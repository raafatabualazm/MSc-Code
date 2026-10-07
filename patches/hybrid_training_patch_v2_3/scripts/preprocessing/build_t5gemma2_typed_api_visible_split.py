#!/usr/bin/env python3
"""Build a sealed visible/complement TRAIN split for typed API rescue.

The existing VeRPO feedback artifact does not cover the expanded 2,776-row
corpus.  This builder derives a task-bound split directly from the pinned gold
TRAIN harnesses and excludes the one known contaminated row.  Ordinary
``expect`` harnesses use the established half-split implementation.  Canonical
stdout harnesses are split by expected output line.  Generated ``_agEval``
harnesses are split by complete call/assertion blocks.  A singleton stdout
oracle cannot be divided without leaking its only answer, so its visible view
is deliberately compile-and-call-only and the sole semantic case remains in
the private complement.

The public file contains visible checks only.  The private file contains the
complement only.  Neither contains gold Dart source, typed/F2 model input, nor
held-out evaluation tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)
from scripts.preprocessing.build_verpo_feedback_view import (
    SPLIT_SCHEMA as EXPECT_SPLIT_SCHEMA,
    extract_expect_spans,
    harness_with_cases,
    split_train_harness,
)
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training.t5gemma2_typed_direct_rs_sft import (
    CONTAMINATED_TRAIN_TASK_ID,
)


PUBLIC_SCHEMA = "t5gemma2-typed-api-visible-train-row-v1"
PRIVATE_SCHEMA = "t5gemma2-typed-api-private-complement-row-v1"
MANIFEST_SCHEMA = "t5gemma2-typed-api-visible-split-manifest-v1"
SPLIT_BINDING_SCHEMA = "t5gemma2-typed-api-visible-split-binding-v1"
EXPECTED_ROWS = 2776
EXPECTED_CLEAN_ROWS = 2775
DEFAULT_SEED = 20260801

_HEX64 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_LITERAL = re.compile(
    r'(?P<prefix>\bconst\s+_expected\s*=\s*)(?P<literal>"(?:\\.|[^"\\])*")(?P<suffix>\s*;)'
)
_ACTUAL_ASSIGNMENT = re.compile(
    r"(?m)^(?P<indent>\s*)final\s+_actual\s*=\s*_captured\.isEmpty\s*\?\s*''\s*:\s*'\$\{_captured\.join\('\\n'\)\}\\n'\s*;\s*$"
)
_AG_CASE_BLOCK = re.compile(
    r"(?m)^(?P<indent>\s*)final\s+_v(?P<index>[0-9]+)\s*=.*?;\s*\r?\n"
    r"(?P=indent)if\s*\(_v(?P=index)\s*!=.*?;\s*$"
)


class VisibleSplitError(ValueError):
    pass


def _pin(path_value: str | Path, expected: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    digest = str(expected or "").strip().lower()
    if not _HEX64.fullmatch(digest):
        raise VisibleSplitError(f"{label} expected digest is not SHA-256")
    if not path.is_file() or sha256_file(path) != digest:
        raise VisibleSplitError(f"{label} digest differs")
    return path


def _tests(row: Mapping[str, Any], task_id: str) -> str:
    value = row.get("acceptance_tests") or row.get("tests")
    if not isinstance(value, str) or not value.strip():
        raise VisibleSplitError(f"{task_id}: complete TRAIN tests are absent")
    return value


def _ranked_indices(
    *, task_id: str, tests_sha256: str, count: int, seed: int, strategy: str
) -> tuple[list[int], list[int]]:
    if count < 2:
        raise VisibleSplitError("semantic half split requires at least two cases")
    ordered = sorted(
        range(count),
        key=lambda index: canonical_sha256(
            {
                "schema": SPLIT_BINDING_SCHEMA,
                "seed": seed,
                "task_id": task_id,
                "tests_sha256": tests_sha256,
                "strategy": strategy,
                "case_index": index,
            }
        ),
    )
    visible_count = count // 2
    visible = sorted(ordered[:visible_count])
    holdback = sorted(set(range(count)) - set(visible))
    if not visible or not holdback:
        raise VisibleSplitError("semantic half split produced an empty side")
    return visible, holdback


def _stdout_literal(tests: str) -> tuple[re.Match[str], str]:
    match = _EXPECTED_LITERAL.search(tests)
    if match is None:
        raise VisibleSplitError("canonical stdout expected literal is absent")
    try:
        value = json.loads(match.group("literal"))
    except json.JSONDecodeError as exc:
        raise VisibleSplitError("canonical stdout expected literal is malformed") from exc
    if not isinstance(value, str):
        raise VisibleSplitError("canonical stdout expected value is not text")
    return match, value


def _replace_stdout_view(tests: str, *, indices: Sequence[int]) -> str:
    literal_match, expected = _stdout_literal(tests)
    lines = expected.splitlines()
    trailing_newline = expected.endswith("\n")
    selected = [lines[index] for index in indices]
    selected_expected = "\n".join(selected) + ("\n" if trailing_newline else "")
    replacement_literal = json.dumps(selected_expected, ensure_ascii=True)
    replaced = (
        tests[: literal_match.start("literal")]
        + replacement_literal
        + tests[literal_match.end("literal") :]
    )
    actual_match = _ACTUAL_ASSIGNMENT.search(replaced)
    if actual_match is None:
        raise VisibleSplitError("canonical stdout actual projection is absent")
    index_literal = ",".join(str(index) for index in indices)
    indent = actual_match.group("indent")
    projection = (
        f"{indent}final _actual = <int>[{index_literal}]\n"
        f"{indent}    .map((i) => i < _captured.length "
        f"? _captured[i] : '<MISSING_OUTPUT_LINE_\u0024i>')\n"
        f"{indent}    .join('\\n') + "
        f"({str(trailing_newline).lower()} ? '\\n' : '');"
    )
    return replaced[: actual_match.start()] + projection + replaced[actual_match.end() :]


def _compile_only_stdout(tests: str) -> str:
    literal_match, _expected = _stdout_literal(tests)
    replaced = (
        tests[: literal_match.start("literal")]
        + '""'
        + tests[literal_match.end("literal") :]
    )
    actual_match = _ACTUAL_ASSIGNMENT.search(replaced)
    if actual_match is None:
        raise VisibleSplitError("singleton stdout actual projection is absent")
    indent = actual_match.group("indent")
    return (
        replaced[: actual_match.start()]
        + f"{indent}final _actual = '';"
        + replaced[actual_match.end() :]
    )


def _split_stdout(
    *, task_id: str, tests: str, seed: int
) -> tuple[str, str, dict[str, Any]]:
    _match, expected = _stdout_literal(tests)
    lines = expected.splitlines()
    tests_sha = hashlib.sha256(tests.encode("utf-8")).hexdigest()
    if len(lines) == 1:
        return _compile_only_stdout(tests), tests, {
            "strategy": "stdout_singleton_compile_and_call_visible",
            "case_count": 1,
            "visible_count": 0,
            "holdback_count": 1,
            "visible_case_indices": [],
            "holdback_case_indices": [0],
            "tests_sha256": tests_sha,
        }
    visible, holdback = _ranked_indices(
        task_id=task_id,
        tests_sha256=tests_sha,
        count=len(lines),
        seed=seed,
        strategy="stdout_line_half",
    )
    return (
        _replace_stdout_view(tests, indices=visible),
        _replace_stdout_view(tests, indices=holdback),
        {
            "strategy": "stdout_line_half",
            "case_count": len(lines),
            "visible_count": len(visible),
            "holdback_count": len(holdback),
            "visible_case_indices": visible,
            "holdback_case_indices": holdback,
            "tests_sha256": tests_sha,
        },
    )


def _split_ag_cases(
    *, task_id: str, tests: str, seed: int
) -> tuple[str, str, dict[str, Any]]:
    matches = list(_AG_CASE_BLOCK.finditer(tests))
    indices = [int(match.group("index")) for match in matches]
    if indices != list(range(len(matches))) or len(matches) < 2:
        raise VisibleSplitError("generated _agEval case blocks are malformed")
    spans = [(match.start(), match.end()) for match in matches]
    tests_sha = hashlib.sha256(tests.encode("utf-8")).hexdigest()
    visible, holdback = _ranked_indices(
        task_id=task_id,
        tests_sha256=tests_sha,
        count=len(matches),
        seed=seed,
        strategy="generated_ag_case_half",
    )
    return (
        harness_with_cases(tests, spans, set(visible)),
        harness_with_cases(tests, spans, set(holdback)),
        {
            "strategy": "generated_ag_case_half",
            "case_count": len(matches),
            "visible_count": len(visible),
            "holdback_count": len(holdback),
            "visible_case_indices": visible,
            "holdback_case_indices": holdback,
            "tests_sha256": tests_sha,
        },
    )


def split_task_harness(
    *, task_id: str, tests: str, seed: int
) -> tuple[str, str, dict[str, Any]]:
    expect_count = len(extract_expect_spans(tests))
    if expect_count >= 2:
        split = split_train_harness(task_id=task_id, tests=tests, seed=seed)
        metadata = {
            key: split[key]
            for key in (
                "tests_sha256",
                "case_count",
                "visible_count",
                "holdback_count",
                "visible_case_indices",
                "holdback_case_indices",
            )
        }
        metadata["strategy"] = "established_expect_half"
        metadata["upstream_schema"] = EXPECT_SPLIT_SCHEMA
        return split["feedback_tests"], split["reward_holdback_tests"], metadata
    if _EXPECTED_LITERAL.search(tests) and "_captured" in tests:
        return _split_stdout(task_id=task_id, tests=tests, seed=seed)
    if "_agEval" in tests:
        return _split_ag_cases(task_id=task_id, tests=tests, seed=seed)
    raise VisibleSplitError(f"{task_id}: no supported non-leaking split strategy")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        (
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise VisibleSplitError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_path = _pin(args.gold_train_jsonl, args.expected_gold_train_sha256, "gold TRAIN")
    rows = base_sft._read_jsonl(train_path)  # noqa: SLF001
    if len(rows) != EXPECTED_ROWS:
        raise VisibleSplitError("gold TRAIN must contain exactly 2,776 rows")
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    strategies: dict[str, int] = {}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        task_id = base_sft._identity(row, index)  # noqa: SLF001
        if task_id in seen:
            raise VisibleSplitError("gold TRAIN task identities are not unique")
        seen.add(task_id)
        if task_id == CONTAMINATED_TRAIN_TASK_ID:
            continue
        visible, holdback, metadata = split_task_harness(
            task_id=task_id, tests=_tests(row, task_id), seed=args.seed
        )
        binding_payload = {
            "schema": SPLIT_BINDING_SCHEMA,
            "seed": args.seed,
            "task_id": task_id,
            **metadata,
            "visible_tests_sha256": hashlib.sha256(visible.encode()).hexdigest(),
            "holdback_tests_sha256": hashlib.sha256(holdback.encode()).hexdigest(),
        }
        binding = canonical_sha256(binding_payload)
        common = {
            "task_id": task_id,
            "split_binding_sha256": binding,
            "split_seed": args.seed,
            **metadata,
        }
        public.append(
            {
                "schema": PUBLIC_SCHEMA,
                **common,
                "visible_tests": visible,
                "private_complement_present": False,
                "gold_present": False,
            }
        )
        private.append(
            {
                "schema": PRIVATE_SCHEMA,
                **common,
                "holdback_tests": holdback,
                "visible_tests_present": False,
                "gold_present": False,
            }
        )
        strategy = str(metadata["strategy"])
        strategies[strategy] = strategies.get(strategy, 0) + 1
    if len(public) != EXPECTED_CLEAN_ROWS or len(private) != EXPECTED_CLEAN_ROWS:
        raise VisibleSplitError("clean visible split must contain 2,775 rows")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "visible_train.jsonl"
    private_path = output_dir / "holdback.private.jsonl"
    _write_jsonl(public_path, public)
    _write_jsonl(private_path, private)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "seed": args.seed,
        "inputs": {
            "gold_train_sha256": sha256_file(train_path),
            "rows": EXPECTED_ROWS,
        },
        "outputs": {
            "visible_train": {
                "path": str(public_path),
                "sha256": sha256_file(public_path),
                "rows": len(public),
            },
            "private_holdback": {
                "path": str(private_path),
                "sha256": sha256_file(private_path),
                "rows": len(private),
            },
        },
        "clean_rows": EXPECTED_CLEAN_ROWS,
        "known_contaminant_excluded": CONTAMINATED_TRAIN_TASK_ID,
        "task_ids_sha256": canonical_sha256([row["task_id"] for row in public]),
        "split_bindings_sha256": canonical_sha256(
            [row["split_binding_sha256"] for row in public]
        ),
        "strategies": strategies,
        "privacy": {
            "visible_file_contains_private_complement": False,
            "private_file_contains_visible_tests": False,
            "gold_source_present": False,
            "heldout_175_opened": False,
            "singleton_stdout_answer_visible": False,
        },
    }
    require_exact_or_write(output_dir / "split_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gold_train_jsonl", required=True)
    parser.add_argument("--expected_gold_train_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
