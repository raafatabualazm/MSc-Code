"""Build a deterministic visible/hidden test split for repair evaluation.

The input benchmark is never modified. Candidate-bearing expect/assert
statements are partitioned per task, while the candidate binding and helper
functions are preserved byte-for-byte in both harnesses. The visible sidecar
contains no scoring-test field; the hidden output remains a normal graph JSONL
that downstream pass@k tools can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.repair_loop_antigravity import (
    _candidate_calls,
    _resolve_dart_binary,
    run_dart_tests,
    validate_dart_binary,
    validate_visible_test_boundary,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def task_identity(row: dict[str, Any], index: int) -> str:
    for key in ("task_id", "id", "filename"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    raise ValueError(f"input row {index + 1} has no stable task identifier")


def _balanced_call_close(text: str, open_paren: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    index = open_paren
    while index < len(text):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def candidate_case_spans(harness: str) -> list[tuple[int, int]]:
    """Return spans for complete expect/assert statements that call candidate."""
    import re

    spans: list[tuple[int, int]] = []
    cursor = 0
    pattern = re.compile(r"\b(?:expect|assert)\s*\(")
    while True:
        match = pattern.search(harness, cursor)
        if match is None:
            break
        open_paren = harness.find("(", match.start(), match.end())
        close = _balanced_call_close(harness, open_paren)
        if close is None:
            raise ValueError(
                f"unbalanced or unterminated test statement near offset {match.start()}"
            )
        semicolon = close
        while semicolon < len(harness) and harness[semicolon].isspace():
            semicolon += 1
        if semicolon >= len(harness) or harness[semicolon] != ";":
            # Function declarations such as `void expect(...) {` are helpers,
            # not candidate test statements.
            cursor = close
            continue
        end = semicolon + 1
        statement = harness[match.start():end]
        if _candidate_calls(statement):
            spans.append((match.start(), end))
        cursor = end
    return spans


def retain_cases(
    harness: str,
    spans: list[tuple[int, int]],
    keep_indices: set[int],
) -> str:
    chunks: list[str] = []
    cursor = 0
    for case_index, (start, end) in enumerate(spans):
        chunks.append(harness[cursor:start])
        statement = harness[start:end]
        if case_index in keep_indices:
            chunks.append(statement)
        else:
            # Preserve line count and surrounding layout for auditable diffs.
            chunks.append("".join("\n" if char == "\n" else " " for char in statement))
        cursor = end
    chunks.append(harness[cursor:])
    return "".join(chunks)


def split_harness(
    harness: str,
    task_id: str,
    seed: int,
    min_visible: int = 1,
    min_hidden: int = 1,
) -> tuple[str, str, dict[str, Any]]:
    spans = candidate_case_spans(harness)
    groups: dict[tuple[str, ...], list[int]] = {}
    for case_index, (start, end) in enumerate(spans):
        input_key = tuple(sorted(_candidate_calls(harness[start:end])))
        groups.setdefault(input_key, []).append(case_index)
    required_groups = min_visible + min_hidden
    if len(groups) < required_groups:
        raise ValueError(
            f"task {task_id} has {len(groups)} unique candidate-input group(s); "
            f"at least {required_groups} are required for min_visible="
            f"{min_visible}, min_hidden={min_hidden}"
        )

    ranked = sorted(
        groups,
        key=lambda input_key: hashlib.sha256(
            (
                f"{seed}|{task_id}|" + "|".join(input_key)
            ).encode("utf-8")
        ).digest(),
    )
    visible_group_count = min(
        max(min_visible, len(groups) // 2),
        len(groups) - min_hidden,
    )
    visible_groups = set(ranked[:visible_group_count])
    visible_indices = {
        case_index
        for input_key in visible_groups
        for case_index in groups[input_key]
    }
    hidden_indices = set(range(len(spans))) - visible_indices
    visible = retain_cases(harness, spans, visible_indices)
    hidden = retain_cases(harness, spans, hidden_indices)
    validate_visible_test_boundary(visible, hidden, task_id)

    metadata = {
        "total_cases": len(spans),
        "unique_input_groups": len(groups),
        "visible_cases": len(visible_indices),
        "hidden_cases": len(hidden_indices),
        "visible_input_groups": len(visible_groups),
        "hidden_input_groups": len(groups) - len(visible_groups),
        "min_visible_input_groups": min_visible,
        "min_hidden_input_groups": min_hidden,
        "visible_case_indices": sorted(visible_indices),
        "hidden_case_indices": sorted(hidden_indices),
        "original_tests_sha256": sha256_text(harness),
        "visible_tests_sha256": sha256_text(visible),
        "hidden_tests_sha256": sha256_text(hidden),
    }
    return visible, hidden, metadata


def build_split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    drop_unsplittable: bool,
    min_visible: int = 1,
    min_hidden: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    visible_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        task_id = task_identity(row, index)
        harness = row.get("tests", "") or ""
        try:
            visible, hidden, metadata = split_harness(
                harness,
                task_id,
                seed,
                min_visible,
                min_hidden,
            )
        except ValueError as exc:
            if not drop_unsplittable:
                raise
            dropped.append({
                "task_id": task_id,
                "source_line": index + 1,
                "reason": str(exc),
            })
            continue

        visible_row = {
            key: row[key]
            for key in ("task_id", "id", "filename", "function")
            if row.get(key) not in (None, "")
        }
        visible_row["visible_tests"] = visible
        visible_row["split_metadata"] = metadata
        hidden_row = dict(row)
        hidden_row["tests"] = hidden
        visible_rows.append(visible_row)
        hidden_rows.append(hidden_row)
        task_records.append({
            "task_id": task_id,
            "source_line": index + 1,
            **metadata,
        })

    return visible_rows, hidden_rows, task_records, dropped


def validate_references(
    source_rows: list[dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    visible_rows: list[dict[str, Any]],
    dart_bin: str,
    timeout: int,
    workers: int,
) -> dict[str, Any]:
    source_by_id = {
        task_identity(row, index): row
        for index, row in enumerate(source_rows)
    }
    jobs: list[tuple[str, str, str, str]] = []
    for hidden, visible in zip(hidden_rows, visible_rows, strict=True):
        task_id = str(hidden.get("task_id", hidden.get("id", hidden.get("filename"))))
        original = source_by_id[task_id]
        source = hidden.get("dart_source", hidden.get("source", "")) or ""
        jobs.append((task_id, "original", source, original.get("tests", "") or ""))
        jobs.append((task_id, "visible", source, visible["visible_tests"]))
        jobs.append((task_id, "hidden", source, hidden["tests"]))

    results: dict[tuple[str, str], tuple[bool, str]] = {}

    def run_one(job: tuple[str, str, str, str]) -> tuple[str, str, bool, str]:
        task_id, side, source, harness = job
        ok, diagnostic = run_dart_tests(
            source,
            harness,
            f"split_{task_id}_{side}",
            dart_bin,
            timeout,
        )
        return task_id, side, ok, diagnostic

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            task_id, side, ok, diagnostic = future.result()
            results[(task_id, side)] = (ok, diagnostic)

    preexisting_failures: list[dict[str, Any]] = []
    introduced_failures: list[dict[str, Any]] = []
    for hidden in hidden_rows:
        task_id = str(hidden.get("task_id", hidden.get("id", hidden.get("filename"))))
        original_ok, original_diag = results[(task_id, "original")]
        visible_ok, visible_diag = results[(task_id, "visible")]
        hidden_ok, hidden_diag = results[(task_id, "hidden")]
        if not original_ok:
            preexisting_failures.append({
                "task_id": task_id,
                "original_diagnostic": original_diag[-1000:],
                "visible_passed": visible_ok,
                "hidden_passed": hidden_ok,
            })
        elif not visible_ok or not hidden_ok:
            introduced_failures.append({
                "task_id": task_id,
                "visible_passed": visible_ok,
                "hidden_passed": hidden_ok,
                "visible_diagnostic": visible_diag[-1000:],
                "hidden_diagnostic": hidden_diag[-1000:],
            })

    if introduced_failures:
        raise RuntimeError(
            f"the split introduced failures for {len(introduced_failures)} task(s); "
            f"first failures: {json.dumps(introduced_failures[:5], ensure_ascii=False)}"
        )
    return {
        "jobs": len(jobs),
        "original_passed": len(hidden_rows) - len(preexisting_failures),
        "original_failed_preexisting": len(preexisting_failures),
        "visible_passed": sum(results[(str(row.get("task_id", row.get("id", row.get("filename")))), "visible")][0] for row in hidden_rows),
        "hidden_passed": sum(results[(str(row.get("task_id", row.get("id", row.get("filename")))), "hidden")][0] for row in hidden_rows),
        "introduced_failures": [],
        "preexisting_reference_failures": preexisting_failures,
    }


def exclude_reference_failures(
    source_rows: list[dict[str, Any]],
    visible_rows: list[dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    task_records: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    runtime_validation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop tasks whose original reference fails its unsplit harness."""
    failures = runtime_validation["preexisting_reference_failures"]
    failure_by_id = {str(record["task_id"]): record for record in failures}
    if not failure_by_id:
        runtime_validation["reference_failure_policy"] = "drop"
        runtime_validation["excluded_reference_failures"] = 0
        runtime_validation["final_rows"] = len(hidden_rows)
        runtime_validation["final_original_passed"] = len(hidden_rows)
        runtime_validation["final_visible_passed"] = len(hidden_rows)
        runtime_validation["final_hidden_passed"] = len(hidden_rows)
        return visible_rows, hidden_rows, task_records

    source_line_by_id = {
        task_identity(row, index): index + 1
        for index, row in enumerate(source_rows)
    }
    for task_id, failure in sorted(failure_by_id.items()):
        dropped.append({
            "task_id": task_id,
            "source_line": source_line_by_id[task_id],
            "reason": "original reference failed the unsplit test harness",
            "diagnostic": failure["original_diagnostic"],
        })

    visible_rows = [
        row
        for index, row in enumerate(visible_rows)
        if task_identity(row, index) not in failure_by_id
    ]
    hidden_rows = [
        row
        for index, row in enumerate(hidden_rows)
        if task_identity(row, index) not in failure_by_id
    ]
    task_records = [
        record
        for record in task_records
        if str(record["task_id"]) not in failure_by_id
    ]
    runtime_validation["reference_failure_policy"] = "drop"
    runtime_validation["excluded_reference_failures"] = len(failure_by_id)
    runtime_validation["excluded_reference_failure_task_ids"] = sorted(
        failure_by_id
    )
    runtime_validation["final_rows"] = len(hidden_rows)
    runtime_validation["final_original_passed"] = len(hidden_rows)
    runtime_validation["final_visible_passed"] = len(hidden_rows)
    runtime_validation["final_hidden_passed"] = len(hidden_rows)
    return visible_rows, hidden_rows, task_records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--visible_out", required=True, type=Path)
    parser.add_argument("--hidden_out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_visible", type=int, default=1)
    parser.add_argument("--min_hidden", type=int, default=1)
    parser.add_argument("--drop_unsplittable", action="store_true")
    parser.add_argument("--run_tests", action="store_true")
    parser.add_argument(
        "--drop_reference_failures",
        action="store_true",
        help=(
            "after --run_tests, exclude tasks whose original reference fails "
            "the unsplit harness"
        ),
    )
    parser.add_argument("--dart", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.min_visible <= 0 or args.min_hidden <= 0:
        raise SystemExit("ERROR: --min_visible and --min_hidden must be positive")
    if args.drop_reference_failures and not args.run_tests:
        raise SystemExit(
            "ERROR: --drop_reference_failures requires --run_tests"
        )

    outputs = [args.visible_out, args.hidden_out, args.manifest]
    resolved = [path.resolve() for path in [args.input, *outputs]]
    if len(set(resolved)) != len(resolved):
        raise SystemExit("ERROR: input and output paths must all be distinct")
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"ERROR: outputs already exist (use --force): {existing}")

    rows = load_jsonl(args.input)
    try:
        visible_rows, hidden_rows, task_records, dropped = build_split_rows(
            rows,
            args.seed,
            args.drop_unsplittable,
            args.min_visible,
            args.min_hidden,
        )
    except ValueError as exc:
        raise SystemExit(
            f"ERROR: {exc}. Supply an independent visible-test override in a "
            "future split, or rerun with --drop_unsplittable and report the "
            "reduced task count."
        ) from None
    structural_output_rows = len(hidden_rows)
    runtime_validation = None
    dart_bin = None
    if args.run_tests:
        dart_bin = _resolve_dart_binary(args.dart)
        validate_dart_binary(dart_bin)
        runtime_validation = validate_references(
            rows,
            hidden_rows,
            visible_rows,
            dart_bin,
            args.timeout,
            args.workers,
        )
        runtime_validation["structural_output_rows"] = structural_output_rows
        runtime_validation["reference_failure_policy"] = "retain"
        runtime_validation["excluded_reference_failures"] = 0
        runtime_validation["final_rows"] = structural_output_rows
        if args.drop_reference_failures:
            visible_rows, hidden_rows, task_records = exclude_reference_failures(
                rows,
                visible_rows,
                hidden_rows,
                task_records,
                dropped,
                runtime_validation,
            )

    write_jsonl(args.visible_out, visible_rows)
    write_jsonl(args.hidden_out, hidden_rows)
    manifest = {
        "schema_version": 1,
        "stage": "repair_test_split",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "strategy": "sha256_ranked_constrained_split",
        "min_visible_input_groups": args.min_visible,
        "min_hidden_input_groups": args.min_hidden,
        "drop_unsplittable": bool(args.drop_unsplittable),
        "drop_reference_failures": bool(args.drop_reference_failures),
        "input_rows": len(rows),
        "structural_output_rows": structural_output_rows,
        "output_rows": len(hidden_rows),
        "input": file_record(args.input),
        "visible_output": file_record(args.visible_out),
        "hidden_output": file_record(args.hidden_out),
        "source_files": [
            file_record(Path(__file__)),
            file_record(Path(__file__).with_name("repair_loop_antigravity.py")),
        ],
        "dropped": dropped,
        "tasks": task_records,
        "runtime_validation": runtime_validation,
        "dart_binary": dart_bin,
        "invariants": {
            "input_unmodified": True,
            "visible_sidecar_has_no_scoring_tests_field": True,
            "candidate_bindings_match": True,
            "visible_hidden_candidate_inputs_disjoint": True,
            "helpers_preserved_in_both_harnesses": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"split complete: input={len(rows)} output={len(hidden_rows)} "
        f"dropped={len(dropped)}"
    )
    print(f"visible: {args.visible_out}")
    print(f"hidden: {args.hidden_out}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
