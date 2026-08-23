#!/usr/bin/env python3
"""Recertify and seal the 1,196 supplemental Phase-0 gold targets.

This is the train-only counterpart of ``sanitize_compact_targets.py``.  It
uses the same forbidden-runtime policy and the same production evaluator, but
binds every row to the independently sealed 2,776-fit membership expansion.
All 1,196 rows remain available to sequence imitation.  A target that truly
requires a forbidden runtime library is omitted only from the executable
reward view and is named explicitly in the output seal.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUILD_SCHEMA = "phase0-supplemental-target-sanitation-v1"
SPLIT_SEAL_SCHEMA = "compact-public-private-join-seal-v1"
SELECTION_SCHEMA = "multifunction-phase0-fit-expansion-selection-v1"
SELECTION_ROW_SCHEMA = "multifunction-phase0-fit-membership-v1"
TARGET_ROW_SCHEMA = "phase0-supplemental-target-row-v1"
EXPECTED_ROWS = 1_196
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FORBIDDEN_IMPORT_RE = re.compile(
    r"""(?mx)
    ^[ \t]*import[ \t]+
    (?P<quote>['"])
    (?P<uri>
      dart:(?:ffi|io|mirrors)
      |package:ffi/ffi\.dart
    )
    (?P=quote)
    (?P<combinators>
      (?:
        [ \t]+as[ \t]+[A-Za-z_$][A-Za-z0-9_$]*
        |[ \t]+(?:show|hide)[ \t]+
          [A-Za-z_$][A-Za-z0-9_$]*
          (?:[ \t]*,[ \t]*[A-Za-z_$][A-Za-z0-9_$]*)*
      )*
    )
    [ \t]*;[ \t]*(?:\r?\n|$)
    """
)


class SupplementalSanitationError(ValueError):
    """Supplemental target sanitation failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def require_file_hash(
    path: str | Path, expected: str, label: str
) -> dict[str, Any]:
    expected = str(expected).strip().lower()
    if SHA256_RE.fullmatch(expected) is None:
        raise SupplementalSanitationError(
            f"{label} expected SHA-256 is malformed"
        )
    record = file_record(path)
    if record["sha256"] != expected:
        raise SupplementalSanitationError(
            f"{label} hash mismatch: expected {expected}, "
            f"observed {record['sha256']}"
        )
    return record


def read_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementalSanitationError(
            f"cannot read {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SupplementalSanitationError(f"{label} is not an object")
    return value


def read_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SupplementalSanitationError(
                    f"{label} has a blank row at line {line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SupplementalSanitationError(
                    f"{label} has invalid JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise SupplementalSanitationError(
                    f"{label} row {line_number} is not an object"
                )
            rows.append(value)
    if not rows:
        raise SupplementalSanitationError(f"{label} is empty")
    return rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )


def atomic_write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    _atomic_write_bytes(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


def ordered_ids(
    rows: Sequence[Mapping[str, Any]], label: str
) -> list[str]:
    values = [str(row.get("task_id") or "") for row in rows]
    if any(not value for value in values):
        raise SupplementalSanitationError(f"{label} has an empty task_id")
    if len(set(values)) != len(values):
        raise SupplementalSanitationError(f"{label} has duplicate task IDs")
    return values


def load_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "phase0_supplemental_evaluator", path
    )
    if spec is None or spec.loader is None:
        raise SupplementalSanitationError(f"cannot import evaluator {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "DART_BIN",
        "COMPLETION_ATTESTATION_ID",
        "disallowed_dart_test_runtime_library",
        "evaluate_dart_jit_tests_detail",
        "validate_dart_binary",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise SupplementalSanitationError(
            f"evaluator lacks required symbols: {missing}"
        )
    module.validate_dart_binary()
    return module


def dart_version(dart_binary: str) -> str:
    result = subprocess.run(
        [dart_binary, "--version"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    value = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0 or not value:
        raise SupplementalSanitationError(
            f"cannot identify Dart runtime: {value}"
        )
    return value


def strip_forbidden_imports(source: str) -> tuple[str, list[str]]:
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        removed.append(str(match.group("uri")).lower())
        return ""

    return FORBIDDEN_IMPORT_RE.sub(replace, source), removed


def row_tests(row: Mapping[str, Any], task_id: str) -> str:
    tests = row.get("tests")
    acceptance = row.get("acceptance_tests")
    if not isinstance(tests, str) or not tests.strip():
        raise SupplementalSanitationError(f"{task_id}: tests are missing")
    if not isinstance(acceptance, str) or acceptance != tests:
        raise SupplementalSanitationError(
            f"{task_id}: acceptance_tests must equal tests before partitioning"
        )
    return tests


def evaluate_row(
    evaluator: Any,
    row: Mapping[str, Any],
    *,
    task_suffix: str,
    timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    source = str(row.get("dart_source") or "")
    tests = row_tests(row, task_id)
    # The generation-oriented evaluator deliberately trims leading prose until
    # it sees a likely function signature.  A canonical gold program may begin
    # with an enum, typedef, extension, or class needed by fn0; passing it as a
    # raw generation would therefore drop those declarations.  A Dart fence is
    # the evaluator's documented lossless full-program input form.
    fenced_source = f"```dart\n{source.rstrip()}\n```"
    compiled, passed, diagnostic, _combined = (
        evaluator.evaluate_dart_jit_tests_detail(
            fenced_source,
            tests,
            f"{task_id}_{task_suffix}",
            timeout=timeout,
            stability_runs=stability_runs,
        )
    )
    return {
        "task_id": task_id,
        "compiled": bool(compiled),
        "passed": bool(passed),
        "diagnostic": str(diagnostic or "")[:2_000],
        "source_sha256": sha256_text(source),
        "tests_sha256": sha256_text(tests),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--input-seal", required=True, type=Path)
    parser.add_argument("--expected-input-seal-sha256", required=True)
    parser.add_argument("--supplemental-manifest", required=True, type=Path)
    parser.add_argument("--expected-supplemental-manifest-sha256", required=True)
    parser.add_argument("--selection-seal", required=True, type=Path)
    parser.add_argument("--expected-selection-seal-sha256", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--expected-dart-version", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability-runs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def sanitize(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "input": args.input,
            "input_seal": args.input_seal,
            "supplemental_manifest": args.supplemental_manifest,
            "selection_seal": args.selection_seal,
            "contract": args.contract,
            "evaluator": args.evaluator,
        }.items()
    }
    inputs = {
        "input": require_file_hash(
            paths["input"], args.expected_input_sha256, "input"
        ),
        "input_seal": require_file_hash(
            paths["input_seal"],
            args.expected_input_seal_sha256,
            "input seal",
        ),
        "supplemental_manifest": require_file_hash(
            paths["supplemental_manifest"],
            args.expected_supplemental_manifest_sha256,
            "supplemental manifest",
        ),
        "selection_seal": require_file_hash(
            paths["selection_seal"],
            args.expected_selection_seal_sha256,
            "selection seal",
        ),
        "contract": require_file_hash(
            paths["contract"], args.expected_contract_sha256, "contract"
        ),
        "evaluator": require_file_hash(
            paths["evaluator"], args.expected_evaluator_sha256, "evaluator"
        ),
    }
    input_seal = read_json(paths["input_seal"], "input seal")
    selection = read_json(paths["selection_seal"], "selection seal")
    if (
        input_seal.get("schema") != SPLIT_SEAL_SCHEMA
        or input_seal.get("selected_role") != "fit"
        or input_seal.get("training_allowed") is not True
        or int(input_seal.get("rows", -1)) != EXPECTED_ROWS
        or input_seal.get("output_sha256") != inputs["input"]["sha256"]
        or input_seal.get("contract_sha256") != inputs["contract"]["sha256"]
    ):
        raise SupplementalSanitationError("input seal contract failed")
    counts = selection.get("counts")
    artifacts = selection.get("artifacts")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("passed") is not True
        or not isinstance(counts, Mapping)
        or int(counts.get("supplemental_rows", -1)) != EXPECTED_ROWS
        or not isinstance(artifacts, Mapping)
        or not isinstance(artifacts.get("supplemental_task_manifest"), Mapping)
        or artifacts["supplemental_task_manifest"].get("sha256")
        != inputs["supplemental_manifest"]["sha256"]
    ):
        raise SupplementalSanitationError("selection seal contract failed")

    rows = read_jsonl(paths["input"], "supplemental target input")
    membership = read_jsonl(
        paths["supplemental_manifest"], "supplemental membership"
    )
    row_ids = ordered_ids(rows, "supplemental target input")
    membership_ids = ordered_ids(membership, "supplemental membership")
    if (
        len(rows) != EXPECTED_ROWS
        or len(membership) != EXPECTED_ROWS
        or row_ids != membership_ids
        or stable_sha256(row_ids)
        != (selection.get("digests") or {}).get(
            "supplemental_ordered_task_ids_sha256"
        )
    ):
        raise SupplementalSanitationError(
            "supplemental target/membership order differs"
        )
    for index, (row, member) in enumerate(zip(rows, membership, strict=True)):
        task_id = row_ids[index]
        if (
            row.get("schema") != TARGET_ROW_SCHEMA
            or row.get("function") != "fn0"
            or member.get("schema") != SELECTION_ROW_SCHEMA
            or member.get("partition") != "supplemental"
            or int(member.get("supplemental_row", -1)) != index
        ):
            raise SupplementalSanitationError(
                f"{task_id}: supplemental target/membership contract mismatch"
            )
        row_tests(row, task_id)

    evaluator = load_evaluator(paths["evaluator"])
    observed_dart_version = dart_version(str(evaluator.DART_BIN))
    if observed_dart_version != args.expected_dart_version:
        raise SupplementalSanitationError(
            "Dart runtime mismatch: expected "
            f"{args.expected_dart_version!r}, observed {observed_dart_version!r}"
        )

    imitation_rows: list[dict[str, Any]] = []
    executable_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        task_id = str(row["task_id"])
        source = str(row.get("dart_source") or "")
        forbidden_before = evaluator.disallowed_dart_test_runtime_library(source)
        if not forbidden_before:
            imitation_rows.append(row)
            executable_rows.append(row)
            continue
        sanitized_source, removed = strip_forbidden_imports(source)
        forbidden_after = evaluator.disallowed_dart_test_runtime_library(
            sanitized_source
        )
        if not removed or forbidden_after:
            raise SupplementalSanitationError(
                f"{task_id}: forbidden import cannot be removed exactly"
            )
        candidate = dict(row)
        candidate["dart_source"] = sanitized_source
        outcome = evaluate_row(
            evaluator,
            candidate,
            task_suffix="sanitation_probe",
            timeout=args.timeout,
            stability_runs=args.stability_runs,
        )
        receipt = {
            "task_id": task_id,
            "removed_import_uris": sorted(set(removed)),
            "source_sha256_before": sha256_text(source),
            "source_sha256_after": sha256_text(sanitized_source),
            "tests_sha256": outcome["tests_sha256"],
            "compiled_after_removal": outcome["compiled"],
            "passed_after_removal": outcome["passed"],
            "diagnostic": outcome["diagnostic"],
        }
        if outcome["compiled"] and outcome["passed"]:
            imitation_rows.append(candidate)
            executable_rows.append(candidate)
            changes.append(receipt)
        else:
            imitation_rows.append(row)
            quarantine.append(
                {
                    **receipt,
                    "reason": (
                        "forbidden_import_is_semantically_required;"
                        "sequence_imitation_only"
                    ),
                    "original_row_sha256": stable_sha256(row),
                }
            )
    if len(imitation_rows) != EXPECTED_ROWS:
        raise SupplementalSanitationError("imitation rows were dropped")

    def recertify(row: Mapping[str, Any]) -> dict[str, Any]:
        return evaluate_row(
            evaluator,
            row,
            task_suffix="gold_recertify",
            timeout=args.timeout,
            stability_runs=args.stability_runs,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as pool:
        recertification = list(pool.map(recertify, executable_rows))
    failures = [item for item in recertification if not item["passed"]]
    if failures:
        raise SupplementalSanitationError(
            f"{len(failures)} executable supplemental gold targets failed: "
            f"{failures[:10]!r}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_paths = {
        "imitation": output_dir / "supplemental_train_imitation_1196.jsonl",
        "imitation_seal": (
            output_dir / "supplemental_train_imitation_1196.seal.json"
        ),
        "executable": output_dir / "supplemental_train_executable.jsonl",
        "executable_seal": (
            output_dir / "supplemental_train_executable.seal.json"
        ),
        "quarantine": output_dir / "execution_ineligible.jsonl",
        "report": output_dir / "sanitation_report.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing sanitation outputs: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_paths["imitation"], imitation_rows)
    atomic_write_jsonl(output_paths["executable"], executable_rows)
    atomic_write_jsonl(output_paths["quarantine"], quarantine)
    sanitizer_sha = sha256_file(Path(__file__).resolve())
    ineligible_ids = sorted(str(item["task_id"]) for item in quarantine)
    common = {
        "schema": SPLIT_SEAL_SCHEMA,
        "selected_role": "fit",
        "training_allowed": True,
        "contract_sha256": inputs["contract"]["sha256"],
        "sanitation_schema": BUILD_SCHEMA,
        "sanitizer_sha256": sanitizer_sha,
        "evaluator_sha256": inputs["evaluator"]["sha256"],
        "completion_attestation_id": evaluator.COMPLETION_ATTESTATION_ID,
        "dart_version": observed_dart_version,
        "stability_runs": int(args.stability_runs),
        "selection_seal_sha256": inputs["selection_seal"]["sha256"],
        "supplemental_manifest_sha256": inputs[
            "supplemental_manifest"
        ]["sha256"],
        "ordered_task_ids_sha256": stable_sha256(row_ids),
        "task_set_sha256": stable_sha256(row_ids),
        "sorted_task_set_sha256": stable_sha256(sorted(row_ids)),
        "quarantine_sha256": sha256_file(output_paths["quarantine"]),
    }
    atomic_write_json(
        output_paths["imitation_seal"],
        {
            **common,
            "training_objective_scope": "sequence_imitation_all_train",
            "output_sha256": sha256_file(output_paths["imitation"]),
            "input_sha256": inputs["input"]["sha256"],
            "rows": EXPECTED_ROWS,
            "executable_reward_eligible_rows": len(executable_rows),
            "execution_ineligible_task_ids": ineligible_ids,
        },
    )
    atomic_write_json(
        output_paths["executable_seal"],
        {
            **common,
            "training_objective_scope": "executable_reward_only",
            "output_sha256": sha256_file(output_paths["executable"]),
            "input_sha256": inputs["input"]["sha256"],
            "rows": len(executable_rows),
            "execution_ineligible_task_ids": ineligible_ids,
        },
    )
    report = {
        "schema": BUILD_SCHEMA,
        "inputs": inputs,
        "runtime": {
            "dart_binary": str(evaluator.DART_BIN),
            "dart_version": observed_dart_version,
            "completion_attestation_id": evaluator.COMPLETION_ATTESTATION_ID,
            "stability_runs": int(args.stability_runs),
        },
        "counts": {
            "input_rows": EXPECTED_ROWS,
            "imitation_rows": len(imitation_rows),
            "executable_rows": len(executable_rows),
            "sanitized_rows": len(changes),
            "execution_ineligible_rows": len(ineligible_ids),
            "recertified_rows": len(recertification),
            "recertification_failures": 0,
        },
        "execution_ineligible_task_ids": ineligible_ids,
        "changes": sorted(changes, key=lambda item: item["task_id"]),
        "quarantine": sorted(quarantine, key=lambda item: item["task_id"]),
        "outputs": {
            key: file_record(path)
            for key, path in output_paths.items()
            if key != "report"
        },
        "invariants": {
            "all_imitation_rows_retained": True,
            "all_executable_rows_recertified": True,
            "forbidden_imports_removed_only_after_pass": True,
            "required_forbidden_imports_sequence_imitation_only": True,
            "expected_outputs_not_rewritten": True,
            "supplemental_membership_and_order_bound": True,
        },
        "passed": True,
    }
    atomic_write_json(output_paths["report"], report)
    print(
        "PHASE0_SUPPLEMENTAL_TARGETS_SANITIZED "
        f"imitation={len(imitation_rows)} executable={len(executable_rows)} "
        f"ineligible={len(ineligible_ids)} "
        f"imitation_sha256={sha256_file(output_paths['imitation'])}",
        flush=True,
    )
    return report


def main() -> int:
    sanitize(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
