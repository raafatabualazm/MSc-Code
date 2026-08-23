#!/usr/bin/env python3
"""Seal compact targets whose trusted Dart harness is executable by construction.

The scrubbed-data builder certified combined programs after supplying
``dart:async`` and ``dart:convert`` to the test harness, while the serialized
harness omitted those imports.  The shared evaluator supplies that trusted
prelude.  This second, deliberately separate stage handles model-target source:

* remove only forbidden SDK import directives whose removal still compiles and
  passes the exact hidden harness;
* retain train rows that genuinely require forbidden runtime libraries in a
  sequence-imitation-only view, while excluding them from executable reward;
* reject any dev quarantine; and
* recertify every emitted gold target with the exact production evaluator.

No expected output is rewritten.  Platform-sensitive oracles must pass on the
pinned target Dart runtime; the build report records that runtime verbatim.
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


BUILD_SCHEMA = "compact-target-harness-sanitation-v1"
SEAL_SCHEMA = "compact-public-private-join-seal-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_IMPORT_RE = re.compile(
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


class SanitationError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    expected = expected.strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise SanitationError(f"{label} expected SHA-256 is malformed")
    record = file_record(path)
    if record["sha256"] != expected:
        raise SanitationError(
            f"{label} hash mismatch: expected {expected}, "
            f"observed {record['sha256']}"
        )
    return record


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SanitationError(f"{label} is not a JSON object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SanitationError(
                    f"{label} has a blank row at line {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SanitationError(
                    f"{label} row {line_number} is not an object"
                )
            rows.append(value)
    if not rows:
        raise SanitationError(f"{label} is empty")
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_task_ids(value: str, label: str) -> set[str]:
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise SanitationError(f"{label} may not be empty")
    return result


def validate_input_seal(
    *,
    dataset: Path,
    seal_path: Path,
    contract_sha256: str,
    expected_role: str,
) -> dict[str, Any]:
    seal = read_json(seal_path, f"{expected_role} input seal")
    if seal.get("schema") != SEAL_SCHEMA:
        raise SanitationError(f"{expected_role} input seal schema mismatch")
    if seal.get("selected_role") != expected_role:
        raise SanitationError(f"{expected_role} input seal role mismatch")
    observed_dataset = sha256_file(dataset)
    if seal.get("output_sha256") != observed_dataset:
        raise SanitationError(f"{expected_role} input seal dataset mismatch")
    if seal.get("contract_sha256") != contract_sha256:
        raise SanitationError(f"{expected_role} input seal contract mismatch")
    row_count = sum(
        1 for line in dataset.open(encoding="utf-8") if line.strip()
    )
    if int(seal.get("rows", -1)) != row_count:
        raise SanitationError(f"{expected_role} input seal row-count mismatch")
    return file_record(seal_path)


def strip_forbidden_imports(source: str) -> tuple[str, list[str]]:
    """Remove syntactically simple forbidden directives, preserving all else."""

    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        removed.append(str(match.group("uri")).lower())
        return ""

    sanitized = _FORBIDDEN_IMPORT_RE.sub(replace, source)
    return sanitized, removed


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location(
        "sealed_target_harness_evaluator", path
    )
    if spec is None or spec.loader is None:
        raise SanitationError(f"cannot load evaluator {path}")
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
        raise SanitationError(f"evaluator lacks required symbols: {missing}")
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
    text = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0 or not text:
        raise SanitationError(f"cannot identify Dart runtime: {text}")
    return text


def row_tests(row: Mapping[str, Any], task_id: str) -> str:
    tests = row.get("tests")
    acceptance = row.get("acceptance_tests")
    if not isinstance(tests, str) or not tests.strip():
        raise SanitationError(f"{task_id}: missing tests")
    if not isinstance(acceptance, str) or not acceptance.strip():
        raise SanitationError(f"{task_id}: missing acceptance_tests")
    if tests != acceptance:
        raise SanitationError(f"{task_id}: tests/acceptance_tests differ")
    return tests


def evaluate_row(
    evaluator: Any,
    row: Mapping[str, Any],
    *,
    task_suffix: str,
    timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    task_id = str(row.get("task_id") or "")
    source = str(row.get("dart_source") or "")
    tests = row_tests(row, task_id)
    compiled, passed, diagnostic, _combined = (
        evaluator.evaluate_dart_jit_tests_detail(
            source,
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
        "diagnostic": str(diagnostic or "")[:2000],
        "source_sha256": sha256_text(source),
        "tests_sha256": sha256_text(tests),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-train", required=True, type=Path)
    parser.add_argument("--expected-input-train-sha256", required=True)
    parser.add_argument("--input-train-seal", required=True, type=Path)
    parser.add_argument("--expected-input-train-seal-sha256", required=True)
    parser.add_argument("--input-dev", required=True, type=Path)
    parser.add_argument("--expected-input-dev-sha256", required=True)
    parser.add_argument("--input-dev-seal", required=True, type=Path)
    parser.add_argument("--expected-input-dev-seal-sha256", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument(
        "--output-imitation-train",
        required=True,
        type=Path,
        help="All train rows; unsafe rows are sequence-imitation-only.",
    )
    parser.add_argument(
        "--output-imitation-train-seal", required=True, type=Path
    )
    parser.add_argument(
        "--output-train",
        required=True,
        type=Path,
        help="Executable-reward train view; unsafe rows are absent.",
    )
    parser.add_argument("--output-train-seal", required=True, type=Path)
    parser.add_argument("--output-dev", required=True, type=Path)
    parser.add_argument("--output-dev-seal", required=True, type=Path)
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-input-train-rows", type=int, default=1580)
    parser.add_argument("--expected-output-train-rows", type=int, default=1578)
    parser.add_argument("--expected-dev-rows", type=int, default=175)
    parser.add_argument("--expected-sanitized-train-task-ids", required=True)
    parser.add_argument("--expected-quarantined-train-task-ids", required=True)
    parser.add_argument("--expected-sanitized-dev-task-ids", required=True)
    parser.add_argument("--expected-dart-version", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability-runs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "input_train": args.input_train.expanduser().resolve(),
        "input_train_seal": args.input_train_seal.expanduser().resolve(),
        "input_dev": args.input_dev.expanduser().resolve(),
        "input_dev_seal": args.input_dev_seal.expanduser().resolve(),
        "contract": args.contract.expanduser().resolve(),
        "evaluator": args.evaluator.expanduser().resolve(),
    }
    inputs = {
        "train": require_hash(
            paths["input_train"],
            args.expected_input_train_sha256,
            "input train",
        ),
        "train_seal": require_hash(
            paths["input_train_seal"],
            args.expected_input_train_seal_sha256,
            "input train seal",
        ),
        "dev": require_hash(
            paths["input_dev"],
            args.expected_input_dev_sha256,
            "input dev",
        ),
        "dev_seal": require_hash(
            paths["input_dev_seal"],
            args.expected_input_dev_seal_sha256,
            "input dev seal",
        ),
        "contract": require_hash(
            paths["contract"], args.expected_contract_sha256, "contract"
        ),
        "evaluator": require_hash(
            paths["evaluator"],
            args.expected_evaluator_sha256,
            "evaluator",
        ),
    }
    contract_sha = inputs["contract"]["sha256"]
    validate_input_seal(
        dataset=paths["input_train"],
        seal_path=paths["input_train_seal"],
        contract_sha256=contract_sha,
        expected_role="fit",
    )
    validate_input_seal(
        dataset=paths["input_dev"],
        seal_path=paths["input_dev_seal"],
        contract_sha256=contract_sha,
        expected_role="measure",
    )
    evaluator = load_evaluator(paths["evaluator"])
    observed_dart_version = dart_version(str(evaluator.DART_BIN))
    if observed_dart_version != args.expected_dart_version:
        raise SanitationError(
            "Dart runtime mismatch: expected "
            f"{args.expected_dart_version!r}, observed {observed_dart_version!r}"
        )

    train_rows = read_jsonl(paths["input_train"], "input train")
    dev_rows = read_jsonl(paths["input_dev"], "input dev")
    if len(train_rows) != args.expected_input_train_rows:
        raise SanitationError("input train row count mismatch")
    if len(dev_rows) != args.expected_dev_rows:
        raise SanitationError("input dev row count mismatch")
    all_ids = [
        str(row.get("task_id") or "") for row in train_rows + dev_rows
    ]
    if any(not task_id for task_id in all_ids):
        raise SanitationError("input row has no task_id")
    if len(set(all_ids)) != len(all_ids):
        raise SanitationError("input task IDs are not globally unique")

    expected_sanitized_train = parse_task_ids(
        args.expected_sanitized_train_task_ids,
        "expected sanitized train IDs",
    )
    expected_quarantined_train = parse_task_ids(
        args.expected_quarantined_train_task_ids,
        "expected quarantined train IDs",
    )
    expected_sanitized_dev = parse_task_ids(
        args.expected_sanitized_dev_task_ids,
        "expected sanitized dev IDs",
    )

    changes: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    def sanitize_split(
        rows: Sequence[Mapping[str, Any]], split: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        imitation: list[dict[str, Any]] = []
        executable: list[dict[str, Any]] = []
        for raw in rows:
            row = dict(raw)
            task_id = str(row["task_id"])
            source = row.get("dart_source")
            if not isinstance(source, str) or not source.strip():
                raise SanitationError(f"{task_id}: missing dart_source")
            row_tests(row, task_id)
            forbidden_before = evaluator.disallowed_dart_test_runtime_library(
                source
            )
            if not forbidden_before:
                imitation.append(row)
                executable.append(row)
                continue
            sanitized_source, removed = strip_forbidden_imports(source)
            forbidden_after = evaluator.disallowed_dart_test_runtime_library(
                sanitized_source
            )
            if not removed or forbidden_after:
                raise SanitationError(
                    f"{task_id}: forbidden import could not be removed "
                    f"exactly (before={forbidden_before!r}, "
                    f"after={forbidden_after!r})"
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
            record = {
                "task_id": task_id,
                "split": split,
                "removed_import_uris": sorted(set(removed)),
                "source_sha256_before": sha256_text(source),
                "source_sha256_after": sha256_text(sanitized_source),
                "tests_sha256": outcome["tests_sha256"],
                "compiled_after_removal": outcome["compiled"],
                "passed_after_removal": outcome["passed"],
                "diagnostic": outcome["diagnostic"],
            }
            if outcome["compiled"] and outcome["passed"]:
                imitation.append(candidate)
                executable.append(candidate)
                changes.append(record)
            elif split == "train":
                # Sequence imitation is an offline likelihood objective and
                # does not execute target code. Keep the complete fit
                # distribution there; only the executable-reward view omits
                # this row.
                imitation.append(row)
                quarantine.append(
                    {
                        **record,
                        "reason": (
                            "forbidden_import_is_semantically_required;"
                            "sequence_imitation_only"
                        ),
                        "original_row": row,
                    }
                )
            else:
                raise SanitationError(
                    f"{task_id}: dev row requires forbidden runtime import: "
                    f"{outcome['diagnostic']}"
                )
        return imitation, executable

    imitation_train_rows, output_train_rows = sanitize_split(
        train_rows, "train"
    )
    imitation_dev_rows, output_dev_rows = sanitize_split(dev_rows, "dev")
    if imitation_dev_rows != output_dev_rows:
        raise SanitationError("dev imitation/executable views diverged")
    sanitized_train = {
        item["task_id"] for item in changes if item["split"] == "train"
    }
    sanitized_dev = {
        item["task_id"] for item in changes if item["split"] == "dev"
    }
    quarantined_train = {item["task_id"] for item in quarantine}
    if sanitized_train != expected_sanitized_train:
        raise SanitationError(
            "sanitized train task set mismatch: "
            f"expected={sorted(expected_sanitized_train)} "
            f"observed={sorted(sanitized_train)}"
        )
    if sanitized_dev != expected_sanitized_dev:
        raise SanitationError(
            "sanitized dev task set mismatch: "
            f"expected={sorted(expected_sanitized_dev)} "
            f"observed={sorted(sanitized_dev)}"
        )
    if quarantined_train != expected_quarantined_train:
        raise SanitationError(
            "quarantined train task set mismatch: "
            f"expected={sorted(expected_quarantined_train)} "
            f"observed={sorted(quarantined_train)}"
        )
    if len(output_train_rows) != args.expected_output_train_rows:
        raise SanitationError("output train row count mismatch")
    if len(imitation_train_rows) != args.expected_input_train_rows:
        raise SanitationError("imitation train row count mismatch")
    if len(output_dev_rows) != args.expected_dev_rows:
        raise SanitationError("output dev row count mismatch")

    # Recertify every emitted gold target, including unchanged rows.  This is
    # deliberately after task-set assertions so no failure can be hidden by an
    # accidental row drop.
    jobs = [
        ("train", row) for row in output_train_rows
    ] + [("dev", row) for row in output_dev_rows]

    def recertify(item: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
        split, row = item
        return {
            "split": split,
            **evaluate_row(
                evaluator,
                row,
                task_suffix="gold_recertify",
                timeout=args.timeout,
                stability_runs=args.stability_runs,
            ),
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as pool:
        recertification = list(pool.map(recertify, jobs))
    failures = [item for item in recertification if not item["passed"]]
    if failures:
        preview = [
            {
                "split": item["split"],
                "task_id": item["task_id"],
                "compiled": item["compiled"],
                "diagnostic": item["diagnostic"][:500],
            }
            for item in failures[:10]
        ]
        raise SanitationError(
            f"{len(failures)} emitted gold targets failed recertification: "
            f"{preview}"
        )

    output_paths = {
        "imitation_train": args.output_imitation_train.expanduser().resolve(),
        "imitation_train_seal": (
            args.output_imitation_train_seal.expanduser().resolve()
        ),
        "train": args.output_train.expanduser().resolve(),
        "train_seal": args.output_train_seal.expanduser().resolve(),
        "dev": args.output_dev.expanduser().resolve(),
        "dev_seal": args.output_dev_seal.expanduser().resolve(),
        "quarantine": args.quarantine.expanduser().resolve(),
        "report": args.report.expanduser().resolve(),
    }
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    atomic_write_jsonl(
        output_paths["imitation_train"], imitation_train_rows
    )
    atomic_write_jsonl(output_paths["train"], output_train_rows)
    atomic_write_jsonl(output_paths["dev"], output_dev_rows)
    atomic_write_jsonl(output_paths["quarantine"], quarantine)
    sanitizer_sha = sha256_file(Path(__file__).resolve())
    quarantine_record = file_record(output_paths["quarantine"])
    common_seal = {
        "schema": SEAL_SCHEMA,
        "contract_sha256": contract_sha,
        "sanitation_schema": BUILD_SCHEMA,
        "sanitizer_sha256": sanitizer_sha,
        "evaluator_sha256": inputs["evaluator"]["sha256"],
        "completion_attestation_id": evaluator.COMPLETION_ATTESTATION_ID,
        "dart_version": observed_dart_version,
        "stability_runs": args.stability_runs,
        "quarantine_sha256": quarantine_record["sha256"],
    }
    atomic_write_json(
        output_paths["imitation_train_seal"],
        {
            **common_seal,
            "selected_role": "fit",
            "training_objective_scope": "sequence_imitation_all_train",
            "output_sha256": sha256_file(output_paths["imitation_train"]),
            "input_sha256": inputs["train"]["sha256"],
            "rows": len(imitation_train_rows),
            "executable_reward_eligible_rows": len(output_train_rows),
            "execution_ineligible_task_ids": sorted(quarantined_train),
        },
    )
    atomic_write_json(
        output_paths["train_seal"],
        {
            **common_seal,
            "selected_role": "fit",
            "training_objective_scope": "executable_reward_only",
            "output_sha256": sha256_file(output_paths["train"]),
            "input_sha256": inputs["train"]["sha256"],
            "rows": len(output_train_rows),
        },
    )
    atomic_write_json(
        output_paths["dev_seal"],
        {
            **common_seal,
            "selected_role": "measure",
            "output_sha256": sha256_file(output_paths["dev"]),
            "input_sha256": inputs["dev"]["sha256"],
            "rows": len(output_dev_rows),
        },
    )
    report = {
        "schema": BUILD_SCHEMA,
        "inputs": inputs,
        "sanitizer_sha256": sanitizer_sha,
        "runtime": {
            "dart_binary": str(evaluator.DART_BIN),
            "dart_version": observed_dart_version,
            "evaluator_sha256": inputs["evaluator"]["sha256"],
            "completion_attestation_id": evaluator.COMPLETION_ATTESTATION_ID,
            "stability_runs": args.stability_runs,
        },
        "policy": {
            "trusted_harness_imports": ["dart:async", "dart:convert"],
            "forbidden_imports_removed_only_after_compile_and_test_pass": True,
            "required_forbidden_imports_quarantined_from_train": True,
            "required_forbidden_imports_retained_for_sequence_imitation": True,
            "dev_quarantine_allowed": False,
            "expected_output_rewritten": False,
            "all_emitted_gold_targets_recertified": True,
        },
        "counts": {
            "input_train": len(train_rows),
            "output_imitation_train": len(imitation_train_rows),
            "output_train": len(output_train_rows),
            "input_dev": len(dev_rows),
            "output_dev": len(output_dev_rows),
            "sanitized_train": len(sanitized_train),
            "sanitized_dev": len(sanitized_dev),
            "quarantined_train": len(quarantined_train),
            "recertified": len(recertification),
            "recertification_failures": 0,
        },
        "task_sets": {
            "sanitized_train": sorted(sanitized_train),
            "sanitized_dev": sorted(sanitized_dev),
            "quarantined_train": sorted(quarantined_train),
        },
        "changes": sorted(changes, key=lambda item: item["task_id"]),
        "quarantine_summary": [
            {
                key: item[key]
                for key in (
                    "task_id",
                    "split",
                    "removed_import_uris",
                    "source_sha256_before",
                    "source_sha256_after",
                    "tests_sha256",
                    "reason",
                    "diagnostic",
                )
            }
            for item in sorted(quarantine, key=lambda value: value["task_id"])
        ],
        "outputs": {
            "imitation_train": file_record(
                output_paths["imitation_train"]
            ),
            "imitation_train_seal": file_record(
                output_paths["imitation_train_seal"]
            ),
            "train": file_record(output_paths["train"]),
            "train_seal": file_record(output_paths["train_seal"]),
            "dev": file_record(output_paths["dev"]),
            "dev_seal": file_record(output_paths["dev_seal"]),
            "quarantine": quarantine_record,
        },
    }
    atomic_write_json(output_paths["report"], report)
    return report


def main() -> int:
    report = build(parse_args())
    print(
        "COMPACT_TARGET_SANITATION "
        f"imitation_train={report['counts']['output_imitation_train']} "
        f"train={report['counts']['output_train']} "
        f"dev={report['counts']['output_dev']} "
        f"sanitized={report['counts']['sanitized_train'] + report['counts']['sanitized_dev']} "
        f"quarantined={report['counts']['quarantined_train']} "
        f"train_sha256={report['outputs']['train']['sha256']} "
        f"dev_sha256={report['outputs']['dev']['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
