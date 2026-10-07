#!/usr/bin/env python3
"""Derive the exact executable-reward view from the 1,580-row imitation build.

Qwen sequence imitation is allowed to use every sealed training target.  Local
execution rewards are not: two audited programs have filesystem/FFI side
effects and must never reach RS candidate replay or VeRPO.  This module derives
one 1,578-row compact+F2 view, preserving row order and every representation
byte for the remaining tasks, and cryptographically binds it to the complete
multi-function build and the untouched 175-row measure split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "binary-multifunction-executable-view-v1"
DERIVATION_SCHEMA = "binary-multifunction-executable-subset-v1"
PARENT_BUILD_SCHEMA = "binary-multifunction-compact-build-v2"
REPRESENTATION_SCHEMA = "binary-multifunction-v1-semantic-adapter-v1"
JOIN_SEAL_SCHEMA = "compact-public-private-join-seal-v1"
F2_MANIFEST_SCHEMA = "verified-api-readable-compact-v2"
F2_REPRESENTATION_SCHEMA = "lossless-semantic-f2"
PARENT_TRAIN_SCOPE = "sequence_imitation_all_train"
EXECUTABLE_SCOPE = "executable_reward_only"

EXPECTED_PARENT_ROWS = 1580
EXPECTED_EXECUTABLE_ROWS = 1578
EXPECTED_HELDOUT_ROWS = 175
EXECUTION_INELIGIBLE_TASK_IDS = frozenset(
    {
        "sigless_bfde11b99b84",  # audited filesystem write
        "sigless_67bb88ce699e",  # audited dart:ffi/native access
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ExecutableViewError(ValueError):
    """The executable subset cannot be proven from sealed parent artifacts."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ExecutableViewError(f"{label} is not a lowercase SHA-256")
    return digest


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    size = resolved.stat().st_size
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": size,
        "size_bytes": size,
    }


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExecutableViewError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutableViewError(f"{label} is not a JSON object")
    return value


def load_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ExecutableViewError(
                    f"{label}:{line_number}: blank rows are forbidden"
                )
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ExecutableViewError(
                    f"{label}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ExecutableViewError(
                    f"{label}:{line_number}: row is not an object"
                )
            rows.append(row)
    if not rows:
        raise ExecutableViewError(f"{label}: no rows")
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validated_record(
    value: Any,
    *,
    label: str,
    expected_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ExecutableViewError(f"{label} is not a file record")
    raw_path = str(value.get("path") or "")
    expected_sha = require_sha256(value.get("sha256"), f"{label} SHA-256")
    if not raw_path:
        raise ExecutableViewError(f"{label} has no path")
    path = Path(raw_path).expanduser().resolve()
    if expected_path is not None and path != Path(expected_path).expanduser().resolve():
        raise ExecutableViewError(f"{label} path mismatch")
    observed = file_record(path)
    if observed["sha256"] != expected_sha:
        raise ExecutableViewError(f"{label} content hash mismatch")
    expected_size = value.get("size_bytes", value.get("bytes"))
    if expected_size is not None and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size != observed["size_bytes"]
    ):
        raise ExecutableViewError(f"{label} byte-size mismatch")
    return path, observed


def _task_ids(
    rows: list[dict[str, Any]],
    *,
    label: str,
    expected_rows: int,
) -> list[str]:
    if len(rows) != expected_rows:
        raise ExecutableViewError(
            f"{label} has {len(rows)} rows, expected {expected_rows}"
        )
    result: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            raise ExecutableViewError(
                f"{label} row {index} has a missing/duplicate task_id"
            )
        seen.add(task_id)
        result.append(task_id)
    return result


def _validate_parent(
    report_path: Path,
    *,
    expected_report_sha256: str,
) -> dict[str, Any]:
    expected = require_sha256(
        expected_report_sha256, "expected parent build report SHA-256"
    )
    if sha256_file(report_path) != expected:
        raise ExecutableViewError("parent build report hash mismatch")
    report = load_json(report_path, "parent multi-function build report")
    invariants = report.get("invariants")
    counts = report.get("counts")
    if (
        report.get("schema") != PARENT_BUILD_SCHEMA
        or report.get("representation_schema") != REPRESENTATION_SCHEMA
        or report.get("passed") is not True
        or not isinstance(invariants, Mapping)
        or not isinstance(counts, Mapping)
        or int(counts.get("train_rows", -1)) != EXPECTED_PARENT_ROWS
        or int(counts.get("dev_rows", -1)) != EXPECTED_HELDOUT_ROWS
        or int(counts.get("excluded_rows", -1)) != 0
        or int(counts.get("truncated_rows", -1)) != 0
    ):
        raise ExecutableViewError("parent multi-function build contract failed")
    required_invariants = (
        "all_user_functions_retained",
        "all_machine_instructions_retained",
        "all_cfg_edges_retained_with_global_offsets",
        "all_external_aliases_and_exact_definitions_retained",
        "source_token_id_set_preserved_from_parent",
        "block_and_control_token_ids_preserved_from_parent",
        "instruction_codebook_refit_from_train_only",
        "warmstart_overlay_rows_reusable_only_when_expansions_match",
        "inline_cfg_source_is_current_containing_block",
        "inline_cfg_omits_only_redundant_edge_source_tokens",
        "all_inline_cfg_text_and_token_roundtrips_verified",
        "all_f2_semantic_roundtrips_verified",
        "all_student_rows_within_9000",
        "all_api_prompts_within_12000",
        "zero_excluded_rows",
        "zero_truncated_rows",
        "train_dev_task_sets_disjoint",
        "dev_is_measure_only_and_not_training",
        "train_dev_representation_contract_identical",
    )
    if any(invariants.get(field) is not True for field in required_invariants):
        raise ExecutableViewError(
            "parent multi-function build invariants are incomplete"
        )
    if int(invariants.get("heldout_rows_used_for_instruction_codebook_fit", -1)) != 0:
        raise ExecutableViewError(
            "heldout rows influenced the parent instruction-codebook refit"
        )
    derived = report.get("derived_representation")
    outputs = report.get("outputs")
    if not isinstance(derived, Mapping) or not isinstance(outputs, Mapping):
        raise ExecutableViewError(
            "parent build lacks the derived v2 representation records"
        )
    for name in ("contract", "codebook"):
        derived_record = derived.get(name)
        output_record = outputs.get(name)
        if (
            not isinstance(derived_record, Mapping)
            or not isinstance(output_record, Mapping)
            or derived_record.get("sha256") != output_record.get("sha256")
        ):
            raise ExecutableViewError(
                f"parent derived representation {name} record mismatch"
            )
    return report


def _resolve_parent_artifacts(
    report: Mapping[str, Any],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    outputs = report.get("outputs")
    inputs = report.get("inputs")
    if not isinstance(outputs, Mapping) or not isinstance(inputs, Mapping):
        raise ExecutableViewError("parent build has no sealed input/output records")
    result = {
        name: _validated_record(outputs.get(name), label=f"parent {name}")
        for name in (
            "train",
            "train_seal",
            "train_f2",
            "train_f2_manifest",
            "dev",
            "dev_seal",
        )
    }
    # Build-v2 refits instruction atoms on train only and emits a new contract
    # whose expansion table binds those meanings.  The old input contract is
    # provenance only; downstream training must consume the derived output.
    result["contract"] = _validated_record(
        outputs.get("contract"), label="derived representation contract"
    )
    result["codebook"] = _validated_record(
        outputs.get("codebook"), label="derived representation codebook"
    )
    return result


def build_executable_view(
    *,
    parent_build_report: str | Path,
    expected_parent_build_report_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    report_path = Path(parent_build_report).expanduser().resolve()
    parent = _validate_parent(
        report_path,
        expected_report_sha256=expected_parent_build_report_sha256,
    )
    artifacts = _resolve_parent_artifacts(parent)
    train_path = artifacts["train"][0]
    train_seal_path = artifacts["train_seal"][0]
    f2_path = artifacts["train_f2"][0]
    f2_manifest_path = artifacts["train_f2_manifest"][0]
    dev_path = artifacts["dev"][0]
    dev_seal_path = artifacts["dev_seal"][0]

    train = load_jsonl(train_path, "parent train")
    train_ids = _task_ids(
        train, label="parent train", expected_rows=EXPECTED_PARENT_ROWS
    )
    f2 = load_jsonl(f2_path, "parent train F2")
    f2_ids = _task_ids(
        f2, label="parent train F2", expected_rows=EXPECTED_PARENT_ROWS
    )
    if f2_ids != train_ids:
        raise ExecutableViewError("parent compact/F2 train order differs")
    dev = load_jsonl(dev_path, "parent heldout")
    dev_ids = _task_ids(
        dev, label="parent heldout", expected_rows=EXPECTED_HELDOUT_ROWS
    )
    if set(train_ids).intersection(dev_ids):
        raise ExecutableViewError("parent train and heldout task sets overlap")

    train_seal = load_json(train_seal_path, "parent train seal")
    dev_seal = load_json(dev_seal_path, "parent heldout seal")
    if (
        train_seal.get("schema") != JOIN_SEAL_SCHEMA
        or train_seal.get("selected_role") != "fit"
        or train_seal.get("training_allowed") is not True
        or train_seal.get("training_objective_scope") != PARENT_TRAIN_SCOPE
        or int(train_seal.get("rows", -1)) != EXPECTED_PARENT_ROWS
        or train_seal.get("output_sha256") != artifacts["train"][1]["sha256"]
        or int(train_seal.get("executable_reward_eligible_rows", -1))
        != EXPECTED_EXECUTABLE_ROWS
        or set(train_seal.get("execution_ineligible_task_ids") or [])
        != EXECUTION_INELIGIBLE_TASK_IDS
        or train_seal.get("contract_sha256")
        != artifacts["contract"][1]["sha256"]
    ):
        raise ExecutableViewError("parent train seal eligibility contract failed")
    if (
        dev_seal.get("schema") != JOIN_SEAL_SCHEMA
        or dev_seal.get("selected_role") != "measure"
        or dev_seal.get("training_allowed") is not False
        or dev_seal.get("heldout_measure_only") is not True
        or int(dev_seal.get("rows", -1)) != EXPECTED_HELDOUT_ROWS
        or dev_seal.get("output_sha256") != artifacts["dev"][1]["sha256"]
        or dev_seal.get("contract_sha256")
        != artifacts["contract"][1]["sha256"]
    ):
        raise ExecutableViewError("parent heldout seal contract failed")
    if not EXECUTION_INELIGIBLE_TASK_IDS.issubset(train_ids):
        raise ExecutableViewError(
            "the two audited execution-ineligible tasks are absent from parent train"
        )

    parent_f2_manifest = load_json(
        f2_manifest_path, "parent train F2 manifest"
    )
    if (
        parent_f2_manifest.get("schema") != F2_MANIFEST_SCHEMA
        or int(parent_f2_manifest.get("rows", -1)) != EXPECTED_PARENT_ROWS
        or (parent_f2_manifest.get("dataset") or {}).get("sha256")
        != artifacts["train"][1]["sha256"]
        or (parent_f2_manifest.get("output") or {}).get("sha256")
        != artifacts["train_f2"][1]["sha256"]
        or (
            parent_f2_manifest.get("f2_prompt_contract") or {}
        ).get("representation_schema")
        != F2_REPRESENTATION_SCHEMA
    ):
        raise ExecutableViewError("parent F2 manifest contract failed")

    executable_train = [
        row
        for row in train
        if str(row["task_id"]) not in EXECUTION_INELIGIBLE_TASK_IDS
    ]
    executable_f2 = [
        row
        for row in f2
        if str(row["task_id"]) not in EXECUTION_INELIGIBLE_TASK_IDS
    ]
    executable_ids = _task_ids(
        executable_train,
        label="derived executable train",
        expected_rows=EXPECTED_EXECUTABLE_ROWS,
    )
    if _task_ids(
        executable_f2,
        label="derived executable train F2",
        expected_rows=EXPECTED_EXECUTABLE_ROWS,
    ) != executable_ids:
        raise AssertionError("derived compact/F2 order changed")
    if set(executable_ids).intersection(dev_ids):
        raise AssertionError("derived executable train overlaps heldout")

    destination = Path(output_dir).expanduser().resolve()
    paths = {
        "dataset": destination / "train_multifunction_binary_executable.jsonl",
        "seal": destination
        / "train_multifunction_binary_executable.seal.json",
        "f2": destination
        / "train_multifunction_binary_executable_f2.jsonl",
        "f2_manifest": destination
        / "train_multifunction_binary_executable_f2.jsonl.manifest.json",
        "report": destination / "executable_view.build.json",
        "contract": destination / "compact_contract.json",
    }
    if any(path.exists() for path in paths.values()):
        existing = [str(path) for path in paths.values() if path.exists()]
        raise FileExistsError(
            "refusing to overwrite executable-view artifacts: "
            + ", ".join(existing)
        )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(paths["dataset"], executable_train)
    _atomic_jsonl(paths["f2"], executable_f2)
    shutil.copyfile(artifacts["contract"][0], paths["contract"])
    if sha256_file(paths["contract"]) != artifacts["contract"][1]["sha256"]:
        raise RuntimeError("executable-view contract copy is not byte-identical")

    parent_record = file_record(report_path)
    derivation = {
        "schema": DERIVATION_SCHEMA,
        "parent_build_report": parent_record,
        "parent_dataset": artifacts["train"][1],
        "parent_dataset_seal": artifacts["train_seal"][1],
        "parent_f2": artifacts["train_f2"][1],
        "parent_f2_manifest": artifacts["train_f2_manifest"][1],
        "parent_rows": EXPECTED_PARENT_ROWS,
        "output_rows": EXPECTED_EXECUTABLE_ROWS,
        "excluded_task_ids": sorted(EXECUTION_INELIGIBLE_TASK_IDS),
        "excluded_task_ids_sha256": stable_sha256(
            sorted(EXECUTION_INELIGIBLE_TASK_IDS)
        ),
        "selection": "stable_parent_order_minus_exact_audited_ids",
        "row_content_transform": "identity",
    }
    f2_manifest = dict(parent_f2_manifest)
    f2_manifest.update(
        {
            "created_at": utc_now(),
            "rows": EXPECTED_EXECUTABLE_ROWS,
            "dataset": file_record(paths["dataset"]),
            "task_set_sha256": stable_sha256(executable_ids),
            "output": file_record(paths["f2"]),
            "derivation": derivation,
            "training_objective_scope": EXECUTABLE_SCOPE,
        }
    )
    f2_manifest["invariants"] = dict(
        parent_f2_manifest.get("invariants") or {}
    ) | {
        "exact_audited_execution_exclusions_applied": True,
        "all_remaining_rows_byte_identical_to_parent": True,
        "heldout_175_disjoint_and_untouched": True,
    }
    _atomic_json(paths["f2_manifest"], f2_manifest)

    copied_seal_fields = (
        "contract_sha256",
        "representation_schema",
        "frontier_f2_schema",
        "adapter_contract_sha256",
        "adapter_script_sha256",
        "source_function_bundles_sha256",
        "source_symbol_attestation_used",
        "source_symbol_attestation_is_keyed",
        "source_symbol_attestation_file_sha256",
        "source_symbol_attestation_key_id_sha256",
        "raw_source_names_serialized",
        "sanitation_schema",
        "sanitizer_sha256",
        "evaluator_sha256",
        "completion_attestation_id",
        "dart_version",
        "stability_runs",
        "quarantine_sha256",
    )
    seal = {
        "schema": JOIN_SEAL_SCHEMA,
        "selected_role": "fit",
        "training_allowed": True,
        "heldout_measure_only": False,
        "rows": EXPECTED_EXECUTABLE_ROWS,
        "task_set_sha256": stable_sha256(executable_ids),
        "output_sha256": sha256_file(paths["dataset"]),
        "output": file_record(paths["dataset"]),
        "f2_output": file_record(paths["f2"]),
        "f2_manifest": file_record(paths["f2_manifest"]),
        "training_objective_scope": EXECUTABLE_SCOPE,
        "executable_reward_eligible_rows": EXPECTED_EXECUTABLE_ROWS,
        "execution_ineligible_task_ids": [],
        "excluded_from_parent_task_ids": sorted(
            EXECUTION_INELIGIBLE_TASK_IDS
        ),
        "derivation": derivation,
    }
    for field in copied_seal_fields:
        if field not in train_seal:
            raise ExecutableViewError(
                f"parent train seal lacks required field {field!r}"
            )
        seal[field] = train_seal[field]
    _atomic_json(paths["seal"], seal)

    result = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "passed": True,
        "representation_schema": REPRESENTATION_SCHEMA,
        "training_objective_scope": EXECUTABLE_SCOPE,
        "parent": {
            "build_report": parent_record,
            "train": artifacts["train"][1],
            "train_seal": artifacts["train_seal"][1],
            "train_f2": artifacts["train_f2"][1],
            "train_f2_manifest": artifacts["train_f2_manifest"][1],
        },
        "heldout_measure_only": {
            "dataset": artifacts["dev"][1],
            "seal": artifacts["dev_seal"][1],
            "rows": EXPECTED_HELDOUT_ROWS,
            "task_set_sha256": stable_sha256(dev_ids),
            "untouched": True,
        },
        "contract": artifacts["contract"][1],
        "counts": {
            "parent_train_rows": EXPECTED_PARENT_ROWS,
            "execution_ineligible_rows": len(
                EXECUTION_INELIGIBLE_TASK_IDS
            ),
            "executable_train_rows": EXPECTED_EXECUTABLE_ROWS,
            "heldout_rows": EXPECTED_HELDOUT_ROWS,
        },
        "excluded_task_ids": sorted(EXECUTION_INELIGIBLE_TASK_IDS),
        "outputs": {
            "dataset": file_record(paths["dataset"]),
            "seal": file_record(paths["seal"]),
            "f2": file_record(paths["f2"]),
            "f2_manifest": file_record(paths["f2_manifest"]),
            "contract": file_record(paths["contract"]),
        },
        "digests": {
            "parent_train_task_ids_sha256": stable_sha256(train_ids),
            "executable_train_task_ids_sha256": stable_sha256(executable_ids),
            "heldout_task_ids_sha256": stable_sha256(dev_ids),
        },
        "invariants": {
            "only_exact_audited_fs_ffi_rows_excluded": True,
            "all_remaining_compact_rows_byte_identical_to_parent": True,
            "all_remaining_f2_rows_byte_identical_to_parent": True,
            "compact_and_f2_task_order_identical": True,
            "executable_train_has_1578_unique_tasks": True,
            "heldout_has_175_unique_tasks": True,
            "train_heldout_disjoint": True,
            "heldout_not_rewritten": True,
            "parent_full_imitation_view_retained": True,
        },
    }
    _atomic_json(paths["report"], result)
    print(
        "MULTIFUNCTION_EXECUTABLE_VIEW "
        f"train={EXPECTED_EXECUTABLE_ROWS} heldout={EXPECTED_HELDOUT_ROWS} "
        f"train_sha256={result['outputs']['dataset']['sha256']} "
        f"f2_sha256={result['outputs']['f2']['sha256']}",
        flush=True,
    )
    return result


def validate_executable_view(
    *,
    dataset: str | Path,
    seal: str | Path,
    f2: str | Path,
    f2_manifest: str | Path,
    build_report: str | Path,
    expected_build_report_sha256: str | None = None,
    contract: str | Path | None = None,
    verify_heldout: bool = True,
) -> dict[str, Any]:
    """Validate a materialized view and return its sealed provenance."""

    paths = {
        "dataset": Path(dataset).expanduser().resolve(),
        "seal": Path(seal).expanduser().resolve(),
        "f2": Path(f2).expanduser().resolve(),
        "f2_manifest": Path(f2_manifest).expanduser().resolve(),
        "report": Path(build_report).expanduser().resolve(),
    }
    if expected_build_report_sha256 is not None:
        expected = require_sha256(
            expected_build_report_sha256,
            "expected executable-view build report SHA-256",
        )
        if sha256_file(paths["report"]) != expected:
            raise ExecutableViewError(
                "executable-view build report hash mismatch"
            )
    report = load_json(paths["report"], "executable-view build report")
    if (
        report.get("schema") != SCHEMA
        or report.get("passed") is not True
        or report.get("representation_schema") != REPRESENTATION_SCHEMA
        or report.get("training_objective_scope") != EXECUTABLE_SCOPE
        or report.get("excluded_task_ids")
        != sorted(EXECUTION_INELIGIBLE_TASK_IDS)
        or int((report.get("counts") or {}).get("executable_train_rows", -1))
        != EXPECTED_EXECUTABLE_ROWS
        or int((report.get("counts") or {}).get("heldout_rows", -1))
        != EXPECTED_HELDOUT_ROWS
    ):
        raise ExecutableViewError("executable-view build report contract failed")
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ExecutableViewError("executable-view report has no outputs")
    observed_records: dict[str, dict[str, Any]] = {}
    for name in ("dataset", "seal", "f2", "f2_manifest"):
        _path, record = _validated_record(
            outputs.get(name), label=f"executable {name}", expected_path=paths[name]
        )
        observed_records[name] = record
    report_contract = (report.get("outputs") or {}).get("contract")
    if not isinstance(report_contract, Mapping):
        raise ExecutableViewError("executable-view report has no contract copy")
    contract_path_from_report, contract_record = _validated_record(
        report_contract, label="executable contract"
    )

    rows = load_jsonl(paths["dataset"], "executable dataset")
    row_ids = _task_ids(
        rows,
        label="executable dataset",
        expected_rows=EXPECTED_EXECUTABLE_ROWS,
    )
    f2_rows = load_jsonl(paths["f2"], "executable F2")
    f2_ids = _task_ids(
        f2_rows,
        label="executable F2",
        expected_rows=EXPECTED_EXECUTABLE_ROWS,
    )
    if f2_ids != row_ids:
        raise ExecutableViewError("executable compact/F2 task order differs")
    if set(row_ids).intersection(EXECUTION_INELIGIBLE_TASK_IDS):
        raise ExecutableViewError(
            "an audited execution-ineligible task entered the executable view"
        )
    for index, row in enumerate(rows):
        if row.get("binary_multifunction_schema") != REPRESENTATION_SCHEMA:
            raise ExecutableViewError(
                f"executable row {index} is not the multi-function representation"
            )

    seal_value = load_json(paths["seal"], "executable seal")
    if (
        seal_value.get("schema") != JOIN_SEAL_SCHEMA
        or seal_value.get("selected_role") != "fit"
        or seal_value.get("training_allowed") is not True
        or seal_value.get("heldout_measure_only") is not False
        or seal_value.get("training_objective_scope") != EXECUTABLE_SCOPE
        or int(seal_value.get("rows", -1)) != EXPECTED_EXECUTABLE_ROWS
        or int(seal_value.get("executable_reward_eligible_rows", -1))
        != EXPECTED_EXECUTABLE_ROWS
        or seal_value.get("execution_ineligible_task_ids") != []
        or seal_value.get("excluded_from_parent_task_ids")
        != sorted(EXECUTION_INELIGIBLE_TASK_IDS)
        or seal_value.get("output_sha256")
        != observed_records["dataset"]["sha256"]
        or seal_value.get("representation_schema") != REPRESENTATION_SCHEMA
    ):
        raise ExecutableViewError("executable seal contract failed")
    if contract is not None:
        contract_path = Path(contract).expanduser().resolve()
        if (
            sha256_file(contract_path)
            != require_sha256(
                seal_value.get("contract_sha256"),
                "executable seal contract SHA-256",
            )
        ):
            raise ExecutableViewError("executable contract hash mismatch")

    f2_value = load_json(paths["f2_manifest"], "executable F2 manifest")
    derivation = f2_value.get("derivation")
    if (
        f2_value.get("schema") != F2_MANIFEST_SCHEMA
        or f2_value.get("training_objective_scope") != EXECUTABLE_SCOPE
        or int(f2_value.get("rows", -1)) != EXPECTED_EXECUTABLE_ROWS
        or (f2_value.get("dataset") or {}).get("sha256")
        != observed_records["dataset"]["sha256"]
        or (f2_value.get("output") or {}).get("sha256")
        != observed_records["f2"]["sha256"]
        or not isinstance(derivation, Mapping)
        or derivation.get("schema") != DERIVATION_SCHEMA
        or derivation.get("excluded_task_ids")
        != sorted(EXECUTION_INELIGIBLE_TASK_IDS)
        or int(derivation.get("parent_rows", -1)) != EXPECTED_PARENT_ROWS
        or int(derivation.get("output_rows", -1))
        != EXPECTED_EXECUTABLE_ROWS
    ):
        raise ExecutableViewError("executable F2 derivation contract failed")
    parent_prompt = derivation.get("parent_f2")
    parent_manifest = derivation.get("parent_f2_manifest")
    _validated_record(parent_prompt, label="parent full F2")
    _validated_record(parent_manifest, label="parent full F2 manifest")

    heldout = report.get("heldout_measure_only")
    if (
        not isinstance(heldout, Mapping)
        or heldout.get("untouched") is not True
        or int(heldout.get("rows", -1)) != EXPECTED_HELDOUT_ROWS
    ):
        raise ExecutableViewError("heldout-175 attestation is absent")
    heldout_dataset_value = heldout.get("dataset")
    heldout_seal_value = heldout.get("seal")
    if not isinstance(heldout_dataset_value, Mapping) or not isinstance(
        heldout_seal_value, Mapping
    ):
        raise ExecutableViewError("heldout-175 file records are absent")
    require_sha256(
        heldout_dataset_value.get("sha256"), "heldout dataset SHA-256"
    )
    require_sha256(heldout_seal_value.get("sha256"), "heldout seal SHA-256")
    heldout_record = dict(heldout_dataset_value)
    heldout_seal_record = dict(heldout_seal_value)
    heldout_task_ids_sha256 = str(
        report.get("digests", {}).get("heldout_task_ids_sha256") or ""
    )
    require_sha256(
        heldout_task_ids_sha256, "heldout task-set attestation SHA-256"
    )
    if verify_heldout:
        heldout_path, heldout_record = _validated_record(
            heldout_dataset_value, label="heldout dataset"
        )
        heldout_seal_path, heldout_seal_record = _validated_record(
            heldout_seal_value, label="heldout seal"
        )
        heldout_rows = load_jsonl(heldout_path, "heldout dataset")
        heldout_ids = _task_ids(
            heldout_rows,
            label="heldout dataset",
            expected_rows=EXPECTED_HELDOUT_ROWS,
        )
        if set(row_ids).intersection(heldout_ids):
            raise ExecutableViewError("executable train overlaps heldout-175")
        heldout_seal = load_json(heldout_seal_path, "heldout seal")
        if (
            heldout_seal.get("selected_role") != "measure"
            or heldout_seal.get("training_allowed") is not False
            or heldout_seal.get("heldout_measure_only") is not True
            or int(heldout_seal.get("rows", -1)) != EXPECTED_HELDOUT_ROWS
            or heldout_seal.get("output_sha256") != heldout_record["sha256"]
        ):
            raise ExecutableViewError("heldout-175 seal contract failed")
        heldout_task_ids_sha256 = stable_sha256(heldout_ids)

    return {
        "schema": SCHEMA,
        "report": file_record(paths["report"]),
        "dataset": observed_records["dataset"],
        "seal": observed_records["seal"],
        "f2": observed_records["f2"],
        "f2_manifest": observed_records["f2_manifest"],
        "contract": contract_record,
        "parent_f2": dict(parent_prompt),
        "parent_f2_manifest": dict(parent_manifest),
        "heldout": heldout_record,
        "heldout_seal": heldout_seal_record,
        "task_ids_sha256": stable_sha256(row_ids),
        "heldout_task_ids_sha256": heldout_task_ids_sha256,
        "heldout_bytes_opened_during_validation": bool(verify_heldout),
        "excluded_task_ids": sorted(EXECUTION_INELIGIBLE_TASK_IDS),
        "rows": EXPECTED_EXECUTABLE_ROWS,
        "heldout_rows": EXPECTED_HELDOUT_ROWS,
        "representation_schema": REPRESENTATION_SCHEMA,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--parent-build-report", required=True)
    parser.add_argument(
        "--expected-parent-build-report-sha256", required=True
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_executable_view(
        parent_build_report=args.parent_build_report,
        expected_parent_build_report_sha256=(
            args.expected_parent_build_report_sha256
        ),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
