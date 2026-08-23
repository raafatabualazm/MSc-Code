#!/usr/bin/env python3
"""Derive an executable-reward view from a sealed imitation fit universe.

Qwen sequence imitation is allowed to use every sealed training target. Local
execution rewards are not: every task named by the parent fit seal's audited
execution-ineligible set is removed before RS candidate replay or VeRPO. This
module preserves parent row order and representation bytes and binds the
derived view to both the complete fit universe and the untouched historical
175-row measure split.

The legacy build-v2 schema remains supported exactly (1,580 -> 1,578). The
expanded production schema is required to attest the 2,776-task append-only
fit union and its selection seal; counts and task-ID commitments are read from
those sealed artifacts rather than copied from legacy constants.
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
EXPANDED_PARENT_BUILD_SCHEMA = "binary-multifunction-compact-expansion-v1"
EXPANSION_SELECTION_SCHEMA = (
    "multifunction-phase0-fit-expansion-selection-v1"
)
EXPANDED_PARENT_SEAL_SCHEMA = "multifunction-phase0-fit-expansion-seal-v1"
REPRESENTATION_SCHEMA = "binary-multifunction-v1-semantic-adapter-v1"
JOIN_SEAL_SCHEMA = "compact-public-private-join-seal-v1"
F2_MANIFEST_SCHEMA = "verified-api-readable-compact-v2"
F2_REPRESENTATION_SCHEMA = "lossless-semantic-f2"
PARENT_TRAIN_SCOPE = "sequence_imitation_all_train"
EXECUTABLE_SCOPE = "executable_reward_only"

EXPECTED_PARENT_ROWS = 1580
EXPECTED_EXECUTABLE_ROWS = 1578
EXPECTED_EXPANDED_PARENT_ROWS = 2776
EXPECTED_SUPPLEMENTAL_ROWS = 1196
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
    expected_rows: int | None = None,
) -> list[str]:
    if expected_rows is not None and len(rows) != expected_rows:
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


def _plain_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutableViewError(f"{label} is not a non-negative integer")
    return value


def _parent_dimensions(report: Mapping[str, Any]) -> dict[str, int]:
    """Return schema-bound parent counts without consulting row contents."""

    schema = report.get("schema")
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise ExecutableViewError("parent build has no sealed counts")
    if schema == PARENT_BUILD_SCHEMA:
        dimensions = {
            "parent_fit_rows": _plain_count(
                counts.get("train_rows"), "legacy parent train_rows"
            ),
            "heldout_rows": _plain_count(
                counts.get("dev_rows"), "legacy parent dev_rows"
            ),
            "supplemental_rows": 0,
        }
        if (
            dimensions["parent_fit_rows"] != EXPECTED_PARENT_ROWS
            or dimensions["heldout_rows"] != EXPECTED_HELDOUT_ROWS
            or _plain_count(
                counts.get("excluded_rows"), "legacy parent excluded_rows"
            )
            != 0
            or _plain_count(
                counts.get("truncated_rows"), "legacy parent truncated_rows"
            )
            != 0
        ):
            raise ExecutableViewError("legacy parent row accounting failed")
        return dimensions
    if schema == EXPANDED_PARENT_BUILD_SCHEMA:
        dimensions = {
            "parent_fit_rows": _plain_count(
                counts.get("expanded_rows"),
                "expanded parent expanded_rows",
            ),
            "heldout_rows": _plain_count(
                counts.get("heldout_rows"), "expanded parent heldout_rows"
            ),
            "supplemental_rows": _plain_count(
                counts.get("supplemental_rows"),
                "expanded parent supplemental_rows",
            ),
        }
        parent_rows = _plain_count(
            counts.get("parent_rows"), "expanded parent parent_rows"
        )
        if (
            dimensions["parent_fit_rows"] != EXPECTED_EXPANDED_PARENT_ROWS
            or parent_rows != EXPECTED_PARENT_ROWS
            or dimensions["supplemental_rows"] != EXPECTED_SUPPLEMENTAL_ROWS
            or parent_rows + dimensions["supplemental_rows"]
            != dimensions["parent_fit_rows"]
            or dimensions["heldout_rows"] != EXPECTED_HELDOUT_ROWS
        ):
            raise ExecutableViewError(
                "expanded parent is not the sealed 1,580 + 1,196 = 2,776 "
                "fit accounting with historical heldout175"
            )
        for optional_zero in ("excluded_rows", "truncated_rows"):
            if optional_zero in counts and _plain_count(
                counts.get(optional_zero),
                f"expanded parent {optional_zero}",
            ):
                raise ExecutableViewError(
                    f"expanded parent {optional_zero} must be zero"
                )
        return dimensions
    raise ExecutableViewError(f"unsupported parent build schema: {schema!r}")


def _record_from_parent(
    report: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one unambiguous record from explicit schema-key aliases."""

    candidates: list[Any] = []
    for container_name in ("outputs", "artifacts", "inputs"):
        container = report.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            if key in container:
                candidates.append(container[key])
    if not candidates:
        raise ExecutableViewError(
            f"parent build lacks {label}; expected one of {', '.join(keys)}"
        )
    normalized: list[tuple[Path, dict[str, Any]]] = [
        _validated_record(value, label=label) for value in candidates
    ]
    first_path, first_record = normalized[0]
    if any(
        path != first_path or record["sha256"] != first_record["sha256"]
        for path, record in normalized[1:]
    ):
        raise ExecutableViewError(f"parent build has conflicting {label} records")
    return first_path, first_record


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
    schema = report.get("schema")
    if (
        schema not in {PARENT_BUILD_SCHEMA, EXPANDED_PARENT_BUILD_SCHEMA}
        or report.get("passed") is not True
        or (
            schema == PARENT_BUILD_SCHEMA
            and (
                report.get("representation_schema")
                != REPRESENTATION_SCHEMA
                or not isinstance(invariants, Mapping)
            )
        )
    ):
        raise ExecutableViewError("parent multi-function build contract failed")
    _parent_dimensions(report)
    if schema == EXPANDED_PARENT_BUILD_SCHEMA:
        # The expanded report is a new append-only composition contract. Its
        # row/task commitments are checked against the separately hashed
        # selection seal below rather than pretending it has every legacy-v2
        # build invariant.
        expansion_seal_value = report.get("expansion_seal")
        expansion_seal_path, _expansion_record = _validated_record(
            expansion_seal_value,
            label="expanded representation build seal",
        )
        expansion_seal = load_json(
            expansion_seal_path, "expanded representation build seal"
        )
        expansion_counts = expansion_seal.get("counts")
        if (
            expansion_seal.get("schema")
            != EXPANDED_PARENT_SEAL_SCHEMA
            or expansion_seal.get("passed") is not True
            or not isinstance(expansion_counts, Mapping)
            or expansion_counts.get("parent_rows") != EXPECTED_PARENT_ROWS
            or expansion_counts.get("supplemental_rows")
            != EXPECTED_SUPPLEMENTAL_ROWS
            or expansion_counts.get("expanded_rows")
            != EXPECTED_EXPANDED_PARENT_ROWS
            or expansion_counts.get("heldout_rows")
            != EXPECTED_HELDOUT_ROWS
            or (expansion_seal.get("invariants") or {}).get(
                "heldout_not_present_in_fit"
            )
            is not True
        ):
            raise ExecutableViewError(
                "expanded representation build seal contract failed"
            )
        return report
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
    schema = report.get("schema")
    if schema == PARENT_BUILD_SCHEMA:
        outputs = report.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ExecutableViewError("parent build has no sealed output records")
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
        result["contract"] = _validated_record(
            outputs.get("contract"), label="derived representation contract"
        )
        result["codebook"] = _validated_record(
            outputs.get("codebook"), label="derived representation codebook"
        )
        return result
    if schema != EXPANDED_PARENT_BUILD_SCHEMA:
        raise ExecutableViewError(f"unsupported parent schema: {schema!r}")
    result = {
        "train": _record_from_parent(
            report,
            ("expanded_dataset", "train"),
            label="expanded fit dataset",
        ),
        "train_seal": _record_from_parent(
            report,
            ("expanded_seal", "expanded_dataset_seal", "train_seal"),
            label="expanded fit seal",
        ),
        "train_f2": _record_from_parent(
            report,
            ("expanded_f2", "train_f2"),
            label="expanded fit F2",
        ),
        "train_f2_manifest": _record_from_parent(
            report,
            ("expanded_f2_manifest", "train_f2_manifest"),
            label="expanded fit F2 manifest",
        ),
        "dev": _record_from_parent(
            report,
            ("heldout_dataset", "dev"),
            label="historical heldout175 dataset",
        ),
        "dev_seal": _record_from_parent(
            report,
            ("heldout_seal", "dev_seal"),
            label="historical heldout175 seal",
        ),
        "contract": _record_from_parent(
            report,
            ("frozen_contract", "contract"),
            label="frozen representation contract",
        ),
        "codebook": _record_from_parent(
            report,
            ("frozen_codebook", "codebook"),
            label="frozen representation codebook",
        ),
        "selection_seal": _record_from_parent(
            report,
            ("selection_seal",),
            label="expanded fit selection seal",
        ),
    }
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
    parent_schema = str(parent["schema"])
    dimensions = _parent_dimensions(parent)
    parent_rows = dimensions["parent_fit_rows"]
    heldout_rows_expected = dimensions["heldout_rows"]
    artifacts = _resolve_parent_artifacts(parent)
    train_path = artifacts["train"][0]
    train_seal_path = artifacts["train_seal"][0]
    f2_path = artifacts["train_f2"][0]
    f2_manifest_path = artifacts["train_f2_manifest"][0]
    dev_path = artifacts["dev"][0]
    dev_seal_path = artifacts["dev_seal"][0]

    train = load_jsonl(train_path, "parent train")
    train_ids = _task_ids(
        train, label="parent train", expected_rows=parent_rows
    )
    f2 = load_jsonl(f2_path, "parent train F2")
    f2_ids = _task_ids(
        f2, label="parent train F2", expected_rows=parent_rows
    )
    if f2_ids != train_ids:
        raise ExecutableViewError("parent compact/F2 train order differs")
    dev = load_jsonl(dev_path, "parent heldout")
    dev_ids = _task_ids(
        dev, label="parent heldout", expected_rows=heldout_rows_expected
    )
    if set(train_ids).intersection(dev_ids):
        raise ExecutableViewError("parent train and heldout task sets overlap")

    train_seal = load_json(train_seal_path, "parent train seal")
    dev_seal = load_json(dev_seal_path, "parent heldout seal")
    raw_ineligible = train_seal.get("execution_ineligible_task_ids")
    if (
        not isinstance(raw_ineligible, list)
        or any(not isinstance(task_id, str) or not task_id for task_id in raw_ineligible)
        or len(set(raw_ineligible)) != len(raw_ineligible)
    ):
        raise ExecutableViewError(
            "parent train seal has no exact execution-ineligible task-ID list"
        )
    execution_ineligible_task_ids = frozenset(raw_ineligible)
    executable_rows = _plain_count(
        train_seal.get("executable_reward_eligible_rows"),
        "parent executable_reward_eligible_rows",
    )
    allowed_train_seal_schemas = {JOIN_SEAL_SCHEMA}
    if parent_schema == EXPANDED_PARENT_BUILD_SCHEMA:
        allowed_train_seal_schemas.add(EXPANDED_PARENT_SEAL_SCHEMA)
    if (
        train_seal.get("schema") not in allowed_train_seal_schemas
        or train_seal.get("selected_role") != "fit"
        or train_seal.get("training_allowed") is not True
        or (
            parent_schema == PARENT_BUILD_SCHEMA
            and train_seal.get("training_objective_scope")
            != PARENT_TRAIN_SCOPE
        )
        or int(train_seal.get("rows", -1)) != parent_rows
        or train_seal.get("output_sha256") != artifacts["train"][1]["sha256"]
        or executable_rows
        != parent_rows - len(execution_ineligible_task_ids)
        or train_seal.get("contract_sha256")
        != artifacts["contract"][1]["sha256"]
    ):
        raise ExecutableViewError("parent train seal eligibility contract failed")
    if (
        parent_schema == PARENT_BUILD_SCHEMA
        and execution_ineligible_task_ids != EXECUTION_INELIGIBLE_TASK_IDS
    ):
        raise ExecutableViewError(
            "legacy parent execution exclusions differ from the audited pair"
        )
    if (
        dev_seal.get("schema") != JOIN_SEAL_SCHEMA
        or dev_seal.get("selected_role") != "measure"
        or dev_seal.get("training_allowed") is not False
        or dev_seal.get("heldout_measure_only") is not True
        or int(dev_seal.get("rows", -1)) != heldout_rows_expected
        or dev_seal.get("output_sha256") != artifacts["dev"][1]["sha256"]
        or dev_seal.get("contract_sha256")
        != artifacts["contract"][1]["sha256"]
    ):
        raise ExecutableViewError("parent heldout seal contract failed")
    if not execution_ineligible_task_ids.issubset(train_ids):
        raise ExecutableViewError(
            "sealed execution-ineligible tasks are absent from parent train"
        )

    selection_seal_record: dict[str, Any] | None = None
    if parent_schema == EXPANDED_PARENT_BUILD_SCHEMA:
        selection_seal_path, selection_seal_record = artifacts["selection_seal"]
        selection = load_json(
            selection_seal_path, "expanded fit selection seal"
        )
        selection_counts = selection.get("counts")
        selection_artifacts = selection.get("artifacts")
        selection_digests = selection.get("digests")
        if (
            selection.get("schema") != EXPANSION_SELECTION_SCHEMA
            or not isinstance(selection_counts, Mapping)
            or not isinstance(selection_artifacts, Mapping)
            or not isinstance(selection_digests, Mapping)
            or _plain_count(
                selection_counts.get("expanded_fit_rows"),
                "selection expanded_fit_rows",
            )
            != parent_rows
            or _plain_count(
                selection_counts.get("heldout_rows"),
                "selection heldout_rows",
            )
            != heldout_rows_expected
            or _plain_count(
                selection_counts.get("parent_fit_rows"),
                "selection parent_fit_rows",
            )
            != EXPECTED_PARENT_ROWS
            or _plain_count(
                selection_counts.get("supplemental_rows"),
                "selection supplemental_rows",
            )
            != EXPECTED_SUPPLEMENTAL_ROWS
            or require_sha256(
                selection_digests.get(
                    "expanded_fit_ordered_task_ids_sha256"
                ),
                "selection expanded fit ordered task IDs SHA-256",
            )
            != stable_sha256(train_ids)
            or require_sha256(
                selection_digests.get(
                    "heldout_ordered_task_ids_sha256"
                ),
                "selection heldout ordered task IDs SHA-256",
            )
            != stable_sha256(dev_ids)
        ):
            raise ExecutableViewError(
                "expanded selection seal accounting/task commitments failed"
            )
        fit_manifest_path, _fit_manifest_record = _validated_record(
            selection_artifacts.get("fit_task_manifest"),
            label="expanded fit task manifest",
        )
        fit_manifest_ids = _task_ids(
            load_jsonl(fit_manifest_path, "expanded fit task manifest"),
            label="expanded fit task manifest",
            expected_rows=parent_rows,
        )
        if fit_manifest_ids != train_ids:
            raise ExecutableViewError(
                "expanded representation order differs from selection seal"
            )
        heldout_dataset_record = selection_artifacts.get(
            "heldout_dataset", selection_artifacts.get("heldout")
        )
        heldout_seal_record = selection_artifacts.get("heldout_seal")
        if (
            not isinstance(heldout_dataset_record, Mapping)
            or heldout_dataset_record.get("sha256")
            != artifacts["dev"][1]["sha256"]
            or not isinstance(heldout_seal_record, Mapping)
            or heldout_seal_record.get("sha256")
            != artifacts["dev_seal"][1]["sha256"]
        ):
            raise ExecutableViewError(
                "expanded selection seal does not bind historical heldout175"
            )
        expansion_seal_path, _expansion_record = _validated_record(
            parent.get("expansion_seal"),
            label="expanded representation build seal",
        )
        expansion_seal = load_json(
            expansion_seal_path, "expanded representation build seal"
        )
        expansion_digests = expansion_seal.get("digests")
        expansion_artifacts = expansion_seal.get("artifacts")
        if (
            not isinstance(expansion_digests, Mapping)
            or not isinstance(expansion_artifacts, Mapping)
            or expansion_digests.get("expanded_ordered_task_ids_sha256")
            != stable_sha256(train_ids)
            or expansion_digests.get("heldout_ordered_task_ids_sha256")
            != stable_sha256(dev_ids)
            or expansion_seal.get("execution_ineligible_task_ids")
            != sorted(execution_ineligible_task_ids)
            or (expansion_artifacts.get("expanded_dataset") or {}).get(
                "sha256"
            )
            != artifacts["train"][1]["sha256"]
            or (expansion_artifacts.get("expanded_seal") or {}).get(
                "sha256"
            )
            != artifacts["train_seal"][1]["sha256"]
        ):
            raise ExecutableViewError(
                "expanded representation seal task/artifact binding failed"
            )

    parent_f2_manifest = load_json(
        f2_manifest_path, "parent train F2 manifest"
    )
    if (
        parent_f2_manifest.get("schema") != F2_MANIFEST_SCHEMA
        or int(parent_f2_manifest.get("rows", -1)) != parent_rows
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
        if str(row["task_id"]) not in execution_ineligible_task_ids
    ]
    executable_f2 = [
        row
        for row in f2
        if str(row["task_id"]) not in execution_ineligible_task_ids
    ]
    executable_ids = _task_ids(
        executable_train,
        label="derived executable train",
        expected_rows=executable_rows,
    )
    if _task_ids(
        executable_f2,
        label="derived executable train F2",
        expected_rows=executable_rows,
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
        "parent_build_schema": parent_schema,
        "parent_build_report": parent_record,
        "parent_dataset": artifacts["train"][1],
        "parent_dataset_seal": artifacts["train_seal"][1],
        "parent_f2": artifacts["train_f2"][1],
        "parent_f2_manifest": artifacts["train_f2_manifest"][1],
        "parent_rows": parent_rows,
        "parent_task_ids_sha256": stable_sha256(train_ids),
        "output_rows": executable_rows,
        "output_task_ids_sha256": stable_sha256(executable_ids),
        "excluded_task_ids": sorted(execution_ineligible_task_ids),
        "excluded_task_ids_sha256": stable_sha256(
            sorted(execution_ineligible_task_ids)
        ),
        "selection": "stable_parent_order_minus_exact_audited_ids",
        "row_content_transform": "identity",
    }
    if selection_seal_record is not None:
        derivation["expanded_selection_seal"] = selection_seal_record
    f2_manifest = dict(parent_f2_manifest)
    f2_manifest.update(
        {
            "created_at": utc_now(),
            "rows": executable_rows,
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

    legacy_copied_seal_fields = (
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
    expanded_copied_seal_fields = (
        "contract_sha256",
        "representation_schema",
        "frontier_f2_schema",
        "adapter_contract_sha256",
        "adapter_script_sha256",
        "expansion_builder_sha256",
        "selection_seal_sha256",
        "source_symbol_attestation_used",
        "source_symbol_attestation_is_keyed",
        "source_symbol_attestation_key_id_sha256",
        "raw_source_names_serialized",
        "heldout_commitment",
        "sanitation_partitions",
        "append_only_prefix",
    )
    copied_seal_fields = (
        legacy_copied_seal_fields
        if parent_schema == PARENT_BUILD_SCHEMA
        else expanded_copied_seal_fields
    )
    seal = {
        "schema": JOIN_SEAL_SCHEMA,
        "selected_role": "fit",
        "training_allowed": True,
        "heldout_measure_only": False,
        "rows": executable_rows,
        "task_set_sha256": stable_sha256(executable_ids),
        "output_sha256": sha256_file(paths["dataset"]),
        "output": file_record(paths["dataset"]),
        "f2_output": file_record(paths["f2"]),
        "f2_manifest": file_record(paths["f2_manifest"]),
        "training_objective_scope": EXECUTABLE_SCOPE,
        "executable_reward_eligible_rows": executable_rows,
        "execution_ineligible_task_ids": [],
        "excluded_from_parent_task_ids": sorted(
            execution_ineligible_task_ids
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
            "schema": parent_schema,
            "build_report": parent_record,
            "train": artifacts["train"][1],
            "train_seal": artifacts["train_seal"][1],
            "train_f2": artifacts["train_f2"][1],
            "train_f2_manifest": artifacts["train_f2_manifest"][1],
        },
        "heldout_measure_only": {
            "dataset": artifacts["dev"][1],
            "seal": artifacts["dev_seal"][1],
            "rows": heldout_rows_expected,
            "task_set_sha256": stable_sha256(dev_ids),
            "untouched": True,
        },
        "contract": artifacts["contract"][1],
        "counts": {
            "parent_train_rows": parent_rows,
            "supplemental_train_rows": dimensions["supplemental_rows"],
            "execution_ineligible_rows": len(
                execution_ineligible_task_ids
            ),
            "executable_train_rows": executable_rows,
            "heldout_rows": heldout_rows_expected,
        },
        "excluded_task_ids": sorted(execution_ineligible_task_ids),
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
            "only_parent_seal_audited_execution_rows_excluded": True,
            "legacy_exact_audited_fs_ffi_rows_excluded": (
                parent_schema == PARENT_BUILD_SCHEMA
            ),
            "all_remaining_compact_rows_byte_identical_to_parent": True,
            "all_remaining_f2_rows_byte_identical_to_parent": True,
            "compact_and_f2_task_order_identical": True,
            "executable_train_has_unique_tasks": True,
            "executable_train_has_1578_unique_tasks": (
                executable_rows == EXPECTED_EXECUTABLE_ROWS
            ),
            "heldout_has_175_unique_tasks": True,
            "train_heldout_disjoint": True,
            "heldout_not_rewritten": True,
            "parent_full_imitation_view_retained": True,
        },
    }
    _atomic_json(paths["report"], result)
    print(
        "MULTIFUNCTION_EXECUTABLE_VIEW "
        f"parent={parent_rows} train={executable_rows} "
        f"heldout={heldout_rows_expected} "
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
    expected_parent_rows: int | None = None,
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
    counts = report.get("counts")
    excluded_task_ids = report.get("excluded_task_ids")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(excluded_task_ids, list)
        or any(
            not isinstance(task_id, str) or not task_id
            for task_id in excluded_task_ids
        )
        or len(set(excluded_task_ids)) != len(excluded_task_ids)
        or excluded_task_ids != sorted(excluded_task_ids)
    ):
        raise ExecutableViewError(
            "executable-view report accounting/exclusion list is malformed"
        )
    parent_rows = _plain_count(
        counts.get("parent_train_rows"),
        "executable-view parent_train_rows",
    )
    executable_rows = _plain_count(
        counts.get("executable_train_rows"),
        "executable-view executable_train_rows",
    )
    heldout_rows_expected = _plain_count(
        counts.get("heldout_rows"),
        "executable-view heldout_rows",
    )
    ineligible_rows = _plain_count(
        counts.get("execution_ineligible_rows"),
        "executable-view execution_ineligible_rows",
    )
    if (
        report.get("schema") != SCHEMA
        or report.get("passed") is not True
        or report.get("representation_schema") != REPRESENTATION_SCHEMA
        or report.get("training_objective_scope") != EXECUTABLE_SCOPE
        or executable_rows <= 0
        or heldout_rows_expected != EXPECTED_HELDOUT_ROWS
        or ineligible_rows != len(excluded_task_ids)
        or parent_rows - executable_rows != ineligible_rows
        or (
            expected_parent_rows is not None
            and parent_rows != expected_parent_rows
        )
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
        expected_rows=executable_rows,
    )
    f2_rows = load_jsonl(paths["f2"], "executable F2")
    f2_ids = _task_ids(
        f2_rows,
        label="executable F2",
        expected_rows=executable_rows,
    )
    if f2_ids != row_ids:
        raise ExecutableViewError("executable compact/F2 task order differs")
    if set(row_ids).intersection(excluded_task_ids):
        raise ExecutableViewError(
            "an audited execution-ineligible task entered the executable view"
        )
    for index, row in enumerate(rows):
        if row.get("binary_multifunction_schema") != REPRESENTATION_SCHEMA:
            raise ExecutableViewError(
                f"executable row {index} is not the multi-function representation"
            )
    digests = report.get("digests")
    parent_task_ids_sha256 = (
        require_sha256(
            (digests or {}).get("parent_train_task_ids_sha256"),
            "parent train task IDs SHA-256",
        )
        if isinstance(digests, Mapping)
        else ""
    )
    if (
        not isinstance(digests, Mapping)
        or digests.get("executable_train_task_ids_sha256")
        != stable_sha256(row_ids)
    ):
        raise ExecutableViewError(
            "executable-view report task-ID commitment failed"
        )

    seal_value = load_json(paths["seal"], "executable seal")
    if (
        seal_value.get("schema") != JOIN_SEAL_SCHEMA
        or seal_value.get("selected_role") != "fit"
        or seal_value.get("training_allowed") is not True
        or seal_value.get("heldout_measure_only") is not False
        or seal_value.get("training_objective_scope") != EXECUTABLE_SCOPE
        or int(seal_value.get("rows", -1)) != executable_rows
        or int(seal_value.get("executable_reward_eligible_rows", -1))
        != executable_rows
        or seal_value.get("execution_ineligible_task_ids") != []
        or seal_value.get("excluded_from_parent_task_ids")
        != excluded_task_ids
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
        or int(f2_value.get("rows", -1)) != executable_rows
        or (f2_value.get("dataset") or {}).get("sha256")
        != observed_records["dataset"]["sha256"]
        or (f2_value.get("output") or {}).get("sha256")
        != observed_records["f2"]["sha256"]
        or not isinstance(derivation, Mapping)
        or derivation.get("schema") != DERIVATION_SCHEMA
        or derivation.get("excluded_task_ids")
        != excluded_task_ids
        or int(derivation.get("parent_rows", -1)) != parent_rows
        or int(derivation.get("output_rows", -1))
        != executable_rows
        or (
            derivation.get("parent_task_ids_sha256") is not None
            and derivation.get("parent_task_ids_sha256")
            != parent_task_ids_sha256
        )
        or (
            derivation.get("output_task_ids_sha256") is not None
            and derivation.get("output_task_ids_sha256")
            != stable_sha256(row_ids)
        )
    ):
        raise ExecutableViewError("executable F2 derivation contract failed")
    parent_prompt = derivation.get("parent_f2")
    parent_manifest = derivation.get("parent_f2_manifest")
    _validated_record(parent_prompt, label="parent full F2")
    _validated_record(parent_manifest, label="parent full F2 manifest")
    selection_record = derivation.get("expanded_selection_seal")
    if selection_record is not None:
        selection_path, _selection_file_record = _validated_record(
            selection_record,
            label="expanded fit selection seal",
        )
        selection = load_json(selection_path, "expanded fit selection seal")
        selection_counts = selection.get("counts")
        selection_digests = selection.get("digests")
        if (
            selection.get("schema") != EXPANSION_SELECTION_SCHEMA
            or not isinstance(selection_counts, Mapping)
            or not isinstance(selection_digests, Mapping)
            or int(selection_counts.get("expanded_fit_rows", -1))
            != parent_rows
            or selection_digests.get(
                "expanded_fit_ordered_task_ids_sha256"
            )
            != parent_task_ids_sha256
        ):
            raise ExecutableViewError(
                "expanded selection seal no longer binds executable parent"
            )

    heldout = report.get("heldout_measure_only")
    if (
        not isinstance(heldout, Mapping)
        or heldout.get("untouched") is not True
        or int(heldout.get("rows", -1)) != heldout_rows_expected
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
            expected_rows=heldout_rows_expected,
        )
        if set(row_ids).intersection(heldout_ids):
            raise ExecutableViewError("executable train overlaps heldout-175")
        heldout_seal = load_json(heldout_seal_path, "heldout seal")
        if (
            heldout_seal.get("selected_role") != "measure"
            or heldout_seal.get("training_allowed") is not False
            or heldout_seal.get("heldout_measure_only") is not True
            or int(heldout_seal.get("rows", -1)) != heldout_rows_expected
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
        "excluded_task_ids": excluded_task_ids,
        "rows": executable_rows,
        "parent_rows": parent_rows,
        "parent_task_ids_sha256": parent_task_ids_sha256,
        "heldout_rows": heldout_rows_expected,
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
