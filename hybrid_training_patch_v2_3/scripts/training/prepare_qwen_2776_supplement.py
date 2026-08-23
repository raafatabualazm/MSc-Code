#!/usr/bin/env python3
"""Derive the exact 2,776-task fit set and its 1,196-task Qwen supplement.

The live 1,580-task Qwen journal is an immutable parent.  This tool accepts a
new complete multi-function compact/F2 view (either all 2,951 Phase-0 train
tasks or the already filtered 2,776 fit tasks), removes the sealed 175-task
experiment holdout when necessary, and proves that every legacy task has the
same student compact IDs and byte-identical API prompt.

Only the set difference is written to the supplemental collection artifacts.
No teacher journal is read or modified here.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.direct_compact_causal import (  # noqa: E402
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.training.build_qwen_sequence_kd import (  # noqa: E402
    compact_ids_sha256,
    exact_output_seal,
    load_student_tokenizer,
    require_file_hash,
    strict_json,
    target_text,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    ArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    load_f2_prompt_contract,
    load_verified_prompt_rows,
    read_jsonl,
    stable_sha256,
)


MANIFEST_SCHEMA = "qwen-2776-supplement-derivation-v1"
SUBSET_PROMPT_SCHEMA = "verified-api-readable-compact-v2"
EXPANSION_SEAL_SCHEMA = "multifunction-phase0-fit-expansion-seal-v1"
EXPANSION_REPORT_SCHEMA = "binary-multifunction-compact-expansion-v1"
SELECTION_SCHEMA = "multifunction-phase0-fit-expansion-selection-v1"
EXPECTED_CANDIDATE_COUNTS = frozenset((2776, 2951))
EXPECTED_FIT_TASKS = 2776
EXPECTED_LEGACY_TASKS = 1580
EXPECTED_SUPPLEMENT_TASKS = 1196
EXPECTED_HOLDOUT_TASKS = 175
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate-compact-jsonl", required=True, type=Path)
    parser.add_argument("--expected-candidate-compact-sha256", required=True)
    parser.add_argument("--candidate-compact-seal", required=True, type=Path)
    parser.add_argument("--expected-candidate-compact-seal-sha256", required=True)
    parser.add_argument("--candidate-prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-candidate-prompt-sha256", required=True)
    parser.add_argument("--candidate-prompt-manifest", required=True, type=Path)
    parser.add_argument("--expected-candidate-prompt-manifest-sha256", required=True)
    parser.add_argument("--legacy-compact-jsonl", required=True, type=Path)
    parser.add_argument("--expected-legacy-compact-sha256", required=True)
    parser.add_argument("--legacy-compact-seal", required=True, type=Path)
    parser.add_argument("--expected-legacy-compact-seal-sha256", required=True)
    parser.add_argument("--legacy-prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-legacy-prompt-sha256", required=True)
    parser.add_argument("--legacy-prompt-manifest", required=True, type=Path)
    parser.add_argument("--expected-legacy-prompt-manifest-sha256", required=True)
    parser.add_argument("--heldout-jsonl", required=True, type=Path)
    parser.add_argument("--expected-heldout-sha256", required=True)
    parser.add_argument("--heldout-seal", required=True, type=Path)
    parser.add_argument("--expected-heldout-seal-sha256", required=True)
    parser.add_argument("--expansion-build-seal", required=True, type=Path)
    parser.add_argument("--expected-expansion-build-seal-sha256", required=True)
    parser.add_argument("--expansion-build-report", required=True, type=Path)
    parser.add_argument("--expected-expansion-build-report-sha256", required=True)
    parser.add_argument(
        "--candidate-contract",
        type=Path,
        help=(
            "Contract bound by the candidate input seal. Defaults to --contract. "
            "A base-capacity contract is accepted only when every field except "
            "max_target_tokens/max_total_tokens is byte-semantically identical."
        ),
    )
    parser.add_argument("--expected-candidate-contract-sha256", default="")
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    parser.add_argument("--fit-compact-output", required=True, type=Path)
    parser.add_argument("--fit-compact-seal-output", required=True, type=Path)
    parser.add_argument("--fit-prompt-output", required=True, type=Path)
    parser.add_argument("--fit-prompt-manifest-output", required=True, type=Path)
    parser.add_argument("--supplement-compact-output", required=True, type=Path)
    parser.add_argument("--supplement-compact-seal-output", required=True, type=Path)
    parser.add_argument("--supplement-prompt-output", required=True, type=Path)
    parser.add_argument(
        "--supplement-prompt-manifest-output", required=True, type=Path
    )
    parser.add_argument("--derivation-manifest-output", required=True, type=Path)
    return parser.parse_args()


def _task_id(row: Mapping[str, Any], label: str) -> str:
    value = str(row.get("task_id") or row.get("id") or "")
    if not value:
        raise ArtifactError(f"{label} has no task_id")
    return value


def _indexed(
    rows: Sequence[dict[str, Any]], label: str
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        task_id = _task_id(row, f"{label} row {index}")
        if task_id in result:
            raise ArtifactError(f"{label} has duplicate task_id {task_id}")
        ordered.append(task_id)
        result[task_id] = row
    return ordered, result


def _prompt_manifest(
    *,
    parent_manifest: Mapping[str, Any],
    parent_manifest_record: Mapping[str, Any],
    prompt_path: Path,
    compact_path: Path,
    rows: int,
    scope: str,
    ordered_task_ids: Sequence[str],
    prompt_rows: Sequence[Mapping[str, Any]],
    heldout_record: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(parent_manifest))
    if len(prompt_rows) != len(ordered_task_ids) or not prompt_rows:
        raise ArtifactError("subset prompt rows/task IDs differ")
    max_row = max(
        prompt_rows,
        key=lambda row: int(
            (row.get("prompt_preflight") or {}).get(
                "estimated_prompt_tokens", -1
            )
        ),
    )
    maximum_tokens = int(
        (max_row.get("prompt_preflight") or {}).get(
            "estimated_prompt_tokens", -1
        )
    )
    max_prompt_tokens = int(
        (manifest.get("f2_prompt_contract") or {}).get(
            "max_prompt_tokens", -1
        )
    )
    if maximum_tokens < 0 or maximum_tokens > max_prompt_tokens:
        raise ArtifactError("subset prompt preflight exceeds its F2 contract")
    manifest["schema"] = SUBSET_PROMPT_SCHEMA
    manifest["rows"] = int(rows)
    manifest["output"] = file_record(prompt_path)
    manifest["dataset"] = file_record(compact_path)
    manifest["task_set_sha256"] = stable_sha256(list(ordered_task_ids))
    manifest["binary_constant_extraction_errors"] = {
        "count": 0,
        "task_ids": [],
    }
    source_attestation = manifest.get("source_symbol_attestation")
    if not isinstance(source_attestation, Mapping):
        raise ArtifactError("parent prompt manifest lacks source attestation")
    binding_digests = [
        str(row.get("source_symbol_attestation_binding_sha256") or "")
        for row in prompt_rows
    ]
    if any(not SHA256_RE.fullmatch(value) for value in binding_digests):
        raise ArtifactError("subset prompt row lacks keyed attestation binding")
    manifest["source_symbol_attestation"] = {
        **dict(source_attestation),
        "binding_sha256_sequence": stable_sha256(binding_digests),
    }
    manifest["f2_prompt_contract"] = {
        **dict(manifest["f2_prompt_contract"]),
        "maximum_estimated_prompt_tokens": maximum_tokens,
        "maximum_task_id": str(max_row.get("task_id") or ""),
        "all_rows_within_limit": True,
    }
    manifest["subset_derivation"] = {
        "schema": MANIFEST_SCHEMA,
        "scope": scope,
        "parent_prompt_manifest": dict(parent_manifest_record),
        "ordered_task_ids_sha256": stable_sha256(list(ordered_task_ids)),
        "task_set_sha256": stable_sha256(sorted(ordered_task_ids)),
        "task_count": len(ordered_task_ids),
        "heldout_artifact": dict(heldout_record),
        "heldout_intersection_count": 0,
        "selection_depends_only_on_task_id_membership": True,
        "prompt_rows_reencoded": False,
        "prompt_text_modified": False,
    }
    return manifest


def _shared_compatibility(
    *,
    legacy_ids: Sequence[str],
    legacy_compact: Mapping[str, dict[str, Any]],
    fit_compact: Mapping[str, dict[str, Any]],
    legacy_prompts: Mapping[str, dict[str, Any]],
    fit_prompts: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    compact_bindings: list[dict[str, str]] = []
    prompt_bindings: list[dict[str, str]] = []
    target_differences = 0
    for task_id in legacy_ids:
        old_compact = legacy_compact[task_id]
        new_compact = fit_compact[task_id]
        old_compact_sha = compact_ids_sha256(old_compact, f"legacy-{task_id}")
        new_compact_sha = compact_ids_sha256(new_compact, f"fit-{task_id}")
        if old_compact_sha != new_compact_sha:
            raise ArtifactError(
                f"{task_id}: expanded compact IDs differ from the live parent"
            )
        old_prompt = legacy_prompts[task_id]
        new_prompt = fit_prompts[task_id]
        for field in (
            "text",
            "text_sha256",
            "compact_ids_sha256",
            "representation_schema",
            "system_prompt_sha256",
        ):
            if old_prompt.get(field) != new_prompt.get(field):
                raise ArtifactError(
                    f"{task_id}: expanded F2 prompt differs in {field}"
                )
        if target_text(old_compact, f"legacy-{task_id}") != target_text(
            new_compact, f"fit-{task_id}"
        ):
            target_differences += 1
        compact_bindings.append(
            {"task_id": task_id, "compact_ids_sha256": old_compact_sha}
        )
        prompt_bindings.append(
            {
                "task_id": task_id,
                "text_sha256": str(old_prompt["text_sha256"]),
                "system_prompt_sha256": str(old_prompt["system_prompt_sha256"]),
            }
        )
    return {
        "shared_tasks": len(legacy_ids),
        "all_compact_ids_byte_identical": True,
        "all_api_prompt_text_byte_identical": True,
        "all_system_prompt_bindings_identical": True,
        "gold_target_difference_count": target_differences,
        "compact_binding_sha256": stable_sha256(compact_bindings),
        "prompt_binding_sha256": stable_sha256(prompt_bindings),
    }


def derive_partition(
    *,
    candidate_order: Sequence[str],
    legacy_order: Sequence[str],
    heldout_order: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return fit/supplement order after proving the exact set equations."""

    if len(candidate_order) not in EXPECTED_CANDIDATE_COUNTS:
        raise ArtifactError(
            "candidate task count must be exactly 2,951 or 2,776"
        )
    if (
        len(candidate_order) != len(set(candidate_order))
        or len(legacy_order) != len(set(legacy_order))
        or len(heldout_order) != len(set(heldout_order))
    ):
        raise ArtifactError("candidate/legacy/heldout task IDs are not unique")
    if len(legacy_order) != EXPECTED_LEGACY_TASKS:
        raise ArtifactError("legacy task count must be exactly 1,580")
    if len(heldout_order) != EXPECTED_HOLDOUT_TASKS:
        raise ArtifactError("heldout task count must be exactly 175")
    candidate_ids = set(candidate_order)
    heldout_ids = set(heldout_order)
    if len(candidate_order) == 2951:
        if not heldout_ids.issubset(candidate_ids):
            raise ArtifactError("full candidate task set lacks heldout IDs")
        fit_order = [
            task_id for task_id in candidate_order if task_id not in heldout_ids
        ]
    else:
        if candidate_ids.intersection(heldout_ids):
            raise ArtifactError("filtered candidate task set contains heldout IDs")
        fit_order = list(candidate_order)
    fit_ids = set(fit_order)
    legacy_ids = set(legacy_order)
    if len(fit_order) != EXPECTED_FIT_TASKS:
        raise ArtifactError("fit task count must be exactly 2,776")
    if not legacy_ids.issubset(fit_ids):
        raise ArtifactError("fit task set lacks legacy parent IDs")
    supplement_order = [
        task_id for task_id in fit_order if task_id not in legacy_ids
    ]
    if (
        len(supplement_order) != EXPECTED_SUPPLEMENT_TASKS
        or legacy_ids.intersection(supplement_order)
    ):
        raise ArtifactError("legacy/supplement partition is not 1,580 + 1,196")
    return fit_order, supplement_order


def _artifact_sha(
    artifacts: Mapping[str, Any], key: str, label: str
) -> str:
    value = artifacts.get(key)
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} lacks artifact record {key}")
    digest = str(value.get("sha256") or "")
    if SHA256_RE.fullmatch(digest) is None:
        raise ArtifactError(f"{label} artifact {key} has malformed SHA-256")
    return digest


def _bound_artifact_path(
    artifacts: Mapping[str, Any], key: str, label: str
) -> Path:
    value = artifacts.get(key)
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} lacks artifact record {key}")
    path_value = str(value.get("path") or "")
    if not path_value:
        raise ArtifactError(f"{label} artifact {key} has no path")
    path = Path(path_value).expanduser().resolve()
    expected_sha = _artifact_sha(artifacts, key, label)
    observed = file_record(path)
    if observed["sha256"] != expected_sha:
        raise ArtifactError(
            f"{label} artifact {key} hash mismatch: "
            f"{observed['sha256']} != {expected_sha}"
        )
    return path


def _load_bound_artifact(
    artifacts: Mapping[str, Any], key: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _bound_artifact_path(artifacts, key, label)
    return path, strict_json(path)


def validate_expansion_evidence(
    *,
    expansion_seal_path: Path,
    expansion_report_path: Path,
    candidate_compact_record: Mapping[str, Any],
    candidate_seal_record: Mapping[str, Any],
    candidate_prompt_record: Mapping[str, Any],
    candidate_prompt_manifest_record: Mapping[str, Any],
    legacy_compact_record: Mapping[str, Any],
    heldout_record: Mapping[str, Any],
    heldout_seal_record: Mapping[str, Any],
    candidate_contract_record: Mapping[str, Any],
    candidate_order: Sequence[str],
    legacy_order: Sequence[str],
    supplement_order: Sequence[str],
    heldout_order: Sequence[str],
) -> dict[str, Any]:
    """Prove the Qwen candidate is the exact sealed Phase-0 fit expansion."""

    seal = strict_json(expansion_seal_path)
    report = strict_json(expansion_report_path)
    expected_counts = {
        "parent_rows": EXPECTED_LEGACY_TASKS,
        "supplemental_rows": EXPECTED_SUPPLEMENT_TASKS,
        "expanded_rows": EXPECTED_FIT_TASKS,
        "heldout_rows": EXPECTED_HOLDOUT_TASKS,
    }
    seal_counts = seal.get("counts")
    report_counts = report.get("counts")
    seal_artifacts = seal.get("artifacts")
    seal_digests = seal.get("digests")
    invariants = seal.get("invariants")
    heldout_commitment = seal.get("heldout_commitment")
    report_outputs = report.get("outputs")
    if (
        seal.get("schema") != EXPANSION_SEAL_SCHEMA
        or seal.get("passed") is not True
        or report.get("schema") != EXPANSION_REPORT_SCHEMA
        or report.get("passed") is not True
        or not isinstance(seal_counts, Mapping)
        or not isinstance(report_counts, Mapping)
        or any(
            int(seal_counts.get(key, -1)) != value
            or int(report_counts.get(key, -1)) != value
            for key, value in expected_counts.items()
        )
        or not isinstance(seal_artifacts, Mapping)
        or not isinstance(seal_digests, Mapping)
        or not isinstance(invariants, Mapping)
        or not isinstance(heldout_commitment, Mapping)
        or not isinstance(report_outputs, Mapping)
    ):
        raise ArtifactError("expansion build seal/report contract mismatch")

    expected_artifacts = {
        "expanded_dataset": candidate_compact_record,
        "expanded_seal": candidate_seal_record,
        "expanded_f2": candidate_prompt_record,
        "expanded_f2_manifest": candidate_prompt_manifest_record,
        "parent_dataset": legacy_compact_record,
        "heldout_dataset": heldout_record,
        "heldout_seal": heldout_seal_record,
        "frozen_contract": candidate_contract_record,
    }
    for key, record in expected_artifacts.items():
        if _artifact_sha(seal_artifacts, key, "expansion seal") != str(
            record.get("sha256") or ""
        ):
            raise ArtifactError(f"expansion seal artifact {key} differs")
    for key in (
        "expanded_dataset",
        "expanded_seal",
        "expanded_f2",
        "expanded_f2_manifest",
    ):
        if _artifact_sha(report_outputs, key, "expansion report") != str(
            expected_artifacts[key].get("sha256") or ""
        ):
            raise ArtifactError(f"expansion report output {key} differs")
    # The expansion builder binds the immutable base-capacity parent seal,
    # whereas the Qwen trainer consumes its monotonically enlarged target24k
    # counterpart. Verify both independently against the same parent bytes.
    base_parent_seal_path = _bound_artifact_path(
        seal_artifacts, "parent_seal", "expansion seal"
    )
    validate_join_seal(
        Path(str(legacy_compact_record["path"])),
        base_parent_seal_path,
        Path(str(candidate_contract_record["path"])),
        expected_role="fit",
    )

    expected_digests = {
        "parent_ordered_task_ids_sha256": stable_sha256(list(legacy_order)),
        "supplemental_ordered_task_ids_sha256": stable_sha256(
            list(supplement_order)
        ),
        "expanded_ordered_task_ids_sha256": stable_sha256(
            list(candidate_order)
        ),
        "heldout_ordered_task_ids_sha256": stable_sha256(list(heldout_order)),
    }
    if any(
        seal_digests.get(key) != value
        for key, value in expected_digests.items()
    ):
        raise ArtifactError("expansion build ordered membership digest differs")
    required_invariants = (
        "parent_dataset_bytes_exact_prefix",
        "parent_f2_bytes_exact_prefix",
        "parent_rows_not_reencoded",
        "parent_compact_ids_unchanged",
        "parent_prompt_text_unchanged",
        "frozen_contract_unchanged",
        "frozen_codebook_unchanged",
        "no_source_token_id_reassigned",
        "unknown_instructions_use_lossless_raw_fallback",
        "supplemental_zero_exclusion",
        "heldout_membership_unchanged",
        "heldout_not_present_in_fit",
        "all_student_sources_within_9000",
        "all_api_prompts_within_12000",
    )
    if any(invariants.get(key) is not True for key in required_invariants):
        raise ArtifactError("expansion build invariant is not sealed true")
    if (
        heldout_commitment.get("dataset_sha256")
        != heldout_record.get("sha256")
        or heldout_commitment.get("seal_sha256")
        != heldout_seal_record.get("sha256")
        or int(heldout_commitment.get("rows", -1))
        != EXPECTED_HOLDOUT_TASKS
        or heldout_commitment.get("measure_only") is not True
    ):
        raise ArtifactError("expansion heldout commitment differs")

    selection_path, selection = _load_bound_artifact(
        seal_artifacts, "selection_seal", "expansion seal"
    )
    selection_counts = selection.get("counts")
    selection_digests = selection.get("digests")
    selection_artifacts = selection.get("artifacts")
    selection_invariants = selection.get("invariants")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("passed") is not True
        or not isinstance(selection_counts, Mapping)
        or int(selection_counts.get("phase0_train_rows", -1)) != 2951
        or int(selection_counts.get("parent_fit_rows", -1))
        != EXPECTED_LEGACY_TASKS
        or int(selection_counts.get("heldout_rows", -1))
        != EXPECTED_HOLDOUT_TASKS
        or int(selection_counts.get("supplemental_rows", -1))
        != EXPECTED_SUPPLEMENT_TASKS
        or int(selection_counts.get("expanded_fit_rows", -1))
        != EXPECTED_FIT_TASKS
        or not isinstance(selection_digests, Mapping)
        or not isinstance(selection_artifacts, Mapping)
        or not isinstance(selection_invariants, Mapping)
    ):
        raise ArtifactError("expansion selection seal contract mismatch")
    selection_expected_digests = {
        "parent_fit_ordered_task_ids_sha256": stable_sha256(
            list(legacy_order)
        ),
        "supplemental_ordered_task_ids_sha256": stable_sha256(
            list(supplement_order)
        ),
        "expanded_fit_ordered_task_ids_sha256": stable_sha256(
            list(candidate_order)
        ),
        "heldout_ordered_task_ids_sha256": stable_sha256(list(heldout_order)),
    }
    if any(
        selection_digests.get(key) != value
        for key, value in selection_expected_digests.items()
    ):
        raise ArtifactError("expansion selection ordered membership differs")
    required_selection_invariants = (
        "phase0_train_equals_expanded_fit_union_heldout",
        "expanded_fit_and_heldout_disjoint",
        "parent_fit_and_supplemental_disjoint",
        "parent_fit_order_unchanged",
        "heldout_membership_unchanged",
        "supplemental_aot_is_exact_full_manifest_projection",
        "canonical_source_rows_byte_hash_verified",
        "raw_semantic_function_names_not_serialized",
    )
    if any(
        selection_invariants.get(key) is not True
        for key in required_selection_invariants
    ):
        raise ArtifactError("expansion selection invariant is not sealed true")
    if (
        _artifact_sha(selection_artifacts, "parent_fit", "selection seal")
        != legacy_compact_record.get("sha256")
        or _artifact_sha(selection_artifacts, "heldout", "selection seal")
        != heldout_record.get("sha256")
        or _artifact_sha(
            selection_artifacts, "heldout_seal", "selection seal"
        )
        != heldout_seal_record.get("sha256")
        or _artifact_sha(
            selection_artifacts, "frozen_contract", "selection seal"
        )
        != candidate_contract_record.get("sha256")
    ):
        raise ArtifactError("expansion selection input binding differs")

    fit_manifest_path = _bound_artifact_path(
        selection_artifacts, "fit_task_manifest", "selection seal"
    )
    fit_manifest_order, _ = _indexed(
        read_jsonl(fit_manifest_path), "sealed fit task manifest"
    )
    supplemental_manifest_path = _bound_artifact_path(
        selection_artifacts, "supplemental_task_manifest", "selection seal"
    )
    supplemental_manifest_order, _ = _indexed(
        read_jsonl(supplemental_manifest_path),
        "sealed supplemental task manifest",
    )
    if (
        fit_manifest_order != list(candidate_order)
        or supplemental_manifest_order != list(supplement_order)
    ):
        raise ArtifactError(
            "candidate membership differs from sealed Phase-0 manifests"
        )
    if (
        not isinstance(report.get("expansion_seal"), Mapping)
        or report["expansion_seal"].get("sha256")
        != sha256_file(expansion_seal_path)
    ):
        raise ArtifactError("expansion report does not bind expansion seal")
    return {
        "selection_seal": file_record(selection_path),
        "phase0_train_rows": 2951,
        "fit_rows": EXPECTED_FIT_TASKS,
        "supplement_rows": EXPECTED_SUPPLEMENT_TASKS,
        "heldout_rows": EXPECTED_HOLDOUT_TASKS,
        "exact_membership_proved": True,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = args.contract.expanduser().resolve()
    contract_record = require_file_hash(
        contract_path, args.expected_contract_sha256, "compact contract"
    )
    contract = DirectCompactContract.load(contract_path)
    candidate_contract_arg = getattr(args, "candidate_contract", None)
    expected_candidate_contract = str(
        getattr(args, "expected_candidate_contract_sha256", "") or ""
    )
    if bool(candidate_contract_arg) != bool(expected_candidate_contract):
        raise ArtifactError(
            "--candidate-contract and its expected SHA-256 must be supplied together"
        )
    candidate_contract_path = (
        contract_path
        if candidate_contract_arg is None
        else candidate_contract_arg.expanduser().resolve()
    )
    candidate_contract_record = (
        contract_record
        if candidate_contract_arg is None
        else require_file_hash(
            candidate_contract_path,
            expected_candidate_contract,
            "candidate compact contract",
        )
    )
    if candidate_contract_path != contract_path:
        candidate_contract_value = strict_json(candidate_contract_path)
        target_contract_value = strict_json(contract_path)
        allowed_capacity_fields = {"max_target_tokens", "max_total_tokens"}
        candidate_fixed = {
            key: value
            for key, value in candidate_contract_value.items()
            if key not in allowed_capacity_fields
        }
        target_fixed = {
            key: value
            for key, value in target_contract_value.items()
            if key not in allowed_capacity_fields
        }
        if (
            candidate_fixed != target_fixed
            or int(target_contract_value.get("max_target_tokens", -1))
            < int(candidate_contract_value.get("max_target_tokens", -1))
            or int(target_contract_value.get("max_total_tokens", -1))
            < int(candidate_contract_value.get("max_total_tokens", -1))
        ):
            raise ArtifactError(
                "candidate/target contracts differ beyond monotonic capacity"
            )
    tokenizer, tokenizer_record = load_student_tokenizer(
        contract,
        args.student_tokenizer_json.expanduser().resolve(),
        args.expected_student_tokenizer_sha256,
    )

    candidate_compact_path = args.candidate_compact_jsonl.expanduser().resolve()
    candidate_seal_path = args.candidate_compact_seal.expanduser().resolve()
    legacy_compact_path = args.legacy_compact_jsonl.expanduser().resolve()
    legacy_seal_path = args.legacy_compact_seal.expanduser().resolve()
    candidate_compact_record = require_file_hash(
        candidate_compact_path,
        args.expected_candidate_compact_sha256,
        "candidate compact train",
    )
    candidate_seal_record = require_file_hash(
        candidate_seal_path,
        args.expected_candidate_compact_seal_sha256,
        "candidate compact seal",
    )
    legacy_compact_record = require_file_hash(
        legacy_compact_path,
        args.expected_legacy_compact_sha256,
        "legacy compact train",
    )
    legacy_seal_record = require_file_hash(
        legacy_seal_path,
        args.expected_legacy_compact_seal_sha256,
        "legacy compact seal",
    )
    validate_join_seal(
        candidate_compact_path,
        candidate_seal_path,
        candidate_contract_path,
        expected_role="fit",
    )
    validate_join_seal(
        legacy_compact_path,
        legacy_seal_path,
        contract_path,
        expected_role="fit",
    )
    candidate_rows = read_jsonl(candidate_compact_path)
    legacy_rows = read_jsonl(legacy_compact_path)
    if len(candidate_rows) not in EXPECTED_CANDIDATE_COUNTS:
        raise ArtifactError(
            "candidate compact rows must be the exact 2,951 Phase-0 train "
            "or already-filtered 2,776 fit rows"
        )
    if len(legacy_rows) != EXPECTED_LEGACY_TASKS:
        raise ArtifactError(
            f"legacy compact rows={len(legacy_rows)}, expected 1580"
        )
    candidate_order, candidate_by_task = _indexed(
        candidate_rows, "candidate compact"
    )
    legacy_order, legacy_by_task = _indexed(legacy_rows, "legacy compact")
    for index, row in enumerate(candidate_rows):
        contract.validate_row(row, f"candidate-compact-{index}")
        target_text(row, f"candidate-compact-{index}")
    for index, row in enumerate(legacy_rows):
        contract.validate_row(row, f"legacy-compact-{index}")
        target_text(row, f"legacy-compact-{index}")

    heldout_path = args.heldout_jsonl.expanduser().resolve()
    heldout_record = require_file_hash(
        heldout_path, args.expected_heldout_sha256, "heldout175"
    )
    heldout_rows = read_jsonl(heldout_path)
    heldout_seal_path = args.heldout_seal.expanduser().resolve()
    heldout_seal_record = require_file_hash(
        heldout_seal_path,
        args.expected_heldout_seal_sha256,
        "heldout175 seal",
    )
    validate_join_seal(
        heldout_path,
        heldout_seal_path,
        candidate_contract_path,
        expected_role="measure",
    )
    expansion_build_seal_path = args.expansion_build_seal.expanduser().resolve()
    expansion_build_report_path = (
        args.expansion_build_report.expanduser().resolve()
    )
    expansion_build_seal_record = require_file_hash(
        expansion_build_seal_path,
        args.expected_expansion_build_seal_sha256,
        "expansion build seal",
    )
    expansion_build_report_record = require_file_hash(
        expansion_build_report_path,
        args.expected_expansion_build_report_sha256,
        "expansion build report",
    )
    heldout_order, heldout_by_task = _indexed(heldout_rows, "heldout175")
    if len(heldout_order) != EXPECTED_HOLDOUT_TASKS:
        raise ArtifactError(
            f"heldout task count={len(heldout_order)}, expected 175"
        )
    heldout_ids = set(heldout_by_task)
    candidate_ids = set(candidate_by_task)
    fit_order, supplement_order = derive_partition(
        candidate_order=candidate_order,
        legacy_order=legacy_order,
        heldout_order=heldout_order,
    )
    fit_ids = set(fit_order)
    legacy_ids = set(legacy_order)

    candidate_prompt_path = args.candidate_prompt_jsonl.expanduser().resolve()
    legacy_prompt_path = args.legacy_prompt_jsonl.expanduser().resolve()
    candidate_prompts, candidate_prompt_record = load_verified_prompt_rows(
        candidate_prompt_path,
        expected_sha256=args.expected_candidate_prompt_sha256,
        expected_rows=len(candidate_rows),
    )
    legacy_prompts, legacy_prompt_record = load_verified_prompt_rows(
        legacy_prompt_path,
        expected_sha256=args.expected_legacy_prompt_sha256,
        expected_rows=EXPECTED_LEGACY_TASKS,
    )
    candidate_prompt_rows = read_jsonl(candidate_prompt_path)
    legacy_prompt_rows = read_jsonl(legacy_prompt_path)
    candidate_prompt_order, candidate_prompt_by_task = _indexed(
        candidate_prompt_rows, "candidate prompt"
    )
    legacy_prompt_order, legacy_prompt_by_task = _indexed(
        legacy_prompt_rows, "legacy prompt"
    )
    if candidate_prompt_order != candidate_order:
        raise ArtifactError(
            "candidate compact and F2 prompt files have different task order"
        )
    if legacy_prompt_order != legacy_order:
        raise ArtifactError(
            "legacy compact and F2 prompt files have different task order"
        )
    if {row.task_id for row in candidate_prompts} != candidate_ids:
        raise ArtifactError("candidate verified prompt task set drifted")
    if {row.task_id for row in legacy_prompts} != legacy_ids:
        raise ArtifactError("legacy verified prompt task set drifted")
    for task_id in candidate_order:
        expected = compact_ids_sha256(
            candidate_by_task[task_id], f"candidate-{task_id}"
        )
        if candidate_prompt_by_task[task_id].get("compact_ids_sha256") != expected:
            raise ArtifactError(
                f"{task_id}: candidate F2 prompt compact join key differs"
            )
    for task_id in legacy_order:
        expected = compact_ids_sha256(
            legacy_by_task[task_id], f"legacy-{task_id}"
        )
        if legacy_prompt_by_task[task_id].get("compact_ids_sha256") != expected:
            raise ArtifactError(
                f"{task_id}: legacy F2 prompt compact join key differs"
            )

    candidate_prompt_manifest_path = (
        args.candidate_prompt_manifest.expanduser().resolve()
    )
    legacy_prompt_manifest_path = args.legacy_prompt_manifest.expanduser().resolve()
    candidate_system, candidate_manifest_record, candidate_manifest = (
        load_f2_prompt_contract(
            candidate_prompt_manifest_path,
            expected_sha256=args.expected_candidate_prompt_manifest_sha256,
            prompt_record=candidate_prompt_record,
            expected_rows=len(candidate_rows),
            student_tokenizer_sha256=tokenizer_record["sha256"],
        )
    )
    legacy_system, legacy_manifest_record, legacy_manifest = (
        load_f2_prompt_contract(
            legacy_prompt_manifest_path,
            expected_sha256=args.expected_legacy_prompt_manifest_sha256,
            prompt_record=legacy_prompt_record,
            expected_rows=EXPECTED_LEGACY_TASKS,
            student_tokenizer_sha256=tokenizer_record["sha256"],
        )
    )
    if candidate_system != legacy_system:
        raise ArtifactError(
            "candidate and live-parent F2 system prompts are not byte-identical"
        )
    candidate_f2 = candidate_manifest["f2_prompt_contract"]
    legacy_f2 = legacy_manifest["f2_prompt_contract"]
    # Corpus-wide maximum/task diagnostics legitimately change when rows are
    # added.  The grammar, tokenizer, and fixed capacity contract may not.
    immutable_f2_fields = (
        "representation_schema",
        "system_prompt",
        "system_prompt_sha256",
        "tokenizer_sha256",
        "max_prompt_tokens",
        "chat_overhead_reserve",
        "constant_prefix_token_cap",
        "all_rows_within_limit",
    )
    changed_f2 = [
        field
        for field in immutable_f2_fields
        if candidate_f2.get(field) != legacy_f2.get(field)
    ]
    if changed_f2:
        raise ArtifactError(
            "candidate and live-parent immutable F2 prompt contract differs: "
            + ", ".join(changed_f2)
        )
    expansion_membership = validate_expansion_evidence(
        expansion_seal_path=expansion_build_seal_path,
        expansion_report_path=expansion_build_report_path,
        candidate_compact_record=candidate_compact_record,
        candidate_seal_record=candidate_seal_record,
        candidate_prompt_record=candidate_prompt_record,
        candidate_prompt_manifest_record=candidate_manifest_record,
        legacy_compact_record=legacy_compact_record,
        heldout_record=heldout_record,
        heldout_seal_record=heldout_seal_record,
        candidate_contract_record=candidate_contract_record,
        candidate_order=candidate_order,
        legacy_order=legacy_order,
        supplement_order=supplement_order,
        heldout_order=heldout_order,
    )

    fit_compact_by_task = {
        task_id: candidate_by_task[task_id] for task_id in fit_order
    }
    fit_prompt_by_task = {
        task_id: candidate_prompt_by_task[task_id] for task_id in fit_order
    }
    compatibility = _shared_compatibility(
        legacy_ids=legacy_order,
        legacy_compact=legacy_by_task,
        fit_compact=fit_compact_by_task,
        legacy_prompts=legacy_prompt_by_task,
        fit_prompts=fit_prompt_by_task,
    )

    fit_compact_rows = [fit_compact_by_task[task_id] for task_id in fit_order]
    supplement_compact_rows = [
        fit_compact_by_task[task_id] for task_id in supplement_order
    ]
    fit_prompt_rows = [fit_prompt_by_task[task_id] for task_id in fit_order]
    supplement_prompt_rows = [
        fit_prompt_by_task[task_id] for task_id in supplement_order
    ]
    fit_compact_output = args.fit_compact_output.expanduser().resolve()
    fit_seal_output = args.fit_compact_seal_output.expanduser().resolve()
    fit_prompt_output = args.fit_prompt_output.expanduser().resolve()
    fit_prompt_manifest_output = (
        args.fit_prompt_manifest_output.expanduser().resolve()
    )
    supplement_compact_output = args.supplement_compact_output.expanduser().resolve()
    supplement_seal_output = (
        args.supplement_compact_seal_output.expanduser().resolve()
    )
    supplement_prompt_output = args.supplement_prompt_output.expanduser().resolve()
    supplement_prompt_manifest_output = (
        args.supplement_prompt_manifest_output.expanduser().resolve()
    )
    atomic_write_jsonl(fit_compact_output, fit_compact_rows)
    atomic_write_jsonl(supplement_compact_output, supplement_compact_rows)
    atomic_write_jsonl(fit_prompt_output, fit_prompt_rows)
    atomic_write_jsonl(supplement_prompt_output, supplement_prompt_rows)
    fit_seal = exact_output_seal(
        output_path=fit_compact_output,
        contract_path=contract_path,
        contract=contract,
        rows=fit_compact_rows,
        tokenizer=tokenizer,
    )
    supplement_seal = exact_output_seal(
        output_path=supplement_compact_output,
        contract_path=contract_path,
        contract=contract,
        rows=supplement_compact_rows,
        tokenizer=tokenizer,
    )
    atomic_write_json(fit_seal_output, fit_seal)
    atomic_write_json(supplement_seal_output, supplement_seal)
    validate_join_seal(
        fit_compact_output,
        fit_seal_output,
        contract_path,
        expected_role="fit",
    )
    validate_join_seal(
        supplement_compact_output,
        supplement_seal_output,
        contract_path,
        expected_role="fit",
    )
    fit_prompt_manifest = _prompt_manifest(
        parent_manifest=candidate_manifest,
        parent_manifest_record=candidate_manifest_record,
        prompt_path=fit_prompt_output,
        compact_path=fit_compact_output,
        rows=len(fit_order),
        scope="phase0_train_minus_heldout175",
        ordered_task_ids=fit_order,
        prompt_rows=fit_prompt_rows,
        heldout_record=heldout_record,
    )
    supplement_prompt_manifest = _prompt_manifest(
        parent_manifest=candidate_manifest,
        parent_manifest_record=candidate_manifest_record,
        prompt_path=supplement_prompt_output,
        compact_path=supplement_compact_output,
        rows=len(supplement_order),
        scope="fit2776_minus_live_parent1580",
        ordered_task_ids=supplement_order,
        prompt_rows=supplement_prompt_rows,
        heldout_record=heldout_record,
    )
    atomic_write_json(fit_prompt_manifest_output, fit_prompt_manifest)
    atomic_write_json(
        supplement_prompt_manifest_output, supplement_prompt_manifest
    )
    # Re-read through the production collector's manifest gate.
    load_f2_prompt_contract(
        fit_prompt_manifest_output,
        expected_sha256=sha256_file(fit_prompt_manifest_output),
        prompt_record=file_record(fit_prompt_output),
        expected_rows=EXPECTED_FIT_TASKS,
        student_tokenizer_sha256=tokenizer_record["sha256"],
    )
    load_f2_prompt_contract(
        supplement_prompt_manifest_output,
        expected_sha256=sha256_file(supplement_prompt_manifest_output),
        prompt_record=file_record(supplement_prompt_output),
        expected_rows=EXPECTED_SUPPLEMENT_TASKS,
        student_tokenizer_sha256=tokenizer_record["sha256"],
    )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "fit_scope": "phase0_train_minus_heldout175",
        "counts": {
            "candidate_tasks": len(candidate_order),
            "heldout_tasks": len(heldout_order),
            "fit_tasks": len(fit_order),
            "legacy_parent_tasks": len(legacy_order),
            "supplement_tasks": len(supplement_order),
            "samples_per_task": 8,
            "legacy_teacher_slots": len(legacy_order) * 8,
            "supplement_teacher_slots": len(supplement_order) * 8,
            "union_teacher_slots": len(fit_order) * 8,
        },
        "ordered_task_ids": fit_order,
        "ordered_task_ids_sha256": stable_sha256(fit_order),
        "fit_task_set_sha256": stable_sha256(sorted(fit_order)),
        "legacy_ordered_task_ids_sha256": stable_sha256(legacy_order),
        "legacy_task_set_sha256": stable_sha256(sorted(legacy_order)),
        "supplement_ordered_task_ids": supplement_order,
        "supplement_ordered_task_ids_sha256": stable_sha256(supplement_order),
        "supplement_task_set_sha256": stable_sha256(sorted(supplement_order)),
        "heldout_ordered_task_ids_sha256": stable_sha256(heldout_order),
        "heldout_task_set_sha256": stable_sha256(sorted(heldout_order)),
        "heldout_intersection_count": len(fit_ids.intersection(heldout_ids)),
        "expansion_membership": expansion_membership,
        "set_equations": {
            "fit_equals_candidate_minus_heldout": True,
            "fit_equals_legacy_disjoint_union_supplement": True,
            "legacy_supplement_intersection_count": len(
                legacy_ids.intersection(supplement_order)
            ),
        },
        "live_parent_compatibility": compatibility,
        "inputs": {
            "candidate_compact": candidate_compact_record,
            "candidate_compact_seal": candidate_seal_record,
            "candidate_prompt": candidate_prompt_record,
            "candidate_prompt_manifest": candidate_manifest_record,
            "legacy_compact": legacy_compact_record,
            "legacy_compact_seal": legacy_seal_record,
            "legacy_prompt": legacy_prompt_record,
            "legacy_prompt_manifest": legacy_manifest_record,
            "heldout175": heldout_record,
            "heldout175_seal": heldout_seal_record,
            "expansion_build_seal": expansion_build_seal_record,
            "expansion_build_report": expansion_build_report_record,
            "candidate_contract": candidate_contract_record,
            "contract": contract_record,
            "student_tokenizer": tokenizer_record,
        },
        "outputs": {
            "fit_compact": file_record(fit_compact_output),
            "fit_compact_seal": file_record(fit_seal_output),
            "fit_prompt": file_record(fit_prompt_output),
            "fit_prompt_manifest": file_record(fit_prompt_manifest_output),
            "supplement_compact": file_record(supplement_compact_output),
            "supplement_compact_seal": file_record(supplement_seal_output),
            "supplement_prompt": file_record(supplement_prompt_output),
            "supplement_prompt_manifest": file_record(
                supplement_prompt_manifest_output
            ),
        },
        "invariants": {
            "live_journal_read": False,
            "live_journal_modified": False,
            "candidate_rows_reencoded": False,
            "candidate_prompt_rows_reencoded": False,
            "heldout_used_for_teacher_collection": False,
            "heldout_used_for_fit": False,
            "task_selection_depends_only_on_sealed_membership": True,
            "exact_counts_passed": True,
        },
    }
    if manifest["heldout_intersection_count"] != 0:
        raise AssertionError("heldout leaked into fit")
    atomic_write_json(args.derivation_manifest_output.expanduser().resolve(), manifest)
    return manifest


def main() -> int:
    manifest = build(parse_args())
    print(
        "QWEN_2776_SUPPLEMENT "
        f"fit={manifest['counts']['fit_tasks']} "
        f"legacy={manifest['counts']['legacy_parent_tasks']} "
        f"supplement={manifest['counts']['supplement_tasks']} "
        f"slots={manifest['counts']['union_teacher_slots']} "
        "live_journal_modified=false heldout_intersection=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
