#!/usr/bin/env python3
"""Harvest ChatGPT repairs only for failed direct-compact training tasks.

The API receives the same verified, API-readable compressed assembly, binary
constants, and explicit compressed CFG used by the audited frontier runner.
Private tests and gold Dart are never included in the request.  Every returned
candidate is independently replayed through the completion-attested Dart
harness before it can enter ``verified_repairs.jsonl``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import socket
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from models.direct_compact_causal import sha256_artifact
from scripts.preprocessing.build_multifunction_executable_view import (
    F2_REPRESENTATION_SCHEMA,
    REPRESENTATION_SCHEMA,
    validate_executable_view,
)
from scripts.evaluation.graph_compile_at_k_antigravity import (
    evaluate_dart_jit_tests_detail,
)


SCHEMA = "direct-compact-openai-rs-harvest-v2"
RUN_CONTRACT_SCHEMA = "direct-compact-openai-rs-run-contract-v1"
QWEN_BUILD_SCHEMA = "direct-compact-mc-sequence-forward-kl-nll-build-v1"
QWEN_AUDIT_SCHEMA = "qwen-direct-compact-teacher-audit-v1"
QWEN_STAGE_MODEL = "qwen3.8-max-preview"
QWEN_TASK_ROWS = 1580
QWEN_DRAWS_PER_TASK = 8
QWEN_TEACHER_DRAW_ROWS = QWEN_TASK_ROWS * QWEN_DRAWS_PER_TASK
QWEN_GOLD_REPLAY_FRACTION = 0.0
QWEN_GOLD_REPLAY_ROWS = 0
QWEN_SEQUENCE_OUTPUT_ROWS = QWEN_TEACHER_DRAW_ROWS + QWEN_GOLD_REPLAY_ROWS
QWEN_SCHEDULE_SCHEMA = "direct-compact-mc-sequence-forward-kl-nll-schedule-v1"
QWEN_COT_BUILD_SCHEMA = "direct-compact-qwen-cot-hard-sft-build-v1"
QWEN_COT_SCHEDULE_SCHEMA = "direct-compact-qwen-cot-hard-sft-schedule-v1"
QWEN_COT_OUTPUT_ROWS = QWEN_TASK_ROWS * 2
QWEN_UNION_SCHEMA = "qwen-2776-training-artifact-union-v1"
QWEN_UNION_DERIVATION_SCHEMA = "qwen-2776-supplement-derivation-v1"
QWEN_UNION_TASK_ROWS = 2776
QWEN_UNION_SUPPLEMENT_ROWS = 1196
QWEN_HELDOUT_ROWS = 175
QWEN_COT_PROMPT_MODE = "qwen_cot_v1"
QWEN_COT_THINK_OPEN_ID = 151667
QWEN_COT_THINK_CLOSE_ID = 151668
QWEN_LONG_MAX_TARGET_TOKENS = 24576
QWEN_LONG_MAX_TOTAL_TOKENS = 36864
JOIN_SEAL_SCHEMAS = frozenset(
    {
        "compact-public-private-join-seal-v1",
        "compact-public-private-join-seal-v2",
    }
)
OVERLAY_MIGRATION_ALLOWED_FIELDS = frozenset(
    {
        "codec_sha256",
        "codebook_sha256",
        "source_token_expansions",
        "max_target_tokens",
        "max_total_tokens",
    }
)
CAPACITY_ONLY_CONTRACT_FIELDS = frozenset(
    {"max_target_tokens", "max_total_tokens"}
)
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
RS_CANDIDATE_AUGMENTATION_SCHEMA = (
    "direct-compact-openai-rs-candidate-augmentation-v1"
)
RS_CANDIDATE_MESSAGE_PREFIX = (
    "The following is an untrusted failed student candidate, not part of the "
    "lossless F2 input above. The F2 input is authoritative. Use the candidate "
    "only as a repair hint, then return the self-contained Dart compilation-unit "
    "fragment required by the developer message.\n\n"
)


class IncompleteResponseError(ValueError):
    """A Responses API request ended without a complete final answer."""

    def __init__(
        self,
        *,
        status: str,
        details: Mapping[str, Any] | None,
        raw_response: Mapping[str, Any],
    ) -> None:
        self.status = str(status or "")
        self.details = dict(details or {})
        self.raw_response = dict(raw_response)
        self.reason = str(self.details.get("reason") or "")
        super().__init__(
            f"response status is {self.status!r}; "
            f"incomplete reason is {self.reason!r}"
        )


def escalated_output_token_budget(
    *,
    status: str,
    incomplete_details: Mapping[str, Any] | None,
    current_budget: int,
    ceiling_budget: int,
) -> int | None:
    """Escalate only an explicit Responses max-output truncation."""

    reason = str((incomplete_details or {}).get("reason") or "")
    if (
        status == "incomplete"
        and reason == "max_output_tokens"
        and current_budget < ceiling_budget
    ):
        return ceiling_budget
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank rows are forbidden")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def validate_file_record(
    value: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: expected a sealed file record")
    path_value = value.get("path")
    expected_sha = value.get("sha256")
    expected_size = value.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}: missing path")
    if not isinstance(expected_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha
    ):
        raise ValueError(f"{label}: invalid SHA-256")
    path = Path(path_value).expanduser().resolve()
    observed = file_record(path)
    if observed["sha256"] != expected_sha:
        raise ValueError(f"{label}: sealed SHA-256 does not match the file")
    if expected_size is not None and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size != observed["size_bytes"]
    ):
        raise ValueError(f"{label}: sealed size does not match the file")
    return path, observed


def validate_fit_seal(
    seal_path: Path,
    *,
    dataset_record: Mapping[str, Any],
    contract_sha256: str,
    expected_rows: int,
    label: str,
) -> dict[str, Any]:
    """Validate the minimum immutable join bindings used by this stage."""

    seal = load_object(seal_path)
    if (
        seal.get("schema") not in JOIN_SEAL_SCHEMAS
        or seal.get("selected_role") != "fit"
        or isinstance(seal.get("rows"), bool)
        or seal.get("rows") != expected_rows
        or seal.get("output_sha256") != dataset_record.get("sha256")
        or seal.get("contract_sha256") != contract_sha256
    ):
        raise ValueError(
            f"{label} does not bind the fit dataset, row count, and current contract"
        )
    output_size = seal.get("output_size_bytes")
    if output_size is not None and output_size != dataset_record.get("size_bytes"):
        raise ValueError(f"{label} sealed output size differs")
    return seal


def _validate_qwen_fit_contract(
    build: Mapping[str, Any],
    *,
    samples_per_task: int,
    label: str,
) -> dict[str, Any]:
    """Resolve legacy or expanded fit membership from the build attestation."""

    union = build.get("union_2776")
    if union is None:
        return {
            "expanded": False,
            "task_count": QWEN_TASK_ROWS,
            "ordered_task_ids": None,
            "ordered_task_ids_sha256": None,
            "heldout_task_count": QWEN_HELDOUT_ROWS,
            "heldout_intersection_count": 0,
            "samples_per_task": samples_per_task,
            "expected_grid_rows": QWEN_TASK_ROWS * samples_per_task,
            "derivation": None,
            "derivation_record": None,
            "parent_builds": None,
        }
    if not isinstance(union, Mapping):
        raise ValueError(f"{label} union_2776 is not an object")
    ordered = union.get("ordered_task_ids")
    invariants = union.get("invariants")
    if (
        union.get("schema") != QWEN_UNION_SCHEMA
        or union.get("fit_scope") != "phase0_train_minus_heldout175"
        or union.get("task_count") != QWEN_UNION_TASK_ROWS
        or union.get("legacy_task_count") != QWEN_TASK_ROWS
        or union.get("supplement_task_count")
        != QWEN_UNION_SUPPLEMENT_ROWS
        or union.get("heldout_task_count") != QWEN_HELDOUT_ROWS
        or union.get("heldout_intersection_count") != 0
        or union.get("samples_per_task") != samples_per_task
        or union.get("expected_grid_rows")
        != QWEN_UNION_TASK_ROWS * samples_per_task
        or union.get("observed_grid_rows")
        != QWEN_UNION_TASK_ROWS * samples_per_task
        or not isinstance(ordered, list)
        or len(ordered) != QWEN_UNION_TASK_ROWS
        or any(not isinstance(task_id, str) or not task_id for task_id in ordered)
        or len(set(ordered)) != len(ordered)
        or stable_sha256(ordered)
        != union.get("ordered_task_ids_sha256")
        or stable_sha256(sorted(ordered)) != union.get("task_set_sha256")
        or not isinstance(invariants, Mapping)
        or invariants.get("parent_journals_modified") is not False
        or invariants.get("parent_rows_modified") is not False
        or invariants.get("teacher_targets_filtered") is not False
        or invariants.get("teacher_targets_resampled") is not False
        or invariants.get("heldout_used_for_fit") is not False
        or invariants.get("heldout_used_for_teacher_collection") is not False
        or invariants.get("exact_task_partition") is not True
    ):
        raise ValueError(f"{label} expanded fit-union contract failed")
    derivation_path, derivation_record = validate_file_record(
        union.get("derivation_manifest"),
        label=f"{label} 2,776-task derivation manifest",
    )
    derivation = load_object(derivation_path)
    counts = derivation.get("counts")
    if (
        derivation.get("schema") != QWEN_UNION_DERIVATION_SCHEMA
        or derivation.get("fit_scope")
        != "phase0_train_minus_heldout175"
        or not isinstance(counts, Mapping)
        or counts.get("fit_tasks") != QWEN_UNION_TASK_ROWS
        or counts.get("legacy_parent_tasks") != QWEN_TASK_ROWS
        or counts.get("supplement_tasks")
        != QWEN_UNION_SUPPLEMENT_ROWS
        or counts.get("heldout_tasks") != QWEN_HELDOUT_ROWS
        or derivation.get("ordered_task_ids") != ordered
        or derivation.get("ordered_task_ids_sha256")
        != union.get("ordered_task_ids_sha256")
        or derivation.get("heldout_intersection_count") != 0
        or (derivation.get("set_equations") or {}).get(
            "fit_equals_legacy_disjoint_union_supplement"
        )
        is not True
        or (derivation.get("invariants") or {}).get(
            "heldout_used_for_fit"
        )
        is not False
    ):
        raise ValueError(f"{label} derivation membership contract failed")
    inputs = build.get("inputs")
    parent_builds = (
        inputs.get("parent_builds") if isinstance(inputs, Mapping) else None
    )
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("union_derivation") != derivation_record
        or not isinstance(parent_builds, list)
        or len(parent_builds) != 2
        or union.get("parents") != parent_builds
        or sum(
            int(parent.get("task_count", -1))
            for parent in parent_builds
            if isinstance(parent, Mapping)
        )
        != QWEN_UNION_TASK_ROWS
    ):
        raise ValueError(f"{label} parent-build union binding failed")
    derivation_outputs = derivation.get("outputs")
    derivation_inputs = derivation.get("inputs")
    if (
        not isinstance(derivation_outputs, Mapping)
        or not isinstance(derivation_inputs, Mapping)
        or inputs.get("compact_train")
        != derivation_outputs.get("fit_compact")
        or inputs.get("compact_train_seal")
        != derivation_outputs.get("fit_compact_seal")
        or inputs.get("contract") != derivation_inputs.get("contract")
        or union.get("contract") != inputs.get("contract")
        or union.get("student_tokenizer")
        != inputs.get("student_tokenizer")
    ):
        raise ValueError(f"{label} fit artifact derivation binding failed")
    for index, parent in enumerate(parent_builds):
        if not isinstance(parent, Mapping):
            raise ValueError(f"{label} parent build {index} is malformed")
        for key in ("dataset", "seal", "schedule", "build_manifest"):
            validate_file_record(
                parent.get(key),
                label=f"{label} parent {index} {key}",
            )
    return {
        "expanded": True,
        "task_count": QWEN_UNION_TASK_ROWS,
        "ordered_task_ids": list(ordered),
        "ordered_task_ids_sha256": str(
            union["ordered_task_ids_sha256"]
        ),
        "heldout_task_count": QWEN_HELDOUT_ROWS,
        "heldout_intersection_count": 0,
        "samples_per_task": samples_per_task,
        "expected_grid_rows": QWEN_UNION_TASK_ROWS * samples_per_task,
        "derivation": derivation,
        "derivation_record": derivation_record,
        "parent_builds": list(parent_builds),
    }


def _checkpoint_paths(root: Path, *, label: str) -> dict[str, Path]:
    paths = {
        "root": root,
        "adapter": root / "decoder_adapter",
        "overlay": root / "source_embedding_overlay.pt",
        "contract": root / "compact_contract.json",
        "provenance": root / "run_provenance.json",
    }
    for path in (
        paths["adapter"] / "adapter_config.json",
        paths["overlay"],
        paths["contract"],
        paths["provenance"],
    ):
        if not path.is_file():
            raise ValueError(f"{label} is incomplete: {path}")
    return paths


def _validate_checkpoint_binding(
    binding: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    label: str,
) -> dict[str, str]:
    expected = {
        "path": str(paths["root"]),
        "decoder_adapter_sha256": sha256_artifact(paths["adapter"]),
        "source_overlay_sha256": sha256_file(paths["overlay"]),
        "contract_sha256": sha256_file(paths["contract"]),
        "provenance_sha256": sha256_file(paths["provenance"]),
    }
    if dict(binding) != expected:
        raise ValueError(f"{label} artifact binding differs")
    return expected


def _validate_gold_sft_checkpoint(
    paths: Mapping[str, Path],
    *,
    binding: Mapping[str, Any],
    compact_train_record: Mapping[str, Any],
    compact_train_seal_path: Path,
    compact_train_seal_record: Mapping[str, Any],
    expected_contract_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected_binding = _validate_checkpoint_binding(binding, paths, label=label)
    provenance = load_object(paths["provenance"])
    contract_sha = sha256_file(paths["contract"])
    if contract_sha != expected_contract_sha256:
        raise ValueError(f"{label} compact contract differs")
    validate_fit_seal(
        compact_train_seal_path,
        dataset_record=compact_train_record,
        contract_sha256=contract_sha,
        expected_rows=QWEN_TASK_ROWS,
        label=f"{label} train seal",
    )
    loss_contract = provenance.get("loss_contract")
    if (
        provenance.get("schema") != "direct-compact-run-provenance-v1"
        or provenance.get("architecture")
        != "qwen-causal-compact-tokens-no-encoder"
        or provenance.get("decoder_adapter_sha256")
        != expected_binding["decoder_adapter_sha256"]
        or provenance.get("source_overlay_sha256")
        != expected_binding["source_overlay_sha256"]
        or provenance.get("contract_sha256") != contract_sha
        or not isinstance(loss_contract, Mapping)
        or loss_contract.get("sequence_distribution_nll") is not False
        or loss_contract.get("primary_reduction")
        != "base_causal_lm_token_mean"
        or provenance.get("train_file_sha256")
        != compact_train_record.get("sha256")
        or provenance.get("train_seal_sha256")
        != compact_train_seal_record.get("sha256")
        or int(provenance.get("train_sealed_rows", -1)) != QWEN_TASK_ROWS
        or provenance.get("heldout_loaded_during_training") is not False
        or provenance.get("eval_file_sha256") is not None
        or provenance.get("eval_seal_sha256") is not None
        or provenance.get("eval_sealed_rows") is not None
        or provenance.get("eval_strategy") != "no"
    ):
        raise ValueError(
            f"{label} is not the sealed train-only 1,580-row gold-SFT stage"
        )
    return {
        "path": str(paths["root"]),
        "run_provenance": file_record(paths["provenance"]),
        "decoder_adapter_sha256": expected_binding["decoder_adapter_sha256"],
        "source_embedding_overlay_sha256": expected_binding[
            "source_overlay_sha256"
        ],
        "compact_contract_sha256": contract_sha,
        "train_seal": file_record(compact_train_seal_path),
    }


def _find_sha256_file(
    directory: Path,
    expected_sha256: str,
    *,
    pattern: str,
    label: str,
) -> Path:
    matches = [
        candidate.resolve()
        for candidate in sorted(directory.glob(pattern))
        if candidate.is_file() and sha256_file(candidate) == expected_sha256
    ]
    if not matches:
        raise ValueError(f"{label} is missing from {directory}")
    return matches[0]


def _source_token_contract(
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[list[int], dict[int, list[int]]]:
    raw = value.get("source_token_expansions")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{label} has no source-token expansion contract")
    expansions: dict[int, list[int]] = {}
    for source_id_value, expansion_value in raw.items():
        try:
            source_id = int(source_id_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} has an invalid source-token ID") from exc
        if (
            isinstance(expansion_value, (str, bytes))
            or not isinstance(expansion_value, Sequence)
            or not expansion_value
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in expansion_value
            )
        ):
            raise ValueError(
                f"{label} has an invalid expansion for source token {source_id}"
            )
        if source_id in expansions:
            raise ValueError(f"{label} repeats source token {source_id}")
        expansions[source_id] = [int(token_id) for token_id in expansion_value]
    source_ids = sorted(expansions)
    return source_ids, expansions


def _expansion_sha256(expansions: Mapping[int, Sequence[int]]) -> str:
    return stable_sha256(
        {
            str(source_id): list(expansions[source_id])
            for source_id in sorted(expansions)
        }
    )


def validate_capacity_only_contracts(
    old_contract: Mapping[str, Any],
    new_contract: Mapping[str, Any],
) -> list[str]:
    """Return changed capacity fields, rejecting any semantic contract change."""

    changed_fields = sorted(
        key
        for key in set(old_contract).union(new_contract)
        if old_contract.get(key) != new_contract.get(key)
    )
    if (
        not changed_fields
        or not set(changed_fields).issubset(CAPACITY_ONLY_CONTRACT_FIELDS)
    ):
        raise ValueError(
            "Qwen gold migration is not capacity-only; changed contract fields: "
            + ", ".join(changed_fields)
        )
    for field in CAPACITY_ONLY_CONTRACT_FIELDS:
        old_value = old_contract.get(field)
        new_value = new_contract.get(field)
        if (
            isinstance(old_value, bool)
            or not isinstance(old_value, int)
            or isinstance(new_value, bool)
            or not isinstance(new_value, int)
            or new_value < old_value
        ):
            raise ValueError(
                f"Qwen gold migration did not monotonically increase {field}"
            )
    return changed_fields


def _validate_capacity_migrated_gold_checkpoint(
    paths: Mapping[str, Path],
    *,
    binding: Mapping[str, Any],
    compact_train_record: Mapping[str, Any],
    current_train_seal_path: Path,
    current_train_seal_record: Mapping[str, Any],
    current_contract_sha256: str,
) -> dict[str, Any]:
    """Validate a zero-step, capacity-only wrapper around the gold-SFT run."""

    migrated_binding = _validate_checkpoint_binding(
        binding, paths, label="Qwen capacity-migrated gold warmstart"
    )
    provenance = load_object(paths["provenance"])
    receipt_path = paths["root"] / "overlay_migration_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(
            "Qwen capacity-migrated gold warmstart has no migration receipt"
        )
    receipt = load_object(receipt_path)
    source_binding = receipt.get("source_checkpoint")
    if not isinstance(source_binding, Mapping):
        raise ValueError("Qwen gold migration receipt has no source checkpoint")
    source_root_value = source_binding.get("path")
    if not isinstance(source_root_value, str) or not source_root_value:
        raise ValueError("Qwen gold migration receipt has no source path")
    source_paths = _checkpoint_paths(
        Path(source_root_value).expanduser().resolve(),
        label="nested source gold-SFT checkpoint",
    )
    expected_source_binding = _validate_checkpoint_binding(
        source_binding,
        source_paths,
        label="nested source gold-SFT checkpoint",
    )

    old_contract = load_object(source_paths["contract"])
    new_contract = load_object(paths["contract"])
    changed_fields = validate_capacity_only_contracts(
        old_contract, new_contract
    )
    if sha256_file(paths["contract"]) != current_contract_sha256:
        raise ValueError(
            "Qwen capacity-migrated gold warmstart lacks the current contract"
        )

    source_ids, old_expansions = _source_token_contract(
        old_contract, label="source gold compact contract"
    )
    new_source_ids, new_expansions = _source_token_contract(
        new_contract, label="migrated gold compact contract"
    )
    if source_ids != new_source_ids or old_expansions != new_expansions:
        raise ValueError("Qwen capacity migration changed source-token semantics")
    base_vocab_size = new_contract.get("base_vocab_size")
    if (
        isinstance(base_vocab_size, bool)
        or not isinstance(base_vocab_size, int)
        or base_vocab_size <= 0
        or old_contract.get("base_vocab_size") != base_vocab_size
    ):
        raise ValueError("Qwen capacity migration changed the base vocabulary")

    expected_compatibility = {
        "schema": "direct-compact-overlay-migration-compatibility-v1",
        "old_contract_sha256": sha256_file(source_paths["contract"]),
        "new_contract_sha256": sha256_file(paths["contract"]),
        "allowed_changed_fields": sorted(OVERLAY_MIGRATION_ALLOWED_FIELDS),
        "observed_changed_fields": changed_fields,
        "source_token_rows": len(source_ids),
        "identical_expansion_rows": len(source_ids),
        "changed_expansion_rows": 0,
        "identical_expansion_source_token_ids": source_ids,
        "changed_expansion_source_token_ids": [],
        "all_non_migratable_contract_fields_identical": True,
        "stable_source_token_id_sequence_identical": True,
        "base_vocab_size_identical": True,
    }
    expected_outputs = {
        "decoder_adapter_sha256": migrated_binding["decoder_adapter_sha256"],
        "source_overlay_sha256": migrated_binding["source_overlay_sha256"],
        "compact_contract_sha256": migrated_binding["contract_sha256"],
        "codebook_sha256": new_contract.get("codebook_sha256"),
        "codec_sha256": new_contract.get("codec_sha256"),
    }
    expected_invariants = {
        "no_training_or_optimizer_step_performed": True,
        "decoder_adapter_tree_byte_identical": True,
        "old_overlay_row_reused_only_for_identical_expansion": True,
        "changed_rows_use_new_codebook_mean_initialization": True,
        "new_contract_copied_byte_identically": True,
        "heldout_data_opened": False,
    }
    expected_migration = {
        "schema": "source-token-overlay-expansion-migration-v1",
        "policy": (
            "reuse_learned_row_iff_source_id_and_ordered_base_token_"
            "expansion_are_identical_else_new_codebook_mean"
        ),
        "base_vocab_size": base_vocab_size,
        "source_token_ids": source_ids,
        "source_token_ids_sha256": stable_sha256(source_ids),
        "old_source_token_expansions_sha256": _expansion_sha256(
            old_expansions
        ),
        "new_source_token_expansions_sha256": _expansion_sha256(
            new_expansions
        ),
        "rows": {
            "total": len(source_ids),
            "reused_identical_expansion": len(source_ids),
            "reinitialized_new_codebook_mean": 0,
        },
        "reused_source_token_ids": source_ids,
        "reinitialized_source_token_ids": [],
        "invariants": {
            "stable_source_token_id_set_identical": True,
            "changed_expansion_rows_copied_from_old_overlay": False,
            "changed_expansion_rows_initialized_from_new_codebook_mean": True,
            "base_embedding_and_lm_head_not_resized": True,
        },
    }
    if (
        receipt.get("schema")
        != "direct-compact-overlay-migration-receipt-v1"
        or receipt.get("training_steps") != 0
        or receipt.get("source_checkpoint") != expected_source_binding
        or receipt.get("contract_compatibility") != expected_compatibility
        or receipt.get("overlay_migration") != expected_migration
        or receipt.get("outputs") != expected_outputs
        or receipt.get("invariants") != expected_invariants
    ):
        raise ValueError("Qwen capacity migration receipt contract failed")
    if (
        migrated_binding["decoder_adapter_sha256"]
        != expected_source_binding["decoder_adapter_sha256"]
        or migrated_binding["source_overlay_sha256"]
        != expected_source_binding["source_overlay_sha256"]
    ):
        raise ValueError(
            "Qwen capacity migration changed learned adapter or overlay bytes"
        )

    expected_provenance = {
        "schema": "direct-compact-run-provenance-v1",
        "architecture": "qwen-causal-compact-tokens-no-encoder",
        "checkpoint_stage": "contract-overlay-migration-only",
        "contract_sha256": current_contract_sha256,
        "decoder_adapter_sha256": migrated_binding[
            "decoder_adapter_sha256"
        ],
        "source_overlay_sha256": migrated_binding[
            "source_overlay_sha256"
        ],
        "codebook_sha256": new_contract.get("codebook_sha256"),
        "codec_sha256": new_contract.get("codec_sha256"),
        "source_embedding_overlay_rows": len(source_ids),
        "lm_head_rows": base_vocab_size,
        "training_performed": False,
        "heldout_loaded_during_migration": False,
        "overlay_migration_receipt_sha256": sha256_file(receipt_path),
        "warmstart_checkpoint": expected_source_binding,
    }
    mismatches = [
        key
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "Qwen capacity-migrated gold provenance differs: "
            + ", ".join(mismatches)
        )

    source_provenance = load_object(source_paths["provenance"])
    source_train_seal_sha = source_provenance.get("train_seal_sha256")
    if not isinstance(source_train_seal_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_train_seal_sha
    ):
        raise ValueError("nested source gold-SFT provenance has no train seal")
    source_train_seal_path = _find_sha256_file(
        current_train_seal_path.parent,
        source_train_seal_sha,
        pattern="*.seal.json",
        label="nested source gold-SFT train seal",
    )
    source_train_seal_record = file_record(source_train_seal_path)
    source_gold = _validate_gold_sft_checkpoint(
        source_paths,
        binding=expected_source_binding,
        compact_train_record=compact_train_record,
        compact_train_seal_path=source_train_seal_path,
        compact_train_seal_record=source_train_seal_record,
        expected_contract_sha256=sha256_file(source_paths["contract"]),
        label="nested source gold-SFT checkpoint",
    )
    old_seal = load_object(source_train_seal_path)
    new_seal = load_object(current_train_seal_path)
    old_without_contract = {
        key: value for key, value in old_seal.items() if key != "contract_sha256"
    }
    new_without_contract = {
        key: value for key, value in new_seal.items() if key != "contract_sha256"
    }
    if old_without_contract != new_without_contract:
        raise ValueError(
            "current compact train seal is not the capacity-contract reseal "
            "of the nested gold-SFT dataset"
        )
    if current_train_seal_record.get("sha256") != sha256_file(
        current_train_seal_path
    ):
        raise ValueError("current compact train seal record changed")

    return {
        "path": str(paths["root"]),
        "run_provenance": file_record(paths["provenance"]),
        "decoder_adapter_sha256": migrated_binding[
            "decoder_adapter_sha256"
        ],
        "source_embedding_overlay_sha256": migrated_binding[
            "source_overlay_sha256"
        ],
        "compact_contract_sha256": current_contract_sha256,
        "capacity_only_migration": {
            "receipt": file_record(receipt_path),
            "changed_contract_fields": changed_fields,
            "reused_source_token_rows": len(source_ids),
            "reinitialized_source_token_rows": 0,
            "nested_source_gold_sft": source_gold,
        },
    }


def _validate_union_gold_continuation_checkpoint(
    paths: Mapping[str, Path],
    *,
    binding: Mapping[str, Any],
    fit_contract: Mapping[str, Any],
    current_contract_sha256: str,
) -> dict[str, Any]:
    """Validate the supplemental-only gold continuation used by union2776."""

    expected_binding = _validate_checkpoint_binding(
        binding, paths, label="Qwen union supplemental-gold warmstart"
    )
    derivation = fit_contract.get("derivation")
    derivation_record = fit_contract.get("derivation_record")
    if not isinstance(derivation, Mapping) or not isinstance(
        derivation_record, Mapping
    ):
        raise ValueError("Qwen union has no supplemental derivation")
    provenance = load_object(paths["provenance"])
    loss_contract = provenance.get("loss_contract")
    stage_contract = provenance.get("stage_contract")
    if (
        provenance.get("schema") != "direct-compact-run-provenance-v1"
        or provenance.get("architecture")
        != "qwen-causal-compact-tokens-no-encoder"
        or provenance.get("decoder_adapter_sha256")
        != expected_binding["decoder_adapter_sha256"]
        or provenance.get("source_overlay_sha256")
        != expected_binding["source_overlay_sha256"]
        or provenance.get("contract_sha256") != current_contract_sha256
        or sha256_file(paths["contract"]) != current_contract_sha256
        or not isinstance(loss_contract, Mapping)
        or loss_contract.get("sequence_distribution_nll") is not False
        or loss_contract.get("primary_reduction")
        != "base_causal_lm_token_mean"
        or provenance.get("heldout_loaded_during_training") is not False
        or provenance.get("eval_file_sha256") is not None
        or provenance.get("eval_seal_sha256") is not None
        or provenance.get("eval_sealed_rows") is not None
        or provenance.get("eval_strategy") != "no"
        or not isinstance(stage_contract, Mapping)
        or stage_contract.get("sha256") != derivation_record.get("sha256")
        or Path(str(stage_contract.get("path") or "")).expanduser().resolve()
        != Path(str(derivation_record.get("path") or "")).expanduser().resolve()
    ):
        raise ValueError(
            "Qwen union warmstart is not the sealed supplemental-only "
            "gold continuation"
        )
    outputs = derivation.get("outputs")
    inputs = derivation.get("inputs")
    if not isinstance(outputs, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("Qwen union derivation has no sealed inputs/outputs")
    supplement_path, supplement_record = validate_file_record(
        outputs.get("supplement_compact"),
        label="Qwen union supplemental compact train",
    )
    supplement_seal_path, supplement_seal_record = validate_file_record(
        outputs.get("supplement_compact_seal"),
        label="Qwen union supplemental compact train seal",
    )
    validate_fit_seal(
        supplement_seal_path,
        dataset_record=supplement_record,
        contract_sha256=current_contract_sha256,
        expected_rows=QWEN_UNION_SUPPLEMENT_ROWS,
        label="Qwen union supplemental compact train seal",
    )
    if (
        provenance.get("train_file_sha256")
        != supplement_record["sha256"]
        or provenance.get("train_seal_sha256")
        != supplement_seal_record["sha256"]
        or provenance.get("train_sealed_rows")
        != QWEN_UNION_SUPPLEMENT_ROWS
    ):
        raise ValueError(
            "Qwen supplemental-gold checkpoint train binding failed"
        )
    legacy_path, legacy_record = validate_file_record(
        inputs.get("legacy_compact"),
        label="Qwen legacy gold compact train",
    )
    legacy_seal_path, legacy_seal_record = validate_file_record(
        inputs.get("legacy_compact_seal"),
        label="Qwen legacy gold compact train seal",
    )
    warmstart = provenance.get("warmstart_checkpoint")
    if not isinstance(warmstart, Mapping):
        raise ValueError(
            "Qwen supplemental-gold checkpoint lacks legacy warmstart"
        )
    legacy_root = Path(
        str(warmstart.get("path") or "")
    ).expanduser().resolve()
    legacy_paths = _checkpoint_paths(
        legacy_root, label="Qwen legacy gold warmstart"
    )
    legacy_gold = _validate_gold_sft_checkpoint(
        legacy_paths,
        binding=warmstart,
        compact_train_record=legacy_record,
        compact_train_seal_path=legacy_seal_path,
        compact_train_seal_record=legacy_seal_record,
        expected_contract_sha256=current_contract_sha256,
        label="Qwen legacy gold warmstart",
    )
    return {
        "path": str(paths["root"]),
        "run_provenance": file_record(paths["provenance"]),
        "decoder_adapter_sha256": expected_binding[
            "decoder_adapter_sha256"
        ],
        "source_embedding_overlay_sha256": expected_binding[
            "source_overlay_sha256"
        ],
        "compact_contract_sha256": current_contract_sha256,
        "supplemental_gold_continuation": {
            "derivation_manifest": dict(derivation_record),
            "train": supplement_record,
            "train_seal": supplement_seal_record,
            "rows": QWEN_UNION_SUPPLEMENT_ROWS,
            "historical_rows_replayed": 0,
            "heldout_loaded_during_training": False,
            "legacy_gold_warmstart": legacy_gold,
        },
    }


def _validate_qwen_sequence_student_checkpoint(
    checkpoint_path: str | Path,
    *,
    qwen_build_manifest: str | Path | None = None,
    inference_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless ``checkpoint_path`` is the sealed Qwen stage output.

    The training script itself has a generic direct-compact provenance schema,
    so the proof must join it to the Qwen sequence-KD build manifest and the
    audited Qwen teacher artifact that produced the checkpoint's exact train
    file.  Merely naming a directory "qwen" is deliberately insufficient.
    """

    root = Path(checkpoint_path).expanduser().resolve()
    adapter = root / "decoder_adapter"
    overlay = root / "source_embedding_overlay.pt"
    contract = root / "compact_contract.json"
    provenance_path = root / "run_provenance.json"
    required = (
        adapter / "adapter_config.json",
        overlay,
        contract,
        provenance_path,
    )
    for path in required:
        if not path.is_file():
            raise ValueError(f"Qwen student checkpoint is incomplete: {path}")

    provenance = load_object(provenance_path)
    if (
        provenance.get("schema") != "direct-compact-run-provenance-v1"
        or provenance.get("architecture")
        != "qwen-causal-compact-tokens-no-encoder"
    ):
        raise ValueError("student checkpoint is not a direct-compact checkpoint")
    adapter_sha = sha256_artifact(adapter)
    overlay_sha = sha256_file(overlay)
    contract_sha = sha256_file(contract)
    if provenance.get("decoder_adapter_sha256") != adapter_sha:
        raise ValueError("student adapter differs from checkpoint provenance")
    if provenance.get("source_overlay_sha256") != overlay_sha:
        raise ValueError("student source overlay differs from checkpoint provenance")
    if provenance.get("contract_sha256") != contract_sha:
        raise ValueError("student compact contract differs from checkpoint provenance")
    loss_contract = provenance.get("loss_contract")
    if (
        not isinstance(loss_contract, Mapping)
        or loss_contract.get("sequence_distribution_nll") is not True
        or loss_contract.get("primary_reduction")
        != "equal_weight_mean_of_eos_inclusive_per_sequence_nll_sums"
        or provenance.get("heldout_loaded_during_training") is not False
        or provenance.get("eval_file_sha256") is not None
        or provenance.get("eval_seal_sha256") is not None
        or provenance.get("eval_sealed_rows") is not None
        or provenance.get("eval_strategy") != "no"
    ):
        raise ValueError(
            "Qwen checkpoint lacks sequence-KD loss or touched heldout175"
        )

    build_path = (
        Path(qwen_build_manifest).expanduser().resolve()
        if qwen_build_manifest is not None
        else root.parent / "qwen_mc_sequence_train.build.json"
    )
    if not build_path.is_file():
        raise ValueError(f"Qwen sequence build manifest is missing: {build_path}")
    build = load_object(build_path)
    fit_contract = _validate_qwen_fit_contract(
        build,
        samples_per_task=QWEN_DRAWS_PER_TASK,
        label="Qwen sequence build",
    )
    qwen_task_rows = int(fit_contract["task_count"])
    qwen_teacher_draw_rows = int(fit_contract["expected_grid_rows"])
    qwen_sequence_output_rows = qwen_teacher_draw_rows
    if fit_contract["expanded"]:
        stage_contract = provenance.get("stage_contract")
        if (
            not isinstance(stage_contract, Mapping)
            or Path(str(stage_contract.get("path") or "")).expanduser().resolve()
            != build_path
            or stage_contract.get("sha256") != sha256_file(build_path)
        ):
            raise ValueError(
                "expanded Qwen sequence checkpoint is not bound to its "
                "union build manifest"
            )
    objective = build.get("objective")
    teacher_sampling = (
        objective.get("teacher_sampling")
        if isinstance(objective, Mapping)
        else None
    )
    counts = build.get("counts")
    gold_replay = build.get("gold_replay")
    if (
        build.get("schema") != QWEN_BUILD_SCHEMA
        or not isinstance(objective, Mapping)
        or objective.get("name") != "monte_carlo_sequence_forward_kl_nll"
        or objective.get("every_teacher_draw_emitted_exactly_once") is not True
        or objective.get("all_k8_draws_required_and_emitted") is not True
        or objective.get("parseability_filtering") is not False
        or objective.get("correctness_filtering") is not False
        or objective.get("gold_targets_mixed_into_sequence_objective")
        is not False
        or objective.get("target_transform") != "trim_outer_whitespace"
        or not isinstance(teacher_sampling, Mapping)
        or teacher_sampling.get("temperature") != 1.0
        or teacher_sampling.get("top_p") != 1.0
        or teacher_sampling.get("top_k") != 101
        or teacher_sampling.get("tempered") is not False
        or teacher_sampling.get("truncated") is not False
        or objective.get("dense_full_vocabulary_kl") is not False
        or not isinstance(counts, Mapping)
        or int(counts.get("teacher_draw_rows", -1))
        != qwen_teacher_draw_rows
        or int(counts.get("unique_teacher_candidate_ids", -1))
        != qwen_teacher_draw_rows
        or int(counts.get("gold_replay_rows", -1))
        != QWEN_GOLD_REPLAY_ROWS
        or int(counts.get("output_rows", -1))
        != qwen_sequence_output_rows
        or not isinstance(gold_replay, Mapping)
        or gold_replay.get("required_zero_for_sequence_only") is not True
        or float(gold_replay.get("requested_final_fraction", -1.0))
        != QWEN_GOLD_REPLAY_FRACTION
        or int(gold_replay.get("rows", -1)) != QWEN_GOLD_REPLAY_ROWS
        or float(gold_replay.get("realized_final_fraction", -1.0))
        != QWEN_GOLD_REPLAY_ROWS / qwen_sequence_output_rows
    ):
        raise ValueError("student was not produced from the sealed Qwen sequence stage")

    build_outputs = build.get("outputs")
    if not isinstance(build_outputs, Mapping):
        raise ValueError("Qwen build manifest has no sealed outputs")
    sequence_train_path, sequence_train_record = validate_file_record(
        build_outputs.get("dataset"),
        label="Qwen sequence train dataset",
    )
    sequence_train_seal_path, sequence_train_seal_record = validate_file_record(
        build_outputs.get("standard_direct_compact_seal"),
        label="Qwen sequence train seal",
    )
    sequence_schedule_path, sequence_schedule_record = validate_file_record(
        build_outputs.get("schedule"),
        label="Qwen sequence schedule",
    )
    train_path, train_record = sequence_train_path, sequence_train_record
    train_seal_path, train_seal_record = (
        sequence_train_seal_path,
        sequence_train_seal_record,
    )
    objective_mode = str(objective.get("objective_mode") or "")
    sparse_manifest_record: dict[str, Any] | None = None
    if objective_mode == "require_top5":
        sparse_manifest_path = (
            build_path.parent / "qwen_mc_sequence_plus_sparse_topk_tail.manifest.json"
        )
        sparse_train_path = (
            build_path.parent / "qwen_mc_sequence_plus_sparse_topk_tail.jsonl"
        )
        sparse_seal_path = (
            build_path.parent / "qwen_mc_sequence_plus_sparse_topk_tail.seal.json"
        )
        for path in (sparse_manifest_path, sparse_train_path, sparse_seal_path):
            if not path.is_file():
                raise ValueError(f"required Qwen sparse-KL artifact is missing: {path}")
        sparse = load_object(sparse_manifest_path)
        sparse_manifest_record = file_record(sparse_manifest_path)
        sequence_manifest_input = (
            (sparse.get("inputs") or {}).get("sequence_build_manifest") or {}
        )
        if (
            sparse.get("schema")
            != "direct-compact-sparse-topk-tail-manifest-v1"
            or sparse.get("objective") != "coarsened_topk_plus_tail_forward_kl"
            or sparse.get("sequence_monte_carlo_forward_kl_nll_primary") is not True
            or sparse.get("dense_full_vocabulary_kl") is not False
            or sparse.get("full_vocabulary_kd") is not False
            or sparse.get("global_provider_tokenizer_identity_claimed") is not False
            or sparse.get("target_transform")
            != "trim_trailing_outer_whitespace_on_provider_token_boundaries"
            or int(sparse.get("rows", -1))
            != qwen_sequence_output_rows
            or int(sparse.get("teacher_draw_rows", -1))
            != qwen_teacher_draw_rows
            or int(sparse.get("rows_with_sparse_auxiliary", 0)) <= 0
            or float(sparse.get("eligible_fraction", 0.0))
            < float(sparse.get("minimum_eligible_fraction", 1.0))
            or sequence_manifest_input.get("sha256") != sha256_file(build_path)
            or sparse.get("dataset_sha256") != sha256_file(sparse_train_path)
            or sparse.get("dataset_seal_sha256") != sha256_file(sparse_seal_path)
        ):
            raise ValueError("Qwen sparse top5+tail manifest contract failed")
        train_path, train_record = sparse_train_path, file_record(sparse_train_path)
        train_seal_path, train_seal_record = (
            sparse_seal_path,
            file_record(sparse_seal_path),
        )
    elif objective_mode != "sequence_only":
        raise ValueError("Qwen sequence build has an unsupported objective mode")
    sparse_provenance = provenance.get("sparse_topk_tail_auxiliary")
    if (
        (objective_mode == "require_top5")
        != isinstance(sparse_provenance, Mapping)
        or (
            isinstance(sparse_provenance, Mapping)
            and (
                sparse_provenance.get("dense_full_vocabulary_kl") is not False
                or sparse_provenance.get(
                    "sequence_monte_carlo_forward_kl_nll_primary"
                )
                is not True
                or sparse_provenance.get("manifest_sha256")
                != (
                    None
                    if sparse_manifest_record is None
                    else sparse_manifest_record["sha256"]
                )
            )
        )
    ):
        raise ValueError("Qwen checkpoint sparse auxiliary provenance is inconsistent")
    if provenance.get("train_file_sha256") != train_record["sha256"]:
        raise ValueError("checkpoint was not trained on its sealed Qwen objective dataset")
    if provenance.get("train_seal_sha256") != train_seal_record["sha256"]:
        raise ValueError("checkpoint was not trained with its sealed Qwen objective seal")

    build_inputs = build.get("inputs")
    if not isinstance(build_inputs, Mapping):
        raise ValueError("Qwen build manifest has no sealed inputs")
    if fit_contract["expanded"]:
        derivation_outputs = (
            (fit_contract.get("derivation") or {}).get("outputs") or {}
        )
        qwen_prompt = derivation_outputs.get("fit_prompt")
        qwen_prompt_manifest = derivation_outputs.get(
            "fit_prompt_manifest"
        )
        qwen_f2_contract: Any = None
    else:
        qwen_prompt = build_inputs.get("prompt_artifact")
        qwen_prompt_manifest = build_inputs.get("prompt_manifest")
        qwen_f2_contract = build_inputs.get("f2_prompt_contract")
    if (
        not isinstance(qwen_prompt, Mapping)
        or len(str(qwen_prompt.get("sha256") or "")) != 64
        or not isinstance(qwen_prompt_manifest, Mapping)
        or len(str(qwen_prompt_manifest.get("sha256") or "")) != 64
    ):
        raise ValueError("Qwen build has no sealed F2 prompt contract")
    qwen_prompt_path, qwen_prompt_record = validate_file_record(
        qwen_prompt,
        label="Qwen full multi-function F2 prompt",
    )
    qwen_prompt_manifest_path, qwen_prompt_manifest_record = (
        validate_file_record(
            qwen_prompt_manifest,
            label="Qwen full multi-function F2 prompt manifest",
        )
    )
    prompt_manifest_value = load_object(qwen_prompt_manifest_path)
    if fit_contract["expanded"]:
        qwen_f2_contract = prompt_manifest_value.get(
            "f2_prompt_contract"
        )
    if (
        not isinstance(qwen_f2_contract, Mapping)
        or qwen_f2_contract.get("representation_schema")
        != "lossless-semantic-f2"
        or len(str(qwen_f2_contract.get("system_prompt_sha256") or ""))
        != 64
    ):
        raise ValueError("Qwen build has no sealed F2 prompt contract")
    prompt_invariants = prompt_manifest_value.get("invariants")
    if (
        prompt_manifest_value.get("schema")
        != "verified-api-readable-compact-v2"
        or int(prompt_manifest_value.get("rows", -1)) != qwen_task_rows
        or (prompt_manifest_value.get("output") or {}).get("sha256")
        != qwen_prompt_record["sha256"]
        or not isinstance(prompt_invariants, Mapping)
        or prompt_invariants.get("all_user_functions_retained") is not True
        or prompt_invariants.get("all_external_symbols_retained") is not True
        or prompt_invariants.get("transfer_table_redundancy_proven") is not True
        or prompt_invariants.get("train_dev_representation_contract_identical")
        is not True
        or (
            prompt_manifest_value.get("f2_prompt_contract") or {}
        ).get("representation_schema")
        != F2_REPRESENTATION_SCHEMA
    ):
        raise ValueError(
            "Qwen student was not conditioned on the complete sealed "
            "multi-function F2 representation"
        )
    compact_train = build_inputs.get("compact_train")
    compact_train_path, compact_train_record = validate_file_record(
        compact_train,
        label="Qwen full multi-function compact train",
    )
    compact_train_seal_path, compact_train_seal_record = validate_file_record(
        build_inputs.get("compact_train_seal"),
        label="Qwen full multi-function compact train seal",
    )
    build_contract_path, build_contract_record = validate_file_record(
        build_inputs.get("contract"),
        label="Qwen current compact contract",
    )
    if (
        build_contract_record["sha256"] != contract_sha
        or sha256_file(build_contract_path) != sha256_file(contract)
    ):
        raise ValueError(
            "Qwen build/checkpoint compact contracts are not byte-identical"
        )
    validate_fit_seal(
        compact_train_seal_path,
        dataset_record=compact_train_record,
        contract_sha256=contract_sha,
        expected_rows=qwen_task_rows,
        label="Qwen full multi-function compact train seal",
    )
    compact_rows = load_jsonl(compact_train_path)
    compact_task_ids = [
        str(row.get("task_id") or row.get("id") or "") for row in compact_rows
    ]
    if (
        len(compact_rows) != qwen_task_rows
        or any(not task_id for task_id in compact_task_ids)
        or len(set(compact_task_ids)) != qwen_task_rows
        or (
            fit_contract["ordered_task_ids"] is not None
            and compact_task_ids != fit_contract["ordered_task_ids"]
        )
        or any(
            row.get("binary_multifunction_schema") != REPRESENTATION_SCHEMA
            for row in compact_rows
        )
        or (prompt_manifest_value.get("dataset") or {}).get("sha256")
        != compact_train_record["sha256"]
    ):
        raise ValueError(
            "Qwen compact train is not the complete sealed fit view"
        )

    sequence_rows = load_jsonl(sequence_train_path)
    sequence_schedule = load_jsonl(sequence_schedule_path)
    sequence_seal = load_object(sequence_train_seal_path)
    validate_fit_seal(
        sequence_train_seal_path,
        dataset_record=sequence_train_record,
        contract_sha256=contract_sha,
        expected_rows=qwen_sequence_output_rows,
        label="Qwen sequence train seal",
    )
    if train_seal_path != sequence_train_seal_path:
        validate_fit_seal(
            train_seal_path,
            dataset_record=train_record,
            contract_sha256=contract_sha,
            expected_rows=qwen_sequence_output_rows,
            label="Qwen selected objective train seal",
        )
    if (
        len(sequence_rows) != qwen_sequence_output_rows
        or len(sequence_schedule) != qwen_sequence_output_rows
        or sequence_seal.get("selected_role") != "fit"
        or int(sequence_seal.get("rows", -1))
        != qwen_sequence_output_rows
        or sequence_seal.get("output_sha256")
        != sequence_train_record["sha256"]
        or stable_sha256(sequence_schedule) != build.get("schedule_sha256")
    ):
        raise ValueError(
            "Qwen sequence dataset/seal/schedule is not the exact sealed "
            "complete teacher grid"
        )

    teacher_counts: Counter[str] = Counter()
    teacher_samples: dict[str, set[int]] = defaultdict(set)
    candidate_ids: set[str] = set()
    teacher_rows = 0
    gold_rows = 0
    for position, (scheduled, output_row) in enumerate(
        zip(sequence_schedule, sequence_rows, strict=True)
    ):
        task_id = str(scheduled.get("task_id") or "")
        base_index = scheduled.get("base_row_index")
        if (
            scheduled.get("schema") != QWEN_SCHEDULE_SCHEMA
            or int(scheduled.get("position", -1)) != position
            or isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or not 0 <= base_index < qwen_task_rows
            or task_id != compact_task_ids[base_index]
            or str(output_row.get("task_id") or output_row.get("id") or "")
            != task_id
            or scheduled.get("draw_weight") != 1.0
        ):
            raise ValueError(f"Qwen schedule row {position} join contract failed")
        if "compact_input_ids" in compact_rows[base_index] and (
            output_row.get("compact_input_ids")
            != compact_rows[base_index].get("compact_input_ids")
        ):
            raise ValueError(
                f"Qwen schedule row {position} changed compact conditioning"
            )
        kind = scheduled.get("kind")
        if kind == "teacher_draw":
            candidate_id = str(scheduled.get("candidate_id") or "")
            sample_index = scheduled.get("sample_index")
            if (
                not candidate_id
                or candidate_id in candidate_ids
                or isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or not 0 <= sample_index < QWEN_DRAWS_PER_TASK
            ):
                raise ValueError(
                    f"Qwen teacher schedule row {position} is malformed"
                )
            candidate_ids.add(candidate_id)
            teacher_counts[task_id] += 1
            teacher_samples[task_id].add(sample_index)
            teacher_rows += 1
        elif kind == "gold_replay":
            if (
                scheduled.get("candidate_id") is not None
                or scheduled.get("sample_index") is not None
            ):
                raise ValueError(
                    f"Qwen gold replay schedule row {position} is malformed"
                )
            gold_rows += 1
        else:
            raise ValueError(f"Qwen schedule row {position} has unknown kind")
    expected_samples = set(range(QWEN_DRAWS_PER_TASK))
    if (
        teacher_rows != qwen_teacher_draw_rows
        or gold_rows != QWEN_GOLD_REPLAY_ROWS
        or len(candidate_ids) != qwen_teacher_draw_rows
        or set(teacher_counts) != set(compact_task_ids)
        or any(
            teacher_counts[task_id] != QWEN_DRAWS_PER_TASK
            or teacher_samples[task_id] != expected_samples
            for task_id in compact_task_ids
        )
    ):
        raise ValueError(
            "Qwen schedule does not contain sample_index 0..7 exactly once "
            "for every sealed fit task"
        )
    if (
        int(provenance.get("train_sealed_rows", -1))
        != qwen_sequence_output_rows
    ):
        raise ValueError(
            "Qwen checkpoint provenance has the wrong sequence-stage row count"
        )
    gold_warmstart = provenance.get("warmstart_checkpoint")
    if not isinstance(gold_warmstart, Mapping):
        raise ValueError("Qwen sequence checkpoint has no gold-adaptation warmstart")
    gold_root = Path(str(gold_warmstart.get("path") or "")).expanduser().resolve()
    gold_paths = _checkpoint_paths(
        gold_root, label="Qwen gold-adaptation warmstart"
    )
    gold_provenance = load_object(gold_paths["provenance"])
    if fit_contract["expanded"]:
        gold_adaptation = _validate_union_gold_continuation_checkpoint(
            gold_paths,
            binding=gold_warmstart,
            fit_contract=fit_contract,
            current_contract_sha256=contract_sha,
        )
    elif gold_provenance.get("checkpoint_stage") == "contract-overlay-migration-only":
        gold_adaptation = _validate_capacity_migrated_gold_checkpoint(
            gold_paths,
            binding=gold_warmstart,
            compact_train_record=compact_train_record,
            current_train_seal_path=compact_train_seal_path,
            current_train_seal_record=compact_train_seal_record,
            current_contract_sha256=contract_sha,
        )
    else:
        gold_adaptation = _validate_gold_sft_checkpoint(
            gold_paths,
            binding=gold_warmstart,
            compact_train_record=compact_train_record,
            compact_train_seal_path=compact_train_seal_path,
            compact_train_seal_record=compact_train_seal_record,
            expected_contract_sha256=contract_sha,
            label="Qwen gold-adaptation warmstart",
        )
    if fit_contract["expanded"]:
        parent_audits: list[dict[str, Any]] = []
        parent_teacher_artifacts: list[dict[str, Any]] = []
        audited_teacher_rows = 0
        for index, parent_binding in enumerate(
            fit_contract["parent_builds"] or []
        ):
            parent_build_path, _parent_build_record = validate_file_record(
                parent_binding.get("build_manifest"),
                label=f"Qwen sequence union parent {index} build",
            )
            parent_build = load_object(parent_build_path)
            parent_inputs = parent_build.get("inputs")
            parent_task_count = int(parent_binding.get("task_count", -1))
            if (
                parent_build.get("schema") != QWEN_BUILD_SCHEMA
                or not isinstance(parent_inputs, Mapping)
                or int((parent_build.get("counts") or {}).get(
                    "teacher_draw_rows", -1
                ))
                != parent_task_count * QWEN_DRAWS_PER_TASK
            ):
                raise ValueError(
                    f"Qwen sequence union parent {index} build failed"
                )
            audit_path, audit_item = validate_file_record(
                parent_inputs.get("teacher_audit"),
                label=f"Qwen sequence union parent {index} audit",
            )
            teacher_path, teacher_item = validate_file_record(
                parent_inputs.get("teacher_parseable"),
                label=f"Qwen sequence union parent {index} teacher rows",
            )
            audit = load_object(audit_path)
            teacher_rows = load_jsonl(teacher_path)
            requested_models = {
                str(
                    (row.get("backend_identity") or {}).get(
                        "requested_model"
                    )
                    or ""
                )
                for row in audit.get("homogeneous_backend_shards") or []
                if isinstance(row, Mapping)
            }
            expected_parent_rows = (
                parent_task_count * QWEN_DRAWS_PER_TASK
            )
            if (
                audit.get("schema") != QWEN_AUDIT_SCHEMA
                or (audit.get("production_readiness") or {}).get(
                    "mc_sequence_forward_kl_nll"
                )
                is not True
                or (audit.get("capabilities") or {}).get(
                    "dense_full_vocabulary_kl"
                )
                is not False
                or int((audit.get("coverage") or {}).get("candidates", -1))
                != expected_parent_rows
                or len(teacher_rows) != expected_parent_rows
                or requested_models != {QWEN_STAGE_MODEL}
            ):
                raise ValueError(
                    f"Qwen sequence union parent {index} audit failed"
                )
            audited_teacher_rows += len(teacher_rows)
            parent_audits.append(audit_item)
            parent_teacher_artifacts.append(teacher_item)
        if audited_teacher_rows != qwen_teacher_draw_rows:
            raise ValueError("Qwen union parent teacher accounting failed")
        audit_record: Any = {
            "schema": "qwen-union-parent-audits-v1",
            "artifacts": parent_audits,
        }
        teacher_parseable_record: Any = {
            "schema": "qwen-union-parent-teacher-artifacts-v1",
            "artifacts": parent_teacher_artifacts,
            "rows": audited_teacher_rows,
        }
    else:
        audit_path, audit_record = validate_file_record(
            build_inputs.get("teacher_audit"),
            label="Qwen teacher audit",
        )
        teacher_parseable_path, teacher_parseable_record = (
            validate_file_record(
                build_inputs.get("teacher_parseable"),
                label="Qwen complete teacher sequence artifact",
            )
        )
        teacher_parseable_rows = load_jsonl(teacher_parseable_path)
        audit = load_object(audit_path)
        requested_models = {
            str(
                (row.get("backend_identity") or {}).get("requested_model")
                or ""
            )
            for row in audit.get("homogeneous_backend_shards") or []
            if isinstance(row, Mapping)
        }
        if (
            audit.get("schema") != QWEN_AUDIT_SCHEMA
            or audit.get("objective_mode") != objective_mode
            or (audit.get("production_readiness") or {}).get(
                "mc_sequence_forward_kl_nll"
            )
            is not True
            or (audit.get("capabilities") or {}).get(
                "dense_full_vocabulary_kl"
            )
            is not False
            or int((audit.get("coverage") or {}).get("candidates", -1))
            != qwen_teacher_draw_rows
            or int((audit.get("coverage") or {}).get(
                "sequence_candidates", -1
            ))
            != qwen_teacher_draw_rows
            or len(teacher_parseable_rows) != qwen_teacher_draw_rows
            or requested_models != {QWEN_STAGE_MODEL}
        ):
            raise ValueError(
                "checkpoint Qwen teacher audit is missing, unready, or from "
                f"a model other than {QWEN_STAGE_MODEL}"
            )

    if inference_provenance is not None:
        if inference_provenance.get("schema") != "direct-compact-inference-v1":
            raise ValueError("student prediction provenance has an unknown schema")
        adapter_value = inference_provenance.get("decoder_adapter")
        if not isinstance(adapter_value, str) or (
            Path(adapter_value).expanduser().resolve() != adapter
        ):
            raise ValueError(
                "student predictions were not generated from the Qwen checkpoint"
            )
        if inference_provenance.get("decoder_adapter_sha256") != adapter_sha:
            raise ValueError("prediction adapter hash differs from Qwen checkpoint")
        if inference_provenance.get("source_overlay_sha256") != overlay_sha:
            raise ValueError("prediction overlay hash differs from Qwen checkpoint")
        if inference_provenance.get("selected_role") != "fit":
            raise ValueError("RS failures must be measured on the fit split")

    return {
        "stage": "qwen3.8-max-preview-mc-sequence-forward-kl-nll",
        "qwen_objective_mode": str(
            (build.get("objective") or {}).get("objective_mode") or ""
        ),
        "checkpoint": {
            "path": str(root),
            "run_provenance": file_record(provenance_path),
            "decoder_adapter_sha256": adapter_sha,
            "source_embedding_overlay_sha256": overlay_sha,
            "compact_contract_sha256": contract_sha,
        },
        "qwen_build_manifest": file_record(build_path),
        "qwen_sparse_topk_tail_manifest": sparse_manifest_record,
        "qwen_teacher_audit": audit_record,
        "qwen_teacher_sequence_artifact": teacher_parseable_record,
        "qwen_sequence_schedule": sequence_schedule_record,
        "qwen_train_dataset": train_record,
        "qwen_train_seal": train_seal_record,
        "qwen_train_paths": {
            "dataset": str(train_path),
            "seal": str(train_seal_path),
        },
        "qwen_gold_adaptation": {
            **gold_adaptation,
            "heldout_loaded_during_training": False,
        },
        "qwen_prompt_artifact": qwen_prompt_record,
        "qwen_prompt_manifest": qwen_prompt_manifest_record,
        "qwen_f2_prompt_contract": dict(qwen_f2_contract),
        "fit_task_count": qwen_task_rows,
        "fit_ordered_task_ids_sha256": (
            fit_contract["ordered_task_ids_sha256"]
            or stable_sha256(compact_task_ids)
        ),
        "heldout_task_count": fit_contract["heldout_task_count"],
        "heldout_intersection_count": fit_contract[
            "heldout_intersection_count"
        ],
        "qwen_union_derivation": fit_contract["derivation_record"],
        "requested_teacher_model": QWEN_STAGE_MODEL,
    }


def _cot_target_text(row: Mapping[str, Any], *, position: int) -> str:
    fields = ("supervised_target", "dart_source", "source")
    values = [row[field] for field in fields if field in row]
    if (
        not values
        or any(not isinstance(value, str) or not value for value in values)
        or any(value != values[0] for value in values[1:])
    ):
        raise ValueError(
            f"Qwen CoT output row {position} has no single byte-exact target"
        )
    return str(values[0])


def _validate_qwen_cot_student_checkpoint(
    checkpoint_path: str | Path,
    *,
    qwen_sequence_build_manifest: str | Path | None,
    inference_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the hard-SFT CoT stage and its complete sequence-KL parent."""

    root = Path(checkpoint_path).expanduser().resolve()
    paths = _checkpoint_paths(root, label="Qwen CoT student checkpoint")
    provenance = load_object(paths["provenance"])
    adapter_sha = sha256_artifact(paths["adapter"])
    overlay_sha = sha256_file(paths["overlay"])
    contract_sha = sha256_file(paths["contract"])
    contract_value = load_object(paths["contract"])
    loss_contract = provenance.get("loss_contract")
    if (
        provenance.get("schema") != "direct-compact-run-provenance-v1"
        or provenance.get("architecture")
        != "qwen-causal-compact-tokens-no-encoder"
        or provenance.get("decoder_adapter_sha256") != adapter_sha
        or provenance.get("source_overlay_sha256") != overlay_sha
        or provenance.get("contract_sha256") != contract_sha
        or not isinstance(loss_contract, Mapping)
        or loss_contract.get("sequence_distribution_nll") is not False
        or loss_contract.get("sequence_target_suffix_logits_only") is not False
        or loss_contract.get("primary_reduction") != "base_causal_lm_token_mean"
        or provenance.get("heldout_loaded_during_training") is not False
        or provenance.get("eval_file_sha256") is not None
        or provenance.get("eval_seal_sha256") is not None
        or provenance.get("eval_sealed_rows") is not None
        or provenance.get("eval_strategy") != "no"
        or provenance.get("sparse_topk_tail_auxiliary") is not None
    ):
        raise ValueError(
            "Qwen CoT checkpoint is not a token-mean, train-only hard-SFT stage"
        )
    if (
        contract_value.get("max_target_tokens")
        != QWEN_LONG_MAX_TARGET_TOKENS
        or contract_value.get("max_total_tokens")
        != QWEN_LONG_MAX_TOTAL_TOKENS
    ):
        raise ValueError("Qwen CoT checkpoint does not use the sealed 24K contract")

    cot_build_path = root.parent / "qwen_cot_sft_train.build.json"
    if not cot_build_path.is_file():
        raise ValueError(f"Qwen CoT build manifest is missing: {cot_build_path}")
    cot_build = load_object(cot_build_path)
    fit_contract = _validate_qwen_fit_contract(
        cot_build,
        samples_per_task=2,
        label="Qwen CoT build",
    )
    qwen_task_rows = int(fit_contract["task_count"])
    qwen_cot_output_rows = int(fit_contract["expected_grid_rows"])
    stage_contract = provenance.get("stage_contract")
    if (
        not isinstance(stage_contract, Mapping)
        or Path(str(stage_contract.get("path") or "")).expanduser().resolve()
        != cot_build_path
        or stage_contract.get("sha256") != sha256_file(cot_build_path)
        or int(stage_contract.get("size_bytes", -1))
        != cot_build_path.stat().st_size
    ):
        raise ValueError(
            "Qwen CoT checkpoint is not bound to its exact build manifest"
        )
    objective = cot_build.get("objective")
    if (
        cot_build.get("schema") != QWEN_COT_BUILD_SCHEMA
        or cot_build.get("build_completed") is not True
        or not isinstance(objective, Mapping)
        or objective.get("name") != "qwen_cot_hard_sft"
        or objective.get("ordinary_hard_sft") is not True
        or objective.get("dense_token_kl") is not False
        or objective.get("sequence_forward_kl") is not False
        or objective.get("reasoning_logprobs_available") is not False
        or objective.get("pure_sequence_kl_artifact_modified") is not False
        or objective.get("direct_prompt_mode") != QWEN_COT_PROMPT_MODE
        or objective.get("target_template")
        != (
            "<think>\\n + raw_reasoning_content + "
            "\\n</think>\\n + raw_final_content"
        )
        or objective.get("target_transform")
        != "none_byte_exact_provider_strings"
        or objective.get("samples_per_task") != 2
        or objective.get("selected_sample_indices") != [0, 1]
        or objective.get("selection_depends_only_on")
        != ["task_id", "sample_index"]
        or objective.get("correctness_filtering") is not False
        or objective.get("compile_filtering") is not False
        or objective.get("parseability_filtering") is not False
        or objective.get("confidence_filtering") is not False
        or objective.get("logprob_filtering") is not False
        or objective.get("empty_reasoning_filtering") is not False
        or objective.get("resampling") is not False
        or objective.get("provider_calls") is not False
    ):
        raise ValueError("Qwen CoT build objective contract failed")

    cot_inputs = cot_build.get("inputs")
    if not isinstance(cot_inputs, Mapping):
        raise ValueError("Qwen CoT build has no sealed inputs")
    input_records: dict[str, tuple[Path, dict[str, Any]]] = {}
    required_input_records = [
        ("compact_train", "Qwen CoT compact train"),
        ("compact_train_seal", "Qwen CoT compact train seal"),
        ("contract", "Qwen CoT 24K contract"),
        ("student_tokenizer", "Qwen CoT student tokenizer"),
    ]
    if not fit_contract["expanded"]:
        required_input_records.extend(
            [
                ("prompt_artifact", "Qwen CoT F2 prompt artifact"),
                ("prompt_manifest", "Qwen CoT F2 prompt manifest"),
                ("teacher_journal", "Qwen CoT teacher journal"),
                (
                    "teacher_journal_chain_head",
                    "Qwen CoT journal chain head",
                ),
                ("teacher_audit", "Qwen CoT teacher audit"),
            ]
        )
    for key, label in required_input_records:
        input_records[key] = validate_file_record(
            cot_inputs.get(key),
            label=label,
        )
    if input_records["contract"][1]["sha256"] != contract_sha:
        raise ValueError("Qwen CoT input/checkpoint contracts are not byte-identical")
    native_think_tokens = cot_inputs.get("native_think_tokens")
    if not fit_contract["expanded"] and native_think_tokens != {
        "open_token_id": QWEN_COT_THINK_OPEN_ID,
        "close_token_id": QWEN_COT_THINK_CLOSE_ID,
    }:
        raise ValueError("Qwen CoT build lacks the native Qwen3 think token IDs")
    f2_contract = cot_inputs.get("f2_prompt_contract")
    if fit_contract["expanded"]:
        derivation_outputs = (
            (fit_contract.get("derivation") or {}).get("outputs") or {}
        )
        prompt_manifest_path, _prompt_manifest_record = (
            validate_file_record(
                derivation_outputs.get("fit_prompt_manifest"),
                label="Qwen CoT union fit prompt manifest",
            )
        )
        f2_contract = load_object(prompt_manifest_path).get(
            "f2_prompt_contract"
        )
    if (
        not isinstance(f2_contract, Mapping)
        or f2_contract.get("representation_schema") != F2_REPRESENTATION_SCHEMA
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(f2_contract.get("system_prompt_sha256") or ""),
        )
    ):
        raise ValueError("Qwen CoT build lacks the sealed F2 prompt contract")

    outputs = cot_build.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Qwen CoT build has no sealed outputs")
    cot_train_path, cot_train_record = validate_file_record(
        outputs.get("dataset"),
        label="Qwen CoT hard-SFT train dataset",
    )
    cot_seal_path, cot_seal_record = validate_file_record(
        outputs.get("standard_direct_compact_seal"),
        label="Qwen CoT hard-SFT train seal",
    )
    cot_schedule_path, cot_schedule_record = validate_file_record(
        outputs.get("schedule"),
        label="Qwen CoT hard-SFT schedule",
    )
    validate_fit_seal(
        cot_seal_path,
        dataset_record=cot_train_record,
        contract_sha256=contract_sha,
        expected_rows=qwen_cot_output_rows,
        label="Qwen CoT hard-SFT train seal",
    )
    counts = cot_build.get("counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("tasks") != qwen_task_rows
        or counts.get("rows") != qwen_cot_output_rows
        or counts.get("rows_per_task") != 2
        or counts.get("unique_candidate_ids") != qwen_cot_output_rows
        or provenance.get("train_file_sha256") != cot_train_record["sha256"]
        or provenance.get("train_seal_sha256") != cot_seal_record["sha256"]
        or provenance.get("train_sealed_rows") != qwen_cot_output_rows
    ):
        raise ValueError(
            "Qwen CoT checkpoint is not bound to the exact sealed corpus"
        )

    compact_train_path = input_records["compact_train"][0]
    compact_train_record = input_records["compact_train"][1]
    compact_seal_path = input_records["compact_train_seal"][0]
    compact_rows = load_jsonl(compact_train_path)
    validate_fit_seal(
        compact_seal_path,
        dataset_record=compact_train_record,
        contract_sha256=contract_sha,
        expected_rows=qwen_task_rows,
        label="Qwen CoT source compact train seal",
    )
    compact_task_ids = [
        str(row.get("task_id") or row.get("id") or "") for row in compact_rows
    ]
    if (
        len(compact_rows) != qwen_task_rows
        or any(not task_id for task_id in compact_task_ids)
        or len(set(compact_task_ids)) != qwen_task_rows
        or (
            fit_contract["ordered_task_ids"] is not None
            and compact_task_ids != fit_contract["ordered_task_ids"]
        )
    ):
        raise ValueError("Qwen CoT compact input is not the complete fit set")

    cot_rows = load_jsonl(cot_train_path)
    cot_schedule = load_jsonl(cot_schedule_path)
    if (
        len(cot_rows) != qwen_cot_output_rows
        or len(cot_schedule) != qwen_cot_output_rows
        or stable_sha256(cot_schedule) != cot_build.get("schedule_sha256")
    ):
        raise ValueError("Qwen CoT output/schedule completeness contract failed")
    samples_by_task: dict[str, set[int]] = defaultdict(set)
    candidate_ids: set[str] = set()
    observed_empty_reasoning = 0
    for position, (scheduled, output_row) in enumerate(
        zip(cot_schedule, cot_rows, strict=True)
    ):
        base_index = scheduled.get("base_row_index")
        sample_index = scheduled.get("sample_index")
        task_id = str(scheduled.get("task_id") or "")
        candidate_id = str(scheduled.get("candidate_id") or "")
        if (
            scheduled.get("schema") != QWEN_COT_SCHEDULE_SCHEMA
            or scheduled.get("position") != position
            or isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or not 0 <= base_index < qwen_task_rows
            or task_id != compact_task_ids[base_index]
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index not in (0, 1)
            or not candidate_id
            or candidate_id in candidate_ids
            or str(output_row.get("task_id") or output_row.get("id") or "")
            != task_id
            or output_row.get("direct_prompt_mode") != QWEN_COT_PROMPT_MODE
            or scheduled.get("selection_rule")
            != "sealed_sample_index_in_[0,1]"
            or scheduled.get("selected_without_outcome_inspection") is not True
        ):
            raise ValueError(f"Qwen CoT schedule row {position} join contract failed")
        compact_ids = compact_rows[base_index].get("compact_input_ids")
        if isinstance(compact_ids, list):
            expected_compact_hash = stable_sha256(
                [int(value) for value in compact_ids]
            )
            if (
                output_row.get("compact_input_ids") != compact_ids
                or scheduled.get("compact_ids_sha256") != expected_compact_hash
            ):
                raise ValueError(
                    f"Qwen CoT schedule row {position} changed compact conditioning"
                )
        target = _cot_target_text(output_row, position=position)
        target_sha = hashlib.sha256(target.encode("utf-8")).hexdigest()
        evidence = scheduled.get("target_length_evidence")
        if (
            not target.startswith("<think>\n")
            or "\n</think>\n" not in target
            or scheduled.get("cot_target_sha256") != target_sha
            or not isinstance(evidence, Mapping)
            or evidence.get("sequence_target_sha256") != target_sha
            or evidence.get("max_target_tokens")
            != QWEN_LONG_MAX_TARGET_TOKENS
            or evidence.get("max_total_tokens") != QWEN_LONG_MAX_TOTAL_TOKENS
            or evidence.get("within_contract") is not True
            or evidence.get("within_total_contract") is not True
            or evidence.get("truncated") is not False
        ):
            raise ValueError(
                f"Qwen CoT schedule row {position} target/length contract failed"
            )
        for hash_field in (
            "reasoning_content_sha256",
            "raw_final_content_sha256",
        ):
            if not re.fullmatch(
                r"[0-9a-f]{64}", str(scheduled.get(hash_field) or "")
            ):
                raise ValueError(
                    f"Qwen CoT schedule row {position} has an invalid provider hash"
                )
        reasoning_empty = scheduled.get("reasoning_content_empty")
        if not isinstance(reasoning_empty, bool):
            raise ValueError(
                f"Qwen CoT schedule row {position} lacks reasoning accounting"
            )
        observed_empty_reasoning += int(reasoning_empty)
        samples_by_task[task_id].add(sample_index)
        candidate_ids.add(candidate_id)
    if (
        set(samples_by_task) != set(compact_task_ids)
        or any(samples != {0, 1} for samples in samples_by_task.values())
        or len(candidate_ids) != qwen_cot_output_rows
    ):
        raise ValueError(
            "Qwen CoT schedule does not contain slots 0 and 1 exactly once "
            "for every sealed fit task"
        )

    coverage = cot_build.get("coverage_gate")
    if not isinstance(coverage, Mapping):
        raise ValueError("Qwen CoT build has no coverage gate")
    minimum_fraction = coverage.get("minimum_nonempty_reasoning_fraction")
    if fit_contract["expanded"]:
        parent_minimums: list[float] = []
        for index, parent in enumerate(fit_contract["parent_builds"] or []):
            parent_path, _parent_record = validate_file_record(
                parent.get("build_manifest"),
                label=f"Qwen CoT union parent {index} build",
            )
            parent_coverage = (
                load_object(parent_path).get("coverage_gate") or {}
            )
            value = parent_coverage.get(
                "minimum_nonempty_reasoning_fraction"
            )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(
                    f"Qwen CoT union parent {index} coverage floor failed"
                )
            parent_minimums.append(float(value))
        minimum_fraction = min(parent_minimums)
    nonempty_rows = qwen_cot_output_rows - observed_empty_reasoning
    nonempty_fraction = nonempty_rows / qwen_cot_output_rows
    empty_diagnostics = coverage.get("empty_reasoning_diagnostics")
    overflow_diagnostics = coverage.get("overflow_diagnostics")
    pilot_prior = coverage.get("pilot_prior")
    if (
        coverage.get("passed") is not True
        or coverage.get("expected_tasks") != qwen_task_rows
        or coverage.get("selected_tasks") != qwen_task_rows
        or coverage.get("expected_rows") != qwen_cot_output_rows
        or coverage.get("selected_rows") != qwen_cot_output_rows
        or coverage.get("exact_kcot_coverage_fraction") != 1.0
        or coverage.get("nonempty_reasoning_rows") != nonempty_rows
        or coverage.get("empty_reasoning_rows") != observed_empty_reasoning
        or coverage.get("nonempty_reasoning_fraction") != nonempty_fraction
        or not 0.0 < float(minimum_fraction) <= nonempty_fraction
        or not isinstance(empty_diagnostics, list)
        or len(empty_diagnostics) != observed_empty_reasoning
        or coverage.get("max_target_tokens")
        != QWEN_LONG_MAX_TARGET_TOKENS
        or coverage.get("max_total_tokens") != QWEN_LONG_MAX_TOTAL_TOKENS
        or coverage.get("overflow_rows") != 0
        or overflow_diagnostics != []
        or coverage.get("target_length_evidence_sha256")
        != stable_sha256(
            [row.get("target_length_evidence") for row in cot_schedule]
        )
        or counts.get("empty_reasoning_rows_retained")
        != observed_empty_reasoning
    ):
        raise ValueError("Qwen CoT coverage/24K target-length gate failed")
    if not fit_contract["expanded"] and (
        coverage.get("empty_rows_retained_if_gate_passes") is not True
        or coverage.get("overflow_rows_retained_or_replaced") is not False
        or coverage.get("overflow_policy")
        != "abort_build_without_filtering_or_resampling"
        or not isinstance(pilot_prior, Mapping)
        or pilot_prior.get("selected_rows") != 128
        or pilot_prior.get("nonempty_reasoning_rows") != 128
        or pilot_prior.get("nonempty_reasoning_fraction") != 1.0
    ):
        raise ValueError("Qwen CoT legacy pilot coverage gate failed")

    sequence_build_path = (
        Path(qwen_sequence_build_manifest).expanduser().resolve()
        if qwen_sequence_build_manifest is not None
        else root.parent / "qwen_mc_sequence_train.build.json"
    )
    if not sequence_build_path.is_file():
        raise ValueError(
            f"nested Qwen sequence build manifest is missing: {sequence_build_path}"
        )
    sequence_build = load_object(sequence_build_path)
    sequence_inputs = sequence_build.get("inputs")
    if not isinstance(sequence_inputs, Mapping):
        raise ValueError("nested Qwen sequence build has no sealed inputs")
    shared_input_keys = [
        "compact_train",
        "compact_train_seal",
        "contract",
        "student_tokenizer",
    ]
    if fit_contract["expanded"]:
        shared_input_keys.append("union_derivation")
        sequence_fit_contract = _validate_qwen_fit_contract(
            sequence_build,
            samples_per_task=QWEN_DRAWS_PER_TASK,
            label="nested Qwen sequence build",
        )
        if (
            sequence_fit_contract["ordered_task_ids_sha256"]
            != fit_contract["ordered_task_ids_sha256"]
            or sequence_fit_contract["derivation_record"]
            != fit_contract["derivation_record"]
        ):
            raise ValueError(
                "Qwen CoT/sequence union membership commitments differ"
            )
    else:
        shared_input_keys.extend(
            [
                "prompt_artifact",
                "prompt_manifest",
                "f2_prompt_contract",
                "teacher_journal",
                "teacher_audit",
            ]
        )
    for key in shared_input_keys:
        if cot_inputs.get(key) != sequence_inputs.get(key):
            raise ValueError(
                f"Qwen CoT input {key} differs from its sequence-KL parent"
            )

    warmstart = provenance.get("warmstart_checkpoint")
    if not isinstance(warmstart, Mapping):
        raise ValueError("Qwen CoT checkpoint has no sequence-KL warmstart")
    sequence_root = (
        Path(str(warmstart.get("path") or "")).expanduser().resolve()
    )
    if sequence_root == root:
        raise ValueError("Qwen CoT checkpoint cannot warmstart from itself")
    sequence_paths = _checkpoint_paths(
        sequence_root,
        label="nested Qwen sequence-KL warmstart",
    )
    _validate_checkpoint_binding(
        warmstart,
        sequence_paths,
        label="nested Qwen sequence-KL warmstart",
    )
    sequence_provenance = load_object(sequence_paths["provenance"])
    sequence_loss = sequence_provenance.get("loss_contract")
    if (
        not isinstance(sequence_loss, Mapping)
        or sequence_loss.get("sequence_distribution_nll") is not True
    ):
        raise ValueError("Qwen CoT warmstart is not the sequence-forward-KL stage")
    nested_sequence = validate_qwen_student_checkpoint(
        sequence_root,
        qwen_build_manifest=sequence_build_path,
    )
    if (
        nested_sequence.get("stage")
        != "qwen3.8-max-preview-mc-sequence-forward-kl-nll"
        or (nested_sequence.get("checkpoint") or {}).get(
            "compact_contract_sha256"
        )
        != contract_sha
    ):
        raise ValueError("Qwen CoT nested sequence-forward-KL validation failed")

    if inference_provenance is not None:
        if inference_provenance.get("schema") != "direct-compact-inference-v1":
            raise ValueError("student prediction provenance has an unknown schema")
        adapter_value = inference_provenance.get("decoder_adapter")
        if not isinstance(adapter_value, str) or (
            Path(adapter_value).expanduser().resolve() != paths["adapter"]
        ):
            raise ValueError(
                "student predictions were not generated from the Qwen checkpoint"
            )
        if inference_provenance.get("decoder_adapter_sha256") != adapter_sha:
            raise ValueError("prediction adapter hash differs from Qwen checkpoint")
        if inference_provenance.get("source_overlay_sha256") != overlay_sha:
            raise ValueError("prediction overlay hash differs from Qwen checkpoint")
        if inference_provenance.get("selected_role") != "fit":
            raise ValueError("RS failures must be measured on the fit split")

    return {
        **nested_sequence,
        "stage": (
            "qwen3.8-max-preview-mc-sequence-forward-kl-nll-"
            "plus-cot-hard-sft"
        ),
        "checkpoint": {
            "path": str(root),
            "run_provenance": file_record(paths["provenance"]),
            "decoder_adapter_sha256": adapter_sha,
            "source_embedding_overlay_sha256": overlay_sha,
            "compact_contract_sha256": contract_sha,
        },
        "qwen_cot_build_manifest": file_record(cot_build_path),
        "qwen_cot_train_dataset": cot_train_record,
        "qwen_cot_train_seal": cot_seal_record,
        "qwen_cot_schedule": cot_schedule_record,
        "qwen_cot_coverage_gate": dict(coverage),
        "qwen_train_dataset": cot_train_record,
        "qwen_train_seal": cot_seal_record,
        "qwen_train_paths": {
            "dataset": str(cot_train_path),
            "seal": str(cot_seal_path),
        },
        "qwen_sequence_stage": nested_sequence,
        "qwen_sequence_warmstart": nested_sequence["checkpoint"],
    }


def validate_qwen_student_checkpoint(
    checkpoint_path: str | Path,
    *,
    qwen_build_manifest: str | Path | None = None,
    inference_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate either the sequence-KL stage or its final CoT hard-SFT child.

    ``qwen_build_manifest`` continues to mean the sealed sequence-forward-KL
    manifest so existing launchers remain byte-for-byte compatible.  A
    token-mean checkpoint must additionally carry a sibling
    ``qwen_cot_sft_train.build.json`` and an exact warmstart binding to the
    recursively validated sequence checkpoint.
    """

    root = Path(checkpoint_path).expanduser().resolve()
    paths = _checkpoint_paths(root, label="Qwen student checkpoint")
    provenance = load_object(paths["provenance"])
    loss_contract = provenance.get("loss_contract")
    sequence_distribution_nll = (
        loss_contract.get("sequence_distribution_nll")
        if isinstance(loss_contract, Mapping)
        else None
    )
    if sequence_distribution_nll is True:
        return _validate_qwen_sequence_student_checkpoint(
            root,
            qwen_build_manifest=qwen_build_manifest,
            inference_provenance=inference_provenance,
        )
    if (
        sequence_distribution_nll is False
        and isinstance(loss_contract, Mapping)
        and loss_contract.get("primary_reduction") == "base_causal_lm_token_mean"
    ):
        return _validate_qwen_cot_student_checkpoint(
            root,
            qwen_sequence_build_manifest=qwen_build_manifest,
            inference_provenance=inference_provenance,
        )
    raise ValueError(
        "Qwen student checkpoint has neither the sequence-KL nor CoT-SFT loss"
    )


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}.{threading.get_ident()}"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def append(self, row: Mapping[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self.lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


def ensure_run_contract(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected_payload = dict(payload)
    expected_hash = stable_sha256(expected_payload)
    if path.is_file():
        observed = load_object(path)
        if (
            observed.get("schema") != RUN_CONTRACT_SCHEMA
            or observed.get("payload_sha256") != expected_hash
            or observed.get("payload") != expected_payload
        ):
            raise ValueError(
                "resume refused: immutable OpenAI RS run contract changed"
            )
        return observed
    value = {
        "schema": RUN_CONTRACT_SCHEMA,
        "created_at": utc_now(),
        "payload_sha256": expected_hash,
        "payload": expected_payload,
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return value


def validate_openai_base_url(value: str) -> str:
    endpoint = str(value or "").strip().rstrip("/")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.openai.com"
        or parsed.port not in {None, 443}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "this launcher is sealed to the official OpenAI endpoint "
            "https://api.openai.com/v1"
        )
    return endpoint


def run_sample_major_bounded(
    *,
    failed_ids: Sequence[str],
    samples_per_task: int,
    terminal_slots: set[tuple[str, int]],
    verified_task_ids: set[str],
    minimum_verified_tasks: int,
    workers: int,
    harvest: Callable[[tuple[str, int]], Mapping[str, Any]],
    progress: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run bounded task-wide sampling rounds and stop at the coverage target.

    At most ``workers`` requests are in flight.  Round 0 is offered once for
    every failed task before any round-1 slot is scheduled.  Once coverage is
    reached, the already-running bounded window is drained but no new API call
    is submitted.
    """

    if workers <= 0 or samples_per_task <= 0 or minimum_verified_tasks <= 0:
        raise ValueError("scheduler counts must be positive")
    completed_now = 0
    scheduled_now = 0
    stop_reason = "exhausted_configured_slots"

    if len(verified_task_ids) >= minimum_verified_tasks:
        return {
            "completed_now": 0,
            "scheduled_now": 0,
            "stop_reason": "coverage_target_already_met",
            "verified_task_ids": set(verified_task_ids),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for sample_index in range(samples_per_task):
            if len(verified_task_ids) >= minimum_verified_tasks:
                stop_reason = "coverage_target_reached"
                break
            jobs = deque(
                (task_id, sample_index)
                for task_id in failed_ids
                if (task_id, sample_index) not in terminal_slots
            )
            in_flight: dict[
                concurrent.futures.Future[Mapping[str, Any]], tuple[str, int]
            ] = {}

            def fill_window() -> None:
                nonlocal scheduled_now
                while (
                    jobs
                    and len(in_flight) < workers
                    and len(verified_task_ids) < minimum_verified_tasks
                ):
                    job = jobs.popleft()
                    in_flight[pool.submit(harvest, job)] = job
                    scheduled_now += 1

            fill_window()
            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    job = in_flight.pop(future)
                    row = future.result()
                    completed_now += 1
                    if row.get("terminal") is True:
                        terminal_slots.add(job)
                        if row.get("passed") is True:
                            verified_task_ids.add(job[0])
                    if progress is not None:
                        progress(
                            completed_now,
                            scheduled_now,
                            sample_index,
                            len(verified_task_ids),
                        )
                # If the target was reached, drain only this bounded window.
                # No second wave is submitted, limiting overshoot to workers-1.
                fill_window()
            if len(verified_task_ids) >= minimum_verified_tasks:
                stop_reason = "coverage_target_reached"
                break

    return {
        "completed_now": completed_now,
        "scheduled_now": scheduled_now,
        "stop_reason": stop_reason,
        "verified_task_ids": set(verified_task_ids),
    }


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"malformed environment line {path}:{line_number}")
        key, value = stripped.split("=", 1)
        os.environ.setdefault(
            key.strip(), value.strip().strip('"').strip("'")
        )


def extract_code(text: str) -> str:
    result = str(text or "").strip()
    fenced = re.findall(
        r"```(?:dart)?\s*\n?(.*?)```",
        result,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        result = max(fenced, key=len).strip()
    result = re.sub(
        r"^\s*(?:Dart(?:\s+source)?|Answer)\s*:\s*",
        "",
        result,
        flags=re.IGNORECASE,
    )
    return result.strip()


def response_text(raw: Mapping[str, Any], sdk_response: Any) -> str:
    direct = str(getattr(sdk_response, "output_text", "") or "").strip()
    if direct:
        return direct
    pieces: list[str] = []
    for output in raw.get("output") or []:
        if not isinstance(output, Mapping):
            continue
        for content in output.get("content") or []:
            if not isinstance(content, Mapping):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "\n".join(pieces).strip()


def object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
        if isinstance(result, dict):
            return result
    if hasattr(value, "dict"):
        result = value.dict()
        if isinstance(result, dict):
            return result
    raise ValueError("SDK response cannot be serialized")


def validate_serialized_inputs(
    serialized_path: Path,
    train_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    manifest = load_object(manifest_path)
    if manifest.get("schema") != "verified-api-readable-compact-v2":
        raise ValueError("serialized-input manifest has an unknown schema")
    if manifest.get("output", {}).get("sha256") != sha256_file(serialized_path):
        raise ValueError("serialized compact inputs do not match their manifest")
    if manifest.get("dataset", {}).get("sha256") != sha256_file(train_path):
        raise ValueError("serialized inputs were built from a different train file")
    invariants = manifest.get("invariants")
    required_invariants = {
        "all_artifact_hashes_verified",
        "all_row_contract_hashes_verified",
        "all_codec_roundtrips_verified",
        "all_student_constant_prefixes_verified",
        "all_f2_semantic_roundtrips_verified",
        "f2_system_prompt_self_contained_and_hashed",
        "all_complete_prompts_within_limit",
        "opaque_source_ids_expanded",
        "cfg_explicit",
        "all_user_functions_retained",
        "all_external_symbols_retained",
        "transfer_table_redundancy_proven",
        "train_dev_representation_contract_identical",
        "exact_audited_execution_exclusions_applied",
        "all_remaining_rows_byte_identical_to_parent",
        "heldout_175_disjoint_and_untouched",
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(name) is not True for name in required_invariants
    ):
        raise ValueError("serialized compact input invariants are incomplete")
    f2_contract = manifest.get("f2_prompt_contract")
    if not isinstance(f2_contract, Mapping):
        raise ValueError("serialized inputs have no F2 prompt contract")
    system_prompt = f2_contract.get("system_prompt")
    system_prompt_sha256 = str(
        f2_contract.get("system_prompt_sha256") or ""
    )
    if (
        f2_contract.get("representation_schema") != F2_REPRESENTATION_SCHEMA
        or f2_contract.get("tokenizer_sha256") != sha256_file(tokenizer_path)
        or f2_contract.get("all_rows_within_limit") is not True
        or not isinstance(system_prompt, str)
        or not system_prompt.strip()
        or hashlib.sha256(system_prompt.encode()).hexdigest()
        != system_prompt_sha256
    ):
        raise ValueError("serialized F2 system-prompt binding is invalid")
    rows = load_jsonl(serialized_path)
    derivation = manifest.get("derivation")
    if (
        manifest.get("training_objective_scope") != "executable_reward_only"
        or len(rows) != int(manifest.get("rows", -1))
        or not isinstance(derivation, Mapping)
        or derivation.get("schema")
        != "binary-multifunction-executable-subset-v1"
        or int(derivation.get("output_rows", -1)) != len(rows)
    ):
        raise ValueError("serialized compact row count differs from manifest")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        text = str(row.get("text") or "")
        verified = row.get("verified")
        if (
            not task_id
            or task_id in result
            or not text
            or hashlib.sha256(text.encode()).hexdigest()
            != row.get("text_sha256")
            or not isinstance(verified, Mapping)
            or verified.get("per_task_instruction_dictionary_roundtrip") is not True
            or verified.get("compact_semantic_f2_roundtrip") is not True
            or verified.get("branch_targets_reconstructed_from_cfg") is not True
            or row.get("representation_schema") != F2_REPRESENTATION_SCHEMA
            or row.get("system_prompt_sha256") != system_prompt_sha256
            or verified.get("opaque_custom_ids_in_text") is not False
        ):
            raise ValueError(
                f"invalid or duplicate serialized compact row: {task_id!r}"
            )
        result[task_id] = row
    return result, system_prompt, manifest


def validate_failure_inputs(
    *,
    train_path: Path,
    score_path: Path,
    prediction_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    train_rows = load_jsonl(train_path)
    tests: dict[str, str] = {}
    for row in train_rows:
        task_id = str(row.get("task_id") or "")
        test_code = str(
            row.get("acceptance_tests")
            or row.get("tests")
            or row.get("feedback_tests")
            or ""
        )
        if not task_id or task_id in tests or not test_code:
            raise ValueError("train file has missing/duplicate task or tests")
        tests[task_id] = test_code

    score = load_object(score_path)
    if score.get("schema") != "direct-compact-attested-passk-v1":
        raise ValueError("score report has an unknown schema")
    if score.get("evaluator", {}).get("completion_attestation") != (
        "per-run-256-bit-marker-exactly-once-v1"
    ):
        raise ValueError("score report does not attest hardened Dart completion")
    if score.get("evaluation", {}).get("sha256") != sha256_file(train_path):
        raise ValueError("score report was not computed on the selected train file")
    if Path(score.get("predictions", {}).get("path", "")).resolve() != (
        prediction_path.resolve()
    ):
        raise ValueError("score report points at a different prediction file")
    prediction_sha = sha256_file(prediction_path)
    if score.get("predictions", {}).get("sha256") != prediction_sha:
        raise ValueError("prediction file does not match score report")
    provenance_path = Path(str(prediction_path) + ".provenance.json")
    if score.get("predictions", {}).get("provenance_sha256") != sha256_file(
        provenance_path
    ):
        raise ValueError("prediction provenance does not match score report")
    provenance = load_object(provenance_path)
    if (
        provenance.get("schema") != "direct-compact-inference-v1"
        or provenance.get("output_sha256") != prediction_sha
    ):
        raise ValueError("prediction provenance is stale or invalid")

    predictions_value = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(predictions_value, list):
        raise ValueError("predictions must be a JSON array")
    predictions: dict[str, list[str]] = {}
    for row in predictions_value:
        task_id = str(row.get("id") or "") if isinstance(row, Mapping) else ""
        samples = row.get("predictions") if isinstance(row, Mapping) else None
        if (
            not task_id
            or task_id in predictions
            or not isinstance(samples, list)
            or len(samples) != int(score.get("k", -1))
        ):
            raise ValueError(f"malformed prediction row {task_id!r}")
        predictions[task_id] = [str(sample or "") for sample in samples]
    if set(predictions) != set(tests):
        raise ValueError("train predictions do not cover the exact train task set")

    task_results: dict[str, dict[str, Any]] = {}
    for row in score.get("task_results") or []:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in task_results:
            raise ValueError("score report has missing/duplicate task result")
        task_results[task_id] = dict(row)
    if set(task_results) != set(tests):
        raise ValueError("score task set differs from train task set")
    return tests, task_results, predictions


def count_prompt_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    reserve: int,
) -> int:
    total = reserve
    for message in messages:
        encoded = tokenizer.encode(
            str(message.get("content") or ""), add_special_tokens=False
        )
        total += len(encoded.ids if hasattr(encoded, "ids") else encoded)
    return total


def build_repair_prompt(
    *,
    tokenizer: Any,
    serialized_text: str,
    student_predictions: Sequence[str],
    include_student_candidate: bool,
    max_prompt_tokens: int,
    chat_overhead_reserve: int,
    task_id: str,
    system_prompt: str,
) -> tuple[list[dict[str, str]], int, dict[str, Any]]:
    """Build one lossless F2 repair prompt with an optional whole candidate."""

    # The base request is byte-for-byte the same manifest-bound system + F2
    # user message that was preflighted for the Qwen teacher.  In particular,
    # do not add a redundant wrapper or instruction after the F2 ``X`` record:
    # the longest sealed production rows intentionally sit close to the cap.
    base_messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": serialized_text},
    ]
    base_count = count_prompt_tokens(
        tokenizer, base_messages, chat_overhead_reserve
    )
    if base_count > max_prompt_tokens:
        raise RuntimeError(
            f"{task_id}: compressed enriched source+CFG alone needs "
            f"{base_count} sealed-Qwen tokens, cap is {max_prompt_tokens}; "
            "source truncation is forbidden"
        )
    current = base_messages
    count = base_count
    inclusion: dict[str, Any] = {
        "requested": bool(include_student_candidate),
        "included": False,
        "reason": "disabled" if not include_student_candidate else "",
        "base_prompt_tokens": base_count,
        "base_prompt_sha256": stable_sha256(base_messages),
    }
    if include_student_candidate:
        candidates = [
            str(raw)
            for raw in student_predictions
            if str(raw).strip()
        ]
        if candidates:
            candidate = min(
                candidates,
                key=lambda value: (
                    len(value),
                    hashlib.sha256(value.encode()).hexdigest(),
                ),
            )
            augmentation_content = RS_CANDIDATE_MESSAGE_PREFIX + candidate
            candidate_messages = base_messages + [
                {"role": "user", "content": augmentation_content}
            ]
            candidate_count = count_prompt_tokens(
                tokenizer,
                candidate_messages,
                chat_overhead_reserve,
            )
            inclusion.update(
                {
                    "candidate_sha256": hashlib.sha256(
                        candidate.encode()
                    ).hexdigest(),
                    "candidate_characters": len(candidate),
                    "candidate_prompt_tokens": candidate_count,
                    "augmentation_message_sha256": hashlib.sha256(
                        augmentation_content.encode()
                    ).hexdigest(),
                }
            )
            if candidate_count <= max_prompt_tokens:
                current = candidate_messages
                count = candidate_count
                inclusion.update({"included": True, "reason": "fits"})
            else:
                inclusion["reason"] = "optional_candidate_exceeds_cap"
        else:
            inclusion["reason"] = "no_nonempty_student_candidate"
    return current, count, inclusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--serialized_inputs", required=True)
    parser.add_argument("--serialized_manifest", default="")
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--train_seal", required=True)
    parser.add_argument("--score_report", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--executable_view_report", required=True)
    parser.add_argument(
        "--expected_executable_view_report_sha256",
        required=True,
    )
    parser.add_argument(
        "--qwen_build_manifest",
        default="",
        help=(
            "Defaults to qwen_mc_sequence_train.build.json beside the Qwen "
            "stage checkpoint directory."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--env_file", default="/workspace/OpenAI.env")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning_effort",
        choices=REASONING_EFFORTS,
        default="high",
    )
    parser.add_argument("--samples_per_task", type=int, default=4)
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument(
        "--max_output_tokens_ceiling",
        type=int,
        default=12288,
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=12000)
    parser.add_argument("--chat_overhead_reserve", type=int, default=256)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--api_timeout", type=int, default=600)
    parser.add_argument("--api_retries", type=int, default=4)
    parser.add_argument("--retry_base_seconds", type=float, default=2.0)
    parser.add_argument("--retry_max_seconds", type=float, default=30.0)
    parser.add_argument("--eval_timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=2)
    parser.add_argument("--min_verified_tasks", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--include_student_candidate",
        action="store_true",
        help=(
            "Include the shortest failed student candidate only when the full "
            "prompt remains within the sealed token cap."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positive = {
        "samples_per_task": args.samples_per_task,
        "max_output_tokens": args.max_output_tokens,
        "max_output_tokens_ceiling": args.max_output_tokens_ceiling,
        "max_prompt_tokens": args.max_prompt_tokens,
        "workers": args.workers,
        "api_timeout": args.api_timeout,
        "api_retries": args.api_retries,
        "eval_timeout": args.eval_timeout,
        "stability_runs": args.stability_runs,
        "min_verified_tasks": args.min_verified_tasks,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("positive arguments required: " + ", ".join(invalid))
    if args.model != "gpt-5.6-sol":
        raise ValueError("this RS-SFT collector is sealed to gpt-5.6-sol")
    if (
        args.max_output_tokens != 8192
        or args.max_output_tokens_ceiling != 12288
    ):
        raise ValueError(
            "production GPT-5.6-sol RS uses the sealed 8192 -> 12288 "
            "max-output-token escalation"
        )
    if args.limit < 0 or args.chat_overhead_reserve < 0:
        raise ValueError("limit and chat overhead reserve must be non-negative")
    if (
        args.retry_base_seconds < 0
        or args.retry_max_seconds < args.retry_base_seconds
    ):
        raise ValueError("invalid API retry backoff")

    serialized_path = Path(args.serialized_inputs).expanduser().resolve()
    manifest_path = (
        Path(args.serialized_manifest).expanduser().resolve()
        if args.serialized_manifest
        else Path(str(serialized_path) + ".manifest.json")
    )
    tokenizer_path = Path(args.tokenizer_json).expanduser().resolve()
    train_path = Path(args.train_file).expanduser().resolve()
    train_seal_path = Path(args.train_seal).expanduser().resolve()
    score_path = Path(args.score_report).expanduser().resolve()
    prediction_path = Path(args.predictions).expanduser().resolve()
    student_checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    qwen_build_manifest = (
        Path(args.qwen_build_manifest).expanduser().resolve()
        if args.qwen_build_manifest
        else student_checkpoint.parent / "qwen_mc_sequence_train.build.json"
    )
    executable_view_report = Path(
        args.executable_view_report
    ).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path in (
        serialized_path,
        manifest_path,
        tokenizer_path,
        train_path,
        train_seal_path,
        score_path,
        prediction_path,
        Path(str(prediction_path) + ".provenance.json"),
        qwen_build_manifest,
        executable_view_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise ValueError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "attempts.jsonl"
    if journal_path.exists() and not args.resume:
        raise ValueError(f"refusing to overwrite journal: {journal_path}")

    serialized, f2_system_prompt, serialized_manifest = (
        validate_serialized_inputs(
            serialized_path, train_path, manifest_path, tokenizer_path
        )
    )
    tests, task_results, predictions = validate_failure_inputs(
        train_path=train_path,
        score_path=score_path,
        prediction_path=prediction_path,
    )
    prediction_provenance = load_object(
        Path(str(prediction_path) + ".provenance.json")
    )
    qwen_student = validate_qwen_student_checkpoint(
        student_checkpoint,
        qwen_build_manifest=qwen_build_manifest,
        inference_provenance=prediction_provenance,
    )
    executable_view = validate_executable_view(
        dataset=train_path,
        seal=train_seal_path,
        f2=serialized_path,
        f2_manifest=manifest_path,
        build_report=executable_view_report,
        expected_build_report_sha256=(
            args.expected_executable_view_report_sha256
        ),
        contract=student_checkpoint / "compact_contract.json",
        verify_heldout=False,
    )
    if (
        executable_view["parent_f2"]["sha256"]
        != qwen_student["qwen_prompt_artifact"]["sha256"]
        or executable_view["parent_f2_manifest"]["sha256"]
        != qwen_student["qwen_prompt_manifest"]["sha256"]
        or qwen_student["qwen_f2_prompt_contract"]["system_prompt_sha256"]
        != serialized_manifest["f2_prompt_contract"]["system_prompt_sha256"]
    ):
        raise ValueError(
            "OpenAI RS inputs are not the exact executable subset of the "
            "full F2 prompt contract used by the Qwen student"
        )
    if (
        qwen_student.get("fit_task_count")
        != executable_view["parent_rows"]
        or qwen_student.get("fit_ordered_task_ids_sha256")
        != executable_view["parent_task_ids_sha256"]
        or qwen_student.get("heldout_task_count") != QWEN_HELDOUT_ROWS
        or qwen_student.get("heldout_intersection_count") != 0
    ):
        raise ValueError(
            "Qwen student and executable view have different sealed fit or "
            "heldout commitments"
        )
    if set(serialized) != set(tests):
        raise ValueError("serialized compact inputs do not match train task set")

    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError("install tokenizers for prompt preflight") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    failed_ids = sorted(
        task_id
        for task_id, result in task_results.items()
        if not bool(result.get("pass_at_k"))
    )
    if args.limit:
        failed_ids = failed_ids[: args.limit]
    if not failed_ids:
        raise RuntimeError("no failed training task was selected")
    if len(failed_ids) < args.min_verified_tasks:
        raise RuntimeError(
            f"only {len(failed_ids)} failed training tasks are eligible, so "
            f"{args.min_verified_tasks} unique verified repairs are impossible"
        )

    messages: dict[str, list[dict[str, str]]] = {}
    prompt_counts: dict[str, int] = {}
    candidate_inclusion: dict[str, dict[str, Any]] = {}
    for task_id in failed_ids:
        current, count, inclusion = build_repair_prompt(
            tokenizer=tokenizer,
            serialized_text=serialized[task_id]["text"],
            student_predictions=predictions[task_id],
            include_student_candidate=bool(args.include_student_candidate),
            max_prompt_tokens=args.max_prompt_tokens,
            chat_overhead_reserve=args.chat_overhead_reserve,
            task_id=task_id,
            system_prompt=f2_system_prompt,
        )
        messages[task_id] = current
        prompt_counts[task_id] = count
        candidate_inclusion[task_id] = inclusion

    endpoint = validate_openai_base_url(args.base_url)
    request_parameters = {
        "max_output_tokens_initial": args.max_output_tokens,
        "max_output_tokens_ceiling": args.max_output_tokens_ceiling,
        "max_output_tokens_escalation": {
            "status": "incomplete",
            "incomplete_details_reason": "max_output_tokens",
            "otherwise_escalate": False,
        },
        "reasoning": {"effort": args.reasoning_effort},
        "store": False,
    }
    run_payload = {
        "provider": "openai",
        "api": "responses",
        "base_url": endpoint,
        "requested_model": args.model,
        "request_parameters": request_parameters,
        "samples_per_task": args.samples_per_task,
        "minimum_unique_verified_tasks": args.min_verified_tasks,
        "scheduler": {
            "order": "sample-major-task-wide-rounds",
            "workers": args.workers,
            "bounded_in_flight": True,
            "stop_after_unique_verified_tasks": args.min_verified_tasks,
            "drain_only_already_running_window_after_stop": True,
        },
        "selected_failed_task_ids": failed_ids,
        "prompt_sha256_by_task": {
            task_id: stable_sha256(messages[task_id]) for task_id in failed_ids
        },
        "prompt_token_estimate_by_task": {
            task_id: prompt_counts[task_id] for task_id in failed_ids
        },
        "prompt_contract": {
            "system_prompt_sha256": hashlib.sha256(
                f2_system_prompt.encode()
            ).hexdigest(),
            "serialized_manifest": file_record(manifest_path),
            "representation_schema": (
                serialized_manifest["f2_prompt_contract"][
                    "representation_schema"
                ]
            ),
            "max_prompt_tokens": args.max_prompt_tokens,
            "chat_overhead_reserve": args.chat_overhead_reserve,
            "include_student_candidate_requested": bool(
                args.include_student_candidate
            ),
            "base_messages_are_exact_manifest_system_plus_f2_text": True,
            "optional_candidate_augmentation": {
                "schema": RS_CANDIDATE_AUGMENTATION_SCHEMA,
                "role": "user",
                "message_prefix": RS_CANDIDATE_MESSAGE_PREFIX,
                "message_prefix_sha256": hashlib.sha256(
                    RS_CANDIDATE_MESSAGE_PREFIX.encode()
                ).hexdigest(),
                "whole_candidate_only": True,
                "fully_assembled_message_preflight_required": True,
            },
            "student_candidate_inclusion_by_task": candidate_inclusion,
            "student_candidate_tasks_included": sum(
                bool(row["included"]) for row in candidate_inclusion.values()
            ),
            "student_candidate_tasks_omitted": sum(
                not bool(row["included"]) for row in candidate_inclusion.values()
            ),
            "optional_student_candidate_truncation_permitted": False,
            "compressed_enriched_source_or_cfg_truncation_permitted": False,
            "private_tests_exposed_to_api": False,
            "gold_source_exposed_to_api": False,
        },
        "verifier": {
            "completion_attestation": (
                "per-run-256-bit-marker-exactly-once-v1"
            ),
            "timeout_seconds": args.eval_timeout,
            "stability_runs": args.stability_runs,
        },
        "retry_policy": {
            "api_timeout_seconds": args.api_timeout,
            "api_retries": args.api_retries,
            "retry_base_seconds": args.retry_base_seconds,
            "retry_max_seconds": args.retry_max_seconds,
            "max_output_tokens_initial": args.max_output_tokens,
            "max_output_tokens_ceiling": (
                args.max_output_tokens_ceiling
            ),
            "token_budget_escalation_consumes_request_attempt": True,
        },
        "inputs": {
            "serialized_inputs": file_record(serialized_path),
            "serialized_manifest": file_record(manifest_path),
            "student_tokenizer": file_record(tokenizer_path),
            "train_file": file_record(train_path),
            "train_seal": file_record(train_seal_path),
            "score_report": file_record(score_path),
            "predictions": file_record(prediction_path),
            "prediction_provenance": file_record(
                Path(str(prediction_path) + ".provenance.json")
            ),
            "executable_view": executable_view,
        },
        "qwen_student_stage": qwen_student,
    }
    run_contract_path = output_dir / "run_contract.json"
    if journal_path.is_file() and not run_contract_path.is_file():
        raise ValueError(
            "resume refused: attempts journal has no immutable run contract"
        )
    ensure_run_contract(run_contract_path, run_payload)
    run_contract_sha = sha256_file(run_contract_path)

    existing = load_jsonl(journal_path) if journal_path.is_file() else []
    terminal: set[tuple[str, int]] = set()
    existing_verified_tasks: set[str] = set()
    resolved_models: set[str] = set()
    selected_ids = set(failed_ids)
    for row in existing:
        task_id = str(row.get("task_id") or "")
        sample_index = row.get("sample_index")
        if (
            row.get("schema") != SCHEMA
            or row.get("run_contract_sha256") != run_contract_sha
            or row.get("provider") != "openai"
            or row.get("requested_model") != args.model
            or row.get("request_parameters") != request_parameters
            or task_id not in selected_ids
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < args.samples_per_task
            or row.get("prompt_sha256")
            != run_payload["prompt_sha256_by_task"][task_id]
        ):
            raise ValueError("resume journal contains an incompatible event")
        key = (task_id, sample_index)
        if row.get("terminal") is True:
            if key in terminal:
                raise ValueError(f"duplicate terminal journal record: {key}")
            terminal.add(key)
            if row.get("passed") is True:
                existing_verified_tasks.add(task_id)
            resolved_model = str(row.get("resolved_model") or "")
            if not resolved_model:
                raise ValueError("terminal journal event lacks resolved model")
            resolved_models.add(resolved_model)
        elif row.get("terminal") is not False:
            raise ValueError("journal event has a non-boolean terminal field")
    if len(resolved_models) > 1:
        raise ValueError("resume journal mixes multiple resolved OpenAI models")

    journal = Journal(journal_path)
    client: Any = None
    if len(existing_verified_tasks) < args.min_verified_tasks:
        load_env_file(Path(args.env_file).expanduser().resolve())
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"missing {args.api_key_env}")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("install the official OpenAI SDK") from exc
        client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=args.api_timeout,
        )

    def harvest(job: tuple[str, int]) -> dict[str, Any]:
        task_id, sample_index = job
        prompt = messages[task_id]
        prompt_sha = stable_sha256(prompt)
        current_output_budget = args.max_output_tokens
        for api_attempt in range(1, args.api_retries + 1):
            base = {
                "schema": SCHEMA,
                "run_contract_sha256": run_contract_sha,
                "created_at": utc_now(),
                "host": socket.gethostname(),
                "provider": "openai",
                "requested_model": args.model,
                "task_id": task_id,
                "sample_index": sample_index,
                "api_attempt": api_attempt,
                "prompt_sha256": prompt_sha,
                "prompt_tokens_estimate": prompt_counts[task_id],
                "request_parameters": request_parameters,
                "request_max_output_tokens": current_output_budget,
            }
            try:
                response = client.responses.create(
                    model=args.model,
                    input=prompt,
                    max_output_tokens=current_output_budget,
                    reasoning={"effort": args.reasoning_effort},
                    store=False,
                )
                raw = object_dict(response)
                content = response_text(raw, response)
                code = extract_code(content)
                status = str(raw.get("status") or "")
                if status != "completed":
                    details = raw.get("incomplete_details")
                    if details is not None and not isinstance(
                        details, Mapping
                    ):
                        raise ValueError(
                            "response incomplete_details is not an object"
                        )
                    raise IncompleteResponseError(
                        status=status,
                        details=details,
                        raw_response=raw,
                    )
                response_id = str(raw.get("id") or "")
                resolved_model = str(raw.get("model") or "")
                if not response_id or not resolved_model or not content or not code:
                    raise ValueError(
                        "response lacks id, resolved model, content, or code"
                    )
                usage = raw.get("usage")
                if not isinstance(usage, Mapping):
                    raise ValueError("response has no token usage")
                input_tokens = usage.get(
                    "input_tokens", usage.get("prompt_tokens")
                )
                output_tokens = usage.get(
                    "output_tokens", usage.get("completion_tokens")
                )
                total_tokens = usage.get("total_tokens")
                if (
                    isinstance(input_tokens, bool)
                    or not isinstance(input_tokens, (int, float))
                    or input_tokens <= 0
                    or isinstance(output_tokens, bool)
                    or not isinstance(output_tokens, (int, float))
                    or output_tokens <= 0
                    or isinstance(total_tokens, bool)
                    or not isinstance(total_tokens, (int, float))
                    or total_tokens < input_tokens + output_tokens
                ):
                    raise ValueError("response token usage is zero or inconsistent")
                compiled, passed, diagnostic, _source = (
                    evaluate_dart_jit_tests_detail(
                        code,
                        tests[task_id],
                        f"{task_id}_gpt_{sample_index}",
                        timeout=args.eval_timeout,
                        stability_runs=args.stability_runs,
                    )
                )
                result = {
                    **base,
                    "terminal": True,
                    "api_ok": True,
                    "resolved_model": resolved_model,
                    "response_id": response_id,
                    "response_status": status or "completed",
                    "usage": usage,
                    "raw_response": raw,
                    "content": content,
                    "content_sha256": hashlib.sha256(
                        content.encode()
                    ).hexdigest(),
                    "code": code,
                    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "compiled": bool(compiled),
                    "passed": bool(passed),
                    "verifier_diagnostic": str(diagnostic)[:2000],
                }
                journal.append(result)
                return result
            except Exception as exc:
                error_row = {
                    **base,
                    "terminal": False,
                    "api_ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "traceback_sha256": hashlib.sha256(
                        traceback.format_exc().encode()
                    ).hexdigest(),
                }
                next_output_budget = None
                if isinstance(exc, IncompleteResponseError):
                    next_output_budget = escalated_output_token_budget(
                        status=exc.status,
                        incomplete_details=exc.details,
                        current_budget=current_output_budget,
                        ceiling_budget=args.max_output_tokens_ceiling,
                    )
                    error_row.update(
                        {
                            "response_status": exc.status,
                            "incomplete_details": exc.details,
                            "incomplete_reason": exc.reason,
                            "response_id": str(
                                exc.raw_response.get("id") or ""
                            ),
                            "resolved_model": str(
                                exc.raw_response.get("model") or ""
                            ),
                            "usage": exc.raw_response.get("usage"),
                            "raw_response": exc.raw_response,
                            "max_output_tokens_escalated": (
                                next_output_budget is not None
                            ),
                            "next_request_max_output_tokens": (
                                next_output_budget
                            ),
                        }
                    )
                journal.append(error_row)
                if api_attempt == args.api_retries:
                    return error_row
                if next_output_budget is not None:
                    current_output_budget = next_output_budget
                    continue
                delay = min(
                    args.retry_max_seconds,
                    args.retry_base_seconds * (2 ** (api_attempt - 1)),
                )
                delay *= 0.75 + 0.5 * random.Random(
                    stable_sha256([task_id, sample_index, api_attempt])
                ).random()
                time.sleep(delay)
        raise AssertionError("unreachable")

    def progress(
        completed_now: int,
        scheduled_now: int,
        sample_index: int,
        verified_count: int,
    ) -> None:
        if completed_now % 20 == 0:
            print(
                "OPENAI_RS_PROGRESS "
                f"completed={completed_now} scheduled={scheduled_now} "
                f"round={sample_index} verified_tasks={verified_count}/"
                f"{args.min_verified_tasks}",
                flush=True,
            )

    scheduling = run_sample_major_bounded(
        failed_ids=failed_ids,
        samples_per_task=args.samples_per_task,
        terminal_slots=terminal,
        verified_task_ids=existing_verified_tasks,
        minimum_verified_tasks=args.min_verified_tasks,
        workers=args.workers,
        harvest=harvest,
        progress=progress,
    )

    journal_rows = load_jsonl(journal_path)
    latest_terminal: dict[tuple[str, int], dict[str, Any]] = {}
    attempted_slots: set[tuple[str, int]] = set()
    final_resolved_models: set[str] = set()
    response_ids: set[str] = set()
    for row in journal_rows:
        task_id = str(row.get("task_id") or "")
        sample_index = row.get("sample_index")
        if (
            row.get("schema") != SCHEMA
            or row.get("run_contract_sha256") != run_contract_sha
            or row.get("provider") != "openai"
            or row.get("requested_model") != args.model
            or row.get("request_parameters") != request_parameters
            or task_id not in selected_ids
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < args.samples_per_task
            or row.get("prompt_sha256")
            != run_payload["prompt_sha256_by_task"][task_id]
        ):
            raise RuntimeError("journal changed or contains an incompatible event")
        key = (task_id, sample_index)
        attempted_slots.add(key)
        if row.get("terminal") is True:
            if key in latest_terminal:
                raise RuntimeError(f"duplicate terminal journal record: {key}")
            latest_terminal[key] = row
            resolved_model = str(row.get("resolved_model") or "")
            response_id = str(row.get("response_id") or "")
            if not resolved_model or not response_id:
                raise RuntimeError("terminal event lacks backend identity")
            if response_id in response_ids:
                raise RuntimeError(f"duplicate OpenAI response id: {response_id}")
            response_ids.add(response_id)
            final_resolved_models.add(resolved_model)
        elif row.get("terminal") is not False:
            raise RuntimeError("journal event has a non-boolean terminal field")
    if len(final_resolved_models) > 1:
        raise RuntimeError("journal mixes multiple resolved OpenAI models")

    expected_slots = {
        (task_id, sample_index)
        for task_id in failed_ids
        for sample_index in range(args.samples_per_task)
    }
    missing_slots = sorted(expected_slots - set(latest_terminal))
    unattempted_slots = sorted(expected_slots - attempted_slots)
    retryable_slots = sorted(attempted_slots - set(latest_terminal))
    verified = sorted(
        (
            {
                "schema": SCHEMA,
                "run_contract_sha256": run_contract_sha,
                "provider": row["provider"],
                "requested_model": row["requested_model"],
                "resolved_model": row["resolved_model"],
                "response_id": row["response_id"],
                "task_id": task_id,
                "sample_index": sample_index,
                "prompt_sha256": row["prompt_sha256"],
                "code": row["code"],
                "code_sha256": row["code_sha256"],
                "ok": True,
                "independently_completion_attested": True,
                "stability_runs": args.stability_runs,
                "reasoning_effort": args.reasoning_effort,
            }
            for (task_id, sample_index), row in latest_terminal.items()
            if row.get("passed") is True
        ),
        key=lambda row: (row["task_id"], row["sample_index"]),
    )
    unique_verified = len({row["task_id"] for row in verified})
    atomic_jsonl(output_dir / "verified_repairs.jsonl", verified)
    report = {
        "schema": SCHEMA,
        "status": (
            "complete"
            if unique_verified >= args.min_verified_tasks
            else "incomplete"
        ),
        "completed_at": utc_now(),
        "provider": "openai",
        "api": "responses",
        "base_url": endpoint,
        "requested_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "failed_training_tasks": len(failed_ids),
        "samples_per_task": args.samples_per_task,
        "expected_slots": len(expected_slots),
        "terminal_slots": len(latest_terminal),
        "attempted_slots": len(attempted_slots),
        "scheduled_this_invocation": scheduling["scheduled_now"],
        "completed_this_invocation": scheduling["completed_now"],
        "stop_reason": scheduling["stop_reason"],
        "missing_slots_count": len(missing_slots),
        "missing_slots_preview": [
            {"task_id": task_id, "sample_index": sample_index}
            for task_id, sample_index in missing_slots[:100]
        ],
        "missing_slots_preview_truncated": len(missing_slots) > 100,
        "unattempted_slots_count": len(unattempted_slots),
        "retryable_slots_count": len(retryable_slots),
        "max_output_token_escalations": sum(
            row.get("max_output_tokens_escalated") is True
            for row in journal_rows
        ),
        "incomplete_max_output_token_events": sum(
            row.get("response_status") == "incomplete"
            and row.get("incomplete_reason") == "max_output_tokens"
            for row in journal_rows
        ),
        "verified_candidates": len(verified),
        "unique_verified_tasks": unique_verified,
        "minimum_unique_verified_tasks": args.min_verified_tasks,
        "production_coverage_met": unique_verified >= args.min_verified_tasks,
        "resolved_models": sorted(final_resolved_models),
        "request_parameters": request_parameters,
        "prompt": {
            "include_student_candidate_requested": bool(
                args.include_student_candidate
            ),
            "student_candidate_tasks_included": sum(
                bool(row["included"]) for row in candidate_inclusion.values()
            ),
            "student_candidate_tasks_omitted": sum(
                not bool(row["included"]) for row in candidate_inclusion.values()
            ),
            "student_candidate_inclusion_by_task": candidate_inclusion,
            "max_prompt_tokens": args.max_prompt_tokens,
            "chat_overhead_reserve": args.chat_overhead_reserve,
            "maximum_estimated_tokens": max(prompt_counts.values()),
            "prompt_hashes": {
                task_id: stable_sha256(messages[task_id])
                for task_id in failed_ids
            },
            "private_tests_exposed_to_api": False,
            "gold_source_exposed_to_api": False,
            "representation": (
                "verified compressed enriched assembly plus explicit compressed CFG"
            ),
        },
        "inputs": {
            "serialized_inputs": {
                "path": str(serialized_path),
                "sha256": sha256_file(serialized_path),
            },
            "serialized_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "train_file": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
            },
            "train_seal": file_record(train_seal_path),
            "score_report": {
                "path": str(score_path),
                "sha256": sha256_file(score_path),
            },
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "qwen_student_stage": qwen_student,
            "executable_view": executable_view,
            "run_contract": file_record(run_contract_path),
        },
        "outputs": {
            "attempts_sha256": sha256_file(journal_path),
            "verified_repairs_sha256": sha256_file(
                output_dir / "verified_repairs.jsonl"
            ),
        },
    }
    atomic_json(output_dir / "report.json", report)
    print(
        f"CHATGPT_RS_RESULT failed_tasks={len(failed_ids)} "
        f"verified_tasks={unique_verified} verified_candidates={len(verified)} "
        f"required={args.min_verified_tasks} stop={scheduling['stop_reason']} "
        f"missing_slots={len(missing_slots)}",
        flush=True,
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
