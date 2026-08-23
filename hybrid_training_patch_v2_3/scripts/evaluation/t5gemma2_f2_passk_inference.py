#!/usr/bin/env python3
"""Sealed pre/post-SFT T5Gemma 2 inference on the held-out F2 view.

The encoder receives only the same verified F2 text used for SFT. Gold Dart
and tests are never serialized to the model. Generation is journaled one
candidate at a time so an interruption resumes at the exact next seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.training.seq2seq_verpo_core import normalize_generated_seq2seq_ids
from scripts.training.t5gemma2_compiler_feedback_verpo import (
    _decode_candidate,
    _decoder_special_ids,
    _encode_source,
    _generation_kwargs,
)
from scripts.training.t5gemma2_enriched_sft import (
    RUN_SCHEMA as SFT_RUN_SCHEMA,
    _adapter_weight_target_modules,
    build_encoder_source,
)


INFERENCE_SCHEMA = "t5gemma2-f2-heldout-inference-v1"
JOURNAL_SCHEMA = "t5gemma2-f2-heldout-generation-journal-v1"
PROVENANCE_SCHEMA = "direct-compact-inference-v1"
MODEL_NAME = "google/t5gemma-2-4b-4b"
MODEL_REVISION = "487d4acf21a4d70c70bf534265b5263c9424979e"
HELDOUT_ROWS = 175
DATASET_SHA256 = "abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7"
DATASET_SEAL_SHA256 = "5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a"
F2_SHA256 = "6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab"
F2_MANIFEST_SHA256 = "777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44"
TASK_SET_SHA256 = "9b93767fd4d0b4057bc752113faeb1efda9faa609e537e189350a6d874d6e38e"
MIXED_RS_SFT_RUN_SCHEMA = "t5gemma2-mixed-rs-sft-run-v1"
TYPED_CONTRACT_SFT_RUN_SCHEMA = "t5gemma2-typed-opaque-contract-sft-run-v1"
TYPED_DIRECT_RS_SFT_RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-run-v1"
TYPED_DIRECT_RS_SFT_PASS2_RUN_SCHEMA = (
    "t5gemma2-typed-direct-rs-sft-pass2-run-v1"
)
TYPED_DIRECT_RS_SFT_PASS3_RUN_SCHEMA = (
    "t5gemma2-typed-direct-rs-sft-pass3-run-v1"
)
TYPED_FOLD_RS_SFT_UNION_RUN_SCHEMA = "t5gemma2-typed-fold-rs-sft-union-run-v1"
KNOWN_TYPED_CONTAMINANT = "sigless_6b1dd0c6b6fc"
PASS3_TRAINER_SHA256 = "2274a500e73b6e37f3fdc3144b6d70cb28aa5bb3ec463682a5a38df9ac7bd54f"
UPDATE58_RUN_CONTRACT_SHA256 = "0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f"
UPDATE58_ADAPTER_WEIGHTS_SHA256 = "62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec"
UPDATE58_ADAPTER_CONFIG_SHA256 = "b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b"
FOLD_TRAINER_SHA256 = "2ae23d69f5dffe816d6b88d0356dc16d88bec16964a1d5dbe66db19c72afdd3c"
TYPED_SFT_RUN_CONTRACT_SHA256 = "3cb25d54f12743ed43572b219e119667f264abab94ec4cbfac72a94407fbdfc7"
TYPED_SFT_ADAPTER_WEIGHTS_SHA256 = "71078435105dc29aff1aba5942abd5c272e78ef817896081f6e994938da9d77a"
TYPED_SFT_ADAPTER_CONFIG_SHA256 = "f3701f13cb66b6b5952cd1dd2a71b17206e77c1c646ec806f6dd43d7e059a92d"
SUPPORTED_ADAPTER_RUN_SCHEMAS = frozenset(
    {
        SFT_RUN_SCHEMA,
        MIXED_RS_SFT_RUN_SCHEMA,
        TYPED_CONTRACT_SFT_RUN_SCHEMA,
        TYPED_DIRECT_RS_SFT_RUN_SCHEMA,
        TYPED_DIRECT_RS_SFT_PASS2_RUN_SCHEMA,
        TYPED_DIRECT_RS_SFT_PASS3_RUN_SCHEMA,
        TYPED_FOLD_RS_SFT_UNION_RUN_SCHEMA,
    }
)


@dataclass(frozen=True)
class EvaluationRow:
    task_id: str
    source: str
    source_sha256: str


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_typed_no_leakage_privacy(
    contract: Mapping[str, Any], *, label: str
) -> None:
    privacy = contract.get("privacy")
    if (
        not isinstance(privacy, Mapping)
        or privacy.get("heldout_overlap") != 0
        or privacy.get("heldout_content_model_visible") is not False
        or privacy.get("tests_model_visible") is not False
        or privacy.get("private_feedback_model_visible") is not False
        or privacy.get("reasoning_persisted") is not False
    ):
        raise ValueError(f"{label} privacy binding failed")


def _require_typed_schedule(dataset: Mapping[str, Any], *, rows: int, label: str) -> None:
    schedule = dataset.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != rows:
        raise ValueError(f"{label} schedule coverage differs")
    task_ids = [str(row.get("source_task_id") or "") for row in schedule]
    pair_ids = [str(row.get("pair_id") or "") for row in schedule]
    if (
        not all(task_ids)
        or len(task_ids) != len(set(task_ids))
        or not all(pair_ids)
        or len(pair_ids) != len(set(pair_ids))
        or KNOWN_TYPED_CONTAMINANT in task_ids
        or dataset.get("schedule_sha256") != canonical_sha256(schedule)
        or dataset.get("task_ids_sha256") != canonical_sha256(task_ids)
    ):
        raise ValueError(f"{label} schedule identity differs")


def _require_typed_pass3_contract(contract: Mapping[str, Any]) -> None:
    dataset = contract.get("dataset")
    privacy = contract.get("privacy")
    warmstart = contract.get("warmstart")
    optimization = contract.get("optimization")
    runtime = contract.get("runtime")
    lora = contract.get("lora")
    if not all(
        isinstance(value, Mapping)
        for value in (dataset, privacy, warmstart, optimization, runtime, lora)
    ):
        raise ValueError("typed pass-3 contract is structurally incomplete")
    rows = int(dataset.get("rows", -1))
    composition = dataset.get("composition")
    verification = dataset.get("full_acceptance_reverification")
    typed_train = dataset.get("typed_train")
    exclusions = (
        typed_train.get("training_exclusions")
        if isinstance(typed_train, Mapping)
        else None
    )
    c002_rows = (
        int(composition.get("kimi_c002_tail", -1))
        if isinstance(composition, Mapping)
        else -1
    )
    if (
        dataset.get("schema") != "t5gemma2-typed-direct-rs-sft-pass3-dataset-v1"
        or dataset.get("architecture") != "native_encoder_decoder"
        or not 13 <= rows <= 60
        or not isinstance(composition, Mapping)
        or int(composition.get("verified_direct", -1)) != rows
        or int(composition.get("kimi_c001", -1)) != 12
        or int(composition.get("kimi_c002_prefix", -1)) != 1
        or c002_rows != rows - 13
        or not 0 <= c002_rows <= 47
        or any(
            int(composition.get(name, -1)) != 0
            for name in (
                "prior_225_replay",
                "pass2_209_replay",
                "gold_replay",
                "repair_conditioned",
                "reasoning_rows",
            )
        )
        or dataset.get("heldout_overlap") != 0
        or dataset.get("heldout_175_model_visible") is not False
        or dataset.get("tests_model_visible") is not False
        or dataset.get("private_feedback_model_visible") is not False
        or dataset.get("repair_conditioned_prefixes_visible") is not False
        or dataset.get("reasoning_model_visible") is not False
        or dataset.get("known_contaminant_excluded") != KNOWN_TYPED_CONTAMINANT
        or dataset.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or dataset.get("task_id_deduplication")
        != "reject_any_cross_source_or_prior_overlap"
        or dataset.get("all_targets_bound_to_provider_or_zero_api_verification_journals")
        is not True
        or dataset.get("production_floor_eligible") is not True
        or not isinstance(verification, Mapping)
        or int(verification.get("rows", -1)) != rows
        or int(verification.get("passed", -1)) != rows
        or verification.get("tests_model_visible") is not False
        or verification.get("diagnostics_persisted") is not False
        or not isinstance(typed_train, Mapping)
        or typed_train.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or not isinstance(exclusions, Mapping)
        or KNOWN_TYPED_CONTAMINANT not in (exclusions.get("task_ids") or [])
    ):
        raise ValueError("typed pass-3 direct-only dataset binding failed")
    _require_typed_schedule(dataset, rows=rows, label="typed pass-3")
    _require_typed_no_leakage_privacy(contract, label="typed pass-3")
    expected_updates_per_epoch = math.ceil(rows / 8)
    if (
        runtime.get("trainer_sha256") != PASS3_TRAINER_SHA256
        or int(warmstart.get("update", -1)) != 58
        or warmstart.get("run_contract_sha256") != UPDATE58_RUN_CONTRACT_SHA256
        or warmstart.get("adapter_weights_sha256")
        != UPDATE58_ADAPTER_WEIGHTS_SHA256
        or warmstart.get("adapter_config_sha256") != UPDATE58_ADAPTER_CONFIG_SHA256
        or int(optimization.get("epochs", -1)) != 2
        or int(optimization.get("batch_size", -1)) != 1
        or int(optimization.get("gradient_accumulation", -1)) != 8
        or not math.isclose(
            float(optimization.get("learning_rate", -1.0)),
            2e-5,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(optimization.get("warmup_updates", -1)) != 0
        or int(optimization.get("seed", -1)) != 42
        or int(optimization.get("updates_per_epoch", -1))
        != expected_updates_per_epoch
        or int(optimization.get("planned_updates", -1))
        != 2 * expected_updates_per_epoch
        or lora.get("new_adapter_attached") is not False
        or lora.get("warmstart_weights_continued") is not True
        or contract.get("production_floor_eligible") is not True
    ):
        raise ValueError("typed pass-3 lineage/optimizer binding failed")


def _require_typed_fold_contract(contract: Mapping[str, Any]) -> None:
    dataset = contract.get("dataset")
    warmstart = contract.get("warmstart")
    optimization = contract.get("optimization")
    runtime = contract.get("runtime")
    lora = contract.get("lora")
    if not all(
        isinstance(value, Mapping)
        for value in (dataset, warmstart, optimization, runtime, lora)
    ):
        raise ValueError("typed folded-union contract is structurally incomplete")
    rows = int(dataset.get("rows", -1))
    composition = dataset.get("composition")
    verification = dataset.get("full_acceptance_reverification")
    cross = dataset.get("equivalent_typed_source_cross_acceptance")
    c002_rows = (
        int(composition.get("kimi_c002_tail", -1))
        if isinstance(composition, Mapping)
        else -1
    )
    if (
        dataset.get("schema") != "t5gemma2-typed-fold-rs-sft-union-dataset-v1"
        or dataset.get("arm") != "B_fold_only"
        or dataset.get("fresh_branch_from") != "typed_sft_optstep348"
        or not 447 <= rows <= 494
        or not isinstance(composition, Mapping)
        or int(composition.get("verified_direct", -1)) != rows
        or int(composition.get("pass1_225", -1)) != 225
        or int(composition.get("pass2_209", -1)) != 209
        or int(composition.get("kimi_c001", -1)) != 12
        or int(composition.get("kimi_c002_prefix", -1)) != 1
        or c002_rows != rows - 447
        or not 0 <= c002_rows <= 47
        or any(
            int(composition.get(name, -1)) != 0
            for name in ("gold_replay", "repair_conditioned", "reasoning_rows")
        )
        or dataset.get("heldout_overlap") != 0
        or dataset.get("heldout_175_model_visible") is not False
        or dataset.get("tests_model_visible") is not False
        or dataset.get("private_feedback_model_visible") is not False
        or dataset.get("repair_conditioned_prefixes_visible") is not False
        or dataset.get("reasoning_model_visible") is not False
        or dataset.get("gold_implementation_model_visible") is not False
        or dataset.get("known_contaminant_excluded") != KNOWN_TYPED_CONTAMINANT
        or dataset.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or dataset.get("task_id_deduplication")
        != "reject_any_duplicate_across_all_source_cohorts"
        or dataset.get("target_code_deduplication")
        != "none_retain_same_code_for_distinct_tasks"
        or int(dataset.get("equivalent_typed_source_groups", -1)) != 1
        or int(dataset.get("equivalent_typed_source_rows", -1)) != 2
        or not isinstance(cross, Mapping)
        or int(cross.get("rows", -1)) != 4
        or int(cross.get("passed", -1)) != 4
        or cross.get("tests_model_visible") is not False
        or not isinstance(verification, Mapping)
        or int(verification.get("rows", -1)) != rows
        or int(verification.get("passed", -1)) != rows
        or verification.get("tests_model_visible") is not False
        or verification.get("diagnostics_persisted") is not False
        or dataset.get("automatic_promotion_permitted") is not False
        or dataset.get("promotion_status")
        != "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
        or dataset.get("verpo_status") != "HOLD"
        or dataset.get("production_floor_eligible") is not True
    ):
        raise ValueError("typed folded-union dataset binding failed")
    _require_typed_schedule(dataset, rows=rows, label="typed folded-union")
    _require_typed_no_leakage_privacy(contract, label="typed folded-union")
    expected_updates = math.ceil(rows / 8)
    if (
        runtime.get("trainer_sha256") != FOLD_TRAINER_SHA256
        or warmstart.get("checkpoint_name") != "checkpoint-optstep-000348"
        or int(warmstart.get("update", -1)) != 348
        or warmstart.get("run_contract_sha256")
        != TYPED_SFT_RUN_CONTRACT_SHA256
        or warmstart.get("adapter_weights_sha256")
        != TYPED_SFT_ADAPTER_WEIGHTS_SHA256
        or warmstart.get("adapter_config_sha256") != TYPED_SFT_ADAPTER_CONFIG_SHA256
        or contract.get("warmstart_contract_schema") != TYPED_CONTRACT_SFT_RUN_SCHEMA
        or int(optimization.get("epochs", -1)) != 1
        or int(optimization.get("batch_size", -1)) != 1
        or int(optimization.get("gradient_accumulation", -1)) != 8
        or not math.isclose(
            float(optimization.get("learning_rate", -1.0)),
            5e-6,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(optimization.get("warmup_updates", -1)) != 0
        or int(optimization.get("seed", -1)) != 42
        or int(optimization.get("updates_per_epoch", -1)) != expected_updates
        or int(optimization.get("planned_updates", -1)) != expected_updates
        or lora.get("new_adapter_attached") is not False
        or lora.get("warmstart_weights_continued") is not True
        or contract.get("production_floor_eligible") is not True
    ):
        raise ValueError("typed folded-union lineage/optimizer binding failed")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )


def load_heldout_rows(
    *,
    dataset: str | Path,
    dataset_seal: str | Path,
    f2_jsonl: str | Path,
    f2_manifest: str | Path,
    limit: int = 0,
) -> tuple[list[EvaluationRow], dict[str, Any]]:
    dataset_path = Path(dataset).expanduser().resolve()
    seal_path = Path(dataset_seal).expanduser().resolve()
    f2_path = Path(f2_jsonl).expanduser().resolve()
    manifest_path = Path(f2_manifest).expanduser().resolve()
    _require_sha(dataset_path, DATASET_SHA256, "held-out dataset")
    _require_sha(seal_path, DATASET_SEAL_SHA256, "held-out seal")
    _require_sha(f2_path, F2_SHA256, "held-out F2")
    _require_sha(manifest_path, F2_MANIFEST_SHA256, "held-out F2 manifest")

    seal = _read_json(seal_path, "held-out seal")
    manifest = _read_json(manifest_path, "held-out F2 manifest")
    if (
        seal.get("schema") != "compact-public-private-join-seal-v1"
        or seal.get("heldout_measure_only") is not True
        or seal.get("training_allowed") is not False
        or seal.get("selected_role") != "measure"
        or int(seal.get("rows", -1)) != HELDOUT_ROWS
        or seal.get("output_sha256") != DATASET_SHA256
        or (seal.get("f2_output") or {}).get("sha256") != F2_SHA256
        or (seal.get("f2_manifest") or {}).get("sha256") != F2_MANIFEST_SHA256
        or seal.get("task_set_sha256") != TASK_SET_SHA256
        or seal.get("completion_attestation_id")
        != "per-run-256-bit-marker-exactly-once-v1"
    ):
        raise ValueError("held-out measure-only seal contract failed")
    if (
        manifest.get("schema") != "verified-api-readable-compact-v2"
        or int(manifest.get("rows", -1)) != HELDOUT_ROWS
        or (manifest.get("dataset") or {}).get("sha256") != DATASET_SHA256
        or (manifest.get("output") or {}).get("sha256") != F2_SHA256
        or manifest.get("task_set_sha256") != TASK_SET_SHA256
        or (manifest.get("invariants") or {}).get(
            "train_dev_representation_contract_identical"
        )
        is not True
    ):
        raise ValueError("held-out F2 manifest contract failed")

    dataset_rows = _read_jsonl(dataset_path, "held-out dataset")
    f2_rows = _read_jsonl(f2_path, "held-out F2")
    if len(dataset_rows) != HELDOUT_ROWS or len(f2_rows) != HELDOUT_ROWS:
        raise ValueError("held-out row count differs from the sealed 175")
    dataset_ids = [str(row.get("task_id") or "") for row in dataset_rows]
    f2_ids = [str(row.get("task_id") or "") for row in f2_rows]
    if (
        not all(dataset_ids)
        or len(set(dataset_ids)) != HELDOUT_ROWS
        or dataset_ids != f2_ids
        or canonical_sha256(dataset_ids) != seal.get("task_set_sha256")
    ):
        raise ValueError("held-out dataset/F2 task order or identity differs")

    rows: list[EvaluationRow] = []
    for task_id, f2_row in zip(dataset_ids, f2_rows, strict=True):
        source = build_encoder_source(f2_row, task_id)
        rows.append(
            EvaluationRow(
                task_id=task_id,
                source=source,
                source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )
        )
    if limit:
        if limit <= 0 or limit > len(rows):
            raise ValueError("limit lies outside the held-out row count")
        rows = rows[:limit]
    return rows, {
        "dataset": {"sha256": DATASET_SHA256, "rows": HELDOUT_ROWS},
        "dataset_seal": {"sha256": DATASET_SEAL_SHA256},
        "f2": {"sha256": F2_SHA256, "rows": HELDOUT_ROWS},
        "f2_manifest": {"sha256": F2_MANIFEST_SHA256},
        "task_set_sha256": TASK_SET_SHA256,
        "selected_rows": len(rows),
        "selected_ordered_task_ids_sha256": canonical_sha256(
            [row.task_id for row in rows]
        ),
        "model_visible_fields": ["F2.text"],
        "tests_serialized_to_model": False,
        "gold_targets_serialized_to_model": False,
    }


def _checkpoint_record(
    checkpoint: Path, arm: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json(
        checkpoint / "run_contract.json", "T5Gemma 2 adapter run contract"
    )
    stage_schema = str(contract.get("schema") or "")
    if stage_schema not in SUPPORTED_ADAPTER_RUN_SCHEMAS:
        raise ValueError(
            "checkpoint is not a supported native T5Gemma 2 adapter checkpoint"
        )
    if (
        contract.get("architecture") != "native_encoder_decoder"
        or contract.get("status") != "training"
    ):
        raise ValueError("T5Gemma 2 adapter checkpoint architecture/status differs")
    if stage_schema == TYPED_DIRECT_RS_SFT_PASS3_RUN_SCHEMA:
        _require_typed_pass3_contract(contract)
    if stage_schema == TYPED_FOLD_RS_SFT_UNION_RUN_SCHEMA:
        _require_typed_fold_contract(contract)
    if stage_schema == MIXED_RS_SFT_RUN_SCHEMA:
        privacy = contract.get("privacy")
        dataset = contract.get("dataset")
        if (
            not isinstance(privacy, Mapping)
            or privacy.get("heldout_overlap") != 0
            or privacy.get("heldout_content_model_visible") is not False
            or privacy.get("tests_model_visible") is not False
            or privacy.get("private_feedback_model_visible") is not False
            or not isinstance(dataset, Mapping)
            or dataset.get("schema") != "t5gemma2-mixed-rs-sft-dataset-v1"
            or dataset.get("heldout_overlap") != 0
        ):
            raise ValueError("mixed RS-SFT checkpoint privacy binding failed")
    if stage_schema == TYPED_CONTRACT_SFT_RUN_SCHEMA:
        dataset = contract.get("dataset")
        heldout = dataset.get("heldout") if isinstance(dataset, Mapping) else None
        exclusions = (
            dataset.get("training_exclusions")
            if isinstance(dataset, Mapping)
            else None
        )
        if (
            not isinstance(dataset, Mapping)
            or dataset.get("rows") != 2775
            or dataset.get("input_rows") != 2776
            or dataset.get("model_visible_fields")
            != ["opaque_typed_contract", "F2.text"]
            or not isinstance(heldout, Mapping)
            or heldout.get("model_visible") is not False
            or heldout.get("task_id_overlap") != 0
            or heldout.get("exact_gold_source_overlap") != 0
            or heldout.get("exact_acceptance_test_overlap") != 0
            or not isinstance(exclusions, Mapping)
            or exclusions.get("count") != 1
            or exclusions.get("task_ids") != ["sigless_6b1dd0c6b6fc"]
        ):
            raise ValueError("typed-contract checkpoint privacy binding failed")
    if stage_schema == TYPED_DIRECT_RS_SFT_RUN_SCHEMA:
        privacy = contract.get("privacy")
        dataset = contract.get("dataset")
        composition = (
            dataset.get("composition") if isinstance(dataset, Mapping) else None
        )
        verification = (
            dataset.get("full_acceptance_reverification")
            if isinstance(dataset, Mapping)
            else None
        )
        typed_train = (
            dataset.get("typed_train") if isinstance(dataset, Mapping) else None
        )
        exclusions = (
            typed_train.get("training_exclusions")
            if isinstance(typed_train, Mapping)
            else None
        )
        if (
            not isinstance(privacy, Mapping)
            or privacy.get("heldout_overlap") != 0
            or privacy.get("heldout_content_model_visible") is not False
            or privacy.get("tests_model_visible") is not False
            or privacy.get("private_feedback_model_visible") is not False
            or not isinstance(dataset, Mapping)
            or dataset.get("schema") != "t5gemma2-typed-direct-rs-sft-dataset-v1"
            or dataset.get("rows") != 225
            or dataset.get("heldout_overlap") != 0
            or dataset.get("known_contaminant_excluded")
            != "sigless_6b1dd0c6b6fc"
            or dataset.get("model_visible_fields")
            != ["opaque_typed_contract", "F2.text"]
            or dataset.get("tests_model_visible") is not False
            or dataset.get("private_feedback_model_visible") is not False
            or dataset.get("repair_conditioned_prefixes_visible") is not False
            or not isinstance(composition, Mapping)
            or composition.get("verified_direct") != 225
            or composition.get("local_student_direct") != 141
            or composition.get("external_teacher_direct") != 84
            or composition.get("repair_conditioned") != 0
            or composition.get("gold_replay") != 0
            or not isinstance(verification, Mapping)
            or verification.get("rows") != 225
            or verification.get("passed") != 225
            or verification.get("tests_model_visible") is not False
            or verification.get("diagnostics_persisted") is not False
            or not isinstance(exclusions, Mapping)
            or exclusions.get("count") != 1
            or exclusions.get("task_ids") != ["sigless_6b1dd0c6b6fc"]
        ):
            raise ValueError("typed direct RS-SFT checkpoint privacy binding failed")
    base = contract.get("base_model") or {}
    revision = str(base.get("resolved_commit") or base.get("requested_revision") or "")
    if (
        base.get("name") != MODEL_NAME
        or revision != MODEL_REVISION
        or base.get("is_encoder_decoder") is not True
    ):
        raise ValueError("SFT checkpoint does not bind the pinned T5Gemma 2 base")
    record: dict[str, Any] = {
        "name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "config_sha256": str(base.get("config_sha256") or ""),
        "arm": arm,
        "training_stage_schema": stage_schema,
        "production_floor_eligible": (
            contract.get("production_floor_eligible", True) is True
        ),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer" / "tokenizer.json"),
        "warmstart_contract_sha256": canonical_sha256(contract),
    }
    if arm == "sft":
        expected_targets = contract.get("lora", {}).get("targets")
        if not isinstance(expected_targets, list) or not expected_targets:
            raise ValueError("SFT checkpoint lacks exact LoRA targets")
        weighted_targets = _adapter_weight_target_modules(checkpoint)
        if weighted_targets != set(map(str, expected_targets)):
            raise ValueError("SFT adapter weights differ from its exact target set")
        record["adapter"] = {
            "adapter_config_sha256": sha256_file(
                checkpoint / "adapter" / "adapter_config.json"
            ),
            "adapter_weights_sha256": sha256_file(
                checkpoint / "adapter" / "adapter_model.safetensors"
            ),
            "run_contract_sha256": canonical_sha256(contract),
            "target_modules": len(expected_targets),
        }
    else:
        record["adapter"] = None
    return contract, record


def load_policy(
    *,
    checkpoint: Path,
    arm: str,
    bf16: bool,
    attn_implementation: str,
) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    contract, record = _checkpoint_record(checkpoint, arm)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "tokenizer",
        trust_remote_code=False,
        local_files_only=True,
    )
    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
    )
    # SFT seals the base config after enabling gradient checkpointing, which
    # disables both cache flags. Normalize to that sealed state before the
    # hash comparison; inference re-enables cache only after validation.
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    if hasattr(base.config, "decoder"):
        base.config.decoder.use_cache = False
    if canonical_sha256(base.config.to_dict()) != record["config_sha256"]:
        raise ValueError("loaded base config differs from the SFT contract")
    if arm == "sft":
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            base,
            checkpoint / "adapter",
            is_trainable=False,
            local_files_only=True,
        )
    else:
        model = base
    if not bool(getattr(model.config, "is_encoder_decoder", False)):
        raise ValueError("loaded policy is not encoder-decoder")
    model.to(torch.device("cuda"))
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if hasattr(model.config, "decoder"):
        model.config.decoder.use_cache = True
    return model, tokenizer, record


def sample_seed(seed: int, task_index: int, sample_index: int) -> int:
    if min(seed, task_index, sample_index) < 0:
        raise ValueError("sampling seed coordinates must be non-negative")
    return int(seed + task_index * 100_003 + sample_index)


def generate_candidate_batch(
    *,
    model: Any,
    tokenizer: Any,
    encoder_outputs: Any,
    attention_mask: torch.Tensor,
    decoder_start: int,
    pad_id: int,
    eos_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("generation batch count must be positive")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prefix = torch.tensor([[decoder_start]], dtype=torch.long, device="cuda")
    kwargs = _generation_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=pad_id,
        eos_token_ids=eos_ids,
    )
    kwargs["top_p"] = float(top_p)
    kwargs["num_return_sequences"] = int(count)
    with torch.no_grad():
        generated = model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=prefix,
            **kwargs,
        )
    results: list[dict[str, Any]] = []
    for batch_position, sequence in enumerate(generated.sequences.detach().cpu()):
        actions = normalize_generated_seq2seq_ids(
            sequence,
            decoder_prefix_ids=[decoder_start],
            eos_token_ids=eos_ids,
            pad_token_id=pad_id,
        )
        text = _decode_candidate(tokenizer, actions)
        eos_observed = actions[-1] in set(eos_ids)
        results.append(
            {
                "seed": seed,
                "batch_position": batch_position,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "action_tokens": len(actions),
                "eos_observed": eos_observed,
                "max_token_completion": (
                    not eos_observed and len(actions) >= max_new_tokens
                ),
            }
        )
    if len(results) != count:
        raise ValueError("generate returned a different number of sequences")
    return results


def _journal_state(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    rows: Sequence[EvaluationRow],
    num_samples: int,
) -> tuple[list[dict[str, Any]], bool]:
    if not events:
        return [], False
    header = events[0]
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("generation journal header differs from the exact run")
    terminals: list[dict[str, Any]] = []
    for event in events[1:]:
        if event.get("event") == "complete":
            if (
                len(terminals) != len(rows)
                or event.get("schema") != JOURNAL_SCHEMA
                or int(event.get("rows", -1)) != len(rows)
                or event.get("predictions_canonical_sha256")
                != canonical_sha256(
                    [
                        {
                            "id": row["task_id"],
                            "predictions": [
                                candidate["text"] for candidate in row["candidates"]
                            ],
                        }
                        for row in terminals
                    ]
                )
            ):
                raise ValueError("generation completion event is inconsistent")
            if event is not events[-1]:
                raise ValueError("events appear after generation completion")
            return terminals, True
        position = len(terminals)
        expected = rows[position] if position < len(rows) else None
        candidates = event.get("candidates")
        if (
            expected is None
            or event.get("event") != "task_terminal"
            or event.get("schema") != JOURNAL_SCHEMA
            or int(event.get("task_index", -1)) != position
            or event.get("task_id") != expected.task_id
            or event.get("source_sha256") != expected.source_sha256
            or not isinstance(candidates, list)
            or len(candidates) != num_samples
        ):
            raise ValueError("generation task terminal differs from the schedule")
        for sample_index, candidate in enumerate(candidates):
            if (
                not isinstance(candidate, dict)
                or int(candidate.get("sample_index", -1)) != sample_index
                or not isinstance(candidate.get("text"), str)
                or candidate.get("text_sha256")
                != hashlib.sha256(candidate["text"].encode("utf-8")).hexdigest()
                or type(candidate.get("action_tokens")) is not int
                or int(candidate["action_tokens"]) <= 0
                or type(candidate.get("eos_observed")) is not bool
                or type(candidate.get("max_token_completion")) is not bool
            ):
                raise ValueError("generation candidate journal record is invalid")
        terminals.append(dict(event))
    return terminals, False


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("T5Gemma pass@k inference requires CUDA")
    rows, data_record = load_heldout_rows(
        dataset=args.dataset,
        dataset_seal=args.dataset_seal,
        f2_jsonl=args.f2_jsonl,
        f2_manifest=args.f2_manifest,
        limit=args.limit,
    )
    checkpoint = Path(args.sft_checkpoint).expanduser().resolve()
    checkpoint_contract, model_record = _checkpoint_record(checkpoint, args.arm)
    output_path = Path(args.output).expanduser().resolve()
    provenance_path = Path(str(output_path) + ".provenance.json")
    journal_path = (
        Path(args.journal or (str(output_path) + ".generation.journal.jsonl"))
        .expanduser()
        .resolve()
    )
    contract = {
        "schema": INFERENCE_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "arm": args.arm,
        "model": model_record,
        "heldout": data_record,
        "sampling": {
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": 0,
            "max_source_tokens": args.max_source_tokens,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "seed_policy": "seed+task_index*100003+batch_start",
            "generation_batch_size": args.generation_batch_size,
            "decoder_prefix_is_not_output": True,
            "sampled_eos_retained": True,
            "fabricated_eos": False,
        },
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or ""),
            "bf16": args.bf16,
            "attn_implementation": args.attn_implementation,
        },
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "targets_exposed_to_model": False,
        "source_truncation": False,
    }
    events = load_journal(journal_path)
    if not events:
        if output_path.exists() or provenance_path.exists():
            raise ValueError("published output exists without its generation journal")
        append_event(
            journal_path,
            {
                "event": "header",
                "schema": JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
        events = load_journal(journal_path)
    terminals, complete = _journal_state(
        events,
        contract=contract,
        rows=rows,
        num_samples=args.num_samples,
    )

    if not complete:
        model, tokenizer, loaded_record = load_policy(
            checkpoint=checkpoint,
            arm=args.arm,
            bf16=args.bf16,
            attn_implementation=args.attn_implementation,
        )
        if loaded_record != model_record:
            raise ValueError("loaded model record differs from preflight")
        decoder_start, pad_id, eos_ids = _decoder_special_ids(model, tokenizer)
        device = torch.device("cuda")
        for task_index in range(len(terminals), len(rows)):
            row = rows[task_index]
            input_ids, attention_mask = _encode_source(
                tokenizer,
                row.source,
                max_source_tokens=args.max_source_tokens,
                device=device,
            )
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            candidates: list[dict[str, Any]] = []
            for batch_start in range(0, args.num_samples, args.generation_batch_size):
                count = min(
                    args.generation_batch_size,
                    args.num_samples - batch_start,
                )
                generated_batch = generate_candidate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_start=decoder_start,
                    pad_id=pad_id,
                    eos_ids=eos_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=sample_seed(args.seed, task_index, batch_start),
                    count=count,
                )
                for batch_position, candidate in enumerate(generated_batch):
                    candidates.append(
                        {
                            "sample_index": batch_start + batch_position,
                            **candidate,
                        }
                    )
            terminal = append_event(
                journal_path,
                {
                    "event": "task_terminal",
                    "schema": JOURNAL_SCHEMA,
                    "task_index": task_index,
                    "task_id": row.task_id,
                    "source_sha256": row.source_sha256,
                    "encoder_tokens": int(input_ids.size(1)),
                    "candidates": candidates,
                },
            )
            terminals.append(terminal)
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "task": task_index + 1,
                        "tasks": len(rows),
                        "task_id": row.task_id,
                        "max_token_completions": sum(
                            candidate["max_token_completion"]
                            for candidate in candidates
                        ),
                        "mean_action_tokens": sum(
                            candidate["action_tokens"] for candidate in candidates
                        )
                        / len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        predictions = [
            {
                "id": terminal["task_id"],
                "predictions": [
                    candidate["text"] for candidate in terminal["candidates"]
                ],
            }
            for terminal in terminals
        ]
        append_event(
            journal_path,
            {
                "event": "complete",
                "schema": JOURNAL_SCHEMA,
                "rows": len(rows),
                "predictions_canonical_sha256": canonical_sha256(predictions),
            },
        )
        events = load_journal(journal_path)
        terminals, complete = _journal_state(
            events,
            contract=contract,
            rows=rows,
            num_samples=args.num_samples,
        )
    if not complete:
        raise RuntimeError("generation journal did not reach completion")

    predictions = [
        {
            "id": terminal["task_id"],
            "predictions": [candidate["text"] for candidate in terminal["candidates"]],
        }
        for terminal in terminals
    ]
    require_exact_or_write(output_path, predictions)
    capped = sum(
        candidate["max_token_completion"]
        for terminal in terminals
        for candidate in terminal["candidates"]
    )
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "architecture": "native_t5gemma2_encoder_decoder",
        "arm": args.arm,
        "output_sha256": sha256_file(output_path),
        "num_rows": len(predictions),
        "num_samples": args.num_samples,
        "model": model_record,
        "heldout": data_record,
        "sampling": contract["sampling"],
        "max_token_completions": capped,
        "generation_journal": journal_record(journal_path),
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "targets_exposed_to_model": False,
        "sft_checkpoint_contract_sha256": canonical_sha256(checkpoint_contract),
    }
    require_exact_or_write(provenance_path, provenance)
    result = {
        "arm": args.arm,
        "rows": len(predictions),
        "samples": len(predictions) * args.num_samples,
        "max_token_completions": capped,
        "output": str(output_path),
        "output_sha256": provenance["output_sha256"],
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_seal", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--f2_manifest", required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--arm", choices=["base", "sft"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", default="")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--generation_batch_size", type=int, default=10)
    parser.add_argument("--max_source_tokens", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--attn_implementation", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(argv)
    if args.num_samples <= 0 or args.generation_batch_size <= 0:
        parser.error("--num_samples and --generation_batch_size must be positive")
    if args.generation_batch_size > args.num_samples:
        parser.error("--generation_batch_size cannot exceed --num_samples")
    if args.max_source_tokens <= 0 or args.max_new_tokens <= 0:
        parser.error("token limits must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0.0 < args.top_p <= 1.0:
        parser.error("--top_p must be in (0, 1]")
    if args.seed < 0 or args.limit < 0:
        parser.error("seed and limit must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
