#!/usr/bin/env python3
"""Sealed typed-contract inference adapter for amended fold+gold Arm C2 only.

The shared measurement runner remains byte-identical for the active Arm-B
evaluation.  This adapter admits exactly the Arm-C schema after validating its
916-row replay contract, then delegates generation to that shared runner.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation import t5gemma2_f2_passk_inference as base
from scripts.evaluation import t5gemma2_measurement_audit_inference as measurement
from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file
from scripts.training import t5gemma2_typed_contract_sft as typed_sft
from scripts.training import t5gemma2_typed_fold_gold_replay_v1 as arm_c
from scripts.training import t5gemma2_typed_fold_rs_sft_union_v1 as arm_b


def _require_arm_c_contract(contract: Mapping[str, Any]) -> None:
    dataset = contract.get("dataset")
    warmstart = contract.get("warmstart")
    optimization = contract.get("optimization")
    runtime = contract.get("runtime")
    lora = contract.get("lora")
    privacy = contract.get("privacy")
    checkpointing = contract.get("checkpointing")
    if not all(
        isinstance(value, Mapping)
        for value in (
            dataset,
            warmstart,
            optimization,
            runtime,
            lora,
            privacy,
            checkpointing,
        )
    ):
        raise ValueError("typed Arm C contract is structurally incomplete")
    composition = dataset.get("composition")
    direct = dataset.get("direct_union")
    replay = dataset.get("gold_replay")
    amendment = dataset.get("arm_c2_amendment")
    production_admissibility = (
        replay.get("production_admissibility")
        if isinstance(replay, Mapping)
        else None
    )
    verification = dataset.get("full_acceptance_reverification")
    replay_verification = dataset.get("gold_replay_acceptance_reverification")
    cross = dataset.get("equivalent_typed_source_cross_acceptance")
    typed_train = dataset.get("typed_train")
    exclusions = (
        typed_train.get("training_exclusions")
        if isinstance(typed_train, Mapping)
        else None
    )
    heldout = typed_train.get("heldout") if isinstance(typed_train, Mapping) else None
    schedule = dataset.get("schedule")
    if (
        contract.get("schema") != arm_c.RUN_SCHEMA
        or contract.get("status") != "training"
        or contract.get("architecture") != "native_encoder_decoder"
        or dataset.get("schema") != arm_c.DATASET_SCHEMA
        or dataset.get("arm")
        != "C2_fold_plus_production_eligible_typed_gold_replay_1to1"
        or dataset.get("estimand")
        != "practical_B_plus_1to1_production_eligible_typed_gold_replay_recipe"
        or dataset.get("original_arm_c_status")
        != "TERMINATED_PREFLIGHT_INFEASIBLE"
        or not isinstance(amendment, Mapping)
        or amendment.get("document_sha256") != arm_c.ARM_C2_AMENDMENT_SHA256
        or amendment.get("seal_sha256") != arm_c.ARM_C2_AMENDMENT_SEAL_SHA256
        or amendment.get("fixed_before_optimizer_step") is not True
        or int(amendment.get("original_arm_c_optimizer_steps", -1)) != 0
        or dataset.get("pure_gold_content_causal_claim_permitted") is not False
        or dataset.get("mechanism_control_required")
        != "duplicated_direct_GA16_without_gold_replay"
        or dataset.get("fresh_branch_from") != "typed_sft_optstep348"
        or int(dataset.get("rows", -1)) != arm_c.EXPECTED_TOTAL_ROWS
        or not isinstance(composition, Mapping)
        or int(composition.get("verified_direct", -1)) != arm_c.EXPECTED_DIRECT_ROWS
        or int(composition.get("gold_replay", -1)) != arm_c.EXPECTED_GOLD_REPLAY_ROWS
        or int(composition.get("repair_conditioned", -1)) != 0
        or int(composition.get("reasoning_rows", -1)) != 0
        or not isinstance(direct, Mapping)
        or int(direct.get("rows", -1)) != arm_c.EXPECTED_DIRECT_ROWS
        or direct.get("dataset_schema") != arm_b.DATASET_SCHEMA
        or not isinstance(direct.get("dataset_manifest"), Mapping)
        or direct["dataset_manifest"].get("rows") != arm_c.EXPECTED_DIRECT_ROWS
        or not isinstance(replay, Mapping)
        or int(replay.get("selected_rows", -1)) != arm_c.EXPECTED_GOLD_REPLAY_ROWS
        or not math.isclose(
            float(replay.get("ratio_to_direct", -1.0)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or replay.get("direct_tasks_excluded") is not True
        or replay.get("direct_typed_source_sha256s_excluded") is not True
        or replay.get("unique_typed_source_sha256_within_replay") is not True
        or replay.get("selection")
        != (
            "corpus_wide_production_verifier_eligibility_then_"
            "deterministic_seeded_hash_order"
        )
        or int(replay.get("seed", -1)) != 42
        or replay.get("ranking_schema") != arm_c.REPLAY_RANKING_SCHEMA
        or int(replay.get("eligible_rows_after_direct_task_and_source_exclusion", -1))
        != arm_c.EXPECTED_REPLAY_ELIGIBLE_ROWS
        or int(replay.get("eligible_unique_typed_sources", -1))
        != arm_c.EXPECTED_REPLAY_ELIGIBLE_UNIQUE_SOURCES
        or int(replay.get("production_verifier_eligible_unique_typed_sources", -1))
        != arm_c.EXPECTED_REPLAY_PRODUCTION_ELIGIBLE_UNIQUE_SOURCES
        or int(replay.get("direct_unique_typed_sources", -1))
        != arm_c.EXPECTED_DIRECT_SOURCE_SHA256S
        or replay.get("selection_key_fields")
        != ["schema", "seed", "kind", "task_id", "source_sha256"]
        or replay.get("selected_task_ids_sha256")
        != arm_c.EXPECTED_REPLAY_TASK_IDS_SHA256
        or replay.get("selected_source_sha256s_sha256")
        != arm_c.EXPECTED_REPLAY_SOURCE_SHA256S_SHA256
        or replay.get("selected_target_sha256s_sha256")
        != arm_c.EXPECTED_REPLAY_TARGET_SHA256S_SHA256
        or replay.get("selection_fixed_before_arm_b_result") is not False
        or replay.get("base_ranking_fixed_before_arm_b_result") is not True
        or replay.get("selection_amended_after_arm_b_result") is not True
        or not isinstance(production_admissibility, Mapping)
        or production_admissibility.get("scope")
        != "all_post_direct_disjoint_source_unique_replay_candidates"
        or production_admissibility.get("verifier")
        != "exact_production_complete_acceptance_verifier"
        or int(production_admissibility.get("candidates_checked", -1))
        != arm_c.EXPECTED_REPLAY_ELIGIBLE_UNIQUE_SOURCES
        or int(production_admissibility.get("eligible", -1))
        != arm_c.EXPECTED_REPLAY_PRODUCTION_ELIGIBLE_UNIQUE_SOURCES
        or int(production_admissibility.get("rejected", -1))
        != len(arm_c.EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS)
        or production_admissibility.get("rejected_task_ids_sha256")
        != arm_c.EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS_SHA256
        or production_admissibility.get("tests_model_visible") is not False
        or production_admissibility.get("diagnostics_persisted") is not False
        or production_admissibility.get("uses_arm_b_predictions_or_scores")
        is not False
        or production_admissibility.get(
            "selection_amendment_fixed_after_arm_b_result"
        )
        is not True
        or dataset.get("task_identity_policy")
        != "direct_and_gold_replay_are_globally_task_disjoint"
        or dataset.get("typed_source_identity_policy")
        != "replay_excludes_all_direct_source_sha256s_and_is_unique_within_replay"
        or dataset.get("model_visible_encoder_fields")
        != ["opaque_typed_contract", "F2.text"]
        or dataset.get("decoder_supervision_fields")
        != ["verified_direct_dart_source", "gold_replay_dart_source"]
        or dataset.get("gold_implementation_encoder_visible") is not False
        or dataset.get("gold_replay_decoder_supervision") is not True
        or dataset.get("tests_model_visible") is not False
        or dataset.get("private_feedback_model_visible") is not False
        or dataset.get("repair_conditioned_prefixes_visible") is not False
        or dataset.get("reasoning_model_visible") is not False
        or dataset.get("heldout_overlap") != 0
        or dataset.get("heldout_175_model_visible") is not False
        or dataset.get("known_contaminant_excluded") != base.KNOWN_TYPED_CONTAMINANT
        or not isinstance(verification, Mapping)
        or int(verification.get("rows", -1)) != arm_c.EXPECTED_TOTAL_ROWS
        or int(verification.get("passed", -1)) != arm_c.EXPECTED_TOTAL_ROWS
        or verification.get("tests_model_visible") is not False
        or verification.get("diagnostics_persisted") is not False
        or not isinstance(replay_verification, Mapping)
        or int(replay_verification.get("rows", -1)) != arm_c.EXPECTED_GOLD_REPLAY_ROWS
        or int(replay_verification.get("passed", -1)) != arm_c.EXPECTED_GOLD_REPLAY_ROWS
        or not isinstance(cross, Mapping)
        or int(cross.get("passed", -1)) != int(cross.get("rows", -2))
        or cross.get("tests_model_visible") is not False
        or not isinstance(typed_train, Mapping)
        or typed_train.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or not isinstance(exclusions, Mapping)
        or base.KNOWN_TYPED_CONTAMINANT not in (exclusions.get("task_ids") or [])
        or not isinstance(heldout, Mapping)
        or heldout.get("task_id_overlap") != 0
        or heldout.get("exact_gold_source_overlap") != 0
        or heldout.get("exact_acceptance_test_overlap") != 0
        or heldout.get("model_visible") is not False
        or dataset.get("automatic_promotion_permitted") is not False
        or dataset.get("promotion_status") != "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
        or dataset.get("verpo_status") != "HOLD"
        or dataset.get("production_floor_eligible") is not True
        or not isinstance(schedule, list)
        or len(schedule) != arm_c.EXPECTED_TOTAL_ROWS
    ):
        raise ValueError("typed Arm C dataset/privacy binding failed")

    task_ids = [str(row.get("source_task_id") or "") for row in schedule]
    pair_ids = [str(row.get("pair_id") or "") for row in schedule]
    direct_schedule_rows = sum(row.get("kind") == "verified_direct" for row in schedule)
    replay_schedule_rows = sum(row.get("kind") == "gold_replay" for row in schedule)
    source_by_kind = {
        "verified_direct": {
            str(row.get("source_sha256") or "")
            for row in schedule
            if row.get("kind") == "verified_direct"
        },
        "gold_replay": {
            str(row.get("source_sha256") or "")
            for row in schedule
            if row.get("kind") == "gold_replay"
        },
    }
    if (
        not all(task_ids)
        or len(set(task_ids)) != arm_c.EXPECTED_TOTAL_ROWS
        or not all(pair_ids)
        or len(set(pair_ids)) != arm_c.EXPECTED_TOTAL_ROWS
        or base.KNOWN_TYPED_CONTAMINANT in task_ids
        or direct_schedule_rows != arm_c.EXPECTED_DIRECT_ROWS
        or replay_schedule_rows != arm_c.EXPECTED_GOLD_REPLAY_ROWS
        or len(source_by_kind["gold_replay"]) != arm_c.EXPECTED_GOLD_REPLAY_ROWS
        or source_by_kind["verified_direct"] & source_by_kind["gold_replay"]
        or dataset.get("schedule_sha256") != canonical_sha256(schedule)
        or dataset.get("task_ids_sha256") != canonical_sha256(task_ids)
        or dataset.get("pair_ids_sha256") != canonical_sha256(pair_ids)
    ):
        raise ValueError("typed Arm C schedule identity differs")

    if (
        runtime.get("trainer_sha256") != sha256_file(Path(arm_c.__file__).resolve())
        or runtime.get("fold_union_builder_sha256")
        != sha256_file(Path(arm_b.__file__).resolve())
        or runtime.get("typed_source_builder_sha256")
        != sha256_file(Path(typed_sft.__file__).resolve())
        or runtime.get("trainer_profile")
        != "typed_fold_plus_production_eligible_gold_replay_arm_c2_v2"
        or warmstart.get("checkpoint_name") != "checkpoint-optstep-000348"
        or int(warmstart.get("update", -1)) != 348
        or warmstart.get("run_contract_sha256") != base.TYPED_SFT_RUN_CONTRACT_SHA256
        or warmstart.get("adapter_weights_sha256")
        != base.TYPED_SFT_ADAPTER_WEIGHTS_SHA256
        or warmstart.get("adapter_config_sha256")
        != base.TYPED_SFT_ADAPTER_CONFIG_SHA256
        or contract.get("warmstart_contract_schema")
        != base.TYPED_CONTRACT_SFT_RUN_SCHEMA
        or int(optimization.get("epochs", -1)) != 1
        or int(optimization.get("batch_size", -1)) != 1
        or int(optimization.get("gradient_accumulation", -1)) != 16
        or not math.isclose(
            float(optimization.get("learning_rate", -1.0)),
            5e-6,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(optimization.get("warmup_updates", -1)) != 0
        or int(optimization.get("updates_per_epoch", -1))
        != arm_c.EXPECTED_PLANNED_UPDATES
        or int(optimization.get("planned_updates", -1))
        != arm_c.EXPECTED_PLANNED_UPDATES
        or int(optimization.get("seed", -1)) != 42
        or lora.get("new_adapter_attached") is not False
        or lora.get("warmstart_weights_continued") is not True
        or privacy.get("heldout_overlap") != 0
        or privacy.get("heldout_content_model_visible") is not False
        or privacy.get("tests_model_visible") is not False
        or privacy.get("private_feedback_model_visible") is not False
        or privacy.get("reasoning_persisted") is not False
        or int(checkpointing.get("interval", -1)) != arm_c.EXPECTED_CHECKPOINT_INTERVAL
        or contract.get("production_floor_eligible") is not True
    ):
        raise ValueError("typed Arm C lineage/optimizer binding failed")


def run(args: Any) -> dict[str, Any]:
    checkpoint = Path(args.sft_checkpoint).expanduser().resolve()
    contract = base._read_json(  # noqa: SLF001
        checkpoint / "run_contract.json", "typed Arm C run contract"
    )
    _require_arm_c_contract(contract)
    original_supported = base.SUPPORTED_ADAPTER_RUN_SCHEMAS
    original_checkpoint_record = base._checkpoint_record  # noqa: SLF001

    def guarded_checkpoint_record(path: Path, arm: str):
        observed = base._read_json(  # noqa: SLF001
            path / "run_contract.json", "typed Arm C run contract"
        )
        _require_arm_c_contract(observed)
        return original_checkpoint_record(path, arm)

    base.SUPPORTED_ADAPTER_RUN_SCHEMAS = frozenset(
        set(original_supported) | {arm_c.RUN_SCHEMA}
    )
    base._checkpoint_record = guarded_checkpoint_record  # type: ignore[assignment]  # noqa: SLF001
    try:
        return measurement.run(args)
    finally:
        base._checkpoint_record = original_checkpoint_record  # type: ignore[assignment]  # noqa: SLF001
        base.SUPPORTED_ADAPTER_RUN_SCHEMAS = original_supported


def main(argv: Sequence[str] | None = None) -> int:
    run(measurement.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
