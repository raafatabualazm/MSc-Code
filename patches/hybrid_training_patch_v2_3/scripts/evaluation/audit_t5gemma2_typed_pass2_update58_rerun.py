#!/usr/bin/env python3
"""Audit a current-stack update58 rerun against the completed typed pass-2 arm.

``pass2-arm`` is an ordered preflight gate: it validates the completed pass-2
K=10 generation and both full-175 and clean-174 scores before update58 is
loaded. ``compare`` validates a fresh update58 rerun, revalidates pass-2, seals
their exact inference/scoring-code identity, and emits paired metrics.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.evaluation import audit_t5gemma2_typed_rs_sft as matched
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_typed_direct_rs_sft_pass2 as pass2_profile


PASS2_ARM_AUDIT_SCHEMA = "t5gemma2-typed-pass2-current-stack-arm-audit-v1"
COMPARISON_SCHEMA = (
    "t5gemma2-typed-pass2-vs-update58-current-stack-matched-audit-v1"
)
PASS1_RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-run-v1"
PASS1_DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-dataset-v1"
PASS2_RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-run-v1"
PASS2_DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-dataset-v1"
PASS2_TRAINING_AUDIT_SCHEMA = (
    "t5gemma2-typed-direct-rs-sft-pass2-training-audit-v1"
)
EXPECTED_ROWS = 175
EXPECTED_CLEAN_ROWS = 174
EXPECTED_K = 10
EXPECTED_EXCLUDED = ("sigless_8bf7f40ca356",)
EXPECTED_CONTAMINANT = "sigless_6b1dd0c6b6fc"
EXPECTED_PASS2_TRAINER_SHA256 = sha256_file(Path(pass2_profile.__file__).resolve())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _validate_pass2_checkpoint(
    contract_path: Path, result_path: Path
) -> dict[str, Any]:
    record = matched._checkpoint_paths_record(contract_path, result_path)  # noqa: SLF001
    contract = record["contract"]
    result = record["result"]
    dataset = contract.get("dataset")
    composition = dataset.get("composition") if isinstance(dataset, Mapping) else None
    verification = (
        dataset.get("full_acceptance_reverification")
        if isinstance(dataset, Mapping)
        else None
    )
    optimization = contract.get("optimization")
    privacy = contract.get("privacy")
    runtime = contract.get("runtime")
    warmstart = contract.get("warmstart")
    lora = contract.get("lora")
    _require(
        contract.get("schema") == PASS2_RUN_SCHEMA
        and contract.get("status") == "training"
        and contract.get("architecture") == "native_encoder_decoder"
        and isinstance(runtime, Mapping)
        and runtime.get("trainer_sha256") == EXPECTED_PASS2_TRAINER_SHA256
        and runtime.get("trainer_profile")
        == "typed_direct_only_rs_sft_pass2_local190_plus_dual_api"
        and isinstance(optimization, Mapping)
        and optimization.get("epochs") == 2
        and optimization.get("batch_size") == 1
        and optimization.get("gradient_accumulation") == 8
        and optimization.get("updates_per_epoch") == 27
        and optimization.get("planned_updates") == 54
        and optimization.get("learning_rate") == 0.00002
        and optimization.get("warmup_updates") == 0
        and optimization.get("seed") == 42
        and isinstance(dataset, Mapping)
        and dataset.get("schema") == PASS2_DATASET_SCHEMA
        and dataset.get("rows") == 209
        and dataset.get("heldout_overlap") == 0
        and dataset.get("known_contaminant_excluded") == EXPECTED_CONTAMINANT
        and dataset.get("model_visible_fields")
        == ["opaque_typed_contract", "F2.text"]
        and dataset.get("tests_model_visible") is False
        and dataset.get("private_feedback_model_visible") is False
        and dataset.get("repair_conditioned_prefixes_visible") is False
        and dataset.get("reasoning_model_visible") is False
        and isinstance(composition, Mapping)
        and composition.get("verified_direct") == 209
        and composition.get("local_student_new") == 190
        and composition.get("external_teacher_new") == 19
        and composition.get("prior_225_replay") == 0
        and composition.get("gold_replay") == 0
        and composition.get("repair_conditioned") == 0
        and composition.get("reasoning_rows") == 0
        and composition.get("gold_source_replay") == 0
        and isinstance(verification, Mapping)
        and verification.get("rows") == 209
        and verification.get("passed") == 209
        and verification.get("stability_runs") == 2
        and verification.get("timeout_seconds") == 30
        and verification.get("diagnostics_persisted") is False
        and verification.get("tests_model_visible") is False
        and isinstance(warmstart, Mapping)
        and warmstart.get("checkpoint_name") == "checkpoint-optstep-000058"
        and warmstart.get("update") == 58
        and contract.get("warmstart_contract_schema") == PASS1_RUN_SCHEMA
        and isinstance(lora, Mapping)
        and lora.get("new_adapter_attached") is False
        and lora.get("warmstart_weights_continued") is True
        and lora.get("encoder_and_decoder_trainable") is True
        and lora.get("vision_trainable") is False
        and isinstance(privacy, Mapping)
        and privacy.get("heldout_overlap") == 0
        and privacy.get("heldout_content_model_visible") is False
        and privacy.get("tests_model_visible") is False
        and privacy.get("private_feedback_model_visible") is False
        and privacy.get("reasoning_persisted") is False
        and result.get("schema") == PASS2_RUN_SCHEMA
        and result.get("status") == "complete"
        and result.get("updates") == 54
        and result.get("planned_updates") == 54
        and result.get("rows") == 209
        and result.get("latest_checkpoint") == "checkpoint-optstep-000054"
        and result.get("production_floor_eligible") is True,
        "pass-2 checkpoint/result contract differs",
    )
    return record


def _validate_update58_checkpoint(
    contract_path: Path, result_path: Path
) -> dict[str, Any]:
    record = matched._checkpoint_paths_record(contract_path, result_path)  # noqa: SLF001
    contract = record["contract"]
    result = record["result"]
    dataset = contract.get("dataset")
    composition = dataset.get("composition") if isinstance(dataset, Mapping) else None
    verification = (
        dataset.get("full_acceptance_reverification")
        if isinstance(dataset, Mapping)
        else None
    )
    optimization = contract.get("optimization")
    privacy = contract.get("privacy")
    _require(
        contract.get("schema") == PASS1_RUN_SCHEMA
        and contract.get("status") == "training"
        and contract.get("architecture") == "native_encoder_decoder"
        and isinstance(optimization, Mapping)
        and optimization.get("epochs") == 2
        and optimization.get("planned_updates") == 58
        and optimization.get("gradient_accumulation") == 8
        and optimization.get("learning_rate") == 0.00002
        and optimization.get("warmup_updates") == 0
        and optimization.get("seed") == 42
        and isinstance(dataset, Mapping)
        and dataset.get("schema") == PASS1_DATASET_SCHEMA
        and dataset.get("rows") == 225
        and dataset.get("heldout_overlap") == 0
        and dataset.get("known_contaminant_excluded") == EXPECTED_CONTAMINANT
        and dataset.get("model_visible_fields")
        == ["opaque_typed_contract", "F2.text"]
        and dataset.get("tests_model_visible") is False
        and dataset.get("private_feedback_model_visible") is False
        and dataset.get("repair_conditioned_prefixes_visible") is False
        and isinstance(composition, Mapping)
        and composition.get("verified_direct") == 225
        and composition.get("local_student_direct") == 141
        and composition.get("external_teacher_direct") == 84
        and composition.get("repair_conditioned") == 0
        and composition.get("gold_replay") == 0
        and isinstance(verification, Mapping)
        and verification.get("rows") == 225
        and verification.get("passed") == 225
        and verification.get("tests_model_visible") is False
        and verification.get("diagnostics_persisted") is False
        and isinstance(privacy, Mapping)
        and privacy.get("heldout_overlap") == 0
        and privacy.get("heldout_content_model_visible") is False
        and privacy.get("tests_model_visible") is False
        and privacy.get("private_feedback_model_visible") is False
        and result.get("schema") == PASS1_RUN_SCHEMA
        and result.get("status") == "complete"
        and result.get("updates") == 58
        and result.get("planned_updates") == 58
        and result.get("rows") == 225
        and result.get("latest_checkpoint") == "checkpoint-optstep-000058",
        "update58 checkpoint/result contract differs",
    )
    return record


def _validate_training_audit(
    path: Path, expected_sha256: str, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    _require(_is_sha256(expected_sha256), "pass-2 training audit SHA is malformed")
    _require(sha256_file(path) == expected_sha256, "pass-2 training audit SHA differs")
    value = _read_object(path, "pass-2 training audit")
    composition = value.get("composition")
    checkpoint_record = value.get("checkpoint")
    artifacts = value.get("artifacts")
    _require(
        value.get("schema") == PASS2_TRAINING_AUDIT_SCHEMA
        and value.get("status") == "pass"
        and value.get("contract", {}).get("rows") == 209
        and value.get("contract", {}).get("local_rows") == 190
        and value.get("contract", {}).get("api_rows") == 19
        and value.get("contract", {}).get("planned_updates") == 54
        and value.get("contract", {}).get("warmstart_update") == 58
        and isinstance(composition, Mapping)
        and composition.get("rows") == 209
        and composition.get("local_student_new") == 190
        and composition.get("external_teacher_new") == 19
        and composition.get("prior_225_replay") == 0
        and composition.get("gold_replay") == 0
        and composition.get("heldout_overlap") == 0
        and isinstance(checkpoint_record, Mapping)
        and checkpoint_record.get("name") == "checkpoint-optstep-000054"
        and checkpoint_record.get("update") == 54
        and checkpoint_record.get("run_contract_canonical_sha256")
        == checkpoint["canonical_sha256"]
        and isinstance(artifacts, Mapping)
        and artifacts.get("checkpoint_contract_sha256")
        == checkpoint["file_sha256"]
        and artifacts.get("result_sha256") == checkpoint["result_sha256"],
        "pass-2 training audit contract differs",
    )
    return value


def _validate_arm(
    *,
    label: str,
    predictions: Path,
    full_score: Path,
    clean_score: Path,
    checkpoint: Mapping[str, Any],
    evaluation: Path,
    evaluator: Path,
    wrapper: Path,
    base_inference: Path,
) -> dict[str, Any]:
    arm = matched._validate_arm(  # noqa: SLF001
        label=label,
        prediction_path=predictions,
        full_score_path=full_score,
        clean_score_path=clean_score,
        checkpoint=checkpoint,
        evaluation_path=evaluation,
        evaluator_path=evaluator,
        expected_rows=EXPECTED_ROWS,
        expected_k=EXPECTED_K,
        expected_sampling=matched.EXPECTED_SAMPLING,
        expected_excluded=EXPECTED_EXCLUDED,
    )
    header = arm["generation"]["header_contract"]
    _require(
        header.get("script_sha256") == sha256_file(wrapper)
        and header.get("base_inference_script_sha256")
        == sha256_file(base_inference),
        f"{label} was not generated by the supplied current inference stack",
    )
    return arm


def _arm_report(
    arm: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    header = arm["generation"]["header_contract"]
    return {
        "checkpoint": {
            "schema": checkpoint["contract"]["schema"],
            "contract_file_sha256": checkpoint["file_sha256"],
            "contract_canonical_sha256": checkpoint["canonical_sha256"],
            "result_sha256": checkpoint["result_sha256"],
        },
        "artifacts": matched._artifact_hashes(arm),  # noqa: SLF001
        "metrics": {
            "full175": {
                name: matched._metric(count, EXPECTED_ROWS)  # noqa: SLF001
                for name, count in arm["full"]["counts"].items()
            },
            "clean174": {
                name: matched._metric(count, EXPECTED_CLEAN_ROWS)  # noqa: SLF001
                for name, count in arm["clean"]["counts"].items()
            },
        },
        "max_token_completions": arm["generation"]["max_token_completions"],
        "inference_code": {
            "wrapper_sha256": header["script_sha256"],
            "base_inference_sha256": header["base_inference_script_sha256"],
        },
    }


def _paired_metric(
    task_order: Sequence[str],
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    source_metric: str,
) -> dict[str, Any]:
    before_positive: list[str] = []
    after_positive: list[str] = []
    gains: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    for task_id in task_order:
        left = bool(before[task_id][source_metric])
        right = bool(after[task_id][source_metric])
        if left:
            before_positive.append(task_id)
        if right:
            after_positive.append(task_id)
        if right and not left:
            gains.append(task_id)
        elif left and not right:
            losses.append(task_id)
        else:
            ties.append(task_id)
    return {
        "source_metric": source_metric,
        "tasks": len(task_order),
        "update58_before": matched._hash_list(before_positive),  # noqa: SLF001
        "pass2_after": matched._hash_list(after_positive),  # noqa: SLF001
        "absolute_count_delta": len(after_positive) - len(before_positive),
        "rate_delta": (len(after_positive) - len(before_positive)) / len(task_order),
        "gains": matched._hash_list(gains),  # noqa: SLF001
        "losses": matched._hash_list(losses),  # noqa: SLF001
        "ties": matched._hash_list(ties),  # noqa: SLF001
        "discordant_tasks": len(gains) + len(losses),
        "exact_two_sided_sign_mcnemar_p": matched._exact_two_sided_sign_mcnemar(  # noqa: SLF001
            len(gains), len(losses)
        ),
    }


def _comparison(
    task_order: Sequence[str],
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "tasks": len(task_order),
        "ordered_task_ids": list(task_order),
        "ordered_task_ids_sha256": canonical_sha256(list(task_order)),
        "metrics": {
            "pass_at_1": _paired_metric(task_order, before, after, "pass_at_1"),
            "pass_at_10": _paired_metric(task_order, before, after, "pass_at_k"),
            "compile_at_10": _paired_metric(
                task_order, before, after, "compile_at_k"
            ),
        },
    }


def audit_pass2_arm(args: argparse.Namespace) -> dict[str, Any]:
    evaluation = Path(args.evaluation_file).resolve()
    evaluator = Path(args.evaluator_file).resolve()
    wrapper = Path(args.wrapper_file).resolve()
    base_inference = Path(args.base_inference_file).resolve()
    task_ids = matched._read_evaluation_task_ids(evaluation, EXPECTED_ROWS)  # noqa: SLF001
    checkpoint = _validate_pass2_checkpoint(
        Path(args.pass2_checkpoint_contract).resolve(),
        Path(args.pass2_training_result).resolve(),
    )
    training_audit = _validate_training_audit(
        Path(args.pass2_training_audit).resolve(),
        args.expected_pass2_training_audit_sha256,
        checkpoint,
    )
    arm = _validate_arm(
        label="typed_direct_rs_sft_pass2_update54",
        predictions=Path(args.pass2_predictions).resolve(),
        full_score=Path(args.pass2_full_score).resolve(),
        clean_score=Path(args.pass2_clean_score).resolve(),
        checkpoint=checkpoint,
        evaluation=evaluation,
        evaluator=evaluator,
        wrapper=wrapper,
        base_inference=base_inference,
    )
    _require(
        arm["generation"]["task_ids"] == task_ids,
        "pass-2 arm task order differs from heldout evaluation",
    )
    report = {
        "schema": PASS2_ARM_AUDIT_SCHEMA,
        "status": "pass",
        "contract": {
            "tasks": EXPECTED_ROWS,
            "clean_tasks": EXPECTED_CLEAN_ROWS,
            "k": EXPECTED_K,
            "input_view": "typed_opaque_contract",
            "sampling": dict(matched.EXPECTED_SAMPLING),
            "excluded_task_ids": list(EXPECTED_EXCLUDED),
        },
        "checks": {
            "pass2_training_audit_validated": True,
            "pass2_checkpoint_validated": True,
            "generation_hash_chain_validated": True,
            "score_hash_chain_validated": True,
            "full175_complete": True,
            "clean174_complete": True,
            "current_inference_code_validated": True,
            "no_source_truncation": True,
        },
        "training_audit": {
            "sha256": sha256_file(Path(args.pass2_training_audit).resolve()),
            "schema": training_audit["schema"],
        },
        "arm": _arm_report(arm, checkpoint),
    }
    require_exact_or_write(Path(args.output), report)
    return report


def audit_comparison(args: argparse.Namespace) -> dict[str, Any]:
    evaluation = Path(args.evaluation_file).resolve()
    evaluator = Path(args.evaluator_file).resolve()
    wrapper = Path(args.wrapper_file).resolve()
    base_inference = Path(args.base_inference_file).resolve()
    task_ids = matched._read_evaluation_task_ids(evaluation, EXPECTED_ROWS)  # noqa: SLF001
    before_checkpoint = _validate_update58_checkpoint(
        Path(args.before_checkpoint_contract).resolve(),
        Path(args.before_training_result).resolve(),
    )
    after_checkpoint = _validate_pass2_checkpoint(
        Path(args.after_checkpoint_contract).resolve(),
        Path(args.after_training_result).resolve(),
    )
    before = _validate_arm(
        label="typed_direct_rs_sft_update58_current_stack",
        predictions=Path(args.before_predictions).resolve(),
        full_score=Path(args.before_full_score).resolve(),
        clean_score=Path(args.before_clean_score).resolve(),
        checkpoint=before_checkpoint,
        evaluation=evaluation,
        evaluator=evaluator,
        wrapper=wrapper,
        base_inference=base_inference,
    )
    after = _validate_arm(
        label="typed_direct_rs_sft_pass2_update54_current_stack",
        predictions=Path(args.after_predictions).resolve(),
        full_score=Path(args.after_full_score).resolve(),
        clean_score=Path(args.after_clean_score).resolve(),
        checkpoint=after_checkpoint,
        evaluation=evaluation,
        evaluator=evaluator,
        wrapper=wrapper,
        base_inference=base_inference,
    )
    arm_audit_path = Path(args.pass2_arm_audit).resolve()
    arm_audit = _read_object(arm_audit_path, "pass-2 arm audit")
    _require(
        _is_sha256(args.expected_pass2_arm_audit_sha256)
        and sha256_file(arm_audit_path) == args.expected_pass2_arm_audit_sha256
        and arm_audit.get("schema") == PASS2_ARM_AUDIT_SCHEMA
        and arm_audit.get("status") == "pass"
        and arm_audit.get("arm") == _arm_report(after, after_checkpoint),
        "pass-2 preflight audit does not bind the supplied after arm",
    )
    bg = before["generation"]
    ag = after["generation"]
    bs = before["full"]
    ass = after["full"]
    _require(
        bg["task_ids"] == ag["task_ids"] == task_ids
        and bg["source_sha256s"] == ag["source_sha256s"]
        and bg["encoder_tokens"] == ag["encoder_tokens"]
        and bg["header_contract"]["sampling"]
        == ag["header_contract"]["sampling"]
        and bg["header_contract"]["script_sha256"]
        == ag["header_contract"]["script_sha256"]
        and bg["header_contract"]["base_inference_script_sha256"]
        == ag["header_contract"]["base_inference_script_sha256"]
        and bg["header_contract"]["runtime"] == ag["header_contract"]["runtime"]
        and before["provenance"]["heldout"] == after["provenance"]["heldout"]
        and before["provenance"]["model"]["tokenizer_sha256"]
        == after["provenance"]["model"]["tokenizer_sha256"]
        and bs["task_order"] == ass["task_order"]
        and bs["candidate_slots"] == ass["candidate_slots"],
        "update58/pass-2 generation or slot pairing is not exact",
    )
    warmstart = after_checkpoint["contract"]["warmstart"]
    before_adapter = before["provenance"]["model"]["adapter"]
    _require(
        warmstart.get("run_contract_sha256") == before_checkpoint["canonical_sha256"]
        and warmstart.get("adapter_weights_sha256")
        == before_adapter.get("adapter_weights_sha256")
        and warmstart.get("adapter_config_sha256")
        == before_adapter.get("adapter_config_sha256")
        and before_checkpoint["contract"].get("base_model")
        == after_checkpoint["contract"].get("base_model")
        and before_checkpoint["contract"].get("model")
        == after_checkpoint["contract"].get("model")
        and before_checkpoint["contract"].get("model_revision")
        == after_checkpoint["contract"].get("model_revision")
        and after_checkpoint["contract"]["lora"].get("targets")
        == before_checkpoint["contract"]["lora"].get("targets"),
        "pass-2 is not the exact update58 adapter continuation",
    )
    score_fields = (
        "schema",
        "evaluation_sha256",
        "evaluator_sha256",
        "completion_attestation",
        "k",
        "workers",
        "batch_size",
        "timeout",
        "stability_runs",
        "ordered_slot_ids_sha256",
        "slots",
        "started_without_terminal_policy",
    )
    _require(
        all(
            bs["journal_contract"].get(key) == ass["journal_contract"].get(key)
            for key in score_fields
        ),
        "update58/pass-2 scoring contracts differ",
    )
    clean_order = [task_id for task_id in task_ids if task_id not in EXPECTED_EXCLUDED]
    report = {
        "schema": COMPARISON_SCHEMA,
        "status": "pass",
        "exact_pairing_validated": True,
        "historical_update58_predictions_reused": False,
        "contract": {
            "tasks": EXPECTED_ROWS,
            "clean_tasks": EXPECTED_CLEAN_ROWS,
            "k": EXPECTED_K,
            "input_view": "typed_opaque_contract",
            "sampling": dict(matched.EXPECTED_SAMPLING),
            "excluded_task_ids": list(EXPECTED_EXCLUDED),
        },
        "checks": {
            "pass2_preflight_audit_validated": True,
            "fresh_update58_current_stack_rerun_validated": True,
            "checkpoint_lineage_validated": True,
            "generation_hash_chains_validated": True,
            "score_hash_chains_validated": True,
            "same_task_order_and_sources": True,
            "same_seed_coordinates_and_sampling": True,
            "same_wrapper_and_base_inference_code": True,
            "same_typed_input_view": True,
            "same_tokenizer_and_encoder_lengths": True,
            "same_scorer_and_scoring_settings": True,
            "no_source_truncation": True,
        },
        "pass2_arm_audit": {
            "schema": arm_audit["schema"],
            "sha256": sha256_file(arm_audit_path),
        },
        "arms": {
            "typed_direct_rs_sft_update58_before": _arm_report(
                before, before_checkpoint
            ),
            "typed_direct_rs_sft_pass2_update54_after": _arm_report(
                after, after_checkpoint
            ),
        },
        "paired": {
            "full175": _comparison(
                task_ids, before["full"]["by_task"], after["full"]["by_task"]
            ),
            "clean174": _comparison(
                clean_order,
                before["clean"]["by_task"],
                after["clean"]["by_task"],
            ),
        },
    }
    require_exact_or_write(Path(args.output), report)
    return report


def _add_arm_paths(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-predictions", required=True)
    parser.add_argument(f"--{prefix}-full-score", required=True)
    parser.add_argument(f"--{prefix}-clean-score", required=True)


def _add_common_code_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-file", required=True)
    parser.add_argument("--evaluator-file", required=True)
    parser.add_argument("--wrapper-file", required=True)
    parser.add_argument("--base-inference-file", required=True)
    parser.add_argument("--output", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("pass2-arm", allow_abbrev=False)
    _add_arm_paths(arm, "pass2")
    arm.add_argument("--pass2-checkpoint-contract", required=True)
    arm.add_argument("--pass2-training-result", required=True)
    arm.add_argument("--pass2-training-audit", required=True)
    arm.add_argument("--expected-pass2-training-audit-sha256", required=True)
    _add_common_code_paths(arm)
    compare = subparsers.add_parser("compare", allow_abbrev=False)
    _add_arm_paths(compare, "before")
    compare.add_argument("--before-checkpoint-contract", required=True)
    compare.add_argument("--before-training-result", required=True)
    _add_arm_paths(compare, "after")
    compare.add_argument("--after-checkpoint-contract", required=True)
    compare.add_argument("--after-training-result", required=True)
    compare.add_argument("--pass2-arm-audit", required=True)
    compare.add_argument("--expected-pass2-arm-audit-sha256", required=True)
    _add_common_code_paths(compare)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if name.startswith("expected_") and name.endswith("_sha256"):
            _require(_is_sha256(value), f"{name} is not a lowercase SHA-256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = (
            audit_pass2_arm(args)
            if args.command == "pass2-arm"
            else audit_comparison(args)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"T5GEMMA_TYPED_PASS2_UPDATE58_RERUN_AUDIT_FAILED {exc}", flush=True)
        return 78
    print(
        "T5GEMMA_TYPED_PASS2_UPDATE58_RERUN_AUDIT_PASS "
        + json.dumps(
            {
                "command": args.command,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
