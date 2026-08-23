#!/usr/bin/env python3
"""Audit typed direct RS-SFT pass-2 and compare it with update58.

``training`` is the pre-evaluation gate.  It validates the stable, late-bound
209-row pass-2 result, root/checkpoint contracts, checkpoint pointer/state and
adapter/tokenizer artifacts.  ``compare`` reuses the existing rigorous typed
K=10 generation/scoring validators and publishes an exactly paired update58
versus pass-2 comparison for full-175 and known-clean-174.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from scripts.evaluation import audit_t5gemma2_typed_rs_sft as matched
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_typed_direct_rs_sft_pass2 as profile


TRAINING_AUDIT_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-training-audit-v1"
COMPARISON_SCHEMA = "t5gemma2-typed-pass2-vs-update58-matched-eval-audit-v1"
PASS1_RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-run-v1"
PASS1_DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-dataset-v1"
PASS2_RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-run-v1"
PASS2_CHECKPOINT_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-checkpoint-v1"
PASS2_DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-dataset-v1"
PREVIOUS_AUDIT_SCHEMA = "t5gemma2-typed-rs-sft-matched-eval-audit-v1"
EXPECTED_ROWS = 209
EXPECTED_LOCAL_ROWS = 190
EXPECTED_API_ROWS = 19
EXPECTED_UPDATES = 54
EXPECTED_PRIOR_ROWS = 225
EXPECTED_PASS1_UPDATE = 58
EXPECTED_CONTAMINANT = "sigless_6b1dd0c6b6fc"
EXPECTED_CLEAN_EXCLUSION = ("sigless_8bf7f40ca356",)
EXPECTED_PASS2_SCRIPT_SHA256 = sha256_file(Path(profile.__file__).resolve())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_object(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {resolved}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _pin(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"missing {label}: {resolved}")
    _require(
        isinstance(expected, str)
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected),
        f"{label} expected SHA-256 is malformed",
    )
    _require(sha256_file(resolved) == expected, f"{label} SHA-256 differs")
    return resolved


def _load_training_state(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, Mapping), "pass-2 training state is not a mapping")
    return value


def _validate_pass2_contract(
    contract: Mapping[str, Any],
    *,
    expected_canonical_sha256: str | None = None,
) -> dict[str, Any]:
    dataset = contract.get("dataset")
    composition = dataset.get("composition") if isinstance(dataset, Mapping) else None
    local_harvest = (
        dataset.get("local_harvest") if isinstance(dataset, Mapping) else None
    )
    dual_harvest = (
        dataset.get("dual_api_harvest") if isinstance(dataset, Mapping) else None
    )
    dual_targets = (
        dual_harvest.get("targets") if isinstance(dual_harvest, Mapping) else None
    )
    prior = (
        dataset.get("prior_225_exclusion") if isinstance(dataset, Mapping) else None
    )
    heldout = (
        dataset.get("heldout_identity_audit")
        if isinstance(dataset, Mapping)
        else None
    )
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
    schedule = dataset.get("schedule") if isinstance(dataset, Mapping) else None
    canonical = canonical_sha256(contract)
    _require(
        contract.get("schema") == PASS2_RUN_SCHEMA
        and contract.get("status") == "training"
        and contract.get("architecture") == "native_encoder_decoder"
        and (expected_canonical_sha256 is None or canonical == expected_canonical_sha256)
        and isinstance(runtime, Mapping)
        and runtime.get("trainer_sha256") == EXPECTED_PASS2_SCRIPT_SHA256
        and runtime.get("trainer_profile")
        == "typed_direct_only_rs_sft_pass2_local190_plus_dual_api"
        and isinstance(optimization, Mapping)
        and optimization.get("epochs") == 2
        and optimization.get("batch_size") == 1
        and optimization.get("gradient_accumulation") == 8
        and optimization.get("planned_updates") == EXPECTED_UPDATES
        and optimization.get("updates_per_epoch") == 27
        and optimization.get("learning_rate") == 0.00002
        and optimization.get("warmup_ratio") == 0
        and optimization.get("warmup_updates") == 0
        and optimization.get("seed") == 42
        and optimization.get("bf16") is True
        and optimization.get("gradient_checkpointing") is True
        and optimization.get("attn_implementation") == "sdpa"
        and isinstance(dataset, Mapping)
        and dataset.get("schema") == PASS2_DATASET_SCHEMA
        and dataset.get("rows") == EXPECTED_ROWS
        and dataset.get("architecture") == "native_encoder_decoder"
        and dataset.get("heldout_overlap") == 0
        and dataset.get("known_contaminant_excluded") == EXPECTED_CONTAMINANT
        and dataset.get("model_visible_fields")
        == ["opaque_typed_contract", "F2.text"]
        and dataset.get("tests_model_visible") is False
        and dataset.get("private_feedback_model_visible") is False
        and dataset.get("repair_conditioned_prefixes_visible") is False
        and dataset.get("reasoning_model_visible") is False
        and dataset.get("all_targets_bound_to_generation_journals") is True
        and dataset.get("production_floor_eligible") is True
        and isinstance(composition, Mapping)
        and composition.get("verified_direct") == EXPECTED_ROWS
        and composition.get("local_student_new") == EXPECTED_LOCAL_ROWS
        and composition.get("external_teacher_new") == EXPECTED_API_ROWS
        and composition.get("prior_225_replay") == 0
        and composition.get("gold_replay") == 0
        and composition.get("repair_conditioned") == 0
        and composition.get("reasoning_rows") == 0
        and composition.get("gold_source_replay") == 0
        and isinstance(composition.get("independently_generated_exact_gold_matches"), int)
        and 0 <= composition["independently_generated_exact_gold_matches"] <= EXPECTED_ROWS
        and isinstance(local_harvest, Mapping)
        and local_harvest.get("rows") == EXPECTED_LOCAL_ROWS
        and isinstance(dual_harvest, Mapping)
        and dual_harvest.get("schema")
        == "t5gemma2-typed-dual-api-pass2-input-audit-v1"
        and dual_harvest.get("status") == "complete"
        and dual_harvest.get("direct_code_only") is True
        and dual_harvest.get("gold_source_replay") is False
        and dual_harvest.get("heldout_175_model_visible") is False
        and dual_harvest.get("heldout_175_used_for_generation_or_selection") is False
        and isinstance(dual_targets, Mapping)
        and dual_targets.get("rows") == EXPECTED_API_ROWS
        and isinstance(prior, Mapping)
        and prior.get("rows") == EXPECTED_PRIOR_ROWS
        and isinstance(heldout, Mapping)
        and heldout.get("rows") == 175
        and isinstance(verification, Mapping)
        and verification.get("rows") == EXPECTED_ROWS
        and verification.get("passed") == EXPECTED_ROWS
        and verification.get("stability_runs") == 2
        and verification.get("timeout_seconds") == 30
        and verification.get("diagnostics_persisted") is False
        and verification.get("tests_model_visible") is False
        and isinstance(schedule, list)
        and len(schedule) == EXPECTED_ROWS
        and dataset.get("schedule_sha256") == canonical_sha256(schedule)
        and len({row.get("source_task_id") for row in schedule}) == EXPECTED_ROWS
        and sum(row.get("source_category") == "local_student_new" for row in schedule)
        == EXPECTED_LOCAL_ROWS
        and sum(row.get("source_category") == "external_teacher_new" for row in schedule)
        == EXPECTED_API_ROWS
        and isinstance(warmstart, Mapping)
        and warmstart.get("checkpoint_name") == "checkpoint-optstep-000058"
        and warmstart.get("update") == EXPECTED_PASS1_UPDATE
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
        and privacy.get("reasoning_persisted") is False,
        "pass-2 run/dataset/privacy/lineage contract differs",
    )
    return {
        "canonical_sha256": canonical,
        "rows": dataset["rows"],
        "local_rows": composition["local_student_new"],
        "api_rows": composition["external_teacher_new"],
        "planned_updates": optimization["planned_updates"],
        "warmstart_update": warmstart["update"],
    }


def audit_training(args: argparse.Namespace) -> dict[str, Any]:
    files = {
        "result": _pin(args.result, args.expected_result_sha256, "pass-2 result"),
        "root_contract": _pin(
            args.root_contract,
            args.expected_root_contract_sha256,
            "pass-2 root run contract",
        ),
        "dataset_manifest": _pin(
            args.dataset_manifest,
            args.expected_dataset_manifest_sha256,
            "pass-2 dataset manifest",
        ),
        "latest_pointer": _pin(
            args.latest_pointer,
            args.expected_latest_pointer_sha256,
            "pass-2 checkpoint pointer",
        ),
        "checkpoint_contract": _pin(
            args.checkpoint_contract,
            args.expected_checkpoint_contract_sha256,
            "pass-2 checkpoint run contract",
        ),
        "training_state": _pin(
            args.training_state,
            args.expected_training_state_sha256,
            "pass-2 training state",
        ),
        "adapter_weights": _pin(
            args.adapter_weights,
            args.expected_adapter_weights_sha256,
            "pass-2 adapter weights",
        ),
        "adapter_config": _pin(
            args.adapter_config,
            args.expected_adapter_config_sha256,
            "pass-2 adapter config",
        ),
        "tokenizer": _pin(
            args.tokenizer,
            args.expected_tokenizer_sha256,
            "pass-2 tokenizer",
        ),
    }
    result = _read_object(files["result"], "pass-2 result")
    root_contract = _read_object(files["root_contract"], "pass-2 root contract")
    checkpoint_contract = _read_object(
        files["checkpoint_contract"], "pass-2 checkpoint contract"
    )
    manifest = _read_object(files["dataset_manifest"], "pass-2 dataset manifest")
    pointer = _read_object(files["latest_pointer"], "pass-2 checkpoint pointer")
    contract_record = _validate_pass2_contract(root_contract)
    contract_sha = contract_record["canonical_sha256"]
    checkpoint_dir = files["checkpoint_contract"].parent
    _require(
        checkpoint_contract == root_contract
        and manifest == root_contract.get("dataset"),
        "pass-2 root/checkpoint/dataset contracts are not byte-semantic equals",
    )
    _require(
        result.get("schema") == PASS2_RUN_SCHEMA
        and result.get("status") == "complete"
        and result.get("updates") == EXPECTED_UPDATES
        and result.get("planned_updates") == EXPECTED_UPDATES
        and result.get("rows") == EXPECTED_ROWS
        and result.get("latest_checkpoint") == "checkpoint-optstep-000054"
        and result.get("production_floor_eligible") is True,
        "pass-2 result contract differs",
    )
    _require(
        pointer.get("schema") == PASS2_CHECKPOINT_SCHEMA
        and pointer.get("update") == EXPECTED_UPDATES
        and pointer.get("run_contract_sha256") == contract_sha
        and Path(str(pointer.get("path") or "")).expanduser().resolve()
        == checkpoint_dir.resolve()
        and checkpoint_dir.name == "checkpoint-optstep-000054",
        "pass-2 final checkpoint pointer differs",
    )
    state = _load_training_state(files["training_state"])
    _require(
        state.get("schema") == PASS2_CHECKPOINT_SCHEMA
        and state.get("update") == EXPECTED_UPDATES
        and state.get("epoch") == 2
        and state.get("next_row") == 0
        and state.get("run_contract_sha256") == contract_sha
        and isinstance(state.get("optimizer"), Mapping)
        and isinstance(state.get("scheduler"), Mapping)
        and isinstance(state.get("rng"), Mapping),
        "pass-2 final training state differs",
    )
    report = {
        "schema": TRAINING_AUDIT_SCHEMA,
        "status": "pass",
        "contract": contract_record,
        "result": {
            "rows": result["rows"],
            "updates": result["updates"],
            "latest_checkpoint": result["latest_checkpoint"],
        },
        "checkpoint": {
            "name": checkpoint_dir.name,
            "update": state["update"],
            "run_contract_canonical_sha256": contract_sha,
        },
        "artifacts": {
            name + "_sha256": sha256_file(path) for name, path in files.items()
        },
        "composition": {
            "rows": EXPECTED_ROWS,
            "local_student_new": EXPECTED_LOCAL_ROWS,
            "external_teacher_new": EXPECTED_API_ROWS,
            "prior_225_replay": 0,
            "gold_replay": 0,
            "heldout_overlap": 0,
        },
        "privacy": {
            "heldout_content_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "reasoning_persisted": False,
        },
    }
    require_exact_or_write(Path(args.output), report)
    return report


def _validate_update58_checkpoint(
    contract_path: Path, result_path: Path
) -> dict[str, Any]:
    record = matched._checkpoint_paths_record(contract_path, result_path)  # noqa: SLF001
    contract = record["contract"]
    dataset = contract.get("dataset")
    composition = dataset.get("composition") if isinstance(dataset, Mapping) else None
    verification = (
        dataset.get("full_acceptance_reverification")
        if isinstance(dataset, Mapping)
        else None
    )
    optimization = contract.get("optimization")
    privacy = contract.get("privacy")
    result = record["result"]
    _require(
        contract.get("schema") == PASS1_RUN_SCHEMA
        and contract.get("status") == "training"
        and contract.get("architecture") == "native_encoder_decoder"
        and isinstance(optimization, Mapping)
        and optimization.get("planned_updates") == 58
        and optimization.get("seed") == 42
        and isinstance(dataset, Mapping)
        and dataset.get("schema") == PASS1_DATASET_SCHEMA
        and dataset.get("rows") == 225
        and dataset.get("heldout_overlap") == 0
        and isinstance(composition, Mapping)
        and composition.get("verified_direct") == 225
        and composition.get("local_student_direct") == 141
        and composition.get("external_teacher_direct") == 84
        and composition.get("repair_conditioned") == 0
        and composition.get("gold_replay") == 0
        and isinstance(verification, Mapping)
        and verification.get("passed") == 225
        and isinstance(privacy, Mapping)
        and privacy.get("heldout_overlap") == 0
        and privacy.get("heldout_content_model_visible") is False
        and result.get("schema") == PASS1_RUN_SCHEMA
        and result.get("status") == "complete"
        and result.get("updates") == 58
        and result.get("planned_updates") == 58
        and result.get("rows") == 225
        and result.get("latest_checkpoint") == "checkpoint-optstep-000058",
        "update58 checkpoint/result contract differs",
    )
    return record


def _arm_report(arm: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
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
                name: matched._metric(count, 175)  # noqa: SLF001
                for name, count in arm["full"]["counts"].items()
            },
            "clean174": {
                name: matched._metric(count, 174)  # noqa: SLF001
                for name, count in arm["clean"]["counts"].items()
            },
        },
        "max_token_completions": arm["generation"]["max_token_completions"],
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


def audit_comparison(args: argparse.Namespace) -> dict[str, Any]:
    evaluation = Path(args.evaluation_file).resolve()
    evaluator = Path(args.evaluator_file).resolve()
    task_ids = matched._read_evaluation_task_ids(evaluation, 175)  # noqa: SLF001
    before_checkpoint = _validate_update58_checkpoint(
        Path(args.before_checkpoint_contract).resolve(),
        Path(args.before_training_result).resolve(),
    )
    after_checkpoint = matched._checkpoint_paths_record(  # noqa: SLF001
        Path(args.after_checkpoint_contract).resolve(),
        Path(args.after_training_result).resolve(),
    )
    _validate_pass2_contract(
        after_checkpoint["contract"],
        expected_canonical_sha256=after_checkpoint["canonical_sha256"],
    )
    before = matched._validate_arm(  # noqa: SLF001
        label="typed_direct_rs_sft_update58_before",
        prediction_path=Path(args.before_predictions).resolve(),
        full_score_path=Path(args.before_full_score).resolve(),
        clean_score_path=Path(args.before_clean_score).resolve(),
        checkpoint=before_checkpoint,
        evaluation_path=evaluation,
        evaluator_path=evaluator,
        expected_rows=175,
        expected_k=10,
        expected_sampling=matched.EXPECTED_SAMPLING,
        expected_excluded=EXPECTED_CLEAN_EXCLUSION,
    )
    after = matched._validate_arm(  # noqa: SLF001
        label="typed_direct_rs_sft_pass2_update54_after",
        prediction_path=Path(args.after_predictions).resolve(),
        full_score_path=Path(args.after_full_score).resolve(),
        clean_score_path=Path(args.after_clean_score).resolve(),
        checkpoint=after_checkpoint,
        evaluation_path=evaluation,
        evaluator_path=evaluator,
        expected_rows=175,
        expected_k=10,
        expected_sampling=matched.EXPECTED_SAMPLING,
        expected_excluded=EXPECTED_CLEAN_EXCLUSION,
    )
    previous_audit_path = Path(args.previous_update58_audit).resolve()
    previous_audit = _read_object(previous_audit_path, "previous update58 audit")
    previous_arm = previous_audit.get("arms", {}).get(
        "typed_direct_rs_sft_update58"
    )
    _require(
        sha256_file(previous_audit_path) == args.expected_previous_update58_audit_sha256
        and previous_audit.get("schema") == PREVIOUS_AUDIT_SCHEMA
        and previous_audit.get("status") == "pass"
        and previous_audit.get("exact_pairing_validated") is True
        and isinstance(previous_arm, Mapping)
        and previous_arm.get("artifacts") == matched._artifact_hashes(before)  # noqa: SLF001
        and previous_arm.get("checkpoint", {}).get("contract_file_sha256")
        == before_checkpoint["file_sha256"]
        and previous_arm.get("checkpoint", {}).get("contract_canonical_sha256")
        == before_checkpoint["canonical_sha256"]
        and previous_arm.get("checkpoint", {}).get("result_sha256")
        == before_checkpoint["result_sha256"],
        "previous audited update58 root differs from supplied before arm",
    )
    bg = before["generation"]
    ag = after["generation"]
    bs = before["full"]
    ass = after["full"]
    _require(
        bg["task_ids"] == ag["task_ids"] == task_ids
        and bg["source_sha256s"] == ag["source_sha256s"]
        and bg["encoder_tokens"] == ag["encoder_tokens"]
        and bg["header_contract"]["sampling"] == ag["header_contract"]["sampling"]
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
    clean_order = [task_id for task_id in task_ids if task_id not in EXPECTED_CLEAN_EXCLUSION]
    report = {
        "schema": COMPARISON_SCHEMA,
        "status": "pass",
        "exact_pairing_validated": True,
        "contract": {
            "tasks": 175,
            "k": 10,
            "clean_tasks": 174,
            "input_view": "typed_opaque_contract",
            "sampling": dict(matched.EXPECTED_SAMPLING),
            "excluded_task_ids": list(EXPECTED_CLEAN_EXCLUSION),
        },
        "previous_update58_audit": {
            "sha256": sha256_file(previous_audit_path),
            "schema": previous_audit["schema"],
        },
        "checks": {
            "previous_update58_audit_validated": True,
            "pass2_training_contract_validated": True,
            "checkpoint_lineage_validated": True,
            "generation_hash_chains_validated": True,
            "score_hash_chains_validated": True,
            "same_task_order_and_sources": True,
            "same_seed_coordinates_and_sampling": True,
            "same_typed_input_view": True,
            "same_tokenizer_and_encoder_lengths": True,
            "same_scorer_and_scoring_settings": True,
            "no_source_truncation": True,
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


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    pairs = (
        "result",
        "root-contract",
        "dataset-manifest",
        "latest-pointer",
        "checkpoint-contract",
        "training-state",
        "adapter-weights",
        "adapter-config",
        "tokenizer",
    )
    for name in pairs:
        parser.add_argument(f"--{name}", required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("training", allow_abbrev=False)
    _add_training_args(training)
    compare = subparsers.add_parser("compare", allow_abbrev=False)
    compare.add_argument("--before-predictions", required=True)
    compare.add_argument("--before-full-score", required=True)
    compare.add_argument("--before-clean-score", required=True)
    compare.add_argument("--before-checkpoint-contract", required=True)
    compare.add_argument("--before-training-result", required=True)
    compare.add_argument("--previous-update58-audit", required=True)
    compare.add_argument("--expected-previous-update58-audit-sha256", required=True)
    compare.add_argument("--after-predictions", required=True)
    compare.add_argument("--after-full-score", required=True)
    compare.add_argument("--after-clean-score", required=True)
    compare.add_argument("--after-checkpoint-contract", required=True)
    compare.add_argument("--after-training-result", required=True)
    compare.add_argument("--evaluation-file", required=True)
    compare.add_argument(
        "--evaluator-file",
        default=str(Path(__file__).with_name("graph_compile_at_k_antigravity.py")),
    )
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if name.startswith("expected_") and name.endswith("_sha256"):
            _require(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"{name} is not a lowercase SHA-256",
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_training(args) if args.command == "training" else audit_comparison(args)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"T5GEMMA_TYPED_PASS2_AUDIT_FAILED {exc}", flush=True)
        return 78
    print(
        "T5GEMMA_TYPED_PASS2_AUDIT_PASS "
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
