#!/usr/bin/env python3
"""Seal the Qwen -> matched RS/control -> VeRPO chain before either fit runs.

The measure split is represented only by the immutable file records already
attested by the executable-view build.  Its bytes and outcomes are not read;
all stage order, checkpoints, output paths, and hyperparameters are fixed from
train-side artifacts before RS-SFT or VeRPO can begin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.preprocessing.build_multifunction_executable_view import (
    EXPECTED_HELDOUT_ROWS,
    file_record,
    sha256_file,
    stable_sha256,
    validate_executable_view,
)
from scripts.preprocessing.build_verpo_feedback_view import (
    PRODUCTION_ELIGIBLE_TASK_IDS_SHA256,
    PRODUCTION_EXCLUDED_TASK_IDS_SHA256,
    PRODUCTION_EXPECTED_ACCOUNTING,
    validate_feedback_view,
)
from scripts.training.collect_chatgpt_compact_rs import (
    validate_qwen_student_checkpoint,
)
from scripts.training.direct_compact_qwen_decompiler import (
    DIRECT_PROMPT_MODE_CODE_ONLY_V1,
    DIRECT_PROMPT_MODE_QWEN_COT_V1,
    DIRECT_PROMPT_MODES,
)


SCHEMA = "post-qwen-predeclared-training-chain-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def load_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{resolved}: expected one JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--qwen-checkpoint", required=True)
    parser.add_argument("--qwen-build-manifest", required=True)
    parser.add_argument("--executable-dataset", required=True)
    parser.add_argument("--executable-seal", required=True)
    parser.add_argument("--executable-f2", required=True)
    parser.add_argument("--executable-f2-manifest", required=True)
    parser.add_argument("--executable-view-report", required=True)
    parser.add_argument(
        "--expected-executable-view-report-sha256", required=True
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument(
        "--expected-parent-fit-rows",
        type=int,
        default=1580,
        help=(
            "Full sealed fit-universe count; expanded production passes 2776."
        ),
    )
    parser.add_argument("--verpo-rollout", required=True)
    parser.add_argument("--verpo-rollout-seal", required=True)
    parser.add_argument("--verpo-teacher-f2", required=True)
    parser.add_argument("--verpo-teacher-f2-manifest", required=True)
    parser.add_argument("--verpo-feedback-view-report", required=True)
    parser.add_argument(
        "--expected-verpo-feedback-view-report-sha256", required=True
    )
    parser.add_argument("--verpo-feedback-public-manifest", required=True)
    parser.add_argument(
        "--expected-verpo-feedback-public-manifest-sha256", required=True
    )
    parser.add_argument(
        "--expected-verpo-eligible-rows",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["eligible_rows"],
    )
    parser.add_argument(
        "--expected-verpo-excluded-rows",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["excluded_rows"],
    )
    parser.add_argument(
        "--expected-verpo-source-expect-cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["source_expect_cases"],
    )
    parser.add_argument(
        "--expected-verpo-visible-expect-cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["visible_expect_cases"],
    )
    parser.add_argument(
        "--expected-verpo-holdback-expect-cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["holdback_expect_cases"],
    )
    parser.add_argument(
        "--expected-verpo-odd-case-tasks",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["odd_case_tasks"],
    )
    parser.add_argument(
        "--expected-verpo-eligible-task-ids-sha256",
        default=PRODUCTION_ELIGIBLE_TASK_IDS_SHA256,
    )
    parser.add_argument(
        "--expected-verpo-excluded-task-ids-sha256",
        default=PRODUCTION_EXCLUDED_TASK_IDS_SHA256,
    )
    parser.add_argument(
        "--derive-verpo-accounting-from-sealed-manifest",
        action="store_true",
        help=(
            "Use the hash-pinned feedback public/report manifests as the "
            "accounting and membership predeclaration."
        ),
    )
    parser.add_argument("--repair-artifact", required=True)
    parser.add_argument("--repair-report", required=True)
    parser.add_argument("--rs-output-root", required=True)
    parser.add_argument("--control-output-root", required=True)
    parser.add_argument("--verpo-output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows-per-rs-arm", type=int, default=3156)
    parser.add_argument(
        "--rows-per-rs-arm-from-sealed-parent",
        action="store_true",
    )
    parser.add_argument("--rs-min-unique-repairs", type=int, default=400)
    parser.add_argument(
        "--rs-harvest-verified-target", type=int, default=450
    )
    parser.add_argument("--rs-recertify-timeout", type=int, default=30)
    parser.add_argument(
        "--rs-recertify-stability-runs", type=int, default=2
    )
    parser.add_argument("--rs-learning-rate", type=float, default=2e-5)
    parser.add_argument("--rs-epochs", type=float, default=1.0)
    parser.add_argument("--rs-max-steps", type=int, default=-1)
    parser.add_argument("--rs-batch-size", type=int, default=1)
    parser.add_argument("--rs-grad-accum", type=int, default=16)
    parser.add_argument("--rs-lora-r", type=int, default=64)
    parser.add_argument("--rs-lora-alpha", type=int, default=128)
    parser.add_argument("--rs-lora-dropout", type=float, default=0.05)
    parser.add_argument("--verpo-group-size", type=int, default=8)
    parser.add_argument("--verpo-rollout-batch-size", type=int, default=1)
    parser.add_argument("--verpo-temperature", type=float, default=0.8)
    parser.add_argument("--verpo-max-updates", type=int, default=1232)
    parser.add_argument("--verpo-checkpoint-interval", type=int, default=154)
    parser.add_argument("--verpo-learning-rate", type=float, default=1e-6)
    parser.add_argument("--verpo-weight-decay", type=float, default=0.0)
    parser.add_argument("--verpo-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--verpo-ppo-clip", type=float, default=0.0)
    parser.add_argument(
        "--verpo-sft-replay-weight", type=float, default=0.05
    )
    parser.add_argument(
        "--verpo-on-policy-logprob-tolerance",
        type=float,
        default=2e-4,
    )
    parser.add_argument("--verpo-alpha", type=float, default=2.0)
    parser.add_argument("--verpo-beta", type=float, default=1.0)
    parser.add_argument("--verpo-max-new-tokens", type=int, default=2048)
    parser.add_argument("--verpo-reward-workers", type=int, default=16)
    parser.add_argument("--verpo-reward-timeout", type=int, default=30)
    parser.add_argument(
        "--verpo-reward-stability-runs", type=int, default=1
    )
    parser.add_argument("--verpo-judge-weight", type=float, default=0.25)
    parser.add_argument(
        "--verpo-judge-mode",
        choices=["off", "sparse_inline", "offline_queue"],
        default="sparse_inline",
    )
    parser.add_argument(
        "--verpo-judge-model", default="gpt-5.6-terra"
    )
    parser.add_argument(
        "--verpo-judge-api-style",
        choices=["openai_responses", "openai_compatible_chat"],
        default="openai_responses",
    )
    parser.add_argument(
        "--verpo-judge-base-url", default="https://api.openai.com/v1"
    )
    parser.add_argument("--verpo-judge-concurrency", type=int, default=1)
    parser.add_argument("--verpo-judge-max-tokens", type=int, default=12288)
    parser.add_argument(
        "--verpo-judge-completion-retries", type=int, default=0
    )
    parser.add_argument(
        "--verpo-judge-retry-max-tokens", type=int, default=12288
    )
    parser.add_argument(
        "--verpo-judge-thinking-mode", default="provider_default"
    )
    parser.add_argument(
        "--verpo-judge-reasoning-mode",
        choices=["standard", "pro"],
        default="standard",
    )
    parser.add_argument(
        "--verpo-judge-reasoning-effort",
        choices=["high", "max"],
        default="high",
    )
    parser.add_argument(
        "--verpo-judge-timeout-seconds", type=float, default=60.0
    )
    parser.add_argument("--verpo-judge-max-retries", type=int, default=0)
    parser.add_argument("--verpo-judge-interval", type=int, default=8)
    parser.add_argument("--verpo-judge-group-top-n", type=int, default=2)
    parser.add_argument(
        "--verpo-judge-deadline-seconds", type=float, default=60.0
    )
    parser.add_argument(
        "--verpo-judge-failure-policy",
        choices=["local_only"],
        default="local_only",
    )
    parser.add_argument("--verpo-judge-max-calls", type=int, default=0)
    parser.add_argument("--verpo-judge-escalation-queue", default="")
    parser.add_argument("--evaluation-k", type=int, default=10)
    parser.add_argument(
        "--evaluation-max-new-tokens", type=int, default=1024
    )
    parser.add_argument("--evaluation-temperature", type=float, default=0.8)
    parser.add_argument("--evaluation-top-p", type=float, default=0.95)
    parser.add_argument("--evaluation-top-k", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument(
        "--qwen-evaluation-direct-prompt-mode",
        choices=sorted(DIRECT_PROMPT_MODES),
        default=DIRECT_PROMPT_MODE_QWEN_COT_V1,
    )
    parser.add_argument(
        "--post-rs-evaluation-direct-prompt-mode",
        choices=sorted(DIRECT_PROMPT_MODES),
        default=DIRECT_PROMPT_MODE_CODE_ONLY_V1,
    )
    parser.add_argument("--evaluation-workers", type=int, default=48)
    parser.add_argument("--evaluation-timeout", type=int, default=30)
    parser.add_argument(
        "--evaluation-stability-runs", type=int, default=2
    )
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_new_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    expected_hash = stable_sha256(payload)
    envelope = {
        "schema": SCHEMA,
        "payload_sha256": expected_hash,
        "payload": dict(payload),
    }
    if path.is_file():
        observed = load_object(path)
        if observed != envelope:
            raise ValueError(
                "existing predeclared chain differs from current train-side inputs"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    if (
        not args.rows_per_rs_arm_from_sealed_parent
        and (args.rows_per_rs_arm <= 0 or args.rows_per_rs_arm % 2)
    ):
        raise ValueError("matched RS/control arm rows must be positive and even")
    if args.expected_parent_fit_rows <= 0:
        raise ValueError("expected parent fit rows must be positive")
    positive = {
        "rs_min_unique_repairs": args.rs_min_unique_repairs,
        "rs_recertify_timeout": args.rs_recertify_timeout,
        "rs_recertify_stability_runs": args.rs_recertify_stability_runs,
        "rs_learning_rate": args.rs_learning_rate,
        "rs_epochs": args.rs_epochs,
        "rs_batch_size": args.rs_batch_size,
        "rs_grad_accum": args.rs_grad_accum,
        "rs_lora_r": args.rs_lora_r,
        "rs_lora_alpha": args.rs_lora_alpha,
        "verpo_group_size": args.verpo_group_size,
        "verpo_rollout_batch_size": args.verpo_rollout_batch_size,
        "verpo_max_updates": args.verpo_max_updates,
        "verpo_checkpoint_interval": args.verpo_checkpoint_interval,
        "verpo_learning_rate": args.verpo_learning_rate,
        "verpo_max_new_tokens": args.verpo_max_new_tokens,
        "verpo_max_grad_norm": args.verpo_max_grad_norm,
        "verpo_on_policy_logprob_tolerance": (
            args.verpo_on_policy_logprob_tolerance
        ),
        "verpo_alpha": args.verpo_alpha,
        "verpo_reward_workers": args.verpo_reward_workers,
        "verpo_reward_timeout": args.verpo_reward_timeout,
        "verpo_reward_stability_runs": args.verpo_reward_stability_runs,
        "verpo_judge_concurrency": args.verpo_judge_concurrency,
        "verpo_judge_max_tokens": args.verpo_judge_max_tokens,
        "verpo_judge_retry_max_tokens": args.verpo_judge_retry_max_tokens,
        "verpo_judge_timeout_seconds": args.verpo_judge_timeout_seconds,
        "verpo_judge_interval": args.verpo_judge_interval,
        "verpo_judge_group_top_n": args.verpo_judge_group_top_n,
        "verpo_judge_deadline_seconds": args.verpo_judge_deadline_seconds,
        "evaluation_k": args.evaluation_k,
        "evaluation_max_new_tokens": args.evaluation_max_new_tokens,
        "evaluation_temperature": args.evaluation_temperature,
        "evaluation_batch_size": args.evaluation_batch_size,
        "evaluation_workers": args.evaluation_workers,
        "evaluation_timeout": args.evaluation_timeout,
        "evaluation_stability_runs": args.evaluation_stability_runs,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("all positive chain hyperparameters must be positive")
    if args.rs_max_steps == 0 or args.rs_max_steps < -1:
        raise ValueError("RS max_steps must be -1 or positive")
    if args.rs_harvest_verified_target <= args.rs_min_unique_repairs:
        raise ValueError(
            "GPT harvest target must exceed the independent recertification floor"
        )
    bounded_zero_one = {
        "rs_lora_dropout": args.rs_lora_dropout,
        "verpo_ppo_clip": args.verpo_ppo_clip,
        "verpo_sft_replay_weight": args.verpo_sft_replay_weight,
        "verpo_judge_weight": args.verpo_judge_weight,
        "evaluation_top_p": args.evaluation_top_p,
    }
    if any(
        value < 0.0 or value > 1.0
        for value in bounded_zero_one.values()
    ):
        raise ValueError("probability/weight parameters must be in [0,1]")
    nonnegative = {
        "verpo_weight_decay": args.verpo_weight_decay,
        "verpo_beta": args.verpo_beta,
        "verpo_judge_completion_retries": (
            args.verpo_judge_completion_retries
        ),
        "verpo_judge_max_retries": args.verpo_judge_max_retries,
        "evaluation_top_k": args.evaluation_top_k,
    }
    if any(value < 0 for value in nonnegative.values()):
        raise ValueError("nonnegative chain hyperparameters cannot be negative")
    if args.verpo_temperature <= 0.0:
        raise ValueError("VeRPO temperature must be positive")
    if (
        args.verpo_judge_mode == "sparse_inline"
        and args.verpo_judge_reasoning_mode != "standard"
    ):
        raise ValueError(
            "Pro mode is offline-only and cannot block sparse inline VeRPO"
        )
    if args.verpo_judge_failure_policy != "local_only":
        raise ValueError("teacher failures must fall back to local rewards")
    if args.verpo_judge_mode == "sparse_inline" and (
        args.verpo_judge_completion_retries != 0
        or args.verpo_judge_max_retries != 0
    ):
        raise ValueError("sparse inline VeRPO permits no blocking API retries")
    if not 2 <= args.verpo_judge_group_top_n <= args.verpo_group_size:
        raise ValueError("judge top-N must lie in [2, rollout group size]")
    if (
        args.verpo_judge_timeout_seconds
        != args.verpo_judge_deadline_seconds
        or args.verpo_judge_deadline_seconds > 60.0
    ):
        raise ValueError(
            "bounded judge timeout/deadline must match and cannot exceed 60s"
        )
    if args.verpo_judge_max_calls < 0:
        raise ValueError("judge max calls cannot be negative")
    if args.evaluation_top_k != 0:
        raise ValueError("matched evaluation requires top_k=0")
    if (
        args.qwen_evaluation_direct_prompt_mode
        != DIRECT_PROMPT_MODE_QWEN_COT_V1
        or args.post_rs_evaluation_direct_prompt_mode
        != DIRECT_PROMPT_MODE_CODE_ONLY_V1
    ):
        raise ValueError(
            "evaluation prompt modes are fixed by checkpoint conditioning: "
            "Qwen CoT=qwen_cot_v1 and control/RS/VeRPO=code_only_v1"
        )
    if not (0.0 < args.evaluation_top_p <= 1.0):
        raise ValueError("matched evaluation top_p must be in (0,1]")
    paths = {
        "qwen_checkpoint": Path(args.qwen_checkpoint).expanduser().resolve(),
        "qwen_build": Path(args.qwen_build_manifest).expanduser().resolve(),
        "dataset": Path(args.executable_dataset).expanduser().resolve(),
        "seal": Path(args.executable_seal).expanduser().resolve(),
        "f2": Path(args.executable_f2).expanduser().resolve(),
        "f2_manifest": Path(args.executable_f2_manifest).expanduser().resolve(),
        "view_report": Path(
            args.executable_view_report
        ).expanduser().resolve(),
        "contract": Path(args.contract).expanduser().resolve(),
        "verpo_rollout": Path(args.verpo_rollout).expanduser().resolve(),
        "verpo_rollout_seal": Path(
            args.verpo_rollout_seal
        ).expanduser().resolve(),
        "verpo_f2": Path(args.verpo_teacher_f2).expanduser().resolve(),
        "verpo_f2_manifest": Path(
            args.verpo_teacher_f2_manifest
        ).expanduser().resolve(),
        "verpo_feedback_report": Path(
            args.verpo_feedback_view_report
        ).expanduser().resolve(),
        "verpo_feedback_public": Path(
            args.verpo_feedback_public_manifest
        ).expanduser().resolve(),
        "repair": Path(args.repair_artifact).expanduser().resolve(),
        "repair_report": Path(args.repair_report).expanduser().resolve(),
    }
    for name, path in paths.items():
        if name == "qwen_checkpoint":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    if not SHA256_RE.fullmatch(
        args.expected_executable_view_report_sha256.strip().lower()
    ):
        raise ValueError("expected executable-view report hash is invalid")
    if not SHA256_RE.fullmatch(
        args.expected_verpo_feedback_view_report_sha256.strip().lower()
    ):
        raise ValueError("expected VeRPO feedback-view report hash is invalid")
    if not SHA256_RE.fullmatch(
        args.expected_verpo_feedback_public_manifest_sha256.strip().lower()
    ):
        raise ValueError(
            "expected VeRPO feedback public-manifest hash is invalid"
        )

    qwen = validate_qwen_student_checkpoint(
        paths["qwen_checkpoint"],
        qwen_build_manifest=paths["qwen_build"],
    )
    view = validate_executable_view(
        dataset=paths["dataset"],
        seal=paths["seal"],
        f2=paths["f2"],
        f2_manifest=paths["f2_manifest"],
        build_report=paths["view_report"],
        expected_build_report_sha256=(
            args.expected_executable_view_report_sha256
        ),
        contract=paths["contract"],
        verify_heldout=False,
        expected_parent_rows=args.expected_parent_fit_rows,
    )
    if view["heldout_bytes_opened_during_validation"]:
        raise AssertionError("predeclaration unexpectedly opened heldout bytes")
    if args.expected_parent_fit_rows == 2776 and (
        qwen.get("fit_task_count") != view["parent_rows"]
        or qwen.get("fit_ordered_task_ids_sha256")
        != view["parent_task_ids_sha256"]
        or qwen.get("heldout_task_count") != EXPECTED_HELDOUT_ROWS
        or qwen.get("heldout_intersection_count") != 0
    ):
        raise ValueError(
            "expanded Qwen checkpoint and executable view do not share the "
            "same sealed 2,776-task fit commitment"
        )
    expected_feedback_accounting: Mapping[str, Any] | None = {
        "parent_rows": view["rows"],
        "eligible_rows": args.expected_verpo_eligible_rows,
        "excluded_rows": args.expected_verpo_excluded_rows,
        "source_expect_cases": args.expected_verpo_source_expect_cases,
        "visible_expect_cases": args.expected_verpo_visible_expect_cases,
        "holdback_expect_cases": (
            args.expected_verpo_holdback_expect_cases
        ),
        "odd_case_tasks": args.expected_verpo_odd_case_tasks,
    }
    expected_eligible_digest: str | None = (
        args.expected_verpo_eligible_task_ids_sha256
    )
    expected_excluded_digest: str | None = (
        args.expected_verpo_excluded_task_ids_sha256
    )
    if args.derive_verpo_accounting_from_sealed_manifest:
        expected_feedback_accounting = None
        expected_eligible_digest = None
        expected_excluded_digest = None
    feedback_view = validate_feedback_view(
        rollout=paths["verpo_rollout"],
        seal=paths["verpo_rollout_seal"],
        f2=paths["verpo_f2"],
        f2_manifest=paths["verpo_f2_manifest"],
        build_report=paths["verpo_feedback_report"],
        expected_build_report_sha256=(
            args.expected_verpo_feedback_view_report_sha256
        ),
        public_manifest=paths["verpo_feedback_public"],
        expected_public_manifest_sha256=(
            args.expected_verpo_feedback_public_manifest_sha256
        ),
        executable_dataset=paths["dataset"],
        executable_seal=paths["seal"],
        executable_f2=paths["f2"],
        executable_f2_manifest=paths["f2_manifest"],
        executable_view_report=paths["view_report"],
        expected_executable_view_report_sha256=(
            args.expected_executable_view_report_sha256
        ),
        contract=paths["contract"],
        expected_accounting=expected_feedback_accounting,
        expected_eligible_task_ids_sha256=expected_eligible_digest,
        expected_excluded_task_ids_sha256=expected_excluded_digest,
        expected_parent_fit_rows=args.expected_parent_fit_rows,
    )
    if feedback_view["heldout_bytes_opened_during_validation"]:
        raise AssertionError("feedback-view predeclaration opened heldout bytes")
    effective_feedback = feedback_view["accounting"]
    eligible_rows = int(effective_feedback["eligible_rows"])
    excluded_rows = int(effective_feedback["excluded_rows"])
    if int(effective_feedback["parent_rows"]) != view["rows"]:
        raise ValueError(
            "feedback manifest parent rows differ from executable-view seal"
        )
    if args.rows_per_rs_arm_from_sealed_parent:
        args.rows_per_rs_arm = 2 * view["rows"]
    if args.rows_per_rs_arm != 2 * view["rows"]:
        raise ValueError(
            "matched RS/control arms must be twice the sealed executable rows"
        )
    if args.rs_min_unique_repairs > view["rows"]:
        raise ValueError(
            "RS unique-repair floor exceeds sealed executable task count"
        )
    planned_rollout_groups = (
        args.verpo_max_updates * args.verpo_rollout_batch_size
    )
    if planned_rollout_groups < eligible_rows:
        raise ValueError(
            "production VeRPO must predeclare at least one complete "
            "deterministic cycle over all eligible training tasks"
        )
    planned_unique_tasks = min(eligible_rows, planned_rollout_groups)
    judge_max_calls = (
        args.verpo_judge_max_calls
        if args.verpo_judge_max_calls > 0
        else (
            planned_rollout_groups + args.verpo_judge_interval - 1
        )
        // args.verpo_judge_interval
    )
    judge_escalation_queue = (
        Path(args.verpo_judge_escalation_queue).expanduser().resolve()
        if args.verpo_judge_escalation_queue
        else (
            Path(args.verpo_output_root).expanduser().resolve()
            / "offline_teacher_escalations.jsonl"
        )
    )
    feedback_excluded_digest = str(
        feedback_view.get("excluded_task_ids_sha256") or ""
    )
    if not SHA256_RE.fullmatch(feedback_excluded_digest):
        raise ValueError(
            "feedback public manifest lacks excluded task-ID commitment"
        )

    repair_report = load_object(paths["repair_report"])
    inputs = repair_report.get("inputs")
    outputs = repair_report.get("outputs")
    qwen_stage = (
        inputs.get("qwen_student_stage")
        if isinstance(inputs, Mapping)
        else None
    )
    executable_stage = (
        inputs.get("executable_view") if isinstance(inputs, Mapping) else None
    )
    if (
        repair_report.get("schema") != "direct-compact-openai-rs-harvest-v2"
        or repair_report.get("status") != "complete"
        or repair_report.get("provider") != "openai"
        or repair_report.get("api") != "responses"
        or repair_report.get("base_url") != "https://api.openai.com/v1"
        or repair_report.get("requested_model") != "gpt-5.6-sol"
        or (repair_report.get("request_parameters") or {})
        != {
            "max_output_tokens_initial": 8192,
            "max_output_tokens_ceiling": 12288,
            "max_output_tokens_escalation": {
                "status": "incomplete",
                "incomplete_details_reason": "max_output_tokens",
                "otherwise_escalate": False,
            },
            "reasoning": {
                "effort": repair_report.get("reasoning_effort")
            },
            "store": False,
        }
        or repair_report.get("production_coverage_met") is not True
        or int(
            repair_report.get("minimum_unique_verified_tasks", -1)
        )
        != args.rs_harvest_verified_target
        or not isinstance(outputs, Mapping)
        or outputs.get("verified_repairs_sha256")
        != sha256_file(paths["repair"])
        or not isinstance(qwen_stage, Mapping)
        or (qwen_stage.get("checkpoint") or {}).get("run_provenance", {}).get(
            "sha256"
        )
        != qwen["checkpoint"]["run_provenance"]["sha256"]
        or not isinstance(executable_stage, Mapping)
        or (executable_stage.get("report") or {}).get("sha256")
        != view["report"]["sha256"]
    ):
        raise ValueError(
            "GPT repair report is not bound to Qwen + sealed executable view"
        )

    payload = {
        "stage_order": [
            "qwen_sequence_kd_complete",
            "gpt_5_6_sol_rs_sft_and_matched_gold_control",
            "sparse_bounded_teacher_verpo",
            "heldout_175_evaluation_all_predeclared_arms",
        ],
        "stage_order_predeclared": True,
        "checkpoint_selection_from_heldout": False,
        "early_stopping_from_heldout": False,
        "launch_decisions_from_heldout": False,
        "heldout_bytes_opened_while_predeclaring": False,
        "qwen_stage": qwen,
        "executable_train": view,
        "gpt_repairs": {
            "artifact": file_record(paths["repair"]),
            "report": file_record(paths["repair_report"]),
            "provider": "openai",
            "api": "responses",
            "model": "gpt-5.6-sol",
            "harvest_verified_task_target": (
                args.rs_harvest_verified_target
            ),
            "independent_recertification_floor": (
                args.rs_min_unique_repairs
            ),
            "azure": False,
            "batch": False,
            "max_output_tokens_initial": 8192,
            "max_output_tokens_ceiling": 12288,
            "escalate_only_on_incomplete_max_output_tokens": True,
        },
        "rs_sft": {
            "warmstart_checkpoint": str(paths["qwen_checkpoint"]),
            "intervention_output": str(
                Path(args.rs_output_root).expanduser().resolve()
            ),
            "matched_control_output": str(
                Path(args.control_output_root).expanduser().resolve()
            ),
            "rows_per_arm": args.rows_per_rs_arm,
            "min_unique_repairs": args.rs_min_unique_repairs,
            "source_harvest_verified_task_target": (
                args.rs_harvest_verified_target
            ),
            "recertification_timeout": args.rs_recertify_timeout,
            "recertification_stability_runs": (
                args.rs_recertify_stability_runs
            ),
            "learning_rate": args.rs_learning_rate,
            "epochs": args.rs_epochs,
            "max_steps": args.rs_max_steps,
            "batch_size": args.rs_batch_size,
            "grad_accum": args.rs_grad_accum,
            "lora_r": args.rs_lora_r,
            "lora_alpha": args.rs_lora_alpha,
            "lora_dropout": args.rs_lora_dropout,
            "decoder_model": "Qwen/Qwen3-8B",
            "decoder_revision": (
                "b968826d9c46dd6066d109eabc6255188de91218"
            ),
            "attn_implementation": "flash_attention_2",
            "gradient_checkpointing": True,
            "bf16": True,
            "fp16": False,
            "load_4bit": False,
            "sequence_distribution_nll": False,
            "eval_strategy": "no",
            "seed": args.seed,
            "heldout_loaded_during_training": False,
        },
        "verpo": {
            "warmstart": "predeclared_rs_intervention_checkpoint",
            "output": str(
                Path(args.verpo_output_root).expanduser().resolve()
            ),
            "rollout_dataset": feedback_view["rollout"],
            "rollout_seal": feedback_view["seal"],
            "feedback_view_report": feedback_view["report"],
            "feedback_public_manifest": feedback_view["public_manifest"],
            "rollout_rows": feedback_view["rows"],
            "parent_safe_rows": view["rows"],
            "parent_fit_rows": view["parent_rows"],
            "feedback_accounting": feedback_view["accounting"],
            "feedback_eligible_task_ids_sha256": (
                feedback_view["task_ids_sha256"]
            ),
            "feedback_excluded_task_ids_sha256": (
                feedback_excluded_digest
            ),
            "task_sampling_policy": (
                "stable_sha256_epoch_permutation_without_replacement_then_cycle"
            ),
            "planned_rollout_groups": planned_rollout_groups,
            "planned_unique_tasks": planned_unique_tasks,
            "planned_unique_fraction": (
                planned_unique_tasks / eligible_rows
            ),
            "complete_dataset_cycles": (
                planned_rollout_groups // eligible_rows
            ),
            "partial_cycle_groups": (
                planned_rollout_groups % eligible_rows
            ),
            "teacher": args.verpo_judge_model,
            "teacher_source": feedback_view["f2"],
            "teacher_source_manifest": feedback_view["f2_manifest"],
            "group_size": args.verpo_group_size,
            "rollout_batch_size": args.verpo_rollout_batch_size,
            "temperature": args.verpo_temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_updates": args.verpo_max_updates,
            "checkpoint_interval": args.verpo_checkpoint_interval,
            "learning_rate": args.verpo_learning_rate,
            "weight_decay": args.verpo_weight_decay,
            "max_grad_norm": args.verpo_max_grad_norm,
            "ppo_clip": args.verpo_ppo_clip,
            "sft_replay_weight": args.verpo_sft_replay_weight,
            "on_policy_logprob_tolerance": (
                args.verpo_on_policy_logprob_tolerance
            ),
            "verpo_alpha": args.verpo_alpha,
            "verpo_beta": args.verpo_beta,
            "max_new_tokens": args.verpo_max_new_tokens,
            "reward_workers": args.verpo_reward_workers,
            "reward_timeout": args.verpo_reward_timeout,
            "reward_stability_runs": (
                args.verpo_reward_stability_runs
            ),
            "judge_weight": args.verpo_judge_weight,
            "advantage_contract": {
                "global": "full_pass_minus_group_mean",
                "local": "density_calibrated_reward_minus_group_mean",
                "teacher": (
                    "observed_selected_signal_minus_selected_mean_"
                    "unobserved_zero"
                ),
                "unified": (
                    "A_global + verpo_beta*A_local + "
                    "judge_weight*A_teacher"
                ),
                "normalization_factor": 1,
                "population_std_division": False,
                "teacher_is_separately_centered_paper_extension": True,
                "missing_teacher_signal_is_neutral": True,
            },
            "judge_mode": args.verpo_judge_mode,
            "judge_api_style": args.verpo_judge_api_style,
            "judge_base_url": args.verpo_judge_base_url.rstrip("/"),
            "judge_concurrency": args.verpo_judge_concurrency,
            "judge_max_tokens": args.verpo_judge_max_tokens,
            "judge_completion_retries": (
                args.verpo_judge_completion_retries
            ),
            "judge_retry_max_tokens": (
                args.verpo_judge_retry_max_tokens
            ),
            "judge_thinking_mode": args.verpo_judge_thinking_mode,
            "judge_reasoning_mode": args.verpo_judge_reasoning_mode,
            "judge_reasoning_effort": (
                args.verpo_judge_reasoning_effort
            ),
            "judge_timeout_seconds": args.verpo_judge_timeout_seconds,
            "judge_max_retries": args.verpo_judge_max_retries,
            "judge_interval": args.verpo_judge_interval,
            "judge_group_top_n": args.verpo_judge_group_top_n,
            "judge_deadline_seconds": args.verpo_judge_deadline_seconds,
            "judge_failure_policy": args.verpo_judge_failure_policy,
            "judge_max_calls": judge_max_calls,
            "judge_escalation_queue": str(judge_escalation_queue),
            "decoder_model": "Qwen/Qwen3-8B",
            "decoder_revision": (
                "b968826d9c46dd6066d109eabc6255188de91218"
            ),
            "attn_implementation": "flash_attention_2",
            "bf16": True,
            "fp16": False,
            "load_4bit": False,
            "resume_within_declared_output_only": True,
            "seed": args.seed,
            "heldout_loaded_during_training": False,
        },
        "evaluation": {
            "after_all_training_stages": True,
            "arms": [
                "qwen_sequence_kd",
                "matched_gold_control",
                "gpt_5_6_sol_rs_sft",
                "sparse_teacher_verpo",
            ],
            "direct_prompt_modes": {
                "qwen_sequence_kd": (
                    args.qwen_evaluation_direct_prompt_mode
                ),
                "matched_gold_control": (
                    args.post_rs_evaluation_direct_prompt_mode
                ),
                "gpt_5_6_sol_rs_sft": (
                    args.post_rs_evaluation_direct_prompt_mode
                ),
                "sparse_teacher_verpo": (
                    args.post_rs_evaluation_direct_prompt_mode
                ),
            },
            "k": args.evaluation_k,
            "num_samples": args.evaluation_k,
            "max_new_tokens": args.evaluation_max_new_tokens,
            "temperature": args.evaluation_temperature,
            "top_p": args.evaluation_top_p,
            "top_k": args.evaluation_top_k,
            "batch_size": args.evaluation_batch_size,
            "limit": 0,
            "score_workers": args.evaluation_workers,
            "timeout": args.evaluation_timeout,
            "stability_runs": args.evaluation_stability_runs,
            "decoder_model": "Qwen/Qwen3-8B",
            "decoder_revision": (
                "b968826d9c46dd6066d109eabc6255188de91218"
            ),
            "attn_implementation": "flash_attention_2",
            "bf16": True,
            "seed": args.seed,
            "heldout": view["heldout"],
            "heldout_seal": view["heldout_seal"],
            "heldout_rows": view["heldout_rows"],
            "heldout_task_ids_sha256": view[
                "heldout_task_ids_sha256"
            ],
            "performance_can_change_stage_order": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_new_or_validate(output, payload)
    print(
        "POST_QWEN_CHAIN_PREDECLARED "
        f"fit_parent={view['parent_rows']} executable={view['rows']} "
        f"heldout={view['heldout_rows']} "
        f"payload_sha256={stable_sha256(payload)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
