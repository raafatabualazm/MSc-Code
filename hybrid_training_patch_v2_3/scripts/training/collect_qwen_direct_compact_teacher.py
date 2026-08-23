#!/usr/bin/env python3
"""Collect an offline K=8 Qwen teacher journal for direct-compact training.

The input must be a pre-serialized API-readable prompt JSONL produced by the
corrected CompactArtifactBundle path. Hidden verifier rows are loaded from a
separate file and are never placed in an API message.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    DEFAULT_MODEL,
    OBJECTIVE_MODES,
    OBJECTIVE_MODE_REQUIRE_TOP5,
    OBJECTIVE_MODE_SEQUENCE_ONLY,
    SAMPLE_SEED_ALGORITHM,
    SAMPLES_PER_TASK,
    TOP_LOGPROBS,
    ArtifactError,
    StudentTokenizerBinding,
    build_messages,
    collect_candidates,
    count_prompt_tokens,
    file_record,
    load_f2_prompt_contract,
    load_target_length_contract,
    load_verified_prompt_rows,
    materialize_artifacts,
    read_jsonl,
    sha256_file,
    sha256_text,
    stable_sha256,
    validate_alibaba_model_studio_base_url,
)


def load_verifier_rows(
    path: Path,
    *,
    expected_sha256: str,
    expected_task_ids: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    record = file_record(path)
    if record["sha256"] != expected_sha256.strip().lower():
        raise ArtifactError(
            "verifier artifact hash mismatch: "
            f"expected {expected_sha256}, got {record['sha256']}"
        )
    tests: dict[str, str] = {}
    for index, row in enumerate(read_jsonl(path)):
        task_id = str(row.get("task_id") or "")
        harness = row.get("acceptance_tests") or row.get("tests")
        if not task_id or not isinstance(harness, str) or not harness.strip():
            raise ArtifactError(f"verifier row {index} lacks task_id/tests")
        if task_id in tests:
            raise ArtifactError(f"duplicate verifier task_id: {task_id}")
        tests[task_id] = harness
    missing = sorted(expected_task_ids.difference(tests))
    if missing:
        raise ArtifactError(
            f"verifier artifact is missing selected tasks: {missing[:3]}"
        )
    # A full sealed verifier file may intentionally be paired with --max-tasks
    # for a smoke run. Keep only selected tasks in memory; the complete source
    # file remains hash-bound in the run header.
    return {task_id: tests[task_id] for task_id in sorted(expected_task_ids)}, record


def make_dart_verifier(
    tests_by_task: Mapping[str, str],
    *,
    timeout_seconds: int,
):
    from scripts.training import teacher_repair_dataset_antigravity as repair
    from scripts.evaluation import graph_compile_at_k_antigravity as evaluator

    implementation = {
        "repair_wrapper": sha256_file(Path(inspect.getsourcefile(repair) or "")),
        "exact_evaluator": sha256_file(Path(inspect.getsourcefile(evaluator) or "")),
        "timeout_seconds": int(timeout_seconds),
    }
    verifier_sha = stable_sha256(implementation)

    def verify(candidate: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(candidate["task_id"])
        tests = tests_by_task[task_id]
        code = str((candidate.get("parse") or {}).get("code") or "")
        if not code:
            return {
                "compiled": False,
                "passed": False,
                "harness_completion_attested": False,
                "diagnostic": "not_parseable",
                "verifier_id": "dart-hidden-acceptance-completion-v1",
                "verifier_sha256": verifier_sha,
                "tests_sha256": sha256_text(tests),
            }
        result = repair.evaluate_candidate(
            code,
            tests,
            f"qwen_teacher_{candidate['candidate_id']}",
            int(timeout_seconds),
        )
        # The exact evaluator makes a zero exit status contingent on its private
        # end-of-harness nonce. A passed result therefore carries an independent
        # completion attestation, not just a provider finish_reason.
        return {
            "compiled": bool(result.compiled),
            "passed": bool(result.passed),
            "harness_completion_attested": bool(result.passed),
            "diagnostic": str(result.diagnostic or ""),
            "verifier_id": "dart-hidden-acceptance-completion-v1",
            "verifier_sha256": verifier_sha,
            "tests_sha256": sha256_text(tests),
        }

    return verify, implementation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--expected-prompt-rows", type=int)
    parser.add_argument("--prompt-manifest", required=True, type=Path)
    parser.add_argument("--expected-prompt-manifest-sha256", required=True)
    parser.add_argument("--verifier-jsonl", required=True, type=Path)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    parser.add_argument("--student-eos-token-id", required=True, type=int)
    parser.add_argument("--target-contract", required=True, type=Path)
    parser.add_argument("--expected-target-contract-sha256", required=True)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--parseable-output", required=True, type=Path)
    parser.add_argument("--rs-sft-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument(
        "--quality-gate-json",
        type=Path,
        help=(
            "Optional passed pilot-quality gate to bind into the production "
            "journal before collection."
        ),
    )
    parser.add_argument("--expected-quality-gate-sha256", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--objective-mode",
        choices=sorted(OBJECTIVE_MODES),
        default=OBJECTIVE_MODE_REQUIRE_TOP5,
        help=(
            "require_top5 fails unless every content token has exact top-5 "
            "logprobs/bytes; sequence_only samples the same teacher without "
            "requesting logprobs and can only build the sequence-NLL stage"
        ),
    )
    parser.add_argument("--required-function", default="fn0")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Must be 1.0 for untempered teacher-distribution sampling",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=101)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument(
        "--length-max-token-escalation",
        type=int,
        nargs="*",
        default=[16384, 24576],
        help=(
            "Larger final-output capacities tried, in order, only when the "
            "provider returns finish_reason=length. Completed draws are never "
            "reissued."
        ),
    )
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Qwen reasoning tokens. This is compatible with "
            "sequence_only, but require_top5 must use --no-enable-thinking "
            "because hidden reasoning would change the token-logprob prefix."
        ),
    )
    parser.add_argument("--seed-base", type=int, default=44)
    parser.add_argument("--max-prompt-tokens", type=int, default=12000)
    parser.add_argument("--chat-overhead-reserve", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--verifier-timeout-seconds", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("QWEN_TEACHER_WORKERS", "16")),
    )
    parser.add_argument(
        "--verifier-workers",
        type=int,
        default=int(os.environ.get("QWEN_VERIFIER_WORKERS", "16")),
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument(
        "--task-selection-strategy",
        choices=("prefix", "deterministic_hash"),
        default="prefix",
        help=(
            "Selection used only with --max-tasks. deterministic_hash gives a "
            "stable corpus-wide pilot rather than a file-prefix smoke test."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DASHSCOPE_ENDPOINT", ""),
        help="OpenAI-compatible endpoint; defaults to DASHSCOPE_ENDPOINT",
    )
    parser.add_argument(
        "--token-plan-automation-authorized",
        action="store_true",
        help=(
            "Attest that Alibaba explicitly authorized automated research use "
            "of this account's Token Plan endpoint. The attestation is sealed "
            "into the journal header."
        ),
    )
    parser.add_argument(
        "--authorize-orphan-reissue-with-duplicate-billing-risk",
        action="store_true",
        help=(
            "Explicitly authorize recovery of fsynced teacher_slot_started "
            "events that have no terminal outcome. The exact logical slot, "
            "request parameters, and seed are reused, but the original "
            "provider request may already have billed or completed. Requires "
            "--token-plan-automation-authorized."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="QWEN_API_KEY",
        help="Name of the environment variable containing the API key",
    )
    parser.add_argument(
        "--split-homogeneous-shards",
        action="store_true",
        help=(
            "Permit an observed returned-model/backend change only by writing "
            "separate homogeneous output shards"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the sealed plan without making calls or writes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_url = validate_alibaba_model_studio_base_url(
        args.base_url,
        token_plan_automation_authorized=(
            args.token_plan_automation_authorized
        ),
    )
    if not args.model.strip():
        raise ArtifactError("--model cannot be empty")
    if (
        args.authorize_orphan_reissue_with_duplicate_billing_risk
        and not args.token_plan_automation_authorized
    ):
        raise ArtifactError(
            "orphan reissue requires --token-plan-automation-authorized"
        )
    if args.temperature != 1.0 or args.top_p != 1.0 or args.top_k != 101:
        raise ArtifactError(
            "production MC sequence KL requires temperature=1.0, "
            "top_p=1.0, and Alibaba-disabled top_k=101"
        )
    if (
        args.objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
        and args.enable_thinking
    ):
        raise ArtifactError(
            "require_top5 cannot use hidden Qwen reasoning tokens: final-content "
            "logprobs would be conditioned on a prefix unavailable to the "
            "student; use --no-enable-thinking or objective_mode=sequence_only"
        )
    if (
        args.max_tokens <= 0
        or args.max_prompt_tokens <= 0
        or args.timeout_seconds <= 0
    ):
        raise ArtifactError("token and timeout budgets must be positive")
    length_capacities = [int(args.max_tokens), *args.length_max_token_escalation]
    if (
        any(value <= 0 for value in length_capacities)
        or length_capacities != sorted(set(length_capacities))
    ):
        raise ArtifactError(
            "length max-token capacities must be unique and strictly increasing"
        )
    if args.enable_thinking and not 1 <= args.thinking_budget < args.max_tokens:
        raise ArtifactError("--thinking-budget must be in [1, max-tokens)")
    if args.objective_mode == OBJECTIVE_MODE_SEQUENCE_ONLY and (
        args.model != DEFAULT_MODEL
        or not args.enable_thinking
        or args.thinking_budget != 8192
        or args.max_tokens != 12288
    ):
        raise ArtifactError(
            "sequence_only production collection is pinned to "
            "model=qwen3.8-max-preview, enable_thinking=true, "
            "thinking_budget=8192, and max_tokens=12288"
        )
    if not 0 <= args.seed_base < 2**31:
        raise ArtifactError("--seed-base must be in [0, 2^31)")
    if args.chat_overhead_reserve < 0:
        raise ArtifactError("--chat-overhead-reserve cannot be negative")
    if args.workers < 1 or args.verifier_workers < 1:
        raise ArtifactError("worker counts must be positive")
    if args.progress_every < 0:
        raise ArtifactError("--progress-every cannot be negative")
    quality_gate_record: dict[str, Any] | None = None
    quality_gate: dict[str, Any] | None = None
    if bool(args.quality_gate_json) != bool(
        args.expected_quality_gate_sha256.strip()
    ):
        raise ArtifactError(
            "--quality-gate-json and --expected-quality-gate-sha256 must "
            "be provided together"
        )
    exact_quality_pilot = bool(
        args.max_tasks == 16
        and args.task_selection_strategy == "deterministic_hash"
    )
    if not args.dry_run:
        if args.quality_gate_json is None and not exact_quality_pilot:
            raise ArtifactError(
                "paid collection without a passed quality gate is allowed only "
                "for the exact deterministic 16-task K=8 pilot"
            )
        if args.quality_gate_json is not None and args.max_tasks != 0:
            raise ArtifactError(
                "a passed pilot gate may authorize only the complete sealed "
                "production task set, not another selected subset"
            )
    if args.quality_gate_json is not None:
        quality_gate_record = file_record(args.quality_gate_json)
        if quality_gate_record["sha256"] != (
            args.expected_quality_gate_sha256.strip().lower()
        ):
            raise ArtifactError("Qwen pilot quality-gate hash mismatch")
        try:
            quality_gate = json.loads(
                args.quality_gate_json.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read Qwen pilot quality gate: {exc}") from exc
        if (
            not isinstance(quality_gate, Mapping)
            or quality_gate.get("schema") != "qwen-teacher-quality-gate-v1"
            or quality_gate.get("passed") is not True
            or int(quality_gate.get("pilot_tasks", 0)) != 16
            or int(quality_gate.get("candidates", 0))
            != int(quality_gate.get("pilot_tasks", 0)) * SAMPLES_PER_TASK
            or int(quality_gate.get("verified_tasks", 0))
            < int(quality_gate.get("minimum_verified_tasks", 1))
            or not 0.0
            <= float(quality_gate.get("minimum_parseable_fraction", -1.0))
            <= 1.0
            or int(quality_gate.get("parseable_candidates", -1))
            / max(1, int(quality_gate.get("candidates", 0)))
            < float(quality_gate.get("minimum_parseable_fraction", 1.0))
        ):
            raise ArtifactError("Qwen pilot quality gate is not a passed K=8 gate")
        sampling_diversity = quality_gate.get("sampling_diversity")
        unique_by_task = (
            sampling_diversity.get("unique_final_sequences_per_task")
            if isinstance(sampling_diversity, Mapping)
            else None
        )
        if (
            not isinstance(sampling_diversity, Mapping)
            or not isinstance(unique_by_task, Mapping)
            or len(unique_by_task) != 16
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= SAMPLES_PER_TASK
                for value in unique_by_task.values()
            )
            or sampling_diversity.get(
                "pathological_all_tasks_have_identical_k8_draws"
            )
            is not False
            or sampling_diversity.get("duplicate_draws_filtered") is not False
            or int(
                sampling_diversity.get(
                    "tasks_with_all_k8_draws_identical", -1
                )
            )
            != sum(value == 1 for value in unique_by_task.values())
            or int(
                sampling_diversity.get(
                    "minimum_unique_final_sequences_per_task", -1
                )
            )
            != min(unique_by_task.values())
            or int(
                sampling_diversity.get(
                    "maximum_unique_final_sequences_per_task", -1
                )
            )
            != max(unique_by_task.values())
        ):
            raise ArtifactError(
                "Qwen pilot quality gate has an invalid diversity contract"
            )
        pilot_target_gate = quality_gate.get("target_length_gate")
        if (
            not isinstance(pilot_target_gate, Mapping)
            or pilot_target_gate.get("passed") is not True
            or int(pilot_target_gate.get("overflow_count", -1)) != 0
            or (
                args.objective_mode != OBJECTIVE_MODE_SEQUENCE_ONLY
                and int(
                    pilot_target_gate.get("non_code_target_count", -1)
                )
                != 0
            )
            or pilot_target_gate.get("final_dart_code_only_required")
            != (args.objective_mode != OBJECTIVE_MODE_SEQUENCE_ONLY)
            or pilot_target_gate.get("truncate") is not False
            or pilot_target_gate.get("filter_draw") is not False
            or pilot_target_gate.get("resample_draw") is not False
            or not isinstance(
                pilot_target_gate.get("max_target_tokens"), int
            )
        ):
            raise ArtifactError(
                "Qwen pilot quality gate lacks the passed objective-specific "
                "target-length/content contract"
            )

    prompts, prompt_record = load_verified_prompt_rows(
        args.prompt_jsonl,
        expected_sha256=args.expected_prompt_sha256,
        expected_rows=args.expected_prompt_rows,
    )
    if args.max_tasks:
        if args.max_tasks < 1:
            raise ArtifactError("--max-tasks must be positive or zero")
        if args.task_selection_strategy == "deterministic_hash":
            prompts = sorted(
                prompts,
                key=lambda row: stable_sha256(
                    {
                        "selection": "qwen-quality-pilot-v1",
                        "seed_base": int(args.seed_base),
                        "task_id": row.task_id,
                    }
                ),
            )
        prompts = prompts[: args.max_tasks]
    if quality_gate is not None:
        expected_pilot_ids = {
            row.task_id
            for row in sorted(
                prompts,
                key=lambda row: stable_sha256(
                    {
                        "selection": "qwen-quality-pilot-v1",
                        "seed_base": int(args.seed_base),
                        "task_id": row.task_id,
                    }
                ),
            )[:16]
        }
        gated_ids = set(
            (
                quality_gate["sampling_diversity"][
                    "unique_final_sequences_per_task"
                ]
            ).keys()
        )
        if gated_ids != expected_pilot_ids:
            raise ArtifactError(
                "Qwen pilot diversity gate was not produced from the exact "
                "deterministic 16-task subset of this prompt artifact"
            )
    task_ids = [prompt.task_id for prompt in prompts]
    tests, verifier_record = load_verifier_rows(
        args.verifier_jsonl,
        expected_sha256=args.expected_verifier_sha256,
        expected_task_ids=set(task_ids),
    )
    binding = StudentTokenizerBinding.from_file(
        args.student_tokenizer_json,
        expected_sha256=args.expected_student_tokenizer_sha256,
        eos_token_id=args.student_eos_token_id,
    )
    target_length_contract = load_target_length_contract(
        args.target_contract,
        expected_sha256=args.expected_target_contract_sha256,
        binding=binding,
    )
    target_length_contract["target_source"][
        "final_dart_code_only_required"
    ] = args.objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
    if quality_gate is not None:
        pilot_target_gate = quality_gate["target_length_gate"]
        if (
            pilot_target_gate.get("target_contract_sha256")
            != target_length_contract["trainer_contract"]["sha256"]
            or int(pilot_target_gate.get("max_target_tokens", -1))
            != int(target_length_contract["max_target_tokens"])
            or not isinstance(
                pilot_target_gate.get("target_length_evidence_sha256"), str
            )
        ):
            raise ArtifactError(
                "Qwen pilot target-length gate differs from the full-run "
                "trainer contract"
            )
    system_prompt, prompt_manifest_record, prompt_manifest = (
        load_f2_prompt_contract(
            args.prompt_manifest,
            expected_sha256=args.expected_prompt_manifest_sha256,
            prompt_record=prompt_record,
            expected_rows=(
                int(args.expected_prompt_rows)
                if args.expected_prompt_rows is not None
                else len(prompts)
            ),
            student_tokenizer_sha256=binding.tokenizer_record["sha256"],
        )
    )
    system_prompt_sha256 = sha256_text(system_prompt)
    if any(
        prompt.representation_schema != "lossless-semantic-f2"
        or prompt.system_prompt_sha256 != system_prompt_sha256
        for prompt in prompts
    ):
        raise ArtifactError(
            "one or more F2 rows differs from the manifest-bound prompt contract"
        )
    verifier, verifier_implementation = make_dart_verifier(
        tests,
        timeout_seconds=args.verifier_timeout_seconds,
    )
    prompt_token_counts: dict[str, dict[str, int]] = {}
    over_budget: list[tuple[int, str]] = []
    for prompt in prompts:
        count = count_prompt_tokens(
            build_messages(system_prompt, prompt),
            binding.tokenizer,
            chat_overhead_reserve=args.chat_overhead_reserve,
        )
        if count["estimated_prompt_tokens"] > args.max_prompt_tokens:
            over_budget.append(
                (count["estimated_prompt_tokens"], prompt.task_id)
            )
        prompt_token_counts[prompt.task_id] = count
    if over_budget:
        worst = sorted(over_budget, reverse=True)
        examples = ", ".join(
            f"{task_id}:{tokens}" for tokens, task_id in worst[:8]
        )
        raise ArtifactError(
            f"{len(over_budget)} prompts exceed the {args.max_prompt_tokens}-token "
            f"cap including reserve; maximum={worst[0][0]}; "
            f"worst={examples}; no truncation is permitted"
        )
    generation_parameters = {
        "n": 1,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "extra_body": {
            "enable_thinking": bool(args.enable_thinking),
            # top_k is Alibaba-specific and must travel inside extra_body;
            # the OpenAI SDK rejects it as an unknown top-level argument.
            "top_k": int(args.top_k),
        },
    }
    if args.enable_thinking:
        generation_parameters["extra_body"]["thinking_budget"] = int(
            args.thinking_budget
        )
    if args.objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5:
        generation_parameters.update(
            {"logprobs": True, "top_logprobs": TOP_LOGPROBS}
        )
    elif args.objective_mode != OBJECTIVE_MODE_SEQUENCE_ONLY:
        raise AssertionError("argparse accepted an unknown objective mode")
    prompt_bindings = [
        {
            "task_id": prompt.task_id,
            "serializer_text_sha256": prompt.text_sha256,
            "source_record_sha256": prompt.source_record_sha256,
            "request_messages_sha256": stable_sha256(
                build_messages(system_prompt, prompt)
            ),
            "token_count": prompt_token_counts[prompt.task_id],
        }
        for prompt in prompts
    ]
    header_payload = {
        "collection_mode": "offline_precompute_only_no_gradient_loop",
        "prompt_artifact": prompt_record,
        "prompt_manifest": prompt_manifest_record,
        "f2_prompt_contract": dict(prompt_manifest["f2_prompt_contract"]),
        "task_ids": task_ids,
        "task_selection": {
            "max_tasks": int(args.max_tasks),
            "strategy": args.task_selection_strategy,
        },
        "task_set_sha256": stable_sha256(task_ids),
        "prompt_bindings": prompt_bindings,
        "prompt_row_seals_sha256": stable_sha256(prompt_bindings),
        "student_tokenizer": binding.tokenizer_record,
        "student_eos_token_id": binding.eos_token_id,
        "target_length_contract": target_length_contract,
        "verifier_artifact": verifier_record,
        "verifier_implementation": verifier_implementation,
        "requested_model": args.model,
        "returned_model_must_equal_requested": True,
        "objective_mode": args.objective_mode,
        "base_url": args.base_url.rstrip("/"),
        "provider_authorization": {
            "token_plan_automation_authorized": bool(
                args.token_plan_automation_authorized
            ),
            "attested_by": "workspace_operator",
            "scope": "automated_research_teacher_harvest",
        },
        "transport": {
            "timeout_seconds": float(args.timeout_seconds),
            "sdk_max_retries": 0,
            "application_max_retries_per_slot": int(args.max_retries),
            "api_mode": "synchronous_chat_completions_n_equals_1",
            "api_workers": int(args.workers),
            "local_verifier_workers": int(args.verifier_workers),
            "length_capped_response_policy": {
                "same_task_draw_only": True,
                "completed_draws_reissued": False,
                "max_token_capacities": length_capacities,
                "capped_responses_retained_by_hash": True,
            },
        },
        "implementation": {
            "collector": file_record(Path(__file__).resolve()),
            "artifact_core": file_record(
                Path(__file__).resolve().with_name(
                    "qwen_direct_compact_teacher_artifact.py"
                )
            ),
        },
        "system_prompt_sha256": system_prompt_sha256,
        "prompt_budget": {
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "chat_overhead_reserve": int(args.chat_overhead_reserve),
            "maximum_estimated_prompt_tokens": max(
                (
                    count["estimated_prompt_tokens"]
                    for count in prompt_token_counts.values()
                ),
                default=0,
            ),
            "truncation_permitted": False,
        },
        "required_function": args.required_function,
        "samples_per_task": SAMPLES_PER_TASK,
        "independence": (
            "eight_separate_n1_sampled_api_requests_with_distinct_requested_"
            "seeds; provider_seed_honor_is_not_assumed_without_response_echo"
        ),
        "sampling_seed_contract": {
            "algorithm": SAMPLE_SEED_ALGORITHM,
            "seed_base": int(args.seed_base),
            "unique_seed_required_per_task_draw": True,
            "provider_seed_honor_not_assumed": True,
            "response_seed_echo_required_to_attest_honor": True,
        },
        "generation_parameters": generation_parameters,
        "pilot_quality_gate": quality_gate_record,
        "pilot_quality_gate_contract": (
            {
                key: quality_gate[key]
                for key in (
                    "schema",
                    "passed",
                    "pilot_tasks",
                    "candidates",
                    "parseable_candidates",
                    "verified_tasks",
                    "minimum_verified_tasks",
                    "minimum_parseable_fraction",
                    "sampling_diversity",
                    "target_length_gate",
                    "pilot_audit_sha256",
                    "pilot_verified_only_sha256",
                )
            }
            if quality_gate is not None
            else None
        ),
        "objective_contract": {
            "monte_carlo_sequence_forward_kl_nll": True,
            "untempered_untruncated_teacher_sampling": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 101,
            "correctness_conditioned_rs_sft": True,
            "provider_top5_logprobs_required": (
                args.objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
            ),
            "hidden_reasoning_tokens_enabled": bool(args.enable_thinking),
            "content_logprob_prefix_fully_visible_to_student": (
                not args.enable_thinking
            ),
            "sparse_top5_tail_auxiliary_possible": (
                args.objective_mode == OBJECTIVE_MODE_REQUIRE_TOP5
            ),
            "dense_full_vocabulary_kl": False,
        },
    }
    if args.dry_run:
        print(
            "QWEN_TEACHER_DRY_RUN "
            f"tasks={len(prompts)} K={SAMPLES_PER_TASK} model={args.model} "
            f"objective_mode={args.objective_mode} "
            f"max_prompt_tokens="
            f"{header_payload['prompt_budget']['maximum_estimated_prompt_tokens']} "
            f"header_sha256={stable_sha256(header_payload)}",
            flush=True,
        )
        return 0
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ArtifactError(
            f"API key environment variable {args.api_key_env!r} is not set"
        )
    try:
        from openai import OpenAI
    except Exception as exc:
        raise ArtifactError("the openai package is required for collection") from exc
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=float(args.timeout_seconds),
        max_retries=0,
    )
    state = collect_candidates(
        prompts=prompts,
        client=client,
        journal_path=args.journal,
        header_payload=header_payload,
        system_prompt=system_prompt,
        requested_model=args.model,
        generation_parameters=generation_parameters,
        required_function=args.required_function,
        verifier=verifier,
        max_retries=args.max_retries,
        allow_homogeneous_shards=args.split_homogeneous_shards,
        workers=args.workers,
        verifier_workers=args.verifier_workers,
        progress_every=args.progress_every,
        seed_base=args.seed_base,
        require_returned_model_exact=True,
        authorize_orphan_reissue_with_duplicate_billing_risk=(
            args.authorize_orphan_reissue_with_duplicate_billing_risk
        ),
    )
    audit = materialize_artifacts(
        journal_path=args.journal,
        binding=binding,
        parseable_output=args.parseable_output,
        rs_sft_output=args.rs_sft_output,
        audit_output=args.audit_output,
        allow_homogeneous_shards=args.split_homogeneous_shards,
    )
    print(
        "QWEN_TEACHER_COLLECTION "
        f"tasks={len(task_ids)} candidates={len(state.candidates)} "
        f"parseable={audit['coverage']['parseable_candidates']} "
        f"rs_sft={audit['coverage']['rs_sft_candidates']} "
        f"target_overflows={audit['target_length_gate']['overflow_count']} "
        f"non_code_targets="
        f"{audit['target_length_gate']['non_code_target_count']} "
        f"sealed_audit={args.audit_output} "
        f"production_ready={audit['production_ready']}",
        flush=True,
    )
    return 0 if audit["production_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
