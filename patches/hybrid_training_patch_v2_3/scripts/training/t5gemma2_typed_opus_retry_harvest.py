#!/usr/bin/env python3
"""Fail-closed Claude Opus retry harvest for typed direct RS-SFT.

This extension consumes the completed typed Kimi/Kimi-retry/Sonnet cascade and
schedules one Claude Opus candidate for each of the exact 16 Sonnet responses
that were non-code or ended at the 16K token ceiling.  All previously verified
identities remain excluded.  It deliberately reuses the typed
cascade's sealed input loaders, prompt builder, durable provider transaction,
visible-then-private verifier, and direct-only publisher.

The held-out 175-task evaluation split may be opened only by the inherited
lineage audit that proves zero overlap.  It is never model-visible and is not
consulted for eligibility, ordering, selection, verification, or transfer.
"""

from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.graph_compile_at_k_antigravity import validate_dart_binary
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade


PHASE = "opus_retry"
MODEL = "claude-opus-5"
TASKS = 16
MAX_INPUT_TOKENS = 32_768
MAX_OUTPUT_TOKENS = 8_192
MAX_INPUT_TOTAL = TASKS * MAX_INPUT_TOKENS
MAX_OUTPUT_TOTAL = TASKS * MAX_OUTPUT_TOKENS
MAX_TOTAL = MAX_INPUT_TOTAL + MAX_OUTPUT_TOTAL
INPUT_USD_PER_MILLION = Decimal("5")
OUTPUT_USD_PER_MILLION = Decimal("25")
MAX_USD = Decimal("5.89824")
EXPECTED_RETRY_TASK_IDS_SHA256 = (
    "839ac9fa414f0434b68561f76acb3f3a0ca3a0f4a8588a476e0d09684a5428fe"
)
EXPECTED_PRIOR_PHASES = (
    (cascade.PHASE_KIMI_INITIAL, 0),
    (cascade.PHASE_KIMI_RETRY, 0),
    (cascade.PHASE_SONNET_RESIDUAL, 0),
)


def validate_profile(args: argparse.Namespace) -> None:
    """Reject any relaxation of the fixed 16-call Opus retry pilot."""

    exact = {
        "provider": "anthropic",
        "model": MODEL,
        "api_key_env": "ANTHROPIC_API_KEY",
        "anthropic_thinking": "adaptive",
        "anthropic_effort": "high",
        "seed": 20260801,
        "max_tasks": TASKS,
        "max_parents_per_task": 1,
        "samples_per_parent": 1,
        "max_calls": TASKS,
        "max_input_tokens_per_call": MAX_INPUT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_input_tokens_total": MAX_INPUT_TOTAL,
        "max_output_tokens_total": MAX_OUTPUT_TOTAL,
        "max_total_tokens": MAX_TOTAL,
        "stability_runs": cascade.STABILITY_RUNS,
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(f"Opus retry fixes --{name}={expected}")
    if (
        args.evaluation_only
        or args.exploratory_terminal_prefix
        or args.allow_unpinned_inputs
        or args.retry_parse_failures_or_truncations_report
    ):
        raise ValueError("Opus retry permits only pinned direct production mode")
    if not args.abort_on_provider_error:
        raise ValueError("Opus retry requires --abort_on_provider_error")
    if len(args.prior_success_report) != len(EXPECTED_PRIOR_PHASES):
        raise ValueError("Opus retry requires exactly three completed cascade reports")
    prices = (
        Decimal(str(args.max_usd)),
        Decimal(str(args.input_usd_per_million)),
        Decimal(str(args.output_usd_per_million)),
    )
    if prices != (MAX_USD, INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION):
        raise ValueError("Opus retry token-price reservation differs")


def select_exact_sonnet_retry(
    *,
    all_api_eligible: Sequence[tuple[int, Any, Mapping[str, Any]]],
    prior_records: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, Any, Mapping[str, Any]]], dict[str, Any]]:
    """Select only the sealed Sonnet non-code/length terminal set."""

    phases = tuple(
        (str(row.get("phase") or ""), int(row.get("cohort_index", -1)))
        for row in prior_records
    )
    if phases != EXPECTED_PRIOR_PHASES:
        raise ValueError("prior cascade phase order/completeness differs")
    eligible_ids = [row[1].task_id for row in all_api_eligible]
    by_id = {row[1].task_id: row for row in all_api_eligible}
    eligible_set = set(by_id)
    scheduled_union: set[str] = set()
    verified_union: set[str] = set()
    phase_accounting: list[dict[str, Any]] = []
    for record in prior_records:
        scheduled = [str(value) for value in record["scheduled_task_ids"]]
        verified = [str(value) for value in record["verified_task_ids"]]
        if not set(verified).issubset(set(scheduled)):
            raise ValueError("prior verified identity is absent from its schedule")
        if not set(scheduled).issubset(eligible_set):
            raise ValueError("prior cascade scheduled a non-eligible task")
        scheduled_union.update(scheduled)
        verified_union.update(verified)
        phase_accounting.append(
            {
                "phase": record["phase"],
                "cohort_index": record["cohort_index"],
                "scheduled_tasks": len(scheduled),
                "scheduled_task_ids_sha256": canonical_sha256(scheduled),
                "verified_tasks": len(verified),
                "verified_task_ids_sha256": canonical_sha256(verified),
            }
        )
    if not verified_union.issubset(scheduled_union):
        raise ValueError("prior verified union is absent from prior schedule union")
    sonnet = prior_records[-1]
    retry_ids = [str(value) for value in sonnet["retry_eligible_task_ids"]]
    if (
        len(retry_ids) != TASKS
        or len(set(retry_ids)) != TASKS
        or canonical_sha256(retry_ids) != EXPECTED_RETRY_TASK_IDS_SHA256
        or not set(retry_ids).issubset(set(sonnet["scheduled_task_ids"]))
        or set(retry_ids) & verified_union
        or any(task_id not in by_id for task_id in retry_ids)
    ):
        raise ValueError("exact Sonnet non-code/length retry cohort differs")
    selected = [by_id[task_id] for task_id in retry_ids]
    return selected, {
        "mode": PHASE,
        "retry_source_phase": cascade.PHASE_SONNET_RESIDUAL,
        "targeted_non_code_or_length_only": True,
        "accepted_nontruncated_sonnet_responses_regenerated": False,
        "prior_verified_excluded": True,
        "prior_phase_accounting": phase_accounting,
        "prior_scheduled_unique_tasks_excluded": len(scheduled_union),
        "prior_scheduled_unique_task_ids_sha256": canonical_sha256(
            [task_id for task_id in eligible_ids if task_id in scheduled_union]
        ),
        "prior_verified_unique_tasks_excluded": len(verified_union),
        "prior_verified_unique_task_ids_sha256": canonical_sha256(
            [task_id for task_id in eligible_ids if task_id in verified_union]
        ),
        "retry_tasks": TASKS,
        "retry_task_ids_sha256": EXPECTED_RETRY_TASK_IDS_SHA256,
        "automatic_followup_retry": False,
        "selection_uses_heldout_175": False,
    }


def run(
    args: argparse.Namespace,
    *,
    transport: base.ProviderTransport | None = None,
    evaluate: base.EvaluateFn | None = None,
) -> dict[str, Any]:
    validate_profile(args)
    context = cascade.load_typed_source_context(args)
    projection_terminals, projection_record = cascade.load_visible_projection(
        args, context=context
    )
    existing_ids, existing_record = cascade.load_existing_225_exclusions(
        args.existing_direct_manifest,
        args.expected_existing_direct_manifest_sha256,
    )
    input_record = {
        "source": context.input_record,
        "visible_failure_projection": projection_record,
        "existing_225_exclusion": existing_record,
    }
    prior_records = cascade.load_prior_cascade_reports(
        report_paths=list(args.prior_success_report),
        expected_sha256s=list(args.expected_prior_success_report_sha256),
        input_record=input_record,
        source_journal_record=context.source_journal_record,
    )
    all_api_eligible = cascade.select_visible_zero_tasks(
        context=context,
        projection_terminals=projection_terminals,
        seed=args.seed,
        excluded_ids=set(existing_ids),
    )
    selected, selection_record = select_exact_sonnet_retry(
        all_api_eligible=all_api_eligible,
        prior_records=prior_records,
    )
    selected_ids = {row[1].task_id for row in selected}
    if selected_ids & set(existing_ids) or cascade.KNOWN_CONTAMINANT in selected_ids:
        raise ValueError("Opus retry overlaps the sealed 225 or known contaminant")
    plans, diagnostic_record = cascade.build_visible_only_plans(
        selected=selected,
        gates=context.gates,
    )
    if len(plans) != TASKS:
        raise ValueError("Opus retry lacks one usable visible-only parent per task")

    input_price = Decimal(str(args.input_usd_per_million))
    output_price = Decimal(str(args.output_usd_per_million))
    max_usd = Decimal(str(args.max_usd))
    capacity, budget_contract = base.schedule_capacity(
        max_calls=args.max_calls,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        max_output_tokens_per_call=args.max_output_tokens,
        max_input_tokens_total=args.max_input_tokens_total,
        max_output_tokens_total=args.max_output_tokens_total,
        max_total_tokens=args.max_total_tokens,
        max_usd=max_usd,
        input_usd_per_million=input_price,
        output_usd_per_million=output_price,
    )
    if capacity != TASKS:
        raise ValueError("Opus reservation does not cover exactly 16 complete calls")
    slots = cascade.build_typed_slots(plans, samples_per_parent=1)
    if len(slots) != TASKS:
        raise ValueError("Opus pilot is not exactly one candidate per task")
    prompt_byte_upper_bounds = [
        len(slot.prompt.encode("utf-8"))
        + len(base.SYSTEM_PROMPT.encode("utf-8"))
        + 1024
        for slot in slots
    ]
    max_prompt_byte_upper_bound = max(prompt_byte_upper_bounds)
    if max_prompt_byte_upper_bound > args.max_input_tokens_per_call:
        raise ValueError(
            "Opus prompt byte upper bound exceeds the sealed input reservation"
        )
    schedule_ids = [plan.task.task_id for plan in plans]
    schedule_sha = canonical_sha256(schedule_ids)
    if args.expected_scheduled_task_ids_sha256:
        cascade._require_digest(  # noqa: SLF001
            args.expected_scheduled_task_ids_sha256,
            "expected Opus scheduled task IDs",
        )
        if schedule_sha != args.expected_scheduled_task_ids_sha256:
            raise ValueError("Opus scheduled task digest differs")

    if args.plan_only_output:
        plan_record = {
            "schema": cascade.PLAN_SCHEMA,
            "status": "complete",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "phase": PHASE,
            "cohort_index": 0,
            "inputs_sha256": canonical_sha256(input_record),
            "source_journal_sha256": context.source_journal_record.get("sha256"),
            "prior_reports": [
                {
                    "report_sha256": row["report_sha256"],
                    "phase": row["phase"],
                    "cohort_index": row["cohort_index"],
                    "journal_sha256": row["journal_sha256"],
                    "targets_sha256": row["targets_sha256"],
                }
                for row in prior_records
            ],
            "selection": {
                **selection_record,
                "scheduled_tasks": TASKS,
                "scheduled_calls": TASKS,
                "task_ids": schedule_ids,
                "task_ids_sha256": schedule_sha,
                "max_prompt_byte_upper_bound": max_prompt_byte_upper_bound,
                "prompt_byte_upper_bounds_sha256": canonical_sha256(
                    prompt_byte_upper_bounds
                ),
            },
            "budget": budget_contract,
            "provider_credentials_read": False,
            "frontier_api_calls": False,
            "heldout_175_used_for_generation_or_selection": False,
        }
        path = Path(args.plan_only_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        require_exact_or_write(path, plan_record)
        print(json.dumps(plan_record, sort_keys=True), flush=True)
        return plan_record

    if not args.expected_scheduled_task_ids_sha256:
        raise ValueError("live Opus run requires a schedule digest from credential-free planning")
    base_url = base.validate_provider_endpoint(
        provider=args.provider,
        base_url=args.base_url,
        api_version=args.api_version,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = (
        "repair_policy_targets.jsonl",
        "repair_policy_sources.jsonl",
        "repair_policy_manifest.json",
        "direct_hard_targets_f2.jsonl",
    )
    if any((output_dir / name).exists() for name in forbidden):
        raise ValueError("Opus direct-only output directory contains forbidden artifacts")
    journal_path = output_dir / "typed_api_rescue.journal.jsonl"
    provider_contract = base._provider_contract(args, base_url)  # noqa: SLF001
    contract = {
        "schema": cascade.RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "phase": PHASE,
        "cohort_index": 0,
        "source_local_harvest_journal": context.source_journal_record,
        "inputs": input_record,
        "prior_reports": [
            {
                key: row[key]
                for key in (
                    "report_sha256",
                    "phase",
                    "cohort_index",
                    "journal_sha256",
                    "targets_sha256",
                )
            }
            for row in prior_records
        ],
        "selection": {
            **selection_record,
            "seed": args.seed,
            "eligibility": "sealed_visible_TRAIN_all_zero_K4_only",
            "private_complete_outcome_used_for_eligibility": False,
            "api_eligible_tasks": len(all_api_eligible),
            "scheduled_tasks": TASKS,
            "scheduled_slots": TASKS,
            "task_ids_sha256": schedule_sha,
            "max_prompt_byte_upper_bound": max_prompt_byte_upper_bound,
            "prompt_byte_upper_bounds_sha256": canonical_sha256(
                prompt_byte_upper_bounds
            ),
            "slot_bindings_sha256": canonical_sha256(
                [base._slot_binding(slot) for slot in slots]  # noqa: SLF001
            ),
            "max_parents_per_task": 1,
            "samples_per_parent": 1,
        },
        "visible_diagnostic_provenance": diagnostic_record,
        "provider": provider_contract,
        "budget": budget_contract,
        "verification": {
            "visible_before_private": True,
            "all_api_calls_before_any_private_gate": True,
            "private_gate": "complete_TRAIN_acceptance",
            "stability_runs": cascade.STABILITY_RUNS,
            "private_failure_triggers_api_call": False,
            "private_gate_can_only_reject_transfer": True,
        },
        "privacy": {
            "api_input_fields": [
                "opaque_typed_contract",
                "compressed_enriched_F2",
                "failed_local_candidate",
                "visible_split_derived_diagnostic",
                "visible_TRAIN_checks",
            ],
            "private_complete_acceptance_sent_to_provider": False,
            "private_split_holdback_sent_to_provider": False,
            "gold_sent_to_provider": False,
            "heldout_175_model_visible": False,
            "heldout_175_used_for_generation_or_selection": False,
            "api_credentials_persisted": False,
            "plaintext_reasoning_persisted": False,
        },
        "training_outputs": {
            "direct_verified_code_targets": True,
            "repair_conditioned_rows": 0,
            "gold_replay_rows": 0,
            "reasoning_rows": 0,
            "tests_in_training_outputs": False,
            "diagnostics_in_training_outputs": False,
            "production_floor_eligible": True,
        },
        "heldout_175_opened_for_lineage_overlap_audit": True,
        "heldout_175_used_for_generation_or_selection": False,
    }
    base._assert_secret_free(contract)  # noqa: SLF001

    with cascade._typed_base_schemas():  # noqa: SLF001
        events = load_journal(journal_path)
        if not events:
            base._append_safe(  # noqa: SLF001
                journal_path,
                {
                    "event": "header",
                    "schema": cascade.JOURNAL_SCHEMA,
                    "contract": contract,
                    "contract_sha256": canonical_sha256(contract),
                },
            )
        else:
            base.validate_rescue_journal(
                events,
                contract=contract,
                plans=plans,
                slots=slots,
            )
        api_key = str(os.environ.get(args.api_key_env) or "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is empty")
        if transport is None:
            transport = base._build_transport(  # noqa: SLF001
                args,
                api_key=api_key,
                base_url=base_url,
            )
        slot_results = base.execute_api_phase(
            journal_path=journal_path,
            contract=contract,
            plans=plans,
            slots=slots,
            transport=transport,
            api_key=api_key,
            max_input_tokens=args.max_input_tokens_per_call,
            max_output_tokens=args.max_output_tokens,
            input_usd_per_million=input_price,
            output_usd_per_million=output_price,
            inter_call_delay_seconds=args.inter_call_delay_seconds,
            abort_on_provider_error=args.abort_on_provider_error,
            provider_max_attempts=args.provider_max_attempts,
            provider_retry_base_seconds=args.provider_retry_base_seconds,
            provider_retry_max_seconds=args.provider_retry_max_seconds,
        )
        if evaluate is None:
            validate_dart_binary()
            evaluate = base._runtime_evaluator(  # noqa: SLF001
                timeout=args.timeout,
                stability_runs=cascade.STABILITY_RUNS,
            )
        verifications = base.execute_verification_phase(
            journal_path=journal_path,
            contract=contract,
            plans=plans,
            slots=slots,
            evaluate=evaluate,
            api_key=api_key,
        )
        final_state = base.validate_rescue_journal(
            load_journal(journal_path),
            contract=contract,
            plans=plans,
            slots=slots,
        )
    if not final_state["complete"]:
        raise RuntimeError("Opus retry journal did not complete")

    contract_sha = canonical_sha256(contract)
    outputs = cascade.publish_direct_only(
        output_dir=output_dir,
        plans=plans,
        verifications=verifications,
        contract_sha256=contract_sha,
        provider_phase=PHASE,
        provider_model=args.model,
        stability_runs=cascade.STABILITY_RUNS,
    )
    charged_input = sum(row["usage"]["charged_input_tokens"] for row in slot_results)
    charged_output = sum(row["usage"]["charged_output_tokens"] for row in slot_results)
    charged_nanos = sum(row["usage"]["charged_usd_nanos"] for row in slot_results)
    verified_ids = [row["task_id"] for row in outputs["rows"]]
    verified_set = set(verified_ids)
    retry_eligible: list[str] = []
    for row in slot_results:
        response = row.get("response")
        finish = (
            str(response.get("finish_reason") or "")
            if isinstance(response, Mapping)
            else ""
        )
        task_id = str(row.get("task_id") or "")
        if task_id not in verified_set and (
            row.get("parse_accepted") is not True or finish == "length"
        ):
            retry_eligible.append(task_id)
    report = {
        "schema": cascade.REPORT_SCHEMA,
        "status": "complete",
        "phase": PHASE,
        "cohort_index": 0,
        "run_contract_sha256": contract_sha,
        "provider": provider_contract,
        "schedule": {
            "api_eligible_tasks": len(all_api_eligible),
            "scheduled_tasks": TASKS,
            "scheduled_calls": TASKS,
            "task_ids_sha256": schedule_sha,
            "provider_responses": sum(
                row.get("status") == "response" for row in slot_results
            ),
            "code_only_responses": sum(
                row.get("parse_accepted") is True for row in slot_results
            ),
            "retry_eligible_non_code_or_length_tasks": len(retry_eligible),
            "retry_eligible_task_ids_sha256": canonical_sha256(retry_eligible),
        },
        "verification": {
            "visible_passes": sum(
                candidate["passed"]
                for event in verifications
                for candidate in event["visible_results"]
            ),
            "private_full_acceptance_passes": len(verified_ids),
            "verified_unique_hard_targets": len(verified_ids),
            "verified_task_ids_sha256": canonical_sha256(verified_ids),
        },
        "budget_charged": {
            "calls": len(slot_results),
            "input_tokens": charged_input,
            "output_tokens": charged_output,
            "total_tokens": charged_input + charged_output,
            "estimated_usd_nanos": charged_nanos,
            "estimated_usd": (
                f"{Decimal(charged_nanos) / Decimal(1_000_000_000):.9f}"
            ),
            "unknown_usage_failures_charged_at_full_reservation": True,
            "within_contract": charged_nanos
            <= int(
                (max_usd * Decimal(1_000_000_000)).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        },
        "outputs": outputs["files"],
        "direct_manifest": outputs["manifest"],
        "repair_policy_manifest": None,
        "cohort_decision": None,
        "journal": journal_record(journal_path),
        "privacy_invariants": contract["privacy"],
        "heldout_175_opened_for_lineage_overlap_audit": True,
        "heldout_175_model_visible": False,
        "heldout_175_used_for_generation_or_selection": False,
    }
    base._assert_secret_free(report, api_key=api_key)  # noqa: SLF001
    require_exact_or_write(output_dir / "typed_api_rescue_report.json", report)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "tasks": TASKS,
                "verified_targets": len(verified_ids),
                "estimated_usd": report["budget_charged"]["estimated_usd"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--local_harvest_report", required=True)
    pre.add_argument("--expected_local_harvest_report_sha256", required=True)
    pre.add_argument("--expected_local_harvest_journal_sha256", required=True)
    pre.add_argument("--local_harvest_targets", required=True)
    pre.add_argument("--expected_local_harvest_targets_sha256", required=True)
    pre.add_argument("--existing_direct_manifest", required=True)
    pre.add_argument("--expected_existing_direct_manifest_sha256", required=True)
    pre.add_argument("--gold_train_jsonl", required=True)
    pre.add_argument("--expected_gold_train_sha256", required=True)
    pre.add_argument("--gold_f2_jsonl", required=True)
    pre.add_argument("--expected_gold_f2_sha256", required=True)
    pre.add_argument("--gold_train_rows", dest="expected_gold_rows", type=int, default=2776)
    pre.add_argument("--heldout_jsonl", required=True)
    pre.add_argument("--expected_heldout_sha256", required=True)
    pre.add_argument("--expected_heldout_rows", type=int, default=175)
    pre.add_argument("--visible_split_manifest", required=True)
    pre.add_argument("--expected_visible_split_manifest_sha256", required=True)
    pre.add_argument("--visible_train", required=True)
    pre.add_argument("--expected_visible_train_sha256", required=True)
    pre.add_argument("--private_split_holdback", required=True)
    pre.add_argument("--expected_private_split_holdback_sha256", required=True)
    pre.add_argument("--visible_projection_report", required=True)
    pre.add_argument("--expected_visible_projection_report_sha256", required=True)
    pre.add_argument("--visible_projection_journal", required=True)
    pre.add_argument("--expected_visible_projection_journal_sha256", required=True)
    pre.add_argument("--plan_only_output", default="")
    extras, remaining = pre.parse_known_args(argv)
    args = base.parse_args(remaining)
    for name, value in vars(extras).items():
        setattr(args, name, value)
    for name in (
        "expected_local_harvest_report_sha256",
        "expected_local_harvest_journal_sha256",
        "expected_local_harvest_targets_sha256",
        "expected_existing_direct_manifest_sha256",
        "expected_gold_train_sha256",
        "expected_gold_f2_sha256",
        "expected_heldout_sha256",
        "expected_visible_split_manifest_sha256",
        "expected_visible_train_sha256",
        "expected_private_split_holdback_sha256",
        "expected_visible_projection_report_sha256",
        "expected_visible_projection_journal_sha256",
    ):
        cascade._require_digest(getattr(args, name), name)  # noqa: SLF001
    aliases = (
        (args.rollout_file, args.visible_train, "rollout_file/visible_train"),
        (args.f2_jsonl, args.gold_f2_jsonl, "f2_jsonl/gold_f2_jsonl"),
        (
            args.private_holdback,
            args.private_split_holdback,
            "private_holdback/private_split_holdback",
        ),
    )
    for left, right, label in aliases:
        if Path(left).expanduser().resolve() != Path(right).expanduser().resolve():
            raise ValueError(f"{label} aliases must identify the same pinned file")
    if (
        args.expected_rollout_sha256 != args.expected_visible_train_sha256
        or args.expected_f2_sha256 != args.expected_gold_f2_sha256
        or args.expected_private_holdback_sha256
        != args.expected_private_split_holdback_sha256
    ):
        raise ValueError("base compatibility digest aliases differ")
    validate_profile(args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
