#!/usr/bin/env python3
"""Resume the exact unpaid 47-call tail of interrupted typed Kimi cohort 2."""

from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_dual_api_orchestrator as dual
from scripts.training import t5gemma2_typed_kimi_continuation_c002 as c002
from scripts.training import t5gemma2_typed_api_rescue_c002_resume47 as adapter


RUN_SCHEMA = "t5gemma2-typed-kimi-c002-resume47-run-v1"
REPORT_SCHEMA = "t5gemma2-typed-kimi-c002-resume47-report-v1"
STATED_BALANCE = Decimal("12.44")
EXPECTED_PRE_PREFIX_SPEND = Decimal("4.230768")
PREFIX_SPEND = Decimal("0.094941")
EXPECTED_CUMULATIVE_PREFIX_SPEND = Decimal("4.325709")
INITIAL_WORST_USD = Decimal("7.219200")
RETRY_WORST_USD_PER_TASK = Decimal("0.215040")
MAX_RETRY_TASKS = 4


def _retry_should_skip(*, retry_tasks: int, remaining: Decimal) -> bool:
    if retry_tasks <= 0:
        return False
    required = RETRY_WORST_USD_PER_TASK * retry_tasks
    return retry_tasks > MAX_RETRY_TASKS or required > remaining


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _append_once(path: Path, event: Mapping[str, Any]) -> None:
    identity = str(event["event"])
    rows = [row for row in load_journal(path) if row.get("event") == identity]
    if not rows:
        append_event(path, dict(event))
    elif len(rows) != 1 or any(rows[0].get(key) != value for key, value in event.items()):
        raise ValueError(f"resume controller {identity} evidence differs")


def execute_phase(
    *,
    launcher: Path,
    root: Path,
    journal_path: Path,
    phase: str,
    output_dir: Path,
    max_usd: Decimal,
    max_tasks: int,
    max_output_tokens: int,
    base_env: Mapping[str, str],
    invoke: Callable[[Path, Mapping[str, str]], None],
    source_plan: Path,
    source_journal: Path,
    source_chain_head: Path,
    prior_index: Mapping[str, Any] | None = None,
    retry_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env = dual._phase_env(  # noqa: SLF001
        base_env,
        phase=phase,
        output_dir=output_dir,
        max_usd=max_usd,
        max_tasks=max_tasks,
        prior_index=prior_index,
        retry_source=retry_source,
    )
    preflight_path = root / f"prompt_preflight_{phase}.json"
    env.update(
        {
            "T5GEMMA_TYPED_API_COHORT_INDEX": "2",
            "T5GEMMA_TYPED_API_FIXED_KIMI_COHORT_LIMIT": "3",
            "T5GEMMA_TYPED_API_MAX_INPUT_TOKENS_PER_CALL": str(
                adapter.MAX_INPUT_TOKENS
            ),
            "T5GEMMA_TYPED_API_INITIAL_MAX_OUTPUT_TOKENS": str(max_output_tokens),
            "T5GEMMA_TYPED_C002_SOURCE_PLAN": str(source_plan),
            "T5GEMMA_TYPED_C002_SOURCE_JOURNAL": str(source_journal),
            "T5GEMMA_TYPED_C002_SOURCE_CHAIN_HEAD": str(source_chain_head),
            "T5GEMMA_TYPED_C002_PROMPT_PREFLIGHT": str(preflight_path),
        }
    )
    plan_path = root / f"plan_{phase}.json"
    plan_env = dict(env)
    plan_env.pop("OPENROUTER_API_KEY", None)
    plan_env["T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT"] = str(plan_path)
    invoke(launcher, plan_env)
    plan = _read_object(plan_path, "resume phase plan")
    preflight = _read_object(preflight_path, "resume prompt preflight")
    expected_schedule = (
        adapter.TAIL_SCHEDULE_SHA256
        if phase == cascade.PHASE_KIMI_INITIAL
        else str(plan.get("selection", {}).get("task_ids_sha256") or "")
    )
    if (
        plan.get("schema") != cascade.PLAN_SCHEMA
        or plan.get("status") != "complete"
        or plan.get("phase") != phase
        or plan.get("cohort_index") != 2
        or plan.get("provider_credentials_read") is not False
        or plan.get("frontier_api_calls") is not False
        or plan.get("selection", {}).get("scheduled_tasks") != max_tasks
        or plan.get("selection", {}).get("task_ids_sha256") != expected_schedule
        or plan.get("budget", {}).get("max_input_tokens_per_call")
        != adapter.MAX_INPUT_TOKENS
        or plan.get("budget", {}).get("max_output_tokens_per_call")
        != max_output_tokens
        or preflight.get("schema") != adapter.PREFLIGHT_SCHEMA
        or preflight.get("phase") != phase
        or preflight.get("slots_checked") != max_tasks
        or preflight.get("task_ids_sha256") != expected_schedule
        or preflight.get("within_reservation") is not True
        or preflight.get("all_selected_slots_checked_before_first_live_call")
        is not True
    ):
        raise ValueError("credential-free resume plan/preflight differs")
    completed = [
        row
        for row in load_journal(journal_path)
        if row.get("event") == f"{phase}_complete"
    ]
    if len(completed) > 1:
        raise ValueError("duplicate resume phase completion")
    if not completed:
        _append_once(
            journal_path,
            {
                "event": f"{phase}_start",
                "plan_sha256": sha256_file(plan_path),
                "preflight_sha256": sha256_file(preflight_path),
                "schedule_sha256": expected_schedule,
                "max_tasks": max_tasks,
                "max_usd": format(max_usd, "f"),
            },
        )
        run_env = dict(env)
        run_env["T5GEMMA_TYPED_API_SCHEDULE_SHA256"] = expected_schedule
        run_env["T5GEMMA_TYPED_C002_PROMPT_PREFLIGHT_SHA256"] = sha256_file(
            preflight_path
        )
        invoke(launcher, run_env)
    record = c002.c001.inspect_phase_report(
        output_dir / "typed_api_rescue_report.json",
        expected_phase=phase,
        expected_cohort=2,
    )
    if record["spent"] > max_usd:
        raise ValueError("resume phase exceeded its exact reservation")
    _append_once(
        journal_path,
        {
            "event": f"{phase}_complete",
            "report_sha256": record["report_sha256"],
            "journal_sha256": record["journal_sha256"],
            "targets_sha256": record["targets_sha256"],
            "spent": format(record["spent"], "f"),
        },
    )
    return record


def run(
    args: argparse.Namespace,
    *,
    invoke: Callable[[Path, Mapping[str, str]], None] = dual._invoke_default,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    launcher = Path(args.phase_launcher).expanduser().resolve()
    root = Path(args.output_root).expanduser().resolve()
    source_plan = Path(args.source_plan).expanduser().resolve()
    source_journal = Path(args.source_journal).expanduser().resolve()
    source_chain = Path(args.source_chain_head).expanduser().resolve()
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    source = adapter.load_source_evidence(
        plan_path=source_plan,
        journal_path=source_journal,
        chain_head_path=source_chain,
    )
    prior_records, c001_remaining, prior = c002.load_completed_c001(
        report_path=args.prior_continuation_report,
        expected_report_sha256=args.expected_prior_continuation_report_sha256,
    )
    c001_spend = STATED_BALANCE - c001_remaining
    if (
        c001_spend != EXPECTED_PRE_PREFIX_SPEND
        or c001_spend + PREFIX_SPEND != EXPECTED_CUMULATIVE_PREFIX_SPEND
    ):
        raise ValueError("cumulative OpenRouter ledger before resume differs")
    available = STATED_BALANCE - EXPECTED_CUMULATIVE_PREFIX_SPEND
    if available < INITIAL_WORST_USD:
        raise ValueError("exact tail-47 initial reservation no longer fits")
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "resume.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "phase_launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
        "source_partial": {
            "plan_path": str(source.plan_path),
            "plan_sha256": adapter.SOURCE_PLAN_SHA256,
            "journal_path": str(source.journal_path),
            "journal_sha256": adapter.SOURCE_JOURNAL_SHA256,
            "chain_head_path": str(source.chain_head_path),
            "chain_head_sha256": adapter.SOURCE_CHAIN_HEAD_SHA256,
            "paid_prefix_tasks": adapter.PAID_PREFIX_TASKS,
            "paid_prefix_task_ids_sha256": canonical_sha256(
                list(adapter.PREFIX_TASK_IDS)
            ),
            "paid_prefix_spend_usd": format(PREFIX_SPEND, "f"),
            "source_files_modified": False,
        },
        "original_schedule_sha256": adapter.ORIGINAL_SCHEDULE_SHA256,
        "tail_schedule_sha256": adapter.TAIL_SCHEDULE_SHA256,
        "tail_calls": adapter.TAIL_TASKS,
        "input_cap": adapter.MAX_INPUT_TOKENS,
        "initial_output_cap": adapter.INITIAL_MAX_OUTPUT_TOKENS,
        "retry_output_cap": adapter.RETRY_MAX_OUTPUT_TOKENS,
        "retry_policy": "exact_all_or_none_non_code_or_length_only",
        "max_retry_tasks": MAX_RETRY_TASKS,
        "budget": {
            "stated_balance_usd": format(STATED_BALANCE, "f"),
            "cumulative_before_resume_usd": format(
                EXPECTED_CUMULATIVE_PREFIX_SPEND, "f"
            ),
            "available_for_resume_usd": format(available, "f"),
            "initial_worst_case_usd": format(INITIAL_WORST_USD, "f"),
            "retry_worst_case_usd_per_task": format(
                RETRY_WORST_USD_PER_TASK, "f"
            ),
        },
        "heldout_175_opened": False,
    }
    events = load_journal(journal_path)
    if not events:
        append_event(
            journal_path,
            {
                "event": "header",
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
    elif events[0].get("contract") != contract:
        raise ValueError("resume controller contract differs")
    prior_index = {
        "tsv": {
            "path": prior["full_prior_index_tsv_path"],
            "sha256": prior["full_prior_index_tsv_sha256"],
        }
    }
    env = dict(os.environ if base_env is None else base_env)
    initial = execute_phase(
        launcher=launcher,
        root=root,
        journal_path=journal_path,
        phase=cascade.PHASE_KIMI_INITIAL,
        output_dir=root / "kimi_initial_tail47",
        max_usd=INITIAL_WORST_USD,
        max_tasks=adapter.TAIL_TASKS,
        max_output_tokens=adapter.INITIAL_MAX_OUTPUT_TOKENS,
        base_env=env,
        invoke=invoke,
        source_plan=source_plan,
        source_journal=source_journal,
        source_chain_head=source_chain,
        prior_index=prior_index,
    )
    records = [initial]
    retry_ids = list(initial["retry_eligible_task_ids"])
    remaining = available - initial["spent"]
    retry_required = RETRY_WORST_USD_PER_TASK * len(retry_ids)
    retry_skipped = _retry_should_skip(
        retry_tasks=len(retry_ids), remaining=remaining
    )
    if retry_ids and not retry_skipped:
        records.append(
            execute_phase(
                launcher=launcher,
                root=root,
                journal_path=journal_path,
                phase=cascade.PHASE_KIMI_RETRY,
                output_dir=root / "kimi_retry_tail47",
                max_usd=retry_required,
                max_tasks=len(retry_ids),
                max_output_tokens=adapter.RETRY_MAX_OUTPUT_TOKENS,
                base_env=env,
                invoke=invoke,
                source_plan=source_plan,
                source_journal=source_journal,
                source_chain_head=source_chain,
                retry_source=initial,
            )
        )
    elif retry_skipped:
        _append_once(
            journal_path,
            {
                "event": "kimi_retry_budget_skip",
                "tasks": len(retry_ids),
                "task_ids_sha256": canonical_sha256(retry_ids),
                "required_worst_case_usd": format(retry_required, "f"),
                "remaining_usd": format(remaining, "f"),
                "partial_retry_executed": False,
            },
        )
    tail_spend = sum((row["spent"] for row in records), Decimal(0))
    cumulative = EXPECTED_CUMULATIVE_PREFIX_SPEND + tail_spend
    if cumulative > STATED_BALANCE:
        raise ValueError("cumulative OpenRouter spend exceeds the stated ledger")
    manifest = dual.publish_aggregate(root, records)
    full_index = dual.publish_prior_index(root, "after_resume47", prior_records + records)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": canonical_sha256(contract),
        "source_partial": contract["source_partial"],
        "tail_phases": [
            {
                key: row[key]
                for key in (
                    "phase",
                    "cohort_index",
                    "report_sha256",
                    "journal_sha256",
                    "targets_sha256",
                )
            }
            for row in records
        ],
        "retry": {
            "eligible_tasks": len(retry_ids),
            "eligible_task_ids_sha256": canonical_sha256(retry_ids),
            "complete_exact_set_executed": bool(retry_ids) and not retry_skipped,
            "not_needed": not retry_ids,
            "budget_skipped_entire_set": retry_skipped,
            "partial_retry_executed": False,
            "required_worst_case_usd": format(retry_required, "f"),
            "max_retry_tasks": MAX_RETRY_TASKS,
        },
        "budget": {
            "prefix_and_predecessor_charged_usd": format(
                EXPECTED_CUMULATIVE_PREFIX_SPEND, "f"
            ),
            "resume_tail_charged_usd": format(tail_spend, "f"),
            "cumulative_charged_usd": format(cumulative, "f"),
            "remaining_from_stated_balance_usd": format(
                STATED_BALANCE - cumulative, "f"
            ),
            "within_contract": True,
        },
        "new_tail_direct_manifest": manifest,
        "full_prior_report_index": full_index,
        "prefix_disposition": {
            "paid_results": adapter.PAID_PREFIX_TASKS,
            "private_verified_in_this_stage": False,
            "training_used_in_this_stage": False,
            "reason": "source_partial_remains_immutable; local_prefix_verification_is_a_separate_zero_API_stage",
        },
        "journal": journal_record(journal_path),
        "heldout_175_opened": False,
    }
    require_exact_or_write(root / "resume_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase-launcher", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prior-continuation-report", required=True)
    parser.add_argument("--expected-prior-continuation-report-sha256", required=True)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--source-journal", required=True)
    parser.add_argument("--source-chain-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": report["status"],
                "tail_verified": report["new_tail_direct_manifest"]["rows"],
                "cumulative_usd": report["budget"]["cumulative_charged_usd"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
