#!/usr/bin/env python3
"""One bounded, budget-sealed Kimi continuation for typed direct RS-SFT.

This controller resumes after the completed cohort-0 Kimi/retry/Sonnet run,
runs exactly one new 50-task Kimi cohort, and then either runs the *entire*
parse-failure/length-only retry set or skips that retry set when its complete
worst-case reservation cannot fit the remaining OpenRouter cap.  It never
starts cohort two automatically.

Provider credentials remain owned by the phase launcher.  The controller
first creates a credential-free plan and then binds execution to that exact
task digest.  Published training data contains verified Dart code only.
"""

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


RUN_SCHEMA = "t5gemma2-typed-kimi-continuation-run-v1"
REPORT_SCHEMA = "t5gemma2-typed-kimi-continuation-report-v1"
COHORT_INDEX = 1
FIXED_KIMI_COHORT_LIMIT = 2
COHORT_TASKS = 50
MAX_INPUT_TOKENS = 16384
INITIAL_MAX_OUTPUT_TOKENS = 2048
RETRY_MAX_OUTPUT_TOKENS = 8192
INPUT_USD_PER_MILLION = Decimal("3")
OUTPUT_USD_PER_MILLION = Decimal("15")
OPENROUTER_BALANCE_BEFORE_COHORT0 = Decimal("12.44")
CONTINUATION_CAP = Decimal("10.30")
EXPECTED_PRIOR_OPENROUTER_SPEND = Decimal("2.103075000")


def _worst_case_usd(*, calls: int, output_tokens: int) -> Decimal:
    return (
        Decimal(calls)
        * (
            Decimal(MAX_INPUT_TOKENS) * INPUT_USD_PER_MILLION
            + Decimal(output_tokens) * OUTPUT_USD_PER_MILLION
        )
        / Decimal(1_000_000)
    )


INITIAL_WORST_USD = _worst_case_usd(
    calls=COHORT_TASKS, output_tokens=INITIAL_MAX_OUTPUT_TOKENS
)
RETRY_WORST_USD_PER_TASK = _worst_case_usd(
    calls=1, output_tokens=RETRY_MAX_OUTPUT_TOKENS
)


def _require_digest(value: str, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} requires an exact lowercase SHA-256")
    return text


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _pinned(path_value: str | Path, expected: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    expected = _require_digest(expected, label)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} differs from its pin")
    return path


def load_prior_run(
    *,
    orchestration_report_path: str | Path,
    expected_orchestration_report_sha256: str,
    prior_index_path: str | Path,
    expected_prior_index_sha256: str,
) -> tuple[list[dict[str, Any]], Decimal, dict[str, Any]]:
    """Validate cohort zero, its exact retry, Sonnet, and their spend ledger."""

    report_path = _pinned(
        orchestration_report_path,
        expected_orchestration_report_sha256,
        "prior orchestration report",
    )
    index_path = _pinned(
        prior_index_path, expected_prior_index_sha256, "prior report index"
    )
    report = _read_object(report_path, "prior orchestration report")
    index_sha = sha256_file(index_path)
    if (
        report.get("schema") != dual.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("direct_manifest", {}).get("direct_only") is not True
        or report.get("direct_manifest", {}).get("reasoning_rows") != 0
        or report.get("direct_manifest", {}).get("repair_conditioned_rows") != 0
        or report.get("direct_manifest", {}).get("gold_replay_rows") != 0
        or report.get("heldout_175_model_visible") is not False
        or report.get("heldout_175_used_for_generation_or_selection") is not False
        or report.get("prior_report_index", {}).get("tsv", {}).get("sha256")
        != index_sha
    ):
        raise ValueError("prior dual-provider run is not a safe completed direct run")

    entries: list[tuple[str, Path]] = []
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("prior report index cannot be read") from exc
    for line_number, line in enumerate(lines, 1):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"prior report index:{line_number}: malformed row")
        digest, path_text = fields
        path = _pinned(path_text, digest, f"prior phase report {line_number}")
        entries.append((digest, path))
    expected_order = [
        (cascade.PHASE_KIMI_INITIAL, 0),
        (cascade.PHASE_KIMI_RETRY, 0),
        (cascade.PHASE_SONNET_RESIDUAL, 0),
    ]
    if len(entries) != len(expected_order):
        raise ValueError("prior phase index is not the exact completed three-phase run")

    records: list[dict[str, Any]] = []
    for (digest, phase_path), (phase, cohort) in zip(
        entries, expected_order, strict=True
    ):
        record = inspect_phase_report(
            phase_path, expected_phase=phase, expected_cohort=cohort
        )
        if record["report_sha256"] != digest:
            raise ValueError("prior phase report changed during inspection")
        records.append(record)

    initial, retry, sonnet = records
    if set(retry["scheduled_task_ids"]) != set(
        initial["retry_eligible_task_ids"]
    ) or len(retry["scheduled_task_ids"]) != len(
        initial["retry_eligible_task_ids"]
    ):
        raise ValueError("cohort-zero retry is not the exact parse/length set")
    verified_sets = [set(row["verified_task_ids"]) for row in records]
    if any(
        verified_sets[left] & verified_sets[right]
        for left in range(len(verified_sets))
        for right in range(left + 1, len(verified_sets))
    ):
        raise ValueError("prior providers published duplicate verified identities")
    kimi_verified = initial["verified_task_ids"] + retry["verified_task_ids"]
    if len(kimi_verified) < cascade.KIMI_CONTINUE_MIN_YIELD:
        raise ValueError("cohort-zero Kimi yield does not authorize continuation")

    prior_spend = initial["spent"] + retry["spent"]
    report_spend = Decimal(
        str(report.get("providers", {}).get("openrouter", {}).get("charged_usd"))
    )
    if (
        not report_spend.is_finite()
        or report_spend != prior_spend
        or report_spend != EXPECTED_PRIOR_OPENROUTER_SPEND
    ):
        raise ValueError("prior OpenRouter spend ledger differs")
    report_phases = report.get("phases")
    if not isinstance(report_phases, list) or [
        (row.get("phase"), row.get("cohort_index"), row.get("report_sha256"))
        for row in report_phases
    ] != [
        (record["phase"], record["cohort_index"], record["report_sha256"])
        for record in records
    ]:
        raise ValueError("prior orchestration/phase index binding differs")

    all_scheduled = {
        task_id for record in records for task_id in record["scheduled_task_ids"]
    }
    all_verified = {
        task_id for record in records for task_id in record["verified_task_ids"]
    }
    return records, prior_spend, {
        "orchestration_report_sha256": sha256_file(report_path),
        "prior_index_sha256": index_sha,
        "prior_phase_report_sha256s": [row[0] for row in entries],
        "cohort0_kimi_verified": len(kimi_verified),
        "all_provider_scheduled_tasks": len(all_scheduled),
        "all_provider_scheduled_task_ids_sha256": canonical_sha256(
            sorted(all_scheduled)
        ),
        "all_provider_verified_tasks": len(all_verified),
        "all_provider_verified_task_ids_sha256": canonical_sha256(
            sorted(all_verified)
        ),
    }


def _append_once(
    journal_path: Path, event: Mapping[str, Any], *, identity: str
) -> None:
    existing = [row for row in load_journal(journal_path) if row.get("event") == identity]
    if not existing:
        append_event(journal_path, dict(event))
    elif len(existing) != 1 or any(
        existing[0].get(key) != value for key, value in event.items()
    ):
        raise ValueError(f"continuation journal {identity} evidence differs")


def inspect_phase_report(
    report_path: Path, *, expected_phase: str, expected_cohort: int
) -> dict[str, Any]:
    """Add schedule-identity validation to the shared direct-only audit."""

    record = dual.inspect_phase_report(
        report_path,
        expected_phase=expected_phase,
        expected_cohort=expected_cohort,
    )
    events = load_journal(report_path.parent / "typed_api_rescue.journal.jsonl")
    scheduled_ids = [
        str(row.get("task_id") or "")
        for row in events
        if row.get("event") == "call_intent"
    ]
    report = _read_object(report_path, "phase report")
    schedule = report.get("schedule", {})
    if (
        any(not task_id for task_id in scheduled_ids)
        or len(scheduled_ids) != len(set(scheduled_ids))
        or schedule.get("scheduled_tasks") != len(scheduled_ids)
        or schedule.get("task_ids_sha256") != canonical_sha256(scheduled_ids)
    ):
        raise ValueError("phase scheduled-task accounting differs")
    return {**record, "scheduled_task_ids": scheduled_ids}


def execute_phase(
    *,
    launcher: Path,
    root: Path,
    journal_path: Path,
    phase: str,
    output_dir: Path,
    max_usd: Decimal,
    max_tasks: int,
    base_env: Mapping[str, str],
    invoke: Callable[[Path, Mapping[str, str]], None],
    prior_index: Mapping[str, Any] | None = None,
    retry_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Credential-free plan then exact cohort-1 execution."""

    env = dual._phase_env(  # noqa: SLF001
        base_env,
        phase=phase,
        output_dir=output_dir,
        max_usd=max_usd,
        max_tasks=max_tasks,
        prior_index=prior_index,
        retry_source=retry_source,
    )
    env.update(
        {
            "T5GEMMA_TYPED_API_COHORT_INDEX": str(COHORT_INDEX),
            "T5GEMMA_TYPED_API_FIXED_KIMI_COHORT_LIMIT": str(
                FIXED_KIMI_COHORT_LIMIT
            ),
            "T5GEMMA_TYPED_API_MAX_INPUT_TOKENS_PER_CALL": str(
                MAX_INPUT_TOKENS
            ),
        }
    )
    plan_path = root / f"plan_{phase}_c{COHORT_INDEX:03d}.json"
    plan_env = dict(env)
    plan_env["T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT"] = str(plan_path.resolve())
    invoke(launcher, plan_env)
    plan = _read_object(plan_path, "continuation phase plan")
    schedule_sha = str(plan.get("selection", {}).get("task_ids_sha256") or "")
    planned_tasks = plan.get("selection", {}).get("scheduled_tasks")
    if (
        plan.get("schema") != cascade.PLAN_SCHEMA
        or plan.get("status") != "complete"
        or plan.get("phase") != phase
        or plan.get("cohort_index") != COHORT_INDEX
        or plan.get("fixed_kimi_cohort_limit") != FIXED_KIMI_COHORT_LIMIT
        or plan.get("provider_credentials_read") is not False
        or plan.get("frontier_api_calls") is not False
        or not isinstance(planned_tasks, int)
        or planned_tasks != max_tasks
        or len(schedule_sha) != 64
    ):
        raise ValueError("credential-free continuation plan differs")

    events = load_journal(journal_path)
    completed = [
        row
        for row in events
        if row.get("event") == "phase_complete"
        and row.get("phase") == phase
        and row.get("cohort_index") == COHORT_INDEX
    ]
    if len(completed) > 1:
        raise ValueError("continuation journal contains duplicate phase completion")
    if not completed:
        started = [
            row
            for row in events
            if row.get("event") == "phase_start"
            and row.get("phase") == phase
            and row.get("cohort_index") == COHORT_INDEX
        ]
        start_event = {
            "event": "phase_start",
            "phase": phase,
            "cohort_index": COHORT_INDEX,
            "plan_sha256": sha256_file(plan_path),
            "schedule_sha256": schedule_sha,
            "max_usd": format(max_usd, "f"),
            "max_tasks": max_tasks,
            "max_input_tokens_per_call": MAX_INPUT_TOKENS,
        }
        if not started:
            append_event(journal_path, start_event)
        elif len(started) != 1 or any(
            started[0].get(key) != value for key, value in start_event.items()
        ):
            raise ValueError("continuation phase-start evidence differs")
        run_env = dict(env)
        run_env["T5GEMMA_TYPED_API_SCHEDULE_SHA256"] = schedule_sha
        invoke(launcher, run_env)

    report_path = output_dir / "typed_api_rescue_report.json"
    record = inspect_phase_report(
        report_path, expected_phase=phase, expected_cohort=COHORT_INDEX
    )
    if record["spent"] > max_usd:
        raise ValueError("continuation phase exceeds its sealed reservation")
    complete_event = {
        "event": "phase_complete",
        "phase": phase,
        "cohort_index": COHORT_INDEX,
        "report_sha256": record["report_sha256"],
        "journal_sha256": record["journal_sha256"],
        "targets_sha256": record["targets_sha256"],
        "spent": format(record["spent"], "f"),
    }
    if not completed:
        append_event(journal_path, complete_event)
    elif any(completed[0].get(key) != value for key, value in complete_event.items()):
        raise ValueError("completed continuation phase evidence changed")
    return record


def run(
    args: argparse.Namespace,
    *,
    invoke: Callable[[Path, Mapping[str, str]], None] = dual._invoke_default,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    launcher = Path(args.phase_launcher).expanduser().resolve()
    root = Path(args.output_root).expanduser().resolve()
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    if (
        args.openrouter_balance_before_cohort0
        != OPENROUTER_BALANCE_BEFORE_COHORT0
        or args.continuation_max_usd != CONTINUATION_CAP
    ):
        raise ValueError("continuation credit ledger is fixed at $12.44 / $10.30")
    prior_records, prior_spend, prior_record = load_prior_run(
        orchestration_report_path=args.prior_orchestration_report,
        expected_orchestration_report_sha256=(
            args.expected_prior_orchestration_report_sha256
        ),
        prior_index_path=args.prior_index,
        expected_prior_index_sha256=args.expected_prior_index_sha256,
    )
    available = args.openrouter_balance_before_cohort0 - prior_spend
    if CONTINUATION_CAP > available or INITIAL_WORST_USD > CONTINUATION_CAP:
        raise ValueError("sealed continuation reservation exceeds available credit")

    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "continuation.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "phase_launcher": {
            "path": str(launcher),
            "sha256": sha256_file(launcher),
        },
        "prior": prior_record,
        "cohort_index": COHORT_INDEX,
        "fixed_kimi_cohort_limit": FIXED_KIMI_COHORT_LIMIT,
        "cohort_tasks": COHORT_TASKS,
        "input_token_cap_per_call": MAX_INPUT_TOKENS,
        "measured_prior_prompt_max_tokens": 7833,
        "initial_output_token_cap": INITIAL_MAX_OUTPUT_TOKENS,
        "retry_output_token_cap": RETRY_MAX_OUTPUT_TOKENS,
        "retry_policy": "exact_non_code_or_length_only_or_budget_skip_entire_set",
        "stop_after_this_cohort": True,
        "credit": {
            "balance_before_cohort0_usd": format(
                args.openrouter_balance_before_cohort0, "f"
            ),
            "audited_prior_spend_usd": format(prior_spend, "f"),
            "available_before_continuation_usd": format(available, "f"),
            "continuation_cap_usd": format(CONTINUATION_CAP, "f"),
            "unreserved_buffer_usd": format(available - CONTINUATION_CAP, "f"),
            "initial_worst_case_usd": format(INITIAL_WORST_USD, "f"),
            "retry_worst_case_usd_per_task": format(
                RETRY_WORST_USD_PER_TASK, "f"
            ),
        },
        "training_output": "new_direct_visible_and_private_verified_code_only",
        "privacy": {
            "gold_sent_to_provider": False,
            "private_acceptance_sent_to_provider": False,
            "private_holdback_sent_to_provider": False,
            "heldout_175_sent_to_provider": False,
            "reasoning_published": False,
            "diagnostics_published": False,
        },
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
        raise ValueError("continuation run contract differs")

    prior_index_record = {
        "tsv": {
            "path": str(Path(args.prior_index).expanduser().resolve()),
            "sha256": args.expected_prior_index_sha256,
        }
    }
    env = dict(os.environ if base_env is None else base_env)
    initial = execute_phase(
        launcher=launcher,
        root=root,
        journal_path=journal_path,
        phase=cascade.PHASE_KIMI_INITIAL,
        output_dir=root / "kimi_initial_c001",
        max_usd=INITIAL_WORST_USD,
        max_tasks=COHORT_TASKS,
        base_env=env,
        invoke=invoke,
        prior_index=prior_index_record,
    )
    continuation_records = [initial]
    retry_ids = list(initial["retry_eligible_task_ids"])
    remaining = CONTINUATION_CAP - initial["spent"]
    if remaining < 0:
        raise ValueError("cohort-one spend exceeds continuation cap")
    retry_required = RETRY_WORST_USD_PER_TASK * len(retry_ids)
    retry_skipped = False
    if retry_ids and retry_required <= remaining:
        retry = execute_phase(
            launcher=launcher,
            root=root,
            journal_path=journal_path,
            phase=cascade.PHASE_KIMI_RETRY,
            output_dir=root / "kimi_retry_c001",
            max_usd=remaining,
            max_tasks=len(retry_ids),
            base_env=env,
            invoke=invoke,
            retry_source=initial,
        )
        continuation_records.append(retry)
    elif retry_ids:
        retry_skipped = True
        _append_once(
            journal_path,
            {
                "event": "kimi_retry_budget_skip",
                "cohort_index": COHORT_INDEX,
                "tasks": len(retry_ids),
                "task_ids_sha256": canonical_sha256(retry_ids),
                "required_worst_case_usd": format(retry_required, "f"),
                "remaining_continuation_usd": format(remaining, "f"),
                "reason": "exact_retry_set_does_not_fit_remaining_cap",
            },
            identity="kimi_retry_budget_skip",
        )

    spent = sum((row["spent"] for row in continuation_records), Decimal(0))
    if spent > CONTINUATION_CAP or prior_spend + spent > args.openrouter_balance_before_cohort0:
        raise ValueError("cumulative OpenRouter spend exceeds its sealed ledger")
    prior_scheduled = {
        task_id
        for record in prior_records
        for task_id in record["scheduled_task_ids"]
    }
    current_initial_scheduled = set(initial["scheduled_task_ids"])
    if prior_scheduled & current_initial_scheduled:
        raise ValueError("cohort one rescheduled a prior provider identity")

    retry_verified = (
        continuation_records[1]["verified_task_ids"]
        if len(continuation_records) == 2
        else []
    )
    decision = cascade.cohort_decision(
        initial_verified_ids=initial["verified_task_ids"],
        retry_verified_ids=retry_verified,
    )
    # The >=8 decision is reported for scientific accounting only.  This
    # bounded controller deliberately never starts another cohort.
    decision["next_cohort_started"] = False
    decision["retry_complete"] = not retry_skipped
    decision["eligible_to_continue_after_bounded_run"] = (
        decision["continue_kimi"] and not retry_skipped
    )

    new_manifest = dual.publish_aggregate(root, continuation_records)
    full_index = dual.publish_prior_index(
        root, "after_cohort1", prior_records + continuation_records
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": canonical_sha256(contract),
        "cohort_index": COHORT_INDEX,
        "phases": [
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
            for row in continuation_records
        ],
        "retry": {
            "eligible_tasks": len(retry_ids),
            "eligible_task_ids_sha256": canonical_sha256(retry_ids),
            "complete_exact_set_executed": bool(retry_ids) and not retry_skipped,
            "not_needed": not retry_ids,
            "budget_skipped_entire_set": retry_skipped,
            "partial_retry_executed": False,
            "required_worst_case_usd": format(retry_required, "f"),
        },
        "cohort_decision": decision,
        "budget": {
            "prior_openrouter_spend_usd": format(prior_spend, "f"),
            "available_before_continuation_usd": format(available, "f"),
            "continuation_cap_usd": format(CONTINUATION_CAP, "f"),
            "continuation_charged_usd": format(spent, "f"),
            "remaining_from_continuation_cap_usd": format(
                CONTINUATION_CAP - spent, "f"
            ),
            "remaining_from_stated_balance_usd": format(
                args.openrouter_balance_before_cohort0 - prior_spend - spent,
                "f",
            ),
            "within_contract": True,
        },
        "new_direct_manifest": new_manifest,
        "full_prior_report_index": full_index,
        "journal": journal_record(journal_path),
        "privacy": contract["privacy"],
        "heldout_175_opened_by_controller": False,
        "heldout_175_model_visible": False,
    }
    require_exact_or_write(root / "continuation_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase-launcher", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prior-orchestration-report", required=True)
    parser.add_argument(
        "--expected-prior-orchestration-report-sha256", required=True
    )
    parser.add_argument("--prior-index", required=True)
    parser.add_argument("--expected-prior-index-sha256", required=True)
    parser.add_argument(
        "--openrouter-balance-before-cohort0",
        type=Decimal,
        default=OPENROUTER_BALANCE_BEFORE_COHORT0,
    )
    parser.add_argument(
        "--continuation-max-usd", type=Decimal, default=CONTINUATION_CAP
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": report["status"],
                "new_verified_targets": report["new_direct_manifest"]["rows"],
                "charged_usd": report["budget"]["continuation_charged_usd"],
                "eligible_to_continue": report["cohort_decision"][
                    "eligible_to_continue_after_bounded_run"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
