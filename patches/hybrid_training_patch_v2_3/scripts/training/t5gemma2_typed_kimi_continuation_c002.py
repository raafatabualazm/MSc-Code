#!/usr/bin/env python3
"""Budget-sealed Kimi cohort 2, chained only after completed cohort 1.

The controller consumes the completed cohort-1 continuation report as its
sole mutable predecessor.  It validates that report's complete prior index,
OpenRouter ledger, retry disposition, and >=8 verified-target continuation
gate before scheduling exactly 50 previously unscheduled TRAIN identities.
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
from scripts.training import t5gemma2_typed_kimi_continuation as c001


RUN_SCHEMA = "t5gemma2-typed-kimi-continuation-c002-run-v1"
REPORT_SCHEMA = "t5gemma2-typed-kimi-continuation-c002-report-v1"
COHORT_INDEX = 2
FIXED_KIMI_COHORT_LIMIT = 3
COHORT_TASKS = 50
MAX_INPUT_TOKENS = 16384
PREFERRED_INITIAL_MAX_OUTPUT_TOKENS = 4096
FALLBACK_INITIAL_MAX_OUTPUT_TOKENS = 2048
RETRY_MAX_OUTPUT_TOKENS = 8192
INPUT_USD_PER_MILLION = Decimal("3")
OUTPUT_USD_PER_MILLION = Decimal("15")
STATED_OPENROUTER_BALANCE = Decimal("12.44")


def _worst_case_usd(*, calls: int, output_tokens: int) -> Decimal:
    return (
        Decimal(calls)
        * (
            Decimal(MAX_INPUT_TOKENS) * INPUT_USD_PER_MILLION
            + Decimal(output_tokens) * OUTPUT_USD_PER_MILLION
        )
        / Decimal(1_000_000)
    )


PREFERRED_INITIAL_WORST_USD = _worst_case_usd(
    calls=COHORT_TASKS, output_tokens=PREFERRED_INITIAL_MAX_OUTPUT_TOKENS
)
FALLBACK_INITIAL_WORST_USD = _worst_case_usd(
    calls=COHORT_TASKS, output_tokens=FALLBACK_INITIAL_MAX_OUTPUT_TOKENS
)
RETRY_WORST_USD_PER_TASK = _worst_case_usd(
    calls=1, output_tokens=RETRY_MAX_OUTPUT_TOKENS
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _digest(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} requires an exact lowercase SHA-256")
    return text


def _pinned(path_value: str | Path, expected: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    expected = _digest(expected, label)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} differs from its pin")
    return path


def _decimal(value: object, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - Decimal raises several types
        raise ValueError(f"{label} is malformed") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} is malformed")
    return result


def load_completed_c001(
    *, report_path: str | Path, expected_report_sha256: str
) -> tuple[list[dict[str, Any]], Decimal, dict[str, Any]]:
    """Validate the complete c001 chain and return its exact residual credit."""

    path = _pinned(report_path, expected_report_sha256, "cohort-1 report")
    report = _read_object(path, "cohort-1 report")
    if (
        report.get("schema") != c001.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("cohort_index") != 1
        or report.get("heldout_175_opened_by_controller") is not False
        or report.get("heldout_175_model_visible") is not False
    ):
        raise ValueError("cohort-1 report is not a safe completed predecessor")
    privacy = report.get("privacy")
    if not isinstance(privacy, Mapping) or any(bool(value) for value in privacy.values()):
        raise ValueError("cohort-1 privacy attestation differs")
    manifest = report.get("new_direct_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("direct_only") is not True
        or manifest.get("reasoning_rows") != 0
        or manifest.get("repair_conditioned_rows") != 0
        or manifest.get("gold_replay_rows") != 0
    ):
        raise ValueError("cohort-1 training output is not direct verified code only")
    retry = report.get("retry")
    decision = report.get("cohort_decision")
    if (
        not isinstance(retry, Mapping)
        or retry.get("partial_retry_executed") is not False
        or retry.get("budget_skipped_entire_set") is not False
        or not (
            retry.get("complete_exact_set_executed") is True
            or retry.get("not_needed") is True
        )
        or not isinstance(decision, Mapping)
        or decision.get("continue_kimi") is not True
        or decision.get("retry_complete") is not True
        or decision.get("eligible_to_continue_after_bounded_run") is not True
        or decision.get("next_cohort_started") is not False
    ):
        raise ValueError("cohort-1 retry/yield gate does not authorize cohort 2")

    actual_journal = journal_record(path.parent / "continuation.journal.jsonl")
    embedded_journal = report.get("journal")
    if not isinstance(embedded_journal, Mapping) or any(
        embedded_journal.get(key) != actual_journal.get(key)
        for key in ("sha256", "chain_head_sha256", "event_count", "head_event_sha256")
    ):
        raise ValueError("cohort-1 continuation journal binding differs")

    index = report.get("full_prior_report_index")
    if (
        not isinstance(index, Mapping)
        or index.get("schema") != dual.INDEX_SCHEMA
        or index.get("status") != "complete"
        or not isinstance(index.get("entries"), list)
        or not isinstance(index.get("tsv"), Mapping)
    ):
        raise ValueError("cohort-1 full prior index is malformed")
    entries = list(index["entries"])
    if index.get("entries_sha256") != canonical_sha256(entries):
        raise ValueError("cohort-1 full prior index digest differs")
    tsv = index["tsv"]
    tsv_path = _pinned(tsv.get("path", ""), tsv.get("sha256", ""), "cohort-1 prior TSV")
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(entries):
        raise ValueError("cohort-1 prior TSV length differs")

    expected_prefix = [
        (cascade.PHASE_KIMI_INITIAL, 0),
        (cascade.PHASE_KIMI_RETRY, 0),
        (cascade.PHASE_SONNET_RESIDUAL, 0),
        (cascade.PHASE_KIMI_INITIAL, 1),
    ]
    observed = [(row.get("phase"), row.get("cohort_index")) for row in entries]
    if observed not in (
        expected_prefix,
        expected_prefix + [(cascade.PHASE_KIMI_RETRY, 1)],
    ):
        raise ValueError("cohort-1 full prior index is not the exact phase prefix")

    records: list[dict[str, Any]] = []
    for position, (entry, line) in enumerate(zip(entries, lines, strict=True)):
        if entry.get("position") != position:
            raise ValueError("cohort-1 prior index positions differ")
        digest = _digest(entry.get("report_sha256"), "indexed phase report")
        report_path = _pinned(entry.get("report_path", ""), digest, "indexed phase report")
        if line != f"{digest}\t{report_path}":
            raise ValueError("cohort-1 prior TSV entry differs")
        record = c001.inspect_phase_report(
            report_path,
            expected_phase=str(entry.get("phase") or ""),
            expected_cohort=int(entry.get("cohort_index", -1)),
        )
        if (
            record["report_sha256"] != digest
            or record["journal_sha256"] != entry.get("journal_sha256")
            or record["targets_sha256"] != entry.get("targets_sha256")
        ):
            raise ValueError("cohort-1 indexed phase evidence differs")
        records.append(record)

    report_phases = report.get("phases")
    current_records = [row for row in records if row["cohort_index"] == 1]
    if not isinstance(report_phases, list) or [
        (row.get("phase"), row.get("cohort_index"), row.get("report_sha256"))
        for row in report_phases
    ] != [
        (row["phase"], row["cohort_index"], row["report_sha256"])
        for row in current_records
    ]:
        raise ValueError("cohort-1 report/index phase binding differs")

    openrouter_records = [
        row
        for row in records
        if row["phase"] in (cascade.PHASE_KIMI_INITIAL, cascade.PHASE_KIMI_RETRY)
    ]
    cumulative_spend = sum((row["spent"] for row in openrouter_records), Decimal(0))
    prior_spend = sum(
        (row["spent"] for row in openrouter_records if row["cohort_index"] == 0),
        Decimal(0),
    )
    c001_spend = sum(
        (row["spent"] for row in openrouter_records if row["cohort_index"] == 1),
        Decimal(0),
    )
    budget = report.get("budget")
    if (
        not isinstance(budget, Mapping)
        or _decimal(budget.get("prior_openrouter_spend_usd"), "prior spend") != prior_spend
        or _decimal(budget.get("continuation_charged_usd"), "cohort-1 spend") != c001_spend
        or _decimal(budget.get("available_before_continuation_usd"), "cohort-1 available")
        != STATED_OPENROUTER_BALANCE - prior_spend
    ):
        raise ValueError("cohort-1 cumulative OpenRouter ledger differs")
    remaining = STATED_OPENROUTER_BALANCE - cumulative_spend
    if (
        remaining < 0
        or _decimal(budget.get("remaining_from_stated_balance_usd"), "remaining credit")
        != remaining
    ):
        raise ValueError("cohort-1 residual OpenRouter credit differs")

    scheduled = {task for row in records for task in row["scheduled_task_ids"]}
    verified = {task for row in records for task in row["verified_task_ids"]}
    return records, remaining, {
        "continuation_report_sha256": sha256_file(path),
        "full_prior_index_tsv_path": str(tsv_path),
        "full_prior_index_tsv_sha256": sha256_file(tsv_path),
        "prior_phase_report_sha256s": [row["report_sha256"] for row in records],
        "cumulative_openrouter_spend_usd": format(cumulative_spend, "f"),
        "remaining_openrouter_usd": format(remaining, "f"),
        "all_provider_scheduled_tasks": len(scheduled),
        "all_provider_scheduled_task_ids_sha256": canonical_sha256(sorted(scheduled)),
        "all_provider_verified_tasks": len(verified),
        "all_provider_verified_task_ids_sha256": canonical_sha256(sorted(verified)),
    }


def _append_once(journal_path: Path, event: Mapping[str, Any], identity: str) -> None:
    existing = [row for row in load_journal(journal_path) if row.get("event") == identity]
    if not existing:
        append_event(journal_path, dict(event))
    elif len(existing) != 1 or any(existing[0].get(k) != v for k, v in event.items()):
        raise ValueError(f"cohort-2 journal {identity} evidence differs")


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
    env.update(
        {
            "T5GEMMA_TYPED_API_COHORT_INDEX": str(COHORT_INDEX),
            "T5GEMMA_TYPED_API_FIXED_KIMI_COHORT_LIMIT": str(FIXED_KIMI_COHORT_LIMIT),
            "T5GEMMA_TYPED_API_MAX_INPUT_TOKENS_PER_CALL": str(MAX_INPUT_TOKENS),
            "T5GEMMA_TYPED_API_INITIAL_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        }
    )
    plan_path = root / f"plan_{phase}_c{COHORT_INDEX:03d}.json"
    plan_env = dict(env)
    plan_env.pop("OPENROUTER_API_KEY", None)
    plan_env["T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT"] = str(plan_path.resolve())
    invoke(launcher, plan_env)
    plan = _read_object(plan_path, "cohort-2 phase plan")
    schedule_sha = str(plan.get("selection", {}).get("task_ids_sha256") or "")
    if (
        plan.get("schema") != cascade.PLAN_SCHEMA
        or plan.get("status") != "complete"
        or plan.get("phase") != phase
        or plan.get("cohort_index") != COHORT_INDEX
        or plan.get("fixed_kimi_cohort_limit") != FIXED_KIMI_COHORT_LIMIT
        or plan.get("provider_credentials_read") is not False
        or plan.get("frontier_api_calls") is not False
        or plan.get("selection", {}).get("scheduled_tasks") != max_tasks
        or plan.get("budget", {}).get("max_output_tokens_per_call") != max_output_tokens
        or len(schedule_sha) != 64
    ):
        raise ValueError("credential-free cohort-2 plan differs")

    events = load_journal(journal_path)
    completed = [
        row
        for row in events
        if row.get("event") == "phase_complete"
        and row.get("phase") == phase
        and row.get("cohort_index") == COHORT_INDEX
    ]
    if len(completed) > 1:
        raise ValueError("cohort-2 journal contains duplicate phase completion")
    if not completed:
        started = [
            row
            for row in events
            if row.get("event") == "phase_start"
            and row.get("phase") == phase
            and row.get("cohort_index") == COHORT_INDEX
        ]
        start = {
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
            append_event(journal_path, start)
        elif len(started) != 1 or any(started[0].get(k) != v for k, v in start.items()):
            raise ValueError("cohort-2 phase-start evidence differs")
        run_env = dict(env)
        run_env["T5GEMMA_TYPED_API_SCHEDULE_SHA256"] = schedule_sha
        invoke(launcher, run_env)

    report_path = output_dir / "typed_api_rescue_report.json"
    record = c001.inspect_phase_report(
        report_path, expected_phase=phase, expected_cohort=COHORT_INDEX
    )
    if record["spent"] > max_usd:
        raise ValueError("cohort-2 phase exceeds its sealed reservation")
    complete = {
        "event": "phase_complete",
        "phase": phase,
        "cohort_index": COHORT_INDEX,
        "report_sha256": record["report_sha256"],
        "journal_sha256": record["journal_sha256"],
        "targets_sha256": record["targets_sha256"],
        "spent": format(record["spent"], "f"),
    }
    if not completed:
        append_event(journal_path, complete)
    elif any(completed[0].get(k) != v for k, v in complete.items()):
        raise ValueError("completed cohort-2 phase evidence changed")
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
    if args.stated_openrouter_balance != STATED_OPENROUTER_BALANCE:
        raise ValueError("cohort-2 stated OpenRouter balance is fixed at $12.44")
    prior_records, available, prior = load_completed_c001(
        report_path=args.prior_continuation_report,
        expected_report_sha256=args.expected_prior_continuation_report_sha256,
    )
    if available >= PREFERRED_INITIAL_WORST_USD:
        initial_max_output_tokens = PREFERRED_INITIAL_MAX_OUTPUT_TOKENS
        initial_worst_usd = PREFERRED_INITIAL_WORST_USD
    elif available >= FALLBACK_INITIAL_WORST_USD:
        initial_max_output_tokens = FALLBACK_INITIAL_MAX_OUTPUT_TOKENS
        initial_worst_usd = FALLBACK_INITIAL_WORST_USD
    else:
        raise ValueError("strict 50-call cohort-2 reservation does not fit residual credit")

    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "continuation.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "phase_launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
        "prior": prior,
        "cohort_index": COHORT_INDEX,
        "fixed_kimi_cohort_limit": FIXED_KIMI_COHORT_LIMIT,
        "cohort_tasks": COHORT_TASKS,
        "input_token_cap_per_call": MAX_INPUT_TOKENS,
        "initial_output_token_cap": initial_max_output_tokens,
        "initial_output_token_cap_policy": "prefer_4096_when_full_reservation_fits_else_2048",
        "retry_output_token_cap": RETRY_MAX_OUTPUT_TOKENS,
        "retry_policy": "exact_non_code_or_length_only_or_budget_skip_entire_set",
        "stop_after_this_cohort": True,
        "credit": {
            "stated_balance_usd": format(STATED_OPENROUTER_BALANCE, "f"),
            "available_before_cohort2_usd": format(available, "f"),
            "initial_worst_case_usd": format(initial_worst_usd, "f"),
            "retry_worst_case_usd_per_task": format(RETRY_WORST_USD_PER_TASK, "f"),
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
            {"event": "header", "contract": contract, "contract_sha256": canonical_sha256(contract)},
        )
    elif events[0].get("contract") != contract:
        raise ValueError("cohort-2 run contract differs")

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
        output_dir=root / "kimi_initial_c002",
        max_usd=initial_worst_usd,
        max_tasks=COHORT_TASKS,
        max_output_tokens=initial_max_output_tokens,
        base_env=env,
        invoke=invoke,
        prior_index=prior_index,
    )
    records = [initial]
    retry_ids = list(initial["retry_eligible_task_ids"])
    remaining = available - initial["spent"]
    if remaining < 0:
        raise ValueError("cohort-2 initial phase exceeded residual credit")
    retry_required = RETRY_WORST_USD_PER_TASK * len(retry_ids)
    retry_skipped = False
    if retry_ids and retry_required <= remaining:
        records.append(
            execute_phase(
                launcher=launcher,
                root=root,
                journal_path=journal_path,
                phase=cascade.PHASE_KIMI_RETRY,
                output_dir=root / "kimi_retry_c002",
                max_usd=remaining,
                max_tasks=len(retry_ids),
                max_output_tokens=RETRY_MAX_OUTPUT_TOKENS,
                base_env=env,
                invoke=invoke,
                retry_source=initial,
            )
        )
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
                "remaining_credit_usd": format(remaining, "f"),
                "reason": "exact_retry_set_does_not_fit_remaining_credit",
            },
            "kimi_retry_budget_skip",
        )

    spent = sum((row["spent"] for row in records), Decimal(0))
    if spent > available:
        raise ValueError("cohort-2 cumulative spend exceeds residual credit")
    prior_scheduled = {task for row in prior_records for task in row["scheduled_task_ids"]}
    current_scheduled = set(initial["scheduled_task_ids"])
    if len(current_scheduled) != COHORT_TASKS or prior_scheduled & current_scheduled:
        raise ValueError("cohort-2 did not schedule 50 fresh identities")

    retry_verified = records[1]["verified_task_ids"] if len(records) == 2 else []
    decision = cascade.cohort_decision(
        initial_verified_ids=initial["verified_task_ids"],
        retry_verified_ids=retry_verified,
    )
    decision.update(
        {
            "next_cohort_started": False,
            "retry_complete": not retry_skipped,
            "eligible_to_continue_after_bounded_run": decision["continue_kimi"] and not retry_skipped,
        }
    )
    manifest = dual.publish_aggregate(root, records)
    full_index = dual.publish_prior_index(root, "after_cohort2", prior_records + records)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": canonical_sha256(contract),
        "cohort_index": COHORT_INDEX,
        "phases": [
            {key: row[key] for key in ("phase", "cohort_index", "report_sha256", "journal_sha256", "targets_sha256")}
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
        },
        "cohort_decision": decision,
        "budget": {
            "available_before_cohort2_usd": format(available, "f"),
            "cohort2_charged_usd": format(spent, "f"),
            "remaining_from_stated_balance_usd": format(available - spent, "f"),
            "within_contract": True,
        },
        "new_direct_manifest": manifest,
        "full_prior_report_index": full_index,
        "journal": journal_record(journal_path),
        "privacy": contract["privacy"],
        "heldout_175_opened_by_controller": False,
        "heldout_175_model_visible": False,
    }
    require_exact_or_write(root / "continuation_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-launcher", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prior-continuation-report", required=True)
    parser.add_argument("--expected-prior-continuation-report-sha256", required=True)
    parser.add_argument(
        "--stated-openrouter-balance", type=Decimal, default=STATED_OPENROUTER_BALANCE
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": report["status"],
                "new_verified_targets": report["new_direct_manifest"]["rows"],
                "charged_usd": report["budget"]["cohort2_charged_usd"],
                "remaining_usd": report["budget"]["remaining_from_stated_balance_usd"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
