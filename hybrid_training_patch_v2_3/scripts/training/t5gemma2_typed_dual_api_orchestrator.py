#!/usr/bin/env python3
"""Durable Kimi-then-Sonnet controller for typed direct RS-SFT harvests.

The controller owns the *cross-phase* invariants that a single provider run
cannot enforce: one Kimi cohort, only parse/length retries, cumulative provider
budgets, a hash-pinned prior-report index, and a direct-only aggregate.  Every
phase is planned without loading a credential, then executed against the exact
planned schedule digest.  Completed phase journals are the resume boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from decimal import Decimal, ROUND_FLOOR
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


RUN_SCHEMA = "t5gemma2-typed-dual-api-orchestration-run-v1"
INDEX_SCHEMA = "t5gemma2-typed-api-rescue-cascade-prior-index-v1"
REPORT_SCHEMA = "t5gemma2-typed-dual-api-orchestration-report-v1"
AGGREGATE_SCHEMA = "t5gemma2-typed-dual-api-direct-manifest-v1"
OPENROUTER_CAP = Decimal("12.0")
ANTHROPIC_CAP = Decimal("11.5")
KIMI_RETRY_WORST = (Decimal(65536) * 3 + Decimal(8192) * 15) / Decimal(1_000_000)
SONNET_WORST = (Decimal(65536) * 2 + Decimal(16384) * 10) / Decimal(1_000_000)
FORBIDDEN_OUTPUTS = (
    "repair_policy_targets.jsonl",
    "repair_policy_sources.jsonl",
    "repair_policy_manifest.json",
    "direct_hard_targets_f2.jsonl",
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{label} is absent") from exc
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ValueError(f"{label}:{line_number}: blank row")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _exact_text(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"sealed artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def inspect_phase_report(
    report_path: Path,
    *,
    expected_phase: str,
    expected_cohort: int,
) -> dict[str, Any]:
    """Validate a completed direct-only phase and expose safe orchestration data."""

    report = _read_object(report_path, "phase report")
    output_dir = report_path.parent
    if any((output_dir / name).exists() for name in FORBIDDEN_OUTPUTS):
        raise ValueError("phase directory contains a forbidden training artifact")
    journal_path = output_dir / "typed_api_rescue.journal.jsonl"
    targets_path = output_dir / "direct_targets.jsonl"
    manifest_path = output_dir / "direct_manifest.json"
    events = load_journal(journal_path)
    if not events or not isinstance(events[0].get("contract"), Mapping):
        raise ValueError("phase journal lacks a sealed run contract")
    contract = events[0]["contract"]
    manifest = _read_object(manifest_path, "direct manifest")
    rows = _read_jsonl(targets_path, "direct targets")
    actual_journal = journal_record(journal_path)
    report_journal = report.get("journal")
    output_record = report.get("outputs", {}).get("direct_targets")
    charged = report.get("budget_charged")
    if (
        report.get("schema") != cascade.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("phase") != expected_phase
        or report.get("cohort_index") != expected_cohort
        or contract.get("schema") != cascade.RUN_SCHEMA
        or contract.get("phase") != expected_phase
        or contract.get("cohort_index") != expected_cohort
        or report.get("run_contract_sha256") != canonical_sha256(contract)
        or not isinstance(report_journal, Mapping)
        or report_journal.get("sha256") != actual_journal["sha256"]
        or not isinstance(output_record, Mapping)
        or output_record.get("sha256") != sha256_file(targets_path)
        or not isinstance(charged, Mapping)
        or charged.get("within_contract") is not True
        or report.get("repair_policy_manifest") is not None
        or manifest.get("schema") != cascade.DIRECT_MANIFEST_SCHEMA
        or manifest.get("direct_only") is not True
        or manifest.get("repair_conditioned_rows") != 0
        or manifest.get("gold_replay_rows") != 0
        or manifest.get("reasoning_rows") != 0
        or manifest.get("tests_in_training_output") is not False
        or manifest.get("private_feedback_in_training_output") is not False
        or manifest.get("targets") != output_record
        or manifest.get("rows") != len(rows)
    ):
        raise ValueError("phase report/direct-only contract differs")
    task_ids: list[str] = []
    for row in rows:
        code = str(row.get("dart_source") or "")
        task_id = str(row.get("task_id") or "")
        if (
            row.get("schema") != cascade.DIRECT_TARGET_SCHEMA
            or not task_id
            or not code.strip()
            or row.get("origin") != "external_teacher_direct_verified"
            or row.get("visible_train_passed") is not True
            or row.get("private_full_acceptance_passed") is not True
            or row.get("stability_runs") != cascade.STABILITY_RUNS
            or row.get("reasoning_present") is not False
            or row.get("repair_conditioned_training_source_present") is not False
            or row.get("gold_replay") is not False
            or cascade.base.sha256_text(code) != row.get("dart_source_sha256")
        ):
            raise ValueError("unsafe or unverified direct target")
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)) or manifest.get(
        "task_ids_sha256"
    ) != canonical_sha256(task_ids):
        raise ValueError("phase target identity accounting differs")
    results = [row for row in events if row.get("event") == "call_result"]
    intents = [row for row in events if row.get("event") == "call_intent"]
    verifications = [row for row in events if row.get("event") == "task_verification"]
    if not (len(results) == len(intents) == len(verifications)):
        raise ValueError("phase call/verification accounting differs")
    verified = set(task_ids)
    retry_ids: list[str] = []
    for result in results:
        response = result.get("response")
        finish_reason = (
            str(response.get("finish_reason") or "")
            if isinstance(response, Mapping)
            else ""
        )
        task_id = str(result.get("task_id") or "")
        if task_id not in verified and (
            result.get("parse_accepted") is not True or finish_reason == "length"
        ):
            retry_ids.append(task_id)
    schedule = report.get("schedule", {})
    if (
        len(retry_ids) != len(set(retry_ids))
        or schedule.get("retry_eligible_non_code_or_length_tasks") != len(retry_ids)
        or schedule.get("retry_eligible_task_ids_sha256")
        != canonical_sha256(retry_ids)
    ):
        raise ValueError("targeted retry evidence differs")
    spent = Decimal(str(charged.get("estimated_usd")))
    if not spent.is_finite() or spent < 0:
        raise ValueError("phase spend is invalid")
    return {
        "path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "phase": expected_phase,
        "cohort_index": expected_cohort,
        "journal_sha256": actual_journal["sha256"],
        "targets_path": str(targets_path.resolve()),
        "targets_sha256": sha256_file(targets_path),
        "verified_task_ids": task_ids,
        "retry_eligible_task_ids": retry_ids,
        "spent": spent,
    }


def publish_prior_index(root: Path, name: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            "position": index,
            "phase": row["phase"],
            "cohort_index": row["cohort_index"],
            "report_path": row["path"],
            "report_sha256": row["report_sha256"],
            "journal_sha256": row["journal_sha256"],
            "targets_sha256": row["targets_sha256"],
        }
        for index, row in enumerate(records)
    ]
    tsv_path = root / f"prior_reports_{name}.tsv"
    _exact_text(
        tsv_path,
        "".join(f"{row['report_sha256']}\t{row['report_path']}\n" for row in entries),
    )
    record = {
        "schema": INDEX_SCHEMA,
        "status": "complete",
        "entries": entries,
        "entries_sha256": canonical_sha256(entries),
        "tsv": {"path": str(tsv_path.resolve()), "sha256": sha256_file(tsv_path)},
    }
    require_exact_or_write(root / f"prior_reports_{name}.json", record)
    return record


def _invoke_default(launcher: Path, env: Mapping[str, str]) -> None:
    result = subprocess.run([str(launcher)], env=dict(env), check=False)
    if result.returncode:
        raise RuntimeError(f"phase launcher exited {result.returncode}")


def _phase_env(
    base_env: Mapping[str, str],
    *,
    phase: str,
    output_dir: Path,
    max_usd: Decimal,
    max_tasks: int,
    prior_index: Mapping[str, Any] | None = None,
    retry_source: Mapping[str, Any] | None = None,
    budget_skipped_retry_ids: Sequence[str] = (),
) -> dict[str, str]:
    env = dict(base_env)
    env.update(
        {
            "T5GEMMA_TYPED_API_PHASE": phase,
            "T5GEMMA_TYPED_API_COHORT_INDEX": "0",
            "T5GEMMA_TYPED_API_OUTPUT_DIR": str(output_dir.resolve()),
            "T5GEMMA_TYPED_API_FIXED_KIMI_COHORT_LIMIT": "1",
            "T5GEMMA_TYPED_API_MAX_USD": format(max_usd, "f"),
            "T5GEMMA_TYPED_API_MAX_TASKS": str(max_tasks),
        }
    )
    for name in (
        "T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT",
        "T5GEMMA_TYPED_API_SCHEDULE_SHA256",
        "T5GEMMA_TYPED_API_PRIOR_INDEX",
        "T5GEMMA_TYPED_API_PRIOR_INDEX_SHA256",
        "T5GEMMA_TYPED_API_RETRY_SOURCE_REPORT",
        "T5GEMMA_TYPED_API_RETRY_SOURCE_SHA256",
        "T5GEMMA_TYPED_API_RETRY_TASKS",
        "T5GEMMA_TYPED_API_RETRY_IDS_SHA256",
        "T5GEMMA_TYPED_API_BUDGET_SKIPPED_KIMI_RETRY_TASKS",
        "T5GEMMA_TYPED_API_BUDGET_SKIPPED_KIMI_RETRY_IDS_SHA256",
    ):
        env.pop(name, None)
    if prior_index is not None:
        env["T5GEMMA_TYPED_API_PRIOR_INDEX"] = prior_index["tsv"]["path"]
        env["T5GEMMA_TYPED_API_PRIOR_INDEX_SHA256"] = prior_index["tsv"]["sha256"]
    if retry_source is not None:
        ids = list(retry_source["retry_eligible_task_ids"])
        env.update(
            {
                "T5GEMMA_TYPED_API_RETRY_SOURCE_REPORT": retry_source["path"],
                "T5GEMMA_TYPED_API_RETRY_SOURCE_SHA256": retry_source["report_sha256"],
                "T5GEMMA_TYPED_API_RETRY_TASKS": str(len(ids)),
                "T5GEMMA_TYPED_API_RETRY_IDS_SHA256": canonical_sha256(ids),
            }
        )
    if budget_skipped_retry_ids:
        ids = list(budget_skipped_retry_ids)
        env["T5GEMMA_TYPED_API_BUDGET_SKIPPED_KIMI_RETRY_TASKS"] = str(len(ids))
        env["T5GEMMA_TYPED_API_BUDGET_SKIPPED_KIMI_RETRY_IDS_SHA256"] = canonical_sha256(ids)
    return env


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
    expected_initial_schedule_sha256: str = "",
    prior_index: Mapping[str, Any] | None = None,
    retry_source: Mapping[str, Any] | None = None,
    budget_skipped_retry_ids: Sequence[str] = (),
) -> dict[str, Any]:
    env = _phase_env(
        base_env,
        phase=phase,
        output_dir=output_dir,
        max_usd=max_usd,
        max_tasks=max_tasks,
        prior_index=prior_index,
        retry_source=retry_source,
        budget_skipped_retry_ids=budget_skipped_retry_ids,
    )
    plan_path = root / f"plan_{phase}_c000.json"
    plan_env = dict(env)
    plan_env["T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT"] = str(plan_path.resolve())
    invoke(launcher, plan_env)
    plan = _read_object(plan_path, "phase plan")
    schedule_sha = str(plan.get("selection", {}).get("task_ids_sha256") or "")
    planned_tasks = plan.get("selection", {}).get("scheduled_tasks")
    exact_task_count = phase in (
        cascade.PHASE_KIMI_INITIAL,
        cascade.PHASE_KIMI_RETRY,
    )
    if (
        plan.get("schema") != cascade.PLAN_SCHEMA
        or plan.get("status") != "complete"
        or plan.get("phase") != phase
        or plan.get("cohort_index") != 0
        or plan.get("fixed_kimi_cohort_limit") != 1
        or plan.get("provider_credentials_read") is not False
        or plan.get("frontier_api_calls") is not False
        or len(schedule_sha) != 64
        or not isinstance(planned_tasks, int)
        or planned_tasks < 0
        or planned_tasks > max_tasks
        or (exact_task_count and planned_tasks != max_tasks)
    ):
        raise ValueError("credential-free phase plan differs")
    if expected_initial_schedule_sha256 and schedule_sha != expected_initial_schedule_sha256:
        raise ValueError("Kimi cohort-0 plan differs from the independent projection audit")
    report_path = output_dir / "typed_api_rescue_report.json"
    events = load_journal(journal_path)
    if planned_tasks == 0:
        if phase != cascade.PHASE_SONNET_RESIDUAL:
            raise ValueError("only Sonnet may have an empty residual schedule")
        skipped = [
            row
            for row in events
            if row.get("event") == "phase_skipped_no_residual"
            and row.get("phase") == phase
        ]
        if not skipped:
            append_event(
                journal_path,
                {
                    "event": "phase_skipped_no_residual",
                    "phase": phase,
                    "cohort_index": 0,
                    "plan_sha256": sha256_file(plan_path),
                    "schedule_sha256": schedule_sha,
                },
            )
        elif len(skipped) != 1 or skipped[0].get("schedule_sha256") != schedule_sha:
            raise ValueError("Sonnet no-residual skip evidence differs")
        return {
            "path": "",
            "report_sha256": "",
            "phase": phase,
            "cohort_index": 0,
            "journal_sha256": "",
            "targets_path": "",
            "targets_sha256": "",
            "verified_task_ids": [],
            "retry_eligible_task_ids": [],
            "spent": Decimal(0),
            "skipped_no_residual": True,
        }
    completed = [
        row for row in events if row.get("event") == "phase_complete" and row.get("phase") == phase
    ]
    if len(completed) > 1:
        raise ValueError("orchestrator journal contains duplicate phase completion")
    if not completed:
        started = [
            row for row in events if row.get("event") == "phase_start" and row.get("phase") == phase
        ]
        if not started:
            append_event(
                journal_path,
                {
                    "event": "phase_start",
                    "phase": phase,
                    "cohort_index": 0,
                    "plan_sha256": sha256_file(plan_path),
                    "schedule_sha256": schedule_sha,
                    "max_usd": format(max_usd, "f"),
                    "max_tasks": max_tasks,
                },
            )
        run_env = dict(env)
        run_env["T5GEMMA_TYPED_API_SCHEDULE_SHA256"] = schedule_sha
        invoke(launcher, run_env)
    record = inspect_phase_report(
        report_path, expected_phase=phase, expected_cohort=0
    )
    if record["spent"] > max_usd:
        raise ValueError("phase charged spend exceeds its sealed reservation")
    if not completed:
        append_event(
            journal_path,
            {
                "event": "phase_complete",
                "phase": phase,
                "cohort_index": 0,
                "report_sha256": record["report_sha256"],
                "journal_sha256": record["journal_sha256"],
                "targets_sha256": record["targets_sha256"],
                "spent": format(record["spent"], "f"),
            },
        )
    elif completed[0].get("report_sha256") != record["report_sha256"]:
        raise ValueError("completed phase report changed after sealing")
    return record


def publish_aggregate(root: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if record.get("skipped_no_residual"):
            continue
        for row in _read_jsonl(Path(record["targets_path"]), "phase direct targets"):
            task_id = str(row.get("task_id") or "")
            if task_id in seen:
                raise ValueError("provider phases produced a duplicate verified task")
            seen.add(task_id)
            rows.append(row)
    target_path = root / "direct_targets.jsonl"
    _exact_text(
        target_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )
    manifest = {
        "schema": AGGREGATE_SCHEMA,
        "rows": len(rows),
        "targets": {
            "path": str(target_path.resolve()),
            "sha256": sha256_file(target_path),
            "rows": len(rows),
        },
        "source_reports": [
            {
                "phase": row["phase"],
                "report_sha256": row["report_sha256"],
                "targets_sha256": row["targets_sha256"],
            }
            for row in records
        ],
        "task_ids_sha256": canonical_sha256([row["task_id"] for row in rows]),
        "direct_only": True,
        "visible_and_private_verified": True,
        "reasoning_rows": 0,
        "repair_conditioned_rows": 0,
        "gold_replay_rows": 0,
        "tests_in_training_output": False,
        "diagnostics_in_training_output": False,
        "production_floor_eligible": True,
    }
    require_exact_or_write(root / "direct_manifest.json", manifest)
    return manifest


def run(
    args: argparse.Namespace,
    *,
    invoke: Callable[[Path, Mapping[str, str]], None] = _invoke_default,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    launcher = Path(args.phase_launcher).expanduser().resolve()
    root = Path(args.output_root).expanduser().resolve()
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    if args.openrouter_max_usd != OPENROUTER_CAP or args.anthropic_max_usd != ANTHROPIC_CAP:
        raise ValueError("dual-provider budgets are fixed at OpenRouter $12 / Anthropic $11.50")
    if not isinstance(args.initial_schedule_sha256, str) or len(args.initial_schedule_sha256) != 64:
        raise ValueError("initial Kimi schedule requires an exact SHA-256")
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "orchestration.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "phase_launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
        "initial_schedule_sha256": args.initial_schedule_sha256,
        "kimi": {"cohorts": 1, "openrouter_max_usd": "12.0", "retry": "non_code_or_length_only"},
        "sonnet": {"model": cascade.SONNET_MODEL, "anthropic_max_usd": "11.5", "max_output_tokens": 16384},
        "training_output": "direct_verified_code_only",
    }
    events = load_journal(journal_path)
    if not events:
        append_event(journal_path, {"event": "header", "contract": contract, "contract_sha256": canonical_sha256(contract)})
    elif events[0].get("contract") != contract:
        raise ValueError("orchestration run contract differs")
    env = dict(os.environ if base_env is None else base_env)
    initial = execute_phase(
        launcher=launcher,
        root=root,
        journal_path=journal_path,
        phase=cascade.PHASE_KIMI_INITIAL,
        output_dir=root / "kimi_initial_c000",
        max_usd=OPENROUTER_CAP,
        max_tasks=50,
        base_env=env,
        invoke=invoke,
        expected_initial_schedule_sha256=args.initial_schedule_sha256,
    )
    records = [initial]
    retry_ids = initial["retry_eligible_task_ids"]
    budget_skipped_retry_ids: list[str] = []
    if retry_ids:
        remaining = OPENROUTER_CAP - initial["spent"]
        required = KIMI_RETRY_WORST * len(retry_ids)
        if required > remaining:
            budget_skipped_retry_ids = list(retry_ids)
            events = load_journal(journal_path)
            skips = [row for row in events if row.get("event") == "kimi_retry_budget_skip"]
            skip_record = {
                "event": "kimi_retry_budget_skip",
                "tasks": len(budget_skipped_retry_ids),
                "task_ids_sha256": canonical_sha256(budget_skipped_retry_ids),
                "required_worst_case_usd": format(required, "f"),
                "remaining_openrouter_usd": format(remaining, "f"),
                "reason": "exact_targeted_retry_set_does_not_fit_remaining_provider_cap",
            }
            if not skips:
                append_event(journal_path, skip_record)
            elif len(skips) != 1 or any(
                skips[0].get(key) != value for key, value in skip_record.items()
            ):
                raise ValueError("Kimi budget-skip evidence differs")
        else:
            retry = execute_phase(
                launcher=launcher,
                root=root,
                journal_path=journal_path,
                phase=cascade.PHASE_KIMI_RETRY,
                output_dir=root / "kimi_retry_c000",
                max_usd=remaining,
                max_tasks=len(retry_ids),
                base_env=env,
                invoke=invoke,
                retry_source=initial,
            )
            records.append(retry)
    openrouter_spent = sum((row["spent"] for row in records), Decimal(0))
    if openrouter_spent > OPENROUTER_CAP:
        raise ValueError("cumulative OpenRouter spend exceeds $12")
    kimi_index = publish_prior_index(root, "after_kimi", records)
    sonnet_tasks = int((ANTHROPIC_CAP / SONNET_WORST).to_integral_value(rounding=ROUND_FLOOR))
    sonnet = execute_phase(
        launcher=launcher,
        root=root,
        journal_path=journal_path,
        phase=cascade.PHASE_SONNET_RESIDUAL,
        output_dir=root / "sonnet_residual_c000",
        max_usd=ANTHROPIC_CAP,
        max_tasks=sonnet_tasks,
        base_env=env,
        invoke=invoke,
        prior_index=kimi_index,
        budget_skipped_retry_ids=budget_skipped_retry_ids,
    )
    if sonnet["spent"] > ANTHROPIC_CAP:
        raise ValueError("cumulative Anthropic spend exceeds $11.50")
    records.append(sonnet)
    final_index = publish_prior_index(
        root,
        "final",
        [row for row in records if not row.get("skipped_no_residual")],
    )
    aggregate = publish_aggregate(root, records)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": canonical_sha256(contract),
        "providers": {
            "openrouter": {"max_usd": "12.0", "charged_usd": format(openrouter_spent, "f")},
            "anthropic": {"max_usd": "11.5", "charged_usd": format(sonnet["spent"], "f")},
        },
        "kimi_retry": {
            "eligible_tasks": len(retry_ids),
            "budget_skipped_tasks": len(budget_skipped_retry_ids),
            "budget_skipped_task_ids_sha256": canonical_sha256(budget_skipped_retry_ids),
            "skip_reason": (
                "exact_targeted_retry_set_does_not_fit_remaining_provider_cap"
                if budget_skipped_retry_ids
                else None
            ),
        },
        "phases": [
            {
                key: row[key]
                for key in ("phase", "cohort_index", "report_sha256", "journal_sha256", "targets_sha256")
            }
            for row in records
        ],
        "prior_report_index": final_index,
        "direct_manifest": aggregate,
        "journal": journal_record(journal_path),
        "heldout_175_opened_for_exclusion_audit": True,
        "heldout_175_model_visible": False,
        "heldout_175_used_for_generation_or_selection": False,
    }
    require_exact_or_write(root / "orchestration_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase-launcher", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--initial-schedule-sha256", required=True)
    parser.add_argument("--openrouter-max-usd", type=Decimal, default=OPENROUTER_CAP)
    parser.add_argument("--anthropic-max-usd", type=Decimal, default=ANTHROPIC_CAP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"status": report["status"], "rows": report["direct_manifest"]["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
