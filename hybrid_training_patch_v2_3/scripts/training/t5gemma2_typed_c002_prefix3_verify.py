#!/usr/bin/env python3
"""Zero-API verification of the immutable three-result Kimi c002 prefix.

The interrupted source phase contains three paid, parsed responses but no
private verification events.  This program reconstructs the exact original
50-task schedule, validates those three call results against it, waits on a
SHA-pinned completed tail-47 report, and only then evaluates the three saved
programs locally on visible and complete TRAIN acceptance tests.  It never
reads a provider credential and emits direct verified code only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.graph_compile_at_k_antigravity import validate_dart_binary
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_api_rescue_continuation_c002 as c002_adapter
from scripts.training import t5gemma2_typed_api_rescue_c002_resume47 as resume_adapter
from scripts.training import t5gemma2_typed_direct_rs_sft_pass3 as pass3
from scripts.training import t5gemma2_typed_kimi_continuation as c001
from scripts.training import t5gemma2_typed_kimi_continuation_c002 as c002
from scripts.training import t5gemma2_typed_kimi_c002_resume47 as resume47


RUN_SCHEMA = "t5gemma2-typed-c002-prefix3-verification-run-v1"
REPORT_SCHEMA = pass3.PREFIX_REPORT_SCHEMA
MANIFEST_SCHEMA = pass3.PREFIX_MANIFEST_SCHEMA
TARGET_SCHEMA = pass3.PREFIX_TARGET_SCHEMA
JOURNAL_SCHEMA = "t5gemma2-typed-c002-prefix3-verification-journal-v1"
EXPECTED_VERIFIED_IDS = (pass3.EXPECTED_PREFIX_TASK_ID,)


def _pin(path_value: str | Path, expected: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    value = str(expected or "")
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or not path.is_file()
        or sha256_file(path) != value
    ):
        raise ValueError(f"{label} differs from its exact SHA-256 pin")
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _write_journal_exact(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    existing = load_journal(path)
    expected = [dict(row) for row in events]
    if existing:
        payloads = [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "journal_event_index",
                    "journal_previous_event_sha256",
                    "journal_event_sha256",
                }
            }
            for row in existing
        ]
        if payloads != expected:
            raise ValueError("prefix verification journal differs from exact rerun")
        return
    for event in expected:
        append_event(path, event)


def _validate_completed_tail(
    report_path: Path, report_sha256: str
) -> dict[str, Any]:
    report = _read_object(_pin(report_path, report_sha256, "resume47 report"), "resume47 report")
    phases = report.get("tail_phases")
    if (
        report.get("schema") != resume47.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("heldout_175_opened") is not False
        or not isinstance(phases, list)
        or not phases
        or phases[0].get("phase") != cascade.PHASE_KIMI_INITIAL
        or phases[0].get("cohort_index") != 2
        or report.get("source_partial", {}).get("paid_prefix_tasks") != 3
        or report.get("source_partial", {}).get("source_files_modified") is not False
        or report.get("prefix_disposition", {}).get("training_used_in_this_stage") is not False
    ):
        raise ValueError("resume47 is not the completed immutable-tail predecessor")
    initial_report = report_path.parent / "kimi_initial_tail47" / "typed_api_rescue_report.json"
    record = c001.inspect_phase_report(
        initial_report,
        expected_phase=cascade.PHASE_KIMI_INITIAL,
        expected_cohort=2,
    )
    if (
        record["report_sha256"] != phases[0].get("report_sha256")
        or record["journal_sha256"] != phases[0].get("journal_sha256")
        or record["targets_sha256"] != phases[0].get("targets_sha256")
        or len(record["scheduled_task_ids"]) != resume_adapter.TAIL_TASKS
        or canonical_sha256(record["scheduled_task_ids"])
        != resume_adapter.TAIL_SCHEDULE_SHA256
    ):
        raise ValueError("resume47 initial phase is not the exact completed tail")
    return {
        "path": str(report_path),
        "sha256": report_sha256,
        "tail_initial_report_sha256": record["report_sha256"],
        "tail_initial_journal_sha256": record["journal_sha256"],
        "tail_schedule_sha256": resume_adapter.TAIL_SCHEDULE_SHA256,
        "all_original_generation_complete_before_private_prefix_gate": True,
    }


def _context_args(parser: argparse.ArgumentParser) -> None:
    for name in (
        "gold_train_jsonl",
        "gold_f2_jsonl",
        "expected_gold_train_sha256",
        "expected_gold_f2_sha256",
        "heldout_jsonl",
        "expected_heldout_sha256",
        "local_harvest_report",
        "expected_local_harvest_report_sha256",
        "pilot_journal",
        "expected_local_harvest_journal_sha256",
        "local_harvest_targets",
        "expected_local_harvest_targets_sha256",
        "existing_direct_manifest",
        "expected_existing_direct_manifest_sha256",
        "visible_split_manifest",
        "expected_visible_split_manifest_sha256",
        "visible_train",
        "expected_visible_train_sha256",
        "private_split_holdback",
        "expected_private_split_holdback_sha256",
        "visible_projection_report",
        "expected_visible_projection_report_sha256",
        "visible_projection_journal",
        "expected_visible_projection_journal_sha256",
    ):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    parser.add_argument("--expected-gold-rows", type=int, default=2776)
    parser.add_argument("--expected-heldout-rows", type=int, default=175)


def _reconstruct(
    args: argparse.Namespace,
    *,
    evidence: resume_adapter.SourceEvidence,
    prior_records: Sequence[Mapping[str, Any]],
) -> tuple[list[base.RescuePlan], list[base.ApiSlot], list[dict[str, Any]], Any, dict[str, Any]]:
    context = cascade.load_typed_source_context(args)
    projection, projection_record = cascade.load_visible_projection(args, context=context)
    existing_ids, _existing_record = cascade.load_existing_225_exclusions(
        args.existing_direct_manifest,
        args.expected_existing_direct_manifest_sha256,
    )
    seed = int(evidence.contract.get("selection", {}).get("seed", -1))
    if seed < 0:
        raise ValueError("source prefix selection seed is absent")
    eligible = cascade.select_visible_zero_tasks(
        context=context,
        projection_terminals=projection,
        seed=seed,
        excluded_ids=set(existing_ids),
    )
    selection_args = argparse.Namespace(
        phase=cascade.PHASE_KIMI_INITIAL,
        cohort_index=2,
        fixed_kimi_cohort_limit=3,
        max_tasks=50,
        budget_skipped_kimi_retry_tasks=0,
        budget_skipped_kimi_retry_task_ids_sha256="",
    )
    selected, selection_record = c002_adapter.phase_selection(
        args=selection_args,
        all_visible_zero=eligible,
        prior_records=prior_records,
    )
    plans, _diagnostic = cascade.build_visible_only_plans(
        selected=selected,
        gates=context.gates,
    )
    slots = cascade.build_typed_slots(plans, samples_per_parent=1)
    if (
        len(plans) != resume_adapter.ORIGINAL_TASKS
        or len(slots) != resume_adapter.ORIGINAL_TASKS
        or canonical_sha256([plan.task.task_id for plan in plans])
        != resume_adapter.ORIGINAL_SCHEDULE_SHA256
        or tuple(plan.task.task_id for plan in plans[:3])
        != resume_adapter.PREFIX_TASK_IDS
        or canonical_sha256([base._slot_binding(slot) for slot in slots])  # noqa: SLF001
        != evidence.contract.get("selection", {}).get("slot_bindings_sha256")
    ):
        raise ValueError("original cohort-2 schedule reconstruction differs")
    with cascade._typed_base_schemas():  # noqa: SLF001
        state = base.validate_rescue_journal(
            evidence.events,
            contract=evidence.contract,
            plans=plans,
            slots=slots,
        )
    if (
        len(state["slot_results"]) != 3
        or state["verification_events"]
        or state["complete"]
    ):
        raise ValueError("immutable source is not exactly three unverified results")
    return plans, slots, list(state["slot_results"]), context, {
        "selection": selection_record,
        "projection": projection_record,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_journal = Path(args.source_journal).expanduser().resolve()
    source_before = sha256_file(source_journal)
    evidence = resume_adapter.load_source_evidence(
        plan_path=args.source_plan,
        journal_path=source_journal,
        chain_head_path=args.source_chain_head,
    )
    prior_records, _remaining, c001_record = c002.load_completed_c001(
        report_path=args.c001_report,
        expected_report_sha256=args.expected_c001_report_sha256,
    )
    tail_record = _validate_completed_tail(
        Path(args.resume47_report).expanduser().resolve(),
        args.expected_resume47_report_sha256,
    )
    plans, slots, results, _context, reconstruction = _reconstruct(
        args, evidence=evidence, prior_records=prior_records
    )
    validate_dart_binary()
    evaluate = base._runtime_evaluator(timeout=args.timeout, stability_runs=2)  # noqa: SLF001
    verification_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for position in range(3):
        plan, slot, result = plans[position], slots[position], results[position]
        code = str(result.get("code") or "")
        visible = evaluate(code, plan.task.visible_tests, f"c002-prefix-visible-{position}")
        visible_passed = bool(visible.compiled and visible.passed)
        private_passed = False
        if visible_passed:
            private = evaluate(code, plan.gate.tests, f"c002-prefix-private-{position}")
            private_passed = bool(private.compiled and private.passed)
        accepted = visible_passed and private_passed
        verification_rows.append(
            {
                "position": position,
                "task_id": plan.task.task_id,
                "source_sha256": plan.task.source_sha256,
                "code_sha256": result["code_sha256"],
                "visible_passed": visible_passed,
                "private_complete_passed": private_passed,
                "accepted": accepted,
                "diagnostic_persisted": False,
            }
        )
        if accepted:
            selected_rows.append(
                {
                    "schema": TARGET_SCHEMA,
                    "task_id": plan.task.task_id,
                    "source_sha256": plan.task.source_sha256,
                    "dart_source": code,
                    "dart_source_sha256": result["code_sha256"],
                    "origin": "external_teacher_direct_verified_zero_api_prefix_recovery",
                    "provider_phase": cascade.PHASE_KIMI_INITIAL,
                    "provider_model": evidence.contract.get("provider", {}).get("model"),
                    "visible_train_passed": True,
                    "private_full_acceptance_passed": True,
                    "stability_runs": 2,
                    "reasoning_present": False,
                    "repair_conditioned_training_source_present": False,
                    "gold_replay": False,
                    "provenance": {
                        "source_journal_sha256": resume_adapter.SOURCE_JOURNAL_SHA256,
                        "source_contract_sha256": canonical_sha256(evidence.contract),
                        "slot_position": slot.slot_position,
                        "request_sha256": result["request_sha256"],
                        "parent_code_sha256": slot.parent.code_sha256,
                        "diagnostic_sha256": slot.parent.diagnostic_sha256,
                    },
                }
            )
    verified_ids = tuple(row["task_id"] for row in selected_rows)
    if verified_ids != EXPECTED_VERIFIED_IDS:
        raise ValueError(
            "prefix private verification outcome differs from its sealed expected disposition"
        )
    if sha256_file(source_journal) != source_before:
        raise ValueError("immutable source journal changed during zero-API verification")

    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source": {
            "plan_sha256": resume_adapter.SOURCE_PLAN_SHA256,
            "journal_sha256": resume_adapter.SOURCE_JOURNAL_SHA256,
            "chain_head_sha256": resume_adapter.SOURCE_CHAIN_HEAD_SHA256,
            "prefix_task_ids_sha256": canonical_sha256(list(resume_adapter.PREFIX_TASK_IDS)),
            "source_files_modified": False,
        },
        "completed_tail": tail_record,
        "c001_predecessor": c001_record,
        "reconstruction_sha256": canonical_sha256(reconstruction),
        "provider_calls": 0,
        "provider_credentials_read": False,
        "private_gate": "complete_TRAIN_acceptance_local_only",
        "heldout_175_opened": False,
        "tests_model_visible": False,
    }
    journal_events: list[dict[str, Any]] = [
        {
            "event": "header",
            "schema": JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        }
    ]
    journal_events.extend(
        {"event": "task_verification", "schema": JOURNAL_SCHEMA, **row}
        for row in verification_rows
    )
    journal_events.append(
        {
            "event": "complete",
            "schema": JOURNAL_SCHEMA,
            "tasks": 3,
            "verified_targets": len(selected_rows),
            "verified_task_ids_sha256": canonical_sha256(list(verified_ids)),
            "provider_calls": 0,
            "private_diagnostics_persisted": False,
        }
    )
    journal_path = output_dir / "prefix_verification.journal.jsonl"
    _write_journal_exact(journal_path, journal_events)
    target_path = output_dir / "direct_targets.jsonl"
    base._exact_write_jsonl(target_path, selected_rows)  # noqa: SLF001
    target_record = {
        "path": str(target_path),
        "sha256": sha256_file(target_path),
        "rows": len(selected_rows),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "rows": len(selected_rows),
        "targets": target_record,
        "task_ids_sha256": canonical_sha256(list(verified_ids)),
        "direct_only": True,
        "visible_and_private_verified": True,
        "reasoning_rows": 0,
        "repair_conditioned_rows": 0,
        "gold_replay_rows": 0,
        "tests_in_training_output": False,
        "diagnostics_in_training_output": False,
        "provider_calls": 0,
        "production_floor_eligible": True,
    }
    require_exact_or_write(output_dir / "direct_manifest.json", manifest)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": canonical_sha256(contract),
        "paid_prefix_tasks": 3,
        "verified_targets": len(selected_rows),
        "verified_task_ids": list(verified_ids),
        "provider_calls": 0,
        "provider_credentials_read": False,
        "heldout_175_opened": False,
        "private_diagnostics_persisted": False,
        "tests_in_training_output": False,
        "source_journal_modified": False,
        "direct_manifest": manifest,
        "journal": journal_record(journal_path),
    }
    require_exact_or_write(output_dir / "prefix_verification_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    _context_args(parser)
    parser.add_argument("--c001-report", required=True)
    parser.add_argument("--expected-c001-report-sha256", required=True)
    parser.add_argument("--resume47-report", required=True)
    parser.add_argument("--expected-resume47-report-sha256", required=True)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--source-journal", required=True)
    parser.add_argument("--source-chain-head", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if args.expected_gold_rows != 2776 or args.expected_heldout_rows != 175:
        parser.error("prefix verification fixes TRAIN=2776 and heldout=175 identities")
    if args.timeout != 30:
        parser.error("prefix verification fixes --timeout=30")
    for name, value in vars(args).items():
        if name.startswith("expected_") and name.endswith("_sha256"):
            if len(str(value)) != 64 or any(ch not in "0123456789abcdef" for ch in str(value)):
                parser.error(f"--{name.replace('_', '-')} is not a lowercase SHA-256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"T5GEMMA_TYPED_C002_PREFIX3_VERIFY_BLOCKED {exc}", flush=True)
        return 78
    print(
        "T5GEMMA_TYPED_C002_PREFIX3_VERIFY_COMPLETE "
        + json.dumps(
            {"verified": report["verified_targets"], "provider_calls": 0},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
