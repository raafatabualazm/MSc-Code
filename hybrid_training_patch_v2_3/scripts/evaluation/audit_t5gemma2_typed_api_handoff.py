#!/usr/bin/env python3
"""Fail-closed audits for the typed local-harvest -> API-rescue handoff.

The ``harvest`` audit reconstructs the complete 2,550-task ledger from the
sealed TRAIN inputs, checks the exact 225-task predecessor exclusion, validates
the hash chain and every published target, and independently re-executes every
accepted target on its complete TRAIN acceptance suite.

The ``projection`` audit runs only after the CPU-only visible projection has
completed.  It validates the split/projection ledgers and seals the exact first
Kimi cohort schedule.  Neither command reads a provider credential or makes an
API request.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.graph_compile_at_k_antigravity import (
    evaluate_dart_jit_tests_detail,
    validate_dart_binary,
)
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_local_direct_harvest as local


HARVEST_AUDIT_SCHEMA = "t5gemma2-typed-local-harvest-handoff-audit-v1"
PROJECTION_AUDIT_SCHEMA = "t5gemma2-typed-api-projection-handoff-audit-v1"
EXPECTED_LOCAL_TASKS = 2550
EXPECTED_SAMPLES_PER_TASK = 4
EXPECTED_PREVIOUS_TARGETS = 225
EXPECTED_STABILITY_RUNS = 2
EXPECTED_LOCAL_HARVEST_SCRIPT_SHA256 = (
    "875517222f2aa3a1cd823d476b44cd51f49fb2a7dff8f2e4a5cb18466622264a"
)
EXPECTED_CHECKPOINT = {
    "checkpoint_stage": "typed_direct",
    "checkpoint_update": 58,
    "run_contract_sha256": (
        "0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f"
    ),
    "training_state_sha256": (
        "6960bc8bdd4b8bafc8e732fc36ac011ccdf8a8f6246a0d3f29f5996235717e89"
    ),
    "adapter_weights_sha256": (
        "62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec"
    ),
    "adapter_config_sha256": (
        "b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b"
    ),
}


VerifyFn = Callable[[str, str, str, int], bool]


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _runtime_verify(code: str, tests: str, slot: str, timeout: int) -> bool:
    compiled, passed, _diagnostic, _details = evaluate_dart_jit_tests_detail(
        code,
        tests,
        slot,
        timeout=timeout,
        stability_runs=EXPECTED_STABILITY_RUNS,
    )
    return bool(compiled and passed)


def _validate_checkpoint_lineage(report: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    adapter = checkpoint.get("adapter") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("checkpoint_stage")
        != EXPECTED_CHECKPOINT["checkpoint_stage"]
        or checkpoint.get("checkpoint_update")
        != EXPECTED_CHECKPOINT["checkpoint_update"]
        or checkpoint.get("training_state_sha256")
        != EXPECTED_CHECKPOINT["training_state_sha256"]
        or not isinstance(adapter, Mapping)
        or adapter.get("run_contract_sha256")
        != EXPECTED_CHECKPOINT["run_contract_sha256"]
        or adapter.get("adapter_weights_sha256")
        != EXPECTED_CHECKPOINT["adapter_weights_sha256"]
        or adapter.get("adapter_config_sha256")
        != EXPECTED_CHECKPOINT["adapter_config_sha256"]
    ):
        raise ValueError("typed local handoff checkpoint lineage differs")
    return {
        "checkpoint_stage": checkpoint["checkpoint_stage"],
        "checkpoint_update": checkpoint["checkpoint_update"],
        "run_contract_sha256": adapter["run_contract_sha256"],
        "training_state_sha256": checkpoint["training_state_sha256"],
        "adapter_weights_sha256": adapter["adapter_weights_sha256"],
        "adapter_config_sha256": adapter["adapter_config_sha256"],
    }


def _validate_harvest_contract(journal_path: str | Path) -> dict[str, Any]:
    events = load_journal(Path(journal_path))
    contract = events[0].get("contract") if events else None
    sampling = contract.get("sampling") if isinstance(contract, Mapping) else None
    schedule = contract.get("schedule") if isinstance(contract, Mapping) else None
    verification = (
        contract.get("verification") if isinstance(contract, Mapping) else None
    )
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema") != local.RUN_SCHEMA
        or contract.get("script_sha256") != EXPECTED_LOCAL_HARVEST_SCRIPT_SHA256
        or sha256_file(Path(local.__file__).resolve())
        != EXPECTED_LOCAL_HARVEST_SCRIPT_SHA256
        or contract.get("checkpoint_stage") != "typed_direct"
        or not isinstance(schedule, Mapping)
        or schedule.get("seed") != 42
        or schedule.get("clean_train_rows") != 2775
        or schedule.get("excluded_previous_direct_tasks") != 225
        or schedule.get("scheduled_tasks") != 2550
        or not isinstance(sampling, Mapping)
        or sampling.get("samples_per_task") != 4
        or sampling.get("repair_samples") != 0
        or sampling.get("max_repair_parents") != 0
        or sampling.get("temperature") != 0.8
        or sampling.get("top_p") != 0.95
        or sampling.get("max_source_tokens") != 32768
        or sampling.get("max_new_tokens") != 4096
        or sampling.get("generation_batch_size") != 4
        or sampling.get("silent_source_truncation") is not False
        or not isinstance(verification, Mapping)
        or verification.get("suite") != "complete_train_acceptance"
        or verification.get("stability_runs") != 2
        or verification.get("timeout_seconds") != 30
        or verification.get("workers") != 8
        or verification.get("all_k_generated_before_gate") is not True
        or verification.get("gate_can_only_reject_transfer") is not True
        or verification.get("diagnostics_persisted") is not False
        or contract.get("complete_acceptance_model_visible") is not False
        or contract.get("frontier_api_calls") is not False
        or contract.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or contract.get("outputs", {}).get("repair_conditioned_rows") != 0
        or contract.get("outputs", {}).get("gold_replay_rows") != 0
    ):
        raise ValueError("typed local handoff generation/verification contract differs")
    return {
        "script_sha256": contract["script_sha256"],
        "seed": schedule["seed"],
        "samples_per_task": sampling["samples_per_task"],
        "repair_samples": sampling["repair_samples"],
        "max_source_tokens": sampling["max_source_tokens"],
        "max_new_tokens": sampling["max_new_tokens"],
        "all_k_generated_before_gate": verification[
            "all_k_generated_before_gate"
        ],
    }


def audit_harvest(
    args: argparse.Namespace,
    *,
    verify: VerifyFn | None = None,
) -> dict[str, Any]:
    """Deeply validate and independently re-run a completed local harvest."""

    loaded = local.load_completed_harvest_artifacts(
        report_path=args.local_harvest_report,
        expected_report_sha256=args.expected_local_harvest_report_sha256,
        journal_path=args.local_harvest_journal,
        expected_journal_sha256=args.expected_local_harvest_journal_sha256,
        targets_path=args.local_harvest_targets,
        expected_targets_sha256=args.expected_local_harvest_targets_sha256,
        gold_train_jsonl=args.gold_train_jsonl,
        expected_gold_train_sha256=args.expected_gold_train_sha256,
        gold_f2_jsonl=args.gold_f2_jsonl,
        expected_gold_f2_sha256=args.expected_gold_f2_sha256,
        heldout_jsonl=args.heldout_jsonl,
        expected_heldout_sha256=args.expected_heldout_sha256,
        expected_gold_rows=2776,
        expected_heldout_rows=175,
    )
    tasks, gates, scheduled, terminals, _input_record, source_record = loaded
    excluded_ids, exclusion_record = cascade.load_existing_225_exclusions(
        args.existing_direct_manifest,
        args.expected_existing_direct_manifest_sha256,
    )
    all_ids = {task.task_id for task in tasks}
    scheduled_ids = [task.task_id for task in scheduled]
    if (
        len(tasks) != 2775
        or len(scheduled) != EXPECTED_LOCAL_TASKS
        or len(terminals) != EXPECTED_LOCAL_TASKS
        or len(excluded_ids) != EXPECTED_PREVIOUS_TARGETS
        or all_ids - set(scheduled_ids) != set(excluded_ids)
    ):
        raise ValueError("typed local handoff schedule/exclusion differs")

    report = _read_json(args.local_harvest_report, "typed local harvest report")
    lineage = _validate_checkpoint_lineage(report)
    generation_contract = _validate_harvest_contract(args.local_harvest_journal)
    selected: list[tuple[str, str, str]] = []
    for terminal in terminals:
        target = terminal.get("selected_target")
        if isinstance(target, Mapping):
            task_id = str(terminal["task_id"])
            selected.append((task_id, str(target["code"]), gates[task_id].tests))

    if verify is None:
        validate_dart_binary()
        verify = _runtime_verify

    def one(item: tuple[str, str, str]) -> tuple[str, bool]:
        task_id, code, tests = item
        return task_id, verify(
            code,
            tests,
            f"typed-handoff-audit-{task_id}",
            args.timeout,
        )

    if args.evaluation_workers == 1:
        outcomes = [one(item) for item in selected]
    else:
        with ThreadPoolExecutor(max_workers=args.evaluation_workers) as pool:
            outcomes = list(pool.map(one, selected))
    failed = [task_id for task_id, passed in outcomes if not passed]
    if failed:
        raise ValueError(
            "typed local handoff independent acceptance recheck failed for "
            f"{len(failed)} target(s); first={failed[0]}"
        )

    output_inventory = report.get("outputs")
    if not isinstance(output_inventory, Mapping):
        raise ValueError("typed local handoff output inventory is absent")
    direct_f2 = Path(args.local_harvest_report).resolve().parent / "direct_f2.jsonl"
    schedule_manifest = (
        Path(args.local_harvest_report).resolve().parent / "schedule_manifest.jsonl"
    )
    audit = {
        "schema": HARVEST_AUDIT_SCHEMA,
        "status": "pass",
        "source_supervisor_required_state": "EXITED_after_sealed_completion",
        "artifacts": {
            "report_sha256": sha256_file(Path(args.local_harvest_report)),
            "journal_sha256": sha256_file(Path(args.local_harvest_journal)),
            "direct_targets_sha256": sha256_file(Path(args.local_harvest_targets)),
            "direct_f2_sha256": sha256_file(direct_f2),
            "schedule_manifest_sha256": sha256_file(schedule_manifest),
            "existing_225_manifest_sha256": sha256_file(
                Path(args.existing_direct_manifest)
            ),
        },
        "lineage": lineage,
        "generation_contract": generation_contract,
        "schedule": {
            "clean_train_tasks": len(tasks),
            "excluded_previous_direct_tasks": len(excluded_ids),
            "scheduled_tasks": len(scheduled),
            "samples_per_task": EXPECTED_SAMPLES_PER_TASK,
            "task_ids_sha256": canonical_sha256(scheduled_ids),
            "terminal_events": len(terminals),
        },
        "accepted": {
            "direct_targets": len(selected),
            "task_ids_sha256": canonical_sha256([row[0] for row in selected]),
            "independently_reverified": len(outcomes),
            "independent_failures": 0,
            "stability_runs": EXPECTED_STABILITY_RUNS,
        },
        "exclusion": exclusion_record,
        "journal": {
            key: source_record[key]
            for key in (
                "sha256",
                "chain_head_sha256",
                "event_count",
                "head_event_sha256",
                "run_contract_sha256",
            )
        },
        "privacy": {
            "provider_credentials_read": False,
            "frontier_api_calls": False,
            "heldout_175_opened_for_exclusion_audit": True,
            "heldout_175_model_visible": False,
            "heldout_175_used_for_generation_or_selection": False,
            "private_diagnostics_persisted": False,
        },
    }
    require_exact_or_write(Path(args.output), audit)
    return audit


def audit_projection(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the visible-only projection and seal Kimi cohort zero."""

    context = cascade.load_typed_source_context(args)
    terminals, projection_record = cascade.load_visible_projection(
        args, context=context
    )
    existing_ids, exclusion_record = cascade.load_existing_225_exclusions(
        args.existing_direct_manifest,
        args.expected_existing_direct_manifest_sha256,
    )
    if (
        len(context.scheduled_tasks) != EXPECTED_LOCAL_TASKS
        or len(terminals) != EXPECTED_LOCAL_TASKS
    ):
        raise ValueError("typed API handoff projection task count differs")
    eligible = cascade.select_visible_zero_tasks(
        context=context,
        projection_terminals=terminals,
        seed=20260801,
        excluded_ids=set(existing_ids),
    )
    first_cohort = eligible[: cascade.KIMI_COHORT_SIZE]
    if len(first_cohort) != min(cascade.KIMI_COHORT_SIZE, len(eligible)):
        raise ValueError("typed API handoff first cohort selection differs")
    if not first_cohort:
        raise ValueError("typed API handoff has no visible-failure tasks")
    plans, diagnostic_record = cascade.build_visible_only_plans(
        selected=first_cohort,
        gates=context.gates,
    )
    if len(plans) != len(first_cohort):
        raise ValueError("typed API handoff cohort lacks a usable K=4 parent")
    schedule_ids = [plan.task.task_id for plan in plans]
    audit = {
        "schema": PROJECTION_AUDIT_SCHEMA,
        "status": "pass",
        "phase": cascade.PHASE_KIMI_INITIAL,
        "cohort_index": 0,
        "projection": projection_record,
        "source_local_harvest_journal": context.source_journal_record,
        "existing_225_exclusion": exclusion_record,
        "eligible_tasks": len(eligible),
        "first_cohort": {
            "tasks": len(plans),
            "task_ids_sha256": canonical_sha256(schedule_ids),
            "one_parent_per_task": True,
            "calls_reserved": len(plans),
        },
        "visible_diagnostic_provenance": diagnostic_record,
        "privacy": {
            "eligibility_uses_visible_train_split_only": True,
            "private_complete_outcome_used_for_eligibility": False,
            "private_complete_diagnostic_used": False,
            "provider_credentials_read": False,
            "frontier_api_calls": False,
            "heldout_175_opened_for_exclusion_audit": True,
            "heldout_175_model_visible": False,
            "heldout_175_used_for_generation_or_selection": False,
        },
    }
    require_exact_or_write(Path(args.output), audit)
    return audit


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local-harvest-report", required=True)
    parser.add_argument("--expected-local-harvest-report-sha256", required=True)
    parser.add_argument("--local-harvest-journal", required=True)
    parser.add_argument("--expected-local-harvest-journal-sha256", required=True)
    parser.add_argument("--local-harvest-targets", required=True)
    parser.add_argument("--expected-local-harvest-targets-sha256", required=True)
    parser.add_argument("--existing-direct-manifest", required=True)
    parser.add_argument("--expected-existing-direct-manifest-sha256", required=True)
    parser.add_argument("--gold-train-jsonl", required=True)
    parser.add_argument("--expected-gold-train-sha256", required=True)
    parser.add_argument("--gold-f2-jsonl", required=True)
    parser.add_argument("--expected-gold-f2-sha256", required=True)
    parser.add_argument("--heldout-jsonl", required=True)
    parser.add_argument("--expected-heldout-sha256", required=True)
    parser.add_argument("--output", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    harvest = subparsers.add_parser("harvest", allow_abbrev=False)
    _add_source_args(harvest)
    harvest.add_argument("--timeout", type=int, default=30)
    harvest.add_argument("--evaluation-workers", type=int, default=16)

    projection = subparsers.add_parser("projection", allow_abbrev=False)
    _add_source_args(projection)
    projection.add_argument("--visible-split-manifest", required=True)
    projection.add_argument("--expected-visible-split-manifest-sha256", required=True)
    projection.add_argument("--visible-train", required=True)
    projection.add_argument("--expected-visible-train-sha256", required=True)
    projection.add_argument("--private-split-holdback", required=True)
    projection.add_argument("--expected-private-split-holdback-sha256", required=True)
    projection.add_argument("--visible-projection-report", required=True)
    projection.add_argument("--expected-visible-projection-report-sha256", required=True)
    projection.add_argument("--visible-projection-journal", required=True)
    projection.add_argument("--expected-visible-projection-journal-sha256", required=True)

    args = parser.parse_args(argv)
    digest_fields = [
        name
        for name in vars(args)
        if name.startswith("expected_") and name.endswith("_sha256")
    ]
    for name in digest_fields:
        cascade._require_digest(getattr(args, name), name)  # noqa: SLF001
    if args.command == "harvest" and (
        args.timeout <= 0 or args.evaluation_workers <= 0
    ):
        parser.error("runtime audit caps must be positive")
    # The cascade loaders use this compatibility name.
    args.pilot_journal = args.local_harvest_journal
    args.expected_gold_rows = 2776
    args.expected_heldout_rows = 175
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "harvest":
        report = audit_harvest(args)
    else:
        report = audit_projection(args)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
