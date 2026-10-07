#!/usr/bin/env python3
"""Freeze and attest an exact Kimi rescue schedule from a completed local run."""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
)
from scripts.training import t5gemma2_api_rs_sft_rescue as rescue


SCHEMA = "t5gemma2-kimi-rescue-schedule-attestation-v1"


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} differs")


def build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    events = load_journal(args.pilot_journal)
    if not events:
        raise ValueError("local pilot journal is empty")
    header = events[0]
    contract = header.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("local pilot contract is missing")
    compatibility = contract.get("checkpoint_loader_compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("journal-bound checkpoint loader compatibility is missing")

    exact_contract_values = (
        (contract.get("script_sha256"), args.expected_pilot_sha256, "pilot SHA"),
        (
            contract.get("checkpoint_contract_sha256"),
            args.expected_checkpoint_contract_sha256,
            "checkpoint contract SHA",
        ),
        (
            compatibility.get("wrapper_sha256"),
            args.expected_wrapper_sha256,
            "wrapper SHA",
        ),
        (
            compatibility.get("inference_core_sha256"),
            args.expected_inference_sha256,
            "inference SHA",
        ),
        (
            compatibility.get("mixed_loader_sha256"),
            args.expected_mixed_loader_sha256,
            "mixed loader SHA",
        ),
    )
    for observed, expected, label in exact_contract_values:
        _require_exact(observed, expected, label)
    _require_exact(
        compatibility.get("schema"),
        "t5gemma2-mixed-local-rs-sft-loader-compat-v1",
        "loader compatibility schema",
    )
    _require_exact(contract.get("heldout_175_opened"), False, "heldout isolation")
    _require_exact(contract.get("no_frontier_api"), True, "local-only generation")

    schedule = contract.get("schedule")
    sampling = contract.get("sampling")
    training_build = contract.get("training_build")
    if not all(
        isinstance(value, Mapping)
        for value in (schedule, sampling, training_build)
    ):
        raise ValueError("local schedule/sampling/training contract is malformed")
    required_controls = (
        (schedule.get("seed"), args.local_seed, "local seed"),
        (schedule.get("pilot_offset"), 0, "local offset"),
        (schedule.get("pilot_tasks"), 100, "local task count"),
        (sampling.get("base_samples"), 4, "local base samples"),
        (sampling.get("repair_samples"), 0, "local repair samples"),
        (sampling.get("max_repair_parents"), 0, "local repair parents"),
        (sampling.get("repair_enabled"), False, "local repair disabled"),
        (training_build.get("gold_replay_ratio"), 0, "local gold replay"),
    )
    for observed, expected, label in required_controls:
        _require_exact(observed, expected, label)

    loader_args = Namespace(
        rollout_file=args.rollout_file,
        f2_jsonl=args.f2_jsonl,
        private_holdback=args.private_holdback,
        expected_rollout_sha256=args.expected_rollout_sha256,
        expected_f2_sha256=args.expected_f2_sha256,
        expected_private_holdback_sha256=args.expected_private_holdback_sha256,
        allow_unpinned_inputs=False,
        pilot_journal=args.pilot_journal,
        exploratory_terminal_prefix=0,
    )
    (
        _all_tasks,
        gates,
        scheduled_tasks,
        terminals,
        input_record,
        source_record,
    ) = rescue._load_completed_local_run(loader_args)
    plans = rescue.select_rescue_plans(
        scheduled_tasks=scheduled_tasks,
        gates=gates,
        terminals=terminals,
        seed=args.rescue_seed,
        max_tasks=0,
        max_parents_per_task=args.max_parents_per_task,
        eligible_task_offset=0,
    )
    if len(plans) < args.tasks:
        raise ValueError(
            f"only {len(plans)} all-zero tasks have usable parents; "
            f"{args.tasks} required"
        )
    selected = plans[: args.tasks]
    slots = rescue.build_slots(selected, samples_per_parent=1)
    max_prompt_byte_upper_bound = max(
        len(slot.prompt.encode("utf-8"))
        + len(rescue.SYSTEM_PROMPT.encode("utf-8"))
        + 1024
        for slot in slots
    )
    if max_prompt_byte_upper_bound > args.max_input_reservation:
        raise ValueError(
            "selected Kimi prompt exceeds the fail-closed input reservation: "
            f"{max_prompt_byte_upper_bound} > {args.max_input_reservation}"
        )
    task_ids = [plan.task.task_id for plan in selected]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "source_local_pilot_journal": source_record,
        "source_local_pilot_journal_current": journal_record(args.pilot_journal),
        "inputs": input_record,
        "checkpoint_contract_sha256": args.expected_checkpoint_contract_sha256,
        "loader_compatibility": dict(compatibility),
        "selection": {
            "rescue_seed": args.rescue_seed,
            "max_parents_per_task": args.max_parents_per_task,
            "eligible_all_zero_tasks": len(plans),
            "scheduled_tasks": len(selected),
            "task_ids": task_ids,
            "task_ids_sha256": canonical_sha256(task_ids),
            "slot_bindings_sha256": canonical_sha256(
                [rescue._slot_binding(slot) for slot in slots]
            ),
            "max_prompt_byte_upper_bound": max_prompt_byte_upper_bound,
            "max_input_reservation": args.max_input_reservation,
            "deterministic_all_zero_order": True,
            "usable_parent_required": True,
        },
        "heldout_175_opened": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pilot_journal", type=Path, required=True)
    parser.add_argument("--rollout_file", type=Path, required=True)
    parser.add_argument("--f2_jsonl", type=Path, required=True)
    parser.add_argument("--private_holdback", type=Path, required=True)
    parser.add_argument("--expected_rollout_sha256", required=True)
    parser.add_argument("--expected_f2_sha256", required=True)
    parser.add_argument("--expected_private_holdback_sha256", required=True)
    parser.add_argument("--expected_pilot_sha256", required=True)
    parser.add_argument("--expected_wrapper_sha256", required=True)
    parser.add_argument("--expected_inference_sha256", required=True)
    parser.add_argument("--expected_mixed_loader_sha256", required=True)
    parser.add_argument("--expected_checkpoint_contract_sha256", required=True)
    parser.add_argument("--local_seed", type=int, required=True)
    parser.add_argument("--rescue_seed", type=int, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--max_parents_per_task", type=int, required=True)
    parser.add_argument("--max_input_reservation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in (
        "expected_rollout_sha256",
        "expected_f2_sha256",
        "expected_private_holdback_sha256",
        "expected_pilot_sha256",
        "expected_wrapper_sha256",
        "expected_inference_sha256",
        "expected_mixed_loader_sha256",
        "expected_checkpoint_contract_sha256",
    ):
        value = getattr(args, name)
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            parser.error(f"--{name} requires a lowercase SHA-256")
    if (
        args.local_seed < 0
        or args.rescue_seed < 0
        or args.tasks <= 0
        or args.max_parents_per_task <= 0
        or args.max_input_reservation <= 0
    ):
        parser.error("seeds must be non-negative and caps must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    attestation = build_attestation(args)
    require_exact_or_write(args.output.expanduser().resolve(), attestation)
    print(attestation["selection"]["task_ids_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
