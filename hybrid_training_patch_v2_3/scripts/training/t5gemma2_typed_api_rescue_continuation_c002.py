#!/usr/bin/env python3
"""Cohort-2 selection adapter for the sealed typed Kimi rescue cascade."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training.t5gemma2_api_rs_sft_rescue import PilotTask


COHORT_INDEX = 2
FIXED_KIMI_COHORT_LIMIT = 3


def phase_selection(
    *,
    args: Any,
    all_visible_zero: Sequence[tuple[int, PilotTask, Mapping[str, Any]]],
    prior_records: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, PilotTask, Mapping[str, Any]]], dict[str, Any]]:
    """Apply the base yield gate, then exclude every prior API schedule."""

    selected, record = _ORIGINAL_PHASE_SELECTION(
        args=args,
        all_visible_zero=all_visible_zero,
        prior_records=prior_records,
    )
    if args.phase != cascade.PHASE_KIMI_INITIAL:
        return selected, record
    all_prior_scheduled = {
        str(task_id)
        for prior in prior_records
        for task_id in prior["scheduled_task_ids"]
    }
    residual = [
        row for row in all_visible_zero if row[1].task_id not in all_prior_scheduled
    ]
    selected = residual[: args.max_tasks]
    if len(selected) != args.max_tasks:
        raise ValueError("cohort 2 lacks a complete 50-task residual cohort")
    selected_ids = [row[1].task_id for row in selected]
    if all_prior_scheduled & set(selected_ids):
        raise ValueError("cohort-2 selection overlaps a prior provider schedule")
    all_ids = [row[1].task_id for row in all_visible_zero]
    return selected, {
        **record,
        "selection_adapter": "exclude_all_prior_provider_schedules",
        "base_selection_replaced": True,
        "prior_all_provider_scheduled_tasks_excluded": len(all_prior_scheduled),
        "prior_all_provider_scheduled_task_ids_sha256": canonical_sha256(
            [task_id for task_id in all_ids if task_id in all_prior_scheduled]
        ),
        "selected_task_ids_sha256": canonical_sha256(selected_ids),
    }


_ORIGINAL_PHASE_SELECTION = cascade._phase_selection  # noqa: SLF001


def _required_option(argv: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"cohort-2 adapter requires exactly one {name}")
    return str(argv[positions[0] + 1])


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    requested_phase = _required_option(raw_argv, "--phase")
    try:
        requested_output = int(_required_option(raw_argv, "--max_output_tokens"))
    except ValueError as exc:
        raise ValueError("cohort-2 output cap is malformed") from exc
    original_selection = cascade._phase_selection  # noqa: SLF001
    original_file = cascade.__file__
    original_initial_max_output = cascade.KIMI_INITIAL_MAX_OUTPUT
    try:
        # parse_args validates against this module constant.  Seal the c002
        # controller's already-budgeted adaptive choice *before* parsing.
        if requested_phase == cascade.PHASE_KIMI_INITIAL:
            if requested_output not in (2048, 4096):
                raise ValueError("cohort-2 initial output cap must be 4,096 or 2,048")
            cascade.KIMI_INITIAL_MAX_OUTPUT = requested_output
        args = cascade.parse_args(raw_argv)
        if args.phase not in (cascade.PHASE_KIMI_INITIAL, cascade.PHASE_KIMI_RETRY):
            raise ValueError("cohort-2 adapter permits Kimi phases only")
        if (
            args.cohort_index != COHORT_INDEX
            or args.fixed_kimi_cohort_limit != FIXED_KIMI_COHORT_LIMIT
        ):
            raise ValueError("cohort-2 adapter is sealed to cohort 2 of 3")
        if args.max_input_tokens_per_call != 16384:
            raise ValueError("cohort-2 adapter fixes a 16,384-token input cap")
        cascade._phase_selection = phase_selection  # noqa: SLF001
        cascade.__file__ = str(Path(__file__).resolve())
        return cascade.run(args)
    finally:
        cascade._phase_selection = original_selection  # noqa: SLF001
        cascade.__file__ = original_file
        cascade.KIMI_INITIAL_MAX_OUTPUT = original_initial_max_output


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
