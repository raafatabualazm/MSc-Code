#!/usr/bin/env python3
"""Attested cohort-continuation adapter for the typed API cascade.

The original cascade remains byte-identical so completed cohort-0 and pass-2
artifacts retain their producer binding.  This adapter changes exactly one
selection rule for later Kimi cohorts: every identity scheduled by any prior
provider phase is excluded, including Sonnet residual schedules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training.t5gemma2_local_rs_sft_pilot import PilotTask


def phase_selection(
    *,
    args: Any,
    all_visible_zero: Sequence[tuple[int, PilotTask, Mapping[str, Any]]],
    prior_records: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, PilotTask, Mapping[str, Any]]], dict[str, Any]]:
    """Validate with the base rule, then exclude every prior API schedule."""

    selected, record = _ORIGINAL_PHASE_SELECTION(
        args=args,
        all_visible_zero=all_visible_zero,
        prior_records=prior_records,
    )
    if args.phase != cascade.PHASE_KIMI_INITIAL or args.cohort_index == 0:
        return selected, record
    all_prior_scheduled = {
        str(task_id)
        for prior in prior_records
        for task_id in prior["scheduled_task_ids"]
    }
    residual = [
        row
        for row in all_visible_zero
        if row[1].task_id not in all_prior_scheduled
    ]
    selected = residual[: args.max_tasks]
    if len(selected) != args.max_tasks:
        raise ValueError("continuation lacks a complete 50-task residual cohort")
    selected_ids = [row[1].task_id for row in selected]
    if all_prior_scheduled & set(selected_ids):
        raise ValueError("continuation selection overlaps a prior provider schedule")
    all_ids = [row[1].task_id for row in all_visible_zero]
    record = {
        **record,
        "selection_adapter": "exclude_all_prior_provider_schedules",
        "base_selection_replaced": True,
        "prior_all_provider_scheduled_tasks_excluded": len(
            all_prior_scheduled
        ),
        "prior_all_provider_scheduled_task_ids_sha256": canonical_sha256(
            [task_id for task_id in all_ids if task_id in all_prior_scheduled]
        ),
        "selected_task_ids_sha256": canonical_sha256(selected_ids),
    }
    return selected, record


_ORIGINAL_PHASE_SELECTION = cascade._phase_selection  # noqa: SLF001


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = cascade.parse_args(argv)
    if args.phase not in (
        cascade.PHASE_KIMI_INITIAL,
        cascade.PHASE_KIMI_RETRY,
    ):
        raise ValueError("continuation adapter permits Kimi phases only")
    if args.cohort_index != 1 or args.fixed_kimi_cohort_limit != 2:
        raise ValueError("continuation adapter is sealed to cohort 1 of 2")
    if args.max_input_tokens_per_call != 16384:
        raise ValueError("continuation adapter fixes a 16,384-token input cap")

    original_selection = cascade._phase_selection  # noqa: SLF001
    original_file = cascade.__file__
    try:
        cascade._phase_selection = phase_selection  # noqa: SLF001
        # The underlying run records sha256(Path(__file__)); point that module
        # global at this adapter so plans/reports attest the behavior in use.
        cascade.__file__ = str(Path(__file__).resolve())
        return cascade.run(args)
    finally:
        cascade._phase_selection = original_selection  # noqa: SLF001
        cascade.__file__ = original_file


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
