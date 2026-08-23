#!/usr/bin/env python3
"""Fail-closed adapter for the unpaid 47-call tail of Kimi cohort 2.

The original cohort-2 phase is preserved as immutable evidence.  This adapter
reconstructs its exact fifty-task schedule, validates the sealed three-result
prefix, and schedules only positions 3..49 into a new phase journal.  It also
preflights every prompt before the cascade can read a provider credential.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_api_rescue_continuation_c002 as c002
from scripts.training.t5gemma2_local_rs_sft_pilot import PrivateGate


COHORT_INDEX = 2
FIXED_KIMI_COHORT_LIMIT = 3
ORIGINAL_TASKS = 50
PAID_PREFIX_TASKS = 3
TAIL_TASKS = ORIGINAL_TASKS - PAID_PREFIX_TASKS
MAX_INPUT_TOKENS = 30_720
INITIAL_MAX_OUTPUT_TOKENS = 4_096
RETRY_MAX_OUTPUT_TOKENS = 8_192
EXPECTED_MAX_TAIL_PROMPT_BYTE_UPPER_BOUND = 29_901

ORIGINAL_SCHEDULE_SHA256 = (
    "9dbbd75cd3d1bf231557179f3a318b49ea861cfbdefcd69d6190a354f1e7c40b"
)
TAIL_SCHEDULE_SHA256 = (
    "c9316309ecba02471fe62846d30476220d4f42a478e46acbd87d5be5b439eb49"
)
PREFIX_TASK_IDS = (
    "sigless_cd4741deab08",
    "sigless_a069da56acfa",
    "fresh-eval-dba1fc9af285",
)
SOURCE_PLAN_SHA256 = (
    "273e94b78074a68bb1e9dfa057d4620802bb9a787821805ae810e3e18d20ccd0"
)
SOURCE_JOURNAL_SHA256 = (
    "5005e6d090e7a7091b65d816abf5c387ca4f2459c49e49cbf686369580f57da4"
)
SOURCE_CHAIN_HEAD_SHA256 = (
    "5c224d735b9476acc98d77454f241cd4390261787613d6a253b7787fa33c3d3a"
)
PREFIX_CHARGED_USD_NANOS = 94_941_000
PREFIX_CHARGED_INPUT_TOKENS = 7_287
PREFIX_CHARGED_OUTPUT_TOKENS = 4_872
PREFLIGHT_SCHEMA = "t5gemma2-typed-kimi-c002-resume47-prompt-preflight-v1"


@dataclass(frozen=True)
class SourceEvidence:
    plan_path: Path
    journal_path: Path
    chain_head_path: Path
    plan: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    contract: dict[str, Any]
    journal_record: dict[str, Any]


_ORIGINAL_BUILD_SLOTS = cascade.build_typed_slots
_PREFLIGHT_PATH: Path | None = None
_SOURCE_EVIDENCE: SourceEvidence | None = None
_REQUESTED_PHASE = ""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _pin(path_value: str | Path, expected: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(f"{label} pin is malformed")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} differs from its immutable pin")
    return path


def load_source_evidence(
    *, plan_path: str | Path, journal_path: str | Path, chain_head_path: str | Path
) -> SourceEvidence:
    """Validate the immutable paid prefix without opening private tests."""

    plan_path = _pin(plan_path, SOURCE_PLAN_SHA256, "source cohort-2 plan")
    journal_path = _pin(
        journal_path, SOURCE_JOURNAL_SHA256, "source cohort-2 phase journal"
    )
    chain_head_path = _pin(
        chain_head_path, SOURCE_CHAIN_HEAD_SHA256, "source journal chain head"
    )
    plan = _read_object(plan_path, "source cohort-2 plan")
    events = load_journal(journal_path)
    current = journal_record(journal_path)
    if Path(current.get("chain_head_path", "")).resolve() != chain_head_path:
        raise ValueError("source journal adjacent chain-head path differs")
    if current.get("sha256") != SOURCE_JOURNAL_SHA256 or current.get(
        "chain_head_sha256"
    ) != SOURCE_CHAIN_HEAD_SHA256:
        raise ValueError("source journal record differs from immutable pins")
    if (
        plan.get("schema") != cascade.PLAN_SCHEMA
        or plan.get("status") != "complete"
        or plan.get("phase") != cascade.PHASE_KIMI_INITIAL
        or plan.get("cohort_index") != COHORT_INDEX
        or plan.get("provider_credentials_read") is not False
        or plan.get("frontier_api_calls") is not False
        or plan.get("selection", {}).get("scheduled_tasks") != ORIGINAL_TASKS
        or plan.get("selection", {}).get("scheduled_calls") != ORIGINAL_TASKS
        or plan.get("selection", {}).get("task_ids_sha256")
        != ORIGINAL_SCHEDULE_SHA256
        or plan.get("budget", {}).get("max_input_tokens_per_call") != 16_384
        or plan.get("budget", {}).get("max_output_tokens_per_call") != 4_096
    ):
        raise ValueError("source cohort-2 plan contract differs")
    if len(events) != 1 + 2 * PAID_PREFIX_TASKS:
        raise ValueError("source journal is not the exact three-result prefix")
    header = events[0]
    contract = header.get("contract")
    if (
        header.get("event") != "header"
        or not isinstance(contract, Mapping)
        or header.get("contract_sha256") != canonical_sha256(contract)
        or contract.get("schema") != cascade.RUN_SCHEMA
        or contract.get("phase") != cascade.PHASE_KIMI_INITIAL
        or contract.get("cohort_index") != COHORT_INDEX
        or contract.get("selection", {}).get("scheduled_tasks") != ORIGINAL_TASKS
        or contract.get("selection", {}).get("scheduled_slots") != ORIGINAL_TASKS
        or contract.get("selection", {}).get("task_ids_sha256")
        != ORIGINAL_SCHEDULE_SHA256
        or contract.get("budget") != plan.get("budget")
        or canonical_sha256(contract.get("inputs")) != plan.get("inputs_sha256")
    ):
        raise ValueError("source phase journal header differs from its plan")
    results: list[Mapping[str, Any]] = []
    for index in range(PAID_PREFIX_TASKS):
        intent = events[1 + 2 * index]
        result = events[2 + 2 * index]
        task_id = PREFIX_TASK_IDS[index]
        if (
            intent.get("event") != "call_intent"
            or result.get("event") != "call_result"
            or intent.get("slot_position") != index
            or result.get("slot_position") != index
            or intent.get("task_id") != task_id
            or result.get("task_id") != task_id
            or intent.get("request_sha256") != result.get("request_sha256")
            or result.get("status") != "response"
            or result.get("parse_accepted") is not True
            or not str(result.get("code") or "").strip()
            or base.sha256_text(str(result.get("code")))
            != result.get("code_sha256")
        ):
            raise ValueError("source paid-prefix event binding differs")
        results.append(result)
    if (
        sum(row["usage"]["charged_usd_nanos"] for row in results)
        != PREFIX_CHARGED_USD_NANOS
        or sum(row["usage"]["charged_input_tokens"] for row in results)
        != PREFIX_CHARGED_INPUT_TOKENS
        or sum(row["usage"]["charged_output_tokens"] for row in results)
        != PREFIX_CHARGED_OUTPUT_TOKENS
        or any(row.get("event") == "task_verification" for row in events)
    ):
        raise ValueError("source paid-prefix charge or verification state differs")
    return SourceEvidence(
        plan_path=plan_path,
        journal_path=journal_path,
        chain_head_path=chain_head_path,
        plan=plan,
        events=tuple(dict(row) for row in events),
        contract=dict(contract),
        journal_record=current,
    )


def _dummy_gates(
    selected: Sequence[tuple[int, Any, Mapping[str, Any]]],
) -> dict[str, PrivateGate]:
    return {
        row[1].task_id: PrivateGate(
            task_id=row[1].task_id,
            tests="not-opened-during-source-reconstruction",
            split_binding_sha256=row[1].split_binding_sha256,
        )
        for row in selected
    }


def _validate_reconstructed_source(
    selected: Sequence[tuple[int, Any, Mapping[str, Any]]],
) -> tuple[list[base.RescuePlan], list[base.ApiSlot]]:
    evidence = _SOURCE_EVIDENCE
    if evidence is None:
        raise ValueError("resume source evidence was not initialized")
    plans, _ = cascade.build_visible_only_plans(
        selected=selected, gates=_dummy_gates(selected)
    )
    slots = _ORIGINAL_BUILD_SLOTS(plans, samples_per_parent=1)
    task_ids = [plan.task.task_id for plan in plans]
    if (
        len(plans) != ORIGINAL_TASKS
        or canonical_sha256(task_ids) != ORIGINAL_SCHEDULE_SHA256
        or tuple(task_ids[:PAID_PREFIX_TASKS]) != PREFIX_TASK_IDS
        or canonical_sha256([base._slot_binding(slot) for slot in slots])
        != evidence.contract.get("selection", {}).get("slot_bindings_sha256")
    ):
        raise ValueError("reconstructed original cohort-2 schedule differs")
    # The source phase temporarily installed the typed cascade's system/schema
    # constants while hashing its call requests; reconstruct in the same scope.
    with cascade._typed_base_schemas():  # noqa: SLF001
        state = base.validate_rescue_journal(
            evidence.events,
            contract=evidence.contract,
            plans=plans,
            slots=slots,
        )
    if (
        len(state["slot_results"]) != PAID_PREFIX_TASKS
        or state["verification_events"]
        or state["complete"]
    ):
        raise ValueError("source journal is not an unverified three-result prefix")
    return plans, slots


def phase_selection(
    *,
    args: Any,
    all_visible_zero: Sequence[tuple[int, Any, Mapping[str, Any]]],
    prior_records: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, Any, Mapping[str, Any]]], dict[str, Any]]:
    if args.phase != cascade.PHASE_KIMI_INITIAL:
        return _ORIGINAL_PHASE_SELECTION(
            args=args,
            all_visible_zero=all_visible_zero,
            prior_records=prior_records,
        )
    # The old c002 adapter first reconstructs the exact original 50, including
    # all predecessor yield gates and exclusions.  Only then may we slice it.
    original, record = c002.phase_selection(
        args=argparse.Namespace(**{**vars(args), "max_tasks": ORIGINAL_TASKS}),
        all_visible_zero=all_visible_zero,
        prior_records=prior_records,
    )
    _validate_reconstructed_source(original)
    tail = list(original[PAID_PREFIX_TASKS:])
    tail_ids = [row[1].task_id for row in tail]
    if len(tail) != TAIL_TASKS or canonical_sha256(tail_ids) != TAIL_SCHEDULE_SHA256:
        raise ValueError("unpaid cohort-2 tail differs from its exact schedule pin")
    if set(PREFIX_TASK_IDS) & set(tail_ids):
        raise ValueError("paid prefix would be called again")
    return tail, {
        **record,
        "selection_adapter": "immutable_c002_prefix3_then_exact_tail47",
        "base_selection_replaced": True,
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "source_journal_sha256": SOURCE_JOURNAL_SHA256,
        "source_chain_head_sha256": SOURCE_CHAIN_HEAD_SHA256,
        "original_scheduled_tasks": ORIGINAL_TASKS,
        "original_task_ids_sha256": ORIGINAL_SCHEDULE_SHA256,
        "paid_prefix_tasks": PAID_PREFIX_TASKS,
        "paid_prefix_task_ids_sha256": canonical_sha256(list(PREFIX_TASK_IDS)),
        "paid_prefix_recalled": False,
        "selected_task_ids_sha256": TAIL_SCHEDULE_SHA256,
    }


_ORIGINAL_PHASE_SELECTION = cascade._phase_selection  # noqa: SLF001


def _preflight_build_slots(
    plans: Sequence[base.RescuePlan], *, samples_per_parent: int
) -> list[base.ApiSlot]:
    slots = _ORIGINAL_BUILD_SLOTS(plans, samples_per_parent=samples_per_parent)
    bounds = [
        len(slot.prompt.encode("utf-8"))
        + len(base.SYSTEM_PROMPT.encode("utf-8"))
        + 1024
        for slot in slots
    ]
    if not bounds or len(bounds) != len(slots):
        raise ValueError("resume prompt preflight did not cover every selected slot")
    maximum = max(bounds)
    if maximum > MAX_INPUT_TOKENS:
        raise ValueError(
            f"resume prompt preflight exceeds {MAX_INPUT_TOKENS}: {maximum}"
        )
    if _REQUESTED_PHASE == cascade.PHASE_KIMI_INITIAL and (
        len(slots) != TAIL_TASKS
        or canonical_sha256([slot.task_id for slot in slots])
        != TAIL_SCHEDULE_SHA256
    ):
        raise ValueError("full tail-47 prompt-bound audit differs")
    record = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete",
        "phase": _REQUESTED_PHASE,
        "slots_checked": len(slots),
        "all_selected_slots_checked_before_first_live_call": True,
        "task_ids_sha256": canonical_sha256([slot.task_id for slot in slots]),
        "slot_bindings_sha256": canonical_sha256(
            [base._slot_binding(slot) for slot in slots]
        ),
        "prompt_byte_upper_bounds_sha256": canonical_sha256(bounds),
        "max_prompt_byte_upper_bound": maximum,
        "max_input_tokens_per_call": MAX_INPUT_TOKENS,
        "within_reservation": True,
        "provider_credentials_required": False,
        "frontier_api_calls": False,
    }
    if _PREFLIGHT_PATH is None:
        raise ValueError("resume prompt-preflight output path is absent")
    require_exact_or_write(_PREFLIGHT_PATH, record)
    return slots


def _parse_custom(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--resume_source_plan", required=True)
    parser.add_argument("--resume_source_journal", required=True)
    parser.add_argument("--resume_source_chain_head", required=True)
    parser.add_argument("--prompt_preflight_output", required=True)
    return parser.parse_known_args(argv)


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    global _PREFLIGHT_PATH, _REQUESTED_PHASE, _SOURCE_EVIDENCE
    raw = list(sys.argv[1:] if argv is None else argv)
    custom, remaining = _parse_custom(raw)
    requested_phase = c002._required_option(remaining, "--phase")  # noqa: SLF001
    try:
        requested_output = int(
            c002._required_option(remaining, "--max_output_tokens")  # noqa: SLF001
        )
    except ValueError as exc:
        raise ValueError("resume output cap is malformed") from exc
    _SOURCE_EVIDENCE = load_source_evidence(
        plan_path=custom.resume_source_plan,
        journal_path=custom.resume_source_journal,
        chain_head_path=custom.resume_source_chain_head,
    )
    _PREFLIGHT_PATH = Path(custom.prompt_preflight_output).expanduser().resolve()
    _REQUESTED_PHASE = requested_phase
    original_selection = cascade._phase_selection  # noqa: SLF001
    original_build_slots = cascade.build_typed_slots
    original_file = cascade.__file__
    original_cohort_size = cascade.KIMI_COHORT_SIZE
    original_initial_output = cascade.KIMI_INITIAL_MAX_OUTPUT
    try:
        cascade.KIMI_COHORT_SIZE = TAIL_TASKS
        cascade.KIMI_INITIAL_MAX_OUTPUT = INITIAL_MAX_OUTPUT_TOKENS
        args = cascade.parse_args(remaining)
        if args.phase not in (cascade.PHASE_KIMI_INITIAL, cascade.PHASE_KIMI_RETRY):
            raise ValueError("resume adapter permits Kimi phases only")
        if (
            args.cohort_index != COHORT_INDEX
            or args.fixed_kimi_cohort_limit != FIXED_KIMI_COHORT_LIMIT
            or args.max_input_tokens_per_call != MAX_INPUT_TOKENS
        ):
            raise ValueError("resume adapter cohort/input contract differs")
        if args.phase == cascade.PHASE_KIMI_INITIAL:
            if args.max_tasks != TAIL_TASKS or requested_output != INITIAL_MAX_OUTPUT_TOKENS:
                raise ValueError("resume initial phase is exactly 47 calls at 4,096")
        elif requested_output != RETRY_MAX_OUTPUT_TOKENS:
            raise ValueError("resume retry phase is fixed at 8,192 output tokens")
        cascade._phase_selection = phase_selection  # noqa: SLF001
        cascade.build_typed_slots = _preflight_build_slots
        cascade.__file__ = str(Path(__file__).resolve())
        return cascade.run(args)
    finally:
        cascade._phase_selection = original_selection  # noqa: SLF001
        cascade.build_typed_slots = original_build_slots
        cascade.__file__ = original_file
        cascade.KIMI_COHORT_SIZE = original_cohort_size
        cascade.KIMI_INITIAL_MAX_OUTPUT = original_initial_output
        _PREFLIGHT_PATH = None
        _SOURCE_EVIDENCE = None
        _REQUESTED_PHASE = ""


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
