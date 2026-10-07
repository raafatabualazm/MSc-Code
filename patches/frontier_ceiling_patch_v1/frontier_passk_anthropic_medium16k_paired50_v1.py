#!/usr/bin/env python3
"""Sealed paired Sonnet 5 medium-effort pilot over 50 calls per F2 arm.

The cohort contains 25 held-out tasks for which both K=2 high-effort responses
hit the 16,384-token ceiling in both enrichment arms.  The task selection is a
deterministic hash-ranked subset of that common failure cohort.  This module
reuses the audited Message Batches transport, response normalizer, evaluator,
and append-only journals, but:

* sends exactly 25 tasks x K=2 = 50 requests per arm;
* uses adaptive thinking with ``effort=medium`` and a fixed 16,384-token cap;
* treats a length stop at that cap as a terminal pilot observation;
* has no retry/escalation cap after 16K;
* writes into a distinct run root selected by the launcher; and
* validates the immutable high-effort source journals before any API action.

The inherited runner retains historical ``primary_8192`` filenames.  The
authoritative corrected pilot result is ``medium_16384_pilot_summary.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import frontier_passk_anthropic_batch_budget100_v2 as runner
from frontier_core import (
    atomic_write_json,
    file_record,
    load_jsonl,
    sha256_file,
    stable_sha256,
    utc_now,
)


SCHEMA = "anthropic-sonnet5-medium16k-paired50-v1"
SELECTION_SCHEMA = "anthropic-sonnet5-medium16k-common-k2-selection-v1"
SELECTION_DOMAIN = "anthropic-sonnet5-medium-pilot-k2-tasks-v1"
SELECTION_DIGEST = "c69baa54717e80c771e6067c32c121d475db010a2157bb144471d288a9131b5a"
EFFORT = "medium"
K = 2
CAP = 16_384
PILOT_ARM_COST_CAP_USD = 4.9152
EXPECTED_CALLS_PER_ARM = 50

SOURCE_ROOT = Path(
    "/workspace/artifacts/frontier_ceiling_two_enrichments/runs/"
    "anthropic_sonnet5_benign_f2_k2_budget100_v2"
)
SOURCE_ATTEMPT_SHA256 = {
    "opus": "870fd46a77234fdc4e142c0bf781824d9ac7e848dfee5f2a0e6966db027249b4",
    "codex": "70ad3d56b837ba6b40da719cdd89c7042849230f87a4abbd76f553db5f09f440",
}
SOURCE_PROGRESS_SHA256 = {
    "opus": "1cff48dfd383e812b41fb53e80a9eba61304bb7094858a9ded67c8be13a9ff0e",
    "codex": "41afa03792095fde6fedc3d90d702df20c131062bb21b22e5798f567669d43fc",
}
PRIOR_HIGH_LIST_USD = {
    "opus": 40.811956,
    "codex": 43.528088,
}

# Rank order under SELECTION_DOMAIN.  Every task has both sample indices 0/1
# censored at 16K in both source arms.
SELECTED_TASKS: tuple[tuple[int, str], ...] = (
    (174, "sigless_54c0d3ca981f"),
    (127, "sigless_fc85f3ba4f9a"),
    (167, "sigless_b721d6a533f8"),
    (87, "sigless_254a6441f2f8"),
    (42, "sigless_8d0cf57a5b8f"),
    (40, "sigless_b03bf3bb9c5b"),
    (67, "sigless_6dfdaa80a5a8"),
    (168, "sigless_bcb6f4de3081"),
    (66, "sigless_79bfe84158a3"),
    (103, "sigless_bd986d86cebb"),
    (82, "sigless_4708b0b01b11"),
    (109, "sigless_95105e48c1e8"),
    (171, "sigless_bf3de39765c8"),
    (131, "sigless_2d693beca98e"),
    (34, "sigless_f19a1cde385c"),
    (70, "sigless_083954eb2fb3"),
    (161, "sigless_7412fc8a275b"),
    (21, "sigless_217ef71865c3"),
    (122, "sigless_897c175fe34d"),
    (33, "sigless_842d8bd3822e"),
    (36, "sigless_6ca6f0196a67"),
    (57, "sigless_5a81bb01e5b7"),
    (143, "sigless_4b8ae6c364f9"),
    (93, "sigless_35088d24eba1"),
    (55, "sigless_ae3f6b074a1d"),
)
SELECTED_TASK_IDS = frozenset(task_id for _, task_id in SELECTED_TASKS)


def selection_claim() -> dict[str, Any]:
    return {
        "schema": SELECTION_SCHEMA,
        "domain": SELECTION_DOMAIN,
        "selection_digest": SELECTION_DIGEST,
        "selection_digest_serialization": "rank-ordered UTF-8 lines: TTT,task_id\\n",
        "tasks": len(SELECTED_TASKS),
        "k": K,
        "calls_per_arm": EXPECTED_CALLS_PER_ARM,
        "selected_tasks_rank_order": [
            {"source_task_index": index, "task_id": task_id}
            for index, task_id in SELECTED_TASKS
        ],
        "source_attempt_sha256": dict(SOURCE_ATTEMPT_SHA256),
        "source_progress_sha256": dict(SOURCE_PROGRESS_SHA256),
        "source_required_cap": CAP,
        "source_required_finish_reason": "length",
        "source_required_native_stop_reason": "max_tokens",
    }


def validate_source_selection() -> dict[str, Any]:
    serialized = "".join(
        f"{index:03d},{task_id}\n" for index, task_id in SELECTED_TASKS
    )
    import hashlib

    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != SELECTION_DIGEST:
        raise runner.audited.RunFailure(
            "embedded medium-pilot selection digest mismatch"
        )
    if len(SELECTED_TASKS) != 25 or len(SELECTED_TASK_IDS) != 25:
        raise runner.audited.RunFailure("medium-pilot selection must contain 25 tasks")

    records: dict[str, Any] = {}
    expected_keys = {
        (task_id, sample_index)
        for _, task_id in SELECTED_TASKS
        for sample_index in range(K)
    }
    expected_indices = {task_id: index for index, task_id in SELECTED_TASKS}
    for arm in ("opus", "codex"):
        attempts_path = SOURCE_ROOT / arm / "batch_slot_attempts.jsonl"
        progress_path = SOURCE_ROOT / arm / "progress.json"
        if sha256_file(attempts_path) != SOURCE_ATTEMPT_SHA256[arm]:
            raise runner.audited.RunFailure(
                f"{arm} high-effort attempt source hash changed"
            )
        if sha256_file(progress_path) != SOURCE_PROGRESS_SHA256[arm]:
            raise runner.audited.RunFailure(
                f"{arm} high-effort progress source hash changed"
            )
        matching: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in load_jsonl(attempts_path, f"{arm} high-effort attempts"):
            key = (str(row.get("task_id") or ""), int(row.get("sample_index", -1)))
            if key not in expected_keys:
                continue
            if (
                int(row.get("requested_max_tokens", -1)) == CAP
                and row.get("result_type") == "succeeded"
                and row.get("finish_reason") == "length"
                and row.get("native_stop_reason") == "max_tokens"
            ):
                if key in matching:
                    raise runner.audited.RunFailure(
                        f"duplicate qualifying source response for {arm}/{key}"
                    )
                if int(row.get("task_index", -1)) != expected_indices[key[0]]:
                    raise runner.audited.RunFailure(
                        f"source task index mismatch for {arm}/{key}"
                    )
                matching[key] = row
        if set(matching) != expected_keys:
            missing = sorted(expected_keys - set(matching))
            raise runner.audited.RunFailure(
                f"{arm} source lacks {len(missing)} selected high-effort truncations"
            )
        records[arm] = {
            "attempts": file_record(attempts_path),
            "progress": file_record(progress_path),
            "qualifying_slots": len(matching),
        }
    return records


# Preserve references before installing the pilot overrides.
_ORIGINAL_PENDING = runner.pending_request_specs
_ORIGINAL_WRITE_PROGRESS = runner._write_progress_or_summary
_ORIGINAL_FIXED_SLOT_POLICY = runner.fixed_slot_policy
_ORIGINAL_CONFIG_FOR_HASH = runner.config_for_hash
_ORIGINAL_PARSE_ARGS = runner.parse_args
_ORIGINAL_SUBMIT = runner._submit

# All inherited functions resolve these module globals dynamically.
runner.SCHEMA = SCHEMA
runner.EFFORT = EFFORT
runner.K = K
runner.CAP_LADDER = (CAP,)
runner.DEFAULT_ARM_COST_CAP_USD = PILOT_ARM_COST_CAP_USD
# Bind runtime identity to this companion rather than the inherited source.
runner.__file__ = __file__


def parse_args(argv: Sequence[str] | None = None) -> Any:
    args = _ORIGINAL_PARSE_ARGS(argv)
    if args.action == "auto":
        raise SystemExit(
            "ACTION=auto is disabled for the one-batch medium-effort pilot"
        )
    return args


def pending_request_specs(
    plans: Sequence[Mapping[str, Any]],
    slot_attempts: Sequence[Mapping[str, Any]],
    terminals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    plan_ids = [str(plan.get("task_id") or "") for plan in plans]
    if not SELECTED_TASK_IDS.issubset(plan_ids):
        raise runner.audited.RunFailure("sealed plan set lacks a selected pilot task")
    specs = _ORIGINAL_PENDING(plans, slot_attempts, terminals)
    selected = [spec for spec in specs if str(spec["task_id"]) in SELECTED_TASK_IDS]
    if not slot_attempts and not terminals and len(selected) != EXPECTED_CALLS_PER_ARM:
        raise runner.audited.RunFailure(
            f"pilot preflight produced {len(selected)} requests, expected "
            f"{EXPECTED_CALLS_PER_ARM}"
        )
    return selected


def _submit(
    args: Any,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
) -> dict[str, Any]:
    events = runner._batch_events(out, config_sha)
    if any(row.get("event_type") == "batch_submitted" for row in events):
        raise runner.audited.RunFailure(
            "the medium-effort pilot permits exactly one paid batch per arm"
        )
    specs = pending_request_specs(
        plans,
        runner._slot_attempts(out, config_sha),
        runner._terminal_rows(out, config_sha),
    )
    if len(specs) != EXPECTED_CALLS_PER_ARM:
        raise runner.audited.RunFailure(
            f"pilot submission has {len(specs)} requests, expected "
            f"{EXPECTED_CALLS_PER_ARM}"
        )
    if any(int(spec["cap"]) != CAP for spec in specs):
        raise runner.audited.RunFailure("pilot submission contains a non-16K request")
    return _ORIGINAL_SUBMIT(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
    )


def fixed_slot_policy(args: Any) -> dict[str, Any]:
    policy = _ORIGINAL_FIXED_SLOT_POLICY(args)
    policy["pilot_selection"] = selection_claim()
    policy["fixed_medium_16k_only"] = True
    policy["no_capacity_retry"] = True
    return policy


def config_for_hash(args: Any) -> dict[str, Any]:
    config = _ORIGINAL_CONFIG_FOR_HASH(args)
    screen = dict(config["anthropic_batch_screen"])
    screen["pilot_selection"] = selection_claim()
    screen["fixed_medium_16k_only"] = True
    screen["no_capacity_retry"] = True
    screen["prior_high_list_usd"] = dict(PRIOR_HIGH_LIST_USD)
    screen["paired_combined_budget_usd"] = 100.0
    config["anthropic_batch_screen"] = screen
    return config


def _arm_from_args(args: Any) -> str:
    if args.pair_arm_key == "opus_real_fn0_cfg":
        return "opus"
    if args.pair_arm_key == "codex_multifunction_cfg":
        return "codex"
    raise runner.audited.RunFailure(f"unknown paired arm {args.pair_arm_key!r}")


def _write_progress_or_summary(
    args: Any,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
    evaluator_record: Mapping[str, Any],
) -> dict[str, Any]:
    selected_plans = [
        plan for plan in plans if str(plan.get("task_id") or "") in SELECTED_TASK_IDS
    ]
    if len(selected_plans) != len(SELECTED_TASKS):
        raise runner.audited.RunFailure("selected pilot plan count changed")
    result = _ORIGINAL_WRITE_PROGRESS(
        args,
        out=out,
        plans=selected_plans,
        config_sha=config_sha,
        provenance=provenance,
        evaluator_record=evaluator_record,
    )
    if result.get("status") != "complete":
        return result

    arm = _arm_from_args(args)
    attempts = runner._slot_attempts(out, config_sha)
    terminals = runner._terminal_rows(out, config_sha)
    reporting = runner._rows_with_native_stop_metadata(terminals, attempts)
    stop_counts: dict[str, int] = {}
    for row in reporting:
        reason = str(row.get("native_stop_reason") or "missing")
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
    medium_cost = dict(result["usage_and_list_cost"])
    medium_list_usd = float(medium_cost["estimated_total_usd"])
    corrected = {
        "schema": SCHEMA,
        "status": "complete",
        "completed_at": utc_now(),
        "config_sha256": config_sha,
        "arm": arm,
        "model": runner.MODEL_ID,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
        "max_output_tokens": CAP,
        "tasks": len(selected_plans),
        "k": K,
        "logical_slots": EXPECTED_CALLS_PER_ARM,
        "selection": selection_claim(),
        "native_stop_reason_counts": stop_counts,
        "normal_completion_slots": stop_counts.get("end_turn", 0),
        "length_censored_slots": stop_counts.get("max_tokens", 0),
        "refusal_slots": stop_counts.get("refusal", 0),
        "pass_at_2_medium_16384": result["capacity_adaptive_pass_at_2"],
        "compile_at_2_medium_16384": result["capacity_adaptive_compile_at_2"],
        "pilot_usage_and_list_cost": medium_cost,
        "prior_high_usage_list_usd": PRIOR_HIGH_LIST_USD[arm],
        "cumulative_arm_list_usd": PRIOR_HIGH_LIST_USD[arm] + medium_list_usd,
        "screen_arm_cost_cap_usd": 50.0,
        "paired_combined_budget_usd": 100.0,
        "source_high_effort": validate_source_selection()[arm],
        "base_summary": file_record(out / "summary.json"),
        "terminal_slots": file_record(out / "terminal_slots.jsonl"),
        "outcomes": file_record(out / "outcomes.jsonl"),
    }
    corrected["summary_sha256"] = stable_sha256(corrected)
    atomic_write_json(out / "medium_16384_pilot_summary.json", corrected)
    return corrected


# Install all overrides used by inherited main/transport code.
runner.pending_request_specs = pending_request_specs
runner.fixed_slot_policy = fixed_slot_policy
runner.config_for_hash = config_for_hash
runner._write_progress_or_summary = _write_progress_or_summary
runner.parse_args = parse_args
runner._submit = _submit


def main(argv: Sequence[str] | None = None) -> int:
    source = validate_source_selection()
    print(
        "MEDIUM16K_PILOT_SELECTION_OK "
        f"tasks={len(SELECTED_TASKS)} calls_per_arm={EXPECTED_CALLS_PER_ARM} "
        f"digest={SELECTION_DIGEST} "
        f"source={stable_sha256(source)}",
        flush=True,
    )
    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
