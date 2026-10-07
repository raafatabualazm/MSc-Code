#!/usr/bin/env python3
"""CPU-only partial-test reward audit for the sealed typed pass-1 k=4 harvest.

This audit reuses candidate programs that were already sampled from the typed
pass-1 policy.  It never loads a model.  For a deterministic subset of the
harvest/reward-view intersection it runs the exact production Dart scorer and
the exact VeRPO advantage function, then records restart-safe, hash-chained
task results and a sealed aggregate summary.

The result is deliberately labelled a proxy.  The source harvest used
``top_p=0.95`` and typed-direct update 58, whereas the currently implemented
native VeRPO sampler uses ``top_p=1.0`` and is not wired to the typed lineage.
The stored candidates also lack on-policy token IDs/log-probabilities and must
never be used for a policy update.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


AUDIT_CONTRACT_SCHEMA = "t5gemma2-typed-proxy-reward-audit-contract-v1"
AUDIT_JOURNAL_SCHEMA = "t5gemma2-typed-proxy-reward-audit-journal-v1"
AUDIT_SUMMARY_SCHEMA = "t5gemma2-typed-proxy-reward-audit-summary-v1"
SELECTION_SCHEMA = "t5gemma2-typed-proxy-reward-audit-selection-v1"

SOURCE_RUN_SCHEMA = "t5gemma2-typed-local-direct-harvest-run-v1"
SOURCE_JOURNAL_SCHEMA = "t5gemma2-typed-local-direct-harvest-journal-v1"
SOURCE_TRAINING_SCHEMA = "t5gemma2-typed-direct-rs-sft-run-v1"
FEEDBACK_SPLIT_SCHEMA = "task-bound-expect-half-split-v1"

EXPECTED_HARVEST_JOURNAL_SHA256 = (
    "ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc"
)
EXPECTED_HARVEST_CHAIN_HEAD_SHA256 = (
    "a8b41e0855cb73874c05ec0f57ca29c43449756b92a7cf7ccc034c4351d22a57"
)
EXPECTED_FEEDBACK_SHA256 = (
    "14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c"
)
EXPECTED_HARVEST_TASKS = 2550
EXPECTED_FEEDBACK_TASKS = 2386
EXPECTED_INTERSECTION_TASKS = 2161
EXPECTED_GROUP_SIZE = 4
EXPECTED_SAMPLE_SIZE = 150
EXPECTED_SAMPLE_SEED = 42
EXPECTED_TIMEOUT = 30
EXPECTED_STABILITY_RUNS = 1
EXPECTED_ALPHA = 2.0
EXPECTED_LOCAL_WEIGHT = 1.0
EXPECTED_COMPILE_WEIGHT = 0.25
ZERO_EPSILON = 1e-12
P_NEW_TARGET = 0.10
P_NEW_MINIMUM = 0.05
R_UNIQUE_TARGET = 0.20
R_UNIQUE_MINIMUM = 0.10
CONFIDENCE_LEVEL = 0.95
WILSON_Z_95 = 1.959963984540054
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
RESIDUAL_SQUARED_TOLERANCE_SCALE = 1e-10

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_SOURCE_PAYLOAD_KEYS = frozenset(
    {
        "tests",
        "acceptance_tests",
        "feedback_tests",
        "reward_holdback_tests",
        "private_tests",
        "diagnostic",
        "private_diagnostic",
        "holdback_diagnostic",
    }
)

ScoreFn = Callable[..., dict[str, Any]]
RewardFn = Callable[..., dict[str, list[float]]]
SplitFn = Callable[[str], list[str]]


@dataclass(frozen=True)
class CandidateGroup:
    task_id: str
    source_task_position: int
    source_sha256: str
    typed_contract_sha256: str
    candidate_codes: tuple[str, ...]
    candidate_code_sha256s: tuple[str, ...]
    feedback_tests: str
    visible_test_cases: int


@dataclass(frozen=True)
class AuditInputs:
    groups: tuple[CandidateGroup, ...]
    harvest_record: dict[str, Any]
    feedback_record: dict[str, Any]
    harvest_contract: dict[str, Any]
    intersection_tasks: int
    intersection_task_ids_sha256: str
    sample_seed: int


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label} has a blank line at {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(value)
    return rows


def _validate_source_contract(contract: Mapping[str, Any]) -> None:
    checkpoint = contract.get("checkpoint")
    sampling = contract.get("sampling")
    input_contract = contract.get("input")
    if (
        contract.get("schema") != SOURCE_RUN_SCHEMA
        or contract.get("checkpoint_stage") != "typed_direct"
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("checkpoint_update") != 58
        or checkpoint.get("training_stage_schema") != SOURCE_TRAINING_SCHEMA
        or not isinstance(sampling, Mapping)
        or sampling.get("samples_per_task") != EXPECTED_GROUP_SIZE
        or sampling.get("generation_batch_size") != EXPECTED_GROUP_SIZE
        or sampling.get("max_repair_parents") != 0
        or sampling.get("repair_samples") != 0
        or sampling.get("temperature") != 0.8
        or sampling.get("top_p") != 0.95
        or sampling.get("max_source_tokens") != 32768
        or sampling.get("max_new_tokens") != 4096
        or not isinstance(input_contract, Mapping)
        or input_contract.get("heldout_175_opened") is not False
        or input_contract.get("complete_acceptance_model_visible") is not False
        or contract.get("frontier_api_calls") is not False
        or contract.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
    ):
        raise ValueError("typed pass-1 k=4 harvest contract differs")


def _validate_source_journal(
    events: Sequence[Mapping[str, Any]], *, expected_tasks: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if len(events) != expected_tasks + 2:
        raise ValueError("typed harvest journal event count differs")
    header = events[0]
    contract = header.get("contract")
    if (
        header.get("event") != "header"
        or header.get("schema") != SOURCE_JOURNAL_SCHEMA
        or not isinstance(contract, dict)
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("typed harvest journal header differs")
    _validate_source_contract(contract)

    terminals: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for position, event in enumerate(events[1:-1]):
        task_id = str(event.get("task_id") or "")
        candidates = event.get("base_candidates")
        if (
            event.get("event") != "task_terminal"
            or event.get("schema") != SOURCE_JOURNAL_SCHEMA
            or event.get("task_position") != position
            or not task_id
            or task_id in terminals
            or not _SHA256_RE.fullmatch(str(event.get("source_sha256") or ""))
            or not _SHA256_RE.fullmatch(
                str(event.get("typed_contract_sha256") or "")
            )
            or event.get("repair_groups") != []
            or event.get("all_generation_completed_before_private_gate") is not True
            or event.get("private_feedback_serialized_to_model") is not False
            or event.get("private_failure_triggers_generation") is not False
            or event.get("binary_field_semantics")
            != "complete_train_acceptance_private_gate"
            or not isinstance(candidates, list)
            or len(candidates) != EXPECTED_GROUP_SIZE
        ):
            raise ValueError(f"typed harvest terminal {position} differs")
        for sample_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"typed harvest candidate {position}:{sample_index} malformed")
            code = candidate.get("code")
            code_sha256 = str(candidate.get("code_sha256") or "")
            visible = candidate.get("visible")
            if (
                not isinstance(code, str)
                or candidate.get("sample_index") != sample_index
                or candidate.get("origin") != "local_student_direct"
                or _sha256_text(code) != code_sha256
                or not isinstance(visible, Mapping)
                or type(visible.get("compiled")) is not bool
                or type(visible.get("passed")) is not bool
                or (visible.get("passed") and not visible.get("compiled"))
                or any(key in candidate for key in _FORBIDDEN_SOURCE_PAYLOAD_KEYS)
            ):
                raise ValueError(
                    f"typed harvest candidate {position}:{sample_index} differs"
                )
        terminals[task_id] = dict(event)
        ordered_ids.append(task_id)

    complete = events[-1]
    if (
        complete.get("event") != "complete"
        or complete.get("schema") != SOURCE_JOURNAL_SCHEMA
        or complete.get("tasks") != expected_tasks
        or complete.get("terminal_task_ids_sha256")
        != canonical_sha256(ordered_ids)
    ):
        raise ValueError("typed harvest completion differs")
    return dict(contract), terminals


def deterministic_sample(
    task_ids: Sequence[str], *, sample_size: int, seed: int
) -> list[str]:
    if sample_size <= 0 or sample_size > len(task_ids):
        raise ValueError("audit sample size is outside the intersection")
    if len(task_ids) != len(set(task_ids)) or any(not value for value in task_ids):
        raise ValueError("audit intersection task IDs are invalid")
    ranked = sorted(
        task_ids,
        key=lambda task_id: (
            canonical_sha256(
                {
                    "schema": SELECTION_SCHEMA,
                    "seed": int(seed),
                    "task_id": task_id,
                }
            ),
            task_id,
        ),
    )
    return ranked[:sample_size]


def load_audit_inputs(
    harvest_journal: str | Path,
    feedback_jsonl: str | Path,
    *,
    split_fn: SplitFn,
    expected_harvest_journal_sha256: str,
    expected_harvest_chain_head_sha256: str,
    expected_feedback_sha256: str,
    expected_harvest_tasks: int = EXPECTED_HARVEST_TASKS,
    expected_feedback_tasks: int = EXPECTED_FEEDBACK_TASKS,
    expected_intersection_tasks: int = EXPECTED_INTERSECTION_TASKS,
    sample_size: int = EXPECTED_SAMPLE_SIZE,
    sample_seed: int = EXPECTED_SAMPLE_SEED,
) -> AuditInputs:
    harvest_path = Path(harvest_journal).expanduser().resolve()
    feedback_path = Path(feedback_jsonl).expanduser().resolve()
    harvest_sha = _require_sha256(
        expected_harvest_journal_sha256, "expected harvest journal SHA-256"
    )
    harvest_head_sha = _require_sha256(
        expected_harvest_chain_head_sha256,
        "expected harvest chain-head SHA-256",
    )
    feedback_sha = _require_sha256(
        expected_feedback_sha256, "expected feedback SHA-256"
    )
    if sha256_file(harvest_path) != harvest_sha:
        raise ValueError("typed harvest journal SHA-256 differs")
    harvest_head_path = Path(str(harvest_path) + ".chain-head.json")
    if sha256_file(harvest_head_path) != harvest_head_sha:
        raise ValueError("typed harvest journal chain-head SHA-256 differs")
    if sha256_file(feedback_path) != feedback_sha:
        raise ValueError("visible feedback JSONL SHA-256 differs")

    events = load_journal(harvest_path)
    harvest_contract, terminal_by_id = _validate_source_journal(
        events, expected_tasks=expected_harvest_tasks
    )
    feedback_rows = _read_jsonl(feedback_path, "visible feedback JSONL")
    if len(feedback_rows) != expected_feedback_tasks:
        raise ValueError("visible feedback row count differs")
    feedback_by_id: dict[str, tuple[str, int]] = {}
    for row_index, row in enumerate(feedback_rows):
        task_id = str(row.get("task_id") or "")
        tests = row.get("feedback_tests")
        if (
            not task_id
            or task_id in feedback_by_id
            or not isinstance(tests, str)
            or not tests.strip()
            or row.get("verpo_feedback_split_schema") != FEEDBACK_SPLIT_SCHEMA
        ):
            raise ValueError(f"visible feedback row {row_index} differs")
        variants = split_fn(tests)
        if not variants or any(not isinstance(value, str) or not value for value in variants):
            raise ValueError(f"visible feedback row {row_index} has no valid cases")
        feedback_by_id[task_id] = (tests, len(variants))

    intersection = sorted(set(terminal_by_id).intersection(feedback_by_id))
    if len(intersection) != expected_intersection_tasks:
        raise ValueError("typed harvest/feedback intersection count differs")
    selected = deterministic_sample(
        intersection, sample_size=sample_size, seed=sample_seed
    )
    groups: list[CandidateGroup] = []
    for task_id in selected:
        terminal = terminal_by_id[task_id]
        candidates = terminal["base_candidates"]
        tests, test_cases = feedback_by_id[task_id]
        groups.append(
            CandidateGroup(
                task_id=task_id,
                source_task_position=int(terminal["task_position"]),
                source_sha256=str(terminal["source_sha256"]),
                typed_contract_sha256=str(terminal["typed_contract_sha256"]),
                candidate_codes=tuple(str(value["code"]) for value in candidates),
                candidate_code_sha256s=tuple(
                    str(value["code_sha256"]) for value in candidates
                ),
                feedback_tests=tests,
                visible_test_cases=test_cases,
            )
        )
    selected_sources = [group.source_sha256 for group in groups]
    if len(selected_sources) != len(set(selected_sources)):
        raise ValueError("deterministic proxy sample has duplicate source SHA-256s")

    feedback_record = {
        "path": str(feedback_path),
        "sha256": feedback_sha,
        "rows": len(feedback_rows),
    }
    return AuditInputs(
        groups=tuple(groups),
        harvest_record=journal_record(harvest_path),
        feedback_record=feedback_record,
        harvest_contract=harvest_contract,
        intersection_tasks=len(intersection),
        intersection_task_ids_sha256=canonical_sha256(intersection),
        sample_seed=int(sample_seed),
    )


def build_audit_contract(
    inputs: AuditInputs,
    *,
    production_code: Mapping[str, Any],
    dart_bin: str,
    timeout: int = EXPECTED_TIMEOUT,
    stability_runs: int = EXPECTED_STABILITY_RUNS,
) -> dict[str, Any]:
    selected_ids = [group.task_id for group in inputs.groups]
    selected_projection = [
        {
            "position": position,
            "task_id": group.task_id,
            "source_sha256": group.source_sha256,
            "candidate_code_sha256s": list(group.candidate_code_sha256s),
        }
        for position, group in enumerate(inputs.groups)
    ]
    return {
        "schema": AUDIT_CONTRACT_SCHEMA,
        "execution": {
            "model_loaded": False,
            "gpu_used": False,
            "cuda_visible_devices": "-1",
            "dart_bin": str(Path(dart_bin).expanduser().resolve()),
            "timeout_seconds": int(timeout),
            "stability_runs": int(stability_runs),
        },
        "source": {
            "harvest_journal": inputs.harvest_record,
            "visible_feedback": inputs.feedback_record,
            "harvest_checkpoint": inputs.harvest_contract["checkpoint"],
            "harvest_sampling": inputs.harvest_contract["sampling"],
            "intersection_tasks": inputs.intersection_tasks,
            "intersection_task_ids_sha256": inputs.intersection_task_ids_sha256,
        },
        "selection": {
            "schema": SELECTION_SCHEMA,
            "seed": inputs.sample_seed,
            "sample_size": len(inputs.groups),
            "algorithm": "ascending_sha256_of_schema_seed_task_id_then_task_id",
            "ordered_task_ids_sha256": canonical_sha256(selected_ids),
            "unique_source_sha256s": len(
                {group.source_sha256 for group in inputs.groups}
            ),
            "ordered_task_candidate_hash_seal_sha256": canonical_sha256(
                selected_projection
            ),
        },
        "reward": {
            "scorer": "production_score_dart_candidate",
            "advantage": "production_verpo_execution_compile_advantages",
            "alpha": EXPECTED_ALPHA,
            "local_weight": EXPECTED_LOCAL_WEIGHT,
            "compile_weight": EXPECTED_COMPILE_WEIGHT,
            "components_independently_mean_centered": True,
            "diagnostics_persisted": False,
        },
        "decision_preregistration": {
            "schema": "t5gemma2-typed-proxy-reward-audit-decision-v1",
            "p_new": {
                "definition": (
                    "fraction of groups with active local advantage while both "
                    "global-full-pass and compile advantages are flat"
                ),
                "target": P_NEW_TARGET,
                "minimum": P_NEW_MINIMUM,
                "interval": "Wilson two-sided 95%",
                "z": WILSON_Z_95,
            },
            "r_unique": {
                "definition": (
                    "sum(||L-Proj_span(C,P)(L)||^2) / "
                    "sum(||P+0.25C||^2+||L||^2)"
                ),
                "target": R_UNIQUE_TARGET,
                "minimum": R_UNIQUE_MINIMUM,
                "residual_squared_tolerance": (
                    "1e-10 * max(1, ||L||^2)"
                ),
                "interval": "task bootstrap percentile two-sided 95%",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "per_metric_rule": {
                "GO": "point_estimate >= target AND lower_95_ci >= minimum",
                "STOP": "upper_95_ci < target",
                "HOLD": "otherwise",
            },
            "intersection_rule": {
                "GO": "both p_new and r_unique are GO",
                "STOP": "either p_new or r_unique is STOP",
                "HOLD": "otherwise",
            },
        },
        "production_code": dict(production_code),
        "interpretation": {
            "proxy_only": True,
            "not_eligible_for_on_policy_update": True,
            "mismatches_to_current_native_verpo": [
                "typed_direct_update58_not_eventual_promoted_checkpoint",
                "harvest_top_p_0.95_vs_native_verpo_top_p_1.0",
                "stored_candidates_have_no_action_token_ids_or_saved_logprobs",
            ],
        },
    }


def _clean_detail(detail: Mapping[str, Any], expected_tests: int) -> dict[str, Any]:
    compiled = detail.get("compiled")
    full_pass = detail.get("full_pass")
    test_passes = detail.get("test_passes")
    if (
        type(compiled) is not bool
        or type(full_pass) is not bool
        or not isinstance(test_passes, list)
        or len(test_passes) != expected_tests
        or any(type(value) is not bool for value in test_passes)
        or (full_pass and (not compiled or not all(test_passes)))
    ):
        raise ValueError("production scorer returned inconsistent evidence")
    return {
        "compiled": compiled,
        "full_pass": full_pass,
        "test_passes": list(test_passes),
    }


def _validate_reward(
    reward: Mapping[str, Any], *, group_size: int
) -> dict[str, list[float]]:
    required = (
        "global_rewards",
        "local_rewards",
        "compile_rewards",
        "global_advantages",
        "local_advantages",
        "compile_advantages",
        "unified_advantages",
    )
    normalized: dict[str, list[float]] = {}
    if set(reward) != set(required):
        raise ValueError("production reward returned unexpected components")
    for key in required:
        values = reward.get(key)
        if (
            not isinstance(values, list)
            or len(values) != group_size
            or any(not isinstance(value, (int, float)) for value in values)
            or any(not math.isfinite(float(value)) for value in values)
        ):
            raise ValueError(f"production reward component {key} is invalid")
        normalized[key] = [float(value) for value in values]
    return normalized


def _score_group(
    item: tuple[int, CandidateGroup],
    *,
    score_fn: ScoreFn,
    reward_fn: RewardFn,
    timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    position, group = item
    details = [
        _clean_detail(
            score_fn(
                code,
                group.feedback_tests,
                f"typed-proxy-{position:04d}-{sample_index}",
                timeout=timeout,
                stability_runs=stability_runs,
            ),
            group.visible_test_cases,
        )
        for sample_index, code in enumerate(group.candidate_codes)
    ]
    reward = _validate_reward(
        reward_fn(
            details,
            alpha=EXPECTED_ALPHA,
            local_weight=EXPECTED_LOCAL_WEIGHT,
            compile_weight=EXPECTED_COMPILE_WEIGHT,
        ),
        group_size=len(group.candidate_codes),
    )
    return {
        "schema": AUDIT_JOURNAL_SCHEMA,
        "event": "task_terminal",
        "task_position": position,
        "task_id": group.task_id,
        "source_task_position": group.source_task_position,
        "source_sha256": group.source_sha256,
        "typed_contract_sha256": group.typed_contract_sha256,
        "candidate_code_sha256s": list(group.candidate_code_sha256s),
        "visible_test_cases": group.visible_test_cases,
        "details": details,
        "reward": reward,
        "diagnostics_persisted": False,
        "model_or_gpu_used": False,
    }


def _same_float_lists(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_sha256(left) == canonical_sha256(right)


def _validate_output_journal(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    groups: Sequence[CandidateGroup],
    reward_fn: RewardFn,
) -> tuple[list[dict[str, Any]], bool]:
    if not events:
        return [], False
    header = events[0]
    if (
        header.get("schema") != AUDIT_JOURNAL_SCHEMA
        or header.get("event") != "header"
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("proxy reward audit journal header differs")
    terminals: list[dict[str, Any]] = []
    complete = False
    for event in events[1:]:
        if event.get("event") == "complete":
            if complete or len(terminals) != len(groups):
                raise ValueError("proxy reward audit completion is early/duplicate")
            if (
                event.get("schema") != AUDIT_JOURNAL_SCHEMA
                or event.get("tasks") != len(groups)
                or event.get("terminal_task_ids_sha256")
                != canonical_sha256([value["task_id"] for value in terminals])
                or event.get("terminal_results_sha256")
                != canonical_sha256(terminals)
            ):
                raise ValueError("proxy reward audit completion differs")
            complete = True
            continue
        if complete or event.get("event") != "task_terminal":
            raise ValueError("proxy reward audit event ordering differs")
        position = len(terminals)
        if position >= len(groups):
            raise ValueError("proxy reward audit has excess terminals")
        group = groups[position]
        details = event.get("details")
        reward = event.get("reward")
        if (
            event.get("schema") != AUDIT_JOURNAL_SCHEMA
            or event.get("task_position") != position
            or event.get("task_id") != group.task_id
            or event.get("source_task_position") != group.source_task_position
            or event.get("source_sha256") != group.source_sha256
            or event.get("typed_contract_sha256") != group.typed_contract_sha256
            or event.get("candidate_code_sha256s")
            != list(group.candidate_code_sha256s)
            or event.get("visible_test_cases") != group.visible_test_cases
            or event.get("diagnostics_persisted") is not False
            or event.get("model_or_gpu_used") is not False
            or not isinstance(details, list)
            or len(details) != EXPECTED_GROUP_SIZE
            or not isinstance(reward, Mapping)
        ):
            raise ValueError(f"proxy reward audit terminal {position} differs")
        cleaned = [
            _clean_detail(value, group.visible_test_cases)
            for value in details
            if isinstance(value, Mapping)
        ]
        if len(cleaned) != EXPECTED_GROUP_SIZE or cleaned != details:
            raise ValueError(f"proxy reward audit terminal {position} evidence differs")
        observed_reward = _validate_reward(reward, group_size=EXPECTED_GROUP_SIZE)
        expected_reward = _validate_reward(
            reward_fn(
                cleaned,
                alpha=EXPECTED_ALPHA,
                local_weight=EXPECTED_LOCAL_WEIGHT,
                compile_weight=EXPECTED_COMPILE_WEIGHT,
            ),
            group_size=EXPECTED_GROUP_SIZE,
        )
        if not _same_float_lists(observed_reward, expected_reward):
            raise ValueError(f"proxy reward audit terminal {position} reward differs")
        terminals.append(dict(event))
    return terminals, complete


def _active(values: Sequence[float]) -> bool:
    return any(abs(float(value)) > ZERO_EPSILON for value in values)


def _component_summary(
    terminals: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    groups = [list(value["reward"][key]) for value in terminals]
    group_active = sum(_active(values) for values in groups)
    trajectory_nonzero = sum(
        abs(float(value)) > ZERO_EPSILON for values in groups for value in values
    )
    trajectories = sum(len(values) for values in groups)
    return {
        "groups_active": group_active,
        "groups_total": len(groups),
        "group_active_fraction": group_active / len(groups) if groups else 0.0,
        "trajectories_nonzero": trajectory_nonzero,
        "trajectories_total": trajectories,
        "trajectory_nonzero_fraction": (
            trajectory_nonzero / trajectories if trajectories else 0.0
        ),
    }


def _orthogonal_residual_squared(
    value: Sequence[float], spans: Sequence[Sequence[float]]
) -> tuple[float, float]:
    vector = [float(item) for item in value]
    basis: list[list[float]] = []
    for raw in spans:
        candidate = [float(item) for item in raw]
        for unit in basis:
            coefficient = sum(left * right for left, right in zip(candidate, unit))
            candidate = [
                left - coefficient * right
                for left, right in zip(candidate, unit)
            ]
        norm = math.sqrt(sum(item * item for item in candidate))
        if norm > ZERO_EPSILON:
            basis.append([item / norm for item in candidate])
    residual = list(vector)
    for unit in basis:
        coefficient = sum(left * right for left, right in zip(residual, unit))
        residual = [
            left - coefficient * right for left, right in zip(residual, unit)
        ]
    return sum(item * item for item in residual), sum(item * item for item in vector)


def _squared_norm(value: Sequence[float]) -> float:
    return sum(float(item) * float(item) for item in value)


def _wilson_interval(successes: int, total: int) -> dict[str, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval counts are invalid")
    proportion = successes / total
    z2 = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile input is invalid")
    coordinate = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(coordinate))
    upper = int(math.ceil(coordinate))
    if lower == upper:
        return float(sorted_values[lower])
    weight = coordinate - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(
        sorted_values[upper]
    ) * weight


def _bootstrap_ratio_interval(
    numerators: Sequence[float], denominators: Sequence[float]
) -> dict[str, Any]:
    if (
        not numerators
        or len(numerators) != len(denominators)
        or any(value < 0.0 or not math.isfinite(value) for value in numerators)
        or any(value < 0.0 or not math.isfinite(value) for value in denominators)
    ):
        raise ValueError("bootstrap ratio contributions are invalid")
    randomizer = random.Random(BOOTSTRAP_SEED)
    count = len(numerators)
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        numerator = 0.0
        denominator = 0.0
        for _draw in range(count):
            index = randomizer.randrange(count)
            numerator += numerators[index]
            denominator += denominators[index]
        replicates.append(numerator / denominator if denominator > 0.0 else 0.0)
    replicates.sort()
    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "task_bootstrap_percentile",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "lower": _percentile(replicates, 0.025),
        "upper": _percentile(replicates, 0.975),
    }


def _metric_decision(
    estimate: float,
    interval: Mapping[str, float],
    *,
    target: float,
    minimum: float,
) -> str:
    if float(estimate) >= target and float(interval["lower"]) >= minimum:
        return "GO"
    if float(interval["upper"]) < target:
        return "STOP"
    return "HOLD"


def _binary_distribution(values: Sequence[bool]) -> str:
    count = sum(bool(value) for value in values)
    if count == 0:
        return "zero"
    if count == len(values):
        return "all"
    return "mixed"


def summarize_terminals(
    terminals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not terminals:
        raise ValueError("cannot summarize an empty proxy audit")
    candidate_count = 0
    compiled_count = 0
    full_pass_count = 0
    isolated_cases = 0
    isolated_passes = 0
    compile_groups = {"zero": 0, "mixed": 0, "all": 0}
    full_groups = {"zero": 0, "mixed": 0, "all": 0}
    zero_visible_test_groups = 0
    compile_homogeneous_groups = 0
    local_active_compile_homogeneous = 0
    local_active_binary_ties = 0
    local_active_groups = 0
    local_noncollinear_groups = 0
    residual_l2_sum = 0.0
    local_l2_sum = 0.0
    residual_squared_contributions: list[float] = []
    total_signal_squared_contributions: list[float] = []
    visible_test_counts: list[int] = []

    for terminal in terminals:
        details = terminal["details"]
        reward = terminal["reward"]
        compiled = [bool(value["compiled"]) for value in details]
        full = [bool(value["full_pass"]) for value in details]
        tests = [list(value["test_passes"]) for value in details]
        candidate_count += len(details)
        compiled_count += sum(compiled)
        full_pass_count += sum(full)
        isolated_cases += sum(len(value) for value in tests)
        isolated_passes += sum(sum(bool(item) for item in value) for value in tests)
        compile_groups[_binary_distribution(compiled)] += 1
        full_groups[_binary_distribution(full)] += 1
        zero_visible_test_groups += int(not any(any(value) for value in tests))
        visible_test_counts.append(int(terminal["visible_test_cases"]))

        local = reward["local_advantages"]
        global_advantage = reward["global_advantages"]
        compile_advantage = reward["compile_advantages"]
        local_is_active = _active(local)
        global_is_active = _active(global_advantage)
        compile_is_active = _active(compile_advantage)
        homogeneous = len(set(compiled)) == 1
        compile_homogeneous_groups += int(homogeneous)
        local_active_groups += int(local_is_active)
        local_active_compile_homogeneous += int(local_is_active and homogeneous)
        local_active_binary_ties += int(
            local_is_active and not global_is_active and not compile_is_active
        )
        residual_squared, local_squared = _orthogonal_residual_squared(
            local, [global_advantage, compile_advantage]
        )
        binary_combined = [
            float(pass_value) + EXPECTED_COMPILE_WEIGHT * float(compile_value)
            for pass_value, compile_value in zip(
                global_advantage, compile_advantage, strict=True
            )
        ]
        denominator_squared = _squared_norm(binary_combined) + local_squared
        residual_squared_contributions.append(residual_squared)
        total_signal_squared_contributions.append(denominator_squared)
        residual_l2_sum += math.sqrt(residual_squared)
        local_l2_sum += math.sqrt(local_squared)
        threshold = RESIDUAL_SQUARED_TOLERANCE_SCALE * max(1.0, local_squared)
        local_noncollinear_groups += int(
            local_is_active and residual_squared > threshold
        )

    groups_total = len(terminals)
    p_new = local_active_binary_ties / groups_total
    p_new_interval = _wilson_interval(local_active_binary_ties, groups_total)
    residual_squared_sum = sum(residual_squared_contributions)
    total_signal_squared_sum = sum(total_signal_squared_contributions)
    r_unique = (
        residual_squared_sum / total_signal_squared_sum
        if total_signal_squared_sum > 0.0
        else 0.0
    )
    r_unique_interval = _bootstrap_ratio_interval(
        residual_squared_contributions, total_signal_squared_contributions
    )
    p_new_decision = _metric_decision(
        p_new,
        p_new_interval,
        target=P_NEW_TARGET,
        minimum=P_NEW_MINIMUM,
    )
    r_unique_decision = _metric_decision(
        r_unique,
        r_unique_interval,
        target=R_UNIQUE_TARGET,
        minimum=R_UNIQUE_MINIMUM,
    )
    if "STOP" in (p_new_decision, r_unique_decision):
        overall_decision = "STOP"
    elif p_new_decision == r_unique_decision == "GO":
        overall_decision = "GO"
    else:
        overall_decision = "HOLD"

    return {
        "components": {
            "global": _component_summary(terminals, "global_advantages"),
            "local": _component_summary(terminals, "local_advantages"),
            "compile": _component_summary(terminals, "compile_advantages"),
            "unified": _component_summary(terminals, "unified_advantages"),
        },
        "candidate_execution": {
            "candidates": candidate_count,
            "compiled": compiled_count,
            "compile_rate": compiled_count / candidate_count,
            "full_pass": full_pass_count,
            "full_pass_rate": full_pass_count / candidate_count,
            "isolated_visible_test_cases": isolated_cases,
            "isolated_visible_test_passes": isolated_passes,
            "isolated_visible_test_pass_rate": (
                isolated_passes / isolated_cases if isolated_cases else 0.0
            ),
        },
        "group_execution": {
            "groups": groups_total,
            "compile": compile_groups,
            "full_pass": full_groups,
            "all_zero_visible_test_groups": zero_visible_test_groups,
            "all_zero_visible_test_group_fraction": (
                zero_visible_test_groups / groups_total
            ),
            "visible_test_cases_per_task": {
                "minimum": min(visible_test_counts),
                "maximum": max(visible_test_counts),
                "mean": sum(visible_test_counts) / groups_total,
            },
        },
        "local_signal_beyond_binary": {
            "definition": (
                "local advantage residual after Gram-Schmidt projection onto "
                "the centered global-full-pass and compile-advantage span"
            ),
            "compile_homogeneous_groups": compile_homogeneous_groups,
            "local_active_on_compile_homogeneous_groups": (
                local_active_compile_homogeneous
            ),
            "local_active_on_compile_homogeneous_fraction_of_all": (
                local_active_compile_homogeneous / groups_total
            ),
            "local_active_with_both_binary_components_tied": local_active_binary_ties,
            "local_active_with_both_binary_components_tied_fraction_of_all": (
                local_active_binary_ties / groups_total
            ),
            "local_active_groups": local_active_groups,
            "local_noncollinear_groups": local_noncollinear_groups,
            "local_noncollinear_fraction_of_all": (
                local_noncollinear_groups / groups_total
            ),
            "local_noncollinear_fraction_of_local_active": (
                local_noncollinear_groups / local_active_groups
                if local_active_groups
                else 0.0
            ),
            "sum_local_residual_l2": residual_l2_sum,
            "sum_local_l2": local_l2_sum,
            "sum_local_residual_squared": residual_squared_sum,
            "sum_total_signal_squared": total_signal_squared_sum,
        },
        "preregistered_statistics": {
            "p_new": {
                "successes": local_active_binary_ties,
                "tasks": groups_total,
                "estimate": p_new,
                "interval": p_new_interval,
                "target": P_NEW_TARGET,
                "minimum": P_NEW_MINIMUM,
                "decision": p_new_decision,
            },
            "r_unique": {
                "estimate": r_unique,
                "interval": r_unique_interval,
                "numerator_sum_residual_squared": residual_squared_sum,
                "denominator_sum_total_signal_squared": total_signal_squared_sum,
                "target": R_UNIQUE_TARGET,
                "minimum": R_UNIQUE_MINIMUM,
                "decision": r_unique_decision,
            },
            "overall_decision": overall_decision,
            "decision_rule": (
                "per metric GO iff point>=target and lower95>=minimum; "
                "STOP iff upper95<target; overall GO=intersection and "
                "STOP=either; else HOLD"
            ),
        },
    }


def run_audit(
    inputs: AuditInputs,
    *,
    contract: Mapping[str, Any],
    output_journal: str | Path,
    output_summary: str | Path,
    score_fn: ScoreFn,
    reward_fn: RewardFn,
    workers: int,
    timeout: int = EXPECTED_TIMEOUT,
    stability_runs: int = EXPECTED_STABILITY_RUNS,
    stop_after_new_tasks: int | None = None,
) -> dict[str, Any] | None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if timeout != EXPECTED_TIMEOUT or stability_runs != EXPECTED_STABILITY_RUNS:
        raise ValueError("proxy audit must use the production VeRPO reward settings")
    journal_path = Path(output_journal).expanduser().resolve()
    summary_path = Path(output_summary).expanduser().resolve()
    if summary_path.exists() and not journal_path.exists():
        raise ValueError("proxy summary exists without its journal")
    events = load_journal(journal_path)
    if not events:
        append_event(
            journal_path,
            {
                "schema": AUDIT_JOURNAL_SCHEMA,
                "event": "header",
                "contract": dict(contract),
                "contract_sha256": canonical_sha256(contract),
            },
        )
        events = load_journal(journal_path)
    terminals, complete = _validate_output_journal(
        events, contract=contract, groups=inputs.groups, reward_fn=reward_fn
    )
    if not complete:
        remaining = list(enumerate(inputs.groups))[len(terminals) :]
        if stop_after_new_tasks is not None:
            if stop_after_new_tasks < 0:
                raise ValueError("stop_after_new_tasks must be non-negative")
            remaining = remaining[:stop_after_new_tasks]
        if remaining:
            def execute(item: tuple[int, CandidateGroup]) -> dict[str, Any]:
                return _score_group(
                    item,
                    score_fn=score_fn,
                    reward_fn=reward_fn,
                    timeout=timeout,
                    stability_runs=stability_runs,
                )

            with ThreadPoolExecutor(max_workers=min(workers, len(remaining))) as pool:
                for completed, terminal in enumerate(pool.map(execute, remaining), 1):
                    append_event(journal_path, terminal)
                    absolute = len(terminals) + completed
                    if absolute % 10 == 0 or absolute == len(inputs.groups):
                        print(
                            f"typed proxy reward audit {absolute}/{len(inputs.groups)}",
                            flush=True,
                        )
        events = load_journal(journal_path)
        terminals, complete = _validate_output_journal(
            events, contract=contract, groups=inputs.groups, reward_fn=reward_fn
        )
        if not complete and len(terminals) == len(inputs.groups):
            append_event(
                journal_path,
                {
                    "schema": AUDIT_JOURNAL_SCHEMA,
                    "event": "complete",
                    "tasks": len(terminals),
                    "terminal_task_ids_sha256": canonical_sha256(
                        [value["task_id"] for value in terminals]
                    ),
                    "terminal_results_sha256": canonical_sha256(terminals),
                },
            )
            events = load_journal(journal_path)
            terminals, complete = _validate_output_journal(
                events,
                contract=contract,
                groups=inputs.groups,
                reward_fn=reward_fn,
            )
    if not complete:
        return None

    body = {
        "schema": AUDIT_SUMMARY_SCHEMA,
        "status": "complete",
        "contract_sha256": canonical_sha256(contract),
        "journal": journal_record(journal_path),
        "selection": dict(contract["selection"]),
        "metrics": summarize_terminals(terminals),
        "decision_preregistration": dict(contract["decision_preregistration"]),
        "interpretation": dict(contract["interpretation"]),
    }
    body["decision"] = body["metrics"]["preregistered_statistics"]
    summary = {**body, "summary_sha256": canonical_sha256(body)}
    require_exact_or_write(summary_path, summary)
    return summary


def load_production_components() -> tuple[ScoreFn, RewardFn, SplitFn, dict[str, Any]]:
    # CUDA is hidden before importing torch transitively through the trainer.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be -1 before production imports")
    from scripts.evaluation import graph_compile_at_k_antigravity as evaluator
    from scripts.training import t5gemma2_compiler_feedback_verpo as trainer
    from scripts.training import seq2seq_verpo_core as core

    evaluator.validate_dart_binary()
    try:
        import torch

        if torch.cuda.is_available():
            raise RuntimeError("GPU is visible in the CPU-only proxy audit")
    except ImportError:
        pass

    code: dict[str, Any] = {}
    audit_source = Path(__file__).resolve()
    code["proxy_audit"] = {
        "path": str(audit_source),
        "sha256": sha256_file(audit_source),
    }
    for name, value in {
        "score_dart_candidate": trainer.score_dart_candidate,
        "verpo_execution_compile_advantages": (
            core.verpo_execution_compile_advantages
        ),
        "dart_evaluator": evaluator.evaluate_dart_jit_tests_detail,
    }.items():
        source = Path(str(inspect.getsourcefile(value))).resolve()
        code[name] = {"path": str(source), "sha256": sha256_file(source)}
    return (
        trainer.score_dart_candidate,
        core.verpo_execution_compile_advantages,
        trainer.split_visible_expect_harnesses,
        code,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harvest_journal", required=True)
    parser.add_argument("--feedback_jsonl", required=True)
    parser.add_argument("--output_journal", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument(
        "--dart_bin",
        default="/workspace/tools/dart-3.12.2/usr/lib/dart/bin/dart",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--expected_harvest_journal_sha256",
        default=EXPECTED_HARVEST_JOURNAL_SHA256,
    )
    parser.add_argument(
        "--expected_harvest_chain_head_sha256",
        default=EXPECTED_HARVEST_CHAIN_HEAD_SHA256,
    )
    parser.add_argument(
        "--expected_feedback_sha256", default=EXPECTED_FEEDBACK_SHA256
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in [1, 32]")

    dart_bin = Path(args.dart_bin).expanduser().resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["DART_BIN"] = str(dart_bin)
    os.environ["PATH"] = str(dart_bin.parent) + os.pathsep + os.environ.get("PATH", "")
    score_fn, reward_fn, split_fn, production_code = load_production_components()
    inputs = load_audit_inputs(
        args.harvest_journal,
        args.feedback_jsonl,
        split_fn=split_fn,
        expected_harvest_journal_sha256=args.expected_harvest_journal_sha256,
        expected_harvest_chain_head_sha256=(
            args.expected_harvest_chain_head_sha256
        ),
        expected_feedback_sha256=args.expected_feedback_sha256,
    )
    contract = build_audit_contract(
        inputs, production_code=production_code, dart_bin=str(dart_bin)
    )
    summary = run_audit(
        inputs,
        contract=contract,
        output_journal=args.output_journal,
        output_summary=args.output_summary,
        score_fn=score_fn,
        reward_fn=reward_fn,
        workers=args.workers,
    )
    if summary is None:
        raise RuntimeError("proxy reward audit stopped before completion")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
