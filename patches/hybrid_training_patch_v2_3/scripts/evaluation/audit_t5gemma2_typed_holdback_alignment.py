#!/usr/bin/env python3
"""One-shot private-holdback alignment audit for the typed reward proxy.

The model is never loaded.  The exact 600 candidates and their visible reward
components come from the sealed 150-task proxy audit.  Candidate programs are
read only in memory from the earlier harvest and scored once against the
private complement.  Neither private tests, candidate code, nor diagnostics
are written.  The restart journal is private evidence; the published summary
contains aggregate statistics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.evaluation import audit_t5gemma2_typed_proxy_reward_surface as proxy
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.preprocessing import build_verpo_feedback_view as feedback_builder


CONTRACT_SCHEMA = "t5gemma2-typed-holdback-alignment-contract-v1"
JOURNAL_SCHEMA = "t5gemma2-typed-holdback-alignment-private-journal-v1"
SUMMARY_SCHEMA = "t5gemma2-typed-holdback-alignment-summary-v1"
PRIVATE_SPLIT_SCHEMA = "task-bound-expect-half-split-v1"

EXPECTED_PROXY_JOURNAL_SHA256 = (
    "b63250b1db1ca53fdf033cd3935824b4a96a76c37ef4f1f390dabd72370be1f4"
)
EXPECTED_PROXY_CHAIN_HEAD_SHA256 = (
    "76b4bcc98ef7f16fd57a76d0501a7c91617c9ac80d1506b2beeb8763a2ab8172"
)
EXPECTED_PROXY_SUMMARY_SHA256 = (
    "b0d73acb0391adea3844afa6f36589f4035e5fa4e73751f25836b318f43d9435"
)
EXPECTED_HOLDBACK_SHA256 = (
    "dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f"
)
EXPECTED_FEEDBACK_BUILD_SHA256 = (
    "1a73ee3df03d1fda97d819e8536ab42435dfa1cbc802335987e21ed48cd196e2"
)
EXPECTED_PRIVATE_ROWS = 2386
EXPECTED_SELECTED_GROUPS = 150
EXPECTED_CANDIDATES_PER_GROUP = 4
EXPECTED_P_NEW_GROUPS = 61
EXPECTED_TIMEOUT = 30
EXPECTED_STABILITY_RUNS = 1
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
ZERO_EPSILON = 1e-12

UPLIFT_TARGET = 0.02
RANK_TARGET = 0.55
RANK_NULL = 0.50

EXPECTED_BUILD_ACCOUNTING = {
    "parent_rows": 2774,
    "eligible_rows": 2386,
    "excluded_rows": 388,
    "source_expect_cases": 22051,
    "visible_expect_cases": 10531,
    "holdback_expect_cases": 11520,
    "odd_case_tasks": 989,
}
EXPECTED_BUILD_INVARIANTS = {
    "parent_is_sealed_executable_view": True,
    "parent_is_exact_safe1578": False,
    "dev175_bytes_opened": False,
    "acceptance_tests_read_or_used": False,
    "rollout_contains_no_acceptance_or_holdback_fields": True,
    "deepseek_f2_contains_no_tests": True,
    "visible_and_holdback_nonempty_for_every_eligible_task": True,
    "all_expect_cases_accounted_exactly_once": True,
    "compact_model_binding_fields_unchanged": True,
    "holdback_is_not_a_trainer_input": True,
}
EXPECTED_BUILD_OUTPUT_NAMES = {
    "rollout": "verpo_rollout_feedback.jsonl",
    "seal": "verpo_rollout_feedback.seal.json",
    "f2": "verpo_teacher_f2.jsonl",
    "f2_manifest": "verpo_teacher_f2.jsonl.manifest.json",
    "reward_holdback_private": "reward_holdback.private.jsonl",
    "excluded": "excluded_feedback_tasks.jsonl",
}
EXPECTED_ELIGIBLE_TASK_IDS_SHA256 = (
    "6916b6883ebff97a909a87db7a50f0de818fff3038e1b5b0495fa9adb79a8eeb"
)
EXPECTED_EXCLUDED_TASK_IDS_SHA256 = (
    "fc8f5e7bd74c9ca617abff13da3c21f2523613ef3f7d25af5e3bd7e5adf94b0d"
)
PRIVATE_ROW_KEYS = frozenset(
    {
        "task_id",
        "schema",
        "tests_sha256",
        "case_count",
        "visible_count",
        "holdback_count",
        "visible_case_indices",
        "holdback_case_indices",
        "feedback_tests",
        "reward_holdback_tests",
    }
)

ScoreFn = Callable[..., dict[str, Any]]
RewardFn = Callable[..., dict[str, list[float]]]
SplitFn = Callable[[str], list[str]]


@dataclass(frozen=True)
class AlignmentGroup:
    task_id: str
    task_position: int
    source_sha256: str
    typed_contract_sha256: str
    candidate_codes: tuple[str, ...]
    candidate_code_sha256s: tuple[str, ...]
    visible_local_rewards: tuple[float, ...]
    visible_unified_advantages: tuple[float, ...]
    p_new: bool
    holdback_tests: str
    holdback_cases: int


@dataclass(frozen=True)
class AlignmentInputs:
    groups: tuple[AlignmentGroup, ...]
    source_inputs: proxy.AuditInputs
    proxy_contract: dict[str, Any]
    proxy_journal_record: dict[str, Any]
    proxy_summary_record: dict[str, Any]
    holdback_record: dict[str, Any]
    feedback_build_record: dict[str, Any]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable/malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _pin(path: str | Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if sha256_file(resolved) != str(expected_sha256).lower():
        raise ValueError(f"{label} SHA-256 differs")
    return resolved


def _enforce_private_journal_permissions(path: Path) -> None:
    for private_path in (path, Path(str(path) + ".chain-head.json")):
        if not private_path.exists():
            continue
        os.chmod(private_path, 0o600)
        if os.name != "nt" and stat.S_IMODE(private_path.stat().st_mode) != 0o600:
            raise PermissionError(f"private audit permissions differ: {private_path}")


def _append_private_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    value = append_event(path, event)
    _enforce_private_journal_permissions(path)
    return value


def _validate_proxy_summary(
    summary: Mapping[str, Any],
    *,
    proxy_journal_record: Mapping[str, Any],
    proxy_contract: Mapping[str, Any],
) -> None:
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    decision = summary.get("decision")
    selection = summary.get("selection")
    journal = summary.get("journal")
    if (
        summary.get("schema") != proxy.AUDIT_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("summary_sha256") != canonical_sha256(body)
        or summary.get("contract_sha256") != canonical_sha256(proxy_contract)
        or not isinstance(decision, Mapping)
        or decision.get("overall_decision") != "GO"
        or not isinstance(decision.get("p_new"), Mapping)
        or decision["p_new"].get("successes") != EXPECTED_P_NEW_GROUPS
        or not isinstance(selection, Mapping)
        or selection != proxy_contract.get("selection")
        or not isinstance(journal, Mapping)
        or journal.get("sha256") != proxy_journal_record.get("sha256")
        or journal.get("chain_head_sha256")
        != proxy_journal_record.get("chain_head_sha256")
    ):
        raise ValueError("completed proxy summary/decision binding differs")


def _actual_file_record(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": size,
        "bytes": size,
    }


def _validate_feedback_build_report(
    report: Mapping[str, Any],
    *,
    feedback_build_path: Path,
    feedback_path: Path,
    holdback_path: Path,
) -> None:
    outputs = report.get("outputs")
    predeclared = report.get("predeclared_expectation")
    digests = report.get("digests")
    expected_split_policy = {
        "schema": PRIVATE_SPLIT_SCHEMA,
        "seed": 42,
        "source_field": "tests",
        "acceptance_tests_used": False,
        "minimum_cases": 2,
        "even_policy": "N/2 visible; N/2 holdback",
        "odd_policy": "floor(N/2) visible; ceil(N/2) holdback",
        "single_case_policy": "exclude",
        "no_expect_policy": "exclude",
        "malformed_policy": "exclude",
        "selection": "lowest task-bound SHA-256 ranks become visible",
    }
    if (
        report.get("schema") != "verpo-train-feedback-view-v1"
        or report.get("status") != "complete"
        or report.get("split_policy") != expected_split_policy
        or report.get("accounting") != EXPECTED_BUILD_ACCOUNTING
        or report.get("invariants") != EXPECTED_BUILD_INVARIANTS
        or not isinstance(predeclared, Mapping)
        or predeclared.get("accounting") != EXPECTED_BUILD_ACCOUNTING
        or predeclared.get("eligible_task_ids_sha256")
        != EXPECTED_ELIGIBLE_TASK_IDS_SHA256
        or predeclared.get("excluded_task_ids_sha256")
        != EXPECTED_EXCLUDED_TASK_IDS_SHA256
        or digests
        != {
            "eligible_task_ids_sha256": EXPECTED_ELIGIBLE_TASK_IDS_SHA256,
            "excluded_task_ids_sha256": EXPECTED_EXCLUDED_TASK_IDS_SHA256,
            "script_sha256": sha256_file(Path(feedback_builder.__file__).resolve()),
        }
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(EXPECTED_BUILD_OUTPUT_NAMES)
    ):
        raise ValueError("private holdback build accounting/invariants differ")

    root = feedback_build_path.parent.resolve()
    for key, basename in EXPECTED_BUILD_OUTPUT_NAMES.items():
        record = outputs.get(key)
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
            "bytes",
        }:
            raise ValueError(f"feedback build output record {key} differs")
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            path.parent != root
            or path.name != basename
            or not path.is_file()
            or dict(record) != _actual_file_record(path)
        ):
            raise ValueError(f"feedback build output bytes {key} differ")
    if (
        Path(str(outputs["rollout"]["path"])).resolve() != feedback_path.resolve()
        or Path(str(outputs["reward_holdback_private"]["path"])).resolve()
        != holdback_path.resolve()
    ):
        raise ValueError("feedback/holdback paths differ from build outputs")


def _reconstruct_complementary_harness(visible: str, holdback: str) -> str:
    if len(visible) != len(holdback):
        raise ValueError("complementary harness lengths differ")
    rebuilt: list[str] = []
    for left, right in zip(visible, holdback, strict=True):
        if left == right:
            rebuilt.append(left)
        elif left == " ":
            rebuilt.append(right)
        elif right == " ":
            rebuilt.append(left)
        else:
            raise ValueError("complementary harness masks conflict")
    return "".join(rebuilt)


def _validate_private_row(
    visible: Mapping[str, Any],
    private: Mapping[str, Any],
    *,
    position: int,
    split_fn: SplitFn,
) -> tuple[str, str, int, int]:
    task_id = str(visible.get("task_id") or "")
    visible_tests = visible.get("feedback_tests")
    holdback = private.get("reward_holdback_tests")
    visible_indices = private.get("visible_case_indices")
    holdback_indices = private.get("holdback_case_indices")
    visible_count = private.get("visible_count")
    holdback_count = private.get("holdback_count")
    case_count = private.get("case_count")
    tests_sha256 = str(private.get("tests_sha256") or "")
    valid_visible_indices = (
        isinstance(visible_indices, list)
        and all(type(value) is int for value in visible_indices)
        and visible_indices == sorted(visible_indices)
        and len(visible_indices) == len(set(visible_indices))
    )
    valid_holdback_indices = (
        isinstance(holdback_indices, list)
        and all(type(value) is int for value in holdback_indices)
        and holdback_indices == sorted(holdback_indices)
        and len(holdback_indices) == len(set(holdback_indices))
    )
    if (
        set(private) != PRIVATE_ROW_KEYS
        or not task_id
        or private.get("schema") != PRIVATE_SPLIT_SCHEMA
        or private.get("task_id") != task_id
        or private.get("feedback_tests") != visible_tests
        or visible.get("verpo_feedback_split_schema") != PRIVATE_SPLIT_SCHEMA
        or not isinstance(visible_tests, str)
        or not visible_tests.strip()
        or not isinstance(holdback, str)
        or not holdback.strip()
        or len(tests_sha256) != 64
        or any(value not in "0123456789abcdef" for value in tests_sha256)
        or type(visible_count) is not int
        or type(holdback_count) is not int
        or type(case_count) is not int
        or case_count < 2
        or visible_count != case_count // 2
        or holdback_count != case_count - visible_count
        or not valid_visible_indices
        or not valid_holdback_indices
        or len(visible_indices) != visible_count
        or len(holdback_indices) != holdback_count
        or set(visible_indices) & set(holdback_indices)
        or set(visible_indices) | set(holdback_indices) != set(range(case_count))
    ):
        raise ValueError(f"private holdback row {position} differs")

    binding = feedback_builder.stable_sha256(
        {
            "tests_sha256": tests_sha256,
            "case_count": case_count,
            "visible_count": visible_count,
            "holdback_count": holdback_count,
            "visible_case_indices": visible_indices,
            "holdback_case_indices": holdback_indices,
        }
    )
    if visible.get("verpo_feedback_split_binding_sha256") != binding:
        raise ValueError(f"private split binding differs at row {position}")

    rebuilt = _reconstruct_complementary_harness(visible_tests, holdback)
    if _sha256_text(rebuilt) != tests_sha256:
        raise ValueError(f"complementary harness digest differs at row {position}")
    spans = feedback_builder.extract_expect_spans(rebuilt)
    if (
        len(spans) != case_count
        or feedback_builder.harness_with_cases(
            rebuilt, spans, set(visible_indices)
        )
        != visible_tests
        or feedback_builder.harness_with_cases(
            rebuilt, spans, set(holdback_indices)
        )
        != holdback
        or len(split_fn(visible_tests)) != visible_count
        or len(split_fn(holdback)) != holdback_count
    ):
        raise ValueError(f"complementary harness reconstruction differs at row {position}")
    return task_id, holdback, holdback_count, visible_count


def _load_private_rows(
    *,
    holdback_path: Path,
    feedback_path: Path,
    feedback_build_path: Path,
    split_fn: SplitFn,
) -> tuple[dict[str, tuple[str, int, int]], dict[str, Any]]:
    report = _read_json(feedback_build_path, "feedback build report")
    _validate_feedback_build_report(
        report,
        feedback_build_path=feedback_build_path,
        feedback_path=feedback_path,
        holdback_path=holdback_path,
    )

    visible_rows = proxy._read_jsonl(feedback_path, "visible feedback")  # noqa: SLF001
    private_rows = proxy._read_jsonl(holdback_path, "private holdback")  # noqa: SLF001
    if len(visible_rows) != len(private_rows) or len(private_rows) != EXPECTED_PRIVATE_ROWS:
        raise ValueError("visible/private feedback row counts differ")

    holdback_by_id: dict[str, tuple[str, int, int]] = {}
    for position, (visible, private) in enumerate(
        zip(visible_rows, private_rows, strict=True)
    ):
        task_id, holdback, holdback_count, visible_count = _validate_private_row(
            visible, private, position=position, split_fn=split_fn
        )
        if task_id in holdback_by_id:
            raise ValueError(f"duplicate private holdback task {task_id}")
        holdback_by_id[task_id] = (holdback, holdback_count, visible_count)
    return holdback_by_id, report


def load_alignment_inputs(
    *,
    harvest_journal: str | Path,
    feedback_jsonl: str | Path,
    proxy_journal: str | Path,
    proxy_summary: str | Path,
    holdback_jsonl: str | Path,
    feedback_build_report: str | Path,
    score_reward_fn: RewardFn,
    split_fn: SplitFn,
    expected_proxy_journal_sha256: str = EXPECTED_PROXY_JOURNAL_SHA256,
    expected_proxy_chain_head_sha256: str = EXPECTED_PROXY_CHAIN_HEAD_SHA256,
    expected_proxy_summary_sha256: str = EXPECTED_PROXY_SUMMARY_SHA256,
    expected_holdback_sha256: str = EXPECTED_HOLDBACK_SHA256,
    expected_feedback_build_sha256: str = EXPECTED_FEEDBACK_BUILD_SHA256,
) -> AlignmentInputs:
    source_inputs = proxy.load_audit_inputs(
        harvest_journal,
        feedback_jsonl,
        split_fn=split_fn,
        expected_harvest_journal_sha256=proxy.EXPECTED_HARVEST_JOURNAL_SHA256,
        expected_harvest_chain_head_sha256=proxy.EXPECTED_HARVEST_CHAIN_HEAD_SHA256,
        expected_feedback_sha256=proxy.EXPECTED_FEEDBACK_SHA256,
    )
    proxy_path = _pin(proxy_journal, expected_proxy_journal_sha256, "proxy journal")
    proxy_head = Path(str(proxy_path) + ".chain-head.json")
    _pin(proxy_head, expected_proxy_chain_head_sha256, "proxy chain head")
    proxy_events = load_journal(proxy_path)
    if not proxy_events or not isinstance(proxy_events[0].get("contract"), dict):
        raise ValueError("proxy journal lacks its run contract")
    proxy_contract = dict(proxy_events[0]["contract"])
    visible_terminals, complete = proxy._validate_output_journal(  # noqa: SLF001
        proxy_events,
        contract=proxy_contract,
        groups=source_inputs.groups,
        reward_fn=score_reward_fn,
    )
    if not complete or len(visible_terminals) != EXPECTED_SELECTED_GROUPS:
        raise ValueError("proxy reward journal is incomplete")
    proxy_record = journal_record(proxy_path)

    summary_path = _pin(proxy_summary, expected_proxy_summary_sha256, "proxy summary")
    summary = _read_json(summary_path, "proxy summary")
    _validate_proxy_summary(
        summary,
        proxy_journal_record=proxy_record,
        proxy_contract=proxy_contract,
    )

    holdback_path = _pin(holdback_jsonl, expected_holdback_sha256, "private holdback")
    build_path = _pin(
        feedback_build_report,
        expected_feedback_build_sha256,
        "feedback build report",
    )
    feedback_path = Path(feedback_jsonl).expanduser().resolve()
    holdback_by_id, _report = _load_private_rows(
        holdback_path=holdback_path,
        feedback_path=feedback_path,
        feedback_build_path=build_path,
        split_fn=split_fn,
    )

    groups: list[AlignmentGroup] = []
    for position, (source, visible) in enumerate(
        zip(source_inputs.groups, visible_terminals, strict=True)
    ):
        if source.task_id != visible.get("task_id"):
            raise ValueError("proxy source/reward task order differs")
        holdback = holdback_by_id.get(source.task_id)
        if holdback is None:
            raise ValueError(f"{source.task_id}: private complement is absent")
        if (
            source.visible_test_cases != holdback[2]
            or visible.get("visible_test_cases") != holdback[2]
        ):
            raise ValueError(
                f"{source.task_id}: proxy visible cases differ from private split"
            )
        reward = visible["reward"]
        local = tuple(float(value) for value in reward["local_rewards"])
        unified = tuple(float(value) for value in reward["unified_advantages"])
        p_new = (
            proxy._active(reward["local_advantages"])  # noqa: SLF001
            and not proxy._active(reward["global_advantages"])  # noqa: SLF001
            and not proxy._active(reward["compile_advantages"])  # noqa: SLF001
        )
        groups.append(
            AlignmentGroup(
                task_id=source.task_id,
                task_position=position,
                source_sha256=source.source_sha256,
                typed_contract_sha256=source.typed_contract_sha256,
                candidate_codes=source.candidate_codes,
                candidate_code_sha256s=source.candidate_code_sha256s,
                visible_local_rewards=local,
                visible_unified_advantages=unified,
                p_new=p_new,
                holdback_tests=holdback[0],
                holdback_cases=holdback[1],
            )
        )
    if len(groups) != EXPECTED_SELECTED_GROUPS:
        raise ValueError("holdback alignment selection is not 150 groups")
    if sum(group.p_new for group in groups) != EXPECTED_P_NEW_GROUPS:
        raise ValueError("holdback alignment p_new subset is not the sealed 61")

    return AlignmentInputs(
        groups=tuple(groups),
        source_inputs=source_inputs,
        proxy_contract=proxy_contract,
        proxy_journal_record=proxy_record,
        proxy_summary_record={
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        holdback_record={
            "path": str(holdback_path),
            "sha256": sha256_file(holdback_path),
            "rows": EXPECTED_PRIVATE_ROWS,
        },
        feedback_build_record={
            "path": str(build_path),
            "sha256": sha256_file(build_path),
        },
    )


def build_contract(
    inputs: AlignmentInputs,
    *,
    production_code: Mapping[str, Any],
    dart_bin: str,
) -> dict[str, Any]:
    proxy_reward = inputs.proxy_contract["reward"]
    selection = inputs.proxy_contract["selection"]
    return {
        "schema": CONTRACT_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "execution": {
            "model_loaded": False,
            "gpu_used": False,
            "cuda_visible_devices": "-1",
            "dart_bin": str(Path(dart_bin).expanduser().resolve()),
            "timeout_seconds": EXPECTED_TIMEOUT,
            "stability_runs": EXPECTED_STABILITY_RUNS,
        },
        "source": {
            "candidate_harvest": inputs.source_inputs.harvest_record,
            "visible_feedback": inputs.source_inputs.feedback_record,
            "visible_reward_journal": inputs.proxy_journal_record,
            "visible_reward_summary": inputs.proxy_summary_record,
            "private_holdback": inputs.holdback_record,
            "feedback_build_report": inputs.feedback_build_record,
        },
        "selection": dict(selection),
        "reward_weights_frozen": {
            "verpo_alpha": proxy_reward["alpha"],
            "local_weight": proxy_reward["local_weight"],
            "compile_weight": proxy_reward["compile_weight"],
            "fixed_before_private_holdback_read": True,
            "frozen_after_private_holdback_read": True,
            "future_weight_tuning_from_holdback_forbidden": True,
        },
        "preregistration": {
            "population": "sealed_61_p_new_groups_with_local_active_and_binary_ties",
            "uplift": {
                "definition": (
                    "within-task mean holdback pass fraction among all candidates "
                    "tied at maximum visible local reward minus unweighted all-k mean"
                ),
                "task_aggregation": "unweighted_mean",
                "target": UPLIFT_TARGET,
                "GO": "point>=0.02 and task-bootstrap lower95>0",
                "STOP": "task-bootstrap upper95<=0",
            },
            "pairwise_rank_accuracy": {
                "definition": (
                    "task-equal mean accuracy over all six unordered candidate "
                    "pairs; either a visible-local tie or holdback tie scores 0.5"
                ),
                "pairs_per_task": 6,
                "task_aggregation": "unweighted_mean",
                "target": RANK_TARGET,
                "null": RANK_NULL,
                "GO": "point>=0.55 and task-bootstrap lower95>0.50",
                "STOP": "task-bootstrap upper95<=0.50",
            },
            "bootstrap": {
                "unit": "task",
                "method": "percentile_two_sided_95",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
            "intersection_rule": {
                "GO": "both metrics GO",
                "STOP": "either metric STOP",
                "HOLD": "otherwise",
            },
        },
        "privacy": {
            "artifact_classification": "private_objective_selection_audit",
            "holdback_tests_model_visible": False,
            "holdback_tests_reward_visible": False,
            "holdback_tests_repair_visible": False,
            "candidate_code_persisted": False,
            "holdback_test_text_persisted": False,
            "diagnostics_persisted": False,
            "published_summary_aggregate_only": True,
        },
        "one_shot_policy": {
            "private_holdback_consumed_for_objective_selection": True,
            "future_objective_selection_on_this_holdback_forbidden": True,
            "future_reward_weight_tuning_on_this_holdback_forbidden": True,
            "journal_must_remain_private": True,
        },
        "production_code": dict(production_code),
        "no_policy_updates": True,
        "no_frontier_api": True,
    }


def _clean_holdback_detail(value: Mapping[str, Any], expected_cases: int) -> dict[str, Any]:
    return proxy._clean_detail(value, expected_cases)  # noqa: SLF001


def _score_one_group(
    item: tuple[int, AlignmentGroup],
    *,
    score_fn: ScoreFn,
) -> dict[str, Any]:
    position, group = item
    details = [
        _clean_holdback_detail(
            score_fn(
                code,
                group.holdback_tests,
                f"typed-holdback-{position:04d}-{candidate_position}",
                timeout=EXPECTED_TIMEOUT,
                stability_runs=EXPECTED_STABILITY_RUNS,
            ),
            group.holdback_cases,
        )
        for candidate_position, code in enumerate(group.candidate_codes)
    ]
    return {
        "schema": JOURNAL_SCHEMA,
        "event": "task_terminal",
        "task_position": position,
        "task_id": group.task_id,
        "source_sha256": group.source_sha256,
        "typed_contract_sha256": group.typed_contract_sha256,
        "candidate_code_sha256s": list(group.candidate_code_sha256s),
        "p_new": group.p_new,
        "holdback_cases": group.holdback_cases,
        "holdback_details": details,
        "candidate_code_persisted": False,
        "holdback_test_text_persisted": False,
        "diagnostics_persisted": False,
    }


def _validate_private_journal(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    groups: Sequence[AlignmentGroup],
) -> tuple[list[dict[str, Any]], bool]:
    if not events:
        return [], False
    header = events[0]
    if (
        header.get("schema") != JOURNAL_SCHEMA
        or header.get("event") != "header"
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
        or header.get("private_artifact") is not True
    ):
        raise ValueError("private holdback journal header differs")
    terminals: list[dict[str, Any]] = []
    complete = False
    for event in events[1:]:
        if event.get("event") == "complete":
            if (
                complete
                or len(terminals) != len(groups)
                or event.get("schema") != JOURNAL_SCHEMA
                or event.get("tasks") != len(groups)
                or event.get("terminal_results_sha256") != canonical_sha256(terminals)
            ):
                raise ValueError("private holdback completion differs")
            complete = True
            continue
        position = len(terminals)
        if complete or position >= len(groups):
            raise ValueError("private holdback journal ordering differs")
        group = groups[position]
        details = event.get("holdback_details")
        if (
            event.get("schema") != JOURNAL_SCHEMA
            or event.get("event") != "task_terminal"
            or event.get("task_position") != position
            or event.get("task_id") != group.task_id
            or event.get("source_sha256") != group.source_sha256
            or event.get("typed_contract_sha256") != group.typed_contract_sha256
            or event.get("candidate_code_sha256s")
            != list(group.candidate_code_sha256s)
            or event.get("p_new") is not group.p_new
            or event.get("holdback_cases") != group.holdback_cases
            or event.get("candidate_code_persisted") is not False
            or event.get("holdback_test_text_persisted") is not False
            or event.get("diagnostics_persisted") is not False
            or not isinstance(details, list)
            or len(details) != EXPECTED_CANDIDATES_PER_GROUP
        ):
            raise ValueError(f"private holdback terminal {position} differs")
        cleaned = [
            _clean_holdback_detail(value, group.holdback_cases)
            for value in details
            if isinstance(value, Mapping)
        ]
        if cleaned != details or len(cleaned) != EXPECTED_CANDIDATES_PER_GROUP:
            raise ValueError(f"private holdback evidence {position} differs")
        forbidden = ("candidate", "code", "tests", "diagnostic")
        if any(key in json.dumps(event).lower() for key in forbidden):
            # Schema fields deliberately contain *_persisted; inspect keys
            # structurally instead of rejecting those declarations.
            for key in event:
                if key in {"candidate", "code", "tests", "diagnostic"}:
                    raise ValueError("private payload entered holdback journal")
        terminals.append(dict(event))
    return terminals, complete


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(float(value) for value in values) / len(values)


def _holdback_fraction(detail: Mapping[str, Any]) -> float:
    passes = detail["test_passes"]
    return sum(bool(value) for value in passes) / len(passes)


def _argmax_tie_average(values: Sequence[float], outcomes: Sequence[float]) -> float:
    if len(values) != len(outcomes) or not values:
        raise ValueError("argmax tie-average inputs differ")
    maximum = max(float(value) for value in values)
    chosen = [
        float(outcome)
        for value, outcome in zip(values, outcomes, strict=True)
        if float(value) == maximum
    ]
    return _mean(chosen)


def _pairwise_contribution(
    visible: Sequence[float], holdback: Sequence[float]
) -> tuple[float, int]:
    if len(visible) != EXPECTED_CANDIDATES_PER_GROUP or len(holdback) != len(visible):
        raise ValueError("pairwise accuracy requires exactly four aligned candidates")
    correct = 0.0
    pairs = 0
    for left in range(len(visible)):
        for right in range(left + 1, len(visible)):
            difference = float(visible[left]) - float(visible[right])
            outcome = float(holdback[left]) - float(holdback[right])
            pairs += 1
            if difference == 0.0 or outcome == 0.0:
                correct += 0.5
            elif difference * outcome > 0.0:
                correct += 1.0
    if pairs != 6:
        raise ValueError("four-candidate task did not produce exactly six pairs")
    return correct, pairs


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    return proxy._percentile(ordered, quantile)  # noqa: SLF001


def _bootstrap(
    task_rows: Sequence[tuple[float, float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not task_rows or replicates <= 0:
        raise ValueError("bootstrap inputs are empty")
    generator = random.Random(seed)
    uplift_values: list[float] = []
    rank_values: list[float] = []
    for _ in range(replicates):
        sample = [task_rows[generator.randrange(len(task_rows))] for _ in task_rows]
        uplift_values.append(_mean([row[0] for row in sample]))
        rank_values.append(_mean([row[1] for row in sample]))
    common = {
        "method": "task_bootstrap_percentile",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
    }
    return (
        {
            **common,
            "lower": _percentile(uplift_values, 0.025),
            "upper": _percentile(uplift_values, 0.975),
        },
        {
            **common,
            "lower": _percentile(rank_values, 0.025),
            "upper": _percentile(rank_values, 0.975),
        },
    )


def _uplift_decision(point: float, interval: Mapping[str, float]) -> str:
    if point >= UPLIFT_TARGET and float(interval["lower"]) > 0.0:
        return "GO"
    if float(interval["upper"]) <= 0.0:
        return "STOP"
    return "HOLD"


def _rank_decision(point: float, interval: Mapping[str, float]) -> str:
    if point >= RANK_TARGET and float(interval["lower"]) > RANK_NULL:
        return "GO"
    if float(interval["upper"]) <= RANK_NULL:
        return "STOP"
    return "HOLD"


def summarize(
    groups: Sequence[AlignmentGroup],
    terminals: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if len(groups) != len(terminals) or not groups:
        raise ValueError("holdback summary inputs differ")
    p_new_rows: list[tuple[float, float]] = []
    all_candidate_fractions: list[float] = []
    all_candidate_full: list[float] = []
    unified_selected_fraction: list[float] = []
    unified_selected_full: list[float] = []
    random_fraction: list[float] = []
    random_full: list[float] = []

    for group, terminal in zip(groups, terminals, strict=True):
        details = terminal["holdback_details"]
        fractions = [_holdback_fraction(value) for value in details]
        full = [float(bool(value["full_pass"])) for value in details]
        all_candidate_fractions.extend(fractions)
        all_candidate_full.extend(full)
        random_fraction.append(_mean(fractions))
        random_full.append(_mean(full))
        unified_selected_fraction.append(
            _argmax_tie_average(group.visible_unified_advantages, fractions)
        )
        unified_selected_full.append(
            _argmax_tie_average(group.visible_unified_advantages, full)
        )
        if group.p_new:
            selected = _argmax_tie_average(group.visible_local_rewards, fractions)
            uplift = selected - _mean(fractions)
            pair_correct, pair_count = _pairwise_contribution(
                group.visible_local_rewards, fractions
            )
            if pair_count != 6:
                raise ValueError("p_new task pair count differs")
            p_new_rows.append((uplift, pair_correct / pair_count))

    if len(p_new_rows) != EXPECTED_P_NEW_GROUPS:
        raise ValueError("summary p_new group count differs")
    uplift_point = _mean([row[0] for row in p_new_rows])
    rank_point = _mean([row[1] for row in p_new_rows])
    uplift_interval, rank_interval = _bootstrap(
        p_new_rows,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    uplift_decision = _uplift_decision(uplift_point, uplift_interval)
    rank_decision = _rank_decision(rank_point, rank_interval)
    if "STOP" in (uplift_decision, rank_decision):
        overall = "STOP"
    elif uplift_decision == rank_decision == "GO":
        overall = "GO"
    else:
        overall = "HOLD"

    return {
        "preregistered_61_p_new": {
            "groups": len(p_new_rows),
            "tie_averaged_visible_local_argmax_uplift": {
                "point": uplift_point,
                "interval": uplift_interval,
                "target": UPLIFT_TARGET,
                "decision": uplift_decision,
            },
            "pairwise_rank_accuracy": {
                "point": rank_point,
                "task_equal": True,
                "pairs_per_task": 6,
                "pairs_descriptively": len(p_new_rows) * 6,
                "interval": rank_interval,
                "target": RANK_TARGET,
                "null": RANK_NULL,
                "visible_or_holdback_ties_score": 0.5,
                "decision": rank_decision,
            },
            "overall_decision": overall,
        },
        "descriptive_all_150": {
            "groups": len(groups),
            "candidates": len(all_candidate_fractions),
            "candidate_holdback_case_pass_fraction": _mean(all_candidate_fractions),
            "candidate_full_holdback_pass_fraction": _mean(all_candidate_full),
            "visible_unified_argmax_tie_averaged_holdback_case_pass_fraction": (
                _mean(unified_selected_fraction)
            ),
            "visible_unified_argmax_uplift_over_random_choice": (
                _mean(unified_selected_fraction) - _mean(random_fraction)
            ),
            "visible_unified_argmax_tie_averaged_full_holdback_pass_fraction": (
                _mean(unified_selected_full)
            ),
            "visible_unified_argmax_full_holdback_uplift_over_random_choice": (
                _mean(unified_selected_full) - _mean(random_full)
            ),
            "groups_with_any_full_holdback_pass": sum(
                any(value["full_pass"] for value in terminal["holdback_details"])
                for terminal in terminals
            ),
        },
    }


def run_audit(
    inputs: AlignmentInputs,
    *,
    contract: Mapping[str, Any],
    output_journal: str | Path,
    output_summary: str | Path,
    score_fn: ScoreFn,
    workers: int,
    stop_after_new_tasks: int | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any] | None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    journal_path = Path(output_journal).expanduser().resolve()
    summary_path = Path(output_summary).expanduser().resolve()
    if summary_path.exists() and not journal_path.exists():
        raise ValueError("published summary exists without private journal")
    _enforce_private_journal_permissions(journal_path)
    events = load_journal(journal_path)
    if not events:
        _append_private_event(
            journal_path,
            {
                "schema": JOURNAL_SCHEMA,
                "event": "header",
                "private_artifact": True,
                "contract": dict(contract),
                "contract_sha256": canonical_sha256(contract),
            },
        )
        events = load_journal(journal_path)
    terminals, complete = _validate_private_journal(
        events, contract=contract, groups=inputs.groups
    )
    if not complete:
        remaining = list(enumerate(inputs.groups))[len(terminals) :]
        if stop_after_new_tasks is not None:
            if stop_after_new_tasks < 0:
                raise ValueError("stop_after_new_tasks must be non-negative")
            remaining = remaining[:stop_after_new_tasks]
        with ThreadPoolExecutor(max_workers=min(workers, len(remaining) or 1)) as pool:
            for completed, terminal in enumerate(
                pool.map(lambda item: _score_one_group(item, score_fn=score_fn), remaining),
                1,
            ):
                _append_private_event(journal_path, terminal)
                absolute = len(terminals) + completed
                if absolute % 10 == 0 or absolute == len(inputs.groups):
                    print(f"typed private holdback audit {absolute}/{len(inputs.groups)}", flush=True)
        events = load_journal(journal_path)
        terminals, complete = _validate_private_journal(
            events, contract=contract, groups=inputs.groups
        )
        if not complete and len(terminals) == len(inputs.groups):
            _append_private_event(
                journal_path,
                {
                    "schema": JOURNAL_SCHEMA,
                    "event": "complete",
                    "tasks": len(terminals),
                    "terminal_results_sha256": canonical_sha256(terminals),
                },
            )
            events = load_journal(journal_path)
            terminals, complete = _validate_private_journal(
                events, contract=contract, groups=inputs.groups
            )
    if not complete:
        return None

    metrics = summarize(
        inputs.groups,
        terminals,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    body = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "contract_sha256": canonical_sha256(contract),
        "private_journal": {
            **journal_record(journal_path),
            "artifact_classification": "private_do_not_publish_or_train",
        },
        "metrics": metrics,
        "decision": metrics["preregistered_61_p_new"]["overall_decision"],
        "privacy": {
            "aggregate_only": True,
            "per_task_outcomes_published": False,
            "candidate_code_published": False,
            "holdback_test_text_published": False,
            "diagnostics_published": False,
        },
        "one_shot_policy": dict(contract["one_shot_policy"]),
        "reward_weights_frozen": dict(contract["reward_weights_frozen"]),
        "next_action": {
            "GO": "typed_C2_discardable_32_update_pilot_may_be_reviewed",
            "HOLD": "do_not_launch_training_without_new_preregistered_direction",
            "STOP": "do_not_launch_typed_C2_VeRPO_on_this_reward",
        }[metrics["preregistered_61_p_new"]["overall_decision"]],
    }
    summary = {**body, "summary_sha256": canonical_sha256(body)}
    require_exact_or_write(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--harvest_journal", required=True)
    parser.add_argument("--feedback_jsonl", required=True)
    parser.add_argument("--proxy_journal", required=True)
    parser.add_argument("--proxy_summary", required=True)
    parser.add_argument("--holdback_jsonl", required=True)
    parser.add_argument("--feedback_build_report", required=True)
    parser.add_argument("--output_journal", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument(
        "--dart_bin",
        default="/workspace/tools/dart-3.12.2/usr/lib/dart/bin/dart",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in [1, 32]")

    dart_bin = Path(args.dart_bin).expanduser().resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["DART_BIN"] = str(dart_bin)
    os.environ["PATH"] = str(dart_bin.parent) + os.pathsep + os.environ.get("PATH", "")
    score_fn, reward_fn, split_fn, production_code = proxy.load_production_components()
    production_code["holdback_alignment_audit"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    inputs = load_alignment_inputs(
        harvest_journal=args.harvest_journal,
        feedback_jsonl=args.feedback_jsonl,
        proxy_journal=args.proxy_journal,
        proxy_summary=args.proxy_summary,
        holdback_jsonl=args.holdback_jsonl,
        feedback_build_report=args.feedback_build_report,
        score_reward_fn=reward_fn,
        split_fn=split_fn,
    )
    contract = build_contract(inputs, production_code=production_code, dart_bin=str(dart_bin))
    summary = run_audit(
        inputs,
        contract=contract,
        output_journal=args.output_journal,
        output_summary=args.output_summary,
        score_fn=score_fn,
        workers=args.workers,
    )
    if summary is None:
        raise RuntimeError("private holdback alignment audit stopped before completion")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
