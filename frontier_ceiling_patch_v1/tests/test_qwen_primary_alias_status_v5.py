from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import qwen37_primary_alias_status_v5 as status


def test_meta_contract_and_code_mapping_are_exact() -> None:
    contract = status.validate_meta_contract(PATCH)
    assert contract["primary_global_k_per_arm"] == 10
    assert len(status.SHARDS) == 4
    assert len(
        {
            shard.directory_template.format(arm=arm)
            for shard in status.SHARDS
            for arm in status.ARMS
        }
    ) == 8
    assert sorted(
        index for shard in status.SHARDS for index in shard.global_indices
    ) == list(range(10))
    assert all(
        shard.model not in status.QUARANTINED_MODELS
        for shard in status.SHARDS
    )


def valid_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 100,
        "completion_tokens": 12_298,
        "total_tokens": 12_398,
        "reasoning_tokens": 8_192,
        "answer_tokens": 4_106,
    }


def test_usage_accepts_exact_tolerance_boundary() -> None:
    status.validate_usage(valid_usage())


@pytest.mark.parametrize(
    "field,value",
    [
        ("completion_tokens", 12_299),
        ("reasoning_tokens", 0),
        ("reasoning_tokens", 8_193),
        ("answer_tokens", 4_105),
        ("prompt_tokens", 12_001),
    ],
)
def test_usage_rejects_contract_drift(field: str, value: int) -> None:
    usage = valid_usage()
    usage[field] = value
    if field in {"completion_tokens", "prompt_tokens"}:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    with pytest.raises(status.AuditError):
        status.validate_usage(usage)


def task_ids() -> tuple[str, ...]:
    return tuple(f"task_{index:03d}" for index in range(status.EXPECTED_TASKS))


def all_outcomes() -> dict[tuple[str, str, int], dict[str, bool]]:
    return {
        (arm, task_id, sample_index): {
            "passed": sample_index == 0 and task_id.endswith("000"),
            "compiled": sample_index == 0,
        }
        for arm in status.ARMS
        for task_id in task_ids()
        for sample_index in range(status.EXPECTED_K)
    }


def test_metrics_are_withheld_for_any_missing_global_slot() -> None:
    outcomes = all_outcomes()
    outcomes.pop(next(iter(outcomes)))
    assert status.metrics_if_complete(outcomes, task_ids()) is None


def test_metrics_emit_only_for_exact_3500_global_slots() -> None:
    metrics = status.metrics_if_complete(all_outcomes(), task_ids())
    assert metrics is not None
    for arm in status.ARMS:
        assert metrics[arm]["pass_at_10"]["successes"] == 1
        assert metrics[arm]["compile_at_10"]["successes"] == 175
        assert metrics[arm]["pass_at_10"]["total"] == 175


def test_metrics_reject_foreign_or_overfull_global_slot() -> None:
    outcomes = all_outcomes()
    outcomes[("opus", "foreign", 0)] = copy.deepcopy(next(iter(outcomes.values())))
    with pytest.raises(status.AuditError):
        status.metrics_if_complete(outcomes, task_ids())
