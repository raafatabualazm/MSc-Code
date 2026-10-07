from __future__ import annotations

import sys
from pathlib import Path
import hashlib

import pytest


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import qwen37_primary_effective_status_v7 as status


def test_optional_failure_receipt_hash_is_null_when_absent(
    tmp_path: Path,
) -> None:
    failure = tmp_path / "failure.json"
    assert status.sha256_file(failure) is None
    raw = b'{"status":"failed_closed"}\n'
    failure.write_bytes(raw)
    assert status.sha256_file(failure) == hashlib.sha256(raw).hexdigest()


def test_adaptive_mapping_replaces_length_without_increasing_k() -> None:
    original_length = ("opus", "task", 0)
    capacity_length = ("opus", "task", 1)
    ordinary = ("opus", "task", 2)
    terminals = {
        original_length: {"finish_reason": "length"},
        capacity_length: {"finish_reason": "length"},
        ordinary: {"finish_reason": "stop"},
    }
    fixed = {
        original_length: {"passed": False},
        capacity_length: {"passed": False},
        ordinary: {"passed": True},
    }
    adaptive, waiting = status.build_adaptive_outcomes(
        fixed_terminals=terminals,
        fixed_outcomes=fixed,
        original_keys={original_length, ordinary},
        capacity_keys={capacity_length},
        original_replacements={original_length: {"passed": True}},
        capacity_replacements={capacity_length: {"passed": True}},
    )
    assert waiting == []
    assert set(adaptive) == set(terminals)
    assert adaptive[original_length]["passed"] is True
    assert adaptive[capacity_length]["passed"] is True
    assert adaptive[ordinary] is fixed[ordinary]


def test_adaptive_mapping_withholds_missing_length_repair() -> None:
    key = ("codex", "task", 9)
    adaptive, waiting = status.build_adaptive_outcomes(
        fixed_terminals={key: {"finish_reason": "length"}},
        fixed_outcomes={key: {"passed": False}},
        original_keys=set(),
        capacity_keys={key},
        original_replacements={},
        capacity_replacements={},
    )
    assert adaptive == {}
    assert waiting == [key]


def test_adaptive_mapping_rejects_nonlength_or_foreign_repair() -> None:
    key = ("opus", "task", 0)
    with pytest.raises(status.AuditError):
        status.build_adaptive_outcomes(
            fixed_terminals={key: {"finish_reason": "stop"}},
            fixed_outcomes={key: {"passed": False}},
            original_keys={key},
            capacity_keys=set(),
            original_replacements={key: {"passed": True}},
            capacity_replacements={},
        )
    with pytest.raises(status.AuditError):
        status.build_adaptive_outcomes(
            fixed_terminals={key: {"finish_reason": "length"}},
            fixed_outcomes={key: {"passed": False}},
            original_keys={key},
            capacity_keys=set(),
            original_replacements={
                ("opus", "foreign", 0): {"passed": True}
            },
            capacity_replacements={},
        )


def test_metrics_withhold_until_exact_3500_slots() -> None:
    task_ids = tuple(
        f"task_{index:03d}" for index in range(status.EXPECTED_TASKS)
    )
    outcomes = {
        (arm, task_id, sample_index): {
            "passed": sample_index == 0,
            "compiled": sample_index == 0,
        }
        for arm in status.ARMS
        for task_id in task_ids
        for sample_index in range(status.EXPECTED_K)
    }
    metrics = status.metrics_if_complete(outcomes, task_ids)
    assert metrics is not None
    assert metrics["opus"]["pass_at_10"]["successes"] == 175
    outcomes.pop(next(iter(outcomes)))
    assert status.metrics_if_complete(outcomes, task_ids) is None
