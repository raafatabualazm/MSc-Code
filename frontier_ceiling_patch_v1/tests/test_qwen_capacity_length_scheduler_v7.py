from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import qwen37_capacity_length_scheduler_v7 as scheduler
import qwen37_capacity_length_repair_v7 as repair


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_scheduler_tracks_last_v6_stage_even_when_complete(
    tmp_path: Path,
) -> None:
    transitions = (
        tmp_path
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "qwen37_capacity_v6_scheduler"
        / "transitions.jsonl"
    )
    write_jsonl(
        transitions,
        [
            {
                "record_type": "stage_preflight",
                "stage": "current_preview",
            },
            {
                "record_type": "stage_preflight",
                "stage": "current_0517",
            }
        ],
    )
    assert scheduler.v6_stage(tmp_path) == "source"
    write_jsonl(
        transitions,
        [
            {
                "record_type": "stage_preflight",
                "stage": "fallback_all_five",
            }
        ],
    )
    assert scheduler.v6_stage(tmp_path) == "fallback"
    run_root = (
        tmp_path
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
    )
    for partition in scheduler.PARTITIONS:
        for arm in scheduler.ARMS:
            write_json(
                run_root
                / f"qwen37_capacity_v6_{partition}_{arm}_mc12k_tb8k"
                / "status.json",
                {"status": "complete"},
            )
    transitions.unlink()
    assert scheduler.v6_stage(tmp_path) == "dated"


def test_scheduler_accepts_only_exact_quota_as_epoch_boundary(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / "failure.json",
        {"status": "failed_closed", "error": "quota"},
    )
    write_jsonl(
        tmp_path / "repair_attempts.jsonl",
        [
            {
                "schema": repair.ATTEMPT_SCHEMA,
                "response_received": False,
                "retryable_transport": False,
                "transport_error": (
                    "Error code: 403 AllocationQuota.FreeTierOnly: "
                    "The free quota has been exhausted"
                ),
            }
        ],
    )
    assert scheduler.exact_quota_failure(tmp_path) is True
    assert scheduler.hard_failure(tmp_path) is None
    write_jsonl(
        tmp_path / "repair_attempts.jsonl",
        [
            {
                "schema": repair.ATTEMPT_SCHEMA,
                "response_received": False,
                "retryable_transport": False,
                "transport_error": "Error code: 429 rate limit",
            }
        ],
    )
    assert scheduler.exact_quota_failure(tmp_path) is False
    assert scheduler.hard_failure(tmp_path) == "quota"


def test_repair_epoch_output_names_are_disjoint(tmp_path: Path) -> None:
    paths = {
        scheduler.repair_root(
            tmp_path,
            partition="0520",
            arm="opus",
            stage=stage,
        )
        for stage in scheduler.STAGES
    }
    assert len(paths) == len(scheduler.STAGES)


def test_current_completion_never_launches_generic_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(scheduler, "validate_dependencies", lambda _w: None)
    monkeypatch.setattr(scheduler, "v6_stage", lambda _w: "dated")
    monkeypatch.setattr(scheduler, "capacity_complete", lambda _w: True)
    monkeypatch.setattr(
        scheduler,
        "ensure_prior_preflights",
        lambda _w, *, stage, partition, arm: calls.append(
            ("preflight", stage, f"{partition}/{arm}")
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "call_launcher",
        lambda _w, *, operation, stage, partition, arm: (
            calls.append((operation, stage, f"{partition}/{arm}")) or 0
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "repair_root",
        lambda *_args, **_kwargs: tmp_path / "repair",
    )
    monkeypatch.setattr(
        scheduler,
        "all_required_repairs_complete",
        lambda _w, *, stage: stage == "dated",
    )
    monkeypatch.setattr(scheduler, "append_journal", lambda *_a, **_k: None)
    result = scheduler.run_scheduler(
        SimpleNamespace(
            workspace=tmp_path,
            poll_seconds=1,
            timeout_seconds=10,
        )
    )
    assert result == 0
    assert calls
    assert {stage for _operation, stage, _target in calls} == {"dated"}


def test_repair_stage_prefix_mirrors_all_current_alias_stages() -> None:
    assert scheduler.STAGES == (
        "dated",
        "preview",
        "source",
        "generic",
        "fallback",
    )
    assert scheduler.V6_STAGE_TO_REPAIR_STAGE == {
        "current_dated": "dated",
        "current_preview": "preview",
        "current_0517": "source",
        "current_generic": "generic",
        "fallback_all_five": "fallback",
    }
    assert scheduler.required_repair_stages("fallback") == scheduler.STAGES
    assert scheduler.required_repair_stages("source") == (
        "dated",
        "preview",
        "source",
    )


def test_fallback_preflight_requires_all_current_repair_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        scheduler,
        "ensure_preflight",
        lambda _workspace, *, stage, partition, arm: calls.append(stage),
    )
    scheduler.ensure_prior_preflights(
        tmp_path,
        stage="fallback",
        partition="0520",
        arm="opus",
    )
    assert calls == list(scheduler.STAGES)
