from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import qwen37_capacity_scheduler_v6 as scheduler


def _append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _quota_row(
    *,
    stage: scheduler.Stage,
    alias: str,
    attempt_id: str,
) -> dict[str, object]:
    return {
        "schema": "qwen37-capacity-fallback-v6",
        "record_type": "capacity_route_attempt",
        "capacity_epoch": stage.epoch,
        "attempt_id": attempt_id,
        "requested_model": alias,
        "response_received": False,
        "http_status": 403,
        "route_result": "quota_exhausted_403",
        "error": (
            "PermissionDeniedError: Error code: 403 - "
            "The free quota has been exhausted. "
            "AllocationQuota.FreeTierOnly"
        ),
    }


def test_exact_quota_rejects_429_and_generic_403() -> None:
    stage = scheduler.STAGES[0]
    exact = _quota_row(stage=stage, alias="x", attempt_id="a")
    assert scheduler.exact_quota_403(exact)
    rate_limit = {**exact, "http_status": 429, "error": "Error code: 429"}
    assert not scheduler.exact_quota_403(rate_limit)
    forbidden = {
        **exact,
        "error": "PermissionDeniedError: Error code: 403 - forbidden",
    }
    assert not scheduler.exact_quota_403(forbidden)


def test_epoch_evidence_requires_each_alias_and_matched_dispatch(
    tmp_path: Path,
) -> None:
    stage = scheduler.STAGES[0]
    out = scheduler.output_dirs(tmp_path)[0]
    for index, alias in enumerate(stage.required_exact_aliases):
        attempt_id = f"attempt-{index}"
        _append(
            out / "dispatches.jsonl",
            {
                "capacity_epoch": stage.epoch,
                "attempt_id": attempt_id,
            },
        )
        _append(
            out / "route_attempts.jsonl",
            _quota_row(
                stage=stage,
                alias=alias,
                attempt_id=attempt_id,
            ),
        )
    evidence = scheduler.epoch_exact_evidence(tmp_path, stage)
    assert set(evidence) == set(stage.required_exact_aliases)
    assert all(len(rows) == 1 for rows in evidence.values())

    _append(
        out / "dispatches.jsonl",
        {
            "capacity_epoch": stage.epoch,
            "attempt_id": "unmatched",
        },
    )
    with pytest.raises(scheduler.SchedulerError, match="without terminal"):
        scheduler.epoch_exact_evidence(tmp_path, stage)


def test_inactive_incomplete_stage_without_all_evidence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = scheduler.STAGES[0]
    monkeypatch.setattr(scheduler, "all_complete", lambda workspace: False)
    monkeypatch.setattr(
        scheduler,
        "units_inactive",
        lambda value: (True, [{"ActiveState": "failed"}]),
    )
    with pytest.raises(
        scheduler.SchedulerError,
        match="stopped without every exact quota boundary",
    ):
        scheduler.wait_stage(
            tmp_path,
            stage,
            poll_seconds=1,
            timeout_seconds=2,
        )


def test_complete_stage_bypasses_quota_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = scheduler.STAGES[0]
    monkeypatch.setattr(scheduler, "all_complete", lambda workspace: True)
    monkeypatch.setattr(
        scheduler,
        "units_inactive",
        lambda value: (True, [{"ActiveState": "inactive"}]),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("quota evidence must not be read after completion")

    monkeypatch.setattr(scheduler, "epoch_exact_evidence", forbidden)
    status, evidence, _ = scheduler.wait_stage(
        tmp_path,
        stage,
        poll_seconds=1,
        timeout_seconds=2,
    )
    assert status == "complete"
    assert evidence == {}


def test_all_current_workspace_alias_stages_precede_fallback() -> None:
    assert [stage.name for stage in scheduler.STAGES] == [
        "current_dated",
        "current_preview",
        "current_0517",
        "current_generic",
        "fallback_all_five",
    ]
    current_aliases = {
        alias
        for stage in scheduler.STAGES[:-1]
        for alias in stage.required_exact_aliases
    }
    assert current_aliases == {
        "qwen3.7-max-2026-05-17",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-preview",
        "qwen3.7-max",
    }


def test_scheduler_cannot_reach_fallback_before_every_current_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[str, str]] = []
    terminals: list[str] = []
    monkeypatch.setattr(scheduler, "sha256_file", lambda _path: "sealed")
    for name in (
        "OVERLAY_ENTRY_SHA256",
        "LAUNCHER_SHA256",
        "CONTRACT_SHA256",
        "EXTENSION_CONTRACT_SHA256",
    ):
        monkeypatch.setattr(scheduler, name, "sealed")
    monkeypatch.setattr(
        scheduler,
        "all_complete",
        lambda _workspace: False,
    )
    monkeypatch.setattr(
        scheduler,
        "launcher_action",
        lambda _workspace, *, operation, stage: launches.append(
            (operation, stage.name)
        ),
    )

    def stop_before_preview(
        _workspace: Path,
        stage: scheduler.Stage,
        **_kwargs: object,
    ) -> tuple[
        str,
        dict[str, list[dict[str, str]]],
        list[dict[str, str]],
    ]:
        terminals.append(stage.name)
        if stage.name == "current_preview":
            raise scheduler.SchedulerError("transport error cannot advance")
        return (
            "exact_quota_exhausted",
            {alias: [{"attempt_id": alias}] for alias in stage.required_exact_aliases},
            [],
        )

    monkeypatch.setattr(scheduler, "wait_stage", stop_before_preview)
    with pytest.raises(scheduler.SchedulerError, match="cannot advance"):
        scheduler.run_scheduler(
            SimpleNamespace(
                workspace=tmp_path,
                poll_seconds=1,
                stage_timeout_seconds=10,
            )
        )
    assert terminals == ["current_dated", "current_preview"]
    assert all(stage != "fallback_all_five" for _op, stage in launches)


@pytest.mark.parametrize(
    ("http_status", "error"),
    [
        (429, "Error code: 429 - rate limit"),
        (None, "APIConnectionError: connection reset"),
    ],
)
def test_429_or_transport_evidence_cannot_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_status: int | None,
    error: str,
) -> None:
    stage = scheduler.STAGES[1]
    out = scheduler.output_dirs(tmp_path)[0]
    attempt_id = "non-boundary"
    _append(
        out / "dispatches.jsonl",
        {"capacity_epoch": stage.epoch, "attempt_id": attempt_id},
    )
    row = _quota_row(
        stage=stage,
        alias=stage.required_exact_aliases[-1],
        attempt_id=attempt_id,
    )
    row.update(
        {
            "http_status": http_status,
            "route_result": "transport_or_provider_error",
            "error": error,
        }
    )
    _append(out / "route_attempts.jsonl", row)
    monkeypatch.setattr(scheduler, "all_complete", lambda _workspace: False)
    monkeypatch.setattr(
        scheduler,
        "units_inactive",
        lambda _stage: (True, [{"ActiveState": "failed"}]),
    )
    with pytest.raises(
        scheduler.SchedulerError,
        match="stopped without every exact quota boundary",
    ):
        scheduler.wait_stage(
            tmp_path,
            stage,
            poll_seconds=1,
            timeout_seconds=2,
        )
