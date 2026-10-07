from __future__ import annotations

import json
import sys
from pathlib import Path

PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import frontier_passk as runner
import qwen37_model_quota_guard as guard


PREVIEW = "qwen3.7-max-preview"


def exact_row(model: str = PREVIEW) -> dict[str, object]:
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "record_type": "api_attempt",
        "provider": "qwen",
        "requested_model": model,
        "response_received": False,
        "slot_terminal": False,
        "candidate_valid": None,
        "terminal_reason": None,
        "transport_retry": True,
        "retryable_transport": False,
        "fatal_response_contract": False,
        "usage": None,
        "response": None,
        "task_id": "sigless_test",
        "sample_index": 0,
        "attempt_index": 0,
        "attempt_id": "sigless_test.s0.a0.test",
        "transport_error": (
            "api_exception:PermissionDeniedError:Error code: 403 - "
            "{'error': {'message': 'The free quota has been exhausted. More.', "
            "'type': 'AllocationQuota.FreeTierOnly', 'param': None, "
            "'code': 'AllocationQuota.FreeTierOnly'}, "
            "'id': 'chatcmpl-test', 'request_id': 'test'}"
        ),
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_preview_scope_is_exactly_four_units_and_no_deepseek() -> None:
    targets = guard.targets_for_model(PREVIEW)
    assert len(targets) == 4
    assert all(target.model == PREVIEW for target in targets)
    assert all("preview" in target.unit for target in targets)
    assert all("deepseek" not in target.unit.lower() for target in targets)


def test_diagnostic_scope_excludes_moderation_rejected_opus_units() -> None:
    may = guard.targets_for_model("qwen3.7-max-2026-05-20")
    june = guard.targets_for_model("qwen3.7-max-2026-06-08")
    assert len(may) == 1 and "-codex-" in may[0].unit
    assert len(june) == 1 and "-codex-" in june[0].unit


def test_scan_finds_only_exact_model_boundary(tmp_path: Path) -> None:
    preview_target = guard.targets_for_model(PREVIEW)[0]
    append_row(
        tmp_path / preview_target.run_id / "attempts.jsonl",
        exact_row(),
    )
    evidence = guard.scan_model(tmp_path, PREVIEW, set())
    assert len(evidence) == 1
    assert evidence[0]["unit"] == preview_target.unit
    assert evidence[0]["http_status"] == 403
    assert evidence[0]["provider_error_variant"] == (
        "AllocationQuota.FreeTierOnly"
    )


def test_scan_ignores_429_rate_limit(tmp_path: Path) -> None:
    target = guard.targets_for_model(PREVIEW)[0]
    row = exact_row()
    row["transport_error"] = str(row["transport_error"]).replace(
        "Error code: 403", "Error code: 429"
    )
    append_row(tmp_path / target.run_id / "attempts.jsonl", row)
    assert guard.scan_model(tmp_path, PREVIEW, set()) == []


def test_processed_receipt_prevents_retrip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = guard.targets_for_model(PREVIEW)[0]
    append_row(tmp_path / "runs" / target.run_id / "attempts.jsonl", exact_row())
    evidence = guard.scan_model(tmp_path / "runs", PREVIEW, set())
    stopped: list[str] = []

    def fake_stop(model: str):
        stopped.append(model)
        return [{"unit": item.unit, "stop_returncode": 0} for item in (
            guard.targets_for_model(model)
        )]

    monkeypatch.setattr(guard, "stop_model_units", fake_stop)
    receipt = guard.trip_guard(
        model=PREVIEW,
        evidence=evidence,
        receipt_dir=tmp_path / "receipts",
    )
    assert receipt["scope"] == "exact_requested_model_only"
    assert receipt["deepseek_units_targeted"] is False
    assert stopped == [PREVIEW]
    processed = guard.load_processed_evidence(tmp_path / "receipts")
    assert guard.scan_model(tmp_path / "runs", PREVIEW, processed) == []
