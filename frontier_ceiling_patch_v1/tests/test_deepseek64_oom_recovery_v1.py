from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import deepseek64_oom_recovery_v1 as recovery
import frontier_passk as runner


def _line(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path, dict, str]:
    out = tmp_path / "out"
    out.mkdir(parents=True)
    config = "c" * 64
    policy = "s" * 64
    prompt = "p" * 64
    intent = {
        "schema": "deepseek-64k-dispatch-v1",
        "record_type": "dispatch_intent",
        "config_sha256": config,
        "dispatch_id": "dispatch-1",
        "task_id": "task-a",
        "sample_index": 2,
        "attempt_index": 0,
        "attempt_id": "attempt-1",
        "requested_max_tokens": 65536,
        "created_at": "2026-07-26T12:35:40Z",
    }
    dispatch_raw = _line(intent)
    (out / "attempts64.jsonl").write_bytes(b"")
    (out / "outcomes64.jsonl").write_bytes(b"")
    (out / "dispatches64.jsonl").write_bytes(dispatch_raw)
    contract = {
        "fixed_policy": {
            "provider": "deepseek",
            "requested_model": "deepseek-v4-pro",
            "config_sha256": config,
            "slot_policy_sha256": policy,
            "prompt_sha256": prompt,
            "max_prompt_tokens": 12000,
            "requested_max_tokens": 65536,
        },
        "baseline_journals": {
            "attempts64.jsonl": {"lines": 0, "bytes": 0, "sha256": _sha(b"")},
            "outcomes64.jsonl": {"lines": 0, "bytes": 0, "sha256": _sha(b"")},
            "dispatches64.jsonl": {
                "lines": 1,
                "bytes": len(dispatch_raw),
                "sha256": _sha(dispatch_raw),
            },
        },
        "expected_dangling_intents": [
            {
                key: intent[key]
                for key in (
                    "dispatch_id",
                    "task_id",
                    "sample_index",
                    "attempt_index",
                    "attempt_id",
                    "config_sha256",
                    "requested_max_tokens",
                    "created_at",
                )
            }
        ],
    }
    return contract, out, {"task-a": {"prompt_sha256": prompt}}, "e" * 64


def _recover(
    contract: dict,
    out: Path,
    prompt_map: dict,
    evidence_sha: str,
    *,
    apply: bool = True,
    next_attempt: int = 0,
) -> dict:
    return recovery.recover_exact_dangling(
        contract=contract,
        contract_sha256="k" * 64,
        out_root=out,
        prompt_map=prompt_map,
        source_terminal={},
        overlay_terminal={},
        next_attempt={("task-a", 2): next_attempt},
        evidence_sha256=evidence_sha,
        apply=apply,
    )


def test_exact_recovery_writes_conservative_receipt_and_settlement(
    tmp_path: Path,
) -> None:
    contract, out, prompt_map, evidence_sha = _fixture(tmp_path)
    result = _recover(contract, out, prompt_map, evidence_sha)
    assert result["recovered_count"] == 1
    assert result["full_reservation_tokens_per_attempt"] == 77536
    attempt = runner.load_jsonl(out / "attempts64.jsonl", "attempts")[0]
    assert attempt["response_received"] is False
    assert attempt["slot_terminal"] is False
    assert attempt["retryable_transport"] is True
    assert attempt["transport_error"] == "process_oom_unknown_response"
    assert attempt["budget_charge_tokens"] == 12000 + 65536
    assert attempt["prompt_sha256"] == "p" * 64
    assert attempt["slot_policy_sha256"] == "s" * 64
    assert attempt["provider_charge_status"] == "unknown_may_have_been_billed"
    assert "may have reached the provider" in attempt["duplicate_call_warning"]
    dispatches = runner.load_jsonl(out / "dispatches64.jsonl", "dispatches")
    settlement = dispatches[1]
    assert settlement["record_type"] == "dispatch_settlement"
    assert settlement["dispatch_id"] == "dispatch-1"
    assert settlement["attempt_recorded"] is True
    assert settlement["recovery_reason"] == "process_oom_unknown_response"


def test_missing_oom_evidence_refuses_before_journal_change(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    lock = {"pid": 42, "host": "test-host", "created_at": "x"}
    payload = _line(lock)
    (archive / "overlay.run.lock").write_bytes(payload)
    contract = {
        "dead_owner": {"pid": 42, "host": "test-host"},
        "oom_evidence": {
            "required_journal_substrings": ["unit failed with result oom-kill"],
            "archived_lock_dir": str(archive),
            "archived_locks": {"overlay.run.lock": _sha(payload)},
            "live_locks_must_be_absent": [],
        },
    }
    with pytest.raises(recovery.RecoveryError, match="incomplete"):
        recovery.verify_oom_evidence(
            contract,
            workspace=tmp_path,
            journal_text="no OOM here",
            process_alive=lambda _pid: False,
            hostname="test-host",
        )


def test_foreign_or_duplicate_dangling_intent_is_rejected(
    tmp_path: Path,
) -> None:
    contract, out, prompt_map, evidence_sha = _fixture(tmp_path)
    foreign = copy.deepcopy(contract["expected_dangling_intents"][0])
    foreign["dispatch_id"] = "foreign-dispatch"
    foreign["attempt_id"] = "foreign-attempt"
    with (out / "dispatches64.jsonl").open("ab") as handle:
        handle.write(
            _line(
                {
                    "schema": "deepseek-64k-dispatch-v1",
                    "record_type": "dispatch_intent",
                    **foreign,
                }
            )
        )
    with pytest.raises(
        recovery.RecoveryError, match="non-settlement|exact sealed"
    ):
        _recover(contract, out, prompt_map, evidence_sha)

    contract2, out2, prompt_map2, evidence_sha2 = _fixture(tmp_path / "two")
    baseline = (out2 / "dispatches64.jsonl").read_bytes()
    (out2 / "dispatches64.jsonl").write_bytes(baseline + baseline)
    contract2["baseline_journals"]["dispatches64.jsonl"] = {
        "lines": 2,
        "bytes": len(baseline) * 2,
        "sha256": _sha(baseline + baseline),
    }
    with pytest.raises(recovery.RecoveryError, match="duplicate"):
        _recover(contract2, out2, prompt_map2, evidence_sha2)


def test_recovery_is_idempotent(tmp_path: Path) -> None:
    contract, out, prompt_map, evidence_sha = _fixture(tmp_path)
    _recover(contract, out, prompt_map, evidence_sha)
    before_attempts = (out / "attempts64.jsonl").read_bytes()
    before_dispatches = (out / "dispatches64.jsonl").read_bytes()
    _recover(
        contract,
        out,
        prompt_map,
        evidence_sha,
        next_attempt=1,
    )
    assert (out / "attempts64.jsonl").read_bytes() == before_attempts
    assert (out / "dispatches64.jsonl").read_bytes() == before_dispatches
