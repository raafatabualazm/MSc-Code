from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"
sys.path.insert(0, str(PATCH))

import deepseek64_continuation_v1 as continuation
import frontier_passk as runner


def _raw(value: dict, *, spacing: bool = False) -> bytes:
    if spacing:
        return (json.dumps(value, separators=(", ", ": ")) + "\n").encode()
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _snapshot(
    tmp_path: Path,
    *,
    with_source_outcome: bool,
) -> tuple[continuation.SourceSnapshot, bytes, bytes | None]:
    source_root = tmp_path / "source"
    out_root = tmp_path / "overlay"
    source_root.mkdir()
    out_root.mkdir()
    task_id = "task-a"
    terminal = {
        "attempt_id": "source-attempt",
        "response_id": "source-response",
        "finish_reason": "stop",
        "candidate_valid": True,
        "terminal_reason": "candidate_valid",
        "code_sha256": "a" * 64,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
    attempt_raw = _raw(terminal, spacing=True)
    attempt_row = continuation.RawRow(value=terminal, raw=attempt_raw)
    outcome = {
        "task_id": task_id,
        "sample_index": 0,
        "attempt_id": "source-attempt",
        "compiled": True,
        "passed": False,
    }
    outcome_raw = _raw(outcome, spacing=True) if with_source_outcome else None
    outcome_rows = (
        [continuation.RawRow(value=outcome, raw=outcome_raw)]
        if outcome_raw is not None
        else []
    )
    policy = {
        "k": 2,
        "source_max_output_tokens": 131072,
        "continuation_max_output_tokens": 65536,
        "evaluator_sha256": "e" * 64,
        "requested_model": "deepseek-v4-pro",
    }
    source_outcomes = (
        {(task_id, 0, "source-attempt"): outcome}
        if with_source_outcome
        else {}
    )
    source_outcome_raw = (
        {(task_id, 0, "source-attempt"): outcome_raw}
        if outcome_raw is not None
        else {}
    )
    snapshot = continuation.SourceSnapshot(
        arm="opus",
        source_root=source_root,
        out_root=out_root,
        provenance={},
        prompts=[],
        tasks=[
            continuation.RawRow(
                value={"task_id": task_id},
                raw=b'{"task_id":"task-a"}\n',
            )
        ],
        prompt_map={
            task_id: {
                "prompt_sha256": "p" * 64,
                "messages": [{"role": "user", "content": "exact prompt"}],
            }
        },
        eval_map={task_id: {"acceptance_tests": "test"}},
        expected_slots={(task_id, 0), (task_id, 1)},
        source_terminal={(task_id, 0): terminal},
        source_terminal_raw={(task_id, 0): attempt_raw},
        source_outcomes=source_outcomes,
        source_outcome_raw=source_outcome_raw,
        source_attempt_rows=[attempt_row],
        source_outcome_rows=outcome_rows,
        overlay_config_sha256="c" * 64,
        overlay_slot_policy={},
        overlay_slot_policy_sha256="s" * 64,
        arm_contract={
            "source_config_sha256": "b" * 64,
            "files": {},
        },
        policy=policy,
        contract_sha256="d" * 64,
    )
    return snapshot, attempt_raw, outcome_raw


def _request_args() -> SimpleNamespace:
    return SimpleNamespace(
        provider="deepseek",
        model="deepseek-v4-pro",
        k=2,
        workers=1,
        max_prompt_tokens=12000,
        max_output_tokens=65536,
        max_attempts_per_sample=6,
        retry_base_seconds=0.0,
        retry_max_seconds=0.0,
        eval_stability_runs=2,
        eval_timeout_seconds=30,
        temperature=0.8,
        top_p=0.95,
        timeout_seconds=7200,
        extra_body={},
        deepseek_env_file=Path("unused"),
        api_key="",
        base_url="",
    )


def test_paid_mode_requires_exact_confirmation() -> None:
    with pytest.raises(SystemExit):
        continuation.parse_args(["--arm", "opus", "--mode", "run"])
    args = continuation.parse_args(
        [
            "--arm",
            "opus",
            "--mode",
            "run",
            "--paid-confirmation",
            continuation.PAID_CONFIRMATION,
        ]
    )
    assert args.mode == "run"


def test_default_contract_seals_both_stopped_snapshots() -> None:
    contract, _digest = continuation.load_contract(continuation.DEFAULT_CONTRACT)
    assert contract["policy"]["source_max_output_tokens"] == 131072
    assert contract["policy"]["continuation_max_output_tokens"] == 65536
    assert contract["arms"]["opus"]["expected_snapshot_counts"] == {
        "terminal_slots": 976,
        "source_outcomes": 930,
        "terminal_without_outcome": 46,
        "missing_slots": 774,
    }
    assert contract["arms"]["codex"]["expected_snapshot_counts"] == {
        "terminal_slots": 1014,
        "source_outcomes": 970,
        "terminal_without_outcome": 44,
        "missing_slots": 736,
    }


def test_overlay_policy_changes_cap_without_mutating_source() -> None:
    source = {
        "config": {
            "slot_policy": {
                "schema": runner.FIXED_SLOT_POLICY_SCHEMA,
                "requested_model": "deepseek-v4-pro",
                "k": 10,
                "fixed_max_output_tokens": 131072,
                "max_prompt_tokens": 12000,
                "temperature": 0.8,
                "top_p": 0.95,
                "request_timeout_seconds": 7200,
                "max_transport_attempts_per_slot": 6,
                "finish_reason_length_consumes_slot": True,
            }
        }
    }
    policy = {
        "requested_model": "deepseek-v4-pro",
        "k": 10,
        "source_max_output_tokens": 131072,
        "continuation_max_output_tokens": 65536,
        "max_prompt_tokens": 12000,
        "temperature": 0.8,
        "top_p": 0.95,
        "timeout_seconds": 7200,
        "max_attempts_per_slot": 6,
    }
    overlay, digest = continuation.overlay_policy(source, policy)
    assert source["config"]["slot_policy"]["fixed_max_output_tokens"] == 131072
    assert overlay["fixed_max_output_tokens"] == 65536
    assert overlay["finish_reason_length_consumes_slot"] is True
    assert digest == runner.stable_sha256(overlay)


def test_raw_jsonl_round_trips_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    payload = b'{"z": 1, "a": "unicode \\u03bb"}\n'
    path.write_bytes(payload)
    rows = continuation.load_raw_jsonl(path, "fixture")
    assert len(rows) == 1
    assert rows[0].raw == payload
    out = tmp_path / "copy.jsonl"
    continuation.atomic_write_raw(out, [rows[0].raw])
    assert out.read_bytes() == payload


def test_effective_feed_copies_source_lines_byte_for_byte(tmp_path: Path) -> None:
    snapshot, attempt_raw, outcome_raw = _snapshot(
        tmp_path, with_source_outcome=True
    )
    status = continuation.build_effective(snapshot, _request_args())
    assert status["counts"]["effective_terminals"] == 1
    assert status["counts"]["effective_outcomes"] == 1
    assert status["counts"]["missing_terminals"] == 1
    assert (snapshot.out_root / "effective_attempts.jsonl").read_bytes() == attempt_raw
    assert (snapshot.out_root / "effective_outcomes.jsonl").read_bytes() == outcome_raw
    index = runner.load_jsonl(
        snapshot.out_root / "effective_slot_index.jsonl", "index"
    )[0]
    assert index["attempt_origin_file"] == str(
        snapshot.source_root / "attempts.jsonl"
    )
    assert index["attempt_origin_line"] == 1
    assert index["outcome_origin_line"] == 1
    assert index["stratum"] == "source_128k"


def test_terminal_presence_not_outcome_presence_selects_completion(
    tmp_path: Path,
) -> None:
    snapshot, attempt_raw, _ = _snapshot(
        tmp_path, with_source_outcome=False
    )
    status = continuation.build_effective(snapshot, _request_args())
    assert status["counts"] == {
        "expected_slots": 2,
        "source_128k_terminals": 1,
        "continuation_64k_terminals": 0,
        "effective_terminals": 1,
        "effective_outcomes": 0,
        "missing_terminals": 1,
        "terminal_without_outcome": 1,
    }
    assert (snapshot.out_root / "effective_attempts.jsonl").read_bytes() == attempt_raw
    assert (snapshot.out_root / "effective_outcomes.jsonl").read_bytes() == b""


def test_dangling_dispatch_intent_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.jsonl"
    row = {
        "schema": continuation.DISPATCH_SCHEMA,
        "record_type": "dispatch_intent",
        "config_sha256": "c" * 64,
        "dispatch_id": "dispatch-1",
        "task_id": "t",
        "sample_index": 0,
        "attempt_index": 0,
        "attempt_id": "attempt-1",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(
        continuation.ContinuationError, match="possible duplicate provider call"
    ):
        continuation.load_dispatches(path, config_sha256="c" * 64)


def test_source_hash_change_is_rejected(tmp_path: Path) -> None:
    snapshot, _attempt, _outcome = _snapshot(
        tmp_path, with_source_outcome=False
    )
    source_file = snapshot.source_root / "attempts.jsonl"
    source_file.write_text("changed\n", encoding="utf-8")
    snapshot.arm_contract["files"] = {"attempts.jsonl": "0" * 64}
    with pytest.raises(continuation.ContinuationError, match="hash mismatch"):
        continuation.verify_source_still_frozen(snapshot)


def test_provider_dispatches_only_missing_slot_at_exact_64k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _attempt, _outcome = _snapshot(
        tmp_path, with_source_outcome=False
    )
    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "id": "continuation-response",
                "model": "deepseek-v4-pro",
                "created": 123,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "",
                            "reasoning_content": "reason",
                            "refusal": None,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

    class OpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setattr(
        continuation.runner,
        "api_credentials",
        lambda _args: ("secret-not-persisted", "https://example.invalid"),
    )
    monkeypatch.setattr(
        continuation,
        "import_evaluator",
        lambda *_args, **_kwargs: (
            lambda *_a, **_k: None,
            {
                "sha256": snapshot.policy["evaluator_sha256"],
                "entrypoint": "evaluate_dart_jit_tests_detail",
            },
        ),
    )
    args = _request_args()
    continuation.run_provider_calls(snapshot, args)
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 65536
    assert calls[0]["model"] == "deepseek-v4-pro"
    assert calls[0]["messages"] == [{"role": "user", "content": "exact prompt"}]
    attempts = runner.load_jsonl(
        snapshot.out_root / "attempts64.jsonl", "64K attempts"
    )
    assert {(row["task_id"], row["sample_index"]) for row in attempts} == {
        ("task-a", 1)
    }
    # Resume sees the terminal slot and must not make a second provider call.
    continuation.run_provider_calls(snapshot, args)
    assert len(calls) == 1
    outcome_path = snapshot.out_root / "outcomes64.jsonl"
    outcome_rows = runner.load_jsonl(outcome_path, "64K outcomes")
    outcome_rows[0]["response_id"] = "tampered"
    runner.atomic_write_jsonl(outcome_path, outcome_rows)
    with pytest.raises(
        continuation.ContinuationError, match="outcome receipt mismatch"
    ):
        continuation.load_overlay_state(snapshot, args)
