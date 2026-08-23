from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
from openai import OpenAI


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import frontier_core as core
import frontier_passk as runner
import frontier_passk_qwen_completion as qwen_entry
import qwen37_length_repair_v5 as repair


MODEL = "qwen3.7-max-preview"


def response(
    *,
    completion_tokens: int = 24_586,
    reasoning_tokens: int = 8_192,
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "id": "repair-response",
        "model": MODEL,
        "created": 123,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": "dynamic fn0() => 7;",
                    "reasoning_content": "sealed reasoning",
                    "refusal": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "total_tokens": 100 + completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
        },
    }


def test_repair_wire_uses_only_doubled_completion_cap() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, request=request, json=response())

    client = OpenAI(
        api_key="test",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    repair.make_repair_request(
        client,
        model=MODEL,
        messages=[{"role": "user", "content": "sealed prompt"}],
    )
    [wire] = captured
    assert wire["max_completion_tokens"] == 24_576
    assert "max_tokens" not in wire
    assert wire["enable_thinking"] is True
    assert wire["thinking_budget"] == 8_192
    assert "extra_body" not in wire


def test_repair_usage_accepts_only_documented_tolerance() -> None:
    terminal = repair.classify_repair_response(
        response(),
        expected_model=MODEL,
    )
    assert terminal.usage["completion_tokens"] == 24_586
    assert terminal.usage["reasoning_tokens"] == 8_192
    assert terminal.usage["answer_tokens"] == 16_394
    with pytest.raises(core.ResponseContractError):
        repair.classify_repair_response(
            response(completion_tokens=24_587),
            expected_model=MODEL,
        )


@pytest.mark.parametrize("reasoning_tokens", [0, 8_193])
def test_repair_requires_bounded_reasoning_tokens(
    reasoning_tokens: int,
) -> None:
    with pytest.raises(core.ResponseContractError):
        repair.classify_repair_response(
            response(reasoning_tokens=reasoning_tokens),
            expected_model=MODEL,
        )


def test_source_slot_key_binds_every_source_identity_field() -> None:
    kwargs: dict[str, object] = {
        "shard_key": "base_preview_k2",
        "arm": "opus",
        "task_id": "task",
        "local_sample_index": 0,
        "global_sample_index": 3,
        "original_attempt_id": "attempt",
        "original_response_id": "response",
        "source_config_sha256": "a" * 64,
        "source_slot_policy_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "source_terminal_row_sha256": "d" * 64,
    }
    baseline = repair.source_slot_key(**kwargs)
    replacements: dict[str, object] = {
        "shard_key": "other",
        "arm": "codex",
        "task_id": "other",
        "local_sample_index": 1,
        "global_sample_index": 4,
        "original_attempt_id": "other",
        "original_response_id": "other",
        "source_config_sha256": "e" * 64,
        "source_slot_policy_sha256": "f" * 64,
        "prompt_sha256": "0" * 64,
        "source_terminal_row_sha256": "1" * 64,
    }
    for field, value in replacements.items():
        changed = dict(kwargs)
        changed[field] = value
        assert repair.source_slot_key(**changed) != baseline


def source_terminal_row() -> dict[str, object]:
    qwen_entry.install_qwen_completion_policy()
    raw = response(completion_tokens=100, reasoning_tokens=50)
    terminal = qwen_entry.classify_qwen_terminal_response(
        raw,
        expected_model=MODEL,
        max_prompt_tokens=12_000,
        requested_max_tokens=12_288,
    )
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "config_sha256": "a" * 64,
        "slot_policy_sha256": "b" * 64,
        "task_id": "terminal",
        "sample_index": 0,
        "attempt_index": 0,
        "prompt_sha256": "c" * 64,
        "requested_max_tokens": 12_288,
        "budget_charge_tokens": terminal.usage["total_tokens"],
        "response_received": True,
        "slot_terminal": True,
        "transport_retry": False,
        "transport_error": None,
        "fatal_response_contract": False,
        "response": raw,
        "response_id": terminal.response_id,
        "resolved_model": terminal.response_model,
        "response_created": terminal.response_created,
        "finish_reason": terminal.finish_reason,
        "candidate_valid": terminal.candidate_valid,
        "terminal_reason": terminal.terminal_reason,
        "content": terminal.content,
        "reasoning_content": terminal.reasoning_content,
        "code": terminal.code,
        "code_sha256": terminal.code_sha256,
        "usage": terminal.usage,
    }


def response_less_row() -> dict[str, object]:
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "config_sha256": "a" * 64,
        "slot_policy_sha256": "b" * 64,
        "task_id": "unresolved",
        "sample_index": 0,
        "attempt_index": 0,
        "prompt_sha256": "d" * 64,
        "requested_max_tokens": 12_288,
        "budget_charge_tokens": 24_288,
        "response_received": False,
        "slot_terminal": False,
        "candidate_valid": None,
        "terminal_reason": None,
        "response": None,
        "usage": None,
        "transport_retry": True,
        "retryable_transport": False,
        "fatal_response_contract": False,
        "transport_error": "api_exception:PermissionDeniedError:capacity",
    }


def test_source_scan_tolerates_but_never_selects_unresolved_403(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempts.jsonl"
    rows = [source_terminal_row(), response_less_row()]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    terminal = repair.load_source_terminals_outcome_blind(
        path,
        config_sha256="a" * 64,
        prompt_map={
            "terminal": {"prompt_sha256": "c" * 64},
            "unresolved": {"prompt_sha256": "d" * 64},
        },
        requested_model=MODEL,
        local_k=1,
        slot_policy_sha256="b" * 64,
        response_ids=set(),
    )
    assert set(terminal) == {("terminal", 0)}


def test_source_scan_rejects_malformed_response_less_row(
    tmp_path: Path,
) -> None:
    row = response_less_row()
    row["retryable_transport"] = None
    path = tmp_path / "attempts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(repair.RepairError):
        repair.load_source_terminals_outcome_blind(
            path,
            config_sha256="a" * 64,
            prompt_map={"unresolved": {"prompt_sha256": "d" * 64}},
            requested_model=MODEL,
            local_k=1,
            slot_policy_sha256="b" * 64,
            response_ids=set(),
        )
