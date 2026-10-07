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
import qwen37_capacity_fallback_v6 as capacity
import qwen37_capacity_length_repair_v7 as repair


MODEL = "qwen3.7-max-2026-05-20"
CONTRACT_SHA = "a" * 64


def raw_response(
    *,
    finish_reason: str = "length",
    completion_tokens: int = 12_298,
    reasoning_tokens: int = 8_192,
) -> dict[str, object]:
    return {
        "id": f"response-{finish_reason}",
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


def feed_row(*, finish_reason: str = "length") -> dict[str, object]:
    qwen_entry.install_qwen_completion_policy()
    raw = raw_response(finish_reason=finish_reason)
    terminal = qwen_entry.classify_qwen_terminal_response(
        raw,
        expected_model=MODEL,
        max_prompt_tokens=12_000,
        requested_max_tokens=12_288,
    )
    row: dict[str, object] = {
        "schema": capacity.SCHEMA,
        "record_type": "capacity_effective_terminal_feed",
        "source_kind": "capacity_v6",
        "selection_id": f"selection-{finish_reason}",
        "overlay_contract_sha256": CONTRACT_SHA,
        "parent_contract_sha256": "b" * 64,
        "overlay_config_sha256": "c" * 64,
        "arm": "opus",
        "pair_status": "paired_missing",
        "originating_shard_key": "base_0517_k3",
        "originating_source_directory": "/workspace/source",
        "originating_source_config_sha256": "d" * 64,
        "originating_source_slot_policy_sha256": "e" * 64,
        "originating_local_sample_index": 0,
        "global_sample_index": 0,
        "task_id": "task",
        "prompt_sha256": "f" * 64,
        "effective_origin": "fresh_capacity_alias_response",
        "effective_source_directory": "/workspace/capacity",
        "effective_source_config_sha256": "1" * 64,
        "effective_source_slot_policy_sha256": "2" * 64,
        "effective_endpoint_sha256": "3" * 64,
        "capacity_epoch": "epoch",
        "effective_attempt_id": "attempt",
        "response_id": terminal.response_id,
        "requested_model": MODEL,
        "resolved_model": MODEL,
        "finish_reason": terminal.finish_reason,
        "candidate_valid": terminal.candidate_valid,
        "terminal_reason": terminal.terminal_reason,
        "code_sha256": terminal.code_sha256,
        "validated_usage": terminal.usage,
        "reasoning_content": terminal.reasoning_content,
        "content": terminal.content,
        "raw_response": raw,
        "effective_terminal_canonical_row_sha256": "4" * 64,
        "request_max_completion_tokens": 12_288,
        "thinking_budget": 8_192,
        "selection_reads_outcomes": False,
        "published_at": "2026-01-01T00:00:00Z",
    }
    immutable = dict(row)
    immutable.pop("published_at")
    row["terminal_feed_payload_sha256"] = capacity.canonical_sha(immutable)
    return row


def test_repair_wire_uses_doubled_completion_cap_only() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            request=request,
            json=raw_response(
                finish_reason="stop",
                completion_tokens=20_000,
            ),
        )

    client = OpenAI(
        api_key="test",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    repair.make_repair_request(
        client,
        model=MODEL,
        messages=[{"role": "user", "content": "sealed"}],
    )
    [wire] = captured
    assert wire["max_completion_tokens"] == 24_576
    assert "max_tokens" not in wire
    assert wire["enable_thinking"] is True
    assert wire["thinking_budget"] == 8_192
    assert "extra_body" not in wire


def test_repair_classifier_enforces_tolerance_and_reasoning() -> None:
    terminal = repair.classify_repair_response(
        raw_response(
            finish_reason="stop",
            completion_tokens=24_586,
        ),
        expected_model=MODEL,
    )
    assert terminal.usage["answer_tokens"] == 16_394
    with pytest.raises(core.ResponseContractError):
        repair.classify_repair_response(
            raw_response(
                finish_reason="stop",
                completion_tokens=24_587,
            ),
            expected_model=MODEL,
        )
    with pytest.raises(core.ResponseContractError):
        repair.classify_repair_response(
            raw_response(
                finish_reason="stop",
                completion_tokens=10_000,
                reasoning_tokens=0,
            ),
            expected_model=MODEL,
        )


def test_feed_validation_reclassifies_raw_terminal() -> None:
    row = feed_row()
    terminal = repair.validate_feed_terminal(
        row,
        expected_capacity_contract_sha256=CONTRACT_SHA,
    )
    assert terminal.finish_reason == "length"
    row["validated_usage"] = dict(row["validated_usage"])
    row["validated_usage"]["answer_tokens"] += 1
    with pytest.raises(repair.RepairError):
        repair.validate_feed_terminal(
            row,
            expected_capacity_contract_sha256=CONTRACT_SHA,
        )


def test_scan_is_outcome_blind_and_selects_only_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    length = feed_row()
    stop = feed_row(finish_reason="stop")
    stop["global_sample_index"] = 1
    immutable = dict(stop)
    immutable.pop("terminal_feed_payload_sha256")
    immutable.pop("published_at")
    stop["terminal_feed_payload_sha256"] = capacity.canonical_sha(immutable)
    feed = tmp_path / "effective_terminal_feed.jsonl"
    feed.write_text(
        json.dumps(length) + "\n" + json.dumps(stop) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        repair,
        "source_material",
        lambda _row: ([{"role": "user", "content": "sealed"}], "tests"),
    )
    selected = repair.scan_capacity_length_sources(
        feed,
        expected_capacity_contract_sha256=CONTRACT_SHA,
    )
    assert len(selected) == 1
    assert selected[0]["feed"]["finish_reason"] == "length"
    assert not any("outcome" in key for key in selected[0])


def test_capacity_source_key_binds_feed_payload() -> None:
    row = feed_row()
    baseline = repair.capacity_source_key(row)
    row["terminal_feed_payload_sha256"] = "0" * 64
    assert repair.capacity_source_key(row) != baseline


def test_source_material_uses_hash_sealed_eval_not_public_tasks(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    source = run_root / "source"
    source.mkdir(parents=True)
    eval_path = tmp_path / "eval.jsonl"
    task_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    for index in range(capacity.EXPECTED_TASKS):
        task_id = f"task_{index:03d}"
        tests = f"void main() {{ assert(fn{index}() == {index}); }}"
        task_rows.append(
            {
                "task_id": task_id,
                "tests_equal_acceptance_tests": True,
                "tests_sha256": runner.sha256_text(tests),
                "acceptance_tests_sha256": runner.sha256_text(tests),
            }
        )
        eval_rows.append(
            {
                "task_id": task_id,
                "tests": tests,
                "acceptance_tests": tests,
            }
        )
    (source / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in task_rows),
        encoding="utf-8",
    )
    eval_path.write_text(
        "".join(json.dumps(row) + "\n" for row in eval_rows),
        encoding="utf-8",
    )
    prompt_sha = "f" * 64
    (source / "prompts.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task_000",
                "prompt_sha256": prompt_sha,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "sealed"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "slot_policy_sha256": "e" * 64,
        "sealed_inputs": {
            "eval_jsonl": str(eval_path.resolve()),
            "eval_jsonl_sha256": runner.sha256_file(eval_path),
        },
    }
    (source / "provenance.json").write_text(
        json.dumps({"config": config}) + "\n",
        encoding="utf-8",
    )
    row = feed_row()
    row.update(
        {
            "originating_source_directory": str(source.resolve()),
            "originating_source_config_sha256": runner.stable_sha256(config),
            "originating_source_slot_policy_sha256": "e" * 64,
            "task_id": "task_000",
            "prompt_sha256": prompt_sha,
        }
    )
    capacity._TEST_BUNDLE_CACHE.clear()
    messages, tests = repair.source_material(row, run_root=run_root)
    assert messages[1]["content"] == "sealed"
    assert tests == "void main() { assert(fn0() == 0); }"
    assert "acceptance_tests" not in task_rows[0]


def test_prior_epoch_accepts_only_exact_quota_boundary(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    row = {
        "schema": repair.ATTEMPT_SCHEMA,
        "config_sha256": "a" * 64,
        "capacity_source_key": "source",
        "attempt_index": 0,
        "response_received": False,
        "retryable_transport": False,
        "transport_error": (
            "api_exception:PermissionDeniedError:Error code: 403 - "
            "AllocationQuota.FreeTierOnly: The free quota has been exhausted"
        ),
        "response": None,
        "usage": None,
    }
    attempts.write_text(json.dumps(row) + "\n", encoding="utf-8")
    outcomes.write_text("", encoding="utf-8")
    terminal, loaded_outcomes = repair.load_existing(
        attempts,
        outcomes,
        config_sha256="a" * 64,
        allow_sealed_quota_boundary=True,
    )
    assert terminal == {}
    assert loaded_outcomes == {}
    with pytest.raises(repair.RepairError):
        repair.load_existing(
            attempts,
            outcomes,
            config_sha256="a" * 64,
            allow_sealed_quota_boundary=False,
        )
    row["transport_error"] = (
        "api_exception:RateLimitError:Error code: 429 - rate limit"
    )
    attempts.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(repair.RepairError):
        repair.load_existing(
            attempts,
            outcomes,
            config_sha256="a" * 64,
            allow_sealed_quota_boundary=True,
        )
