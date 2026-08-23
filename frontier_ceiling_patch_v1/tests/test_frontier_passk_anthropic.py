from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frontier_passk_anthropic as anthro
from frontier_core import ResponseContractError, classify_terminal_provider_response


def _native_message(
    *,
    response_id: str = "msg_test_1",
    model: str = anthro.MODEL_ID,
    stop_reason: str = "end_turn",
    stop_details: object | None = None,
) -> dict:
    result = {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "thinking",
                "thinking": "private scratch",
                "signature": "signed",
            },
            {
                "type": "text",
                "text": "```dart\nint fn0(int x) => x + 1;\n```",
            },
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 123, "output_tokens": 456},
    }
    if stop_details is not None:
        result["stop_details"] = stop_details
    return result


def test_normalize_native_message_preserves_native_and_usage() -> None:
    native = _native_message()
    result = anthro.normalize_anthropic_response(native)

    assert result["id"] == "msg_test_1"
    assert result["model"] == anthro.MODEL_ID
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["message"]["reasoning_content"] == "private scratch"
    assert result["usage"] == {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579,
    }
    assert result["provider_native_response"] == native


def test_normalize_maps_max_tokens_to_length() -> None:
    result = anthro.normalize_anthropic_response(
        _native_message(stop_reason="max_tokens")
    )
    assert result["choices"][0]["finish_reason"] == "length"


def test_normalize_preserves_refusal_compatibility_and_native_category() -> None:
    result = anthro.normalize_anthropic_response(
        _native_message(
            stop_reason="refusal",
            stop_details={"type": "refusal", "reason": "cyber"},
        )
    )

    assert result["choices"][0]["finish_reason"] == "content_filter"
    assert result["choices"][0]["message"]["refusal"] == "anthropic_refusal"
    assert result["native_stop_reason"] == "refusal"
    assert result["native_stop_details_category"] == "cyber"
    assert result["native_stop_category"] == "refusal:cyber"


def test_refusal_without_provider_category_is_explicitly_unspecified() -> None:
    result = anthro.normalize_anthropic_response(_native_message(stop_reason="refusal"))
    assert result["choices"][0]["finish_reason"] == "content_filter"
    assert result["native_stop_details_category"] is None
    assert result["native_stop_category"] == "refusal:unspecified"


def _terminal_record(
    task_id: str,
    sample_index: int,
    *,
    stop_reason: str,
    stop_details: object | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "sample_index": sample_index,
        # Exercise reconstruction from the preserved normalized response, as
        # required for durable journals written before additive stop fields.
        "response": anthro.normalize_anthropic_response(
            _native_message(
                response_id=f"msg_{task_id}_{sample_index}",
                stop_reason=stop_reason,
                stop_details=stop_details,
            )
        ),
    }


def test_refusal_dominated_metric_is_not_reported_as_capability_ceiling() -> None:
    records = [
        _terminal_record(
            "t0",
            0,
            stop_reason="refusal",
            stop_details={"category": "cyber"},
        ),
        _terminal_record("t1", 0, stop_reason="refusal"),
        _terminal_record("t2", 0, stop_reason="refusal"),
        _terminal_record("t3", 0, stop_reason="end_turn"),
    ]
    report = anthro.anthropic_metric_transparency(
        records,
        task_ids=["t0", "t1", "t2", "t3"],
        k=1,
        complete=True,
    )

    stops = report["anthropic_native_stop_report"]
    assert stops["native_stop_reason_counts"] == {
        "end_turn": 1,
        "refusal": 3,
    }
    assert stops["native_stop_category_counts"] == {
        "end_turn": 1,
        "refusal:cyber": 1,
        "refusal:unspecified": 2,
    }
    assert stops["refusal_dominated"] is True

    assessment = report["capability_metric_assessment"]
    assert assessment["status"] == "invalid_refusal_dominated"
    assert assessment["capability_metric_valid"] is False
    assert assessment["valid_as_capability_ceiling"] is False
    assert assessment["ceiling_claim_allowed"] is False
    assert assessment["coverage"]["non_refusal_slots"] == 1
    assert assessment["coverage"]["non_refusal_slot_rate"] == pytest.approx(0.25)
    assert assessment["coverage"]["tasks_with_all_slots_refused"] == 3


def test_any_refusal_invalidates_ceiling_but_no_refusals_is_valid() -> None:
    one_refusal = [
        _terminal_record("t0", 0, stop_reason="refusal"),
        _terminal_record("t1", 0, stop_reason="end_turn"),
        _terminal_record("t2", 0, stop_reason="max_tokens"),
    ]
    invalid = anthro.anthropic_metric_transparency(
        one_refusal,
        task_ids=["t0", "t1", "t2"],
        k=1,
        complete=True,
    )["capability_metric_assessment"]
    assert invalid["status"] == "invalid_refusal_present"
    assert invalid["valid_as_capability_ceiling"] is False

    refusal_free = [
        _terminal_record("t0", 0, stop_reason="end_turn"),
        _terminal_record("t1", 0, stop_reason="max_tokens"),
        _terminal_record("t2", 0, stop_reason="end_turn"),
    ]
    valid = anthro.anthropic_metric_transparency(
        refusal_free,
        task_ids=["t0", "t1", "t2"],
        k=1,
        complete=True,
    )["capability_metric_assessment"]
    assert valid["status"] == "valid"
    assert valid["valid_as_capability_ceiling"] is True
    # A max-token stop is an observed fixed-cap capability outcome, not a
    # policy refusal, so it does not invalidate the ceiling interpretation.
    assert valid["coverage"]["non_refusal_slots"] == 3


def test_incomplete_progress_is_not_a_capability_ceiling() -> None:
    report = anthro.anthropic_metric_transparency(
        [_terminal_record("t0", 0, stop_reason="end_turn")],
        task_ids=["t0", "t1"],
        k=1,
        complete=False,
    )
    assert report["capability_metric_assessment"]["status"] == "invalid_incomplete"
    assert report["capability_metric_assessment"]["ceiling_claim_allowed"] is False


def test_audited_classifier_enforces_exact_model_and_provider_usage() -> None:
    normalized = anthro.normalize_anthropic_response(_native_message())
    terminal = classify_terminal_provider_response(
        normalized,
        expected_model=anthro.MODEL_ID,
        max_prompt_tokens=12_000,
        requested_max_tokens=anthro.MAX_OUTPUT_TOKENS,
    )
    assert terminal.response_model == anthro.MODEL_ID
    assert terminal.usage["total_tokens"] == 579

    normalized["model"] = "claude-sonnet-5-unexpected-alias"
    with pytest.raises(ResponseContractError, match="does not equal requested"):
        classify_terminal_provider_response(
            normalized,
            expected_model=anthro.MODEL_ID,
            max_prompt_tokens=12_000,
            requested_max_tokens=anthro.MAX_OUTPUT_TOKENS,
        )


def test_normalize_rejects_missing_usage() -> None:
    native = _native_message()
    del native["usage"]
    with pytest.raises(Exception, match="usage is missing"):
        anthro.normalize_anthropic_response(native)


def test_split_messages_moves_exact_system_to_top_level() -> None:
    system, messages = anthro._split_anthropic_messages(
        [
            {"role": "system", "content": "sealed system"},
            {"role": "user", "content": "sealed F2"},
        ]
    )
    assert system == "sealed system"
    assert messages == [{"role": "user", "content": "sealed F2"}]


def test_split_messages_rejects_second_system() -> None:
    with pytest.raises(Exception, match="system message"):
        anthro._split_anthropic_messages(
            [
                {"role": "system", "content": "one"},
                {"role": "user", "content": "question"},
                {"role": "system", "content": "two"},
            ]
        )


def test_transport_omits_sampling_and_uses_adaptive_effort(monkeypatch) -> None:
    captured: dict = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _native_message()

    class FakeClient:
        messages = Messages()

    class FakeAnthropicModule:
        @staticmethod
        def Anthropic(**kwargs):
            captured["client"] = kwargs
            return FakeClient()

    monkeypatch.setitem(__import__("sys").modules, "anthropic", FakeAnthropicModule)
    args = SimpleNamespace(
        model=anthro.MODEL_ID,
        max_output_tokens=anthro.MAX_OUTPUT_TOKENS,
        anthropic_effort="max",
        timeout_seconds=600,
    )
    transport = anthro.AnthropicMessagesTransport(
        args, "secret-test-key", anthro.DEFAULT_BASE_URL
    )
    response = transport.create(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ]
    )

    assert response["id"] == "msg_test_1"
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "max"}
    assert captured["max_tokens"] == 65_536
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured


def test_cost_estimate_uses_provider_usage_only() -> None:
    rows = [
        {
            "response_received": True,
            "budget_charge_tokens": 300,
            "usage": {"prompt_tokens": 100, "completion_tokens": 200},
        },
        {
            "response_received": False,
            "transport_retry": True,
            "budget_charge_tokens": 700,
            "usage": None,
        },
    ]
    cost = anthro.cost_estimate_from_attempts(rows)
    assert cost["provider_reported_prompt_tokens"] == 100
    assert cost["provider_reported_completion_tokens"] == 200
    assert cost["unknown_billing_transport_attempts"] == 1
    assert cost["conservative_budget_charge_tokens"] == 1000
    assert cost["estimated_total_usd"] == pytest.approx(0.0022)


def test_api_provenance_never_contains_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-value")
    args = SimpleNamespace(
        model=anthro.MODEL_ID,
        anthropic_effort="high",
        max_output_tokens=anthro.MAX_OUTPUT_TOKENS,
    )
    serialized = json.dumps(
        anthro._api_provenance(args, anthro.DEFAULT_BASE_URL),
        sort_keys=True,
    )
    assert "super-secret-value" not in serialized
    assert "ANTHROPIC_API_KEY environment variable" in serialized


def test_fixed_slot_policy_binds_dispatch_but_not_tranche() -> None:
    args = SimpleNamespace(
        model=anthro.MODEL_ID,
        k=10,
        max_output_tokens=anthro.MAX_OUTPUT_TOKENS,
        max_prompt_tokens=12_000,
        temperature=1.0,
        top_p=1.0,
        extra_body={},
        timeout_seconds=600,
        max_attempts_per_sample=6,
        anthropic_effort="max",
    )
    policy = anthro.fixed_slot_policy(args)
    assert policy["dispatch_order"] == "sample_index_then_sealed_task_order"
    assert policy["thinking"] == {"type": "adaptive"}
    assert policy["max_new_terminal_slots_is_resume_operational_only"] is True
    assert 20 not in policy.values()
