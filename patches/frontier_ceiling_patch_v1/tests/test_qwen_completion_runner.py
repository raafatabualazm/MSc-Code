from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from openai import OpenAI


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import frontier_core as core
import frontier_passk_qwen_completion as qwen_entry


MODEL = "qwen3.7-max-2026-05-17"


def args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "provider": "qwen",
        "model": MODEL,
        "max_output_tokens": 12_288,
        "budget": 0,
        "extra_body": {
            "enable_thinking": True,
            "thinking_budget": 8_192,
        },
        "temperature": 0.8,
        "top_p": 0.95,
        "timeout_seconds": 1_800,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return object()


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def response(completion_tokens: int) -> dict[str, object]:
    return {
        "id": "resp-qwen-contract",
        "model": MODEL,
        "created": 123,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "dynamic fn0() => 7;",
                    "reasoning_content": "reason",
                    "refusal": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "total_tokens": 100 + completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": min(8_192, completion_tokens),
            },
        },
    }


def test_request_uses_only_max_completion_tokens() -> None:
    client = FakeClient()
    messages = [{"role": "user", "content": "sealed prompt"}]
    qwen_entry.make_qwen_completion_request(
        client,
        args(),
        messages,
        requested_max_tokens=12_288,
    )
    [request] = client.chat.completions.calls
    assert request["max_completion_tokens"] == 12_288
    assert "max_tokens" not in request
    assert request["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 8_192,
    }
    assert request["model"] == MODEL
    assert request["messages"] is messages


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "deepseek"},
        {"model": "qwen3.8-max-preview"},
        {"max_output_tokens": 12_289},
        {"budget": 1},
        {"extra_body": {"enable_thinking": False, "thinking_budget": 8_192}},
        {"extra_body": {"enable_thinking": True, "thinking_budget": 8_191}},
        {
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget": 8_192,
                "max_tokens": 99,
            }
        },
        {
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget": 8_192,
                "max_completion_tokens": 99,
            }
        },
        {
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget": 8_192,
                "unsealed_option": True,
            }
        },
    ],
)
def test_request_contract_rejects_policy_drift(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(core.PreflightError):
        qwen_entry.request_contract(args(**overrides))


def test_request_rejects_runner_cap_disagreement_before_call() -> None:
    client = FakeClient()
    with pytest.raises(core.PreflightError):
        qwen_entry.make_qwen_completion_request(
            client,
            args(),
            [{"role": "user", "content": "sealed prompt"}],
            requested_max_tokens=12_287,
        )
    assert client.chat.completions.calls == []


def test_config_hash_input_binds_entry_and_request_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = {
        "runtime_identity": {"runner_sha256": "a" * 64},
        "slot_policy": {"schema": "fixed-cap-exact-response-slot-v1"},
        "slot_policy_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        qwen_entry,
        "_BASE_CONFIG_FOR_HASH",
        lambda _args: copy.deepcopy(base_config),
    )
    config = qwen_entry.qwen_config_for_hash(args())
    assert (
        config["runtime_identity"]["qwen_completion_entry_sha256"]
        == core.sha256_file(qwen_entry.ENTRY_PATH)
    )
    assert config["qwen_request_contract"]["request_cap_parameter"] == (
        "max_completion_tokens"
    )
    assert config["slot_policy"]["max_tokens_absent"] is True
    assert config["slot_policy"]["total_completion_cap"] == 12_288
    assert config["slot_policy"]["provider_completion_tolerance"] == 10
    assert config["slot_policy"]["completion_usage_validation_cap"] == 12_298
    assert (
        config["slot_policy"][
            "reasoning_tokens_usage_required_positive_and_bounded"
        ]
        is True
    )
    assert config["slot_policy"]["reasoning_content_required_nonempty"] is True
    assert config["slot_policy"]["exact_extra_body_keys"] == [
        "enable_thinking",
        "thinking_budget",
    ]
    assert config["slot_policy"]["finite_runner_budget_forbidden"] is True
    assert config["slot_policy_sha256"] == core.stable_sha256(
        config["slot_policy"]
    )


def test_terminal_usage_enforces_documented_tolerance_and_preserves_split() -> None:
    accepted = qwen_entry.classify_qwen_terminal_response(
        response(12_298),
        expected_model=MODEL,
        max_prompt_tokens=12_000,
        requested_max_tokens=12_288,
    )
    assert accepted.usage["completion_tokens"] == 12_298
    assert accepted.usage["reasoning_tokens"] == 8_192
    assert accepted.usage["answer_tokens"] == 4_106
    with pytest.raises(core.ResponseContractError):
        qwen_entry.classify_qwen_terminal_response(
            response(12_299),
            expected_model=MODEL,
            max_prompt_tokens=12_000,
            requested_max_tokens=12_288,
        )


def test_terminal_usage_requires_bounded_reasoning_detail() -> None:
    missing = response(10_000)
    missing["usage"].pop("completion_tokens_details")
    with pytest.raises(core.ResponseContractError):
        qwen_entry.classify_qwen_terminal_response(
            missing,
            expected_model=MODEL,
            max_prompt_tokens=12_000,
            requested_max_tokens=12_288,
        )

    zero_reasoning = response(10_000)
    zero_reasoning["usage"]["completion_tokens_details"]["reasoning_tokens"] = 0
    with pytest.raises(core.ResponseContractError):
        qwen_entry.classify_qwen_terminal_response(
            zero_reasoning,
            expected_model=MODEL,
            max_prompt_tokens=12_000,
            requested_max_tokens=12_288,
        )

    no_reasoning_content = response(10_000)
    no_reasoning_content["choices"][0]["message"]["reasoning_content"] = ""
    with pytest.raises(core.ResponseContractError):
        qwen_entry.classify_qwen_terminal_response(
            no_reasoning_content,
            expected_model=MODEL,
            max_prompt_tokens=12_000,
            requested_max_tokens=12_288,
        )


def test_openai_mock_transport_wire_json_has_only_completion_cap() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "resp-wire",
                "object": "chat.completion",
                "created": 123,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "dynamic fn0() => 7;",
                            "reasoning_content": "reason",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        with OpenAI(
            api_key="test-key",
            base_url="https://qwen.invalid/v1",
            http_client=http_client,
        ) as client:
            qwen_entry.make_qwen_completion_request(
                client,
                args(),
                [{"role": "user", "content": "sealed prompt"}],
                requested_max_tokens=12_288,
            )
    [wire] = captured
    assert wire["max_completion_tokens"] == 12_288
    assert "max_tokens" not in wire
    assert "extra_body" not in wire
    assert wire["enable_thinking"] is True
    assert wire["thinking_budget"] == 8_192

    over_budget = response(10_000)
    over_budget["usage"]["completion_tokens_details"]["reasoning_tokens"] = 8_193
    with pytest.raises(core.ResponseContractError):
        qwen_entry.classify_qwen_terminal_response(
            over_budget,
            expected_model=MODEL,
            max_prompt_tokens=12_000,
            requested_max_tokens=12_288,
        )
