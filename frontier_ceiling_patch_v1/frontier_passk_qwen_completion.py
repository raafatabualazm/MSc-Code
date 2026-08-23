#!/usr/bin/env python3
"""Qwen-only audited entry using the thinking-model completion-token cap.

The shared frontier runner sends ``max_tokens`` because that is the request
shape used by the DeepSeek runs. Alibaba's thinking-model contract instead
requires ``max_completion_tokens`` when reasoning and final-answer tokens must
share one hard cap. This entry leaves ``frontier_passk.py`` byte-identical,
replaces only its request constructor, and binds this entry and request policy
into the run configuration hash and provenance.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import frontier_passk as base


REQUEST_CONTRACT_SCHEMA = "qwen-thinking-max-completion-tokens-v1"
TOTAL_COMPLETION_CAP = 12_288
PROVIDER_COMPLETION_TOLERANCE = 10
COMPLETION_USAGE_VALIDATION_CAP = (
    TOTAL_COMPLETION_CAP + PROVIDER_COMPLETION_TOLERANCE
)
THINKING_BUDGET = 8_192
ALLOWED_MODELS = frozenset(
    {
        "qwen3.7-max-2026-05-17",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-preview",
    }
)
EXACT_EXTRA_BODY_KEYS = frozenset({"enable_thinking", "thinking_budget"})
ENTRY_PATH = Path(__file__).resolve()

_BASE_CONFIG_FOR_HASH = base.config_for_hash
_BASE_PREPARE_RUN = base.prepare_run
_BASE_CLASSIFY_TERMINAL_RESPONSE = base.classify_terminal_provider_response


def request_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.provider != "qwen":
        raise base.PreflightError("Qwen completion entry requires --provider qwen")
    if args.model not in ALLOWED_MODELS:
        raise base.PreflightError(
            "Qwen completion entry permits only the sealed primary models: "
            + ", ".join(sorted(ALLOWED_MODELS))
        )
    if args.max_output_tokens != TOTAL_COMPLETION_CAP:
        raise base.PreflightError(
            "Qwen completion entry requires --max-output-tokens "
            f"{TOTAL_COMPLETION_CAP}"
        )
    if args.budget != 0:
        raise base.PreflightError(
            "Qwen completion entry requires --budget 0 because the shared "
            "finite-budget reservation does not include provider tolerance"
        )
    extra_body = args.extra_body
    if not isinstance(extra_body, dict):
        raise base.PreflightError("Qwen thinking extra_body must be an object")
    if set(extra_body) != EXACT_EXTRA_BODY_KEYS:
        raise base.PreflightError(
            "Qwen thinking extra_body must contain exactly "
            "{enable_thinking, thinking_budget}"
        )
    conflicting_cap_fields = sorted(
        key
        for key in ("max_tokens", "max_completion_tokens")
        if key in extra_body
    )
    if conflicting_cap_fields:
        raise base.PreflightError(
            "Qwen thinking extra_body must not override completion caps: "
            + ", ".join(conflicting_cap_fields)
        )
    if extra_body.get("enable_thinking") is not True:
        raise base.PreflightError("Qwen completion entry requires enable_thinking=true")
    thinking_budget = extra_body.get("thinking_budget")
    if (
        not isinstance(thinking_budget, int)
        or isinstance(thinking_budget, bool)
        or thinking_budget != THINKING_BUDGET
    ):
        raise base.PreflightError(
            "Qwen completion entry requires thinking_budget="
            f"{THINKING_BUDGET}"
        )
    return {
        "schema": REQUEST_CONTRACT_SCHEMA,
        "provider": "qwen",
        "allowed_models": sorted(ALLOWED_MODELS),
        "request_cap_parameter": "max_completion_tokens",
        "forbidden_request_cap_parameter": "max_tokens",
        "total_completion_cap": TOTAL_COMPLETION_CAP,
        "provider_completion_tolerance": PROVIDER_COMPLETION_TOLERANCE,
        "completion_usage_validation_cap": COMPLETION_USAGE_VALIDATION_CAP,
        "thinking_budget": THINKING_BUDGET,
        "exact_extra_body_keys": sorted(EXACT_EXTRA_BODY_KEYS),
        "reasoning_and_final_share_completion_cap": True,
        "reasoning_tokens_usage_required_positive_and_bounded": True,
        "reasoning_content_required_nonempty": True,
        "answer_tokens_derived_from_usage": True,
        "finite_runner_budget_forbidden": True,
    }


def make_qwen_completion_request(
    client: Any,
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    *,
    requested_max_tokens: int,
) -> Any:
    contract = request_contract(args)
    if requested_max_tokens != contract["total_completion_cap"]:
        raise base.PreflightError(
            "runner/request completion-cap disagreement: "
            f"{requested_max_tokens} != {contract['total_completion_cap']}"
        )
    request: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_completion_tokens": requested_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.timeout_seconds,
        "extra_body": dict(args.extra_body),
    }
    if "max_tokens" in request:
        raise AssertionError("Qwen thinking request must not contain max_tokens")
    return client.chat.completions.create(**request)


def classify_qwen_terminal_response(
    response: Any,
    *,
    expected_model: str,
    max_prompt_tokens: int,
    requested_max_tokens: int,
) -> base.TerminalProviderResponse:
    if requested_max_tokens != TOTAL_COMPLETION_CAP:
        raise base.ResponseContractError(
            "Qwen terminal classifier requires requested max_completion_tokens="
            f"{TOTAL_COMPLETION_CAP}"
        )
    raw = base.response_to_dict(response)
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise base.ResponseContractError("response has no token usage")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, bool) or not isinstance(
        completion_tokens, int
    ):
        raise base.ResponseContractError("usage.completion_tokens is missing")
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        raise base.ResponseContractError(
            "usage.completion_tokens_details is missing"
        )
    reasoning_tokens = details.get("reasoning_tokens")
    if isinstance(reasoning_tokens, bool) or not isinstance(
        reasoning_tokens, int
    ):
        raise base.ResponseContractError(
            "usage.completion_tokens_details.reasoning_tokens is missing"
        )
    if reasoning_tokens <= 0:
        raise base.ResponseContractError("usage reasoning_tokens is not positive")
    if reasoning_tokens > min(completion_tokens, THINKING_BUDGET):
        raise base.ResponseContractError(
            f"provider counted {reasoning_tokens} reasoning tokens, thinking "
            f"limit is min(completion_tokens, {THINKING_BUDGET})"
        )
    answer_tokens = completion_tokens - reasoning_tokens
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise base.ResponseContractError(
            "Qwen reasoning contract requires exactly one response choice"
        )
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    reasoning_content = (
        message.get("reasoning_content") if isinstance(message, dict) else None
    )
    if not isinstance(reasoning_content, str) or not reasoning_content.strip():
        raise base.ResponseContractError(
            "Qwen response reasoning_content is empty or missing"
        )
    terminal = _BASE_CLASSIFY_TERMINAL_RESPONSE(
        response,
        expected_model=expected_model,
        max_prompt_tokens=max_prompt_tokens,
        requested_max_tokens=COMPLETION_USAGE_VALIDATION_CAP,
    )
    normalized_usage = dict(terminal.usage)
    normalized_usage["reasoning_tokens"] = reasoning_tokens
    normalized_usage["answer_tokens"] = answer_tokens
    return dataclasses.replace(terminal, usage=normalized_usage)


def qwen_config_for_hash(args: argparse.Namespace) -> dict[str, Any]:
    config = _BASE_CONFIG_FOR_HASH(args)
    contract = request_contract(args)
    runtime_identity = config.get("runtime_identity")
    if not isinstance(runtime_identity, dict):
        raise base.PreflightError("base runtime identity is missing")
    runtime_identity["qwen_completion_entry_sha256"] = base.sha256_file(ENTRY_PATH)
    config["qwen_request_contract"] = contract
    slot_policy = config.get("slot_policy")
    if not isinstance(slot_policy, dict):
        raise base.PreflightError("base exact-slot policy is missing")
    slot_policy["request_cap_parameter"] = "max_completion_tokens"
    slot_policy["max_tokens_absent"] = True
    slot_policy["total_completion_cap"] = TOTAL_COMPLETION_CAP
    slot_policy["provider_completion_tolerance"] = (
        PROVIDER_COMPLETION_TOLERANCE
    )
    slot_policy["completion_usage_validation_cap"] = (
        COMPLETION_USAGE_VALIDATION_CAP
    )
    slot_policy["requested_max_tokens_field_semantics"] = (
        "The journal field records the top-level max_completion_tokens request "
        "cap; response usage validation separately includes the documented "
        "provider tolerance."
    )
    slot_policy["reasoning_tokens_usage_required_positive_and_bounded"] = True
    slot_policy["reasoning_content_required_nonempty"] = True
    slot_policy["exact_extra_body_keys"] = sorted(EXACT_EXTRA_BODY_KEYS)
    slot_policy["normalized_usage_includes_reasoning_and_answer_tokens"] = True
    slot_policy["finite_runner_budget_forbidden"] = True
    config["slot_policy_sha256"] = base.stable_sha256(slot_policy)
    return config


def qwen_prepare_run(
    args: argparse.Namespace,
    out: Path,
) -> tuple[Any, list[dict[str, Any]], dict[str, dict[str, Any]], str, dict[str, Any]]:
    prepared = _BASE_PREPARE_RUN(args, out)
    bundle, plans, prompt_map, config_sha, provenance = prepared
    provenance = dict(provenance)
    provenance["qwen_completion_entry"] = base.file_record(ENTRY_PATH)
    provenance["qwen_request_contract"] = request_contract(args)
    provenance["preflight_invariants"] = dict(
        provenance.get("preflight_invariants") or {}
    )
    provenance["preflight_invariants"].update(
        {
            "qwen_uses_max_completion_tokens": True,
            "qwen_request_omits_max_tokens": True,
            "reasoning_and_final_share_12288_completion_cap": True,
            "provider_completion_tolerance_is_10": True,
            "completion_usage_validation_cap_is_12298": True,
            "reasoning_tokens_usage_required_and_bounded": True,
            "reasoning_content_required_nonempty": True,
            "extra_body_keyset_exact": True,
            "answer_tokens_derived_and_preserved": True,
            "finite_runner_budget_disabled": True,
        }
    )
    base.atomic_write_json(out / "provenance.json", provenance)
    return bundle, plans, prompt_map, config_sha, provenance


def install_qwen_completion_policy() -> None:
    base.make_request = make_qwen_completion_request
    base.config_for_hash = qwen_config_for_hash
    base.prepare_run = qwen_prepare_run
    base.classify_terminal_provider_response = classify_qwen_terminal_response


def main() -> int:
    install_qwen_completion_policy()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
