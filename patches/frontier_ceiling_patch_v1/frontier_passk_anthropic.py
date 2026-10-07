#!/usr/bin/env python3
"""Additive direct-Anthropic transport for the audited frontier runner.

This module deliberately does not modify ``frontier_passk.py`` or
``frontier_core.py``.  It reuses their sealed F2 preflight, journals, hardened
evaluator, fixed-slot semantics, and finalization while replacing only the
provider transport with Anthropic's native Messages API.

Claude Sonnet 5 is locked to adaptive thinking.  Sampling parameters are
omitted because Sonnet 5 rejects non-default temperature/top-p/top-k values.
The native response (including thinking/redacted-thinking blocks) is retained
inside the normalized response envelope so an audited run remains replayable.

The synchronous transport is intentionally budget gated.  A paid invocation
requires a positive ``--budget`` token ceiling.  Use
``--max-new-terminal-slots`` to stop cleanly after a deterministic tranche;
zero means all remaining slots.  The tranche size is operational state and is
excluded from the semantic config hash, so a run can resume with another
tranche without changing the sealed K experiment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import frontier_passk as audited
from frontier_core import (
    JsonlJournal,
    PreflightError,
    ResponseContractError,
    TokenBudget,
    atomic_write_json,
    file_record,
    load_json,
    load_jsonl,
    sha256_file,
    sha256_text,
    stable_sha256,
    utc_now,
)

MODEL_ID = "claude-sonnet-5"
PROVIDER = "anthropic"
MAX_OUTPUT_TOKENS = 65_536
DEFAULT_BASE_URL = "https://api.anthropic.com"
TRANSPORT_SCHEMA = "direct-anthropic-messages-adaptive-v1"
DISPATCH_SCHEMA = "sealed-round-robin-terminal-slot-tranches-v1"
NATIVE_STOP_REPORT_SCHEMA = "anthropic-native-stop-report-v1"
CAPABILITY_METRIC_ASSESSMENT_SCHEMA = "anthropic-capability-metric-assessment-v1"
ALLOWED_EFFORTS = ("medium", "high", "xhigh", "max")

# Introductory Claude Sonnet 5 list prices through 2026-08-31.  This is an
# estimate of list cost, not a claim about credits, negotiated rates, taxes, or
# the amount ultimately billed.
INPUT_USD_PER_MILLION = 2.0
OUTPUT_USD_PER_MILLION = 10.0
PRICE_VALID_THROUGH = "2026-08-31"
PRICE_SOURCE = (
    "https://platform.claude.com/docs/en/about-claude/models/" "whats-new-sonnet-5"
)

_ORIGINAL_PARSE_ARGS = audited.parse_args
_ORIGINAL_FIXED_SLOT_POLICY = audited.fixed_slot_policy
_ORIGINAL_CONFIG_FOR_HASH = audited.config_for_hash
_ORIGINAL_RUN_API_AND_EVALUATION = audited.run_api_and_evaluation


class BudgetPause(RuntimeError):
    """The sealed token ceiling cannot reserve another transport attempt."""


def _flag_present(argv: Sequence[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in argv)


def _temporary_environment(
    updates: Mapping[str, str]
) -> tuple[dict[str, str], set[str]]:
    previous: dict[str, str] = {}
    missing: set[str] = set()
    for key, value in updates.items():
        if key in os.environ:
            previous[key] = os.environ[key]
        else:
            missing.add(key)
        os.environ[key] = value
    return previous, missing


def _restore_environment(previous: Mapping[str, str], missing: set[str]) -> None:
    for key in missing:
        os.environ.pop(key, None)
    for key, value in previous.items():
        os.environ[key] = value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the audited runner arguments plus Anthropic-only controls."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    front = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    front.add_argument(
        "--anthropic-effort",
        choices=ALLOWED_EFFORTS,
        default=os.environ.get("ANTHROPIC_EFFORT", "high"),
    )
    front.add_argument(
        "--anthropic-base-url",
        default=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL),
    )
    front.add_argument(
        "--max-new-terminal-slots",
        type=int,
        default=int(os.environ.get("MAX_NEW_TERMINAL_SLOTS", "0")),
        help=(
            "Generate at most this many new terminal slots in sealed "
            "sample-major/task-order; 0 completes all remaining slots."
        ),
    )
    anthro, remaining = front.parse_known_args(raw_argv)

    for forbidden in (
        "--provider",
        "--api-key",
        "--qwen-env-file",
        "--deepseek-env-file",
    ):
        if _flag_present(remaining, forbidden):
            front.error(
                f"{forbidden} is disabled in the Anthropic runner; use only "
                "ANTHROPIC_API_KEY for credentials"
            )

    # Reuse the audited parser without widening its pinned provider choices.
    # Temporary defaults are restored before this function returns.
    env_updates = {
        "PROVIDER": "qwen",
        "MODEL": os.environ.get("MODEL", MODEL_ID),
        "MAXTOK": os.environ.get("MAXTOK", str(MAX_OUTPUT_TOKENS)),
        "TEMPERATURE": os.environ.get("TEMPERATURE", "1"),
        "TOP_P": os.environ.get("TOP_P", "1"),
    }
    previous, missing = _temporary_environment(env_updates)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = old_argv
        _restore_environment(previous, missing)

    args.provider = PROVIDER
    args.anthropic_effort = anthro.anthropic_effort
    args.anthropic_base_url = anthro.anthropic_base_url.rstrip("/")
    args.max_new_terminal_slots = anthro.max_new_terminal_slots

    if args.model != MODEL_ID:
        front.error(f"--model must be exactly {MODEL_ID!r}")
    if args.max_output_tokens != MAX_OUTPUT_TOKENS:
        front.error(
            f"--max-output-tokens must be exactly {MAX_OUTPUT_TOKENS} for "
            "the 64K ceiling contract"
        )
    if args.temperature != 1.0 or args.top_p != 1.0:
        front.error(
            "Claude Sonnet 5 requires sampling parameters at their defaults; "
            "temperature and top-p must both be 1 (and are omitted on wire)"
        )
    if args.extra_body:
        front.error("--extra-body-json is not supported by the native transport")
    if args.max_new_terminal_slots < 0:
        front.error("--max-new-terminal-slots cannot be negative")
    if not args.preflight_only and args.budget <= 0:
        front.error("paid Anthropic runs require a positive --budget token ceiling")
    if not args.anthropic_base_url:
        front.error("--anthropic-base-url cannot be empty")
    return args


def resolve_api_configuration(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve credentials without reading or persisting their value."""

    return (
        os.environ.get("ANTHROPIC_API_KEY", ""),
        str(args.anthropic_base_url).rstrip("/"),
    )


def fixed_slot_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy = _ORIGINAL_FIXED_SLOT_POLICY(args)
    policy.pop("temperature", None)
    policy.pop("top_p", None)
    policy.pop("extra_body", None)
    policy.update(
        {
            "transport_schema": TRANSPORT_SCHEMA,
            "native_api": "anthropic_messages",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": args.anthropic_effort},
            "sampling_parameters_on_wire": {
                "temperature": "omitted",
                "top_p": "omitted",
                "top_k": "omitted",
            },
            "native_response_preserved": True,
            "dispatch_schema": DISPATCH_SCHEMA,
            "dispatch_order": "sample_index_then_sealed_task_order",
            "max_new_terminal_slots_is_resume_operational_only": True,
        }
    )
    return policy


def config_for_hash(args: argparse.Namespace) -> dict[str, Any]:
    """Extend the semantic config without including the resumable tranche."""

    config = _ORIGINAL_CONFIG_FOR_HASH(args)
    runtime = dict(config.get("runtime_identity") or {})
    runtime.pop("openai_sdk_version", None)
    try:
        anthropic_version: str | None = importlib.metadata.version("anthropic")
    except importlib.metadata.PackageNotFoundError:
        anthropic_version = None
    runtime.update(
        {
            "audited_runner_sha256": runtime.pop("runner_sha256"),
            "anthropic_runner_sha256": sha256_file(Path(__file__).resolve()),
            "anthropic_sdk_version": anthropic_version,
        }
    )
    config["runtime_identity"] = runtime
    config["provider"] = PROVIDER
    config["anthropic_transport"] = {
        "schema": TRANSPORT_SCHEMA,
        "native_endpoint": "/v1/messages",
        "model": MODEL_ID,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": args.anthropic_effort},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "sampling_parameters_omitted": True,
        "dispatch_schema": DISPATCH_SCHEMA,
    }
    # Deliberately absent: args.max_new_terminal_slots.  It controls only how
    # much of an already sealed slot schedule this process invocation executes.
    return config


def install_audited_hooks() -> None:
    """Install additive provider hooks in this process only."""

    audited.resolve_api_configuration = resolve_api_configuration
    audited.fixed_slot_policy = fixed_slot_policy
    audited.config_for_hash = config_for_hash


def _native_response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        value = response.model_dump()
        if isinstance(value, dict):
            return value
    if hasattr(response, "dict"):
        value = response.dict()
        if isinstance(value, dict):
            return value
    raise ResponseContractError(
        f"Anthropic response cannot be serialized: {type(response).__name__}"
    )


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponseContractError(f"Anthropic {label} is missing or invalid")
    return value


def _stop_category_component(value: Any) -> str | None:
    """Return a stable, non-secret provider stop-category component."""

    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    normalized = re.sub(r"[^a-z0-9._-]+", "_", text).strip("_")
    return normalized or None


def _stop_details_category(details: Any, stop_reason: str) -> str | None:
    """Extract a provider-declared category without inferring one from text."""

    direct = _stop_category_component(details)
    if direct is not None:
        return direct
    if not isinstance(details, Mapping):
        return None

    # Category/reason fields are semantically more specific than a generic
    # details ``type`` such as "refusal".
    for key in (
        "category",
        "policy_category",
        "refusal_category",
        "safety_category",
        "reason",
        "code",
    ):
        value = _stop_category_component(details.get(key))
        if value is not None and value != stop_reason:
            return value
    for key in ("refusal", "safety", "details", "policy"):
        nested = _stop_details_category(details.get(key), stop_reason)
        if nested is not None and nested != stop_reason:
            return nested
    value = _stop_category_component(details.get("type"))
    if value is not None and value != stop_reason:
        return value
    return None


def native_stop_metadata(response: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Expose the native Anthropic stop without changing normalized semantics.

    Anthropic's native ``stop_reason=refusal`` remains normalized to the
    OpenAI-compatible ``finish_reason=content_filter`` elsewhere.  This helper
    retains the provider-native reason and, when the provider supplies one,
    its stop-details category.  It deliberately never guesses a category from
    prompt or response prose.
    """

    value = _native_response_dict(response)
    native_value = value.get("provider_native_response")
    native = native_value if isinstance(native_value, Mapping) else value
    raw_reason = native.get("stop_reason")
    stop_reason = (
        str(raw_reason).strip().lower()
        if isinstance(raw_reason, str) and raw_reason.strip()
        else None
    )
    stop_details = native.get("stop_details")
    category = _stop_details_category(stop_details, stop_reason or "")

    # Some Anthropic-compatible deployments attach the category beside
    # stop_details or on a refusal content block.  Accept only explicit
    # provider metadata; never classify the response text ourselves.
    if category is None:
        for key in (
            "stop_category",
            "refusal_category",
            "safety_category",
            "policy_category",
        ):
            category = _stop_category_component(native.get(key))
            if category is not None:
                break
    if category is None:
        blocks = native.get("content")
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, Mapping):
                    continue
                block_type = _stop_category_component(block.get("type"))
                if block_type not in {"refusal", "safety_refusal"}:
                    continue
                category = _stop_details_category(block, stop_reason or "")
                if category is not None:
                    break

    if stop_reason == "refusal":
        combined = f"refusal:{category or 'unspecified'}"
    elif stop_reason is None:
        combined = "missing"
    elif category is None:
        combined = stop_reason
    else:
        combined = f"{stop_reason}:{category}"
    return {
        "native_stop_reason": stop_reason,
        "native_stop_details_category": category,
        "native_stop_category": combined,
    }


def native_stop_metadata_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Read additive stop fields or reconstruct them from a preserved response."""

    reason = record.get("native_stop_reason")
    combined = record.get("native_stop_category")
    category = record.get("native_stop_details_category")
    if (
        (reason is None or isinstance(reason, str))
        and isinstance(combined, str)
        and combined
        and (category is None or isinstance(category, str))
    ):
        return {
            "native_stop_reason": reason,
            "native_stop_details_category": category,
            "native_stop_category": combined,
        }
    for field in ("normalized_response", "response"):
        response = record.get(field)
        if isinstance(response, Mapping):
            return native_stop_metadata(response)
    return {
        "native_stop_reason": None,
        "native_stop_details_category": None,
        "native_stop_category": "missing",
    }


def anthropic_metric_transparency(
    records: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    k: int,
    complete: bool | None = None,
) -> dict[str, Any]:
    """Build refusal-transparent stop counts and ceiling-validity coverage.

    Execution metrics retain their sealed denominator.  A refusal, however,
    means that denominator result is not a capability ceiling: policy prevented
    observation of the requested sample.  We therefore expose both the lower
    bound and the exact non-refusal coverage instead of silently interpreting
    refusals as capability failures.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ResponseContractError("Anthropic transparency K must be positive")
    ordered_tasks = [str(task_id) for task_id in task_ids]
    if not ordered_tasks or any(not task_id for task_id in ordered_tasks):
        raise ResponseContractError(
            "Anthropic transparency requires non-empty task IDs"
        )
    if len(set(ordered_tasks)) != len(ordered_tasks):
        raise ResponseContractError("Anthropic transparency task IDs are not unique")
    expected_keys = {
        (task_id, sample_index)
        for task_id in ordered_tasks
        for sample_index in range(k)
    }
    by_slot: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        task_id = str(record.get("task_id") or "")
        sample_index = record.get("sample_index")
        if (
            not task_id
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
        ):
            raise ResponseContractError(
                f"Anthropic transparency record {index} has no logical slot"
            )
        key = (task_id, sample_index)
        if key not in expected_keys:
            raise ResponseContractError(
                f"Anthropic transparency has unexpected logical slot {key!r}"
            )
        if key in by_slot:
            raise ResponseContractError(
                f"Anthropic transparency has duplicate logical slot {key!r}"
            )
        by_slot[key] = record

    expected_slots = len(expected_keys)
    observed_slots = len(by_slot)
    inferred_complete = observed_slots == expected_slots
    is_complete = inferred_complete if complete is None else bool(complete)
    if is_complete and not inferred_complete:
        raise ResponseContractError(
            "Anthropic transparency was marked complete with missing slots"
        )

    reason_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    slot_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    missing_metadata_slots = 0
    for key, record in by_slot.items():
        metadata = native_stop_metadata_from_record(record)
        slot_metadata[key] = metadata
        reason = metadata["native_stop_reason"]
        reason_key = str(reason) if reason is not None else "missing"
        category_key = str(metadata["native_stop_category"])
        reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        category_counts[category_key] = category_counts.get(category_key, 0) + 1
        if reason is None:
            missing_metadata_slots += 1

    refusal_keys = {
        key
        for key, metadata in slot_metadata.items()
        if metadata["native_stop_reason"] == "refusal"
    }
    non_refusal_keys = set(by_slot) - refusal_keys
    refusal_slots = len(refusal_keys)
    non_refusal_slots = len(non_refusal_keys)
    refusal_dominated = observed_slots > 0 and refusal_slots * 2 > observed_slots

    tasks_with_any_non_refusal = 0
    tasks_with_all_k_non_refusal = 0
    tasks_with_all_slots_refused = 0
    tasks_with_all_slots_observed = 0
    for task_id in ordered_tasks:
        keys = {(task_id, sample_index) for sample_index in range(k)}
        observed = keys & set(by_slot)
        if observed == keys:
            tasks_with_all_slots_observed += 1
        non_refused = keys & non_refusal_keys
        refused = keys & refusal_keys
        if non_refused:
            tasks_with_any_non_refusal += 1
        if non_refused == keys:
            tasks_with_all_k_non_refusal += 1
        if refused == keys:
            tasks_with_all_slots_refused += 1

    if not is_complete:
        validity = "invalid_incomplete"
        invalid_reasons = ["logical_slot_coverage_incomplete"]
    elif missing_metadata_slots:
        validity = "invalid_native_stop_metadata_missing"
        invalid_reasons = ["native_stop_reason_missing"]
    elif refusal_dominated:
        validity = "invalid_refusal_dominated"
        invalid_reasons = ["provider_refusals_dominate_observed_slots"]
    elif refusal_slots:
        validity = "invalid_refusal_present"
        invalid_reasons = ["provider_refusal_prevents_full_capability_observation"]
    else:
        validity = "valid"
        invalid_reasons = []
    valid_as_ceiling = validity == "valid"

    stop_report = {
        "schema": NATIVE_STOP_REPORT_SCHEMA,
        "observed_terminal_slots": observed_slots,
        "expected_terminal_slots": expected_slots,
        "native_stop_reason_counts": dict(sorted(reason_counts.items())),
        "native_stop_category_counts": dict(sorted(category_counts.items())),
        "native_stop_reason_missing_slots": missing_metadata_slots,
        "refusal_slots": refusal_slots,
        "refusal_rate_observed": (
            refusal_slots / observed_slots if observed_slots else None
        ),
        "refusal_rate_expected": refusal_slots / expected_slots,
        "refusal_dominated": refusal_dominated,
        "category_semantics": (
            "provider-declared stop_details only; refusal:unspecified means "
            "the provider supplied no category"
        ),
    }
    assessment = {
        "schema": CAPABILITY_METRIC_ASSESSMENT_SCHEMA,
        "status": validity,
        "capability_metric_valid": valid_as_ceiling,
        "valid_as_capability_ceiling": valid_as_ceiling,
        "ceiling_claim_allowed": valid_as_ceiling,
        "invalid_reasons": invalid_reasons,
        "execution_metric_interpretation": (
            "capability_ceiling"
            if valid_as_ceiling
            else "sealed_denominator_lower_bound_not_capability_ceiling"
        ),
        "sealed_execution_denominator_unchanged": True,
        "refusal_is_not_counted_as_observed_capability_failure": True,
        "coverage": {
            "expected_tasks": len(ordered_tasks),
            "k": k,
            "expected_slots": expected_slots,
            "observed_terminal_slots": observed_slots,
            "observed_terminal_slot_rate": observed_slots / expected_slots,
            "non_refusal_slots": non_refusal_slots,
            "non_refusal_slot_rate": non_refusal_slots / expected_slots,
            "tasks_with_any_non_refusal": tasks_with_any_non_refusal,
            "tasks_with_any_non_refusal_rate": (
                tasks_with_any_non_refusal / len(ordered_tasks)
            ),
            "tasks_with_all_k_non_refusal": tasks_with_all_k_non_refusal,
            "tasks_with_all_k_non_refusal_rate": (
                tasks_with_all_k_non_refusal / len(ordered_tasks)
            ),
            "tasks_with_all_slots_refused": tasks_with_all_slots_refused,
            "tasks_with_all_slots_observed": tasks_with_all_slots_observed,
        },
    }
    return {
        "anthropic_native_stop_report": stop_report,
        "capability_metric_assessment": assessment,
    }


def normalize_anthropic_response(response: Any) -> dict[str, Any]:
    """Convert a native Message into the audited terminal-response envelope."""

    native = _native_response_dict(response)
    response_id = str(native.get("id") or "")
    model = str(native.get("model") or "")
    if not response_id:
        raise ResponseContractError("Anthropic response id is missing")
    if not model:
        raise ResponseContractError("Anthropic resolved model is missing")

    usage_native = native.get("usage")
    if not isinstance(usage_native, Mapping):
        raise ResponseContractError("Anthropic response usage is missing")
    input_tokens = _nonnegative_int(
        usage_native.get("input_tokens"), "usage.input_tokens"
    )
    output_tokens = _nonnegative_int(
        usage_native.get("output_tokens"), "usage.output_tokens"
    )

    blocks = native.get("content")
    if not isinstance(blocks, list):
        raise ResponseContractError("Anthropic content is not a block list")
    text_blocks: list[str] = []
    thinking_blocks: list[str] = []
    refusal_present = str(native.get("stop_reason") or "") == "refusal"
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ResponseContractError("Anthropic content block is malformed")
        block_type = str(block.get("type") or "")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ResponseContractError("Anthropic text block is malformed")
            text_blocks.append(text)
        elif block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str):
                thinking_blocks.append(thinking)
        elif block_type == "redacted_thinking":
            # The untouched native block remains in provider_native_response.
            continue
        elif block_type in {"refusal", "safety_refusal"}:
            refusal_present = True

    stop_reason = str(native.get("stop_reason") or "")
    finish_reason = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "refusal": "content_filter",
    }.get(stop_reason, f"anthropic:{stop_reason or 'missing'}")
    normalized = {
        "id": response_id,
        "model": model,
        "created": None,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "\n\n".join(text_blocks),
                    "reasoning_content": "\n\n".join(thinking_blocks),
                    "refusal": "anthropic_refusal" if refusal_present else None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "provider_native_response": native,
        "provider_transport_schema": TRANSPORT_SCHEMA,
    }
    normalized.update(native_stop_metadata(native))
    return normalized


def _split_anthropic_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system_values: list[str] = []
    native_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        content = message.get("content")
        if not isinstance(content, str):
            raise PreflightError(f"message {index} content is not a string")
        if role == "system":
            if native_messages:
                raise PreflightError(
                    "system message appears after conversation content"
                )
            system_values.append(content)
        elif role in {"user", "assistant"}:
            native_messages.append({"role": role, "content": content})
        else:
            raise PreflightError(f"unsupported Anthropic message role: {role!r}")
    if len(system_values) != 1:
        raise PreflightError("exactly one system message is required")
    if not native_messages or native_messages[-1]["role"] != "user":
        raise PreflightError("Anthropic conversation must end in a user message")
    return system_values[0], native_messages


class AnthropicMessagesTransport:
    """Small native Messages client with no compatibility endpoint."""

    def __init__(self, args: argparse.Namespace, api_key: str, base_url: str) -> None:
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover - depends on remote environment
            raise PreflightError(
                "the anthropic Python package is required for a paid run"
            ) from exc
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._args = args

    def create(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        system, native_messages = _split_anthropic_messages(messages)
        response = self._client.messages.create(
            model=self._args.model,
            max_tokens=self._args.max_output_tokens,
            system=system,
            messages=native_messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self._args.anthropic_effort},
            timeout=self._args.timeout_seconds,
        )
        return normalize_anthropic_response(response)


def _redact_exception(exc: Exception, api_key: str) -> str:
    value = f"{type(exc).__name__}:{str(exc)[:1000]}"
    if api_key:
        value = value.replace(api_key, "[REDACTED]")
    return value


def cost_estimate_from_attempts(
    attempt_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    terminal_responses = 0
    unknown_billing_attempts = 0
    conservative_budget_charge = 0
    for row in attempt_rows:
        charge = row.get("budget_charge_tokens")
        if isinstance(charge, int) and not isinstance(charge, bool) and charge >= 0:
            conservative_budget_charge += charge
        if row.get("response_received") is True:
            terminal_responses += 1
            usage = row.get("usage")
            if isinstance(usage, Mapping):
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if isinstance(prompt, int) and not isinstance(prompt, bool):
                    prompt_tokens += prompt
                if isinstance(completion, int) and not isinstance(completion, bool):
                    completion_tokens += completion
        elif row.get("transport_retry") is True:
            unknown_billing_attempts += 1
    input_cost = prompt_tokens * INPUT_USD_PER_MILLION / 1_000_000
    output_cost = completion_tokens * OUTPUT_USD_PER_MILLION / 1_000_000
    return {
        "schema": "anthropic-list-cost-estimate-v1",
        "provider_reported_prompt_tokens": prompt_tokens,
        "provider_reported_completion_tokens": completion_tokens,
        "provider_reported_total_tokens": prompt_tokens + completion_tokens,
        "terminal_responses": terminal_responses,
        "unknown_billing_transport_attempts": unknown_billing_attempts,
        "conservative_budget_charge_tokens": conservative_budget_charge,
        "input_usd_per_million": INPUT_USD_PER_MILLION,
        "output_usd_per_million": OUTPUT_USD_PER_MILLION,
        "estimated_input_usd": input_cost,
        "estimated_output_usd": output_cost,
        "estimated_total_usd": input_cost + output_cost,
        "price_valid_through": PRICE_VALID_THROUGH,
        "price_source": PRICE_SOURCE,
        "estimate_not_invoice": True,
    }


def _api_provenance(
    args: argparse.Namespace, base_url: str, cost: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    try:
        version: str | None = importlib.metadata.version("anthropic")
    except importlib.metadata.PackageNotFoundError:
        version = None
    result: dict[str, Any] = {
        "provider": PROVIDER,
        "transport_schema": TRANSPORT_SCHEMA,
        "native_api": "anthropic_messages",
        "base_url_redacted": audited.redact_api_endpoint(base_url),
        "base_url_sha256": sha256_text(base_url.rstrip("/")),
        "requested_model": args.model,
        "anthropic_package_version": version,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": args.anthropic_effort},
        "max_tokens": args.max_output_tokens,
        "sampling_parameters_omitted": True,
        "credentials_source": "ANTHROPIC_API_KEY environment variable",
        "credentials_persisted": False,
    }
    if cost is not None:
        result["usage_and_list_cost"] = dict(cost)
    return result


def _rewrite_final_attestation(
    out: Path,
    args: argparse.Namespace,
    base_url: str,
) -> dict[str, Any]:
    """Replace compatibility-only API metadata and reseal final artifacts."""

    attempts = load_jsonl(out / "attempts.jsonl", "completed attempt journal")
    cost = cost_estimate_from_attempts(attempts)
    summary = load_json(out / "summary.json", "completed summary")
    terminal_attempts = [
        row
        for row in attempts
        if row.get("response_received") is True and row.get("slot_terminal") is True
    ]
    task_results = summary.get("task_results")
    if not isinstance(task_results, list):
        raise ResponseContractError("completed Anthropic summary has no task results")
    task_ids = [
        str(row.get("task_id") or "")
        for row in task_results
        if isinstance(row, Mapping)
    ]
    summary.update(
        anthropic_metric_transparency(
            terminal_attempts,
            task_ids=task_ids,
            k=args.k,
            complete=True,
        )
    )
    summary["anthropic_transport"] = _api_provenance(args, base_url, cost)
    summary["usage_and_list_cost"] = cost
    atomic_write_json(out / "summary.json", summary)

    provenance = load_json(out / "provenance.json", "completed provenance")
    provenance["api"] = _api_provenance(args, base_url, cost)
    provenance["summary_sha256"] = sha256_file(out / "summary.json")
    atomic_write_json(out / "provenance.json", provenance)

    files = (
        "provenance.json",
        "tasks.jsonl",
        "prompts.jsonl",
        "attempts.jsonl",
        "outcomes.jsonl",
        "summary.json",
    )
    atomic_write_json(
        out / "manifest.json",
        {
            "schema": audited.RUN_SCHEMA_VERSION,
            "created_at": utc_now(),
            "files": {name: file_record(out / name) for name in files},
        },
    )
    return summary


def _finalize_with_audited_runner(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    prompt_map: dict[str, dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Use the unmodified audited aggregator once every slot exists."""

    # All slots already exist, so the audited function performs no provider
    # call.  It still imports/constructs OpenAI; provide a fail-closed sentinel
    # so a regression cannot silently contact a compatibility endpoint.
    import openai

    original_openai = openai.OpenAI

    class NoCallOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = self
            self.completions = self

        def create(self, **_: Any) -> Any:  # pragma: no cover - safety sentinel
            raise audited.RunFailure(
                "audited finalization unexpectedly attempted an API request"
            )

    openai.OpenAI = NoCallOpenAI
    try:
        summary = _ORIGINAL_RUN_API_AND_EVALUATION(
            args,
            out=out,
            plans=plans,
            prompt_map=prompt_map,
            config_sha=config_sha,
            provenance=provenance,
        )
    finally:
        openai.OpenAI = original_openai
    del summary
    _key, base_url = resolve_api_configuration(args)
    return _rewrite_final_attestation(out, args, base_url)


def run_incremental(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    prompt_map: dict[str, dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Generate a deterministic tranche and finalize only at exact K."""

    key, base_url = audited.api_credentials(args)
    transport = AnthropicMessagesTransport(args, key, base_url)
    if not args.expected_evaluator_sha256.strip():
        raise PreflightError("paid evaluation requires --expected-evaluator-sha256")
    if not args.expected_dart_sha256.strip():
        raise PreflightError("paid evaluation requires --expected-dart-sha256")
    evaluator_module, evaluator_record = audited.import_evaluator(
        args.evaluator_module,
        args.expected_evaluator_sha256,
        dart_binary=args.dart,
        expected_dart_hash=args.expected_dart_sha256,
        validate_dart=True,
    )
    evaluator = evaluator_module.evaluate_dart_jit_tests_detail
    if evaluator_record["sha256"] != provenance["evaluator"]["sha256"]:
        raise PreflightError("evaluator changed after prompt preflight")

    provenance = dict(provenance)
    provenance["evaluator"] = evaluator_record
    provenance["api"] = _api_provenance(args, base_url)
    provenance["status"] = "running"
    provenance["started_at"] = utc_now()
    atomic_write_json(out / "provenance.json", provenance)

    attempts_path = out / "attempts.jsonl"
    outcomes_path = out / "outcomes.jsonl"
    # Empty journals are valid durable state when the configured budget cannot
    # reserve even one request; creating them also makes progress attestation
    # uniform across zero- and nonzero-completion tranches.
    attempts_path.touch(exist_ok=True)
    outcomes_path.touch(exist_ok=True)
    budget = TokenBudget(args.budget)
    slot_policy_sha = provenance["config"]["slot_policy_sha256"]
    terminal_resume, next_attempt = audited.load_resume_attempts(
        attempts_path,
        config_sha=config_sha,
        prompt_map=prompt_map,
        budget=budget,
        requested_model=args.model,
        k=args.k,
        max_prompt_tokens=args.max_prompt_tokens,
        requested_max_tokens=args.max_output_tokens,
        max_transport_attempts_per_slot=args.max_attempts_per_sample,
        slot_policy_sha256=slot_policy_sha,
    )
    resumed_outcomes = audited.load_resume_outcomes(
        outcomes_path,
        config_sha=config_sha,
        evaluator_sha256=evaluator_record["sha256"],
    )
    terminal_attempt_keys = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in terminal_resume.items()
    }
    orphan_outcomes = sorted(set(resumed_outcomes) - terminal_attempt_keys)
    if orphan_outcomes:
        raise audited.RunFailure(
            f"outcome journal has orphan record(s); first={orphan_outcomes[0]}"
        )

    plan_by_task = {str(plan["task_id"]): plan for plan in plans}
    ordered_slots = [
        (str(plan["task_id"]), sample_index)
        for sample_index in range(args.k)
        for plan in plans
    ]
    pending_slots = [slot for slot in ordered_slots if slot not in terminal_resume]
    tranche_limit = args.max_new_terminal_slots or len(pending_slots)
    selected_slots = pending_slots[:tranche_limit]

    attempts = JsonlJournal(attempts_path)
    outcomes = JsonlJournal(outcomes_path)
    response_ids = {str(row["response_id"]) for row in terminal_resume.values()}
    response_id_lock = threading.Lock()
    worst_case = args.max_prompt_tokens + args.max_output_tokens
    budget_paused = threading.Event()

    def execute_slot(slot: tuple[str, int]) -> dict[str, Any]:
        task_id, sample_index = slot
        plan = plan_by_task[task_id]
        terminal_record: dict[str, Any] | None = None
        first_attempt = next_attempt.get(slot, 0)
        for attempt_index in range(first_attempt, args.max_attempts_per_sample):
            if not budget.reserve(worst_case):
                budget_paused.set()
                raise BudgetPause("token budget cannot reserve another request")
            attempt_id = (
                f"{audited.safe_label(task_id)}.s{sample_index}.a{attempt_index}."
                f"{uuid.uuid4().hex[:10]}"
            )
            base_record: dict[str, Any] = {
                "schema": audited.RUN_SCHEMA_VERSION,
                "record_type": "api_attempt",
                "attempt_id": attempt_id,
                "config_sha256": config_sha,
                "task_id": task_id,
                "sample_index": sample_index,
                "attempt_index": attempt_index,
                "prompt_sha256": plan["prompt_sha256"],
                "requested_model": args.model,
                "requested_max_tokens": args.max_output_tokens,
                "provider": args.provider,
                "slot_policy_sha256": slot_policy_sha,
                "started_at": utc_now(),
            }
            reservation_open = True
            try:
                response = transport.create(plan["messages"])
                settled = audited.usage_total(response, worst_case)
                budget.settle(worst_case, settled)
                reservation_open = False
                try:
                    terminal = audited.classify_terminal_provider_response(
                        response,
                        expected_model=args.model,
                        max_prompt_tokens=args.max_prompt_tokens,
                        requested_max_tokens=args.max_output_tokens,
                    )
                except ResponseContractError as exc:
                    record = dict(base_record)
                    record.update(
                        {
                            "finished_at": utc_now(),
                            "response_received": True,
                            "slot_terminal": True,
                            "candidate_valid": False,
                            "terminal_reason": "fatal_response_contract:" + str(exc),
                            "transport_retry": False,
                            "transport_error": None,
                            "fatal_response_contract": True,
                            "budget_charge_tokens": settled,
                            "usage": response.get("usage"),
                            "response": response,
                        }
                    )
                    attempts.append(record)
                    raise audited.RunFailure(
                        f"Anthropic response violates fatal contract: {exc}"
                    ) from exc
                with response_id_lock:
                    duplicate = terminal.response_id in response_ids
                    if not duplicate:
                        response_ids.add(terminal.response_id)
                record = dict(base_record)
                stop_metadata = native_stop_metadata(response)
                record.update(
                    {
                        "finished_at": utc_now(),
                        "response_received": True,
                        "slot_terminal": True,
                        "candidate_valid": terminal.candidate_valid,
                        "terminal_reason": terminal.terminal_reason,
                        "transport_retry": False,
                        "transport_error": None,
                        "fatal_response_contract": False,
                        "response_id": terminal.response_id,
                        "resolved_model": terminal.response_model,
                        "response_created": terminal.response_created,
                        "finish_reason": terminal.finish_reason,
                        **stop_metadata,
                        "budget_charge_tokens": settled,
                        "usage": terminal.usage,
                        "content": terminal.content,
                        "reasoning_content": terminal.reasoning_content,
                        "code": terminal.code,
                        "code_sha256": terminal.code_sha256,
                        "response": terminal.raw_response,
                    }
                )
                attempts.append(record)
                if duplicate:
                    raise audited.RunFailure(
                        f"duplicate terminal response id: {terminal.response_id}"
                    )
                terminal_record = record
                break
            except audited.RunFailure:
                if reservation_open:
                    budget.settle(worst_case, worst_case)
                raise
            except Exception as exc:
                if reservation_open:
                    budget.settle(worst_case, worst_case)
                    reservation_open = False
                retryable = audited.is_retryable_api_exception(exc)
                record = dict(base_record)
                record.update(
                    {
                        "finished_at": utc_now(),
                        "response_received": False,
                        "slot_terminal": False,
                        "candidate_valid": None,
                        "terminal_reason": None,
                        "transport_retry": True,
                        "retryable_transport": retryable,
                        "transport_error": "api_exception:"
                        + _redact_exception(exc, key),
                        "fatal_response_contract": False,
                        "budget_charge_tokens": worst_case,
                        "usage": None,
                        "response": None,
                    }
                )
                attempts.append(record)
                if not retryable:
                    raise audited.RunFailure(
                        f"non-retryable Anthropic exception: "
                        f"{_redact_exception(exc, key)}"
                    ) from exc
            if (
                terminal_record is None
                and attempt_index + 1 < args.max_attempts_per_sample
            ):
                time.sleep(audited.retry_delay(args, attempt_index))

        if terminal_record is None:
            raise audited.RunFailure(
                f"slot {slot} received no response in "
                f"{args.max_attempts_per_sample} attempts"
            )
        if terminal_record["candidate_valid"]:
            evaluation = audited.evaluate_candidate_stably(
                evaluator,
                code=terminal_record["code"],
                tests=plan["row"]["acceptance_tests"],
                task_id=task_id,
                sample_index=sample_index,
                stability_runs=args.eval_stability_runs,
                timeout=args.eval_timeout_seconds,
            )
            evaluation_performed = True
        else:
            evaluation = {
                "compiled": False,
                "passed": False,
                "completion_attestation_id": audited.REQUIRED_ATTESTATION_ID,
                "completion_attestation_enforced": False,
                "completion_attestation_satisfied_all_runs": False,
                "stability_runs": [],
            }
            evaluation_performed = False
        outcome = {
            "schema": audited.RUN_SCHEMA_VERSION,
            "record_type": "candidate_outcome",
            "config_sha256": config_sha,
            "task_id": task_id,
            "sample_index": sample_index,
            "attempt_id": terminal_record["attempt_id"],
            "response_id": terminal_record["response_id"],
            "finish_reason": terminal_record["finish_reason"],
            "native_stop_reason": terminal_record.get("native_stop_reason"),
            "native_stop_details_category": terminal_record.get(
                "native_stop_details_category"
            ),
            "native_stop_category": terminal_record.get(
                "native_stop_category", "missing"
            ),
            "candidate_valid": terminal_record["candidate_valid"],
            "terminal_reason": terminal_record["terminal_reason"],
            "code_sha256": terminal_record["code_sha256"],
            "evaluator_sha256": evaluator_record["sha256"],
            "evaluator_entrypoint": evaluator_record["entrypoint"],
            "evaluation_performed": evaluation_performed,
            "completion_attestation_id": evaluation["completion_attestation_id"],
            "completion_attestation_enforced": evaluation[
                "completion_attestation_enforced"
            ],
            "completion_attestation_satisfied_all_runs": evaluation[
                "completion_attestation_satisfied_all_runs"
            ],
            "compiled": evaluation["compiled"],
            "passed": evaluation["passed"],
            "stability_runs": evaluation["stability_runs"],
            "evaluated_at": utc_now(),
        }
        outcomes.append(outcome)
        return {"slot": slot, "attempt": terminal_record, "outcome": outcome}

    completed_new = 0
    failures: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(selected_slots) and not failures:
        wave = selected_slots[cursor : cursor + args.workers]
        cursor += len(wave)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(wave))
        ) as pool:
            future_map = {pool.submit(execute_slot, slot): slot for slot in wave}
            for future in concurrent.futures.as_completed(future_map):
                slot = future_map[future]
                try:
                    future.result()
                except BudgetPause:
                    budget_paused.set()
                except Exception as exc:
                    failures.append(
                        {
                            "slot": list(slot),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                else:
                    completed_new += 1
        if budget_paused.is_set():
            break
    if failures:
        raise audited.RunFailure(f"{len(failures)} slot(s) failed; first={failures[0]}")

    # Re-validate journals before deciding whether the sealed run is complete.
    audit_budget = TokenBudget(0)
    terminal_now, _ = audited.load_resume_attempts(
        attempts_path,
        config_sha=config_sha,
        prompt_map=prompt_map,
        budget=audit_budget,
        requested_model=args.model,
        k=args.k,
        max_prompt_tokens=args.max_prompt_tokens,
        requested_max_tokens=args.max_output_tokens,
        max_transport_attempts_per_slot=args.max_attempts_per_sample,
        slot_policy_sha256=slot_policy_sha,
    )
    outcomes_now = audited.load_resume_outcomes(
        outcomes_path,
        config_sha=config_sha,
        evaluator_sha256=evaluator_record["sha256"],
    )
    expected_outcome_keys = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in terminal_now.items()
    }
    if set(outcomes_now) != expected_outcome_keys:
        raise audited.RunFailure("attempt/outcome journals are not one-to-one")

    total_slots = len(ordered_slots)
    completed_slots = len(terminal_now)
    attempt_rows = load_jsonl(attempts_path, "attempt journal")
    cost = cost_estimate_from_attempts(attempt_rows)
    progress = {
        "schema": "anthropic-frontier-progress-v1",
        "updated_at": utc_now(),
        "config_sha256": config_sha,
        "transport_schema": TRANSPORT_SCHEMA,
        "dispatch_schema": DISPATCH_SCHEMA,
        "dispatch_order": "sample_index_then_sealed_task_order",
        "k": args.k,
        "tasks": len(plans),
        "total_slots": total_slots,
        "completed_terminal_slots": completed_slots,
        "remaining_terminal_slots": total_slots - completed_slots,
        "new_terminal_slots_this_invocation": completed_new,
        "requested_max_new_terminal_slots": args.max_new_terminal_slots,
        "paused_for_budget": budget_paused.is_set(),
        "budget": budget.snapshot(),
        "usage_and_list_cost": cost,
        "attempts": file_record(attempts_path),
        "outcomes": file_record(outcomes_path),
    }
    progress.update(
        anthropic_metric_transparency(
            list(terminal_now.values()),
            task_ids=[str(plan["task_id"]) for plan in plans],
            k=args.k,
            complete=completed_slots == total_slots,
        )
    )
    atomic_write_json(out / "progress.json", progress)

    if completed_slots == total_slots:
        return {
            "status": "complete",
            "summary": _finalize_with_audited_runner(
                args,
                out=out,
                plans=plans,
                prompt_map=prompt_map,
                config_sha=config_sha,
                provenance=provenance,
            ),
            "progress": progress,
        }

    provenance["status"] = (
        "paused_token_budget"
        if budget_paused.is_set()
        else "paused_terminal_slot_tranche"
    )
    provenance["paused_at"] = utc_now()
    provenance["api"] = _api_provenance(args, base_url, cost)
    provenance["progress_sha256"] = sha256_file(out / "progress.json")
    atomic_write_json(out / "provenance.json", provenance)
    return {"status": provenance["status"], "progress": progress}


def main(argv: Sequence[str] | None = None) -> int:
    install_audited_hooks()
    args = parse_args(argv)
    out = audited.choose_output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    try:
        audited.enforce_output_state_policy(args, out)
        if not args.resume and (out / "progress.json").exists():
            raise audited.RunFailure("--no-resume refuses existing progress.json")
    except Exception as exc:
        print(
            f"FRONTIER_ANTHROPIC_FAILED_CLOSED error={type(exc).__name__}: "
            f"{exc} out={out}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    with audited.RunLock(out / ".run.lock"):
        try:
            bundle, plans, prompt_map, config_sha, provenance = audited.prepare_run(
                args, out
            )
            del bundle
            max_estimate = max(
                int(
                    prompt_map[plan["task_id"]]["token_count"][
                        "estimated_prompt_tokens"
                    ]
                )
                for plan in plans
            )
            print(
                f"PREFLIGHT_OK provider={PROVIDER} model={MODEL_ID} "
                f"effort={args.anthropic_effort} tasks={len(plans)} "
                f"K={args.k} max_output_tokens={args.max_output_tokens} "
                f"max_sealed_qwen_estimate_tokens={max_estimate} out={out}",
                flush=True,
            )
            if args.preflight_only:
                provenance["status"] = "preflight_only_complete"
                provenance["completed_at"] = utc_now()
                atomic_write_json(out / "provenance.json", provenance)
                return 0
            result = run_incremental(
                args,
                out=out,
                plans=plans,
                prompt_map=prompt_map,
                config_sha=config_sha,
                provenance=provenance,
            )
        except Exception as exc:
            failure = {
                "schema": audited.RUN_SCHEMA_VERSION,
                "status": "failed_closed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(out / "failure.json", failure)
            print(
                f"FRONTIER_ANTHROPIC_FAILED_CLOSED "
                f"error={type(exc).__name__}: {exc} out={out}",
                file=sys.stderr,
                flush=True,
            )
            return 2

    if result["status"] == "complete":
        summary = result["summary"]
        pass_result = summary["pass_at_k"]
        compile_result = summary["compile_at_k"]
        print(
            f"FRONTIER_ANTHROPIC_PASSK model={MODEL_ID} K={args.k} "
            f"tasks={summary['tasks']} pass@{args.k}="
            f"{pass_result['successes']}/{pass_result['total']}="
            f"{pass_result['rate']:.4f} compile@{args.k}="
            f"{compile_result['successes']}/{compile_result['total']}="
            f"{compile_result['rate']:.4f} "
            f"estimated_list_usd="
            f"{summary['usage_and_list_cost']['estimated_total_usd']:.4f} "
            f"capability_status="
            f"{summary['capability_metric_assessment']['status']} "
            f"non_refusal_slot_coverage="
            f"{summary['capability_metric_assessment']['coverage']['non_refusal_slots']}/"
            f"{summary['capability_metric_assessment']['coverage']['expected_slots']} "
            f"out={out}",
            flush=True,
        )
    else:
        progress = result["progress"]
        print(
            f"FRONTIER_ANTHROPIC_PAUSED status={result['status']} "
            f"slots={progress['completed_terminal_slots']}/"
            f"{progress['total_slots']} "
            f"new={progress['new_terminal_slots_this_invocation']} "
            f"estimated_list_usd="
            f"{progress['usage_and_list_cost']['estimated_total_usd']:.4f} "
            f"capability_status="
            f"{progress['capability_metric_assessment']['status']} "
            f"non_refusal_slot_coverage="
            f"{progress['capability_metric_assessment']['coverage']['non_refusal_slots']}/"
            f"{progress['capability_metric_assessment']['coverage']['expected_slots']} "
            f"out={out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
