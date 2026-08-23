#!/usr/bin/env python3
"""Budget-extended Anthropic Message Batches continuation for the sealed F2 pair.

Protocol:

* Claude Sonnet 5, adaptive thinking, high effort;
* K=2 over all 175 held-out tasks for one sealed enrichment arm;
* initial max_tokens=8192 through the first-party Message Batches API;
* a logical slot ending at ``max_tokens`` is retried at
  16384 -> 32768 -> 65536, without becoming a new sample;
* all native batch metadata/results and every intermediate response are
  persisted; only the final response for a logical slot is evaluated;
* provider model identity and usage are fail-closed;
* batch list-cost accounting uses the 2026 introductory $1/$5 per MTok
  input/output rates and a 16K per-request provider-input audit ceiling;
* each arm is capped at $50 including an explicitly adopted 8K source stage,
  keeping the paired continuation within the operator's $100 Anthropic budget.

This is a separate experimental protocol.  It does not modify the pinned
Qwen/DeepSeek runner or the unrestricted synchronous K=10/64K ceiling runner.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import frontier_passk as audited
import frontier_passk_anthropic as sync
from frontier_core import (
    JsonlJournal,
    PreflightError,
    ResponseContractError,
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    load_json,
    load_jsonl,
    sha256_file,
    sha256_text,
    stable_sha256,
    utc_now,
    wilson_interval,
)

SCHEMA = "anthropic-frontier-batch-budget100-v2"
TRANSPORT_SCHEMA = "anthropic-message-batches-adaptive-v1"
MODEL_ID = sync.MODEL_ID
EFFORT = "high"
K = 2
CAP_LADDER = (8192, 16384, 32768, 65536)
PROVIDER_PROMPT_AUDIT_CAP = 16_384
BATCH_INPUT_USD_PER_MILLION = 1.0
BATCH_OUTPUT_USD_PER_MILLION = 5.0
DEFAULT_ARM_COST_CAP_USD = 50.0
PRICE_VALID_THROUGH = sync.PRICE_VALID_THROUGH
PRICE_SOURCE = (
    "https://platform.claude.com/docs/en/build-with-claude/" "batch-processing"
)
MAX_RETRIES_PER_CAP = 3
BAD_GATE_MAX_SUCCESSES = 9
TOKEN_COUNT_MAX_ATTEMPTS = 5
TOKEN_COUNT_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0, 4.0)
ANTHROPIC_HTTP_TIMEOUT_SECONDS = 60.0

_BASE_PARSE_ARGS = sync._ORIGINAL_PARSE_ARGS
_BASE_FIXED_SLOT_POLICY = sync._ORIGINAL_FIXED_SLOT_POLICY
_BASE_CONFIG_FOR_HASH = sync._ORIGINAL_CONFIG_FOR_HASH


def _flag_present(argv: Sequence[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in argv)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    front = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    front.add_argument(
        "--action",
        choices=("preflight", "submit", "status", "harvest", "auto"),
        default=os.environ.get("BATCH_ACTION", "preflight"),
    )
    front.add_argument(
        "--anthropic-base-url",
        default=os.environ.get("ANTHROPIC_BASE_URL", sync.DEFAULT_BASE_URL),
    )
    front.add_argument(
        "--screen-cost-cap-usd",
        type=float,
        default=float(
            os.environ.get(
                "ANTHROPIC_SCREEN_ARM_COST_CAP_USD",
                str(DEFAULT_ARM_COST_CAP_USD),
            )
        ),
    )
    batch, remaining = front.parse_known_args(raw)
    for forbidden in (
        "--provider",
        "--api-key",
        "--qwen-env-file",
        "--deepseek-env-file",
        "--anthropic-effort",
        "--max-new-terminal-slots",
    ):
        if _flag_present(remaining, forbidden):
            front.error(f"{forbidden} is disabled by the batch-screen contract")

    updates = {
        "PROVIDER": "qwen",
        "MODEL": os.environ.get("MODEL", MODEL_ID),
        "K": os.environ.get("K", str(K)),
        "MAXTOK": os.environ.get("MAXTOK", str(CAP_LADDER[0])),
        "TEMPERATURE": os.environ.get("TEMPERATURE", "1"),
        "TOP_P": os.environ.get("TOP_P", "1"),
        "BUDGET": "0",
    }
    previous, missing = sync._temporary_environment(updates)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = old_argv
        sync._restore_environment(previous, missing)

    args.provider = sync.PROVIDER
    args.anthropic_base_url = batch.anthropic_base_url.rstrip("/")
    args.anthropic_effort = EFFORT
    args.action = batch.action
    args.screen_cost_cap_usd = batch.screen_cost_cap_usd
    if args.model != MODEL_ID:
        front.error(f"--model must be exactly {MODEL_ID!r}")
    if args.k != K:
        front.error(f"--k must be exactly {K}")
    if args.max_output_tokens != CAP_LADDER[0]:
        front.error(
            f"--max-output-tokens must be exactly {CAP_LADDER[0]} "
            "for the initial screen"
        )
    if args.temperature != 1.0 or args.top_p != 1.0:
        front.error("temperature/top-p must be defaults and are omitted on wire")
    if args.extra_body:
        front.error("--extra-body-json is unsupported")
    if args.input_mode != "prematerialized_f2":
        front.error("batch screen requires --input-mode prematerialized_f2")
    if args.limit:
        front.error("batch screen forbids --limit")
    if not 0 < args.screen_cost_cap_usd <= DEFAULT_ARM_COST_CAP_USD:
        front.error(f"--screen-cost-cap-usd must be in (0,{DEFAULT_ARM_COST_CAP_USD}]")
    if args.action != "preflight" and not os.environ.get("ANTHROPIC_API_KEY"):
        front.error("ANTHROPIC_API_KEY is required for batch API actions")
    return args


def resolve_api_configuration(args: argparse.Namespace) -> tuple[str, str]:
    return (
        os.environ.get("ANTHROPIC_API_KEY", ""),
        str(args.anthropic_base_url).rstrip("/"),
    )


def fixed_slot_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy = _BASE_FIXED_SLOT_POLICY(args)
    policy.pop("temperature", None)
    policy.pop("top_p", None)
    policy.pop("extra_body", None)
    policy.update(
        {
            "schema": "anthropic-batch-exact-logical-slot-v1",
            "transport_schema": TRANSPORT_SCHEMA,
            "k": K,
            "initial_max_output_tokens": CAP_LADDER[0],
            "length_retry_cap_ladder": list(CAP_LADDER),
            "length_retry_preserves_logical_slot": True,
            "only_final_logical_slot_response_is_evaluated": True,
            "token_count_max_attempts": TOKEN_COUNT_MAX_ATTEMPTS,
            "token_count_retry_delays_seconds": list(TOKEN_COUNT_RETRY_DELAYS_SECONDS),
            "token_count_retry_scope": (
                "free idempotent count_tokens transport, timeout, rate-limit, "
                "and provider-5xx errors only"
            ),
            "provider_connection_warmup": (
                "free models.list call on the shared client before token counting"
            ),
            "anthropic_http_timeout_seconds": ANTHROPIC_HTTP_TIMEOUT_SECONDS,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": EFFORT},
            "sampling_parameters_on_wire": {
                "temperature": "omitted",
                "top_p": "omitted",
                "top_k": "omitted",
            },
            "dispatch_order": "sample_index_then_sealed_task_order",
            "provider_prompt_audit_cap": PROVIDER_PROMPT_AUDIT_CAP,
            "batch_cost_cap_usd": args.screen_cost_cap_usd,
        }
    )
    return policy


def config_for_hash(args: argparse.Namespace) -> dict[str, Any]:
    config = _BASE_CONFIG_FOR_HASH(args)
    runtime = dict(config.get("runtime_identity") or {})
    runtime.pop("openai_sdk_version", None)
    try:
        version: str | None = __import__("importlib").metadata.version("anthropic")
    except Exception:
        version = None
    runtime.update(
        {
            "audited_runner_sha256": runtime.pop("runner_sha256"),
            "batch_runner_sha256": sha256_file(Path(__file__).resolve()),
            "anthropic_sync_normalizer_sha256": sha256_file(
                Path(sync.__file__).resolve()
            ),
            "anthropic_sdk_version": version,
        }
    )
    config["runtime_identity"] = runtime
    config["provider"] = sync.PROVIDER
    config["anthropic_batch_screen"] = {
        "schema": SCHEMA,
        "transport_schema": TRANSPORT_SCHEMA,
        "model": MODEL_ID,
        "k": K,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
        "cap_ladder": list(CAP_LADDER),
        "provider_prompt_audit_cap": PROVIDER_PROMPT_AUDIT_CAP,
        "batch_cost_cap_usd": args.screen_cost_cap_usd,
        "batch_input_usd_per_million": BATCH_INPUT_USD_PER_MILLION,
        "batch_output_usd_per_million": BATCH_OUTPUT_USD_PER_MILLION,
        "price_valid_through": PRICE_VALID_THROUGH,
        "sampling_parameters_omitted": True,
    }
    # action is operational state and intentionally excluded.
    return config


def install_hooks() -> None:
    audited.resolve_api_configuration = resolve_api_configuration
    audited.fixed_slot_policy = fixed_slot_policy
    audited.config_for_hash = config_for_hash


def _jsonable(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            result = value.model_dump(mode="json")
        except TypeError:
            result = value.model_dump()
        if isinstance(result, dict):
            return json.loads(json.dumps(result, default=str))
    if hasattr(value, "dict"):
        result = value.dict()
        if isinstance(result, dict):
            return json.loads(json.dumps(result, default=str))
    raise audited.RunFailure(f"cannot serialize {type(value).__name__}")


def _client(args: argparse.Namespace) -> tuple[Any, str, str]:
    key, base_url = audited.api_credentials(args)
    try:
        import anthropic
    except Exception as exc:
        raise PreflightError("the anthropic Python package is required") from exc
    return (
        anthropic.Anthropic(
            api_key=key,
            base_url=base_url,
            max_retries=0,
            timeout=ANTHROPIC_HTTP_TIMEOUT_SECONDS,
        ),
        key,
        base_url,
    )


def _retryable_token_count_error(exc: BaseException) -> bool:
    """Limit free/idempotent token-count retries to transient provider failures."""

    try:
        import anthropic
    except Exception:
        return False
    retryable_types = tuple(
        error_type
        for error_type in (
            getattr(anthropic, "APIConnectionError", None),
            getattr(anthropic, "APITimeoutError", None),
            getattr(anthropic, "RateLimitError", None),
            getattr(anthropic, "InternalServerError", None),
        )
        if isinstance(error_type, type)
    )
    return bool(retryable_types) and isinstance(exc, retryable_types)


def _count_tokens_with_retries(client: Any, request: Mapping[str, Any]) -> Any:
    """Execute one free token count with a bounded, explicit retry policy."""

    for attempt in range(TOKEN_COUNT_MAX_ATTEMPTS):
        try:
            return client.messages.count_tokens(**dict(request))
        except Exception as exc:
            if (
                not _retryable_token_count_error(exc)
                or attempt + 1 >= TOKEN_COUNT_MAX_ATTEMPTS
            ):
                raise
            time.sleep(TOKEN_COUNT_RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("unreachable token-count retry state")


def _warm_anthropic_connection(client: Any) -> dict[str, Any]:
    """Warm the shared client and attest that the exact target model exists."""

    for attempt in range(TOKEN_COUNT_MAX_ATTEMPTS):
        try:
            response = client.models.list(limit=100)
            raw = _jsonable(response)
            model_ids = [
                str(row.get("id") or "")
                for row in raw.get("data", [])
                if isinstance(row, Mapping)
            ]
            if MODEL_ID not in model_ids:
                raise audited.RunFailure(
                    f"target model {MODEL_ID!r} is absent from provider model catalog"
                )
            return {
                "target_model": MODEL_ID,
                "target_model_present": True,
                "catalog_size": len(model_ids),
            }
        except Exception as exc:
            if isinstance(exc, audited.RunFailure):
                raise
            if (
                not _retryable_token_count_error(exc)
                or attempt + 1 >= TOKEN_COUNT_MAX_ATTEMPTS
            ):
                raise
            time.sleep(TOKEN_COUNT_RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("unreachable provider warm-up retry state")


def _count_input_tokens(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: Sequence[Mapping[str, Any]],
    config_sha: str,
    client: Any,
    api_key: str,
) -> dict[str, Any]:
    """Count every unique sealed prompt with the exact target model/request."""

    model_catalog_attestation = _warm_anthropic_connection(client)
    rows_path = out / "anthropic_input_token_counts.jsonl"
    audit_path = out / "anthropic_input_token_audit.json"
    existing = (
        load_jsonl(rows_path, "Anthropic input-token counts")
        if rows_path.is_file()
        else []
    )
    by_task: dict[str, dict[str, Any]] = {}
    for row in existing:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure("foreign Anthropic token-count journal")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in by_task:
            raise audited.RunFailure("duplicate/malformed Anthropic token count")
        by_task[task_id] = row

    journal = JsonlJournal(rows_path)

    def count_one(index_plan: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
        index, plan = index_plan
        task_id = str(plan["task_id"])
        system, messages = sync._split_anthropic_messages(plan["messages"])
        try:
            response = _count_tokens_with_retries(
                client,
                {
                    "model": args.model,
                    "system": system,
                    "messages": messages,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": EFFORT},
                },
            )
        except Exception as exc:
            raise audited.RunFailure(
                f"Anthropic token counting failed for {task_id}: "
                + sync._redact_exception(exc, api_key)
            ) from exc
        raw = _jsonable(response)
        count = raw.get("input_tokens")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise audited.RunFailure(
                f"Anthropic token count is invalid for {task_id}: {count!r}"
            )
        return {
            "schema": SCHEMA,
            "record_type": "anthropic_input_token_count",
            "counted_at": utc_now(),
            "config_sha256": config_sha,
            "task_id": task_id,
            "task_index": index,
            "prompt_sha256": plan["prompt_sha256"],
            "model": args.model,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": EFFORT},
            "input_tokens": count,
            "response": raw,
        }

    missing = [
        (index, plan)
        for index, plan in enumerate(plans)
        if str(plan["task_id"]) not in by_task
    ]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(missing))
        ) as pool:
            futures = {pool.submit(count_one, item): item for item in missing}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                journal.append(row)
                by_task[str(row["task_id"])] = row

    ordered: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        task_id = str(plan["task_id"])
        row = by_task.get(task_id)
        if row is None:
            raise audited.RunFailure(f"missing Anthropic token count for {task_id}")
        if int(row.get("task_index", -1)) != index:
            raise audited.RunFailure("Anthropic token-count task order mismatch")
        if row.get("prompt_sha256") != plan["prompt_sha256"]:
            raise audited.RunFailure("Anthropic token-count prompt hash mismatch")
        if row.get("model") != args.model:
            raise audited.RunFailure("Anthropic token-count model mismatch")
        count = int(row["input_tokens"])
        if count > PROVIDER_PROMPT_AUDIT_CAP:
            raise audited.RunFailure(
                f"task {task_id} has {count} Anthropic input tokens, exceeding "
                f"the sealed provider cap {PROVIDER_PROMPT_AUDIT_CAP}"
            )
        ordered.append(row)
    counts = [int(row["input_tokens"]) for row in ordered]
    audit = {
        "schema": SCHEMA,
        "record_type": "anthropic_input_token_audit",
        "completed_at": utc_now(),
        "config_sha256": config_sha,
        "model": args.model,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
        "unique_prompts_counted": len(ordered),
        "logical_requests_covered": len(ordered) * K,
        "provider_prompt_audit_cap": PROVIDER_PROMPT_AUDIT_CAP,
        "minimum_input_tokens": min(counts),
        "maximum_input_tokens": max(counts),
        "total_unique_prompt_input_tokens": sum(counts),
        "all_counts_within_cap": True,
        "model_catalog_attestation": model_catalog_attestation,
        "ordered_count_rows_sha256": stable_sha256(ordered),
        "count_journal": file_record(rows_path),
        "token_counting_is_free_but_rate_limited": True,
    }
    if audit_path.exists():
        prior = load_json(audit_path, "Anthropic input-token audit")
        comparable_keys = (
            "schema",
            "record_type",
            "config_sha256",
            "model",
            "thinking",
            "output_config",
            "unique_prompts_counted",
            "logical_requests_covered",
            "provider_prompt_audit_cap",
            "minimum_input_tokens",
            "maximum_input_tokens",
            "total_unique_prompt_input_tokens",
            "all_counts_within_cap",
            "model_catalog_attestation",
            "ordered_count_rows_sha256",
            "count_journal",
            "token_counting_is_free_but_rate_limited",
        )
        if any(prior.get(key) != audit.get(key) for key in comparable_keys):
            raise audited.RunFailure(
                "persisted Anthropic input-token audit no longer matches counts"
            )
        return prior
    atomic_write_json(audit_path, audit)
    return audit


def _custom_id(
    task_index: int,
    sample_index: int,
    cap: int,
    attempt_index: int,
) -> str:
    value = f"a{attempt_index:02d}_s{sample_index:02d}_t{task_index:03d}_c{cap:05d}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise audited.RunFailure(f"invalid batch custom_id: {value!r}")
    return value


def _slot_key(task_id: str, sample_index: int) -> str:
    return f"{task_id}\x1f{sample_index}"


def _batch_events(out: Path, config_sha: str) -> list[dict[str, Any]]:
    path = out / "batch_events.jsonl"
    rows = load_jsonl(path, "batch event journal") if path.is_file() else []
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure("foreign batch event journal")
    return rows


def _slot_attempts(out: Path, config_sha: str) -> list[dict[str, Any]]:
    path = out / "batch_slot_attempts.jsonl"
    rows = load_jsonl(path, "batch slot-attempt journal") if path.is_file() else []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure("foreign batch slot-attempt journal")
        key = (str(row.get("batch_id") or ""), str(row.get("custom_id") or ""))
        if not all(key) or key in seen:
            raise audited.RunFailure("duplicate/malformed batch slot-attempt")
        seen.add(key)
    return rows


def _terminal_rows(out: Path, config_sha: str) -> list[dict[str, Any]]:
    path = out / "terminal_slots.jsonl"
    rows = load_jsonl(path, "terminal slot journal") if path.is_file() else []
    seen: set[str] = set()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure("foreign terminal slot journal")
        key = _slot_key(str(row.get("task_id") or ""), int(row["sample_index"]))
        if key in seen:
            raise audited.RunFailure("duplicate terminal logical slot")
        seen.add(key)
    return rows


def _outcome_rows(out: Path, config_sha: str) -> list[dict[str, Any]]:
    path = out / "outcomes.jsonl"
    rows = load_jsonl(path, "batch-screen outcome journal") if path.is_file() else []
    seen: set[str] = set()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure("foreign batch-screen outcome journal")
        key = _slot_key(str(row.get("task_id") or ""), int(row["sample_index"]))
        if key in seen:
            raise audited.RunFailure("duplicate logical-slot outcome")
        seen.add(key)
    return rows


def _primary_outcome_rows(out: Path, config_sha: str) -> list[dict[str, Any]]:
    path = out / "primary_8192_outcomes.jsonl"
    rows = load_jsonl(path, "primary 8192 outcome journal") if path.is_file() else []
    seen: set[str] = set()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure("foreign primary 8192 outcome journal")
        if int(row.get("requested_max_tokens", -1)) != CAP_LADDER[0]:
            raise audited.RunFailure("primary outcome is not from the 8192 cap")
        key = _slot_key(str(row.get("task_id") or ""), int(row["sample_index"]))
        if key in seen:
            raise audited.RunFailure("duplicate primary 8192 logical slot")
        seen.add(key)
    return rows


def actual_batch_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    successful_responses = 0
    for row in rows:
        if row.get("result_type") != "succeeded":
            continue
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            raise audited.RunFailure("successful batch result has no usage")
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if (
            isinstance(prompt, bool)
            or not isinstance(prompt, int)
            or isinstance(completion, bool)
            or not isinstance(completion, int)
        ):
            raise audited.RunFailure("successful batch result usage is malformed")
        input_tokens += prompt
        output_tokens += completion
        successful_responses += 1
    input_usd = input_tokens * BATCH_INPUT_USD_PER_MILLION / 1_000_000
    output_usd = output_tokens * BATCH_OUTPUT_USD_PER_MILLION / 1_000_000
    return {
        "schema": "anthropic-batch-list-cost-v1",
        "successful_responses": successful_responses,
        "provider_reported_input_tokens": input_tokens,
        "provider_reported_output_tokens": output_tokens,
        "provider_reported_total_tokens": input_tokens + output_tokens,
        "input_usd_per_million": BATCH_INPUT_USD_PER_MILLION,
        "output_usd_per_million": BATCH_OUTPUT_USD_PER_MILLION,
        "estimated_input_usd": input_usd,
        "estimated_output_usd": output_usd,
        "estimated_total_usd": input_usd + output_usd,
        "price_valid_through": PRICE_VALID_THROUGH,
        "price_source": PRICE_SOURCE,
        "estimate_not_invoice": True,
    }


def worst_batch_cost(request_specs: Sequence[Mapping[str, Any]]) -> float:
    return sum(
        (
            PROVIDER_PROMPT_AUDIT_CAP * BATCH_INPUT_USD_PER_MILLION
            + int(spec["cap"]) * BATCH_OUTPUT_USD_PER_MILLION
        )
        / 1_000_000
        for spec in request_specs
    )


def _active_submission(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    submissions = [row for row in events if row.get("event_type") == "batch_submitted"]
    harvested = {
        str(row.get("batch_id") or "")
        for row in events
        if row.get("event_type") == "batch_harvested"
    }
    active = [row for row in submissions if str(row["batch_id"]) not in harvested]
    if len(active) > 1:
        raise audited.RunFailure("more than one unharvested batch exists")
    return active[0] if active else None


def pending_request_specs(
    plans: Sequence[Mapping[str, Any]],
    slot_attempts: Sequence[Mapping[str, Any]],
    terminals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    terminal_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"])) for row in terminals
    }
    attempts_by_slot: dict[str, list[Mapping[str, Any]]] = {}
    for row in slot_attempts:
        attempts_by_slot.setdefault(
            _slot_key(str(row["task_id"]), int(row["sample_index"])), []
        ).append(row)
    specs: list[dict[str, Any]] = []
    for sample_index in range(K):
        for task_index, plan in enumerate(plans):
            task_id = str(plan["task_id"])
            key = _slot_key(task_id, sample_index)
            if key in terminal_keys:
                continue
            history = attempts_by_slot.get(key, [])
            if not history:
                cap = CAP_LADDER[0]
                attempt_index = 0
            else:
                last = history[-1]
                prior_cap = int(last["requested_max_tokens"])
                result_type = str(last["result_type"])
                if result_type == "succeeded":
                    if last.get("finish_reason") != "length":
                        raise audited.RunFailure(
                            "non-length success is missing terminal row"
                        )
                    try:
                        cap = CAP_LADDER[CAP_LADDER.index(prior_cap) + 1]
                    except (ValueError, IndexError) as exc:
                        raise audited.RunFailure(
                            "final-cap length result is missing terminal row"
                        ) from exc
                    attempt_index = 0
                elif result_type in {"errored", "canceled", "expired"}:
                    cap = prior_cap
                    attempt_index = int(last["cap_attempt_index"]) + 1
                    if attempt_index >= MAX_RETRIES_PER_CAP:
                        raise audited.RunFailure(
                            f"slot {task_id}/{sample_index} exhausted batch retries"
                        )
                else:
                    raise audited.RunFailure(f"unknown result type {result_type!r}")
            specs.append(
                {
                    "task_id": task_id,
                    "task_index": task_index,
                    "sample_index": sample_index,
                    "cap": cap,
                    "cap_attempt_index": attempt_index,
                    "custom_id": _custom_id(
                        task_index, sample_index, cap, attempt_index
                    ),
                }
            )
    return specs


def _request_for_spec(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    system, messages = sync._split_anthropic_messages(plan["messages"])
    params = {
        "model": args.model,
        "max_tokens": int(spec["cap"]),
        "system": system,
        "messages": messages,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
    }
    return {"custom_id": str(spec["custom_id"]), "params": params}


def _submit(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
) -> dict[str, Any]:
    events = _batch_events(out, config_sha)
    if _active_submission(events) is not None:
        raise audited.RunFailure("an unharvested batch already exists")
    slot_attempts = _slot_attempts(out, config_sha)
    terminals = _terminal_rows(out, config_sha)
    specs = pending_request_specs(plans, slot_attempts, terminals)
    if not specs:
        return {"status": "nothing_pending"}
    client, key, _base = _client(args)
    token_audit = _count_input_tokens(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
        client=client,
        api_key=key,
    )
    actual_cost = actual_batch_cost(slot_attempts)
    worst_next = worst_batch_cost(specs)
    projected = actual_cost["estimated_total_usd"] + worst_next
    if projected > args.screen_cost_cap_usd + 1e-12:
        raise audited.RunFailure(
            "batch submission blocked by cost gate: actual list estimate "
            f"${actual_cost['estimated_total_usd']:.6f} + worst next batch "
            f"${worst_next:.6f} > arm cap ${args.screen_cost_cap_usd:.6f}"
        )

    plan_by_task = {str(plan["task_id"]): plan for plan in plans}
    requests = [
        _request_for_spec(args, plan_by_task[str(spec["task_id"])], spec)
        for spec in specs
    ]
    request_commitment = stable_sha256(requests)
    intent_path = out / "batch_submission_intent.json"
    if intent_path.exists():
        prior = load_json(intent_path, "batch submission intent")
        if prior.get("status") == "submitted":
            matching_event = any(
                row.get("event_type") == "batch_submitted"
                and row.get("batch_id") == prior.get("batch_id")
                and row.get("request_commitment_sha256")
                == prior.get("request_commitment_sha256")
                for row in events
            )
            if matching_event:
                intent_path.unlink()
            else:
                raise audited.RunFailure(
                    "submitted intent exists without event reconciliation"
                )
        if not intent_path.exists():
            prior = {}
        elif prior.get("request_commitment_sha256") != request_commitment:
            raise audited.RunFailure(
                "unresolved batch intent has a different request set"
            )
        else:
            raise audited.RunFailure(
                "unresolved pre-submit intent exists; reconcile workspace batches "
                "before risking a duplicate paid submission"
            )
    intent = {
        "schema": SCHEMA,
        "status": "prepared",
        "created_at": utc_now(),
        "config_sha256": config_sha,
        "request_commitment_sha256": request_commitment,
        "request_count": len(requests),
        "request_specs": specs,
        "anthropic_input_token_audit": file_record(
            out / "anthropic_input_token_audit.json"
        ),
        "worst_next_batch_list_usd": worst_next,
        "actual_prior_list_usd": actual_cost["estimated_total_usd"],
        "projected_list_usd": projected,
    }
    atomic_write_json(intent_path, intent)

    try:
        response = client.messages.batches.create(requests=requests)
    except Exception as exc:
        # The call may have reached the service.  Keep the prepared intent and
        # require reconciliation rather than blindly resubmitting.
        raise audited.RunFailure(
            "batch submission returned no receipt; reconcile before retry: "
            + sync._redact_exception(exc, key)
        ) from exc
    batch = _jsonable(response)
    batch_id = str(batch.get("id") or "")
    if not batch_id:
        raise audited.RunFailure("batch creation receipt has no id")
    intent["status"] = "submitted"
    intent["batch_id"] = batch_id
    intent["submitted_at"] = utc_now()
    atomic_write_json(intent_path, intent)
    event = {
        "schema": SCHEMA,
        "event_type": "batch_submitted",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "batch_id": batch_id,
        "request_commitment_sha256": request_commitment,
        "request_count": len(requests),
        "request_specs": specs,
        "anthropic_input_token_audit": file_record(
            out / "anthropic_input_token_audit.json"
        ),
        "anthropic_input_token_audit_claims": {
            "model": token_audit["model"],
            "unique_prompts_counted": token_audit["unique_prompts_counted"],
            "logical_requests_covered": token_audit["logical_requests_covered"],
            "maximum_input_tokens": token_audit["maximum_input_tokens"],
            "provider_prompt_audit_cap": token_audit["provider_prompt_audit_cap"],
            "all_counts_within_cap": token_audit["all_counts_within_cap"],
        },
        "worst_batch_list_usd": worst_next,
        "projected_cumulative_list_usd": projected,
        "batch": batch,
    }
    JsonlJournal(out / "batch_events.jsonl").append(event)
    intent_path.unlink()
    return {
        "status": "submitted",
        "batch_id": batch_id,
        "requests": len(requests),
        "projected_list_usd": projected,
    }


def _retrieve_active(
    args: argparse.Namespace,
    *,
    out: Path,
    config_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    events = _batch_events(out, config_sha)
    active = _active_submission(events)
    if active is None:
        raise audited.RunFailure("there is no unharvested batch")
    client, key, _base = _client(args)
    try:
        response = client.messages.batches.retrieve(str(active["batch_id"]))
    except Exception as exc:
        raise audited.RunFailure(
            "batch status retrieval failed: " + sync._redact_exception(exc, key)
        ) from exc
    batch = _jsonable(response)
    if str(batch.get("id") or "") != str(active["batch_id"]):
        raise audited.RunFailure("retrieved batch id differs from submitted id")
    event = {
        "schema": SCHEMA,
        "event_type": "batch_status",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "batch_id": active["batch_id"],
        "processing_status": batch.get("processing_status"),
        "request_counts": batch.get("request_counts"),
        "batch": batch,
    }
    JsonlJournal(out / "batch_events.jsonl").append(event)
    return active, batch, client


def _status(
    args: argparse.Namespace,
    *,
    out: Path,
    config_sha: str,
) -> dict[str, Any]:
    active, batch, _client_value = _retrieve_active(
        args, out=out, config_sha=config_sha
    )
    return {
        "status": str(batch.get("processing_status") or "unknown"),
        "batch_id": active["batch_id"],
        "request_counts": batch.get("request_counts"),
    }


def _result_payload(result: Mapping[str, Any]) -> tuple[str, Any]:
    result_body = result.get("result")
    if not isinstance(result_body, Mapping):
        raise audited.RunFailure("batch result has no result object")
    result_type = str(result_body.get("type") or "")
    if result_type == "succeeded":
        message = result_body.get("message")
        if not isinstance(message, Mapping):
            raise audited.RunFailure("succeeded batch result has no message")
        return result_type, message
    if result_type in {"errored", "canceled", "expired"}:
        return result_type, result_body.get("error")
    raise audited.RunFailure(f"unknown batch result type: {result_type!r}")


def _failed_evaluation() -> dict[str, Any]:
    return {
        "compiled": False,
        "passed": False,
        "completion_attestation_id": audited.REQUIRED_ATTESTATION_ID,
        "completion_attestation_enforced": False,
        "completion_attestation_satisfied_all_runs": False,
        "stability_runs": [],
    }


def _evaluate_terminal(
    args: argparse.Namespace,
    *,
    evaluator: Any,
    plan: Mapping[str, Any],
    terminal: Any,
    task_id: str,
    sample_index: int,
    force_capacity_failure: bool,
) -> tuple[dict[str, Any], bool]:
    if force_capacity_failure or not terminal.candidate_valid:
        return _failed_evaluation(), False
    return (
        audited.evaluate_candidate_stably(
            evaluator,
            code=terminal.code,
            tests=plan["row"]["acceptance_tests"],
            task_id=task_id,
            sample_index=sample_index,
            stability_runs=args.eval_stability_runs,
            timeout=args.eval_timeout_seconds,
        ),
        True,
    )


def _outcome_payload(
    *,
    config_sha: str,
    task_id: str,
    task_index: int,
    sample_index: int,
    batch_id: str,
    custom_id: str,
    cap: int,
    terminal: Any,
    evaluator_record: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evaluation_performed: bool,
    metric_role: str,
    capacity_exhausted: bool,
    native_stop_reason: str | None = None,
    native_stop_details_category: str | None = None,
    native_stop_category: str = "missing",
) -> dict[str, Any]:
    candidate_valid = bool(terminal.candidate_valid) and not capacity_exhausted
    terminal_reason = (
        f"capacity_exhausted_at_{cap}"
        if capacity_exhausted
        else terminal.terminal_reason
    )
    return {
        "schema": SCHEMA,
        "record_type": "logical_slot_outcome",
        "metric_role": metric_role,
        "evaluated_at": utc_now(),
        "config_sha256": config_sha,
        "task_id": task_id,
        "task_index": task_index,
        "sample_index": sample_index,
        "batch_id": batch_id,
        "custom_id": custom_id,
        "requested_max_tokens": cap,
        "response_id": terminal.response_id,
        "finish_reason": terminal.finish_reason,
        "native_stop_reason": native_stop_reason,
        "native_stop_details_category": native_stop_details_category,
        "native_stop_category": native_stop_category,
        "capacity_exhausted": capacity_exhausted,
        "candidate_valid": candidate_valid,
        "terminal_reason": terminal_reason,
        "code_sha256": terminal.code_sha256,
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
    }


def _harvest(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    prompt_map: dict[str, dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    active, batch, client = _retrieve_active(args, out=out, config_sha=config_sha)
    if batch.get("processing_status") != "ended":
        return {
            "status": str(batch.get("processing_status") or "unknown"),
            "batch_id": active["batch_id"],
        }
    batch_id = str(active["batch_id"])
    try:
        result_values = [
            _jsonable(value) for value in client.messages.batches.results(batch_id)
        ]
    except Exception as exc:
        key, _base = resolve_api_configuration(args)
        raise audited.RunFailure(
            "batch result retrieval failed: " + sync._redact_exception(exc, key)
        ) from exc

    plan_by_task = {str(plan["task_id"]): plan for plan in plans}
    reconstructed_requests = [
        _request_for_spec(
            args,
            plan_by_task[str(spec["task_id"])],
            spec,
        )
        for spec in active["request_specs"]
    ]
    if stable_sha256(reconstructed_requests) != active.get("request_commitment_sha256"):
        raise audited.RunFailure("submitted batch request commitment was tampered")
    expected_specs = {str(spec["custom_id"]): spec for spec in active["request_specs"]}
    observed_ids = [str(row.get("custom_id") or "") for row in result_values]
    if len(observed_ids) != len(set(observed_ids)):
        raise audited.RunFailure("batch result contains duplicate custom_id")
    if set(observed_ids) != set(expected_specs):
        raise audited.RunFailure("batch result custom_id set differs from request set")
    results_path = out / f"batch_results_{batch_id}.jsonl"
    if results_path.exists():
        existing = load_jsonl(results_path, "persisted raw batch results")
        if stable_sha256(existing) != stable_sha256(result_values):
            raise audited.RunFailure(
                "re-fetched batch results differ from persisted copy"
            )
    else:
        atomic_write_jsonl(results_path, result_values)

    if not args.expected_evaluator_sha256.strip():
        raise PreflightError("--expected-evaluator-sha256 is required")
    if not args.expected_dart_sha256.strip():
        raise PreflightError("--expected-dart-sha256 is required")
    evaluator_module, evaluator_record = audited.import_evaluator(
        args.evaluator_module,
        args.expected_evaluator_sha256,
        dart_binary=args.dart,
        expected_dart_hash=args.expected_dart_sha256,
        validate_dart=True,
    )
    evaluator = evaluator_module.evaluate_dart_jit_tests_detail
    if evaluator_record["sha256"] != provenance["evaluator"]["sha256"]:
        raise PreflightError("evaluator changed after sealed preflight")

    slot_rows = _slot_attempts(out, config_sha)
    attempt_by_transport = {
        (str(row["batch_id"]), str(row["custom_id"])): row for row in slot_rows
    }
    terminal_rows = _terminal_rows(out, config_sha)
    terminal_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"]))
        for row in terminal_rows
    }
    outcome_rows = _outcome_rows(out, config_sha)
    outcome_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"])) for row in outcome_rows
    }
    primary_rows = _primary_outcome_rows(out, config_sha)
    primary_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"])) for row in primary_rows
    }
    attempts_journal = JsonlJournal(out / "batch_slot_attempts.jsonl")
    terminals_journal = JsonlJournal(out / "terminal_slots.jsonl")
    outcomes_journal = JsonlJournal(out / "outcomes.jsonl")
    primary_journal = JsonlJournal(out / "primary_8192_outcomes.jsonl")

    for raw_result in result_values:
        custom_id = str(raw_result["custom_id"])
        spec = expected_specs[custom_id]
        task_id = str(spec["task_id"])
        sample_index = int(spec["sample_index"])
        cap = int(spec["cap"])
        slot_key = _slot_key(task_id, sample_index)
        transport_key = (batch_id, custom_id)
        existing_attempt = attempt_by_transport.get(transport_key)
        if existing_attempt is not None:
            result_type = str(existing_attempt["result_type"])
            normalized_value = existing_attempt.get("normalized_response")
            normalized = (
                dict(normalized_value)
                if isinstance(normalized_value, Mapping)
                else None
            )
        else:
            result_type, payload = _result_payload(raw_result)
            normalized = None
            finish_reason: str | None = None
            usage: dict[str, int] | None = None
            error_type: str | None = None
            error_text: str | None = None
            if result_type == "succeeded":
                normalized = sync.normalize_anthropic_response(payload)
                classified = audited.classify_terminal_provider_response(
                    normalized,
                    expected_model=args.model,
                    max_prompt_tokens=PROVIDER_PROMPT_AUDIT_CAP,
                    requested_max_tokens=cap,
                )
                finish_reason = classified.finish_reason
                usage = classified.usage
                stop_metadata = sync.native_stop_metadata(normalized)
            else:
                stop_metadata = {
                    "native_stop_reason": None,
                    "native_stop_details_category": None,
                    "native_stop_category": "missing",
                }
                error_object = payload if isinstance(payload, Mapping) else {}
                nested = (
                    error_object.get("error")
                    if isinstance(error_object.get("error"), Mapping)
                    else error_object
                )
                error_type = str(
                    (nested or {}).get("type")
                    if isinstance(nested, Mapping)
                    else result_type
                )
                error_text = str(
                    (nested or {}).get("message")
                    if isinstance(nested, Mapping)
                    else result_type
                )[:1000]
                if error_type == "invalid_request_error":
                    raise audited.RunFailure(
                        f"batch validation failed for {custom_id}: {error_text}"
                    )
            attempt_row = {
                "schema": SCHEMA,
                "record_type": "batch_slot_attempt",
                "recorded_at": utc_now(),
                "config_sha256": config_sha,
                "batch_id": batch_id,
                "custom_id": custom_id,
                "task_id": task_id,
                "task_index": int(spec["task_index"]),
                "sample_index": sample_index,
                "requested_max_tokens": cap,
                "cap_attempt_index": int(spec["cap_attempt_index"]),
                "result_type": result_type,
                "finish_reason": finish_reason,
                **stop_metadata,
                "usage": usage,
                "error_type": error_type,
                "error": error_text,
                "normalized_response": normalized,
                "native_batch_result": raw_result,
            }
            attempts_journal.append(attempt_row)
            attempt_by_transport[transport_key] = attempt_row
        if result_type != "succeeded":
            continue
        if normalized is None:
            raise audited.RunFailure(
                "succeeded batch attempt lacks normalized response"
            )
        terminal = audited.classify_terminal_provider_response(
            normalized,
            expected_model=args.model,
            max_prompt_tokens=PROVIDER_PROMPT_AUDIT_CAP,
            requested_max_tokens=cap,
        )
        stop_metadata = sync.native_stop_metadata(normalized)

        primary_evaluation: dict[str, Any] | None = None
        primary_evaluation_performed = False
        if cap == CAP_LADDER[0] and slot_key not in primary_keys:
            primary_capacity_exhausted = terminal.finish_reason == "length"
            (
                primary_evaluation,
                primary_evaluation_performed,
            ) = _evaluate_terminal(
                args,
                evaluator=evaluator,
                plan=plan_by_task[task_id],
                terminal=terminal,
                task_id=task_id,
                sample_index=sample_index,
                force_capacity_failure=primary_capacity_exhausted,
            )
            primary_outcome = _outcome_payload(
                config_sha=config_sha,
                task_id=task_id,
                task_index=int(spec["task_index"]),
                sample_index=sample_index,
                batch_id=batch_id,
                custom_id=custom_id,
                cap=cap,
                terminal=terminal,
                evaluator_record=evaluator_record,
                evaluation=primary_evaluation,
                evaluation_performed=primary_evaluation_performed,
                metric_role="primary_fixed_cap_8192",
                capacity_exhausted=primary_capacity_exhausted,
                **stop_metadata,
            )
            primary_journal.append(primary_outcome)
            primary_keys.add(slot_key)

        is_final = terminal.finish_reason != "length" or cap == CAP_LADDER[-1]
        if not is_final:
            continue
        capacity_exhausted = terminal.finish_reason == "length"
        if slot_key not in outcome_keys:
            if (
                cap == CAP_LADDER[0]
                and primary_evaluation is not None
                and capacity_exhausted == (terminal.finish_reason == "length")
            ):
                evaluation = primary_evaluation
                evaluation_performed = primary_evaluation_performed
            else:
                evaluation, evaluation_performed = _evaluate_terminal(
                    args,
                    evaluator=evaluator,
                    plan=plan_by_task[task_id],
                    terminal=terminal,
                    task_id=task_id,
                    sample_index=sample_index,
                    force_capacity_failure=capacity_exhausted,
                )
            outcome = _outcome_payload(
                config_sha=config_sha,
                task_id=task_id,
                task_index=int(spec["task_index"]),
                sample_index=sample_index,
                batch_id=batch_id,
                custom_id=custom_id,
                cap=cap,
                terminal=terminal,
                evaluator_record=evaluator_record,
                evaluation=evaluation,
                evaluation_performed=evaluation_performed,
                metric_role="capacity_adaptive",
                capacity_exhausted=capacity_exhausted,
                **stop_metadata,
            )
            outcomes_journal.append(outcome)
            outcome_keys.add(slot_key)
        if slot_key in terminal_keys:
            continue
        terminal_row = {
            "schema": SCHEMA,
            "record_type": "terminal_logical_slot",
            "recorded_at": utc_now(),
            "config_sha256": config_sha,
            "batch_id": batch_id,
            "custom_id": custom_id,
            "task_id": task_id,
            "task_index": int(spec["task_index"]),
            "sample_index": sample_index,
            "requested_max_tokens": cap,
            "response_id": terminal.response_id,
            "resolved_model": terminal.response_model,
            "finish_reason": terminal.finish_reason,
            **stop_metadata,
            "capacity_exhausted": capacity_exhausted,
            "candidate_valid": (
                bool(terminal.candidate_valid) and not capacity_exhausted
            ),
            "terminal_reason": (
                f"capacity_exhausted_at_{cap}"
                if capacity_exhausted
                else terminal.terminal_reason
            ),
            "code": terminal.code,
            "code_sha256": terminal.code_sha256,
            "usage": terminal.usage,
        }
        terminals_journal.append(terminal_row)
        terminal_keys.add(slot_key)

    event = {
        "schema": SCHEMA,
        "event_type": "batch_harvested",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "batch_id": batch_id,
        "request_count": len(result_values),
        "raw_results": file_record(results_path),
    }
    JsonlJournal(out / "batch_events.jsonl").append(event)
    return _write_progress_or_summary(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
        provenance=provenance,
        evaluator_record=evaluator_record,
    )


def _rows_with_native_stop_metadata(
    rows: Sequence[Mapping[str, Any]],
    slot_attempts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach native stop metadata, including for pre-patch durable journals."""

    attempt_by_transport: dict[tuple[str, str], Mapping[str, Any]] = {}
    for attempt in slot_attempts:
        batch_id = str(attempt.get("batch_id") or "")
        custom_id = str(attempt.get("custom_id") or "")
        if not batch_id or not custom_id:
            continue
        key = (batch_id, custom_id)
        if key in attempt_by_transport:
            raise audited.RunFailure(
                f"duplicate batch attempt transport identity: {key!r}"
            )
        attempt_by_transport[key] = attempt

    enriched: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        metadata = sync.native_stop_metadata_from_record(value)
        if metadata["native_stop_reason"] is None:
            key = (
                str(value.get("batch_id") or ""),
                str(value.get("custom_id") or ""),
            )
            attempt = attempt_by_transport.get(key)
            if attempt is not None:
                metadata = sync.native_stop_metadata_from_record(attempt)
        value.update(metadata)
        enriched.append(value)
    return enriched


def _write_progress_or_summary(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
    evaluator_record: Mapping[str, Any],
) -> dict[str, Any]:
    slot_attempts = _slot_attempts(out, config_sha)
    terminals = _terminal_rows(out, config_sha)
    outcomes = _outcome_rows(out, config_sha)
    primary_outcomes = _primary_outcome_rows(out, config_sha)
    if len(terminals) != len(outcomes):
        raise audited.RunFailure("terminal and outcome counts differ")
    terminal_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"])) for row in terminals
    }
    outcome_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"])) for row in outcomes
    }
    if terminal_keys != outcome_keys:
        raise audited.RunFailure("terminal and outcome logical slots differ")
    total_slots = len(plans) * K
    cost = actual_batch_cost(slot_attempts)
    pending = pending_request_specs(plans, slot_attempts, terminals)
    task_ids = [str(plan["task_id"]) for plan in plans]
    terminal_reporting_rows = _rows_with_native_stop_metadata(terminals, slot_attempts)
    primary_reporting_rows = _rows_with_native_stop_metadata(
        primary_outcomes, slot_attempts
    )
    terminal_transparency = sync.anthropic_metric_transparency(
        terminal_reporting_rows,
        task_ids=task_ids,
        k=K,
        complete=len(terminals) == total_slots,
    )
    primary_transparency = sync.anthropic_metric_transparency(
        primary_reporting_rows,
        task_ids=task_ids,
        k=K,
        complete=len(primary_outcomes) == total_slots,
    )
    primary_summary: dict[str, Any] | None = None
    if len(primary_outcomes) == total_slots:
        primary_by_slot = {
            (str(row["task_id"]), int(row["sample_index"])): row
            for row in primary_outcomes
        }
        expected_primary_keys = {
            (str(plan["task_id"]), sample_index)
            for plan in plans
            for sample_index in range(K)
        }
        if set(primary_by_slot) != expected_primary_keys:
            raise audited.RunFailure("primary 8192 outcome slot set is incomplete")
        primary_task_results: list[dict[str, Any]] = []
        for plan in plans:
            task_id = str(plan["task_id"])
            candidates = [
                primary_by_slot[(task_id, sample_index)] for sample_index in range(K)
            ]
            primary_task_results.append(
                {
                    "task_id": task_id,
                    "compiled": any(bool(row["compiled"]) for row in candidates),
                    "passed": any(bool(row["passed"]) for row in candidates),
                    "candidate_outcomes": candidates,
                }
            )
        primary_passed = sum(bool(row["passed"]) for row in primary_task_results)
        primary_compiled = sum(bool(row["compiled"]) for row in primary_task_results)
        primary_summary = {
            "schema": SCHEMA,
            "status": "primary_fixed_cap_complete",
            "completed_at": utc_now(),
            "config_sha256": config_sha,
            "metric": "primary_fixed_cap_8192",
            "capacity_adaptive": False,
            "length_is_failure": True,
            "model": MODEL_ID,
            "effort": EFFORT,
            "thinking": {"type": "adaptive"},
            "k": K,
            "fixed_max_output_tokens": CAP_LADDER[0],
            "tasks": len(plans),
            "logical_slots": total_slots,
            "pass_at_2_fixed_8192": {
                "successes": primary_passed,
                "total": len(plans),
                "rate": primary_passed / len(plans),
                "wilson_95": wilson_interval(primary_passed, len(plans)),
            },
            "compile_at_2_fixed_8192": {
                "successes": primary_compiled,
                "total": len(plans),
                "rate": primary_compiled / len(plans),
                "wilson_95": wilson_interval(primary_compiled, len(plans)),
            },
            "bad_gate_threshold_successes": BAD_GATE_MAX_SUCCESSES,
            "bad_gate_triggered": primary_passed <= BAD_GATE_MAX_SUCCESSES,
            "usage_and_list_cost_so_far": cost,
            "task_results": primary_task_results,
            "outcomes": file_record(out / "primary_8192_outcomes.jsonl"),
        }
        primary_summary.update(primary_transparency)
        atomic_write_json(out / "primary_8192_summary.json", primary_summary)
    progress = {
        "schema": SCHEMA,
        "status": "complete" if len(terminals) == total_slots else "incomplete",
        "updated_at": utc_now(),
        "config_sha256": config_sha,
        "model": MODEL_ID,
        "effort": EFFORT,
        "thinking": {"type": "adaptive"},
        "k": K,
        "tasks": len(plans),
        "total_logical_slots": total_slots,
        "terminal_logical_slots": len(terminals),
        "remaining_logical_slots": total_slots - len(terminals),
        "pending_next_batch_requests": len(pending),
        "next_batch_worst_list_usd": worst_batch_cost(pending),
        "usage_and_list_cost": cost,
        "screen_cost_cap_usd": args.screen_cost_cap_usd,
        "primary_fixed_cap_8192": (
            {
                "status": "complete",
                "summary": file_record(out / "primary_8192_summary.json"),
                "pass_at_2_fixed_8192": primary_summary["pass_at_2_fixed_8192"],
                "bad_gate_triggered": primary_summary["bad_gate_triggered"],
            }
            if primary_summary is not None
            else {
                "status": "incomplete",
                "terminal_slots": len(primary_outcomes),
                "required_slots": total_slots,
            }
        ),
    }
    progress.update(terminal_transparency)
    progress["primary_fixed_cap_8192"].update(primary_transparency)
    atomic_write_json(out / "progress.json", progress)
    if len(terminals) != total_slots:
        provenance = dict(provenance)
        provenance["status"] = "batch_screen_incomplete"
        provenance["progress_sha256"] = sha256_file(out / "progress.json")
        atomic_write_json(out / "provenance.json", provenance)
        return progress

    outcome_by_slot = {
        (str(row["task_id"]), int(row["sample_index"])): row for row in outcomes
    }
    task_results: list[dict[str, Any]] = []
    for plan in plans:
        task_id = str(plan["task_id"])
        candidates = [outcome_by_slot[(task_id, index)] for index in range(K)]
        task_results.append(
            {
                "task_id": task_id,
                "compiled": any(bool(row["compiled"]) for row in candidates),
                "passed": any(bool(row["passed"]) for row in candidates),
                "candidate_outcomes": candidates,
            }
        )
    passed = sum(bool(row["passed"]) for row in task_results)
    compiled = sum(bool(row["compiled"]) for row in task_results)
    cap_counts: dict[str, int] = {}
    for row in terminals:
        key = str(row["requested_max_tokens"])
        cap_counts[key] = cap_counts.get(key, 0) + 1
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "completed_at": utc_now(),
        "config_sha256": config_sha,
        "run_id": out.name,
        "dataset_label": args.dataset_label,
        "dataset_sha256": provenance["dataset"]["sha256"],
        "task_set_sha256": provenance["task_set_sha256"],
        "input_mode": args.input_mode,
        "pair_arm_key": args.pair_arm_key,
        "pair_manifest_sha256": provenance["artifacts"]["pair_manifest"]["sha256"],
        "acceptance_test_sequence_sha256": provenance[
            "acceptance_test_sequence_sha256"
        ],
        "provider": sync.PROVIDER,
        "transport_schema": TRANSPORT_SCHEMA,
        "requested_model": MODEL_ID,
        "resolved_models": sorted({str(row["resolved_model"]) for row in terminals}),
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
        "k": K,
        "tasks": len(plans),
        "logical_slots": total_slots,
        "cap_ladder": list(CAP_LADDER),
        "terminal_cap_counts": cap_counts,
        "capacity_adaptive_pass_at_2": {
            "successes": passed,
            "total": len(plans),
            "rate": passed / len(plans),
            "wilson_95": wilson_interval(passed, len(plans)),
        },
        "capacity_adaptive_compile_at_2": {
            "successes": compiled,
            "total": len(plans),
            "rate": compiled / len(plans),
            "wilson_95": wilson_interval(compiled, len(plans)),
        },
        "usage_and_list_cost": cost,
        "screen_cost_cap_usd": args.screen_cost_cap_usd,
        "evaluator": dict(evaluator_record),
        "task_results": task_results,
        "artifacts": {
            name: file_record(out / name)
            for name in (
                "tasks.jsonl",
                "prompts.jsonl",
                "batch_events.jsonl",
                "batch_slot_attempts.jsonl",
                "anthropic_input_token_counts.jsonl",
                "anthropic_input_token_audit.json",
                "primary_8192_outcomes.jsonl",
                "primary_8192_summary.json",
                "terminal_slots.jsonl",
                "outcomes.jsonl",
                "progress.json",
            )
        },
    }
    summary.update(terminal_transparency)
    if summary["resolved_models"] != [MODEL_ID]:
        raise audited.RunFailure(
            f"resolved batch models are {summary['resolved_models']!r}"
        )
    atomic_write_json(out / "summary.json", summary)
    provenance = dict(provenance)
    provenance["status"] = "complete"
    provenance["completed_at"] = summary["completed_at"]
    provenance["summary_sha256"] = sha256_file(out / "summary.json")
    atomic_write_json(out / "provenance.json", provenance)
    atomic_write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "created_at": utc_now(),
            "files": {
                name: file_record(out / name)
                for name in (
                    "provenance.json",
                    "tasks.jsonl",
                    "prompts.jsonl",
                    "batch_events.jsonl",
                    "batch_slot_attempts.jsonl",
                    "anthropic_input_token_counts.jsonl",
                    "anthropic_input_token_audit.json",
                    "primary_8192_outcomes.jsonl",
                    "primary_8192_summary.json",
                    "terminal_slots.jsonl",
                    "outcomes.jsonl",
                    "progress.json",
                    "summary.json",
                )
            },
        },
    )
    return summary


def _dispatch_action(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    prompt_map: dict[str, dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if args.action == "submit":
        return _submit(args, out=out, plans=plans, config_sha=config_sha)
    if args.action == "status":
        return _status(args, out=out, config_sha=config_sha)
    if args.action == "harvest":
        return _harvest(
            args,
            out=out,
            plans=plans,
            prompt_map=prompt_map,
            config_sha=config_sha,
            provenance=provenance,
        )
    if args.action == "auto":
        active = _active_submission(_batch_events(out, config_sha))
        if active is None:
            return _submit(args, out=out, plans=plans, config_sha=config_sha)
        status = _status(args, out=out, config_sha=config_sha)
        if status["status"] == "ended":
            return _harvest(
                args,
                out=out,
                plans=plans,
                prompt_map=prompt_map,
                config_sha=config_sha,
                provenance=provenance,
            )
        return status
    raise audited.RunFailure(f"unsupported action: {args.action}")


def main(argv: Sequence[str] | None = None) -> int:
    install_hooks()
    args = parse_args(argv)
    out = audited.choose_output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    try:
        audited.enforce_output_state_policy(args, out)
    except Exception as exc:
        print(
            f"ANTHROPIC_BATCH_FAILED_CLOSED error={type(exc).__name__}: "
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
            initial_specs = pending_request_specs(plans, [], [])
            initial_worst = worst_batch_cost(initial_specs)
            print(
                f"BATCH_PREFLIGHT_OK model={MODEL_ID} effort={EFFORT} "
                f"tasks={len(plans)} K={K} initial_requests={len(initial_specs)} "
                f"initial_worst_list_usd={initial_worst:.6f} "
                f"arm_cost_cap_usd={args.screen_cost_cap_usd:.6f} out={out}",
                flush=True,
            )
            if initial_worst > args.screen_cost_cap_usd + 1e-12:
                raise audited.RunFailure(
                    "initial batch exceeds configured per-arm cost cap"
                )
            if args.action == "preflight":
                provenance["status"] = "preflight_only_complete"
                provenance["completed_at"] = utc_now()
                atomic_write_json(out / "provenance.json", provenance)
                return 0
            result = _dispatch_action(
                args,
                out=out,
                plans=plans,
                prompt_map=prompt_map,
                config_sha=config_sha,
                provenance=provenance,
            )
        except Exception as exc:
            failure = {
                "schema": SCHEMA,
                "status": "failed_closed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(out / "failure.json", failure)
            print(
                f"ANTHROPIC_BATCH_FAILED_CLOSED error={type(exc).__name__}: "
                f"{exc} out={out}",
                file=sys.stderr,
                flush=True,
            )
            return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
