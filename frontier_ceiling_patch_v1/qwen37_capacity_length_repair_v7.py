#!/usr/bin/env python3
"""Outcome-blind 24K length repair for v6 capacity-effective terminals."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import time
import traceback
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Mapping

import frontier_core as core
import frontier_passk as runner
import frontier_passk_qwen_completion as qwen_entry
import qwen37_capacity_fallback_v6 as capacity


SCHEMA = "qwen37-capacity-length-repair-v7"
ATTEMPT_SCHEMA = "qwen37-capacity-length-repair-attempt-v7"
OUTCOME_SCHEMA = "qwen37-capacity-length-repair-outcome-v7"
PROVENANCE_SCHEMA = "qwen37-capacity-length-repair-provenance-v7"
SUMMARY_SCHEMA = "qwen37-capacity-length-repair-summary-v7"
REQUEST_CAP = 24_576
PROVIDER_TOLERANCE = 10
VALIDATION_CAP = REQUEST_CAP + PROVIDER_TOLERANCE
THINKING_BUDGET = 8_192
MAX_TRANSPORT_ATTEMPTS = 6
AUTHORIZED_MODELS = frozenset(
    {
        "qwen3.7-max-2026-05-17",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-max-preview",
        "qwen3.7-max",
    }
)
EXPECTED_EVALUATOR_SHA256 = (
    "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
)
EXPECTED_DART_SHA256 = (
    "c03ad868b5c53e31461b0fef22dc6eb6aeb56b7567efff6ca488ce9c4a6f8a6a"
)
EVALUATOR_PATH = Path(
    "/workspace/hybrid_training_patch_v2_3/scripts/evaluation/"
    "graph_compile_at_k_antigravity.py"
)
DART_PATH = Path("/usr/lib/dart/bin/dart")


class RepairError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    value = runner.sha256_file(path)
    if not value:
        raise RepairError(f"cannot hash required file: {path}")
    return value


def redact_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip().rstrip("/"))
    if parsed.scheme != "https" or not parsed.hostname:
        raise RepairError("QWEN_BASE_URL must be a valid HTTPS endpoint")
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path.rstrip("/"), "", "")
    )


def classify_repair_response(
    response: Any,
    *,
    expected_model: str,
) -> core.TerminalProviderResponse:
    raw = runner.response_to_dict(response)
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise core.ResponseContractError("repair response has no token usage")
    completion = usage.get("completion_tokens")
    if isinstance(completion, bool) or not isinstance(completion, int):
        raise core.ResponseContractError(
            "repair usage.completion_tokens is missing"
        )
    details = usage.get("completion_tokens_details")
    reasoning = (
        details.get("reasoning_tokens") if isinstance(details, dict) else None
    )
    if (
        isinstance(reasoning, bool)
        or not isinstance(reasoning, int)
        or reasoning <= 0
        or reasoning > min(completion, THINKING_BUDGET)
    ):
        raise core.ResponseContractError(
            "repair reasoning_tokens is missing or outside the sealed bound"
        )
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    message = choice.get("message") if isinstance(choice, dict) else None
    reasoning_content = (
        message.get("reasoning_content") if isinstance(message, dict) else None
    )
    if not isinstance(reasoning_content, str) or not reasoning_content.strip():
        raise core.ResponseContractError(
            "repair reasoning_content is empty or missing"
        )
    terminal = core.classify_terminal_provider_response(
        response,
        expected_model=expected_model,
        max_prompt_tokens=12_000,
        requested_max_tokens=VALIDATION_CAP,
    )
    normalized = dict(terminal.usage)
    normalized["reasoning_tokens"] = reasoning
    normalized["answer_tokens"] = completion - reasoning
    return dataclasses.replace(terminal, usage=normalized)


def make_repair_request(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
) -> Any:
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=REQUEST_CAP,
        temperature=0.8,
        top_p=0.95,
        timeout=1_800,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": THINKING_BUDGET,
        },
    )


def capacity_source_key(feed: Mapping[str, Any]) -> str:
    bound_fields = {
        key: feed.get(key)
        for key in (
            "selection_id",
            "overlay_contract_sha256",
            "parent_contract_sha256",
            "overlay_config_sha256",
            "arm",
            "pair_status",
            "originating_shard_key",
            "originating_source_directory",
            "originating_source_config_sha256",
            "originating_source_slot_policy_sha256",
            "originating_local_sample_index",
            "global_sample_index",
            "task_id",
            "prompt_sha256",
            "effective_origin",
            "effective_source_directory",
            "effective_source_config_sha256",
            "effective_source_slot_policy_sha256",
            "effective_endpoint_sha256",
            "capacity_epoch",
            "effective_attempt_id",
            "response_id",
            "requested_model",
            "resolved_model",
            "finish_reason",
            "candidate_valid",
            "terminal_reason",
            "code_sha256",
            "effective_terminal_canonical_row_sha256",
            "terminal_feed_payload_sha256",
        )
    }
    return runner.stable_sha256(
        {
            "schema": SCHEMA,
            "source_kind": "capacity_v6",
            "feed": bound_fields,
        }
    )


def validate_feed_terminal(
    row: Mapping[str, Any],
    *,
    expected_capacity_contract_sha256: str,
) -> core.TerminalProviderResponse:
    if (
        row.get("schema") != capacity.SCHEMA
        or row.get("record_type") != "capacity_effective_terminal_feed"
        or row.get("source_kind") != "capacity_v6"
        or row.get("overlay_contract_sha256")
        != expected_capacity_contract_sha256
        or row.get("selection_reads_outcomes") is not False
        or row.get("request_max_completion_tokens") != 12_288
        or row.get("thinking_budget") != THINKING_BUDGET
    ):
        raise RepairError("capacity terminal-feed contract mismatch")
    model = str(row.get("requested_model") or "")
    if not model or row.get("resolved_model") != model:
        raise RepairError("capacity terminal-feed model identity mismatch")
    raw = row.get("raw_response")
    if not isinstance(raw, Mapping):
        raise RepairError("capacity terminal feed lacks raw response")
    qwen_entry.install_qwen_completion_policy()
    terminal = qwen_entry.classify_qwen_terminal_response(
        dict(raw),
        expected_model=model,
        max_prompt_tokens=12_000,
        requested_max_tokens=12_288,
    )
    expected = {
        "response_id": terminal.response_id,
        "resolved_model": terminal.response_model,
        "finish_reason": terminal.finish_reason,
        "candidate_valid": terminal.candidate_valid,
        "terminal_reason": terminal.terminal_reason,
        "code_sha256": terminal.code_sha256,
        "validated_usage": terminal.usage,
        "reasoning_content": terminal.reasoning_content,
        "content": terminal.content,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise RepairError(
                f"capacity terminal-feed field {field!r} was tampered"
            )
    return terminal


def source_material(
    feed: Mapping[str, Any],
    *,
    run_root: Path | None = None,
) -> tuple[list[dict[str, str]], str]:
    source_root = Path(
        str(feed.get("originating_source_directory") or "")
    ).resolve()
    if run_root is None:
        run_root = Path(
            "/workspace/artifacts/frontier_ceiling_two_enrichments/runs"
        )
    run_root = run_root.resolve()
    if source_root.parent != run_root:
        raise RepairError("capacity feed originating directory is out of scope")
    provenance_path = source_root / "provenance.json"
    if not provenance_path.is_file():
        raise RepairError("originating source provenance is missing")
    provenance = capacity.read_json(provenance_path)
    config = provenance.get("config")
    if not isinstance(config, Mapping):
        raise RepairError("originating source config is missing")
    if (
        runner.stable_sha256(dict(config))
        != feed.get("originating_source_config_sha256")
        or config.get("slot_policy_sha256")
        != feed.get("originating_source_slot_policy_sha256")
    ):
        raise RepairError("originating source config binding mismatch")
    task_id = str(feed.get("task_id") or "")
    try:
        messages = capacity._load_messages(source_root, feed)
        tests = capacity._load_tests(source_root, task_id)
    except Exception as exc:
        raise RepairError(
            f"sealed originating prompt/eval material failed: {exc}"
        ) from exc
    return messages, tests


def scan_capacity_length_sources(
    feed_path: Path,
    *,
    expected_capacity_contract_sha256: str,
) -> list[dict[str, Any]]:
    feed_rows = capacity._load_terminal_feed(feed_path)
    found: list[dict[str, Any]] = []
    for row in feed_rows.values():
        terminal = validate_feed_terminal(
            row,
            expected_capacity_contract_sha256=(
                expected_capacity_contract_sha256
            ),
        )
        if terminal.finish_reason != "length":
            continue
        messages, tests = source_material(row)
        found.append(
            {
                "capacity_source_key": capacity_source_key(row),
                "feed": dict(row),
                "messages": messages,
                "acceptance_tests": tests,
            }
        )
    found.sort(
        key=lambda value: (
            str(value["feed"]["arm"]),
            str(value["feed"]["task_id"]),
            int(value["feed"]["global_sample_index"]),
        )
    )
    return found


def load_existing(
    attempts_path: Path,
    outcomes_path: Path,
    *,
    config_sha256: str,
    allow_sealed_quota_boundary: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    response_ids: set[str] = set()
    for row in capacity.read_jsonl(attempts_path):
        if (
            row.get("schema") != ATTEMPT_SCHEMA
            or row.get("config_sha256") != config_sha256
        ):
            raise RepairError("capacity-length attempt has a foreign contract")
        key = str(row.get("capacity_source_key") or "")
        index = row.get("attempt_index")
        if (
            not key
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
        ):
            raise RepairError("capacity-length attempt identity is malformed")
        grouped.setdefault(key, []).append(row)
    terminal: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["attempt_index"]))
        if [int(row["attempt_index"]) for row in ordered] != list(
            range(len(ordered))
        ):
            raise RepairError("capacity-length attempt indices are not contiguous")
        if len(ordered) > MAX_TRANSPORT_ATTEMPTS:
            raise RepairError("capacity-length transport-attempt cap exceeded")
        saw_terminal = False
        for row in ordered:
            if saw_terminal:
                raise RepairError("capacity-length journal is post-terminal")
            response_received = row.get("response_received")
            if type(response_received) is not bool:
                raise RepairError("capacity-length response flag is malformed")
            if response_received:
                raw = row.get("response")
                if not isinstance(raw, Mapping):
                    raise RepairError("capacity-length terminal lacks raw response")
                classified = classify_repair_response(
                    dict(raw),
                    expected_model=str(row.get("requested_model") or ""),
                )
                expected = {
                    "response_id": classified.response_id,
                    "resolved_model": classified.response_model,
                    "finish_reason": classified.finish_reason,
                    "candidate_valid": classified.candidate_valid,
                    "terminal_reason": classified.terminal_reason,
                    "code_sha256": classified.code_sha256,
                    "usage": classified.usage,
                    "reasoning_content": classified.reasoning_content,
                }
                for field, value in expected.items():
                    if row.get(field) != value:
                        raise RepairError(
                            f"capacity-length field {field!r} was tampered"
                        )
                if classified.response_id in response_ids:
                    raise RepairError("duplicate capacity-length response ID")
                response_ids.add(classified.response_id)
                if classified.finish_reason != "stop":
                    raise RepairError(
                        "capacity-length repair did not finish with stop"
                    )
                terminal[key] = row
                saw_terminal = True
            else:
                retryable = row.get("retryable_transport")
                transport_error = str(row.get("transport_error") or "")
                accepted_quota_boundary = (
                    allow_sealed_quota_boundary
                    and retryable is False
                    and capacity.exact_quota_403(transport_error)
                )
                if (
                    (retryable is not True and not accepted_quota_boundary)
                    or row.get("response") is not None
                    or row.get("usage") is not None
                    or not transport_error
                ):
                    raise RepairError(
                        "capacity-length response-less attempt is neither "
                        "retryable nor an exact sealed quota boundary"
                    )
    outcomes: dict[str, dict[str, Any]] = {}
    for row in capacity.read_jsonl(outcomes_path):
        if (
            row.get("schema") != OUTCOME_SCHEMA
            or row.get("config_sha256") != config_sha256
        ):
            raise RepairError("capacity-length outcome has a foreign contract")
        key = str(row.get("capacity_source_key") or "")
        terminal_row = terminal.get(key)
        if (
            not key
            or key in outcomes
            or terminal_row is None
            or row.get("repair_attempt_id")
            != terminal_row.get("repair_attempt_id")
            or row.get("response_id") != terminal_row.get("response_id")
            or row.get("code_sha256") != terminal_row.get("code_sha256")
        ):
            raise RepairError("capacity-length outcome is not terminal-backed")
        outcomes[key] = row
    return terminal, outcomes


def load_prior_repairs(
    roots: list[Path],
    *,
    expected_script_sha256: str,
    expected_contract_sha256: str,
    expected_capacity_script_sha256: str,
    expected_capacity_contract_sha256: str,
    capacity_out: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    all_terminal: dict[str, dict[str, Any]] = {}
    all_outcomes: dict[str, dict[str, Any]] = {}
    response_ids: set[str] = set()
    for root in roots:
        root = root.resolve()
        provenance = capacity.read_json(root / "provenance.json")
        if provenance.get("schema") != PROVENANCE_SCHEMA:
            raise RepairError("prior repair provenance uses a foreign schema")
        prior_config = provenance.get("config")
        if not isinstance(prior_config, Mapping) or (
            prior_config.get("script_sha256") != expected_script_sha256
            or prior_config.get("contract_sha256")
            != expected_contract_sha256
            or prior_config.get("capacity_script_sha256")
            != expected_capacity_script_sha256
            or prior_config.get("capacity_contract_sha256")
            != expected_capacity_contract_sha256
            or prior_config.get("capacity_out") != str(capacity_out.resolve())
        ):
            raise RepairError("prior repair runtime/source binding mismatch")
        config_sha = str(provenance.get("config_sha256") or "")
        terminal, outcomes = load_existing(
            root / "repair_attempts.jsonl",
            root / "repair_outcomes.jsonl",
            config_sha256=config_sha,
            allow_sealed_quota_boundary=True,
        )
        for key, row in terminal.items():
            response_id = str(row.get("response_id") or "")
            if key in all_terminal or not response_id or response_id in response_ids:
                raise RepairError("duplicate prior capacity-length repair")
            if key not in outcomes:
                raise RepairError(
                    "prior repair has a terminal response without outcome"
                )
            all_terminal[key] = row
            all_outcomes[key] = outcomes[key]
            response_ids.add(response_id)
    return all_terminal, all_outcomes


def repair_one(
    source: Mapping[str, Any],
    *,
    client: Any,
    attempts: runner.JsonlJournal,
    outcomes: runner.JsonlJournal,
    existing_attempts: list[dict[str, Any]],
    evaluator: Any,
    config_sha256: str,
) -> None:
    feed = source["feed"]
    key = str(source["capacity_source_key"])
    prior = [
        row for row in existing_attempts if row.get("capacity_source_key") == key
    ]
    for attempt_index in range(len(prior), MAX_TRANSPORT_ATTEMPTS):
        repair_attempt_id = (
            f"capacity-repair.{feed['task_id']}."
            f"g{feed['global_sample_index']}.a{attempt_index}."
            f"{uuid.uuid4().hex[:10]}"
        )
        source_projection = {
            field: feed.get(field)
            for field in (
                "selection_id",
                "terminal_feed_payload_sha256",
                "overlay_contract_sha256",
                "parent_contract_sha256",
                "overlay_config_sha256",
                "arm",
                "pair_status",
                "originating_shard_key",
                "originating_source_directory",
                "originating_source_config_sha256",
                "originating_source_slot_policy_sha256",
                "originating_local_sample_index",
                "global_sample_index",
                "task_id",
                "prompt_sha256",
                "effective_origin",
                "effective_source_directory",
                "effective_source_config_sha256",
                "effective_source_slot_policy_sha256",
                "effective_endpoint_sha256",
                "capacity_epoch",
                "effective_attempt_id",
                "response_id",
                "effective_terminal_canonical_row_sha256",
                "finish_reason",
            )
        }
        base = {
            "schema": ATTEMPT_SCHEMA,
            "record_type": "capacity_length_repair_api_attempt",
            "config_sha256": config_sha256,
            "repair_attempt_id": repair_attempt_id,
            "attempt_index": attempt_index,
            "capacity_source_key": key,
            "source_kind": "capacity_v6",
            "selection_basis": "effective_feed_finish_reason_length_only",
            "capacity_outcome_consulted_for_selection": False,
            "source": source_projection,
            "requested_model": feed["requested_model"],
            "request_cap_parameter": "max_completion_tokens",
            "requested_max_completion_tokens": REQUEST_CAP,
            "provider_completion_tolerance": PROVIDER_TOLERANCE,
            "completion_usage_validation_cap": VALIDATION_CAP,
            "thinking_budget": THINKING_BUDGET,
            "started_at": runner.utc_now(),
        }
        try:
            response = make_repair_request(
                client,
                model=str(feed["requested_model"]),
                messages=list(source["messages"]),
            )
        except Exception as exc:
            retryable = runner.is_retryable_api_exception(exc)
            record = {
                **base,
                "finished_at": runner.utc_now(),
                "response_received": False,
                "slot_terminal": False,
                "retryable_transport": retryable,
                "transport_error": (
                    f"api_exception:{type(exc).__name__}:{str(exc)[:1000]}"
                ),
                "usage": None,
                "response": None,
            }
            attempts.append(record)
            existing_attempts.append(record)
            if not retryable:
                raise RepairError(
                    "capacity-length repair API failed non-retryably: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            continue
        raw = runner.response_to_dict(response)
        try:
            terminal = classify_repair_response(
                response,
                expected_model=str(feed["requested_model"]),
            )
        except Exception as exc:
            record = {
                **base,
                "finished_at": runner.utc_now(),
                "response_received": True,
                "slot_terminal": True,
                "repair_accepted": False,
                "fatal_response_contract": True,
                "terminal_reason": f"fatal_response_contract:{exc}",
                "usage": raw.get("usage"),
                "response": raw,
            }
            attempts.append(record)
            existing_attempts.append(record)
            raise RepairError(
                f"capacity-length response contract failed: {exc}"
            ) from exc
        record = {
            **base,
            "finished_at": runner.utc_now(),
            "response_received": True,
            "slot_terminal": True,
            "repair_accepted": terminal.finish_reason == "stop",
            "fatal_response_contract": False,
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
            "response": terminal.raw_response,
        }
        attempts.append(record)
        existing_attempts.append(record)
        if terminal.finish_reason != "stop":
            raise RepairError(
                "capacity-length doubled cap still returned "
                f"finish_reason={terminal.finish_reason!r}"
            )
        if terminal.candidate_valid:
            evaluation = runner.evaluate_candidate_stably(
                evaluator,
                code=terminal.code,
                tests=str(source["acceptance_tests"]),
                task_id=str(feed["task_id"]),
                sample_index=int(feed["global_sample_index"]),
                stability_runs=2,
                timeout=30,
            )
            evaluation_performed = True
        else:
            evaluation = {
                "compiled": False,
                "passed": False,
                "completion_attestation_id": runner.REQUIRED_ATTESTATION_ID,
                "completion_attestation_enforced": False,
                "completion_attestation_satisfied_all_runs": False,
                "stability_runs": [],
            }
            evaluation_performed = False
        outcome = {
            "schema": OUTCOME_SCHEMA,
            "record_type": "capacity_length_repair_outcome",
            "config_sha256": config_sha256,
            "capacity_source_key": key,
            "source_selection_id": feed["selection_id"],
            "source_terminal_feed_payload_sha256": feed[
                "terminal_feed_payload_sha256"
            ],
            "selection_basis": "effective_feed_finish_reason_length_only",
            "capacity_outcome_consulted_for_selection": False,
            "repair_attempt_id": repair_attempt_id,
            "response_id": terminal.response_id,
            "resolved_model": terminal.response_model,
            "finish_reason": terminal.finish_reason,
            "candidate_valid": terminal.candidate_valid,
            "terminal_reason": terminal.terminal_reason,
            "code_sha256": terminal.code_sha256,
            "task_id": feed["task_id"],
            "arm": feed["arm"],
            "global_sample_index": feed["global_sample_index"],
            "evaluation_performed": evaluation_performed,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            **evaluation,
            "evaluated_at": runner.utc_now(),
        }
        outcomes.append(outcome)
        return
    raise RepairError(
        "capacity-length repair exhausted response-less transport attempts"
    )


def build_config(
    args: argparse.Namespace,
    *,
    script_sha256: str,
    contract_sha256: str,
    endpoint: str,
) -> dict[str, Any]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "contract": str(args.contract.resolve()),
        "contract_sha256": contract_sha256,
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_sha256,
        "capacity_script": str(Path(capacity.__file__).resolve()),
        "capacity_script_sha256": args.expected_capacity_script_sha256,
        "capacity_contract_sha256": (
            args.expected_capacity_contract_sha256
        ),
        "capacity_out": str(args.capacity_out.resolve()),
        "capacity_terminal_feed": str(
            (args.capacity_out.resolve() / "effective_terminal_feed.jsonl")
        ),
        "request_policy": {
            "request_cap_parameter": "max_completion_tokens",
            "max_tokens_absent": True,
            "requested_cap": REQUEST_CAP,
            "provider_completion_tolerance": PROVIDER_TOLERANCE,
            "validation_cap": VALIDATION_CAP,
            "thinking_budget": THINKING_BUDGET,
            "temperature": 0.8,
            "top_p": 0.95,
            "finite_budget": 0,
        },
        "selection_policy": {
            "source": "v6 outcome-free effective terminal feed",
            "trigger": "finish_reason == length",
            "outcome_blind": True,
            "non_length_resampling": False,
            "maximum_terminal_repair_responses_per_source_slot": 1,
            "required_repair_finish_reason": "stop",
        },
        "repair_endpoint_epoch": args.repair_epoch,
        "allowed_models_this_repair_epoch": sorted(args.allowed_model),
        "prior_repair_outputs": [
            str(path.resolve()) for path in args.prior_repair_out
        ],
        "endpoint_independence": {
            "source_capacity_endpoint_is_not_the_repair_endpoint": True,
            "repair_endpoint_is_bound_by_this_config": True,
            "a_new_exact_quota_epoch_uses_a_fresh_output_directory": True,
        },
        "api_endpoint_sha256": runner.sha256_text(endpoint.rstrip("/")),
        "api_endpoint_redacted": redact_endpoint(endpoint),
        "runtime": {
            "shared_runner_sha256": sha256_file(
                args.workspace.resolve()
                / "frontier_ceiling_patch_v1"
                / "frontier_passk.py"
            ),
            "core_sha256": sha256_file(
                args.workspace.resolve()
                / "frontier_ceiling_patch_v1"
                / "frontier_core.py"
            ),
            "qwen_entry_sha256": sha256_file(
                args.workspace.resolve()
                / "frontier_ceiling_patch_v1"
                / "frontier_passk_qwen_completion.py"
            ),
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "dart_sha256": EXPECTED_DART_SHA256,
            "openai_sdk_version": importlib.metadata.version("openai"),
        },
    }


def run(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    contract_sha = sha256_file(args.contract.resolve())
    capacity_script_sha = sha256_file(Path(capacity.__file__).resolve())
    capacity_contract = (
        args.workspace.resolve()
        / "frontier_ceiling_patch_v1"
        / "qwen37_capacity_fallback_contract_v6.json"
    )
    if (
        script_sha != args.expected_script_sha256
        or contract_sha != args.expected_contract_sha256
        or capacity_script_sha != args.expected_capacity_script_sha256
        or sha256_file(capacity_contract)
        != args.expected_capacity_contract_sha256
    ):
        raise RepairError("v7 or capacity-v6 runtime hash mismatch")
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    endpoint = os.environ.get("QWEN_BASE_URL", "").strip().rstrip("/")
    if not api_key or not endpoint:
        raise RepairError("QWEN_API_KEY/QWEN_BASE_URL are required")
    config = build_config(
        args,
        script_sha256=script_sha,
        contract_sha256=contract_sha,
        endpoint=endpoint,
    )
    config_sha = runner.stable_sha256(config)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / "provenance.json"
    if provenance_path.is_file():
        provenance = capacity.read_json(provenance_path)
        if (
            provenance.get("schema") != PROVENANCE_SCHEMA
            or provenance.get("config_sha256") != config_sha
        ):
            raise RepairError("existing v7 provenance is incompatible")
    else:
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "status": "watching",
            "created_at": runner.utc_now(),
            "config": config,
            "config_sha256": config_sha,
        }
    runner.atomic_write_json(provenance_path, provenance)
    copied_contract = out / args.contract.name
    if copied_contract.is_file():
        if sha256_file(copied_contract) != contract_sha:
            raise RepairError("copied v7 contract hash mismatch")
    else:
        copied_contract.write_bytes(args.contract.resolve().read_bytes())
        copied_contract.chmod(0o444)

    feed_path = args.capacity_out.resolve() / "effective_terminal_feed.jsonl"
    all_sources = scan_capacity_length_sources(
        feed_path,
        expected_capacity_contract_sha256=(
            args.expected_capacity_contract_sha256
        ),
    )
    prior_terminal, prior_outcomes = load_prior_repairs(
        list(args.prior_repair_out),
        expected_script_sha256=script_sha,
        expected_contract_sha256=contract_sha,
        expected_capacity_script_sha256=(
            args.expected_capacity_script_sha256
        ),
        expected_capacity_contract_sha256=(
            args.expected_capacity_contract_sha256
        ),
        capacity_out=args.capacity_out,
    )
    if args.preflight_only:
        runner.atomic_write_json(
            out / "preflight.json",
            {
                "schema": PROVENANCE_SCHEMA,
                "status": "preflight_complete",
                "checked_at": runner.utc_now(),
                "config_sha256": config_sha,
                "capacity_length_slots_observed": len(all_sources),
                "eligible_slots_this_repair_epoch": sum(
                    source["feed"]["requested_model"]
                    in set(args.allowed_model)
                    for source in all_sources
                ),
                "prior_epoch_repairs": len(prior_outcomes),
                "provider_calls": 0,
            },
        )
        return 0

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RepairError("the openai package is required") from exc
    client = OpenAI(api_key=api_key, base_url=endpoint, max_retries=0)
    evaluator_module, evaluator_record = runner.import_evaluator(
        EVALUATOR_PATH,
        EXPECTED_EVALUATOR_SHA256,
        dart_binary=DART_PATH,
        expected_dart_hash=EXPECTED_DART_SHA256,
        validate_dart=True,
    )
    evaluator = evaluator_module.evaluate_dart_jit_tests_detail
    attempts_path = out / "repair_attempts.jsonl"
    outcomes_path = out / "repair_outcomes.jsonl"
    attempts_path.touch(exist_ok=True)
    outcomes_path.touch(exist_ok=True)
    attempts = runner.JsonlJournal(attempts_path)
    outcomes = runner.JsonlJournal(outcomes_path)

    while True:
        all_sources = scan_capacity_length_sources(
            feed_path,
            expected_capacity_contract_sha256=(
                args.expected_capacity_contract_sha256
            ),
        )
        sources = [
            source
            for source in all_sources
            if source["feed"]["requested_model"] in set(args.allowed_model)
        ]
        terminal, existing_outcomes = load_existing(
            attempts_path,
            outcomes_path,
            config_sha256=config_sha,
        )
        all_source_keys = {
            str(source["capacity_source_key"]) for source in all_sources
        }
        source_keys = {
            str(source["capacity_source_key"]) for source in sources
        }
        if (
            not set(terminal).issubset(source_keys)
            or not set(prior_terminal).issubset(all_source_keys)
        ):
            raise RepairError("v7 journal contains a non-length capacity slot")
        if set(terminal).intersection(prior_terminal):
            raise RepairError("v7 repair duplicates a prior endpoint epoch")
        existing_attempts = capacity.read_jsonl(attempts_path)
        for source in sources:
            key = str(source["capacity_source_key"])
            if key in existing_outcomes or key in prior_outcomes:
                continue
            if key in terminal:
                raise RepairError("v7 terminal is missing its outcome")
            repair_one(
                source,
                client=client,
                attempts=attempts,
                outcomes=outcomes,
                existing_attempts=existing_attempts,
                evaluator=evaluator,
                config_sha256=config_sha,
            )
        terminal_after, outcomes_after = load_existing(
            attempts_path,
            outcomes_path,
            config_sha256=config_sha,
        )
        repaired_keys = set(outcomes_after).union(prior_outcomes)
        pending = source_keys - repaired_keys
        capacity_status_path = args.capacity_out.resolve() / "status.json"
        capacity_complete = False
        if capacity_status_path.is_file():
            capacity_status = capacity.read_json(capacity_status_path)
            capacity_complete = capacity_status.get("status") == "complete"
        provenance.update(
            {
                "status": "watching",
                "last_scan_at": runner.utc_now(),
                "capacity_length_slots_observed": len(source_keys),
                "all_capacity_length_slots_observed": len(all_source_keys),
                "repairs_terminal": len(terminal_after),
                "repairs_evaluated": len(outcomes_after),
                "prior_epoch_repairs": len(prior_outcomes),
                "effective_repairs_across_epochs": len(repaired_keys),
                "pending_repairs": len(pending),
                "capacity_overlay_complete": capacity_complete,
                "evaluator": evaluator_record,
            }
        )
        runner.atomic_write_json(provenance_path, provenance)
        if capacity_complete and not pending:
            summary = {
                "schema": SUMMARY_SCHEMA,
                "status": "complete",
                "completed_at": runner.utc_now(),
                "config_sha256": config_sha,
                "capacity_length_slots_observed": len(source_keys),
                "all_capacity_length_slots_observed": len(all_source_keys),
                "repairs_terminal": len(terminal_after),
                "repairs_evaluated": len(outcomes_after),
                "prior_epoch_repairs": len(prior_outcomes),
                "effective_repairs_across_epochs": len(repaired_keys),
                "extra_provider_calls_with_responses": len(terminal_after),
                "repair_usage": {
                    field: sum(
                        int(row["usage"][field])
                        for row in terminal_after.values()
                    )
                    for field in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "reasoning_tokens",
                        "answer_tokens",
                    )
                },
                "artifacts": {
                    "repair_attempts_sha256": sha256_file(attempts_path),
                    "repair_outcomes_sha256": sha256_file(outcomes_path),
                },
            }
            runner.atomic_write_json(out / "summary.json", summary)
            provenance["status"] = "complete"
            provenance["completed_at"] = summary["completed_at"]
            provenance["summary_sha256"] = sha256_file(out / "summary.json")
            runner.atomic_write_json(provenance_path, provenance)
            return 0
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--capacity-out", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--expected-capacity-contract-sha256", required=True)
    parser.add_argument("--expected-capacity-script-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repair-epoch", required=True)
    parser.add_argument(
        "--allowed-model",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--prior-repair-out",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "expected_contract_sha256",
        "expected_script_sha256",
        "expected_capacity_contract_sha256",
        "expected_capacity_script_sha256",
    ):
        value = str(getattr(args, name)).strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            parser.error(f"--{name.replace('_', '-')} must be a SHA-256")
        setattr(args, name, value)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not args.repair_epoch.strip():
        parser.error("--repair-epoch must be nonempty")
    if (
        not args.allowed_model
        or len(args.allowed_model) != len(set(args.allowed_model))
        or not set(args.allowed_model).issubset(AUTHORIZED_MODELS)
    ):
        parser.error(
            "--allowed-model must be a unique nonempty subset of the five "
            "authorized Qwen3.7 aliases"
        )
    resolved_priors = [path.resolve() for path in args.prior_repair_out]
    if len(resolved_priors) != len(set(resolved_priors)):
        parser.error("--prior-repair-out contains duplicates")
    if args.out.resolve() in set(resolved_priors):
        parser.error("--out cannot also be a prior repair output")
    return args


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        with runner.RunLock(out / ".run.lock"):
            return run(args)
    except Exception as exc:
        runner.atomic_write_json(
            out / "failure.json",
            {
                "schema": PROVENANCE_SCHEMA,
                "status": "failed_closed",
                "failed_at": runner.utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"QWEN37_CAPACITY_LENGTH_REPAIR_FAILED "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
