#!/usr/bin/env python3
"""Outcome-blind, dynamic length repair for the Qwen v5 primary estimator."""

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
import qwen37_primary_alias_status_v5 as primary_status


PROVENANCE_SCHEMA = "qwen37-length-repair-provenance-v5"
ATTEMPT_SCHEMA = "qwen37-length-repair-attempt-v5"
OUTCOME_SCHEMA = "qwen37-length-repair-outcome-v5"
SUMMARY_SCHEMA = "qwen37-length-repair-summary-v5"
REQUEST_CAP = 24_576
PROVIDER_TOLERANCE = 10
VALIDATION_CAP = REQUEST_CAP + PROVIDER_TOLERANCE
THINKING_BUDGET = 8_192
MAX_TRANSPORT_ATTEMPTS = 6
EXPECTED_EVALUATOR_SHA256 = primary_status.EXPECTED_EVALUATOR_SHA256
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


def response_to_dict(response: Any) -> dict[str, Any]:
    return runner.response_to_dict(response)


def classify_repair_response(
    response: Any,
    *,
    expected_model: str,
) -> core.TerminalProviderResponse:
    raw = response_to_dict(response)
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
            "repair response reasoning_content is empty or missing"
        )
    terminal = core.classify_terminal_provider_response(
        response,
        expected_model=expected_model,
        max_prompt_tokens=12_000,
        requested_max_tokens=VALIDATION_CAP,
    )
    normalized_usage = dict(terminal.usage)
    normalized_usage["reasoning_tokens"] = reasoning
    normalized_usage["answer_tokens"] = completion - reasoning
    return dataclasses.replace(terminal, usage=normalized_usage)


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


def source_slot_key(
    *,
    shard_key: str,
    arm: str,
    task_id: str,
    local_sample_index: int,
    global_sample_index: int,
    original_attempt_id: str,
    original_response_id: str,
    source_config_sha256: str,
    source_slot_policy_sha256: str,
    prompt_sha256: str,
    source_terminal_row_sha256: str,
) -> str:
    return runner.stable_sha256(
        {
            "meta_contract_sha256": (
                primary_status.EXPECTED_META_CONTRACT_SHA256
            ),
            "shard_key": shard_key,
            "arm": arm,
            "task_id": task_id,
            "local_sample_index": local_sample_index,
            "global_sample_index": global_sample_index,
            "original_attempt_id": original_attempt_id,
            "original_response_id": original_response_id,
            "source_config_sha256": source_config_sha256,
            "source_slot_policy_sha256": source_slot_policy_sha256,
            "prompt_sha256": prompt_sha256,
            "source_terminal_row_sha256": source_terminal_row_sha256,
        }
    )


def load_existing_repairs(
    attempts_path: Path,
    outcomes_path: Path,
    *,
    config_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    terminal: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    response_ids: set[str] = set()
    for row in primary_status.read_jsonl(attempts_path):
        if (
            row.get("schema") != ATTEMPT_SCHEMA
            or row.get("config_sha256") != config_sha256
        ):
            raise RepairError("repair attempt journal has a foreign contract")
        key = str(row.get("source_slot_key") or "")
        attempt_index = row.get("attempt_index")
        if (
            not key
            or isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise RepairError("repair attempt identity is malformed")
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["attempt_index"]))
        if [int(row["attempt_index"]) for row in ordered] != list(
            range(len(ordered))
        ):
            raise RepairError("repair attempt indices are not contiguous")
        if len(ordered) > MAX_TRANSPORT_ATTEMPTS:
            raise RepairError("repair transport-attempt cap exceeded")
        saw_terminal = False
        for row in ordered:
            if saw_terminal:
                raise RepairError("repair journal has a post-terminal attempt")
            response_received = row.get("response_received")
            if type(response_received) is not bool:
                raise RepairError("repair response flag is malformed")
            if response_received:
                raw = row.get("response")
                if not isinstance(raw, Mapping):
                    raise RepairError("repair terminal lacks raw response")
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
                            f"repair terminal field {field!r} was tampered"
                        )
                if classified.response_id in response_ids:
                    raise RepairError("duplicate repair response ID")
                response_ids.add(classified.response_id)
                if classified.finish_reason != "stop":
                    raise RepairError(
                        "doubled-cap repair returned non-stop finish_reason"
                    )
                terminal[key] = row
                saw_terminal = True
            else:
                if (
                    row.get("retryable_transport") is not True
                    or row.get("response") is not None
                    or row.get("usage") is not None
                ):
                    raise RepairError(
                        "repair response-less attempt is not safely retryable"
                    )
    outcomes: dict[str, dict[str, Any]] = {}
    for row in primary_status.read_jsonl(outcomes_path):
        if (
            row.get("schema") != OUTCOME_SCHEMA
            or row.get("config_sha256") != config_sha256
        ):
            raise RepairError("repair outcome journal has a foreign contract")
        key = str(row.get("source_slot_key") or "")
        if not key or key in outcomes:
            raise RepairError("repair outcome identity is malformed/duplicate")
        terminal_row = terminal.get(key)
        if (
            terminal_row is None
            or row.get("repair_attempt_id")
            != terminal_row.get("repair_attempt_id")
            or row.get("response_id") != terminal_row.get("response_id")
            or row.get("code_sha256") != terminal_row.get("code_sha256")
        ):
            raise RepairError("repair outcome is not terminal-backed")
        outcomes[key] = row
    return terminal, outcomes


def load_source_terminals_outcome_blind(
    path: Path,
    *,
    config_sha256: str,
    prompt_map: Mapping[str, Mapping[str, Any]],
    requested_model: str,
    local_k: int,
    slot_policy_sha256: str,
    response_ids: set[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate source attempts while tolerating unresolved response-less slots.

    A source runner stops on a provider-side 403 and records that response-less
    attempt as non-retryable.  Such a row is neither a completion nor a length
    trigger, so the provisional length overlay may ignore it while a separate
    sealed capacity-fallback overlay resolves the missing source slot.  Returned
    responses remain fully reclassified and tamper checked here.  No outcome
    journal is opened.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in primary_status.read_jsonl(path):
        if row.get("schema") != runner.RUN_SCHEMA_VERSION:
            raise RepairError("source attempt journal uses a foreign schema")
        if row.get("config_sha256") != config_sha256:
            raise RepairError("source attempt journal has a foreign config")
        if row.get("slot_policy_sha256") != slot_policy_sha256:
            raise RepairError("source attempt journal has a foreign slot policy")
        task_id = str(row.get("task_id") or "")
        sample_index = row.get("sample_index")
        attempt_index = row.get("attempt_index")
        if (
            task_id not in prompt_map
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
            or sample_index >= local_k
            or isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise RepairError("source attempt identity is malformed")
        if row.get("prompt_sha256") != prompt_map[task_id].get("prompt_sha256"):
            raise RepairError("source attempt prompt fingerprint mismatch")
        if row.get("requested_max_tokens") != 12_288:
            raise RepairError("source attempt completion cap mismatch")
        charge = row.get("budget_charge_tokens")
        if (
            isinstance(charge, bool)
            or not isinstance(charge, int)
            or charge < 0
        ):
            raise RepairError("source attempt budget charge is malformed")
        grouped.setdefault((task_id, sample_index), []).append(row)

    terminal: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["attempt_index"]))
        if [int(row["attempt_index"]) for row in ordered] != list(
            range(len(ordered))
        ):
            raise RepairError("source attempt indices are not contiguous")
        if len(ordered) > MAX_TRANSPORT_ATTEMPTS:
            raise RepairError("source transport-attempt cap exceeded")
        saw_terminal = False
        for position, row in enumerate(ordered):
            response_received = row.get("response_received")
            slot_terminal = row.get("slot_terminal")
            if type(response_received) is not bool or type(slot_terminal) is not bool:
                raise RepairError("source attempt lacks exact response flags")
            if saw_terminal:
                raise RepairError("source journal has a post-terminal attempt")
            if response_received:
                if slot_terminal is not True:
                    raise RepairError("source provider response is not terminal")
                raw = row.get("response")
                if not isinstance(raw, Mapping):
                    raise RepairError("source terminal lacks a raw response")
                try:
                    classified = runner.classify_terminal_provider_response(
                        dict(raw),
                        expected_model=requested_model,
                        max_prompt_tokens=12_000,
                        requested_max_tokens=12_288,
                    )
                except core.ResponseContractError as exc:
                    raise RepairError(
                        f"source terminal violates response contract: {exc}"
                    ) from exc
                expected = {
                    "response_id": classified.response_id,
                    "resolved_model": classified.response_model,
                    "response_created": classified.response_created,
                    "finish_reason": classified.finish_reason,
                    "candidate_valid": classified.candidate_valid,
                    "terminal_reason": classified.terminal_reason,
                    "content": classified.content,
                    "reasoning_content": classified.reasoning_content,
                    "code": classified.code,
                    "code_sha256": classified.code_sha256,
                    "usage": classified.usage,
                }
                for field, value in expected.items():
                    if row.get(field) != value:
                        raise RepairError(
                            f"source terminal field {field!r} was tampered"
                        )
                if classified.response_id in response_ids:
                    raise RepairError("duplicate source response ID")
                response_ids.add(classified.response_id)
                if (
                    row.get("transport_retry") is not False
                    or row.get("transport_error") is not None
                    or row.get("fatal_response_contract") is not False
                    or row.get("budget_charge_tokens")
                    != classified.usage["total_tokens"]
                ):
                    raise RepairError("source terminal transport fields are malformed")
                terminal[key] = row
                saw_terminal = True
            else:
                retryable = row.get("retryable_transport")
                if (
                    slot_terminal is not False
                    or row.get("candidate_valid") is not None
                    or row.get("terminal_reason") is not None
                    or row.get("response") is not None
                    or row.get("usage") is not None
                    or row.get("transport_retry") is not True
                    or type(retryable) is not bool
                    or row.get("fatal_response_contract") is not False
                    or not str(row.get("transport_error") or "")
                    or row.get("budget_charge_tokens") != 24_288
                ):
                    raise RepairError(
                        "source response-less transport row is malformed"
                    )
                if retryable is False and position != len(ordered) - 1:
                    raise RepairError(
                        "source has an attempt after a non-retryable failure"
                    )
    return terminal


def scan_length_sources(
    workspace: Path,
) -> tuple[list[dict[str, Any]], bool]:
    qwen_entry.install_qwen_completion_policy()
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    found: list[dict[str, Any]] = []
    all_sources_final = True
    response_ids: set[str] = set()
    for shard in primary_status.SHARDS:
        for arm in primary_status.ARMS:
            root = run_root / shard.directory_template.format(arm=arm)
            provenance = primary_status.read_json(root / "provenance.json")
            config_sha, _endpoint_hash = (
                primary_status.validate_config_and_provenance(
                    provenance,
                    shard=shard,
                    arm=arm,
                )
            )
            tasks = primary_status.read_jsonl(root / "tasks.jsonl")
            prompts = primary_status.read_jsonl(root / "prompts.jsonl")
            prompt_map = {str(row["task_id"]): row for row in prompts}
            if len(tasks) != 175 or len(prompt_map) != 175:
                raise RepairError("source task/prompt rows are incomplete")
            terminal = load_source_terminals_outcome_blind(
                root / "attempts.jsonl",
                config_sha256=config_sha,
                prompt_map=prompt_map,
                requested_model=shard.model,
                slot_policy_sha256=str(
                    provenance["config"]["slot_policy_sha256"]
                ),
                local_k=shard.local_k,
                response_ids=response_ids,
            )
            eval_path = Path(
                str(provenance["config"]["sealed_inputs"]["eval_jsonl"])
            )
            eval_rows = {
                str(row["task_id"]): row
                for row in primary_status.read_jsonl(eval_path)
            }
            if len(eval_rows) != 175:
                raise RepairError("source evaluator rows are incomplete")
            for (task_id, local_index), row in terminal.items():
                if row.get("finish_reason") != "length":
                    continue
                original_attempt_id = str(row.get("attempt_id") or "")
                original_response_id = str(row.get("response_id") or "")
                prompt_sha256 = str(row.get("prompt_sha256") or "")
                source_slot_policy_sha256 = str(
                    provenance["config"]["slot_policy_sha256"]
                )
                source_terminal_row_sha256 = runner.stable_sha256(row)
                global_sample_index = shard.global_indices[local_index]
                key = source_slot_key(
                    shard_key=shard.key,
                    arm=arm,
                    task_id=task_id,
                    local_sample_index=local_index,
                    global_sample_index=global_sample_index,
                    original_attempt_id=original_attempt_id,
                    original_response_id=original_response_id,
                    source_config_sha256=config_sha,
                    source_slot_policy_sha256=source_slot_policy_sha256,
                    prompt_sha256=prompt_sha256,
                    source_terminal_row_sha256=source_terminal_row_sha256,
                )
                found.append(
                    {
                        "source_slot_key": key,
                        "shard_key": shard.key,
                        "arm": arm,
                        "source_root": str(root),
                        "source_config_sha256": config_sha,
                        "task_id": task_id,
                        "local_sample_index": local_index,
                        "global_sample_index": global_sample_index,
                        "original_attempt_id": original_attempt_id,
                        "original_response_id": original_response_id,
                        "original_code_sha256": row.get("code_sha256"),
                        "original_finish_reason": "length",
                        "requested_model": shard.model,
                        "prompt_sha256": prompt_sha256,
                        "source_slot_policy_sha256": (
                            source_slot_policy_sha256
                        ),
                        "source_terminal_row_sha256": (
                            source_terminal_row_sha256
                        ),
                        "messages": prompt_map[task_id]["messages"],
                        "acceptance_tests": eval_rows[task_id][
                            "acceptance_tests"
                        ],
                    }
                )
            if not (root / "manifest.json").is_file():
                all_sources_final = False
    found.sort(
        key=lambda row: (
            str(row["arm"]),
            str(row["task_id"]),
            int(row["global_sample_index"]),
        )
    )
    return found, all_sources_final


def repair_one(
    source: dict[str, Any],
    *,
    client: Any,
    attempts: runner.JsonlJournal,
    outcomes: runner.JsonlJournal,
    existing_attempt_rows: list[dict[str, Any]],
    evaluator: Any,
    config_sha256: str,
) -> None:
    prior = [
        row
        for row in existing_attempt_rows
        if row.get("source_slot_key") == source["source_slot_key"]
    ]
    first_attempt = len(prior)
    for attempt_index in range(first_attempt, MAX_TRANSPORT_ATTEMPTS):
        repair_attempt_id = (
            f"repair.{str(source['task_id'])}.g{source['global_sample_index']}."
            f"a{attempt_index}.{uuid.uuid4().hex[:10]}"
        )
        base_record = {
            "schema": ATTEMPT_SCHEMA,
            "record_type": "length_repair_api_attempt",
            "config_sha256": config_sha256,
            "repair_attempt_id": repair_attempt_id,
            "attempt_index": attempt_index,
            "source_slot_key": source["source_slot_key"],
            "selection_basis": "finish_reason_length_only",
            "original_outcome_consulted_for_selection": False,
            **{
                key: source[key]
                for key in (
                    "shard_key",
                    "arm",
                    "source_root",
                    "source_config_sha256",
                    "source_slot_policy_sha256",
                    "source_terminal_row_sha256",
                    "task_id",
                    "local_sample_index",
                    "global_sample_index",
                    "original_attempt_id",
                    "original_response_id",
                    "original_code_sha256",
                    "original_finish_reason",
                    "prompt_sha256",
                )
            },
            "requested_model": source["requested_model"],
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
                model=str(source["requested_model"]),
                messages=list(source["messages"]),
            )
        except Exception as exc:
            retryable = runner.is_retryable_api_exception(exc)
            record = {
                **base_record,
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
            existing_attempt_rows.append(record)
            if not retryable:
                raise RepairError(
                    "length repair API rejected max_completion_tokens=24576: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            continue
        raw = response_to_dict(response)
        try:
            terminal = classify_repair_response(
                response,
                expected_model=str(source["requested_model"]),
            )
        except Exception as exc:
            record = {
                **base_record,
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
            existing_attempt_rows.append(record)
            raise RepairError(f"repair response contract failed: {exc}") from exc
        record = {
            **base_record,
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
        existing_attempt_rows.append(record)
        if terminal.finish_reason != "stop":
            raise RepairError(
                "doubled total completion cap still returned finish_reason="
                f"{terminal.finish_reason!r}"
            )
        if terminal.candidate_valid:
            evaluation = runner.evaluate_candidate_stably(
                evaluator,
                code=terminal.code,
                tests=str(source["acceptance_tests"]),
                task_id=str(source["task_id"]),
                sample_index=int(source["global_sample_index"]),
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
            "record_type": "length_repair_outcome",
            "config_sha256": config_sha256,
            "source_slot_key": source["source_slot_key"],
            "selection_basis": "finish_reason_length_only",
            "original_outcome_consulted_for_selection": False,
            "repair_attempt_id": repair_attempt_id,
            "response_id": terminal.response_id,
            "resolved_model": terminal.response_model,
            "finish_reason": terminal.finish_reason,
            "candidate_valid": terminal.candidate_valid,
            "terminal_reason": terminal.terminal_reason,
            "code_sha256": terminal.code_sha256,
            "task_id": source["task_id"],
            "arm": source["arm"],
            "shard_key": source["shard_key"],
            "local_sample_index": source["local_sample_index"],
            "global_sample_index": source["global_sample_index"],
            "original_attempt_id": source["original_attempt_id"],
            "original_response_id": source["original_response_id"],
            "evaluation_performed": evaluation_performed,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            **evaluation,
            "evaluated_at": runner.utc_now(),
        }
        outcomes.append(outcome)
        return
    raise RepairError("length repair exhausted response-less transport attempts")


def build_config(
    *,
    workspace: Path,
    script_path: Path,
    script_sha256: str,
    status_sha256: str,
    contract_path: Path,
    contract_sha256: str,
    endpoint: str,
) -> dict[str, Any]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "workspace": str(workspace),
        "meta_contract_sha256": primary_status.EXPECTED_META_CONTRACT_SHA256,
        "length_repair_contract": str(contract_path),
        "length_repair_contract_sha256": contract_sha256,
        "script": str(script_path),
        "script_sha256": script_sha256,
        "primary_status_sha256": status_sha256,
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
            "trigger": "finish_reason == length",
            "outcome_blind": True,
            "non_length_resampling": False,
            "maximum_terminal_repair_responses_per_source_slot": 1,
            "required_repair_finish_reason": "stop",
        },
        "api_endpoint_sha256": runner.sha256_text(endpoint.rstrip("/")),
        "api_endpoint_redacted": redact_endpoint(endpoint),
        "runtime": {
            "shared_runner_sha256": sha256_file(
                workspace / "frontier_ceiling_patch_v1" / "frontier_passk.py"
            ),
            "core_sha256": sha256_file(
                workspace / "frontier_ceiling_patch_v1" / "frontier_core.py"
            ),
            "qwen_entry_sha256": sha256_file(
                workspace
                / "frontier_ceiling_patch_v1"
                / "frontier_passk_qwen_completion.py"
            ),
            "openai_sdk_version": importlib.metadata.version("openai"),
        },
    }


def run(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    script_path = Path(__file__).resolve()
    status_path = (
        workspace
        / "frontier_ceiling_patch_v1"
        / "qwen37_primary_alias_status_v5.py"
    )
    contract_path = args.contract.expanduser().resolve()
    script_sha = sha256_file(script_path)
    status_sha = sha256_file(status_path)
    contract_sha = sha256_file(contract_path)
    if script_sha != args.expected_script_sha256:
        raise RepairError("length-repair script hash mismatch")
    if status_sha != args.expected_status_sha256:
        raise RepairError("primary status helper hash mismatch")
    if contract_sha != args.expected_contract_sha256:
        raise RepairError("length-repair contract hash mismatch")
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    endpoint = os.environ.get("QWEN_BASE_URL", "").strip().rstrip("/")
    if not api_key or not endpoint:
        raise RepairError("QWEN_API_KEY/QWEN_BASE_URL are required")
    config = build_config(
        workspace=workspace,
        script_path=script_path,
        script_sha256=script_sha,
        status_sha256=status_sha,
        contract_path=contract_path,
        contract_sha256=contract_sha,
        endpoint=endpoint,
    )
    config_sha = runner.stable_sha256(config)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / "provenance.json"
    if provenance_path.is_file():
        provenance = primary_status.read_json(provenance_path)
        if (
            provenance.get("schema") != PROVENANCE_SCHEMA
            or provenance.get("config_sha256") != config_sha
        ):
            raise RepairError("existing repair provenance is incompatible")
    else:
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "status": "watching",
            "created_at": runner.utc_now(),
            "config": config,
            "config_sha256": config_sha,
        }
    runner.atomic_write_json(provenance_path, provenance)
    copied_contract = out / contract_path.name
    if copied_contract.is_file():
        if sha256_file(copied_contract) != contract_sha:
            raise RepairError("copied repair contract hash mismatch")
    else:
        copied_contract.write_bytes(contract_path.read_bytes())
        copied_contract.chmod(0o444)

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
    attempts = runner.JsonlJournal(attempts_path)
    outcomes = runner.JsonlJournal(outcomes_path)

    while True:
        existing_terminal, existing_outcomes = load_existing_repairs(
            attempts_path,
            outcomes_path,
            config_sha256=config_sha,
        )
        sources, all_sources_final = scan_length_sources(workspace)
        source_keys = {str(source["source_slot_key"]) for source in sources}
        foreign_repairs = set(existing_terminal) - source_keys
        if foreign_repairs:
            raise RepairError("repair journal contains a non-length source slot")
        existing_attempt_rows = primary_status.read_jsonl(attempts_path)
        for source in sources:
            key = str(source["source_slot_key"])
            if key in existing_outcomes:
                continue
            if key in existing_terminal:
                raise RepairError("repair terminal is missing its outcome")
            repair_one(
                source,
                client=client,
                attempts=attempts,
                outcomes=outcomes,
                existing_attempt_rows=existing_attempt_rows,
                evaluator=evaluator,
                config_sha256=config_sha,
            )
        terminal_after, outcomes_after = load_existing_repairs(
            attempts_path,
            outcomes_path,
            config_sha256=config_sha,
        )
        pending = source_keys - set(outcomes_after)
        provenance.update(
            {
                "status": "watching",
                "last_scan_at": runner.utc_now(),
                "length_slots_observed": len(source_keys),
                "repairs_terminal": len(terminal_after),
                "repairs_evaluated": len(outcomes_after),
                "pending_repairs": len(pending),
                "evaluator": evaluator_record,
            }
        )
        runner.atomic_write_json(provenance_path, provenance)
        if all_sources_final and not pending:
            summary = {
                "schema": SUMMARY_SCHEMA,
                "status": "complete",
                "completed_at": runner.utc_now(),
                "config_sha256": config_sha,
                "length_slots_observed": len(source_keys),
                "repairs_terminal": len(terminal_after),
                "repairs_evaluated": len(outcomes_after),
                "extra_provider_calls_with_responses": len(terminal_after),
                "repair_usage": {
                    key: sum(
                        int(row["usage"][key])
                        for row in terminal_after.values()
                    )
                    for key in (
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
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--expected-status-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    for name in (
        "expected_contract_sha256",
        "expected_script_sha256",
        "expected_status_sha256",
    ):
        value = str(getattr(args, name)).strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            parser.error(f"--{name.replace('_', '-')} must be a SHA-256")
        setattr(args, name, value)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    out = args.out.expanduser().resolve()
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
            f"QWEN37_LENGTH_REPAIR_FAILED error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
