#!/usr/bin/env python3
"""Audited outcome-blind Qwen alias-capacity fallback overlay.

This program never edits a primary or diagnostic run directory.  It selects
only global slots absent from the four failed 05-17 primary shards after each
shard recorded an exact response-less quota HTTP 403.  A selected slot is
filled by either adopting its pre-existing clean diagnostic response or by a
fresh request to the sealed alias pool.  Pass/compile outcomes are not opened
until after an effective terminal has already been selected.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from openai import OpenAI

import frontier_passk as runner
import frontier_passk_qwen_completion as qwen


SCHEMA = "qwen37-capacity-fallback-v6"
CONTRACT_NAME = "qwen37_capacity_fallback_contract_v6.json"
EXPECTED_CONTRACT_SHA256 = (
    "cea8acaa785ddc2685a5da8b4426dce41837a25af2e7dd9639dd70f632d59631"
)
EXPECTED_PARENT_CONTRACT_SHA256 = (
    "6218118c8e9e7b67079df2b848626ea8a71deaf128353bbc8757d9553a6cdbae"
)
EXPECTED_RUNNER_SHA256 = (
    "8d3e3ad160d9ed389a9e212dacb76556ab7af59f1559418d45d9802402d9dead"
)
EXPECTED_CORE_SHA256 = (
    "f502e958a6fa3fb564d17327c2c4c77bc9cf4f5182546235970b1a4498a60258"
)
EXPECTED_QWEN_ENTRY_SHA256 = (
    "5055eabac3898d529beb6209b3792256378d509239265cb44eaa2cf7f46b5e15"
)
EXPECTED_EVALUATOR_SHA256 = (
    "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
)
EXPECTED_DART_SHA256 = (
    "c03ad868b5c53e31461b0fef22dc6eb6aeb56b7567efff6ca488ce9c4a6f8a6a"
)
EXPECTED_TASKS = 175
SOURCE_MODEL = "qwen3.7-max-2026-05-17"
ALLOWED_ALIASES = (
    "qwen3.7-max-2026-05-20",
    "qwen3.7-max-2026-06-08",
    "qwen3.7-max-preview",
    "qwen3.7-max",
    SOURCE_MODEL,
)
VALIDATED_MODELS = frozenset((*ALLOWED_ALIASES, SOURCE_MODEL))
CAP = 12_288
VALIDATION_CAP = 12_298
THINKING_BUDGET = 8_192
PROMPT_CAP = 12_000
EVALUATOR_RELATIVE = (
    "hybrid_training_patch_v2_3/scripts/evaluation/"
    "graph_compile_at_k_antigravity.py"
)
DART_PATH = Path("/usr/lib/dart/bin/dart")


class AuditError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class SourceShard:
    key: str
    source_model: str
    local_k: int
    global_indices: tuple[int, ...]
    directory_template: str
    partition: str
    service_template: str
    trigger: str


SOURCE_SHARDS = (
    SourceShard(
        "base_0517_k3",
        SOURCE_MODEL,
        3,
        (0, 1, 2),
        "qwen37_clean_v4_0517_{arm}_k3_mc12k_tol10_tb8k",
        "0520",
        "frontier-qwen37-clean-v4-0517-{arm}-k3-mc12k-tol10-tb8k.service",
        "per_shard_quota_403",
    ),
    SourceShard(
        "base_preview_k2",
        "qwen3.7-max-preview",
        2,
        (3, 4),
        "qwen37_clean_v4_preview_{arm}_k2_mc12k_tol10_tb8k",
        "0520",
        "frontier-qwen37-clean-v4-preview-{arm}-k2-mc12k-tol10-tb8k.service",
        "preview_model_guard",
    ),
    SourceShard(
        "supplement_0517_k2",
        SOURCE_MODEL,
        2,
        (5, 6),
        "qwen37_clean_v5_supplement_0517_{arm}_k2_mc12k_tol10_tb8k",
        "0608",
        (
            "frontier-qwen37-clean-v5-supplement-0517-{arm}-k2-"
            "mc12k-tol10-tb8k.service"
        ),
        "per_shard_quota_403",
    ),
    SourceShard(
        "supplement_preview_k3",
        "qwen3.7-max-preview",
        3,
        (7, 8, 9),
        "qwen37_clean_v5_supplement_preview_{arm}_k3_mc12k_tol10_tb8k",
        "0608",
        (
            "frontier-qwen37-clean-v5-supplement-preview-{arm}-k3-"
            "mc12k-tol10-tb8k.service"
        ),
        "preview_model_guard",
    ),
)

PARTITIONS = {
    "0520": {
        "indices": (0, 1, 2, 3, 4),
        "diagnostic_reuse_indices": (0, 1, 2),
        "aliases": (
            "qwen3.7-max-2026-05-20",
            "qwen3.7-max-2026-06-08",
            "qwen3.7-max-preview",
            "qwen3.7-max",
        ),
        "diagnostic_model": "qwen3.7-max-2026-05-20",
        "diagnostic_dir": (
            "qwen37_clean_v4_0520_{arm}_k3_mc12k_tol10_tb8k"
        ),
        "diagnostic_service": (
            "frontier-qwen37-clean-v4-0520-{arm}-k3-mc12k-tol10-tb8k.service"
        ),
    },
    "0608": {
        "indices": (5, 6, 7, 8, 9),
        "diagnostic_reuse_indices": (5, 6),
        "aliases": (
            "qwen3.7-max-2026-06-08",
            "qwen3.7-max-2026-05-20",
            "qwen3.7-max-preview",
            "qwen3.7-max",
        ),
        "diagnostic_model": "qwen3.7-max-2026-06-08",
        "diagnostic_dir": (
            "qwen37_clean_v4_0608_{arm}_k2_mc12k_tol10_tb8k"
        ),
        "diagnostic_service": (
            "frontier-qwen37-clean-v4-0608-{arm}-k2-mc12k-tol10-tb8k.service"
        ),
    },
}

QUOTA_CODES = ("AllocationQuota.FreeTierOnly", "insufficient_quota")
PREVIEW_GUARD_RELATIVE = (
    "artifacts/frontier_ceiling_two_enrichments/qwen37_quota_guard_v1/"
    "event_60c698b8b65c65964b24edcbac389fcc4f520a4bde83aedf00b489a757f53ce6.json"
)
PREVIEW_GUARD_SHA256 = (
    "3585b5382a8b57cf7f00fbcc137812872b0f9f9552b66c9ce7df16dc8e7974e3"
)
PREVIEW_GUARD_MAPPING_SHA256 = (
    "b8b6ef9afc0de3ab44881d63aa54d9a1561468098248647e74014012b7c4d722"
)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise AuditError(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Mapping[str, Any]) -> str:
    return runner.stable_sha256(dict(value))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def exact_quota_403(text: str) -> bool:
    return (
        "Error code: 403" in text
        and any(code in text for code in QUOTA_CODES)
        and "free quota has been exhausted" in text.lower()
    )


def data_inspection_failed(exc_or_text: Any) -> bool:
    text = str(exc_or_text)
    return "data_inspection_failed" in text and (
        "Error code: 400" in text or "status_code=400" in text
    )


def exception_status(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r"Error code:\s*(\d{3})", str(exc))
    return int(match.group(1)) if match else None


def quota_exception(exc: BaseException) -> bool:
    return exception_status(exc) == 403 and exact_quota_403(str(exc))


def ensure_undated_identity_probe(
    client: Any,
    *,
    out: Path,
    capacity_epoch: str,
    endpoint_sha256: str,
) -> dict[str, Any]:
    path = out / "identity_probes.jsonl"
    matches = [
        row
        for row in read_jsonl(path)
        if row.get("capacity_epoch") == capacity_epoch
        and row.get("requested_model") == "qwen3.7-max"
    ]
    if len(matches) > 1:
        raise AuditError("duplicate undated-alias identity probe")
    if matches:
        row = matches[0]
        if (
            row.get("endpoint_sha256") != endpoint_sha256
            or row.get("resolved_model") != "qwen3.7-max"
            or not str(row.get("response_id") or "")
        ):
            raise AuditError("undated-alias identity probe receipt mismatch")
        return row
    response = client.chat.completions.create(
        model="qwen3.7-max",
        messages=[
            {
                "role": "system",
                "content": "Identity probe. Return exactly OK.",
            },
            {"role": "user", "content": "OK"},
        ],
        max_completion_tokens=64,
        temperature=0,
        top_p=1,
        timeout=120,
        extra_body={"enable_thinking": False},
    )
    raw = runner.response_to_dict(response)
    resolved = str(raw.get("model") or "")
    response_id = str(raw.get("id") or "")
    if resolved != "qwen3.7-max" or not response_id:
        raise AuditError(
            "undated alias did not resolve exactly to qwen3.7-max: "
            f"{resolved!r}"
        )
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise AuditError("identity probe lacks usage")
    record = {
        "schema": SCHEMA,
        "record_type": "capacity_alias_identity_probe",
        "capacity_epoch": capacity_epoch,
        "endpoint_sha256": endpoint_sha256,
        "requested_model": "qwen3.7-max",
        "resolved_model": resolved,
        "response_id": response_id,
        "usage": usage,
        "content_persisted": False,
        "probed_at": runner.utc_now(),
    }
    runner.JsonlJournal(path).append(record)
    return record


def make_v6_request(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
) -> Any:
    if model not in ALLOWED_ALIASES:
        raise AuditError(f"unsealed capacity alias: {model}")
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=CAP,
        temperature=0.8,
        top_p=0.95,
        timeout=1800,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": THINKING_BUDGET,
        },
    )


def classify_v6_terminal_response(
    response: Any,
    *,
    expected_model: str,
) -> runner.TerminalProviderResponse:
    if expected_model not in VALIDATED_MODELS:
        raise runner.ResponseContractError("unsealed capacity alias")
    raw = runner.response_to_dict(response)
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise runner.ResponseContractError("response has no token usage")
    completion = usage.get("completion_tokens")
    if isinstance(completion, bool) or not isinstance(completion, int):
        raise runner.ResponseContractError("usage.completion_tokens is missing")
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        raise runner.ResponseContractError(
            "usage.completion_tokens_details is missing"
        )
    reasoning = details.get("reasoning_tokens")
    if (
        isinstance(reasoning, bool)
        or not isinstance(reasoning, int)
        or reasoning <= 0
        or reasoning > min(completion, THINKING_BUDGET)
    ):
        raise runner.ResponseContractError(
            "usage reasoning_tokens violates the v6 thinking bound"
        )
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise runner.ResponseContractError(
            "v6 requires exactly one response choice"
        )
    message = (
        choices[0].get("message")
        if isinstance(choices[0], dict)
        else None
    )
    reasoning_content = (
        message.get("reasoning_content")
        if isinstance(message, dict)
        else None
    )
    if (
        not isinstance(reasoning_content, str)
        or not reasoning_content.strip()
    ):
        raise runner.ResponseContractError(
            "v6 response reasoning_content is empty or missing"
        )
    terminal = runner.classify_terminal_provider_response(
        response,
        expected_model=expected_model,
        max_prompt_tokens=PROMPT_CAP,
        requested_max_tokens=VALIDATION_CAP,
    )
    normalized = dict(terminal.usage)
    normalized["reasoning_tokens"] = reasoning
    normalized["answer_tokens"] = completion - reasoning
    return dataclasses.replace(terminal, usage=normalized)


def service_active(service: str) -> bool:
    completed = subprocess.run(
        ("systemctl", "is-active", "--quiet", service),
        check=False,
        timeout=10,
    )
    return completed.returncode == 0


def validate_preview_guard(workspace: Path) -> dict[str, Any]:
    path = workspace / PREVIEW_GUARD_RELATIVE
    if sha256_file(path) != PREVIEW_GUARD_SHA256:
        raise AuditError("Preview quota-guard receipt SHA mismatch")
    receipt = read_json(path)
    exact = {
        "schema": "qwen37-model-free-quota-stop-receipt-v1",
        "status": "stopped_model_units",
        "model": "qwen3.7-max-preview",
        "scope": "exact_requested_model_only",
        "mapping_sha256": PREVIEW_GUARD_MAPPING_SHA256,
        "deepseek_units_targeted": False,
    }
    for key, value in exact.items():
        if receipt.get(key) != value:
            raise AuditError(f"Preview guard mismatch for {key}")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 2:
        raise AuditError("Preview guard must bind exactly two 403 evidence rows")
    for item in evidence:
        if (
            not isinstance(item, dict)
            or item.get("http_status") != 403
            or item.get("response_received") is not False
            or item.get("provider_error_variant") != "insufficient_quota"
            or len(str(item.get("row_sha256") or "")) != 64
        ):
            raise AuditError("Preview guard evidence is malformed")
    stop_results = receipt.get("stop_results")
    expected_units = {
        shard.service_template.format(arm=arm)
        for shard in SOURCE_SHARDS
        if shard.source_model == "qwen3.7-max-preview"
        for arm in ("opus", "codex")
    }
    if not isinstance(stop_results, list) or {
        str(item.get("unit") or "")
        for item in stop_results
        if isinstance(item, dict)
    } != expected_units:
        raise AuditError("Preview guard stopped-unit mapping mismatch")
    for item in stop_results:
        after = item.get("after")
        if (
            not isinstance(after, dict)
            or after.get("active_state") != "inactive"
            or item.get("stop_returncode") != 0
        ):
            raise AuditError("Preview guard stop receipt is incomplete")
    if any(service_active(unit) for unit in expected_units):
        raise AuditError("a Preview source unit became active after guard stop")
    return {
        "path": str(path),
        "sha256": PREVIEW_GUARD_SHA256,
        "mapping_sha256": PREVIEW_GUARD_MAPPING_SHA256,
        "evidence_row_sha256s": sorted(
            str(item["row_sha256"]) for item in evidence
        ),
    }


def selection_id(
    *,
    arm: str,
    shard_key: str,
    task_id: str,
    global_index: int,
    source_config_sha256: str,
    source_model: str = SOURCE_MODEL,
    trigger: str = "per_shard_quota_403",
) -> str:
    value = {
        "schema": SCHEMA,
        "parent_contract_sha256": EXPECTED_PARENT_CONTRACT_SHA256,
        "arm": arm,
        "source_shard_key": shard_key,
        "task_id": task_id,
        "global_sample_index": global_index,
        "source_config_sha256": source_config_sha256,
        "source_model": source_model,
        "trigger": trigger,
    }
    return canonical_sha(value)


def _validate_source_config(
    provenance: Mapping[str, Any], shard: SourceShard, arm: str
) -> tuple[str, str]:
    config = provenance.get("config")
    if not isinstance(config, dict):
        raise AuditError("source provenance lacks config")
    config_sha = str(provenance.get("config_sha256") or "")
    if config_sha != canonical_sha(config):
        raise AuditError("source config SHA mismatch")
    expected = {
        "provider": "qwen",
        "model_requested": shard.source_model,
        "k": shard.local_k,
        "max_output_tokens": CAP,
        "max_prompt_tokens": PROMPT_CAP,
        "temperature": 0.8,
        "top_p": 0.95,
        "budget": 0,
        "extra_body": {
            "enable_thinking": True,
            "thinking_budget": THINKING_BUDGET,
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise AuditError(f"source config mismatch for {key}")
    pair_key = (
        "opus_real_fn0_cfg" if arm == "opus" else "codex_multifunction_cfg"
    )
    if config.get("pair_arm_key") != pair_key:
        raise AuditError("source pair-arm key mismatch")
    slot_policy = config.get("slot_policy")
    if not isinstance(slot_policy, dict):
        raise AuditError("source slot policy missing")
    policy_sha = str(config.get("slot_policy_sha256") or "")
    if policy_sha != canonical_sha(slot_policy):
        raise AuditError("source slot-policy SHA mismatch")
    return config_sha, policy_sha


def _validate_terminal(
    row: Mapping[str, Any],
    *,
    model: str,
    prompt_sha: str,
) -> None:
    if (
        row.get("response_received") is not True
        or row.get("slot_terminal") is not True
        or row.get("requested_model") != model
        or row.get("resolved_model") != model
        or row.get("prompt_sha256") != prompt_sha
    ):
        raise AuditError("terminal identity contract mismatch")
    raw = row.get("response")
    if not isinstance(raw, Mapping):
        raise AuditError("terminal raw response missing")
    classified = classify_v6_terminal_response(
        dict(raw),
        expected_model=model,
    )
    exact = {
        "response_id": classified.response_id,
        "finish_reason": classified.finish_reason,
        "candidate_valid": classified.candidate_valid,
        "terminal_reason": classified.terminal_reason,
        "code_sha256": classified.code_sha256,
        "usage": classified.usage,
    }
    for key, value in exact.items():
        if row.get(key) != value:
            raise AuditError(f"terminal field mismatch for {key}")


def source_missing_map(
    workspace: Path,
) -> dict[tuple[str, str, int], bool]:
    """Return missing state for pair-aware selection without reading outcomes."""
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    state: dict[tuple[str, str, int], bool] = {}
    for shard in SOURCE_SHARDS:
        for arm in ("opus", "codex"):
            root = run_root / shard.directory_template.format(arm=arm)
            tasks = read_jsonl(root / "tasks.jsonl")
            attempts = read_jsonl(root / "attempts.jsonl")
            terminal = {
                (str(row.get("task_id") or ""), int(row.get("sample_index", -1)))
                for row in attempts
                if row.get("response_received") is True
                and row.get("slot_terminal") is True
            }
            for task in tasks:
                task_id = str(task.get("task_id") or "")
                for local_index, global_index in enumerate(shard.global_indices):
                    state[(arm, task_id, global_index)] = (
                        (task_id, local_index) not in terminal
                    )
    return state


def build_targets(
    workspace: Path,
    *,
    arm: str,
    partition: str,
) -> list[dict[str, Any]]:
    if arm not in {"opus", "codex"}:
        raise AuditError("arm must be opus or codex")
    if partition not in PARTITIONS:
        raise AuditError("unknown partition")
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    preview_guard = validate_preview_guard(workspace)
    pair_missing = source_missing_map(workspace)
    targets: list[dict[str, Any]] = []
    for shard in SOURCE_SHARDS:
        if shard.partition != partition:
            continue
        root = run_root / shard.directory_template.format(arm=arm)
        if (
            shard.trigger == "per_shard_quota_403"
            and not (root / "failure.json").is_file()
        ):
            raise AuditError(f"source shard lacks failed-closed receipt: {root}")
        provenance = read_json(root / "provenance.json")
        config_sha, policy_sha = _validate_source_config(provenance, shard, arm)
        tasks = read_jsonl(root / "tasks.jsonl")
        prompts = read_jsonl(root / "prompts.jsonl")
        attempts = read_jsonl(root / "attempts.jsonl")
        if len(tasks) != EXPECTED_TASKS or len(prompts) != EXPECTED_TASKS:
            raise AuditError("source task/prompt count mismatch")
        _load_test_bundle(root)
        task_ids = [str(row.get("task_id") or "") for row in tasks]
        prompt_ids = [str(row.get("task_id") or "") for row in prompts]
        if task_ids != prompt_ids or len(set(task_ids)) != EXPECTED_TASKS:
            raise AuditError("source ordered task/prompt identity mismatch")
        prompt_map = {str(row["task_id"]): row for row in prompts}
        quota_rows = [
            row
            for row in attempts
            if row.get("response_received") is False
            and exact_quota_403(str(row.get("transport_error") or ""))
        ]
        other_response_less = [
            row
            for row in attempts
            if row.get("response_received") is False and row not in quota_rows
        ]
        if other_response_less:
            raise AuditError(
                "source selection requires only exact response-less quota-403 "
                "boundary rows"
            )
        if shard.trigger == "per_shard_quota_403" and not quota_rows:
            raise AuditError("05-17 source shard lacks its exact quota boundary")
        if shard.trigger == "preview_model_guard" and service_active(
            shard.service_template.format(arm=arm)
        ):
            raise AuditError("Preview source unit is active after guard receipt")
        boundary_hashes = sorted(canonical_sha(row) for row in quota_rows)
        terminal: dict[tuple[str, int], dict[str, Any]] = {}
        for row in attempts:
            if row.get("response_received") is not True:
                continue
            task_id = str(row.get("task_id") or "")
            local_index = int(row.get("sample_index", -1))
            if (task_id, local_index) in terminal:
                raise AuditError("duplicate terminal in source shard")
            prompt = prompt_map.get(task_id)
            if not isinstance(prompt, dict):
                raise AuditError("source terminal has unknown task")
            _validate_terminal(
                row,
                model=shard.source_model,
                prompt_sha=str(prompt.get("prompt_sha256") or ""),
            )
            terminal[(task_id, local_index)] = row
        for task in tasks:
            task_id = str(task["task_id"])
            prompt = prompt_map[task_id]
            for local_index, global_index in enumerate(shard.global_indices):
                if (task_id, local_index) in terminal:
                    continue
                exact_rows = [
                    row
                    for row in quota_rows
                    if row.get("task_id") == task_id
                    and row.get("sample_index") == local_index
                ]
                other_arm = "codex" if arm == "opus" else "opus"
                paired_missing = pair_missing.get(
                    (other_arm, task_id, global_index), False
                )
                target = {
                    "schema": SCHEMA,
                    "record_type": "capacity_target",
                    "selection_id": selection_id(
                        arm=arm,
                        shard_key=shard.key,
                        task_id=task_id,
                        global_index=global_index,
                        source_config_sha256=config_sha,
                        source_model=shard.source_model,
                        trigger=shard.trigger,
                    ),
                    "parent_contract_sha256": EXPECTED_PARENT_CONTRACT_SHA256,
                    "overlay_contract_sha256": EXPECTED_CONTRACT_SHA256,
                    "arm": arm,
                    "pair_status": (
                        "both_arms_missing"
                        if paired_missing
                        else "only_this_arm_missing"
                    ),
                    "source_shard_key": shard.key,
                    "source_directory": str(root),
                    "source_model": shard.source_model,
                    "source_local_sample_index": local_index,
                    "global_sample_index": global_index,
                    "task_id": task_id,
                    "prompt_sha256": prompt["prompt_sha256"],
                    "source_config_sha256": config_sha,
                    "source_slot_policy_sha256": policy_sha,
                    "source_failure_sha256": (
                        sha256_file(root / "failure.json")
                        if (root / "failure.json").is_file()
                        else None
                    ),
                    "preview_guard_receipt": (
                        preview_guard
                        if shard.trigger == "preview_model_guard"
                        else None
                    ),
                    "source_boundary_row_sha256s": boundary_hashes,
                    "slot_exact_boundary_row_sha256s": sorted(
                        canonical_sha(row) for row in exact_rows
                    ),
                    "trigger": (
                        "missing_after_failed_shard_exact_response_less_quota_403"
                        if shard.trigger == "per_shard_quota_403"
                        else "missing_after_model_scoped_preview_guard_403"
                    ),
                    "alias_order": list(PARTITIONS[partition]["aliases"]),
                    "selection_reads_outcomes": False,
                }
                target["selection_record_sha256"] = canonical_sha(target)
                targets.append(target)
    targets.sort(
        key=lambda row: (
            str(row["task_id"]),
            int(row["global_sample_index"]),
        )
    )
    if not targets:
        raise AuditError("capacity target selection is empty")
    return targets


def overlay_config(
    workspace: Path,
    out: Path,
    *,
    arm: str,
    partition: str,
    targets_sha256: str,
) -> dict[str, Any]:
    this_path = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "arm": arm,
        "partition": partition,
        "targets_sha256": targets_sha256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "parent_contract_sha256": EXPECTED_PARENT_CONTRACT_SHA256,
        "out": str(out),
        "request_policy": {
            "max_completion_tokens": CAP,
            "provider_tolerance": 10,
            "completion_validation_cap": VALIDATION_CAP,
            "thinking_budget": THINKING_BUDGET,
            "temperature": 0.8,
            "top_p": 0.95,
            "prompt_cap": PROMPT_CAP,
        },
        "runtime_identity": {
            "capacity_runner_sha256": sha256_file(this_path),
            "frontier_runner_sha256": sha256_file(
                this_path.with_name("frontier_passk.py")
            ),
            "frontier_core_sha256": sha256_file(
                this_path.with_name("frontier_core.py")
            ),
            "qwen_entry_sha256": sha256_file(
                this_path.with_name("frontier_passk_qwen_completion.py")
            ),
            "openai_sdk_version": importlib.metadata.version("openai"),
        },
    }


def register_capacity_epoch(
    out: Path,
    *,
    capacity_epoch: str,
    credential_source: str,
    credential_env_file: Path,
    allow_preview: bool,
    include_undated_alias: bool,
    include_source_alias: bool,
    only_undated_alias: bool,
) -> dict[str, Any]:
    endpoint = os.environ.get("QWEN_BASE_URL", "").rstrip("/")
    if not endpoint.startswith("https://"):
        raise AuditError("QWEN_BASE_URL must be provided by a sealed wrapper")
    if credential_source not in {"secondary_qwen_env", "generic_fallback_env"}:
        raise AuditError("unsealed credential source")
    env_path = credential_env_file.resolve()
    if not env_path.is_file():
        raise AuditError(f"credential env file is missing: {env_path}")
    record = {
        "schema": SCHEMA,
        "record_type": "capacity_epoch",
        "capacity_epoch": capacity_epoch,
        "endpoint_sha256": runner.sha256_text(endpoint),
        "credential_source": credential_source,
        "credential_env_file": str(env_path),
        "api_key_persisted": False,
        "preview_alias_enabled": bool(allow_preview),
        "undated_alias_enabled": bool(include_undated_alias),
        "source_0517_alias_enabled": bool(include_source_alias),
        "undated_alias_only": bool(only_undated_alias),
        "registered_at": runner.utc_now(),
    }
    epoch_path = out / "capacity_epochs.jsonl"
    existing = read_jsonl(epoch_path)
    matches = [
        row for row in existing if row.get("capacity_epoch") == capacity_epoch
    ]
    if len(matches) > 1:
        raise AuditError("capacity epoch is duplicated")
    if matches:
        left = dict(matches[0])
        right = dict(record)
        left.pop("registered_at", None)
        right.pop("registered_at", None)
        if left != right:
            raise AuditError(
                "capacity epoch name was rebound to different endpoint semantics"
            )
        return matches[0]
    runner.JsonlJournal(epoch_path).append(record)
    return record


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.only_undated_alias and (
        not args.include_undated_alias
        or args.allow_preview
        or args.include_source_alias
    ):
        raise AuditError(
            "only-undated routing requires exactly --include-undated-alias"
        )
    workspace = args.workspace.resolve()
    patch_root = workspace / "frontier_ceiling_patch_v1"
    contract = patch_root / CONTRACT_NAME
    if sha256_file(contract) != EXPECTED_CONTRACT_SHA256:
        raise AuditError("v6 capacity contract hash mismatch")
    parent = patch_root / "qwen37_primary_alias_fallback_contract_v5.json"
    if sha256_file(parent) != EXPECTED_PARENT_CONTRACT_SHA256:
        raise AuditError("parent v5 contract hash mismatch")
    if sha256_file(patch_root / "frontier_passk.py") != EXPECTED_RUNNER_SHA256:
        raise AuditError("frontier runner hash mismatch")
    if sha256_file(patch_root / "frontier_core.py") != EXPECTED_CORE_SHA256:
        raise AuditError("frontier core hash mismatch")
    if (
        sha256_file(patch_root / "frontier_passk_qwen_completion.py")
        != EXPECTED_QWEN_ENTRY_SHA256
    ):
        raise AuditError("Qwen entry hash mismatch")
    targets = build_targets(
        workspace, arm=args.arm, partition=args.partition
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    epoch_record = register_capacity_epoch(
        out,
        capacity_epoch=args.capacity_epoch,
        credential_source=args.credential_source,
        credential_env_file=args.credential_env_file,
        allow_preview=args.allow_preview,
        include_undated_alias=args.include_undated_alias,
        include_source_alias=args.include_source_alias,
        only_undated_alias=args.only_undated_alias,
    )
    target_path = out / "targets.jsonl"
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in targets
    )
    if target_path.exists():
        if target_path.read_text(encoding="utf-8") != encoded:
            raise AuditError("sealed target selection changed on resume")
    else:
        target_path.write_text(encoded, encoding="utf-8", newline="\n")
    target_sha = sha256_file(target_path)
    config = overlay_config(
        workspace,
        out,
        arm=args.arm,
        partition=args.partition,
        targets_sha256=target_sha,
    )
    config_sha = canonical_sha(config)
    provenance = {
        "schema": SCHEMA,
        "status": "preflight_complete",
        "created_at": runner.utc_now(),
        "contract": runner.file_record(contract),
        "parent_contract": runner.file_record(parent),
        "targets": runner.file_record(target_path),
        "target_count": len(targets),
        "config": config,
        "config_sha256": config_sha,
        "selection_outcome_blind": True,
        "base_journals_modified": False,
    }
    provenance_path = out / "provenance.json"
    if provenance_path.exists():
        existing = read_json(provenance_path)
        if existing.get("config_sha256") != config_sha:
            raise AuditError("overlay provenance/config mismatch on resume")
    else:
        runner.atomic_write_json(provenance_path, provenance)
    copied = out / CONTRACT_NAME
    if copied.exists():
        if sha256_file(copied) != EXPECTED_CONTRACT_SHA256:
            raise AuditError("copied overlay contract mismatch")
    else:
        copied.write_bytes(contract.read_bytes())
    return {
        "targets": targets,
        "config": config,
        "config_sha256": config_sha,
        "provenance": provenance,
        "capacity_epoch_record": epoch_record,
    }


def _diagnostic_local_index(partition: str, global_index: int) -> int:
    return global_index if partition == "0520" else global_index - 5


def _matching_outcome(
    rows: Iterable[Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> dict[str, Any] | None:
    matches = [
        dict(row)
        for row in rows
        if row.get("task_id") == terminal.get("task_id")
        and row.get("sample_index") == terminal.get("sample_index")
        and row.get("attempt_id") == terminal.get("attempt_id")
        and row.get("response_id") == terminal.get("response_id")
    ]
    if len(matches) > 1:
        raise AuditError("duplicate diagnostic outcome")
    return matches[0] if matches else None


def adopt_diagnostic(
    workspace: Path,
    target: Mapping[str, Any],
    *,
    partition: str,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """Return ADOPTED/WAIT/NONE/INSPECTION and immutable source rows."""
    spec = PARTITIONS[partition]
    if int(target["global_sample_index"]) not in set(
        spec["diagnostic_reuse_indices"]
    ):
        return "NONE", None, None
    arm = str(target["arm"])
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    root = run_root / str(spec["diagnostic_dir"]).format(arm=arm)
    provenance = read_json(root / "provenance.json")
    config = provenance.get("config")
    if not isinstance(config, dict):
        raise AuditError("diagnostic provenance lacks config")
    config_sha = str(provenance.get("config_sha256") or "")
    if config_sha != canonical_sha(config):
        raise AuditError("diagnostic config SHA mismatch")
    slot_policy = config.get("slot_policy")
    if (
        not isinstance(slot_policy, dict)
        or config.get("slot_policy_sha256") != canonical_sha(slot_policy)
    ):
        raise AuditError("diagnostic slot-policy SHA mismatch")
    local_index = _diagnostic_local_index(
        partition, int(target["global_sample_index"])
    )
    attempts = read_jsonl(root / "attempts.jsonl")
    candidates = [
        row
        for row in attempts
        if row.get("task_id") == target["task_id"]
        and row.get("sample_index") == local_index
    ]
    terminal = [
        row
        for row in candidates
        if row.get("response_received") is True
        and row.get("slot_terminal") is True
    ]
    if len(terminal) > 1:
        raise AuditError("diagnostic slot has multiple terminal responses")
    if terminal:
        row = terminal[0]
        if (
            row.get("config_sha256") != config_sha
            or row.get("slot_policy_sha256")
            != config.get("slot_policy_sha256")
        ):
            raise AuditError("diagnostic terminal provenance mismatch")
        _validate_terminal(
            row,
            model=str(spec["diagnostic_model"]),
            prompt_sha=str(target["prompt_sha256"]),
        )
        outcome = _matching_outcome(
            read_jsonl(root / "outcomes.jsonl"), row
        )
        if outcome is None:
            return "TERMINAL_WAIT_OUTCOME", row, None
        exact_outcome = {
            "config_sha256": config_sha,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "response_id": row["response_id"],
            "finish_reason": row["finish_reason"],
            "candidate_valid": row["candidate_valid"],
            "terminal_reason": row["terminal_reason"],
            "code_sha256": row["code_sha256"],
        }
        for key, value in exact_outcome.items():
            if outcome.get(key) != value:
                raise AuditError(
                    f"diagnostic outcome mismatch for {key}"
                )
        if (
            type(outcome.get("compiled")) is not bool
            or type(outcome.get("passed")) is not bool
            or (
                outcome.get("passed") is True
                and outcome.get("compiled") is not True
            )
        ):
            raise AuditError("diagnostic outcome booleans are invalid")
        expected_runs = 2 if row.get("candidate_valid") is True else 0
        runs = outcome.get("stability_runs")
        if not isinstance(runs, list) or len(runs) != expected_runs:
            raise AuditError("diagnostic outcome stability count mismatch")
        return "ADOPTED", row, outcome
    for row in candidates:
        if row.get("response_received") is False and data_inspection_failed(
            row.get("transport_error")
        ):
            return "INSPECTION", row, None
    service = str(spec["diagnostic_service"]).format(arm=arm)
    if service_active(service):
        return "WAIT", None, None
    return "NONE", None, None


def _request_args(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        provider="qwen",
        model=model,
        max_output_tokens=CAP,
        budget=0,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": THINKING_BUDGET,
        },
        temperature=0.8,
        top_p=0.95,
        timeout_seconds=1800,
    )


def _make_outcome(
    *,
    terminal: Mapping[str, Any],
    target: Mapping[str, Any],
    evaluator: Any,
    evaluator_sha: str,
    tests: str,
    selection_id_value: str,
) -> dict[str, Any]:
    if terminal.get("candidate_valid") is True:
        evaluation = runner.evaluate_candidate_stably(
            evaluator,
            code=str(terminal.get("code") or ""),
            tests=tests,
            task_id=str(target["task_id"]),
            sample_index=int(target["global_sample_index"]),
            stability_runs=2,
            timeout=30,
        )
        performed = True
    else:
        evaluation = {
            "compiled": False,
            "passed": False,
            "completion_attestation_id": runner.REQUIRED_ATTESTATION_ID,
            "completion_attestation_enforced": False,
            "completion_attestation_satisfied_all_runs": False,
            "stability_runs": [],
        }
        performed = False
    return {
        "schema": SCHEMA,
        "record_type": "capacity_candidate_outcome",
        "selection_id": selection_id_value,
        "task_id": target["task_id"],
        "global_sample_index": target["global_sample_index"],
        "arm": target["arm"],
        "attempt_id": terminal["attempt_id"],
        "response_id": terminal["response_id"],
        "model": terminal["resolved_model"],
        "finish_reason": terminal["finish_reason"],
        "candidate_valid": terminal["candidate_valid"],
        "terminal_reason": terminal["terminal_reason"],
        "code_sha256": terminal["code_sha256"],
        "evaluator_sha256": evaluator_sha,
        "evaluation_performed": performed,
        "compiled": evaluation["compiled"],
        "passed": evaluation["passed"],
        "completion_attestation_id": evaluation[
            "completion_attestation_id"
        ],
        "completion_attestation_enforced": evaluation[
            "completion_attestation_enforced"
        ],
        "completion_attestation_satisfied_all_runs": evaluation[
            "completion_attestation_satisfied_all_runs"
        ],
        "stability_runs": evaluation["stability_runs"],
        "evaluated_at": runner.utc_now(),
    }


def _effective_record(
    *,
    target: Mapping[str, Any],
    terminal: Mapping[str, Any],
    outcome: Mapping[str, Any],
    origin: str,
    effective_source_directory: str,
    effective_config_sha256: str,
    effective_slot_policy_sha256: str,
    effective_endpoint_sha256: str,
) -> dict[str, Any]:
    row = {
        "schema": SCHEMA,
        "record_type": "effective_capacity_terminal",
        "selection_id": target["selection_id"],
        "overlay_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "parent_contract_sha256": EXPECTED_PARENT_CONTRACT_SHA256,
        "arm": target["arm"],
        "pair_status": target["pair_status"],
        "source_shard_key": target["source_shard_key"],
        "source_directory": target["source_directory"],
        "source_config_sha256": target["source_config_sha256"],
        "source_slot_policy_sha256": target[
            "source_slot_policy_sha256"
        ],
        "source_local_sample_index": target[
            "source_local_sample_index"
        ],
        "global_sample_index": target["global_sample_index"],
        "task_id": target["task_id"],
        "prompt_sha256": target["prompt_sha256"],
        "origin": origin,
        "effective_source_directory": effective_source_directory,
        "effective_source_config_sha256": effective_config_sha256,
        "effective_source_slot_policy_sha256": effective_slot_policy_sha256,
        "effective_endpoint_sha256": effective_endpoint_sha256,
        "capacity_epoch": terminal.get("capacity_epoch"),
        "effective_attempt_id": terminal["attempt_id"],
        "response_id": terminal["response_id"],
        "requested_model": terminal["requested_model"],
        "resolved_model": terminal["resolved_model"],
        "finish_reason": terminal["finish_reason"],
        "candidate_valid": terminal["candidate_valid"],
        "terminal_reason": terminal["terminal_reason"],
        "code_sha256": terminal["code_sha256"],
        "usage": terminal["usage"],
        "canonical_terminal_row_sha256": canonical_sha(terminal),
        "canonical_outcome_row_sha256": canonical_sha(outcome),
        "compiled": outcome["compiled"],
        "passed": outcome["passed"],
        "selected_without_outcome_inspection": True,
        "effective_at": runner.utc_now(),
    }
    return row


def _terminal_feed_record(
    *,
    target: Mapping[str, Any],
    terminal: Mapping[str, Any],
    origin: str,
    overlay_config_sha256: str,
    effective_source_directory: str,
    effective_config_sha256: str,
    effective_slot_policy_sha256: str,
    effective_endpoint_sha256: str,
) -> dict[str, Any]:
    """Outcome-free effective-terminal projection consumed by length repair."""
    row = {
        "schema": SCHEMA,
        "record_type": "capacity_effective_terminal_feed",
        "source_kind": "capacity_v6",
        "selection_id": target["selection_id"],
        "overlay_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "parent_contract_sha256": EXPECTED_PARENT_CONTRACT_SHA256,
        "overlay_config_sha256": overlay_config_sha256,
        "arm": target["arm"],
        "pair_status": target["pair_status"],
        "originating_shard_key": target["source_shard_key"],
        "originating_source_directory": target["source_directory"],
        "originating_source_config_sha256": target[
            "source_config_sha256"
        ],
        "originating_source_slot_policy_sha256": target[
            "source_slot_policy_sha256"
        ],
        "originating_local_sample_index": target[
            "source_local_sample_index"
        ],
        "global_sample_index": target["global_sample_index"],
        "task_id": target["task_id"],
        "prompt_sha256": target["prompt_sha256"],
        "effective_origin": origin,
        "effective_source_directory": effective_source_directory,
        "effective_source_config_sha256": effective_config_sha256,
        "effective_source_slot_policy_sha256": effective_slot_policy_sha256,
        "effective_endpoint_sha256": effective_endpoint_sha256,
        "capacity_epoch": terminal.get("capacity_epoch"),
        "effective_attempt_id": terminal["attempt_id"],
        "response_id": terminal["response_id"],
        "requested_model": terminal["requested_model"],
        "resolved_model": terminal["resolved_model"],
        "finish_reason": terminal["finish_reason"],
        "candidate_valid": terminal["candidate_valid"],
        "terminal_reason": terminal["terminal_reason"],
        "code_sha256": terminal["code_sha256"],
        "validated_usage": terminal["usage"],
        "reasoning_content": terminal["reasoning_content"],
        "content": terminal["content"],
        "raw_response": terminal["response"],
        "effective_terminal_canonical_row_sha256": canonical_sha(terminal),
        "request_max_completion_tokens": CAP,
        "thinking_budget": THINKING_BUDGET,
        "selection_reads_outcomes": False,
        "published_at": runner.utc_now(),
    }
    immutable = dict(row)
    immutable.pop("published_at")
    row["terminal_feed_payload_sha256"] = canonical_sha(immutable)
    return row


def _load_terminal_feed(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    keys: set[tuple[str, str, int]] = set()
    response_ids: set[str] = set()
    for row in read_jsonl(path):
        sid = str(row.get("selection_id") or "")
        key = (
            str(row.get("arm") or ""),
            str(row.get("task_id") or ""),
            int(row.get("global_sample_index", -1)),
        )
        response_id = str(row.get("response_id") or "")
        if not sid or sid in result or key in keys:
            raise AuditError("duplicate/missing terminal-feed selection/key")
        if not response_id or response_id in response_ids:
            raise AuditError("duplicate/missing terminal-feed response ID")
        immutable = dict(row)
        observed_payload_sha = immutable.pop(
            "terminal_feed_payload_sha256", None
        )
        immutable.pop("published_at", None)
        if observed_payload_sha != canonical_sha(immutable):
            raise AuditError("terminal-feed payload SHA mismatch")
        result[sid] = row
        keys.add(key)
        response_ids.add(response_id)
    return result


def _publish_terminal_feed(
    *,
    journal: Any,
    feed: dict[str, dict[str, Any]],
    target: Mapping[str, Any],
    terminal: Mapping[str, Any],
    origin: str,
    overlay_config_sha256: str,
    effective_source_directory: str,
    effective_config_sha256: str,
    effective_slot_policy_sha256: str,
    effective_endpoint_sha256: str,
) -> dict[str, Any]:
    row = _terminal_feed_record(
        target=target,
        terminal=terminal,
        origin=origin,
        overlay_config_sha256=overlay_config_sha256,
        effective_source_directory=effective_source_directory,
        effective_config_sha256=effective_config_sha256,
        effective_slot_policy_sha256=effective_slot_policy_sha256,
        effective_endpoint_sha256=effective_endpoint_sha256,
    )
    sid = str(target["selection_id"])
    existing = feed.get(sid)
    if existing is not None:
        left = dict(existing)
        right = dict(row)
        for value in (left, right):
            value.pop("published_at", None)
            value.pop("terminal_feed_payload_sha256", None)
        if left != right:
            raise AuditError("terminal feed changed for an existing selection")
        return existing
    journal.append(row)
    feed[sid] = row
    return row


def _load_effective(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    response_ids: set[str] = set()
    for row in read_jsonl(path):
        sid = str(row.get("selection_id") or "")
        rid = str(row.get("response_id") or "")
        if not sid or sid in result:
            raise AuditError("duplicate/missing effective selection ID")
        if not rid or rid in response_ids:
            raise AuditError("duplicate/missing effective response ID")
        result[sid] = row
        response_ids.add(rid)
    return result


_TEST_BUNDLE_CACHE: dict[Path, dict[str, str]] = {}


def _load_test_bundle(source_root: Path) -> dict[str, str]:
    """Load tests from the source run's hash-sealed eval dataset.

    ``tasks.jsonl`` intentionally excludes private plaintext tests.  It binds
    each task to the tests with hashes, while provenance binds the complete
    eval JSONL.  Validate both layers before exposing a test to evaluation.
    """
    source_root = source_root.resolve()
    cached = _TEST_BUNDLE_CACHE.get(source_root)
    if cached is not None:
        return cached

    provenance = read_json(source_root / "provenance.json")
    config = provenance.get("config")
    if not isinstance(config, dict):
        raise AuditError("source provenance lacks config for sealed tests")
    sealed = config.get("sealed_inputs")
    if not isinstance(sealed, dict):
        raise AuditError("source provenance lacks sealed_inputs")
    eval_path_raw = sealed.get("eval_jsonl")
    eval_sha = str(sealed.get("eval_jsonl_sha256") or "")
    if not isinstance(eval_path_raw, str) or not eval_path_raw.strip():
        raise AuditError("source sealed eval JSONL path is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", eval_sha):
        raise AuditError("source sealed eval JSONL SHA is malformed")
    eval_path = Path(eval_path_raw).resolve()
    if sha256_file(eval_path) != eval_sha:
        raise AuditError("source sealed eval JSONL SHA mismatch")

    tasks = read_jsonl(source_root / "tasks.jsonl")
    eval_rows = read_jsonl(eval_path)
    if len(tasks) != EXPECTED_TASKS or len(eval_rows) != EXPECTED_TASKS:
        raise AuditError("sealed test task/eval count mismatch")
    task_ids = [str(row.get("task_id") or "") for row in tasks]
    eval_ids = [str(row.get("task_id") or "") for row in eval_rows]
    if (
        task_ids != eval_ids
        or len(set(task_ids)) != EXPECTED_TASKS
        or any(not task_id for task_id in task_ids)
    ):
        raise AuditError("sealed test ordered task identity mismatch")

    result: dict[str, str] = {}
    for task, eval_row in zip(tasks, eval_rows, strict=True):
        task_id = str(task["task_id"])
        tests = eval_row.get("tests")
        acceptance = eval_row.get("acceptance_tests")
        if (
            not isinstance(tests, str)
            or not tests.strip()
            or not isinstance(acceptance, str)
            or not acceptance.strip()
        ):
            raise AuditError("sealed eval acceptance tests missing")
        if tests != acceptance:
            raise AuditError("sealed eval tests/acceptance_tests differ")
        tests_sha = runner.sha256_text(tests)
        acceptance_sha = runner.sha256_text(acceptance)
        if (
            task.get("tests_equal_acceptance_tests") is not True
            or task.get("tests_sha256") != tests_sha
            or task.get("acceptance_tests_sha256") != acceptance_sha
        ):
            raise AuditError("source task test-hash binding mismatch")
        result[task_id] = acceptance
    _TEST_BUNDLE_CACHE[source_root] = result
    return result


def _load_tests(source_root: Path, task_id: str) -> str:
    tests = _load_test_bundle(source_root).get(task_id)
    if tests is None:
        raise AuditError("target task missing from sealed eval tests")
    return tests


def _load_messages(source_root: Path, target: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = read_jsonl(source_root / "prompts.jsonl")
    matches = [row for row in rows if row.get("task_id") == target["task_id"]]
    if len(matches) != 1:
        raise AuditError("target prompt missing")
    prompt = matches[0]
    if prompt.get("prompt_sha256") != target["prompt_sha256"]:
        raise AuditError("target prompt SHA changed")
    messages = prompt.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise AuditError("target prompt messages malformed")
    return [dict(value) for value in messages]


def _enabled_route_aliases(
    *,
    partition: str,
    allow_preview: bool,
    include_undated_alias: bool,
    include_source_alias: bool,
    only_undated_alias: bool,
) -> list[str]:
    route_aliases = list(PARTITIONS[partition]["aliases"])
    if include_source_alias:
        route_aliases.append(SOURCE_MODEL)
    if len(route_aliases) != len(set(route_aliases)):
        raise AuditError("capacity alias route contains duplicates")
    result: list[str] = []
    for model in route_aliases:
        if only_undated_alias and model != "qwen3.7-max":
            continue
        if model == "qwen3.7-max-preview" and not allow_preview:
            continue
        if model == "qwen3.7-max" and not include_undated_alias:
            continue
        if model == SOURCE_MODEL and not include_source_alias:
            continue
        result.append(model)
    if not result:
        raise AuditError("capacity route has no enabled aliases")
    return result


def _all_enabled_aliases_exactly_unavailable(
    enabled_aliases: Iterable[str],
    unavailable_aliases: set[str],
) -> bool:
    enabled = tuple(enabled_aliases)
    return bool(enabled) and all(
        model in unavailable_aliases for model in enabled
    )


def run_overlay(args: argparse.Namespace) -> int:
    prepared = preflight(args)
    workspace = args.workspace.resolve()
    out = args.out.resolve()
    config_sha = prepared["config_sha256"]
    targets = prepared["targets"]
    effective_path = out / "effective_terminals.jsonl"
    route_path = out / "route_attempts.jsonl"
    outcomes_path = out / "outcomes.jsonl"
    dispatch_path = out / "dispatches.jsonl"
    terminal_feed_path = out / "effective_terminal_feed.jsonl"
    effective_journal = runner.JsonlJournal(effective_path)
    route_journal = runner.JsonlJournal(route_path)
    outcome_journal = runner.JsonlJournal(outcomes_path)
    dispatch_journal = runner.JsonlJournal(dispatch_path)
    terminal_feed_journal = runner.JsonlJournal(terminal_feed_path)
    effective = _load_effective(effective_path)
    terminal_feed = _load_terminal_feed(terminal_feed_path)
    epoch_rows = read_jsonl(out / "capacity_epochs.jsonl")
    epoch_endpoints = {
        str(row.get("capacity_epoch") or ""): str(
            row.get("endpoint_sha256") or ""
        )
        for row in epoch_rows
    }
    if len(epoch_endpoints) != len(epoch_rows):
        raise AuditError("duplicate capacity epoch registry entry")
    route_rows = read_jsonl(route_path)
    direct_terminals: dict[str, dict[str, Any]] = {}
    for row in route_rows:
        if (
            row.get("record_type") == "capacity_terminal_response"
            and row.get("response_received") is True
            and row.get("slot_terminal") is True
        ):
            sid = str(row.get("selection_id") or "")
            if not sid or sid in direct_terminals:
                raise AuditError("duplicate/missing direct terminal selection")
            direct_terminals[sid] = row
    overlay_outcomes = {
        str(row.get("selection_id") or ""): row
        for row in read_jsonl(outcomes_path)
    }
    if len(overlay_outcomes) != len(read_jsonl(outcomes_path)):
        raise AuditError("duplicate overlay outcome selection")
    permanent_inspection = {
        (str(row.get("selection_id")), str(row.get("requested_model")))
        for row in route_rows
        if row.get("route_result") == "data_inspection_failed"
    }
    unavailable: set[str] = {
        str(row.get("requested_model"))
        for row in route_rows
        if row.get("capacity_epoch") == args.capacity_epoch
        and row.get("route_result") == "quota_exhausted_403"
    }
    evaluator_path = workspace / EVALUATOR_RELATIVE
    evaluator_module, evaluator_record = runner.import_evaluator(
        evaluator_path,
        EXPECTED_EVALUATOR_SHA256,
        dart_binary=DART_PATH,
        expected_dart_hash=EXPECTED_DART_SHA256,
        validate_dart=True,
    )
    evaluator = evaluator_module.evaluate_dart_jit_tests_detail
    api_key = os.environ.get("QWEN_API_KEY", "")
    base_url = os.environ.get("QWEN_BASE_URL", "").rstrip("/")
    if not api_key or not base_url:
        raise AuditError("credential wrapper did not provide Qwen credentials")
    client = OpenAI(api_key=api_key, base_url=base_url)
    epoch_record = prepared["capacity_epoch_record"]
    endpoint_sha256 = str(epoch_record["endpoint_sha256"])
    if runner.sha256_text(base_url) != endpoint_sha256:
        raise AuditError("runtime endpoint disagrees with sealed capacity epoch")
    enabled_aliases = _enabled_route_aliases(
        partition=args.partition,
        allow_preview=args.allow_preview,
        include_undated_alias=args.include_undated_alias,
        include_source_alias=args.include_source_alias,
        only_undated_alias=args.only_undated_alias,
    )
    idle_cycles = 0
    completed_new = 0
    while True:
        progress = False
        waiting = False
        for target in targets:
            sid = str(target["selection_id"])
            if sid in effective:
                continue
            if args.max_new and completed_new >= args.max_new:
                break
            source_root = Path(str(target["source_directory"]))
            tests = _load_tests(source_root, str(target["task_id"]))
            resumed_terminal = direct_terminals.get(sid)
            if resumed_terminal is not None:
                terminal_epoch = str(
                    resumed_terminal.get("capacity_epoch") or ""
                )
                if (
                    not terminal_epoch
                    or epoch_endpoints.get(terminal_epoch)
                    != resumed_terminal.get("endpoint_sha256")
                ):
                    raise AuditError(
                        "resumed terminal endpoint/epoch binding mismatch"
                    )
                _validate_terminal(
                    resumed_terminal,
                    model=str(resumed_terminal.get("requested_model") or ""),
                    prompt_sha=str(target["prompt_sha256"]),
                )
                _publish_terminal_feed(
                    journal=terminal_feed_journal,
                    feed=terminal_feed,
                    target=target,
                    terminal=resumed_terminal,
                    origin="fresh_capacity_alias_response",
                    overlay_config_sha256=config_sha,
                    effective_source_directory=str(out),
                    effective_config_sha256=config_sha,
                    effective_slot_policy_sha256=canonical_sha(
                        prepared["config"]["request_policy"]
                    ),
                    effective_endpoint_sha256=endpoint_sha256,
                )
                outcome_row = overlay_outcomes.get(sid)
                if outcome_row is None:
                    outcome_row = _make_outcome(
                        terminal=resumed_terminal,
                        target=target,
                        evaluator=evaluator,
                        evaluator_sha=evaluator_record["sha256"],
                        tests=tests,
                        selection_id_value=sid,
                    )
                    outcome_journal.append(outcome_row)
                    overlay_outcomes[sid] = outcome_row
                effective_row = _effective_record(
                    target=target,
                    terminal=resumed_terminal,
                    outcome=outcome_row,
                    origin="fresh_capacity_alias_response",
                    effective_source_directory=str(out),
                    effective_config_sha256=config_sha,
                    effective_slot_policy_sha256=canonical_sha(
                        prepared["config"]["request_policy"]
                    ),
                    effective_endpoint_sha256=endpoint_sha256,
                )
                effective_journal.append(effective_row)
                effective[sid] = effective_row
                progress = True
                completed_new += 1
                continue
            diagnostic_state, diagnostic_terminal, diagnostic_outcome = (
                adopt_diagnostic(
                    workspace, target, partition=args.partition
                )
            )
            preferred_model = str(PARTITIONS[args.partition]["diagnostic_model"])
            if diagnostic_state == "ADOPTED":
                assert diagnostic_terminal is not None
                assert diagnostic_outcome is not None
                diag_root = (
                    workspace
                    / "artifacts"
                    / "frontier_ceiling_two_enrichments"
                    / "runs"
                    / str(PARTITIONS[args.partition]["diagnostic_dir"]).format(
                        arm=args.arm
                    )
                )
                diag_prov = read_json(diag_root / "provenance.json")
                diag_cfg = diag_prov.get("config")
                if not isinstance(diag_cfg, dict):
                    raise AuditError("diagnostic config missing")
                _publish_terminal_feed(
                    journal=terminal_feed_journal,
                    feed=terminal_feed,
                    target=target,
                    terminal=diagnostic_terminal,
                    origin="adopted_clean_diagnostic",
                    overlay_config_sha256=config_sha,
                    effective_source_directory=str(diag_root),
                    effective_config_sha256=str(
                        diag_prov.get("config_sha256") or ""
                    ),
                    effective_slot_policy_sha256=str(
                        diag_cfg.get("slot_policy_sha256") or ""
                    ),
                    effective_endpoint_sha256=str(
                        diag_cfg.get("api_base_url_sha256") or ""
                    ),
                )
                effective_row = _effective_record(
                    target=target,
                    terminal=diagnostic_terminal,
                    outcome=diagnostic_outcome,
                    origin="adopted_clean_diagnostic",
                    effective_source_directory=str(diag_root),
                    effective_config_sha256=str(
                        diag_prov.get("config_sha256") or ""
                    ),
                    effective_slot_policy_sha256=str(
                        diag_cfg.get("slot_policy_sha256") or ""
                    ),
                    effective_endpoint_sha256=str(
                        diag_cfg.get("api_base_url_sha256") or ""
                    ),
                )
                effective_journal.append(effective_row)
                effective[sid] = effective_row
                progress = True
                completed_new += 1
                continue
            if diagnostic_state == "TERMINAL_WAIT_OUTCOME":
                assert diagnostic_terminal is not None
                diag_root = (
                    workspace
                    / "artifacts"
                    / "frontier_ceiling_two_enrichments"
                    / "runs"
                    / str(PARTITIONS[args.partition]["diagnostic_dir"]).format(
                        arm=args.arm
                    )
                )
                diag_prov = read_json(diag_root / "provenance.json")
                diag_cfg = diag_prov.get("config")
                if not isinstance(diag_cfg, dict):
                    raise AuditError("diagnostic config missing")
                _publish_terminal_feed(
                    journal=terminal_feed_journal,
                    feed=terminal_feed,
                    target=target,
                    terminal=diagnostic_terminal,
                    origin="adopted_clean_diagnostic",
                    overlay_config_sha256=config_sha,
                    effective_source_directory=str(diag_root),
                    effective_config_sha256=str(
                        diag_prov.get("config_sha256") or ""
                    ),
                    effective_slot_policy_sha256=str(
                        diag_cfg.get("slot_policy_sha256") or ""
                    ),
                    effective_endpoint_sha256=str(
                        diag_cfg.get("api_base_url_sha256") or ""
                    ),
                )
                waiting = True
                continue
            if diagnostic_state == "WAIT":
                waiting = True
                continue
            if diagnostic_state == "INSPECTION":
                permanent_inspection.add((sid, preferred_model))
                if not any(
                    row.get("selection_id") == sid
                    and row.get("requested_model") == preferred_model
                    and row.get("route_result") == "data_inspection_failed"
                    for row in route_rows
                ):
                    route = {
                        "schema": SCHEMA,
                        "record_type": "capacity_route_attempt",
                        "config_sha256": config_sha,
                        "capacity_epoch": args.capacity_epoch,
                        "endpoint_sha256": endpoint_sha256,
                        "selection_id": sid,
                        "task_id": target["task_id"],
                        "global_sample_index": target[
                            "global_sample_index"
                        ],
                        "requested_model": preferred_model,
                        "response_received": False,
                        "route_result": "data_inspection_failed",
                        "prompt_transformed": False,
                        "retry_authorized": False,
                        "source": "clean_diagnostic_journal",
                        "recorded_at": runner.utc_now(),
                    }
                    route_journal.append(route)
                    route_rows.append(route)
            terminal_row: dict[str, Any] | None = None
            outcome_row: dict[str, Any] | None = None
            for alias_rank, model in enumerate(enabled_aliases):
                model = str(model)
                if (sid, model) in permanent_inspection or model in unavailable:
                    continue
                messages = _load_messages(source_root, target)
                attempt_id = (
                    f"{sid[:12]}.{args.capacity_epoch}."
                    f"{model.replace('.', '_')}.{uuid.uuid4().hex[:10]}"
                )
                dispatch_journal.append(
                    {
                        "schema": SCHEMA,
                        "record_type": "capacity_request_dispatch",
                        "config_sha256": config_sha,
                        "capacity_epoch": args.capacity_epoch,
                        "endpoint_sha256": endpoint_sha256,
                        "selection_id": sid,
                        "attempt_id": attempt_id,
                        "task_id": target["task_id"],
                        "global_sample_index": target[
                            "global_sample_index"
                        ],
                        "requested_model": model,
                        "alias_rank": alias_rank,
                        "prompt_sha256": target["prompt_sha256"],
                        "max_completion_tokens": CAP,
                        "thinking_budget": THINKING_BUDGET,
                        "dispatched_at": runner.utc_now(),
                    }
                )
                started = runner.utc_now()
                try:
                    if model == "qwen3.7-max":
                        ensure_undated_identity_probe(
                            client,
                            out=out,
                            capacity_epoch=args.capacity_epoch,
                            endpoint_sha256=endpoint_sha256,
                        )
                    response = make_v6_request(
                        client,
                        model=model,
                        messages=messages,
                    )
                except Exception as exc:
                    if data_inspection_failed(exc):
                        result = "data_inspection_failed"
                        permanent_inspection.add((sid, model))
                    elif quota_exception(exc):
                        result = "quota_exhausted_403"
                        unavailable.add(model)
                    else:
                        result = "transport_or_provider_error"
                    route = {
                        "schema": SCHEMA,
                        "record_type": "capacity_route_attempt",
                        "config_sha256": config_sha,
                        "capacity_epoch": args.capacity_epoch,
                        "endpoint_sha256": endpoint_sha256,
                        "selection_id": sid,
                        "attempt_id": attempt_id,
                        "task_id": target["task_id"],
                        "global_sample_index": target[
                            "global_sample_index"
                        ],
                        "requested_model": model,
                        "alias_rank": alias_rank,
                        "response_received": False,
                        "http_status": exception_status(exc),
                        "route_result": result,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:4000],
                        "prompt_transformed": False,
                        "retry_authorized": False,
                        "started_at": started,
                        "finished_at": runner.utc_now(),
                    }
                    route_journal.append(route)
                    route_rows.append(route)
                    continue
                classified = classify_v6_terminal_response(
                    response,
                    expected_model=model,
                )
                terminal_row = {
                    "schema": SCHEMA,
                    "record_type": "capacity_terminal_response",
                    "config_sha256": config_sha,
                    "capacity_epoch": args.capacity_epoch,
                    "endpoint_sha256": endpoint_sha256,
                    "selection_id": sid,
                    "attempt_id": attempt_id,
                    "task_id": target["task_id"],
                    "global_sample_index": target[
                        "global_sample_index"
                    ],
                    "prompt_sha256": target["prompt_sha256"],
                    "requested_model": model,
                    "resolved_model": classified.response_model,
                    "response_id": classified.response_id,
                    "response_created": classified.response_created,
                    "finish_reason": classified.finish_reason,
                    "candidate_valid": classified.candidate_valid,
                    "terminal_reason": classified.terminal_reason,
                    "content": classified.content,
                    "reasoning_content": classified.reasoning_content,
                    "code": classified.code,
                    "code_sha256": classified.code_sha256,
                    "usage": classified.usage,
                    "response_received": True,
                    "slot_terminal": True,
                    "response": classified.raw_response,
                    "started_at": started,
                    "finished_at": runner.utc_now(),
                }
                route_journal.append(terminal_row)
                direct_terminals[sid] = terminal_row
                _publish_terminal_feed(
                    journal=terminal_feed_journal,
                    feed=terminal_feed,
                    target=target,
                    terminal=terminal_row,
                    origin="fresh_capacity_alias_response",
                    overlay_config_sha256=config_sha,
                    effective_source_directory=str(out),
                    effective_config_sha256=config_sha,
                    effective_slot_policy_sha256=canonical_sha(
                        prepared["config"]["request_policy"]
                    ),
                    effective_endpoint_sha256=endpoint_sha256,
                )
                outcome_row = _make_outcome(
                    terminal=terminal_row,
                    target=target,
                    evaluator=evaluator,
                    evaluator_sha=evaluator_record["sha256"],
                    tests=tests,
                    selection_id_value=sid,
                )
                outcome_journal.append(outcome_row)
                overlay_outcomes[sid] = outcome_row
                effective_row = _effective_record(
                    target=target,
                    terminal=terminal_row,
                    outcome=outcome_row,
                    origin="fresh_capacity_alias_response",
                    effective_source_directory=str(out),
                    effective_config_sha256=config_sha,
                    effective_slot_policy_sha256=canonical_sha(
                        prepared["config"]["request_policy"]
                    ),
                    effective_endpoint_sha256=endpoint_sha256,
                )
                effective_journal.append(effective_row)
                effective[sid] = effective_row
                progress = True
                completed_new += 1
                break
            if terminal_row is None and diagnostic_state != "WAIT":
                waiting = True
        remaining = len(targets) - len(effective)
        exactly_exhausted = (
            remaining > 0
            and _all_enabled_aliases_exactly_unavailable(
                enabled_aliases,
                unavailable,
            )
        )
        status = {
            "schema": SCHEMA,
            "status": (
                "complete"
                if remaining == 0
                else (
                    "capacity_epoch_exact_quota_exhausted"
                    if exactly_exhausted
                    else "in_progress"
                )
            ),
            "arm": args.arm,
            "partition": args.partition,
            "capacity_epoch": args.capacity_epoch,
            "targets": len(targets),
            "effective": len(effective),
            "remaining": remaining,
            "unavailable_aliases_this_epoch": sorted(unavailable),
            "preview_routing_enabled": bool(args.allow_preview),
            "undated_alias_enabled": bool(args.include_undated_alias),
            "source_0517_alias_enabled": bool(args.include_source_alias),
            "undated_alias_only": bool(args.only_undated_alias),
            "enabled_aliases": enabled_aliases,
            "endpoint_sha256": endpoint_sha256,
            "terminal_feed_rows": len(terminal_feed),
            "length_overlay_pending_selections": sum(
                row.get("finish_reason") == "length"
                for row in terminal_feed.values()
            ),
            "updated_at": runner.utc_now(),
        }
        runner.atomic_write_json(out / "status.json", status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if remaining == 0 or (args.max_new and completed_new >= args.max_new):
            return 0
        if exactly_exhausted:
            return 75
        if progress:
            idle_cycles = 0
            continue
        idle_cycles += 1
        if not waiting or idle_cycles >= args.max_idle_cycles:
            return 75
        time.sleep(args.poll_seconds)


def status(args: argparse.Namespace) -> int:
    prepared = preflight(args)
    effective = _load_effective(args.out.resolve() / "effective_terminals.jsonl")
    report = {
        "schema": SCHEMA,
        "status": (
            "complete"
            if len(effective) == len(prepared["targets"])
            else "in_progress"
        ),
        "arm": args.arm,
        "partition": args.partition,
        "targets": len(prepared["targets"]),
        "effective": len(effective),
        "remaining": len(prepared["targets"]) - len(effective),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run", "status"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("opus", "codex"), required=True)
    parser.add_argument("--partition", choices=tuple(PARTITIONS), required=True)
    parser.add_argument("--capacity-epoch", required=True)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-idle-cycles", type=int, default=30)
    parser.add_argument("--allow-preview", action="store_true")
    parser.add_argument("--include-undated-alias", action="store_true")
    parser.add_argument("--include-source-alias", action="store_true")
    parser.add_argument("--only-undated-alias", action="store_true")
    parser.add_argument(
        "--credential-source",
        choices=("secondary_qwen_env", "generic_fallback_env"),
        default="secondary_qwen_env",
    )
    parser.add_argument(
        "--credential-env-file",
        type=Path,
        default=Path("/workspace/Qwen.env"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "preflight":
            prepared = preflight(args)
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "status": "preflight_ok",
                        "arm": args.arm,
                        "partition": args.partition,
                        "targets": len(prepared["targets"]),
                        "config_sha256": prepared["config_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "status":
            return status(args)
        return run_overlay(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
