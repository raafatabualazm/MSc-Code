#!/usr/bin/env python3
"""Fail-closed progress/final aggregator for the paired Qwen v5 primary pool."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import frontier_passk as runner
import frontier_passk_qwen_completion as qwen_entry


SCHEMA = "qwen37-frontier-primary-alias-status-v5"
EXPECTED_META_CONTRACT_SHA256 = (
    "6218118c8e9e7b67079df2b848626ea8a71deaf128353bbc8757d9553a6cdbae"
)
EXPECTED_V4_CONTRACT_SHA256 = (
    "5c183322702c6e6a171400882f3aeb768525301c5168dc1ea8de7315ce006be0"
)
EXPECTED_ENTRY_SHA256 = (
    "5055eabac3898d529beb6209b3792256378d509239265cb44eaa2cf7f46b5e15"
)
EXPECTED_RUNNER_SHA256 = (
    "8d3e3ad160d9ed389a9e212dacb76556ab7af59f1559418d45d9802402d9dead"
)
EXPECTED_CORE_SHA256 = (
    "f502e958a6fa3fb564d17327c2c4c77bc9cf4f5182546235970b1a4498a60258"
)
EXPECTED_F2_SHA256 = (
    "097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce"
)
EXPECTED_EVALUATOR_SHA256 = (
    "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
)
EXPECTED_PAIR_MANIFEST_SHA256 = (
    "35f4cfcaf0732928312bed3f2f27c3f3e347525c0076921caeab7ee6539c132e"
)
EXPECTED_TASKS = 175
EXPECTED_K = 10
ARMS = ("opus", "codex")
QUARANTINED_MODELS = frozenset(
    {"qwen3.7-max-2026-05-20", "qwen3.7-max-2026-06-08"}
)


class AuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Shard:
    key: str
    model: str
    local_k: int
    global_indices: tuple[int, ...]
    directory_template: str
    service_template: str
    copied_contract: str
    copied_contract_sha256: str


SHARDS = (
    Shard(
        key="base_0517_k3",
        model="qwen3.7-max-2026-05-17",
        local_k=3,
        global_indices=(0, 1, 2),
        directory_template="qwen37_clean_v4_0517_{arm}_k3_mc12k_tol10_tb8k",
        service_template=(
            "frontier-qwen37-clean-v4-0517-{arm}-k3-mc12k-tol10-tb8k.service"
        ),
        copied_contract="qwen37_primary_clean_contract_v4.json",
        copied_contract_sha256=EXPECTED_V4_CONTRACT_SHA256,
    ),
    Shard(
        key="base_preview_k2",
        model="qwen3.7-max-preview",
        local_k=2,
        global_indices=(3, 4),
        directory_template=(
            "qwen37_clean_v4_preview_{arm}_k2_mc12k_tol10_tb8k"
        ),
        service_template=(
            "frontier-qwen37-clean-v4-preview-{arm}-k2-mc12k-tol10-tb8k.service"
        ),
        copied_contract="qwen37_primary_clean_contract_v4.json",
        copied_contract_sha256=EXPECTED_V4_CONTRACT_SHA256,
    ),
    Shard(
        key="supplement_0517_k2",
        model="qwen3.7-max-2026-05-17",
        local_k=2,
        global_indices=(5, 6),
        directory_template=(
            "qwen37_clean_v5_supplement_0517_{arm}_k2_mc12k_tol10_tb8k"
        ),
        service_template=(
            "frontier-qwen37-clean-v5-supplement-0517-{arm}-k2-"
            "mc12k-tol10-tb8k.service"
        ),
        copied_contract="qwen37_primary_alias_fallback_contract_v5.json",
        copied_contract_sha256=EXPECTED_META_CONTRACT_SHA256,
    ),
    Shard(
        key="supplement_preview_k3",
        model="qwen3.7-max-preview",
        local_k=3,
        global_indices=(7, 8, 9),
        directory_template=(
            "qwen37_clean_v5_supplement_preview_{arm}_k3_mc12k_tol10_tb8k"
        ),
        service_template=(
            "frontier-qwen37-clean-v5-supplement-preview-{arm}-k3-"
            "mc12k-tol10-tb8k.service"
        ),
        copied_contract="qwen37_primary_alias_fallback_contract_v5.json",
        copied_contract_sha256=EXPECTED_META_CONTRACT_SHA256,
    ),
)

ARM_EXPECTATIONS = {
    "opus": {
        "dataset_label": "opus_real_fn0_cfg_175",
        "pair_arm_key": "opus_real_fn0_cfg",
        "prompt_jsonl_sha256": (
            "4aae71997aa98b4a273fdedca17d1df2266f18dd5a03fe164b9cf81e342648cd"
        ),
        "prompt_manifest_sha256": (
            "35e25fa9d7a2bd813b6aec55a1149304d4dd160c82b27b691f27c4cb0bd6068b"
        ),
        "eval_jsonl_sha256": (
            "a4ed1cf185d52c3d212e2d7348fdb2a1dffd0035f4c395e2e897fd072fa70001"
        ),
        "eval_seal_sha256": (
            "2909d279d7c87279b5b0e59cdcd7598742b25a2bd111382f6c8216103f906799"
        ),
    },
    "codex": {
        "dataset_label": "codex_multifunction_cfg_175",
        "pair_arm_key": "codex_multifunction_cfg",
        "prompt_jsonl_sha256": (
            "6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab"
        ),
        "prompt_manifest_sha256": (
            "777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44"
        ),
        "eval_jsonl_sha256": (
            "abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7"
        ),
        "eval_seal_sha256": (
            "5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a"
        ),
    },
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON artifact is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def validate_usage(usage: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuditError(f"invalid normalized usage.{key}")
    completion = int(usage["completion_tokens"])
    reasoning = usage.get("reasoning_tokens")
    answer = usage.get("answer_tokens")
    if (
        isinstance(reasoning, bool)
        or not isinstance(reasoning, int)
        or reasoning <= 0
        or reasoning > min(completion, 8_192)
    ):
        raise AuditError("normalized reasoning_tokens is outside the sealed bound")
    if (
        isinstance(answer, bool)
        or not isinstance(answer, int)
        or answer != completion - reasoning
    ):
        raise AuditError("normalized answer_tokens is not completion-reasoning")
    if completion > 12_298:
        raise AuditError("completion usage exceeds the sealed 12,298 cap")
    if usage["prompt_tokens"] > 12_000:
        raise AuditError("prompt usage exceeds the sealed 12,000 cap")
    if usage["total_tokens"] != usage["prompt_tokens"] + completion:
        raise AuditError("normalized total token usage is inconsistent")


def service_state(service: str) -> dict[str, str | None]:
    completed = subprocess.run(
        (
            "systemctl",
            "show",
            service,
            "--property=ActiveState",
            "--property=SubState",
            "--property=Result",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {
        "active_state": values.get("ActiveState"),
        "sub_state": values.get("SubState"),
        "result": values.get("Result"),
    }


def validate_meta_contract(patch_root: Path) -> dict[str, Any]:
    path = patch_root / "qwen37_primary_alias_fallback_contract_v5.json"
    if sha256_file(path) != EXPECTED_META_CONTRACT_SHA256:
        raise AuditError("v5 meta-contract hash mismatch")
    contract = read_json(path)
    allocation = contract.get("primary_global_sample_allocation")
    expected = [
        {
            "model": shard.model,
            "local_k": shard.local_k,
            "global_sample_indices": list(shard.global_indices),
            "run_directory_pattern": shard.directory_template,
        }
        for shard in SHARDS
    ]
    if not isinstance(allocation, list) or len(allocation) != len(expected):
        raise AuditError("v5 meta-contract allocation length mismatch")
    for row, expected_row in zip(allocation, expected):
        if not isinstance(row, dict):
            raise AuditError("v5 meta-contract allocation row is malformed")
        for key, value in expected_row.items():
            if row.get(key) != value:
                raise AuditError(
                    f"v5 meta-contract allocation mismatch for {key}"
                )
    indices = [
        index for shard in SHARDS for index in shard.global_indices
    ]
    if sorted(indices) != list(range(EXPECTED_K)) or len(set(indices)) != EXPECTED_K:
        raise AuditError("code-level global-index mapping is invalid")
    if contract.get("primary_global_k_per_arm") != EXPECTED_K:
        raise AuditError("v5 meta-contract K mismatch")
    return contract


def validate_config_and_provenance(
    provenance: dict[str, Any],
    *,
    shard: Shard,
    arm: str,
) -> tuple[str, str]:
    config = provenance.get("config")
    if not isinstance(config, dict):
        raise AuditError("provenance config is missing")
    config_sha = provenance.get("config_sha256")
    if config_sha != runner.stable_sha256(config):
        raise AuditError("provenance config fingerprint mismatch")
    expected_arm = ARM_EXPECTATIONS[arm]
    exact_values = {
        "schema": runner.RUN_SCHEMA_VERSION,
        "provider": "qwen",
        "model_requested": shard.model,
        "arm": "compact",
        "input_mode": "prematerialized_f2",
        "pair_arm_key": expected_arm["pair_arm_key"],
        "k": shard.local_k,
        "workers": 1,
        "limit": 0,
        "max_output_tokens": 12_288,
        "max_prompt_tokens": 12_000,
        "budget": 0,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_attempts_per_sample": 6,
        "eval_stability_runs": 2,
        "dataset_label": expected_arm["dataset_label"],
        "expected_task_count": EXPECTED_TASKS,
        "extra_body": {
            "enable_thinking": True,
            "thinking_budget": 8_192,
        },
    }
    for key, value in exact_values.items():
        if config.get(key) != value:
            raise AuditError(f"config field {key!r} disagrees with v5 mapping")
    runtime = config.get("runtime_identity")
    expected_runtime = {
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "core_sha256": EXPECTED_CORE_SHA256,
        "frontier_f2_sha256": EXPECTED_F2_SHA256,
        "qwen_completion_entry_sha256": EXPECTED_ENTRY_SHA256,
    }
    if not isinstance(runtime, dict):
        raise AuditError("config runtime identity is missing")
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            raise AuditError(f"runtime identity mismatch for {key}")
    request_contract = config.get("qwen_request_contract")
    if not isinstance(request_contract, dict):
        raise AuditError("Qwen request contract is missing")
    request_expected = {
        "schema": qwen_entry.REQUEST_CONTRACT_SCHEMA,
        "request_cap_parameter": "max_completion_tokens",
        "forbidden_request_cap_parameter": "max_tokens",
        "total_completion_cap": 12_288,
        "provider_completion_tolerance": 10,
        "completion_usage_validation_cap": 12_298,
        "thinking_budget": 8_192,
        "finite_runner_budget_forbidden": True,
        "reasoning_content_required_nonempty": True,
        "reasoning_tokens_usage_required_positive_and_bounded": True,
        "exact_extra_body_keys": ["enable_thinking", "thinking_budget"],
    }
    for key, value in request_expected.items():
        if request_contract.get(key) != value:
            raise AuditError(f"Qwen request contract mismatch for {key}")
    slot_policy = config.get("slot_policy")
    if not isinstance(slot_policy, dict):
        raise AuditError("slot policy is missing")
    if config.get("slot_policy_sha256") != runner.stable_sha256(slot_policy):
        raise AuditError("slot-policy fingerprint mismatch")
    slot_expected = {
        "request_cap_parameter": "max_completion_tokens",
        "max_tokens_absent": True,
        "total_completion_cap": 12_288,
        "provider_completion_tolerance": 10,
        "completion_usage_validation_cap": 12_298,
        "reasoning_content_required_nonempty": True,
        "reasoning_tokens_usage_required_positive_and_bounded": True,
        "finite_runner_budget_forbidden": True,
        "requested_model": shard.model,
        "k": shard.local_k,
    }
    for key, value in slot_expected.items():
        if slot_policy.get(key) != value:
            raise AuditError(f"slot-policy mismatch for {key}")
    sealed = config.get("sealed_inputs")
    if not isinstance(sealed, dict):
        raise AuditError("sealed input config is missing")
    for key in (
        "prompt_jsonl_sha256",
        "prompt_manifest_sha256",
        "eval_jsonl_sha256",
        "eval_seal_sha256",
    ):
        if sealed.get(key) != expected_arm[key]:
            raise AuditError(f"sealed input hash mismatch for {key}")
    if sealed.get("pair_manifest_sha256") != EXPECTED_PAIR_MANIFEST_SHA256:
        raise AuditError("pair manifest hash mismatch")
    entry_record = provenance.get("qwen_completion_entry")
    if (
        not isinstance(entry_record, dict)
        or entry_record.get("sha256") != EXPECTED_ENTRY_SHA256
    ):
        raise AuditError("top-level Qwen entry provenance mismatch")
    runner_record = provenance.get("runner")
    core_record = provenance.get("core")
    evaluator = provenance.get("evaluator")
    if not isinstance(runner_record, dict) or runner_record.get(
        "sha256"
    ) != EXPECTED_RUNNER_SHA256:
        raise AuditError("runner provenance mismatch")
    if not isinstance(core_record, dict) or core_record.get(
        "sha256"
    ) != EXPECTED_CORE_SHA256:
        raise AuditError("core provenance mismatch")
    if not isinstance(evaluator, dict) or evaluator.get(
        "sha256"
    ) != EXPECTED_EVALUATOR_SHA256:
        raise AuditError("evaluator provenance mismatch")
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict):
        raise AuditError("artifact provenance is missing")
    artifact_expected = {
        "prompt_jsonl": expected_arm["prompt_jsonl_sha256"],
        "prompt_manifest": expected_arm["prompt_manifest_sha256"],
        "eval_jsonl": expected_arm["eval_jsonl_sha256"],
        "eval_seal": expected_arm["eval_seal_sha256"],
        "pair_manifest": EXPECTED_PAIR_MANIFEST_SHA256,
        "frontier_f2": EXPECTED_F2_SHA256,
    }
    for key, value in artifact_expected.items():
        record = artifacts.get(key)
        if not isinstance(record, dict) or record.get("sha256") != value:
            raise AuditError(f"artifact provenance mismatch for {key}")
    endpoint_hash = config.get("api_base_url_sha256")
    if not isinstance(endpoint_hash, str) or len(endpoint_hash) != 64:
        raise AuditError("API endpoint fingerprint is malformed")
    return str(config_sha), endpoint_hash


def validate_final_files(
    root: Path,
    provenance: dict[str, Any],
    *,
    shard: Shard,
) -> None:
    summary = read_json(root / "summary.json")
    if provenance.get("status") != "complete":
        raise AuditError("final shard provenance is not complete")
    if provenance.get("summary_sha256") != sha256_file(root / "summary.json"):
        raise AuditError("summary hash is not bound by provenance")
    if (
        summary.get("status") != "complete"
        or summary.get("requested_model") != shard.model
        or summary.get("resolved_models") != [shard.model]
        or summary.get("k") != shard.local_k
        or summary.get("tasks") != EXPECTED_TASKS
        or summary.get("terminal_responses") != EXPECTED_TASKS * shard.local_k
    ):
        raise AuditError("final shard summary contract mismatch")
    manifest = read_json(root / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise AuditError("final manifest has no file map")
    for name in (
        "provenance.json",
        "tasks.jsonl",
        "prompts.jsonl",
        "attempts.jsonl",
        "outcomes.jsonl",
        "summary.json",
    ):
        record = files.get(name)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != sha256_file(root / name)
        ):
            raise AuditError(f"final manifest hash mismatch for {name}")


def metrics_if_complete(
    outcomes: Mapping[tuple[str, str, int], Mapping[str, Any]],
    task_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    expected_keys = {
        (arm, task_id, sample_index)
        for arm in ARMS
        for task_id in task_ids
        for sample_index in range(EXPECTED_K)
    }
    if set(outcomes) != expected_keys:
        if len(outcomes) > len(expected_keys) or not set(outcomes).issubset(
            expected_keys
        ):
            raise AuditError("primary global outcome slots are foreign or overfull")
        return None
    pooled: dict[str, Any] = {}
    for arm in ARMS:
        passed = 0
        compiled = 0
        for task_id in task_ids:
            rows = [
                outcomes[(arm, task_id, sample_index)]
                for sample_index in range(EXPECTED_K)
            ]
            passed += any(row.get("passed") is True for row in rows)
            compiled += any(row.get("compiled") is True for row in rows)
        pooled[arm] = {
            "pass_at_10": {
                "successes": passed,
                "total": EXPECTED_TASKS,
                "rate": passed / EXPECTED_TASKS,
            },
            "compile_at_10": {
                "successes": compiled,
                "total": EXPECTED_TASKS,
                "rate": compiled / EXPECTED_TASKS,
            },
        }
    return pooled


def aggregate(workspace: Path) -> dict[str, Any]:
    qwen_entry.install_qwen_completion_policy()
    patch_root = workspace / "frontier_ceiling_patch_v1"
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    validate_meta_contract(patch_root)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "meta_contract_sha256": EXPECTED_META_CONTRACT_SHA256,
        "expected": {
            "tasks_per_arm": EXPECTED_TASKS,
            "global_k_per_arm": EXPECTED_K,
            "terminal_outcomes_per_arm": EXPECTED_TASKS * EXPECTED_K,
            "terminal_outcomes_total": len(ARMS) * EXPECTED_TASKS * EXPECTED_K,
        },
        "shards": [],
    }
    canonical_task_ids: tuple[str, ...] | None = None
    endpoint_hash: str | None = None
    global_response_ids: set[str] = set()
    global_terminals: dict[tuple[str, str, int], dict[str, Any]] = {}
    global_outcomes: dict[tuple[str, str, int], dict[str, Any]] = {}
    all_final_files_present = True

    for shard in SHARDS:
        if shard.model in QUARANTINED_MODELS:
            raise AuditError("quarantined model appears in primary shard mapping")
        for arm in ARMS:
            root = run_root / shard.directory_template.format(arm=arm)
            if sha256_file(root / shard.copied_contract) != (
                shard.copied_contract_sha256
            ):
                raise AuditError(f"copied contract mismatch in {root}")
            if (root / "failure.json").is_file():
                raise AuditError(f"primary shard has a failure record: {root}")
            provenance = read_json(root / "provenance.json")
            config_sha, observed_endpoint_hash = validate_config_and_provenance(
                provenance,
                shard=shard,
                arm=arm,
            )
            if endpoint_hash is None:
                endpoint_hash = observed_endpoint_hash
            elif endpoint_hash != observed_endpoint_hash:
                raise AuditError("primary shards use different API endpoints")
            tasks = read_jsonl(root / "tasks.jsonl")
            prompts = read_jsonl(root / "prompts.jsonl")
            if len(tasks) != EXPECTED_TASKS or len(prompts) != EXPECTED_TASKS:
                raise AuditError(f"sealed task/prompt row count mismatch in {root}")
            task_ids = tuple(str(row.get("task_id") or "") for row in tasks)
            prompt_ids = tuple(str(row.get("task_id") or "") for row in prompts)
            if (
                not all(task_ids)
                or len(set(task_ids)) != EXPECTED_TASKS
                or prompt_ids != task_ids
            ):
                raise AuditError(f"task/prompt identity mismatch in {root}")
            if canonical_task_ids is None:
                canonical_task_ids = task_ids
            elif task_ids != canonical_task_ids:
                raise AuditError("primary shards have different ordered task IDs")
            prompt_map = {str(row["task_id"]): row for row in prompts}
            budget = runner.TokenBudget(0)
            terminal, _next_attempt = runner.load_resume_attempts(
                root / "attempts.jsonl",
                config_sha=config_sha,
                prompt_map=prompt_map,
                budget=budget,
                requested_model=shard.model,
                k=shard.local_k,
                max_prompt_tokens=12_000,
                requested_max_tokens=12_288,
                max_transport_attempts_per_slot=6,
                slot_policy_sha256=str(
                    provenance["config"]["slot_policy_sha256"]
                ),
            )
            outcomes = runner.load_resume_outcomes(
                root / "outcomes.jsonl",
                config_sha=config_sha,
                evaluator_sha256=EXPECTED_EVALUATOR_SHA256,
            )
            attempts = read_jsonl(root / "attempts.jsonl")
            for row in attempts:
                if (
                    row.get("provider") != "qwen"
                    or row.get("requested_model") != shard.model
                    or row.get("requested_model") in QUARANTINED_MODELS
                    or row.get("resolved_model") in QUARANTINED_MODELS
                ):
                    raise AuditError("foreign/quarantined model in primary attempts")
            for (task_id, local_index), row in terminal.items():
                if task_id not in prompt_map:
                    raise AuditError("foreign task in terminal map")
                response_id = str(row.get("response_id") or "")
                if not response_id or response_id in global_response_ids:
                    raise AuditError("duplicate/missing response ID across primary")
                global_response_ids.add(response_id)
                usage = row.get("usage")
                if not isinstance(usage, Mapping):
                    raise AuditError("terminal row has no normalized usage")
                validate_usage(usage)
                if not str(row.get("reasoning_content") or "").strip():
                    raise AuditError("terminal row has empty reasoning content")
                global_index = shard.global_indices[local_index]
                global_key = (arm, task_id, global_index)
                if global_key in global_terminals:
                    raise AuditError("duplicate primary global terminal slot")
                global_terminals[global_key] = row
            for (task_id, local_index, attempt_id), row in outcomes.items():
                terminal_row = terminal.get((task_id, local_index))
                if terminal_row is None:
                    raise AuditError("orphan primary outcome")
                if (
                    terminal_row.get("attempt_id") != attempt_id
                    or terminal_row.get("response_id") != row.get("response_id")
                    or terminal_row.get("code_sha256") != row.get("code_sha256")
                    or terminal_row.get("finish_reason") != row.get("finish_reason")
                    or terminal_row.get("candidate_valid")
                    != row.get("candidate_valid")
                    or terminal_row.get("terminal_reason")
                    != row.get("terminal_reason")
                ):
                    raise AuditError("outcome does not match its terminal receipt")
                global_index = shard.global_indices[local_index]
                global_key = (arm, task_id, global_index)
                if global_key in global_outcomes:
                    raise AuditError("duplicate primary global outcome slot")
                global_outcomes[global_key] = row
            final_files = all(
                (root / name).is_file()
                for name in ("summary.json", "manifest.json")
            )
            if final_files:
                validate_final_files(root, provenance, shard=shard)
            else:
                all_final_files_present = False
            report["shards"].append(
                {
                    "key": shard.key,
                    "arm": arm,
                    "model": shard.model,
                    "local_k": shard.local_k,
                    "global_sample_indices": list(shard.global_indices),
                    "root": str(root),
                    "service": shard.service_template.format(arm=arm),
                    "service_state": service_state(
                        shard.service_template.format(arm=arm)
                    ),
                    "config_sha256": config_sha,
                    "attempt_rows": len(attempts),
                    "terminal_responses": len(terminal),
                    "candidate_outcomes": len(outcomes),
                    "expected_terminal_responses": (
                        EXPECTED_TASKS * shard.local_k
                    ),
                    "final_manifest_present": final_files,
                }
            )

    if canonical_task_ids is None:
        raise AuditError("no primary task sequence was loaded")
    expected_global_keys = {
        (arm, task_id, sample_index)
        for arm in ARMS
        for task_id in canonical_task_ids
        for sample_index in range(EXPECTED_K)
    }
    if not set(global_terminals).issubset(expected_global_keys):
        raise AuditError("foreign primary global terminal slot")
    if not set(global_outcomes).issubset(set(global_terminals)):
        raise AuditError("primary global outcomes are not terminal-backed")
    progress_by_arm: dict[str, Any] = {}
    for arm in ARMS:
        terminal_count = sum(key[0] == arm for key in global_terminals)
        outcome_count = sum(key[0] == arm for key in global_outcomes)
        progress_by_arm[arm] = {
            "terminal_responses": terminal_count,
            "candidate_outcomes": outcome_count,
            "expected": EXPECTED_TASKS * EXPECTED_K,
            "remaining_outcomes": EXPECTED_TASKS * EXPECTED_K - outcome_count,
        }
    report["progress"] = progress_by_arm
    pooled = metrics_if_complete(global_outcomes, canonical_task_ids)
    if pooled is None or set(global_terminals) != expected_global_keys:
        report["status"] = "in_progress"
        report["metrics_withheld_until_complete"] = True
        return report
    if not all_final_files_present:
        report["status"] = "awaiting_final_manifests"
        report["metrics_withheld_until_complete"] = True
        return report
    report["status"] = "complete"
    report["metrics_withheld_until_complete"] = False
    report["pooled_metrics"] = pooled
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = aggregate(args.workspace.expanduser().resolve())
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
            )
        )
        return 2
    print(
        json.dumps(
            report,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
