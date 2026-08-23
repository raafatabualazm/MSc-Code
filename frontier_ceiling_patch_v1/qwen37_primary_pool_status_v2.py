#!/usr/bin/env python3
"""Read-only status audit for the Qwen3.7 primary-v2 K=10 pool."""

from __future__ import annotations

import collections
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path("/workspace")
PATCH_ROOT = WORKSPACE / "frontier_ceiling_patch_v1"
RUN_ROOT = (
    WORKSPACE / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
)
CONTRACT_PATH = PATCH_ROOT / "qwen37_primary_pool_contract_v2.json"
EXPECTED_CONTRACT_SHA256 = (
    "b36835319a942461f9b38cf893e9dc3fb5fd68b24018cdf347507f08aeb08b47"
)
EXPECTED_TASKS = 175
ARMS = ("opus", "codex")

PRIMARY_SHARDS = (
    {
        "wave": 1,
        "model": "qwen3.7-max-2026-05-17",
        "slug": "0517",
        "local_k": 3,
        "global_sample_indices": (0, 1, 2),
        "run_template": "qwen37_pool_0517_{arm}_k3_tb8k",
        "service_template": "frontier-qwen37-0517-{arm}-k3-tb8k.service",
        "copied_contract": "qwen37_pooled_contract.json",
        "copied_contract_sha256": (
            "68ec0e3f2f84b92ebbfd5169636914822fb06c422946e85aa68c5e5adbb6925b"
        ),
    },
    {
        "wave": 1,
        "model": "qwen3.7-max-preview",
        "slug": "preview",
        "local_k": 2,
        "global_sample_indices": (3, 4),
        "run_template": "qwen37_pool_preview_{arm}_k2_tb8k",
        "service_template": "frontier-qwen37-preview-{arm}-k2-tb8k.service",
        "copied_contract": "qwen37_pooled_contract.json",
        "copied_contract_sha256": (
            "68ec0e3f2f84b92ebbfd5169636914822fb06c422946e85aa68c5e5adbb6925b"
        ),
    },
    {
        "wave": 2,
        "model": "qwen3.7-max-2026-05-17",
        "slug": "0517",
        "local_k": 2,
        "global_sample_indices": (5, 6),
        "run_template": "qwen37_primary_v2_w2_0517_{arm}_k2_tb8k",
        "service_template": (
            "frontier-qwen37-primary-v2-w2-0517-{arm}-k2-tb8k.service"
        ),
        "copied_contract": "qwen37_primary_pool_contract_v2.json",
        "copied_contract_sha256": EXPECTED_CONTRACT_SHA256,
    },
    {
        "wave": 2,
        "model": "qwen3.7-max-preview",
        "slug": "preview",
        "local_k": 3,
        "global_sample_indices": (7, 8, 9),
        "run_template": "qwen37_primary_v2_w2_preview_{arm}_k3_tb8k",
        "service_template": (
            "frontier-qwen37-primary-v2-w2-preview-{arm}-k3-tb8k.service"
        ),
        "copied_contract": "qwen37_primary_pool_contract_v2.json",
        "copied_contract_sha256": EXPECTED_CONTRACT_SHA256,
    },
)

DIAGNOSTIC_SHARDS = (
    {
        "model": "qwen3.7-max-2026-05-20",
        "slug": "0520",
        "local_k": 3,
        "run_template": "qwen37_pool_0520_{arm}_k3_tb8k",
        "service_template": "frontier-qwen37-0520-{arm}-k3-tb8k.service",
    },
    {
        "model": "qwen3.7-max-2026-06-08",
        "slug": "0608",
        "local_k": 2,
        "run_template": "qwen37_pool_0608_{arm}_k2_tb8k",
        "service_template": "frontier-qwen37-0608-{arm}-k2-tb8k.service",
    },
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return ()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def usage_total(row: dict[str, Any]) -> int:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get("total_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def reasoning_tokens(row: dict[str, Any]) -> int:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return 0
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        response = row.get("response")
        if isinstance(response, dict):
            raw_usage = response.get("usage")
            if isinstance(raw_usage, dict):
                details = raw_usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return 0
    value = details.get("reasoning_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def service_state(service: str) -> dict[str, str | None]:
    command = (
        "systemctl",
        "show",
        service,
        "--property=ActiveState",
        "--property=SubState",
        "--property=Result",
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "active_state": None,
            "sub_state": None,
            "result": None,
            "status_error": type(exc).__name__,
        }
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {
        "active_state": values.get("ActiveState"),
        "sub_state": values.get("SubState"),
        "result": values.get("Result"),
        "status_error": (
            None
            if completed.returncode == 0
            else f"systemctl_exit_{completed.returncode}"
        ),
    }


def inspect_shard(
    shard: dict[str, Any],
    arm: str,
    *,
    primary: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    root = RUN_ROOT / str(shard["run_template"]).format(arm=arm)
    attempts = list(read_jsonl(root / "attempts.jsonl"))
    outcomes = list(read_jsonl(root / "outcomes.jsonl"))
    terminal = [
        row
        for row in attempts
        if row.get("response_received") is True
        and row.get("slot_terminal") is True
    ]
    response_ids = [
        str(row.get("response_id"))
        for row in terminal
        if isinstance(row.get("response_id"), str) and row.get("response_id")
    ]
    terminal_without_response_id = len(terminal) - len(response_ids)
    if len(response_ids) != len(set(response_ids)):
        raise RuntimeError(f"duplicate terminal response id in {root}")
    expected_model = str(shard["model"])
    wrong_requested_models = sorted(
        {
            str(row.get("requested_model") or "")
            for row in terminal
            if row.get("requested_model") != expected_model
        }
    )
    if wrong_requested_models:
        raise RuntimeError(
            f"requested-model mismatch in {root}: {wrong_requested_models}"
        )
    task_candidate_counts = collections.Counter(
        str(row["task_id"]) for row in outcomes
    )
    local_k = int(shard["local_k"])
    overfull = {
        task_id: count
        for task_id, count in task_candidate_counts.items()
        if count > local_k
    }
    if overfull:
        raise RuntimeError(f"overfull shard tasks in {root}: {overfull}")

    copied_contract = None
    copied_contract_expected = None
    copied_contract_matches = None
    if primary:
        copied_contract = sha256_file(root / str(shard["copied_contract"]))
        copied_contract_expected = str(shard["copied_contract_sha256"])
        copied_contract_matches = copied_contract == copied_contract_expected

    service = str(shard["service_template"]).format(arm=arm)
    report = {
        "role": "primary" if primary else "diagnostic_quarantined",
        "excluded_from_primary": not primary,
        "wave": shard.get("wave"),
        "model": expected_model,
        "slug": shard["slug"],
        "arm": arm,
        "local_k": local_k,
        "global_sample_indices": list(shard.get("global_sample_indices", ())),
        "root": str(root),
        "service": service,
        "service_state": service_state(service),
        "attempt_rows": len(attempts),
        "terminal_responses": len(terminal),
        "terminal_responses_with_ids": len(response_ids),
        "terminal_responses_without_ids": terminal_without_response_id,
        "fatal_response_contracts": sum(
            row.get("fatal_response_contract") is True for row in terminal
        ),
        "expected_terminal_responses": EXPECTED_TASKS * local_k,
        "response_less_attempts": len(attempts) - len(terminal),
        "complete_tasks": sum(
            count == local_k for count in task_candidate_counts.values()
        ),
        "candidate_outcomes": len(outcomes),
        "valid_candidates": sum(
            row.get("candidate_valid") is True for row in outcomes
        ),
        "compiled_candidates": sum(
            row.get("compiled") is True for row in outcomes
        ),
        "passed_candidates": sum(row.get("passed") is True for row in outcomes),
        "finish_reasons": dict(
            sorted(
                collections.Counter(
                    str(row.get("finish_reason")) for row in terminal
                ).items()
            )
        ),
        "resolved_models": sorted(
            {str(row.get("resolved_model") or "") for row in terminal}
        ),
        "total_tokens": sum(usage_total(row) for row in terminal),
        "reasoning_tokens": sum(reasoning_tokens(row) for row in terminal),
        "failure_present": (root / "failure.json").is_file(),
        "summary_present": (root / "summary.json").is_file(),
        "copied_contract_sha256": copied_contract,
        "expected_copied_contract_sha256": copied_contract_expected,
        "copied_contract_matches": copied_contract_matches,
    }
    return report, outcomes, response_ids


def main() -> None:
    contract_sha256 = sha256_file(CONTRACT_PATH)
    if contract_sha256 != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(
            "primary-v2 contract hash mismatch: "
            f"expected={EXPECTED_CONTRACT_SHA256} actual={contract_sha256}"
        )
    global_indices = [
        index
        for shard in PRIMARY_SHARDS
        for index in shard["global_sample_indices"]
    ]
    if sorted(global_indices) != list(range(10)) or len(set(global_indices)) != 10:
        raise RuntimeError(f"invalid primary global sample map: {global_indices}")

    report: dict[str, Any] = {
        "schema": "qwen37_frontier_primary_pool_status_v2",
        "contract": str(CONTRACT_PATH),
        "contract_sha256": contract_sha256,
        "expected": {
            "tasks": EXPECTED_TASKS,
            "global_k_per_arm": 10,
            "terminal_slots_per_arm": EXPECTED_TASKS * 10,
        },
        "primary_shards": [],
        "diagnostic_quarantine": [],
    }
    outcomes_by_arm_task_shard: dict[
        str, dict[str, dict[str, list[dict[str, Any]]]]
    ] = {
        arm: collections.defaultdict(lambda: collections.defaultdict(list))
        for arm in ARMS
    }
    primary_response_ids: list[str] = []

    for shard_index, shard in enumerate(PRIMARY_SHARDS):
        shard_key = f"w{shard['wave']}_{shard['slug']}_k{shard['local_k']}"
        for arm in ARMS:
            shard_report, outcomes, response_ids = inspect_shard(
                shard, arm, primary=True
            )
            shard_report["shard_key"] = shard_key
            shard_report["shard_index"] = shard_index
            report["primary_shards"].append(shard_report)
            primary_response_ids.extend(response_ids)
            for row in outcomes:
                outcomes_by_arm_task_shard[arm][str(row["task_id"])][
                    shard_key
                ].append(row)

    if len(primary_response_ids) != len(set(primary_response_ids)):
        raise RuntimeError("duplicate terminal response id across primary shards")

    for shard in DIAGNOSTIC_SHARDS:
        for arm in ARMS:
            shard_report, _outcomes, _response_ids = inspect_shard(
                shard, arm, primary=False
            )
            report["diagnostic_quarantine"].append(shard_report)

    pooled: dict[str, Any] = {}
    expected_by_shard = {
        f"w{shard['wave']}_{shard['slug']}_k{shard['local_k']}": int(
            shard["local_k"]
        )
        for shard in PRIMARY_SHARDS
    }
    for arm in ARMS:
        by_task = outcomes_by_arm_task_shard[arm]
        complete: dict[str, list[dict[str, Any]]] = {}
        for task_id, by_shard in by_task.items():
            if all(
                len(by_shard.get(shard_key, ())) == expected_k
                for shard_key, expected_k in expected_by_shard.items()
            ):
                complete[task_id] = [
                    row
                    for shard_key in expected_by_shard
                    for row in by_shard[shard_key]
                ]
        passed_tasks = sum(
            any(row.get("passed") is True for row in rows)
            for rows in complete.values()
        )
        compiled_tasks = sum(
            any(row.get("compiled") is True for row in rows)
            for rows in complete.values()
        )
        pooled[arm] = {
            "terminal_candidate_outcomes": sum(
                len(rows)
                for by_shard in by_task.values()
                for rows in by_shard.values()
            ),
            "fully_pooled_tasks": len(complete),
            "expected_tasks": EXPECTED_TASKS,
            "passed_tasks": passed_tasks,
            "compiled_tasks": compiled_tasks,
            "provisional_pass_at_10": (
                passed_tasks / len(complete) if complete else None
            ),
            "provisional_compile_at_10": (
                compiled_tasks / len(complete) if complete else None
            ),
        }
    report["primary_pooled"] = pooled
    report["primary_terminal_responses"] = len(primary_response_ids)
    report["primary_total_tokens"] = sum(
        int(shard["total_tokens"]) for shard in report["primary_shards"]
    )
    report["primary_total_reasoning_tokens"] = sum(
        int(shard["reasoning_tokens"]) for shard in report["primary_shards"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
