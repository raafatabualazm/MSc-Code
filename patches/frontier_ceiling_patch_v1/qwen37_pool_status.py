#!/usr/bin/env python3
"""Read-only progress/status audit for the declared Qwen3.7 pooled frontier."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path("/workspace")
RUN_ROOT = (
    WORKSPACE / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
)
SHARDS = (
    ("qwen3.7-max-2026-05-17", "0517", 3),
    ("qwen3.7-max-preview", "preview", 2),
    ("qwen3.7-max-2026-05-20", "0520", 3),
    ("qwen3.7-max-2026-06-08", "0608", 2),
)
ARMS = ("opus", "codex")
EXPECTED_TASKS = 175


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


def main() -> None:
    report: dict[str, Any] = {
        "schema": "qwen37_frontier_pool_status_v1",
        "expected": {
            "tasks": EXPECTED_TASKS,
            "global_k_per_arm": sum(k for _model, _slug, k in SHARDS),
            "terminal_slots_per_arm": EXPECTED_TASKS
            * sum(k for _model, _slug, k in SHARDS),
        },
        "shards": [],
    }
    outcomes_by_arm_task: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = {arm: collections.defaultdict(list) for arm in ARMS}
    model_usage: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"terminal_responses": 0, "total_tokens": 0, "reasoning_tokens": 0}
    )

    for model, slug, local_k in SHARDS:
        for arm in ARMS:
            root = RUN_ROOT / f"qwen37_pool_{slug}_{arm}_k{local_k}_tb8k"
            attempts = list(read_jsonl(root / "attempts.jsonl"))
            outcomes = list(read_jsonl(root / "outcomes.jsonl"))
            terminal = [
                row
                for row in attempts
                if row.get("response_received") is True
                and row.get("slot_terminal") is True
            ]
            response_ids = [str(row.get("response_id") or "") for row in terminal]
            if len(response_ids) != len(set(response_ids)):
                raise RuntimeError(f"duplicate terminal response id in {root}")
            model_usage[model]["terminal_responses"] += len(terminal)
            model_usage[model]["total_tokens"] += sum(
                usage_total(row) for row in terminal
            )
            model_usage[model]["reasoning_tokens"] += sum(
                reasoning_tokens(row) for row in terminal
            )
            finish_reasons = collections.Counter(
                str(row.get("finish_reason")) for row in terminal
            )
            resolved_models = sorted(
                {str(row.get("resolved_model") or "") for row in terminal}
            )
            for row in outcomes:
                outcomes_by_arm_task[arm][str(row["task_id"])].append(row)
            task_candidate_counts = collections.Counter(
                str(row["task_id"]) for row in outcomes
            )
            complete_tasks = sum(
                count == local_k for count in task_candidate_counts.values()
            )
            report["shards"].append(
                {
                    "model": model,
                    "slug": slug,
                    "arm": arm,
                    "local_k": local_k,
                    "root": str(root),
                    "attempt_rows": len(attempts),
                    "terminal_responses": len(terminal),
                    "expected_terminal_responses": EXPECTED_TASKS * local_k,
                    "response_less_attempts": len(attempts) - len(terminal),
                    "complete_tasks": complete_tasks,
                    "candidate_outcomes": len(outcomes),
                    "valid_candidates": sum(
                        row.get("candidate_valid") is True for row in outcomes
                    ),
                    "compiled_candidates": sum(
                        row.get("compiled") is True for row in outcomes
                    ),
                    "passed_candidates": sum(
                        row.get("passed") is True for row in outcomes
                    ),
                    "finish_reasons": dict(sorted(finish_reasons.items())),
                    "resolved_models": resolved_models,
                    "total_tokens": sum(usage_total(row) for row in terminal),
                    "reasoning_tokens": sum(
                        reasoning_tokens(row) for row in terminal
                    ),
                    "failure_present": (root / "failure.json").is_file(),
                    "summary_present": (root / "summary.json").is_file(),
                }
            )

    pooled: dict[str, Any] = {}
    expected_global_k = sum(k for _model, _slug, k in SHARDS)
    for arm in ARMS:
        by_task = outcomes_by_arm_task[arm]
        complete = {
            task_id: rows
            for task_id, rows in by_task.items()
            if len(rows) == expected_global_k
        }
        overfull = {
            task_id: len(rows)
            for task_id, rows in by_task.items()
            if len(rows) > expected_global_k
        }
        if overfull:
            raise RuntimeError(f"overfull pooled tasks for {arm}: {overfull}")
        passed_tasks = sum(
            any(row.get("passed") is True for row in rows)
            for rows in complete.values()
        )
        compiled_tasks = sum(
            any(row.get("compiled") is True for row in rows)
            for rows in complete.values()
        )
        pooled[arm] = {
            "terminal_candidate_outcomes": sum(len(rows) for rows in by_task.values()),
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
    report["pooled"] = pooled
    report["model_usage"] = dict(model_usage)
    report["total_terminal_responses"] = sum(
        row["terminal_responses"] for row in report["shards"]
    )
    report["total_tokens"] = sum(
        values["total_tokens"] for values in model_usage.values()
    )
    report["total_reasoning_tokens"] = sum(
        values["reasoning_tokens"] for values in model_usage.values()
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
