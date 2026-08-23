#!/usr/bin/env python3
"""Derive a hash-bound pass@k sensitivity report after named task exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)


SCHEMA = "direct-compact-passk-exclusion-sensitivity-v1"


def derive(score: dict[str, Any], excluded_task_ids: Sequence[str]) -> dict[str, Any]:
    if score.get("schema") != "direct-compact-attested-passk-v1":
        raise ValueError("input is not an attested direct-compact score")
    excluded = list(excluded_task_ids)
    if not excluded or len(set(excluded)) != len(excluded):
        raise ValueError("exclusions must be nonempty and unique")
    task_rows = score.get("task_results")
    candidate_rows = score.get("candidate_results")
    if not isinstance(task_rows, list) or not isinstance(candidate_rows, list):
        raise ValueError("input score lacks task/candidate detail")
    known = {str(row.get("task_id") or "") for row in task_rows}
    missing = sorted(set(excluded) - known)
    if missing:
        raise ValueError(f"excluded task IDs are absent: {missing}")
    retained_tasks = [
        row for row in task_rows if str(row.get("task_id") or "") not in set(excluded)
    ]
    retained_candidates = [
        row
        for row in candidate_rows
        if str(row.get("task_id") or "") not in set(excluded)
    ]
    k = int(score.get("k", 0))
    if k <= 0 or len(retained_candidates) != len(retained_tasks) * k:
        raise ValueError("retained candidate coverage is inconsistent")
    tasks = len(retained_tasks)
    if tasks <= 0:
        raise ValueError("exclusions removed every task")

    def metric(name: str) -> dict[str, float | int]:
        count = sum(bool(row.get(name)) for row in retained_tasks)
        return {"count": count, "rate": count / tasks}

    return {
        "schema": SCHEMA,
        "source_score_schema": score["schema"],
        "k": k,
        "tasks": tasks,
        "excluded_task_ids": excluded,
        "excluded_task_ids_sha256": canonical_sha256(excluded),
        "exclusion_reason": "known train/heldout exact acceptance-test duplicate in comparator training set",
        "pass_at_1": metric("pass_at_1"),
        "pass_at_k": metric("pass_at_k"),
        "compile_at_k": metric("compile_at_k"),
        "task_results": retained_tasks,
        "candidate_results": retained_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--score", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_task_id", action="append", required=True)
    args = parser.parse_args()
    score_path = Path(args.score).expanduser().resolve()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if not isinstance(score, dict):
        raise ValueError("score is not a JSON object")
    report = derive(score, args.exclude_task_id)
    report["source_score"] = {
        "path": str(score_path),
        "sha256": sha256_file(score_path),
    }
    require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    print(json.dumps({key: report[key] for key in ("tasks", "pass_at_1", "pass_at_k", "compile_at_k")}, sort_keys=True))


if __name__ == "__main__":
    main()
