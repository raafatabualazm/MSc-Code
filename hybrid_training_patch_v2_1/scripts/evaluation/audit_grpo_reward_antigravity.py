#!/usr/bin/env python3
"""Fail-closed static and executable audit for hybrid training pools.

Static mode validates contracts, provenance, and test extraction without a GPU.
``--run_references -1`` additionally checks every reference under each requested
harness through the exact evaluator used by pass@k.  Run this before any SFT and
again after verified RS-SFT construction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.hybrid_data_controls import (  # noqa: E402
    assert_training_approved,
    candidate_expect_lines,
    infer_function_name,
    read_jsonl_many,
    sanitize_verifier_diagnostic,
    source_text,
    task_key,
    verified_origin,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_evaluator():
    try:
        from scripts.evaluation.graph_compile_at_k_antigravity import (  # type: ignore
            evaluate_dart_jit_tests_detail,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("project-aligned pass@k evaluator is unavailable") from exc
    return evaluate_dart_jit_tests_detail


def static_audit(
    rows: list[dict[str, Any]],
    test_fields: list[str],
    *,
    require_phase0: bool,
    require_neutral: bool,
    require_verified: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    counts: dict[str, list[int]] = {field: [] for field in test_fields}
    for index, row in enumerate(rows):
        reasons: list[str] = []
        metadata = row.get("hybrid_metadata") or {}
        if metadata.get("evaluation_only") is True and (require_phase0 or require_verified):
            reasons.append("evaluation-only row in training audit")
        if require_phase0:
            try:
                assert_training_approved(row)
            except Exception as exc:
                reasons.append(str(exc))
        if require_neutral:
            if metadata.get("neutral_contract") is not True or infer_function_name(row) != "fn0":
                reasons.append("missing neutral fn0 contract")
        if require_verified and not verified_origin(row):
            reasons.append("invalid verified-alternative provenance")
        if not source_text(row).strip():
            reasons.append("missing reference source")
        if not str(row.get("assembly") or "").strip():
            reasons.append("missing assembly")
        for field in test_fields:
            tests = str(row.get(field) or "")
            assertions = candidate_expect_lines(tests)
            counts[field].append(len(assertions))
            optional_empty_feedback = field == "feedback_tests" and not tests
            if not tests and not optional_empty_feedback:
                reasons.append(f"missing {field}")
            elif tests and not assertions:
                reasons.append(f"{field}: no expect(candidate(...)) assertions")
            if tests and not re.search(r"\bfinal\s+candidate\s*=\s*[A-Za-z_]\w*\s*;", tests):
                reasons.append(f"{field}: missing candidate alias")
        if reasons:
            failures.append({"index": index, "task_key": task_key(row, index), "reasons": reasons})
    summary = {
        "rows": len(rows),
        "static_failures": len(failures),
        "test_fields": {
            field: {
                "assertions_min": min(values) if values else 0,
                "assertions_max": max(values) if values else 0,
            }
            for field, values in counts.items()
        },
    }
    return failures, summary


def run_reference_audit(
    rows: list[dict[str, Any]],
    test_fields: list[str],
    limit: int,
    workers: int,
    timeout: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    selected = rows if limit < 0 else rows[:limit]
    evaluator = load_evaluator()

    def check(item: tuple[int, dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        index, row = item
        problems: list[dict[str, Any]] = []
        for field in test_fields:
            tests = str(row.get(field) or "")
            if field == "feedback_tests" and not tests:
                continue
            try:
                compiled, passed, diagnostic, _source = evaluator(
                    source_text(row),
                    tests,
                    f"audit_{index}_{field}",
                    timeout=timeout,
                )
                if not bool(compiled and passed):
                    problems.append(
                        {
                            "field": field,
                            "compiled": bool(compiled),
                            "passed": bool(passed),
                            "diagnostic": sanitize_verifier_diagnostic(str(diagnostic or "")),
                        }
                    )
            except Exception as exc:
                problems.append({"field": field, "error": repr(exc)})
        return index, problems

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(check, item): item for item in enumerate(selected)}
        for done, future in enumerate(as_completed(futures), 1):
            index, problems = future.result()
            if problems:
                failures.append(
                    {
                        "index": index,
                        "task_key": task_key(selected[index], index),
                        "failures": problems,
                    }
                )
            if done % 100 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] reference parity failures={len(failures)}")
    failures.sort(key=lambda item: item["index"])
    checked_harnesses = sum(
        1
        for row in selected
        for field in test_fields
        if str(row.get(field) or "").strip()
    )
    return len(selected), checked_harnesses, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dataset", required=True,
                        help="One JSONL or comma-separated JSONL files")
    parser.add_argument("--test_fields", default="tests",
                        help="Comma-separated harness fields, e.g. feedback_tests,acceptance_tests,tests")
    parser.add_argument("--run_references", type=int, default=0,
                        help="0=static only, -1=all, positive=N first rows")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--require_phase0_approved", action="store_true")
    parser.add_argument("--require_neutral_contract", action="store_true")
    parser.add_argument("--require_verified_origin", action="store_true")
    args = parser.parse_args()

    fields = [value.strip() for value in args.test_fields.split(",") if value.strip()]
    if not fields:
        parser.error("--test_fields cannot be empty")
    rows = read_jsonl_many(args.dataset)
    failures, summary = static_audit(
        rows,
        fields,
        require_phase0=args.require_phase0_approved,
        require_neutral=args.require_neutral_contract,
        require_verified=args.require_verified_origin,
    )
    report: dict[str, Any] = {**summary, "static_failure_rows": failures}
    if args.run_references != 0:
        checked, checked_harnesses, parity = run_reference_audit(
            rows, fields, args.run_references, args.workers, args.timeout
        )
        report["reference_reward_parity"] = {
            "checked_rows": checked,
            "checked_harnesses": checked_harnesses,
            "passed_rows": checked - len(parity),
            "failures": parity,
        }
    report_path = args.report or Path(args.dataset.split(",")[0]).with_suffix(".reward_audit.json")
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures or report.get("reference_reward_parity", {}).get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
