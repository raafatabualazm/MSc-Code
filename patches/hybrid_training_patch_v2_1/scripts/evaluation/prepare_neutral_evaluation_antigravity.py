#!/usr/bin/env python3
"""Create an evaluation-only neutral-contract copy of a frozen benchmark.

The output is explicitly marked ``evaluation_only`` and never receives the
Phase-0 training approval bit.  This lets the same typed ``fn0`` contract be
used by the graph-causality gate without turning the frozen benchmark into a
training artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.hybrid_data_controls import (
    SCHEMA_VERSION,
    facts_comment,
    file_record,
    mechanical_facts,
    neutralize_training_row,
    read_jsonl_many,
    sanitize_verifier_diagnostic,
    source_text,
    task_identity,
    write_jsonl,
)


def _evaluator():
    from scripts.evaluation.graph_compile_at_k_antigravity import (  # type: ignore
        evaluate_dart_jit_tests_detail,
    )
    return evaluate_dart_jit_tests_detail


def _verify(row: dict[str, Any], timeout: int) -> dict[str, Any]:
    compiled, passed, diagnostic, _source = _evaluator()(
        source_text(row),
        str(row.get("tests") or ""),
        f"neutral_eval_{task_identity(row)}",
        timeout=timeout,
    )
    return {
        "compiled": bool(compiled),
        "passed": bool(passed),
        "diagnostic": sanitize_verifier_diagnostic(str(diagnostic or "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--neutral_name", default="fn0")
    parser.add_argument("--min_rows", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4) - 1)),
    )
    parser.add_argument(
        "--allow_reference_failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep failing benchmark rows but report them; off for confirmatory gates",
    )
    args = parser.parse_args()

    rows = read_jsonl_many(args.input)
    prepared: list[dict[str, Any]] = []
    preparation_failures: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        try:
            row = neutralize_training_row(raw, neutral_name=args.neutral_name)
            facts = mechanical_facts(row)
            row["binary_facts"] = facts
            row["facts_target_comment"] = facts_comment(facts)
            metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
            metadata.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "evaluation_only": True,
                    "phase0_approved": False,
                    "frozen_evaluation_copy": True,
                }
            )
            row["hybrid_metadata"] = metadata
            prepared.append(row)
        except Exception as exc:
            preparation_failures.append(
                {"task": task_identity(raw, index), "error": f"{type(exc).__name__}: {exc}"}
            )

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda row: _verify(row, args.timeout), prepared):
            results.append(result)
    failures: list[dict[str, Any]] = []
    for row, result in zip(prepared, results):
        metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
        metadata["reference_test_replay"] = result
        row["hybrid_metadata"] = metadata
        if not result["passed"]:
            failures.append({"task": task_identity(row), "result": result})

    write_jsonl(args.output, prepared)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepare_neutral_evaluation",
        "input_rows": len(rows),
        "output_rows": len(prepared),
        "preparation_failures": preparation_failures,
        "reference_failures": failures,
        "input": file_record(args.input),
        "output": file_record(args.output),
        "arguments": vars(args),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_rows": len(prepared), "reference_failures": len(failures)}, indent=2))

    fatal: list[str] = []
    if preparation_failures:
        fatal.append(f"{len(preparation_failures)} benchmark rows could not be neutralised")
    if len(prepared) < args.min_rows:
        fatal.append(f"only {len(prepared)} rows; minimum is {args.min_rows}")
    if failures and not args.allow_reference_failures:
        fatal.append(f"{len(failures)} neutralised references fail their own tests")
    if fatal:
        raise SystemExit("neutral evaluation gate failed: " + "; ".join(fatal))


if __name__ == "__main__":
    main()
