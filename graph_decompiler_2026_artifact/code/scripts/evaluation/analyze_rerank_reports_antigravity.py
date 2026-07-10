"""
Compare Antigravity compile-reranker and oracle-reranker JSON reports.

This consumes the JSON reports written by rerank_predictions_antigravity.py.
It does not run Dart. The goal is to split failures into:

  * coverage failures: no candidate in the sampled pool passes tests
  * selection failures: a passing candidate exists, but compile reranking misses it

The oracle report must come from --mode test. The compile report should come
from --mode compile over the same prediction file and candidate ordering.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in data:
        raise SystemExit(f"ERROR: report has no rows: {path}")
    return data


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def candidate_passes(oracle_row: dict[str, Any]) -> list[int]:
    return [
        int(candidate.get("index", idx))
        for idx, candidate in enumerate(oracle_row.get("candidate_evaluations", []))
        if candidate.get("pass") is True
    ]


def candidate_at(row: dict[str, Any], index: int) -> dict[str, Any]:
    for candidate in row.get("candidate_evaluations", []):
        if int(candidate.get("index", -1)) == index:
            return candidate
    return {}


def bucket_first_pass(index: int | None) -> str:
    if index is None:
        return "none"
    one_based = index + 1
    if one_based == 1:
        return "1"
    if one_based <= 5:
        return "2-5"
    if one_based <= 10:
        return "6-10"
    if one_based <= 25:
        return "11-25"
    return "26+"


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    features = candidate.get("features", {}) or {}
    return {
        "index": candidate.get("index"),
        "score": candidate.get("score"),
        "compile": candidate.get("compile"),
        "pass": candidate.get("pass"),
        "name_match": features.get("name_match"),
        "has_main": features.get("has_main"),
        "has_markdown": features.get("has_markdown"),
        "has_balanced_braces": features.get("has_balanced_braces"),
        "length": features.get("length"),
        "diagnostic": candidate.get("diagnostic"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile_report", required=True, type=Path)
    parser.add_argument("--oracle_report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top_examples", type=int, default=25)
    args = parser.parse_args()

    compile_report = load_report(args.compile_report)
    oracle_report = load_report(args.oracle_report)
    compile_rows = compile_report["rows"]
    oracle_rows = oracle_report["rows"]

    if len(compile_rows) != len(oracle_rows):
        raise SystemExit(
            f"ERROR: report row counts differ: compile={len(compile_rows)} oracle={len(oracle_rows)}"
        )

    total = len(oracle_rows)
    passable_rows = []
    zero_pass_rows = []
    compile_hits = []
    compile_misses = []
    original_hits = []
    first_pass_indices: list[int] = []
    first_pass_buckets: Counter[str] = Counter()

    for row_idx, (compile_row, oracle_row) in enumerate(zip(compile_rows, oracle_rows)):
        passing_indices = candidate_passes(oracle_row)
        first_pass = min(passing_indices) if passing_indices else None
        first_pass_buckets[bucket_first_pass(first_pass)] += 1
        if first_pass is not None:
            first_pass_indices.append(first_pass + 1)

        compile_idx = int(compile_row.get("best_original_index", 0))
        oracle_idx = int(oracle_row.get("best_original_index", 0))
        original_pass = 0 in passing_indices
        compile_selected_pass = compile_idx in passing_indices
        passable = bool(passing_indices)

        base = {
            "row_index": row_idx,
            "problem_id": oracle_row.get("problem_id"),
            "filename": oracle_row.get("filename"),
            "target_name": oracle_row.get("target_name"),
            "passing_indices_0_based": passing_indices,
            "passing_indices_1_based": [idx + 1 for idx in passing_indices],
            "first_pass_index_1_based": first_pass + 1 if first_pass is not None else None,
            "original_pass": original_pass,
            "compile_selected_pass": compile_selected_pass,
            "compile_selected_index_1_based": compile_idx + 1,
            "oracle_selected_index_1_based": oracle_idx + 1,
            "compile_selected": compact_candidate(candidate_at(oracle_row, compile_idx)),
            "oracle_selected": compact_candidate(candidate_at(oracle_row, oracle_idx)),
        }

        if original_pass:
            original_hits.append(base)
        if passable:
            passable_rows.append(base)
            if compile_selected_pass:
                compile_hits.append(base)
            else:
                compile_misses.append(base)
        else:
            zero_pass_rows.append(base)

    missed_sorted = sorted(
        compile_misses,
        key=lambda item: (
            item["first_pass_index_1_based"] or 10**9,
            item["row_index"],
        ),
    )
    zero_sorted = sorted(zero_pass_rows, key=lambda item: item["row_index"])

    result = {
        "compile_report": str(args.compile_report),
        "oracle_report": str(args.oracle_report),
        "summary": {
            "total_problems": total,
            "candidate0_pass_count": len(original_hits),
            "candidate0_pass_rate": rate(len(original_hits), total),
            "oracle_passable_count": len(passable_rows),
            "oracle_passable_rate": rate(len(passable_rows), total),
            "compile_selected_pass_count": len(compile_hits),
            "compile_selected_pass_rate": rate(len(compile_hits), total),
            "compile_missed_passable_count": len(compile_misses),
            "compile_missed_passable_rate": rate(len(compile_misses), total),
            "zero_pass_count": len(zero_pass_rows),
            "zero_pass_rate": rate(len(zero_pass_rows), total),
            "compile_recovered_fraction_of_passable": rate(len(compile_hits), len(passable_rows)),
            "mean_first_pass_index_1_based": mean(first_pass_indices) if first_pass_indices else None,
            "median_first_pass_index_1_based": median(first_pass_indices) if first_pass_indices else None,
            "first_pass_bucket_counts": dict(first_pass_buckets),
        },
        "compile_missed_passable_examples": missed_sorted[: args.top_examples],
        "zero_pass_examples": zero_sorted[: args.top_examples],
        "all_compile_missed_passable": compile_misses,
        "all_zero_pass": zero_pass_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved analysis to: {args.output}")


if __name__ == "__main__":
    main()
