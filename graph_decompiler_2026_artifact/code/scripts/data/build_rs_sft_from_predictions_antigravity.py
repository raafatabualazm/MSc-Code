"""Build RS-SFT rows from arbitrary Antigravity pass prediction pools.

This is the generic version of the older 9B-specific harvester. It scans a
results directory for pass-prediction JSON files, pairs each one with the
matching per-candidate pass_stats CSV, harvests candidates marked as passing,
dedupes them per task, and writes JSONL rows that preserve the original
assembly/tests/CFG fields while replacing dart_source with the passing
candidate.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from rerank_predictions_antigravity import _extract_code, _normalize_code  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {path} ({len(rows)} rows)")


def flag(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    try:
        return float(text) >= 0.5
    except ValueError:
        return False


def prediction_stem(path: Path) -> str:
    name = path.name
    if name.endswith("_pass_predictions.json"):
        return name[: -len("_pass_predictions.json")]
    return path.stem


def candidate_stats_path(results_dir: Path, prediction_path: Path) -> Path:
    return results_dir / "sweeps_antigravity" / f"{prediction_stem(prediction_path)}_pass_stats.csv"


def include_path(path: Path, includes: list[str], excludes: list[str]) -> bool:
    text = path.as_posix()
    if includes and not any(fnmatch.fnmatch(text, pat) or fnmatch.fnmatch(path.name, pat) for pat in includes):
        return False
    return not any(fnmatch.fnmatch(text, pat) or fnmatch.fnmatch(path.name, pat) for pat in excludes)


def load_split_indices(path: Path | None, split_key: str) -> set[int] | None:
    if not path or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get(split_key)
    if values is None:
        raise SystemExit(f"split file {path} does not contain key {split_key!r}")
    return {int(v) for v in values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path,
                        help="Reference task JSONL, usually data/testing/grpo_data_cfg.jsonl")
    parser.add_argument("--results_dir", required=True, type=Path,
                        help="Directory containing predictions and sweeps_antigravity stats")
    parser.add_argument("--out_prefix", required=True, type=Path,
                        help="Output prefix; writes *_all.jsonl, *_all_plus_refs.jsonl, optional *_train_half.jsonl")
    parser.add_argument("--include", action="append", default=[],
                        help="Glob include pattern. Can be repeated. Example: '*qwen3-8b*x86*g3*pass_predictions.json'")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Glob exclude pattern. Can be repeated.")
    parser.add_argument("--max_per_task", type=int, default=4)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--split_key", default="train_half_indices")
    parser.add_argument("--min_code_chars", type=int, default=10)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    tasks = read_jsonl(args.data)
    train_half = load_split_indices(args.split, args.split_key)

    excludes = list(args.exclude) + [
        "*selected*",
        "*rerank*",
        "*oracle*",
        "*k50*",
    ]
    pred_files = [
        path
        for path in sorted(args.results_dir.glob("*_pass_predictions.json"))
        if include_path(path, args.include, excludes)
    ]
    if not pred_files:
        raise SystemExit("no matching pass prediction files found")

    # candidates[task_idx][normalized_code] = {code, count, pools}
    candidates: list[dict[str, dict[str, Any]]] = [dict() for _ in tasks]
    pool_summaries: list[dict[str, Any]] = []

    for pred_file in pred_files:
        stats_file = candidate_stats_path(args.results_dir, pred_file)
        label = prediction_stem(pred_file)
        if not stats_file.is_file():
            print(f"WARNING: {label}: missing stats {stats_file}; skipped")
            continue
        try:
            preds = json.loads(pred_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"WARNING: {label}: cannot parse predictions ({exc}); skipped")
            continue
        with stats_file.open("r", encoding="utf-8", newline="") as handle:
            stats = list(csv.DictReader(handle))
        if len(preds) != len(tasks) or len(stats) != len(tasks):
            print(
                f"WARNING: {label}: row mismatch "
                f"(preds={len(preds)} stats={len(stats)} tasks={len(tasks)}); skipped"
            )
            continue

        pool_pass_candidates = 0
        pool_tasks = set()
        for task_idx, (pred_row, stat_row) in enumerate(zip(preds, stats)):
            raw_candidates = pred_row.get("predictions") or []
            for cand_idx, candidate in enumerate(raw_candidates):
                if not flag(stat_row.get(f"cand_{cand_idx + 1}_pass", "0")):
                    continue
                code = _extract_code(candidate).strip()
                if len(code) < args.min_code_chars:
                    continue
                key = _normalize_code(code)
                if not key:
                    continue
                bucket = candidates[task_idx]
                if key in bucket:
                    prev = bucket[key]
                    if len(code) < len(prev["code"]):
                        prev["code"] = code
                    prev["count"] += 1
                    prev["pools"].add(label)
                else:
                    bucket[key] = {"code": code, "count": 1, "pools": {label}}
                pool_pass_candidates += 1
                pool_tasks.add(task_idx)

        pool_summaries.append({
            "label": label,
            "predictions": str(pred_file),
            "stats": str(stats_file),
            "passing_candidates": pool_pass_candidates,
            "tasks_with_pass": len(pool_tasks),
        })
        print(f"pool {label}: {pool_pass_candidates} passing candidates, {len(pool_tasks)} tasks")

    harvested: list[tuple[int, dict[str, Any]]] = []
    covered_indices = []
    per_task_distinct = Counter()
    for task_idx, task in enumerate(tasks):
        bucket = candidates[task_idx]
        if not bucket:
            continue
        covered_indices.append(task_idx)
        per_task_distinct[len(bucket)] += 1
        ranked = sorted(bucket.values(), key=lambda item: (-item["count"], len(item["code"])))
        for item in ranked[: args.max_per_task]:
            row = dict(task)
            row["dart_source"] = item["code"]
            row["rs_sft_source_pools"] = sorted(item["pools"])
            row["rs_sft_pool_count"] = item["count"]
            row["rs_sft_task_index"] = task_idx
            harvested.append((task_idx, row))

    zero_pass_indices = [idx for idx in range(len(tasks)) if not candidates[idx]]
    plus_refs = harvested + [(idx, dict(tasks[idx])) for idx in zero_pass_indices]

    all_rows = [row for _idx, row in harvested]
    plus_ref_rows = [row for _idx, row in plus_refs]
    write_jsonl(args.out_prefix.with_name(args.out_prefix.name + "_all.jsonl"), all_rows)
    write_jsonl(args.out_prefix.with_name(args.out_prefix.name + "_all_plus_refs.jsonl"), plus_ref_rows)

    train_half_rows = None
    if train_half is not None:
        train_half_rows = [row for idx, row in plus_refs if idx in train_half]
        write_jsonl(args.out_prefix.with_name(args.out_prefix.name + "_train_half.jsonl"), train_half_rows)

    summary = {
        "data": str(args.data),
        "results_dir": str(args.results_dir),
        "prediction_files_used": len(pool_summaries),
        "pools": pool_summaries,
        "union_tasks_with_pass": len(covered_indices),
        "zero_pass_tasks": len(zero_pass_indices),
        "covered_indices": covered_indices,
        "zero_pass_indices": zero_pass_indices,
        "passing_candidates_total": sum(pool["passing_candidates"] for pool in pool_summaries),
        "harvested_rows_after_dedupe_cap": len(harvested),
        "max_per_task": args.max_per_task,
        "distinct_solutions_per_covered_task": dict(sorted(per_task_distinct.items())),
        "all_rows": len(all_rows),
        "all_plus_refs_rows": len(plus_ref_rows),
        "train_half_rows": len(train_half_rows) if train_half_rows is not None else None,
    }
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
