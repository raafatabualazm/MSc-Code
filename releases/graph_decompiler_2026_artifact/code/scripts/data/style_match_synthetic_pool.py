"""Sample a synthetic pool to better match a real Antigravity dataset.

The old large synthetic pool is schema/protocol compatible with the GRPO data,
but its distribution can drift: too many Maps/Sets/doubles, longer functions,
or much larger assembly. This script treats the synthetic pool as a reservoir
and selects rows whose return types and rough size profile resemble the real
set used for evaluation/training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def return_type(row: dict[str, Any]) -> str:
    sig = row.get("dart_function_signature", "") or ""
    match = re.match(r"\s*([\w<>?, ]+)\s+([A-Za-z_]\w*)\s*\(", sig)
    if not match:
        return "unknown"
    return re.sub(r"\s+", "", match.group(1))


def source_words(row: dict[str, Any]) -> int:
    return len((row.get("dart_source") or row.get("source") or "").split())


def expect_count(row: dict[str, Any]) -> int:
    tests = row.get("tests") or ""
    return tests.split("void expect(", 1)[0].count("expect(")


def asm_lines(row: dict[str, Any]) -> int:
    return len((row.get("assembly") or "").splitlines())


def median(values: list[int]) -> float:
    return float(stats.median(values)) if values else 0.0


def score_row(row: dict[str, Any], real_profile: dict[str, float]) -> float:
    src = source_words(row)
    asm = asm_lines(row)
    ex = expect_count(row)
    # Log distance avoids letting huge assembly dominate the score.
    src_score = abs(math.log((src + 1) / (real_profile["src_median"] + 1)))
    asm_score = abs(math.log((asm + 1) / (real_profile["asm_median"] + 1)))
    ex_score = abs(ex - real_profile["expect_median"]) / max(real_profile["expect_median"], 1.0)
    return src_score + 0.75 * asm_score + 0.5 * ex_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", required=True, type=Path)
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--target_rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_source_words", type=int, default=220)
    parser.add_argument("--max_asm_lines", type=int, default=700)
    parser.add_argument("--min_expects", type=int, default=3)
    parser.add_argument("--allow_extra_return_types", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    synthetic_rows = read_jsonl(args.synthetic)
    real_rows = read_jsonl(args.real)

    real_counts = Counter(return_type(row) for row in real_rows)
    real_profile = {
        "src_median": median([source_words(row) for row in real_rows]),
        "asm_median": median([asm_lines(row) for row in real_rows]),
        "expect_median": median([expect_count(row) for row in real_rows]),
    }

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejects = Counter()
    for row in synthetic_rows:
        rt = return_type(row)
        if not args.allow_extra_return_types and rt not in real_counts:
            rejects["return_type"] += 1
            continue
        if source_words(row) > args.max_source_words:
            rejects["source_words"] += 1
            continue
        if asm_lines(row) > args.max_asm_lines:
            rejects["asm_lines"] += 1
            continue
        if expect_count(row) < args.min_expects:
            rejects["expects"] += 1
            continue
        buckets[rt].append(row)

    for rows in buckets.values():
        rows.sort(key=lambda row: score_row(row, real_profile))

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    total_real = sum(real_counts.values())
    quotas = {
        rt: max(1, round(args.target_rows * count / total_real))
        for rt, count in real_counts.items()
    }

    for rt, quota in quotas.items():
        candidates = buckets.get(rt, [])
        take = min(quota, len(candidates))
        selected.extend(candidates[:take])

    if len(selected) < args.target_rows and args.allow_extra_return_types:
        used_ids = {id(row) for row in selected}
        extras = [
            row
            for rows in buckets.values()
            for row in rows
            if id(row) not in used_ids
        ]
        extras.sort(key=lambda row: score_row(row, real_profile))
        selected.extend(extras[: args.target_rows - len(selected)])

    rng.shuffle(selected)
    write_jsonl(args.output, selected)

    summary = {
        "synthetic": str(args.synthetic),
        "real": str(args.real),
        "output": str(args.output),
        "target_rows": args.target_rows,
        "selected_rows": len(selected),
        "real_return_counts": dict(real_counts),
        "selected_return_counts": dict(Counter(return_type(row) for row in selected)),
        "real_profile": real_profile,
        "selected_profile": {
            "src_median": median([source_words(row) for row in selected]),
            "asm_median": median([asm_lines(row) for row in selected]),
            "expect_median": median([expect_count(row) for row in selected]),
        },
        "rejects": dict(rejects),
        "bucket_sizes_after_filters": {rt: len(rows) for rt, rows in buckets.items()},
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
