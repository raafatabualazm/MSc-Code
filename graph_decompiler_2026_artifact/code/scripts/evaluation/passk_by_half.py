"""Per-half pass@k from a *_pass_stats.csv and the half-split JSON.

Recomputes the harness's unbiased pass@k estimator from the stored
per-candidate pass flags, decomposed into the RL train half and the held-out
half defined by split_grpo_train_eval.py. The "overall" block must reproduce
the harness numbers exactly - if it does not, the CSV and split do not
belong to the same eval layout.

Usage:
  python scripts/evaluation/passk_by_half.py \
    --stats results-.../sweeps_antigravity/NAME_pass_stats.csv \
    --split data/testing/grpo_split_halves.json --label NAME
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from math import comb
from pathlib import Path


def _flag(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    try:
        return 1 if float(text) >= 0.5 else 0
    except ValueError:
        return 0


def pass_at_k(c: int, n: int, k: int) -> float:
    if n < k:
        k = n  # fall back to @n, matching the harness
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    with args.stats.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split = json.loads(args.split.read_text(encoding="utf-8"))

    if split["total"] != len(rows):
        raise SystemExit(f"ERROR: split covers {split['total']} rows, CSV has {len(rows)}")

    n_cands = max(
        int(match.group(1))
        for match in (re.match(r"cand_(\d+)_pass$", col) for col in rows[0])
        if match
    )
    pass_counts = [
        sum(_flag(row.get(f"cand_{i + 1}_pass", "0")) for i in range(n_cands))
        for row in rows
    ]

    groups = {
        "overall": list(range(len(rows))),
        "train_half": split["train_half_indices"],
        "heldout_half": split["heldout_half_indices"],
    }
    out: dict = {"label": args.label or args.stats.stem, "n_candidates": n_cands}
    for name, indices in groups.items():
        block = {
            f"pass_at_{k}": round(
                sum(pass_at_k(pass_counts[i], n_cands, k) for i in indices) / len(indices), 4
            )
            for k in args.k
        }
        block["tasks"] = len(indices)
        block["any_pass"] = sum(1 for i in indices if pass_counts[i] > 0)
        out[name] = block

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
