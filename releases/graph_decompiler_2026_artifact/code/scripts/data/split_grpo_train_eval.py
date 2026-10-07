"""Deterministic half-split of grpo_data.jsonl for the RL generalization run.

The RL train file has always equaled the 154-task eval set (recorded
confound). For a defensible generalization claim, train RL on the even-index
half only and report eval metrics per half offline (passk_by_half.py).

Writes:
  data/testing/grpo_data_rl_train_half.jsonl   even-index rows (RL training)
  data/testing/grpo_split_halves.json          both halves' indices + ids

Usage:  python scripts/data/split_grpo_train_eval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/testing/grpo_data.jsonl", type=Path)
    parser.add_argument("--train_output", default="data/testing/grpo_data_rl_train_half.jsonl", type=Path)
    parser.add_argument("--split_output", default="data/testing/grpo_split_halves.json", type=Path)
    args = parser.parse_args()

    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_indices: list[int] = []
    heldout_indices: list[int] = []
    train_lines: list[str] = []
    row_meta: list[dict] = []

    for index, line in enumerate(lines):
        row = json.loads(line)
        identifier = row.get("id", row.get("filename", index))
        row_meta.append({"index": index, "id": identifier})
        if index % 2 == 0:
            train_indices.append(index)
            train_lines.append(line)
        else:
            heldout_indices.append(index)

    args.train_output.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    split = {
        "rule": "even row index -> RL train half; odd row index -> held out",
        "source": str(args.input),
        "total": len(lines),
        "train_half_indices": train_indices,
        "heldout_half_indices": heldout_indices,
        "rows": row_meta,
    }
    args.split_output.write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(f"total={len(lines)} train_half={len(train_indices)} heldout={len(heldout_indices)}")
    print(f"wrote {args.train_output}")
    print(f"wrote {args.split_output}")


if __name__ == "__main__":
    main()
