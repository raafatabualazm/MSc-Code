"""
Build mixed Antigravity GRPO/SFT JSONL files from synthetic and real test rows.

The useful training pressure after the k50 audit is:
  * broad synthetic tasks for algorithmic coverage
  * all 154 real unit-test tasks for target-distribution anchoring
  * extra repeats of the real zero-pass tasks so training does not spend most
    updates on already-solved/easy prompts

This script preserves the normal row schema used by the trainers. It only adds
small _mix_* metadata fields, which the training code ignores.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def clone_row(row: dict[str, Any], source: str, repeat_index: int = 0) -> dict[str, Any]:
    cloned = dict(row)
    cloned["_mix_source"] = source
    cloned["_mix_repeat_index"] = repeat_index
    return cloned


def load_gap_indices(path: Path | None, key: str) -> list[int]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get(key, [])
    return [int(row["row_index"]) for row in rows if "row_index" in row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, default=Path("data/datasets/synthetic_pool_train576.jsonl"))
    parser.add_argument("--real", type=Path, default=Path("data/testing/grpo_data.jsonl"))
    parser.add_argument("--gap_analysis", type=Path, default=None,
                        help="Optional k50 rerank gap analysis JSON with all_zero_pass rows.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--synthetic_limit", type=int, default=0,
                        help="0 means all synthetic rows from --synthetic.")
    parser.add_argument("--synthetic_repeat", type=int, default=1)
    parser.add_argument("--real_repeat", type=int, default=1)
    parser.add_argument("--zero_pass_repeat", type=int, default=2,
                        help="Extra repeats for real rows listed in all_zero_pass.")
    parser.add_argument("--missed_passable_repeat", type=int, default=0,
                        help="Extra repeats for real rows listed in all_compile_missed_passable.")
    parser.add_argument("--shuffle", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    synthetic_rows = read_jsonl(args.synthetic)
    if args.synthetic_limit and args.synthetic_limit > 0:
        synthetic_rows = synthetic_rows[: args.synthetic_limit]
    real_rows = read_jsonl(args.real)

    zero_pass_indices = load_gap_indices(args.gap_analysis, "all_zero_pass")
    missed_passable_indices = load_gap_indices(args.gap_analysis, "all_compile_missed_passable")

    output_rows: list[dict[str, Any]] = []
    for repeat in range(args.synthetic_repeat):
        output_rows.extend(clone_row(row, "synthetic", repeat) for row in synthetic_rows)
    for repeat in range(args.real_repeat):
        output_rows.extend(clone_row(row, "real154", repeat) for row in real_rows)
    for repeat in range(args.zero_pass_repeat):
        for idx in zero_pass_indices:
            if 0 <= idx < len(real_rows):
                output_rows.append(clone_row(real_rows[idx], "real154_zero_pass", repeat))
    for repeat in range(args.missed_passable_repeat):
        for idx in missed_passable_indices:
            if 0 <= idx < len(real_rows):
                output_rows.append(clone_row(real_rows[idx], "real154_missed_passable", repeat))

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(output_rows)

    write_jsonl(args.output, output_rows)
    summary = {
        "output": str(args.output),
        "rows": len(output_rows),
        "synthetic": str(args.synthetic),
        "synthetic_rows_used": len(synthetic_rows),
        "synthetic_repeat": args.synthetic_repeat,
        "real": str(args.real),
        "real_rows": len(real_rows),
        "real_repeat": args.real_repeat,
        "gap_analysis": str(args.gap_analysis) if args.gap_analysis else None,
        "zero_pass_indices": len(zero_pass_indices),
        "zero_pass_repeat": args.zero_pass_repeat,
        "missed_passable_indices": len(missed_passable_indices),
        "missed_passable_repeat": args.missed_passable_repeat,
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
