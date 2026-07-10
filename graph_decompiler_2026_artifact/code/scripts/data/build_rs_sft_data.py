"""Build rejection-sampling SFT data from every archived candidate pool.

Harvests every PASSING candidate across all *_pass_predictions.json /
*_pass_stats.csv pairs (same 154-task eval order), dedupes per task by
normalized code, caps per task preferring (cross-pool frequency desc,
length asc), and writes grpo_data.jsonl-schema rows with dart_source
replaced by the harvested candidate. The original assembly/tests/metadata
fields are preserved verbatim so the existing SFT pipeline consumes the
files unchanged.

Arms written (under data/testing/):
  rs_sft_all.jsonl             harvested passing candidates, all tasks
  rs_sft_all_plus_refs.jsonl   + the ORIGINAL row for every task with zero
                               passing candidates in the union (reference
                               as target - note _fitut already trained on
                               these refs; this arm is repetition pressure,
                               not new knowledge)
  rs_sft_train_half.jsonl      the plus_refs arm restricted to even-index
                               tasks (grpo_split_halves.json), for the
                               half-split generalization protocol

Offline; no Dart, no GPU.

Usage:
  python scripts/data/build_rs_sft_data.py --max_per_task 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from rerank_predictions_antigravity import _extract_code, _normalize_code  # noqa: E402

POOLS = [
    ("base", "results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_pass_predictions.json",
     "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_pass_stats.csv"),
    ("ut", "results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_pass_predictions.json",
     "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_pass_stats.csv"),
    ("ut_gentle", "results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo_pass_predictions.json",
     "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo_pass_stats.csv"),
    ("ut_gentle2", "results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle2_grpo_pass_predictions.json",
     "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle2_grpo_pass_stats.csv"),
    ("ut_grpo", "results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_grpo_pass_predictions.json",
     "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_grpo_pass_stats.csv"),
    ("ut_rewardfix", "results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardfix_grpo_pass_predictions.json",
     "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardfix_grpo_pass_stats.csv"),
    ("ut_rewardsoft", "results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_pass_predictions.json",
     "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_pass_stats.csv"),
    ("fitut", "results-qwen-9b-stageC/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut_pass_predictions.json",
     "results-qwen-9b-stageC/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut_pass_stats.csv"),
    ("fitut_grpo", "results-qwen-9b-stageC/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut_grpo_pass_predictions.json",
     "results-qwen-9b-stageC/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut_grpo_pass_stats.csv"),
    ("pk5", "results-qwen-9b-stageD/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo_pass_predictions.json",
     "results-qwen-9b-stageD/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo_pass_stats.csv"),
    ("pk5h", "results-qwen-9b-stageE/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5h_grpo_pass_predictions.json",
     "results-qwen-9b-stageE/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5h_grpo_pass_stats.csv"),
    ("pk5g16", "results-qwen-9b-stageF/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5g16_grpo_pass_predictions.json",
     "results-qwen-9b-stageF/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5g16_grpo_pass_stats.csv"),
    ("pk5dapo", "results-qwen-9b-stageG/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5dapo_grpo_pass_predictions.json",
     "results-qwen-9b-stageG/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5dapo_grpo_pass_stats.csv"),
    ("pk5gspo", "results-qwen-9b-stageG/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5gspo_grpo_pass_predictions.json",
     "results-qwen-9b-stageG/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5gspo_grpo_pass_stats.csv"),
]


def _flag(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    try:
        return 1 if float(text) >= 0.5 else 0
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/testing/grpo_data.jsonl", type=Path)
    parser.add_argument("--split", default="data/testing/grpo_split_halves.json", type=Path)
    parser.add_argument("--out_dir", default="data/testing", type=Path)
    parser.add_argument("--max_per_task", type=int, default=4)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_half = set(json.loads(args.split.read_text(encoding="utf-8"))["train_half_indices"])

    # candidates[task_idx] -> {normalized: (best_text, count_across_pools)}
    candidates: list[dict] = [dict() for _ in tasks]
    pool_pass_counts: dict[str, int] = {}

    for label, pred_path, stats_path in POOLS:
        pred_file, stats_file = Path(pred_path), Path(stats_path)
        if not pred_file.is_file() or not stats_file.is_file():
            print(f"WARNING: pool {label} missing files - skipped")
            continue
        preds = json.loads(pred_file.read_text(encoding="utf-8"))
        with stats_file.open("r", encoding="utf-8", newline="") as handle:
            stats = list(csv.DictReader(handle))
        if len(preds) != len(tasks) or len(stats) != len(tasks):
            print(f"WARNING: pool {label} row mismatch (preds={len(preds)} stats={len(stats)}) - skipped")
            continue
        n_pass = 0
        for idx, (pred_row, stat_row) in enumerate(zip(preds, stats)):
            cands = pred_row.get("predictions", [])
            for cand_idx, cand in enumerate(cands):
                if not _flag(stat_row.get(f"cand_{cand_idx + 1}_pass", "0")):
                    continue
                n_pass += 1
                code = _extract_code(cand).strip()
                if len(code) < 10:
                    continue
                key = _normalize_code(cand)
                if not key:
                    continue
                if key in candidates[idx]:
                    text, count = candidates[idx][key]
                    # keep the shorter surface form of the same normalized code
                    candidates[idx][key] = (min(text, code, key=len), count + 1)
                else:
                    candidates[idx][key] = (code, 1)
        pool_pass_counts[label] = n_pass
        print(f"pool {label}: {n_pass} passing candidates")

    harvested_rows = []
    tasks_covered = 0
    for idx, task in enumerate(tasks):
        if not candidates[idx]:
            continue
        tasks_covered += 1
        ranked = sorted(candidates[idx].values(), key=lambda item: (-item[1], len(item[0])))
        for code, _count in ranked[: args.max_per_task]:
            row = dict(task)
            row["dart_source"] = code
            harvested_rows.append((idx, row))

    zero_pass_indices = [idx for idx in range(len(tasks)) if not candidates[idx]]
    ref_rows = [(idx, dict(tasks[idx])) for idx in zero_pass_indices]

    def write_jsonl(path: Path, rows) -> None:
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for _, r in rows) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(rows)} rows)")

    all_rows = harvested_rows
    plus_refs = harvested_rows + ref_rows
    half_rows = [(idx, row) for idx, row in plus_refs if idx in train_half]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "rs_sft_all.jsonl", all_rows)
    write_jsonl(args.out_dir / "rs_sft_all_plus_refs.jsonl", plus_refs)
    write_jsonl(args.out_dir / "rs_sft_train_half.jsonl", half_rows)

    dedup_sizes = Counter(len(c) for c in candidates if c)
    print(json.dumps({
        "pools_used": list(pool_pass_counts.keys()),
        "passing_candidates_total": sum(pool_pass_counts.values()),
        "union_tasks_with_pass": tasks_covered,
        "zero_pass_tasks": len(zero_pass_indices),
        "harvested_rows_after_dedupe_cap": len(harvested_rows),
        "distinct_solutions_per_covered_task": dict(sorted(dedup_sizes.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
