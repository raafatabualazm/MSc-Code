#!/usr/bin/env python3
"""Build an explicit 50/50 gold-to-verified RS-SFT epoch.

Simple JSONL concatenation makes a small verified pool disappear inside thousands
of reference rows.  This builder materialises the intended sampling policy and
fails if doing so would require pathological oversampling of a tiny verified
set.  The result can be consumed by the unmodified HF Trainer shuffle logic.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path

from scripts.training.hybrid_data_controls import (
    SCHEMA_VERSION,
    file_record,
    read_jsonl_many,
    assert_training_approved,
    task_identity,
    verified_origin,
    write_jsonl,
)


def _cycle_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not rows and count:
        raise ValueError("cannot sample from an empty pool")
    result: list[dict] = []
    while len(result) < count:
        epoch = list(rows)
        rng.shuffle(epoch)
        result.extend(epoch[: count - len(result)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--gold_control_out",
        default="",
        help=(
            "Optional gold-only matched-step control. It contains exactly the same "
            "number of rows as the 50/50 mix and is used for a like-for-like RS-SFT "
            "improvement baseline."
        ),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--rows_per_epoch",
        type=int,
        default=0,
        help=(
            "Even output size. 0 emits exactly 2 * len(gold), so every approved "
            "gold row appears once and the verified half is sampled with replacement."
        ),
    )
    parser.add_argument(
        "--allow_partial_gold_coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Research-only escape hatch allowing an explicit --rows_per_epoch whose "
            "gold half is smaller than the approved gold pool. Disabled by default "
            "because a static epoch file would otherwise silently discard gold rows."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_verified_rows", type=int, default=64)
    parser.add_argument("--min_verified_unique_tasks", type=int, default=64)
    parser.add_argument("--min_verified_length_bins", type=int, default=3)
    parser.add_argument("--max_verified_oversample_factor", type=float, default=4.0)
    args = parser.parse_args()

    gold = read_jsonl_many(args.gold)
    verified_raw = read_jsonl_many(args.verified)
    if not gold:
        raise SystemExit("gold pool is empty")
    bad_gold: list[str] = []
    for index, row in enumerate(gold):
        try:
            assert_training_approved(row)
        except Exception as exc:
            bad_gold.append(f"{task_identity(row, index)}: {exc}")
    if bad_gold:
        raise SystemExit("gold mix contains unsafe rows: " + "; ".join(bad_gold[:8]))
    verified = [row for row in verified_raw if verified_origin(row)]
    if len(verified) != len(verified_raw):
        raise SystemExit(
            f"verified pool contains {len(verified_raw) - len(verified)} rows without strict "
            "verifier/hidden/facts provenance"
        )

    unique_tasks = {task_identity(row, i) for i, row in enumerate(verified)}
    raw_bins = [
        str((row.get("hybrid_metadata") or {}).get("length_bin") or "unknown")
        for row in verified
    ]
    if any(value == "unknown" for value in raw_bins):
        raise SystemExit(
            "verified pool contains rows without a Phase-0 length_bin; coverage cannot be audited"
        )
    bins = set(raw_bins)
    if len(verified) < args.min_verified_rows:
        raise SystemExit(
            f"only {len(verified)} verified rows; minimum is {args.min_verified_rows}"
        )
    if len(unique_tasks) < args.min_verified_unique_tasks:
        raise SystemExit(
            f"only {len(unique_tasks)} unique verified tasks; minimum is "
            f"{args.min_verified_unique_tasks}"
        )
    if len(bins) < args.min_verified_length_bins:
        raise SystemExit(
            f"verified pool covers only {len(bins)} length bins {sorted(bins)}; minimum is "
            f"{args.min_verified_length_bins}"
        )

    if args.max_verified_oversample_factor <= 0.0:
        raise SystemExit("--max_verified_oversample_factor must be positive")

    # A materialised epoch is not a weighted sampler: rows omitted here can never
    # appear during Trainer shuffling. Therefore the default must cover the entire
    # approved gold pool, not silently shrink gold until a tiny verified pool fits.
    rows_per_epoch = args.rows_per_epoch or (2 * len(gold))
    if rows_per_epoch <= 0 or rows_per_epoch % 2:
        raise SystemExit("--rows_per_epoch must be a positive even integer")
    half = rows_per_epoch // 2
    if half < len(gold) and not args.allow_partial_gold_coverage:
        raise SystemExit(
            f"requested epoch has only {half} gold slots for {len(gold)} approved gold "
            "rows. This would silently omit training rows; increase --rows_per_epoch "
            "to at least 2*len(gold), or explicitly pass --allow_partial_gold_coverage "
            "for a research-only subsampled epoch."
        )
    required_verified_oversample = half / max(1, len(verified))
    if required_verified_oversample > args.max_verified_oversample_factor + 1e-12:
        minimum_verified = math.ceil(half / args.max_verified_oversample_factor)
        raise SystemExit(
            f"50/50 full-coverage mix needs {half} verified slots from "
            f"{len(verified)} verified rows ({required_verified_oversample:.2f}x), "
            f"above --max_verified_oversample_factor="
            f"{args.max_verified_oversample_factor}. Harvest at least "
            f"{minimum_verified} verified rows, raise the factor explicitly, or "
            "reduce the approved gold pool before this stage."
        )

    rng = random.Random(args.seed)
    chosen_gold = _cycle_sample(gold, half, rng)
    chosen_verified = _cycle_sample(verified, half, rng)
    mixed: list[dict] = []
    for bucket, chosen in (("gold", chosen_gold), ("verified", chosen_verified)):
        for row in chosen:
            copied = copy.deepcopy(row)
            metadata = copy.deepcopy(copied.get("hybrid_metadata") or {})
            metadata.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sft_mix_bucket": bucket,
                    "sft_mix_ratio_target": 0.5,
                    "sft_mix_seed": args.seed,
                }
            )
            copied["hybrid_metadata"] = metadata
            mixed.append(copied)
    rng.shuffle(mixed)

    # A scientifically valid +6 pp gate compares RS-SFT against a control that
    # starts from the same checkpoint, uses the same full prompt, optimiser,
    # epochs, and number of training examples, but contains gold only.
    gold_control: list[dict] = []
    if args.gold_control_out:
        control_rng = random.Random(args.seed + 104729)
        for row in _cycle_sample(gold, rows_per_epoch, control_rng):
            copied = copy.deepcopy(row)
            metadata = copy.deepcopy(copied.get("hybrid_metadata") or {})
            metadata.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sft_mix_bucket": "gold_control",
                    "sft_mix_ratio_target": 0.0,
                    "sft_mix_seed": args.seed,
                    "matched_step_control_for": str(Path(args.out).expanduser().resolve()),
                }
            )
            copied["hybrid_metadata"] = metadata
            gold_control.append(copied)
        write_jsonl(args.gold_control_out, gold_control)

    count = write_jsonl(args.out, mixed)
    bucket_counts = Counter((row.get("hybrid_metadata") or {}).get("sft_mix_bucket") for row in mixed)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "balanced_rs_sft_mix",
        "gold_pool_rows": len(gold),
        "verified_pool_rows": len(verified),
        "verified_unique_tasks": len(unique_tasks),
        "verified_length_bins": sorted(bins),
        "output_rows": count,
        "bucket_counts": dict(bucket_counts),
        "realized_verified_ratio": bucket_counts.get("verified", 0) / max(1, count),
        "gold_coverage_fraction": min(1.0, half / max(1, len(gold))),
        "gold_full_coverage_enforced": not args.allow_partial_gold_coverage,
        "verified_oversample_factor": half / max(1, len(verified)),
        "inputs": {
            "gold": file_record(args.gold),
            "verified": file_record(args.verified),
        },
        "output": file_record(args.out),
        "gold_control_output": (
            file_record(args.gold_control_out) if args.gold_control_out else None
        ),
        "gold_control_rows": len(gold_control),
        "gold_control_matches_training_examples": (
            len(gold_control) == count if args.gold_control_out else None
        ),
        "arguments": vars(args),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["realized_verified_ratio"] != 0.5:
        raise RuntimeError("internal error: output mix is not exactly 50/50")
    if args.gold_control_out and len(gold_control) != count:
        raise RuntimeError("internal error: gold-only control does not match RS-SFT row count")


if __name__ == "__main__":
    main()
