#!/usr/bin/env python3
"""
Length-stratified, source-deduped, in-distribution train/held-out split for the
Flutter ARM64 corpus (Flagship B).

Guardrails baked in:
  - clean held-out: a function (keyed by exact dart_source) lands wholly on ONE
    side; eval rows never appear in train (and RS-SFT must harvest from train only).
  - length stays represented in eval: stratify by instruction-count bins so the
    >200-instruction cliff region B targets has adequate eval support.
  - deterministic: fixed seed; re-running reproduces the identical split.

Length = total real instructions (sum of CFG block instruction_count; falls back
to counting disasm lines if a row has no precomputed cfg).

Usage:
  python scripts/data/split_flutter_stratified.py \
      --input  data/datasets/flutter_function_assembly_pool_cfg.jsonl \
      --train-out data/datasets/flutter_train_cfg.jsonl \
      --eval-out  data/datasets/flutter_eval_cfg.jsonl \
      --eval-frac 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

BINS = [("<50", 0, 50), ("50-100", 50, 100), ("100-200", 100, 200),
        ("200-500", 200, 500), (">500", 500, 10**9)]


def instr_count(row: dict) -> int:
    cfg = row.get("cfg")
    if cfg:
        return sum(int(b.get("instruction_count") or len(b.get("instructions") or [])) for b in cfg)
    asm = row.get("assembly") or ""
    return sum(1 for l in asm.splitlines() if re.match(r'\s*(?:0x)?[0-9a-fA-F]+:\s', l))


def bin_of(n: int) -> str:
    for name, lo, hi in BINS:
        if lo <= n < hi:
            return name
    return ">500"


def src_key(row: dict) -> str:
    return hashlib.sha256((row.get("dart_source") or row.get("function") or "").encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--train-out", required=True, type=Path)
    ap.add_argument("--eval-out", required=True, type=Path)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]

    # Group by exact source so duplicates can never straddle the split.
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[src_key(r)].append(r)

    # Each group -> one bin (by the group's first row's length).
    by_bin: dict[str, list[str]] = defaultdict(list)
    for k, members in groups.items():
        by_bin[bin_of(instr_count(members[0]))].append(k)

    rng = random.Random(args.seed)
    train_keys: set[str] = set()
    eval_keys: set[str] = set()
    for name, _, _ in BINS:
        keys = sorted(by_bin.get(name, []))  # sorted for determinism pre-shuffle
        rng.shuffle(keys)
        n_eval = round(len(keys) * args.eval_frac)
        eval_keys.update(keys[:n_eval])
        train_keys.update(keys[n_eval:])

    train_rows = [r for r in rows if src_key(r) in train_keys]
    eval_rows = [r for r in rows if src_key(r) in eval_keys]

    # Hard invariant: zero source overlap.
    overlap = train_keys & eval_keys
    assert not overlap, f"LEAK: {len(overlap)} sources on both sides"

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    # Manifest.
    def bin_hist(rs):
        c = Counter(bin_of(instr_count(r)) for r in rs)
        return {name: c.get(name, 0) for name, _, _ in BINS}

    th, eh = bin_hist(train_rows), bin_hist(eval_rows)
    print(f"input rows: {len(rows)}  unique sources: {len(groups)}")
    print(f"train: {len(train_rows)}   eval: {len(eval_rows)}   (eval_frac={args.eval_frac}, seed={args.seed})")
    print(f"source overlap: {len(overlap)}  (must be 0)")
    print(f"{'bin':<10}{'train':>8}{'eval':>8}{'eval%':>8}")
    for name, _, _ in BINS:
        tot = th[name] + eh[name]
        print(f"{name:<10}{th[name]:>8}{eh[name]:>8}{(100*eh[name]/tot if tot else 0):>7.1f}%")
    print(f"\nwrote:\n  {args.train_out}\n  {args.eval_out}")


if __name__ == "__main__":
    main()
