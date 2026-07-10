#!/usr/bin/env python3
"""
Validate ARM64 (and, as a regression guard, x86) CFG extraction.

Runs AssemblyCFGExtractor over a JSONL pool and reports:
  - block / edge count distributions
  - single-block rate, split by whether the function actually contains branches
    (a branchy function that still collapses to 1 block == the dead-CFG bug)
  - intra-procedural branch resolution rate (b.<cond>/cbz/tbz targets that land
    inside the dumped function vs. fall outside / are indirect)
  - per-block instruction-count distribution (must fit GraphCodeBERT's window)
  - edge-type histogram and structural integrity (valid indices, entry-reachable)
  - a couple of fully-resolved sample CFGs for eyeballing

Usage:
  python scripts/data/validate_arm64_cfg.py --input data/datasets/flutter_function_assembly_pool.jsonl
  python scripts/data/validate_arm64_cfg.py --input data/testing/grpo_data.jsonl --samples 0   # x86 regression
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_extractor import (  # noqa: E402
    AssemblyCFGExtractor,
    CONDITIONAL_JUMPS,
    UNCONDITIONAL_JUMPS,
    CALL_MNEMONICS,
)

BRANCH_MNEMONICS = (CONDITIONAL_JUMPS | UNCONDITIONAL_JUMPS) - {'b.al', 'b.nv'}


def opcodes_in(assembly: str):
    ops = Counter()
    for line in assembly.splitlines():
        s = line.split(';')[0].strip()
        m = re.match(r'(?:0x[0-9a-fA-F]+|[0-9a-fA-F]+:)', s)
        if not m:
            continue
        rest = s.split(':', 1)[1].strip() if ':' in s else s[m.end():].strip()
        if rest:
            ops[rest.split()[0].lower()] += 1
    return ops


def pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def quantiles(xs):
    if not xs:
        return (0, 0, 0, 0, 0)
    xs = sorted(xs)
    return (xs[0], statistics.median(xs), round(statistics.mean(xs), 1),
            xs[int(0.95 * (len(xs) - 1))], xs[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--samples', type=int, default=2, help='detailed CFG dumps to print')
    ap.add_argument('--block-token-est', type=float, default=1.5,
                    help='chars/token estimate to flag blocks that may exceed 512 tokens')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding='utf-8') if l.strip()]
    n = len(rows)
    print(f"== CFG extraction validation: {args.input} ({n} rows) ==\n")

    n_blocks, n_edges = [], []
    block_sizes = []  # instructions per block
    single_block = 0
    single_block_branchy = []  # (idx, fn) collapsed despite having branches
    parse_empty = 0
    extract_err = 0
    bad_index_rows = 0
    unreachable_rows = 0
    edge_types = Counter()
    branch_total = 0
    branch_resolved = 0
    big_block_rows = 0  # any block likely > 512 tokens

    detailed = []

    for idx, r in enumerate(rows):
        asm = r.get('assembly') or ''
        ops = opcodes_in(asm)
        has_branch = any(ops.get(b, 0) for b in BRANCH_MNEMONICS)
        n_branch_instr = sum(ops.get(b, 0) for b in BRANCH_MNEMONICS)

        try:
            ex = AssemblyCFGExtractor(asm)
            blocks, edges, integrity = ex.build_blocks()
        except Exception as e:
            extract_err += 1
            continue

        if not blocks:
            parse_empty += 1
            continue

        nb = len(blocks)
        n_blocks.append(nb)
        n_edges.append(len(edges))
        for b in blocks:
            block_sizes.append(b.instruction_count)
            if b.instruction_count and (b.instruction_count * 4 * args.block_token_est) > 512 * args.block_token_est:
                pass  # placeholder; real check below per-row

        # per-row big-block check (chars-based estimate)
        if any(sum(len(i) for i in b.instructions) / max(args.block_token_est, 1e-9) > 512 for b in blocks):
            big_block_rows += 1

        for e in edges:
            edge_types[e.edge_type] += 1

        # index validity
        if any(not (0 <= e.source < nb and 0 <= e.target < nb) for e in edges):
            bad_index_rows += 1
        if integrity.get('unreachable_nodes'):
            unreachable_rows += 1

        if nb == 1:
            single_block += 1
            if has_branch:
                single_block_branchy.append((idx, r.get('function'), n_branch_instr))

        # intra-procedural branch resolution: how many conditional/uncond
        # branch instructions produced a real in-function edge target
        addr_set = {ins['address'] for ins in ex.instructions}
        for ins in ex.instructions:
            op = ins['instruction'].split()[0].lower()
            if op in BRANCH_MNEMONICS:
                branch_total += 1
                tgt = ex.extract_jump_target(ins['instruction'])
                if tgt is not None and tgt in addr_set:
                    branch_resolved += 1

        if len(detailed) < args.samples and nb >= 4 and len(edges) >= nb:
            detailed.append((idx, r.get('function'), blocks, edges))

    ok = len(n_blocks)
    print(f"extracted ok:        {ok}/{n}  ({pct(ok, n)})")
    print(f"parse empty:         {parse_empty}")
    print(f"extractor errors:    {extract_err}")
    print()
    bmin, bmed, bmean, b95, bmax = quantiles(n_blocks)
    emin, emed, emean, e95, emax = quantiles(n_edges)
    print(f"blocks/row  min/med/mean/p95/max:  {bmin} / {bmed} / {bmean} / {b95} / {bmax}")
    print(f"edges/row   min/med/mean/p95/max:  {emin} / {emed} / {emean} / {e95} / {emax}")
    smin, smed, smean, s95, smax = quantiles(block_sizes)
    print(f"instr/block min/med/mean/p95/max:  {smin} / {smed} / {smean} / {s95} / {smax}")
    print()
    print(f"single-block rows:           {single_block}  ({pct(single_block, ok)})")
    print(f"  -> of which are BRANCHY:    {len(single_block_branchy)}   <-- must be ~0 (else dead-CFG bug)")
    print(f"rows w/ block likely >512tok: {big_block_rows}  ({pct(big_block_rows, ok)})")
    print(f"rows w/ bad edge indices:     {bad_index_rows}")
    print(f"rows w/ unreachable nodes:    {unreachable_rows}")
    print()
    print(f"intra-proc branch resolution: {branch_resolved}/{branch_total}  ({pct(branch_resolved, branch_total)})")
    print(f"edge-type histogram:          {dict(edge_types.most_common())}")
    print()
    if single_block_branchy[:8]:
        print("sample branchy-but-collapsed rows (idx, fn, #branch_instr):")
        for row in single_block_branchy[:8]:
            print("   ", row)
        print()

    for idx, fn, blocks, edges in detailed:
        print(f"--- sample CFG  row#{idx}  {fn}  ({len(blocks)} blocks, {len(edges)} edges) ---")
        for b in blocks[:12]:
            last = b.instructions[-1] if b.instructions else ''
            print(f"   block {b.id:>2} [{b.block_type:<11}] {b.instruction_count:>2} instr  | last: {last[:48]}")
        et = Counter(e.edge_type for e in edges)
        print(f"   edges: {[(e.source, e.target, e.edge_type) for e in edges[:14]]}")
        print(f"   edge-types: {dict(et)}\n")


if __name__ == '__main__':
    main()
