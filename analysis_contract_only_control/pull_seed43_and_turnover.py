#!/usr/bin/env python3
"""Read-only: pull the missing seed-43 fields, and run the baseline-turnover control.

Writes nothing. Touches no sealed artifact. Run on the pod:

    python3 pull_seed43_and_turnover.py

Part 1 reports what the two-seed report already contains but was not relayed:
seed-43 diversity and both paired McNemar contrasts.

Part 2 is the control that decides whether ANY task-level statement about this
arm is admissible. Contract-only and baseline solved sets share 1 task of 11 at
seed 42. That is only interpretable against how much the baseline's own solved
set turns over when nothing changes but the sampling seed. If baseline-vs-
baseline overlap across seeds is as low as contract-only-vs-baseline overlap,
the disjointness carries no channel information and every solved-set claim has
to be withdrawn.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import statistics

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
ART = os.path.join(WORKSPACE, "artifacts")

REPORT = os.path.join(
    ART, "t5gemma2_contract_only_control_v1", "contract_only_two_seed_report.json"
)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def solved(score_path):
    """Task ids with pass_at_k true, from a sealed score file."""
    data = load(score_path)
    return {
        str(row.get("task_id"))
        for row in (data.get("task_results") or [])
        if row.get("pass_at_k")
    }


def jaccard(a, b):
    union = a | b
    return (len(a & b) / len(union)) if union else float("nan")


# ---------------------------------------------------------------- part 1
print("=" * 72)
print("PART 1 - fields present in the sealed report but not yet relayed")
print("=" * 72)

report = load(REPORT)
seeds = report["typed_contract_only"]["seeds"]

for seed in ("42", "43"):
    arm = seeds[seed]
    div = arm["diversity"]
    print("\nseed %s" % seed)
    print("  pass@10 %-4d compile@10 %-4d pass@1 %d" % (
        arm["metrics"]["pass_at_k"]["count"],
        arm["metrics"]["compile_at_k"]["count"],
        arm["metrics"]["pass_at_1"]["count"],
    ))
    print("  distinct/10 mean %.3f   tasks below k: %d" % (
        div["mean_distinct_per_k"], div["tasks_below_k"]))
    for label, key in (
        ("vs baseline ", "paired_vs_same_seed_baseline"),
        ("vs typed+F2", "paired_vs_same_seed_typed_plus_f2"),
    ):
        for metric in ("pass_at_k", "compile_at_k"):
            cell = arm[key][metric]
            print("  %s %-12s control-only %-3d comparator %-3d  p=%.4f" % (
                label, metric,
                cell["left_only"], cell["right_only"],
                cell["exact_two_sided_p"]))

gate = report["preregistered_interpretation_gate"]
print("\ninterpretation class : %s" % gate["interpretation_class"])
print("defensible claim     : %s" % gate["defensible_claim"])
print("compile>=160 each    : %s" % gate["compile_at_10_at_least_160_each_seed"])
print("pass<=3 each         : %s" % gate["pass_at_10_at_most_3_each_seed"])
print("typed McNemar sig    : %s"
      % gate["typed_plus_f2_pass_at_10_mcnemar_p_le_0_05_each_seed"])

dist = report["typed_contract_only"]["count_distributions"]
print("\ntwo-seed pass@10  : %s  mean %.1f" % (
    dist["pass_at_k"]["values"], dist["pass_at_k"]["mean"]))
print("two-seed compile  : %s  mean %.1f" % (
    dist["compile_at_k"]["values"], dist["compile_at_k"]["mean"]))

# ---------------------------------------------------------------- part 2
print()
print("=" * 72)
print("PART 2 - baseline turnover control (the null for solved-set disjointness)")
print("=" * 72)

candidates = []
for pattern in (
    os.path.join(ART, "t5gemma2_sft_epoch_ablation_passk_v1", "two_epoch_k10_score.json"),
    os.path.join(ART, "t5gemma2_f2_measurement_audit_v1", "baseline_seed*_k10_score.json"),
    os.path.join(ART, "t5gemma2_f2_intervention_multiseed_v1", "baseline_seed*_k10_score.json"),
):
    candidates.extend(sorted(glob.glob(pattern)))

baseline = {}
for path in candidates:
    name = os.path.basename(path)
    seed = 42 if "two_epoch" in name else int(name.split("seed")[1].split("_")[0])
    baseline.setdefault(seed, path)

print("\nbaseline score files found:")
for seed in sorted(baseline):
    print("  seed %-3d %s" % (seed, baseline[seed]))

if len(baseline) < 2:
    print("\nFEWER THAN TWO BASELINE SEEDS FOUND - adjust the glob and rerun.")
    raise SystemExit(1)

sets = {seed: solved(path) for seed, path in sorted(baseline.items())}
print("\nbaseline solved counts: %s"
      % {seed: len(value) for seed, value in sets.items()})

print("\npairwise baseline overlap (same arm, same input, seed differs only):")
overlaps = []
for left, right in itertools.combinations(sorted(sets), 2):
    a, b = sets[left], sets[right]
    j = jaccard(a, b)
    overlaps.append(j)
    print("  %d vs %d : shared %-3d union %-3d  jaccard %.3f"
          % (left, right, len(a & b), len(a | b), j))

print("\nbaseline self-consistency: mean jaccard %.3f  (n=%d pairs)"
      % (statistics.mean(overlaps), len(overlaps)))

# contract-only vs same-seed baseline, for direct comparison
ctrl_dir = os.path.join(ART, "t5gemma2_contract_only_control_v1")
for seed in (42, 43):
    hits = sorted(glob.glob(os.path.join(ctrl_dir, "*seed%d*_k10_score.json" % seed)))
    if not hits or seed not in sets:
        continue
    c = solved(hits[0])
    b = sets[seed]
    print("\ncontract-only vs baseline, seed %d: shared %d union %d  jaccard %.3f"
          % (seed, len(c & b), len(c | b), jaccard(c, b)))

print("""
READ THIS BEFORE INTERPRETING:

  If mean baseline jaccard is comparable to the contract-only-vs-baseline
  jaccard, the solved sets turn over just as much when NOTHING changes but the
  seed. Disjointness then carries no channel information, and no task-level
  claim about this arm survives - including 'the channels solve different
  tasks' and 'adding the binary loses tasks types alone solved'.

  Only if the baseline is substantially self-consistent across seeds does
  cross-arm disjointness become interpretable.
""")
