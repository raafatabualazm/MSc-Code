#!/usr/bin/env python3
"""Read-only: solved-set stability vs pass rate, across every sealed arm.

Turns a single turnover number into a relationship. If stability rises with the
solve count, turnover is a function of pass rate (a low-signal regime), not a
property of this benchmark. Either answer is publishable; they are different
papers.

Writes nothing. Run on the pod:  python3 turnover_all_arms.py
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import re
import statistics
from collections import defaultdict

ART = os.path.join(os.environ.get("WORKSPACE", "/workspace"), "artifacts")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def solved(path):
    return {
        str(row.get("task_id"))
        for row in (load(path).get("task_results") or [])
        if row.get("pass_at_k")
    }


def compiled(path):
    return {
        str(row.get("task_id"))
        for row in (load(path).get("task_results") or [])
        if row.get("compile_at_k")
    }


def jaccard(a, b):
    union = a | b
    return (len(a & b) / len(union)) if union else float("nan")


# ------------------------------------------------------- collect arms
arms = defaultdict(dict)
for path in sorted(glob.glob(os.path.join(ART, "*", "*_k10_score.json"))):
    name = os.path.basename(path)
    match = re.match(r"(.+?)_seed(\d+)_k10_score\.json$", name)
    if match:
        arms[match.group(1)][int(match.group(2))] = path
    elif name == "two_epoch_k10_score.json":
        arms["baseline"][42] = path

# fold the seed-42 baseline alias into the baseline arm
if "baseline" in arms:
    for other in list(arms):
        if other != "baseline" and other.startswith("baseline"):
            arms["baseline"].update(arms.pop(other))

print("=" * 78)
print("SOLVED-SET STABILITY vs PASS RATE  (same arm, same input, seed differs only)")
print("=" * 78)

rows = []
for arm in sorted(arms):
    seeds = arms[arm]
    if len(seeds) < 2:
        continue
    sets = {s: solved(p) for s, p in sorted(seeds.items())}
    counts = [len(v) for v in sets.values()]
    pairs = list(itertools.combinations(sorted(sets), 2))
    js = [jaccard(sets[a], sets[b]) for a, b in pairs]
    shared = [len(sets[a] & sets[b]) for a, b in pairs]

    # stable core = tasks solved in EVERY seed; rotating = solved in exactly one
    everywhere = set.intersection(*sets.values()) if sets else set()
    anywhere = set.union(*sets.values()) if sets else set()
    once = {t for t in anywhere if sum(t in v for v in sets.values()) == 1}

    rows.append((arm, len(seeds), counts, js, everywhere, anywhere, once))

    print("\n%s   (%d seeds)" % (arm, len(seeds)))
    print("  solved per seed : %s   mean %.1f  SD %s" % (
        counts, statistics.mean(counts),
        "%.2f" % statistics.stdev(counts) if len(counts) > 1 else "n/a"))
    print("  mean jaccard    : %.3f   (pairs %d, shared %s)" % (
        statistics.mean(js), len(pairs), shared))
    print("  solved in ALL seeds : %-3d   in ANY : %-3d   in exactly ONE : %d"
          % (len(everywhere), len(anywhere), len(once)))
    if anywhere:
        print("  -> %.0f%% of the union is seed-specific (solved by one run only)"
              % (100.0 * len(once) / len(anywhere)))

# ------------------------------------------------------- compilation contrast
print()
print("=" * 78)
print("SAME ANALYSIS ON COMPILATION  (the high-rate control)")
print("=" * 78)
for arm in sorted(arms):
    seeds = arms[arm]
    if len(seeds) < 2:
        continue
    sets = {s: compiled(p) for s, p in sorted(seeds.items())}
    js = [jaccard(sets[a], sets[b])
          for a, b in itertools.combinations(sorted(sets), 2)]
    counts = [len(v) for v in sets.values()]
    print("  %-34s compile mean %6.1f   jaccard %.3f"
          % (arm, statistics.mean(counts), statistics.mean(js)))

# ------------------------------------------------------- the relationship
print()
print("=" * 78)
print("THE RELATIONSHIP")
print("=" * 78)
print("  %-34s %8s %10s" % ("arm", "mean n", "jaccard"))
for arm, _, counts, js, _, _, _ in sorted(rows, key=lambda r: statistics.mean(r[2])):
    print("  %-34s %8.1f %10.3f" % (arm, statistics.mean(counts), statistics.mean(js)))
print("""
  If jaccard rises with mean n, solved-set instability is a LOW-SIGNAL-REGIME
  effect: it indicts pass@k reporting at low absolute rates, which is where this
  whole subfield operates. If jaccard stays flat and low even for the strongest
  arm, it indicts the benchmark itself. Compare against the compilation block
  above, which runs at ~70-100% and should be near-stable if the mechanism is
  rate-driven.
""")
