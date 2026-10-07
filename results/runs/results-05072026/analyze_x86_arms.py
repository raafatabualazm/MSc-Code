"""Paired significance (McNemar) + vote rerank across the x86 ablation arms.
Runs entirely off the downloaded pass_stats.csv (per-candidate compile/pass
flags) + pass_predictions.json (candidate texts). No GPU, no Dart."""
import csv
import json
import math
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128"
ARMS = [
    ("R  base",        "x86_ref_base"),
    ("G1 text-only",   "x86_g1_textonly"),
    ("G0 null",        "x86_g0_null"),
    ("G2 graph+text",  "x86_g2_graphtext"),
    ("G2c cfg+text",   "x86_g2c_cfgonly"),
    ("p128 (diverged)","x86_g3_p128"),
    ("p128r wide",     "x86_g3_p128r"),
    ("G3 graph-only",  "x86_g3_graphonly"),
]
N_CAND = 10


def load_arm(suffix):
    csv_path = os.path.join(HERE, "sweeps_antigravity", f"{BASE}_{suffix}_pass_stats.csv")
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = int(r["problem_id"])
            comp = [int(float(r[f"cand_{i}_compile"])) for i in range(1, N_CAND + 1)]
            pas = [int(float(r[f"cand_{i}_pass"])) for i in range(1, N_CAND + 1)]
            rows[pid] = {"compile": comp, "pass": pas}
    pred_path = os.path.join(HERE, f"{BASE}_{suffix}_pass_predictions.json")
    preds = json.load(open(pred_path, encoding="utf-8"))
    for i, item in enumerate(preds):
        if i in rows:
            rows[i]["texts"] = item.get("predictions", [])
    return rows


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip()).lower()


def realized(rows):
    ids = sorted(rows)
    n = len(ids)
    pass_any = [1 if any(rows[i]["pass"]) else 0 for i in ids]      # pass@10
    pass1 = [rows[i]["pass"][0] for i in ids]                        # pass@1 (cand_1)
    comp_any = [1 if any(rows[i]["compile"]) else 0 for i in ids]   # compile@10
    comp1 = [rows[i]["compile"][0] for i in ids]
    return {
        "n": n, "any_pass": pass_any,
        "pass_at_1": sum(pass1) / n, "pass_at_10": sum(pass_any) / n,
        "compile_at_1": sum(comp1) / n, "compile_at_10": sum(comp_any) / n,
        "solved": sum(pass_any),
    }


def mcnemar_exact(a_flags, b_flags):
    """Two-sided exact McNemar over paired binary outcomes."""
    b = sum(1 for x, y in zip(a_flags, b_flags) if x == 1 and y == 0)  # A wins
    c = sum(1 for x, y in zip(a_flags, b_flags) if x == 0 and y == 1)  # B wins
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * tail)
    return b, c, p


def select_pass(rows, mode):
    """Return realized pass@1 under a selection rule over the 10 candidates."""
    ids = sorted(rows)
    hits = 0
    for i in ids:
        comp, pas, texts = rows[i]["compile"], rows[i]["pass"], rows[i].get("texts", [])
        pick = 0  # default: first candidate
        if mode == "first":
            pick = 0
        elif mode == "stats_compile":
            pick = next((j for j in range(N_CAND) if comp[j]), 0)
        elif mode == "stats_compile_vote":
            comp_idx = [j for j in range(N_CAND) if comp[j] and j < len(texts)]
            if comp_idx:
                clusters = defaultdict(list)
                for j in comp_idx:
                    clusters[norm(texts[j])].append(j)
                best = max(clusters.values(), key=len)
                pick = best[0]
            else:
                pick = 0
        hits += pas[pick] if pick < len(pas) else 0
    return hits / len(ids)


data = {name: load_arm(sfx) for name, sfx in ARMS}
stats = {name: realized(rows) for name, rows in data.items()}

print("=" * 92)
print(f"{'arm':18s} {'pass@1':>7s} {'pass@10':>8s} {'solved/154':>11s} "
      f"{'compile@1':>10s} {'compile@10':>11s}")
print("-" * 92)
for name, _ in ARMS:
    s = stats[name]
    print(f"{name:18s} {s['pass_at_1']:7.3f} {s['pass_at_10']:8.3f} "
          f"{s['solved']:>7d}/154   {s['compile_at_1']:10.3f} {s['compile_at_10']:11.3f}")

print("\n" + "=" * 92)
print("PAIRED McNemar (exact, two-sided) on per-task any-pass (pass@10) flags")
print("-" * 92)
pairs = [
    ("G3 graph-only", "G0 null",         "graph carries assembly info?"),
    ("G3 graph-only", "G1 text-only",    "graph vs raw text"),
    ("G3 graph-only", "R  base",         "graph-FT vs untrained base"),
    ("G3 graph-only", "G2 graph+text",   "graph-only vs graph+text (text harmful?)"),
    ("G3 graph-only", "p128r wide",      "width helps?"),
    ("G2 graph+text", "G2c cfg+text",    "DFG edges add pass?"),
    ("G0 null",       "G1 text-only",    "adding text to null (harmful?)"),
]
for a, b, q in pairs:
    fa = stats[a]["any_pass"]
    fb = stats[b]["any_pass"]
    bw, cw, p = mcnemar_exact(fa, fb)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
    print(f"{a:16s} vs {b:16s} | only-A {bw:2d}  only-B {cw:2d}  p={p:.3f} {sig:3s} | {q}")

print("\n" + "=" * 92)
print("SELECTED pass@1 (rerank the 10-candidate pool to one output)")
print("-" * 92)
print(f"{'arm':18s} {'first':>7s} {'compile':>8s} {'comp+vote':>10s}  (raw pass@1 for ref)")
print("-" * 92)
for name, _ in ARMS:
    rows = data[name]
    print(f"{name:18s} {select_pass(rows,'first'):7.3f} "
          f"{select_pass(rows,'stats_compile'):8.3f} "
          f"{select_pass(rows,'stats_compile_vote'):10.3f}   ({stats[name]['pass_at_1']:.3f})")
