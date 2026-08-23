"""Pairwise overlap (McNemar) + union coverage across the x86 arms.
Answers: do simko-GRPO / style1036-SFT solve tasks G3 misses? What is the
total solvable-task ceiling available to harvest (RS-SFT) or ensemble?"""
import csv, math, os
BASE = "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128"
ARMS = {
    "R_base": "x86_ref_base", "G1_text": "x86_g1_textonly", "G0_null": "x86_g0_null",
    "G2_gtext": "x86_g2_graphtext", "G2c_cfg": "x86_g2c_cfgonly",
    "G3_base": "x86_g3_graphonly", "p128r": "x86_g3_p128r",
    "simko_grpo": "x86_g3_simko_eval_grpo", "style1036": "x86_g3_style1036_sft",
}

def flags(sfx):
    p = f"sweeps_antigravity/{BASE}_{sfx}_pass_stats.csv"
    if not os.path.exists(p): return None
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        pid = int(r["problem_id"])
        out[pid] = 1 if any(int(float(r[f"cand_{i}_pass"])) for i in range(1, 11)) else 0
    return out

F = {k: flags(v) for k, v in ARMS.items()}
F = {k: v for k, v in F.items() if v is not None}
ids = sorted(set.intersection(*[set(v) for v in F.values()]))
print(f"arms loaded: {list(F)}  | tasks: {len(ids)}\n")

def solved(k): return {i for i in ids if F[k][i]}

print("standalone solved:")
for k in F: print(f"  {k:12s} {len(solved(k)):3d}")

def mcnemar(a, b):
    A, B = solved(a), solved(b)
    only_a, only_b = len(A - B), len(B - A)
    n = only_a + only_b
    if n == 0: return only_a, only_b, 1.0
    kk = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(kk + 1)) * 0.5 ** n)
    return only_a, only_b, p

print("\npaired vs G3_base (McNemar):")
for k in ["simko_grpo", "style1036", "p128r"]:
    if k in F:
        oa, ob, p = mcnemar("G3_base", k)
        print(f"  G3 vs {k:12s} G3-only {oa:2d}  {k}-only {ob:2d}  p={p:.3f}  "
              f"({k} adds {ob} tasks G3 misses)")

print("\nunion coverage (harvest / ensemble ceiling):")
def U(*ks):
    s = set()
    for k in ks: s |= solved(k)
    return len(s)
print(f"  G3 alone                       {U('G3_base')}")
print(f"  G3 + style1036                 {U('G3_base','style1036')}")
print(f"  G3 + simko                     {U('G3_base','simko_grpo')}")
print(f"  G3 + style1036 + simko         {U('G3_base','style1036','simko_grpo')}")
print(f"  ALL x86 arms union             {U(*F.keys())}")
allsolved = set()
for k in F: allsolved |= solved(k)
never = [i for i in ids if i not in allsolved]
print(f"  tasks NO arm ever solves       {len(never)}/{len(ids)}  (the hard floor)")

# which tasks does each add over G3
for k in ["style1036", "simko_grpo"]:
    if k in F:
        add = sorted(solved(k) - solved("G3_base"))
        print(f"\n  {k} solves {len(add)} tasks G3 misses: {add}")
