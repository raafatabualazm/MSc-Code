"""
Fit a beta-binomial pass@k scaling model (Levi 2024 / Kazdan et al. 2510.05197)
to per-problem success counts (c_i out of n_i samples), and extrapolate pass@k.

Input: JSON from graph_pass_at_k_antigravity.py (list of per-task records with
n and passed/c). Robust to a few key spellings.

Model: pass_i@1 ~ Beta(alpha, beta). Then
    pass_D@k = 1 - E_p[(1-p)^k] = 1 - B(alpha, beta+k)/B(alpha, beta)
Empirical (Chen et al. 2021 unbiased): pass@k = mean_i [1 - C(n_i-c_i,k)/C(n_i,k)].
"""
import json, sys, math
from math import lgamma

def betaln(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)

def load_counts(path):
    raw = open(path, encoding="utf-8").read()
    # graph_pass_at_k mixes progress lines + a trailing JSON blob on stdout.
    # Try clean JSON first; else parse the 'n=.., passed=..' progress lines.
    try:
        data = json.loads(raw)
    except Exception:
        import re
        pairs = re.findall(r"n=(\d+),\s*passed=(\d+)", raw)
        if pairs:
            return [(int(c), int(n)) for n, c in pairs]
        # last resort: extract the trailing JSON array/object
        for opener in ("[", "{"):
            idx = raw.rfind("\n" + opener)
            if idx != -1:
                try:
                    data = json.loads(raw[idx+1:]); break
                except Exception:
                    continue
        else:
            raise SystemExit("could not parse counts from %s" % path)
    # graph_pass_at_k dumps a dict or list; find per-task records
    rows = None
    if isinstance(data, dict):
        for key in ("per_task", "tasks", "results", "rows"):
            if isinstance(data.get(key), list):
                rows = data[key]; break
        if rows is None and "n" in data:  # single record edge case
            rows = [data]
    elif isinstance(data, list):
        rows = data
    if rows is None:
        raise SystemExit("could not locate per-task list in JSON; keys=%s" % (list(data.keys()) if isinstance(data, dict) else type(data)))
    counts = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        n = r.get("n") or r.get("num_samples") or r.get("samples")
        c = r.get("passed", r.get("c", r.get("successes", r.get("num_pass"))))
        if n is None or c is None:
            continue
        counts.append((int(c), int(n)))
    if not counts:
        raise SystemExit("no (c,n) pairs parsed; sample row=%s" % (rows[0] if rows else None))
    return counts

def empirical_pass_at_k(counts, k):
    # unbiased estimator, mean over tasks; requires k <= n_i
    tot = 0.0; m = 0
    for c, n in counts:
        if k > n:
            continue
        # 1 - C(n-c,k)/C(n,k)
        if c == 0:
            val = 0.0
        elif n - c < k:
            val = 1.0
        else:
            lognum = lgamma(n-c+1)-lgamma(n-c-k+1)
            logden = lgamma(n+1)-lgamma(n-k+1)
            val = 1.0 - math.exp(lognum - logden)
        tot += val; m += 1
    return tot / m if m else float("nan")

def bb_negloglik(alpha, beta, counts):
    ll = 0.0
    for c, n in counts:
        # log C(n,c) + betaln(c+alpha, n-c+beta) - betaln(alpha,beta)
        logC = lgamma(n+1) - lgamma(c+1) - lgamma(n-c+1)
        ll += logC + betaln(c+alpha, n-c+beta) - betaln(alpha, beta)
    return -ll

def fit_bb(counts):
    # coarse log-grid then local refine (no scipy dependency)
    best = None
    grid = [10**e for e in [x/4 for x in range(-24, 13)]]  # 1e-6 .. ~1e3
    for a in grid:
        for b in grid:
            nll = bb_negloglik(a, b, counts)
            if best is None or nll < best[0]:
                best = (nll, a, b)
    # refine: coordinate descent, shrinking multiplicative steps
    nll, a, b = best
    step = 2.0
    for _ in range(60):
        improved = False
        for (da, db) in [(step,1),(1/step,1),(1,step),(1,1/step)]:
            na, nb = a*da, b*db
            if na <= 0 or nb <= 0:
                continue
            v = bb_negloglik(na, nb, counts)
            if v < nll:
                nll, a, b = v, na, nb; improved = True
        if not improved:
            step = step ** 0.5
            if step < 1.0001:
                break
    return a, b, nll

def pred_pass_at_k(alpha, beta, k):
    return 1.0 - math.exp(betaln(alpha, beta+k) - betaln(alpha, beta))

def main():
    path = sys.argv[1]
    counts = load_counts(path)
    n_tasks = len(counts)
    n_samp = counts[0][1]
    zeros = sum(1 for c, n in counts if c == 0)
    maxc = max(c for c, n in counts)
    total_pass = sum(c for c, n in counts)
    total_draw = sum(n for c, n in counts)
    print("="*70)
    print("BETA-BINOMIAL pass@k SCALING  (%d tasks, %d samples each)" % (n_tasks, n_samp))
    print("  tasks with 0 successes : %d / %d (%.1f%%)" % (zeros, n_tasks, 100*zeros/n_tasks))
    print("  max successes on a task: %d / %d" % (maxc, n_samp))
    print("  pooled pass_1 estimate : %d/%d = %.4f%%" % (total_pass, total_draw, 100*total_pass/total_draw))
    print("="*70)
    print("EMPIRICAL pass@k (Chen unbiased, k<=%d):" % n_samp)
    for k in [1, 2, 5, 10]:
        if k <= n_samp:
            print("  pass@%-4d = %.4f%%" % (k, 100*empirical_pass_at_k(counts, k)))
    a, b, nll = fit_bb(counts)
    print("-"*70)
    print("FIT: alpha=%.4g  beta=%.4g   (neg-loglik=%.2f)" % (a, b, nll))
    print("  mean pass_1 under fit  = alpha/(alpha+beta) = %.5f%%" % (100*a/(a+b)))
    print("EXTRAPOLATED pass@k (1 - B(a,b+k)/B(a,b)):")
    for k in [1, 10, 100, 1000, 10000, 100000]:
        print("  pass@%-7d = %.3f%%" % (k, 100*pred_pass_at_k(a, b, k)))
    print("-"*70)
    # honest diagnosis
    print("DIAGNOSIS:")
    if maxc == 0:
        print("  ALL tasks 0/%d -> hard-zero at this budget. alpha unidentifiable;" % n_samp)
        print("  the fit's high-k prediction is an EXTRAPOLATION UNSUPPORTED by data.")
        print("  A targeted high-sample run is required to distinguish hard-zero from tiny-p.")
    elif zeros >= n_tasks - 3:
        print("  Only %d task(s) ever pass -> fit driven by a handful of points;" % (n_tasks-zeros))
        print("  high-k extrapolation is WEAK. Treat as order-of-magnitude, verify with")
        print("  a targeted high-sample (>=100) run on the hard tail (Kazdan Thm 1).")
    else:
        print("  Enough signal to fit; extrapolation is indicative but validate at k=100.")

if __name__ == "__main__":
    main()
