"""
Graph-aware decompiler pass@k evaluation script (Antigravity version).
Computes unbiased pass@k for multiple candidate predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pseudo_pass(reference: str, prediction: str) -> bool:
    reference_tokens = set(reference.lower().split())
    prediction_tokens = set(prediction.lower().split())

    if not reference_tokens:
        return False

    overlap = len(reference_tokens & prediction_tokens) / len(reference_tokens)
    return overlap >= 0.3


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    # Compute C(n-c, k) / C(n, k)
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--k_values', default='1,5')
    args = parser.parse_args()

    rows = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    k_list = [int(x.strip()) for x in args.k_values.split(',')]

    pass_sums = {k: 0.0 for k in k_list}
    total = len(rows)

    for idx, row in enumerate(rows):
        reference = row.get('reference', '')
        candidates = row.get('predictions', [row.get('prediction', '')])

        if not candidates:
            candidates = [row.get('prediction', '')]

        n = len(candidates)
        c = 0

        for cand in candidates:
            if pseudo_pass(reference, cand):
                c += 1

        for k in k_list:
            if n >= k:
                val = pass_at_k_estimator(n, c, k)
                pass_sums[k] += val
            else:
                val = pass_at_k_estimator(n, c, n)
                pass_sums[k] += val

        print(f'[{idx + 1}/{len(rows)}] n={n}, passed={c}, pass@1={c/n:.4f}')

    results = {}
    for k in k_list:
        results[f'pass_at_{k}'] = pass_sums[k] / max(total, 1)

    results['total_problems'] = total
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
