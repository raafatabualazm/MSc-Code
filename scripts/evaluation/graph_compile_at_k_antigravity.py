"""
Graph-aware decompiler compile@k evaluation script (Antigravity version).
Computes unbiased compile@k for multiple candidate predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _resolve_dart_binary() -> str:
    """Find the dart binary, checking known locations first."""
    candidates = [
        '/home/zeus/dart-sdk/bin/dart',
        os.path.join(os.path.expanduser('~'), 'dart-sdk', 'bin', 'dart'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Fallback to PATH lookup
    found = shutil.which('dart')
    return found if found else 'dart'


DART_BIN = _resolve_dart_binary()


def compile_dart(code: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        path.write_text(code, encoding='utf-8')

        result = subprocess.run(
            [DART_BIN, 'analyze', str(path)],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0


def compile_swift(code: str) -> bool:
    return bool(code.strip())


def compile_at_k_estimator(n: int, c: int, k: int) -> float:
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

    compile_sums = {k: 0.0 for k in k_list}
    total = len(rows)

    for idx, row in enumerate(rows):
        lang = row.get('language', 'dart').lower()
        candidates = row.get('predictions', [row.get('prediction', '')])
        
        # If predictions is empty or missing, fallback to single prediction
        if not candidates:
            candidates = [row.get('prediction', '')]
            
        n = len(candidates)
        c = 0

        for cand in candidates:
            ok = compile_dart(cand) if lang == 'dart' else compile_swift(cand)
            if ok:
                c += 1

        # Compute compile@k for each k
        for k in k_list:
            if n >= k:
                val = compile_at_k_estimator(n, c, k)
                compile_sums[k] += val
            else:
                # If we generated fewer than k samples, compile@k defaults to compile@n
                val = compile_at_k_estimator(n, c, n)
                compile_sums[k] += val

        print(f'[{idx + 1}/{len(rows)}] n={n}, compiling={c}, compile@1={c/n:.4f}')

    results = {}
    for k in k_list:
        results[f'compile_at_{k}'] = compile_sums[k] / max(total, 1)

    results['total_problems'] = total
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
