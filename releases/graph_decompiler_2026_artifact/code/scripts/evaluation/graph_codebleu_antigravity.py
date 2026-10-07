"""
Graph-aware decompiler CodeBLEU evaluation script (Antigravity version).
Computes maximum CodeBLEU score across multiple candidates.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from codebleu import CodeBLEUCalculator
from graph_compile_at_k_antigravity import compile_dart


_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--compiled_only', action='store_true',
                        help="Match old compiled CodeBLEU: score only candidates that pass compile@k harness")
    parser.add_argument('--workers', type=int, default=1,
                        help="Parallel row workers for compiled-only CodeBLEU compile filtering")
    args = parser.parse_args()

    rows = json.loads(open(args.predictions, 'r', encoding='utf-8').read())

    calculators = {
        'dart': CodeBLEUCalculator('dart'),
        'swift': CodeBLEUCalculator('swift'),
    }

    scores = []
    samples_with_success = 0

    workers = max(1, int(args.workers))

    def score_row(row: dict) -> tuple[bool, float]:
        lang = row.get('language', 'dart').lower()
        calc = calculators.get(lang, calculators['dart'])
        reference = row.get('reference', '')

        candidates = row.get('predictions', [row.get('prediction', '')])
        if not candidates:
            candidates = [row.get('prediction', '')]

        candidate_scores = []
        for cand in candidates:
            code = _extract_code(cand)
            if args.compiled_only and lang == 'dart' and not compile_dart(code):
                continue
            try:
                result = calc.compute_codebleu(reference, code)
                candidate_scores.append(result['codebleu'])
            except Exception:
                candidate_scores.append(0.0)

        if candidate_scores:
            return True, max(candidate_scores)
        if not args.compiled_only:
            return True, 0.0
        return False, 0.0

    if args.compiled_only and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            row_scores = list(pool.map(score_row, rows))
    else:
        row_scores = [score_row(row) for row in rows]

    for ok, score in row_scores:
        if ok:
            samples_with_success += 1
            scores.append(score)

    print(json.dumps({
        'count': len(rows),
        'samples_with_success': samples_with_success if args.compiled_only else len(scores),
        'mean_codebleu': statistics.mean(scores) if scores else 0.0,
        'min_codebleu': min(scores) if scores else 0.0,
        'max_codebleu': max(scores) if scores else 0.0,
    }, indent=2))


if __name__ == '__main__':
    main()
