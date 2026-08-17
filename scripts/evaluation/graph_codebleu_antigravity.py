"""
Graph-aware decompiler CodeBLEU evaluation script (Antigravity version).
Computes maximum CodeBLEU score across multiple candidates.
"""

from __future__ import annotations

import argparse
import json
import statistics
from codebleu import CodeBLEUCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()

    rows = json.loads(open(args.predictions, 'r', encoding='utf-8').read())

    calculators = {
        'dart': CodeBLEUCalculator('dart'),
        'swift': CodeBLEUCalculator('swift'),
    }

    scores = []

    for row in rows:
        lang = row.get('language', 'dart').lower()
        calc = calculators.get(lang, calculators['dart'])
        reference = row.get('reference', '')
        
        candidates = row.get('predictions', [row.get('prediction', '')])
        if not candidates:
            candidates = [row.get('prediction', '')]
            
        candidate_scores = []
        for cand in candidates:
            try:
                result = calc.compute_codebleu(reference, cand)
                candidate_scores.append(result['codebleu'])
            except Exception:
                candidate_scores.append(0.0)
                
        # Take the maximum CodeBLEU score among all candidates
        best_score = max(candidate_scores) if candidate_scores else 0.0
        scores.append(best_score)

    print(json.dumps({
        'count': len(scores),
        'mean_codebleu': statistics.mean(scores) if scores else 0.0,
        'min_codebleu': min(scores) if scores else 0.0,
        'max_codebleu': max(scores) if scores else 0.0,
    }, indent=2))


if __name__ == '__main__':
    main()
