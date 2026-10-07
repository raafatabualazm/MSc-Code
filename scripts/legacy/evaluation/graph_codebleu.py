
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

        result = calc.compute_codebleu(
            row['reference'],
            row['prediction'],
        )

        score = result['codebleu']
        scores.append(score)

    print(json.dumps({
        'count': len(scores),
        'mean_codebleu': statistics.mean(scores) if scores else 0.0,
        'min_codebleu': min(scores) if scores else 0.0,
        'max_codebleu': max(scores) if scores else 0.0,
    }, indent=2))


if __name__ == '__main__':
    main()
