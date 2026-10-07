
from __future__ import annotations

import argparse
import json


def pseudo_pass(reference: str, prediction: str) -> bool:
    reference_tokens = set(reference.lower().split())
    prediction_tokens = set(prediction.lower().split())

    if not reference_tokens:
        return False

    overlap = len(reference_tokens & prediction_tokens) / len(reference_tokens)
    return overlap >= 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()

    rows = json.loads(open(args.predictions, 'r', encoding='utf-8').read())

    passed = 0

    for row in rows:
        if pseudo_pass(row['reference'], row['prediction']):
            passed += 1

    total = max(len(rows), 1)

    print(json.dumps({
        'pass_at_1': passed / total,
        'passed': passed,
        'total': total,
    }, indent=2))


if __name__ == '__main__':
    main()
