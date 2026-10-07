
from __future__ import annotations

import json
from collections import Counter


def compute_cfg_statistics(path: str):
    block_counts = []
    edge_counter = Counter()
    block_types = Counter()

    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)

            block_counts.append(len(record['cfg']))

            for block in record['cfg']:
                block_types[block['block_type']] += 1

            for edge in record['edges']:
                edge_counter[edge['edge_type']] += 1

    print('Average blocks per function:', sum(block_counts) / len(block_counts))
    print('Block types:', dict(block_types))
    print('Edge types:', dict(edge_counter))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    compute_cfg_statistics(args.input)
