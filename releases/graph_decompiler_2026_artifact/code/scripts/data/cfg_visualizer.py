
"""
CFG visualization utilities.
"""

from __future__ import annotations

import json
import networkx as nx
import matplotlib.pyplot as plt


def render_cfg(record: dict, output_path: str):
    graph = nx.DiGraph()

    for block in record['cfg']:
        graph.add_node(
            block['id'],
            label=f"{block['block_type']}\n{block['instruction_count']} ins"
        )

    for edge in record['edges']:
        graph.add_edge(
            edge['source'],
            edge['target'],
            edge_type=edge['edge_type'],
        )

    positions = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(graph, positions, node_size=1800)
    nx.draw_networkx_labels(graph, positions)
    nx.draw_networkx_edges(graph, positions, arrows=True)

    edge_labels = {
        (u, v): d['edge_type']
        for u, v, d in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as handle:
        first = json.loads(handle.readline())

    render_cfg(first, args.output)
