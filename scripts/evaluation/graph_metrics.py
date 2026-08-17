
from __future__ import annotations


def cfg_similarity(reference_edges, predicted_edges):
    reference = set((e['source'], e['target'], e['edge_type']) for e in reference_edges)
    predicted = set((e['source'], e['target'], e['edge_type']) for e in predicted_edges)

    if not reference:
        return 0.0

    return len(reference & predicted) / len(reference)


def dfg_similarity(reference_edges, predicted_edges):
    reference = set((e['source'], e['target']) for e in reference_edges)
    predicted = set((e['source'], e['target']) for e in predicted_edges)

    if not reference:
        return 0.0

    return len(reference & predicted) / len(reference)
