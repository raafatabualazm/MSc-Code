
from __future__ import annotations

import sys

import torch
from torch_geometric.data import Data

# Edge-type vocabulary for the GNN edge embedding (num_edge_types=8).
# 'call', 'runtime_stub', and 'error_path' are never emitted by
# scripts/data/cfg_extractor.py; 'dataflow' (cross-block def-use edges,
# GRAPH_DFG_MODE=edges) reuses the untrained 'call' slot so the embedding
# table keeps shape (8, hidden) and older checkpoints still load.
EDGE_TYPE_TO_IDX = {
    'linear_fallthrough': 0,
    'conditional_true': 1,
    'conditional_false': 2,
    'unconditional_jump': 3,
    'loop_backedge': 4,
    'call': 5,
    'dataflow': 5,
    'runtime_stub': 6,
    'error_path': 7,
}

_WARNED_UNKNOWN_EDGE_TYPES = set()


def cfg_to_pyg(record: dict, node_embeddings: torch.Tensor):
    edge_index = []
    edge_attr = []

    for edge in record['edges']:
        edge_type = edge['edge_type']
        type_index = EDGE_TYPE_TO_IDX.get(edge_type)
        if type_index is None:
            # An unknown type means a producer/consumer contract drift; falling
            # back silently is how dead-graph regressions hide, so say it once.
            if edge_type not in _WARNED_UNKNOWN_EDGE_TYPES:
                _WARNED_UNKNOWN_EDGE_TYPES.add(edge_type)
                print(
                    f"[cfg_to_pyg] WARNING: unknown edge_type {edge_type!r}; "
                    f"mapping to linear_fallthrough(0). Known: {sorted(EDGE_TYPE_TO_IDX)}",
                    file=sys.stderr,
                )
            type_index = 0
        edge_index.append([edge['source'], edge['target']])
        edge_attr.append(type_index)

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(
        x=node_embeddings,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
