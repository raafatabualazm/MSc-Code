
from __future__ import annotations

import torch
from torch_geometric.data import Data

EDGE_TYPE_TO_IDX = {
    'linear_fallthrough': 0,
    'conditional_true': 1,
    'conditional_false': 2,
    'unconditional_jump': 3,
    'loop_backedge': 4,
    'call': 5,
    'runtime_stub': 6,
    'error_path': 7,
}


def cfg_to_pyg(record: dict, node_embeddings: torch.Tensor):
    edge_index = []
    edge_attr = []

    for edge in record['edges']:
        edge_index.append([edge['source'], edge['target']])
        edge_attr.append(EDGE_TYPE_TO_IDX.get(edge['edge_type'], 0))

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
