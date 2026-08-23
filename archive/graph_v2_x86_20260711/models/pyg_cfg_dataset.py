
from __future__ import annotations

import hashlib
import os
import random

import torch
from torch_geometric.data import Data

# Edge-type vocabulary for graph-v2 checkpoints. Dataflow has its own slot;
# aliasing it with `call` made future call edges silently indistinguishable.
EDGE_TYPE_TO_IDX = {
    'linear_fallthrough': 0,
    'conditional_true': 1,
    'conditional_false': 2,
    'unconditional_jump': 3,
    'loop_backedge': 4,
    'call': 5,
    'runtime_stub': 6,
    'error_path': 7,
    'dataflow': 8,
    'reverse_linear_fallthrough': 9,
    'reverse_conditional_true': 10,
    'reverse_conditional_false': 11,
    'reverse_unconditional_jump': 12,
    'reverse_loop_backedge': 13,
    'reverse_call': 14,
    'reverse_runtime_stub': 15,
    'reverse_error_path': 16,
    'reverse_dataflow': 17,
}

REVERSE_EDGE_TYPE = {
    edge_type: f'reverse_{edge_type}'
    for edge_type in (
        'linear_fallthrough',
        'conditional_true',
        'conditional_false',
        'unconditional_jump',
        'loop_backedge',
        'call',
        'runtime_stub',
        'error_path',
        'dataflow',
    )
}

def cfg_to_pyg(record: dict, node_embeddings: torch.Tensor):
    edge_index = []
    edge_attr = []

    edges = list(record.get('edges', []))
    node_count = int(node_embeddings.size(0))
    malformed = []
    for index, edge in enumerate(edges):
        source = edge.get('source')
        target = edge.get('target')
        if (
            not isinstance(source, int)
            or not isinstance(target, int)
            or not (0 <= source < node_count)
            or not (0 <= target < node_count)
        ):
            malformed.append((index, source, target, edge.get('edge_type')))
    if malformed:
        raise ValueError(
            f"CFG edge endpoints outside this graph's {node_count} nodes: "
            f"{malformed[:8]}{' ...' if len(malformed) > 8 else ''}"
        )
    ablation = os.environ.get('GRAPH_EDGE_ABLATION', 'full').strip().lower()
    if ablation == 'none':
        edges = []
    elif ablation == 'cfg':
        edges = [edge for edge in edges if edge.get('edge_type') != 'dataflow']
    elif ablation == 'dfg':
        edges = [edge for edge in edges if edge.get('edge_type') == 'dataflow']
    elif ablation == 'shuffle' and len(edges) > 1:
        # Deterministic per-graph target permutation. This preserves edge count
        # and edge-type marginals while destroying the original topology.
        signature = repr(
            [(edge.get('source'), edge.get('target'), edge.get('edge_type')) for edge in edges]
        )
        seed_text = f"{os.environ.get('GRAPH_SEED', '42')}|{signature}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode('utf-8')).digest()[:8], 'big')
        rng = random.Random(seed)
        targets = [edge.get('target') for edge in edges]
        rng.shuffle(targets)
        edges = [dict(edge, target=target) for edge, target in zip(edges, targets)]
    elif ablation not in {'', 'full'}:
        raise ValueError(
            f"Unknown GRAPH_EDGE_ABLATION={ablation!r}; use full, none, cfg, dfg, or shuffle"
        )

    if os.environ.get('GRAPH_ADD_REVERSE_EDGES', '0') == '1':
        reverse_edges = []
        for edge in edges:
            edge_type = edge.get('edge_type')
            reverse_type = REVERSE_EDGE_TYPE.get(edge_type)
            if reverse_type is None:
                raise ValueError(f"Cannot reverse unknown edge type {edge_type!r}")
            reverse_edges.append({
                **edge,
                'source': edge['target'],
                'target': edge['source'],
                'edge_type': reverse_type,
            })
        edges = edges + reverse_edges

    for edge in edges:
        edge_type = edge['edge_type']
        type_index = EDGE_TYPE_TO_IDX.get(edge_type)
        if type_index is None:
            raise ValueError(
                f"Unknown edge_type {edge_type!r}; known types: {sorted(EDGE_TYPE_TO_IDX)}"
            )
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
