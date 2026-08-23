
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

REGION_CHAIN_EDGE_TYPES = {
    'linear_fallthrough',
    'unconditional_jump',
}


def build_linear_region_ids(
    node_count: int,
    edges: list[dict],
    max_region_blocks: int = 8,
    block_types: list[str] | None = None,
) -> torch.Tensor:
    """Partition a CFG into bounded maximal straight-line regions.

    Data-flow edges do not define control regions. A CFG edge can join two
    blocks only when it is the source's sole control successor and the
    target's sole control predecessor. Branches, joins, calls, backedges, and
    exceptional edges therefore form explicit region boundaries.
    """
    if max_region_blocks < 1:
        raise ValueError(
            f"max_region_blocks must be positive, got {max_region_blocks}"
        )
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    if node_count == 0:
        return torch.empty((0,), dtype=torch.long)
    if block_types is not None and len(block_types) != node_count:
        raise ValueError(
            f"Expected {node_count} block types, received {len(block_types)}"
        )

    structural_edges = [
        edge for edge in edges
        if edge.get('edge_type') != 'dataflow'
        and not str(edge.get('edge_type', '')).startswith('reverse_')
    ]
    outgoing = [[] for _ in range(node_count)]
    incoming = [[] for _ in range(node_count)]
    for edge in structural_edges:
        source = edge.get('source')
        target = edge.get('target')
        if (
            not isinstance(source, int)
            or not isinstance(target, int)
            or not (0 <= source < node_count)
            or not (0 <= target < node_count)
        ):
            raise ValueError(
                f"Cannot construct regions from malformed edge: {edge!r}"
            )
        outgoing[source].append(edge)
        incoming[target].append(edge)

    chain_next: dict[int, int] = {}
    chain_previous: dict[int, int] = {}
    for edge in structural_edges:
        source = edge['source']
        target = edge['target']
        if (
            source != target
            and edge.get('edge_type') in REGION_CHAIN_EDGE_TYPES
            and len(outgoing[source]) == 1
            and len(incoming[target]) == 1
            and (
                block_types is None
                or block_types[source] not in {'call', 'runtime_stub', 'return', 'trap'}
            )
        ):
            chain_next[source] = target
            chain_previous[target] = source

    region_ids = torch.full((node_count,), -1, dtype=torch.long)
    next_region = 0

    def assign_chain(start: int) -> None:
        nonlocal next_region
        current = start
        while current is not None and region_ids[current].item() < 0:
            assigned = 0
            while region_ids[current].item() < 0 and assigned < max_region_blocks:
                region_ids[current] = next_region
                assigned += 1
                successor = chain_next.get(current)
                if successor is None or region_ids[successor].item() >= 0:
                    current = None
                    break
                current = successor
            next_region += 1

    # Canonical starts make acyclic regions deterministic. The second pass
    # handles max-size splits and the unlikely case of a closed jump cycle.
    for node in range(node_count):
        if node not in chain_previous and region_ids[node].item() < 0:
            assign_chain(node)
    for node in range(node_count):
        if region_ids[node].item() < 0:
            assign_chain(node)

    return region_ids

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

    cfg_blocks = record.get('cfg')
    block_types = None
    if cfg_blocks is not None:
        block_types = [str(block.get('block_type', '')) for block in cfg_blocks]
    region_ids = build_linear_region_ids(
        node_count,
        edges,
        max_region_blocks=int(os.environ.get('GRAPH_REGION_MAX_BLOCKS', '8')),
        block_types=block_types,
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
        region_id=region_ids,
    )
