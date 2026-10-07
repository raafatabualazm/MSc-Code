#!/usr/bin/env python3
"""Utilities for antigravity-compact-graph-v1.0 datasets.

The JSON representation stores one opcode record per instruction. Operands are
small typed records, while CFG and DFG topology remain sparse side arrays. The
model should not feed the JSON text to Qwen. Convert it to tensors and either:

1. pool operands into each instruction vector (recommended), or
2. use one opcode position plus one position per operand.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

K_REG = 0
K_IMM = 1
K_MEM = 2
K_SYMBOL = 3
K_BLOCK = 4
K_RAW = 5


def read_jsonl_gz(path: str | Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def load_vocab(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    if vocab.get("schema") != "antigravity-compact-graph-v1.0-vocab":
        raise ValueError("Unexpected compact graph vocabulary schema")
    return vocab


def validate_compact_graph(graph: Mapping[str, Any]) -> None:
    if graph.get("schema") != "antigravity-compact-graph-v1.0":
        raise ValueError("Unexpected compact graph schema")

    n = int(graph["n"])
    b = int(graph["b"])
    instructions = graph["i"]
    block_ptr = graph["bp"]
    block_types = graph["bt"]

    if len(instructions) != n:
        raise ValueError(f"Instruction count mismatch: n={n}, len(i)={len(instructions)}")
    if len(block_ptr) != b + 1 or block_ptr[0] != 0 or block_ptr[-1] != n:
        raise ValueError("Invalid block pointer array")
    if any(block_ptr[i] > block_ptr[i + 1] for i in range(len(block_ptr) - 1)):
        raise ValueError("Block pointers are not monotonic")
    if len(block_types) != b:
        raise ValueError("Block type count mismatch")

    for edge in graph["c"]:
        if len(edge) != 3 or not (0 <= edge[0] < b and 0 <= edge[1] < b):
            raise ValueError(f"Invalid control edge: {edge}")
    for edge in graph["d"]:
        if len(edge) != 2 or not (0 <= edge[0] < b and 0 <= edge[1] < b):
            raise ValueError(f"Invalid data-flow edge: {edge}")

    operand_count = 0
    for inst in instructions:
        if not inst:
            raise ValueError("Empty instruction record")
        for operand in inst[1:]:
            operand_count += 1
            kind = operand[0]
            expected = {K_REG: 2, K_IMM: 2, K_MEM: 6, K_SYMBOL: 2, K_BLOCK: 2, K_RAW: 2}
            if kind not in expected or len(operand) != expected[kind]:
                raise ValueError(f"Malformed operand record: {operand}")

    if operand_count != int(graph["operand_count"]):
        raise ValueError("Operand count mismatch")
    if int(graph["positions_flat"]) != n + operand_count + b + 1:
        raise ValueError("Flat position count mismatch")
    if int(graph["positions_pooled"]) != n + b + 1:
        raise ValueError("Pooled position count mismatch")


def to_columnar(graph: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Convert nested instruction records into arrays suitable for PyTorch.

    Missing fields are represented by zero. A kind mask tells the encoder which
    fields are active. Numeric immediates and displacements stay as integer side
    features; they do not consume vocabulary positions.
    """
    validate_compact_graph(graph)

    op_ids: List[int] = []
    operand_ptr: List[int] = [0]
    kind: List[int] = []
    reg_id: List[int] = []
    immediate: List[int] = []
    width: List[int] = []
    base_reg: List[int] = []
    index_reg: List[int] = []
    scale: List[int] = []
    displacement: List[int] = []
    symbol_id: List[int] = []
    block_target: List[int] = []
    raw_id: List[int] = []

    for inst in graph["i"]:
        op_ids.append(int(inst[0]))
        for operand in inst[1:]:
            k = int(operand[0])
            kind.append(k)
            reg_id.append(int(operand[1]) if k == K_REG else 0)
            immediate.append(int(operand[1]) if k == K_IMM else 0)
            width.append(int(operand[1]) if k == K_MEM else 0)
            base_reg.append(int(operand[2]) if k == K_MEM else 0)
            index_reg.append(int(operand[3]) if k == K_MEM else 0)
            scale.append(int(operand[4]) if k == K_MEM else 0)
            displacement.append(int(operand[5]) if k == K_MEM else 0)
            symbol_id.append(int(operand[1]) if k == K_SYMBOL else 0)
            block_target.append(int(operand[1]) if k == K_BLOCK else 0)
            raw_id.append(int(operand[1]) if k == K_RAW else 0)
        operand_ptr.append(len(kind))

    return {
        "op_ids": op_ids,
        "operand_ptr": operand_ptr,
        "operand_kind": kind,
        "operand_reg_id": reg_id,
        "operand_immediate": immediate,
        "operand_width": width,
        "operand_base_reg": base_reg,
        "operand_index_reg": index_reg,
        "operand_scale": scale,
        "operand_displacement": displacement,
        "operand_symbol_id": symbol_id,
        "operand_block_target": block_target,
        "operand_raw_id": raw_id,
        "block_ptr": [int(x) for x in graph["bp"]],
        "block_type": [int(x) for x in graph["bt"]],
        "cfg_src": [int(x[0]) for x in graph["c"]],
        "cfg_dst": [int(x[1]) for x in graph["c"]],
        "cfg_type": [int(x[2]) for x in graph["c"]],
        "dfg_src": [int(x[0]) for x in graph["d"]],
        "dfg_dst": [int(x[1]) for x in graph["d"]],
    }


def to_torch(graph: Mapping[str, Any], device: str | None = None) -> Dict[str, Any]:
    """Return int64 tensors. PyTorch is imported only when this is called."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for to_torch()") from exc

    arrays = to_columnar(graph)
    return {
        name: torch.tensor(values, dtype=torch.long, device=device)
        for name, values in arrays.items()
    }


def pooled_position_count(graph: Mapping[str, Any]) -> int:
    """One source vector per instruction, one per block, and one global token."""
    return int(graph["positions_pooled"])


def flat_position_count(graph: Mapping[str, Any]) -> int:
    """One opcode position, one position per operand, blocks, and one global token."""
    return int(graph["positions_flat"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Compact .jsonl.gz dataset")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    count = 0
    max_flat = 0
    max_pooled = 0
    for row in read_jsonl_gz(args.dataset):
        graph = row["compact_graph_v1"]
        validate_compact_graph(graph)
        max_flat = max(max_flat, flat_position_count(graph))
        max_pooled = max(max_pooled, pooled_position_count(graph))
        count += 1
        if args.limit and count >= args.limit:
            break

    print(json.dumps({"validated_rows": count, "max_flat": max_flat, "max_pooled": max_pooled}))
