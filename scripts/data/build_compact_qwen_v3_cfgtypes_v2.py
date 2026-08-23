#!/usr/bin/env python3
"""Lossless compact-Qwen v3 codec with inline source-implied CFG edges.

The original v3 stream writes ``edge_type, source_block, target_block`` for
every CFG edge.  This versioned subcodec places each edge directly after its
source block, so the source is structural rather than repeated.  The pinned
CFG invariant makes conditional-false and linear-fallthrough targets exactly
the next block; those two target atoms are implicit.  Every edge still has an
explicit type atom, and every call keeps both ``<CC>`` and its target block.

The pool encoding, instruction codebook, public four-field schema, and v3
trainer contract are unchanged.  ``<G2C3>`` and ``<CFG>`` stay in their
original positions so the existing no-encoder trainer can validate the stream;
the exact semantics are versioned by this module's SHA and ``cfg_encoding`` in
the sealed contract.  The original ``build_compact_qwen_v3.py`` remains
byte-compatible with prior releases.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.data import build_compact_qwen_v3 as base


# Public codec surface consumed by the release builder and auditor.
CODEBOOK_SCHEMA = base.CODEBOOK_SCHEMA
CONTRACT_SCHEMA = base.CONTRACT_SCHEMA
CONTROL = base.CONTROL
EDGE_TOKEN = base.EDGE_TOKEN
TOKEN_EDGE = base.TOKEN_EDGE
POOL_ENCODING = base.POOL_ENCODING
POOL_SCHEMA = base.POOL_SCHEMA
POOL_SCOPE = base.POOL_SCOPE
PREFLIGHT_SCHEMA = base.PREFLIGHT_SCHEMA
MAX_COMPOSITE_DEPTH = base.MAX_COMPOSITE_DEPTH
MAX_COMPOSITE_NODES = base.MAX_COMPOSITE_NODES
NESTED_NONLITERAL_PROFILE_KIND = base.NESTED_NONLITERAL_PROFILE_KIND
ROUTE_CURRENT = base.ROUTE_CURRENT
ROUTE_LEGACY = base.ROUTE_LEGACY
ROUTE_SPECS = base.ROUTE_SPECS
ROUTE_BY_ATOM = base.ROUTE_BY_ATOM
RUNTIME_POLICY = base.RUNTIME_POLICY
TARGET_FUNCTION = base.TARGET_FUNCTION
STREAM_START = base.STREAM_START
POOL_START = base.POOL_START
POOL_END = base.POOL_END
STREAM_END = base.STREAM_END

CFG_ENCODING = "inline-source-implicit-next-fallthrough-targets-v2"
_IMPLICIT_NEXT_TYPES = frozenset({"conditional_false", "linear_fallthrough"})
_BLOCK_ATOM_RE = re.compile(r"<B([0-9]+)>")
_INSTRUCTION_ATOM_RE = re.compile(r"<I([0-9]+)>")

canonical_pool_json = base.canonical_pool_json
canonicalize = base.canonicalize
canonicalize_pool_uses = base.canonicalize_pool_uses
compact_ids = base.compact_ids
fixed_r15_offsets = base.fixed_r15_offsets
pool_envelope = base.pool_envelope
source_token_contract = base.source_token_contract


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inline_cfg(canonical: Mapping[str, Any]) -> None:
    blocks = canonical.get("blocks")
    edges = canonical.get("cfg_edges")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("inline_cfg_missing_blocks")
    if [block.get("id") for block in blocks] != list(range(len(blocks))):
        raise ValueError("inline_cfg_blocks_not_contiguous")
    if not isinstance(edges, list):
        raise ValueError("inline_cfg_edges_must_be_list")
    sources = [edge.get("source") for edge in edges]
    if sources != sorted(sources):
        raise ValueError("inline_cfg_edges_not_source_grouped")
    for index, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("edge_type")
        if not isinstance(source, int) or isinstance(source, bool):
            raise ValueError(f"inline_cfg_source_not_integer:{index}")
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError(f"inline_cfg_target_not_integer:{index}")
        if not 0 <= source < len(blocks) or not 0 <= target < len(blocks):
            raise ValueError(f"inline_cfg_endpoint_out_of_range:{index}")
        if edge_type not in EDGE_TOKEN:
            raise ValueError(f"inline_cfg_unknown_edge_type:{index}")
        if edge_type == "call" and not ROUTE_SPECS[
            str(canonical.get("dfg_route"))
        ].allow_call_edges:
            raise ValueError("inline_cfg_call_edge_not_allowed_for_route")
        if edge_type in _IMPLICIT_NEXT_TYPES and target != source + 1:
            raise ValueError(
                f"inline_cfg_{edge_type}_target_is_not_next_block:{index}"
            )


def encode(canonical: dict[str, Any], code: dict[str, int]) -> str:
    if "binary_pool" not in canonical:
        raise ValueError("missing_canonical_binary_pool")
    route = str(canonical.get("dfg_route") or "")
    if route not in ROUTE_SPECS:
        raise ValueError("unknown_canonical_route")
    _validate_inline_cfg(canonical)

    output = [STREAM_START, "<AX64>", ROUTE_SPECS[route].atom, "<ENTRY>"]
    output.extend(f"<B{value}>" for value in canonical["entry_blocks"])
    output.append("<BLOCKS>")
    edge_position = 0
    edges = canonical["cfg_edges"]
    for block in canonical["blocks"]:
        block_id = int(block["id"])
        output.append(f"<B{block_id}>")
        for instruction in block["instructions"]:
            if instruction in code:
                output.append(f"<I{code[instruction]}>")
            else:
                output.extend(("<R>", instruction, "<E>"))
        while edge_position < len(edges) and edges[edge_position]["source"] == block_id:
            edge = edges[edge_position]
            edge_type = edge["edge_type"]
            output.append(EDGE_TOKEN[edge_type])
            if edge_type not in _IMPLICIT_NEXT_TYPES:
                output.append(f"<B{edge['target']}>")
            edge_position += 1
    if edge_position != len(edges):
        raise ValueError("inline_cfg_unconsumed_edges")
    output.append("<CFG>")

    pool = base._validate_pool_envelope(
        canonical["binary_pool"], blocks=canonical.get("blocks")
    )
    return (
        "".join(output)
        + POOL_START
        + canonical_pool_json(pool)
        + POOL_END
        + STREAM_END
    )


def decode(text: str, expansions: list[str]) -> dict[str, Any]:
    if not isinstance(text, str) or not text.startswith(STREAM_START):
        raise ValueError("missing_v3_inline_cfg_stream_start")
    if not text.endswith(POOL_END + STREAM_END):
        raise ValueError("missing_v3_inline_cfg_pool_or_stream_end")
    if text.count(POOL_START) != 1 or text.count(POOL_END) != 1:
        raise ValueError("pool_marker_count_mismatch")
    pool_start = text.find(POOL_START)
    pool_end = text.find(POOL_END, pool_start + len(POOL_START))
    if pool_start < len(STREAM_START) or pool_end < pool_start:
        raise ValueError("invalid_pool_marker_order")

    graph_text = text[:pool_start]
    tags = list(base.TAG_RE.finditer(graph_text))
    position = 0

    def take(expected: str | None = None) -> str:
        nonlocal position
        if position >= len(tags):
            raise ValueError("unexpected_inline_cfg_eof")
        value = tags[position].group()
        position += 1
        if expected is not None and value != expected:
            raise ValueError(f"expected_{expected}_got_{value}")
        return value

    take(STREAM_START)
    take("<AX64>")
    route_atom = take()
    if route_atom not in ROUTE_BY_ATOM:
        raise ValueError("missing_or_unknown_extractor_route_atom")
    route = ROUTE_BY_ATOM[route_atom]
    take("<ENTRY>")
    entries: list[int] = []
    while position < len(tags) and tags[position].group() != "<BLOCKS>":
        match = _BLOCK_ATOM_RE.fullmatch(take())
        if match is None:
            raise ValueError("invalid_entry_block_atom")
        entries.append(int(match.group(1)))
    take("<BLOCKS>")

    blocks: list[dict[str, Any]] = []
    cfg_edges: list[dict[str, Any]] = []
    while position < len(tags) and tags[position].group() != "<CFG>":
        block_match = _BLOCK_ATOM_RE.fullmatch(take())
        if block_match is None:
            raise ValueError("invalid_inline_block_atom")
        block_id = int(block_match.group(1))
        instructions: list[str] = []
        while position < len(tags):
            token = tags[position].group()
            if token == "<CFG>" or _BLOCK_ATOM_RE.fullmatch(token) or token in TOKEN_EDGE:
                break
            token = take()
            instruction_match = _INSTRUCTION_ATOM_RE.fullmatch(token)
            if instruction_match is not None:
                expansion_index = int(instruction_match.group(1))
                if expansion_index >= len(expansions):
                    raise ValueError("instruction_atom_out_of_range")
                instructions.append(expansions[expansion_index])
            elif token == "<R>":
                start = tags[position - 1].end()
                if position >= len(tags):
                    raise ValueError("unterminated_raw_instruction")
                end = tags[position].start()
                instructions.append(graph_text[start:end])
                take("<E>")
            else:
                raise ValueError("bad_instruction_token:" + token)
        if not instructions:
            raise ValueError("decoded_inline_block_is_empty")
        blocks.append({"id": block_id, "instructions": instructions})

        while position < len(tags) and tags[position].group() in TOKEN_EDGE:
            edge_atom = take()
            edge_type = TOKEN_EDGE[edge_atom]
            if edge_type == "call" and not ROUTE_SPECS[route].allow_call_edges:
                raise ValueError("call_edge_not_allowed_for_legacy_route")
            if edge_type in _IMPLICIT_NEXT_TYPES:
                target = block_id + 1
                if target >= 4096:
                    raise ValueError("implicit_next_cfg_target_out_of_range")
            else:
                target_match = _BLOCK_ATOM_RE.fullmatch(take())
                if target_match is None:
                    raise ValueError("explicit_cfg_target_atom_missing")
                target = int(target_match.group(1))
            cfg_edges.append(
                {"source": block_id, "target": target, "edge_type": edge_type}
            )
    take("<CFG>")
    if position != len(tags):
        raise ValueError("trailing_inline_cfg_atoms")
    if [block["id"] for block in blocks] != list(range(len(blocks))):
        raise ValueError("decoded_block_ids_not_contiguous")
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("invalid_entry_blocks")
    if any(value < 0 or value >= len(blocks) for value in entries):
        raise ValueError("entry_block_out_of_range")
    if any(
        edge["source"] >= len(blocks) or edge["target"] >= len(blocks)
        for edge in cfg_edges
    ):
        raise ValueError("decoded_inline_cfg_endpoint_out_of_range")

    pool_text = text[pool_start + len(POOL_START) : pool_end]
    try:
        raw_pool = json.loads(pool_text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid_binary_pool_json") from error
    canonical_pool = base._validate_pool_envelope(
        base._decode_pool_positional(raw_pool), blocks=blocks
    )
    if canonical_pool_json(canonical_pool) != pool_text:
        raise ValueError("binary_pool_json_not_canonical")
    return {
        "architecture": "x86_64",
        "dfg_route": route,
        "entry_blocks": entries,
        "blocks": blocks,
        "cfg_edges": cfg_edges,
        "binary_pool": canonical_pool,
    }


def regenerate_dfg(
    decoded: dict[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return base.regenerate_dfg(decoded, registry)


def graph_codec_sha256() -> str:
    return base.graph_codec_sha256()


def codec_contract(**kwargs: Any) -> dict[str, Any]:
    contract = base.codec_contract(**kwargs)
    contract.update(
        {
            "cfg_encoding": CFG_ENCODING,
            "cfg_inline_encoding": {
                "source": "enclosing_canonical_block",
                "conditional_false_target": "next_canonical_block",
                "linear_fallthrough_target": "next_canonical_block",
                "other_targets": "explicit_<B#>_atom",
                "edge_order": "preserved_exactly_within_source_grouped_stream",
                "edge_type_atoms": "one_explicit_atom_per_ordered_cfg_edge",
                "call_edges": "explicit_<CC>_and_explicit_target_<B#>",
                "encode_gates": [
                    "sources_non_decreasing",
                    "all_endpoints_in_range",
                    "implicit_targets_equal_source_plus_one",
                ],
                "decode_gate": "all_inline_edges_reconstructed_before_dfg_regeneration",
            },
            "base_pool_codec_sha256": _sha256_file(Path(base.__file__)),
        }
    )
    return contract


__all__ = [
    "CFG_ENCODING",
    "CODEBOOK_SCHEMA",
    "CONTRACT_SCHEMA",
    "CONTROL",
    "EDGE_TOKEN",
    "POOL_ENCODING",
    "POOL_SCHEMA",
    "POOL_SCOPE",
    "PREFLIGHT_SCHEMA",
    "MAX_COMPOSITE_DEPTH",
    "MAX_COMPOSITE_NODES",
    "NESTED_NONLITERAL_PROFILE_KIND",
    "ROUTE_CURRENT",
    "ROUTE_LEGACY",
    "ROUTE_SPECS",
    "RUNTIME_POLICY",
    "TARGET_FUNCTION",
    "canonical_pool_json",
    "canonicalize",
    "canonicalize_pool_uses",
    "codec_contract",
    "compact_ids",
    "decode",
    "encode",
    "fixed_r15_offsets",
    "graph_codec_sha256",
    "pool_envelope",
    "regenerate_dfg",
    "source_token_contract",
]
