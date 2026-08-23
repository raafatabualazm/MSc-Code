#!/usr/bin/env python3
"""Lossless direct-compact stream with source-implicit inline CFG edges.

This codec reuses the established direct-compact-v1 atom IDs and therefore does
not enlarge Qwen's output vocabulary or shift block/control token IDs.  Its
two-token ``<G2C1><CFG>`` sentinel distinguishes the format from v1.

Each block is emitted once in contiguous ID order.  Its instructions are
followed immediately by zero or more ``edge-type, target-block`` pairs; the
current block is the exact source, so serializing the source block on every CFG
edge is redundant.  Decoding reconstructs the original ordered edge list and
the caller must prove an exact canonical round trip.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


EDGE_TOKEN = {
    "conditional_true": "<CT>",
    "conditional_false": "<CF>",
    "linear_fallthrough": "<CN>",
    "loop_backedge": "<CL>",
    "unconditional": "<CU>",
    "unconditional_jump": "<CJ>",
}
TOKEN_EDGE = {value: key for key, value in EDGE_TOKEN.items()}
TAG_RE = re.compile(r"<[^>]+>")
FORMAT_SENTINEL = ("<G2C1>", "<CFG>")


def compact_ids(
    text: str,
    base_tokenizer: Any,
    atom_ids: Mapping[str, int],
) -> list[int]:
    """Encode compact atoms outside the decoder vocabulary plus raw text."""

    output: list[int] = []
    cursor = 0
    for match in TAG_RE.finditer(text):
        if match.start() > cursor:
            encoded = base_tokenizer.encode(
                text[cursor : match.start()], add_special_tokens=False
            )
            values = encoded.ids if hasattr(encoded, "ids") else encoded
            output.extend(int(value) for value in values)
        atom = match.group()
        if atom not in atom_ids:
            raise ValueError(f"unknown compact atom: {atom}")
        output.append(int(atom_ids[atom]))
        cursor = match.end()
    if cursor < len(text):
        encoded = base_tokenizer.encode(text[cursor:], add_special_tokens=False)
        values = encoded.ids if hasattr(encoded, "ids") else encoded
        output.extend(int(value) for value in values)
    return output


def _ordered_edges_by_source(
    canonical: Mapping[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    blocks = canonical.get("blocks")
    edges = canonical.get("cfg_edges")
    if not isinstance(blocks, list) or not isinstance(edges, list):
        raise ValueError("canonical graph has no block/CFG arrays")
    if [int(block.get("id", -1)) for block in blocks] != list(
        range(len(blocks))
    ):
        raise ValueError("canonical block IDs are not contiguous")
    grouped = {block_id: [] for block_id in range(len(blocks))}
    previous_source = -1
    normalized: list[dict[str, Any]] = []
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        edge_type = str(edge["edge_type"])
        if (
            source not in grouped
            or not 0 <= target < len(blocks)
            or edge_type not in EDGE_TOKEN
        ):
            raise ValueError("canonical CFG edge is out of domain")
        if source < previous_source:
            raise ValueError(
                "canonical CFG edges are not grouped in source-block order"
            )
        previous_source = source
        record = {
            "source": source,
            "target": target,
            "edge_type": edge_type,
        }
        grouped[source].append(record)
        normalized.append(record)
    if normalized != [
        {
            "source": int(edge["source"]),
            "target": int(edge["target"]),
            "edge_type": str(edge["edge_type"]),
        }
        for edge in edges
    ]:
        raise ValueError("canonical CFG normalization drifted")
    return grouped


def encode(canonical: Mapping[str, Any], code: Mapping[str, int]) -> str:
    """Encode one canonical x86-64 graph without dropping any CFG edge."""

    if canonical.get("architecture") != "x86_64":
        raise ValueError("inline CFG v2 supports only x86_64")
    blocks = canonical.get("blocks")
    entries = canonical.get("entry_blocks")
    if not isinstance(blocks, list) or not isinstance(entries, list) or not entries:
        raise ValueError("canonical graph has invalid blocks/entries")
    if len(set(int(value) for value in entries)) != len(entries) or any(
        not 0 <= int(value) < len(blocks) for value in entries
    ):
        raise ValueError("canonical entry block is invalid")
    edges_by_source = _ordered_edges_by_source(canonical)

    output = [
        *FORMAT_SENTINEL,
        "<AX64>",
        "<ENTRY>",
        *(f"<B{int(value)}>" for value in entries),
        "<BLOCKS>",
    ]
    for block_id, block in enumerate(blocks):
        output.append(f"<B{block_id}>")
        instructions = block.get("instructions")
        if not isinstance(instructions, list):
            raise ValueError(f"block {block_id} has no instruction array")
        for raw_instruction in instructions:
            instruction = str(raw_instruction)
            if instruction in code:
                output.append(f"<I{int(code[instruction])}>")
            else:
                if "<" in instruction or ">" in instruction:
                    raise ValueError(
                        "raw fallback instruction contains a compact delimiter"
                    )
                output.extend(("<R>", instruction, "<E>"))
        for edge in edges_by_source[block_id]:
            output.extend(
                (
                    EDGE_TOKEN[edge["edge_type"]],
                    f"<B{edge['target']}>",
                )
            )
    output.append("<END>")
    return "".join(output)


def decode(text: str, expansions: Sequence[str]) -> dict[str, Any]:
    """Decode v2 and reconstruct every implicit CFG source exactly."""

    tags = list(TAG_RE.finditer(text))
    position = 0

    def peek() -> str:
        if position >= len(tags):
            raise ValueError("unexpected end of compact stream")
        return tags[position].group()

    def take(expected: str | None = None) -> str:
        nonlocal position
        value = peek()
        position += 1
        if expected is not None and value != expected:
            raise ValueError(f"expected {expected}, got {value}")
        return value

    take(FORMAT_SENTINEL[0])
    take(FORMAT_SENTINEL[1])
    take("<AX64>")
    take("<ENTRY>")
    entries: list[int] = []
    while peek() != "<BLOCKS>":
        value = take()
        if not re.fullmatch(r"<B[0-9]+>", value):
            raise ValueError("entry list contains a non-block atom")
        entries.append(int(value[2:-1]))
    take("<BLOCKS>")

    blocks: list[dict[str, Any]] = []
    cfg_edges: list[dict[str, Any]] = []
    while peek() != "<END>":
        start = take()
        expected_start = f"<B{len(blocks)}>"
        if start != expected_start:
            raise ValueError(
                f"expected contiguous block {expected_start}, got {start}"
            )
        block_id = len(blocks)
        instructions: list[str] = []
        while True:
            value = peek()
            if value.startswith("<I"):
                token = take()
                match = re.fullmatch(r"<I([0-9]+)>", token)
                if match is None:
                    raise ValueError("malformed instruction atom")
                index = int(match.group(1))
                if not 0 <= index < len(expansions):
                    raise ValueError("instruction atom is outside the codebook")
                instructions.append(str(expansions[index]))
            elif value == "<R>":
                take("<R>")
                raw_start = tags[position - 1].end()
                if position >= len(tags):
                    raise ValueError("raw fallback instruction is unterminated")
                raw_end = tags[position].start()
                instructions.append(text[raw_start:raw_end])
                take("<E>")
            else:
                break
        while peek() in TOKEN_EDGE:
            edge_type = TOKEN_EDGE[take()]
            target_token = take()
            match = re.fullmatch(r"<B([0-9]+)>", target_token)
            if match is None:
                raise ValueError("inline CFG edge has no target block")
            cfg_edges.append(
                {
                    "source": block_id,
                    "target": int(match.group(1)),
                    "edge_type": edge_type,
                }
            )
        blocks.append({"id": block_id, "instructions": instructions})

    take("<END>")
    if position != len(tags) or (tags and tags[-1].end() != len(text)):
        raise ValueError("compact stream has trailing text or atoms")
    if not entries or any(not 0 <= value < len(blocks) for value in entries):
        raise ValueError("decoded entry block is invalid")
    if any(
        not 0 <= int(edge["target"]) < len(blocks) for edge in cfg_edges
    ):
        raise ValueError("decoded CFG target is out of range")
    return {
        "architecture": "x86_64",
        "entry_blocks": entries,
        "blocks": blocks,
        "cfg_edges": cfg_edges,
    }

