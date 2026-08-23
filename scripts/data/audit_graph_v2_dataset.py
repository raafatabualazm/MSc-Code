"""Fail-closed structural and tokenizer audit for graph-v2 JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    from scripts.data.build_graph_v2_jsonl import (
        ALLOWED_EDGE_TYPES,
        GRAPH_SCHEMA_VERSION,
        extractor_sha256,
        file_sha256,
        sha256_bytes,
    )
except ModuleNotFoundError:
    from build_graph_v2_jsonl import (
        ALLOWED_EDGE_TYPES,
        GRAPH_SCHEMA_VERSION,
        extractor_sha256,
        file_sha256,
        sha256_bytes,
    )


SOURCE_DECLARATION_RE = re.compile(
    r"^(?:static\b|void\b|class\b|int\s|double\s|bool\s|String\b|List<|Map<|Set<)"
)


def parse_dataset_spec(value: str) -> tuple[Path, int]:
    try:
        path_text, expected_text = value.rsplit("=", 1)
        expected = int(expected_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "dataset must be PATH=EXPECTED_ROWS"
        ) from exc
    return Path(path_text), expected


def audit_dataset(
    path: Path,
    expected_rows: int,
    tokenizer: Any,
    instruction_token_cache: dict[str, int],
    *,
    max_block_instrs: int,
    max_code_tokens: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    problems: list[dict[str, Any]] = []
    current_extractor_hash = extractor_sha256()

    def problem(line_number: int, identity: str, message: str) -> None:
        if len(problems) < 100:
            problems.append({
                "line_number": line_number,
                "identity": identity,
                "message": message,
            })
        counts["problem_count"] += 1

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counts["rows"] += 1
            row = json.loads(line)
            identity = str(
                row.get("task_id")
                or row.get("filename")
                or row.get("name")
                or line_number
            )
            assembly = row.get("assembly") or ""
            metadata = row.get("graph_v2") or {}
            integrity = row.get("integrity") or {}
            blocks = row.get("cfg") or []
            edges = row.get("edges") or []

            if metadata.get("schema") != GRAPH_SCHEMA_VERSION:
                problem(line_number, identity, "missing graph-v2 schema")
            if metadata.get("assembly_sha256") != sha256_bytes(assembly.encode("utf-8")):
                problem(line_number, identity, "assembly hash mismatch")
            if metadata.get("extractor_sha256") != current_extractor_hash:
                problem(line_number, identity, "extractor hash mismatch")
            if metadata.get("max_block_instrs") != max_block_instrs:
                problem(
                    line_number,
                    identity,
                    f"max_block_instrs is {metadata.get('max_block_instrs')}, "
                    f"expected {max_block_instrs}",
                )
            if integrity.get("valid") is not True:
                problem(line_number, identity, "integrity.valid is not true")
            if integrity.get("networkx_available") is not True:
                problem(line_number, identity, "graph was built without NetworkX dominator analysis")
            entry_blocks = integrity.get("entry_blocks")
            if not isinstance(entry_blocks, list) or not entry_blocks:
                problem(line_number, identity, "entry_blocks is missing or empty")
            elif integrity.get("entry_block") != entry_blocks[0]:
                problem(line_number, identity, "primary entry block disagrees with entry_blocks")
            elif any(
                not isinstance(block_id, int) or not 0 <= block_id < len(blocks)
                for block_id in entry_blocks
            ):
                problem(line_number, identity, "entry block is outside the graph")
            symbol_entries = metadata.get("symbol_entry_addresses") or []
            if len(entry_blocks or []) != len(symbol_entries):
                problem(line_number, identity, "not every symbol entry resolved to a block")
            if int(integrity.get("unresolved_direct_branch_count") or 0) != 0:
                problem(line_number, identity, "unresolved direct branch remains")
            if integrity.get("unknown_branch_mnemonics"):
                problem(line_number, identity, "unknown branch mnemonic remains")
            external_count = int(integrity.get("external_direct_branch_count") or 0)
            counts["external_direct_branches"] += external_count
            counts["max_external_direct_branches_per_row"] = max(
                counts["max_external_direct_branches_per_row"], external_count
            )
            counts["pruned_unreachable_blocks"] += int(
                integrity.get("pruned_unreachable_block_count") or 0
            )
            if not blocks:
                problem(line_number, identity, "empty CFG")
                continue

            counts["blocks"] += len(blocks)
            counts["max_blocks_per_row"] = max(counts["max_blocks_per_row"], len(blocks))
            for block_id, block in enumerate(blocks):
                instructions = block.get("instructions") or []
                if block.get("id") != block_id:
                    problem(line_number, identity, f"non-contiguous block id at {block_id}")
                if not instructions:
                    problem(line_number, identity, f"empty block {block_id}")
                if len(instructions) > max_block_instrs:
                    problem(
                        line_number,
                        identity,
                        f"block {block_id} exceeds {max_block_instrs} instructions",
                    )
                code_tokens = 0
                for instruction in instructions:
                    if SOURCE_DECLARATION_RE.match(instruction.strip()):
                        problem(
                            line_number,
                            identity,
                            f"source declaration parsed in block {block_id}: {instruction[:80]}",
                        )
                    if instruction not in instruction_token_cache:
                        instruction_token_cache[instruction] = len(tokenizer.tokenize(instruction))
                    code_tokens += instruction_token_cache[instruction]
                counts["max_code_tokens_per_block"] = max(
                    counts["max_code_tokens_per_block"], code_tokens
                )
                if code_tokens > max_code_tokens:
                    problem(
                        line_number,
                        identity,
                        f"block {block_id} has {code_tokens} GraphCodeBERT tokens; "
                        f"limit is {max_code_tokens}",
                    )

            seen_edges: set[tuple[int, int, str]] = set()
            cfg_count = 0
            dfg_count = 0
            for edge_index, edge in enumerate(edges):
                source = edge.get("source")
                target = edge.get("target")
                edge_type = edge.get("edge_type")
                if (
                    not isinstance(source, int)
                    or not isinstance(target, int)
                    or not 0 <= source < len(blocks)
                    or not 0 <= target < len(blocks)
                ):
                    problem(line_number, identity, f"bad edge {edge_index}: {edge}")
                    continue
                if edge_type not in ALLOWED_EDGE_TYPES:
                    problem(line_number, identity, f"unknown edge type {edge_type!r}")
                signature = (source, target, str(edge_type))
                if signature in seen_edges:
                    problem(line_number, identity, f"duplicate edge {signature}")
                seen_edges.add(signature)
                if edge_type == "dataflow":
                    dfg_count += 1
                    locations = edge.get("locations")
                    if not isinstance(locations, list) or not locations:
                        problem(line_number, identity, "dataflow edge lacks locations")
                    elif edge.get("dependency_count") != len(locations):
                        problem(line_number, identity, "dataflow dependency count mismatch")
                else:
                    cfg_count += 1
            counts["cfg_edges"] += cfg_count
            counts["dataflow_edges"] += dfg_count
            counts["max_cfg_edges_per_row"] = max(counts["max_cfg_edges_per_row"], cfg_count)
            counts["max_dataflow_edges_per_row"] = max(
                counts["max_dataflow_edges_per_row"], dfg_count
            )
            configured_cap = int(metadata.get("max_dataflow_edges") or 0)
            if configured_cap > 0 and dfg_count >= configured_cap:
                counts["rows_at_dataflow_edge_cap"] += 1
            if integrity.get("cfg_edge_count") != cfg_count:
                problem(line_number, identity, "stored CFG edge count mismatch")
            if integrity.get("dataflow_edge_count") != dfg_count:
                problem(line_number, identity, "stored DFG edge count mismatch")

    if counts["rows"] != expected_rows:
        problem(0, str(path), f"expected {expected_rows} rows, found {counts['rows']}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "expected_rows": expected_rows,
        **dict(counts),
        "problems": problems,
        "passed": counts["problem_count"] == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_spec,
        metavar="PATH=ROWS",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", default="microsoft/graphcodebert-base")
    parser.add_argument("--tokenizer_revision", required=True)
    parser.add_argument("--max_block_instrs", type=int, default=20)
    parser.add_argument("--max_code_tokens", type=int, default=510)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    cache: dict[str, int] = {}
    reports = [
        audit_dataset(
            path,
            expected,
            tokenizer,
            cache,
            max_block_instrs=args.max_block_instrs,
            max_code_tokens=args.max_code_tokens,
        )
        for path, expected in args.dataset
    ]
    payload = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "max_block_instrs": args.max_block_instrs,
        "max_code_tokens": args.max_code_tokens,
        "instruction_token_cache_size": len(cache),
        "datasets": reports,
        "passed": all(report["passed"] for report in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
