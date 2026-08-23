"""Build immutable, audited graph-v2 JSONL datasets from raw assembly.

Legacy ``*_cfg.jsonl`` files are treated only as containers for the original
records. Their precomputed ``cfg``/``edges`` fields are always replaced.
Incomplete assembly dumps fail integrity; callers may explicitly drop those
rows for training corpora, but benchmark builds fail instead of changing the
evaluation denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from scripts.data.cfg_extractor import AssemblyCFGExtractor
    from scripts.data.dfg_extractor import build_cross_block_dfg
except ModuleNotFoundError:
    # Direct execution sets sys.path to scripts/data rather than the repo root.
    from cfg_extractor import AssemblyCFGExtractor
    from dfg_extractor import build_cross_block_dfg


GRAPH_SCHEMA_VERSION = "antigravity-graph-v2.1"
ALLOWED_EDGE_TYPES = {
    "linear_fallthrough",
    "conditional_true",
    "conditional_false",
    "unconditional_jump",
    "loop_backedge",
    "call",
    "runtime_stub",
    "error_path",
    "dataflow",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extractor_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).with_name("cfg_extractor.py"),
        Path(__file__).with_name("dfg_extractor.py"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def row_identity(record: dict[str, Any], line_number: int) -> str:
    for key in ("task_id", "filename", "name", "function"):
        value = record.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return f"line:{line_number}"


def normalized_source_sha256(record: dict[str, Any]) -> str | None:
    source = record.get("source") or record.get("dart_source")
    if not isinstance(source, str) or not source.strip():
        return None
    normalized = re.sub(r"\s+", " ", source).strip()
    return sha256_bytes(normalized.encode("utf-8"))


def symbol_entry_addresses(record: dict[str, Any]) -> list[str]:
    ranges = record.get("flutter_function_symbol_ranges") or []
    if not isinstance(ranges, list):
        return []
    usable = [
        item for item in ranges
        if isinstance(item, dict) and item.get("address") not in (None, "")
    ]
    function_name = str(record.get("function") or "")
    exact = [item for item in usable if str(item.get("name") or "") == function_name]
    ordered = exact + [item for item in usable if item not in exact]
    addresses = []
    for item in ordered:
        address = str(item["address"])
        if address not in addresses:
            addresses.append(address)
    return addresses


def validate_graph(
    blocks: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    integrity: dict[str, Any],
    max_block_instrs: int,
) -> None:
    if integrity.get("valid") is not True:
        raise ValueError(f"CFG integrity failed: {integrity}")
    if not blocks:
        raise ValueError("CFG has no blocks")
    for expected_id, block in enumerate(blocks):
        if block.get("id") != expected_id:
            raise ValueError(
                f"non-contiguous block id {block.get('id')} at position {expected_id}"
            )
        instructions = block.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            raise ValueError(f"block {expected_id} has no instructions")
        if max_block_instrs > 0 and len(instructions) > max_block_instrs:
            raise ValueError(
                f"block {expected_id} has {len(instructions)} instructions; "
                f"limit is {max_block_instrs}"
            )

    block_count = len(blocks)
    for edge_index, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("edge_type")
        if not isinstance(source, int) or not isinstance(target, int):
            raise ValueError(f"edge {edge_index} has non-integer endpoints: {edge}")
        if not (0 <= source < block_count and 0 <= target < block_count):
            raise ValueError(
                f"edge {edge_index} is outside {block_count} blocks: {edge}"
            )
        if edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(f"edge {edge_index} has unknown type: {edge_type!r}")


def build_record(
    record: dict[str, Any],
    *,
    max_block_instrs: int,
    max_dataflow_edges: int,
    extractor_hash: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    assembly = record.get("assembly") or ""
    if not assembly.strip():
        raise ValueError("missing assembly")

    os.environ["GRAPH_MAX_BLOCK_INSTRS"] = str(max_block_instrs)
    symbol_entries = symbol_entry_addresses(record)
    blocks, cfg_edges, integrity = AssemblyCFGExtractor(
        assembly, entry_addresses=symbol_entries
    ).build_blocks()
    effective_symbol_entries = symbol_entries or list(integrity.get("entry_addresses") or [])
    block_dicts = [asdict(block) for block in blocks]
    cfg_edge_dicts = [asdict(edge) for edge in cfg_edges]
    dfg_edges = build_cross_block_dfg(
        block_dicts,
        cfg_edge_dicts,
        max_edges=max_dataflow_edges,
    )
    edges = cfg_edge_dicts + dfg_edges
    validate_graph(block_dicts, edges, integrity, max_block_instrs)

    output = dict(record)
    output["cfg"] = block_dicts
    output["edges"] = edges
    output["integrity"] = {
        **integrity,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "cfg_edge_count": len(cfg_edge_dicts),
        "dataflow_edge_count": len(dfg_edges),
        "max_block_instrs": max_block_instrs,
        "max_dataflow_edges": max_dataflow_edges,
        "symbol_entry_addresses": effective_symbol_entries,
    }
    output["graph_v2"] = {
        "schema": GRAPH_SCHEMA_VERSION,
        "assembly_sha256": sha256_bytes(assembly.encode("utf-8")),
        "extractor_sha256": extractor_hash,
        "max_block_instrs": max_block_instrs,
        "max_dataflow_edges": max_dataflow_edges,
        "symbol_entry_addresses": effective_symbol_entries,
    }
    return output, {
        "blocks": len(block_dicts),
        "cfg_edges": len(cfg_edge_dicts),
        "dataflow_edges": len(dfg_edges),
        "pruned_blocks": int(integrity.get("pruned_unreachable_block_count") or 0),
        "external_direct_branches": int(
            integrity.get("external_direct_branch_count") or 0
        ),
        "internal_direct_calls": int(integrity.get("internal_direct_call_count") or 0),
        "external_direct_calls": int(integrity.get("external_direct_call_count") or 0),
        "indirect_calls": int(integrity.get("indirect_call_count") or 0),
        "symbol_entries": len(effective_symbol_entries),
    }


def build_dataset(
    input_path: Path,
    output_path: Path,
    *,
    rejected_path: Path,
    summary_path: Path,
    drop_invalid: bool,
    expected_input_rows: int | None,
    expected_output_rows: int | None,
    max_block_instrs: int,
    max_dataflow_edges: int,
    dedupe_source: bool,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    extractor_hash = extractor_sha256()
    rejected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with input_path.open("r", encoding="utf-8-sig") as source, temp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                stats["input_rows"] += 1
                record = json.loads(line)
                if dedupe_source:
                    source_hash = normalized_source_sha256(record)
                    if source_hash is not None and source_hash in seen_sources:
                        stats["duplicate_source_rows"] += 1
                        continue
                    if source_hash is not None:
                        seen_sources.add(source_hash)
                try:
                    converted, row_stats = build_record(
                        record,
                        max_block_instrs=max_block_instrs,
                        max_dataflow_edges=max_dataflow_edges,
                        extractor_hash=extractor_hash,
                    )
                except Exception as exc:
                    stats["rejected_rows"] += 1
                    rejected.append({
                        "line_number": line_number,
                        "identity": row_identity(record, line_number),
                        "error": str(exc),
                    })
                    if not drop_invalid:
                        raise RuntimeError(
                            f"invalid graph row {input_path}:{line_number} "
                            f"({row_identity(record, line_number)}): {exc}"
                        ) from exc
                    continue

                destination.write(
                    json.dumps(converted, ensure_ascii=False, sort_keys=True) + "\n"
                )
                stats["output_rows"] += 1
                for key, value in row_stats.items():
                    stats[key] += value

        if expected_input_rows is not None and stats["input_rows"] != expected_input_rows:
            raise RuntimeError(
                f"expected {expected_input_rows} input rows, found {stats['input_rows']}"
            )
        if expected_output_rows is not None and stats["output_rows"] != expected_output_rows:
            raise RuntimeError(
                f"expected {expected_output_rows} output rows, found {stats['output_rows']}"
            )
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    with rejected_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "rejected": str(rejected_path),
        "rejected_sha256": file_sha256(rejected_path),
        "extractor_sha256": extractor_hash,
        "drop_invalid": drop_invalid,
        "dedupe_source": dedupe_source,
        "max_block_instrs": max_block_instrs,
        "max_dataflow_edges": max_dataflow_edges,
        **dict(stats),
    }
    if stats["output_rows"]:
        summary["mean_blocks_per_row"] = stats["blocks"] / stats["output_rows"]
        summary["mean_cfg_edges_per_row"] = stats["cfg_edges"] / stats["output_rows"]
        summary["mean_dataflow_edges_per_row"] = (
            stats["dataflow_edges"] / stats["output_rows"]
        )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--drop_invalid", action="store_true")
    parser.add_argument(
        "--dedupe_source",
        action="store_true",
        help="Keep the first row for each whitespace-normalized source program",
    )
    parser.add_argument("--expected_input_rows", type=int)
    parser.add_argument("--expected_output_rows", type=int)
    parser.add_argument("--max_block_instrs", type=int, default=24)
    parser.add_argument(
        "--max_dataflow_edges",
        type=int,
        default=0,
        help="0 keeps every DFG edge; positive values apply an explicit cap",
    )
    args = parser.parse_args()

    summary = build_dataset(
        args.input,
        args.output,
        rejected_path=args.rejected,
        summary_path=args.summary,
        drop_invalid=args.drop_invalid,
        expected_input_rows=args.expected_input_rows,
        expected_output_rows=args.expected_output_rows,
        max_block_instrs=args.max_block_instrs,
        max_dataflow_edges=args.max_dataflow_edges,
        dedupe_source=args.dedupe_source,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
