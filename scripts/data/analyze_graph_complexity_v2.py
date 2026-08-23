"""Summarize graph-v2 size, complexity, and truncation-risk distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0,
        "mean": (sum(values) / len(values)) if values else 0.0,
    }


def identity(row: dict[str, Any], line_number: int) -> str:
    return str(
        row.get("task_id")
        or row.get("filename")
        or row.get("name")
        or row.get("function")
        or line_number
    )


def summarize(path: Path, top: int) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            blocks = row.get("cfg") or []
            edges = row.get("edges") or []
            cfg_edges = [edge for edge in edges if edge.get("edge_type") != "dataflow"]
            dfg_edges = [edge for edge in edges if edge.get("edge_type") == "dataflow"]
            instruction_count = sum(len(block.get("instructions") or []) for block in blocks)
            metrics.append(
                {
                    "line_number": line_number,
                    "identity": identity(row, line_number),
                    "blocks": len(blocks),
                    "cfg_edges": len(cfg_edges),
                    "dfg_edges": len(dfg_edges),
                    "instructions": instruction_count,
                    "cyclomatic": max(1, len(cfg_edges) - len(blocks) + 2),
                    "loop_backedges": sum(
                        edge.get("edge_type") == "loop_backedge" for edge in cfg_edges
                    ),
                    "conditional_edges": sum(
                        edge.get("edge_type") in {"conditional_true", "conditional_false"}
                        for edge in cfg_edges
                    ),
                    "external_direct_branches": int(
                        (row.get("integrity") or {}).get("external_direct_branch_count") or 0
                    ),
                    "pruned_unreachable_blocks": int(
                        (row.get("integrity") or {}).get("pruned_unreachable_block_count") or 0
                    ),
                }
            )

    fields = (
        "blocks",
        "cfg_edges",
        "dfg_edges",
        "instructions",
        "cyclomatic",
        "loop_backedges",
        "conditional_edges",
        "external_direct_branches",
        "pruned_unreachable_blocks",
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": len(metrics),
        "distributions": {
            field: distribution([int(item[field]) for item in metrics])
            for field in fields
        },
        "top_by_blocks": sorted(metrics, key=lambda item: item["blocks"], reverse=True)[:top],
        "top_by_cyclomatic": sorted(
            metrics, key=lambda item: item["cyclomatic"], reverse=True
        )[:top],
        "top_by_dfg_edges": sorted(
            metrics, key=lambda item: item["dfg_edges"], reverse=True
        )[:top],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    payload = {
        "schema": "antigravity-graph-v2-complexity-audit-v1",
        "datasets": [summarize(path, args.top) for path in args.dataset],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
