"""Safely add CFG/edge fields to Antigravity JSONL datasets.

This is the safe dataset preprocessor for SFT/GRPO rows.  It preserves every
original field (dart_source, tests, dart_function_signature, assembly, task_id,
metadata) and only adds or refreshes cfg/edges/integrity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from scripts.data.cfg_extractor import AssemblyCFGExtractor
except ModuleNotFoundError:
    # Supports direct execution as `python scripts/data/build_cfg_jsonl.py`.
    from cfg_extractor import AssemblyCFGExtractor


def _has_cfg(record: dict[str, Any]) -> bool:
    cfg = record.get("cfg")
    return isinstance(cfg, list) and len(cfg) > 0


def _extract_cfg(assembly: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    extractor = AssemblyCFGExtractor(assembly)
    blocks, edges, integrity = extractor.build_blocks()
    return [asdict(block) for block in blocks], [asdict(edge) for edge in edges], integrity


def safe_add_cfg(input_path: Path, output_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    requested_output_path = output_path
    requested_output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        same_path = input_path.resolve() == requested_output_path.resolve()
    except FileNotFoundError:
        same_path = os.path.abspath(input_path) == os.path.abspath(requested_output_path)

    temp_path: Path | None = None
    if same_path:
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{requested_output_path.name}.",
            suffix=".tmp",
            dir=str(requested_output_path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        output_path = temp_path

    try:
        with input_path.open("r", encoding="utf-8-sig") as f_in, output_path.open("w", encoding="utf-8") as f_out:
            for line_number, line in enumerate(f_in, start=1):
                if not line.strip():
                    continue

                stats["rows"] += 1
                record = json.loads(line)

                if _has_cfg(record) and not overwrite:
                    stats["kept_existing_cfg"] += 1
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                assembly = record.get("assembly") or ""
                if not assembly.strip():
                    stats["missing_assembly"] += 1
                    record.setdefault("cfg", [])
                    record.setdefault("edges", [])
                    record.setdefault("integrity", {"valid": False, "error": "missing assembly"})
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                try:
                    cfg, edges, integrity = _extract_cfg(assembly)
                    record["cfg"] = cfg
                    record["edges"] = edges
                    record["integrity"] = integrity
                    stats["extracted_cfg"] += 1
                    stats["total_blocks"] += len(cfg)
                    stats["total_edges"] += len(edges)
                except Exception as exc:
                    stats["failed_cfg"] += 1
                    record["cfg"] = []
                    record["edges"] = []
                    record["integrity"] = {
                        "valid": False,
                        "error": str(exc),
                        "line_number": line_number,
                    }
                    print(
                        f"Failed to extract CFG at {input_path}:{line_number} "
                        f"({record.get('filename') or record.get('task_id') or 'unknown'}): {exc}",
                        file=sys.stderr,
                    )

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        if temp_path is not None:
            temp_path.replace(requested_output_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise

    summary = {
        "input": str(input_path),
        "output": str(requested_output_path),
        "overwrite": overwrite,
        "in_place": same_path,
        **dict(stats),
    }
    rows = max(int(stats.get("extracted_cfg", 0)), 1)
    if stats.get("extracted_cfg"):
        summary["mean_blocks_per_extracted_row"] = stats["total_blocks"] / rows
        summary["mean_edges_per_extracted_row"] = stats["total_edges"] / rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract CFG even if the input row already has non-empty cfg.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    args = parser.parse_args()

    summary = safe_add_cfg(args.input, args.output, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
