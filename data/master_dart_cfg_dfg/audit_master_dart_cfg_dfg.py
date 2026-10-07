#!/usr/bin/env python3
"""Independent streaming audit for the unified master Dart CFG+DFG dataset."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterator


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("master_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"{path}:{line_no}: {exc}") from exc
            yield row


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_gzip_payload(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(path: Path, builder) -> tuple[dict[str, Any], set[str], set[str]]:
    ids: set[str] = set()
    assembly_hashes: set[str] = set()
    source_hashes: set[str] = set()
    errors: list[dict[str, Any]] = []
    counts = collections.Counter()
    dataset_counts = collections.Counter()
    test_kinds = collections.Counter()
    test_origins = collections.Counter()
    runtime_status = collections.Counter()
    dfg_origins = collections.Counter()
    total_nodes = total_control = total_data = total_cases = 0

    for line_no, row in enumerate(iter_jsonl(path), 1):
        counts["rows"] += 1
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            errors.append({"line": line_no, "error": "missing id"})
        elif rid in ids:
            errors.append({"line": line_no, "error": "duplicate id", "id": rid})
        else:
            ids.add(rid)

        if row.get("schema") != builder.SCHEMA:
            errors.append({"line": line_no, "id": rid, "error": "record schema mismatch"})
        source = row.get("dart_source")
        assembly = row.get("assembly")
        cfg = row.get("cfg")
        edges = row.get("edges")
        if not isinstance(source, str) or not source.strip():
            errors.append({"line": line_no, "id": rid, "error": "empty source"})
            continue
        if not isinstance(assembly, str) or not assembly.strip():
            errors.append({"line": line_no, "id": rid, "error": "empty assembly"})
        if not isinstance(cfg, list) or not cfg:
            errors.append({"line": line_no, "id": rid, "error": "empty cfg"})
            continue
        if not isinstance(edges, list):
            errors.append({"line": line_no, "id": rid, "error": "edges not list"})
            continue

        expected_source = builder.source_fingerprint(source)
        expected_assembly = builder.assembly_fingerprint(assembly, cfg)
        expected_graph = builder.sha256_text(builder.json_compact({"cfg": cfg, "edges": edges}))
        fingerprints = row.get("fingerprints") or {}
        for key, expected in (
            ("source_sha256", expected_source),
            ("assembly_sha256", expected_assembly),
            ("graph_sha256", expected_graph),
        ):
            if fingerprints.get(key) != expected:
                errors.append({"line": line_no, "id": rid, "error": f"{key} mismatch"})
        if expected_source in source_hashes:
            errors.append({"line": line_no, "id": rid, "error": "duplicate normalized source"})
        if expected_assembly in assembly_hashes:
            errors.append({"line": line_no, "id": rid, "error": "duplicate normalized assembly"})
        source_hashes.add(expected_source)
        assembly_hashes.add(expected_assembly)

        n = len(cfg)
        total_nodes += n
        for index, block in enumerate(cfg):
            if set(block) != {"id", "label", "start_address", "instructions", "instruction_count", "block_type", "predecessors", "successors", "edge_types"}:
                errors.append({"line": line_no, "id": rid, "error": f"block {index} schema mismatch"})
            if block.get("id") != index:
                errors.append({"line": line_no, "id": rid, "error": f"block {index} id mismatch"})
            if not isinstance(block.get("instructions"), list) or not block.get("instructions"):
                errors.append({"line": line_no, "id": rid, "error": f"block {index} empty"})
            if block.get("instruction_count") != len(block.get("instructions") or []):
                errors.append({"line": line_no, "id": rid, "error": f"block {index} instruction_count mismatch"})

        edge_keys = {"source", "target", "edge_family", "edge_type", "locations", "dependency_count"}
        seen_edges: set[str] = set()
        for index, edge in enumerate(edges):
            if set(edge) != edge_keys:
                errors.append({"line": line_no, "id": rid, "error": f"edge {index} schema mismatch"})
            source_id, target_id = edge.get("source"), edge.get("target")
            if not isinstance(source_id, int) or not isinstance(target_id, int) or not (0 <= source_id < n and 0 <= target_id < n):
                errors.append({"line": line_no, "id": rid, "error": f"edge {index} out of range"})
            family = edge.get("edge_family")
            if family == "control":
                total_control += 1
                if edge.get("locations") != [] or edge.get("dependency_count") != 0:
                    errors.append({"line": line_no, "id": rid, "error": f"control edge {index} carries data fields"})
            elif family == "data":
                total_data += 1
                if edge.get("edge_type") != "dataflow":
                    errors.append({"line": line_no, "id": rid, "error": f"data edge {index} wrong type"})
                locations = edge.get("locations")
                if not isinstance(locations, list) or locations != sorted(set(locations)):
                    errors.append({"line": line_no, "id": rid, "error": f"data edge {index} locations not canonical"})
                if locations and edge.get("dependency_count") != len(locations):
                    errors.append({"line": line_no, "id": rid, "error": f"data edge {index} dependency_count mismatch"})
            else:
                errors.append({"line": line_no, "id": rid, "error": f"edge {index} bad family"})
            edge_sig = json.dumps(edge, sort_keys=True, separators=(",", ":"))
            if edge_sig in seen_edges:
                errors.append({"line": line_no, "id": rid, "error": f"duplicate edge {index}"})
            seen_edges.add(edge_sig)

        graph_meta = row.get("graph_v2") or {}
        if graph_meta.get("schema") != builder.GRAPH_SCHEMA:
            errors.append({"line": line_no, "id": rid, "error": "graph schema mismatch"})
        dfg_origins[str(graph_meta.get("dataflow_origin"))] += 1

        tests = row.get("tests")
        if not isinstance(tests, dict) or tests.get("schema") != builder.TEST_SCHEMA:
            errors.append({"line": line_no, "id": rid, "error": "tests schema mismatch"})
        else:
            if tests.get("validation", {}).get("static", {}).get("status") != "passed":
                errors.append({"line": line_no, "id": rid, "error": "tests static validation not passed"})
            test_kinds[str(tests.get("kind"))] += 1
            test_origins[str(tests.get("origin"))] += 1
            runtime_status[str(tests.get("validation", {}).get("runtime", {}).get("status"))] += 1
            total_cases += int(tests.get("case_count") or 0)

        dataset_counts[str((row.get("provenance") or {}).get("source_dataset"))] += 1
        counts["flutter_rows"] += int(row.get("framework") == "flutter")
        counts["rows_with_dataflow"] += int(any(e.get("edge_family") == "data" for e in edges))

    return (
        {
            "path": str(path),
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
            "rows": counts["rows"],
            "unique_ids": len(ids),
            "unique_source_fingerprints": len(source_hashes),
            "unique_assembly_fingerprints": len(assembly_hashes),
            "source_datasets": dict(dataset_counts),
            "test_kinds": dict(test_kinds),
            "test_origins": dict(test_origins),
            "runtime_status": dict(runtime_status),
            "dataflow_origins": dict(dfg_origins),
            "rows_with_dataflow": counts["rows_with_dataflow"],
            "flutter_rows": counts["flutter_rows"],
            "total_nodes": total_nodes,
            "total_control_edges": total_control,
            "total_dataflow_edges": total_data,
            "total_test_cases": total_cases,
            "error_count": len(errors),
            "errors": errors[:100],
        },
        source_hashes,
        assembly_hashes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.directory
    builder = load_builder(root / "build_master_dart_cfg_dfg.py")
    train_path = root / "master_dart_cfg_dfg_train.jsonl"
    heldout_path = root / "master_dart_cfg_dfg_heldout.jsonl"
    train, train_src, train_asm = scan(train_path, builder)
    heldout, heldout_src, heldout_asm = scan(heldout_path, builder)

    gzip_checks = {}
    for raw in (train_path, heldout_path):
        gz = raw.with_suffix(raw.suffix + ".gz")
        gzip_checks[gz.name] = {
            "exists": gz.is_file(),
            "compressed_sha256": sha256_path(gz) if gz.is_file() else None,
            "payload_sha256": sha256_gzip_payload(gz) if gz.is_file() else None,
            "raw_sha256": sha256_path(raw),
            "payload_matches_raw": bool(gz.is_file() and sha256_gzip_payload(gz) == sha256_path(raw)),
        }

    rejections = collections.Counter()
    rejection_rows = 0
    for row in iter_jsonl(root / "master_dart_cfg_dfg_rejected.jsonl"):
        rejection_rows += 1
        rejections[str(row.get("reason"))] += 1

    manifest = json.loads((root / "master_dart_cfg_dfg_manifest.json").read_text(encoding="utf-8"))
    manifest_checks = {
        "train_rows_match": manifest.get("train", {}).get("rows") == train["rows"],
        "heldout_rows_match": manifest.get("heldout", {}).get("rows") == heldout["rows"],
        "source_overlap_match": manifest.get("leakage", {}).get("source_overlap_count") == len(train_src & heldout_src),
        "assembly_overlap_match": manifest.get("leakage", {}).get("assembly_overlap_count") == len(train_asm & heldout_asm),
    }

    audit = {
        "schema": "antigravity-master-dataset-audit-v1",
        "train": train,
        "heldout": heldout,
        "leakage": {
            "source_overlap_count": len(train_src & heldout_src),
            "assembly_overlap_count": len(train_asm & heldout_asm),
        },
        "gzip_checks": gzip_checks,
        "rejections": {"rows": rejection_rows, "reasons": dict(rejections)},
        "manifest_checks": manifest_checks,
    }
    audit["passed"] = bool(
        train["error_count"] == 0
        and heldout["error_count"] == 0
        and not (train_src & heldout_src)
        and not (train_asm & heldout_asm)
        and all(item["payload_matches_raw"] for item in gzip_checks.values())
        and all(manifest_checks.values())
    )
    output = args.output or (root / "master_dart_cfg_dfg_audit.json")
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
