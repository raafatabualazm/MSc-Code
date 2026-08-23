#!/usr/bin/env python3
"""Harmonize only the DFG edges of a scrubbed HumanEval public/private pair.

The direct compact-Qwen codec does not serialize DFG edges.  It regenerates
them from its privacy-canonicalized instruction/CFG domain with a SHA-pinned
extractor.  Older HumanEval graph files were built with a different extractor,
so they must be harmonized before they can be measured by that codec.

This program is deliberately fail-closed.  Apart from ``edges`` whose
``edge_type`` is ``dataflow``, every JSON value is required to remain exactly
equal.  Public/private rows are joined by task ID (never by row position), and
their binary evidence must agree before either output is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DFG = ROOT / "scrubbed_master_v2_release" / "extractors" / "dfg_extractor.py"
DEFAULT_CODEC = ROOT / "scripts" / "data" / "build_compact_qwen_v1.py"
EXPECTED_DFG_SHA256 = "beb237cf2ad8e3d65a536e8d30b698e14486ade36a019c247d580c372b858000"
PRIVATE_ONLY_KEYS = {
    "dart_source",
    "tests",
    "evaluation_only_dart_function_signature",
    "original_source",
    "semantic_function_name",
    "original_function_name",
    "private",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_symbol(path: Path, module_name: str, symbol: str) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, symbol, None)
    if not callable(value):
        raise ValueError(f"{path} does not define callable {symbol}")
    return value


def read_rows(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"{path}:{line_number}: missing task_id")
            result.append(row)
    if len(result) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, observed {len(result)}")
    task_ids = [row["task_id"] for row in result]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{path}: duplicate task_id")
    return result


def non_dataflow_edges(row: dict[str, Any]) -> list[dict[str, Any]]:
    edges = row.get("edges")
    if not isinstance(edges, list):
        raise ValueError(f"{row.get('task_id')}: edges must be a list")
    return [edge for edge in edges if edge.get("edge_type") != "dataflow"]


def dataflow_edges(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [edge for edge in row["edges"] if edge.get("edge_type") == "dataflow"]


def protected_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Everything that the transform is forbidden to alter."""
    return {
        "row_without_edges": {key: value for key, value in row.items() if key != "edges"},
        "non_dataflow_edges": non_dataflow_edges(row),
    }


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from recursive_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from recursive_keys(nested)


def validate_public_privacy(public_rows: list[dict[str, Any]]) -> None:
    for row in public_rows:
        leaked = sorted(set(recursive_keys(row)) & PRIVATE_ONLY_KEYS)
        if leaked:
            raise ValueError(f"{row['task_id']}: private keys in public row: {leaked}")


def validate_edge(edge: dict[str, Any], block_count: int, task_id: str) -> None:
    if set(edge) != {"source", "target", "edge_type"}:
        raise ValueError(f"{task_id}: regenerated DFG has unexpected fields: {sorted(edge)}")
    if edge["edge_type"] != "dataflow":
        raise ValueError(f"{task_id}: regenerated non-dataflow edge")
    for endpoint in ("source", "target"):
        value = edge[endpoint]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < block_count:
            raise ValueError(f"{task_id}: invalid DFG {endpoint}: {value!r}")


def replace_dfg(
    row: dict[str, Any],
    canonicalize: Callable[..., dict[str, Any]],
    build_dfg: Callable[..., list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_id = row["task_id"]
    canonical = canonicalize(row, "runtime_aware")
    regenerated = build_dfg(
        canonical["blocks"],
        canonical["cfg_edges"],
        max_edges=100000,
    )
    regenerated = sorted(
        (dict(edge) for edge in regenerated),
        key=lambda edge: (edge["source"], edge["target"], edge["edge_type"]),
    )
    for edge in regenerated:
        validate_edge(edge, len(canonical["blocks"]), task_id)
    if len({(e["source"], e["target"], e["edge_type"]) for e in regenerated}) != len(regenerated):
        raise ValueError(f"{task_id}: duplicate regenerated DFG edge")

    output = copy.deepcopy(row)
    output["edges"] = copy.deepcopy(non_dataflow_edges(row)) + regenerated
    if protected_projection(output) != protected_projection(row):
        raise AssertionError(f"{task_id}: protected content changed")

    # This is the exact equality tested by build_compact_qwen_v1 after decoding.
    recanonicalized = canonicalize(output, "runtime_aware")
    expected = sorted(
        recanonicalized["dfg_edges"],
        key=lambda edge: (edge["source"], edge["target"], edge["edge_type"]),
    )
    if expected != regenerated:
        raise AssertionError(f"{task_id}: output is outside compact codec DFG domain")
    return output, regenerated


def encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    # Stable serialization makes reruns byte-identical, not merely equivalent.
    return b"".join(stable_bytes(row) + b"\n" for row in rows)


def relative_display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dfg-extractor", type=Path, default=DEFAULT_DFG)
    parser.add_argument("--compact-codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--expected-rows", type=int, default=154)
    parser.add_argument(
        "--expected-dfg-sha256",
        default=EXPECTED_DFG_SHA256,
        help="Fail if the frozen DFG extractor bytes differ",
    )
    args = parser.parse_args()

    observed_dfg_sha = sha256_file(args.dfg_extractor)
    if observed_dfg_sha != args.expected_dfg_sha256:
        raise ValueError(
            "DFG extractor SHA-256 mismatch: "
            f"expected {args.expected_dfg_sha256}, observed {observed_dfg_sha}"
        )
    canonicalize = load_symbol(args.compact_codec, "harmonize_compact_codec", "canonicalize")
    build_dfg = load_symbol(args.dfg_extractor, "harmonize_frozen_dfg", "build_cross_block_dfg")

    public_rows = read_rows(args.public, args.expected_rows)
    private_rows = read_rows(args.private, args.expected_rows)
    validate_public_privacy(public_rows)
    public_by_id = {row["task_id"]: row for row in public_rows}
    private_by_id = {row["task_id"]: row for row in private_rows}
    if set(public_by_id) != set(private_by_id):
        only_public = sorted(set(public_by_id) - set(private_by_id))[:10]
        only_private = sorted(set(private_by_id) - set(public_by_id))[:10]
        raise ValueError(
            f"public/private task ID mismatch: only_public={only_public}, "
            f"only_private={only_private}"
        )

    output_public: list[dict[str, Any]] = []
    output_private_by_id: dict[str, dict[str, Any]] = {}
    row_audit: list[dict[str, Any]] = []
    old_public_dfg = old_private_dfg = new_dfg_count = 0
    for public in public_rows:
        task_id = public["task_id"]
        private = private_by_id[task_id]
        # "Binary parity" means the exact assembly, parsed instructions/blocks,
        # CFG metadata, and non-dataflow edges agree after the task-ID join.
        for key in ("assembly", "cfg"):
            if public.get(key) != private.get(key):
                raise ValueError(f"{task_id}: public/private {key} mismatch")
        if non_dataflow_edges(public) != non_dataflow_edges(private):
            raise ValueError(f"{task_id}: public/private non-dataflow edge mismatch")

        public_output, public_dfg = replace_dfg(public, canonicalize, build_dfg)
        private_output, private_dfg = replace_dfg(private, canonicalize, build_dfg)
        if public_dfg != private_dfg:
            raise ValueError(f"{task_id}: regenerated public/private DFG mismatch")
        if public_output["assembly"] != private_output["assembly"] or public_output["cfg"] != private_output["cfg"]:
            raise AssertionError(f"{task_id}: output binary parity failure")

        old_pub = len(dataflow_edges(public))
        old_priv = len(dataflow_edges(private))
        old_public_dfg += old_pub
        old_private_dfg += old_priv
        new_dfg_count += len(public_dfg)
        row_audit.append(
            {
                "task_id": task_id,
                "assembly_utf8_sha256": sha256_bytes(public["assembly"].encode("utf-8")),
                "cfg_semantic_sha256": sha256_bytes(stable_bytes(public["cfg"])),
                "non_dataflow_edges_semantic_sha256": sha256_bytes(stable_bytes(non_dataflow_edges(public))),
                "public_protected_semantic_sha256": sha256_bytes(stable_bytes(protected_projection(public))),
                "private_protected_semantic_sha256": sha256_bytes(stable_bytes(protected_projection(private))),
                "old_public_dfg_edges": old_pub,
                "old_private_dfg_edges": old_priv,
                "new_dfg_edges": len(public_dfg),
                "new_dfg_semantic_sha256": sha256_bytes(stable_bytes(public_dfg)),
            }
        )
        output_public.append(public_output)
        output_private_by_id[task_id] = private_output

    # Preserve the independent original row order of each side.
    output_private = [output_private_by_id[row["task_id"]] for row in private_rows]
    validate_public_privacy(output_public)
    public_payload = encode_jsonl(output_public)
    private_payload = encode_jsonl(output_private)
    # Encode twice as a direct deterministic-serialization gate.
    if public_payload != encode_jsonl(output_public) or private_payload != encode_jsonl(output_private):
        raise AssertionError("non-deterministic JSONL serialization")

    public_name = "humaneval_v2_nameonly_dfg_harmonized_public.jsonl"
    private_name = "humaneval_v2_nameonly_dfg_harmonized_private.jsonl"
    manifest_name = "harmonization_manifest.json"
    public_output_path = args.output_dir / public_name
    private_output_path = args.output_dir / private_name
    manifest_path = args.output_dir / manifest_name
    write_atomic(public_output_path, public_payload)
    write_atomic(private_output_path, private_payload)

    public_protected_before = [protected_projection(row) for row in public_rows]
    public_protected_after = [protected_projection(row) for row in output_public]
    private_protected_before = [protected_projection(row) for row in private_rows]
    private_protected_after = [protected_projection(row) for row in output_private]
    if public_protected_before != public_protected_after or private_protected_before != private_protected_after:
        raise AssertionError("protected aggregate changed")

    manifest = {
        "schema": "scrubbed-humaneval-dfg-harmonization-v1",
        "transform": "replace_dataflow_edges_only",
        "harmonizer": {
            "path": relative_display(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "row_count": args.expected_rows,
        "public_private_join": "task_id",
        "public_task_order_sha256": sha256_bytes(stable_bytes([r["task_id"] for r in public_rows])),
        "private_task_order_sha256": sha256_bytes(stable_bytes([r["task_id"] for r in private_rows])),
        "inputs": {
            "public": {"path": relative_display(args.public), "sha256": sha256_file(args.public)},
            "private": {"path": relative_display(args.private), "sha256": sha256_file(args.private)},
        },
        "outputs": {
            "public": {"path": public_name, "sha256": sha256_bytes(public_payload)},
            "private": {"path": private_name, "sha256": sha256_bytes(private_payload)},
        },
        "frozen_dfg_extractor": {
            "path": relative_display(args.dfg_extractor),
            "sha256": observed_dfg_sha,
            "max_edges": 100000,
        },
        "compact_canonicalization": {
            "path": relative_display(args.compact_codec),
            "sha256": sha256_file(args.compact_codec),
            "function": "canonicalize",
            "symbol_policy": "runtime_aware",
        },
        "edge_counts": {
            "old_public_dataflow": old_public_dfg,
            "old_private_dataflow": old_private_dfg,
            "harmonized_dataflow_each_side": new_dfg_count,
        },
        "gates": {
            "task_id_sets_equal": True,
            "public_private_binary_parity": True,
            "instructions_cfg_non_dataflow_preserved": True,
            "tests_labels_all_non_edge_fields_preserved": True,
            "public_has_no_private_keys": True,
            "canonical_dfg_regenerated_exactly": True,
            "deterministic_serialization": True,
            "legacy_graph_metadata_preserved": True,
        },
        "metadata_policy": (
            "graph_v2/integrity describe the frozen binary/CFG build and are preserved byte-semantically; "
            "this manifest's frozen_dfg_extractor and edge_counts are authoritative for harmonized DFG"
        ),
        "protected_aggregate_sha256": {
            "public_before": sha256_bytes(stable_bytes(public_protected_before)),
            "public_after": sha256_bytes(stable_bytes(public_protected_after)),
            "private_before": sha256_bytes(stable_bytes(private_protected_before)),
            "private_after": sha256_bytes(stable_bytes(private_protected_after)),
        },
        "row_audit_sha256": sha256_bytes(stable_bytes(row_audit)),
        "rows": row_audit,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    write_atomic(manifest_path, manifest_payload)
    checksums = "".join(
        f"{sha256_file(args.output_dir / name)}  {name}\n"
        for name in (public_name, private_name, manifest_name)
    ).encode("ascii")
    write_atomic(args.output_dir / "SHA256SUMS.txt", checksums)
    print(json.dumps({
        "rows": args.expected_rows,
        "dfg_extractor_sha256": observed_dfg_sha,
        "old_public_dfg": old_public_dfg,
        "old_private_dfg": old_private_dfg,
        "harmonized_dfg": new_dfg_count,
        "public_sha256": sha256_bytes(public_payload),
        "private_sha256": sha256_bytes(private_payload),
        "passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
