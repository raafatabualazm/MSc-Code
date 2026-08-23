#!/usr/bin/env python3
"""Seal the complete Phase-0 x64 AOT/pool build before compact encoding."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "phase0-s44-binary-pool-build-seal-v1"
RECONCILIATION_SCHEMA = "phase0-s44-pool-reconciliation-row-v1"
AOT_SCHEMA = "phase0-s44-source-only-aot-row-v1"
EXPECTED = {"train": 2951, "dev": 326}
TARGET = "candidate"
CURRENT_EXTRACTOR = "7a89b10f74754a8ff43580dba0cfb3348cd8e7b370e325ba8d31667c60ac04c1"
PINNED_DART_VERSION_PREFIX = "Dart SDK version: 3.12.2 (stable)"
EXPECTED_LAYOUT_CONTRACT = "dart-3.12.2-linux-x64-object-layout-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value).rstrip(b"\n"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank_jsonl_line:{path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non_object_jsonl_line:{path}:{line_number}")
            result.append(row)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_bytes(row))


def entry_offset(entry: dict[str, Any]) -> int:
    value = entry.get("pp_offset")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(str(entry.get("pool_offset")), 16)


def exact_uses(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        use
        for use in entry.get("uses") or []
        if use.get("function_id") == TARGET
    ]


def supported_entry(
    static_entry: dict[str, Any], runtime_entry: dict[str, Any]
) -> bool:
    if static_entry.get("category") in {"literal", "composite"}:
        return True
    return runtime_entry.get("category") == "literal"


def validate_toolchain_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "dart-aot-toolchain-manifest-v1":
        raise ValueError("toolchain_manifest_schema_mismatch")
    if (
        manifest.get("target_architecture") != "x86_64"
        or manifest.get("target_os") != "linux"
    ):
        raise ValueError("toolchain_target_mismatch")
    version = str(manifest.get("dart_version") or "")
    if not version.startswith(PINNED_DART_VERSION_PREFIX):
        raise ValueError(f"toolchain_dart_version_mismatch:{version}")
    runtime_sha = str(
        ((manifest.get("files") or {}).get("runtime") or {}).get("sha256") or ""
    )
    if len(runtime_sha) != 64 or any(ch not in "0123456789abcdef" for ch in runtime_sha):
        raise ValueError("toolchain_runtime_sha256_invalid")
    analysis_tools = manifest.get("analysis_tools") or {}
    if set(analysis_tools) != {"gdb", "nm", "objdump", "readelf"}:
        raise ValueError("toolchain_analysis_tools_missing")
    for name, record in analysis_tools.items():
        digest = str((record or {}).get("sha256") or "")
        if (
            len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or not str((record or {}).get("version") or "").strip()
            or not str((record or {}).get("resolved_path") or "").strip()
        ):
            raise ValueError(f"toolchain_analysis_tool_invalid:{name}")
    return manifest, runtime_sha


def validate_row(
    *,
    build_root: Path,
    source_row: dict[str, Any],
    row: dict[str, Any],
    split: str,
    position: int,
    verify_aot: bool,
    expected_runtime_sha256: str,
    expected_layout_contract: str,
    expected_toolchain_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if row.get("schema") != "phase0-s44-binary-pool-aot-row-v1":
        raise ValueError(f"{split}:{position}:row_schema_mismatch")
    if (
        row.get("split") != split
        or row.get("split_row") != position
        or row.get("task_id") != source_row.get("task_id")
    ):
        raise ValueError(f"{split}:{position}:source_output_alignment_mismatch")
    if row.get("function") != TARGET or row.get("lang") != "dart":
        raise ValueError(f"{split}:{position}:target_or_language_drift")
    if row.get("compact_private_metadata") != source_row.get("compact_private_metadata"):
        raise ValueError(f"{split}:{position}:private_metadata_drift")
    if row.get("build_input_sha256") != canonical_sha(source_row):
        raise ValueError(f"{split}:{position}:build_input_hash_mismatch")
    if row.get("analysis_program_sha256") != source_row.get("analysis_program_sha256"):
        raise ValueError(f"{split}:{position}:analysis_program_hash_mismatch")
    if row.get("toolchain_manifest_sha256") != expected_toolchain_manifest_sha256:
        raise ValueError(f"{split}:{position}:row_toolchain_binding_mismatch")
    if (row.get("graph_v2") or {}).get("extractor_sha256") != CURRENT_EXTRACTOR:
        raise ValueError(f"{split}:{position}:graph_extractor_route_mismatch")

    receipt_bundle = row.get("binary_pool_private_receipt") or {}
    if receipt_bundle.get("schema") != "dart-aot-reconciled-pool-receipts-v1":
        raise ValueError(f"{split}:{position}:receipt_bundle_schema_mismatch")
    static = receipt_bundle.get("static") or {}
    runtime = receipt_bundle.get("runtime") or {}
    if (
        static.get("source_blind") is not True
        or runtime.get("source_blind") is not True
        or static.get("target_function") != TARGET
        or runtime.get("target_function") != TARGET
    ):
        raise ValueError(f"{split}:{position}:receipt_not_source_blind_or_neutral")
    if runtime.get("layout_contract") != expected_layout_contract:
        raise ValueError(f"{split}:{position}:runtime_layout_contract_mismatch")
    if (runtime.get("inputs") or {}).get("dartaotruntime_sha256") != expected_runtime_sha256:
        raise ValueError(f"{split}:{position}:runtime_toolchain_binding_mismatch")
    projection = row.get("binary_pool_uses")
    if not isinstance(projection, list):
        raise ValueError(f"{split}:{position}:projection_not_list")
    projection_sha = canonical_sha(projection)
    if receipt_bundle.get("projection_sha256") != projection_sha:
        raise ValueError(f"{split}:{position}:projection_hash_mismatch")

    accounting = row.get("pool_projection_accounting") or {}
    if (
        accounting.get("scope") != "canonical_graph_retained_fixed_r15_xrefs"
        or accounting.get("all_target_xrefs_accounted") is not True
    ):
        raise ValueError(f"{split}:{position}:pool_accounting_gate_failed")
    excluded = accounting.get("excluded_non_graph_xrefs")
    if not isinstance(excluded, list) or len(excluded) != accounting.get(
        "excluded_non_graph_xref_count"
    ):
        raise ValueError(f"{split}:{position}:excluded_xref_accounting_mismatch")

    static_by_offset = {entry_offset(entry): entry for entry in static.get("entries") or []}
    runtime_by_offset = {entry_offset(entry): entry for entry in runtime.get("entries") or []}
    if set(static_by_offset) != set(runtime_by_offset):
        raise ValueError(f"{split}:{position}:receipt_offset_set_mismatch")
    candidate_base = int(str((row.get("graph_v2") or {})["symbol_entry_addresses"][0]), 16)
    excluded_keys = {
        (int(item["pp_offset"]), int(item["function_offset"])) for item in excluded
    }
    supported_graph_xrefs = 0
    supported_excluded_xrefs = 0
    target_xrefs = 0
    for offset in sorted(runtime_by_offset):
        static_entry = static_by_offset[offset]
        runtime_entry = runtime_by_offset[offset]
        supported = supported_entry(static_entry, runtime_entry)
        for use in exact_uses(runtime_entry):
            target_xrefs += 1
            function_offset = int(str(use["pc"]), 16) - candidate_base
            if (offset, function_offset) in excluded_keys:
                supported_excluded_xrefs += int(supported)
            else:
                supported_graph_xrefs += int(supported)
    represented_xrefs = sum(
        len(record.get("use_sites") or []) for record in projection
    )
    if target_xrefs != accounting.get("target_exact_xrefs"):
        raise ValueError(f"{split}:{position}:target_xref_count_mismatch")
    if represented_xrefs != supported_graph_xrefs:
        raise ValueError(
            f"{split}:{position}:supported_graph_literal_omission:"
            f"{represented_xrefs}!={supported_graph_xrefs}"
        )
    if len(excluded_keys) != accounting.get("excluded_non_graph_xref_count"):
        raise ValueError(f"{split}:{position}:duplicate_excluded_xref")

    aot = row.get("aot") or {}
    relative_path = Path(str(aot.get("path") or ""))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{split}:{position}:unsafe_aot_path")
    aot_path = (build_root / relative_path).resolve()
    if build_root.resolve() not in aot_path.parents or not aot_path.is_file():
        raise ValueError(f"{split}:{position}:aot_missing_or_outside_root")
    if aot_path.stat().st_size != aot.get("size_bytes"):
        raise ValueError(f"{split}:{position}:aot_size_mismatch")
    if verify_aot and sha256_file(aot_path) != aot.get("sha256"):
        raise ValueError(f"{split}:{position}:aot_hash_mismatch")
    if (runtime.get("inputs") or {}).get("aot_sha256") != aot.get("sha256"):
        raise ValueError(f"{split}:{position}:runtime_aot_binding_mismatch")

    reconciliation = {
        "schema": RECONCILIATION_SCHEMA,
        "split": split,
        "split_row": position,
        "task_id": row["task_id"],
        "family": row.get("family"),
        "target_function": TARGET,
        "source_blind": True,
        "build_input_sha256": row["build_input_sha256"],
        "aot_sha256": aot["sha256"],
        "static_receipt_sha256": canonical_sha(static),
        "runtime_receipt_sha256": canonical_sha(runtime),
        "runtime_layout_contract": expected_layout_contract,
        "dartaotruntime_sha256": expected_runtime_sha256,
        "toolchain_manifest_sha256": expected_toolchain_manifest_sha256,
        "projection_sha256": projection_sha,
        "projection_records": len(projection),
        "represented_literal_xrefs": represented_xrefs,
        "target_exact_xrefs": target_xrefs,
        "supported_graph_literal_xrefs": supported_graph_xrefs,
        "supported_excluded_non_graph_xrefs": supported_excluded_xrefs,
        "pool_projection_accounting": accounting,
    }
    aot_manifest_row = {
        "schema": AOT_SCHEMA,
        "split": split,
        "split_row": position,
        "task_id": row["task_id"],
        "analysis_program_sha256": row["analysis_program_sha256"],
        "function_source_sha256": row["function_source_sha256"],
        "aot_path": aot.as_posix() if isinstance(aot, Path) else str(relative_path).replace("\\", "/"),
        "aot_sha256": aot["sha256"],
        "aot_size_bytes": aot["size_bytes"],
        "producer": row.get("producer"),
    }
    counts = {
        "projection_records": len(projection),
        "represented_literal_xrefs": represented_xrefs,
        "target_exact_xrefs": target_xrefs,
        "supported_graph_literal_xrefs": supported_graph_xrefs,
        "excluded_non_graph_xrefs": len(excluded_keys),
        "aot_bytes": int(aot["size_bytes"]),
    }
    return reconciliation, aot_manifest_row, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--skip-aot-hash-verification", action="store_true")
    args = parser.parse_args()
    build_root = args.build_root.resolve()
    source_root = args.source_root.resolve()
    project_root = args.project_root.resolve()

    toolchain_path = build_root / "dart_toolchain_manifest.json"
    toolchain, runtime_sha256 = validate_toolchain_manifest(toolchain_path)
    toolchain_manifest_sha256 = sha256_file(toolchain_path)

    reconciliation_rows: list[dict[str, Any]] = []
    aot_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    split_outputs: dict[str, Any] = {}
    producer_hashes: set[str] = set()
    static_extractor_hashes: set[str] = set()
    runtime_extractor_hashes: set[str] = set()
    graph_builder_hashes: set[str] = set()
    for split, expected in EXPECTED.items():
        source_path = source_root / f"private_build_inputs/{split}.jsonl"
        row_path = build_root / f"prepared/{split}_codec_private.jsonl"
        split_manifest_path = build_root / f"manifests/{split}.json"
        sources = load_jsonl(source_path)
        rows = load_jsonl(row_path)
        if len(sources) != expected or len(rows) != expected:
            raise ValueError(
                f"{split}:row_count_mismatch:{len(sources)}:{len(rows)}:{expected}"
            )
        split_counts: Counter[str] = Counter()
        for position, (source_row, row) in enumerate(zip(sources, rows, strict=True)):
            reconciliation, aot_row, counts = validate_row(
                build_root=build_root,
                source_row=source_row,
                row=row,
                split=split,
                position=position,
                verify_aot=not args.skip_aot_hash_verification,
                expected_runtime_sha256=runtime_sha256,
                expected_layout_contract=EXPECTED_LAYOUT_CONTRACT,
                expected_toolchain_manifest_sha256=toolchain_manifest_sha256,
            )
            reconciliation_rows.append(reconciliation)
            aot_rows.append(aot_row)
            split_counts.update(counts)
            producer = row.get("producer") or {}
            producer_hashes.add(str(producer.get("script_sha256") or ""))
            static_extractor_hashes.add(str(producer.get("pool_extractor_sha256") or ""))
            runtime_extractor_hashes.add(
                str(producer.get("runtime_pool_extractor_sha256") or "")
            )
            graph_builder_hashes.add(str(producer.get("graph_builder_sha256") or ""))
        totals.update(split_counts)
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if (
            split_manifest.get("counts", {}).get("built_or_resumed") != expected
            or split_manifest.get("counts", {}).get("failed") != 0
            or not all((split_manifest.get("gates") or {}).values())
        ):
            raise ValueError(f"{split}:split_manifest_gate_failed")
        split_outputs[split] = {
            "source_input": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "codec_private": {
                "path": str(row_path.relative_to(build_root)),
                "sha256": sha256_file(row_path),
            },
            "split_manifest": {
                "path": str(split_manifest_path.relative_to(build_root)),
                "sha256": sha256_file(split_manifest_path),
            },
            "rows": expected,
            "counts": dict(split_counts),
        }

    for label, values in {
        "producer": producer_hashes,
        "static_pool_extractor": static_extractor_hashes,
        "runtime_pool_extractor": runtime_extractor_hashes,
        "graph_builder": graph_builder_hashes,
    }.items():
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"nonuniform_{label}_hashes:{sorted(values)}")
    expected_project_hashes = {
        "producer": sha256_file(
            project_root / "scripts/data/build_phase0_binary_pool_graphs.py"
        ),
        "static_pool_extractor": sha256_file(
            project_root / "scripts/data/extract_dart_aot_pool_receipt.py"
        ),
        "runtime_pool_extractor": sha256_file(
            project_root / "scripts/data/extract_dart_aot_pool_runtime_receipt.py"
        ),
        "graph_builder": sha256_file(
            project_root / "scrubbed_master_v2_release/build_scrubbed_dataset.py"
        ),
    }
    observed_project_hashes = {
        "producer": next(iter(producer_hashes)),
        "static_pool_extractor": next(iter(static_extractor_hashes)),
        "runtime_pool_extractor": next(iter(runtime_extractor_hashes)),
        "graph_builder": next(iter(graph_builder_hashes)),
    }
    if expected_project_hashes != observed_project_hashes:
        raise ValueError(
            f"project_producer_hash_mismatch:{observed_project_hashes}:"
            f"{expected_project_hashes}"
        )

    reconciliation_path = build_root / "pool_reconciliation_private.jsonl"
    aot_manifest_path = build_root / "aot_manifest.jsonl"
    write_jsonl(reconciliation_path, reconciliation_rows)
    write_jsonl(aot_manifest_path, aot_rows)

    extractor_manifest = {
        "schema": "dart-aot-pool-extractor-manifest-v1",
        "static": {
            "path": "scripts/data/extract_dart_aot_pool_receipt.py",
            "sha256": next(iter(static_extractor_hashes)),
        },
        "runtime": {
            "path": "scripts/data/extract_dart_aot_pool_runtime_receipt.py",
            "sha256": next(iter(runtime_extractor_hashes)),
        },
        "reconciliation_producer": {
            "path": "scripts/data/build_phase0_binary_pool_graphs.py",
            "sha256": next(iter(producer_hashes)),
        },
        "graph_builder": {
            "path": "scrubbed_master_v2_release/build_scrubbed_dataset.py",
            "sha256": next(iter(graph_builder_hashes)),
        },
        "target_function": TARGET,
        "target_architecture": "x86_64",
        "source_blind_after_aot": True,
    }
    extractor_manifest_path = build_root / "pool_extractor_manifest.json"
    write_json(extractor_manifest_path, extractor_manifest)

    source_preparation_path = source_root / "source_preparation_manifest.json"
    manifest = {
        "schema": SCHEMA,
        "rows": sum(EXPECTED.values()),
        "splits": split_outputs,
        "counts": dict(totals),
        "toolchain": {
            "dart_version": toolchain["dart_version"],
            "target_architecture": toolchain["target_architecture"],
            "target_os": toolchain["target_os"],
            "dartaotruntime_sha256": runtime_sha256,
            "runtime_layout_contract": EXPECTED_LAYOUT_CONTRACT,
            "manifest_sha256": toolchain_manifest_sha256,
            "analysis_tools": toolchain["analysis_tools"],
        },
        "artifacts": {
            "aot_manifest": {
                "path": aot_manifest_path.name,
                "sha256": sha256_file(aot_manifest_path),
            },
            "pool_reconciliation_private": {
                "path": reconciliation_path.name,
                "sha256": sha256_file(reconciliation_path),
            },
            "pool_extractor_manifest": {
                "path": extractor_manifest_path.name,
                "sha256": sha256_file(extractor_manifest_path),
            },
            "dart_toolchain_manifest": {
                "path": toolchain_path.name,
                "sha256": sha256_file(toolchain_path),
            },
            "source_preparation_manifest": {
                "path": str(source_preparation_path),
                "sha256": sha256_file(source_preparation_path),
            },
        },
        "producer": {
            "path": "scrubbed_master_v2_release/finalize_phase0_binary_pool_v3.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "gates": {
            "all_3277_rows_present": len(reconciliation_rows) == 3277,
            "all_aots_present_and_hash_valid": not args.skip_aot_hash_verification,
            "all_target_xrefs_accounted": all(
                row["pool_projection_accounting"]["all_target_xrefs_accounted"]
                for row in reconciliation_rows
            ),
            "all_graph_supported_literals_represented": totals[
                "represented_literal_xrefs"
            ]
            == totals["supported_graph_literal_xrefs"],
            "all_pool_receipts_source_blind": all(
                row["source_blind"] for row in reconciliation_rows
            ),
            "current_call_edge_extractor_uniform": True,
            "dart_3_12_2_toolchain_pinned": True,
            "runtime_layout_contract_uniform": all(
                row["runtime_layout_contract"] == EXPECTED_LAYOUT_CONTRACT
                for row in reconciliation_rows
            ),
            "runtime_receipts_bound_to_toolchain": all(
                row["dartaotruntime_sha256"] == runtime_sha256
                for row in reconciliation_rows
            ),
            "all_rows_bound_to_toolchain_manifest": all(
                row["toolchain_manifest_sha256"] == toolchain_manifest_sha256
                for row in reconciliation_rows
            ),
        },
    }
    if not all(manifest["gates"].values()):
        raise ValueError(f"final_build_gate_failed:{manifest['gates']}")
    manifest_path = build_root / "binary_build_manifest.json"
    write_json(manifest_path, manifest)
    sealed = [
        manifest_path,
        aot_manifest_path,
        reconciliation_path,
        extractor_manifest_path,
        toolchain_path,
        *(build_root / f"manifests/{split}.json" for split in EXPECTED),
        *(build_root / f"prepared/{split}_codec_private.jsonl" for split in EXPECTED),
        *(build_root / f"quarantine/{split}.jsonl" for split in EXPECTED),
    ]
    sums_path = build_root / "BINARY_BUILD_SHA256SUMS.txt"
    sums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(build_root).as_posix()}\n"
            for path in sealed
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
