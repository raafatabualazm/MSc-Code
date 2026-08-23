import json
from pathlib import Path

import pytest

from scrubbed_master_v2_release.finalize_phase0_binary_pool_v3 import (
    CURRENT_EXTRACTOR,
    EXPECTED_LAYOUT_CONTRACT,
    canonical_sha,
    sha256_file,
    validate_toolchain_manifest,
    validate_row,
)


RUNTIME_SHA256 = "c" * 64
TOOLCHAIN_SHA256 = "e" * 64


def fixture(tmp_path: Path):
    aot_path = tmp_path / "aot/train/row.aot"
    aot_path.parent.mkdir(parents=True)
    aot_path.write_bytes(b"aot")
    aot_sha = sha256_file(aot_path)
    metadata = {
        "family": "master",
        "phase0_split": "train",
        "target_function": "candidate",
    }
    source = {
        "task_id": "sigless_test",
        "analysis_program_sha256": "a" * 64,
        "compact_private_metadata": metadata,
    }
    projection = [
        {
            "pp_offset": 0x787,
            "kind": "string",
            "payload": {"code_units": [65]},
            "use_sites": [{"block": 0, "instruction": 0}],
        }
    ]
    static = {
        "source_blind": True,
        "target_function": "candidate",
        "entries": [
            {
                "pool_offset": "0x787",
                "category": "literal",
                "literal": {"type": "string", "code_units": [65]},
                "uses": [{"function_id": "candidate", "pc": "0x10"}],
            }
        ],
    }
    runtime = {
        "source_blind": True,
        "target_function": "candidate",
        "layout_contract": EXPECTED_LAYOUT_CONTRACT,
        "inputs": {
            "aot_sha256": aot_sha,
            "dartaotruntime_sha256": RUNTIME_SHA256,
        },
        "entries": [
            {
                "pp_offset": 0x787,
                "category": "literal",
                "literal": {"type": "string", "code_units": [65]},
                "uses": [{"function_id": "candidate", "pc": "0x1010"}],
            }
        ],
    }
    row = {
        "schema": "phase0-s44-binary-pool-aot-row-v1",
        "split": "train",
        "split_row": 0,
        "task_id": "sigless_test",
        "function": "candidate",
        "lang": "dart",
        "compact_private_metadata": metadata,
        "build_input_sha256": canonical_sha(source),
        "analysis_program_sha256": "a" * 64,
        "function_source_sha256": "b" * 64,
        "toolchain_manifest_sha256": TOOLCHAIN_SHA256,
        "graph_v2": {
            "extractor_sha256": CURRENT_EXTRACTOR,
            "symbol_entry_addresses": ["0x1000"],
        },
        "binary_pool_uses": projection,
        "binary_pool_private_receipt": {
            "schema": "dart-aot-reconciled-pool-receipts-v1",
            "static": static,
            "runtime": runtime,
            "projection_sha256": canonical_sha(projection),
        },
        "pool_projection_accounting": {
            "scope": "canonical_graph_retained_fixed_r15_xrefs",
            "all_target_xrefs_accounted": True,
            "target_exact_xrefs": 1,
            "excluded_non_graph_xrefs": [],
            "excluded_non_graph_xref_count": 0,
        },
        "aot": {
            "path": "aot/train/row.aot",
            "sha256": aot_sha,
            "size_bytes": 3,
        },
        "producer": {},
    }
    return source, row


def test_validate_row_proves_supported_literal_coverage(tmp_path: Path) -> None:
    source, row = fixture(tmp_path)
    reconciliation, _, counts = validate_row(
        build_root=tmp_path,
        source_row=source,
        row=row,
        split="train",
        position=0,
        verify_aot=True,
        expected_runtime_sha256=RUNTIME_SHA256,
        expected_layout_contract=EXPECTED_LAYOUT_CONTRACT,
        expected_toolchain_manifest_sha256=TOOLCHAIN_SHA256,
    )
    assert counts["represented_literal_xrefs"] == 1
    assert counts["supported_graph_literal_xrefs"] == 1
    assert reconciliation["source_blind"] is True


def test_validate_row_rejects_silent_supported_literal_omission(tmp_path: Path) -> None:
    source, row = fixture(tmp_path)
    row["binary_pool_uses"] = []
    row["binary_pool_private_receipt"]["projection_sha256"] = canonical_sha([])
    with pytest.raises(ValueError, match="supported_graph_literal_omission"):
        validate_row(
            build_root=tmp_path,
            source_row=source,
            row=row,
            split="train",
            position=0,
            verify_aot=True,
            expected_runtime_sha256=RUNTIME_SHA256,
            expected_layout_contract=EXPECTED_LAYOUT_CONTRACT,
            expected_toolchain_manifest_sha256=TOOLCHAIN_SHA256,
        )


def test_validate_row_rejects_runtime_layout_drift(tmp_path: Path) -> None:
    source, row = fixture(tmp_path)
    row["binary_pool_private_receipt"]["runtime"]["layout_contract"] = (
        "dart-3.11.5-linux-x64-object-layout-v1"
    )
    with pytest.raises(ValueError, match="runtime_layout_contract_mismatch"):
        validate_row(
            build_root=tmp_path,
            source_row=source,
            row=row,
            split="train",
            position=0,
            verify_aot=True,
            expected_runtime_sha256=RUNTIME_SHA256,
            expected_layout_contract=EXPECTED_LAYOUT_CONTRACT,
            expected_toolchain_manifest_sha256=TOOLCHAIN_SHA256,
        )


def test_validate_row_rejects_runtime_toolchain_drift(tmp_path: Path) -> None:
    source, row = fixture(tmp_path)
    row["binary_pool_private_receipt"]["runtime"]["inputs"][
        "dartaotruntime_sha256"
    ] = "d" * 64
    with pytest.raises(ValueError, match="runtime_toolchain_binding_mismatch"):
        validate_row(
            build_root=tmp_path,
            source_row=source,
            row=row,
            split="train",
            position=0,
            verify_aot=True,
            expected_runtime_sha256=RUNTIME_SHA256,
            expected_layout_contract=EXPECTED_LAYOUT_CONTRACT,
            expected_toolchain_manifest_sha256=TOOLCHAIN_SHA256,
        )


def test_toolchain_manifest_requires_dart_3_12_2_and_analysis_tools(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dart_toolchain_manifest.json"
    manifest = {
        "schema": "dart-aot-toolchain-manifest-v1",
        "dart_version": (
            "Dart SDK version: 3.12.2 (stable) (fixture) on \"linux_x64\""
        ),
        "target_architecture": "x86_64",
        "target_os": "linux",
        "files": {"runtime": {"sha256": RUNTIME_SHA256}},
        "analysis_tools": {
            name: {
                "resolved_path": f"/usr/bin/{name}",
                "sha256": char * 64,
                "version": f"{name} fixture",
            }
            for name, char in (
                ("gdb", "1"),
                ("nm", "2"),
                ("objdump", "3"),
                ("readelf", "4"),
            )
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, runtime_sha = validate_toolchain_manifest(path)
    assert loaded == manifest
    assert runtime_sha == RUNTIME_SHA256

    manifest["dart_version"] = "Dart SDK version: 3.11.5 (stable)"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="toolchain_dart_version_mismatch"):
        validate_toolchain_manifest(path)
