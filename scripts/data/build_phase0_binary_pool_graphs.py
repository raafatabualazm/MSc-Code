#!/usr/bin/env python3
"""Build source-only x64 AOT graphs and binary-derived literal-pool rows.

This is private build infrastructure.  It reads the already-neutralized,
test-free analysis programs prepared by ``prepare_phase0_binary_pool_v3.py``.
Model-side graph and pool fields are then recovered from the resulting AOT and
the matching pinned compiler artifacts; no literal is read from Dart source.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "scrubbed_master_v2_release"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))

import build_scrubbed_dataset as graph_builder
from scripts.data.cfg_extractor import AssemblyCFGExtractor as CurrentCFGExtractor
from scripts.data.dfg_extractor import build_cross_block_dfg as build_current_dfg
from scripts.data.extract_dart_aot_pool_receipt import (
    PoolReceiptError,
    build_pool_receipt,
)
from scripts.data.extract_dart_aot_pool_runtime_receipt import (
    RuntimePoolReceiptError,
    SDK_LAYOUT_CONTRACT,
    build_runtime_receipt,
)


ROW_SCHEMA = "phase0-s44-binary-pool-aot-row-v1"
MANIFEST_SCHEMA = "phase0-s44-binary-pool-aot-manifest-v1"
GRAPH_SCHEMA = "antigravity-graph-v2.1"
TARGET = "candidate"
PINNED_DART_VERSION_PREFIX = "Dart SDK version: 3.12.2 (stable)"
ASSEMBLY_ADDRESS_RE = re.compile(r"^\s*(0x[0-9a-fA-F]+)\s+<\+\d+>:\s*(.*)$")
FIXED_R15_RE = re.compile(
    r"\[\s*r15\s*(?:(?P<sign>[+-])\s*(?P<amount>0x[0-9a-fA-F]+|[0-9]+))?\s*\]",
    re.IGNORECASE,
)

# Exact source-blind descriptors that the pinned V8 profile decoder may retain
# inside otherwise complete Array/Map storage.  Keep this finite and in lockstep
# with ``build_compact_qwen_v3.NESTED_NONLITERAL_PROFILE_KIND``: arbitrary
# profile type strings are not safe model input because they can carry names.
NESTED_NONLITERAL_PROFILE_KIND = {
    "Type": "type_metadata",
    "TypeArguments": "type_metadata",
    "FunctionType": "type_metadata",
    "TypeParameter": "type_metadata",
    "TypeParameters": "type_metadata",
    "Code": "code",
    "Function": "callable",
    "Closure": "callable",
    "ClosureData": "callable",
    "Field": "declaration_metadata",
    "PatchClass": "declaration_metadata",
    "Class": "declaration_metadata",
    "ArgumentsDescriptor": "calling_convention_metadata",
    "SubtypeTestCache": "calling_convention_metadata",
    "UnlinkedCall": "calling_convention_metadata",
    "Instance": "runtime_object",
    "Record": "runtime_object",
}


class BuildError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "DART_SUPPRESS_ANALYTICS": "1",
            "PUB_ENVIRONMENT": "compact_qwen_v3_binary_build",
        }
    )
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise BuildError(f"command_timeout:{Path(command[0]).name}:{timeout}") from error
    if process.returncode:
        stderr = process.stderr[-4000:].replace(str(cwd), "<workdir>")
        raise BuildError(
            f"command_failed:{Path(command[0]).name}:{process.returncode}:{stderr}"
        )
    return process


def _toolchain_paths(dart_sdk: Path) -> dict[str, Path]:
    paths = {
        "dart": dart_sdk / "bin/dart",
        "runtime": dart_sdk / "bin/dartaotruntime",
        "gen_snapshot": dart_sdk / "bin/utils/gen_snapshot",
        "gen_kernel": dart_sdk / "bin/snapshots/gen_kernel_aot.dart.snapshot",
        "platform": dart_sdk / "lib/_internal/vm_platform_product.dill",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise BuildError(f"missing_toolchain_files:{missing}")
    return paths


def _analysis_tool_manifest(command: str, *, cwd: Path) -> dict[str, str]:
    supplied = Path(command)
    resolved_text = (
        str(supplied.resolve())
        if supplied.is_file()
        else str(Path(shutil.which(command) or "").resolve())
    )
    resolved = Path(resolved_text)
    if not resolved.is_file():
        raise BuildError(f"analysis_tool_missing:{command}")
    version = run([str(resolved), "--version"], cwd=cwd, timeout=20)
    version_text = (version.stdout + version.stderr).strip().splitlines()
    if not version_text:
        raise BuildError(f"analysis_tool_version_missing:{command}")
    return {
        "command": command,
        "resolved_path": str(resolved),
        "sha256": sha256_file(resolved),
        "version": version_text[0],
    }


def toolchain_manifest(
    dart_sdk: Path, *, gdb: str, nm: str, objdump: str, readelf: str
) -> dict[str, Any]:
    paths = _toolchain_paths(dart_sdk)
    version = run([str(paths["dart"]), "--version"], cwd=dart_sdk, timeout=20)
    version_text = (version.stdout + version.stderr).strip()
    if not version_text.startswith(PINNED_DART_VERSION_PREFIX):
        raise BuildError(f"unpinned_dart_version:{version_text}")
    return {
        "schema": "dart-aot-toolchain-manifest-v1",
        "dart_version": version_text,
        "target_os": "linux",
        "target_architecture": "x86_64",
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "analysis_tools": {
            "gdb": _analysis_tool_manifest(gdb, cwd=dart_sdk),
            "nm": _analysis_tool_manifest(nm, cwd=dart_sdk),
            "objdump": _analysis_tool_manifest(objdump, cwd=dart_sdk),
            "readelf": _analysis_tool_manifest(readelf, cwd=dart_sdk),
        },
        "commands": {
            "kernel": [
                "dartaotruntime",
                "gen_kernel_aot.dart.snapshot",
                "--platform",
                "vm_platform_product.dill",
                "--aot",
                "--tfa",
                "--target-os",
                "linux",
            ],
            "snapshot": [
                "gen_snapshot",
                "--snapshot_kind=app-aot-elf",
                "--disassemble",
                "--disassemble-optimized",
                "--disassemble-relative",
                "--code-comments",
                "--write-v8-snapshot-profile-to",
            ],
        },
    }


def _parse_assembly_instructions(assembly: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for line in assembly.splitlines():
        match = ASSEMBLY_ADDRESS_RE.match(line)
        if match:
            result.append((int(match.group(1), 16), match.group(2).strip()))
    if not result:
        raise BuildError("scrubbed_assembly_has_no_instructions")
    return result


def instruction_sites(
    assembly: str, cfg: list[dict[str, Any]]
) -> tuple[int, dict[int, dict[str, int]], dict[str, int]]:
    instructions = _parse_assembly_instructions(assembly)
    by_address = {address: position for position, (address, _) in enumerate(instructions)}
    if len(by_address) != len(instructions):
        raise BuildError("duplicate_objdump_instruction_address")
    sites: dict[int, dict[str, int]] = {}
    cursor = 0
    for block_position, block in enumerate(cfg):
        if int(block.get("id", -1)) != block_position:
            raise BuildError("cfg_blocks_not_position_aligned")
        start = int(str(block.get("start_address")), 16)
        if start not in by_address:
            raise BuildError(f"cfg_start_not_in_objdump:0x{start:x}")
        cursor = by_address[start]
        block_instructions = block.get("instructions")
        if not isinstance(block_instructions, list) or not block_instructions:
            raise BuildError(f"empty_cfg_block:{block_position}")
        for instruction_position, expected in enumerate(block_instructions):
            if cursor >= len(instructions):
                raise BuildError("cfg_instruction_overruns_objdump")
            address, observed = instructions[cursor]
            if observed != str(expected).strip():
                raise BuildError(
                    "cfg_objdump_instruction_mismatch:"
                    f"{block_position}:{instruction_position}:{observed!r}!={expected!r}"
                )
            sites[address] = {
                "block": block_position,
                "instruction": instruction_position,
            }
            cursor += 1
    graph_instruction_count = sum(
        len(block.get("instructions") or []) for block in cfg
    )
    if len(sites) != graph_instruction_count:
        raise BuildError(
            f"cfg_site_count_mismatch:{len(sites)}!={graph_instruction_count}"
        )
    # The pinned CFG extractor can deliberately prune unreachable machine-code
    # islands.  Those instructions remain accounted for here; any target pool
    # xref inside one still fails later because it has no canonical graph site.
    return instructions[0][0], sites, {
        "objdump_instruction_count": len(instructions),
        "graph_instruction_count": graph_instruction_count,
        "non_graph_instruction_count": len(instructions) - graph_instruction_count,
    }


def graph_from_current_extractor(
    assembly: str, entry_addresses: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    old = os.environ.get("GRAPH_MAX_BLOCK_INSTRS")
    os.environ["GRAPH_MAX_BLOCK_INSTRS"] = str(graph_builder.MAX_BLOCK_INSTRS)
    try:
        blocks, control, extractor_integrity = CurrentCFGExtractor(assembly).build_blocks()
    finally:
        if old is None:
            os.environ.pop("GRAPH_MAX_BLOCK_INSTRS", None)
        else:
            os.environ["GRAPH_MAX_BLOCK_INSTRS"] = old
    cfg = [dataclasses.asdict(block) for block in blocks]
    control_edges = [dataclasses.asdict(edge) for edge in control]
    dfg = build_current_dfg(cfg, control_edges, max_edges=100000)
    edges = control_edges + dfg
    block_count = len(cfg)
    all_in_range = all(
        isinstance(edge.get("source"), int)
        and isinstance(edge.get("target"), int)
        and 0 <= edge["source"] < block_count
        and 0 <= edge["target"] < block_count
        for edge in edges
    )
    valid = bool(
        cfg
        and all_in_range
        and all(block.get("instructions") for block in cfg)
    )
    integrity = {
        **extractor_integrity,
        "isolated_nodes": extractor_integrity.get("isolated_nodes", []),
        "isolated_nonentry_nodes": [
            node
            for node in extractor_integrity.get("isolated_nodes", [])
            if node != 0
        ],
        "entry_nodes": [0] if cfg else [],
        "entry_address": entry_addresses[0] if entry_addresses else None,
        "entry_addresses": entry_addresses,
        "entry_block": 0 if cfg else None,
        "entry_blocks": [0] if cfg else [],
        "requested_entry_address_count": len(entry_addresses),
        "resolved_entry_address_count": len(entry_addresses) if cfg else 0,
        "unresolved_entry_addresses": [],
        "has_entry": bool(cfg),
        "all_edges_in_range": all_in_range,
        "all_blocks_nonempty": all(bool(block.get("instructions")) for block in cfg),
        "parsed_instruction_count": sum(
            len(block.get("instructions") or []) for block in cfg
        ),
        "candidate_line_count": sum(
            len(block.get("instructions") or []) for block in cfg
        ),
        "graph_schema_version": GRAPH_SCHEMA,
        "cfg_edge_count": len(control_edges),
        "dataflow_edge_count": len(dfg),
        "max_block_instrs": graph_builder.MAX_BLOCK_INSTRS,
        "max_dataflow_edges": 0,
        "symbol_entry_addresses": entry_addresses,
        "valid": valid,
    }
    if not valid:
        raise BuildError("invalid_current_graph")
    return cfg, edges, integrity


def current_extractor_hashes() -> dict[str, str]:
    cfg_path = ROOT / "scripts/data/cfg_extractor.py"
    dfg_path = ROOT / "scripts/data/dfg_extractor.py"
    combined = hashlib.sha256()
    for path in (cfg_path, dfg_path):
        combined.update(path.name.encode("utf-8"))
        combined.update(path.read_bytes())
    result = {
        "cfg": sha256_file(cfg_path),
        "dfg": sha256_file(dfg_path),
        "combined": combined.hexdigest(),
    }
    expected = {
        "cfg": "daebbbfa7ac53fed9104e66396bc861bc837a8cea5a948548204d34439ee553c",
        "dfg": "603c052e8a79e7f6f689e97acdfc9c87245505b4fbf497bc2c49c2343fb0ed12",
        "combined": "7a89b10f74754a8ff43580dba0cfb3348cd8e7b370e325ba8d31667c60ac04c1",
    }
    if result != expected:
        raise BuildError(f"current_extractor_hash_mismatch:{result}")
    return result


def fixed_r15_offsets(instruction: str) -> list[int]:
    offsets: list[int] = []
    for match in FIXED_R15_RE.finditer(instruction):
        amount = match.group("amount")
        if amount is None:
            offsets.append(0)
            continue
        value = int(amount, 0)
        offsets.append(-value if match.group("sign") == "-" else value)
    return offsets


def dart_utf16_code_units(value: str) -> list[int]:
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    return [int.from_bytes(encoded[index : index + 2], "little") for index in range(0, len(encoded), 2)]


def literal_payload(literal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    literal_type = literal.get("type")
    if literal_type == "string":
        code_units = literal.get("code_units")
        if isinstance(code_units, list):
            if any(
                not isinstance(unit, int)
                or isinstance(unit, bool)
                or not 0 <= unit <= 0xFFFF
                for unit in code_units
            ):
                raise BuildError("pool_string_code_units_invalid")
            return "string", {"code_units": list(code_units)}
        value = literal.get("value")
        if not isinstance(value, str):
            raise BuildError("pool_string_value_not_text")
        return "string", {"code_units": dart_utf16_code_units(value)}
    if literal_type == "int":
        decimal = literal.get("decimal")
        if isinstance(decimal, str):
            try:
                value = int(decimal, 10)
            except ValueError as error:
                raise BuildError("pool_int_decimal_invalid") from error
            if str(value) != decimal:
                raise BuildError("pool_int_decimal_not_canonical")
            return "int", {"decimal": decimal}
        value = literal.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise BuildError("pool_int_value_not_integer")
        return "int", {"decimal": str(value)}
    if literal_type == "double":
        bits_hex = literal.get("bits_hex")
        if isinstance(bits_hex, str):
            if not re.fullmatch(r"[0-9a-f]{16}", bits_hex):
                raise BuildError("pool_double_bits_not_canonical")
            return "double", {"bits_hex": bits_hex}
        value = literal.get("value")
        if isinstance(value, str):
            if value == "Infinity":
                numeric = math.inf
            elif value == "-Infinity":
                numeric = -math.inf
            elif value == "NaN":
                raise BuildError("pool_nan_requires_runtime_bit_receipt")
            else:
                raise BuildError(f"unsupported_pool_double:{value}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
        else:
            raise BuildError("pool_double_value_not_numeric")
        return "double", {"bits_hex": struct.pack(">d", numeric).hex()}
    if literal_type == "bool":
        value = literal.get("value")
        if not isinstance(value, bool):
            raise BuildError("pool_bool_value_not_boolean")
        return "bool", {"value": value}
    if literal_type == "null":
        return "null", {}
    raise BuildError(f"unsupported_pool_literal_type:{literal_type}")


def composite_value(node: dict[str, Any], *, depth: int = 0) -> dict[str, Any]:
    """Convert a complete static V8-profile node to the strict v3 value shape."""

    if depth > 64:
        raise BuildError("pool_composite_depth_exceeded")
    category = node.get("category")
    if category == "literal":
        kind, payload = literal_payload(node.get("literal") or {})
        return {"kind": kind, "payload": payload}
    if category == "nonliteral":
        if set(node) != {"category", "nonliteral_kind", "profile_type"}:
            raise BuildError("pool_nonliteral_descriptor_shape_invalid")
        nonliteral_kind = node.get("nonliteral_kind")
        profile_type = node.get("profile_type")
        if not isinstance(nonliteral_kind, str) or not isinstance(profile_type, str):
            raise BuildError("pool_nonliteral_descriptor_values_not_strings")
        expected_kind = NESTED_NONLITERAL_PROFILE_KIND.get(profile_type)
        if expected_kind is None or nonliteral_kind != expected_kind:
            raise BuildError(
                "unsupported_pool_nonliteral_descriptor_pair:"
                f"{nonliteral_kind}:{profile_type}"
            )
        return {
            "kind": "nonliteral",
            "payload": {
                "nonliteral_kind": nonliteral_kind,
                "profile_type": profile_type,
            },
        }
    if category != "composite":
        raise BuildError(
            f"unsupported_pool_composite_node:{category}:{canonical_json(node)}"
        )
    if node.get("complete") is not True:
        raise BuildError("incomplete_pool_composite")
    composite_type = node.get("composite_type")
    if composite_type not in {"array_storage", "map_storage"}:
        raise BuildError(f"unsupported_pool_composite_type:{composite_type}")
    raw_elements = node.get("elements")
    if not isinstance(raw_elements, list):
        raise BuildError("pool_composite_elements_not_list")
    elements: list[dict[str, Any]] = []
    for element in raw_elements:
        if not isinstance(element, dict) or set(element) != {"index", "value"}:
            raise BuildError("pool_composite_element_shape_invalid")
        index = element["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise BuildError("pool_composite_element_index_invalid")
        value = element["value"]
        if not isinstance(value, dict):
            raise BuildError("pool_composite_element_value_invalid")
        elements.append(
            {
                "index": index,
                "value": composite_value(value, depth=depth + 1),
            }
        )
    raw_omitted = node.get("omitted_edge_counts")
    if not isinstance(raw_omitted, dict):
        raise BuildError("pool_composite_omitted_edges_invalid")
    omitted: dict[str, int] = {}
    for key, value in raw_omitted.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise BuildError("pool_composite_omitted_edge_invalid")
        omitted[key] = value
    return {
        "kind": "composite",
        "payload": {
            "complete": True,
            "composite_type": composite_type,
            "elements": elements,
            "omitted_edge_counts": omitted,
        },
    }


def _entry_offset(entry: dict[str, Any]) -> int:
    numeric = entry.get("pp_offset")
    if isinstance(numeric, int) and not isinstance(numeric, bool):
        return numeric
    return int(str(entry.get("pool_offset")), 16)


def _exact_static_xrefs(receipt: dict[str, Any]) -> list[tuple[int, int]]:
    return sorted(
        [
        (_entry_offset(entry), int(str(use["pc"]), 16))
        for entry in receipt.get("entries") or []
        for use in entry.get("uses") or []
        if use.get("function_id") == TARGET
        ]
    )


def _exact_runtime_xrefs(
    receipt: dict[str, Any], *, candidate_base: int
) -> list[tuple[int, int]]:
    return sorted(
        [
        (_entry_offset(entry), int(str(use["pc"]), 16) - candidate_base)
        for entry in receipt.get("entries") or []
        for use in entry.get("uses") or []
        if use.get("function_id") == TARGET
        ]
    )


def _reconciled_value(
    static_entry: dict[str, Any], runtime_entry: dict[str, Any]
) -> dict[str, Any] | None:
    static_category = static_entry.get("category")
    runtime_category = runtime_entry.get("category")
    if static_category == "composite":
        return composite_value(static_entry)
    if runtime_category == "literal":
        runtime_kind, runtime_payload = literal_payload(
            runtime_entry.get("literal") or {}
        )
        runtime_value = {"kind": runtime_kind, "payload": runtime_payload}
        if static_category == "literal":
            static_literal = static_entry.get("literal") or {}
            static_kind = str(static_literal.get("type") or "")
            if static_kind != runtime_value["kind"]:
                raise BuildError(
                    f"static_runtime_literal_kind_mismatch:{static_kind}:{runtime_value['kind']}"
                )
            # Static doubles deliberately lack exact bits.  All other primitive
            # payloads must agree byte-for-byte with the AOT-only runtime view.
            if static_kind != "double":
                _, static_payload = literal_payload(static_literal)
                if static_payload != runtime_value["payload"]:
                    raise BuildError("static_runtime_literal_payload_mismatch")
        return runtime_value
    if static_category == "literal":
        static_kind, static_payload = literal_payload(static_entry.get("literal") or {})
        # Tagged small integers are stored directly in the pool slot, so the
        # runtime audit conservatively calls them ``untagged_or_smi``.  The
        # compiler profile proves the entry is an integer and carries its exact
        # canonical decimal value.
        if (
            static_kind == "int"
            and runtime_category == "nonliteral"
            and runtime_entry.get("nonliteral_kind") == "untagged_or_smi"
        ):
            return {"kind": static_kind, "payload": static_payload}
        raise BuildError(
            f"static_literal_missing_runtime_confirmation:{static_kind}:{runtime_category}"
        )
    if runtime_category == "literal":  # pragma: no cover - handled above
        raise AssertionError("unreachable_runtime_literal_branch")
    if static_category in {"nonliteral", "unresolved_object"} and runtime_category in {
        "nonliteral",
        "unresolved",
    }:
        return None
    raise BuildError(
        f"unreconciled_pool_entry:{static_category}:{runtime_category}"
    )


def model_pool_uses(
    static_receipt: dict[str, Any],
    runtime_receipt: dict[str, Any],
    *,
    candidate_base: int,
    sites: dict[int, dict[str, int]],
    cfg: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    static_xrefs = _exact_static_xrefs(static_receipt)
    runtime_xrefs = _exact_runtime_xrefs(
        runtime_receipt, candidate_base=candidate_base
    )
    if static_xrefs != runtime_xrefs:
        raise BuildError(
            f"static_runtime_xref_mismatch:{len(static_xrefs)}:{len(runtime_xrefs)}"
        )
    static_by_offset = {
        _entry_offset(entry): entry for entry in static_receipt.get("entries") or []
    }
    runtime_by_offset = {
        _entry_offset(entry): entry for entry in runtime_receipt.get("entries") or []
    }
    if set(static_by_offset) != set(runtime_by_offset):
        raise BuildError("static_runtime_pool_offset_set_mismatch")
    records: list[dict[str, Any]] = []
    excluded_non_graph_xrefs: list[dict[str, Any]] = []
    retained_xrefs = 0
    for pp_offset in sorted(static_by_offset):
        static_entry = static_by_offset[pp_offset]
        runtime_entry = runtime_by_offset[pp_offset]
        exact_uses = [
            use
            for use in runtime_entry.get("uses") or []
            if use.get("function_id") == TARGET
        ]
        if not exact_uses:
            continue
        use_sites: list[dict[str, int]] = []
        for use in exact_uses:
            absolute_pc = int(str(use.get("pc")), 16)
            if absolute_pc not in sites:
                excluded_non_graph_xrefs.append(
                    {
                        "pp_offset": pp_offset,
                        "function_offset": absolute_pc - candidate_base,
                        "reason": "deterministically_pruned_non_graph_instruction",
                        "static_category": static_entry.get("category"),
                        "runtime_category": runtime_entry.get("category"),
                    }
                )
                continue
            site = dict(sites[absolute_pc])
            instruction = str(cfg[site["block"]]["instructions"][site["instruction"]])
            if pp_offset not in fixed_r15_offsets(instruction):
                raise BuildError(
                    "pool_offset_not_present_at_graph_site:"
                    f"0x{pp_offset:x}:{site['block']}:{site['instruction']}:{instruction}"
                )
            use_sites.append(site)
            retained_xrefs += 1
        if not use_sites:
            continue
        value = _reconciled_value(static_entry, runtime_entry)
        if value is None:
            continue
        records.append(
            {
                "pp_offset": pp_offset,
                "kind": value["kind"],
                "payload": value["payload"],
                "use_sites": use_sites,
            }
        )
    total_exact = len(runtime_xrefs)
    accounted = retained_xrefs + len(excluded_non_graph_xrefs)
    if accounted != total_exact:
        raise BuildError(f"pool_xref_accounting_mismatch:{accounted}!={total_exact}")
    return records, {
        "scope": "canonical_graph_retained_fixed_r15_xrefs",
        "target_exact_xrefs": total_exact,
        "graph_retained_xrefs": retained_xrefs,
        "excluded_non_graph_xrefs": excluded_non_graph_xrefs,
        "excluded_non_graph_xref_count": len(excluded_non_graph_xrefs),
        "all_target_xrefs_accounted": True,
    }


def _artifact_name(row: dict[str, Any]) -> str:
    digest = sha256_text(str(row["task_id"]))[:16]
    return f"{int(row['split_row']):06d}_{digest}"


def _build_one(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload["row"]
    output = Path(payload["output"])
    dart_sdk = Path(payload["dart_sdk"])
    keep_aot = bool(payload["keep_aot"])
    paths = _toolchain_paths(dart_sdk)
    name = _artifact_name(row)
    split = str(row["split"])
    receipt_path = output / "rows" / split / f"{name}.json"
    expected_input_hash = sha256_text(canonical_json(row))
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        producer = existing.get("producer") or {}
        if (
            existing.get("schema") == ROW_SCHEMA
            and existing.get("build_input_sha256") == expected_input_hash
            and producer.get("script_sha256") == sha256_file(Path(__file__))
            and producer.get("pool_extractor_sha256")
            == sha256_file(ROOT / "scripts/data/extract_dart_aot_pool_receipt.py")
            and producer.get("runtime_pool_extractor_sha256")
            == sha256_file(
                ROOT / "scripts/data/extract_dart_aot_pool_runtime_receipt.py"
            )
            and producer.get("graph_builder_sha256")
            == sha256_file(RELEASE / "build_scrubbed_dataset.py")
            and existing.get("toolchain_manifest_sha256")
            == payload["toolchain_manifest_sha256"]
        ):
            return {"status": "resumed", "path": str(receipt_path), "row": existing}
        raise BuildError(f"stale_existing_receipt:{receipt_path}")

    started = time.monotonic()
    work_root = output / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{name}_", dir=work_root) as temporary:
        work = Path(temporary)
        source = work / "program.dart"
        kernel = work / "program.aot.dill"
        aot = work / "program.aot"
        profile = work / "profile.json"
        source.write_text(str(row["analysis_program"]), encoding="utf-8", newline="\n")
        if sha256_file(source) != row["analysis_program_sha256"]:
            raise BuildError("analysis_program_hash_mismatch")

        kernel_started = time.monotonic()
        run(
            [
                str(paths["runtime"]),
                str(paths["gen_kernel"]),
                "--platform",
                str(paths["platform"]),
                "--aot",
                "--tfa",
                "--target-os",
                "linux",
                "-o",
                str(kernel),
                str(source),
            ],
            cwd=work,
            timeout=float(payload["compile_timeout"]),
        )
        kernel_seconds = time.monotonic() - kernel_started

        snapshot_started = time.monotonic()
        snapshot = run(
            [
                str(paths["gen_snapshot"]),
                "--snapshot_kind=app-aot-elf",
                f"--elf={aot}",
                "--disassemble",
                "--disassemble-optimized",
                "--disassemble-relative",
                "--code-comments",
                f"--write-v8-snapshot-profile-to={profile}",
                str(kernel),
            ],
            cwd=work,
            timeout=float(payload["compile_timeout"]),
        )
        snapshot_seconds = time.monotonic() - snapshot_started
        disassembly_bytes = snapshot.stderr.encode("utf-8")
        profile_bytes = profile.read_bytes()
        static_receipt = build_pool_receipt(
            snapshot.stderr,
            json.loads(profile_bytes),
            target=TARGET,
            disassembly_sha256=sha256_bytes(disassembly_bytes),
            profile_sha256=sha256_bytes(profile_bytes),
        )

        run([str(paths["runtime"]), str(aot)], cwd=work, timeout=30)
        assembly, entries = graph_builder.extract_symbol_assembly(
            aot,
            TARGET,
            "program.dart",
            row.get("source_symbols") or {"functions": [TARGET], "types": []},
            readelf=str(payload["readelf"]),
            objdump=str(payload["objdump"]),
        )
        cfg, edges, integrity = graph_from_current_extractor(assembly, entries)
        graph_builder.validate_cfg_mnemonics(cfg)
        candidate_base, sites, site_accounting = instruction_sites(assembly, cfg)
        integrity["instruction_site_accounting"] = site_accounting
        if entries != [f"0x{candidate_base:x}"]:
            raise BuildError(f"candidate_entry_mismatch:{entries}:0x{candidate_base:x}")
        runtime_started = time.monotonic()
        if static_receipt["summary"]["pool_uses"]:
            runtime_receipt = build_runtime_receipt(
                aot,
                paths["runtime"],
                target=TARGET,
                nm=str(payload["nm"]),
                objdump=str(payload["objdump"]),
                gdb=str(payload["gdb"]),
            )
        else:
            runtime_receipt = {
                "schema": "dart-aot-runtime-pool-receipt-v1",
                "source_blind": True,
                "target_function": TARGET,
                "layout_contract": SDK_LAYOUT_CONTRACT,
                "inputs": {
                    "aot_sha256": sha256_file(aot),
                    "dartaotruntime_sha256": sha256_file(paths["runtime"]),
                },
                "target_scope": [],
                "entries": [],
                "pool_literal_presence": {"bool": False, "null": False},
                "summary": {
                    "target_functions": 0,
                    "unique_pool_entries": 0,
                    "pool_uses": 0,
                    "literal_entries": 0,
                    "bool_pool_entries": 0,
                    "null_pool_entries": 0,
                    "nonliteral_entries": 0,
                    "unresolved_entries": 0,
                },
            }
        runtime_seconds = time.monotonic() - runtime_started
        binary_pool_uses, pool_projection_accounting = model_pool_uses(
            static_receipt,
            runtime_receipt,
            candidate_base=candidate_base,
            sites=sites,
            cfg=cfg,
        )

        aot_hash = sha256_file(aot)
        destination: Path | None = None
        if keep_aot:
            destination = output / "aot" / split / f"{name}.aot"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(aot, destination)
            if sha256_file(destination) != aot_hash:
                raise BuildError("copied_aot_hash_mismatch")

        cfg_path = ROOT / "scripts/data/cfg_extractor.py"
        dfg_path = ROOT / "scripts/data/dfg_extractor.py"
        extractor_hashes = current_extractor_hashes()
        extractor_sha = extractor_hashes["combined"]
        row_output = {
            "schema": ROW_SCHEMA,
            "task_id": row["task_id"],
            "split": split,
            "split_row": row["split_row"],
            "family": row.get("family"),
            "compact_private_metadata": row.get("compact_private_metadata"),
            "function": TARGET,
            "lang": "dart",
            "cfg": cfg,
            "edges": edges,
            "graph_v2": {
                "schema": GRAPH_SCHEMA,
                "assembly_sha256": sha256_text(assembly),
                "extractor_sha256": extractor_sha,
                "max_block_instrs": graph_builder.MAX_BLOCK_INSTRS,
                "max_dataflow_edges": 0,
                "symbol_entry_addresses": entries,
            },
            "integrity": integrity,
            "binary_pool_uses": binary_pool_uses,
            "pool_projection_accounting": pool_projection_accounting,
            "binary_pool_private_receipt": {
                "schema": "dart-aot-reconciled-pool-receipts-v1",
                "static": static_receipt,
                "runtime": runtime_receipt,
                "projection_sha256": sha256_text(canonical_json(binary_pool_uses)),
            },
            "build_input_sha256": expected_input_hash,
            "analysis_program_sha256": row["analysis_program_sha256"],
            "function_source_sha256": row["function_source_sha256"],
            "toolchain_manifest_sha256": payload["toolchain_manifest_sha256"],
            "aot": {
                "path": str(destination.relative_to(output)) if destination else None,
                "sha256": aot_hash,
                "size_bytes": aot.stat().st_size,
            },
            "producer": {
                "script_sha256": sha256_file(Path(__file__)),
                "pool_extractor_sha256": sha256_file(
                    ROOT / "scripts/data/extract_dart_aot_pool_receipt.py"
                ),
                "runtime_pool_extractor_sha256": sha256_file(
                    ROOT / "scripts/data/extract_dart_aot_pool_runtime_receipt.py"
                ),
                "cfg_extractor_sha256": sha256_file(cfg_path),
                "dfg_extractor_sha256": sha256_file(dfg_path),
                "graph_builder_sha256": sha256_file(
                    RELEASE / "build_scrubbed_dataset.py"
                ),
            },
            "timing_seconds": {
                "kernel": round(kernel_seconds, 4),
                "snapshot": round(snapshot_seconds, 4),
                "runtime_pool": round(runtime_seconds, 4),
                "total": round(time.monotonic() - started, 4),
            },
        }
        write_json_atomic(receipt_path, row_output)
        return {"status": "built", "path": str(receipt_path), "row": row_output}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BuildError(f"blank_jsonl_line:{path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BuildError(f"non_object_jsonl_line:{path}:{line_number}")
            result.append(value)
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def _validate_input_rows(rows: list[dict[str, Any]], split: str) -> None:
    task_ids: set[str] = set()
    for position, row in enumerate(rows):
        if row.get("schema") != "dart-source-only-aot-build-input-v1":
            raise BuildError(f"input_schema_mismatch:{position}")
        if row.get("split") != split or row.get("split_row") != position:
            raise BuildError(f"input_split_alignment_mismatch:{position}")
        if row.get("function") != TARGET:
            raise BuildError(f"input_target_mismatch:{position}")
        metadata = row.get("compact_private_metadata")
        if not isinstance(metadata, dict):
            raise BuildError(f"input_compact_private_metadata_invalid:{position}")
        if (
            metadata.get("target_function") != TARGET
            or metadata.get("phase0_split") != split
            or metadata.get("family") != row.get("family")
        ):
            raise BuildError(f"input_compact_private_metadata_drift:{position}")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in task_ids:
            raise BuildError(f"input_task_id_invalid_or_duplicate:{position}:{task_id}")
        task_ids.add(task_id)
        program = str(row.get("analysis_program") or "")
        if sha256_text(program) != row.get("analysis_program_sha256"):
            raise BuildError(f"input_program_hash_mismatch:{position}")
        if not program.rstrip().endswith("void main() {}"):
            raise BuildError(f"analysis_main_not_neutral:{position}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--dart-sdk", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--compile-timeout", type=float, default=240.0)
    parser.add_argument("--gdb", default="/usr/bin/gdb")
    parser.add_argument("--nm", default="nm")
    parser.add_argument("--objdump", default="x86_64-linux-gnu-objdump")
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--no-keep-aot", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.jobs < 1 or args.jobs > 8:
        raise SystemExit("jobs_must_be_between_1_and_8")
    rows = load_jsonl(args.input)
    _validate_input_rows(rows, args.split)
    if args.start < 0 or args.start >= len(rows):
        raise SystemExit("start_must_select_an_input_row")
    rows = rows[args.start :]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("limit_must_be_positive")
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    toolchain = toolchain_manifest(
        args.dart_sdk,
        gdb=args.gdb,
        nm=args.nm,
        objdump=args.objdump,
        readelf=args.readelf,
    )
    toolchain_path = args.output / "dart_toolchain_manifest.json"
    if toolchain_path.exists():
        observed = json.loads(toolchain_path.read_text(encoding="utf-8"))
        if observed != toolchain:
            raise SystemExit("existing_toolchain_manifest_mismatch")
    else:
        write_json_atomic(toolchain_path, toolchain)
    toolchain_manifest_sha256 = sha256_file(toolchain_path)

    payloads = [
        {
            "row": row,
            "output": str(args.output),
            "dart_sdk": str(args.dart_sdk),
            "keep_aot": not args.no_keep_aot,
            "compile_timeout": args.compile_timeout,
            "gdb": args.gdb,
            "nm": args.nm,
            "objdump": args.objdump,
            "readelf": args.readelf,
            "toolchain_manifest_sha256": toolchain_manifest_sha256,
        }
        for row in rows
    ]
    completed: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    with futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        pending = {executor.submit(_build_one, payload): payload for payload in payloads}
        for ordinal, future in enumerate(futures.as_completed(pending), 1):
            payload = pending[future]
            row = payload["row"]
            try:
                result = future.result()
                completed[int(row["split_row"])] = result["row"]
                summary = result["row"]["binary_pool_private_receipt"]["static"]["summary"]
                print(
                    canonical_json(
                        {
                            "done": ordinal,
                            "total": len(payloads),
                            "status": result["status"],
                            "split_row": row["split_row"],
                            "task_id": row["task_id"],
                            "pool_literals": summary["literal_entries"],
                            "pool_uses": summary["pool_uses"],
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "split_row": row["split_row"],
                    "task_id": row["task_id"],
                    "error": f"{type(error).__name__}:{error}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(canonical_json({"status": "failed", **failure}), flush=True)

    ordered = [completed[index] for index in sorted(completed)]
    prepared = args.output / "prepared" / f"{args.split}_codec_private.jsonl"
    write_jsonl(prepared, ordered)
    failure_path = args.output / "quarantine" / f"{args.split}.jsonl"
    write_jsonl(failure_path, failures)

    manifest_path = args.output / "manifests" / f"{args.split}.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "split": args.split,
        "input": {
            "path": str(args.input),
            "sha256": sha256_file(args.input),
            "rows_selected": len(rows),
        },
        "counts": {
            "built_or_resumed": len(ordered),
            "failed": len(failures),
            "literal_pool_records": sum(len(row["binary_pool_uses"]) for row in ordered),
            "literal_pool_use_sites": sum(
                len(record["use_sites"])
                for row in ordered
                for record in row["binary_pool_uses"]
            ),
            "excluded_non_graph_pool_xrefs": sum(
                row["pool_projection_accounting"]["excluded_non_graph_xref_count"]
                for row in ordered
            ),
        },
        "outputs": {
            "prepared_codec_private": {
                "path": str(prepared.relative_to(args.output)),
                "sha256": sha256_file(prepared),
            },
            "quarantine": {
                "path": str(failure_path.relative_to(args.output)),
                "sha256": sha256_file(failure_path),
            },
            "dart_toolchain_manifest": {
                "path": str(toolchain_path.relative_to(args.output)),
                "sha256": sha256_file(toolchain_path),
            },
        },
        "producer_sha256": sha256_file(Path(__file__)),
        "pool_extractor_sha256": sha256_file(
            ROOT / "scripts/data/extract_dart_aot_pool_receipt.py"
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gates": {
            "all_selected_rows_built": len(ordered) == len(rows),
            "zero_failures": not failures,
            "source_blind_pool_extraction": all(
                row["binary_pool_private_receipt"]["static"].get("source_blind") is True
                and row["binary_pool_private_receipt"]["runtime"].get("source_blind") is True
                for row in ordered
            ),
            "all_target_pool_xrefs_accounted": all(
                row["pool_projection_accounting"].get("all_target_xrefs_accounted")
                is True
                for row in ordered
            ),
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if failures and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
