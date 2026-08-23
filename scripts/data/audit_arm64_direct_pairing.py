#!/usr/bin/env python3
"""Audit ARM64/x86 source pairing and direct-Qwen compression readiness.

This is intentionally an audit, not a dataset builder.  It never emits model
rows and it never uses held-out rows to fit the prospective instruction
codebook.  The ARM training split supplies codebook entries; the ARM evaluation
split is measurement-only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
ARM_FULL = ROOT / "data/datasets/arm64_graphv2/flutter_function_assembly_pool_graphv2.jsonl"
ARM_TRAIN = ROOT / "data/datasets/arm64_graphv2/flutter_train_graphv2.jsonl"
ARM_EVAL = ROOT / "data/datasets/arm64_graphv2/flutter_eval_graphv2.jsonl"
X86_SYNTHETIC = ROOT / "data/datasets/synthetic_pool_reward_clean_graphv2.jsonl"
MASTER = ROOT / "master_dart_cfg_dfg/master_dart_cfg_dfg_train.jsonl"
LEDGER = ROOT / "scrubbed_master_v2_release/master_dart_graphv2_compile_ledger.jsonl"
QUARANTINE = ROOT / "scrubbed_master_v2_release/master_dart_graphv2_quarantine.jsonl"
CURRENT_DFG = ROOT / "scripts/data/dfg_extractor.py"
RELEASE_DFG = ROOT / "scrubbed_master_v2_release/extractors/dfg_extractor.py"
CURRENT_CFG = ROOT / "scripts/data/cfg_extractor.py"

TARGET_ANNOTATION = re.compile(r"0x([0-9a-fA-F]+)\s*<([^>]+)>")
BRANCH_CALL_OPS = {"b", "bl", "blr", "br", "ret", "cbz", "cbnz", "tbz", "tbnz"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def raw_source_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("dart_source") or "").encode()).hexdigest()


def load_dfg(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_cross_block_dfg


def combined_extractor_sha256(cfg_path: Path, dfg_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (cfg_path, dfg_path):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def canonicalize_arm_row(row: dict[str, Any]) -> list[list[str]]:
    """Leakage-safe prospective ARM representation used only for token audit.

    Control-transfer destinations inside the slice become block references.
    External stripped-snapshot destinations become one architecture-neutral
    marker.  Numeric immediates on ordinary instructions remain untouched.
    """

    starts = {
        str(block.get("start_address", "")).lower().removeprefix("0x"): int(block["id"])
        for block in row.get("cfg") or []
    }
    result: list[list[str]] = []
    for block in row.get("cfg") or []:
        instructions: list[str] = []
        for raw in block.get("instructions") or []:
            text = " ".join(str(raw).strip().split())
            if not text:
                continue
            opcode = text.split(None, 1)[0].lower()
            if opcode in BRANCH_CALL_OPS or opcode.startswith("b."):
                def replace_target(match: re.Match[str]) -> str:
                    address = match.group(1).lower()
                    return f"@B{starts[address]}" if address in starts else "@EXT"

                text = TARGET_ANNOTATION.sub(replace_target, text)
            else:
                text = re.sub(r"<[^>]+>", "@SYM", text)
            instructions.append(re.sub(r"\s*,\s*", ",", text))
        result.append(instructions)
    return result


def prospective_length(
    row: dict[str, Any],
    canonical_blocks: list[list[str]],
    codebook: set[str],
    native_token_lengths: dict[str, int],
) -> tuple[int, int]:
    # Version, ISA, entry marker, entry IDs, blocks marker.
    length = 4 + len(row.get("integrity", {}).get("entry_blocks") or [0])
    fallback = 0
    for instructions in canonical_blocks:
        length += 1  # block atom
        for instruction in instructions:
            if instruction in codebook:
                length += 1
            else:
                # Raw-start + raw-end atoms plus native Qwen tokens.
                length += 2 + native_token_lengths[instruction]
                fallback += 1
    cfg_edges = sum(edge.get("edge_type") != "dataflow" for edge in row.get("edges") or [])
    length += 2 + 3 * cfg_edges  # CFG marker, triples, end marker.
    return length, fallback


def distribution(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)

    def percentile(q: float) -> int:
        return ordered[round((len(ordered) - 1) * q)]

    return {
        "min": min(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(ordered),
        "over_9000": sum(value > 9000 for value in ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/arm64_direct_pairing_audit_20260719.json",
    )
    args = parser.parse_args()
    started = time.time()

    arm_rows = [row for _, row in jsonl(ARM_FULL)]
    arm_train_rows = [row for _, row in jsonl(ARM_TRAIN)]
    arm_eval_rows = [row for _, row in jsonl(ARM_EVAL)]
    x86_rows = [row for _, row in jsonl(X86_SYNTHETIC)]

    arm_by_source = {raw_source_sha256(row): row for row in arm_rows}
    x86_by_source = {raw_source_sha256(row): row for row in x86_rows}
    common_sources = set(arm_by_source) & set(x86_by_source)
    source_exact = sum(
        arm_by_source[digest].get("dart_source") == x86_by_source[digest].get("dart_source")
        for digest in common_sources
    )
    original_assembly_link_exact = sum(
        arm_by_source[digest].get("original_dart_aot_assembly_sha256")
        == x86_by_source[digest].get("graph_v2", {}).get("assembly_sha256")
        for digest in common_sources
    )

    x86_by_index = {index: row for index, row in enumerate(x86_rows)}
    master_index_to_source: dict[int, str] = {}
    for master_index, row in jsonl(MASTER):
        provenance = row.get("provenance") or {}
        if provenance.get("source_dataset") != "synthetic_pool_reward_clean_graphv2":
            continue
        source_index = int(provenance["source_index"])
        master_index_to_source[master_index] = raw_source_sha256(x86_by_index[source_index])
    retained_indices = {
        int(row["source_index"])
        for _, row in jsonl(LEDGER)
        if row.get("status") == "retained"
    }
    quarantine_indices = {int(row["source_index"]) for _, row in jsonl(QUARANTINE)}
    retained_paired_sources = {
        master_index_to_source[index]
        for index in retained_indices
        if master_index_to_source.get(index) in common_sources
    }
    quarantined_arm_sources = {
        master_index_to_source[index]
        for index in quarantine_indices
        if master_index_to_source.get(index) in common_sources
    }
    arm_train_sources = {raw_source_sha256(row) for row in arm_train_rows}
    arm_eval_sources = {raw_source_sha256(row) for row in arm_eval_rows}

    private_field_counts: collections.Counter[str] = collections.Counter()
    exact_symbol_assembly_rows = 0
    exact_symbol_cfg_rows = 0
    assembly_libapp_hash_rows = 0
    for row in arm_rows:
        for field in (
            "dart_source",
            "tests",
            "dart_function_signature",
            "function",
            "filename",
            "flutter_function_symbol_ranges",
            "flutter_project_dir",
            "flutter_artifact",
            "flutter_libapp_so",
            "flutter_split_debug_info",
        ):
            if row.get(field) not in (None, "", [], {}):
                private_field_counts[field] += 1
        assembly = str(row.get("assembly") or "")
        cfg_text = "\n".join(
            instruction
            for block in row.get("cfg") or []
            for instruction in block.get("instructions") or []
        )
        names = [
            str(item.get("name") or "")
            for item in row.get("flutter_function_symbol_ranges") or []
            if item.get("name")
        ]
        exact_symbol_assembly_rows += any(name in assembly for name in names)
        exact_symbol_cfg_rows += any(name in cfg_text for name in names)
        assembly_libapp_hash_rows += "libapp_sha256:" in assembly

    current_build_dfg = load_dfg(CURRENT_DFG, "arm_pair_current_dfg")
    release_build_dfg = load_dfg(RELEASE_DFG, "arm_pair_release_dfg")
    current_exact = 0
    release_exact = 0
    release_abs_edge_count_delta = 0
    for row in arm_rows:
        cfg_edges = [edge for edge in row["edges"] if edge["edge_type"] != "dataflow"]
        stored = sorted(
            (edge["source"], edge["target"], "dataflow")
            for edge in row["edges"]
            if edge["edge_type"] == "dataflow"
        )
        current = sorted(
            (edge["source"], edge["target"], "dataflow")
            for edge in current_build_dfg(row["cfg"], cfg_edges, max_edges=100000)
        )
        release = sorted(
            (edge["source"], edge["target"], "dataflow")
            for edge in release_build_dfg(row["cfg"], cfg_edges, max_edges=100000)
        )
        current_exact += current == stored
        release_exact += release == stored
        release_abs_edge_count_delta += abs(len(release) - len(stored))

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    train_canonical = [(row, canonicalize_arm_row(row)) for row in arm_train_rows]
    eval_canonical = [(row, canonicalize_arm_row(row)) for row in arm_eval_rows]
    eval_instruction_set = {
        instruction
        for _, blocks in eval_canonical
        for instructions in blocks
        for instruction in instructions
    }
    native_token_lengths = {
        instruction: len(tokenizer.encode(instruction, add_special_tokens=False).ids)
        for instruction in eval_instruction_set
    }
    frequency = collections.Counter(
        instruction
        for _, blocks in train_canonical
        for instructions in blocks
        for instruction in instructions
    )
    compression: dict[str, Any] = {}
    for size in (0, 1024, 4096, 8192, 16384):
        codebook = {instruction for instruction, _ in frequency.most_common(size)}
        lengths: list[int] = []
        fallbacks: list[int] = []
        for row, blocks in eval_canonical:
            length, fallback = prospective_length(row, blocks, codebook, native_token_lengths)
            lengths.append(length)
            fallbacks.append(fallback)
        compression[str(size)] = {
            **distribution(lengths),
            "codebook_entries": len(codebook),
            "fallback_instructions_total": sum(fallbacks),
            "fallback_instructions_max_row": max(fallbacks),
        }

    opcodes = collections.Counter(
        instruction.split(None, 1)[0].lower()
        for _, blocks in train_canonical + eval_canonical
        for instructions in blocks
        for instruction in instructions
    )
    report = {
        "schema": "arm64-direct-pairing-audit-v1",
        "inputs": {
            (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)): {
                "sha256": file_sha256(path)
            }
            for path in (
                ARM_FULL,
                ARM_TRAIN,
                ARM_EVAL,
                X86_SYNTHETIC,
                MASTER,
                LEDGER,
                QUARANTINE,
                CURRENT_CFG,
                CURRENT_DFG,
                RELEASE_DFG,
                args.tokenizer_json,
            )
        },
        "pairing": {
            "arm_rows": len(arm_rows),
            "x86_synthetic_rows": len(x86_rows),
            "same_raw_source_pairs": len(common_sources),
            "byte_exact_raw_source_pairs": source_exact,
            "arm_original_x86_assembly_hash_links_exact": original_assembly_link_exact,
            "release_retained_same_original_source_pairs": len(retained_paired_sources),
            "release_retained_pairs_in_old_arm_train": len(retained_paired_sources & arm_train_sources),
            "release_retained_pairs_in_old_arm_eval": len(retained_paired_sources & arm_eval_sources),
            "release_quarantined_arm_sources": len(quarantined_arm_sources),
            "strong_same_neutral_source_rebuild_pairs": 0,
            "strong_pair_reason": (
                "Existing ARM binaries were built from original named sources; scrubbed x86 release "
                "binaries were rebuilt from candidate-neutral sources. No shared neutral-source build "
                "manifest binds both architecture artifacts."
            ),
        },
        "leakage": {
            "rows_with_private_fields": dict(sorted(private_field_counts.items())),
            "rows_with_exact_recorded_symbol_name_in_assembly": exact_symbol_assembly_rows,
            "rows_with_exact_recorded_symbol_name_in_cfg_instructions": exact_symbol_cfg_rows,
            "rows_with_linkable_libapp_sha256_in_assembly_header": assembly_libapp_hash_rows,
            "safe_as_model_input": False,
            "required_policy": "strict compact-field allowlist; all IDs, source, tests, paths and pair bindings remain private",
        },
        "dfg_reproduction": {
            "arm_rows": len(arm_rows),
            "recorded_combined_cfg_dfg_sha256": arm_rows[0]["graph_v2"]["extractor_sha256"],
            "current_combined_cfg_dfg_sha256": combined_extractor_sha256(CURRENT_CFG, CURRENT_DFG),
            "current_dfg_exact_rows": current_exact,
            "release_pinned_dfg_sha256": file_sha256(RELEASE_DFG),
            "release_pinned_dfg_exact_rows": release_exact,
            "release_pinned_absolute_edge_count_delta": release_abs_edge_count_delta,
        },
        "prospective_direct_qwen_arm64_compression": {
            "codebook_fit": str(ARM_TRAIN.relative_to(ROOT)),
            "measurement_only": str(ARM_EVAL.relative_to(ROOT)),
            "tokenizer_json_sha256": file_sha256(args.tokenizer_json),
            "canonical_train_unique_instructions": len(frequency),
            "observed_opcode_count": len(opcodes),
            "observed_opcodes": dict(sorted(opcodes.items())),
            "logical_token_accounting": (
                "one source-only atom per codebook instruction/block/control marker; native Qwen "
                "tokens plus two delimiters for fallback instructions; explicit CFG triples; DFG regenerated"
            ),
            "measurements": compression,
        },
        "blockers": [
            "The current compact codec is x86-only (hard-coded x86 mnemonic allowlist, AX64 marker and x86_64 decode result).",
            "The codec's default frozen release DFG extractor reproduces only 4/1714 ARM rows; ARM graph-v2.1 requires the recorded 7a89 combined extractor family.",
            "The ARM JSONL is an unsplit private record, not a public model-input artifact.",
            "The original Flutter/APK build-and-slice script, source manifest, APKs and libapp.so files are absent from this workspace and the CPU build VM.",
            "No manifest records the original Flutter/Dart SDK revisions, NDK/binutils revisions, or complete build command.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
