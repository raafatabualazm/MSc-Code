#!/usr/bin/env python3
"""Fail-closed audit of a harmonized HumanEval compact-Qwen measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_FIELDS = {
    "compact_input_ids",
    "compact_codec_sha256",
    "compact_codebook_sha256",
    "compact_tokenizer_sha256",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--compact-dir", required=True, type=Path)
    parser.add_argument("--harmonized-public", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-codebook-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=154)
    parser.add_argument("--max-source-tokens", type=int, default=9000)
    args = parser.parse_args()

    report_path = args.compact_dir / "preflight_report.json"
    alignment_path = args.compact_dir / "alignment_private.jsonl"
    model_path = args.compact_dir / "compact_model_inputs.jsonl"
    codebook_path = args.compact_dir / "codebook.json"
    failures_path = args.compact_dir / "failures.jsonl"
    quarantine_path = args.compact_dir / "quarantine.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    alignment = rows(alignment_path)
    model = rows(model_path)
    public = rows(args.harmonized_public)
    failures = rows(failures_path)
    quarantine = rows(quarantine_path)
    if sha(codebook_path) != args.expected_codebook_sha256:
        raise ValueError("sealed training codebook SHA-256 mismatch")
    if report.get("contract", {}).get("compact_codebook_sha256") != args.expected_codebook_sha256:
        raise ValueError("preflight report codebook binding mismatch")
    if report.get("passed") is not True or failures or quarantine:
        raise ValueError("compact builder did not pass cleanly")
    if len(alignment) != len(model):
        raise ValueError("model/private alignment row-count mismatch")
    for index, (private, public_model) in enumerate(zip(alignment, model)):
        if private.get("model_row") != index:
            raise ValueError(f"non-contiguous model_row at {index}")
        if set(public_model) != MODEL_FIELDS:
            raise ValueError(f"model row {index} violates strict field allowlist")
        if public_model["compact_codebook_sha256"] != args.expected_codebook_sha256:
            raise ValueError(f"model row {index} codebook mismatch")
    measured = [row for row in alignment if row.get("role") == "measure"]
    if len(measured) != args.expected_rows or len(public) != args.expected_rows:
        raise ValueError("HumanEval measure row count mismatch")
    measured_ids = [row.get("task_id") for row in measured]
    public_ids = [row.get("task_id") for row in public]
    if len(set(measured_ids)) != args.expected_rows or set(measured_ids) != set(public_ids):
        raise ValueError("HumanEval task-ID coverage mismatch")
    lengths = [int(row["source_tokens"]) for row in measured]
    if max(lengths) > args.max_source_tokens:
        raise ValueError("HumanEval compact source exceeds token contract")
    invariants = report.get("lossless_invariants", {})
    if invariants.get("unknown_tokens") != 0 or invariants.get("truncated_rows") != 0:
        raise ValueError("lossless invariant failed")
    if invariants.get("raw_fallback_is_reversible") is not True:
        raise ValueError("raw fallback is not certified reversible")
    if report.get("rows_by_role", {}).get("measure") != args.expected_rows:
        raise ValueError("preflight role count mismatch")

    measured_model_rows = [int(row["model_row"]) for row in measured]
    measured_model = [model[index] for index in measured_model_rows]
    base_vocab_size = int(json.loads(codebook_path.read_text(encoding="utf-8"))["base_vocab_size"])
    compact_positions = [
        token
        for row in measured_model
        for token in row["compact_input_ids"]
    ]
    custom_positions = sum(token >= base_vocab_size for token in compact_positions)
    fallback_counts = [int(row.get("fallback_instructions", 0)) for row in measured]
    canonical_instruction_positions = sum(
        1
        for row in public
        for block in row.get("cfg", [])
        for instruction in block.get("instructions", [])
        if str(instruction).strip() != "static void candidate(void)"
    )
    fallback_instructions = sum(fallback_counts)
    if canonical_instruction_positions <= 0:
        raise ValueError("HumanEval contains no canonical instruction positions")
    if fallback_instructions > canonical_instruction_positions:
        raise ValueError("fallback instruction count exceeds canonical instructions")

    gate = {
        "schema": "compact-humaneval-measure-gate-v1",
        "passed": True,
        "lossless_domain": invariants.get("lossless_domain"),
        "measure_rows": len(measured),
        "measure_tasks": len(set(measured_ids)),
        "failures": 0,
        "quarantined": 0,
        "unknown_tokens": 0,
        "truncated_rows": 0,
        "raw_fallback_is_reversible": True,
        "train_only_codebook_generalization": {
            "rows_with_fallback": sum(value > 0 for value in fallback_counts),
            "row_fallback_rate": sum(value > 0 for value in fallback_counts) / len(fallback_counts),
            "fallback_instructions": fallback_instructions,
            "canonical_instruction_positions": canonical_instruction_positions,
            "instruction_fallback_rate": fallback_instructions / canonical_instruction_positions,
            "fallback_per_row_p50": percentile(fallback_counts, 0.50),
            "fallback_per_row_p95": percentile(fallback_counts, 0.95),
            "fallback_per_row_max": max(fallback_counts),
            "compact_positions": len(compact_positions),
            "custom_atom_positions": custom_positions,
            "custom_atom_position_fraction": custom_positions / len(compact_positions),
            "interpretation": "Fallback is exact native-Qwen tokenization, not UNK or information loss.",
        },
        "source_tokens": {
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths),
            "limit": args.max_source_tokens,
        },
        "bindings": {
            "harmonized_public_sha256": sha(args.harmonized_public),
            "codebook_sha256": sha(codebook_path),
            "preflight_report_sha256": sha(report_path),
            "compact_model_inputs_sha256": sha(model_path),
            "alignment_private_sha256": sha(alignment_path),
            "dfg_extractor_sha256": invariants.get("dfg_extractor_sha256"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
