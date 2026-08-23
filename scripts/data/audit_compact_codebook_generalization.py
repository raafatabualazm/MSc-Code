#!/usr/bin/env python3
"""Audit train-only compact-codebook coverage without treating fallback as loss.

The compact codec has no UNK path: an instruction absent from the fitted
one-token codebook is emitted between reversible raw delimiters using Qwen's
native tokenizer.  This audit reports both row-level fallback incidence and the
more meaningful instruction-position fallback rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_compact_qwen_v1 import canonicalize


MODEL_FIELDS = {
    "compact_input_ids",
    "compact_codec_sha256",
    "compact_codebook_sha256",
    "compact_tokenizer_sha256",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--compact-dir", required=True, type=Path)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=["fit", "measure"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-codebook-sha256", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()

    codebook_path = args.compact_dir / "codebook.json"
    alignment_path = args.compact_dir / "alignment_private.jsonl"
    model_path = args.compact_dir / "compact_model_inputs.jsonl"
    preflight_path = args.compact_dir / "preflight_report.json"
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    alignment = read_jsonl(alignment_path)
    model = read_jsonl(model_path)
    public = read_jsonl(args.public)
    if sha256(codebook_path) != args.expected_codebook_sha256:
        raise ValueError("codebook SHA-256 mismatch")
    if len(alignment) != len(model):
        raise ValueError("alignment/model row-count mismatch")
    for index, (audit, visible) in enumerate(zip(alignment, model, strict=True)):
        if audit.get("model_row") != index:
            raise ValueError(f"invalid model_row at row {index}")
        if set(visible) != MODEL_FIELDS:
            raise ValueError(f"strict model schema violation at row {index}")

    selected = [row for row in alignment if row.get("role") == args.role]
    if len(selected) != args.expected_rows or len(public) != args.expected_rows:
        raise ValueError("selected/public row-count mismatch")
    selected_ids = [str(row.get("task_id") or "") for row in selected]
    public_ids = [str(row.get("task_id") or "") for row in public]
    if len(set(selected_ids)) != args.expected_rows or set(selected_ids) != set(public_ids):
        raise ValueError("selected/public task-ID coverage mismatch")

    selected_model = [model[int(row["model_row"])] for row in selected]
    base_vocab = int(codebook["base_vocab_size"])
    instruction_atoms = len(codebook["expansions"])
    atom_ids = {str(key): int(value) for key, value in codebook["source_atom_ids"].items()}
    block_atoms = {
        atom_ids[f"<B{index}>"] for index in range(int(codebook["max_blocks"]))
    }
    source_ids = {int(value) for value in codebook["source_token_expansions"]}
    positions = [token for row in selected_model for token in row["compact_input_ids"]]
    custom = [token for token in positions if token >= base_vocab]
    unique_custom = set(custom)
    unique_instruction_atoms = {
        token for token in unique_custom if base_vocab <= token < base_vocab + instruction_atoms
    }
    unique_block_atoms = unique_custom & block_atoms
    unique_control_atoms = unique_custom - unique_instruction_atoms - unique_block_atoms

    fallback_counts = [int(row.get("fallback_instructions", 0)) for row in selected]
    source_lengths = [int(row["source_tokens"]) for row in selected]
    instruction_frequency: Counter[str] = Counter()
    for row in public:
        compact = canonicalize(row)
        instruction_frequency.update(
            instruction
            for block in compact["blocks"]
            for instruction in block["instructions"]
        )
    known = set(str(value) for value in codebook["expansions"])
    oov = {
        instruction: count
        for instruction, count in instruction_frequency.items()
        if instruction not in known
    }
    fallback_instructions = sum(fallback_counts)
    oov_positions = sum(oov.values())
    if fallback_instructions != oov_positions:
        raise ValueError(
            f"alignment fallback count {fallback_instructions} != canonical OOV {oov_positions}"
        )
    instruction_positions = sum(instruction_frequency.values())
    if not positions or instruction_positions <= 0:
        raise ValueError("empty compact or instruction stream")

    fit_custom = {
        token
        for index, row in enumerate(alignment)
        if row.get("role") == "fit"
        for token in model[index]["compact_input_ids"]
        if token >= base_vocab
    }
    report = {
        "schema": "compact-codebook-generalization-audit-v1",
        "passed": True,
        "role": args.role,
        "rows": len(selected),
        "lossless_fallback": True,
        "fallback_interpretation": (
            "Exact canonical instruction text is encoded with native Qwen tokens; "
            "fallback is neither UNK nor truncation."
        ),
        "fallback": {
            "rows": sum(value > 0 for value in fallback_counts),
            "row_rate": sum(value > 0 for value in fallback_counts) / len(fallback_counts),
            "instructions": fallback_instructions,
            "instruction_positions": instruction_positions,
            "instruction_rate": fallback_instructions / instruction_positions,
            "unique_instructions": len(oov),
            "per_row_p50": percentile(fallback_counts, 0.50),
            "per_row_p95": percentile(fallback_counts, 0.95),
            "per_row_max": max(fallback_counts),
        },
        "source_tokens": {
            "min": min(source_lengths),
            "p50": percentile(source_lengths, 0.50),
            "p95": percentile(source_lengths, 0.95),
            "p99": percentile(source_lengths, 0.99),
            "max": max(source_lengths),
        },
        "positions": {
            "total": len(positions),
            "custom": len(custom),
            "native_fallback": len(positions) - len(custom),
            "custom_fraction": len(custom) / len(positions),
        },
        "embedding_exposure": {
            "contract_source_rows": len(source_ids),
            "unique_custom_rows_in_role": len(unique_custom),
            "unique_instruction_rows_in_role": len(unique_instruction_atoms),
            "unique_block_rows_in_role": len(unique_block_atoms),
            "unique_control_rows_in_role": len(unique_control_atoms),
            "custom_rows_unseen_in_fit": len(unique_custom - fit_custom),
            "contract_rows_unseen_in_fit": len(source_ids - fit_custom),
        },
        "instruction_vocabulary": {
            "positions": instruction_positions,
            "unique": len(instruction_frequency),
            "singleton_unique": sum(count == 1 for count in instruction_frequency.values()),
            "at_most_two_unique": sum(count <= 2 for count in instruction_frequency.values()),
        },
        "bindings": {
            "public_sha256": sha256(args.public),
            "codebook_sha256": sha256(codebook_path),
            "alignment_sha256": sha256(alignment_path),
            "model_inputs_sha256": sha256(model_path),
            "preflight_sha256": sha256(preflight_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
