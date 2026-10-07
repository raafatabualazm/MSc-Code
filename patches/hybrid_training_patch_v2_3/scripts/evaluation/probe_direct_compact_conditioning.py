#!/usr/bin/env python3
"""Measure correct versus permuted versus absent compact-source target NLL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.direct_compact_causal import (
    DirectCompactBatchCollator,
    DirectCompactCausalLM,
    DirectCompactContract,
    matched_permutation_indices,
    per_sequence_causal_nll,
    restore_source_embedding_overlay,
    resolve_decoder_config_path,
    sha256_artifact,
    sha256_file,
    validate_base_model_vocab,
)
from scripts.training.direct_compact_qwen_decompiler import CompactJsonlDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codec_artifact", required=True)
    parser.add_argument("--source_overlay", required=True)
    parser.add_argument("--decoder_model", default="")
    parser.add_argument("--decoder_revision", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer_revision", default="")
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--decoder_adapter", default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contract = DirectCompactContract.load(args.contract)
    decoder_model = args.decoder_model.strip() or contract.decoder_model
    decoder_revision = args.decoder_revision.strip() or contract.decoder_revision
    decoder_config_path = resolve_decoder_config_path(decoder_model, decoder_revision)
    contract.validate_decoder_binding(
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        model_config_path=decoder_config_path,
    )
    tokenizer_name = args.tokenizer or decoder_model
    tokenizer_revision = (
        args.tokenizer_revision.strip()
        or (decoder_revision if tokenizer_name == decoder_model else "")
        or None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    contract.validate_artifacts(
        tokenizer=tokenizer,
        tokenizer_json_path=args.tokenizer_json,
        codec_path=args.codec_artifact,
        codebook_path=args.codebook,
    )
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        decoder_model,
        revision=decoder_revision,
        **model_kwargs,
    )
    validate_base_model_vocab(model, contract)
    if args.decoder_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.decoder_adapter)
    model.to(args.device)
    restore_source_embedding_overlay(
        model,
        dict(contract.source_token_expansions),
        args.source_overlay,
        base_vocab_size=int(contract.base_vocab_size or 0),
    )
    model.eval()
    wrapped = DirectCompactCausalLM(model)
    dataset_path = Path(args.dataset).resolve()
    dataset = CompactJsonlDataset(
        dataset_path, tokenizer=tokenizer, contract=contract
    )
    rows = list(dataset.rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    if len(rows) < 2:
        raise ValueError("conditioning probe requires at least two rows")
    permutation = matched_permutation_indices(
        [len(row["compact_input_ids"]) for row in rows], seed=args.seed
    )
    arms = {
        "correct": rows,
        "permuted": [
            {**row, "compact_input_ids": rows[permutation[index]]["compact_input_ids"]}
            for index, row in enumerate(rows)
        ],
        # Keep source positions and the attention mask identical to the correct
        # arm. DirectCompactCausalLM zeros only those source embeddings.
        "null": rows,
    }
    collators = {
        name: DirectCompactBatchCollator(
            pad_token_id=tokenizer.pad_token_id,
            max_source_tokens=contract.max_source_tokens,
            max_target_tokens=contract.max_target_tokens,
            max_total_tokens=contract.max_total_tokens,
            source_token_ids=contract.source_token_ids,
        )
        for name in arms
    }
    nll: dict[str, list[float]] = {name: [] for name in arms}
    batch_size = max(1, args.batch_size)
    with torch.no_grad():
        for name, arm_rows in arms.items():
            for start in range(0, len(arm_rows), batch_size):
                batch = collators[name](arm_rows[start : start + batch_size])
                batch = {key: value.to(args.device) for key, value in batch.items()}
                outputs = wrapped(
                    **batch, zero_source_embeddings=name == "null"
                )
                values = per_sequence_causal_nll(outputs["logits"], batch["labels"])
                nll[name].extend(float(value) for value in values.detach().cpu())

    correct = torch.tensor(nll["correct"])
    permuted = torch.tensor(nll["permuted"])
    null = torch.tensor(nll["null"])
    report = {
        "schema": "direct-compact-conditioning-probe-v2",
        "rows": len(rows),
        "seed": args.seed,
        "dataset_sha256": sha256_file(dataset_path),
        "contract_sha256": sha256_file(args.contract),
        "decoder_model": decoder_model,
        "decoder_revision": decoder_revision,
        "model_config_sha256": sha256_file(decoder_config_path),
        "attn_implementation": args.attn_implementation,
        "source_overlay_sha256": sha256_file(args.source_overlay),
        "decoder_adapter_sha256": (
            sha256_artifact(args.decoder_adapter) if args.decoder_adapter else None
        ),
        "mean_nll": {name: sum(values) / len(values) for name, values in nll.items()},
        "permuted_minus_correct_nll": float((permuted - correct).mean()),
        "null_minus_correct_nll": float((null - correct).mean()),
        "correct_better_than_permuted_fraction": float((correct < permuted).float().mean()),
        "correct_better_than_null_fraction": float((correct < null).float().mean()),
        "permutation": permutation,
        "permutation_length_matching": {
            "method": "minimum-total-absolute-length-derangement",
            "correct_source_lengths": [
                len(row["compact_input_ids"]) for row in rows
            ],
            "donor_source_lengths": [
                len(rows[donor]["compact_input_ids"]) for donor in permutation
            ],
            "total_absolute_difference": sum(
                abs(
                    len(rows[index]["compact_input_ids"])
                    - len(rows[donor]["compact_input_ids"])
                )
                for index, donor in enumerate(permutation)
            ),
        },
        "null_ablation": "position-matched-zero-source-embeddings",
        "authoritative_behavioral_gate": False,
        "note": "NLL is diagnostic; free-running functional permutation remains authoritative.",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
