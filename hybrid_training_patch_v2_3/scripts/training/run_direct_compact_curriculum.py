#!/usr/bin/env python3
"""Minimal fail-closed runner for the encoder-free compact-source SFT stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from models.direct_compact_causal import (
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--train_seal", required=True)
    parser.add_argument("--dev_seal", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codec_artifact", required=True)
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
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolved_file(path: str, label: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = resolved_file(args.train_file, "training JSONL")
    dev = resolved_file(args.dev_file, "development JSONL")
    train_seal = resolved_file(args.train_seal, "training join seal")
    dev_seal = resolved_file(args.dev_seal, "development join seal")
    contract_path = resolved_file(args.contract, "compact contract")
    codebook = resolved_file(args.codebook, "compact codebook")
    codec = resolved_file(args.codec_artifact, "compact codec")
    tokenizer_json = resolved_file(args.tokenizer_json, "base tokenizer.json")
    contract = DirectCompactContract.load(contract_path)
    validate_join_seal(train, train_seal, contract_path, expected_role="fit")
    validate_join_seal(dev, dev_seal, contract_path, expected_role="measure")
    decoder_model = args.decoder_model.strip() or contract.decoder_model
    decoder_revision = args.decoder_revision.strip() or contract.decoder_revision
    contract.validate_decoder_binding(
        decoder_model=decoder_model, decoder_revision=decoder_revision
    )
    if sha256_file(codebook) != contract.codebook_sha256:
        raise ValueError("codebook file does not match the compact contract")
    if sha256_file(codec) != contract.codec_sha256:
        raise ValueError("codec artifact does not match the compact contract")
    if sha256_file(tokenizer_json) != contract.tokenizer_json_sha256:
        raise ValueError("tokenizer.json does not match the compact contract")

    stage = output / "01_direct_compact_sft"
    command = [
        args.python,
        "-m",
        "scripts.training.direct_compact_qwen_decompiler",
        "--train_file",
        str(train),
        "--eval_file",
        str(dev),
        "--train_seal",
        str(train_seal),
        "--eval_seal",
        str(dev_seal),
        "--output_dir",
        str(stage),
        "--contract",
        str(contract_path),
        "--codebook",
        str(codebook),
        "--codec_artifact",
        str(codec),
        "--tokenizer_json",
        str(tokenizer_json),
        "--attn_implementation",
        args.attn_implementation,
        "--decoder_model",
        decoder_model,
        "--decoder_revision",
        decoder_revision,
        "--learning_rate",
        str(args.learning_rate),
        "--epochs",
        str(args.epochs),
        "--max_steps",
        str(args.max_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--eval_strategy",
        args.eval_strategy,
        "--batch_size",
        str(args.batch_size),
        "--grad_accum",
        str(args.grad_accum),
        "--seed",
        str(args.seed),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
    ]
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.tokenizer_revision:
        command.extend(["--tokenizer_revision", args.tokenizer_revision])
    for enabled, flag in (
        (args.load_4bit, "--load_4bit"),
        (args.gradient_checkpointing, "--gradient_checkpointing"),
        (args.bf16, "--bf16"),
        (args.fp16, "--fp16"),
    ):
        if enabled:
            command.append(flag)

    manifest = {
        "schema": "direct-compact-curriculum-v1",
        "architecture": "decoder-only-compact-source",
        "train_file": str(train),
        "train_sha256": sha256_file(train),
        "train_seal_sha256": sha256_file(train_seal),
        "dev_file": str(dev),
        "dev_sha256": sha256_file(dev),
        "dev_seal_sha256": sha256_file(dev_seal),
        "contract": contract.as_dict(),
        "contract_file_sha256": sha256_file(contract_path),
        "command": command,
        "dry_run": bool(args.dry_run),
    }
    (output / "direct_compact_curriculum_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=root, check=True)
    provenance = stage / "run_provenance.json"
    if not provenance.is_file():
        raise RuntimeError(f"direct compact stage completed without {provenance}")


if __name__ == "__main__":
    main()
