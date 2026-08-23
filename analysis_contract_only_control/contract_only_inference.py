#!/usr/bin/env python3
"""Sealed T5Gemma 2 inference for the opaque typed-contract-only control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from analysis_contract_only_control.contract_only_view import (
    EMPTY_BINARY_PAYLOAD,
    EMPTY_SHA256,
    OOD_CAVEAT,
    ORACLE_CAVEAT,
    VIEW,
    build_input_view,
)
from scripts.evaluation import t5gemma2_f2_passk_inference as base
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.training.t5gemma2_compiler_feedback_verpo import (
    _decoder_special_ids,
    _encode_source,
)


INFERENCE_SCHEMA = "t5gemma2-contract-only-inference-v1"
PROVENANCE_SCHEMA = "t5gemma2-f2-measurement-ablation-provenance-v1"
EXPECTED_CHECKPOINT_NAME = "checkpoint-optstep-000348"
EXPECTED_CHECKPOINT_FILE_SHA256 = {
    "run_contract.json": "562c3da5f89428e6a7263ad8ec79dde9c8b6eb25c77949606277d7d80aecea4f",
    "adapter/adapter_model.safetensors": "83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc",
    "adapter/adapter_config.json": "c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3",
    "tokenizer/tokenizer.json": "f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d",
}
EXPECTED_RUN_CONTRACT_CANONICAL_SHA256 = (
    "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3"
)


def _require_original_enriched_checkpoint(
    checkpoint: Path,
    checkpoint_contract: dict[str, Any],
    model_record: dict[str, Any],
) -> dict[str, Any]:
    """Reject every checkpoint except the frozen original enriched optstep348."""

    if checkpoint.name != EXPECTED_CHECKPOINT_NAME:
        raise ValueError("checkpoint path is not frozen optstep348")
    observed_files: dict[str, str] = {}
    for relative, expected in EXPECTED_CHECKPOINT_FILE_SHA256.items():
        path = checkpoint / relative
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"frozen checkpoint file differs: {relative}")
        observed_files[relative] = observed
    if (
        canonical_sha256(checkpoint_contract)
        != EXPECTED_RUN_CONTRACT_CANONICAL_SHA256
        or checkpoint_contract.get("schema") != "t5gemma2-enriched-sft-run-v1"
        or checkpoint_contract.get("architecture") != "native_encoder_decoder"
        or (checkpoint_contract.get("optimization") or {}).get("epochs") != 2
        or (checkpoint_contract.get("optimization") or {}).get("planned_updates")
        != 348
        or (checkpoint_contract.get("optimization") or {}).get("seed") != 42
        or model_record.get("training_stage_schema")
        != "t5gemma2-enriched-sft-run-v1"
        or model_record.get("warmstart_contract_sha256")
        != EXPECTED_RUN_CONTRACT_CANONICAL_SHA256
        or (model_record.get("adapter") or {}).get("adapter_weights_sha256")
        != EXPECTED_CHECKPOINT_FILE_SHA256["adapter/adapter_model.safetensors"]
    ):
        raise ValueError("checkpoint is not the frozen original enriched SFT run")
    return {
        "checkpoint_name": EXPECTED_CHECKPOINT_NAME,
        "run_contract_canonical_sha256": EXPECTED_RUN_CONTRACT_CANONICAL_SHA256,
        "files": observed_files,
    }


def load_contract_only_rows(
    args: argparse.Namespace,
) -> tuple[list[Any], dict[str, Any]]:
    """Load sealed rows, then replace all model-visible F2 with empty payloads."""

    baseline_rows, heldout = base.load_heldout_rows(
        dataset=args.dataset,
        dataset_seal=args.dataset_seal,
        f2_jsonl=args.f2_jsonl,
        f2_manifest=args.f2_manifest,
        limit=0,
    )
    dataset_rows = base._read_jsonl(  # noqa: SLF001
        Path(args.dataset).expanduser().resolve(), "held-out dataset"
    )
    f2_rows = base._read_jsonl(  # noqa: SLF001
        Path(args.f2_jsonl).expanduser().resolve(), "held-out F2"
    )
    sources, view_record = build_input_view(
        dataset_rows=dataset_rows,
        f2_rows=f2_rows,
    )
    if len(sources) != len(baseline_rows):
        raise ValueError("contract-only row count differs from held-out set")
    rows = [
        base.EvaluationRow(
            task_id=row.task_id,
            source=source,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )
        for row, source in zip(baseline_rows, sources, strict=True)
    ]
    if args.limit:
        if not 0 < args.limit <= len(rows):
            raise ValueError("limit lies outside the held-out row count")
        rows = rows[: args.limit]
    heldout = dict(heldout)
    heldout.update(
        {
            "selected_rows": len(rows),
            "selected_ordered_task_ids_sha256": canonical_sha256(
                [row.task_id for row in rows]
            ),
            "selected_ordered_source_sha256s_sha256": canonical_sha256(
                [row.source_sha256 for row in rows]
            ),
            "input_view": view_record,
            "model_visible_fields": [
                "gold_derived_opaque_types_and_arity",
                "task_invariant_empty_binary_payload",
            ],
            "binary_payload": {
                "text": EMPTY_BINARY_PAYLOAD,
                "utf8_hex": "",
                "utf8_bytes": 0,
                "sha256": EMPTY_SHA256,
                "task_invariant": True,
            },
            "f2_serialized_to_model": False,
            "recovered_constants_serialized_to_model": False,
            "f2_structure_serialized_to_model": False,
            "external_call_identities_serialized_to_model": False,
            "tests_serialized_to_model": False,
            "full_gold_targets_serialized_to_model": False,
            "gold_interface_types_and_arity_serialized_to_model": True,
            "gold_derived_oracle_control": True,
            "deployable_type_recovery_frontend_evaluated": False,
            "oracle_caveat": ORACLE_CAVEAT,
            "out_of_distribution_caveat": OOD_CAVEAT,
        }
    )
    return rows, heldout


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_view != VIEW:
        raise ValueError("contract-only inference received another input view")
    if not torch.cuda.is_available():
        raise RuntimeError("T5Gemma contract-only inference requires CUDA")
    rows, data_record = load_contract_only_rows(args)
    checkpoint = Path(args.sft_checkpoint).expanduser().resolve()
    checkpoint_contract, model_record = base._checkpoint_record(  # noqa: SLF001
        checkpoint, args.arm
    )
    checkpoint_identity = _require_original_enriched_checkpoint(
        checkpoint, checkpoint_contract, model_record
    )
    output_path = Path(args.output).expanduser().resolve()
    provenance_path = Path(str(output_path) + ".provenance.json")
    journal_path = Path(
        args.journal or (str(output_path) + ".generation.journal.jsonl")
    ).expanduser().resolve()
    contract = {
        "schema": INFERENCE_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_builder_script_sha256": sha256_file(
            Path(__file__).resolve().with_name("contract_only_view.py")
        ),
        "base_inference_script_sha256": sha256_file(Path(base.__file__).resolve()),
        "arm": args.arm,
        "input_view": VIEW,
        "model": model_record,
        "checkpoint_identity": checkpoint_identity,
        "heldout": data_record,
        "sampling": {
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": 0,
            "max_source_tokens": args.max_source_tokens,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "seed_policy": "seed+task_index*100003+batch_start",
            "generation_batch_size": args.generation_batch_size,
            "decoder_prefix_is_not_output": True,
            "sampled_eos_retained": True,
            "fabricated_eos": False,
        },
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or ""),
            "bf16": args.bf16,
            "attn_implementation": args.attn_implementation,
        },
        "oracle_control": {
            "gold_derived": True,
            "deployable_type_recovery_frontend_evaluated": False,
            "caveat": ORACLE_CAVEAT,
            "out_of_distribution_caveat": OOD_CAVEAT,
        },
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "gold_interface_types_and_arity_exposed_to_model": True,
        "f2_exposed_to_model": False,
        "source_truncation": False,
        "no_training_or_checkpoint_write": True,
    }
    events = load_journal(journal_path)
    if not events:
        if output_path.exists() or provenance_path.exists():
            raise ValueError("published output exists without its generation journal")
        append_event(
            journal_path,
            {
                "event": "header",
                "schema": base.JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
        events = load_journal(journal_path)
    terminals, complete = base._journal_state(  # noqa: SLF001
        events,
        contract=contract,
        rows=rows,
        num_samples=args.num_samples,
    )

    if not complete:
        model, tokenizer, loaded_record = base.load_policy(
            checkpoint=checkpoint,
            arm=args.arm,
            bf16=args.bf16,
            attn_implementation=args.attn_implementation,
        )
        if loaded_record != model_record:
            raise ValueError("loaded model record differs from preflight")
        decoder_start, pad_id, eos_ids = _decoder_special_ids(model, tokenizer)
        device = torch.device("cuda")
        for task_index in range(len(terminals), len(rows)):
            row = rows[task_index]
            input_ids, attention_mask = _encode_source(
                tokenizer,
                row.source,
                max_source_tokens=args.max_source_tokens,
                device=device,
            )
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            candidates: list[dict[str, Any]] = []
            for batch_start in range(
                0, args.num_samples, args.generation_batch_size
            ):
                count = min(
                    args.generation_batch_size,
                    args.num_samples - batch_start,
                )
                generated = base.generate_candidate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_start=decoder_start,
                    pad_id=pad_id,
                    eos_ids=eos_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=base.sample_seed(args.seed, task_index, batch_start),
                    count=count,
                )
                candidates.extend(
                    {
                        "sample_index": batch_start + batch_position,
                        **candidate,
                    }
                    for batch_position, candidate in enumerate(generated)
                )
            terminal = append_event(
                journal_path,
                {
                    "event": "task_terminal",
                    "schema": base.JOURNAL_SCHEMA,
                    "task_index": task_index,
                    "task_id": row.task_id,
                    "source_sha256": row.source_sha256,
                    "encoder_tokens": int(input_ids.size(1)),
                    "candidates": candidates,
                },
            )
            terminals.append(terminal)
            print(
                json.dumps(
                    {
                        "input_view": VIEW,
                        "task": task_index + 1,
                        "tasks": len(rows),
                        "task_id": row.task_id,
                        "encoder_tokens": int(input_ids.size(1)),
                        "max_token_completions": sum(
                            candidate["max_token_completion"]
                            for candidate in candidates
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        predictions = [
            {
                "id": terminal["task_id"],
                "predictions": [
                    candidate["text"] for candidate in terminal["candidates"]
                ],
            }
            for terminal in terminals
        ]
        append_event(
            journal_path,
            {
                "event": "complete",
                "schema": base.JOURNAL_SCHEMA,
                "rows": len(rows),
                "predictions_canonical_sha256": canonical_sha256(predictions),
            },
        )
        events = load_journal(journal_path)
        terminals, complete = base._journal_state(  # noqa: SLF001
            events,
            contract=contract,
            rows=rows,
            num_samples=args.num_samples,
        )
    if not complete:
        raise RuntimeError("generation journal did not reach completion")

    predictions = [
        {
            "id": terminal["task_id"],
            "predictions": [
                candidate["text"] for candidate in terminal["candidates"]
            ],
        }
        for terminal in terminals
    ]
    require_exact_or_write(output_path, predictions)
    capped = sum(
        candidate["max_token_completion"]
        for terminal in terminals
        for candidate in terminal["candidates"]
    )
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "architecture": "native_t5gemma2_encoder_decoder",
        "arm": args.arm,
        "input_view": VIEW,
        "output_sha256": sha256_file(output_path),
        "num_rows": len(predictions),
        "num_samples": args.num_samples,
        "model": model_record,
        "checkpoint_identity": checkpoint_identity,
        "heldout": data_record,
        "sampling": contract["sampling"],
        "max_token_completions": capped,
        "generation_journal": journal_record(journal_path),
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "gold_interface_types_and_arity_exposed_to_model": True,
        "f2_exposed_to_model": False,
        "recovered_constants_exposed_to_model": False,
        "f2_structure_exposed_to_model": False,
        "external_call_identities_exposed_to_model": False,
        "gold_derived_oracle_control": True,
        "deployable_type_recovery_frontend_evaluated": False,
        "oracle_caveat": ORACLE_CAVEAT,
        "out_of_distribution_caveat": OOD_CAVEAT,
        "sft_checkpoint_contract_sha256": canonical_sha256(checkpoint_contract),
    }
    require_exact_or_write(provenance_path, provenance)
    result = {
        "input_view": VIEW,
        "rows": len(predictions),
        "samples": len(predictions) * args.num_samples,
        "max_token_completions": capped,
        "output": str(output_path),
        "output_sha256": provenance["output_sha256"],
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_seal", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--f2_manifest", required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--arm", choices=["sft"], default="sft")
    parser.add_argument("--input_view", choices=[VIEW], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", default="")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--generation_batch_size", type=int, default=10)
    parser.add_argument("--max_source_tokens", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--attn_implementation", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(argv)
    if args.num_samples <= 0 or args.generation_batch_size <= 0:
        parser.error("sample counts must be positive")
    if args.generation_batch_size > args.num_samples:
        parser.error("generation batch cannot exceed sample count")
    if args.max_source_tokens <= 0 or args.max_new_tokens <= 0:
        parser.error("token limits must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("top_p must be in (0, 1]")
    if args.seed < 0 or args.limit < 0:
        parser.error("seed and limit must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
