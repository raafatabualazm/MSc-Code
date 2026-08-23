#!/usr/bin/env python3
"""Load a sealed compiler-VeRPO checkpoint with the historical F2 evaluator.

This wrapper changes only the checkpoint-contract loader.  Sampling, decoding,
journaling, and provenance publication continue to run in
``t5gemma2_f2_passk_inference.py`` so a launcher can require the exact historical
inference hash used by the promoted two-epoch SFT baseline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation.durable_evaluation_journal import require_exact_or_write


VERPO_RUN_SCHEMA = "t5gemma2-compiler-feedback-verpo-run-v1"
VERPO_RUNTIME_SCHEMA = (
    "t5gemma2-compiler-feedback-verpo-runtime-provenance-v1"
)
SFT_RUN_SCHEMA = "t5gemma2-enriched-sft-run-v1"
COMPAT_SCHEMA = "t5gemma2-verpo-passk-loader-compat-v1"


def _pop_flag_value(arguments: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"VeRPO evaluation requires exactly one {flag}")
    index = positions[0]
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def _warmstart_contract(
    contract: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    warm = contract.get("warmstart")
    if not isinstance(warm, Mapping):
        raise ValueError("VeRPO checkpoint has no sealed warm-start")
    warm_path = Path(str(warm.get("path") or "")).expanduser().resolve()
    warm_contract = inference._read_json(
        warm_path / "run_contract.json", "VeRPO warm-start run contract"
    )
    warm_sha256 = inference.canonical_sha256(warm_contract)
    if (
        warm.get("stage_schema") != SFT_RUN_SCHEMA
        or warm.get("run_contract_sha256") != warm_sha256
        or warm.get("production_floor_eligible") is not True
        or warm_contract.get("schema") != SFT_RUN_SCHEMA
        or warm_contract.get("architecture") != "native_encoder_decoder"
        or warm_contract.get("status") != "training"
    ):
        raise ValueError("VeRPO warm-start contract binding failed")
    return warm_path, warm_contract


def _verpo_checkpoint_record(
    checkpoint: Path, arm: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = inference._read_json(
        checkpoint / "run_contract.json", "compiler-VeRPO run contract"
    )
    feedback = contract.get("feedback_boundary")
    runtime = contract.get("runtime_provenance")
    if (
        arm != "sft"
        or contract.get("schema") != VERPO_RUN_SCHEMA
        or contract.get("architecture") != "native_t5gemma2_encoder_decoder"
        or contract.get("objective")
        != "on_policy_visible_execution_verpo_plus_local_compiler_repair"
        or contract.get("no_frontier_api") is not True
        or contract.get("llm_judge") is not False
        or contract.get("acceptance_tests_exposed") is not False
        or contract.get("private_holdback_exposed") is not False
        or not isinstance(feedback, Mapping)
        or feedback.get("schema") != "verpo-train-feedback-view-v1"
        or feedback.get("acceptance_tests_exposed") is not False
        or feedback.get("reward_holdback_exposed") is not False
        or feedback.get("heldout_bytes_opened_during_validation") is not False
        or feedback.get("parent_or_private_bytes_opened_during_validation")
        is not False
        or not isinstance(runtime, Mapping)
        or runtime.get("schema") != VERPO_RUNTIME_SCHEMA
        or not isinstance(runtime.get("code"), Mapping)
        or runtime.get("code_bundle_sha256")
        != inference.canonical_sha256(runtime["code"])
    ):
        raise ValueError("compiler-VeRPO checkpoint contract/privacy binding failed")

    warm_path, warm_contract = _warmstart_contract(contract)
    base = warm_contract.get("base_model") or {}
    revision = str(base.get("resolved_commit") or base.get("requested_revision") or "")
    if (
        base.get("name") != inference.MODEL_NAME
        or revision != inference.MODEL_REVISION
        or base.get("is_encoder_decoder") is not True
    ):
        raise ValueError("VeRPO warm-start does not bind pinned T5Gemma 2")

    expected_targets = (warm_contract.get("lora") or {}).get("targets")
    if not isinstance(expected_targets, list) or not expected_targets:
        raise ValueError("VeRPO warm-start lacks exact LoRA targets")
    weighted_targets = inference._adapter_weight_target_modules(checkpoint)
    if weighted_targets != set(map(str, expected_targets)):
        raise ValueError("VeRPO adapter weights differ from warm-start targets")

    tokenizer_sha256 = inference.sha256_file(
        checkpoint / "tokenizer" / "tokenizer.json"
    )
    warm_tokenizer_sha256 = inference.sha256_file(
        warm_path / "tokenizer" / "tokenizer.json"
    )
    if tokenizer_sha256 != warm_tokenizer_sha256:
        raise ValueError("VeRPO checkpoint tokenizer differs from warm-start")

    contract_sha256 = inference.canonical_sha256(contract)
    record: dict[str, Any] = {
        "name": inference.MODEL_NAME,
        "revision": inference.MODEL_REVISION,
        "config_sha256": str(base.get("config_sha256") or ""),
        "arm": arm,
        "training_stage_schema": VERPO_RUN_SCHEMA,
        "production_floor_eligible": True,
        "tokenizer_sha256": tokenizer_sha256,
        "warmstart_contract_sha256": contract_sha256,
        "source_warmstart": {
            "path": str(warm_path),
            "run_contract_sha256": inference.canonical_sha256(warm_contract),
            "tokenizer_sha256": warm_tokenizer_sha256,
        },
        "training_runtime_code_bundle_sha256": runtime["code_bundle_sha256"],
        "adapter": {
            "adapter_config_sha256": inference.sha256_file(
                checkpoint / "adapter" / "adapter_config.json"
            ),
            "adapter_weights_sha256": inference.sha256_file(
                checkpoint / "adapter" / "adapter_model.safetensors"
            ),
            "run_contract_sha256": contract_sha256,
            "target_modules": len(expected_targets),
        },
    }
    return contract, record


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    compat_record = Path(
        _pop_flag_value(arguments, "--compat_record")
    ).expanduser().resolve()
    checkpoint = Path(
        _pop_flag_value(arguments, "--compat_checkpoint")
    ).expanduser().resolve()
    arguments.extend(["--sft_checkpoint", str(checkpoint)])

    contract, record = _verpo_checkpoint_record(checkpoint, "sft")
    warm = record["source_warmstart"]
    core_path = Path(inference.__file__).resolve()
    wrapper_path = Path(__file__).resolve()
    compatibility = {
        "schema": COMPAT_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_run_contract_file_sha256": inference.sha256_file(
            checkpoint / "run_contract.json"
        ),
        "checkpoint_run_contract_sha256": inference.canonical_sha256(contract),
        "checkpoint_adapter_config_sha256": record["adapter"][
            "adapter_config_sha256"
        ],
        "checkpoint_adapter_weights_sha256": record["adapter"][
            "adapter_weights_sha256"
        ],
        "checkpoint_tokenizer_sha256": record["tokenizer_sha256"],
        "source_warmstart": warm,
        "training_runtime_code_bundle_sha256": record[
            "training_runtime_code_bundle_sha256"
        ],
        "core_inference_path": str(core_path),
        "core_inference_sha256": inference.sha256_file(core_path),
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": hashlib.sha256(wrapper_path.read_bytes()).hexdigest(),
        "scope": "checkpoint_contract_loader_only",
        "sampling_code_changed": False,
        "generation_code_changed": False,
        "scoring_code_changed": False,
    }
    require_exact_or_write(compat_record, compatibility)
    inference._checkpoint_record = _verpo_checkpoint_record
    print(json.dumps(compatibility, sort_keys=True), flush=True)
    return inference.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
