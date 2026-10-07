#!/usr/bin/env python3
"""Load a mixed RS-SFT checkpoint with the sealed legacy inference engine."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation.durable_evaluation_journal import require_exact_or_write

MIXED_RUN_SCHEMA = "t5gemma2-mixed-rs-sft-run-v1"
MIXED_DATASET_SCHEMA = "t5gemma2-mixed-rs-sft-dataset-v1"
COMPAT_SCHEMA = "t5gemma2-mixed-passk-loader-compat-v1"


def _pop_flag_value(arguments: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"mixed evaluation compatibility requires exactly one {flag}")
    index = positions[0]
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def _mixed_checkpoint_record(
    checkpoint: Path, arm: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = inference._read_json(
        checkpoint / "run_contract.json", "mixed RS-SFT run contract"
    )
    privacy = contract.get("privacy")
    dataset = contract.get("dataset")
    if (
        contract.get("schema") != MIXED_RUN_SCHEMA
        or contract.get("architecture") != "native_encoder_decoder"
        or contract.get("status") != "training"
        or not isinstance(privacy, Mapping)
        or privacy.get("heldout_overlap") != 0
        or privacy.get("heldout_content_model_visible") is not False
        or privacy.get("tests_model_visible") is not False
        or privacy.get("private_feedback_model_visible") is not False
        or not isinstance(dataset, Mapping)
        or dataset.get("schema") != MIXED_DATASET_SCHEMA
        or dataset.get("heldout_overlap") != 0
    ):
        raise ValueError("mixed RS-SFT checkpoint contract/privacy binding failed")
    base = contract.get("base_model") or {}
    revision = str(base.get("resolved_commit") or base.get("requested_revision") or "")
    if (
        base.get("name") != inference.MODEL_NAME
        or revision != inference.MODEL_REVISION
        or base.get("is_encoder_decoder") is not True
    ):
        raise ValueError("mixed checkpoint does not bind the pinned T5Gemma 2 base")
    record: dict[str, Any] = {
        "name": inference.MODEL_NAME,
        "revision": inference.MODEL_REVISION,
        "config_sha256": str(base.get("config_sha256") or ""),
        "arm": arm,
        "training_stage_schema": MIXED_RUN_SCHEMA,
        "production_floor_eligible": (
            contract.get("production_floor_eligible") is True
        ),
        "tokenizer_sha256": inference.sha256_file(
            checkpoint / "tokenizer" / "tokenizer.json"
        ),
        "warmstart_contract_sha256": inference.canonical_sha256(contract),
    }
    if arm != "sft":
        raise ValueError("mixed RS-SFT checkpoint requires the sft inference arm")
    expected_targets = contract.get("lora", {}).get("targets")
    if not isinstance(expected_targets, list) or not expected_targets:
        raise ValueError("mixed checkpoint lacks exact LoRA targets")
    weighted_targets = inference._adapter_weight_target_modules(checkpoint)
    if weighted_targets != set(map(str, expected_targets)):
        raise ValueError("mixed adapter weights differ from its exact target set")
    record["adapter"] = {
        "adapter_config_sha256": inference.sha256_file(
            checkpoint / "adapter" / "adapter_config.json"
        ),
        "adapter_weights_sha256": inference.sha256_file(
            checkpoint / "adapter" / "adapter_model.safetensors"
        ),
        "run_contract_sha256": inference.canonical_sha256(contract),
        "target_modules": len(expected_targets),
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
    contract, _record = _mixed_checkpoint_record(checkpoint, "sft")
    core_path = Path(inference.__file__).resolve()
    wrapper_path = Path(__file__).resolve()
    compatibility = {
        "schema": COMPAT_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_run_contract_sha256": inference.canonical_sha256(contract),
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
    inference._checkpoint_record = _mixed_checkpoint_record
    print(json.dumps(compatibility, sort_keys=True), flush=True)
    return inference.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
