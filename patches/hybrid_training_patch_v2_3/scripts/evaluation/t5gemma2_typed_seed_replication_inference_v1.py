#!/usr/bin/env python3
"""Versioned adapter for typed seed-replication inference.

Known typed-SFT and update58 checkpoints are pinned by the launcher.  A future
pass-3 checkpoint must additionally arrive with an externally SHA-pinned
manifest; this adapter validates that manifest before admitting a new training
schema to the otherwise frozen inference implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation import t5gemma2_f2_passk_inference as base
from scripts.evaluation import t5gemma2_measurement_audit_inference as inference
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)


ADAPTER_SCHEMA = "t5gemma2-typed-seed-replication-adapter-v1"
CHECKPOINT_MANIFEST_SCHEMA = (
    "t5gemma2-typed-seed-replication-pass3-checkpoint-manifest-v1"
)
UPDATE58_ADAPTER_SHA256 = (
    "62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec"
)
REQUIRED_CHECKPOINT_FILES = frozenset(
    {
        "run_contract.json",
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "tokenizer/tokenizer.json",
    }
)
HEX64 = re.compile(r"[0-9a-f]{64}")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _named_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for candidate_key, candidate_value in value.items():
            if candidate_key == key:
                found.append(candidate_value)
            found.extend(_named_values(candidate_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_named_values(item, key))
    return found


def _validate_file_record(
    record: Mapping[str, Any], *, expected_path: Path, label: str
) -> None:
    observed_path = Path(str(record.get("path") or "")).expanduser().resolve()
    expected_sha = str(record.get("sha256") or "")
    if (
        observed_path != expected_path.resolve()
        or HEX64.fullmatch(expected_sha) is None
        or not expected_path.is_file()
        or sha256_file(expected_path) != expected_sha
    ):
        raise ValueError(f"pass3 {label} file binding differs")


def validate_pass3_manifest(
    *, manifest_path: Path, expected_sha256: str, checkpoint: Path
) -> dict[str, Any]:
    if HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("pass3 manifest SHA is not a lowercase SHA256")
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("pass3 checkpoint manifest differs from its handoff SHA")
    manifest = _read_object(manifest_path, "pass3 checkpoint manifest")
    checkpoint = checkpoint.resolve()
    files = manifest.get("checkpoint_files")
    privacy = manifest.get("privacy")
    lineage = manifest.get("lineage")
    if (
        manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or manifest.get("arm") != "pass3"
        or Path(str(manifest.get("checkpoint") or "")).expanduser().resolve()
        != checkpoint
        or not isinstance(files, Mapping)
        or set(files) != REQUIRED_CHECKPOINT_FILES
        or not isinstance(privacy, Mapping)
        or privacy.get("heldout_175_model_visible") is not False
        or privacy.get("tests_model_visible") is not False
        or privacy.get("private_feedback_model_visible") is not False
        or privacy.get("gold_implementation_model_visible") is not False
        or privacy.get("semantic_parameter_names_model_visible") is not False
        or privacy.get("prior_success_exclusion_applied") is not True
        or privacy.get("known_contaminant_excluded")
        != "sigless_6b1dd0c6b6fc"
        or not isinstance(lineage, Mapping)
        or lineage.get("parent_arm") != "incumbent_update58"
        or lineage.get("parent_adapter_weights_sha256")
        != UPDATE58_ADAPTER_SHA256
        or manifest.get("no_automatic_promotion") is not True
    ):
        raise ValueError("pass3 checkpoint manifest policy contract differs")
    for relative_path, record in files.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"pass3 checkpoint file record is invalid: {relative_path}")
        _validate_file_record(
            record,
            expected_path=checkpoint / relative_path,
            label=relative_path,
        )
    for field in ("training_result", "training_audit"):
        record = manifest.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"pass3 manifest lacks {field}")
        _validate_file_record(
            record,
            expected_path=Path(str(record.get("path") or "")).expanduser().resolve(),
            label=field,
        )
    contract = _read_object(checkpoint / "run_contract.json", "pass3 run contract")
    contract_schema = str(contract.get("schema") or "")
    if (
        manifest.get("run_contract_canonical_sha256") != canonical_sha256(contract)
        or manifest.get("run_contract_schema") != contract_schema
        or not contract_schema
        or contract.get("architecture") != "native_encoder_decoder"
        or contract.get("status") != "training"
        or (contract.get("base_model") or {}).get("name")
        != "google/t5gemma-2-4b-4b"
        or str(
            (contract.get("base_model") or {}).get("resolved_commit")
            or (contract.get("base_model") or {}).get("requested_revision")
            or ""
        )
        != "487d4acf21a4d70c70bf534265b5263c9424979e"
    ):
        raise ValueError("pass3 run-contract/base-model binding differs")
    test_visibility = _named_values(contract, "tests_model_visible")
    heldout_overlap = _named_values(contract, "heldout_overlap")
    model_visible_fields = _named_values(contract, "model_visible_fields")
    if (
        not test_visibility
        or any(value is not False for value in test_visibility)
        or not heldout_overlap
        or any(value != 0 for value in heldout_overlap)
        or ["opaque_typed_contract", "F2.text"] not in model_visible_fields
    ):
        raise ValueError("pass3 run contract does not independently prove privacy")
    return manifest


def parse_adapter_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--replication-arm",
        choices=["typed_sft", "incumbent", "pass3"],
        required=True,
    )
    parser.add_argument("--checkpoint-manifest", default="")
    parser.add_argument("--expected-checkpoint-manifest-sha256", default="")
    known, remainder = parser.parse_known_args(argv)
    if known.replication_arm == "pass3":
        if not known.checkpoint_manifest or not known.expected_checkpoint_manifest_sha256:
            parser.error("pass3 requires a checkpoint manifest and handoff SHA")
    elif known.checkpoint_manifest or known.expected_checkpoint_manifest_sha256:
        parser.error("known arms do not accept a late-bound checkpoint manifest")
    return known, remainder


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    adapter_args, inference_argv = parse_adapter_args(argv)
    args = inference.parse_args(inference_argv)
    checkpoint = Path(args.sft_checkpoint).expanduser().resolve()
    manifest_record: dict[str, Any] | None = None
    if adapter_args.replication_arm == "pass3":
        manifest_path = Path(adapter_args.checkpoint_manifest).expanduser().resolve()
        manifest = validate_pass3_manifest(
            manifest_path=manifest_path,
            expected_sha256=adapter_args.expected_checkpoint_manifest_sha256,
            checkpoint=checkpoint,
        )
        stage_schema = str(manifest["run_contract_schema"])
        base.SUPPORTED_ADAPTER_RUN_SCHEMAS = frozenset(
            {*base.SUPPORTED_ADAPTER_RUN_SCHEMAS, stage_schema}
        )
        manifest_record = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "run_contract_schema": stage_schema,
        }
    result = inference.run(args)
    output_path = Path(args.output).expanduser().resolve()
    provenance = _read_object(
        Path(str(output_path) + ".provenance.json"), "inference provenance"
    )
    sidecar = {
        "schema": ADAPTER_SCHEMA,
        "replication_arm": adapter_args.replication_arm,
        "seed": args.seed,
        "adapter_script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint_manifest": manifest_record,
        "predictions_sha256": sha256_file(output_path),
        "provenance_sha256": sha256_file(
            Path(str(output_path) + ".provenance.json")
        ),
        "model_sha256": canonical_sha256(provenance.get("model") or {}),
        "input_view": "typed_opaque_contract",
        "tests_model_visible": False,
        "full_gold_implementation_model_visible": False,
        "automatic_promotion_performed": False,
    }
    require_exact_or_write(
        Path(str(output_path) + ".typed_seed_replication.json"), sidecar
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
