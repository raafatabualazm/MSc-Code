#!/usr/bin/env python3
"""Fail-closed seal for the preregistered typed RS-SFT pass-3 checkpoint.

The seed-replication runner intentionally accepts a new pass-3 checkpoint only
through an externally SHA-pinned manifest.  This program constructs that
handoff after independently checking the completed training stage, its exact
update-58 lineage, the direct-only/no-replay dataset contract, and all files
needed to reload the adapter.  It performs no inference and no promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)


AUDIT_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass3-training-audit-v1"
MANIFEST_SCHEMA = "t5gemma2-typed-seed-replication-pass3-checkpoint-manifest-v1"
RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass3-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass3-checkpoint-v1"
DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass3-dataset-v1"
MODEL = "google/t5gemma-2-4b-4b"
MODEL_REVISION = "487d4acf21a4d70c70bf534265b5263c9424979e"
UPDATE58_RUN_CONTRACT_SHA256 = (
    "0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f"
)
UPDATE58_ADAPTER_SHA256 = (
    "62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec"
)
UPDATE58_ADAPTER_CONFIG_SHA256 = (
    "b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b"
)
KNOWN_CONTAMINANT = "sigless_6b1dd0c6b6fc"
EXPECTED_VISIBLE_FIELDS = ["opaque_typed_contract", "F2.text"]
REQUIRED_CHECKPOINT_FILES = (
    "run_contract.json",
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "tokenizer/tokenizer.json",
)
AUDITED_CHECKPOINT_FILES = (*REQUIRED_CHECKPOINT_FILES, "training_state.pt")
AUDITED_STAGE_FILES = (
    "result.json",
    "run_contract.json",
    "dataset_manifest.json",
    "latest_checkpoint.json",
)
CHECKPOINT_RE = re.compile(r"checkpoint-optstep-([0-9]{6})")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _resolve_real_file(root: Path, relative: str, label: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} is missing or linked: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes its sealed root: {candidate}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {candidate}")
    return resolved


def _adapter_weight_target_modules(checkpoint: Path) -> set[str]:
    """Recover authoritative LoRA module names from the saved A/B weights.

    PEFT minimizes ``adapter_config.json.target_modules`` into a suffix-matching
    program when it saves an adapter.  Those strings are therefore not the
    explicit module-name set recorded by the training contract.  The weight
    keys retain that exact set and are the correct artifact to audit.
    """

    from safetensors import safe_open

    weights = checkpoint / "adapter" / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = [str(key) for key in handle.keys()]
    pattern = re.compile(
        r"^(?:base_model\.model\.)?"
        r"(?P<module>.+)\.lora_(?P<branch>A|B)"
        r"(?:\.[^.]+)?\.weight$"
    )
    branches: dict[str, set[str]] = {}
    unexpected: list[str] = []
    for key in keys:
        match = pattern.fullmatch(key)
        if match is None:
            unexpected.append(key)
            continue
        branches.setdefault(match.group("module"), set()).add(match.group("branch"))
    incomplete = sorted(
        module for module, observed in branches.items() if observed != {"A", "B"}
    )
    if unexpected or not branches or incomplete:
        raise ValueError(
            "pass-3 adapter weights do not contain a complete canonical LoRA target set: "
            f"unexpected={unexpected[:10]} incomplete={incomplete[:10]}"
        )
    return set(branches)


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        stat_before = path.stat()
        digest = sha256_file(path)
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            or stat_before.st_ino != stat_after.st_ino
        ):
            raise ValueError(f"file changed while it was sealed: {path}")
        result[str(path.resolve())] = {
            "sha256": digest,
            "size_bytes": stat_after.st_size,
        }
    # A second complete digest pass prevents a cross-file snapshot from
    # silently mixing two training states.
    for path in paths:
        if sha256_file(path) != result[str(path.resolve())]["sha256"]:
            raise ValueError(f"file set changed while it was sealed: {path}")
    return result


def _require_pass3_dataset(dataset: Mapping[str, Any], *, rows: int) -> None:
    composition = dataset.get("composition")
    schedule = dataset.get("schedule")
    typed_train = dataset.get("typed_train")
    prior = dataset.get("prior_225_exclusion")
    if not isinstance(composition, Mapping):
        raise ValueError("pass-3 dataset lacks composition")
    c002_rows = int(composition.get("kimi_c002_tail", -1))
    if (
        dataset.get("schema") != DATASET_SCHEMA
        or dataset.get("architecture") != "native_encoder_decoder"
        or int(dataset.get("rows", -1)) != rows
        or rows < 13
        or rows > 60
        or int(composition.get("verified_direct", -1)) != rows
        or int(composition.get("kimi_c001", -1)) != 12
        or int(composition.get("kimi_c002_prefix", -1)) != 1
        or c002_rows != rows - 13
        or not 0 <= c002_rows <= 47
        or any(
            int(composition.get(name, -1)) != 0
            for name in (
                "prior_225_replay",
                "pass2_209_replay",
                "gold_replay",
                "repair_conditioned",
                "reasoning_rows",
            )
        )
        or dataset.get("heldout_overlap") != 0
        or dataset.get("heldout_175_model_visible") is not False
        or dataset.get("tests_model_visible") is not False
        or dataset.get("private_feedback_model_visible") is not False
        or dataset.get("repair_conditioned_prefixes_visible") is not False
        or dataset.get("reasoning_model_visible") is not False
        or dataset.get("known_contaminant_excluded") != KNOWN_CONTAMINANT
        or dataset.get("model_visible_fields") != EXPECTED_VISIBLE_FIELDS
        or dataset.get("task_id_deduplication")
        != "reject_any_cross_source_or_prior_overlap"
        or dataset.get("all_targets_bound_to_provider_or_zero_api_verification_journals")
        is not True
        or dataset.get("production_floor_eligible") is not True
        or not isinstance(schedule, list)
        or len(schedule) != rows
        or not isinstance(typed_train, Mapping)
        or not isinstance(prior, Mapping)
    ):
        raise ValueError("pass-3 direct-only/no-replay dataset contract differs")
    categories = [str(row.get("source_category") or "") for row in schedule]
    task_ids = [str(row.get("source_task_id") or "") for row in schedule]
    pair_ids = [str(row.get("pair_id") or "") for row in schedule]
    if (
        categories.count("kimi_c001") != 12
        or categories.count("kimi_c002_tail") != c002_rows
        or categories.count("kimi_c002_prefix") != 1
        or not all(task_ids)
        or len(task_ids) != len(set(task_ids))
        or not all(pair_ids)
        or len(pair_ids) != len(set(pair_ids))
        or KNOWN_CONTAMINANT in task_ids
        or dataset.get("schedule_sha256") != canonical_sha256(schedule)
        or dataset.get("task_ids_sha256") != canonical_sha256(task_ids)
    ):
        raise ValueError("pass-3 schedule identity/composition differs")
    opaque = typed_train.get("opaque_contract")
    exclusions = typed_train.get("training_exclusions")
    if (
        typed_train.get("model_visible_fields") != EXPECTED_VISIBLE_FIELDS
        or not isinstance(opaque, Mapping)
        or opaque.get("parameter_name_policy") != "p{zero_based_index}"
        or opaque.get("semantic_function_name_exposed") is not False
        or opaque.get("semantic_parameter_names_exposed") is not False
        or not isinstance(exclusions, Mapping)
        or KNOWN_CONTAMINANT not in (exclusions.get("task_ids") or [])
    ):
        raise ValueError("pass-3 opaque typed-source privacy contract differs")


def _require_run_contract(
    contract: Mapping[str, Any], *, expected_trainer_sha256: str
) -> tuple[int, int]:
    if HEX64.fullmatch(expected_trainer_sha256) is None:
        raise ValueError("expected trainer SHA is malformed")
    base_model = contract.get("base_model")
    runtime = contract.get("runtime")
    warmstart = contract.get("warmstart")
    optimization = contract.get("optimization")
    privacy = contract.get("privacy")
    dataset = contract.get("dataset")
    if not all(
        isinstance(value, Mapping)
        for value in (base_model, runtime, warmstart, optimization, privacy, dataset)
    ):
        raise ValueError("pass-3 run contract is structurally incomplete")
    rows = int(dataset.get("rows", -1))
    if (
        contract.get("schema") != RUN_SCHEMA
        or contract.get("status") != "training"
        or contract.get("architecture") != "native_encoder_decoder"
        or contract.get("model") != MODEL
        or contract.get("model_revision") != MODEL_REVISION
        or base_model.get("name") != MODEL
        or str(base_model.get("resolved_commit") or base_model.get("requested_revision") or "")
        != MODEL_REVISION
        or runtime.get("trainer_sha256") != expected_trainer_sha256
        or int(warmstart.get("update", -1)) != 58
        or warmstart.get("run_contract_sha256") != UPDATE58_RUN_CONTRACT_SHA256
        or warmstart.get("adapter_weights_sha256") != UPDATE58_ADAPTER_SHA256
        or warmstart.get("adapter_config_sha256") != UPDATE58_ADAPTER_CONFIG_SHA256
        or int(optimization.get("epochs", -1)) != 2
        or int(optimization.get("batch_size", -1)) != 1
        or int(optimization.get("gradient_accumulation", -1)) != 8
        or not math.isclose(
            float(optimization.get("learning_rate", -1.0)),
            2e-5,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or float(optimization.get("warmup_ratio", -1.0)) != 0.0
        or int(optimization.get("warmup_updates", -1)) != 0
        or int(optimization.get("seed", -1)) != 42
        or privacy.get("heldout_overlap") != 0
        or privacy.get("heldout_content_model_visible") is not False
        or privacy.get("tests_model_visible") is not False
        or privacy.get("private_feedback_model_visible") is not False
        or privacy.get("reasoning_persisted") is not False
        or contract.get("production_floor_eligible") is not True
    ):
        raise ValueError("pass-3 run/lineage/privacy contract differs")
    _require_pass3_dataset(dataset, rows=rows)
    expected_updates_per_epoch = math.ceil(rows / 8)
    expected_planned_updates = 2 * expected_updates_per_epoch
    if (
        int(optimization.get("updates_per_epoch", -1)) != expected_updates_per_epoch
        or int(optimization.get("planned_updates", -1)) != expected_planned_updates
    ):
        raise ValueError("pass-3 optimizer schedule differs")
    return rows, expected_planned_updates


def seal(args: argparse.Namespace) -> dict[str, Any]:
    stage_candidate = Path(args.stage_dir).expanduser()
    if stage_candidate.is_symlink() or not stage_candidate.is_dir():
        raise ValueError("pass-3 stage root is missing or linked")
    stage = stage_candidate.resolve(strict=True)
    stage_files = {
        relative: _resolve_real_file(stage, relative, f"stage {relative}")
        for relative in AUDITED_STAGE_FILES
    }
    contract = _read_object(stage_files["run_contract.json"], "stage run contract")
    rows, planned_updates = _require_run_contract(
        contract, expected_trainer_sha256=args.expected_trainer_sha256
    )
    dataset = _read_object(stage_files["dataset_manifest.json"], "dataset manifest")
    result = _read_object(stage_files["result.json"], "training result")
    latest = _read_object(stage_files["latest_checkpoint.json"], "checkpoint pointer")
    if canonical_sha256(dataset) != canonical_sha256(contract["dataset"]):
        raise ValueError("stage dataset manifest differs from the run contract")
    checkpoint_name = str(result.get("latest_checkpoint") or "")
    match = CHECKPOINT_RE.fullmatch(checkpoint_name)
    if (
        result.get("schema") != RUN_SCHEMA
        or result.get("status") != "complete"
        or int(result.get("rows", -1)) != rows
        or int(result.get("updates", -1)) != planned_updates
        or int(result.get("planned_updates", -1)) != planned_updates
        or result.get("production_floor_eligible") is not True
        or match is None
        or int(match.group(1)) != planned_updates
    ):
        raise ValueError("pass-3 completion result differs")
    checkpoint_candidate = stage / checkpoint_name
    if checkpoint_candidate.is_symlink() or not checkpoint_candidate.is_dir():
        raise ValueError("pass-3 checkpoint escapes the stage root or is linked")
    checkpoint = checkpoint_candidate.resolve(strict=True)
    if checkpoint.parent != stage:
        raise ValueError("pass-3 checkpoint escapes the stage root or is linked")
    checkpoint_files = {
        relative: _resolve_real_file(
            checkpoint, relative, f"checkpoint {relative}"
        )
        for relative in AUDITED_CHECKPOINT_FILES
    }
    contract_sha = canonical_sha256(contract)
    checkpoint_contract = _read_object(
        checkpoint_files["run_contract.json"], "checkpoint run contract"
    )
    if canonical_sha256(checkpoint_contract) != contract_sha:
        raise ValueError("checkpoint and stage run contracts differ")
    if (
        latest.get("schema") != CHECKPOINT_SCHEMA
        or Path(str(latest.get("path") or "")).expanduser().resolve() != checkpoint
        or int(latest.get("update", -1)) != planned_updates
        or latest.get("run_contract_sha256") != contract_sha
    ):
        raise ValueError("latest-checkpoint pointer differs")
    adapter = _read_object(
        checkpoint_files["adapter/adapter_config.json"], "adapter config"
    )
    lora_contract = contract.get("lora", {})
    configured_targets = adapter.get("target_modules")
    expected_configured_targets_sha256 = str(
        args.expected_adapter_target_modules_sha256
    )
    if HEX64.fullmatch(expected_configured_targets_sha256) is None:
        raise ValueError("expected adapter target-module set SHA is malformed")
    if (
        adapter.get("task_type") != "SEQ_2_SEQ_LM"
        or adapter.get("peft_type") != "LORA"
        or adapter.get("base_model_name_or_path") != MODEL
        or adapter.get("bias") != "none"
        or adapter.get("use_dora") is not False
        or adapter.get("use_rslora") is not False
        or adapter.get("modules_to_save") is not None
        or int(adapter.get("r", -1)) != int(lora_contract.get("rank", -2))
        or int(adapter.get("lora_alpha", -1))
        != int(lora_contract.get("alpha", -2))
        or float(adapter.get("lora_dropout", -1.0))
        != float(lora_contract.get("dropout", -2.0))
        or not isinstance(configured_targets, list)
        or not configured_targets
        or any(not isinstance(value, str) or not value for value in configured_targets)
        or len(configured_targets) != len(set(configured_targets))
        or canonical_sha256(sorted(configured_targets))
        != expected_configured_targets_sha256
    ):
        raise ValueError("pass-3 adapter configuration differs from run contract")
    contract_targets = set(map(str, lora_contract.get("targets") or []))
    weighted_targets = _adapter_weight_target_modules(checkpoint)
    if not contract_targets or weighted_targets != contract_targets:
        raise ValueError("pass-3 adapter weight targets differ from run contract")

    all_paths = [*stage_files.values(), *checkpoint_files.values()]
    stable_snapshot = _snapshot(all_paths)
    stage_records = {key: _file_record(value) for key, value in stage_files.items()}
    checkpoint_records = {
        key: _file_record(value) for key, value in checkpoint_files.items()
    }
    if any(
        stable_snapshot[record["path"]]["sha256"] != record["sha256"]
        for record in [*stage_records.values(), *checkpoint_records.values()]
    ):
        raise ValueError("pass-3 files changed while records were constructed")

    audit = {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "stage": str(stage),
        "checkpoint": str(checkpoint),
        "stage_files": stage_records,
        "checkpoint_files": checkpoint_records,
        "run_contract_schema": RUN_SCHEMA,
        "run_contract_canonical_sha256": contract_sha,
        "rows": rows,
        "optimizer_updates": planned_updates,
        "adapter_weight_targets": {
            "count": len(weighted_targets),
            "sorted_names_sha256": canonical_sha256(sorted(weighted_targets)),
            "matches_run_contract": True,
        },
        "lineage": {
            "parent_arm": "incumbent_update58",
            "parent_update": 58,
            "parent_run_contract_sha256": UPDATE58_RUN_CONTRACT_SHA256,
            "parent_adapter_weights_sha256": UPDATE58_ADAPTER_SHA256,
            "parent_adapter_config_sha256": UPDATE58_ADAPTER_CONFIG_SHA256,
        },
        "training_profile": {
            "epochs": 2,
            "learning_rate": 2e-5,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "warmup_updates": 0,
            "seed": 42,
            "gold_replay_rows": 0,
            "prior_success_replay_rows": 0,
            "pass2_replay_rows": 0,
            "only_new_direct_verified_targets": True,
        },
        "privacy": {
            "heldout_175_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "gold_implementation_model_visible": False,
            "semantic_parameter_names_model_visible": False,
            "prior_success_exclusion_applied": True,
            "known_contaminant_excluded": KNOWN_CONTAMINANT,
        },
        "automatic_promotion_performed": False,
        "verpo_launched": False,
    }
    audit_path = Path(args.output_audit).expanduser().resolve()
    require_exact_or_write(audit_path, audit)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "arm": "pass3",
        "checkpoint": str(checkpoint),
        "checkpoint_files": {
            key: checkpoint_records[key] for key in REQUIRED_CHECKPOINT_FILES
        },
        "training_result": stage_records["result.json"],
        "training_audit": _file_record(audit_path),
        "run_contract_schema": RUN_SCHEMA,
        "run_contract_canonical_sha256": contract_sha,
        "lineage": {
            "parent_arm": "incumbent_update58",
            "parent_adapter_weights_sha256": UPDATE58_ADAPTER_SHA256,
        },
        "privacy": dict(audit["privacy"]),
        "no_automatic_promotion": True,
    }
    manifest_path = Path(args.output_manifest).expanduser().resolve()
    require_exact_or_write(manifest_path, manifest)
    seal_result = {
        "schema": "t5gemma2-typed-pass3-checkpoint-seal-result-v1",
        "status": "complete",
        "training_audit": _file_record(audit_path),
        "checkpoint_manifest": _file_record(manifest_path),
        "checkpoint": str(checkpoint),
        "promotion_status": "HOLD_REQUIRES_3PLUS_SEEDS",
        "verpo_status": "HOLD",
    }
    require_exact_or_write(Path(args.output_result).expanduser().resolve(), seal_result)
    return seal_result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--expected-trainer-sha256", required=True)
    parser.add_argument("--expected-adapter-target-modules-sha256", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-result", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(seal(parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
