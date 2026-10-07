#!/usr/bin/env python3
"""Phased production launcher for the asynchronous VeRPO rescue experiment.

The launcher intentionally cannot run the experiment as one opaque job. The
paid judge phase is separated from GPU inference by a required, sealed
operator acknowledgement that the GPU instance was released. GPU repair
generation is provisioned again only after diagnoses have been materialized.

No API key is accepted in the JSON config or command line. Provider
credentials remain process-environment secrets consumed by ``VerpoJudge``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PATCH_ROOT = Path(__file__).resolve().parents[2]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import direct_compact_verpo_rescue as rescue
from models.direct_compact_causal import (
    CONTRACT_SCHEMA_V3,
    DirectCompactContract,
    validate_join_seal,
)


CONFIG_SCHEMA = "direct-compact-verpo-rescue-launch-config-v1"
RUN_CONTRACT_SCHEMA = "direct-compact-verpo-rescue-launch-contract-v1"
GPU_RELEASE_SCHEMA = "direct-compact-verpo-rescue-gpu-release-v1"
RUN_CONTRACT_HASH_FIELD = "run_contract_sha256"
GPU_RELEASE_HASH_FIELD = "gpu_release_sha256"

REQUIRED_INPUTS = (
    "base_inference",
    "base_provenance",
    "rollout",
    "rollout_seal",
    "f2",
    "f2_manifest",
    "feedback_view_report",
    "private_holdback",
    "contract",
    "dataset",
    "alignment",
    "codebook",
    "codec_artifact",
    "tokenizer_json",
    "source_overlay",
)
PHASE_ORDER = (
    "preflight",
    "plan",
    "gpu_release",
    "diagnose",
    "materialize",
    "generate",
    "bundle",
    "score",
    "transfer",
)
PAID_PHASES = frozenset({"diagnose"})
GPU_PHASES = frozenset({"generate"})


class LaunchError(ValueError):
    """A launch contract or phase transition failed closed."""


def _read_json(path: str | Path, label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {label} {path}: {exc}") from exc


def _path_layout(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root).expanduser().resolve()
    return {
        "root": root,
        "contract": root / "00_preflight" / "run_contract.json",
        "plan": root / "01_plan" / "pilot_plan.json",
        "gpu_release": root / "02_gpu_release" / "released.json",
        "diagnoses": root / "03_diagnose" / "diagnoses.json",
        "diagnosis_journal": (
            root / "03_diagnose" / "diagnoses.journal.jsonl"
        ),
        "materialized": root / "04_materialized",
        "repairs": root / "05_repairs",
        "bundle": root / "06_bundle" / "repair_bundle.json",
        "score": root / "07_score" / "score_report.json",
        "rs_sft": root / "07_score" / "rs_sft_targets.jsonl",
        "preferences": root / "07_score" / "partial_preferences.jsonl",
        "transfer": root / "08_transfer",
        "transfer_report": root / "08_transfer" / "build_report.json",
    }


def _strict_object(
    value: Any,
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaunchError(f"{label} must be an object")
    allowed = required | (optional or set())
    missing = required - set(value)
    extra = set(value) - allowed
    if missing or extra:
        raise LaunchError(
            f"{label} keys differ; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return value


def _pinned_input(
    raw: Any,
    *,
    name: str,
) -> tuple[Path, dict[str, Any]]:
    value = _strict_object(
        raw,
        label=f"inputs.{name}",
        required={"path", "sha256"},
    )
    path = Path(str(value["path"])).expanduser().resolve()
    expected_sha = str(value["sha256"]).strip().lower()
    if not path.is_file():
        raise LaunchError(f"pinned input does not exist: {path}")
    observed = rescue.file_record(path)
    if observed["sha256"] != expected_sha:
        raise LaunchError(f"pinned input hash mismatch: {name}")
    return path, observed


def _row_identity(row: Mapping[str, Any], label: str) -> str:
    value = row.get("task_id") or row.get("id")
    if not isinstance(value, str) or not value:
        raise LaunchError(f"{label} has no task identity")
    return value


def _ordered_unique_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise LaunchError(f"{label} row {index + 1} is not an object")
        identity = _row_identity(row, f"{label} row {index + 1}")
        if identity in seen:
            raise LaunchError(f"{label} duplicates task {identity!r}")
        seen.add(identity)
        result.append(identity)
    return result


def _validate_fit_only_launch_view(
    paths: Mapping[str, Path],
    *,
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject measure/final-holdout routing before any plan or API call."""

    if inference.get("role") != "fit":
        raise LaunchError(
            "rescue inference role must be exactly 'fit'; measure/all can "
            "expose the final measure_175 stratum"
        )
    contract = DirectCompactContract.load(paths["contract"])
    if contract.schema == CONTRACT_SCHEMA_V3:
        raise LaunchError(
            "production rescue transfer does not support contract v3; "
            "rejecting before plan/API/GPU work"
        )
    try:
        rollout_seal = validate_join_seal(
            paths["rollout"],
            paths["rollout_seal"],
            paths["contract"],
            expected_role="fit",
        )
    except Exception as exc:
        raise LaunchError(f"sealed fit rollout validation failed: {exc}") from exc
    if (
        rollout_seal.get("training_allowed") is not True
        or rollout_seal.get("heldout_measure_only") is not False
        or rollout_seal.get("selected_role") != "fit"
    ):
        raise LaunchError(
            "rollout seal is not an explicitly training-allowed fit view"
        )

    public_rows = rescue.read_jsonl(paths["dataset"])
    alignment_rows = rescue.read_jsonl(paths["alignment"])
    if len(public_rows) != len(alignment_rows):
        raise LaunchError("compact public/alignment row counts differ")
    fit_ids: list[str] = []
    measure_ids: list[str] = []
    seen_alignment: set[str] = set()
    for index, row in enumerate(alignment_rows):
        if row.get("model_row") != index:
            raise LaunchError(
                f"alignment row {index + 1} has the wrong model_row"
            )
        identity = _row_identity(row, f"alignment row {index + 1}")
        if identity in seen_alignment:
            raise LaunchError(f"alignment duplicates task {identity!r}")
        seen_alignment.add(identity)
        role = str(row.get("role") or "").strip().lower()
        if role == "fit":
            fit_ids.append(identity)
        elif role == "measure":
            measure_ids.append(identity)
        else:
            raise LaunchError(
                f"alignment row {identity!r} has invalid role {role!r}"
            )
        for key in ("split", "partition", "dataset_role", "holdout_role"):
            marker = str(row.get(key) or "").strip().lower()
            if role == "fit" and marker in {
                "measure",
                "measure_175",
                "heldout",
                "holdout",
                "final_175",
            }:
                raise LaunchError(
                    f"alignment task {identity!r} is fit-labeled but carries "
                    f"the forbidden {key}={marker!r} marker"
                )

    rollout_rows = rescue.read_jsonl(paths["rollout"])
    f2_rows = rescue.read_jsonl(paths["f2"])
    holdback_rows = rescue.read_jsonl(paths["private_holdback"])
    rollout_ids = _ordered_unique_ids(rollout_rows, label="fit rollout")
    f2_ids = _ordered_unique_ids(f2_rows, label="fit F2")
    holdback_ids = _ordered_unique_ids(
        holdback_rows, label="development reward holdback"
    )
    fit_set = set(fit_ids)
    measure_set = set(measure_ids)
    if (
        rollout_ids != f2_ids
        or rollout_ids != holdback_ids
        or not set(rollout_ids) <= fit_set
        or set(rollout_ids) & measure_set
    ):
        raise LaunchError(
            "rollout/F2/holdback tasks are not the same sealed fit-only view"
        )
    base_inference = _read_json(
        paths["base_inference"], "base inference"
    )
    if not isinstance(base_inference, list) or not base_inference:
        raise LaunchError("base inference must be a nonempty task list")
    base_ids = _ordered_unique_ids(
        base_inference, label="base inference"
    )
    if not set(base_ids) <= set(rollout_ids):
        raise LaunchError(
            "base inference contains non-fit or final measure_175 tasks"
        )
    return {
        "contract_schema": contract.schema,
        "public_rows": len(public_rows),
        "alignment_fit_rows": len(fit_ids),
        "alignment_measure_rows_not_selected": len(measure_ids),
        "sealed_rollout_rows": len(rollout_ids),
        "base_inference_rows": len(base_ids),
        "selected_role": "fit",
        "final_measure_175_selected": False,
    }


def validate_launch_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate every file pin and return a normalized secret-free config."""

    source = Path(config_path).expanduser().resolve()
    raw = _read_json(source, "launch config")
    config = _strict_object(
        raw,
        label="launch config",
        required={
            "schema",
            "project_root",
            "output_root",
            "inputs",
            "plan",
            "judge",
            "inference",
            "score",
            "transfer",
        },
    )
    if config["schema"] != CONFIG_SCHEMA:
        raise LaunchError("launch config schema is invalid")
    if any("key" in str(key).lower() for key in config):
        raise LaunchError("API keys must not be placed in launch config")
    project_root = Path(str(config["project_root"])).expanduser().resolve()
    if not project_root.is_dir():
        raise LaunchError(f"project root does not exist: {project_root}")
    output_root = Path(str(config["output_root"])).expanduser().resolve()
    inputs = _strict_object(
        config["inputs"],
        label="inputs",
        required=set(REQUIRED_INPUTS),
    )
    normalized_inputs: dict[str, dict[str, Any]] = {}
    resolved_paths: dict[str, Path] = {}
    for name in REQUIRED_INPUTS:
        path, record = _pinned_input(inputs[name], name=name)
        resolved_paths[name] = path
        normalized_inputs[name] = record

    feedback_sha = normalized_inputs["feedback_view_report"]["sha256"]
    feedback = rescue.validate_feedback_view_report(
        resolved_paths["feedback_view_report"],
        feedback_sha,
        expected_outputs={
            "rollout": resolved_paths["rollout"],
            "seal": resolved_paths["rollout_seal"],
            "f2": resolved_paths["f2"],
            "f2_manifest": resolved_paths["f2_manifest"],
            "reward_holdback_private": resolved_paths["private_holdback"],
        },
    )
    feedback_invariants = feedback["report"].get("invariants")
    if (
        not isinstance(feedback_invariants, Mapping)
        or feedback_invariants.get("dev175_bytes_opened") is not False
        or feedback_invariants.get("acceptance_tests_read_or_used") is not False
        or feedback_invariants.get(
            "rollout_contains_no_acceptance_or_holdback_fields"
        )
        is not True
        or feedback_invariants.get("holdback_is_not_a_trainer_input")
        is not True
    ):
        raise LaunchError(
            "feedback-view report lacks fit/final-holdout privacy invariants"
        )
    provenance = _read_json(
        resolved_paths["base_provenance"], "base inference provenance"
    )
    if not isinstance(provenance, Mapping):
        raise LaunchError("base inference provenance is not an object")
    if (
        provenance.get("output_sha256")
        != normalized_inputs["base_inference"]["sha256"]
    ):
        raise LaunchError("base inference and provenance hashes disagree")

    plan = _strict_object(
        config["plan"],
        label="plan",
        required={
            "select_k",
            "repairs_per_candidate",
            "max_groups",
            "seed",
            "reward_timeout",
            "stability_runs",
            "workers",
            "mcnemar_minimum_difference",
            "mcnemar_assumed_discordance",
            "mcnemar_alpha",
            "mcnemar_power",
        },
    )
    judge = _strict_object(
        config["judge"],
        label="judge",
        required={
            "model",
            "base_url",
            "api_style",
            "max_tokens",
            "timeout_seconds",
            "max_retries",
            "thinking_mode",
            "reasoning_effort",
        },
    )
    if any(
        "key" in str(key).lower() or "token" == str(key).lower()
        for key in judge
    ):
        raise LaunchError("judge config must not contain credentials")
    inference = _strict_object(
        config["inference"],
        label="inference",
        required={
            "decoder_model",
            "decoder_revision",
            "tokenizer",
            "tokenizer_revision",
            "decoder_adapter",
            "attn_implementation",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "role",
            "direct_prompt_mode",
            "precision",
            "device",
        },
    )
    score = _strict_object(
        config["score"],
        label="score",
        required={"reward_timeout", "stability_runs", "workers"},
    )
    transfer = _strict_object(
        config["transfer"],
        label="transfer",
        required={"min_unique_repairs", "allow_low_coverage_smoke"},
    )
    if int(plan["select_k"]) <= 0 or int(plan["repairs_per_candidate"]) <= 0:
        raise LaunchError("plan K and R must be positive")
    if inference["precision"] not in {"bf16", "fp16", "fp32"}:
        raise LaunchError("inference precision must be bf16, fp16, or fp32")
    if int(inference["top_k"]) != 0:
        raise LaunchError("rescue generation requires top_k=0")
    if int(judge["max_retries"]) != 0:
        raise LaunchError(
            "judge max_retries must be zero for exactly one billed attempt "
            "per sealed group"
        )
    if inference["direct_prompt_mode"] != provenance.get(
        "direct_prompt_mode"
    ):
        raise LaunchError(
            "repair prompt mode differs from base student provenance"
        )
    for input_name, provenance_field in (
        ("contract", "contract_sha256"),
        ("codebook", "codebook_sha256"),
        ("codec_artifact", "codec_sha256"),
        ("tokenizer_json", "tokenizer_json_sha256"),
        ("source_overlay", "source_overlay_sha256"),
    ):
        if (
            normalized_inputs[input_name]["sha256"]
            != provenance.get(provenance_field)
        ):
            raise LaunchError(
                f"{input_name} differs from base student provenance"
            )
    fit_view = _validate_fit_only_launch_view(
        resolved_paths,
        inference=inference,
    )

    normalized = {
        "schema": CONFIG_SCHEMA,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "inputs": normalized_inputs,
        "plan": dict(plan),
        "judge": dict(judge),
        "inference": dict(inference),
        "score": dict(score),
        "transfer": dict(transfer),
        "fit_only_preflight": fit_view,
        "config_file": rescue.file_record(source),
        "secrets_persisted": False,
    }
    return normalized


def _run_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": RUN_CONTRACT_SCHEMA,
        "status": "complete",
        "config": dict(config),
        "phase_order": list(PHASE_ORDER),
        "phase_boundaries": {
            "judge_requires_gpu_release_ack": True,
            "judge_uses_gpu": False,
            "gpu_reprovisioned_only_for_generate": True,
            "score_opens_private_holdback": True,
            "provider_never_receives_private_holdback": True,
        },
    }
    return rescue.seal_artifact(body, RUN_CONTRACT_HASH_FIELD)


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = validate_launch_config(config_path)
    layout = _path_layout(config["output_root"])
    expected = _run_contract(config)
    destination = layout["contract"]
    if (
        not destination.exists()
        and layout["root"].exists()
        and any(layout["root"].iterdir())
    ):
        raise LaunchError(
            "output_root is nonempty but has no matching run contract"
        )
    if destination.exists():
        observed = _read_json(destination, "run contract")
        if observed != expected:
            raise LaunchError(
                "existing run contract differs from pinned launch config"
            )
    else:
        rescue.write_json_new(destination, expected)
    return expected


def load_run_contract(config_path: str | Path) -> dict[str, Any]:
    config = validate_launch_config(config_path)
    layout = _path_layout(config["output_root"])
    value = _read_json(layout["contract"], "run contract")
    if not isinstance(value, Mapping):
        raise LaunchError("run contract is not an object")
    rescue.require_artifact_digest(
        value,
        schema=RUN_CONTRACT_SCHEMA,
        hash_field=RUN_CONTRACT_HASH_FIELD,
    )
    if value != _run_contract(config):
        raise LaunchError("run contract/config binding changed")
    return dict(value)


def _module_command(module: str, arguments: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def _run(command: Sequence[str], *, project_root: str | Path) -> None:
    resolved_root = Path(project_root).resolve()
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(resolved_root), existing_pythonpath)
        if value
    )
    subprocess.run(
        list(command),
        cwd=resolved_root,
        check=True,
        env=environment,
    )


def _input_path(contract: Mapping[str, Any], name: str) -> str:
    return str(contract["config"]["inputs"][name]["path"])


def _plan_phase(contract: Mapping[str, Any]) -> None:
    config = contract["config"]
    layout = _path_layout(config["output_root"])
    if layout["plan"].exists():
        plan = _read_json(layout["plan"], "pilot plan")
        rescue.require_artifact_digest(
            plan,
            schema=rescue.PILOT_PLAN_SCHEMA,
            hash_field=rescue.PLAN_HASH_FIELD,
        )
        return
    settings = config["plan"]
    command = _module_command(
        "scripts.training.direct_compact_verpo_rescue",
        [
            "plan",
            "--base-inference",
            _input_path(contract, "base_inference"),
            "--base-provenance",
            _input_path(contract, "base_provenance"),
            "--rollout",
            _input_path(contract, "rollout"),
            "--f2",
            _input_path(contract, "f2"),
            "--f2-manifest",
            _input_path(contract, "f2_manifest"),
            "--feedback-view-report",
            _input_path(contract, "feedback_view_report"),
            "--expected-feedback-view-report-sha256",
            contract["config"]["inputs"]["feedback_view_report"]["sha256"],
            "--output",
            str(layout["plan"]),
            "--select-k",
            str(settings["select_k"]),
            "--repairs-per-candidate",
            str(settings["repairs_per_candidate"]),
            "--max-groups",
            str(settings["max_groups"]),
            "--seed",
            str(settings["seed"]),
            "--reward-timeout",
            str(settings["reward_timeout"]),
            "--stability-runs",
            str(settings["stability_runs"]),
            "--workers",
            str(settings["workers"]),
            "--mcnemar-minimum-difference",
            str(settings["mcnemar_minimum_difference"]),
            "--mcnemar-assumed-discordance",
            str(settings["mcnemar_assumed_discordance"]),
            "--mcnemar-alpha",
            str(settings["mcnemar_alpha"]),
            "--mcnemar-power",
            str(settings["mcnemar_power"]),
        ],
    )
    _run(command, project_root=config["project_root"])


def _gpu_release_phase(
    contract: Mapping[str, Any],
    confirmation: str,
) -> None:
    text = confirmation.strip()
    if len(text) < 8:
        raise LaunchError(
            "GPU release confirmation must identify the terminated instance"
        )
    layout = _path_layout(contract["config"]["output_root"])
    if not layout["plan"].is_file():
        raise LaunchError("plan must complete before releasing the GPU")
    body = {
        "schema": GPU_RELEASE_SCHEMA,
        "status": "operator_confirmed",
        "source_run_contract_sha256": contract[RUN_CONTRACT_HASH_FIELD],
        "source_plan_sha256": rescue.require_artifact_digest(
            _read_json(layout["plan"], "pilot plan"),
            schema=rescue.PILOT_PLAN_SCHEMA,
            hash_field=rescue.PLAN_HASH_FIELD,
        ),
        "confirmation": text,
        "launcher_did_not_claim_cloud_control": True,
        "operator_asserts_gpu_instance_released": True,
        "judge_phase_requires_no_gpu": True,
    }
    marker = rescue.seal_artifact(body, GPU_RELEASE_HASH_FIELD)
    if layout["gpu_release"].exists():
        if _read_json(layout["gpu_release"], "GPU release marker") != marker:
            raise LaunchError("GPU release marker already exists and differs")
    else:
        rescue.write_json_new(layout["gpu_release"], marker)


def _require_gpu_release(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    layout = _path_layout(contract["config"]["output_root"])
    marker = _read_json(layout["gpu_release"], "GPU release marker")
    if not isinstance(marker, Mapping):
        raise LaunchError("GPU release marker is not an object")
    rescue.require_artifact_digest(
        marker,
        schema=GPU_RELEASE_SCHEMA,
        hash_field=GPU_RELEASE_HASH_FIELD,
    )
    if (
        marker.get("source_run_contract_sha256")
        != contract[RUN_CONTRACT_HASH_FIELD]
        or marker.get("operator_asserts_gpu_instance_released") is not True
    ):
        raise LaunchError("GPU release marker belongs to another run")
    return marker


def _diagnose_phase(contract: Mapping[str, Any]) -> None:
    config = contract["config"]
    layout = _path_layout(config["output_root"])
    _require_gpu_release(contract)
    settings = config["judge"]
    command = _module_command(
        "scripts.training.direct_compact_verpo_rescue",
        [
            "diagnose",
            "--plan",
            str(layout["plan"]),
            "--output",
            str(layout["diagnoses"]),
            "--receipt-journal",
            str(layout["diagnosis_journal"]),
            "--model",
            str(settings["model"]),
            "--base-url",
            str(settings["base_url"]),
            "--api-style",
            str(settings["api_style"]),
            "--max-tokens",
            str(settings["max_tokens"]),
            "--timeout-seconds",
            str(settings["timeout_seconds"]),
            "--max-retries",
            str(settings["max_retries"]),
            "--thinking-mode",
            str(settings["thinking_mode"]),
            "--reasoning-effort",
            str(settings["reasoning_effort"]),
        ],
    )
    _run(command, project_root=config["project_root"])


def _materialize_phase(contract: Mapping[str, Any]) -> None:
    config = contract["config"]
    layout = _path_layout(config["output_root"])
    manifest = layout["materialized"] / "materialization.json"
    if manifest.exists():
        rescue._load_materialized_dir(layout["materialized"])
        return
    command = _module_command(
        "scripts.training.direct_compact_verpo_rescue",
        [
            "materialize",
            "--plan",
            str(layout["plan"]),
            "--diagnoses",
            str(layout["diagnoses"]),
            "--output-dir",
            str(layout["materialized"]),
        ],
    )
    _run(command, project_root=config["project_root"])


def generation_commands(
    contract: Mapping[str, Any],
) -> list[tuple[str, list[str]]]:
    """Return exact per-arm/rank GPU commands without executing them."""

    config = contract["config"]
    layout = _path_layout(config["output_root"])
    materialized = rescue._load_materialized_dir(layout["materialized"])
    settings = config["inference"]
    repairs = int(materialized["manifest"]["repairs_per_candidate"])
    commands: list[tuple[str, list[str]]] = []
    for record in materialized["manifest"]["plans"]:
        if int(record["generatable_rows"]) <= 0:
            continue
        key = str(record["key"])
        stem = key.replace(":", "__")
        output = layout["repairs"] / f"{stem}.json"
        journal = layout["repairs"] / f"{stem}.journal.jsonl"
        command = _module_command(
            "scripts.evaluation.direct_compact_qwen_inference",
            [
                "--dataset",
                _input_path(contract, "dataset"),
                "--alignment",
                _input_path(contract, "alignment"),
                "--output",
                str(output),
                "--rescue_conditioning_plan",
                str(record["artifact"]["path"]),
                "--journal",
                str(journal),
                "--contract",
                _input_path(contract, "contract"),
                "--codebook",
                _input_path(contract, "codebook"),
                "--codec_artifact",
                _input_path(contract, "codec_artifact"),
                "--decoder_model",
                str(settings["decoder_model"]),
                "--decoder_revision",
                str(settings["decoder_revision"]),
                "--tokenizer",
                str(settings["tokenizer"]),
                "--tokenizer_revision",
                str(settings["tokenizer_revision"]),
                "--tokenizer_json",
                _input_path(contract, "tokenizer_json"),
                "--attn_implementation",
                str(settings["attn_implementation"]),
                "--decoder_adapter",
                str(settings["decoder_adapter"]),
                "--source_overlay",
                _input_path(contract, "source_overlay"),
                "--batch_size",
                "1",
                "--max_new_tokens",
                str(settings["max_new_tokens"]),
                "--num_samples",
                str(repairs),
                "--temperature",
                str(settings["temperature"]),
                "--top_p",
                str(settings["top_p"]),
                "--top_k",
                str(settings["top_k"]),
                "--seed",
                str(settings["seed"]),
                "--limit",
                "0",
                "--role",
                str(settings["role"]),
                "--direct_prompt_mode",
                str(settings["direct_prompt_mode"]),
                "--device",
                str(settings["device"]),
            ],
        )
        if settings["precision"] == "bf16":
            command.append("--bf16")
        elif settings["precision"] == "fp16":
            command.append("--fp16")
        commands.append((key, command))
    return commands


def _generate_phase(contract: Mapping[str, Any]) -> None:
    config = contract["config"]
    layout = _path_layout(config["output_root"])
    layout["repairs"].mkdir(parents=True, exist_ok=True)
    plan = _read_json(layout["plan"], "pilot plan")
    materialized = rescue._load_materialized_dir(layout["materialized"])
    # Validate every already-present run before spending more GPU time.
    rescue.build_inference_bundle(
        plan,
        materialized,
        layout["repairs"],
        allow_missing=True,
    )
    for key, command in generation_commands(contract):
        stem = key.replace(":", "__")
        output = layout["repairs"] / f"{stem}.json"
        provenance = layout["repairs"] / f"{stem}.json.provenance.json"
        if output.exists() and provenance.exists():
            continue
        if output.exists() != provenance.exists():
            raise LaunchError(
                f"{key}: output/provenance are only partially present"
            )
        _run(command, project_root=config["project_root"])


def _bundle_phase(
    contract: Mapping[str, Any],
    *,
    allow_missing: bool,
) -> None:
    config = contract["config"]
    layout = _path_layout(config["output_root"])
    if layout["bundle"].exists():
        plan = _read_json(layout["plan"], "pilot plan")
        materialized = rescue._load_materialized_dir(
            layout["materialized"]
        )
        rescue._load_repair_bundle(
            layout["bundle"],
            expected_plan_sha256=str(plan[rescue.PLAN_HASH_FIELD]),
            expected_materialization_sha256=str(
                materialized["manifest"][
                    rescue.MATERIALIZATION_HASH_FIELD
                ]
            ),
            expected_plan_records=materialized["manifest"]["plans"],
        )
        return
    arguments = [
        "bundle",
        "--plan",
        str(layout["plan"]),
        "--materialized-dir",
        str(layout["materialized"]),
        "--repair-output-dir",
        str(layout["repairs"]),
        "--output",
        str(layout["bundle"]),
    ]
    if allow_missing:
        arguments.append("--allow-missing")
    _run(
        _module_command(
            "scripts.training.direct_compact_verpo_rescue",
            arguments,
        ),
        project_root=config["project_root"],
    )


def _score_phase(contract: Mapping[str, Any]) -> None:
    config = contract["config"]
    layout = _path_layout(config["output_root"])
    if layout["score"].exists():
        report = _read_json(layout["score"], "score report")
        rescue.require_artifact_digest(
            report,
            schema=rescue.SCORE_ARTIFACT_SCHEMA,
            hash_field=rescue.SCORE_HASH_FIELD,
        )
        exports = report.get("export_artifacts")
        if (
            not isinstance(exports, Mapping)
            or exports.get("rs_sft_targets")
            != rescue.file_record(layout["rs_sft"])
            or exports.get("preference_pairs")
            != rescue.file_record(layout["preferences"])
        ):
            raise LaunchError("existing score exports differ from report")
        return
    settings = config["score"]
    _run(
        _module_command(
            "scripts.training.direct_compact_verpo_rescue",
            [
                "score",
                "--plan",
                str(layout["plan"]),
                "--materialized-dir",
                str(layout["materialized"]),
                "--repair-bundle",
                str(layout["bundle"]),
                "--private-holdback",
                _input_path(contract, "private_holdback"),
                "--feedback-view-report",
                _input_path(contract, "feedback_view_report"),
                "--expected-feedback-view-report-sha256",
                config["inputs"]["feedback_view_report"]["sha256"],
                "--output",
                str(layout["score"]),
                "--rs-sft-output",
                str(layout["rs_sft"]),
                "--preferences-output",
                str(layout["preferences"]),
                "--reward-timeout",
                str(settings["reward_timeout"]),
                "--stability-runs",
                str(settings["stability_runs"]),
                "--workers",
                str(settings["workers"]),
            ],
        ),
        project_root=config["project_root"],
    )


def _transfer_phase(contract: Mapping[str, Any]) -> None:
    config = contract["config"]
    layout = _path_layout(config["output_root"])
    if layout["transfer_report"].exists():
        report = _read_json(layout["transfer_report"], "transfer report")
        if (
            not isinstance(report, Mapping)
            or report.get("schema")
            != "direct-compact-verpo-rescue-transfer-build-v1"
            or report.get("status") != "complete"
        ):
            raise LaunchError("existing transfer report is invalid")
        return
    settings = config["transfer"]
    arguments = [
        "--rollout",
        _input_path(contract, "rollout"),
        "--rollout-seal",
        _input_path(contract, "rollout_seal"),
        "--contract",
        _input_path(contract, "contract"),
        "--repairs",
        str(layout["rs_sft"]),
        "--score-report",
        str(layout["score"]),
        "--partial-preferences",
        str(layout["preferences"]),
        "--min-unique-repairs",
        str(settings["min_unique_repairs"]),
        "--output-dir",
        str(layout["transfer"]),
    ]
    if settings["allow_low_coverage_smoke"]:
        arguments.append("--allow-low-coverage-smoke")
    _run(
        _module_command(
            "scripts.training.build_verpo_rescue_transfer",
            arguments,
        ),
        project_root=config["project_root"],
    )


def phase_status(contract: Mapping[str, Any]) -> dict[str, Any]:
    layout = _path_layout(contract["config"]["output_root"])
    result: dict[str, Any] = {
        "schema": "direct-compact-verpo-rescue-launch-status-v1",
        "run_contract_sha256": contract[RUN_CONTRACT_HASH_FIELD],
        "phases": {},
    }
    checks = {"preflight": layout["contract"].is_file()}
    plan: Mapping[str, Any] | None = None
    if layout["plan"].is_file():
        raw_plan = _read_json(layout["plan"], "pilot plan")
        if not isinstance(raw_plan, Mapping):
            raise LaunchError("pilot plan is not an object")
        rescue.require_artifact_digest(
            raw_plan,
            schema=rescue.PILOT_PLAN_SCHEMA,
            hash_field=rescue.PLAN_HASH_FIELD,
        )
        plan = raw_plan
    checks["plan"] = plan is not None
    if layout["gpu_release"].is_file():
        _require_gpu_release(contract)
        checks["gpu_release"] = True
    else:
        checks["gpu_release"] = False
    if layout["diagnoses"].is_file():
        diagnoses = _read_json(layout["diagnoses"], "diagnoses")
        if not isinstance(diagnoses, Mapping):
            raise LaunchError("diagnosis artifact is not an object")
        rescue.require_artifact_digest(
            diagnoses,
            schema=rescue.DIAGNOSIS_ARTIFACT_SCHEMA,
            hash_field=rescue.DIAGNOSIS_HASH_FIELD,
        )
        if (
            plan is None
            or diagnoses.get("source_plan_sha256")
            != plan[rescue.PLAN_HASH_FIELD]
        ):
            raise LaunchError("diagnosis artifact belongs to another plan")
        checks["diagnose"] = True
    else:
        checks["diagnose"] = False
    materialized: Mapping[str, Any] | None = None
    if (layout["materialized"] / "materialization.json").is_file():
        materialized = rescue._load_materialized_dir(
            layout["materialized"]
        )
    checks["materialize"] = materialized is not None
    checks["bundle"] = layout["bundle"].is_file()
    if checks["bundle"] and plan is not None and materialized is not None:
        rescue._load_repair_bundle(
            layout["bundle"],
            expected_plan_sha256=str(plan[rescue.PLAN_HASH_FIELD]),
            expected_materialization_sha256=str(
                materialized["manifest"][
                    rescue.MATERIALIZATION_HASH_FIELD
                ]
            ),
            expected_plan_records=materialized["manifest"]["plans"],
        )
    checks["score"] = (
        layout["score"].is_file()
        and layout["rs_sft"].is_file()
        and layout["preferences"].is_file()
    )
    if checks["score"]:
        score = _read_json(layout["score"], "score report")
        if not isinstance(score, Mapping):
            raise LaunchError("score report is not an object")
        rescue.require_artifact_digest(
            score,
            schema=rescue.SCORE_ARTIFACT_SCHEMA,
            hash_field=rescue.SCORE_HASH_FIELD,
        )
        exports = score.get("export_artifacts")
        if (
            not isinstance(exports, Mapping)
            or exports.get("rs_sft_targets")
            != rescue.file_record(layout["rs_sft"])
            or exports.get("preference_pairs")
            != rescue.file_record(layout["preferences"])
        ):
            raise LaunchError("score export files differ from report")
    checks["transfer"] = layout["transfer_report"].is_file()
    if checks["transfer"]:
        transfer = _read_json(layout["transfer_report"], "transfer report")
        if (
            not isinstance(transfer, Mapping)
            or transfer.get("schema")
            != "direct-compact-verpo-rescue-transfer-build-v1"
            or transfer.get("status") != "complete"
        ):
            raise LaunchError("transfer report schema/status is invalid")
    generated = 0
    required = 0
    if materialized is not None:
        for record in materialized["manifest"]["plans"]:
            if int(record["generatable_rows"]) <= 0:
                continue
            required += 1
            stem = str(record["key"]).replace(":", "__")
            if (
                (layout["repairs"] / f"{stem}.json").is_file()
                and (
                    layout["repairs"]
                    / f"{stem}.json.provenance.json"
                ).is_file()
            ):
                generated += 1
    checks["generate"] = required > 0 and generated == required
    for phase in PHASE_ORDER:
        result["phases"][phase] = {
            "complete": bool(checks.get(phase, False))
        }
    result["phases"]["generate"].update(
        {"completed_runs": generated, "required_runs": required}
    )
    result["next_phase"] = next(
        (
            phase
            for phase in PHASE_ORDER
            if not result["phases"][phase]["complete"]
        ),
        None,
    )
    result["gpu_should_be_allocated_now"] = (
        result["next_phase"] in GPU_PHASES
    )
    result["paid_api_authorization_required"] = (
        result["next_phase"] in PAID_PHASES
    )
    config_path = str(contract["config"]["config_file"]["path"])
    if result["next_phase"] == "gpu_release":
        result["next_command"] = (
            f"{sys.executable} -m "
            "scripts.training.launch_direct_compact_verpo_rescue "
            f"--config {json.dumps(config_path)} gpu-release "
            '--confirmation "terminated instance REPLACE_ME"'
        )
    elif result["next_phase"] in PAID_PHASES:
        result["next_command"] = (
            f"{sys.executable} -m "
            "scripts.training.launch_direct_compact_verpo_rescue "
            f"--config {json.dumps(config_path)} resume --allow-paid-api"
        )
    elif result["next_phase"] in GPU_PHASES:
        result["next_command"] = (
            f"{sys.executable} -m "
            "scripts.training.launch_direct_compact_verpo_rescue "
            f"--config {json.dumps(config_path)} resume --allow-gpu"
        )
    elif result["next_phase"] is not None:
        result["next_command"] = (
            f"{sys.executable} -m "
            "scripts.training.launch_direct_compact_verpo_rescue "
            f"--config {json.dumps(config_path)} resume"
        )
    else:
        result["next_command"] = None
    return result


def _execute_phase(
    contract: Mapping[str, Any],
    phase: str,
    *,
    confirmation: str = "",
    allow_missing: bool = False,
) -> None:
    if phase == "plan":
        _plan_phase(contract)
    elif phase == "gpu_release":
        _gpu_release_phase(contract, confirmation)
    elif phase == "diagnose":
        _diagnose_phase(contract)
    elif phase == "materialize":
        _materialize_phase(contract)
    elif phase == "generate":
        _generate_phase(contract)
    elif phase == "bundle":
        _bundle_phase(contract, allow_missing=allow_missing)
    elif phase == "score":
        _score_phase(contract)
    elif phase == "transfer":
        _transfer_phase(contract)
    else:
        raise LaunchError(f"unsupported executable phase {phase!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phased direct-compact VeRPO rescue launcher"
    )
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("status")
    sub.add_parser("plan")
    release = sub.add_parser("gpu-release")
    release.add_argument("--confirmation", required=True)
    sub.add_parser("diagnose")
    sub.add_parser("materialize")
    sub.add_parser("generate")
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--allow-missing", action="store_true")
    sub.add_parser("score")
    sub.add_parser("transfer")
    resume = sub.add_parser("resume")
    resume.add_argument("--allow-paid-api", action="store_true")
    resume.add_argument("--allow-gpu", action="store_true")
    resume.add_argument("--allow-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        contract = preflight(args.config)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "run_contract_sha256": contract[
                        RUN_CONTRACT_HASH_FIELD
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    contract = load_run_contract(args.config)
    if args.command == "status":
        print(json.dumps(phase_status(contract), indent=2, sort_keys=True))
        return 0
    if args.command == "resume":
        status = phase_status(contract)
        phase = status["next_phase"]
        if phase is None:
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0
        if phase == "gpu_release":
            raise LaunchError(
                "terminate the GPU instance, then run gpu-release with an "
                "instance-identifying --confirmation"
            )
        if phase in PAID_PHASES and not args.allow_paid_api:
            raise LaunchError(
                "resume reached paid diagnosis; repeat with --allow-paid-api"
            )
        if phase in GPU_PHASES and not args.allow_gpu:
            raise LaunchError(
                "resume reached GPU generation; provision the checkpoint GPU "
                "and repeat with --allow-gpu"
            )
        _execute_phase(
            contract,
            phase,
            allow_missing=args.allow_missing,
        )
        return 0
    phase = args.command.replace("-", "_")
    _execute_phase(
        contract,
        phase,
        confirmation=getattr(args, "confirmation", ""),
        allow_missing=getattr(args, "allow_missing", False),
    )
    return 0


__all__ = [
    "CONFIG_SCHEMA",
    "GPU_RELEASE_SCHEMA",
    "LaunchError",
    "PHASE_ORDER",
    "RUN_CONTRACT_SCHEMA",
    "generation_commands",
    "load_run_contract",
    "main",
    "parse_args",
    "phase_status",
    "preflight",
    "validate_launch_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
