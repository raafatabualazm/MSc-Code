#!/usr/bin/env python3
"""Sealed typed-contract inference adapter for the final C2 VeRPO pilot.

Only a successful update-150 checkpoint from the discardable pilot is
admitted.  Generation is delegated unchanged to the existing typed
measurement runner.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.evaluation import t5gemma2_f2_passk_inference as base
from scripts.evaluation import t5gemma2_measurement_audit_inference as measurement
from scripts.evaluation import t5gemma2_typed_fold_gold_replay_inference_v1 as c2_guard
from scripts.evaluation.audit_t5gemma2_typed_c2_verpo_multiseed import SIDECAR_SCHEMA
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)


RUN_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-checkpoint-v1"
BASELINE_SCHEMA = "t5gemma2-typed-fold-gold-replay-run-v2"
EXPECTED_FINAL_UPDATE = 150
EXPECTED_CHECKPOINT = "checkpoint-optstep-000150"
EXPECTED_SEEDS = frozenset({42, 43, 44})


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _required_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is absent or empty: {resolved}")
    return resolved


def _require_profile(contract: Mapping[str, Any]) -> None:
    optimization = contract.get("optimization")
    warmstart = contract.get("warmstart")
    selection = contract.get("selection")
    input_view = contract.get("input_view")
    pilot = contract.get("pilot")
    base_model = contract.get("base_model")
    lora = contract.get("lora")
    checkpoint = contract.get("checkpoint")
    sampling = contract.get("sampling")
    reward = contract.get("reward")
    runtime = contract.get("runtime_provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (
            optimization,
            warmstart,
            selection,
            input_view,
            pilot,
            base_model,
            lora,
            checkpoint,
            sampling,
            reward,
            runtime,
        )
    ):
        raise ValueError("VeRPO run contract is structurally incomplete")
    runtime_code = runtime.get("code")
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "t5gemma2_typed_c2_verpo_pilot150.py"
    )
    profile_record = (
        runtime_code.get("typed_c2_pilot_profile")
        if isinstance(runtime_code, Mapping)
        else None
    )
    if (
        contract.get("schema") != RUN_SCHEMA
        or contract.get("status") != "training"
        or contract.get("architecture") != "native_encoder_decoder"
        or contract.get("policy_architecture") != "native_t5gemma2_encoder_decoder"
        or contract.get("objective")
        != "on_policy_visible_execution_verpo_plus_local_compiler_repair"
        or contract.get("automatic_promotion_permitted") is not False
        or contract.get("production_floor_eligible") is not False
        or contract.get("private_holdback_exposed") is not False
        or contract.get("no_frontier_api") is not True
        or contract.get("llm_judge") is not False
        or contract.get("acceptance_tests_exposed") is not False
        or input_view.get("view")
        != "opaque_typed_contract_plus_compressed_enriched_F2"
        or input_view.get("function_name") != "fn0"
        or input_view.get("parameter_name_policy") != "p{zero_based_index}"
        or input_view.get("semantic_names_visible") is not False
        or selection.get("tasks") != 150
        or selection.get("stored_candidates_actions_logprobs_rewards_reused")
        is not False
        or warmstart.get("stage_schema") != BASELINE_SCHEMA
        or warmstart.get("production_floor_eligible") is not True
        or optimization.get("max_updates") != EXPECTED_FINAL_UPDATE
        or optimization.get("tasks_per_update") != 1
        or not math.isclose(
            float(optimization.get("learning_rate", -1.0)),
            1e-6,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(optimization.get("sft_replay_weight", -1.0)),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or optimization.get("objective_profile") != "pure_execution_reward"
        or optimization.get("gold_or_sft_replay_gradient") is not False
        or not math.isclose(float(optimization.get("ppo_clip", -1.0)), 0.0)
        or pilot.get("disposition")
        != "discardable_mechanics_pilot_not_a_promotion_arm"
        or pilot.get("maximum_updates") != EXPECTED_FINAL_UPDATE
        or pilot.get("automatic_promotion_permitted") is not False
        or pilot.get("private_holdback_read") is not False
        or sampling.get("group_size") != 4
        or sampling.get("temperature") != 0.8
        or sampling.get("top_p") != 1.0
        or sampling.get("max_new_tokens") != 8192
        or sampling.get("max_source_tokens") != 32768
        or sampling.get("distribution_truncated") is not False
        or reward.get("visible_tests_only") is not True
        or reward.get("global_full_pass") is not True
        or reward.get("density_calibrated_partial_tests") is not True
        or base_model.get("name") != base.MODEL_NAME
        or (base_model.get("resolved_commit") or base_model.get("requested_revision"))
        != base.MODEL_REVISION
        or base_model.get("is_encoder_decoder") is not True
        or not isinstance(lora.get("targets"), list)
        or not lora.get("targets")
        or checkpoint.get("base_model_duplicated") is not False
        or not isinstance(profile_record, Mapping)
        or profile_record.get("sha256") != sha256_file(profile_path)
        or runtime.get("code_bundle_sha256") != canonical_sha256(runtime_code)
    ):
        raise ValueError("VeRPO pilot profile differs")


def _require_final_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    root = checkpoint.parent.resolve()
    if checkpoint.name != EXPECTED_CHECKPOINT or checkpoint.parent != root:
        raise ValueError("evaluation accepts only the final update-150 checkpoint")
    root_contract_path = _required_file(root / "run_contract.json", "root run contract")
    checkpoint_contract_path = _required_file(
        checkpoint / "run_contract.json", "checkpoint run contract"
    )
    result_path = _required_file(root / "result.json", "pilot result")
    latest_path = _required_file(root / "latest_checkpoint.json", "latest checkpoint record")
    state_path = _required_file(checkpoint / "training_state.pt", "training state")
    adapter_weights = _required_file(
        checkpoint / "adapter" / "adapter_model.safetensors", "adapter weights"
    )
    adapter_config = _required_file(
        checkpoint / "adapter" / "adapter_config.json", "adapter config"
    )
    tokenizer = _required_file(
        checkpoint / "tokenizer" / "tokenizer.json", "tokenizer"
    )
    root_contract = _read_object(root_contract_path, "root run contract")
    checkpoint_contract = _read_object(checkpoint_contract_path, "checkpoint run contract")
    result = _read_object(result_path, "pilot result")
    latest = _read_object(latest_path, "latest checkpoint record")
    _require_profile(root_contract)
    if canonical_sha256(checkpoint_contract) != canonical_sha256(root_contract):
        raise ValueError("root/checkpoint run contracts differ")
    contract_sha = canonical_sha256(root_contract)
    # The optimizer state is large and is needed only for its sealed metadata.
    # Memory-map it so evaluation preflight does not materialize another copy.
    state = torch.load(
        state_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(state, Mapping):
        raise ValueError("training state is not a mapping")
    latest_target = Path(str(latest.get("path") or "")).expanduser().resolve()
    if (
        result.get("schema") != RUN_SCHEMA
        or result.get("status") != "complete"
        or result.get("updates") != EXPECTED_FINAL_UPDATE
        or result.get("latest_checkpoint") != EXPECTED_CHECKPOINT
        or result.get("mechanics_gate") != "GO"
        or result.get("automatic_promotion_performed") is not False
        or result.get("production_floor_eligible") is not False
        or result.get("pilot_disposition")
        != "discardable_not_for_automatic_promotion"
        or result.get("run_contract_sha256") != contract_sha
        or latest.get("schema") != CHECKPOINT_SCHEMA
        or latest.get("update") != EXPECTED_FINAL_UPDATE
        or latest.get("run_contract_sha256") != contract_sha
        or latest_target != checkpoint
        or state.get("schema") != CHECKPOINT_SCHEMA
        or state.get("update") != EXPECTED_FINAL_UPDATE
        or state.get("run_contract_sha256") != contract_sha
    ):
        raise ValueError("VeRPO final result/checkpoint seal differs")

    warmstart = root_contract["warmstart"]
    warm_path = Path(str(warmstart.get("path") or "")).expanduser().resolve()
    if warm_path.name != "checkpoint-optstep-000058":
        raise ValueError("VeRPO warmstart is not the final C2 checkpoint")
    warm_contract_path = _required_file(warm_path / "run_contract.json", "C2 run contract")
    warm_contract = _read_object(warm_contract_path, "C2 run contract")
    c2_guard._require_arm_c_contract(warm_contract)  # noqa: SLF001
    warm_files = warmstart.get("checkpoint_files")
    if not isinstance(warm_files, Mapping) or (
        warm_files.get("run_contract_sha256") != sha256_file(warm_contract_path)
        or warm_files.get("adapter_weights_sha256")
        != sha256_file(_required_file(warm_path / "adapter" / "adapter_model.safetensors", "C2 adapter weights"))
        or warm_files.get("adapter_config_sha256")
        != sha256_file(_required_file(warm_path / "adapter" / "adapter_config.json", "C2 adapter config"))
        or warm_files.get("tokenizer_sha256")
        != sha256_file(_required_file(warm_path / "tokenizer" / "tokenizer.json", "C2 tokenizer"))
    ):
        raise ValueError("VeRPO-to-C2 warmstart file binding differs")
    manifest = {
        "run_contract": sha256_file(root_contract_path),
        "result": sha256_file(result_path),
        "latest_checkpoint": sha256_file(latest_path),
        "checkpoint_run_contract": sha256_file(checkpoint_contract_path),
        "training_state": sha256_file(state_path),
        "adapter_weights": sha256_file(adapter_weights),
        "adapter_config": sha256_file(adapter_config),
        "tokenizer": sha256_file(tokenizer),
    }
    return {
        "contract": root_contract,
        "contract_sha256": contract_sha,
        "manifest": manifest,
        "manifest_sha256": canonical_sha256(manifest),
    }


def _require_exact_eval_profile(args: Any) -> None:
    expected = {
        "arm": "sft",
        "input_view": "typed_opaque_contract",
        "num_samples": 10,
        "generation_batch_size": 10,
        "max_source_tokens": 32768,
        "max_new_tokens": 8192,
        "temperature": 0.8,
        "top_p": 0.95,
        "limit": 0,
        "attn_implementation": "sdpa",
        "bf16": True,
    }
    differences = {
        key: (getattr(args, key), value)
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if differences or args.seed not in EXPECTED_SEEDS:
        raise ValueError(
            f"VeRPO matched-evaluation profile differs: {differences}, seed={args.seed}"
        )


def run(args: Any) -> dict[str, Any]:
    _require_exact_eval_profile(args)
    checkpoint = Path(args.sft_checkpoint).expanduser().resolve()
    checkpoint_record = _require_final_checkpoint(checkpoint)
    original_supported = base.SUPPORTED_ADAPTER_RUN_SCHEMAS
    original_checkpoint_record = base._checkpoint_record  # noqa: SLF001

    def guarded_checkpoint_record(path: Path, arm: str):
        observed = _require_final_checkpoint(path)
        if observed["manifest_sha256"] != checkpoint_record["manifest_sha256"]:
            raise ValueError("VeRPO checkpoint changed between validation and load")
        return original_checkpoint_record(path, arm)

    base.SUPPORTED_ADAPTER_RUN_SCHEMAS = frozenset(
        set(original_supported) | {RUN_SCHEMA}
    )
    base._checkpoint_record = guarded_checkpoint_record  # type: ignore[assignment]  # noqa: SLF001
    try:
        result = measurement.run(args)
    finally:
        base._checkpoint_record = original_checkpoint_record  # type: ignore[assignment]  # noqa: SLF001
        base.SUPPORTED_ADAPTER_RUN_SCHEMAS = original_supported

    predictions = Path(args.output).expanduser().resolve()
    provenance = Path(str(predictions) + ".provenance.json")
    post = _require_final_checkpoint(checkpoint)
    if post["manifest_sha256"] != checkpoint_record["manifest_sha256"]:
        raise ValueError("VeRPO checkpoint changed during inference")
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "status": "complete",
        "seed": args.seed,
        "training_stage_schema": RUN_SCHEMA,
        "final_update": EXPECTED_FINAL_UPDATE,
        "checkpoint_manifest": checkpoint_record["manifest"],
        "checkpoint_manifest_sha256": checkpoint_record["manifest_sha256"],
        "checkpoint_contract_sha256": checkpoint_record["contract_sha256"],
        "predictions_sha256": sha256_file(predictions),
        "provenance_sha256": sha256_file(provenance),
        "model_sha256": canonical_sha256(
            _read_object(provenance, "evaluation provenance")["model"]
        ),
        "adapter_script_sha256": sha256_file(Path(__file__).resolve()),
        "automatic_promotion_performed": False,
    }
    require_exact_or_write(
        Path(str(predictions) + ".typed_c2_verpo_eval.json"), sidecar
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(measurement.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
