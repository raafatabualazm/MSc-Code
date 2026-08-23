from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from scripts.evaluation import audit_t5gemma2_typed_c2_verpo_multiseed as audit
from scripts.evaluation import t5gemma2_typed_c2_verpo_inference_v1 as inference
from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _contract(warm_path: Path) -> dict:
    for relative in (
        "run_contract.json",
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "tokenizer/tokenizer.json",
    ):
        _write(warm_path / relative, b"{}" if relative.endswith(".json") else b"x")
    runtime_code = {
        "typed_c2_pilot_profile": {
            "relative_path": "scripts/training/t5gemma2_typed_c2_verpo_pilot150.py",
            "sha256": sha256_file(
                Path(inference.__file__).resolve().parents[1]
                / "training/t5gemma2_typed_c2_verpo_pilot150.py"
            ),
        }
    }
    return {
        "schema": inference.RUN_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "policy_architecture": "native_t5gemma2_encoder_decoder",
        "objective": "on_policy_visible_execution_verpo_plus_local_compiler_repair",
        "automatic_promotion_permitted": False,
        "production_floor_eligible": False,
        "private_holdback_exposed": False,
        "no_frontier_api": True,
        "llm_judge": False,
        "acceptance_tests_exposed": False,
        "runtime_provenance": {
            "code": runtime_code,
            "code_bundle_sha256": canonical_sha256(runtime_code),
        },
        "input_view": {
            "view": "opaque_typed_contract_plus_compressed_enriched_F2",
            "function_name": "fn0",
            "parameter_name_policy": "p{zero_based_index}",
            "semantic_names_visible": False,
        },
        "selection": {
            "tasks": 150,
            "stored_candidates_actions_logprobs_rewards_reused": False,
        },
        "warmstart": {
            "path": str(warm_path.resolve()),
            "stage_schema": inference.BASELINE_SCHEMA,
            "production_floor_eligible": True,
            "checkpoint_files": {
                "run_contract_sha256": sha256_file(warm_path / "run_contract.json"),
                "adapter_weights_sha256": sha256_file(
                    warm_path / "adapter/adapter_model.safetensors"
                ),
                "adapter_config_sha256": sha256_file(
                    warm_path / "adapter/adapter_config.json"
                ),
                "tokenizer_sha256": sha256_file(warm_path / "tokenizer/tokenizer.json"),
            },
        },
        "optimization": {
            "max_updates": 150,
            "tasks_per_update": 1,
            "learning_rate": 1e-6,
            "sft_replay_weight": 0.0,
            "objective_profile": "pure_execution_reward",
            "gold_or_sft_replay_gradient": False,
            "ppo_clip": 0.0,
        },
        "pilot": {
            "disposition": "discardable_mechanics_pilot_not_a_promotion_arm",
            "maximum_updates": 150,
            "automatic_promotion_permitted": False,
            "private_holdback_read": False,
        },
        "sampling": {
            "group_size": 4,
            "temperature": 0.8,
            "top_p": 1.0,
            "max_new_tokens": 8192,
            "max_source_tokens": 32768,
            "distribution_truncated": False,
        },
        "reward": {
            "visible_tests_only": True,
            "global_full_pass": True,
            "density_calibrated_partial_tests": True,
        },
        "base_model": {
            "name": inference.base.MODEL_NAME,
            "resolved_commit": inference.base.MODEL_REVISION,
            "is_encoder_decoder": True,
        },
        "lora": {"targets": ["encoder.block.0.layer.0.SelfAttention.q"]},
        "checkpoint": {"base_model_duplicated": False},
    }


def _sealed_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    warm = tmp_path / "c2" / "checkpoint-optstep-000058"
    contract = _contract(warm)
    root = tmp_path / "pilot"
    checkpoint = root / inference.EXPECTED_CHECKPOINT
    checkpoint.mkdir(parents=True)
    (root / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (checkpoint / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    for relative in (
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "tokenizer/tokenizer.json",
    ):
        _write(checkpoint / relative, b"{}" if relative.endswith(".json") else b"x")
    contract_sha = canonical_sha256(contract)
    torch.save(
        {
            "schema": inference.CHECKPOINT_SCHEMA,
            "update": 150,
            "run_contract_sha256": contract_sha,
        },
        checkpoint / "training_state.pt",
    )
    (root / "latest_checkpoint.json").write_text(
        json.dumps(
            {
                "schema": inference.CHECKPOINT_SCHEMA,
                "update": 150,
                "path": str(checkpoint.resolve()),
                "run_contract_sha256": contract_sha,
            }
        ),
        encoding="utf-8",
    )
    (root / "result.json").write_text(
        json.dumps(
            {
                "schema": inference.RUN_SCHEMA,
                "status": "complete",
                "updates": 150,
                "latest_checkpoint": inference.EXPECTED_CHECKPOINT,
                "mechanics_gate": "GO",
                "automatic_promotion_performed": False,
                "production_floor_eligible": False,
                "pilot_disposition": "discardable_not_for_automatic_promotion",
                "run_contract_sha256": contract_sha,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(inference.c2_guard, "_require_arm_c_contract", lambda _: None)
    return checkpoint


def test_final_checkpoint_accepts_only_complete_update150(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _sealed_checkpoint(tmp_path, monkeypatch)
    record = inference._require_final_checkpoint(checkpoint)
    assert record["contract"]["schema"] == inference.RUN_SCHEMA
    assert len(record["manifest_sha256"]) == 64

    result_path = checkpoint.parent / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "stopped_at_window_gate"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="final result/checkpoint seal differs"):
        inference._require_final_checkpoint(checkpoint)


def test_profile_rejects_replay_and_private_holdback(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "c2" / "checkpoint-optstep-000058")
    inference._require_profile(contract)
    contract["optimization"]["sft_replay_weight"] = 0.02
    with pytest.raises(ValueError, match="profile differs"):
        inference._require_profile(contract)
    contract["optimization"]["sft_replay_weight"] = 0.0
    contract["private_holdback_exposed"] = True
    with pytest.raises(ValueError, match="profile differs"):
        inference._require_profile(contract)


def test_eval_profile_is_three_seed_8192_only() -> None:
    args = argparse.Namespace(
        arm="sft",
        input_view="typed_opaque_contract",
        num_samples=10,
        generation_batch_size=10,
        max_source_tokens=32768,
        max_new_tokens=8192,
        temperature=0.8,
        top_p=0.95,
        seed=44,
        limit=0,
        attn_implementation="sdpa",
        bf16=True,
    )
    inference._require_exact_eval_profile(args)
    args.max_new_tokens = 4096
    with pytest.raises(ValueError, match="profile differs"):
        inference._require_exact_eval_profile(args)


def _fake_arm(delta_task: str | None = None) -> dict[int, dict]:
    tasks = ("a", "b")
    result = {}
    for seed in audit.EXPECTED_SEEDS:
        values = {
            task_id: {
                "pass_at_1": 0.0,
                "pass_at_10": 0.0,
                "compile_at_10": 1.0,
                "candidate_pass_rate": 0.0,
                "candidate_compile_rate": 0.8,
                "distinct_per_10": 10.0,
            }
            for task_id in tasks
        }
        if delta_task is not None:
            values[delta_task]["pass_at_10"] = 1.0
            values[delta_task]["candidate_pass_rate"] = 0.1
        result[seed] = {"task_values": values}
    return result


def test_cluster_bootstrap_keeps_all_seeds_inside_task_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "BOOTSTRAP_REPLICATES", 200)
    report = audit._cluster_bootstrap(
        {"c2_baseline": _fake_arm(), "c2_verpo": _fake_arm("a")},
        ("a", "b"),
    )
    assert report["unit"].startswith("heldout_task_with_all_three")
    assert report["candidates_per_cluster_per_arm"] == 30
    assert report["metrics"]["pass_at_10"]["difference_verpo_minus_baseline"] == 0.5
    assert report["metrics"]["pass_at_30"]["difference_verpo_minus_baseline"] == 0.5


def test_report_policy_cannot_promote() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert '"automatic_promotion_performed": False' in source
    assert '"promoted_checkpoint": None' in source
    assert '"promotion_permitted_from_this_report": False' in source
    assert audit.EXPECTED_SEEDS == (42, 43, 44)


def test_build_report_is_matched_three_seed_and_nonpromoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "BOOTSTRAP_REPLICATES", 50)
    task_ids = ["a", "b"]

    def fake_load(*, label, seed, predictions_path, score_path):
        arm = _fake_arm("a" if label == "c2_verpo" else None)[seed]
        return {
            "label": label,
            "seed": seed,
            "task_ids": task_ids,
            "task_values": arm["task_values"],
            "score": {},
            "model_sha256": "b" * 64 if label == "c2_baseline" else "v" * 64,
            "checkpoint_contract_sha256": (
                "c" * 64 if label == "c2_baseline" else "d" * 64
            ),
            "metrics": {
                "pass_at_1": {"count": 0, "rate": 0.0},
                "pass_at_k": {
                    "count": 0 if label == "c2_baseline" else 1,
                    "rate": 0.0 if label == "c2_baseline" else 0.5,
                },
                "compile_at_k": {"count": 2, "rate": 1.0},
            },
            "distinct_mean": 10.0,
            "tasks_below_10_distinct": 0,
            "distinct_histogram": {str(index): int(index == 10) * 2 for index in range(1, 11)},
            "artifacts": {},
        }

    monkeypatch.setattr(audit, "_load_artifact", fake_load)
    args = argparse.Namespace(
        artifact=[
            f"{label}|{seed}|pred-{label}-{seed}|score-{label}-{seed}"
            for label in audit.EXPECTED_LABELS
            for seed in audit.EXPECTED_SEEDS
        ]
    )
    report = audit.build_report(args)
    assert report["status"] == "complete"
    assert set(report["paired_by_seed"]) == {"42", "43", "44"}
    assert report["paired_task_cluster_bootstrap"]["candidates_per_cluster_per_arm"] == 30
    assert report["decision"]["automatic_promotion_performed"] is False
    assert report["decision"]["promoted_checkpoint"] is None


def test_launcher_requires_final_go_and_runs_exact_matched_eval() -> None:
    project = Path(__file__).resolve().parents[1]
    launcher = project / "deploy/vast/t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh"
    conf = project / "deploy/vast/t5gemma2-typed-c2-verpo-matched-eval8192-v1.conf"
    text = launcher.read_text(encoding="utf-8")
    conf_text = conf.read_text(encoding="utf-8")
    assert 'checkpoint-optstep-000150' in text
    assert '.status == "complete"' in text
    assert '.updates == 150' in text
    assert '.window_gates_passed == [16,32,48,64,80,96,112,128,144]' in text
    assert 'for gate_update in 16 32 48 64 80 96 112 128 144' in text
    assert 'and .decision == "GO"' in text
    assert 'for seed in 42 43 44' in text
    assert text.count('--max_new_tokens 8192') == 2
    assert 't5gemma2_typed_fold_gold_replay_inference_v1.py' in text
    assert 't5gemma2_typed_c2_verpo_inference_v1.py' in text
    assert 'audit_t5gemma2_typed_c2_verpo_multiseed.py' in text
    assert '.decision.automatic_promotion_performed == false' in text
    assert '.decision.promoted_checkpoint == null' in text
    assert '[program:t5gemma2-typed-c2-verpo-matched-eval8192-v1]' in conf_text
    assert 'autostart=false' in conf_text
    assert 'autorestart=false' in conf_text


def test_launcher_pins_current_eval_and_training_code() -> None:
    project = Path(__file__).resolve().parents[1]
    launcher = project / "deploy/vast/t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh"
    text = launcher.read_text(encoding="utf-8")
    for relative in (
        "scripts/training/t5gemma2_typed_c2_verpo_pilot150.py",
        "scripts/evaluation/t5gemma2_typed_c2_verpo_inference_v1.py",
        "scripts/evaluation/audit_t5gemma2_typed_c2_verpo_multiseed.py",
        "scripts/evaluation/t5gemma2_measurement_audit_inference.py",
        "scripts/evaluation/t5gemma2_f2_passk_inference.py",
        "scripts/evaluation/score_direct_compact_passk.py",
    ):
        assert sha256_file(project / relative) in text
