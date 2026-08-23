from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation import audit_t5gemma2_typed_pass3_promotion as gate
from scripts.evaluation import seal_t5gemma2_typed_pass3_checkpoint as checkpoint_seal
from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation import (
    t5gemma2_typed_seed_replication_inference_v1 as replication_adapter,
)
from scripts.evaluation.durable_evaluation_journal import load_journal, sha256_file
from scripts.training import t5gemma2_typed_c002_prefix3_verify as prefix
from scripts.training import t5gemma2_typed_direct_rs_sft_pass3 as pass3


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def _score(*, pass_tasks: int, nine_distinct_tasks: int) -> dict:
    candidates = []
    task_results = []
    for task_position in range(175):
        task_id = f"task-{task_position:03d}"
        passed_task = task_position < pass_tasks
        for sample in range(10):
            distinct_sample = 8 if task_position < nine_distinct_tasks and sample == 9 else sample
            digest = hashlib.sha256(
                f"{task_id}-candidate-{distinct_sample}".encode()
            ).hexdigest()
            passed = passed_task and sample == 0
            candidates.append(
                {
                    "task_id": task_id,
                    "sample_index": sample,
                    "code_sha256": digest,
                    "raw_sha256": digest,
                    "compiled": True,
                    "passed": passed,
                    "diagnostic": "",
                }
            )
        task_results.append(
            {
                "task_id": task_id,
                "pass_at_1": passed_task,
                "pass_at_k": passed_task,
                "compile_at_k": True,
                "passing_samples": 1 if passed_task else 0,
                "compiling_samples": 10,
            }
        )
    return {
        "schema": gate.SCORE_SCHEMA,
        "tasks": 175,
        "k": 10,
        "stability_runs": 2,
        "evaluation": {"sha256": gate.EVALUATION_SHA256, "path": "/sealed/dev.jsonl"},
        "candidate_results": candidates,
        "task_results": task_results,
        "pass_at_1": {"count": pass_tasks, "rate": pass_tasks / 175},
        "pass_at_k": {"count": pass_tasks, "rate": pass_tasks / 175},
        "compile_at_k": {"count": 175, "rate": 1.0},
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _pass3_stage(tmp_path: Path) -> tuple[Path, str]:
    stage = tmp_path / "pass3-stage"
    checkpoint = stage / "checkpoint-optstep-000004"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    trainer_sha = inference.PASS3_TRAINER_SHA256
    schedule = []
    for index in range(13):
        category = "kimi_c001" if index < 12 else "kimi_c002_prefix"
        schedule.append(
            {
                "position": index,
                "pair_id": f"task-{index}::pass3::{category}",
                "source_task_id": f"task-{index}",
                "source_category": category,
            }
        )
    dataset = {
        "schema": checkpoint_seal.DATASET_SCHEMA,
        "rows": 13,
        "architecture": "native_encoder_decoder",
        "composition": {
            "verified_direct": 13,
            "kimi_c001": 12,
            "kimi_c002_tail": 0,
            "kimi_c002_prefix": 1,
            "prior_225_replay": 0,
            "pass2_209_replay": 0,
            "gold_replay": 0,
            "repair_conditioned": 0,
            "reasoning_rows": 0,
        },
        "typed_train": {
            "model_visible_fields": checkpoint_seal.EXPECTED_VISIBLE_FIELDS,
            "opaque_contract": {
                "parameter_name_policy": "p{zero_based_index}",
                "semantic_function_name_exposed": False,
                "semantic_parameter_names_exposed": False,
            },
            "training_exclusions": {
                "task_ids": [checkpoint_seal.KNOWN_CONTAMINANT]
            },
        },
        "prior_225_exclusion": {"rows": 225},
        "heldout_overlap": 0,
        "heldout_175_model_visible": False,
        "tests_model_visible": False,
        "private_feedback_model_visible": False,
        "repair_conditioned_prefixes_visible": False,
        "reasoning_model_visible": False,
        "known_contaminant_excluded": checkpoint_seal.KNOWN_CONTAMINANT,
        "model_visible_fields": checkpoint_seal.EXPECTED_VISIBLE_FIELDS,
        "task_id_deduplication": "reject_any_cross_source_or_prior_overlap",
        "all_targets_bound_to_provider_or_zero_api_verification_journals": True,
        "production_floor_eligible": True,
        "full_acceptance_reverification": {
            "rows": 13,
            "passed": 13,
            "tests_model_visible": False,
            "diagnostics_persisted": False,
        },
        "schedule": schedule,
        "schedule_sha256": checkpoint_seal.canonical_sha256(schedule),
        "task_ids_sha256": checkpoint_seal.canonical_sha256(
            [row["source_task_id"] for row in schedule]
        ),
    }
    contract = {
        "schema": checkpoint_seal.RUN_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "model": checkpoint_seal.MODEL,
        "model_revision": checkpoint_seal.MODEL_REVISION,
        "base_model": {
            "name": checkpoint_seal.MODEL,
            "resolved_commit": checkpoint_seal.MODEL_REVISION,
            "is_encoder_decoder": True,
            "config_sha256": "b97320af5c9c921ae14cc6f73b365c936d6ddd14fd70992fd872c9a905048b5f",
        },
        "runtime": {"trainer_sha256": trainer_sha},
        "warmstart": {
            "update": 58,
            "run_contract_sha256": checkpoint_seal.UPDATE58_RUN_CONTRACT_SHA256,
            "adapter_weights_sha256": checkpoint_seal.UPDATE58_ADAPTER_SHA256,
            "adapter_config_sha256": checkpoint_seal.UPDATE58_ADAPTER_CONFIG_SHA256,
        },
        "dataset": dataset,
        "optimization": {
            "epochs": 2,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.0,
            "warmup_updates": 0,
            "updates_per_epoch": 2,
            "planned_updates": 4,
            "seed": 42,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "targets": ["q_proj"],
            "new_adapter_attached": False,
            "warmstart_weights_continued": True,
        },
        "privacy": {
            "heldout_overlap": 0,
            "heldout_content_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "reasoning_persisted": False,
        },
        "production_floor_eligible": True,
    }
    result = {
        "schema": checkpoint_seal.RUN_SCHEMA,
        "status": "complete",
        "rows": 13,
        "updates": 4,
        "planned_updates": 4,
        "latest_checkpoint": checkpoint.name,
        "production_floor_eligible": True,
    }
    latest = {
        "schema": checkpoint_seal.CHECKPOINT_SCHEMA,
        "path": str(checkpoint.resolve()),
        "update": 4,
        "run_contract_sha256": checkpoint_seal.canonical_sha256(contract),
    }
    _write_json(stage / "run_contract.json", contract)
    _write_json(stage / "dataset_manifest.json", dataset)
    _write_json(stage / "result.json", result)
    _write_json(stage / "latest_checkpoint.json", latest)
    _write_json(checkpoint / "run_contract.json", contract)
    _write_json(
        checkpoint / "adapter" / "adapter_config.json",
        {
            "base_model_name_or_path": checkpoint_seal.MODEL,
            "bias": "none",
            "peft_type": "LORA",
            "task_type": "SEQ_2_SEQ_LM",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj"],
            "use_dora": False,
            "use_rslora": False,
        },
    )
    from safetensors.torch import save_file
    import torch

    save_file(
        {
            "base_model.model.q_proj.lora_A.weight": torch.zeros((1, 1)),
            "base_model.model.q_proj.lora_B.weight": torch.zeros((1, 1)),
        },
        checkpoint / "adapter" / "adapter_model.safetensors",
    )
    _write_json(checkpoint / "tokenizer" / "tokenizer.json", {"version": "1"})
    (checkpoint / "training_state.pt").write_bytes(b"training-state")
    return stage, trainer_sha


def _generation(tmp_path: Path, label: str) -> tuple[Path, Path]:
    predictions_path = tmp_path / f"{label}_predictions.json"
    predictions = [
        {
            "id": f"task-{task_position:03d}",
            "predictions": [f"candidate-{sample}" for sample in range(10)],
        }
        for task_position in range(175)
    ]
    _write_json(predictions_path, predictions)
    provenance_path = Path(str(predictions_path) + ".provenance.json")
    provenance = {
        "schema": "t5gemma2-f2-measurement-ablation-provenance-v1",
        "num_rows": 175,
        "num_samples": 10,
        "input_view": "typed_opaque_contract",
        "output_sha256": sha256_file(predictions_path),
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "sampling": {
            "seed": 42,
            "seed_policy": "seed+task_index*100003+batch_start",
            "num_samples": 10,
            "generation_batch_size": 10,
            "max_source_tokens": 32768,
            "max_new_tokens": 4096,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 0,
            "sampled_eos_retained": True,
            "fabricated_eos": False,
        },
        "heldout": {
            "dataset": {"sha256": gate.EVALUATION_SHA256},
            "selected_rows": 175,
            "tests_serialized_to_model": False,
            "gold_targets_serialized_to_model": False,
            "input_view": {"view": "typed_opaque_contract"},
        },
    }
    _write_json(provenance_path, provenance)
    return predictions_path, provenance_path


def test_single_seed_can_clear_diagnostics_but_never_promotes(tmp_path: Path) -> None:
    # 21 duplicate tasks gives (1750 - 21) / 175 = 9.88 exactly.
    baseline_path = tmp_path / "update58.json"
    current_path = tmp_path / "pass3.json"
    baseline_predictions, baseline_provenance = _generation(tmp_path, "update58")
    current_predictions, current_provenance = _generation(tmp_path, "pass3")
    baseline_score = _score(pass_tasks=18, nine_distinct_tasks=21)
    current_score = _score(pass_tasks=19, nine_distinct_tasks=0)
    baseline_score["predictions"] = {
        "sha256": sha256_file(baseline_predictions),
        "provenance_sha256": sha256_file(baseline_provenance),
    }
    current_score["predictions"] = {
        "sha256": sha256_file(current_predictions),
        "provenance_sha256": sha256_file(current_provenance),
    }
    _write_json(baseline_path, baseline_score)
    _write_json(current_path, current_score)
    checker = WORKSPACE / "analysis_rs_sft_fold" / "check_collapse.py"
    args = argparse.Namespace(
        pass3_score=str(current_path),
        expected_pass3_score_sha256=sha256_file(current_path),
        pass3_predictions=str(current_predictions),
        expected_pass3_predictions_sha256=sha256_file(current_predictions),
        pass3_provenance=str(current_provenance),
        expected_pass3_provenance_sha256=sha256_file(current_provenance),
        update58_score=str(baseline_path),
        expected_update58_score_sha256=sha256_file(baseline_path),
        update58_predictions=str(baseline_predictions),
        expected_update58_predictions_sha256=sha256_file(baseline_predictions),
        update58_provenance=str(baseline_provenance),
        expected_update58_provenance_sha256=sha256_file(baseline_provenance),
        collapse_checker=str(checker),
        expected_collapse_checker_sha256=sha256_file(checker),
        output=str(tmp_path / "decision.json"),
    )
    report = gate.audit(args)
    assert report["decision"]["seed42_diagnostic_eligible"] is True
    assert report["decision"]["promotion_status"] == "HOLD_REQUIRES_3PLUS_SEEDS"
    assert report["decision"]["promoted_checkpoint"] is None
    assert report["decision"]["verpo_status"] == "HOLD"
    assert report["decision"]["update58_diversity_eligible_under_9_90_bar"] is False
    assert report["contract"]["primary_diagnostic_read"] == (
        "mean_distinct_extracted_code_sha256_per_10"
    )
    assert report["replication_status"]["additional_matched_pass3_seeds_required"] == 2
    assert report["human_readable_check_collapse"]["return_code_used_as_promotion_gate"] is False


def test_gate_rejects_incomplete_sample_coverage(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    score = _score(pass_tasks=18, nine_distinct_tasks=21)
    score["candidate_results"].pop()
    _write_json(path, score)
    with pytest.raises(ValueError, match="full175/k10"):
        gate._validate_score(path, label="broken")


def test_pass3_profile_is_exact_update58_two_epoch_no_replay() -> None:
    args = argparse.Namespace(
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=13,
        min_repair_conditioned_targets=0,
        expected_warmstart_update=58,
        epochs=2,
        batch_size=1,
        gradient_accumulation=8,
        max_updates=0,
        learning_rate=2e-5,
        warmup_ratio=0.0,
        seed=42,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        local_report=["prior"],
        api_report=list(range(9)),
    )
    pass3._validate_profile_args(args)
    args.gold_replay_rows = 1
    with pytest.raises(ValueError, match="gold_replay_rows"):
        pass3._validate_profile_args(args)


def test_prefix_journal_is_zero_api_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "prefix.journal.jsonl"
    events = [
        {"event": "header", "schema": prefix.JOURNAL_SCHEMA, "provider_calls": 0},
        {"event": "complete", "schema": prefix.JOURNAL_SCHEMA, "provider_calls": 0},
    ]
    prefix._write_journal_exact(path, events)
    first = path.read_bytes()
    prefix._write_journal_exact(path, events)
    assert path.read_bytes() == first
    assert len(load_journal(path)) == 2


def test_checkpoint_seal_emits_replication_manifest_without_promotion(
    tmp_path: Path,
) -> None:
    stage, trainer_sha = _pass3_stage(tmp_path)
    audit = tmp_path / "training-audit.json"
    manifest = tmp_path / "checkpoint-manifest.json"
    result = tmp_path / "checkpoint-seal-result.json"
    sealed = checkpoint_seal.seal(
        argparse.Namespace(
            stage_dir=str(stage),
            expected_trainer_sha256=trainer_sha,
            expected_adapter_target_modules_sha256=checkpoint_seal.canonical_sha256(
                ["q_proj"]
            ),
            output_audit=str(audit),
            output_manifest=str(manifest),
            output_result=str(result),
        )
    )
    assert sealed["promotion_status"] == "HOLD_REQUIRES_3PLUS_SEEDS"
    assert sealed["verpo_status"] == "HOLD"
    checkpoint = stage / "checkpoint-optstep-000004"
    admitted = replication_adapter.validate_pass3_manifest(
        manifest_path=manifest,
        expected_sha256=sha256_file(manifest),
        checkpoint=checkpoint,
    )
    assert admitted["no_automatic_promotion"] is True
    assert admitted["privacy"]["tests_model_visible"] is False
    assert json.loads(audit.read_text())["training_profile"]["gold_replay_rows"] == 0


def test_checkpoint_seal_rejects_replay(tmp_path: Path) -> None:
    stage, trainer_sha = _pass3_stage(tmp_path)
    contract_path = stage / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["dataset"]["composition"]["gold_replay"] = 1
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match="direct-only/no-replay"):
        checkpoint_seal.seal(
            argparse.Namespace(
                stage_dir=str(stage),
                expected_trainer_sha256=trainer_sha,
                expected_adapter_target_modules_sha256=checkpoint_seal.canonical_sha256(
                    ["q_proj"]
                ),
                output_audit=str(tmp_path / "audit.json"),
                output_manifest=str(tmp_path / "manifest.json"),
                output_result=str(tmp_path / "result.json"),
            )
        )


def test_checkpoint_seal_uses_weight_keys_when_peft_minimizes_targets(
    tmp_path: Path,
) -> None:
    stage, trainer_sha = _pass3_stage(tmp_path)
    config_path = (
        stage / "checkpoint-optstep-000004" / "adapter" / "adapter_config.json"
    )
    config = json.loads(config_path.read_text())
    config["target_modules"] = ["proj"]
    _write_json(config_path, config)
    sealed = checkpoint_seal.seal(
        argparse.Namespace(
            stage_dir=str(stage),
            expected_trainer_sha256=trainer_sha,
            expected_adapter_target_modules_sha256=checkpoint_seal.canonical_sha256(
                ["proj"]
            ),
            output_audit=str(tmp_path / "audit.json"),
            output_manifest=str(tmp_path / "manifest.json"),
            output_result=str(tmp_path / "result.json"),
        )
    )
    assert sealed["status"] == "complete"
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["adapter_weight_targets"] == {
        "count": 1,
        "matches_run_contract": True,
        "sorted_names_sha256": checkpoint_seal.canonical_sha256(["q_proj"]),
    }


def test_checkpoint_seal_rejects_weight_target_mismatch(tmp_path: Path) -> None:
    stage, trainer_sha = _pass3_stage(tmp_path)
    from safetensors.torch import save_file
    import torch

    weights = (
        stage
        / "checkpoint-optstep-000004"
        / "adapter"
        / "adapter_model.safetensors"
    )
    save_file(
        {
            "base_model.model.other_proj.lora_A.weight": torch.zeros((1, 1)),
            "base_model.model.other_proj.lora_B.weight": torch.zeros((1, 1)),
        },
        weights,
    )
    with pytest.raises(ValueError, match="adapter weight targets"):
        checkpoint_seal.seal(
            argparse.Namespace(
                stage_dir=str(stage),
                expected_trainer_sha256=trainer_sha,
                expected_adapter_target_modules_sha256=checkpoint_seal.canonical_sha256(
                    ["q_proj"]
                ),
                output_audit=str(tmp_path / "audit.json"),
                output_manifest=str(tmp_path / "manifest.json"),
                output_result=str(tmp_path / "result.json"),
            )
        )


def test_checkpoint_seal_rejects_unmatched_adapter_weight_key(tmp_path: Path) -> None:
    stage, trainer_sha = _pass3_stage(tmp_path)
    from safetensors.torch import save_file
    import torch

    weights = (
        stage
        / "checkpoint-optstep-000004"
        / "adapter"
        / "adapter_model.safetensors"
    )
    save_file(
        {
            "base_model.model.q_proj.lora_A.weight": torch.zeros((1, 1)),
            "base_model.model.q_proj.lora_B.weight": torch.zeros((1, 1)),
            "base_model.model.unexpected.weight": torch.zeros((1, 1)),
        },
        weights,
    )
    with pytest.raises(ValueError, match="canonical LoRA target set"):
        checkpoint_seal.seal(
            argparse.Namespace(
                stage_dir=str(stage),
                expected_trainer_sha256=trainer_sha,
                expected_adapter_target_modules_sha256=checkpoint_seal.canonical_sha256(
                    ["q_proj"]
                ),
                output_audit=str(tmp_path / "audit.json"),
                output_manifest=str(tmp_path / "manifest.json"),
                output_result=str(tmp_path / "result.json"),
            )
        )


def test_shared_inference_accepts_only_exact_pass3_contract(tmp_path: Path) -> None:
    stage, _trainer_sha = _pass3_stage(tmp_path)
    checkpoint = stage / "checkpoint-optstep-000004"
    contract, record = inference._checkpoint_record(checkpoint, "sft")  # noqa: SLF001
    assert contract["schema"] == inference.TYPED_DIRECT_RS_SFT_PASS3_RUN_SCHEMA
    assert record["training_stage_schema"] == contract["schema"]
    assert record["adapter"]["target_modules"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda c: c["runtime"].update(trainer_sha256="0" * 64), "lineage"),
        (lambda c: c["warmstart"].update(run_contract_sha256="0" * 64), "lineage"),
        (lambda c: c["privacy"].update(tests_model_visible=True), "privacy"),
        (
            lambda c: c["dataset"]["composition"].update(gold_replay=1),
            "dataset",
        ),
    ],
)
def test_shared_inference_rejects_pass3_contract_mutations(
    tmp_path: Path, mutation, match: str
) -> None:
    stage, _trainer_sha = _pass3_stage(tmp_path)
    checkpoint = stage / "checkpoint-optstep-000004"
    contract_path = checkpoint / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    mutation(contract)
    _write_json(contract_path, contract)
    with pytest.raises(ValueError, match=match):
        inference._checkpoint_record(checkpoint, "sft")  # noqa: SLF001


def test_launch_chain_holds_promotion_and_verpo() -> None:
    handoff = (ROOT / "deploy/vast/t5gemma2_typed_pass3_after_c002_handoff.sh").read_text()
    evaluation = (
        ROOT / "deploy/vast/t5gemma2_typed_direct_rs_sft_pass3_eval_gate.sh"
    ).read_text()
    trainer = (ROOT / "deploy/vast/t5gemma2_typed_direct_rs_sft_pass3.sh").read_text()
    conf = (ROOT / "deploy/vast/t5gemma2-typed-pass3-after-c002-handoff.conf").read_text()
    assert "t5gemma2-typed-kimi-c002-resume47" in handoff
    assert '"${PREFIX_LAUNCHER}"' in handoff
    assert '"${TRAIN_LAUNCHER}"' in handoff
    assert '"${EVAL_LAUNCHER}"' in handoff
    assert "checkpoint_manifest_sha256" in handoff
    assert "HOLD_REQUIRES_3PLUS_SEEDS" in evaluation
    assert "verpo=HOLD" in evaluation
    assert "supervisorctl start" not in evaluation.lower()
    assert "seal_t5gemma2_typed_pass3_checkpoint.py" in evaluation
    assert "--expected_warmstart_update 58" in trainer
    assert "--epochs 2" in trainer and "--learning_rate 2e-5" in trainer
    assert "--gold_replay_rows 0" in trainer
    assert "autostart=false" in conf and "autorestart=false" in conf
    assert "exitcodes=0,78" in conf


def test_new_files_have_no_unresolved_hash_placeholders() -> None:
    paths = (
        ROOT / "deploy/vast/t5gemma2_typed_c002_prefix3_verify.sh",
        ROOT / "deploy/vast/t5gemma2_typed_direct_rs_sft_pass3.sh",
        ROOT / "deploy/vast/t5gemma2_typed_direct_rs_sft_pass3_eval_gate.sh",
        ROOT / "deploy/vast/t5gemma2_typed_pass3_after_c002_handoff.sh",
    )
    assert all("__" not in path.read_text() for path in paths)
