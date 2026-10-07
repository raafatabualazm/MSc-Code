from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.evaluation import audit_t5gemma2_typed_fold_promotion as fold_gate
from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation.durable_evaluation_journal import sha256_file
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_fold_rs_sft_union_v1 as fold


ROOT = Path(__file__).resolve().parents[1]


def _profile_args() -> argparse.Namespace:
    return argparse.Namespace(
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=447,
        min_repair_conditioned_targets=0,
        expected_warmstart_update=348,
        epochs=1,
        batch_size=1,
        gradient_accumulation=8,
        max_updates=0,
        learning_rate=5e-6,
        warmup_ratio=0.0,
        seed=42,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        local_report=list(range(9)),
        api_report=list(range(19)),
    )


def _minimum_union() -> tuple[
    list[tuple[str, mixed.MixedPair]],
    list[base_sft.TextPair],
    dict[str, str],
    dict[str, str],
]:
    categories = (
        [("pass1_225", 225), ("pass2_209", 209), ("kimi_c001", 12), ("kimi_c002_prefix", 1)]
    )
    collected: list[tuple[str, mixed.MixedPair]] = []
    typed: list[base_sft.TextPair] = []
    tests: dict[str, str] = {}
    gold: dict[str, str] = {}
    global_position = 0
    for category, count in categories:
        for local_position in range(count):
            task_id = f"task-{global_position:03d}"
            source = (
                "typed opaque equivalent source"
                if global_position < 2
                else f"typed opaque source {task_id}"
            )
            # The first two tasks share typed input but have different targets;
            # the next two have distinct input but retain byte-identical code.
            if global_position < 2:
                target = f"int fn0() => {9000 + global_position};"
            elif global_position < 4:
                target = "int fn0() => 9999;"
            else:
                target = f"int fn0() => {global_position};"
            pair = mixed._make_pair(  # noqa: SLF001
                pair_id=f"upstream::{task_id}",
                source_task_id=task_id,
                kind="verified_direct",
                source=source,
                target=target,
                provenance=(("upstream", category),),
            )
            collected.append((category, pair))
            typed.append(
                base_sft.TextPair(
                    task_id=task_id,
                    source=source,
                    target=f"int fn0() => {-global_position};",
                    source_sha256=mixed.sha256_text(source),
                    target_sha256=mixed.sha256_text(f"int fn0() => {-global_position};"),
                )
            )
            tests[task_id] = f"private tests {task_id}"
            gold[task_id] = f"int fn0() => {-global_position};"
            global_position += 1
    return collected, typed, tests, gold


def _typed_manifest() -> dict:
    return {
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "opaque_contract": {
            "parameter_name_policy": "p{zero_based_index}",
            "semantic_function_name_exposed": False,
            "semantic_parameter_names_exposed": False,
        },
        "training_exclusions": {"task_ids": [fold.pass1.CONTAMINATED_TRAIN_TASK_ID]},
        "heldout": {
            "task_id_overlap": 0,
            "exact_gold_source_overlap": 0,
            "exact_acceptance_test_overlap": 0,
            "model_visible": False,
        },
    }


def test_profile_is_fresh_typed_sft_one_epoch_low_lr_no_replay() -> None:
    args = _profile_args()
    fold._validate_profile_args(args)
    for name, bad in (
        ("expected_warmstart_update", 58),
        ("epochs", 2),
        ("learning_rate", 2e-5),
        ("gold_replay_rows", 1),
    ):
        changed = argparse.Namespace(**vars(args))
        setattr(changed, name, bad)
        with pytest.raises(ValueError, match=name):
            fold._validate_profile_args(changed)


def test_shared_inference_validates_exact_fold_contract() -> None:
    collected, typed, tests, gold = _minimum_union()
    _pairs, manifest = fold._rebind_union(  # noqa: SLF001
        collected=collected,
        typed_pairs=typed,
        heldout_ids=set(),
        tests_by_id=tests,
        gold_target_by_id=gold,
        seed=42,
        source_audits={"sealed": True},
        typed_manifest=_typed_manifest(),
        heldout_record={"rows": 175},
        verify=lambda _code, _tests, _slot: True,
        verification_workers=1,
    )
    contract = {
        "schema": inference.TYPED_FOLD_RS_SFT_UNION_RUN_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "runtime": {"trainer_sha256": inference.FOLD_TRAINER_SHA256},
        "warmstart": {
            "checkpoint_name": "checkpoint-optstep-000348",
            "update": 348,
            "run_contract_sha256": inference.TYPED_SFT_RUN_CONTRACT_SHA256,
            "adapter_weights_sha256": inference.TYPED_SFT_ADAPTER_WEIGHTS_SHA256,
            "adapter_config_sha256": inference.TYPED_SFT_ADAPTER_CONFIG_SHA256,
        },
        "warmstart_contract_schema": inference.TYPED_CONTRACT_SFT_RUN_SCHEMA,
        "dataset": manifest,
        "optimization": {
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "learning_rate": 5e-6,
            "warmup_updates": 0,
            "updates_per_epoch": 56,
            "planned_updates": 56,
            "seed": 42,
        },
        "lora": {
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
    assert contract["schema"] in inference.SUPPORTED_ADAPTER_RUN_SCHEMAS
    inference._require_typed_fold_contract(contract)  # noqa: SLF001
    contract["dataset"]["composition"]["gold_replay"] = 1
    with pytest.raises(ValueError, match="dataset"):
        inference._require_typed_fold_contract(contract)  # noqa: SLF001


def test_fold_rejects_duplicate_tasks_but_retains_equal_code_for_distinct_tasks() -> None:
    collected, typed, tests, gold = _minimum_union()
    pairs, manifest = fold._rebind_union(  # noqa: SLF001
        collected=collected,
        typed_pairs=typed,
        heldout_ids=set(),
        tests_by_id=tests,
        gold_target_by_id=gold,
        seed=42,
        source_audits={"sealed": True},
        typed_manifest=_typed_manifest(),
        heldout_record={"rows": 175},
        verify=lambda _code, _tests, _slot: True,
        verification_workers=1,
    )
    assert len(pairs) == 447
    assert manifest["composition"] == {
        "verified_direct": 447,
        "pass1_225": 225,
        "pass2_209": 209,
        "kimi_c001": 12,
        "kimi_c002_prefix": 1,
        "kimi_c002_tail": 0,
        "gold_replay": 0,
        "repair_conditioned": 0,
        "reasoning_rows": 0,
        "independently_generated_exact_gold_matches": 0,
    }
    assert manifest["shared_target_code_groups"] == 1
    assert manifest["shared_target_code_rows"] == 2
    assert manifest["target_code_deduplication"] == "none_retain_same_code_for_distinct_tasks"
    assert manifest["equivalent_typed_source_groups"] == 1
    assert manifest["equivalent_typed_source_rows"] == 2
    assert manifest["equivalent_typed_source_cross_acceptance"]["rows"] == 4
    assert manifest["equivalent_typed_source_cross_acceptance"]["passed"] == 4
    assert manifest["promotion_status"] == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"

    duplicate = list(collected)
    duplicate[-1] = (duplicate[-1][0], duplicate[0][1])
    with pytest.raises(ValueError, match="duplicate task IDs"):
        fold._rebind_union(  # noqa: SLF001
            collected=duplicate,
            typed_pairs=typed,
            heldout_ids=set(),
            tests_by_id=tests,
            gold_target_by_id=gold,
            seed=42,
            source_audits={},
            typed_manifest=_typed_manifest(),
            heldout_record={},
            verify=lambda _code, _tests, _slot: True,
            verification_workers=1,
        )


def test_fold_rejects_heldout_and_known_contaminant() -> None:
    collected, typed, tests, gold = _minimum_union()
    with pytest.raises(ValueError, match="held-out"):
        fold._rebind_union(  # noqa: SLF001
            collected=collected,
            typed_pairs=typed,
            heldout_ids={collected[0][1].source_task_id},
            tests_by_id=tests,
            gold_target_by_id=gold,
            seed=42,
            source_audits={},
            typed_manifest=_typed_manifest(),
            heldout_record={},
            verify=lambda _code, _tests, _slot: True,
            verification_workers=1,
        )


def test_equivalent_typed_source_requires_cartesian_private_acceptance() -> None:
    collected, typed, tests, gold = _minimum_union()

    def reject_cross(code: str, private_tests: str, _slot: str) -> bool:
        return not ("9000" in code and private_tests.endswith("task-001"))

    with pytest.raises(ValueError, match="acceptance re-verification"):
        fold._rebind_union(  # noqa: SLF001
            collected=collected,
            typed_pairs=typed,
            heldout_ids=set(),
            tests_by_id=tests,
            gold_target_by_id=gold,
            seed=42,
            source_audits={},
            typed_manifest=_typed_manifest(),
            heldout_record={},
            verify=reject_cross,
            verification_workers=1,
        )


def test_published_manifest_is_a_seal_not_a_target_text_source(tmp_path: Path) -> None:
    value = {"schema": "x", "rows": 225, "schedule": [{"target_sha256": "a" * 64}]}
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    record = fold._require_manifest_reconstruction(  # noqa: SLF001
        value, (path, sha256_file(path)), "pass-1"
    )
    assert record["target_text_source"] == "sealed_producer_reports_not_dataset_manifest"
    with pytest.raises(ValueError, match="reconstruction differs"):
        fold._require_manifest_reconstruction(  # noqa: SLF001
            {**value, "rows": 224}, (path, sha256_file(path)), "pass-1"
        )


def test_fold_gate_relabels_shared_gate_and_never_promotes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fold_gate.shared,
        "audit",
        lambda _args: {
            "schema": "old",
            "status": "pass",
            "inputs": {"pass3": {"x": 1}, "update58": {}},
            "metrics": {"pass3": {"pass_at_10": 18}, "update58": {}},
            "decision": {"promotion_status": "old", "promoted_checkpoint": "bad"},
            "replication_status": {"minimum_required_for_promotion": 3},
        },
    )
    args = argparse.Namespace(
        fold_score="x", expected_fold_score_sha256="x",
        fold_predictions="x", expected_fold_predictions_sha256="x",
        fold_provenance="x", expected_fold_provenance_sha256="x",
        update58_score="x", expected_update58_score_sha256="x",
        update58_predictions="x", expected_update58_predictions_sha256="x",
        update58_provenance="x", expected_update58_provenance_sha256="x",
        collapse_checker="x", expected_collapse_checker_sha256="x",
    )
    report = fold_gate.audit(args)
    assert "fold" in report["inputs"] and "pass3" not in report["inputs"]
    assert "fold" in report["metrics"] and "pass3" not in report["metrics"]
    assert report["decision"]["promotion_status"] == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
    assert report["decision"]["promoted_checkpoint"] is None
    assert report["decision"]["verpo_status"] == "HOLD"


def test_launch_stack_is_separate_versioned_fold_arm() -> None:
    trainer = (ROOT / "deploy/vast/t5gemma2_typed_fold_rs_sft_union_v1.sh").read_text()
    evaluation = (ROOT / "deploy/vast/t5gemma2_typed_fold_rs_sft_union_v1_eval_gate.sh").read_text()
    handoff = (ROOT / "deploy/vast/t5gemma2_typed_fold_after_c002_handoff_v1.sh").read_text()
    conf = (ROOT / "deploy/vast/t5gemma2-typed-fold-after-c002-handoff-v1.conf").read_text()
    assert "--expected_warmstart_update 348" in trainer
    assert "--epochs 1" in trainer and "--learning_rate 5e-6" in trainer
    assert "--min_verified_direct_targets 447" in trainer
    assert "--gold_replay_rows 0" in trainer
    assert "typed_fold_union_seed42_k10" in evaluation
    assert "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS" in evaluation
    assert "verpo=HOLD" in evaluation
    assert '"${TRAIN_LAUNCHER}"' in handoff and '"${EVAL_LAUNCHER}"' in handoff
    assert "t5gemma2-typed-pass3-after-c002-handoff" in handoff
    assert "t5gemma2-typed-pass3-eval-retry-v1" in handoff
    assert "pass-3 handoff exited without sealed decision artifacts" in handoff
    assert "HOLD_REQUIRES_3PLUS_SEEDS" in handoff
    assert "pass3_snapshot_one" in handoff
    assert "autostart=false" in conf and "autorestart=false" in conf
    assert "exitcodes=0,78" in conf
    assert all("__" not in text for text in (trainer, evaluation, handoff))
