from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.evaluation import audit_t5gemma2_typed_fold_gold_replay_promotion as gate
from scripts.evaluation import t5gemma2_typed_fold_gold_replay_inference_v1 as inference
from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_fold_gold_replay_v1 as arm_c
from scripts.training import t5gemma2_typed_fold_rs_sft_union_v1 as arm_b


ROOT = Path(__file__).resolve().parents[1]


def _profile_args() -> argparse.Namespace:
    return argparse.Namespace(
        gold_replay_ratio=1.0,
        gold_replay_rows=458,
        min_verified_direct_targets=458,
        min_repair_conditioned_targets=0,
        expected_warmstart_update=348,
        epochs=1,
        batch_size=1,
        gradient_accumulation=16,
        max_updates=0,
        learning_rate=5e-6,
        weight_decay=0.0,
        warmup_ratio=0.0,
        max_source_tokens=32768,
        max_target_tokens=32768,
        checkpoint_interval=20,
        seed=42,
        attn_implementation="sdpa",
        bf16=True,
        gradient_checkpointing=True,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        local_report=list(range(9)),
        api_report=list(range(19)),
        arm_b_dataset_manifest="arm-b.json",
        expected_arm_b_dataset_manifest_sha256="a" * 64,
    )


def _text_pair(task_id: str, source: str, target: str) -> base_sft.TextPair:
    return base_sft.TextPair(
        task_id=task_id,
        source=source,
        target=target,
        source_sha256=mixed.sha256_text(source),
        target_sha256=mixed.sha256_text(target),
    )


def _synthetic_corpus() -> tuple[list[mixed.MixedPair], list[base_sft.TextPair]]:
    direct: list[mixed.MixedPair] = []
    typed: list[base_sft.TextPair] = []
    for index in range(arm_c.EXPECTED_DIRECT_ROWS):
        task_id = f"direct-{index:04d}"
        source = "direct-shared-source" if index < 2 else f"direct-source-{index:04d}"
        target = f"int fn0() => {index};"
        direct.append(
            mixed._make_pair(  # noqa: SLF001
                pair_id=f"arm-b::{task_id}",
                source_task_id=task_id,
                kind="verified_direct",
                source=source,
                target=target,
                provenance=(("source", "arm-b"),),
            )
        )
        typed.append(_text_pair(task_id, source, f"int fn0() => {-index};"))

    # There are 2,317 non-direct IDs. Two are aliases of direct model-visible
    # sources and are excluded; the remaining 2,315 rows contain one duplicate
    # source pair, yielding 2,314 eligible unique typed inputs.
    for index in range(2317):
        task_id = f"pool-{index:04d}"
        if index == 0:
            source = "direct-shared-source"
        elif index == 1:
            source = "direct-source-0002"
        elif index in (2, 3):
            source = "remaining-shared-source"
        else:
            source = f"pool-source-{index:04d}"
        typed.append(_text_pair(task_id, source, f"int fn0() => {10000 + index};"))
    assert len(typed) == 2775
    return direct, typed


def _typed_manifest() -> dict:
    return {
        "dataset": {"sha256": "1" * 64},
        "f2": {"sha256": "2" * 64},
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "training_exclusions": {"task_ids": [arm_c.pass1.CONTAMINATED_TRAIN_TASK_ID]},
        "heldout": {
            "task_id_overlap": 0,
            "exact_gold_source_overlap": 0,
            "exact_acceptance_test_overlap": 0,
            "model_visible": False,
        },
    }


def _direct_manifest(direct: list[mixed.MixedPair]) -> dict:
    schedule = [
        {
            "position": position,
            "pair_id": pair.pair_id,
            "source_task_id": pair.source_task_id,
            "kind": pair.kind,
            "source_sha256": pair.source_sha256,
            "target_sha256": pair.target_sha256,
            "provenance": dict(pair.provenance),
        }
        for position, pair in enumerate(direct)
    ]
    return {
        "schema": arm_b.DATASET_SCHEMA,
        "arm": "B_fold_only",
        "rows": 458,
        "composition": {
            "verified_direct": 458,
            "kimi_c002_tail": 11,
            "gold_replay": 0,
        },
        "schedule": schedule,
        "schedule_sha256": canonical_sha256(schedule),
        "pair_ids_sha256": canonical_sha256([pair.pair_id for pair in direct]),
        "source_sha256s_sha256": canonical_sha256(
            [pair.source_sha256 for pair in direct]
        ),
        "target_sha256s_sha256": canonical_sha256(
            [pair.target_sha256 for pair in direct]
        ),
        "full_acceptance_reverification": {
            "rows": 458,
            "passed": 458,
            "tests_model_visible": False,
            "diagnostics_persisted": False,
        },
    }


def _selected_replay(typed: list[base_sft.TextPair], direct: list[mixed.MixedPair]):
    direct_ids = {pair.source_task_id for pair in direct}
    direct_sources = {pair.source_sha256 for pair in direct}
    pool = [
        pair
        for pair in typed
        if pair.task_id not in direct_ids and pair.source_sha256 not in direct_sources
    ]
    pool.sort(
        key=lambda pair: canonical_sha256(
            {
                "schema": arm_c.REPLAY_RANKING_SCHEMA,
                "seed": 42,
                "kind": "gold_replay",
                "task_id": pair.task_id,
                "source_sha256": pair.source_sha256,
            }
        )
    )
    selected = []
    seen = set()
    for pair in pool:
        if pair.source_sha256 in seen:
            continue
        seen.add(pair.source_sha256)
        selected.append(pair)
    return pool, selected[:458]


def _compose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    direct, typed = _synthetic_corpus()
    direct_manifest = _direct_manifest(direct)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(direct_manifest), encoding="utf-8")
    pool, selected = _selected_replay(typed, direct)
    monkeypatch.setattr(
        arm_c,
        "EXPECTED_REPLAY_TASK_IDS_SHA256",
        canonical_sha256([pair.task_id for pair in selected]),
    )
    monkeypatch.setattr(
        arm_c,
        "EXPECTED_REPLAY_SOURCE_SHA256S_SHA256",
        canonical_sha256([pair.source_sha256 for pair in selected]),
    )
    monkeypatch.setattr(
        arm_c,
        "EXPECTED_REPLAY_TARGET_SHA256S_SHA256",
        canonical_sha256([pair.target_sha256 for pair in selected]),
    )
    monkeypatch.setattr(
        arm_c,
        "EXPECTED_REPLAY_PRODUCTION_ELIGIBLE_UNIQUE_SOURCES",
        2314,
    )
    monkeypatch.setattr(
        arm_c,
        "EXPECTED_REPLAY_PRODUCTION_INELIGIBLE_TASK_IDS",
        frozenset(),
    )
    tests = {pair.task_id: f"private-{pair.task_id}" for pair in typed}
    pairs, manifest = arm_c._compose_arm_c(  # noqa: SLF001
        direct_pairs=direct,
        direct_manifest=direct_manifest,
        arm_b_manifest_path=manifest_path,
        expected_arm_b_manifest_sha256=sha256_file(manifest_path),
        typed_pairs=typed,
        typed_manifest=_typed_manifest(),
        tests_by_id=tests,
        seed=42,
        verify=lambda _code, _tests, _slot: True,
        verification_workers=1,
    )
    return direct, typed, pool, selected, pairs, manifest


def test_profile_and_schedule_are_exact_replay_recipe() -> None:
    args = _profile_args()
    arm_c._validate_profile_args(args)  # noqa: SLF001
    schedule = base_sft.calculate_training_schedule(
        rows=916,
        epochs=1,
        batch_size=1,
        gradient_accumulation=16,
        max_updates=0,
        warmup_ratio=0.0,
    )
    assert schedule["planned_updates"] == 58
    for name, bad in (
        ("gold_replay_rows", 457),
        ("gradient_accumulation", 8),
        ("learning_rate", 2e-5),
        ("checkpoint_interval", 5),
        ("seed", 43),
    ):
        changed = argparse.Namespace(**vars(args))
        setattr(changed, name, bad)
        with pytest.raises(ValueError, match=name):
            arm_c._validate_profile_args(changed)  # noqa: SLF001


def test_amended_production_eligible_replay_digests_are_hard_pinned() -> None:
    assert arm_c.EXPECTED_REPLAY_TASK_IDS_SHA256 == (
        "6da49d120c902fde194c09fa14f7718bb379d8e676da8071211e1ac95da8e9df"
    )
    assert arm_c.EXPECTED_REPLAY_SOURCE_SHA256S_SHA256 == (
        "1c818f33808c4142eb7b148733ce6879a779f795f1855e66533832baa99b31d6"
    )
    assert arm_c.EXPECTED_REPLAY_TARGET_SHA256S_SHA256 == (
        "c7031487c72a2edba0baca1d8fe9eadc76232136c439a0f1fc95b25e2044e8f6"
    )


def test_arm_c_replay_is_exact_task_and_typed_source_disjoint_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    direct, _typed, pool, selected, pairs, manifest = _compose(tmp_path, monkeypatch)
    assert len(pool) == 2315
    assert len({pair.source_sha256 for pair in pool}) == 2314
    assert len(selected) == 458
    assert len(pairs) == 916
    assert manifest["composition"] == {
        "verified_direct": 458,
        "gold_replay": 458,
        "repair_conditioned": 0,
        "reasoning_rows": 0,
    }
    direct_ids = {pair.source_task_id for pair in direct}
    direct_sources = {pair.source_sha256 for pair in direct}
    replay = [pair for pair in pairs if pair.kind == "gold_replay"]
    assert not direct_ids & {pair.source_task_id for pair in replay}
    assert not direct_sources & {pair.source_sha256 for pair in replay}
    assert len({pair.source_sha256 for pair in replay}) == 458
    assert manifest["gold_replay"]["selection_key_fields"] == [
        "schema",
        "seed",
        "kind",
        "task_id",
        "source_sha256",
    ]
    assert manifest["gold_replay"]["selection_fixed_before_arm_b_result"] is False
    assert manifest["gold_replay"]["base_ranking_fixed_before_arm_b_result"] is True
    assert manifest["gold_replay"]["selection_amended_after_arm_b_result"] is True
    assert manifest["gold_replay"]["production_admissibility"]["eligible"] == 2314
    assert manifest["direct_union"]["canonical_manifest_sha256"] == canonical_sha256(
        _direct_manifest(direct)
    )
    assert manifest["pure_gold_content_causal_claim_permitted"] is False


def test_arm_b_manifest_is_a_required_byte_and_canonical_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    direct, _typed = _synthetic_corpus()
    manifest = _direct_manifest(direct)
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    record = arm_c._require_arm_b_manifest(  # noqa: SLF001
        manifest, path=path, expected_sha256=sha256_file(path)
    )
    assert (
        record["identity"] == "byte_pinned_manifest_and_canonical_reconstruction_match"
    )
    with pytest.raises(ValueError, match="canonical|reconstructed"):
        arm_c._require_arm_b_manifest(  # noqa: SLF001
            {**manifest, "rows": 457}, path=path, expected_sha256=sha256_file(path)
        )


def test_dedicated_inference_rejects_replay_source_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _direct, _typed, _pool, _selected, _pairs, manifest = _compose(
        tmp_path, monkeypatch
    )
    contract = {
        "schema": arm_c.RUN_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "runtime": {
            "trainer_sha256": sha256_file(Path(arm_c.__file__).resolve()),
            "fold_union_builder_sha256": sha256_file(Path(arm_b.__file__).resolve()),
            "typed_source_builder_sha256": sha256_file(
                Path(inference.typed_sft.__file__).resolve()
            ),
            "trainer_profile": (
                "typed_fold_plus_production_eligible_gold_replay_arm_c2_v2"
            ),
        },
        "warmstart": {
            "checkpoint_name": "checkpoint-optstep-000348",
            "update": 348,
            "run_contract_sha256": inference.base.TYPED_SFT_RUN_CONTRACT_SHA256,
            "adapter_weights_sha256": inference.base.TYPED_SFT_ADAPTER_WEIGHTS_SHA256,
            "adapter_config_sha256": inference.base.TYPED_SFT_ADAPTER_CONFIG_SHA256,
        },
        "warmstart_contract_schema": inference.base.TYPED_CONTRACT_SFT_RUN_SCHEMA,
        "dataset": manifest,
        "optimization": {
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "learning_rate": 5e-6,
            "warmup_updates": 0,
            "updates_per_epoch": 58,
            "planned_updates": 58,
            "seed": 42,
        },
        "checkpointing": {"interval": 20},
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
    inference._require_arm_c_contract(contract)  # noqa: SLF001
    contract["dataset"]["gold_replay"]["direct_typed_source_sha256s_excluded"] = False
    with pytest.raises(ValueError, match="dataset/privacy"):
        inference._require_arm_c_contract(contract)  # noqa: SLF001


def test_arm_c_gate_pairs_but_always_holds_single_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {}
    for name in (
        "arm_c_score",
        "arm_c_predictions",
        "arm_c_provenance",
        "arm_b_score",
        "arm_b_predictions",
        "arm_b_provenance",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        files[name] = path
    decision_path = tmp_path / "arm_b_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema": gate.ARM_B_DECISION_SCHEMA,
                "status": "pass",
                "automatic_promotion_performed": False,
                "verpo_launched": False,
                "decision": {"promotion_status": "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"},
                "inputs": {
                    "fold": {
                        "sha256": sha256_file(files["arm_b_score"]),
                        "generation": {
                            "predictions": {
                                "sha256": sha256_file(files["arm_b_predictions"])
                            },
                            "provenance": {
                                "sha256": sha256_file(files["arm_b_provenance"])
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    task_order = ["a", "b"]
    coordinates = canonical_sha256([("a", 0), ("b", 0)])

    def score(label: str) -> dict:
        current = label.startswith("Arm C")
        task_metrics = {
            "a": {"passk": current, "pass1": False, "compk": True},
            "b": {"passk": not current, "pass1": False, "compk": True},
        }
        return {
            "task_order": task_order,
            "task_order_sha256": canonical_sha256(task_order),
            "sample_coordinates_sha256": coordinates,
            "task_metrics": task_metrics,
            "pass_at_1": 0,
            "pass_at_10": 1,
            "compile_at_10": 2,
            "distinct_mean": 10.0,
            "distinct_counts_sha256": "d" * 64,
            "tasks_below_10_distinct": 0,
        }

    monkeypatch.setattr(
        gate.shared,
        "audit",
        lambda _args: {
            "schema": "shared",
            "status": "pass",
            "inputs": {"pass3": {}, "update58": {}},
            "metrics": {"pass3": {}, "update58": {}, "paired": {}},
            "decision": {
                "seed42_diagnostic_eligible": True,
                "promotion_status": "old",
            },
            "replication_status": {
                "validated_pass3_seeds": [42],
                "validated_seed_count": 1,
                "minimum_required_for_promotion": 3,
                "additional_matched_pass3_seeds_required": 2,
            },
        },
    )
    monkeypatch.setattr(
        gate.shared,
        "_validate_score",
        lambda _path, *, label: score(label),
    )
    monkeypatch.setattr(
        gate.shared,
        "_validate_generation",
        lambda **_kwargs: {"matched_seed42_contract_validated": True},
    )
    args = argparse.Namespace(
        **{name: str(path) for name, path in files.items()},
        **{
            f"expected_{name}_sha256": sha256_file(path) for name, path in files.items()
        },
        arm_b_decision=str(decision_path),
        expected_arm_b_decision_sha256=sha256_file(decision_path),
        update58_score="unused",
        expected_update58_score_sha256="unused",
        update58_predictions="unused",
        expected_update58_predictions_sha256="unused",
        update58_provenance="unused",
        expected_update58_provenance_sha256="unused",
        collapse_checker="unused",
        expected_collapse_checker_sha256="unused",
    )
    report = gate.audit(args)
    assert report["decision"]["promotion_status"] == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
    assert report["decision"]["promoted_checkpoint"] is None
    assert report["decision"]["verpo_status"] == "HOLD"
    assert report["replication_status"]["arm_b_same_seed_comparators_required"] is True
    paired = report["metrics"]["arm_c_vs_arm_b_paired"]["passk"]
    assert paired["gains"] == 1 and paired["losses"] == 1


def test_launch_stack_waits_for_b_and_never_auto_starts_or_deletes() -> None:
    train = (ROOT / "deploy/vast/t5gemma2_typed_fold_gold_replay_v1.sh").read_text()
    evaluation = (
        ROOT / "deploy/vast/t5gemma2_typed_fold_gold_replay_v1_eval_gate.sh"
    ).read_text()
    handoff = (
        ROOT / "deploy/vast/t5gemma2_typed_fold_gold_replay_handoff_v1.sh"
    ).read_text()
    conf = (ROOT / "deploy/vast/t5gemma2-typed-fold-gold-replay-v2.conf").read_text()
    trainer_path = ROOT / "scripts/training/t5gemma2_typed_fold_gold_replay_v1.py"
    inference_path = (
        ROOT / "scripts/evaluation/t5gemma2_typed_fold_gold_replay_inference_v1.py"
    )
    audit_path = (
        ROOT / "scripts/evaluation/audit_t5gemma2_typed_fold_gold_replay_promotion.py"
    )
    train_path = ROOT / "deploy/vast/t5gemma2_typed_fold_gold_replay_v1.sh"
    evaluation_path = (
        ROOT / "deploy/vast/t5gemma2_typed_fold_gold_replay_v1_eval_gate.sh"
    )
    assert "--gold_replay_rows 458" in train
    assert "--gradient_accumulation 16" in train
    assert "--checkpoint_interval 20" in train
    assert "f1accbf1db6ab326583b8bdc789250c021db34028690b8bab6d014b69437ac05" in train
    assert "aa8fb9b3ba258a0ee117e8c7f98acb55d92fba2d79ef4b0df7b093d57135dcf6" in train
    assert "8226d0ebd55476088d2e2a5cbfb06e573e92539012c4dc4ba551417158e261ed" in train
    assert "c02793912f998dc8c2a85a45a3fcaf5d221561a9bd919256bcf6510bf2caf542" in train
    assert "insufficient free storage" in train and "automatic deletion" in train
    assert (
        sha256_file(trainer_path) in train and sha256_file(trainer_path) in evaluation
    )
    assert sha256_file(inference_path) in evaluation
    assert sha256_file(audit_path) in evaluation
    assert "typed_fold_gold_replay_c2_seed42_k10" in evaluation
    assert "arm-c-score" in evaluation and "arm-b-score" in evaluation
    assert "Arm B supervisor must be EXITED" in handoff
    assert '"${TRAIN_LAUNCHER}"' in handoff and '"${EVAL_LAUNCHER}"' in handoff
    assert "cleanup is never automatic" in handoff
    assert sha256_file(train_path) in handoff
    assert sha256_file(evaluation_path) in handoff
    assert "autostart=false" in conf and "autorestart=false" in conf
    assert "exitcodes=0,78" in conf
    assert "\nrm " not in handoff and "\nrm " not in train
