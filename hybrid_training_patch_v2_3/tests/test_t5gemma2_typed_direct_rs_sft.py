from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.training import t5gemma2_enriched_sft as base
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_direct_rs_sft as profile


LOCAL_COUNTS = [9, 48, 49, 35]
API_COUNTS = [23, 42, 3, 2, 1, 10, 3]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _historical_pair(task_id: str, *, category: str) -> mixed.MixedPair:
    target = f"int fn0() => {int(task_id.split('-')[-1]) + 1000};"
    return mixed.MixedPair(
        pair_id=f"{task_id}::{category}",
        source_task_id=task_id,
        kind="verified_direct",
        source=f"UNTYPED::{task_id}",
        target=target,
        source_sha256=_sha(f"UNTYPED::{task_id}"),
        target_sha256=_sha(target),
        provenance=(("view", "direct"),),
    )


def _patch_corpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    overlap: bool = False,
) -> tuple[Path, list[tuple[Path, str]], list[tuple[Path, str]]]:
    task_ids = [f"task-{index}" for index in range(225)]
    typed_pairs = [
        base.TextPair(
            task_id=task_id,
            source=f"TYPED::{task_id}",
            target=f"int fn0() => {index};",
            source_sha256=_sha(f"TYPED::{task_id}"),
            target_sha256=_sha(f"int fn0() => {index};"),
        )
        for index, task_id in enumerate(task_ids)
    ]
    typed_manifest = {
        "schema": "synthetic-typed",
        "rows": 225,
        "training_exclusions": {
            "count": 1,
            "task_ids": [profile.CONTAMINATED_TRAIN_TASK_ID],
        },
    }
    monkeypatch.setattr(
        profile.typed_sft,
        "load_typed_text_pairs",
        lambda *args, **kwargs: (typed_pairs, typed_manifest),
    )

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "dart_source": f"int fn0() => {index};",
                    "acceptance_tests": f"FULL_TEST::{task_id}",
                },
                separators=(",", ":"),
            )
            + "\n"
            for index, task_id in enumerate(task_ids)
        ),
        encoding="utf-8",
    )

    local_specs = [(tmp_path / f"local-{index}.json", f"l{index}") for index in range(4)]
    api_specs = [(tmp_path / f"api-{index}.json", f"a{index}") for index in range(7)]
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for path, _digest in local_specs:
        index = int(path.stem.split("-")[-1])
        offsets[str(path)] = (cursor, LOCAL_COUNTS[index])
        cursor += LOCAL_COUNTS[index]
    for path, _digest in api_specs:
        index = int(path.stem.split("-")[-1])
        offsets[str(path)] = (cursor, API_COUNTS[index])
        cursor += API_COUNTS[index]
    assert cursor == 225

    def fake_local(*, report_path: Path, report_sha256: str, warmstart):
        del report_sha256, warmstart
        start, count = offsets[str(report_path)]
        pairs = [
            _historical_pair(task_ids[position], category="local")
            for position in range(start, start + count)
        ]
        return pairs, {
            "schema": mixed.LOCAL_REPORT_SCHEMA,
            "accepted_direct_rows": count,
            "private_gate_bound": True,
            "warmstart_bound": True,
        }

    def fake_api(*, report_path: Path, report_sha256: str, allow_exploratory: bool):
        del report_sha256
        assert allow_exploratory is False
        start, count = offsets[str(report_path)]
        ids = task_ids[start : start + count]
        if overlap and report_path == api_specs[0][0]:
            ids[0] = task_ids[0]
        direct = [_historical_pair(task_id, category="api") for task_id in ids]
        conditioned = [
            mixed.MixedPair(
                pair_id=f"{task_id}::conditioned",
                source_task_id=task_id,
                kind="repair_conditioned",
                source=f"REPAIR::{task_id}",
                target=pair.target,
                source_sha256=_sha(f"REPAIR::{task_id}"),
                target_sha256=pair.target_sha256,
                provenance=(("view", "repair"),),
            )
            for task_id, pair in zip(ids, direct, strict=True)
        ]
        return direct + conditioned, {
            "schema": mixed.API_REPORT_SCHEMA,
            "direct_rows": count,
            "repair_conditioned_rows": count,
            "private_gate_bound": True,
        }

    monkeypatch.setattr(profile.mixed, "_validate_local_report", fake_local)
    monkeypatch.setattr(profile.mixed, "_validate_api_report", fake_api)
    return train_path, local_specs, api_specs


def _build(
    train_path: Path,
    local_specs: list[tuple[Path, str]],
    api_specs: list[tuple[Path, str]],
    *,
    verify=lambda code, tests, slot: True,
):
    return profile.build_typed_direct_pairs(
        gold_train_jsonl=train_path,
        gold_f2_jsonl=Path("unused-f2"),
        expected_gold_train_sha256="0" * 64,
        expected_gold_f2_sha256="1" * 64,
        expected_gold_rows=2776,
        heldout_jsonl=Path("unused-heldout"),
        expected_heldout_sha256="2" * 64,
        expected_heldout_rows=175,
        local_reports=local_specs,
        api_reports=api_specs,
        warmstart=profile._source_sft_identity(),
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=225,
        min_repair_conditioned_targets=0,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        seed=42,
        verify=verify,
        verification_workers=1,
    )


def test_direct_profile_rebinds_exactly_225_without_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train, local_specs, api_specs = _patch_corpus(monkeypatch, tmp_path)
    seen: list[tuple[str, str]] = []

    def verify(code: str, tests: str, slot: str) -> bool:
        assert code.startswith("int fn0()")
        assert tests.startswith("FULL_TEST::task-")
        seen.append((tests, slot))
        return True

    pairs, manifest = _build(train, local_specs, api_specs, verify=verify)
    assert len(pairs) == 225
    assert len({pair.source_task_id for pair in pairs}) == 225
    assert all(pair.source.startswith("TYPED::") for pair in pairs)
    assert all(pair.kind == "verified_direct" for pair in pairs)
    assert len(seen) == 225
    assert manifest["schema"] == profile.DATASET_SCHEMA
    assert manifest["composition"]["local_student_direct"] == 141
    assert manifest["composition"]["external_teacher_direct"] == 84
    assert manifest["composition"]["repair_conditioned"] == 0
    assert manifest["composition"]["gold_replay"] == 0
    assert manifest["full_acceptance_reverification"]["passed"] == 225
    assert manifest["tests_model_visible"] is False
    assert "FULL_TEST::task" not in json.dumps(manifest)


def test_overlapping_ledgers_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train, local_specs, api_specs = _patch_corpus(
        monkeypatch, tmp_path, overlap=True
    )
    with pytest.raises(ValueError, match="overlap by source task"):
        _build(train, local_specs, api_specs)


def test_full_acceptance_failure_rejects_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train, local_specs, api_specs = _patch_corpus(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="failed complete acceptance"):
        _build(
            train,
            local_specs,
            api_specs,
            verify=lambda code, tests, slot: tests != "FULL_TEST::task-17",
        )


def test_profile_hyperparameters_are_fail_closed() -> None:
    args = argparse.Namespace(
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=225,
        min_repair_conditioned_targets=0,
        epochs=2,
        batch_size=1,
        gradient_accumulation=8,
        max_updates=0,
        learning_rate=2e-5,
        warmup_ratio=0.0,
        seed=42,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        local_report=["x"] * 4,
        api_report=["x"] * 7,
    )
    profile._validate_profile_args(args)
    assert base.calculate_training_schedule(
        rows=225,
        epochs=2,
        batch_size=1,
        gradient_accumulation=8,
        max_updates=0,
        warmup_ratio=0.0,
    ) == {
        "microbatches_per_epoch": 225,
        "updates_per_epoch": 29,
        "available_updates": 58,
        "planned_updates": 58,
        "warmup_updates": 0,
    }
    args.gold_replay_rows = 1
    with pytest.raises(ValueError, match="gold_replay_rows"):
        profile._validate_profile_args(args)


def test_launcher_pins_all_reports_and_direct_only_recipe() -> None:
    launcher = (
        PATCH_ROOT / "deploy" / "vast" / "t5gemma2_typed_direct_rs_sft.sh"
    ).read_text(encoding="utf-8")
    config = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2-typed-direct-rs-sft-225.conf"
    ).read_text(encoding="utf-8")
    assert launcher.count("--local_report") == 1
    assert launcher.count("--api_report") == 1
    assert "LOCAL_REPORTS=(" in launcher and "API_REPORTS=(" in launcher
    assert "kimi_k3_mixed_paired50_v12" in launcher
    assert "kimi_k3_retry17_8k_v1" in launcher
    assert "--gold_replay_rows 0" in launcher
    assert "--min_verified_direct_targets 225" in launcher
    assert "--min_repair_conditioned_targets 0" in launcher
    assert "--epochs 2" in launcher
    assert "--gradient_accumulation 8" in launcher
    assert "--learning_rate 2e-5" in launcher
    assert "--warmup_ratio 0" in launcher
    assert "nonempty foreign/incomplete output" in launcher
    assert "latest_checkpoint.json" in launcher
    assert "3cb25d54f12743ed43572b219e119667f264abab94ec4cbfac72a94407fbdfc7" in launcher
    assert "[program:t5gemma2-typed-direct-rs-sft-225]" in config
    assert profile.RUN_SCHEMA in inference.SUPPORTED_ADAPTER_RUN_SCHEMAS

    evaluation = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2_typed_direct_rs_sft_eval.sh"
    ).read_text(encoding="utf-8")
    assert "checkpoint-optstep-000058" in evaluation
    assert "--input_view typed_opaque_contract" in evaluation
    assert "typed_direct_rs_sft_seed42_k10_score_full175.json" in evaluation
    assert "typed_direct_rs_sft_seed42_k10_score_clean174.json" in evaluation
    assert "--exclude_task_id sigless_8bf7f40ca356" in evaluation
