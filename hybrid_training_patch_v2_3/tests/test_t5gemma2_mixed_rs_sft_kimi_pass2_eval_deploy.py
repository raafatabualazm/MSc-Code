from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluation.t5gemma2_kimi_pass2_compare import (
    build_historical_regression_block,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "deploy" / "vast" / "t5gemma2_mixed_rs_sft_kimi_pass2_eval.sh"
)
SUPERVISOR = (
    ROOT / "deploy" / "vast" / "t5gemma2-mixed-rs-sft-kimi-pass2-eval.conf"
)


def test_launcher_seals_exact_pass2_training_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "checkpoint-optstep-000013",
        ".updates == 13",
        ".planned_updates == 13",
        ".rows == 104",
        ".optimization.epochs == 1",
        ".optimization.learning_rate == 0.00001",
        ".dataset.composition.verified_direct == 13",
        ".dataset.composition.repair_conditioned == 13",
        ".dataset.composition.gold_replay == 78",
        '.warmstart.checkpoint_name == "checkpoint-optstep-000426"',
        ".warmstart.update == 426",
    ):
        assert fragment in text
    assert "fe9bcd00c6774432b7911129246c8b2837523d85b1c94efb29c03f85ae860205" in text
    assert "fe2941885767f7c4abb3012d1a49c22a934a6b67d8f1f9626bf09e44a3d633d0" in text


def test_launcher_uses_exact_paired_heldout_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "t5gemma2_mixed_rs_sft_passk_v1",
        "post_mixed_k10_predictions.json",
        "t5gemma2_f2_passk_mixed_compat.py",
        "--num_samples 10",
        "--generation_batch_size 10",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--temperature 0.8",
        "--top_p 0.95",
        "--seed 42",
        "--attn_implementation sdpa",
        "--bf16",
        "--k 10",
        "--workers 32",
        "--timeout 30",
        "--stability_runs 2",
    ):
        assert fragment in text


def test_regression_analysis_is_post_score_only_and_never_model_visible() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    inference = text.index("t5gemma2_f2_passk_mixed_compat.py")
    scoring = text.index("score_direct_compact_passk.py")
    regression_analysis = text.index("t5gemma2_kimi_pass2_compare.py")
    assert inference < scoring < regression_analysis
    assert "No historical" in text
    assert "enters inference" in text
    assert "--historical_score" not in text[:inference]
    assert "--pre_score" not in text[:inference]


def test_historical_five_are_derived_and_recovery_is_reported() -> None:
    ids = [f"task-{index}" for index in range(8)]
    historical = {
        task_id: {"pass_at_k": index < 5}
        for index, task_id in enumerate(ids)
    }
    stage1 = {task_id: {"pass_at_k": False} for task_id in ids}
    pass2 = {
        task_id: {"pass_at_k": index in {1, 4}}
        for index, task_id in enumerate(ids)
    }
    report = build_historical_regression_block(
        task_order=ids,
        historical_by_task=historical,
        stage1_by_task=stage1,
        pass2_by_task=pass2,
    )
    assert report["expected_and_observed_tasks"] == 5
    assert report["recovered_tasks"] == 2
    assert report["still_regressed_tasks"] == 3
    assert report["used_as_model_input_or_feedback"] is False
    assert [row["task_id"] for row in report["tasks"]] == ids[:5]


def test_historical_regression_count_drift_fails_closed() -> None:
    ids = [f"task-{index}" for index in range(4)]
    historical = {task_id: {"pass_at_k": True} for task_id in ids}
    stage1 = {task_id: {"pass_at_k": False} for task_id in ids}
    pass2 = {task_id: {"pass_at_k": False} for task_id in ids}
    with pytest.raises(ValueError, match="expected=5, observed=4"):
        build_historical_regression_block(
            task_order=ids,
            historical_by_task=historical,
            stage1_by_task=stage1,
            pass2_by_task=pass2,
        )


def test_supervisor_job_is_manual_and_fail_closed() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-mixed-rs-sft-kimi-pass2-eval]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
