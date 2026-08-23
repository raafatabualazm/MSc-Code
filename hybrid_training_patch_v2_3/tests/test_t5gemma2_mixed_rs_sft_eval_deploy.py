from __future__ import annotations

from pathlib import Path


def test_mixed_eval_is_exact_paired_heldout_evaluation() -> None:
    project = Path(__file__).resolve().parents[1]
    launcher = (
        project / "deploy" / "vast" / "t5gemma2_mixed_rs_sft_eval.sh"
    ).read_text(encoding="utf-8")
    assert "checkpoint-optstep-000426" in launcher
    assert "two_epoch_k10_predictions.json" in launcher
    assert "t5gemma2_f2_passk_mixed_compat.py" in launcher
    assert "checkpoint-loader-compat.json" in launcher
    assert "--num_samples 10" in launcher
    assert "--max_new_tokens 4096" in launcher
    assert "--seed 42" in launcher
    assert "same_sampling_and_slot_seeds" in launcher
    assert "tests_exposed_to_model" in launcher
    assert "targets_exposed_to_model" in launcher
    assert "frontier" in launcher.lower()
    assert "api_rs_sft_rescue" not in launcher
    assert "mixed_rs_sft.py" not in launcher


def test_mixed_eval_supervisor_is_manual_and_fail_closed() -> None:
    project = Path(__file__).resolve().parents[1]
    config = (
        project / "deploy" / "vast" / "t5gemma2-mixed-rs-sft-eval.conf"
    ).read_text(encoding="utf-8")
    assert "autostart=false" in config
    assert "autorestart=unexpected" in config
    assert "exitcodes=0,78" in config
    assert "stopasgroup=true" in config
    assert "killasgroup=true" in config
