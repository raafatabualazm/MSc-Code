from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT / "scripts" / "training" / "t5gemma2_local_rs_sft_mixed_compat.py"
)
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_local_stage2_kimi_probe.sh"
SUPERVISOR = (
    ROOT / "deploy" / "vast" / "t5gemma2-local-stage2-kimi-probe.conf"
)


def test_wrapper_patches_both_checkpoint_loader_references_only() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "pilot._checkpoint_record = _mixed_checkpoint_record" in text
    assert "pilot.CHECKPOINT_LOADER_COMPATIBILITY = compatibility" in text
    assert "inference._checkpoint_record = _mixed_checkpoint_record" in text
    assert '"scope": "checkpoint_contract_loader_only"' in text
    assert '"mixed_loader_sha256": sha256_file(mixed_loader_path)' in text
    assert '"sampling_code_changed": False' in text
    assert '"scoring_code_changed": False' in text
    assert '"heldout_175_opened": False' in text


def test_probe_is_train_only_four_sample_no_repair_harvest() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = (
        "--pilot_tasks 100",
        "--pilot_offset 0",
        "--base_samples 4",
        "--repair_samples 0",
        "--max_repair_parents 0",
        "--gold_replay_ratio 0",
        "--seed 20260730",
        "verpo_rollout_feedback.jsonl",
        "verpo_teacher_f2.jsonl",
        "reward_holdback.private.jsonl",
        "checkpoint-optstep-000426",
    )
    for value in required:
        assert value in text
    assert "dev_multifunction_binary" not in text
    assert "mixed_rs_sft_passk" not in text
    assert "OPENAI_API_KEY" not in text
    assert "OPENROUTER_API_KEY" not in text


def test_supervisor_is_manual_and_owns_process_group() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-local-stage2-kimi-probe]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
