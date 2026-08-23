from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_rs_sft_pilot.sh"
SUPERVISOR = ROOT / "deploy" / "vast" / "t5gemma2-rs-sft-pilot.conf"
TWO_EPOCH_SUPERVISOR = (
    ROOT / "deploy" / "vast" / "t5gemma2-rs-sft-pilot-2epoch.conf"
)


def test_launcher_pins_actual_training_feedback_and_final_sft_checkpoint() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    expected = {
        "14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c":
            "verpo_rollout_feedback.jsonl",
        "c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3":
            "verpo_teacher_f2.jsonl",
        "dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f":
            "reward_holdback.private.jsonl",
    }
    for digest, filename in expected.items():
        assert digest in text
        assert filename in text
    assert "checkpoint-optstep-000174" in text
    assert "T5GEMMA_SFT_CHECKPOINT_NAME" in text
    assert ".latest_checkpoint // empty" in text
    assert ".no_frontier_api // false" in text
    assert "--allow_unpinned_inputs" not in text
    assert "| sha256sum -c -" in text


def test_launcher_is_a_200_task_local_only_resumable_pilot() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required_arguments = (
        "--pilot_tasks 200",
        "--base_samples 4",
        "--repair_samples 4",
        "--max_repair_parents 2",
        "--gold_replay_ratio 3",
        "--production_min_unique_targets 200",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--generation_batch_size 4",
        "--stability_runs 2",
    )
    for argument in required_arguments:
        assert argument in text
    assert "post_sft_k10_score.json" in text
    assert "T5GEMMA_RS_SFT_READY_SCORE" in text
    assert "harvest_report.json" in text
    assert "T5GEMMA_RS_SFT_PILOT_ALREADY_COMPLETE" in text
    assert "OPENAI" not in text
    assert "ANTHROPIC" not in text
    assert "DASHSCOPE" not in text
    assert "curl " not in text


def test_launcher_exports_dart_path_and_supervisor_owns_process_group() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    assert 'DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"' in launcher
    assert 'export PATH="$(dirname "${DART_BIN}"):${PATH}"' in launcher
    assert "export DART_BIN" in launcher
    assert "[program:t5gemma-rs-sft-pilot]" in supervisor
    assert "command=/opt/supervisor-scripts/t5gemma2_rs_sft_pilot.sh" in supervisor
    assert "autostart=false" in supervisor
    assert "autorestart=unexpected" in supervisor
    assert "exitcodes=0,78" in supervisor
    assert "stopasgroup=true" in supervisor
    assert "killasgroup=true" in supervisor
    assert "stdout_logfile=/workspace/logs/t5gemma-rs-sft-pilot.log" in supervisor


def test_two_epoch_pilot_is_isolated_and_uses_the_completed_ablation_arm() -> None:
    text = TWO_EPOCH_SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-rs-sft-pilot-2epoch]" in text
    assert 'T5GEMMA_SFT_CHECKPOINT_NAME="checkpoint-optstep-000348"' in text
    assert (
        'T5GEMMA_SFT_OUTPUT_DIR="/workspace/artifacts/'
        't5gemma2_4b4b_enriched_sft_2epoch_v1"'
    ) in text
    assert (
        'T5GEMMA_RS_SFT_READY_SCORE="/workspace/artifacts/'
        't5gemma2_sft_epoch_ablation_passk_v1/two_epoch_k10_score.json"'
    ) in text
    assert (
        'T5GEMMA_RS_SFT_PILOT_OUTPUT_DIR="/workspace/artifacts/'
        't5gemma2_local_rs_sft_pilot_2epoch_v1"'
    ) in text
    assert "autostart=false" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
