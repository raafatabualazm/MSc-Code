from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_sft_2epoch_eval.sh"
SUPERVISOR = ROOT / "deploy" / "vast" / "t5gemma2-sft-2epoch-eval.conf"


def test_handoff_only_observes_the_trainer_and_requires_exact_final_state() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'SUPERVISORCTL="${T5GEMMA_2E_SUPERVISORCTL:-/usr/local/bin/supervisorctl}"' in text
    assert '"${SUPERVISORCTL}" status "${TRAIN_PROGRAM}"' in text
    assert '[[ ! -x "${SUPERVISORCTL}" ]]' in text
    assert "status_rc=$?" in text
    assert "empty Supervisor response" in text
    assert '"${status_line}" )" ||' not in text
    assert "supervisorctl start" not in text
    assert "supervisorctl stop" not in text
    assert "supervisorctl restart" not in text
    assert ".updates == 348" in text
    assert '.latest_checkpoint == "checkpoint-optstep-000348"' in text
    assert 'train_state}" == EXITED' in text
    assert "T5GEMMA_2E_EVAL_MAX_WAIT_SECONDS" in text


def test_epoch_arms_use_the_same_sealed_175_task_decoding_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "--num_samples 10",
        "--generation_batch_size 10",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--temperature 0.8",
        "--top_p 0.95",
        "--seed 42",
        "--k 10",
        "--timeout 30",
        "--stability_runs 2",
    ):
        assert fragment in text
    assert "dev_multifunction_binary.seal.json" in text
    assert "dev_multifunction_binary_f2.jsonl.manifest.json" in text
    assert "no_frontier_api" in text
    assert "tests_exposed_to_model" in text
    assert "same_sampling_and_slot_seeds" in text
    assert "slot_coordinates[0] == slot_coordinates[1] == slot_coordinates[2]" in text


def test_completed_reference_arms_are_reused_and_two_epoch_is_isolated() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "t5gemma2_prepost_passk_v1" in text
    assert "pre_base_k10_predictions.json" in text
    assert "post_sft_k10_predictions.json" in text
    assert "t5gemma2_sft_epoch_ablation_passk_v1" in text
    assert "two_epoch_k10_predictions.json" in text
    assert "require_exact_or_write(output, report)" in text


def test_supervisor_job_is_manual_and_fail_closed() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-sft-2epoch-eval]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
    assert 'T5GEMMA_2E_TRAIN_PROGRAM="t5gemma-sft-2epoch"' in text
