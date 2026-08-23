import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_verpo_pilot16_2epoch.sh"
CONFIG = ROOT / "deploy" / "vast" / "t5gemma2-verpo-pilot16-2epoch.conf"


def test_pilot_uses_exact_promoted_two_epoch_checkpoint() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "enriched_sft_2epoch_v1/checkpoint-optstep-000348" in text
    assert "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3" in text
    assert "83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc" in text
    assert "c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3" in text
    assert "tokenization\", {}).get(\"truncated_rows\") != 0" in text
    assert "source_tokens\", {}).get(\"max\") != 13253" in text
    assert "target_tokens\", {}).get(\"max\") != 3343" in text


def test_pilot_is_fresh_bounded_sixteen_update_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "compiler_verpo_pilot16_2epoch_v1" in text
    for fragment in (
        "MAX_UPDATES=16",
        "--group_size 4",
        "--repair_group_size 4",
        "--max_repair_parents 2",
        "--tasks_per_update 1",
        '--max_updates "${MAX_UPDATES}"',
        "--temperature 0.8",
        "--max_new_tokens 4096",
        "--max_source_tokens 32768",
        "--max_target_tokens 32768",
        "--verpo_alpha 2.0",
        "--local_weight 1.0",
        "--compile_weight 0.25",
        "--learning_rate 1e-6",
        "--ppo_clip 0.0",
        "--sft_replay_weight 0.02",
        "--on_policy_logprob_tolerance 2e-4",
        "--checkpoint_interval 1",
        "--keep_last_checkpoints 2",
        "--seed 42",
    ):
        assert fragment in text
    assert "--max_new_tokens 32767" not in text
    assert "T5GEMMA_SFT_OUTPUT_DIR" not in text


def test_pilot_fails_closed_on_runtime_or_resume_discontinuity() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "runtime source differs",
        "runtime code bundle differs",
        'sampling.get("suppressed_token_ids") == [0]',
        'sampling.get("sampling_support_constraint_exactly_recomputed") is True',
        'sampling.get("pad_before_eos_fail_closed") is True',
        "metrics/checkpoint update mismatch",
        "row.get(\"run_contract_sha256\") != contract_sha256",
        "row.get(\"update\") != index",
        'item["actions"][-1] != 1',
        "checkpoint.parent != output_dir",
        "checkpoint contract differs",
        "final checkpoint has no completed result",
        "validate_pilot_state",
        "post-run validation failed",
        "validation_status=$?",
        'exit 78',
    ):
        assert fragment in text


def test_embedded_python_preflights_parse() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    scripts = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
    assert len(scripts) == 2
    for script in scripts:
        ast.parse(script)


def test_supervisor_job_is_manual_and_restart_safe() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "[program:t5gemma-verpo-pilot16-2epoch]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "startretries=1" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
