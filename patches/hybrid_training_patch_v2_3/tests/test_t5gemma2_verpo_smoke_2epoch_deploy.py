from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_verpo_smoke_2epoch.sh"
CONFIG = ROOT / "deploy" / "vast" / "t5gemma2-verpo-smoke-2epoch.conf"


def test_smoke_uses_promoted_two_epoch_checkpoint_and_exact_identity() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "enriched_sft_2epoch_v1/checkpoint-optstep-000348" in text
    assert "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3" in text
    assert "83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc" in text
    assert "c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3" in text
    assert "len(contract.get(\"lora\", {}).get(\"targets\") or []) != 476" in text


def test_smoke_is_one_update_with_bounded_untruncated_sampling() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "--group_size 4",
        "--repair_group_size 4",
        "--max_repair_parents 2",
        "--tasks_per_update 1",
        "--max_updates 1",
        "--max_new_tokens 4096",
        "--max_source_tokens 32768",
        "--ppo_clip 0.0",
        "--on_policy_logprob_tolerance 2e-4",
        "--checkpoint_interval 1",
    ):
        assert fragment in text
    assert "--max_new_tokens 32767" not in text


def test_supervisor_job_is_manual_and_restart_safe() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "[program:t5gemma-verpo-smoke-2epoch]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
