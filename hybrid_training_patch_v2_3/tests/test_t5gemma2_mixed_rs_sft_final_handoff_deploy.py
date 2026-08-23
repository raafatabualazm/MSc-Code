from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "deploy" / "vast" / "t5gemma2_mixed_rs_sft_final_handoff.sh"
CONF = ROOT / "deploy" / "vast" / "t5gemma2-mixed-rs-sft-final-handoff.conf"


def test_handoff_waits_without_treating_clean_exited_as_failure() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "t5gemma-base-harvest-residual-chain-2epoch" in text
    assert 'supervisorctl status "${RESIDUAL_SERVICE}" 2>/dev/null || true' in text
    assert "RUNNING|STARTING" in text
    assert "EXITED|STOPPED" in text
    assert "unexpected ${RESIDUAL_SERVICE}" in text


def test_handoff_late_binds_only_sealed_residual_hashes() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert 'residual_report_sha="$(sha256sum "${RESIDUAL_REPORT}"' in text
    assert 'residual_journal_sha="$(sha256sum "${RESIDUAL_JOURNAL}"' in text
    assert '"${recorded_journal_sha}" != "${residual_journal_sha}"' in text
    assert '.pilot.tasks == 1500' in text
    assert ".contract.schedule.pilot_tasks == 1500" in text
    assert ".contract.sampling.base_samples == 4" in text
    assert ".contract.sampling.repair_samples == 0" in text
    assert 'select(.event == "complete"' in text


def test_handoff_pins_every_production_report_and_journal() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    expected = {
        "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab":
            "t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json",
        "8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50":
            "t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest_report.json",
        "883aeeab6d6a62d4bab41c62f77c8578866cf1a3b9355c4fd74e1de06c048cae":
            "t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest_report.json",
        "fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad":
            "t5gemma2_api_rs_sft_claude_production_2epoch_v1/api_rescue_report.json",
        "99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4":
            "t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1/api_rescue_report.json",
        "f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b":
            "t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue_report.json",
        "fa0c70c73767a525f2ca710fd822cb2bdca60140f133696ad15b87e71d2751d1":
            "t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1/api_rescue_report.json",
        "5c610a4073122e209e26af8e689a683258405c00e58a23c6e9a109c76f9c4c6c":
            "t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1/api_rescue.journal.jsonl",
        "336874a72569f6a82bbc844260b772e7f3dc631c399e23c979d54502713ea727":
            "t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue_report.json",
        "33bf539f37beb285459511ee5349f8eec34b8335ff4c07339ce8a95467379cf0":
            "t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue.journal.jsonl",
        "06af6f49ea45d485e6c61b0e4a8b783894ffb4a1491235c56fb2c0428cf0e683":
            "t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue.journal.jsonl.chain-head.json",
        "aa22e905037222a34eb01964eb2f6b6a9826ffbb19376490ff1c130a2d8bf18b":
            "t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/direct_hard_targets.jsonl",
        "a8c9bc693a27d46c5d83d7b2beb4dddcdae6e1d46d64916d163688de3a3ba557":
            "t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/direct_hard_targets_f2.jsonl",
        "77cae6c03ca0dd1e80e303afedf2fb551fd1e8ea7ceee0844ecf8448877b423e":
            "t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/repair_policy_sources.jsonl",
        "903fd33974f37fb6144267eac84e39f7d5d8ffcf437bf96db79920fd1f9b6924":
            "t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/repair_policy_targets.jsonl",
    }
    for digest, path in expected.items():
        assert digest in text
        assert path in text
    assert "claude_probe_prefix10" not in text


def test_handoff_enforces_final_training_contract() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "T5GEMMA_MIXED_MIN_DIRECT_TARGETS=200" in text
    assert "T5GEMMA_MIXED_MIN_REPAIR_TARGETS=71" in text
    assert "T5GEMMA_MIXED_GOLD_REPLAY_RATIO=3" in text
    assert "T5GEMMA_MIXED_EPOCHS=3" in text
    assert "T5GEMMA_MIXED_LEARNING_RATE=5e-5" in text
    assert "t5gemma2_4b4b_mixed_rs_sft_final_v1" in text
    assert 'exec "${MIXED_LAUNCHER}"' in text


def test_handoff_supervisor_is_manual_and_exit78_is_expected() -> None:
    text = CONF.read_text(encoding="utf-8")
    assert "[program:t5gemma-mixed-rs-sft-final-handoff]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
