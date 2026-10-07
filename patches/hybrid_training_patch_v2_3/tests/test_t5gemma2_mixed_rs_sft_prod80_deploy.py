from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "deploy" / "vast" / "t5gemma2-mixed-rs-sft-prod80.conf"


def test_prod80_stage_pins_completed_non_exploratory_reports() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    expected = {
        "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab":
            "t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json",
        "8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50":
            "t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest_report.json",
        "fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad":
            "t5gemma2_api_rs_sft_claude_production_2epoch_v1/api_rescue_report.json",
    }
    for digest, path_fragment in expected.items():
        assert digest in text
        assert path_fragment in text
    assert 'T5GEMMA_MIXED_ALLOW_EXPLORATORY_INPUTS="0"' in text
    assert 'T5GEMMA_MIXED_OUTPUT_DIR="/workspace/artifacts/t5gemma2_4b4b_mixed_rs_sft_prod80_v1"' in text


def test_prod80_supervisor_is_manual_and_process_group_owned() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "[program:t5gemma-mixed-rs-sft-prod80]" in text
    assert "command=/opt/supervisor-scripts/t5gemma2_mixed_rs_sft.sh" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
