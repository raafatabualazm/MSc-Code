from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_sft.sh"
SUPERVISOR = ROOT / "deploy" / "vast" / "t5gemma2-sft-2epoch.conf"


def test_sft_launcher_supports_a_separate_epoch_ablation() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'SFT_EPOCHS="${T5GEMMA_SFT_EPOCHS:-1}"' in text
    assert 'SFT_LEARNING_RATE="${T5GEMMA_SFT_LEARNING_RATE:-2e-4}"' in text
    assert '--epochs "${SFT_EPOCHS}"' in text
    assert '--learning_rate "${SFT_LEARNING_RATE}"' in text


def test_two_epoch_arm_preserves_the_original_output() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-sft-2epoch]" in text
    assert 'T5GEMMA_SFT_EPOCHS="2"' in text
    assert 'T5GEMMA_SFT_LEARNING_RATE="2e-4"' in text
    assert (
        'T5GEMMA_SFT_OUTPUT_DIR="/workspace/artifacts/'
        't5gemma2_4b4b_enriched_sft_2epoch_v1"'
    ) in text
    assert "t5gemma2_4b4b_enriched_sft_v1" not in text
    assert "autostart=false" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
