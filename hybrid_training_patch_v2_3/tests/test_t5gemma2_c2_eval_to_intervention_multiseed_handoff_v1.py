from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = (
    ROOT
    / "deploy/vast/t5gemma2_c2_eval_to_intervention_multiseed_handoff_v1.sh"
)
CONF = (
    ROOT
    / "deploy/vast/t5gemma2-c2-eval-to-intervention-multiseed-handoff-v1.conf"
)


def test_handoff_requires_exact_exited_sealed_eval_before_start() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "t5gemma2-typed-c2-verpo-matched-eval8192-v1" in text
    assert "t5gemma2-measurement-intervention-multiseed-v1" in text
    assert "RUNNING|STARTING" in text and "EXITED) break" in text
    assert "STOPPED|FATAL|BACKOFF|UNKNOWN" in text
    assert text.index("EXITED) break") < text.index('[[ -s "${REPORT}" ]]')
    assert text.index('"${PYTHON_BIN}" "${AUDITOR}"') < text.index(
        '"${SUPERVISORCTL}" start "${NEXT_PROGRAM}"'
    )
    for fragment in (
        '.contract.seeds == [42,43,44]',
        '.contract.sampling.max_new_tokens == 8192',
        '.checks.all_generation_and_scoring_hash_chains_validated == true',
        '.decision.status == "STOP_AFTER_MATCHED_EVALUATION"',
        '.decision.automatic_promotion_performed == false',
        '.decision.promoted_checkpoint == null',
        '.decision.promotion_permitted_from_this_report == false',
    ):
        assert fragment in text


def test_handoff_requires_storage_and_gpu_empty_without_mutation() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "MIN_FREE_KIB" in text and "5242880" in text
    assert "nvidia-smi --query-compute-apps=pid" in text
    assert text.index('[[ -z "${gpu_pids}" ]]') < text.index(
        '"${SUPERVISORCTL}" start "${NEXT_PROGRAM}"'
    )
    assert "private_holdback_used_for_selection_or_training == false" in text
    assert "holdback_alignment" not in text
    assert "rm " not in text and "Remove-Item" not in text
    assert "cp " not in text and "mv " not in text


def test_handoff_supervisor_is_manual_and_fail_closed() -> None:
    text = CONF.read_text(encoding="utf-8")
    assert "[program:t5gemma2-c2-eval-to-intervention-multiseed-handoff-v1]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text and "killasgroup=true" in text


def test_handoff_seals_installed_next_launcher_and_config() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "INSTALLED_NEXT_LAUNCHER" in text
    assert 'cmp -s "${NEXT_LAUNCHER}" "${INSTALLED_NEXT_LAUNCHER}"' in text
    assert 'cmp -s "${NEXT_CONF}" "${INSTALLED_NEXT_CONF}"' in text
    assert "fcc1bcc088c6b2e59bdfb29eb195c08af830a4c99a7652e297c4f44a9b0b0453" in text
    assert "14c5888e2646219dd1771933b0e4dc90c6cec18ca08d8879667ae73d16eb7577" in text


def test_handoff_seals_installed_upstream_launcher_and_config() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "INSTALLED_EVAL_LAUNCHER" in text
    assert "INSTALLED_EVAL_CONF" in text
    assert "a6eb165a6b641fcae21a0f1d3e64a8bba7858ea9ad890e2df05dfe7051acccaa" in text
    assert "8c5205349e885a9218a55d9ca1e959f313dc38ba9514695fa8aea9f13e1bb0c5" in text
