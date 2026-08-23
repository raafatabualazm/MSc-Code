from __future__ import annotations

import hashlib
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parents[1]


def test_handoff_pins_both_sides_and_waits_for_sealed_report() -> None:
    launcher_path = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2_measurement_intervention_multiseed_v1.sh"
    )
    reporter_path = (
        PATCH_ROOT
        / "scripts"
        / "evaluation"
        / "t5gemma2_measurement_intervention_multiseed_report_v1.py"
    )
    checker_path = (
        PATCH_ROOT
        / "scripts"
        / "evaluation"
        / "verify_t5gemma2_measurement_runtime_compat_v1.py"
    )
    downstream_config = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2-measurement-intervention-multiseed-v1.conf"
    )
    upstream_launcher = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh"
    )
    upstream_config = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2-typed-c2-verpo-matched-eval8192-v1.conf"
    )
    handoff = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2_measurement_intervention_after_verpo_handoff_v1.sh"
    ).read_text(encoding="utf-8")
    for path in (
        launcher_path,
        reporter_path,
        checker_path,
        downstream_config,
        upstream_launcher,
        upstream_config,
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() in handoff
    assert "t5gemma2-typed-c2-verpo-matched-eval8192-v1" in handoff
    assert "t5gemma2-measurement-intervention-multiseed-v1" in handoff
    assert 'RUNNING|STARTING|STOPPING)' in handoff
    assert 'STOPPED|FATAL|BACKOFF|UNKNOWN)' in handoff
    assert "upstream exited without its sealed three-seed report" in handoff
    assert 'and .contract.seeds == [42,43,44]' in handoff
    assert 'and .decision.status == "STOP_AFTER_MATCHED_EVALUATION"' in handoff
    assert '"${SUPERVISORCTL}" start "${DOWNSTREAM_PROGRAM}"' in handoff
    assert "downstream previously exited without its final report" in handoff


def test_handoff_supervisor_is_explicitly_armed_only() -> None:
    config = (
        PATCH_ROOT
        / "deploy"
        / "vast"
        / "t5gemma2-measurement-intervention-after-verpo-handoff-v1.conf"
    ).read_text(encoding="utf-8")
    assert "autostart=false" in config
    assert "autorestart=false" in config
    assert "startretries=0" in config
    assert "stopasgroup=true" in config
    assert "killasgroup=true" in config
