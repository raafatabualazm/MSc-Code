from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_typed_dual_to_pass2_handoff.sh"
CONF = ROOT / "deploy" / "vast" / "t5gemma2-typed-dual-to-pass2-handoff.conf"


def test_handoff_waits_for_successful_dual_exit_and_stable_artifacts() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'RUNNING|STARTING)' in text
    assert 'EXITED)' in text
    assert 'STOPPED)' in text
    assert 'exit 78' in text
    assert 'snapshot_one="$(sha256sum "${sealed_files[@]}")"' in text
    assert 'snapshot_two="$(sha256sum "${sealed_files[@]}")"' in text
    assert '.schema == "t5gemma2-typed-dual-api-orchestration-report-v1"' in text
    assert '.status == "complete"' in text
    assert 'exec "${PASS2_LAUNCHER}"' in text


def test_handoff_pins_pass2_and_is_not_autostarted() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert (
        "b47b1330c08a3581b82cbc05ba98cd8048b02e5661b4ea57ac2126293ab73d43"
        in text
    )
    conf = CONF.read_text(encoding="utf-8")
    assert "[program:t5gemma2-typed-dual-to-pass2-handoff]" in conf
    assert "autostart=false" in conf
    assert "autorestart=unexpected" in conf
    assert "exitcodes=0,78" in conf
