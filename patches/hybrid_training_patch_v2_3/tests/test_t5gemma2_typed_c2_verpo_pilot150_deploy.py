import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_typed_c2_verpo_pilot150_v1.sh"
CONFIG = ROOT / "deploy" / "vast" / "t5gemma2-typed-c2-verpo-pilot150-v1.conf"


def test_launcher_pins_parent_go_gate_and_preregistration() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for value in (
        "checkpoint-optstep-000058",
        "a3d325af70ac9cd0a0c55cb7e66f4df2b390f78fab3ca6a70a930093ac989a00",
        "80b50fab88e076d3e14771d09b5d1706baffeb2fd6c0c9d51b8841dd4135a004",
        "2bd9aa7a0c4ce5740e670a7ab7a702a6522790f4dc60ec39355a4a7647f13117",
        "bc515b7dd7efb4d2458da3af407028eca572cf2a7a1af6616e0a8f8797c134a9",
        "5bfbe6359d0b84ecabe43542eaebadd77bde0f9e959a210682ebcfa453445c80",
    ):
        assert value in text
    assert 'holdback.get("decision") == "GO"' in text
    assert 'aggregate_only") is True' in text


def test_launcher_materializes_target_free_view_before_gpu_training() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    builder = text.index('"${PYTHON_BIN}" "${VIEW_BUILDER}"')
    trainer = text.index('"${PYTHON_BIN}" "${TRAINER}" "${args[@]}"')
    assert builder < trainer
    assert "task_view.public.jsonl" in text
    assert "--typed_task_view" in text
    assert "--typed_task_view_manifest" in text
    assert "--proxy_audit_journal" in text


def test_launcher_freezes_pure_reward_150_update_profile() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "--max_updates 150",
        "--tasks_per_update 1",
        "--group_size 4",
        "--repair_group_size 4",
        "--max_repair_parents 2",
        "--max_new_tokens 8192",
        "--max_source_tokens 32768",
        "--verpo_alpha 2.0",
        "--local_weight 1.0",
        "--compile_weight 0.25",
        "--learning_rate 1e-6",
        "--ppo_clip 0.0",
        "--sft_replay_weight 0.0",
        "--checkpoint_interval 1",
        "--keep_last_checkpoints 2",
        "--seed 42",
    ):
        assert fragment in text
    assert "--sft_replay_weight 0.02" not in text


def test_launcher_forces_phase16_process_boundary_and_exact_resume() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "phase16-go:",
        "resume-pre16:",
        "resume-continue:",
        "--continue_after_gate16",
        "--resume_checkpoint",
        "mandatory update-16 GO boundary",
        "checkpoint optimizer/RNG state differs",
        "metrics/checkpoint update mismatch",
        "differs from recomputation",
        "pilot.evaluate_mechanics_gate",
    ):
        assert fragment in text


def test_launcher_never_performs_storage_cleanup_or_promotion() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "no launcher cleanup is permitted" in text
    assert "automatic_promotion_performed" in text
    for destructive in ("rm -", "rm --", "find ${", "Remove-Item"):
        assert destructive not in text


def test_embedded_python_blocks_parse() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    scripts = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY", text, flags=re.DOTALL)
    assert len(scripts) == 3
    for script in scripts:
        ast.parse(script)


def test_supervisor_is_manual_restart_safe_and_group_terminating() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "[program:t5gemma2-typed-c2-verpo-pilot150-v1]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "startretries=1" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
    assert "stopwaitsecs=180" in text
