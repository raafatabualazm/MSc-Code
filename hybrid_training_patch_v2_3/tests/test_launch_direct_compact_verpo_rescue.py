from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import direct_compact_verpo_rescue as rescue
from scripts.training import launch_direct_compact_verpo_rescue as launcher
from models.direct_compact_causal import DirectCompactContract


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_config(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "run"
    paths: dict[str, Path] = {}
    for name in launcher.REQUIRED_INPUTS:
        suffix = ".json" if name in {"base_inference", "base_provenance"} else ".bin"
        paths[name] = _write(tmp_path / "inputs" / f"{name}{suffix}", name)
    contract = DirectCompactContract(
        schema="direct-compact-causal-v1",
        codec_sha256="a" * 64,
        codebook_sha256="b" * 64,
        tokenizer_json_sha256="c" * 64,
        tokenizer_fingerprint_sha256="d" * 64,
        model_config_sha256="e" * 64,
        decoder_model="student",
        decoder_revision="revision",
        target_function="fn0",
        target_language="Dart",
        dfg_extractor_sha256="f" * 64,
        lossless_domain="scrubbed_canonical_graph",
        base_vocab_size=4,
        source_token_ids=(4,),
        source_token_expansions=((4, (2,)),),
    )
    paths["contract"].write_text(
        json.dumps(contract.as_dict()), encoding="utf-8"
    )
    paths["base_inference"].write_text(
        json.dumps(
            [{"task_id": "fit-0", "predictions": ["int fn0() => 0;"]}]
        ),
        encoding="utf-8",
    )
    paths["rollout"].write_text(
        '{"task_id":"fit-0","feedback_tests":"VISIBLE"}\n',
        encoding="utf-8",
    )
    paths["f2"].write_text('{"task_id":"fit-0"}\n', encoding="utf-8")
    paths["f2_manifest"].write_text("{}", encoding="utf-8")
    paths["private_holdback"].write_text(
        '{"task_id":"fit-0","reward_holdback_tests":"PRIVATE"}\n',
        encoding="utf-8",
    )
    paths["dataset"].write_text("{}\n", encoding="utf-8")
    paths["alignment"].write_text(
        '{"model_row":0,"task_id":"fit-0","role":"fit"}\n',
        encoding="utf-8",
    )
    rollout_seal = {
        "schema": "compact-public-private-join-seal-v1",
        "selected_role": "fit",
        "training_allowed": True,
        "heldout_measure_only": False,
        "output_sha256": rescue.sha256_file(paths["rollout"]),
        "contract_sha256": rescue.sha256_file(paths["contract"]),
        "rows": 1,
    }
    paths["rollout_seal"].write_text(
        json.dumps(rollout_seal), encoding="utf-8"
    )

    feedback = {
        "schema": rescue.FEEDBACK_VIEW_REPORT_SCHEMA,
        "status": "complete",
        "outputs": {
            "rollout": rescue.file_record(paths["rollout"]),
            "seal": rescue.file_record(paths["rollout_seal"]),
            "f2": rescue.file_record(paths["f2"]),
            "f2_manifest": rescue.file_record(paths["f2_manifest"]),
            "reward_holdback_private": rescue.file_record(
                paths["private_holdback"]
            ),
        },
        "invariants": {
            "dev175_bytes_opened": False,
            "acceptance_tests_read_or_used": False,
            "rollout_contains_no_acceptance_or_holdback_fields": True,
            "holdback_is_not_a_trainer_input": True,
        },
    }
    paths["feedback_view_report"].write_text(
        json.dumps(feedback), encoding="utf-8"
    )
    provenance = {
        "output_sha256": rescue.sha256_file(paths["base_inference"]),
        "direct_prompt_mode": "code_only_v1",
        "contract_sha256": rescue.sha256_file(paths["contract"]),
        "codebook_sha256": rescue.sha256_file(paths["codebook"]),
        "codec_sha256": rescue.sha256_file(paths["codec_artifact"]),
        "tokenizer_json_sha256": rescue.sha256_file(paths["tokenizer_json"]),
        "source_overlay_sha256": rescue.sha256_file(paths["source_overlay"]),
    }
    paths["base_provenance"].write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    inputs = {
        name: {
            "path": str(path.resolve()),
            "sha256": rescue.sha256_file(path),
        }
        for name, path in paths.items()
    }
    config = {
        "schema": launcher.CONFIG_SCHEMA,
        "project_root": str(project),
        "output_root": str(output),
        "inputs": inputs,
        "plan": {
            "select_k": 4,
            "repairs_per_candidate": 4,
            "max_groups": 100,
            "seed": 7,
            "reward_timeout": 20,
            "stability_runs": 1,
            "workers": 8,
            "mcnemar_minimum_difference": 0.05,
            "mcnemar_assumed_discordance": 0.10,
            "mcnemar_alpha": 0.05,
            "mcnemar_power": 0.80,
        },
        "judge": {
            "model": "judge-model",
            "base_url": "https://judge.invalid/v1",
            "api_style": "openai_compatible_chat",
            "max_tokens": 4096,
            "timeout_seconds": 180,
            "max_retries": 0,
            "thinking_mode": "enabled",
            "reasoning_effort": "high",
        },
        "inference": {
            "decoder_model": "student",
            "decoder_revision": "revision",
            "tokenizer": "student",
            "tokenizer_revision": "revision",
            "decoder_adapter": "adapter",
            "attn_implementation": "flash_attention_2",
            "max_new_tokens": 4096,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 11,
            "role": "fit",
            "direct_prompt_mode": "code_only_v1",
            "precision": "bf16",
            "device": "cuda",
        },
        "score": {
            "reward_timeout": 20,
            "stability_runs": 1,
            "workers": 8,
        },
        "transfer": {
            "min_unique_repairs": 400,
            "allow_low_coverage_smoke": False,
        },
    }
    config_path = tmp_path / "launch.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _install_fake_plan(contract: dict[str, Any]) -> dict[str, Any]:
    layout = launcher._path_layout(contract["config"]["output_root"])
    plan = rescue.seal_artifact(
        {
            "schema": rescue.PILOT_PLAN_SCHEMA,
            "status": "complete",
            "groups": [{"task_id": "t0"}],
        },
        rescue.PLAN_HASH_FIELD,
    )
    rescue.write_json_new(layout["plan"], plan)
    return plan


def test_preflight_pins_every_input_and_is_idempotent(
    tmp_path: Path,
) -> None:
    config_path = _make_config(tmp_path)
    first = launcher.preflight(config_path)
    second = launcher.preflight(config_path)

    assert first == second
    assert first["phase_boundaries"]["judge_requires_gpu_release_ack"] is True
    assert first["phase_boundaries"]["judge_uses_gpu"] is False
    assert first["config"]["secrets_persisted"] is False
    assert set(first["config"]["inputs"]) == set(launcher.REQUIRED_INPUTS)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["inputs"]["rollout"]["path"]).write_text(
        '{"changed":true}\n', encoding="utf-8"
    )
    with pytest.raises(launcher.LaunchError, match="hash mismatch"):
        launcher.validate_launch_config(config_path)


def test_preflight_rejects_measure_role_before_any_phase(
    tmp_path: Path,
) -> None:
    config_path = _make_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inference"]["role"] = "measure"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="exactly 'fit'"):
        launcher.preflight(config_path)
    assert not (tmp_path / "run").exists()


def test_preflight_rejects_alignment_measure_175_mismatch(
    tmp_path: Path,
) -> None:
    config_path = _make_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    alignment = Path(config["inputs"]["alignment"]["path"])
    alignment.write_text(
        '{"model_row":0,"task_id":"fit-0","role":"measure",'
        '"split":"measure_175"}\n',
        encoding="utf-8",
    )
    config["inputs"]["alignment"]["sha256"] = rescue.sha256_file(alignment)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="fit-only view"):
        launcher.preflight(config_path)
    assert not (tmp_path / "run").exists()


def test_preflight_rejects_paid_retry_budget(tmp_path: Path) -> None:
    config_path = _make_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["judge"]["max_retries"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="exactly one billed"):
        launcher.preflight(config_path)


def test_diagnose_requires_sealed_gpu_release_and_sends_no_holdback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _make_config(tmp_path)
    contract = launcher.preflight(config_path)
    _install_fake_plan(contract)
    captured: list[list[str]] = []
    monkeypatch.setattr(
        launcher,
        "_run",
        lambda command, **_kwargs: captured.append(list(command)),
    )

    with pytest.raises(launcher.LaunchError, match="cannot read GPU release"):
        launcher._diagnose_phase(contract)
    launcher._gpu_release_phase(
        contract, "terminated gpu-instance-1234"
    )
    launcher._diagnose_phase(contract)

    assert len(captured) == 1
    serialized = "\n".join(captured[0]).lower()
    assert "private_holdback" not in serialized
    assert "reward_holdback" not in serialized
    assert "diagnose" in captured[0]


def test_status_stops_at_gpu_release_boundary(tmp_path: Path) -> None:
    config_path = _make_config(tmp_path)
    contract = launcher.preflight(config_path)
    assert launcher.phase_status(contract)["next_phase"] == "plan"
    _install_fake_plan(contract)
    status = launcher.phase_status(contract)
    assert status["next_phase"] == "gpu_release"
    assert status["gpu_should_be_allocated_now"] is False
    assert "gpu-release" in status["next_command"]

    launcher._gpu_release_phase(
        contract, "terminated gpu-instance-5678"
    )
    status = launcher.phase_status(contract)
    assert status["next_phase"] == "diagnose"
    assert status["paid_api_authorization_required"] is True
    assert "--allow-paid-api" in status["next_command"]


def test_subprocess_runner_prepends_project_root_to_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> None:
        observed["command"] = command
        observed.update(kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "prior")
    launcher._run(
        [sys.executable, "-c", "pass"],
        project_root=tmp_path,
    )

    values = observed["env"]["PYTHONPATH"].split(os.pathsep)
    assert values[0] == str(tmp_path.resolve())
    assert values[1] == "prior"
    assert observed["cwd"] == tmp_path.resolve()
    assert observed["check"] is True


def test_resume_requires_explicit_paid_and_gpu_authorization() -> None:
    args = launcher.parse_args(
        ["--config", "launch.json", "resume"]
    )
    assert args.allow_paid_api is False
    assert args.allow_gpu is False
    assert args.allow_missing is False
