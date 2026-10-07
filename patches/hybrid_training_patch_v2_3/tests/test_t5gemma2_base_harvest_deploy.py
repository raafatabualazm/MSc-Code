from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_base_harvest_2epoch.sh"
SUPERVISOR = ROOT / "deploy" / "vast" / "t5gemma2-base-harvest-2epoch.conf"
REMAINING_SUPERVISOR = (
    ROOT / "deploy" / "vast" / "t5gemma2-base-harvest-remaining-2epoch.conf"
)
RESIDUAL_LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_base_residual_harvest_2epoch.sh"
RESIDUAL_SUPERVISOR = (
    ROOT / "deploy" / "vast" / "t5gemma2-base-harvest-residual-2epoch.conf"
)


def test_base_harvest_pins_training_pool_and_exact_two_epoch_checkpoint() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    expected = {
        "f0d02161da9fac96d31085eb8b569ab44dc42902853db3cf1095d6643dd26dbe":
            "result.json",
        "562c3da5f89428e6a7263ad8ec79dde9c8b6eb25c77949606277d7d80aecea4f":
            "run_contract.json",
        "c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3":
            "adapter_config.json",
        "83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc":
            "adapter_model.safetensors",
        "f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d":
            "tokenizer.json",
        "14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c":
            "verpo_rollout_feedback.jsonl",
        "c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3":
            "verpo_teacher_f2.jsonl",
        "dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f":
            "reward_holdback.private.jsonl",
    }
    for digest, filename in expected.items():
        assert digest in text
        assert filename in text
    assert "checkpoint-optstep-000348" in text
    assert '"487d4acf21a4d70c70bf534265b5263c9424979e"' in text
    assert '"8c3e78cb0fc5a2483a01029a13be9f0536c203de0d28c3302b87eba34b36f3d0"' in text
    assert "| sha256sum -c -" in text
    assert "cmp -s" in text
    assert "/usr/bin/wc -l" in text
    assert '-ne 2386' in text
    assert "feedback pool is not 2,386 rows" in text
    assert "--allow_unpinned_inputs" not in text


def test_base_harvest_is_disjoint_expansion_without_local_repairs() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = (
        '--pilot_offset "${PILOT_OFFSET}"',
        '--pilot_tasks "${PILOT_TASKS}"',
        "--base_samples 4",
        "--repair_samples 0",
        "--max_repair_parents 0",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--generation_batch_size 4",
        "--stability_runs 2",
        "--seed 42",
    )
    for argument in required:
        assert argument in text
    assert "t5gemma2_local_base_harvest_2epoch_1000x4_v1" in text
    assert "T5GEMMA_BASE_HARVEST_OFFSET:-200" in text
    assert "T5GEMMA_BASE_HARVEST_TASKS:-1000" in text
    assert "PILOT_OFFSET + PILOT_TASKS > 2386" in text
    assert "dev_multifunction" not in text
    assert "heldout_175" not in text
    assert "OPENAI" not in text
    assert "ANTHROPIC" not in text
    assert "DASHSCOPE" not in text
    assert "curl " not in text


def test_base_harvest_supervisor_is_manual_resumable_and_group_owned() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    assert 'DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"' in launcher
    assert 'export PATH="$(dirname "${DART_BIN}"):${PATH}"' in launcher
    assert "[program:t5gemma-base-harvest-2epoch]" in supervisor
    assert (
        "command=/opt/supervisor-scripts/t5gemma2_base_harvest_2epoch.sh"
        in supervisor
    )
    assert "autostart=false" in supervisor
    assert "autorestart=unexpected" in supervisor
    assert "exitcodes=0,78" in supervisor
    assert "stopasgroup=true" in supervisor
    assert "killasgroup=true" in supervisor
    assert (
        "stdout_logfile=/workspace/logs/t5gemma-base-harvest-2epoch.log"
        in supervisor
    )


def test_remaining_training_pool_arm_is_exactly_disjoint_and_exhaustive() -> None:
    text = REMAINING_SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-base-harvest-remaining-2epoch]" in text
    assert 'T5GEMMA_BASE_HARVEST_OFFSET="1200"' in text
    assert 'T5GEMMA_BASE_HARVEST_TASKS="1186"' in text
    assert 1200 + 1186 == 2386
    assert (
        'T5GEMMA_BASE_HARVEST_OUTPUT_DIR="/workspace/artifacts/'
        't5gemma2_local_base_harvest_2epoch_remaining1186x4_v1"'
    ) in text
    assert "autostart=false" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text


def test_residual_harvest_is_fail_closed_and_base_only() -> None:
    launcher = RESIDUAL_LAUNCHER.read_text(encoding="utf-8")
    supervisor = RESIDUAL_SUPERVISOR.read_text(encoding="utf-8")
    for argument in (
        "--exclude_verified_report \"${PILOT_REPORT}\"",
        "--exclude_verified_report \"${EXPANDED_REPORT}\"",
        "--exclude_verified_report \"${REMAINING_REPORT}\"",
        "--base_samples 4",
        "--repair_samples 0",
        "--max_repair_parents 0",
        "--pilot_offset 0",
        "--pilot_tasks 1500",
    ):
        assert argument in launcher
    assert "T5GEMMA_RESIDUAL_BLOCKED fill exact" in launcher
    assert "T5GEMMA_RESIDUAL_REMAINING_REPORT_SHA:-" in launcher
    assert "heldout_175" not in launcher
    assert "OPENAI" not in launcher
    assert "ANTHROPIC" not in launcher
    assert "DASHSCOPE" not in launcher
    assert "curl " not in launcher
    assert "[program:t5gemma-base-harvest-residual-2epoch]" in supervisor
    assert "autostart=false" in supervisor
    assert "stopasgroup=true" in supervisor
    assert "killasgroup=true" in supervisor
