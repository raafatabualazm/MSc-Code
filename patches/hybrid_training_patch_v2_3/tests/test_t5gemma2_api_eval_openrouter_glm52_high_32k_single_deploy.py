from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    sha256_file,
)
from scripts.training.t5gemma2_api_rs_sft_rescue import (
    RUN_SCHEMA,
    schedule_capacity,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LAUNCHER = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2_api_eval_openrouter_glm52_high_32k_single.sh"
)
CONF = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-eval-openrouter-glm52-high-32k-single.conf"
)
RUNNER = ROOT / "scripts" / "training" / "t5gemma2_api_rs_sft_rescue.py"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_glm52_high_32k_single_is_evaluation_only_and_under_twelve_cents() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    conf = CONF.read_text(encoding="utf-8")

    for fragment in (
        "--provider openrouter_chat",
        "--model z-ai/glm-5.2",
        "--base_url https://openrouter.ai/api/v1",
        "--chat_token_parameter max_tokens",
        "--openrouter_provider_only novita/fp8",
        "--openrouter_require_parameters",
        "--evaluation_only",
        "--openrouter_reasoning enabled",
        "--openrouter_reasoning_effort high",
        "--openrouter_include_reasoning",
        "--eligible_task_offset 101",
        "--max_tasks 1",
        "--max_parents_per_task 1",
        "--samples_per_parent 1",
        "--max_calls 1",
        "--max_input_tokens_per_call 49152",
        "--max_output_tokens 32768",
        "--max_input_tokens_total 49152",
        "--max_output_tokens_total 32768",
        "--max_total_tokens 81920",
        "--max_usd 0.12",
        "--input_usd_per_million 0.70",
        "--output_usd_per_million 2.20",
        (
            "--expected_scheduled_task_ids_sha256 "
            "1751df08e563a3b07c0c9b968e7a5723cf4d3227402596485306c8f3d5595247"
        ),
    ):
        assert fragment in launcher
    assert "--openrouter_enforce_distillable_text" not in launcher
    assert "external_license" not in launcher
    assert "--openrouter_allow_fallbacks" not in launcher
    assert "OPENROUTER_API_KEY" in launcher
    assert 'source "${SECRET_FILE}"' not in launcher
    assert "t5gemma-api-eval-openrouter-glm52-high-32k-single" in conf
    assert "t5gemma2_api_eval_openrouter_glm52_high_32k_single.sh" in conf

    capacity, contract = schedule_capacity(
        max_calls=1,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=32768,
        max_input_tokens_total=49152,
        max_output_tokens_total=32768,
        max_total_tokens=81920,
        max_usd=Decimal("0.12"),
        input_usd_per_million=Decimal("0.70"),
        output_usd_per_million=Decimal("2.20"),
    )
    assert capacity == 1
    reserved = (
        Decimal(contract["worst_case_usd_nanos_per_call"])
        / Decimal(1_000_000_000)
    )
    assert reserved == Decimal("0.106496")
    assert reserved < Decimal("0.12")


def test_glm52_high_32k_single_is_exact_fresh_residual_position_101() -> None:
    pilot = (
        WORKSPACE
        / "artifacts"
        / "t5gemma2_local_rs_sft_pilot_2epoch_v1"
        / "harvest.journal.jsonl"
    )
    terminals = [
        row
        for row in _jsonl(pilot)
        if row.get("event") == "task_terminal"
        and row.get("visible_unique_passes") == 0
    ]
    ordered = sorted(
        ((int(row["task_position"]), str(row["task_id"])) for row in terminals),
        key=lambda item: canonical_sha256(
            {
                "schema": RUN_SCHEMA,
                "seed": 42,
                "task_id": item[1],
                "local_task_position": item[0],
            }
        ),
    )
    sonnet_successes: set[str] = set()
    for dirname in (
        "t5gemma2_api_rs_sft_claude_production_2epoch_v1",
        "t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1",
    ):
        sonnet_successes.update(
            str(row["task_id"])
            for row in _jsonl(
                WORKSPACE / "artifacts" / dirname / "direct_hard_targets.jsonl"
            )
        )
    residual = [
        (task_position, task_id)
        for task_position, task_id in ordered
        if task_id not in sonnet_successes
    ]
    assert len(sonnet_successes) == 65
    assert len(residual) == 123
    assert residual[101] == (16, "fresh-eval-e0ef0e1c780c")
    assert canonical_sha256([residual[101][1]]) == (
        "1751df08e563a3b07c0c9b968e7a5723cf4d3227402596485306c8f3d5595247"
    )
    lower_cap_task_id = residual[100][1]
    assert residual[101][1] != lower_cap_task_id


def test_glm52_high_32k_single_pins_runner_and_has_no_trainable_paths() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert sha256_file(RUNNER) == (
        "f655a8cb0253033f5674845b83f71f44df35ba7c962448b349e449aa04b79dda"
    )
    assert launcher.count(
        "f655a8cb0253033f5674845b83f71f44df35ba7c962448b349e449aa04b79dda"
    ) == 1
    assert "direct_hard_targets" not in launcher
    assert "repair_policy_" not in launcher
    assert "T5GEMMA_OPENROUTER_GLM52_HIGH_SINGLE_OUTPUT_DIR" not in launcher
    assert "t5gemma2_api_eval_openrouter_glm52_high_single_2epoch_v1" not in launcher
