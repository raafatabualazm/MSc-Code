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
    / "t5gemma2_api_rs_sft_openrouter_glm52_residual_probe.sh"
)
CONF = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-eval-openrouter-glm52-residual-probe.conf"
)
RUNNER = ROOT / "scripts" / "training" / "t5gemma2_api_rs_sft_rescue.py"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_glm52_probe_is_evaluation_only_and_under_two_dollars() -> None:
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
        "--openrouter_reasoning_effort xhigh",
        "--openrouter_include_reasoning",
        "--eligible_task_offset 80",
        "--max_tasks 20",
        "--max_calls 20",
        "--max_input_tokens_per_call 49152",
        "--max_output_tokens 16384",
        "--max_input_tokens_total 983040",
        "--max_output_tokens_total 327680",
        "--max_total_tokens 1310720",
        "--max_usd 2",
        "--input_usd_per_million 0.70",
        "--output_usd_per_million 2.20",
        (
            "--expected_scheduled_task_ids_sha256 "
            "380b7dc9603a5da7367859d897ebf312c8660374326cdd187e8d1df0dc7b0f51"
        ),
    ):
        assert fragment in launcher
    assert "--openrouter_enforce_distillable_text" not in launcher
    assert "external_license" not in launcher
    assert "--openrouter_allow_fallbacks" not in launcher
    assert '--prior_success_report "${M3_REPORT}"' not in launcher
    assert "OPENROUTER_API_KEY" in launcher
    assert 'source "${SECRET_FILE}"' not in launcher
    assert "t5gemma-api-eval-openrouter-glm52-residual-probe" in conf

    capacity, contract = schedule_capacity(
        max_calls=20,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=16384,
        max_input_tokens_total=983040,
        max_output_tokens_total=327680,
        max_total_tokens=1310720,
        max_usd=Decimal("2"),
        input_usd_per_million=Decimal("0.70"),
        output_usd_per_million=Decimal("2.20"),
    )
    assert capacity == 20
    reserved = (
        Decimal(contract["worst_case_usd_nanos_per_call"])
        * Decimal(capacity)
        / Decimal(1_000_000_000)
    )
    assert reserved == Decimal("1.409024")
    assert reserved < Decimal("2")


def test_glm52_schedule_is_exact_sonnet_residual_positions_80_to_99() -> None:
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
    residual = [task_id for _, task_id in ordered if task_id not in sonnet_successes]
    assert len(sonnet_successes) == 65
    assert len(residual) == 123
    assert canonical_sha256(residual[80:100]) == (
        "380b7dc9603a5da7367859d897ebf312c8660374326cdd187e8d1df0dc7b0f51"
    )


def test_glm52_launcher_pins_runner_and_rejected_m3_evidence() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert sha256_file(RUNNER) == (
        "f655a8cb0253033f5674845b83f71f44df35ba7c962448b349e449aa04b79dda"
    )
    for digest in (
        "c474ac844acd027a02bf48015447b76a0955052c85c44a9cd698a020e88caef4",
        "b4794de43b6fd74073754671579a049ecd0e6caa6f3ca5b03c9707f51eb670e8",
        "97329655958e0cde43c328990c2d115b749a8f4cb17647b314f819bb0c3fb137",
        "5369a707643d953ef4b13542ad95f9fcd3d606e2d4a1d77eb051c1e7ae2d8b9d",
        "745226bea7c3cddda88c45d48b2efd72709fc828cffb49644723f210b35c6570",
    ):
        assert digest in launcher
    assert launcher.count('"${M3_JOURNAL}"') == 1
    assert launcher.count('"${M3_JOURNAL}.chain-head.json"') == 1
    assert launcher.count('"${M3_REPORT}"') == 2
    assert ".schedule.provider_responses == 0" in launcher
    assert ".verification.verified_unique_hard_targets == 0" in launcher
