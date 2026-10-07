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
    / "t5gemma2_api_rs_sft_openrouter_minimax_m3_residual_probe.sh"
)
CONF = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-rs-sft-openrouter-minimax-m3-residual-probe.conf"
)
RUNNER = ROOT / "scripts" / "training" / "t5gemma2_api_rs_sft_rescue.py"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_openrouter_minimax_probe_contract_is_exact_and_under_one_dollar() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    conf = CONF.read_text(encoding="utf-8")

    for fragment in (
        "--provider openrouter_chat",
        "--model minimax/minimax-m3",
        "--base_url https://openrouter.ai/api/v1",
        "--chat_token_parameter max_tokens",
        "--openrouter_provider_only gmicloud/fp8",
        "--openrouter_require_parameters",
        "--openrouter_enforce_distillable_text",
        "--openrouter_reasoning enabled",
        "--openrouter_include_reasoning",
        "--eligible_task_offset 60",
        "--max_tasks 20",
        "--max_calls 20",
        "--max_input_tokens_per_call 49152",
        "--max_output_tokens 16384",
        "--max_input_tokens_total 983040",
        "--max_output_tokens_total 327680",
        "--max_total_tokens 1310720",
        "--max_usd 1",
        "--input_usd_per_million 0.30",
        "--output_usd_per_million 1.20",
        (
            "--expected_scheduled_task_ids_sha256 "
            "5bac934b9e0e56a81f19c22aca69d1f0dccdf2a1f7038d443171dfa087f5e14e"
        ),
    ):
        assert fragment in launcher
    assert "--openrouter_allow_fallbacks" not in launcher
    assert "OPENROUTER_API_KEY" in launcher
    assert 'source "${SECRET_FILE}"' not in launcher
    assert "openrouter-minimax-m3-residual-probe" in conf

    capacity, contract = schedule_capacity(
        max_calls=20,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=16384,
        max_input_tokens_total=983040,
        max_output_tokens_total=327680,
        max_total_tokens=1310720,
        max_usd=Decimal("1"),
        input_usd_per_million=Decimal("0.30"),
        output_usd_per_million=Decimal("1.20"),
    )
    assert capacity == 20
    reserved = (
        Decimal(contract["worst_case_usd_nanos_per_call"])
        * Decimal(capacity)
        / Decimal(1_000_000_000)
    )
    assert reserved == Decimal("0.688128")
    assert reserved < Decimal("1")


def test_openrouter_schedule_is_exact_positions_60_to_79() -> None:
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
    assert len(residual) == 123
    tranches = [set(residual[start : start + 20]) for start in (0, 20, 40, 60)]
    for left in range(len(tranches)):
        for right in range(left + 1, len(tranches)):
            assert not (tranches[left] & tranches[right])
    assert canonical_sha256(residual[60:80]) == (
        "5bac934b9e0e56a81f19c22aca69d1f0dccdf2a1f7038d443171dfa087f5e14e"
    )


def test_launcher_pins_runner_azure_and_all_prior_training_artifacts() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert sha256_file(RUNNER) == (
        "f655a8cb0253033f5674845b83f71f44df35ba7c962448b349e449aa04b79dda"
    )
    # Exact Azure report plus journal, output views, and manifests.
    for digest in (
        "33bf539f37beb285459511ee5349f8eec34b8335ff4c07339ce8a95467379cf0",
        "06af6f49ea45d485e6c61b0e4a8b783894ffb4a1491235c56fb2c0428cf0e683",
        "336874a72569f6a82bbc844260b772e7f3dc631c399e23c979d54502713ea727",
        "aa22e905037222a34eb01964eb2f6b6a9826ffbb19376490ff1c130a2d8bf18b",
        "a8c9bc693a27d46c5d83d7b2beb4dddcdae6e1d46d64916d163688de3a3ba557",
        "2ff8c7cfef215075fa8e3d2a867084b422ee1e3ed0a04c7059c0b13d3ef5d75c",
        "6b2840eb0e270b1bc246350130d9cc1c2de671a66beecc2f1ae506a580dbebdd",
        "77cae6c03ca0dd1e80e303afedf2fb551fd1e8ea7ceee0844ecf8448877b423e",
        "903fd33974f37fb6144267eac84e39f7d5d8ffcf437bf96db79920fd1f9b6924",
    ):
        assert digest in launcher
    # Each earlier Sonnet/Opus tranche has journal, chain, report, two direct
    # views, two manifests, and two repair views pinned by the launcher.
    for variable in (
        "TRANCHE1_JOURNAL",
        "TRANCHE2_JOURNAL",
        "FIRST_OPUS_JOURNAL",
        "SECOND_OPUS_JOURNAL",
    ):
        assert launcher.count(f'"${{{variable}}}"') == 1
        assert launcher.count(f'"${{{variable}}}.chain-head.json"') == 1
    for dirname in (
        "TRANCHE1_DIR",
        "TRANCHE2_DIR",
        "FIRST_OPUS_DIR",
        "SECOND_OPUS_DIR",
    ):
        for filename in (
            "direct_hard_targets.jsonl",
            "direct_hard_targets_f2.jsonl",
            "direct_manifest.json",
            "repair_policy_manifest.json",
            "repair_policy_sources.jsonl",
            "repair_policy_targets.jsonl",
        ):
            assert f'"${{{dirname}}}/{filename}"' in launcher
