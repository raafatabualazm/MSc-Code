from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    sha256_file,
)
from scripts.training.seq2seq_verpo_core import sha256_text
from scripts.training.t5gemma2_api_rs_sft_rescue import (
    RUN_SCHEMA,
    SYSTEM_PROMPT,
    schedule_capacity,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LAUNCHER = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2.sh"
)
CONF = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-rs-sft-claude-opus-production-residual-tranche2.conf"
)
SCRIPT = ROOT / "scripts" / "training" / "t5gemma2_api_rs_sft_rescue.py"
PILOT_JOURNAL = (
    WORKSPACE
    / "artifacts"
    / "t5gemma2_local_rs_sft_pilot_2epoch_v1"
    / "harvest.journal.jsonl"
)
SONNET_DIRS = (
    "t5gemma2_api_rs_sft_claude_production_2epoch_v1",
    "t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1",
)
OPUS_FIRST_DIR = (
    WORKSPACE
    / "artifacts"
    / "t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1"
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_residual_tranche2_launcher_is_exactly_sealed_and_affordable() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    conf = CONF.read_text(encoding="utf-8")

    assert sha256_file(SCRIPT) == (
        "4900c5704f1488a55369d2149f637ce8c2346443330524f0a5c58b0a16820ad8"
    )
    assert sha256_text(SYSTEM_PROMPT) == (
        "0118f9f452ff23093b11abab0076a6297f6d91be7d0777e7a91b1720f183bb5e"
    )
    assert "residual_tranche2_2epoch_v1" in launcher
    assert "--model claude-opus-5" in launcher
    assert "--anthropic_thinking adaptive" in launcher
    assert "--anthropic_effort high" in launcher
    assert "--eligible_task_offset 20" in launcher
    assert "--max_tasks 20" in launcher
    assert "--max_calls 20" in launcher
    assert "--max_input_tokens_per_call 49152" in launcher
    assert "--max_output_tokens 16384" in launcher
    assert "--max_input_tokens_total 983040" in launcher
    assert "--max_output_tokens_total 327680" in launcher
    assert "--max_total_tokens 1310720" in launcher
    assert "--max_usd 13.1072" in launcher
    assert "f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b" in launcher
    assert "49b97de386b759955497e3f9ab7b4358ca5e74ebf3a877fb6c7f3d98e39275b6" in launcher
    assert "79b2d7a95dadabc5b6701d063192df684054bfbbd322ee1faf4d9171ed6e186c" in launcher
    assert "15ef808838ed01347e646e9b4462f48ae88d4afcb467d144f6c6283576abf180" in launcher
    assert '.budget_charged.estimated_usd == "1.776690000"' in launcher
    assert '.verification.verified_unique_hard_targets == 3' in launcher
    assert '.provider.max_output_tokens == 8192' in launcher
    assert (
        "--expected_scheduled_task_ids_sha256 "
        "f6d83bab2b4ff9dcb4a8f0ba1c1935b6dd36a79b044ca576bb6940aaa10e8655"
        in launcher
    )
    assert "production-residual-tranche2" in conf

    capacity, contract = schedule_capacity(
        max_calls=20,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=16384,
        max_input_tokens_total=983040,
        max_output_tokens_total=327680,
        max_total_tokens=1310720,
        max_usd=Decimal("13.1072"),
        input_usd_per_million=Decimal("5"),
        output_usd_per_million=Decimal("25"),
    )
    assert capacity == 20
    reserved = (
        Decimal(contract["worst_case_usd_nanos_per_call"])
        * Decimal(capacity)
        / Decimal(1_000_000_000)
    )
    assert reserved == Decimal("13.1072")
    assert Decimal("11.758550") + reserved == Decimal("24.865750")
    assert Decimal("24.865750000") < Decimal("30")


def test_residual_positions_20_to_39_are_exact_and_disjoint() -> None:
    terminals = [
        row
        for row in _jsonl(PILOT_JOURNAL)
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
    for dirname in SONNET_DIRS:
        rows = _jsonl(
            WORKSPACE / "artifacts" / dirname / "direct_hard_targets.jsonl"
        )
        sonnet_successes.update(str(row["task_id"]) for row in rows)

    residual = [task_id for _, task_id in ordered if task_id not in sonnet_successes]
    first = residual[:20]
    second = residual[20:40]
    assert len(ordered) == 188
    assert len(sonnet_successes) == 65
    assert len(residual) == 123
    assert set(first).isdisjoint(second)
    assert canonical_sha256(second) == (
        "f6d83bab2b4ff9dcb4a8f0ba1c1935b6dd36a79b044ca576bb6940aaa10e8655"
    )

    prior_ids = [
        str(row["task_id"])
        for row in _jsonl(OPUS_FIRST_DIR / "api_rescue.journal.jsonl")
        if row.get("event") == "call_result"
    ]
    assert prior_ids == first
    assert set(prior_ids).isdisjoint(second)


def test_launcher_preserves_private_gate_ordering_contract() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "--prior_success_report \"${TRANCHE1_REPORT}\"" in launcher
    assert "--prior_success_report \"${TRANCHE2_REPORT}\"" in launcher
    assert "--require_prior_schedules_disjoint" in launcher
    assert "--require_prior_schedule_complete_coverage" in launcher
    assert "--expected_prior_scheduled_tasks 188" in launcher
    assert "--expected_prior_verified_tasks 65" in launcher
    assert "--expected_residual_tasks 123" in launcher
    assert "--stability_runs 2" in launcher

    script = SCRIPT.read_text(encoding="utf-8")
    assert '"all_api_calls_before_any_private_gate": True' in script
    assert '"private_failure_triggers_api_call": False' in script
    assert '"private_holdback_sent_to_provider": False' in script
