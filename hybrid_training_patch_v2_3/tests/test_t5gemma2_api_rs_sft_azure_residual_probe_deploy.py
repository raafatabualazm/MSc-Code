from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
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
    / "t5gemma2_api_rs_sft_azure_production_residual_probe.sh"
)
CONF = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-rs-sft-azure-production-residual-probe.conf"
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_azure_residual_probe_contract_is_exact_and_isolated() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    conf = CONF.read_text(encoding="utf-8")
    assert "--provider azure_v1_chat" in launcher
    assert "--model gpt-chat-latest" in launcher
    assert "--chat_token_parameter max_completion_tokens" in launcher
    assert "--reasoning_effort" not in launcher
    assert "--eligible_task_offset 40" in launcher
    assert "--max_tasks 20" in launcher
    assert "--max_calls 20" in launcher
    assert "--max_input_tokens_per_call 49152" in launcher
    assert "--max_output_tokens 8192" in launcher
    assert "--max_input_tokens_total 983040" in launcher
    assert "--max_output_tokens_total 163840" in launcher
    assert "--max_total_tokens 1146880" in launcher
    assert "--max_usd 9.0112" in launcher
    assert (
        "--expected_scheduled_task_ids_sha256 "
        "05c8f8052b820113dfa881c2181982fbca7f007de4df86af2ba2f0d96c0c30c7"
        in launcher
    )
    assert "AZURE_OPENAI_ENDPOINT" in launcher
    assert "AZURE_OPENAI_API_KEY" in launcher
    assert "source \"${SECRET_FILE}\"" not in launcher
    assert "azure-production-residual-probe" in conf
    assert "claude-opus-production-residual-tranche2" not in conf

    capacity, contract = schedule_capacity(
        max_calls=20,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=8192,
        max_input_tokens_total=983040,
        max_output_tokens_total=163840,
        max_total_tokens=1146880,
        max_usd=Decimal("9.0112"),
        input_usd_per_million=Decimal("5"),
        output_usd_per_million=Decimal("25"),
    )
    assert capacity == 20
    reserved = (
        Decimal(contract["worst_case_usd_nanos_per_call"])
        * Decimal(capacity)
        / Decimal(1_000_000_000)
    )
    assert reserved == Decimal("9.0112")
    assert reserved < Decimal("23")


def test_azure_schedule_is_exact_sonnet_residual_positions_40_to_59() -> None:
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
    first_60 = [set(residual[start : start + 20]) for start in (0, 20, 40)]
    assert not (first_60[0] & first_60[1])
    assert not (first_60[0] & first_60[2])
    assert not (first_60[1] & first_60[2])
    assert canonical_sha256(residual[40:60]) == (
        "05c8f8052b820113dfa881c2181982fbca7f007de4df86af2ba2f0d96c0c30c7"
    )


def test_azure_launcher_pins_both_completed_opus_tranches() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for digest in (
        "f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b",
        "fa0c70c73767a525f2ca710fd822cb2bdca60140f133696ad15b87e71d2751d1",
        "5c610a4073122e209e26af8e689a683258405c00e58a23c6e9a109c76f9c4c6c",
        "d5da7c8ed6ec045239602f08d290dac7e123d44c9edc17b1bf121763db1b1511",
        "2e02a7db60d0baf9d64afdc9b5bb211fcd0253186b490e9af911de8d49b87bf7",
        "5fd2d1cd56b0c2de0ed79fd4cfdb3017244774f038517e574a7089a39ea51a91",
    ):
        assert digest in launcher
