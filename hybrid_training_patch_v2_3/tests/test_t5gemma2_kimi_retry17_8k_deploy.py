from __future__ import annotations

from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2_api_rs_sft_openrouter_kimi_k3_retry17_8k.sh"
)
SUPERVISOR = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-rs-sft-openrouter-kimi-k3-retry17-8k.conf"
)


def test_launcher_retries_only_the_sealed_failure_or_truncation_union() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = (
        "ad681aaa68db63dbc64ce847f32f18e2740e4db2050d1211e5d5457fdc6dff69",
        "fe9bcd00c6774432b7911129246c8b2837523d85b1c94efb29c03f85ae860205",
        "--retry_parse_failures_or_truncations_report",
        "--expected_retry_parse_failures_or_truncations_tasks 17",
        "15a66c43b97fa72fd47702689babc3b6ab33ee48f24813a34a7b62f2d9ccc00a",
        "--max_tasks 17",
        "--max_parents_per_task 1",
        "--samples_per_parent 1",
        "--max_calls 17",
        "--max_input_tokens_per_call 13312",
        "--max_output_tokens 8192",
        "--max_input_tokens_total 226304",
        "--max_output_tokens_total 139264",
        "--max_total_tokens 365568",
        "--max_usd 3",
        "--input_usd_per_million 3",
        "--output_usd_per_million 15",
        "--openrouter_allow_fallbacks",
        "--openrouter_enforce_distillable_text",
        "--abort_on_provider_error",
    )
    for value in required:
        assert value in text
    assert "--evaluation_only" not in text
    assert "--prior_success_report" not in text
    assert "mixed_paired50_v12" in text
    assert "retry17_8k_v1" in text


def test_retry_worst_case_is_inside_three_dollar_hard_cap() -> None:
    cost = (
        Decimal(226_304) * Decimal(3)
        + Decimal(139_264) * Decimal(15)
    ) / Decimal(1_000_000)
    assert cost == Decimal("2.767872")
    assert cost < Decimal(3)


def test_retry_supervisor_is_manual_and_fail_closed() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-api-rs-sft-openrouter-kimi-k3-retry17-8k]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
