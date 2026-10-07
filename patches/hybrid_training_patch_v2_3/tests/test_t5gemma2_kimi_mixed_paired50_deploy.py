from __future__ import annotations

from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTESTOR = (
    ROOT / "scripts" / "training" / "t5gemma2_kimi_schedule_attestation.py"
)
LAUNCHER = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2_api_rs_sft_openrouter_kimi_k3_mixed_paired50.sh"
)
SUPERVISOR = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-rs-sft-openrouter-kimi-k3-mixed-paired50.conf"
)


def test_attestor_reuses_exact_rescue_schedule_and_binds_loader() -> None:
    text = ATTESTOR.read_text(encoding="utf-8")
    assert "rescue._load_completed_local_run(loader_args)" in text
    assert "rescue.select_rescue_plans(" in text
    assert '"mixed_loader_sha256"' in text
    assert '"heldout_175_opened": False' in text
    assert "plans[: args.tasks]" in text


def test_launcher_is_exact_capped_kimi_train_only_arm() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = (
        "--model moonshotai/kimi-k3",
        "--openrouter_provider_only baseten/fp8",
        "--openrouter_provider_only modal/mxfp4",
        "--openrouter_provider_only digitalocean",
        "--openrouter_provider_only together",
        "--openrouter_provider_only fireworks",
        "--openrouter_provider_only moonshotai/mxfp4",
        "--openrouter_provider_order baseten/fp8",
        "--openrouter_provider_order modal/mxfp4",
        "--openrouter_provider_order digitalocean",
        "--openrouter_provider_order together",
        "--openrouter_provider_order fireworks",
        "--openrouter_provider_order moonshotai/mxfp4",
        "--openrouter_allow_fallbacks",
        "--openrouter_require_parameters",
        "--openrouter_enforce_distillable_text",
        "--openrouter_reasoning enabled",
        "--openrouter_reasoning_effort low",
        "--openrouter_include_reasoning",
        "--max_tasks 50",
        "--max_parents_per_task 1",
        "--samples_per_parent 1",
        "--max_calls 50",
        "--max_input_tokens_per_call 13312",
        "--max_output_tokens 2048",
        "--max_input_tokens_total 665600",
        "--max_output_tokens_total 102400",
        "--max_total_tokens 768000",
        "--max_usd 4",
        "--input_usd_per_million 3",
        "--output_usd_per_million 15",
        "--inter_call_delay_seconds 10",
        "--abort_on_provider_error",
        "--provider_max_attempts 8",
        "--provider_retry_base_seconds 2",
        "--provider_retry_max_seconds 30",
        "--expected_scheduled_task_ids_sha256",
    )
    for value in required:
        assert value in text
    assert "--evaluation_only" not in text
    assert "--prior_success_report" not in text
    assert "dev_multifunction_binary" not in text
    assert ".privacy_invariants.heldout_175_opened == false" in text
    assert "T5GEMMA_KIMI_ROUTE_CANARY_OK" not in text
    assert "T5GEMMA_KIMI_FAIL_FAST" in text
    assert "mixed_paired50_v12" in text


def test_worst_case_cost_is_below_hard_cap() -> None:
    cost = (
        Decimal(665_600) * Decimal(3)
        + Decimal(102_400) * Decimal(15)
    ) / Decimal(1_000_000)
    assert cost == Decimal("3.5328")
    assert cost < Decimal(4)


def test_supervisor_waiter_is_manual_and_fail_closed() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma-api-rs-sft-openrouter-kimi-k3-mixed-paired50]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
