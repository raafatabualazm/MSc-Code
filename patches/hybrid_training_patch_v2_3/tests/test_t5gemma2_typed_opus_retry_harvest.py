from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_opus_retry_harvest as opus


RETRY_IDS = [
    "sigless_b78f065b5f17",
    "sigless_59690a61e2a4",
    "fresh-eval-204d6fbaef33",
    "fresh-eval-d983d52b9c1f",
    "fresh-eval-a8b68ea26a73",
    "fresh-eval-9ee989a731f8",
    "sigless_b8c3d7d38830",
    "sigless_4128d40d5043",
    "fresh-eval-b18403dc7da1",
    "fresh-eval-bc74f5897f24",
    "fresh-eval-927b729728de",
    "sigless_731f6c240879",
    "sigless_ac279ba1425a",
    "sigless_82f7067efda2",
    "fresh-eval-616b875ce7cf",
    "sigless_ae7501aa7a1f",
]


def _eligible(task_id: str, position: int) -> tuple[int, SimpleNamespace, dict]:
    return (
        position,
        SimpleNamespace(task_id=task_id),
        {"task_id": task_id, "api_eligible": True},
    )


def _prior_records() -> list[dict]:
    kimi = [f"kimi-{index}" for index in range(50)]
    sonnet_verified = ["sonnet-pass-0", "sonnet-pass-1"]
    return [
        {
            "phase": cascade.PHASE_KIMI_INITIAL,
            "cohort_index": 0,
            "scheduled_task_ids": kimi,
            "verified_task_ids": kimi[:15],
            "retry_eligible_task_ids": kimi[15:30],
        },
        {
            "phase": cascade.PHASE_KIMI_RETRY,
            "cohort_index": 0,
            "scheduled_task_ids": kimi[15:30],
            "verified_task_ids": kimi[15:17],
            "retry_eligible_task_ids": kimi[17:30],
        },
        {
            "phase": cascade.PHASE_SONNET_RESIDUAL,
            "cohort_index": 0,
            "scheduled_task_ids": RETRY_IDS + sonnet_verified + kimi[30:50],
            "verified_task_ids": sonnet_verified,
            "retry_eligible_task_ids": RETRY_IDS,
        },
    ]


def test_exact_sonnet_retry_cohort_is_sealed_and_excludes_all_successes() -> None:
    prior = _prior_records()
    ids = list(
        dict.fromkeys(
            task_id
            for row in prior
            for task_id in row["scheduled_task_ids"]
        )
    ) + ["unused-residual"]
    selected, record = opus.select_exact_sonnet_retry(
        all_api_eligible=[_eligible(task_id, i) for i, task_id in enumerate(ids)],
        prior_records=prior,
    )
    selected_ids = [row[1].task_id for row in selected]
    verified = {
        task_id for row in prior for task_id in row["verified_task_ids"]
    }
    assert selected_ids == RETRY_IDS
    assert canonical_sha256(selected_ids) == opus.EXPECTED_RETRY_TASK_IDS_SHA256
    assert not (set(selected_ids) & verified)
    assert record["targeted_non_code_or_length_only"] is True
    assert record["accepted_nontruncated_sonnet_responses_regenerated"] is False
    assert record["selection_uses_heldout_175"] is False


def test_retry_cohort_fails_closed_on_digest_or_verified_overlap() -> None:
    prior = _prior_records()
    ids = list(dict.fromkeys(RETRY_IDS + ["sonnet-pass-0", "sonnet-pass-1"] + [f"kimi-{i}" for i in range(50)]))
    eligible = [_eligible(task_id, i) for i, task_id in enumerate(ids)]
    wrong = _prior_records()
    wrong[-1]["retry_eligible_task_ids"] = list(reversed(RETRY_IDS))
    with pytest.raises(ValueError, match="retry cohort differs"):
        opus.select_exact_sonnet_retry(
            all_api_eligible=eligible,
            prior_records=wrong,
        )
    contaminated = _prior_records()
    contaminated[-1]["verified_task_ids"].append(RETRY_IDS[0])
    with pytest.raises(ValueError, match="retry cohort differs"):
        opus.select_exact_sonnet_retry(
            all_api_eligible=eligible,
            prior_records=contaminated,
        )


def _profile(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "provider": "anthropic",
        "model": opus.MODEL,
        "api_key_env": "ANTHROPIC_API_KEY",
        "anthropic_thinking": "adaptive",
        "anthropic_effort": "high",
        "seed": 20260801,
        "max_tasks": 16,
        "max_parents_per_task": 1,
        "samples_per_parent": 1,
        "max_calls": 16,
        "max_input_tokens_per_call": 32768,
        "max_output_tokens": 8192,
        "max_input_tokens_total": 524288,
        "max_output_tokens_total": 131072,
        "max_total_tokens": 655360,
        "stability_runs": 2,
        "evaluation_only": False,
        "exploratory_terminal_prefix": 0,
        "allow_unpinned_inputs": False,
        "retry_parse_failures_or_truncations_report": "",
        "abort_on_provider_error": True,
        "prior_success_report": ["kimi", "retry", "sonnet"],
        "max_usd": "5.89824",
        "input_usd_per_million": "5",
        "output_usd_per_million": "25",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_profile_and_budget_are_exact() -> None:
    opus.validate_profile(_profile())
    capacity, contract = base.schedule_capacity(
        max_calls=16,
        max_input_tokens_per_call=32768,
        max_output_tokens_per_call=8192,
        max_input_tokens_total=524288,
        max_output_tokens_total=131072,
        max_total_tokens=655360,
        max_usd=Decimal("5.89824"),
        input_usd_per_million=Decimal("5"),
        output_usd_per_million=Decimal("25"),
    )
    assert capacity == 16
    assert contract["worst_case_usd_nanos_per_call"] == 368640000
    with pytest.raises(ValueError, match="max_output_tokens"):
        opus.validate_profile(_profile(max_output_tokens=16384))
    with pytest.raises(ValueError, match="token-price reservation"):
        opus.validate_profile(_profile(max_usd="4.60"))


def test_launcher_is_manual_plan_first_direct_only_and_secret_free() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/training/t5gemma2_typed_opus_retry_harvest.py"
    launcher = root / "deploy/vast/t5gemma2_typed_opus_retry_harvest.sh"
    conf = root / "deploy/vast/t5gemma2-typed-opus-retry-harvest.conf"
    text = launcher.read_text(encoding="utf-8")
    assert sha256_file(script) in text
    assert text.index("--plan_only_output") < text.index("anthropic_key=")
    assert "--model claude-opus-5" in text
    assert "--anthropic_thinking adaptive --anthropic_effort high" in text
    assert "--max_tasks 16" in text and "--max_calls 16" in text
    assert "--max_input_tokens_per_call 32768" in text
    assert "--max_output_tokens 8192" in text
    assert "--max_usd 5.89824" in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "exec nice -n 10 /venv/main/bin/python" in text
    assert opus.EXPECTED_RETRY_TASK_IDS_SHA256 in text
    assert "repair_policy_targets" not in text
    assert "sk-ant-" not in text and "chat-pasted" not in text
    conf_text = conf.read_text(encoding="utf-8")
    assert "autostart=false" in conf_text
    assert "autorestart=false" in conf_text
    assert "exitcodes=0,78" in conf_text
