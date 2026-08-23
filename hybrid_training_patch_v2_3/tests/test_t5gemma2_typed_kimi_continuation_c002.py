from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_api_rescue_continuation_c002 as adapter
from scripts.training import t5gemma2_typed_kimi_continuation_c002 as c002


def _record(
    *,
    phase: str,
    cohort: int,
    scheduled: list[str],
    verified: list[str],
    retry: list[str],
    spent: str,
) -> dict:
    return {
        "path": f"/{phase}-{cohort}.json",
        "phase": phase,
        "cohort_index": cohort,
        "report_sha256": ("a" if cohort == 0 else "b") * 64,
        "journal_sha256": "c" * 64,
        "targets_sha256": "d" * 64,
        "targets_path": f"/{phase}-{cohort}.jsonl",
        "scheduled_task_ids": scheduled,
        "verified_task_ids": verified,
        "retry_eligible_task_ids": retry,
        "spent": Decimal(spent),
    }


def _prior_records() -> list[dict]:
    c0 = [f"old-{index}" for index in range(50)]
    c1 = [f"seen-{index}" for index in range(50)]
    return [
        _record(
            phase=cascade.PHASE_KIMI_INITIAL,
            cohort=0,
            scheduled=c0,
            verified=c0[:15],
            retry=c0[-2:],
            spent="1.20",
        ),
        _record(
            phase=cascade.PHASE_KIMI_RETRY,
            cohort=0,
            scheduled=c0[-2:],
            verified=c0[-1:],
            retry=[],
            spent="0.30",
        ),
        _record(
            phase=cascade.PHASE_SONNET_RESIDUAL,
            cohort=0,
            scheduled=c0[30:] + ["sonnet-only"],
            verified=["sonnet-only"],
            retry=[],
            spent="2.00",
        ),
        _record(
            phase=cascade.PHASE_KIMI_INITIAL,
            cohort=1,
            scheduled=c1,
            verified=c1[:9],
            retry=c1[-2:],
            spent="1.00",
        ),
        _record(
            phase=cascade.PHASE_KIMI_RETRY,
            cohort=1,
            scheduled=c1[-2:],
            verified=c1[-1:],
            retry=[],
            spent="0.20",
        ),
    ]


def test_budget_constants_and_adaptive_initial_cap() -> None:
    assert c002.PREFERRED_INITIAL_WORST_USD == Decimal("5.5296")
    assert c002.FALLBACK_INITIAL_WORST_USD == Decimal("3.9936")
    assert c002.RETRY_WORST_USD_PER_TASK == Decimal("0.172032")


@pytest.mark.parametrize(
    ("available", "expected_cap", "expected_reserve"),
    [
        (Decimal("6.00"), 4096, Decimal("5.5296")),
        (Decimal("5.00"), 2048, Decimal("3.9936")),
    ],
)
def test_run_selects_sealed_cap_and_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: Decimal,
    expected_cap: int,
    expected_reserve: Decimal,
) -> None:
    prior = _prior_records()
    launcher = tmp_path / "phase.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    c001_report = tmp_path / "c001.json"
    c001_report.write_text("{}", encoding="utf-8")
    prior_record = {
        "full_prior_index_tsv_path": str(tmp_path / "prior.tsv"),
        "full_prior_index_tsv_sha256": "e" * 64,
    }
    calls: list[tuple[str, int, Decimal]] = []

    monkeypatch.setattr(
        c002,
        "load_completed_c001",
        lambda **_kwargs: (prior, available, prior_record),
    )

    fresh = [f"fresh-{index}" for index in range(50)]

    def fake_execute_phase(**kwargs):
        calls.append(
            (
                kwargs["phase"],
                kwargs["max_output_tokens"],
                kwargs["max_usd"],
            )
        )
        if kwargs["phase"] == cascade.PHASE_KIMI_INITIAL:
            return _record(
                phase=cascade.PHASE_KIMI_INITIAL,
                cohort=2,
                scheduled=fresh,
                verified=fresh[:8],
                retry=fresh[-2:],
                spent="1.00",
            )
        return _record(
            phase=cascade.PHASE_KIMI_RETRY,
            cohort=2,
            scheduled=fresh[-2:],
            verified=fresh[-1:],
            retry=[],
            spent="0.20",
        )

    monkeypatch.setattr(c002, "execute_phase", fake_execute_phase)
    monkeypatch.setattr(
        c002.dual,
        "publish_aggregate",
        lambda _root, records: {
            "direct_only": True,
            "rows": sum(len(row["verified_task_ids"]) for row in records),
        },
    )
    monkeypatch.setattr(
        c002.dual,
        "publish_prior_index",
        lambda *_args: {"schema": "test", "status": "complete"},
    )
    args = c002.parse_args(
        [
            "--phase-launcher",
            str(launcher),
            "--output-root",
            str(tmp_path / "out"),
            "--prior-continuation-report",
            str(c001_report),
            "--expected-prior-continuation-report-sha256",
            "f" * 64,
        ]
    )
    report = c002.run(args, base_env={})
    assert calls[0] == (cascade.PHASE_KIMI_INITIAL, expected_cap, expected_reserve)
    assert calls[1][0:2] == (cascade.PHASE_KIMI_RETRY, 8192)
    assert report["retry"]["complete_exact_set_executed"] is True
    assert report["retry"]["partial_retry_executed"] is False
    assert report["budget"]["cohort2_charged_usd"] == "1.20"


def test_run_refuses_when_even_2048_reservation_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "phase.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(
        c002,
        "load_completed_c001",
        lambda **_kwargs: (_prior_records(), Decimal("3.99"), {}),
    )
    args = c002.parse_args(
        [
            "--phase-launcher",
            str(launcher),
            "--output-root",
            str(tmp_path / "out"),
            "--prior-continuation-report",
            str(tmp_path / "unused.json"),
            "--expected-prior-continuation-report-sha256",
            "f" * 64,
        ]
    )
    with pytest.raises(ValueError, match="strict 50-call"):
        c002.run(args, base_env={})


def test_adapter_excludes_union_of_all_prior_schedules(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        (index, SimpleNamespace(task_id=f"task-{index}"), {}) for index in range(120)
    ]
    prior = [
        {"scheduled_task_ids": [f"task-{index}" for index in range(10)]},
        {"scheduled_task_ids": ["task-10", "task-11", "task-12"]},
    ]
    monkeypatch.setattr(
        adapter,
        "_ORIGINAL_PHASE_SELECTION",
        lambda **_kwargs: (rows[:50], {"mode": cascade.PHASE_KIMI_INITIAL}),
    )
    selected, record = adapter.phase_selection(
        args=SimpleNamespace(phase=cascade.PHASE_KIMI_INITIAL, max_tasks=50),
        all_visible_zero=rows,
        prior_records=prior,
    )
    ids = [row[1].task_id for row in selected]
    assert ids == [f"task-{index}" for index in range(13, 63)]
    assert record["prior_all_provider_scheduled_tasks_excluded"] == 13
    assert record["selected_task_ids_sha256"] == canonical_sha256(ids)


def test_adapter_seals_4096_for_base_validation_and_restores_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cascade.KIMI_INITIAL_MAX_OUTPUT
    args = SimpleNamespace(
        phase=cascade.PHASE_KIMI_INITIAL,
        cohort_index=2,
        fixed_kimi_cohort_limit=3,
        max_input_tokens_per_call=16384,
        max_output_tokens=4096,
    )
    def fake_parse(_argv):
        # This is the ordering regression: base argument validation must see
        # the c002 override, not the cascade's original 2K constant.
        assert cascade.KIMI_INITIAL_MAX_OUTPUT == 4096
        return args

    monkeypatch.setattr(cascade, "parse_args", fake_parse)

    def fake_run(_args):
        assert cascade.KIMI_INITIAL_MAX_OUTPUT == 4096
        return {"status": "complete"}

    monkeypatch.setattr(cascade, "run", fake_run)
    assert adapter.run(
        [
            "--phase",
            cascade.PHASE_KIMI_INITIAL,
            "--max_output_tokens",
            "4096",
        ]
    ) == {"status": "complete"}
    assert cascade.KIMI_INITIAL_MAX_OUTPUT == original


def test_launchers_are_nonconcurrent_gpu_isolated_and_secret_free() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "deploy/vast/t5gemma2_typed_kimi_continuation_c002.sh").read_text(
        encoding="utf-8"
    )
    phase = (root / "deploy/vast/t5gemma2_typed_api_rescue_continuation_c002.sh").read_text(
        encoding="utf-8"
    )
    handoff = (
        root / "deploy/vast/t5gemma2_typed_kimi_continuation_c002_handoff.sh"
    ).read_text(encoding="utf-8")
    conf = (
        root / "deploy/vast/t5gemma2-typed-kimi-continuation-c002-handoff.conf"
    ).read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES=""' in launcher
    assert 'exec nice -n 10 "${PYTHON_BIN}"' in launcher
    assert "OPENROUTER_API_KEY=" not in launcher
    assert "cohort_index 2" in phase
    assert "fixed_kimi_cohort_limit 3" in phase
    assert 'INITIAL_MAX_OUTPUT}" == 4096' in phase
    assert "pgrep -f '[t]5gemma2_typed_kimi_continuation.py'" in handoff
    assert "sleep 30" in handoff
    assert "autostart=false" in conf
    assert "autorestart=false" in conf
    assert "exitcodes=0,78" in conf
