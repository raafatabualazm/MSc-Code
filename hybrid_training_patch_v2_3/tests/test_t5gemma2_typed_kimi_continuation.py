from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_dual_api_orchestrator as dual
from scripts.training import t5gemma2_typed_kimi_continuation as continuation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _phase_fixture(
    output_dir: Path,
    *,
    phase: str,
    cohort: int,
    scheduled_ids: list[str],
    verified_ids: list[str],
    retry_ids: list[str],
    spent: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": cascade.RUN_SCHEMA,
        "phase": phase,
        "cohort_index": cohort,
        "training_outputs": {
            "direct_verified_code_targets": True,
            "reasoning_rows": 0,
            "repair_conditioned_rows": 0,
        },
    }
    journal = output_dir / "typed_api_rescue.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": cascade.JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    rows = []
    verified = set(verified_ids)
    retry = set(retry_ids)
    for index, task_id in enumerate(scheduled_ids):
        append_event(journal, {"event": "call_intent", "task_id": task_id})
        append_event(
            journal,
            {
                "event": "call_result",
                "task_id": task_id,
                "parse_accepted": task_id not in retry,
                "response": {
                    "finish_reason": "length" if task_id in retry else "stop"
                },
            },
        )
        selected = None
        if task_id in verified:
            code = f"int fn0(int p0) => {index};"
            selected = {"task_id": task_id}
            rows.append(
                {
                    "schema": cascade.DIRECT_TARGET_SCHEMA,
                    "task_id": task_id,
                    "source_sha256": "a" * 64,
                    "dart_source": code,
                    "dart_source_sha256": base.sha256_text(code),
                    "origin": "external_teacher_direct_verified",
                    "provider_phase": phase,
                    "provider_model": cascade.KIMI_MODEL,
                    "visible_train_passed": True,
                    "private_full_acceptance_passed": True,
                    "stability_runs": cascade.STABILITY_RUNS,
                    "reasoning_present": False,
                    "repair_conditioned_training_source_present": False,
                    "gold_replay": False,
                    "provenance": {
                        "run_contract_sha256": canonical_sha256(contract)
                    },
                }
            )
        append_event(
            journal,
            {
                "event": "task_verification",
                "task_id": task_id,
                "selected_target": selected,
            },
        )
    targets = output_dir / "direct_targets.jsonl"
    _write_jsonl(targets, rows)
    target_record = {
        "path": str(targets),
        "sha256": sha256_file(targets),
        "rows": len(rows),
    }
    manifest = {
        "schema": cascade.DIRECT_MANIFEST_SCHEMA,
        "run_contract_sha256": canonical_sha256(contract),
        "rows": len(rows),
        "targets": target_record,
        "direct_only": True,
        "repair_conditioned_rows": 0,
        "gold_replay_rows": 0,
        "reasoning_rows": 0,
        "tests_in_training_output": False,
        "private_feedback_in_training_output": False,
        "stability_runs": cascade.STABILITY_RUNS,
        "task_ids_sha256": canonical_sha256(verified_ids),
    }
    require_exact_or_write(output_dir / "direct_manifest.json", manifest)
    report = {
        "schema": cascade.REPORT_SCHEMA,
        "status": "complete",
        "phase": phase,
        "cohort_index": cohort,
        "run_contract_sha256": canonical_sha256(contract),
        "schedule": {
            "scheduled_tasks": len(scheduled_ids),
            "task_ids_sha256": canonical_sha256(scheduled_ids),
            "retry_eligible_non_code_or_length_tasks": len(retry_ids),
            "retry_eligible_task_ids_sha256": canonical_sha256(retry_ids),
        },
        "budget_charged": {"estimated_usd": spent, "within_contract": True},
        "outputs": {"direct_targets": target_record},
        "direct_manifest": manifest,
        "repair_policy_manifest": None,
        "journal": journal_record(journal),
    }
    report_path = output_dir / "typed_api_rescue_report.json"
    require_exact_or_write(report_path, report)
    return dual.inspect_phase_report(
        report_path, expected_phase=phase, expected_cohort=cohort
    )


def _prior_run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "prior"
    initial_ids = [f"prior-{index}" for index in range(50)]
    retry_ids = initial_ids[35:50]
    records = [
        _phase_fixture(
            root / "kimi_initial_c000",
            phase=cascade.PHASE_KIMI_INITIAL,
            cohort=0,
            scheduled_ids=initial_ids,
            verified_ids=initial_ids[:15],
            retry_ids=retry_ids,
            spent="1.248090000",
        ),
        _phase_fixture(
            root / "kimi_retry_c000",
            phase=cascade.PHASE_KIMI_RETRY,
            cohort=0,
            scheduled_ids=retry_ids,
            verified_ids=retry_ids[:2],
            retry_ids=[],
            spent="0.854985000",
        ),
        _phase_fixture(
            root / "sonnet_residual_c000",
            phase=cascade.PHASE_SONNET_RESIDUAL,
            cohort=0,
            scheduled_ids=[f"prior-{index}" for index in range(17, 55)],
            verified_ids=["prior-53", "prior-54"],
            retry_ids=[],
            spent="4.296234000",
        ),
    ]
    index = dual.publish_prior_index(root, "final", records)
    report = {
        "schema": dual.REPORT_SCHEMA,
        "status": "complete",
        "providers": {
            "openrouter": {"charged_usd": "2.103075000"},
            "anthropic": {"charged_usd": "4.296234000"},
        },
        "phases": [
            {
                key: row[key]
                for key in (
                    "phase",
                    "cohort_index",
                    "report_sha256",
                    "journal_sha256",
                    "targets_sha256",
                )
            }
            for row in records
        ],
        "prior_report_index": index,
        "direct_manifest": {
            "direct_only": True,
            "rows": 19,
            "reasoning_rows": 0,
            "repair_conditioned_rows": 0,
            "gold_replay_rows": 0,
        },
        "heldout_175_model_visible": False,
        "heldout_175_used_for_generation_or_selection": False,
    }
    report_path = root / "orchestration_report.json"
    require_exact_or_write(report_path, report)
    return report_path, Path(index["tsv"]["path"])


def test_continuation_runs_one_50_task_cohort_and_exact_retry(tmp_path: Path) -> None:
    prior_report, prior_index = _prior_run(tmp_path)
    launcher = tmp_path / "phase.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    actual_phases: list[str] = []

    def invoke(_launcher: Path, env: dict[str, str]) -> None:
        assert env["T5GEMMA_TYPED_API_COHORT_INDEX"] == "1"
        assert env["T5GEMMA_TYPED_API_FIXED_KIMI_COHORT_LIMIT"] == "2"
        assert env["T5GEMMA_TYPED_API_MAX_INPUT_TOKENS_PER_CALL"] == "16384"
        phase = env["T5GEMMA_TYPED_API_PHASE"]
        if phase == cascade.PHASE_KIMI_INITIAL:
            ids = [f"new-{index}" for index in range(50)]
        else:
            ids = ["new-48", "new-49"]
        plan_path = env.get("T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT")
        if plan_path:
            assert "OPENROUTER_API_KEY" not in env
            require_exact_or_write(
                plan_path,
                {
                    "schema": cascade.PLAN_SCHEMA,
                    "status": "complete",
                    "phase": phase,
                    "cohort_index": 1,
                    "fixed_kimi_cohort_limit": 2,
                    "selection": {
                        "scheduled_tasks": len(ids),
                        "task_ids_sha256": canonical_sha256(ids),
                    },
                    "provider_credentials_read": False,
                    "frontier_api_calls": False,
                },
            )
            return
        actual_phases.append(phase)
        if phase == cascade.PHASE_KIMI_INITIAL:
            verified, retry, spent = ids[:9], ids[48:], "1.250000000"
        else:
            verified, retry, spent = ids[:1], [], "0.200000000"
        _phase_fixture(
            Path(env["T5GEMMA_TYPED_API_OUTPUT_DIR"]),
            phase=phase,
            cohort=1,
            scheduled_ids=ids,
            verified_ids=verified,
            retry_ids=retry,
            spent=spent,
        )

    args = continuation.parse_args(
        [
            "--phase-launcher",
            str(launcher),
            "--output-root",
            str(tmp_path / "continuation"),
            "--prior-orchestration-report",
            str(prior_report),
            "--expected-prior-orchestration-report-sha256",
            sha256_file(prior_report),
            "--prior-index",
            str(prior_index),
            "--expected-prior-index-sha256",
            sha256_file(prior_index),
        ]
    )
    report = continuation.run(args, invoke=invoke, base_env={})
    assert actual_phases == [
        cascade.PHASE_KIMI_INITIAL,
        cascade.PHASE_KIMI_RETRY,
    ]
    assert report["new_direct_manifest"]["rows"] == 10
    assert report["retry"]["complete_exact_set_executed"] is True
    assert report["retry"]["partial_retry_executed"] is False
    assert report["cohort_decision"]["eligible_to_continue_after_bounded_run"] is True
    assert report["cohort_decision"]["next_cohort_started"] is False
    assert report["budget"]["continuation_charged_usd"] == "1.450000000"

    # Resume validates sealed artifacts and makes no second provider call.
    continuation.run(args, invoke=invoke, base_env={})
    assert actual_phases == [
        cascade.PHASE_KIMI_INITIAL,
        cascade.PHASE_KIMI_RETRY,
    ]


def test_budget_constants_fit_remaining_credit_and_forbid_partial_retry() -> None:
    available = (
        continuation.OPENROUTER_BALANCE_BEFORE_COHORT0
        - continuation.EXPECTED_PRIOR_OPENROUTER_SPEND
    )
    assert continuation.CONTINUATION_CAP == Decimal("10.30")
    assert available == Decimal("10.336925000")
    assert continuation.INITIAL_WORST_USD == Decimal("3.9936")
    assert continuation.RETRY_WORST_USD_PER_TASK == Decimal("0.172032")
    assert continuation.CONTINUATION_CAP < available


def test_launcher_and_supervisor_are_manual_sealed_and_secret_free() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "deploy/vast/t5gemma2_typed_kimi_continuation.sh"
    ).read_text(encoding="utf-8")
    phase_launcher = (
        root / "deploy/vast/t5gemma2_typed_api_rescue_continuation.sh"
    ).read_text(encoding="utf-8")
    conf = (
        root / "deploy/vast/t5gemma2-typed-kimi-continuation.conf"
    ).read_text(encoding="utf-8")
    assert "--openrouter-balance-before-cohort0 12.44" in launcher
    assert "--continuation-max-usd 10.30" in launcher
    assert "T5GEMMA_TYPED_API_MAX_INPUT_TOKENS_PER_CALL" not in launcher
    assert "OPENROUTER_API_KEY=" not in launcher
    assert "Anthropic.env" not in launcher
    assert 'export CUDA_VISIBLE_DEVICES=""' in launcher
    assert 'exec nice -n 10 "${PYTHON_BIN}"' in launcher
    assert "--max_input_tokens_per_call 16384" in phase_launcher
    assert "t5gemma2_typed_api_rescue_continuation.py" in phase_launcher
    assert "sonnet_residual" not in phase_launcher
    assert "ANTHROPIC_API_KEY" not in phase_launcher
    assert "autostart=false" in conf
    assert "autorestart=false" in conf
    assert "exitcodes=0,78" in conf
    assert "redirect_stderr=true" in conf
    assert "stopasgroup=true" in conf
    assert "killasgroup=true" in conf
