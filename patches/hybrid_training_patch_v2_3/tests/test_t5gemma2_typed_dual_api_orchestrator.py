from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_dual_api_orchestrator as orchestration


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _phase_fixture(
    output_dir: Path,
    *,
    phase: str,
    scheduled: int,
    spent: str,
    verified_indices: tuple[int, ...] = (0,),
    retry_indices: tuple[int, ...] = (),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": cascade.RUN_SCHEMA,
        "phase": phase,
        "cohort_index": 0,
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
    retry_ids = []
    scheduled_ids = []
    for index in range(scheduled):
        task_id = f"{phase}-{index}"
        scheduled_ids.append(task_id)
        code = f"int fn0(int p0) => {index};"
        append_event(journal, {"event": "call_intent", "task_id": task_id})
        parse_accepted = index not in retry_indices
        finish = "length" if index in retry_indices else "stop"
        append_event(
            journal,
            {
                "event": "call_result",
                "task_id": task_id,
                "parse_accepted": parse_accepted,
                "response": {"finish_reason": finish},
            },
        )
        selected = None
        if index in verified_indices:
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
                    "provider_model": cascade.KIMI_MODEL if phase.startswith("kimi") else cascade.SONNET_MODEL,
                    "visible_train_passed": True,
                    "private_full_acceptance_passed": True,
                    "stability_runs": 2,
                    "reasoning_present": False,
                    "repair_conditioned_training_source_present": False,
                    "gold_replay": False,
                    "provenance": {"run_contract_sha256": canonical_sha256(contract)},
                }
            )
        elif index in retry_indices:
            retry_ids.append(task_id)
        append_event(
            journal,
            {"event": "task_verification", "task_id": task_id, "selected_target": selected},
        )
    targets = output_dir / "direct_targets.jsonl"
    _write_jsonl(targets, rows)
    target_record = {"path": str(targets), "sha256": sha256_file(targets), "rows": len(rows)}
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
        "stability_runs": 2,
        "task_ids_sha256": canonical_sha256([row["task_id"] for row in rows]),
    }
    require_exact_or_write(output_dir / "direct_manifest.json", manifest)
    report = {
        "schema": cascade.REPORT_SCHEMA,
        "status": "complete",
        "phase": phase,
        "cohort_index": 0,
        "run_contract_sha256": canonical_sha256(contract),
        "schedule": {
            "scheduled_tasks": scheduled,
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
    require_exact_or_write(output_dir / "typed_api_rescue_report.json", report)


def test_phase_inspection_limits_retry_to_parse_or_length(tmp_path: Path) -> None:
    output = tmp_path / "phase"
    _phase_fixture(
        output,
        phase=cascade.PHASE_KIMI_INITIAL,
        scheduled=4,
        spent="1.25",
        verified_indices=(0,),
        retry_indices=(2,),
    )
    record = orchestration.inspect_phase_report(
        output / "typed_api_rescue_report.json",
        expected_phase=cascade.PHASE_KIMI_INITIAL,
        expected_cohort=0,
    )
    assert record["retry_eligible_task_ids"] == ["kimi_initial-2"]
    assert record["spent"] == Decimal("1.25")
    (output / "repair_policy_targets.jsonl").write_text("leak", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        orchestration.inspect_phase_report(
            output / "typed_api_rescue_report.json",
            expected_phase=cascade.PHASE_KIMI_INITIAL,
            expected_cohort=0,
        )


def test_prior_index_is_hash_pinned_and_immutable(tmp_path: Path) -> None:
    record = {
        "phase": "kimi_initial",
        "cohort_index": 0,
        "path": str(tmp_path / "r.json"),
        "report_sha256": "a" * 64,
        "journal_sha256": "b" * 64,
        "targets_sha256": "c" * 64,
    }
    index = orchestration.publish_prior_index(tmp_path, "sealed", [record])
    assert sha256_file(index["tsv"]["path"]) == index["tsv"]["sha256"]
    with pytest.raises(ValueError, match="sealed artifact differs"):
        orchestration.publish_prior_index(tmp_path, "sealed", [{**record, "report_sha256": "d" * 64}])


def test_dual_orchestration_plans_without_credentials_and_resumes(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    output_root = tmp_path / "run"
    calls: list[dict[str, str]] = []

    def invoke(_launcher: Path, env: dict[str, str]) -> None:
        calls.append(dict(env))
        phase = env["T5GEMMA_TYPED_API_PHASE"]
        max_tasks = int(env["T5GEMMA_TYPED_API_MAX_TASKS"])
        plan_path = env.get("T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT")
        schedule_sha = "a" * 64 if phase == cascade.PHASE_KIMI_INITIAL else "b" * 64
        if plan_path:
            assert "OPENROUTER_API_KEY" not in env
            assert "ANTHROPIC_API_KEY" not in env
            require_exact_or_write(
                plan_path,
                {
                    "schema": cascade.PLAN_SCHEMA,
                    "status": "complete",
                    "phase": phase,
                    "cohort_index": 0,
                    "fixed_kimi_cohort_limit": 1,
                    "selection": {"scheduled_tasks": max_tasks, "task_ids_sha256": schedule_sha},
                    "provider_credentials_read": False,
                    "frontier_api_calls": False,
                },
            )
            return
        assert env["T5GEMMA_TYPED_API_SCHEDULE_SHA256"] == schedule_sha
        _phase_fixture(
            Path(env["T5GEMMA_TYPED_API_OUTPUT_DIR"]),
            phase=phase,
            scheduled=max_tasks,
            spent="1.0" if phase == cascade.PHASE_KIMI_INITIAL else "2.0",
        )

    args = SimpleNamespace(
        phase_launcher=str(launcher),
        output_root=str(output_root),
        initial_schedule_sha256="a" * 64,
        openrouter_max_usd=Decimal("12.0"),
        anthropic_max_usd=Decimal("11.5"),
    )
    report = orchestration.run(args, invoke=invoke, base_env={})
    assert report["status"] == "complete"
    assert report["direct_manifest"]["rows"] == 2
    assert report["direct_manifest"]["reasoning_rows"] == 0
    assert [row["phase"] for row in report["phases"]] == [
        cascade.PHASE_KIMI_INITIAL,
        cascade.PHASE_SONNET_RESIDUAL,
    ]
    assert sum("T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT" not in call for call in calls) == 2
    orchestration.run(args, invoke=invoke, base_env={})
    assert sum("T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT" not in call for call in calls) == 2


def test_unaffordable_exact_kimi_retry_skips_safely_to_sonnet(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    actual_phases: list[str] = []
    sonnet_skip_count = 0

    def invoke(_launcher: Path, env: dict[str, str]) -> None:
        nonlocal sonnet_skip_count
        phase = env["T5GEMMA_TYPED_API_PHASE"]
        max_tasks = int(env["T5GEMMA_TYPED_API_MAX_TASKS"])
        plan_path = env.get("T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT")
        schedule_sha = "a" * 64 if phase == cascade.PHASE_KIMI_INITIAL else "b" * 64
        if plan_path:
            require_exact_or_write(
                plan_path,
                {
                    "schema": cascade.PLAN_SCHEMA,
                    "status": "complete",
                    "phase": phase,
                    "cohort_index": 0,
                    "fixed_kimi_cohort_limit": 1,
                    "selection": {"scheduled_tasks": max_tasks, "task_ids_sha256": schedule_sha},
                    "provider_credentials_read": False,
                    "frontier_api_calls": False,
                },
            )
            if phase == cascade.PHASE_SONNET_RESIDUAL:
                sonnet_skip_count = int(
                    env["T5GEMMA_TYPED_API_BUDGET_SKIPPED_KIMI_RETRY_TASKS"]
                )
            return
        actual_phases.append(phase)
        _phase_fixture(
            Path(env["T5GEMMA_TYPED_API_OUTPUT_DIR"]),
            phase=phase,
            scheduled=max_tasks,
            spent="11.90" if phase == cascade.PHASE_KIMI_INITIAL else "1.0",
            retry_indices=(1,) if phase == cascade.PHASE_KIMI_INITIAL else (),
        )

    report = orchestration.run(
        SimpleNamespace(
            phase_launcher=str(launcher),
            output_root=str(tmp_path / "run"),
            initial_schedule_sha256="a" * 64,
            openrouter_max_usd=Decimal("12.0"),
            anthropic_max_usd=Decimal("11.5"),
        ),
        invoke=invoke,
        base_env={},
    )
    assert actual_phases == [
        cascade.PHASE_KIMI_INITIAL,
        cascade.PHASE_SONNET_RESIDUAL,
    ]
    assert sonnet_skip_count == 1
    assert report["kimi_retry"]["budget_skipped_tasks"] == 1
    assert report["providers"]["openrouter"]["charged_usd"] == "11.90"
    assert report["heldout_175_opened_for_exclusion_audit"] is True
    assert report["heldout_175_model_visible"] is False


def test_phase_profiles_enforce_reduced_provider_caps() -> None:
    common = dict(
        phase=cascade.PHASE_SONNET_RESIDUAL,
        fixed_kimi_cohort_limit=1,
        max_parents_per_task=1,
        samples_per_parent=1,
        stability_runs=2,
        evaluation_only=False,
        exploratory_terminal_prefix=0,
        allow_unpinned_inputs=False,
        provider="anthropic",
        model=cascade.SONNET_MODEL,
        max_output_tokens=16384,
        anthropic_thinking="adaptive",
        anthropic_effort="high",
        retry_parse_failures_or_truncations_report="",
        max_calls=38,
        max_tasks=38,
        max_input_tokens_per_call=65536,
        max_input_tokens_total=65536 * 38,
        max_output_tokens_total=16384 * 38,
        max_total_tokens=(65536 + 16384) * 38,
        max_usd="11.5",
        input_usd_per_million="2",
        output_usd_per_million="10",
    )
    cascade.validate_phase_profile(SimpleNamespace(**common))
    with pytest.raises(ValueError, match="11.50"):
        cascade.validate_phase_profile(SimpleNamespace(**{**common, "max_usd": "11.51"}))


def test_dual_launcher_and_supervisor_are_sealed_and_manual() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "deploy/vast/t5gemma2_typed_dual_api_orchestrator.sh").read_text()
    conf = (root / "deploy/vast/t5gemma2-typed-dual-api-orchestrator.conf").read_text()
    assert "--openrouter-max-usd 12.0" in launcher
    assert "--anthropic-max-usd 11.5" in launcher
    assert "T5GEMMA_TYPED_API_SCHEDULE_SHA256" in launcher
    assert "OPENROUTER_API_KEY" not in launcher
    assert "ANTHROPIC_API_KEY" not in launcher
    assert "autostart=false" in conf
    assert "autorestart=false" in conf
