from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_api_rs_sft_rescue as api_base
from scripts.training import t5gemma2_enriched_sft as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_direct_rs_sft as pass1
from scripts.training import t5gemma2_typed_direct_rs_sft_pass2 as profile
from scripts.training import t5gemma2_typed_dual_api_orchestrator as dual
from scripts.training import t5gemma2_typed_local_direct_harvest as local


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _local_row(task_id: str, value: int) -> dict:
    source = f"TYPED::{task_id}"
    code = f"int fn0() => {value};"
    return {
        "schema": local.TARGET_SCHEMA,
        "task_id": task_id,
        "dart_source": code,
        "dart_source_sha256": _sha(code),
        "source_sha256": _sha(source),
        "origin": "local_student_direct",
        "full_acceptance_passed": True,
        "stability_runs": local.EXPECTED_STABILITY_RUNS,
        "repair_conditioned": False,
        "gold_replay": False,
    }


def _api_row(
    task_id: str,
    value: int,
    *,
    provider_phase: str | None = None,
    provider_model: str | None = None,
    run_contract_sha256: str | None = None,
    slot_position: int = 0,
) -> dict:
    code = f"int fn0() => {value};"
    row = {
        "schema": cascade.DIRECT_TARGET_SCHEMA,
        "task_id": task_id,
        "source_sha256": _sha(f"TYPED::{task_id}"),
        "dart_source": code,
        "dart_source_sha256": _sha(code),
        "origin": "external_teacher_direct_verified",
        "visible_train_passed": True,
        "private_full_acceptance_passed": True,
        "stability_runs": cascade.STABILITY_RUNS,
        "reasoning_present": False,
        "repair_conditioned_training_source_present": False,
        "gold_replay": False,
    }
    if provider_phase is not None:
        row.update(
            {
                "provider_phase": provider_phase,
                "provider_model": provider_model,
                "provenance": {
                    "run_contract_sha256": run_contract_sha256,
                    "slot_position": slot_position,
                    "parent_code_sha256": "a" * 64,
                    "diagnostic_sha256": "b" * 64,
                },
            }
        )
    return row


def _synthetic_specs(
    tmp_path: Path,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[dict]]:
    local_dir = tmp_path / "local"
    api_dir = tmp_path / "api"
    local_dir.mkdir()
    api_dir.mkdir()
    local_rows = [_local_row(f"local-{index}", index) for index in range(190)]
    files = [
        local_dir / "harvest_report.json",
        local_dir / "harvest.journal.jsonl",
        local_dir / "direct_targets.jsonl",
        tmp_path / "dataset_manifest.json",
    ]
    files[0].write_text("{}\n", encoding="utf-8")
    files[1].write_text("{}\n", encoding="utf-8")
    _write_jsonl(files[2], local_rows)
    files[3].write_text("{}\n", encoding="utf-8")
    api_files = [
        api_dir / "orchestration_report.json",
        api_dir / "direct_manifest.json",
        api_dir / "direct_targets.jsonl",
    ]
    api_files[0].write_text("{}\n", encoding="utf-8")
    api_files[1].write_text("{}\n", encoding="utf-8")
    api_files[2].write_text("", encoding="utf-8")
    return (
        [(path, sha256_file(path)) for path in files],
        [(path, sha256_file(path)) for path in api_files],
        local_rows,
    )


def _patch_builder_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    local_rows: list[dict],
    api_rows: list[dict],
    *,
    prior_ids: set[str] | None = None,
    exact_gold_ids: set[str] | None = None,
) -> None:
    all_ids = [row["task_id"] for row in local_rows + api_rows]
    target_sha_by_id = {
        row["task_id"]: row["dart_source_sha256"] for row in local_rows + api_rows
    }
    exact_gold_ids = set() if exact_gold_ids is None else set(exact_gold_ids)
    tasks = [
        local.HarvestTask(
            task_id=task_id,
            source=f"TYPED::{task_id}",
            source_sha256=_sha(f"TYPED::{task_id}"),
            f2_row={},
            gold_target_sha256=(
                target_sha_by_id[task_id] if task_id in exact_gold_ids else "f" * 64
            ),
            typed_contract_sha256="e" * 64,
        )
        for task_id in dict.fromkeys(all_ids)
    ]
    gates = {
        task.task_id: local.PrivateAcceptanceGate(
            task_id=task.task_id,
            tests=f"TEST::{task.task_id}",
            tests_sha256=_sha(f"TEST::{task.task_id}"),
        )
        for task in tasks
    }
    monkeypatch.setattr(
        profile.local_harvest,
        "load_completed_harvest_artifacts",
        lambda **kwargs: (
            tasks,
            gates,
            [],
            [],
            {"schema": "typed-input"},
            {"schema": "local-audit"},
        ),
    )
    monkeypatch.setattr(
        profile,
        "_load_prior_225_manifest",
        lambda path, digest: (
            set() if prior_ids is None else set(prior_ids),
            {"rows": 225, "sha256": digest, "used_for_training_in_this_stage": False},
        ),
    )
    monkeypatch.setattr(
        profile,
        "audit_completed_dual_orchestration",
        lambda **kwargs: (
            api_rows,
            {
                "schema": "dual-audit",
                "status": "complete",
                "targets": {"rows": len(api_rows)},
                "row_count_late_bound_after_complete_audit": True,
            },
        ),
    )
    monkeypatch.setattr(
        profile.mixed,
        "_load_heldout_ids",
        lambda *args, **kwargs: (set(), {"rows": 175, "content_model_visible": False}),
    )
    monkeypatch.setattr(
        profile.pass1,
        "_verify_all",
        lambda pairs, **kwargs: {
            "rows": len(pairs),
            "passed": len(pairs),
            "tests_model_visible": False,
        },
    )


def _build(local_specs, api_specs):
    return profile.build_typed_direct_pass2_pairs(
        gold_train_jsonl=Path("train.jsonl"),
        gold_f2_jsonl=Path("f2.jsonl"),
        expected_gold_train_sha256="0" * 64,
        expected_gold_f2_sha256="1" * 64,
        expected_gold_rows=2776,
        heldout_jsonl=Path("heldout.jsonl"),
        expected_heldout_sha256="2" * 64,
        expected_heldout_rows=175,
        local_reports=local_specs,
        api_reports=api_specs,
        warmstart=SimpleNamespace(),
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=190,
        min_repair_conditioned_targets=0,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        seed=42,
    )


def test_pass2_builds_only_new_direct_rows_and_late_binds_api_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_specs, api_specs, local_rows = _synthetic_specs(tmp_path)
    api_rows = [_api_row("api-0", 900), _api_row("api-1", 901)]
    _patch_builder_dependencies(monkeypatch, local_rows, api_rows)
    pairs, manifest = _build(local_specs, api_specs)
    assert len(pairs) == 192
    assert len({pair.source_task_id for pair in pairs}) == 192
    assert all(pair.kind == "verified_direct" for pair in pairs)
    assert manifest["composition"] == {
        "verified_direct": 192,
        "local_student_new": 190,
        "external_teacher_new": 2,
        "prior_225_replay": 0,
        "gold_replay": 0,
        "repair_conditioned": 0,
        "reasoning_rows": 0,
        "independently_generated_exact_gold_matches": 0,
        "gold_source_replay": 0,
    }
    assert (
        manifest["api_row_count_policy"]
        == "late_bound_after_audited_complete_orchestration"
    )
    assert manifest["tests_model_visible"] is False
    assert manifest["heldout_overlap"] == 0
    assert manifest["all_targets_bound_to_generation_journals"] is True


def test_independently_generated_exact_gold_match_is_not_gold_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_specs, api_specs, local_rows = _synthetic_specs(tmp_path)
    _patch_builder_dependencies(
        monkeypatch,
        local_rows,
        [],
        exact_gold_ids={"local-0"},
    )
    pairs, manifest = _build(local_specs, api_specs)
    assert len(pairs) == 190
    assert manifest["composition"]["independently_generated_exact_gold_matches"] == 1
    assert manifest["composition"]["gold_source_replay"] == 0
    assert "cryptographically_bound" in manifest["exact_gold_match_policy"]


def test_pass2_rejects_cross_source_and_prior_225_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_specs, api_specs, local_rows = _synthetic_specs(tmp_path)
    duplicate_api = [_api_row("local-0", 999)]
    _patch_builder_dependencies(monkeypatch, local_rows, duplicate_api)
    with pytest.raises(ValueError, match="overlap by task_id"):
        _build(local_specs, api_specs)

    _patch_builder_dependencies(
        monkeypatch, local_rows, [_api_row("api-0", 999)], prior_ids={"local-7"}
    )
    with pytest.raises(ValueError, match="prior-225"):
        _build(local_specs, api_specs)


def test_pass2_accepts_audited_empty_api_aggregate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_specs, api_specs, local_rows = _synthetic_specs(tmp_path)
    _patch_builder_dependencies(monkeypatch, local_rows, [])
    pairs, manifest = _build(local_specs, api_specs)
    assert len(pairs) == 190
    assert manifest["composition"]["external_teacher_new"] == 0


def _phase_fixture(
    output_dir: Path,
    phase: str,
    task_prefix: str,
    spent: str,
    *,
    scheduled_tasks: int,
    input_record: dict,
    source_journal_record: dict,
    prior_records: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = [f"{task_prefix}-{index:03d}" for index in range(scheduled_tasks)]
    provider = {
        **profile._phase_provider_expectation(phase),
        "credential_source": "environment_value_not_persisted",
        "one_candidate_per_call": True,
    }
    phase_source_journal = {
        **source_journal_record,
        "mode": cascade.LOCAL_SOURCE_MODE,
        "exploratory_prefix": False,
        "production_floor_eligible": True,
        "terminal_prefix_length": None,
        "source_journal_modified": False,
    }
    contract = {
        "schema": cascade.RUN_SCHEMA,
        "script_sha256": profile.EXPECTED_CASCADE_PRODUCER_SHA256,
        "phase": phase,
        "cohort_index": 0,
        "source_local_harvest_journal": phase_source_journal,
        "inputs": {
            "source": {
                "typed_local_harvest": {
                    **input_record,
                    "existing_225_manifest": {"schema": pass1.DATASET_SCHEMA},
                },
                "permitted_visible_train_split": {"schema": "synthetic-visible"},
                "task_ids_sha256": "3" * 64,
                "complete_acceptance_text_serialized": False,
                "private_holdback_text_serialized": False,
            },
            "visible_failure_projection": {"schema": "synthetic-projection"},
            "existing_225_exclusion": {"schema": pass1.DATASET_SCHEMA},
        },
        "prior_reports": prior_records,
        "selection": {
            "scheduled_tasks": scheduled_tasks,
            "scheduled_slots": scheduled_tasks,
            "task_ids_sha256": canonical_sha256(task_ids),
            "max_parents_per_task": 1,
            "samples_per_parent": 1,
        },
        "provider": provider,
        "verification": {
            "visible_before_private": True,
            "all_api_calls_before_any_private_gate": True,
            "private_gate": "complete_TRAIN_acceptance",
            "stability_runs": cascade.STABILITY_RUNS,
            "private_failure_triggers_api_call": False,
            "private_gate_can_only_reject_transfer": True,
        },
        "privacy": {
            "private_complete_acceptance_sent_to_provider": False,
            "private_split_holdback_sent_to_provider": False,
            "gold_sent_to_provider": False,
            "heldout_175_opened": False,
        },
        "training_outputs": {
            "direct_verified_code_targets": True,
            "repair_conditioned_rows": 0,
            "gold_replay_rows": 0,
            "reasoning_rows": 0,
            "tests_in_training_outputs": False,
            "production_floor_eligible": True,
        },
        "heldout_175_opened": False,
    }
    contract_sha = canonical_sha256(contract)
    journal = output_dir / "typed_api_rescue.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": cascade.JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": contract_sha,
        },
    )
    result_rows: list[dict] = []
    for position, task_id in enumerate(task_ids):
        code = f"int fn0() => {position + 1};"
        binding = {
            "slot_position": position,
            "task_position": position,
            "task_id": task_id,
            "parent_position": 0,
            "sample_index": 0,
            "parent_code_sha256": "a" * 64,
            "diagnostic_sha256": "b" * 64,
            "feedback_source_sha256": "c" * 64,
            "prompt_sha256": "d" * 64,
        }
        append_event(
            journal,
            {"event": "call_intent", "schema": cascade.JOURNAL_SCHEMA, **binding},
        )
        result = {
            "event": "call_result",
            "schema": cascade.JOURNAL_SCHEMA,
            **binding,
            "status": "response",
            "parse_accepted": True,
            "code": code,
            "code_sha256": _sha(code),
            "response": {"finish_reason": "stop"},
            "usage": {
                "charged_input_tokens": 0,
                "charged_output_tokens": 0,
                "charged_usd_nanos": (
                    int(Decimal(spent) * Decimal(1_000_000_000)) if position == 0 else 0
                ),
            },
        }
        append_event(journal, result)
        result_rows.append(result)
    selected_task = task_ids[0]
    selected_code = result_rows[0]["code"]
    selected = {
        "schema": cascade.DIRECT_TARGET_SCHEMA,
        "task_id": selected_task,
        "source_sha256": _sha(f"TYPED::{selected_task}"),
        "code": selected_code,
        "code_sha256": _sha(selected_code),
        "slot_position": 0,
        "parent_position": 0,
        "parent_code_sha256": "a" * 64,
        "diagnostic_sha256": "b" * 64,
        "feedback_source_sha256": "c" * 64,
        "visible_passed": True,
        "private_gate_passed": True,
        "exploratory_prefix": False,
        "production_floor_eligible": True,
        "training_use_forbidden": False,
    }
    for position, task_id in enumerate(task_ids):
        source_sha = _sha(f"TYPED::{task_id}")
        event_selected = selected if position == 0 else None
        visible = []
        private = []
        if position == 0:
            visible = [
                {
                    "slot_position": 0,
                    "code": selected_code,
                    "code_sha256": _sha(selected_code),
                    "passed": True,
                }
            ]
            private = [
                {
                    "slot_position": 0,
                    "code_sha256": _sha(selected_code),
                    "private_gate_passed": True,
                }
            ]
        append_event(
            journal,
            {
                "event": "task_verification",
                "schema": cascade.JOURNAL_SCHEMA,
                "task_position": position,
                "task_id": task_id,
                "source_sha256": source_sha,
                "visible_results": visible,
                "private_gate_results": private,
                "selected_target": event_selected,
                "all_api_generation_completed_before_private_gate": True,
                "private_feedback_serialized_to_model": False,
                "holdback_failure_triggers_generation": False,
                "private_diagnostics_persisted": False,
            },
        )
    append_event(
        journal,
        {
            "event": "complete",
            "schema": cascade.JOURNAL_SCHEMA,
            "tasks": scheduled_tasks,
            "slots": scheduled_tasks,
            "verified_targets": 1,
            "exploratory_prefix": False,
            "production_floor_eligible": True,
        },
    )
    rows = [
        _api_row(
            selected_task,
            1,
            provider_phase=phase,
            provider_model=provider["model"],
            run_contract_sha256=contract_sha,
        )
    ]
    targets = output_dir / "direct_targets.jsonl"
    _write_jsonl(targets, rows)
    target_record = {"path": str(targets), "sha256": sha256_file(targets), "rows": 1}
    manifest = {
        "schema": cascade.DIRECT_MANIFEST_SCHEMA,
        "run_contract_sha256": contract_sha,
        "rows": 1,
        "targets": target_record,
        "direct_only": True,
        "repair_conditioned_rows": 0,
        "gold_replay_rows": 0,
        "reasoning_rows": 0,
        "tests_in_training_output": False,
        "private_feedback_in_training_output": False,
        "full_acceptance_reverified": True,
        "stability_runs": cascade.STABILITY_RUNS,
        "task_ids_sha256": canonical_sha256([selected_task]),
        "production_floor_eligible": True,
    }
    require_exact_or_write(output_dir / "direct_manifest.json", manifest)
    report = {
        "schema": cascade.REPORT_SCHEMA,
        "status": "complete",
        "phase": phase,
        "cohort_index": 0,
        "run_contract_sha256": contract_sha,
        "provider": provider,
        "schedule": {
            "api_eligible_tasks": scheduled_tasks,
            "scheduled_tasks": scheduled_tasks,
            "scheduled_calls": scheduled_tasks,
            "task_ids_sha256": canonical_sha256(task_ids),
            "provider_responses": scheduled_tasks,
            "code_only_responses": scheduled_tasks,
            "retry_eligible_non_code_or_length_tasks": 0,
            "retry_eligible_task_ids_sha256": canonical_sha256([]),
        },
        "verification": {
            "visible_passes": 1,
            "private_full_acceptance_passes": 1,
            "verified_unique_hard_targets": 1,
            "verified_task_ids_sha256": canonical_sha256([selected_task]),
        },
        "budget_charged": {
            "calls": scheduled_tasks,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_usd_nanos": int(Decimal(spent) * Decimal(1_000_000_000)),
            "estimated_usd": spent,
            "within_contract": True,
        },
        "outputs": {"direct_targets": target_record},
        "direct_manifest": manifest,
        "repair_policy_manifest": None,
        "journal": journal_record(journal),
        "privacy_invariants": contract["privacy"],
        "heldout_175_opened": False,
    }
    require_exact_or_write(output_dir / "typed_api_rescue_report.json", report)


def test_completed_dual_audit_precedes_and_binds_variable_row_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = tmp_path / "t5gemma2_typed_api_rescue_cascade.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(
        profile, "EXPECTED_PHASE_LAUNCHER_SHA256", sha256_file(launcher)
    )
    root = tmp_path / "dual"
    input_record = {"schema": "synthetic-typed-input", "sha256": "1" * 64}
    source_journal_record = {
        "schema": local.JOURNAL_SCHEMA,
        "sha256": "2" * 64,
        "source_journal_modified": False,
    }

    def prior_records(env: dict[str, str]) -> list[dict]:
        index_path = env.get("T5GEMMA_TYPED_API_PRIOR_INDEX")
        if not index_path:
            return []
        result = []
        for line in Path(index_path).read_text(encoding="utf-8").splitlines():
            report_sha, raw_path = line.split("\t", 1)
            phase_report = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            result.append(
                {
                    "report_sha256": report_sha,
                    "phase": phase_report["phase"],
                    "cohort_index": phase_report["cohort_index"],
                    "journal_sha256": phase_report["journal"]["sha256"],
                    "targets_sha256": phase_report["outputs"]["direct_targets"][
                        "sha256"
                    ],
                }
            )
        return result

    def invoke(_launcher: Path, env: dict[str, str]) -> None:
        phase = env["T5GEMMA_TYPED_API_PHASE"]
        plan_path = env.get("T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT")
        max_tasks = int(env["T5GEMMA_TYPED_API_MAX_TASKS"])
        task_ids = [f"{phase}-verified-{index:03d}" for index in range(max_tasks)]
        if plan_path:
            require_exact_or_write(
                plan_path,
                {
                    "schema": cascade.PLAN_SCHEMA,
                    "status": "complete",
                    "phase": phase,
                    "cohort_index": 0,
                    "fixed_kimi_cohort_limit": 1,
                    "selection": {
                        "scheduled_tasks": max_tasks,
                        "task_ids_sha256": canonical_sha256(task_ids),
                    },
                    "provider_credentials_read": False,
                    "frontier_api_calls": False,
                },
            )
            return
        _phase_fixture(
            Path(env["T5GEMMA_TYPED_API_OUTPUT_DIR"]),
            phase,
            f"{phase}-verified",
            "1.0",
            scheduled_tasks=max_tasks,
            input_record=input_record,
            source_journal_record=source_journal_record,
            prior_records=prior_records(env),
        )

    args = SimpleNamespace(
        phase_launcher=str(launcher),
        output_root=str(root),
        initial_schedule_sha256=canonical_sha256(
            [f"kimi_initial-verified-{index:03d}" for index in range(50)]
        ),
        openrouter_max_usd=Decimal("12.0"),
        anthropic_max_usd=Decimal("11.5"),
    )
    dual.run(args, invoke=invoke, base_env={})
    report = root / "orchestration_report.json"
    manifest = root / "direct_manifest.json"
    targets = root / "direct_targets.jsonl"
    rows, audit = profile.audit_completed_dual_orchestration(
        report_path=report,
        report_sha256=sha256_file(report),
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        targets_path=targets,
        targets_sha256=sha256_file(targets),
        expected_input_record=input_record,
        expected_source_journal_record=source_journal_record,
    )
    assert len(rows) == 2
    assert audit["status"] == "complete"
    assert audit["row_count_late_bound_after_complete_audit"] is True

    original_report = report.read_text(encoding="utf-8")
    tampered = json.loads(original_report)
    tampered["phases"] = list(reversed(tampered["phases"]))
    report.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="phase order"):
        profile.audit_completed_dual_orchestration(
            report_path=report,
            report_sha256=sha256_file(report),
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            targets_path=targets,
            targets_sha256=sha256_file(targets),
            expected_input_record=input_record,
            expected_source_journal_record=source_journal_record,
        )

    report.write_text(original_report, encoding="utf-8")
    tampered = json.loads(original_report)
    tampered["status"] = "running"
    report.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate contract"):
        profile.audit_completed_dual_orchestration(
            report_path=report,
            report_sha256=sha256_file(report),
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            targets_path=targets,
            targets_sha256=sha256_file(targets),
            expected_input_record=input_record,
            expected_source_journal_record=source_journal_record,
        )


def test_phase_audit_rejects_target_not_selected_in_paid_call_journal(
    tmp_path: Path,
) -> None:
    input_record = {"schema": "synthetic-typed-input", "sha256": "1" * 64}
    source_record = {"schema": local.JOURNAL_SCHEMA, "sha256": "2" * 64}
    output_dir = tmp_path / "phase"
    _phase_fixture(
        output_dir,
        cascade.PHASE_KIMI_INITIAL,
        "paid-call",
        "1.0",
        scheduled_tasks=1,
        input_record=input_record,
        source_journal_record=source_record,
        prior_records=[],
    )
    injected = _api_row(
        "paid-call-000",
        999,
        provider_phase=cascade.PHASE_KIMI_INITIAL,
        provider_model=cascade.KIMI_MODEL,
        run_contract_sha256=json.loads(
            (output_dir / "typed_api_rescue_report.json").read_text(encoding="utf-8")
        )["run_contract_sha256"],
    )
    target_path = output_dir / "direct_targets.jsonl"
    _write_jsonl(target_path, [injected])
    target_record = {
        "path": str(target_path),
        "sha256": sha256_file(target_path),
        "rows": 1,
    }
    manifest_path = output_dir / "direct_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["targets"] = target_record
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "typed_api_rescue_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outputs"]["direct_targets"] = target_record
    report["direct_manifest"] = manifest
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    # The old shallow audit accepted this fully re-hashed injection because it
    # compared only counts/flags, never code against selected_target.
    dual.inspect_phase_report(
        report_path,
        expected_phase=cascade.PHASE_KIMI_INITIAL,
        expected_cohort=0,
    )
    with pytest.raises(ValueError, match="target reconstruction"):
        profile._audit_phase_evidence(
            report_path,
            expected_phase=cascade.PHASE_KIMI_INITIAL,
            expected_cohort=0,
            expected_input_record=input_record,
            expected_source_journal_record=source_record,
            expected_prior_records=[],
        )


def test_phase_audit_rejects_unbound_provider_identity(tmp_path: Path) -> None:
    input_record = {"schema": "synthetic-typed-input", "sha256": "1" * 64}
    source_record = {"schema": local.JOURNAL_SCHEMA, "sha256": "2" * 64}
    output_dir = tmp_path / "phase"
    _phase_fixture(
        output_dir,
        cascade.PHASE_KIMI_INITIAL,
        "provider-call",
        "1.0",
        scheduled_tasks=1,
        input_record=input_record,
        source_journal_record=source_record,
        prior_records=[],
    )
    report_path = output_dir / "typed_api_rescue_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provider"] = {**report["provider"], "model": "unreviewed-model"}
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    dual.inspect_phase_report(
        report_path,
        expected_phase=cascade.PHASE_KIMI_INITIAL,
        expected_cohort=0,
    )
    with pytest.raises(ValueError, match="provider/model"):
        profile._audit_phase_evidence(
            report_path,
            expected_phase=cascade.PHASE_KIMI_INITIAL,
            expected_cohort=0,
            expected_input_record=input_record,
            expected_source_journal_record=source_record,
            expected_prior_records=[],
        )


def test_profile_and_launcher_are_fail_closed() -> None:
    args = argparse.Namespace(
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=190,
        min_repair_conditioned_targets=0,
        expected_warmstart_update=58,
        epochs=2,
        batch_size=1,
        gradient_accumulation=8,
        max_updates=0,
        learning_rate=2e-5,
        warmup_ratio=0.0,
        seed=42,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        local_report=["x"] * 4,
        api_report=["x"] * 3,
    )
    profile._validate_profile_args(args)
    args.gold_replay_rows = 1
    with pytest.raises(ValueError, match="gold_replay_rows"):
        profile._validate_profile_args(args)

    launcher = (
        PATCH_ROOT / "deploy" / "vast" / "t5gemma2_typed_direct_rs_sft_pass2.sh"
    ).read_text(encoding="utf-8")
    config = (
        PATCH_ROOT / "deploy" / "vast" / "t5gemma2-typed-direct-rs-sft-pass2.conf"
    ).read_text(encoding="utf-8")
    assert "checkpoint-optstep-000058" in launcher
    assert "--expected_warmstart_update 58" in launcher
    assert "--min_verified_direct_targets 190" in launcher
    assert "--gold_replay_rows 0" in launcher
    assert launcher.count("--local_report") == 4
    assert launcher.count("--api_report") == 3
    assert "API_REPORT_SHA256" in launcher and "orchestration_report.json" in launcher
    assert (
        "1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3" in launcher
    )
    assert "T5GEMMA_TYPED_DIRECT_RS_SFT_PASS2_ALREADY_COMPLETE" not in launcher
    assert '"$(dirname "${resolved_resume}")" == "${resolved_output}"' in launcher
    assert "malformed checkpoint pointer" in launcher
    assert "[program:t5gemma2-typed-direct-rs-sft-pass2]" in config
    assert (
        sha256_file(Path(dual.__file__).resolve())
        == profile.EXPECTED_DUAL_PRODUCER_SHA256
    )
    assert (
        sha256_file(Path(cascade.__file__).resolve())
        == profile.EXPECTED_CASCADE_PRODUCER_SHA256
    )
    assert (
        sha256_file(
            PATCH_ROOT / "deploy" / "vast" / "t5gemma2_typed_api_rescue_cascade.sh"
        )
        == profile.EXPECTED_PHASE_LAUNCHER_SHA256
    )
    assert profile.RUN_SCHEMA in inference.SUPPORTED_ADAPTER_RUN_SCHEMAS


def test_warmstart_validator_temporarily_uses_all_pass1_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = (
        profile.mixed.RUN_SCHEMA,
        profile.mixed.CHECKPOINT_SCHEMA,
        profile.mixed.DATASET_SCHEMA,
    )
    profile.mixed.RUN_SCHEMA = profile.RUN_SCHEMA
    profile.mixed.CHECKPOINT_SCHEMA = profile.CHECKPOINT_SCHEMA
    profile.mixed.DATASET_SCHEMA = profile.DATASET_SCHEMA

    def fake(checkpoint: Path, **kwargs):
        assert checkpoint == Path("checkpoint-optstep-000058")
        assert profile.mixed.RUN_SCHEMA == pass1.RUN_SCHEMA
        assert profile.mixed.CHECKPOINT_SCHEMA == pass1.CHECKPOINT_SCHEMA
        assert profile.mixed.DATASET_SCHEMA == pass1.DATASET_SCHEMA
        return SimpleNamespace(update=58), {"schema": pass1.RUN_SCHEMA}

    monkeypatch.setattr(profile, "_MIXED_VALIDATE_WARMSTART", fake)
    try:
        identity, contract = profile.validate_pass2_warmstart(
            Path("checkpoint-optstep-000058"),
            expected_update=58,
            expected_run_contract_sha256="0" * 64,
            expected_adapter_weights_sha256="1" * 64,
            expected_adapter_config_sha256="2" * 64,
            model="model",
            model_revision="revision",
        )
        assert identity.update == 58 and contract["schema"] == pass1.RUN_SCHEMA
        assert (
            profile.mixed.RUN_SCHEMA,
            profile.mixed.CHECKPOINT_SCHEMA,
            profile.mixed.DATASET_SCHEMA,
        ) == (profile.RUN_SCHEMA, profile.CHECKPOINT_SCHEMA, profile.DATASET_SCHEMA)
    finally:
        (
            profile.mixed.RUN_SCHEMA,
            profile.mixed.CHECKPOINT_SCHEMA,
            profile.mixed.DATASET_SCHEMA,
        ) = original
