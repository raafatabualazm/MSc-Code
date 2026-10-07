from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    require_exact_or_write,
    sha256_file,
)
from scripts.training import t5gemma2_typed_local_direct_harvest as harvest


def _task(index: int) -> harvest.HarvestTask:
    task_id = f"train-{index:04d}"
    source = f"typed source {index}"
    return harvest.HarvestTask(
        task_id=task_id,
        source=source,
        source_sha256=harvest.sha256_text(source),
        f2_row={"task_id": task_id, "text": f"F2 {index}"},
        gold_target_sha256=harvest.sha256_text(f"gold {index}"),
        typed_contract_sha256=harvest.sha256_text(f"contract {index}"),
    )


def _gate(task: harvest.HarvestTask) -> harvest.PrivateAcceptanceGate:
    tests = f"PRIVATE COMPLETE TESTS {task.task_id}"
    return harvest.PrivateAcceptanceGate(
        task_id=task.task_id,
        tests=tests,
        tests_sha256=harvest.sha256_text(tests),
    )


def _generate(_source: str, count: int, _seed: int):
    assert count == 4
    return [{"text": f"code-{index}", "finish_reason": "eos"} for index in range(4)]


def _evaluate(code: str, tests: str, _slot: str) -> harvest.Evaluation:
    assert tests.startswith("PRIVATE COMPLETE TESTS")
    if code == "code-0":
        return harvest.Evaluation(False, False, "Error: undefined name at /tmp/x.dart:9")
    return harvest.Evaluation(True, code in {"code-1", "code-3"}, "")


def test_process_task_generates_all_k_then_selects_one_without_private_payload() -> None:
    task = _task(0)
    gate = _gate(task)
    calls: list[str] = []

    def generate(source: str, count: int, seed: int):
        calls.append("generate")
        return _generate(source, count, seed)

    def evaluate(code: str, tests: str, slot: str):
        assert calls == ["generate"]
        calls.append("evaluate")
        result = _evaluate(code, tests, slot)
        calls.pop()
        return result

    event = harvest.process_task(
        task=task,
        gate=gate,
        task_position=0,
        seed=42,
        generate=generate,
        evaluate=evaluate,
        evaluation_workers=1,
    )
    assert calls == ["generate"]
    assert len(event["base_candidates"]) == 4
    assert event["repair_groups"] == []
    assert event["visible_unique_passes"] == 2
    assert event["selected_target"]["code"] == "code-1"
    assert event["selected_target"]["origin"] == "local_student_direct"
    assert event["all_generation_completed_before_private_gate"] is True
    payload = json.dumps(event, sort_keys=True)
    assert gate.tests not in payload
    assert "/tmp/x.dart" not in payload


def test_journal_resume_validation_is_exact(tmp_path: Path) -> None:
    task = _task(0)
    gate = _gate(task)
    contract = {
        "schema": harvest.RUN_SCHEMA,
        "sampling": {"samples_per_task": 4},
    }
    journal = tmp_path / "harvest.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": harvest.JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    terminal = harvest.process_task(
        task=task,
        gate=gate,
        task_position=0,
        seed=42,
        generate=_generate,
        evaluate=_evaluate,
        evaluation_workers=1,
    )
    append_event(journal, terminal)
    events = harvest.load_journal(journal)
    terminals, complete = harvest.validate_journal_state(
        events,
        contract=contract,
        scheduled_tasks=[task],
        gates={task.task_id: gate},
    )
    assert len(terminals) == 1
    assert complete is False
    append_event(
        journal,
        {
            "event": "complete",
            "schema": harvest.JOURNAL_SCHEMA,
            "tasks": 1,
            "terminal_task_ids_sha256": canonical_sha256([task.task_id]),
        },
    )
    terminals, complete = harvest.validate_journal_state(
        harvest.load_journal(journal),
        contract=contract,
        scheduled_tasks=[task],
        gates={task.task_id: gate},
    )
    assert complete is True
    tampered = [dict(row) for row in terminals]
    tampered[0]["selected_target"] = None
    with pytest.raises(ValueError, match="target differs"):
        harvest.validate_journal_state(
            [events[0], *tampered],
            contract=contract,
            scheduled_tasks=[task],
            gates={task.task_id: gate},
        )


def test_full_clean_schedule_excludes_exact_predecessor_set() -> None:
    tasks = [_task(index) for index in range(harvest.EXPECTED_CLEAN_ROWS)]
    excluded = {task.task_id for task in tasks[: harvest.EXPECTED_PREVIOUS_DIRECT_TASKS]}
    scheduled = harvest.build_schedule(tasks, excluded_task_ids=excluded, seed=42)
    assert len(scheduled) == harvest.EXPECTED_SCHEDULED_TASKS
    assert not excluded.intersection(task.task_id for task in scheduled)
    assert scheduled == harvest.build_schedule(tasks, excluded_task_ids=excluded, seed=42)
    with pytest.raises(ValueError, match="exactly 225"):
        harvest.build_schedule(tasks, excluded_task_ids=set(), seed=42)


def test_checkpoint_stage_selects_fixed_update() -> None:
    common = [
        "--gold_train_jsonl", "train.jsonl",
        "--gold_f2_jsonl", "f2.jsonl",
        "--expected_gold_train_sha256", "1" * 64,
        "--expected_gold_f2_sha256", "2" * 64,
        "--heldout_jsonl", "heldout.jsonl",
        "--expected_heldout_sha256", "3" * 64,
        "--checkpoint", "checkpoint-optstep-000348",
        "--checkpoint_stage", "typed_sft",
        "--expected_checkpoint_update", "348",
        "--expected_checkpoint_run_contract_sha256", "4" * 64,
        "--expected_checkpoint_run_contract_file_sha256", "5" * 64,
        "--expected_checkpoint_training_state_sha256", "a" * 64,
        "--expected_checkpoint_adapter_weights_sha256", "6" * 64,
        "--expected_checkpoint_adapter_config_sha256", "7" * 64,
        "--output_dir", "out",
    ]
    for index in range(4):
        common += ["--local_report", f"{'8' * 64}=local-{index}.json"]
    for index in range(7):
        common += ["--api_report", f"{'9' * 64}=api-{index}.json"]
    assert harvest.parse_args(common).expected_checkpoint_update == 348
    bad = list(common)
    bad[bad.index("348")] = "58"
    with pytest.raises(SystemExit):
        harvest.parse_args(bad)


def test_loaded_checkpoint_match_excludes_only_preflight_provenance() -> None:
    loaded = {
        "name": "google/t5gemma-2-4b-4b",
        "revision": "pinned-revision",
        "config_sha256": "1" * 64,
        "arm": "sft",
        "training_stage_schema": "typed-direct",
        "production_floor_eligible": True,
        "tokenizer_sha256": "2" * 64,
        "warmstart_contract_sha256": "3" * 64,
        "adapter": {
            "adapter_config_sha256": "4" * 64,
            "adapter_weights_sha256": "5" * 64,
            "run_contract_sha256": "3" * 64,
            "target_modules": 32,
        },
    }
    preflight = {
        **loaded,
        "checkpoint_stage": "typed_direct",
        "checkpoint_update": 58,
        "training_state_sha256": "6" * 64,
    }

    # This is the exact shape produced by validate_checkpoint followed by
    # inference.load_policy; preflight-only provenance must not cause failure.
    harvest.assert_loaded_checkpoint_matches_preflight(loaded, preflight)

    changed_adapter = json.loads(json.dumps(loaded))
    changed_adapter["adapter"]["adapter_weights_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="adapter differs from preflight"):
        harvest.assert_loaded_checkpoint_matches_preflight(
            changed_adapter, preflight
        )

    missing_provenance = dict(preflight)
    missing_provenance.pop("training_state_sha256")
    with pytest.raises(ValueError, match="lacks provenance"):
        harvest.assert_loaded_checkpoint_matches_preflight(
            loaded, missing_provenance
        )


def test_completed_artifact_adapter_reconstructs_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harvest, "EXPECTED_CLEAN_ROWS", 3)
    monkeypatch.setattr(harvest, "EXPECTED_PREVIOUS_DIRECT_TASKS", 1)
    monkeypatch.setattr(harvest, "EXPECTED_SCHEDULED_TASKS", 2)
    tasks = [_task(index) for index in range(3)]
    gates = {task.task_id: _gate(task) for task in tasks}
    input_record = {"sealed": True}
    monkeypatch.setattr(
        harvest,
        "load_harvest_inputs",
        lambda **_kwargs: (tasks, gates, input_record),
    )
    excluded = {tasks[0].task_id}
    scheduled = harvest.build_schedule(tasks, excluded_task_ids=excluded, seed=42)
    contract = {
        "schema": harvest.RUN_SCHEMA,
        "input": input_record,
        "checkpoint": {"checkpoint_stage": "typed_direct", "checkpoint_update": 58},
        "previous_direct_exclusion": {
            "rows": 1,
            "task_ids_sha256": canonical_sha256(sorted(excluded)),
        },
        "schedule": {
            "schema": harvest.SCHEDULE_SCHEMA,
            "seed": 42,
            "clean_train_rows": 3,
            "excluded_previous_direct_tasks": 1,
            "scheduled_tasks": 2,
            "task_ids_sha256": canonical_sha256([task.task_id for task in scheduled]),
            "source_sha256s_sha256": canonical_sha256(
                [task.source_sha256 for task in scheduled]
            ),
        },
    }
    journal = tmp_path / "harvest.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": harvest.JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    terminals = []
    for position, task in enumerate(scheduled):
        terminal = harvest.process_task(
            task=task,
            gate=gates[task.task_id],
            task_position=position,
            seed=42,
            generate=_generate,
            evaluate=_evaluate,
            evaluation_workers=1,
        )
        terminals.append(append_event(journal, terminal))
    append_event(
        journal,
        {
            "event": "complete",
            "schema": harvest.JOURNAL_SCHEMA,
            "tasks": 2,
            "terminal_task_ids_sha256": canonical_sha256(
                [task.task_id for task in scheduled]
            ),
        },
    )
    targets = []
    direct_f2 = []
    schedule_rows = []
    for task, terminal in zip(scheduled, terminals, strict=True):
        selected = terminal["selected_target"]
        targets.append(
            {
                "schema": harvest.TARGET_SCHEMA,
                "task_id": task.task_id,
                "dart_source": selected["code"],
                "dart_source_sha256": selected["code_sha256"],
                "source_sha256": task.source_sha256,
                "origin": "local_student_direct",
                "full_acceptance_passed": True,
                "stability_runs": 2,
                "repair_conditioned": False,
                "gold_replay": False,
            }
        )
        direct_f2.append(task.f2_row)
        schedule_rows.append(
            {
                "schema": harvest.SCHEDULE_SCHEMA,
                "position": terminal["task_position"],
                "task_id": task.task_id,
                "source_sha256": task.source_sha256,
                "typed_contract_sha256": task.typed_contract_sha256,
                "complete_acceptance_sha256": gates[task.task_id].tests_sha256,
                "candidate_code_sha256s": [
                    row["code_sha256"] for row in terminal["base_candidates"]
                ],
                "unique_full_passes": terminal["visible_unique_passes"],
                "selected_target_sha256": selected["code_sha256"],
            }
        )
    target_file = tmp_path / "direct_targets.jsonl"
    f2_file = tmp_path / "direct_f2.jsonl"
    schedule_file = tmp_path / "schedule_manifest.jsonl"
    harvest._atomic_write_jsonl(target_file, targets)
    harvest._atomic_write_jsonl(f2_file, direct_f2)
    harvest._atomic_write_jsonl(schedule_file, schedule_rows)
    report = {
        "schema": harvest.REPORT_SCHEMA,
        "status": "complete",
        "production_floor_eligible": True,
        "run_contract_sha256": canonical_sha256(contract),
        "checkpoint": contract["checkpoint"],
        "schedule": {
            "clean_train_rows": 3,
            "excluded_previous_direct_tasks": 1,
            "tasks": 2,
            "samples_per_task": 4,
            "candidate_generations": 8,
            "task_ids_sha256": canonical_sha256(
                [task.task_id for task in scheduled]
            ),
        },
        "accepted": {
            "unique_direct_targets": 2,
            "task_ids_sha256": canonical_sha256(
                [row["task_id"] for row in targets]
            ),
            "exact_gold_targets": 0,
            "at_most_one_per_task": True,
        },
        "composition": {
            "local_student_direct": 2,
            "repair_conditioned": 0,
            "gold_replay": 0,
        },
        "verification": {
            "suite": "complete_train_acceptance",
            "stability_runs": 2,
            "tests_model_visible": False,
            "tests_persisted": False,
            "diagnostics_persisted": False,
        },
        "privacy": {
            "model_visible_fields": ["opaque_typed_contract", "F2.text"],
            "complete_acceptance_model_visible": False,
            "heldout_175_opened": False,
            "frontier_api_calls": False,
        },
        "outputs": {
            "direct_targets": {
                "path": str(target_file), "sha256": sha256_file(target_file), "rows": 2
            },
            "direct_f2": {
                "path": str(f2_file), "sha256": sha256_file(f2_file), "rows": 2
            },
            "schedule_manifest": {
                "path": str(schedule_file), "sha256": sha256_file(schedule_file), "rows": 2
            },
        },
        "journal": journal_record(journal),
    }
    report_file = tmp_path / "harvest_report.json"
    require_exact_or_write(report_file, report)
    result = harvest.load_completed_harvest_artifacts(
        report_path=report_file,
        expected_report_sha256=sha256_file(report_file),
        journal_path=journal,
        expected_journal_sha256=sha256_file(journal),
        targets_path=target_file,
        expected_targets_sha256=sha256_file(target_file),
        gold_train_jsonl="unused-train",
        expected_gold_train_sha256="a" * 64,
        gold_f2_jsonl="unused-f2",
        expected_gold_f2_sha256="b" * 64,
        heldout_jsonl="unused-heldout",
        expected_heldout_sha256="c" * 64,
        expected_gold_rows=3,
        expected_heldout_rows=1,
    )
    assert [task.task_id for task in result[2]] == [task.task_id for task in scheduled]
    assert len(result[3]) == 2
    tampered = json.loads(report_file.read_text())
    tampered["composition"]["gold_replay"] = 1
    bad_report = tmp_path / "bad_report.json"
    require_exact_or_write(bad_report, tampered)
    with pytest.raises(ValueError, match="accounting differs"):
        harvest.load_completed_harvest_artifacts(
            report_path=bad_report,
            expected_report_sha256=sha256_file(bad_report),
            journal_path=journal,
            expected_journal_sha256=sha256_file(journal),
            targets_path=target_file,
            expected_targets_sha256=sha256_file(target_file),
            gold_train_jsonl="unused-train",
            expected_gold_train_sha256="a" * 64,
            gold_f2_jsonl="unused-f2",
            expected_gold_f2_sha256="b" * 64,
            heldout_jsonl="unused-heldout",
            expected_heldout_sha256="c" * 64,
            expected_gold_rows=3,
            expected_heldout_rows=1,
        )


def test_launcher_and_supervisor_are_sealed_and_do_not_autostart() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "deploy/vast/t5gemma2_typed_local_direct_harvest.sh").read_text()
    config = (root / "deploy/vast/t5gemma2-typed-local-direct-harvest.conf").read_text()
    for required in (
        "typed_direct)",
        "typed_sft)",
        "--samples_per_task 4",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--stability_runs 2",
        "0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f",
        "3cb25d54f12743ed43572b219e119667f264abab94ec4cbfac72a94407fbdfc7",
    ):
        assert required in launcher
    assert "OPENAI" not in launcher
    assert "ANTHROPIC" not in launcher
    assert "autostart=false" in config
    assert "autorestart=unexpected" in config
