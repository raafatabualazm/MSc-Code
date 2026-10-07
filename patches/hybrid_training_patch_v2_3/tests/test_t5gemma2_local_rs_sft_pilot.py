from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    journal_record,
    load_journal,
    sha256_file,
)
from scripts.preprocessing.build_verpo_feedback_view import SPLIT_SCHEMA
from scripts.training.t5gemma2_enriched_sft import (
    F2_ROW_SCHEMA,
    REPRESENTATION_SCHEMA,
    _REQUIRED_F2_ATTESTATIONS,
)
from scripts.training.t5gemma2_local_rs_sft_pilot import (
    JOURNAL_SCHEMA,
    RUN_SCHEMA,
    Evaluation,
    PilotTask,
    PrivateGate,
    build_matched_training_rows,
    deterministic_pilot_indices,
    deterministic_residual_indices,
    load_excluded_api_verified_tasks,
    load_excluded_verified_tasks,
    load_pilot_inputs,
    parse_args,
    process_task,
    validate_journal_state,
)
from scripts.training.seq2seq_verpo_core import canonical_json, sha256_text


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_rows(task_id: str = "fit-1") -> tuple[dict[str, Any], ...]:
    visible = "void main() {\n  expect(fn0(2), 4);\n}\n"
    holdback = "void main() {\n  expect(fn0(3), 6);\n}\n"
    metadata = {
        "tests_sha256": "a" * 64,
        "case_count": 2,
        "visible_count": 1,
        "holdback_count": 1,
        "visible_case_indices": [0],
        "holdback_case_indices": [1],
    }
    binding = sha256_text(canonical_json(metadata))
    rollout = {
        "task_id": task_id,
        "dart_source": "int fn0(int x) => x * 2;",
        "feedback_tests": visible,
        "verpo_feedback_split_schema": SPLIT_SCHEMA,
        "verpo_feedback_split_binding_sha256": binding,
    }
    text = "F2 TEST-FREE REPRESENTATION"
    f2 = {
        "schema": F2_ROW_SCHEMA,
        "representation_schema": REPRESENTATION_SCHEMA,
        "task_id": task_id,
        "text": text,
        "text_sha256": sha256_text(text),
        "verified": dict(_REQUIRED_F2_ATTESTATIONS),
    }
    private = {
        "task_id": task_id,
        "schema": SPLIT_SCHEMA,
        **metadata,
        "feedback_tests": visible,
        "reward_holdback_tests": holdback,
    }
    return rollout, f2, private


def _task(task_id: str = "fit-1") -> PilotTask:
    source = f"source::{task_id}"
    gold = f"int fn0_{task_id.replace('-', '_')}(int x) => x;"
    return PilotTask(
        task_id=task_id,
        source=source,
        source_sha256=sha256_text(source),
        visible_tests="VISIBLE",
        gold_target=gold,
        gold_target_sha256=sha256_text(gold),
        f2_row={"task_id": task_id, "text": source},
        split_binding_sha256="b" * 64,
    )


def test_load_pilot_inputs_binds_original_f2_and_complementary_gate(
    tmp_path: Path,
) -> None:
    rollout, f2, private = _fixture_rows()
    rollout_path = tmp_path / "rollout.jsonl"
    f2_path = tmp_path / "f2.jsonl"
    private_path = tmp_path / "private.jsonl"
    _write_jsonl(rollout_path, [rollout])
    _write_jsonl(f2_path, [f2])
    _write_jsonl(private_path, [private])

    tasks, gates, record = load_pilot_inputs(
        rollout_file=rollout_path,
        f2_jsonl=f2_path,
        private_holdback=private_path,
        allow_unpinned_inputs=True,
    )

    assert len(tasks) == 1
    assert tasks[0].task_id == "fit-1"
    assert tasks[0].gold_target == rollout["dart_source"]
    assert tasks[0].f2_row == f2
    assert gates["fit-1"].tests == private["reward_holdback_tests"]
    assert record["private_path_serialized"] is False
    assert "path" not in record["private_holdback"]


def test_load_pilot_inputs_rejects_private_field_in_public_rollout(
    tmp_path: Path,
) -> None:
    rollout, f2, private = _fixture_rows()
    rollout["tests"] = "leak"
    paths = [tmp_path / name for name in ("rollout.jsonl", "f2.jsonl", "private.jsonl")]
    for path, row in zip(paths, (rollout, f2, private), strict=True):
        _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="leaked to rollout"):
        load_pilot_inputs(
            rollout_file=paths[0],
            f2_jsonl=paths[1],
            private_holdback=paths[2],
            allow_unpinned_inputs=True,
        )


def test_process_task_completes_all_generation_before_private_gate() -> None:
    task = _task()
    gate = PrivateGate(
        task_id=task.task_id,
        tests="PRIVATE_SENTINEL",
        split_binding_sha256=task.split_binding_sha256,
    )
    trace: list[tuple[str, str]] = []
    repair_calls = 0

    def generate(source: str, count: int, seed: int) -> list[dict[str, Any]]:
        nonlocal repair_calls
        trace.append(("generate", source))
        if "COMPILER_REPAIR_CONTEXT_JSON" not in source:
            return [{"text": f"bad_{index}", "seed": seed} for index in range(count)]
        call = repair_calls
        repair_calls += 1
        return [
            {"text": f"fixed_{call}_{index}", "seed": seed}
            for index in range(count)
        ]

    def evaluate(code: str, tests: str, slot: str) -> Evaluation:
        trace.append(("private" if tests == gate.tests else "visible", code))
        if tests == gate.tests:
            return Evaluation(True, code == "fixed_0_1", "SECRET_DIAGNOSTIC")
        if code.startswith("bad_"):
            return Evaluation(False, False, "Error: Expected ';' at <path>")
        return Evaluation(True, True, "")

    event = process_task(
        task=task,
        gate=gate,
        task_position=0,
        seed=42,
        base_samples=2,
        repair_samples=2,
        max_repair_parents=2,
        evaluation_workers=1,
        generate=generate,
        evaluate=evaluate,
    )

    first_private = next(index for index, item in enumerate(trace) if item[0] == "private")
    assert all(item[0] != "generate" for item in trace[first_private:])
    assert all("PRIVATE_SENTINEL" not in source for kind, source in trace if kind == "generate")
    assert event["selected_target"]["code"] == "fixed_0_1"
    serialized = canonical_json(event)
    assert "PRIVATE_SENTINEL" not in serialized
    assert "SECRET_DIAGNOSTIC" not in serialized
    assert event["all_generation_completed_before_private_gate"] is True
    assert len(event["repair_groups"]) == 2


def test_private_failure_only_rejects_and_never_triggers_repair() -> None:
    task = _task()
    gate = PrivateGate(task.task_id, "PRIVATE", task.split_binding_sha256)
    generated_sources: list[str] = []

    def generate(source: str, count: int, seed: int) -> list[dict[str, Any]]:
        generated_sources.append(source)
        return [{"text": f"visible_pass_{index}"} for index in range(count)]

    def evaluate(code: str, tests: str, slot: str) -> Evaluation:
        return Evaluation(True, tests != gate.tests, "private failure details")

    event = process_task(
        task=task,
        gate=gate,
        task_position=0,
        seed=1,
        base_samples=2,
        repair_samples=2,
        max_repair_parents=1,
        evaluation_workers=1,
        generate=generate,
        evaluate=evaluate,
    )

    assert generated_sources == [task.source]
    assert event["repair_groups"] == []
    assert event["selected_target"] is None
    assert all(
        result["private_gate_passed"] is False
        for result in event["private_gate_results"]
    )


def test_process_task_supports_explicit_base_only_mode() -> None:
    task = _task()
    gate = PrivateGate(task.task_id, "PRIVATE", task.split_binding_sha256)
    generated_sources: list[str] = []

    def generate(source: str, count: int, seed: int) -> list[dict[str, Any]]:
        generated_sources.append(source)
        return [
            {"text": "does_not_compile"},
            {"text": "visible_and_private_pass"},
        ]

    def evaluate(code: str, tests: str, slot: str) -> Evaluation:
        if code == "visible_and_private_pass":
            return Evaluation(True, True, "")
        return Evaluation(False, False, "Error: Expected ';' at <path>")

    event = process_task(
        task=task,
        gate=gate,
        task_position=0,
        seed=42,
        base_samples=2,
        repair_samples=0,
        max_repair_parents=0,
        evaluation_workers=1,
        generate=generate,
        evaluate=evaluate,
    )

    assert generated_sources == [task.source]
    assert event["repair_groups"] == []
    assert event["visible_unique_passes"] == 1
    assert event["selected_target"]["origin"] == "base"


@pytest.mark.parametrize(
    ("repair_samples", "max_repair_parents"),
    ((0, 1), (1, 0)),
)
def test_process_task_rejects_half_disabled_repair_mode(
    repair_samples: int, max_repair_parents: int
) -> None:
    task = _task()
    gate = PrivateGate(task.task_id, "PRIVATE", task.split_binding_sha256)

    with pytest.raises(ValueError, match="both be zero or both positive"):
        process_task(
            task=task,
            gate=gate,
            task_position=0,
            seed=42,
            base_samples=2,
            repair_samples=repair_samples,
            max_repair_parents=max_repair_parents,
            evaluation_workers=1,
            generate=lambda *_: [],
            evaluate=lambda *_: Evaluation(False, False, ""),
        )


def test_schedule_offset_extends_seeded_pilot_without_overlap() -> None:
    tasks = [_task(f"fit-{index}") for index in range(30)]
    first = deterministic_pilot_indices(tasks, seed=42, limit=5)
    expanded = deterministic_pilot_indices(
        tasks, seed=42, offset=5, limit=20
    )
    entire = deterministic_pilot_indices(tasks, seed=42, limit=25)

    assert not set(first) & set(expanded)
    assert [*first, *expanded] == entire


def _completed_base_only_harvest(
    tmp_path: Path, tasks: list[PilotTask], accepted_ids: set[str]
) -> tuple[Path, Path]:
    journal = tmp_path / "source.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "sampling": {
            "base_samples": 1,
            "repair_samples": 0,
            "max_repair_parents": 0,
        },
    }
    append_event(
        journal,
        {
            "event": "header",
            "schema": JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": sha256_text(canonical_json(contract)),
        },
    )
    for position, task in enumerate(tasks):
        gate = PrivateGate(task.task_id, f"PRIVATE-{task.task_id}", task.split_binding_sha256)

        def generate(_source: str, count: int, _seed: int, *, task_id: str = task.task_id) -> list[dict[str, Any]]:
            return [{"text": "pass" if task_id in accepted_ids else "fail"} for _ in range(count)]

        def evaluate(code: str, _tests: str, _slot: str) -> Evaluation:
            return Evaluation(code == "pass", code == "pass", "")

        append_event(
            journal,
            process_task(
                task=task,
                gate=gate,
                task_position=position,
                seed=42,
                base_samples=1,
                repair_samples=0,
                max_repair_parents=0,
                evaluation_workers=1,
                generate=generate,
                evaluate=evaluate,
            ),
        )
    append_event(
        journal,
        {
            "event": "complete",
            "schema": JOURNAL_SCHEMA,
            "tasks": len(tasks),
            "terminal_task_ids_sha256": sha256_text(
                canonical_json([task.task_id for task in tasks])
            ),
        },
    )
    report = tmp_path / "harvest_report.json"
    report.write_text(
        canonical_json(
            {
                "schema": "t5gemma2-local-rs-sft-pilot-report-v1",
                "status": "complete",
                "pilot": {
                    "tasks": len(tasks),
                    "accepted_unique_targets": len(accepted_ids),
                },
                "journal": journal_record(journal),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return report, journal


def test_residual_schedule_and_exclusion_are_pinned_to_private_acceptance(
    tmp_path: Path,
) -> None:
    tasks = [_task(f"fit-{index}") for index in range(4)]
    report, journal = _completed_base_only_harvest(
        tmp_path, tasks[:2], {"fit-0"}
    )
    excluded, records = load_excluded_verified_tasks(
        reports=[report],
        journals=[journal],
        expected_report_sha256=[sha256_file(report)],
        expected_journal_sha256=[sha256_file(journal)],
        tasks=tasks,
    )
    assert excluded == {"fit-0"}
    assert records[0]["accepted_unique_targets"] == 1
    residual = deterministic_residual_indices(
        tasks, excluded_task_ids=excluded, seed=42
    )
    assert {tasks[index].task_id for index in residual} == {
        "fit-1",
        "fit-2",
        "fit-3",
    }


def test_excluded_harvest_rejects_report_journal_mismatch(tmp_path: Path) -> None:
    tasks = [_task("fit-0")]
    report, journal = _completed_base_only_harvest(tmp_path, tasks, {"fit-0"})
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["journal"]["sha256"] = "0" * 64
    report.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind its journal"):
        load_excluded_verified_tasks(
            reports=[report],
            journals=[journal],
            expected_report_sha256=[sha256_file(report)],
            expected_journal_sha256=[sha256_file(journal)],
            tasks=tasks,
        )


def test_api_exclusion_requires_pinned_sealed_private_targets(tmp_path: Path) -> None:
    tasks = [_task("fit-0"), _task("fit-1")]
    journal = tmp_path / "api.journal.jsonl"
    append_event(journal, {"event": "header", "schema": "api-test"})
    targets = tmp_path / "direct_targets.jsonl"
    _write_jsonl(
        targets,
        [
            {
                "schema": "t5gemma2-api-rs-sft-direct-target-v1",
                "task_id": "fit-1",
                "private_gate_passed": True,
                "production_floor_eligible": True,
            }
        ],
    )
    report = tmp_path / "api_report.json"
    report.write_text(
        canonical_json(
            {
                "schema": "t5gemma2-api-rs-sft-rescue-report-v1",
                "status": "complete",
                "journal": journal_record(journal),
                "outputs": {
                    "direct_targets": {"sha256": sha256_file(targets), "rows": 1}
                },
                "verification": {"verified_unique_hard_targets": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    excluded, records = load_excluded_api_verified_tasks(
        reports=[report],
        journals=[journal],
        target_files=[targets],
        expected_report_sha256=[sha256_file(report)],
        expected_journal_sha256=[sha256_file(journal)],
        expected_target_sha256=[sha256_file(targets)],
        tasks=tasks,
    )
    assert excluded == {"fit-1"}
    assert records[0]["accepted_unique_targets"] == 1


def test_cli_accepts_only_coherent_base_only_repair_counts() -> None:
    required = [
        "--rollout_file",
        "rollout.jsonl",
        "--f2_jsonl",
        "f2.jsonl",
        "--private_holdback",
        "private.jsonl",
        "--sft_checkpoint",
        "checkpoint",
        "--output_dir",
        "output",
    ]
    args = parse_args(
        [
            *required,
            "--pilot_offset",
            "200",
            "--repair_samples",
            "0",
            "--max_repair_parents",
            "0",
        ]
    )
    assert args.pilot_offset == 200
    assert args.repair_samples == 0
    assert args.max_repair_parents == 0
    with pytest.raises(SystemExit):
        parse_args(
            [
                *required,
                "--repair_samples",
                "0",
                "--max_repair_parents",
                "1",
            ]
        )


def test_matched_training_rows_keep_original_f2_and_strong_gold_replay() -> None:
    tasks = [_task(f"fit-{index}") for index in range(5)]
    accepted = tasks[0]
    terminals = [
        {
            "task_id": accepted.task_id,
            "selected_target": {
                "code": "int repaired(int x) => x + 1;",
                "origin": "compiler_repair",
            },
        }
    ]

    rows = build_matched_training_rows(
        all_tasks=tasks,
        terminals=terminals,
        gold_replay_ratio=3,
        seed=42,
    )

    assert len(rows["repairs"]) == 1
    assert len(rows["intervention"]) == 4
    assert len(rows["control"]) == 4
    assert len(rows["matched_f2"]) == 4
    ids = [row["task_id"] for row in rows["intervention"]]
    assert len(ids) == len(set(ids))
    assert ids == [row["task_id"] for row in rows["control"]]
    assert ids == [row["task_id"] for row in rows["matched_f2"]]
    accepted_position = ids.index(accepted.task_id)
    assert rows["intervention"][accepted_position]["dart_source"].startswith(
        "int repaired"
    )
    assert (
        rows["control"][accepted_position]["dart_source"]
        == accepted.gold_target
    )
    assert sum(row["kind"] == "gold_replay" for row in rows["schedule"]) == 3


def test_hash_chained_journal_resumes_only_the_exact_schedule(
    tmp_path: Path,
) -> None:
    task = _task()
    contract = {
        "schema": RUN_SCHEMA,
        "seed": 42,
        "sampling": {
            "base_samples": 1,
            "repair_samples": 0,
            "max_repair_parents": 0,
        },
    }
    journal = tmp_path / "harvest.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": sha256_text(canonical_json(contract)),
        },
    )
    terminal = {
        "event": "task_terminal",
        "schema": JOURNAL_SCHEMA,
        "task_position": 0,
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "split_binding_sha256": task.split_binding_sha256,
        "base_candidates": [
            {
                "origin": "base",
                "sample_index": 0,
                "code": "bad",
                "code_sha256": sha256_text("bad"),
                "generation": {},
                "visible": {"compiled": False, "passed": False},
            }
        ],
        "repair_groups": [],
        "visible_unique_passes": 0,
        "private_gate_results": [],
        "selected_target": None,
        "all_generation_completed_before_private_gate": True,
        "private_feedback_serialized_to_model": False,
        "holdback_failure_triggers_generation": False,
    }
    append_event(journal, terminal)
    append_event(
        journal,
        {
            "event": "complete",
            "schema": JOURNAL_SCHEMA,
            "tasks": 1,
            "terminal_task_ids_sha256": sha256_text(
                canonical_json([task.task_id])
            ),
        },
    )

    terminals, complete = validate_journal_state(
        load_journal(journal), contract=contract, scheduled_tasks=[task]
    )
    assert complete is True
    assert len(terminals) == 1
    with pytest.raises(ValueError, match="header differs"):
        validate_journal_state(
            load_journal(journal),
            contract={**contract, "seed": 43},
            scheduled_tasks=[task],
        )
