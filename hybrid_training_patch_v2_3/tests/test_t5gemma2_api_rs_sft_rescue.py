from __future__ import annotations

import argparse
import sys
import types
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from scripts.training import t5gemma2_api_rs_sft_rescue as rescue_module
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.training.seq2seq_verpo_core import sha256_text
from scripts.training.t5gemma2_api_rs_sft_rescue import (
    COMPLETED_LOCAL_RUN_MODE,
    DIRECT_TARGET_SCHEMA,
    EXPLORATORY_PREFIX_RUN_MODE,
    JOURNAL_SCHEMA,
    REPORT_SCHEMA,
    RUN_SCHEMA,
    ApiSlot,
    PendingProviderCall,
    ProviderResponse,
    RescueParent,
    RescuePlan,
    RetryableProviderPayloadError,
    SYSTEM_PROMPT,
    OpenAITransport,
    _provider_contract,
    _publish_training_outputs,
    build_provider_prompt,
    build_slots,
    cap_plans_to_budget,
    execute_api_phase,
    execute_verification_phase,
    exclude_prior_verified_plans,
    freeze_local_terminal_prefix,
    load_retry_parse_failures_or_truncations_source,
    load_prior_success_exclusions,
    parse_code_only,
    parse_args,
    schedule_capacity,
    select_rescue_plans,
    slice_rescue_plans,
    validate_rescue_journal,
    validate_provider_endpoint,
)
from scripts.training.t5gemma2_local_rs_sft_pilot import (
    Evaluation,
    JOURNAL_SCHEMA as LOCAL_JOURNAL_SCHEMA,
    PilotTask,
    PrivateGate,
)


def _task(task_id: str) -> PilotTask:
    source = f"<F2>{task_id}: assembly cfg dfg only</F2>"
    gold = f"int fn_{task_id.replace('-', '_')}(int x) => x + 999;"
    return PilotTask(
        task_id=task_id,
        source=source,
        source_sha256=sha256_text(source),
        visible_tests=f"VISIBLE_TEST_SECRET::{task_id}",
        gold_target=gold,
        gold_target_sha256=sha256_text(gold),
        f2_row={"task_id": task_id, "text": f"f2::{task_id}"},
        split_binding_sha256=sha256_text(f"split::{task_id}"),
    )


def _gate(task: PilotTask) -> PrivateGate:
    return PrivateGate(
        task_id=task.task_id,
        tests=f"PRIVATE_HOLDBACK_SECRET::{task.task_id}",
        split_binding_sha256=task.split_binding_sha256,
    )


def _candidate(
    code: str,
    *,
    compiled: bool,
    passed: bool = False,
    diagnostic: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "origin": "base",
        "sample_index": 0,
        "code": code,
        "code_sha256": sha256_text(code),
        "visible": {"compiled": compiled, "passed": passed},
    }
    if not compiled and diagnostic:
        row["safe_compiler_feedback"] = diagnostic
        row["safe_compiler_feedback_sha256"] = sha256_text(diagnostic)
    return row


def _terminal(task: PilotTask, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "journal_event_sha256": sha256_text(f"terminal::{task.task_id}"),
        "base_candidates": candidates,
        "repair_groups": [],
        "visible_unique_passes": sum(
            candidate["visible"]["passed"] for candidate in candidates
        ),
        "selected_target": None,
    }


def _local_terminal_event(
    task: PilotTask,
    position: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    terminal = _terminal(task, candidates)
    terminal.update(
        {
            "event": "task_terminal",
            "schema": LOCAL_JOURNAL_SCHEMA,
            "task_position": position,
            "source_sha256": task.source_sha256,
            "split_binding_sha256": task.split_binding_sha256,
            "all_generation_completed_before_private_gate": True,
            "private_feedback_serialized_to_model": False,
            "holdback_failure_triggers_generation": False,
            "private_gate_results": [],
        }
    )
    terminal.pop("journal_event_sha256")
    return terminal


def test_selection_is_all_zero_deterministic_diverse_and_prefers_noncompilers() -> None:
    tasks = [_task(f"fit-{index}") for index in range(3)]
    terminals = [
        _terminal(
            tasks[0],
            [
                _candidate(
                    "int fn0(int x) => x + ;",
                    compiled=False,
                    diagnostic="Error: Expected an expression at C:\\secret\\a.dart:1",
                ),
                _candidate(
                    "int fn0(int x) { return x * ; }",
                    compiled=False,
                    diagnostic="Error: Expected an expression",
                ),
                _candidate("int fn0(int x) => x;", compiled=True),
            ],
        ),
        _terminal(
            tasks[1],
            [_candidate("int fn1(int x) => x;", compiled=True, passed=True)],
        ),
        _terminal(
            tasks[2],
            [_candidate("int fn2(int x) => x - 1;", compiled=True)],
        ),
    ]
    # Eligibility is defined only by the all-zero visible group.  A stale or
    # future auxiliary selection field must never silently change the cohort.
    terminals[2]["selected_target"] = {"stale_auxiliary_field": True}
    gates = {task.task_id: _gate(task) for task in tasks}

    first = select_rescue_plans(
        scheduled_tasks=tasks,
        gates=gates,
        terminals=terminals,
        seed=17,
        max_tasks=0,
        max_parents_per_task=2,
    )
    second = select_rescue_plans(
        scheduled_tasks=tasks,
        gates=gates,
        terminals=terminals,
        seed=17,
        max_tasks=0,
        max_parents_per_task=2,
    )

    assert [
        (plan.task.task_id, [parent.code_sha256 for parent in plan.parents])
        for plan in first
    ] == [
        (plan.task.task_id, [parent.code_sha256 for parent in plan.parents])
        for plan in second
    ]
    assert {plan.task.task_id for plan in first} == {"fit-0", "fit-2"}
    fit0 = next(plan for plan in first if plan.task.task_id == "fit-0")
    assert len(fit0.parents) == 2
    assert all(parent.compiled is False for parent in fit0.parents)
    assert all(
        "<path>/" in parent.diagnostic or "Expected" in parent.diagnostic
        for parent in fit0.parents
    )
    assert len({parent.code_sha256 for parent in fit0.parents}) == 2


def test_selection_requires_exact_integer_zero_visible_pass_count() -> None:
    task = _task("fit-strict-zero")
    terminal = _terminal(task, [_candidate("int fn0(int x) => x;", compiled=True)])
    terminal["visible_unique_passes"] = False
    with pytest.raises(ValueError, match="non-negative integer"):
        select_rescue_plans(
            scheduled_tasks=[task],
            gates={task.task_id: _gate(task)},
            terminals=[terminal],
            seed=1,
            max_tasks=1,
            max_parents_per_task=1,
        )


def test_eligible_task_offset_is_post_sort_disjoint_and_fail_closed() -> None:
    tasks = [_task(f"fit-offset-{index}") for index in range(7)]
    terminals = [
        _terminal(
            task,
            [_candidate(f"int fn{index}(int x) => x;", compiled=True)],
        )
        for index, task in enumerate(tasks)
    ]
    gates = {task.task_id: _gate(task) for task in tasks}

    complete = select_rescue_plans(
        scheduled_tasks=tasks,
        gates=gates,
        terminals=terminals,
        seed=19,
        max_tasks=0,
        max_parents_per_task=1,
    )
    first_tranche = select_rescue_plans(
        scheduled_tasks=tasks,
        gates=gates,
        terminals=terminals,
        seed=19,
        max_tasks=3,
        max_parents_per_task=1,
        eligible_task_offset=0,
    )
    second_tranche = select_rescue_plans(
        scheduled_tasks=tasks,
        gates=gates,
        terminals=terminals,
        seed=19,
        max_tasks=0,
        max_parents_per_task=1,
        eligible_task_offset=3,
    )

    complete_ids = [plan.task.task_id for plan in complete]
    first_ids = [plan.task.task_id for plan in first_tranche]
    second_ids = [plan.task.task_id for plan in second_tranche]
    assert first_ids == complete_ids[:3]
    assert second_ids == complete_ids[3:]
    assert set(first_ids).isdisjoint(second_ids)
    assert first_ids + second_ids == complete_ids

    with pytest.raises(ValueError, match="offset exceeds"):
        select_rescue_plans(
            scheduled_tasks=tasks,
            gates=gates,
            terminals=terminals,
            seed=19,
            max_tasks=1,
            max_parents_per_task=1,
            eligible_task_offset=len(tasks) + 1,
        )


def test_frozen_incomplete_terminal_prefix_is_exact_and_append_stable(
    tmp_path: Path,
) -> None:
    tasks = [_task(f"fit-prefix-{index}") for index in range(3)]
    contract = {
        "schema": "local-prefix-test",
        "schedule": {"tasks": 3},
        "sampling": {
            "base_samples": 1,
            "repair_samples": 0,
            "max_repair_parents": 0,
        },
    }
    journal = tmp_path / "local-pilot.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": LOCAL_JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    for position in range(2):
        append_event(
            journal,
            _local_terminal_event(
                tasks[position],
                position,
                [
                    _candidate(
                        f"int fn{position}(int x) => x;",
                        compiled=True,
                    )
                ],
            ),
        )

    first_tasks, first_terminals, first_record = freeze_local_terminal_prefix(
        load_journal(journal),
        contract=contract,
        scheduled_tasks=tasks,
        terminal_prefix_length=2,
    )
    assert [task.task_id for task in first_tasks] == [
        "fit-prefix-0",
        "fit-prefix-1",
    ]
    assert len(first_terminals) == 2
    assert first_record["mode"] == EXPLORATORY_PREFIX_RUN_MODE
    assert first_record["terminal_prefix_length"] == 2
    assert first_record["exploratory_prefix"] is True
    assert first_record["production_floor_eligible"] is False

    # The source journal remains append-only. Later local work must not change
    # the already frozen exploratory prefix binding.
    append_event(
        journal,
        _local_terminal_event(
            tasks[2],
            2,
            [_candidate("int fn2(int x) => x;", compiled=True)],
        ),
    )
    _, _, after_append = freeze_local_terminal_prefix(
        load_journal(journal),
        contract=contract,
        scheduled_tasks=tasks,
        terminal_prefix_length=2,
    )
    assert after_append == first_record


def test_frozen_prefix_rejects_missing_or_malformed_terminal(
    tmp_path: Path,
) -> None:
    tasks = [_task("fit-prefix-0"), _task("fit-prefix-1")]
    contract = {
        "schema": "local-prefix-test",
        "sampling": {
            "base_samples": 1,
            "repair_samples": 0,
            "max_repair_parents": 0,
        },
    }
    journal = tmp_path / "malformed-local.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": LOCAL_JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    bad = _local_terminal_event(
        tasks[0],
        0,
        [_candidate("int fn0(int x) => x;", compiled=True)],
    )
    bad["task_id"] = "wrong-task"
    append_event(journal, bad)
    events = load_journal(journal)  # hash chain itself is valid
    with pytest.raises(ValueError, match="differs from schedule"):
        freeze_local_terminal_prefix(
            events,
            contract=contract,
            scheduled_tasks=tasks,
            terminal_prefix_length=1,
        )

    clean_journal = tmp_path / "short-local.journal.jsonl"
    append_event(
        clean_journal,
        {
            "event": "header",
            "schema": LOCAL_JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    append_event(
        clean_journal,
        _local_terminal_event(
            tasks[0],
            0,
            [_candidate("int fn0(int x) => x;", compiled=True)],
        ),
    )
    with pytest.raises(ValueError, match="fewer validated terminals"):
        freeze_local_terminal_prefix(
            load_journal(clean_journal),
            contract=contract,
            scheduled_tasks=tasks,
            terminal_prefix_length=2,
        )


def test_prompt_contains_visible_train_checks_but_no_holdback_or_gold() -> None:
    task = _task("fit-privacy")
    terminal = _terminal(
        task,
        [
            _candidate(
                "int fn0(int x) => x + ;",
                compiled=False,
                diagnostic=(
                    "C:\\tmp\\candidate.dart:1 Error: Expected ';'\n"
                    "Expected: 7\nActual: 6"
                ),
            )
        ],
    )
    plan = select_rescue_plans(
        scheduled_tasks=[task],
        gates={task.task_id: _gate(task)},
        terminals=[terminal],
        seed=1,
        max_tasks=1,
        max_parents_per_task=1,
    )[0]
    prompt = build_provider_prompt(plan, plan.parents[0])

    assert plan.parents[0].feedback_source in prompt
    assert task.source in prompt
    assert task.visible_tests in prompt
    assert plan.gate.tests not in prompt
    assert task.gold_target not in prompt
    assert "Expected: 7" not in prompt
    assert "Actual: 6" not in prompt
    assert "test-oracle values redacted" in prompt


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("int fn0(int x) => x + 1;", True),
        ("```dart\nint fn0(int x) => x + 1;\n```", True),
        ("Here is the fixed code:\n```dart\nint fn0(int x) => x;\n```", False),
        ("<analysis>reason</analysis>\nint fn0(int x) => x;", False),
        ("```dart\nint a() => 1;\n```\n```dart\nint b() => 2;\n```", False),
    ],
)
def test_code_only_parser_rejects_explanations(text: str, accepted: bool) -> None:
    code, error = parse_code_only(text)
    assert bool(code) is accepted
    assert (error is None) is accepted


def test_worst_case_caps_truncate_only_on_complete_task_boundaries() -> None:
    task0, task1 = _task("fit-0"), _task("fit-1")

    def parent(task: PilotTask, index: int) -> RescueParent:
        code = f"int fn{index}(int x) => x;"
        source = task.source + f"\nrepair::{index}"
        return RescueParent(
            task_id=task.task_id,
            parent_index=index,
            code=code,
            code_sha256=sha256_text(code),
            compiled=False,
            diagnostic="Error",
            diagnostic_sha256=sha256_text("Error"),
            origin="base",
            feedback_source=source,
            feedback_source_sha256=sha256_text(source),
        )

    plans = [
        RescuePlan(0, task0, _gate(task0), "a" * 64, (parent(task0, 0),)),
        RescuePlan(
            1,
            task1,
            _gate(task1),
            "b" * 64,
            (parent(task1, 1), parent(task1, 2)),
        ),
    ]
    capacity, record = schedule_capacity(
        max_calls=3,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=50,
        max_input_tokens_total=300,
        max_output_tokens_total=150,
        max_total_tokens=450,
        max_usd=Decimal("1"),
        input_usd_per_million=Decimal("1"),
        output_usd_per_million=Decimal("1"),
    )
    capped = cap_plans_to_budget(plans, samples_per_parent=2, call_capacity=capacity)

    assert capacity == 3
    assert [plan.task.task_id for plan in capped] == ["fit-0"]
    assert record["reservation_policy"].startswith("full_per_call")


class _FakeTransport:
    def __init__(self, code: str) -> None:
        self.code = code
        self.calls: list[tuple[str, str, int]] = []

    def create(
        self, *, system: str, user: str, max_output_tokens: int
    ) -> ProviderResponse:
        self.calls.append((system, user, max_output_tokens))
        return ProviderResponse(
            text=f"```dart\n{self.code}\n```",
            response_id=f"response-{len(self.calls)}",
            model="fake-model",
            input_tokens=200,
            output_tokens=30,
            finish_reason="stop",
        )


def test_retryable_empty_choice_payload_is_retried_without_advancing_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _single_plan()
    slots = build_slots([plan], samples_per_parent=1)
    contract = {"schema": "retry-test", "slots": 1}
    journal = tmp_path / "retry.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )

    class FlakyTransport:
        def __init__(self) -> None:
            self.calls = 0

        def create(
            self, *, system: str, user: str, max_output_tokens: int
        ) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                raise RetryableProviderPayloadError("zero choices")
            return ProviderResponse(
                text="```dart\nint fn0(int x) => x + 1;\n```",
                response_id="retry-success",
                model="fake-model",
                input_tokens=200,
                output_tokens=30,
                finish_reason="stop",
            )

    delays: list[float] = []
    monkeypatch.setattr(rescue_module.time, "sleep", delays.append)
    transport = FlakyTransport()
    results = execute_api_phase(
        journal_path=journal,
        contract=contract,
        plans=[plan],
        slots=slots,
        transport=transport,
        api_key="unit-test-credential-value",
        max_input_tokens=4096,
        max_output_tokens=512,
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
        provider_max_attempts=2,
        provider_retry_base_seconds=2,
        provider_retry_max_seconds=30,
    )

    assert transport.calls == 2
    assert delays == [2.0]
    assert results[0]["status"] == "response"
    assert results[0]["provider_attempts"] == 2
    assert results[0]["provider_retry_delays_seconds"] == [2.0]


def test_offset_is_bound_into_run_contract_and_final_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [_task(f"fit-contract-offset-{index}") for index in range(5)]
    terminals = [
        _terminal(
            task,
            [_candidate(f"int fn{index}(int x) => x;", compiled=True)],
        )
        for index, task in enumerate(tasks)
    ]
    gates = {task.task_id: _gate(task) for task in tasks}
    source_record = {
        "mode": COMPLETED_LOCAL_RUN_MODE,
        "terminal_prefix_length": None,
        "journal_sha256": "a" * 64,
    }

    def fake_local_run(
        _args: argparse.Namespace,
    ) -> tuple[
        list[PilotTask],
        dict[str, PrivateGate],
        list[PilotTask],
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        return (
            tasks,
            gates,
            tasks,
            terminals,
            {"fixture_inputs_sha256": "b" * 64},
            source_record,
        )

    output_dir = tmp_path / "offset-contract"
    args = parse_args(
        [
            "--pilot_journal",
            "pilot.jsonl",
            "--rollout_file",
            "rollout.jsonl",
            "--f2_jsonl",
            "f2.jsonl",
            "--private_holdback",
            "private.jsonl",
            "--output_dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--model",
            "fixture-model",
            "--base_url",
            "https://api.anthropic.com",
            "--seed",
            "23",
            "--eligible_task_offset",
            "2",
            "--max_tasks",
            "2",
            "--max_parents_per_task",
            "1",
            "--samples_per_parent",
            "1",
            "--max_calls",
            "2",
            "--max_input_tokens_per_call",
            "4096",
            "--max_output_tokens",
            "512",
            "--max_input_tokens_total",
            "8192",
            "--max_output_tokens_total",
            "1024",
            "--max_total_tokens",
            "9216",
            "--max_usd",
            "1",
            "--input_usd_per_million",
            "1",
            "--output_usd_per_million",
            "1",
        ]
    )
    monkeypatch.setattr(rescue_module, "_load_completed_local_run", fake_local_run)
    monkeypatch.setenv("RS_SFT_API_KEY", "unit-test-credential-value")

    report = rescue_module.run(
        args,
        transport=_FakeTransport("int fn0(int x) => x + 1;"),
        evaluate=lambda _code, _tests, _slot: Evaluation(True, True, ""),
    )
    header = load_journal(output_dir / "api_rescue.journal.jsonl")[0]
    selection = header["contract"]["selection"]
    assert selection["eligible_task_offset"] == 2
    assert selection["eligible_task_offset_applied_after_deterministic_sort"]
    assert selection["eligible_tasks_before_offset"] == 5
    assert selection["eligible_tasks_after_offset_before_task_cap"] == 3
    assert selection["eligible_tasks_after_task_cap_before_budget"] == 2
    assert selection["scheduled_tasks"] == 2

    schedule = report["schedule"]
    assert schedule["eligible_all_zero_tasks_before_offset"] == 5
    assert schedule["eligible_task_offset"] == 2
    assert schedule["eligible_task_offset_applied_after_deterministic_sort"]
    assert schedule["eligible_all_zero_tasks_after_offset_before_task_cap"] == 3
    assert schedule["eligible_all_zero_tasks_before_caps"] == 2
    assert schedule["scheduled_tasks"] == 2
    assert schedule["scheduled_calls"] == 2
    assert schedule["task_ids_sha256"] == selection["task_ids_sha256"]


def test_evaluation_only_scores_holdback_without_training_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task("evaluation-only")
    terminal = _terminal(
        task,
        [_candidate("int evaluation_only(int x) => x;", compiled=True)],
    )
    source_record = {
        "mode": COMPLETED_LOCAL_RUN_MODE,
        "terminal_prefix_length": None,
        "journal_sha256": "a" * 64,
    }

    def fake_local_run(
        _args: argparse.Namespace,
    ) -> tuple[
        list[PilotTask],
        dict[str, PrivateGate],
        list[PilotTask],
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        return (
            [task],
            {task.task_id: _gate(task)},
            [task],
            [terminal],
            {"fixture_inputs_sha256": "b" * 64},
            source_record,
        )

    output_dir = tmp_path / "evaluation-only"
    args = parse_args(
        [
            "--pilot_journal",
            "pilot.jsonl",
            "--rollout_file",
            "rollout.jsonl",
            "--f2_jsonl",
            "f2.jsonl",
            "--private_holdback",
            "private.jsonl",
            "--output_dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--model",
            "fixture-model",
            "--base_url",
            "https://api.anthropic.com",
            "--evaluation_only",
            "--max_tasks",
            "1",
            "--max_parents_per_task",
            "1",
            "--samples_per_parent",
            "1",
            "--max_calls",
            "1",
            "--max_input_tokens_per_call",
            "4096",
            "--max_output_tokens",
            "512",
            "--max_input_tokens_total",
            "4096",
            "--max_output_tokens_total",
            "512",
            "--max_total_tokens",
            "4608",
            "--max_usd",
            "1",
            "--input_usd_per_million",
            "1",
            "--output_usd_per_million",
            "1",
        ]
    )
    monkeypatch.setattr(rescue_module, "_load_completed_local_run", fake_local_run)
    monkeypatch.setenv("RS_SFT_API_KEY", "unit-test-credential-value")

    report = rescue_module.run(
        args,
        transport=_FakeTransport("int evaluation_only(int x) => x + 1;"),
        evaluate=lambda _code, _tests, _slot: Evaluation(True, True, ""),
    )

    assert report["execution_mode"] == "evaluation_only"
    assert report["evaluation_only"] is True
    assert report["training_use_forbidden"] is True
    assert report["production_floor_eligible"] is False
    assert report["may_count_toward_production_min_unique_targets"] is False
    assert report["verification"]["private_holdback_passes"] == 1
    assert report["verification"]["verified_unique_hard_targets"] == 0
    assert report["outputs"] == {}
    assert report["direct_manifest"] is None
    assert report["repair_policy_manifest"] is None
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "api_rescue.journal.jsonl",
        "api_rescue.journal.jsonl.chain-head.json",
        "api_rescue_report.json",
    ]
    journal = load_journal(output_dir / "api_rescue.journal.jsonl")
    selected = next(
        row["selected_target"]
        for row in journal
        if row.get("event") == "task_verification"
    )
    assert selected["training_use_forbidden"] is True
    assert selected["production_floor_eligible"] is False
    assert journal[-1]["training_use_forbidden"] is True
    assert journal[-1]["production_floor_eligible"] is False


def _single_plan() -> RescuePlan:
    task = _task("fit-resume")
    terminal = _terminal(
        task,
        [
            _candidate(
                "int fn0(int x) => x + ;",
                compiled=False,
                diagnostic="Error: Expected an expression",
            )
        ],
    )
    return select_rescue_plans(
        scheduled_tasks=[task],
        gates={task.task_id: _gate(task)},
        terminals=[terminal],
        seed=7,
        max_tasks=1,
        max_parents_per_task=1,
    )[0]


def _write_prior_success_fixture(
    output_dir: Path,
    *,
    task_ids: list[str],
    verified_task_ids: set[str],
    input_record: dict[str, Any],
    source_record: dict[str, Any],
    result_specs: dict[str, tuple[bool, str]] | None = None,
) -> tuple[Path, str, list[RescuePlan]]:
    plans: list[RescuePlan] = []
    for task_id in task_ids:
        task = _task(task_id)
        selected = select_rescue_plans(
            scheduled_tasks=[task],
            gates={task.task_id: _gate(task)},
            terminals=[
                _terminal(
                    task,
                    [_candidate(f"int {task_id}() => 0;", compiled=True)],
                )
            ],
            seed=31,
            max_tasks=1,
            max_parents_per_task=1,
        )[0]
        plans.append(
            RescuePlan(
                task_position=len(plans),
                task=selected.task,
                gate=selected.gate,
                local_terminal_sha256=selected.local_terminal_sha256,
                parents=selected.parents,
            )
        )
    slots = build_slots(plans, samples_per_parent=1)
    contract = {
        "schema": RUN_SCHEMA,
        "inputs": input_record,
        "source_local_pilot_journal": source_record,
        "local_source": {
            "mode": COMPLETED_LOCAL_RUN_MODE,
            "exploratory_prefix": False,
            "production_floor_eligible": True,
        },
        "selection": {
            "scheduled_tasks": len(plans),
            "scheduled_slots": len(slots),
            "task_ids_sha256": canonical_sha256(task_ids),
        },
        "provider": {"provider": "fixture", "model": "fixture"},
        "privacy": {
            "private_holdback_sent_to_provider": False,
            "gold_sent_to_provider": False,
        },
        "training_outputs": {"production_floor_eligible": True},
        "heldout_175_opened": False,
    }
    contract_sha256 = canonical_sha256(contract)
    journal = output_dir / "api_rescue.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": contract_sha256,
        },
    )
    for slot in slots:
        binding = rescue_module._slot_binding(slot)
        request_sha256 = canonical_sha256(
            {
                "system_sha256": sha256_text(SYSTEM_PROMPT),
                **binding,
            }
        )
        parse_accepted, finish_reason = (result_specs or {}).get(
            slot.task_id, (True, "stop")
        )
        response_code = (
            f"int response_{slot.slot_position}() => {slot.slot_position};"
            if parse_accepted
            else ""
        )
        append_event(
            journal,
            {
                "event": "call_intent",
                "schema": JOURNAL_SCHEMA,
                **binding,
                "request_sha256": request_sha256,
            },
        )
        append_event(
            journal,
            {
                "event": "call_result",
                "schema": JOURNAL_SCHEMA,
                **binding,
                "request_sha256": request_sha256,
                "status": "response",
                "response": {
                    "response_id_sha256": sha256_text(
                        f"response::{slot.slot_position}"
                    ),
                    "resolved_model": "fixture",
                    "finish_reason": finish_reason,
                    "raw_text_sha256": sha256_text(
                        response_code or f"prose::{slot.slot_position}"
                    ),
                },
                "parse_accepted": parse_accepted,
                "parse_rejection": (
                    None if parse_accepted else "response_is_not_code_only"
                ),
                "code": response_code,
                "code_sha256": (
                    sha256_text(response_code) if response_code else None
                ),
                "usage": {
                    "charged_input_tokens": 10,
                    "charged_output_tokens": 10,
                    "charged_usd_nanos": 20,
                },
            },
        )
    verifications: list[dict[str, Any]] = []
    for task_position, plan in enumerate(plans):
        selected_target = None
        if plan.task.task_id in verified_task_ids:
            code = f"int verified_{task_position}() => {task_position};"
            selected_target = {
                "schema": DIRECT_TARGET_SCHEMA,
                "task_id": plan.task.task_id,
                "source_sha256": plan.task.source_sha256,
                "code": code,
                "code_sha256": sha256_text(code),
                "visible_passed": True,
                "private_gate_passed": True,
                "exploratory_prefix": False,
                "production_floor_eligible": True,
                "parent_position": 0,
                "slot_position": task_position,
            }
        verification = {
            "event": "task_verification",
            "schema": JOURNAL_SCHEMA,
            "task_position": task_position,
            "task_id": plan.task.task_id,
            "source_sha256": plan.task.source_sha256,
            "split_binding_sha256": plan.task.split_binding_sha256,
            "all_api_generation_completed_before_private_gate": True,
            "private_feedback_serialized_to_model": False,
            "holdback_failure_triggers_generation": False,
            "selected_target": selected_target,
        }
        append_event(journal, verification)
        verifications.append(verification)
    append_event(
        journal,
        {
            "event": "complete",
            "schema": JOURNAL_SCHEMA,
            "tasks": len(plans),
            "slots": len(slots),
            "verified_targets": len(verified_task_ids),
            "exploratory_prefix": False,
            "production_floor_eligible": True,
        },
    )
    outputs = _publish_training_outputs(
        output_dir=output_dir,
        plans=plans,
        verifications=verifications,
        contract_sha256=contract_sha256,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": contract_sha256,
        "execution_mode": "rs_sft_rescue",
        "exploratory_prefix": False,
        "evaluation_only": False,
        "training_use_forbidden": False,
        "production_floor_eligible": True,
        "may_count_toward_production_min_unique_targets": True,
        "heldout_175_opened": False,
        "schedule": {
            "scheduled_tasks": len(plans),
            "scheduled_calls": len(slots),
            "task_ids_sha256": canonical_sha256(task_ids),
            "provider_responses": len(slots),
            "code_only_responses": sum(
                (result_specs or {}).get(task_id, (True, "stop"))[0]
                for task_id in task_ids
            ),
        },
        "verification": {"verified_unique_hard_targets": len(verified_task_ids)},
        "budget_charged": {
            "calls": len(slots),
            "within_contract": True,
        },
        "outputs": outputs["files"],
        "direct_manifest": outputs["direct_manifest"],
        "repair_policy_manifest": outputs["repair_manifest"],
        "journal": journal_record(journal),
    }
    report_path = output_dir / "api_rescue_report.json"
    require_exact_or_write(report_path, report)
    return report_path, sha256_file(report_path), plans


def test_pinned_prior_success_exclusion_verifies_artifacts_and_residual(
    tmp_path: Path,
) -> None:
    input_record = {"sealed_inputs": "fixture"}
    source_record = {
        "mode": COMPLETED_LOCAL_RUN_MODE,
        "sha256": "a" * 64,
    }
    first_path, first_sha, first_plans = _write_prior_success_fixture(
        tmp_path / "first",
        task_ids=["prior-a", "prior-b"],
        verified_task_ids={"prior-a"},
        input_record=input_record,
        source_record=source_record,
    )
    second_path, second_sha, second_plans = _write_prior_success_fixture(
        tmp_path / "second",
        task_ids=["prior-c"],
        verified_task_ids={"prior-c"},
        input_record=input_record,
        source_record=source_record,
    )
    exclusions = load_prior_success_exclusions(
        report_paths=[first_path, second_path],
        expected_report_sha256s=[first_sha, second_sha],
        current_eligible_task_ids=["prior-a", "prior-b", "prior-c"],
        input_record=input_record,
        source_journal_record=source_record,
        require_disjoint_schedules=True,
        require_complete_coverage=True,
    )
    assert exclusions.scheduled_task_ids == {
        "prior-a",
        "prior-b",
        "prior-c",
    }
    assert exclusions.verified_task_ids == {"prior-a", "prior-c"}
    residual = exclude_prior_verified_plans(
        [*first_plans, *second_plans], exclusions.verified_task_ids
    )
    assert [plan.task.task_id for plan in residual] == ["prior-b"]
    assert slice_rescue_plans(residual, offset=0, max_tasks=1)[0].task_position == 0

    with pytest.raises(ValueError, match="do not exactly cover"):
        load_prior_success_exclusions(
            report_paths=[first_path],
            expected_report_sha256s=[first_sha],
            current_eligible_task_ids=["prior-a", "prior-b", "prior-c"],
            input_record=input_record,
            source_journal_record=source_record,
            require_complete_coverage=True,
        )
    overlap_path, overlap_sha, _ = _write_prior_success_fixture(
        tmp_path / "overlap",
        task_ids=["prior-b"],
        verified_task_ids=set(),
        input_record=input_record,
        source_record=source_record,
    )
    with pytest.raises(ValueError, match="not disjoint"):
        load_prior_success_exclusions(
            report_paths=[first_path, overlap_path],
            expected_report_sha256s=[first_sha, overlap_sha],
            current_eligible_task_ids=["prior-a", "prior-b"],
            input_record=input_record,
            source_journal_record=source_record,
            require_disjoint_schedules=True,
        )
    with pytest.raises(ValueError, match="report 0 digest differs"):
        load_prior_success_exclusions(
            report_paths=[first_path],
            expected_report_sha256s=["f" * 64],
            current_eligible_task_ids=["prior-a", "prior-b"],
            input_record=input_record,
            source_journal_record=source_record,
        )
    with pytest.raises(ValueError, match="sealed inputs"):
        load_prior_success_exclusions(
            report_paths=[first_path],
            expected_report_sha256s=[first_sha],
            current_eligible_task_ids=["prior-a", "prior-b"],
            input_record={"sealed_inputs": "drifted"},
            source_journal_record=source_record,
        )

    with (tmp_path / "first" / "direct_hard_targets.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="direct_targets digest differs"):
        load_prior_success_exclusions(
            report_paths=[first_path],
            expected_report_sha256s=[first_sha],
            current_eligible_task_ids=["prior-a", "prior-b"],
            input_record=input_record,
            source_journal_record=source_record,
        )


def test_prior_success_cli_requires_exact_paired_report_digests() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_minimal_cli("azure_v1_responses"),
                "--prior_success_report",
                "prior.json",
            ]
        )
    parsed = parse_args(
        [
            *_minimal_cli("azure_v1_responses"),
            "--prior_success_report",
            "prior.json",
            "--expected_prior_success_report_sha256",
            "a" * 64,
        ]
    )
    assert parsed.prior_success_report == ["prior.json"]


def test_pinned_retry_selects_parse_failures_or_truncations_in_source_order(
    tmp_path: Path,
) -> None:
    input_record = {"sealed_inputs": "retry-fixture"}
    source_record = {
        "mode": COMPLETED_LOCAL_RUN_MODE,
        "sha256": "a" * 64,
    }
    task_ids = ["retry-a", "retry-b", "retry-c", "retry-d"]
    report_path, report_sha256, plans = _write_prior_success_fixture(
        tmp_path / "retry-source",
        task_ids=task_ids,
        verified_task_ids=set(),
        input_record=input_record,
        source_record=source_record,
        result_specs={
            "retry-a": (False, "length"),
            "retry-b": (False, "stop"),
            "retry-c": (True, "length"),
            "retry-d": (True, "stop"),
        },
    )

    selected = load_retry_parse_failures_or_truncations_source(
        report_path=report_path,
        expected_report_sha256=report_sha256,
        current_eligible_plans=plans,
        input_record=input_record,
        source_journal_record=source_record,
    )

    selected_ids = [plan.task.task_id for plan in selected.plans]
    assert selected_ids == ["retry-a", "retry-b", "retry-c"]
    assert selected.record["retry_tasks"] == 3
    assert selected.record["retry_task_ids_sha256"] == canonical_sha256(
        selected_ids
    )
    assert selected.record["qualifying_result_event_count"] == 3
    assert selected.record["finish_reason_counts"] == {
        "length": 2,
        "stop": 1,
    }
    assert selected.record["parse_failure_count"] == 2
    assert selected.record["truncation_count"] == 2
    assert selected.record["parse_failure_and_truncation_count"] == 1
    assert selected.record["accepted_nontruncated_responses_regenerated"] is False


def test_retry_cli_requires_complete_pin_and_one_slot_per_task() -> None:
    base = [
        *_minimal_cli("azure_v1_responses"),
        "--retry_parse_failures_or_truncations_report",
        "source.json",
    ]
    with pytest.raises(SystemExit):
        parse_args(base)
    pinned = [
        *base,
        "--expected_retry_parse_failures_or_truncations_report_sha256",
        "a" * 64,
        "--expected_retry_parse_failures_or_truncations_tasks",
        "17",
        "--expected_retry_parse_failures_or_truncations_task_ids_sha256",
        "b" * 64,
        "--max_parents_per_task",
        "1",
    ]
    parsed = parse_args(pinned)
    assert parsed.expected_retry_parse_failures_or_truncations_tasks == 17
    with pytest.raises(SystemExit):
        parse_args([*pinned, "--samples_per_parent", "2"])


def test_exact_resume_verifies_then_emits_both_training_views(
    tmp_path: Path,
) -> None:
    plan = _single_plan()
    slots = build_slots([plan], samples_per_parent=1)
    contract = {"schema": "test-contract", "slots": 1}
    journal = tmp_path / "rescue.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    # canonical JSON for this simple ASCII mapping matches the helper digest.
    transport = _FakeTransport("int fn0(int x) => x + 1;")
    first = execute_api_phase(
        journal_path=journal,
        contract=contract,
        plans=[plan],
        slots=slots,
        transport=transport,
        api_key="unit-test-credential-value",
        max_input_tokens=4096,
        max_output_tokens=512,
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
    )
    second = execute_api_phase(
        journal_path=journal,
        contract=contract,
        plans=[plan],
        slots=slots,
        transport=transport,
        api_key="unit-test-credential-value",
        max_input_tokens=4096,
        max_output_tokens=512,
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
    )
    assert len(first) == len(second) == 1
    assert len(transport.calls) == 1

    trace: list[str] = []

    def evaluate(code: str, tests: str, slot: str) -> Evaluation:
        if tests == plan.task.visible_tests:
            trace.append("visible")
            return Evaluation(True, True, "")
        assert tests == plan.gate.tests
        trace.append("private")
        return Evaluation(True, True, "SECRET_PRIVATE_DIAGNOSTIC")

    verifications = execute_verification_phase(
        journal_path=journal,
        contract=contract,
        plans=[plan],
        slots=slots,
        evaluate=evaluate,
        api_key="unit-test-credential-value",
    )
    assert trace == ["visible", "private"]
    assert verifications[0]["selected_target"]["schema"] == DIRECT_TARGET_SCHEMA
    assert "SECRET_PRIVATE_DIAGNOSTIC" not in journal.read_text(encoding="utf-8")

    outputs = _publish_training_outputs(
        output_dir=tmp_path,
        plans=[plan],
        verifications=verifications,
        contract_sha256=sha256_text("contract"),
    )
    assert len(outputs["rows"]["direct_targets"]) == 1
    assert len(outputs["rows"]["repair_targets"]) == 1
    assert (
        outputs["rows"]["direct_targets"][0]["dart_source"]
        == outputs["rows"]["repair_targets"][0]["dart_source"]
    )
    assert (
        outputs["rows"]["repair_sources"][0]["encoder_source"]
        == plan.parents[0].feedback_source
    )
    assert outputs["repair_manifest"]["source_is_exact_model_input"] is True
    assert outputs["direct_manifest"]["production_floor_eligible"] is True
    assert load_journal(journal)[-1]["event"] == "complete"


def test_exploratory_prefix_outputs_can_never_count_toward_production_floor(
    tmp_path: Path,
) -> None:
    plan = _single_plan()
    code = "int fn0(int x) => x + 1;"
    verification = {
        "selected_target": {
            "task_id": plan.task.task_id,
            "parent_position": 0,
            "slot_position": 0,
            "code": code,
        }
    }
    outputs = _publish_training_outputs(
        output_dir=tmp_path,
        plans=[plan],
        verifications=[verification],
        contract_sha256=sha256_text("exploratory-contract"),
        exploratory_prefix=True,
    )

    for manifest_name in ("direct_manifest", "repair_manifest"):
        manifest = outputs[manifest_name]
        assert manifest["exploratory_prefix"] is True
        assert manifest["production_floor_eligible"] is False
        assert manifest["may_count_toward_production_min_unique_targets"] is False
    for name in ("direct_targets", "repair_targets", "repair_sources"):
        assert outputs["rows"][name][0]["exploratory_prefix"] is True
        assert outputs["rows"][name][0]["production_floor_eligible"] is False


def test_vast_claude_probe_is_small_separate_and_hard_capped() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "vast"
        / "t5gemma2_api_rs_sft_claude_probe.sh"
    ).read_text(encoding="utf-8")
    assert "${WORKSPACE}/secrets/Anthropic.env" in launcher
    assert "--exploratory_terminal_prefix 10" in launcher
    assert "--model claude-sonnet-5" in launcher
    assert "--max_tasks 5" in launcher
    assert "--max_calls 5" in launcher
    assert "--max_parents_per_task 1" in launcher
    assert "--samples_per_parent 1" in launcher
    assert "--max_output_tokens 8192" in launcher
    assert "--max_usd 0.95" in launcher
    assert "--input_usd_per_million 2" in launcher
    assert "--output_usd_per_million 10" in launcher
    assert "t5gemma2_api_rs_sft_claude_probe_prefix10_v1" in launcher

    capacity, _ = schedule_capacity(
        max_calls=5,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=8192,
        max_input_tokens_total=245760,
        max_output_tokens_total=40960,
        max_total_tokens=286720,
        max_usd=Decimal("0.95"),
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
    )
    assert capacity == 5


def test_orphan_call_intent_blocks_duplicate_billing(tmp_path: Path) -> None:
    plan = _single_plan()
    slot: ApiSlot = build_slots([plan], samples_per_parent=1)[0]
    contract = {"schema": "pending-test"}
    journal = tmp_path / "pending.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": JOURNAL_SCHEMA,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    binding = {
        "slot_position": slot.slot_position,
        "task_position": slot.task_position,
        "task_id": slot.task_id,
        "parent_position": slot.parent_position,
        "sample_index": slot.sample_index,
        "parent_code_sha256": slot.parent.code_sha256,
        "diagnostic_sha256": slot.parent.diagnostic_sha256,
        "feedback_source_sha256": slot.parent.feedback_source_sha256,
        "prompt_sha256": slot.prompt_sha256,
    }
    append_event(
        journal,
        {
            "event": "call_intent",
            "schema": JOURNAL_SCHEMA,
            **binding,
            "request_sha256": canonical_sha256(
                {
                    "system_sha256": sha256_text(SYSTEM_PROMPT),
                    **binding,
                }
            ),
        },
    )

    with pytest.raises(PendingProviderCall, match="ambiguous"):
        validate_rescue_journal(
            load_journal(journal),
            contract=contract,
            plans=[plan],
            slots=[slot],
        )


def test_azure_v1_endpoint_contract_is_explicit_and_versionless() -> None:
    endpoint = validate_provider_endpoint(
        provider="azure_v1_responses",
        base_url="https://example.openai.azure.com/openai/v1/",
        api_version="",
    )
    assert endpoint == "https://example.openai.azure.com/openai/v1"
    with pytest.raises(ValueError, match="must not receive --api_version"):
        validate_provider_endpoint(
            provider="azure_v1_chat",
            base_url="https://example.openai.azure.com/openai/v1",
            api_version="2025-04-01-preview",
        )
    with pytest.raises(ValueError, match="must end in /openai/v1"):
        validate_provider_endpoint(
            provider="azure_v1_chat",
            base_url="https://example.openai.azure.com",
            api_version="",
        )
    with pytest.raises(ValueError, match="requires --api_version"):
        validate_provider_endpoint(
            provider="azure_chat",
            base_url="https://example.openai.azure.com",
            api_version="",
        )


def test_openrouter_endpoint_is_exact_and_cannot_exfiltrate_key() -> None:
    assert validate_provider_endpoint(
        provider="openrouter_chat",
        base_url="https://openrouter.ai/api/v1/",
        api_version="",
    ) == "https://openrouter.ai/api/v1"
    for endpoint in (
        "https://attacker.example/api/v1",
        "https://openrouter.ai/api/v1/chat/completions",
        "https://openrouter.ai:443/api/v1",
    ):
        with pytest.raises(ValueError, match="must be exactly"):
            validate_provider_endpoint(
                provider="openrouter_chat",
                base_url=endpoint,
                api_version="",
            )
    with pytest.raises(ValueError, match="unencrypted"):
        validate_provider_endpoint(
            provider="openrouter_chat",
            base_url="http://openrouter.ai/api/v1",
            api_version="",
        )
    with pytest.raises(ValueError, match="must not receive --api_version"):
        validate_provider_endpoint(
            provider="openrouter_chat",
            base_url="https://openrouter.ai/api/v1",
            api_version="2025-01-01",
        )


def test_openrouter_chat_seals_routing_reasoning_and_distillation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return types.SimpleNamespace(
                id="response-id",
                model="minimax/minimax-m3",
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="int f(int x) => x + 1;",
                            reasoning="private reasoning that must not be persisted",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=200,
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=FakeOpenAI, AzureOpenAI=object),
    )
    transport = OpenAITransport(
        api_key="credential",
        base_url="https://openrouter.ai/api/v1",
        model="minimax/minimax-m3",
        timeout=600,
        provider="openrouter_chat",
        api_version="",
        reasoning_effort="",
        chat_token_parameter="max_tokens",
        openrouter_reasoning="enabled",
        openrouter_provider_only=("gmicloud/fp8",),
        openrouter_provider_order=("modal", "baseten"),
        openrouter_allow_fallbacks=False,
        openrouter_require_parameters=True,
        openrouter_include_reasoning=True,
        openrouter_enforce_distillable_text=True,
    )
    response = transport.create(
        system="system",
        user="user",
        max_output_tokens=16384,
    )

    assert response.text == "int f(int x) => x + 1;"
    assert response.output_tokens == 200
    assert requests == [
        {
            "model": "minimax/minimax-m3",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "max_tokens": 16384,
            "extra_body": {
                "provider": {
                    "only": ["gmicloud/fp8"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    "enforce_distillable_text": True,
                    "order": ["modal", "baseten"],
                },
                "reasoning": {"enabled": True, "exclude": False},
                "include_reasoning": True,
            },
        }
    ]


def test_openrouter_empty_choices_are_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            return types.SimpleNamespace(choices=[])

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=FakeOpenAI, AzureOpenAI=object),
    )
    transport = OpenAITransport(
        api_key="credential",
        base_url="https://openrouter.ai/api/v1",
        model="moonshotai/kimi-k3",
        timeout=600,
        provider="openrouter_chat",
        api_version="",
        reasoning_effort="",
        chat_token_parameter="max_tokens",
        openrouter_provider_only=("together",),
        openrouter_require_parameters=True,
        openrouter_enforce_distillable_text=True,
    )

    with pytest.raises(RetryableProviderPayloadError) as exc_info:
        transport.create(system="system", user="user", max_output_tokens=2048)
    assert exc_info.value.status_code == 503


def test_openrouter_chat_seals_xhigh_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return types.SimpleNamespace(
                id="response-id",
                model="z-ai/glm-5.2",
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="void main() {}"),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=200,
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=FakeOpenAI, AzureOpenAI=object),
    )
    transport = OpenAITransport(
        api_key="credential",
        base_url="https://openrouter.ai/api/v1",
        model="z-ai/glm-5.2",
        timeout=600,
        provider="openrouter_chat",
        api_version="",
        reasoning_effort="",
        chat_token_parameter="max_tokens",
        openrouter_reasoning="enabled",
        openrouter_provider_only=("novita/fp8",),
        openrouter_require_parameters=True,
        openrouter_include_reasoning=True,
        openrouter_enforce_distillable_text=False,
        openrouter_reasoning_effort="xhigh",
    )
    transport.create(system="system", user="user", max_output_tokens=16384)

    assert requests[0]["extra_body"] == {
        "provider": {
            "only": ["novita/fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "enforce_distillable_text": False,
        },
        "reasoning": {
            "enabled": True,
            "exclude": False,
            "effort": "xhigh",
        },
        "include_reasoning": True,
    }


def test_azure_v1_uses_standard_openai_client_and_preserves_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("OpenAI", kwargs))

    class FakeAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("AzureOpenAI", kwargs))

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=FakeOpenAI, AzureOpenAI=FakeAzureOpenAI),
    )
    OpenAITransport(
        api_key="credential",
        base_url="https://example.openai.azure.com/openai/v1",
        model="gpt-chat-latest",
        timeout=60,
        provider="azure_v1_responses",
        api_version="",
        reasoning_effort="medium",
        chat_token_parameter="max_completion_tokens",
    )
    OpenAITransport(
        api_key="credential",
        base_url="https://example.openai.azure.com",
        model="gpt-chat-latest",
        timeout=60,
        provider="azure_responses",
        api_version="2025-04-01-preview",
        reasoning_effort="medium",
        chat_token_parameter="max_completion_tokens",
    )

    assert calls[0][0] == "OpenAI"
    assert calls[0][1]["base_url"].endswith("/openai/v1")
    assert "api_version" not in calls[0][1]
    assert calls[1][0] == "AzureOpenAI"
    assert calls[1][1]["api_version"] == "2025-04-01-preview"

    args = argparse.Namespace(
        provider="azure_v1_responses",
        model="gpt-chat-latest",
        api_version="",
        max_output_tokens=4096,
        timeout_seconds=60,
        reasoning_effort="medium",
        chat_token_parameter="max_completion_tokens",
        anthropic_thinking="disabled",
        anthropic_effort="high",
    )
    provenance = _provider_contract(args, "https://example.openai.azure.com/openai/v1")
    assert provenance["azure"] is True
    assert provenance["client"] == "OpenAI"
    assert (
        provenance["azure_endpoint_mode"]
        == "openai_v1_standard_client_without_api_version"
    )
    assert provenance["model_semantics"] == "azure_deployment_name"
    assert provenance["api_version"] is None


def _minimal_cli(provider: str) -> list[str]:
    return [
        "--pilot_journal",
        "pilot.jsonl",
        "--rollout_file",
        "rollout.jsonl",
        "--f2_jsonl",
        "f2.jsonl",
        "--private_holdback",
        "private.jsonl",
        "--output_dir",
        "out",
        "--provider",
        provider,
        "--model",
        "gpt-chat-latest",
        "--base_url",
        "https://example.openai.azure.com/openai/v1",
        "--max_calls",
        "1",
        "--max_usd",
        "1",
        "--input_usd_per_million",
        "1",
        "--output_usd_per_million",
        "1",
    ]


def test_cli_rejects_ambiguous_azure_version_modes() -> None:
    with pytest.raises(SystemExit):
        parse_args(_minimal_cli("azure_responses"))
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_minimal_cli("azure_v1_responses"),
                "--api_version",
                "2025-04-01-preview",
            ]
        )
    parsed = parse_args(_minimal_cli("azure_v1_responses"))
    assert parsed.provider == "azure_v1_responses"
    assert parsed.eligible_task_offset == 0


def test_openrouter_cli_requires_pinned_distillable_routing() -> None:
    base = [
        *_minimal_cli("openrouter_chat"),
        "--base_url",
        "https://openrouter.ai/api/v1",
        "--openrouter_provider_only",
        "gmicloud/fp8",
        "--openrouter_provider_only",
        "modal/mxfp4",
        "--openrouter_provider_only",
        "baseten/fp8",
        "--openrouter_provider_order",
        "modal/mxfp4",
        "--openrouter_provider_order",
        "baseten/fp8",
    ]
    # The later duplicate --base_url wins in argparse; the exact endpoint
    # validator still protects the credential destination.
    with pytest.raises(SystemExit):
        parse_args(base)
    parsed = parse_args(
        [
            *base,
            "--openrouter_require_parameters",
            "--openrouter_enforce_distillable_text",
            "--openrouter_reasoning",
            "enabled",
            "--openrouter_include_reasoning",
            "--chat_token_parameter",
            "max_tokens",
        ]
    )
    assert parsed.provider == "openrouter_chat"
    assert parsed.openrouter_provider_only == [
        "gmicloud/fp8",
        "modal/mxfp4",
        "baseten/fp8",
    ]
    assert parsed.openrouter_provider_order == [
        "modal/mxfp4",
        "baseten/fp8",
    ]
    assert parsed.openrouter_allow_fallbacks is False
    provenance = _provider_contract(parsed, "https://openrouter.ai/api/v1")
    assert provenance["openrouter_routing"] == {
        "only": ["gmicloud/fp8", "modal/mxfp4", "baseten/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "enforce_distillable_text": True,
        "order": ["modal/mxfp4", "baseten/fp8"],
    }
    assert provenance["openrouter_reasoning"] == {
        "enabled": True,
        "included_in_response": True,
    }


def test_openrouter_evaluation_only_allows_non_distillable_route() -> None:
    base = [
        *_minimal_cli("openrouter_chat"),
        "--model",
        "z-ai/glm-5.2",
        "--base_url",
        "https://openrouter.ai/api/v1",
        "--openrouter_provider_only",
        "novita/fp8",
        "--openrouter_require_parameters",
        "--evaluation_only",
        "--openrouter_reasoning",
        "enabled",
        "--openrouter_reasoning_effort",
        "xhigh",
        "--openrouter_include_reasoning",
        "--chat_token_parameter",
        "max_tokens",
    ]
    parsed = parse_args(base)
    assert parsed.evaluation_only is True
    assert parsed.openrouter_enforce_distillable_text is False
    provenance = _provider_contract(parsed, "https://openrouter.ai/api/v1")
    assert provenance["openrouter_reasoning"] == {
        "enabled": True,
        "included_in_response": True,
        "effort": "xhigh",
    }
    assert provenance["openrouter_routing"] == {
        "only": ["novita/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "enforce_distillable_text": False,
    }


def test_openrouter_non_distillable_route_requires_evaluation_only() -> None:
    common = [
        *_minimal_cli("openrouter_chat"),
        "--base_url",
        "https://openrouter.ai/api/v1",
        "--openrouter_provider_only",
        "novita/fp8",
        "--openrouter_require_parameters",
    ]
    with pytest.raises(SystemExit):
        parse_args(
            [
                *common,
                "--model",
                "z-ai/glm-5.2",
            ]
        )


@pytest.mark.parametrize(
    "provider_order",
    [
        ("Modal",),
        ("modal", "modal"),
        ("modal/too/many",),
    ],
)
def test_openrouter_cli_rejects_invalid_or_duplicate_provider_order(
    provider_order: tuple[str, ...],
) -> None:
    argv = [
        *_minimal_cli("openrouter_chat"),
        "--base_url",
        "https://openrouter.ai/api/v1",
        "--openrouter_provider_only",
        "together",
        "--openrouter_require_parameters",
        "--openrouter_enforce_distillable_text",
    ]
    for value in provider_order:
        argv.extend(["--openrouter_provider_order", value])
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_openrouter_cli_rejects_provider_order_outside_allowlist() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_minimal_cli("openrouter_chat"),
                "--base_url",
                "https://openrouter.ai/api/v1",
                "--openrouter_provider_only",
                "together",
                "--openrouter_provider_order",
                "modal/mxfp4",
                "--openrouter_require_parameters",
                "--openrouter_enforce_distillable_text",
            ]
        )


def test_openrouter_provider_order_is_rejected_for_other_providers() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_minimal_cli("azure_v1_responses"),
                "--openrouter_provider_order",
                "modal",
            ]
        )


def test_cli_rejects_negative_eligible_task_offset() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_minimal_cli("azure_v1_responses"),
                "--eligible_task_offset",
                "-1",
            ]
        )
