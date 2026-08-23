from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.preprocessing import build_t5gemma2_typed_api_visible_split as split
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_api_rescue_continuation as continuation_cascade
from scripts.training import t5gemma2_typed_visible_failure_projection as projection
from scripts.training.t5gemma2_local_rs_sft_pilot import PilotTask, PrivateGate


def _task(task_id: str = "task-0") -> PilotTask:
    source = "Implement fn0.\nF2\ncompressed enriched assembly"
    gold = "int fn0(int p0) => 9;"
    return PilotTask(
        task_id=task_id,
        source=source,
        source_sha256=base.sha256_text(source),
        visible_tests="void main() { expect(fn0(1), 2); }",
        gold_target=gold,
        gold_target_sha256=base.sha256_text(gold),
        f2_row={"task_id": task_id, "text": "F2\ncompressed enriched assembly"},
        split_binding_sha256="a" * 64,
    )


def _candidate(code: str, index: int, *, compiled: bool, passed: bool) -> dict:
    diagnostic = base.COMPILED_NO_DIAGNOSTIC if compiled else "Error: expected ';'"
    return {
        "sample_index": index,
        "origin": "local_student_direct",
        "code": code,
        "code_sha256": base.sha256_text(code),
        "visible": {"compiled": compiled, "passed": passed},
        "safe_visible_diagnostic": diagnostic,
        "safe_visible_diagnostic_sha256": base.sha256_text(diagnostic),
        "diagnostic_source": "sealed_visible_TRAIN_split",
        "private_complete_diagnostic_consumed": False,
    }


def _projection_terminal(
    task: PilotTask, passes: int = 0, *, singleton_compile_only: bool = False
) -> dict:
    candidates = [
        _candidate(
            f"int fn0(int p0) => {index};",
            index,
            compiled=True,
            passed=index < passes,
        )
        for index in range(4)
    ]
    return {
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "visible_unique_passes": passes,
        "base_candidates": candidates,
        "repair_groups": [],
        "journal_event_sha256": "b" * 64,
        "api_eligibility_stratum": (
            "singleton_stdout_compile_call_only"
            if singleton_compile_only
            else "semantic_visible_all_zero"
        ),
        "api_eligible": singleton_compile_only or passes == 0,
    }


def test_split_expect_stdout_and_singleton_without_leaking_single_answer() -> None:
    expect_tests = "void main() { expect(fn0(1), 2); expect(fn0(2), 3); }"
    visible, holdback, metadata = split.split_task_harness(
        task_id="expect", tests=expect_tests, seed=7
    )
    assert metadata["strategy"] == "established_expect_half"
    assert metadata["visible_count"] == metadata["holdback_count"] == 1
    assert visible != holdback

    stdout = """
Future<void> main() async {
  final _captured = <String>[];
  await runZoned(() async { await Future.sync(() => fn0()); },
    zoneSpecification: ZoneSpecification(print: (self, parent, zone, line) { _captured.add(line); }));
  final _actual = _captured.isEmpty ? '' : '${_captured.join('\\n')}\\n';
  const _expected = "secret-answer\\n";
  if (_actual != _expected) { throw StateError('bad'); }
}
"""
    visible, holdback, metadata = split.split_task_harness(
        task_id="stdout-one", tests=stdout, seed=7
    )
    assert metadata["strategy"] == "stdout_singleton_compile_and_call_visible"
    assert "secret-answer" not in visible
    assert "fn0()" in visible
    assert "secret-answer" in holdback


def test_generated_ag_case_split_is_disjoint() -> None:
    tests = """
Future<void> main() async {
  final _v0=await _agEval(() => fn0(0));
  if (_v0 != "ok:0") throw StateError('case 0');
  final _v1=await _agEval(() => fn0(1));
  if (_v1 != "ok:1") throw StateError('case 1');
  final _v2=await _agEval(() => fn0(2));
  if (_v2 != "ok:2") throw StateError('case 2');
}
"""
    visible, holdback, metadata = split.split_task_harness(
        task_id="ag", tests=tests, seed=11
    )
    assert metadata["strategy"] == "generated_ag_case_half"
    assert metadata["visible_count"] == 1
    assert metadata["holdback_count"] == 2
    for index in range(3):
        marker = f"final _v{index}="
        assert (marker in visible) is (index in metadata["visible_case_indices"])
        assert (marker in holdback) is (index in metadata["holdback_case_indices"])


def test_projection_ignores_private_binary_and_recomputes_visible() -> None:
    task = _task()
    # The local binary says every candidate passed private full acceptance.
    # The projection evaluator still decides eligibility exclusively from the
    # public split supplied to it.
    local_terminal = {
        "base_candidates": [
            {
                "sample_index": index,
                "origin": "local_student_direct",
                "code": f"int fn0(int p0) => {index};",
                "code_sha256": base.sha256_text(f"int fn0(int p0) => {index};"),
                "visible": {"compiled": True, "passed": True},
            }
            for index in range(4)
        ]
    }
    original = projection._evaluate
    projection._evaluate = lambda *_args, **_kwargs: base.Evaluation(True, False, "")
    try:
        row = projection.project_task(
            task=task,
            terminal=local_terminal,
            task_position=0,
            timeout=1,
            workers=1,
            visible_metadata={
                "strategy": "established_expect_half",
                "semantic_visible_cases": 1,
            },
        )
    finally:
        projection._evaluate = original
    assert row["visible_unique_passes"] == 0
    assert row["private_complete_outcome_consumed_for_eligibility"] is False
    assert all(not candidate["visible"]["passed"] for candidate in row["base_candidates"])


def test_projection_run_binds_visible_metadata_to_scheduled_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task()
    local_terminal = {
        "base_candidates": [
            {
                "sample_index": index,
                "origin": "local_student_direct",
                "code": f"int fn0(int p0) => {index};",
                "code_sha256": base.sha256_text(f"int fn0(int p0) => {index};"),
                "visible": {"compiled": True, "passed": True},
            }
            for index in range(4)
        ]
    }
    metadata = {
        "strategy": "established_expect_half",
        "semantic_visible_cases": 1,
    }
    context = SimpleNamespace(
        scheduled_tasks=[task],
        terminals=[local_terminal],
        visible_metadata={task.task_id: metadata},
        source_journal_record={},
        input_record={},
    )
    monkeypatch.setattr(projection, "EXPECTED_TASKS", 1)
    monkeypatch.setattr(projection, "validate_dart_binary", lambda: None)
    monkeypatch.setattr(
        projection,
        "_evaluate",
        lambda *_args, **_kwargs: base.Evaluation(True, False, ""),
    )

    report = projection.run(
        SimpleNamespace(output_dir=str(tmp_path), timeout=1, evaluation_workers=1),
        context_loader=lambda _args: context,
    )

    assert report["status"] == "complete"
    terminal = json.loads(
        (tmp_path / "visible_projection.journal.jsonl").read_text().splitlines()[1]
    )
    assert terminal["task_id"] == task.task_id
    assert terminal["visible_split_strategy"] == metadata["strategy"]


def test_singleton_stdout_is_api_eligible_even_when_compile_call_passes() -> None:
    task = _task("singleton")
    context = cascade.TypedSourceContext(
        [task],
        {task.task_id: PrivateGate(task.task_id, "PRIVATE", task.split_binding_sha256)},
        [task],
        [{"task_id": task.task_id, "visible_unique_passes": 4}],
        {},
        {},
    )
    terminal = _projection_terminal(task, passes=4, singleton_compile_only=True)
    selected = cascade.select_visible_zero_tasks(
        context=context,
        projection_terminals=[terminal],
        seed=1,
        excluded_ids=set(),
    )
    assert [row[1].task_id for row in selected] == [task.task_id]
    plans, _ = cascade.build_visible_only_plans(
        selected=selected,
        gates=context.gates,
    )
    assert len(plans) == 1


def test_visible_projection_alone_controls_api_eligibility() -> None:
    first = _task("first")
    second = _task("second")
    context = cascade.TypedSourceContext(
        [first, second],
        {
            first.task_id: PrivateGate(first.task_id, "PRIVATE-A", first.split_binding_sha256),
            second.task_id: PrivateGate(second.task_id, "PRIVATE-B", second.split_binding_sha256),
        },
        [first, second],
        # Deliberately contradictory local/private pass fields.  Selection must
        # not read them.
        [
            {"task_id": first.task_id, "visible_unique_passes": 4},
            {"task_id": second.task_id, "visible_unique_passes": 0},
        ],
        {},
        {},
    )
    selected = cascade.select_visible_zero_tasks(
        context=context,
        projection_terminals=[
            _projection_terminal(first, passes=0),
            _projection_terminal(second, passes=1),
        ],
        seed=3,
        excluded_ids=set(),
    )
    assert [row[1].task_id for row in selected] == ["first"]


def test_prompt_contains_public_evidence_but_not_private_or_gold() -> None:
    task = _task()
    gate = PrivateGate(task.task_id, "PRIVATE COMPLETE SUITE", task.split_binding_sha256)
    terminal = _projection_terminal(task, passes=0)
    plans, provenance = cascade.build_visible_only_plans(
        selected=[(0, task, terminal)], gates={task.task_id: gate}
    )
    prompt = cascade.build_typed_provider_prompt(plans[0], plans[0].parents[0])
    assert task.source in prompt
    assert task.visible_tests in prompt
    assert plans[0].parents[0].code in prompt
    assert "PRIVATE COMPLETE SUITE" not in prompt
    assert task.gold_target not in prompt
    assert provenance["complete_private_suite_used_for_diagnostic"] is False


def test_direct_publisher_emits_no_repair_reasoning_or_gold(tmp_path: Path) -> None:
    task = _task()
    gate = PrivateGate(task.task_id, "PRIVATE", task.split_binding_sha256)
    plans, _ = cascade.build_visible_only_plans(
        selected=[(0, task, _projection_terminal(task))],
        gates={task.task_id: gate},
    )
    code = "int fn0(int p0) => 2;"
    selected = {
        "task_id": task.task_id,
        "code": code,
        "code_sha256": base.sha256_text(code),
        "slot_position": 0,
        "parent_code_sha256": plans[0].parents[0].code_sha256,
        "diagnostic_sha256": plans[0].parents[0].diagnostic_sha256,
        "visible_passed": True,
        "private_gate_passed": True,
    }
    output = cascade.publish_direct_only(
        output_dir=tmp_path,
        plans=plans,
        verifications=[{"selected_target": selected}],
        contract_sha256="c" * 64,
        provider_phase=cascade.PHASE_KIMI_INITIAL,
        provider_model=cascade.KIMI_MODEL,
        stability_runs=2,
    )
    assert set(output["files"]) == {"direct_targets"}
    assert not (tmp_path / "repair_policy_targets.jsonl").exists()
    row = json.loads((tmp_path / "direct_targets.jsonl").read_text().strip())
    assert row["reasoning_present"] is False
    assert row["repair_conditioned_training_source_present"] is False
    assert row["gold_replay"] is False


def test_kimi_yield_gate_and_targeted_retry_selection() -> None:
    assert cascade.cohort_decision(
        initial_verified_ids=[f"a{i}" for i in range(7)]
    )["continue_kimi"] is False
    assert cascade.cohort_decision(
        initial_verified_ids=[f"a{i}" for i in range(7)],
        retry_verified_ids=["r0"],
    )["continue_kimi"] is True


def test_kimi_continuation_excludes_all_prior_provider_schedules() -> None:
    tasks = [_task(f"task-{index}") for index in range(110)]
    visible = [
        (index, task, _projection_terminal(task))
        for index, task in enumerate(tasks)
    ]
    prior = [
        {
            "phase": cascade.PHASE_KIMI_INITIAL,
            "cohort_index": 0,
            "scheduled_task_ids": [task.task_id for task in tasks[:50]],
            "verified_task_ids": [task.task_id for task in tasks[:8]],
            "retry_eligible_task_ids": [],
        },
        {
            "phase": cascade.PHASE_SONNET_RESIDUAL,
            "cohort_index": 0,
            "scheduled_task_ids": [task.task_id for task in tasks[48:58]],
            "verified_task_ids": [tasks[55].task_id],
            "retry_eligible_task_ids": [],
        },
    ]
    selected, record = continuation_cascade.phase_selection(
        args=SimpleNamespace(
            phase=cascade.PHASE_KIMI_INITIAL,
            cohort_index=1,
            max_tasks=50,
            fixed_kimi_cohort_limit=2,
            budget_skipped_kimi_retry_tasks=0,
            budget_skipped_kimi_retry_task_ids_sha256="",
        ),
        all_visible_zero=visible,
        prior_records=prior,
    )
    assert [row[1].task_id for row in selected] == [
        f"task-{index}" for index in range(58, 108)
    ]
    assert record["prior_scheduled_tasks_excluded"] == 50
    assert record["prior_all_provider_scheduled_tasks_excluded"] == 58


def test_continuation_adapter_attests_itself_and_restores_base_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        phase=cascade.PHASE_KIMI_INITIAL,
        cohort_index=1,
        fixed_kimi_cohort_limit=2,
        max_input_tokens_per_call=16384,
    )
    original_file = cascade.__file__
    original_selection = cascade._phase_selection
    observed: dict[str, object] = {}
    monkeypatch.setattr(cascade, "parse_args", lambda _argv: args)

    def fake_run(value: object) -> dict:
        observed["args"] = value
        observed["file"] = cascade.__file__
        observed["selection"] = cascade._phase_selection
        return {"status": "complete"}

    monkeypatch.setattr(cascade, "run", fake_run)
    assert continuation_cascade.run([]) == {"status": "complete"}
    assert observed["args"] is args
    assert Path(str(observed["file"])).name == Path(
        continuation_cascade.__file__
    ).name
    assert observed["selection"] is continuation_cascade.phase_selection
    assert cascade.__file__ == original_file
    assert cascade._phase_selection is original_selection


def test_fixed_one_cohort_budget_handoff_allows_sonnet_after_high_kimi_yield() -> None:
    tasks = [_task(f"task-{index}") for index in range(12)]
    visible = [(index, task, _projection_terminal(task)) for index, task in enumerate(tasks)]
    prior = [
        {
            "phase": cascade.PHASE_KIMI_INITIAL,
            "cohort_index": 0,
            "scheduled_task_ids": [task.task_id for task in tasks[:10]],
            "verified_task_ids": [task.task_id for task in tasks[:8]],
            "retry_eligible_task_ids": [],
        }
    ]
    selected, record = cascade._phase_selection(
        args=SimpleNamespace(
            phase=cascade.PHASE_SONNET_RESIDUAL,
            max_tasks=4,
            fixed_kimi_cohort_limit=1,
            budget_skipped_kimi_retry_tasks=0,
            budget_skipped_kimi_retry_task_ids_sha256="",
        ),
        all_visible_zero=visible,
        prior_records=prior,
    )
    assert record["kimi_stopped_for_fixed_cohort_limit"] is True
    assert [row[1].task_id for row in selected] == [
        "task-8",
        "task-9",
        "task-10",
        "task-11",
    ]


def test_phase_profiles_lock_models_lengths_prices_and_caps() -> None:
    common = dict(
        phase=cascade.PHASE_KIMI_INITIAL,
        max_parents_per_task=1,
        samples_per_parent=1,
        stability_runs=2,
        evaluation_only=False,
        exploratory_terminal_prefix=0,
        allow_unpinned_inputs=False,
        provider="openrouter_chat",
        model=cascade.KIMI_MODEL,
        max_output_tokens=2048,
        openrouter_reasoning="enabled",
        openrouter_reasoning_effort="low",
        chat_token_parameter="max_tokens",
        max_tasks=50,
        retry_parse_failures_or_truncations_report="",
        openrouter_require_parameters=True,
        openrouter_enforce_distillable_text=True,
        openrouter_provider_only=["together"],
        max_calls=50,
        max_input_tokens_per_call=65536,
        max_input_tokens_total=3276800,
        max_output_tokens_total=102400,
        max_total_tokens=3379200,
        max_usd="12",
        input_usd_per_million="3",
        output_usd_per_million="15",
        expected_retry_parse_failures_or_truncations_tasks=-1,
    )
    cascade.validate_phase_profile(SimpleNamespace(**common))
    wrong = dict(common, max_output_tokens=8192)
    with pytest.raises(ValueError, match="max_output_tokens"):
        cascade.validate_phase_profile(SimpleNamespace(**wrong))
    too_expensive = dict(common, max_usd="12.01")
    with pytest.raises(ValueError, match="spend cap"):
        cascade.validate_phase_profile(SimpleNamespace(**too_expensive))


def test_launchers_are_manual_and_encode_the_sealed_profiles() -> None:
    root = Path(__file__).resolve().parents[1]
    prepare = (root / "deploy/vast/t5gemma2_typed_api_rescue_prepare.sh").read_text()
    runner = (root / "deploy/vast/t5gemma2_typed_api_rescue_cascade.sh").read_text()
    prepare_conf = (root / "deploy/vast/t5gemma2-typed-api-rescue-prepare.conf").read_text()
    runner_conf = (root / "deploy/vast/t5gemma2-typed-api-rescue-cascade.conf").read_text()
    assert "t5gemma2_typed_visible_failure_projection.py" in prepare
    assert "OPENROUTER_API_KEY" not in prepare
    assert "ANTHROPIC_API_KEY" not in prepare
    assert "--model moonshotai/kimi-k3" in runner
    assert "--openrouter_reasoning_effort low" in runner
    assert "MAX_OUTPUT=2048" in runner
    assert "MAX_OUTPUT=8192" in runner
    assert "--model claude-sonnet-5" in runner
    assert "--anthropic_thinking adaptive --anthropic_effort high" in runner
    assert "MAX_OUTPUT=16384" in runner
    assert "T5GEMMA_TYPED_API_SCHEDULE_SHA256" in runner
    assert "autostart=false" in prepare_conf
    assert "autostart=false" in runner_conf
