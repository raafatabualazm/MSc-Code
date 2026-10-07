from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import torch


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training import seq2seq_verpo_core as core
from scripts.training import t5gemma2_compiler_feedback_verpo as runner
from scripts.training import t5gemma2_enriched_sft as sft


VISIBLE_TESTS = """void main() {
  final candidate = fn0;
  expect(candidate(1), 2); // PUBLIC_TEST_SENTINEL
  expect(candidate(2), 3);
}

void expect(dynamic actual, dynamic expected) {
  if (actual == expected) return;
  throw '$actual != $expected';
}
"""


def test_per_case_scoring_runs_after_combined_compile_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    answers = iter(
        [
            (False, False, "combined static failure", "combined source"),
            (True, True, "", "first source"),
            (False, False, "second static failure", "second source"),
        ]
    )

    def fake_evaluate(
        _candidate: str,
        test_code: str,
        task_id: str,
        *,
        timeout: int,
        stability_runs: int,
    ):
        del test_code, timeout, stability_runs
        calls.append(task_id)
        return next(answers)

    monkeypatch.setattr(
        runner,
        "evaluate_dart_jit_tests_detail",
        fake_evaluate,
    )
    detail = runner.score_dart_candidate(
        "int fn0(int x) => x;",
        VISIBLE_TESTS,
        "task-1",
        timeout=30,
        stability_runs=1,
    )
    assert calls == [
        "task-1-full",
        "task-1-test-0",
        "task-1-test-1",
    ]
    assert detail == {
        "compiled": False,
        "full_pass": False,
        "test_passes": [True, False],
        "diagnostic": "combined static failure",
    }


def test_runtime_provenance_hashes_every_training_dependency() -> None:
    provenance = runner.runtime_provenance()
    assert provenance["schema"] == runner.RUNTIME_PROVENANCE_SCHEMA
    code = provenance["code"]
    assert set(code) == {
        "trainer",
        "seq2seq_core",
        "dart_evaluator",
        "feedback_boundary_builder",
        "enriched_sft_helper",
    }
    project_root = Path(runner.__file__).resolve().parents[2]
    for record in code.values():
        assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        assert (
            runner._sha256_file(project_root / record["relative_path"])
            == record["sha256"]
        )
    assert provenance["code_bundle_sha256"] == sft.canonical_sha256(code)
    assert provenance["torch"]["version"] == str(torch.__version__)
    assert provenance["cuda"]["available"] is torch.cuda.is_available()
    assert provenance["cuda"]["device_count"] == torch.cuda.device_count()


def test_metric_and_result_binding_uses_canonical_run_contract() -> None:
    contract = {
        "schema": runner.RUN_SCHEMA,
        "runtime_provenance": {"code_bundle_sha256": "a" * 64},
    }
    source = {"schema": runner.ROLLOUT_SCHEMA, "update": 1}
    bound = runner.bind_run_contract(source, contract)
    assert source == {"schema": runner.ROLLOUT_SCHEMA, "update": 1}
    assert bound["run_contract_sha256"] == sft.canonical_sha256(contract)
    with pytest.raises(ValueError, match="already contains"):
        runner.bind_run_contract(bound, contract)


def trajectory(
    candidate: str,
    *,
    compiled: bool,
    passes: list[bool],
    state_kind: str = "base",
    diagnostic: str = "",
) -> runner.RolloutTrajectory:
    return runner.RolloutTrajectory(
        task_id="task-1",
        state_kind=state_kind,
        source="SEALED_BINARY_SOURCE",
        source_sha256=core.sha256_text("SEALED_BINARY_SOURCE"),
        actions=(7, 1),
        candidate=candidate,
        detail={
            "compiled": compiled,
            "full_pass": bool(compiled and all(passes)),
            "test_passes": list(passes),
            "diagnostic": diagnostic,
        },
        saved_logprobs=torch.tensor([-1.0, -2.0]),
    )


def test_compiler_repairs_require_all_zero_and_select_noncompiling_diversity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated: list[dict[str, object]] = []

    def fake_generate_group(**kwargs):
        generated.append(kwargs)
        return [
            trajectory(
                "int fn0(int x) => x;",
                compiled=True,
                passes=[False, False],
                state_kind="compiler_repair",
            )
        ]

    monkeypatch.setattr(runner, "generate_group", fake_generate_group)
    task = runner.VerpoTask(
        task_id="task-1",
        source="SEALED_BINARY_SOURCE",
        source_sha256=core.sha256_text("SEALED_BINARY_SOURCE"),
        feedback_tests=VISIBLE_TESTS,
        replay_target="int fn0(int x) => x + 1;",
    )

    # A visible partial pass is not a dead group, even if another parent fails
    # compilation.
    live_group = [
        trajectory(
            "int fn0(int x) => ;",
            compiled=False,
            passes=[False, False],
            diagnostic="Error: Expected an expression.",
        ),
        trajectory(
            "int fn0(int x) => x + 1;",
            compiled=True,
            passes=[True, False],
        ),
    ]
    assert (
        runner.build_repair_groups(
            model=None,
            tokenizer=None,
            task=task,
            base_group=live_group,
            max_parents=2,
            repair_group_size=3,
            max_source_tokens=1000,
            max_new_tokens=100,
            temperature=0.8,
            device=torch.device("cpu"),
        )
        == []
    )

    # All-zero but entirely compiling groups have no compiler feedback state.
    compiling_zero = [
        trajectory(
            f"int fn0(int x) => {value};",
            compiled=True,
            passes=[False, False],
        )
        for value in range(3)
    ]
    assert (
        runner.build_repair_groups(
            model=None,
            tokenizer=None,
            task=task,
            base_group=compiling_zero,
            max_parents=2,
            repair_group_size=3,
            max_source_tokens=1000,
            max_new_tokens=100,
            temperature=0.8,
            device=torch.device("cpu"),
        )
        == []
    )

    candidates = [
        "int fn0(int x) { return 0 + ; }",
        "int fn0(int x) { return 1 + ; }",
        ("int fn0(int x) { while (x > 0) { x--; } " "return x + ; }"),
    ]
    dead_group = [
        trajectory(
            candidate,
            compiled=False,
            passes=[False, False],
            diagnostic=(
                "C:\\tmp\\test.dart:1:7: Error: Expected an expression.\n"
                "Expected: <99>\nActual: <0>"
            ),
        )
        for candidate in candidates
    ]
    expected_indices = core.max_min_diverse_indices(candidates, 2)
    expected_parent_hashes = {
        core.sha256_text(candidates[index]) for index in expected_indices
    }
    groups = runner.build_repair_groups(
        model=None,
        tokenizer=None,
        task=task,
        base_group=dead_group,
        max_parents=2,
        repair_group_size=3,
        max_source_tokens=1000,
        max_new_tokens=100,
        temperature=0.8,
        device=torch.device("cpu"),
    )
    assert len(groups) == 2
    selected_parent_hashes = {
        str(call["parent_candidate_sha256"]) for call in generated
    }
    assert selected_parent_hashes == expected_parent_hashes
    assert all(call["state_kind"] == "compiler_repair" for call in generated)
    assert all(call["group_size"] == 3 for call in generated)
    assert all(
        str(call["source"]).startswith("SEALED_BINARY_SOURCE\n") for call in generated
    )
    assert all("PUBLIC_TEST_SENTINEL" not in str(call["source"]) for call in generated)
    assert all("Expected: <99>" not in str(call["source"]) for call in generated)


def test_repair_advantages_never_mutate_base_actions_or_advantages() -> None:
    base = [
        trajectory(
            "int fn0(int x) => ;",
            compiled=False,
            passes=[False, False],
        ),
        trajectory(
            "int fn0(int x) => x;",
            compiled=True,
            passes=[True, False],
        ),
    ]
    repair = [
        trajectory(
            "int fn0(int x) => x;",
            compiled=True,
            passes=[False, False],
            state_kind="compiler_repair",
        ),
        trajectory(
            "int fn0(int x) => x + 1;",
            compiled=True,
            passes=[True, True],
            state_kind="compiler_repair",
        ),
    ]
    runner.assign_advantages(
        base,
        alpha=2.0,
        local_weight=1.0,
        compile_weight=0.25,
    )
    base_snapshot = [(item.actions, item.advantage, item.state_kind) for item in base]
    runner.assign_advantages(
        repair,
        alpha=2.0,
        local_weight=1.0,
        compile_weight=0.25,
    )
    assert [
        (item.actions, item.advantage, item.state_kind) for item in base
    ] == base_snapshot
    assert all(item.state_kind == "base" for item in base)
    assert all(item.state_kind == "compiler_repair" for item in repair)
    assert repair[1].advantage > repair[0].advantage


def test_trajectory_record_persists_pad_before_eos_incident_count() -> None:
    item = trajectory(
        "int fn0(int x) => x;",
        compiled=True,
        passes=[False, False],
    )
    item.sampled_pad_before_eos = 0
    record = runner._trajectory_record(item)
    assert record["sampled_pad_before_eos"] == 0


def test_one_optimizer_step_uses_predeclared_slot_normalization() -> None:
    slots = runner.declared_step_trajectory_slots(
        tasks_per_update=2,
        group_size=4,
        max_repair_parents=2,
        repair_group_size=4,
    )
    assert slots == 24

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    # Only two of the 24 sealed slots happened to carry nonzero signal.
    first = runner.normalize_step_slot_loss(
        parameter,
        declared_slots=slots,
    )
    second = runner.normalize_step_slot_loss(
        3.0 * parameter,
        declared_slots=slots,
    )
    first.backward(retain_graph=True)
    second.backward()
    assert float(parameter.grad) == pytest.approx(4.0 / 24.0)


def test_generation_kwargs_explicitly_disable_distribution_truncation() -> None:
    kwargs = runner._generation_kwargs(
        max_new_tokens=123,
        temperature=0.8,
        pad_token_id=0,
        eos_token_ids=(1, 11),
    )
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 0
    assert kwargs["typical_p"] == 1.0
    assert kwargs["min_p"] is None
    assert kwargs["top_h"] is None
    assert kwargs["epsilon_cutoff"] == 0.0
    assert kwargs["eta_cutoff"] == 0.0
    assert kwargs["repetition_penalty"] == 1.0
    assert kwargs["encoder_repetition_penalty"] == 1.0
    assert kwargs["no_repeat_ngram_size"] == 0
    assert kwargs["encoder_no_repeat_ngram_size"] == 0
    assert kwargs["bad_words_ids"] is None
    assert kwargs["suppress_tokens"] == [0]
    assert kwargs["constraints"] is None
    assert kwargs["force_words_ids"] is None
    assert kwargs["forced_bos_token_id"] is None
    assert kwargs["forced_eos_token_id"] is None
    assert kwargs["sequence_bias"] is None
    assert kwargs["watermarking_config"] is None
    assert kwargs["stop_strings"] is None
    assert kwargs["max_new_tokens"] == 123
    assert "max_length" not in kwargs
    assert kwargs["pad_token_id"] == 0
    assert kwargs["eos_token_id"] == [1, 11]

    shared = runner._generation_kwargs(
        max_new_tokens=123,
        temperature=0.8,
        pad_token_id=1,
        eos_token_ids=(1,),
    )
    assert shared["suppress_tokens"] is None


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _f2_row() -> dict[str, object]:
    text = "F2\nBINARY_SOURCE_SENTINEL\n"
    return {
        "schema": sft.F2_ROW_SCHEMA,
        "representation_schema": sft.REPRESENTATION_SCHEMA,
        "task_id": "task-1",
        "text": text,
        "text_sha256": core.sha256_text(text),
        "verified": dict(sft._REQUIRED_F2_ATTESTATIONS),
    }


def test_loaded_encoder_source_excludes_visible_private_and_gold_text(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.jsonl"
    f2 = tmp_path / "f2.jsonl"
    row = {
        "task_id": "task-1",
        "feedback_tests": VISIBLE_TESTS,
        "supervised_target": "GOLD_TARGET_SENTINEL",
    }
    _write_jsonl(rollout, [row])
    _write_jsonl(f2, [_f2_row()])
    tasks = runner.load_verpo_tasks(rollout, f2)
    assert len(tasks) == 1
    assert "BINARY_SOURCE_SENTINEL" in tasks[0].source
    assert "PUBLIC_TEST_SENTINEL" not in tasks[0].source
    assert "GOLD_TARGET_SENTINEL" not in tasks[0].source
    assert "PUBLIC_TEST_SENTINEL" in tasks[0].feedback_tests
    assert tasks[0].replay_target == "GOLD_TARGET_SENTINEL"

    _write_jsonl(
        rollout,
        [{**row, "acceptance_tests": "PRIVATE_TEST_SENTINEL"}],
    )
    with pytest.raises(ValueError, match="forbidden private/test fields"):
        runner.load_verpo_tasks(rollout, f2)

    bad_f2 = {**_f2_row(), "feedback_tests": VISIBLE_TESTS}
    _write_jsonl(rollout, [row])
    _write_jsonl(f2, [bad_f2])
    with pytest.raises(ValueError, match="forbidden field"):
        runner.load_verpo_tasks(rollout, f2)


def _checkpoint_dir(
    root: Path,
    update: int,
    contract: dict[str, object],
) -> Path:
    path = root / f"checkpoint-optstep-{update:06d}"
    path.mkdir()
    (path / "adapter").mkdir()
    (path / "tokenizer").mkdir()
    (path / "training_state.pt").write_bytes(b"sealed fixture")
    (path / "run_contract.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    return path


def test_checkpoint_pruning_validates_all_dirs_before_refusing_foreign(
    tmp_path: Path,
) -> None:
    root_contract = {"schema": runner.RUN_SCHEMA, "seed": 7}
    foreign_contract = {"schema": runner.RUN_SCHEMA, "seed": 999}
    (tmp_path / "run_contract.json").write_text(
        json.dumps(root_contract),
        encoding="utf-8",
    )
    first = _checkpoint_dir(tmp_path, 1, root_contract)
    foreign = _checkpoint_dir(tmp_path, 2, foreign_contract)
    latest = _checkpoint_dir(tmp_path, 3, root_contract)

    with pytest.raises(ValueError, match="foreign checkpoint"):
        runner._prune_checkpoints(tmp_path, keep=1)
    # Validation is two-phase: encountering a foreign directory cannot leave
    # the legitimate older checkpoint half-pruned.
    assert first.is_dir()
    assert foreign.is_dir()
    assert latest.is_dir()

    (foreign / "run_contract.json").write_text(
        json.dumps(root_contract),
        encoding="utf-8",
    )
    runner._prune_checkpoints(tmp_path, keep=1)
    assert not first.exists()
    assert not foreign.exists()
    assert latest.is_dir()
