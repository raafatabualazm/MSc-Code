from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.audit_t5gemma2_typed_proxy_reward_surface import (
    AUDIT_JOURNAL_SCHEMA,
    FEEDBACK_SPLIT_SCHEMA,
    SOURCE_JOURNAL_SCHEMA,
    SOURCE_RUN_SCHEMA,
    SOURCE_TRAINING_SCHEMA,
    build_audit_contract,
    deterministic_sample,
    load_audit_inputs,
    _metric_decision,
    run_audit,
)
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    load_journal,
    sha256_file,
)
from scripts.training.seq2seq_verpo_core import (
    verpo_execution_compile_advantages,
)


def _write_feedback(path: Path, task_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task_id in task_ids:
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "feedback_tests": "case-0||case-1||case-2",
                        "verpo_feedback_split_schema": FEEDBACK_SPLIT_SCHEMA,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _source_contract() -> dict:
    return {
        "schema": SOURCE_RUN_SCHEMA,
        "checkpoint_stage": "typed_direct",
        "checkpoint": {
            "checkpoint_update": 58,
            "training_stage_schema": SOURCE_TRAINING_SCHEMA,
        },
        "sampling": {
            "samples_per_task": 4,
            "generation_batch_size": 4,
            "max_repair_parents": 0,
            "repair_samples": 0,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_source_tokens": 32768,
            "max_new_tokens": 4096,
        },
        "input": {
            "heldout_175_opened": False,
            "complete_acceptance_model_visible": False,
        },
        "frontier_api_calls": False,
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
    }


def _write_harvest(path: Path, task_ids: list[str]) -> None:
    contract = _source_contract()
    append_event(
        path,
        {
            "schema": SOURCE_JOURNAL_SCHEMA,
            "event": "header",
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    )
    for position, task_id in enumerate(task_ids):
        candidates = []
        for sample_index in range(4):
            code = f"{task_id}-candidate-{sample_index}"
            candidates.append(
                {
                    "code": code,
                    "code_sha256": __import__("hashlib").sha256(
                        code.encode("utf-8")
                    ).hexdigest(),
                    "origin": "local_student_direct",
                    "sample_index": sample_index,
                    "visible": {"compiled": True, "passed": False},
                }
            )
        append_event(
            path,
            {
                "schema": SOURCE_JOURNAL_SCHEMA,
                "event": "task_terminal",
                "task_position": position,
                "task_id": task_id,
                "source_sha256": f"{position + 1:064x}",
                "typed_contract_sha256": f"{position + 100:064x}",
                "base_candidates": candidates,
                "repair_groups": [],
                "all_generation_completed_before_private_gate": True,
                "private_feedback_serialized_to_model": False,
                "private_failure_triggers_generation": False,
                "binary_field_semantics": "complete_train_acceptance_private_gate",
            },
        )
    append_event(
        path,
        {
            "schema": SOURCE_JOURNAL_SCHEMA,
            "event": "complete",
            "tasks": len(task_ids),
            "terminal_task_ids_sha256": canonical_sha256(task_ids),
        },
    )


def _inputs(tmp_path: Path, task_ids: list[str], sample_size: int):
    harvest = tmp_path / "harvest.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_harvest(harvest, task_ids)
    _write_feedback(feedback, list(reversed(task_ids)))
    return load_audit_inputs(
        harvest,
        feedback,
        split_fn=lambda value: value.split("||"),
        expected_harvest_journal_sha256=sha256_file(harvest),
        expected_harvest_chain_head_sha256=sha256_file(
            Path(str(harvest) + ".chain-head.json")
        ),
        expected_feedback_sha256=sha256_file(feedback),
        expected_harvest_tasks=len(task_ids),
        expected_feedback_tasks=len(task_ids),
        expected_intersection_tasks=len(task_ids),
        sample_size=sample_size,
        sample_seed=42,
    )


def _fake_score(code: str, _tests: str, _task: str, **_kwargs):
    task_id, raw_index = code.rsplit("-candidate-", 1)
    index = int(raw_index)
    if task_id == "task-a":
        return {
            "compiled": True,
            "full_pass": False,
            "test_passes": [index == 2, False, index == 2],
            "diagnostic": "must not persist",
        }
    if index == 0:
        return {
            "compiled": True,
            "full_pass": True,
            "test_passes": [True, True, True],
            "diagnostic": "must not persist",
        }
    return {
        "compiled": index == 2,
        "full_pass": False,
        "test_passes": [index == 2, False, False],
        "diagnostic": "must not persist",
    }


def test_deterministic_sample_is_order_invariant() -> None:
    task_ids = ["c", "a", "d", "b"]
    first = deterministic_sample(task_ids, sample_size=3, seed=42)
    second = deterministic_sample(list(reversed(task_ids)), sample_size=3, seed=42)
    assert first == second
    assert len(first) == len(set(first)) == 3


def test_preregistered_metric_decision_boundaries() -> None:
    assert (
        _metric_decision(
            0.12, {"lower": 0.06, "upper": 0.20}, target=0.10, minimum=0.05
        )
        == "GO"
    )
    assert (
        _metric_decision(
            0.12, {"lower": 0.04, "upper": 0.20}, target=0.10, minimum=0.05
        )
        == "HOLD"
    )
    assert (
        _metric_decision(
            0.09, {"lower": 0.06, "upper": 0.11}, target=0.10, minimum=0.05
        )
        == "HOLD"
    )
    assert (
        _metric_decision(
            0.02, {"lower": 0.00, "upper": 0.08}, target=0.10, minimum=0.05
        )
        == "STOP"
    )


def test_cpu_proxy_audit_resumes_and_reports_partial_signal(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, ["task-a", "task-b"], sample_size=2)
    contract = build_audit_contract(
        inputs,
        production_code={"test_double": {"sha256": "a" * 64}},
        dart_bin=str(tmp_path / "dart"),
    )
    prereg = contract["decision_preregistration"]
    assert prereg["p_new"]["target"] == 0.10
    assert prereg["p_new"]["minimum"] == 0.05
    assert prereg["r_unique"]["target"] == 0.20
    assert prereg["r_unique"]["minimum"] == 0.10
    assert prereg["r_unique"]["bootstrap_replicates"] == 10_000
    assert contract["selection"]["unique_source_sha256s"] == 2
    assert len(contract["selection"]["ordered_task_candidate_hash_seal_sha256"]) == 64
    journal = tmp_path / "audit.jsonl"
    summary_path = tmp_path / "summary.json"

    assert (
        run_audit(
            inputs,
            contract=contract,
            output_journal=journal,
            output_summary=summary_path,
            score_fn=_fake_score,
            reward_fn=verpo_execution_compile_advantages,
            workers=2,
            stop_after_new_tasks=1,
        )
        is None
    )
    assert [event["event"] for event in load_journal(journal)] == [
        "header",
        "task_terminal",
    ]

    summary = run_audit(
        inputs,
        contract=contract,
        output_journal=journal,
        output_summary=summary_path,
        score_fn=_fake_score,
        reward_fn=verpo_execution_compile_advantages,
        workers=2,
    )
    assert summary is not None
    assert [event["event"] for event in load_journal(journal)] == [
        "header",
        "task_terminal",
        "task_terminal",
        "complete",
    ]
    metrics = summary["metrics"]
    assert metrics["components"]["global"]["groups_active"] == 1
    assert metrics["components"]["compile"]["groups_active"] == 1
    assert metrics["components"]["local"]["groups_active"] == 2
    beyond = metrics["local_signal_beyond_binary"]
    assert beyond["local_active_on_compile_homogeneous_groups"] >= 1
    assert beyond["local_active_with_both_binary_components_tied"] >= 1
    assert beyond["local_noncollinear_groups"] >= 1
    prereg_stats = metrics["preregistered_statistics"]
    assert prereg_stats["p_new"]["successes"] >= 1
    assert prereg_stats["p_new"]["interval"]["confidence_level"] == 0.95
    assert prereg_stats["r_unique"]["interval"]["replicates"] == 10_000
    assert prereg_stats["r_unique"]["interval"]["seed"] == 42
    assert prereg_stats["overall_decision"] in {"GO", "STOP", "HOLD"}
    assert metrics["candidate_execution"]["candidates"] == 8
    assert metrics["candidate_execution"]["full_pass"] == 1
    terminals = [
        event for event in load_journal(journal) if event["event"] == "task_terminal"
    ]
    assert all("must not persist" not in json.dumps(value) for value in terminals)
    assert summary["summary_sha256"] == canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )

    # A completed exact resume is idempotent.
    assert (
        run_audit(
            inputs,
            contract=contract,
            output_journal=journal,
            output_summary=summary_path,
            score_fn=_fake_score,
            reward_fn=verpo_execution_compile_advantages,
            workers=1,
        )
        == summary
    )


def test_input_hash_pin_and_output_chain_fail_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, ["task-a"], sample_size=1)
    harvest_path = Path(inputs.harvest_record["path"])
    feedback_path = Path(inputs.feedback_record["path"])
    with pytest.raises(ValueError, match="journal SHA-256 differs"):
        load_audit_inputs(
            harvest_path,
            feedback_path,
            split_fn=lambda value: value.split("||"),
            expected_harvest_journal_sha256="0" * 64,
            expected_harvest_chain_head_sha256=inputs.harvest_record[
                "chain_head_sha256"
            ],
            expected_feedback_sha256=inputs.feedback_record["sha256"],
            expected_harvest_tasks=1,
            expected_feedback_tasks=1,
            expected_intersection_tasks=1,
            sample_size=1,
        )

    contract = build_audit_contract(
        inputs,
        production_code={"test_double": {"sha256": "b" * 64}},
        dart_bin=str(tmp_path / "dart"),
    )
    journal = tmp_path / "audit.jsonl"
    run_audit(
        inputs,
        contract=contract,
        output_journal=journal,
        output_summary=tmp_path / "summary.json",
        score_fn=_fake_score,
        reward_fn=verpo_execution_compile_advantages,
        workers=1,
    )
    payload = journal.read_text(encoding="utf-8")
    journal.write_text(payload.replace(AUDIT_JOURNAL_SCHEMA, "tampered", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain|chain head"):
        load_journal(journal)
