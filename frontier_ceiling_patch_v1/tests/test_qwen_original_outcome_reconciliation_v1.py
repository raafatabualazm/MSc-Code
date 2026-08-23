from __future__ import annotations

import sys
from pathlib import Path

import pytest


PATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH))

import frontier_passk as runner
import qwen37_original_outcome_reconciliation_v1 as reconciliation


def test_contract_pins_exact_six_unique_original_terminals() -> None:
    contract = reconciliation.load_contract(PATCH)
    rows = contract["expected_orphans"]
    assert len(rows) == 6
    assert len({row["response_id"] for row in rows}) == 6
    assert {
        (row["arm"], row["task_id"], row["global_sample_index"])
        for row in rows
    } == {
        ("opus", "sigless_9dcc77211b03", 0),
        ("opus", "sigless_9dcc77211b03", 1),
        ("codex", "sigless_f2b582d1df68", 3),
        ("opus", "sigless_9dcc77211b03", 5),
        ("codex", "sigless_ef71ae50a5fb", 7),
        ("codex", "sigless_ef71ae50a5fb", 8),
    }


def test_reconciliation_source_has_no_provider_client_or_call() -> None:
    reconciliation.assert_provider_free_source()


def test_provider_import_is_rejected_statically(tmp_path: Path) -> None:
    script = tmp_path / "unsafe.py"
    script.write_text("from openai import OpenAI\n", encoding="utf-8")
    with pytest.raises(
        reconciliation.ReconciliationError,
        match="provider client import",
    ):
        reconciliation.assert_provider_free_source(script)


def test_selection_id_ignores_outcomes_but_binds_terminal() -> None:
    row = {
        "arm": "opus",
        "source_shard_key": "base_0517_k3",
        "task_id": "task",
        "local_sample_index": 0,
        "global_sample_index": 0,
        "attempt_id": "attempt",
        "response_id": "response",
        "terminal_row_sha256": "a" * 64,
    }
    baseline = reconciliation.selection_id(row)
    row["passed"] = True
    row["compiled"] = True
    assert reconciliation.selection_id(row) == baseline
    row["terminal_row_sha256"] = "b" * 64
    assert reconciliation.selection_id(row) != baseline


def _selection(*, candidate_valid: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": reconciliation.SCHEMA,
        "record_type": "outcome_blind_selection",
        "contract_sha256": reconciliation.EXPECTED_CONTRACT_SHA256,
        "arm": "opus",
        "source_shard_key": "base_0517_k3",
        "source_directory": "source",
        "task_id": "task",
        "local_sample_index": 0,
        "global_sample_index": 0,
        "attempt_id": "attempt",
        "response_id": "response",
        "finish_reason": "stop",
        "candidate_valid": candidate_valid,
        "terminal_reason": (
            "candidate_valid" if candidate_valid else "invalid"
        ),
        "code_sha256": "c" * 64,
        "terminal_row_sha256": "d" * 64,
        "selection_uses_pass_or_compile": False,
    }
    row["selection_id"] = reconciliation.selection_id(row)
    return row


def test_invalid_candidate_reconciliation_is_fail_closed() -> None:
    selection = _selection(candidate_valid=False)
    outcome = {
        "schema": reconciliation.SCHEMA,
        "record_type": "reconciled_candidate_outcome",
        "contract_sha256": reconciliation.EXPECTED_CONTRACT_SHA256,
        **{
            key: selection[key]
            for key in (
                "selection_id",
                "arm",
                "task_id",
                "local_sample_index",
                "global_sample_index",
                "attempt_id",
                "response_id",
                "finish_reason",
                "candidate_valid",
                "terminal_reason",
                "code_sha256",
            )
        },
        "evaluator_sha256": reconciliation.EXPECTED_EVALUATOR_SHA256,
        "evaluation_performed": False,
        "completion_attestation_id": runner.REQUIRED_ATTESTATION_ID,
        "completion_attestation_enforced": False,
        "completion_attestation_satisfied_all_runs": False,
        "compiled": False,
        "passed": False,
        "stability_runs": [],
    }
    reconciliation.validate_reconciled_outcome(outcome, selection)
    outcome["passed"] = True
    with pytest.raises(reconciliation.ReconciliationError):
        reconciliation.validate_reconciled_outcome(outcome, selection)


def test_valid_candidate_requires_two_stable_attested_runs() -> None:
    selection = _selection(candidate_valid=True)
    run = {
        "compiled": True,
        "passed": False,
        "completion_attestation_id": runner.REQUIRED_ATTESTATION_ID,
        "completion_attestation_required": True,
        "completion_attestation_satisfied": False,
    }
    outcome = {
        "schema": reconciliation.SCHEMA,
        "record_type": "reconciled_candidate_outcome",
        "contract_sha256": reconciliation.EXPECTED_CONTRACT_SHA256,
        **{
            key: selection[key]
            for key in (
                "selection_id",
                "arm",
                "task_id",
                "local_sample_index",
                "global_sample_index",
                "attempt_id",
                "response_id",
                "finish_reason",
                "candidate_valid",
                "terminal_reason",
                "code_sha256",
            )
        },
        "evaluator_sha256": reconciliation.EXPECTED_EVALUATOR_SHA256,
        "evaluation_performed": True,
        "completion_attestation_id": runner.REQUIRED_ATTESTATION_ID,
        "completion_attestation_enforced": True,
        "completion_attestation_satisfied_all_runs": False,
        "compiled": True,
        "passed": False,
        "stability_runs": [dict(run), dict(run)],
    }
    reconciliation.validate_reconciled_outcome(outcome, selection)
    outcome["stability_runs"].pop()
    with pytest.raises(reconciliation.ReconciliationError):
        reconciliation.validate_reconciled_outcome(outcome, selection)
