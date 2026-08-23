from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from scripts.evaluation import audit_t5gemma2_typed_holdback_alignment as audit
from scripts.evaluation import audit_t5gemma2_typed_proxy_reward_surface as proxy
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    load_journal,
    sha256_file,
)
from scripts.preprocessing import build_verpo_feedback_view as feedback_builder


def _groups() -> tuple[audit.AlignmentGroup, ...]:
    groups: list[audit.AlignmentGroup] = []
    for position in range(audit.EXPECTED_SELECTED_GROUPS):
        codes = tuple(f"private-code-{position:03d}-{index}" for index in range(4))
        groups.append(
            audit.AlignmentGroup(
                task_id=f"task-{position:03d}",
                task_position=position,
                source_sha256=f"{position + 1:064x}",
                typed_contract_sha256=f"{position + 1000:064x}",
                candidate_codes=codes,
                candidate_code_sha256s=tuple(
                    audit._sha256_text(value) for value in codes  # noqa: SLF001
                ),
                visible_local_rewards=(0.0, 1.0, 2.0, 3.0),
                visible_unified_advantages=(0.0, 1.0, 2.0, 3.0),
                p_new=position < audit.EXPECTED_P_NEW_GROUPS,
                holdback_tests="PRIVATE TEST TEXT MUST NOT PERSIST",
                holdback_cases=3,
            )
        )
    return tuple(groups)


def _inputs() -> audit.AlignmentInputs:
    source = proxy.AuditInputs(
        groups=(),
        harvest_record={},
        feedback_record={},
        harvest_contract={},
        intersection_tasks=0,
        intersection_task_ids_sha256="0" * 64,
        sample_seed=42,
    )
    return audit.AlignmentInputs(
        groups=_groups(),
        source_inputs=source,
        proxy_contract={},
        proxy_journal_record={},
        proxy_summary_record={},
        holdback_record={},
        feedback_build_record={},
    )


def _contract() -> dict:
    return {
        "schema": audit.CONTRACT_SCHEMA,
        "one_shot_policy": {
            "private_holdback_consumed_for_objective_selection": True,
            "future_objective_selection_on_this_holdback_forbidden": True,
            "future_reward_weight_tuning_on_this_holdback_forbidden": True,
            "journal_must_remain_private": True,
        },
        "reward_weights_frozen": {
            "verpo_alpha": 2.0,
            "local_weight": 1.0,
            "compile_weight": 0.25,
            "fixed_before_private_holdback_read": True,
            "frozen_after_private_holdback_read": True,
            "future_weight_tuning_from_holdback_forbidden": True,
        },
    }


def _fake_score(code: str, tests: str, _slot: str, **_kwargs):
    assert tests == "PRIVATE TEST TEXT MUST NOT PERSIST"
    index = int(code.rsplit("-", 1)[1])
    passes = [candidate < index for candidate in range(3)]
    return {
        "compiled": True,
        "full_pass": all(passes),
        "test_passes": passes,
        "diagnostic": "PRIVATE DIAGNOSTIC MUST NOT PERSIST",
    }


def test_tie_averaged_argmax_and_pairwise_ties() -> None:
    selected = audit._argmax_tie_average(  # noqa: SLF001
        [0.0, 2.0, 2.0, 1.0], [0.0, 0.5, 1.0, 0.0]
    )
    assert selected == 0.75
    correct, pairs = audit._pairwise_contribution(  # noqa: SLF001
        [0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 0.5, 1.0]
    )
    assert pairs == 6
    assert correct == 5.5
    visible_tie_correct, visible_tie_pairs = audit._pairwise_contribution(  # noqa: SLF001
        [0.0, 1.0, 1.0, 3.0], [0.0, 0.25, 0.75, 1.0]
    )
    assert visible_tie_pairs == 6
    assert visible_tie_correct == 5.5


def test_preregistered_decision_boundaries_are_strict() -> None:
    assert audit._uplift_decision(0.02, {"lower": 0.0001, "upper": 0.1}) == "GO"  # noqa: SLF001
    assert audit._uplift_decision(0.02, {"lower": 0.0, "upper": 0.1}) == "HOLD"  # noqa: SLF001
    assert audit._uplift_decision(0.0, {"lower": -0.1, "upper": 0.0}) == "STOP"  # noqa: SLF001
    assert audit._rank_decision(0.55, {"lower": 0.5001, "upper": 0.7}) == "GO"  # noqa: SLF001
    assert audit._rank_decision(0.55, {"lower": 0.5, "upper": 0.7}) == "HOLD"  # noqa: SLF001
    assert audit._rank_decision(0.4, {"lower": 0.2, "upper": 0.5}) == "STOP"  # noqa: SLF001


def test_private_audit_resumes_and_publishes_only_aggregates(tmp_path: Path) -> None:
    inputs = _inputs()
    contract = _contract()
    journal = tmp_path / "private.journal.jsonl"
    summary_path = tmp_path / "summary.json"

    assert (
        audit.run_audit(
            inputs,
            contract=contract,
            output_journal=journal,
            output_summary=summary_path,
            score_fn=_fake_score,
            workers=2,
            stop_after_new_tasks=1,
            bootstrap_replicates=100,
        )
        is None
    )
    assert [row["event"] for row in load_journal(journal)] == [
        "header",
        "task_terminal",
    ]

    summary = audit.run_audit(
        inputs,
        contract=contract,
        output_journal=journal,
        output_summary=summary_path,
        score_fn=_fake_score,
        workers=4,
        bootstrap_replicates=100,
    )
    assert summary is not None
    assert summary["decision"] == "GO"
    assert summary["metrics"]["preregistered_61_p_new"]["groups"] == 61
    assert summary["metrics"]["descriptive_all_150"]["candidates"] == 600
    assert summary["privacy"]["aggregate_only"] is True
    assert summary["one_shot_policy"][
        "future_objective_selection_on_this_holdback_forbidden"
    ] is True
    assert summary["reward_weights_frozen"][
        "future_weight_tuning_from_holdback_forbidden"
    ] is True
    assert summary["reward_weights_frozen"][
        "frozen_after_private_holdback_read"
    ] is True
    assert summary["summary_sha256"] == canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )

    private_text = journal.read_text(encoding="utf-8")
    public_text = summary_path.read_text(encoding="utf-8")
    for forbidden in (
        "PRIVATE TEST TEXT MUST NOT PERSIST",
        "PRIVATE DIAGNOSTIC MUST NOT PERSIST",
        "private-code-000-0",
    ):
        assert forbidden not in private_text
        assert forbidden not in public_text
    assert "task-000" not in public_text
    assert len(load_journal(journal)) == audit.EXPECTED_SELECTED_GROUPS + 2
    if os.name != "nt":
        assert stat.S_IMODE(journal.stat().st_mode) == 0o600
        assert stat.S_IMODE(
            Path(str(journal) + ".chain-head.json").stat().st_mode
        ) == 0o600

    # An exact completed resume is idempotent.
    assert (
        audit.run_audit(
            inputs,
            contract=contract,
            output_journal=journal,
            output_summary=summary_path,
            score_fn=_fake_score,
            workers=1,
            bootstrap_replicates=100,
        )
        == summary
    )


def test_private_journal_chain_fails_closed(tmp_path: Path) -> None:
    inputs = _inputs()
    journal = tmp_path / "private.journal.jsonl"
    audit.run_audit(
        inputs,
        contract=_contract(),
        output_journal=journal,
        output_summary=tmp_path / "summary.json",
        score_fn=_fake_score,
        workers=4,
        bootstrap_replicates=20,
    )
    payload = journal.read_text(encoding="utf-8")
    journal.write_text(payload.replace('"p_new":true', '"p_new":false', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain|chain head"):
        load_journal(journal)


def test_launcher_pins_private_inputs_and_cpu_only_runtime() -> None:
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts/evaluation/audit_t5gemma2_typed_holdback_alignment.py"
    launcher = project / "deploy/vast/t5gemma2_typed_holdback_alignment_audit_v1.sh"
    text = launcher.read_text(encoding="utf-8")
    assert sha256_file(script) in text
    for digest in (
        audit.EXPECTED_PROXY_JOURNAL_SHA256,
        audit.EXPECTED_PROXY_CHAIN_HEAD_SHA256,
        audit.EXPECTED_PROXY_SUMMARY_SHA256,
        audit.EXPECTED_HOLDBACK_SHA256,
        audit.EXPECTED_FEEDBACK_BUILD_SHA256,
        proxy.EXPECTED_HARVEST_JOURNAL_SHA256,
        proxy.EXPECTED_HARVEST_CHAIN_HEAD_SHA256,
        proxy.EXPECTED_FEEDBACK_SHA256,
    ):
        assert digest in text
    assert "export CUDA_VISIBLE_DEVICES=-1" in text
    assert "umask 077" in text
    assert "--holdback_jsonl" in text
    assert "--output_journal" in text
    assert "--output_summary" in text
    supervisor = (
        project
        / "deploy/vast/t5gemma2-typed-holdback-alignment-audit-v1.conf"
    ).read_text(encoding="utf-8")
    assert "autostart=false" in supervisor
    assert 'CUDA_VISIBLE_DEVICES="-1"' in supervisor
    assert "t5gemma2_typed_holdback_alignment_audit_v1.sh" in supervisor


def _private_split_fixture() -> tuple[dict, dict]:
    tests = """void main() {
  expect(fn0(0), 0);
  expect(fn0(1), 1);
  expect(fn0(2), 2);
  expect(fn0(3), 3);
}
void expect(dynamic a, dynamic b) { if (a != b) throw 'bad'; }
"""
    split = feedback_builder.split_train_harness(
        task_id="task-private", tests=tests, seed=42
    )
    private = {
        "task_id": "task-private",
        "schema": audit.PRIVATE_SPLIT_SCHEMA,
        **split,
    }
    binding = feedback_builder.stable_sha256(
        {
            key: split[key]
            for key in (
                "tests_sha256",
                "case_count",
                "visible_count",
                "holdback_count",
                "visible_case_indices",
                "holdback_case_indices",
            )
        }
    )
    visible = {
        "task_id": "task-private",
        "feedback_tests": split["feedback_tests"],
        "verpo_feedback_split_schema": audit.PRIVATE_SPLIT_SCHEMA,
        "verpo_feedback_split_binding_sha256": binding,
    }
    return visible, private


def test_private_split_exact_keys_indices_binding_and_reconstruction() -> None:
    visible, private = _private_split_fixture()
    split_fn = lambda value: [object() for _ in feedback_builder.extract_expect_spans(value)]
    task_id, _tests, holdback_count, visible_count = audit._validate_private_row(  # noqa: SLF001
        visible, private, position=0, split_fn=split_fn
    )
    assert task_id == "task-private"
    assert visible_count == holdback_count == 2

    with_extra = {**private, "unexpected": False}
    with pytest.raises(ValueError, match="row 0 differs"):
        audit._validate_private_row(  # noqa: SLF001
            visible, with_extra, position=0, split_fn=split_fn
        )

    unsorted = {**private, "visible_case_indices": list(reversed(private["visible_case_indices"]))}
    with pytest.raises(ValueError, match="row 0 differs"):
        audit._validate_private_row(  # noqa: SLF001
            visible, unsorted, position=0, split_fn=split_fn
        )

    broken_binding = {**visible, "verpo_feedback_split_binding_sha256": "0" * 64}
    with pytest.raises(ValueError, match="split binding"):
        audit._validate_private_row(  # noqa: SLF001
            broken_binding, private, position=0, split_fn=split_fn
        )

    broken_holdback = {
        **private,
        "reward_holdback_tests": private["reward_holdback_tests"].replace(
            "fn0(", "gn0(", 1
        ),
    }
    with pytest.raises(ValueError, match="masks conflict|digest differs|reconstruction differs"):
        audit._validate_private_row(  # noqa: SLF001
            visible, broken_holdback, position=0, split_fn=split_fn
        )


def test_build_report_requires_exact_accounting_invariants_and_outputs(
    tmp_path: Path,
) -> None:
    records = {}
    for key, basename in audit.EXPECTED_BUILD_OUTPUT_NAMES.items():
        path = tmp_path / basename
        path.write_text(f"{key}\n", encoding="utf-8")
        records[key] = audit._actual_file_record(path)  # noqa: SLF001
    report_path = tmp_path / "verpo_feedback_view.build.json"
    split_policy = {
        "schema": audit.PRIVATE_SPLIT_SCHEMA,
        "seed": 42,
        "source_field": "tests",
        "acceptance_tests_used": False,
        "minimum_cases": 2,
        "even_policy": "N/2 visible; N/2 holdback",
        "odd_policy": "floor(N/2) visible; ceil(N/2) holdback",
        "single_case_policy": "exclude",
        "no_expect_policy": "exclude",
        "malformed_policy": "exclude",
        "selection": "lowest task-bound SHA-256 ranks become visible",
    }
    report = {
        "schema": "verpo-train-feedback-view-v1",
        "status": "complete",
        "split_policy": split_policy,
        "accounting": dict(audit.EXPECTED_BUILD_ACCOUNTING),
        "invariants": dict(audit.EXPECTED_BUILD_INVARIANTS),
        "predeclared_expectation": {
            "accounting": dict(audit.EXPECTED_BUILD_ACCOUNTING),
            "eligible_task_ids_sha256": audit.EXPECTED_ELIGIBLE_TASK_IDS_SHA256,
            "excluded_task_ids_sha256": audit.EXPECTED_EXCLUDED_TASK_IDS_SHA256,
        },
        "digests": {
            "eligible_task_ids_sha256": audit.EXPECTED_ELIGIBLE_TASK_IDS_SHA256,
            "excluded_task_ids_sha256": audit.EXPECTED_EXCLUDED_TASK_IDS_SHA256,
            "script_sha256": sha256_file(Path(feedback_builder.__file__).resolve()),
        },
        "outputs": records,
    }
    audit._validate_feedback_build_report(  # noqa: SLF001
        report,
        feedback_build_path=report_path,
        feedback_path=tmp_path / audit.EXPECTED_BUILD_OUTPUT_NAMES["rollout"],
        holdback_path=tmp_path
        / audit.EXPECTED_BUILD_OUTPUT_NAMES["reward_holdback_private"],
    )
    bad = {**report, "invariants": {**report["invariants"], "extra": True}}
    with pytest.raises(ValueError, match="accounting/invariants"):
        audit._validate_feedback_build_report(  # noqa: SLF001
            bad,
            feedback_build_path=report_path,
            feedback_path=tmp_path / audit.EXPECTED_BUILD_OUTPUT_NAMES["rollout"],
            holdback_path=tmp_path
            / audit.EXPECTED_BUILD_OUTPUT_NAMES["reward_holdback_private"],
        )
