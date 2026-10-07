import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation import audit_t5gemma2_typed_api_handoff as handoff
from scripts.evaluation.durable_evaluation_journal import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_typed_local_api_handoff.sh"
CONF = ROOT / "deploy" / "vast" / "t5gemma2-typed-local-api-handoff.conf"


def _lineage() -> dict:
    return {
        "checkpoint_stage": "typed_direct",
        "checkpoint_update": 58,
        "training_state_sha256": handoff.EXPECTED_CHECKPOINT[
            "training_state_sha256"
        ],
        "adapter": {
            "run_contract_sha256": handoff.EXPECTED_CHECKPOINT[
                "run_contract_sha256"
            ],
            "adapter_weights_sha256": handoff.EXPECTED_CHECKPOINT[
                "adapter_weights_sha256"
            ],
            "adapter_config_sha256": handoff.EXPECTED_CHECKPOINT[
                "adapter_config_sha256"
            ],
        },
    }


def _write(path: Path, value: str = "sealed\n") -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def test_checkpoint_lineage_is_exact() -> None:
    assert handoff._validate_checkpoint_lineage({"checkpoint": _lineage()})[
        "checkpoint_update"
    ] == 58
    bad = _lineage()
    bad["adapter"]["adapter_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint lineage differs"):
        handoff._validate_checkpoint_lineage({"checkpoint": bad})


def test_harvest_audit_rechecks_targets_and_exact_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "harvest_report.json"
    report_path.write_text(
        json.dumps({"checkpoint": _lineage(), "outputs": {"sealed": True}}),
        encoding="utf-8",
    )
    journal = _write(tmp_path / "harvest.journal.jsonl")
    targets = _write(tmp_path / "direct_targets.jsonl")
    manifest = _write(tmp_path / "dataset_manifest.json")
    _write(tmp_path / "direct_f2.jsonl")
    _write(tmp_path / "schedule_manifest.jsonl")

    tasks = [SimpleNamespace(task_id=f"task-{index:04d}") for index in range(2775)]
    excluded = frozenset(task.task_id for task in tasks[-225:])
    scheduled = tasks[:2550]
    gates = {
        task.task_id: SimpleNamespace(tests=f"tests-{task.task_id}")
        for task in tasks
    }
    terminals = [
        {
            "task_id": task.task_id,
            "selected_target": {"code": f"void fn0() {{ /* {task.task_id} */ }}"},
        }
        if position < 2
        else {"task_id": task.task_id, "selected_target": None}
        for position, task in enumerate(scheduled)
    ]
    source_record = {
        "sha256": "1" * 64,
        "chain_head_sha256": "2" * 64,
        "event_count": 2552,
        "head_event_sha256": "3" * 64,
        "run_contract_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        handoff.local,
        "load_completed_harvest_artifacts",
        lambda **_kwargs: (
            tasks,
            gates,
            scheduled,
            terminals,
            {"sealed": True},
            source_record,
        ),
    )
    monkeypatch.setattr(
        handoff.cascade,
        "load_existing_225_exclusions",
        lambda *_args: (
            excluded,
            {"schema": "sealed-225", "rows": 225, "sha256": "5" * 64},
        ),
    )
    monkeypatch.setattr(
        handoff,
        "_validate_harvest_contract",
        lambda _path: {
            "script_sha256": handoff.EXPECTED_LOCAL_HARVEST_SCRIPT_SHA256,
            "samples_per_task": 4,
        },
    )
    observed: list[tuple[str, str, int]] = []

    def verify(code: str, tests_text: str, slot: str, timeout: int) -> bool:
        observed.append((code, tests_text, timeout))
        assert slot.startswith("typed-handoff-audit-task-")
        return True

    args = argparse.Namespace(
        local_harvest_report=str(report_path),
        expected_local_harvest_report_sha256=sha256_file(report_path),
        local_harvest_journal=str(journal),
        expected_local_harvest_journal_sha256=sha256_file(journal),
        local_harvest_targets=str(targets),
        expected_local_harvest_targets_sha256=sha256_file(targets),
        existing_direct_manifest=str(manifest),
        expected_existing_direct_manifest_sha256=sha256_file(manifest),
        gold_train_jsonl="gold.jsonl",
        expected_gold_train_sha256="a" * 64,
        gold_f2_jsonl="gold-f2.jsonl",
        expected_gold_f2_sha256="b" * 64,
        heldout_jsonl="heldout.jsonl",
        expected_heldout_sha256="c" * 64,
        timeout=30,
        evaluation_workers=1,
        output=str(tmp_path / "audit.json"),
    )
    audit = handoff.audit_harvest(args, verify=verify)
    assert audit["status"] == "pass"
    assert audit["schedule"]["scheduled_tasks"] == 2550
    assert audit["schedule"]["excluded_previous_direct_tasks"] == 225
    assert audit["accepted"]["direct_targets"] == 2
    assert audit["accepted"]["independently_reverified"] == 2
    assert len(observed) == 2


def test_harvest_audit_rejects_wrong_predecessor_complement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The full loader may be internally consistent, but the independent 225
    # manifest must identify exactly the complement of the 2,550 schedule.
    tasks = [SimpleNamespace(task_id=f"task-{index}") for index in range(2775)]
    scheduled = tasks[:2550]
    monkeypatch.setattr(
        handoff.local,
        "load_completed_harvest_artifacts",
        lambda **_kwargs: (tasks, {}, scheduled, [{}] * 2550, {}, {}),
    )
    monkeypatch.setattr(
        handoff.cascade,
        "load_existing_225_exclusions",
        lambda *_args: (frozenset(task.task_id for task in tasks[:225]), {}),
    )
    args = argparse.Namespace(
        local_harvest_report="unused",
        expected_local_harvest_report_sha256="a" * 64,
        local_harvest_journal="unused",
        expected_local_harvest_journal_sha256="b" * 64,
        local_harvest_targets="unused",
        expected_local_harvest_targets_sha256="c" * 64,
        existing_direct_manifest="unused",
        expected_existing_direct_manifest_sha256="d" * 64,
        gold_train_jsonl="unused",
        expected_gold_train_sha256="e" * 64,
        gold_f2_jsonl="unused",
        expected_gold_f2_sha256="f" * 64,
        heldout_jsonl="unused",
        expected_heldout_sha256="0" * 64,
        timeout=30,
        evaluation_workers=1,
        output=str(tmp_path / "unused.json"),
    )
    with pytest.raises(ValueError, match="schedule/exclusion differs"):
        handoff.audit_harvest(args, verify=lambda *_args: True)


def test_projection_audit_seals_exact_first_kimi_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = [SimpleNamespace(task_id=f"task-{index:04d}") for index in range(2550)]
    context = SimpleNamespace(
        scheduled_tasks=tasks,
        gates={task.task_id: object() for task in tasks},
        source_journal_record={"sha256": "1" * 64},
    )
    terminals = [{"task_id": task.task_id} for task in tasks]
    selected = [(index, task, terminals[index]) for index, task in enumerate(tasks[:75])]
    monkeypatch.setattr(
        handoff.cascade, "load_typed_source_context", lambda _args: context
    )
    monkeypatch.setattr(
        handoff.cascade,
        "load_visible_projection",
        lambda _args, context: (terminals, {"tasks": 2550, "report_sha256": "2" * 64}),
    )
    monkeypatch.setattr(
        handoff.cascade,
        "load_existing_225_exclusions",
        lambda *_args: (frozenset(), {"rows": 225, "sha256": "3" * 64}),
    )
    monkeypatch.setattr(
        handoff.cascade,
        "select_visible_zero_tasks",
        lambda **_kwargs: selected,
    )
    monkeypatch.setattr(
        handoff.cascade,
        "build_visible_only_plans",
        lambda **kwargs: (
            [SimpleNamespace(task=row[1]) for row in kwargs["selected"]],
            {"complete_private_suite_used_for_diagnostic": False},
        ),
    )
    args = argparse.Namespace(
        existing_direct_manifest="unused",
        expected_existing_direct_manifest_sha256="a" * 64,
        output=str(tmp_path / "projection-audit.json"),
    )
    audit = handoff.audit_projection(args)
    expected_ids = [task.task_id for task in tasks[:50]]
    assert audit["status"] == "pass"
    assert audit["first_cohort"]["tasks"] == 50
    assert audit["first_cohort"]["task_ids_sha256"] == canonical_sha256(expected_ids)
    assert audit["privacy"]["frontier_api_calls"] is False
    assert audit["privacy"]["heldout_175_opened_for_exclusion_audit"] is True


def test_supervisor_handoff_is_ordered_and_fails_closed() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    harvest_audit = text.index("audit_t5gemma2_typed_api_handoff.py harvest")
    prepare = text.index('! "${PREPARE_LAUNCHER}"')
    projection_audit = text.index("audit_t5gemma2_typed_api_handoff.py projection")
    controller = text.index('exec "${DUAL_API_LAUNCHER}"')
    assert harvest_audit < prepare < projection_audit < controller
    assert 'case "${state}"' in text
    assert "RUNNING|STARTING" in text
    assert "EXITED)" in text
    assert 'STOPPED)\n      blocked "harvest was stopped rather than completed"' in text
    assert "snapshot_one" in text and "snapshot_two" in text
    assert "3714c845574bf3eae8250d79078c34ac009a8f07460ddf767ee0fa6d5f0add33" in text
    assert "a2c2c3adaa96467a8de1025697222c11146ef94fe63c0f85e9cfbf8beebd753c" in text
    assert "c69e845cfefcd91555171813a66492dba0b2b5c9d44bbd8efd21175f5f7f2e14" in text
    assert "83fb363aa04f9f8993d44d8b085707897699f0eeebdd0c4539b279184b8b2796" in text
    assert handoff.EXPECTED_LOCAL_HARVEST_SCRIPT_SHA256 in text
    assert "independently_reverified == .accepted.direct_targets" in text
    assert "T5GEMMA_TYPED_DUAL_API_OUTPUT_ROOT" in text
    assert "projection_complete" in text
    assert "OPENROUTER_API_KEY" not in text
    assert "sk-" not in text


def test_handoff_supervisor_is_manual_and_exit78_is_expected() -> None:
    text = CONF.read_text(encoding="utf-8")
    assert "[program:t5gemma2-typed-local-api-handoff]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
