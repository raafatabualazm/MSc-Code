from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.evaluation import audit_t5gemma2_typed_pass2_update58_rerun as audit
from scripts.evaluation.durable_evaluation_journal import sha256_file


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = (
    ROOT
    / "scripts"
    / "evaluation"
    / "audit_t5gemma2_typed_pass2_update58_rerun.py"
)
RUNNER = (
    ROOT / "deploy" / "vast" / "t5gemma2_typed_pass2_update58_matched_eval.sh"
)
HANDOFF = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2_typed_pass2_update58_matched_eval_handoff.sh"
)
SUPERVISOR = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-typed-pass2-update58-matched-eval-handoff.conf"
)


def _update58_contract() -> dict:
    return {
        "schema": audit.PASS1_RUN_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "optimization": {
            "epochs": 2,
            "planned_updates": 58,
            "gradient_accumulation": 8,
            "learning_rate": 0.00002,
            "warmup_updates": 0,
            "seed": 42,
        },
        "dataset": {
            "schema": audit.PASS1_DATASET_SCHEMA,
            "rows": 225,
            "heldout_overlap": 0,
            "known_contaminant_excluded": audit.EXPECTED_CONTAMINANT,
            "model_visible_fields": ["opaque_typed_contract", "F2.text"],
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "repair_conditioned_prefixes_visible": False,
            "composition": {
                "verified_direct": 225,
                "local_student_direct": 141,
                "external_teacher_direct": 84,
                "repair_conditioned": 0,
                "gold_replay": 0,
            },
            "full_acceptance_reverification": {
                "rows": 225,
                "passed": 225,
                "tests_model_visible": False,
                "diagnostics_persisted": False,
            },
        },
        "privacy": {
            "heldout_overlap": 0,
            "heldout_content_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
        },
    }


def test_update58_checkpoint_gate_rejects_replay_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _update58_contract()
    result = {
        "schema": audit.PASS1_RUN_SCHEMA,
        "status": "complete",
        "updates": 58,
        "planned_updates": 58,
        "rows": 225,
        "latest_checkpoint": "checkpoint-optstep-000058",
    }
    record = {"contract": contract, "result": result}
    monkeypatch.setattr(
        audit.matched, "_checkpoint_paths_record", lambda *_args: record
    )
    assert audit._validate_update58_checkpoint(Path("contract"), Path("result")) is record  # noqa: SLF001
    tampered = copy.deepcopy(contract)
    tampered["dataset"]["composition"]["gold_replay"] = 1
    record["contract"] = tampered
    with pytest.raises(ValueError, match="update58 checkpoint/result"):
        audit._validate_update58_checkpoint(Path("contract"), Path("result"))  # noqa: SLF001


def test_pass2_training_audit_is_hash_and_checkpoint_bound(tmp_path: Path) -> None:
    checkpoint = {
        "canonical_sha256": "1" * 64,
        "file_sha256": "2" * 64,
        "result_sha256": "3" * 64,
    }
    value = {
        "schema": audit.PASS2_TRAINING_AUDIT_SCHEMA,
        "status": "pass",
        "contract": {
            "rows": 209,
            "local_rows": 190,
            "api_rows": 19,
            "planned_updates": 54,
            "warmstart_update": 58,
        },
        "composition": {
            "rows": 209,
            "local_student_new": 190,
            "external_teacher_new": 19,
            "prior_225_replay": 0,
            "gold_replay": 0,
            "heldout_overlap": 0,
        },
        "checkpoint": {
            "name": "checkpoint-optstep-000054",
            "update": 54,
            "run_contract_canonical_sha256": "1" * 64,
        },
        "artifacts": {
            "checkpoint_contract_sha256": "2" * 64,
            "result_sha256": "3" * 64,
        },
    }
    path = tmp_path / "training_audit.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert audit._validate_training_audit(path, sha256_file(path), checkpoint) == value  # noqa: SLF001
    with pytest.raises(ValueError, match="SHA differs"):
        audit._validate_training_audit(path, "f" * 64, checkpoint)  # noqa: SLF001


def test_handoff_waits_for_exact_exit_and_audits_before_gpu_runner() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "t5gemma2-typed-pass2-eval-handoff" in text
    assert "RUNNING|STARTING" in text and "EXITED) break" in text
    assert 'STOPPED) blocked' in text and 'FATAL|BACKOFF|UNKNOWN|"") blocked' in text
    assert 'pass2_snapshot_one="$(sha256sum "${pass2_files[@]}")"' in text
    assert 'pass2_snapshot_two="$(sha256sum "${pass2_files[@]}")"' in text
    assert "00d8d92dff815479bbab8357f899ef76cf416e71e1a9dd9b9b5680edaeb659a4" in text
    assert text.index('"${AUDITOR}" pass2-arm') < text.index('exec "${RUNNER}"')
    for fragment in (
        '"${PREDICTIONS}.provenance.json"',
        '"${PREDICTIONS}.generation.journal.jsonl.chain-head.json"',
        "typed_direct_pass2_seed42_k10_score_full175.json",
        '"${FULL_SCORE}.evaluation.journal.jsonl.chain-head.json"',
        "typed_direct_pass2_seed42_k10_score_clean174.json",
        ".checks.full175_complete == true",
        ".checks.clean174_complete == true",
    ):
        assert fragment in text


def test_runner_reruns_update58_under_exact_current_stack_then_compares() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    inference = text.index('/venv/main/bin/python "${WRAPPER}"')
    scoring = text.index('/venv/main/bin/python "${SCORER}"')
    clean = text.index('/venv/main/bin/python "${DERIVE_CLEAN}"')
    compare = text.index('"${AUDITOR}" compare')
    assert inference < scoring < clean < compare
    for fragment in (
        "checkpoint-optstep-000058",
        "--input_view typed_opaque_contract",
        "--num_samples 10",
        "--generation_batch_size 10",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--temperature 0.8",
        "--top_p 0.95",
        "--seed 42",
        "--k 10 --workers 32 --timeout 30 --stability_runs 2",
        "--exclude_task_id sigless_8bf7f40ca356",
        ".historical_update58_predictions_reused == false",
        ".checks.same_wrapper_and_base_inference_code == true",
        "62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec",
    ):
        assert fragment in text
    assert "typed_direct_rs_sft_seed42_k10_predictions.json" not in text


def test_auditor_requires_same_wrapper_and_base_inference_hashes() -> None:
    text = AUDITOR.read_text(encoding="utf-8")
    assert (
        'bg["header_contract"]["script_sha256"]\n'
        '        == ag["header_contract"]["script_sha256"]'
    ) in text
    assert (
        'bg["header_contract"]["base_inference_script_sha256"]\n'
        '        == ag["header_contract"]["base_inference_script_sha256"]'
    ) in text
    assert '"historical_update58_predictions_reused": False' in text
    assert "previous-update58-audit" not in text


def test_paired_metric_reports_gains_losses_and_exact_test() -> None:
    order = ["a", "b", "c", "d"]
    before = {
        "a": {"pass_at_k": False},
        "b": {"pass_at_k": True},
        "c": {"pass_at_k": False},
        "d": {"pass_at_k": True},
    }
    after = {
        "a": {"pass_at_k": True},
        "b": {"pass_at_k": False},
        "c": {"pass_at_k": True},
        "d": {"pass_at_k": True},
    }
    result = audit._paired_metric(order, before, after, "pass_at_k")  # noqa: SLF001
    assert result["gains"]["task_ids"] == ["a", "c"]
    assert result["losses"]["task_ids"] == ["b"]
    assert result["discordant_tasks"] == 3
    assert "exact_two_sided_sign_mcnemar_p" in result


def test_supervisor_job_is_new_manual_fail_closed_chain() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")
    assert "[program:t5gemma2-typed-pass2-update58-matched-eval-handoff]" in text
    assert "t5gemma2_typed_pass2_update58_matched_eval_handoff.sh" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text and "killasgroup=true" in text
