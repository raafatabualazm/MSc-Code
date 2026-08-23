from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation.audit_t5gemma2_typed_rs_sft import (
    BASELINE_CHECKPOINT_SCHEMA,
    BASELINE_SEAL_SCHEMA,
    CLEAN_SCORE_SCHEMA,
    EXPECTED_COMPLETION_ATTESTATION,
    EXPECTED_EXCLUDED_TASK_IDS,
    EXPECTED_INPUT_VIEW,
    EXPECTED_RETRY_POLICY,
    EXPECTED_SAMPLING,
    GENERATION_JOURNAL_SCHEMA,
    INFERENCE_SCHEMA,
    PROVENANCE_SCHEMA,
    SCORE_JOURNAL_SCHEMA,
    SCORE_SCHEMA,
    UPDATE_CHECKPOINT_SCHEMA,
    UPDATE_DATASET_SCHEMA,
    _exact_two_sided_sign_mcnemar,
    _score_jobs,
    audit,
)
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    atomic_write_json,
    canonical_sha256,
    journal_record,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluation" / "graph_compile_at_k_antigravity.py"
HANDOFF = ROOT / "deploy" / "vast" / "t5gemma2_typed_rs_sft_eval_audit_handoff.sh"
CONF = ROOT / "deploy" / "vast" / "t5gemma2-typed-rs-sft-eval-audit-handoff.conf"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_evaluation(path: Path, task_ids: list[str]) -> None:
    path.write_text(
        "".join(
            f'{{"acceptance_tests":"void main() {{}}","task_id":"{task_id}"}}\n'
            for task_id in task_ids
        ),
        encoding="utf-8",
    )


def _checkpoint_contracts(tmp_path: Path) -> dict[str, Path]:
    shared_base = {
        "name": "google/t5gemma-2-4b-4b",
        "resolved_commit": "4" * 40,
        "is_encoder_decoder": True,
        "config_sha256": "1" * 64,
    }
    baseline_contract: dict[str, Any] = {
        "schema": BASELINE_CHECKPOINT_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "model": "google/t5gemma-2-4b-4b",
        "model_revision": "4" * 40,
        "base_model": shared_base,
        "optimization": {"epochs": 2, "planned_updates": 348, "seed": 42},
        "dataset": {
            "schema": BASELINE_CHECKPOINT_SCHEMA,
            "rows": 2775,
            "input_rows": 2776,
            "model_visible_fields": ["opaque_typed_contract", "F2.text"],
            "heldout": {
                "rows": 175,
                "model_visible": False,
                "task_id_overlap": 0,
                "exact_gold_source_overlap": 0,
                "exact_acceptance_test_overlap": 0,
            },
            "training_exclusions": {
                "count": 1,
                "task_ids": ["sigless_6b1dd0c6b6fc"],
            },
        },
        "lora": {"targets": ["encoder.q_proj", "decoder.q_proj"]},
    }
    baseline_digest = canonical_sha256(baseline_contract)
    update_contract: dict[str, Any] = {
        "schema": UPDATE_CHECKPOINT_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "model": baseline_contract["model"],
        "model_revision": baseline_contract["model_revision"],
        "base_model": shared_base,
        "optimization": {
            "epochs": 2,
            "planned_updates": 58,
            "gradient_accumulation": 8,
            "learning_rate": 0.00002,
            "warmup_updates": 0,
            "seed": 42,
        },
        "dataset": {
            "schema": UPDATE_DATASET_SCHEMA,
            "rows": 225,
            "architecture": "native_encoder_decoder",
            "heldout_overlap": 0,
            "known_contaminant_excluded": "sigless_6b1dd0c6b6fc",
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
            "typed_train": {
                "training_exclusions": {
                    "count": 1,
                    "task_ids": ["sigless_6b1dd0c6b6fc"],
                }
            },
        },
        "privacy": {
            "heldout_overlap": 0,
            "heldout_content_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
        },
        "warmstart": {
            "checkpoint_name": "checkpoint-optstep-000348",
            "update": 348,
            "run_contract_sha256": baseline_digest,
        },
        "lora": {
            "targets": baseline_contract["lora"]["targets"],
            "new_adapter_attached": False,
            "warmstart_weights_continued": True,
        },
    }
    paths = {
        "baseline_contract": tmp_path / "baseline_run_contract.json",
        "baseline_result": tmp_path / "baseline_result.json",
        "update_contract": tmp_path / "update_run_contract.json",
        "update_result": tmp_path / "update_result.json",
    }
    atomic_write_json(paths["baseline_contract"], baseline_contract)
    atomic_write_json(
        paths["baseline_result"],
        {
            "schema": BASELINE_CHECKPOINT_SCHEMA,
            "status": "complete",
            "updates": 348,
            "planned_updates": 348,
            "rows": 2775,
            "latest_checkpoint": "checkpoint-optstep-000348",
        },
    )
    atomic_write_json(paths["update_contract"], update_contract)
    atomic_write_json(
        paths["update_result"],
        {
            "schema": UPDATE_CHECKPOINT_SCHEMA,
            "status": "complete",
            "updates": 58,
            "planned_updates": 58,
            "rows": 225,
            "latest_checkpoint": "checkpoint-optstep-000058",
        },
    )
    return paths


def _outcome(arm: str, task_id: str, sample_index: int) -> tuple[bool, bool]:
    excluded = EXPECTED_EXCLUDED_TASK_IDS[0]
    if arm == "baseline":
        passed = task_id == "task_b" and sample_index == 0
        compiled = passed or (task_id == excluded and sample_index == 0)
    else:
        passed = (
            (task_id == "task_a" and sample_index == 0)
            or (task_id == excluded and sample_index == 1)
        )
        compiled = passed or (task_id == "task_b" and sample_index == 0)
    return compiled, passed


def _write_arm(
    *,
    root: Path,
    arm: str,
    task_ids: list[str],
    evaluation: Path,
    checkpoint_contract: Path,
) -> dict[str, Path]:
    root.mkdir(parents=True)
    prediction_path = root / f"{arm}_predictions.json"
    full_path = root / f"{arm}_full.json"
    clean_path = root / f"{arm}_clean.json"
    provenance_path = Path(str(prediction_path) + ".provenance.json")
    generation_path = Path(str(prediction_path) + ".generation.journal.jsonl")
    score_journal_path = Path(str(full_path) + ".evaluation.journal.jsonl")
    checkpoint = __import__("json").loads(
        checkpoint_contract.read_text(encoding="utf-8")
    )
    checkpoint_digest = canonical_sha256(checkpoint)
    sources = [_sha(f"source:{task_id}") for task_id in task_ids]
    task_digest = canonical_sha256(task_ids)
    source_digest = canonical_sha256(sources)
    input_view = {
        "schema": "t5gemma2-f2-measurement-input-view-v1",
        "view": EXPECTED_INPUT_VIEW,
        "rows": len(task_ids),
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "ordered_task_ids_sha256": task_digest,
        "ordered_source_sha256s_sha256": source_digest,
        "summary": {
            "intervention": "gold_derived_types_and_arity_only",
            "gold_implementation_body_exposed_to_model": False,
            "gold_semantic_parameter_names_exposed_to_model": False,
        },
    }
    heldout = {
        "dataset": {"sha256": sha256_file(evaluation), "rows": len(task_ids)},
        "dataset_seal": {"sha256": "2" * 64},
        "f2": {"sha256": "3" * 64, "rows": len(task_ids)},
        "f2_manifest": {"sha256": "4" * 64},
        "task_set_sha256": task_digest,
        "selected_rows": len(task_ids),
        "selected_ordered_task_ids_sha256": task_digest,
        "selected_ordered_source_sha256s_sha256": source_digest,
        "input_view": input_view,
        "model_visible_fields": [
            "transformed_F2.text",
            "gold_derived_types_and_arity",
        ],
        "tests_serialized_to_model": False,
        "full_gold_targets_serialized_to_model": False,
        "gold_targets_serialized_to_model": False,
        "gold_interface_types_and_arity_serialized_to_model": True,
    }
    model = {
        "name": "google/t5gemma-2-4b-4b",
        "revision": "4" * 40,
        "training_stage_schema": checkpoint["schema"],
        "tokenizer_sha256": "5" * 64,
        "warmstart_contract_sha256": checkpoint_digest,
        "adapter": {
            "run_contract_sha256": checkpoint_digest,
            "adapter_weights_sha256": ("6" if arm == "baseline" else "7") * 64,
            "adapter_config_sha256": "8" * 64,
        },
    }
    generation_contract = {
        "schema": INFERENCE_SCHEMA,
        "script_sha256": "9" * 64,
        "base_inference_script_sha256": ("a" if arm == "baseline" else "b") * 64,
        "arm": "sft",
        "model": model,
        "heldout": heldout,
        "sampling": dict(EXPECTED_SAMPLING),
        "runtime": {
            "torch": "fixture",
            "cuda": "fixture",
            "bf16": True,
            "attn_implementation": "sdpa",
        },
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "source_truncation": False,
    }
    append_event(
        generation_path,
        {
            "event": "header",
            "schema": GENERATION_JOURNAL_SCHEMA,
            "contract": generation_contract,
            "contract_sha256": canonical_sha256(generation_contract),
        },
    )
    prediction_rows: list[dict[str, Any]] = []
    capped = 0
    for task_index, (task_id, source_sha) in enumerate(zip(task_ids, sources, strict=True)):
        candidates = []
        texts = []
        for sample_index in range(10):
            text = f"void fn0() => print('{arm}:{task_id}:{sample_index}');"
            texts.append(text)
            candidates.append(
                {
                    "sample_index": sample_index,
                    "seed": 42 + task_index * 100_003,
                    "batch_position": sample_index,
                    "text": text,
                    "text_sha256": _sha(text),
                    "action_tokens": 12,
                    "eos_observed": True,
                    "max_token_completion": False,
                }
            )
        append_event(
            generation_path,
            {
                "event": "task_terminal",
                "schema": GENERATION_JOURNAL_SCHEMA,
                "task_index": task_index,
                "task_id": task_id,
                "source_sha256": source_sha,
                "encoder_tokens": 100 + task_index,
                "candidates": candidates,
            },
        )
        prediction_rows.append({"id": task_id, "predictions": texts})
    append_event(
        generation_path,
        {
            "event": "complete",
            "schema": GENERATION_JOURNAL_SCHEMA,
            "rows": len(task_ids),
            "predictions_canonical_sha256": canonical_sha256(prediction_rows),
        },
    )
    atomic_write_json(prediction_path, prediction_rows)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "architecture": "native_t5gemma2_encoder_decoder",
        "arm": "sft",
        "input_view": EXPECTED_INPUT_VIEW,
        "output_sha256": sha256_file(prediction_path),
        "num_rows": len(task_ids),
        "num_samples": 10,
        "model": model,
        "heldout": heldout,
        "sampling": dict(EXPECTED_SAMPLING),
        "max_token_completions": capped,
        "generation_journal": journal_record(generation_path),
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "gold_interface_types_and_arity_exposed_to_model": True,
        "sft_checkpoint_contract_sha256": checkpoint_digest,
    }
    atomic_write_json(provenance_path, provenance)

    jobs = _score_jobs(prediction_rows)
    slots = [f"{job['task_id']}:{job['sample_index']}" for job in jobs]
    score_contract = {
        "schema": SCORE_JOURNAL_SCHEMA,
        "predictions_sha256": sha256_file(prediction_path),
        "prediction_provenance_sha256": sha256_file(provenance_path),
        "evaluation_sha256": sha256_file(evaluation),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "completion_attestation": EXPECTED_COMPLETION_ATTESTATION,
        "k": 10,
        "workers": 32,
        "batch_size": 32,
        "timeout": 30,
        "stability_runs": 2,
        "ordered_slot_ids_sha256": canonical_sha256(slots),
        "slots": len(slots),
        "started_without_terminal_policy": EXPECTED_RETRY_POLICY,
    }
    append_event(
        score_journal_path,
        {
            "event": "score_header",
            "schema": SCORE_JOURNAL_SCHEMA,
            "contract": score_contract,
            "contract_sha256": canonical_sha256(score_contract),
        },
    )
    candidate_results: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(jobs), 32)):
        batch_jobs = jobs[start : start + 32]
        batch_slots = [
            f"{job['task_id']}:{job['sample_index']}" for job in batch_jobs
        ]
        started = append_event(
            score_journal_path,
            {
                "event": "score_batch_started",
                "schema": SCORE_JOURNAL_SCHEMA,
                "batch_index": batch_index,
                "slot_ids": batch_slots,
                "jobs_canonical_sha256": canonical_sha256(batch_jobs),
            },
        )
        results = []
        for job in batch_jobs:
            compiled, passed = _outcome(
                arm, str(job["task_id"]), int(job["sample_index"])
            )
            results.append(
                {
                    **job,
                    "compiled": compiled,
                    "passed": passed,
                    "diagnostic": "fixture",
                }
            )
        append_event(
            score_journal_path,
            {
                "event": "score_batch_terminal",
                "schema": SCORE_JOURNAL_SCHEMA,
                "batch_index": batch_index,
                "started_event_sha256": started["journal_event_sha256"],
                "retry_count": 0,
                "latest_retry_event_sha256": None,
                "candidate_results": results,
                "candidate_results_canonical_sha256": canonical_sha256(results),
            },
        )
        candidate_results.extend(results)
    append_event(
        score_journal_path,
        {
            "event": "score_complete",
            "schema": SCORE_JOURNAL_SCHEMA,
            "slots": len(jobs),
            "candidate_results_canonical_sha256": canonical_sha256(candidate_results),
            "rerun_slots": 0,
            "orphan_retry_events": 0,
            "orphan_rerun_slots": 0,
        },
    )
    by_task: dict[str, list[dict[str, Any]]] = {}
    for result in candidate_results:
        by_task.setdefault(str(result["task_id"]), []).append(result)
    task_results = []
    for task_id in sorted(by_task):
        rows = by_task[task_id]
        task_results.append(
            {
                "task_id": task_id,
                "pass_at_1": rows[0]["passed"],
                "pass_at_k": any(row["passed"] for row in rows),
                "compile_at_k": any(row["compiled"] for row in rows),
                "passing_samples": sum(row["passed"] for row in rows),
                "compiling_samples": sum(row["compiled"] for row in rows),
            }
        )
    counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in task_results),
        "pass_at_k": sum(row["pass_at_k"] for row in task_results),
        "compile_at_k": sum(row["compile_at_k"] for row in task_results),
    }
    score = {
        "schema": SCORE_SCHEMA,
        "predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "provenance_sha256": sha256_file(provenance_path),
        },
        "evaluation": {
            "path": str(evaluation.resolve()),
            "sha256": sha256_file(evaluation),
        },
        "k": 10,
        "timeout": 30,
        "stability_runs": 2,
        "evaluator": {
            "path": str(EVALUATOR.resolve()),
            "sha256": sha256_file(EVALUATOR),
            "completion_attestation": EXPECTED_COMPLETION_ATTESTATION,
        },
        "evaluation_journal": journal_record(score_journal_path),
        "started_without_terminal_policy": EXPECTED_RETRY_POLICY,
        "rerun_slots": 0,
        "orphan_retry_events": 0,
        "orphan_rerun_slots": 0,
        "tasks": len(task_ids),
        **{
            name: {"count": value, "rate": value / len(task_ids)}
            for name, value in counts.items()
        },
        "task_results": task_results,
        "candidate_results": candidate_results,
    }
    atomic_write_json(full_path, score)
    excluded = set(EXPECTED_EXCLUDED_TASK_IDS)
    clean_tasks = [row for row in task_results if row["task_id"] not in excluded]
    clean_candidates = [
        row for row in candidate_results if row["task_id"] not in excluded
    ]
    clean_counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in clean_tasks),
        "pass_at_k": sum(row["pass_at_k"] for row in clean_tasks),
        "compile_at_k": sum(row["compile_at_k"] for row in clean_tasks),
    }
    clean = {
        "schema": CLEAN_SCORE_SCHEMA,
        "source_score": {
            "path": str(full_path.resolve()),
            "sha256": sha256_file(full_path),
        },
        "source_score_schema": SCORE_SCHEMA,
        "excluded_task_ids": list(EXPECTED_EXCLUDED_TASK_IDS),
        "excluded_task_ids_sha256": canonical_sha256(
            list(EXPECTED_EXCLUDED_TASK_IDS)
        ),
        "exclusion_reason": (
            "known train/heldout exact acceptance-test duplicate in comparator training set"
        ),
        "tasks": len(clean_tasks),
        "k": 10,
        **{
            name: {"count": value, "rate": value / len(clean_tasks)}
            for name, value in clean_counts.items()
        },
        "task_results": clean_tasks,
        "candidate_results": clean_candidates,
    }
    atomic_write_json(clean_path, clean)
    return {
        "predictions": prediction_path,
        "provenance": provenance_path,
        "generation": generation_path,
        "full": full_path,
        "score_journal": score_journal_path,
        "clean": clean_path,
        "full_counts": counts,
        "clean_counts": clean_counts,
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    task_ids = ["task_b", EXPECTED_EXCLUDED_TASK_IDS[0], "task_a"]
    evaluation = tmp_path / "evaluation.jsonl"
    _write_evaluation(evaluation, task_ids)
    checkpoints = _checkpoint_contracts(tmp_path)
    baseline = _write_arm(
        root=tmp_path / "baseline",
        arm="baseline",
        task_ids=task_ids,
        evaluation=evaluation,
        checkpoint_contract=checkpoints["baseline_contract"],
    )
    update = _write_arm(
        root=tmp_path / "update",
        arm="update",
        task_ids=task_ids,
        evaluation=evaluation,
        checkpoint_contract=checkpoints["update_contract"],
    )
    baseline_seal = tmp_path / "baseline_seal.json"
    atomic_write_json(
        baseline_seal,
        {
            "schema": BASELINE_SEAL_SCHEMA,
            "status": "sealed",
            "evaluation_sha256": sha256_file(evaluation),
            "ordered_task_ids_sha256": canonical_sha256(task_ids),
            "checkpoint_contract_canonical_sha256": canonical_sha256(
                __import__("json").loads(
                    checkpoints["baseline_contract"].read_text(encoding="utf-8")
                )
            ),
            "artifacts": {
                "predictions_sha256": sha256_file(baseline["predictions"]),
                "provenance_sha256": sha256_file(baseline["provenance"]),
                "generation_journal_sha256": sha256_file(baseline["generation"]),
                "generation_journal_chain_head_sha256": sha256_file(
                    Path(str(baseline["generation"]) + ".chain-head.json")
                ),
                "full_score_sha256": sha256_file(baseline["full"]),
                "clean_score_sha256": sha256_file(baseline["clean"]),
                "evaluation_journal_sha256": sha256_file(
                    baseline["score_journal"]
                ),
                "evaluation_journal_chain_head_sha256": sha256_file(
                    Path(str(baseline["score_journal"]) + ".chain-head.json")
                ),
                "checkpoint_contract_sha256": sha256_file(
                    checkpoints["baseline_contract"]
                ),
                "training_result_sha256": sha256_file(
                    checkpoints["baseline_result"]
                ),
            },
            "metrics": {
                "full175": baseline["full_counts"],
                "clean174": baseline["clean_counts"],
            },
        },
    )
    return {
        "task_ids": task_ids,
        "evaluation": evaluation,
        "checkpoints": checkpoints,
        "baseline": baseline,
        "update": update,
        "baseline_seal": baseline_seal,
        "output": tmp_path / "audit.json",
    }


def _run_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    checkpoint = fixture["checkpoints"]
    return audit(
        baseline_predictions=fixture["baseline"]["predictions"],
        baseline_full_score=fixture["baseline"]["full"],
        baseline_clean_score=fixture["baseline"]["clean"],
        baseline_checkpoint_contract=checkpoint["baseline_contract"],
        baseline_training_result=checkpoint["baseline_result"],
        baseline_seal=fixture["baseline_seal"],
        update_predictions=fixture["update"]["predictions"],
        update_full_score=fixture["update"]["full"],
        update_clean_score=fixture["update"]["clean"],
        update_checkpoint_contract=checkpoint["update_contract"],
        update_training_result=checkpoint["update_result"],
        evaluation_file=fixture["evaluation"],
        evaluator_file=EVALUATOR,
        output=fixture["output"],
        expected_rows=3,
    )


def test_end_to_end_matched_audit_emits_exact_paired_partitions(
    tmp_path: Path,
) -> None:
    report = _run_fixture(_fixture(tmp_path))
    assert report["status"] == "pass"
    assert report["checks"]["no_source_truncation"] is True
    full = report["paired"]["full175"]["metrics"]
    assert full["pass_at_1"]["gains"]["task_ids"] == ["task_a"]
    assert full["pass_at_1"]["losses"]["task_ids"] == ["task_b"]
    assert full["pass_at_1"]["exact_two_sided_sign_mcnemar_p"]["fraction"] == "1/1"
    assert full["pass_at_10"]["gains"]["task_ids"] == [
        EXPECTED_EXCLUDED_TASK_IDS[0],
        "task_a",
    ]
    assert report["paired"]["clean174"]["tasks"] == 2


def test_audit_fails_closed_when_update_prediction_bytes_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prediction_path = fixture["update"]["predictions"]
    prediction_path.write_bytes(prediction_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="provenance|output_sha256|hash|predictions"):
        _run_fixture(fixture)
    assert not fixture["output"].exists()


def test_exact_sign_mcnemar_is_reported_as_reduced_rational() -> None:
    assert _exact_two_sided_sign_mcnemar(5, 0)["fraction"] == "1/16"
    assert _exact_two_sided_sign_mcnemar(4, 1)["fraction"] == "3/8"
    assert _exact_two_sided_sign_mcnemar(0, 0)["fraction"] == "1/1"


def test_supervisor_handoff_is_manual_fail_closed_and_uses_actual_next_program() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "t5gemma2-typed-local-direct-harvest" in text
    assert "t5gemma2-typed-rs-sft-full-train-harvest-k4" not in text
    assert 'if ! "${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_rs_sft.py' in text
    assert 'supervisorctl start "${NEXT_PROGRAM}"' in text
    assert "next program was not started" in text
    assert "RUNNING \"*|*\" STARTING" in text
    conf = CONF.read_text(encoding="utf-8")
    assert "[program:t5gemma2-typed-rs-sft-eval-audit-handoff]" in conf
    assert "autostart=false" in conf
    assert "autorestart=unexpected" in conf
    assert "exitcodes=0,78" in conf
