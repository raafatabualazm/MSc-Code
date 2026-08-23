from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"
sys.path.insert(0, str(PATCH))

import paired_compare as paired


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary(
    task_ids: list[str],
    passed: list[bool],
    *,
    k: int = 2,
    run_id: str = "run",
) -> dict[str, object]:
    cap = 131_072
    slot_policy = {
        "schema": paired.SLOT_POLICY_SCHEMA,
        "requested_model": "deepseek-v4-pro",
        "resolved_model_must_equal_requested": True,
        "k": k,
        "fixed_max_output_tokens": cap,
        "max_prompt_tokens": 12_000,
        "temperature": 0.8,
        "top_p": 0.95,
        "extra_body": {},
        "request_timeout_seconds": 7200,
        "max_transport_attempts_per_slot": 6,
        "every_returned_response_consumes_one_slot": True,
        "retry_only_when_no_provider_response": True,
        "finish_reason_length_consumes_slot": True,
        "finish_reason_does_not_blanket_invalidate_extractable_code": True,
        "safe_extractable_fn0_is_evaluated": True,
        "unusable_candidate_is_terminal_failure": True,
        "no_candidate_resampling": True,
        "duplicate_response_id_is_fatal": True,
        "response_identity_and_usage_contract_is_fatal": True,
        "early_stopping": False,
    }
    policy_sha = _stable_sha256(slot_policy)
    code_sha = hashlib.sha256(
        b"dynamic fn0() => 1;"
    ).hexdigest()
    task_results = []
    for task_id, task_passed in zip(task_ids, passed):
        outcomes = [
            {
                "task_id": task_id,
                "sample_index": sample_index,
                "attempt_id": f"{task_id}.s{sample_index}.a0",
                "response_id": f"{run_id}-{task_id}-{sample_index}",
                "finish_reason": "stop",
                "candidate_valid": True,
                "terminal_reason": "candidate_valid",
                "code_sha256": code_sha,
                "evaluation_performed": True,
                "compiled": task_passed and sample_index == 0,
                "passed": task_passed and sample_index == 0,
            }
            for sample_index in range(k)
        ]
        task_results.append(
            {
                "task_id": task_id,
                "terminal_responses": k,
                "evaluable_candidates": k,
                "invalid_candidates": 0,
                "length_slots": 0,
                "compiled": task_passed,
                "passed": task_passed,
                "candidate_outcomes": outcomes,
            }
        )
    successes = sum(passed)
    return {
        "schema": paired.RUN_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "dataset_label": run_id,
        "arm": "compact",
        "provider": "deepseek",
        "requested_model": "deepseek-v4-pro",
        "resolved_models": ["deepseek-v4-pro"],
        "fixed_max_output_tokens": cap,
        "slot_policy": slot_policy,
        "slot_policy_sha256": policy_sha,
        "task_set_sha256": _stable_sha256(task_ids),
        "k": k,
        "tasks": len(task_ids),
        "terminal_responses": len(task_ids) * k,
        "evaluable_candidates": len(task_ids) * k,
        "invalid_candidates": 0,
        "transport_retries": 0,
        "length_slots": 0,
        "model_invalid_responses": 0,
        "discarded_terminal_responses": 0,
        "terminal_reasons": {"candidate_valid": len(task_ids) * k},
        "all_tasks_have_exactly_k_terminal_provider_responses": True,
        "every_terminal_provider_response_has_exactly_one_outcome": True,
        "returned_responses_resampled": False,
        "transport_failures_only_retried": True,
        "early_stopping_used": False,
        "prompt_truncation_used": False,
        "usage": {
            "prompt_tokens": len(task_ids) * k * 100,
            "completion_tokens": len(task_ids) * k * 20,
            "total_tokens": len(task_ids) * k * 120,
        },
        "budget": {
            "limit": 0,
            "spent": len(task_ids) * k * 120,
            "reserved": 0,
        },
        "recorded_budget_charge_tokens": len(task_ids) * k * 120,
        "evaluator": {
            "sha256": "e" * 64,
            "entrypoint": "evaluate_dart_jit_tests_detail",
        },
        "completion_attestation_id": (
            "per-run-256-bit-marker-exactly-once-v1"
        ),
        "pass_at_k": {
            "successes": successes,
            "total": len(task_ids),
            "rate": successes / len(task_ids),
        },
        "compile_at_k": {
            "successes": successes,
            "total": len(task_ids),
            "rate": successes / len(task_ids),
        },
        "task_results": task_results,
    }


def _write_summary(
    tmp_path: Path,
    name: str,
    task_ids: list[str],
    passed: list[bool],
    *,
    k: int = 2,
) -> Path:
    return _write_completed_directory(
        tmp_path,
        name,
        task_ids,
        passed,
        k=k,
    )


def _config_for_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "schema": paired.RUN_SCHEMA,
        "provider": "deepseek",
        "model_requested": "deepseek-v4-pro",
        "max_output_tokens": summary["fixed_max_output_tokens"],
        "slot_policy": summary["slot_policy"],
        "slot_policy_sha256": summary["slot_policy_sha256"],
        "api_base_url_sha256": "1" * 64,
        "runtime_identity": {
            "runner_sha256": "2" * 64,
            "core_sha256": "3" * 64,
            "frontier_f2_sha256": "4" * 64,
            "openai_sdk_version": "1.0",
        },
        "expected_evaluator_sha256": "e" * 64,
        "expected_dart_sha256": "f" * 64,
        "dart_binary": "/dart",
        "timeout_seconds": 7200,
        "eval_timeout_seconds": 30,
        "eval_stability_runs": 2,
    }


def _write_completed_directory(
    tmp_path: Path,
    name: str,
    task_ids: list[str],
    passed: list[bool],
    *,
    k: int = 2,
) -> Path:
    run = tmp_path / name
    run.mkdir()
    summary = _summary(task_ids, passed, k=k, run_id=name)
    _write_json(run / "summary.json", summary)
    config = _config_for_summary(summary)
    config_sha = _stable_sha256(config)
    (run / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
    prompt_sha_by_task = {
        task_id: _stable_sha256(
            [{"role": "user", "content": f"prompt:{task_id}"}]
        )
        for task_id in task_ids
    }
    (run / "prompts.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema": paired.RUN_SCHEMA,
                    "task_id": task_id,
                    "prompt_sha256": prompt_sha_by_task[task_id],
                },
                sort_keys=True,
            )
            + "\n"
            for task_id in task_ids
        ),
        encoding="utf-8",
    )
    attempt_rows = []
    outcome_rows = []
    for task_result in summary["task_results"]:
        for outcome in task_result["candidate_outcomes"]:
            response_id = outcome["response_id"]
            attempt_rows.append(
                {
                    "schema": paired.RUN_SCHEMA,
                    "task_id": task_result["task_id"],
                    "sample_index": outcome["sample_index"],
                    "attempt_index": 0,
                    "attempt_id": outcome["attempt_id"],
                    "config_sha256": config_sha,
                    "prompt_sha256": prompt_sha_by_task[
                        task_result["task_id"]
                    ],
                    "requested_max_tokens": summary[
                        "fixed_max_output_tokens"
                    ],
                    "slot_policy_sha256": summary["slot_policy_sha256"],
                    "response_received": True,
                    "slot_terminal": True,
                    "candidate_valid": True,
                    "terminal_reason": "candidate_valid",
                    "transport_retry": False,
                    "transport_error": None,
                    "fatal_response_contract": False,
                    "response_id": response_id,
                    "resolved_model": "deepseek-v4-pro",
                    "finish_reason": "stop",
                    "response_created": None,
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                    "code_sha256": outcome["code_sha256"],
                    "content": "dynamic fn0() => 1;",
                    "reasoning_content": "",
                    "code": "dynamic fn0() => 1;",
                    "budget_charge_tokens": 120,
                    "response": {
                        "id": response_id,
                        "model": "deepseek-v4-pro",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": "dynamic fn0() => 1;"
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        },
                    },
                }
            )
            outcome_rows.append(
                {
                    "schema": paired.RUN_SCHEMA,
                    "config_sha256": config_sha,
                    "evaluator_sha256": "e" * 64,
                    "evaluation_performed": True,
                    "completion_attestation_enforced": True,
                    "completion_attestation_id": (
                        "per-run-256-bit-marker-exactly-once-v1"
                    ),
                    "completion_attestation_satisfied_all_runs": outcome[
                        "passed"
                    ],
                    "stability_runs": [
                        {
                            "compiled": outcome["compiled"],
                            "passed": outcome["passed"],
                            "completion_attestation_id": (
                                "per-run-256-bit-marker-exactly-once-v1"
                            ),
                            "completion_attestation_required": True,
                            "completion_attestation_satisfied": outcome[
                                "passed"
                            ],
                        }
                        for _ in range(2)
                    ],
                    **outcome,
                }
            )
    (run / "attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempt_rows),
        encoding="utf-8",
    )
    (run / "outcomes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in outcome_rows),
        encoding="utf-8",
    )
    _reseal_directory(run)
    return run


def _reseal_directory(run: Path) -> None:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = _config_for_summary(summary)
    provenance = {
        "schema": paired.RUN_SCHEMA,
        "status": "complete",
        "summary_sha256": _sha256(run / "summary.json"),
        "config": config,
        "config_sha256": _stable_sha256(config),
    }
    _write_json(run / "provenance.json", provenance)
    files = {}
    for filename in paired.REQUIRED_RUN_FILES:
        path = run / filename
        files[filename] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    _write_json(
        run / "manifest.json",
        {
            "schema": paired.RUN_SCHEMA,
            "files": files,
        },
    )


def test_paired_comparison_rederives_task_delta_bootstrap_and_mcnemar(
    tmp_path: Path,
) -> None:
    task_ids = ["t0", "t1", "t2", "t3"]
    path_a = _write_summary(
        tmp_path,
        "a",
        task_ids,
        [False, True, False, True],
    )
    path_b = _write_summary(
        tmp_path,
        "b",
        task_ids,
        [True, True, False, False],
    )
    arm_a = paired.load_completed_run(path_a)
    arm_b = paired.load_completed_run(path_b)

    first = paired.compare_completed_runs(
        arm_a,
        arm_b,
        bootstrap_replicates=500,
        bootstrap_seed=19,
    )
    second = paired.compare_completed_runs(
        arm_a,
        arm_b,
        bootstrap_replicates=500,
        bootstrap_seed=19,
    )

    assert first["pass_at_k"]["delta_arm_b_minus_arm_a"]["rate"] == 0.0
    assert first["paired_contingency"] == {
        "neither_passed": 1,
        "only_arm_a_passed": 1,
        "only_arm_b_passed": 1,
        "both_passed": 1,
    }
    assert first["mcnemar_exact"]["p_value"] == 1.0
    assert (
        first["pass_at_k"]["paired_bootstrap_95"]
        == second["pass_at_k"]["paired_bootstrap_95"]
    )
    assert [
        row["delta_arm_b_minus_arm_a"] for row in first["task_results"]
    ] == [1, 0, 0, -1]


def test_exact_mcnemar_reports_exact_fraction() -> None:
    result = paired.exact_mcnemar(0, 5)

    assert result["discordant_pairs"] == 5
    assert result["p_value"] == 0.0625
    assert result["p_value_exact_fraction"] == "1/16"


def test_rejects_same_task_set_in_different_order(tmp_path: Path) -> None:
    path_a = _write_summary(tmp_path, "a", ["t0", "t1"], [False, True])
    path_b = _write_summary(tmp_path, "b", ["t1", "t0"], [True, False])

    with pytest.raises(paired.ComparisonError, match="ordered task IDs mismatch"):
        paired.compare_completed_runs(
            paired.load_completed_run(path_a),
            paired.load_completed_run(path_b),
            bootstrap_replicates=10,
        )


def test_rejects_mismatched_k(tmp_path: Path) -> None:
    path_a = _write_summary(tmp_path, "a", ["t0", "t1"], [False, True], k=2)
    path_b = _write_summary(tmp_path, "b", ["t0", "t1"], [True, False], k=3)

    with pytest.raises(paired.ComparisonError, match="K mismatch"):
        paired.compare_completed_runs(
            paired.load_completed_run(path_a),
            paired.load_completed_run(path_b),
            bootstrap_replicates=10,
        )


def test_rejects_task_aggregate_that_disagrees_with_candidates(
    tmp_path: Path,
) -> None:
    run = _write_completed_directory(tmp_path, "corrupt", ["t0"], [True])
    value = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    value["task_results"][0]["candidate_outcomes"][0]["passed"] = False
    _write_json(run / "summary.json", value)
    _reseal_directory(run)

    with pytest.raises(paired.ComparisonError, match="pass@K flag disagrees"):
        paired.load_completed_run(run)


def test_directory_input_verifies_final_manifest(tmp_path: Path) -> None:
    run = _write_completed_directory(tmp_path, "run", ["t0"], [True])
    loaded = paired.load_completed_run(run)

    assert loaded.source_kind == "verified_run_directory"

    with (run / "outcomes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(paired.ComparisonError, match="byte-size mismatch"):
        paired.load_completed_run(run)


def test_prematerialized_comparison_requires_shared_pair_seal(
    tmp_path: Path,
) -> None:
    task_ids = ["t0", "t1"]
    path_a = _write_completed_directory(
        tmp_path, "opus", task_ids, [False, True]
    )
    path_b = _write_completed_directory(
        tmp_path, "codex", task_ids, [True, True]
    )
    value_a = json.loads((path_a / "summary.json").read_text(encoding="utf-8"))
    value_b = json.loads((path_b / "summary.json").read_text(encoding="utf-8"))
    pair_sha = "a" * 64
    acceptance_sha = "b" * 64
    value_a.update(
        {
            "input_mode": "prematerialized_f2",
            "pair_arm_key": "opus_real_fn0_cfg",
            "pair_manifest_sha256": pair_sha,
            "acceptance_test_sequence_sha256": acceptance_sha,
        }
    )
    value_b.update(
        {
            "input_mode": "prematerialized_f2",
            "pair_arm_key": "codex_multifunction_cfg",
            "pair_manifest_sha256": pair_sha,
            "acceptance_test_sequence_sha256": acceptance_sha,
        }
    )
    _write_json(path_a / "summary.json", value_a)
    _write_json(path_b / "summary.json", value_b)
    _reseal_directory(path_a)
    _reseal_directory(path_b)
    result = paired.compare_completed_runs(
        paired.load_completed_run(path_a),
        paired.load_completed_run(path_b),
        bootstrap_replicates=10,
    )
    assert result["pair_manifest_sha256"] == pair_sha
    assert result["acceptance_test_sequence_sha256"] == acceptance_sha

    value_b["pair_manifest_sha256"] = "c" * 64
    _write_json(path_b / "summary.json", value_b)
    _reseal_directory(path_b)
    with pytest.raises(paired.ComparisonError, match="different pair manifests"):
        paired.compare_completed_runs(
            paired.load_completed_run(path_a),
            paired.load_completed_run(path_b),
            bootstrap_replicates=10,
        )
