#!/usr/bin/env python3
"""Fail-closed, single-seed diagnostic gate for typed RS-SFT pass 3.

This is intentionally stricter than ``analysis_rs_sft_fold/check_collapse.py``.
The latter is run for its human-readable diagnostic only and always exits 0.
Here both score bytes are SHA-pinned, full 175 x 10 coverage and exact pairing
are validated, extracted-code diversity is recomputed, and a machine-readable
decision is written.  One seed can veto; it can never promote.  Promotion
requires at least three matched seeds under the revised analysis policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)


REPORT_SCHEMA = "t5gemma2-typed-pass3-single-seed-promotion-gate-v1"
SCORE_SCHEMA = "direct-compact-attested-passk-v1"
EVALUATION_SHA256 = "abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7"
TASKS = 175
K = 10
SEED = 42
INCUMBENT_PASS_AT_10 = 18
DIVERSITY_BAR = 9.90
COLLAPSE_FLOOR = 9.50
MIN_PROMOTION_SEEDS = 3
TYPED_SFT_REFERENCE_DISTINCT = 10.00
UPDATE58_REFERENCE_DISTINCT = 9.88


def _pin(path_value: str | Path, expected: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    text = str(expected or "")
    if (
        len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
        or not path.is_file()
        or sha256_file(path) != text
    ):
        raise ValueError(f"{label} differs from its exact SHA-256 pin")
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_array(path: Path, label: str) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _hex64(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _exact_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _validate_score(path: Path, *, label: str) -> dict[str, Any]:
    score = _read_object(path, label)
    candidates = score.get("candidate_results")
    tasks = score.get("task_results")
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("tasks") != TASKS
        or score.get("k") != K
        or score.get("stability_runs") != 2
        or score.get("evaluation", {}).get("sha256") != EVALUATION_SHA256
        or not isinstance(candidates, list)
        or len(candidates) != TASKS * K
        or not isinstance(tasks, list)
        or len(tasks) != TASKS
    ):
        raise ValueError(f"{label} does not have the sealed full175/k10 contract")
    task_order = [str(row.get("task_id") or "") for row in tasks]
    if any(not task_id for task_id in task_order) or len(set(task_order)) != TASKS:
        raise ValueError(f"{label} task-result identities differ")
    by_id: dict[str, list[Mapping[str, Any]]] = {task_id: [] for task_id in task_order}
    coordinates: list[tuple[str, int]] = []
    for row in candidates:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} candidate row is malformed")
        task_id = str(row.get("task_id") or "")
        sample = row.get("sample_index")
        digest = row.get("code_sha256")
        if (
            task_id not in by_id
            or type(sample) is not int
            or not 0 <= sample < K
            or not _hex64(digest)
            or type(row.get("compiled")) is not bool
            or type(row.get("passed")) is not bool
            or (row["passed"] and not row["compiled"])
        ):
            raise ValueError(f"{label} candidate binding differs")
        by_id[task_id].append(row)
        coordinates.append((task_id, sample))
    if len(set(coordinates)) != TASKS * K:
        raise ValueError(f"{label} contains duplicate sample coordinates")

    task_metrics: dict[str, dict[str, Any]] = {}
    distinct_counts: list[int] = []
    for task_row in tasks:
        task_id = str(task_row["task_id"])
        rows = sorted(by_id[task_id], key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in rows] != list(range(K)):
            raise ValueError(f"{label} does not cover exactly samples 0..9")
        pass1 = bool(rows[0]["passed"])
        passk = any(bool(row["passed"]) for row in rows)
        compk = any(bool(row["compiled"]) for row in rows)
        passing = sum(bool(row["passed"]) for row in rows)
        compiling = sum(bool(row["compiled"]) for row in rows)
        if (
            task_row.get("pass_at_1") is not pass1
            or task_row.get("pass_at_k") is not passk
            or task_row.get("compile_at_k") is not compk
            or task_row.get("passing_samples") != passing
            or task_row.get("compiling_samples") != compiling
        ):
            raise ValueError(f"{label} per-task summary differs from candidates")
        distinct = len({str(row["code_sha256"]) for row in rows})
        distinct_counts.append(distinct)
        task_metrics[task_id] = {
            "pass1": pass1,
            "passk": passk,
            "compk": compk,
            "distinct": distinct,
        }
    pass1_count = sum(row["pass1"] for row in task_metrics.values())
    passk_count = sum(row["passk"] for row in task_metrics.values())
    compk_count = sum(row["compk"] for row in task_metrics.values())
    for key, count in (
        ("pass_at_1", pass1_count),
        ("pass_at_k", passk_count),
        ("compile_at_k", compk_count),
    ):
        metric = score.get(key)
        if not isinstance(metric, Mapping) or metric.get("count") != count:
            raise ValueError(f"{label} headline {key} differs from candidates")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "task_order": task_order,
        "task_order_sha256": canonical_sha256(task_order),
        "sample_coordinates_sha256": canonical_sha256(coordinates),
        "pass_at_1": pass1_count,
        "pass_at_10": passk_count,
        "compile_at_10": compk_count,
        "distinct_mean": sum(distinct_counts) / TASKS,
        "distinct_counts_sha256": canonical_sha256(distinct_counts),
        "tasks_below_10_distinct": sum(value < K for value in distinct_counts),
        "task_metrics": task_metrics,
        "prediction_record": dict(score.get("predictions") or {}),
    }


def _validate_generation(
    *,
    predictions_path: Path,
    provenance_path: Path,
    score: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    predictions = _read_array(predictions_path, f"{label} predictions")
    provenance = _read_object(provenance_path, f"{label} provenance")
    prediction_sha = sha256_file(predictions_path)
    provenance_sha = sha256_file(provenance_path)
    prediction_record = score.get("prediction_record")
    if not isinstance(prediction_record, Mapping):
        raise ValueError(f"{label} score lacks prediction provenance")
    ids: list[str] = []
    for row in predictions:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} prediction row is malformed")
        task_id = str(row.get("id") or "")
        values = row.get("predictions")
        if (
            not task_id
            or not isinstance(values, list)
            or len(values) != K
            or any(not isinstance(value, str) for value in values)
        ):
            raise ValueError(f"{label} prediction coverage differs")
        ids.append(task_id)
    sampling = provenance.get("sampling")
    heldout = provenance.get("heldout")
    if (
        len(ids) != TASKS
        or len(set(ids)) != TASKS
        or set(ids) != set(score["task_order"])
        or prediction_record.get("sha256") != prediction_sha
        or prediction_record.get("provenance_sha256") != provenance_sha
        or provenance.get("schema") != "t5gemma2-f2-measurement-ablation-provenance-v1"
        or provenance.get("num_rows") != TASKS
        or provenance.get("num_samples") != K
        or provenance.get("input_view") != "typed_opaque_contract"
        or provenance.get("output_sha256") != prediction_sha
        or provenance.get("no_frontier_api") is not True
        or provenance.get("tests_exposed_to_model") is not False
        or provenance.get("full_gold_targets_exposed_to_model") is not False
        or not isinstance(sampling, Mapping)
        or sampling.get("seed") != SEED
        or sampling.get("seed_policy") != "seed+task_index*100003+batch_start"
        or sampling.get("num_samples") != K
        or sampling.get("generation_batch_size") != K
        or sampling.get("max_source_tokens") != 32768
        or sampling.get("max_new_tokens") != 4096
        or not math.isclose(float(sampling.get("temperature", -1)), 0.8, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(sampling.get("top_p", -1)), 0.95, rel_tol=0.0, abs_tol=1e-12)
        or sampling.get("top_k") != 0
        or sampling.get("sampled_eos_retained") is not True
        or sampling.get("fabricated_eos") is not False
        or not isinstance(heldout, Mapping)
        or heldout.get("dataset", {}).get("sha256") != EVALUATION_SHA256
        or heldout.get("selected_rows") != TASKS
        or heldout.get("tests_serialized_to_model") is not False
        or heldout.get("gold_targets_serialized_to_model") is not False
        or heldout.get("input_view", {}).get("view") != "typed_opaque_contract"
    ):
        raise ValueError(f"{label} generation/provenance contract differs")
    return {
        "predictions": {"path": str(predictions_path), "sha256": prediction_sha},
        "provenance": {"path": str(provenance_path), "sha256": provenance_sha},
        "task_ids_sha256": canonical_sha256(ids),
        "sampling": dict(sampling),
        "matched_seed42_contract_validated": True,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    current_path = _pin(args.pass3_score, args.expected_pass3_score_sha256, "pass-3 score")
    baseline_path = _pin(
        args.update58_score,
        args.expected_update58_score_sha256,
        "fresh current-stack update58 score",
    )
    checker_path = _pin(
        args.collapse_checker,
        args.expected_collapse_checker_sha256,
        "collapse diagnostic script",
    )
    current_predictions = _pin(
        args.pass3_predictions,
        args.expected_pass3_predictions_sha256,
        "pass-3 predictions",
    )
    current_provenance = _pin(
        args.pass3_provenance,
        args.expected_pass3_provenance_sha256,
        "pass-3 generation provenance",
    )
    baseline_predictions = _pin(
        args.update58_predictions,
        args.expected_update58_predictions_sha256,
        "update58 predictions",
    )
    baseline_provenance = _pin(
        args.update58_provenance,
        args.expected_update58_provenance_sha256,
        "update58 generation provenance",
    )
    current = _validate_score(current_path, label="pass-3 score")
    baseline = _validate_score(baseline_path, label="update58 score")
    current_generation = _validate_generation(
        predictions_path=current_predictions,
        provenance_path=current_provenance,
        score=current,
        label="pass-3",
    )
    baseline_generation = _validate_generation(
        predictions_path=baseline_predictions,
        provenance_path=baseline_provenance,
        score=baseline,
        label="update58",
    )
    if (
        current["task_order"] != baseline["task_order"]
        or current["sample_coordinates_sha256"] != baseline["sample_coordinates_sha256"]
        or baseline["pass_at_10"] != INCUMBENT_PASS_AT_10
        or not math.isclose(
            baseline["distinct_mean"],
            UPDATE58_REFERENCE_DISTINCT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("pass-3/update58 exact pairing or incumbent count differs")

    paired: dict[str, dict[str, Any]] = {}
    for key in ("passk", "pass1", "compk"):
        gains = sum(
            current["task_metrics"][task_id][key]
            and not baseline["task_metrics"][task_id][key]
            for task_id in current["task_order"]
        )
        losses = sum(
            baseline["task_metrics"][task_id][key]
            and not current["task_metrics"][task_id][key]
            for task_id in current["task_order"]
        )
        paired[key] = {
            "gains": gains,
            "losses": losses,
            "discordant": gains + losses,
            "exact_two_sided_p": _exact_p(gains, losses),
        }

    # Run the supplied checker only after the formal score audit above.  Its
    # return code is recorded but never trusted as a gate.
    diagnostic = subprocess.run(
        [sys.executable, str(checker_path), str(current_path), str(baseline_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    diagnostic_stdout = diagnostic.stdout
    diagnostic_stderr = diagnostic.stderr
    if diagnostic.returncode != 0:
        raise ValueError("human-readable collapse diagnostic failed to execute")

    distinct_ok = current["distinct_mean"] >= DIVERSITY_BAR
    passk_ok = current["pass_at_10"] >= INCUMBENT_PASS_AT_10
    collapsed = current["distinct_mean"] < COLLAPSE_FLOOR
    seed42_diagnostic_eligible = bool(distinct_ok and passk_ok and not collapsed)
    vetoes: list[str] = []
    if not passk_ok:
        vetoes.append("pass_at_10_below_update58_18_of_175")
    if not distinct_ok:
        vetoes.append("distinct_extracted_code_hashes_mean_below_9_90")
    if collapsed:
        vetoes.append("collapsed_distinct_mean_below_9_50")
    vetoes.append("only_one_seed_observed_minimum_three_required")

    return {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "contract": {
            "tasks": TASKS,
            "k": K,
            "seed": SEED,
            "evaluation_sha256": EVALUATION_SHA256,
            "primary_diagnostic_read": "mean_distinct_extracted_code_sha256_per_10",
            "pass_at_10_promotion_endpoint_requires_3plus_seeds": True,
            "pass_at_10_seed42_preregistered_diagnostic_floor": INCUMBENT_PASS_AT_10,
            "diversity_floor": DIVERSITY_BAR,
            "collapse_floor": COLLAPSE_FLOOR,
            "minimum_promotion_seeds": MIN_PROMOTION_SEEDS,
            "compile_at_10_and_pass_at_1_can_promote": False,
        },
        "inputs": {
            "pass3": {key: current[key] for key in (
                "path", "sha256", "task_order_sha256", "sample_coordinates_sha256"
            )} | {"generation": current_generation},
            "update58": {key: baseline[key] for key in (
                "path", "sha256", "task_order_sha256", "sample_coordinates_sha256"
            )} | {"generation": baseline_generation},
            "collapse_checker": {
                "path": str(checker_path),
                "sha256": sha256_file(checker_path),
                "return_code_not_used_as_gate": True,
            },
        },
        "metrics": {
            "pass3": {key: current[key] for key in (
                "pass_at_1", "pass_at_10", "compile_at_10", "distinct_mean",
                "distinct_counts_sha256", "tasks_below_10_distinct"
            )},
            "update58": {key: baseline[key] for key in (
                "pass_at_1", "pass_at_10", "compile_at_10", "distinct_mean",
                "distinct_counts_sha256", "tasks_below_10_distinct"
            )},
            "paired": paired,
        },
        "decision": {
            "seed42_diagnostic_eligible": seed42_diagnostic_eligible,
            "promotion_status": "HOLD_REQUIRES_3PLUS_SEEDS",
            "promoted_checkpoint": None,
            "vetoes_and_holds": vetoes,
            "single_seed_may_veto_but_may_not_promote": True,
            "differences_below_about_three_tasks_treated_as_noise_pending_replicates": True,
            "update58_is_comparator_not_fallback": True,
            "update58_diversity_eligible_under_9_90_bar": False,
            "update58_reference_distinct": UPDATE58_REFERENCE_DISTINCT,
            "current_measured_arm_within_diversity_bar": {
                "arm": "typed_contract_sft_2epoch_optstep348",
                "reference_distinct": TYPED_SFT_REFERENCE_DISTINCT,
                "reference_source": "sealed_preregistration_not_an_input_to_this_pairwise_gate",
            },
            "verpo_status": "HOLD",
            "verpo_hold_reason": "await_pass3_read_and_at_least_three_matched_seeds",
            "compiler_reward_saturation_claim_made": False,
        },
        "replication_status": {
            "validated_pass3_seeds": [SEED],
            "validated_seed_count": 1,
            "minimum_required_for_promotion": MIN_PROMOTION_SEEDS,
            "additional_matched_pass3_seeds_required": MIN_PROMOTION_SEEDS - 1,
            "seed43_to_46_measurement_audit_not_assumed_complete": True,
        },
        "human_readable_check_collapse": {
            "ran_after_formal_validation": True,
            "return_code": diagnostic.returncode,
            "stdout_sha256": hashlib.sha256(diagnostic_stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(diagnostic_stderr.encode("utf-8")).hexdigest(),
            "stdout": diagnostic_stdout,
            "stderr": diagnostic_stderr,
            "return_code_used_as_promotion_gate": False,
        },
        "exact_pairing_validated": True,
        "full175_k10_coverage_validated": True,
        "score_bytes_pinned_before_metrics_read": True,
        "matched_seed42_generation_provenance_validated": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pass3-score", required=True)
    parser.add_argument("--expected-pass3-score-sha256", required=True)
    parser.add_argument("--pass3-predictions", required=True)
    parser.add_argument("--expected-pass3-predictions-sha256", required=True)
    parser.add_argument("--pass3-provenance", required=True)
    parser.add_argument("--expected-pass3-provenance-sha256", required=True)
    parser.add_argument("--update58-score", required=True)
    parser.add_argument("--expected-update58-score-sha256", required=True)
    parser.add_argument("--update58-predictions", required=True)
    parser.add_argument("--expected-update58-predictions-sha256", required=True)
    parser.add_argument("--update58-provenance", required=True)
    parser.add_argument("--expected-update58-provenance-sha256", required=True)
    parser.add_argument("--collapse-checker", required=True)
    parser.add_argument("--expected-collapse-checker-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit(args)
        require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"T5GEMMA_TYPED_PASS3_PROMOTION_GATE_BLOCKED {exc}", flush=True)
        return 78
    print(
        "T5GEMMA_TYPED_PASS3_PROMOTION_GATE_COMPLETE "
        + json.dumps(
            {
                "promotion_status": report["decision"]["promotion_status"],
                "seed42_diagnostic_eligible": report["decision"]["seed42_diagnostic_eligible"],
                "verpo_status": report["decision"]["verpo_status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
