#!/usr/bin/env python3
"""Fail-closed three-seed comparison of typed Arm-C2 and its VeRPO pilot.

The statistical unit is the held-out task.  All ten candidates from all three
sampling seeds remain in the same task cluster during the paired bootstrap.
This report is descriptive/exploratory and cannot promote either checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation import t5gemma2_f2_passk_inference as base_inference
from scripts.evaluation import t5gemma2_measurement_audit_inference as measurement
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)


REPORT_SCHEMA = "t5gemma2-typed-c2-verpo-multiseed-eval-v1"
SIDECAR_SCHEMA = "t5gemma2-typed-c2-verpo-eval-adapter-v1"
BASELINE_SCHEMA = "t5gemma2-typed-fold-gold-replay-run-v2"
VERPO_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-run-v1"
PROVENANCE_SCHEMA = "t5gemma2-f2-measurement-ablation-provenance-v1"
SCORE_SCHEMA = "direct-compact-attested-passk-v1"
EXPECTED_LABELS = ("c2_baseline", "c2_verpo")
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_TASKS = 175
EXPECTED_K = 10
EXPECTED_DATASET_SHA256 = (
    "abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7"
)
EXPECTED_DATASET_SEAL_SHA256 = (
    "5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a"
)
EXPECTED_F2_SHA256 = (
    "6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab"
)
EXPECTED_F2_MANIFEST_SHA256 = (
    "777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44"
)
EXPECTED_TASK_ORDER_SHA256 = (
    "9b93767fd4d0b4057bc752113faeb1efda9faa609e537e189350a6d874d6e38e"
)
EXPECTED_TYPED_SOURCE_ORDER_SHA256 = (
    "b687b1c41aab33e0e634f8b4279386fc2dd2528cd3f789d88b79d0cd1c298b22"
)
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_REPLICATES = 10_000
HEX64 = re.compile(r"[0-9a-f]{64}")


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {path}") from exc


def _parse_artifact(value: str) -> tuple[str, int, Path, Path]:
    parts = value.split("|", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(
            "--artifact must be label|seed|predictions|full175_score"
        )
    label, seed_text, predictions, score = parts
    if label not in EXPECTED_LABELS:
        raise ValueError(f"unsupported arm label: {label}")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError(f"invalid sampling seed: {seed_text}") from exc
    return (
        label,
        seed,
        Path(predictions).expanduser().resolve(),
        Path(score).expanduser().resolve(),
    )


def _journal_content(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("sha256", "chain_head_sha256", "event_count", "head_event_sha256")
    }


def _exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _validate_score_aggregates(
    score: Mapping[str, Any], task_ids: Sequence[str]
) -> None:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in score.get("candidate_results") or []:
        by_task.setdefault(str(row.get("task_id") or ""), []).append(row)
    expected_task_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        rows = sorted(
            by_task.get(task_id, []), key=lambda row: int(row.get("sample_index", -1))
        )
        if [int(row.get("sample_index", -1)) for row in rows] != list(range(EXPECTED_K)):
            raise ValueError(f"{task_id}: score sample coverage differs")
        if any(HEX64.fullmatch(str(row.get("code_sha256") or "")) is None for row in rows):
            raise ValueError(f"{task_id}: invalid extracted-code hash")
        expected_task_rows.append(
            {
                "task_id": task_id,
                "pass_at_1": bool(rows[0].get("passed")),
                "pass_at_k": any(bool(row.get("passed")) for row in rows),
                "compile_at_k": any(bool(row.get("compiled")) for row in rows),
                "passing_samples": sum(bool(row.get("passed")) for row in rows),
                "compiling_samples": sum(bool(row.get("compiled")) for row in rows),
            }
        )
    observed = {
        str(row.get("task_id") or ""): row for row in score.get("task_results") or []
    }
    if len(observed) != EXPECTED_TASKS or any(
        observed.get(row["task_id"]) != row for row in expected_task_rows
    ):
        raise ValueError("score task aggregates are inconsistent")
    counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in expected_task_rows),
        "pass_at_k": sum(row["pass_at_k"] for row in expected_task_rows),
        "compile_at_k": sum(row["compile_at_k"] for row in expected_task_rows),
    }
    if any((score.get(metric) or {}).get("count") != count for metric, count in counts.items()):
        raise ValueError("score global aggregates are inconsistent")


def _candidate_views(score: Mapping[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in score["candidate_results"]:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    task_values: dict[str, dict[str, float]] = {}
    histogram = {str(value): 0 for value in range(1, EXPECTED_K + 1)}
    for task_id, raw_rows in grouped.items():
        rows = sorted(raw_rows, key=lambda row: int(row["sample_index"]))
        distinct = len({str(row["code_sha256"]) for row in rows})
        histogram[str(distinct)] += 1
        task_values[task_id] = {
            "pass_at_1": float(bool(rows[0]["passed"])),
            "pass_at_10": float(any(bool(row["passed"]) for row in rows)),
            "compile_at_10": float(any(bool(row["compiled"]) for row in rows)),
            "candidate_pass_rate": sum(bool(row["passed"]) for row in rows) / EXPECTED_K,
            "candidate_compile_rate": sum(bool(row["compiled"]) for row in rows) / EXPECTED_K,
            "distinct_per_10": float(distinct),
        }
    return task_values, histogram


def _load_artifact(
    *, label: str, seed: int, predictions_path: Path, score_path: Path
) -> dict[str, Any]:
    provenance_path = Path(str(predictions_path) + ".provenance.json")
    generation_journal_path = Path(str(predictions_path) + ".generation.journal.jsonl")
    score_journal_path = Path(str(score_path) + ".evaluation.journal.jsonl")
    predictions = _read_json(predictions_path, f"{label}/{seed} predictions")
    provenance = _read_json(provenance_path, f"{label}/{seed} provenance")
    score = _read_json(score_path, f"{label}/{seed} score")
    generation_journal = load_journal(generation_journal_path)
    score_journal = load_journal(score_journal_path)
    if not isinstance(predictions, list):
        raise ValueError(f"{label}/{seed}: predictions are not a list")
    task_ids = [str(row.get("id") or "") for row in predictions]
    heldout = provenance.get("heldout") or {}
    view = heldout.get("input_view") or {}
    sampling = provenance.get("sampling") or {}
    expected_sampling = {
        "num_samples": EXPECTED_K,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "max_source_tokens": 32768,
        "max_new_tokens": 8192,
        "seed": seed,
        "seed_policy": "seed+task_index*100003+batch_start",
        "generation_batch_size": 10,
        "decoder_prefix_is_not_output": True,
        "sampled_eos_retained": True,
        "fabricated_eos": False,
    }
    expected_schema = BASELINE_SCHEMA if label == "c2_baseline" else VERPO_SCHEMA
    expected_floor = label == "c2_baseline"
    model = provenance.get("model") or {}
    if (
        len(predictions) != EXPECTED_TASKS
        or len(set(task_ids)) != EXPECTED_TASKS
        or canonical_sha256(task_ids) != EXPECTED_TASK_ORDER_SHA256
        or any(len(row.get("predictions") or []) != EXPECTED_K for row in predictions)
        or provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("input_view") != "typed_opaque_contract"
        or provenance.get("num_rows") != EXPECTED_TASKS
        or provenance.get("num_samples") != EXPECTED_K
        or provenance.get("output_sha256") != sha256_file(predictions_path)
        or provenance.get("tests_exposed_to_model") is not False
        or provenance.get("full_gold_targets_exposed_to_model") is not False
        or provenance.get("gold_interface_types_and_arity_exposed_to_model") is not True
        or heldout.get("selected_rows") != EXPECTED_TASKS
        or heldout.get("selected_ordered_task_ids_sha256") != EXPECTED_TASK_ORDER_SHA256
        or heldout.get("selected_ordered_source_sha256s_sha256") != EXPECTED_TYPED_SOURCE_ORDER_SHA256
        or heldout.get("model_visible_fields") != ["transformed_F2.text", "gold_derived_types_and_arity"]
        or heldout.get("tests_serialized_to_model") is not False
        or heldout.get("full_gold_targets_serialized_to_model") is not False
        or heldout.get("gold_interface_types_and_arity_serialized_to_model") is not True
        or (heldout.get("dataset") or {}).get("sha256") != EXPECTED_DATASET_SHA256
        or (heldout.get("dataset_seal") or {}).get("sha256") != EXPECTED_DATASET_SEAL_SHA256
        or (heldout.get("f2") or {}).get("sha256") != EXPECTED_F2_SHA256
        or (heldout.get("f2_manifest") or {}).get("sha256") != EXPECTED_F2_MANIFEST_SHA256
        or view.get("view") != "typed_opaque_contract"
        or view.get("tests_exposed_to_model") is not False
        or view.get("full_gold_targets_exposed_to_model") is not False
        or (view.get("summary") or {}).get("function_name") != "fn0"
        or (view.get("summary") or {}).get("parameter_name_policy") != "p{zero_based_index}"
        or (view.get("summary") or {}).get("gold_implementation_body_exposed_to_model") is not False
        or (view.get("summary") or {}).get("gold_semantic_parameter_names_exposed_to_model") is not False
        or sampling != expected_sampling
        or model.get("training_stage_schema") != expected_schema
        or model.get("production_floor_eligible") is not expected_floor
    ):
        raise ValueError(f"{label}/{seed}: typed evaluation contract differs")

    expected_evaluator_sha = sha256_file(
        Path(base_inference.__file__).with_name("graph_compile_at_k_antigravity.py")
    )
    generation_contract = generation_journal[0].get("contract") if generation_journal else None
    score_contract = score_journal[0].get("contract") if score_journal else None
    if (
        not isinstance(generation_contract, Mapping)
        or generation_journal[0].get("event") != "header"
        or generation_journal[-1].get("event") != "complete"
        or generation_contract.get("script_sha256") != sha256_file(Path(measurement.__file__).resolve())
        or generation_contract.get("base_inference_script_sha256") != sha256_file(Path(base_inference.__file__).resolve())
        or generation_contract.get("sampling") != expected_sampling
        or (generation_contract.get("runtime") or {}).get("bf16") is not True
        or (generation_contract.get("runtime") or {}).get("attn_implementation") != "sdpa"
        or generation_contract.get("source_truncation") is not False
        or generation_contract.get("tests_exposed_to_model") is not False
        or generation_contract.get("full_gold_targets_exposed_to_model") is not False
        or _journal_content(provenance.get("generation_journal") or {})
        != _journal_content(journal_record(generation_journal_path))
        or not isinstance(score_contract, Mapping)
        or score_journal[0].get("event") != "score_header"
        or score_journal[-1].get("event") != "score_complete"
        or score_contract.get("evaluation_sha256") != EXPECTED_DATASET_SHA256
        or score_contract.get("evaluator_sha256") != expected_evaluator_sha
        or score_contract.get("k") != EXPECTED_K
        or score_contract.get("slots") != EXPECTED_TASKS * EXPECTED_K
        or score_contract.get("workers") != 32
        or score_contract.get("timeout") != 30
        or score_contract.get("stability_runs") != 2
        or score.get("schema") != SCORE_SCHEMA
        or score.get("tasks") != EXPECTED_TASKS
        or score.get("k") != EXPECTED_K
        or score.get("timeout") != 30
        or score.get("stability_runs") != 2
        or (score.get("evaluation") or {}).get("sha256") != EXPECTED_DATASET_SHA256
        or (score.get("evaluator") or {}).get("sha256") != expected_evaluator_sha
        or (score.get("predictions") or {}).get("sha256") != sha256_file(predictions_path)
        or (score.get("predictions") or {}).get("provenance_sha256") != sha256_file(provenance_path)
        or _journal_content(score.get("evaluation_journal") or {})
        != _journal_content(journal_record(score_journal_path))
        or len(score.get("candidate_results") or []) != EXPECTED_TASKS * EXPECTED_K
        or len(score.get("task_results") or []) != EXPECTED_TASKS
    ):
        raise ValueError(f"{label}/{seed}: evaluation hash chain differs")

    terminals = generation_journal[1:-1]
    if len(terminals) != EXPECTED_TASKS:
        raise ValueError(f"{label}/{seed}: generation coverage differs")
    for task_index, (task_id, terminal) in enumerate(zip(task_ids, terminals, strict=True)):
        candidates = terminal.get("candidates") or []
        if (
            terminal.get("task_index") != task_index
            or terminal.get("task_id") != task_id
            or len(candidates) != EXPECTED_K
            or [candidate.get("sample_index") for candidate in candidates] != list(range(EXPECTED_K))
            or [candidate.get("batch_position") for candidate in candidates] != list(range(EXPECTED_K))
            or any(candidate.get("seed") != seed + task_index * 100_003 for candidate in candidates)
        ):
            raise ValueError(f"{label}/{seed}: sample coordinates differ")
    _validate_score_aggregates(score, task_ids)
    task_values, histogram = _candidate_views(score)

    sidecar_record: dict[str, Any] | None = None
    sidecar_path = Path(str(predictions_path) + ".typed_c2_verpo_eval.json")
    if label == "c2_verpo":
        sidecar = _read_json(sidecar_path, f"{label}/{seed} adapter sidecar")
        if (
            not isinstance(sidecar, Mapping)
            or sidecar.get("schema") != SIDECAR_SCHEMA
            or sidecar.get("seed") != seed
            or sidecar.get("training_stage_schema") != VERPO_SCHEMA
            or sidecar.get("final_update") != 150
            or sidecar.get("predictions_sha256") != sha256_file(predictions_path)
            or sidecar.get("provenance_sha256") != sha256_file(provenance_path)
            or sidecar.get("model_sha256") != canonical_sha256(model)
            or sidecar.get("automatic_promotion_performed") is not False
        ):
            raise ValueError(f"{label}/{seed}: VeRPO adapter sidecar differs")
        sidecar_record = {
            "path": str(sidecar_path),
            "sha256": sha256_file(sidecar_path),
            "checkpoint_manifest_sha256": sidecar.get("checkpoint_manifest_sha256"),
        }
    return {
        "label": label,
        "seed": seed,
        "task_ids": task_ids,
        "task_values": task_values,
        "score": score,
        "model_sha256": canonical_sha256(model),
        "checkpoint_contract_sha256": provenance.get("sft_checkpoint_contract_sha256"),
        "metrics": {
            metric: {
                "count": int(score[metric]["count"]),
                "rate": float(score[metric]["rate"]),
            }
            for metric in ("pass_at_1", "pass_at_k", "compile_at_k")
        },
        "distinct_mean": statistics.mean(
            values["distinct_per_10"] for values in task_values.values()
        ),
        "tasks_below_10_distinct": sum(
            values["distinct_per_10"] < EXPECTED_K for values in task_values.values()
        ),
        "distinct_histogram": histogram,
        "artifacts": {
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
            "provenance": {"path": str(provenance_path), "sha256": sha256_file(provenance_path)},
            "generation_journal": _journal_content(journal_record(generation_journal_path)),
            "score": {"path": str(score_path), "sha256": sha256_file(score_path)},
            "score_journal": _journal_content(journal_record(score_journal_path)),
            "adapter_sidecar": sidecar_record,
        },
    }


def _paired_seed(baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    if baseline["task_ids"] != treatment["task_ids"]:
        raise ValueError("paired task identity/order differs")
    result: dict[str, Any] = {}
    for key in ("pass_at_1", "pass_at_10", "compile_at_10"):
        gains = sum(
            treatment["task_values"][task_id][key] > baseline["task_values"][task_id][key]
            for task_id in baseline["task_ids"]
        )
        losses = sum(
            treatment["task_values"][task_id][key] < baseline["task_values"][task_id][key]
            for task_id in baseline["task_ids"]
        )
        result[key] = {
            "gains": gains,
            "losses": losses,
            "discordant": gains + losses,
            "exact_two_sided_mcnemar_p": _exact_mcnemar(gains, losses),
        }
    result["distinct_per_10"] = {
        "baseline": baseline["distinct_mean"],
        "verpo": treatment["distinct_mean"],
        "difference": treatment["distinct_mean"] - baseline["distinct_mean"],
    }
    return result


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _cluster_bootstrap(
    by_arm: Mapping[str, Mapping[int, Mapping[str, Any]]], task_ids: Sequence[str]
) -> dict[str, Any]:
    metrics = (
        "pass_at_1",
        "pass_at_10",
        "compile_at_10",
        "candidate_pass_rate",
        "candidate_compile_rate",
        "distinct_per_10",
        "pass_at_30",
    )
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    arm_points: dict[str, dict[str, float]] = {label: {} for label in EXPECTED_LABELS}
    clustered: dict[str, dict[str, dict[str, float]]] = {label: {} for label in EXPECTED_LABELS}
    for label in EXPECTED_LABELS:
        for task_id in task_ids:
            per_seed = [by_arm[label][seed]["task_values"][task_id] for seed in EXPECTED_SEEDS]
            clustered[label][task_id] = {
                metric: (
                    float(any(row["pass_at_10"] > 0.0 for row in per_seed))
                    if metric == "pass_at_30"
                    else statistics.mean(row[metric] for row in per_seed)
                )
                for metric in metrics
            }
        arm_points[label] = {
            metric: statistics.mean(clustered[label][task_id][metric] for task_id in task_ids)
            for metric in metrics
        }
    for task_id in task_ids:
        for metric in metrics:
            deltas[metric].append(
                clustered["c2_verpo"][task_id][metric]
                - clustered["c2_baseline"][task_id][metric]
            )
    rng = random.Random(BOOTSTRAP_SEED)
    distributions = {metric: [] for metric in metrics}
    task_count = len(task_ids)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(task_count) for _ in range(task_count)]
        for metric in metrics:
            distributions[metric].append(
                sum(deltas[metric][index] for index in indices) / task_count
            )
    comparison: dict[str, Any] = {}
    for metric in metrics:
        values = sorted(distributions[metric])
        point = statistics.mean(deltas[metric])
        comparison[metric] = {
            "baseline": arm_points["c2_baseline"][metric],
            "verpo": arm_points["c2_verpo"][metric],
            "difference_verpo_minus_baseline": point,
            "paired_task_cluster_bootstrap_95_percentile_ci": [
                _percentile(values, 0.025),
                _percentile(values, 0.975),
            ],
            "bootstrap_probability_difference_gt_0": sum(value > 0.0 for value in values) / len(values),
        }
    return {
        "unit": "heldout_task_with_all_three_sampling_seeds_and_all_candidates_kept_together",
        "task_clusters": task_count,
        "seeds_per_cluster": len(EXPECTED_SEEDS),
        "candidates_per_cluster_per_arm": len(EXPECTED_SEEDS) * EXPECTED_K,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "interval": "paired_nonparametric_task_cluster_percentile_95pct",
        "metrics": comparison,
    }


def _public_arm(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": artifact["seed"],
        **artifact["metrics"],
        "distinct_extracted_code_per_10": artifact["distinct_mean"],
        "tasks_below_10_distinct": artifact["tasks_below_10_distinct"],
        "distinct_histogram": artifact["distinct_histogram"],
        "model_sha256": artifact["model_sha256"],
        "checkpoint_contract_sha256": artifact["checkpoint_contract_sha256"],
        "artifacts": artifact["artifacts"],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    specs = [_parse_artifact(value) for value in args.artifact]
    by_arm: dict[str, dict[int, dict[str, Any]]] = {label: {} for label in EXPECTED_LABELS}
    for label, seed, predictions, score in specs:
        if seed in by_arm[label]:
            raise ValueError(f"duplicate artifact: {label}/{seed}")
        by_arm[label][seed] = _load_artifact(
            label=label, seed=seed, predictions_path=predictions, score_path=score
        )
    for label in EXPECTED_LABELS:
        if tuple(sorted(by_arm[label])) != EXPECTED_SEEDS:
            raise ValueError(f"{label}: seeds must be exactly 42, 43, and 44")
        if len({by_arm[label][seed]["model_sha256"] for seed in EXPECTED_SEEDS}) != 1:
            raise ValueError(f"{label}: checkpoint identity differs across seeds")
        if len({by_arm[label][seed]["checkpoint_contract_sha256"] for seed in EXPECTED_SEEDS}) != 1:
            raise ValueError(f"{label}: checkpoint contract differs across seeds")
    task_ids = by_arm["c2_baseline"][42]["task_ids"]
    if any(
        by_arm[label][seed]["task_ids"] != task_ids
        for label in EXPECTED_LABELS
        for seed in EXPECTED_SEEDS
    ):
        raise ValueError("cross-arm task order differs")
    if by_arm["c2_baseline"][42]["model_sha256"] == by_arm["c2_verpo"][42]["model_sha256"]:
        raise ValueError("baseline and VeRPO checkpoint identities are equal")

    arms = {
        label: {
            "training_stage_schema": BASELINE_SCHEMA if label == "c2_baseline" else VERPO_SCHEMA,
            "seeds": {str(seed): _public_arm(by_arm[label][seed]) for seed in EXPECTED_SEEDS},
            "three_seed_mean": {
                metric: statistics.mean(by_arm[label][seed]["metrics"][metric]["rate"] for seed in EXPECTED_SEEDS)
                for metric in ("pass_at_1", "pass_at_k", "compile_at_k")
            },
            "three_seed_mean_distinct_per_10": statistics.mean(
                by_arm[label][seed]["distinct_mean"] for seed in EXPECTED_SEEDS
            ),
        }
        for label in EXPECTED_LABELS
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "contract": {
            "heldout_tasks": EXPECTED_TASKS,
            "k": EXPECTED_K,
            "seeds": list(EXPECTED_SEEDS),
            "input_view": "typed_opaque_contract",
            "sampling": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_source_tokens": 32768,
                "max_new_tokens": 8192,
            },
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "task_order_sha256": EXPECTED_TASK_ORDER_SHA256,
        },
        "checks": {
            "same_byte_identical_measurement_inference_core": True,
            "same_byte_identical_passk_scorer": True,
            "same_task_order_and_typed_sources": True,
            "same_sampling_coordinates_within_each_seed": True,
            "all_generation_and_scoring_hash_chains_validated": True,
            "heldout_175_model_visible_during_training": False,
            "tests_model_visible_during_generation": False,
            "private_holdback_used_for_selection_or_training": False,
        },
        "arms": arms,
        "paired_by_seed": {
            str(seed): _paired_seed(by_arm["c2_baseline"][seed], by_arm["c2_verpo"][seed])
            for seed in EXPECTED_SEEDS
        },
        "paired_task_cluster_bootstrap": _cluster_bootstrap(by_arm, task_ids),
        "decision": {
            "status": "STOP_AFTER_MATCHED_EVALUATION",
            "automatic_promotion_performed": False,
            "promoted_checkpoint": None,
            "promotion_permitted_from_this_report": False,
            "reason": "discardable VeRPO experiment; three-seed report is exploratory and requires human interpretation",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
        require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"T5GEMMA_TYPED_C2_VERPO_MULTISEED_EVAL_BLOCKED {exc}", flush=True)
        return 78
    print(
        "T5GEMMA_TYPED_C2_VERPO_MULTISEED_EVAL_COMPLETE "
        + json.dumps(
            {
                "decision": report["decision"]["status"],
                "automatic_promotion_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
