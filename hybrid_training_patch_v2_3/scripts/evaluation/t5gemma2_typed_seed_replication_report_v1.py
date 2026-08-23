#!/usr/bin/env python3
"""Audit typed T5Gemma-2 seed replications without selecting a winner.

Seed 42 is retained as a diagnostic link to the sealed historical result.  The
confirmatory replication consists of seeds 43..46, all of which must use the
same current inference and scoring stack.  The report deliberately computes no
promotion decision: pass@10 and extracted-code diversity remain the primary
quantities for a later, human-reviewed decision.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)


REPORT_SCHEMA = "t5gemma2-typed-seed-replication-report-v1"
PROVENANCE_SCHEMA = "t5gemma2-f2-measurement-ablation-provenance-v1"
SCORE_SCHEMA = "direct-compact-attested-passk-v1"
REQUIRED_ARMS = frozenset({"typed_sft", "incumbent", "pass3"})
OPTIONAL_ARMS = frozenset({"pass2"})
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
CONFIRMATORY_SEEDS = (43, 44, 45, 46)
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
HEX64 = re.compile(r"[0-9a-f]{64}")


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {path}") from exc


def _parse_arm(value: str) -> tuple[str, int, Path, Path]:
    parts = value.split("|", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError("--arm must be label|seed|predictions|full175_score")
    label, seed_text, predictions, score = parts
    if label not in REQUIRED_ARMS | OPTIONAL_ARMS:
        raise ValueError(f"unsupported arm label: {label}")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError(f"arm seed is not an integer: {seed_text}") from exc
    return (
        label,
        seed,
        Path(predictions).expanduser().resolve(),
        Path(score).expanduser().resolve(),
    )


def _journal_content_record(value: Mapping[str, Any]) -> dict[str, Any]:
    # Artifact paths can differ after a safe copy.  Content bindings cannot.
    return {
        key: value.get(key)
        for key in (
            "sha256",
            "chain_head_sha256",
            "event_count",
            "head_event_sha256",
        )
    }


def _metric(score: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = score.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"score lacks metric {name}")
    return {"count": int(value["count"]), "rate": float(value["rate"])}


def _exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _summarize_candidates(score: Mapping[str, Any], *, k: int) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in score.get("candidate_results") or []:
        grouped.setdefault(str(row.get("task_id") or ""), []).append(row)
    if len(grouped) != EXPECTED_TASKS:
        raise ValueError("score candidate task coverage differs")
    distinct_counts: list[int] = []
    passing_counts: list[int] = []
    for task_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row.get("sample_index", -1)))
        if [int(row.get("sample_index", -1)) for row in ordered] != list(range(k)):
            raise ValueError(f"{task_id}: score sample coverage differs")
        code_hashes = [str(row.get("code_sha256") or "") for row in ordered]
        if any(HEX64.fullmatch(value) is None for value in code_hashes):
            raise ValueError(f"{task_id}: invalid extracted-code SHA")
        distinct_counts.append(len(set(code_hashes)))
        passing_counts.append(sum(bool(row.get("passed")) for row in ordered))
    mean_distinct = sum(distinct_counts) / len(distinct_counts)
    return {
        "distinct_extracted_code_per_10": mean_distinct,
        "tasks_below_10_distinct": sum(value < k for value in distinct_counts),
        "distinct_histogram": {
            str(value): distinct_counts.count(value) for value in range(1, k + 1)
        },
        "successes_per_solved_task": sorted(value for value in passing_counts if value),
        "diversity_guardrail": {
            "reference": 10.0,
            "maximum_allowed_drop": 0.10,
            "minimum_allowed": 9.90,
            "passes": mean_distinct >= 9.90,
            "informational_only_not_a_promotion_decision": True,
        },
    }


def _validate_score_aggregates(
    score: Mapping[str, Any], *, ordered_task_ids: Sequence[str], k: int
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in score.get("candidate_results") or []:
        grouped.setdefault(str(row.get("task_id") or ""), []).append(row)
    expected_rows: list[dict[str, Any]] = []
    for task_id in ordered_task_ids:
        rows = sorted(
            grouped.get(task_id, []), key=lambda row: int(row.get("sample_index", -1))
        )
        if [int(row.get("sample_index", -1)) for row in rows] != list(range(k)):
            raise ValueError(f"{task_id}: score candidate slots differ")
        expected_rows.append(
            {
                "task_id": task_id,
                "pass_at_1": bool(rows[0].get("passed")),
                "pass_at_k": any(bool(row.get("passed")) for row in rows),
                "compile_at_k": any(bool(row.get("compiled")) for row in rows),
                "passing_samples": sum(bool(row.get("passed")) for row in rows),
                "compiling_samples": sum(bool(row.get("compiled")) for row in rows),
            }
        )
    observed_rows = {
        str(row.get("task_id") or ""): row for row in score.get("task_results") or []
    }
    if len(observed_rows) != len(expected_rows):
        raise ValueError("score task aggregates have different coverage")
    if any(observed_rows[row["task_id"]] != row for row in expected_rows):
        raise ValueError("score task aggregates are inconsistent")
    counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in expected_rows),
        "pass_at_k": sum(row["pass_at_k"] for row in expected_rows),
        "compile_at_k": sum(row["compile_at_k"] for row in expected_rows),
    }
    if any((score.get(name) or {}).get("count") != value for name, value in counts.items()):
        raise ValueError("score global aggregates are inconsistent")


def _load_arm(
    *,
    label: str,
    seed: int,
    predictions_path: Path,
    score_path: Path,
    evaluation_file: Path,
    expected_wrapper_sha256: str,
    expected_base_sha256: str,
    expected_evaluator_sha256: str,
    expected_adapter_sha256: str,
    enforce_current_code: bool,
) -> dict[str, Any]:
    provenance_path = Path(str(predictions_path) + ".provenance.json")
    generation_journal_path = Path(
        str(predictions_path) + ".generation.journal.jsonl"
    )
    score_journal_path = Path(str(score_path) + ".evaluation.journal.jsonl")
    adapter_sidecar_path = Path(
        str(predictions_path) + ".typed_seed_replication.json"
    )
    predictions = _read_json(predictions_path, f"{label} seed {seed} predictions")
    provenance = _read_json(provenance_path, f"{label} seed {seed} provenance")
    score = _read_json(score_path, f"{label} seed {seed} score")
    generation_journal = load_journal(generation_journal_path)
    score_journal = load_journal(score_journal_path)
    task_ids = [str(row.get("id") or "") for row in predictions]
    heldout = provenance.get("heldout") or {}
    input_view = heldout.get("input_view") or {}
    sampling = provenance.get("sampling") or {}
    if (
        not isinstance(predictions, list)
        or len(predictions) != EXPECTED_TASKS
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
        or heldout.get("selected_ordered_task_ids_sha256")
        != EXPECTED_TASK_ORDER_SHA256
        or heldout.get("selected_ordered_source_sha256s_sha256")
        != EXPECTED_TYPED_SOURCE_ORDER_SHA256
        or heldout.get("model_visible_fields")
        != ["transformed_F2.text", "gold_derived_types_and_arity"]
        or heldout.get("tests_serialized_to_model") is not False
        or heldout.get("full_gold_targets_serialized_to_model") is not False
        or heldout.get("gold_interface_types_and_arity_serialized_to_model")
        is not True
        or (heldout.get("dataset") or {}).get("sha256")
        != EXPECTED_DATASET_SHA256
        or (heldout.get("dataset_seal") or {}).get("sha256")
        != EXPECTED_DATASET_SEAL_SHA256
        or (heldout.get("f2") or {}).get("sha256") != EXPECTED_F2_SHA256
        or (heldout.get("f2_manifest") or {}).get("sha256")
        != EXPECTED_F2_MANIFEST_SHA256
        or input_view.get("view") != "typed_opaque_contract"
        or input_view.get("tests_exposed_to_model") is not False
        or input_view.get("full_gold_targets_exposed_to_model") is not False
        or (input_view.get("summary") or {}).get("function_name") != "fn0"
        or (input_view.get("summary") or {}).get("parameter_name_policy")
        != "p{zero_based_index}"
        or (input_view.get("summary") or {}).get(
            "gold_implementation_body_exposed_to_model"
        )
        is not False
        or (input_view.get("summary") or {}).get(
            "gold_semantic_parameter_names_exposed_to_model"
        )
        is not False
    ):
        raise ValueError(f"{label} seed {seed}: typed privacy/input contract failed")
    expected_sampling = {
        "num_samples": EXPECTED_K,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "max_source_tokens": 32768,
        "max_new_tokens": 4096,
        "seed": seed,
        "seed_policy": "seed+task_index*100003+batch_start",
        "generation_batch_size": 10,
        "decoder_prefix_is_not_output": True,
        "sampled_eos_retained": True,
        "fabricated_eos": False,
    }
    if sampling != expected_sampling:
        raise ValueError(f"{label} seed {seed}: sampling contract differs")
    if not generation_journal or not score_journal:
        raise ValueError(f"{label} seed {seed}: evaluation journal is empty")
    generation_contract = (generation_journal[0].get("contract") or {})
    score_contract = (score_journal[0].get("contract") or {})
    if (
        generation_journal[0].get("event") != "header"
        or generation_journal[-1].get("event") != "complete"
        or generation_contract.get("source_truncation") is not False
        or generation_contract.get("tests_exposed_to_model") is not False
        or generation_contract.get("full_gold_targets_exposed_to_model") is not False
        or generation_contract.get("sampling") != expected_sampling
        or _journal_content_record(provenance.get("generation_journal") or {})
        != _journal_content_record(journal_record(generation_journal_path))
        or score_journal[0].get("event") != "score_header"
        or score_journal[-1].get("event") != "score_complete"
        or score_contract.get("evaluation_sha256") != EXPECTED_DATASET_SHA256
        or score_contract.get("evaluator_sha256") != expected_evaluator_sha256
        or score_contract.get("k") != EXPECTED_K
        or score_contract.get("slots") != EXPECTED_TASKS * EXPECTED_K
        or score_contract.get("timeout") != 30
        or score_contract.get("stability_runs") != 2
        or score_contract.get("workers") != 32
        or score.get("schema") != SCORE_SCHEMA
        or score.get("tasks") != EXPECTED_TASKS
        or score.get("k") != EXPECTED_K
        or score.get("timeout") != 30
        or score.get("stability_runs") != 2
        or (score.get("evaluation") or {}).get("sha256")
        != EXPECTED_DATASET_SHA256
        or (score.get("evaluator") or {}).get("sha256")
        != expected_evaluator_sha256
        or (score.get("predictions") or {}).get("sha256")
        != sha256_file(predictions_path)
        or (score.get("predictions") or {}).get("provenance_sha256")
        != sha256_file(provenance_path)
        or _journal_content_record(score.get("evaluation_journal") or {})
        != _journal_content_record(journal_record(score_journal_path))
        or len(score.get("candidate_results") or [])
        != EXPECTED_TASKS * EXPECTED_K
        or len(score.get("task_results") or []) != EXPECTED_TASKS
    ):
        raise ValueError(f"{label} seed {seed}: full175 hash-chain contract failed")
    wrapper_sha = str(generation_contract.get("script_sha256") or "")
    base_sha = str(generation_contract.get("base_inference_script_sha256") or "")
    if enforce_current_code and (
        wrapper_sha != expected_wrapper_sha256 or base_sha != expected_base_sha256
    ):
        raise ValueError(f"{label} seed {seed}: current inference code differs")
    adapter_sidecar: dict[str, Any] | None = None
    if enforce_current_code:
        adapter_sidecar = _read_json(
            adapter_sidecar_path, f"{label} seed {seed} replication sidecar"
        )
        if (
            not isinstance(adapter_sidecar, Mapping)
            or adapter_sidecar.get("schema")
            != "t5gemma2-typed-seed-replication-adapter-v1"
            or adapter_sidecar.get("replication_arm") != label
            or adapter_sidecar.get("seed") != seed
            or adapter_sidecar.get("adapter_script_sha256")
            != expected_adapter_sha256
            or adapter_sidecar.get("predictions_sha256")
            != sha256_file(predictions_path)
            or adapter_sidecar.get("provenance_sha256")
            != sha256_file(provenance_path)
            or adapter_sidecar.get("model_sha256")
            != canonical_sha256(provenance.get("model") or {})
            or adapter_sidecar.get("input_view") != "typed_opaque_contract"
            or adapter_sidecar.get("tests_model_visible") is not False
            or adapter_sidecar.get("full_gold_implementation_model_visible")
            is not False
            or adapter_sidecar.get("automatic_promotion_performed") is not False
            or (
                label == "pass3"
                and not isinstance(adapter_sidecar.get("checkpoint_manifest"), Mapping)
            )
            or (
                label == "pass3"
                and (
                    HEX64.fullmatch(
                        str(
                            (adapter_sidecar.get("checkpoint_manifest") or {}).get(
                                "sha256"
                            )
                            or ""
                        )
                    )
                    is None
                    or not str(
                        (adapter_sidecar.get("checkpoint_manifest") or {}).get("path")
                        or ""
                    )
                    or not str(
                        (adapter_sidecar.get("checkpoint_manifest") or {}).get(
                            "run_contract_schema"
                        )
                        or ""
                    )
                )
            )
            or (
                label != "pass3"
                and adapter_sidecar.get("checkpoint_manifest") is not None
            )
        ):
            raise ValueError(f"{label} seed {seed}: replication sidecar differs")
    terminals = generation_journal[1:-1]
    if len(terminals) != EXPECTED_TASKS:
        raise ValueError(f"{label} seed {seed}: generation task coverage differs")
    for task_index, (task_id, terminal) in enumerate(zip(task_ids, terminals, strict=True)):
        candidates = terminal.get("candidates") or []
        expected_seed = seed + task_index * 100_003
        if (
            terminal.get("task_index") != task_index
            or terminal.get("task_id") != task_id
            or len(candidates) != EXPECTED_K
            or [candidate.get("sample_index") for candidate in candidates]
            != list(range(EXPECTED_K))
            or [candidate.get("batch_position") for candidate in candidates]
            != list(range(EXPECTED_K))
            or any(candidate.get("seed") != expected_seed for candidate in candidates)
        ):
            raise ValueError(f"{label} seed {seed}: generation seed coordinates differ")
    _validate_score_aggregates(score, ordered_task_ids=task_ids, k=EXPECTED_K)
    candidate_summary = _summarize_candidates(score, k=EXPECTED_K)
    return {
        "label": label,
        "seed": seed,
        "predictions": predictions_path,
        "score_path": score_path,
        "score": score,
        "provenance": provenance,
        "task_ids": task_ids,
        "model_sha256": canonical_sha256(provenance.get("model") or {}),
        "wrapper_sha256": wrapper_sha,
        "base_inference_sha256": base_sha,
        "checkpoint_manifest_sha256": (
            str((adapter_sidecar.get("checkpoint_manifest") or {}).get("sha256"))
            if label == "pass3" and adapter_sidecar is not None
            else None
        ),
        "metrics": {
            name: _metric(score, name)
            for name in ("pass_at_1", "pass_at_k", "compile_at_k")
        },
        **candidate_summary,
        "artifacts": {
            "predictions_sha256": sha256_file(predictions_path),
            "provenance_sha256": sha256_file(provenance_path),
            "generation_journal": _journal_content_record(
                journal_record(generation_journal_path)
            ),
            "score_sha256": sha256_file(score_path),
            "score_journal": _journal_content_record(
                journal_record(score_journal_path)
            ),
            "replication_sidecar_sha256": (
                sha256_file(adapter_sidecar_path) if enforce_current_code else None
            ),
        },
    }


def _paired(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_rows = {row["task_id"]: row for row in left["score"]["task_results"]}
    right_rows = {row["task_id"]: row for row in right["score"]["task_results"]}
    if set(left_rows) != set(right_rows) or left["task_ids"] != right["task_ids"]:
        raise ValueError("paired arm task identity/order differs")
    report: dict[str, Any] = {"tasks": len(left_rows)}
    for metric in ("pass_at_1", "pass_at_k", "compile_at_k"):
        gains = sum(
            bool(right_rows[task_id][metric]) and not bool(left_rows[task_id][metric])
            for task_id in left["task_ids"]
        )
        losses = sum(
            bool(left_rows[task_id][metric]) and not bool(right_rows[task_id][metric])
            for task_id in left["task_ids"]
        )
        report[metric] = {
            "incumbent_count": left["metrics"][metric]["count"],
            "pass3_count": right["metrics"][metric]["count"],
            "gains": gains,
            "losses": losses,
            "discordant": gains + losses,
            "exact_two_sided_mcnemar_p": _exact_mcnemar(gains, losses),
        }
    report["distinct_extracted_code_per_10"] = {
        "incumbent": left["distinct_extracted_code_per_10"],
        "pass3": right["distinct_extracted_code_per_10"],
        "difference_pass3_minus_incumbent": (
            right["distinct_extracted_code_per_10"]
            - left["distinct_extracted_code_per_10"]
        ),
    }
    return report


def _public_arm_record(arm: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": arm["seed"],
        "role": "diagnostic_only" if arm["seed"] == 42 else "confirmatory_replication",
        **arm["metrics"],
        "distinct_extracted_code_per_10": arm[
            "distinct_extracted_code_per_10"
        ],
        "tasks_below_10_distinct": arm["tasks_below_10_distinct"],
        "distinct_histogram": arm["distinct_histogram"],
        "successes_per_solved_task": arm["successes_per_solved_task"],
        "diversity_guardrail": arm["diversity_guardrail"],
        "model_sha256": arm["model_sha256"],
        "inference_code": {
            "wrapper_sha256": arm["wrapper_sha256"],
            "base_inference_sha256": arm["base_inference_sha256"],
        },
        "artifacts": arm["artifacts"],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_file = Path(args.evaluation_file).expanduser().resolve()
    if sha256_file(evaluation_file) != EXPECTED_DATASET_SHA256:
        raise ValueError("full175 evaluation file differs")
    specs = [_parse_arm(value) for value in args.arm]
    labels = {label for label, _, _, _ in specs}
    if not REQUIRED_ARMS <= labels or labels - REQUIRED_ARMS - OPTIONAL_ARMS:
        raise ValueError("typed_sft, incumbent, and pass3 arms are required")
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for label, seed, predictions, score in specs:
        if seed in by_arm.setdefault(label, {}):
            raise ValueError(f"duplicate arm/seed: {label}/{seed}")
        by_arm[label][seed] = _load_arm(
            label=label,
            seed=seed,
            predictions_path=predictions,
            score_path=score,
            evaluation_file=evaluation_file,
            expected_wrapper_sha256=args.expected_wrapper_sha256,
            expected_base_sha256=args.expected_base_inference_sha256,
            expected_evaluator_sha256=args.expected_evaluator_sha256,
            expected_adapter_sha256=args.expected_adapter_sha256,
            enforce_current_code=seed in CONFIRMATORY_SEEDS,
        )
    for label, arms in by_arm.items():
        if tuple(sorted(arms)) != EXPECTED_SEEDS:
            raise ValueError(f"{label}: seeds must be exactly 42..46")
        if len({arms[seed]["model_sha256"] for seed in EXPECTED_SEEDS}) != 1:
            raise ValueError(f"{label}: checkpoint identity differs across seeds")
        reference_order = arms[42]["task_ids"]
        if any(arms[seed]["task_ids"] != reference_order for seed in EXPECTED_SEEDS):
            raise ValueError(f"{label}: task order differs across seeds")
        if label == "pass3" and len(
            {
                arms[seed]["checkpoint_manifest_sha256"]
                for seed in CONFIRMATORY_SEEDS
            }
        ) != 1:
            raise ValueError("pass3: checkpoint manifest differs across confirmatory seeds")
    reference_order = by_arm["incumbent"][42]["task_ids"]
    if any(
        arm["task_ids"] != reference_order
        for arms in by_arm.values()
        for arm in arms.values()
    ):
        raise ValueError("cross-arm task order differs")
    if len(
        {
            by_arm[label][43]["model_sha256"]
            for label in ("typed_sft", "incumbent", "pass3")
        }
    ) != 3:
        raise ValueError("three required arms do not bind three distinct checkpoints")

    arms_report: dict[str, Any] = {}
    for label, arms in sorted(by_arm.items()):
        records = {str(seed): _public_arm_record(arms[seed]) for seed in EXPECTED_SEEDS}
        confirmatory = [arms[seed] for seed in CONFIRMATORY_SEEDS]
        arms_report[label] = {
            "seeds": records,
            "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
            "seed42_role": "diagnostic_only_not_used_for_promotion",
            "confirmatory_distribution": {
                metric: {
                    "counts": [arm["metrics"][metric]["count"] for arm in confirmatory],
                    "minimum": min(arm["metrics"][metric]["count"] for arm in confirmatory),
                    "maximum": max(arm["metrics"][metric]["count"] for arm in confirmatory),
                    "mean": statistics.mean(
                        arm["metrics"][metric]["count"] for arm in confirmatory
                    ),
                }
                for metric in ("pass_at_1", "pass_at_k", "compile_at_k")
            },
            "confirmatory_distinct_extracted_code_per_10": {
                "values": [arm["distinct_extracted_code_per_10"] for arm in confirmatory],
                "minimum": min(
                    arm["distinct_extracted_code_per_10"] for arm in confirmatory
                ),
                "maximum": max(
                    arm["distinct_extracted_code_per_10"] for arm in confirmatory
                ),
                "mean": statistics.mean(
                    arm["distinct_extracted_code_per_10"] for arm in confirmatory
                ),
                "all_seeds_meet_9_90_guardrail": all(
                    arm["distinct_extracted_code_per_10"] >= 9.90
                    for arm in confirmatory
                ),
            },
        }
    paired = {
        str(seed): _paired(by_arm["incumbent"][seed], by_arm["pass3"][seed])
        for seed in EXPECTED_SEEDS
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "contract": {
            "tasks": EXPECTED_TASKS,
            "k": EXPECTED_K,
            "diagnostic_seed": 42,
            "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
            "input_view": "typed_opaque_contract",
            "temperature": 0.8,
            "top_p": 0.95,
            "max_source_tokens": 32768,
            "max_new_tokens": 4096,
            "seed_policy": "seed+task_index*100003+batch_start",
            "timeout": 30,
            "stability_runs": 2,
            "expected_current_code": {
                "wrapper_sha256": args.expected_wrapper_sha256,
                "base_inference_sha256": args.expected_base_inference_sha256,
                "evaluator_sha256": args.expected_evaluator_sha256,
                "replication_adapter_sha256": args.expected_adapter_sha256,
            },
        },
        "checks": {
            "full175_hash_chains_validated": True,
            "confirmatory_current_inference_code_identical": True,
            "checkpoint_identity_stable_within_each_arm": True,
            "three_required_checkpoints_distinct": True,
            "task_order_and_typed_sources_identical": True,
            "heldout_175_model_visible": False,
            "tests_model_visible": False,
            "full_gold_implementation_model_visible": False,
            "semantic_parameter_names_model_visible": False,
            "only_opaque_types_and_arity_added": True,
        },
        "arms": arms_report,
        "paired_incumbent_vs_pass3_by_seed": paired,
        "analysis_policy": {
            "primary_metric": "pass_at_k",
            "primary_k": 10,
            "primary_diversity_measure": "distinct extracted-code SHA256 values per task/10",
            "minimum_confirmatory_seeds_for_any_pass_at_k_decision": 3,
            "available_confirmatory_seeds": len(CONFIRMATORY_SEEDS),
            "differences_below_approximately_three_tasks_do_not_promote": True,
            "diversity_veto_below": 9.90,
            "old_12250_generation_audit_role": "empirical_noise_floor_context_only",
            "automatic_promotion_performed": False,
            "promotion_decision": "not_performed",
            "compiler_or_pass_at_1_cannot_override_pass_at_10_or_diversity": True,
        },
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--evaluation-file", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-base-inference-sha256", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    for name in (
        "expected_wrapper_sha256",
        "expected_base_inference_sha256",
        "expected_evaluator_sha256",
        "expected_adapter_sha256",
    ):
        if HEX64.fullmatch(str(getattr(args, name))) is None:
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
