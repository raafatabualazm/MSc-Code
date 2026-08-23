#!/usr/bin/env python3
"""Validate and summarize the frozen-checkpoint F2 measurement audit."""

from __future__ import annotations

import argparse
import json
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


REPORT_SCHEMA = "t5gemma2-f2-measurement-audit-report-v1"
BASE_PROVENANCE_SCHEMA = "direct-compact-inference-v1"
ABLATION_PROVENANCE_SCHEMA = (
    "t5gemma2-f2-measurement-ablation-provenance-v1"
)
SCORE_SCHEMA = "direct-compact-attested-passk-v1"
_ARITY_DIAGNOSTIC = re.compile(
    r"too (?:many|few) positional arguments|positional arguments?:|"
    r"required named parameter|no named parameter|missing required argument",
    re.IGNORECASE,
)
_TYPE_DIAGNOSTIC = re.compile(
    r"(?:argument|value) of type .+ can't be assigned|the argument type|"
    r"is not a subtype|isn't a type|return type",
    re.IGNORECASE,
)


def _read(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {path}") from exc


def _parse_spec(value: str, *, numeric: bool) -> tuple[str, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("artifact spec must be label|predictions|score")
    label = parts[0]
    if numeric:
        int(label)
    return label, Path(parts[1]).expanduser().resolve(), Path(parts[2]).expanduser().resolve()


def _journal_content_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare journal content across machines without trusting copied paths."""

    return {
        key: value.get(key)
        for key in (
            "sha256",
            "chain_head_sha256",
            "event_count",
            "head_event_sha256",
        )
    }


def _load_arm(
    *,
    label: str,
    predictions_path: Path,
    score_path: Path,
    expected_tasks: int,
    expected_k: int,
    expected_provenance_schema: str,
) -> dict[str, Any]:
    provenance_path = Path(str(predictions_path) + ".provenance.json")
    journal_path = Path(str(predictions_path) + ".generation.journal.jsonl")
    predictions = _read(predictions_path, f"{label} predictions")
    score = _read(score_path, f"{label} score")
    provenance = _read(provenance_path, f"{label} provenance")
    journal = load_journal(journal_path)
    task_ids = [str(row.get("id") or "") for row in predictions]
    score_ids = [str(row.get("task_id") or "") for row in score.get("task_results") or []]
    if (
        not isinstance(predictions, list)
        or len(predictions) != expected_tasks
        or len(set(task_ids)) != expected_tasks
        or any(len(row.get("predictions") or []) != expected_k for row in predictions)
        or provenance.get("schema") != expected_provenance_schema
        or provenance.get("num_rows") != expected_tasks
        or provenance.get("num_samples") != expected_k
        or provenance.get("output_sha256") != sha256_file(predictions_path)
        or _journal_content_record(provenance.get("generation_journal") or {})
        != _journal_content_record(journal_record(journal_path))
        or provenance.get("no_frontier_api") is not True
        or provenance.get("tests_exposed_to_model") is not False
        or not journal
        or journal[0].get("event") != "header"
        or journal[-1].get("event") != "complete"
        or score.get("schema") != SCORE_SCHEMA
        or score.get("tasks") != expected_tasks
        or score.get("k") != expected_k
        or score.get("predictions", {}).get("sha256") != sha256_file(predictions_path)
        or len(set(score_ids)) != expected_tasks
        or set(score_ids) != set(task_ids)
        or len(score.get("candidate_results") or [])
        != expected_tasks * expected_k
    ):
        raise ValueError(f"{label}: sealed evaluation contract failed")
    terminals = journal[1:-1]
    coordinates = [
        (
            terminal.get("task_id"),
            tuple(
                (candidate.get("sample_index"), candidate.get("seed"))
                for candidate in terminal.get("candidates") or []
            ),
        )
        for terminal in terminals
    ]
    if len(coordinates) != expected_tasks:
        raise ValueError(f"{label}: journal terminal count differs")
    return {
        "label": label,
        "predictions_path": predictions_path,
        "score_path": score_path,
        "predictions": predictions,
        "score": score,
        "provenance": provenance,
        "journal": journal,
        "task_ids": task_ids,
        "coordinates": coordinates,
    }


def _metric(score: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = score.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"score lacks metric {name}")
    return {"count": int(value["count"]), "rate": float(value["rate"])}


def _paired(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    task_subset: Sequence[str] | None = None,
) -> dict[str, Any]:
    left_rows = {row["task_id"]: row for row in left["task_results"]}
    right_rows = {row["task_id"]: row for row in right["task_results"]}
    if set(left_rows) != set(right_rows):
        raise ValueError("paired score task sets differ")
    selected = sorted(set(task_subset) if task_subset is not None else set(left_rows))
    if not set(selected) <= set(left_rows):
        raise ValueError("paired subset is outside the score task set")
    result: dict[str, Any] = {"tasks": len(selected)}
    for metric in ("pass_at_1", "pass_at_k", "compile_at_k"):
        gain = loss = tie = 0
        left_count = right_count = 0
        for task_id in selected:
            lhs = bool(left_rows[task_id][metric])
            rhs = bool(right_rows[task_id][metric])
            left_count += lhs
            right_count += rhs
            gain += lhs and not rhs
            loss += rhs and not lhs
            tie += lhs == rhs
        result[metric] = {
            "left_only": gain,
            "right_only": loss,
            "equal": tie,
            "discordant": gain + loss,
            "left_count": left_count,
            "right_count": right_count,
        }
    return result


def _candidate_metrics(score: Mapping[str, Any]) -> dict[str, Any]:
    rows = score.get("candidate_results") or []
    total = len(rows)
    if total <= 0:
        raise ValueError("score has no candidate results")
    compiled = sum(bool(row.get("compiled")) for row in rows)
    passed = sum(bool(row.get("passed")) for row in rows)
    arity = type_failures = union = 0
    for row in rows:
        diagnostic = str(row.get("diagnostic") or "")
        arity_match = _ARITY_DIAGNOSTIC.search(diagnostic) is not None
        type_match = _TYPE_DIAGNOSTIC.search(diagnostic) is not None
        arity += arity_match
        type_failures += type_match
        union += arity_match or type_match
    return {
        "candidates": total,
        "compiled": {"count": compiled, "rate": compiled / total},
        "passed": {"count": passed, "rate": passed / total},
        "diagnostic_candidate_counts": {
            "arity_or_parameter_shape": arity,
            "type": type_failures,
            "arity_or_type_union": union,
        },
        "diagnostic_taxonomy": {
            "arity_regex": _ARITY_DIAGNOSTIC.pattern,
            "type_regex": _TYPE_DIAGNOSTIC.pattern,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    gold_path = Path(args.gold_score).expanduser().resolve()
    gold = _read(gold_path, "Rank-0 gold score")
    if (
        gold.get("schema") != SCORE_SCHEMA
        or gold.get("tasks") != args.expected_tasks
        or gold.get("k") != 1
        or (gold.get("pass_at_1") or {}).get("count") != args.expected_tasks
        or (gold.get("pass_at_k") or {}).get("count") != args.expected_tasks
        or (gold.get("compile_at_k") or {}).get("count") != args.expected_tasks
    ):
        raise ValueError("Rank-0 gold round-trip is incomplete")
    baseline_specs = [_parse_spec(value, numeric=True) for value in args.baseline]
    seeds = [int(label) for label, _, _ in baseline_specs]
    if sorted(seeds) != [42, 43, 44, 45, 46] or len(set(seeds)) != 5:
        raise ValueError("baseline replicates must be exactly seeds 42..46")
    baselines = {
        int(label): _load_arm(
            label=f"baseline_seed{label}",
            predictions_path=predictions,
            score_path=score,
            expected_tasks=args.expected_tasks,
            expected_k=args.k,
            expected_provenance_schema=BASE_PROVENANCE_SCHEMA,
        )
        for label, predictions, score in baseline_specs
    }
    ablation_specs = [_parse_spec(value, numeric=False) for value in args.ablation]
    expected_views = {
        "semantic_body_swap",
        "constants_stripped",
        "typed_opaque_contract",
    }
    if {label for label, _, _ in ablation_specs} != expected_views:
        raise ValueError("all three declared input views are required")
    ablations = {
        label: _load_arm(
            label=label,
            predictions_path=predictions,
            score_path=score,
            expected_tasks=args.expected_tasks,
            expected_k=args.k,
            expected_provenance_schema=ABLATION_PROVENANCE_SCHEMA,
        )
        for label, predictions, score in ablation_specs
    }

    reference = baselines[42]
    reference_model = canonical_sha256(reference["provenance"]["model"])
    reference_tasks = reference["task_ids"]
    reference_score_contract = (
        reference["score"]["evaluation"]["sha256"],
        reference["score"]["k"],
        reference["score"]["timeout"],
        reference["score"]["stability_runs"],
    )
    for arm in [*baselines.values(), *ablations.values()]:
        score_contract = (
            arm["score"]["evaluation"]["sha256"],
            arm["score"]["k"],
            arm["score"]["timeout"],
            arm["score"]["stability_runs"],
        )
        if (
            arm["task_ids"] != reference_tasks
            or canonical_sha256(arm["provenance"]["model"]) != reference_model
            or score_contract != reference_score_contract
        ):
            raise ValueError(f"{arm['label']}: pairing/model/scoring contract differs")
    for view, arm in ablations.items():
        provenance = arm["provenance"]
        sampling = dict(provenance["sampling"])
        baseline_sampling = dict(reference["provenance"]["sampling"])
        if (
            provenance.get("input_view") != view
            or sampling != baseline_sampling
            or arm["coordinates"] != reference["coordinates"]
            or provenance.get("full_gold_targets_exposed_to_model") is not False
        ):
            raise ValueError(f"{view}: seed-slot pairing or privacy contract differs")

    constants_summary = ablations["constants_stripped"]["provenance"]["heldout"][
        "input_view"
    ]["summary"]
    changed_task_ids = list(constants_summary.get("changed_task_ids") or [])
    unchanged_task_ids = list(constants_summary.get("unchanged_task_ids") or [])
    if (
        len(changed_task_ids) != int(constants_summary.get("changed_rows", -1))
        or len(unchanged_task_ids)
        != int(constants_summary.get("unchanged_no_literal_rows", -1))
        or set(changed_task_ids) & set(unchanged_task_ids)
        or set(changed_task_ids) | set(unchanged_task_ids) != set(reference_tasks)
        or constants_summary.get("changed_task_ids_sha256")
        != canonical_sha256(changed_task_ids)
        or constants_summary.get("unchanged_task_ids_sha256")
        != canonical_sha256(unchanged_task_ids)
    ):
        raise ValueError("constants-stripped changed/unchanged partition is invalid")
    baseline_predictions = {
        row["id"]: row["predictions"] for row in reference["predictions"]
    }
    constants_predictions = {
        row["id"]: row["predictions"]
        for row in ablations["constants_stripped"]["predictions"]
    }
    nonidentical_unchanged = [
        task_id
        for task_id in unchanged_task_ids
        if constants_predictions[task_id] != baseline_predictions[task_id]
    ]
    if nonidentical_unchanged:
        raise ValueError(
            "constants-stripped byte-identical sources produced different paired "
            f"predictions: {nonidentical_unchanged[:5]}"
        )

    metrics = ("pass_at_1", "pass_at_k", "compile_at_k")
    baseline_counts = {
        metric: [_metric(baselines[seed]["score"], metric)["count"] for seed in seeds]
        for metric in metrics
    }
    task_solve_frequency = {task_id: 0 for task_id in reference_tasks}
    for arm in baselines.values():
        for row in arm["score"]["task_results"]:
            task_solve_frequency[row["task_id"]] += int(bool(row["pass_at_k"]))
    frequency_histogram = {
        str(frequency): sum(
            count == frequency for count in task_solve_frequency.values()
        )
        for frequency in range(6)
    }
    baseline_summary = {
        "seeds": seeds,
        "arms": {
            str(seed): {
                **{
                    metric: _metric(baselines[seed]["score"], metric)
                    for metric in metrics
                },
                "candidate_level": _candidate_metrics(baselines[seed]["score"]),
            }
            for seed in seeds
        },
        "count_distribution": {
            metric: {
                "minimum": min(values),
                "maximum": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            }
            for metric, values in baseline_counts.items()
        },
        "pass_at_k_task_frequency_histogram": frequency_histogram,
        "unique_tasks_solved_across_five_seeds": sum(
            value > 0 for value in task_solve_frequency.values()
        ),
        "prior_six_to_seven_solve_band_spanned": (
            min(baseline_counts["pass_at_k"]) <= 6
            and max(baseline_counts["pass_at_k"]) >= 7
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "heldout_tasks": args.expected_tasks,
        "k": args.k,
        "checkpoint_sha256": reference_model,
        "exact_task_order_model_and_scoring_pairing_validated": True,
        "ablation_seed_and_sample_coordinates_paired_to_seed42": True,
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "rank0_gold_roundtrip": {
            "score": str(gold_path),
            "score_sha256": sha256_file(gold_path),
            "tasks": args.expected_tasks,
            "pass_at_1": args.expected_tasks,
            "pass_at_k": args.expected_tasks,
            "compile_at_k": args.expected_tasks,
        },
        "baseline_replicates": baseline_summary,
        "input_ablations": {
            view: {
                "metrics": {
                    metric: _metric(arm["score"], metric) for metric in metrics
                },
                "paired_vs_baseline_seed42": _paired(
                    arm["score"], reference["score"]
                ),
                "candidate_level": _candidate_metrics(arm["score"]),
                "input_view_provenance": arm["provenance"]["heldout"][
                    "input_view"
                ],
                "predictions_sha256": sha256_file(arm["predictions_path"]),
                "score_sha256": sha256_file(arm["score_path"]),
            }
            for view, arm in ablations.items()
        },
        "interpretation_gate": {
            "prior_small_arm_differences_uninterpretable_if_six_to_seven_band_spanned": True,
            "triggered": baseline_summary["prior_six_to_seven_solve_band_spanned"],
        },
    }
    report["input_ablations"]["constants_stripped"].update(
        {
            "changed_source_tasks": len(changed_task_ids),
            "unchanged_source_tasks": len(unchanged_task_ids),
            "unchanged_source_predictions_byte_identical_to_baseline": True,
            "paired_changed_tasks_only": _paired(
                ablations["constants_stripped"]["score"],
                reference["score"],
                changed_task_ids,
            ),
        }
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--ablation", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gold_score", required=True)
    parser.add_argument("--expected_tasks", type=int, default=175)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.expected_tasks <= 0 or args.k <= 0:
        parser.error("expected_tasks and k must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
