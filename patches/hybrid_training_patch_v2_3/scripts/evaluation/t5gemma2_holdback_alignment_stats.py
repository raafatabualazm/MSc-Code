#!/usr/bin/env python3
"""Pure task-level statistics for visible-reward/holdback alignment.

The helpers in this module deliberately perform no file or process I/O.  They
measure whether a visible reward ranks candidates in a way that transfers to a
separate holdback score while preserving the task, rather than candidate pair,
as the unit of analysis.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA = "t5gemma2-holdback-alignment-stats-v1"
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class HoldbackAlignmentTask:
    """One task's candidate-level visible and holdback scores."""

    task_id: str
    visible_scores: tuple[float, ...]
    holdback_scores: tuple[float, ...]


@dataclass(frozen=True)
class AlignmentDecisionThresholds:
    """Pre-registered target and minimum-useful values for both metrics."""

    argmax_uplift_target: float
    argmax_uplift_minimum: float
    pairwise_accuracy_target: float
    pairwise_accuracy_minimum: float


def _finite_scores(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _validate_score_pair(
    visible_scores: Sequence[float],
    holdback_scores: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    visible = _finite_scores(visible_scores, label="visible_scores")
    holdback = _finite_scores(holdback_scores, label="holdback_scores")
    if len(visible) != len(holdback):
        raise ValueError("visible_scores and holdback_scores must have equal length")
    return visible, holdback


def _validate_tie_tolerance(tie_tolerance: float) -> float:
    tolerance = float(tie_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    return tolerance


def tie_averaged_argmax_uplift(
    visible_scores: Sequence[float],
    holdback_scores: Sequence[float],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> float:
    """Return holdback uplift of visible argmax selection over uniform choice.

    All candidates tied for the maximum visible score receive equal selection
    weight.  This avoids letting candidate order act as an undeclared tie
    breaker.  The baseline is the mean holdback score over every candidate.
    """

    visible, holdback = _validate_score_pair(visible_scores, holdback_scores)
    tolerance = _validate_tie_tolerance(tie_tolerance)
    maximum = max(visible)
    selected = [
        holdback[index]
        for index, score in enumerate(visible)
        if abs(score - maximum) <= tolerance
    ]
    if not selected:  # Defensive: max itself must always be selected.
        raise AssertionError("visible argmax selection is unexpectedly empty")
    selected_mean = sum(selected) / len(selected)
    baseline_mean = sum(holdback) / len(holdback)
    return selected_mean - baseline_mean


def _pairwise_rank_record(
    visible_scores: Sequence[float],
    holdback_scores: Sequence[float],
    *,
    tie_tolerance: float,
) -> tuple[float | None, int, float]:
    visible, holdback = _validate_score_pair(visible_scores, holdback_scores)
    tolerance = _validate_tie_tolerance(tie_tolerance)
    credit = 0.0
    informative_pairs = 0
    for left in range(len(visible)):
        for right in range(left + 1, len(visible)):
            holdback_delta = holdback[left] - holdback[right]
            if abs(holdback_delta) <= tolerance:
                continue
            informative_pairs += 1
            visible_delta = visible[left] - visible[right]
            if abs(visible_delta) <= tolerance:
                credit += 0.5
            elif (visible_delta > 0.0) == (holdback_delta > 0.0):
                credit += 1.0
    if informative_pairs == 0:
        return None, 0, 0.0
    return credit / informative_pairs, informative_pairs, credit


def task_alignment_metrics(
    task: HoldbackAlignmentTask,
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Compute the two alignment metrics for one task."""

    if not isinstance(task.task_id, str) or not task.task_id:
        raise ValueError("task_id must be a non-empty string")
    visible, holdback = _validate_score_pair(
        task.visible_scores, task.holdback_scores
    )
    tolerance = _validate_tie_tolerance(tie_tolerance)
    maximum = max(visible)
    selected_indices = [
        index
        for index, score in enumerate(visible)
        if abs(score - maximum) <= tolerance
    ]
    baseline = sum(holdback) / len(holdback)
    selected = sum(holdback[index] for index in selected_indices) / len(
        selected_indices
    )
    pairwise, informative_pairs, pairwise_credit = _pairwise_rank_record(
        visible,
        holdback,
        tie_tolerance=tolerance,
    )
    return {
        "task_id": task.task_id,
        "candidates": len(visible),
        "argmax_tie_count": len(selected_indices),
        "uniform_holdback_mean": baseline,
        "tie_averaged_argmax_holdback_mean": selected,
        "tie_averaged_argmax_uplift": selected - baseline,
        "pairwise_rank_accuracy": pairwise,
        "pairwise_informative_pairs": informative_pairs,
        "pairwise_credit": pairwise_credit,
    }


def task_equal_pairwise_rank_accuracy(
    tasks: Sequence[HoldbackAlignmentTask],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Average per-task pairwise accuracy over tasks with informative pairs.

    Holdback-tied pairs are excluded because they have no true ordering.
    Visible-score ties earn half credit.  Each eligible task receives equal
    aggregate weight regardless of how many informative candidate pairs it
    contains.
    """

    records = _validated_task_records(tasks, tie_tolerance=tie_tolerance)
    eligible = [
        record
        for record in records
        if record["pairwise_rank_accuracy"] is not None
    ]
    if not eligible:
        raise ValueError("no task contains a holdback-informative candidate pair")
    return {
        "estimate": sum(
            float(record["pairwise_rank_accuracy"]) for record in eligible
        )
        / len(eligible),
        "eligible_tasks": len(eligible),
        "ineligible_tasks": len(records) - len(eligible),
        "informative_pairs": sum(
            int(record["pairwise_informative_pairs"]) for record in eligible
        ),
    }


def _validated_task_records(
    tasks: Sequence[HoldbackAlignmentTask],
    *,
    tie_tolerance: float,
) -> list[dict[str, Any]]:
    if not tasks:
        raise ValueError("holdback alignment needs at least one task")
    records = [
        task_alignment_metrics(task, tie_tolerance=tie_tolerance)
        for task in tasks
    ]
    task_ids = [str(record["task_id"]) for record in records]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("holdback alignment task_ids must be unique")
    return records


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile input is invalid")
    coordinate = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(coordinate))
    upper = int(math.ceil(coordinate))
    if lower == upper:
        return float(sorted_values[lower])
    weight = coordinate - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(
        sorted_values[upper]
    ) * weight


def task_bootstrap_mean_interval(
    task_values: Sequence[float],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Return a deterministic percentile interval from task resampling."""

    values = _finite_scores(task_values, label="task_values")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    randomizer = random.Random(seed)
    count = len(values)
    samples: list[float] = []
    for _ in range(replicates):
        samples.append(
            sum(values[randomizer.randrange(count)] for _draw in range(count))
            / count
        )
    samples.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "confidence_level": confidence,
        "method": "task_bootstrap_percentile",
        "replicates": replicates,
        "seed": seed,
        "lower": _percentile(samples, tail),
        "upper": _percentile(samples, 1.0 - tail),
    }


def metric_decision(
    estimate: float,
    interval: Mapping[str, float],
    *,
    target: float,
    minimum: float,
) -> str:
    """Apply the preregisterable GO/HOLD/STOP interval rule."""

    values = {
        "estimate": float(estimate),
        "lower": float(interval["lower"]),
        "upper": float(interval["upper"]),
        "target": float(target),
        "minimum": float(minimum),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("decision inputs must be finite")
    if values["lower"] > values["upper"]:
        raise ValueError("decision interval lower bound exceeds upper bound")
    if values["minimum"] > values["target"]:
        raise ValueError("decision minimum must not exceed target")
    if (
        values["estimate"] >= values["target"]
        and values["lower"] >= values["minimum"]
    ):
        return "GO"
    if values["upper"] < values["target"]:
        return "STOP"
    return "HOLD"


def _validate_thresholds(thresholds: AlignmentDecisionThresholds) -> None:
    metric_decision(
        thresholds.argmax_uplift_target,
        {
            "lower": thresholds.argmax_uplift_minimum,
            "upper": thresholds.argmax_uplift_target,
        },
        target=thresholds.argmax_uplift_target,
        minimum=thresholds.argmax_uplift_minimum,
    )
    if not 0.0 <= thresholds.pairwise_accuracy_minimum <= 1.0:
        raise ValueError("pairwise accuracy minimum must lie in [0, 1]")
    if not 0.0 <= thresholds.pairwise_accuracy_target <= 1.0:
        raise ValueError("pairwise accuracy target must lie in [0, 1]")
    if thresholds.pairwise_accuracy_minimum > thresholds.pairwise_accuracy_target:
        raise ValueError("pairwise accuracy minimum must not exceed target")


def summarize_holdback_alignment(
    tasks: Sequence[HoldbackAlignmentTask],
    thresholds: AlignmentDecisionThresholds,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Summarize alignment and apply an intersection GO/any-STOP rule."""

    _validate_thresholds(thresholds)
    records = _validated_task_records(tasks, tie_tolerance=tie_tolerance)
    uplift_values = [
        float(record["tie_averaged_argmax_uplift"]) for record in records
    ]
    pairwise_records = [
        record
        for record in records
        if record["pairwise_rank_accuracy"] is not None
    ]
    if not pairwise_records:
        raise ValueError("no task contains a holdback-informative candidate pair")
    pairwise_values = [
        float(record["pairwise_rank_accuracy"])
        for record in pairwise_records
    ]
    uplift_estimate = sum(uplift_values) / len(uplift_values)
    pairwise_estimate = sum(pairwise_values) / len(pairwise_values)
    uplift_interval = task_bootstrap_mean_interval(
        uplift_values,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    pairwise_interval = task_bootstrap_mean_interval(
        pairwise_values,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    uplift_decision = metric_decision(
        uplift_estimate,
        uplift_interval,
        target=thresholds.argmax_uplift_target,
        minimum=thresholds.argmax_uplift_minimum,
    )
    pairwise_decision = metric_decision(
        pairwise_estimate,
        pairwise_interval,
        target=thresholds.pairwise_accuracy_target,
        minimum=thresholds.pairwise_accuracy_minimum,
    )
    decisions = (uplift_decision, pairwise_decision)
    if "STOP" in decisions:
        overall_decision = "STOP"
    elif decisions == ("GO", "GO"):
        overall_decision = "GO"
    else:
        overall_decision = "HOLD"

    return {
        "schema": SUMMARY_SCHEMA,
        "tasks": len(records),
        "tie_tolerance": float(tie_tolerance),
        "bootstrap": {
            "method": "task_bootstrap_percentile",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "tie_averaged_argmax_uplift": {
            "estimate": uplift_estimate,
            "interval": uplift_interval,
            "tasks": len(records),
            "tasks_with_argmax_ties": sum(
                int(record["argmax_tie_count"]) > 1 for record in records
            ),
            "target": thresholds.argmax_uplift_target,
            "minimum": thresholds.argmax_uplift_minimum,
            "decision": uplift_decision,
        },
        "task_equal_pairwise_rank_accuracy": {
            "estimate": pairwise_estimate,
            "interval": pairwise_interval,
            "eligible_tasks": len(pairwise_records),
            "ineligible_tasks": len(records) - len(pairwise_records),
            "informative_pairs": sum(
                int(record["pairwise_informative_pairs"])
                for record in pairwise_records
            ),
            "target": thresholds.pairwise_accuracy_target,
            "minimum": thresholds.pairwise_accuracy_minimum,
            "decision": pairwise_decision,
        },
        "overall_decision": overall_decision,
        "decision_rule": (
            "per metric GO iff estimate>=target and lower95>=minimum; "
            "STOP iff upper95<target; overall GO=intersection and "
            "STOP=either; otherwise HOLD"
        ),
    }


__all__ = [
    "AlignmentDecisionThresholds",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_TIE_TOLERANCE",
    "HoldbackAlignmentTask",
    "SUMMARY_SCHEMA",
    "metric_decision",
    "summarize_holdback_alignment",
    "task_alignment_metrics",
    "task_bootstrap_mean_interval",
    "task_equal_pairwise_rank_accuracy",
    "tie_averaged_argmax_uplift",
]
