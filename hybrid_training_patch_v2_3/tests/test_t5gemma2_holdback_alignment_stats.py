from __future__ import annotations

import math

import pytest

from scripts.evaluation.t5gemma2_holdback_alignment_stats import (
    AlignmentDecisionThresholds,
    HoldbackAlignmentTask,
    metric_decision,
    summarize_holdback_alignment,
    task_alignment_metrics,
    task_bootstrap_mean_interval,
    task_equal_pairwise_rank_accuracy,
    tie_averaged_argmax_uplift,
)


def _task(task_id: str, visible: list[float], holdback: list[float]):
    return HoldbackAlignmentTask(
        task_id=task_id,
        visible_scores=tuple(visible),
        holdback_scores=tuple(holdback),
    )


def test_argmax_uplift_averages_visible_ties() -> None:
    uplift = tie_averaged_argmax_uplift(
        [2.0, 2.0, 0.0],
        [1.0, 0.5, 0.0],
    )
    assert uplift == pytest.approx(0.25)

    record = task_alignment_metrics(
        _task("tie", [2.0, 2.0, 0.0], [1.0, 0.5, 0.0])
    )
    assert record["argmax_tie_count"] == 2
    assert record["tie_averaged_argmax_holdback_mean"] == pytest.approx(0.75)
    assert record["uniform_holdback_mean"] == pytest.approx(0.5)


def test_pairwise_accuracy_is_task_equal_not_pair_pooled() -> None:
    tasks = [
        _task("one-pair-correct", [0.0, 1.0], [0.0, 1.0]),
        _task(
            "six-pairs-wrong",
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ),
    ]
    result = task_equal_pairwise_rank_accuracy(tasks)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["eligible_tasks"] == 2
    assert result["informative_pairs"] == 7
    assert result["estimate"] != pytest.approx(1.0 / 7.0)


def test_pairwise_visible_ties_get_half_credit_and_holdback_ties_are_ignored() -> None:
    tied_prediction = task_alignment_metrics(
        _task("predicted-tie", [0.0, 0.0], [0.0, 1.0])
    )
    assert tied_prediction["pairwise_rank_accuracy"] == pytest.approx(0.5)
    assert tied_prediction["pairwise_informative_pairs"] == 1

    true_tie = task_alignment_metrics(
        _task("true-tie", [0.0, 1.0], [1.0, 1.0])
    )
    assert true_tie["pairwise_rank_accuracy"] is None
    assert true_tie["pairwise_informative_pairs"] == 0


def test_task_bootstrap_is_deterministic_and_defaults_to_10k() -> None:
    first = task_bootstrap_mean_interval([0.0, 0.5, 1.0], seed=17)
    second = task_bootstrap_mean_interval([0.0, 0.5, 1.0], seed=17)
    assert first == second
    assert first["replicates"] == 10_000
    assert first["method"] == "task_bootstrap_percentile"
    assert 0.0 <= first["lower"] <= first["upper"] <= 1.0


def test_summary_go_and_stop_use_intersection_rule() -> None:
    thresholds = AlignmentDecisionThresholds(
        argmax_uplift_target=0.4,
        argmax_uplift_minimum=0.3,
        pairwise_accuracy_target=0.8,
        pairwise_accuracy_minimum=0.6,
    )
    aligned = [
        _task(f"aligned-{index}", [0.0, 1.0], [0.0, 1.0])
        for index in range(8)
    ]
    go = summarize_holdback_alignment(
        aligned,
        thresholds,
        bootstrap_replicates=100,
    )
    assert go["tie_averaged_argmax_uplift"]["estimate"] == pytest.approx(0.5)
    assert go["task_equal_pairwise_rank_accuracy"]["estimate"] == 1.0
    assert go["overall_decision"] == "GO"

    reversed_tasks = [
        _task(f"reversed-{index}", [0.0, 1.0], [1.0, 0.0])
        for index in range(8)
    ]
    stop = summarize_holdback_alignment(
        reversed_tasks,
        thresholds,
        bootstrap_replicates=100,
    )
    assert stop["overall_decision"] == "STOP"
    assert stop["tie_averaged_argmax_uplift"]["decision"] == "STOP"
    assert stop["task_equal_pairwise_rank_accuracy"]["decision"] == "STOP"


def test_metric_decision_exposes_hold_region() -> None:
    assert (
        metric_decision(
            0.20,
            {"lower": 0.05, "upper": 0.30},
            target=0.20,
            minimum=0.10,
        )
        == "HOLD"
    )
    assert (
        metric_decision(
            0.25,
            {"lower": 0.12, "upper": 0.35},
            target=0.20,
            minimum=0.10,
        )
        == "GO"
    )
    assert (
        metric_decision(
            0.08,
            {"lower": 0.01, "upper": 0.19},
            target=0.20,
            minimum=0.10,
        )
        == "STOP"
    )


def test_input_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="equal length"):
        tie_averaged_argmax_uplift([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        tie_averaged_argmax_uplift([math.nan], [0.0])
    with pytest.raises(ValueError, match="unique"):
        summarize_holdback_alignment(
            [
                _task("duplicate", [0.0, 1.0], [0.0, 1.0]),
                _task("duplicate", [0.0, 1.0], [0.0, 1.0]),
            ],
            AlignmentDecisionThresholds(0.1, 0.0, 0.6, 0.5),
            bootstrap_replicates=10,
        )
    with pytest.raises(ValueError, match="holdback-informative"):
        summarize_holdback_alignment(
            [_task("no-order", [0.0, 1.0], [1.0, 1.0])],
            AlignmentDecisionThresholds(0.1, 0.0, 0.6, 0.5),
            bootstrap_replicates=10,
        )

