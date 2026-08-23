#!/usr/bin/env python3
"""Compare the sealed stage-1 and Kimi pass-2 T5Gemma held-out arms.

The historical two-epoch arm is included only to derive the five pass@10
regressions observed after stage 1.  Regression identities and outcomes are
loaded after generation/scoring and are never used to select prompts, tasks,
samples, or model-visible feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.evaluation.durable_evaluation_journal import (
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)


EXPECTED_ROWS = 175
EXPECTED_SAMPLES = 10
EXPECTED_REGRESSIONS = 5
EXPECTED_SAMPLING = {
    "num_samples": 10,
    "generation_batch_size": 10,
    "max_source_tokens": 32768,
    "max_new_tokens": 4096,
    "temperature": 0.8,
    "top_k": 0,
    "top_p": 0.95,
    "seed": 42,
    "seed_policy": "seed+task_index*100003+batch_start",
    "decoder_prefix_is_not_output": True,
    "fabricated_eos": False,
    "sampled_eos_retained": True,
}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_metric(metric: Any, label: str) -> None:
    if (
        not isinstance(metric, Mapping)
        or not isinstance(metric.get("count"), int)
        or isinstance(metric.get("count"), bool)
        or not 0 <= int(metric["count"]) <= EXPECTED_ROWS
        or not isinstance(metric.get("rate"), (int, float))
        or isinstance(metric.get("rate"), bool)
        or abs(float(metric["rate"]) - int(metric["count"]) / EXPECTED_ROWS)
        > 1e-15
    ):
        raise ValueError(f"{label} metric contract failed")


def _validate_arm(
    *,
    label: str,
    prediction_path: Path,
    score_path: Path,
) -> dict[str, Any]:
    provenance_path = Path(f"{prediction_path}.provenance.json")
    journal_path = Path(f"{prediction_path}.generation.journal.jsonl")
    provenance = _read_object(provenance_path, f"{label} provenance")
    score = _read_object(score_path, f"{label} score")
    try:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} predictions are not readable JSON") from exc
    if not isinstance(prediction, list):
        raise ValueError(f"{label} predictions are not a JSON list")
    journal = load_journal(journal_path)
    if (
        provenance.get("schema") != "direct-compact-inference-v1"
        or provenance.get("architecture")
        != "native_t5gemma2_encoder_decoder"
        or provenance.get("arm") != "sft"
        or provenance.get("num_rows") != EXPECTED_ROWS
        or provenance.get("num_samples") != EXPECTED_SAMPLES
        or provenance.get("sampling") != EXPECTED_SAMPLING
        or provenance.get("output_sha256") != sha256_file(prediction_path)
        or provenance.get("generation_journal") != journal_record(journal_path)
        or provenance.get("no_frontier_api") is not True
        or provenance.get("tests_exposed_to_model") is not False
        or provenance.get("targets_exposed_to_model") is not False
        or len(journal) != EXPECTED_ROWS + 2
        or journal[0].get("event") != "header"
        or journal[-1].get("event") != "complete"
        or any(item.get("event") != "task_terminal" for item in journal[1:-1])
        or journal[-1].get("rows") != EXPECTED_ROWS
        or journal[0].get("contract", {}).get("source_truncation") is not False
        or journal[0].get("contract", {}).get("runtime", {}).get(
            "attn_implementation"
        )
        != "sdpa"
        or journal[0].get("contract", {}).get("runtime", {}).get("bf16")
        is not True
        or score.get("schema") != "direct-compact-attested-passk-v1"
        or score.get("tasks") != EXPECTED_ROWS
        or score.get("k") != EXPECTED_SAMPLES
        or score.get("timeout") != 30
        or score.get("stability_runs") != 2
        or score.get("predictions", {}).get("sha256")
        != sha256_file(prediction_path)
        or len(prediction) != EXPECTED_ROWS
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("id"), str)
            or len(row.get("predictions") or []) != EXPECTED_SAMPLES
            for row in prediction
        )
    ):
        raise ValueError(f"{label}: sealed evaluation contract failed")
    for metric_name in ("pass_at_1", "pass_at_k", "compile_at_k"):
        _require_metric(score.get(metric_name), f"{label} {metric_name}")
    task_results = score.get("task_results")
    candidate_results = score.get("candidate_results")
    if (
        not isinstance(task_results, list)
        or len(task_results) != EXPECTED_ROWS
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("task_id"), str)
            or not isinstance(row.get("pass_at_1"), bool)
            or not isinstance(row.get("pass_at_k"), bool)
            or not isinstance(row.get("compile_at_k"), bool)
            or not isinstance(row.get("passing_samples"), int)
            or not 0 <= int(row["passing_samples"]) <= EXPECTED_SAMPLES
            for row in task_results
        )
        or not isinstance(candidate_results, list)
        or len(candidate_results) != EXPECTED_ROWS * EXPECTED_SAMPLES
    ):
        raise ValueError(f"{label}: score task/candidate contract failed")
    task_order = [str(row["id"]) for row in prediction]
    score_order = [str(row["task_id"]) for row in task_results]
    terminal_order = [str(row["task_id"]) for row in journal[1:-1]]
    if (
        len(set(task_order)) != EXPECTED_ROWS
        or len(set(score_order)) != EXPECTED_ROWS
        or set(task_order) != set(score_order)
        or task_order != terminal_order
    ):
        raise ValueError(f"{label}: task identities or generation order differ")
    return {
        "label": label,
        "prediction_path": prediction_path,
        "score_path": score_path,
        "provenance": provenance,
        "journal": journal,
        "score": score,
        "predictions": prediction,
        "task_order": task_order,
        "by_task": {str(row["task_id"]): row for row in task_results},
    }


def _slot_coordinates(arm: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            terminal["task_id"],
            terminal["source_sha256"],
            tuple(
                (candidate["sample_index"], candidate["seed"])
                for candidate in terminal["candidates"]
            ),
        )
        for terminal in arm["journal"][1:-1]
    ]


def _score_contract(arm: Mapping[str, Any]) -> tuple[Any, ...]:
    score = arm["score"]
    return (
        score["evaluation"]["sha256"],
        score["evaluator"]["sha256"],
        score["k"],
        score["timeout"],
        score["stability_runs"],
    )


def _metric_block(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: score[key]
        for key in ("pass_at_1", "pass_at_k", "compile_at_k")
    }


def _paired_metrics(
    *,
    task_order: Sequence[str],
    before_by_task: Mapping[str, Mapping[str, Any]],
    after_by_task: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for metric in ("pass_at_1", "pass_at_k", "compile_at_k"):
        gains = losses = ties = 0
        for task_id in task_order:
            before = bool(before_by_task[task_id][metric])
            after = bool(after_by_task[task_id][metric])
            gains += after and not before
            losses += before and not after
            ties += before == after
        paired[metric] = {
            "post_above_pre_tasks": gains,
            "pre_above_post_tasks": losses,
            "equal_tasks": ties,
        }
    return paired


def build_historical_regression_block(
    *,
    task_order: Sequence[str],
    historical_by_task: Mapping[str, Mapping[str, Any]],
    stage1_by_task: Mapping[str, Mapping[str, Any]],
    pass2_by_task: Mapping[str, Mapping[str, Any]],
    expected_regressions: int = EXPECTED_REGRESSIONS,
) -> dict[str, Any]:
    """Derive the historical pass@10 losses and classify pass-2 recovery."""

    regressions = [
        task_id
        for task_id in task_order
        if bool(historical_by_task[task_id]["pass_at_k"])
        and not bool(stage1_by_task[task_id]["pass_at_k"])
    ]
    if len(regressions) != expected_regressions:
        raise ValueError(
            "historical stage1-vs-two-epoch pass@10 regression count differs: "
            f"expected={expected_regressions}, observed={len(regressions)}"
        )
    tasks = [
        {
            "task_id": task_id,
            "status": (
                "recovered_by_kimi_pass2"
                if bool(pass2_by_task[task_id]["pass_at_k"])
                else "still_regressed"
            ),
            "kimi_pass2_pass_at_k": bool(pass2_by_task[task_id]["pass_at_k"]),
        }
        for task_id in regressions
    ]
    recovered = sum(
        row["status"] == "recovered_by_kimi_pass2" for row in tasks
    )
    return {
        "metric": "pass_at_k",
        "historical_reference": "two_epoch_sft",
        "pre_arm": "stage1_mixed_rs_sft",
        "post_arm": "kimi_pass2",
        "derived_after_post_generation_and_scoring": True,
        "used_for_task_selection": False,
        "used_for_sampling": False,
        "used_as_model_input_or_feedback": False,
        "expected_and_observed_tasks": expected_regressions,
        "ordered_task_ids_sha256": _canonical_sha256(regressions),
        "recovered_tasks": recovered,
        "still_regressed_tasks": expected_regressions - recovered,
        "tasks": tasks,
    }


def compare(
    *,
    historical_predictions: Path,
    historical_score: Path,
    pre_predictions: Path,
    pre_score: Path,
    post_predictions: Path,
    post_score: Path,
    post_compat: Path,
    output: Path,
) -> dict[str, Any]:
    historical = _validate_arm(
        label="two_epoch_sft",
        prediction_path=historical_predictions,
        score_path=historical_score,
    )
    pre = _validate_arm(
        label="stage1_mixed_rs_sft",
        prediction_path=pre_predictions,
        score_path=pre_score,
    )
    post = _validate_arm(
        label="kimi_pass2",
        prediction_path=post_predictions,
        score_path=post_score,
    )
    arms = (historical, pre, post)
    script_hashes = [
        arm["journal"][0]["contract"]["script_sha256"] for arm in arms
    ]
    compat = _read_object(post_compat, "post checkpoint-loader compatibility")
    compat_wrapper = Path(str(compat.get("wrapper_path") or ""))
    if (
        compat.get("schema") != "t5gemma2-mixed-passk-loader-compat-v1"
        or compat.get("scope") != "checkpoint_contract_loader_only"
        or compat.get("sampling_code_changed") is not False
        or compat.get("generation_code_changed") is not False
        or compat.get("scoring_code_changed") is not False
        or compat.get("core_inference_sha256") != script_hashes[2]
        or not compat_wrapper.is_file()
        or compat.get("wrapper_sha256") != sha256_file(compat_wrapper)
        or compat.get("checkpoint_run_contract_sha256")
        != post["provenance"]["model"]["adapter"]["run_contract_sha256"]
    ):
        raise ValueError("post checkpoint-loader compatibility binding failed")

    heldout_contracts = [arm["provenance"]["heldout"] for arm in arms]
    score_orders = [
        [row["task_id"] for row in arm["score"]["task_results"]] for arm in arms
    ]
    score_contracts = [_score_contract(arm) for arm in arms]
    tokenizers = [
        arm["provenance"]["model"]["tokenizer_sha256"] for arm in arms
    ]
    if not (
        heldout_contracts[0] == heldout_contracts[1] == heldout_contracts[2]
        and len(set(script_hashes)) == 1
        and historical["task_order"] == pre["task_order"] == post["task_order"]
        and score_orders[0] == score_orders[1] == score_orders[2]
        and _slot_coordinates(historical)
        == _slot_coordinates(pre)
        == _slot_coordinates(post)
        and score_contracts[0] == score_contracts[1] == score_contracts[2]
        and len(set(tokenizers)) == 1
    ):
        raise ValueError(
            "historical/stage1/pass2 held-out arms are not exactly paired"
        )

    regression_block = build_historical_regression_block(
        task_order=pre["task_order"],
        historical_by_task=historical["by_task"],
        stage1_by_task=pre["by_task"],
        pass2_by_task=post["by_task"],
    )
    report = {
        "schema": "t5gemma2-kimi-pass2-paired-comparison-v1",
        "status": "complete",
        "heldout_tasks": EXPECTED_ROWS,
        "k": EXPECTED_SAMPLES,
        "exact_pairing_validated": True,
        "same_inference_code": True,
        "same_task_order_and_sources": True,
        "same_sampling_and_slot_seeds": True,
        "same_scoring_contract": True,
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "targets_exposed_to_model": False,
        "post_checkpoint_loader_compat": {
            "path": str(post_compat.resolve()),
            "sha256": sha256_file(post_compat),
            "wrapper_sha256": compat["wrapper_sha256"],
            "scope": compat["scope"],
        },
        "arms": {
            arm["label"]: {
                "predictions": str(arm["prediction_path"].resolve()),
                "predictions_sha256": sha256_file(arm["prediction_path"]),
                "score": str(arm["score_path"].resolve()),
                "score_sha256": sha256_file(arm["score_path"]),
                "metrics": _metric_block(arm["score"]),
            }
            for arm in arms
        },
        "paired_kimi_pass2_vs_stage1": _paired_metrics(
            task_order=pre["task_order"],
            before_by_task=pre["by_task"],
            after_by_task=post["by_task"],
        ),
        "historical_stage1_regressions": regression_block,
    }
    require_exact_or_write(output, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--historical_predictions", type=Path, required=True)
    parser.add_argument("--historical_score", type=Path, required=True)
    parser.add_argument("--pre_predictions", type=Path, required=True)
    parser.add_argument("--pre_score", type=Path, required=True)
    parser.add_argument("--post_predictions", type=Path, required=True)
    parser.add_argument("--post_score", type=Path, required=True)
    parser.add_argument("--post_compat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = compare(
        historical_predictions=args.historical_predictions.resolve(),
        historical_score=args.historical_score.resolve(),
        pre_predictions=args.pre_predictions.resolve(),
        pre_score=args.pre_score.resolve(),
        post_predictions=args.post_predictions.resolve(),
        post_score=args.post_score.resolve(),
        post_compat=args.post_compat.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
