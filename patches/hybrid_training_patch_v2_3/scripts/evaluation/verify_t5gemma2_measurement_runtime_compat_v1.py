#!/usr/bin/env python3
"""Attest current generation/scoring semantics against the sealed seed-42 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    load_journal,
    require_exact_or_write,
    sha256_file,
)


SCHEMA = "t5gemma2-measurement-runtime-compat-v1"
ABLATION_SCHEMA = "t5gemma2-f2-measurement-ablation-provenance-v1"
SCORE_SCHEMA = "direct-compact-attested-passk-v1"
HISTORICAL_WRAPPER = "27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60"
HISTORICAL_BASE = "564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9"
CURRENT_BASE = "30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d"
HISTORICAL_EVALUATOR = "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
CURRENT_EVALUATOR = "5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d"


def _read(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"could not read {label}: {path}") from exc


def _generation_contract(predictions: Path) -> Mapping[str, Any]:
    journal = load_journal(Path(str(predictions) + ".generation.journal.jsonl"))
    if not journal or journal[0].get("event") != "header" or journal[-1].get("event") != "complete":
        raise ValueError(f"generation journal is incomplete: {predictions}")
    return journal[0].get("contract") or {}


def _candidate_projection(score: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(row.get("task_id") or ""),
            int(row.get("sample_index", -1)),
            str(row.get("code_sha256") or ""),
            str(row.get("raw_sha256") or ""),
            bool(row.get("compiled")),
            bool(row.get("passed")),
        )
        for row in score.get("candidate_results") or []
    )


def _task_projection(score: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(row.get("task_id") or ""),
            bool(row.get("pass_at_1")),
            bool(row.get("pass_at_k")),
            bool(row.get("compile_at_k")),
        )
        for row in score.get("task_results") or []
    )


def _model_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project away newly added loader attestations, never model identity."""

    return {
        key: value.get(key)
        for key in (
            "name",
            "revision",
            "config_sha256",
            "arm",
            "tokenizer_sha256",
            "warmstart_contract_sha256",
            "adapter",
        )
    }


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    historical_predictions = Path(args.historical_predictions).expanduser().resolve()
    replay_predictions = Path(args.replay_predictions).expanduser().resolve()
    historical_score = Path(args.historical_score).expanduser().resolve()
    rescored = Path(args.rescored).expanduser().resolve()

    historical_rows = _read(historical_predictions, "historical predictions")
    replay_rows = _read(replay_predictions, "generation replay")
    historical_provenance = _read(
        Path(str(historical_predictions) + ".provenance.json"),
        "historical provenance",
    )
    replay_provenance = _read(
        Path(str(replay_predictions) + ".provenance.json"),
        "replay provenance",
    )
    old_contract = _generation_contract(historical_predictions)
    new_contract = _generation_contract(replay_predictions)

    if (
        not isinstance(historical_rows, list)
        or len(historical_rows) != args.historical_rows
        or not isinstance(replay_rows, list)
        or len(replay_rows) != args.replay_rows
        or historical_provenance.get("schema") != ABLATION_SCHEMA
        or replay_provenance.get("schema") != ABLATION_SCHEMA
        or historical_provenance.get("input_view") != args.input_view
        or replay_provenance.get("input_view") != args.input_view
        or old_contract.get("script_sha256") != HISTORICAL_WRAPPER
        or new_contract.get("script_sha256") != HISTORICAL_WRAPPER
        or old_contract.get("base_inference_script_sha256") != HISTORICAL_BASE
        or new_contract.get("base_inference_script_sha256") != CURRENT_BASE
        or old_contract.get("sampling") != new_contract.get("sampling")
        or _model_identity(old_contract.get("model") or {})
        != _model_identity(new_contract.get("model") or {})
        or old_contract.get("tests_exposed_to_model") is not False
        or new_contract.get("tests_exposed_to_model") is not False
        or old_contract.get("full_gold_targets_exposed_to_model") is not False
        or new_contract.get("full_gold_targets_exposed_to_model") is not False
    ):
        raise ValueError("generation replay contract differs")
    if replay_rows != historical_rows[: args.replay_rows]:
        raise ValueError("current generation path did not reproduce the sealed prefix exactly")

    historical_score_value = _read(historical_score, "historical score")
    rescored_value = _read(rescored, "rescored historical predictions")
    metric_names = ("pass_at_1", "pass_at_k", "compile_at_k")
    if (
        historical_score_value.get("schema") != SCORE_SCHEMA
        or rescored_value.get("schema") != SCORE_SCHEMA
        or historical_score_value.get("tasks") != args.historical_rows
        or rescored_value.get("tasks") != args.historical_rows
        or historical_score_value.get("k") != 10
        or rescored_value.get("k") != 10
        or historical_score_value.get("predictions", {}).get("sha256")
        != sha256_file(historical_predictions)
        or rescored_value.get("predictions", {}).get("sha256")
        != sha256_file(historical_predictions)
        or historical_score_value.get("evaluator", {}).get("sha256")
        != HISTORICAL_EVALUATOR
        or rescored_value.get("evaluator", {}).get("sha256") != CURRENT_EVALUATOR
        or any(
            historical_score_value.get(metric) != rescored_value.get(metric)
            for metric in metric_names
        )
        or _candidate_projection(historical_score_value)
        != _candidate_projection(rescored_value)
        or _task_projection(historical_score_value) != _task_projection(rescored_value)
    ):
        raise ValueError("current scoring path did not reproduce sealed decisions exactly")

    return {
        "schema": SCHEMA,
        "status": "pass",
        "historical_generation": {
            "predictions_sha256": sha256_file(historical_predictions),
            "wrapper_sha256": HISTORICAL_WRAPPER,
            "base_inference_sha256": HISTORICAL_BASE,
            "rows": args.historical_rows,
        },
        "current_generation_replay": {
            "predictions_sha256": sha256_file(replay_predictions),
            "wrapper_sha256": HISTORICAL_WRAPPER,
            "base_inference_sha256": CURRENT_BASE,
            "rows": args.replay_rows,
            "candidates": args.replay_rows * 10,
            "exact_prefix_reproduction": True,
            "model_identity_projection_identical": True,
        },
        "historical_scoring": {
            "score_sha256": sha256_file(historical_score),
            "evaluator_sha256": HISTORICAL_EVALUATOR,
        },
        "current_scoring_replay": {
            "score_sha256": sha256_file(rescored),
            "evaluator_sha256": CURRENT_EVALUATOR,
            "candidate_compile_pass_decisions_identical": True,
            "task_metrics_identical": True,
        },
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--historical_predictions", required=True)
    parser.add_argument("--replay_predictions", required=True)
    parser.add_argument("--historical_score", required=True)
    parser.add_argument("--rescored", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input_view", default="typed_opaque_contract")
    parser.add_argument("--historical_rows", type=int, default=175)
    parser.add_argument("--replay_rows", type=int, default=5)
    args = parser.parse_args(argv)
    if args.historical_rows <= 0 or not 0 < args.replay_rows <= args.historical_rows:
        parser.error("row counts are invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_record(args)
    require_exact_or_write(Path(args.output).expanduser().resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
