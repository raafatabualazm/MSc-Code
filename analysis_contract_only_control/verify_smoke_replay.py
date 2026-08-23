#!/usr/bin/env python3
"""Verify that the scored five-task smoke exactly replays in full seed 42."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis_contract_only_control.contract_only_view import VIEW
from analysis_contract_only_control.score_contract_only import (
    PROVENANCE_SCHEMA,
    validate_existing_contract_only_score,
)
from scripts.evaluation import t5gemma2_measurement_audit_report as audit_report
from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)


SCHEMA = "t5gemma2-contract-only-smoke-replay-gate-v1"


def _model_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "name",
            "revision",
            "config_sha256",
            "arm",
            "training_stage_schema",
            "tokenizer_sha256",
            "warmstart_contract_sha256",
            "adapter",
        )
    }


def _score_projection(score: Mapping[str, Any], task_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row.get("task_id"),
            "sample_index": row.get("sample_index"),
            "raw_sha256": row.get("raw_sha256"),
            "code_sha256": row.get("code_sha256"),
            "compiled": row.get("compiled"),
            "passed": row.get("passed"),
        }
        for row in score.get("candidate_results") or []
        if row.get("task_id") in task_ids
    ]


def _expected_first_n_jsonl_bytes(path: Path, rows: int) -> bytes:
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < rows or any(not line.strip() for line in lines):
        raise ValueError("full evaluation JSONL is incomplete or contains blanks")
    payload = b"".join(lines[:rows])
    if payload and not payload.endswith((b"\n", b"\r")):
        payload += b"\n"
    return payload


def build_gate(args: argparse.Namespace) -> dict[str, Any]:
    smoke_predictions = Path(args.smoke_predictions).expanduser().resolve()
    smoke_score = Path(args.smoke_score).expanduser().resolve()
    full_predictions = Path(args.full_predictions).expanduser().resolve()
    full_score = Path(args.full_score).expanduser().resolve()
    smoke_evaluation = Path(args.smoke_evaluation).expanduser().resolve()
    full_evaluation = Path(args.full_evaluation).expanduser().resolve()
    if smoke_evaluation.read_bytes() != _expected_first_n_jsonl_bytes(
        full_evaluation, args.smoke_tasks
    ):
        raise ValueError("smoke evaluation is not the exact first-N full JSONL bytes")
    smoke = audit_report._load_arm(  # noqa: SLF001
        label="contract_only_smoke_seed42",
        predictions_path=smoke_predictions,
        score_path=smoke_score,
        expected_tasks=args.smoke_tasks,
        expected_k=args.k,
        expected_provenance_schema=PROVENANCE_SCHEMA,
    )
    full = audit_report._load_arm(  # noqa: SLF001
        label="contract_only_full_seed42",
        predictions_path=full_predictions,
        score_path=full_score,
        expected_tasks=args.full_tasks,
        expected_k=args.k,
        expected_provenance_schema=PROVENANCE_SCHEMA,
    )
    validate_existing_contract_only_score(
        predictions=smoke_predictions,
        evaluation_file=smoke_evaluation,
        output=smoke_score,
        k=args.k,
        timeout=args.timeout,
        stability_runs=args.stability_runs,
    )
    validate_existing_contract_only_score(
        predictions=full_predictions,
        evaluation_file=full_evaluation,
        output=full_score,
        k=args.k,
        timeout=args.timeout,
        stability_runs=args.stability_runs,
    )
    smoke_ids = set(smoke["task_ids"])
    smoke_terminals = smoke["journal"][1:-1]
    full_terminals = full["journal"][1 : 1 + args.smoke_tasks]
    smoke_candidates = _score_projection(smoke["score"], smoke_ids)
    full_candidates = _score_projection(full["score"], smoke_ids)
    if (
        smoke["provenance"].get("input_view") != VIEW
        or full["provenance"].get("input_view") != VIEW
        or smoke["task_ids"] != full["task_ids"][: args.smoke_tasks]
        or smoke["predictions"] != full["predictions"][: args.smoke_tasks]
        or smoke["coordinates"] != full["coordinates"][: args.smoke_tasks]
        or [row.get("source_sha256") for row in smoke_terminals]
        != [row.get("source_sha256") for row in full_terminals]
        or canonical_sha256(_model_identity(smoke["provenance"]["model"]))
        != canonical_sha256(_model_identity(full["provenance"]["model"]))
        or smoke["provenance"].get("sampling")
        != full["provenance"].get("sampling")
        or smoke_candidates != full_candidates
    ):
        raise ValueError("five-task smoke does not exactly replay in full seed 42")
    return {
        "schema": SCHEMA,
        "status": "pass",
        "smoke_tasks": args.smoke_tasks,
        "k": args.k,
        "candidate_slots": args.smoke_tasks * args.k,
        "predictions_exact_prefix_reproduction": True,
        "sample_coordinates_identical": True,
        "source_hashes_identical": True,
        "model_identity_identical": True,
        "sampling_identical": True,
        "candidate_compile_pass_decisions_identical": True,
        "smoke_evaluation_exact_first_n_bytes": True,
        "smoke_evaluation_sha256": sha256_file(smoke_evaluation),
        "full_evaluation_sha256": sha256_file(full_evaluation),
        "smoke": {
            "predictions_sha256": sha256_file(smoke_predictions),
            "score_sha256": sha256_file(smoke_score),
        },
        "full_seed42": {
            "predictions_sha256": sha256_file(full_predictions),
            "score_sha256": sha256_file(full_score),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--smoke_predictions", required=True)
    parser.add_argument("--smoke_score", required=True)
    parser.add_argument("--smoke_evaluation", required=True)
    parser.add_argument("--full_predictions", required=True)
    parser.add_argument("--full_score", required=True)
    parser.add_argument("--full_evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke_tasks", type=int, default=5)
    parser.add_argument("--full_tasks", type=int, default=175)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=2)
    args = parser.parse_args(argv)
    if min(
        args.smoke_tasks,
        args.full_tasks,
        args.k,
        args.timeout,
        args.stability_runs,
    ) <= 0 or args.smoke_tasks >= args.full_tasks:
        parser.error("invalid smoke/full scoring dimensions")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gate = build_gate(args)
    require_exact_or_write(Path(args.output).expanduser().resolve(), gate)
    print(json.dumps(gate, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
