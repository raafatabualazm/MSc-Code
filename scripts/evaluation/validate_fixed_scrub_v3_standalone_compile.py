#!/usr/bin/env python3
"""Validate and seal the secondary fixed-scrub-v3 standalone compile result."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from scripts.evaluation.project_fixed_scrub_v3_standalone_pool import (
        EXPECTED_ROWS,
        EXPECTED_SAMPLES,
        SCHEMA_VERSION as PROJECTION_SCHEMA_VERSION,
        candidate_stream_sha256,
        file_record,
        load_candidate_pool,
        load_prediction_candidate_rows,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct execution from scripts/evaluation.
    from project_fixed_scrub_v3_standalone_pool import (  # type: ignore[no-redef]
        EXPECTED_ROWS,
        EXPECTED_SAMPLES,
        SCHEMA_VERSION as PROJECTION_SCHEMA_VERSION,
        candidate_stream_sha256,
        file_record,
        load_candidate_pool,
        load_prediction_candidate_rows,
        sha256_file,
    )


SCHEMA_VERSION = "fixed-scrub-v3-standalone-compile-provenance-v1"
EXPECTED_K = (1, 5, 10)
PINNED_DART_PREFIX = "Dart SDK version: 3.11.5 (stable)"
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
PROGRESS_RE = re.compile(
    r"^\[(\d+)/(\d+)\] n=(\d+), compiling=(\d+), "
    r"compile@1=([0-9]+(?:\.[0-9]+)?)$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def require_hash(value: str, label: str) -> str:
    require(bool(SHA256_RE.fullmatch(value)), f"{label} must be a SHA-256 digest")
    return value.lower()


def require_record_matches(record: Any, path: Path, label: str) -> None:
    require(isinstance(record, dict), f"projection provenance lacks {label} record")
    require(record.get("sha256") == sha256_file(path), f"{label} SHA-256 mismatch")
    require(record.get("size_bytes") == path.stat().st_size, f"{label} size mismatch")


def compile_at_k_estimator(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    product = 1.0
    for index in range(k):
        product *= (n - c - index) / (n - index)
    return 1.0 - product


def extract_final_result(stdout_text: str) -> dict[str, Any]:
    lines = stdout_text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() != "{":
            continue
        candidate = "\n".join(lines[index:]).strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("compile stdout lacks a final JSON object")


def validate_stdout(stdout_path: Path) -> tuple[dict[str, Any], list[int]]:
    try:
        text = stdout_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read compile stdout {stdout_path}: {exc}") from exc

    progress = []
    for line in text.splitlines():
        match = PROGRESS_RE.fullmatch(line.strip())
        if match:
            progress.append(tuple(match.groups()))
    require(
        len(progress) == EXPECTED_ROWS,
        f"compile stdout: expected {EXPECTED_ROWS} row summaries, found {len(progress)}",
    )

    compile_counts: list[int] = []
    for expected_index, groups in enumerate(progress, 1):
        row_index, denominator, samples, compiling, displayed_at_1 = groups
        require(int(row_index) == expected_index, "compile stdout row order is incomplete")
        require(int(denominator) == EXPECTED_ROWS, "compile stdout row denominator mismatch")
        require(int(samples) == EXPECTED_SAMPLES, "compile stdout sample count mismatch")
        compiled = int(compiling)
        require(0 <= compiled <= EXPECTED_SAMPLES, "invalid per-row compile count")
        displayed = float(displayed_at_1)
        require(
            math.isclose(
                displayed,
                compiled / EXPECTED_SAMPLES,
                rel_tol=0.0,
                abs_tol=5.1e-5,
            ),
            f"compile stdout row {expected_index}: displayed compile@1 mismatch",
        )
        compile_counts.append(compiled)

    result = extract_final_result(text)
    expected_keys = {*(f"compile_at_{k}" for k in EXPECTED_K), "total_problems"}
    require(set(result) == expected_keys, "compile stdout final JSON has unexpected keys")
    total = result.get("total_problems")
    require(
        isinstance(total, int) and not isinstance(total, bool) and total == EXPECTED_ROWS,
        "compile stdout total_problems mismatch",
    )

    for k in EXPECTED_K:
        key = f"compile_at_{k}"
        value = result.get(key)
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0,
            f"compile stdout {key} is invalid",
        )
        recomputed = sum(
            compile_at_k_estimator(EXPECTED_SAMPLES, count, k)
            for count in compile_counts
        ) / EXPECTED_ROWS
        require(
            math.isclose(float(value), recomputed, rel_tol=0.0, abs_tol=1e-12),
            f"compile stdout {key} does not match per-row counts",
        )
    return result, compile_counts


def validate_and_seal(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = (
        args.source_predictions,
        args.candidate_pool,
        args.projection_provenance,
        args.compile_stdout,
        args.scorer,
        args.projector,
        args.dart_version_file,
    )
    require(
        args.output.resolve() not in {path.resolve() for path in input_paths},
        "standalone provenance output must not overwrite an input",
    )
    expected_pool_hash = require_hash(args.expected_pool_sha256, "expected pool hash")
    expected_projection_hash = require_hash(
        args.expected_projection_provenance_sha256,
        "expected projection provenance hash",
    )
    expected_scorer_hash = require_hash(args.expected_scorer_sha256, "expected scorer hash")

    require(sha256_file(args.candidate_pool) == expected_pool_hash, "candidate pool pin mismatch")
    require(
        sha256_file(args.projection_provenance) == expected_projection_hash,
        "projection provenance pin mismatch",
    )
    require(sha256_file(args.scorer) == expected_scorer_hash, "scorer pin mismatch")

    source_candidates = load_prediction_candidate_rows(args.source_predictions)
    pool = load_candidate_pool(args.candidate_pool)
    pool_candidates = [row["predictions"] for row in pool]
    source_digest = candidate_stream_sha256(source_candidates)
    pool_digest = candidate_stream_sha256(pool_candidates)
    require(pool_candidates == source_candidates, "candidate stream differs from source predictions")
    require(pool_digest == source_digest, "candidate stream digest mismatch")

    projection = load_json_object(args.projection_provenance, "projection provenance")
    require(
        projection.get("schema_version") == PROJECTION_SCHEMA_VERSION,
        "projection schema mismatch",
    )
    require(projection.get("diagnostic_role") == "secondary_standalone_compile", "bad projection role")
    require(projection.get("row_count") == EXPECTED_ROWS, "projection row count mismatch")
    require(projection.get("samples_per_row") == EXPECTED_SAMPLES, "projection sample count mismatch")
    require(
        projection.get("candidate_count") == EXPECTED_ROWS * EXPECTED_SAMPLES,
        "projection candidate count mismatch",
    )
    require(projection.get("candidate_stream_sha256") == source_digest, "projection stream hash mismatch")
    require(projection.get("candidate_stream_preserved") is True, "projection did not preserve stream")
    require(projection.get("pool_is_candidate_only") is True, "projection pool is not candidate-only")
    require(projection.get("pool_row_keys") == ["id", "predictions"], "projection pool keys changed")
    require(projection.get("semantic_identifiers_preserved") is False, "semantic IDs retained")
    require(
        projection.get("hidden_evaluation_metadata_preserved") is False,
        "hidden evaluation metadata retained",
    )
    require_record_matches(
        (projection.get("inputs") or {}).get("predictions"),
        args.source_predictions,
        "source predictions",
    )
    require_record_matches(
        (projection.get("output") or {}).get("candidate_pool"),
        args.candidate_pool,
        "candidate pool",
    )
    require_record_matches(projection.get("projector"), args.projector, "projector")

    dart_version = args.dart_version_file.read_text(encoding="utf-8").strip()
    require(dart_version.startswith(PINNED_DART_PREFIX), f"unexpected Dart SDK: {dart_version!r}")
    result, compile_counts = validate_stdout(args.compile_stdout)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_role": "secondary_standalone_compile",
        "diagnostic_only": True,
        "standalone_compile_mode": "legacy",
        "k_values": list(EXPECTED_K),
        "row_count": EXPECTED_ROWS,
        "samples_per_row": EXPECTED_SAMPLES,
        "candidate_count": EXPECTED_ROWS * EXPECTED_SAMPLES,
        "candidate_stream_sha256": source_digest,
        "scorer_sha256": expected_scorer_hash,
        "dart_sdk_version": dart_version,
        "result": result,
        "validated_invariants": {
            "candidate_only_pool": True,
            "candidate_stream_preserved": True,
            "opaque_row_identifiers": True,
            "tests_absent": True,
            "all_row_counts_present": len(compile_counts) == EXPECTED_ROWS,
            "compile_at_k_recomputed_from_row_counts": True,
            "primary_aligned_jit_analysis_modified": False,
        },
        "inputs": {
            "source_predictions": file_record(args.source_predictions),
            "candidate_pool": file_record(args.candidate_pool),
            "projection_provenance": file_record(args.projection_provenance),
            "compile_stdout": file_record(args.compile_stdout),
            "scorer": file_record(args.scorer),
            "projector": file_record(args.projector),
            "dart_version": file_record(args.dart_version_file),
        },
    }
    require("compile_mode" not in payload, "reserved primary-metric provenance key used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_predictions", required=True, type=Path)
    parser.add_argument("--candidate_pool", required=True, type=Path)
    parser.add_argument("--projection_provenance", required=True, type=Path)
    parser.add_argument("--compile_stdout", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--projector", required=True, type=Path)
    parser.add_argument("--dart_version_file", required=True, type=Path)
    parser.add_argument("--expected_pool_sha256", required=True)
    parser.add_argument("--expected_projection_provenance_sha256", required=True)
    parser.add_argument("--expected_scorer_sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    payload = validate_and_seal(parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
