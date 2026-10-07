#!/usr/bin/env python3
"""Audit legacy Qwen logprobs and validate the strict sparse-KD schema.

The legacy audit is intentionally non-destructive: verified source code may be
valuable RS-SFT data even when the associated logprobs are not trustworthy KD
targets.  The sparse validator fails closed and is used by the trainer before
loading a model or allocating GPU memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


SPARSE_MANIFEST_SCHEMA = "sparse-topk-tail-kd-manifest-v1"
SPARSE_ROW_SCHEMA = "sparse-topk-tail-kd-row-v1"

REQUIRED_SPARSE_MANIFEST_FIELDS = {
    "schema",
    "data_sha256",
    "rows",
    "source_dataset_sha256",
    "contract_sha256",
    "collector_sha256",
    "teacher_model",
    "teacher_revision",
    "teacher_tokenizer_sha256",
    "student_tokenizer_sha256",
    "student_vocab_size",
    "logprob_base",
    "probability_temperature",
    "logprob_rounding",
    "includes_eos",
    "topk_tail_partition",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no non-empty JSONL rows")
    return rows


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p05": at(0.05),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.mean(ordered)),
        "p95": at(0.95),
        "max": float(ordered[-1]),
    }


def audit_legacy_qwen_logprobs(
    path: str | Path,
    *,
    expected_training_rows: int | None = None,
    student_tokenizer_json: str | Path | None = None,
) -> dict[str, Any]:
    """Describe legacy Chat Completions token strings without blessing them."""

    rows = read_jsonl(path)
    task_ids: set[str] = set()
    positions = 0
    top_candidates = 0
    selected_in_top = 0
    top_masses: list[float] = []
    top_lengths: dict[int, int] = {}
    exact_text_joins = 0
    single_token_positions = 0
    single_token_candidates = 0
    decoded_exact_rows = 0
    tokenizer = None
    if student_tokenizer_json is not None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "the optional token mapping audit requires the `tokenizers` package"
            ) from exc
        tokenizer = Tokenizer.from_file(str(student_tokenizer_json))

    for row_index, row in enumerate(rows):
        task_id = str(row.get("task_id") or "")
        if task_id:
            task_ids.add(task_id)
        token_strings = row.get("teacher_tokens")
        distributions = row.get("teacher_logprobs")
        if not isinstance(token_strings, list) or not isinstance(distributions, list):
            raise ValueError(
                f"row {row_index}: expected teacher_tokens and teacher_logprobs lists"
            )
        if "".join(str(value) for value in token_strings) == str(
            row.get("repair") or ""
        ):
            exact_text_joins += 1
        mapped_target_ids: list[int] = []
        mapped_row = tokenizer is not None
        for position_index, distribution in enumerate(distributions):
            if not isinstance(distribution, Mapping):
                raise ValueError(
                    f"row {row_index} position {position_index}: distribution "
                    "must be an object"
                )
            observed = distribution.get("t")
            observed_logprob = distribution.get("lp")
            top = distribution.get("top")
            if not isinstance(observed, str) or not isinstance(
                observed_logprob, (int, float)
            ):
                raise ValueError(
                    f"row {row_index} position {position_index}: invalid observed token"
                )
            if not isinstance(top, list) or not top:
                raise ValueError(
                    f"row {row_index} position {position_index}: missing top candidates"
                )
            positions += 1
            top_candidates += len(top)
            top_lengths[len(top)] = top_lengths.get(len(top), 0) + 1
            mass = 0.0
            found = False
            for candidate in top:
                if not isinstance(candidate, Mapping):
                    raise ValueError("top candidate must be an object")
                token_text = candidate.get("t")
                logprob = candidate.get("lp")
                if not isinstance(token_text, str) or not isinstance(
                    logprob, (int, float)
                ):
                    raise ValueError("top candidate needs string t and numeric lp")
                mass += math.exp(float(logprob))
                found = found or token_text == observed
                if tokenizer is not None:
                    candidate_ids = tokenizer.encode(
                        token_text, add_special_tokens=False
                    ).ids
                    single_token_candidates += int(len(candidate_ids) == 1)
            selected_in_top += int(found)
            top_masses.append(mass)
            if tokenizer is not None:
                observed_ids = tokenizer.encode(
                    observed, add_special_tokens=False
                ).ids
                if len(observed_ids) == 1:
                    single_token_positions += 1
                    mapped_target_ids.append(int(observed_ids[0]))
                else:
                    mapped_row = False
        if (
            tokenizer is not None
            and mapped_row
            and tokenizer.decode(mapped_target_ids, skip_special_tokens=False)
            == str(row.get("repair") or "")
        ):
            decoded_exact_rows += 1

    missing_provenance = sorted(
        {
            field
            for field in (
                "schema",
                "teacher_model",
                "teacher_revision",
                "teacher_tokenizer_sha256",
                "student_tokenizer_sha256",
                "student_vocab_size",
                "finish_reason",
                "includes_eos",
                "collector_sha256",
            )
            if any(field not in row for row in rows)
        }
    )
    tail_masses = [1.0 - value for value in top_masses]
    coverage = (
        None
        if expected_training_rows is None
        else len(rows) / float(expected_training_rows)
    )
    reasons = [
        "only top-k logprobs are present; the complete teacher vocabulary "
        "distribution is unavailable",
        "top-k probabilities were rounded and no explicit aggregate tail mass "
        "was stored",
        "decoded token strings were stored without API token bytes or token IDs",
        "teacher/student tokenizer identity and vocabulary equality are not sealed",
        "EOS has no teacher distribution",
    ]
    if expected_training_rows is not None and len(rows) != expected_training_rows:
        reasons.append(
            f"artifact covers {len(rows)}/{expected_training_rows} training rows"
        )
    return {
        "schema": "legacy-qwen-logprob-audit-v1",
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "unique_tasks": len(task_ids),
        "expected_training_rows": expected_training_rows,
        "row_coverage": coverage,
        "positions": positions,
        "average_positions_per_row": positions / len(rows),
        "top_candidates": top_candidates,
        "top_length_counts": {str(key): value for key, value in sorted(top_lengths.items())},
        "selected_token_in_top_count": selected_in_top,
        "top_probability_mass": _quantiles(top_masses),
        "inferred_tail_mass": _quantiles(tail_masses),
        "rows_where_tokens_join_to_repair": exact_text_joins,
        "missing_provenance_fields": missing_provenance,
        "student_string_retokenization": (
            None
            if tokenizer is None
            else {
                "single_token_observed_positions": single_token_positions,
                "single_token_observed_fraction": single_token_positions / positions,
                "single_token_top_candidates": single_token_candidates,
                "single_token_top_candidate_fraction": (
                    single_token_candidates / top_candidates
                ),
                "rows_decoding_exactly_to_repair": decoded_exact_rows,
                "warning": (
                    "retokenizing decoded strings does not prove the API teacher "
                    "used the same token IDs"
                ),
            }
        ),
        "usable_for_dense_full_kl": False,
        "usable_for_sparse_topk_tail_kl": False,
        "usable_verified_code_for_rs_sft": all(
            bool(str(row.get("code") or "").strip()) and row.get("ok") is True
            for row in rows
        ),
        "kd_rejection_reasons": reasons,
    }


def _require_manifest_string(manifest: Mapping[str, Any], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sparse manifest field {name!r} must be a non-empty string")
    return value


def _finite_number(value: Any, identity: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{identity} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{identity} must be finite")
    return result


def validate_sparse_topk_tail_dataset(
    data_path: str | Path,
    manifest_path: str | Path,
    *,
    student_tokenizer_json: str | Path,
    expected_contract_sha256: str,
    probability_tolerance: float = 1e-6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the strict sparse schema and return its manifest and rows."""

    if probability_tolerance <= 0:
        raise ValueError("probability_tolerance must be positive")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("sparse manifest must be a JSON object")
    missing = sorted(REQUIRED_SPARSE_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError(f"sparse manifest is missing fields: {missing}")
    if manifest.get("schema") != SPARSE_MANIFEST_SCHEMA:
        raise ValueError(
            f"sparse manifest schema must be {SPARSE_MANIFEST_SCHEMA!r}"
        )
    data_sha = sha256_file(data_path)
    if _require_manifest_string(manifest, "data_sha256") != data_sha:
        raise ValueError("sparse data SHA-256 does not match its manifest")
    tokenizer_sha = sha256_file(student_tokenizer_json)
    if _require_manifest_string(manifest, "student_tokenizer_sha256") != tokenizer_sha:
        raise ValueError("student tokenizer SHA-256 does not match the manifest")
    if _require_manifest_string(manifest, "teacher_tokenizer_sha256") != tokenizer_sha:
        raise ValueError(
            "teacher and student tokenizer JSON must be byte-identical for sparse KD"
        )
    if _require_manifest_string(manifest, "contract_sha256") != str(
        expected_contract_sha256
    ):
        raise ValueError("sparse manifest contract SHA-256 mismatch")
    _require_manifest_string(manifest, "source_dataset_sha256")
    _require_manifest_string(manifest, "collector_sha256")
    _require_manifest_string(manifest, "teacher_model")
    _require_manifest_string(manifest, "teacher_revision")
    vocab_size = manifest.get("student_vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 1:
        raise ValueError("student_vocab_size must be an integer greater than one")
    if manifest.get("logprob_base") != "natural":
        raise ValueError("only natural-log teacher probabilities are supported")
    if float(manifest.get("probability_temperature")) != 1.0:
        raise ValueError(
            "sparse top-k+tail data must be collected at probability temperature 1"
        )
    if manifest.get("logprob_rounding") != "none":
        raise ValueError("sparse teacher logprobs must be stored without rounding")
    if manifest.get("includes_eos") is not True:
        raise ValueError("sparse data must include a teacher distribution for EOS")
    if manifest.get("topk_tail_partition") is not True:
        raise ValueError("manifest must explicitly declare the top-k+tail partition")

    rows = read_jsonl(data_path)
    if manifest.get("rows") != len(rows):
        raise ValueError("sparse manifest row count does not match the data")
    eos_id = manifest.get("eos_token_id")
    if isinstance(eos_id, bool) or not isinstance(eos_id, int):
        raise ValueError("sparse manifest must contain an integer eos_token_id")
    if eos_id < 0 or eos_id >= vocab_size:
        raise ValueError("manifest EOS token lies outside the vocabulary")

    for row_index, row in enumerate(rows):
        identity = str(row.get("task_id") or row.get("id") or f"row-{row_index}")
        if row.get("schema") != SPARSE_ROW_SCHEMA:
            raise ValueError(
                f"{identity}: row schema must be {SPARSE_ROW_SCHEMA!r}"
            )
        targets = row.get("target_input_ids")
        positions = row.get("teacher_positions")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"{identity}: target_input_ids must be non-empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= vocab_size
            for value in targets
        ):
            raise ValueError(f"{identity}: invalid target token ID")
        if targets[-1] != eos_id:
            raise ValueError(f"{identity}: target must end with the sealed EOS token")
        if not isinstance(positions, list) or len(positions) != len(targets):
            raise ValueError(
                f"{identity}: one teacher distribution is required per target token"
            )
        for position_index, (target_id, position) in enumerate(
            zip(targets, positions)
        ):
            if not isinstance(position, dict):
                raise ValueError(
                    f"{identity}:{position_index}: teacher position must be an object"
                )
            observed = position.get("observed_token_id")
            top_ids = position.get("top_token_ids")
            top_logprobs = position.get("top_logprobs")
            tail_mass = _finite_number(
                position.get("tail_mass"),
                f"{identity}:{position_index}:tail_mass",
            )
            if observed != target_id:
                raise ValueError(
                    f"{identity}:{position_index}: observed and target IDs differ"
                )
            if (
                not isinstance(top_ids, list)
                or not top_ids
                or not isinstance(top_logprobs, list)
                or len(top_ids) != len(top_logprobs)
            ):
                raise ValueError(
                    f"{identity}:{position_index}: invalid top-k arrays"
                )
            if len(set(top_ids)) != len(top_ids):
                raise ValueError(
                    f"{identity}:{position_index}: duplicate top token IDs"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= vocab_size
                for value in top_ids
            ):
                raise ValueError(
                    f"{identity}:{position_index}: top token ID outside vocabulary"
                )
            if observed not in top_ids:
                raise ValueError(
                    f"{identity}:{position_index}: observed token is absent from top-k"
                )
            logprobs = [
                _finite_number(
                    value, f"{identity}:{position_index}:top_logprobs"
                )
                for value in top_logprobs
            ]
            if not 0 <= tail_mass <= 1:
                raise ValueError(
                    f"{identity}:{position_index}: tail mass must lie in [0, 1]"
                )
            total = sum(math.exp(value) for value in logprobs) + tail_mass
            if abs(total - 1.0) > probability_tolerance:
                raise ValueError(
                    f"{identity}:{position_index}: top probabilities plus tail "
                    f"sum to {total:.12g}, not one"
                )
    return manifest, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    legacy = subparsers.add_parser("legacy-qwen-audit")
    legacy.add_argument("--input", required=True)
    legacy.add_argument("--expected_training_rows", type=int)
    legacy.add_argument("--student_tokenizer_json")
    legacy.add_argument("--report")

    sparse = subparsers.add_parser("sparse-validate")
    sparse.add_argument("--input", required=True)
    sparse.add_argument("--manifest", required=True)
    sparse.add_argument("--student_tokenizer_json", required=True)
    sparse.add_argument("--contract", required=True)
    sparse.add_argument("--report")
    return parser


def _emit_report(report: Mapping[str, Any], destination: str | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if destination:
        Path(destination).write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "legacy-qwen-audit":
        report = audit_legacy_qwen_logprobs(
            args.input,
            expected_training_rows=args.expected_training_rows,
            student_tokenizer_json=args.student_tokenizer_json,
        )
        _emit_report(report, args.report)
        return
    try:
        manifest, rows = validate_sparse_topk_tail_dataset(
            args.input,
            args.manifest,
            student_tokenizer_json=args.student_tokenizer_json,
            expected_contract_sha256=sha256_file(args.contract),
        )
    except Exception as exc:
        _emit_report(
            {
                "schema": "sparse-topk-tail-validation-report-v1",
                "valid": False,
                "error": str(exc),
            },
            args.report,
        )
        raise SystemExit(2) from exc
    _emit_report(
        {
            "schema": "sparse-topk-tail-validation-report-v1",
            "valid": True,
            "rows": len(rows),
            "data_sha256": sha256_file(args.input),
            "manifest_sha256": sha256_file(args.manifest),
            "teacher_model": manifest["teacher_model"],
            "teacher_revision": manifest["teacher_revision"],
        },
        args.report,
    )


if __name__ == "__main__":
    main()
