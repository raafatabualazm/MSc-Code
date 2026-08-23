#!/usr/bin/env python3
"""Build a hidden-test-safe VeRPO in-context-repair dataset.

The builder joins inference output to the original Phase-0 rows using any
shared task identifier, executes only ``feedback_tests``, and strips hidden or
legacy full-harness fields from emitted rows.  A low join rate is an error, not
a successful empty/partial repair corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IDENTITY_FIELDS = ("task_id", "id", "source_line")
FORBIDDEN_OUTPUT_TEST_FIELDS = (
    "tests",
    "acceptance_tests",
    "hidden_tests",
    "private_tests",
)


def _load(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_predictions(path: str | Path) -> list[dict[str, Any]]:
    # Accept JSONL or one JSON array/object with a ``rows``/``predictions`` list.
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("rows", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"unsupported prediction container in {path}")


def _identity_values(row: dict[str, Any]) -> set[str]:
    """Canonical aliases; inference ``id`` may correspond to source ``task_id``."""
    values: set[str] = set()
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            values.add(str(value).strip())
    return values


def _prediction_index(
    predictions: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    index: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for prediction in predictions:
        for identity in _identity_values(prediction):
            prior = index.get(identity)
            if prior is not None and prior is not prediction:
                ambiguous.add(identity)
                continue
            index[identity] = prediction
    return index, ambiguous


def _match_prediction(
    row: dict[str, Any],
    index: dict[str, dict[str, Any]],
    ambiguous: set[str],
) -> dict[str, Any] | None:
    identities = _identity_values(row)
    bad = sorted(identities & ambiguous)
    if bad:
        raise ValueError(f"ambiguous prediction identities for row: {bad}")
    matches = {id(index[value]): index[value] for value in identities if value in index}
    if len(matches) > 1:
        raise ValueError(
            "row identities resolve to different prediction records: "
            + ", ".join(sorted(identities))
        )
    return next(iter(matches.values()), None)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach a fail-closed LLM critique (default: enabled).",
    )
    parser.add_argument(
        "--max_per_task",
        type=int,
        default=8,
        help="Maximum leading predictions inspected per task (default: 8).",
    )
    parser.add_argument(
        "--min_match_rate",
        type=float,
        default=0.95,
        help="Minimum fraction of source rows that must join to prediction records.",
    )
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if args.max_per_task <= 0:
        parser.error("--max_per_task must be positive")
    if not 0.0 < args.min_match_rate <= 1.0:
        parser.error("--min_match_rate must be in (0, 1]")

    rows = _load(args.rows)
    predictions = _load_predictions(args.predictions)
    if not rows:
        raise SystemExit("source VeRPO rows are empty")
    if not predictions:
        raise SystemExit("VeRPO prediction file is empty")
    prediction_index, ambiguous = _prediction_index(predictions)

    eligible_pairs: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
    skipped_no_prediction = 0
    skipped_no_feedback_tests = 0
    for row_index, row in enumerate(rows):
        feedback_tests = str(row.get("feedback_tests") or "")
        if not feedback_tests.strip():
            skipped_no_feedback_tests += 1
            continue
        prediction = _match_prediction(row, prediction_index, ambiguous)
        if not prediction or not isinstance(prediction.get("predictions"), list):
            skipped_no_prediction += 1
            continue
        eligible_pairs.append((row_index, row, prediction, feedback_tests))

    eligible_rows = len(rows) - skipped_no_feedback_tests
    matched_rows = len(eligible_pairs)
    match_rate = matched_rows / eligible_rows if eligible_rows else 0.0
    overall_match_rate = matched_rows / len(rows)
    join_report = {
        "schema_version": 2,
        "rows_in": len(rows),
        "eligible_rows": eligible_rows,
        "prediction_rows": len(predictions),
        "matched_rows": matched_rows,
        "match_rate_eligible": match_rate,
        "match_rate_overall": overall_match_rate,
        "skipped_no_prediction": skipped_no_prediction,
        "skipped_no_feedback_tests": skipped_no_feedback_tests,
        "ambiguous_prediction_identities": sorted(ambiguous),
    }
    if match_rate < args.min_match_rate:
        failure_report = {
            **join_report,
            "status": "failed_join_rate",
            "minimum_match_rate": args.min_match_rate,
        }
        print(json.dumps(failure_report, sort_keys=True))
        if args.report:
            _write_json(args.report, failure_report)
        print(
            f"ERROR: eligible-row prediction join rate {match_rate:.3f} is below "
            f"--min_match_rate={args.min_match_rate:.3f}",
            file=sys.stderr,
        )
        return 2

    # Import the heavyweight executable reward only after cheap schema/join
    # validation helpers have loaded; this also keeps unit tests lightweight.
    from graph_grpo_decompiler_antigravity import TruePerTestReward

    scorer = TruePerTestReward()
    scorer.reward_mode = "binary"
    judge = None
    if args.judge:
        from verpo_judge_antigravity import VerpoJudge

        judge = VerpoJudge()
        judge.validate_configuration()

    out_rows: list[dict[str, Any]] = []
    skipped_already_pass = 0
    for row_index, row, prediction, feedback_tests in eligible_pairs:
        # Prefer a compiling failed attempt because it has the richest
        # diagnostic. Fall back to the first failed attempt among the bounded
        # leading candidates.
        chosen: tuple[str, dict[str, Any]] | None = None
        fallback: tuple[str, dict[str, Any]] | None = None
        for raw_candidate in prediction["predictions"][: args.max_per_task]:
            candidate = str(raw_candidate or "").strip()
            if not candidate:
                continue
            details = scorer.compute_reward_details(candidate, feedback_tests)
            if details.get("full_pass"):
                continue
            fallback = fallback or (candidate, details)
            if details.get("compiled"):
                chosen = (candidate, details)
                break
        chosen = chosen or fallback
        if chosen is None:
            skipped_already_pass += 1
            continue

        candidate, details = chosen
        feedback = str(
            details.get("diagnostic") or details.get("status") or "visible tests failed"
        ).strip()
        if judge is not None:
            critique = judge.critique(
                [
                    {
                        "tests": feedback_tests,
                        "candidate": candidate,
                        "diagnostic": feedback,
                        "compiled": bool(details.get("compiled")),
                        "full_pass": False,
                    }
                ]
            )[0]
            feedback = feedback + "\n\nReviewer: " + critique.strip()

        repaired_row = dict(row)
        for field in FORBIDDEN_OUTPUT_TEST_FIELDS:
            repaired_row.pop(field, None)
        repaired_row["prior_attempt"] = candidate
        repaired_row["repair_feedback"] = feedback
        repaired_row["verpo_repair"] = True
        repaired_row["verpo_repair_metadata"] = {
            "schema_version": 2,
            "source_row_index": row_index,
            "reward_test_field": "feedback_tests",
            "hidden_tests_exposed": False,
            "judge_enabled": bool(judge),
        }
        out_rows.append(repaired_row)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        **join_report,
        "status": "complete",
        "repair_rows": len(out_rows),
        "skipped_already_pass": skipped_already_pass,
        "reward_test_field": "feedback_tests",
        "hidden_tests_exposed": False,
        "judge": judge.telemetry() if judge is not None else {"enabled": False},
        "inputs": {
            "rows": {"path": str(Path(args.rows).resolve()), "sha256": _sha256(args.rows)},
            "predictions": {
                "path": str(Path(args.predictions).resolve()),
                "sha256": _sha256(args.predictions),
            },
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
        },
    }
    print(json.dumps(report, sort_keys=True))
    if args.report:
        _write_json(args.report, report)
    if not out_rows:
        print("ERROR: produced 0 repair rows", file=sys.stderr)
        return 1
    if judge is not None:
        judge.assert_healthy(require_success=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
