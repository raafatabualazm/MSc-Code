"""Stability-audit every archived passing candidate without mutating raw CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.graph_compile_at_k_antigravity import evaluate_dart_jit_tests_detail

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--stem_glob", default="*graphv2_clean*")
    parser.add_argument("--stability_runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sweeps = args.results_dir / "sweeps_antigravity"
    jobs: list[dict[str, Any]] = []
    run_count = 0
    for stats_path in sorted(sweeps.glob(f"{args.stem_glob}_pass_stats.csv")):
        stem = stats_path.name.removesuffix("_pass_stats.csv")
        prediction_path = args.results_dir / f"{stem}_pass_predictions.json"
        if not prediction_path.is_file():
            continue
        with stats_path.open("r", encoding="utf-8", newline="") as handle:
            stats_rows = list(csv.DictReader(handle))
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        if len(stats_rows) != len(predictions):
            raise ValueError(f"Row-count mismatch for {stem}")
        run_count += 1
        pass_columns = sorted(
            (
                (int(match.group(1)), name)
                for name in stats_rows[0]
                if (match := re.fullmatch(r"cand_(\d+)_pass", name))
            ),
            key=lambda item: item[0],
        )
        for row_index, (stats_row, prediction_row) in enumerate(
            zip(stats_rows, predictions), start=1
        ):
            candidates = prediction_row.get("predictions") or [
                prediction_row.get("prediction", "")
            ]
            tests = str(prediction_row.get("tests") or "")
            task_id = str(
                prediction_row.get("task_id", prediction_row.get("id", row_index))
            )
            for candidate_index, column in pass_columns:
                if int(float(stats_row.get(column) or 0)) != 1:
                    continue
                jobs.append(
                    {
                        "stem": stem,
                        "row_one_based": row_index,
                        "problem_id": str(stats_row.get("problem_id", task_id)),
                        "candidate_one_based": candidate_index,
                        "candidate": candidates[candidate_index - 1],
                        "tests": tests,
                        "task_id": task_id,
                    }
                )

    def score(job: dict[str, Any]) -> dict[str, Any]:
        compiled, passed, diagnostic, _source = evaluate_dart_jit_tests_detail(
            job["candidate"],
            job["tests"],
            job["task_id"],
            timeout=args.timeout,
            stability_runs=args.stability_runs,
        )
        return {
            **{key: value for key, value in job.items() if key not in {"candidate", "tests"}},
            "compiled": compiled,
            "passed": passed,
            "diagnostic": diagnostic[:1000],
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        iterator = pool.map(score, jobs)
        outcomes = list(
            tqdm(iterator, total=len(jobs), desc="stability audit", unit="candidate")
            if tqdm is not None else iterator
        )

    invalidated = [item for item in outcomes if not item["passed"]]
    corrections = [
        {
            "stem": item["stem"],
            "row_one_based": item["row_one_based"],
            "problem_id": item["problem_id"],
            "candidate_one_based": item["candidate_one_based"],
            "field": "pass",
            "archived_value": 1,
            "corrected_value": 0,
            "reason": "Archived one-run pass did not survive stability-qualified replay.",
            "diagnostic": item["diagnostic"],
        }
        for item in invalidated
    ]
    report = {
        "schema": "antigravity-pass-stability-audit-v1",
        "results_dir": str(args.results_dir.resolve()),
        "stem_glob": args.stem_glob,
        "run_count": run_count,
        "stability_runs": args.stability_runs,
        "archived_passing_candidates_checked": len(jobs),
        "stable_passing_candidates": len(jobs) - len(invalidated),
        "invalidated_candidates": len(invalidated),
        "corrections": corrections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "corrections"}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
