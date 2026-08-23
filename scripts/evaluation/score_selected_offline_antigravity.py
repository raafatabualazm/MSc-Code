"""
Score a rerank report offline against the stored per-candidate stats CSV.

rerank_predictions_antigravity.py only knows pass outcomes in stats_pass_oracle
and test modes; in heuristic and stats_compile modes its report has no pass
rates. This tool joins the reranker's chosen candidate (best_original_index
per row in the --report JSON) with the stored cand_N_pass / cand_N_compile
flags in the matching *_pass_stats.csv, so selected pass@1 and compile@1 can
be measured for ANY mode with no Dart execution.

It also replays the standard estimators from the CSV alone as cross-checks
against the recorded harness numbers:
  pass_at_1_replay   mean over rows of (passing candidates / n)  == harness pass@1
  any_pass_rate      fraction of rows with >=1 passing candidate == pass@n oracle

Usage:
  python scripts/evaluation/score_selected_offline_antigravity.py \
    --report  results-.../analysis/NAME_reranker_MODE_report.json \
    --stats   results-.../sweeps_antigravity/NAME_pass_stats.csv \
    --label   "NAME MODE"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _parse_flag(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    try:
        return 1 if float(text) >= 0.5 else 0
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path,
                        help="Report JSON written by rerank_predictions_antigravity.py --report")
    parser.add_argument("--stats", required=True, type=Path,
                        help="The *_pass_stats.csv matching the SAME predictions file")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    report_rows = report["rows"]

    with args.stats.open("r", encoding="utf-8", newline="") as handle:
        stats_rows = list(csv.DictReader(handle))

    if len(stats_rows) != len(report_rows):
        raise SystemExit(
            f"ERROR: report rows ({len(report_rows)}) and stats rows ({len(stats_rows)}) differ - "
            "the report and CSV must come from the same predictions file"
        )

    cand_indices = sorted(
        int(match.group(1))
        for match in (re.match(r"cand_(\d+)_pass$", col) for col in stats_rows[0])
        if match
    )
    if not cand_indices:
        raise SystemExit("ERROR: no cand_N_pass columns found in the stats CSV")
    n_cands = max(cand_indices)

    def flag(row_idx: int, cand_idx: int, field: str) -> int:
        return _parse_flag(stats_rows[row_idx].get(f"cand_{cand_idx + 1}_{field}", "0"))

    selected_pass = []
    selected_compile = []
    original_pass = []
    original_compile = []
    pass_at_1_replay = []
    compile_at_1_replay = []
    any_pass = []
    any_compile = []

    for row_idx, row_report in enumerate(report_rows):
        best_idx = int(row_report.get("best_original_index", 0))
        if best_idx >= n_cands:
            raise SystemExit(
                f"ERROR: row {row_idx} selected candidate {best_idx + 1} but the CSV "
                f"only has {n_cands} candidates - report/CSV mismatch"
            )
        selected_pass.append(flag(row_idx, best_idx, "pass"))
        selected_compile.append(flag(row_idx, best_idx, "compile"))
        original_pass.append(flag(row_idx, 0, "pass"))
        original_compile.append(flag(row_idx, 0, "compile"))

        row_pass = [flag(row_idx, idx, "pass") for idx in range(n_cands)]
        row_compile = [flag(row_idx, idx, "compile") for idx in range(n_cands)]
        pass_at_1_replay.append(sum(row_pass) / n_cands)
        compile_at_1_replay.append(sum(row_compile) / n_cands)
        any_pass.append(1 if any(row_pass) else 0)
        any_compile.append(1 if any(row_compile) else 0)

    total = len(report_rows)
    moved = sum(1 for row_report in report_rows if row_report.get("moved"))

    def rate(values) -> float:
        return sum(values) / total if total else 0.0

    summary = {
        "label": args.label or args.report.stem,
        "rerank_mode": report.get("summary", {}).get("mode"),
        "total_problems": total,
        "n_candidates": n_cands,
        "moved_rate": round(moved / total, 4) if total else 0.0,
        "selected_pass_at_1": round(rate(selected_pass), 4),
        "selected_compile_at_1": round(rate(selected_compile), 4),
        "first_candidate_pass_at_1": round(rate(original_pass), 4),
        "first_candidate_compile_at_1": round(rate(original_compile), 4),
        "delta_pass_pts_vs_first_candidate": round(
            100.0 * (rate(selected_pass) - rate(original_pass)), 2
        ),
        "csv_replay_cross_checks": {
            "pass_at_1_estimator": round(rate(pass_at_1_replay), 4),
            "compile_at_1_estimator": round(rate(compile_at_1_replay), 4),
            "any_pass_rate_pass_at_n_oracle": round(rate(any_pass), 4),
            "any_compile_rate": round(rate(any_compile), 4),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
