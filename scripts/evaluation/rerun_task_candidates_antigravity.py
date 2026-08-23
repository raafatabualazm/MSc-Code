"""Rerun one prediction row to diagnose nondeterministic Dart outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.graph_pass_at_k_antigravity import run_dart_test_detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--row", required=True, type=int, help="One-based prediction row")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.row < 1 or args.row > len(rows):
        raise SystemExit(f"--row must be in 1..{len(rows)}")
    row = rows[args.row - 1]
    candidates = row.get("predictions") or [row.get("prediction", "")]
    tests = str(row.get("tests") or "")
    task_id = str(row.get("task_id", row.get("id", args.row)))

    report = {
        "row": args.row,
        "task_id": task_id,
        "problem_id": row.get("id"),
        "runs": [],
    }
    for repeat in range(args.repeats):
        def score(item: tuple[int, str]) -> dict:
            index, candidate = item
            passed, diagnostic, _source = run_dart_test_detail(
                candidate, tests, task_id, timeout=args.timeout
            )
            return {
                "candidate": index + 1,
                "passed": passed,
                "diagnostic": diagnostic[:500],
            }

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            outcomes = list(pool.map(score, enumerate(candidates)))
        report["runs"].append(
            {
                "repeat": repeat + 1,
                "passed": sum(item["passed"] for item in outcomes),
                "outcomes": outcomes,
            }
        )

    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
