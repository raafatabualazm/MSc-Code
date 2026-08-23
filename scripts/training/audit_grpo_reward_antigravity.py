#!/usr/bin/env python3
"""Audit GRPO reward compatibility and optional full-harness parity.

Static mode is fast and requires no GPU. ``--run_references`` additionally
checks that reference source receives a binary perfect reward under the same
full Dart harness used by pass@k. Run the full pool on Linux before GRPO.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main_assertions(test_code: str) -> list[str]:
    assertions: list[str] = []
    in_main = False
    depth = 0
    for line in test_code.splitlines():
        stripped = line.strip()
        if not in_main and re.match(r"^void\s+main\s*\([^)]*\)\s*\{", stripped):
            in_main = True
            depth = line.count("{") - line.count("}")
            continue
        if in_main:
            if stripped.startswith("expect("):
                assertions.append(stripped)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                in_main = False
    return assertions


def static_audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    assertion_counts: list[int] = []
    helper_names: set[str] = set()
    for index, row in enumerate(rows):
        tests = str(row.get("tests") or "")
        assertions = main_assertions(tests)
        extracted = [line for line in assertions if "candidate(" in line]
        assertion_counts.append(len(assertions))
        for line in tests.splitlines():
            match = re.match(
                r"^\s*(?:void|bool|dynamic|int|double|String|List(?:<[^>]+>)?|"
                r"Map(?:<[^>]+>)?|Set(?:<[^>]+>)?)\s+(expect\w*)\s*\(",
                line,
            )
            if match:
                helper_names.add(match.group(1))
        reasons = []
        if not tests:
            reasons.append("missing tests")
        if len(assertions) < 1:
            reasons.append("no main assertions")
        if len(extracted) != len(assertions):
            reasons.append(f"candidate extraction {len(extracted)}/{len(assertions)}")
        if not re.search(r"\bfinal\s+candidate\s*=\s*[A-Za-z_]\w*\s*;", tests):
            reasons.append("missing candidate alias")
        if reasons:
            failures.append(
                {
                    "index": index,
                    "task_id": row.get("task_id"),
                    "function": row.get("function"),
                    "reasons": reasons,
                }
            )
    summary = {
        "rows": len(rows),
        "static_failures": len(failures),
        "assertions_min": min(assertion_counts) if assertion_counts else 0,
        "assertions_max": max(assertion_counts) if assertion_counts else 0,
        "helper_names": sorted(helper_names),
    }
    return failures, summary


def run_reference_audit(
    rows: list[dict[str, Any]], limit: int, workers: int
) -> tuple[int, list[dict[str, Any]]]:
    os.environ["GRPO_REWARD_MODE"] = "binary"
    os.environ["GRPO_PERFECT_BASE_REWARD"] = "1.0"
    os.environ["GRPO_PERFECT_BONUS"] = "0.0"
    os.environ["GRPO_BINARY_FAIL_REWARD"] = "-1.0"
    from scripts.training.graph_grpo_decompiler_antigravity import TruePerTestReward

    selected = rows if limit < 0 else rows[:limit]

    def check(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any] | None]:
        index, row = item
        scorer = TruePerTestReward()
        try:
            details = scorer.compute_reward_details(
                str(row.get("dart_source") or ""), str(row.get("tests") or "")
            )
            if details["reward"] != 1.0 or details["pass_ratio"] != 1.0:
                return index, {
                    "index": index,
                    "task_id": row.get("task_id"),
                    "function": row.get("function"),
                    "details": details,
                }
            return index, None
        except Exception as exc:
            return index, {
                "index": index,
                "task_id": row.get("task_id"),
                "function": row.get("function"),
                "error": repr(exc),
            }

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(check, item) for item in enumerate(selected)]
        for done, future in enumerate(as_completed(futures), start=1):
            _, failure = future.result()
            if failure:
                failures.append(failure)
            if done % 100 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] reward parity failures={len(failures)}")
    failures.sort(key=lambda item: item["index"])
    return len(selected), failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run_references", type=int, default=0,
                        help="0=static only, -1=all, positive=N first rows")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    failures, summary = static_audit(rows)
    report: dict[str, Any] = {**summary, "static_failure_rows": failures}

    if args.run_references != 0:
        checked, parity_failures = run_reference_audit(rows, args.run_references, args.workers)
        report["reference_reward_parity"] = {
            "checked": checked,
            "passed": checked - len(parity_failures),
            "failures": parity_failures,
        }

    report_path = args.report or args.dataset.with_suffix(args.dataset.suffix + ".reward_audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    parity_failures = report.get("reference_reward_parity", {}).get("failures", [])
    if failures or parity_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
