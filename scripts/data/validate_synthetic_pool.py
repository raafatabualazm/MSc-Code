"""Validate a synthetic (asm, dart_source, tests) pool before it is used.

Checks, in order of severity:
  1. Schema: required keys per row; reports key-set variants.
  2. Uniqueness: task_id / filename / normalized-source duplicates.
  3. Conventions: tests contain the REQUIRED 'final candidate = NAME;'
     line and NAME == row['function']; tests define their own expect
     helper; dart_source contains the function; assembly has the GDB
     header and mentions the function.
  4. Leakage vs the 154-task eval set (function names + normalized
     sources vs grpo_data.jsonl) - the synthetic pool's whole value is
     being confound-free.
  5. FUNCTIONAL GATE (--run_tests N, -1 = all): execute every sampled
     row's dart_source against its own tests with the SAME harness the
     reward/eval uses (run_dart_tests from the reranker; local Dart).
     References that fail their own tests are unusable rows.

Usage:
  python scripts/data/validate_synthetic_pool.py --pool data/datasets/synthetic_pool.jsonl --run_tests 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.rerank_predictions_antigravity import (  # noqa: E402
    _normalize_code,
    _resolve_dart_binary,
    run_dart_tests,
    validate_dart_binary,
)

REQUIRED_KEYS = [
    "filename", "function", "dart_function_signature", "dart_source",
    "assembly", "lang", "task_id", "tests",
]

_CANDIDATE_RE = re.compile(r"\bfinal\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--eval_data", default="data/testing/grpo_data.jsonl", type=Path)
    parser.add_argument("--run_tests", type=int, default=0,
                        help="N rows to functionally test (-1 = all, 0 = skip)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems: list[str] = []

    # 1. schema
    key_sets = Counter(tuple(sorted(r.keys())) for r in rows)
    missing_required = [
        (i, [k for k in REQUIRED_KEYS if k not in r])
        for i, r in enumerate(rows)
        if any(k not in r for k in REQUIRED_KEYS)
    ]
    if missing_required:
        problems.append(f"{len(missing_required)} rows missing required keys, first: {missing_required[:3]}")
    if len(key_sets) > 1:
        sets = list(key_sets.items())
        base = set(sets[0][0])
        for keys, count in sets[1:]:
            diff_plus = sorted(set(keys) - base)
            diff_minus = sorted(base - set(keys))
            print(f"key-set variant ({count} rows): +{diff_plus} -{diff_minus}")

    # 2. uniqueness
    dup_ids = [k for k, c in Counter(r.get("task_id") for r in rows).items() if c > 1]
    dup_files = [k for k, c in Counter(r.get("filename") for r in rows).items() if c > 1]
    norm_counter = Counter(_normalize_code(r.get("dart_source", "")) for r in rows)
    dup_sources = sum(c - 1 for c in norm_counter.values() if c > 1)
    if dup_ids:
        problems.append(f"duplicate task_ids: {len(dup_ids)} (e.g. {dup_ids[:5]})")
    if dup_files:
        problems.append(f"duplicate filenames: {len(dup_files)}")
    if dup_sources:
        print(f"note: {dup_sources} rows are normalized-source duplicates of another row")

    # 3. conventions
    bad_candidate_line = []
    bad_expect = []
    bad_source_name = []
    bad_assembly = []
    expect_counts = []
    for i, r in enumerate(rows):
        tests = r.get("tests", "")
        match = _CANDIDATE_RE.search(tests)
        if not match or match.group(1) != r.get("function"):
            bad_candidate_line.append(i)
        if "void expect(" not in tests:
            bad_expect.append(i)
        expect_counts.append(tests.split("void expect(", 1)[0].count("expect("))
        if r.get("function", "") not in r.get("dart_source", ""):
            bad_source_name.append(i)
        asm = r.get("assembly", "")
        valid_header = asm.startswith("All functions matching regular expression") or \
            asm.startswith("Dump of assembler code for function")
        if not valid_header or r.get("function", "") not in asm:
            bad_assembly.append(i)
    for name, bad in [("candidate-line/function mismatch", bad_candidate_line),
                      ("missing self-contained expect helper", bad_expect),
                      ("function name absent from dart_source", bad_source_name),
                      ("assembly header/function check failed", bad_assembly)]:
        if bad:
            problems.append(f"{name}: {len(bad)} rows (first: {bad[:5]})")

    # 4. leakage vs eval set
    eval_rows = [json.loads(line) for line in args.eval_data.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_funcs = {r.get("function") for r in eval_rows}
    eval_norms = {_normalize_code(r.get("dart_source", "")) for r in eval_rows}
    func_overlap = [r.get("function") for r in rows if r.get("function") in eval_funcs]
    norm_overlap = sum(1 for r in rows if _normalize_code(r.get("dart_source", "")) in eval_norms)
    if func_overlap:
        print(f"LEAKAGE WARNING: {len(func_overlap)} rows share a function name with the eval set: {sorted(set(func_overlap))[:10]}")
    if norm_overlap:
        problems.append(f"LEAKAGE: {norm_overlap} rows have normalized sources identical to eval references")

    print(f"rows={len(rows)} | dup_ids={len(dup_ids)} | expects/task: med {sorted(expect_counts)[len(expect_counts)//2]} min {min(expect_counts)}")
    print(f"convention failures: candidate_line={len(bad_candidate_line)} expect_helper={len(bad_expect)} "
          f"source_name={len(bad_source_name)} assembly={len(bad_assembly)}")
    print(f"eval-set leakage: function-name overlap={len(func_overlap)} identical-source={norm_overlap}")

    # 5. functional gate
    test_results = None
    if args.run_tests != 0:
        sample = rows if args.run_tests < 0 else rows[: args.run_tests]
        dart_bin = _resolve_dart_binary(None)
        validate_dart_binary(dart_bin)
        print(f"functional gate: running {len(sample)} references against their own tests ({args.workers} workers)...")
        passed, failures = 0, []

        def run_one(idx_row):
            idx, row = idx_row
            ok, diag = run_dart_tests(row["dart_source"], row["tests"], str(row.get("task_id", idx)), dart_bin, args.timeout)
            return idx, row.get("task_id"), ok, diag

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, (i, r)) for i, r in enumerate(sample)]
            done = 0
            for future in as_completed(futures):
                idx, task_id, ok, diag = future.result()
                done += 1
                if ok:
                    passed += 1
                else:
                    failures.append({"index": idx, "task_id": task_id, "diagnostic": diag[:300]})
                if done % 100 == 0:
                    print(f"  [{done}/{len(sample)}] passing so far: {passed}")
        rate = passed / max(len(sample), 1)
        print(f"FUNCTIONAL: {passed}/{len(sample)} references pass their own tests ({rate:.1%})")
        if failures:
            print(f"first failures: {json.dumps(failures[:5], indent=2)}")
        if rate < 0.95:
            problems.append(f"functional pass rate {rate:.1%} below 95% - clean the failing rows before use")
        test_results = {"sampled": len(sample), "passed": passed, "failures": failures}

    verdict = "OK" if not problems else "PROBLEMS"
    print(f"\nVERDICT: {verdict}")
    for p in problems:
        print(f"  - {p}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "pool": str(args.pool), "rows": len(rows), "problems": problems,
            "functional": test_results,
        }, indent=2), encoding="utf-8")
        print(f"report: {args.report}")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
