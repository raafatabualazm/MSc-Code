"""Compatibility front end for the canonical repair-test split builder.

The implementation lives in ``scripts/evaluation/build_repair_test_split.py``.
This entry point keeps the concise data-oriented command while ensuring both
paths use the same balanced parser, input-group split, runtime validation, and
provenance manifest. By default, tasks whose original reference fails its full
test harness are excluded from the emitted pilot set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BUILDER = ROOT / "scripts/evaluation/build_repair_test_split.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def self_test() -> int:
    from scripts.evaluation.build_repair_test_split import split_harness
    from scripts.evaluation.repair_loop_antigravity import (
        _candidate_calls,
        validate_visible_test_boundary,
    )

    harness = """void main() {
  final candidate = square;
  expect(candidate(0), 0);
  expect(candidate(1), 1);
  expect(candidate(2), 4);
  expect(candidate(3), 9);
  expect(candidate(4), 16);
  expect(candidate(2), 2 + 2);
}

void expect(dynamic actual, dynamic expected) {
  if (actual is List && expected is List) {
    expect(actual[0], expected[0]);
  }
  if (actual != expected) throw '$actual != $expected';
}
"""
    visible, hidden, metadata = split_harness(
        harness,
        "self-test",
        seed=42,
        min_visible=2,
        min_hidden=3,
    )
    validate_visible_test_boundary(visible, hidden, "self-test")
    assert metadata["visible_input_groups"] == 2
    assert metadata["hidden_input_groups"] == 3
    assert metadata["total_cases"] == 6
    assert not (_candidate_calls(visible) & _candidate_calls(hidden))
    print(
        "self_test OK: canonical balanced parser, grouped duplicate inputs, "
        "2/3 minimums, helper exclusion, and boundary validation"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/testing/grpo_data_graphv2.jsonl",
    )
    parser.add_argument(
        "--out_scoring",
        type=Path,
        default=(
            ROOT
            / "data/testing/grpo_data_graphv2_repair_hidden_minvis2_minhid3_s42.jsonl"
        ),
    )
    parser.add_argument(
        "--out_visible",
        type=Path,
        default=(
            ROOT
            / "data/testing/grpo_data_graphv2_repair_visible_minvis2_minhid3_s42.jsonl"
        ),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_visible", type=int, default=2)
    parser.add_argument("--min_hidden", type=int, default=3)
    parser.add_argument("--dart", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include_reference_failures", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    manifest = args.manifest or args.out_scoring.with_suffix(".manifest.json")
    command = [
        sys.executable,
        str(CANONICAL_BUILDER),
        "--input",
        str(args.benchmark),
        "--visible_out",
        str(args.out_visible),
        "--hidden_out",
        str(args.out_scoring),
        "--manifest",
        str(manifest),
        "--seed",
        str(args.seed),
        "--min_visible",
        str(args.min_visible),
        "--min_hidden",
        str(args.min_hidden),
        "--drop_unsplittable",
        "--run_tests",
        "--timeout",
        str(args.timeout),
        "--workers",
        str(args.workers),
    ]
    if not args.include_reference_failures:
        command.append("--drop_reference_failures")
    if args.dart:
        command.extend(["--dart", args.dart])
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
