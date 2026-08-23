#!/usr/bin/env python3
"""Canonicalize and split synthetic Dart rows for leakage-safe SFT/GRPO.

The historical 1,726-row pool contains valid reference programs, but some test
rows call the reference function directly instead of the ``candidate`` alias.
Those rows pass the full reference harness while the per-test reward extractor
sees zero tests. This tool rewrites only assertion call sites inside ``main``,
fails closed on any remaining incompatibility, preserves CFG/assembly fields,
and creates source-grouped train/validation/test splits.

No evaluation rows are added. The 126-row and 154-row benchmark corpora must
remain outside these outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_ALIAS_RE = re.compile(r"\bfinal\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;")
_EXPECT_LINE_RE = re.compile(r"^\s*expect\s*\(")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalized_source(source: str) -> str:
    source = re.sub(r"//.*?$", "", source or "", flags=re.MULTILINE)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"\s+", " ", source).strip().lower()


def canonicalize_tests(row: dict[str, Any]) -> tuple[str, int]:
    tests = str(row.get("tests") or "")
    function = str(row.get("function") or "")
    alias = _ALIAS_RE.search(tests)
    if not function or not alias:
        raise ValueError("missing function or final candidate alias")
    if alias.group(1) != function:
        raise ValueError(f"candidate alias {alias.group(1)!r} != function {function!r}")

    lines = tests.splitlines()
    in_main = False
    depth = 0
    rewritten = 0
    assertions = 0
    output: list[str] = []
    direct_pattern = re.compile(
        rf"^(\s*expect\s*\(\s*){re.escape(function)}\s*\("
    )

    for line in lines:
        stripped = line.strip()
        if not in_main and re.match(r"^void\s+main\s*\([^)]*\)\s*\{", stripped):
            in_main = True
            depth = line.count("{") - line.count("}")
            output.append(line)
            continue

        if in_main:
            if _EXPECT_LINE_RE.match(line):
                assertions += 1
                new_line, count = direct_pattern.subn(r"\1candidate(", line, count=1)
                line = new_line
                rewritten += count
                if "candidate(" not in line:
                    raise ValueError(f"non-candidate assertion remains: {stripped[:160]}")
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                in_main = False
        output.append(line)

    if assertions < 4:
        raise ValueError(f"only {assertions} candidate assertions; require at least 4")
    canonical = "\n".join(output).rstrip() + "\n"
    extracted = sum(
        1
        for line in canonical.splitlines()
        if line.strip().startswith("expect(") and "candidate(" in line
    )
    if extracted != assertions:
        raise ValueError(f"reward extraction mismatch: assertions={assertions}, extracted={extracted}")
    return canonical, rewritten


def grouped_split(
    rows: list[dict[str, Any]], seed: int, train_fraction: float, validation_fraction: float
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = hashlib.sha256(normalized_source(str(row.get("dart_source") or "")).encode()).hexdigest()
        groups[key].append(row)

    items = list(groups.items())
    random.Random(seed).shuffle(items)
    targets = {
        "train": round(len(rows) * train_fraction),
        "validation": round(len(rows) * validation_fraction),
    }
    targets["test"] = len(rows) - targets["train"] - targets["validation"]
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}

    for _, group in items:
        deficits = {name: targets[name] - len(splits[name]) for name in splits}
        destination = max(deficits, key=lambda name: (deficits[name], -len(splits[name])))
        splits[destination].extend(group)

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: str(row.get("task_id") or row.get("filename") or ""))
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split_dir", type=Path, default=None)
    parser.add_argument("--split_prefix", default="synthetic_reward")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--drop_invalid", action="store_true",
                        help="Drop and report incompatible rows instead of aborting")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.validation_fraction < 0:
        raise SystemExit("invalid split fractions")
    if args.train_fraction + args.validation_fraction >= 1:
        raise SystemExit("train_fraction + validation_fraction must be < 1")

    rows = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    rewritten_rows = 0
    rewritten_calls = 0
    helper_counts: Counter[str] = Counter()

    for index, source_row in enumerate(rows):
        row = dict(source_row)
        try:
            row["tests"], changed = canonicalize_tests(row)
            rewritten_rows += int(changed > 0)
            rewritten_calls += changed
            seen_helpers = set()
            for line in row["tests"].splitlines():
                match = re.match(
                    r"^\s*(?:void|bool|dynamic|int|double|String|List(?:<[^>]+>)?|"
                    r"Map(?:<[^>]+>)?|Set(?:<[^>]+>)?)\s+(expect\w*)\s*\(",
                    line,
                )
                if match:
                    seen_helpers.add(match.group(1))
            helper_counts.update(seen_helpers)
            output.append(row)
        except Exception as exc:
            rejects.append(
                {
                    "index": index,
                    "task_id": row.get("task_id"),
                    "function": row.get("function"),
                    "error": str(exc),
                }
            )

    if rejects and not args.drop_invalid:
        raise SystemExit(
            f"Rejected {len(rejects)} rows; first failures: {json.dumps(rejects[:5], indent=2)}"
        )

    write_jsonl(args.output, output)
    report: dict[str, Any] = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output),
        "rewritten_rows": rewritten_rows,
        "rewritten_calls": rewritten_calls,
        "rejected_rows": len(rejects),
        "rejects": rejects,
        "helper_definition_rows": dict(sorted(helper_counts.items())),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }

    if args.split_dir:
        splits = grouped_split(output, args.seed, args.train_fraction, args.validation_fraction)
        report["splits"] = {}
        for name, split_rows in splits.items():
            path = args.split_dir / f"{args.split_prefix}_{name}.jsonl"
            write_jsonl(path, split_rows)
            report["splits"][name] = {
                "path": str(path),
                "rows": len(split_rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
