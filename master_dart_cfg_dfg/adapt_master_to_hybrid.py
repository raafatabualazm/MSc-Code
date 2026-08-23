#!/usr/bin/env python3
"""Adapt master_dart_cfg_dfg rows into the hybrid Phase-0 input schema.

The hybrid Phase-0 (`prepare_hybrid_training_data_antigravity.py`) expects, per
row: `dart_source`, a typed `dart_function_signature`, `assembly`, `cfg`,
`edges`, and `tests` as an *executable Dart harness STRING* (it partitions the
assertions itself). The master schema instead nests the harness under a
structured `tests` dict (`antigravity-behavior-tests-v1`) and mixes three task
shapes. This adapter selects only the rows that fit the fn0 function-contract
model and lifts them into the expected shape.

Kept  : `dart_harness` rows (function-level, typed signature, `expect()` harness).
Dropped: `differential_program` (whole-program `void main()`, stdout oracle) and
         `differential_function` (untyped, reference-oracle, no harness string) —
         wrong task shape for fn0 and an execution model Phase-0 does not run.

Every drop is recorded with a reason. This does NOT scrub; Phase-0 does the fn0
neutralization, FACTS, partition, leakage rejection, and reference replay.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def instr_count(row: dict) -> int:
    return sum(len(b.get("instructions", [])) for b in (row.get("cfg") or []))


def typed_signature(sig: str) -> bool:
    """A signature Phase-0 can neutralise: has a return type AND a name(...) form.

    `int gcd(int u, int v)` -> ok; `gcd(u,v)` (untyped) -> not ok;
    `void main()` -> ok syntactically but excluded upstream by shape.
    """
    m = re.match(r"^\s*[A-Za-z_][\w<>,\[\]?\s]*\s+[A-Za-z_]\w*\s*\(", sig or "")
    return bool(m)


def adapt(row: dict) -> tuple[dict | None, str]:
    tests = row.get("tests") or {}
    kind = tests.get("kind")
    if kind != "dart_harness":
        return None, f"unsupported_test_kind:{kind}"
    fn = str(row.get("function") or "")
    if fn == "main" or not fn:
        return None, "whole_program_or_missing_function"
    sig = str(row.get("dart_function_signature") or "")
    if not typed_signature(sig):
        return None, "untyped_or_unparseable_signature"
    harness = tests.get("harness")
    if not isinstance(harness, str) or "expect(" not in harness:
        return None, "no_executable_expect_harness"
    out = {
        "task_id": row.get("id"),
        "function": fn,
        "dart_function_signature": sig,
        "dart_source": row.get("dart_source"),
        "assembly": row.get("assembly"),
        "cfg": row.get("cfg"),
        "edges": row.get("edges"),
        "graph_v2": row.get("graph_v2"),
        # Phase-0 consumes `tests` as the full harness string and splits it.
        "tests": harness,
        "provenance": {
            **(row.get("provenance") or {}),
            "master_id": row.get("id"),
            "master_schema": row.get("schema"),
            "master_test_kind": kind,
            "parsed_instruction_count": instr_count(row),
        },
    }
    return out, "kept"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dropped", required=True, type=Path)
    args = ap.parse_args()

    kept = 0
    reasons: dict[str, int] = {}
    with args.input.open(encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout, \
         args.dropped.open("w", encoding="utf-8") as fdrop:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            adapted, reason = adapt(row)
            if adapted is not None:
                fout.write(json.dumps(adapted, ensure_ascii=False) + "\n")
                kept += 1
            else:
                fdrop.write(json.dumps({"id": row.get("id"), "reason": reason}) + "\n")
            reasons[reason] = reasons.get(reason, 0) + 1

    summary = {"kept": kept, "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}
    print(json.dumps(summary, indent=2))
    (args.output.with_suffix(args.output.suffix + ".adapt_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
