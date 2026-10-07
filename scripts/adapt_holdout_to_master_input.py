#!/usr/bin/env python3
"""Adapt the s44 holdout (fresh_eval schema) into the master build script's
input schema so build_scrubbed_dataset.py can regenerate its assembly/CFG on
Dart 3.12.2 with objdump + the master extractor + fn0 neutralization.

Holdout row -> master input row:
  task_id            -> id
  function           -> function        (the semantic name to neutralize away)
  dart_source        -> dart_source
  tests (string)     -> tests = {kind: 'dart_harness', harness: <string>}

The tests harness already binds `final candidate = <semanticName>;`; the master
dart_harness path renames <semanticName> -> candidate, matching the neutralized
source. Semantics/tests are preserved; only the binary representation is rebuilt.
"""
import json, sys

inp, outp = sys.argv[1], sys.argv[2]
n = 0
with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        harness = r.get("tests") or ""
        if not harness.strip():
            continue
        adapted = {
            "id": r.get("task_id"),
            "function": r.get("function") or r.get("camel_case_function_name"),
            "dart_source": r.get("dart_source"),
            "tests": {"kind": "dart_harness", "harness": harness},
        }
        g.write(json.dumps(adapted, ensure_ascii=False) + "\n")
        n += 1
print(f"adapted {n} rows -> {outp}")
