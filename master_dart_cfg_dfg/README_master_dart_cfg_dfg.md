# Master Dart/Flutter CFG+DFG Dataset

## Outputs

- `master_dart_cfg_dfg_train.jsonl`: deduplicated training split.
- `master_dart_cfg_dfg_heldout.jsonl`: deduplicated held-out split built from the three test-set variants.
- `master_dart_cfg_dfg_rejected.jsonl`: duplicate/conflict decisions with kept/rejected IDs.
- `master_dart_cfg_dfg_manifest.json`: input/output hashes, counts, leakage checks, graph and test statistics.
- `master_dart_cfg_dfg_sample.jsonl`: seven-row inspection sample.

## Canonical row schema

Every row uses `antigravity-master-dart-cfg-dfg-v1` and contains:

- `assembly`: original AOT disassembly text.
- `dart_source`: target Dart/Flutter source.
- `cfg`: normalized basic blocks with stable positional IDs.
- `edges`: one fixed edge schema. `edge_family=control` represents CFG edges; `edge_family=data` and `edge_type=dataflow` represent reaching-definition dependencies. Data edges retain `locations` and `dependency_count`.
- `tests`: one fixed `antigravity-behavior-tests-v1` object. Existing harnesses use `kind=dart_harness`; generated suites use differential reference oracles.
- `fingerprints`: normalized source, assembly, and graph SHA-256 values.
- `provenance`: original dataset and row identity.

## Deduplication and split safety

Training rows are deduplicated first by normalized assembly input and then by normalized source. Same-assembly/different-source groups are treated as conflicting labels: one deterministic best-quality representative is retained and every removed row is recorded in the rejection file. The held-out test files are treated as three representations of the same tasks, not three additional training datasets.

## Test validation

The build performs structural/static validation for every suite. The current execution environment does not contain a Dart or Flutter executable, so runtime validation is deliberately recorded as `not_run`, never as passed. Run:

```bash
python validate_master_dart_tests.py \
  --input master_dart_cfg_dfg_train.jsonl \
  --output master_dart_cfg_dfg_train.runtime_validated.jsonl \
  --dart /path/to/dart
```

The validator executes provided harnesses and differential reference oracles, checks determinism, filters invalid generated stdin cases, and writes per-row runtime results.
