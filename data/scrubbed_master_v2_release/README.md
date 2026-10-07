# Master Dart CFG+DFG — Signature-Scrubbed HumanEval-Style Training Set

## Release result

- Input rows: **2474**
- Final retained rows: **2211**
- Quarantined rows: **263**
- Former `main` rows converted to callable `candidate`: **495**
- Provided assertion-harness rows retained: **1703**
- Generated function-oracle rows retained: **13**

## Real Dart validation

Every retained row passed all three gates with:

`Dart SDK version: 3.12.2 (stable) (Tue Jun 9 01:11:39 2026 -0700) on "linux_x64"`

1. `dart run` of the rewritten source plus its tests.
2. `dart compile aot-snapshot` using the actual Dart AOT compiler.
3. `dartaotruntime` execution of that snapshot.

The function assembly, CFG, and DFG were regenerated from the rewritten neutral function. Original `main` assembly and graphs were not reused.

## HumanEval-style transformation

- The exposed target is always named `candidate`.
- Main-program logic is moved into a top-level callable `candidate(...)` function.
- The private tests call `candidate`; no training target contains a top-level `main`.
- Programs requiring process-global `stdin`, `stdout`, `stderr`, `exit`, or subprocess behavior are quarantined because they cannot be converted into a strict direct-call unit-test task without changing semantics.
- Comments are removed from target source before compilation to reduce semantic leakage.

## Public/private split

`master_dart_graphv2_signature_scrubbed_public.jsonl` matches the supplied public reference schema. It exposes only neutral task identity, assembly, CFG/DFG, graph integrity, and protocol metadata. It withholds source, tests, expected values, signature details, semantic function names, original task IDs, and reasoning.

`master_dart_graphv2_signature_scrubbed_private.jsonl` matches the supplied private reference schema and adds the withheld neutralized source, evaluation-only signature, and executable Dart test harness.

## Audit result

The independent audit passed with:

- **2211** aligned public rows, private rows, and compile-ledger records.
- **109,608** CFG blocks.
- **153,667** control-flow edges.
- **201,798** data-flow edges.
- Zero forbidden public fields.
- Zero residual temporary `c_<hash>` symbols.
- Zero top-level `main` declarations in private target source.
- Zero duplicate retained source or assembly fingerprints.
- Three fresh post-release JIT/AOT recompilation samples passed.

See `master_dart_graphv2_quarantine.jsonl` for every rejected row and its reason, and `master_dart_graphv2_compile_ledger.jsonl` for retained-row evidence.
