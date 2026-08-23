# master_dart_cfg_dfg — scrub + suitability for hybrid v2.1 training (2026-07-17)

## TL;DR
- **Leakage vs frozen 154-task HumanEval benchmark: 0** (independently verified;
  the manifest's own leakage check was only train-vs-`test_set_cfg_clean`, a
  different held-out set — so this was checked separately and is clean).
- **Do not hand-roll a scrubber.** The hybrid patch's Phase-0
  (`prepare_hybrid_training_data_antigravity.py`) already does the fn0 scrub
  across source/signature/assembly/CFG-symbols/tests, with dual-fingerprint
  leakage rejection, FACTS extraction, and feedback/acceptance partition. It is
  stronger than the v3 builder. Route master through it.
- **Master is NOT plug-compatible.** Its `tests` is a structured
  `antigravity-behavior-tests-v1` dict; Phase-0 wants an executable Dart harness
  STRING. And 30% of rows are the wrong task shape. An adapter is required first.

## Composition of `master_dart_cfg_dfg_train.jsonl` (2,474 rows)
| Test kind | Rows | Shape | fn0-suitable? |
|---|---:|---|---|
| `dart_harness` | 1,725 | function-level, typed sig, `expect(candidate(...))` | **YES** |
| `differential_program` | 720 | whole-program `void main()`, stdout oracle | NO |
| `differential_function` | 29 | function-level but **untyped** (`gcd(u,v)`), oracle | NO |

The 749 non-suitable rows are the `dart_all` half. They cannot use an fn0 typed
function contract (renaming `main`→`fn0` breaks the entrypoint; untyped sigs have
no types to expose/withhold), and their reference-oracle stdout comparison is an
execution model the hybrid assertion-harness evaluator does not run.

## What was produced (this session, CPU-only, no Dart)
1. `adapt_master_to_hybrid.py` — filters to the 1,725 `dart_harness` rows and
   lifts `tests.harness` (string) into the Phase-0 `tests` field. Records every
   drop with a reason.
   - Output: `master_hybrid_input.jsonl` (1,725 rows).
   - Dropped: `master_hybrid_dropped.jsonl` (749 = 720 program + 29 untyped).
2. Scrub validation on a sample (direct call into `hybrid_data_controls`):
   `nearestDollarFirst` -> `int fn0(List<int> a)`; **no semantic-name leak** in
   source, signature, or tests; FACTS_JSON extracted from the binary; dual
   fingerprints (`neutral_sha256`, `alpha_structural_sha256`) present; test
   partition works.

## Effective training size (the number that matters)
Length routing (plan's <=150-instruction one-shot cap) over the 1,725 suitable rows:
| Bucket | Rows | Use |
|---|---:|---|
| short (<=150 instr) | **663** | one-shot RS-SFT / SFT training pool |
| bridge (151–199) | 270 | holdout (separate hierarchical track) |
| long (>=200) | 792 | holdout (length-cliff track) |

**So the usable supervised training pool is ~663 rows, not 2,474.** 46% of the
corpus (792 rows) is in the >=200-instruction cliff regime that the plan does not
claim to solve one-shot. The 663 is modest but clean/typed/verified/scrubbed;
it is the gold pool that Phase-2 RS-SFT (k=16 rejection sampling) amplifies.

## Remaining step (requires the Dart box)
Phase-0 **fails closed** without a Dart SDK: `--skip_reference_execution` marks
references as unverified (not passed), approving 0 rows. That is correct — no
row trains until its reference is proven to pass its own scrubbed harness. Run
the real Phase-0 on `167.172.150.125` (Dart 3.11.5) or the pod (3.12.2):

```bash
python -m scripts.training.prepare_hybrid_training_data_antigravity \
  --input master_hybrid_input.jsonl \
  --output phase0/approved_short.jsonl --bridge_output phase0/bridge.jsonl \
  --long_output phase0/long.jsonl --rejected_output phase0/rejected.jsonl \
  --report phase0/report.json \
  --forbidden_eval <all frozen-154 eval files> \
  --neutral_name fn0 --max_instructions 150
```
Expect the approved count to drop below 663 (the master build never executed the
generated harnesses — all `runtime_status: not_run` — so some will fail replay).

## Minor bug to flag to the patch author (not a blocker)
`mechanical_facts` put register-indirect operands into `direct_callees`
(`"QWORD PTR [r14"`, `"QWORD PTR [rax"`). The README says FACTS excludes
runtime-layout numbers; the callee extractor should also reject `PTR`/register
memory operands so FACTS callees are real symbol names only.
