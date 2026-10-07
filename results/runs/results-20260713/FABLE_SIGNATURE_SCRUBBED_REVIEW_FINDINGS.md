# Fable Review — Regions16 Signature-Scrubbed Evaluation

Review of the bundle indexed by `FABLE_SIGNATURE_SCRUBBED_REVIEW.md`. Performed 2026-07-16
against the locally synced artifacts; all checks re-executed independently (hashes recomputed,
metrics re-derived from per-task CSVs, generations re-audited, and 260 Dart programs re-executed
through the project's own `evaluate_dart_jit_tests_detail` on the pinned Dart 3.11.5 SDK).

## Verdict

**The evaluation is sound and the headline result is real.** Every integrity claim in the index
verifies; the paired table reproduces to four decimals; the harness was demonstrably passable
(151/154 neutral references compile, 142/154 pass), so the 0/1,540 outcome is authentic model
behavior, not a pipeline defect. Findings below are protocol-hygiene and reuse caveats, not
threats to this run's conclusion.

## Confirmations (all green)

| Check | Result |
|---|---|
| Manifest hashes (11 locally verifiable of 12; checkpoint is pod-side, matches log + provenance) | all match |
| Public/private dataset SHA-256 vs index | match |
| Rows/candidates: 154 x 10 in raw and scored; 0 rejects at build | confirmed |
| Candidate-stream digest `b4c0be6e...` identical raw vs scored | confirmed |
| Provenance chain: dataset binding, checkpoint binding, `scoring_tests_visible_to_policy:false`, join input hashes | closes end-to-end |
| Public-row neutrality (no private fields, `candidate` name, empty signature, `name_only`) | 154/154 |
| Leak scan: all 154 original identifiers vs full public rows | 0 leaks (669 regex hits are all the mnemonic `add` inside instruction text) |
| Test rewrite: alias handling, shadowing, dangling `implementation(` | clean; 152/154 alias-style, 0 defects |
| Neutral `evaluation_only_dart_function_signature` == rename(original signature) | 154/154 |
| Comparator table (pass@1/5/10, compile@1/5) recomputed from per-task CSV | 17.6623 / 24.1136 / 27.2727 / 77.0779 / 93.8028 — exact |
| Paired counts | 42 losses + 112 ties (pass), 147 + 7 (compile) — exact |
| Metric-mode parity: both arms scored with `--compile_mode jit_tests` (aligned-JIT classifier) | confirmed (scrubbed log + comparator sweep JSON) |
| Scrubbed stats CSV all-zero compile/pass across 154 rows | confirmed |
| Gold check: neutral references under rewritten tests (local pinned Dart 3.11.5) | 151/154 compile, 142/154 pass → **the 0% ceiling was reachable; failure is the model's** |
| Re-execution of all 106 candidate-defining generations | 0 pass; 0 true compiles (see F4) |

## Findings (ranked)

**F1 — medium (validity for future reuse): the scrubbed evidence channel is not input-identical
to the comparator's.** The rebuild ran on the pod (build log 04:29Z) whose `dart --version` is
3.12.2, while the frozen benchmark assembly comes from the pinned 3.11.5 pipeline; additionally
the graph rows were built with a different extractor revision (`7a89b10f...` vs the benchmark's
`3b522afc...`). Result: 14/154 tasks differ in CFG block count and 13/154 in instruction count —
e.g. `sumOddInEvenPosition` 13→74 instructions (+469%), `specialFilter` 128→185, `maximum` 15→38.
This does not affect this run's conclusion (the collapse is prompt-side), but the frozen scrubbed
set must **not** be treated as "identical except the signature" in decomposition arms that
attribute deltas to signature components. Rebuild pinned (Dart 3.11.5 / GDB 17.1, benchmark
extractor revision) or document the drift in the index.

**F2 — low (provenance gap): `benchmark_protocol.assembly_build` records block/edge/instruction
counts but no Dart/GDB/extractor version stamps.** This is exactly why F1 is inferable only from
circumstantial evidence. Add toolchain version fields to `assembly_build` in
`build_signature_scrubbed_eval.py`.

**F3 — low (reuse hygiene): the public file carries recoverable fingerprints of withheld
secrets.** Public rows keep `semantic_function_name_sha256` (unsalted SHA-256 of the lowercased
withheld name — dictionary-attackable given the benchmark's public name list),
`original_source_sha256`, and the row order is 1:1 with the original benchmark file (verified
154/154). Harmless for this offline policy (the prompt builder never touches
`benchmark_protocol`), but the public file as-is is **not** safe input for tool-using or
retrieval-capable external baselines (P8): a frontier agent could deanonymize every task by
position or dictionary. Strip/salt these fields and shuffle rows for any external release.

**F4 — low (harness edge): Dart front-end crashes are classified as "compiled".**
`_is_dart_jit_static_error` does not match `Crash when compiling:` diagnostics, so a compiler
crash (nonzero exit, non-static-pattern output) falls through to compiled-but-failing. Observed
locally on Dart 3.11.5: 3/106 candidate re-executions (`RangeError ... growable_array` front-end
crash) count as compiled. The pod run (3.12.2) was unaffected — compile@k there is exactly 0 —
but the classifier should treat `Crash when compiling` as a static failure. One-line regex fix in
`graph_compile_at_k_antigravity.py`.

**F5 — info (benchmark debt, pre-existing): 3 neutral references fail to compile, while 4
tasks reject a bare contract-compatible candidate stub; 12 references fail overall.**
`intersection` (orig 127: tests pass Dart records to `List<int>`), `strongestExtension` (orig
153: tests alias the undefined `Strongest_Extension`), and `solve` (orig 161: unescaped `$a`
interpolation in a test string) are reference compile failures. `sumOddInEvenPosition` (orig
121) is the fourth stub failure because its tests bind an undefined `solution` wrapper. These
are zero-compile ties on the comparator too: inherited legacy defects, not scrub regressions.
Nine additional references compile but fail their own tests. The builder's
keep-the-denominator policy is reasonable, but results should also report the valid-150
sensitivity and preserve an exact known-defect list so nobody re-diagnoses these.

**F6 — info (index doc nits).** (a) The static-audit prose doesn't sum: "1,415 lacked ... 120
declared" leaves 5 candidates unaccounted (1,540 − 1,415 = 125); presumably 5 declared
`candidate(` non-top-level — state it. (b) Audit counts are definition-sensitive: an independent
strict recount (harness `_extract_code`, top-level declarations with a body) gives 106 declaring /
18 arity-matched (vs 120 / 22), fences 1,531 (exact match), package-imports/Flutter/`class
Candidate` within ±10. Same conclusions; pin the counting script if the numbers will be cited.
(c) Comparator `_pass_stats.csv` and `_compile_stats.csv` are byte-identical duplicates of one
file — harmless, worth a note.

**F7 — interpretation (corrected after auditing the actual SFT rows).** The caution correctly
notes that name, return type, parameter types, and arity were removed simultaneously. The earlier
claim that all 770 SFT rows carried exact signatures was wrong: **0/770 rows contain a signature
field**; the prompt's function-name branch was used, with `name == main` on 741 rows and
`name == candidate` on none. Therefore the literal name-only branch was not unseen. The relevant
format/lexical shift was from a training distribution dominated by `main` programs to the
semantically loaded identifier `candidate`, combined with the old protocol's missing exact-name,
top-level, anti-demo, and signature-inference constraints. Independently, this arm withholds assembly
text from the decoder prompt (`GRAPH_PROMPT_ASSEMBLY_MODE=none`), so after scrubbing the only
task-specific binary signal is the 64 gated soft-prefix vectors. The defensible conclusion is
dependence on the visible signature contract **for this arm and protocol**; it does not establish
that the graph channel contains zero usable evidence. A matched opaque-name/typed-signature arm,
an opaque name-only arm with the same structural output contract, or a full-assembly-text arm are
the appropriate decompositions.

## Addendum — what compile@k actually compiles (and the no-tests counterfactual)

"Aligned JIT compile@k" is **not** standalone compilation of the generation. In `jit_tests` mode
(used by both arms) the harness builds one program = extracted candidate code (imports re-added,
`void main` stripped) + the hidden test driver, then runs `dart run`; front-end diagnostics count
as compile failure. The tests call `candidate(...)` with hidden caller shapes, so under
signature scrubbing the metric requires top-level symbol and caller compatibility (including
name/arity and the types statically constrained by those callers), not just Dart
well-formedness. It does not prove exact ABI recovery. That is what makes exactly-zero reachable.
Static kill-paths over the 1,540
scrubbed candidates: 1,522 lack a compatible top-level `candidate(...)`; 1,044 re-add
unresolvable `package:` imports; 43 leave an unstripped non-`void` `main()` that collides with
the test driver's. Only 14 candidates avoid all three, and re-executing them shows all still fail
on type-level front-end errors — 0 true compiles, matching the pod result.

Measured counterfactual (legacy standalone compile, candidate #1 per task, local Dart 3.11.5):
**scrubbed standalone-compile@1 = 14/154 = 9.1%** (failures: 83 other static errors, 56
unresolvable package imports), comparator = 52.6% under the same stricter standalone/AOT mode
(not comparable to its 77.08% jit_tests figure). So the scrubbed model does emit *some*
syntactically valid Dart; the 0.00% is specific to the with-tests metric. Recommend one sentence
in the index stating that compile@k is candidate+tests (interface-inclusive), optionally
reporting standalone compile as a secondary diagnostic so readers do not conclude the model lost
the ability to write Dart at all.

## Corrective experiment status (v3)

The first v2 repair was rejected after review because it retained `candidate`, used different
public permutations across arms, did not preserve all graph metadata, and lacked a matched
same-SDK comparator rescore. The replacement v3 protocol is documented in
`RUNBOOK_FIXED_SCRUB_V3.md` and is queued behind an unrelated GPU job; no v3 model result is
claimed yet.

The final 56-file GPU bundle (`918e1caeâ€¦`) verifies 56/56, passes shell syntax validation and
106/106 tests in a faithful staging-only Linux overlay, and independently verifies the frozen
154x10 comparator pool against prompt stream `55adb80eâ€¦`. The corrected waiter is PID 2286577;
it has not deployed canonical files while the ARM64 job remains active.

V3 uses opaque `fn0`, one shared task permutation, frozen-assembly symbol renaming, identical
structural output constraints, and two arms: name-only (signature withheld) and neutral-exact (typed
signature, neutral parameter names). Its rebuilt 154-row pair passes exact frozen CFG/edge
parity, public/private hygiene including helper symbols, prompt-digest locks, Dart 3.11.5/GDB
17.1 provenance, contract stubs 150/154, and neutral references 151 compile / 142 pass. The
current exact renderer reproduces the frozen comparator prompt digest `55adb80e…` byte-for-byte;
the comparator will therefore be rescored rather than regenerated. All new arms and the frozen
comparator use the patched aligned-JIT classifier and the same pinned Dart 3.11.5 SDK. A separate
candidate-only legacy wrapped standalone-AOT acceptance@1/5/10 diagnostic is also sealed for all
three pools so the primary candidate-plus-callers metric is not mistaken for standalone syntax
compilation.

## Recommended actions

1. Rebuild the scrubbed set on the pinned toolchain + benchmark extractor revision before any
   decomposition arms (F1), stamping toolchain versions in `assembly_build` (F2).
2. Patch `_is_dart_jit_static_error` to classify `Crash when compiling` as static failure (F4).
3. For external/frontier reuse of the public file: strip `semantic_function_name_sha256` /
   `original_source_sha256`, shuffle rows, re-salt IDs (F3).
4. Correct the SFT-distribution statement as in F7, use a structurally matched opaque-name
   protocol for the decomposition, and record the exact known-defect task list (F5).
