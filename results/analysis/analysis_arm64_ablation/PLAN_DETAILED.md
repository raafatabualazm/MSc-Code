# ARM64 cross-ISA ablation — detailed execution plan

Goal: turn the signature-leakage result from a Dart/x86-64 finding into a claim
about the task, replicated on a second ISA using 343 held-out functions from
real, obfuscated Flutter release APKs. This is the generalisation axis an
ICSE/FSE reviewer will demand.

---

## STEP 0 — BLOCKER: the ARM64 checkpoint is missing

Searched and **not found**:

| location | result |
|---|---|
| local workspace | no `adapter_model.safetensors` / `.bin` under any `*arm64*` path |
| runner 167.172.150.125 | no `arm64v21` directory |
| training pod 104.189.178.113 | no `arm64` artifacts |
| `exports/arm64_regions16_s42_gpt56_analysis_20260716.zip` | 51 entries, **0 weight files** — analysis bundle only |

Everything downstream depends on resolving this. Three paths:

### Path A — recover the checkpoint (cheapest, if it exists)
The 2026-07-16 run was on a pod that may since have been destroyed. Check any
surviving pod, cloud bucket, or external drive for
`qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_arm64v21_s42_prefix_no_gine_regions16`.
Only the LoRA adapter is needed (rank 64), not the base model.
**If found: skip to Step 2. Cost ~0.**

### Path B — retrain the graph model (not recommended)
Revives an architecture already abandoned (2026-07-22 pivot; VeRPO OOM'd on it
at >=200 basic blocks). Reintroduces a dependency you deliberately dropped.

### Path C — train T5Gemma 2 on ARM64 (RECOMMENDED)
Do not revive the graph stack. Train the **current** model on ARM64 and replicate
the x86 study exactly.

Why this is better science, not just convenience: using the old graph checkpoint
would leave **architecture confounded with ISA** — any x86/ARM64 difference could
be T5Gemma-vs-graph rather than x86-vs-ARM64. Training T5Gemma on ARM64 removes
that confound entirely and lets you say "same model, same tokenizer, same
harness, same interventions, two instruction sets."

It also means the whole measurement toolchain already works: the audit scripts,
the collapse checker, the seed-replication reporter, the scorer.

Cost: ~1,371 ARM64 training rows (1,714 total - 343 held out). x86 typed SFT took
348 updates on 2,775 rows, so ARM64 lands near ~170 updates at 2 epochs —
**order 2-4 GPU-hours**, plus data conversion (below).

---

## STEP 1 — Data conversion (Path C only)

ARM64 rows are **graphv2**; T5Gemma consumes **F2**. Fields available per row:
`assembly` (raw ARM64 text), `cfg` (blocks), `edges`, `dart_source`, `tests`,
`dart_function_signature`, `function`.

Required: a converter emitting F2-format rows from graphv2, mirroring
`multifunction_v1` construction. Validate with:

- byte-round-trip on 10 rows (F2 parse -> render -> identical)
- token-length histogram against the 32,768 source cap (ARM64 assembly is long;
  check the `>=200 instruction` tail specifically — that cliff was total on ARM64,
  0/168 solved, so truncation there would be silently catastrophic)
- gold round-trip: score gold `dart_source` through the evaluator, expect 343/343

**Gate:** if gold round-trip is not 343/343, stop. That is the Rank-0 check that
eliminated "the evaluator is broken" on x86 and it must pass here first.

---

## STEP 2 — Input views (BUILT AND VALIDATED)

`build_arm64_signature_views.py` emits four views, 343/343 rows, 0 failures,
verified: 0 rows retain the semantic name, assembly byte-identical to `exact`.

| view | prompt shows | rewritten? |
|---|---|---|
| `exact` | `String sortPasswordViolationsBySeverity(List<String> violations)` | no |
| `typed_opaque` | `String fn0(List<String> p0)` | yes (name -> fn0 across signature, source, tests) |
| `name_only` | name, no typed signature | no |
| `none` | neither | no |

`prompt_signature_mode` is read by `build_decoder_prompt`, which
`graph_inference_antigravity.py` imports directly — **the flag flows through
inference, verified.** For T5Gemma the equivalent hook is the typed-contract
path already used on x86.

Arity `{1:190, 2:101, 3:28, 4:14, 5:6, 6:2, 7:1, 11:1}`; modal return type
`bool` at 60/343, so guessing it is wrong 82.5% of the time.

---

## STEP 3 — The binary channel is ALSO already instrumented

`graph_inference_antigravity.py --graph_input_ablation` accepts
`none | null | cyclic_shift | matched_permutation | shuffle_blocks`.

- `matched_permutation` = the cross-task structure swap (the x86 `semantic_body_swap`)
- `null` = binary channel zeroed = "shown only the signature, never the binary"

**No new code.** `null` in particular replicates the paper's *second* independent
design (the signature-only control, x86: -0.0195, p=0.549) on a new ISA.

---

## STEP 4 — Factorial design

Signature view x binary channel. Seven cells worth running of twelve possible:

| # | signature | channel | what it measures |
|---|---|---|---|
| 1 | `exact` | full | baseline; must reproduce the published 5.54% |
| 2 | `typed_opaque` | full | **headline**: name removed, types kept |
| 3 | `name_only` | full | name without types |
| 4 | `none` | full | neither |
| 5 | `exact` | `null` | signature-only control (never shown the binary) |
| 6 | `exact` | `matched_permutation` | structure permuted, constants intact |
| 7 | `typed_opaque` | `null` | floor: no name, no binary |

Cell 1 doubles as a reproduction check on the published number. If it does not
land near 19/343, something in the pipeline differs from 2026-07 and everything
else is suspect.

### Tiered seeds (budget-aware)

| tier | cells | seeds | runs | ~GPU-h |
|---|---|---|---|---|
| 1 headline | 1, 2 | 42-46 (5) | 10 | 25 |
| 2 ladder | 3, 4 | 42-44 (3) | 6 | 15 |
| 3 channel | 5, 6, 7 | 42-43 (2) | 6 | 15 |
| | | | **22** | **~55** |

At 343 tasks x k=10 = 3,430 generations per run, ~150 min each (x86 was ~75 min
for 1,750). Scoring runs on the **CPU runner** (Dart 3.12.2 installed and
verified) in parallel with generation on the GPU pod.

---

## STEP 5 — Base-model gate (cheap, run before Tier 2/3)

On x86 the pre-SFT base scored **0 pass@1 / 0 pass@10 / 0 compile@10** — it could
not emit compilable Dart at all. Before spending on a base-model ladder, run
**cell 1 only, 1 seed, on the untrained base**.

- `compile@10` materially non-zero -> the full base ladder becomes very valuable
  (it would show the channel effect does not require task-specific training)
- `compile@10` ~ 0 -> drop the arm, report in one sentence

---

## STEP 6 — Measurement and decision rules (fixed now)

**Primary endpoint: pass@10.** compile@10 and pass@1 are secondary and cannot
carry a promotion on their own.

**Noise floor.** x86 measured SD 1.79 at 5.8 solved. Poisson-scaled to ARM64's 19
solved: **SD ~3.2 tasks (~0.9 pp)**. Any difference under ~3 tasks is reported as
within the floor, full stop.

**Power.** The expected name-removal effect is large: x86 went 42 -> 7 solved. The
ARM64 analogue (19 -> ~4) is ~4.7 SD, comfortably detectable. The study is
powered for the headline and **underpowered for subtle inter-cell differences** —
pre-commit to reporting only the former.

**Diversity guardrail.** Report mean distinct-candidates/10 per cell
(`analysis_rs_sft_fold/check_collapse.py`). It is far better powered than a
343-task binary count and catches degenerate decoding before it distorts pass@k.

**Statistics.** Exact McNemar on discordant tasks for each cell vs cell 1, plus
the seed SD. Report both; neither alone is sufficient.

---

## STEP 7 — Optional: constants-stripped (new code, ~half a day)

The x86 `constants_stripped` transform operates on the F2 constants prefix. The
ARM64 analogue must strip string and numeric literals from the `assembly` text.
This is the one intervention with no existing implementation.

Worth it because on x86 it was the arm that mattered: pass@10 7 -> 3 overall and
4 -> 0 on the 96 affected tasks, while permuting structure cost almost nothing.
If that ordering replicates on ARM64, "constant hygiene beats control-flow
obfuscation" becomes a two-ISA claim — and that is the sentence with real security
relevance.

Control to include: tasks with no literals must come back **byte-identical** to
baseline, exactly as the 79 unchanged x86 tasks did.

---

## Sequence

```
0. Resolve the checkpoint            <- BLOCKING. Path A if recoverable, else C.
1. (Path C) graphv2 -> F2 converter + gold round-trip gate 343/343
2. Build views                        DONE
3. Base-model probe (cell 1, 1 seed)  -> decides Tier 2/3 base arms
4. Tier 1: cells 1-2 x 5 seeds        -> headline + reproduction check
5. Tier 2: cells 3-4 x 3 seeds        -> full signature ladder
6. Tier 3: cells 5-7 x 2 seeds        -> binary-channel decomposition
7. Optional: constants-stripped
```

Stop after Tier 1 if cell 1 fails to reproduce ~19/343 — diagnose before
spending the remaining budget.

---

## Risks

| risk | detection | mitigation |
|---|---|---|
| checkpoint unrecoverable | Step 0 | Path C (recommended anyway) |
| F2 conversion silently truncates long ARM64 rows | token histogram vs 32,768 cap; `>=200`-instruction tail | fail-closed on any truncated row |
| cell 1 does not reproduce 5.54% | Tier 1 | halt; pipeline differs from 2026-07 |
| ARM64 pass@10 too low for the ladder to resolve | cells 3-4 near zero | report as a floor effect; the x86 ladder still carries the claim |
| rename inconsistency in `typed_opaque` | already checked: 0 leaks, tests rebound | none needed |
