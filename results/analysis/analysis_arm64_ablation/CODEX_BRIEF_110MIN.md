# ARM64 cross-ISA headline — 110-minute paired run

Budget: **110 minutes of GPU wall clock**, starting when the contract-only seed 43
run reaches `EXITED`. Scoring is CPU and does not consume this budget.

This is not the tiered study in `PLAN_DETAILED.md`. That needs ~55 GPU-hours. This
is the single highest-value cell of it, designed so that **running out of time
still leaves a complete, publishable result** rather than a truncated arm.

---

## Objective

One comparison, paired per task, on the frozen ARM64 checkpoint:

| view | prompt shows | |
|---|---|---|
| `exact` | `String sortPasswordViolationsBySeverity(List<String> violations)` | name present |
| `typed_opaque` | `String fn0(List<String> p0)` | name removed, types kept |

Endpoint: **pass@10**, seed 42, k=10. Everything else is secondary.

`exact` doubles as the reproduction check on the published 5.54% / 19-of-343.

---

## Hard gates — abort rules, read these first

| time | gate | action if failed |
|---|---|---|
| T+15 | checkpoint downloaded, sha256 verified | abort, report |
| T+35 | 5-task smoke has produced scored output on `exact` | **abort the whole run**, report environment blocker |
| T+40 | measured throughput extrapolated, scope chosen | — |
| T+105 | stop generation wherever it is | score what exists |

The T+35 abort is not optional. The dominant risk here is environment (PyTorch
Geometric, model download, repo state), not science. If it has not produced
tokens by T+35 there is no path to a usable result and continuing wastes money.
Report the blocker; it is a legitimate outcome.

---

## Already done — do not redo

- `build_arm64_signature_views.py` is built and validated: 343/343 rows, 0 failures,
  0 rows retain the semantic name, assembly byte-identical to `exact`. The rename in
  `typed_opaque` is applied consistently across signature, `dart_source` **and
  `tests`** (tests bind by name).
- `prompt_signature_mode` is already consumed by `build_decoder_prompt`, which
  `graph_inference_antigravity.py` imports. **No training-code change is required.**
- Base revisions are pinned and recorded. Do not resolve to hub HEAD.

## Artifacts and hashes — fail closed on any mismatch

```
checkpoint  hf://raafatabualazm/antigravity-qwen3-8b-artifacts
            artifacts/checkpoints/qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_arm64v21_s42_prefix_no_gine_regions16/pytorch_model.bin
            924,910,291 bytes   sha256 bc7ccbc6dbcae4755e93be7e685dd123b20a422085f9ec7793c1f456b6391085
decoder     Qwen/Qwen3-8B                  @ b968826d9c46dd6066d109eabc6255188de91218
encoder     microsoft/graphcodebert-base   @ 2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d
eval        flutter_eval_graphv2.jsonl     343 rows, sha256 864dc0bb7e9ee305ba0fc4be6e5d5ecbbeb7c17fd15bb3e41cfcc6d9aaf65fac
```

Public repo, no auth needed. **Verify the checkpoint by hash, not by name** — the
sibling `…graphv2_clean_s42…/pytorch_model.bin` is byte-identical in size and is
the wrong model.

The run directory is named `qwen3-8b-base…` but `GRAPH_DECODER_MODEL` is
`Qwen/Qwen3-8B`. Trust the provenance, not the directory name.

Settings from the 2026-07 run — do not improvise: decoder prompt max **2048**,
max new tokens **768**, k=10, bf16, SDPA, prompt assembly mode `none` (graph-only
conditioning), `scoring_tests_visible_to_policy=false`.

---

## Protocol: interleaved and paired

**Do not run view-major.** Run **task-major, both views back to back**:

```
for task in TASK_LIST:            # fixed order, frozen before generation
    generate(task, view=exact,         seed=42, k=10)  -> append, flush
    generate(task, view=typed_opaque,  seed=42, k=10)  -> append, flush
```

Same seed coordinates for both views so the only difference is the input, mirroring
`same_seed_sample_coordinates: true` from the x86 protocol.

This is the whole trick. View-major execution truncated at 70% yields one complete
arm and one partial arm, which cannot be compared. Interleaved execution truncated
at *any* point yields a complete, equal-n, perfectly paired experiment on however
many tasks finished. Flush after every task so a hard kill loses at most one.

### Task list

Freeze it before generating, in this priority order:

1. **Short-reference stratum first** (reference length < 1000 chars, 177 tasks). The
   long stratum contributed **0 passes out of 166** in the 2026-07 run, so it costs
   ~48% of the budget and cannot move pass@10.
2. If throughput allows, append the long stratum.

State in the pre-registration that the stratum is defined by reference length — a
covariate fixed before scoring — and that compile@10 on this subset is therefore
**not** comparable to the published 339/343.

### Scope selection at T+40

From the smoke, compute seconds per (task × 2 views). Choose the largest task count
that fits in the remaining budget with 10% headroom. Log the chosen n and the
dropped tasks explicitly — silent truncation reads as full coverage.

---

## Pre-register before generating

Write `PREREGISTRATION.json` and hash it **before the first full generation**. It
must contain:

- **Primary**: paired pass@10, `exact` vs `typed_opaque`, exact two-sided McNemar on
  discordant tasks.
- **Prediction**: `exact` ≈ 10.7% (19/177 scaled to n); `typed_opaque` materially
  lower; direction `exact` > `typed_opaque`. On x86 the analogous drop was 42 → 7.
- **Minimum reportable n**: **90 paired tasks**. Below that, report as a pilot and
  make no channel claim.
- **Reproduction gate**: on the `exact` arm, overlap of the solved-task set with
  `results/…_pass_predictions.json` from the 2026-07 bundle. Set the criterion as
  *substantial overlap*, not byte-identity — library and batching differences perturb
  individual candidates at a fixed seed, and demanding exact reproduction would fail
  for reasons unrelated to the pipeline.
- **Kill rule**: if `exact` solves **fewer than 5** tasks in the first 90, the
  pipeline differs from 2026-07. Report a failed reproduction and make **no**
  name-channel claim. This is the one that stops us reporting a broken pipeline as
  a name effect.
- **Diversity guardrail**: mean distinct-candidates/10 per arm
  (`analysis_rs_sft_fold/check_collapse.py`). Catches degenerate decoding before it
  distorts pass@k.

---

## Scoring

CPU only, on the runner, Dart 3.12.2, **in parallel with generation**. It does not
touch the GPU budget. Score the paired prefix only — every task must have both views
present or be excluded from the comparison.

---

## Do NOT do any of these

- Any training. The checkpoint exists; this is inference-only.
- `constants_stripped` — it has no ARM64 implementation and needs new code.
- The base-model probe, `name_only`, `none`, or any Tier 2/3 cell.
- compile@10 as a headline. Baseline any-compile is **339/343 = 98.8%**, at ceiling.
  Interventions that *reduce* compilation are measurable; there are only 4 tasks of
  upward headroom. Report it as secondary, with the ceiling stated.
- Editing anything under the x86 measurement harness. Keep everything in a new
  directory.

---

## Deliverable

1. `PREREGISTRATION.json` + its hash, written before generation.
2. Per-task paired results, both views, with provenance sidecars and journal chain heads.
3. Counts: `exact` solved / `typed_opaque` solved / discordant pairs / McNemar p.
4. distinct-candidates/10 per arm.
5. Reproduction-gate overlap against the 2026-07 solved set.
6. Explicit n, and the list of tasks dropped for time.

## Why this is worth 110 minutes

If the x86 effect size replicates, even a truncated run is decisive. At n=90 we
expect ~10 solved under `exact` and ~2 under `typed_opaque`; ~9 discordant pairs all
in one direction gives p ≈ 0.004 on an exact sign test. At n=177, p is beyond 1e-4.

The claim it buys: the name channel dominates on a second ISA, a different model
architecture, a different input representation, and **real obfuscated production
binaries rather than a HumanEval port**. That is a conceptual replication, not a
retest — which is exactly the generalisation axis a SANER reviewer will ask for.
