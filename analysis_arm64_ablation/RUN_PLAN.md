# ARM64 signature ablation — cross-ISA replication

Target: convert the signature-leakage finding from a Dart/x86-64 result into a
claim about the task, measured on a second ISA and on real production binaries.
This is the generalisation axis an ICSE/FSE reviewer will ask for.

## Why this corpus and not the x86 one

The x86 175-task set **cannot** run this ablation: its gold functions are already
named `fn0`, so there is no semantic name left to remove. The ARM64 corpus is the
only one in the project that still carries original names — and it is 343
held-out functions from real, obfuscated Flutter release APKs (arm64-v8a),
not a HumanEval port.

**The published ARM64 5.54% pass@10 was measured with the full semantic
signature supplied** (`prompt_signature_mode` defaults to `exact`). That number
sits in exactly the condition the paper criticises, which is worth stating
plainly rather than discovering in review.

## Views (built, validated, 343/343 rows, 0 failures)

`build_arm64_signature_views.py` emits four inputs. `prompt_signature_mode` is
already consumed by `graph_encoder_decoder_decompiler_v2_antigravity.py:361` —
**no training-code change is required.**

| view | prompt shows | record rewritten? |
|---|---|---|
| `exact` | `String sortPasswordViolationsBySeverity(List<String> violations)` | no |
| `typed_opaque` | `String fn0(List<String> p0)` | yes — see below |
| `name_only` | name, no typed signature | no |
| `none` | neither | no |

Only `typed_opaque` rewrites content, so any effect from `name_only`/`none`
cannot be a rewriting artefact. The rewrite renames the function consistently
across signature, `dart_source` **and `tests`** — the tests bind by name
(`final candidate = <name>;`), so an inconsistent rename would fail every task
for reasons unrelated to the ablation. Verified: 0 rows retain the semantic name,
assembly byte-identical to `exact`.

## Design

Primary, **inference-only** on the existing ARM64 checkpoint
(`qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_arm64v21_s42_prefix_no_gine_regions16`):

```
4 views x 5 seeds (42-46) x 343 tasks x k=10
```

No training, no API. Seeds 42–46 match the x86 replication so the two studies
share a seed protocol.

## Power

x86 measured pass@10 SD = 1.79 at mean 5.8 solved. Poisson-scaled to ARM64's 19
solved: **SD ~3.2 tasks (~0.9 pp)**. Differences under ~3 tasks are noise and
cannot promote a claim.

The expected effect is far larger: x86 went 42 -> 7 solved when the name was
removed. The ARM64 analogue would be 19 -> ~4, about 4.7 SD. **The study is
adequately powered for the headline and underpowered for anything subtle** —
so pre-commit to reporting only the name-removal effect, not small inter-view
differences.

## Decision rules (fixed before results)

- Primary endpoint: pass@10, reported with the 5-seed SD.
- `exact` vs `typed_opaque` is the headline comparison; McNemar exact on
  discordant tasks, plus the seed spread.
- Report `distinct-candidates/10` per arm. It is far better powered than a
  343-task binary count and catches degenerate decoding before it distorts pass@k.
- Any difference under 3 tasks is reported as within the noise floor, full stop.

## Answers to the three questions asked

**Run on the fine-tuned checkpoint?** Yes — mandatory. It is the only ARM64 model
with a non-zero pass@10. A channel decomposition on a model that solves nothing
yields four zeros and no information.

**Run before fine-tuning too?** Gate it. On x86 the pre-SFT base scored
**0 pass@1 / 0 pass@10 / 0 compile@10** — it could not emit compilable Dart in
the expected format at all. If the ARM64 base behaves the same, all four views
return zero and the run is wasted. So: run `exact` only, 1 seed, on the base
model first. If `compile@10` is materially non-zero, the full 4-view base ladder
becomes very valuable (it would show the channel effect does not depend on
task-specific training). If it is ~0, drop the arm and say so.

**Run T5Gemma 2 on ARM64?** No. T5Gemma is trained on x86 Dart in **F2** format;
ARM64 is **graphv2**. Running it changes ISA *and* input representation at once,
so a null is uninterpretable and a non-null is unattributable. It also very
likely returns zeros. Skip it.

## The stronger "before/after" that is worth running

The interesting axis is not pre- vs post-fine-tuning. It is **how much of the
achievable gain is information versus training** — and x86 already answers it:

```
fine-tuned + baseline input          compile@10 124
fine-tuned + typed contract (infer)  compile@10 170     <- +46, no training
fine-tuned ON typed contract (train) compile@10 171     <- +1 over inference-only
```

Supplying the information captures essentially the entire gain; training on it
adds one task. Replicating that triple on ARM64 needs the two inference arms
above plus **one** training run (fine-tune on `typed_opaque`). If training again
adds ~nothing on a second ISA, the claim generalises:

> Supplying recoverable type information at inference captures nearly all of the
> attainable improvement; training on it adds almost nothing.

That is a general, quantified, falsifiable statement about the task rather than
about a model — which is the bar this paper needs to clear.

## Sequence

1. Build views (done) — `build_arm64_signature_views.py`
2. Base-model probe: `exact`, 1 seed → decide on the base ladder
3. Primary: 4 views x 5 seeds, inference-only, fine-tuned checkpoint
4. Conditional: one training run on `typed_opaque`, evaluate, complete the triple
5. Secondary (needs a graphv2 port of the F2 transforms): `constants_stripped`
   and `structure_permuted`. Lower priority — the signature ladder is the
   headline and it is config-only.
