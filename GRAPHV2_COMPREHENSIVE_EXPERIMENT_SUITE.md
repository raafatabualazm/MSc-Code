# Graph-v2 Comprehensive Experiment Suite

This document fixes the remaining Graph-v2 study before the final x86 model is
selected. The aim is to answer the strongest foreseeable reviewer objections,
not to keep searching until one configuration wins. Every completed cell is
reported, including negative results.

## Freeze Rules

1. Seed 42 is the architecture-screen seed. It may select a representation and
   prefix configuration, but it is not confirmatory evidence by itself.
2. The complete x86 configuration is frozen before seeds 43-46, the fresh
   signature-scrubbed holdout, or ARM64 results are inspected.
3. The fresh holdout manifest is hashed before its tasks are generated or
   compiled. No architecture or hyperparameter changes are allowed afterward.
4. ARM64 is a replication dataset, not another tuning split.
5. All candidate pools contain ten candidates per task and are reused for
   pass@k, aligned JIT compile, and CodeBLEU.
6. Failed provenance, changed task denominators, visible scoring tests, or
   mismatched candidate pools invalidate a run rather than becoming exclusions.

## Reviewer-Objection Matrix

| Objection | Controlled evidence |
|---|---|
| The decoder solves tasks from signatures alone | Untuned, signature-only, raw-assembly text, and signature-scrubbed evaluations |
| The learned binary channel is ignored | Zero-gate and cross-task prefix-permutation controls |
| Improvements come from prompt length or extra parameters | No-attention, frozen-encoder, and matched prefix-density controls |
| CFG/DFG topology is unnecessary or harmful | No-edge, CFG, CFG+DFG, shuffled-edge, no-GINE, and GINE-depth controls |
| Four GINE layers oversmooth | GINE depth 0, 2, and 4 |
| Global attention or block order is doing all the work | No-attention, no-position, and block-order-shuffle controls |
| One-vector pooling discards semantics | CLS and multi-query pooling with 2, 4, and 8 vectors per block |
| GraphCodeBERT is an arbitrary encoder choice | Pinned GraphCodeBERT and pinned CLAP-ASM |
| Hierarchical compression failed only because of one region size | Region maximums 4, 8, and 16, reported as a fixed sensitivity family |
| Prefix capacity or gate initialization was arbitrarily chosen | Fixed density x gate grid on the frozen representation |
| Results are seed noise | Five matched x86 seeds for the winner and strongest fixed baseline |
| Results depend on the HumanEval-style task signature | Existing signature ablation plus a post-freeze fresh signature-scrubbed holdout |
| Results are x86- or short-function-specific | Frozen ARM64 Graph-v2.1 replication on the 343-task longer-function split |
| Accuracy gains hide excessive cost | Parameters, training time, inference time, peak VRAM, graph vectors, and prefix-token counts |
| Results exploit benchmark overlap | Exact/near-overlap, component-split, source-hash, prompt-integrity, and hidden-test audits |

## Execution Order

### Phase 0: Protocol Lock

- Pin decoder, encoder, tokenizer, Dart SDK, graph extractor, and evaluator
  revisions.
- Run `py_compile`, graph integrity tests, protocol integrity tests, dataset
  audits, overlap audits, and the three-run pass-stability evaluator.
- Record immutable hashes for every train, validation, test, and holdout file.

### Phase 1: Complete the Seed-42 Representation Screen

Report all four primary representations against `prefix_no_gine`:

- `prefix_no_gine` (GraphCodeBERT CLS, one vector per block)
- `prefix_no_gine_regions` (already complete)
- `prefix_no_gine_clap` (currently running)
- `prefix_no_gine_multivector4` (queued)

Primary selection order is pass@1, pass@5, pass@10, then aligned compile and
cost. A representation is not selected from CodeBLEU alone.

### Phase 2: Complete Component Isolations

Run every central architectural isolation at seed 42:

- `prefix_no_gine_no_attention`
- `prefix_no_edges_gine2`
- `prefix_no_gine_no_positions`
- `prefix_no_gine_frozen_encoder`

Retain the already completed no-GINE, four-layer GINE, shuffled-edge,
zero-gate, prefix-permutation, block-order, and region controls.

### Phase 3: Resolve Compression Families

Run the complete fixed representation sensitivities, even if their center cell
is negative:

- Multi-query vectors per block: 2, 4, 8; CLS is the one-vector reference.
- Region maximum blocks: 4, 8, 16; `8` is the completed center cell.

These are family-level sensitivity results. All cells remain in the paper and
supplement; no failed cell is silently dropped.

### Phase 4: Prefix Density and Gate Grid

Freeze the representation family, then run the full seed-42 grid:

| Dynamic prefix density | Gate initialization |
|---:|---:|
| 2 | 0.1, 0.2, 0.3 |
| 4 | 0.1, 0.2, 0.3 |
| 6 | 0.1, 0.2, 0.3 |

The existing density `4`, gate `0.2` run is reused when its representation
matches the frozen winner. Record the learned final gate distribution and the
actual prefix-token distribution, including the fraction of tasks hitting the
minimum or maximum.

### Phase 5: Five-Seed Matched Confirmation

Freeze the complete x86 configuration. Evaluate seeds 42-46 for:

- the selected complete configuration;
- the strongest fixed pre-grid baseline (`prefix_no_gine`, density 4, gate
  0.2).

Reuse existing valid seeds rather than retraining them. The confirmatory claim
uses the matched five-seed comparison, not the best seed.

### Phase 6: Multi-Seed Causal Confirmation

For every selected-config checkpoint at seeds 42-46, reuse the checkpoint and
run:

- prefix gate forced to zero;
- graph/prefix input cyclically shifted across tasks;
- block order deterministically shuffled with topology remapped.

This separates retraining variance from inference-time causal dependence.

### Phase 7: Fresh Signature-Scrubbed Holdout

After the x86 freeze manifest is written, build a new private/public benchmark
with `scripts/data/build_signature_scrubbed_eval.py --benchmark_kind
fresh_holdout`. The prompt exposes only neutral target name `candidate` plus the
binary representation. Types, semantic names, source, and tests remain private.

Evaluate without retraining:

- signature-only decoder baseline;
- raw-assembly text baseline;
- strongest fixed prefix baseline;
- selected complete configuration;
- any feasible external model baseline under exactly the same public prompt.

### Phase 8: Same-Task External Baselines

External comparisons must target Dart on the same task IDs; C-only benchmark
numbers are context, not direct baselines. Include, where licensing and API
access permit:

- an untuned decoder-only Qwen3-8B baseline;
- a frontier general-purpose LLM prompted with the same public Dart task and
  assembly budget;
- a conventional-decompiler-pseudocode-to-Dart pipeline using the same decoder;
- a train-set nearest-neighbor/retrieval baseline to expose memorization.

Record prompt, model revision, decoding parameters, token usage, cost, and all
raw candidate pools.

### Phase 9: Frozen ARM64 Replication

Do not alter the selected representation, density, gate, training schedule, or
decoder after seeing ARM64 results.

1. Run the seven-arm ARM64 pilot at seed 42.
2. Run the frozen selected x86 configuration at seeds 42-46.
3. Repeat the strongest ARM64 baseline at the same seeds.
4. Run zero-gate, cross-task permutation, and block-order controls for the
   selected ARM64 checkpoints.
5. Report the 343-task result separately from x86 and as a joint cross-ISA
   random-effects summary.

### Phase 10: Robustness and Cost Report

For x86 and ARM64, report:

- pass@1, pass@5, pass@10;
- aligned JIT compile@1, @5, @10;
- CodeBLEU as a secondary diagnostic;
- solved-task gains/losses and candidate diversity;
- low/mid/high strata for block count, instruction count, branch count, and
  data-flow edge count;
- training wall time, peak VRAM, checkpoint size, inference candidates/second,
  graph vectors/function, and decoder prefix tokens/function;
- failure taxonomy: syntax, static type, runtime, timeout, and wrong output.

## Statistical Plan

- Report per-seed values and mean +/- sample standard deviation.
- Use task-paired bootstrap intervals within a seed and a hierarchical
  seed-then-task bootstrap for multi-seed effects.
- Use paired solved-task gain/loss counts with exact McNemar tests.
- Apply Holm correction within each prespecified family: topology, component
  isolation, representation, prefix grid, and external baselines.
- Report effect sizes and intervals even when corrected tests are not
  significant.
- Treat pass@1 as deployment quality, pass@10 as candidate-pool potential, and
  compile as necessary but not sufficient.

## Decision Rule

The final model must satisfy all of the following:

1. Positive mean pass@1 and pass@10 effects over the strongest fixed baseline
   across five matched x86 seeds.
2. A meaningful drop under zero-gate or cross-task permutation, showing that
   the binary channel is causally used.
3. No unacceptable regression in aligned compile, high-complexity tasks, or
   runtime/representation cost.
4. Directionally consistent improvement on the untouched signature-scrubbed
   holdout.
5. Directionally consistent improvement on ARM64 without ARM64-specific
   retuning.

If a criterion fails, report the failure and narrow the claim. Do not select a
new configuration from the holdout or ARM64 results.
