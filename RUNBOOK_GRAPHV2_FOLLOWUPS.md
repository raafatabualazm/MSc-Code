# Graph-v2 Follow-up Runbook

This runbook continues the leakage-free x86 study and then performs the
prepared ARM64 Graph-v2.1 replication. All commands are resumable at the
completed-result level: a stage is skipped only when its 10-candidate pool,
summary, and leakage-free provenance are complete.

The frozen objection matrix, five-seed confirmation rules, fresh-holdout
boundary, and external-baseline requirements are defined in
`GRAPHV2_COMPREHENSIVE_EXPERIMENT_SUITE.md`.

The follow-up runners set `save_strategy=no`: every training stage saves its
final `pytorch_model.bin`, but does not retain multi-gigabyte optimizer snapshots
at each epoch. This is deliberate because the active Vast host is ephemeral and
disk-limited. Enable Hugging Face output for durable final adapters and metrics.

## Scientific Order

1. Finish the seed-42 CLAP and four-vector encoder screen.
2. Complete the central component controls: no global attention, GINE depth 2,
   no block positions, and a frozen local encoder.
3. Complete the fixed region-size 4/8/16 and multi-vector 2/4/8 families.
4. Run causal inference controls on the selected seed-42 checkpoint:
   - prefix gate forced to zero;
   - graph input cyclically permuted across tasks;
   - block order deterministically shuffled with topology remapped.

   The default matrix applies these controls to both no-GINE and no-edge
   checkpoints, avoiding a post-hoc choice based on one seed.
5. Freeze the representation and run the density 2/4/6 by gate 0.1/0.2/0.3
   grid at seed 42.
6. Freeze the complete configuration and run matched selected/baseline seeds
   42-46.
7. Repeat zero-gate, cross-task permutation, and block-order controls for all
   five selected checkpoints.
8. Evaluate the post-freeze signature-scrubbed holdout without retraining.
9. Run same-task Dart external baselines.
10. Replicate the frozen configuration on ARM64 without ARM64-specific tuning.

Do not interpret a single seed as architecture confirmation. The zero-gate and
cross-task permutation controls test whether the prefix carries task-specific
information at all; they do not replace independent training seeds.

## Transfer and Preflight

Create the transfer archive from the workspace root:

```bash
tar -czf graphv2_followups_bundle.tar.gz -T upload_clean_study_filelist.txt
```

After extraction on the GPU host:

```bash
python -m py_compile \
  scripts/run_graphv2_followups.py \
  scripts/run_arm64_graphv21_study.py \
  scripts/evaluation/graph_inference_antigravity.py \
  configs/run_sweeps_antigravity.py

python -m unittest \
  scripts.evaluation.test_graph_input_ablation_antigravity \
  scripts.data.test_graph_architecture_ablation
```

## x86 Commands

Dry-run the complete follow-up matrix:

```bash
python scripts/run_graphv2_followups.py \
  --phase all \
  --selected_architecture prefix_no_gine \
  --causal_architectures prefix_no_gine,prefix_no_edges \
  --repeat_seeds 43,44 \
  --capacity_values 2,6 \
  --budget_hours 20
```

Launch it in `tmux` after the no-GINE seed-42 result is complete:

```bash
tmux new-session -d -s graphv2_followups \
  "cd /workspace && \
   python scripts/run_graphv2_followups.py \
     --phase all \
     --execute \
     --selected_architecture prefix_no_gine \
     --causal_architectures prefix_no_gine,prefix_no_edges \
     --repeat_seeds 43,44 \
     --capacity_values 2,6 \
     --metric_workers 64 \
     --budget_hours 20 \
     --wait_for_model qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine \
     --poll_seconds 120 \
     2>&1 | tee logs/graphv2_followups/all.log"
```

If no-GINE clearly loses to `prefix_no_edges`, use
`--selected_architecture prefix_no_edges` for the causal and capacity phases.
The `all` phase still records no-GINE repeats because that negative result needs
replication.

Capacity confirmation is a separate decision after the seed-42 sweep:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-capacity-repeat \
  --execute \
  --selected_architecture prefix_no_gine \
  --capacity_winner 2 \
  --repeat_seeds 43,44 \
  --metric_workers 64 \
  --budget_hours 6
```

Replace `2` with `6` only if that arm wins the seed-42 capacity sweep. Do not
repeat a capacity arm that fails to improve the balanced functional result.

### Comprehensive Screen

After the active encoder queue exits, dry-run the remaining seed-42 screen:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-comprehensive-screen \
  --isolation_variants prefix_no_gine_no_attention,prefix_no_edges_gine2,prefix_no_gine_regions,prefix_no_gine_no_positions,prefix_no_gine_frozen_encoder \
  --encoder_variants prefix_no_gine_clap,prefix_no_gine_multivector4 \
  --vector_values 2,4,8 \
  --region_values 4,8,16 \
  --metric_workers 64 \
  --budget_hours 20
```

The runner skips complete CLAP, vector-4, and region-8 artifacts. Add
`--execute` only after confirming that the active queue has fully exited.

After the representation winner is frozen, run the complete prefix grid:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-prefix-grid \
  --execute \
  --selected_representation prefix_no_gine_multivector4 \
  --prefix_density_values 2,4,6 \
  --prefix_gate_values 0.1,0.2,0.3 \
  --metric_workers 64 \
  --budget_hours 20
```

Replace the example representation with the frozen winner. Then freeze the
complete density/gate cell and confirm it across five seeds:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-confirm \
  --execute \
  --selected_representation prefix_no_gine_multivector4 \
  --selected_prefix_density 4 \
  --selected_gate_init 0.2 \
  --confirm_seeds 42,43,44,45,46 \
  --metric_workers 64 \
  --budget_hours 12
```

Finally, run the three causal controls for each selected checkpoint:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-confirm-causal \
  --execute \
  --selected_representation prefix_no_gine_multivector4 \
  --selected_prefix_density 4 \
  --selected_gate_init 0.2 \
  --confirm_seeds 42,43,44,45,46 \
  --metric_workers 64 \
  --budget_hours 15
```

## x86 Analysis

After synchronizing results into a dated folder:

```bash
python scripts/evaluation/analyze_graphv2_clean_study.py \
  --results_dir results-YYYYMMDD \
  --benchmark data/testing/grpo_data_graphv2.jsonl \
  --output_json results-YYYYMMDD/graphv2_clean_study_analysis.json \
  --output_md results-YYYYMMDD/GRAPHV2_CLEAN_STUDY_ANALYSIS.md
```

Acceptance checks:

- A useful prefix should beat its zero-gate and cross-task permutation controls.
- A topology claim requires intact edges to beat no-edge and shuffled-edge
  controls across seeds; one seed is insufficient.
- If no-GINE matches or beats GINE, describe the contribution as learned block
  encoding and prefix compression, not graph message passing.
- Prefer pass@10 and aligned JIT compile; CodeBLEU is diagnostic.

## ARM64 Graph-v2.1

The runner verifies immutable hashes, row counts, graph schema, and overlap
audits before printing or executing any stage. The seed-42 pilot is seven arms
and is estimated at 16.8 GPU-hours:

```bash
python scripts/run_arm64_graphv21_study.py \
  --phase pilot \
  --execute \
  --seed 42 \
  --generation_batch_size 10 \
  --metric_workers 64 \
  --budget_hours 20
```

The pilot contains signature-only, raw cleaned assembly, no-edge prefix, CFG,
CFG+DFG, shuffled edges, and no-GINE. It uses 1,371 training rows and evaluates
all arms on the untouched 343-task ARM64 split.

Repeat the five useful architecture families only after inspecting the pilot:

```bash
python scripts/run_arm64_graphv21_study.py \
  --phase repeat \
  --execute \
  --repeat_seeds 43,44 \
  --repeat_arms text,prefix_no_edges,prefix_cfg,prefix_cfg_dfg,prefix_no_gine \
  --metric_workers 64 \
  --budget_hours 30
```

Then apply the causal controls to the selected ARM64 checkpoint:

```bash
python scripts/run_arm64_graphv21_study.py \
  --phase causal \
  --execute \
  --selected_architecture prefix_no_gine \
  --selected_prefix_density 4 \
  --selected_gate_init 0.2 \
  --causal_seeds 42,43,44,45,46 \
  --metric_workers 64 \
  --budget_hours 25
```

For the frozen x86 winner, use the `selected` phase with the same architecture,
density, and gate values and `--selected_seeds 42,43,44,45,46`. The ARM64
runner also accepts multi-vector winners with 2, 4, or 8 vectors per block.

Analyze synchronized ARM64 results offline:

```bash
python scripts/evaluation/analyze_arm64_graphv21_study.py \
  --results_dir results-YYYYMMDD \
  --output_json results-YYYYMMDD/arm64_graphv21_study_analysis.json \
  --output_md results-YYYYMMDD/ARM64_GRAPHV21_STUDY_ANALYSIS.md
```

ARM64 is a cross-ISA, longer-function external replication. Its semantics are
synthetic algorithmic Dart tasks extracted from real Flutter ARM64 release
binaries; it is not evidence about organic application business logic.

## Deferred Experiments

- Additional GRPO epochs are deferred until the final SFT architecture is
  selected. The clean seed-42 binary-GRPO run produced only a one-task pass@10
  gain and does not justify repeated RL yet.
- GNN depth above four is not scheduled. If four-layer GINE loses to no-GINE,
  two layers tests oversmoothing; adding depth would move in the wrong direction.
- A 128-token fixed-prefix run is not scheduled. Dynamic-prefix scale changes
  affect the benchmark materially; cap-only 32/64 changes almost no tasks.
