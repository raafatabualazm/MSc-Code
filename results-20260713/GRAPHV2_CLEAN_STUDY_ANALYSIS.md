# Graph-v2 Leakage-Free Study Analysis

This report recomputes metrics from archived candidate pools; no model inference was rerun. Dart was rerun only for the explicit multi-run pass-stability audit described below.

## Protocol

- Provenance audit: **FAIL** across 38 complete runs.
- All audited runs use 154 tasks, 10 candidates per task, the pinned Qwen3-8B and GraphCodeBERT revisions, and `antigravity-v2-no-test-hints`.
- Compile and pass metrics reuse the same candidate pool and use the pass-aligned JIT/test harness.
- Candidate-level CSV replay is canonical for paired effects, task coverage, and the tables below.
- Pass stability requires 3 successful executions. The raw archives are preserved; 1 documented stochastic false positive is corrected through an overlay.
- The bulk stability audit replayed 3450 archived positives across 15 runs; 3449 remained passing and 1 was invalidated.
- CodeBLEU uses extracted Dart code. The evaluator has been unified with the compile/statistics extractor; archived aggregate JSONs produced by the older raw-text extractor are audited below rather than mixed into the analysis.
- The legacy standalone-AOT compiled-only CodeBLEU count is retained only as a diagnostic and is not treated as the aligned compile metric.

## Three-Seed Core Ablation

| Variant | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | compile@10 | CodeBLEU | Solved tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefix, no edges | 0.1604 +/- 0.0096 | 0.2330 +/- 0.0136 | 0.2597 +/- 0.0234 | 0.7563 +/- 0.0069 | 0.9430 +/- 0.0070 | 0.9589 +/- 0.0037 | 0.6722 +/- 0.0033 | 40.0 +/- 3.6 |
| Prefix + CFG | 0.1589 +/- 0.0143 | 0.2309 +/- 0.0113 | 0.2554 +/- 0.0135 | 0.7470 +/- 0.0262 | 0.9480 +/- 0.0041 | 0.9675 +/- 0.0065 | 0.6669 +/- 0.0134 | 39.3 +/- 2.1 |
| Prefix + CFG + DFG | 0.1535 +/- 0.0177 | 0.2281 +/- 0.0163 | 0.2554 +/- 0.0270 | 0.7316 +/- 0.0860 | 0.9397 +/- 0.0238 | 0.9632 +/- 0.0037 | 0.6616 +/- 0.0085 | 39.3 +/- 4.2 |

## Edge Effects

- **prefix_cfg vs prefix_no_edges**: pass@10 difference -0.0043, hierarchical bootstrap 95% CI [-0.0455, +0.0325].
- **prefix_cfg_dfg vs prefix_cfg**: pass@10 difference +0.0000, hierarchical bootstrap 95% CI [-0.0390, +0.0433].
- **prefix_cfg_dfg vs prefix_no_edges**: pass@10 difference -0.0043, hierarchical bootstrap 95% CI [-0.0281, +0.0195].
- Seed 44 illustrates the instability: CFG versus no-edge has 1 gains/8 losses (exact p=0.0391), while adding DFG to CFG has 7 gains/0 losses (exact p=0.0156). The directions reverse within one seed rather than reproducing across seeds.

The intervals all cross zero. The current evidence supports the learned block-prefix representation, but does not support a causal benefit from CFG or DFG edges.

## Signature-Only Control

- Trained raw assembly vs signature-only base: pass@10 difference -0.0195, paired task 95% CI [-0.0649, +0.0260], 4 gains/7 losses (exact p=0.5488).
- No-edge prefix vs signature-only base: pass@10 difference +0.0195, paired task 95% CI [-0.0195, +0.0584], 7 gains/4 losses (exact p=0.5488).
- CFG prefix vs signature-only base: pass@10 difference +0.0260, paired task 95% CI [-0.0130, +0.0714], 8 gains/4 losses (exact p=0.3877).

The exact HumanEval-style signature is itself a strong task-recognition cue. Prefix arms improve over raw-assembly SFT in this fixed seed, but their small pass@10 advantage over the signature-only base is not distinguishable from zero. Claims must therefore be about this benchmark and representation pipeline, not general binary-semantic recovery.

## Complexity Stratification

| Block-count stratum | Range | Prefix, no edges | Prefix + CFG | Prefix + CFG + DFG |
|---|---:|---:|---:|---:|
| low | 3-14 blocks | 0.3399 | 0.3203 | 0.3268 |
| mid | 14-24 blocks | 0.2614 | 0.2810 | 0.2810 |
| high | 25-458 blocks | 0.1795 | 0.1667 | 0.1603 |

All variants degrade sharply on larger graphs, and explicit edges do not rescue the high-complexity stratum. This is evidence of a remaining representation/capacity ceiling, not evidence that topology is unnecessary in principle.

## Seed-42 Diagnostics

| Variant | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | CodeBLEU | Solved | Mean unique/10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signature_only_base | 0.1487 | 0.2078 | 0.2338 | 0.5883 | 0.8644 | 0.6475 | 36 | 9.64 |
| untuned | 0.1195 | 0.1881 | 0.2013 | 0.5922 | 0.8339 | 0.6468 | 31 | 9.73 |
| text | 0.1104 | 0.1892 | 0.2143 | 0.6065 | 0.8862 | 0.6410 | 33 | 9.70 |
| prefix_no_edges | 0.1688 | 0.2300 | 0.2532 | 0.7636 | 0.9459 | 0.6759 | 39 | 9.19 |
| prefix_cfg | 0.1734 | 0.2343 | 0.2597 | 0.7675 | 0.9460 | 0.6821 | 40 | 8.90 |
| prefix_cfg_dfg | 0.1649 | 0.2187 | 0.2338 | 0.7955 | 0.9482 | 0.6538 | 36 | 9.51 |
| prefix_no_gine | 0.1617 | 0.2314 | 0.2597 | 0.7831 | 0.9527 | 0.6726 | 40 | 9.10 |
| prefix_no_gine_clap | 0.1656 | 0.2318 | 0.2532 | 0.7708 | 0.9501 | 0.6715 | 39 | 9.06 |
| prefix_no_gine_multivector2 | 0.1448 | 0.2279 | 0.2532 | 0.6955 | 0.9498 | 0.6679 | 39 | 8.87 |
| prefix_no_gine_multivector4 | 0.1584 | 0.2362 | 0.2597 | 0.7649 | 0.9479 | 0.6670 | 40 | 9.11 |
| prefix_no_gine_multivector8 | 0.1578 | 0.2353 | 0.2468 | 0.7123 | 0.9388 | 0.6689 | 38 | 9.17 |
| prefix_no_gine_regions4 | 0.1708 | 0.2333 | 0.2597 | 0.7721 | 0.9439 | 0.6803 | 40 | 8.71 |
| prefix_no_gine_regions | 0.1604 | 0.2227 | 0.2403 | 0.7578 | 0.9420 | 0.6732 | 37 | 9.12 |
| prefix_no_gine_regions16 | 0.1766 | 0.2411 | 0.2727 | 0.7708 | 0.9380 | 0.6785 | 42 | 8.45 |
| prefix_no_gine_no_attention | 0.1675 | 0.2312 | 0.2468 | 0.7734 | 0.9422 | 0.6771 | 38 | 8.92 |
| prefix_no_gine_no_positions | 0.1747 | 0.2286 | 0.2403 | 0.7805 | 0.9528 | 0.6752 | 37 | 9.46 |
| prefix_no_edges_gine2 | 0.1604 | 0.2370 | 0.2597 | 0.7708 | 0.9453 | 0.6695 | 40 | 9.06 |
| prefix_no_gine_eval_gate_zero | 0.1584 | 0.2106 | 0.2143 | 0.7000 | 0.9152 | 0.6562 | 33 | 9.52 |
| prefix_no_gine_eval_prefix_permuted | 0.1714 | 0.2341 | 0.2468 | 0.8117 | 0.9559 | 0.6782 | 38 | 9.21 |
| prefix_no_gine_eval_block_order_shuffled | 0.1617 | 0.2321 | 0.2532 | 0.7721 | 0.9481 | 0.6790 | 39 | 9.13 |
| prefix_no_edges_eval_gate_zero | 0.1539 | 0.2101 | 0.2338 | 0.5935 | 0.8543 | 0.6619 | 36 | 9.36 |
| prefix_shuffled | 0.1734 | 0.2286 | 0.2468 | 0.7838 | 0.9370 | 0.6718 | 38 | 8.90 |
| prefix_cfg_dfg_text | 0.1188 | 0.2161 | 0.2338 | 0.6545 | 0.9393 | 0.6709 | 36 | 9.92 |
| prefix_cfg_dfg_expanded | 0.1506 | 0.2255 | 0.2403 | 0.7812 | 0.9474 | 0.6542 | 37 | 9.29 |

## Coverage Across Seeds

- No-edge family union: 48/154 tasks.
- CFG family union: 45/154 tasks.
- CFG+DFG family union: 47/154 tasks.
- All nine core pools: 54/154 tasks; 100 tasks remain unsolved.

These are oracle candidate-pool ceilings, not deployable pass@1 results.

## Expanded SFT and GRPO Readiness

- Expanded SFT changes pass@10 by +0.0065 (95% paired task bootstrap [-0.0195, +0.0390]).
- GRPO reward preflight found signal in 37.5% of groups, with 16.4% perfect samples; this is sufficient to run GRPO, not evidence that GRPO improves held-out performance.

## GINE and Prefix Causal Follow-ups

- No-GINE vs no-edge GINE: pass@10 difference +0.0065, paired task 95% CI [-0.0260, +0.0390], 4 gains/3 losses (exact p=1.0000).
- No-GINE vs CFG+DFG GINE: pass@10 difference +0.0260, paired task 95% CI [+0.0000, +0.0584], 5 gains/1 losses (exact p=0.2188).
- CLAP-ASM vs GraphCodeBERT no-GINE: pass@10 difference -0.0065, paired task 95% CI [-0.0325, +0.0195], 2 gains/3 losses (exact p=1.0000).
- Four block vectors vs one CLS vector: pass@10 difference +0.0000, paired task 95% CI [-0.0390, +0.0325], 4 gains/4 losses (exact p=1.0000).
- Two block vectors vs one CLS vector: pass@10 difference -0.0065, paired task 95% CI [-0.0325, +0.0195], 2 gains/3 losses (exact p=1.0000).
- Eight block vectors vs one CLS vector: pass@10 difference -0.0130, paired task 95% CI [-0.0455, +0.0195], 2 gains/4 losses (exact p=0.6875).
- Hierarchical regions vs no-GINE block prefix: pass@10 difference -0.0195, paired task 95% CI [-0.0519, +0.0065], 1 gains/4 losses (exact p=0.3750).
- Region maximum 4 vs no regions: pass@10 difference +0.0000, paired task 95% CI [-0.0260, +0.0260], 2 gains/2 losses (exact p=1.0000).
- Region maximum 16 vs no regions: pass@10 difference +0.0130, paired task 95% CI [-0.0195, +0.0455], 4 gains/2 losses (exact p=0.6875).
- No global attention vs no-GINE: pass@10 difference -0.0130, paired task 95% CI [-0.0390, +0.0130], 1 gains/3 losses (exact p=0.6250).
- No block positions vs sinusoidal positions: pass@10 difference -0.0195, paired task 95% CI [-0.0519, +0.0065], 1 gains/4 losses (exact p=0.3750).
- Two-layer GINE vs four-layer no-edge GINE: pass@10 difference +0.0065, paired task 95% CI [-0.0260, +0.0390], 4 gains/3 losses (exact p=1.0000).
- Zero prefix gate vs intact no-GINE prefix: pass@10 difference -0.0455, paired task 95% CI [-0.0779, -0.0130], 0 gains/7 losses (exact p=0.0156).
- Cross-task permuted graph vs intact graph: pass@10 difference -0.0130, paired task 95% CI [-0.0325, +0.0000], 0 gains/2 losses (exact p=0.5000).
- Shuffled block order vs intact block order: pass@10 difference -0.0065, paired task 95% CI [-0.0325, +0.0195], 2 gains/3 losses (exact p=1.0000).
- Zero prefix gate vs intact no-edge prefix: pass@10 difference -0.0195, paired task 95% CI [-0.0584, +0.0195], 3 gains/6 losses (exact p=0.5078).
These are seed-42 causal diagnostics. The winning architecture must be repeated at seeds 43-46 before it becomes the five-seed confirmatory configuration.

## Interpretation

1. The leakage-free graph-prefix family has the strongest fixed-run local results, particularly relative to raw full-assembly prompting, but the advantage over the signature-only base is modest at k=10.
2. Explicit topology is not validated: no-edge has the best three-seed mean pass@10, shuffled edges do not hurt at seed 42, and CFG+DFG does not consistently beat CFG.
3. The likely useful component is learned assembly-to-prefix compression through the block encoder and adapter. Calling the observed gain a graph-topology gain would overstate the evidence.
4. Adding raw assembly back to CFG+DFG raises CodeBLEU but does not improve pass@10, consistent with context overload or competing conditioning channels.
5. Expanded synthetic SFT is essentially neutral on this benchmark. The pending GRPO run is justified by the reward-signal audit, but must be accepted only on held-out leakage-free metrics.

## Reporting Caveats

- The legacy core has three seeds; the frozen selected configuration requires five matched seeds. Report all per-seed values, mean +/- sample SD, and hierarchical bootstrap intervals.
- Seed-42 controls, shuffled edges, graph+text, and expanded SFT remain single-run diagnostics.
- CodeBLEU and aligned JIT compile measure different properties from functional pass@k; functional conclusions should lead with pass@k.

## Artifact Consistency

- 10 of 38 runs contain at least one aggregate-summary/candidate-replay disagreement.
- CodeBLEU extraction differences occur in 5 runs and are resolved by the shared code extractor.
- Functional metric differences occur in 5 runs.
- Task 158 diagnosed the cause: one-run evaluation passed in 1/20 replays, while three-run stability passed in 0/20 replays. The candidate used random tie-breaking on a deterministic task.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_multivector4` `pass_at_1` differs by -0.000649; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_multivector4` `pass_at_5` differs by -0.003247; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_multivector4` `pass_at_10` differs by -0.006494; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_multivector8` `pass_at_1` differs by -0.001948; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_multivector8` `pass_at_5` differs by -0.005952; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_multivector8` `pass_at_10` differs by -0.006494; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_cfg_dfg` `pass_at_1` differs by -0.000649; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_cfg_dfg` `pass_at_5` differs by -0.003247; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_cfg_dfg` `pass_at_10` differs by -0.006494; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_no_edges` `pass_at_1` differs by -0.000649; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_no_edges` `pass_at_5` differs by -0.003247; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_no_edges` `pass_at_10` differs by -0.006494; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_no_gine` `pass_at_1` differs by -0.000649; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_no_gine` `pass_at_5` differs by -0.003247; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
- `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s44_prefix_no_gine` `pass_at_10` differs by -0.006494; the archived aggregate is retained, while the corrected candidate replay is used for analysis.
