# Claude Summary: Graph-Conditioned Dart AOT Neural Decompilation

Date: 2026-07-07

This summary is for continuing the paper-writing and experiment-analysis work on the Antigravity Dart AOT decompilation project. It assumes the current workspace is `C:\Users\Raafat Abualazm\Desktop\Train Data\experiment_workspace`.

## Immediate Context

The prior SANER ERA paper is:

- Raafat Abualazm and Ayman Abo Elhassan, "LLMs as Idiomatic Decompilers: Recovering High-Level Code from x86-64 Assembly for Dart", arXiv:2604.02278, accepted at SANER 2026 ERA Track.

The TOSEM submission to cite as the evaluation baseline is:

- Raafat Abualazm, Ayman AboElhassan, and Amr G. Wassal, "Evaluating Fine-Tuning and Metrics for Neural Decompilation of Dart AOT Binaries", arXiv submission 7795177 / TOSEM preprint under review. Local PDF: `C:\Users\Raafat Abualazm\Desktop\Comments\tosem_paper_arxiv_notice_nolinenums.pdf`.

The new work is not just another fine-tuning sweep. The story has shifted to a graph-conditioned architecture and an inference-time selection/ensemble analysis:

- GraphCodeBERT encoder over CFG/DFG-derived representations.
- Qwen3-8B decoder with learned graph prefix tokens.
- LoRA on both encoder and decoder.
- Explicit comparison of architecture ablations, GRPO, RS-SFT, reranking, and multi-arm candidate pooling.
- Functional correctness remains measured by pass@k on 154 HumanEval-Dart tasks.

## Best Current Architectural Arm

The clean architecture baseline is:

`qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_graphonly`

Key flags:

- `--qwen_prefix_tokens 16`
- `--qwen_prefix_rms_match 1`
- `--prompt_assembly_mode none`
- `--prompt_fit_assembly 1`
- `--dfg_mode edges`
- `--position_scheme roberta`
- `--auto_cfg 0`
- `--max_block_instrs 24`
- `GRAPH_MAX_DATAFLOW_EDGES=4096`
- LoRA rank 64, alpha 128 on both GraphCodeBERT encoder and Qwen3-8B decoder.

This arm is the best paper-clean model because it keeps CodeBLEU and compile@k high while improving pass@k relative to weaker graph/text/null baselines.

## Main Standalone Results

All results are for Qwen3-8B-base plus GraphCodeBERT-based variants on the x86 Dart AOT benchmark.

| Arm | CodeBLEU | Compiled Rows | compile@5 | pass@1 | pass@5 | pass@10 |
|---|---:|---:|---:|---:|---:|---:|
| x86_g3_binary_pk10_g32_grpo_grpo | 0.5438 | 33 | 0.2619 | 0.0409 | 0.1280 | 0.1818 |
| x86_g3_rs_sft_allarms | 0.5740 | 53 | 0.4286 | 0.0474 | 0.1296 | 0.1688 |
| x86_g3_binary_rs_sft_ref | 0.5323 | 43 | 0.3492 | 0.0357 | 0.1121 | 0.1558 |
| x86_g3_graphonly | 0.6383 | 84 | 0.6667 | 0.0266 | 0.0996 | 0.1558 |
| x86_g3_style_repair_ultralite | 0.5299 | 26 | 0.1984 | 0.0240 | 0.0918 | 0.1429 |
| x86_g3_p128r | 0.6349 | 84 | 0.6667 | 0.0292 | 0.0878 | 0.1364 |
| x86_g3_style1036_sft | 0.5709 | 55 | 0.4365 | 0.0325 | 0.0972 | 0.1364 |
| x86_g2c_cfgonly | 0.6063 | 72 | 0.5714 | 0.0201 | 0.0671 | 0.1104 |
| x86_g3_simko_eval_grpo | 0.6360 | 84 | 0.6667 | 0.0221 | 0.0716 | 0.1104 |
| x86_g2_graphtext | 0.6253 | 74 | 0.5873 | 0.0227 | 0.0691 | 0.0974 |
| x86_ref_base | 0.5383 | 26 | 0.2063 | 0.0156 | 0.0563 | 0.0909 |
| x86_g0_null | 0.5459 | 52 | 0.4127 | 0.0136 | 0.0494 | 0.0779 |
| x86_g1_textonly | 0.5193 | 41 | 0.3254 | 0.0052 | 0.0260 | 0.0519 |
| x86_g3_p128 | 0.4481 | 5 | 0.0397 | 0.0000 | 0.0000 | 0.0000 |

Interpretation:

- The clean G3 graph-only architecture is the best balanced model: high CodeBLEU, high compile@5, and competitive pass@10.
- Binary GRPO gives the best pass@10 but badly damages compile@k and CodeBLEU. It should be framed as an exploratory RL arm or harvest source, not as the flagship model.
- RS-SFT and repair arms often move pass behavior but damage compile and CodeBLEU. Their main value is complementarity.

## Union / Harvest Results

The strongest result is not a single checkpoint. It is the complementary coverage across independently adapted arms.

Latest all-arms harvest report:

- Prediction files used: 17
- Union tasks with at least one passing candidate: 43 / 154
- Zero-pass tasks remaining: 111 / 154
- Passing candidates total: 587
- Harvested rows after dedupe cap: 142
- All-plus-reference RS-SFT rows: 253
- Train-half rows: 131

Important progression:

- Earlier all-arms union: 39 / 154
- After B RS-SFT arm: 40 / 154
- After H100 ultralite repair arms: 43 / 154

New tasks added by H100 arms:

- Task 31, `34.dart`, `List<int> unique(List<int> l)`, covered by `style_repair_ultralite`
- Task 56, `61.dart`, `bool correctBracketing(String brackets)`, covered by `binary_repair_ultralite`
- Task 77, `82.dart`, `bool primeLength(String string)`, covered by `style_repair_ultralite` and `withB_ref_lite`

The paper framing should be:

> Individual adapted policies are weak and unstable as standalone models, but their error sets are partially complementary. Multi-arm candidate pooling raises the functional coverage ceiling from the best single arm to 43/154 tasks.

## Reranking Results

Reranking has two meanings and they should not be conflated.

1. Intra-model reranking:
   Generate k candidates from one model and select the best candidate using compile/shape/cluster heuristics.

2. Multi-arm candidate ensembling:
   Pool candidates from multiple checkpoints/configurations and apply reranking over the union.

Deployable reranker:

- `compile_cluster_vote`
- Uses real Dart compilation and candidate clustering/shape heuristics.
- Does not use hidden unit tests.

Oracle reranker:

- `stats_pass_oracle` or `test` mode.
- Uses unit-test outcomes to estimate the best possible selected pass@1 from a sampled pool.
- Not deployable unless tests are public.

H100 rerank results for the four newest weak arms:

| Arm | Deployable selected compile | Deployable selected pass@1 | Oracle selected pass@1 |
|---|---:|---:|---:|
| withB_ref_lite | 0.4351 | 0.0325 | 0.0519 |
| withB_ref_ultralite | 0.4351 | 0.0584 | 0.0779 |
| binary_repair_ultralite | 0.4026 | 0.0844 | 0.1104 |
| style_repair_ultralite | 0.3442 | 0.0844 | 0.1364 |

Interpretation:

- Compile-aware reranking recovers syntax/compilability strongly even for weak arms.
- It only partially recovers functional correctness.
- Oracle gaps show that selection is not the only bottleneck; many tasks still have no passing candidate at all.

## GRPO Diagnosis

GRPO did not fail because the code was simply broken. It found some pass@10 signal, but the reward and data regime push against the metric we care about.

Observed behavior:

- Binary GRPO improved pass@10 to 0.1818, the best standalone pass@10.
- It reduced CodeBLEU to 0.5438 and compile@5 to 0.2619.
- Subsequent RS-SFT repair from GRPO washed out the useful tail rather than preserving it.

Likely reason:

- GRPO optimizes expected per-sample reward.
- pass@k rewards diversity and tail coverage.
- Small groups, sparse correctness rewards, and tiny datasets encourage policy narrowing or movement toward partial/fragile solutions.

Paper-safe claim:

> Online RL provides evidence that correctness rewards can shift the model toward passing solutions, but under this small-data decompilation setting the gains are unstable and trade off sharply against compilability and surface similarity.

## Do Not Overclaim

Avoid saying:

- "We solved Dart decompilation."
- "GRPO improves the decompiler."
- "The final model is deployable."
- "Reranking alone solves pass@k."

Say instead:

- "Graph-conditioned decoding improves the balanced CodeBLEU/compile/pass profile."
- "Pass@k exposes failures hidden by CodeBLEU and compile@k."
- "Reward-tuned and RS-SFT arms expose complementary correct candidates, even when their standalone metrics regress."
- "Multi-arm candidate ensembling raises the recoverable functional coverage ceiling."

## Suggested Paper Thesis

The next paper should be an architecture-plus-evaluation paper:

> We introduce a graph-conditioned neural decompilation architecture for Dart AOT binaries and show that although individual fine-tuned policies remain brittle, graph conditioning and multi-arm candidate ensembling reveal complementary semantic recovery behavior that is invisible under CodeBLEU or compile@k alone.

## Suggested Paper Structure

1. Introduction
2. Background and Prior Work
3. Graph-Conditioned Decompilation Architecture
4. Training and Adaptation Strategies
5. Evaluation Protocol
6. Results
7. Reranking and Multi-Arm Ensembling
8. Error Analysis and Discussion
9. Threats to Validity
10. Conclusion

## Files Created For This Handoff

- `CLAUDE_SUMMARY_20260707.md`
- `PAPER_DRAFT_GRAPH_ENSEMBLE_20260707.md`
- `paper_graph_ensemble_refs.bib`

