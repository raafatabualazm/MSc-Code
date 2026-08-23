# Graph-Conditioned Neural Decompilation of Dart AOT Binaries: Pass@k, Reward Adaptation, and Multi-Arm Candidate Ensembling

Raafat Abualazm, Ayman AboElHassan, Amr G. Wassal

## Abstract

Neural decompilation is increasingly framed as a code-generation problem, but recent evidence shows that surface similarity and syntactic validity can diverge sharply from functional correctness. Prior work on Dart AOT decompilation established that small fine-tuned language models can produce idiomatic Dart and that CodeBLEU and compile@k are useful but incomplete indicators of decompilation quality. A subsequent empirical study introduced pass@k evaluation for HumanEval-Dart and found that conventional fine-tuning often fails to improve functional correctness, even when CodeBLEU and compile@k increase. This paper investigates whether explicit binary structure and inference-time candidate selection can recover functional behavior that sequence-only fine-tuning misses.

We introduce a graph-conditioned decompilation architecture that encodes control-flow and data-flow structure from x86-64 Dart AOT assembly using a GraphCodeBERT-based encoder and conditions a Qwen3-8B decoder through learned graph prefix tokens. We evaluate architecture ablations, reward-tuned GRPO variants, rejection-sampling supervised fine-tuning, deployable compile-aware reranking, and multi-arm candidate ensembling on a 154-task HumanEval-Dart benchmark. The best balanced graph-conditioned model reaches 0.638 CodeBLEU, compile@5 of 0.667, and pass@10 of 0.162. A binary-reward GRPO arm achieves the highest standalone pass@10, 0.182, but sharply regresses compile@5 to 0.262, demonstrating the instability of sparse correctness rewards in this setting. Although no single adapted arm dominates, their errors are complementary: pooling 17 independently adapted arms yields at least one passing candidate for 43 of 154 tasks, compared with 25 tasks for the clean graph-conditioned arm and 28 for the best GRPO arm. Finally, a deployable compile-cluster reranker raises selected compile rates above 0.40 for several weak arms, while oracle reranking exposes the remaining candidate-generation ceiling. These results suggest that graph conditioning and multi-policy candidate ensembling are more reliable research directions for Dart AOT decompilation than single-checkpoint reward optimization alone.

## 1. Introduction

Decompilation reconstructs high-level source-like programs from low-level binaries. For reverse engineering, malware analysis, security auditing, and software maintenance, the ideal output is not merely compilable pseudocode but source code that is readable, idiomatic, and functionally faithful to the original program. Traditional decompilers such as Ghidra and Hex-Rays recover useful control and data-flow information, but their outputs often use generic identifiers, low-level control structures, and C-like representations that are far removed from the original source language.

Recent neural decompilation work treats decompilation as a translation or code-generation task. Large language models and specialized sequence-to-sequence models have improved readability and, in some settings, re-executability for C binaries. However, modern managed languages remain underexplored. Dart is an especially challenging target: production Flutter applications are compiled Ahead-of-Time (AOT), and the resulting optimized native code reflects type specialization, inlining, and runtime-specific object representations. Prior Dart-focused work showed that small specialized models can generate readable Dart and can approach much larger code models on CodeBLEU under limited-data conditions [@abualazm2026idiomatic]. A larger evaluation study later showed that this apparent success is incomplete: fine-tuning can improve CodeBLEU and compile@k while failing to improve, or even regressing, pass@k functional correctness [@abualazm2026tosem].

This paper starts from that negative result. If sequence-only fine-tuning is unreliable for functional correctness, what additional signals and inference strategies can expose or improve semantic recovery? We investigate two complementary ideas. First, decompilation is not plain text translation: assembly has explicit control-flow and data-flow structure. We therefore condition a causal decoder on graph-derived representations built from extracted basic blocks, control-flow edges, and lightweight data-flow edges. Second, pass@k is a tail metric: it rewards the existence of at least one correct candidate among many. A single checkpoint optimized for expected per-sample reward may narrow its output distribution and reduce the diversity needed for pass@k. We therefore distinguish single-model performance from multi-arm candidate coverage and reranking.

The central finding is that graph conditioning improves the balanced decompilation profile, but the largest functional gains appear as complementary candidates spread across multiple imperfect adaptation arms. The best standalone pass@10 is obtained by a GRPO arm, but it damages compile@k and CodeBLEU. The clean graph-conditioned architecture remains the best balanced model. Meanwhile, a union of candidates from 17 arms covers 43 tasks with at least one passing candidate, substantially more than any single arm. This reframes the problem: current models often contain correct solutions in their sampled tails, but no single policy or reranker fully captures them. Clean single-arm comparisons support the architectural claim, while reranking and multi-arm union results are candidate-discovery and selection analyses rather than clean generalization evidence.

## 2. Background and Prior Work

### 2.1 Dart AOT Decompilation

Dart AOT compilation converts high-level Dart functions into optimized native code. Compared with unoptimized C or toy algorithmic programs, this setting introduces several difficulties: dynamic runtime conventions, object layouts, tagged values, inlining, allocation paths, and type-specialized operations. Prior Dart decompilation work introduced the first idiomatic Dart neural decompilation study and evaluated small specialized models using CodeBLEU and compile@k [@abualazm2026idiomatic]. That work established the feasibility of producing readable Dart but explicitly left semantic pass@k evaluation for future work.

The arXiv preprint "Evaluating Fine-Tuning and Metrics for Neural Decompilation of Dart AOT Binaries", currently under TOSEM review, addressed this gap by introducing HumanEval-Dart, a 154-task pass@k benchmark for Dart decompilation, and by showing that fine-tuning can improve surface metrics while regressing functional correctness [@abualazm2026tosem]. This paper builds directly on those findings. Rather than treating CodeBLEU or compile@k as sufficient, we use pass@k and task-level coverage as the primary measures of semantic recovery.

### 2.2 Metric Divergence

CodeBLEU captures lexical, syntactic, and data-flow similarity to a reference implementation. compile@k measures whether at least one of k generated candidates is syntactically valid Dart. pass@k measures whether at least one of k candidates passes unit tests. These metrics answer different questions. A decompiler can produce reference-like code that does not compile; compilable code that fails tests; or a low-CodeBLEU implementation that is functionally correct. The TOSEM study found that metric divergence is not incidental: fine-tuned variants can improve CodeBLEU and compile@k while pass@k regresses [@abualazm2026tosem].

This motivates two design decisions in the present study. First, pass@k is the primary semantic metric. Second, we report CodeBLEU and compile@k as supporting metrics, not as substitutes for correctness.

### 2.3 Graph Conditioning

Assembly is structured. Basic blocks, control-flow edges, and data-flow relationships provide information that may be obscured in a flat token sequence. We therefore use a GraphCodeBERT-based local encoder to process extracted graph representations, and expose the graph representation to a Qwen3-8B decoder through learned prefix tokens. The key hypothesis is not that the graph alone solves decompilation, but that graph conditioning provides a useful structural prior that stabilizes syntax and control-flow reconstruction.

### 2.4 Reward Adaptation and Candidate Selection

Reward-based fine-tuning such as GRPO can optimize task-specific correctness signals, but pass@k creates a tension: the metric rewards tail diversity, while online reward optimization often sharpens the policy toward high-reward modes. We therefore analyze GRPO as one adaptation arm rather than assuming it should dominate. We also evaluate rejection-sampling SFT (RS-SFT), deployable reranking, oracle reranking, and multi-arm candidate ensembling.

## 3. Method

### 3.1 Task and Data

The benchmark consists of 154 HumanEval-Dart tasks with function signatures, reference Dart implementations, AOT-derived x86-64 assembly, CFG/DFG-derived graph features, and unit tests. compile@k is evaluated on a 126-task compile/CodeBLEU set, and pass@k is evaluated on the 154-task HumanEval-Dart pass set.

### 3.2 Graph Extraction

The graph pipeline extracts:

- Basic blocks from assembly using jump targets and fall-through structure.
- Control-flow edges between blocks.
- Lightweight data-flow edges tracking register/value relationships where available.
- Per-block instruction summaries with bounded instruction counts.

The graph records are merged into the original JSONL rows without dropping fields such as `dart_source`, `tests`, `dart_function_signature`, and `assembly`. This is important because earlier conversion code could create graph-only records and accidentally destroy the SFT/GRPO fields required by training and evaluation.

### 3.3 Model Architecture

The best clean architecture uses:

- GraphCodeBERT as the graph/local block encoder.
- Qwen3-8B-Base as the causal decoder.
- Learned graph prefix tokens injected into the decoder context.
- LoRA on both encoder and decoder with rank 64 and alpha 128.

The primary graph-conditioned configuration is:

- `qwen_prefix_tokens = 16`
- `qwen_prefix_rms_match = 0` for the reported 16-prefix G3 row
- `prompt_assembly_mode = none`
- `prompt_fit_assembly = 1` (inert when `prompt_assembly_mode = none`)
- `dfg_mode = edges`
- `position_scheme = roberta`
- `auto_cfg = 0`
- `max_block_instrs = 24`
- `GRAPH_MAX_DATAFLOW_EDGES = 4096`

The `prompt_assembly_mode = none` setting is deliberate: the model receives structure through the graph channel rather than duplicating long assembly text in the decoder prompt. Prefix RMS matching is reported as a stability mechanism for the wide-prefix ablation, not as an assumed property of the archived 16-prefix G3 row.

### 3.4 Adaptation Arms

We evaluate several adaptation strategies:

- Null and text-only controls.
- Graph-text and CFG-only variants.
- Clean graph-only G3 architecture.
- Wider prefix variants.
- GRPO reward tuning with binary/full-pass reward variants.
- RS-SFT from all passing candidates harvested across arms.
- Ultralite repair arms from GRPO, RS-SFT, and style-SFT checkpoints.

The purpose is not to find one magical checkpoint but to map how each intervention changes the tradeoff among CodeBLEU, compile@k, and pass@k.

### 3.5 Reranking and Ensembling

We distinguish:

- Intra-model reranking: generate k candidates from one checkpoint and select one candidate with deployable heuristics.
- Multi-arm candidate ensembling: pool candidates from multiple independently adapted checkpoints, then measure union coverage or rerank over the pool.

The deployable reranker, `compile_cluster_vote`, uses Dart compilation, structural heuristics, and candidate clustering. It does not use hidden tests. Oracle reranking uses pass/fail outcomes only to estimate the upper bound available in a sampled candidate pool.

## 4. Experimental Setup

All main experiments use Qwen3-8B-Base as decoder and GraphCodeBERT as encoder. Unless otherwise stated, evaluation uses:

- 5 samples for compile@k/CodeBLEU.
- 10 samples for pass@k.
- Maximum generation length 768 tokens.
- Decoder prompt budget 2048 tokens.
- LoRA rank 64, alpha 128.
- Graph prefix tokens 16.

We report:

- CodeBLEU on the compile/CodeBLEU set.
- Compiled-only CodeBLEU, counting only tasks with at least one compilable candidate.
- compile@1 and compile@5.
- pass@1, pass@5, and pass@10.
- Task-level union coverage across arms.
- Deployable and oracle reranking results.

## 5. Results

### 5.1 Standalone Model Results

| Arm | CodeBLEU | Compiled Rows | compile@5 | pass@1 | pass@5 | pass@10 |
|---|---:|---:|---:|---:|---:|---:|
| binary GRPO | 0.5438 | 33 | 0.2619 | 0.0409 | 0.1280 | 0.1818 |
| RS-SFT all-arms | 0.5740 | 54 | 0.4286 | 0.0474 | 0.1296 | 0.1688 |
| binary RS-SFT ref | 0.5323 | 44 | 0.3492 | 0.0357 | 0.1121 | 0.1558 |
| clean G3 graph-only | 0.6383 | 84 | 0.6667 | 0.0273 | 0.1028 | 0.1623 |
| style repair ultralite | 0.5299 | 25 | 0.1984 | 0.0221 | 0.0849 | 0.1364 |
| G3 p128r | 0.6349 | 84 | 0.6667 | 0.0292 | 0.0878 | 0.1364 |
| style1036 SFT | 0.5709 | 55 | 0.4365 | 0.0325 | 0.0972 | 0.1364 |
| CFG-only | 0.6063 | 72 | 0.5714 | 0.0201 | 0.0671 | 0.1104 |
| SimKO-style GRPO | 0.6360 | 84 | 0.6667 | 0.0221 | 0.0716 | 0.1104 |
| graph-text | 0.6253 | 74 | 0.5873 | 0.0227 | 0.0691 | 0.0974 |
| base reference | 0.5383 | 26 | 0.2063 | 0.0156 | 0.0563 | 0.0909 |
| null graph | 0.5459 | 52 | 0.4127 | 0.0136 | 0.0494 | 0.0779 |
| text-only | 0.5193 | 41 | 0.3254 | 0.0052 | 0.0260 | 0.0519 |
| p128 without RMS | 0.4481 | 5 | 0.0397 | 0.0000 | 0.0000 | 0.0000 |

The clean G3 graph-only model is the best balanced checkpoint. It achieves the strongest combination of CodeBLEU, compiled rows, compile@5, and competitive pass@10. The binary GRPO arm gives the highest standalone pass@10 but severely regresses compile@5 and CodeBLEU, making it unsuitable as the flagship model.

### 5.2 Architecture Ablations

The graph-conditioned arms outperform text-only and null controls on pass@10 and compile@k. The clean G3 graph-only model reaches pass@10 of 0.1623, compared with 0.0519 for text-only and 0.0779 for null graph. This suggests that the graph channel carries useful information beyond the decoder prior.

The current ablations do not prove that DFG edges help on their own. CFG-only outperforms graph-text on pass@10 (0.1104 versus 0.0974), and these rows do not isolate DFG under the same graph-only regime as G3. We therefore interpret the result as evidence for graph conditioning overall, while treating DFG contribution as unresolved and requiring a future controlled ablation.

The p128 experiment shows that simply increasing prefix capacity is unsafe. Without RMS matching, the 128-prefix variant collapses to zero pass@k and very low compile@5. With RMS matching, p128r recovers compile@5 and CodeBLEU but still underperforms the 16-prefix G3 model on pass@10. The lesson is that graph-prefix scale and conditioning stability matter more than raw prefix width.

### 5.3 GRPO and RS-SFT

GRPO improves standalone pass@10 to 0.1818 but damages compile@5, reducing it to 0.2619. RS-SFT all-arms improves pass@1 and pass@5 relative to clean G3 but lowers compile@5 and CodeBLEU. Attempts to repair GRPO with RS-SFT recover some compile behavior but wash out the useful pass@10 tail.

This supports a careful interpretation: correctness rewards can move probability mass toward passing solutions, but the setting is small, sparse, and unstable. GRPO optimizes expected per-sample reward, whereas pass@k rewards the existence of diverse correct candidates in the tail. The objectives are related but not identical.

### 5.4 Multi-Arm Candidate Ensembling

The strongest result appears when candidates are pooled across arms. Across 17 prediction files:

- 43 / 154 tasks have at least one passing candidate.
- 111 / 154 tasks remain zero-pass.
- 587 passing candidates are found before deduplication.
- 142 RS-SFT rows remain after dedupe and per-task capping.
- 253 rows are available in the all-plus-reference RS-SFT set.

The multi-arm pool combines the 14 primary model families in the standalone table with three auxiliary repair/rerank pools: `withB ref lite`, `withB ref ultralite`, and `binary repair ultralite`.

This is substantially higher than the clean G3 single-arm coverage of 25 tasks and the best GRPO single-arm coverage of 28 tasks. These are candidate-coverage counts, and the comparison is not sample-budget matched: the union pools many more candidates than a single 10-sample arm. We therefore interpret it as a coverage ceiling and diversity diagnostic, not as evidence that a 17-arm system is the deployable default. The multi-arm result shows that the models fail differently. Even weak standalone arms can add new tasks to the union.

The H100 ultralite arms added three new tasks to the union:

- `List<int> unique(List<int> l)`
- `bool correctBracketing(String brackets)`
- `bool primeLength(String string)`

This complementarity motivates treating candidate ensembling as a first-class inference-time method rather than as an afterthought.

### 5.5 Reranking

Deployable compile-cluster reranking improves selected compile rates substantially. Rates are measured on the 154-task pass-set candidate pools, not the 126-task compile/CodeBLEU set. Original compile denotes the first/original candidate, selected compile denotes the compile-cluster-selected candidate, and oracle selected pass is a hidden-test upper-bound diagnostic.

| Arm | Original compile | Selected compile | Original pass@1 | Selected pass@1 | Oracle selected pass@1 |
|---|---:|---:|---:|---:|---:|
| withB_ref_lite | 0.0584 | 0.4351 | 0.0130 | 0.0325 | 0.0519 |
| withB_ref_ultralite | 0.0909 | 0.4351 | 0.0130 | 0.0584 | 0.0779 |
| binary_repair_ultralite | 0.0649 | 0.4026 | 0.0000 | 0.0844 | 0.1104 |
| style_repair_ultralite | 0.0779 | 0.3442 | 0.0260 | 0.0844 | 0.1364 |

The deployable reranker does not reach the oracle ceiling, but it materially improves selection. The oracle gap quantifies how much additional benefit would require better semantic selection or public tests. More importantly, for several weak arms, the reranker recovers enough compile behavior to make their candidate pools useful for downstream harvesting and analysis.

The large compile jump should be read as a reranking effect, not as a new model compile@1 probability: the reranker chooses among an existing sampled pool, so selected compile can be much higher than the original first-candidate compile rate.

## 6. Discussion

### 6.1 What Graph Conditioning Buys

Graph conditioning improves the balanced model profile. It does not make pass@k high in absolute terms, but it moves the model beyond text-only decoding and stabilizes compile behavior. For a decompilation system, this matters: a checkpoint that slightly improves pass@10 while destroying compile@k is not a practical decompiler. The clean G3 model is therefore the best architectural contribution.

### 6.2 Why GRPO Is Not Enough

The GRPO result is tempting because it has the highest pass@10. However, its compile and CodeBLEU regressions show that it is not simply "better." The likely failure mode is reward-policy mismatch. Correctness rewards identify useful candidates, but repeated policy updates narrow the generation distribution and can damage syntax and style. The best use of GRPO in this setting may be as a candidate discovery arm rather than as the final deployable checkpoint.

### 6.3 Why Ensembling Matters

pass@k is a coverage metric. If different adaptation strategies solve different tasks, then a multi-arm ensemble can reveal a larger recoverable set than any single model. The union result, 43/154, demonstrates that several apparently weak arms are still scientifically valuable because they contribute unique passing candidates.

### 6.4 Reranking as a Bridge Between Research and Deployment

Oracle reranking measures the sampled pool ceiling; deployable reranking tries to approximate it using available signals. Our compile-cluster reranker strongly improves selected compile rates but only partially improves selected pass@1. This suggests a clear next step: semantic rerankers that do not require hidden tests, such as learned failure predictors, signature/type consistency models, lightweight symbolic checks, or public-example execution when available.

## 7. Artifact and Reproducibility Notes

This draft is backed by concrete, checkable materials rather than a description of what an artifact should contain. The working repository includes the full training and evaluation code, JSONL data manifests, generated prediction files, sweep summaries, and per-candidate metric CSVs for the main table and paired tests. A consolidated, deduplicated log of the command-line invocations is maintained alongside the code. A pinned local verification environment (`env_manifest_local_verification.txt`, 468 packages; Python 3.12.7; `torch` 2.11.0+cu128, `transformers` 5.9.0, `peft` 0.18.0, `torch-geometric` 2.7.0, `accelerate` 1.6.0) runs an offline regression suite: 56 preprocessing checks and 9 GRPO gradient-equivalence self-checks.

Two gaps remain. First, checkpoint weights and SHA-256 hashes are not yet part of the archived materials: the 8B table checkpoints were trained on ephemeral rented GPU pods, and only prediction outputs were routinely retrieved to control storage and transfer cost. The pod-side training environment is likewise not yet captured as a pinned manifest. Second, the command log is complete for the standalone table arms and the union/harvesting pipeline, but the smaller repair-set arms used only for reranking were run interactively on a pod and their literal invocations are still being reconstructed from pod-side logs.

We commit to releasing the complete artifact -- code, data manifests, the full run-command log, both environment manifests, checkpoint SHA-256 hashes, and, hosting permitting, the LoRA adapter weights themselves -- at an anonymized repository URL upon acceptance, consistent with double-blind review.

## 8. Threats to Validity

Dataset size remains small: 154 pass@k tasks are enough to expose failure patterns but not enough to claim deployment readiness. Several training and evaluation arms share data, so harvest and ensemble results should be treated as optimization/coverage analysis rather than clean generalization. Some synthetic data is style-matched, but synthetic distributions may not reflect production Flutter binaries. The graph extractor is lightweight and may miss important data-flow relationships. Finally, oracle reranking uses test outcomes and is not deployable unless those tests are public; it is reported only as an upper bound on candidate-pool quality.

## 9. Future Work

The most promising next directions are:

1. Larger and more realistic Dart/Flutter AOT datasets.
2. Optimization-matched Swift and Dart cross-lingual experiments.
3. Learned semantic rerankers that approximate oracle selection without hidden tests.
4. Reasoning-SFT with explicit, structured `<think>...</think>` targets followed by fenced Dart code, rather than enabling untrained reasoning at inference.
5. Graph improvements, including richer DFG extraction, block summarization, and alignment between assembly blocks and source-level constructs.
6. Multi-arm ensemble evaluation at k=25 and k=50 to estimate the true candidate-generation ceiling.
7. Clean held-out train/test splits for RS-SFT and harvest experiments, separating optimization experiments from generalization claims.

## 10. Conclusion

This study extends Dart AOT neural decompilation beyond sequence-only fine-tuning by adding graph-conditioned decoding, reward adaptation, rejection-sampling fine-tuning, reranking, and multi-arm candidate ensembling. The clean graph-conditioned G3 model is the best balanced architecture, while binary GRPO provides the strongest but unstable standalone pass@10. The most important finding is complementarity: multiple imperfect arms collectively cover substantially more tasks than any single checkpoint. This suggests that future neural decompilers should be evaluated not only as single models but also as candidate-generation systems paired with deployable selection mechanisms.

## References

[@abualazm2026idiomatic] Raafat Abualazm and Ayman AboElHassan. 2026. LLMs as Idiomatic Decompilers: Recovering High-Level Code from x86-64 Assembly for Dart. arXiv:2604.02278. Accepted at SANER 2026 ERA Track.

[@abualazm2026tosem] Raafat Abualazm, Ayman AboElHassan, and Amr G. Wassal. 2026. Evaluating Fine-Tuning and Metrics for Neural Decompilation of Dart AOT Binaries. arXiv:2607.06125. Under review at TOSEM.


