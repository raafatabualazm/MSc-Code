# Graph-v2 Construction and Training Audit

Date: 2026-07-11

## Decision

The historical `_cfg` datasets, graph checkpoints, and graph-conditioned
prediction pools are not valid evidence for the confirmatory study. They were
built before the graph-v2 parser, DFG, edge-vocabulary, and encoder-to-decoder
bridge corrections. They must not be resumed or mixed with graph-v2 results.

The graph-v2 implementation is now suitable for a fresh confirmatory run, with
the limitations in this document stated explicitly. Passing the structural
preflight establishes implementation integrity; it does not establish that the
new architecture improves pass@k. That remains an empirical question for the
controlled multi-seed study.

## Fresh Corpus Audit

All graph-v2 files are rebuilt from the raw assembly field. Existing `cfg` and
`edges` fields are ignored. Each row records its assembly hash, extractor hash,
schema version, and integrity report.

| Dataset | Input | Valid captures | Source rejects | Duplicate/overlap drops | Final rows | Blocks | CFG edges | DFG edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval-Dart evaluation | 154 | 154 | 0 | 0 | 154 | 3,917 | 5,435 | 13,373 |
| Original SFT train | 1,081 | 941 | 140 | 171 | 770 | 42,095 | 59,606 | 137,346 |
| Original SFT validation | 114 | 83 | 15 | 16 | 83 | 5,795 | 8,262 | 23,294 |
| Synthetic source pool | 1,726 | 1,726 | 0 | 0 | 1,726 | 86,085 | 120,582 | 280,591 |

The historical 513/58 complete-case fallback is retired. The original SFT
source was recompiled with Dart 3.11.5 and disassembled with GDB 17.1. The first
train pass recovered 897 rows, a dependency/import retry recovered 40, and a
targeted entry-point retry recovered 4. The remaining 140 train and 15
validation fragments are missing companion declarations, malformed, or lack
source; the builder rejects them rather than inventing code. Exact normalized
source deduplication removed 168 train and 16 validation rows. A token-normalized
7-gram audit then removed three train programs near-duplicating four validation
programs. The final 770/83 split has zero exact or near source overlap internally
and zero exact or near overlap with the 154-task benchmark.

The evaluation and synthetic corpora remain complete. Closed unreachable
regions are pruned only after direct-target closure succeeds: 1 block in the
benchmark, 1,192 across rebuilt train, 31 across validation, and 7 across the
synthetic corpus. Every count is retained in row-level integrity metadata.
Rejecting context-dependent source fragments may still favor self-contained
programs and must be disclosed as a data-selection limitation.

The exact pinned-tokenizer audit passed across 137,892 blocks. Straight-line
runs are split into at most 20-instruction fallthrough sub-blocks; no instruction
is discarded. The largest block has 430 GraphCodeBERT code tokens, below the
510-token budget, so graph-v2 never relies on runtime token truncation. The
benchmark's largest graph has 458 blocks, 696 CFG edges and 3,708 DFG edges.
Graph-v2 preprocessing retains every DFG edge rather than selecting a
source-order-biased prefix.

Current immutable output hashes are in:

- `data/testing/grpo_data_graphv2.summary.json`
- `data/datasets/dart_all_graphv2_train.summary.json`
- `data/datasets/dart_all_graphv2_validation.summary.json`
- `data/datasets/synthetic_pool_graphv2.summary.json`
- `results/graph_v2_dataset_audit.json`
- `results/sft_source_train_validation_filter_graphv2.json`
- `results/sft_train_validation_overlap_graphv2.json`

## Critique Disposition

### CFG extraction

| Finding | Status | Resolution |
|---|---|---|
| GDB `static void ...` declarations parsed as instructions | Fixed | Parse only selected disassembly regions and reject symbol-list metadata. |
| Incomplete x86 condition set | Fixed | Added overflow, sign, parity, count-zero, and loop branch families; unknown branch-like mnemonics invalidate the row. |
| `=>` current-instruction marker skipped | Fixed | Marker is stripped while preserving the instruction and `<+0>` entry. |
| Indirect jump displacement treated as a direct target | Fixed | Memory-indirect operands do not produce direct targets. |
| Trap instructions fall through | Fixed | `ud2`, `int3`, `hlt`, `brk`, and `eret` terminate blocks. |
| ARM bytes and bare targets parsed incorrectly | Fixed | LLVM byte columns are removed and bare direct branch targets are supported. |
| Multiple/repeated GDB symbol regions collide | Fixed | Select the final requested `<+0>` function region before duplicate filtering. |
| Entry assumed from stream/address order | Fixed | Canonical GDB `<+offset>` order and the true `<+0>` entry are recorded. |
| Address-based loop heuristic mislabels non-monotonic dumps | Fixed | Loop backedges are assigned only through the dominator chain. |
| Unreachable or unresolved blocks can pass integrity | Fixed | Entry reachability, direct-target closure, endpoints, duplicates, and nonempty blocks are mandatory. Closed dead regions may be pruned and reported; unresolved branches are never laundered by pruning. |
| Metadata becomes stale after edge relabeling | Fixed | Predecessors, successors, and edge-type metadata are rebuilt from final edges. |
| Old precomputed files survive extractor fixes | Fixed | `build_graph_v2_jsonl.py` always reconstructs from assembly and writes atomically. |
| GDB output clipped to a 4,000-character diagnostic tail | Fixed | GDB capture is now unlimited; `rebuild_sft_assembly.py` recompiles source rows and validates complete control-flow closure. |
| External runtime tail exits treated as truncated same-function branches | Fixed | Named external targets are counted as external exits; missing `<function+offset>` targets remain fatal. |
| Instruction-count splitting can still overflow the encoder token window | Fixed for graph-v2 | Long straight-line runs are split at 20 instructions and an exact pinned-tokenizer audit fails above 510 code tokens; observed maximum is 430. |
| Jump-table/indirect-target recovery | Open | Indirect branches are represented as sinks, not fabricated direct edges. Full jump-table recovery remains future work. |

### DFG construction

| Finding | Status | Resolution |
|---|---|---|
| Calls have no effects and preserve stale return definitions | Fixed | Calls consume Dart argument registers, define/clobber volatile registers, and define return registers. |
| Generic ABI used instead of Dart ABI | Fixed | x64 uses `rdi,rsi,rdx,rbx,r8,r9`; ARM64 uses `x1,x2,x3,x5,x6,x7`, following Dart VM constants. |
| Return blocks do not read return values | Fixed | `ret` reads `rax`/`x0`, permitting def-to-return flow. |
| x86 8-bit aliases and `setcc` are invisible | Fixed | Low/high byte aliases, flags, `setcc`, and `cmovcc` are modeled. |
| Implicit integer operands omitted | Fixed | Integer multiply/divide and sign-extension families model their implicit registers. |
| Flags and branch predicates omitted | Fixed | Flags are a pseudo-location consumed by conditional materialization/branches. |
| Stack store/load flow is absent | Partly fixed | Stable frame slots such as `[rbp-8]` and `[x29+8]` are symbolic locations. Moving `rsp`/`sp` offsets are deliberately excluded until stack-delta analysis exists. |
| Plumbing registers create dense noise | Fixed for emitted edges | `sp/rsp`, frame pointers, link/thread/pool registers are filtered from DFG edge locations. Stable frame-slot locations remain. |
| Fixed 64-pass convergence cap | Fixed | Reaching definitions use a worklist to a true fixed point. |
| Blind low edge cap | Fixed for graph-v2 | `max_dataflow_edges=0` keeps every DFG edge; the audit would report any future positive-cap saturation. |
| Dataflow aliases the `call` edge type | Fixed | `dataflow` has dedicated forward type 8; reverse relations occupy 9-17. |
| Register identity and multiplicity reach the GNN | Open | Locations and dependency counts are retained in JSON, but PyG currently consumes only the block-pair `dataflow` type. |
| Heap/field aliasing | Open | General `[object+offset]` may-alias analysis is not implemented. |
| SIMD/floating-point def-use | Open | Current DFG is general-purpose-register, flags, call, return, and stable-frame-slot may-flow only. |
| SSA-level/instruction-level DFG | Open by design | The graph is a block-level may-reaching-definition relation; intra-block flow remains in the block encoder. |

The Dart register choices are grounded in the official
[x64 VM constants](https://raw.githubusercontent.com/dart-lang/sdk/main/runtime/vm/constants_x64.h)
and [ARM64 VM constants](https://raw.githubusercontent.com/dart-lang/sdk/main/runtime/vm/constants_arm64.h).

### Graph consumer and bridge

| Finding | Status | Resolution |
|---|---|---|
| Invalid endpoints can leak across functions after PyG batching | Fixed | Every edge endpoint is validated against its graph before batching; forward-time batch pointers and graph sizes are also asserted. |
| GNN message passing is forward-only | Fixed for graph-v2 | Every selected forward relation receives a distinct reverse relation. |
| Global attention is blind to block order | Fixed for graph-v2 | Canonical block-order sinusoidal positions are injected before global attention with a learned scale. |
| Legacy DFG `<unk>` tokens are non-informative | Explicitly retired | Graph-v2 uses `dfg_mode=edges`; legacy mode remains only for old-checkpoint reproduction and must not be described as GraphCodeBERT graph-guided attention. |
| RoBERTa positions use untrained rows | Fixed for graph-v2 | New arms use `position_scheme=roberta`; strict audit enforces the 512-token block contract. |
| One-node graph yields identical prefix slots | Fixed | Learned queries remain in the prefix residual: `query + cross_attention + FFN`. |
| Arbitrary graphs compressed to fixed 16 slots | Fixed for the new study | Prefix capacity is dynamic from 4 to 64 active slots using `4*ceil(log2(max(blocks,2)))`, with padding masked. |
| One scalar gate throttles all prefix slots | Fixed for the new study | Graph-v2 uses one learned gate per prefix slot and logs the active gate values. |
| Wide-prefix scale mismatch | Fixed for the new study | Prefix vectors are RMS-matched to Qwen token embeddings. |
| Repeated decoder cross-attention to all block states | Open | The current bridge remains an input-prefix resampler. Adding decoder-layer cross-attention is a separate, checkpoint-breaking architecture and future experiment. |
| Multiple semantic tokens per block / auxiliary graph losses | Open | Current node state is one GraphCodeBERT CLS vector per block; richer role tokens and structural objectives remain future work. |

## Causal Experiment Contract

The new study compares equal-data, equal-budget graph arms using the same block
encoder and prefix bridge:

1. Signature-only untuned Qwen baseline.
2. Untuned Qwen with cleaned assembly.
3. SFT text-only cleaned assembly.
4. Prefix path with no graph edges.
5. Prefix path with CFG edges only.
6. Prefix path with CFG+DFG edges.
7. Shuffled-edge prefix control.
8. CFG+DFG plus raw assembly, to measure channel competition.

The no-edge, CFG-only, and CFG+DFG arms run with training/generation seeds 42,
43, and 44. All graph-only arms use the same dynamic prefix, reverse relations,
block positions, per-token gates, RMS matching, training data, prompt budget,
and candidate budget. This isolates topology and DFG more cleanly than the
historical graph-versus-text comparison.

`prefix_no_gine` is available through `--full_matrix` but is lower priority
than completing the three-seed causal comparison inside the 20-hour budget.
Future graph-sensitivity work should add shuffled edge types, shuffled blocks,
an unrelated-function graph, and a zero-gate checkpoint replay.

## GRPO Audit

The confirmatory reward path is intentionally simpler than the historical
reward variants:

`R(x,y) = +1` when the generated function makes the complete private Dart
harness exit successfully, and `R(x,y) = -1` otherwise. Within each 16-sample
group, `A_i = R_i - mean(R_group)`. The sequence-normalized policy loss is the
negative mean of `A_i * mean_t(log pi(y_i,t | x,y_i,<t))` over update-eligible
samples. Groups with constant reward provide zero policy signal.

- tests are never inserted into the policy prompt;
- the 154 evaluation tasks are never used for SFT or GRPO;
- GRPO uses a synthetic-only 256-task subset disjoint from benchmark tasks;
- one evaluator-aligned `dart run` executes the complete harness per completion;
- reward is `+1` only for a complete pass and `-1` otherwise;
- no partial credit, overlap reward, uniqueness bonus, entropy bonus, SimKO,
  or top-k smoothing is active;
- group size 16, temperature 0.7, `top_p=1.0`, one epoch, LR `5e-7`;
- overlong generations are excluded from policy updates;
- gradient accumulation flushes partial final windows correctly;
- the local block encoder is frozen while GNN, projection, prefix resampler,
  per-token gates, and LoRA policy parameters remain trainable;
- strict graph checks reject missing, empty, or misaligned graph batches.

The implemented one-update objective is group-relative REINFORCE. The detached
old log-probability comes from the same scoring pass, so the ratio is one and
PPO clipping is inert. It must not be called a full PPO/GRPO trust-region
implementation in the paper. GRPO runs only if reward preflight observes mixed
outcomes in at least 5 percent of prompt groups.

## Verification Evidence

The following passed on 2026-07-11:

- 29 graph-v2 adversarial/gradient unit tests;
- 16 protocol-integrity tests;
- 62 preprocessing and tensor-builder checks;
- all GRPO objective, reward, chunking, and accumulation self-checks;
- synthetic generator compile/disassembly self-test;
- deterministic rebuild and audit of all four graph-v2 datasets;
- zero leaked scoring-test rows in 154 prompts;
- zero exact or near overlap between train and validation, and between each training source and the benchmark;
- 1,725-row synthetic static reward audit with zero failures;
- 1,725/1,725 executable reference rewards passed locally with zero failures.

The executable parity audit remains mandatory on the remote CPU before paid GPU
work because Dart/platform drift can change the harness outcome. Run the command
in `RUNBOOK_LEAKAGE_FREE_20H.md` and stop on any failure.

## Remaining Claims Boundary

The code and data are now internally auditable, but no new graph-v2 model has
yet been trained. Therefore:

- do not claim graph-v2 improves pass@k until the controlled runs finish;
- do not reuse historical graph metrics as graph-v2 evidence;
- do not call the DFG precise, SSA-based, heap-aware, or interprocedural;
- do not claim ARM64 effectiveness from parser unit tests alone;
- do not claim extractor precision/recall against Ghidra/IDA without a
  hand-labeled external graph study;
- report that 155 context-dependent, malformed, or source-less SFT fragments
  remain rejected and may bias the corpus toward self-contained programs;
- report multi-seed variability and task-level paired effects after rerunning;
- keep decoder-layer cross-attention and richer node tokens as future work.
