# ARM64 Graph-v2.1 Data Preparation

Prepared on 2026-07-12 using the CPU-only DigitalOcean host at
`root@167.172.150.125:/workspace`. The final artifacts were pulled back and
verified locally. No model training has been run on these files yet.

## Protocol

- Input: 1,714 real Flutter Android ARM64 release-binary slices with synthetic
  algorithmic Dart semantics and executable tests.
- Schema: `antigravity-graph-v2.1`.
- Maximum instructions per block: 20.
- DFG edge cap: 0 (all edges retained).
- GraphCodeBERT tokenizer revision:
  `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`.
- Split: seed 42, 80/20, stratified jointly by task category and assembly
  length, with normalized-source and 7-token-shingle near-clone components kept
  wholly on one side (`Jaccard >= 0.8`).
- Scoring tests remain dataset fields for the external evaluator and must never
  enter policy prompts.

## Graph-v2.1 Corrections

The first remote build exposed two ARM-specific correctness failures and was
discarded:

1. NetworkX was absent, which silently disabled dominator analysis and produced
   zero loop-backedge labels. Graph construction and auditing now fail closed
   without NetworkX; the prepared environment pins `networkx==3.6.1`.
2. Flutter slices may contain an entry wrapper, the implementation body, local
   helpers, and closures as multiple symbol ranges. A single-entry CFG pruned
   legitimate semantic bodies. Graph-v2.1 treats every recorded symbol range as
   a legitimate graph entry, keeps the exact top-level symbol as the primary
   entry, computes dominators from a virtual analysis root, and adds explicit
   `call` edges for statically resolved in-slice `bl`/`call` targets.

The exact extractor used by the ongoing x86 graph-v2 study is preserved under
`archive/graph_v2_x86_20260711/`. Do not replace scripts on that x86 pod while
its runs are active; graph-v2.1 has a different schema and extractor hash.

## Final Counts

| Item | Full | Train | Evaluation |
|---|---:|---:|---:|
| Rows | 1,714 | 1,371 | 343 |
| Blocks | 91,895 | 73,502 | 18,393 |
| CFG edges | 126,931 | 101,596 | 25,335 |
| DFG edges | 306,958 | 246,851 | 60,107 |

Additional full-corpus facts:

- 2,339 symbol entries, all resolved.
- 2,963 dominator-confirmed loop backedges across 1,491 rows.
- 100 statically resolved in-slice call edges.
- Seven unreachable runtime-tail blocks pruned and recorded.
- Zero rejected rows.
- Maximum block token count: 349 of the allowed 510.
- Maximum DFG edges in one row: 1,954; no cap was applied.

Evaluation length distribution:

| Instructions | Tasks |
|---|---:|
| `<50` | 19 |
| `50-99` | 46 |
| `100-199` | 110 |
| `200-499` | 134 |
| `500+` | 34 |

All 17 semantic categories are represented in evaluation.

## Leakage Audits

- Train/evaluation exact source overlap: 0.
- Train/evaluation near-source overlap at 0.8: 0.
- ARM64 full corpus versus x86 154-task benchmark exact overlap: 0.
- ARM64 full corpus versus x86 benchmark near overlap at 0.8: 0.
- Eight near-clone pairs exist inside the ARM64 corpus; their seven connected
  components are kept wholly in train or evaluation.

## Immutable Files

- `data/datasets/arm64_graphv2/flutter_function_assembly_pool_graphv2.jsonl`
  - SHA-256: `bd64a1e8d24dc93a89f05d7f58cbaa9b4a09c7232e0a85555561f9dbeaa1519b`
- `data/datasets/arm64_graphv2/flutter_train_graphv2.jsonl`
  - SHA-256: `f21782dd60edc11988867659dd2d16a5f6b6d2f550594cae09ad7cf92b68dcb7`
- `data/datasets/arm64_graphv2/flutter_eval_graphv2.jsonl`
  - SHA-256: `864dc0bb7e9ee305ba0fc4be6e5d5ecbbeb7c17fd15bb3e41cfcc6d9aaf65fac`
- `data/datasets/arm64_graphv2/flutter_function_assembly_pool_graphv2.rejected.jsonl`
  - Empty-file SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The complete internal file list is in `results/arm64_graphv21_SHA256SUMS`.

## Audits and Package

- Structural/tokenizer audit: `results/arm64_graph_v2_dataset_audit.json`
- Complexity audit: `results/arm64_graph_v2_complexity_audit.json`
- Train/evaluation overlap: `results/arm64_graphv2_train_eval_overlap.json`
- ARM64/x86 overlap: `results/arm64_graphv2_x86_benchmark_overlap.json`
- Split manifest: `data/datasets/arm64_graphv2/flutter_split_graphv2.manifest.json`
- Environment: `results/arm64_graphv21_environment.txt`
- Transfer package: `arm64_graphv21_prepared.tar.gz`
- Package SHA-256:
  `ba93ec630615a4948ddc8af769e3fd20863a11dcfdb1a16e44170728bd9f6efe`

## Remaining Scientific Scope

This is a real Flutter/ARM64 binary distribution, but the function semantics are
synthetic algorithmic tasks. It supports a cross-ISA and long-function
replication claim; it does not by itself establish performance on organic
application business logic, malware, or framework-heavy Flutter code.

The old `RUN_PLAN_ARM64.md` predates graph-v2.1 and uses a different 9B lineage.
Do not run it as written. The confirmatory ARM study should use the same pinned
Qwen3-8B/GraphCodeBERT revisions as the corrected x86 study and begin with
text-only, no-edge, CFG-only, CFG+DFG, shuffled-edge, and no-GINE controls.
