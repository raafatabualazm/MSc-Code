# Regions16 Signature-Scrubbed Review Index

## Scope

Frozen seed-42 trainable-GCB/no-GINE/Regions16 checkpoint evaluated once on the
154-row neutral-name, signature-scrubbed x86 set with 10 candidates per row.
Inference used only the public dataset; hidden source/tests were joined after
generation.

## Immutable Inputs

- Public policy input: `data/testing/grpo_data_graphv2_signature_scrubbed_public.jsonl`
  - SHA-256: `45756d00c0fc7f749081ca1236e715b1b9d33412a454cea27342ddf0ac2d657e`
- Private evaluator sidecar: `data/testing/grpo_data_graphv2_signature_scrubbed_private.jsonl`
  - SHA-256: `91dc2eee2f06602e6cd95873c81802e0997fd3ff758b395aeab9c3da114df252`
- Dataset summary: `data/testing/grpo_data_graphv2_signature_scrubbed_private.jsonl.summary.json`
- Dataset rejects: `data/testing/grpo_data_graphv2_signature_scrubbed_rejects.json`
- Dataset build log: `logs/grpo_data_graphv2_signature_scrubbed_build.log`
- Frozen checkpoint SHA-256:
  `e8e872608f22ae8e1c5607d6179feeb5f133401fb2a7d1fc40fe8894d8c347fc`

## Canonical Evaluation Bundle

Stem:
`qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_regions16_signature_scrubbed`

Under `results-20260713/`:

- `<stem>_raw_predictions.json`
- `<stem>_raw_predictions.json.provenance.json`
- `<stem>_predictions.json`
- `<stem>_predictions.json.provenance.json`
- `<stem>_codebleu.txt`
- `<stem>_compiled_codebleu.txt`
- `<stem>_compile_at_k.txt`
- `<stem>_pass_at_k.txt`
- `<stem>_sha256.txt`
- `<stem>.status`

Additional files:

- Stats: `results-20260713/sweeps_antigravity/<stem>_stats.csv`
- Active-run log: `results-20260713/logs/signature_scrubbed_eval/<stem>.log`

The SHA-256 manifest verifies the datasets, checkpoint, raw/scored predictions,
both provenance files, metric outputs, and stats CSV. The raw and scored files
contain 154 rows with exactly 10 candidates each. Their candidate-stream digest
is unchanged:
`b4c0be6ec347329bf5cbb71910d07c66dd0fe8b1392a3bb3610f3dbf61bdb32e`.

## Exact-Signature Comparator

Stem:
`qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_regions16`

- Sweep summary: `results-20260713/sweeps_antigravity/<stem>.json`
- Compile predictions/provenance: `results-20260713/<stem>_compile_predictions.json*`
- Pass predictions/provenance: `results-20260713/<stem>_pass_predictions.json*`
- Compile stats: `results-20260713/sweeps_antigravity/<stem>_compile_stats.csv`
- Pass stats: `results-20260713/sweeps_antigravity/<stem>_pass_stats.csv`

Paired results over all 154 tasks:

| Metric | Exact signature | Scrubbed | Delta |
|---|---:|---:|---:|
| pass@1 | 17.66% | 0.00% | -17.66 pp |
| pass@5 | 24.11% | 0.00% | -24.11 pp |
| pass@10 | 27.27% | 0.00% | -27.27 pp |
| aligned JIT compile@1 | 77.08% | 0.00% | -77.08 pp |
| aligned JIT compile@5 | 93.80% | 0.00% | -93.80 pp |

Static generation audit of the 1,540 scrubbed candidates:

- 1,415 lacked a detected top-level lowercase `candidate(...)` function.
- 120 declared that target at top level; 22 matched the hidden argument count.
- 1,050 imported external packages; 578 referenced Flutter.
- 965 declared `main()`, 939 declared a `Candidate` class, and 1,531 used fences.

## Protocol Code

- Dataset builder: `scripts/data/build_signature_scrubbed_eval.py`
- Hidden-label join: `scripts/evaluation/rehydrate_signature_scrubbed_predictions.py`
- Builder tests: `scripts/data/test_signature_scrubbed_eval.py`
- Join tests: `scripts/evaluation/test_rehydrate_signature_scrubbed_predictions.py`
- Prompt/model path: `scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py`
- Aligned compile harness: `scripts/evaluation/graph_compile_at_k_antigravity.py`
- Functional harness: `scripts/evaluation/graph_pass_at_k_antigravity.py`

## Broader Experiment Context

- Full clean-study report: `results-20260713/GRAPHV2_CLEAN_STUDY_ANALYSIS.md`
- Full clean-study data: `results-20260713/graphv2_clean_study_analysis.json`
- Interaction-study data: `results-20260713/graphv2_interaction_study_analysis.json`

Interpretation caution: this stress test removes semantic name, return type,
parameter types, and arity simultaneously, while replacing the name with the
common noun `candidate`. This decoder arm also receives no assembly text; its
only task-specific binary input is the 64-vector graph prefix. The SFT corpus
does use the function-name prompt branch (0/770 rows carry signatures), but is
dominated by `name == main` (741/770) and contains no `candidate` targets. The
result therefore proves dependence on the visible signature contract for this
arm and protocol; it does not show that the graph channel contains zero usable
evidence or isolate semantic-name reliance from signature/arity reliance. Also note
that aligned JIT compile@k compiles candidate + hidden tests, so it includes
top-level symbol/caller compatibility rather than measuring standalone Dart
syntax alone; it does not prove exact ABI recovery. The corrective v3 protocol
therefore seals a separate candidate-only legacy wrapped standalone-AOT
acceptance@1/5/10 diagnostic for every arm.
