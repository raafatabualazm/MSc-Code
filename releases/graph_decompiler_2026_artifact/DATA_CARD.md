# Data Card

## Primary Functional Corpus

Path: `data/benchmark/grpo_data_cfg.jsonl`

- Rows: 154
- Unit: HumanEval-Dart function
- Key fields: `task_id`, `filename`, `function`, `dart_function_signature`, `dart_source`, `assembly`, `tests`, `cfg`, `edges`, `integrity`
- Uses: pass@k, aligned JIT/pass-harness compile@k, reranking, task-level paired analysis

Each candidate is normalized, combined with the stored tests, and executed through `dart run`. The aligned compile classifier counts Dart front-end syntax/type diagnostics as compile failures and runtime assertions/timeouts as compiled-but-not-passing.

## Historical CodeBLEU Corpus

Path: `data/benchmark/compile-test2_cfg.jsonl`

- Rows: 126
- Unit: standalone Dart program, usually a top-level `main`
- Key fields: `filename`, `function`, `source`, `assembly`, `language`, `cfg`, `edges`, `integrity`
- Uses: CodeBLEU and historical standalone AOT compile diagnostics only

This file has no HumanEval `tests` field and shares no filenames with the 154-task corpus. It is not a filtered subset of HumanEval-Dart.

## Harvested RS-SFT Data

- `data/harvest/rs_sft_x86_8b_allarms_with_h100_all.jsonl`: 142 deduplicated passing rows after per-task capping.
- `data/harvest/rs_sft_x86_8b_allarms_with_h100_all_plus_refs.jsonl`: 253 rows after adding references.
- `data/harvest/rs_sft_x86_8b_allarms_with_h100_report.json`: harvest provenance and counts.

These rows were harvested from the same benchmark family. They support optimization and candidate-coverage analysis, not clean held-out generalization.

## Future-Work Assets

- `data/future/synthetic_pool_clean.jsonl.gz`: 1,726 synthetic source/assembly rows with executable tests.
- `data/future/flutter_function_assembly_pool_cfg.jsonl.gz`: 1,714 ARM64 Flutter release-build function slices with tests and extracted graphs.

The ARM64 binary distribution is real, but its function semantics are controlled/synthetic. It is a stage-one cross-ISA artifact, not evidence of recovery on production application logic.

`data/fixtures/flutter_train_cfg_first.jsonl` is a single-row ARM64 regression fixture used by `test_graph_preprocessing_fixes.py`; it is not an evaluation set.

## Leakage and Validity

The benchmark is public and HumanEval-derived. Function signatures may identify familiar tasks to large pretrained models. The study did not run a signature-only control, so hosted-model results measure practitioner-visible performance on this benchmark rather than assembly-only semantic inference.
