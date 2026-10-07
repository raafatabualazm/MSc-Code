# Typed seed replication v1

This stack reruns the frozen typed-SFT optstep348 checkpoint, the incumbent
typed RS-SFT update58 checkpoint, and the eventual pass-3 checkpoint at seeds
43–46. Seed 42 remains diagnostic context. The confirmatory arms use 175 tasks,
K=10, temperature 0.8, top-p 0.95, the typed opaque contract, 32,768 source
tokens, 4,096 output tokens, and the existing hash-chained inference/scorer.

The three artifact roots are distinct and all arm launchers share a GPU lock.
Incomplete generation and scoring journals resume at the exact next sealed
task/batch. Starting two arms does not run them concurrently. An unrelated GPU
process is also detected before model loading.

Programs:

- `t5gemma2-typed-seed-repl-incumbent-v1`
- `t5gemma2-typed-seed-repl-sft-v1`
- `t5gemma2-typed-seed-repl-pass3-v1`
- `t5gemma2-typed-seed-repl-report-v1`

The pass-3 program is intentionally fail-closed until its config placeholders
are replaced. Its SHA-pinned manifest has schema
`t5gemma2-typed-seed-replication-pass3-checkpoint-manifest-v1` and contains:

- `arm: "pass3"`, the absolute `checkpoint`, and exactly four
  `checkpoint_files` records (`run_contract.json`, adapter weights/config, and
  tokenizer), each with absolute `path` and `sha256`;
- `training_result` and `training_audit` path/SHA records;
- `run_contract_schema` and `run_contract_canonical_sha256`;
- lineage to `incumbent_update58` and parent adapter SHA
  `62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec`;
- false model-visibility flags for heldout/tests/private feedback/gold body/
  semantic parameter names, true prior-success exclusion, the known contaminant
  exclusion, and `no_automatic_promotion: true`.

The actual pass-3 run contract is independently checked for zero heldout
overlap, false test visibility, the typed-only visible fields, pinned base model,
and native encoder-decoder architecture.

Before the report program is installed, late-bind the completed pass-3 seed-42
prediction and full175 score paths. The report requires all three arms at seeds
42–46; pass2 is optional and admitted only as a complete five-seed arm. It
reports pass@1, pass@10, compile@10, extracted-code distinct/10, and paired exact
McNemar results for incumbent versus pass3 at every seed. It performs no
promotion. Seed 42 is labelled diagnostic-only; seeds 43–46 are confirmatory.

Observed current-stack timing is roughly 75–85 minutes per seed. Budget about
5–6 hours for one four-seed arm and 15–17 hours for all three, serialized on one
GPU. The JSON report itself takes seconds after all artifacts exist.
