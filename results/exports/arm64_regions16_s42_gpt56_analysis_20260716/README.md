# ARM64 Regions16 Seed-42 Analysis Bundle

This bundle is a self-contained handoff for deeper analysis of the verified Dart AOT ARM64 replication. It contains the immutable train/eval graph splits, all 343 tasks x 10 candidates, compile/pass outcomes, exact run provenance, matching architecture source snapshots, audits, logs, and derived failure tables.

Start with:

1. `docs/PROMPT_FOR_GPT_5_6_PRO.md`
2. `docs/KNOWN_FINDINGS.md`
3. `docs/ARCHITECTURE.md`
4. `derived/per_task_analysis.csv`
5. `results/*_pass_predictions.json` and `results/*_pass_stats.csv`

## Directory map

- `datasets/`: Full immutable ARM64 train and eval graph-v2.1 splits plus split metadata. The duplicate full pool is intentionally omitted because train + eval already contain all 1,714 rows.
- `results/`: Verified summary, raw candidate records, compile/pass statistics, provenance, status, and logs.
- `derived/`: Analysis-friendly per-task features, stratified metrics, long-reference failures, semantic successes, and metric-reproduction checks.
- `source/`: Exact runner, training, encoder, tensor-builder, dataset, and CFG/DFG source used by or directly relevant to the run.
- `audits/`: Dataset integrity, overlap, complexity, environment, and original hash reports.
- `tools/`: Reproducible script used to generate `derived/`.

## Integrity

`SHA256SUMS.txt` hashes every file in the bundle except itself. The ZIP has a separate adjacent `.sha256` file.

## Important scope notes

- The 924,910,291-byte model checkpoint is not duplicated. Its SHA-256, Hugging Face location, and full provenance are in `results/run_provenance.json`.
- The exact training implementation is the archived snapshot in `source/scripts/training/`. Its hash matches the source hash recorded by the run. A newer workspace copy was deliberately not substituted.
- `reference_length` comes from the evaluation statistics and is a useful complexity proxy. It is not proof that length itself caused failure.
- Hidden scoring tests were not visible to the policy during generation.
