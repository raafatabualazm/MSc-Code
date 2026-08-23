# Suggested Prompt for GPT-5.6 Pro

You are reviewing a verified ARM64 Dart AOT decompiler experiment. Analyze why the model compiles often but almost never passes semantic tests, with special attention to the observed zero pass@10 across all 166 tasks whose reference-length proxy is at least 1,000.

Use the files in this ZIP as primary evidence. Begin with `README.md`, `docs/ARCHITECTURE.md`, `docs/KNOWN_FINDINGS.md`, `derived/analysis_summary.json`, `derived/per_task_analysis.csv`, the full eval split, and the raw prediction records. The full immutable train split is included for distribution comparisons.

Please deliver:

1. A concise diagnosis separating syntax/compilation failure from semantic failure.
2. Ranked hypotheses with direct evidence, counterevidence, confidence, and the files/fields used.
3. Offline analyses that can discriminate among region-compression pressure, graph-prefix density, context/output truncation, graph construction limits, training-distribution mismatch, and intrinsic task complexity.
4. A taxonomy of long-task candidate failures based on representative raw predictions.
5. A comparison of successful and unsuccessful tasks controlling for reference length, CFG blocks, parsed instructions, category, difficulty, test count, and assembly size.
6. Any architecture or provenance inconsistency that could invalidate interpretation, including the gradient-checkpointing serialization discrepancy.
7. The smallest next experiment that would be decision-relevant, but only after exhausting the included offline evidence. Do not recommend broad grids or many new GPU arms.

Important constraints:

- Treat `reference_length` as a proxy, not an established cause.
- Use task-level uncertainty; ten candidates from one task are not ten independent tasks.
- Hidden scoring tests were not visible to the policy.
- Do not confuse the separate low-stratum fresh-holdout source generation with training data for this model.
- Cite exact task IDs and candidate indices when discussing examples.
- Clearly separate verified facts, calculations from the bundle, and speculation.
