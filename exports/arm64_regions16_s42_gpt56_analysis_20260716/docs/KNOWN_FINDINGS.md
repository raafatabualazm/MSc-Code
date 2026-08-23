# Known Findings and Caveats

## Overall verified metrics

| Metric | Value |
|---|---:|
| CodeBLEU | 0.6723237273 |
| Compiled-only CodeBLEU | 0.6634723335 |
| compile@1 | 0.5294460641 |
| compile@5 | 0.9307464482 |
| pass@1 | 0.0212827988 |
| pass@5 | 0.0452820584 |
| pass@10 | 0.0553935860 |

The model often produced syntactically compilable Dart but rarely preserved behavior. Nineteen of 343 tasks had at least one passing candidate.

## Reference-length stratification

The `@k` values below use the same unbiased estimator as the official result.

| Reference-length proxy | Rows | compile@1 | compile@5 | pass@1 | pass@5 | pass@10 |
|---|---:|---:|---:|---:|---:|---:|
| <700 | 94 | 0.6489 | 0.9771 | 0.0362 | 0.0934 | 0.1170 |
| 700-999 | 83 | 0.5771 | 0.9625 | 0.0470 | 0.0814 | 0.0964 |
| 1,000-1,299 | 78 | 0.4897 | 0.9186 | 0 | 0 | 0 |
| >=1,300 | 88 | 0.3920 | 0.8620 | 0 | 0 | 0 |

All 166 tasks with `reference_length >= 1000` failed all ten semantic test attempts. This is an observation, not a causal identification result. Length covaries with CFG size, instruction count, category, test complexity, output length, and other factors.

## High-value questions

1. Is the collapse better explained by region compression pressure, decoder context allocation, output truncation, training-distribution scarcity, or test/algorithm complexity?
2. Does compile probability decay smoothly while semantic success exhibits a threshold, suggesting separate syntax and information bottlenecks?
3. Are long tasks failing because candidates omit logic, truncate, mishandle types/state, or reproduce the wrong algorithm?
4. Does `region_max_blocks=16` discard or overcompress control/dataflow precisely where semantic success disappears?
5. Are graph-prefix tokens too sparse relative to longer functions, even though the prefix is dynamic?
6. Does the graph-only prompt channel remove textual assembly cues needed for long functions?

## Caveats

- This is one seed of one selected architecture on one ARM64 split.
- `reference_length` is not target-function length alone.
- Passing tasks are only 19 observations, so fine-grained subgroup estimates are unstable.
- Compile and pass records are aligned to the same 343 tasks and 10 candidates; do not treat the 3,430 candidates as independent task-level observations.
- The evaluation failure does not imply that adding low-stratum holdout tasks would train or repair this model. That separate data job is for evaluation supply.
