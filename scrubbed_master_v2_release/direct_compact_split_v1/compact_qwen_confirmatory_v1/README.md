# Confirmatory direct-Qwen compact bundle

This bundle fits the instruction codebook on the sealed 1,975-row training
split and only measures the 219-row development split. Development rows do not
affect the codebook.

`compact_model_inputs.jsonl` contains both roles in fit-then-measure order and
has exactly four model-facing fields. `alignment_private.jsonl` provides the
withheld task IDs, role, original line, and audit digests. The role-aware sealed
join produced `train_supervised.jsonl` and `dev_supervised.jsonl`; their seals
bind every source artifact, mapping, row count, and output hash.

`compact_input_ids` contains only compact source IDs. The direct-Qwen stack adds
the natural-language prompt exactly once and takes its fixed language (`Dart`)
and neutral target name (`candidate`) from `compact_contract.json`.

## Verified compression and reversibility

- 2,194 / 2,194 rows round-trip exactly in the
  `scrubbed_canonical_graph` domain.
- Instructions, entry blocks, and CFG edges reconstruct exactly.
- 201,033 DFG edges regenerate and match edge-for-edge with the SHA-pinned DFG
  extractor.
- There is no UNK substitution or truncation; raw fallback is reversible.
- Privacy canonicalization defining the scrubbed domain is the only intentional
  irreversibility.
- Actual compact-source tokens: min 29, p50 388, p95 1,432, p99 2,253,
  max 7,254. No row exceeds the 9,000-token contract.

This bundle verifies data/codec correctness. It does not by itself prove model
conditioning; correct/permuted/null NLL and free-running functional gates remain
required after training.

## Train-only codebook generalization

The fitted codebook contains 16,365 instruction atoms. Every instruction atom
is observed in training, but 6,275 occur only once; each atom is initialized
from the mean of Qwen's native tokenization of its exact instruction text rather
than from a random vector. Of 20,475 total source rows, 3,394 are not exercised
by training; these are almost entirely reserved high block IDs plus the raw
fallback delimiters. They retain their semantic initialization.

On the 219-row sealed development split, 804 of 59,170 instructions (1.36%) are
outside the train-only one-token codebook. They occur on 134 rows and are encoded
losslessly with native Qwen tokens; development still has no UNK or truncation,
and its maximum source length is 6,175. See
`dev_codebook_generalization.json`.

On the 154-task harmonized HumanEval measurement, 1,418 of 18,359 instructions
(7.72%) use the same reversible fallback. Every task has at least one fallback,
but all remain below 9,000 source tokens (maximum 7,766). See
`data/testing/direct_compact_humaneval_v2_nameonly_harmonized/`
`humaneval_codebook_generalization.json`.

The compact stream is the only task-specific model input, but that does not make
causal use automatic under teacher forcing. Post-training source permutation
and null-source gates remain load-bearing.
