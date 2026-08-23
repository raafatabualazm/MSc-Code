# Phase-0 s44 compact-Qwen v2 release

This release holds representation constant against the exact Phase-0 corpus
used by the v2.3 pipeline. The canonical 3,306 input rows are reconciled once,
in original input order, against the supplied 3,305-row Phase-0 assignment.
Task IDs are unchanged. The one unlisted row is excluded rather than inferred
into either split.

## Population and sealing

- Included train: 2,951 rows.
- Included dev: 326 rows, including 138 retained `in_long_dev_ge200` rows.
- Quarantined: 14 corrupt `local_N` mnemonic rows. This includes
  `sigless_179318026249` and `sigless_b51cb0fef9eb`; two quarantines belonged
  to the 140-row long-dev slice.
- Excluded: 14 forbidden-family near-clones plus the one row absent from the
  Phase-0 manifest (`sigless_4901067c13b9`).

The 14 near-clone exclusions are mandatory. Although an earlier audit reported
zero ID/content overlap, the required compact-v1 fingerprint policy found these
rows at the sealed alpha-token 5-gram Jaccard threshold. They are explicitly
recorded in `prepared/forbidden_overlap_audit.jsonl`; no ID-only assumption can
override that content gate.

`family` is experiment metadata, not a task-ID rewrite. Its three requested
values are `master`, `topup_s45`, and `topup_s46`. `topup_s45` is deliberately a
coarse pre-s46 umbrella covering the exact source pools `base_llm`,
`topup_s44`, and `topup_s45`; it must not be read as literal generator
provenance. Exact `source_pool` and raw Phase-0 family remain in private
alignment metadata. The `topup_s46` family maps only to its exact source pool.

The codebook was fit on the 2,951 included train rows only. Dev, all scrubbed
HumanEval variants, `fresh_graphv2_holdout_s44`, and the rebuilt 490-row
functional evaluation were excluded from fit and swept by exact,
alpha-structural, and near-clone fingerprints. `runtime-symbol-policy-v1` is
applied uniformly with target function `candidate`, including relabeled top-up
targets.

## Versioned, lossless compact stream

The compact stream starts with `<G2C2><AX64>` and an explicit extractor-route
atom: `<DX0>` for the legacy graph/DFG pair or `<DX1>` for the current combined
extractor. Current graphs encode every explicit `call` CFG edge with `<CC>`;
legacy graphs reject call edges. Both CFG and DFG extractor hashes are pinned in
`compact_qwen_phase0_s44_v2/compact_contract.json`.

Within the declared `scrubbed_canonical_graph_v2` domain, compression is
lossless. All 3,277 retained rows reconstruct instructions, entry, blocks, CFG,
and extractor route exactly; DFG is deterministically regenerated with the
route-pinned extractor and all 334,849 edges match. Reversible native-token raw
fallback handles unseen dev/evaluation instructions. The privacy scrub is the
only intentional irreversibility. There are zero unknown substitutions and
zero truncated rows.

Compact-source tokens are min 30, p50 355, p95 1,261, p99 2,080, max 7,255;
all rows pass the 9,000-token gate. The dev, scrubbed-HumanEval, and family/source
pool fallback reports are sealed under `audits/`.

## Artifacts

- `prepared/`: train/dev codec-private rows, private labels, the 3,306-row
  reconciliation, quarantine and forbidden-overlap audit, plus its seal.
- `compact_qwen_phase0_s44_v2/`: strict four-field model inputs, private
  alignment, codebook, compact contract, preflight, and nested checksums.
- `supervised/`: bijective train/dev joins and per-join hash/mapping seals.
- `audits/`: aggregate generalization reports and their audit seal.
- `release_manifest.json` and root `SHA256SUMS.txt`: deterministic umbrella
  seal. The checksum covers every shipped file except itself.

## Rebuild and verify

Run from the workspace root. Replace the tokenizer/config placeholders with the
pinned Qwen snapshot paths shown in the compact contract.

```text
python scrubbed_master_v2_release/prepare_phase0_compact_qwen_v2.py

python scripts/data/build_compact_qwen_v2.py --fit scrubbed_master_v2_release/direct_compact_phase0_s44_v2/prepared/train_codec_private.jsonl --measure scrubbed_master_v2_release/direct_compact_phase0_s44_v2/prepared/dev_codec_private.jsonl --output-dir scrubbed_master_v2_release/direct_compact_phase0_s44_v2/compact_qwen_phase0_s44_v2 --tokenizer-json <QWEN_SNAPSHOT>/tokenizer.json --model-config <QWEN_SNAPSHOT>/config.json --codebook-size 20000 --tokenizer-fingerprint-sha256 c1e79a88eb4b3e2a96972edf99f6b28e6f39ed832ae95e4727ecdb425a25de4a --decoder-revision b968826d9c46dd6066d109eabc6255188de91218

python scrubbed_master_v2_release/audit_phase0_compact_qwen_v2_generalization.py

python scrubbed_master_v2_release/finalize_phase0_compact_qwen_v2.py
```

The two supervised joins are created with
`hybrid_training_patch_v2_3/scripts/training/join_compact_public_private.py`,
selecting `--role fit` for `prepared/train_private_labels.jsonl` and
`--role measure` for `prepared/dev_private_labels.jsonl`. The v2 contract makes
the private-label bijection mandatory.
