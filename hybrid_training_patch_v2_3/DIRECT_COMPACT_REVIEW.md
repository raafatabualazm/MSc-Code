# Encoder-free direct-Qwen review

Date: 2026-07-19

## Verdict

The path is now genuinely decoder-only: compact source IDs enter Qwen's input
embedding stream directly. There is no GraphCodeBERT model, GNN, CFG tensorizer,
or soft-prefix injection in training or inference. The corrected confirmatory
bundle passes its codec, leakage, no-truncation, graph round-trip, and sealed
join gates.

No claim of learned decompilation performance is made yet. A real Qwen3-8B
optimizer smoke and the correct/permuted/null behavioral gates still need to run
on the GPU pod.

## Blocking integration defects corrected

1. The codec previously prepended a natural-language prompt inside
   `compact_input_ids`, and the trainer prepended another prompt. Compact IDs now
   contain only the source representation; train and inference add one prompt.
2. Strict model rows intentionally omit task IDs, but the old join required a
   task ID in those rows. The join now requires the private alignment sidecar,
   verifies `model_row`, and emits only an allowlisted supervised schema.
3. One combined codec output could not be separated into train/dev. The join
   now selects `fit` or `measure` by the sealed alignment role.
4. Strict inference had neither identity rehydration nor role selection and
   would emit numeric indices. It now requires the alignment sidecar for output
   IDs only; sidecar data never enters the prompt.
5. Training requested `candidate`, while strict inference rows had no function
   field and fell back to `fn0`. The signed contract now fixes both target name
   (`candidate`) and exact language spelling (`Dart`) in every path.
6. The builder emitted no usable `DirectCompactContract`. It now emits a full
   contract containing source-token expansions and binds codec, codebook, raw
   tokenizer, logical tokenizer fingerprint, model config, model ID, immutable
   revision, DFG extractor, limits, and target contract.
7. Source IDs formerly risked colliding with Qwen tokenizer-only boundaries.
   The boundary is the Qwen model/config vocabulary (151,936); compact IDs start
   at 151,936. The untied LM head remains at 151,936 rows.
8. Overlay restore needlessly recomputed roughly 20,000 codebook means on the
   GPU before overwriting them. Restore now installs uninitialized rows and
   immediately loads the checkpoint.
9. Bounded smoke controls now include `max_steps`, `logging_steps`, and
   `eval_strategy=no`; FlashAttention 2 and BF16 loading are exposed explicitly.
10. The GPU gate now verifies both checksum manifests and the sealed train/dev
    joins before loading Qwen. Training requires the two join seals, and run,
    NLL, and generation provenance hash-bind the exact adapter and source
    overlay.
11. The diagnostic permutation now uses a minimum-total-absolute-length
    derangement. The null arm preserves every source position and attention-mask
    bit while zeroing only source embeddings, avoiding the old target-position
    confound.

## Confirmatory artifact evidence

Canonical directory:
`scrubbed_master_v2_release/direct_compact_split_v1/compact_qwen_confirmatory_v1`

- Fit rows: 1,975; measured dev rows: 219; quarantine/failures: 0/0.
- Exact canonical graph round trips: 2,194 / 2,194.
- DFG edges regenerated and matched: 201,033.
- Compact source tokens: min 29, p50 388, p95 1,432, p99 2,253, max 7,254.
- Rows over 9,000, truncated rows, and unknown tokens: 0 / 0 / 0.
- Codebook expansion leak scan (`candidate`, file URI, absolute symbol target,
  private-field terms): all zero.
- Contract SHA-256:
  `df4ba9a5bb7ce2a7f03188ec5630ba255a893bb98e2baa379498cd5ec351769f`.
- Codebook SHA-256:
  `d44f9be95debe6e7d8766bf434cf9aeabd89a3d6ca5b09a06e3c50272543e76c`.
- Sealed train output: 1,975 rows,
  `e5e00169ec0f4c9955e060fc5e3ceec540f8f36ea8abd5f89e3557e187fbee08`.
- Sealed dev output: 219 rows,
  `72fd71ef405b0bce1a343b0a790531f4d1a8e5ecb053fdc5ca88c37de7a94a3e`.

The representation is not semantic summarization. It is lossless over the
post-privacy `scrubbed_canonical_graph`: instruction text, entries, and CFG are
encoded reversibly; DFG is omitted only because the SHA-pinned extractor
regenerates it exactly and every row is checked edge-for-edge.

## Train-only coverage follow-up

The confirmatory codebook has 16,365 instruction atoms, all observed in the
1,975 training rows and initialized from their exact native-Qwen expansion;
6,275 are singletons. Of 20,475 total source embedding rows, 3,394 are not
exercised during training, mostly reserved high block IDs.

Development fallback is 804 / 59,170 instructions (1.36%, 134 / 219 rows).
Harmonized scrubbed HumanEval fallback is 1,418 / 18,359 instructions (7.72%,
154 / 154 rows). In both cases fallback preserves the exact canonical
instruction with native Qwen tokens; it is not UNK or truncation. Direct tokens
also do not guarantee conditioning under teacher forcing, so the permutation
and null gates below remain mandatory.

## Runtime verification

- Direct-path CPU tests: 24 / 24 passed; full patch suite: 62 / 62 passed.
- Codec unit tests: 4 / 4 passed.
- Python compilation: passed for every direct train/eval module.
- On the Linux GPU pod's actual PyTorch/Transformers/PEFT environment, a tiny
  LoRA causal model successfully trained the external source overlay, saved the
  adapter and overlay separately, restored both, and generated with KV caching
  while retaining the original LM-head size.

## Remaining gates

1. Complete the queued one-step Qwen3-8B BF16/FlashAttention smoke with
   evaluation disabled, then verify adapter/overlay reload and cross-provenance
   hashes. It is deliberately behind the forced graph gate and CLEAN@12K.
2. Run correct versus matched-permuted versus null source NLL.
3. Run the authoritative free-generation compile/pass permutation gate.
4. Build the full separately requested ARM64/x86-64 paired corpus; the
   resumable builder and real import-sensitive two-pair resume smoke pass and a
   clean full v2 build is running, but this compact bundle is still x86-64 only.
