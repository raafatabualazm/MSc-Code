# Arm C2 production-eligibility amendment

Status: fixed after Arm B's seed-42 result and before any Arm C optimizer step.
The original Arm C process terminated during CPU preflight; it never loaded the
model, touched the GPU, or changed weights.

## Why Arm C cannot run unchanged

The original seeded selection contains `sigless_bfde11b99b84`. Its sealed gold
decoder target imports `dart:io` and writes a file. The exact production
completion-attested verifier rejects every `dart:io`/`dart:ffi`/`dart:mirrors`
candidate before execution because process and introspection capabilities can
spoof or bypass the completion marker. Eight repeated checks produced the same
pre-execution rejection. A trusted-gold-only exception was rejected because it
would train a target that a student can never score with at evaluation time.

The investigation also found that the production code extractor did not
recognize class-first Dart. It discarded an enclosing class and began at its
first method, falsely classifying otherwise valid targets as compile failures.
That evaluator bug is patched and regression-tested before this amendment is
used. Historical checkpoints must be rescored from their stored generations
under the corrected extractor before any promotion comparison is interpreted.

## Amended replay-selection contract

Arm C2 retains the original Arm C architecture, parent checkpoint, direct
union, optimizer configuration, update count, privacy boundaries, and v1
ranking key. Only replay eligibility changes:

1. Reconstruct the same 2,315 replay rows that are disjoint from Arm B by task
   identity and byte-identical typed-source hash.
2. Sort with the original v1 key: v1 dataset schema, seed 42, `gold_replay`
   kind, task ID, and typed-source SHA-256.
3. Retain the first-ranked representative of each typed-source hash, producing
   2,314 source-unique candidates.
4. Run the exact corrected production verifier, with the complete private
   TRAIN acceptance suite, 30-second timeout, and two stability executions, on
   every one of the 2,314 candidates. Tests and diagnostics remain private and
   are never model-visible or persisted. This gate uses neither Arm B
   predictions nor Arm B scores.
5. Preserve ranked order among the 2,312 verifier-eligible candidates and take
   the first 458.

The two corpus-wide exclusions are verifier-forbidden capability programs:
`sigless_bfde11b99b84` and `sigless_67bb88ce699e`. The former was in the
original 458; its deterministic replacement is
`fresh-eval-ef0cf897e22d`. This is an amended post-result dataset and must be
reported as Arm C2, never as unchanged or preregistered Arm C.

## Sealed selected-replay identities

- Task IDs SHA-256:
  `6da49d120c902fde194c09fa14f7718bb379d8e676da8071211e1ac95da8e9df`
- Typed-source SHA-256 list SHA-256:
  `1c818f33808c4142eb7b148733ce6879a779f795f1855e66533832baa99b31d6`
- Decoder-target SHA-256 list SHA-256:
  `c7031487c72a2edba0baca1d8fe9eadc76232136c439a0f1fc95b25e2044e8f6`

## Training and decision contract

Arm C2 remains a fresh branch from typed SFT optstep348 with Arm B's exact 458
direct targets plus 458 production-eligible gold rows, one epoch, batch size 1,
gradient accumulation 16, learning rate `5e-6`, no warmup, no weight decay,
seed 42, and 58 optimizer updates. Seed 42 may veto or trigger replication but
cannot promote. Promotion still requires at least three matched seeds, and
VeRPO remains held.
