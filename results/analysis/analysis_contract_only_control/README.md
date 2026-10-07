# Opaque typed-contract-only oracle control

Status: locally staged for review. Nothing in this directory has been deployed
or launched on the GPU pod.

This control evaluates the frozen original enriched-SFT checkpoint at
`checkpoint-optstep-000348` while exposing only the gold-derived opaque Dart
types and arity. It preserves the exact historical typed instruction and
encoder tags. The payload between `<enriched_binary>` and
`</enriched_binary>` is the task-invariant empty UTF-8 byte string (`0` bytes,
SHA-256 `e3b0c442...b855`). No F2 constants, external identities, or structure
are serialized to the model.

This is an oracle control, not a deployable system result: it assumes perfect
gold type/arity recovery while evaluating no recovery front-end. The empty
input is also out of distribution. A drop can establish dependence on the
task-specific binary-channel condition under this frozen policy; it cannot by
itself prove semantic decoding of F2.

## Run design

1. A separate, non-autostart Supervisor handoff waits for the active
   `t5gemma2-measurement-intervention-multiseed-v1` program to reach exact
   `EXITED`. `STOPPED`, `FATAL`, `BACKOFF`, or an unknown state blocks.
2. The handoff recomputes the predecessor's five-seed report over all source
   artifacts, requires a stable complete report hash, at least 5 GiB of disk,
   and an empty GPU with at least 5 GiB free.
3. It starts the distinct `t5gemma2-contract-only-control-v1` program with a
   duplicate-process guard, after publishing a deterministic exact-EXITED
   handoff attestation. The runner refuses a direct start without that record.
4. The runner repeats bundle/code/data/checkpoint/Rank-0/runtime/privacy/disk
   gates, takes the canonical shared GPU lock, and rechecks GPU availability.
5. A separate seed-42 first-five, K=10 smoke is generated and scored against an
   exact first-five evaluation JSONL.
6. Full seed 42 is generated/scored. All 50 smoke predictions and compile/pass
   decisions must exactly match the full-run prefix before seed 43 starts.
7. Full seed 43 is generated/scored. The final report validates every journal
   chain and reports per-seed exact McNemar comparisons against the same-seed
   baseline and typed+F2 arms.

The full settings are K=10, generation batch 10, temperature 0.8, top-p 0.95,
32,768 source tokens, 4,096 output tokens, SDPA, bf16, scorer timeout 30 seconds,
32 workers, and two stability runs.

## Files

- `contract_only_view.py`: exact source builder and first-five materializer.
- `contract_only_inference.py`: truthful control-specific provenance plus the
  existing native T5Gemma generation/resume path.
- `score_contract_only.py`: strict fourth-view provenance admission; all
  scoring semantics remain in the unchanged project scorer.
- `verify_smoke_replay.py`: scored smoke-to-full seed-42 replay gate.
- `contract_only_report.py`: sealed n=2 paired report and interpretation gate.
- `preregistration.json`: immutable pre-result hypotheses, thresholds,
  contrasts, limitations, settings, and checkpoint/input hashes.
- `run_contract_only_control.sh`: actual evaluation runner.
- `handoff_after_current.sh`: CPU-only post-current Supervisor transition.
- `handoff_attestation.py`: binds exact upstream EXITED, corrected predecessor
  report SHA, stable-hash/resource gates, and the reviewed bundle digest.
- `t5gemma2-contract-only-*.conf`: separate non-autostart Supervisor programs.
- `bundle.sha256`: reviewed payload hashes. Its own SHA-256 is the required
  approval token baked into both Supervisor configurations.

## Review and deployment rule

Do not deploy until the parent review accepts the exact `bundle.sha256` digest.
Both Supervisor programs default to `autostart=false`. Deployment should copy
the reviewed directory without editing it, install the two provided Supervisor
configs, run Supervisor reread/update, and manually start only the handoff
program. Starting the actual control program directly bypasses the intended
exact-EXITED transition and is not the registered procedure.

The renderer has an explicit binary-payload boundary so a future, separately
reviewed full-F2 donor-permutation control can preserve prompt grammar. No such
control is included in this preregistration or runner.
