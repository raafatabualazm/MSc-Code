# Changelog v2.3

v2.3 corrects two compute-invalidating defects in v2.2 and makes the historical
long-function truncation result observable independently of the graph gate.

## 2026-07-27 asynchronous direct-compact VeRPO rescue

- Added a grounded four-arm rescue experiment for visible all-zero rollout
  groups with fixed ITT repair slots and deterministic max-min code diversity.
- Added exactly-one-attempt-per-group judge diagnosis with SDK and completion
  retries disabled, derived diagnosis-only feedback, hash-chained paid-call
  receipts (including rejected receipts without accepted response IDs), exact
  terminal-call resume, and fail-closed handling of indeterminate started calls.
- Added visible-only repair selection, sealed development-holdback scoring,
  full-pass-only RS-SFT export, and separately sealed off-policy partial
  preferences. Byte-identical copies of original gold are excluded from
  transfer coverage and mixture counts.
- Added a phased production launcher, pinned launch config, explicit GPU
  release acknowledgement before paid diagnosis, resumable GPU generation,
  bundle validation, scoring, and matched transfer construction. Preflight
  rejects final-measure data before work begins; generation binds precision,
  attention implementation, batching, and deterministic RNG provenance across
  arms; scoring seals the predeclared task-paired McNemar contrast.

## 2026-07-23 RS-SFT and VeRPO integrity correction

- VeRPO applies the compile-gated LLM score after the group reward instead of
  overwriting it; a verifier-confirmed full pass always remains dominant.
- The paper bandwidth is restored (`sigma = std(rho) / 2`) and the non-paper
  reward deadband is opt-in (`0` by default).
- Judge authentication, transport, timeout, and parsing fail closed. Cumulative
  API/cache/error telemetry is written during training and into provenance.
- DeepSeek thinking remains enabled for score/critique calls with a 12,288-token
  initial budget; empty or length-truncated completions retry up to a bounded
  32,768-token budget before failing closed.
- Long Qwen rollouts are generation-chunked. Weight snapshots are written only
  after completed optimizer updates (not during partial accumulation windows).
- Rollout sampling is untruncated (`top_p=1`, `top_k=0`) so policy-gradient
  log-probabilities exactly match the sampled temperature-scaled distribution.
- Text-SFT and VeRPO shortcut checkpoints now participate in the same mandatory
  structural-provenance checks as the main curriculum checkpoints.
- Missing legacy defaults are recovered only for two explicitly allowlisted
  trainer source hashes and only while that exact trainer remains installed;
  unknown provenance still fails closed.
- The default VeRPO anchor is the independently recertified 06a artifact rather
  than the legacy pre-recertification harvest.
- V2 repair rows join across `task_id`/`id` aliases, use only
  `feedback_tests`, and reject/strip hidden harness fields.
- Resume signatures now include non-secret `VERPO_*` configuration and verify
  the output artifact digest before reuse.
- Dart pass/reward execution now requires a per-run 256-bit completion
  attestation emitted only after the trusted sync/async test main returns.
  Candidate-controlled `exit(0)`, isolate termination, duplicate markers, and
  marker-spoof capabilities fail closed. VeRPO's full-suite and per-assertion
  subprocesses use the same contract.
- The installer now ships the judge, repair builder, and hardened Dart
  evaluator instead of treating the evaluator as an unversioned prerequisite.

## Dynamic padding

- SFT tokenization now stores variable-length labels and decoder prompts with
  `padding=False`.
- `models/graph_data_collator.py` dynamically right-pads prompts to the current
  batch maximum and pads labels with `-100`.
- GRPO uses the same collator for policy and verified-anchor batches.
- Inference no longer pads a single row to the global prompt ceiling and aborts
  rather than silently truncating an oversized prompt.

## Budget consistency

- The default long generation floor is now 3,072 tokens, matching the default
  3,072-token admitted target ceiling.
- The runner validates that the target ceiling fits the effective gate,
  rollout, and GRPO generation budgets before Phase 0.
- A complete target can no longer pass SFT admission while being impossible to
  emit in every evaluation path.

## Token-distribution preflight

- Added `report_token_lengths_antigravity.py`.
- Reports target/prompt percentiles, historical 768/1024/2048 truncation counts,
  per-stratum and per-task distributions, unique code-target distributions,
  largest examples, and all configured-budget overflows.
- The curriculum runs this preflight before any GPU stage and fails closed on
  overflow.

## Direct-control ordering

- The matched direct-code control and its code-recovery stage now run before the
  Stage-1 causal graph gate.
- An absolute ≥200 direct-control pass@k report is produced even if the latent
  prefix gate fails.
- Hierarchical training still requires the causal gate to pass.

## Validation

- 66/66 hybrid, VeRPO, and evaluator-integrity tests pass; the complete package
  suite is 107/107.
- Added regressions for dynamic batch padding, absence of global max-length
  padding, generation-budget consistency, token-preflight wiring, and direct
  control ordering.
