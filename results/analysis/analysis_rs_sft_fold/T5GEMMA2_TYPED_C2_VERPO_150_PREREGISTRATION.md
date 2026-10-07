# Typed C2 VeRPO-150 preregistration

Status: frozen before the first C2 VeRPO rollout or optimizer update.

## Purpose and disposition

This is one bounded, discardable test of whether on-policy visible execution
reward can improve the typed T5Gemma 2 Arm-C2 policy.  It is not an automatic
promotion arm.  Arm C2 remains the rollback checkpoint under every outcome,
and training stops after the matched evaluation.

The separate private-holdback alignment audit returned `GO` before this run:
visible-local argmax uplift `0.0778688524590164` with 95% task-bootstrap CI
`[0.033355532786885245, 0.1247728825136612]`, and task-equal pairwise rank
accuracy `0.6174863387978141` with CI
`[0.5724043715846994, 0.6598360655737705]`.  Its public aggregate file has
SHA-256 `2bd9aa7a0c4ce5740e670a7ab7a702a6522790f4dc60ec39355a4a7647f13117`.
The private journal is consumed, remains private, and is forbidden for
training, early stopping, reward tuning, checkpoint selection, or evaluation.

## Frozen parent and data boundary

- Parent: Arm C2 final `checkpoint-optstep-000058`.
- Parent run-contract canonical SHA-256:
  `a3d325af70ac9cd0a0c55cb7e66f4df2b390f78fab3ca6a70a930093ac989a00`.
- Parent adapter SHA-256:
  `80b50fab88e076d3e14771d09b5d1706baffeb2fd6c0c9d51b8841dd4135a004`.
- Input view: byte-identical opaque typed contract plus compressed enriched F2
  (`fn0`, parameters `p0`, `p1`, ...; semantic names absent).
- Cohort: the 150 unique training task identities selected before outcomes by
  the proxy-reward audit, in its sealed order.  Stored proxy candidates,
  actions, log probabilities, and rewards are forbidden for updates.
- A CPU-only preflight materializes a sealed target-free view containing only
  task identity, typed encoder source, visible feedback harness, and hashes.
  The GPU trainer must neither require nor load a gold target.
- Acceptance/private/holdback tests and their diagnostics are forbidden from
  the model and optimizer.

## Frozen optimizer and reward

- Planned updates: 150, one unique scheduled task per update (one pass).
- Fresh on-policy base group: 4 samples.
- Sampling: temperature `0.8`, top-p `1.0`, top-k `0`, maximum 8192 new
  tokens; sampled EOS is an action; PAD-before-EOS and source truncation fail
  closed.
- Reward components are independently centered: full visible-harness pass,
  density-calibrated isolated-visible-test reward, and compile reward.
- Weights: full-pass `1.0`, local `1.0`, compile `0.25`; local-test density
  calibration alpha is `2.0`.  These were
  frozen before the private holdback read and cannot be retuned from it.
- Compiler repair: only for base groups with no visible full or partial pass;
  at most two diverse noncompiling parents, four repairs per parent.  The
  diagnostic is produced by candidate-only compilation with a neutral
  `main()` and may not contain the visible harness.  Repair advantage updates
  only its repair-conditioned prefix.
- AdamW learning rate `1e-6`, zero weight decay, maximum gradient norm `1.0`,
  PPO clip `0.0`, no KL term, and SFT/gold replay weight exactly `0.0`.
- Saved rollout action log probabilities must match pre-update recomputation
  within `2e-4` under byte-identical conditioning and sampling support.

## Phase gates

The immutable run contract plans all 150 updates.  Execution first stops after
update 16 and resumes from that checkpoint with the same optimizer, RNG,
schedule, and run-contract hash only on `GO`; it never restarts training.

For each complete 16-update window (1-16, 17-32, ..., 129-144), all of the
following are required:

- at least 8 base groups have a unified centered advantage with absolute value
  above `1e-12`;
- at least 4 base groups have local advantage outside the centered span of
  full-pass and compile advantages, using
  `residual_sq > 1e-10 * max(1, local_l2_sq)`;
- at most 8 updates have zero active policy trajectories;
- an update has `optimizer_step=false` iff it has zero active trajectories;
- all active losses and gradient norms are finite and nonzero where required;
- all EOS, PAD, truncation, privacy, source, runtime, schedule, checkpoint, and
  on-policy-logprob invariants pass.

The final six-update window is descriptive but all integrity invariants remain
mandatory.  A failed phase gate stops the run and retains Arm C2.  Full-pass
count is reported but is not itself a mechanics gate.

## Evaluation and stopping rule

Only the final update-150 checkpoint is eligible for evaluation; heldout data
cannot select an intermediate checkpoint.  Compare Arm C2 and post-VeRPO on
the same 175 tasks and paired task/slot seeds, `k=10`, seeds 42, 43, and 44,
temperature `0.8`, top-p `0.95`, and maximum 8192 new tokens.  Report each
seed's pass@1/pass@10, candidate and task compile rates, candidate pass rate,
distinct programs per 10, and paired discordances.

The primary statistic is the per-task mean of the three seed-specific pass@10
indicators.  Its 95% interval uses 10,000 task-cluster bootstrap replicates
with bootstrap seed 42.  A `positive pilot signal` requires all three:

1. mean pass@10 gain at least `3/175`;
2. clustered-bootstrap lower bound greater than zero; and
3. pooled distinct-programs/10 drop no more than `0.10` versus matched C2.

If the interval excludes zero downward, classify `regression`; otherwise
classify `unresolved/small`.  Secondary metrics cannot overturn the primary
classification.  No outcome automatically promotes the VeRPO checkpoint.
Stop model training after this evaluation.
