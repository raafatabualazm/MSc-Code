# Typed RS-SFT Arm B: operational claim and decision amendment

Status: fixed before Arm B training. This document supplements, and does not
rewrite, `RS_SFT_FOLD_PREREGISTRATION.md`. It was recorded while Arm A's
seed-42 evaluation was running and before the Arm B stage emitted a run
contract.

## 1. What Arm B tests

Arm B is a practical bundled-recipe comparison, not a single-factor causal
test of folding.

Relative to the stacked sequence it changes all of the following together:

- restart point: typed SFT `optstep348`, rather than pass-1 `update58`;
- data arrangement: one union of all verified direct targets, rather than
  sequential new-only stages;
- schedule: one epoch at `5e-6`, rather than the later stacked stages' two
  epochs at `2e-5`;
- data volume per stage: the final union is `458 = 447 + 11` unique tasks.

Therefore the supported positive claim is:

> A fresh, low-LR, one-pass fold of all verified direct targets works better
> than the observed high-LR sequential-stacking recipe.

Arm B cannot by itself support:

> Folding, independently of learning rate, restart point, and schedule, is the
> mechanism that caused the difference.

That causal claim would require at least an LR-matched folded control and a
low-LR stacked control.

## 2. Exact Arm B composition

Arm B has zero gold replay, but it is not pure self-distillation:

- 331 student-generated verified targets;
- 127 externally generated verified targets;
- 458 total verified direct targets, or 16.5% of the 2,776-row typed training
  universe.

It is still a strongly success-selected slice. A null result therefore tests
the direct-only folded recipe; it does not test replay-assisted RS-SFT.

## 3. Fixed seed-42 gate

Seed 42 may veto an arm, but may not promote it.

- Primary collapse read: mean distinct extracted-code hashes per 10 samples.
- Diversity eligibility: `distinct/10 >= 9.90`.
- Formal collapse floor: `distinct/10 < 9.50`.
- `pass@10 >= 18/175` at seed 42 is only a trigger for matched replication,
  not evidence that Arm B improves.
- Compile@10 and pass@1 are secondary and cannot carry promotion.
- Promotion requires at least three matched seeds. Differences below roughly
  three solved tasks are treated as sampling noise pending those replicates.

The earlier `9.88` value is update58's observed diversity, not the eligibility
bar. Under the fixed 9.90 rule, update58 itself is diversity-ineligible.

## 4. Decision tree

### B1. Seed-42 diversity is below 9.90

The fold-only bundled recipe fails the diversity gate. Below 9.50 it is
reported as collapsed. Arm C becomes the priority test because replay may
stabilize the distribution. This outcome does **not** establish that replay
was load-bearing; it establishes only that fold-only was insufficient.

### B2. Seed-42 pass@10 is at least 18 and diversity is at least 9.90

Run matched seeds 43 and 44 before interpreting pass@10. Only the matched
three-seed result may support the practical claim that the bundled folded
recipe helps. It still cannot identify folding as the sole mechanism.

### B3. Diversity is preserved but the matched result is within the noise
floor of typed SFT

Conclude that direct-only RS-SFT is inert under this folded recipe. This closes
the direct-only line, not the replay-assisted line. Arm C is required before
claiming RS-SFT is inert regardless of arrangement.

### B4. The matched folded recipe improves

Retain Arm B as the practical winner. Arm C then tests whether adding replay
provides further benefit; it is an additivity test, not required to establish
the practical Arm B result.

## 5. Arm C, if triggered

Arm C is a separate experiment: the same 458-row direct union plus 1:1 typed
gold replay, from typed SFT `optstep348`, one epoch at `5e-6`. Gradient
accumulation should be 16 so its optimizer-step count matches Arm B despite
twice the rows. Replay must exclude direct task IDs and direct source hashes.
Arm C is not part of Arm B and is not currently running.
