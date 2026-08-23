# x86 intervention multi-seed staging

## Fixed experiment

This is the 12-run experiment needed for the SANER measurement and input-channel
sections.  It does **not** train or promote a checkpoint.

- frozen checkpoint: original untyped T5Gemma 2 two-epoch SFT,
  `checkpoint-optstep-000348`
- held-out set: sealed 175 x86-64 tasks, K=10
- decoding: temperature 0.8, top-p 0.95, 32,768 source tokens, 4,096 output
  tokens
- interventions: `typed_opaque_contract`, `constants_stripped`, and
  `semantic_body_swap`
- fresh seeds: 43, 44, 45, 46
- reused context: each intervention at seed 42 and baseline seeds 42--46

Therefore: 3 interventions x 4 new seeds = **12 fresh runs**.  This must not be
confused with `t5gemma2_typed_seed_replication_arm_v1.sh`, which evaluates
different, typed-trained checkpoints and answers a different question.

## Time and storage

The earlier 15-hour estimate was optimistic.  Six observable run boundaries on
the same RTX Pro 6000 took 96.97--116.62 minutes of generation each (mean
108.54), before roughly 2--4 minutes of CPU scoring.  The defensible budget is
**20--24 GPU-hours** for all 12.  A 14-hour rental is likely to finish about
seven runs, not twelve.

The artifacts are small: the three existing seed-42 arms occupy 25.83 MB, which
projects to about **99 MB** for the 12 fresh arms.  The current 13 GiB free is
ample if VeRPO does not consume the remaining disk unexpectedly.

## Staged files

- `hybrid_training_patch_v2_3/deploy/vast/t5gemma2_measurement_intervention_multiseed_v1.sh`
- `hybrid_training_patch_v2_3/deploy/vast/t5gemma2-measurement-intervention-multiseed-v1.conf`
- `hybrid_training_patch_v2_3/scripts/evaluation/t5gemma2_measurement_intervention_multiseed_report_v1.py`
- `hybrid_training_patch_v2_3/scripts/evaluation/verify_t5gemma2_measurement_runtime_compat_v1.py`
- `hybrid_training_patch_v2_3/deploy/vast/t5gemma2_measurement_intervention_after_verpo_handoff_v1.sh`
- `hybrid_training_patch_v2_3/deploy/vast/t5gemma2-measurement-intervention-after-verpo-handoff-v1.conf`
- `hybrid_training_patch_v2_3/tests/test_t5gemma2_measurement_intervention_multiseed_v1.py`

The launcher is exact-resumable through the existing hash-chained inference and
scoring journals.  It pins the checkpoint, datasets, evaluator, transformation
code, seed-42 report, and Rank-0 175/175 gold round-trip.  It acquires the shared
evaluation lock, waits for an idle GPU, requires at least 5 GiB free, and
snapshots the frozen checkpoint before and after the run.

The historical runs predate the current checkpoint-loader and evaluator
extensions.  A mandatory compatibility preflight therefore regenerates the
first five typed seed-42 tasks (50 candidates) and requires byte-identical
outputs, then rescores all 1,750 historical typed seed-42 candidates and
requires identical candidate compile/pass decisions and task metrics.  A
mismatch stops before the first fresh run.

Runs are view-major in scientific priority:

1. typed opaque contract, seeds 43--46;
2. constants stripped, seeds 43--46;
3. semantic body swap, seeds 43--46.

This ordering makes a time-limited partial run useful: it completes whole
five-seed intervention arms rather than leaving every arm under-replicated.

The final report rejects mismatched task order, model, scorer, source view,
sampling coordinates, privacy flags, or seed.  For constants-stripped tasks
whose source did not change, it additionally requires byte-identical predictions
to the corresponding same-seed baseline for all five seeds.

## Exact next action

Deploy the staged runtime/report/launcher files, run the focused tests plus
`bash -n`, and install both Supervisor configs with `autostart=false`.  Do not
start the x86 program directly while VeRPO evaluation is active, and do not
lower the 4,096-token cap: either change would break the paired
frozen-checkpoint design.

For the pre-armed automatic transition, start
`t5gemma2-measurement-intervention-after-verpo-handoff-v1` while the VeRPO
matched evaluation is running.  The handoff is CPU-only.  It requires the
upstream supervisor to reach `EXITED`, checks the stable three-seed report and
its fixed `STOP_AFTER_MATCHED_EVALUATION`/no-promotion disposition, and only
then starts the x86 program.  An upstream exit without that report blocks the
transition; GPU idleness alone is deliberately insufficient.
