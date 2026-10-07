# Fixed RS-SFT and VeRPO launchers

These foreground launchers are the production entry points installed at
`/workspace/run_finish_rs_sft.sh`, `/workspace/run_verpo_v2.sh`, and
`/workspace/run_rs_sft_then_verpo.sh`.

- Every launcher uses `set -Eeuo pipefail`, returns the Python exit status, and
  uses a non-destructive lock instead of killing unrelated jobs.
- RS-SFT writes to `/workspace/artifacts/text_arm_v2_s44_fixed`, re-certifies
  the legacy harvest, trains a same-seed/same-step gold-only control, and only
  passes its paired gate at an improvement of at least six percentage points.
- VeRPO starts only from the fixed RS-SFT checkpoint. DeepSeek judging is
  fail-closed, post-transform, bounded, and compile-gated. Generation and
  scoring are chunked one candidate at a time, and recovery checkpoints are
  written every optimizer step.
- `run_rs_sft_then_verpo.sh` stops if the RS-SFT causal gate fails. This is
  deliberate: VeRPO must not silently continue from a rejected checkpoint.

Run the stages in the foreground:

```bash
bash /workspace/run_finish_rs_sft.sh
bash /workspace/run_verpo_v2.sh
```

Or chain them:

```bash
bash /workspace/run_rs_sft_then_verpo.sh
```

True distribution KD is exposed at `/workspace/run_true_kd.sh`. The legacy
`soft_kd_trainer.py` and `build_softkd_data.py` entry points now fail closed;
their rounded top-k artifacts must not be mistaken for full KD.

The most useful safe overrides are `OUTPUT_ROOT`, `MAX_STEPS`,
`GRPO_GROUP_SIZE`, `GRPO_GRAD_ACCUM`, `GRPO_MAX_NEW_TOKENS`, and
`VERPO_JUDGE_MODEL`. Additional runner arguments can be appended directly.
