# RS-SFT fold-vs-stack: pre-registration and patch spec

Written 2026-08-02, **before** the Kimi c002 tranche lands and before pass 3 runs.
Nothing in `hybrid_training_patch_v2_3/` is modified by this document.

## 1. What the sealed data already shows

All arms below are the same 175 held-out tasks, same order, same eval source
(`dev_multifunction_binary.jsonl`, `abc8499f…`), seed 42, k=10. Verified, not assumed.

| arm | pass@1 | pass@10 | compile@10 | distinct/10 |
|---|---|---|---|---|
| untyped 2-epoch SFT | 2 | 7 | 124 | 9.99 |
| typed-contract 2-epoch SFT (optstep348) | 5 | 14 | 171 | 10.00 |
| typed RS-SFT pass 1 (update58) | 5 | 18 | 169 | 9.88 |
| typed RS-SFT pass 2 (update54) | 6 | 12 | 172 | 9.45 |

Exact McNemar (two-sided sign test on discordant pairs):

```
untyped   -> typed      compile@10  124 -> 171   49 gain,  2 loss   p = 1.2e-12
untyped   -> typed      pass@10       7 ->  14   11 gain,  4 loss   p = 0.118
typed SFT -> pass 1     pass@10      14 ->  18    6 gain,  2 loss   p = 0.289
pass 1    -> pass 2     pass@10      18 ->  12    1 gain,  7 loss   p = 0.070
```

The typed contract is the largest effect in the programme. Pass 1 did not regress.
**Pass 2 is the regression, and it is the only near-significant result in the series.**

### Mechanism: measured, not inferred

Distinct programs emitted across the 10 samples, per task:

```
                          3    4    5    6    7    8    9   10
typed SFT 2ep             0    0    0    0    0    0    0  175
typed RS-SFT pass 1       0    1    0    0    0    4    7  163
typed RS-SFT pass 2       1    0    3    4    3   13   23  128
```

Successes per solved task:

```
typed SFT 2ep    [1,1,1,1,1,1,1,1,2,2,2,2,4,6]   8 of 14 solves are a single lucky draw
typed RS-SFT p2  [1,1,2,3,4,5,6,6,8,9,9,10]      median 5.5, one perfect 10/10
```

Pass 2 became more reliable on what it knew (pass@1 5→6, compile@10 169→172) and lost
seven of the eight tasks whose only solve was a 1-in-10 tail event. RS-SFT sharpens the
mode; pass@10 measures the tail. They are in direct opposition.

### Why it could not self-correct

From `t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1/dataset_manifest.json`:

```
warmstart_update: 58        started from the pass-1 checkpoint, not the SFT baseline
prior_225_replay: 0         never saw pass-1's rows
gold_replay: 0              never saw gold
rows: 209 (190 student + 19 teacher), 54 updates
```

Plus `visible_all_zero_tasks: 2238` of 2550 — the harvest can only draw from the easiest
~12% of the pool, and 190 of the 209 rows are the student's own successes, which carry no
new information and all of the sharpening pressure.

### The replay block is structural, not a config slip

```python
# scripts/training/t5gemma2_typed_direct_rs_sft_pass2.py:1197
raise ValueError("pass-2 accepts only its sealed direct-only no-replay profile")

# scripts/training/t5gemma2_typed_direct_rs_sft.py:424   (pass 1)
"gold_replay": {"selected_rows": 0, "forbidden": True}
```

`scripts/training/t5gemma2_mixed_rs_sft.py` implements replay fully (`replay_pairs`,
rows = ratio x len(rescue_pairs), lines ~925–1099) — but that is the **untyped** lineage.
The typed contract and gold replay have never been in the same pipeline.

## 2. Pre-registered prediction for pass 3

Pass 3 as currently planned — branch from `update58`, genuinely-new targets only, replay
unavailable on this code path — is structurally identical to pass 2.

> **Prediction.** pass@10 falls or stays flat versus update58's 18/175; compile@10 and
> pass@1 stay flat or rise; mean distinct-candidates/10 drops below 9.88.

The diagnostic signature is the **divergence**: two metrics improving while pass@10 falls
is entropy collapse, not progress. Pass 2 showed exactly this and reads as improvement on
two of three headline numbers.

Falsification: if pass 3 raises pass@10 above 18 **and** holds distinct/10 at or above
9.88, the stacking theory is wrong and stacked RS-SFT is viable here.

## 3. Decision rule, fixed before results

- pass@10 is the **primary** endpoint. compile@10 and pass@1 are secondary and cannot
  carry a promotion decision on their own.
- Promote a checkpoint only if pass@10 is at or above the incumbent **and** mean
  distinct/10 has not fallen by more than 0.10 from the typed SFT baseline's 10.00.
- Any arm whose distinct/10 falls below 9.5 is reported as collapsed regardless of its
  other metrics.
- All numbers here are seed 42 only. The 14 / 18 / 12 spread across checkpoints is not
  established as larger than seed noise; the running replication is what settles it.
  The compile@10 typed-contract result (49 gain, 2 loss) is far too large to be seed noise.

## 4. Patch spec for the fold arm (arm B)

Two factors change together relative to arm A, so run both or the comparison is confounded.

| | arm A (as planned) | arm B (fold) |
|---|---|---|
| warmstart | `…typed_direct_rs_sft_225_v1/checkpoint-optstep-000058` | `…typed_contract_sft_2epoch_v1/checkpoint-optstep-000348` |
| targets | new Kimi tranche only | 225 (pass 1) + 209 (pass 2) + new Kimi |
| gold replay | forbidden by the script | 3:1 against harvest rows |
| epochs / LR | 2 / 2e-5 | 1 / 5e-6 |

Arm B cannot be expressed as flags on the typed-direct path. Cheapest honest route:

1. Copy `t5gemma2_typed_direct_rs_sft_pass2.py` to a **new** filename
   (`t5gemma2_typed_fold_rs_sft.py`) — do not edit the original.
2. Remove the `gold_replay_ratio != 0.0 or gold_replay_rows != 0` clause from the profile
   guard (~line 1191) and the hardcoded `"gold_replay_ratio": 0.0 / "gold_replay_rows": 0`
   in the emitted contract (~line 1421).
3. Change `_load_prior_225_manifest` from `used_for_exclusion_only: True` (~line 631) to
   loading those rows as training pairs.
4. Lift the replay selection block from `t5gemma2_mixed_rs_sft.py` (~1029–1048) verbatim;
   it already computes `ceil(ratio * len(rescue_pairs))` and tags pairs `kind="gold_replay"`.
5. Keep every leakage guard untouched: `heldout_175_model_visible`, `tests_model_visible`,
   `prior_success_exclusion`, and the sha256 pinning in the launcher. These are well built.

## 5. VeRPO

The typed contract pushed compile@10 to 171/175 = 97.7%. A compiler-only reward is now
**saturated**: nearly every rollout compiles, so within-group reward variance is ~0,
advantage is ~0, and there is no gradient. The untyped pilot at compile@10 = 71% had more
variance to work with and still produced 7 -> 6 with minimum attainable p = 0.0625.

> **Prediction.** Compiler-only VeRPO on the typed arm is more degenerate than the untyped
> pilot, and moves pass@10 by no more than +-2 tasks.

A test-based reward gives a real but thin surface: only the ~18/175 solved tasks produce
mixed rollout groups (per-task success counts run 1–9 of 10, none saturated), so ~90% of
groups still have zero advantage. GRPO is additionally entropy-reducing; without an
explicit KL or entropy term it sharpens exactly as RS-SFT did, against a tail metric.

Minimum bar before VeRPO is worth GPU time: a graded reward with variance in the operating
regime, a KL/entropy regulariser, and distinct/10 tracked as a stopping guardrail.

## 6. Teacher yield ledger (why more teacher spend is not the lever)

| teacher | condition | calls | verified | yield |
|---|---|---|---|---|
| Sonnet-5 high | fresh pool | 193 | 67 | 34.7% |
| Kimi-K3 | pass-2 initial | 50 | 15 | 30.0% |
| Kimi-K3 | pass-2 retry | 15 | 2 | 13.3% |
| Opus-5 high | picked-over residual | 40 | 5 | 12.5% |
| Sonnet-5 high | pass-2 residual | 38 | 2 | 5.3% |
| GPT (Azure) | residual | 20 | 1 | 5.0% |
| MiniMax-M3 | residual | 20 | 0 | 0.0% |

Already spent in the `api_rs_sft` series: 2.38M input + 1.38M output tokens.
73 verified teacher rows sit on disk (72 unique, **zero overlap with the 175 held-out**).

Yield falls as the residual is picked over *while the teacher gets stronger*
(Sonnet 34.7% fresh -> Opus-5 high 12.5% residual). That is a task-difficulty gradient,
not a capability gradient, and it independently corroborates the sealed frontier ceiling
(DeepSeek-V4-Pro, 4/175 = 2.29% pass@10).
