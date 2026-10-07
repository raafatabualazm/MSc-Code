# Review: codex KD trainers (local snapshot + 167.172.150.125)

Method: 6 parallel reviewers over the KD stack → 3-lens adversarial verification per finding
(2/3 refutes kills) → this synthesis. 48 findings raised, 11 survived, 24 refuted, 9 lost their
verifiers to a session limit (flagged UNVERIFIED below — treat as unconfirmed, not dismissed).

---

## VERDICT

The *derivation* is sound and codex is unusually honest about what it is: coarsening the vocab
into {top-5, complement} and taking forward KL on that partition is a legitimate data-processing
**lower bound** on full-vocabulary KL; the direction is right, the `temperature==1` restriction is
correctly motivated and enforced, the docstring says outright *"It is never dense/full-vocabulary
KD"*, and the manifest validator **fail-closes if the manifest claims dense KL**.

But the objective you flagged has **never executed and cannot execute as written**. Three
independent blockers: (1) a logits-indexing bug that crashes on the first micro-batch carrying
sparse positions, (2) a NaN/-inf trap in the tail term, (3) a data gate whose tolerance (1e-8) is
~20x tighter than the provider's own numeric noise (measured median inferred tail **-2.06e-07**),
so ~50-95% of positions are rejected and a whole response dies on any one of them. Even if all
three were fixed, the auxiliary is **numerically inert**: a per-position *mean* KL times 0.10 added
to a per-sequence *summed* NLL is ~0.03% of the gradient.

Meanwhile the arm that actually runs is not top-k KD at all — it is sequence-level distillation on
**unverified teacher text measured at 0.89% correct / 8.8% compiling**, with gold replay hard-
forbidden in three places and evaluation disabled. That is a strict downgrade of a 100%-correct
gold corpus and should be expected to *reduce* pass@k.

**Recommendation: do not run either arm as configured.** Nothing has been burned yet — the harvest
has been idle 27h and the box has no torch.

---

## BLOCKING DEFECTS

### A. The top-k KL arm (never ran; would crash or corrupt)

1. **[CRITICAL, 0/3 refuted] Absolute label positions indexed into a truncated logits suffix**
   `direct_compact_sparse_topk_tail.py:554`
   With `sequence_sum_nll` on (the sole launcher hardcodes it), `DirectCompactCausalLM` sets
   `logits_to_keep = labels.size(1) - first_prediction`, so the decoder returns only the last
   ~302 positions — but KD gathers `logits[batch, prediction_tensor]` with absolute indices
   (~1499-1798). **Fix:** subtract the offset `labels.size(1) - logits.size(1)` at :531/:554.
   Fails loudly (IndexError / CUDA device-side assert) in real shapes; silent misalignment only
   in degenerate short-prompt shapes.

2. **[CRITICAL, 0/3 refuted] NaN backward from the tail term**
   `direct_compact_sparse_topk_tail.py:337-350, 430`
   When `teacher_tail == 0.0` *and* `student_top_log_mass == 0.0` (fp32), `_log_one_minus_exp`
   yields -inf and the discarded `torch.where` branch evaluates `0*inf`. The forward finite-check
   at :436 **passes**, then backward writes NaN across the whole logit row → every trainable
   parameter, bf16 so no GradScaler skip, AdamW state permanently poisoned. Verifiers corrected
   the mechanism: the trap lives *in* `_log_one_minus_exp`, merely *exposed* by the `where`.
   **Fix:** compute the tail term in a numerically safe branch (mask before multiply), and treat
   `teacher_tail == 0` as "skip this position", not "zero the term".

3. **[HIGH, 0/3 refuted] `_log_one_minus_exp` returns -inf at exactly 0.0**
   `:349` — guards check `> 1e-7` and `> 0` but not `== 0`; falls through to `log(-expm1(0)) =
   -inf` → `FloatingPointError` mid-run. Measured: at a top-1 logit gap of 20-24 nats,
   `logsumexp(top5)` is **exactly 0.0** in fp32 for 12/12 seeds at V=172000. A warm-started
   student hits this within an epoch.

4. **[CRITICAL, 1/3 refuted — both upholders said it is *understated*] Data yield is ~0**
   `qwen_direct_compact_teacher_artifact.py:41,1142` — `NEGATIVE_TAIL_TOLERANCE = 1e-8` vs the
   audited provider distribution (median inferred tail **-2.06e-07**, p05 -1.33e-05, p95 still
   negative at -1.84e-10). One bad position rejects the **entire response**, and the handler
   consumes the draw with no retry. `build_qwen_sparse_topk_tail_auxiliary.py:151` is stricter
   still (`tail < 0.0`, zero tolerance). With ~292 positions/response, P(clean response) ≈ 0.
   **Fix:** set the tolerance from a measured characterization of provider precision, and reject
   *positions*, not whole responses.

5. **[HIGH, 0/3 refuted] The auxiliary cannot influence the optimizer**
   `:565-566` — `auxiliary = auxiliary_sum / position_count` (mean) added as
   `primary + 0.10 * auxiliary` where `primary` is a per-sequence **sum** (~90 nats for a
   300-token target). Effective KD coefficient ≈ `w / L_target` ≈ **3e-4**, and it varies
   inversely with target length across the dataset. A null here would say nothing about KD.

6. **[HIGH, 0/3 refuted] Green tests on unrunnable code**
   `tests/test_qwen_sparse_topk_tail.py:86` — the stub LM does `del kwargs` and returns
   full-length logits, so defect #1 is structurally invisible. A sibling stub
   (`tests/test_direct_compact_path.py:97-104`) *does* honor `logits_to_keep`, proving the authors
   knew it must be. **Fix:** make the stub honor the kwarg; the test then fails as it should.

> Note on causality: the top-k data does not exist **not because the provider refused**, but
> because the production path is contractually forbidden from asking — `require_top5` demands
> `enable_thinking=false`, and the sealed launcher pins `QWEN_OBJECTIVE_MODE=sequence_only`.

### B. The arm that actually runs (sequence-KD) — the higher-stakes problem

7. **[CRITICAL, 1/3 refuted] Unverified teacher text used as gold**
   `build_qwen_sequence_kd.py:8` — docstring verbatim: *"No correctness, confidence, parseability,
   or logprob filter is applied"*; manifest seals `correctness_filtering: False` and
   `verified_only_rs_sft_consumed: False`. Measured over 3,477 production draws: **0.89% correct,
   8.80% compiling**. One verifier called the true state *worse* than claimed; another downgraded
   to "high, prospective" because the harvest is still partial and no train artifact exists on
   disk yet. Either way the student (69% compile) would be fit toward an 8.8%-compile target.

8. **[CRITICAL, 1/3 refuted] Length weighting concentrates gradient on the worst outputs**
   `models/direct_compact_causal.py:2448/2532` — per-sequence **sum** NLL, mean over draws.
   Failing draws average ~7x longer; one 24,541-token degenerate draw carries ≈ the gradient of
   **68 correct 358-token answers**, and only 31 of 3,477 draws are correct.
   *Important correction from verifiers:* sum-within-sequence is the **mathematically correct**
   unbiased MC estimator of sequence forward-KL — token-mean would be biased. **The defect is the
   corpus, not the loss kernel.** Fix the corpus, not the reduction.

9. **[SURVIVED, 0/2 refuted] `reasoning_content_excluded` seal does not constrain the text**
   `build_qwen_sequence_kd.py:577` — the code-only gate is applied only under `require_top5`, so in
   the production `sequence_only` arm non-code targets are retained *by design* (pilot: 76/128 =
   **59% non-code**), while the manifest still asserts `reasoning_content_excluded: True`. The
   student is trained to emit `think>` prose and `<tool_call>` markup as its answer.

10. **[SURVIVED, 1/3 refuted] The quality gate cannot fail**
    `run_qwen38_supplemental_harvest.sh:277` — `--minimum-verified-tasks` defaults to **0**; the
    recorded gate has `verified_tasks: 0, passed: true` with an empty verified-only file
    (sha `e3b0c442…` = empty). It authorized a 9,568-call paid fan-out on a teacher that solved
    nothing.

11. **[SURVIVED, 0/2 refuted] Migration guarantees dangling entry points**
    `experiment_workspace/fixed_training_launchers/migrate_workspace_167.sh:39` — copies **all**
    `run_*.sh` by glob but payload dirs by whitelist, so launchers arrive without their targets.

### C. UNVERIFIED (verifiers died on a session limit — unconfirmed, worth checking)

- No control arm for the KD stage: the four eval arms are sequential points on one chain, so the
  KD stage's own effect is structurally unattributable (`seal_post_qwen_chain.py:768`).
- KD arm evaluated at **1024 max_new_tokens** in `qwen_cot_v1` prompt mode while its supervision
  teaches multi-thousand-token `<think>` blocks → pass@10 ≈ 0 by truncation. *This is the exact
  failure mode we already hit and fixed in the frontier eval.*
- Study not powered: on 175 tasks 1 task = **0.571pp**; the exact one-sided McNemar floor is ~5
  zero-regression wins (2.86pp), realistically 8-11 net tasks (4.6-6.3pp). Every historical
  imitation delta in our table is 1-2 tasks — inside noise. No min-improvement gate covers the KD
  arms (the 6pp McNemar gate exists but only for RS vs matched-gold).

---

## WHAT CODEX GOT RIGHT (verified, do not re-litigate)

- **No test leakage.** train 1580 / dev 175: **0** task_id overlap, **0** target-text overlap,
  **0** compact-input-id overlap; the 2776 and 1196 corpora are also 0/0/0 vs dev; max 5-gram
  Jaccard 0.257. Enforced *structurally* — every teacher task must bijectively join a
  seal-validated `expected_role="fit"` row or the build raises. The teacher is genuinely sigless.
- **The coarsening math is correct** — valid data-processing lower bound, correct KL direction,
  correct `T^2` pairing in the dense variant, and the `temperature==1` restriction is right (you
  genuinely cannot re-temper an aggregate tail).
- **Honest labeling and fail-closed seals** — manifest rejects dense-KL/full-KD claims and global
  tokenizer-identity claims; `negative_teacher_tail_policy: "reject_never_clamp"` refuses the
  renormalization fudge; EOS correctly excluded from the auxiliary and left to the primary NLL.
- **Byte-based token-ID recovery** (a real improvement over re-tokenizing decoded strings), failing
  closed on multi-token/non-standalone-UTF8 pieces.
- **Vocabulary safety** — the +20.5K compact codebook IDs are input-only, the LM head is asserted at
  `contract.base_vocab_size`, and teacher IDs ≥ that are rejected, so the teacher distribution is
  well-defined over the student's output vocab with no codebook collision.
- **Position mapping is re-derived at runtime** from `labels != -100` and cross-checked
  token-for-token against the sealed `observed_token_id`; padding correctly `-100`.
- The `verified_repairs_lp.jsonl` audit artifact is **not** lost — our 2026-07-23 snapshot
  preserves it and its sha256 re-verifies (`e9892bfe1d12…`).

---

## RECOMMENDATION

1. **Do not launch the sequence-KD arm as configured.** It is not a neutral test of KD; it is a
   ~99%-wrong corpus with no gold anchor and no eval, warm-started from your best gold-adapted
   checkpoint. If run, expect a large negative and an uninterpretable one.
2. **If you want a defensible imitation arm at all**, the only honest version is: verified draws
   only (31/3,477 today), gold replay mixed back in (currently forbidden in 3 places), eval
   enabled, plus a pre-KD control arm and the 6pp McNemar gate extended to the KD arms. Note that
   verified-only ≈ RS-SFT, which we already measured at **1.71% (null)**.
3. **The top-k KL arm is not worth the repair cost.** Its own math shows the tail carries ~1.6e-5
   of the mass, so it is numerically ≈ renormalized top-5 CE ≈ hard-target RS-SFT — the thing that
   already returned null — and it needs 3 code fixes plus a data path that can actually yield rows
   before it would even be inert-but-runnable.
4. **Spend the compute on the two levers with real evidence instead:** enrichment (the only lever
   that ever moved pass@k: 0.57 → 5.71, and the one that doubled the *frontier* ceiling
   8.33% → 16.67% on v3's richer inputs), and VeRPO/execution reward — the only non-imitation
   lever, which needs no teacher and optimizes pass@k directly.
5. **Practical:** the GPU pod (98.218.15.126) is gone; 167.172.150.125 is CPU-only with no torch.
   Any training needs a new GPU host, so this is the cheapest possible moment to change course.
