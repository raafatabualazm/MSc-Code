# Production Qwen, RS-SFT, and VeRPO launchers

The five predeclared training/evaluation steps are:

1. Build and seal the train-only inline-CFG v2 representation, migrate the
   old direct-compact overlay under the new contract, and begin the independent
   1,580-row gold adaptation.
2. Run the Qwen contract probe, deterministic 16-task K=8 quality pilot, and
   full 1,580-task K=8 `qwen3.8-max-preview` harvest.
3. Join the completed gold adaptation and Qwen corpus, then fit the student
   with equal-draw, EOS-inclusive sequence Monte Carlo forward-KL.
4. Generate fresh student failures, collect synchronous `gpt-5.6-sol`
   repairs, and train the matched RS-SFT intervention and gold-only control.
5. Run sparse bounded-teacher VeRPO on the sealed 1,232-task
   feedback-eligible view, then and only then open heldout175 for the
   descriptive four-arm evaluation.

The four operational commands implementing those steps are, in order:

1. `/workspace/run_qwen38_sequence_kd.sh`
2. `/workspace/run_collect_chatgpt_compact_rs.sh`
3. `/workspace/run_finish_rs_sft.sh`
4. `/workspace/run_verpo_v2.sh`

These are command numbers, not alternate step numbers. In particular, the
second command performs the GPT portion of Step 4; Step 2 is always the Qwen
pilot/full harvest.

The Qwen stage serializes the full lossless compressed enriched assembly and
explicit compressed CFG for the target and every retained user helper, checks
every prompt against the 12K input cap, makes
one bounded contract probe, then harvests K=8 independent
`qwen3.8-max-preview` draws per training task. Before the full fan-out, a
separate 16-task K=8 quality pilot must meet both the parseable-output floor and
at least one functionally verified task; its passed, hashed gate is bound into
the production journal. The collector itself refuses any paid full-corpus call
without that exact deterministic 16-task gate. The production launcher is
pinned to the exact preview model, sequence-only objective, 8,192-token thinking
budget, and 12,288-token output budget. The default
`QWEN_OBJECTIVE_MODE=sequence_only` uses thinking mode because it is the only
mathematically valid way to distill the reasoning model's marginal final-output
distribution from the available API.

Because the complete user-function representation is new, the Qwen launcher
first adapts the direct-compact student on all 1,580 sealed gold training rows
(the 175 held-out rows are evaluation-only). After contract/overlay migration
is dry-validated, gold adaptation may use the GPU while the paid contract
probe, 16-task pilot, and full Qwen harvest run concurrently. The final
sequence fit starts only after both sides complete and every K=8 draw has been
tokenized by the exact pinned student tokenizer, had EOS appended exactly as
the trainer does, and passed `max_target_tokens`.

In `sequence_only`, every completion-attested, trim-only final-content draw is
retained, including prose or fenced output: filtering those draws would change
the teacher distribution. The pilot instead requires zero length overflow, at
least the declared final-code/parseable fraction (default 50%), at least one
verified task, and non-pathological K=8 diversity. Hidden
`reasoning_content` is audit metadata only. Draws are never truncated,
filtered, resampled, or silently replaced.

The production base warm-start is the byte-preserving recovered checkpoint
`/workspace/artifacts/direct_compact_fn0_real_sft_v1_self_sealed_recovered`.
Its adapter, overlay, and run provenance are unchanged from the original; only
the mismatching contract copy was replaced by the contract already bound in
that immutable provenance. The installer and Qwen launcher require
`/workspace/direct_compact_fn0_self_seal_recovery.json` at SHA-256
`41e0dd7ecf68ebb0b560c66266d686b49afe9e59c6390f1e4079854dca6a7c9b`,
revalidate every recorded artifact, and never modify the source checkpoint.

The explicit `QWEN_OBJECTIVE_MODE=require_top5` alternative fails before
fan-out unless the exact preview endpoint returns usable content logprobs and
five raw-byte alternatives per position. Production sampling is sealed to the
exact identity transform
`temperature=1`, `top_p=1`, and Alibaba `top_k=101` (the provider's documented
top-k-disable setting); any other value is rejected because it would estimate
the wrong teacher distribution. There is
no automatic downgrade. This mode also requires `enable_thinking=false`:
content-token logprobs produced after a hidden reasoning trace are conditioned
on a prefix the student never sees and therefore are not a valid token-KL
target.

`QWEN_OBJECTIVE_MODE=sequence_only` omits the logprob request fields and trains
EOS-inclusive summed sequence NLL, averaged with equal weight over
all K=8 sampled final-content sequences. It deliberately does not use the base
model's length-normalized token-mean loss. This is a Monte Carlo estimator of
sequence-level forward cross-entropy/KL after an attested trim-only transform;
it is not dense token KL and cannot enable the sparse top-5+tail auxiliary.
Thinking is enabled by default in this mode because sampled final sequences
remain valid draws from the reasoning model's marginal final-output
distribution. Neither mode is ever described as dense/full-vocabulary KD.

The OpenAI RS collector refuses to run before the resulting Qwen checkpoint,
dataset, audit, and build manifest are cryptographically joined. It derives
fresh failures only on the exact 1,578-row executable multi-function view; the
two audited filesystem/FFI tasks can be imitated by Qwen but can never enter
candidate replay or execution reward. It uses the
official synchronous Responses API with `gpt-5.6-sol`, high reasoning effort,
and no temperature parameter. For each failed training task it includes the
shortest whole failed student attempt when the exact prompt remains within
12K; otherwise it omits only that optional attempt. It never truncates the F2
assembly/CFG representation. Private tests and gold source are not sent to the
API. The first request uses 8,192 output tokens and escalates once to 12,288
only when the Responses API explicitly reports `incomplete` because
`max_output_tokens` was reached. The default resumable output is
`/workspace/artifacts/chatgpt_rs_qwen38_inline_cfg_v2_gpt56`.

RS-SFT trains a source-matched gold control and a 50/50 repair intervention
from the exact same direct-compact Qwen adapter plus source-embedding overlay.
Before either fit begins, the launcher seals the complete Qwen -> matched
RS/control -> VeRPO -> evaluation order and all train-side artifacts and
hyperparameters. Neither arm loads heldout175 while fitting, and heldout
performance never selects a checkpoint or decides whether VeRPO launches.

VeRPO continues the direct-compact architecture, never the retired graph/text
arm. Its rollout universe is the exact 1,232-task feedback-eligible subset of
the sealed 1,578-row executable view—not the duplicated RS epoch. The
predeclared schedule covers every eligible task before cycling. Each
current-policy group has eight samples by default with `top_p=1` and
`top_k=0`. Full and per-test rewards use the hardened Dart completion
attestation. The live teacher is `gpt-5.6-terra` through the Responses API in
standard mode with high reasoning effort. It receives the same task-joined F2
compressed enriched assembly, compressed CFG, and visible feedback harness;
references and hidden acceptance tests remain private.

The live teacher is deliberately sparse and bounded. Only every eighth group
is eligible, a group is skipped when any candidate fully passes or fewer than
two candidates compile, and one request compares only the two strongest
compiling failures. The request has a 60-second deadline, zero transport
retries, zero completion retries, and a global call budget derived from the
number of rollout groups. A timeout, malformed response, API error, or
exhausted call budget contributes no teacher advantage: training continues
immediately using only execution rewards and writes the unresolved group to
the idempotent offline escalation queue. Missing teacher scores are never
converted to numeric zero scores; they are masked with exactly zero advantage,
while observed scores are centered only over the observed pair.

`reasoning_mode=pro` is rejected in `sparse_inline`. Pro may consume the
offline queue later, but its result is never attached retroactively to a
rollout whose optimizer update has already run. Privacy-safe per-call receipts
form a hash chain across resumes. OpenAI Responses receipts require a unique
response ID and positive, internally consistent usage; exact response-model
echo and a system fingerprint are required only for compatible-chat providers
that promise them. Every optimizer checkpoint seals the receipt chain,
response-ID set, adapter, compact overlay, optimizer/RNG state,
run/data/model contract, rollout logprobs, candidate teacher-selection masks,
and judge telemetry.

Only after VeRPO publishes its final optimizer checkpoint does the launcher
open the measure split. Qwen sequence-KD, matched gold control, GPT RS-SFT, and
sparse-teacher VeRPO are then generated and scored with byte-identical settings
on the same sealed 175 tasks. That four-arm report is descriptive only and is
explicitly marked as unused for stage launch, stopping, or checkpoint choice.
Generation and scoring each use an append-only, externally headed hash-chain
journal. Completed batches resume exactly; each generation batch has an
independent hash-derived seed. A batch consumed without a terminal record is
indeterminate and fails closed—its K slots are never replaced or rerun.

The launchers pin the sealed parent build report
`7a9fdd032fef34c43ac5e7b8217b6b0b4c986b7dfdf0f1b4b6897aec01df241f`
and exact 1,578-row executable-view report
`9a2621ef0262db7d05f795d79347d67cd01230a864af2f1154b744f6f01132f4`.
This prevents fallback to an older `fn0`, graph, or compressed-input artifact.

The no-network GPU initialization is the non-autostarting Supervisor program
`qwen38_gold_only`; it exports `QWEN_GOLD_ONLY=1`, runs the canonical launcher,
validates the completed gold checkpoint, and exits before any API probe. The
paid Qwen path is the separate non-autostarting `qwen38_kd` program. Both share
the same exclusive lock, and the paid path reuses the validated gold
checkpoint. A Token Plan endpoint is rejected for automated harvesting; use an
authorized Alibaba Model Studio PAYG endpoint/key. The remaining synchronous
OpenAI harvest, matched RS-SFT, and VeRPO chain is the non-autostarting
`post_qwen_pipeline` program.

Installation also requires the staged
`frontier_ceiling_patch_v1/frontier_f2.py` to match SHA-256
`097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce`.
Only that pinned file is installed and receipted; the surrounding frontier
package is neither replaced nor pruned.
Install the full canonical runtime from a staged bundle with:

```bash
bash fixed_training_launchers/install_remote_fixes.sh /path/to/staging
# Optional while the paid endpoint is unavailable:
supervisorctl start qwen38_gold_only
# Later, with an authorized PAYG Qwen endpoint:
supervisorctl start qwen38_kd
# After qwen38_kd exits successfully and OPENAI_API_KEY is present:
supervisorctl start post_qwen_pipeline
```

True dense distribution KD remains available only through
`/workspace/run_true_kd.sh` with a compatible local teacher. API top-5
logprobs are never relabeled as dense/full-vocabulary KL. The legacy graph
launchers are retained solely for forensic reproduction and are not called by
the production chain.
