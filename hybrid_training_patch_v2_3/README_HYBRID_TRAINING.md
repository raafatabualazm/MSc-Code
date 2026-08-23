# Hybrid Decompiler Training Patch v2.3

This overlay replaces the v2.1 rule that routed functions above 199 parsed
instructions out of supervised training. In v2.3, every Phase-0-approved length
stratum participates in training.

The ≥200-instruction population receives:

1. direct whole-function supervised fine-tuning;
2. deterministic CFG-region-plan supervision;
3. plan-conditioned whole-function reconstruction;
4. length-balanced code-only recovery;
5. verifier-bounded rejection-sampling SFT;
6. optional verified-only VeRPO.

Only a deterministic, proportionally length-stratified development slice is
withheld. The bridge and long JSONL files are training-stratum artifacts for
reporting and sampling, not quarantine files.

## Complete binary multi-function input

`scripts/data/extract_dart_aot_user_function_bundle.py` disassembles the root
and every producer-attested same-file user helper/closure without reading Dart
source. `scripts/preprocessing/build_multifunction_binary_compact.py` converts
that source-neutral bundle into one frozen-v1-compatible graph: F0 is marked
`fn @SELF`, helpers are marked `fn @U<n>`, local block IDs and CFG edges receive
global offsets, and external runtime targets stay in the separate `@X<n>`
namespace with their audited dictionary in the binary-enrichment prefix.
Completeness uses the extractor's private keyed source-symbol attestation:
the builder requires its per-row name-free binding, ordered `T<n>`/`AF<n>`
alias inventories, and recursive-call accounting. Reports state this use
explicitly. Raw attested names and the private key are never serialized into
student or API inputs.

`scripts/preprocessing/sanitize_compact_targets.py` first produces the
role-sealed target views. The builder accepts only its all-1,580
`sequence_imitation_all_train` view and its full held-out dev175 measure view;
their seals and hashes are mandatory inputs. It then emits disjoint,
role-sealed multi-function compact rows plus F2 prompt artifacts for both.

The separately sealed executable-reward train1578 membership is joined
downstream only for RS-SFT and VeRPO. It must not shrink the 1,580-row Qwen
imitation/F2 representation. The builder aborts the entire build on the wrong
sanitized view, an extractor omission, hash drift, codec/F2 round-trip failure,
9K student overflow, or 12K API-prompt overflow; it never truncates or drops a
row.

## Why v2.1 held long rows out—and why v2.3 changes that

The v2.1 split was intended to isolate the representation experiment from the
known one-shot length cliff. That was defensible for a narrowly scoped ablation,
but inappropriate as the default training policy when approximately half of the
approved corpus occupies the failing regime. The observed cliff is evidence that
the architecture and curriculum must change; it is not a reason to discard the
very supervision needed to learn that regime.

v2.3 therefore makes the short/bridge/long thresholds curriculum boundaries:

| Parsed instructions | Role in v2.3 |
|---:|---|
| ≤150 | direct SFT, replay in hierarchical stage, code recovery |
| 151–199 | direct SFT, region planning, plan-conditioned code, 2× recovery |
| ≥200 | direct SFT, region planning, plan-conditioned code, 3× recovery |

The repeat factors are configurable.

## All-length architecture

All SFT, inference, and RL stages instantiate the same structural configuration:

- `GRAPH_REGION_COMPRESSION=linear_residual`
- `GRAPH_REGION_MAX_BLOCKS=8`
- `GRAPH_BLOCK_POOLING=multi_query`
- `GRAPH_BLOCK_VECTORS_PER_BLOCK=4`
- `GRAPH_QWEN_PREFIX_TOKENS=64`
- dynamic prefix allocation with at least 16 active slots
- prefix RMS matching
- typed reverse CFG/DFG edges
- sinusoidal block positions
- RoBERTa-compatible block token positions
- Qwen attention+MLP LoRA
- decoder and local-encoder gradient checkpointing

The architecture is created before Stage 1 and remains unchanged. Region or
prefix parameters are not inserted halfway through the run.

A legacy Regions16 checkpoint will generally not contain these new structural
parameters. It must not be loaded with permissive `strict=False` semantics. Use
a checkpoint whose provenance matches the v2.3 architecture, or begin the new
all-length run from the pretrained base. The existing attention-only to
attention+MLP LoRA migration remains narrow and does not pretend to migrate an
old 16-prefix/CLS/no-region checkpoint into the new hierarchy.

## Phase 0: all-length approval

`prepare_hybrid_training_data_antigravity.py` still performs the v2.1 controls:

- exact and alpha-structural frozen-evaluation overlap blocking;
- within-pool deduplication;
- neutral typed `fn0` contracts;
- feedback/hidden-acceptance test partitioning;
- deterministic assembly facts;
- full, feedback, and hidden reference replay;
- immutable Phase-0 provenance.

It now proportionally apportions the development set across all fine-grained
length bins using deterministic Hamilton-style allocation. The remaining rows
are emitted as:

- `approved_all_length_train.jsonl`
- `approved_short_le150.jsonl`
- `approved_bridge_151_199.jsonl`
- `approved_long_ge200.jsonl`
- `approved_all_length_dev.jsonl`

The three stratum files are subsets of the all-length training file.

## Hierarchical dataset construction

`build_hierarchical_long_training_antigravity.py` consumes Phase-0-approved
rows and creates four training datasets.

### 1. Direct all-length SFT

Every row maps the complete binary representation to its complete verified Dart
function.

### 2. Region-plan multitask SFT

Each bridge/long function creates three examples:

- `region_plan`: binary/graph → compact deterministic CFG-region JSON;
- `code_from_region_plan`: binary/graph + region plan → complete Dart;
- `code`: binary/graph → complete Dart.

The plan is computed from the existing CFG, edge types, and bounded maximal
straight-line regions. It records region order, member blocks, predecessor and
successor regions, inter-region edge types, calls, constants, strings,
comparisons, conditional branches, and returns.

The plan is an auxiliary target. It is not obtained from a frontier model and
cannot leak reference source. The final deployed task remains direct code
generation.

### 3. Matched direct-SFT control

For every hierarchical multitask example, the builder creates one direct-code
control example from the same source task. The hierarchical and direct-control
stages therefore have identical example counts, starting checkpoint,
architecture, learning rate, epoch count, and recovery stage.

This is a matched-step/example control, not an assertion of exactly equal FLOPs;
plan and code targets can have different token lengths.

### 4. Code-only recovery

The final recovery dataset contains only `training_task=code` examples. It
removes dependence on teacher-forced plans and oversamples the hard strata by
default:

- short: 1×
- bridge: 2×
- long: 3×

## Long-function causal/performance gate

Before frontier-teacher harvesting or RL, v2.3 evaluates the recovered
hierarchical model against the recovered matched direct-SFT control on the
**≥200-instruction development subset only**.

The gate uses the same statistical controls introduced in v2.1:

- free-running pass@k;
- correct versus permuted versus null graph arms;
- exact one-sided paired sign/McNemar test;
- paired task bootstrap confidence bound;
- minimum discordant pairs;
- prompt-stream and generation-provenance equality;
- checkpoint architecture validation.

The candidate must both retain causal graph use and beat the direct-code control
on the long stratum. The default practical improvement floor is zero because the
paired statistical test and positive confidence bound remain mandatory. A
positive floor can be pre-registered with:

```bash
--hierarchical_long_min_improvement_pp 3.0
```

The default minimum long-development size is 64. With about 792 long rows and a
10% proportional dev split, the expected long gate contains roughly 79 tasks.

## No silent truncation, impossible targets, or global 3k padding

The trainer first tokenizes every complete target and prompt with
`truncation=False` and `padding=False`.

It then:

1. aborts when a complete target exceeds `GRAPH_MAX_TARGET_LENGTH`;
2. aborts when a complete prompt exceeds the configured prompt ceiling;
3. dynamically pads labels and prompts only to the current batch maximum;
4. requires the target ceiling to fit every effective gate, rollout, and GRPO
   generation budget.

At batch size 1, a 200-token target remains 200 tokens rather than being padded
to 3,072. At larger batch sizes, labels are padded with `-100` and decoder
prompts are right-padded with the decoder pad token and a zero attention mask.
The shared cumulative position-id implementation remains active, so dynamic
right padding does not reintroduce the earlier mid-sequence RoPE mismatch.

The all-length defaults are now aligned:

- target budget: 3,072 tokens;
- prompt budget: 3,072 tokens;
- effective gate/rollout/GRPO generation budget: at least 3,072 new tokens.

The runner exits before Phase 0 if the admitted target ceiling is greater than
any generation path. `GRAPH_ALLOW_TARGET_TRUNCATION=1` remains an unsafe
reproduction-only override and must not be used for confirmatory training.

## Token-length preflight

Before any GPU stage,
`scripts/evaluation/report_token_lengths_antigravity.py` measures:

- target and prompt min/mean/p50/p75/p90/p95/p99/max;
- counts and fractions above historical 768, 1,024, 2,048, and 3,072 limits;
- distributions by short/bridge/long stratum;
- distributions by training task and input dataset;
- deduplicated complete-code target distribution;
- the largest 25 prompts and targets;
- every target or prompt that exceeds the configured budgets.

The runner produces two reports:

```text
00_phase0/raw_token_length_distribution.json
00_phase0/token_length_distribution.json
```

The raw report runs before Phase 0 and is descriptive, so rows that Phase 0 will
later reject do not block preprocessing. The second report measures the exact
direct, hierarchical, recovery, and development views that will reach training;
any overflow there is fail-closed. The deduplicated complete-code section
quantifies how much of the original target population exceeded the historical
768-token ceiling without being inflated by recovery replay.

## Complete curriculum

1. Measure the raw-corpus target and prompt token distribution.
2. Run Phase-0 all-length approval and full reward audit.
3. Build direct, hierarchical, matched-control, and recovery datasets.
4. Run the fail-closed token-budget report over the exact training views.
5. Prepare and audit the neutral held-out functional evaluation set.
6. Run the compatible-checkpoint probe, or probe immediately after Stage 1.
7. Train graph-only direct all-length SFT.
8. Train and evaluate the matched direct-code long-function control **even if
   the subsequent graph-causality gate fails**.
9. Run the free-running Stage-1 graph-causality gate.
10. Only after that gate passes, train region-plan multitask SFT.
11. Run code-only all-length recovery.
12. Evaluate hierarchy versus the matched direct control on the ≥200 stratum.
13. Generate all-length k=16 rollouts.
14. Run redacted frontier diagnosis and repair.
15. Replay hidden acceptance, complete harnesses, and assembly-facts checks.
16. Train exact 50/50 gold/verified RS-SFT with a matched gold-only control.
17. Apply the statistical RS-SFT kill switch.
18. Optionally run verified-only VeRPO and the final gate.

VeRPO's LLM judge is enabled by default and fails closed. Set
`DEEPSEEK_API_KEY` (or the provider-neutral `VERPO_JUDGE_API_KEY`) before the
GRPO stage. A successful run writes `verpo_judge_telemetry.json`; zero
successful judgements cannot produce the final checkpoint.

For DeepSeek models that expose a thinking toggle, reasoning is enabled by
default with a 12,288-token initial completion budget. Empty or
length-truncated completions are retried with successively larger token
budgets, bounded at 32,768 tokens by default, and still fail closed if no
complete final answer is returned.

Rollouts use `top_p=1.0` and explicitly disable top-k truncation. This is a
correctness constraint: VeRPO scores the full temperature-scaled policy
softmax, so sampling from a truncated distribution would bias its policy
gradient. Intermediate `checkpoint-optstep-*` directories are weight-only
warm-start snapshots written after optimizer updates; the final output root is
the canonical checkpoint.

Dart test success requires both process exit code zero and a unique 256-bit
completion marker emitted by a wrapper only after the trusted test `main`
returns. The wrapper preserves synchronous and asynchronous harnesses.
Marker-introspection/native-process capabilities (`dart:io`, `dart:ffi`, and
`dart:mirrors`) are outside this pure-function verifier contract and fail
closed. `dart:isolate` remains usable, but isolate termination cannot pass
without the wrapper's marker. Consequently, `exit(0)`, `Isolate.exit`, and
marker-source spoofing cannot receive RS-SFT certification or VeRPO reward.

Legacy text checkpoints that predate explicit recording of architecture
defaults can be migrated only when their recorded trainer SHA-256 is in the
runner's fixed allowlist and the same trainer bytes remain installed. In that
case the source-code defaults are made explicit in the new run environment;
unknown or changed trainer provenance remains a hard error.

V2 in-context repair is explicit. First build a visible-test-only repair file:

```bash
python -m scripts.training.build_verpo_repair_dataset_antigravity \
  --rows data/prepared/phase0_approved.jsonl \
  --predictions artifacts/rollouts/deployment_current.json \
  --out artifacts/verpo_repair.jsonl \
  --report artifacts/verpo_repair.report.json
```

Then pass both `--verpo_repair` and
`--verpo_repair_file artifacts/verpo_repair.jsonl`. The runner rejects a repair
flag without a validated file, hidden harness fields, and low prediction-join
coverage.

The direct control is intentionally placed before the causal gate. It produces
an absolute ≥200 pass@k result even if the latent graph-prefix experiment stops,
allowing the no-truncation/direct-SFT result to be reported independently. It is
still a control within the v2.3 architecture; a claim that truncation alone
caused the entire historical cliff would require a separately matched
historical-architecture truncation ablation.

## Asynchronous direct-compact VeRPO rescue

The all-zero-group rescue experiment is a separate, off-policy stage. It does
not run inside the on-policy VeRPO optimizer. Its four matched arms are plain
resampling, compiler/test-diagnostic-conditioned repair, grounded judge
diagnosis with repair steps removed, and that same diagnosis with repair
steps. The diagnosis-only arm is derived from the exact same provider response
as diagnosis-plus-steps: there is one paid judge call per dead task group, not
one call per candidate or arm.

Use the phased launcher for production:

```bash
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json preflight
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json status
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json plan
```

At this point, terminate the GPU instance. The launcher deliberately cannot
cross into the paid API phase until that release is acknowledged:

```bash
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json gpu-release \
  --confirmation "terminated instance gpu-1234"

# Run these on a CPU/API host. The API key stays in the environment.
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json diagnose
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json materialize
```

Provision a GPU again only for student repair generation:

```bash
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json generate

# GPU generation is finished; the remaining phases are CPU/Dart work.
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json bundle
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json score
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json transfer
```

`diagnose` uses a hash-chained append-only journal. A completed paid call is
reused exactly after a restart. A journal ending after `diagnosis_started` but
before a terminal event fails closed because the provider offers no
idempotency guarantee. Rescue diagnosis permits exactly one provider attempt
per sealed group: both SDK transport retries and completion retries are fixed
at zero. Empty, rejected, malformed, or length-truncated responses are recorded
as terminal ITT failures rather than billed again. Rejected receipts advance
the receipt chain even when they do not add an accepted provider response ID.
Repair inference has its own per-plan exact-resume journal.

`resume` runs only the next incomplete phase and will not cross a cost boundary
without an explicit acknowledgement:

```bash
# CPU-only, non-paid next phase
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json resume

# Required when the next phase is diagnosis
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json resume --allow-paid-api

# Required when the next phase is repair generation
python -m scripts.training.launch_direct_compact_verpo_rescue \
  --config configs/verpo_rescue_launch.json resume --allow-gpu
```

The bundle phase fails if any generatable arm/rank run is absent.
`--allow-missing` is only for an explicitly incomplete ITT analysis; absent
runs remain failures and never shrink the denominator.

The launch config schema is
`direct-compact-verpo-rescue-launch-config-v1`. Every entry under `inputs` is
an exact `{path, sha256}` pin. Required names are:

```text
base_inference, base_provenance, rollout, rollout_seal, f2, f2_manifest,
feedback_view_report, private_holdback, contract, dataset, alignment,
codebook, codec_artifact, tokenizer_json, source_overlay
```

Copy `configs/verpo_rescue_launch.example.json` and replace every path, digest,
immutable model revision, and endpoint placeholder before preflight.

Preflight checks every byte hash; binds base inference to its provenance;
requires rollout, F2, F2 manifest, and private reward holdback to be the exact
outputs of the pinned `verpo-train-feedback-view-v1` report; and checks the
contract, codebook, codec, tokenizer, source overlay, and prompt mode against
the base student provenance. It rejects a final-measure or `measure_175` view
before creating the run root or permitting API/GPU work, and requires the
rollout, F2, private development holdback, and base inference task IDs to be a
single sealed fit-only view. Provider model and endpoint settings are pinned,
but API keys must never appear in the config or persisted artifacts.

The fixed phase outputs below the configured `output_root` are:

```text
00_preflight/run_contract.json
01_plan/pilot_plan.json
02_gpu_release/released.json
03_diagnose/diagnoses.json
03_diagnose/diagnoses.journal.jsonl
04_materialized/
05_repairs/
06_bundle/repair_bundle.json
07_score/score_report.json
07_score/rs_sft_targets.jsonl
07_score/partial_preferences.jsonl
08_transfer/build_report.json
```

The scorer selects one repair using visible tests only, then evaluates that
same repair and its originating base attempt on the sealed development reward
holdback. Only full visible-plus-holdback student passes enter RS-SFT. Partial
preferences require improvement on both views, remain separate and off-policy,
and never enter the on-policy VeRPO update. Repairs byte-identical to the
original gold are excluded before the 400-target coverage and 50/50 transfer
calculations; a genuine non-gold repair is preferred and an all-gold-only
transfer fails closed. The score report seals the predeclared task-paired
McNemar contrast between `plain_resample` and `diagnosis_and_steps`, including
the four-cell discordance table, continuity-corrected statistic, and exact
two-sided binomial p-value. The final 175-task evaluation holdout is not an
input to any rescue phase.

## Installation

```bash
unzip hybrid_training_patch_v2_3_20260717.zip
cd hybrid_training_patch_v2_3

python apply_hybrid_patch.py \
  --project-root /path/to/antigravity
```

## Example run

```bash
cd /path/to/antigravity

python -m scripts.training.run_hybrid_curriculum_antigravity \
  --project_root . \
  --output_root artifacts/hybrid_v2_3 \
  --train_file data/prepared/all_gold.jsonl \
  --eval_file data/testing/frozen_154.jsonl \
  --functional_eval_file data/testing/fresh_eval.jsonl \
  --teacher_model gpt-5.6 \
  --max_train_instructions 150 \
  --max_bridge_instructions 199 \
  --min_long_rows 64 \
  --dev_fraction 0.10 \
  --min_long_gate_rows 64 \
  --long_target_max_tokens 3072 \
  --long_prompt_max_tokens 3072 \
  --long_generation_max_tokens 3072
```

For a new hierarchy, omit legacy `--initial_checkpoint` and
`--stage1_checkpoint` unless their recorded graph architecture exactly matches
v2.3.

## Important interpretation

v2.3 is an implementation for training the ≥200 regime, not proof that it is
solved. A credible result requires separate reporting for:

- ≤150;
- 151–199;
- 200–299;
- ≥300 instructions.

The key confirmatory comparison is hierarchical recovery versus its matched
direct-SFT control on the ≥200 development/evaluation stratum. If that paired
gate fails, the conclusion is that the implemented hierarchy did not overcome
the cliff—not that the long rows were absent from training.

## Validation performed in this archive

- every Python file parses and compiles;
- 66 focused hybrid/RS-SFT/VeRPO/evaluator tests pass, and the complete package
  suite passes 107/107;
- tests cover all-length Phase-0 routing, deterministic hierarchical dataset
  generation, dynamic batch padding, generation-budget consistency,
  token-distribution preflight wiring, matched-control cardinality, long-row
  recovery oversampling, statistical gates, checkpoint contracts, teacher
  leakage controls, completion-attested Dart execution, and the corrected
  runner ordering;
- installer copy, backup, and SHA-256 verification are tested separately.

Not executed in this environment:

- Qwen3-8B/GraphCodeBERT GPU training;
- real Dart replay over the pending prepared corpus;
- live frontier API calls;
- empirical ≥200 pass@k improvement.
