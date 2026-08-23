# Fail-Closed Hybrid Decompiler Training Patch v2.1

This overlay supersedes v2 with the Critique-2 statistical-gate and Qwen
read-path corrections. Its go/no-go decisions
are based on **free-running executable behavior**, data-integrity checks, and
independent verifier replay—not teacher-forced NLL alone.

It is designed for the current Antigravity GraphCodeBERT/GNN/Qwen decompiler.
The patch does not claim that the continuous graph prefix is already the right
primary representation. It first tests whether the representation retains
facts and whether free-running decoding causally depends on it. The full-prompt
RS-SFT stage then uses compact mechanical binary evidence plus assembly text,
while the latent graph channel remains subject to ablation.

## What changed from v1 and v2

The review findings are implemented as code-level controls:

- Wrong-graph NLL is only an optional Stage-1 auxiliary objective. It cannot
  satisfy the stage gate.
- The full-text Stage-2 wrong-graph objective is disabled. The trainer aborts if
  it is accidentally enabled with a full assembly prompt unless an explicit
  unsafe research override is used.
- The graph-use gate is free-running: correct graph versus shape-matched
  cross-task graph versus null graph, on held-out neutral-contract tasks.
- The RS-SFT performance gate uses an automatically trained **gold-only,
  matched-step, matched-modality control**. A graph-only checkpoint or raw warm
  start is not accepted as a like-for-like baseline.
- Frozen-evaluation overlap is blocked by two source fingerprints: an exact
  neutralized-source hash and a stronger identifier-alpha-normalized structural
  hash. Alpha-renamed benchmark copies are therefore rejected.
- Phase 0 executes before GPU training and replays every accepted reference on
  feedback, hidden acceptance, and complete test harnesses.
- Frontier-teacher diagnostics redact assertion values. Test source and hidden
  acceptance outcomes are never sent to the teacher.
- Every rollout/repair is independently replayed against feedback tests, hidden
  acceptance tests, and the full harness, then checked against a deterministic
  assembly FACTS contract.
- RS-SFT training data is materialized as an exact 50/50 gold-to-verified epoch,
  with unique-task, length-bin, and bounded-oversampling kill switches.
- GRPO uses a separate verified-only anchor loader. Ordinary RL-row reference
  labels are never used as anchor targets.
- Zero-variance RL groups are resampled for a bounded number of attempts. A
  persistently dead group is skipped by default rather than converted to plain
  reference CE.
- The functional and RS-SFT gates now require paired statistical evidence:
  an exact one-sided paired sign test (exact McNemar when pass@k is binary),
  a task-paired bootstrap lower confidence bound above zero, at least eight
  discordant task pairs, and at least 96 held-out tasks by default.
- The old arbitrary 1-point graph-permutation default is removed. A positive
  practical-effect floor is accepted only as an explicit, pre-registered value
  calibrated from an external repeatability/noise study.
- Qwen decoder LoRA now targets both attention and MLP projections by default:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj`.
- Historical attention-only Qwen checkpoints may expand only during a newly
  trained Stage 1. The newly introduced MLP LoRA-B tensors must be exactly zero,
  so loading preserves the old policy before training; every unrelated missing
  trainable tensor still aborts.
- Checkpoint architecture provenance is pinned and validated across SFT,
  inference, probe, and GRPO entry points, including the Qwen LoRA target set.
- Functions above the configured short-function cap are emitted as bridge and
  long-function holdouts instead of being silently mixed into one-shot training.

## Pipeline and enforced gates

### 0. Architecture contract

Warm checkpoints in this project contain trainable tensors rather than a full
pretrained model. Loading them with `strict=False` can silently succeed under a
wrong prefix, GNN, PEFT, LoRA-target, or prompt architecture. v2.1 therefore:

1. loads the structural `graph_environment` from an adjacent
   `run_provenance.json`, or from `--architecture_env_json`;
2. imports only an allowlist of architecture settings—never output paths,
   upload settings, secrets, dataset paths, or objective knobs;
3. requires the complete warm-start architecture contract unless
   `--allow_unpinned_architecture` is explicitly supplied for a research-only
   run;
4. checks that every currently trainable tensor exists in the checkpoint and
   that every checkpoint tensor is recognized by the instantiated model;
5. records `GRAPH_QWEN_LORA_TARGETS=attention_mlp` for new Qwen runs.

For a historical attention-only Qwen LoRA checkpoint, supply it as
`--initial_checkpoint`, not `--stage1_checkpoint`. The runner infers the legacy
`attention` target set from old provenance, constructs the expanded
attention+MLP adapter, and permits only the new `gate_proj`/`up_proj`/`down_proj`
LoRA A/B tensors to be absent. Every new LoRA-B tensor is checked to be all-zero
before the warm start is accepted. This is an explicit behavior-preserving
migration, not a general partial-load escape hatch.

`GRAPH_ALLOW_PARTIAL_CHECKPOINT=1` is an additional research-only escape hatch
inside individual model entry points. It should not be used for confirmatory
experiments.

### 1. Phase-0 CPU preflight

`prepare_hybrid_training_data_antigravity.py`:

1. computes exact-neutral and alpha-structural source fingerprints;
2. rejects overlap with every `--forbidden_eval` file;
3. deduplicates within the input pool;
4. rewrites the typed contract, executable source, tests, assembly symbol, and
   CFG instruction symbols to the opaque name `fn0`, while preserving literals;
5. deterministically partitions assertions into teacher-visible
   `feedback_tests` and hidden `acceptance_tests`;
6. extracts deterministic mechanical facts from the assembly while excluding
   branch targets, frame/stack offsets, object-pool offsets, and other runtime
   layout numbers;
7. executes the reference against feedback, hidden acceptance, and complete
   harnesses;
8. emits immutable Phase-0 metadata only for fully approved rows.

The orchestrator then runs the independent reward audit over **all** approved
rows before any model training.

Default length routing is:

| Parsed instruction count | Destination | Used by one-shot curriculum? |
|---:|---|:---:|
| `<=150` | approved short-function pool | Yes |
| `151–199` | bridge holdout | No |
| `>=200` | long-function holdout | No |

The thresholds are configurable. This routing is a safety measure, not a claim
that the length cliff is solved.

### 2. Frozen representation probe

`probe_graph_representations_antigravity.py` extracts four representations from
a frozen checkpoint:

- local block encoder;
- post-GNN/global-attention representation;
- projected graph representation;
- final decoder prefix.

It fits held-out closed-form ridge probes for mechanical properties, compares
them with permuted-label controls, and measures projected-graph-to-prefix
retrieval through a train-fitted mapping. This distinguishes:

- facts absent at the local encoder;
- facts lost by GNN/pooling/projection;
- facts present in the final prefix but ignored by the decoder.

The probe tests representational availability. It is **not** proof that
free-running generation uses the graph.

### 3. Graph-only Stage-1 SFT

Stage 1 uses:

- a graph-only decoder prompt;
- an opaque typed `fn0` contract;
- a target containing `// FACTS_JSON: ...` followed by Dart code;
- Qwen attention+MLP LoRA by default, so the modality-read path is not capped
  at attention projections;
- optional own-graph versus wrong-graph NLL margin training.

The NLL margin is a training auxiliary and diagnostic only. A mismatch detector
can satisfy it, so it never acts as the acceptance criterion.

### 4. Functional graph-use gate

`functional_graph_gate_antigravity.py` generates from the same held-out neutral
rows under three graph-only conditions:

1. correct graph;
2. length/shape-matched, one-to-one deranged cross-task graph;
3. null final graph context.

The gate fails closed unless:

- all arms use the same row set, rendered prompt stream, checkpoint contract,
  generation settings, and expected number of candidates;
- the permutation has no self-mapped rows;
- at least 96 held-out tasks are present by default;
- correct-versus-permuted and correct-versus-null each contain at least eight
  discordant task pairs;
- the pre-registered one-sided exact paired test has `p <= 0.05`;
- the 95% task-paired bootstrap one-sided lower bound on the pass@k difference
  is strictly above zero;
- any explicitly configured practical-effect floor is also met;
- optional FACTS@k statistical and practical thresholds are met.

At the default `k=10`, `num_samples=10`, per-task pass@10 is binary, so the
exact paired test is exact McNemar on discordant task outcomes. With fractional
per-task pass@k it is the exact paired sign test. The bootstrap resamples tasks,
not generated candidates. It addresses finite held-out-task uncertainty; final
paper estimates should still repeat generation under multiple seeds.

The graph-permutation practical floor defaults to zero rather than inventing a
1-point threshold. Use `--stage1_min_permutation_drop_pp` only with a value
pre-registered from an external repeatability/noise study. Statistical evidence
remains mandatory either way.

This is the authoritative answer to “does the decoder actually read the graph?”

### 5. Verifier feedback and frontier repair

`teacher_repair_dataset_antigravity.py` has three subcommands:

- `collect`: score policy candidates against feedback tests and record compile
  status, pass ratio, and per-test pass vectors;
- `teacher`: request a strict JSON diagnosis/repair through synchronous
  Responses API calls or `/v1/responses` Batch JSONL;
- `build`: independently verify candidates and construct RS-SFT/preference data.

The teacher receives assembly, the typed contract, deterministic mechanical
facts, the failed candidate, and redacted failure categories. It receives no
raw test source, assertion input, expected value, actual value, or hidden
acceptance result.

A model rollout or teacher repair enters RS-SFT only when it:

1. passes all feedback tests;
2. passes all hidden acceptance tests;
3. passes the complete harness;
4. satisfies the typed `fn0` contract;
5. satisfies the independent assembly FACTS gate;
6. carries strict replay provenance.

### 6. Exact 50/50 RS-SFT and matched control

`build_balanced_sft_mix_antigravity.py` produces:

- an exact 50% Phase-0 gold / 50% verified-alternative epoch;
- a gold-only control file with exactly the same number of training examples.

The default kill switches require at least:

- 64 verified alternatives;
- 64 unique verified tasks;
- three represented length bins.

The complete approved gold pool is covered by default. Verified oversampling is
bounded; the builder aborts rather than silently omit gold rows or amplify a tiny
verified pool without an explicit override.

Both Stage-2 runs start from the same Stage-1 checkpoint and use the same:

- full structured-text prompt;
- graph channel;
- learning rate;
- epochs;
- number of training examples.

The only intended difference is gold-only versus 50/50 gold/verified data.
Stage 2 always sets:

```text
GRAPH_PREFIX_DEPENDENCE_WEIGHT=0
```

The subsequent neutral-evaluation kill switch requires the RS-SFT checkpoint
to improve pass@k by **at least 6 percentage points** over that matched control.
It also requires the same paired exact test, paired-bootstrap lower bound above
zero, minimum discordant-pair count, and held-out-row floor. Thus two flipped
tasks on a tiny set cannot satisfy the `+6 pp` rule. An external
`--rs_sft_baseline_checkpoint` may replace the automatic control only when it is
genuinely like-for-like. A scalar `--baseline_pass_at_k` is supported but is
reported as weaker because baseline uncertainty and task pairing are absent.

### 7. Optional VeRPO/GRPO

The RL stage separates:

- `GRPO_TRAIN_FILE`: Phase-0-approved executable tasks;
- `GRPO_VERIFIED_ANCHOR_FILE`: independently verified alternative targets only.

The anchor dataloader rejects gold references and rows without feedback replay,
hidden acceptance replay, full-harness success, and FACTS success. The anchor
uses the same `FACTS_JSON + DART` target schema as SFT.

For a dead reward group, the trainer samples fresh completions up to
`GRPO_DYNAMIC_RESAMPLE_ATTEMPTS`. If the group remains zero-variance,
`GRPO_SFT_ANCHOR_ON_NO_SIGNAL=0` skips it. The verified-only CE anchor is used as
an anti-forgetting term on signal-bearing RL updates; it is not presented as
new failure-specific evidence.

### 8. Long-function track

This patch does not claim to fix the observed one-shot collapse at roughly
`>=200` parsed instructions. It produces bridge and long-function artifacts for
a separate hierarchical approach, such as CFG-region/SCC decomposition,
region-level facts or IR, dependency-aware composition, and whole-function
verification.

## Files installed into the project

| File | Purpose |
|---|---|
| `scripts/training/graph_positions.py` | Shared active-token causal positions. |
| `scripts/training/checkpoint_contract.py` | Fail-closed trainable-checkpoint validation. |
| `scripts/training/hybrid_data_controls.py` | Neutralization, dual fingerprints, test partitioning, FACTS, redaction, and provenance. |
| `scripts/training/prepare_hybrid_training_data_antigravity.py` | Phase-0 leakage/reference/length gate. |
| `scripts/training/build_balanced_sft_mix_antigravity.py` | Exact 50/50 mix plus matched gold-only control. |
| `scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py` | Facts-first SFT and guarded dependence objective. |
| `scripts/training/graph_grpo_decompiler_antigravity.py` | Dynamic resampling and separate verified-only anchor. |
| `scripts/training/teacher_repair_dataset_antigravity.py` | Safe collect/teacher/build workflow. |
| `scripts/training/run_hybrid_curriculum_antigravity.py` | Resumable fail-closed orchestration. |
| `scripts/evaluation/audit_grpo_reward_antigravity.py` | Full reference and provenance audits. |
| `scripts/evaluation/prepare_neutral_evaluation_antigravity.py` | Evaluation-only `fn0` dataset builder. |
| `scripts/evaluation/probe_graph_representations_antigravity.py` | Frozen multi-stage representation probe. |
| `scripts/evaluation/functional_graph_gate_antigravity.py` | Correct/permuted/null executable gate. |
| `scripts/evaluation/graph_inference_antigravity.py` | Matched graph derangement, null ablation, and prompt provenance. |
| `scripts/evaluation/run_sweeps_antigravity.py` | Exposes v2.1 controls to ordinary sweep runs. |

## Installation

Extract the archive and install the overlay:

```bash
cd hybrid_training_patch_v2_1
python apply_hybrid_patch.py --project-root /path/to/msc-code
cd /path/to/msc-code
python -m pip install -r /path/to/hybrid_training_patch_v2_1/requirements-hybrid.txt
```

Changed targets are backed up under:

```text
.hybrid_patch_backups/<timestamp>/
```

The installer parses every overlay Python file before copying and verifies the
SHA-256 of every installed target. It also checks the project-owned model,
data-pipeline, evaluator, and provenance prerequisites.

## Local validation

From the extracted patch directory:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m compileall -q -f .
```

The packaged suite currently contains 31 tests. It uses fake evaluators and
small synthetic records for CPU-only controls; it does not pretend to replace a
real Dart/GPU/API integration run.

## Confirmatory run

Pass every frozen evaluation pool to the Phase-0 blocklist. Warm checkpoints
must have adjacent `run_provenance.json` architecture metadata, or an explicit
provenance file must be supplied.

```bash
export OPENAI_API_KEY='...'

python -m scripts.training.run_hybrid_curriculum_antigravity \
  --project_root . \
  --output_root artifacts/hybrid_v2_1_confirmatory \
  --train_file data/training/master_train.jsonl \
  --eval_file data/evaluation/frozen_154.jsonl \
  --functional_eval_file data/evaluation/neutral_exact.jsonl \
  --frozen_eval_file data/evaluation/fresh_eval.jsonl \
  --initial_checkpoint artifacts/regions16/pytorch_model.bin \
  --probe_checkpoint artifacts/regions16/pytorch_model.bin \
  --architecture_env_json artifacts/regions16/run_provenance.json \
  --decoder_model Qwen/Qwen3-8B \
  --encoder_model microsoft/graphcodebert-base \
  --teacher_model '<frontier-model-id>' \
  --teacher_mode sync \
  --qwen_lora_targets attention_mlp \
  --max_train_instructions 150 \
  --max_bridge_instructions 199 \
  --min_gate_rows 96 \
  --gate_bootstrap_iterations 10000 \
  --gate_max_sign_test_p_value 0.05 \
  --gate_min_causal_effective_pairs 8 \
  --gate_min_permutation_ci_lower_pp 0.0 \
  --rs_sft_min_improvement_pp 6.0 \
  --min_verified_rows 64 \
  --min_verified_unique_tasks 64 \
  --min_verified_length_bins 3
```

When no `--rs_sft_baseline_checkpoint` or absolute
`--baseline_pass_at_k` is supplied, the runner automatically trains the
matched-step gold-only Stage-2 control.

Inspect the complete DAG without executing expensive stages:

```bash
python -m scripts.training.run_hybrid_curriculum_antigravity \
  --project_root . \
  --output_root artifacts/hybrid_v2_1_dry_run \
  --train_file data/training/master_train.jsonl \
  --eval_file data/evaluation/frozen_154.jsonl \
  --teacher_model '<frontier-model-id>' \
  --dry_run
```

### Batch teacher mode

Produce Batch JSONL without sending requests:

```bash
python -m scripts.training.run_hybrid_curriculum_antigravity \
  ... \
  --teacher_model '<frontier-model-id>' \
  --teacher_mode batch
```

The runner exits with status 2 after writing the request file and an
`awaiting_batch_responses` manifest. After retrieving completed output, resume
with:

```bash
--teacher_responses /path/to/completed_batch_output.jsonl
```

## Validation boundary

The packaged checks cover source parsing, orchestration, provenance, synthetic
verifier flows, and installer integrity. They do **not** establish empirical
improvement. The following still require the target repository and runtime:

- real Qwen/GraphCodeBERT checkpoint loading and GPU backward passes;
- full Dart replay over the actual training and frozen evaluation corpora;
- real frontier Responses or Batch requests;
- measured correct/permuted/null pass@10 deltas;
- measured RS-SFT improvement over the matched gold-only control;
- multi-seed confirmatory estimates beyond the single-seed task bootstrap;
- a separate hierarchical experiment for the `>=200`-instruction regime.
