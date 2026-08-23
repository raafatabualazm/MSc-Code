# Leakage-Free 20-Hour Runbook (RTX PRO 6000 96 GB)

## Decision

The historical local 154-task functional results must be rerun. Do not rerun
all 17 arms. Do not train on either evaluation corpus:

- `data/testing/compile-test2_cfg.jsonl` (126 standalone legacy tasks)
- `data/testing/grpo_data_graphv2.jsonl` (154 HumanEval-Dart tasks)

Those files remain evaluation-only. Combining them into training would make
the new pass@k and compile@k numbers in-sample again.

The confirmatory study has three layers:

1. **Causal architecture controls:** train on 770 recompiled, deduplicated,
   validation-disjoint assembly/source pairs, validate on 83 pairs, and
   evaluate on all 154 untouched tasks.
2. **Expanded-data arm:** train one CFG+DFG model on the 770-row original
   train split plus the 1,380-row synthetic train split.
3. **Optional GRPO:** start from the expanded CFG+DFG checkpoint, optimize on a
   256-row synthetic reward subset, and evaluate on all 154 untouched tasks.

## Prepared Data

Preflight first removes three train programs that near-duplicate validation,
then rebuilds CFG+DFG with `build_graph_v2_jsonl.py`. It preserves all 154
benchmark rows and all 1,726 synthetic rows, and emits immutable hashes plus
rejection manifests. The supervised source files were recompiled from source
with complete GDB capture; exact source duplicates were removed. They yield:

| File | Rows | Role |
|---|---:|---|
| `data/datasets/dart_all_graphv2_train.jsonl` | 770 | confirmatory SFT train |
| `data/datasets/dart_all_graphv2_validation.jsonl` | 83 | confirmatory validation |
| `data/testing/grpo_data_graphv2.jsonl` | 154 | evaluation only |
| `data/datasets/synthetic_pool_graphv2.jsonl` | 1,726 | synthetic source master |

`prepare_synthetic_reward_pool.py` then finds and repairs 172 rows whose tests
called the reference function directly instead of the `candidate` alias. One
row with only two assertions was rejected. The clean pool is:

| File | Rows | Role |
|---|---:|---|
| `data/datasets/synthetic_pool_reward_clean_graphv2.jsonl` | 1,725 | canonical master |
| `data/datasets/synthetic_reward_graphv2_splits/synthetic_reward_graphv2_train.jsonl` | 1,380 | expanded SFT |
| `data/datasets/synthetic_reward_graphv2_splits/synthetic_reward_graphv2_validation.jsonl` | 172 | expanded validation |
| `data/datasets/synthetic_reward_graphv2_splits/synthetic_reward_graphv2_test.jsonl` | 173 | synthetic held-out diagnostic |
| `data/datasets/synthetic_reward_graphv2_splits/synthetic_reward_graphv2_grpo256.jsonl` | 256 | optional GRPO training |

The synthetic split is grouped by normalized source so duplicate
implementations cannot cross split boundaries. The original 770/83 split also
passes exact and token-normalized near-overlap audits.

## One Python Entry Point

`upload_clean_study_filelist.txt` is the exact 58-file transfer manifest for a
fresh machine. From Git Bash/WSL, a hashable bundle can be built with:

```bash
tar czf clean_study_bundle_graphv2.tar.gz -T upload_clean_study_filelist.txt
sha256sum clean_study_bundle_graphv2.tar.gz
```

On a fresh CUDA 12.8 / PyTorch 2.8 machine, recreate the captured environment
and verify the two non-Python tools before reserving GPU time:

```bash
python -m pip install -r requirements.txt
dart --version
gdb --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

`requirements.txt` is the captured working environment; replace any
machine-local wheel paths if that exact path is unavailable on the new host.
Authenticate once with `hf auth login`; do not put the token in a logged shell
command. Pass the destination repository directly to the driver, then run from
the repository root inside `tmux` or `screen`:

```bash
tmux new -s antigravity-clean
python scripts/run_leakage_free_study.py --phase plan
```

The script is resumable. It skips a stage only when the result summary and both
prediction provenance sidecars exist, state `scoring_tests_visible_to_policy =
false`, use prompt schema `antigravity-v2-no-test-hints`, and match the seed.
Logs and state are stored under `logs/leakage_free_graphv2/`.

## Phase 1: CPU Preflight

```bash
python scripts/run_leakage_free_study.py \
  --phase preflight \
  --execute \
  --full_reward_audit \
  --reward_workers 32
```

This reconstructs the final 770-row train split, rebuilds CFG+DFG from the four
raw files, and verifies the graph-v2 schema, assembly/extractor hashes, entry nodes, edge
types/ranges, duplicate edges, source-line exclusion, DFG counts, and every
block against the pinned GraphCodeBERT tokenizer. It then runs protocol and
GRPO self-checks, audits all 154 prompts for test leakage, checks both training
SFT train/validation split and both training pools for exact/near overlap,
rebuilds synthetic splits, validates
reward compatibility, and optionally executes all 1,725 references.

Do not reuse any historical graph checkpoint or prediction pool. Graph-v2
changes CFG parsing, DFG semantics, edge-type IDs, reverse relations, block
positions, dynamic prefix resampling, per-token gates, and RMS matching;
compatibility would be scientifically invalid even if tensor loading succeeds
with `strict=False`.

**Stop if any preflight command fails.** Do not start paid GPU work.

## Phase 2: One Guarded GPU Reservation

```bash
python scripts/run_leakage_free_study.py \
  --phase gpu \
  --execute \
  --seeds 42,43,44 \
  --budget_hours 20 \
  --metric_workers 64 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

This single invocation keeps one continuous 20-hour wall-clock budget across
the causal core, expanded SFT, and reward preflight. It stops before starting
a stage that would exceed the reservation and resumes provenance-valid stages
on the next machine.

The default priority profile runs the untuned reference, text control,
signature-only base control, and graph controls at seed 42. It then repeats
the causal graph comparison with independent training/generation seeds 43 and
44:

- signature-only untuned Qwen base (seed 42; benchmark-recognition floor)
- text-only cleaned assembly (seed 42)
- identical block encoder and prefix with no edges (seeds 42, 43, 44)
- CFG-only edges (seeds 42, 43, 44)
- CFG+DFG edges (seeds 42, 43, 44)

Every graph arm uses at most 20 instructions per lossless fallthrough-split
block and a maximum 64-slot prefix with 4-64 active slots selected
from graph size, distinct forward/reverse edge relations, canonical block-order
positions, one learned gate per prefix slot, and prefix/token RMS matching.
Shuffled edges and CFG+DFG-plus-assembly run on seed 42 as auxiliary diagnostics.
The latter tests whether raw assembly causes the decoder to ignore the graph
channel. No-GINE pooling is retained in the runner but deferred to
`--full_matrix`; that later mode also repeats raw-base, signature-only, text,
and auxiliary controls for every seed.

The runner generates the 154-task, 10-candidate pool once and reuses that exact
pool for JIT compile@k and pass@k. Older runs generated compile and pass pools
separately, roughly doubling GPU time and making paired interpretation harder.

## Individual Resume: Expanded SFT

Run only after the core arms complete:

```bash
python scripts/run_leakage_free_study.py \
  --phase expanded \
  --execute \
  --seeds 42 \
  --budget_hours 20 \
  --metric_workers 64 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

This is a separate data-scale result, not a replacement for the causal
ablation. It trains CFG+DFG from the pinned base on original plus synthetic
training rows, while the 154-task benchmark remains unseen.

## Individual Resume: GRPO Reward Preflight

```bash
python scripts/run_leakage_free_study.py \
  --phase reward-preflight \
  --execute \
  --seeds 42 \
  --reward_preflight_batches 8 \
  --min_signal_group_rate 0.05 \
  --reward_workers 32 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

This generates and executes 8 groups without calling `backward()` or
`optimizer.step()`. It writes `reward_preflight.json` beside the temporary
preflight output. GRPO is approved only if at least 5% of prompt groups contain
mixed binary outcomes. If the signal rate is lower, binary GRPO cannot learn
reliably from this checkpoint and dataset; stop and use rejection-sampling SFT
or frontier-teacher distillation instead.

## Phase 3: Optional GRPO (Usually a Second Reservation)

```bash
python scripts/run_leakage_free_study.py \
  --phase grpo \
  --execute \
  --seeds 42 \
  --budget_hours 6 \
  --reward_workers 32 \
  --metric_workers 64 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

The confirmatory GRPO configuration is deliberately plain:

- exact full-harness binary reward: pass `+1`, fail `-1`
- group size 16, scoring chunk 4
- temperature 0.7 with `top_p = 1.0`, so rollout and scored distributions match
- one epoch, learning rate `5e-7`
- sequence-normalized loss
- mean-centered group advantages
- no overlap reward, partial reward, duplicate/unique bonus, entropy term, or
  legacy top-K smoothing
- no KL term, because adapter disabling exposes the pre-SFT base rather than a
  valid frozen copy of the SFT starting policy
- overlong samples excluded from policy updates
- 768-token rollout/evaluation budgets, avoiding the old train/eval length mismatch
- frozen local block encoder, with GNN, projection, dynamic prefix adapter,
  per-slot gates, and policy LoRA parameters trainable

Binary reward uses one evaluator-aligned `dart run` per completion. It does not
run a second compile subprocess; a successful harness run necessarily compiled,
while compile status for a failed run is reported as unknown rather than guessed.

The current one-update loop is group-relative REINFORCE, not full PPO: the old
log-probability is detached from the same scoring pass, so the likelihood ratio
is one and clipping is inert. The paper and thesis must use this precise name
unless a genuine previous-policy/reference implementation is added.

## Recommended 20-Hour Order

| Priority | Work | Approx. time |
|---:|---|---:|
| 1 | CPU preflight and full reward parity | 0.4-1.0 h |
| 2 | Seed-42 full controls + three-seed causal graph comparison | 17.2 h |
| 3 | Expanded CFG+DFG SFT | 2.0 h |
| 4 | GRPO reward preflight | 0.5 h |
| 5 | One synthetic-trained GRPO run, only if approved | a later 4 h reservation |

The paid `--phase gpu` estimate is 19.7 hours before the optional GRPO run;
CPU preflight is run separately. Historical generation pools varied from about
one to two and a half hours, so the 20-hour figure is a priority budget, not a
completion guarantee. Actual time is recorded in the runbook logs; the Python
driver stops before starting a stage whose estimate exceeds the requested
budget and resumes cleanly on the next machine. GRPO is deliberately last and
will usually require a second reservation after the leakage-free causal result
is secured.

## Synthetic Generation

The generator was not lost:

- `generate_synthetic_tasks.py`
- `generate_synthetic_tasks_parallel.py`

The parallel generator now defaults to a compact HumanEval-like style without
copying HumanEval names, wording, tests, or tasks. It requires 8-12 unique
`expect(candidate(...), ...)` assertions and compiles/disassembles candidates
outside the acceptance lock, so `--parallel` can use multiple CPU cores.

Generate a **new** pool only after the clean 1,725-row experiment is complete;
otherwise data changes and architecture changes are confounded. Example:

```bash
python generate_synthetic_tasks_parallel.py \
  --profile humaneval \
  --eval-jsonl data/testing/grpo_data_graphv2.jsonl \
  --out data/datasets/synthetic_humaneval_v2_raw.jsonl \
  --providers or_gpt,or_sonnet,or_deepseek,or_qwen_max \
  --per-provider 300 \
  --parallel 8 \
  --task-prefix synth_he_v2
```

Then canonicalize, split, validate all references, build CFG if needed, and
record hashes before training. A stronger v3 generator should use an
independent test-author model and mutation-score gate; simply increasing row
count is less valuable than adding tests that kill plausible wrong solutions.

## Reporting Rules

- Main local results: three-seed mean and standard deviation plus task-and-seed
  intervals on the untouched 154 tasks.
- DFG claim: compare CFG-only versus CFG+DFG under the same block/prefix path.
- Topology claim: compare no-edge versus CFG and shuffled-edge versus CFG+DFG;
  do not substitute the historical graph-versus-text comparison.
- Channel-competition claim: compare graph-only CFG+DFG against the seed-42
  CFG+DFG-plus-cleaned-assembly auxiliary arm.
- GRPO claim: compare expanded SFT versus synthetic-trained GRPO on the same
  untouched 154 tasks.
- Never merge the 126 or 154 rows into training.
- Keep all historical local pass@k, reranking, and 17-arm union tables labelled
  contaminated and graph-invalid; do not combine them with the graph-v2 rerun.
- Use `GRAPH_CONSTRUCTION_AUDIT.md` as the fixed/open claim boundary. In
  particular, do not describe the block-level may-flow DFG as SSA, heap-aware,
  SIMD-complete, instruction-level, or interprocedural.
