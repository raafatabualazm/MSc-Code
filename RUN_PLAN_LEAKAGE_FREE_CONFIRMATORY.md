# Leakage-Free Confirmatory Run Plan

This plan supersedes the historical local functional tables. Do not combine
new results with old candidate pools.

## Fixed Inputs

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `data/datasets/dart_all_cfg_clean.jsonl` | 1,195 | `175e67ac2a343723415f13ab2ed71e79b1b4d0750f578ffd4dd2762747024281` |
| SFT train split | 1,081 | `e3137a9c70570aff43aed0d7dc59b342e199ae6f7b4555d9704c39af3fcbdc0e` |
| SFT validation split | 114 | `7b0a2880140b77c1b99dbebc712e68c3a179d21fbe63e6288d026fe13721630a` |
| Functional benchmark | 154 | `209adb27e762f005652f9f78b1ac2f49c1eb0b366f2fec45aa41ba77e48b1c0f` |
| GRPO train half | 77 | `3a9644cff0c4fc852d8a7007a0806fa2a1c16a9659ff1bec285ff705d6fef673` |
| GRPO held-out half | 77 | `4511a3c342fad5a03f49ffd1802a02d79b8eea06da8b4c7db1f5bc2b7a7490cb` |

Pinned base revisions:

- Qwen decoder: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`
- GraphCodeBERT: `microsoft/graphcodebert-base@2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`

The SFT split groups identical normalized sources; its 188 duplicate rows never
cross train/validation. The overlap audit reports zero exact and zero normalized
token-7-gram matches at Jaccard 0.8 between SFT and the 154-task benchmark.

## Preflight

```bash
python -m unittest scripts.evaluation.test_protocol_integrity_antigravity

python scripts/evaluation/audit_prompt_integrity_antigravity.py \
  --dataset data/testing/grpo_data_cfg.jsonl \
  --output results/protocol_audit/grpo_data_cfg_prompt_integrity.json
```

The audit must report `rows_leaking_scoring_tests: 0` and prompt schema
`antigravity-v2-no-test-hints`.

## Core Study

Run three seeds for the untuned reference and six SFT arms:

```bash
bash scripts/run_clean_confirmatory_x86.sh
```

For a one-seed smoke test:

```bash
SEEDS="42" RUN_GRPO=0 bash scripts/run_clean_confirmatory_x86.sh
```

For only the causal topology controls:

```bash
SEEDS="42 43 44" \
ARMS="prefix_no_edges prefix_shuffled prefix_cfg prefix_cfg_dfg prefix_no_gine" \
RUN_REFERENCE=0 RUN_GRPO=0 \
bash scripts/run_clean_confirmatory_x86.sh
```

The controlled graph arms all use the same GraphCodeBERT block encoder, 16-token
prefix resampler, graph-only decoder cue, training rows, prompt budget, LoRA
configuration, and seed. They differ only as follows:

| Arm | Edge path | GINE |
|---|---|---|
| `prefix_no_edges` | no edges | active node transforms |
| `prefix_shuffled` | deterministic target permutation | active |
| `prefix_cfg` | CFG edges only | active |
| `prefix_cfg_dfg` | CFG + DFG edges | active |
| `prefix_no_gine` | CFG + DFG retained | bypassed |

The separate `text` arm is a representation baseline, not a topology-isolating
control, because it changes the decoder input path.

## GRPO

The driver optionally starts binary GRPO from each seed's clean CFG+DFG SFT
checkpoint. GRPO sees tests only through the external reward process. It trains
on the 77-row train file and generates its reported metrics only on the 77-row
held-out file. Never report its full-154 in-sample aggregate as generalization.

For GRPO only after clean G3 checkpoints exist:

```bash
SEEDS="42 43 44" ARMS="" RUN_REFERENCE=0 RUN_GRPO=1 \
bash scripts/run_clean_confirmatory_x86.sh
```

## Required Result Evidence

Every `*_predictions.json` must have a neighboring
`*_predictions.json.provenance.json` containing:

- prompt schema and prompt-stream SHA-256;
- `scoring_tests_visible_to_policy: false`;
- dataset/checkpoint/output/source SHA-256 hashes;
- exact requested and resolved model revisions;
- seed, generation settings, graph environment, runtime, Git commit, and dirty state.

Each trained artifact must contain `run_provenance.json` with the same model,
dataset, source, seed, and checkpoint evidence.

## Confirmatory Reporting

Report mean and standard deviation across seeds for pass@1/5/10 and aligned JIT
compile@1/5/10. Use task-and-seed hierarchical bootstrap intervals. For paired
coverage, compare per-seed task outcomes and report effect sizes; do not treat a
single fixed candidate-pool McNemar test as training-run uncertainty.

The historical 17-arm union is not part of this confirmatory core. Only after
clean arms exist should ensemble curves compare budgets 10, 20, 50, 100, and
the full pool against a 170-sample single-policy and temperature-mixture
baseline.
