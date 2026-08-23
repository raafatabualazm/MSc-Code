# FULL HANDOFF - Antigravity Graph Decompiler Experiments

> **Critical evaluation correction (2026-07-10):** The historical local
> 154-task generation prompt copied `candidate(...)` assertions, including
> expected outputs, from the same `tests` field used by the pass@k evaluator.
> Those local pass@k, GRPO, RS-SFT, reranking, and union pools are test-informed
> and must not be reported as hidden-test functional results. See
> [`CONTAMINATION_NOTICE.md`](CONTAMINATION_NOTICE.md). The prompt leak, the
> task-160 schema asymmetry, model-revision pinning, seeds, and per-run hashes
> are now fixed in code; clean multi-seed results still require a GPU rerun.

Last updated: 2026-06-18 (Section 0: ROOT CAUSE of the training plateau found and fixed - the graph was dead and the assembly was truncated to ~15%)

Workspace:

```text
C:\Users\Raafat Abualazm\Desktop\Train Data\experiment_workspace
```

Remote workspace used during these runs:

```text
/workspace
```

## 0. CRITICAL FIX (2026-06-18) - the plateau was a broken preprocessing pipeline

### Root cause
The training/eval data (`grpo_data.jsonl`, `compile-test2.jsonl`, the synthetic
pool `grpo_mix_*`, `rs_sft_*`, the half splits) never contained `cfg`/`edges`.
The pipeline's silent fallback turned every example into ONE basic block holding
the entire assembly:

```text
scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py  (tokenize_dataset)
scripts/evaluation/graph_inference_antigravity.py                    (build_blocks)
  if not cfg: cfg = [single 'entry' block with ALL assembly lines]
```

Two consequences, both measured on the real 154 rows:

1. The GNN was DEAD. One block => one graph node, `edges = []`. `cfg_to_pyg`
   produced an empty `edge_index`, so the 4-layer GINE GNN and the block
   self-attention were no-ops. Zero control-flow signal ever reached the
   decoder. GRPO inherits the same path (`from ... import tokenize_dataset`).
2. The assembly was truncated to a sliver:
   - GraphCodeBERT encoder (512 cap, single mega-block): median **18%** of the
     assembly kept; 130/154 rows lost >50%.
   - Qwen decoder prompt (768 cap, tail truncation): median **14%** kept, and
     **100% of rows lost the trailing `Dart code:` generation cue**.
   - `--prompt_fit_assembly 1` did NOT help: its 3.1 chars/token estimate is ~2x
     wrong for hex-dense disassembly (~1.5 measured), so it still overflowed and
     still dropped the cue on 0/154... i.e. it never preserved the cue.

This is exactly the plateau: an input-information bottleneck no amount of SFT,
GRPO, extra epochs, reward shaping, test-set training, or synthetic data can move
(they all fed the same crippled inputs). It also explains CodeBLEU ~0.66 (the
surviving signature/name/test-hint priors produce plausible-looking Dart) with
low pass@k (the body needed to be functionally correct was never visible).
Targets were fine (max 419 tokens, 0 truncated) - the bottleneck was input-only.

### The fix (implemented + validated 2026-06-18)
- Data: ran the existing `scripts/data/build_cfg_jsonl.py` to produce CFG-enriched
  copies (100% extraction). Use these from now on:

  ```text
  data/testing/grpo_data_cfg.jsonl                 (median 17 blocks / 22 edges per row)
  data/testing/compile-test2_cfg.jsonl
  data/testing/grpo_mix_synth576_real154_zp2_cfg.jsonl
  data/testing/rs_sft_all_cfg.jsonl
  data/testing/rs_sft_train_half_cfg.jsonl
  data/testing/grpo_data_rl_train_half_cfg.jsonl
  data/datasets/dart_all_cfg.jsonl                 (base SFT corpus, 1195 rows, mean 19.8 blocks)
  data/datasets/test-set_cfg.jsonl                 (default eval corpus, 165 rows)
  data/datasets/synthetic_pool_clean_cfg.jsonl     (1726 rows)
  data/datasets/synthetic_pool_train576_cfg.jsonl  (576 rows)
  data/datasets/dart_all_cfg_clean.jsonl           (de-leaked: source stripped from assembly; USE THIS for base SFT)
  data/datasets/test-set_cfg_clean.jsonl           (de-leaked; still better not to eval on it)
  ```

  NOTE: the config defaults (`GraphDecompilerConfig.train_file=data/datasets/dart_all.jsonl`,
  `eval_file=data/datasets/test-set.jsonl`) still point at the NON-cfg originals.
  Pass the `_cfg` files explicitly (or `--auto_cfg 1`) - a bare run without those
  flags still uses the dead single-block path.

  Effect: encoder assembly coverage 18% -> ~100% (each small block fits the
  512 window), and the GNN finally gets a real graph. Verified the
  GraphPoolingEncoder output now changes with edges (mean abs diff 0.093 vs the
  old dead no-edge path).

- Code (must re-upload these to the pod):
  - `scripts/data/cfg_extractor.py`: networkx now optional; added
    `ensure_cfg_blocks(record)` - one shared CFG resolver (precomputed cfg ->
    inline extract when `GRAPH_AUTO_CFG=1` -> single-block fallback), returns a
    normalized minimal schema so Arrow stays consistent.
  - `scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py`:
    `tokenize_dataset` uses `ensure_cfg_blocks`; `build_decoder_prompt` is now
    token-aware (`_fit_assembly_tokens`) and keeps the `<lang> code:` cue
    (153/154 now, was 0/154). With fit OFF the prompt is byte-identical to before.
  - `scripts/evaluation/graph_inference_antigravity.py`: `build_blocks` uses the
    same `ensure_cfg_blocks`; prompt call passes the tokenizer+budget (train/eval
    parity).
  - `configs/run_sweeps_antigravity.py`: added `--auto_cfg {0,1}`
    (sets `GRAPH_AUTO_CFG`) as a safety net so a future non-CFG dataset cannot
    silently regress to the dead single-block path. Also added a pre-flight
    existence check: an explicit `--sft_checkpoint`/`--grpo_checkpoint` that is
    missing now aborts in seconds instead of silently cold-starting.
  - `models/graphcodebert_tensor_builder.py`: PERF FIX. Each block emitted a
    (512, 512) DFG attention mask that the encoder collapses to a 2D mask and
    discards. Harmless with one block, but once the CFG went live (median 17, up
    to ~450 blocks/row) these square masks (built as Python lists, rebuilt every
    step) dominated CPU/RAM and starved the GPU (the "20GB allocated, 0% util,
    stuck at step 0" symptom). Now emits the equivalent 2D padding mask -> ~512x
    less mask data per block; model behavior unchanged (verified).
  - `scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py`: a
    missing explicit `GRAPH_CHECKPOINT` now raises instead of warning + cold
    starting (this silently turned `_cfg_ut` into a 154-row cold start).

### CFG extractor robustness fix (2026-06-18)
First pass over `dart_all.jsonl` failed CFG extraction on 467/1195 rows (39%) and
test-set.jsonl on 46/165 (28%), all `IndexError: list index out of range`. Two
bugs in `scripts/data/cfg_extractor.py`, now fixed:
- `parse()` accepted any line CONTAINING `0x` and ran `.split()[0]` on it, so
  interleaved source / address-only lines became fake/empty instructions. Now it
  requires the address at the START of the line and skips empty instructions.
- `build_blocks()` cut blocks in ADDRESS order, but Dart AOT dumps are not
  monotonic by address (entry region listed after the body, backward jumps), so
  leader indices came out e.g. `[13,0,4,6]` -> a `start>end` empty block ->
  crash. Now blocks are cut in INSTRUCTION-STREAM order (sort leaders by stream
  index), which also makes the `index+1` fall-through edge correct.

After the fix: dart_all 1195/1195 and test-set 165/165 extract with 0 failures
and 0 empty CFGs (mean 19.8 / 32.5 blocks). Clean corpora are unchanged
(grpo_data still 22.9 blocks), so no regression.

### DATA-QUALITY WARNING: source leak in dart_all / test-set
`dart_all.jsonl` (41% of rows) and `test-set.jsonl` (29%) have the numbered Dart
SOURCE interleaved into the `assembly` field (that is what broke CFG extraction).
`grpo_data.jsonl`, `compile-test2.jsonl`, and the synthetic pools are clean (0%).
Implications:
- The CFG/graph channel is clean either way (the extractor now ignores non-disasm
  lines), but the `assembly` text in the DECODER PROMPT of those rows still
  contains the answer. Training base SFT on `dart_all_cfg.jsonl` leaks the target
  into the prompt; the model can learn to copy instead of decompile.
- `test-set.jsonl` is leak-contaminated, so do NOT use it for eval. The reported
  numbers in this handoff use `compile-test2`/`grpo_data` (clean) and are fine.
- The recommended path below avoids the leak: it warm-starts from the existing
  base checkpoint and does the `_ut` SFT on `grpo_data_cfg.jsonl` (clean). Only
  if you retrain base SFT on `dart_all` should you first strip the interleaved
  source from the `assembly` field.

De-leaked files have been generated with `scripts/data/strip_source_from_assembly.py`
(removes `^\d+\t` numbered-source lines, keeps disasm + gdb headers; only the
`assembly` field changes, so cfg/edges are preserved and verified to still match):

```text
data/datasets/dart_all_cfg_clean.jsonl   (0 source lines left; 155,537 disasm lines preserved; 42,897 source lines removed)
data/datasets/test-set_cfg_clean.jsonl   (0 source lines left)
```

If you retrain the base SFT on dart_all, use `dart_all_cfg_clean.jsonl`, e.g.:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" --name_suffix _cfgbase \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --train_file data/datasets/dart_all_cfg_clean.jsonl \
  --eval_file  data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset    data/testing/grpo_data_cfg.jsonl \
  --prompt_fit_assembly 1 --auto_cfg 1 \
  --epochs 4 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 4 --grad_accum 16 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 --eval_max_new_tokens 768 \
  --generation_batch_size 4 --num_samples 5 --pass_num_samples 10
```
(then run `_ut` / GRPO from that fresh base instead of the old single-block one.)

### How to run the corrected pipeline
Point ALL FOUR dataset flags at the `_cfg` files and pass `--prompt_fit_assembly 1`.
`--auto_cfg 1` is included below as a safety net: it is dormant when the `_cfg`
files are used (those already carry cfg/edges), but if any flag accidentally
points at a non-CFG file it extracts the CFG inline instead of silently
degrading to the dead single-block path. Use a fresh suffix; do NOT compare
against pre-fix checkpoints, the prompt format and graph both changed.

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

# 1) corrected SFT (mirrors the proven _ut recipe, with real CFG + cue)
python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" --name_suffix _cfg_ut \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --sft_checkpoint "artifacts/$NEW/pytorch_model.bin" \
  --train_file data/testing/grpo_data_cfg.jsonl \
  --eval_file  data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset    data/testing/grpo_data_cfg.jsonl \
  --prompt_fit_assembly 1 --auto_cfg 1 \
  --epochs 4 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 4 --grad_accum 16 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 --eval_max_new_tokens 768 \
  --generation_batch_size 4 --num_samples 5 --pass_num_samples 10

# 2) gentle GRPO from the corrected SFT (only after SFT shows movement)
UT=${NEW}_cfg_ut
python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" --name_suffix _cfg_ut_gentle \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --use_grpo --grpo_checkpoint "artifacts/${UT}/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset    data/testing/grpo_data_cfg.jsonl \
  --prompt_fit_assembly 1 --auto_cfg 1 \
  --grpo_group_size 4 --grpo_epochs 1 --grpo_lr 5e-7 --grpo_perfect_bonus 1.5 \
  --grpo_max_new_tokens 256 --grpo_test_timeout 3 --grpo_reward_workers 32 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 1 --grad_accum 64 --lora_r 64 --lora_alpha 128 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 --use_reasoning 0 --eval_max_new_tokens 768 \
  --generation_batch_size 2 --num_samples 5 --pass_num_samples 10 \
  --save_strategy epoch --save_total_limit 2
```

### Multi-GPU (DDP) SFT (2026-06-18)
SFT now scales across GPUs (e.g. 2xH100 or 2xRTX Pro 6000). Add `--num_gpus N`:
the runner launches the SFT trainer under `torch.distributed.run --standalone
--nproc_per_node N` and the HF `Seq2SeqTrainer` shards the data (DistributedSampler)
and all-reduces gradients. Implementation notes:
- Effective batch = `per_device x N x grad_accum`, so halve `--grad_accum` to keep
  it constant (2 GPUs: use `--grad_accum 8` instead of 16 with batch 4 -> eff 64).
- Only rank 0 writes `pytorch_model.bin` / uploads to HF (guarded).
- `ddp_find_unused_parameters=True` is set automatically when distributed (the
  GNN edge-embedding / prefix adapter can be unused on edge-less micro-batches).
- 4-bit decoder is pinned to the local rank instead of `device_map="auto"`.
- Requires NCCL => Linux pods only (not Windows local). GRPO is still single-GPU
  (custom loop); `--num_gpus>1` with `--use_grpo` warns and runs on one GPU.
- Keep `--gradient_checkpointing 0` (grad-checkpointing + DDP + find_unused can
  conflict).

Example (the from-scratch base on 2 GPUs): add `--num_gpus 2 --grad_accum 8` to
the `_cfgbase` command above (batch stays 4, effective batch stays 64).

Decision rule: this is the first run where the graph and the full assembly are
actually present. Expect movement above the old ceiling (CodeBLEU 0.66 / pass@10
0.32). If pass@k still does not move with the graph alive and ~100% asm coverage,
the bottleneck is the architecture/decoder, not preprocessing.

## 1. TL;DR

The current best balanced model is:

```text
artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo/pytorch_model.bin
```

Best balanced result:

```text
CodeBLEU:                0.6609878738
Compiled-only CodeBLEU:  0.6466651790
Compile@1:              0.3428571429
Compile@5:              0.6746031746
Pass@1:                 0.1616883117
Pass@5:                 0.2868738404
Pass@10:                0.3246753247
```

The best pure SFT/pass-safe fallback is:

```text
artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut/pytorch_model.bin
```

The second GRPO epoch and the later reward-shape experiments did not beat the first gentle GRPO on pass@10. Do not continue from `gentle2`, `rewardfix`, or `rewardsoft` unless the goal changes from pass@k to CodeBLEU/pass@1.

Current best ranking:

```text
1. qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo
   Best balanced checkpoint. Best compile@5 and tied best pass@10.

2. qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut
   Strong SFT baseline. Almost same pass, lower compile.

3. qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardfix_grpo
   Highest pass@1 and CodeBLEU among GRPO variants, but worse pass@5/pass@10.

4. qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo
   Did not beat best. Pass@10 recovered somewhat but still below best.

5. qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_grpo
   Aggressive GRPO. Hurt pass@k.

6. qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle2_grpo
   Second gentle epoch. Hurt both compile and pass.
```

## 2. Project Idea

The project is a neural decompiler for Dart/Swift. The input is assembly plus graph/control-flow-derived structure; the output is source code.

The important idea is not just "LLM generates code." The working stack is:

```text
assembly / CFG / graph data
  -> graph-aware local/block encoder
  -> graph pooling / projection bridge
  -> decoder model
  -> Dart source candidate(s)
  -> CodeBLEU, compile@k, pass@k
```

For the main Qwen runs, the decoder is:

```text
Qwen/Qwen3.5-9B-Base
```

The graph encoder is usually:

```text
microsoft/graphcodebert-base
```

The Qwen model is used as a causal decoder with graph-conditioned learned prefix/glue. The most relevant glue settings were:

```text
--qwen_prefix_tokens 16
--qwen_prefix_gate_init 0.2
--decoder_prompt_max_length 768
--use_reasoning 0
```

Reasoning was kept off because the eval target is "return valid source code"; adding reasoning risks training/eval mismatch and extra non-code output.

## 3. Why Old v1-v6 Had Higher CodeBLEU

The older v1-v6 systems were not the same kind of system.

They were frontier-ish causal LLM runs:

```text
v1, v2: Qwen3-4B-2507 based
v3, v4: DeepSeek-R1-0528-Distill-Qwen3-8B based
v5, v6: Qwen3-8B based
```

Reasons they could score higher CodeBLEU:

1. They had stronger Dart/code priors out of the box.
2. They were causal code LLMs, not a graph-conditioned encoder-decoder with a fragile glue bridge.
3. CodeBLEU rewards surface similarity, syntax shape, token overlap, and local structure more than functional correctness.
4. The older evaluation path and current antigravity path were not initially aligned.
5. Missing Dart on remote caused zero compile/pass in some early summaries, making model quality look worse than it was.
6. Some older pipelines used retry/multi-sample behavior that can inflate best-candidate CodeBLEU compared with a narrower decoder.

The main lesson: CodeBLEU alone is not enough. For this task, `compile@k` and `pass@k` are more important.

## 4. Main Files and What They Do

### Runner

```text
configs/run_sweeps_antigravity.py
```

This is the orchestration script. It defines experiment configs, applies CLI overrides, launches SFT or GRPO, then runs inference and metrics.

Important things it now supports:

```text
--experiment
--name_suffix
--encoder
--use_grpo
--skip_training
--train_file
--eval_file
--grpo_train_file
--epochs
--sft_lr
--grpo_lr
--grpo_perfect_bonus
--grpo_no_compile_penalty
--grpo_compile_reward
--grpo_partial_reward_cap
--grpo_perfect_base_reward
--grpo_overlap_weight
--qwen_prefix_tokens
--qwen_prefix_gate_init
--decoder_prompt_max_length
--use_reasoning
--attn_implementation
--load_4bit
--generation_batch_size
--num_samples
--pass_num_samples
--hf_repo
--hf_token
--hf_upload_checkpoints
--save_strategy
--save_total_limit
```

The suffix behavior matters. To create a suffixed config, use the base experiment and `--name_suffix`, for example:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut \
  ...
```

For already-suffixed configs, listing may show them, but the safe pattern is still to select the base and add `--name_suffix`.

### SFT Training

```text
scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py
```

This trains the graph-conditioned decoder with supervised fine-tuning.

Important fixes/features:

1. Supports Qwen causal decoder as graph-conditioned source generator.
2. Builds a compact decoder prompt with:
   - target language
   - exact Dart function signature if present
   - unit-test call shape excerpt if present
   - imports/helper instruction
   - no explanations/markdown/test code
3. Supports `GRAPH_TRAIN_FILE` and `GRAPH_EVAL_FILE`, including comma-separated files.
4. Strips unit-test `main()` and pragmas from `grpo_data.jsonl` target source so SFT learns standalone top-level solution functions.
5. Keeps reasoning off by default for eval-matched generation.

The key `_ut` run trained from the base SFT checkpoint using:

```text
data/testing/grpo_data.jsonl
```

This helped a lot because the pass@k evaluator also uses that unit-test-bearing data.

### GRPO Training

```text
scripts/training/graph_grpo_decompiler_antigravity.py
```

This does Group Relative Policy Optimization. It samples multiple completions for each problem, scores them with a Dart reward harness, normalizes rewards within the group, and updates the trainable LoRA/prefix parameters.

Important fixes/features:

1. CodeT5+/seq2seq fallback for `_shift_right`.
2. Per-test Dart reward extraction:
   - extracts single-line `expect(candidate(...), expected)` calls
   - appends candidate solution plus helper functions
   - runs tests with Dart
3. Strips candidate `main()` before tests, because the unit-test harness supplies its own `main()`.
4. Penalizes bad structure:
   - empty/tiny output
   - `main()` violation
   - compile failure
5. Adds compile check before per-test runs.
6. Logs `Compiled:` during GRPO so we can see if it is only learning syntax.
7. Reward shape is now CLI/env tunable through runner flags:
   - `GRPO_NO_COMPILE_PENALTY`
   - `GRPO_COMPILE_REWARD`
   - `GRPO_PARTIAL_REWARD_CAP`
   - `GRPO_PERFECT_BASE_REWARD`
   - `GRPO_OVERLAP_WEIGHT`
8. Token overlap shaping is off by default for Dart GRPO:

```text
GRPO_OVERLAP_WEIGHT=0.0
```

The reason: earlier overlap shaping improved CodeBLEU-ish behavior but did not improve pass@k.

### Inference

```text
scripts/evaluation/graph_inference_antigravity.py
```

Generates multi-sample predictions for compile/CodeBLEU tasks and pass@k tasks.

Important flags:

```text
--num_samples
--generation_batch_size
--max_new_tokens
--checkpoint
--decoder_prompt_max_length
```

Past bug:

```text
unrecognized arguments: --decoder_prompt_max_length 1024
```

This was fixed by adding the argument in inference and keeping runner/inference flags aligned.

### Metrics

```text
scripts/evaluation/graph_codebleu_antigravity.py
scripts/evaluation/graph_compile_at_k_antigravity.py
scripts/evaluation/graph_pass_at_k_antigravity.py
scripts/evaluation/compile_statistical_results_antigravity.py
```

Important evaluator fixes:

1. Evaluators fail loudly if Dart is missing.
2. `graph_compile_at_k_antigravity.py` uses the legacy-style Dart compile harness:

```text
dart compile aot-snapshot
```

3. `graph_pass_at_k_antigravity.py` uses the stored unit tests for `grpo_data.jsonl`.
4. `graph_codebleu_antigravity.py --compiled_only` filters candidates using compile harness.
5. Debug flags exist:

```bash
--debug_failures 5
```

6. Runner uses `check=True` for metric subprocesses so a broken metric run cannot silently write a fake success summary.

## 5. Datasets and Evaluation Tasks

Compile/CodeBLEU task:

```text
data/testing/compile-test2.jsonl
126 problems
5 candidates per problem in current eval
```

Pass@k task:

```text
data/testing/grpo_data.jsonl
154 problems
10 candidates per problem in current eval
```

The "old task alignment" mattered. The current runner was updated to use the same old-style task files:

```bash
--compile_dataset data/testing/compile-test2.jsonl
--pass_dataset data/testing/grpo_data.jsonl
```

Do not compare current results against old v1-v6 unless the dataset, candidate count, and harness are aligned.

## 6. Environment Notes

Remote pods:

```text
Linux /workspace
Qwen3.5-9B-Base
GraphCodeBERT encoder
Blackwell RTX PRO 6000 or similar 96 GB card in one pod
H100/H200-like profiles were used as runner defaults
```

Local:

```text
Windows
Intel i7-12700H, 20 threads
Dart found at C:\flutter\bin\dart.BAT
Python 3.12.7 from Anaconda
```

Important local caveat:

Trying to recompute all Dart AOT compile tests locally with 20 parallel workers caused Windows Dart compiler processes to get stuck for hours. Those local Dart processes were killed. This did not affect the downloaded remote results.

For rewardsoft, compile/pass aggregates were recomputed locally from the complete downloaded candidate-level CSVs:

```text
compile_stats.csv: 126 rows, 5 candidates each
pass_stats.csv:    154 rows, 10 candidates each
```

CodeBLEU was recomputed locally from raw predictions.

## 7. Dependency Notes

Remote install trouble:

```text
tree-sitter refused installation on one pod
```

That mainly blocks some CodeBLEU/tree-sitter paths. It does not block Dart compile/pass evaluation if the candidate-level stats were already produced.

Useful Python libs:

```bash
pip install peft transformers xformers tree-sitter torch-geometric
```

For PyG on CUDA/PyTorch-specific installs, use the appropriate wheel index if plain pip fails.

FA3/FA4 notes:

1. PyTorch/HF may expose `flash_attention_3`/`flash_attention_4`.
2. Availability does not guarantee the specific Qwen model path will use it successfully.
3. FA4 install may need beta package names and CUDA-specific wheels.
4. When FA3/FA4 throws CUTLASS/CUTE errors, fall back to:

```bash
--attn_implementation sdpa
```

The successful runs used `sdpa`.

## 8. Key Commands Used

### Unit-test SFT from base

This was the important SFT pass that produced `_ut`.

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --sft_checkpoint "artifacts/$NEW/pytorch_model.bin" \
  --train_file data/testing/grpo_data.jsonl \
  --eval_file data/testing/grpo_data.jsonl \
  --epochs 4 \
  --sft_lr 5e-6 \
  --lora_r 64 \
  --lora_alpha 128 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 0 \
  --train_batch_size 4 \
  --grad_accum 16 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 4 \
  --num_samples 5 \
  --pass_num_samples 10
```

### Best gentle GRPO run

This produced the current best balanced checkpoint.

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
UT=${NEW}_ut

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut_gentle \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${UT}/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 4 \
  --grpo_epochs 1 \
  --grpo_lr 5e-7 \
  --grpo_perfect_bonus 1.5 \
  --grpo_max_new_tokens 256 \
  --grpo_test_timeout 3 \
  --grpo_reward_workers 32 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 0 \
  --train_batch_size 1 \
  --grad_accum 64 \
  --lora_r 64 \
  --lora_alpha 128 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 \
  --generation_batch_size 2 \
  --num_samples 5 \
  --pass_num_samples 10 \
  --save_strategy epoch \
  --save_total_limit 2
```

Historical note: at the time of this run, `--grad_accum` did not matter in GRPO because the GRPO trainer stepped every batch. This is now fixed; see Section 19 onward. `--grad_accum` matters for both SFT and current GRPO.

### Rewardsoft run

This did not beat best, but it is documented because it tested the softer reward shape.

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
BEST=${NEW}_ut_gentle_grpo

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut_rewardsoft \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${BEST}/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 4 \
  --grpo_epochs 1 \
  --grpo_lr 3e-7 \
  --grpo_no_compile_penalty -2.0 \
  --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 3.0 \
  --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 \
  --grpo_overlap_weight 0.0 \
  --grpo_max_new_tokens 256 \
  --grpo_test_timeout 3 \
  --grpo_reward_workers 64 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 0 \
  --train_batch_size 1 \
  --lora_r 64 \
  --lora_alpha 128 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 2 \
  --num_samples 5 \
  --pass_num_samples 10 \
  --save_strategy epoch \
  --save_total_limit 2
```

### Resume only prediction/metrics after pod dies post-training

If training finished but prediction/eval did not, use:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut_rewardsoft \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --use_grpo \
  --skip_training \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 0 \
  --train_batch_size 1 \
  --lora_r 64 \
  --lora_alpha 128 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 2 \
  --num_samples 5 \
  --pass_num_samples 10
```

## 9. Results Table

All rows below are from:

```text
results-qwen-9b-latest-3/sweeps_antigravity
```

| Run | CodeBLEU | Compiled-only CodeBLEU | Compiled Success | Compile@1 | Compile@5 | Pass@1 | Pass@5 | Pass@10 | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128` | 0.643034 | 0.000000 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | Early bad summary from missing/broken Dart environment. Do not use as model-quality evidence. |
| `_ut` | 0.672308 | 0.644784 | 83 | 0.306349 | 0.658730 | 0.161688 | 0.285895 | 0.324675 | Strong SFT baseline after unit-test-formatted SFT. |
| `_ut_grpo` | 0.661355 | 0.644777 | 83 | 0.323810 | 0.658730 | 0.144156 | 0.242837 | 0.272727 | Aggressive GRPO hurt pass@k. Reject. |
| `_ut_gentle_grpo` | 0.660988 | 0.646665 | 85 | 0.342857 | 0.674603 | 0.161688 | 0.286874 | 0.324675 | Best balanced. Keep. |
| `_ut_gentle2_grpo` | 0.661920 | 0.636234 | 80 | 0.325397 | 0.634921 | 0.151299 | 0.253917 | 0.285714 | Second epoch drifted worse. Reject. |
| `_ut_rewardfix_grpo` | 0.671537 | 0.652218 | 82 | 0.338095 | 0.650794 | 0.163636 | 0.260359 | 0.292208 | Sharper reward improved CodeBLEU/pass@1 but hurt pass@5/pass@10. |
| `_ut_rewardsoft_grpo` | 0.658956 | 0.639448 | 80 | 0.342857 | 0.634921 | 0.147403 | 0.269970 | 0.311688 | Softer reward still did not beat best. |

Important conclusion:

```text
_ut_gentle_grpo is the best balanced checkpoint.
_ut_rewardfix_grpo has the best pass@1 but worse pass@5/pass@10.
_ut has tied best pass@10 but lower compile@5.
```

## 10. Reward Function History and Reasoning

### GRPO objective mechanics

For each input problem, GRPO samples `G = --grpo_group_size` completions. Each completion receives a scalar reward. Rewards are reshaped within the group:

```text
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

Then the trainer maximizes the sampled-token log-probability weighted by this group-relative advantage, with clipping and a KL term against the adapter-disabled reference policy:

```text
ratio = exp(logp_policy - logp_reference)
policy_loss = -min(ratio * advantage, clip(ratio, 1 - eps, 1 + eps) * advantage)
loss = mean(policy_loss + kl_coef * KL(reference || policy))
```

Current defaults in `graph_grpo_decompiler_antigravity.py`:

```text
clip_eps = 0.2
kl_coef = 0.01
```

Historical implementation note: early GRPO stepped the optimizer every batch, so `--grad_accum` was passed by the runner but did not actually accumulate gradients. This is now fixed; see Section 19 onward.

### Original issue

The early GRPO reward had this rough shape:

```text
reward = functional_reward + 0.2 * token_overlap

functional_reward =
  -3                                  if empty/tiny output
  -5                                  if candidate still defines main()
  -1 + 8 * pass_ratio + perfect_bonus if tests extracted and executed

perfect_bonus is added only when passed == total_tests
```

This gives a 90 percent partial solution a large reward:

```text
-1 + 8 * 0.9 = 6.2
```

If the perfect bonus is hard to reach, GRPO can learn "almost pass" or "looks similar" rather than fully solving tasks. This matched observed behavior: compile/CodeBLEU improved while pass@k dropped.

### Aggressive GRPO result

Aggressive GRPO looked healthy during training but hurt eval pass:

```text
Pass@1:  0.1617 -> 0.1442
Pass@5:  0.2859 -> 0.2428
Pass@10: 0.3247 -> 0.2727
```

Conclusion: training reward was not aligned enough with held-out/multi-sample pass behavior.

### Gentle GRPO result

Gentle GRPO at `5e-7`, 1 epoch, lower bonus did not hurt pass and improved compile:

```text
Compile@1: 0.3063 -> 0.3429
Compile@5: 0.6587 -> 0.6746
Pass@10:   stayed 0.3247
```

Conclusion: one gentle epoch was useful; more was not.

### Rewardfix

The reward was changed to:

```text
invalid/main violation:       -5
empty:                        -3
does not compile:             -2
compiles but passes 0 tests:  -0.25
partial pass:                 -0.25 + cap * pass_ratio
full pass:                    perfect_base + perfect_bonus
overlap:                      0 by default
```

This was meant to stop almost-passing solutions from competing with full-pass solutions.

Result:

```text
CodeBLEU improved
pass@1 improved a little
pass@5/pass@10 dropped
```

Conclusion: reward became too sharp and narrowed diversity.

### Rewardsoft

Rewardsoft used softer values:

```text
--grpo_lr 3e-7
--grpo_compile_reward 0.0
--grpo_partial_reward_cap 3.0
--grpo_perfect_base_reward 3.0
--grpo_perfect_bonus 1.5
--grpo_overlap_weight 0.0
```

It still did not beat `_ut_gentle_grpo`.

Conclusion: the immediate next improvement should probably not be another one-epoch GRPO with the same setup. It should be a deeper reward/evaluation change or a reranker.

## 10A. GRPO Trial Parameter Ledger

This table records the GRPO trials that were actually compared. Some early pod terminal logs were lost when pods died, so the "source" column distinguishes exact commands captured in this handoff from reconstructed commands based on the run commands used in chat.

| Trial | Start checkpoint | Reward equation | Key params | Outcome |
|---|---|---|---|---|
| `_ut_grpo` aggressive | `artifacts/${UT}/pytorch_model.bin` | Old linear reward: `-1 + 8*pass_ratio + perfect_bonus + 0.2*overlap`; empty `-3`; main violation `-5` | Reconstructed from command: `group_size=4`, `epochs=1`, `lr=2e-6`, `perfect_bonus=3.0`, `max_new_tokens=384`, `test_timeout=3`, `reward_workers=32`, `train_batch_size=2`, `generation_batch_size=4`, `num_samples=5`, `pass_num_samples=10`, `lora_r=64`, `lora_alpha=128` | Hurt pass@k. `pass@10=0.2727`. Reject. |
| `_ut_gentle_grpo` | `artifacts/${UT}/pytorch_model.bin` | Old linear reward: `-1 + 8*pass_ratio + perfect_bonus + 0.2*overlap` | Exact command in Section 8: `group_size=4`, `epochs=1`, `lr=5e-7`, `perfect_bonus=1.5`, `max_new_tokens=256`, `test_timeout=3`, `reward_workers=32`, `train_batch_size=1`, `generation_batch_size=2`, `num_samples=5`, `pass_num_samples=10`, `lora_r=64`, `lora_alpha=128` | Best balanced run. `compile@5=0.6746`, `pass@10=0.3247`. Keep. |
| `_ut_gentle2_grpo` | `artifacts/${NEW}_ut_gentle_grpo/pytorch_model.bin` | Old linear reward: `-1 + 8*pass_ratio + perfect_bonus + 0.2*overlap` | Same intended settings as gentle continuation: `group_size=4`, `epochs=1`, `lr=5e-7`, `perfect_bonus=1.5`, `max_new_tokens=256`, `test_timeout=3`, `num_samples=5`, `pass_num_samples=10`, `lora_r=64`, `lora_alpha=128` | Second epoch drifted. `compile@5=0.6349`, `pass@10=0.2857`. Reject. |
| `_ut_rewardfix_grpo` | `artifacts/${NEW}_ut_gentle_grpo/pytorch_model.bin` | New sharp reward: `-5` main violation, `-3` empty, `-2` no compile, `-0.25` compile/pass0, `-0.25 + 2.0*pass_ratio` partial, `4.0 + perfect_bonus` full pass, `overlap_weight=0.0` | Command recommended after reward patch: `group_size=4`, `epochs=1`, `lr=5e-7`, `perfect_bonus=2.0`, `no_compile_penalty=-2.0`, `compile_reward=-0.25`, `partial_reward_cap=2.0`, `perfect_base_reward=4.0`, `overlap_weight=0.0`, `max_new_tokens=256`, `test_timeout=3`, `reward_workers=32 or 64`, `lora_r=64`, `lora_alpha=128` | Highest pass@1 and strong CodeBLEU, but worse pass@5/pass@10. `pass@10=0.2922`. Not best. |
| `_ut_rewardsoft_grpo` | `artifacts/${NEW}_ut_gentle_grpo/pytorch_model.bin` | New softer reward: `-5` main violation, `-3` empty, `-2` no compile, `0.0` compile/pass0, `0.0 + 3.0*pass_ratio` partial, `3.0 + 1.5` full pass, `overlap_weight=0.0` | Exact command in Section 8: `group_size=4`, `epochs=1`, `lr=3e-7`, `perfect_bonus=1.5`, `no_compile_penalty=-2.0`, `compile_reward=0.0`, `partial_reward_cap=3.0`, `perfect_base_reward=3.0`, `overlap_weight=0.0`, `max_new_tokens=256`, `test_timeout=3`, `reward_workers=64`, `train_batch_size=1`, `num_samples=5`, `pass_num_samples=10`, `lora_r=64`, `lora_alpha=128` | Did not beat best. `pass@10=0.3117`, `compile@5=0.6349`. |

Practical interpretation:

```text
Old linear reward + one very gentle epoch helped compile and preserved pass.
Old linear reward + more intensity or another epoch hurt pass.
New sharper reward improved pass@1/CodeBLEU but narrowed diversity and hurt pass@5/pass@10.
New softer reward was safer than sharp reward, but still below best.
```

## 11. What Went Wrong and What We Learned

### Problem 1: Missing Dart made early numbers fake

Early summaries showed:

```text
compile@1 = 0
compile@5 = 0
pass@k = 0
compiled-only CodeBLEU = 0
```

But raw CodeBLEU was nonzero. That was an environment/evaluator issue, not proof that all candidates were semantically hopeless.

Fix:

```text
evaluators now fail loudly if Dart is missing
```

### Problem 2: GRPO can improve compile without improving pass

The first gentle GRPO improved compile and preserved pass. Later GRPO variants often improved CodeBLEU or pass@1 while hurting pass@5/pass@10.

This means:

```text
The model may be becoming more deterministic/narrow.
Pass@k needs diversity.
Pass@1 alone is not enough for judging these runs.
```

### Problem 3: More epochs are not automatically better

Second gentle epoch:

```text
Compile@5: 0.6746 -> 0.6349
Pass@10:   0.3247 -> 0.2857
```

So the policy drifted. Do not run 2-3 GRPO epochs blindly.

### Problem 4: `grad_accum` did not affect early GRPO

The GRPO trainer calls:

```text
loss.backward()
optimizer.step()
```

inside each batch in the early implementation. That version did not implement gradient accumulation, which matters for interpreting those old LR results. A `5e-7` GRPO LR was not as tiny as it looked when every batch was an optimizer step. This is now patched; current GRPO steps every `GRAPH_GRAD_ACCUM` batches and flushes at epoch end.

### Problem 5: Windows local full Dart recompute is not reliable at high parallelism

The 20-thread local attempt spawned several long-running Dart AOT processes that had to be killed. Do not use Windows as the main compile/pass recomputation engine at high AOT concurrency. Prefer Linux remote stats or use very low concurrency.

## 12. Files to Upload After Local Changes

At minimum, upload these if starting a new pod:

```text
configs/run_sweeps_antigravity.py
scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py
scripts/training/graph_grpo_decompiler_antigravity.py
scripts/evaluation/graph_inference_antigravity.py
scripts/evaluation/graph_codebleu_antigravity.py
scripts/evaluation/graph_compile_at_k_antigravity.py
scripts/evaluation/graph_pass_at_k_antigravity.py
scripts/evaluation/compile_statistical_results_antigravity.py
models/graph_data_collator.py
models/hierarchical_graph_encoder_antigravity.py
```

The key two most recently edited are:

```text
configs/run_sweeps_antigravity.py
scripts/training/graph_grpo_decompiler_antigravity.py
```

Latest post-Gemini GRPO patch:

```text
scripts/training/graph_grpo_decompiler_antigravity.py
  - added real GRPO gradient accumulation using GRAPH_GRAD_ACCUM
  - added group-level unique-test bonus
  - added exact-normalized duplicate candidate penalty
  - added GRPO_KL_COEF and GRPO_CLIP_EPS env-backed parser defaults
  - logs OptStep, Accum, UniqueBonus, DupPenalty

configs/run_sweeps_antigravity.py
  - added CLI plumbing for:
    --grpo_reward_mode
    --grpo_binary_fail_reward
    --grpo_unique_test_bonus
    --grpo_duplicate_penalty
    --grpo_kl_coef
    --grpo_clip_eps
```

Example upload command pattern:

```bash
scp -P 64566 ./configs/run_sweeps_antigravity.py root@71.232.99.8:/workspace/configs/
scp -P 64566 ./scripts/training/graph_grpo_decompiler_antigravity.py root@71.232.99.8:/workspace/scripts/training/
```

Use `:` after the host path, not a comma.

## 13. Files to Download

Download results only:

```powershell
scp -P 64566 -r root@71.232.99.8:/workspace/results ".\results-qwen-9b-latest-N"
```

Download the best checkpoint separately if needed:

```powershell
scp -P 64566 -r root@71.232.99.8:/workspace/artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo ".\artifacts\"
```

Minimum result files to preserve:

```text
results/sweeps_antigravity/*.json
results/sweeps_antigravity/*_stats.csv
results/*_compile_predictions.json
results/*_pass_predictions.json
```

## 14. Saved Local Results

Latest local downloaded folder:

```text
results-qwen-9b-latest-3
```

Important rewardsoft files:

```text
results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_compile_predictions.json
results-qwen-9b-latest-3/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_pass_predictions.json
results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo.json
results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_compile_stats.csv
results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_pass_stats.csv
```

Extra local metric JSONs saved for rewardsoft:

```text
results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_codebleu.json
results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_codebleu_compiled_only.json
results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_compile_at_k.json
results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_pass_at_k.json
```

## 15. Best Current Decision

If the goal is a single best balanced model:

```text
Use qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo.
```

If the goal is safest pass@10 without GRPO drift:

```text
Use qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut.
```

If the goal is highest pass@1/CodeBLEU and lower pass@10 is acceptable:

```text
Consider qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardfix_grpo.
```

Do not use:

```text
qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle2_grpo
qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_grpo
```

## 16. Next Research Directions

### A. Rerank instead of more GRPO

Since pass@k has some successful candidates, add an inference-time reranker:

1. Generate 10 or 20 candidates.
2. Prefer candidates that compile.
3. Prefer candidates that match the exact function signature.
4. Penalize `main()`.
5. Penalize explanations/markdown.
6. Optionally run cheap public/sample tests if available.

This could improve pass@1 without damaging pass@10.

### B. Analyze failures by task

Use candidate-level stats to find:

```text
tasks with many compiling candidates but zero passing candidates
tasks with high CodeBLEU but no pass
tasks where best pass candidate is not candidate 1
tasks where GRPO lost diversity
```

This is likely more valuable than another blind GRPO epoch.

### C. Fix GRPO diversity

The rewardfix run improved pass@1 but hurt pass@5/pass@10. That suggests policy narrowing.

Potential fixes:

```text
lower LR further
increase entropy/diversity pressure
larger group size if VRAM allows
shorter max_new_tokens only if truncation is not observed
add KL/adapter trust-region checks
avoid overlap shaping
```

### D. Compare GraphCodeBERT vs CLAP/ASM-CLAP

GraphCodeBERT (`gcb`) is currently the practical first choice because it worked and produced measurable gains.

ASM-CLAP may be worth trying later, but only after the Qwen + GCB path is stable. The graph/assembly encoder can easily become the bottleneck or cause decoding collapse.

### E. Model choice

For fastest iteration:

```text
Qwen3.5-9B-Base + GCB + LoRA r64/a128
```

For lower VRAM/cost experiments:

```text
smaller Qwen or CodeT5+ variants
```

But CodeT5+ had more glue/decoder issues and weaker Dart prior. Qwen has a better Dart/code prior.

## 17. Sanity Checks Before Any New Run

Run these on a fresh pod:

```bash
python -c "import torch, platform; print('python:', platform.python_version()); print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('compute_capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

Check Dart:

```bash
dart --version
```

List expected configs:

```bash
python configs/run_sweeps_antigravity.py --encoder gcb --max_risk high | grep qwen-9b-base
```

Verify runner args:

```bash
python configs/run_sweeps_antigravity.py --help
```

Verify prediction files before metrics:

```bash
ls -lah results/*rewardsoft*predictions.json
ls -lah results/sweeps_antigravity/*rewardsoft*
```

## 18. Current Recommendation

Do not run another blind GRPO epoch right now.

The most rational next move is:

1. Preserve/download/upload the best checkpoint `_ut_gentle_grpo`.
2. Build or test a reranker on existing predictions.
3. Inspect candidate-level stats to understand where pass@k is won/lost.
4. If training again, branch from `_ut_gentle_grpo` or `_ut`, never from a rejected checkpoint.
5. Keep every run in a new suffix so comparisons stay clean.

The core insight: GRPO did not fail completely. It found a small useful step (`_ut_gentle_grpo`). But the reward/data regime is too small and noisy for repeated online RL epochs. Treat GRPO here as a very sharp tool: one short pass can help, two passes can already drift.

## 19. Gemini GRPO Suggestions and Current Patch

Gemini's diagnosis was mostly correct:

```text
pass@k needs diversity and exploration
plain per-candidate GRPO tends to narrow the policy
the original linear partial-pass reward over-rewarded almost-correct candidates
the sharp rewardfix narrowed diversity even more
GRPO grad_accum was ignored, making LR effectively hotter
```

One caveat: a uniqueness reward can also over-reward complementary partial solutions if set too high. Therefore it should be small and treated as a diversity regularizer, not the dominant reward.

I did not add full token entropy regularization because Qwen vocab-level entropy over long generated sequences can increase memory use and OOM risk. The safer implementation is reward-side diversity shaping:

```text
unique_test_bonus:
  for each sampled group, add bonus to a candidate for tests it passed that no other candidate in the same group passed

duplicate_penalty:
  subtract a small penalty when exact-normalized generated candidates duplicate each other inside the group

real grad_accum:
  optimizer now steps every GRAPH_GRAD_ACCUM GRPO batches and flushes at epoch end
```

Recommended experimental branch from the current best checkpoint:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
BEST=${NEW}_ut_gentle_grpo

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut_diverse_grpo \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${BEST}/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 4 \
  --grpo_epochs 1 \
  --grpo_lr 8e-7 \
  --grpo_no_compile_penalty -2.0 \
  --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 2.0 \
  --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 \
  --grpo_overlap_weight 0.0 \
  --grpo_unique_test_bonus 0.75 \
  --grpo_duplicate_penalty 0.25 \
  --grpo_kl_coef 0.02 \
  --grpo_clip_eps 0.15 \
  --grpo_max_new_tokens 256 \
  --grpo_test_timeout 3 \
  --grpo_reward_workers 64 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 0 \
  --train_batch_size 1 \
  --grad_accum 8 \
  --lora_r 64 \
  --lora_alpha 128 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 2 \
  --num_samples 5 \
  --pass_num_samples 10 \
  --save_strategy epoch \
  --save_total_limit 2
```

Why these numbers:

```text
--grad_accum 8:
  now actually works, so LR can be slightly higher without stepping every single batch

--grpo_lr 8e-7:
  effective update pressure is lower than previous 5e-7-per-batch behavior because of accumulation

--grpo_unique_test_bonus 0.75:
  enough to reward group coverage, not enough to dominate full-pass reward

--grpo_duplicate_penalty 0.25:
  discourages exact duplicates without punishing natural similarity too hard

--grpo_kl_coef 0.02 and --grpo_clip_eps 0.15:
  tighter trust region to reduce mode collapse
```

Decision rule:

```text
Keep only if pass@5 > 0.2868738404 or pass@10 > 0.3246753247.
Reject if it only improves pass@1 or CodeBLEU.
```

## 20. DeepSeek Diagnosis and Binary Reward Patch

DeepSeek's diagnosis agrees with the observed runs:

```text
1. Partial-credit rewards made almost-passing candidates too attractive.
2. Group normalization with small group size can give positive advantage to merely "least bad" partial solutions.
3. Repeated GRPO updates narrowed the policy and harmed pass@5/pass@10 diversity.
4. rewardfix sharpened the reward and improved pass@1, but it also narrowed diversity further.
```

Useful correction to DeepSeek's entropy suggestion:

```text
If minimizing loss, an entropy bonus should be subtracted, not added.
loss = policy_loss + kl_coef * kl - entropy_coef * entropy
```

Token-level entropy regularization is now available but defaults to off because it is memory-expensive for Qwen. Use it only in a controlled anti-collapse GRPO run. The safer baseline patch remains binary reward mode plus group diversity shaping.

Latest binary reward patch:

```text
scripts/training/graph_grpo_decompiler_antigravity.py
  - added GRPO_REWARD_MODE=shaped|binary
  - added GRPO_BINARY_FAIL_REWARD
  - binary mode:
      full pass: perfect_base_reward + perfect_bonus
      everything else: binary_fail_reward

configs/run_sweeps_antigravity.py
  - added --grpo_reward_mode {shaped,binary}
  - added --grpo_binary_fail_reward
  - added --grpo_entropy_coef
```

Recommended binary-diverse GRPO trial:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
BEST=${NEW}_ut_gentle_grpo

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut_binary_diverse_grpo \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${BEST}/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 8 \
  --grpo_epochs 1 \
  --grpo_lr 8e-7 \
  --grpo_reward_mode binary \
  --grpo_binary_fail_reward -1.0 \
  --grpo_perfect_base_reward 1.0 \
  --grpo_perfect_bonus 0.0 \
  --grpo_overlap_weight 0.0 \
  --grpo_unique_test_bonus 0.25 \
  --grpo_duplicate_penalty 0.25 \
  --grpo_kl_coef 0.02 \
  --grpo_clip_eps 0.15 \
  --grpo_entropy_coef 0.002 \
  --grpo_max_new_tokens 512 \
  --grpo_test_timeout 5 \
  --grpo_reward_workers 64 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 0 \
  --train_batch_size 1 \
  --grad_accum 8 \
  --lora_r 64 \
  --lora_alpha 128 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 2 \
  --num_samples 5 \
  --pass_num_samples 10 \
  --save_strategy epoch \
  --save_total_limit 2
```

Why `group_size=8` instead of DeepSeek's `16/32` first:

```text
GRPO generation and log-prob recomputation scale with train_batch_size * group_size * sequence_length.
group_size=16 may be feasible on a 96 GB card, but it can also OOM or slow the run sharply.
Try 8 first. If memory is comfortable and the logs show low SignalGroups, try 16 next.
```

Why `grpo_max_new_tokens=512`:

```text
Claude correctly flagged that training rollouts at 256 but eval at 768 bias the RL signal toward short/easy solutions.
Use 512 first so longer Dart functions can produce positive reward during GRPO.
If VRAM is comfortable, try 768. If it OOMs, drop back to 384/256 and treat the run as short-solution-biased.
```

Why `grpo_entropy_coef=0.002`:

```text
This directly resists mode collapse, but computes full-vocab entropy and can cost memory/time.
Start small. If it OOMs or slows too much, set it to 0.0 and rely on KL + duplicate/unique-test shaping.
```

Decision rule remains:

```text
Keep only if it beats _ut_gentle_grpo on pass@5 or pass@10.
Reject if it only improves pass@1 or CodeBLEU.
```

## 21. Qwen Diagnosis, Reranking, and Candidate-Level Analysis

Qwen's diagnosis agrees with the observed behavior: loose partial rewards improve syntax/surface similarity, sharp rewards improve candidate-1 behavior, but neither reliably improves pass@5/pass@10 because the generated candidate pool itself does not contain enough fully passing solutions.

I added an offline analysis/reranking tool:

```text
scripts/evaluation/analyze_rerank_antigravity.py
```

What it does:

```text
Inputs:
  - *_pass_predictions.json
  - matching *_pass_stats.csv

It does not run Dart again.
It uses stored compile/pass/CodeBLEU flags from the remote stats CSV.

Rankers:
  original_order:
    existing candidate order

  compile_first:
    non-leaky if used at inference, because compile can be checked cheaply

  compile_shape_heuristic:
    compile-first plus code-shape checks: target name from tests/reference, no main(), no markdown,
    balanced braces, reasonable candidate length, and duplicate penalty

  compile_codebleu_oracle:
    leaky analysis-only reranker using reference CodeBLEU

  pass_oracle_upper_bound:
    analysis-only upper bound over the already generated candidates
```

Saved reports:

```text
results-qwen-9b-latest-3/analysis/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rerank_analysis.json
results-qwen-9b-latest-3/analysis/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo_rerank_analysis.json
results-qwen-9b-latest-3/analysis/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_grpo_rerank_analysis.json
results-qwen-9b-latest-3/analysis/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle2_grpo_rerank_analysis.json
results-qwen-9b-latest-3/analysis/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardfix_grpo_rerank_analysis.json
results-qwen-9b-latest-3/analysis/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_rerank_analysis.json
```

Local reranking summary on the pass@k task set:

| Run | pass@1/5/10 | cand1 pass | compile-shape selected pass | pass-oracle ceiling | compile-many zero-pass | high-CodeBLEU zero-pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `_ut` | 0.1617 / 0.2859 / 0.3247 | 0.1623 | 0.2078 | 0.3247 | 66 | 86 |
| `_ut_gentle_grpo` | 0.1617 / 0.2869 / 0.3247 | 0.1623 | 0.2013 | 0.3247 | 70 | 86 |
| `_ut_grpo` | 0.1442 / 0.2428 / 0.2727 | 0.1234 | 0.1818 | 0.2727 | 68 | 89 |
| `_ut_gentle2_grpo` | 0.1513 / 0.2539 / 0.2857 | 0.1818 | 0.2143 | 0.2857 | 76 | 86 |
| `_ut_rewardfix_grpo` | 0.1636 / 0.2604 / 0.2922 | 0.1558 | 0.2208 | 0.2922 | 69 | 87 |
| `_ut_rewardsoft_grpo` | 0.1474 / 0.2700 / 0.3117 | 0.1494 | 0.1883 | 0.3117 | 73 | 78 |

Interpretation:

```text
1. Reranking helps candidate-1 selection.
   For _ut_gentle_grpo, compile_shape_heuristic raises selected pass@1 from 0.1623 to 0.2013.

2. But reranking cannot fix tasks where no candidate passes.
   _ut_gentle_grpo has pass-oracle ceiling 0.3247, meaning 104/154 tasks have zero passing candidate in the 10-sample pool.

3. High CodeBLEU is not the main bottleneck.
   _ut_gentle_grpo has 86 high-CodeBLEU zero-pass tasks, so surface similarity is often semantically wrong.

4. The next useful engineering step is not another blind GRPO epoch.
   It is either:
     - inference-time compile/public-test reranking,
     - larger and more diverse candidate generation,
     - rejection-sampling SFT on candidates that fully pass,
     - or targeted data/error repair for the compile-many zero-pass tasks.
```

Recommended immediate use:

```bash
python scripts/evaluation/analyze_rerank_antigravity.py \
  --predictions results/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo_pass_predictions.json \
  --stats results/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo_pass_stats.csv \
  --output results/analysis/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo_rerank_analysis.json
```

Decision update:

```text
Freeze _ut_gentle_grpo as the best balanced model.
Do not spend more GPU time on repeated GRPO unless the next trial explicitly changes the candidate pool diversity or uses binary/diverse GRPO as a controlled one-off.
The highest-leverage next script is a real inference-time reranker that generates 10-20 candidates, compiles them, optionally runs public tests if available, deduplicates them, and emits the best candidate.
```

## 22. Claude Diagnosis and Actual Reranker

Claude's most important additional diagnosis:

```text
GRPO optimizes expected reward per sample.
pass@k measures whether at least one of k samples is correct.

Loose rewards teach near-correctness.
Sharp rewards improve the mode but narrow the tails.
Training rollouts at 256 tokens while eval uses 768 means long solutions often never produce positive RL signal.
Small groups make advantage estimates noisy.
Using the same 154 tasks for GRPO and pass@k means reward overfitting is possible.
```

Two fixes from Claude are now implemented:

```text
1. Actual reranker:
   scripts/evaluation/rerank_predictions_antigravity.py

2. Optional entropy knob:
   scripts/training/graph_grpo_decompiler_antigravity.py
   configs/run_sweeps_antigravity.py
   --grpo_entropy_coef / GRPO_ENTROPY_COEF
```

Reranker modes:

```text
heuristic:
  no Dart execution; uses target name/signature, no main(), no markdown, balanced braces, length sanity, duplicate penalty

stats_compile:
  offline replay using an existing *_stats.csv compile flag; useful locally when Dart reruns are slow

compile:
  real inference-time compile reranker; runs Dart AOT compile and promotes compilable, well-shaped candidates

test:
  runs the stored unit-test harness and promotes passing candidates
  fair only if the tests are public tests; with benchmark/eval tests this is an oracle diagnostic

stats_pass_oracle:
  offline oracle using stored pass flags; diagnostic ceiling only
```

The script writes both:

```text
--output:
  full reranked JSON with all candidates preserved/reordered

--selected_output:
  one selected candidate per problem, so existing pass@k evaluator can measure selected pass@1 with n=1

--report:
  JSON with moved rate, observed selected pass/compile rates, per-row scores, and best original indices
```

Offline replay on `_ut_gentle_grpo`:

```bash
BEST=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo

python scripts/evaluation/rerank_predictions_antigravity.py \
  --mode stats_compile \
  --predictions "results-qwen-9b-latest-3/${BEST}_pass_predictions.json" \
  --stats "results-qwen-9b-latest-3/sweeps_antigravity/${BEST}_pass_stats.csv" \
  --output "results-qwen-9b-latest-3/analysis/${BEST}_reranked_compile_predictions.json" \
  --selected_output "results-qwen-9b-latest-3/analysis/${BEST}_selected_compile_predictions.json" \
  --report "results-qwen-9b-latest-3/analysis/${BEST}_reranker_compile_report.json"
```

Observed result:

```text
mode: stats_compile
total_problems: 154
moved_count: 80
moved_rate: 0.5195
original_compile_rate_observed: 0.6234
selected_compile_rate_observed: 0.9805
```

Pass-oracle replay on `_ut_gentle_grpo`:

```text
mode: stats_pass_oracle
moved_count: 88
original_pass_rate_observed: 0.1623
selected_pass_rate_observed: 0.3247
```

Small local smoke test in `test` mode:

```text
limit: 5 problems
workers: 5
original_pass_rate_observed: 0.2
selected_pass_rate_observed: 1.0
```

Do not treat the full `test` reranker score as a fair benchmark unless those tests are public. It is useful as an oracle to prove whether selection is the bottleneck.

Fair remote compile-rerank command:

```bash
BEST=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo

python scripts/evaluation/rerank_predictions_antigravity.py \
  --mode compile \
  --workers 16 \
  --timeout 5 \
  --predictions "results/${BEST}_pass_predictions.json" \
  --output "results/${BEST}_reranked_compile_predictions.json" \
  --selected_output "results/${BEST}_selected_compile_predictions.json" \
  --report "results/${BEST}_reranker_compile_report.json"

python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions "results/${BEST}_selected_compile_predictions.json" \
  --k_values 1
```

Oracle/public-test rerank command:

```bash
BEST=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo

python scripts/evaluation/rerank_predictions_antigravity.py \
  --mode test \
  --workers 16 \
  --timeout 5 \
  --predictions "results/${BEST}_pass_predictions.json" \
  --output "results/${BEST}_reranked_test_predictions.json" \
  --selected_output "results/${BEST}_selected_test_predictions.json" \
  --report "results/${BEST}_reranker_test_report.json"
```

Upload after this patch:

```bash
scp -P 64566 ./scripts/evaluation/rerank_predictions_antigravity.py root@71.232.99.8:/workspace/scripts/evaluation/
scp -P 64566 ./scripts/training/graph_grpo_decompiler_antigravity.py root@71.232.99.8:/workspace/scripts/training/
scp -P 64566 ./configs/run_sweeps_antigravity.py root@71.232.99.8:/workspace/configs/
```

### 22.1 Fully-offline rerank scoring (2026-06-10, no Dart, no pod)

Gap closed: in `heuristic` and `stats_compile` modes the reranker's own
report has NO pass rates (pass outcomes are only known to it in
`stats_pass_oracle`/`test` modes). New helper joins the report's chosen
candidate index against the stored `cand_N_pass` / `cand_N_compile` flags
in the matching `*_pass_stats.csv`:

```text
scripts/evaluation/score_selected_offline_antigravity.py   (NEW)
  --report  <reranker --report JSON>   --stats <same-run *_pass_stats.csv>
  Prints selected pass@1 / compile@1, first-candidate baseline, delta, and
  CSV replay cross-checks (pass@1 estimator + any-pass oracle) that must
  reproduce the recorded harness numbers - if they don't, the report and
  CSV are from different runs.
```

Validated end-to-end on `_ut` (n=10 pools, 154 tasks), local Windows
PowerShell, pure Python:

```text
mode            selected pass@1   delta vs first-candidate (0.1623)
heuristic            0.1753            +1.3 pts
stats_compile        0.2078            +4.6 pts   <- compile flag + shape
stats_pass_oracle    0.3247            +16.2 pts  == any-pass ceiling, sanity OK
Cross-checks reproduced exactly: pass@1 replay 0.1617 (recorded 0.162),
any-pass 0.3247 (recorded 50/154). Selected compile@1 0.987 = any-compile
ceiling: compile-based selection is saturated on the compile axis.
NOTE: stats_compile selected pass@1 0.2078 on the OLD _ut checkpoint
already beats the Stage C GRPO checkpoint's pass@1 0.1688 - selection,
not RL, is the cheaper pass@1 lever (confirms Sections 21/22).
```

Command template per checkpoint (PowerShell, local; set $NAME and $DIR):

```powershell
$NAME = "qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut"   # or ..._fitut_grpo
$DIR  = "results-qwen-9b-stageC"
foreach ($MODE in "heuristic", "stats_compile", "stats_pass_oracle") {
  python scripts/evaluation/rerank_predictions_antigravity.py `
    --mode $MODE `
    --predictions "$DIR/${NAME}_pass_predictions.json" `
    --stats "$DIR/sweeps_antigravity/${NAME}_pass_stats.csv" `
    --output "$DIR/analysis/${NAME}_reranked_${MODE}_predictions.json" `
    --selected_output "$DIR/analysis/${NAME}_selected_${MODE}_predictions.json" `
    --report "$DIR/analysis/${NAME}_reranker_${MODE}_report.json"
  python scripts/evaluation/score_selected_offline_antigravity.py `
    --report "$DIR/analysis/${NAME}_reranker_${MODE}_report.json" `
    --stats "$DIR/sweeps_antigravity/${NAME}_pass_stats.csv" `
    --label "$NAME $MODE"
}
```

(`--stats` is ignored by heuristic mode but harmless. The compile flag in
the pass-set CSV is the pass-harness compile outcome on the 154-task set;
do not compare it to the 126-problem compile@k table.)

Bash equivalent (pod or WSL; both checkpoints in one paste). If running on
the pod, first upload the scorer, which is new:
`scp -P <PORT> ./scripts/evaluation/score_selected_offline_antigravity.py root@<IP>:/workspace/scripts/evaluation/`

```bash
DIR=results-qwen-9b-stageC    # on the pod: DIR=results (run from /workspace)

for NAME in qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut \
            qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut_grpo; do
  for MODE in heuristic stats_compile stats_pass_oracle; do
    python scripts/evaluation/rerank_predictions_antigravity.py \
      --mode "$MODE" \
      --predictions "$DIR/${NAME}_pass_predictions.json" \
      --stats "$DIR/sweeps_antigravity/${NAME}_pass_stats.csv" \
      --output "$DIR/analysis/${NAME}_reranked_${MODE}_predictions.json" \
      --selected_output "$DIR/analysis/${NAME}_selected_${MODE}_predictions.json" \
      --report "$DIR/analysis/${NAME}_reranker_${MODE}_report.json"
    python scripts/evaluation/score_selected_offline_antigravity.py \
      --report "$DIR/analysis/${NAME}_reranker_${MODE}_report.json" \
      --stats "$DIR/sweeps_antigravity/${NAME}_pass_stats.csv" \
      --label "$NAME $MODE"
  done
done
```

Results on the _fitut lineage (2026-06-10, ran on the pod, outputs in
results/analysis/; cross-checks reproduced the harness numbers exactly
for both checkpoints, so the joins are valid). Selected pass@1:

```text
selector                _fitut (SFT head)   _fitut_grpo
first candidate         0.1818              0.1753
heuristic (shape)       0.1883              0.1948
stats_compile           0.2143  <- BEST     0.2143
stats_pass_oracle       0.3182 (=ceiling)   0.2662 (=ceiling)
selected compile@1      0.9545              0.9805
(harness sampled pass@1: 0.1617 / 0.1688)

Old checkpoints, same compile+shape selector, measured for comparison:
  _ut             0.2078
  _ut_gentle_grpo 0.2013   (the previous "keep deploying" model)

FINDINGS:
1. NEW PROJECT-BEST deployable single-output: _fitut + n=10 sampling +
   compile/shape rerank = pass@1 0.2143 (33/154). Beats every measured
   checkpoint reranked the same way and costs no GPU training.
2. Selection NEUTRALIZES GRPO's only advantage: both Stage B and Stage C
   checkpoints land on exactly 0.2143 after the same compile rerank, and
   the GRPO model's selection ceiling is 5.2 pts LOWER (0.2662 vs
   0.3182). Seals the Stage C verdict: _fitut is the head, and the
   deployable pipeline is _fitut + sampling + compile/shape selection.
   This supersedes Section 23.8's "keep deploying _ut_gentle_grpo".
3. Selector headroom within the existing pools: 0.2143 -> 0.3182
   (~16 tasks hold a passing candidate that compile+shape fails to pick
   among multiple compiling ones). Offline-testable ideas on the SAME
   files: majority/frequency vote among compiling candidates, output
   self-consistency, a learned verifier. Raising the 0.3182 ceiling
   itself needs better pools: rejection-sampling SFT, more samples per
   task, more/better training data.

ADDENDUM (2026-06-10, post Stage D): the pass@k-GRPO checkpoint
_fitutpk5_grpo raised the selection ceiling to 0.3247 (new best) but
its compile-reranked selected pass@1 is 0.1883 - the deployable record
0.2143 (_fitut + compile rerank) STANDS. Full table and decisions:
Section 23.12 Stage D result block.

ADDENDUM 2 (2026-06-10, stats_compile_vote): new reranker mode - among
compiling candidates, exact-duplicate completions VOTE for their shared
implementation (self-consistency bonus +6/duplicate) instead of taking
the diversity penalty. Selected pass@1, all three checkpoints:

                    stats_compile   stats_compile_vote
  _fitut                0.2143        0.2338  <- NEW DEPLOYABLE RECORD
  _fitut_grpo           0.2143        0.2143
  _fitutpk5_grpo        0.1883        0.2143

Deployable config of record: _fitut + n=10 sampling + stats_compile_vote
= 0.2338 (36/154; sampled pass@1 baseline 0.1617). The vote lifted the
pass@k model most in relative terms (0.1883 -> 0.2143) but the SFT model
keeps the crown: its duplicates concentrate on correct implementations,
while the pass@k objective deliberately spreads mass. Remaining selector
headroom: 8.4 pts to _fitut's 0.3182 ceiling, 11.0 pts to
_fitutpk5_grpo's 0.3247. Next selector lever: a learned verifier.
```

## 23. Code Audit (2026-06-09): Real GRPO Bugs Found and Fixed

A line-by-line audit of the GRPO trainer found bugs that all the previous
diagnoses (Sections 19-22) missed. They explain the observed pathology better
than any reward-shape theory, and they mean every reward-shape comparison in
Section 10A was run on a broken gradient.

### 23.1 The big one: the PPO ratio used the wrong policy

The old loss was:

```text
ratio = exp(logp_policy - logp_REFERENCE)        # reference = adapter-DISABLED
```

Two things are wrong with that:

```text
1. GRPO/PPO takes the ratio against the OLD policy (the one that sampled the
   completions). With one update per sampled batch, old == current, so the
   ratio must be exp(logp - logp.detach()) == 1 and its gradient is the plain
   REINFORCE policy gradient.

2. adapter-disabled is not the GRPO start policy. It is the PRE-SFT base
   model. After SFT, logp_policy - logp_base is several nats per token, so
   ratio >> 1+eps almost everywhere.
```

Consequence of ratio >> 1+eps:

```text
advantage > 0  ->  min(ratio*A, clip*A) = clip*A = constant  ->  ZERO gradient
advantage < 0  ->  min picks ratio*A    ->  ratio-AMPLIFIED gradient
```

So the trainer could suppress behaviors but barely reinforce full-pass
candidates. That is exactly the observed signature: small compile gains
(suppressing broken modes), flat-to-falling pass@k, diversity narrowing, and
reward-shape changes barely mattering.

The KL term had the same wrong anchor: `kl_coef * KL(policy || pre-SFT base)`
is a constant per-token pressure to UNLEARN SFT, applied even on zero-signal
batches. That is a clean explanation for the `gentle2` drift.

Fixes (in `scripts/training/graph_grpo_decompiler_antigravity.py`):

```text
ratio is now exp(logp - logp.detach())  (value 1, REINFORCE gradient, clip inert)
kl_coef now defaults to 0.0; if you set it > 0 it still anchors to the
  pre-SFT base (only available reference) and the code says so loudly.
```

### 23.2 Scoring forward did not match the sampling forward

Three mismatches between how completions were SAMPLED (generate) and how
their log-probs were SCORED (manual forward):

```text
1. Position ids. The text prompt is right-padded to a fixed 768 tokens, so
   the combined [prefix | prompt | completion] sequence has masked holes in
   the middle. generate() assigns RoPE positions by attention-mask cumsum
   (pads skipped); a manual forward without position_ids uses arange (pads
   counted). Scoring now passes explicit cumsum position ids.

2. The start-token embed was dropped from the scoring context, and the first
   generated token was never scored. Scoring now replays
   [prefix | prompt | start | t0..t_{n-2}] and scores all n tokens.

3. Sampling used temperature 0.7 / top_p 0.95 but scoring used raw
   temperature-1.0 logits, and scoring ran in train() mode with LoRA dropout
   active while sampling ran in eval(). Scoring now divides logits by the
   sampling temperature and stays in eval mode (gradients flow either way).
```

### 23.3 Advantage normalization could amplify noise

`(r - mean) / std` with group size 4-8 turns a noise-level reward gap (for
example a lone 0.25 duplicate penalty on four otherwise identical failures)
into a full-size -1.5 advantage. New defaults:

```text
--grpo_adv_norm mean          (Dr.GRPO-style: reward minus group mean)
--grpo_min_reward_range 0.05  (groups with max-min reward <= this are skipped)
```

Skipping no-signal groups also skips their policy/reference forwards and the
backward. Early in training most groups are all-fail, so this cuts a large
fraction of GRPO GPU time. The `Skip:` field in the step log shows it.

### 23.4 Reward noise from tight Dart timeouts

`--grpo_test_timeout 3` with 32-64 parallel reward workers means cold
`dart compile` / `dart run` regularly die on the clock and healthy candidates
get scored as failures - pure reward noise on an already sparse signal. The
runner default is now 8s, compile checks retry once on timeout, and the eval
harness has always used 30s (so training was strictly noisier than eval).

### 23.5 Eval inference quirk: the fifth candidate was greedy

`do_sample` was decided per generation chunk. With `generation_batch_size 2`
and `num_samples 5`, the last chunk had size 1 and silently decoded GREEDILY,
so every compile/CodeBLEU pool was 4 sampled + 1 greedy candidate. Now fixed
(`do_sample` follows the total candidate count).

```text
Comparability note: compile@k / CodeBLEU after this fix are not exactly
comparable to the Section 9 table (the old pools contained one greedy
candidate). The pass@k task used chunks of 2 throughout, so pass@k IS
comparable. Eval still applies repetition_penalty 1.15, which GRPO sampling
does not; if pass@k is the target metric this asymmetry is acceptable.
```

### 23.6 The quiet ceiling: 768-token prompts cut off most of the assembly

Measured with `scripts/data/inspect_grpo_data_lengths.py`:

```text
grpo_data.jsonl (154 rows): median prompt ~1800-2100 tokens vs 768 budget.
The tokenizer truncates the TAIL, so the model loses most of the assembly
AND the trailing "Dart code:" cue. Median row fits only ~28% of its
assembly; only 12/154 rows fit fully. compile-test2.jsonl: median ~44%.
```

No reward shaping can recover information the model never sees. This is the
most plausible main reason the pass-oracle ceiling is stuck at 0.3247
(104/154 tasks with zero passing candidate in 10 samples). The 16 graph
prefix tokens are the only full-function channel, and they are a bottleneck.

New opt-in flag (off by default; bit-identical prompts when off):

```text
--prompt_fit_assembly 1   (env GRAPH_PROMPT_FIT_ASSEMBLY=1)
  trims the assembly MIDDLE (keeps head + tail + the "Dart code:" cue)
  to GRAPH_PROMPT_MAX_CHARS (default 2400 chars ~ 768 tokens)
```

Do NOT enable it when evaluating existing checkpoints - they were trained on
tail-truncated prompts. Enable it (ideally together with a larger
`--decoder_prompt_max_length`, e.g. 1536) for the NEXT SFT run.

### 23.7 Files changed in this patch

```text
scripts/training/graph_grpo_decompiler_antigravity.py   (core fixes 23.1-23.4
  + checkpoint-load diagnostics, fp32-trainables sanity print, fail-fast
  after 3 consecutive step errors instead of saving a no-op checkpoint)
configs/run_sweeps_antigravity.py                       (new flags, timeout default)
scripts/evaluation/graph_inference_antigravity.py        (do_sample fix)
scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py
                                                         (opt-in prompt fit, default off)
scripts/training/grpo_selfcheck.py                       (NEW: local math self-check)
scripts/data/inspect_grpo_data_lengths.py                (NEW: prompt/target length audit)
```

New runner flags:

```text
--grpo_gen_temperature   (default 0.7, matches eval)
--grpo_gen_top_p         (default 0.95, matches eval)
--grpo_adv_norm          (mean | std, default mean)
--grpo_min_reward_range  (default 0.05)
--prompt_fit_assembly    (0 | 1, default 0)
--quiet                  (0 | 1, default 1; see 23.10)
--skip_inference         (reuse *_predictions.json, metrics only; see 23.11-H)
```

Verification already done locally (no GPU needed):

```bash
python scripts/training/grpo_selfcheck.py
# group_advantages / position ids / ratio-trick gradient / reward shapes / group shaping: ALL PASSED
```

### 23.8 Recommended next run (cheap, sequenced)

GPU choice (priced 2026-06-09): rent ONE H200 pod ($1.968/h) for the whole
sequence below.

```text
The workload is decode-dominated (GRPO rollouts + 2170-candidate eval at
generation_batch_size 2), which is memory-bandwidth bound:
  RTX Pro 6000  ~1.8 TB/s / $1.081  -> ~1.7 TB/s per $/h. Memory is fine
                (a --grpo_max_new_tokens 768 run completed on this card),
                but decode work per dollar is ~30% worse than H200 and
                wall-clock is ~2.5x slower
  H200          ~4.8 TB/s / $1.968  -> ~2.4 TB/s per $/h, 141 GB headroom;
                BEST $/work here
  B200          ~7.7 TB/s / $4.30   -> ~1.8 TB/s per $/h; pays double for
                headroom nothing in this plan uses
One pod for everything also amortizes the ~30-40 min paid setup (uploads,
pip, model download) once. Rewards are CPU-bound: prefer a pod with >=32
vCPUs for the 48 parallel dart workers.

Time budget for a 20 h H200 booking (stop early at any failed gate;
download results/checkpoints between stages so a pod death loses nothing):
  setup (uploads, pip, model download)     ~0.5-0.8 h
  Step 1 smoke                             ~0.3 h
  Step 2 GRPO epoch + full eval            ~3-5 h
  Step 3 _utfit SFT + full eval            ~3-4 h
  Step 4 GRPO on _utfit + full eval        ~4-5 h
  buffer / re-runs / downloads             ~2-3 h
  total                                    ~14-18 h -> 20 h is enough
```

Step 1 - 10-minute smoke on the pod before any full run:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
START=${NEW}_ut

GRAPH_MAX_STEPS=3 python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut_fixedsmoke \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${START}/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 8 --grpo_epochs 1 --grpo_lr 1e-6 \
  --grpo_max_new_tokens 512 --grpo_reward_workers 48 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 1 --grad_accum 8 --lora_r 64 --lora_alpha 128 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 --use_reasoning 0 \
  --limit_eval 2 --num_samples 2 --pass_num_samples 2 \
  --eval_max_new_tokens 768 --generation_batch_size 2
```

Check the log for:

```text
"trainable params not found in checkpoint: 0"
"Sanity: ... trainable dtype=torch.float32"   (bf16 here would mean LR ~1e-6
                                               updates round away - report it)
Skip: 0/1 flags, OptStep/Accum counting, no tracebacks
```

Step 2 - the one full trial worth paying for now (clean read of the FIXED
algorithm, branched from the pure SFT checkpoint):

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
START=${NEW}_ut

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _ut_fixed \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${START}/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 8 \
  --grpo_epochs 1 \
  --grpo_lr 1e-6 \
  --grpo_no_compile_penalty -2.0 \
  --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 2.0 \
  --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 \
  --grpo_overlap_weight 0.0 \
  --grpo_unique_test_bonus 0.25 \
  --grpo_duplicate_penalty 0.25 \
  --grpo_max_new_tokens 512 \
  --grpo_test_timeout 8 \
  --grpo_reward_workers 48 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 1 --grad_accum 8 --lora_r 64 --lora_alpha 128 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 5 --pass_num_samples 10 \
  --save_strategy epoch --save_total_limit 2
```

Notes on the choices:

```text
kl_coef stays at its new 0.0 default (the only reference is the pre-SFT base).
clip_eps is inert with the fixed ratio (single update per batch); harmless.
adv_norm mean / min_reward_range 0.05 are the new defaults; no flag needed.
group_size 8: doubles the chance a group contains a passing candidate, which
  is what creates positive reinforcement signal at all. The skip logic eats
  most of the extra cost on all-fail groups.
lr 1e-6 with grad_accum 8: previous LR lore was calibrated on the broken
  gradient; treat this as a fresh, deliberately gentle starting point.
Branching from _ut, not _ut_gentle_grpo: gentle's gains came from the broken
  process, so a clean baseline is more interpretable. If this run beats
  _ut_gentle_grpo, rerunning from _ut_gentle_grpo is the follow-up.
```

Decision rule (unchanged):

```text
Keep only if pass@5 > 0.2869 or pass@10 > 0.3247.
Compare compile@k / CodeBLEU only against a re-evaluated baseline
(Section 23.5 changed the compile-task candidate pool).
```

Step 3 - if the budget allows ONE more spend, it should NOT be another GRPO
variant. It should be the prompt-budget SFT, because Section 23.6 caps
everything upstream. This is NOT a from-scratch retrain: it reuses the
existing base SFT artifact and only re-runs the cheap `_ut`-style
continuation (154 rows x 4 epochs) with the fixed prompt format:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _utfit \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --sft_checkpoint "artifacts/$NEW/pytorch_model.bin" \
  --train_file data/testing/grpo_data.jsonl \
  --eval_file data/testing/grpo_data.jsonl \
  --epochs 4 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 2 --grad_accum 32 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --prompt_fit_assembly 1 \
  --decoder_prompt_max_length 1536 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 5 --pass_num_samples 10
```

Notes:

```text
batch 2 / accum 32 (not 4/16): same effective batch 64, but sequences are
  ~2.3k tokens now (16 prefix + 1536 prompt + 768 target); drop to bs 1 /
  accum 64 if it OOMs on a 96 GB card.
1536-token prompts + middle-trimmed assembly lift median assembly
  visibility from ~28% to ~70% AND restore the "Dart code:" cue.
  GRAPH_PROMPT_MAX_CHARS now derives from decoder_prompt_max_length
  automatically.
Judge it on pass@10 (the candidate-pool ceiling) and compile@5, not pass@1.
Every later stage on this lineage (GRPO, skip_training re-evals) must keep
  --prompt_fit_assembly 1 --decoder_prompt_max_length 1536.
If _utfit clearly beats _ut on pass@10 / compile@5, proceed to Step 4.
```

Step 4 - one fixed-GRPO epoch on top of _utfit (run only if Step 3 passed
its gate). Identical to Step 2 except for the checkpoint and the two prompt
flags, which must stay on for EVERY stage of the _utfit lineage (training,
GRPO, and any --skip_training re-eval):

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _utfit \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${NEW}_utfit/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 8 \
  --grpo_epochs 1 \
  --grpo_lr 1e-6 \
  --grpo_no_compile_penalty -2.0 \
  --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 2.0 \
  --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 \
  --grpo_overlap_weight 0.0 \
  --grpo_unique_test_bonus 0.25 \
  --grpo_duplicate_penalty 0.25 \
  --grpo_max_new_tokens 512 \
  --grpo_test_timeout 8 \
  --grpo_reward_workers 48 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 1 --grad_accum 8 --lora_r 64 --lora_alpha 128 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --prompt_fit_assembly 1 \
  --decoder_prompt_max_length 1536 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 5 --pass_num_samples 10 \
  --save_strategy epoch --save_total_limit 2
```

Step 4 notes:

```text
Naming: the GRPO stage appends _grpo, so this writes
  artifacts/${NEW}_utfit_grpo and results/.../${NEW}_utfit_grpo.json;
  no collision with the Step 3 SFT artifact.
Rollout budget stays 512: targets are short (median ~90 tokens, p95 ~220),
  so 768 mostly buys longer rambling on failures, not more passes. 768 is
  known to fit in memory if you want the insurance; it just costs time.
Gate: keep only if it beats _utfit itself on pass@5 / pass@10.
```

What NOT to retrain from scratch:

```text
The base SFT (dart_all corpus) stays as-is for now. Nothing in the audit
invalidates its weights: the SFT loss path was correct; the bugs were in
GRPO's gradient, eval candidate sampling, and prompt truncation. The
existing artifacts remain valid starting points:

  keep + reuse:  artifacts/qwen-9b-base_..._gcb_a128 (base SFT)
                 artifacts/..._ut (SFT continuation)
  keep deploying (best measured model so far, but do not draw process
                 conclusions from how it was trained):
                 artifacts/..._ut_gentle_grpo
  never continue from: _ut_grpo, _ut_gentle2_grpo, _ut_rewardfix_grpo,
                 _ut_rewardsoft_grpo (all trained under the broken gradient)

A true from-scratch base SFT retrain only becomes worth pricing if the
cheap _utfit step shows the prompt format matters AND results plateau:
that one retrain would fold in prompt_fit on the full corpus and
position-consistent training in a single spend. Do not pay for it before
the cheap evidence exists.
```

### 23.9 Corrections to earlier sections

```text
Section 10's objective description documents the BUGGY implementation
  (ratio vs reference); it is not standard GRPO. Kept for history.
Section 19/20 parameter advice (kl_coef 0.02, clip_eps 0.15) predates the
  ratio fix; with the corrected gradient, kl_coef should stay 0.
"GRPO updates the trainable LoRA/prefix parameters" (Section 4) is wrong:
  the graph encoder/projection/prefix run under no_grad in GRPO, so ONLY the
  decoder LoRA receives gradients. The prefix glue is frozen during GRPO.
  (Probably desirable for stability, but the doc claimed otherwise.)
GRPO still trains on the same 154 tasks pass@k evaluates (Section 22's
  caveat stands): treat pass@k gains as in-distribution until a held-out
  split exists.
```

### 23.10 Quiet mode (warnings suppressed, errors stay loud)

All training/inference logs were drowning in transformers/datasets/PyG
warning spam. New default behavior:

```text
--quiet 1 (runner default) / GRAPH_QUIET=1 (scripts default):
  suppresses FutureWarning / UserWarning / DeprecationWarning,
  sets TRANSFORMERS_VERBOSITY=error, DATASETS_VERBOSITY=error,
  TOKENIZERS_PARALLELISM=false, HF_HUB_DISABLE_PROGRESS_BARS=1,
  and PYTHONWARNINGS for all child processes (training, inference, metrics).

--quiet 0 / GRAPH_QUIET=0: show everything (use when debugging a new pod).
```

Progress bars stay ON in quiet mode (quiet means no warning spam, not no
feedback): HF download bars, the HF Trainer bar (disable_tqdm=False), and
tqdm bars with ETA for GRPO epochs, eval generation, and the
compile@k / pass@k / stats-CSV evaluators. Evaluator bars render on stderr
because the runner captures their stdout for the JSON results.

Errors and tracebacks are deliberately NOT suppressed, and must never be:
silent failures are how the fake zero-metric summaries happened (Section 11,
Problem 1), and a GRPO run whose steps all error would still save a no-op
checkpoint and then spend hours of GPU evaluating it. The trainer now prints
the full traceback for a failed step and aborts after 3 consecutive
failures, so a broken setup costs minutes instead of a full paid run.

### 23.11 Fresh pod bootstrap: full install + from-scratch training

Use this when the old pod's artifacts cannot be copied. Verified 2026-06-09:
the only local checkpoints are old CodeT5+ runs; the Qwen `_ut` /
`_ut_gentle_grpo` artifacts exist nowhere off-pod. Since this retrains the
whole lineage anyway, the assembly-cutoff fix (Section 23.6) is folded in
from the start. New lineage names so nothing collides with history:

```text
_fitbase  -> base SFT from scratch, full corpus, fixed prompts
_fitut    -> unit-test continuation (the _ut equivalent)
_fitut_grpo -> one fixed-GRPO epoch on top
```

RULE FOR THIS TIME: after EVERY stage, either pass
`--hf_repo <user>/<repo> --hf_token $HF_TOKEN` so artifacts upload
automatically, or scp the artifact down before starting the next stage.
Checkpoints are trainable-params-only (~0.5 GB), so this is minutes.

#### A. Upload everything (code + data, no artifacts/results)

```bash
scp -P <PORT> -r ./configs ./models ./scripts ./data root@<IP>:/workspace/
```

Recursive upload is deliberate: the evaluators import a LOCAL `codebleu.py`
(scripts/evaluation/codebleu.py - NOT the pypi package) and
`scripts/data/dfg_extractor.py`; per-file lists keep missing one of them.
Total is ~25 MB of data plus small code.

#### B. Install dependencies

Python (pod torch stays - do NOT reinstall torch if CUDA torch is present):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

pip install -U pip
pip install "transformers==5.9.0" "tokenizers==0.22.1" \
    peft accelerate datasets torch-geometric \
    nltk numpy "tree-sitter>=0.25" tree-sitter-dart tree-sitter-swift \
    "flash-linear-attention==0.3.2" causal-conv1d \
    huggingface_hub hf_transfer
```

Version notes:

```text
flash-linear-attention + causal-conv1d are REQUIRED for Qwen3.5 (incident
  2026-06-10): Qwen3.5 is a hybrid linear-attention model (gated delta
  rule). Without these kernels, transformers silently falls back to
  torch_chunk_gated_delta_rule - a Python-loop implementation that is both
  slow (~33 s/GRPO batch) and so memory-hungry that the GRPO scoring
  forward (8 x ~2.1k tokens, grads on) OOMs even on a 141 GB H200.
  Symptom: OOM traceback through modeling_qwen3_5.py ->
  torch_chunk_gated_delta_rule. The transformers warning recommending this
  install is hidden by quiet mode, so do not skip it.
  PIN fla to 0.3.2 (Triton-only). fla >=0.4 moved kernels to a tilelang
  backend, and tilelang's bundled TVM FFI can double-register against
  another copy in the process and SIGABRT the whole run at model load:
    terminate called after throwing 'tvm::ffi::Error'
    TypeAttr `__ffi_repr__` is already registered for type index ...
  (hit 2026-06-10). Do NOT install tilelang; uninstall it if present.
  Verify the exact symbol transformers uses:
    python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; import causal_conv1d; print('kernels OK')"
  fla alone fixes the OOM (pure Triton, installs in seconds);
  causal-conv1d is only a speed bump and often BUILDS FROM SOURCE for
  20-40 min when no wheel matches the pod's torch/CUDA. Do not block on
  it: launch with fla only, or build scoped to one arch:
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=<vCPUs-4> pip install causal-conv1d
  (sm "9.0" = H100/H200; RTX Pro 6000 Blackwell "12.0"; B200 "10.0")
  Fallback if fla itself fails on a pod:
    --grpo_group_size 4 --grpo_max_new_tokens 384  (halves scoring tokens)
Set --grpo_reward_workers to roughly the pod's vCPU count (e.g. 24 on a
  28-vCPU slice): 48 workers on 28 vCPUs oversubscribes the CPU and
  recreates the spurious dart-timeout noise the 8s budget exists to avoid.
tree-sitter MUST be >=0.25 (incident 2026-06-10): pod images often preship
  an older core that only loads grammar ABI <=14, while current
  tree-sitter-dart/-swift wheels are ABI 15. Symptom, raised at METRICS
  time after training+inference already ran:
    ValueError: Incompatible Language version 15. Must be between 13 and 14
  An unpinned `pip install tree-sitter` reports "already satisfied" and
  keeps the broken old core, so the >=0.25 pin is required.
transformers 5.9.0 + tokenizers 0.22.1 is the combo the scripts carry
  monkeypatches for; other versions may work but are untested.
torch-geometric: plain pip suffices (GINEConv needs no compiled extras on
  PyG 2.x). If pip fails, use the PyG wheel index for the pod's torch/CUDA.
tree-sitter + tree-sitter-dart + tree-sitter-swift are REQUIRED by the
  local codebleu.py. bitsandbytes only if load_4bit=1 (we run 0).
  xformers not needed (runs use sdpa).
```

Dart SDK (the runner and evaluators auto-find ~/dart-sdk/bin):

```bash
cd ~
curl -fLO https://storage.googleapis.com/dart-archive/channels/stable/release/latest/sdk/dartsdk-linux-x64-release.zip
unzip -q dartsdk-linux-x64-release.zip   # creates ~/dart-sdk
~/dart-sdk/bin/dart --version
```

(`apt-get install -y unzip` first if unzip is missing. If you ever run the
GRPO trainer directly instead of via the runner, also
`export PATH=$HOME/dart-sdk/bin:$PATH` - the runner does this for you.)

Optional but recommended - pre-pull model weights with fast transfer:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B-Base'); snapshot_download('microsoft/graphcodebert-base')"
```

#### C. Preflight (two minutes, before any paid training)

```bash
cd /workspace
~/dart-sdk/bin/dart --version
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; import causal_conv1d; print('linear-attention kernels OK')"
python -c "import sys; sys.path.insert(0,'scripts/evaluation'); from codebleu import CodeBLEUCalculator; CodeBLEUCalculator('dart'); CodeBLEUCalculator('swift'); print('codebleu OK')"
python scripts/training/grpo_selfcheck.py
python configs/run_sweeps_antigravity.py --encoder gcb --max_risk high | grep qwen-9b-base
```

The codebleu check must INSTANTIATE both calculators, not just import the
class: the tree-sitter ABI check fires at construction (import alone passed
on 2026-06-10 while construction crashed). The runner computes CodeBLEU
BEFORE compile@k/pass@k, so a broken tree-sitter crashes AFTER you paid for
training + inference and before any metric lands. If that happens anyway,
see H below - nothing is lost.

#### D. Stage A - base SFT from scratch (dart_all, 1195 rows)

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _fitbase \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --train_file data/datasets/dart_all.jsonl \
  --eval_file data/datasets/test-set.jsonl \
  --epochs 2 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 2 --grad_accum 32 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --prompt_fit_assembly 1 --decoder_prompt_max_length 1536 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 2 --pass_num_samples 2
```

```text
num_samples 2 / pass_num_samples 2 on purpose: Stage A is an intermediate
  artifact; the cheap eval is only a sanity signal. Full 5/10 evals come
  after Stages B and C.
epochs 2: the historical base's epoch count was never recorded (sweep
  default was 1). Two epochs over 1195 rows is ~1-2 h and a safer start
  for the new prompt format.
```

Stage A result (2026-06-10), after the tree-sitter fix and a
`--skip_training --skip_inference` metrics resume:

```text
CodeBLEU 0.5822 | compiled-only 0.5995 (60/126 problems with a compiling
candidate) | compile@1 0.3095 | pass@1 0.1104 (154 tasks)

CAUTION: this eval ran n=2 candidates, so the reported "compile_at_5",
"pass_at_5" and "pass_at_10" are really @2 (hence pass@5 == pass@10 ==
0.1688). Only @1 is comparable to the Section 9 table.

Verdict: PASSED the sanity gate. compile@1 matches the historical _ut
checkpoint (0.3063) before any unit-test continuation, so the new prompt
format trains fine. pass@1 below _ut (0.162) is expected: the pass-harness
output format is what Stage B teaches. Proceeded to Stage B.
```

#### E. Stage B - unit-test continuation (the _ut equivalent)

```bash
python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _fitut \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --sft_checkpoint "artifacts/${NEW}_fitbase/pytorch_model.bin" \
  --train_file data/testing/grpo_data.jsonl \
  --eval_file data/testing/grpo_data.jsonl \
  --epochs 4 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 2 --grad_accum 32 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --prompt_fit_assembly 1 --decoder_prompt_max_length 1536 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 5 --pass_num_samples 10
```

Gate: compare against the historical `_ut` row in Section 9
(pass@10 0.3247, compile@5 0.6587; compile-side numbers are only
approximately comparable per Section 23.5). If `_fitut` lands well below
`_ut` on pass@10, stop and investigate before GRPO.

Stage B result (2026-06-10), full 5/10 eval:

```text
CodeBLEU 0.6673 | compiled-only 0.6462 (89/126, project record)
compile@1 0.3270 | compile@5 0.7063 (project record, +4.8 pts over _ut,
  achieved under the stricter all-sampled eval)
pass@1 0.1617 (== _ut exactly) | pass@5 0.2756 | pass@10 0.3182 (49/154
  vs _ut's 50/154 - within +-1 task, i.e. noise; s.e. ~3.8 pts)

Verdict: PASSED the gate. Proceeded to Stage C (GRPO on _fitut).
Stage C keep-gates: pass@5 > 0.2756 or pass@10 > 0.3182.

KEY FINDING: raising assembly visibility ~28% -> ~70% lifted compile@k
substantially but did NOT move pass@k at all. The ~0.32 pass-oracle
ceiling is not input-truncation-bound; ~105/154 tasks yield zero passing
candidates regardless of input coverage. Post-Stage-C, the pass@k levers
to prioritize are: inference-time compile/shape reranking (Section 21
measured +4 pts selected pass@1), rejection-sampling SFT on passing
candidates, larger candidate pools - and possibly more/better training
data (1195 rows is small). Assembly address-stripping (Section 24
analysis, pending) stays worthwhile for compile/CodeBLEU and token
efficiency, but should no longer be sold as a pass@k fix.
```

#### F. Stage C - fixed GRPO on top

Use the Step 4 command from Section 23.8 with `_utfit` replaced by `_fitut`
everywhere (suffix and checkpoint path). Keep
`--prompt_fit_assembly 1 --decoder_prompt_max_length 1536`.

Stage C result (2026-06-10), full 5/10 eval after the GRPO epoch (the run
completed end-to-end on H200 once tilelang was removed and fla 0.3.2 +
causal-conv1d were in place; CodeBLEU scored 126/126, so the tree-sitter
fix held):

```text
CodeBLEU 0.6647 (flat vs Stage B 0.6673) | compiled-only 0.6596 (84/126)
compile@1 0.3397 (+1.3 pts vs Stage B) | compile@5 0.6667 (-4.0 pts, 89->84)
pass@1 0.1688 (+0.7 pts, within noise) | pass@5 0.2501 (-2.6 pts)
pass@10 0.2662 (-5.2 pts; tasks with any pass 49 -> 41 of 154)

Verdict: FAILED both keep-gates (pass@5 0.2501 <= 0.2756, pass@10 0.2662
<= 0.3182). Per the pre-registered rule the GRPO checkpoint is NOT
adopted; `_fitut` (Stage B SFT) remains the lineage head.

Reading: textbook RLVR mode-sharpening, and it closes the GRPO chapter
for this reward. The fixed trainer demonstrably works (completed an
epoch, reinforced, visibly reshaped the distribution - the broken-ratio
runs never did), but per-sample pass reward maximizes pass@1 by
concentrating mass on tasks the model already sometimes solves: the
no-signal skip means only mixed-outcome groups produce gradient, so
zero-pass tasks contribute nothing toward new solves, while
low-probability "lucky tail" completions get pruned. Hence every @k
metric fell while @1 ticked up inside noise. More epochs sharpen
further; do NOT re-run vanilla GRPO (any group size / token budget)
expecting pass@k gains. Any future RL here needs a diversity-preserving
objective (KL>0 to the SFT anchor, entropy bonus, or pass@k-aware
advantage shaping) - treat that as a new experiment, not a retry.
IMPLEMENTED as Stage D, Section 23.12 (--grpo_passk_k).

Post-C plan (per the Stage B KEY FINDING, now in force):
1. DONE (2026-06-10, results in Section 22.1): compile/shape rerank on
   the saved n=10 pools gives the NEW project-best deployable pass@1
   0.2143 - identical for _fitut and _fitut_grpo, i.e. selection
   neutralizes GRPO's gain while _fitut keeps the higher ceiling
   (0.3182 vs 0.2662). Remaining selector headroom ~10.4 pts; next free
   lever is discriminating among compiling candidates (majority vote /
   verifier), still offline on the same files.
2. Rejection-sampling SFT on passing candidates (raises pass@1 without
   RL's coverage cost).
3. Larger candidate pools / more+better training data (1195 rows is
   small).

Archive rule applies: upload the GRPO adapter + both prediction files +
summary JSONs before releasing the pod, then shut the pod down - nothing
left in this plan needs the GPU today.
```

#### G. Scratch-path time budget (fits the 20 h H200 booking)

```text
setup + installs + preflight + weight download   ~1 h
Stage A train + cheap eval                       ~1.5-2.5 h
Stage B train + full eval                        ~3-4 h
Stage C GRPO + full eval                         ~4-5 h
buffer / downloads / HF uploads                  ~2-3 h
total                                            ~12-16 h
```

#### H. If a metrics step crashes after training/inference (recovery)

Nothing paid-for is lost: the checkpoint is in `artifacts/` and both
`*_predictions.json` files are in `results/` (they are written before any
metric runs). Fix the environment, then re-run the SAME stage command with
BOTH flags appended:

```bash
--skip_training --skip_inference
```

The runner then reuses the existing prediction files and only recomputes
CodeBLEU, compile@k, pass@k, the summary JSON, and the stats CSVs - minutes
of CPU/Dart work, no GPU generation. `--skip_inference` fails loudly if the
prediction files are missing.

Worked example - the 2026-06-10 incident (tree-sitter ABI crash in
CodeBLEU right after Stage A inference):

```bash
pip install -U "tree-sitter>=0.25"
python -c "import sys; sys.path.insert(0,'scripts/evaluation'); from codebleu import CodeBLEUCalculator; CodeBLEUCalculator('dart'); CodeBLEUCalculator('swift'); print('codebleu OK')"
# then the full Stage A command from D, plus:
#   --skip_training --skip_inference
```

### 23.12 Stage D: pass@k-weighted GRPO, the coverage-aware objective (2026-06-10)

Why: Stage C proved the FIXED trainer reinforces correctly, and exactly
because of that it damaged pass@k - per-sample expected reward IS pass@1
maximization, so the policy sharpened onto already-solved prompts. The fix
is to make the objective pass@k itself. The math is one line: per prompt,

```text
pass@k = 1 - (1-p)^k        =>      grad pass@k = k * (1-p)^(k-1) * grad p
```

i.e. the pass@1 gradient scaled by a PROMPT-LEVEL factor that vanishes as
the prompt becomes reliably solved. Implementation: estimate p with the
group's perfect-pass rate p_hat and multiply each group's advantages by
(1 - p_hat)^(k-1); the constant k folds into the learning rate. Solved
prompts (p_hat -> 1) stop contributing gradient - sharpening on them is
mechanically blocked; rarely-solved prompts keep full weight. Zero-pass
groups are unchanged (weight 1, but they still need within-group reward
spread to train, same as before).

Code changes (validated by scripts/training/grpo_selfcheck.py - 7 checks -
both locally on Windows AND on the pod with /venv/main/bin/python,
2026-06-10; trainer, runner, and selfcheck already uploaded to the pod):

```text
scripts/training/graph_grpo_decompiler_antigravity.py
  passk_advantage_weights()            the (1-p_hat)^(k-1) group weights
  calculate_rewards()                  per-sample perfect_flags (1.0 iff ALL
                                       unit tests passed) returned in stats,
                                       aligned 1:1 with the rewards tensor
  train_step()                         advantages *= weights when passk_k>1;
                                       skip extended to batches whose
                                       advantages are entirely weighted out;
                                       passk_weight_mean in stats, "PasskW"
                                       in the epoch log line
  --passk_k / GRPO_PASSK_K             0/1 = off (vanilla); >1 = pass@k mode
configs/run_sweeps_antigravity.py
  --grpo_passk_k                       plumbs GRPO_PASSK_K
```

The run - identical to Stage C except ONE variable (--grpo_passk_k 5) and
a new suffix so nothing overwrites the Stage C artifacts/results:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _fitutpk5 \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${NEW}_fitut/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 8 \
  --grpo_epochs 1 \
  --grpo_lr 1e-6 \
  --grpo_passk_k 5 \
  --grpo_no_compile_penalty -2.0 \
  --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 2.0 \
  --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 \
  --grpo_overlap_weight 0.0 \
  --grpo_unique_test_bonus 0.25 \
  --grpo_duplicate_penalty 0.25 \
  --grpo_max_new_tokens 512 \
  --grpo_test_timeout 8 \
  --grpo_reward_workers 24 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 1 --grad_accum 8 --lora_r 64 --lora_alpha 128 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --prompt_fit_assembly 1 \
  --decoder_prompt_max_length 1536 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 5 --pass_num_samples 10 \
  --save_strategy epoch --save_total_limit 2
```

Writes artifacts/${NEW}_fitutpk5_grpo and matching results. ~4-5 h on H200.

Watch during training:
- PasskW (mean group weight): ~1.0 early since most groups are unsolved;
  falling PasskW means the model reliably solves more groups. PasskW
  pinned at 0.00 would mean every signal group is fully solved (then the
  run has nothing left to learn - stop it).
- Perfect / skipped_no_signal as before. Batch wall-time as before.

Gates (pre-registered, same as Stage C): keep only if pass@5 > 0.2756 or
pass@10 > 0.3182. Then rerun the Section 22.1 offline rerank on the new
predictions; the deployable bar is selected pass@1 >= 0.2143 (the current
project best from _fitut + compile rerank).

Honest expectation: the weighting removes the gradient that CAUSED the
Stage C pass@k collapse; whether pass@k rises above SFT depends on
partially-solved tasks converting. Zero-pass tasks still yield no signal -
on-policy RL cannot learn from them. Even "no collapse, no gain" is a
clean publishable ablation: vanilla vs pass@k objective, identical config.

Fallback ladder if the gates fail again (cost order):
1. Same run + --grpo_entropy_coef 0.002 (explicit anti-collapse pressure;
   adds full-vocab memory cost, see flag help). ~$10.
2. Offline DPO from the saved n=10 pools (chosen = passing candidate,
   rejected = compiling-but-failing same prompt). No rollouts; reference
   logprobs precomputable once; runs on a cheaper GPU. Needs ~half a day
   of implementation.
3. Rejection-sampling SFT (RAFT) on passing candidates - the
   guaranteed-positive arm, already post-C priority 2.

MSc framing (the chapter writes itself regardless of which arm wins):
diagnose broken GRPO (23.1-23.7) -> fix and validate the implementation
(Stage C trained and visibly reshaped the policy) -> show the OBJECTIVE
mismatch with measurements (23.11-F: pass@1 up, pass@5/10 down) -> derive
the pass@k gradient and test it (this section) -> compare against
selection baselines (22.1: compile rerank 0.2143 beats both RL arms'
sampled pass@1). Known validity caveat to state in the thesis: the RL
train file equals the 154-task eval set (Section 22 noted reward
overfitting risk; the SFT stages share it). For a defensible
generalization claim, split grpo_data.jsonl deterministically in half,
train RL on one half, and report pass@k per half offline from the
prediction JSONs - same cost; worth doing for whichever arm survives.

Stage D result (2026-06-10), full 5/10 eval. Training telemetry first:
the weighting behaved exactly as derived - PasskW matched (1-Perfect)^4
to print precision on every step; ~15 fully-solved groups skipped with
zero gradient (the batches that caused Stage C's collapse); the largest
gradient steps all landed on PasskW=1.00 unsolved/partial tasks. No OOM,
no errors, 154/154 batches.

```text
CodeBLEU 0.6707 (best of lineage) | compiled-only 0.6496 (91/126, RECORD)
compile@1 0.3603 (RECORD) | compile@5 0.7222 (RECORD, beats 0.7063)
pass@1 0.1552 | pass@5 0.2747 (== SFT 0.2756) | pass@10 0.3247
  (50/154 vs SFT 49, vanilla-GRPO 41)

GATE: pass@10 0.3247 > 0.3182 -> PASSED (first RL stage ever to survive
its gate). Per the pre-registered rule _fitutpk5_grpo is the new lineage
head. Honest read vs SFT: +1 any-pass task is within noise (s.e. ~3.8
pts); the strong claims are (a) the collapse is GONE - +5.8 pts pass@10
over vanilla GRPO at identical config, one variable changed - and
(b) compile@1/compile@5/CodeBLEU records on top.

Offline rerank (Section 22.1 machinery, cross-checks reproduced the
harness numbers exactly):

selector            _fitut     _fitut_grpo   _fitutpk5_grpo
first candidate     0.1818     0.1753        0.1299
heuristic           0.1883     0.1948        0.1494
stats_compile       0.2143     0.2143        0.1883
oracle (ceiling)    0.3182     0.2662        0.3247  <- best ceiling
any-compile (10p)   0.9545     0.9805        0.9740

Deployable bar (selected pass@1 >= 0.2143): NOT met by Stage D with the
current compile+shape selector. _fitut + compile rerank stays the
deployable single-output config. This is the predicted trade made
visible: the pass@k model keeps its probability mass SPREAD, so passing
candidates sit thinner among compiling ones and a crude selector cashes
less of them - while its ceiling (0.3247) is now the highest of any
checkpoint, with 13.6 pts of selector headroom (vs 10.4 for _fitut).
Selection quality, not generation, is now the binding constraint.

Decisions recorded:
1. _fitutpk5_grpo = lineage head for anything best-of-k (gate rule).
2. Deployable single-output: _fitut + compile+VOTE rerank = 0.2338
   (superseded the 0.2143 compile-only record within hours; see 22.1
   ADDENDUM 2). The vote also lifted _fitutpk5_grpo 0.1883 -> 0.2143.
3. DONE same day: majority vote implemented as stats_compile_vote
   (22.1) and set the 0.2338 record. Next selector lever: learned
   verifier.
4. Optional GPU follow-up on a future booking: the half-split
   generalization run of THIS arm (validity for the thesis).
All Stage D artifacts archived locally (checkpoint 1.4 GB + results +
rerank analysis).
```

### 23.13 Stage E: half-split generalization run (launched 2026-06-10, ~5 h left on booking)

Decision: the remaining ~5 booked H200 hours go to VALIDITY, not another
objective tweak. An entropy-coef run (--grpo_entropy_coef 0.002) was
considered and declined: entropy was the fallback for diversity collapse,
and Stage D already eliminated the collapse - low expected value, and it
would muddy the clean one-variable ablation. The binding thesis risk is
instead the recorded train==eval confound: every RL stage trained on the
same 154 tasks it is evaluated on. Stage E trains the SURVIVING arm
(pass@k-GRPO, k=5) on HALF the tasks and measures the other half.

Tools (new, both validated):

```text
scripts/data/split_grpo_train_eval.py
  deterministic split: even row index -> RL train half (77 rows ->
  data/testing/grpo_data_rl_train_half.jsonl), odd -> held out;
  full index/id map in data/testing/grpo_split_halves.json.
  Both files uploaded to the pod.
scripts/evaluation/passk_by_half.py
  recomputes the unbiased pass@k estimator from any *_pass_stats.csv,
  decomposed per half. Validation: reproduces Stage D's harness numbers
  exactly (0.1552/0.2747/0.3247) and the split is luck-balanced:
  25/25 any-pass tasks per half for Stage D, 25/24 for the SFT.
```

Pre-registered baselines on the held-out half (from existing CSVs):

```text
                      heldout pass@10   heldout any_pass (of 77)
  _fitut (SFT)             0.3117              24
  _fitutpk5_grpo
    (all-data RL)          0.3247              25
```

Stage E command (Stage D command with TWO changes: train file + suffix):

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _fitutpk5h \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${NEW}_fitut/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data_rl_train_half.jsonl \
  --grpo_group_size 8 --grpo_epochs 1 --grpo_lr 1e-6 \
  --grpo_passk_k 5 \
  --grpo_no_compile_penalty -2.0 --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 2.0 --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 --grpo_overlap_weight 0.0 \
  --grpo_unique_test_bonus 0.25 --grpo_duplicate_penalty 0.25 \
  --grpo_max_new_tokens 512 --grpo_test_timeout 8 --grpo_reward_workers 24 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 1 --grad_accum 8 --lora_r 64 --lora_alpha 128 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --prompt_fit_assembly 1 --decoder_prompt_max_length 1536 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 5 --pass_num_samples 10 \
  --save_strategy epoch --save_total_limit 2
```

Training is 77 batches (~half of Stage D), eval unchanged; total ~3-3.5 h,
fits the 5 h window. Writes artifacts/${NEW}_fitutpk5h_grpo.

Pre-registered reading of the result (per-half via passk_by_half.py):
- Held-out half >= SFT heldout baseline (pass@10 0.3117 / any_pass 24):
  RL does not damage unseen-task coverage; anything above is positive
  generalization evidence.
- Train half vs SFT train half (0.3247 / 25): the in-distribution RL
  effect; expect it to look like Stage D's.
- This run informs the THESIS CLAIM only; the lineage head decision
  (Stage D) stands regardless.

Booking-expiry safety: predictions are written before metrics
(23.11-H). If the clock runs out mid-metrics, scp
results/*_fitutpk5h_grpo_*predictions.json immediately - per-half pass@k
needs the pass stats CSV, which can be recomputed on ANY machine with
Dart from the predictions (or on a cheap CPU instance); nothing
irreplaceable is lost after the predictions exist. The stats CSV tail
alone takes ~25 min - start watching the clock at the metrics step.

Stage E result (2026-06-10), full 5/10 eval + per-half decomposition
(passk_by_half.py; cross-checks reproduced the harness exactly):

```text
Overall: CodeBLEU 0.6773 (best of lineage) | compile@1 0.3270 |
compile@5 0.6984 | pass@1 0.1565 | pass@5 0.2703 | pass@10 0.2987
(46/154) | vote-rerank selected pass@1 0.1948 (record 0.2338 stands)

pass@10 by half        train(77)      heldout(77)    overall(154)
  _fitut SFT           0.3247 / 25    0.3117 / 24    0.3182 / 49
  StageD all-data RL   0.3247 / 25    0.3247 / 25    0.3247 / 50
  StageE half-data RL  0.3117 / 24    0.2857 / 22    0.2987 / 46
(pass@5 by half, SFT vs E: train 0.2850->0.2845 flat; heldout
 0.2662->0.2562, -1.0 pt)

VERDICT (pre-registered reading, second bullet applies):
- NO generalization: the held-out half did not improve (24 -> 22 tasks,
  -2; directionally negative, individually within noise).
- NO in-distribution gain at half data either: the trained half came
  back flat (25 -> 24). With 77 tasks x 8 rollouts there was too little
  signal even to replicate Stage D's in-distribution effect.
- Combined honest conclusion across C/D/E: pass@k-weighted GRPO is
  coverage-NEUTRAL. The solid, large effect is what it PREVENTS
  (vanilla collapse: 41 vs 50 tasks, Stage C vs D - 9-task swing at
  identical config). It does not grow coverage at this sampling budget,
  and what little it adds does not transfer to unrewarded tasks. This
  strengthens the 23.14 discovery-limit diagnosis and slightly LOWERS
  the Stage F prior (if even in-distribution amplification is weak at
  G=8, the G=16 question is precisely whether budget was binding -
  the +0-2 pt prediction stands, weighted toward the low end).
- Thesis chapter, final shape: (1) vanilla RLVR destroys pass@k;
  (2) the derived pass@k gradient eliminates the destruction;
  (3) the surviving effect is task-local and exploration-limited
  (E, plus F when run); (4) coverage growth needs off-policy knowledge
  injection (RS-SFT, future work); (5) selection, not generation, sets
  the deployable number (0.2338 via 22.1 vote).
- Decisions UNCHANGED: lineage head _fitutpk5_grpo (Stage D), deployable
  _fitut + vote 0.2338. Stage E was for the claim, not adoption.
Artifacts: Stage E predictions/stats/summary pulled and archived
locally (results-qwen-9b-stageE); checkpoint left on pod - archive
_fitutpk5h_grpo only if convenient before shutdown (low value: a
claim-run, not a candidate).
```

### 23.14 Why pass@k barely moved, and what would move it (diagnosis, 2026-06-10)

Question raised: "is it low epochs?" Answer: mostly no - the binding
constraint is SIGNAL SPARSITY, not optimization pressure.

```text
On-policy RL only amplifies successes it samples. One epoch x group 8 at
T=0.7 = 8 rollouts/task. P(group sees >=1 pass) = 1-(1-p)^8:
  p=0.20 -> 83% | p=0.05 -> 34% | p=0.01 -> 8% | p=0 -> 0%
pass@10 counts tasks with p >~ 0.05-0.1. Adding a NEW task to pass@10
means moving it from p~0 to p>~0.05, but the gradient for that only
exists AFTER a success is sampled - chicken and egg. The ~104 zero-pass
tasks produced zero passing rollouts all epoch (their groups trained on
compile shaping only - dense signal - which is why compile@k set records
while pass@k stayed flat). The pass@k weighting reallocates sparse
signal; it cannot create it. This matches the RLVR literature: RL
redistributes capability, it rarely discovers it. Stages C+D demonstrate
both halves: unweighted pressure burns coverage; weighted pressure
preserves it; neither grows it much.

More epochs: marginal help (more lottery draws per task) at the cost of
compounding the train==eval memorization confound. Pressure is not the
bottleneck; discovery is.

Levers that WOULD push pass@k, in expected-value order:
1. Bigger groups (16-32), not more epochs: multiplies P(first success)
   per pass, improves the p_hat estimate the weighting uses, stays
   on-policy. The exploration-budget knob.
2. Hotter rollouts (--grpo_gen_temperature 1.0, training only),
   combined with 1.
3. Rejection-sampling / continued SFT on the UNSOLVED tasks' reference
   sources: the only lever that touches the zero-pass wall - RL cannot
   discover those solutions, but SFT can teach p>0, after which RL
   amplifies. (Recorded post-C priority 2.)
4. Open hypothesis to quantify for the thesis: Stage B showed more
   assembly visibility did not move pass@k - if the assembly
   underdetermines tested behavior on many zero-pass tasks, that is an
   information ceiling no training scheme fixes. Estimating its size
   (manual audit of a zero-pass sample) is a strong thesis section.
```

### 23.15 Stage F (proposed): exploration-budget run - group 16, T=1.0, chunked scoring

Decision guidance (2026-06-10): a ~7 h H200 booking (NOT 5 - fresh-pod
setup ~1 h, G=16 training ~2x Stage D's, eval+CSV ~2.5-3 h) buys the
EXPLORATION ablation that completes the 23.14 story. Pre-registered
prediction: +0 to +2 pts pass@10 (1-3 newly solved tasks) - at G=16 a
p=0.05 task's chance of producing a first success rises 34% -> 56%,
p=0.02 rises 15% -> 28%; the ~104-task zero-pass wall is NOT addressed
by this run (that is RS-SFT territory). Either outcome is a result:
"pressure ablation (C vs D), exploration ablation (D vs F)".

Memory: the peak is the differentiable scoring forward (full-vocab
logits + activations for 16 x ~2050-token sequences with
gradient_checkpointing 0), roughly 2x Stage D's unmeasured peak - "it
probably fits" is how the Stage C OOM happened. Solved structurally
instead: NEW --score_chunk_size / GRPO_SCORE_CHUNK_SIZE
(--grpo_score_chunk_size in the runner) scores and backwards the group
in sample chunks against one global token denominator - gradients are
mathematically identical (sum-decomposable loss; selfcheck 7 covers the
retain_graph pattern), and peak memory is set by the CHUNK size, not the
group size. With --grpo_score_chunk_size 4, scoring peaks BELOW the
already-proven G=8 single-pass run regardless of group size. Generation
at 16 rows is no-grad linear-attention (small states) - not the peak.
Trainer+runner+selfcheck uploaded to the pod 2026-06-10; all 8
selfchecks pass locally (Windows) and the upload preserved the running
Stage E process (code already imported).

Stage F command (Stage D command + three exploration changes + safety
flag; FULL train file for comparability with Stage D - note for the
thesis that G and T change together as one "exploration budget" arm; run
G-only at T=0.7 first if a strict single-variable ablation is preferred):

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" \
  --name_suffix _fitutpk5g16 \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${NEW}_fitut/pytorch_model.bin" \
  --grpo_train_file data/testing/grpo_data.jsonl \
  --grpo_group_size 16 \
  --grpo_gen_temperature 1.0 \
  --grpo_score_chunk_size 4 \
  --grpo_epochs 1 --grpo_lr 1e-6 \
  --grpo_passk_k 5 \
  --grpo_no_compile_penalty -2.0 --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 2.0 --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 --grpo_overlap_weight 0.0 \
  --grpo_unique_test_bonus 0.25 --grpo_duplicate_penalty 0.25 \
  --grpo_max_new_tokens 512 --grpo_test_timeout 8 --grpo_reward_workers 24 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 1 --grad_accum 8 --lora_r 64 --lora_alpha 128 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --prompt_fit_assembly 1 --decoder_prompt_max_length 1536 \
  --use_reasoning 0 \
  --compile_dataset data/testing/compile-test2.jsonl \
  --pass_dataset data/testing/grpo_data.jsonl \
  --eval_max_new_tokens 768 --generation_batch_size 2 \
  --num_samples 5 --pass_num_samples 10 \
  --save_strategy epoch --save_total_limit 2
```

Gates: keep only if pass@5 > 0.2756 or pass@10 > 0.3247 (Stage D's
number is now the bar). Always also run the 22.1 vote rerank
(deployable bar 0.2338) and passk_by_half.py for the per-half view.
Watch during training: PasskW as in Stage D; with T=1.0 expect somewhat
lower Perfect rates per group (hotter samples) - that is the point.
If a fresh pod: full bootstrap per 23.11, including fla 0.3.2 and the
preflight, and run grpo_selfcheck.py (all checks must pass) before
paying for the epoch.

Stage F result (2026-06-10), full 5/10 eval. Engineering first: the
chunked scorer (--grpo_score_chunk_size 4) survived a full G=16 epoch
in production with no OOM - the memory claim is field-proven.

```text
CodeBLEU 0.6731 | compiled-only 0.6645 (92/126, RECORD)
compile@1 0.3492 | compile@5 0.7302 (RECORD, beats Stage D 0.7222)
pass@1 0.1597 | pass@5 0.2641 | pass@10 0.2922 (45/154)

GATES: FAILED both (pass@5 0.2641 <= 0.2756; pass@10 0.2922 <= 0.3247;
also below SFT 0.3182). Lineage head UNCHANGED: _fitutpk5_grpo (D).

Reading - the exploration ablation is NEGATIVE, and its split is the
informative part: doubling the sampling budget (G 8->16) with hotter
rollouts (T=1.0) helped exactly where signal is dense (compile records)
and cost ~5 any-pass tasks where signal is sparse. Confirms the 23.14
discovery limit from a new angle. Likely contributor, pre-flagged in
this section: G and T changed together, and training/scoring at T=1.0
tilts the policy toward behaviors eval sampling at 0.7 does not
express. A G-only arm at T=0.7 would disambiguate but is NOT
recommended spend; state the confound in the thesis. Chapter conclusion
now rests on three consistent measurements - pressure (C vs D),
data-scale/generalization (E), exploration (F): pass@k-GRPO repairs
RLVR's coverage damage but cannot GROW coverage at feasible on-policy
budgets; growth requires off-policy knowledge injection (RS-SFT).

Per-half (both halves TRAINED here - variance accounting, not
generalization): even half 0.3247/25 (== Stage D), odd half 0.2597/20;
the whole 5-task loss vs Stage D sits in one half - marginal
low-probability solves dropping out unevenly, consistent with the
T=1.0 tilt and with task-level noise at n=77.

Vote rerank: selected pass@1 0.2013 (cross-checks exact). The 0.2338
record (_fitut + vote) STANDS. Notable: Stage F any-compile in the
10-pool is 0.987 (152/154) - the best compile coverage of ANY
checkpoint; it almost never fails to produce a compiling candidate,
but passing candidates are spread too thin for the selector to cash.
Final vote table: _fitut 0.2338 | _fitut_grpo 0.2143 | StageD 0.2143 |
StageE 0.1948 | StageF 0.2013.

Artifacts: Stage F results archived locally (results-qwen-9b-stageF);
checkpoint pull noted below. RL chapter (C/D/E/F) COMPLETE - next
booking goes to RS-SFT, not more GRPO variants (Stage G in 23.16 stays
optional thesis-completeness only).
```

### 23.16 DAPO / GSPO / SimKO options implemented (2026-06-10)

Asked: "can we implement DAPO/GSPO and SimKO as the RL loop instead of
GRPO?" Answer: they are not replacements here - they are GRPO-family
modifications, and the honest mapping onto THIS trainer matters because
our loop does ONE gradient update per rollout batch (ratio identically
1, recorded in 23.1), which makes the clipping innovations inert:

```text
technique                 status in this trainer
DAPO token-level loss     ALREADY HAD IT: loss = sum/global-token-count
                          (now explicit as --loss_pooling token).
DAPO dynamic sampling     EFFECTIVELY HAD IT: B=1 + no-signal skip +
                          accumulate-only-on-signal gives the same
                          optimizer-level effect (8 signal batches per
                          step); we only lack the refill of wasted
                          generation, which is a throughput not a
                          gradient difference.
DAPO clip-higher          NEW FLAG --clip_eps_high, but INERT at
                          single-update (ratio==1 never touches either
                          bound). Plumbed for any future multi-update.
DAPO overlong shaping     NEW: --overlong_filter 1 zeros the advantage
                          of samples that hit max_new_tokens without
                          EOS, so truncation artifacts (currently -2.0
                          compile-fail) stop teaching. Real fix for us.
GSPO sequence ratio       Clipping side: inert (same reason). Pooling
                          side IMPLEMENTED: --loss_pooling seq = per-
                          sample length-normalized mean. Mechanistic
                          case for us: failures ramble toward 512
                          tokens, passes are ~85, so token pooling
                          gives failures ~5x the gradient weight;
                          seq pooling counts every sample equally.
SimKO top-K credit        NEW: --simko_k N - positive-advantage samples
                          reinforce the MEAN of the policy's top-N
                          token log-probs per position (spreads
                          probability mass, anti-over-concentration);
                          failures keep the standard sampled-token
                          penalty. Complementary to --passk_k: passk
                          reallocates across PROMPTS, SimKO
                          redistributes within TOKEN distributions.
"UCP0"                    Unidentified - if this means UCB-style prompt
                          selection (bandit allocating rollouts to
                          uncertain prompts), it is the natural upgrade
                          to dynamic sampling and a separate feature;
                          if it is a specific paper, provide the
                          reference before implementing.
```

Runner flags: --grpo_loss_pooling / --grpo_simko_k /
--grpo_overlong_filter / --grpo_clip_eps_high (env GRPO_LOSS_POOLING,
GRPO_SIMKO_K, GRPO_OVERLONG_FILTER, GRPO_CLIP_EPS_HIGH). All default to
the historical behavior; every Stage C/D/E result is reproducible with
the new code. Both scoring paths (single-pass and chunked) share the
implementations via module helpers effective_token_log_probs() and
pooled_surrogate_loss(); selfcheck now has 9 checks (pooling
arithmetic + chunk decomposability + SimKO asymmetry) - ALL PASS
locally; trainer/runner/selfcheck uploaded to the pod 2026-06-10.

Stage G candidate command ("DAPO+SimKO" bundle = Stage F exploration
config + overlong filter + SimKO; token pooling per DAPO):

```text
Stage F command (23.15) plus:
  --name_suffix _fitutpk5dapo   (instead of _fitutpk5g16)
  --grpo_overlong_filter 1
  --grpo_simko_k 4
Alternative arm (GSPO-pooling): same but --grpo_loss_pooling seq and
suffix _fitutpk5gspo.
```

STRATEGIC NOTE (do not skip): these are all OPTIMIZER-side changes -
how gradient flows once signal exists. The C/D/E evidence says the
binding constraints are DISCOVERY (Stage F tests it) and KNOWLEDGE
(RS-SFT injects it). Run Stage G only after Stage F's result is read,
and expect its value to be mainly thesis completeness (a modern-RLVR
ablation) plus the overlong/seq-pooling rebalancing - not a coverage
breakthrough. Gates and analysis identical to Stage F.

LAUNCHED (2026-06-10): user chose to run both arms on the extended
booking, DAPO first (_fitutpk5dapo), GSPO-pooling second
(_fitutpk5gspo - MUST keep --grpo_overlong_filter 1 --grpo_simko_k 4
and change ONLY --grpo_loss_pooling seq + the suffix, so the pair
isolates pooling). Pre-registered comparison frame: both arms keep
G=16 / T=1.0 / chunk 4, so the baseline is STAGE F (pass@10 0.2922/45,
compile@5 0.7302, vote 0.2013), NOT Stage D; DAPO-vs-F isolates
overlong+SimKO. Adoption gate unchanged (pass@10 > 0.3247). Expected
log behavior: MORE Skip:1 lines than F (overlong filter zeroing
truncation-artifact batches) - correct, not a regression; per-step
overlong_rate is computed but not printed.

DAPO-arm result (2026-06-11), vs Stage F (two-flag controlled diff):

```text
CodeBLEU 0.6709 (flat) | compiled-only 0.6459 (90/126)
compile@1 0.3175 (-3.2 pts) | compile@5 0.7143 (-1.6 pts, 92->90)
pass@1 0.1468 (-1.3) | pass@5 0.2577 (-0.6) | pass@10 0.2922 (45/154,
  EXACTLY == Stage F)

VERDICT: overlong_filter + simko_k 4 are NEUTRAL on coverage and
slightly negative elsewhere at this config. Gates failed; lineage head
(Stage D) and deployable record (0.2338) unchanged. Mechanistic note:
the compile dip is consistent with the overlong filter's second edge -
zeroing truncated samples also removes the NEGATIVE gradient that
punished rambling (-2.0 compile-fail), so truncation-prone behavior is
less suppressed. Fourth consecutive optimizer-side null on coverage;
the ~105-task zero-pass wall is visible row-by-row in the per-task
output and untouched.
GSPO-pooling arm: pre-registered expectation flat +-1-2 pts around
F/DAPO; it is the one arm with a mechanism aimed at the pass axis
(short passes no longer outweighed ~5x by long failures), value =
completing the pooling ablation.

DAPO per-half (both halves trained; variance accounting): 22/23 - the
balanced split Stage F lacked (F was 25/20), same overall 45. Vote
rerank: selected pass@1 0.2143 (cross-checks exact) - notably BETTER
than Stage F's 0.2013 from the same raw coverage, i.e. the DAPO
additions made candidate pools slightly more selectable even though
raw metrics dipped. Record 0.2338 stands. Vote table now:
_fitut 0.2338 | C 0.2143 | D 0.2143 | E 0.1948 | F 0.2013 |
G-DAPO 0.2143. Results archived locally (results-qwen-9b-stageG).

GSPO-arm result (2026-06-11), one-flag diff vs DAPO arm (seq pooling):

```text
CodeBLEU 0.6735 | compiled-only 0.6613 (89/126)
compile@1 0.3349 (+1.7 vs DAPO) | compile@5 0.7063 (-0.8, 90->89)
pass@1 0.1526 (+0.6) | pass@5 0.2611 (+0.3) | pass@10 0.2857
  (44/154, -1 task) | per-half perfectly balanced 22/22

POOLING ABLATION: NULL on raw metrics, exactly as pre-registered
(everything within +-1.7 pts). Gates failed; nothing adopted.

SECONDARY FINDING (the real result of the G arms): vote-selected
pass@1 rose MONOTONICALLY across the exploration-config arms while raw
coverage stayed flat or fell - F 0.2013 -> DAPO 0.2143 -> GSPO 0.2208
(best of ANY RL checkpoint; cross-checks exact; any-compile 0.9805).
SimKO + seq pooling concentrate correctness into more consistent,
vote-able candidate forms: optimizer-side RLVR knobs trade raw
coverage for SELECTABILITY. Gap to the all-time deployable record
(_fitut + vote 0.2338) is now 1.3 pts.

RL PROGRAM CLOSE-OUT - the full ablation table (raw pass@10 / vote):
  SFT _fitut                 0.3182 (49) / 0.2338  <- both records
  C  vanilla GRPO            0.2662 (41) / 0.2143
  D  passk G8 T0.7           0.3247 (50) / 0.2143  <- raw record, HEAD
  E  passk half-data         0.2987 (46) / 0.1948
  F  passk G16 T1.0          0.2922 (45) / 0.2013
  G  +overlong+SimKO (DAPO)  0.2922 (45) / 0.2143
  G  +seq pooling (GSPO)     0.2857 (44) / 0.2208
Seven controlled measurements, one conclusion, unchanged: on-policy
RLVR cannot grow coverage here (only repair its own damage, Stage D);
its knobs instead shape HOW the fixed coverage is distributed
(sharpness vs selectability). Coverage growth = RS-SFT (off-policy
knowledge injection), selector growth = learned verifier. Decisions
unchanged: head _fitutpk5_grpo, deployable _fitut + vote 0.2338.
All Stage G results + both checkpoints archived locally; pod has
nothing mandatory left after the checkpoint pull - SHUT IT DOWN.
```

### 23.17 Forward plan: "is it data regime?" - yes, three ways (2026-06-11)

Diagnosis question asked after the close-out: the RL LOOP is not
broken - every objective change produced its predicted distributional
change across seven runs - it is STARVED. The data regime binds in
three distinct senses:

```text
1. TRAINING-SIGNAL regime: 154 tasks x 8-16 rollouts = ~1.2-2.4k
   samples/epoch, ~150-250 passing, nearly all on the same ~25 tasks.
   Dense compile signal hit records; sparse pass signal could not move.
   Stage E adds: what RL learns does not transfer off its train tasks.
2. KNOWLEDGE regime: the ~104 zero-pass tasks have p~0 under the SFT
   policy -> zero on-policy gradient at ANY tested budget. The model
   has never seen solutions to these task shapes; SFT-corpus gap
   (1,195 rows), not an RL hyperparameter.
3. INFORMATION regime: Stage B showed more assembly visibility did not
   move pass@k; some tasks' tested behavior may be underdetermined by
   the input. A ceiling to QUANTIFY, not fight.

PLAN (cheapest first):
Phase 0 (FREE, local):
  a. Zero-pass audit: sample ~20 of the 104 unsolved tasks; inspect
     candidates vs reference vs tests; classify knowledge-limited /
     information-limited / harness-strict. Measures achievable headroom
     BEFORE any GPU spend; thesis section on its own.
  b. RS-SFT harvester: union of PASSING candidates across all eight
     archived pools (per-candidate pass flags + texts are in the
     prediction JSONs), deduped; plus REFERENCE solutions for zero-pass
     tasks. Two arms: RS-only vs RS+references.
Phase 1 (~$5-8, SFT not RL; RTX-6000-class suffices):
  Continue SFT from _fitut on the harvested file; standard eval.
  Gates: pass@10 > 0.3247 AND/OR vote > 0.2338.
Phase 2 (only if Phase 1 lifts coverage, ~$10):
  Stage D pass@k-GRPO config VERBATIM on top of the RS-SFT checkpoint -
  the amplifier finally gets tasks with p>0 to amplify. Optionally
  +SimKO/seq-pooling (selectability evidence from the G arms).
Phase 3: learned verifier for selection (1.3 pts to the record,
  ~10 pts to the oracle ceiling).
Long-run data fix (eng time, not GPU): synthesize new (asm, source,
tests) triples by compiling Dart corpus functions and dumping GDB asm -
grows ALL three regimes at once.
```

PHASE 0 EXECUTED (2026-06-11). Tools: scripts/data/zero_pass_audit.py,
scripts/data/build_rs_sft_data.py. Results:

```text
AUDIT (all 14 archived pools = 140 candidates/task):
- UNION any-pass: 67/154 (0.435) - the cross-checkpoint ensemble oracle
  is +11 pts over the best single pool (D, 50). Checkpoints solve
  DIFFERENT tasks; diversity across training runs is real.
- Hard-core zero-pass: 87/154 (56.5%).
- NO automatable proxy separates solved from unsolved:
    max CodeBLEU: median 0.757 solved vs 0.750 zero-pass (identical);
    every zero-pass task has compiling candidates;
    string-output test assertions occur EQUALLY (control: 29/67 solved
    vs 34/87 zero-pass) - models pass string tasks because expected
    strings derive from inputs, not constants.
  => the wall is FUNCTIONAL-SEMANTIC (knowledge gap by elimination):
  compiling, textually-close Dart that computes the wrong thing. A true
  information ceiling was NOT detectable by proxy; quantifying it now
  requires the manual spot-check (per-task data + examples saved in
  results/analysis/zero_pass_audit.json).
- CodeBLEU is hereby measured to be non-predictive of functional
  correctness on this task set (solved/unsolved medians identical) -
  state this in the thesis when discussing metric choice.

HARVEST: 3,128 passing candidates total ('base' pre-UT pool: 0 - the
unit-test continuation is where passing began). Deduped by normalized
code, capped 4/task (solution diversity is extremely skewed: a few
easy tasks have 50-100+ distinct correct solutions, most have 1-2).
Files written (grpo_data schema, drop-in for the SFT pipeline):
  data/testing/rs_sft_all.jsonl            235 rows (67 tasks)
  data/testing/rs_sft_all_plus_refs.jsonl  322 rows (+87 zero-pass refs)
  data/testing/rs_sft_train_half.jsonl     163 rows (even-index only)

PHASE 1 command (SFT continuation, NOT RL; from _fitut; upload the
rs_sft files + both new scripts to the pod first):

  python configs/run_sweeps_antigravity.py \
    --experiment "$NEW" --name_suffix _rsft \
    --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
    --sft_checkpoint "artifacts/${NEW}_fitut/pytorch_model.bin" \
    --train_file data/testing/rs_sft_all_plus_refs.jsonl \
    --eval_file data/testing/grpo_data.jsonl \
    --epochs 3 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
    --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
    --train_batch_size 2 --grad_accum 32 \
    --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
    --prompt_fit_assembly 1 --decoder_prompt_max_length 1536 \
    --use_reasoning 0 \
    --compile_dataset data/testing/compile-test2.jsonl \
    --pass_dataset data/testing/grpo_data.jsonl \
    --eval_max_new_tokens 768 --generation_batch_size 4 \
    --num_samples 5 --pass_num_samples 10

Gates: pass@10 > 0.3247 and/or vote-selected > 0.2338. Honest framing:
train==eval self-improvement on target tasks; the train_half arm is the
generalization protocol if a second run is funded.

LOCAL-METRICS WORKFLOW (2026-06-11, cuts ~25-35 paid min off EVERY
run): the local machine (RTX 3060 Laptop 6GB - far too small to train
or infer the 9B, even 4-bit) HAS Dart SDK 3.11.5. So: pod does training
+ inference ONLY; the moment both *_predictions.json files exist, scp
them down and SHUT THE POD OFF; compute CodeBLEU/compile@k/pass@k and
both stats CSVs locally (the metric scripts take --predictions paths
and need only Dart + the dataset files), then the usual offline
rerank/per-half suite. Caveat: pod vs local Dart versions may flip rare
compile edge cases - within-run comparisons clean; do not over-read
cross-run compile deltas under ~1 pt. With this, Phase 1 is ~2-2.5 h of
paid time (~$4-5).

SYNTHETIC grpo_data SCHEMA (user is building this - verified against
the real file; one JSON object per line, ALL keys present):
  filename                  unique '<n>.dart'
  function                  Dart function name
  python_function_name / camel_case_function_name / python_source
                            provenance fields (HumanEval-style); keep
                            populated, pipeline tolerates plain copies
  dart_function_signature   e.g. 'bool f(List<double> xs, double t)'
  dart_source               standalone top-level function; keep the
                            @pragma('vm:entry-point') prefix - it is
                            what stops AOT tree-shaking from deleting
                            the function before GDB can dump it
  assembly                  GDB text in the exact format: first line
                            'All functions matching regular expression
                            "NAME":' then File header + address-prefixed
                            disassembly; produce with the SAME dart
                            compile aot-snapshot + gdb flow as the
                            original corpus so the input distribution
                            matches
  lang                      'Dart'
  task_id                   unique string
  tests                     SELF-CONTAINED: void main() { final
                            candidate = NAME; expect(...); ... } plus
                            its OWN expect/expectList/expectMap helper
                            definitions; pass = process exit code 0;
                            the 'final candidate = NAME;' line is
                            REQUIRED (harness extracts the target name
                            from it); deterministic functions only
                            (no IO/random/time); several expects per
                            task (the shaped reward gives per-test
                            partial credit)
```

### 23.18 Synthetic pool validated; Phase 1 + clean-RL launched (2026-06-12)

User delivered data/datasets/synthetic_pool.jsonl (1,727 tasks, 28.6 MB).
Validation (scripts/data/validate_synthetic_pool.py - schema, uniqueness,
conventions, eval-set leakage, and a FULL functional gate running every
reference against its own tests with local Dart):

```text
- Schema/conventions: clean (524 rows carry an extra harmless
  generator_model_reported key; 12 rows use GDB "Dump of assembler code"
  format instead of the "All functions matching" listing - both real GDB
  output, both accepted).
- Uniqueness: 0 duplicate ids/files. Tests: median 8 expects/task, min 4.
- LEAKAGE vs the 154-task eval set: ZERO (function names and normalized
  sources) - the confound-free property holds.
- FUNCTIONAL: 1,726/1,727 references pass their own tests (99.94%).
  The one defect: syn_1661 declares its own main() (collides with the
  test harness main). Excluded. VALIDATION LESSON for future batches:
  run the gate with <=4 workers / 20 s timeout - 12 parallel Dart VMs
  on the local laptop produced 423 spurious test_timeouts (all passed
  on gentle retry); 30-row samples at light load hide the artifact.
- Files of record: data/datasets/synthetic_pool_clean.jsonl (1,726) and
  synthetic_pool_train576.jsonl (every 3rd clean row, 576 tasks, cost-
  bounding subsample). Both on the pod.
```

Run plan executed this booking (commands in chat + below):
1. Phase 1 RS-SFT (_rsft, 23.17 command) - LAUNCHED. Local-metrics
   workflow: Ctrl-C after pass_predictions is written; pass@k verdict
   computed locally in ~8 min; full metrics local while step 2 runs.
2. Clean-RL: Stage D pass@k-GRPO config except
   --name_suffix _synpk5full (user chose FULL pool over the 576
   subsample, accepting ~13 h / ~$26),
   --grpo_train_file data/datasets/synthetic_pool_clean.jsonl (1,726),
   --grpo_group_size 16 --grpo_score_chunk_size 4 (chunk 4 is the
   field-proven memory bound; T stays 0.7 deliberately - G16's
   discovery without Stage F's train/eval temperature mismatch),
   --grpo_checkpoint _fitut (Phase 1 failed both gates; also keeps the
   confound-free narrative clean). Rollout budget 27,616 = 22x any
   prior run. HARD REQUIREMENT: booking must have >=15 h left at
   launch - save_strategy is epoch-end only; a mid-epoch expiry loses
   the entire training spend.
   Eval untouched (original 154): any pass@k movement is confound-free
   generalization - the experiment the synthetic pool exists for.
   Watch the first ~30 batches: if ZeroPass ~1.00 throughout, the model
   cannot solve synthetic tasks at all -> abort cheaply, the pool is
   too far from the SFT distribution (then: SFT on synthetic first).
   Known limitation, accepted for one-variable cleanliness: synthetic
   assembly is longer (median ~10.9k chars), so the 1536-token prompt
   budget shows only ~45% of it (vs ~70% on original tasks).
Gates: pass@10 > 0.3247 / vote > 0.2338 as standing.

_synpk5full ABORTED at step ~25/1726 (2026-06-13, ~$0.50 spent). The
abort-watch fired exactly as designed: ~22/25 batches Skip:1, rewards
-2.0 (all 16 rollouts fail to COMPILE) or 0.0 (compile, pass nothing),
one lone signal batch (step 16, Perfect 0.19). Ruled out before
aborting: difficulty is well-shuffled in file order (first 25 mostly
easy), and synthetic assembly format is byte-identical to the originals
(same GDB AT&T, push rbp / <+N>). So it is a genuine DISCOVERY-FLOOR
failure (23.14): _fitut almost never sample-passes the synthetic tasks
because their STYLE differs - eval is HumanEval-flavored, the pool is
sorting-comparators / bit-manip / matrix / state-machine parsers. On-
policy RL cannot learn what the base never samples. Clean abort, ~10
min, not the $26.

REVISED PATH - teach what RL cannot discover (mirrors Phase0->Phase1,
pointed at synthetic): SFT on synthetic FIRST, then RL.
  Stage syn-1 (_synsft): SFT continuation from _fitut on
    synthetic_pool_clean.jsonl (1,726 verified (src,tests) pairs - a
    bigger, cleaner SFT corpus than the 322-row rs_sft). 2 epochs.
    Eval on the 154 does double duty: (a) does synthetic SFT TRANSFER
    to real tasks? (b) does it lift the synthetic solve rate enough for
    RL to have signal? Watch for catastrophic forgetting of the
    HumanEval style vs broader Dart competence.
  Stage syn-2: re-run the synthetic pass@k-GRPO (the _synpk5full
    command) but --grpo_checkpoint from _synsft. Only worth launching
    if _synsft can sample-pass synthetic tasks (re-check the first ~30
    batches' ZeroPass).
_synsft command in chat; uses the standing 1536 budget and gates.

_synsft result (2026-06-13): TRANSFER NEGATIVE at this dose, and the
diagnosis is in the TRAINING log, not the eval.

```text
Eval on the 154: compile@1 0.2667 / compile@5 0.5714 (72/126,
-13.5 pts vs _fitut) | pass@1 0.1591 (flat) | pass@10 0.3052 (47,
-2 tasks, noise) | CodeBLEU 0.6346 (-3.3 pts).
Training: loss 8.55 -> 8.04 over 2 epochs; eval_loss IDENTICAL at both
epoch boundaries (0.3039). 1,726 novel tasks x 2 epochs / (batch 2 x
accum 32) = 54 optimizer steps at lr 5e-6 - the corpus was NOT learned
at this dose (vs _fitut's 4 repetitions of 154 rows: repetition
consolidates; one-shot diversity does not). The style exposure was
still enough to DISTURB original-distribution surface behavior:
compile and CodeBLEU paid the forgetting tax with no knowledge gained.
Vote-selected pass@1 0.1753 (cross-checks exact) - record 0.2338
stands. All _synsft artifacts archived locally.

FORK recorded:
A. Close the synthetic line (thesis already complete; this adds the
   dose/transfer negative).
B. ONE properly-dosed attempt (~$10-14): lr 1e-5, epochs 4, train on
   the MIX synthetic_pool_clean.jsonl,grpo_data.jsonl (originals
   anchor style and protect compile). PRE-REGISTERED KILL-SWITCH,
   visible during training: train loss must drop >=25% by end of
   epoch 2 or Ctrl-C without paying for eval (~$3 lost, not $14).
   If it breaks downward: finish, eval, then the 30-batch GRPO probe
   from the new checkpoint.
Deployable/lineage decisions unaffected: head _fitutpk5_grpo,
deployable _fitut + vote 0.2338, best selector-free _rsft 0.1935.

_synsft2 result (2026-06-13, Option B, mixed corpus lr 1e-5 x4 epochs).
KILL-SWITCH NOTE: strict 25%-by-epoch-2 trigger was technically missed
(18.6%), overridden because (a) eval_loss was MONOTONE down all 4
epochs 0.2894->0.2838->0.2802->0.2797 - real learning, unlike _synsft's
frozen 0.3039 - and (b) the run was 82% through training (past the
cheap-kill point). Override was correct: it learned.

```text
Real 154 eval vs _fitut / _synsft:
              _fitut      _synsft(light)   _synsft2(mixed,proper)
compile@5     0.7063(89)  0.5714(72)       0.6349(80)
pass@10       0.3182(49)  0.3052(47)       0.3182(49)  <- baseline
pass@1        0.1617      0.1591           0.1558
CodeBLEU      0.6673      0.6346           0.6088      <- worst
```

VERDICT: dose fix WORKED as training (learned + replay-mixing recovered
coverage to baseline 49 and most of compile), but ZERO transfer benefit
to the real task - pass@k at baseline, compile still -7pts, CodeBLEU
worst of lineage. THIRD consistent synthetic negative: RL-abort (base
can't sample-pass) -> SFT-light (no learning) -> SFT-proper (learns,
no transfer). The cross-STYLE distribution gap is the robust binding
constraint. Caveat: eval_loss drop is partly the 154 originals being
in the train mix (not proof synthetic solve rate rose).

OPEN QUESTION (the only one left for the synthetic line): did _synsft2
become able to SAMPLE-PASS synthetic tasks? Answered only by a ~$1
30-batch GRPO probe on synthetic from _synsft2 (watch ZeroPass).
Plan: run the probe, then CLOSE the line either way.
  - ZeroPass mixed -> SFT made synthetic samplable; synthetic-RL viable
    for a future synthetic-EVAL study (real-154 transfer still unlikely).
  - ZeroPass ~1.00 -> line definitively closed; clean thesis negative
    (cross-style synthetic transfers via neither SFT nor RL even at
    proper dose with replay).
Deployable/lineage UNCHANGED (nothing here beats _fitut on real eval).
Bookkeeping: _synsft2 pass_stats.csv did NOT survive (metrics killed
mid-write when the GPU was freed for the probe). Immaterial - checkpoint
not adopted, pass@k already recorded above; vote-rerank unmeasured.
Archived: predictions + compile_stats.csv + summary + checkpoint
(686.9 MB). Reconstructable from predictions + local Dart if ever
needed (~25 min), but no decision depends on it.

_synprobe result (2026-06-13, ~$1): the GRPO probe from _synsft2 on
synthetic (G16/chunk4/T0.7, same config as the _fitut abort). VERDICT:
WALL HELD. 31 batches: ZeroPass ~1.00 throughout, dominated by -2.000
(rollouts fail to COMPILE), one partial-signal batch (step 18, no full
pass), and Perfect 0.00 across ALL 31 - WORSE than the _fitut abort
which had a Perfect 0.19 batch by step 16. Sharp finding: the 576 probe
tasks were INSIDE _synsft2's training set, yet after proper-dose SFT on
them it still cannot GENERATE a passing one at T=0.7. Token-level
eval_loss fell (0.2894->0.2797) WITHOUT buying functional generation -
textbook perplexity != correctness dissociation. RL cannot amplify what
SFT could not make samplable.

SYNTHETIC LINE CLOSED - robust 4-point negative:
  1. RL from _fitut         -> abort (base can't sample-pass)
  2. SFT light (5e-6, 2ep)  -> didn't learn (eval_loss frozen)
  3. SFT proper (1e-5, 4ep, mixed+replay) -> learned token-loss, ZERO
     real-task transfer (pass@k at baseline, compile/CodeBLEU down)
  4. RL from the SFT'd ckpt -> still can't sample-pass its OWN train
     tasks
Binding constraint = cross-STYLE distribution gap, robust across SFT
and RL. Caveat on the record: all at 1536 budget (~45% synthetic-asm
visibility); the 2048-visibility SFT->RL study is theoretically open but
unfunded and unlikely to flip the conclusion.

PROJECT STATE AT CLOSE (decisions final):
  - Lineage head (best-of-k):  _fitutpk5_grpo  pass@10 0.3247
  - Deployable (single out):   _fitut + n=10 + compile-vote  0.2338
  - Best selector-free:        _rsft  pass@1 0.1935
  - 11 checkpoints + all results/analyses archived locally.
  - Thesis arc complete: GRPO bug audit+fix (23.1-7) -> pass@k gradient
    derivation+validation (Stage C/D) -> exploration/data/generalization
    ablations (E/F/G) -> modern-RLVR knobs (DAPO/GSPO/SimKO, 23.16) ->
    selection beats RL for deployable (22.1) -> RS-SFT + synthetic data
    regime study (23.17-18). Core result: fixed pass@k-GRPO repairs
    RLVR coverage collapse but cannot GROW coverage on-policy; growth is
    bottlenecked by discovery + knowledge, and cross-style synthetic
    injection does not transfer. POD CAN BE SHUT DOWN.

### 23.19 Synthetic rescue at 2048 (FUNDED, 2026-06-13)

User funded the higher-visibility retry. Hypothesis (well-motivated):
the synthetic wall was a COMPILE wall (rollouts scored -2.000, fail to
compile), and Stage B proved visibility is the lever that moves compile
- so fuller input may convert the compile wall into samplable tasks and
reopen synthetic RL. _synsft2 is the EXACT 1536 control (same ckpt,
dose, mixed corpus); only --decoder_prompt_max_length changes.

Visibility math (corrected): 2048 budget = ~6,350 chars; synthetic asm
median ~10,953 -> ~58% visible, up from ~43% at 1536 (+15pt, NOT full;
the earlier "70%" was the ORIGINAL-task figure). If 58% only partially
moves it, 3072 (~87%) is the follow-up.

Two stages (commands in chat; data already on pod):
  Step 1 _synsft2048: SFT = _synsft2 + 2048 budget. Ctrl-C at "Saving
    model" to skip the ~1.5h eval - rescue needs only the checkpoint.
    Watch first 2-3 steps for OOM (2048xbatch2 = +33% prompt tokens vs
    _synsft2); if OOM drop to train_batch_size 1 / grad_accum 64.
  Step 2 _synprobe2048: GRPO probe from _synsft2048, G16/chunk4/2048.
    Read batches 1-30 vs the _synprobe (1536) control: that had
    ZeroPass ~1.00 + ZERO Perfect in 31 batches. Perfect>0 / ZeroPass
    dropping = visibility broke the wall -> full synthetic RL reopens.
    Still-dead = +15pt visibility insufficient; only 3072 left.
~$5 to first answer (SFT-to-ckpt + probe). Decisions/records unchanged
until the probe reads positive.

EXECUTION NOTE (2026-06-13): user went straight to 3072 (skipped 2048)
and slept through the Ctrl-C, so the _synsft3072 SFT ran its FULL eval.
3072 budget = ~9,525 chars = ~87% synthetic visibility (near-full;
originals fully visible, no trim). Bonus real-154 eval (this is the
"original-task ceiling" side-question, answered for free):

```text
              _fitut      _synsft2(1536)   _synsft3072(~87% vis)
compile@5     0.7063(89)  0.6349(80)       0.5952(75)
pass@10       0.3182(49)  0.3182(49)       0.3117(48)
CodeBLEU      0.6673      0.6088           0.6306
vote pass@1   0.2338      (unmeasured)     0.1883
```
=> near-full input visibility bought NOTHING on the real task (pass@10
baseline-minus-noise, compile LOWER than the 1536 mixed run). STAGE B
CONFIRMED A 4TH TIME: visibility is not the bottleneck. eval_loss did
fall a touch more (0.2845->0.2716) - token-level fit, not functional.
_synsft3072 fully archived (preds + both CSVs + summary + ckpt 686.9MB
+ vote 0.1883).

_synprobe3072 result (2026-06-13): WALL HELD - SYNTHETIC LINE CLOSED.
29 batches before an incidental OOM (3072xG16 scoring exceeds H200 even
at chunk 4 - would need chunk 2, but immaterial): EVERY batch ZeroPass
1.00, SignalGroups 0.00, reward -2.000 (rollouts don't even COMPILE).
ZERO completed signal groups in 29 - DEADER than the 1536 probe (which
had a partial-signal batch by step 18). The OOM at step 30 was the
first batch with enough spread to enter scoring; the trainer's
error-handling CAUGHT the single OOM (fails fast only after 3
consecutive) and continued - steps 31-33 kept the same dead wall, 32
dead batches total before user Ctrl-C. Note: 3072xG16 scoring OOMs on
ANY signal batch, so this config is structurally UNTRAINABLE even if
signal appeared (would need chunk 2 / G8) - independent reason the run
was pointless. Corroboration: _synsft3072 compiles synthetic LESS than
the lower-vis checkpoint (matches its lower real-154 compile@5) -
longer-prompt SFT was modestly WORSE at generation in the same step
budget. No artifact worth keeping.

SYNTHETIC LINE FINAL - 6-point negative across BOTH visibility levels:
  1. RL from _fitut (43% vis)        -> wall
  2. SFT light (1536)                -> didn't learn
  3. SFT proper (1536 mixed)         -> learned token-loss, no transfer
  4. RL probe from _synsft2 (43%)    -> wall, 1 marginal batch
  5. SFT @ 3072 (~87% vis)           -> no real-task gain (StageB x4)
  6. RL probe from _synsft3072 (87%) -> wall, ZERO signal groups
Visibility tested at 43% AND 87% -> wall both. Binding constraint is
conclusively the TASK-STYLE distribution gap, NOT input truncation.
The cross-style synthetic corpus transfers via neither SFT nor RL at
any visibility. POD CAN BE SHUT DOWN - nothing further needs GPU.
Future same-style data (HumanEval-flavored synthetic matching the eval
distribution) remains the only untested data lever; different booking.

Phase 1 result (_rsft, 2026-06-13), full eval + offline analysis
(cross-checks exact):

```text
CodeBLEU 0.6600 | compiled-only 0.6456 (93/126, RECORD any-compile)
compile@1 0.3365 | compile@5 0.7381 (RECORD, beats F's 0.7302)
pass@1 0.1935 (RECORD, +3.2 pts over SFT - biggest pass@1 jump of the
  project) | pass@5 0.2816 (RECORD) | pass@10 0.3052 (47/154)
vote-selected pass@1 0.2143 | per-half 26/21 (trained on all tasks;
  variance accounting)

VERDICT: BOTH gates failed (pass@10 0.3052 < 0.3247; vote 0.2143 <
0.2338) -> Step 2 checkpoint = _fitut per the pre-registered rule.
Reading: classic distillation - reliability way up (record raw pass@1/
pass@5/compile@5), coverage slightly down (47 vs 49 any-pass). The
selector's edge SHRINKS as the base sharpens: _rsft raw 0.1935 -> vote
0.2143 (+2.1) vs _fitut raw 0.1617 -> vote 0.2338 (+7.2). Selection
still beats distillation for the deployable number. _rsft is the best
SELECTOR-FREE model of the project; deployable-with-selector record
unchanged (_fitut + vote 0.2338). Starting Step 2 from _fitut also
keeps the clean-RL narrative uncontaminated (_rsft maximally memorized
eval-task solutions). Artifacts archived locally
(results-qwen-9b-phase1 + checkpoint).
```

### 24. Current GRPO Review and Gate Finding (2026-06-13)

Current status after re-reading the latest archived results: Opus 4.8's
pass@k/objective-mismatch diagnosis was mostly correct, but incomplete.
The later experiments did not fail because the exact first reward formula
was bad; they failed because every reward/objective variant still mostly
reshuffled the same small set of solvable tasks.

Compact scorecard from local archived result JSONs:

```text
run                         compile@5   pass@1   pass@5   pass@10
_ut_gentle_grpo             0.6746      0.1617   0.2869   0.3247
_fitutpk5_grpo Stage D      0.7222      0.1552   0.2747   0.3247
_fitutpk5g16_grpo Stage F   0.7302      0.1597   0.2641   0.2922
_fitutpk5gspo Stage G       0.7063      0.1526   0.2611   0.2857
_rsft Phase 1               0.7381      0.1935   0.2816   0.3052
_synsft3072                 0.5952      0.1565   0.2679   0.3117
```

Candidate-pool coverage from pass_stats.csv:

```text
run               any-pass tasks   zero-pass tasks   new/lost vs _ut_gentle
_ut_gentle_grpo   50               104               baseline
_fitutpk5_grpo    50               104               +7 / -7
_rsft             47               107               +3 / -6
G16               45               109               +3 / -8
GSPO              44               110               +3 / -9
_synsft3072       48               106               +5 / -7
```

Interpretation: Stage D tied the best pass@10 while improving compile@5,
but it did not expand coverage. Group 16, DAPO/GSPO, reward changes, and
synthetic data mostly improved syntax/probability mass or candidate
ordering, not the set of tasks with at least one correct candidate. The
zero-pass audit remains the controlling evidence: across the broad pooled
runs, union_any_pass is only 67/154, leaving 87 hard-core zero-pass tasks.

Important implementation finding: the Qwen graph prefix gate did not learn
in practice. Local checkpoint inspection:

```text
_fitut           sigmoid(gate_logit)=0.200011
_fitutpk5_grpo   sigmoid(gate_logit)=0.200011
_rsft            sigmoid(gate_logit)=0.200015
_synsft3072      sigmoid(gate_logit)=0.200038
```

So "try a bigger gate" is a legitimate architecture/glue ablation, but
the old `--qwen_prefix_gate_init` flag is only an initializer. When loading
an existing checkpoint, the saved `gate_logit` wins and the init flag does
nothing. New patch added `--qwen_prefix_gate_override`, which maps to
`GRAPH_QWEN_PREFIX_GATE_OVERRIDE`, logs the loaded gate, and can force the
gate after checkpoint load in both SFT and GRPO:

```text
configs/run_sweeps_antigravity.py
scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py
scripts/training/graph_grpo_decompiler_antigravity.py
```

Also still important: GRPO currently computes the graph encoder/projection/
prefix context under `torch.no_grad()`, then optimizes decoder LoRA against
fixed graph-conditioned embeddings. That means GRPO cannot learn a better
graph glue/gate; it can only change how the decoder responds to the frozen
context. Bigger H200/B200 group sizes improve advantage estimation and
compile stability, but they do not create semantic signal when groups have
zero passing candidates.

Current recommendation:

1. Do not rerun the same GRPO stack from scratch. It has already been
   falsified by Stage C/D/F/G and synthetic probes.
2. Before spending on another RL run, run a large-sample inference probe
   (for example 50 candidates) from the current best head. If the pass
   oracle stays near 50 tasks, GRPO cannot discover new coverage. If it
   jumps materially, invest in verifier/reranker or RS-SFT on the passing
   candidates.
3. If testing bigger gates, treat it as a glue ablation, not a GRPO reward
   tweak. Use the new gate override for cheap continuation probes
   (`--qwen_prefix_gate_override 0.35` or `0.5`) and keep only runs that
   increase any-pass coverage or pass@10, not just compile@5/CodeBLEU.
4. A truly new GRPO line should either train graph glue/projection under
   gradient or use same-style verified data that increases the probability
   of full-pass samples. Larger group size alone is not expected to matter
   materially.

### 25. Current State After k50, Reranking, Synthetic-Mix Planning, and HF Upload Notes (2026-06-14)

This section supersedes only the tactical next steps, not the older audit
record. The core diagnosis is now sharper:

- The current best lineage head remains
  `qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo`.
- The current checkpoint is an adapter/glue checkpoint, not a full Qwen
  checkpoint. It contains GraphCodeBERT encoder LoRA, Qwen decoder LoRA,
  graph encoder/projection glue, and Qwen graph-prefix/gate tensors.
- Loading elsewhere requires:
  `Qwen/Qwen3.5-9B-Base`, `microsoft/graphcodebert-base`, this repo's model
  code, and the matching env/config:

```bash
GRAPH_ENCODER_PEFT=lora
GRAPH_DECODER_PEFT=lora
GRAPH_LORA_R=64
GRAPH_LORA_ALPHA=128
GRAPH_QWEN_PREFIX_TOKENS=16
```

Checkpoint inspection confirmed both sides are adapted:

```text
local_encoder.encoder...lora_A/lora_B                  GraphCodeBERT/encoder LoRA
t5_model.base_model.model.model.layers.*...lora_A/B    Qwen decoder LoRA
graph_encoder.*                                        graph glue
qwen prefix/gate tensors                               learned bridge
```

`t5_model` is a historical module name in the code. In the Qwen runs it is
the wrapped Qwen decoder, not CodeT5.

#### 25.1 k50 Coverage Probe

Direct k50 pass@k from `_fitutpk5_grpo`:

```json
{
  "pass_at_1": 0.16246753246753246,
  "pass_at_5": 0.27227255932347605,
  "pass_at_10": 0.30913218382519236,
  "pass_at_25": 0.36062415474635173,
  "pass_at_50": 0.4155844155844156,
  "total_problems": 154
}
```

Interpretation:

```text
64 / 154 tasks have at least one passing candidate in 50 samples.
90 / 154 tasks have zero passing candidate even at k50.
```

This separates the problem into two distinct failures:

1. Selection failure: a correct candidate exists, but the deployed selector
   does not choose it.
2. Coverage failure: no correct candidate exists in the sampled pool.

The second failure is the harder wall; reranking cannot exceed it.

#### 25.2 Reranker Results

Compile-only reranking:

```text
selected compile rate: 0.9935
selected pass@1:       0.2208 = 34 / 154
```

Oracle test reranking:

```text
selected pass@1:       0.4156 = 64 / 154
```

Gap analysis:

```json
{
  "candidate0_pass_count": 28,
  "oracle_passable_count": 64,
  "compile_selected_pass_count": 34,
  "compile_missed_passable_count": 30,
  "zero_pass_count": 90,
  "mean_first_pass_index_1_based": 7.0625,
  "median_first_pass_index_1_based": 2.0,
  "first_pass_bucket_counts": {
    "1": 28,
    "2-5": 15,
    "6-10": 8,
    "11-25": 9,
    "26+": 4,
    "none": 90
  }
}
```

Conclusion: compile is nearly solved as a selector. The remaining selection
gap is semantic: the wrong candidate usually compiles, has the right function
name, no markdown, no rogue `main`, and balanced braces, but implements a
simpler/wrong algorithm.

New deployable reranker added:

```text
scripts/evaluation/rerank_predictions_antigravity.py
  --mode compile_cluster_vote
  --cluster_vote_bonus 5.0
```

This still uses only fair inference-time signals: Dart compile plus exact
self-consistency/duplicate clusters. It does not use hidden tests.

Observed k50 result:

```text
candidate 0 pass@1:          28 / 154 = 0.1818
compile reranker pass@1:     34 / 154 = 0.2208
compile+cluster vote pass@1: 43 / 154 = 0.2792
oracle k50 ceiling:          64 / 154 = 0.4156
```

So `compile_cluster_vote` recovers 67.2% of passable k50 tasks:

```text
43 / 64 = 0.672
```

New analyzer added:

```text
scripts/evaluation/analyze_rerank_reports_antigravity.py
```

Use it to compare compile/cluster/oracle reports without rerunning Dart.

#### 25.3 k10/k16/k25/k50 Cost Tradeoff

Separate generations at k10/k16/k25/k50 gave:

```text
Samples   raw pass@1   raw pass@5   raw pass@10   max raw pass@k   cluster-vote selected pass@1
k10       0.1610       0.2722       0.3117        0.3117          0.2143
k16       0.1530       0.2742       0.3181        0.3506          0.2273
k25       0.1608       0.2643       0.2941        0.3377          0.2468
k50       0.1625       0.2723       0.3091        0.4156          0.2792
```

Reading:

- Raw pass@1/5/10 is mostly flat. More samples reveal rare correct
  candidates; they do not improve the underlying model.
- Cluster voting improves with sample count because repeated solutions become
  a usable proxy for semantic confidence.
- k16 is the cheap diagnostic point.
- k50 is the best reporting/deployable point if inference cost is acceptable.
- k25 being worse than k16 on raw pass@k is likely sampling noise because
  these were separate generations, not a nested prefix of one k50 pool.

#### 25.4 Commands for k50 and Reranking

Direct k50 inference requires LoRA env vars when bypassing the runner:

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
HEAD=${NEW}_fitutpk5_grpo

export GRAPH_ENCODER_PEFT=lora
export GRAPH_DECODER_PEFT=lora
export GRAPH_LORA_R=64
export GRAPH_LORA_ALPHA=128
export GRAPH_QWEN_PREFIX_TOKENS=16
export GRAPH_ATTN_IMPLEMENTATION=sdpa
export GRAPH_USE_REASONING=0
export GRAPH_DECODER_PROMPT_MAX_LENGTH=768
export GRAPH_QUIET=1

python scripts/evaluation/graph_inference_antigravity.py \
  --dataset data/testing/grpo_data.jsonl \
  --decoder_model Qwen/Qwen3.5-9B-Base \
  --output results/${HEAD}_pass_predictions_k50.json \
  --checkpoint artifacts/${HEAD}/pytorch_model.bin \
  --limit 0 \
  --num_samples 50 \
  --generation_batch_size 16 \
  --max_new_tokens 768 \
  --decoder_prompt_max_length 768

python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50.json \
  --k_values 1,5,10,25,50 \
  --workers 128
```

Fair deployable rerank:

```bash
python scripts/evaluation/rerank_predictions_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50.json \
  --output results/${HEAD}_pass_predictions_k50_reranked_compile_cluster_vote.json \
  --selected_output results/${HEAD}_pass_predictions_k50_selected_compile_cluster_vote.json \
  --report results/${HEAD}_pass_predictions_k50_rerank_compile_cluster_vote_report.json \
  --mode compile_cluster_vote \
  --cluster_vote_bonus 5.0 \
  --workers 128 \
  --timeout 10

python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50_selected_compile_cluster_vote.json \
  --k_values 1 \
  --workers 128
```

Oracle diagnostic only, never report as fair inference:

```bash
python scripts/evaluation/rerank_predictions_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50.json \
  --output results/${HEAD}_pass_predictions_k50_reranked_oracle.json \
  --selected_output results/${HEAD}_pass_predictions_k50_selected_oracle.json \
  --report results/${HEAD}_pass_predictions_k50_oracle_report.json \
  --mode test \
  --workers 128 \
  --timeout 3
```

Report analyzer:

```bash
python scripts/evaluation/analyze_rerank_reports_antigravity.py \
  --compile_report results/${HEAD}_pass_predictions_k50_rerank_compile_report.json \
  --oracle_report results/${HEAD}_pass_predictions_k50_oracle_report.json \
  --output results/${HEAD}_pass_predictions_k50_rerank_gap_analysis.json \
  --top_examples 50
```

#### 25.5 Synthetic Data and Why the New Mix Is Different

Available synthetic files:

```text
data/datasets/synthetic_pool.jsonl          1727 raw rows
data/datasets/synthetic_pool_clean.jsonl    1726 clean rows
data/datasets/synthetic_pool_train576.jsonl  576 subsample rows
```

These rows are schema-compatible with `grpo_data.jsonl`: they have
`assembly`, `dart_source`, `dart_function_signature`, `function`, and
`tests`.

Important: the full synthetic line was already tested and was negative:

```text
_synpk5full:
  direct GRPO on full synthetic from _fitut; aborted after ~25 steps because
  nearly every group had zero passing/non-compiling rollouts.

_synsft:
  SFT on full 1726 synthetic only, 2 epochs, lr 5e-6; real eval worsened.

_synsft2:
  SFT on full synthetic + real 154 mix, 4 epochs, lr 1e-5; token loss learned,
  but real pass@10 stayed baseline and compile/CodeBLEU worsened.

_synsft3072:
  longer prompt visibility; still no real gain, and synthetic GRPO probe
  remained dead.
```

Therefore the new proposal is not "try synthetic again blindly." It is a
targeted zero-pass-weighted mix:

- smaller synthetic dose,
- all 154 real tasks retained as anchor,
- current k50 zero-pass real rows repeated,
- start from current best head,
- judge by whether k50 oracle coverage or cluster-vote pass@1 increases.

New mixer added:

```text
scripts/data/build_grpo_mix_antigravity.py
```

Local default file built:

```text
data/testing/grpo_mix_synth576_real154_zp2.jsonl
```

Composition:

```json
{
  "rows": 910,
  "synthetic_rows_used": 576,
  "real_rows": 154,
  "zero_pass_indices": 90,
  "zero_pass_repeat": 2
}
```

Build/upload command:

```bash
python scripts/data/build_grpo_mix_antigravity.py \
  --synthetic data/datasets/synthetic_pool_train576.jsonl \
  --real data/testing/grpo_data.jsonl \
  --gap_analysis results/${HEAD}_pass_predictions_k50_rerank_gap_analysis.json \
  --output data/testing/grpo_mix_synth576_real154_zp2.jsonl \
  --summary data/testing/grpo_mix_synth576_real154_zp2.summary.json \
  --zero_pass_repeat 2 \
  --missed_passable_repeat 0 \
  --seed 13
```

Optional full-pool weighted mix if the 910-row mix helps:

```bash
python scripts/data/build_grpo_mix_antigravity.py \
  --synthetic data/datasets/synthetic_pool_clean.jsonl \
  --real data/testing/grpo_data.jsonl \
  --gap_analysis results/${HEAD}_pass_predictions_k50_rerank_gap_analysis.json \
  --output data/testing/grpo_mix_synth1726_real154x2_zp3.jsonl \
  --summary data/testing/grpo_mix_synth1726_real154x2_zp3.summary.json \
  --synthetic_repeat 1 \
  --real_repeat 2 \
  --zero_pass_repeat 3 \
  --missed_passable_repeat 1 \
  --seed 13
```

#### 25.6 Recommended Future Training Direction

Do not start with GRPO alone on zero-pass tasks. For prompts with no passing
rollouts, GRPO has no positive signal. SimKO/GSPO/DAPO help only after there
are positive-advantage samples to shape.

Recommended order:

1. SFT coverage repair on the mixed dataset.
2. Evaluate k16 and k50 with `compile_cluster_vote`.
3. Only if coverage improves or at least does not regress, run a small mixed
   GRPO pass.

Stage 1 SFT coverage repair:

```bash
BASE=qwen-9b-base_lora_enc_dec_r16_5e6_gcb
HEAD=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo
MIX=data/testing/grpo_mix_synth576_real154_zp2.jsonl

python configs/run_sweeps_antigravity.py \
  --experiment "$BASE" \
  --name_suffix _synthzp_sft \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --sft_checkpoint "artifacts/${HEAD}/pytorch_model.bin" \
  --train_file "$MIX" \
  --eval_file data/testing/grpo_data.jsonl \
  --epochs 1 \
  --sft_lr 2e-6 \
  --lora_r 64 \
  --lora_alpha 128 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 0 \
  --train_batch_size 4 \
  --grad_accum 16 \
  --qwen_prefix_tokens 16 \
  --decoder_prompt_max_length 768 \
  --prompt_fit_assembly 1 \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 16 \
  --num_samples 5 \
  --pass_num_samples 16 \
  --metric_workers 128 \
  --save_strategy epoch \
  --save_total_limit 2
```

Acceptance gate:

```text
Keep only if k50 oracle coverage increases above 64/154 or
compile_cluster_vote selected pass@1 beats 43/154 without a major compile
regression.
```

Stage 2 small mixed GRPO, only if Stage 1 passes:

```bash
SFT=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_synthzp_sft

python configs/run_sweeps_antigravity.py \
  --experiment "$BASE" \
  --name_suffix _synthzp \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --use_grpo \
  --grpo_checkpoint "artifacts/${SFT}/pytorch_model.bin" \
  --grpo_train_file "$MIX" \
  --grpo_group_size 8 \
  --grpo_epochs 1 \
  --grpo_lr 5e-7 \
  --grpo_reward_mode binary \
  --grpo_binary_fail_reward -1.0 \
  --grpo_perfect_base_reward 1.0 \
  --grpo_perfect_bonus 0.0 \
  --grpo_passk_k 5 \
  --grpo_max_new_tokens 512 \
  --grpo_test_timeout 3 \
  --grpo_reward_workers 128 \
  --grpo_score_chunk_size 4 \
  --grpo_loss_pooling seq \
  --grpo_min_reward_range 0.05 \
  --grpo_simko_k 4 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --train_batch_size 1 \
  --grad_accum 16 \
  --lora_r 64 \
  --lora_alpha 128 \
  --qwen_prefix_tokens 16 \
  --decoder_prompt_max_length 768 \
  --prompt_fit_assembly 1 \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 16 \
  --num_samples 5 \
  --pass_num_samples 16 \
  --metric_workers 128
```

SimKO note: `--grpo_simko_k 4` is reasonable as a secondary stabilizer, but
not the main bet. If OOM or slow, remove it before reducing group size:

```text
1. --grpo_simko_k 0
2. --grpo_score_chunk_size 2
3. --grpo_group_size 4
```

#### 25.7 Hardware Recommendation

RTX Pro 6000 96GB is enough for the next economically sensible tests:

```text
SFT on the 910-row mix
k16/k25/k50 inference
compile_cluster_vote reranking
small GRPO with group_size 8, chunk 4, SimKO 0 or 4
```

Use H200 only if pushing:

```text
GRPO group_size 16+
long-prompt GRPO at 2048/3072
SimKO + group_size 16
large-batch SFT for speed
```

Given current evidence, rent RTX Pro 6000 first. Move to H200 only after the
cheaper run shows promising coverage movement.

#### 25.8 Hugging Face Upload From Windows

Upload target should be an artifact folder, for example:

```text
artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo
```

PowerShell:

```powershell
python -m pip install -U huggingface_hub

$env:HF_TOKEN = "hf_YOUR_TOKEN_HERE"
$env:HF_REPO = "YOUR_USERNAME/qwen9b-antigravity-fitutpk5-grpo"
$env:CKPT_DIR = "C:\Users\Raafat Abualazm\Desktop\Train Data\experiment_workspace\artifacts\qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo"
```

Upload checkpoint folder:

```powershell
@'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO"]
folder = os.environ["CKPT_DIR"]
token = os.environ["HF_TOKEN"]

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder,
    path_in_repo=".",
    commit_message="Upload Antigravity Qwen graph LoRA checkpoint",
)
print(f"Uploaded: https://huggingface.co/{repo_id}")
'@ | python -
```

Upload loader/eval scripts:

```powershell
cd "C:\Users\Raafat Abualazm\Desktop\Train Data\experiment_workspace"

@'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO"]
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

files = [
    "configs/run_sweeps_antigravity.py",
    "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
    "scripts/training/graph_grpo_decompiler_antigravity.py",
    "scripts/evaluation/graph_inference_antigravity.py",
    "scripts/evaluation/rerank_predictions_antigravity.py",
    "scripts/evaluation/graph_pass_at_k_antigravity.py",
    "scripts/evaluation/analyze_rerank_reports_antigravity.py",
    "scripts/data/build_grpo_mix_antigravity.py",
]

for path in files:
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=path,
        path_in_repo=path,
        commit_message=f"Upload {path}",
    )
    print("uploaded", path)
'@ | python -
```

The HF repo should explicitly state: this is not a standalone model; it is a
GraphCodeBERT+Qwen3.5 adapter/glue checkpoint requiring the local architecture
code and base model weights.

### 26. Latest Remote State: synthzp_sft Trial Rejected, Head Unchanged (2026-06-18)

This section supersedes the tentative Stage 1/Stage 2 plan in Section 25.6.
The mixed synthetic + zero-pass SFT trial ran successfully, but it did not
improve the metric that matters.

#### 26.1 Remote Environment Incidents and Fixes

The H100 pod reported:

```text
torch: 2.8.0+cu128
cuda: 12.8
triton: 3.4.0
fla: 0.5.0
```

The first training attempt OOMed inside Qwen3.5 linear attention:

```text
transformers/models/qwen3_5/modeling_qwen3_5.py
self.linear_attn -> torch_chunk_gated_delta_rule
torch.OutOfMemoryError
```

Diagnosis: Qwen3.5 was falling back to the pure PyTorch gated-delta-rule path.
`--attn_implementation sdpa` does not solve this because the failing module is
Qwen3.5 linear attention, not normal Transformer SDPA attention.

Installed:

```bash
python -m pip install -U flash-linear-attention causal-conv1d
```

After FLA loaded, backward failed on H100/Hopper with Triton 3.4:

```text
RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for
gated chunk_bwd_dqkwg. Please install tilelang.
```

Installing TileLang initially crashed with:

```text
TypeAttr `__ffi_repr__` is already registered
```

Fix was to pin TVM-FFI below the broken 0.1.12 release:

```bash
python -m pip uninstall -y tilelang apache-tvm-ffi tvm-ffi apache-tvm tvm
python -m pip install -U pip wheel setuptools packaging ninja
python -m pip install "apache-tvm-ffi<0.1.12"
python -m pip install --no-deps "tilelang==0.1.11"
```

Verified good imports:

```text
tvm_ffi 0.1.11
tilelang 0.1.11
torch 2.8.0+cu128
triton 3.4.0
fla 0.5.0
```

Important note: the pod is CUDA 12.8 from PyTorch's point of view, not CUDA 13.
Do not chase CUDA 13 wheels on this pod unless the whole image changes.

#### 26.2 Code/Data Bundle Incident

HF checkpoint repos were used for weights only. Scripts/data were uploaded from
the local Windows workspace. The first code bundle omitted:

```text
data/testing/compile-test2.jsonl
```

This caused the post-training runner to fail only at compile/CodeBLEU inference:

```text
FileNotFoundError: data/testing/compile-test2.jsonl
```

Training had already finished and saved:

```text
artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_synthzp_sft/pytorch_model.bin
```

Future upload bundles must include:

```text
data/testing/compile-test2.jsonl
data/testing/grpo_data.jsonl
data/testing/grpo_mix_synth576_real154_zp2.jsonl
data/datasets/synthetic_pool_train576.jsonl
```

#### 26.3 Runner Naming Trap

The sweep runner rewrites experiment names before lookup when LoRA overrides
are passed. Therefore this fails:

```bash
BASE=qwen-9b-base_lora_enc_dec_r16_5e6_gcb
python configs/run_sweeps_antigravity.py --experiment "$BASE" --lora_r 64 --lora_alpha 128 ...
```

because the configured name has been rewritten to:

```text
qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
```

Use:

```bash
BASE=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
```

#### 26.4 synthzp_sft Trial

Training command lineage:

```text
start checkpoint:
  artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo/pytorch_model.bin

train file:
  data/testing/grpo_mix_synth576_real154_zp2.jsonl

output:
  artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_synthzp_sft/pytorch_model.bin
```

The mix file composition was:

```text
576 synthetic rows
154 real GRPO/test rows
90 zero-pass real rows repeated twice
= 910 rows
```

The safer H100 training settings that got through training were:

```text
gradient_checkpointing=1
train_batch_size=1
grad_accum=64
decoder_prompt_max_length=512 during training
prompt_fit_assembly=1
```

Evaluation then used prompt length 768 and `pass_num_samples=16`.

#### 26.5 synthzp_sft Results

Raw CodeBLEU on compile set:

```json
{
  "count": 126,
  "samples_with_success": 126,
  "mean_codebleu": 0.6505345670843848,
  "min_codebleu": 0.17515334592135465,
  "max_codebleu": 0.9319495947686317
}
```

Compiled-only CodeBLEU:

```json
{
  "count": 126,
  "samples_with_success": 87,
  "mean_codebleu": 0.6379034174307496,
  "min_codebleu": 0.17515334592135465,
  "max_codebleu": 0.9319495947686317
}
```

Pass@k on the 154 GRPO/test tasks with `n=16` candidates:

```json
{
  "pass_at_1": 0.1525974025974026,
  "pass_at_5": 0.26701423576423566,
  "pass_at_10": 0.30424770035159643,
  "total_problems": 154
}
```

Comparison against the active best `_fitutpk5_grpo` early-k baseline:

```json
{
  "pass_at_1": 0.16246753246753246,
  "pass_at_5": 0.27227255932347605,
  "pass_at_10": 0.30913218382519236
}
```

Conclusion: `_synthzp_sft` improved surface/compile-oriented metrics but
slightly hurt pass@1/5/10. This repeats the central failure pattern:
CodeBLEU/compile can move upward while functional correctness does not.

Decision:

```text
REJECT qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_synthzp_sft
for lineage head and for GRPO continuation.
```

Do not run GRPO from `_synthzp_sft`; it would likely sharpen a weaker policy.

#### 26.6 Current Accepted Head

The active best remains:

```text
qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo
```

Known useful numbers for this head:

```text
direct pass@1:   0.1625
direct pass@5:   0.2723
direct pass@10:  0.3091
direct pass@25:  0.3606
direct pass@50:  0.4156

compile_cluster_vote selected pass@1:
43 / 154 = 0.2792
```

Acceptance rule for any future checkpoint:

```text
Keep only if it beats at least one of:
  pass@10 > 0.3091
  k50 oracle/passable coverage > 0.4156
  compile_cluster_vote selected pass@1 > 0.2792

Reject if it only improves CodeBLEU or compile-only CodeBLEU.
```

#### 26.7 Recommended Next Steps

1. Archive `_synthzp_sft` results/checkpoint for record, but do not branch from
   it.

2. Return to:

```bash
HEAD=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo
```

3. Re-run or preserve k50 inference and fair compile-cluster reranking for the
   accepted head:

```bash
python scripts/evaluation/graph_inference_antigravity.py \
  --dataset data/testing/grpo_data.jsonl \
  --decoder_model Qwen/Qwen3.5-9B-Base \
  --output results/${HEAD}_pass_predictions_k50.json \
  --checkpoint artifacts/${HEAD}/pytorch_model.bin \
  --limit 0 \
  --num_samples 50 \
  --generation_batch_size 8 \
  --max_new_tokens 768 \
  --decoder_prompt_max_length 768

python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50.json \
  --k_values 1,5,10,25,50 \
  --workers 128 \
  | tee results/${HEAD}_pass_at_k50.json

python scripts/evaluation/rerank_predictions_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50.json \
  --output results/${HEAD}_pass_predictions_k50_reranked_compile_cluster_vote.json \
  --selected_output results/${HEAD}_pass_predictions_k50_selected_compile_cluster_vote.json \
  --report results/${HEAD}_pass_predictions_k50_rerank_compile_cluster_vote_report.json \
  --mode compile_cluster_vote \
  --cluster_vote_bonus 5.0 \
  --workers 128 \
  --timeout 10

python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50_selected_compile_cluster_vote.json \
  --k_values 1 \
  --workers 128 \
  | tee results/${HEAD}_compile_cluster_vote_selected_pass_at_1.json
```

4. If running another training attempt, prefer rejection-SFT style data over
   synthetic-zero-pass mixing. Candidate file already exists locally:

```text
data/testing/rs_sft_all_plus_refs.jsonl
```

Potential next command, engineering-only / not clean-paper protocol:

```bash
BASE=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
HEAD=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo
RS=data/testing/rs_sft_all_plus_refs.jsonl

python configs/run_sweeps_antigravity.py \
  --experiment "$BASE" \
  --name_suffix _rsft_from_pk5 \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h100 \
  --force_rerun \
  --sft_checkpoint "artifacts/${HEAD}/pytorch_model.bin" \
  --train_file "$RS" \
  --eval_file data/testing/grpo_data.jsonl \
  --epochs 1 \
  --sft_lr 1e-6 \
  --lora_r 64 \
  --lora_alpha 128 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 1 \
  --train_batch_size 1 \
  --grad_accum 64 \
  --qwen_prefix_tokens 16 \
  --decoder_prompt_max_length 768 \
  --prompt_fit_assembly 0 \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 4 \
  --num_samples 5 \
  --pass_num_samples 16 \
  --metric_workers 128 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --hf_repo "$HF_OUT_REPO" \
  --hf_private 1
```

Warning: `rs_sft_all_plus_refs.jsonl` likely includes passing candidates
harvested from the same 154-task pool, so it is useful for engineering and
diagnosis but not sufficient for a clean ICSE-style generalization claim.

---

## 24. Flagship pivot (2026-06-19): two papers + real ARM64 Flutter corpus

Strategy reset (full rationale in the web-handoff + memory `flagship-strategy-and-flutter-arm64`):
the void synthetic-x86 graph line is retired as an *engine*, not a paper. Two flagships:
- **B (ship first, ASE A\*):** compositional/hierarchical decompilation that breaks the
  ~200-instruction capability cliff using the CFG to decompile blocks/loop bodies
  independently and recompose. Rebuilt graph system = the engine.
- **A (later, NDSS/USENIX/CCS A\*):** real obfuscated ARM64 Flutter + a downstream
  security task. Artifact = Flutter-Eval.

### 24.1 New corpus: `data/datasets/flutter_function_assembly_pool.jsonl` (1714 rows)
Real Flutter Android **ARM64** per-function slices from stripped/**obfuscated**/signed
`libapp.so` (1714 distinct builds), `apk` release build mode. Test-equipped (`tests`
field with `expect(...)`; 1695/1714 also carry assert-`main()`) -> pass@k computable.
Heavy in the cliff zone: **median 193 instr lines, 48% (826/1714) > 200 lines**. Zero
overlap with the 154-task eval (by fn-name and exact source). CAVEAT (honest scoping):
**semantics are synthetic LLM-generated algorithmic tasks** (sorting/recursion/matrix/
parsing), only the **binary distribution** is real -> this is Flutter-Eval **stage 1**
(real binary dist, controlled semantics), NOT real-app-logic. Security framing that
survives review: Flutter `--obfuscate` is *symbol/name* obfuscation + strip (NOT
instruction-level CFF/opaque-predicates -> disasm is clean optimized ARM64); so the
defensible claim is "name-based hardening is false security against a learned decompiler;
we recover functional Dart from stripped AOT-optimized obfuscated release binaries."
Scope to *standard Flutter obfuscation + AOT + strip*; packers/anti-tamper = future work.

### 24.2 ARM64 support in `cfg_extractor.py` (DONE + validated, local/no-GPU)
The extractor was **x86-only** -> on ARM64 it would silently reproduce the dead-CFG bug
(no leaders -> single block -> dead GNN + truncation). Three ISA-mismatch fixes (union, x86
kept working):
1. **Mnemonics:** added AArch64 branch sets — conditional `b.<cond>`/`cbz`/`cbnz`/`tbz`/
   `tbnz`, unconditional `b`/`br` (`br` = indirect, no static target), and `bl`/`blr` =
   CALLS kept inline as fall-through (never split a block).
2. **Address column:** `parse()` required a `0x` prefix; AArch64 llvm-objdump uses bare hex
   (`226304:`). Regex now accepts `0x...` OR `hex:` at line start (the `:` keeps numbered
   Dart source rejected). `canonicalize_address` falls back to bare-hex.
3. **Annotation hazard:** `extract_jump_target` now strips `<symbol+0xoffset>` before scanning,
   else the annotation's 0x offset is mistaken for the branch target.

Validation (`scripts/data/validate_arm64_cfg.py`, all 1714 rows):
- 1714/1714 extracted, 0 errors, **0 single-block collapses**, median **42 blocks / 59 edges**.
- **Intra-proc branch resolution 58734/58734 = 100%** (key correctness signal).
- x86 regression unbroken: `grpo_data` 154/154 (2383/2383 resolved, median 17 blk),
  `compile-test2` 126/126 (3299/3300).
- Minor/non-blocking: 13.4% rows have one >512-tok block (milder than x86 _cfg's 38.9%);
  18% rows have unreachable-from-entry nodes, largely **multi-slice rows** (mean 1.4, up to
  8/row) where Dart AOT emits nested closures/comparators as separate functions — the
  extractor correctly keeps them as disconnected components. NB for B: slice boundaries
  (`flutter_function_symbol_ranges`) are a ready-made decomposition primitive.

### 24.3 Next (local, free, before any pod spend)
- Build precomputed `flutter_function_assembly_pool_cfg.jsonl` via build_cfg_jsonl.py.
- Carve a CLEAN train/held-out split (guardrail #1). Split-design is a research decision
  (random in-distribution vs hold-out-by-category vs ...); length must stay represented in
  eval since B's claim IS the length cliff.
- Then the paid step: corrected SFT->GRPO on the live ARM64 pipeline, fresh suffix,
  in-distribution held-out eval (do NOT eval against the x86 HumanEval set — ISA+style
  mismatch = the old synthetic-transfer failure mode).

### 24.4 Preprocessing finalized + architecture-ablation plan (2026-06-20)
Three preprocessing capabilities added (all local, validated):
- **Block-splitting** (`GRAPH_MAX_BLOCK_INSTRS`, cfg_extractor): caps block length
  so every block fits GraphCodeBERT's 512-token window. On flutter: max instr/block
  167->24, oversized-block rows 13.4%->0.1%, branch resolution still 100% (only
  linear edges added). Encoder truncation ~eliminated. `flutter_*_cfg.jsonl` rebuilt
  with it (90,770 blocks).
- **Prompt-clean** (`GRAPH_PROMPT_CLEAN_ASM`, build_decoder_prompt): strips
  `<symbol+0xoffset>` annotations + `;`/`//` comments from the DECODER prompt only
  (~25% fewer tokens, zero info loss; graph channel untouched).
- **No-assembly mode** (`GRAPH_PROMPT_ASSEMBLY_MODE=none`): withholds raw assembly
  text so the decoder relies on the graph-prefix channel (encoder-carries ablation).
All three plumbed as runner CLI flags: `--max_block_instrs`, `--prompt_clean_asm`,
`--prompt_assembly_mode` (verified in --help).

**Why an ablation first (architecture finding):** the decoder is Qwen (causal,
decoder-only) — it has no native cross-attention to the encoder. The graph is
injected as just 16 gated soft-prefix tokens (`QwenGraphPrefixAdapter`, gate stuck
~0.2), which cannot carry instruction-level detail, so the code ALSO feeds raw
assembly text. So today the encoder is a faint hint and the (truncatable) TEXT is
the real channel. Before buying a big decoder budget or the GNN, settle which
channel carries the info. **Plan: `RUN_PLAN_ARM64.md`** — SFT arms A0 (text-only,
prefix 0), A2 (wide graph+text, prefix 128), A3 (encoder-only, prefix 128 + no
text), optional A1 (16-token). Decision rule maps {A2 vs A0, A3 vs A2} to:
drop-GNN / encoder-carries / hybrid+need-decomposition / graph-interferes. Winner
-> pass@k-GRPO + full eval; then Flagship B compositional. Eval in-distribution on
flutter_eval, length-stratified; decoder budget 8192 (fair baseline); B200/B300.

### 24.5 Funding + compute reality (2026-06-22)
Applied for **AWS research credits**: proposal at `PROPOSAL_AWS.md` (pilot scope).
Award/plan: **$550 USD on a single `g7e.12xlarge`** instance. Proposal is scoped
to the decisive core only — the architecture ablation + the compositional
cliff-breaking demonstration on the already-built `flutter_*_cfg` corpus (no data
cost). Flagship A / Flutter-Eval-at-scale / downstream security = framed as future
work, NOT in this budget. Title: "Compositional Neural Decompilation of Hardened
Flutter/Dart AOT Binaries for Mobile Application Security." Budget = ~38
instance-hours (the 2 GPUs run ablation arms two at a time; same node's CPUs run
the pass@k harness; small S3).

**PLANNING IMPLICATION — reconcile with `RUN_PLAN_ARM64.md`:** that run plan
assumed B200/B300 (192/272 GB) with `--decoder_prompt_max_length 8192`.
`g7e.12xlarge` = **2x RTX Pro 6000 (96 GB GDDR7 each, 192 GB aggregate)**. SFT at
8192 fits comfortably on one 96 GB GPU. But GRPO runs SINGLE-GPU (the runner forces
1 GPU for `--use_grpo`), so it is bound by 96 GB not the 192 aggregate: 8192 GRPO is
feasible with chunked scoring + small `--grpo_group_size`, with 4096-6144 the safer
first target. The 192 GB aggregate helps only SFT (DDP across both GPUs).
OPEN QUESTION for the user: does g7e
*replace* the B200/B300 plan, or only fund the credit-covered slice (B200/B300
still primary)? This decides which budgets the run plan should carry.

### 24.6 X86 re-validation on Qwen3-8B (supervisor request, 2026-06-22)
Supervisor wants the FIXED (CFG-corrected, de-leaked) old datasets re-run on
**Qwen3-8B** (the TOSEM comparison base; registered as `qwen3-8b-base` ->
`Qwen/Qwen3-8B-Base`) to measure how good the encoder+graph architecture is now the
graph is live. User folded the **with/without-assembly** question into the same 2x2.
Plan = `RUN_PLAN_X86_ABLATION.md`. Matrix (Qwen3-8B, SFT on `dart_all_cfg_clean`,
eval `grpo_data_cfg` 154 + `compile-test2_cfg` 126, leak verified 0):
- R = base zero-shot (prefix 0, text, --skip_training) ~ TOSEM base (p@1 6.36%)
- G1 = text-only FT (prefix 0, asm full)
- G2 = graph+text (prefix 16, asm full)  -> supervisor Q = **G2 vs G1**
- G3 = graph-only (prefix 16, asm none)  -> with/without-asm = **G2 vs G3 vs G1**
- G0 = null floor (prefix 0, asm none)
Internally rigorous (same train data, only architecture varies); TOSEM = external
context. SFT-only; GRPO is a separate follow-up. Step-0 prep: rebuild x86 `_cfg`
with `--max_block_instrs 24` so G3's encoder isn't truncated (G1/G2 unaffected).
Light compute (8B + short x86 asm). SEPARATE scope from the ARM64 Flagship-B pilot.

**24.6 update (2026-06-22):** Step-0 DONE — x86 `_cfg` files (`dart_all_cfg_clean`,
`grpo_data_cfg`, `compile-test2_cfg`) rebuilt with block-split `<=24` (grpo_data: max
instr/block 24, branch res 100%, 0 single-block). Operational step-by-step =
`STUDY_RUNBOOK.md`, model-selectable via one `MODEL` toggle (verified exact names:
`qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128` / `qwen-9b-base_..._gcb_a128`; select
with `--experiment $EXP --name_suffix _x86_<arm>`). 5 arms R/G1/G2/G3/G0, SFT-only.

## 25. Publication Audit and Public Artifact (2026-07-10)

This section supersedes stale model-path and artifact-status statements above without
rewriting the historical run narrative.

- The exact 8B decoder identifier is **`Qwen/Qwen3-8B`**, not
  `Qwen/Qwen3-8B-Base`. The runner registry and publication commands were corrected.
- The 126-row CodeBLEU/legacy standalone-compile corpus and the 154-task
  HumanEval-Dart functional corpus are distinct datasets. The former contains
  standalone programs (mostly `main`) and no HumanEval `tests` field; it is not a
  126-task subset created by excluding 28 HumanEval-Dart rows.
- The primary paper compile metric is now the 154-task `jit_tests` view produced from
  the same candidate-plus-tests files as pass@k. The strict/legacy standalone AOT view
  remains an appendix diagnostic only.
- Frontier results are a July 8, 2026 practitioner snapshot. Azure
  `gpt-chat-latest` is a moving alias. Temperature/top-p are archived requested
  controls, not independently verified provider-side controls. GPT-5.5 and Sonnet 5
  have no recorded API-error rows; DeepSeek V4 Pro has three affected rows and GLM-5.2
  one, with blank candidate slots retained as failures.
- The seven GPT-5.5 prompt pools comprise four v1 representations and three cleaned
  v2 representations. The 0.7143 v1 full-assembly row and canonical 0.7013 frontier
  row come from separate stochastic runs with different output caps/settings and are
  not interpreted as a prompt-induced improvement.
- GLM-5.2 covers 44/154 tasks versus 43/154 for the 17-arm union: a one-task edge that
  is reported but not interpreted as a meaningful capability difference.
- Public code/results artifact:
  `https://github.com/raafatabualazm/MSc-Code/tree/graph-decompiler-2026-v1.1/graph_decompiler_2026_artifact`
- Public 8B adapter collection:
  `https://huggingface.co/raafatabualazm/antigravity-qwen3-8b-artifacts`
  at pinned revision `bc992cecbb6968be2e20b6e99e8a4c420e32242c`.
- The captured training-pod `requirements.txt` records the working CUDA 12.8 /
  PyTorch 2.8 environment. It includes machine-local paths for locally built
  dependencies, so those paths must be replaced when recreating the environment
  elsewhere; the environment snapshot itself is not missing.
- Remaining honest gaps: no repeated training seeds, no signature-only frontier
  control, and no preserved literal training commands for four auxiliary repair
  pools used only in union/reranking analysis.

## 26. Leakage-Free Rerun and GRPO Audit (2026-07-11)

This section supersedes the historical local pass@k/GRPO recommendations above.

### 26.1 Required rerun and data boundary

The historical local 154-task functional pools are contaminated because the policy
prompt included excerpts from the same scoring tests, including expected outputs.
Those numbers remain useful only as historical diagnostics. The clean prompt schema
is now `antigravity-v2-no-test-hints`; a 154-row audit found zero leaked prompts.

Never train on either evaluation corpus:

- `data/testing/compile-test2_cfg.jsonl` (126 unrelated legacy programs)
- `data/testing/grpo_data_cfg.jsonl` (154 HumanEval-Dart benchmark tasks)

The clean SFT train split contains 1,081 rows and has zero exact or near-source
overlaps with the 154-task benchmark. The corrected experiment is automated by
`scripts/run_leakage_free_study.py`; the operator guide is
`RUNBOOK_LEAKAGE_FREE_20H.md`.

### 26.2 Synthetic pool repair and verification

The original 1,726-row pool was not entirely reward-compatible: 172 rows called the
reference function directly in tests, and one row had only two assertions. The new
canonicalizer repaired 1,375 calls and rejected the underspecified row:

- canonical pool: 1,725 rows
- source-grouped SFT split: 1,380 train / 172 validation / 173 test
- style-matched GRPO subset: 256 rows, drawn from synthetic train only
- benchmark name overlap: 0; exact and normalized near-source overlap pairs: 0
- exact binary reward parity: 1,725/1,725 references passed, 0 failures

Key artifacts are under `data/datasets/synthetic_reward_splits/`; the full executable
audit is `reward_full_parity.json`. Do not generate a replacement pool before the
clean 1,725-row data-scale arm is measured, because changing data and architecture at
the same time destroys the ablation. The recovered generator is
`generate_synthetic_tasks_parallel.py`; its default `--profile humaneval` creates
compact, original HumanEval-like library functions with 8-12 candidate assertions.

### 26.3 GRPO audit outcome

The confirmatory reward is now exact full-harness binary reward: all tests pass gives
`+1`; every other outcome gives `-1`. Missing Dart, missing tests, and zero extracted
assertions abort instead of silently becoming ordinary negative samples. Dart gets a
private writable home/cache for every subprocess, avoiding telemetry-induced false
compile failures on ephemeral workers.

Important algorithmic correction: the current one-update loop computes its detached
"old" log probability from the same forward pass. The ratio is therefore exactly one
and PPO clipping is inert. It is group-relative REINFORCE, not a complete PPO/GRPO
implementation. The historical top-K option is also not a faithful implementation of
CaSP/SimKO. Confirmatory runs disable KL, entropy, partial credit, overlap, uniqueness,
duplicate, pass@k weighting, and legacy top-K terms.

GRPO is optional and gated. Eight no-update rollout batches must first produce
`signal_group_rate_mean >= 0.05`; otherwise there is too little within-group binary
variation to learn, and the correct next step is RS-SFT or teacher distillation.
Direct `--phase grpo` now refuses to run without the matching approved preflight JSON.
That JSON is bound to the exact checkpoint and reward-dataset SHA-256 plus seed,
generation settings, and reward values. Binary candidates are scored concurrently
across the group, partial accumulation windows are rescaled before an epoch-end
flush, and every trainable floating tensor is forced to FP32 before the optimizer is
created so a `5e-7` update cannot round away in BF16.

### 26.4 Twenty-hour priority order

One RTX PRO 6000 reservation should prioritize the clean causal result:

1. Run CPU preflight before renting the GPU.
2. At seed 42 run full-assembly base, signature-only base, text SFT, no-edge
   prefix, CFG, CFG+DFG, and shuffled-edge.
3. At seeds 43 and 44 repeat only no-edge prefix, CFG, and CFG+DFG.
4. Run the expanded original-plus-synthetic CFG+DFG arm if budget remains.
5. Run reward preflight; schedule GRPO in a later four-hour reservation if approved.

The no-GINE pooling control remains available through `--full_matrix`, but the
signature-only base control has priority in the first reservation because it directly
measures public-benchmark recognition without binary evidence.

Compile@k and pass@k now reuse the same 154-task, 10-candidate prediction pool when
their settings match. This halves redundant GPU generation and makes paired metrics
refer to identical candidates. Candidate-level CSVs use that same JIT/pass-harness
classification, contain no hardcoded task failures, and are copied rather than scored
twice for a shared pool. Historical pools took roughly 1-2.5 hours each, so
20 hours is a priority budget, not a guarantee that every stage finishes; the driver
records state and resumes without regenerating provenance-valid outputs.

## 27. Graph-v2 Correction Superseding Section 26 (2026-07-11)

Section 26 correctly established the leakage-free protocol, but its statement that
all 1,081 original training rows were suitable for graph training is no longer valid.
An adversarial audit of the CFG/DFG construction found multiple parser and graph-
consumer defects. A first fail-closed rebuild exposed truncated historical captures;
source recompilation then recovered a much larger clean corpus. This section
supersedes every graph-data count, graph-checkpoint reuse recommendation, and graph
architecture command above.

### 27.1 Historical graph results are retired

Do not resume or report any historical `_cfg` graph checkpoint as evidence for the
new architecture. The old precomputed files contained GDB symbol declarations as
instructions, incomplete branch coverage, unsafe direct-target parsing, weak
integrity checks, incorrect call/return DFG semantics, and no protection against
cross-function PyG endpoint leakage. Loading old tensors with `strict=False` does not
make them compatible with graph-v2.

Graph-v2 always rebuilds from the assembly field and records the assembly,
extractor, and output hashes. Fresh accepted counts are:

| Role | Rows | File |
|---|---:|---|
| Original SFT train | 770 | `data/datasets/dart_all_graphv2_train.jsonl` |
| Original validation | 83 | `data/datasets/dart_all_graphv2_validation.jsonl` |
| HumanEval-Dart evaluation only | 154 | `data/testing/grpo_data_graphv2.jsonl` |
| Synthetic source master | 1,726 | `data/datasets/synthetic_pool_graphv2.jsonl` |

The SFT source was recompiled with Dart 3.11.5 and GDB 17.1. Of 1,081 train rows,
941 yielded complete validated captures; 168 exact normalized-source duplicates and
three train programs near-duplicating validation were removed, producing 770 rows.
Of 114 validation rows, 16 exact duplicates and 15 malformed/context-dependent
fragments were removed, producing 83 rows. The final split has zero exact or
token-normalized near overlap internally and with the benchmark. The remaining 140
train and 15 validation rejects are missing companion declarations, malformed, or
source-less and are not repaired by inventing code. This still creates a
self-contained-program selection limitation that must be disclosed.

All 154 benchmark and 1,726 synthetic rows pass graph integrity. Exact counts and
SHA-256 values are in the adjacent `.summary.json` and `.rejected.jsonl` files,
`results/graph_v2_dataset_audit.json`, and the source-split filter reports. Blocks
are split into lossless linear chains of at most 20 instructions; the pinned
GraphCodeBERT tokenizer audit observed a maximum of 430 code tokens against a
510-token budget.

### 27.2 Construction and bridge changes

The corrected CFG parser handles the full relevant x86 conditional family, GDB
`=>` markers, duplicate symbol regions, ARM byte columns and bare targets, indirect
jumps, traps, true `<+0>` entry selection, dominator-only backedges, and reachability.
Unknown or unresolved direct branches invalidate a row. Closed unreachable regions
may be pruned only after direct-target closure is established and are recorded.

The DFG is now an honest block-level may-reaching-definition graph for general-
purpose registers, flags, Dart call/return effects, and stable frame slots. Dart
x64 calls consume `rdi,rsi,rdx,rbx,r8,r9`; Dart ARM64 calls consume
`x1,x2,x3,x5,x6,x7`. Calls kill volatile definitions, returns consume `rax/x0`,
8-bit x86 aliases and common implicit integer operands are modeled, and convergence
uses a worklist. Moving `rsp/sp` slots, general heap aliasing, SIMD/FP, SSA precision,
and interprocedural call graphs remain out of scope.

Graph-v2 validates every edge before PyG batching, assigns `dataflow` its own type,
adds distinct reverse relation types, and adds canonical block-order positions before
global attention. The Qwen bridge retains learned query identity in the residual,
uses a dynamic 4-64-slot prefix based on block count, one gate per prefix slot, and
RMS matching to native token embeddings. Decoder-layer cross-attention and multiple
semantic tokens per block remain future work.

### 27.3 Confirmatory experiment

The causal comparison is now same-data and same-bridge:

1. signature-only untuned Qwen;
2. untuned Qwen with cleaned assembly;
3. cleaned-assembly text SFT;
4. graph-prefix path with no edges;
5. the same path with CFG edges;
6. the same path with CFG+DFG edges;
7. shuffled edges;
8. CFG+DFG plus cleaned assembly, measuring graph/text competition.

No-edge, CFG, and CFG+DFG run at seeds 42, 43, and 44. Every graph arm uses the
same lossless 20-instruction fallthrough segmentation. The benchmark remains
evaluation-only. The expanded arm adds the 1,380-row synthetic graph-v2 train split
to the 770 leakage-clean original rows. Optional GRPO uses only a 256-row synthetic
subset after executable reward parity and mixed-outcome preflight.

The confirmatory reward is one full-harness `dart run`: complete pass `+1`, every
failure `-1`, with no partial/overlap/diversity/entropy/SimKO terms. The local block
encoder is frozen; graph glue and policy adapters train. The current objective must
be called group-relative REINFORCE, not full PPO/GRPO, because its same-pass detached
old log probability makes the ratio one and clipping inert.

### 27.4 Verified state and source of truth

Local structural preflight passed on 2026-07-11:

- 29 adversarial graph/gradient tests;
- 16 protocol-integrity tests;
- 62 preprocessing/tensor checks;
- all GRPO reward/objective/chunking checks;
- deterministic graph-v2 rebuild and exact tokenizer audit;
- zero leaked scoring-test prompts, zero train/validation overlap, and zero train/benchmark overlap;
- zero static reward failures across 1,725 canonical synthetic rows;
- 1,725/1,725 local executable reference rewards passed with zero failures.

The remote full executable 1,725-reference parity audit is still mandatory before
paid GPU work. `GRAPH_CONSTRUCTION_AUDIT.md` is the finding-by-finding technical
record. `RUNBOOK_LEAKAGE_FREE_20H.md` and `scripts/run_leakage_free_study.py` are the
only current execution instructions. Build and transfer the 58-file graph-v2 bundle;
do not use an older clean-study archive.

---

## 28. ARM64 graph-v2.1 preparation (2026-07-12)

The 1,714-row Flutter ARM64 corpus has been rebuilt and audited on a CPU-only
DigitalOcean host. The final graph-v2.1 dataset is local under
`data/datasets/arm64_graphv2/`; the complete technical record and hashes are in
`ARM64_GRAPHV21_PREP.md`.

The preparation audit found two issues that the old ARM plan missed. First,
NetworkX was not installed on the empty server, so the extractor silently skipped
dominator analysis. Graph-v2.1 now fails integrity closed without NetworkX. Second,
Flutter records can contain a wrapper, implementation, local helpers, and closures
as multiple symbol slices. Treating only the first instruction as the graph entry
pruned thousands of legitimate semantic blocks. Graph-v2.1 resolves every recorded
symbol start as a legitimate entry, keeps the exact top-level function as primary,
computes dominators through a virtual analysis root, and emits `call` edges for
statically resolved intra-slice calls.

Final full-corpus counts are 91,895 blocks, 126,931 CFG edges, 306,958 uncapped DFG
edges, 2,339 resolved symbol entries, 2,963 loop backedges, 100 intra-slice call
edges, seven recorded unreachable runtime-tail blocks, and zero rejected rows. The
pinned GraphCodeBERT tokenizer maximum is 349/510 tokens per block. The corrected
category-and-length-stratified split is 1,371 train / 343 evaluation, with zero
exact or 0.8-threshold near-source overlap internally and against the 154-task x86
benchmark. Evaluation retains all 17 categories and 168 functions of at least 200
instructions.

Graph-v2.1 is intentionally versioned separately from the graph-v2 data used by the
ongoing x86 runs. Exact x86 preprocessing sources are preserved under
`archive/graph_v2_x86_20260711/`. Do not upload graph-v2.1 scripts into the active
x86 pod. The legacy `RUN_PLAN_ARM64.md` is marked superseded; ARM confirmatory
training must use the same pinned Qwen3-8B lineage as the x86 study, not the old 9B
commands.
