# Run Commands

Consolidated command log for the Antigravity decompiler experiments.

This is a deduplicated runbook assembled from `FULL_HANDOFF.md`,
`RUN_PLAN_X86_ABLATION.md`, `RUN_PLAN_ARM64.md`, `STUDY_RUNBOOK.md`, and the
recent paper/thesis work. It preserves the commands we actually used or settled
on. Historical or rejected experiments are labeled as such.

## 0. Basic Environment Checks

### GPU / CUDA / PyTorch check

```bash
python -c "import torch, platform; print('python:', platform.python_version()); print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('compute_capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

### Attention backends exposed by Transformers

```bash
python - <<'PY'
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
print(sorted(ALL_ATTENTION_FUNCTIONS.keys()))
PY
```

### Common Python packages

```bash
pip install peft transformers xformers tree-sitter
pip install torch-geometric
pip install tilelang
python -m pip install -U huggingface_hub
```

### Preload base models on a pod

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B-Base'); snapshot_download('microsoft/graphcodebert-base')"
```

### Compile sanity

```bash
python -m py_compile configs/run_sweeps_antigravity.py
```

## 1. Useful Runner Discovery

### List matching 9B configs

```bash
python configs/run_sweeps_antigravity.py --encoder gcb --max_risk high | grep qwen-9b-base
```

### List matching 8B configs

```bash
python configs/run_sweeps_antigravity.py \
  --encoder gcb \
  --max_risk high \
  --epochs 4 \
  --lora_r 64 \
  --lora_alpha 128 \
  | grep qwen3-8b-base | grep 5e6
```

### Runner help

```bash
python configs/run_sweeps_antigravity.py --help
```

## 2. Sync Commands

### Push patched scripts to an old remote by SCP

```bash
scp -P 64566 ./configs/run_sweeps_antigravity.py root@71.232.99.8:/workspace/configs/
scp -P 64566 ./scripts/training/graph_grpo_decompiler_antigravity.py root@71.232.99.8:/workspace/scripts/training/
scp -P 64566 ./scripts/evaluation/rerank_predictions_antigravity.py root@71.232.99.8:/workspace/scripts/evaluation/
```

### Push full code/data tree to a remote

```bash
scp -P <PORT> -r ./configs ./models ./scripts ./data root@<IP>:/workspace/
```

### Pull all remote results

```bash
scp -P 64566 -r root@71.232.99.8:/workspace/results ".\results-qwen-9b-latest-N"
```

### Pull a checkpoint

```bash
scp -P 64566 -r root@71.232.99.8:/workspace/artifacts/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo ".\artifacts\"
```

### Lightning SSH examples used

```bash
ssh s_01kvb8rpr3wsrywshwmdqbyfkz@ssh.lightning.ai
ssh s_01kwwydfy31vfqr0tt4sav99e3@ssh.lightning.ai
```

## 3. Data Preparation

### Rebuild x86 CFG files with split blocks

```bash
for f in data/datasets/dart_all_cfg_clean.jsonl data/testing/grpo_data_cfg.jsonl data/testing/compile-test2_cfg.jsonl; do
  tmp="${f}.split24.tmp"
  GRAPH_MAX_BLOCK_INSTRS=24 python scripts/data/build_cfg_jsonl.py --input "$f" --output "$tmp" --overwrite
  test -s "$tmp"
  mv "$tmp" "$f"
done

wc -l data/datasets/dart_all_cfg_clean.jsonl data/testing/grpo_data_cfg.jsonl data/testing/compile-test2_cfg.jsonl
```

### Alternate in-place rebuild from `STUDY_RUNBOOK.md`

```bash
for f in data/datasets/dart_all_cfg_clean.jsonl data/testing/grpo_data_cfg.jsonl data/testing/compile-test2_cfg.jsonl; do
  GRAPH_MAX_BLOCK_INSTRS=24 python scripts/data/build_cfg_jsonl.py --input "$f" --output "$f.tmp" --overwrite && mv -f "$f.tmp" "$f"
done
```

### Build CFG for original GRPO data

```bash
python scripts/data/build_cfg_jsonl.py \
  --input data/testing/grpo_data.jsonl \
  --output data/testing/grpo_data_cfg.jsonl \
  --overwrite
```

### Build CFG for compile set

```bash
python scripts/data/build_cfg_jsonl.py \
  --input data/testing/compile-test2.jsonl \
  --output data/testing/compile-test2_cfg.jsonl \
  --overwrite
```

### Style-match synthetic pool

```bash
python scripts/data/style_match_synthetic_pool.py \
  --synthetic data/datasets/synthetic_pool_clean_cfg.jsonl \
  --real data/testing/grpo_data_cfg.jsonl \
  --output data/datasets/synthetic_pool_stylematched800_cfg.jsonl \
  --summary data/datasets/synthetic_pool_stylematched800_cfg.summary.json \
  --target_rows 800 \
  --max_source_words 220 \
  --max_asm_lines 700 \
  --min_expects 3
```

### Build zero-pass weighted synthetic/real GRPO mix

```bash
python scripts/data/build_grpo_mix_antigravity.py \
  --synthetic data/datasets/synthetic_pool_train576_cfg.jsonl \
  --real data/testing/grpo_data_cfg.jsonl \
  --gap_analysis results/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo_pass_predictions_k50_rerank_gap_analysis.json \
  --output data/testing/grpo_mix_synth576_real154_zp2_cfg.jsonl \
  --summary data/testing/grpo_mix_synth576_real154_zp2_cfg.summary.json \
  --synthetic_limit 576 \
  --synthetic_repeat 1 \
  --real_repeat 1 \
  --zero_pass_repeat 2 \
  --missed_passable_repeat 1 \
  --shuffle 1 \
  --seed 13
```

### Build mix from style-matched synthetic pool

```bash
python scripts/data/build_grpo_mix_antigravity.py \
  --synthetic data/datasets/synthetic_pool_stylematched800_cfg.jsonl \
  --real data/testing/grpo_data_cfg.jsonl \
  --output data/testing/grpo_mix_style732_real154_cfg.jsonl \
  --summary data/testing/grpo_mix_style732_real154_cfg.summary.json \
  --synthetic_limit 732 \
  --synthetic_repeat 1 \
  --real_repeat 1 \
  --shuffle 1 \
  --seed 13
```

### Deterministic train/held-out split (feeds GRPO train-half + RS-SFT half-split runs)

Even row index -> RL train half, odd -> held out. Produces the
`grpo_data_rl_train_half_cfg.jsonl` file consumed by the SimKO and binary
pass@10 GRPO commands below (Section 7), and the split index file consumed by
the RS-SFT harvester's `--split half` mode.

```bash
python scripts/data/split_grpo_train_eval.py \
  --input data/testing/grpo_data_cfg.jsonl \
  --train_output data/testing/grpo_data_rl_train_half_cfg.jsonl \
  --split_output data/testing/grpo_split_halves.json
```

### Build RS-SFT data from prediction pools

```bash
python scripts/data/build_rs_sft_from_predictions_antigravity.py \
  --data data/testing/grpo_data_cfg.jsonl \
  --results_dir results-20260706 \
  --out_prefix data/testing/rs_sft_x86_8b_allarms \
  --include "*qwen3-8b*x86*pass_predictions.json" \
  --max_per_task 4 \
  --split half \
  --split_key task_id \
  --min_code_chars 20 \
  --report data/testing/rs_sft_x86_8b_allarms_report.json
```

### Paper statistics / CI / paired tests

```bash
python scripts/evaluation/paper_stats_antigravity.py \
  --results_dir results-20260707/results \
  --union_report data/testing/rs_sft_x86_8b_allarms_with_h100_report.json \
  --output_json results-20260707/results/sweeps_antigravity/x86_8b_paper_statistics.json \
  --output_csv results-20260707/results/sweeps_antigravity/x86_8b_paper_statistics.csv \
  --bootstrap_reps 10000 \
  --seed 13
```

## 4. 9B Qwen3.5 Lineage

### Base SFT on de-leaked `dart_all_cfg_clean`

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" --name_suffix _cfgbase \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --train_file data/datasets/dart_all_cfg_clean.jsonl \
  --eval_file data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset data/testing/grpo_data_cfg.jsonl \
  --prompt_fit_assembly 1 --auto_cfg 1 \
  --epochs 4 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 4 --grad_accum 16 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 --eval_max_new_tokens 768 \
  --generation_batch_size 4 --num_samples 5 --pass_num_samples 10
```

### Original unit-test SFT from base

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

### Corrected CFG unit-test SFT

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py \
  --experiment "$NEW" --name_suffix _cfg_ut \
  --encoder gcb --max_risk high --hardware_profile h200 --force_rerun \
  --sft_checkpoint "artifacts/$NEW/pytorch_model.bin" \
  --train_file data/testing/grpo_data_cfg.jsonl \
  --eval_file data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset data/testing/grpo_data_cfg.jsonl \
  --prompt_fit_assembly 1 --auto_cfg 1 \
  --epochs 4 --sft_lr 5e-6 --lora_r 64 --lora_alpha 128 \
  --load_4bit 0 --attn_implementation sdpa --gradient_checkpointing 0 \
  --train_batch_size 4 --grad_accum 16 \
  --qwen_prefix_tokens 16 --qwen_prefix_gate_init 0.2 \
  --decoder_prompt_max_length 768 --eval_max_new_tokens 768 \
  --generation_batch_size 4 --num_samples 5 --pass_num_samples 10
```

### Best historical gentle GRPO

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

### Rewardsoft GRPO, rejected

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
  --use_reasoning 0
```

### Diverse GRPO branch after grad-accum fix

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

### Binary-diverse GRPO trial

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
  --use_reasoning 0
```

### Fixed-GRPO smoke test

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

### Fixed-GRPO full trial

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

### Prompt-budget SFT `_utfit`

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

### GRPO on `_utfit`

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

### K50 inference and fair reranking for accepted 9B head

```bash
HEAD=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo

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

## 5. 9B Synthetic / RS-SFT Experiments

### SFT coverage repair on synthetic zero-pass mix, rejected

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

### Small mixed GRPO after synthetic SFT, only if SFT gate passes

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

### Engineering-only RS-SFT from accepted 9B head

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
  --metric_workers 128
```

## 6. 8B x86 Architecture Ablation

### Shared setup

```bash
python -m py_compile configs/run_sweeps_antigravity.py

NEW=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128
METRIC_WORKERS=${METRIC_WORKERS:-64}
GEN_BS=${GEN_BS:-4}

python configs/run_sweeps_antigravity.py \
  --encoder gcb \
  --max_risk high \
  --epochs 4 \
  --lora_r 64 \
  --lora_alpha 128 \
  | grep qwen3-8b-base | grep 5e6

COMMON=(
  --experiment "$NEW"
  --encoder gcb
  --max_risk high
  --hardware_profile auto
  --force_rerun
  --train_file data/datasets/dart_all_cfg_clean.jsonl
  --eval_file data/testing/grpo_data_cfg.jsonl
  --compile_dataset data/testing/compile-test2_cfg.jsonl
  --pass_dataset data/testing/grpo_data_cfg.jsonl
  --prompt_fit_assembly 1
  --auto_cfg 1
  --max_block_instrs 24
  --dfg_mode edges
  --position_scheme roberta
  --epochs 4
  --sft_lr 5e-6
  --lora_r 64
  --lora_alpha 128
  --load_4bit 0
  --attn_implementation sdpa
  --gradient_checkpointing 1
  --train_batch_size 2
  --grad_accum 16
  --decoder_prompt_max_length 2048
  --eval_max_new_tokens 768
  --generation_batch_size "$GEN_BS"
  --num_samples 5
  --pass_num_samples 10
  --metric_workers "$METRIC_WORKERS"
)
```

### R reference, no training

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --skip_training \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _x86_ref_base
```

### G1 text-only

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _x86_g1_textonly
```

### G2 graph plus text

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _x86_g2_graphtext
```

### G3 graph-only

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode none \
  --name_suffix _x86_g3_graphonly
```

### G0 null control

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode none \
  --name_suffix _x86_g0_null
```

### G2c CFG-only optional arm

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --dfg_mode off \
  --name_suffix _x86_g2c_cfgonly
```

### Wide-prefix p128 arm, REJECTED (diverges without RMS matching)

Do not rerun as-is. Eval loss rose every epoch after epoch 1 (0.4356 to 0.6831,
grad-norm spikes to 77 to 119) and pass@k collapsed to 0. Kept here only so the
"p128 without RMS" row in the paper table is traceable to a real command.

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode none \
  --name_suffix _x86_g3_p128
```

### Wide-prefix p128r arm (fix: RMS-match the prefix to the decoder embedding scale)

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode none \
  --qwen_prefix_rms_match 1 \
  --qwen_prefix_gate_init 0.05 \
  --save_strategy epoch \
  --save_total_limit 4 \
  --name_suffix _x86_g3_p128r
```

### Metrics-only recovery loop for x86 arms

```bash
for SUFFIX in _x86_ref_base _x86_g1_textonly _x86_g2_graphtext _x86_g3_graphonly _x86_g0_null _x86_g2c_cfgonly; do
  EXTRA=()
  case "$SUFFIX" in
    _x86_ref_base) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _x86_g1_textonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _x86_g2_graphtext) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _x86_g3_graphonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode none) ;;
    _x86_g0_null) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode none) ;;
    _x86_g2c_cfgonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1 --dfg_mode off) ;;
  esac
  [ -f "results/sweeps_antigravity/${NEW}${SUFFIX}.json" ] || [ -f "results/${NEW}${SUFFIX}_pass_predictions.json" ] || continue
  python configs/run_sweeps_antigravity.py "${COMMON[@]}" "${EXTRA[@]}" --name_suffix "$SUFFIX"
done
```

### Print x86 comparison table

```bash
python - <<'PY'
import json
from pathlib import Path

base = "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128"
suffixes = [
    "_x86_ref_base",
    "_x86_g1_textonly",
    "_x86_g2_graphtext",
    "_x86_g3_graphonly",
    "_x86_g0_null",
    "_x86_g2c_cfgonly",
]

print("arm,codebleu,codebleu_compiled,compile@1,compile@5,pass@1,pass@5,pass@10")
for suffix in suffixes:
    path = Path("results/sweeps_antigravity") / f"{base}{suffix}.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    print(",".join([
        suffix.lstrip("_"),
        str(data["codebleu"]["mean_codebleu"]),
        str(data["codebleu_compiled_only"]["mean_codebleu"]),
        str(data["compile_at_k"].get("compile_at_1")),
        str(data["compile_at_k"].get("compile_at_5")),
        str(data["pass_at_k"].get("pass_at_1")),
        str(data["pass_at_k"].get("pass_at_5")),
        str(data["pass_at_k"].get("pass_at_10")),
    ]))
PY
```

## 7. 8B x86 SFT / GRPO Follow-ups

### Style-matched SFT continuation on B200/H200

```bash
python configs/run_sweeps_antigravity.py \
  --experiment qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128 \
  --name_suffix _x86_g3_style1036_sft \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --sft_checkpoint artifacts/qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_graphonly/pytorch_model.bin \
  --train_file data/testing/grpo_mix_style732_real154_cfg.jsonl \
  --eval_file data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset data/testing/grpo_data_cfg.jsonl \
  --epochs 1 \
  --sft_lr 1e-6 \
  --lora_r 64 \
  --lora_alpha 128 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 1 \
  --train_batch_size 4 \
  --grad_accum 16 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --qwen_prefix_rms_match 1 \
  --decoder_prompt_max_length 2048 \
  --prompt_fit_assembly 1 \
  --prompt_assembly_mode none \
  --auto_cfg 0 \
  --max_block_instrs 24 \
  --dfg_mode edges \
  --position_scheme roberta \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 32 \
  --num_samples 5 \
  --pass_num_samples 10 \
  --metric_workers 128 \
  --save_strategy epoch \
  --save_total_limit 2
```

### SimKO-style GRPO on G3 graph-only

```bash
NEW=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --use_grpo \
  --grpo_checkpoint "artifacts/${NEW}_x86_g3_graphonly/pytorch_model.bin" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode none \
  --grpo_train_file data/testing/grpo_data_rl_train_half_cfg.jsonl \
  --grpo_group_size 32 \
  --grpo_score_chunk_size 4 \
  --grpo_gen_temperature 0.7 \
  --grpo_epochs 1 \
  --grpo_lr 1e-6 \
  --grpo_passk_k 5 \
  --grpo_simko_k 4 \
  --grpo_loss_pooling seq \
  --grpo_no_compile_penalty -2.0 \
  --grpo_compile_reward 0.0 \
  --grpo_partial_reward_cap 2.0 \
  --grpo_perfect_base_reward 3.0 \
  --grpo_perfect_bonus 1.5 \
  --grpo_unique_test_bonus 0.25 \
  --grpo_duplicate_penalty 0.25 \
  --grpo_max_new_tokens 512 \
  --grpo_test_timeout 8 \
  --grpo_reward_workers 32 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --name_suffix _x86_g3_simko
```

### Binary pass@10 GRPO on G3 graph-only

```bash
NEW=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128

python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --use_grpo \
  --grpo_checkpoint "artifacts/${NEW}_x86_g3_graphonly/pytorch_model.bin" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode none \
  --grpo_train_file data/testing/grpo_data_rl_train_half_cfg.jsonl \
  --grpo_group_size 32 \
  --grpo_score_chunk_size 4 \
  --grpo_gen_temperature 0.7 \
  --grpo_epochs 1 \
  --grpo_lr 1e-6 \
  --grpo_reward_mode binary \
  --grpo_binary_fail_reward -1.0 \
  --grpo_passk_k 10 \
  --grpo_loss_pooling seq \
  --grpo_max_new_tokens 512 \
  --grpo_test_timeout 8 \
  --grpo_reward_workers 32 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --name_suffix _x86_g3_binary_pk10_g32_grpo
```

### RS-SFT all-arms continuation

```bash
python configs/run_sweeps_antigravity.py \
  --experiment qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128 \
  --name_suffix _x86_g3_rs_sft_allarms \
  --encoder gcb \
  --max_risk high \
  --hardware_profile h200 \
  --force_rerun \
  --sft_checkpoint artifacts/qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_graphonly/pytorch_model.bin \
  --train_file data/testing/rs_sft_x86_8b_allarms_all.jsonl \
  --eval_file data/testing/grpo_data_cfg.jsonl \
  --compile_dataset data/testing/compile-test2_cfg.jsonl \
  --pass_dataset data/testing/grpo_data_cfg.jsonl \
  --epochs 2 \
  --sft_lr 5e-7 \
  --lora_r 64 \
  --lora_alpha 128 \
  --load_4bit 0 \
  --attn_implementation sdpa \
  --gradient_checkpointing 1 \
  --train_batch_size 4 \
  --grad_accum 4 \
  --qwen_prefix_tokens 16 \
  --qwen_prefix_gate_init 0.2 \
  --qwen_prefix_rms_match 1 \
  --decoder_prompt_max_length 2048 \
  --prompt_fit_assembly 1 \
  --prompt_assembly_mode none \
  --auto_cfg 0 \
  --max_block_instrs 24 \
  --dfg_mode edges \
  --position_scheme roberta \
  --use_reasoning 0 \
  --eval_max_new_tokens 768 \
  --generation_batch_size 64 \
  --num_samples 5 \
  --pass_num_samples 10 \
  --metric_workers 192 \
  --save_strategy epoch \
  --save_total_limit 2
```

## 8. Direct Inference / Metrics / Reranking

### Required graph env for direct inference on adapter checkpoints

```bash
export GRAPH_ENCODER_PEFT=lora
export GRAPH_DECODER_PEFT=lora
export GRAPH_LORA_R=64
export GRAPH_LORA_ALPHA=128
export GRAPH_QWEN_PREFIX_TOKENS=16
export GRAPH_ATTN_IMPLEMENTATION=sdpa
export GRAPH_USE_REASONING=0
export GRAPH_DECODER_PROMPT_MAX_LENGTH=768
export GRAPH_QUIET=1
```

### Full DFG visibility during eval

```bash
export GRAPH_MAX_DATAFLOW_EDGES=4096
```

### K50 pass inference

```bash
NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
HEAD=${NEW}_fitutpk5_grpo

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
```

### K50 pass@k metrics

```bash
python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions results/${HEAD}_pass_predictions_k50.json \
  --k_values 1,5,10,25,50 \
  --workers 128
```

### Compile/CodeBLEU inference

```bash
OUT=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_rs_sft_withB_ref_ultralite

python scripts/evaluation/graph_inference_antigravity.py \
  --dataset data/testing/compile-test2_cfg.jsonl \
  --decoder_model Qwen/Qwen3-8B \
  --output results/${OUT}_compile_predictions.json \
  --checkpoint artifacts/${OUT}/pytorch_model.bin \
  --limit 0 \
  --num_samples 5 \
  --generation_batch_size 4 \
  --max_new_tokens 768 \
  --decoder_prompt_max_length 2048
```

### Pass inference for a 8B x86 checkpoint

```bash
OUT=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_rs_sft_withB_ref_ultralite

python scripts/evaluation/graph_inference_antigravity.py \
  --dataset data/testing/grpo_data_cfg.jsonl \
  --decoder_model Qwen/Qwen3-8B \
  --output results/${OUT}_pass_predictions.json \
  --checkpoint artifacts/${OUT}/pytorch_model.bin \
  --limit 0 \
  --num_samples 10 \
  --generation_batch_size 4 \
  --max_new_tokens 768 \
  --decoder_prompt_max_length 2048
```

### CodeBLEU

```bash
SFT=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_synthzp_sft

python scripts/evaluation/graph_codebleu_antigravity.py \
  --predictions results/${SFT}_compile_predictions.json \
  | tee results/sweeps_antigravity/${SFT}_codebleu.json
```

### Compiled-only CodeBLEU

```bash
python scripts/evaluation/graph_codebleu_antigravity.py \
  --predictions results/${SFT}_compile_predictions.json \
  --compiled_only \
  --workers 128 \
  | tee results/sweeps_antigravity/${SFT}_codebleu_compiled_only.json
```

### Compile@k

```bash
python scripts/evaluation/graph_compile_at_k_antigravity.py \
  --predictions results/${SFT}_compile_predictions.json \
  --k_values 1,5 \
  | tee results/sweeps_antigravity/${SFT}_compile_at_k.json
```

### Pass@k

```bash
python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions results/${SFT}_pass_predictions.json \
  --k_values 1,5,10 \
  --workers 128 \
  | tee results/sweeps_antigravity/${SFT}_pass_at_k.json
```

### Fair compile-cluster-vote rerank

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

### Oracle rerank diagnostic only

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

### Analyze rerank reports

```bash
python scripts/evaluation/analyze_rerank_reports_antigravity.py \
  --compile_report results/${HEAD}_pass_predictions_k50_rerank_compile_report.json \
  --oracle_report results/${HEAD}_pass_predictions_k50_oracle_report.json \
  --output results/${HEAD}_pass_predictions_k50_rerank_gap_analysis.json \
  --top_examples 50
```

### Validate whether interrupted prediction JSONs are complete

```bash
export OUT=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_rs_sft_withB_ref_ultralite

python - <<'PY'
import json, os
from pathlib import Path

out = os.environ["OUT"]
for kind, expected in [("pass", 154), ("compile", 126)]:
    p = Path(f"results/{out}_{kind}_predictions.json")
    print("\n", p)
    if not p.exists():
        print("MISSING")
        continue
    try:
        data = json.loads(p.read_text())
        print("valid_json:", True)
        print("rows:", len(data), "expected:", expected)
        print("complete:", len(data) == expected)
    except Exception as e:
        print("valid_json:", False, e)
PY
```

## 9. ARM64 Flutter Plan

### Shared ARM64 settings

```bash
python -m py_compile configs/run_sweeps_antigravity.py

NEW=qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128
METRIC_WORKERS=${METRIC_WORKERS:-64}
GEN_BS=${GEN_BS:-4}

python configs/run_sweeps_antigravity.py \
  --encoder gcb \
  --max_risk high \
  --epochs 4 \
  --lora_r 64 \
  --lora_alpha 128 \
  | grep qwen-9b-base | grep 5e6

COMMON=(
  --experiment "$NEW"
  --encoder gcb
  --max_risk high
  --hardware_profile auto
  --force_rerun
  --train_file data/datasets/flutter_train_cfg.jsonl
  --eval_file data/datasets/flutter_eval_cfg.jsonl
  --compile_dataset data/datasets/flutter_eval_cfg.jsonl
  --pass_dataset data/datasets/flutter_eval_cfg.jsonl
  --prompt_fit_assembly 1
  --auto_cfg 1
  --max_block_instrs 24
  --epochs 4
  --sft_lr 5e-6
  --lora_r 64
  --lora_alpha 128
  --load_4bit 0
  --attn_implementation sdpa
  --gradient_checkpointing 1
  --train_batch_size 1
  --grad_accum 64
  --decoder_prompt_max_length 8192
  --eval_max_new_tokens 1024
  --generation_batch_size "$GEN_BS"
  --num_samples 3
  --pass_num_samples 8
  --metric_workers "$METRIC_WORKERS"
)
```

### ARM64 A0 text-only

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 0 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a0_textonly
```

### ARM64 A2 wide hybrid

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a2_wide
```

### ARM64 A3 encoder-only

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode none \
  --name_suffix _arm_a3_enconly
```

### ARM64 A1 hybrid16

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --qwen_prefix_tokens 16 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a1_hybrid16
```

### ARM64 dry run example

```bash
python configs/run_sweeps_antigravity.py "${COMMON[@]}" \
  --dry_run \
  --qwen_prefix_tokens 128 \
  --prompt_assembly_mode full \
  --prompt_clean_asm 1 \
  --name_suffix _arm_a2_wide_dry
```

### ARM64 metrics-only recovery loop

```bash
for SUFFIX in _arm_a0_textonly _arm_a2_wide _arm_a3_enconly _arm_a1_hybrid16; do
  EXTRA=()
  case "$SUFFIX" in
    _arm_a0_textonly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 0 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _arm_a2_wide) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 128 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
    _arm_a3_enconly) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 128 --prompt_assembly_mode none) ;;
    _arm_a1_hybrid16) EXTRA=(--skip_training --skip_inference --qwen_prefix_tokens 16 --prompt_assembly_mode full --prompt_clean_asm 1) ;;
  esac
  python configs/run_sweeps_antigravity.py "${COMMON[@]}" "${EXTRA[@]}" --name_suffix "$SUFFIX"
done
```

### ARM64 winner GRPO

```bash
WIN=${NEW}_arm_a2_wide
WIN_PREFIX_TOKENS=128
WIN_ASM_MODE=full
WIN_CLEAN_ASM=1

GRPO_COMMON=(
  --experiment "$NEW"
  --name_suffix _arm_grpo
  --encoder gcb
  --max_risk high
  --hardware_profile auto
  --force_rerun
  --use_grpo
  --grpo_checkpoint "artifacts/${WIN}/pytorch_model.bin"
  --grpo_train_file data/datasets/flutter_train_cfg.jsonl
  --compile_dataset data/datasets/flutter_eval_cfg.jsonl
  --pass_dataset data/datasets/flutter_eval_cfg.jsonl
  --prompt_fit_assembly 1
  --auto_cfg 1
  --max_block_instrs 24
  --qwen_prefix_tokens "$WIN_PREFIX_TOKENS"
  --prompt_assembly_mode "$WIN_ASM_MODE"
  --prompt_clean_asm "$WIN_CLEAN_ASM"
  --grpo_passk_k 5
  --grpo_group_size 8
  --grpo_score_chunk_size 2
  --grpo_epochs 1
  --grpo_lr 5e-7
  --grpo_perfect_bonus 1.5
  --grpo_max_new_tokens 1024
  --grpo_test_timeout 6
  --grpo_reward_workers 32
  --load_4bit 0
  --attn_implementation sdpa
  --gradient_checkpointing 1
  --train_batch_size 1
  --grad_accum 64
  --lora_r 64
  --lora_alpha 128
  --decoder_prompt_max_length 8192
  --use_reasoning 0
  --eval_max_new_tokens 1024
  --generation_batch_size 2
  --num_samples 5
  --pass_num_samples 16
  --metric_workers "$METRIC_WORKERS"
  --save_strategy epoch
  --save_total_limit 2
)

python configs/run_sweeps_antigravity.py "${GRPO_COMMON[@]}"
```

### ARM64 GRPO metrics-only recovery

```bash
python configs/run_sweeps_antigravity.py "${GRPO_COMMON[@]}" \
  --skip_training \
  --skip_inference
```

## 10. Hugging Face Upload / Download

### Download all uploaded checkpoint files with new `hf` CLI

```bash
hf download "$HF_REPO" \
  --repo-type model \
  --include "artifacts/*/pytorch_model.bin" \
  --include "artifacts/*/training_args.bin" \
  --local-dir .
```

### Upload one checkpoint folder from Windows PowerShell

```powershell
python -m pip install -U huggingface_hub

$env:HF_TOKEN = "hf_YOUR_TOKEN_HERE"
$env:HF_REPO = "YOUR_USERNAME/qwen9b-antigravity-fitutpk5-grpo"
$env:CKPT_DIR = "C:\Users\Raafat Abualazm\Desktop\Train Data\experiment_workspace\artifacts\qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo"

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

### Upload loader/eval scripts from Windows PowerShell

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

### Upload specific pod output, with `OUT` exported

```bash
export HF_REPO=YOUR_USERNAME/YOUR_REPO
export HF_TOKEN=hf_YOUR_TOKEN
export OUT=qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_rs_sft_withB_ref_lite

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo = os.environ["HF_REPO"]
token = os.environ["HF_TOKEN"]
out = os.environ["OUT"]

api = HfApi(token=token)
api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)

model_dir = Path("artifacts") / out
for name in ["pytorch_model.bin", "training_args.bin"]:
    path = model_dir / name
    if path.is_file():
        print("uploading", path)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"artifacts/{out}/{name}",
            repo_id=repo,
            repo_type="model",
        )

for path in [
    Path("results") / f"{out}_compile_predictions.json",
    Path("results") / f"{out}_pass_predictions.json",
    Path("results/sweeps_antigravity") / f"{out}.json",
    Path("results/sweeps_antigravity") / f"{out}_compile_stats.csv",
    Path("results/sweeps_antigravity") / f"{out}_pass_stats.csv",
]:
    if path.is_file():
        print("uploading", path)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=str(path),
            repo_id=repo,
            repo_type="model",
        )
PY
```

## 11. Local / Azure Commands

### Azure account and regional usage checks

```bash
az account show
az vm list-usage --location eastus -o table
```

### Family-specific quota check

```bash
az vm list-usage --location eastus --query "[?contains(localName, 'Standard')]" -o table
```

## 12. Paper and Thesis Build Commands

### Paper build

```powershell
pdflatex -interaction=nonstopmode -halt-on-error paper_graph_ensemble_20260707.tex
bibtex paper_graph_ensemble_20260707
pdflatex -interaction=nonstopmode -halt-on-error paper_graph_ensemble_20260707.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_graph_ensemble_20260707.tex
```

### Thesis build

```powershell
cd "C:\Users\Raafat Abualazm\Documents\PhD Data\cairo_thesis_merged"
pdflatex -interaction=nonstopmode -halt-on-error mthesis.tex
biber mthesis
pdflatex -interaction=nonstopmode -halt-on-error mthesis.tex
pdflatex -interaction=nonstopmode -halt-on-error mthesis.tex
```

### Check LaTeX logs for unresolved citations/references

```powershell
rg -n "undefined|Citation .* undefined|There were undefined|Label\(s\) may have changed|Please \(re\)run Biber|Rerun Biber|LaTeX Warning:.*Citation|LaTeX Warning:.*Reference" paper_graph_ensemble_20260707.log

rg -n "undefined|Citation .* undefined|There were undefined|Label\(s\) may have changed|Please \(re\)run Biber|Rerun Biber|LaTeX Warning:.*Citation|LaTeX Warning:.*Reference" "C:\Users\Raafat Abualazm\Documents\PhD Data\cairo_thesis_merged\mthesis.log"
```

## 13. Notes on Workers and Throughput

Use these flags rather than environment variables when possible:

```bash
--metric_workers 128
--metric_workers 192
--grpo_reward_workers 32
--grpo_reward_workers 64
--grpo_reward_workers 128
```

`graph_pass_at_k_antigravity.py`, `graph_compile_at_k_antigravity.py`, and
`rerank_predictions_antigravity.py` benefit most from large CPU counts. GRPO
reward scoring benefits, but less linearly, because reward scoring is grouped
around generated completions and Dart subprocesses.

## 14. Session Verification, Sync, and Recovery Commands

Commands actually run to validate and ship the 2026-07 pipeline fixes (graph
gradient checkpointing, prefix RMS matching, DFG wiring) between local Windows
and the training pods. Local test suites are pure CPU/offline; no GPU needed.

### Local test suites (run before every sync)

```bash
python scripts/data/test_graph_preprocessing_fixes.py
python scripts/training/grpo_selfcheck.py
```

### Build a hash-verified sync bundle (Windows/Git Bash)

`upload_filelist.txt` lists the changed files, one per line, repo-relative.

```bash
sha256sum $(cat upload_filelist.txt) > pipeline_fixes_<date>.sha256
tar czf pipeline_fixes_<date>.tar.gz -T upload_filelist.txt pipeline_fixes_<date>.sha256
sha256sum pipeline_fixes_<date>.tar.gz | tee pipeline_fixes_<date>.tar.gz.sha256
```

### Upload and verify on the pod

```bash
scp pipeline_fixes_<date>.tar.gz root@<pod>:/workspace/

# on the pod
sha256sum pipeline_fixes_<date>.tar.gz        # compare against the printed .tar.gz.sha256
tar xzf pipeline_fixes_<date>.tar.gz
sha256sum -c pipeline_fixes_<date>.sha256      # expect OK for every file
```

### Dart SDK install on a fresh Lightning/teamspace pod

Needed before any GRPO reward scoring or compile@k/pass@k metrics step; a
missing `dart` binary makes GRPO fail silently (every reward pinned at the
no-compile penalty) or crashes the metrics stage outright.

```bash
cd ~
curl -O https://storage.googleapis.com/dart-archive/channels/stable/release/latest/sdk/dartsdk-linux-x64-release.zip
unzip -q dartsdk-linux-x64-release.zip
~/dart-sdk/bin/dart --version
```

### CUDA allocator fragmentation guard

Set this before training on any pod not already covered by the runner's
remote-profile default; it prevented a near-miss OOM (994 MiB requested with
871 MiB free) during an x86 SFT run.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Dataflow edge cap (avoid silently dropping most of a dense function's DFG)

The default `GRAPH_MAX_DATAFLOW_EDGES=512` truncates large functions'
cross-block dataflow edges to the first 512 by `(source, target)` order,
which systematically drops edges from later blocks. Raise it before tokenizing
if the tokenization log prints `[build_cross_block_dfg] NOTICE: capping
dataflow edges ...`.

```bash
export GRAPH_MAX_DATAFLOW_EDGES=8192
```

### Metrics-only recovery after a Dart-install crash

If training and inference completed but the metrics stage crashed with
`Dart binary is not runnable`, install Dart (above) and rerun the same arm
command with both flags appended; the runner reuses the existing
`*_predictions.json` files and only recomputes CodeBLEU/compile@k/pass@k.

```bash
--skip_training --skip_inference
```

### Known gap: repair-arm commands not reconstructed here

`withB_ref_lite`, `withB_ref_ultralite`, `binary_repair_ultralite`, and
`style_repair_ultralite` (referenced in `CLAUDE_SUMMARY_20260707.md` and the
paper's Table 2 rerank results, checkpoints archived under
`results-20260707*/`) were run directly against the pod and their exact CLI
invocations were not captured in this transcript. Do not infer their flags
from the RS-SFT templates above by pattern-matching; pull the literal
commands from pod shell history, the other assistant's session log, or
`artifacts/<name>/training_args.bin` before citing them as reproducible.

## 154-task JIT/pass-harness compile correction (2026-07-08)

This fixes the old `pass@1 > compile@1` discrepancy caused by comparing
strict Dart AOT compile against JIT `dart run` pass@k. The corrected metric
uses the same candidate-plus-tests source as pass@k and classifies Dart
front-end syntax/type diagnostics as compile failures while counting runtime
assertion failures/timeouts as compiled-but-not-passing.

```bash
python scripts/evaluation/graph_compile_batch_antigravity.py \
  --jobs results/compile154_passharness/jobs_frontier_and_arms.json \
  --output_dir results/compile154_jitpassharness \
  --compile_mode jit_tests \
  --k_values 1,5,10 \
  --workers 16 \
  --timeout 5 \
  --skip_existing 1
```

Outputs:

```text
results/compile154_jitpassharness/summary_compile_at_k_154_passharness.json
results/compile154_jitpassharness/summary_compile_at_k_154_passharness.csv
results/pass154_same_pools/summary_pass_compile_154_jitpassharness.json
results/pass154_same_pools/summary_pass_compile_154_jitpassharness.csv
```

Validation result: 18/18 prediction pools have `pass_at_k <= jit_compile_at_k`
for k=1,5,10. The strict AOT compile results in
`results/compile154_passharness/` should be retained as deployment diagnostics,
not mixed with pass@k as if the metrics were nested.

## GPT-5.5 cleaned-assembly v2 control (2026-07-10)

This is the assembly-only control for the two v2 graph serializers. In
`llm_baselinev2.py`, `--mode asm` uses the cleaned assembler dump by default;
do not add `--raw_assembly` for this arm. Azure sampling controls are omitted
so all three v2 arms use the deployment defaults recorded in their metadata.

```bash
python llm_baselinev2.py \
  --provider azure \
  --input data/testing/grpo_data_cfg.jsonl \
  --output GPT55Abl/preds_gpt55_asm2 \
  --model gpt-chat-latest \
  --mode asm \
  --samples 10 \
  --max_tokens 8192 \
  --resume

python scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions GPT55Abl/preds_gpt55_asm2 \
  --k_values 1,5,10 \
  --workers 16 \
  | tee GPT55Abl/preds_gpt55_asm2_passk

python scripts/evaluation/graph_compile_at_k_antigravity.py \
  --predictions GPT55Abl/preds_gpt55_asm2 \
  --k_values 1,5,10 \
  --compile_mode jit_tests \
  --workers 16 \
  | tee GPT55Abl/preds_gpt55_asm2_compilek
```

Recorded results: pass@1/5/10 = 0.5792/0.6789/0.7078 and aligned
compile@1/5/10 = 0.9617/0.9838/0.9870. Against this 109-task pass@10
reference, assembly plus address-aligned CFG is `+2/-3` (`p=1.0`) and
CFG-organized block instructions is `+1/-2` (`p=1.0`).

## 15. Leakage-Free Graph-v2 Confirmatory Rerun (2026-07-11)

Stop any process using the historical `_cfg` files or an old graph checkpoint.
Build the fresh 58-file transfer bundle from PowerShell:

```powershell
tar -czf clean_study_bundle_graphv2.tar.gz -T upload_clean_study_filelist.txt
Get-FileHash clean_study_bundle_graphv2.tar.gz -Algorithm SHA256
```

Upload it to the fresh machine and extract it from the repository root. Replace
the SSH target with the current rental's target:

```powershell
scp -P <PORT> .\clean_study_bundle_graphv2.tar.gz root@<HOST>:/workspace/
```

```bash
cd /workspace
tar xzf clean_study_bundle_graphv2.tar.gz
python -m pip install -r requirements.txt
dart --version
gdb --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
tmux new -s antigravity-graphv2
```

Do the CPU preflight before starting paid GPU time:

```bash
python scripts/run_leakage_free_study.py \
  --phase preflight \
  --execute \
  --full_reward_audit \
  --reward_workers 32
```

Run the 20-hour priority matrix on the RTX PRO 6000. The driver is resumable and
reuses one candidate pool for aligned compile@k and pass@k:

```bash
python scripts/run_leakage_free_study.py \
  --phase gpu \
  --execute \
  --seeds 42,43,44 \
  --budget_hours 20 \
  --metric_workers 64 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

Then run the data-scale arm if time remains:

```bash
python scripts/run_leakage_free_study.py \
  --phase expanded \
  --execute \
  --seeds 42 \
  --budget_hours 20 \
  --metric_workers 64 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

GRPO is not automatic. First run no-update reward rollouts:

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

Only if the report clears the threshold, run the optional binary-reward update:

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

The exact data-preparation and reward-audit commands are encoded in the preflight.
Local structural preflight passed with 154/154 benchmark rows, 770 leakage-clean
original train rows, 83 validation rows, and 1,726/1,726 synthetic rows. Source
recompilation recovered 941/1,081 train captures before removing 168 exact
duplicates and three validation near-duplicates; validation retained 83/114 after
16 exact duplicates and 15 malformed/context-dependent fragments were removed.
The local executable audit passed 1,725/1,725 references with zero failures; rerun it
on the remote because Dart/platform drift can change the harness outcome. Never
add `compile-test2_cfg.jsonl`, `grpo_data_cfg.jsonl`, or their
graph-v2 derivatives to training.

The new graph arms use `GRAPH_STRICT_GRAPH=1`, uncapped DFG construction, distinct
reverse relations, sinusoidal block-order positions, dynamic 4-64 prefix slots,
per-slot gates, and RMS matching. These settings make every historical graph
checkpoint incompatible with this study.

### Source recompilation and final graph-v2 rebuild (2026-07-11)

The historical 513/58 truncated-dump fallback was replaced by fresh AOT/GDB
capture. These are the exact recovery commands:

```bash
python -u scripts/data/rebuild_sft_assembly.py \
  --input data/datasets/dart_all_cfg_clean_train.jsonl \
  --output data/datasets/dart_all_rebuilt_train.jsonl \
  --rejected data/datasets/dart_all_rebuilt_train.rejected.json \
  --report data/datasets/dart_all_rebuilt_train.report.json \
  --workers 16 --timeout 240 --retries 1 \
  --expected_input_rows 1081 --allow_rejects

python -u scripts/data/rebuild_sft_assembly.py \
  --input data/datasets/dart_all_cfg_clean_train.jsonl \
  --include_lines_from data/datasets/dart_all_rebuilt_train.rejected.json \
  --output data/datasets/dart_all_rebuilt_train_retry.jsonl \
  --rejected data/datasets/dart_all_rebuilt_train_retry.rejected.json \
  --report data/datasets/dart_all_rebuilt_train_retry.report.json \
  --workers 8 --timeout 240 --retries 1 \
  --expected_input_rows 1081 --allow_rejects

python -u scripts/data/rebuild_sft_assembly.py \
  --input data/datasets/dart_all_cfg_clean_train.jsonl \
  --include_lines_from data/datasets/dart_all_rebuilt_train_tree_shaken.retry.json \
  --output data/datasets/dart_all_rebuilt_train_tree_shaken.jsonl \
  --rejected data/datasets/dart_all_rebuilt_train_tree_shaken.rejected.json \
  --report data/datasets/dart_all_rebuilt_train_tree_shaken.report.json \
  --workers 4 --timeout 240 --retries 1 \
  --expected_input_rows 1081 --allow_rejects

python scripts/data/merge_jsonl_by_source.py \
  --input data/datasets/dart_all_rebuilt_train.jsonl \
  --input data/datasets/dart_all_rebuilt_train_retry.jsonl \
  --input data/datasets/dart_all_rebuilt_train_tree_shaken.jsonl \
  --output data/datasets/dart_all_rebuilt_train_unique.jsonl \
  --report data/datasets/dart_all_rebuilt_train_unique.report.json

python -u scripts/data/rebuild_sft_assembly.py \
  --input data/datasets/dart_all_cfg_clean_validation.jsonl \
  --output data/datasets/dart_all_rebuilt_validation_unique.jsonl \
  --rejected data/datasets/dart_all_rebuilt_validation_unique.rejected.json \
  --report data/datasets/dart_all_rebuilt_validation_unique.report.json \
  --workers 8 --timeout 240 --retries 1 \
  --expected_input_rows 114 --dedupe_source --allow_rejects
```

The driver then drops the three train-side validation near-duplicates, rebuilds
all graph-v2 files with 20-instruction lossless fallthrough splitting and no DFG
edge cap, and runs the full protocol/reward audit:

```bash
python scripts/run_leakage_free_study.py \
  --phase preflight \
  --execute \
  --full_reward_audit \
  --reward_workers 20 \
  --budget_hours 4
```

### Hierarchical region-compression control (2026-07-13)

This arm keeps the selected no-GINE block encoder, global block attention, and
dynamic prefix budget unchanged. It partitions the forward CFG into bounded
maximal straight-line regions, attention-pools each region, and residually
injects the region summary into every member block. Branches, joins, calls,
backedges, exceptional edges, and the eight-block cap form region boundaries;
DFG edges do not define regions.

Do not replace source files on a host while an older experiment queue is still
launching subprocesses. After that queue completes, sync the updated source and
dry-plan the single new arm:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-isolation \
  --isolation_variants prefix_no_gine_regions \
  --seed 42 \
  --metric_workers 64 \
  --budget_hours 2
```

Then launch it explicitly:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-isolation \
  --isolation_variants prefix_no_gine_regions \
  --execute \
  --seed 42 \
  --metric_workers 64 \
  --budget_hours 2 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

The result name is
`qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_prefix_no_gine_regions`.
Its provenance must contain `GRAPH_REGION_COMPRESSION=linear_residual` and
`GRAPH_REGION_MAX_BLOCKS=8`.

### Assembly encoder and multi-vector block controls (2026-07-14)

These two arms keep the no-GINE topology, dynamic Qwen prefix budget, training
data, and ten-candidate evaluation budget fixed:

- `prefix_no_gine_clap` replaces GraphCodeBERT with the assembly-pretrained
  `hustcw/clap-asm` encoder pinned at revision
  `620f4beba2edce172e8f35e263399716494950c9`. Its native instruction tokenizer
  and token-type IDs are used; it is not the historical generic tokenizer
  alias.
- `prefix_no_gine_multivector4` replaces hard one-vector `[CLS]` block pooling
  with four learned masked queries over each block's token states. One mean
  topology vector per block is still used where a graph node is required, but
  all four semantic vectors survive into global attention and the prefix
  resampler. Dynamic prefix allocation remains a function of block count, not
  the expanded four-vector sequence length.

Do not deploy changed source files to a host while an older queue is still
launching subprocesses. Once the queue is complete, dry-plan and launch both
seed-42 controls:

```bash
python scripts/run_graphv2_followups.py \
  --phase x86-encoder \
  --encoder_variants prefix_no_gine_clap,prefix_no_gine_multivector4 \
  --seed 42 \
  --metric_workers 64 \
  --budget_hours 4

python scripts/run_graphv2_followups.py \
  --phase x86-encoder \
  --encoder_variants prefix_no_gine_clap,prefix_no_gine_multivector4 \
  --execute \
  --seed 42 \
  --metric_workers 64 \
  --budget_hours 4 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

The combined estimate is 3.2 GPU hours. Compare these arms with
`prefix_no_gine` and `prefix_no_gine_regions` using:

```bash
python scripts/evaluation/analyze_graphv2_clean_study.py \
  --results_dir results-20260713 \
  --bootstrap_reps 10000
```

Choose the x86 architecture using functional pass@k first, with aligned JIT
compile, paired task changes, complexity strata, and representation cost as
secondary evidence. Confirm the selected x86 arm on seeds 43 and 44 before
opening the ARM64 result. This prevents ARM64 from becoming validation data for
architecture selection.

### Signature-scrubbed evaluation after the x86 freeze (2026-07-14)

The existing 154 tasks can be rebuilt under the neutral symbol `candidate` with
the exact signature withheld from the decoder. This is a robustness ablation,
not a fresh holdout, because the tasks have already influenced model selection.
The script recompiles every source before rebuilding graph-v2; deleting only the
JSON signature would leave semantic function names in the assembly.

```bash
python -m scripts.data.build_signature_scrubbed_eval \
  --input data/testing/grpo_data_graphv2.jsonl \
  --output data/testing/grpo_data_graphv2_signature_scrubbed.jsonl \
  --public_output data/testing/grpo_data_graphv2_signature_scrubbed_public.jsonl \
  --rejects data/testing/grpo_data_graphv2_signature_scrubbed.rejected.json \
  --benchmark_kind existing_ablation \
  --expected_rows 154 \
  --workers 8 \
  --max_block_instrs 20
```

For genuinely fresh evidence, first write the complete frozen x86 configuration,
checkpoint identity, representation winner, prefix density, gate initialization,
and selection timestamp to `results-20260713/graphv2_x86_freeze.json`. Only then
generate a new HumanEval-style set. The generator hashes the freeze manifest,
neutralizes each semantic function name to `candidate` before AOT compilation,
withholds the entire signature, validates 8-12 hidden tests, and decontaminates
against every supplied prior corpus.

```bash
python generate_synthetic_tasks_parallel.py \
  --eval-jsonl data/testing/grpo_data_graphv2.jsonl \
  --decontam-jsonl data/datasets/dart_all_graphv2_train.jsonl \
  --decontam-jsonl data/datasets/dart_all_graphv2_validation.jsonl \
  --decontam-jsonl data/datasets/synthetic_pool_reward_clean_graphv2.jsonl \
  --profile humaneval \
  --signature-scrubbed \
  --freeze-manifest results-20260713/graphv2_x86_freeze.json \
  --target-count 154 \
  --per-provider 400 \
  --parallel 4 \
  --providers <COMMA_SEPARATED_PROVIDER_KEYS> \
  --task-prefix fresh_sigless \
  --out data/testing/fresh_dart_signature_scrubbed_raw.jsonl \
  --public-out data/testing/fresh_dart_signature_scrubbed_public.jsonl

python scripts/data/build_graph_v2_jsonl.py \
  --input data/testing/fresh_dart_signature_scrubbed_raw.jsonl \
  --output data/testing/fresh_dart_signature_scrubbed_graphv2.jsonl \
  --rejected data/testing/fresh_dart_signature_scrubbed_graphv2.rejected.jsonl \
  --summary data/testing/fresh_dart_signature_scrubbed_graphv2.summary.json \
  --expected_input_rows 154 \
  --expected_output_rows 154 \
  --max_block_instrs 20 \
  --max_dataflow_edges 0
```

Evaluate the frozen model on this set exactly once. A disappointing holdout
result is reported, not used to tune another representation, prefix density, or
gate value. Keep the private JSONL for scoring and release the source/tests only
after the confirmatory evaluation is locked.

### ARM64 replication after the x86 architecture freeze (2026-07-14)

The recommended ARM64 phase trains only the frozen x86 winner on the immutable
1,371-row training split and evaluates on the disjoint 343-row split. Accepted
winner labels are `prefix_no_gine`, `prefix_no_edges`, `prefix_cfg`,
`prefix_cfg_dfg`, `prefix_no_gine_regions`,
`prefix_no_gine_multivector4`, and `prefix_no_gine_clap`.

Dry-plan first, replacing `<X86_WINNER>` with the frozen label:

```bash
python scripts/run_arm64_graphv21_study.py \
  --phase selected \
  --selected_architecture <X86_WINNER> \
  --selected_seeds 42,43,44 \
  --metric_workers 64 \
  --budget_hours 10
```

Then execute the same plan:

```bash
python scripts/run_arm64_graphv21_study.py \
  --phase selected \
  --selected_architecture <X86_WINNER> \
  --selected_seeds 42,43,44 \
  --execute \
  --metric_workers 64 \
  --budget_hours 10 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts
```

Three ARM64 seeds are estimated at 7.5 hours for the existing no-GINE arm,
8.1 hours for region compression or CLAP-ASM, and 9.0 hours for four-vector
pooling. ARM64 is an external cross-ISA replication of a frozen architecture,
not another sweep from which to select the method.
