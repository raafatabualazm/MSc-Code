#!/usr/bin/env bash
set -euo pipefail

# Leakage-free confirmatory x86 study. Override, for example:
#   SEEDS="42" ARMS="prefix_no_edges prefix_cfg prefix_cfg_dfg" RUN_GRPO=0 bash scripts/run_clean_confirmatory_x86.sh

BASE="qwen3-8b-base_lora_enc_dec_r16_5e6_gcb"
DECODER_REV="b968826d9c46dd6066d109eabc6255188de91218"
ENCODER_REV="2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"
SFT_TRAIN="data/datasets/dart_all_cfg_clean_train.jsonl"
SFT_VALID="data/datasets/dart_all_cfg_clean_validation.jsonl"
BENCH="data/testing/grpo_data_cfg.jsonl"
RL_TRAIN="data/testing/grpo_data_rl_train_half_cfg.jsonl"
RL_HELDOUT="data/testing/grpo_data_rl_eval_half_cfg.jsonl"
SEEDS="${SEEDS-42 43 44}"
ARMS="${ARMS-text prefix_no_edges prefix_shuffled prefix_cfg prefix_cfg_dfg prefix_no_gine}"
RUN_REFERENCE="${RUN_REFERENCE:-1}"
RUN_GRPO="${RUN_GRPO:-1}"

common=(
  --experiment "$BASE"
  --encoder gcb
  --max_risk high
  --hardware_profile h200
  --force_rerun
  --decoder_revision "$DECODER_REV"
  --encoder_revision "$ENCODER_REV"
  --train_file "$SFT_TRAIN"
  --eval_file "$SFT_VALID"
  --compile_dataset "$BENCH"
  --pass_dataset "$BENCH"
  --compile_mode jit_tests
  --epochs 4
  --sft_lr 5e-6
  --lora_r 64
  --lora_alpha 128
  --load_4bit 0
  --attn_implementation sdpa
  --gradient_checkpointing 1
  --train_batch_size 4
  --grad_accum 16
  --qwen_prefix_gate_init 0.2
  --qwen_prefix_rms_match 0
  --decoder_prompt_max_length 2048
  --prompt_fit_assembly 1
  --auto_cfg 0
  --max_block_instrs 24
  --position_scheme roberta
  --causal_position_ids cumsum
  --use_reasoning 0
  --eval_max_new_tokens 768
  --generation_batch_size 32
  --num_samples 10
  --pass_num_samples 10
  --metric_workers 128
  --save_strategy epoch
  --save_total_limit 2
)

run_arm() {
  local seed="$1"
  local arm="$2"
  local prefix_tokens prompt_mode clean_asm dfg edge gnn
  case "$arm" in
    text)
      prefix_tokens=0; prompt_mode=full; clean_asm=1; dfg=off; edge=none; gnn=identity ;;
    prefix_no_edges)
      prefix_tokens=16; prompt_mode=none; clean_asm=0; dfg=edges; edge=none; gnn=full ;;
    prefix_shuffled)
      prefix_tokens=16; prompt_mode=none; clean_asm=0; dfg=edges; edge=shuffle; gnn=full ;;
    prefix_cfg)
      prefix_tokens=16; prompt_mode=none; clean_asm=0; dfg=edges; edge=cfg; gnn=full ;;
    prefix_cfg_dfg)
      prefix_tokens=16; prompt_mode=none; clean_asm=0; dfg=edges; edge=full; gnn=full ;;
    prefix_no_gine)
      prefix_tokens=16; prompt_mode=none; clean_asm=0; dfg=edges; edge=full; gnn=identity ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
  esac

  python configs/run_sweeps_antigravity.py "${common[@]}" \
    --seed "$seed" \
    --name_suffix "_clean_s${seed}_${arm}" \
    --qwen_prefix_tokens "$prefix_tokens" \
    --prompt_assembly_mode "$prompt_mode" \
    --prompt_clean_asm "$clean_asm" \
    --dfg_mode "$dfg" \
    --edge_ablation "$edge" \
    --gnn_ablation "$gnn"
}

for seed in $SEEDS; do
  if [[ "$RUN_REFERENCE" == "1" ]]; then
    python configs/run_sweeps_antigravity.py "${common[@]}" \
      --seed "$seed" \
      --name_suffix "_clean_s${seed}_untuned" \
      --skip_training \
      --qwen_prefix_tokens 0 \
      --prompt_assembly_mode full \
      --prompt_clean_asm 1 \
      --dfg_mode off \
      --edge_ablation none \
      --gnn_ablation identity
  fi

  for arm in $ARMS; do
    run_arm "$seed" "$arm"
  done

  if [[ "$RUN_GRPO" == "1" ]]; then
    g3="qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_clean_s${seed}_prefix_cfg_dfg"
    python configs/run_sweeps_antigravity.py \
      --experiment "$BASE" \
      --name_suffix "_clean_s${seed}_heldout_binary" \
      --encoder gcb \
      --max_risk high \
      --hardware_profile h200 \
      --force_rerun \
      --use_grpo \
      --grpo_checkpoint "artifacts/${g3}/pytorch_model.bin" \
      --grpo_train_file "$RL_TRAIN" \
      --compile_dataset "$RL_HELDOUT" \
      --pass_dataset "$RL_HELDOUT" \
      --compile_mode jit_tests \
      --decoder_revision "$DECODER_REV" \
      --encoder_revision "$ENCODER_REV" \
      --seed "$seed" \
      --lora_r 64 \
      --lora_alpha 128 \
      --load_4bit 0 \
      --attn_implementation sdpa \
      --qwen_prefix_tokens 16 \
      --qwen_prefix_gate_init 0.2 \
      --qwen_prefix_rms_match 0 \
      --prompt_assembly_mode none \
      --decoder_prompt_max_length 2048 \
      --prompt_fit_assembly 1 \
      --auto_cfg 0 \
      --max_block_instrs 24 \
      --dfg_mode edges \
      --edge_ablation full \
      --gnn_ablation full \
      --position_scheme roberta \
      --causal_position_ids cumsum \
      --use_reasoning 0 \
      --grpo_group_size 16 \
      --grpo_score_chunk_size 4 \
      --grpo_epochs 1 \
      --grpo_lr 5e-7 \
      --grpo_reward_mode binary \
      --grpo_binary_fail_reward -1.0 \
      --grpo_perfect_base_reward 1.0 \
      --grpo_perfect_bonus 0.0 \
      --grpo_kl_coef 0.0 \
      --grpo_clip_eps 0.15 \
      --grpo_adv_norm mean \
      --grpo_loss_pooling seq \
      --grpo_simko_k 0 \
      --grpo_passk_k 0 \
      --grpo_unique_test_bonus 0.0 \
      --grpo_duplicate_penalty 0.0 \
      --grpo_entropy_coef 0.0 \
      --grpo_overlong_filter 1 \
      --grpo_max_new_tokens 512 \
      --grpo_test_timeout 8 \
      --grpo_reward_workers 64 \
      --train_batch_size 1 \
      --grad_accum 8 \
      --eval_max_new_tokens 768 \
      --generation_batch_size 32 \
      --num_samples 10 \
      --pass_num_samples 10 \
      --metric_workers 128 \
      --save_strategy epoch \
      --save_total_limit 2
  fi
done
