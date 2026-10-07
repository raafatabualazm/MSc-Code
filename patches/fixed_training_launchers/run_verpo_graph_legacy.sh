#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

if [[ ! -f /workspace/data.env ]]; then
  printf 'Missing /workspace/data.env\n' >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
source /workspace/data.env
set +a
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required for fail-closed VeRPO judging}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/artifacts/text_arm_v2_s44_fixed}"
VERPO_CHECKPOINT="${VERPO_CHECKPOINT:-${OUTPUT_ROOT}/08b_rejection_sampling_sft/pytorch_model.bin}"
ARCHITECTURE_ENV_JSON="${ARCHITECTURE_ENV_JSON:-${OUTPUT_ROOT}/08b_rejection_sampling_sft/run_provenance.json}"
RS_SFT_FILE="${RS_SFT_FILE:-${OUTPUT_ROOT}/06a_recertified_verified_rs_sft/verified_rs_sft.jsonl}"

for required_file in \
  "${VERPO_CHECKPOINT}" \
  "${ARCHITECTURE_ENV_JSON}" \
  "${RS_SFT_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'Required input is missing: %s\n' "${required_file}" >&2
    exit 2
  fi
done

mkdir -p /workspace/locks
exec 9>/workspace/locks/verpo_v2.lock
if ! flock -n 9; then
  printf 'Another fixed VeRPO run already holds %s\n' \
    /workspace/locks/verpo_v2.lock >&2
  exit 3
fi

trap 'status=$?; printf "[verpo_v2_fixed] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

export GRAPH_SAVE_STRATEGY=steps
export VERPO_JUDGE_MAX_TOKENS="${VERPO_JUDGE_MAX_TOKENS:-12288}"
export VERPO_JUDGE_COMPLETION_RETRIES="${VERPO_JUDGE_COMPLETION_RETRIES:-2}"
export VERPO_JUDGE_RETRY_MAX_TOKENS="${VERPO_JUDGE_RETRY_MAX_TOKENS:-32768}"
export VERPO_JUDGE_THINKING_MODE="${VERPO_JUDGE_THINKING_MODE:-enabled}"
export VERPO_JUDGE_TIMEOUT_SECONDS="${VERPO_JUDGE_TIMEOUT_SECONDS:-180}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRPO_GEN_TOP_P=1.0

/venv/main/bin/python -m scripts.training.run_hybrid_curriculum_antigravity \
  --project_root /workspace \
  --output_root "${OUTPUT_ROOT}" \
  --train_file /workspace/data/training/combined_sealed_v2_train_text.jsonl \
  --eval_file /workspace/data/testing/grpo_data_graphv2.jsonl \
  --functional_eval_file /workspace/data/testing/fresh_graphv2_holdout_s44_rebuilt312.jsonl \
  --teacher_model deepseek-reasoner \
  --teacher_mode batch \
  --seed 44 \
  --max_train_instructions 150 \
  --max_bridge_instructions 199 \
  --min_long_rows 64 \
  --dev_fraction 0.10 \
  --min_long_gate_rows 64 \
  --long_target_max_tokens 3328 \
  --long_generation_max_tokens 3328 \
  --long_prompt_max_tokens 8192 \
  --limit_tasks 1000 \
  --text_only \
  --text_sft_checkpoint /workspace/artifacts/text_arm_v2_s44/02t_text_sft/pytorch_model.bin \
  --text_verpo \
  --verpo_checkpoint "${VERPO_CHECKPOINT}" \
  --architecture_env_json "${ARCHITECTURE_ENV_JSON}" \
  --rs_sft_file "${RS_SFT_FILE}" \
  --verpo_judge \
  --verpo_judge_model "${VERPO_JUDGE_MODEL:-deepseek-v4-pro}" \
  --verpo_judge_concurrency "${VERPO_JUDGE_CONCURRENCY:-8}" \
  --verpo_judge_timeout_seconds "${VERPO_JUDGE_TIMEOUT_SECONDS:-60}" \
  --verpo_judge_max_retries "${VERPO_JUDGE_MAX_RETRIES:-2}" \
  --verpo_judge_weight "${VERPO_JUDGE_WEIGHT:-0.25}" \
  --verpo_full_pass_margin "${VERPO_FULL_PASS_MARGIN:-0.001}" \
  --grpo_reward_mode verpo \
  --grpo_group_size "${GRPO_GROUP_SIZE:-4}" \
  --grpo_batch_size 1 \
  --grpo_grad_accum "${GRPO_GRAD_ACCUM:-4}" \
  --grpo_max_new_tokens "${GRPO_MAX_NEW_TOKENS:-3072}" \
  --grpo_score_chunk_size 1 \
  --grpo_generation_chunk_size 1 \
  --grpo_save_steps 1 \
  --grpo_min_reward_range 0 \
  --max_steps "${MAX_STEPS:-200}" \
  "$@"
