#!/usr/bin/env bash
# Legacy graph/text-arm RS-SFT entry point. This is retained for reproducibility
# only; production direct-compact runs use run_finish_rs_sft.sh.
set -Eeuo pipefail

cd /workspace

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/artifacts/text_arm_v2_s44_fixed}"
SOURCE_ROOT="${SOURCE_ROOT:-/workspace/artifacts/text_arm_v2_s44}"
TEXT_SFT_CHECKPOINT="${TEXT_SFT_CHECKPOINT:-${SOURCE_ROOT}/02t_text_sft/pytorch_model.bin}"
ARCHITECTURE_ENV_JSON="${ARCHITECTURE_ENV_JSON:-${SOURCE_ROOT}/02t_text_sft/run_provenance.json}"
RS_SFT_FILE="${RS_SFT_FILE:-${SOURCE_ROOT}/04_teacher_harvest/verified_rs_sft_214.jsonl}"
RS_SFT_MIN_IMPROVEMENT_PP="${RS_SFT_MIN_IMPROVEMENT_PP:-6.0}"

for required_file in \
  "${TEXT_SFT_CHECKPOINT}" \
  "${ARCHITECTURE_ENV_JSON}" \
  "${RS_SFT_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'Required input is missing: %s\n' "${required_file}" >&2
    exit 2
  fi
done

mkdir -p /workspace/locks
exec 9>/workspace/locks/graph_rs_sft_legacy.lock
if ! flock -n 9; then
  printf 'Another legacy graph RS-SFT run holds the lock\n' >&2
  exit 3
fi

schedule_args=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  schedule_args+=(--max_steps "${MAX_STEPS}")
fi

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
  --long_prompt_max_tokens 12288 \
  --gate_generation_batch_size 8 \
  --rollout_samples 8 \
  --rollout_generation_batch_size 16 \
  --limit_tasks 1000 \
  --text_sft_checkpoint "${TEXT_SFT_CHECKPOINT}" \
  --architecture_env_json "${ARCHITECTURE_ENV_JSON}" \
  --text_only \
  --text_eval_file /workspace/data/training/sealed_v2_dev_text_eval_fn0_clean.jsonl \
  --text_finish_rs_sft \
  --rs_sft_file "${RS_SFT_FILE}" \
  --rs_sft_rows_per_epoch 1712 \
  --rs_sft_allow_partial_gold \
  --rs_sft_recertify_facts_gate_mode signature \
  --rs_sft_min_improvement_pp "${RS_SFT_MIN_IMPROVEMENT_PP}" \
  --stage2_epochs 2 \
  "${schedule_args[@]}" \
  "$@"
