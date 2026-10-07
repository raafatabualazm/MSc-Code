#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SFT_DIR="${T5GEMMA_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_v1}"
SFT_CHECKPOINT_NAME="${T5GEMMA_SFT_CHECKPOINT_NAME:-checkpoint-optstep-000174}"
SFT_CHECKPOINT="${SFT_DIR}/${SFT_CHECKPOINT_NAME}"
EVAL_DIR="${T5GEMMA_PREPOST_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_prepost_passk_v1}"
EVAL_READY_SCORE="${T5GEMMA_RS_SFT_READY_SCORE:-${EVAL_DIR}/post_sft_k10_score.json}"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
OUTPUT_DIR="${T5GEMMA_RS_SFT_PILOT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
WAIT_INTERVAL="${T5GEMMA_RS_SFT_WAIT_INTERVAL:-30}"

if ! [[ "${WAIT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "T5GEMMA_RS_SFT_PILOT_BLOCKED invalid wait interval: ${WAIT_INTERVAL}" >&2
  exit 78
fi
if [[ ! -f "${SFT_DIR}/result.json" ]] \
  || [[ "$(/usr/bin/jq -r '.status // empty' "${SFT_DIR}/result.json")" != complete ]] \
  || [[ "$(/usr/bin/jq -r '.latest_checkpoint // empty' "${SFT_DIR}/result.json")" != "${SFT_CHECKPOINT_NAME}" ]] \
  || [[ "$(/usr/bin/jq -r '.no_frontier_api // false' "${SFT_DIR}/result.json")" != true ]] \
  || [[ ! -d "${SFT_CHECKPOINT}" ]]; then
  echo "T5GEMMA_RS_SFT_PILOT_BLOCKED final SFT checkpoint is absent or incomplete" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_RS_SFT_PILOT_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi

printf '%s  %s\n' \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

# The paired pre/post measurement is required to finish first only to avoid
# two model processes competing for the same GPU. Its outcome never selects
# pilot membership, targets, or hyperparameters.
while [[ ! -s "${EVAL_READY_SCORE}" ]]; do
  echo "T5GEMMA_RS_SFT_PILOT_WAITING for paired post-SFT evaluation"
  sleep "${WAIT_INTERVAL}"
done
sleep 15

mkdir -p "${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/harvest_report.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/harvest_report.json")" == complete ]]; then
  echo "T5GEMMA_RS_SFT_PILOT_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_local_rs_sft_pilot.py \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected_rollout_sha256 14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  --expected_f2_sha256 c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  --expected_private_holdback_sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --sft_checkpoint "${SFT_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --pilot_tasks 200 \
  --base_samples 4 \
  --repair_samples 4 \
  --max_repair_parents 2 \
  --gold_replay_ratio 3 \
  --production_min_unique_targets 200 \
  --max_source_tokens 32768 \
  --max_new_tokens 4096 \
  --generation_batch_size 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --evaluation_workers 8 \
  --timeout 30 \
  --stability_runs 2 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16
