#!/usr/bin/env bash
set -Eeuo pipefail

# Required, intentionally explicit: using the same checkpoint for both variables
# would make the initial KL zero and the Python preflight rejects it.
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to a local compact-conditioned teacher checkpoint}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to the exact local teacher base/model path}"
: "${TEACHER_REVISION:?Set TEACHER_REVISION to an immutable teacher revision}"
: "${STUDENT_WARMSTART_CHECKPOINT:?Set STUDENT_WARMSTART_CHECKPOINT to the student warm-start checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new empty output directory}"

PATCH_ROOT="${TRUE_KD_PATCH_ROOT:-/workspace/true_kd_patch_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/direct_compact_stage/hybrid_training_patch_v2_3}"
REBUILD="${REBUILD:-/workspace/artifacts/compact_fn0_rebuild}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
CODEBOOK="${CODEBOOK:-/workspace/direct_compact_stage/scrubbed_master_v2_release/direct_compact_split_v1/compact_qwen_confirmatory_v1/codebook.json}"
CODEC="${CODEC:-/workspace/direct_compact_stage/scripts/data/build_compact_qwen_v1.py}"

teacher_resolved="$(realpath -m "${TEACHER_CHECKPOINT}")"
student_resolved="$(realpath -m "${STUDENT_WARMSTART_CHECKPOINT}")"
output_resolved="$(realpath -m "${OUTPUT_DIR}")"
if [[ "${output_resolved}" == "${teacher_resolved}" || "${output_resolved}" == "${student_resolved}" ]]; then
  echo "OUTPUT_DIR must be distinct from teacher and student warm-start checkpoints" >&2
  exit 2
fi

schedule_args=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  schedule_args+=(--max_steps "${MAX_STEPS}")
fi

exec /venv/main/bin/python "${PATCH_ROOT}/scripts/training/true_distribution_kd_antigravity.py" \
  --mode dense_full_kl \
  --project_root "${PROJECT_ROOT}" \
  --train_file "${REBUILD}/train_fn0_real.jsonl" \
  --train_seal "${REBUILD}/train_fn0_real.seal.json" \
  --contract "${REBUILD}/fn0_contract.json" \
  --codebook "${CODEBOOK}" \
  --codec_artifact "${CODEC}" \
  --student_model Qwen/Qwen3-8B \
  --student_revision b968826d9c46dd6066d109eabc6255188de91218 \
  --student_tokenizer_json "${TOKENIZER_JSON}" \
  --student_warmstart_checkpoint "${STUDENT_WARMSTART_CHECKPOINT}" \
  --teacher_model "${TEACHER_MODEL}" \
  --teacher_revision "${TEACHER_REVISION}" \
  --teacher_tokenizer_json "${TOKENIZER_JSON}" \
  --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --dtype bf16 \
  --student_load_4bit \
  --teacher_load_4bit \
  --gradient_checkpointing \
  --temperature "${KD_TEMPERATURE:-2.0}" \
  --kd_weight "${KD_WEIGHT:-1.0}" \
  --hard_ce_weight "${HARD_CE_WEIGHT:-0.0}" \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --grad_accum "${GRAD_ACCUM:-16}" \
  --position_chunk_size "${POSITION_CHUNK_SIZE:-32}" \
  --learning_rate "${LEARNING_RATE:-3e-6}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_every_steps "${SAVE_EVERY_STEPS:-50}" \
  "${schedule_args[@]}" \
  "$@"
