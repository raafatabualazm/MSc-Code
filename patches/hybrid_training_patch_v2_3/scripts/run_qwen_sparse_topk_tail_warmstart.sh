#!/usr/bin/env bash
# Offline sparse top-k+tail auxiliary over primary direct-compact sequence SFT.
# No teacher/API process is launched here.
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPARSE_AUXILIARY_WEIGHT="${SPARSE_AUXILIARY_WEIGHT:-0.10}"
MINIMUM_SPARSE_ELIGIBLE_FRACTION="${MINIMUM_SPARSE_ELIGIBLE_FRACTION:-1.0}"
SEED="${SEED:-44}"

: "${SEQUENCE_TRAIN_JSONL:?set SEQUENCE_TRAIN_JSONL}"
: "${SEQUENCE_TRAIN_SEAL:?set SEQUENCE_TRAIN_SEAL}"
: "${SEQUENCE_SCHEDULE_JSONL:?set SEQUENCE_SCHEDULE_JSONL}"
: "${SEQUENCE_BUILD_MANIFEST:?set SEQUENCE_BUILD_MANIFEST}"
: "${TEACHER_PARSEABLE_JSONL:?set TEACHER_PARSEABLE_JSONL}"
: "${TEACHER_AUDIT_JSON:?set TEACHER_AUDIT_JSON}"
: "${STUDENT_TOKENIZER_JSON:?set STUDENT_TOKENIZER_JSON}"
: "${STUDENT_EOS_TOKEN_ID:?set STUDENT_EOS_TOKEN_ID}"
: "${COMPACT_CONTRACT:?set COMPACT_CONTRACT}"
: "${COMPACT_CODEBOOK:?set COMPACT_CODEBOOK}"
: "${COMPACT_CODEC_ARTIFACT:?set COMPACT_CODEC_ARTIFACT}"
: "${DIRECT_COMPACT_WARMSTART:?set DIRECT_COMPACT_WARMSTART}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"
SPARSE_TRAIN="${OUTPUT_ROOT}/qwen_mc_sequence_plus_sparse_topk_tail.jsonl"
SPARSE_SEAL="${OUTPUT_ROOT}/qwen_mc_sequence_plus_sparse_topk_tail.seal.json"
SPARSE_MANIFEST="${OUTPUT_ROOT}/qwen_mc_sequence_plus_sparse_topk_tail.manifest.json"
TRAIN_OUTPUT="${QWEN_TRAIN_OUTPUT:-${OUTPUT_ROOT}/direct_compact_qwen_sparse_topk_tail_warmstart}"

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

cd "${PATCH_ROOT}"
"${PYTHON_BIN}" -m scripts.training.build_qwen_sparse_topk_tail_auxiliary \
  --sequence-train-jsonl "${SEQUENCE_TRAIN_JSONL}" \
  --sequence-train-seal "${SEQUENCE_TRAIN_SEAL}" \
  --sequence-schedule-jsonl "${SEQUENCE_SCHEDULE_JSONL}" \
  --sequence-build-manifest "${SEQUENCE_BUILD_MANIFEST}" \
  --expected-sequence-build-manifest-sha256 "$(sha256_of "${SEQUENCE_BUILD_MANIFEST}")" \
  --teacher-parseable-jsonl "${TEACHER_PARSEABLE_JSONL}" \
  --expected-teacher-parseable-sha256 "$(sha256_of "${TEACHER_PARSEABLE_JSONL}")" \
  --teacher-audit-json "${TEACHER_AUDIT_JSON}" \
  --expected-teacher-audit-sha256 "$(sha256_of "${TEACHER_AUDIT_JSON}")" \
  --student-tokenizer-json "${STUDENT_TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "$(sha256_of "${STUDENT_TOKENIZER_JSON}")" \
  --student-eos-token-id "${STUDENT_EOS_TOKEN_ID}" \
  --contract "${COMPACT_CONTRACT}" \
  --output-jsonl "${SPARSE_TRAIN}" \
  --output-seal "${SPARSE_SEAL}" \
  --output-manifest "${SPARSE_MANIFEST}" \
  --minimum-eligible-fraction "${MINIMUM_SPARSE_ELIGIBLE_FRACTION}"

# The standard direct-compact trainer computes:
#   primary sequence target NLL + weight * coarsened top-k+tail forward KL.
# Sparse metadata never enters the prompt and never replaces EOS supervision.
TRAIN_ARGS=(
  --train_file "${SPARSE_TRAIN}"
  --train_seal "${SPARSE_SEAL}"
  --no_eval_during_training
  --output_dir "${TRAIN_OUTPUT}"
  --contract "${COMPACT_CONTRACT}"
  --codebook "${COMPACT_CODEBOOK}"
  --codec_artifact "${COMPACT_CODEC_ARTIFACT}"
  --tokenizer_json "${STUDENT_TOKENIZER_JSON}"
  --warmstart_checkpoint "${DIRECT_COMPACT_WARMSTART}"
  --sparse_topk_tail_manifest "${SPARSE_MANIFEST}"
  --sparse_topk_tail_weight "${SPARSE_AUXILIARY_WEIGHT}"
  --sparse_topk_tail_position_chunk_size "${SPARSE_POSITION_CHUNK_SIZE:-32}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --epochs "${EPOCHS:-1.0}"
  --batch_size "${BATCH_SIZE:-1}"
  --grad_accum "${GRAD_ACCUM:-16}"
  --eval_strategy no
  --seed "${SEED}"
  --sequence_distribution_nll
  --gradient_checkpointing
  --bf16
)
if [[ -n "${DECODER_MODEL:-}" ]]; then
  TRAIN_ARGS+=(--decoder_model "${DECODER_MODEL}")
fi
if [[ -n "${DECODER_REVISION:-}" ]]; then
  TRAIN_ARGS+=(--decoder_revision "${DECODER_REVISION}")
fi
"${PYTHON_BIN}" -m scripts.training.direct_compact_qwen_decompiler \
  "${TRAIN_ARGS[@]}"

printf 'QWEN_SPARSE_AUX_COMPLETE train=%s manifest=%s output=%s dense_kl=false\n' \
  "${SPARSE_TRAIN}" "${SPARSE_MANIFEST}" "${TRAIN_OUTPUT}"
