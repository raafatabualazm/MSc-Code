#!/usr/bin/env bash
# Separate mode-conditioned Qwen reasoning hard-SFT after sequence forward-KL.
set -Eeuo pipefail

PATCH_ROOT="${PATCH_ROOT:-/workspace/hybrid_training_patch_v2_3}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
RB="${MULTIFUNCTION_BUILD:-/workspace/multifunction_v1/build}"
OUTPUT_ROOT="${QWEN_ROOT:-/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
CONTRACT="${CONTRACT:-${RB}/multifunction_inline_cfg_v2_target24k_contract.json}"
CODEBOOK="${CODEBOOK:-${RB}/multifunction_inline_cfg_v2_codebook.json}"
CODEC="${CODEC:-/workspace/scripts/data/build_multifunction_compact_v2.py}"
TRAIN_JSONL="${TRAIN_JSONL:-${RB}/train_multifunction_binary.jsonl}"
TRAIN_SEAL="${TRAIN_SEAL:-${RB}/train_multifunction_binary_target24k.seal.json}"
PROMPTS="${PROMPTS:-${RB}/train_multifunction_binary_f2.jsonl}"
JOURNAL="${QWEN_JOURNAL:-${OUTPUT_ROOT}/qwen_teacher.journal.jsonl}"
AUDIT="${QWEN_AUDIT:-${OUTPUT_ROOT}/qwen_teacher.audit.json}"
SEQUENCE_CHECKPOINT="${SEQUENCE_CHECKPOINT:-${OUTPUT_ROOT}/direct_compact_qwen_sequence_warmstart}"
SEQUENCE_BUILD_MANIFEST="${SEQUENCE_BUILD_MANIFEST:-${OUTPUT_ROOT}/qwen_mc_sequence_train.build.json}"

COT_TRAIN="${COT_TRAIN:-${OUTPUT_ROOT}/qwen_cot_sft_train.jsonl}"
COT_SEAL="${COT_SEAL:-${OUTPUT_ROOT}/qwen_cot_sft_train.seal.json}"
COT_SCHEDULE="${COT_SCHEDULE:-${OUTPUT_ROOT}/qwen_cot_sft_train.schedule.jsonl}"
COT_BUILD_MANIFEST="${COT_BUILD_MANIFEST:-${OUTPUT_ROOT}/qwen_cot_sft_train.build.json}"
COT_CHECKPOINT="${COT_CHECKPOINT:-${OUTPUT_ROOT}/direct_compact_qwen_cot_sft}"

if (( $# != 0 )); then
  printf 'The sealed Qwen CoT stage accepts no positional arguments\n' >&2
  exit 2
fi
for required in \
  "${PATCH_ROOT}/scripts/training/build_qwen_cot_sft.py" \
  "${PATCH_ROOT}/scripts/training/direct_compact_qwen_decompiler.py" \
  "${PATCH_ROOT}/scripts/evaluation/validate_direct_compact_training_stage.py" \
  "${TRAIN_JSONL}" "${TRAIN_SEAL}" "${PROMPTS}" "${JOURNAL}" "${AUDIT}" \
  "${CONTRACT}" "${CODEBOOK}" "${CODEC}" "${TOKENIZER_JSON}" \
  "${SEQUENCE_BUILD_MANIFEST}" \
  "${SEQUENCE_CHECKPOINT}/decoder_adapter/adapter_config.json" \
  "${SEQUENCE_CHECKPOINT}/source_embedding_overlay.pt" \
  "${SEQUENCE_CHECKPOINT}/compact_contract.json" \
  "${SEQUENCE_CHECKPOINT}/run_provenance.json"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Required Qwen CoT-stage input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" \
   != "f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f" ]]; then
  printf 'Qwen CoT stage requires the exact target24k compact contract\n' >&2
  exit 2
fi

mkdir -p /workspace/locks
exec 9>/workspace/locks/qwen_cot_sft.lock
if ! flock -n 9; then
  printf 'Another Qwen CoT SFT stage holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[qwen_cot_sft] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

export PYTHONPATH="${PATCH_ROOT}:/workspace"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The nested sequence checkpoint must be complete before CoT SFT can start.
"${PYTHON}" -c \
  'import sys; from scripts.training.collect_chatgpt_compact_rs import validate_qwen_student_checkpoint; validate_qwen_student_checkpoint(sys.argv[1], qwen_build_manifest=sys.argv[2]); print("QWEN_SEQUENCE_CHECKPOINT_FOR_COT_VERIFIED", flush=True)' \
  "${SEQUENCE_CHECKPOINT}" "${SEQUENCE_BUILD_MANIFEST}"

JOURNAL_SHA256="$(sha256sum "${JOURNAL}" | awk '{print $1}')"
AUDIT_SHA256="$(sha256sum "${AUDIT}" | awk '{print $1}')"
PROMPT_SHA256="$(sha256sum "${PROMPTS}" | awk '{print $1}')"
TOKENIZER_SHA256="$(sha256sum "${TOKENIZER_JSON}" | awk '{print $1}')"

# Pure deterministic materialization: no provider call, filtering, resampling,
# correctness selection, or modification of the sequence-KL corpus.
"${PYTHON}" -m scripts.training.build_qwen_cot_sft \
  --compact-train-jsonl "${TRAIN_JSONL}" \
  --compact-train-seal "${TRAIN_SEAL}" \
  --contract "${CONTRACT}" \
  --prompt-jsonl "${PROMPTS}" \
  --expected-prompt-sha256 "${PROMPT_SHA256}" \
  --teacher-journal "${JOURNAL}" \
  --expected-teacher-journal-sha256 "${JOURNAL_SHA256}" \
  --teacher-audit-json "${AUDIT}" \
  --expected-teacher-audit-sha256 "${AUDIT_SHA256}" \
  --student-tokenizer-json "${TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "${TOKENIZER_SHA256}" \
  --output-jsonl "${COT_TRAIN}" \
  --output-seal "${COT_SEAL}" \
  --schedule-output "${COT_SCHEDULE}" \
  --build-manifest "${COT_BUILD_MANIFEST}" \
  --min-nonempty-reasoning-fraction 0.90
COT_BUILD_SHA256="$(sha256sum "${COT_BUILD_MANIFEST}" | awk '{print $1}')"

validate_cot_checkpoint() {
  "${PYTHON}" -m scripts.evaluation.validate_direct_compact_training_stage \
    --checkpoint "${COT_CHECKPOINT}" \
    --contract "${CONTRACT}" \
    --train-file "${COT_TRAIN}" \
    --train-seal "${COT_SEAL}" \
    --expected-train-rows 3160 \
    --no-eval-during-training \
    --loss-mode token_mean \
    --base-warmstart "${SEQUENCE_CHECKPOINT}" \
    --stage-contract "${COT_BUILD_MANIFEST}" \
    --expected-stage-contract-sha256 "${COT_BUILD_SHA256}"
}

if [[ -e "${COT_CHECKPOINT}" ]] && validate_cot_checkpoint; then
  printf 'QWEN_COT_SFT_REUSE checkpoint=%s\n' "${COT_CHECKPOINT}"
  exit 0
fi

TRAIN_ARGS=(
  --train_file "${COT_TRAIN}"
  --train_seal "${COT_SEAL}"
  --no_eval_during_training
  --output_dir "${COT_CHECKPOINT}"
  --contract "${CONTRACT}"
  --codebook "${CODEBOOK}"
  --codec_artifact "${CODEC}"
  --tokenizer_json "${TOKENIZER_JSON}"
  --warmstart_checkpoint "${SEQUENCE_CHECKPOINT}"
  --stage_contract "${COT_BUILD_MANIFEST}"
  --expected_stage_contract_sha256 "${COT_BUILD_SHA256}"
  --learning_rate 5e-6
  --epochs 1
  --batch_size 1
  --grad_accum 16
  --eval_strategy no
  --seed 44
  --logging_steps 1
  --save_steps 25
  --gradient_checkpointing
  --bf16
)
if [[ -d "${COT_CHECKPOINT}" ]]; then
  if compgen -G "${COT_CHECKPOINT}/checkpoint-*" >/dev/null; then
    TRAIN_ARGS+=(--resume_from_checkpoint auto)
  else
    printf 'Incomplete CoT checkpoint has no resumable trainer state: %s\n' \
      "${COT_CHECKPOINT}" >&2
    exit 2
  fi
fi
"${PYTHON}" -m scripts.training.direct_compact_qwen_decompiler \
  "${TRAIN_ARGS[@]}"
validate_cot_checkpoint
printf 'QWEN_COT_SFT_COMPLETE checkpoint=%s build=%s\n' \
  "${COT_CHECKPOINT}" "${COT_BUILD_MANIFEST}"
