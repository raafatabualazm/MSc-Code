#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
OUTPUT_DIR="${T5GEMMA_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_v1}"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
MODEL_REVISION=487d4acf21a4d70c70bf534265b5263c9424979e
TRAIN_JSONL="${WORKSPACE}/multifunction_v1/expanded2776/build/train_multifunction_binary_expanded_2776.jsonl"
F2_JSONL="${WORKSPACE}/multifunction_v1/expanded2776/build/train_multifunction_binary_expanded_2776_f2.jsonl"
SFT_EPOCHS="${T5GEMMA_SFT_EPOCHS:-1}"
SFT_LEARNING_RATE="${T5GEMMA_SFT_LEARNING_RATE:-2e-4}"

if ! [[ "${SFT_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "T5GEMMA_SFT_BLOCKED invalid epoch count: ${SFT_EPOCHS}" >&2
  exit 78
fi
if ! /venv/main/bin/python -c \
  'import math,sys; value=float(sys.argv[1]); raise SystemExit(not math.isfinite(value) or value <= 0)' \
  "${SFT_LEARNING_RATE}"; then
  echo "T5GEMMA_SFT_BLOCKED invalid learning rate: ${SFT_LEARNING_RATE}" >&2
  exit 78
fi

if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_SFT_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi

secret_line="$(
  /venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_bytes()
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("utf-16")
lines = [
    line.strip()
    for line in text.splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit("secret must contain exactly one non-comment token line")
print(lines[0], end="")
PY
)"
if [[ "${secret_line}" =~ ^[[:space:]]*(export[[:space:]]+)?HF_TOKEN[[:space:]]*=(.*)$ ]]; then
  HF_TOKEN="${BASH_REMATCH[2]}"
else
  HF_TOKEN="${secret_line}"
fi
HF_TOKEN="${HF_TOKEN#"${HF_TOKEN%%[![:space:]]*}"}"
HF_TOKEN="${HF_TOKEN%"${HF_TOKEN##*[![:space:]]}"}"
if [[ "${HF_TOKEN}" == \"*\" && "${HF_TOKEN}" == *\" ]]; then
  HF_TOKEN="${HF_TOKEN:1:-1}"
elif [[ "${HF_TOKEN}" == \'*\' && "${HF_TOKEN}" == *\' ]]; then
  HF_TOKEN="${HF_TOKEN:1:-1}"
fi
export HF_TOKEN
unset secret_line
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "T5GEMMA_SFT_BLOCKED HF_TOKEN is absent from ${SECRET_FILE}" >&2
  exit 78
fi

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  "${TRAIN_JSONL}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  "${F2_JSONL}" | sha256sum -c -

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
if [[ -f "${OUTPUT_DIR}/result.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/result.json")" == complete ]]; then
  echo "T5GEMMA_SFT_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

resume_args=()
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  resume_checkpoint=$(/usr/bin/jq -r '.path // empty' "${OUTPUT_DIR}/latest_checkpoint.json")
  if [[ -n "${resume_checkpoint}" && -d "${resume_checkpoint}" ]]; then
    resume_args=(--resume_checkpoint "${resume_checkpoint}")
    echo "T5GEMMA_SFT_RESUME checkpoint=${resume_checkpoint}"
  else
    echo "T5GEMMA_SFT_BLOCKED invalid latest checkpoint pointer" >&2
    exit 78
  fi
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_enriched_sft.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --f2_jsonl "${F2_JSONL}" \
  --expected_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_rows 2776 \
  --output_dir "${OUTPUT_DIR}" \
  --model google/t5gemma-2-4b-4b \
  --model_revision "${MODEL_REVISION}" \
  --max_source_tokens 32768 \
  --max_target_tokens 32768 \
  --epochs "${SFT_EPOCHS}" \
  --batch_size 1 \
  --gradient_accumulation 16 \
  --learning_rate "${SFT_LEARNING_RATE}" \
  --warmup_ratio 0.03 \
  --checkpoint_interval 5 \
  --keep_last_checkpoints 2 \
  --resume_from_trainer_sha256 72e54c0e134c44954040f2c6b348f69257994397b42b3f93b4837ad9a0350de8 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --attn_implementation sdpa \
  --bf16 \
  --gradient_checkpointing \
  "${resume_args[@]}"
