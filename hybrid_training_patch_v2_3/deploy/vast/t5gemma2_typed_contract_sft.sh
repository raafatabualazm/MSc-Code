#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
OUTPUT_DIR="${T5GEMMA_TYPED_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_contract_sft_2epoch_v1}"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
MODEL_REVISION=487d4acf21a4d70c70bf534265b5263c9424979e
TRAIN_JSONL="${WORKSPACE}/multifunction_v1/expanded2776/build/train_multifunction_binary_expanded_2776.jsonl"
F2_JSONL="${WORKSPACE}/multifunction_v1/expanded2776/build/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT_JSONL="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"

if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_TYPED_SFT_BLOCKED missing ${SECRET_FILE}" >&2
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
  echo "T5GEMMA_TYPED_SFT_BLOCKED HF_TOKEN is absent from ${SECRET_FILE}" >&2
  exit 78
fi

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  "${TRAIN_JSONL}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  "${F2_JSONL}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  "${HELDOUT_JSONL}" | sha256sum -c -

mkdir -p "${WORKSPACE}/.hf_home"
if [[ -f "${OUTPUT_DIR}/result.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/result.json")" == complete ]]; then
  echo "T5GEMMA_TYPED_SFT_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

resume_args=()
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  resume_checkpoint=$(/usr/bin/jq -r '.path // empty' "${OUTPUT_DIR}/latest_checkpoint.json")
  if [[ -n "${resume_checkpoint}" && -d "${resume_checkpoint}" ]]; then
    resume_args=(--resume_checkpoint "${resume_checkpoint}")
    echo "T5GEMMA_TYPED_SFT_RESUME checkpoint=${resume_checkpoint}"
  else
    echo "T5GEMMA_TYPED_SFT_BLOCKED invalid latest checkpoint pointer" >&2
    exit 78
  fi
elif [[ -d "${OUTPUT_DIR}" ]] && find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "T5GEMMA_TYPED_SFT_BLOCKED nonempty foreign/incomplete output ${OUTPUT_DIR}" >&2
  exit 78
else
  mkdir -p "${OUTPUT_DIR}"
  echo "T5GEMMA_TYPED_SFT_FRESH_BASE output=${OUTPUT_DIR}"
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_typed_contract_sft.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --f2_jsonl "${F2_JSONL}" \
  --heldout_jsonl "${HELDOUT_JSONL}" \
  --expected_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --expected_rows 2776 \
  --expected_heldout_rows 175 \
  --exclude_train_task_id sigless_6b1dd0c6b6fc \
  --output_dir "${OUTPUT_DIR}" \
  --model google/t5gemma-2-4b-4b \
  --model_revision "${MODEL_REVISION}" \
  --max_source_tokens 32768 \
  --max_target_tokens 32768 \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation 16 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --checkpoint_interval 5 \
  --keep_last_checkpoints 2 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --attn_implementation sdpa \
  --bf16 \
  --gradient_checkpointing \
  "${resume_args[@]}"
