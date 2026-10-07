#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
SFT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_v1"
SFT_CHECKPOINT="${SFT_DIR}/checkpoint-optstep-000174"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_prepost_passk_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -f "${SFT_DIR}/result.json" ]] \
  || [[ "$(/usr/bin/jq -r '.status // empty' "${SFT_DIR}/result.json")" != complete ]] \
  || [[ "$(/usr/bin/jq -r '.latest_checkpoint // empty' "${SFT_DIR}/result.json")" != checkpoint-optstep-000174 ]] \
  || [[ ! -d "${SFT_CHECKPOINT}" ]]; then
  echo "T5GEMMA_EVAL_BLOCKED final SFT checkpoint is absent or incomplete" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_EVAL_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi

printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a \
  "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab \
  "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 \
  "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c -

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "${DART_BIN}"):${PATH}"

common_inference=(
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl"
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json"
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl"
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json"
  --sft_checkpoint "${SFT_CHECKPOINT}"
  --num_samples 10
  --generation_batch_size 10
  --max_source_tokens 32768
  --max_new_tokens 4096
  --temperature 0.8
  --top_p 0.95
  --seed 42
  --attn_implementation sdpa
  --bf16
)

cd "${PROJECT}"
/venv/main/bin/python scripts/evaluation/t5gemma2_f2_passk_inference.py \
  "${common_inference[@]}" \
  --arm base \
  --output "${OUTPUT_DIR}/pre_base_k10_predictions.json"

/venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
  --predictions "${OUTPUT_DIR}/pre_base_k10_predictions.json" \
  --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --output "${OUTPUT_DIR}/pre_base_k10_score.json" \
  --k 10 \
  --workers 32 \
  --timeout 30 \
  --stability_runs 2

/venv/main/bin/python scripts/evaluation/t5gemma2_f2_passk_inference.py \
  "${common_inference[@]}" \
  --arm sft \
  --output "${OUTPUT_DIR}/post_sft_k10_predictions.json"

/venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
  --predictions "${OUTPUT_DIR}/post_sft_k10_predictions.json" \
  --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --output "${OUTPUT_DIR}/post_sft_k10_score.json" \
  --k 10 \
  --workers 32 \
  --timeout 30 \
  --stability_runs 2

echo "T5GEMMA_PREPOST_EVAL_COMPLETE output=${OUTPUT_DIR}"
