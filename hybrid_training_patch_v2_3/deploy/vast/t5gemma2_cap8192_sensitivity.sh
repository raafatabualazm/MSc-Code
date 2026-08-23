#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
SFT_CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_v1/checkpoint-optstep-000174"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_prepost_passk_v1"
SOURCE_JOURNAL="${OUTPUT_DIR}/post_sft_k10_predictions.json.generation.journal.jsonl"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -d "${SFT_CHECKPOINT}" ]] || [[ ! -f "${SOURCE_JOURNAL}" ]] \
  || [[ ! -f "${SOURCE_JOURNAL}.chain-head.json" ]]; then
  echo "T5GEMMA_CAP_SENSITIVITY_BLOCKED checkpoint/source journal absent" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_CAP_SENSITIVITY_BLOCKED Dart 3.12.2 is not executable" >&2
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

cd "${PROJECT}"
/venv/main/bin/python scripts/evaluation/t5gemma2_cap_sensitivity.py \
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  --sft_checkpoint "${SFT_CHECKPOINT}" \
  --arm sft \
  --source_journal "${SOURCE_JOURNAL}" \
  --max_new_tokens 8192 \
  --expected_capped 23 \
  --output "${OUTPUT_DIR}/post_sft_k10_cap8192_predictions.json"

/venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
  --predictions "${OUTPUT_DIR}/post_sft_k10_cap8192_predictions.json" \
  --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --output "${OUTPUT_DIR}/post_sft_k10_cap8192_score.json" \
  --k 10 \
  --workers 32 \
  --timeout 30 \
  --stability_runs 2

echo "T5GEMMA_CAP8192_SENSITIVITY_COMPLETE output=${OUTPUT_DIR}"
