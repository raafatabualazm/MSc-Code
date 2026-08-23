#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_contract_sft_2epoch_v1/checkpoint-optstep-000348"
OUTPUT_DIR="${T5GEMMA_TYPED_EVAL_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_contract_sft_2epoch_eval_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
PREDICTIONS="${OUTPUT_DIR}/typed_contract_seed42_k10_predictions.json"
SCORE="${OUTPUT_DIR}/typed_contract_seed42_k10_score_full175.json"
CLEAN_SCORE="${OUTPUT_DIR}/typed_contract_seed42_k10_score_clean174.json"

for required in \
  "${CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${CHECKPOINT}/tokenizer/tokenizer.json" \
  "${CHECKPOINT}/run_contract.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "T5GEMMA_TYPED_EVAL_BLOCKED missing ${required}" >&2
    exit 78
  fi
done
if ! /usr/bin/jq -e '
  .schema == "t5gemma2-typed-opaque-contract-sft-run-v1"
  and .architecture == "native_encoder_decoder"
  and .optimization.epochs == 2
  and .optimization.planned_updates == 348
  and .optimization.seed == 42
  and .dataset.rows == 2775
  and .dataset.input_rows == 2776
  and .dataset.training_exclusions.count == 1
  and .dataset.training_exclusions.task_ids == ["sigless_6b1dd0c6b6fc"]
  and .dataset.heldout.exact_acceptance_test_overlap == 0' \
  "${CHECKPOINT}/run_contract.json" >/dev/null; then
  echo "T5GEMMA_TYPED_EVAL_BLOCKED checkpoint contract differs" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_TYPED_EVAL_BLOCKED pinned Dart binary is absent" >&2
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

/venv/main/bin/python scripts/evaluation/t5gemma2_measurement_audit_inference.py \
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  --sft_checkpoint "${CHECKPOINT}" \
  --arm sft \
  --input_view typed_opaque_contract \
  --num_samples 10 \
  --generation_batch_size 10 \
  --max_source_tokens 32768 \
  --max_new_tokens 4096 \
  --temperature 0.8 \
  --top_p 0.95 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16 \
  --output "${PREDICTIONS}"

/venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
  --predictions "${PREDICTIONS}" \
  --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --output "${SCORE}" \
  --k 10 \
  --workers 32 \
  --timeout 30 \
  --stability_runs 2

/venv/main/bin/python scripts/evaluation/derive_passk_exclusion_sensitivity.py \
  --score "${SCORE}" \
  --output "${CLEAN_SCORE}" \
  --exclude_task_id sigless_8bf7f40ca356

echo "T5GEMMA_TYPED_EVAL_COMPLETE full=${SCORE} clean=${CLEAN_SCORE}"
