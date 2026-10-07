#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_PASS2_EVAL_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
PASS2_STAGE="${T5GEMMA_TYPED_PASS2_STAGE_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1}"
PASS2_CHECKPOINT="${PASS2_STAGE}/checkpoint-optstep-000054"
OUTPUT_DIR="${T5GEMMA_TYPED_PASS2_EVAL_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_direct_rs_sft_pass2_eval_v1}"
TRAINING_AUDIT="${T5GEMMA_TYPED_PASS2_TRAINING_AUDIT:-${OUTPUT_DIR}/pass2_training_audit.json}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

PREDICTIONS="${OUTPUT_DIR}/typed_direct_pass2_seed42_k10_predictions.json"
FULL_SCORE="${OUTPUT_DIR}/typed_direct_pass2_seed42_k10_score_full175.json"
CLEAN_SCORE="${OUTPUT_DIR}/typed_direct_pass2_seed42_k10_score_clean174.json"

blocked() {
  echo "T5GEMMA_TYPED_PASS2_EVAL_BLOCKED $*" >&2
  exit 78
}

pin_names=(
  T5GEMMA_TYPED_PASS2_RESULT_SHA256
  T5GEMMA_TYPED_PASS2_ROOT_CONTRACT_SHA256
  T5GEMMA_TYPED_PASS2_DATASET_MANIFEST_SHA256
  T5GEMMA_TYPED_PASS2_LATEST_POINTER_SHA256
  T5GEMMA_TYPED_PASS2_CHECKPOINT_CONTRACT_SHA256
  T5GEMMA_TYPED_PASS2_TRAINING_STATE_SHA256
  T5GEMMA_TYPED_PASS2_ADAPTER_WEIGHTS_SHA256
  T5GEMMA_TYPED_PASS2_ADAPTER_CONFIG_SHA256
  T5GEMMA_TYPED_PASS2_TOKENIZER_SHA256
  T5GEMMA_TYPED_PASS2_TRAINING_AUDIT_SHA256
)
for name in "${pin_names[@]}"; do
  [[ "${!name:-}" =~ ^[0-9a-f]{64}$ ]] || blocked "${name} is not pinned"
done
[[ -x "${DART_BIN}" ]] || blocked "Dart 3.12.2 is absent"

printf '%s  %s\n' \
  4f84a30603adbffc00634f750c3906745695cbfd6a026c87305bf06325e3f437 "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_pass2.py" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py" \
  2b3c8803307fb8c51304e52d5eb2d81112a5f5ab4f4cf3eaca54afb7eeed02d4 "${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py" \
  2c543c54a0ee5e55b4df708e8fd088cb772e62d012ddd41550c784c20e617cf0 "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  eb418ff372b8f20a5aad3f4eb232b1a66a397c88704cdb109f591f4f2deabede "${PROJECT}/scripts/evaluation/derive_passk_exclusion_sensitivity.py" \
  249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  fa45106da5a260bae44f2df09af8ee24962b9f86f8c08334c82d35e89b7bd713 "${PROJECT}/scripts/training/t5gemma2_typed_direct_rs_sft_pass2.py" \
  | sha256sum -c - || blocked "pass-2 audit/evaluation code differs"

printf '%s  %s\n' \
  "${T5GEMMA_TYPED_PASS2_RESULT_SHA256}" "${PASS2_STAGE}/result.json" \
  "${T5GEMMA_TYPED_PASS2_ROOT_CONTRACT_SHA256}" "${PASS2_STAGE}/run_contract.json" \
  "${T5GEMMA_TYPED_PASS2_DATASET_MANIFEST_SHA256}" "${PASS2_STAGE}/dataset_manifest.json" \
  "${T5GEMMA_TYPED_PASS2_LATEST_POINTER_SHA256}" "${PASS2_STAGE}/latest_checkpoint.json" \
  "${T5GEMMA_TYPED_PASS2_CHECKPOINT_CONTRACT_SHA256}" "${PASS2_CHECKPOINT}/run_contract.json" \
  "${T5GEMMA_TYPED_PASS2_TRAINING_STATE_SHA256}" "${PASS2_CHECKPOINT}/training_state.pt" \
  "${T5GEMMA_TYPED_PASS2_ADAPTER_WEIGHTS_SHA256}" "${PASS2_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${T5GEMMA_TYPED_PASS2_ADAPTER_CONFIG_SHA256}" "${PASS2_CHECKPOINT}/adapter/adapter_config.json" \
  "${T5GEMMA_TYPED_PASS2_TOKENIZER_SHA256}" "${PASS2_CHECKPOINT}/tokenizer/tokenizer.json" \
  "${T5GEMMA_TYPED_PASS2_TRAINING_AUDIT_SHA256}" "${TRAINING_AUDIT}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c - || blocked "sealed training/evaluation input differs"

/usr/bin/jq -e '
  .schema == "t5gemma2-typed-direct-rs-sft-pass2-training-audit-v1"
  and .status == "pass"
  and .contract.rows == 209
  and .contract.local_rows == 190
  and .contract.api_rows == 19
  and .contract.planned_updates == 54
  and .result.rows == 209
  and .result.updates == 54
  and .result.latest_checkpoint == "checkpoint-optstep-000054"
  and .composition.prior_225_replay == 0
  and .composition.gold_replay == 0
  and .composition.heldout_overlap == 0' "${TRAINING_AUDIT}" >/dev/null \
  || blocked "pass-2 training audit gate differs"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"

/venv/main/bin/python scripts/evaluation/t5gemma2_measurement_audit_inference.py \
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  --sft_checkpoint "${PASS2_CHECKPOINT}" \
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
  --output "${FULL_SCORE}" \
  --k 10 --workers 32 --timeout 30 --stability_runs 2

/venv/main/bin/python scripts/evaluation/derive_passk_exclusion_sensitivity.py \
  --score "${FULL_SCORE}" \
  --output "${CLEAN_SCORE}" \
  --exclude_task_id sigless_8bf7f40ca356

echo "T5GEMMA_TYPED_PASS2_EVAL_COMPLETE full=${FULL_SCORE} clean=${CLEAN_SCORE}"
