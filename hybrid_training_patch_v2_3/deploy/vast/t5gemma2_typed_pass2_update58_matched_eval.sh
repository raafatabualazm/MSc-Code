#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_PASS2_UPDATE58_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
PASS2_STAGE="${T5GEMMA_TYPED_PASS2_STAGE_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1}"
PASS2_CHECKPOINT="${PASS2_STAGE}/checkpoint-optstep-000054"
PASS2_EVAL="${T5GEMMA_TYPED_PASS2_EVAL_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_direct_rs_sft_pass2_eval_v1}"
UPDATE_STAGE="${T5GEMMA_TYPED_UPDATE58_STAGE_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1}"
UPDATE_CHECKPOINT="${UPDATE_STAGE}/checkpoint-optstep-000058"
OUTPUT_DIR="${T5GEMMA_TYPED_PASS2_UPDATE58_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_pass2_update58_current_stack_matched_v1}"
PASS2_ARM_AUDIT="${T5GEMMA_TYPED_PASS2_CURRENT_ARM_AUDIT:-${OUTPUT_DIR}/pass2_current_stack_arm_audit.json}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

AUDITOR="${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_pass2_update58_rerun.py"
WRAPPER="${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py"
BASE_INFERENCE="${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py"
SCORER="${PROJECT}/scripts/evaluation/score_direct_compact_passk.py"
DERIVE_CLEAN="${PROJECT}/scripts/evaluation/derive_passk_exclusion_sensitivity.py"
EVALUATOR="${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py"
EVALUATION="${DATA_DIR}/dev_multifunction_binary.jsonl"

PASS2_PREDICTIONS="${PASS2_EVAL}/typed_direct_pass2_seed42_k10_predictions.json"
PASS2_FULL="${PASS2_EVAL}/typed_direct_pass2_seed42_k10_score_full175.json"
PASS2_CLEAN="${PASS2_EVAL}/typed_direct_pass2_seed42_k10_score_clean174.json"
UPDATE_PREDICTIONS="${OUTPUT_DIR}/update58_current_stack_seed42_k10_predictions.json"
UPDATE_FULL="${OUTPUT_DIR}/update58_current_stack_seed42_k10_score_full175.json"
UPDATE_CLEAN="${OUTPUT_DIR}/update58_current_stack_seed42_k10_score_clean174.json"
MATCHED_AUDIT="${OUTPUT_DIR}/update58_vs_pass2_current_stack_matched_audit.json"

blocked() {
  echo "T5GEMMA_TYPED_PASS2_UPDATE58_MATCHED_EVAL_BLOCKED $*" >&2
  exit 78
}

[[ "${T5GEMMA_TYPED_PASS2_CURRENT_ARM_AUDIT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
  || blocked "pass-2 arm audit SHA is not late-bound"
[[ -x "${DART_BIN}" ]] || blocked "Dart 3.12.2 is absent"

printf '%s  %s\n' \
  4206a7d7b382a1035a2e8ba4ec4189c4a8ae2bc921a59eda96220a8daae15f48 "${AUDITOR}" \
  4a6e62900a0ebd6ed1123bfb85eeac2f7404023cc56c36ba5ded52e7354b6311 "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_rs_sft.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${WRAPPER}" \
  2b3c8803307fb8c51304e52d5eb2d81112a5f5ab4f4cf3eaca54afb7eeed02d4 "${BASE_INFERENCE}" \
  2c543c54a0ee5e55b4df708e8fd088cb772e62d012ddd41550c784c20e617cf0 "${SCORER}" \
  eb418ff372b8f20a5aad3f4eb232b1a66a397c88704cdb109f591f4f2deabede "${DERIVE_CLEAN}" \
  249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 "${EVALUATOR}" \
  fa45106da5a260bae44f2df09af8ee24962b9f86f8c08334c82d35e89b7bd713 "${PROJECT}/scripts/training/t5gemma2_typed_direct_rs_sft_pass2.py" \
  | sha256sum -c - || blocked "current matched-evaluation code differs"

printf '%s  %s\n' \
  "${T5GEMMA_TYPED_PASS2_CURRENT_ARM_AUDIT_SHA256}" "${PASS2_ARM_AUDIT}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${EVALUATION}" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c - || blocked "sealed arm audit/evaluation input differs"

update_files=(
  "${UPDATE_STAGE}/result.json"
  "${UPDATE_CHECKPOINT}/run_contract.json"
  "${UPDATE_CHECKPOINT}/adapter/adapter_model.safetensors"
  "${UPDATE_CHECKPOINT}/adapter/adapter_config.json"
  "${UPDATE_CHECKPOINT}/tokenizer/tokenizer.json"
)
for required in "${update_files[@]}"; do
  [[ -s "${required}" ]] || blocked "missing update58 prerequisite ${required}"
done
printf '%s  %s\n' \
  9c5c31ec34f30cf521f27ea8ec27931685d5a28e311154ee9ceaa5cb796f66d1 "${UPDATE_STAGE}/result.json" \
  5f04ad8f4019641bb55831217035de5e744050d908aaa11a4a12b1d52cf3be90 "${UPDATE_CHECKPOINT}/run_contract.json" \
  62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec "${UPDATE_CHECKPOINT}/adapter/adapter_model.safetensors" \
  b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b "${UPDATE_CHECKPOINT}/adapter/adapter_config.json" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d "${UPDATE_CHECKPOINT}/tokenizer/tokenizer.json" \
  | sha256sum -c - || blocked "sealed update58 checkpoint artifacts differ"
update_snapshot_one="$(sha256sum "${update_files[@]}")"
sleep 2
update_snapshot_two="$(sha256sum "${update_files[@]}")"
[[ "${update_snapshot_one}" == "${update_snapshot_two}" ]] \
  || blocked "update58 checkpoint artifacts are not stable"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"

/venv/main/bin/python "${WRAPPER}" \
  --dataset "${EVALUATION}" \
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  --sft_checkpoint "${UPDATE_CHECKPOINT}" \
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
  --output "${UPDATE_PREDICTIONS}"

/venv/main/bin/python "${SCORER}" \
  --predictions "${UPDATE_PREDICTIONS}" \
  --evaluation_file "${EVALUATION}" \
  --output "${UPDATE_FULL}" \
  --k 10 --workers 32 --timeout 30 --stability_runs 2

/venv/main/bin/python "${DERIVE_CLEAN}" \
  --score "${UPDATE_FULL}" \
  --output "${UPDATE_CLEAN}" \
  --exclude_task_id sigless_8bf7f40ca356

[[ "$(sha256sum "${update_files[@]}")" == "${update_snapshot_one}" ]] \
  || blocked "update58 checkpoint changed during evaluation"
printf '%s  %s\n' \
  "${T5GEMMA_TYPED_PASS2_CURRENT_ARM_AUDIT_SHA256}" "${PASS2_ARM_AUDIT}" \
  | sha256sum -c - || blocked "pass-2 arm audit changed during update58 rerun"

if ! /venv/main/bin/python "${AUDITOR}" compare \
  --before-predictions "${UPDATE_PREDICTIONS}" \
  --before-full-score "${UPDATE_FULL}" \
  --before-clean-score "${UPDATE_CLEAN}" \
  --before-checkpoint-contract "${UPDATE_CHECKPOINT}/run_contract.json" \
  --before-training-result "${UPDATE_STAGE}/result.json" \
  --after-predictions "${PASS2_PREDICTIONS}" \
  --after-full-score "${PASS2_FULL}" \
  --after-clean-score "${PASS2_CLEAN}" \
  --after-checkpoint-contract "${PASS2_CHECKPOINT}/run_contract.json" \
  --after-training-result "${PASS2_STAGE}/result.json" \
  --pass2-arm-audit "${PASS2_ARM_AUDIT}" \
  --expected-pass2-arm-audit-sha256 "${T5GEMMA_TYPED_PASS2_CURRENT_ARM_AUDIT_SHA256}" \
  --evaluation-file "${EVALUATION}" \
  --evaluator-file "${EVALUATOR}" \
  --wrapper-file "${WRAPPER}" \
  --base-inference-file "${BASE_INFERENCE}" \
  --output "${MATCHED_AUDIT}"; then
  blocked "current-stack paired audit failed"
fi

/usr/bin/jq -e '
  .schema == "t5gemma2-typed-pass2-vs-update58-current-stack-matched-audit-v1"
  and .status == "pass"
  and .exact_pairing_validated == true
  and .historical_update58_predictions_reused == false
  and .contract.tasks == 175
  and .contract.clean_tasks == 174
  and .contract.k == 10
  and .checks.pass2_preflight_audit_validated == true
  and .checks.fresh_update58_current_stack_rerun_validated == true
  and .checks.checkpoint_lineage_validated == true
  and .checks.same_seed_coordinates_and_sampling == true
  and .checks.same_wrapper_and_base_inference_code == true
  and .checks.same_scorer_and_scoring_settings == true
  and .checks.no_source_truncation == true
  and .paired.full175.tasks == 175
  and .paired.clean174.tasks == 174' "${MATCHED_AUDIT}" >/dev/null \
  || blocked "paired audit output gate differs"

echo "T5GEMMA_TYPED_PASS2_UPDATE58_MATCHED_EVAL_COMPLETE audit=${MATCHED_AUDIT}"
