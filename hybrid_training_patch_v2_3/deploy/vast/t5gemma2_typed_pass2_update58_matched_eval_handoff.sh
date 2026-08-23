#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_PASS2_UPDATE58_HANDOFF_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
PASS2_STAGE="${T5GEMMA_TYPED_PASS2_STAGE_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1}"
PASS2_CHECKPOINT="${PASS2_STAGE}/checkpoint-optstep-000054"
PASS2_EVAL="${T5GEMMA_TYPED_PASS2_EVAL_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_direct_rs_sft_pass2_eval_v1}"
OUTPUT_DIR="${T5GEMMA_TYPED_PASS2_UPDATE58_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_pass2_update58_current_stack_matched_v1}"
PASS2_ARM_AUDIT="${OUTPUT_DIR}/pass2_current_stack_arm_audit.json"
PASS2_PROGRAM="${T5GEMMA_TYPED_PASS2_CURRENT_EVAL_PROGRAM:-t5gemma2-typed-pass2-eval-handoff}"
RUNNER="${PROJECT}/deploy/vast/t5gemma2_typed_pass2_update58_matched_eval.sh"
AUDITOR="${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_pass2_update58_rerun.py"
PYTHON_BIN="${T5GEMMA_TYPED_PASS2_UPDATE58_HANDOFF_PYTHON:-/venv/main/bin/python}"
SUPERVISORCTL="${T5GEMMA_TYPED_PASS2_UPDATE58_SUPERVISORCTL:-supervisorctl}"
POLL_SECONDS="${T5GEMMA_TYPED_PASS2_UPDATE58_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_TYPED_PASS2_UPDATE58_STABILITY_SECONDS:-2}"

PREDICTIONS="${PASS2_EVAL}/typed_direct_pass2_seed42_k10_predictions.json"
FULL_SCORE="${PASS2_EVAL}/typed_direct_pass2_seed42_k10_score_full175.json"
CLEAN_SCORE="${PASS2_EVAL}/typed_direct_pass2_seed42_k10_score_clean174.json"
TRAINING_AUDIT="${PASS2_EVAL}/pass2_training_audit.json"
EVALUATION="${DATA_DIR}/dev_multifunction_binary.jsonl"
EVALUATOR="${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py"
WRAPPER="${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py"
BASE_INFERENCE="${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py"

blocked() {
  echo "T5GEMMA_TYPED_PASS2_UPDATE58_HANDOFF_BLOCKED $*" >&2
  exit 78
}

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  && [[ "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  && [[ "${PASS2_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || blocked "poll/stability/program configuration is invalid"
[[ -x "${PYTHON_BIN}" && -x "${RUNNER}" && -s "${AUDITOR}" ]] \
  || blocked "post-pass2 audit/runner executable is absent"

printf '%s  %s\n' \
  4206a7d7b382a1035a2e8ba4ec4189c4a8ae2bc921a59eda96220a8daae15f48 "${AUDITOR}" \
  8d0485b497bc9a2cea4ebcba78786f89007729ecfc6032a4715bd760b933f1be "${RUNNER}" \
  4a6e62900a0ebd6ed1123bfb85eeac2f7404023cc56c36ba5ded52e7354b6311 "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_rs_sft.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${WRAPPER}" \
  2b3c8803307fb8c51304e52d5eb2d81112a5f5ab4f4cf3eaca54afb7eeed02d4 "${BASE_INFERENCE}" \
  2c543c54a0ee5e55b4df708e8fd088cb772e62d012ddd41550c784c20e617cf0 "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  eb418ff372b8f20a5aad3f4eb232b1a66a397c88704cdb109f591f4f2deabede "${PROJECT}/scripts/evaluation/derive_passk_exclusion_sensitivity.py" \
  249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 "${EVALUATOR}" \
  fa45106da5a260bae44f2df09af8ee24962b9f86f8c08334c82d35e89b7bd713 "${PROJECT}/scripts/training/t5gemma2_typed_direct_rs_sft_pass2.py" \
  | sha256sum -c - || blocked "post-pass2 matched-evaluation code differs"

while true; do
  status_line="$("${SUPERVISORCTL}" status "${PASS2_PROGRAM}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING) sleep "${POLL_SECONDS}" ;;
    EXITED) break ;;
    STOPPED) blocked "pass-2 evaluation was stopped rather than completed" ;;
    FATAL|BACKOFF|UNKNOWN|"") blocked "pass-2 evaluation state=${state:-missing}" ;;
    *) blocked "unexpected pass-2 evaluation state=${state}" ;;
  esac
done

pass2_files=(
  "${PREDICTIONS}"
  "${PREDICTIONS}.provenance.json"
  "${PREDICTIONS}.generation.journal.jsonl"
  "${PREDICTIONS}.generation.journal.jsonl.chain-head.json"
  "${FULL_SCORE}"
  "${FULL_SCORE}.evaluation.journal.jsonl"
  "${FULL_SCORE}.evaluation.journal.jsonl.chain-head.json"
  "${CLEAN_SCORE}"
  "${PASS2_CHECKPOINT}/run_contract.json"
  "${PASS2_STAGE}/result.json"
  "${TRAINING_AUDIT}"
)
for required in "${pass2_files[@]}"; do
  [[ -s "${required}" ]] || blocked "pass-2 evaluation exited without ${required}"
done
printf '%s  %s\n' \
  00d8d92dff815479bbab8357f899ef76cf416e71e1a9dd9b9b5680edaeb659a4 "${TRAINING_AUDIT}" \
  | sha256sum -c - || blocked "sealed pass-2 training audit differs"
pass2_snapshot_one="$(sha256sum "${pass2_files[@]}")"
sleep "${STABILITY_SECONDS}"
pass2_snapshot_two="$(sha256sum "${pass2_files[@]}")"
[[ "${pass2_snapshot_one}" == "${pass2_snapshot_two}" ]] \
  || blocked "pass-2 evaluation artifacts changed after Supervisor EXITED"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
cd "${PROJECT}"
training_audit_sha="$(sha256sum "${TRAINING_AUDIT}" | awk '{print $1}')"
if ! "${PYTHON_BIN}" "${AUDITOR}" pass2-arm \
  --pass2-predictions "${PREDICTIONS}" \
  --pass2-full-score "${FULL_SCORE}" \
  --pass2-clean-score "${CLEAN_SCORE}" \
  --pass2-checkpoint-contract "${PASS2_CHECKPOINT}/run_contract.json" \
  --pass2-training-result "${PASS2_STAGE}/result.json" \
  --pass2-training-audit "${TRAINING_AUDIT}" \
  --expected-pass2-training-audit-sha256 "${training_audit_sha}" \
  --evaluation-file "${EVALUATION}" \
  --evaluator-file "${EVALUATOR}" \
  --wrapper-file "${WRAPPER}" \
  --base-inference-file "${BASE_INFERENCE}" \
  --output "${PASS2_ARM_AUDIT}"; then
  blocked "pass-2 full175/clean174 completion audit failed; update58 was not loaded"
fi

/usr/bin/jq -e '
  .schema == "t5gemma2-typed-pass2-current-stack-arm-audit-v1"
  and .status == "pass"
  and .contract.tasks == 175
  and .contract.clean_tasks == 174
  and .contract.k == 10
  and .checks.pass2_training_audit_validated == true
  and .checks.pass2_checkpoint_validated == true
  and .checks.generation_hash_chain_validated == true
  and .checks.score_hash_chain_validated == true
  and .checks.full175_complete == true
  and .checks.clean174_complete == true
  and .checks.current_inference_code_validated == true
  and .checks.no_source_truncation == true' "${PASS2_ARM_AUDIT}" >/dev/null \
  || blocked "pass-2 arm audit output gate differs; update58 was not loaded"

[[ "$(sha256sum "${pass2_files[@]}")" == "${pass2_snapshot_one}" ]] \
  || blocked "pass-2 artifacts changed during preflight audit"
export T5GEMMA_TYPED_PASS2_CURRENT_ARM_AUDIT="${PASS2_ARM_AUDIT}"
export T5GEMMA_TYPED_PASS2_CURRENT_ARM_AUDIT_SHA256="$(sha256sum "${PASS2_ARM_AUDIT}" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_STAGE_DIR="${PASS2_STAGE}"
export T5GEMMA_TYPED_PASS2_EVAL_DIR="${PASS2_EVAL}"
export T5GEMMA_TYPED_PASS2_UPDATE58_OUTPUT_DIR="${OUTPUT_DIR}"

echo "T5GEMMA_TYPED_PASS2_UPDATE58_HANDOFF_SEALED pass2_arm=${PASS2_ARM_AUDIT}"
exec "${RUNNER}"
