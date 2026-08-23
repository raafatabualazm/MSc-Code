#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_PASS2_HANDOFF_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PASS2_STAGE="${T5GEMMA_TYPED_PASS2_STAGE_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1}"
CHECKPOINT="${PASS2_STAGE}/checkpoint-optstep-000054"
OUTPUT_DIR="${T5GEMMA_TYPED_PASS2_EVAL_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_direct_rs_sft_pass2_eval_v1}"
TRAINING_AUDIT="${OUTPUT_DIR}/pass2_training_audit.json"
TRAIN_PROGRAM="${T5GEMMA_TYPED_PASS2_TRAIN_PROGRAM:-t5gemma2-typed-dual-to-pass2-handoff}"
EVAL_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_pass2_eval.sh"
AUDITOR="${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_pass2.py"
PYTHON_BIN="${T5GEMMA_TYPED_PASS2_HANDOFF_PYTHON:-/venv/main/bin/python}"
SUPERVISORCTL="${T5GEMMA_TYPED_PASS2_HANDOFF_SUPERVISORCTL:-supervisorctl}"
POLL_SECONDS="${T5GEMMA_TYPED_PASS2_HANDOFF_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_TYPED_PASS2_HANDOFF_STABILITY_SECONDS:-2}"

blocked() {
  echo "T5GEMMA_TYPED_PASS2_EVAL_HANDOFF_BLOCKED $*" >&2
  exit 78
}

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  && [[ "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  && [[ "${TRAIN_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || blocked "poll/stability/program configuration is invalid"
[[ -x "${PYTHON_BIN}" && -x "${EVAL_LAUNCHER}" && -s "${AUDITOR}" ]] \
  || blocked "audit/evaluation executable is absent"
printf '%s  %s\n' \
  4f84a30603adbffc00634f750c3906745695cbfd6a026c87305bf06325e3f437 "${AUDITOR}" \
  ada7f62d8d8e5d6267f220bd54733e849e52b1a8cfa21c8be2cf63a6017a57a9 "${EVAL_LAUNCHER}" \
  fa45106da5a260bae44f2df09af8ee24962b9f86f8c08334c82d35e89b7bd713 "${PROJECT}/scripts/training/t5gemma2_typed_direct_rs_sft_pass2.py" \
  | sha256sum -c - || blocked "pass-2 handoff/audit/evaluation code differs"

while true; do
  status_line="$("${SUPERVISORCTL}" status "${TRAIN_PROGRAM}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING) sleep "${POLL_SECONDS}" ;;
    EXITED) break ;;
    STOPPED) blocked "pass-2 chain was stopped rather than completed" ;;
    FATAL|BACKOFF|UNKNOWN|"") blocked "pass-2 chain state=${state:-missing}" ;;
    *) blocked "unexpected pass-2 chain state=${state}" ;;
  esac
done

files=(
  "${PASS2_STAGE}/result.json"
  "${PASS2_STAGE}/run_contract.json"
  "${PASS2_STAGE}/dataset_manifest.json"
  "${PASS2_STAGE}/latest_checkpoint.json"
  "${CHECKPOINT}/run_contract.json"
  "${CHECKPOINT}/training_state.pt"
  "${CHECKPOINT}/adapter/adapter_model.safetensors"
  "${CHECKPOINT}/adapter/adapter_config.json"
  "${CHECKPOINT}/tokenizer/tokenizer.json"
)
for required in "${files[@]}"; do
  [[ -s "${required}" ]] || blocked "training exited without ${required}"
done
snapshot_one="$(sha256sum "${files[@]}")"
sleep "${STABILITY_SECONDS}"
snapshot_two="$(sha256sum "${files[@]}")"
[[ "${snapshot_one}" == "${snapshot_two}" ]] \
  || blocked "pass-2 artifacts changed after Supervisor EXITED"
unset snapshot_one snapshot_two

export T5GEMMA_TYPED_PASS2_RESULT_SHA256="$(sha256sum "${PASS2_STAGE}/result.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_ROOT_CONTRACT_SHA256="$(sha256sum "${PASS2_STAGE}/run_contract.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_DATASET_MANIFEST_SHA256="$(sha256sum "${PASS2_STAGE}/dataset_manifest.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_LATEST_POINTER_SHA256="$(sha256sum "${PASS2_STAGE}/latest_checkpoint.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_CHECKPOINT_CONTRACT_SHA256="$(sha256sum "${CHECKPOINT}/run_contract.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_TRAINING_STATE_SHA256="$(sha256sum "${CHECKPOINT}/training_state.pt" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_ADAPTER_WEIGHTS_SHA256="$(sha256sum "${CHECKPOINT}/adapter/adapter_model.safetensors" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_ADAPTER_CONFIG_SHA256="$(sha256sum "${CHECKPOINT}/adapter/adapter_config.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_TOKENIZER_SHA256="$(sha256sum "${CHECKPOINT}/tokenizer/tokenizer.json" | awk '{print $1}')"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
cd "${PROJECT}"
if ! "${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_pass2.py training \
  --result "${PASS2_STAGE}/result.json" --expected-result-sha256 "${T5GEMMA_TYPED_PASS2_RESULT_SHA256}" \
  --root-contract "${PASS2_STAGE}/run_contract.json" --expected-root-contract-sha256 "${T5GEMMA_TYPED_PASS2_ROOT_CONTRACT_SHA256}" \
  --dataset-manifest "${PASS2_STAGE}/dataset_manifest.json" --expected-dataset-manifest-sha256 "${T5GEMMA_TYPED_PASS2_DATASET_MANIFEST_SHA256}" \
  --latest-pointer "${PASS2_STAGE}/latest_checkpoint.json" --expected-latest-pointer-sha256 "${T5GEMMA_TYPED_PASS2_LATEST_POINTER_SHA256}" \
  --checkpoint-contract "${CHECKPOINT}/run_contract.json" --expected-checkpoint-contract-sha256 "${T5GEMMA_TYPED_PASS2_CHECKPOINT_CONTRACT_SHA256}" \
  --training-state "${CHECKPOINT}/training_state.pt" --expected-training-state-sha256 "${T5GEMMA_TYPED_PASS2_TRAINING_STATE_SHA256}" \
  --adapter-weights "${CHECKPOINT}/adapter/adapter_model.safetensors" --expected-adapter-weights-sha256 "${T5GEMMA_TYPED_PASS2_ADAPTER_WEIGHTS_SHA256}" \
  --adapter-config "${CHECKPOINT}/adapter/adapter_config.json" --expected-adapter-config-sha256 "${T5GEMMA_TYPED_PASS2_ADAPTER_CONFIG_SHA256}" \
  --tokenizer "${CHECKPOINT}/tokenizer/tokenizer.json" --expected-tokenizer-sha256 "${T5GEMMA_TYPED_PASS2_TOKENIZER_SHA256}" \
  --output "${TRAINING_AUDIT}"; then
  blocked "pass-2 training audit failed; evaluation was not started"
fi
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-direct-rs-sft-pass2-training-audit-v1"
  and .status == "pass"
  and .composition.rows == 209
  and .composition.local_student_new == 190
  and .composition.external_teacher_new == 19
  and .composition.prior_225_replay == 0
  and .composition.gold_replay == 0
  and .composition.heldout_overlap == 0
  and .checkpoint.name == "checkpoint-optstep-000054"
  and .checkpoint.update == 54' "${TRAINING_AUDIT}" >/dev/null \
  || blocked "pass-2 training audit output differs"
export T5GEMMA_TYPED_PASS2_TRAINING_AUDIT="${TRAINING_AUDIT}"
export T5GEMMA_TYPED_PASS2_TRAINING_AUDIT_SHA256="$(sha256sum "${TRAINING_AUDIT}" | awk '{print $1}')"
export T5GEMMA_TYPED_PASS2_STAGE_DIR="${PASS2_STAGE}"
export T5GEMMA_TYPED_PASS2_EVAL_OUTPUT_DIR="${OUTPUT_DIR}"

echo "T5GEMMA_TYPED_PASS2_EVAL_HANDOFF_SEALED audit=${TRAINING_AUDIT} checkpoint=${CHECKPOINT}"
exec "${EVAL_LAUNCHER}"
