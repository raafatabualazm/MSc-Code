#!/usr/bin/env bash
set -euo pipefail

# CPU-only, review-gated Supervisor handoff. It may be armed while the current
# intervention multiseed run is active, but it starts the separate control
# program only after exact EXITED plus complete report recomputation.
WORKSPACE="${T5GEMMA_CONTRACT_ONLY_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
CONTROL_DIR="${WORKSPACE}/analysis_contract_only_control"
SUPERVISORCTL="${T5GEMMA_CONTRACT_ONLY_SUPERVISORCTL:-/usr/local/bin/supervisorctl}"
UPSTREAM_PROGRAM="t5gemma2-measurement-intervention-multiseed-v1"
DOWNSTREAM_PROGRAM="t5gemma2-contract-only-control-v1"
POLL_SECONDS="${T5GEMMA_CONTRACT_ONLY_HANDOFF_POLL_SECONDS:-30}"
MAX_WAIT_SECONDS="${T5GEMMA_CONTRACT_ONLY_HANDOFF_MAX_WAIT_SECONDS:-172800}"
STABILITY_SECONDS="${T5GEMMA_CONTRACT_ONLY_HANDOFF_STABILITY_SECONDS:-5}"
MIN_FREE_KIB=5242880
MIN_GPU_FREE_MIB=5120
PYTHON_BIN="${T5GEMMA_CONTRACT_ONLY_PYTHON:-/venv/main/bin/python}"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
REFERENCE_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
AUDIT_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_measurement_audit_v1"
INTERVENTION_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_intervention_multiseed_v1"
CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
CONTROL_OUTPUT="${T5GEMMA_CONTRACT_ONLY_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_contract_only_control_v1}"
CURRENT_REPORT="${INTERVENTION_DIR}/intervention_multiseed_report.json"
RUNTIME_COMPAT="${INTERVENTION_DIR}/runtime_compatibility/runtime_compatibility.json"
GOLD_SCORE="${AUDIT_DIR}/gold_roundtrip/gold_k1_score.json"
REPORTER="${PROJECT}/scripts/evaluation/t5gemma2_measurement_intervention_multiseed_report_v1.py"
CONTROL_REPORTER="${CONTROL_DIR}/contract_only_report.py"
HANDOFF_ATTESTER="${CONTROL_DIR}/handoff_attestation.py"
HANDOFF_ATTESTATION="${CONTROL_OUTPUT}/handoff_attestation.json"
PREREG="${CONTROL_DIR}/preregistration.json"
MANIFEST="${CONTROL_DIR}/bundle.sha256"

blocked() {
  echo "T5GEMMA_CONTRACT_ONLY_HANDOFF_BLOCKED $*" >&2
  exit 78
}

for value_name in POLL_SECONDS MAX_WAIT_SECONDS STABILITY_SECONDS MIN_FREE_KIB MIN_GPU_FREE_MIB; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || blocked "invalid ${value_name}: ${value}"
done
[[ "${STABILITY_SECONDS}" == "5" ]] \
  || blocked "registered report-stability window must remain 5 seconds"
(( MIN_FREE_KIB >= 5242880 && MIN_GPU_FREE_MIB >= 5120 )) \
  || blocked "registered 5-GiB resource floors cannot be lowered"
[[ -x "${SUPERVISORCTL}" && -x "${PYTHON_BIN}" ]] \
  || blocked "Supervisor client or pinned Python is absent"
for command in sha256sum jq nvidia-smi; do
  command -v "${command}" >/dev/null 2>&1 || blocked "required command absent: ${command}"
done
[[ -s "${MANIFEST}" ]] || blocked "reviewed bundle manifest is absent"
manifest_sha256="$(sha256sum "${MANIFEST}" | /usr/bin/awk '{print $1}')"
[[ "${T5GEMMA_CONTRACT_ONLY_REVIEWED_BUNDLE_SHA256:-}" == "${manifest_sha256}" ]] \
  || blocked "explicit approval for this exact bundle is absent"
(
  cd "${CONTROL_DIR}"
  sha256sum -c "${MANIFEST}"
) || blocked "reviewed bundle differs"

printf '%s  %s\n' \
  89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a "${REPORTER}" \
  10098cfac9a6475b1c54320d93b2bd989d7efc2298972d5030c7ff567e06c9db "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_report.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  | sha256sum -c - || blocked "predecessor reporter path differs"

supervisor_state() {
  local program="$1"
  local line rc
  set +e
  line="$("${SUPERVISORCTL}" status "${program}" 2>&1)"
  rc=$?
  set -e
  [[ -n "${line}" ]] || blocked "empty Supervisor response for ${program} (rc=${rc})"
  printf '%s\n' "${line}" | /usr/bin/awk '{print $2}'
}

started_at="$(date +%s)"
while true; do
  upstream_state="$(supervisor_state "${UPSTREAM_PROGRAM}")"
  case "${upstream_state}" in
    RUNNING|STARTING|STOPPING)
      now="$(date +%s)"
      (( now - started_at < MAX_WAIT_SECONDS )) \
        || blocked "timed out waiting for ${UPSTREAM_PROGRAM}"
      echo "T5GEMMA_CONTRACT_ONLY_HANDOFF_WAITING upstream_state=${upstream_state}"
      sleep "${POLL_SECONDS}"
      ;;
    EXITED)
      break
      ;;
    STOPPED|FATAL|BACKOFF|UNKNOWN)
      blocked "upstream state is ${upstream_state}; exact EXITED is required"
      ;;
    *)
      blocked "unrecognized upstream state: ${upstream_state}"
      ;;
  esac
done

# Re-run the sealed reporter over all 20 baseline/intervention arms. Its
# require-exact publication policy makes any discrepancy with CURRENT_REPORT a
# hard failure while revalidating every generation/scoring hash chain.
report_args=(
  --baseline "42|${REFERENCE_DIR}/two_epoch_k10_predictions.json|${REFERENCE_DIR}/two_epoch_k10_score.json"
)
for seed in 43 44 45 46; do
  report_args+=(
    --baseline "${seed}|${AUDIT_DIR}/baseline_seed${seed}_k10_predictions.json|${AUDIT_DIR}/baseline_seed${seed}_k10_score.json"
  )
done
for input_view in typed_opaque_contract constants_stripped semantic_body_swap; do
  report_args+=(
    --arm "${input_view}|42|${AUDIT_DIR}/${input_view}_seed42_k10_predictions.json|${AUDIT_DIR}/${input_view}_seed42_k10_score.json"
  )
  for seed in 43 44 45 46; do
    report_args+=(
      --arm "${input_view}|${seed}|${INTERVENTION_DIR}/${input_view}_seed${seed}_k10_predictions.json|${INTERVENTION_DIR}/${input_view}_seed${seed}_k10_score.json"
    )
  done
done
export PYTHONPATH="${PROJECT}"
"${PYTHON_BIN}" "${REPORTER}" \
  "${report_args[@]}" \
  --seed42_measurement_report "${AUDIT_DIR}/measurement_report.json" \
  --runtime_compatibility "${RUNTIME_COMPAT}" \
  --gold_score "${GOLD_SCORE}" --expected_tasks 175 --k 10 \
  --output "${CURRENT_REPORT}" >/dev/null \
  || blocked "predecessor multiseed report did not reproduce"

[[ -s "${CURRENT_REPORT}" ]] \
  || blocked "upstream EXITED without its sealed report"
report_sha_one="$(sha256sum "${CURRENT_REPORT}" | /usr/bin/awk '{print $1}')"
sleep "${STABILITY_SECONDS}"
report_sha_two="$(sha256sum "${CURRENT_REPORT}" | /usr/bin/awk '{print $1}')"
[[ "${report_sha_one}" == "${report_sha_two}" ]] \
  || blocked "predecessor report hash is not stable"
[[ "${report_sha_two}" == "17645716115052bb48a906a4c7231c76ec28a9d7b66dc55bc53e669e990bee63" ]] \
  || blocked "predecessor report differs from its corrected sealed hash"
/usr/bin/jq -e \
  '.schema == "t5gemma2-f2-intervention-multiseed-report-v1"
   and .status == "complete"
   and .design.seeds == [42,43,44,45,46]
   and .design.fresh_runs == 12
   and .design.tasks_per_run == 175 and .design.k == 10
   and .design.no_training_or_promotion == true
   and .design.historical_to_current_runtime_replay_gate_passed == true
   and .model_visible_source_bytes_identical_across_seeds == true
   and .full_input_view_records_identical_across_seeds == false
   and .allowed_input_view_metadata_drift.field == "row_transformations_sha256"
   and .allowed_input_view_metadata_drift.full_record_identity_not_claimed == true
   and .rank0_gold_roundtrip.passed == 175' \
  "${CURRENT_REPORT}" >/dev/null \
  || blocked "predecessor report contract differs"

available_kib="$(df -Pk "${WORKSPACE}" | /usr/bin/awk 'NR==2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge "${MIN_FREE_KIB}" ]] \
  || blocked "less than ${MIN_FREE_KIB} KiB is free"
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
[[ -z "${gpu_pids}" ]] || blocked "GPU is occupied by pid(s): $(tr '\n' ',' <<<"${gpu_pids}")"
[[ "${gpu_free_mib}" =~ ^[0-9]+$ && "${gpu_free_mib}" -ge "${MIN_GPU_FREE_MIB}" ]] \
  || blocked "GPU free memory is below ${MIN_GPU_FREE_MIB} MiB"
upstream_state="$(supervisor_state "${UPSTREAM_PROGRAM}")"
[[ "${upstream_state}" == "EXITED" ]] \
  || blocked "upstream left exact EXITED before downstream authorization"

mkdir -p "${CONTROL_OUTPUT}"
"${PYTHON_BIN}" "${HANDOFF_ATTESTER}" \
  --mode create \
  --predecessor_report "${CURRENT_REPORT}" \
  --predecessor_reporter_sha256 89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a \
  --stable_report_sha256 "${report_sha_two}" \
  --bundle_manifest "${MANIFEST}" \
  --reviewed_bundle_sha256 "${manifest_sha256}" \
  --upstream_program "${UPSTREAM_PROGRAM}" --upstream_state "${upstream_state}" \
  --stability_seconds "${STABILITY_SECONDS}" \
  --minimum_free_kib "${MIN_FREE_KIB}" --minimum_gpu_free_mib "${MIN_GPU_FREE_MIB}" \
  --output "${HANDOFF_ATTESTATION}" >/dev/null \
  || blocked "could not seal exact-EXITED handoff attestation"
/usr/bin/jq -e \
  '.schema == "t5gemma2-contract-only-handoff-attestation-v1"
   and .status == "pass"
   and .upstream_supervisor.observed_state == "EXITED"
   and .predecessor_report.recomputed_immediately_before_attestation == true
   and .predecessor_report.stable_hash_gate_passed == true
   and .resource_gates.gpu_compute_processes_empty == true
   and .downstream_start_authorized == true' \
  "${HANDOFF_ATTESTATION}" >/dev/null \
  || blocked "published handoff attestation contract failed"

downstream_state="$(supervisor_state "${DOWNSTREAM_PROGRAM}")"
case "${downstream_state}" in
  STOPPED)
    "${SUPERVISORCTL}" start "${DOWNSTREAM_PROGRAM}" \
      || blocked "could not start downstream control"
    sleep 3
    downstream_state="$(supervisor_state "${DOWNSTREAM_PROGRAM}")"
    [[ "${downstream_state}" == "RUNNING" ]] \
      || blocked "downstream did not reach RUNNING (state=${downstream_state})"
    ;;
  RUNNING|STARTING)
    # Idempotent duplicate guard: never create a second process.
    ;;
  EXITED)
    final_report="${CONTROL_OUTPUT}/contract_only_two_seed_report.json"
    if [[ -s "${final_report}" ]]; then
    # Recompute the complete control report in place. require_exact_or_write
    # rejects any stale/tampered same-schema report, while the reporter walks
    # every generation and scoring journal chain again.
    PYTHONPATH="${WORKSPACE}:${PROJECT}" "${PYTHON_BIN}" "${CONTROL_REPORTER}" \
      --baseline "42|${REFERENCE_DIR}/two_epoch_k10_predictions.json|${REFERENCE_DIR}/two_epoch_k10_score.json" \
      --baseline "43|${AUDIT_DIR}/baseline_seed43_k10_predictions.json|${AUDIT_DIR}/baseline_seed43_k10_score.json" \
      --typed "42|${AUDIT_DIR}/typed_opaque_contract_seed42_k10_predictions.json|${AUDIT_DIR}/typed_opaque_contract_seed42_k10_score.json" \
      --typed "43|${INTERVENTION_DIR}/typed_opaque_contract_seed43_k10_predictions.json|${INTERVENTION_DIR}/typed_opaque_contract_seed43_k10_score.json" \
      --control "42|${CONTROL_OUTPUT}/full/typed_contract_only_seed42_k10_predictions.json|${CONTROL_OUTPUT}/full/typed_contract_only_seed42_k10_score.json" \
      --control "43|${CONTROL_OUTPUT}/full/typed_contract_only_seed43_k10_predictions.json|${CONTROL_OUTPUT}/full/typed_contract_only_seed43_k10_score.json" \
      --evaluation "${DATA_DIR}/dev_multifunction_binary.jsonl" \
      --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
      --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
      --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
      --checkpoint "${CHECKPOINT}" --gold_score "${GOLD_SCORE}" \
      --runtime_compatibility "${RUNTIME_COMPAT}" \
        --current_multiseed_report "${CURRENT_REPORT}" \
        --preregistration "${PREREG}" \
        --handoff_attestation "${HANDOFF_ATTESTATION}" \
      --smoke_predictions "${CONTROL_OUTPUT}/smoke/typed_contract_only_seed42_first5_k10_predictions.json" \
      --smoke_score "${CONTROL_OUTPUT}/smoke/typed_contract_only_seed42_first5_k10_score.json" \
      --smoke_evaluation "${CONTROL_OUTPUT}/smoke/evaluation_first5.jsonl" \
      --smoke_gate "${CONTROL_OUTPUT}/smoke/smoke_to_full_seed42_replay_gate.json" \
      --expected_tasks 175 --k 10 --output "${final_report}" >/dev/null \
      || blocked "existing downstream report/artifact chains did not reproduce"
    /usr/bin/jq -e \
      '.schema == "t5gemma2-contract-only-two-seed-report-v1"
       and .status == "complete"
       and .design.fresh_full_seeds == [42,43]' \
      "${final_report}" >/dev/null \
      || blocked "existing downstream report differs"
    else
      # A prior interruption may leave only valid partial journals. The runner
      # owns exact resume and will validate/reuse every completed slot.
      "${SUPERVISORCTL}" start "${DOWNSTREAM_PROGRAM}" \
        || blocked "could not resume incomplete downstream control"
      sleep 3
      downstream_state="$(supervisor_state "${DOWNSTREAM_PROGRAM}")"
      [[ "${downstream_state}" == "RUNNING" ]] \
        || blocked "resumed downstream did not reach RUNNING (state=${downstream_state})"
    fi
    ;;
  STOPPING|FATAL|BACKOFF|UNKNOWN)
    blocked "downstream state is ${downstream_state}"
    ;;
  *)
    blocked "unrecognized downstream state: ${downstream_state}"
    ;;
esac

echo "T5GEMMA_CONTRACT_ONLY_HANDOFF_COMPLETE upstream_report_sha256=$(sha256sum "${CURRENT_REPORT}" | /usr/bin/awk '{print $1}') downstream_state=${downstream_state}"
