#!/usr/bin/env bash
set -euo pipefail

# CPU-only supervisor handoff.  It may be armed while the matched VeRPO
# evaluation is running, but it starts the x86 intervention replication only
# after the sealed three-seed report exists and has the fixed STOP disposition.
WORKSPACE="${T5GEMMA_INTERVENTION_HANDOFF_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SUPERVISORCTL="${T5GEMMA_INTERVENTION_HANDOFF_SUPERVISORCTL:-/usr/local/bin/supervisorctl}"
UPSTREAM_PROGRAM="${T5GEMMA_INTERVENTION_HANDOFF_UPSTREAM_PROGRAM:-t5gemma2-typed-c2-verpo-matched-eval8192-v1}"
DOWNSTREAM_PROGRAM="${T5GEMMA_INTERVENTION_HANDOFF_DOWNSTREAM_PROGRAM:-t5gemma2-measurement-intervention-multiseed-v1}"
POLL_SECONDS="${T5GEMMA_INTERVENTION_HANDOFF_POLL_SECONDS:-30}"
MAX_WAIT_SECONDS="${T5GEMMA_INTERVENTION_HANDOFF_MAX_WAIT_SECONDS:-172800}"
STABILITY_SECONDS="${T5GEMMA_INTERVENTION_HANDOFF_STABILITY_SECONDS:-5}"
UPSTREAM_REPORT="${WORKSPACE}/artifacts/t5gemma2_typed_c2_verpo_matched_eval8192_v1/typed_c2_verpo_seeds42_44_k10_8192_cluster_report.json"
DOWNSTREAM_REPORT="${WORKSPACE}/artifacts/t5gemma2_f2_intervention_multiseed_v1/intervention_multiseed_report.json"

blocked() {
  echo "T5GEMMA_INTERVENTION_AFTER_VERPO_HANDOFF_BLOCKED $*" >&2
  exit 78
}

for value_name in POLL_SECONDS MAX_WAIT_SECONDS STABILITY_SECONDS; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || blocked "invalid ${value_name}: ${value}"
done
[[ -x "${SUPERVISORCTL}" ]] || blocked "Supervisor client is absent"

# Bind both sides of the state transition.  The downstream files must be
# deployed before this handoff is armed; the upstream files are those already
# executing on the pod.
printf '%s  %s\n' \
  a6eb165a6b641fcae21a0f1d3e64a8bba7858ea9ad890e2df05dfe7051acccaa \
    /opt/supervisor-scripts/t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh \
  8c5205349e885a9218a55d9ca1e959f313dc38ba9514695fa8aea9f13e1bb0c5 \
    /etc/supervisor/conf.d/t5gemma2-typed-c2-verpo-matched-eval8192-v1.conf \
  71eb4e512d97f19ae85a139a17188164afa670acd379d9ccfae1c7ae427b642f \
    "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_c2_verpo_multiseed.py" \
  7ae2adecc6cf684f19b9989acfc10d630d5cbf860f59a1f758e0d59a4bf9b520 \
    /opt/supervisor-scripts/t5gemma2_measurement_intervention_multiseed_v1.sh \
  14c5888e2646219dd1771933b0e4dc90c6cec18ca08d8879667ae73d16eb7577 \
    /etc/supervisor/conf.d/t5gemma2-measurement-intervention-multiseed-v1.conf \
  89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a \
    "${PROJECT}/scripts/evaluation/t5gemma2_measurement_intervention_multiseed_report_v1.py" \
  07d14fd62ffd52d361b11e1ef0eb1d816ad89b78edb7a6b62fddfcb52b5a8895 \
    "${PROJECT}/scripts/evaluation/verify_t5gemma2_measurement_runtime_compat_v1.py" \
  | sha256sum -c - || blocked "upstream/downstream deployment differs"

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
      echo "T5GEMMA_INTERVENTION_AFTER_VERPO_HANDOFF_WAITING upstream_state=${upstream_state}"
      sleep "${POLL_SECONDS}"
      ;;
    EXITED)
      break
      ;;
    STOPPED|FATAL|BACKOFF|UNKNOWN)
      blocked "upstream state is ${upstream_state}"
      ;;
    *)
      blocked "unrecognized upstream state: ${upstream_state}"
      ;;
  esac
done

[[ -s "${UPSTREAM_REPORT}" ]] \
  || blocked "upstream exited without its sealed three-seed report"
upstream_sha_one="$(sha256sum "${UPSTREAM_REPORT}")"
sleep "${STABILITY_SECONDS}"
upstream_sha_two="$(sha256sum "${UPSTREAM_REPORT}")"
[[ "${upstream_sha_one}" == "${upstream_sha_two}" ]] \
  || blocked "upstream report is not stable"
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-c2-verpo-multiseed-eval-v1"
  and .status == "complete"
  and .contract.heldout_tasks == 175
  and .contract.k == 10
  and .contract.seeds == [42,43,44]
  and .contract.input_view == "typed_opaque_contract"
  and .contract.sampling.max_new_tokens == 8192
  and .checks.all_generation_and_scoring_hash_chains_validated == true
  and .checks.private_holdback_used_for_selection_or_training == false
  and .decision.status == "STOP_AFTER_MATCHED_EVALUATION"
  and .decision.automatic_promotion_performed == false
  and .decision.promoted_checkpoint == null
  and .decision.promotion_permitted_from_this_report == false
' "${UPSTREAM_REPORT}" >/dev/null \
  || blocked "upstream report disposition differs"

downstream_state="$(supervisor_state "${DOWNSTREAM_PROGRAM}")"
case "${downstream_state}" in
  STOPPED)
    "${SUPERVISORCTL}" start "${DOWNSTREAM_PROGRAM}" \
      || blocked "could not start downstream program"
    sleep 3
    downstream_state="$(supervisor_state "${DOWNSTREAM_PROGRAM}")"
    [[ "${downstream_state}" == "RUNNING" ]] \
      || blocked "downstream did not reach RUNNING (state=${downstream_state})"
    ;;
  RUNNING|STARTING)
    # Idempotent restart of this handoff must never create a duplicate process.
    ;;
  EXITED)
    [[ -s "${DOWNSTREAM_REPORT}" ]] \
      || blocked "downstream previously exited without its final report"
    /usr/bin/jq -e '
      .schema == "t5gemma2-f2-intervention-multiseed-report-v1"
      and .status == "complete"
      and .design.fresh_runs == 12
      and .design.no_training_or_promotion == true
    ' "${DOWNSTREAM_REPORT}" >/dev/null \
      || blocked "existing downstream report differs"
    ;;
  STOPPING|FATAL|BACKOFF|UNKNOWN)
    blocked "downstream state is ${downstream_state}"
    ;;
  *)
    blocked "unrecognized downstream state: ${downstream_state}"
    ;;
esac

echo "T5GEMMA_INTERVENTION_AFTER_VERPO_HANDOFF_COMPLETE upstream_report_sha256=$(sha256sum "${UPSTREAM_REPORT}" | /usr/bin/awk '{print $1}') downstream_state=${downstream_state}"
