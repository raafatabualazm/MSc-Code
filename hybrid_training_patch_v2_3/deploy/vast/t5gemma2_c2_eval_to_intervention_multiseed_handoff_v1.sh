#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="${T5GEMMA_INTERVENTION_HANDOFF_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_INTERVENTION_HANDOFF_PYTHON:-/venv/main/bin/python}"
SUPERVISORCTL="${T5GEMMA_INTERVENTION_HANDOFF_SUPERVISORCTL:-supervisorctl}"
EVAL_PROGRAM="${T5GEMMA_INTERVENTION_HANDOFF_EVAL_PROGRAM:-t5gemma2-typed-c2-verpo-matched-eval8192-v1}"
NEXT_PROGRAM="${T5GEMMA_INTERVENTION_HANDOFF_NEXT_PROGRAM:-t5gemma2-measurement-intervention-multiseed-v1}"
POLL_SECONDS="${T5GEMMA_INTERVENTION_HANDOFF_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_INTERVENTION_HANDOFF_STABILITY_SECONDS:-2}"
MIN_FREE_KIB="${T5GEMMA_INTERVENTION_HANDOFF_MIN_FREE_KIB:-5242880}"
EVAL_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_c2_verpo_matched_eval8192_v1"
REPORT="${EVAL_DIR}/typed_c2_verpo_seeds42_44_k10_8192_cluster_report.json"
AUDITOR="${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_c2_verpo_multiseed.py"
INSTALLED_EVAL_LAUNCHER="/opt/supervisor-scripts/t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh"
INSTALLED_EVAL_CONF="/etc/supervisor/conf.d/t5gemma2-typed-c2-verpo-matched-eval8192-v1.conf"
NEXT_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_measurement_intervention_multiseed_v1.sh"
NEXT_CONF="${PROJECT}/deploy/vast/t5gemma2-measurement-intervention-multiseed-v1.conf"
INSTALLED_NEXT_LAUNCHER="/opt/supervisor-scripts/t5gemma2_measurement_intervention_multiseed_v1.sh"
INSTALLED_NEXT_CONF="/etc/supervisor/conf.d/t5gemma2-measurement-intervention-multiseed-v1.conf"

blocked() {
  echo "T5GEMMA_C2_EVAL_TO_INTERVENTION_MULTISEED_BLOCKED $*" >&2
  exit 78
}

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ \
   && "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ \
   && "${MIN_FREE_KIB}" =~ ^[1-9][0-9]*$ \
   && "${EVAL_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ \
   && "${NEXT_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ \
   && "${EVAL_PROGRAM}" != "${NEXT_PROGRAM}" ]] \
  || blocked "poll/stability/storage/program configuration is invalid"
[[ -x "${PYTHON_BIN}" && -x "${NEXT_LAUNCHER}" && -s "${NEXT_CONF}" \
   && -x "${INSTALLED_EVAL_LAUNCHER}" && -s "${INSTALLED_EVAL_CONF}" \
   && -x "${INSTALLED_NEXT_LAUNCHER}" \
   && -s "${INSTALLED_NEXT_CONF}" && -s "${AUDITOR}" ]] \
  || blocked "pinned auditor or staged next Supervisor deployment is absent"
command -v "${SUPERVISORCTL}" >/dev/null 2>&1 \
  || blocked "supervisorctl is absent"

printf '%s  %s\n' \
  71eb4e512d97f19ae85a139a17188164afa670acd379d9ccfae1c7ae427b642f "${AUDITOR}" \
  a6eb165a6b641fcae21a0f1d3e64a8bba7858ea9ad890e2df05dfe7051acccaa "${PROJECT}/deploy/vast/t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh" \
  a6eb165a6b641fcae21a0f1d3e64a8bba7858ea9ad890e2df05dfe7051acccaa "${INSTALLED_EVAL_LAUNCHER}" \
  8c5205349e885a9218a55d9ca1e959f313dc38ba9514695fa8aea9f13e1bb0c5 "${INSTALLED_EVAL_CONF}" \
  fcc1bcc088c6b2e59bdfb29eb195c08af830a4c99a7652e297c4f44a9b0b0453 "${NEXT_LAUNCHER}" \
  14c5888e2646219dd1771933b0e4dc90c6cec18ca08d8879667ae73d16eb7577 "${NEXT_CONF}" \
  | sha256sum -c - || blocked "pinned handoff dependency differs"
cmp -s "${NEXT_LAUNCHER}" "${INSTALLED_NEXT_LAUNCHER}" \
  || blocked "installed next-program launcher differs"
cmp -s "${NEXT_CONF}" "${INSTALLED_NEXT_CONF}" \
  || blocked "installed next-program Supervisor configuration differs"

# EXITED is required.  STOPPED/FATAL/BACKOFF cannot be interpreted as a
# successful matched evaluation even if partial artifacts exist.
while true; do
  eval_status_line="$("${SUPERVISORCTL}" status "${EVAL_PROGRAM}" 2>/dev/null || true)"
  eval_state="$(awk '{print $2}' <<<"${eval_status_line}")"
  case "${eval_state}" in
    RUNNING|STARTING) sleep "${POLL_SECONDS}" ;;
    EXITED) break ;;
    STOPPED|FATAL|BACKOFF|UNKNOWN|"") \
      blocked "matched evaluation state=${eval_state:-missing}" ;;
    *) blocked "unexpected matched evaluation state=${eval_state}" ;;
  esac
done

[[ -s "${REPORT}" ]] \
  || blocked "matched evaluation EXITED without its final report"

# Re-run the read-only report builder against the six declared artifacts.  Its
# require-exact semantics and hash-chain checks make an existing report a seal;
# a mismatch fails rather than replacing it.
export PYTHONPATH="${PROJECT}"
artifact_args=()
for seed in 42 43 44; do
  artifact_args+=(
    --artifact "c2_baseline|${seed}|${EVAL_DIR}/c2_baseline_seed${seed}_k10_8192_predictions.json|${EVAL_DIR}/c2_baseline_seed${seed}_k10_8192_score_full175.json"
    --artifact "c2_verpo|${seed}|${EVAL_DIR}/c2_verpo_u150_seed${seed}_k10_8192_predictions.json|${EVAL_DIR}/c2_verpo_u150_seed${seed}_k10_8192_score_full175.json"
  )
done
"${PYTHON_BIN}" "${AUDITOR}" "${artifact_args[@]}" --output "${REPORT}" \
  || blocked "matched report/hash-chain revalidation failed"
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-c2-verpo-multiseed-eval-v1"
  and .status == "complete"
  and .contract.heldout_tasks == 175
  and .contract.k == 10
  and .contract.seeds == [42,43,44]
  and .contract.input_view == "typed_opaque_contract"
  and .contract.sampling.temperature == 0.8
  and .contract.sampling.top_p == 0.95
  and .contract.sampling.max_source_tokens == 32768
  and .contract.sampling.max_new_tokens == 8192
  and .checks.all_generation_and_scoring_hash_chains_validated == true
  and .checks.private_holdback_used_for_selection_or_training == false
  and (.arms.c2_baseline.seeds | keys | sort) == ["42","43","44"]
  and (.arms.c2_verpo.seeds | keys | sort) == ["42","43","44"]
  and .decision.status == "STOP_AFTER_MATCHED_EVALUATION"
  and .decision.automatic_promotion_performed == false
  and .decision.promoted_checkpoint == null
  and .decision.promotion_permitted_from_this_report == false
' "${REPORT}" >/dev/null || blocked "matched report terminal contract differs"
report_sha_one="$(sha256sum "${REPORT}" | awk '{print $1}')"
sleep "${STABILITY_SECONDS}"
report_sha_two="$(sha256sum "${REPORT}" | awk '{print $1}')"
[[ "${report_sha_one}" == "${report_sha_two}" ]] \
  || blocked "matched report changed after evaluation Supervisor EXITED"

available_kib="$(df -Pk "${WORKSPACE}" | /usr/bin/awk 'NR==2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge "${MIN_FREE_KIB}" ]] \
  || blocked "free storage ${available_kib:-unknown} KiB is below ${MIN_FREE_KIB} KiB"

# The next launcher repeats both the 5-GiB check and this GPU-empty check after
# acquiring its shared evaluation lock.  Requiring emptiness here prevents a
# handoff from even starting while another GPU workload is live.
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)"
[[ -z "${gpu_pids}" ]] \
  || blocked "GPU still has compute process(es): $(tr '\n' ',' <<<"${gpu_pids}")"

next_status_line="$("${SUPERVISORCTL}" status "${NEXT_PROGRAM}" 2>/dev/null || true)"
next_state="$(awk '{print $2}' <<<"${next_status_line}")"
case "${next_state}" in
  RUNNING|STARTING)
    echo "T5GEMMA_C2_EVAL_TO_INTERVENTION_MULTISEED_ALREADY_RUNNING program=${NEXT_PROGRAM} report_sha256=${report_sha_two}"
    exit 0
    ;;
  STOPPED|EXITED) ;;
  FATAL|BACKOFF|UNKNOWN|"") blocked "next program state=${next_state:-missing}" ;;
  *) blocked "unexpected next program state=${next_state}" ;;
esac

"${SUPERVISORCTL}" start "${NEXT_PROGRAM}" \
  || blocked "Supervisor rejected next-program start"
post_start_line="$("${SUPERVISORCTL}" status "${NEXT_PROGRAM}" 2>/dev/null || true)"
post_start_state="$(awk '{print $2}' <<<"${post_start_line}")"
case "${post_start_state}" in
  RUNNING|STARTING)
    echo "T5GEMMA_C2_EVAL_TO_INTERVENTION_MULTISEED_STARTED program=${NEXT_PROGRAM} report_sha256=${report_sha_two} free_kib=${available_kib}"
    ;;
  *) blocked "next program did not remain started state=${post_start_state:-missing}" ;;
esac
