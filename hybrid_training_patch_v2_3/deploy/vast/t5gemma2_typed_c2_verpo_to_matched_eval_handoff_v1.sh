#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="${T5GEMMA_C2_VERPO_HANDOFF_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_C2_VERPO_HANDOFF_PYTHON:-/venv/main/bin/python}"
SUPERVISORCTL="${T5GEMMA_C2_VERPO_HANDOFF_SUPERVISORCTL:-supervisorctl}"
TRAIN_PROGRAM="${T5GEMMA_C2_VERPO_HANDOFF_TRAIN_PROGRAM:-t5gemma2-typed-c2-verpo-pilot150-v1}"
EVAL_PROGRAM="${T5GEMMA_C2_VERPO_HANDOFF_EVAL_PROGRAM:-t5gemma2-typed-c2-verpo-matched-eval8192-v1}"
POLL_SECONDS="${T5GEMMA_C2_VERPO_HANDOFF_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_C2_VERPO_HANDOFF_STABILITY_SECONDS:-2}"
STAGE="${T5GEMMA_C2_VERPO_HANDOFF_STAGE:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_c2_verpo_pilot150_v1}"
VALIDATOR="${PROJECT}/scripts/evaluation/validate_t5gemma2_typed_c2_verpo_terminal.py"
EVAL_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh"
EVAL_CONF="${PROJECT}/deploy/vast/t5gemma2-typed-c2-verpo-matched-eval8192-v1.conf"
INSTALLED_EVAL_LAUNCHER="/opt/supervisor-scripts/t5gemma2_typed_c2_verpo_matched_eval8192_v1.sh"
INSTALLED_EVAL_CONF="/etc/supervisor/conf.d/t5gemma2-typed-c2-verpo-matched-eval8192-v1.conf"
REPORT="${WORKSPACE}/artifacts/t5gemma2_typed_c2_verpo_matched_eval8192_v1/typed_c2_verpo_seeds42_44_k10_8192_cluster_report.json"

blocked() {
  echo "T5GEMMA_TYPED_C2_VERPO_EVAL_HANDOFF_BLOCKED $*" >&2
  exit 78
}

no_eval() {
  echo "T5GEMMA_TYPED_C2_VERPO_EVAL_HANDOFF_NO_EVAL $*"
  exit 0
}

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  && [[ "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  && [[ "${TRAIN_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]] \
  && [[ "${EVAL_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]] \
  && [[ "${TRAIN_PROGRAM}" != "${EVAL_PROGRAM}" ]] \
  || blocked "poll/stability/program configuration is invalid"
[[ -x "${PYTHON_BIN}" && -x "${EVAL_LAUNCHER}" && -s "${VALIDATOR}" \
   && -s "${EVAL_CONF}" && -s "${INSTALLED_EVAL_LAUNCHER}" \
   && -s "${INSTALLED_EVAL_CONF}" ]] \
  || blocked "validator or matched-evaluation Supervisor deployment is absent"
command -v "${SUPERVISORCTL}" >/dev/null 2>&1 \
  || blocked "supervisorctl is absent"
command -v /usr/bin/jq >/dev/null 2>&1 || blocked "jq is absent"

printf '%s  %s\n' \
  aedeeec5a57d540c13cd9e05501166854c0d214d51c36db691b3c765b03f3766 "${VALIDATOR}" \
  0a88076cda0f6c981e7d07e402b5917966551f7b17f1efb05cc9eb833368fe31 "${PROJECT}/scripts/training/t5gemma2_typed_c2_verpo_pilot150.py" \
  a6eb165a6b641fcae21a0f1d3e64a8bba7858ea9ad890e2df05dfe7051acccaa "${EVAL_LAUNCHER}" \
  8c5205349e885a9218a55d9ca1e959f313dc38ba9514695fa8aea9f13e1bb0c5 "${EVAL_CONF}" \
  | sha256sum -c - || blocked "pinned handoff/evaluation code differs"
cmp -s "${EVAL_LAUNCHER}" "${INSTALLED_EVAL_LAUNCHER}" \
  || blocked "installed matched-evaluation launcher differs"
cmp -s "${EVAL_CONF}" "${INSTALLED_EVAL_CONF}" \
  || blocked "installed matched-evaluation Supervisor configuration differs"

while true; do
  status_line="$("${SUPERVISORCTL}" status "${TRAIN_PROGRAM}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING) sleep "${POLL_SECONDS}" ;;
    EXITED) break ;;
    STOPPED) no_eval "training_supervisor_state=STOPPED" ;;
    FATAL|BACKOFF|UNKNOWN|"") no_eval "training_supervisor_state=${state:-missing}" ;;
    *) no_eval "training_supervisor_state=${state}" ;;
  esac
done

export PYTHONPATH="${PROJECT}"
decision_one="$("${PYTHON_BIN}" "${VALIDATOR}" --stage "${STAGE}")" \
  || blocked "terminal validator failed"
disposition="$(/usr/bin/jq -er '.disposition' <<<"${decision_one}" 2>/dev/null)" \
  || blocked "terminal validator returned malformed output"
case "${disposition}" in
  STOP_NO_EVAL|BLOCKED_NO_EVAL)
    reason="$(/usr/bin/jq -r '.reason' <<<"${decision_one}")"
    no_eval "disposition=${disposition} reason=${reason}"
    ;;
  EVALUATE) ;;
  *) blocked "terminal validator returned unknown disposition=${disposition}" ;;
esac

sleep "${STABILITY_SECONDS}"
decision_two="$("${PYTHON_BIN}" "${VALIDATOR}" --stage "${STAGE}")" \
  || blocked "terminal stability validator failed"
[[ "${decision_one}" == "${decision_two}" ]] \
  || no_eval "disposition=BLOCKED_NO_EVAL reason=terminal_evidence_changed_after_EXITED"
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-c2-verpo-eval-handoff-decision-v1"
  and .status == "complete"
  and .disposition == "EVALUATE"
  and .terminal_update == 150
  and .evaluation_permitted == true
  and .automatic_promotion_performed == false
  and .private_holdback_read == false
  and (.evidence_bundle_sha256 | test("^[0-9a-f]{64}$"))
' <<<"${decision_two}" >/dev/null \
  || blocked "exact update-150 terminal decision differs"

eval_state_line="$("${SUPERVISORCTL}" status "${EVAL_PROGRAM}" 2>/dev/null || true)"
eval_state="$(awk '{print $2}' <<<"${eval_state_line}")"
case "${eval_state}" in
  RUNNING|STARTING)
    echo "T5GEMMA_TYPED_C2_VERPO_EVAL_HANDOFF_ALREADY_RUNNING program=${EVAL_PROGRAM}"
    exit 0
    ;;
  EXITED)
    if [[ -s "${REPORT}" ]] && /usr/bin/jq -e '
      .schema == "t5gemma2-typed-c2-verpo-multiseed-eval-v1"
      and .status == "complete"
      and .decision.status == "STOP_AFTER_MATCHED_EVALUATION"
      and .decision.automatic_promotion_performed == false
    ' "${REPORT}" >/dev/null 2>&1; then
      echo "T5GEMMA_TYPED_C2_VERPO_EVAL_HANDOFF_ALREADY_COMPLETE report=${REPORT}"
      exit 0
    fi
    ;;
  STOPPED) ;;
  FATAL|BACKOFF|UNKNOWN|"") blocked "evaluation program state=${eval_state:-missing}" ;;
  *) blocked "unexpected evaluation program state=${eval_state}" ;;
esac

if ! "${SUPERVISORCTL}" start "${EVAL_PROGRAM}"; then
  blocked "Supervisor rejected matched evaluation start"
fi
post_start_line="$("${SUPERVISORCTL}" status "${EVAL_PROGRAM}" 2>/dev/null || true)"
post_start_state="$(awk '{print $2}' <<<"${post_start_line}")"
case "${post_start_state}" in
  RUNNING|STARTING)
    echo "T5GEMMA_TYPED_C2_VERPO_EVAL_HANDOFF_STARTED program=${EVAL_PROGRAM} update=150 gates=16..144"
    ;;
  *) blocked "matched evaluation did not remain started state=${post_start_state:-missing}" ;;
esac
