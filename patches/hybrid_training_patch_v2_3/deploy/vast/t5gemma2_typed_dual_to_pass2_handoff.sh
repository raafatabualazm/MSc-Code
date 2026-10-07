#!/usr/bin/env bash
set -euo pipefail

# Wait for the durable Kimi -> Sonnet controller to finish successfully, then
# hand the sealed artifacts to the independently auditing RS-SFT pass-2 job.
# This wrapper never reads provider or Hugging Face credentials itself.

WORKSPACE="${T5GEMMA_TYPED_HANDOFF_WORKSPACE:-/workspace}"
DUAL_DIR="${T5GEMMA_TYPED_DUAL_API_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_dual_api_rescue_v1}"
DUAL_REPORT="${DUAL_DIR}/orchestration_report.json"
DUAL_MANIFEST="${DUAL_DIR}/direct_manifest.json"
DUAL_TARGETS="${DUAL_DIR}/direct_targets.jsonl"
PASS2_LAUNCHER="/opt/supervisor-scripts/t5gemma2_typed_direct_rs_sft_pass2.sh"
DUAL_PROGRAM="${T5GEMMA_TYPED_DUAL_PROGRAM:-t5gemma2-typed-local-api-handoff}"
POLL_SECONDS="${T5GEMMA_TYPED_PASS2_HANDOFF_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_TYPED_PASS2_HANDOFF_STABILITY_SECONDS:-2}"
SUPERVISORCTL="${T5GEMMA_TYPED_PASS2_HANDOFF_SUPERVISORCTL:-supervisorctl}"

blocked() {
  echo "T5GEMMA_TYPED_DUAL_TO_PASS2_BLOCKED $*" >&2
  exit 78
}

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${DUAL_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  blocked "poll/stability/program configuration is invalid"
fi
[[ -x "${PASS2_LAUNCHER}" ]] || blocked "pass-2 launcher is absent"
printf '%s  %s\n' \
  b47b1330c08a3581b82cbc05ba98cd8048b02e5661b4ea57ac2126293ab73d43 "${PASS2_LAUNCHER}" \
  | sha256sum -c - || blocked "pass-2 launcher differs"

while true; do
  status_line="$("${SUPERVISORCTL}" status "${DUAL_PROGRAM}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING)
      sleep "${POLL_SECONDS}"
      ;;
    EXITED)
      break
      ;;
    STOPPED)
      blocked "dual API controller was stopped rather than completed"
      ;;
    FATAL|BACKOFF|UNKNOWN|"")
      blocked "dual API controller state=${state:-missing}"
      ;;
    *)
      blocked "unexpected dual API controller state=${state}"
      ;;
  esac
done

for required in "${DUAL_REPORT}" "${DUAL_MANIFEST}" "${DUAL_TARGETS}"; do
  [[ -f "${required}" ]] || blocked "controller exited without ${required}"
done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-dual-api-orchestration-report-v1"
  and .status == "complete"
  and .heldout_175_model_visible == false
  and .heldout_175_used_for_generation_or_selection == false
' "${DUAL_REPORT}" >/dev/null || blocked "dual API report is not complete"

sealed_files=("${DUAL_REPORT}" "${DUAL_MANIFEST}" "${DUAL_TARGETS}")
snapshot_one="$(sha256sum "${sealed_files[@]}")"
sleep "${STABILITY_SECONDS}"
snapshot_two="$(sha256sum "${sealed_files[@]}")"
[[ "${snapshot_one}" == "${snapshot_two}" ]] \
  || blocked "dual API artifacts changed after controller EXITED"
unset snapshot_one snapshot_two

echo "T5GEMMA_TYPED_DUAL_TO_PASS2_STARTING report=${DUAL_REPORT}"
exec "${PASS2_LAUNCHER}"
