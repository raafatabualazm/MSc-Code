#!/usr/bin/env bash
set -euo pipefail

# Keep the GPU hand-off automatic without mutating the currently running
# first-pass process.  This service waits for its immutable report, seals the
# two late-bound hashes, then replaces itself with the residual-only launcher.
CURRENT_SERVICE=t5gemma-base-harvest-remaining-2epoch
CURRENT_DIR=/workspace/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1
CURRENT_REPORT="${CURRENT_DIR}/harvest_report.json"
CURRENT_JOURNAL="${CURRENT_DIR}/harvest.journal.jsonl"
RESIDUAL_LAUNCHER=/opt/supervisor-scripts/t5gemma2_base_residual_harvest_2epoch.sh

while true; do
  # supervisorctl intentionally returns a non-zero status for EXITED/STOPPED;
  # capture its text without letting `set -e` abort the hand-off.
  status_line="$(supervisorctl status "${CURRENT_SERVICE}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING)
      sleep 30
      ;;
    EXITED|STOPPED)
      break
      ;;
    *)
      echo "T5GEMMA_RESIDUAL_CHAIN_BLOCKED unexpected ${CURRENT_SERVICE} state=${state:-missing}" >&2
      exit 78
      ;;
  esac
done

if [[ ! -s "${CURRENT_REPORT}" ]] \
  || [[ ! -s "${CURRENT_JOURNAL}" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-local-rs-sft-pilot-report-v1"
     and .status == "complete"
     and .pilot.tasks == 1186
     and .privacy_invariants.frontier_api_calls == false
     and .privacy_invariants.heldout_175_opened == false
     and .privacy_invariants.private_holdback_text_in_model_input == false' \
    "${CURRENT_REPORT}" >/dev/null; then
  echo "T5GEMMA_RESIDUAL_CHAIN_BLOCKED remaining harvest did not seal successfully" >&2
  exit 78
fi

export T5GEMMA_RESIDUAL_REMAINING_REPORT_SHA
T5GEMMA_RESIDUAL_REMAINING_REPORT_SHA="$(sha256sum "${CURRENT_REPORT}" | awk '{print $1}')"
export T5GEMMA_RESIDUAL_REMAINING_JOURNAL_SHA
T5GEMMA_RESIDUAL_REMAINING_JOURNAL_SHA="$(sha256sum "${CURRENT_JOURNAL}" | awk '{print $1}')"

export T5GEMMA_RESIDUAL_OPUS_REPORT=/workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue_report.json
export T5GEMMA_RESIDUAL_OPUS_JOURNAL=/workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue.journal.jsonl
export T5GEMMA_RESIDUAL_OPUS_TARGETS=/workspace/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/direct_hard_targets.jsonl
export T5GEMMA_RESIDUAL_OPUS_REPORT_SHA=f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b
export T5GEMMA_RESIDUAL_OPUS_JOURNAL_SHA=49b97de386b759955497e3f9ab7b4358ca5e74ebf3a877fb6c7f3d98e39275b6
export T5GEMMA_RESIDUAL_OPUS_TARGETS_SHA=15ef808838ed01347e646e9b4462f48ae88d4afcb467d144f6c6283576abf180

exec "${RESIDUAL_LAUNCHER}"
