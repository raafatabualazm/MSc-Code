#!/usr/bin/env bash
# Poll and harvest the two Sonnet screen batches. This deliberately never
# submits another batch, so length-cap repairs cannot consume the Opus reserve.
set -euo pipefail

PATCH_DIR="/workspace/frontier_ceiling_patch_v1"
RUN_ROOT="/workspace/artifacts/frontier_ceiling_two_enrichments/runs/anthropic_sonnet5_batch_screen_k2_warm_v1"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-1500}"

set -a
source /workspace/Anthropic.env
set +a

for ((poll=1; poll<=MAX_POLLS; poll++)); do
  complete=0
  for arm in opus codex; do
    arm_root="${RUN_ROOT}/${arm}"
    if [[ -s "${arm_root}/primary_8192_summary.json" ]]; then
      complete=$((complete + 1))
      continue
    fi

    if grep -q '"event_type":"batch_harvested"\|"event_type": "batch_harvested"' \
      "${arm_root}/batch_events.jsonl" 2>/dev/null; then
      printf '%s\n' \
        "Primary result is incomplete after harvest; manual same-cap transport reconciliation is required." \
        > "${arm_root}/PRIMARY_INCOMPLETE_REQUIRES_RECONCILIATION.txt"
      exit 3
    fi

    status_output="$(
      ACTION=status ARM="${arm}" \
        bash "${PATCH_DIR}/run_anthropic_sonnet5_batch_screen.sh"
    )"
    printf '%s\n' "${status_output}"
    if grep -q '"status": "ended"' <<<"${status_output}"; then
      ACTION=harvest ARM="${arm}" \
        bash "${PATCH_DIR}/run_anthropic_sonnet5_batch_screen.sh"
      if [[ -s "${arm_root}/primary_8192_summary.json" ]]; then
        complete=$((complete + 1))
      fi
    fi
  done

  if [[ "${complete}" -eq 2 ]]; then
    printf '%s\n' "SONNET5_PRIMARY_BOTH_ARMS_COMPLETE"
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done

printf '%s\n' "Timed out before both primary Sonnet summaries completed." >&2
exit 4
