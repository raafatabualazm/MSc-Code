#!/usr/bin/env bash
# Poll/harvest the exact two-sample primary retry batch. Never submits work.
set -euo pipefail

PATCH_DIR="/workspace/frontier_ceiling_patch_v1"
OUT="/workspace/artifacts/frontier_ceiling_two_enrichments/runs/anthropic_sonnet5_batch_screen_k2_warm_v1/opus"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-30}"
MAX_POLLS="${MAX_POLLS:-2880}"

set -a
source /workspace/Anthropic.env
set +a

for ((poll=1; poll<=MAX_POLLS; poll++)); do
  if [[ -s "${OUT}/primary_8192_summary.json" ]]; then
    echo "SONNET5_TWO_PRIMARY_RETRIES_COMPLETE"
    exit 0
  fi
  status_output="$(
    ACTION=status ARM=opus \
      bash "${PATCH_DIR}/run_anthropic_sonnet5_batch_screen.sh"
  )"
  printf '%s\n' "${status_output}"
  if grep -q '"status": "ended"' <<<"${status_output}"; then
    ACTION=harvest ARM=opus \
      bash "${PATCH_DIR}/run_anthropic_sonnet5_batch_screen.sh"
    test -s "${OUT}/primary_8192_summary.json"
    echo "SONNET5_TWO_PRIMARY_RETRIES_COMPLETE"
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done

echo "Timed out waiting for the two Sonnet primary retries." >&2
exit 4
