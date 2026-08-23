#!/usr/bin/env bash
# Wait for the complete Sonnet primary pair, then run the predeclared Opus O1
# gate over both full arms. No Sonnet-failure selection and no retry batches.
set -euo pipefail

PATCH_DIR="/workspace/frontier_ceiling_patch_v1"
SONNET_ROOT="/workspace/artifacts/frontier_ceiling_two_enrichments/runs/anthropic_sonnet5_batch_screen_k2_warm_v1"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_POLLS="${MAX_POLLS:-1500}"

set -a
source /workspace/Anthropic.env
set +a

for ((poll=1; poll<=MAX_POLLS; poll++)); do
  if [[ -s "${SONNET_ROOT}/opus/primary_8192_summary.json" \
    && -s "${SONNET_ROOT}/codex/primary_8192_summary.json" ]]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done

test -s "${SONNET_ROOT}/opus/primary_8192_summary.json"
test -s "${SONNET_ROOT}/codex/primary_8192_summary.json"

ACTION=preflight bash "${PATCH_DIR}/run_anthropic_opus5_o1_gate.sh"
ACTION=submit bash "${PATCH_DIR}/run_anthropic_opus5_o1_gate.sh"

for ((poll=1; poll<=MAX_POLLS; poll++)); do
  status_output="$(
    ACTION=status bash "${PATCH_DIR}/run_anthropic_opus5_o1_gate.sh"
  )"
  printf '%s\n' "${status_output}"
  ended_count="$(
    grep -c '"status": "ended"' <<<"${status_output}" || true
  )"
  if [[ "${ended_count}" -eq 2 ]]; then
    ACTION=harvest bash "${PATCH_DIR}/run_anthropic_opus5_o1_gate.sh"
    echo "ANTHROPIC_OPUS5_O1_BOTH_ARMS_COMPLETE"
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done

echo "Timed out waiting for the Opus O1 pair." >&2
exit 4
