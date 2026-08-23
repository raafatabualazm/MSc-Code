#!/usr/bin/env bash
# Budget-extended context-readable F2 continuation. Earlier arms are untouched.
#
# This wrapper defaults to the base launcher's free local preflight. Paid API
# actions remain controlled by ACTION plus all original cost/submission gates.
set -euo pipefail

PATCH_DIR="/workspace/frontier_ceiling_patch_v1"

if [[ "${ANTHROPIC_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK:-}" != "1" ]]; then
  echo \
    "Set ANTHROPIC_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK=1 to attest that this controlled benchmark is authorized." \
    >&2
  exit 2
fi

export BATCH_RUNNER="${PATCH_DIR}/frontier_passk_anthropic_benign_budget100_v2.py"
export RUN_ROOT="${RUN_ROOT:-/workspace/artifacts/frontier_ceiling_two_enrichments/runs/anthropic_sonnet5_benign_f2_k2_budget100_v2}"
export SCREEN_COST_CAP_USD="${SCREEN_COST_CAP_USD:-50.0}"

exec "${PATCH_DIR}/run_anthropic_sonnet5_batch_screen.sh"
