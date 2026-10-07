#!/usr/bin/env bash
# Separate Opus 5 O1 contextual arm; the original gated arm is untouched.
set -euo pipefail

PATCH_DIR="/workspace/frontier_ceiling_patch_v1"

if [[ "${ANTHROPIC_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK:-}" != "1" ]]; then
  echo \
    "Set ANTHROPIC_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK=1 to attest that this controlled benchmark is authorized." \
    >&2
  exit 2
fi

export OPUS_RUNNER="${PATCH_DIR}/anthropic_opus5_o1_benign_batch.py"
export RUN_ROOT="${RUN_ROOT:-/workspace/artifacts/frontier_ceiling_two_enrichments/runs/anthropic_opus5_o1_benign_f2_k1_v1}"

exec "${PATCH_DIR}/run_anthropic_opus5_o1_gate.sh"
