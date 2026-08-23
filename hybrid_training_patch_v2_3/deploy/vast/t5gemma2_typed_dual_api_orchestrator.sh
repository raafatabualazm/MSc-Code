#!/usr/bin/env bash
set -euo pipefail

# The local->API handoff supplies all late-bound artifact pins and the
# independently audited cohort-0 schedule.  This process never reads a secret;
# the child phase launcher reads exactly one provider key only after its
# credential-free plan has been sealed.
WORKSPACE="${T5GEMMA_TYPED_HANDOFF_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_API_HANDOFF_PYTHON:-/venv/main/bin/python}"
CONTROLLER="${PROJECT}/scripts/training/t5gemma2_typed_dual_api_orchestrator.py"
PHASE_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_api_rescue_cascade.sh"
OUTPUT_ROOT="${T5GEMMA_TYPED_DUAL_API_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_dual_api_rescue_v1}"
INITIAL_SCHEDULE_SHA="${T5GEMMA_TYPED_API_SCHEDULE_SHA256:-}"

blocked() {
  echo "T5GEMMA_TYPED_DUAL_API_BLOCKED $*" >&2
  exit 78
}

[[ -x "${PYTHON_BIN}" ]] || blocked "Python is absent"
[[ -x "${PHASE_LAUNCHER}" ]] || blocked "phase launcher is absent"
[[ "${INITIAL_SCHEDULE_SHA}" =~ ^[0-9a-f]{64}$ ]] \
  || blocked "independently audited Kimi cohort-0 schedule pin is absent"
printf '%s  %s\n' \
  15020fefa5e617029abdf62832a349a968ac23837c8e244073568ccde0b0d30e "${CONTROLLER}" \
  c69e845cfefcd91555171813a66492dba0b2b5c9d44bbd8efd21175f5f7f2e14 "${PHASE_LAUNCHER}" \
  | sha256sum -c - || blocked "controller/phase-launcher code differs"

export PYTHONPATH="${PROJECT}"
cd "${PROJECT}"
exec "${PYTHON_BIN}" scripts/training/t5gemma2_typed_dual_api_orchestrator.py \
  --phase-launcher "${PHASE_LAUNCHER}" \
  --output-root "${OUTPUT_ROOT}" \
  --initial-schedule-sha256 "${INITIAL_SCHEDULE_SHA}" \
  --openrouter-max-usd 12.0 \
  --anthropic-max-usd 11.5
