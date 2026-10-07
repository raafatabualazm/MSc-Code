#!/usr/bin/env bash
set -euo pipefail

# One manual, bounded Kimi cohort after the sealed cohort-0 Kimi/Sonnet run.
# The Python controller plans without credentials and permits at most $10.30
# of additional OpenRouter spend.  The child phase launcher alone reads the
# OpenRouter key.
WORKSPACE="${T5GEMMA_TYPED_CONTINUATION_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_CONTINUATION_PYTHON:-/venv/main/bin/python}"
CONTROLLER="${PROJECT}/scripts/training/t5gemma2_typed_kimi_continuation.py"
DUAL_CONTROLLER="${PROJECT}/scripts/training/t5gemma2_typed_dual_api_orchestrator.py"
CASCADE="${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_cascade.py"
CASCADE_ADAPTER="${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_continuation.py"
PHASE_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_api_rescue_continuation.sh"
PRIOR_ROOT="${T5GEMMA_TYPED_PRIOR_API_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_dual_api_rescue_v1}"
OUTPUT_ROOT="${T5GEMMA_TYPED_KIMI_CONTINUATION_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c001_v1}"
PRIOR_REPORT="${PRIOR_ROOT}/orchestration_report.json"
PRIOR_INDEX="${PRIOR_ROOT}/prior_reports_final.tsv"

blocked() {
  echo "T5GEMMA_TYPED_KIMI_CONTINUATION_BLOCKED $*" >&2
  exit 78
}

[[ -x "${PYTHON_BIN}" ]] || blocked "Python is absent"
[[ -x "${PHASE_LAUNCHER}" ]] || blocked "phase launcher is absent"
printf '%s  %s\n' \
  8f0fe20ccba5f94aeeaedcb421090845cd3123c9bdb67bf648796b08ef344137 "${CONTROLLER}" \
  15020fefa5e617029abdf62832a349a968ac23837c8e244073568ccde0b0d30e "${DUAL_CONTROLLER}" \
  7a03af003e998497012706361f5cbf0734d8defa82c7e458aa5f87f796e01143 "${CASCADE}" \
  806537c03424ba7bc36cd7644eba4e3ec75b571ac8d7ed20c6b683404c35c2ac "${CASCADE_ADAPTER}" \
  5bd090871cd43a8af3595df547aa508a3a9405c06ba5d8483077f67d3081ed8d "${PHASE_LAUNCHER}" \
  9221e7cc68babbee43c9b4ae2405e1633414cd0e684da942161ed100c848fac3 "${PRIOR_REPORT}" \
  3e04594b4a5b1e4c3cd2aa1c54dfd1537d53f5140c1fb6fe77ee17b7c04cce4b "${PRIOR_INDEX}" \
  | sha256sum -c - || blocked "code or prior-run evidence differs"

# Immutable evidence pins consumed by the phase launcher.  These are hashes,
# never credentials or raw private-test content.
export T5GEMMA_TYPED_LOCAL_REPORT_SHA256=1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1
export T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256=ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc
export T5GEMMA_TYPED_LOCAL_TARGETS_SHA256=c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4
export T5GEMMA_TYPED_225_MANIFEST_SHA256=1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3
export T5GEMMA_TYPED_API_SPLIT_MANIFEST_SHA256=d9694f084c694c6d1f3cc85ffa00b06d91bb953e97f3e5f8b8f74f5708e0afcc
export T5GEMMA_TYPED_API_VISIBLE_SHA256=0f6054d688d1fdd9b7f332cef703ae7ff0c00956f57c47944223c9883055ad23
export T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256=419917fceb8fd98849333309270277b412c877f4b2d7205976532bd532d1494b
export T5GEMMA_TYPED_API_PROJECTION_REPORT_SHA256=2f2ef4a288da49f47fdd659576a1dda67836bdd069bd9546c8b5b1e479a3426c
export T5GEMMA_TYPED_API_PROJECTION_JOURNAL_SHA256=1359c20028418a1a678c70364b6bf522338ac95e9169a8156626bd32af9b8502

export PYTHONPATH="${PROJECT}"
export CUDA_VISIBLE_DEVICES=""
cd "${PROJECT}"
exec nice -n 10 "${PYTHON_BIN}" scripts/training/t5gemma2_typed_kimi_continuation.py \
  --phase-launcher "${PHASE_LAUNCHER}" \
  --output-root "${OUTPUT_ROOT}" \
  --prior-orchestration-report "${PRIOR_REPORT}" \
  --expected-prior-orchestration-report-sha256 9221e7cc68babbee43c9b4ae2405e1633414cd0e684da942161ed100c848fac3 \
  --prior-index "${PRIOR_INDEX}" \
  --expected-prior-index-sha256 3e04594b4a5b1e4c3cd2aa1c54dfd1537d53f5140c1fb6fe77ee17b7c04cce4b \
  --openrouter-balance-before-cohort0 12.44 \
  --continuation-max-usd 10.30
