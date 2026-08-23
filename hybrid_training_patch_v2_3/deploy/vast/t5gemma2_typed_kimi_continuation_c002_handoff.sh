#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_CONTINUATION_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
C001_PROGRAM="t5gemma2-typed-kimi-continuation"
C001_REPORT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c001_v1/continuation_report.json"
LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_kimi_continuation_c002.sh"

blocked() {
  echo "T5GEMMA_TYPED_KIMI_C002_HANDOFF_BLOCKED $*" >&2
  exit 78
}

[[ -x "${LAUNCHER}" ]] || blocked "cohort-2 launcher is absent"
while true; do
  status="$(supervisorctl status "${C001_PROGRAM}" 2>/dev/null || true)"
  if [[ "${status}" == *" RUNNING "* || "${status}" == *" STARTING "* ]] \
    || pgrep -f '[t]5gemma2_typed_kimi_continuation.py' >/dev/null 2>&1; then
    sleep 30
    continue
  fi
  break
done
[[ -s "${C001_REPORT}" ]] || blocked "completed cohort-1 report is absent"
report_sha="$(sha256sum "${C001_REPORT}" | awk '{print $1}')"
[[ "${report_sha}" =~ ^[0-9a-f]{64}$ ]] || blocked "cohort-1 report hash failed"
export T5GEMMA_TYPED_C001_CONTINUATION_REPORT_SHA256="${report_sha}"
unset report_sha status
exec "${LAUNCHER}"
