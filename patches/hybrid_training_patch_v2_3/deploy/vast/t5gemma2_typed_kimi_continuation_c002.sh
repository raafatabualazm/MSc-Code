#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_CONTINUATION_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_CONTINUATION_PYTHON:-/venv/main/bin/python}"
CONTROLLER="${PROJECT}/scripts/training/t5gemma2_typed_kimi_continuation_c002.py"
ADAPTER="${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_continuation_c002.py"
PHASE_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_api_rescue_continuation_c002.sh"
C001_ROOT="${T5GEMMA_TYPED_C001_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c001_v1}"
C001_REPORT="${C001_ROOT}/continuation_report.json"
OUTPUT_ROOT="${T5GEMMA_TYPED_KIMI_C002_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c002_v2}"
EXPECTED_C001_SHA="${T5GEMMA_TYPED_C001_CONTINUATION_REPORT_SHA256:-}"

blocked() {
  echo "T5GEMMA_TYPED_KIMI_C002_BLOCKED $*" >&2
  exit 78
}

[[ -x "${PYTHON_BIN}" ]] || blocked "Python is absent"
[[ -x "${PHASE_LAUNCHER}" ]] || blocked "phase launcher is absent"
[[ "${EXPECTED_C001_SHA}" =~ ^[0-9a-f]{64}$ ]] \
  || blocked "completed cohort-1 report pin is absent"
printf '%s  %s\n' \
  3501cb42a48fae6c4202ed58f4f7b1525812c0dca37319b2f71c6082de09ecc0 "${CONTROLLER}" \
  15020fefa5e617029abdf62832a349a968ac23837c8e244073568ccde0b0d30e "${PROJECT}/scripts/training/t5gemma2_typed_dual_api_orchestrator.py" \
  7a03af003e998497012706361f5cbf0734d8defa82c7e458aa5f87f796e01143 "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_cascade.py" \
  7a29289f6f07caca03df73b7956ffb1782a0c2ec250cc4d5793eedc73e0d910f "${ADAPTER}" \
  3e9943b6a91644662fe8983d76839cc36ce5eaa4510bf217a04deed43c6bad98 "${PHASE_LAUNCHER}" \
  "${EXPECTED_C001_SHA}" "${C001_REPORT}" \
  | sha256sum -c - || blocked "code or completed cohort-1 evidence differs"

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
exec nice -n 10 "${PYTHON_BIN}" scripts/training/t5gemma2_typed_kimi_continuation_c002.py \
  --phase-launcher "${PHASE_LAUNCHER}" \
  --output-root "${OUTPUT_ROOT}" \
  --prior-continuation-report "${C001_REPORT}" \
  --expected-prior-continuation-report-sha256 "${EXPECTED_C001_SHA}" \
  --stated-openrouter-balance 12.44
