#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${T5GEMMA_TYPED_CONTINUATION_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_CONTINUATION_PYTHON:-/venv/main/bin/python}"
CONTROLLER="${PROJECT}/scripts/training/t5gemma2_typed_kimi_c002_resume47.py"
PHASE_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_api_rescue_c002_resume47.sh"
C001_ROOT="${T5GEMMA_TYPED_C001_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c001_v1}"
SOURCE_ROOT="${T5GEMMA_TYPED_C002_SOURCE_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c002_v2}"
OUTPUT_ROOT="${T5GEMMA_TYPED_C002_RESUME47_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_kimi_c002_resume47_v2}"
EXPECTED_C001_SHA="${T5GEMMA_TYPED_C001_CONTINUATION_REPORT_SHA256:-f137edb5a5484f0f4f8a59e54fb327cbfec754ccb2403844d59d2517d8e519d3}"
blocked() { echo "T5GEMMA_TYPED_C002_RESUME47_BLOCKED $*" >&2; exit 78; }
[[ -x "${PYTHON_BIN}" && -x "${PHASE_LAUNCHER}" ]] || blocked "runtime/phase launcher absent"
if pgrep -f '[t]5gemma2_typed_kimi_continuation_c002.py' >/dev/null 2>&1 \
  || pgrep -f '[t]5gemma2_typed_api_rescue_continuation_c002.py' >/dev/null 2>&1; then
  blocked "immutable source c002 process is still running"
fi
[[ -s "${C001_ROOT}/continuation_report.json" ]] || blocked "completed cohort-1 report absent"
if [[ -z "${EXPECTED_C001_SHA}" ]]; then
  EXPECTED_C001_SHA="$(sha256sum "${C001_ROOT}/continuation_report.json" | awk '{print $1}')"
fi
[[ "${EXPECTED_C001_SHA}" =~ ^[0-9a-f]{64}$ ]] || blocked "cohort-1 report pin malformed"
printf '%s  %s\n' \
  cf978fa222e56d43d6450bc0f1f7fb69f8a04ee55f02c1d81f1509646ee019b7 "${CONTROLLER}" \
  11094b27ddbd80f07b358019ad094239cb74181a264b24d81cf3ce3af7f60899 "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_c002_resume47.py" \
  f902c79060043b39d4fd88dc185e15c955980dd5958bee964ae23b49a4aac08b "${PHASE_LAUNCHER}" \
  15020fefa5e617029abdf62832a349a968ac23837c8e244073568ccde0b0d30e "${PROJECT}/scripts/training/t5gemma2_typed_dual_api_orchestrator.py" \
  | sha256sum -c - || blocked "resume controller code differs"
printf '%s  %s\n' "${EXPECTED_C001_SHA}" "${C001_ROOT}/continuation_report.json" \
  273e94b78074a68bb1e9dfa057d4620802bb9a787821805ae810e3e18d20ccd0 "${SOURCE_ROOT}/plan_kimi_initial_c002.json" \
  5005e6d090e7a7091b65d816abf5c387ca4f2459c49e49cbf686369580f57da4 "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl" \
  5c224d735b9476acc98d77454f241cd4390261787613d6a253b7787fa33c3d3a "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl.chain-head.json" \
  | sha256sum -c - || blocked "predecessor/source evidence differs"
export T5GEMMA_TYPED_LOCAL_REPORT_SHA256=1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1
export T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256=ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc
export T5GEMMA_TYPED_LOCAL_TARGETS_SHA256=c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4
export T5GEMMA_TYPED_225_MANIFEST_SHA256=1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3
export T5GEMMA_TYPED_API_SPLIT_MANIFEST_SHA256=d9694f084c694c6d1f3cc85ffa00b06d91bb953e97f3e5f8b8f74f5708e0afcc
export T5GEMMA_TYPED_API_VISIBLE_SHA256=0f6054d688d1fdd9b7f332cef703ae7ff0c00956f57c47944223c9883055ad23
export T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256=419917fceb8fd98849333309270277b412c877f4b2d7205976532bd532d1494b
export T5GEMMA_TYPED_API_PROJECTION_REPORT_SHA256=2f2ef4a288da49f47fdd659576a1dda67836bdd069bd9546c8b5b1e479a3426c
export T5GEMMA_TYPED_API_PROJECTION_JOURNAL_SHA256=1359c20028418a1a678c70364b6bf522338ac95e9169a8156626bd32af9b8502
export PYTHONPATH="${PROJECT}" CUDA_VISIBLE_DEVICES=""
cd "${PROJECT}"
exec nice -n 10 "${PYTHON_BIN}" scripts/training/t5gemma2_typed_kimi_c002_resume47.py \
  --phase-launcher "${PHASE_LAUNCHER}" --output-root "${OUTPUT_ROOT}" \
  --prior-continuation-report "${C001_ROOT}/continuation_report.json" --expected-prior-continuation-report-sha256 "${EXPECTED_C001_SHA}" \
  --source-plan "${SOURCE_ROOT}/plan_kimi_initial_c002.json" \
  --source-journal "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl" \
  --source-chain-head "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl.chain-head.json"
