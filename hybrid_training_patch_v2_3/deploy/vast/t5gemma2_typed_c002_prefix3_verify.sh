#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_PASS3_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_PASS3_PYTHON:-/venv/main/bin/python}"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
LOCAL_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1"
PASS1_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1"
SPLIT_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_api_visible_split_v1"
PROJECTION_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_visible_failure_projection_v1"
C001_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c001_v1"
SOURCE_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c002_v2"
RESUME_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_c002_resume47_v2"
OUTPUT_DIR="${T5GEMMA_TYPED_PREFIX3_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_c002_prefix3_verification_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

blocked() { echo "T5GEMMA_TYPED_C002_PREFIX3_VERIFY_BLOCKED $*" >&2; exit 78; }
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] || blocked "runtime is absent"
[[ "${T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
  || blocked "resume47 report SHA is not late-bound"

printf '%s  %s\n' \
  8d83d545cbdd0a6c0b187421349a4f8d8a5b28746b596f884929d7cc33141ab0 "${PROJECT}/scripts/training/t5gemma2_typed_c002_prefix3_verify.py" \
  2274a500e73b6e37f3fdc3144b6d70cb28aa5bb3ec463682a5a38df9ac7bd54f "${PROJECT}/scripts/training/t5gemma2_typed_direct_rs_sft_pass3.py" \
  11094b27ddbd80f07b358019ad094239cb74181a264b24d81cf3ce3af7f60899 "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_c002_resume47.py" \
  cf978fa222e56d43d6450bc0f1f7fb69f8a04ee55f02c1d81f1509646ee019b7 "${PROJECT}/scripts/training/t5gemma2_typed_kimi_c002_resume47.py" \
  f137edb5a5484f0f4f8a59e54fb327cbfec754ccb2403844d59d2517d8e519d3 "${C001_ROOT}/continuation_report.json" \
  273e94b78074a68bb1e9dfa057d4620802bb9a787821805ae810e3e18d20ccd0 "${SOURCE_ROOT}/plan_kimi_initial_c002.json" \
  5005e6d090e7a7091b65d816abf5c387ca4f2459c49e49cbf686369580f57da4 "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl" \
  5c224d735b9476acc98d77454f241cd4390261787613d6a253b7787fa33c3d3a "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl.chain-head.json" \
  "${T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256}" "${RESUME_ROOT}/resume_report.json" \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1 "${LOCAL_DIR}/harvest_report.json" \
  ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc "${LOCAL_DIR}/harvest.journal.jsonl" \
  c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4 "${LOCAL_DIR}/direct_targets.jsonl" \
  1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3 "${PASS1_DIR}/dataset_manifest.json" \
  d9694f084c694c6d1f3cc85ffa00b06d91bb953e97f3e5f8b8f74f5708e0afcc "${SPLIT_DIR}/split_manifest.json" \
  0f6054d688d1fdd9b7f332cef703ae7ff0c00956f57c47944223c9883055ad23 "${SPLIT_DIR}/visible_train.jsonl" \
  419917fceb8fd98849333309270277b412c877f4b2d7205976532bd532d1494b "${SPLIT_DIR}/holdback.private.jsonl" \
  2f2ef4a288da49f47fdd659576a1dda67836bdd069bd9546c8b5b1e479a3426c "${PROJECTION_DIR}/visible_projection_report.json" \
  1359c20028418a1a678c70364b6bf522338ac95e9169a8156626bd32af9b8502 "${PROJECTION_DIR}/visible_projection.journal.jsonl" \
  | sha256sum -c - || blocked "pinned code/input evidence differs"

export PYTHONPATH="${PROJECT}" CUDA_VISIBLE_DEVICES="" DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
"${PYTHON_BIN}" scripts/training/t5gemma2_typed_c002_prefix3_verify.py \
  --gold-train-jsonl "${GOLD_TRAIN}" --gold-f2-jsonl "${GOLD_F2}" \
  --expected-gold-train-sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected-gold-f2-sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --heldout-jsonl "${HELDOUT}" --expected-heldout-sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --local-harvest-report "${LOCAL_DIR}/harvest_report.json" --expected-local-harvest-report-sha256 1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1 \
  --pilot-journal "${LOCAL_DIR}/harvest.journal.jsonl" --expected-local-harvest-journal-sha256 ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc \
  --local-harvest-targets "${LOCAL_DIR}/direct_targets.jsonl" --expected-local-harvest-targets-sha256 c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4 \
  --existing-direct-manifest "${PASS1_DIR}/dataset_manifest.json" --expected-existing-direct-manifest-sha256 1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3 \
  --visible-split-manifest "${SPLIT_DIR}/split_manifest.json" --expected-visible-split-manifest-sha256 d9694f084c694c6d1f3cc85ffa00b06d91bb953e97f3e5f8b8f74f5708e0afcc \
  --visible-train "${SPLIT_DIR}/visible_train.jsonl" --expected-visible-train-sha256 0f6054d688d1fdd9b7f332cef703ae7ff0c00956f57c47944223c9883055ad23 \
  --private-split-holdback "${SPLIT_DIR}/holdback.private.jsonl" --expected-private-split-holdback-sha256 419917fceb8fd98849333309270277b412c877f4b2d7205976532bd532d1494b \
  --visible-projection-report "${PROJECTION_DIR}/visible_projection_report.json" --expected-visible-projection-report-sha256 2f2ef4a288da49f47fdd659576a1dda67836bdd069bd9546c8b5b1e479a3426c \
  --visible-projection-journal "${PROJECTION_DIR}/visible_projection.journal.jsonl" --expected-visible-projection-journal-sha256 1359c20028418a1a678c70364b6bf522338ac95e9169a8156626bd32af9b8502 \
  --c001-report "${C001_ROOT}/continuation_report.json" --expected-c001-report-sha256 f137edb5a5484f0f4f8a59e54fb327cbfec754ccb2403844d59d2517d8e519d3 \
  --resume47-report "${RESUME_ROOT}/resume_report.json" --expected-resume47-report-sha256 "${T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256}" \
  --source-plan "${SOURCE_ROOT}/plan_kimi_initial_c002.json" \
  --source-journal "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl" \
  --source-chain-head "${SOURCE_ROOT}/kimi_initial_c002/typed_api_rescue.journal.jsonl.chain-head.json" \
  --output-dir "${OUTPUT_DIR}" --timeout 30
