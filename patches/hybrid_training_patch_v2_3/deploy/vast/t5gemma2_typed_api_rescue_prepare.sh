#!/usr/bin/env bash
set -euo pipefail

# CPU-only preparation: build the full clean visible/complement split, then
# project all completed local K=4 candidates onto the visible half.  No API
# credential is read and no frontier call is possible in this launcher.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
LOCAL_DIR="${T5GEMMA_TYPED_LOCAL_HARVEST_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1}"
EXISTING_MANIFEST="${T5GEMMA_TYPED_225_MANIFEST:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1/dataset_manifest.json}"
SPLIT_DIR="${T5GEMMA_TYPED_API_SPLIT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_api_visible_split_v1}"
PROJECTION_DIR="${T5GEMMA_TYPED_API_PROJECTION_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_visible_failure_projection_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

printf '%s  %s\n' \
  0cc6323136e12110e6ee25ef42f8e0baec90359ef7cc972fb191691a12fd7b15 "${PROJECT}/scripts/preprocessing/build_t5gemma2_typed_api_visible_split.py" \
  b7f66f7ee0b1959fe6a6b8bbe6fa422d545ab950971dd6aae86bfb31acde0f88 "${PROJECT}/scripts/training/t5gemma2_typed_visible_failure_projection.py" \
  7a03af003e998497012706361f5cbf0734d8defa82c7e458aa5f87f796e01143 "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_cascade.py" \
  | sha256sum -c -

required_env=(
  T5GEMMA_TYPED_LOCAL_REPORT_SHA256
  T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256
  T5GEMMA_TYPED_LOCAL_TARGETS_SHA256
  T5GEMMA_TYPED_225_MANIFEST_SHA256
)
for name in "${required_env[@]}"; do
  value="${!name:-}"
  if ! [[ "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "T5GEMMA_TYPED_API_PREPARE_BLOCKED ${name} is not pinned" >&2
    exit 78
  fi
done

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  "${T5GEMMA_TYPED_LOCAL_REPORT_SHA256}" "${LOCAL_DIR}/harvest_report.json" \
  "${T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256}" "${LOCAL_DIR}/harvest.journal.jsonl" \
  "${T5GEMMA_TYPED_LOCAL_TARGETS_SHA256}" "${LOCAL_DIR}/direct_targets.jsonl" \
  "${T5GEMMA_TYPED_225_MANIFEST_SHA256}" "${EXISTING_MANIFEST}" \
  | sha256sum -c -
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_TYPED_API_PREPARE_BLOCKED Dart 3.12.2 is absent" >&2
  exit 78
fi

export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
mkdir -p "${SPLIT_DIR}" "${PROJECTION_DIR}"
cd "${PROJECT}"

if [[ ! -s "${SPLIT_DIR}/split_manifest.json" ]]; then
  /venv/main/bin/python scripts/preprocessing/build_t5gemma2_typed_api_visible_split.py \
    --gold_train_jsonl "${GOLD_TRAIN}" \
    --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
    --output_dir "${SPLIT_DIR}" \
    --seed 20260801
fi

SPLIT_MANIFEST_SHA="$(sha256sum "${SPLIT_DIR}/split_manifest.json" | cut -d' ' -f1)"
VISIBLE_SHA="$(sha256sum "${SPLIT_DIR}/visible_train.jsonl" | cut -d' ' -f1)"
PRIVATE_SHA="$(sha256sum "${SPLIT_DIR}/holdback.private.jsonl" | cut -d' ' -f1)"

/venv/main/bin/python scripts/training/t5gemma2_typed_visible_failure_projection.py \
  --local_harvest_report "${LOCAL_DIR}/harvest_report.json" \
  --expected_local_harvest_report_sha256 "${T5GEMMA_TYPED_LOCAL_REPORT_SHA256}" \
  --pilot_journal "${LOCAL_DIR}/harvest.journal.jsonl" \
  --expected_local_harvest_journal_sha256 "${T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256}" \
  --local_harvest_targets "${LOCAL_DIR}/direct_targets.jsonl" \
  --expected_local_harvest_targets_sha256 "${T5GEMMA_TYPED_LOCAL_TARGETS_SHA256}" \
  --existing_direct_manifest "${EXISTING_MANIFEST}" \
  --expected_existing_direct_manifest_sha256 "${T5GEMMA_TYPED_225_MANIFEST_SHA256}" \
  --gold_train_jsonl "${GOLD_TRAIN}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --gold_f2_jsonl "${GOLD_F2}" \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --heldout_jsonl "${HELDOUT}" \
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --visible_split_manifest "${SPLIT_DIR}/split_manifest.json" \
  --expected_visible_split_manifest_sha256 "${SPLIT_MANIFEST_SHA}" \
  --visible_train "${SPLIT_DIR}/visible_train.jsonl" \
  --expected_visible_train_sha256 "${VISIBLE_SHA}" \
  --private_split_holdback "${SPLIT_DIR}/holdback.private.jsonl" \
  --expected_private_split_holdback_sha256 "${PRIVATE_SHA}" \
  --output_dir "${PROJECTION_DIR}" \
  --timeout 30 \
  --evaluation_workers 16

printf 'T5GEMMA_TYPED_API_PREPARE_COMPLETE split_manifest_sha256=%s visible_sha256=%s private_sha256=%s projection_report_sha256=%s projection_journal_sha256=%s\n' \
  "${SPLIT_MANIFEST_SHA}" \
  "${VISIBLE_SHA}" \
  "${PRIVATE_SHA}" \
  "$(sha256sum "${PROJECTION_DIR}/visible_projection_report.json" | cut -d' ' -f1)" \
  "$(sha256sum "${PROJECTION_DIR}/visible_projection.journal.jsonl" | cut -d' ' -f1)"
