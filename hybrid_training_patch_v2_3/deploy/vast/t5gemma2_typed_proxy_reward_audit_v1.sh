#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
HARVEST="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1/harvest.journal.jsonl"
FEEDBACK="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1/verpo_rollout_feedback.jsonl"
OUTPUT="${WORKSPACE}/artifacts/t5gemma2_typed_proxy_reward_audit_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

mkdir -p "${OUTPUT}"
export PYTHONPATH="${PROJECT}"
export CUDA_VISIBLE_DEVICES=-1
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec nice -n 10 /venv/main/bin/python \
  scripts/evaluation/audit_t5gemma2_typed_proxy_reward_surface.py \
  --harvest_journal "${HARVEST}" \
  --feedback_jsonl "${FEEDBACK}" \
  --output_journal "${OUTPUT}/reward_audit.journal.jsonl" \
  --output_summary "${OUTPUT}/reward_audit.summary.json" \
  --dart_bin "${DART_BIN}" \
  --workers 8
