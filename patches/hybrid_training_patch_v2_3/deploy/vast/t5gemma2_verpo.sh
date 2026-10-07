#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SFT_DIR="${T5GEMMA_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_v1}"
OUTPUT_DIR="${T5GEMMA_VERPO_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_compiler_verpo_v1}"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
MAX_UPDATES="${VERPO_MAX_UPDATES:-2386}"
WAIT_INTERVAL="${T5GEMMA_VERPO_WAIT_INTERVAL:-30}"

if ! [[ "${WAIT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "T5GEMMA_VERPO_BLOCKED invalid wait interval: ${WAIT_INTERVAL}" >&2
  exit 78
fi
while [[ ! -f "${SFT_DIR}/result.json" ]] \
  || [[ "$(/usr/bin/jq -r '.status // empty' "${SFT_DIR}/result.json")" != complete ]]; do
  echo "T5GEMMA_VERPO_WAITING for completed SFT: ${SFT_DIR}"
  sleep "${WAIT_INTERVAL}"
done
# The SFT process writes its result immediately before exiting. Give CUDA
# teardown a short deterministic grace period before loading the policy.
sleep 15

warmstart="${SFT_DIR}/$(
  /usr/bin/jq -r '.latest_checkpoint // empty' "${SFT_DIR}/result.json"
)"
if [[ ! -d "${warmstart}" ]]; then
  echo "T5GEMMA_VERPO_BLOCKED missing SFT warmstart ${warmstart}" >&2
  exit 78
fi

public_manifest="${FEEDBACK_DIR}/verpo_feedback_view.public.json"
printf '%s  %s\n' \
  11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614 \
  "${public_manifest}" | sha256sum -c -

mkdir -p "${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/result.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/result.json")" == complete ]]; then
  echo "T5GEMMA_VERPO_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

resume_args=()
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  resume_checkpoint=$(/usr/bin/jq -r '.path // empty' "${OUTPUT_DIR}/latest_checkpoint.json")
  if [[ -n "${resume_checkpoint}" && -d "${resume_checkpoint}" ]]; then
    resume_args=(--resume_checkpoint "${resume_checkpoint}")
    echo "T5GEMMA_VERPO_RESUME checkpoint=${resume_checkpoint}"
  else
    echo "T5GEMMA_VERPO_BLOCKED invalid latest checkpoint pointer" >&2
    exit 78
  fi
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_compiler_feedback_verpo.py \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --rollout_seal "${FEEDBACK_DIR}/verpo_rollout_feedback.seal.json" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --f2_manifest "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl.manifest.json" \
  --feedback_public_manifest "${public_manifest}" \
  --expected_feedback_public_manifest_sha256 11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614 \
  --compact_contract "${WORKSPACE}/multifunction_v1/expanded2776/executable_target24k/compact_contract.json" \
  --warmstart_checkpoint "${warmstart}" \
  --output_dir "${OUTPUT_DIR}" \
  --group_size 4 \
  --repair_group_size 4 \
  --max_repair_parents 2 \
  --tasks_per_update 1 \
  --max_updates "${MAX_UPDATES}" \
  --temperature 0.8 \
  --max_new_tokens 32767 \
  --max_source_tokens 65536 \
  --max_target_tokens 32768 \
  --compile_weight 0.25 \
  --sft_replay_weight 0.02 \
  --checkpoint_interval 1 \
  --keep_last_checkpoints 2 \
  --attn_implementation sdpa \
  --bf16 \
  "${resume_args[@]}"
