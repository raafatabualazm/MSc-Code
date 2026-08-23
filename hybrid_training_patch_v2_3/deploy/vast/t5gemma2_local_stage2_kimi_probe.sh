#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
MIXED_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_mixed_rs_sft_final_v1"
MIXED_CHECKPOINT="${MIXED_DIR}/checkpoint-optstep-000426"
OUTPUT_DIR="${T5GEMMA_STAGE2_KIMI_PROBE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_mixed_kimi_probe_v1}"
COMPAT_RECORD="${OUTPUT_DIR}/checkpoint-loader-compat.json"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
PILOT_CORE="${PROJECT}/scripts/training/t5gemma2_local_rs_sft_pilot.py"
INFERENCE_CORE="${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py"
MIXED_LOADER="${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_mixed_compat.py"
COMPAT_WRAPPER="${PROJECT}/scripts/training/t5gemma2_local_rs_sft_mixed_compat.py"

if [[ ! -s "${MIXED_DIR}/result.json" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-mixed-rs-sft-run-v1"
     and .status == "complete"
     and .architecture == "native_encoder_decoder"
     and .updates == 426
     and .planned_updates == 426
     and .rows == 1132
     and .latest_checkpoint == "checkpoint-optstep-000426"
     and .production_floor_eligible == true' \
    "${MIXED_DIR}/result.json" >/dev/null; then
  echo "T5GEMMA_STAGE2_KIMI_PROBE_BLOCKED completed mixed result changed" >&2
  exit 78
fi
if [[ ! -d "${MIXED_CHECKPOINT}" ]] \
  || [[ ! -s "${MIXED_CHECKPOINT}/run_contract.json" ]] \
  || ! cmp -s "${MIXED_DIR}/run_contract.json" \
    "${MIXED_CHECKPOINT}/run_contract.json"; then
  echo "T5GEMMA_STAGE2_KIMI_PROBE_BLOCKED mixed checkpoint is absent" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_STAGE2_KIMI_PROBE_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi

# Pin the checkpoint, compatibility loader/wrapper, historical
# generation/scoring cores, and TRAIN-only feedback views. No held-out
# evaluation artifact is opened by this launcher or harvester.
printf '%s  %s\n' \
  6b00c4f72405579d6a89f77448dbbcdc84812851ff6c80bdaac3a9424601269b \
  "${MIXED_DIR}/result.json" \
  a6b4e28ba647f09f393e265ad40053c9f227d49f0dd894d5b5d942a1111b1728 \
  "${MIXED_CHECKPOINT}/run_contract.json" \
  be95b3bec613478790facb2bb6ec29243a3625e468560b0b678def8f016a46da \
  "${MIXED_CHECKPOINT}/adapter/adapter_model.safetensors" \
  75aad08619087aa2d7ef5db9d50b1890ca74feccf85a2c104a93dbddb0d7b9e6 \
  "${MIXED_CHECKPOINT}/adapter/adapter_config.json" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d \
  "${MIXED_CHECKPOINT}/tokenizer/tokenizer.json" \
  0a6134c1753e69b75aa46eb4e762ab463b61c411db0c4c3ba7b18fe2f8e96f1d \
  "${PILOT_CORE}" \
  564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9 \
  "${INFERENCE_CORE}" \
  758b0cf37475cacf8789ce9db62d3e6e8f88fe344c6d616c53b0c1d221921972 \
  "${MIXED_LOADER}" \
  8ddf567c42124e6d549c69eb3748ecce256c8ec20522a8aae7f32732167a6477 \
  "${COMPAT_WRAPPER}" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

mkdir -p "${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/harvest_report.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/harvest_report.json")" == complete ]]; then
  echo "T5GEMMA_STAGE2_KIMI_PROBE_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python \
  scripts/training/t5gemma2_local_rs_sft_mixed_compat.py \
  --compat_record "${COMPAT_RECORD}" \
  --compat_checkpoint "${MIXED_CHECKPOINT}" \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected_rollout_sha256 14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  --expected_f2_sha256 c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  --expected_private_holdback_sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --output_dir "${OUTPUT_DIR}" \
  --pilot_tasks 100 \
  --pilot_offset 0 \
  --base_samples 4 \
  --repair_samples 0 \
  --max_repair_parents 0 \
  --gold_replay_ratio 0 \
  --production_min_unique_targets 1 \
  --max_source_tokens 32768 \
  --max_new_tokens 4096 \
  --generation_batch_size 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --evaluation_workers 8 \
  --timeout 30 \
  --stability_runs 2 \
  --seed 20260730 \
  --attn_implementation sdpa \
  --bf16
