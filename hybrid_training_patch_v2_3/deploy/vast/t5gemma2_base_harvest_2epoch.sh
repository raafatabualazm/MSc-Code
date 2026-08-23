#!/usr/bin/env bash
set -euo pipefail

# Base-only expansion over the next 1,000 deterministic TRAIN tasks.  The
# offset skips the 200-task two-epoch pilot while preserving the same seeded
# ordering.  This process owns the GPU but can overlap an API-only rescue job.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SFT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1"
SFT_CHECKPOINT="${SFT_DIR}/checkpoint-optstep-000348"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
OUTPUT_DIR="${T5GEMMA_BASE_HARVEST_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1}"
PILOT_OFFSET="${T5GEMMA_BASE_HARVEST_OFFSET:-200}"
PILOT_TASKS="${T5GEMMA_BASE_HARVEST_TASKS:-1000}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

for value_name in PILOT_OFFSET PILOT_TASKS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "T5GEMMA_BASE_HARVEST_BLOCKED invalid ${value_name}: ${value}" >&2
    exit 78
  fi
done
if (( PILOT_TASKS == 0 || PILOT_OFFSET + PILOT_TASKS > 2386 )); then
  echo "T5GEMMA_BASE_HARVEST_BLOCKED requested slice exceeds the 2,386-task pool" >&2
  exit 78
fi

if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-enriched-sft-run-v1"
   and .status == "complete"
   and .updates == 348
   and .planned_updates == 348
   and .rows == 2776
   and .latest_checkpoint == "checkpoint-optstep-000348"
   and .no_frontier_api == true' \
  "${SFT_DIR}/result.json" >/dev/null; then
  echo "T5GEMMA_BASE_HARVEST_BLOCKED two-epoch SFT result differs" >&2
  exit 78
fi
if [[ ! -d "${SFT_CHECKPOINT}" ]] \
  || [[ ! -s "${SFT_CHECKPOINT}/adapter/adapter_model.safetensors" ]] \
  || [[ ! -s "${SFT_CHECKPOINT}/adapter/adapter_config.json" ]] \
  || [[ ! -s "${SFT_CHECKPOINT}/tokenizer/tokenizer.json" ]] \
  || [[ ! -s "${SFT_CHECKPOINT}/run_contract.json" ]] \
  || ! cmp -s "${SFT_DIR}/run_contract.json" \
    "${SFT_CHECKPOINT}/run_contract.json"; then
  echo "T5GEMMA_BASE_HARVEST_BLOCKED final checkpoint is absent or unbound" >&2
  exit 78
fi
if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-enriched-sft-run-v1"
   and .architecture == "native_encoder_decoder"
   and .base_model.name == "google/t5gemma-2-4b-4b"
   and .base_model.resolved_commit
     == "487d4acf21a4d70c70bf534265b5263c9424979e"
   and .optimization.epochs == 2
   and .optimization.planned_updates == 348
   and .optimization.seed == 42
   and .lora.rank == 64
   and .lora.alpha == 128
   and .lora.encoder_and_decoder_trainable == true
   and .dataset.rows == 2776
   and .dataset.task_ids_sha256
     == "8c3e78cb0fc5a2483a01029a13be9f0536c203de0d28c3302b87eba34b36f3d0"
   and .no_frontier_api == true
   and .tests_exposed_to_model == false' \
  "${SFT_CHECKPOINT}/run_contract.json" >/dev/null; then
  echo "T5GEMMA_BASE_HARVEST_BLOCKED checkpoint contract differs" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_BASE_HARVEST_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi

# Pin every mutable input plus the exact adapter/tokenizer used by the sealed
# two-epoch evaluation. The run itself records a deeper checkpoint contract.
printf '%s  %s\n' \
  f0d02161da9fac96d31085eb8b569ab44dc42902853db3cf1095d6643dd26dbe \
  "${SFT_DIR}/result.json" \
  562c3da5f89428e6a7263ad8ec79dde9c8b6eb25c77949606277d7d80aecea4f \
  "${SFT_CHECKPOINT}/run_contract.json" \
  c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3 \
  "${SFT_CHECKPOINT}/adapter/adapter_config.json" \
  83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc \
  "${SFT_CHECKPOINT}/adapter/adapter_model.safetensors" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d \
  "${SFT_CHECKPOINT}/tokenizer/tokenizer.json" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

for training_file in \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl"; do
  # The enriched SFT corpus has 2,776 rows, while the sealed feedback pool has
  # 2,386 mutually aligned tasks after exclusions.  Do not infer this count
  # from the parent directory name.
  if [[ "$(/usr/bin/wc -l < "${training_file}")" -ne 2386 ]]; then
    echo "T5GEMMA_BASE_HARVEST_BLOCKED feedback pool is not 2,386 rows" >&2
    exit 78
  fi
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_local_rs_sft_pilot.py \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected_rollout_sha256 14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  --expected_f2_sha256 c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  --expected_private_holdback_sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --sft_checkpoint "${SFT_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --pilot_offset "${PILOT_OFFSET}" \
  --pilot_tasks "${PILOT_TASKS}" \
  --base_samples 4 \
  --repair_samples 0 \
  --max_repair_parents 0 \
  --gold_replay_ratio 3 \
  --production_min_unique_targets 200 \
  --max_source_tokens 32768 \
  --max_new_tokens 4096 \
  --generation_batch_size 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --evaluation_workers 8 \
  --timeout 30 \
  --stability_runs 2 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16
