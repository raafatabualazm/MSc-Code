#!/usr/bin/env bash
set -euo pipefail

# Second local base-only pass.  Unlike an offset expansion, it excludes every
# source task already accepted by the private gate in all three earlier local
# stages, then samples exactly four fresh base candidates for each unresolved
# training task.  This file is intentionally separate from the live first-pass
# launcher: deploying it cannot alter that service.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
RESIDUAL_SCRIPT="${T5GEMMA_RESIDUAL_SCRIPT:-${PROJECT}/scripts/training/t5gemma2_local_rs_sft_residual.py}"
SFT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1"
SFT_CHECKPOINT="${SFT_DIR}/checkpoint-optstep-000348"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
OUTPUT_DIR="${T5GEMMA_RESIDUAL_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_local_base_residual_unresolved4_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

PILOT_REPORT="${T5GEMMA_RESIDUAL_PILOT_REPORT:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json}"
PILOT_JOURNAL="${T5GEMMA_RESIDUAL_PILOT_JOURNAL:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest.journal.jsonl}"
EXPANDED_REPORT="${T5GEMMA_RESIDUAL_EXPANDED_REPORT:-${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest_report.json}"
EXPANDED_JOURNAL="${T5GEMMA_RESIDUAL_EXPANDED_JOURNAL:-${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest.journal.jsonl}"
REMAINING_REPORT="${T5GEMMA_RESIDUAL_REMAINING_REPORT:-${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest_report.json}"
REMAINING_JOURNAL="${T5GEMMA_RESIDUAL_REMAINING_JOURNAL:-${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest.journal.jsonl}"
SONNET_ONE_DIR="${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1"
SONNET_TWO_DIR="${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1"
OPUS_REPORT="${T5GEMMA_RESIDUAL_OPUS_REPORT:-}"
OPUS_JOURNAL="${T5GEMMA_RESIDUAL_OPUS_JOURNAL:-}"
OPUS_TARGETS="${T5GEMMA_RESIDUAL_OPUS_TARGETS:-}"
OPUS_REPORT_SHA="${T5GEMMA_RESIDUAL_OPUS_REPORT_SHA:-}"
OPUS_JOURNAL_SHA="${T5GEMMA_RESIDUAL_OPUS_JOURNAL_SHA:-}"
OPUS_TARGETS_SHA="${T5GEMMA_RESIDUAL_OPUS_TARGETS_SHA:-}"

PILOT_REPORT_SHA="${T5GEMMA_RESIDUAL_PILOT_REPORT_SHA:-b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab}"
PILOT_JOURNAL_SHA="${T5GEMMA_RESIDUAL_PILOT_JOURNAL_SHA:-5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58}"
EXPANDED_REPORT_SHA="${T5GEMMA_RESIDUAL_EXPANDED_REPORT_SHA:-8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50}"
EXPANDED_JOURNAL_SHA="${T5GEMMA_RESIDUAL_EXPANDED_JOURNAL_SHA:-80a326b6b2b2c8bdb0cd745f9884ace91baf411971023b1fed2d98192a022024}"
REMAINING_REPORT_SHA="${T5GEMMA_RESIDUAL_REMAINING_REPORT_SHA:-}"
REMAINING_JOURNAL_SHA="${T5GEMMA_RESIDUAL_REMAINING_JOURNAL_SHA:-}"

SONNET_ONE_REPORT_SHA=fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad
SONNET_ONE_JOURNAL_SHA=b2b6dfbb3d0a3efd5cbadee09e134c24fa7594f6df1238833d25a7b671c9af10
SONNET_ONE_TARGETS_SHA=2bbd8ccc486734a7aed738e9cb705105e79778162f2fbb99798895e9142611d3
SONNET_TWO_REPORT_SHA=99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4
SONNET_TWO_JOURNAL_SHA=4bdeb9e6f5a0d3063b6d454d91bde65596ef788a7edd08d67045fa545b6481d6
SONNET_TWO_TARGETS_SHA=e31d438ade29469b5a742c16f4dc4708b6b8491a6aa8843fad29ee20d8114b1b

for value_name in \
  PILOT_REPORT_SHA PILOT_JOURNAL_SHA \
  EXPANDED_REPORT_SHA EXPANDED_JOURNAL_SHA \
  REMAINING_REPORT_SHA REMAINING_JOURNAL_SHA; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "T5GEMMA_RESIDUAL_BLOCKED fill exact ${value_name} after source harvest completion" >&2
    exit 78
  fi
done
if [[ ! -x "${DART_BIN}" ]] || [[ ! -d "${SFT_CHECKPOINT}" ]]; then
  echo "T5GEMMA_RESIDUAL_BLOCKED required checkpoint or Dart runtime is absent" >&2
  exit 78
fi
if [[ ! -s "${RESIDUAL_SCRIPT}" ]]; then
  echo "T5GEMMA_RESIDUAL_BLOCKED residual trainer is absent" >&2
  exit 78
fi
API_EXCLUSION_ARGS=(
  --exclude_verified_api_report "${SONNET_ONE_DIR}/api_rescue_report.json"
  --exclude_verified_api_journal "${SONNET_ONE_DIR}/api_rescue.journal.jsonl"
  --exclude_verified_api_targets "${SONNET_ONE_DIR}/direct_hard_targets.jsonl"
  --expected_exclude_verified_api_report_sha256 "${SONNET_ONE_REPORT_SHA}"
  --expected_exclude_verified_api_journal_sha256 "${SONNET_ONE_JOURNAL_SHA}"
  --expected_exclude_verified_api_targets_sha256 "${SONNET_ONE_TARGETS_SHA}"
  --exclude_verified_api_report "${SONNET_TWO_DIR}/api_rescue_report.json"
  --exclude_verified_api_journal "${SONNET_TWO_DIR}/api_rescue.journal.jsonl"
  --exclude_verified_api_targets "${SONNET_TWO_DIR}/direct_hard_targets.jsonl"
  --expected_exclude_verified_api_report_sha256 "${SONNET_TWO_REPORT_SHA}"
  --expected_exclude_verified_api_journal_sha256 "${SONNET_TWO_JOURNAL_SHA}"
  --expected_exclude_verified_api_targets_sha256 "${SONNET_TWO_TARGETS_SHA}"
)
if [[ -n "${OPUS_REPORT}${OPUS_JOURNAL}${OPUS_TARGETS}${OPUS_REPORT_SHA}${OPUS_JOURNAL_SHA}${OPUS_TARGETS_SHA}" ]]; then
  for value_name in OPUS_REPORT OPUS_JOURNAL OPUS_TARGETS OPUS_REPORT_SHA OPUS_JOURNAL_SHA OPUS_TARGETS_SHA; do
    if [[ -z "${!value_name}" ]]; then
      echo "T5GEMMA_RESIDUAL_BLOCKED Opus exclusion fields must be filled together" >&2
      exit 78
    fi
  done
  for value_name in OPUS_REPORT_SHA OPUS_JOURNAL_SHA OPUS_TARGETS_SHA; do
    if ! [[ "${!value_name}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "T5GEMMA_RESIDUAL_BLOCKED invalid ${value_name}" >&2
      exit 78
    fi
  done
  API_EXCLUSION_ARGS+=(
    --exclude_verified_api_report "${OPUS_REPORT}"
    --exclude_verified_api_journal "${OPUS_JOURNAL}"
    --exclude_verified_api_targets "${OPUS_TARGETS}"
    --expected_exclude_verified_api_report_sha256 "${OPUS_REPORT_SHA}"
    --expected_exclude_verified_api_journal_sha256 "${OPUS_JOURNAL_SHA}"
    --expected_exclude_verified_api_targets_sha256 "${OPUS_TARGETS_SHA}"
  )
fi
if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-enriched-sft-run-v1"
   and .status == "complete"
   and .updates == 348
   and .latest_checkpoint == "checkpoint-optstep-000348"
   and .no_frontier_api == true' \
  "${SFT_DIR}/result.json" >/dev/null; then
  echo "T5GEMMA_RESIDUAL_BLOCKED two-epoch SFT result differs" >&2
  exit 78
fi
printf '%s  %s\n' \
  562c3da5f89428e6a7263ad8ec79dde9c8b6eb25c77949606277d7d80aecea4f "${SFT_CHECKPOINT}/run_contract.json" \
  c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3 "${SFT_CHECKPOINT}/adapter/adapter_config.json" \
  83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc "${SFT_CHECKPOINT}/adapter/adapter_model.safetensors" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d "${SFT_CHECKPOINT}/tokenizer/tokenizer.json" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python "${RESIDUAL_SCRIPT}" \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected_rollout_sha256 14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  --expected_f2_sha256 c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  --expected_private_holdback_sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --exclude_verified_report "${PILOT_REPORT}" \
  --exclude_verified_journal "${PILOT_JOURNAL}" \
  --expected_exclude_verified_report_sha256 "${PILOT_REPORT_SHA}" \
  --expected_exclude_verified_journal_sha256 "${PILOT_JOURNAL_SHA}" \
  --exclude_verified_report "${EXPANDED_REPORT}" \
  --exclude_verified_journal "${EXPANDED_JOURNAL}" \
  --expected_exclude_verified_report_sha256 "${EXPANDED_REPORT_SHA}" \
  --expected_exclude_verified_journal_sha256 "${EXPANDED_JOURNAL_SHA}" \
  --exclude_verified_report "${REMAINING_REPORT}" \
  --exclude_verified_journal "${REMAINING_JOURNAL}" \
  --expected_exclude_verified_report_sha256 "${REMAINING_REPORT_SHA}" \
  --expected_exclude_verified_journal_sha256 "${REMAINING_JOURNAL_SHA}" \
  "${API_EXCLUSION_ARGS[@]}" \
  --sft_checkpoint "${SFT_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --pilot_offset 0 \
  --pilot_tasks 1500 \
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
