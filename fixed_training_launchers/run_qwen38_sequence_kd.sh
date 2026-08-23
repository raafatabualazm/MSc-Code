#!/usr/bin/env bash
# Production Qwen3.8-Max-Preview harvest -> direct-compact sequence KD.
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
PATCH_ROOT="${PATCH_ROOT:-${WORKSPACE}/hybrid_training_patch_v2_3}"
RB="${RB:-${WORKSPACE}/multifunction_v1/build}"
TOKENIZER_SNAPSHOT="${TOKENIZER_SNAPSHOT:-${WORKSPACE}/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/artifacts/direct_compact_qwen38_inline_cfg_v2}"
QWEN_ENV_FILE="${QWEN_ENV_FILE:-${WORKSPACE}/Qwen.env}"

TRAIN_JSONL="${RB}/train_multifunction_binary.jsonl"
BASE_TRAIN_SEAL="${RB}/train_multifunction_binary.seal.json"
TRAIN_SEAL="${RB}/train_multifunction_binary_target24k.seal.json"
BUILD_REPORT="${RB}/build_report.json"
BASE_CONTRACT="${BASE_CONTRACT:-${RB}/multifunction_inline_cfg_v2_contract.json}"
CONTRACT="${CONTRACT:-${RB}/multifunction_inline_cfg_v2_target24k_contract.json}"
CODEBOOK="${CODEBOOK:-${RB}/multifunction_inline_cfg_v2_codebook.json}"
CODEC="${CODEC:-${WORKSPACE}/scripts/data/build_multifunction_compact_v2.py}"
TOKENIZER_JSON="${TOKENIZER_SNAPSHOT}/tokenizer.json"
WARMSTART="${DIRECT_COMPACT_WARMSTART:-${WORKSPACE}/artifacts/direct_compact_fn0_real_sft_v1_self_sealed_recovered}"
WARMSTART_RECOVERY_RECEIPT="${WARMSTART_RECOVERY_RECEIPT:-${WORKSPACE}/direct_compact_fn0_self_seal_recovery.json}"
PROMPTS="${PROMPTS:-${RB}/train_multifunction_binary_f2.jsonl}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-${PROMPTS}.manifest.json}"
EXPECTED_BUILD_REPORT_SHA256="${EXPECTED_BUILD_REPORT_SHA256:-7a9fdd032fef34c43ac5e7b8217b6b0b4c986b7dfdf0f1b4b6897aec01df241f}"
EXPECTED_EXPANDED_CONTRACT_SHA256="${EXPECTED_EXPANDED_CONTRACT_SHA256:-f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f}"
EXPECTED_EXPANDED_TRAIN_SEAL_SHA256="${EXPECTED_EXPANDED_TRAIN_SEAL_SHA256:-0444a09d99d8dba8b4069623f50b584e75b66791c03f9aa1493b9bb0024813ae}"

mkdir -p "${WORKSPACE}/locks"
exec 9>"${WORKSPACE}/locks/qwen38_sequence_kd.lock"
if ! flock -n 9; then
  printf 'Another Qwen3.8 sequence harvest/training run holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[qwen38_sequence_kd] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

sha256_of() {
  local value
  value="$(sha256sum "$1")"
  printf '%s\n' "${value%% *}"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  test -f "${path}" || {
    printf 'Missing %s: %s\n' "${label}" "${path}" >&2
    exit 2
  }
  local actual
  actual="$(sha256_of "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    printf '%s hash mismatch: expected %s, got %s\n' \
      "${label}" "${expected}" "${actual}" >&2
    exit 2
  fi
}

require_sha256 "${BUILD_REPORT}" "${EXPECTED_BUILD_REPORT_SHA256}" \
  "inline-CFG v2 multi-function compact build report"
require_sha256 "${WARMSTART_RECOVERY_RECEIPT}" \
  41e0dd7ecf68ebb0b560c66266d686b49afe9e59c6390f1e4079854dca6a7c9b \
  "recovered direct-compact warm-start receipt"
jq -e \
  --arg recovered "${WARMSTART}" '
  .schema == "direct-compact-self-seal-recovery-v1"
  and .recovered_checkpoint == $recovered
  and .artifacts.decoder_adapter_sha256
    == "5516443eb9efe7a32b9ef180f8ae729a96b897dab22d46207c74c00810979e93"
  and .artifacts.source_embedding_overlay_sha256
    == "cffc58811014f4123c91c305347421f4fc9fd1498024a381ddeb0c0a2240eeb0"
  and .artifacts.run_provenance_sha256
    == "fb097e9a494ff8fa15d6b161ad6b2cd8b08de62890c54b8e2ffeb88b2e96589c"
  and .artifacts.provenance_bound_contract_sha256
    == "4801767387e4312cced559166c2fbf7145242ab21b2b35883a54c9f99f367e02"
  and .invariants.validate_self_sealed_checkpoint_passed == true
  and .invariants.source_checkpoint_modified == false
' "${WARMSTART_RECOVERY_RECEIPT}" >/dev/null
require_sha256 "${WARMSTART}/source_embedding_overlay.pt" \
  cffc58811014f4123c91c305347421f4fc9fd1498024a381ddeb0c0a2240eeb0 \
  "recovered warm-start overlay"
require_sha256 "${WARMSTART}/run_provenance.json" \
  fb097e9a494ff8fa15d6b161ad6b2cd8b08de62890c54b8e2ffeb88b2e96589c \
  "recovered warm-start provenance"
require_sha256 "${WARMSTART}/compact_contract.json" \
  4801767387e4312cced559166c2fbf7145242ab21b2b35883a54c9f99f367e02 \
  "recovered warm-start contract"
EXPECTED_TRAIN_SHA256="$(jq -er '.outputs.train.sha256' "${BUILD_REPORT}")"
EXPECTED_BASE_TRAIN_SEAL_SHA256="$(
  jq -er '.outputs.train_seal.sha256' "${BUILD_REPORT}"
)"
EXPECTED_BASE_CONTRACT_SHA256="$(
  jq -er '.outputs.contract.sha256' "${BUILD_REPORT}"
)"
EXPECTED_CODEBOOK_SHA256="$(
  jq -er '.outputs.codebook.sha256' "${BUILD_REPORT}"
)"
EXPECTED_CODEC_SHA256="$(
  jq -er '.derived_representation.codec.sha256' "${BUILD_REPORT}"
)"
EXPECTED_PROMPTS_SHA256="$(
  jq -er '.outputs.train_f2.sha256' "${BUILD_REPORT}"
)"
EXPECTED_PROMPT_MANIFEST_SHA256="$(
  jq -er '.outputs.train_f2_manifest.sha256' "${BUILD_REPORT}"
)"
require_sha256 "${TRAIN_JSONL}" "${EXPECTED_TRAIN_SHA256}" "compact train"
require_sha256 "${BASE_TRAIN_SEAL}" "${EXPECTED_BASE_TRAIN_SEAL_SHA256}" \
  "base compact train seal"
require_sha256 "${TRAIN_SEAL}" "${EXPECTED_EXPANDED_TRAIN_SEAL_SHA256}" \
  "expanded 24K-target compact train seal"
require_sha256 "${BASE_CONTRACT}" "${EXPECTED_BASE_CONTRACT_SHA256}" \
  "base inline-CFG v2 compact contract"
require_sha256 "${CONTRACT}" "${EXPECTED_EXPANDED_CONTRACT_SHA256}" \
  "expanded 24K-target inline-CFG v2 compact contract"
jq -e '
  .max_source_tokens == 9000
  and .max_target_tokens == 24576
  and .max_total_tokens == 36864
  and .decoder_model == "Qwen/Qwen3-8B"
  and .source_token_expansions
  and .schema == "direct-compact-causal-v1"
' "${CONTRACT}" >/dev/null
cmp -s \
  <(jq -S 'del(.max_target_tokens, .max_total_tokens)' "${BASE_CONTRACT}") \
  <(jq -S 'del(.max_target_tokens, .max_total_tokens)' "${CONTRACT}") || {
  printf 'Expanded contract changes fields other than target/total limits\n' >&2
  exit 2
}
cmp -s \
  <(jq -S 'del(.contract_sha256)' "${BASE_TRAIN_SEAL}") \
  <(jq -S 'del(.contract_sha256)' "${TRAIN_SEAL}") || {
  printf 'Expanded train seal changes fields other than contract hash\n' >&2
  exit 2
}
require_sha256 "${CODEBOOK}" "${EXPECTED_CODEBOOK_SHA256}" \
  "derived train-only inline-CFG v2 codebook"
require_sha256 "${CODEC}" "${EXPECTED_CODEC_SHA256}" \
  "inline-CFG v2 codec"
require_sha256 "${TOKENIZER_JSON}" \
  aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
  "student tokenizer"
require_sha256 "${PROMPTS}" "${EXPECTED_PROMPTS_SHA256}" \
  "inline-CFG v2 F2 train prompts"
require_sha256 "${PROMPT_MANIFEST}" \
  "${EXPECTED_PROMPT_MANIFEST_SHA256}" \
  "multi-function F2 prompt manifest"
test -x "${WORKSPACE}/dart-3.12.2/usr/bin/dart"
test -d "${WARMSTART}/decoder_adapter"
test -f "${WARMSTART}/source_embedding_overlay.pt"
test -f "${WARMSTART}/run_provenance.json"
test -f "${PATCH_ROOT}/scripts/run_qwen_sequence_kd_warmstart.sh"
test -f "${QWEN_ENV_FILE}"

mkdir -p "${OUTPUT_ROOT}"

jq -e '
  .schema == "binary-multifunction-compact-build-v2"
  and .counts.train_rows == 1580
  and .counts.dev_rows == 175
  and .counts.excluded_rows == 0
  and .counts.truncated_rows == 0
  and .invariants.all_user_functions_retained == true
  and .invariants.all_machine_instructions_retained == true
  and .invariants.all_cfg_edges_retained_with_global_offsets == true
  and .invariants.all_global_user_call_aliases_retained == true
  and .invariants.all_external_aliases_and_exact_definitions_retained == true
  and .invariants.source_token_id_set_preserved_from_parent == true
  and .invariants.block_and_control_token_ids_preserved_from_parent == true
  and .invariants.instruction_codebook_refit_from_train_only == true
  and .invariants.heldout_rows_used_for_instruction_codebook_fit == 0
  and .invariants.warmstart_overlay_rows_reusable_only_when_expansions_match == true
  and .invariants.inline_cfg_source_is_current_containing_block == true
  and .invariants.inline_cfg_omits_only_redundant_edge_source_tokens == true
  and .invariants.all_inline_cfg_text_and_token_roundtrips_verified == true
  and .invariants.all_student_rows_within_9000 == true
  and .invariants.all_api_prompts_within_12000 == true
  and .invariants.zero_excluded_rows == true
  and .invariants.zero_truncated_rows == true
  and .invariants.dev_is_measure_only_and_not_training == true
  and .invariants.train_dev_representation_contract_identical == true
  and .passed == true
' "${BUILD_REPORT}" >/dev/null

jq -e \
  --arg train_sha256 "${EXPECTED_TRAIN_SHA256}" \
  --arg prompt_sha256 "${EXPECTED_PROMPTS_SHA256}" \
  --arg contract_sha256 "${EXPECTED_BASE_CONTRACT_SHA256}" \
  --arg codebook_sha256 "${EXPECTED_CODEBOOK_SHA256}" \
  --arg codec_sha256 "${EXPECTED_CODEC_SHA256}" '
  .schema == "verified-api-readable-compact-v2"
  and .rows == 1580
  and .dataset.sha256 == $train_sha256
  and .output.sha256 == $prompt_sha256
  and .artifacts.representation_contract.sha256 == $contract_sha256
  and .artifacts.representation_codebook.sha256 == $codebook_sha256
  and .artifacts.inline_cfg_codec.sha256 == $codec_sha256
  and .f2_prompt_contract.representation_schema == "lossless-semantic-f2"
  and (.f2_prompt_contract.system_prompt_sha256
    | test("^[0-9a-f]{64}$"))
  and .f2_prompt_contract.maximum_estimated_prompt_tokens <= 12000
  and .f2_prompt_contract.all_rows_within_limit == true
  and .invariants.all_artifact_hashes_verified == true
  and .invariants.all_row_contract_hashes_verified == true
  and .invariants.all_codec_roundtrips_verified == true
  and .invariants.all_student_constant_prefixes_verified == true
  and .invariants.all_f2_semantic_roundtrips_verified == true
  and .invariants.f2_system_prompt_self_contained_and_hashed == true
  and .invariants.opaque_source_ids_expanded == true
  and .invariants.cfg_explicit == true
  and .invariants.all_user_functions_retained == true
  and .invariants.all_external_symbols_retained == true
  and .invariants.transfer_table_redundancy_proven == true
  and .invariants.train_dev_representation_contract_identical == true
' "${PROMPT_MANIFEST}" >/dev/null

export QWEN_ENV_FILE
export PYTHON_BIN="${PYTHON_BIN:-${PYTHON}}"
export PROMPT_JSONL="${PROMPTS}"
export PROMPT_SHA256
PROMPT_SHA256="$(sha256_of "${PROMPTS}")"
export PROMPT_ROWS=1580
export PROMPT_MANIFEST
export PROMPT_MANIFEST_SHA256
PROMPT_MANIFEST_SHA256="$(sha256_of "${PROMPT_MANIFEST}")"
export VERIFIER_JSONL="${TRAIN_JSONL}"
export VERIFIER_SHA256
VERIFIER_SHA256="$(sha256_of "${TRAIN_JSONL}")"
export STUDENT_TOKENIZER_JSON="${TOKENIZER_JSON}"
export STUDENT_TOKENIZER_SHA256
STUDENT_TOKENIZER_SHA256="$(sha256_of "${TOKENIZER_JSON}")"
export STUDENT_EOS_TOKEN_ID=151645
export COMPACT_TRAIN_JSONL="${TRAIN_JSONL}"
export COMPACT_TRAIN_SEAL="${TRAIN_SEAL}"
export COMPACT_CONTRACT="${CONTRACT}"
export COMPACT_CODEBOOK="${CODEBOOK}"
export COMPACT_CODEC_ARTIFACT="${CODEC}"
export DIRECT_COMPACT_WARMSTART="${WARMSTART}"
export GOLD_ADAPT_TRAIN_ROWS=1580
export OUTPUT_ROOT
export QWEN_DEFER_STUDENT_PREP=1
export CAPACITY_MIGRATED_GOLD_WARMSTART="${OUTPUT_ROOT}/direct_compact_multifunction_gold_sft_target24k"
export CAPACITY_MIGRATION_SOURCE_GOLD="${OUTPUT_ROOT}/direct_compact_multifunction_gold_sft"

# This launcher is the sealed production sequence-distillation arm, not a
# generic Qwen experiment entry point.  Refuse inherited overrides that would
# quietly turn it into a different teacher/objective or skip the exact pilot.
if [[ "${QWEN_TEACHER_MODEL:-qwen3.8-max-preview}" != "qwen3.8-max-preview" ]]; then
  printf 'QWEN_TEACHER_MODEL must be qwen3.8-max-preview\n' >&2
  exit 2
fi
if [[ "${QWEN_OBJECTIVE_MODE:-sequence_only}" != "sequence_only" ]]; then
  printf 'QWEN_OBJECTIVE_MODE must be sequence_only for thinking-mode KD\n' >&2
  exit 2
fi
case "${QWEN_ENABLE_THINKING:-1}" in
  1|true|TRUE|yes|YES) ;;
  *)
    printf 'QWEN_ENABLE_THINKING must be enabled for this production arm\n' >&2
    exit 2
    ;;
esac
if [[ "${TEACHER_MAX_TOKENS:-12288}" != "12288" ]]; then
  printf 'TEACHER_MAX_TOKENS must be exactly 12288\n' >&2
  exit 2
fi
if [[ "${TEACHER_THINKING_BUDGET:-8192}" != "8192" ]]; then
  printf 'TEACHER_THINKING_BUDGET must be exactly 8192\n' >&2
  exit 2
fi
if [[ "${TEACHER_TEMPERATURE:-1.0}" != "1.0" ]]; then
  printf 'TEACHER_TEMPERATURE must be exactly 1.0\n' >&2
  exit 2
fi
if [[ "${TEACHER_TOP_P:-1.0}" != "1.0" ]]; then
  printf 'TEACHER_TOP_P must be exactly 1.0\n' >&2
  exit 2
fi
if [[ "${TEACHER_TOP_K:-101}" != "101" ]]; then
  printf 'TEACHER_TOP_K must be exactly 101 (Alibaba-disabled top-k)\n' >&2
  exit 2
fi
if [[ "${MAX_PROMPT_TOKENS:-12000}" != "12000" ]]; then
  printf 'MAX_PROMPT_TOKENS must be exactly 12000\n' >&2
  exit 2
fi
if [[ "${CHAT_OVERHEAD_RESERVE:-256}" != "256" ]]; then
  printf 'CHAT_OVERHEAD_RESERVE must be exactly 256\n' >&2
  exit 2
fi
if [[ "${QWEN_PILOT_TASKS:-16}" != "16" ]]; then
  printf 'QWEN_PILOT_TASKS must be exactly 16 before the full harvest\n' >&2
  exit 2
fi
if [[ "${GOLD_REPLAY_FRACTION:-0.0}" != "0" \
   && "${GOLD_REPLAY_FRACTION:-0.0}" != "0.0" ]]; then
  printf 'GOLD_REPLAY_FRACTION must be 0.0 for pure Qwen sequence MC forward-KL\n' >&2
  exit 2
fi
if [[ "${BATCH_SIZE:-1}" != "1" ]]; then
  printf 'BATCH_SIZE must be exactly 1 for the 24K-target sequence stage\n' >&2
  exit 2
fi

export QWEN_TEACHER_MODEL="qwen3.8-max-preview"
export QWEN_OBJECTIVE_MODE="sequence_only"
export QWEN_ENABLE_THINKING=1
# Opt in only at this sealed production entry point. The inner launcher keeps
# a fail-closed default and refuses recovery unless Qwen.env independently
# attests Token Plan automation authorization.
export QWEN_ORPHAN_REISSUE_AUTHORIZED=1
export API_KEY_ENV="QWEN_API_KEY"
export MAX_PROMPT_TOKENS=12000
export CHAT_OVERHEAD_RESERVE=256
export TEACHER_MAX_TOKENS=12288
export TEACHER_THINKING_BUDGET=8192
export TEACHER_SEED_BASE="${TEACHER_SEED_BASE:-44}"
export TEACHER_TEMPERATURE=1.0
export TEACHER_TOP_P=1.0
export TEACHER_TOP_K=101
export QWEN_PILOT_TASKS=16
export GOLD_REPLAY_FRACTION=0.0
export BATCH_SIZE=1
export QWEN_TEACHER_WORKERS="${QWEN_TEACHER_WORKERS:-16}"
export QWEN_VERIFIER_WORKERS="${QWEN_VERIFIER_WORKERS:-16}"

exec bash "${PATCH_ROOT}/scripts/run_qwen_sequence_kd_warmstart.sh"
