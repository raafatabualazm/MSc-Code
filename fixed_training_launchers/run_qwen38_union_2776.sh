#!/usr/bin/env bash
# Materialize and union the two sealed Qwen parents; no API or GPU work.
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
PATCH_ROOT="${PATCH_ROOT:-${WORKSPACE}/hybrid_training_patch_v2_3}"
LEGACY_BUILD_ROOT="${LEGACY_BUILD_ROOT:-${WORKSPACE}/multifunction_v1/build}"
EXPANDED_ROOT="${EXPANDED_ROOT:-${WORKSPACE}/multifunction_v1/expanded2776}"
LEGACY_QWEN_ROOT="${LEGACY_QWEN_ROOT:-${WORKSPACE}/artifacts/direct_compact_qwen38_inline_cfg_v2}"
SUPPLEMENT_QWEN_ROOT="${SUPPLEMENT_QWEN_ROOT:-${WORKSPACE}/artifacts/direct_compact_qwen38_supplement1196}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/artifacts/direct_compact_qwen38_union2776}"
PARENT_ARTIFACT_ROOT="${PARENT_ARTIFACT_ROOT:-${OUTPUT_ROOT}/parents}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${WORKSPACE}/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"

DERIVATION="${DERIVATION:-${EXPANDED_ROOT}/qwen_2776_supplement.derivation.json}"
FIT_COMPACT="${FIT_COMPACT:-${EXPANDED_ROOT}/fit2776_multifunction_binary.jsonl}"
FIT_SEAL="${FIT_SEAL:-${EXPANDED_ROOT}/fit2776_multifunction_binary.target24k.seal.json}"
CONTRACT="${CONTRACT:-${LEGACY_BUILD_ROOT}/multifunction_inline_cfg_v2_target24k_contract.json}"

LEGACY_COMPACT="${LEGACY_COMPACT:-${LEGACY_BUILD_ROOT}/train_multifunction_binary.jsonl}"
LEGACY_SEAL="${LEGACY_SEAL:-${LEGACY_BUILD_ROOT}/train_multifunction_binary_target24k.seal.json}"
LEGACY_PROMPTS="${LEGACY_PROMPTS:-${LEGACY_BUILD_ROOT}/train_multifunction_binary_f2.jsonl}"
LEGACY_JOURNAL="${LEGACY_JOURNAL:-${LEGACY_QWEN_ROOT}/qwen_teacher.journal.jsonl}"
LEGACY_AUDIT="${LEGACY_AUDIT:-${LEGACY_QWEN_ROOT}/qwen_teacher.audit.json}"
LEGACY_PARSEABLE="${LEGACY_PARSEABLE:-${LEGACY_QWEN_ROOT}/qwen_teacher.parseable.mc_sequence.jsonl}"

mkdir -p "${WORKSPACE}/locks" "${OUTPUT_ROOT}" "${PARENT_ARTIFACT_ROOT}"
exec 9>"${WORKSPACE}/locks/qwen38_union2776.lock"
if ! flock -n 9; then
  printf 'Another Qwen 2,776-task union build holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[qwen38_union2776] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

sha256_of() {
  local value
  value="$(sha256sum "$1")"
  printf '%s\n' "${value%% *}"
}

require_file() {
  test -f "$1" || {
    printf 'Required union input is missing: %s\n' "$1" >&2
    exit 2
  }
}

for path in \
  "${DERIVATION}" "${FIT_COMPACT}" "${FIT_SEAL}" "${CONTRACT}" \
  "${TOKENIZER_JSON}" "${LEGACY_COMPACT}" "${LEGACY_SEAL}" \
  "${LEGACY_PROMPTS}" "${LEGACY_JOURNAL}" "${LEGACY_AUDIT}" \
  "${LEGACY_PARSEABLE}" \
  "${SUPPLEMENT_QWEN_ROOT}/qwen_teacher.journal.jsonl" \
  "${SUPPLEMENT_QWEN_ROOT}/qwen_teacher.audit.json" \
  "${SUPPLEMENT_QWEN_ROOT}/qwen_teacher.parseable.mc_sequence.jsonl"; do
  require_file "${path}"
done

jq -e '
  .schema == "qwen-2776-supplement-derivation-v1"
  and .counts.fit_tasks == 2776
  and .counts.legacy_parent_tasks == 1580
  and .counts.supplement_tasks == 1196
  and .counts.heldout_tasks == 175
  and .heldout_intersection_count == 0
  and .invariants.live_journal_modified == false
' "${DERIVATION}" >/dev/null

TOKENIZER_SHA256="$(sha256_of "${TOKENIZER_JSON}")"
cd "${PATCH_ROOT}"

materialize_parent() {
  local label="$1"
  local compact="$2"
  local seal="$3"
  local prompts="$4"
  local journal="$5"
  local audit="$6"
  local parseable="$7"
  local root="$8"
  mkdir -p "${root}"
  local sequence="${root}/qwen_mc_sequence_train.jsonl"
  local sequence_seal="${root}/qwen_mc_sequence_train.seal.json"
  local sequence_schedule="${root}/qwen_mc_sequence_train.schedule.jsonl"
  local sequence_build="${root}/qwen_mc_sequence_train.build.json"
  local cot="${root}/qwen_cot_sft_train.jsonl"
  local cot_seal="${root}/qwen_cot_sft_train.seal.json"
  local cot_schedule="${root}/qwen_cot_sft_train.schedule.jsonl"
  local cot_build="${root}/qwen_cot_sft_train.build.json"
  local prompt_sha journal_sha audit_sha parseable_sha
  prompt_sha="$(sha256_of "${prompts}")"
  journal_sha="$(sha256_of "${journal}")"
  audit_sha="$(sha256_of "${audit}")"
  parseable_sha="$(sha256_of "${parseable}")"
  # Always rebuild these deterministic derivatives from the current final
  # journal/audit hashes. Reusing a stale derivative after journal recovery is
  # forbidden even if a same-named file happens to exist.
  "${PYTHON}" -m scripts.training.build_qwen_sequence_kd \
    --compact-train-jsonl "${compact}" \
    --compact-train-seal "${seal}" \
    --contract "${CONTRACT}" \
    --prompt-jsonl "${prompts}" \
    --expected-prompt-sha256 "${prompt_sha}" \
    --teacher-parseable-jsonl "${parseable}" \
    --expected-teacher-parseable-sha256 "${parseable_sha}" \
    --teacher-journal "${journal}" \
    --expected-teacher-journal-sha256 "${journal_sha}" \
    --teacher-audit-json "${audit}" \
    --expected-teacher-audit-sha256 "${audit_sha}" \
    --student-tokenizer-json "${TOKENIZER_JSON}" \
    --expected-student-tokenizer-sha256 "${TOKENIZER_SHA256}" \
    --output-jsonl "${sequence}" \
    --output-seal "${sequence_seal}" \
    --schedule-output "${sequence_schedule}" \
    --build-manifest "${sequence_build}" \
    --gold-replay-fraction 0.0 \
    --seed 44
  "${PYTHON}" -m scripts.training.build_qwen_cot_sft \
    --compact-train-jsonl "${compact}" \
    --compact-train-seal "${seal}" \
    --contract "${CONTRACT}" \
    --prompt-jsonl "${prompts}" \
    --expected-prompt-sha256 "${prompt_sha}" \
    --teacher-journal "${journal}" \
    --expected-teacher-journal-sha256 "${journal_sha}" \
    --teacher-audit-json "${audit}" \
    --expected-teacher-audit-sha256 "${audit_sha}" \
    --student-tokenizer-json "${TOKENIZER_JSON}" \
    --expected-student-tokenizer-sha256 "${TOKENIZER_SHA256}" \
    --output-jsonl "${cot}" \
    --output-seal "${cot_seal}" \
    --schedule-output "${cot_schedule}" \
    --build-manifest "${cot_build}" \
    --min-nonempty-reasoning-fraction 0.90
  printf 'QWEN_PARENT_MATERIALIZED label=%s sequence=%s cot=%s\n' \
    "${label}" "${sequence_build}" "${cot_build}"
}

materialize_parent \
  legacy "${LEGACY_COMPACT}" "${LEGACY_SEAL}" "${LEGACY_PROMPTS}" \
  "${LEGACY_JOURNAL}" "${LEGACY_AUDIT}" "${LEGACY_PARSEABLE}" \
  "${PARENT_ARTIFACT_ROOT}/legacy"
materialize_parent \
  supplement \
  "${EXPANDED_ROOT}/supplement1196_multifunction_binary.jsonl" \
  "${EXPANDED_ROOT}/supplement1196_multifunction_binary.target24k.seal.json" \
  "${EXPANDED_ROOT}/supplement1196_multifunction_binary_f2.jsonl" \
  "${SUPPLEMENT_QWEN_ROOT}/qwen_teacher.journal.jsonl" \
  "${SUPPLEMENT_QWEN_ROOT}/qwen_teacher.audit.json" \
  "${SUPPLEMENT_QWEN_ROOT}/qwen_teacher.parseable.mc_sequence.jsonl" \
  "${PARENT_ARTIFACT_ROOT}/supplement"

parent_args() {
  local prefix="$1"
  local root="$2"
  local stem="$3"
  PARENT_ARGS+=(
    "--${prefix}-jsonl" "${root}/${stem}_train.jsonl"
    "--${prefix}-seal" "${root}/${stem}_train.seal.json"
    "--${prefix}-schedule" "${root}/${stem}_train.schedule.jsonl"
    "--${prefix}-manifest" "${root}/${stem}_train.build.json"
    "--expected-${prefix}-manifest-sha256"
    "$(sha256_of "${root}/${stem}_train.build.json")"
  )
}

PARENT_ARGS=()
parent_args legacy-sequence "${PARENT_ARTIFACT_ROOT}/legacy" qwen_mc_sequence
parent_args supplement-sequence "${PARENT_ARTIFACT_ROOT}/supplement" qwen_mc_sequence
parent_args legacy-cot "${PARENT_ARTIFACT_ROOT}/legacy" qwen_cot_sft
parent_args supplement-cot "${PARENT_ARTIFACT_ROOT}/supplement" qwen_cot_sft

"${PYTHON}" -m scripts.training.union_qwen_2776_training_artifacts \
  --derivation-manifest "${DERIVATION}" \
  --expected-derivation-manifest-sha256 "$(sha256_of "${DERIVATION}")" \
  --fit-compact-jsonl "${FIT_COMPACT}" \
  --fit-compact-seal "${FIT_SEAL}" \
  --contract "${CONTRACT}" \
  --student-tokenizer-json "${TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "${TOKENIZER_SHA256}" \
  "${PARENT_ARGS[@]}" \
  --sequence-output-jsonl "${OUTPUT_ROOT}/qwen_mc_sequence_train.jsonl" \
  --sequence-output-seal "${OUTPUT_ROOT}/qwen_mc_sequence_train.seal.json" \
  --sequence-output-schedule "${OUTPUT_ROOT}/qwen_mc_sequence_train.schedule.jsonl" \
  --sequence-output-manifest "${OUTPUT_ROOT}/qwen_mc_sequence_train.build.json" \
  --cot-output-jsonl "${OUTPUT_ROOT}/qwen_cot_sft_train.jsonl" \
  --cot-output-seal "${OUTPUT_ROOT}/qwen_cot_sft_train.seal.json" \
  --cot-output-schedule "${OUTPUT_ROOT}/qwen_cot_sft_train.schedule.jsonl" \
  --cot-output-manifest "${OUTPUT_ROOT}/qwen_cot_sft_train.build.json" \
  --seed 44

printf 'QWEN38_UNION2776_COMPLETE tasks=2776 sequence_rows=22208 cot_rows=5552 output=%s heldout_intersection=0\n' \
  "${OUTPUT_ROOT}"
