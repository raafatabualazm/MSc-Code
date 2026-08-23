#!/usr/bin/env bash
# Normalize the expanded builder output into exact fit2776/supplement1196 views.
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
PATCH_ROOT="${PATCH_ROOT:-${WORKSPACE}/hybrid_training_patch_v2_3}"
LEGACY_BUILD_ROOT="${LEGACY_BUILD_ROOT:-${WORKSPACE}/multifunction_v1/build}"
EXPANSION_ROOT="${EXPANSION_ROOT:-${WORKSPACE}/multifunction_v1/expanded2776}"
EXPANSION_BUILD="${EXPANSION_BUILD:-${EXPANSION_ROOT}/build}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${WORKSPACE}/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"

CANDIDATE_COMPACT="${CANDIDATE_COMPACT:-${EXPANSION_BUILD}/train_multifunction_binary_expanded_2776.jsonl}"
CANDIDATE_SEAL="${CANDIDATE_SEAL:-${EXPANSION_BUILD}/train_multifunction_binary_expanded_2776.seal.json}"
CANDIDATE_PROMPT="${CANDIDATE_PROMPT:-${EXPANSION_BUILD}/train_multifunction_binary_expanded_2776_f2.jsonl}"
CANDIDATE_PROMPT_MANIFEST="${CANDIDATE_PROMPT_MANIFEST:-${CANDIDATE_PROMPT}.manifest.json}"
LEGACY_COMPACT="${LEGACY_COMPACT:-${LEGACY_BUILD_ROOT}/train_multifunction_binary.jsonl}"
LEGACY_SEAL="${LEGACY_SEAL:-${LEGACY_BUILD_ROOT}/train_multifunction_binary_target24k.seal.json}"
LEGACY_PROMPT="${LEGACY_PROMPT:-${LEGACY_BUILD_ROOT}/train_multifunction_binary_f2.jsonl}"
LEGACY_PROMPT_MANIFEST="${LEGACY_PROMPT_MANIFEST:-${LEGACY_PROMPT}.manifest.json}"
HELDOUT="${HELDOUT:-${LEGACY_BUILD_ROOT}/dev_multifunction_binary.jsonl}"
HELDOUT_SEAL="${HELDOUT_SEAL:-${LEGACY_BUILD_ROOT}/dev_multifunction_binary.seal.json}"
CANDIDATE_CONTRACT="${CANDIDATE_CONTRACT:-${LEGACY_BUILD_ROOT}/multifunction_inline_cfg_v2_contract.json}"
TARGET_CONTRACT="${TARGET_CONTRACT:-${LEGACY_BUILD_ROOT}/multifunction_inline_cfg_v2_target24k_contract.json}"

mkdir -p "${WORKSPACE}/locks" "${EXPANSION_ROOT}"
exec 9>"${WORKSPACE}/locks/prepare_qwen2776_supplement.lock"
if ! flock -n 9; then
  printf 'Another Qwen 2,776-task supplement preparation holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[prepare_qwen2776_supplement] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

sha256_of() {
  local value
  value="$(sha256sum "$1")"
  printf '%s\n' "${value%% *}"
}

for required in \
  "${EXPANSION_BUILD}/expansion_build.seal.json" \
  "${EXPANSION_BUILD}/build_report.json" \
  "${CANDIDATE_COMPACT}" "${CANDIDATE_SEAL}" \
  "${CANDIDATE_PROMPT}" "${CANDIDATE_PROMPT_MANIFEST}" \
  "${LEGACY_COMPACT}" "${LEGACY_SEAL}" \
  "${LEGACY_PROMPT}" "${LEGACY_PROMPT_MANIFEST}" \
  "${HELDOUT}" "${HELDOUT_SEAL}" \
  "${CANDIDATE_CONTRACT}" "${TARGET_CONTRACT}" \
  "${TOKENIZER_JSON}"; do
  test -f "${required}" || {
    printf 'Required supplement-preparation input is missing: %s\n' \
      "${required}" >&2
    exit 2
  }
done

cd "${PATCH_ROOT}"
"${PYTHON}" -m scripts.training.prepare_qwen_2776_supplement \
  --candidate-compact-jsonl "${CANDIDATE_COMPACT}" \
  --expected-candidate-compact-sha256 "$(sha256_of "${CANDIDATE_COMPACT}")" \
  --candidate-compact-seal "${CANDIDATE_SEAL}" \
  --expected-candidate-compact-seal-sha256 "$(sha256_of "${CANDIDATE_SEAL}")" \
  --candidate-prompt-jsonl "${CANDIDATE_PROMPT}" \
  --expected-candidate-prompt-sha256 "$(sha256_of "${CANDIDATE_PROMPT}")" \
  --candidate-prompt-manifest "${CANDIDATE_PROMPT_MANIFEST}" \
  --expected-candidate-prompt-manifest-sha256 \
    "$(sha256_of "${CANDIDATE_PROMPT_MANIFEST}")" \
  --legacy-compact-jsonl "${LEGACY_COMPACT}" \
  --expected-legacy-compact-sha256 "$(sha256_of "${LEGACY_COMPACT}")" \
  --legacy-compact-seal "${LEGACY_SEAL}" \
  --expected-legacy-compact-seal-sha256 "$(sha256_of "${LEGACY_SEAL}")" \
  --legacy-prompt-jsonl "${LEGACY_PROMPT}" \
  --expected-legacy-prompt-sha256 "$(sha256_of "${LEGACY_PROMPT}")" \
  --legacy-prompt-manifest "${LEGACY_PROMPT_MANIFEST}" \
  --expected-legacy-prompt-manifest-sha256 \
    "$(sha256_of "${LEGACY_PROMPT_MANIFEST}")" \
  --heldout-jsonl "${HELDOUT}" \
  --expected-heldout-sha256 "$(sha256_of "${HELDOUT}")" \
  --heldout-seal "${HELDOUT_SEAL}" \
  --expected-heldout-seal-sha256 "$(sha256_of "${HELDOUT_SEAL}")" \
  --expansion-build-seal "${EXPANSION_BUILD}/expansion_build.seal.json" \
  --expected-expansion-build-seal-sha256 \
    "$(sha256_of "${EXPANSION_BUILD}/expansion_build.seal.json")" \
  --expansion-build-report "${EXPANSION_BUILD}/build_report.json" \
  --expected-expansion-build-report-sha256 \
    "$(sha256_of "${EXPANSION_BUILD}/build_report.json")" \
  --candidate-contract "${CANDIDATE_CONTRACT}" \
  --expected-candidate-contract-sha256 \
    "$(sha256_of "${CANDIDATE_CONTRACT}")" \
  --contract "${TARGET_CONTRACT}" \
  --expected-contract-sha256 "$(sha256_of "${TARGET_CONTRACT}")" \
  --student-tokenizer-json "${TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "$(sha256_of "${TOKENIZER_JSON}")" \
  --fit-compact-output \
    "${EXPANSION_ROOT}/fit2776_multifunction_binary.jsonl" \
  --fit-compact-seal-output \
    "${EXPANSION_ROOT}/fit2776_multifunction_binary.target24k.seal.json" \
  --fit-prompt-output \
    "${EXPANSION_ROOT}/fit2776_multifunction_binary_f2.jsonl" \
  --fit-prompt-manifest-output \
    "${EXPANSION_ROOT}/fit2776_multifunction_binary_f2.jsonl.manifest.json" \
  --supplement-compact-output \
    "${EXPANSION_ROOT}/supplement1196_multifunction_binary.jsonl" \
  --supplement-compact-seal-output \
    "${EXPANSION_ROOT}/supplement1196_multifunction_binary.target24k.seal.json" \
  --supplement-prompt-output \
    "${EXPANSION_ROOT}/supplement1196_multifunction_binary_f2.jsonl" \
  --supplement-prompt-manifest-output \
    "${EXPANSION_ROOT}/supplement1196_multifunction_binary_f2.jsonl.manifest.json" \
  --derivation-manifest-output \
    "${EXPANSION_ROOT}/qwen_2776_supplement.derivation.json"

printf 'QWEN2776_SUPPLEMENT_PREPARED fit=2776 legacy=1580 supplement=1196 heldout=175 root=%s\n' \
  "${EXPANSION_ROOT}"
