#!/usr/bin/env bash
# Collect only the missing 1,196 Qwen K=8 tasks and build parent artifacts.
#
# This launcher never reads, writes, copies, or resumes the live 1,580-task
# journal.  Its output root, lock, pilot, journal, and chain head are separate.
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
PATCH_ROOT="${PATCH_ROOT:-${WORKSPACE}/hybrid_training_patch_v2_3}"
EXPANDED_ROOT="${EXPANDED_ROOT:-${WORKSPACE}/multifunction_v1/expanded2776}"
LEGACY_BUILD_ROOT="${LEGACY_BUILD_ROOT:-${WORKSPACE}/multifunction_v1/build}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/artifacts/direct_compact_qwen38_supplement1196}"
QWEN_ENV_FILE="${QWEN_ENV_FILE:-${WORKSPACE}/Qwen.env}"
TOKENIZER_SNAPSHOT="${TOKENIZER_SNAPSHOT:-${WORKSPACE}/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"

DERIVATION="${DERIVATION:-${EXPANDED_ROOT}/qwen_2776_supplement.derivation.json}"
SUPPLEMENT_COMPACT="${SUPPLEMENT_COMPACT:-${EXPANDED_ROOT}/supplement1196_multifunction_binary.jsonl}"
SUPPLEMENT_SEAL="${SUPPLEMENT_SEAL:-${EXPANDED_ROOT}/supplement1196_multifunction_binary.target24k.seal.json}"
SUPPLEMENT_PROMPTS="${SUPPLEMENT_PROMPTS:-${EXPANDED_ROOT}/supplement1196_multifunction_binary_f2.jsonl}"
SUPPLEMENT_PROMPT_MANIFEST="${SUPPLEMENT_PROMPT_MANIFEST:-${SUPPLEMENT_PROMPTS}.manifest.json}"
CONTRACT="${CONTRACT:-${LEGACY_BUILD_ROOT}/multifunction_inline_cfg_v2_target24k_contract.json}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${TOKENIZER_SNAPSHOT}/tokenizer.json}"

mkdir -p "${WORKSPACE}/locks" "${OUTPUT_ROOT}"
exec 9>"${WORKSPACE}/locks/qwen38_supplement1196.lock"
if ! flock -n 9; then
  printf 'Another Qwen supplemental harvest holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[qwen38_supplement1196] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

sha256_of() {
  local value
  value="$(sha256sum "$1")"
  printf '%s\n' "${value%% *}"
}

require_manifest_output() {
  local key="$1"
  local path="$2"
  test -f "${path}" || {
    printf 'Missing supplemental input: %s\n' "${path}" >&2
    exit 2
  }
  local expected actual
  expected="$(jq -er --arg key "${key}" '.outputs[$key].sha256' "${DERIVATION}")"
  actual="$(sha256_of "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'Supplemental input %s hash mismatch: expected %s got %s\n' \
      "${key}" "${expected}" "${actual}" >&2
    exit 2
  fi
}

test -f "${DERIVATION}" || {
  printf 'Missing sealed 2,776-task derivation: %s\n' "${DERIVATION}" >&2
  exit 2
}
jq -e '
  .schema == "qwen-2776-supplement-derivation-v1"
  and .fit_scope == "phase0_train_minus_heldout175"
  and .counts.fit_tasks == 2776
  and .counts.legacy_parent_tasks == 1580
  and .counts.supplement_tasks == 1196
  and .counts.heldout_tasks == 175
  and .counts.samples_per_task == 8
  and .counts.supplement_teacher_slots == 9568
  and .counts.union_teacher_slots == 22208
  and .heldout_intersection_count == 0
  and .set_equations.fit_equals_legacy_disjoint_union_supplement == true
  and .set_equations.legacy_supplement_intersection_count == 0
  and .live_parent_compatibility.shared_tasks == 1580
  and .live_parent_compatibility.all_compact_ids_byte_identical == true
  and .live_parent_compatibility.all_api_prompt_text_byte_identical == true
  and .invariants.live_journal_read == false
  and .invariants.live_journal_modified == false
  and .invariants.heldout_used_for_teacher_collection == false
  and .invariants.heldout_used_for_fit == false
' "${DERIVATION}" >/dev/null
require_manifest_output "supplement_compact" "${SUPPLEMENT_COMPACT}"
require_manifest_output "supplement_compact_seal" "${SUPPLEMENT_SEAL}"
require_manifest_output "supplement_prompt" "${SUPPLEMENT_PROMPTS}"
require_manifest_output \
  "supplement_prompt_manifest" "${SUPPLEMENT_PROMPT_MANIFEST}"
EXPECTED_CONTRACT_SHA256="$(
  jq -er '.inputs.contract.sha256' "${DERIVATION}"
)"
if [[ "$(sha256_of "${CONTRACT}")" != "${EXPECTED_CONTRACT_SHA256}" ]]; then
  printf 'Supplemental target24k contract hash mismatch\n' >&2
  exit 2
fi
EXPECTED_TOKENIZER_SHA256="$(
  jq -er '.inputs.student_tokenizer.sha256' "${DERIVATION}"
)"
if [[ "$(sha256_of "${TOKENIZER_JSON}")" != "${EXPECTED_TOKENIZER_SHA256}" ]]; then
  printf 'Supplemental tokenizer hash mismatch\n' >&2
  exit 2
fi

# Parse credentials as inert KEY=VALUE data.  Never source this file.
declare -A qwen_env_seen=()
while IFS= read -r qwen_env_line || [[ -n "${qwen_env_line}" ]]; do
  qwen_env_line="${qwen_env_line%$'\r'}"
  if [[ -z "${qwen_env_line}" || "${qwen_env_line}" == \#* ]]; then
    continue
  fi
  if [[ ! "${qwen_env_line}" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
    printf 'Qwen.env contains a non-KEY=VALUE line\n' >&2
    exit 2
  fi
  qwen_env_key="${BASH_REMATCH[1]}"
  qwen_env_value="${BASH_REMATCH[2]}"
  case "${qwen_env_key}" in
    API_KEY|QWEN_API_KEY|DASHSCOPE_ENDPOINT|QWEN_BASE_URL|QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED)
      ;;
    *)
      printf 'Qwen.env contains disallowed key: %s\n' "${qwen_env_key}" >&2
      exit 2
      ;;
  esac
  if [[ -n "${qwen_env_seen[${qwen_env_key}]:-}" ]]; then
    printf 'Qwen.env contains duplicate key: %s\n' "${qwen_env_key}" >&2
    exit 2
  fi
  qwen_env_seen["${qwen_env_key}"]=1
  if [[ "${qwen_env_value}" == *'$('* \
     || "${qwen_env_value}" == *'`'* \
     || "${qwen_env_value}" == *';'* ]]; then
    printf 'Qwen.env value for %s contains shell syntax\n' "${qwen_env_key}" >&2
    exit 2
  fi
  printf -v "${qwen_env_key}" '%s' "${qwen_env_value}"
  export "${qwen_env_key}"
done < "${QWEN_ENV_FILE}"
if [[ -z "${QWEN_API_KEY:-}" && -n "${API_KEY:-}" ]]; then
  export QWEN_API_KEY="${API_KEY}"
fi
if [[ -z "${QWEN_API_KEY:-}" ]]; then
  printf 'QWEN_API_KEY is not set\n' >&2
  exit 2
fi
case "${QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED:-0}" in
  1|true|TRUE|yes|YES) ;;
  *)
    printf 'Supplemental automation requires explicit Token Plan authorization\n' >&2
    exit 2
    ;;
esac
QWEN_BASE_URL="${QWEN_BASE_URL:-${DASHSCOPE_ENDPOINT:-}}"
test -n "${QWEN_BASE_URL}" || {
  printf 'QWEN_BASE_URL or DASHSCOPE_ENDPOINT is required\n' >&2
  exit 2
}

JOURNAL="${OUTPUT_ROOT}/qwen_teacher.journal.jsonl"
PARSEABLE="${OUTPUT_ROOT}/qwen_teacher.parseable.mc_sequence.jsonl"
VERIFIED_RS="${OUTPUT_ROOT}/qwen_teacher.verified_only.rs_sft.jsonl"
AUDIT="${OUTPUT_ROOT}/qwen_teacher.audit.json"
PILOT_ROOT="${OUTPUT_ROOT}/quality_pilot"
PILOT_JOURNAL="${PILOT_ROOT}/qwen_teacher.pilot.journal.jsonl"
PILOT_SEQUENCE="${PILOT_ROOT}/qwen_teacher.pilot.sequence.jsonl"
PILOT_VERIFIED="${PILOT_ROOT}/qwen_teacher.pilot.verified_only.jsonl"
PILOT_AUDIT="${PILOT_ROOT}/qwen_teacher.pilot.audit.json"
PILOT_GATE="${PILOT_ROOT}/qwen_teacher.pilot.quality_gate.json"
SEQUENCE_JSONL="${OUTPUT_ROOT}/qwen_mc_sequence_train.jsonl"
SEQUENCE_SEAL="${OUTPUT_ROOT}/qwen_mc_sequence_train.seal.json"
SEQUENCE_SCHEDULE="${OUTPUT_ROOT}/qwen_mc_sequence_train.schedule.jsonl"
SEQUENCE_BUILD="${OUTPUT_ROOT}/qwen_mc_sequence_train.build.json"
COT_JSONL="${OUTPUT_ROOT}/qwen_cot_sft_train.jsonl"
COT_SEAL="${OUTPUT_ROOT}/qwen_cot_sft_train.seal.json"
COT_SCHEDULE="${OUTPUT_ROOT}/qwen_cot_sft_train.schedule.jsonl"
COT_BUILD="${OUTPUT_ROOT}/qwen_cot_sft_train.build.json"
mkdir -p "${PILOT_ROOT}"

PROMPT_SHA256="$(sha256_of "${SUPPLEMENT_PROMPTS}")"
PROMPT_MANIFEST_SHA256="$(sha256_of "${SUPPLEMENT_PROMPT_MANIFEST}")"
VERIFIER_SHA256="$(sha256_of "${SUPPLEMENT_COMPACT}")"
CONTRACT_SHA256="$(sha256_of "${CONTRACT}")"

cd "${PATCH_ROOT}"

COMMON_ARGS=(
  --token-plan-automation-authorized
  --authorize-orphan-reissue-with-duplicate-billing-risk
  --prompt-jsonl "${SUPPLEMENT_PROMPTS}"
  --expected-prompt-sha256 "${PROMPT_SHA256}"
  --expected-prompt-rows 1196
  --prompt-manifest "${SUPPLEMENT_PROMPT_MANIFEST}"
  --expected-prompt-manifest-sha256 "${PROMPT_MANIFEST_SHA256}"
  --verifier-jsonl "${SUPPLEMENT_COMPACT}"
  --expected-verifier-sha256 "${VERIFIER_SHA256}"
  --student-tokenizer-json "${TOKENIZER_JSON}"
  --expected-student-tokenizer-sha256 "${EXPECTED_TOKENIZER_SHA256}"
  --student-eos-token-id 151645
  --target-contract "${CONTRACT}"
  --expected-target-contract-sha256 "${CONTRACT_SHA256}"
  --model qwen3.8-max-preview
  --objective-mode sequence_only
  --enable-thinking
  --required-function fn0
  --base-url "${QWEN_BASE_URL}"
  --api-key-env QWEN_API_KEY
  --temperature 1.0
  --top-p 1.0
  --top-k 101
  --max-tokens 12288
  --length-max-token-escalation 16384 24576
  --thinking-budget 8192
  --seed-base "${TEACHER_SEED_BASE:-44}"
  --max-prompt-tokens 12000
  --chat-overhead-reserve 256
  --timeout-seconds "${TEACHER_TIMEOUT_SECONDS:-600}"
  --verifier-timeout-seconds "${VERIFIER_TIMEOUT_SECONDS:-45}"
  --max-retries "${TEACHER_MAX_RETRIES:-8}"
  --workers "${QWEN_TEACHER_WORKERS:-16}"
  --verifier-workers "${QWEN_VERIFIER_WORKERS:-16}"
  --progress-every "${QWEN_PROGRESS_EVERY:-50}"
)

"${PYTHON}" -m scripts.training.collect_qwen_direct_compact_teacher \
  "${COMMON_ARGS[@]}" \
  --journal "${JOURNAL}" \
  --parseable-output "${PARSEABLE}" \
  --rs-sft-output "${VERIFIED_RS}" \
  --audit-output "${AUDIT}" \
  --dry-run

# A fresh, genuine deterministic 16-task pilot is intentionally used for the
# new task universe. It costs 128 extra draws and avoids weakening or forging
# the collector's quality-gate contract. On production-journal recovery, the
# exact gate sealed into the run header must be reused: re-materializing the
# pilot changes its audit creation timestamp and therefore its file hash.
if [[ -s "${JOURNAL}" ]]; then
  PILOT_GATE_SHA256="$(
    "${PYTHON}" - "${JOURNAL}" "${PILOT_GATE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

journal = Path(sys.argv[1]).resolve()
gate = Path(sys.argv[2]).resolve()
with journal.open("r", encoding="utf-8") as handle:
    header = json.loads(next(handle))
record = (header.get("payload") or {}).get("pilot_quality_gate") or {}
expected_path = str(record.get("path") or "")
expected_sha256 = str(record.get("sha256") or "")
expected_size = int(record.get("size_bytes", -1))
payload = gate.read_bytes()
if expected_path != str(gate):
    raise SystemExit("supplement production journal pilot-gate path differs")
if len(payload) != expected_size:
    raise SystemExit("supplement production journal pilot-gate size differs")
if hashlib.sha256(payload).hexdigest() != expected_sha256:
    raise SystemExit("supplement production journal pilot-gate hash differs")
print(expected_sha256)
PY
  )"
  printf 'QWEN_SUPPLEMENT_PILOT_REUSE gate_sha256=%s reason=production_journal_resume\n' \
    "${PILOT_GATE_SHA256}"
else
  "${PYTHON}" -m scripts.training.collect_qwen_direct_compact_teacher \
    "${COMMON_ARGS[@]}" \
    --journal "${PILOT_JOURNAL}" \
    --parseable-output "${PILOT_SEQUENCE}" \
    --rs-sft-output "${PILOT_VERIFIED}" \
    --audit-output "${PILOT_AUDIT}" \
    --max-tasks 16 \
    --task-selection-strategy deterministic_hash

  "${PYTHON}" -m scripts.training.build_qwen_quality_gate \
    --pilot-audit "${PILOT_AUDIT}" \
    --pilot-verified-only "${PILOT_VERIFIED}" \
    --output "${PILOT_GATE}" \
    --pilot-tasks 16 \
    --minimum-verified-tasks "${QWEN_PILOT_MIN_VERIFIED_TASKS:-0}" \
    --minimum-parseable-fraction \
      "${QWEN_PILOT_MIN_PARSEABLE_FRACTION:-0.50}"
  PILOT_GATE_SHA256="$(sha256_of "${PILOT_GATE}")"
fi

"${PYTHON}" -m scripts.training.collect_qwen_direct_compact_teacher \
  "${COMMON_ARGS[@]}" \
  --journal "${JOURNAL}" \
  --parseable-output "${PARSEABLE}" \
  --rs-sft-output "${VERIFIED_RS}" \
  --audit-output "${AUDIT}" \
  --quality-gate-json "${PILOT_GATE}" \
  --expected-quality-gate-sha256 "${PILOT_GATE_SHA256}"

# Re-materialize once from the durable journal so all derivative hashes bind
# the final journal chain head.
"${PYTHON}" -m scripts.evaluation.audit_qwen_direct_compact_teacher \
  --journal "${JOURNAL}" \
  --student-tokenizer-json "${TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "${EXPECTED_TOKENIZER_SHA256}" \
  --student-eos-token-id 151645 \
  --parseable-output "${PARSEABLE}" \
  --rs-sft-output "${VERIFIED_RS}" \
  --audit-output "${AUDIT}"

"${PYTHON}" -m scripts.training.build_qwen_sequence_kd \
  --compact-train-jsonl "${SUPPLEMENT_COMPACT}" \
  --compact-train-seal "${SUPPLEMENT_SEAL}" \
  --contract "${CONTRACT}" \
  --prompt-jsonl "${SUPPLEMENT_PROMPTS}" \
  --expected-prompt-sha256 "${PROMPT_SHA256}" \
  --teacher-parseable-jsonl "${PARSEABLE}" \
  --expected-teacher-parseable-sha256 "$(sha256_of "${PARSEABLE}")" \
  --teacher-journal "${JOURNAL}" \
  --expected-teacher-journal-sha256 "$(sha256_of "${JOURNAL}")" \
  --teacher-audit-json "${AUDIT}" \
  --expected-teacher-audit-sha256 "$(sha256_of "${AUDIT}")" \
  --student-tokenizer-json "${TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "${EXPECTED_TOKENIZER_SHA256}" \
  --output-jsonl "${SEQUENCE_JSONL}" \
  --output-seal "${SEQUENCE_SEAL}" \
  --schedule-output "${SEQUENCE_SCHEDULE}" \
  --build-manifest "${SEQUENCE_BUILD}" \
  --gold-replay-fraction 0.0 \
  --seed 44

"${PYTHON}" -m scripts.training.build_qwen_cot_sft \
  --compact-train-jsonl "${SUPPLEMENT_COMPACT}" \
  --compact-train-seal "${SUPPLEMENT_SEAL}" \
  --contract "${CONTRACT}" \
  --prompt-jsonl "${SUPPLEMENT_PROMPTS}" \
  --expected-prompt-sha256 "${PROMPT_SHA256}" \
  --teacher-journal "${JOURNAL}" \
  --expected-teacher-journal-sha256 "$(sha256_of "${JOURNAL}")" \
  --teacher-audit-json "${AUDIT}" \
  --expected-teacher-audit-sha256 "$(sha256_of "${AUDIT}")" \
  --student-tokenizer-json "${TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "${EXPECTED_TOKENIZER_SHA256}" \
  --output-jsonl "${COT_JSONL}" \
  --output-seal "${COT_SEAL}" \
  --schedule-output "${COT_SCHEDULE}" \
  --build-manifest "${COT_BUILD}"

printf 'QWEN38_SUPPLEMENT_COMPLETE tasks=1196 K=8 draws=9568 sequence=%s cot=%s live_journal_modified=false\n' \
  "${SEQUENCE_BUILD}" "${COT_BUILD}"
