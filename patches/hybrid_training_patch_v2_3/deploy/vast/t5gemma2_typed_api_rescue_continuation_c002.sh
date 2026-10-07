#!/usr/bin/env bash
set -euo pipefail

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
PHASE="${T5GEMMA_TYPED_API_PHASE:-}"
COHORT_INDEX="${T5GEMMA_TYPED_API_COHORT_INDEX:-}"
OUTPUT_DIR="${T5GEMMA_TYPED_API_OUTPUT_DIR:-}"
PLAN_ONLY_OUTPUT="${T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT:-}"
FIXED_KIMI_COHORT_LIMIT="${T5GEMMA_TYPED_API_FIXED_KIMI_COHORT_LIMIT:-}"
MAX_INPUT_PER_CALL="${T5GEMMA_TYPED_API_MAX_INPUT_TOKENS_PER_CALL:-}"
INITIAL_MAX_OUTPUT="${T5GEMMA_TYPED_API_INITIAL_MAX_OUTPUT_TOKENS:-}"

blocked() {
  echo "T5GEMMA_TYPED_API_C002_BLOCKED $*" >&2
  exit 78
}

if [[ "${PHASE}" != kimi_initial && "${PHASE}" != kimi_retry ]]; then
  blocked "only Kimi initial/retry phases are permitted"
fi
[[ "${COHORT_INDEX}" == 2 ]] || blocked "cohort index must be 2"
[[ "${FIXED_KIMI_COHORT_LIMIT}" == 3 ]] || blocked "fixed cohort limit must be 3"
[[ "${MAX_INPUT_PER_CALL}" == 16384 ]] || blocked "input cap must be 16384"
[[ -n "${OUTPUT_DIR}" ]] || blocked "output directory is absent"
[[ -x "${DART_BIN}" ]] || blocked "Dart 3.12.2 is absent"
printf '%s  %s\n' \
  7a03af003e998497012706361f5cbf0734d8defa82c7e458aa5f87f796e01143 "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_cascade.py" \
  7a29289f6f07caca03df73b7956ffb1782a0c2ec250cc4d5793eedc73e0d910f "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_continuation_c002.py" \
  0cc6323136e12110e6ee25ef42f8e0baec90359ef7cc972fb191691a12fd7b15 "${PROJECT}/scripts/preprocessing/build_t5gemma2_typed_api_visible_split.py" \
  b7f66f7ee0b1959fe6a6b8bbe6fa422d545ab950971dd6aae86bfb31acde0f88 "${PROJECT}/scripts/training/t5gemma2_typed_visible_failure_projection.py" \
  | sha256sum -c - || blocked "cohort-2 producer code differs"

pin_names=(
  T5GEMMA_TYPED_LOCAL_REPORT_SHA256
  T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256
  T5GEMMA_TYPED_LOCAL_TARGETS_SHA256
  T5GEMMA_TYPED_225_MANIFEST_SHA256
  T5GEMMA_TYPED_API_SPLIT_MANIFEST_SHA256
  T5GEMMA_TYPED_API_VISIBLE_SHA256
  T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256
  T5GEMMA_TYPED_API_PROJECTION_REPORT_SHA256
  T5GEMMA_TYPED_API_PROJECTION_JOURNAL_SHA256
)
for name in "${pin_names[@]}"; do
  [[ "${!name:-}" =~ ^[0-9a-f]{64}$ ]] || blocked "${name} is not pinned"
done
if [[ -z "${PLAN_ONLY_OUTPUT}" ]]; then
  [[ "${T5GEMMA_TYPED_API_SCHEDULE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
    || blocked "executed schedule is not pinned"
fi

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  "${T5GEMMA_TYPED_LOCAL_REPORT_SHA256}" "${LOCAL_DIR}/harvest_report.json" \
  "${T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256}" "${LOCAL_DIR}/harvest.journal.jsonl" \
  "${T5GEMMA_TYPED_LOCAL_TARGETS_SHA256}" "${LOCAL_DIR}/direct_targets.jsonl" \
  "${T5GEMMA_TYPED_225_MANIFEST_SHA256}" "${EXISTING_MANIFEST}" \
  "${T5GEMMA_TYPED_API_SPLIT_MANIFEST_SHA256}" "${SPLIT_DIR}/split_manifest.json" \
  "${T5GEMMA_TYPED_API_VISIBLE_SHA256}" "${SPLIT_DIR}/visible_train.jsonl" \
  "${T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256}" "${SPLIT_DIR}/holdback.private.jsonl" \
  "${T5GEMMA_TYPED_API_PROJECTION_REPORT_SHA256}" "${PROJECTION_DIR}/visible_projection_report.json" \
  "${T5GEMMA_TYPED_API_PROJECTION_JOURNAL_SHA256}" "${PROJECTION_DIR}/visible_projection.journal.jsonl" \
  | sha256sum -c - || blocked "input evidence differs"

prior_args=()
if [[ "${PHASE}" == kimi_initial ]]; then
  MAX_TASKS=50
  MAX_CALLS=50
  if [[ "${INITIAL_MAX_OUTPUT}" == 4096 || "${INITIAL_MAX_OUTPUT}" == 2048 ]]; then
    MAX_OUTPUT="${INITIAL_MAX_OUTPUT}"
  else
    blocked "initial output cap must be sealed to 4096 or 2048"
  fi
  PRIOR_INDEX="${T5GEMMA_TYPED_API_PRIOR_INDEX:-}"
  PRIOR_INDEX_SHA="${T5GEMMA_TYPED_API_PRIOR_INDEX_SHA256:-}"
  [[ -s "${PRIOR_INDEX}" ]] || blocked "prior report index is absent"
  [[ "${PRIOR_INDEX_SHA}" =~ ^[0-9a-f]{64}$ ]] || blocked "prior index pin is absent"
  printf '%s  %s\n' "${PRIOR_INDEX_SHA}" "${PRIOR_INDEX}" | sha256sum -c - \
    || blocked "prior report index differs"
  while IFS=$'\t' read -r digest path; do
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || blocked "prior report digest is malformed"
    [[ -s "${path}" ]] || blocked "prior report is absent"
    printf '%s  %s\n' "${digest}" "${path}" | sha256sum -c - \
      || blocked "prior report differs"
    prior_args+=(--prior_success_report "${path}" --expected_prior_success_report_sha256 "${digest}")
  done <"${PRIOR_INDEX}"
else
  [[ "${INITIAL_MAX_OUTPUT}" == 8192 ]] || blocked "retry output cap must be 8192"
  SOURCE_REPORT="${T5GEMMA_TYPED_API_RETRY_SOURCE_REPORT:-}"
  SOURCE_SHA="${T5GEMMA_TYPED_API_RETRY_SOURCE_SHA256:-}"
  RETRY_TASKS="${T5GEMMA_TYPED_API_RETRY_TASKS:-}"
  RETRY_IDS_SHA="${T5GEMMA_TYPED_API_RETRY_IDS_SHA256:-}"
  [[ -s "${SOURCE_REPORT}" ]] || blocked "retry source is absent"
  [[ "${SOURCE_SHA}" =~ ^[0-9a-f]{64}$ ]] || blocked "retry source pin is absent"
  [[ "${RETRY_TASKS}" =~ ^[1-9][0-9]*$ ]] || blocked "retry count is invalid"
  [[ "${RETRY_IDS_SHA}" =~ ^[0-9a-f]{64}$ ]] || blocked "retry identity pin is absent"
  printf '%s  %s\n' "${SOURCE_SHA}" "${SOURCE_REPORT}" | sha256sum -c - \
    || blocked "retry source differs"
  prior_args+=(
    --retry_parse_failures_or_truncations_report "${SOURCE_REPORT}"
    --expected_retry_parse_failures_or_truncations_report_sha256 "${SOURCE_SHA}"
    --expected_retry_parse_failures_or_truncations_tasks "${RETRY_TASKS}"
    --expected_retry_parse_failures_or_truncations_task_ids_sha256 "${RETRY_IDS_SHA}"
  )
  MAX_TASKS="${RETRY_TASKS}"
  MAX_CALLS="${RETRY_TASKS}"
  MAX_OUTPUT=8192
fi

MAX_INPUT_TOTAL=$((MAX_INPUT_PER_CALL * MAX_TASKS))
MAX_OUTPUT_TOTAL=$((MAX_OUTPUT * MAX_TASKS))
MAX_TOTAL=$((MAX_INPUT_TOTAL + MAX_OUTPUT_TOTAL))
MAX_USD="${T5GEMMA_TYPED_API_MAX_USD:-}"
[[ "${MAX_USD}" =~ ^[0-9]+([.][0-9]+)?$ ]] || blocked "phase budget is invalid"

SECRET_FILE="${T5GEMMA_OPENROUTER_ENV:-${WORKSPACE}/secrets/Openrouter.env}"
if [[ -z "${PLAN_ONLY_OUTPUT}" ]]; then
  [[ -s "${SECRET_FILE}" ]] || blocked "OpenRouter secret file is absent"
  provider_key="$(/venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import re, stat, sys
from pathlib import Path
p=Path(sys.argv[1])
if stat.S_IMODE(p.stat().st_mode) & 0o077:
    raise SystemExit("provider secret file must be mode 0600")
raw=p.read_bytes()
try: text=raw.decode("utf-8-sig")
except UnicodeDecodeError: text=raw.decode("utf-16")
values=[]
for line in text.splitlines():
    line=line.strip()
    if not line or line.startswith("#"): continue
    m=re.fullmatch(r"(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*)", line)
    if m:
        value=m.group(1).strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in "\"'": value=value[1:-1]
        values.append(value)
if len(values)!=1 or not values[0] or any(c.isspace() for c in values[0]):
    raise SystemExit("OPENROUTER_API_KEY must occur exactly once")
print(values[0], end="")
PY
)"
  export OPENROUTER_API_KEY="${provider_key}"
  unset provider_key
fi

schedule_args=()
plan_args=()
if [[ -n "${PLAN_ONLY_OUTPUT}" ]]; then
  plan_args+=(--plan_only_output "${PLAN_ONLY_OUTPUT}")
else
  schedule_args+=(--expected_scheduled_task_ids_sha256 "${T5GEMMA_TYPED_API_SCHEDULE_SHA256}")
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_typed_api_rescue_continuation_c002.py \
  --phase "${PHASE}" --cohort_index 2 --fixed_kimi_cohort_limit 3 \
  --local_harvest_report "${LOCAL_DIR}/harvest_report.json" \
  --expected_local_harvest_report_sha256 "${T5GEMMA_TYPED_LOCAL_REPORT_SHA256}" \
  --pilot_journal "${LOCAL_DIR}/harvest.journal.jsonl" \
  --expected_local_harvest_journal_sha256 "${T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256}" \
  --local_harvest_targets "${LOCAL_DIR}/direct_targets.jsonl" \
  --expected_local_harvest_targets_sha256 "${T5GEMMA_TYPED_LOCAL_TARGETS_SHA256}" \
  --existing_direct_manifest "${EXISTING_MANIFEST}" \
  --expected_existing_direct_manifest_sha256 "${T5GEMMA_TYPED_225_MANIFEST_SHA256}" \
  --gold_train_jsonl "${GOLD_TRAIN}" --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --gold_f2_jsonl "${GOLD_F2}" --f2_jsonl "${GOLD_F2}" \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --heldout_jsonl "${HELDOUT}" --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --visible_split_manifest "${SPLIT_DIR}/split_manifest.json" \
  --expected_visible_split_manifest_sha256 "${T5GEMMA_TYPED_API_SPLIT_MANIFEST_SHA256}" \
  --visible_train "${SPLIT_DIR}/visible_train.jsonl" --rollout_file "${SPLIT_DIR}/visible_train.jsonl" \
  --expected_visible_train_sha256 "${T5GEMMA_TYPED_API_VISIBLE_SHA256}" --expected_rollout_sha256 "${T5GEMMA_TYPED_API_VISIBLE_SHA256}" \
  --private_split_holdback "${SPLIT_DIR}/holdback.private.jsonl" --private_holdback "${SPLIT_DIR}/holdback.private.jsonl" \
  --expected_private_split_holdback_sha256 "${T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256}" \
  --expected_private_holdback_sha256 "${T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256}" \
  --visible_projection_report "${PROJECTION_DIR}/visible_projection_report.json" \
  --expected_visible_projection_report_sha256 "${T5GEMMA_TYPED_API_PROJECTION_REPORT_SHA256}" \
  --visible_projection_journal "${PROJECTION_DIR}/visible_projection.journal.jsonl" \
  --expected_visible_projection_journal_sha256 "${T5GEMMA_TYPED_API_PROJECTION_JOURNAL_SHA256}" \
  --output_dir "${OUTPUT_DIR}" "${prior_args[@]}" \
  --provider openrouter_chat --model moonshotai/kimi-k3 \
  --base_url https://openrouter.ai/api/v1 --api_key_env OPENROUTER_API_KEY \
  --chat_token_parameter max_tokens \
  --openrouter_provider_only baseten/fp8 --openrouter_provider_only modal/mxfp4 \
  --openrouter_provider_only digitalocean --openrouter_provider_only together \
  --openrouter_provider_only fireworks --openrouter_provider_only moonshotai/mxfp4 \
  --openrouter_provider_order baseten/fp8 --openrouter_provider_order modal/mxfp4 \
  --openrouter_provider_order digitalocean --openrouter_provider_order together \
  --openrouter_provider_order fireworks --openrouter_provider_order moonshotai/mxfp4 \
  --openrouter_allow_fallbacks --openrouter_require_parameters --openrouter_enforce_distillable_text \
  --openrouter_reasoning enabled --openrouter_reasoning_effort low --openrouter_include_reasoning \
  --input_usd_per_million 3 --output_usd_per_million 15 \
  --seed 20260801 --max_tasks "${MAX_TASKS}" --max_parents_per_task 1 \
  --samples_per_parent 1 --max_calls "${MAX_CALLS}" \
  --max_input_tokens_per_call 16384 --max_output_tokens "${MAX_OUTPUT}" \
  --max_input_tokens_total "${MAX_INPUT_TOTAL}" --max_output_tokens_total "${MAX_OUTPUT_TOTAL}" \
  --max_total_tokens "${MAX_TOTAL}" --max_usd "${MAX_USD}" \
  --timeout_seconds 900 --inter_call_delay_seconds 2 --abort_on_provider_error \
  --provider_max_attempts 8 --provider_retry_base_seconds 2 --provider_retry_max_seconds 30 \
  --timeout 30 --stability_runs 2 "${schedule_args[@]}" "${plan_args[@]}"
