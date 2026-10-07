#!/usr/bin/env bash
set -euo pipefail

# Manual API phase launcher.  It never derives a digest from mutable evidence:
# every local/split/projection/prior input must be paired with an exact SHA-256.
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
COHORT_INDEX="${T5GEMMA_TYPED_API_COHORT_INDEX:-0}"
OUTPUT_DIR="${T5GEMMA_TYPED_API_OUTPUT_DIR:-}"
PLAN_ONLY_OUTPUT="${T5GEMMA_TYPED_API_PLAN_ONLY_OUTPUT:-}"
FIXED_KIMI_COHORT_LIMIT="${T5GEMMA_TYPED_API_FIXED_KIMI_COHORT_LIMIT:-0}"
BUDGET_SKIPPED_RETRY_TASKS="${T5GEMMA_TYPED_API_BUDGET_SKIPPED_KIMI_RETRY_TASKS:-0}"
BUDGET_SKIPPED_RETRY_IDS_SHA="${T5GEMMA_TYPED_API_BUDGET_SKIPPED_KIMI_RETRY_IDS_SHA256:-}"

printf '%s  %s\n' \
  0cc6323136e12110e6ee25ef42f8e0baec90359ef7cc972fb191691a12fd7b15 "${PROJECT}/scripts/preprocessing/build_t5gemma2_typed_api_visible_split.py" \
  b7f66f7ee0b1959fe6a6b8bbe6fa422d545ab950971dd6aae86bfb31acde0f88 "${PROJECT}/scripts/training/t5gemma2_typed_visible_failure_projection.py" \
  7a03af003e998497012706361f5cbf0734d8defa82c7e458aa5f87f796e01143 "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_cascade.py" \
  | sha256sum -c -

if [[ "${PHASE}" != kimi_initial && "${PHASE}" != kimi_retry && "${PHASE}" != sonnet_residual ]]; then
  echo "T5GEMMA_TYPED_API_BLOCKED invalid T5GEMMA_TYPED_API_PHASE" >&2
  exit 78
fi
if ! [[ "${COHORT_INDEX}" =~ ^[0-9]+$ ]] || [[ -z "${OUTPUT_DIR}" ]]; then
  echo "T5GEMMA_TYPED_API_BLOCKED cohort/output is absent" >&2
  exit 78
fi

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
  if ! [[ "${!name:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "T5GEMMA_TYPED_API_BLOCKED ${name} is not pinned" >&2
    exit 78
  fi
done
if [[ -z "${PLAN_ONLY_OUTPUT}" ]] && ! [[ "${T5GEMMA_TYPED_API_SCHEDULE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "T5GEMMA_TYPED_API_BLOCKED T5GEMMA_TYPED_API_SCHEDULE_SHA256 is not pinned" >&2
  exit 78
fi
if ! [[ "${FIXED_KIMI_COHORT_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "T5GEMMA_TYPED_API_BLOCKED fixed Kimi cohort limit is invalid" >&2
  exit 78
fi
if ! [[ "${BUDGET_SKIPPED_RETRY_TASKS}" =~ ^[0-9]+$ ]]; then
  echo "T5GEMMA_TYPED_API_BLOCKED budget-skipped retry attestation is invalid" >&2
  exit 78
fi
if (( BUDGET_SKIPPED_RETRY_TASKS > 0 )); then
  [[ "${BUDGET_SKIPPED_RETRY_IDS_SHA}" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "T5GEMMA_TYPED_API_BLOCKED budget-skipped retry digest is invalid" >&2; exit 78; }
elif [[ -n "${BUDGET_SKIPPED_RETRY_IDS_SHA}" ]]; then
  echo "T5GEMMA_TYPED_API_BLOCKED budget-skipped retry digest lacks a count" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_TYPED_API_BLOCKED Dart 3.12.2 is absent" >&2
  exit 78
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
  | sha256sum -c -

prior_args=()
if [[ "${PHASE}" == kimi_retry ]]; then
  SOURCE_REPORT="${T5GEMMA_TYPED_API_RETRY_SOURCE_REPORT:-}"
  SOURCE_SHA="${T5GEMMA_TYPED_API_RETRY_SOURCE_SHA256:-}"
  RETRY_TASKS="${T5GEMMA_TYPED_API_RETRY_TASKS:-}"
  RETRY_IDS_SHA="${T5GEMMA_TYPED_API_RETRY_IDS_SHA256:-}"
  if [[ ! -s "${SOURCE_REPORT}" ]] \
    || ! [[ "${SOURCE_SHA}" =~ ^[0-9a-f]{64}$ ]] \
    || ! [[ "${RETRY_TASKS}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${RETRY_IDS_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "T5GEMMA_TYPED_API_BLOCKED exact Kimi retry source is absent" >&2
    exit 78
  fi
  printf '%s  %s\n' "${SOURCE_SHA}" "${SOURCE_REPORT}" | sha256sum -c -
  prior_args+=(
    --retry_parse_failures_or_truncations_report "${SOURCE_REPORT}"
    --expected_retry_parse_failures_or_truncations_report_sha256 "${SOURCE_SHA}"
    --expected_retry_parse_failures_or_truncations_tasks "${RETRY_TASKS}"
    --expected_retry_parse_failures_or_truncations_task_ids_sha256 "${RETRY_IDS_SHA}"
  )
  MAX_TASKS="${RETRY_TASKS}"
  MAX_CALLS="${RETRY_TASKS}"
  MAX_OUTPUT=8192
  MAX_INPUT_TOTAL=$((65536 * RETRY_TASKS))
  MAX_OUTPUT_TOTAL=$((8192 * RETRY_TASKS))
  MAX_TOTAL=$((MAX_INPUT_TOTAL + MAX_OUTPUT_TOTAL))
  MAX_USD="${T5GEMMA_TYPED_API_MAX_USD:-12}"
elif [[ "${PHASE}" == kimi_initial ]]; then
  MAX_TASKS=50
  MAX_CALLS=50
  MAX_OUTPUT=2048
  MAX_INPUT_TOTAL=3276800
  MAX_OUTPUT_TOTAL=102400
  MAX_TOTAL=3379200
  MAX_USD="${T5GEMMA_TYPED_API_MAX_USD:-12}"
else
  MAX_TASKS="${T5GEMMA_TYPED_API_MAX_TASKS:-38}"
  if ! [[ "${MAX_TASKS}" =~ ^[1-9][0-9]*$ ]] || (( MAX_TASKS > 50 )); then
    echo "T5GEMMA_TYPED_API_BLOCKED Sonnet task cap is invalid" >&2
    exit 78
  fi
  MAX_CALLS="${MAX_TASKS}"
  MAX_OUTPUT=16384
  MAX_INPUT_TOTAL=$((65536 * MAX_TASKS))
  MAX_OUTPUT_TOTAL=$((16384 * MAX_TASKS))
  MAX_TOTAL=4096000
  MAX_TOTAL=$((MAX_INPUT_TOTAL + MAX_OUTPUT_TOTAL))
  MAX_USD="${T5GEMMA_TYPED_API_MAX_USD:-11.5}"
fi

if [[ ( "${PHASE}" != "kimi_retry" && "${COHORT_INDEX}" != "0" ) || "${PHASE}" == "sonnet_residual" ]]; then
  PRIOR_INDEX="${T5GEMMA_TYPED_API_PRIOR_INDEX:-}"
  PRIOR_INDEX_SHA="${T5GEMMA_TYPED_API_PRIOR_INDEX_SHA256:-}"
  if [[ ! -s "${PRIOR_INDEX}" ]] || ! [[ "${PRIOR_INDEX_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "T5GEMMA_TYPED_API_BLOCKED pinned prior report index is absent" >&2
    exit 78
  fi
  printf '%s  %s\n' "${PRIOR_INDEX_SHA}" "${PRIOR_INDEX}" | sha256sum -c -
  while IFS=$'\t' read -r digest path; do
    if ! [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || [[ ! -s "${path}" ]]; then
      echo "T5GEMMA_TYPED_API_BLOCKED malformed prior report index" >&2
      exit 78
    fi
    printf '%s  %s\n' "${digest}" "${path}" | sha256sum -c -
    prior_args+=(--prior_success_report "${path}" --expected_prior_success_report_sha256 "${digest}")
  done <"${PRIOR_INDEX}"
fi

if [[ "${PHASE}" == sonnet_residual ]]; then
  SECRET_FILE="${T5GEMMA_ANTHROPIC_ENV:-${WORKSPACE}/secrets/Anthropic.env}"
  KEY_NAME=ANTHROPIC_API_KEY
  provider_args=(
    --provider anthropic --model claude-sonnet-5
    --base_url https://api.anthropic.com --api_key_env ANTHROPIC_API_KEY
    --anthropic_thinking adaptive --anthropic_effort high
    --input_usd_per_million 2 --output_usd_per_million 10
  )
else
  SECRET_FILE="${T5GEMMA_OPENROUTER_ENV:-${WORKSPACE}/secrets/Openrouter.env}"
  KEY_NAME=OPENROUTER_API_KEY
  provider_args=(
    --provider openrouter_chat --model moonshotai/kimi-k3
    --base_url https://openrouter.ai/api/v1 --api_key_env OPENROUTER_API_KEY
    --chat_token_parameter max_tokens
    --openrouter_provider_only baseten/fp8
    --openrouter_provider_only modal/mxfp4
    --openrouter_provider_only digitalocean
    --openrouter_provider_only together
    --openrouter_provider_only fireworks
    --openrouter_provider_only moonshotai/mxfp4
    --openrouter_provider_order baseten/fp8
    --openrouter_provider_order modal/mxfp4
    --openrouter_provider_order digitalocean
    --openrouter_provider_order together
    --openrouter_provider_order fireworks
    --openrouter_provider_order moonshotai/mxfp4
    --openrouter_allow_fallbacks --openrouter_require_parameters
    --openrouter_enforce_distillable_text
    --openrouter_reasoning enabled --openrouter_reasoning_effort low
    --openrouter_include_reasoning
    --input_usd_per_million 3 --output_usd_per_million 15
  )
fi
if [[ -z "${PLAN_ONLY_OUTPUT}" ]]; then
  if [[ ! -s "${SECRET_FILE}" ]]; then
    echo "T5GEMMA_TYPED_API_BLOCKED provider secret file is absent" >&2
    exit 78
  fi

  # Parse exactly one requested key assignment; never source a multi-key file.
  provider_key="$(/venv/main/bin/python - "${SECRET_FILE}" "${KEY_NAME}" <<'PY'
import re, stat, sys
from pathlib import Path
p=Path(sys.argv[1]); name=sys.argv[2]
if stat.S_IMODE(p.stat().st_mode) & 0o077:
    raise SystemExit("provider secret file must be mode 0600")
raw=p.read_bytes()
try: text=raw.decode("utf-8-sig")
except UnicodeDecodeError: text=raw.decode("utf-16")
values=[]
for line in text.splitlines():
    line=line.strip()
    if not line or line.startswith("#"): continue
    m=re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
    if m and m.group(1)==name:
        value=m.group(2).strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in "\"'": value=value[1:-1]
        values.append(value)
if len(values)!=1 or not values[0] or any(c.isspace() for c in values[0]):
    raise SystemExit(f"{name} must occur exactly once and be well formed")
print(values[0], end="")
PY
)"
  export "${KEY_NAME}=${provider_key}"
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
exec /venv/main/bin/python scripts/training/t5gemma2_typed_api_rescue_cascade.py \
  --phase "${PHASE}" --cohort_index "${COHORT_INDEX}" \
  --fixed_kimi_cohort_limit "${FIXED_KIMI_COHORT_LIMIT}" \
  --budget_skipped_kimi_retry_tasks "${BUDGET_SKIPPED_RETRY_TASKS}" \
  --budget_skipped_kimi_retry_task_ids_sha256 "${BUDGET_SKIPPED_RETRY_IDS_SHA}" \
  --local_harvest_report "${LOCAL_DIR}/harvest_report.json" \
  --expected_local_harvest_report_sha256 "${T5GEMMA_TYPED_LOCAL_REPORT_SHA256}" \
  --pilot_journal "${LOCAL_DIR}/harvest.journal.jsonl" \
  --expected_local_harvest_journal_sha256 "${T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256}" \
  --local_harvest_targets "${LOCAL_DIR}/direct_targets.jsonl" \
  --expected_local_harvest_targets_sha256 "${T5GEMMA_TYPED_LOCAL_TARGETS_SHA256}" \
  --existing_direct_manifest "${EXISTING_MANIFEST}" \
  --expected_existing_direct_manifest_sha256 "${T5GEMMA_TYPED_225_MANIFEST_SHA256}" \
  --gold_train_jsonl "${GOLD_TRAIN}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --gold_f2_jsonl "${GOLD_F2}" --f2_jsonl "${GOLD_F2}" \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --heldout_jsonl "${HELDOUT}" \
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --visible_split_manifest "${SPLIT_DIR}/split_manifest.json" \
  --expected_visible_split_manifest_sha256 "${T5GEMMA_TYPED_API_SPLIT_MANIFEST_SHA256}" \
  --visible_train "${SPLIT_DIR}/visible_train.jsonl" --rollout_file "${SPLIT_DIR}/visible_train.jsonl" \
  --expected_visible_train_sha256 "${T5GEMMA_TYPED_API_VISIBLE_SHA256}" \
  --expected_rollout_sha256 "${T5GEMMA_TYPED_API_VISIBLE_SHA256}" \
  --private_split_holdback "${SPLIT_DIR}/holdback.private.jsonl" \
  --private_holdback "${SPLIT_DIR}/holdback.private.jsonl" \
  --expected_private_split_holdback_sha256 "${T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256}" \
  --expected_private_holdback_sha256 "${T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256}" \
  --visible_projection_report "${PROJECTION_DIR}/visible_projection_report.json" \
  --expected_visible_projection_report_sha256 "${T5GEMMA_TYPED_API_PROJECTION_REPORT_SHA256}" \
  --visible_projection_journal "${PROJECTION_DIR}/visible_projection.journal.jsonl" \
  --expected_visible_projection_journal_sha256 "${T5GEMMA_TYPED_API_PROJECTION_JOURNAL_SHA256}" \
  --output_dir "${OUTPUT_DIR}" \
  "${provider_args[@]}" "${prior_args[@]}" \
  --seed 20260801 --max_tasks "${MAX_TASKS}" \
  --max_parents_per_task 1 --samples_per_parent 1 --max_calls "${MAX_CALLS}" \
  --max_input_tokens_per_call 65536 --max_output_tokens "${MAX_OUTPUT}" \
  --max_input_tokens_total "${MAX_INPUT_TOTAL}" \
  --max_output_tokens_total "${MAX_OUTPUT_TOTAL}" --max_total_tokens "${MAX_TOTAL}" \
  --max_usd "${MAX_USD}" --timeout_seconds 900 --inter_call_delay_seconds 2 \
  --abort_on_provider_error --provider_max_attempts 8 \
  --provider_retry_base_seconds 2 --provider_retry_max_seconds 30 \
  --timeout 30 --stability_runs 2 \
  "${schedule_args[@]}" "${plan_args[@]}"
