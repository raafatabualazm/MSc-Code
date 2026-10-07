#!/usr/bin/env bash
# Launch/supervise the quarantined fit2776 Qwen3.7 auxiliary repair collector.
#
# Usage:
#   ./scripts/run_qwen37_auxiliary_repairs.sh plan
#   ./scripts/run_qwen37_auxiliary_repairs.sh run
#   ./scripts/run_qwen37_auxiliary_repairs.sh start
#   ./scripts/run_qwen37_auxiliary_repairs.sh status
#   ./scripts/run_qwen37_auxiliary_repairs.sh wait
#
# This wrapper never imports its output into Qwen3.8 KL/CoT.  The only training
# artifact it produces is verified code for a separate auxiliary RS-SFT stage.
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION="${1:-}"

case "${ACTION}" in
  plan|run|start|status|wait) ;;
  *)
    printf 'usage: %s {plan|run|start|status|wait}\n' "$0" >&2
    exit 2
    ;;
esac

QWEN_ENV_FILE="${QWEN_ENV_FILE:-}"
if [[ -n "${QWEN_ENV_FILE}" && -f "${QWEN_ENV_FILE}" ]]; then
  declare -A qwen_env_seen=()
  while IFS= read -r qwen_env_line || [[ -n "${qwen_env_line}" ]]; do
    qwen_env_line="${qwen_env_line%$'\r'}"
    if [[ -z "${qwen_env_line}" || "${qwen_env_line}" == \#* ]]; then
      continue
    fi
    if [[ ! "${qwen_env_line}" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
      printf 'Qwen env contains a non-KEY=VALUE line\n' >&2
      exit 2
    fi
    qwen_env_key="${BASH_REMATCH[1]}"
    qwen_env_value="${BASH_REMATCH[2]}"
    case "${qwen_env_key}" in
      API_KEY|QWEN_API_KEY|DASHSCOPE_API_KEY|DASHSCOPE_ENDPOINT|QWEN_BASE_URL)
        ;;
      *)
        printf 'Qwen env contains disallowed key: %s\n' \
          "${qwen_env_key}" >&2
        exit 2
        ;;
    esac
    if [[ -n "${qwen_env_seen[${qwen_env_key}]:-}" ]]; then
      printf 'Qwen env contains duplicate key: %s\n' "${qwen_env_key}" >&2
      exit 2
    fi
    qwen_env_seen["${qwen_env_key}"]=1
    if [[ "${qwen_env_value}" == *'$('* \
       || "${qwen_env_value}" == *'`'* \
       || "${qwen_env_value}" == *';'* ]]; then
      printf 'Qwen env value for %s contains shell syntax\n' \
        "${qwen_env_key}" >&2
      exit 2
    fi
    printf -v "${qwen_env_key}" '%s' "${qwen_env_value}"
    export "${qwen_env_key}"
  done < "${QWEN_ENV_FILE}"
fi

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  if [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
    export QWEN_API_KEY="${DASHSCOPE_API_KEY}"
  elif [[ -n "${API_KEY:-}" ]]; then
    export QWEN_API_KEY="${API_KEY}"
  fi
fi

QWEN37_MODEL="${QWEN37_MODEL:-}"
QWEN37_BASE_URL="${QWEN37_BASE_URL:-${QWEN_BASE_URL:-${DASHSCOPE_ENDPOINT:-}}}"
QWEN37_MODE="${QWEN37_MODE:-auxiliary_verified_rs_sft_hard_targets_only}"
QWEN37_WORKERS="${QWEN37_WORKERS:-1}"
QWEN37_BUDGET_TOKENS="${QWEN37_BUDGET_TOKENS:-900000}"
QWEN37_MAX_PROMPT_TOKENS="${QWEN37_MAX_PROMPT_TOKENS:-12288}"
QWEN37_MAX_OUTPUT_TOKENS="${QWEN37_MAX_OUTPUT_TOKENS:-12288}"
QWEN37_THINKING_BUDGET="${QWEN37_THINKING_BUDGET:-8192}"
QWEN37_TOKEN_PLAN_AUTOMATION_AUTHORIZED="${
  QWEN37_TOKEN_PLAN_AUTOMATION_AUTHORIZED:-0
}"

: "${QWEN37_MODEL:?set QWEN37_MODEL to an exact pinned snapshot}"
: "${QWEN37_BASE_URL:?set QWEN37_BASE_URL}"
: "${FIT2776_JSONL:?set FIT2776_JSONL}"
: "${FIT2776_SHA256:?set FIT2776_SHA256}"
: "${FIT2776_SEAL:?set FIT2776_SEAL}"
: "${FIT2776_SEAL_SHA256:?set FIT2776_SEAL_SHA256}"
: "${FROZEN_CONTRACT:?set FROZEN_CONTRACT}"
: "${FROZEN_CONTRACT_SHA256:?set FROZEN_CONTRACT_SHA256}"
: "${FIT2776_PROMPT_JSONL:?set FIT2776_PROMPT_JSONL}"
: "${FIT2776_PROMPT_SHA256:?set FIT2776_PROMPT_SHA256}"
: "${FIT2776_PROMPT_MANIFEST:?set FIT2776_PROMPT_MANIFEST}"
: "${FIT2776_PROMPT_MANIFEST_SHA256:?set FIT2776_PROMPT_MANIFEST_SHA256}"
: "${STUDENT_TOKENIZER_JSON:?set STUDENT_TOKENIZER_JSON}"
: "${STUDENT_TOKENIZER_SHA256:?set STUDENT_TOKENIZER_SHA256}"
: "${FIT2776_PREDICTIONS:?set FIT2776_PREDICTIONS}"
: "${FIT2776_PREDICTIONS_SHA256:?set FIT2776_PREDICTIONS_SHA256}"
: "${FIT2776_SCORE:?set FIT2776_SCORE}"
: "${FIT2776_SCORE_SHA256:?set FIT2776_SCORE_SHA256}"
: "${QWEN37_ARTIFACT_ROOT:?set QWEN37_ARTIFACT_ROOT}"

case "${QWEN37_MODEL}" in
  qwen3.7-max-2026-05-17|qwen3.7-max-2026-05-20|qwen3.7-max-2026-06-08)
    ;;
  *)
    printf 'QWEN37_MODEL must be an exact allowed pinned snapshot\n' >&2
    exit 2
    ;;
esac
if [[ "${QWEN37_MODE}" != "auxiliary_verified_rs_sft_hard_targets_only" ]]; then
  printf 'QWEN37_MODE is restricted to verified RS-SFT hard targets\n' >&2
  exit 2
fi
if [[ "${QWEN37_WORKERS}" != "1" ]]; then
  printf 'QWEN37_WORKERS must be 1 for the hard-budget one-draw journal\n' >&2
  exit 2
fi
if (( QWEN37_BUDGET_TOKENS < 1 || QWEN37_BUDGET_TOKENS > 900000 )); then
  printf 'QWEN37_BUDGET_TOKENS must be in [1, 900000]\n' >&2
  exit 2
fi

TOKEN_PLAN_ARGS=()
case "${QWEN37_TOKEN_PLAN_AUTOMATION_AUTHORIZED,,}" in
  1|true|yes)
    TOKEN_PLAN_ARGS=(--token-plan-automation-authorized)
    ;;
  0|false|no)
    ;;
  *)
    printf 'QWEN37_TOKEN_PLAN_AUTOMATION_AUTHORIZED must be boolean\n' >&2
    exit 2
    ;;
esac

COMMAND=(
  "${PYTHON_BIN}"
  "${PATCH_ROOT}/scripts/training/collect_qwen37_auxiliary_repairs.py"
  --fit-jsonl "${FIT2776_JSONL}"
  --expected-fit-sha256 "${FIT2776_SHA256}"
  --fit-seal "${FIT2776_SEAL}"
  --expected-fit-seal-sha256 "${FIT2776_SEAL_SHA256}"
  --frozen-contract "${FROZEN_CONTRACT}"
  --expected-frozen-contract-sha256 "${FROZEN_CONTRACT_SHA256}"
  --prompt-jsonl "${FIT2776_PROMPT_JSONL}"
  --expected-prompt-sha256 "${FIT2776_PROMPT_SHA256}"
  --prompt-manifest "${FIT2776_PROMPT_MANIFEST}"
  --expected-prompt-manifest-sha256 "${FIT2776_PROMPT_MANIFEST_SHA256}"
  --student-tokenizer-json "${STUDENT_TOKENIZER_JSON}"
  --expected-student-tokenizer-sha256 "${STUDENT_TOKENIZER_SHA256}"
  --predictions "${FIT2776_PREDICTIONS}"
  --expected-predictions-sha256 "${FIT2776_PREDICTIONS_SHA256}"
  --score "${FIT2776_SCORE}"
  --expected-score-sha256 "${FIT2776_SCORE_SHA256}"
  --artifact-root "${QWEN37_ARTIFACT_ROOT}"
  --model "${QWEN37_MODEL}"
  --mode "${QWEN37_MODE}"
  --workers "${QWEN37_WORKERS}"
  --base-url "${QWEN37_BASE_URL}"
  --api-key-env QWEN_API_KEY
  --budget-tokens "${QWEN37_BUDGET_TOKENS}"
  --max-prompt-tokens "${QWEN37_MAX_PROMPT_TOKENS}"
  --max-output-tokens "${QWEN37_MAX_OUTPUT_TOKENS}"
  --thinking-budget "${QWEN37_THINKING_BUDGET}"
  "${TOKEN_PLAN_ARGS[@]}"
)

SUPERVISOR_ROOT="${QWEN37_SUPERVISOR_ROOT:-${
  QWEN37_ARTIFACT_ROOT%/
}.supervisor/${QWEN37_MODEL}}"
mkdir -p "${SUPERVISOR_ROOT}"
PID_FILE="${SUPERVISOR_ROOT}/collector.pid"
LOG_FILE="${SUPERVISOR_ROOT}/collector.log"
LOCK_FILE="${SUPERVISOR_ROOT}/collector.lock"

read_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    tr -dc '0-9' < "${PID_FILE}"
  fi
}

is_running() {
  local pid
  pid="$(read_pid)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

case "${ACTION}" in
  plan)
    "${COMMAND[@]}" --dry-run
    ;;
  run)
    if is_running; then
      printf 'collector already running pid=%s\n' "$(read_pid)" >&2
      exit 1
    fi
    exec flock -n "${LOCK_FILE}" "${COMMAND[@]}"
    ;;
  start)
    if is_running; then
      printf 'collector already running pid=%s\n' "$(read_pid)"
      exit 0
    fi
    nohup flock -n "${LOCK_FILE}" "${COMMAND[@]}" \
      >>"${LOG_FILE}" 2>&1 </dev/null &
    collector_pid=$!
    printf '%s\n' "${collector_pid}" > "${PID_FILE}"
    printf 'QWEN37_AUX_STARTED model=%s pid=%s log=%s\n' \
      "${QWEN37_MODEL}" "${collector_pid}" "${LOG_FILE}"
    ;;
  status)
    if is_running; then
      printf 'QWEN37_AUX_STATUS running model=%s pid=%s log=%s\n' \
        "${QWEN37_MODEL}" "$(read_pid)" "${LOG_FILE}"
    elif [[ -f "${QWEN37_ARTIFACT_ROOT}/${QWEN37_MODEL}/build_report.json" ]]; then
      printf 'QWEN37_AUX_STATUS complete model=%s report=%s\n' \
        "${QWEN37_MODEL}" \
        "${QWEN37_ARTIFACT_ROOT}/${QWEN37_MODEL}/build_report.json"
    else
      printf 'QWEN37_AUX_STATUS stopped model=%s log=%s\n' \
        "${QWEN37_MODEL}" "${LOG_FILE}"
      exit 1
    fi
    ;;
  wait)
    if ! is_running; then
      printf 'collector is not running\n' >&2
      exit 1
    fi
    collector_pid="$(read_pid)"
    while kill -0 "${collector_pid}" 2>/dev/null; do
      sleep 5
    done
    wait "${collector_pid}" 2>/dev/null || true
    if [[ -f "${QWEN37_ARTIFACT_ROOT}/${QWEN37_MODEL}/build_report.json" ]]; then
      printf 'QWEN37_AUX_COMPLETE model=%s report=%s\n' \
        "${QWEN37_MODEL}" \
        "${QWEN37_ARTIFACT_ROOT}/${QWEN37_MODEL}/build_report.json"
    else
      printf 'QWEN37_AUX_FAILED model=%s log=%s\n' \
        "${QWEN37_MODEL}" "${LOG_FILE}" >&2
      exit 1
    fi
    ;;
esac
