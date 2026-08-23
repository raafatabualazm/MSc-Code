#!/bin/bash
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1091
. "${utils}/logging.sh" ""
# shellcheck disable=SC1091
. "${utils}/environment.sh"

source /venv/main/bin/activate
mkdir -p /workspace/logs
cd /workspace
export QWEN_TEACHER_WORKERS="${QWEN_TEACHER_WORKERS:-32}"
export QWEN_VERIFIER_WORKERS="${QWEN_VERIFIER_WORKERS:-32}"

# The paid collector is append-only and fsyncs a slot receipt before every
# provider call.  A launcher retry therefore resumes completed draws and
# explicitly recovers only indeterminate in-flight slots; it never begins the
# dataset again.  Keep transient endpoint/process failures from leaving the
# long harvest idle after a dropped SSH session.
restart_limit="${QWEN_SUPERVISOR_RESUME_LIMIT:-64}"
restart_delay="${QWEN_SUPERVISOR_RESUME_DELAY_SECONDS:-20}"
quota_not_before_utc="${QWEN_QUOTA_NOT_BEFORE_UTC:-1970-01-01T00:00:00Z}"
quota_retry_seconds="${QWEN_QUOTA_RETRY_SECONDS:-900}"
if ! [[ "${restart_limit}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${restart_delay}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${quota_retry_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'QWEN_SUPERVISOR_RESUME_LIMIT and delay must be positive integers\n' >&2
  exit 2
fi

quota_epoch="$(date -u -d "${quota_not_before_utc}" +%s)" || {
  printf 'Invalid QWEN_QUOTA_NOT_BEFORE_UTC: %s\n' \
    "${quota_not_before_utc}" >&2
  exit 2
}
journal=/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2/qwen_teacher.journal.jsonl

last_provider_outcome_is_quota() {
  [[ -s "${journal}" ]] || return 1
  /venv/main/bin/python - "${journal}" <<'PY'
import json
import sys

last_error = (-1, "")
last_terminal = -1
for line in open(sys.argv[1], "r", encoding="utf-8"):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    index = int(row.get("journal_event_index", -1))
    if row.get("event") == "teacher_error":
        last_error = (index, str(row.get("provider_error_code") or ""))
    elif row.get("event") == "teacher_slot_terminal":
        last_terminal = max(last_terminal, index)
raise SystemExit(
    0
    if last_error[0] > last_terminal
    and last_error[1] == "insufficient_quota"
    else 1
)
PY
}

wait_for_quota() {
  local now remaining sleep_for
  now="$(date -u +%s)"
  if (( now < quota_epoch )); then
    printf 'QWEN_QUOTA_PAUSE resume_at=%s journal_resume=true\n' \
      "${quota_not_before_utc}"
    while (( now < quota_epoch )); do
      remaining=$((quota_epoch - now))
      sleep_for=60
      (( remaining < sleep_for )) && sleep_for="${remaining}"
      sleep "${sleep_for}"
      now="$(date -u +%s)"
    done
  else
    printf 'QWEN_QUOTA_BACKOFF seconds=%s journal_resume=true\n' \
      "${quota_retry_seconds}"
    sleep "${quota_retry_seconds}"
  fi
}

attempt=0
while :; do
  if last_provider_outcome_is_quota \
    && (( $(date -u +%s) < quota_epoch )); then
    wait_for_quota
  fi
  printf 'QWEN_RUNTIME_PARALLELISM api_workers=%s verifier_workers=%s operational_only=true\n' \
    "${QWEN_TEACHER_WORKERS}" "${QWEN_VERIFIER_WORKERS}"
  set +e
  bash /workspace/run_qwen38_sequence_kd.sh 2>&1 \
    | tee -a /workspace/logs/qwen38_sequence_kd.log
  status="${PIPESTATUS[0]}"
  set -e
  if (( status == 0 )); then
    exit 0
  fi
  if last_provider_outcome_is_quota; then
    wait_for_quota
    continue
  fi

  # A consumed malformed response is a terminal Monte Carlo draw, not a
  # transport orphan.  The collector intentionally forbids replacing it.
  # Retrying such a journal would only spin, so fail closed.
  if [[ -s "${journal}" ]] && /venv/main/bin/python - "${journal}" <<'PY'
import json
import sys

for line in open(sys.argv[1], "r", encoding="utf-8"):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("event") == "teacher_rejected_draw":
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    printf 'QWEN_SUPERVISOR_STOP terminal_rejected_draw=true exit=%s\n' \
      "${status}" >&2
    exit "${status}"
  fi

  attempt=$((attempt + 1))
  if (( attempt > restart_limit )); then
    printf 'QWEN_SUPERVISOR_STOP resume_limit=%s exit=%s\n' \
      "${restart_limit}" "${status}" >&2
    exit "${status}"
  fi
  printf 'QWEN_SUPERVISOR_JOURNAL_RESUME attempt=%s/%s delay_seconds=%s prior_exit=%s\n' \
    "${attempt}" "${restart_limit}" "${restart_delay}" "${status}"
  sleep "${restart_delay}"
done
