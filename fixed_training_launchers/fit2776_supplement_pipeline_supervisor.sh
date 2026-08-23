#!/bin/bash
# Build the exact missing 1,196 representations, normalize the sealed views,
# then collect their independent Qwen K=8 supplement.
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

bash /workspace/run_expand_multifunction_2776.sh 2>&1 \
  | tee -a /workspace/logs/fit2776_supplement_pipeline.log

derivation=/workspace/multifunction_v1/expanded2776/qwen_2776_supplement.derivation.json
if [[ ! -f "${derivation}" ]]; then
  bash /workspace/run_prepare_qwen2776_supplement.sh 2>&1 \
    | tee -a /workspace/logs/fit2776_supplement_pipeline.log
fi

patch=/workspace/hybrid_training_patch_v2_3
export PYTHONPATH="${patch}:/workspace"
export PYTHONUNBUFFERED=1
expansion=/workspace/multifunction_v1/expanded2776
base_executable="${expansion}/executable"
target_executable="${expansion}/executable_target24k"
if [[ ! -f "${base_executable}/executable_view.build.json" ]]; then
  expansion_report="${expansion}/build/build_report.json"
  /venv/main/bin/python \
    "${patch}/scripts/preprocessing/build_multifunction_executable_view.py" \
    --parent-build-report "${expansion_report}" \
    --expected-parent-build-report-sha256 \
      "$(sha256sum "${expansion_report}" | awk '{print $1}')" \
    --output-dir "${base_executable}" 2>&1 \
    | tee -a /workspace/logs/fit2776_supplement_pipeline.log
fi
test -f "${base_executable}/executable_view.build.json" || {
  printf 'FIT2776_EXECUTABLE_BUILD_MISSING path=%s\n' \
    "${base_executable}/executable_view.build.json" >&2
  exit 2
}
rich_target_seal="${expansion}/fit2776_expansion.target24k.seal.json"
rich_target_receipt="${expansion}/fit2776_expansion.target24k.rebind.json"
/venv/main/bin/python \
  "${patch}/scripts/preprocessing/rebind_multifunction_parent_capacity.py" \
  --source-rich-seal \
    "${expansion}/build/train_multifunction_binary_expanded_2776.seal.json" \
  --source-dataset \
    "${expansion}/build/train_multifunction_binary_expanded_2776.jsonl" \
  --source-contract \
    /workspace/multifunction_v1/build/multifunction_inline_cfg_v2_contract.json \
  --target-dataset "${expansion}/fit2776_multifunction_binary.jsonl" \
  --target-contract \
    /workspace/multifunction_v1/build/multifunction_inline_cfg_v2_target24k_contract.json \
  --generic-target-seal \
    "${expansion}/fit2776_multifunction_binary.target24k.seal.json" \
  --output-seal "${rich_target_seal}" \
  --output-receipt "${rich_target_receipt}" 2>&1 \
  | tee -a /workspace/logs/fit2776_supplement_pipeline.log
if [[ ! -f "${target_executable}/executable_view.build.json" ]]; then
  base_report="${base_executable}/executable_view.build.json"
  /venv/main/bin/python \
    "${patch}/scripts/preprocessing/migrate_multifunction_executable_capacity.py" \
    --source-dir "${base_executable}" \
    --expected-source-report-sha256 \
      "$(sha256sum "${base_report}" | awk '{print $1}')" \
    --target-contract \
      /workspace/multifunction_v1/build/multifunction_inline_cfg_v2_target24k_contract.json \
    --target-parent-train-seal "${rich_target_seal}" \
    --output-dir "${target_executable}" 2>&1 \
    | tee -a /workspace/logs/fit2776_supplement_pipeline.log
fi
test -f "${target_executable}/executable_view.build.json" || {
  printf 'FIT2776_TARGET24K_EXECUTABLE_MISSING path=%s\n' \
    "${target_executable}/executable_view.build.json" >&2
  exit 2
}

restart_limit="${QWEN_SUPPLEMENT_RESUME_LIMIT:-64}"
restart_delay="${QWEN_SUPPLEMENT_RESUME_DELAY_SECONDS:-20}"
quota_not_before_utc="${QWEN_QUOTA_NOT_BEFORE_UTC:-1970-01-01T00:00:00Z}"
quota_retry_seconds="${QWEN_QUOTA_RETRY_SECONDS:-900}"
if ! [[ "${restart_limit}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${restart_delay}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${quota_retry_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'QWEN supplement resume limit and delay must be positive integers\n' >&2
  exit 2
fi

quota_epoch="$(date -u -d "${quota_not_before_utc}" +%s)" || {
  printf 'Invalid QWEN_QUOTA_NOT_BEFORE_UTC: %s\n' \
    "${quota_not_before_utc}" >&2
  exit 2
}
journal=/workspace/artifacts/direct_compact_qwen38_supplement1196/qwen_teacher.journal.jsonl

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
    printf 'QWEN_SUPPLEMENT_QUOTA_PAUSE resume_at=%s journal_resume=true\n' \
      "${quota_not_before_utc}"
    while (( now < quota_epoch )); do
      remaining=$((quota_epoch - now))
      sleep_for=60
      (( remaining < sleep_for )) && sleep_for="${remaining}"
      sleep "${sleep_for}"
      now="$(date -u +%s)"
    done
  else
    printf 'QWEN_SUPPLEMENT_QUOTA_BACKOFF seconds=%s journal_resume=true\n' \
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
  printf 'QWEN_SUPPLEMENT_RUNTIME_PARALLELISM api_workers=%s verifier_workers=%s operational_only=true\n' \
    "${QWEN_TEACHER_WORKERS}" "${QWEN_VERIFIER_WORKERS}"
  set +e
  bash /workspace/run_qwen38_supplemental_harvest.sh 2>&1 \
    | tee -a /workspace/logs/fit2776_supplement_pipeline.log
  status="${PIPESTATUS[0]}"
  set -e
  if (( status == 0 )); then
    exit 0
  fi
  if last_provider_outcome_is_quota; then
    wait_for_quota
    continue
  fi
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
    printf 'QWEN_SUPPLEMENT_STOP terminal_rejected_draw=true exit=%s\n' \
      "${status}" >&2
    exit "${status}"
  fi
  attempt=$((attempt + 1))
  if (( attempt > restart_limit )); then
    printf 'QWEN_SUPPLEMENT_STOP resume_limit=%s exit=%s\n' \
      "${restart_limit}" "${status}" >&2
    exit "${status}"
  fi
  printf 'QWEN_SUPPLEMENT_JOURNAL_RESUME attempt=%s/%s delay_seconds=%s prior_exit=%s\n' \
    "${attempt}" "${restart_limit}" "${restart_delay}" "${status}"
  sleep "${restart_delay}"
done
