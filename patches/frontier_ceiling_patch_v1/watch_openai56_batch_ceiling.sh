#!/usr/bin/env bash
# Resume every deterministic OpenAI Batch shard until the selected sealed jobs
# have complete summaries. Each runner invocation makes at most one state
# transition, so this loop cannot queue a second shard before harvesting the
# first one.
set -euo pipefail

PATCH_DIR="/workspace/frontier_ceiling_patch_v1"
LAUNCHER="${LAUNCHER:-${PATCH_DIR}/run_openai56_batch_ceiling.sh}"
BASH_BIN="${BASH_BIN:-bash}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/.venv/bin/python}"
PAIR_ROOT="/workspace/artifacts/frontier_ceiling_two_enrichments"
RUN_ROOT="${RUN_ROOT:-${PAIR_ROOT}/runs/openai56_batch_k2_32k_max_authorized_v1}"
# Terra is the authorized default. Sol is opt-in and uses an independent cap.
MODEL="${MODEL:-terra}"
ARM="${ARM:-both}"
POLL_SECONDS="${POLL_SECONDS:-180}"
MAX_WATCH_CYCLES="${MAX_WATCH_CYCLES:-0}"

case "${MODEL}" in
  sol|terra|both) ;;
  *) echo "MODEL must be sol, terra, or both" >&2; exit 2 ;;
esac
case "${ARM}" in
  opus|codex|both) ;;
  *) echo "ARM must be opus, codex, or both" >&2; exit 2 ;;
esac
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 30 )); then
  echo "POLL_SECONDS must be an integer >= 30" >&2
  exit 2
fi
if ! [[ "${MAX_WATCH_CYCLES}" =~ ^[0-9]+$ ]]; then
  echo "MAX_WATCH_CYCLES must be a non-negative integer" >&2
  exit 2
fi
[[ "${OPENAI56_AUTHORIZE_PAID_BATCH:-0}" == "1" ]] || {
  echo "Set OPENAI56_AUTHORIZE_PAID_BATCH=1 before starting the watcher" >&2
  exit 2
}

models=()
arms=()
[[ "${MODEL}" == "sol" || "${MODEL}" == "both" ]] && models+=("sol")
[[ "${MODEL}" == "terra" || "${MODEL}" == "both" ]] && models+=("terra")
[[ "${ARM}" == "opus" || "${ARM}" == "both" ]] && arms+=("opus")
[[ "${ARM}" == "codex" || "${ARM}" == "both" ]] && arms+=("codex")

for model_tag in "${models[@]}"; do
  cap_name="OPENAI56_${model_tag^^}_JOB_COST_CAP_USD"
  [[ -n "${!cap_name:-}" ]] || {
    echo "Set ${cap_name} before starting the watcher" >&2
    exit 2
  }
done

# Pair-gate every requested model before any call that can create a paid Batch.
for model_tag in "${models[@]}"; do
  ACTION=preflight MODEL="${model_tag}" ARM=both \
    RUN_ROOT="${RUN_ROOT}" "${BASH_BIN}" "${LAUNCHER}"
done

cycle=0
while true; do
  cycle=$((cycle + 1))

  # Advance both paired arms. The launcher's default 700K shard cap keeps
  # their combined same-model queued input below a 1.5M Tier-1 allowance.
  for model_tag in "${models[@]}"; do
    for arm_name in "${arms[@]}"; do
      summary="${RUN_ROOT}/${model_tag}/${arm_name}/summary.json"
      if [[ -s "${summary}" ]]; then
        continue
      fi
      job_output="$(
        ACTION=auto MODEL="${model_tag}" ARM="${arm_name}" \
          RUN_ROOT="${RUN_ROOT}" "${BASH_BIN}" "${LAUNCHER}"
      )"
      printf '%s\n' "${job_output}"
      if [[ "${job_output}" == *'"status": "all_shards_submitted_and_harvested"'* ]]; then
        progress="${RUN_ROOT}/${model_tag}/${arm_name}/progress.json"
        if [[ -s "${progress}" ]]; then
          progress_status="$(
            "${PYTHON_BIN}" -c \
              'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' \
              "${progress}"
          )"
          if [[ "${progress_status}" == "invalid_incomplete_provider_slot_coverage" ]]; then
            echo "OPENAI56_BATCH_WATCH_INVALID model=${model_tag} arm=${arm_name} progress=${progress}" >&2
            exit 4
          fi
        fi
      fi
    done
  done

  pending=0
  for model_tag in "${models[@]}"; do
    for arm_name in "${arms[@]}"; do
      summary="${RUN_ROOT}/${model_tag}/${arm_name}/summary.json"
      [[ -s "${summary}" ]] || pending=$((pending + 1))
    done
  done

  if (( pending == 0 )); then
    echo "OPENAI56_BATCH_WATCH_COMPLETE cycles=${cycle} run_root=${RUN_ROOT}"
    exit 0
  fi
  if (( MAX_WATCH_CYCLES > 0 && cycle >= MAX_WATCH_CYCLES )); then
    echo "OPENAI56_BATCH_WATCH_LIMIT cycles=${cycle} pending=${pending}" >&2
    exit 3
  fi
  sleep "${POLL_SECONDS}"
done
