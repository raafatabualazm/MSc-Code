#!/usr/bin/env bash
# Staged launch of the FROZEN representation winner (prefix_no_gine) on ARM64 —
# the proper step-9 cross-ISA replication (the earlier ARM64 run was regions16,
# a screen arm, not the winner). Fires AFTER the currently-running GPU work
# finishes. The scrubbed x86 datasets are handled by a separate agent; this
# only concerns the ARM64 pod's GPU.
#
# SAFETY (Rule 0): this script NEVER stops, kills, or preempts a running job.
# It only polls the GPU and waits until it is idle, then launches. Deploy to the
# ARM64 pod (/workspace) and run detached:
#   nohup bash scripts/launch_frozen_winner_after_current.sh > frozen_arm64.out 2>&1 &
#
# Frozen winner on ARM64 = --selected_architecture prefix_no_gine:
#   topology-free (GINE bypassed), region_compression off, CLS block pooling,
#   GraphCodeBERT encoder LoRA trainable. Uses the same run_arm64_graphv21_study
#   path proven by the completed regions16 run. Seed 42, held-out 343-task
#   flutter_eval, no ARM64-specific tuning (per the freeze rules).
#
# NOTE ON CONFIG FIDELITY: the ARM64 runner's prefix_no_gine uses 4 vectors per
# block, matching every other ARM64 arm; the x86 freeze note said single-vector.
# On x86 that knob was null (multivector4 == no_gine, both 0.260), so it will
# not move the ARM64 result. Left at the runner default for pipeline
# consistency; override in the runner if you want a strict single-vector match.
set -euo pipefail
cd /workspace

RUN_NAME="${RUN_NAME:-frozen_winner_arm64_prefix_no_gine_s42}"
SEED="${SEED:-42}"
ARCH="${ARCH:-prefix_no_gine}"
BUDGET_HOURS="${BUDGET_HOURS:-4}"
HF_REPO="${HF_REPO:-raafatabualazm/antigravity-qwen3-8b-artifacts}"
IDLE_CHECKS="${IDLE_CHECKS:-6}"        # consecutive idle polls required to launch
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-12}"
# Optional: wait on specific status files instead of GPU-idle. Space-separated
# globs; launch fires only once every match reads COMPLETE/FAILED/ABORTED. Use
# this if another job may start on this GPU in the gap (GPU-idle could otherwise
# fire between two separately-launched jobs).
WAIT_STATUS_GLOB="${WAIT_STATUS_GLOB:-}"

status="results/${RUN_NAME}.status"
log="logs/frozen_winner/${RUN_NAME}.launcher.log"
mkdir -p results logs/frozen_winner
if [[ -f /workspace/.env ]]; then set -a; source /workspace/.env; set +a; fi

printf 'WAITING started=%s waiting_for=%s\n' \
  "$(date -u +%FT%TZ)" "${WAIT_STATUS_GLOB:-gpu_idle}" > "$status"

gpu_busy() {
  local n
  n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [[ "${n:-0}" -gt 0 ]]
}

status_pending() {
  # true (0) if any watched status file is missing or not terminal
  local g f any
  for g in $WAIT_STATUS_GLOB; do
    any=0
    for f in $g; do
      [[ -e "$f" ]] || return 0
      any=1
      grep -Eq '^(COMPLETE|FAILED|ABORTED)' "$f" || return 0
    done
    [[ "$any" -eq 1 ]] || return 0
  done
  return 1
}

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
idle=0
while :; do
  if (( $(date +%s) > deadline )); then
    printf 'ABORTED reason=max_wait_exceeded at=%s\n' "$(date -u +%FT%TZ)" > "$status"
    echo "$(date -u +%FT%TZ) aborted: waited > ${MAX_WAIT_HOURS}h" | tee -a "$log"
    exit 3
  fi
  if [[ -n "$WAIT_STATUS_GLOB" ]]; then
    if status_pending; then
      echo "$(date -u +%FT%TZ) watched job(s) still running — waiting" | tee -a "$log"
      sleep "$POLL_SECONDS"; continue
    fi
    echo "$(date -u +%FT%TZ) watched job(s) all finished" | tee -a "$log"
    break
  fi
  if gpu_busy; then
    idle=0
    echo "$(date -u +%FT%TZ) GPU busy (a run is active) — waiting, not interfering" | tee -a "$log"
  else
    idle=$(( idle + 1 ))
    echo "$(date -u +%FT%TZ) GPU idle ${idle}/${IDLE_CHECKS}" | tee -a "$log"
    (( idle >= IDLE_CHECKS )) && break
  fi
  sleep "$POLL_SECONDS"
done

echo "$(date -u +%FT%TZ) launching ARM64 frozen winner (${ARCH}, seed ${SEED})" | tee -a "$log"
printf 'RUNNING stage=arm64_frozen_winner arm=%s seed=%s started=%s\n' \
  "$ARCH" "$SEED" "$(date -u +%FT%TZ)" > "$status"

set +e
/venv/main/bin/python scripts/run_arm64_graphv21_study.py \
  --phase selected \
  --selected_seeds "$SEED" \
  --selected_architecture "$ARCH" \
  --budget_hours "$BUDGET_HOURS" \
  --hf_repo "$HF_REPO" \
  --execute 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
set -e

fin="$(date -u +%FT%TZ)"
if [[ $code -eq 0 ]]; then
  printf 'COMPLETE stage=arm64_frozen_winner arm=%s seed=%s finished=%s\n' "$ARCH" "$SEED" "$fin" > "$status"
else
  printf 'FAILED stage=arm64_frozen_winner arm=%s seed=%s exit_code=%s finished=%s\n' "$ARCH" "$SEED" "$code" "$fin" > "$status"
fi
exit "$code"
