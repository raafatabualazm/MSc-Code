#!/usr/bin/env bash
set -euo pipefail

cd /workspace

run_name="arm64_regions16_s42"
status="results/${run_name}.status"
log="logs/arm64_graphv21/${run_name}.launcher.log"

mkdir -p results logs/arm64_graphv21
if [[ -f /workspace/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/.env
  set +a
fi

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'RUNNING stage=arm64_train_eval seed=42 architecture=prefix_no_gine_regions16 started=%s\n' "$started" > "$status"

set +e
/venv/main/bin/python scripts/run_arm64_graphv21_study.py \
  --phase selected \
  --selected_seeds 42 \
  --selected_architecture prefix_no_gine_regions16 \
  --budget_hours 4 \
  --hf_repo raafatabualazm/antigravity-qwen3-8b-artifacts \
  --execute 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
set -e

finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ $code -eq 0 ]]; then
  printf 'COMPLETE stage=arm64_train_eval seed=42 architecture=prefix_no_gine_regions16 finished=%s\n' "$finished" > "$status"
else
  printf 'FAILED stage=arm64_train_eval seed=42 architecture=prefix_no_gine_regions16 exit_code=%s finished=%s\n' "$code" "$finished" > "$status"
fi
exit "$code"
