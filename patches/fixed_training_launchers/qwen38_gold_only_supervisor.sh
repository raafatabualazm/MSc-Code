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
export QWEN_GOLD_ONLY=1
pty bash /workspace/run_qwen38_sequence_kd.sh 2>&1 \
  | tee -a /workspace/logs/qwen38_gold_only.log
