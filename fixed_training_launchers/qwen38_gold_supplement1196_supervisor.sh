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
export PYTHONUNBUFFERED=1
bash \
  /workspace/fixed_training_launchers/run_qwen38_gold_supplement_1196.sh \
  2>&1 | tee -a /workspace/logs/qwen38_gold_supplement1196.log
