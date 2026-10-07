#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_finish_rs_sft.sh"
bash "${SCRIPT_DIR}/run_verpo_v2.sh"
