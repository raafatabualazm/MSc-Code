#!/usr/bin/env bash
# Start or inspect the hash-pinned automatic Qwen capacity epoch scheduler.
set -euo pipefail

action="${1:-}"
if [[ "${action}" != "start" && "${action}" != "status" ]]; then
  printf 'usage: %s {start|status}\n' "$0" >&2
  exit 2
fi

workspace="${WORKSPACE:-/workspace}"
python_bin="${PYTHON_BIN:-${workspace}/.venv/bin/python}"
patch_root="${workspace}/frontier_ceiling_patch_v1"
entry="${patch_root}/qwen37_capacity_scheduler_v6.py"
unit="frontier-qwen37-capacity-scheduler-v6"
expected_sha256="ac87e57264dd341b19b5d80e8862642c06e91f5d62f6cb4dcb2a725f34a0c792"
actual_sha256="$(sha256sum "${entry}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  printf 'scheduler hash mismatch: expected=%s actual=%s\n' \
    "${expected_sha256}" "${actual_sha256}" >&2
  exit 2
fi

if [[ "${action}" == "status" ]]; then
  systemctl show "${unit}.service" \
    --property=LoadState \
    --property=ActiveState \
    --property=SubState \
    --property=Result \
    --property=ExecMainStatus
  exit 0
fi

if systemctl is-active --quiet "${unit}.service"; then
  printf 'ALREADY_RUNNING unit=%s\n' "${unit}"
  exit 0
fi
systemctl reset-failed "${unit}.service" 2>/dev/null || true
systemd-run \
  --unit="${unit}" \
  --description="Qwen3.7 v6 automatic capacity epoch scheduler" \
  --property=WorkingDirectory="${patch_root}" \
  --property=Restart=no \
  --setenv=PYTHONUNBUFFERED=1 \
  -- "${python_bin}" "${entry}" --workspace "${workspace}"
