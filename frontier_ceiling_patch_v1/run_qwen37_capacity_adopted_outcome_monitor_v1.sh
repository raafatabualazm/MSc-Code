#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
case "${action}" in
  once|start|status) ;;
  *)
    printf 'usage: %s {once|start|status}\n' "$0" >&2
    exit 2
    ;;
esac

workspace="${WORKSPACE:-/workspace}"
python_bin="${PYTHON_BIN:-${workspace}/.venv/bin/python}"
patch_root="${workspace}/frontier_ceiling_patch_v1"
entry="${patch_root}/qwen37_capacity_adopted_outcome_monitor_v1.py"
reconciliation_entry="${patch_root}/qwen37_capacity_adopted_outcome_reconciliation_v1.py"
contract="${patch_root}/qwen37_capacity_adopted_outcome_reconciliation_contract_v1.json"
extension="${patch_root}/qwen37_capacity_adopted_outcome_extension_v1.json"
unit="frontier-qwen37-capacity-adopted-outcome-monitor-v1"

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    printf '%s hash mismatch: expected=%s actual=%s path=%s\n' \
      "${label}" "${expected}" "${actual}" "${path}" >&2
    exit 2
  fi
}

require_sha256 \
  "${entry}" \
  "836835ceccfd08c93d2a6f97c7c1a484b9a82e9343d7aca1e5001dc29109b446" \
  "monitor entry"
require_sha256 \
  "${reconciliation_entry}" \
  "a5bc531c71652d0fb881d748863ad91c08f8eb155905250226dc0507346e7457" \
  "reconciliation entry"
require_sha256 \
  "${contract}" \
  "344ac01c949f76e351cb1040d018d075875dcfbc2b08bc16483f9a28661c2b74" \
  "reconciliation contract"
require_sha256 \
  "${extension}" \
  "454f8e7330d35545d04359580a736253da4ec59c78ab62fc96a71ff8dadb7a19" \
  "reconciliation extension"

if [[ "${action}" == "once" ]]; then
  exec "${python_bin}" "${entry}" --workspace "${workspace}" --once
fi

if [[ "${action}" == "status" ]]; then
  systemctl show "${unit}.service" \
    --property=LoadState \
    --property=ActiveState \
    --property=SubState \
    --property=MainPID \
    --property=Result \
    --property=ExecMainStatus
  exit 0
fi

if systemctl is-active --quiet "${unit}.service"; then
  systemctl show "${unit}.service" \
    --property=ActiveState \
    --property=SubState \
    --property=MainPID
  exit 0
fi

systemctl reset-failed "${unit}.service" >/dev/null 2>&1 || true
systemd-run \
  --unit="${unit}" \
  --description="Provider-free Qwen adopted-outcome reconciliation monitor" \
  --property=Type=simple \
  --property=Restart=on-failure \
  --property=RestartSec=15s \
  --property=StartLimitIntervalSec=300 \
  --property=StartLimitBurst=10 \
  "${python_bin}" "${entry}" \
  --workspace "${workspace}" \
  --interval-seconds 15
