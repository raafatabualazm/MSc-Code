#!/usr/bin/env bash
# Launch one audited v6 capacity epoch.  Action-to-route flags are constructed
# once and reused byte-for-byte by preflight, run, and status.
set -euo pipefail

action="${1:-}"
case "${action}" in
  preflight|start|status|\
  preflight-current-preview|start-current-preview|status-current-preview|\
  preflight-current-0517|start-current-0517|status-current-0517|\
  preflight-current-generic|start-current-generic|status-current-generic|\
  preflight-fallback|start-fallback|status-fallback)
    ;;
  *)
    printf 'usage: %s {preflight|start|status|preflight-current-preview|start-current-preview|status-current-preview|preflight-current-0517|start-current-0517|status-current-0517|preflight-current-generic|start-current-generic|status-current-generic|preflight-fallback|start-fallback|status-fallback}\n' "$0" >&2
    exit 2
    ;;
esac

workspace="${WORKSPACE:-/workspace}"
python_bin="${PYTHON_BIN:-${workspace}/.venv/bin/python}"
patch_root="${workspace}/frontier_ceiling_patch_v1"
entry="${patch_root}/qwen37_capacity_fallback_v6.py"
contract="${patch_root}/qwen37_capacity_fallback_contract_v6.json"
extension_contract="${patch_root}/qwen37_current_five_capacity_extension_v1.json"
secondary_wrapper="${patch_root}/run_with_qwen_secondary_env.sh"
generic_wrapper="${patch_root}/run_with_qwen_generic_env_v6.sh"
run_root="${workspace}/artifacts/frontier_ceiling_two_enrichments/runs"

operation="${action%%-*}"
stage="current-dated"
wrapper="${secondary_wrapper}"
capacity_epoch="${CAPACITY_EPOCH:-secondary-workspace-20260726-epoch1}"
credential_source="secondary_qwen_env"
credential_env_file="/workspace/Qwen.env"
route_flags=()
unit_stage=""
if [[ "${action}" == *-current-preview ]]; then
  stage="current-preview"
  capacity_epoch="${CAPACITY_EPOCH:-secondary-workspace-20260726-preview-epoch1}"
  route_flags=(--allow-preview)
  unit_stage="-preview"
elif [[ "${action}" == *-current-0517 ]]; then
  stage="current-0517"
  capacity_epoch="${CAPACITY_EPOCH:-secondary-workspace-20260726-0517-epoch1}"
  route_flags=(--allow-preview --include-source-alias)
  unit_stage="-0517"
elif [[ "${action}" == *-current-generic ]]; then
  stage="current-generic"
  capacity_epoch="${CAPACITY_EPOCH:-secondary-workspace-20260726-generic-epoch1}"
  route_flags=(--include-undated-alias --only-undated-alias)
  unit_stage="-generic"
elif [[ "${action}" == *-fallback ]]; then
  stage="fallback-all-five"
  wrapper="${generic_wrapper}"
  capacity_epoch="${CAPACITY_EPOCH:-fallback-workspace-20260726-epoch1}"
  credential_source="generic_fallback_env"
  credential_env_file="/workspace/Qwen_fallback.env"
  export QWEN_GENERIC_ENV_FILE="${credential_env_file}"
  route_flags=(
    --allow-preview
    --include-undated-alias
    --include-source-alias
  )
  unit_stage="-fallback"
fi

expected_entry_sha256="6b2d642be25bb7b2e97daddf70e9a2245a8ff09e5f0c6e5e32c09afc92159521"
expected_contract_sha256="cea8acaa785ddc2685a5da8b4426dce41837a25af2e7dd9639dd70f632d59631"
expected_extension_contract_sha256="db4ee883e5073f08a0dad160e7e5ea594c54a3a3dd0c31a68fe4479aea1536ef"
expected_wrapper_sha256="95d2cfeeef2eb7fbf35099fad179320a535bba5316b6226517165965ae48516d"
expected_generic_wrapper_sha256="203a0abbd897867252de39f4523bf54037b175a30fe4bb5d4ea11386fdb8e8bc"
expected_runner_sha256="8d3e3ad160d9ed389a9e212dacb76556ab7af59f1559418d45d9802402d9dead"
expected_core_sha256="f502e958a6fa3fb564d17327c2c4c77bc9cf4f5182546235970b1a4498a60258"
expected_qwen_entry_sha256="5055eabac3898d529beb6209b3792256378d509239265cb44eaa2cf7f46b5e15"

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

require_sha256 "${entry}" "${expected_entry_sha256}" "capacity entry"
require_sha256 "${contract}" "${expected_contract_sha256}" "capacity contract"
require_sha256 \
  "${extension_contract}" "${expected_extension_contract_sha256}" \
  "current-five capacity extension contract"
require_sha256 \
  "${secondary_wrapper}" "${expected_wrapper_sha256}" \
  "secondary credential wrapper"
require_sha256 \
  "${generic_wrapper}" "${expected_generic_wrapper_sha256}" \
  "generic credential wrapper"
require_sha256 \
  "${patch_root}/frontier_passk.py" "${expected_runner_sha256}" \
  "frontier runner"
require_sha256 \
  "${patch_root}/frontier_core.py" "${expected_core_sha256}" \
  "frontier core"
require_sha256 \
  "${patch_root}/frontier_passk_qwen_completion.py" \
  "${expected_qwen_entry_sha256}" "Qwen completion entry"

for partition in 0520 0608; do
  for arm in opus codex; do
    out="${run_root}/qwen37_capacity_v6_${partition}_${arm}_mc12k_tb8k"
    unit="frontier-qwen37-capacity-v6-${partition}-${arm}${unit_stage}-mc12k-tb8k"
    common=(
      --workspace "${workspace}"
      --out "${out}"
      --arm "${arm}"
      --partition "${partition}"
      --capacity-epoch "${capacity_epoch}"
      --credential-source "${credential_source}"
      --credential-env-file "${credential_env_file}"
      "${route_flags[@]}"
    )
    case "${operation}" in
      preflight)
        "${wrapper}" "${python_bin}" "${entry}" preflight "${common[@]}"
        ;;
      start)
        if [[ ! -f "${out}/${contract##*/}" ]]; then
          printf 'run matching preflight before start: %s stage=%s\n' \
            "${out}" "${stage}" >&2
          exit 2
        fi
        require_sha256 \
          "${out}/${contract##*/}" "${expected_contract_sha256}" \
          "copied capacity contract"
        if systemctl is-active --quiet "${unit}.service"; then
          printf 'ALREADY_RUNNING unit=%s arm=%s partition=%s stage=%s\n' \
            "${unit}" "${arm}" "${partition}" "${stage}"
          continue
        fi
        command=(
          "${wrapper}" "${python_bin}" "${entry}" run
          "${common[@]}"
          --poll-seconds 10
          --max-idle-cycles 360
        )
        if [[ -n "${MAX_NEW:-}" ]]; then
          command+=(--max-new "${MAX_NEW}")
        fi
        systemctl reset-failed "${unit}.service" 2>/dev/null || true
        systemd-run \
          --unit="${unit}" \
          --description="Qwen3.7 v6 capacity ${stage} ${partition} ${arm}" \
          --property=WorkingDirectory="${patch_root}" \
          --property=Restart=no \
          --setenv=PYTHONUNBUFFERED=1 \
          -- "${command[@]}"
        ;;
      status)
        "${wrapper}" "${python_bin}" "${entry}" status "${common[@]}"
        ;;
    esac
  done
done
