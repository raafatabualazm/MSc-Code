#!/usr/bin/env bash
# Run one outcome-blind capacity-length repair endpoint epoch.
set -euo pipefail

action="${1:-}"
case "${action}" in
  preflight-current|once-current|preflight-dated|once-dated|\
  preflight-preview|once-preview|preflight-source|once-source|\
  preflight-generic|once-generic|\
  preflight-fallback|once-fallback|status)
    ;;
  *)
    printf 'usage: %s {preflight-dated|once-dated|preflight-preview|once-preview|preflight-source|once-source|preflight-generic|once-generic|preflight-fallback|once-fallback|status}\n' "$0" >&2
    exit 2
    ;;
esac

workspace="${WORKSPACE:-/workspace}"
python_bin="${PYTHON_BIN:-${workspace}/.venv/bin/python}"
patch_root="${workspace}/frontier_ceiling_patch_v1"
entry="${patch_root}/qwen37_capacity_length_repair_v7.py"
contract="${patch_root}/qwen37_capacity_length_contract_v7.json"
capacity_entry="${patch_root}/qwen37_capacity_fallback_v6.py"
capacity_contract="${patch_root}/qwen37_capacity_fallback_contract_v6.json"
extension_contract="${patch_root}/qwen37_current_five_capacity_extension_v1.json"
secondary_wrapper="${patch_root}/run_with_qwen_secondary_env.sh"
fallback_wrapper="${patch_root}/run_with_qwen_generic_env_v6.sh"
run_root="${workspace}/artifacts/frontier_ceiling_two_enrichments/runs"
only_partition="${V7_ONLY_PARTITION:-}"
only_arm="${V7_ONLY_ARM:-}"
if [[ -n "${only_partition}" && "${only_partition}" != "0520" && "${only_partition}" != "0608" ]]; then
  printf 'invalid V7_ONLY_PARTITION: %s\n' "${only_partition}" >&2
  exit 2
fi
if [[ -n "${only_arm}" && "${only_arm}" != "opus" && "${only_arm}" != "codex" ]]; then
  printf 'invalid V7_ONLY_ARM: %s\n' "${only_arm}" >&2
  exit 2
fi

expected_entry_sha256="a74cb5595032c5d2b1d7fc325e38903ccf4f8fa844e8caf0f15f30a853ec19a8"
expected_contract_sha256="b69b5cf91f33e785f78e965eb67c814372fa774ded4d19114eddf49ba9149809"
expected_capacity_entry_sha256="6b2d642be25bb7b2e97daddf70e9a2245a8ff09e5f0c6e5e32c09afc92159521"
expected_capacity_contract_sha256="cea8acaa785ddc2685a5da8b4426dce41837a25af2e7dd9639dd70f632d59631"
expected_extension_contract_sha256="db4ee883e5073f08a0dad160e7e5ea594c54a3a3dd0c31a68fe4479aea1536ef"
expected_secondary_wrapper_sha256="95d2cfeeef2eb7fbf35099fad179320a535bba5316b6226517165965ae48516d"
expected_fallback_wrapper_sha256="203a0abbd897867252de39f4523bf54037b175a30fe4bb5d4ea11386fdb8e8bc"

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

require_sha256 "${entry}" "${expected_entry_sha256}" "v7 repair entry"
require_sha256 "${contract}" "${expected_contract_sha256}" "v7 contract"
require_sha256 \
  "${capacity_entry}" "${expected_capacity_entry_sha256}" \
  "v6 capacity entry"
require_sha256 \
  "${capacity_contract}" "${expected_capacity_contract_sha256}" \
  "v6 capacity contract"
require_sha256 \
  "${extension_contract}" "${expected_extension_contract_sha256}" \
  "current-five capacity extension contract"
require_sha256 \
  "${secondary_wrapper}" "${expected_secondary_wrapper_sha256}" \
  "secondary credential wrapper"
require_sha256 \
  "${fallback_wrapper}" "${expected_fallback_wrapper_sha256}" \
  "fallback credential wrapper"

operation="${action%%-*}"
stage="${action#*-}"
if [[ "${action}" == "status" ]]; then
  operation="status"
  stage="all"
elif [[ "${stage}" == "current" ]]; then
  stage="dated"
fi

wrapper="${secondary_wrapper}"
suffix=""
repair_epoch=""
allowed_models=()
case "${stage}" in
  dated)
    suffix="secondary_dated"
    repair_epoch="${REPAIR_EPOCH:-secondary-workspace-20260726-dated-repair1}"
    allowed_models=(
      qwen3.7-max-2026-05-20
      qwen3.7-max-2026-06-08
    )
    ;;
  preview)
    suffix="secondary_preview"
    repair_epoch="${REPAIR_EPOCH:-secondary-workspace-20260726-preview-repair1}"
    allowed_models=(qwen3.7-max-preview)
    ;;
  source)
    suffix="secondary_0517"
    repair_epoch="${REPAIR_EPOCH:-secondary-workspace-20260726-0517-repair1}"
    allowed_models=(qwen3.7-max-2026-05-17)
    ;;
  generic)
    suffix="secondary_generic"
    repair_epoch="${REPAIR_EPOCH:-secondary-workspace-20260726-generic-repair1}"
    allowed_models=(qwen3.7-max)
    ;;
  fallback)
    wrapper="${fallback_wrapper}"
    export QWEN_GENERIC_ENV_FILE="${QWEN_GENERIC_ENV_FILE:-/workspace/Qwen_fallback.env}"
    suffix="fallback_all_five"
    repair_epoch="${REPAIR_EPOCH:-fallback-workspace-20260726-repair1}"
    allowed_models=(
      qwen3.7-max-2026-05-17
      qwen3.7-max-2026-05-20
      qwen3.7-max-2026-06-08
      qwen3.7-max-preview
      qwen3.7-max
    )
    ;;
  all)
    ;;
  *)
    printf 'invalid v7 repair stage: %s\n' "${stage}" >&2
    exit 2
    ;;
esac

for partition in 0520 0608; do
  if [[ -n "${only_partition}" && "${partition}" != "${only_partition}" ]]; then
    continue
  fi
  for arm in opus codex; do
    if [[ -n "${only_arm}" && "${arm}" != "${only_arm}" ]]; then
      continue
    fi
    capacity_out="${run_root}/qwen37_capacity_v6_${partition}_${arm}_mc12k_tb8k"
    if [[ "${operation}" == "status" ]]; then
      for status_suffix in secondary_dated secondary_preview secondary_0517 secondary_generic fallback_all_five; do
        status_out="${run_root}/qwen37_capacity_length_v7_${partition}_${arm}_repair_epoch_${status_suffix}"
        provenance="absent"
        attempts=0
        outcomes=0
        failure="absent"
        summary="absent"
        [[ -f "${status_out}/provenance.json" ]] && provenance="present"
        [[ -f "${status_out}/failure.json" ]] && failure="present"
        [[ -f "${status_out}/summary.json" ]] && summary="present"
        [[ -f "${status_out}/repair_attempts.jsonl" ]] && \
          attempts="$(wc -l < "${status_out}/repair_attempts.jsonl")"
        [[ -f "${status_out}/repair_outcomes.jsonl" ]] && \
          outcomes="$(wc -l < "${status_out}/repair_outcomes.jsonl")"
        printf 'QWEN37_CAPACITY_LENGTH_V7 partition=%s arm=%s epoch=%s provenance=%s attempts=%s outcomes=%s failure=%s summary=%s out=%s\n' \
          "${partition}" "${arm}" "${status_suffix}" "${provenance}" \
          "${attempts}" "${outcomes}" "${failure}" "${summary}" \
          "${status_out}"
      done
      continue
    fi

    out="${run_root}/qwen37_capacity_length_v7_${partition}_${arm}_repair_epoch_${suffix}"
    prior_args=()
    prior_suffixes=()
    case "${stage}" in
      dated)
        ;;
      preview)
        prior_suffixes=(secondary_dated)
        ;;
      source)
        prior_suffixes=(secondary_dated secondary_preview)
        ;;
      generic)
        prior_suffixes=(secondary_dated secondary_preview secondary_0517)
        ;;
      fallback)
        prior_suffixes=(
          secondary_dated
          secondary_preview
          secondary_0517
          secondary_generic
        )
        ;;
    esac
    for prior_suffix in "${prior_suffixes[@]}"; do
      prior="${run_root}/qwen37_capacity_length_v7_${partition}_${arm}_repair_epoch_${prior_suffix}"
      if [[ ! -f "${prior}/provenance.json" ]]; then
        printf 'prior repair epoch was not preflighted: %s\n' "${prior}" >&2
        exit 2
      fi
      prior_args+=(--prior-repair-out "${prior}")
    done
    model_args=()
    for model in "${allowed_models[@]}"; do
      model_args+=(--allowed-model "${model}")
    done
    command=(
      "${wrapper}"
      "${python_bin}"
      "${entry}"
      --workspace "${workspace}"
      --capacity-out "${capacity_out}"
      --contract "${contract}"
      --expected-contract-sha256 "${expected_contract_sha256}"
      --expected-script-sha256 "${expected_entry_sha256}"
      --expected-capacity-contract-sha256 "${expected_capacity_contract_sha256}"
      --expected-capacity-script-sha256 "${expected_capacity_entry_sha256}"
      --out "${out}"
      --repair-epoch "${repair_epoch}"
      "${model_args[@]}"
      "${prior_args[@]}"
    )
    case "${operation}" in
      preflight)
        "${command[@]}" --preflight-only
        ;;
      once)
        if [[ ! -f "${out}/preflight.json" ]]; then
          printf 'run matching v7 preflight first: %s\n' "${out}" >&2
          exit 2
        fi
        "${command[@]}" --once
        ;;
      *)
        printf 'invalid v7 repair operation: %s\n' "${operation}" >&2
        exit 2
        ;;
    esac
  done
done
