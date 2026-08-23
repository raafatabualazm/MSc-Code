#!/usr/bin/env bash
# Launch the clean Qwen3.7 primary estimator: K=5 from the pinned 2026-05-17
# model and K=5 from its documented preview alias, identically for both arms.
# The Qwen-only entry sends max_completion_tokens=12288 and never max_tokens.
set -euo pipefail

action="${1:-}"
case "${action}" in
  preflight|start|status)
    ;;
  *)
    printf 'usage: %s {preflight|start|status}\n' "$0" >&2
    exit 2
    ;;
esac

workspace="${WORKSPACE:-/workspace}"
python_bin="${PYTHON_BIN:-${workspace}/.venv/bin/python}"
patch_root="${workspace}/frontier_ceiling_patch_v1"
qwen_entry="${patch_root}/frontier_passk_qwen_completion.py"
shared_runner="${patch_root}/frontier_passk.py"
core="${patch_root}/frontier_core.py"
f2_codec="${patch_root}/frontier_f2.py"
credential_wrapper="${patch_root}/run_with_qwen_secondary_env.sh"
primary_contract="${patch_root}/qwen37_primary_clean_contract_v3.json"
run_root="${workspace}/artifacts/frontier_ceiling_two_enrichments/runs"
tokenizer_json="${workspace}/.hf_home/hub/models--Qwen--Qwen3-8B/blobs/aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

expected_qwen_entry_sha256="a1115b69abed4db79e6c39db2e1cd83c8175f33a6cb8a5d749ec646826bc5709"
expected_shared_runner_sha256="8d3e3ad160d9ed389a9e212dacb76556ab7af59f1559418d45d9802402d9dead"
expected_core_sha256="f502e958a6fa3fb564d17327c2c4c77bc9cf4f5182546235970b1a4498a60258"
expected_f2_codec_sha256="097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce"
expected_credential_wrapper_sha256="95d2cfeeef2eb7fbf35099fad179320a535bba5316b6226517165965ae48516d"
expected_primary_contract_sha256="b7b3892389fc8cef3618c4153ed562d1460a2b4ca07348708afa2d8cf9ce256f"
expected_tokenizer_sha256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

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

require_sha256 "${qwen_entry}" "${expected_qwen_entry_sha256}" "Qwen entry"
require_sha256 \
  "${shared_runner}" "${expected_shared_runner_sha256}" "shared frontier runner"
require_sha256 "${core}" "${expected_core_sha256}" "frontier core"
require_sha256 "${f2_codec}" "${expected_f2_codec_sha256}" "F2 codec"
require_sha256 \
  "${credential_wrapper}" \
  "${expected_credential_wrapper_sha256}" \
  "Qwen credential wrapper"
require_sha256 \
  "${primary_contract}" \
  "${expected_primary_contract_sha256}" \
  "Qwen clean primary contract"
require_sha256 \
  "${tokenizer_json}" "${expected_tokenizer_sha256}" "Qwen tokenizer"

models=(
  "qwen3.7-max-2026-05-17"
  "qwen3.7-max-preview"
)
model_slugs=("0517" "preview")

common_args=(
  --provider qwen
  --arm compact
  --input-mode prematerialized_f2
  --workers 1
  --max-output-tokens 12288
  --max-prompt-tokens 12000
  --chat-overhead-reserve 256
  --budget 0
  --temperature 0.8
  --top-p 0.95
  --timeout-seconds 1800
  --max-attempts-per-sample 6
  --retry-base-seconds 2
  --retry-max-seconds 30
  --eval-timeout-seconds 30
  --eval-stability-runs 2
  --expected-task-count 175
  --extra-body-json '{"enable_thinking":true,"thinking_budget":8192}'
  --pair-manifest "${workspace}/artifacts/frontier_ceiling_two_enrichments/pair_manifest.json"
  --expected-pair-manifest-sha256 35f4cfcaf0732928312bed3f2f27c3f3e347525c0076921caeab7ee6539c132e
  --evaluator-module "${workspace}/hybrid_training_patch_v2_3/scripts/evaluation/graph_compile_at_k_antigravity.py"
  --expected-evaluator-sha256 249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6
  --dart /usr/lib/dart/bin/dart
  --expected-dart-sha256 c03ad868b5c53e31461b0fef22dc6eb6aeb56b7567efff6ca488ce9c4a6f8a6a
  --tokenizer-json "${tokenizer_json}"
  --qwen-env-file "${workspace}/Qwen.env"
  --resume
)

arm_args() {
  local arm="$1"
  case "${arm}" in
    opus)
      printf '%s\0' \
        --dataset-label opus_real_fn0_cfg_175 \
        --prompt-jsonl "${workspace}/artifacts/frontier_ceiling_two_enrichments/opus/opus_real175_f2.jsonl" \
        --prompt-manifest "${workspace}/artifacts/frontier_ceiling_two_enrichments/opus/opus_real175_f2.jsonl.manifest.json" \
        --eval-jsonl "${workspace}/artifacts/frontier_ceiling_two_enrichments/opus/dev_fn0_real.jsonl" \
        --eval-seal "${workspace}/artifacts/frontier_ceiling_two_enrichments/opus/dev_fn0_real.seal.json" \
        --pair-arm-key opus_real_fn0_cfg \
        --expected-prompt-jsonl-sha256 4aae71997aa98b4a273fdedca17d1df2266f18dd5a03fe164b9cf81e342648cd \
        --expected-prompt-manifest-sha256 35e25fa9d7a2bd813b6aec55a1149304d4dd160c82b27b691f27c4cb0bd6068b \
        --expected-eval-jsonl-sha256 a4ed1cf185d52c3d212e2d7348fdb2a1dffd0035f4c395e2e897fd072fa70001 \
        --expected-eval-seal-sha256 2909d279d7c87279b5b0e59cdcd7598742b25a2bd111382f6c8216103f906799
      ;;
    codex)
      printf '%s\0' \
        --dataset-label codex_multifunction_cfg_175 \
        --prompt-jsonl "${workspace}/multifunction_v1/build/dev_multifunction_binary_f2.jsonl" \
        --prompt-manifest "${workspace}/multifunction_v1/build/dev_multifunction_binary_f2.jsonl.manifest.json" \
        --eval-jsonl "${workspace}/multifunction_v1/build/dev_multifunction_binary.jsonl" \
        --eval-seal "${workspace}/multifunction_v1/build/dev_multifunction_binary.seal.json" \
        --pair-arm-key codex_multifunction_cfg \
        --expected-prompt-jsonl-sha256 6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab \
        --expected-prompt-manifest-sha256 777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 \
        --expected-eval-jsonl-sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
        --expected-eval-seal-sha256 5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a
      ;;
    *)
      return 2
      ;;
  esac
}

for index in "${!models[@]}"; do
  model="${models[index]}"
  slug="${model_slugs[index]}"
  for arm in opus codex; do
    mapfile -d '' -t selected_arm_args < <(arm_args "${arm}")
    out="${run_root}/qwen37_clean_v3_${slug}_${arm}_k5_mc12k_tb8k"
    unit="frontier-qwen37-clean-v3-${slug}-${arm}-k5-mc12k-tb8k"
    command=(
      "${credential_wrapper}"
      "${python_bin}"
      "${qwen_entry}"
      "${common_args[@]}"
      --model "${model}"
      --k 5
      "${selected_arm_args[@]}"
      --out "${out}"
    )

    case "${action}" in
      preflight)
        "${command[@]}" --preflight-only
        if [[ -f "${out}/qwen37_primary_clean_contract_v3.json" ]]; then
          existing_contract_sha256="$(
            sha256sum "${out}/qwen37_primary_clean_contract_v3.json" |
              awk '{print $1}'
          )"
          if [[ "${existing_contract_sha256}" != "${expected_primary_contract_sha256}" ]]; then
            printf 'existing output has a different clean-v3 contract: %s\n' \
              "${out}" >&2
            exit 2
          fi
        else
          install -m 0444 \
            "${primary_contract}" \
            "${out}/qwen37_primary_clean_contract_v3.json"
        fi
        ;;
      start)
        if [[ ! -f "${out}/qwen37_primary_clean_contract_v3.json" ]]; then
          printf 'run preflight before start: %s\n' "${out}" >&2
          exit 2
        fi
        existing_contract_sha256="$(
          sha256sum "${out}/qwen37_primary_clean_contract_v3.json" |
            awk '{print $1}'
        )"
        if [[ "${existing_contract_sha256}" != "${expected_primary_contract_sha256}" ]]; then
          printf 'output clean-v3 contract hash mismatch: %s\n' "${out}" >&2
          exit 2
        fi
        if systemctl is-active --quiet "${unit}.service"; then
          printf 'ALREADY_RUNNING unit=%s model=%s arm=%s k=5\n' \
            "${unit}" "${model}" "${arm}"
          continue
        fi
        systemd-run \
          --unit="${unit}" \
          --description="Qwen3.7 clean-v3 ${model} ${arm} K=5 max_completion_tokens=12288" \
          --property=WorkingDirectory="${patch_root}" \
          --property=Restart=no \
          --setenv=PYTHONUNBUFFERED=1 \
          -- "${command[@]}"
        ;;
      status)
        active="$(systemctl is-active "${unit}.service" 2>/dev/null || true)"
        rows=0
        terminal=0
        if [[ -f "${out}/attempts.jsonl" ]]; then
          rows="$(wc -l < "${out}/attempts.jsonl")"
          terminal="$(
            grep -Ec '"slot_terminal"[[:space:]]*:[[:space:]]*true' \
              "${out}/attempts.jsonl" || true
          )"
        fi
        printf 'QWEN37_CLEAN_V3_STATUS unit=%s state=%s model=%s arm=%s k=5 attempts=%s terminal=%s out=%s\n' \
          "${unit}" "${active}" "${model}" "${arm}" \
          "${rows}" "${terminal}" "${out}"
        ;;
    esac
  done
done
