#!/usr/bin/env bash
# Run the paired 175-task frontier ceiling as a declared Qwen3.7-Max text
# family pool. Exact model identities remain in separate journals. The global
# K=10 allocation is 3+2+3+2, identically applied to both enrichment arms.
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
runner="${patch_root}/frontier_passk.py"
credential_wrapper="${patch_root}/run_with_qwen_secondary_env.sh"
pool_contract="${patch_root}/qwen37_pooled_contract.json"
expected_pool_contract_sha256="68ec0e3f2f84b92ebbfd5169636914822fb06c422946e85aa68c5e5adbb6925b"
run_root="${workspace}/artifacts/frontier_ceiling_two_enrichments/runs"

actual_pool_contract_sha256="$(sha256sum "${pool_contract}" | awk '{print $1}')"
if [[ "${actual_pool_contract_sha256}" != "${expected_pool_contract_sha256}" ]]; then
  printf 'Qwen pool contract hash mismatch: expected=%s actual=%s\n' \
    "${expected_pool_contract_sha256}" "${actual_pool_contract_sha256}" >&2
  exit 2
fi

models=(
  "qwen3.7-max-2026-05-17"
  "qwen3.7-max-preview"
  "qwen3.7-max-2026-05-20"
  "qwen3.7-max-2026-06-08"
)
model_slugs=("0517" "preview" "0520" "0608")
model_k=(3 2 3 2)

common_args=(
  --provider qwen
  --arm compact
  --input-mode prematerialized_f2
  --workers 1
  --max-output-tokens 12288
  --max-prompt-tokens 12000
  --chat-overhead-reserve 256
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
  --tokenizer-json "${workspace}/.hf_home/hub/models--Qwen--Qwen3-8B/blobs/aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
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
  k="${model_k[index]}"
  for arm in opus codex; do
    mapfile -d '' -t selected_arm_args < <(arm_args "${arm}")
    out="${run_root}/qwen37_pool_${slug}_${arm}_k${k}_tb8k"
    unit="frontier-qwen37-${slug}-${arm}-k${k}-tb8k"
    command=(
      "${credential_wrapper}"
      "${python_bin}"
      "${runner}"
      "${common_args[@]}"
      --model "${model}"
      --k "${k}"
      "${selected_arm_args[@]}"
      --out "${out}"
    )

    case "${action}" in
      preflight)
        "${command[@]}" --preflight-only
        if [[ -f "${out}/qwen37_pooled_contract.json" ]]; then
          existing_contract_sha256="$(
            sha256sum "${out}/qwen37_pooled_contract.json" | awk '{print $1}'
          )"
          if [[ "${existing_contract_sha256}" != "${expected_pool_contract_sha256}" ]]; then
            printf 'existing output has a different Qwen pool contract: %s\n' \
              "${out}" >&2
            exit 2
          fi
        else
          install -m 0444 "${pool_contract}" "${out}/qwen37_pooled_contract.json"
        fi
        ;;
      start)
        if [[ ! -f "${out}/qwen37_pooled_contract.json" ]]; then
          printf 'run preflight before start: %s\n' "${out}" >&2
          exit 2
        fi
        existing_contract_sha256="$(
          sha256sum "${out}/qwen37_pooled_contract.json" | awk '{print $1}'
        )"
        if [[ "${existing_contract_sha256}" != "${expected_pool_contract_sha256}" ]]; then
          printf 'output Qwen pool contract hash mismatch: %s\n' "${out}" >&2
          exit 2
        fi
        if systemctl is-active --quiet "${unit}.service"; then
          printf 'ALREADY_RUNNING unit=%s model=%s arm=%s k=%s\n' \
            "${unit}" "${model}" "${arm}" "${k}"
          continue
        fi
        systemd-run \
          --unit="${unit}" \
          --description="Qwen3.7 pooled frontier ${model} ${arm} K=${k}" \
          --property=WorkingDirectory="${patch_root}" \
          --property=Restart=no \
          --setenv=PYTHONUNBUFFERED=1 \
          -- "${command[@]}"
        ;;
      status)
        active="$(systemctl is-active "${unit}.service" 2>/dev/null || true)"
        rows=0
        if [[ -f "${out}/attempts.jsonl" ]]; then
          rows="$(wc -l < "${out}/attempts.jsonl")"
        fi
        printf 'QWEN37_POOL_STATUS unit=%s state=%s model=%s arm=%s k=%s attempts=%s out=%s\n' \
          "${unit}" "${active}" "${model}" "${arm}" "${k}" "${rows}" "${out}"
        ;;
    esac
  done
done
