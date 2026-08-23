#!/usr/bin/env bash
# Sealed GPT-5.6 Sol/Terra K=2 frontier ceiling over both 175-task F2 arms.
# Defaults to API-free preflight. Paid Batch creation requires BOTH
# OPENAI56_AUTHORIZE_PAID_BATCH=1 and an explicit per-job cost cap.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/workspace/.venv/bin/python}"
PATCH_DIR="/workspace/frontier_ceiling_patch_v1"
RUNNER="${RUNNER:-${PATCH_DIR}/openai56_batch_fasttrack.py}"
PAIR_ROOT="/workspace/artifacts/frontier_ceiling_two_enrichments"
RUN_ROOT="${RUN_ROOT:-${PAIR_ROOT}/runs/openai56_batch_k2_32k_max_authorized_v1}"
PAIR_MANIFEST="${PAIR_ROOT}/pair_manifest.json"
PAIR_SHA="35f4cfcaf0732928312bed3f2f27c3f3e347525c0076921caeab7ee6539c132e"
EVALUATOR="/workspace/hybrid_training_patch_v2_3/scripts/evaluation/graph_compile_at_k_antigravity.py"
EVALUATOR_SHA="249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
DART_BIN="/usr/lib/dart/bin/dart"
DART_SHA="c03ad868b5c53e31461b0fef22dc6eb6aeb56b7567efff6ca488ce9c4a6f8a6a"
OPENAI_ENV_FILE="${OPENAI_ENV_FILE:-/workspace/OpenAI.env}"

ACTION="${ACTION:-preflight}"
# Terra is the authorized default. Sol remains available for an explicitly
# separate, separately capped run.
MODEL="${MODEL:-terra}"
ARM="${ARM:-both}"
WORKERS="${WORKERS:-32}"
COUNT_WORKERS="${COUNT_WORKERS:-12}"
# Two same-model arms may be active together. 2 x 700K remains below the
# published 1.5M Tier-1 queued-input allowance.
SHARD_INPUT_TOKEN_CAP="${SHARD_INPUT_TOKEN_CAP:-700000}"

case "${ACTION}" in
  preflight|submit|status|harvest|auto) ;;
  *) echo "ACTION must be preflight, submit, status, harvest, or auto" >&2; exit 2 ;;
esac
case "${MODEL}" in
  sol|terra|both) ;;
  *) echo "MODEL must be sol, terra, or both" >&2; exit 2 ;;
esac
case "${ARM}" in
  opus|codex|both) ;;
  *) echo "ARM must be opus, codex, or both" >&2; exit 2 ;;
esac

if [[ "${ACTION}" != "preflight" ]]; then
  if [[ -f "${OPENAI_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${OPENAI_ENV_FILE}"
    set +a
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for ${ACTION}" >&2
    exit 2
  fi
fi

run_job() {
  local model_id="$1" arm_name="$2"
  local pair_key eval_jsonl eval_sha eval_seal eval_seal_sha
  local prompt_jsonl prompt_sha prompt_manifest prompt_manifest_sha

  if [[ "${arm_name}" == "opus" ]]; then
    pair_key="opus_real_fn0_cfg"
    eval_jsonl="${PAIR_ROOT}/opus/dev_fn0_real.jsonl"
    eval_sha="a4ed1cf185d52c3d212e2d7348fdb2a1dffd0035f4c395e2e897fd072fa70001"
    eval_seal="${PAIR_ROOT}/opus/dev_fn0_real.seal.json"
    eval_seal_sha="2909d279d7c87279b5b0e59cdcd7598742b25a2bd111382f6c8216103f906799"
    prompt_jsonl="${PAIR_ROOT}/opus/opus_real175_f2.jsonl"
    prompt_sha="4aae71997aa98b4a273fdedca17d1df2266f18dd5a03fe164b9cf81e342648cd"
    prompt_manifest="${PAIR_ROOT}/opus/opus_real175_f2.jsonl.manifest.json"
    prompt_manifest_sha="35e25fa9d7a2bd813b6aec55a1149304d4dd160c82b27b691f27c4cb0bd6068b"
  else
    pair_key="codex_multifunction_cfg"
    eval_jsonl="/workspace/multifunction_v1/build/dev_multifunction_binary.jsonl"
    eval_sha="abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7"
    eval_seal="/workspace/multifunction_v1/build/dev_multifunction_binary.seal.json"
    eval_seal_sha="5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a"
    prompt_jsonl="/workspace/multifunction_v1/build/dev_multifunction_binary_f2.jsonl"
    prompt_sha="6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab"
    prompt_manifest="/workspace/multifunction_v1/build/dev_multifunction_binary_f2.jsonl.manifest.json"
    prompt_manifest_sha="777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44"
  fi

  local model_tag="${model_id#gpt-5.6-}"
  local cap_var
  if [[ "${model_tag}" == "sol" ]]; then
    cap_var="${OPENAI56_SOL_JOB_COST_CAP_USD:-}"
  else
    cap_var="${OPENAI56_TERRA_JOB_COST_CAP_USD:-}"
  fi

  local paid_args=()
  if [[ "${ACTION}" == "submit" || "${ACTION}" == "auto" ]]; then
    [[ "${OPENAI56_AUTHORIZE_PAID_BATCH:-0}" == "1" ]] || {
      echo "Set OPENAI56_AUTHORIZE_PAID_BATCH=1 for ${ACTION}" >&2
      exit 2
    }
    [[ -n "${cap_var}" ]] || {
      echo "Set OPENAI56_${model_tag^^}_JOB_COST_CAP_USD for ${ACTION}" >&2
      exit 2
    }
    paid_args+=(--authorize-paid-batch --job-cost-cap-usd "${cap_var}")
  fi

  "${PYTHON_BIN}" "${RUNNER}" \
    --action "${ACTION}" \
    --model "${model_id}" \
    --attest-authorized-benchmark \
    --input-token-workers "${COUNT_WORKERS}" \
    --shard-input-token-cap "${SHARD_INPUT_TOKEN_CAP}" \
    "${paid_args[@]}" \
    --input-mode prematerialized_f2 \
    --workers "${WORKERS}" \
    --max-prompt-tokens 12000 \
    --chat-overhead-reserve 256 \
    --dataset-label "common175_${pair_key}_${model_tag}_batch_k2_32k_max_authorized" \
    --expected-task-count 175 \
    --prompt-jsonl "${prompt_jsonl}" \
    --expected-prompt-jsonl-sha256 "${prompt_sha}" \
    --prompt-manifest "${prompt_manifest}" \
    --expected-prompt-manifest-sha256 "${prompt_manifest_sha}" \
    --eval-jsonl "${eval_jsonl}" \
    --expected-eval-jsonl-sha256 "${eval_sha}" \
    --eval-seal "${eval_seal}" \
    --expected-eval-seal-sha256 "${eval_seal_sha}" \
    --pair-manifest "${PAIR_MANIFEST}" \
    --expected-pair-manifest-sha256 "${PAIR_SHA}" \
    --pair-arm-key "${pair_key}" \
    --evaluator-module "${EVALUATOR}" \
    --expected-evaluator-sha256 "${EVALUATOR_SHA}" \
    --dart "${DART_BIN}" \
    --expected-dart-sha256 "${DART_SHA}" \
    --out "${RUN_ROOT}/${model_tag}/${arm_name}"
}

models=()
arms=()
[[ "${MODEL}" == "sol" || "${MODEL}" == "both" ]] && models+=("gpt-5.6-sol")
[[ "${MODEL}" == "terra" || "${MODEL}" == "both" ]] && models+=("gpt-5.6-terra")
[[ "${ARM}" == "opus" || "${ARM}" == "both" ]] && arms+=("opus")
[[ "${ARM}" == "codex" || "${ARM}" == "both" ]] && arms+=("codex")

# Paid submission is pair-gated: preflight both arms for each requested model
# before creating either arm's Batch.
if [[ "${ACTION}" == "submit" ]]; then
  saved_action="${ACTION}"
  ACTION=preflight
  for model_id in "${models[@]}"; do
    run_job "${model_id}" opus
    run_job "${model_id}" codex
  done
  ACTION="${saved_action}"
fi

for model_id in "${models[@]}"; do
  for arm_name in "${arms[@]}"; do
    run_job "${model_id}" "${arm_name}"
  done
done
