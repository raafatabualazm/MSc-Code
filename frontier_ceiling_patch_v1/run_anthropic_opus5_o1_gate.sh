#!/usr/bin/env bash
# Gated Opus 5 O1 follow-up over both complete sealed 175-task F2 arms.
# This launcher never accepts an arm subset.  For ACTION=submit, it preflights
# both arms before allowing either paid Batch creation.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/workspace/.venv/bin/python}"
PATCH_DIR="/workspace/frontier_ceiling_patch_v1"
OPUS_RUNNER="${OPUS_RUNNER:-${PATCH_DIR}/anthropic_opus5_o1_batch.py}"
PAIR_ROOT="/workspace/artifacts/frontier_ceiling_two_enrichments"
RUN_ROOT="${RUN_ROOT:-${PAIR_ROOT}/runs/anthropic_opus5_o1_k1_warm_v1}"
PAIR_MANIFEST="${PAIR_ROOT}/pair_manifest.json"
PAIR_SHA="35f4cfcaf0732928312bed3f2f27c3f3e347525c0076921caeab7ee6539c132e"
EVALUATOR="/workspace/hybrid_training_patch_v2_3/scripts/evaluation/graph_compile_at_k_antigravity.py"
EVALUATOR_SHA="249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
DART_BIN="/usr/lib/dart/bin/dart"
DART_SHA="c03ad868b5c53e31461b0fef22dc6eb6aeb56b7567efff6ca488ce9c4a6f8a6a"

ACTION="${ACTION:-preflight}"
WORKERS="${WORKERS:-1}"
OPUS_ARM_COST_CAP_USD="25.088"

case "${ACTION}" in
  preflight|submit|status|harvest) ;;
  *)
    echo "ACTION must be preflight, submit, status, or harvest" >&2
    exit 2
    ;;
esac
if ! "${PYTHON_BIN}" -c 'import anthropic' >/dev/null 2>&1; then
  echo "The anthropic Python package is required in ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ "${ACTION}" != "preflight" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is required for ${ACTION}" >&2
  exit 2
fi

run_arm() {
  local arm_name="$1"
  local action="$2"
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

  "${PYTHON_BIN}" "${OPUS_RUNNER}" \
    --action "${action}" \
    --model claude-opus-5 \
    --input-mode prematerialized_f2 \
    --arm compact \
    --k 1 \
    --workers "${WORKERS}" \
    --max-output-tokens 8192 \
    --max-prompt-tokens 12000 \
    --chat-overhead-reserve 256 \
    --screen-cost-cap-usd "${OPUS_ARM_COST_CAP_USD}" \
    --dataset-label "common175_${pair_key}_claude_opus5_o1" \
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
    --out "${RUN_ROOT}/${arm_name}"
}

# A paid submit is allowed only after both complete arms pass the same offline
# gate and artifact preflight.  This prevents a one-arm partial protocol caused
# by a local seal/configuration error.
if [[ "${ACTION}" == "submit" ]]; then
  run_arm opus preflight
  run_arm codex preflight
fi

run_arm opus "${ACTION}"
run_arm codex "${ACTION}"
