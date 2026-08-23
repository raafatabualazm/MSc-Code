#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
PARENT_STAGE="${T5GEMMA_TYPED_LOCAL_PARENT_STAGE:-typed_direct}"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

case "${PARENT_STAGE}" in
  typed_direct)
    CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1/checkpoint-optstep-000058"
    CHECKPOINT_UPDATE=58
    CHECKPOINT_CONTRACT_SHA=0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f
    CHECKPOINT_CONTRACT_FILE_SHA=5f04ad8f4019641bb55831217035de5e744050d908aaa11a4a12b1d52cf3be90
    CHECKPOINT_STATE_SHA=6960bc8bdd4b8bafc8e732fc36ac011ccdf8a8f6246a0d3f29f5996235717e89
    CHECKPOINT_WEIGHTS_SHA=62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec
    CHECKPOINT_CONFIG_SHA=b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b
    DEFAULT_OUTPUT="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1"
    ;;
  typed_sft)
    CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_contract_sft_2epoch_v1/checkpoint-optstep-000348"
    CHECKPOINT_UPDATE=348
    CHECKPOINT_CONTRACT_SHA=3cb25d54f12743ed43572b219e119667f264abab94ec4cbfac72a94407fbdfc7
    CHECKPOINT_CONTRACT_FILE_SHA=3f75898b4bed1b2b6058341d8ac0788b1feb8c66aa0f7aef5e374e6745b9f8ba
    CHECKPOINT_STATE_SHA=97220684aa00213a9fc8a20bc088cd1fa2e017f8bc09f5194e8146065f41b5f9
    CHECKPOINT_WEIGHTS_SHA=71078435105dc29aff1aba5942abd5c272e78ef817896081f6e994938da9d77a
    CHECKPOINT_CONFIG_SHA=f3701f13cb66b6b5952cd1dd2a71b17206e77c1c646ec806f6dd43d7e059a92d
    DEFAULT_OUTPUT="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_sft348_k4_v1"
    ;;
  *)
    echo "T5GEMMA_TYPED_LOCAL_HARVEST_BLOCKED invalid parent stage ${PARENT_STAGE}" >&2
    exit 78
    ;;
esac
OUTPUT_DIR="${T5GEMMA_TYPED_LOCAL_HARVEST_OUTPUT_DIR:-${DEFAULT_OUTPUT}}"

LOCAL_REPORTS=(
  "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab=${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json"
  "8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50=${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest_report.json"
  "883aeeab6d6a62d4bab41c62f77c8578866cf1a3b9355c4fd74e1de06c048cae=${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest_report.json"
  "91c00cd23d7cf5aa923604d416addfd926eb757d58ac856b858c0f5909e35fab=${WORKSPACE}/artifacts/t5gemma2_local_base_residual_unresolved4_v1/harvest_report.json"
)
API_REPORTS=(
  "fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1/api_rescue_report.json"
  "99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1/api_rescue_report.json"
  "f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue_report.json"
  "fa0c70c73767a525f2ca710fd822cb2bdca60140f133696ad15b87e71d2751d1=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1/api_rescue_report.json"
  "336874a72569f6a82bbc844260b772e7f3dc631c399e23c979d54502713ea727=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue_report.json"
  "fe9bcd00c6774432b7911129246c8b2837523d85b1c94efb29c03f85ae860205=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_kimi_k3_mixed_paired50_v12/api_rescue_report.json"
  "fe2941885767f7c4abb3012d1a49c22a934a6b67d8f1f9626bf09e44a3d633d0=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_kimi_k3_retry17_8k_v1/api_rescue_report.json"
)

if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_TYPED_LOCAL_HARVEST_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_TYPED_LOCAL_HARVEST_BLOCKED Dart 3.12.2 is absent" >&2
  exit 78
fi
for required in \
  "${CHECKPOINT}/run_contract.json" \
  "${CHECKPOINT}/training_state.pt" \
  "${CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${CHECKPOINT}/adapter/adapter_config.json" \
  "${CHECKPOINT}/tokenizer/tokenizer.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "T5GEMMA_TYPED_LOCAL_HARVEST_BLOCKED missing ${required}" >&2
    exit 78
  fi
done

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  "${CHECKPOINT_CONTRACT_FILE_SHA}" "${CHECKPOINT}/run_contract.json" \
  "${CHECKPOINT_STATE_SHA}" "${CHECKPOINT}/training_state.pt" \
  "${CHECKPOINT_WEIGHTS_SHA}" "${CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${CHECKPOINT_CONFIG_SHA}" "${CHECKPOINT}/adapter/adapter_config.json" \
  | sha256sum -c -

report_args=()
for spec in "${LOCAL_REPORTS[@]}"; do
  digest="${spec%%=*}"
  path="${spec#*=}"
  printf '%s  %s\n' "${digest}" "${path}" | sha256sum -c -
  report_args+=(--local_report "${spec}")
done
for spec in "${API_REPORTS[@]}"; do
  digest="${spec%%=*}"
  path="${spec#*=}"
  printf '%s  %s\n' "${digest}" "${path}" | sha256sum -c -
  report_args+=(--api_report "${spec}")
done

secret_line="$(
  /venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_bytes()
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("utf-16")
lines = [
    line.strip()
    for line in text.splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit("secret must contain exactly one non-comment token line")
print(lines[0], end="")
PY
)"
if [[ "${secret_line}" =~ ^[[:space:]]*(export[[:space:]]+)?HF_TOKEN[[:space:]]*=(.*)$ ]]; then
  HF_TOKEN="${BASH_REMATCH[2]}"
else
  HF_TOKEN="${secret_line}"
fi
HF_TOKEN="${HF_TOKEN#"${HF_TOKEN%%[![:space:]]*}"}"
HF_TOKEN="${HF_TOKEN%"${HF_TOKEN##*[![:space:]]}"}"
if [[ "${HF_TOKEN}" == \"*\" && "${HF_TOKEN}" == *\" ]]; then
  HF_TOKEN="${HF_TOKEN:1:-1}"
elif [[ "${HF_TOKEN}" == \'*\' && "${HF_TOKEN}" == *\' ]]; then
  HF_TOKEN="${HF_TOKEN:1:-1}"
fi
export HF_TOKEN
unset secret_line
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "T5GEMMA_TYPED_LOCAL_HARVEST_BLOCKED HF_TOKEN is absent" >&2
  exit 78
fi

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
if [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 \
  ! -name 'harvest.journal.jsonl' \
  ! -name 'harvest.journal.jsonl.chain-head.json' \
  ! -name 'direct_targets.jsonl' \
  ! -name 'direct_f2.jsonl' \
  ! -name 'schedule_manifest.jsonl' \
  ! -name 'harvest_report.json' -print -quit)" ]]; then
  echo "T5GEMMA_TYPED_LOCAL_HARVEST_BLOCKED foreign output in ${OUTPUT_DIR}" >&2
  exit 78
fi
if [[ ! -f "${OUTPUT_DIR}/harvest.journal.jsonl" ]] \
  && find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "T5GEMMA_TYPED_LOCAL_HARVEST_BLOCKED outputs exist without journal" >&2
  exit 78
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_typed_local_direct_harvest.py \
  --gold_train_jsonl "${GOLD_TRAIN}" \
  --gold_f2_jsonl "${GOLD_F2}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_gold_rows 2776 \
  --heldout_jsonl "${HELDOUT}" \
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --expected_heldout_rows 175 \
  "${report_args[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_stage "${PARENT_STAGE}" \
  --expected_checkpoint_update "${CHECKPOINT_UPDATE}" \
  --expected_checkpoint_run_contract_sha256 "${CHECKPOINT_CONTRACT_SHA}" \
  --expected_checkpoint_run_contract_file_sha256 "${CHECKPOINT_CONTRACT_FILE_SHA}" \
  --expected_checkpoint_training_state_sha256 "${CHECKPOINT_STATE_SHA}" \
  --expected_checkpoint_adapter_weights_sha256 "${CHECKPOINT_WEIGHTS_SHA}" \
  --expected_checkpoint_adapter_config_sha256 "${CHECKPOINT_CONFIG_SHA}" \
  --output_dir "${OUTPUT_DIR}" \
  --samples_per_task 4 \
  --max_source_tokens 32768 \
  --max_new_tokens 4096 \
  --generation_batch_size 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --evaluation_workers 8 \
  --timeout 30 \
  --stability_runs 2 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16
