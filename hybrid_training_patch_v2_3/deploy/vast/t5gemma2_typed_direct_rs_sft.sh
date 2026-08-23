#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
MODEL_REVISION=487d4acf21a4d70c70bf534265b5263c9424979e
TYPED_SFT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_contract_sft_2epoch_v1"
WARMSTART="${TYPED_SFT_DIR}/checkpoint-optstep-000348"
OUTPUT_DIR="${T5GEMMA_TYPED_DIRECT_RS_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1}"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

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
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_BLOCKED Dart 3.12.2 is absent" >&2
  exit 78
fi
for required in \
  "${WARMSTART}/run_contract.json" \
  "${WARMSTART}/training_state.pt" \
  "${WARMSTART}/adapter/adapter_model.safetensors" \
  "${WARMSTART}/adapter/adapter_config.json" \
  "${WARMSTART}/tokenizer/tokenizer.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "T5GEMMA_TYPED_DIRECT_RS_SFT_BLOCKED missing ${required}" >&2
    exit 78
  fi
done

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  3f75898b4bed1b2b6058341d8ac0788b1feb8c66aa0f7aef5e374e6745b9f8ba "${WARMSTART}/run_contract.json" \
  71078435105dc29aff1aba5942abd5c272e78ef817896081f6e994938da9d77a "${WARMSTART}/adapter/adapter_model.safetensors" \
  f3701f13cb66b6b5952cd1dd2a71b17206e77c1c646ec806f6dd43d7e059a92d "${WARMSTART}/adapter/adapter_config.json" \
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
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_BLOCKED HF_TOKEN is absent" >&2
  exit 78
fi

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
if [[ -f "${OUTPUT_DIR}/result.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/result.json")" == complete ]]; then
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi
resume_args=()
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  resume_checkpoint=$(/usr/bin/jq -r '.path // empty' "${OUTPUT_DIR}/latest_checkpoint.json")
  if [[ -n "${resume_checkpoint}" && -d "${resume_checkpoint}" ]]; then
    resume_args=(--resume_checkpoint "${resume_checkpoint}")
    echo "T5GEMMA_TYPED_DIRECT_RS_SFT_RESUME checkpoint=${resume_checkpoint}"
  else
    echo "T5GEMMA_TYPED_DIRECT_RS_SFT_BLOCKED invalid checkpoint pointer" >&2
    exit 78
  fi
elif find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_BLOCKED nonempty foreign/incomplete output ${OUTPUT_DIR}" >&2
  exit 78
else
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_FRESH output=${OUTPUT_DIR}"
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
exec /venv/main/bin/python scripts/training/t5gemma2_typed_direct_rs_sft.py \
  --gold_train_jsonl "${GOLD_TRAIN}" \
  --gold_f2_jsonl "${GOLD_F2}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_gold_rows 2776 \
  --heldout_jsonl "${HELDOUT}" \
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --expected_heldout_rows 175 \
  "${report_args[@]}" \
  --gold_replay_ratio 0 \
  --gold_replay_rows 0 \
  --min_verified_direct_targets 225 \
  --min_repair_conditioned_targets 0 \
  --warmstart_checkpoint "${WARMSTART}" \
  --expected_warmstart_update 348 \
  --expected_warmstart_run_contract_sha256 3cb25d54f12743ed43572b219e119667f264abab94ec4cbfac72a94407fbdfc7 \
  --expected_warmstart_adapter_weights_sha256 71078435105dc29aff1aba5942abd5c272e78ef817896081f6e994938da9d77a \
  --expected_warmstart_adapter_config_sha256 f3701f13cb66b6b5952cd1dd2a71b17206e77c1c646ec806f6dd43d7e059a92d \
  --output_dir "${OUTPUT_DIR}" \
  --model google/t5gemma-2-4b-4b \
  --model_revision "${MODEL_REVISION}" \
  --max_source_tokens 32768 \
  --max_target_tokens 32768 \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0 \
  --checkpoint_interval 5 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16 \
  --gradient_checkpointing \
  "${resume_args[@]}"
