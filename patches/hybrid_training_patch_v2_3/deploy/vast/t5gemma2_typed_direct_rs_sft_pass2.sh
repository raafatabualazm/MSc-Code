#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
MODEL_REVISION=487d4acf21a4d70c70bf534265b5263c9424979e
PASS1_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1"
WARMSTART="${PASS1_DIR}/checkpoint-optstep-000058"
OUTPUT_DIR="${T5GEMMA_TYPED_DIRECT_RS_SFT_PASS2_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1}"
LOCAL_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1"
DUAL_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_dual_api_rescue_v1"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

blocked() {
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_PASS2_BLOCKED $*" >&2
  exit 78
}

[[ -s "${SECRET_FILE}" ]] || blocked "missing ${SECRET_FILE}"
[[ -x "${DART_BIN}" ]] || blocked "Dart 3.12.2 is absent"
for required in \
  "${WARMSTART}/run_contract.json" \
  "${WARMSTART}/training_state.pt" \
  "${WARMSTART}/adapter/adapter_model.safetensors" \
  "${WARMSTART}/adapter/adapter_config.json" \
  "${WARMSTART}/tokenizer/tokenizer.json" \
  "${LOCAL_DIR}/harvest_report.json" \
  "${LOCAL_DIR}/harvest.journal.jsonl" \
  "${LOCAL_DIR}/direct_targets.jsonl" \
  "${PASS1_DIR}/dataset_manifest.json" \
  "${DUAL_DIR}/orchestration_report.json" \
  "${DUAL_DIR}/direct_manifest.json"; do
  [[ -s "${required}" ]] || blocked "missing ${required}"
done
[[ -f "${DUAL_DIR}/direct_targets.jsonl" ]] \
  || blocked "missing ${DUAL_DIR}/direct_targets.jsonl"

/usr/bin/jq -e '
  .schema == "t5gemma2-typed-dual-api-orchestration-report-v1"
  and .status == "complete"
  and .heldout_175_model_visible == false
  and .heldout_175_used_for_generation_or_selection == false
  and (.direct_manifest.rows | type) == "number"
  and .direct_manifest.rows >= 0
  and .direct_manifest.rows <= 88
' "${DUAL_DIR}/orchestration_report.json" >/dev/null \
  || blocked "dual-provider orchestration is not a completed usable result"

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  5f04ad8f4019641bb55831217035de5e744050d908aaa11a4a12b1d52cf3be90 "${WARMSTART}/run_contract.json" \
  62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec "${WARMSTART}/adapter/adapter_model.safetensors" \
  b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b "${WARMSTART}/adapter/adapter_config.json" \
  1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1 "${LOCAL_DIR}/harvest_report.json" \
  ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc "${LOCAL_DIR}/harvest.journal.jsonl" \
  c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4 "${LOCAL_DIR}/direct_targets.jsonl" \
  1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3 "${PASS1_DIR}/dataset_manifest.json" \
  | sha256sum -c - || blocked "sealed local, parent, TRAIN, or heldout input differs"

# These artifacts cannot be pinned in source before the paid harvest exists.
# Bind their exact completed bytes now; the Python profile then independently
# audits the orchestration journal and every phase before it observes row count.
API_REPORT_SHA256="$(sha256sum "${DUAL_DIR}/orchestration_report.json" | cut -d' ' -f1)"
API_MANIFEST_SHA256="$(sha256sum "${DUAL_DIR}/direct_manifest.json" | cut -d' ' -f1)"
API_TARGETS_SHA256="$(sha256sum "${DUAL_DIR}/direct_targets.jsonl" | cut -d' ' -f1)"

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
[[ -n "${HF_TOKEN:-}" ]] || blocked "HF_TOKEN is absent"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
resume_args=()
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  /usr/bin/jq -e '
    .schema == "t5gemma2-typed-direct-rs-sft-pass2-checkpoint-v1"
    and (.update | type) == "number"
    and .update >= 0
    and (.run_contract_sha256 | type) == "string"
    and (.run_contract_sha256 | test("^[0-9a-f]{64}$"))
    and (.path | type) == "string"
    and (.path | length) > 0
  ' "${OUTPUT_DIR}/latest_checkpoint.json" >/dev/null \
    || blocked "malformed checkpoint pointer"
  resume_checkpoint="$(/usr/bin/jq -r '.path // empty' "${OUTPUT_DIR}/latest_checkpoint.json")"
  resolved_output="$(realpath -e "${OUTPUT_DIR}")"
  resolved_resume="$(realpath -e "${resume_checkpoint}" 2>/dev/null || true)"
  if [[ -n "${resolved_resume}" \
    && "$(dirname "${resolved_resume}")" == "${resolved_output}" \
    && "$(basename "${resolved_resume}")" =~ ^checkpoint-optstep-[0-9]{6}$ \
    && -s "${resolved_resume}/run_contract.json" \
    && -s "${resolved_resume}/training_state.pt" \
    && -s "${resolved_resume}/adapter/adapter_model.safetensors" \
    && -s "${resolved_resume}/adapter/adapter_config.json" \
    && -s "${resolved_resume}/tokenizer/tokenizer_config.json" ]]; then
    resume_checkpoint="${resolved_resume}"
    resume_args=(--resume_checkpoint "${resume_checkpoint}")
    echo "T5GEMMA_TYPED_DIRECT_RS_SFT_PASS2_RESUME checkpoint=${resume_checkpoint}"
  else
    blocked "invalid checkpoint pointer"
  fi
elif find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  blocked "nonempty foreign/incomplete output ${OUTPUT_DIR}"
else
  echo "T5GEMMA_TYPED_DIRECT_RS_SFT_PASS2_FRESH output=${OUTPUT_DIR}"
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
exec /venv/main/bin/python scripts/training/t5gemma2_typed_direct_rs_sft_pass2.py \
  --gold_train_jsonl "${GOLD_TRAIN}" \
  --gold_f2_jsonl "${GOLD_F2}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_gold_rows 2776 \
  --heldout_jsonl "${HELDOUT}" \
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --expected_heldout_rows 175 \
  --local_report "1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1=${LOCAL_DIR}/harvest_report.json" \
  --local_report "ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc=${LOCAL_DIR}/harvest.journal.jsonl" \
  --local_report "c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4=${LOCAL_DIR}/direct_targets.jsonl" \
  --local_report "1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3=${PASS1_DIR}/dataset_manifest.json" \
  --api_report "${API_REPORT_SHA256}=${DUAL_DIR}/orchestration_report.json" \
  --api_report "${API_MANIFEST_SHA256}=${DUAL_DIR}/direct_manifest.json" \
  --api_report "${API_TARGETS_SHA256}=${DUAL_DIR}/direct_targets.jsonl" \
  --gold_replay_ratio 0 \
  --gold_replay_rows 0 \
  --min_verified_direct_targets 190 \
  --min_repair_conditioned_targets 0 \
  --warmstart_checkpoint "${WARMSTART}" \
  --expected_warmstart_update 58 \
  --expected_warmstart_run_contract_sha256 0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f \
  --expected_warmstart_adapter_weights_sha256 62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec \
  --expected_warmstart_adapter_config_sha256 b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b \
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
