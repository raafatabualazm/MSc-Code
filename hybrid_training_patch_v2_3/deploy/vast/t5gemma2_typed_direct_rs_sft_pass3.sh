#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_PASS3_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_PASS3_PYTHON:-/venv/main/bin/python}"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
MODEL_REVISION=487d4acf21a4d70c70bf534265b5263c9424979e
PASS1_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1"
WARMSTART="${PASS1_DIR}/checkpoint-optstep-000058"
C001_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c001_v1"
C002_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_c002_resume47_v2"
PREFIX_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_c002_prefix3_verification_v1"
OUTPUT_DIR="${T5GEMMA_TYPED_PASS3_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_pass3_c001_c002_v1}"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

blocked() { echo "T5GEMMA_TYPED_DIRECT_RS_SFT_PASS3_BLOCKED $*" >&2; exit 78; }
[[ -s "${SECRET_FILE}" && -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] || blocked "runtime/secret is absent"
for name in \
  T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256 T5GEMMA_TYPED_C002_RESUME_MANIFEST_SHA256 T5GEMMA_TYPED_C002_RESUME_TARGETS_SHA256 \
  T5GEMMA_TYPED_PREFIX3_REPORT_SHA256 T5GEMMA_TYPED_PREFIX3_MANIFEST_SHA256 T5GEMMA_TYPED_PREFIX3_TARGETS_SHA256; do
  [[ "${!name:-}" =~ ^[0-9a-f]{64}$ ]] || blocked "${name} is not late-bound"
done

printf '%s  %s\n' \
  2274a500e73b6e37f3fdc3144b6d70cb28aa5bb3ec463682a5a38df9ac7bd54f "${PROJECT}/scripts/training/t5gemma2_typed_direct_rs_sft_pass3.py" \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  5f04ad8f4019641bb55831217035de5e744050d908aaa11a4a12b1d52cf3be90 "${WARMSTART}/run_contract.json" \
  62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec "${WARMSTART}/adapter/adapter_model.safetensors" \
  b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b "${WARMSTART}/adapter/adapter_config.json" \
  1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3 "${PASS1_DIR}/dataset_manifest.json" \
  f137edb5a5484f0f4f8a59e54fb327cbfec754ccb2403844d59d2517d8e519d3 "${C001_ROOT}/continuation_report.json" \
  a24d87391552aa233004ede2836c9e47b44cd08d6b30ec5b35b33ac1f684a370 "${C001_ROOT}/direct_manifest.json" \
  71d93eb89f0915562226e715d189120384ad28831fd453ebf5b8bdb5fd624ab5 "${C001_ROOT}/direct_targets.jsonl" \
  "${T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256}" "${C002_ROOT}/resume_report.json" \
  "${T5GEMMA_TYPED_C002_RESUME_MANIFEST_SHA256}" "${C002_ROOT}/direct_manifest.json" \
  "${T5GEMMA_TYPED_C002_RESUME_TARGETS_SHA256}" "${C002_ROOT}/direct_targets.jsonl" \
  "${T5GEMMA_TYPED_PREFIX3_REPORT_SHA256}" "${PREFIX_ROOT}/prefix_verification_report.json" \
  "${T5GEMMA_TYPED_PREFIX3_MANIFEST_SHA256}" "${PREFIX_ROOT}/direct_manifest.json" \
  "${T5GEMMA_TYPED_PREFIX3_TARGETS_SHA256}" "${PREFIX_ROOT}/direct_targets.jsonl" \
  | sha256sum -c - || blocked "sealed pass-3 code/input evidence differs"

secret_line="$("${PYTHON_BIN}" - "${SECRET_FILE}" <<'PY'
import sys
from pathlib import Path
raw=Path(sys.argv[1]).read_bytes()
try: text=raw.decode('utf-8-sig')
except UnicodeDecodeError: text=raw.decode('utf-16')
lines=[x.strip() for x in text.splitlines() if x.strip() and not x.lstrip().startswith('#')]
if len(lines)!=1: raise SystemExit('secret must contain exactly one non-comment line')
x=lines[0]
if '=' in x and x.split('=',1)[0].replace('export ','').strip()=='HF_TOKEN': x=x.split('=',1)[1].strip()
if len(x)>=2 and x[0]==x[-1] and x[0] in "\"'": x=x[1:-1]
if not x or any(c.isspace() for c in x): raise SystemExit('HF token is malformed')
print(x,end='')
PY
)"
export HF_TOKEN="${secret_line}"
unset secret_line

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
resume_args=()
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  /usr/bin/jq -e '.schema=="t5gemma2-typed-direct-rs-sft-pass3-checkpoint-v1" and (.path|type)=="string" and (.update|type)=="number"' "${OUTPUT_DIR}/latest_checkpoint.json" >/dev/null \
    || blocked "malformed pass-3 checkpoint pointer"
  resume_checkpoint="$(/usr/bin/jq -r .path "${OUTPUT_DIR}/latest_checkpoint.json")"
  resolved_output="$(realpath -e "${OUTPUT_DIR}")"
  resolved_resume="$(realpath -e "${resume_checkpoint}" 2>/dev/null || true)"
  [[ -n "${resolved_resume}" && "$(dirname "${resolved_resume}")" == "${resolved_output}" && "$(basename "${resolved_resume}")" =~ ^checkpoint-optstep-[0-9]{6}$ ]] \
    || blocked "pass-3 resume pointer escapes its output root"
  resume_args=(--resume_checkpoint "${resolved_resume}")
elif find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  blocked "nonempty foreign pass-3 output"
fi

export PYTHONPATH="${PROJECT}" HF_HOME="${WORKSPACE}/.hf_home" HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
"${PYTHON_BIN}" scripts/training/t5gemma2_typed_direct_rs_sft_pass3.py \
  --gold_train_jsonl "${GOLD_TRAIN}" --gold_f2_jsonl "${GOLD_F2}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe --expected_gold_rows 2776 \
  --heldout_jsonl "${HELDOUT}" --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 --expected_heldout_rows 175 \
  --local_report "1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3=${PASS1_DIR}/dataset_manifest.json" \
  --api_report "f137edb5a5484f0f4f8a59e54fb327cbfec754ccb2403844d59d2517d8e519d3=${C001_ROOT}/continuation_report.json" \
  --api_report "a24d87391552aa233004ede2836c9e47b44cd08d6b30ec5b35b33ac1f684a370=${C001_ROOT}/direct_manifest.json" \
  --api_report "71d93eb89f0915562226e715d189120384ad28831fd453ebf5b8bdb5fd624ab5=${C001_ROOT}/direct_targets.jsonl" \
  --api_report "${T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256}=${C002_ROOT}/resume_report.json" \
  --api_report "${T5GEMMA_TYPED_C002_RESUME_MANIFEST_SHA256}=${C002_ROOT}/direct_manifest.json" \
  --api_report "${T5GEMMA_TYPED_C002_RESUME_TARGETS_SHA256}=${C002_ROOT}/direct_targets.jsonl" \
  --api_report "${T5GEMMA_TYPED_PREFIX3_REPORT_SHA256}=${PREFIX_ROOT}/prefix_verification_report.json" \
  --api_report "${T5GEMMA_TYPED_PREFIX3_MANIFEST_SHA256}=${PREFIX_ROOT}/direct_manifest.json" \
  --api_report "${T5GEMMA_TYPED_PREFIX3_TARGETS_SHA256}=${PREFIX_ROOT}/direct_targets.jsonl" \
  --gold_replay_ratio 0 --gold_replay_rows 0 --min_verified_direct_targets 13 --min_repair_conditioned_targets 0 \
  --warmstart_checkpoint "${WARMSTART}" --expected_warmstart_update 58 \
  --expected_warmstart_run_contract_sha256 0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f \
  --expected_warmstart_adapter_weights_sha256 62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec \
  --expected_warmstart_adapter_config_sha256 b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b \
  --output_dir "${OUTPUT_DIR}" --model google/t5gemma-2-4b-4b --model_revision "${MODEL_REVISION}" \
  --max_source_tokens 32768 --max_target_tokens 32768 --epochs 2 --batch_size 1 --gradient_accumulation 8 \
  --learning_rate 2e-5 --warmup_ratio 0 --checkpoint_interval 5 --seed 42 --attn_implementation sdpa --bf16 --gradient_checkpointing \
  "${resume_args[@]}"
