#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_FOLD_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_FOLD_PYTHON:-/venv/main/bin/python}"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
MODEL_REVISION=487d4acf21a4d70c70bf534265b5263c9424979e
TYPED_SFT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_contract_sft_2epoch_v1"
WARMSTART="${TYPED_SFT_DIR}/checkpoint-optstep-000348"
PASS1_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1"
PASS2_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1"
PASS2_LOCAL="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1"
PASS2_API="${WORKSPACE}/artifacts/t5gemma2_typed_dual_api_rescue_v1"
C001_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_continuation_c001_v1"
C002_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_c002_resume47_v2"
PREFIX_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_c002_prefix3_verification_v1"
OUTPUT_DIR="${T5GEMMA_TYPED_FOLD_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_fold_rs_sft_union_v1}"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

blocked() { echo "T5GEMMA_TYPED_FOLD_RS_SFT_UNION_V1_BLOCKED $*" >&2; exit 78; }
[[ -s "${SECRET_FILE}" && -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] || blocked "runtime/secret is absent"
for name in \
  T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256 T5GEMMA_TYPED_C002_RESUME_MANIFEST_SHA256 T5GEMMA_TYPED_C002_RESUME_TARGETS_SHA256 \
  T5GEMMA_TYPED_PREFIX3_REPORT_SHA256 T5GEMMA_TYPED_PREFIX3_MANIFEST_SHA256 T5GEMMA_TYPED_PREFIX3_TARGETS_SHA256; do
  [[ "${!name:-}" =~ ^[0-9a-f]{64}$ ]] || blocked "${name} is not late-bound"
done

PASS1_LOCAL=(
  "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab=${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json"
  "8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50=${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1/harvest_report.json"
  "883aeeab6d6a62d4bab41c62f77c8578866cf1a3b9355c4fd74e1de06c048cae=${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1/harvest_report.json"
  "91c00cd23d7cf5aa923604d416addfd926eb757d58ac856b858c0f5909e35fab=${WORKSPACE}/artifacts/t5gemma2_local_base_residual_unresolved4_v1/harvest_report.json"
)
PASS1_API=(
  "fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1/api_rescue_report.json"
  "99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1/api_rescue_report.json"
  "f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1/api_rescue_report.json"
  "fa0c70c73767a525f2ca710fd822cb2bdca60140f133696ad15b87e71d2751d1=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1/api_rescue_report.json"
  "336874a72569f6a82bbc844260b772e7f3dc631c399e23c979d54502713ea727=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1/api_rescue_report.json"
  "fe9bcd00c6774432b7911129246c8b2837523d85b1c94efb29c03f85ae860205=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_kimi_k3_mixed_paired50_v12/api_rescue_report.json"
  "fe2941885767f7c4abb3012d1a49c22a934a6b67d8f1f9626bf09e44a3d633d0=${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_kimi_k3_retry17_8k_v1/api_rescue_report.json"
)

LOCAL_REPORTS=(
  "${PASS1_LOCAL[@]}"
  "1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3=${PASS1_DIR}/dataset_manifest.json"
  "1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1=${PASS2_LOCAL}/harvest_report.json"
  "ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc=${PASS2_LOCAL}/harvest.journal.jsonl"
  "c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4=${PASS2_LOCAL}/direct_targets.jsonl"
  "dc5b98056f7ba4109c1930aa2165c3ba9050bc37d0ca7d0a5a33f3a5e182c0ee=${PASS2_DIR}/dataset_manifest.json"
)
API_REPORTS=(
  "${PASS1_API[@]}"
  "9221e7cc68babbee43c9b4ae2405e1633414cd0e684da942161ed100c848fac3=${PASS2_API}/orchestration_report.json"
  "cd142c89735ce69755b3b40a92a15bd49d92a117a81e496bd2d4432a25360a5c=${PASS2_API}/direct_manifest.json"
  "b64c8b57ed66ceb05a575cb51bcb33cd5d4c615442504c2382fdc998d493c586=${PASS2_API}/direct_targets.jsonl"
  "f137edb5a5484f0f4f8a59e54fb327cbfec754ccb2403844d59d2517d8e519d3=${C001_ROOT}/continuation_report.json"
  "a24d87391552aa233004ede2836c9e47b44cd08d6b30ec5b35b33ac1f684a370=${C001_ROOT}/direct_manifest.json"
  "71d93eb89f0915562226e715d189120384ad28831fd453ebf5b8bdb5fd624ab5=${C001_ROOT}/direct_targets.jsonl"
  "${T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256}=${C002_ROOT}/resume_report.json"
  "${T5GEMMA_TYPED_C002_RESUME_MANIFEST_SHA256}=${C002_ROOT}/direct_manifest.json"
  "${T5GEMMA_TYPED_C002_RESUME_TARGETS_SHA256}=${C002_ROOT}/direct_targets.jsonl"
  "${T5GEMMA_TYPED_PREFIX3_REPORT_SHA256}=${PREFIX_ROOT}/prefix_verification_report.json"
  "${T5GEMMA_TYPED_PREFIX3_MANIFEST_SHA256}=${PREFIX_ROOT}/direct_manifest.json"
  "${T5GEMMA_TYPED_PREFIX3_TARGETS_SHA256}=${PREFIX_ROOT}/direct_targets.jsonl"
)

printf '%s  %s\n' \
  2ae23d69f5dffe816d6b88d0356dc16d88bec16964a1d5dbe66db19c72afdd3c "${PROJECT}/scripts/training/t5gemma2_typed_fold_rs_sft_union_v1.py" \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  3f75898b4bed1b2b6058341d8ac0788b1feb8c66aa0f7aef5e374e6745b9f8ba "${WARMSTART}/run_contract.json" \
  71078435105dc29aff1aba5942abd5c272e78ef817896081f6e994938da9d77a "${WARMSTART}/adapter/adapter_model.safetensors" \
  f3701f13cb66b6b5952cd1dd2a71b17206e77c1c646ec806f6dd43d7e059a92d "${WARMSTART}/adapter/adapter_config.json" \
  | sha256sum -c - || blocked "sealed code/TRAIN/heldout/warmstart differs"
for spec in "${LOCAL_REPORTS[@]}" "${API_REPORTS[@]}"; do
  printf '%s  %s\n' "${spec%%=*}" "${spec#*=}" | sha256sum -c - || blocked "source artifact differs"
done

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
  /usr/bin/jq -e '.schema=="t5gemma2-typed-fold-rs-sft-union-checkpoint-v1" and (.path|type)=="string" and (.update|type)=="number"' "${OUTPUT_DIR}/latest_checkpoint.json" >/dev/null \
    || blocked "malformed fold checkpoint pointer"
  resume_checkpoint="$(/usr/bin/jq -r .path "${OUTPUT_DIR}/latest_checkpoint.json")"
  resolved_output="$(realpath -e "${OUTPUT_DIR}")"
  resolved_resume="$(realpath -e "${resume_checkpoint}" 2>/dev/null || true)"
  [[ -n "${resolved_resume}" && "$(dirname "${resolved_resume}")" == "${resolved_output}" && "$(basename "${resolved_resume}")" =~ ^checkpoint-optstep-[0-9]{6}$ ]] \
    || blocked "fold resume pointer escapes output root"
  resume_args=(--resume_checkpoint "${resolved_resume}")
elif find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  blocked "nonempty foreign fold output"
fi

report_args=()
for spec in "${LOCAL_REPORTS[@]}"; do report_args+=(--local_report "${spec}"); done
for spec in "${API_REPORTS[@]}"; do report_args+=(--api_report "${spec}"); done

export PYTHONPATH="${PROJECT}" HF_HOME="${WORKSPACE}/.hf_home" HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
exec "${PYTHON_BIN}" scripts/training/t5gemma2_typed_fold_rs_sft_union_v1.py \
  --gold_train_jsonl "${GOLD_TRAIN}" --gold_f2_jsonl "${GOLD_F2}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe --expected_gold_rows 2776 \
  --heldout_jsonl "${HELDOUT}" --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 --expected_heldout_rows 175 \
  "${report_args[@]}" \
  --gold_replay_ratio 0 --gold_replay_rows 0 --min_verified_direct_targets 447 --min_repair_conditioned_targets 0 \
  --warmstart_checkpoint "${WARMSTART}" --expected_warmstart_update 348 \
  --expected_warmstart_run_contract_sha256 3cb25d54f12743ed43572b219e119667f264abab94ec4cbfac72a94407fbdfc7 \
  --expected_warmstart_adapter_weights_sha256 71078435105dc29aff1aba5942abd5c272e78ef817896081f6e994938da9d77a \
  --expected_warmstart_adapter_config_sha256 f3701f13cb66b6b5952cd1dd2a71b17206e77c1c646ec806f6dd43d7e059a92d \
  --output_dir "${OUTPUT_DIR}" --model google/t5gemma-2-4b-4b --model_revision "${MODEL_REVISION}" \
  --max_source_tokens 32768 --max_target_tokens 32768 --epochs 1 --batch_size 1 --gradient_accumulation 8 \
  --learning_rate 5e-6 --warmup_ratio 0 --checkpoint_interval 5 --seed 42 --attn_implementation sdpa --bf16 --gradient_checkpointing \
  "${resume_args[@]}"
