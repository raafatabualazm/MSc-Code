#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
MODEL_REVISION=487d4acf21a4d70c70bf534265b5263c9424979e
SFT_DIR="${T5GEMMA_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1}"
WARMSTART="${T5GEMMA_MIXED_WARMSTART:-${SFT_DIR}/checkpoint-optstep-000348}"
WARMSTART_UPDATE="${T5GEMMA_MIXED_WARMSTART_UPDATE:-348}"
WARMSTART_RUN_CONTRACT_SHA256="${T5GEMMA_MIXED_WARMSTART_RUN_CONTRACT_SHA256:-21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3}"
WARMSTART_ADAPTER_WEIGHTS_SHA256="${T5GEMMA_MIXED_WARMSTART_ADAPTER_WEIGHTS_SHA256:-83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc}"
WARMSTART_ADAPTER_CONFIG_SHA256="${T5GEMMA_MIXED_WARMSTART_ADAPTER_CONFIG_SHA256:-c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3}"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
LOCAL_REPORT="${T5GEMMA_MIXED_LOCAL_REPORT:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1/harvest_report.json}"
LOCAL_REPORT_SHA256="${T5GEMMA_MIXED_LOCAL_REPORT_SHA256:-b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab}"
API_REPORT="${T5GEMMA_MIXED_API_REPORT:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_probe_prefix10_v1/api_rescue_report.json}"
API_REPORT_SHA256="${T5GEMMA_MIXED_API_REPORT_SHA256:-6368e08632119f18353d29fef6f6ed6d728bf09208cb2f42d90baab7775316e2}"
EXTRA_LOCAL_REPORT="${T5GEMMA_MIXED_EXTRA_LOCAL_REPORT:-}"
EXTRA_LOCAL_REPORT_SHA256="${T5GEMMA_MIXED_EXTRA_LOCAL_REPORT_SHA256:-}"
EXTRA_API_REPORT="${T5GEMMA_MIXED_EXTRA_API_REPORT:-}"
EXTRA_API_REPORT_SHA256="${T5GEMMA_MIXED_EXTRA_API_REPORT_SHA256:-}"
# Semicolon-separated SHA256=PATH bindings.  When either list is set, both
# lists must be set and they replace the legacy primary/extra slots.  This is
# the production path for more than two reports of either kind.
LOCAL_REPORT_SPECS="${T5GEMMA_MIXED_LOCAL_REPORT_SPECS:-}"
API_REPORT_SPECS="${T5GEMMA_MIXED_API_REPORT_SPECS:-}"
OUTPUT_DIR="${T5GEMMA_MIXED_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_mixed_rs_sft_exploratory_v1}"
ALLOW_EXPLORATORY="${T5GEMMA_MIXED_ALLOW_EXPLORATORY_INPUTS:-0}"
REQUIRE_LOCAL_FLOOR="${T5GEMMA_MIXED_REQUIRE_LOCAL_PRODUCTION_FLOOR:-0}"
GOLD_REPLAY_RATIO="${T5GEMMA_MIXED_GOLD_REPLAY_RATIO:-3}"
EPOCHS="${T5GEMMA_MIXED_EPOCHS:-3}"
LEARNING_RATE="${T5GEMMA_MIXED_LEARNING_RATE:-5e-5}"
MIN_DIRECT_TARGETS="${T5GEMMA_MIXED_MIN_DIRECT_TARGETS:-1}"
MIN_REPAIR_TARGETS="${T5GEMMA_MIXED_MIN_REPAIR_TARGETS:-1}"
RESUME_COMPAT="${T5GEMMA_MIXED_RESUME_COMPAT:-0}"
PREFLIGHT_ONLY="${T5GEMMA_MIXED_PREFLIGHT_ONLY:-0}"

if [[ "${ALLOW_EXPLORATORY}" != 0 && "${ALLOW_EXPLORATORY}" != 1 ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid exploratory flag" >&2
  exit 78
fi
if [[ "${REQUIRE_LOCAL_FLOOR}" != 0 && "${REQUIRE_LOCAL_FLOOR}" != 1 ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid local-floor flag" >&2
  exit 78
fi
if [[ "${RESUME_COMPAT}" != 0 && "${RESUME_COMPAT}" != 1 ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid resume-compat flag" >&2
  exit 78
fi
if [[ "${PREFLIGHT_ONLY}" != 0 && "${PREFLIGHT_ONLY}" != 1 ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid preflight-only flag" >&2
  exit 78
fi
if ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid epoch count" >&2
  exit 78
fi
if ! [[ "${WARMSTART_UPDATE}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${WARMSTART_RUN_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || ! [[ "${WARMSTART_ADAPTER_WEIGHTS_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || ! [[ "${WARMSTART_ADAPTER_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid warm-start identity" >&2
  exit 78
fi
if ! [[ "${MIN_DIRECT_TARGETS}" =~ ^[0-9]+$ ]] \
  || ! [[ "${MIN_REPAIR_TARGETS}" =~ ^[0-9]+$ ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid minimum target count" >&2
  exit 78
fi
if ! /venv/main/bin/python -c \
  'import math,sys; x=float(sys.argv[1]); raise SystemExit(not math.isfinite(x) or x <= 0)' \
  "${LEARNING_RATE}"; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid learning rate" >&2
  exit 78
fi
if ! /venv/main/bin/python -c \
  'import math,sys; x=float(sys.argv[1]); raise SystemExit(not math.isfinite(x) or x < 0)' \
  "${GOLD_REPLAY_RATIO}"; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid gold replay ratio" >&2
  exit 78
fi
if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi
if [[ ! -d "${WARMSTART}" ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED missing warm-start checkpoint" >&2
  exit 78
fi
report_args=()
append_report_specs() {
  local specs="$1"
  local kind="$2"
  local flag="$3"
  local entry digest path
  local -a entries=()
  IFS=';' read -r -a entries <<< "${specs}"
  if [[ ${#entries[@]} -eq 0 ]]; then
    echo "T5GEMMA_MIXED_RS_SFT_BLOCKED ${kind} report list is empty" >&2
    exit 78
  fi
  for entry in "${entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" != *=* ]]; then
      echo "T5GEMMA_MIXED_RS_SFT_BLOCKED malformed ${kind} report binding" >&2
      exit 78
    fi
    digest="${entry%%=*}"
    path="${entry#*=}"
    if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ || -z "${path}" || ! -s "${path}" ]]; then
      echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid ${kind} report binding" >&2
      exit 78
    fi
    printf '%s  %s\n' "${digest}" "${path}" | sha256sum -c -
    report_args+=("${flag}" "${entry}")
  done
}

# This identity-only read is the independent deny-list used by the trainer.
# Held-out content is never placed in a model input.
printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  "${HELDOUT}" \
  | sha256sum -c -
if [[ -n "${LOCAL_REPORT_SPECS}" || -n "${API_REPORT_SPECS}" ]]; then
  if [[ -n "${EXTRA_LOCAL_REPORT}${EXTRA_LOCAL_REPORT_SHA256}${EXTRA_API_REPORT}${EXTRA_API_REPORT_SHA256}" ]]; then
    echo "T5GEMMA_MIXED_RS_SFT_BLOCKED report-list mode forbids legacy extra slots" >&2
    exit 78
  fi
  if [[ -n "${LOCAL_REPORT_SPECS}" ]]; then
    append_report_specs "${LOCAL_REPORT_SPECS}" local --local_report
  fi
  if [[ -n "${API_REPORT_SPECS}" ]]; then
    append_report_specs "${API_REPORT_SPECS}" api --api_report
  fi
else
  if [[ ! -s "${LOCAL_REPORT}" || ! -s "${API_REPORT}" ]]; then
    echo "T5GEMMA_MIXED_RS_SFT_BLOCKED rescue report is absent" >&2
    exit 78
  fi
  printf '%s  %s\n' "${LOCAL_REPORT_SHA256}" "${LOCAL_REPORT}" | sha256sum -c -
  printf '%s  %s\n' "${API_REPORT_SHA256}" "${API_REPORT}" | sha256sum -c -
  report_args+=(--local_report "${LOCAL_REPORT_SHA256}=${LOCAL_REPORT}")
  report_args+=(--api_report "${API_REPORT_SHA256}=${API_REPORT}")
  if [[ -n "${EXTRA_LOCAL_REPORT}" || -n "${EXTRA_LOCAL_REPORT_SHA256}" ]]; then
    if [[ -z "${EXTRA_LOCAL_REPORT}" || -z "${EXTRA_LOCAL_REPORT_SHA256}" || ! -s "${EXTRA_LOCAL_REPORT}" ]]; then
      echo "T5GEMMA_MIXED_RS_SFT_BLOCKED incomplete extra local report binding" >&2
      exit 78
    fi
    printf '%s  %s\n' "${EXTRA_LOCAL_REPORT_SHA256}" "${EXTRA_LOCAL_REPORT}" | sha256sum -c -
    report_args+=(--local_report "${EXTRA_LOCAL_REPORT_SHA256}=${EXTRA_LOCAL_REPORT}")
  fi
  if [[ -n "${EXTRA_API_REPORT}" || -n "${EXTRA_API_REPORT_SHA256}" ]]; then
    if [[ -z "${EXTRA_API_REPORT}" || -z "${EXTRA_API_REPORT_SHA256}" || ! -s "${EXTRA_API_REPORT}" ]]; then
      echo "T5GEMMA_MIXED_RS_SFT_BLOCKED incomplete extra API report binding" >&2
      exit 78
    fi
    printf '%s  %s\n' "${EXTRA_API_REPORT_SHA256}" "${EXTRA_API_REPORT}" | sha256sum -c -
    report_args+=(--api_report "${EXTRA_API_REPORT_SHA256}=${EXTRA_API_REPORT}")
  fi
fi

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
  echo "T5GEMMA_MIXED_RS_SFT_BLOCKED HF_TOKEN is absent" >&2
  exit 78
fi

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
if [[ -f "${OUTPUT_DIR}/result.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/result.json")" == complete ]]; then
  echo "T5GEMMA_MIXED_RS_SFT_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

resume_args=()
if [[ -f "${OUTPUT_DIR}/latest_checkpoint.json" ]]; then
  resume_checkpoint=$(/usr/bin/jq -r '.path // empty' "${OUTPUT_DIR}/latest_checkpoint.json")
  if [[ -n "${resume_checkpoint}" && -d "${resume_checkpoint}" ]]; then
    resume_args=(--resume_checkpoint "${resume_checkpoint}")
    echo "T5GEMMA_MIXED_RS_SFT_RESUME checkpoint=${resume_checkpoint}"
  else
    echo "T5GEMMA_MIXED_RS_SFT_BLOCKED invalid checkpoint pointer" >&2
    exit 78
  fi
fi
trainer_entrypoint=scripts/training/t5gemma2_mixed_rs_sft.py
if [[ "${RESUME_COMPAT}" == 1 ]]; then
  if [[ ${#resume_args[@]} -eq 0 ]]; then
    echo "T5GEMMA_MIXED_RS_SFT_BLOCKED resume compatibility requires a checkpoint" >&2
    exit 78
  fi
  trainer_entrypoint=scripts/training/t5gemma2_mixed_rs_sft_resume_compat.py
fi
exploratory_args=()
if [[ "${ALLOW_EXPLORATORY}" == 1 ]]; then
  exploratory_args=(--allow_exploratory_inputs)
fi
local_floor_args=()
if [[ "${REQUIRE_LOCAL_FLOOR}" == 1 ]]; then
  local_floor_args=(--require_local_production_floor)
fi
preflight_args=()
if [[ "${PREFLIGHT_ONLY}" == 1 ]]; then
  preflight_args=(--preflight_only)
fi
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd "${PROJECT}"
exec /venv/main/bin/python "${trainer_entrypoint}" \
  --gold_train_jsonl "${GOLD_TRAIN}" \
  --gold_f2_jsonl "${GOLD_F2}" \
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --expected_gold_rows 2776 \
  --heldout_jsonl "${HELDOUT}" \
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --expected_heldout_rows 175 \
  "${report_args[@]}" \
  --gold_replay_ratio "${GOLD_REPLAY_RATIO}" \
  --gold_replay_rows -1 \
  --min_verified_direct_targets "${MIN_DIRECT_TARGETS}" \
  --min_repair_conditioned_targets "${MIN_REPAIR_TARGETS}" \
  --warmstart_checkpoint "${WARMSTART}" \
  --expected_warmstart_update "${WARMSTART_UPDATE}" \
  --expected_warmstart_run_contract_sha256 "${WARMSTART_RUN_CONTRACT_SHA256}" \
  --expected_warmstart_adapter_weights_sha256 "${WARMSTART_ADAPTER_WEIGHTS_SHA256}" \
  --expected_warmstart_adapter_config_sha256 "${WARMSTART_ADAPTER_CONFIG_SHA256}" \
  --output_dir "${OUTPUT_DIR}" \
  --model google/t5gemma-2-4b-4b \
  --model_revision "${MODEL_REVISION}" \
  --max_source_tokens 32768 \
  --max_target_tokens 32768 \
  --epochs "${EPOCHS}" \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio 0.05 \
  --checkpoint_interval 5 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16 \
  --gradient_checkpointing \
  "${exploratory_args[@]}" \
  "${local_floor_args[@]}" \
  "${preflight_args[@]}" \
  "${resume_args[@]}"
