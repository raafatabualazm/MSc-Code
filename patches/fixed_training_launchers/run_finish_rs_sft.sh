#!/usr/bin/env bash
# Predeclared matched RS-SFT/control fitting on the sealed executable view. No heldout file
# is opened and no heldout metric can select, stop, or block either arm.
set -Eeuo pipefail

PATCH_ROOT="${PATCH_ROOT:-/workspace/hybrid_training_patch_v2_3}"
EXEC_ROOT="${MULTIFUNCTION_EXECUTABLE_ROOT:-/workspace/multifunction_v1/expanded2776/executable_target24k}"
MULTIFUNCTION_BUILD="${MULTIFUNCTION_BUILD:-/workspace/multifunction_v1/expanded2776}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-/workspace/artifacts/direct_compact_qwen38_union2776/direct_compact_qwen_cot_sft}"
QWEN_BUILD_MANIFEST="${QWEN_BUILD_MANIFEST:-$(dirname "${WARMSTART_CHECKPOINT}")/qwen_mc_sequence_train.build.json}"
CHATGPT_ROOT="${CHATGPT_ROOT:-/workspace/artifacts/chatgpt_rs_qwen38_union2776_target24k_gpt56}"
CHATGPT_REPAIRS="${CHATGPT_REPAIRS:-${CHATGPT_ROOT}/verified_repairs.jsonl}"
CHATGPT_REPORT="${CHATGPT_REPORT:-${CHATGPT_ROOT}/report.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/artifacts/direct_compact_rs_sft_union2776_target24k}"
CHAIN_CONTRACT="${CHAIN_CONTRACT:-/workspace/artifacts/post_qwen_union2776_target24k_chain.json}"
PYTHON="${PYTHON:-/venv/main/bin/python}"

BASE_TRAIN="${BASE_TRAIN:-${EXEC_ROOT}/train_multifunction_binary_executable.jsonl}"
BASE_TRAIN_SEAL="${BASE_TRAIN_SEAL:-${EXEC_ROOT}/train_multifunction_binary_executable.seal.json}"
EXECUTABLE_VIEW_REPORT="${EXECUTABLE_VIEW_REPORT:-${EXEC_ROOT}/executable_view.build.json}"
CONTRACT="${CONTRACT:-${EXEC_ROOT}/compact_contract.json}"
CODEBOOK="${CODEBOOK:-${MULTIFUNCTION_BUILD}/multifunction_inline_cfg_v2_codebook.json}"
CODEC="${CODEC:-/workspace/scripts/data/build_multifunction_compact_v2.py}"
EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256="${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256:-}"
EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256:-f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f}"
EXPECTED_PARENT_FIT_ROWS="${EXPECTED_PARENT_FIT_ROWS:-2776}"

ROWS_PER_ARM="${ROWS_PER_ARM:-}"
MIN_UNIQUE_REPAIRS="${MIN_UNIQUE_REPAIRS:-400}"
RECERTIFY_TIMEOUT="${RECERTIFY_TIMEOUT:-30}"
RECERTIFY_STABILITY_RUNS="${RECERTIFY_STABILITY_RUNS:-2}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
RS_LORA_DROPOUT="${RS_LORA_DROPOUT:-0.05}"

if (( $# != 0 )); then
  printf 'This sealed RS-SFT launcher accepts no positional overrides\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256 is still an unsealed placeholder\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_CONTRACT_SHA256 is unsealed\n' >&2
  exit 2
fi

for required in \
  "${PATCH_ROOT}/scripts/training/build_direct_compact_rs_sft.py" \
  "${PATCH_ROOT}/scripts/training/direct_compact_qwen_decompiler.py" \
  "${PATCH_ROOT}/scripts/training/seal_post_qwen_chain.py" \
  "${BASE_TRAIN}" "${BASE_TRAIN_SEAL}" "${EXECUTABLE_VIEW_REPORT}" \
  "${CONTRACT}" "${CODEBOOK}" "${CODEC}" "${TOKENIZER_JSON}" \
  "${QWEN_BUILD_MANIFEST}" "${CHATGPT_REPAIRS}" "${CHATGPT_REPORT}" \
  "${CHAIN_CONTRACT}" \
  "${WARMSTART_CHECKPOINT}/decoder_adapter/adapter_config.json" \
  "${WARMSTART_CHECKPOINT}/source_embedding_overlay.pt" \
  "${WARMSTART_CHECKPOINT}/compact_contract.json" \
  "${WARMSTART_CHECKPOINT}/run_provenance.json"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Required multi-function RS-SFT input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
DECLARED_ROWS_PER_ARM="$("${PYTHON}" - "${CHAIN_CONTRACT}" \
  "${EXPECTED_PARENT_FIT_ROWS}" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
payload = value.get("payload") or {}
rs = payload.get("rs_sft") or {}
view = payload.get("executable_train") or {}
rows = int(rs.get("rows_per_arm", -1))
executable_rows = int(view.get("rows", -1))
if (
    int(view.get("parent_rows", -1)) != int(sys.argv[2])
    or int(view.get("heldout_rows", -1)) != 175
    or rows != 2 * executable_rows
):
    raise SystemExit("chain has incoherent sealed fit/RS row accounting")
print(rows)
PY
)"
if [[ -n "${ROWS_PER_ARM}" && "${ROWS_PER_ARM}" != "${DECLARED_ROWS_PER_ARM}" ]]; then
  printf 'ROWS_PER_ARM differs from the sealed chain: %s != %s\n' \
    "${ROWS_PER_ARM}" "${DECLARED_ROWS_PER_ARM}" >&2
  exit 2
fi
ROWS_PER_ARM="${DECLARED_ROWS_PER_ARM}"
if [[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" \
   != "${EXPECTED_CONTRACT_SHA256}" ]]; then
  printf 'Target24k compact contract hash mismatch: %s\n' "${CONTRACT}" >&2
  exit 2
fi
mkdir -p /workspace/locks
exec 9>/workspace/locks/direct_compact_rs_sft.lock
if ! flock -n 9; then
  printf 'Another direct-compact RS-SFT run holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[direct_compact_rs_sft] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

export PYTHONPATH="${PATCH_ROOT}:/workspace"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" -c \
  'import sys; from scripts.training.collect_chatgpt_compact_rs import validate_qwen_student_checkpoint; validate_qwen_student_checkpoint(sys.argv[1], qwen_build_manifest=sys.argv[2]); print("QWEN_RS_WARMSTART_VERIFIED", flush=True)' \
  "${WARMSTART_CHECKPOINT}" "${QWEN_BUILD_MANIFEST}"

DATA_DIR="${OUTPUT_ROOT}/00_matched_data"
CONTROL_DIR="${OUTPUT_ROOT}/02_gold_control"
RS_DIR="${OUTPUT_ROOT}/03_rs_sft"

# Verify that this invocation is exactly the predeclared train-only chain.
"${PYTHON}" - "${CHAIN_CONTRACT}" "${WARMSTART_CHECKPOINT}" \
  "${OUTPUT_ROOT}" "${CONTROL_DIR}" "${RS_DIR}" \
  "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
  "${ROWS_PER_ARM}" "${MIN_UNIQUE_REPAIRS}" "${RECERTIFY_TIMEOUT}" \
  "${RECERTIFY_STABILITY_RUNS}" "${LEARNING_RATE}" "${EPOCHS}" \
  "${MAX_STEPS}" "${TRAIN_SEED}" "${RS_LORA_DROPOUT}" \
  "${EXPECTED_PARENT_FIT_ROWS}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]).resolve()
value = json.loads(path.read_text())
payload = value.get("payload") or {}
rs = payload.get("rs_sft") or {}
view = payload.get("executable_train") or {}
expected = {
    "rows_per_arm": int(sys.argv[7]),
    "min_unique_repairs": int(sys.argv[8]),
    "recertification_timeout": int(sys.argv[9]),
    "recertification_stability_runs": int(sys.argv[10]),
    "learning_rate": float(sys.argv[11]),
    "epochs": float(sys.argv[12]),
    "max_steps": int(sys.argv[13]),
    "batch_size": 1,
    "grad_accum": 16,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": float(sys.argv[15]),
    "decoder_model": "Qwen/Qwen3-8B",
    "decoder_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "attn_implementation": "flash_attention_2",
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "load_4bit": False,
    "sequence_distribution_nll": False,
    "eval_strategy": "no",
    "seed": int(sys.argv[14]),
    "heldout_loaded_during_training": False,
}
if (
    value.get("schema") != "post-qwen-predeclared-training-chain-v1"
    or payload.get("stage_order_predeclared") is not True
    or payload.get("launch_decisions_from_heldout") is not False
    or pathlib.Path((payload.get("qwen_stage") or {}).get("checkpoint", {}).get("path", "")).resolve()
       != pathlib.Path(sys.argv[2]).resolve()
    or pathlib.Path(rs.get("matched_control_output", "")).resolve()
       != pathlib.Path(sys.argv[4]).resolve()
    or pathlib.Path(rs.get("intervention_output", "")).resolve()
       != pathlib.Path(sys.argv[5]).resolve()
    or (view.get("report") or {}).get("sha256") != sys.argv[6]
    or int(view.get("parent_rows", -1)) != int(sys.argv[16])
    or int(view.get("heldout_rows", -1)) != 175
    or int(rs.get("rows_per_arm", -1))
       != 2 * int(view.get("rows", -1))
    or rs.get("heldout_loaded_during_training") is not False
    or any(rs.get(key) != item for key, item in expected.items())
):
    raise SystemExit("RS invocation differs from the predeclared chain")
print("PREDECLARED_RS_CHAIN_VERIFIED", flush=True)
PY

CHAIN_CONTRACT_SHA256="$(sha256sum "${CHAIN_CONTRACT}" | awk '{print $1}')"
mkdir -p "${OUTPUT_ROOT}"

validate_data_dir() {
  "${PYTHON}" - "${DATA_DIR}" "${BASE_TRAIN}" "${BASE_TRAIN_SEAL}" \
    "${CONTRACT}" "${CHATGPT_REPAIRS}" "${CHATGPT_REPORT}" \
    "${EXECUTABLE_VIEW_REPORT}" "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
    "${ROWS_PER_ARM}" "${MIN_UNIQUE_REPAIRS}" "${TRAIN_SEED}" \
    "${EXPECTED_PARENT_FIT_ROWS}" <<'PY'
import hashlib, json, pathlib, sys
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
root=pathlib.Path(sys.argv[1]).resolve()
report_path=root/"build_report.json"
report=json.loads(report_path.read_text())
outputs=report.get("outputs") or {}
expected_files={
    "intervention":(root/"rs_sft_50_50.jsonl", root/"rs_sft_50_50.seal.json"),
    "control":(root/"gold_only_matched.jsonl", root/"gold_only_matched.seal.json"),
}
if (
    report.get("schema") != "direct-compact-rs-sft-matched-build-v1"
    or report.get("seed") != int(sys.argv[11])
    or report.get("contract_sha256") != sha(sys.argv[4])
    or (report.get("base_train") or {}).get("sha256") != sha(sys.argv[2])
    or (report.get("base_train") or {}).get("seal_sha256") != sha(sys.argv[3])
    or (report.get("executable_view") or {}).get("report", {}).get("sha256")
       != sys.argv[8]
    or report.get("low_coverage_smoke_override") is not False
    or int(report.get("unique_recertified_tasks", -1)) < int(sys.argv[10])
    or int(report.get("unique_repair_floor", -1)) != int(sys.argv[10])
    or (report.get("arms") or {}).get("rows_each") != int(sys.argv[9])
    or (report.get("sealed_fit_accounting") or {}).get("parent_fit_rows")
       != int(sys.argv[12])
    or (report.get("sealed_fit_accounting") or {}).get("executable_rows")
       != int(sys.argv[9]) // 2
    or (report.get("arms") or {}).get("source_sequence_exactly_matched") is not True
    or not any(
        item.get("sha256") == sha(sys.argv[5])
        and (item.get("collector_report") or {}).get("sha256") == sha(sys.argv[6])
        for item in report.get("repair_artifacts") or []
    )
):
    raise SystemExit("matched-data report contract failed")
for name,(dataset,seal_path) in expected_files.items():
    record=outputs.get(name) or {}
    seal=json.loads(seal_path.read_text())
    if (
        pathlib.Path(record.get("path","")).resolve() != dataset
        or record.get("sha256") != sha(dataset)
        or record.get("seal_sha256") != sha(seal_path)
        or seal.get("selected_role") != "fit"
        or int(seal.get("rows",-1)) != int(sys.argv[9])
        or seal.get("output_sha256") != sha(dataset)
        or seal.get("contract_sha256") != sha(sys.argv[4])
    ):
        raise SystemExit(f"matched-data {name} artifact contract failed")
if (
    outputs.get("schedule_sha256") != sha(root/"schedule.jsonl")
    or outputs.get("rejections_sha256") != sha(root/"rejected_repairs.jsonl")
):
    raise SystemExit("matched-data audit artifacts differ")
print("MATCHED_RS_DATA_REUSED", flush=True)
PY
}

quarantine_incomplete_dir() {
  local target="$1"
  local label="$2"
  if [[ ! -e "${target}" ]]; then
    return
  fi
  case "$(realpath -m -- "${target}")" in
    "$(realpath -m -- "${OUTPUT_ROOT}")"/*) ;;
    *)
      printf 'Refusing to move %s outside OUTPUT_ROOT: %s\n' "${label}" "${target}" >&2
      exit 2
      ;;
  esac
  local preserved="${target}.incomplete.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mv -- "${target}" "${preserved}"
  printf 'PRESERVED_INCOMPLETE_RS_ARTIFACT label=%s path=%s\n' \
    "${label}" "${preserved}" >&2
}

if [[ -d "${DATA_DIR}" ]] && validate_data_dir; then
  :
else
  quarantine_incomplete_dir "${DATA_DIR}" matched_data
  "${PYTHON}" "${PATCH_ROOT}/scripts/training/build_direct_compact_rs_sft.py" \
    --base_train "${BASE_TRAIN}" \
    --base_train_seal "${BASE_TRAIN_SEAL}" \
    --contract "${CONTRACT}" \
    --executable_view_report "${EXECUTABLE_VIEW_REPORT}" \
    --expected_executable_view_report_sha256 \
      "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
    --repairs "chatgpt=${CHATGPT_REPAIRS}" \
    --repair_report "chatgpt=${CHATGPT_REPORT}" \
    --output_dir "${DATA_DIR}" \
    --rows_per_arm "${ROWS_PER_ARM}" \
    --rows_per_arm_from_sealed_parent \
    --expected_parent_fit_rows "${EXPECTED_PARENT_FIT_ROWS}" \
    --min_unique_repairs "${MIN_UNIQUE_REPAIRS}" \
    --seed "${TRAIN_SEED}" \
    --workers "${RECERTIFY_WORKERS:-32}" \
    --timeout "${RECERTIFY_TIMEOUT}" \
    --stability_runs "${RECERTIFY_STABILITY_RUNS}"
  validate_data_dir
fi

train_arm() {
  local train_file="$1"
  local train_seal="$2"
  local output_dir="$3"
  local resume_mode="${4:-fresh}"
  local train_args=(
    --train_file "${train_file}" \
    --train_seal "${train_seal}" \
    --no_eval_during_training \
    --stage_contract "${CHAIN_CONTRACT}" \
    --expected_stage_contract_sha256 "${CHAIN_CONTRACT_SHA256}" \
    --output_dir "${output_dir}" \
    --contract "${CONTRACT}" \
    --codebook "${CODEBOOK}" \
    --codec_artifact "${CODEC}" \
    --tokenizer_json "${TOKENIZER_JSON}" \
    --decoder_model Qwen/Qwen3-8B \
    --decoder_revision b968826d9c46dd6066d109eabc6255188de91218 \
    --warmstart_checkpoint "${WARMSTART_CHECKPOINT}" \
    --learning_rate "${LEARNING_RATE}" \
    --epochs "${EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --logging_steps 10 \
    --eval_strategy no \
    --batch_size 1 \
    --grad_accum 16 \
    --seed "${TRAIN_SEED}" \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout "${RS_LORA_DROPOUT}" \
    --gradient_checkpointing \
    --bf16 \
    --attn_implementation flash_attention_2
  )
  if [[ "${resume_mode}" == resume ]]; then
    train_args+=(--resume_from_checkpoint auto)
  elif [[ "${resume_mode}" != fresh ]]; then
    printf 'Unknown RS training resume mode: %s\n' "${resume_mode}" >&2
    exit 2
  fi
  "${PYTHON}" -m scripts.training.direct_compact_qwen_decompiler \
    "${train_args[@]}"
}

validate_arm() {
  local train_file="$1"
  local train_seal="$2"
  local output_dir="$3"
  "${PYTHON}" - "${train_file}" "${train_seal}" "${output_dir}" \
    "${WARMSTART_CHECKPOINT}" "${CHAIN_CONTRACT}" "${CONTRACT}" \
    "${CODEBOOK}" "${CODEC}" "${LEARNING_RATE}" "${EPOCHS}" \
    "${MAX_STEPS}" "${TRAIN_SEED}" "${RS_LORA_DROPOUT}" \
    "${ROWS_PER_ARM}" <<'PY'
import hashlib, json, pathlib, sys
from models.direct_compact_causal import sha256_artifact
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
train,seal_path,out,warm,chain,contract,codebook,codec=map(
    lambda x:pathlib.Path(x).resolve(), sys.argv[1:9]
)
prov=json.loads((out/"run_provenance.json").read_text())
seal=json.loads(seal_path.read_text())
schedule=prov.get("training_schedule") or {}
warm_record=prov.get("warmstart_checkpoint") or {}
if (
    prov.get("schema") != "direct-compact-run-provenance-v1"
    or prov.get("architecture") != "qwen-causal-compact-tokens-no-encoder"
    or prov.get("train_file_sha256") != sha(train)
    or prov.get("train_seal_sha256") != sha(seal_path)
    or int(prov.get("train_sealed_rows",-1)) != int(seal.get("rows",-2))
    or int(seal.get("rows",-1)) != int(sys.argv[14])
    or prov.get("heldout_loaded_during_training") is not False
    or prov.get("eval_file_sha256") is not None
    or prov.get("eval_seal_sha256") is not None
    or prov.get("eval_strategy") != "no"
    or (prov.get("loss_contract") or {}).get("sequence_distribution_nll") is not False
    or (prov.get("stage_contract") or {}).get("sha256") != sha(chain)
    or prov.get("contract_sha256") != sha(contract)
    or prov.get("codebook_sha256") != sha(codebook)
    or prov.get("codec_sha256") != sha(codec)
    or pathlib.Path(warm_record.get("path","")).resolve() != warm
    or warm_record.get("decoder_adapter_sha256")
       != sha256_artifact(warm/"decoder_adapter")
    or warm_record.get("source_overlay_sha256")
       != sha(warm/"source_embedding_overlay.pt")
    or prov.get("decoder_adapter_sha256") != sha256_artifact(out/"decoder_adapter")
    or prov.get("source_overlay_sha256") != sha(out/"source_embedding_overlay.pt")
    or schedule != {
        "learning_rate":float(sys.argv[9]),
        "epochs":float(sys.argv[10]),
        "max_steps":int(sys.argv[11]),
        "batch_size":1,
        "grad_accum":16,
        "seed":int(sys.argv[12]),
        "lora_r":64,
        "lora_alpha":128,
        "lora_dropout":float(sys.argv[13]),
        "load_4bit":False,
        "gradient_checkpointing":True,
        "bf16":True,
        "fp16":False,
    }
):
    raise SystemExit("RS/control checkpoint contract failed")
print(f"RS_ARM_REUSED output={out}", flush=True)
PY
}

ensure_arm() {
  local train_file="$1"
  local train_seal="$2"
  local output_dir="$3"
  local label="$4"
  if [[ -d "${output_dir}" ]] \
    && validate_arm "${train_file}" "${train_seal}" "${output_dir}"; then
    return
  fi
  if [[ -d "${output_dir}" ]]; then
    if compgen -G "${output_dir}/checkpoint-*" >/dev/null; then
      printf 'AUTO_RESUME_RS_ARM label=%s output=%s\n' \
        "${label}" "${output_dir}"
      train_arm "${train_file}" "${train_seal}" "${output_dir}" resume
    else
      printf 'Incomplete RS arm has no resumable trainer checkpoint: %s\n' \
        "${output_dir}" >&2
      exit 2
    fi
  else
    train_arm "${train_file}" "${train_seal}" "${output_dir}" fresh
  fi
  validate_arm "${train_file}" "${train_seal}" "${output_dir}"
}

# Each exact completed arm is reused; an interrupted arm resumes from its
# highest sealed Trainer checkpoint and is never moved aside or restarted.
ensure_arm \
  "${DATA_DIR}/gold_only_matched.jsonl" \
  "${DATA_DIR}/gold_only_matched.seal.json" \
  "${CONTROL_DIR}" matched_control
ensure_arm \
  "${DATA_DIR}/rs_sft_50_50.jsonl" \
  "${DATA_DIR}/rs_sft_50_50.seal.json" \
  "${RS_DIR}" rs_intervention

"${PYTHON}" - "${CHAIN_CONTRACT}" "${DATA_DIR}/build_report.json" \
  "${CONTROL_DIR}/run_provenance.json" "${RS_DIR}/run_provenance.json" \
  "${OUTPUT_ROOT}/train_side_handoff.json" <<'PY'
import hashlib, json, os, pathlib, sys
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
chain=json.loads(pathlib.Path(sys.argv[1]).read_text())
build=json.loads(pathlib.Path(sys.argv[2]).read_text())
control=json.loads(pathlib.Path(sys.argv[3]).read_text())
rs=json.loads(pathlib.Path(sys.argv[4]).read_text())
if (
    build.get("low_coverage_smoke_override") is not False
    or build.get("unique_recertified_tasks", 0) < build.get("unique_repair_floor", 1)
    or control.get("heldout_loaded_during_training") is not False
    or rs.get("heldout_loaded_during_training") is not False
    or control.get("warmstart_checkpoint") != rs.get("warmstart_checkpoint")
    or (control.get("stage_contract") or {}).get("sha256") != sha(sys.argv[1])
    or (rs.get("stage_contract") or {}).get("sha256") != sha(sys.argv[1])
):
    raise SystemExit("train-side RS/control handoff failed")
payload={
  "schema":"post-qwen-rs-train-side-handoff-v1",
  "predeclared_chain_sha256":sha(sys.argv[1]),
  "matched_data_build_sha256":sha(sys.argv[2]),
  "control_provenance_sha256":sha(sys.argv[3]),
  "rs_provenance_sha256":sha(sys.argv[4]),
  "heldout_loaded_during_training":False,
  "heldout_metrics_used_for_selection":False,
  "passed":True,
}
out=pathlib.Path(sys.argv[5])
if out.exists():
    if json.loads(out.read_text()) != payload:
        raise SystemExit("existing train-side handoff differs")
    print("RS_TRAIN_SIDE_HANDOFF_REUSED", flush=True)
else:
    with out.open("x", encoding="utf-8") as f:
        json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    print("RS_TRAIN_SIDE_HANDOFF_READY", flush=True)
PY

printf 'Matched RS-SFT/control checkpoints trained without opening heldout175: %s\n' \
  "${OUTPUT_ROOT}/train_side_handoff.json"
