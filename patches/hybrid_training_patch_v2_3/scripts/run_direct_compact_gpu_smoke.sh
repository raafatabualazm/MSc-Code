#!/usr/bin/env bash
set -euo pipefail

# Non-destructive Qwen3-8B smoke for the encoder-free compact-source path.
# It waits for the currently running graph causality evaluation rather than
# contending with it, then performs one optimizer step, reloads the separately
# saved adapter/overlay, runs a small NLL conditioning probe, and exercises
# cached generation.

STAGE_ROOT="${STAGE_ROOT:-/workspace/direct_compact_stage}"
PROJECT_ROOT="${PROJECT_ROOT:-${STAGE_ROOT}/hybrid_training_patch_v2_3}"
DATA_ROOT="${DATA_ROOT:-${STAGE_ROOT}/confirmatory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/artifacts/direct_compact_v1_smoke}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
HF_SNAPSHOT="${HF_SNAPSHOT:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${HF_SNAPSHOT}/tokenizer.json}"
DECODER_MODEL="${DECODER_MODEL:-Qwen/Qwen3-8B}"
DECODER_REVISION="${DECODER_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
WAIT_PATTERN="${WAIT_PATTERN:-functional_graph_gate_antigravity.py}"
PREREQUISITE_DONE="${PREREQUISITE_DONE:-/workspace/artifacts/hybrid_v2_3_s44/clean12k_eval.done}"
PREREQUISITE_PATTERN="${PREREQUISITE_PATTERN:-launch_clean12k_eval.sh}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
EXPECTED_DATA_MANIFEST_SHA256="${EXPECTED_DATA_MANIFEST_SHA256:-cf64b75dff2a502e1881ee4ca58ee2b4ac81d2cf403a9131545c94203f19b7cd}"
EXPECTED_JOINED_MANIFEST_SHA256="${EXPECTED_JOINED_MANIFEST_SHA256:-dfb2477b17a1e63d34a17ca74c0a346dece4458d7e697a1f137f2edc84c7f254}"

STATUS="${OUTPUT_ROOT}.status"
LOG="${OUTPUT_ROOT}.log"
mkdir -p "$(dirname "${OUTPUT_ROOT}")"

on_exit() {
  rc=$?
  if (( rc != 0 )); then
    printf '%s failed_exit_%s\n' "$(date -u +%FT%TZ)" "${rc}" >"${STATUS}"
  fi
}
trap on_exit EXIT

status() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" | tee "${STATUS}"
}

status "waiting_for_active_graph_gate"
while pgrep -af "${WAIT_PATTERN}" | grep -v "run_direct_compact_gpu_smoke" >/dev/null; do
  sleep 30
done

status "waiting_for_clean12k_prerequisite"
while [[ ! -f "${PREREQUISITE_DONE}" ]] || pgrep -af "${PREREQUISITE_PATTERN}" | grep -v "run_direct_compact_gpu_smoke" >/dev/null; do
  sleep 30
done

while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  status "waiting_for_free_vram_${free_mib:-unknown}_MiB"
  sleep 30
done

status "validating_inputs"
"${PYTHON_BIN}" - "${DATA_ROOT}" "${PROJECT_ROOT}" "${TOKENIZER_JSON}" \
  "${EXPECTED_DATA_MANIFEST_SHA256}" "${EXPECTED_JOINED_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data = Path(sys.argv[1])
project = Path(sys.argv[2])
tokenizer = Path(sys.argv[3])
expected_manifest_hashes = {
    "SHA256SUMS.txt": sys.argv[4],
    "JOINED_SHA256SUMS.txt": sys.argv[5],
}
contract_path = data / "compact_contract.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks = {
    data / "codebook.json": contract["codebook_sha256"],
    project.parent / "scripts" / "data" / "build_compact_qwen_v1.py": contract["codec_sha256"],
    tokenizer: contract["tokenizer_json_sha256"],
}
for path, expected in checks.items():
    if not path.is_file():
        raise SystemExit(f"missing bound input: {path}")
    observed = sha(path)
    if observed != expected:
        raise SystemExit(f"SHA mismatch for {path}: {observed} != {expected}")
for checksum_name, checksum_expected in expected_manifest_hashes.items():
    checksum_path = data / checksum_name
    if not checksum_path.is_file():
        raise SystemExit(f"missing checksum manifest: {checksum_path}")
    if sha(checksum_path) != checksum_expected:
        raise SystemExit(f"checksum manifest is not the frozen artifact: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(None, 1)
        name = name.strip()
        if Path(name).name != name:
            raise SystemExit(f"unsafe checksum-manifest path: {name}")
        path = data / name
        if not path.is_file():
            raise SystemExit(f"missing checksummed input: {path}")
        observed = sha(path)
        if observed != expected:
            raise SystemExit(f"SHA mismatch for {path}: {observed} != {expected}")
contract_sha = sha(contract_path)
for stem, role in (("train_supervised", "fit"), ("dev_supervised", "measure")):
    dataset = data / f"{stem}.jsonl"
    seal_path = data / f"{stem}.seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != "compact-public-private-join-seal-v1":
        raise SystemExit(f"invalid join seal schema: {seal_path}")
    if seal.get("selected_role") != role:
        raise SystemExit(f"join seal role mismatch: {seal_path}")
    if seal.get("output_sha256") != sha(dataset):
        raise SystemExit(f"join seal output mismatch: {dataset}")
    if seal.get("contract_sha256") != contract_sha:
        raise SystemExit(f"join seal contract mismatch: {seal_path}")
    if seal.get("public_sha256") != sha(data / "compact_model_inputs.jsonl"):
        raise SystemExit(f"join seal public-source mismatch: {seal_path}")
    if seal.get("alignment_sha256") != sha(data / "alignment_private.jsonl"):
        raise SystemExit(f"join seal alignment mismatch: {seal_path}")
print(json.dumps({"contract_sha256": sha(contract_path), "inputs_valid": True}, sort_keys=True))
PY

status "training_one_step"
cd "${PROJECT_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

{
  "${PYTHON_BIN}" -m scripts.training.direct_compact_qwen_decompiler \
    --train_file "${DATA_ROOT}/train_supervised.jsonl" \
    --eval_file "${DATA_ROOT}/dev_supervised.jsonl" \
    --train_seal "${DATA_ROOT}/train_supervised.seal.json" \
    --eval_seal "${DATA_ROOT}/dev_supervised.seal.json" \
    --output_dir "${OUTPUT_ROOT}" \
    --contract "${DATA_ROOT}/compact_contract.json" \
    --codebook "${DATA_ROOT}/codebook.json" \
    --codec_artifact "${PROJECT_ROOT}/../scripts/data/build_compact_qwen_v1.py" \
    --tokenizer_json "${TOKENIZER_JSON}" \
    --decoder_model "${DECODER_MODEL}" \
    --decoder_revision "${DECODER_REVISION}" \
    --learning_rate 1e-4 \
    --epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --eval_strategy no \
    --batch_size 1 \
    --grad_accum 1 \
    --seed 42 \
    --lora_r 16 \
    --lora_alpha 32 \
    --gradient_checkpointing \
    --bf16 \
    --attn_implementation flash_attention_2

  status "running_conditioning_probe"
  "${PYTHON_BIN}" -m scripts.evaluation.probe_direct_compact_conditioning \
    --dataset "${DATA_ROOT}/dev_supervised.jsonl" \
    --report "${OUTPUT_ROOT}/conditioning_probe_8.json" \
    --contract "${DATA_ROOT}/compact_contract.json" \
    --codebook "${DATA_ROOT}/codebook.json" \
    --codec_artifact "${PROJECT_ROOT}/../scripts/data/build_compact_qwen_v1.py" \
    --source_overlay "${OUTPUT_ROOT}/source_embedding_overlay.pt" \
    --decoder_adapter "${OUTPUT_ROOT}/decoder_adapter" \
    --tokenizer_json "${TOKENIZER_JSON}" \
    --decoder_model "${DECODER_MODEL}" \
    --decoder_revision "${DECODER_REVISION}" \
    --batch_size 1 \
    --limit 8 \
    --seed 42 \
    --bf16 \
    --attn_implementation flash_attention_2

  status "running_cached_generation"
  "${PYTHON_BIN}" -m scripts.evaluation.direct_compact_qwen_inference \
    --dataset "${DATA_ROOT}/compact_model_inputs.jsonl" \
    --alignment "${DATA_ROOT}/alignment_private.jsonl" \
    --role measure \
    --output "${OUTPUT_ROOT}/generation_smoke.json" \
    --contract "${DATA_ROOT}/compact_contract.json" \
    --codebook "${DATA_ROOT}/codebook.json" \
    --codec_artifact "${PROJECT_ROOT}/../scripts/data/build_compact_qwen_v1.py" \
    --source_overlay "${OUTPUT_ROOT}/source_embedding_overlay.pt" \
    --decoder_adapter "${OUTPUT_ROOT}/decoder_adapter" \
    --tokenizer_json "${TOKENIZER_JSON}" \
    --decoder_model "${DECODER_MODEL}" \
    --decoder_revision "${DECODER_REVISION}" \
    --batch_size 1 \
    --limit 2 \
    --max_new_tokens 64 \
    --num_samples 1 \
    --seed 42 \
    --bf16 \
    --attn_implementation flash_attention_2
} >>"${LOG}" 2>&1

status "validating_outputs"
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY' >>"${LOG}" 2>&1
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run = json.loads((root / "run_provenance.json").read_text(encoding="utf-8"))
probe = json.loads((root / "conditioning_probe_8.json").read_text(encoding="utf-8"))
generation = json.loads(
    (root / "generation_smoke.json.provenance.json").read_text(encoding="utf-8")
)
if probe.get("schema") != "direct-compact-conditioning-probe-v2":
    raise SystemExit("conditioning probe is not the position-matched v2 protocol")
if probe.get("null_ablation") != "position-matched-zero-source-embeddings":
    raise SystemExit("conditioning probe null arm is not position matched")
for record, label in ((probe, "probe"), (generation, "generation")):
    if record.get("source_overlay_sha256") != run.get("source_overlay_sha256"):
        raise SystemExit(f"{label} overlay is not bound to the trained checkpoint")
    if record.get("decoder_adapter_sha256") != run.get("decoder_adapter_sha256"):
        raise SystemExit(f"{label} adapter is not bound to the trained checkpoint")
if int(run.get("lm_head_rows", -1)) != 151936:
    raise SystemExit("training provenance reports an expanded LM head")
if run.get("graph_encoder") is not None or run.get("soft_prefix") is not None:
    raise SystemExit("training provenance unexpectedly reports an encoder/prefix")
print(json.dumps({"checkpoint_bindings_valid": True}, sort_keys=True))
PY

status "passed"
