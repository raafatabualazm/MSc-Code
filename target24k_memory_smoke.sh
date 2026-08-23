#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace
PATCH_ROOT="${ROOT}/hybrid_training_patch_v2_3"
SMOKE_ROOT="${ROOT}/artifacts/direct_compact_qwen38_inline_cfg_v2/target24k_memory_smoke"
OUTPUT="${SMOKE_ROOT}/checkpoint"
MEMORY_LOG="${SMOKE_ROOT}/gpu_memory.csv"

test -f "${SMOKE_ROOT}/train.jsonl"
test -f "${SMOKE_ROOT}/train.seal.json"
test -f "${SMOKE_ROOT}/build.report.json"
test -d \
  "${ROOT}/artifacts/direct_compact_qwen38_inline_cfg_v2/direct_compact_multifunction_gold_sft_target24k"
if [[ -e "${OUTPUT}" ]]; then
  printf 'Refusing to overwrite target24k memory-smoke checkpoint: %s\n' \
    "${OUTPUT}" >&2
  exit 2
fi

cd "${PATCH_ROOT}"
printf 'timestamp,memory_used_mib,utilization_gpu_percent\n' >"${MEMORY_LOG}"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >>"${MEMORY_LOG}"
    sleep 2
  done
) &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${PATCH_ROOT}" \
/venv/main/bin/python -m scripts.training.direct_compact_qwen_decompiler \
  --train_file "${SMOKE_ROOT}/train.jsonl" \
  --train_seal "${SMOKE_ROOT}/train.seal.json" \
  --no_eval_during_training \
  --output_dir "${OUTPUT}" \
  --contract \
    "${ROOT}/multifunction_v1/build/multifunction_inline_cfg_v2_target24k_contract.json" \
  --codebook \
    "${ROOT}/multifunction_v1/build/multifunction_inline_cfg_v2_codebook.json" \
  --codec_artifact "${ROOT}/scripts/data/build_multifunction_compact_v2.py" \
  --tokenizer_json \
    "${ROOT}/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json" \
  --warmstart_checkpoint \
    "${ROOT}/artifacts/direct_compact_qwen38_inline_cfg_v2/direct_compact_multifunction_gold_sft_target24k" \
  --learning_rate 1e-6 \
  --epochs 1 \
  --max_steps 1 \
  --batch_size 1 \
  --grad_accum 1 \
  --eval_strategy no \
  --logging_steps 1 \
  --save_steps 1 \
  --sequence_distribution_nll \
  --sequence_nll_position_chunk_size 512 \
  --gradient_checkpointing \
  --bf16

/venv/main/bin/python - <<'PY'
import csv
import json
from pathlib import Path

root = Path(
    "/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2/"
    "target24k_memory_smoke"
)
provenance = json.loads(
    (root / "checkpoint" / "run_provenance.json").read_text(encoding="utf-8")
)
with (root / "gpu_memory.csv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
peak = max(int(row["memory_used_mib"]) for row in rows)
result = {
    "schema": "direct-compact-target24k-memory-smoke-result-v1",
    "max_steps": provenance.get("max_steps"),
    "loss_contract": provenance.get("loss_contract"),
    "train_rows": provenance.get("train_sealed_rows"),
    "heldout_loaded_during_training": provenance.get(
        "heldout_loaded_during_training"
    ),
    "peak_memory_used_mib": peak,
    "passed": (
        provenance.get("max_steps") == 1
        and (provenance.get("loss_contract") or {}).get(
            "sequence_distribution_nll"
        ) is True
        and (provenance.get("loss_contract") or {}).get(
            "sequence_target_suffix_logits_only"
        ) is True
        and provenance.get("train_sealed_rows") == 1
        and provenance.get("heldout_loaded_during_training") is False
    ),
}
if not result["passed"]:
    raise SystemExit("target24k memory-smoke provenance failed")
(root / "result.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    "TARGET24K_MEMORY_SMOKE_PASS "
    f"peak_memory_mib={peak} "
    f"loss_mode={result['loss_contract']['primary_reduction']}",
    flush=True,
)
PY
