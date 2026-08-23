#!/usr/bin/env bash
set -euo pipefail

# Four new frozen-checkpoint replicates.  Seed 42 already exists in the sealed
# two-epoch evaluation and is intentionally reused by the final audit report.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
REFERENCE_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_measurement_audit_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -s "${CHECKPOINT}/adapter/adapter_model.safetensors" \
   || ! -s "${CHECKPOINT}/tokenizer/tokenizer.json" \
   || ! -s "${CHECKPOINT}/run_contract.json" ]]; then
  echo "MEASUREMENT_AUDIT_BLOCKED frozen checkpoint is incomplete" >&2
  exit 78
fi
if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-enriched-sft-run-v1"
   and .architecture == "native_encoder_decoder"
   and .optimization.epochs == 2
   and .optimization.planned_updates == 348
   and .optimization.seed == 42
   and .lora.rank == 64
   and .lora.alpha == 128' \
  "${CHECKPOINT}/run_contract.json" >/dev/null; then
  echo "MEASUREMENT_AUDIT_BLOCKED frozen checkpoint contract differs" >&2
  exit 78
fi
if [[ ! -s "${REFERENCE_DIR}/two_epoch_k10_predictions.json" \
   || ! -s "${REFERENCE_DIR}/two_epoch_k10_score.json" ]]; then
  echo "MEASUREMENT_AUDIT_BLOCKED reusable seed-42 evaluation is absent" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "MEASUREMENT_AUDIT_BLOCKED pinned Dart binary is absent" >&2
  exit 78
fi

printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a \
  "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab \
  "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 \
  "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c -

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
for seed in 43 44 45 46; do
  predictions="${OUTPUT_DIR}/baseline_seed${seed}_k10_predictions.json"
  score="${OUTPUT_DIR}/baseline_seed${seed}_k10_score.json"
  /venv/main/bin/python scripts/evaluation/t5gemma2_f2_passk_inference.py \
    --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
    --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
    --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
    --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
    --sft_checkpoint "${CHECKPOINT}" \
    --arm sft \
    --num_samples 10 \
    --generation_batch_size 10 \
    --max_source_tokens 32768 \
    --max_new_tokens 4096 \
    --temperature 0.8 \
    --top_p 0.95 \
    --seed "${seed}" \
    --attn_implementation sdpa \
    --bf16 \
    --output "${predictions}"
  /venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
    --predictions "${predictions}" \
    --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
    --k 10 \
    --workers 32 \
    --timeout 30 \
    --stability_runs 2 \
    --output "${score}"
  echo "MEASUREMENT_AUDIT_RESEED_COMPLETE seed=${seed} score=${score}"
done

echo "MEASUREMENT_AUDIT_BASELINE_RESEEDS_COMPLETE output=${OUTPUT_DIR}"
