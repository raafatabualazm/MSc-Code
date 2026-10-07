#!/usr/bin/env bash
set -euo pipefail

# Paired held-out evaluation of the sealed stage-1 checkpoint versus the
# adapter-only Kimi pass-2 continuation.  Historical two-epoch outcomes are
# read only after POST generation/scoring to report recovery of the five
# earlier pass@10 regressions; they never influence model-visible inputs.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
HISTORICAL_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
PRE_DIR="${WORKSPACE}/artifacts/t5gemma2_mixed_rs_sft_passk_v1"
PASS2_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_mixed_rs_sft_kimi_pass2_v1"
PASS2_CHECKPOINT="${PASS2_DIR}/checkpoint-optstep-000013"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_mixed_rs_sft_kimi_pass2_passk_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -s "${PASS2_DIR}/result.json" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-mixed-rs-sft-run-v1"
     and .status == "complete"
     and .architecture == "native_encoder_decoder"
     and .updates == 13
     and .planned_updates == 13
     and .rows == 104
     and .latest_checkpoint == "checkpoint-optstep-000013"' \
    "${PASS2_DIR}/result.json" >/dev/null; then
  echo "T5GEMMA_KIMI_PASS2_EVAL_BLOCKED completed 104-row/13-update pass-2 result is absent or changed" >&2
  exit 78
fi
if [[ ! -d "${PASS2_CHECKPOINT}" ]] \
  || [[ ! -s "${PASS2_CHECKPOINT}/adapter/adapter_model.safetensors" ]] \
  || [[ ! -s "${PASS2_CHECKPOINT}/adapter/adapter_config.json" ]] \
  || [[ ! -s "${PASS2_CHECKPOINT}/tokenizer/tokenizer.json" ]] \
  || [[ ! -s "${PASS2_CHECKPOINT}/run_contract.json" ]] \
  || [[ ! -s "${PASS2_DIR}/run_contract.json" ]] \
  || [[ ! -s "${PASS2_DIR}/dataset_manifest.json" ]] \
  || ! cmp -s "${PASS2_DIR}/run_contract.json" \
    "${PASS2_CHECKPOINT}/run_contract.json" \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-mixed-rs-sft-run-v1"
     and .status == "training"
     and .architecture == "native_encoder_decoder"
     and .optimization.planned_updates == 13
     and .optimization.epochs == 1
     and .optimization.batch_size == 1
     and .optimization.gradient_accumulation == 8
     and .optimization.learning_rate == 0.00001
     and .optimization.seed == 42
     and .dataset.rows == 104
     and .dataset.composition.verified_direct == 13
     and .dataset.composition.repair_conditioned == 13
     and .dataset.composition.gold_replay == 78
     and .dataset.gold_replay.ratio_when_rows_is_minus_one == 3
     and .dataset.heldout_overlap == 0
     and ([.dataset.reports[].sha256] | sort)
       == ([
         "fe2941885767f7c4abb3012d1a49c22a934a6b67d8f1f9626bf09e44a3d633d0",
         "fe9bcd00c6774432b7911129246c8b2837523d85b1c94efb29c03f85ae860205"
       ] | sort)
     and .warmstart.checkpoint_name == "checkpoint-optstep-000426"
     and .warmstart.update == 426
     and .warmstart.adapter_weights_sha256
       == "be95b3bec613478790facb2bb6ec29243a3625e468560b0b678def8f016a46da"
     and .warmstart.adapter_config_sha256
       == "75aad08619087aa2d7ef5db9d50b1890ca74feccf85a2c104a93dbddb0d7b9e6"
     and .privacy.heldout_content_model_visible == false
     and .privacy.tests_model_visible == false
     and .privacy.private_feedback_model_visible == false
     and .lora.encoder_and_decoder_trainable == true
     and .lora.new_adapter_attached == false
     and .lora.warmstart_weights_continued == true' \
    "${PASS2_CHECKPOINT}/run_contract.json" >/dev/null \
  || ! /usr/bin/jq -e -s '.[0].dataset == .[1]' \
    "${PASS2_CHECKPOINT}/run_contract.json" \
    "${PASS2_DIR}/dataset_manifest.json" >/dev/null; then
  echo "T5GEMMA_KIMI_PASS2_EVAL_BLOCKED checkpoint/dataset/warm-start contract failed" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_KIMI_PASS2_EVAL_BLOCKED Dart 3.12.2 is not executable" >&2
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

HISTORICAL_PREDICTIONS="${HISTORICAL_DIR}/two_epoch_k10_predictions.json"
HISTORICAL_SCORE="${HISTORICAL_DIR}/two_epoch_k10_score.json"
PRE_PREDICTIONS="${PRE_DIR}/post_mixed_k10_predictions.json"
PRE_SCORE="${PRE_DIR}/post_mixed_k10_score.json"
POST_PREDICTIONS="${OUTPUT_DIR}/post_kimi_pass2_k10_predictions.json"
POST_SCORE="${OUTPUT_DIR}/post_kimi_pass2_k10_score.json"
POST_COMPAT="${POST_PREDICTIONS}.checkpoint-loader-compat.json"
for path in \
  "${HISTORICAL_PREDICTIONS}" \
  "${HISTORICAL_PREDICTIONS}.provenance.json" \
  "${HISTORICAL_PREDICTIONS}.generation.journal.jsonl" \
  "${HISTORICAL_SCORE}" \
  "${PRE_PREDICTIONS}" \
  "${PRE_PREDICTIONS}.provenance.json" \
  "${PRE_PREDICTIONS}.generation.journal.jsonl" \
  "${PRE_SCORE}"; do
  if [[ ! -s "${path}" ]]; then
    echo "T5GEMMA_KIMI_PASS2_EVAL_BLOCKED paired reference artifact is absent: ${path}" >&2
    exit 78
  fi
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
/venv/main/bin/python scripts/evaluation/t5gemma2_f2_passk_mixed_compat.py \
  --compat_record "${POST_COMPAT}" \
  --compat_checkpoint "${PASS2_CHECKPOINT}" \
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  --arm sft \
  --num_samples 10 \
  --generation_batch_size 10 \
  --max_source_tokens 32768 \
  --max_new_tokens 4096 \
  --temperature 0.8 \
  --top_p 0.95 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16 \
  --output "${POST_PREDICTIONS}"

/venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
  --predictions "${POST_PREDICTIONS}" \
  --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --output "${POST_SCORE}" \
  --k 10 \
  --workers 32 \
  --timeout 30 \
  --stability_runs 2

# This is deliberately after POST generation and scoring.  No historical
# task outcome, test, diagnostic, or regression identity enters inference.
/venv/main/bin/python scripts/evaluation/t5gemma2_kimi_pass2_compare.py \
  --historical_predictions "${HISTORICAL_PREDICTIONS}" \
  --historical_score "${HISTORICAL_SCORE}" \
  --pre_predictions "${PRE_PREDICTIONS}" \
  --pre_score "${PRE_SCORE}" \
  --post_predictions "${POST_PREDICTIONS}" \
  --post_score "${POST_SCORE}" \
  --post_compat "${POST_COMPAT}" \
  --output "${OUTPUT_DIR}/comparison.json"

echo "T5GEMMA_KIMI_PASS2_EVAL_COMPLETE output=${OUTPUT_DIR}/comparison.json"
