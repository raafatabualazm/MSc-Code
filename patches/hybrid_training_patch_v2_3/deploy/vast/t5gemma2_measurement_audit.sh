#!/usr/bin/env bash
set -euo pipefail

# Complete frozen-checkpoint measurement audit.  Seed 42 is reused from the
# sealed two-epoch evaluation; seeds 43..46 are exact-resumable if the separate
# baseline supervisor has not already completed them.
WORKSPACE="${T5GEMMA_MEASUREMENT_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
REFERENCE_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
OUTPUT_DIR="${T5GEMMA_MEASUREMENT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_f2_measurement_audit_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
BASELINE_PROGRAM="${T5GEMMA_MEASUREMENT_BASELINE_PROGRAM:-t5gemma-measurement-baseline-reseeds}"
SUPERVISORCTL="${T5GEMMA_MEASUREMENT_SUPERVISORCTL:-/usr/local/bin/supervisorctl}"
WAIT_INTERVAL="${T5GEMMA_MEASUREMENT_WAIT_INTERVAL:-30}"
MAX_WAIT_SECONDS="${T5GEMMA_MEASUREMENT_MAX_WAIT_SECONDS:-86400}"
GOLD_SCORE="${OUTPUT_DIR}/gold_roundtrip/gold_k1_score.json"

for value_name in WAIT_INTERVAL MAX_WAIT_SECONDS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MEASUREMENT_AUDIT_BLOCKED invalid ${value_name}: ${value}" >&2
    exit 78
  fi
done
if [[ ! -x "${SUPERVISORCTL}" ]]; then
  echo "MEASUREMENT_AUDIT_BLOCKED Supervisor client is not executable" >&2
  exit 78
fi

# Never contend with the live reseed process for a durable journal.  Supervisor
# state plus complete artifact presence is the handoff contract.
started_at="$(date +%s)"
while true; do
  set +e
  status_line="$("${SUPERVISORCTL}" status "${BASELINE_PROGRAM}" 2>&1)"
  status_rc=$?
  set -e
  if [[ -z "${status_line}" ]]; then
    echo "MEASUREMENT_AUDIT_BLOCKED empty Supervisor response (rc=${status_rc})" >&2
    exit 78
  fi
  baseline_state="$(printf '%s\n' "${status_line}" | /usr/bin/awk '{print $2}')"
  case "${baseline_state}" in
    RUNNING|STARTING|STOPPING)
      now="$(date +%s)"
      if (( now - started_at >= MAX_WAIT_SECONDS )); then
        echo "MEASUREMENT_AUDIT_BLOCKED timed out waiting for ${BASELINE_PROGRAM}" >&2
        exit 78
      fi
      echo "MEASUREMENT_AUDIT_WAITING program=${BASELINE_PROGRAM} state=${baseline_state}"
      sleep "${WAIT_INTERVAL}"
      ;;
    EXITED)
      for seed in 43 44 45 46; do
        stem="${OUTPUT_DIR}/baseline_seed${seed}_k10"
        for required in \
          "${stem}_predictions.json" \
          "${stem}_predictions.json.provenance.json" \
          "${stem}_predictions.json.generation.journal.jsonl" \
          "${stem}_predictions.json.generation.journal.jsonl.chain-head.json" \
          "${stem}_score.json" \
          "${stem}_score.json.evaluation.journal.jsonl" \
          "${stem}_score.json.evaluation.journal.jsonl.chain-head.json"; do
          if [[ ! -s "${required}" ]]; then
            echo "MEASUREMENT_AUDIT_BLOCKED baseline handoff missing ${required}" >&2
            exit 78
          fi
        done
      done
      break
      ;;
    STOPPED|FATAL|BACKOFF|UNKNOWN)
      echo "MEASUREMENT_AUDIT_BLOCKED ${BASELINE_PROGRAM} state=${baseline_state}" >&2
      exit 78
      ;;
    *)
      echo "MEASUREMENT_AUDIT_BLOCKED unrecognized Supervisor state: ${status_line}" >&2
      exit 78
      ;;
  esac
done

# Rank 0 validates that the evaluator and all private test joins accept the
# exact gold programs before any diagnostic ablation spends GPU time.
if [[ ! -s "${GOLD_SCORE}" ]] || ! /usr/bin/jq -e \
  '.schema == "direct-compact-attested-passk-v1"
   and .tasks == 175
   and .k == 1
   and .pass_at_1.count == 175
   and .pass_at_k.count == 175
   and .compile_at_k.count == 175
   and .timeout == 30
   and .stability_runs == 2' "${GOLD_SCORE}" >/dev/null; then
  echo "MEASUREMENT_AUDIT_BLOCKED Rank-0 gold round-trip is absent or failed" >&2
  exit 78
fi

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

common_data=(
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl"
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json"
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl"
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json"
  --sft_checkpoint "${CHECKPOINT}"
  --arm sft
  --num_samples 10
  --generation_batch_size 10
  --max_source_tokens 32768
  --max_new_tokens 4096
  --temperature 0.8
  --top_p 0.95
  --attn_implementation sdpa
  --bf16
)
common_score=(
  --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl"
  --k 10
  --workers 32
  --timeout 30
  --stability_runs 2
)

# The standalone reseed supervisor normally produces these first.  Invoking
# the full launcher after it exits is O(verification); incomplete journals
# resume at the next exact task rather than silently restarting.
for seed in 43 44 45 46; do
  predictions="${OUTPUT_DIR}/baseline_seed${seed}_k10_predictions.json"
  score="${OUTPUT_DIR}/baseline_seed${seed}_k10_score.json"
  /venv/main/bin/python scripts/evaluation/t5gemma2_f2_passk_inference.py \
    "${common_data[@]}" --seed "${seed}" --output "${predictions}"
  /venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
    --predictions "${predictions}" --output "${score}" "${common_score[@]}"
done

for view in semantic_body_swap constants_stripped typed_opaque_contract; do
  predictions="${OUTPUT_DIR}/${view}_seed42_k10_predictions.json"
  score="${OUTPUT_DIR}/${view}_seed42_k10_score.json"
  /venv/main/bin/python \
    scripts/evaluation/t5gemma2_measurement_audit_inference.py \
    "${common_data[@]}" --input_view "${view}" --seed 42 \
    --output "${predictions}"
  /venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
    --predictions "${predictions}" --output "${score}" "${common_score[@]}"
  echo "MEASUREMENT_AUDIT_ABLATION_COMPLETE view=${view} score=${score}"
done

report_args=(
  --baseline "42|${REFERENCE_DIR}/two_epoch_k10_predictions.json|${REFERENCE_DIR}/two_epoch_k10_score.json"
)
for seed in 43 44 45 46; do
  report_args+=(
    --baseline "${seed}|${OUTPUT_DIR}/baseline_seed${seed}_k10_predictions.json|${OUTPUT_DIR}/baseline_seed${seed}_k10_score.json"
  )
done
for view in semantic_body_swap constants_stripped typed_opaque_contract; do
  report_args+=(
    --ablation "${view}|${OUTPUT_DIR}/${view}_seed42_k10_predictions.json|${OUTPUT_DIR}/${view}_seed42_k10_score.json"
  )
done
/venv/main/bin/python scripts/evaluation/t5gemma2_measurement_audit_report.py \
  "${report_args[@]}" --gold_score "${GOLD_SCORE}" \
  --expected_tasks 175 --k 10 \
  --output "${OUTPUT_DIR}/measurement_report.json"

echo "MEASUREMENT_AUDIT_COMPLETE output=${OUTPUT_DIR}/measurement_report.json"
