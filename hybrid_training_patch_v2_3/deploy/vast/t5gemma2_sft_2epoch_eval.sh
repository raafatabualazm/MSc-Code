#!/usr/bin/env bash
set -euo pipefail

# Read-only handoff from the isolated two-epoch SFT arm to the fixed held-out
# evaluation.  This script never starts, stops, or signals the trainer.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
ONE_EPOCH_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_v1"
TWO_EPOCH_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1"
ONE_EPOCH_CHECKPOINT="${ONE_EPOCH_DIR}/checkpoint-optstep-000174"
TWO_EPOCH_CHECKPOINT="${TWO_EPOCH_DIR}/checkpoint-optstep-000348"
REFERENCE_EVAL_DIR="${WORKSPACE}/artifacts/t5gemma2_prepost_passk_v1"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
TRAIN_PROGRAM="${T5GEMMA_2E_TRAIN_PROGRAM:-t5gemma-sft-2epoch}"
SUPERVISORCTL="${T5GEMMA_2E_SUPERVISORCTL:-/usr/local/bin/supervisorctl}"
WAIT_INTERVAL="${T5GEMMA_2E_EVAL_WAIT_INTERVAL:-30}"
MAX_WAIT_SECONDS="${T5GEMMA_2E_EVAL_MAX_WAIT_SECONDS:-86400}"

for value_name in WAIT_INTERVAL MAX_WAIT_SECONDS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "T5GEMMA_2E_EVAL_BLOCKED invalid ${value_name}: ${value}" >&2
    exit 78
  fi
done
if [[ ! -x "${SUPERVISORCTL}" ]]; then
  echo "T5GEMMA_2E_EVAL_BLOCKED Supervisor client is not executable: ${SUPERVISORCTL}" >&2
  exit 78
fi

started_at="$(date +%s)"
while true; do
  # supervisorctl deliberately returns a non-zero status for terminal process
  # states (for example, rc=3 for EXITED).  Capture that status without letting
  # `set -e` turn a successfully completed trainer into a handoff failure; the
  # parsed Supervisor state below remains the fail-closed source of truth.
  set +e
  status_line="$(
    "${SUPERVISORCTL}" status "${TRAIN_PROGRAM}" 2>&1
  )"
  status_rc=$?
  set -e
  if [[ -z "${status_line}" ]]; then
    echo "T5GEMMA_2E_EVAL_BLOCKED empty Supervisor response (rc=${status_rc})" >&2
    exit 78
  fi
  train_state="$(printf '%s\n' "${status_line}" | /usr/bin/awk '{print $2}')"
  result_complete=false
  if [[ -f "${TWO_EPOCH_DIR}/result.json" ]] \
    && /usr/bin/jq -e \
      '.schema == "t5gemma2-enriched-sft-run-v1"
       and .status == "complete"
       and .updates == 348
       and .planned_updates == 348
       and .rows == 2776
       and .latest_checkpoint == "checkpoint-optstep-000348"
       and .no_frontier_api == true' \
      "${TWO_EPOCH_DIR}/result.json" >/dev/null; then
    result_complete=true
  fi

  if [[ "${result_complete}" == true && "${train_state}" == EXITED ]]; then
    break
  fi
  case "${train_state}" in
    RUNNING|STARTING|STOPPING)
      ;;
    EXITED|FATAL|BACKOFF|STOPPED|UNKNOWN)
      echo "T5GEMMA_2E_EVAL_BLOCKED trainer ${TRAIN_PROGRAM} ended as ${train_state} without a valid completed result" >&2
      exit 78
      ;;
    *)
      echo "T5GEMMA_2E_EVAL_BLOCKED unrecognized trainer state: ${status_line}" >&2
      exit 78
      ;;
  esac
  now="$(date +%s)"
  if (( now - started_at >= MAX_WAIT_SECONDS )); then
    echo "T5GEMMA_2E_EVAL_BLOCKED timed out waiting for ${TRAIN_PROGRAM}" >&2
    exit 78
  fi
  echo "T5GEMMA_2E_EVAL_WAITING trainer=${TRAIN_PROGRAM} state=${train_state}"
  sleep "${WAIT_INTERVAL}"
done

# Bind both checkpoints to the intended immutable SFT contracts.  Inference
# performs deeper adapter/tokenizer validation before loading either model.
if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-enriched-sft-run-v1"
   and .status == "complete"
   and .updates == 174
   and .planned_updates == 174
   and .rows == 2776
   and .latest_checkpoint == "checkpoint-optstep-000174"
   and .no_frontier_api == true' \
  "${ONE_EPOCH_DIR}/result.json" >/dev/null; then
  echo "T5GEMMA_2E_EVAL_BLOCKED one-epoch result contract is absent or changed" >&2
  exit 78
fi
for checkpoint_spec in \
  "${ONE_EPOCH_CHECKPOINT}:1:174" \
  "${TWO_EPOCH_CHECKPOINT}:2:348"; do
  IFS=: read -r checkpoint epochs updates <<<"${checkpoint_spec}"
  if [[ ! -d "${checkpoint}" ]] \
    || [[ ! -s "${checkpoint}/adapter/adapter_model.safetensors" ]] \
    || [[ ! -s "${checkpoint}/adapter/adapter_config.json" ]] \
    || [[ ! -s "${checkpoint}/tokenizer/tokenizer.json" ]] \
    || [[ ! -s "${checkpoint}/run_contract.json" ]] \
    || ! /usr/bin/jq -e \
      --argjson epochs "${epochs}" --argjson updates "${updates}" \
      '.schema == "t5gemma2-enriched-sft-run-v1"
       and .optimization.epochs == $epochs
       and .optimization.planned_updates == $updates
       and .optimization.batch_size == 1
       and .optimization.gradient_accumulation == 16
       and .optimization.seed == 42
       and .lora.rank == 64
       and .lora.alpha == 128
       and .lora.encoder_and_decoder_trainable == true' \
      "${checkpoint}/run_contract.json" >/dev/null; then
    echo "T5GEMMA_2E_EVAL_BLOCKED checkpoint contract failed: ${checkpoint}" >&2
    exit 78
  fi
done
if ! cmp -s "${TWO_EPOCH_DIR}/run_contract.json" \
  "${TWO_EPOCH_CHECKPOINT}/run_contract.json"; then
  echo "T5GEMMA_2E_EVAL_BLOCKED final two-epoch checkpoint/root contracts differ" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_2E_EVAL_BLOCKED Dart 3.12.2 is not executable" >&2
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

common_inference=(
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl"
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json"
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl"
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json"
  --num_samples 10
  --generation_batch_size 10
  --max_source_tokens 32768
  --max_new_tokens 4096
  --temperature 0.8
  --top_p 0.95
  --seed 42
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

run_arm() {
  local label="$1"
  local arm="$2"
  local checkpoint="$3"
  local predictions="$4"
  local score="$5"
  /venv/main/bin/python scripts/evaluation/t5gemma2_f2_passk_inference.py \
    "${common_inference[@]}" \
    --sft_checkpoint "${checkpoint}" \
    --arm "${arm}" \
    --output "${predictions}"
  /venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
    --predictions "${predictions}" \
    --output "${score}" \
    "${common_score[@]}"
  echo "T5GEMMA_2E_EVAL_ARM_COMPLETE arm=${label} score=${score}"
}

cd "${PROJECT}"
# The first two paths are the already-valid paired evaluation.  Their durable
# journals make these calls O(verification) when complete and exact-resume when
# interrupted; they are never silently regenerated under a different contract.
run_arm base base "${ONE_EPOCH_CHECKPOINT}" \
  "${REFERENCE_EVAL_DIR}/pre_base_k10_predictions.json" \
  "${REFERENCE_EVAL_DIR}/pre_base_k10_score.json"
run_arm one_epoch sft "${ONE_EPOCH_CHECKPOINT}" \
  "${REFERENCE_EVAL_DIR}/post_sft_k10_predictions.json" \
  "${REFERENCE_EVAL_DIR}/post_sft_k10_score.json"
run_arm two_epoch sft "${TWO_EPOCH_CHECKPOINT}" \
  "${OUTPUT_DIR}/two_epoch_k10_predictions.json" \
  "${OUTPUT_DIR}/two_epoch_k10_score.json"

/venv/main/bin/python - \
  "${REFERENCE_EVAL_DIR}/pre_base_k10_predictions.json" \
  "${REFERENCE_EVAL_DIR}/post_sft_k10_predictions.json" \
  "${OUTPUT_DIR}/two_epoch_k10_predictions.json" \
  "${REFERENCE_EVAL_DIR}/pre_base_k10_score.json" \
  "${REFERENCE_EVAL_DIR}/post_sft_k10_score.json" \
  "${OUTPUT_DIR}/two_epoch_k10_score.json" \
  "${OUTPUT_DIR}/comparison.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import (
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)

prediction_paths = list(map(Path, sys.argv[1:4]))
score_paths = list(map(Path, sys.argv[4:7]))
output = Path(sys.argv[7])
labels = ("base", "one_epoch", "two_epoch")
expected_arms = ("base", "sft", "sft")
provenances = []
journals = []
scores = []
predictions = []
for label, expected_arm, prediction_path, score_path in zip(
    labels, expected_arms, prediction_paths, score_paths, strict=True
):
    provenance_path = Path(str(prediction_path) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    journal_path = Path(str(prediction_path) + ".generation.journal.jsonl")
    journal = load_journal(journal_path)
    if (
        provenance.get("schema") != "direct-compact-inference-v1"
        or provenance.get("arm") != expected_arm
        or provenance.get("num_rows") != 175
        or provenance.get("num_samples") != 10
        or provenance.get("output_sha256") != sha256_file(prediction_path)
        or provenance.get("generation_journal") != journal_record(journal_path)
        or provenance.get("no_frontier_api") is not True
        or provenance.get("tests_exposed_to_model") is not False
        or provenance.get("targets_exposed_to_model") is not False
        or not journal
        or journal[0].get("event") != "header"
        or journal[-1].get("event") != "complete"
        or score.get("schema") != "direct-compact-attested-passk-v1"
        or score.get("tasks") != 175
        or score.get("k") != 10
        or score.get("timeout") != 30
        or score.get("stability_runs") != 2
        or score.get("predictions", {}).get("sha256") != sha256_file(prediction_path)
        or len(prediction) != 175
        or any(len(row.get("predictions") or []) != 10 for row in prediction)
    ):
        raise SystemExit(f"{label}: sealed evaluation contract failed")
    provenances.append(provenance)
    journals.append(journal)
    scores.append(score)
    predictions.append(prediction)

sampling = [item["sampling"] for item in provenances]
heldout = [item["heldout"] for item in provenances]
script_hashes = [item[0]["contract"]["script_sha256"] for item in journals]
task_orders = [[row["id"] for row in item] for item in predictions]
score_orders = [[row["task_id"] for row in item["task_results"]] for item in scores]
slot_coordinates = [
    [
        (
            terminal["task_id"],
            terminal["source_sha256"],
            tuple(
                (candidate["sample_index"], candidate["seed"])
                for candidate in terminal["candidates"]
            ),
        )
        for terminal in journal[1:-1]
    ]
    for journal in journals
]
score_contracts = [
    (
        item["evaluation"]["sha256"],
        item["evaluator"]["sha256"],
        item["k"],
        item["timeout"],
        item["stability_runs"],
    )
    for item in scores
]
tokenizers = [item["model"]["tokenizer_sha256"] for item in provenances]
if not (
    sampling[0] == sampling[1] == sampling[2]
    and heldout[0] == heldout[1] == heldout[2]
    and len(set(script_hashes)) == 1
    and task_orders[0] == task_orders[1] == task_orders[2]
    and score_orders[0] == score_orders[1] == score_orders[2]
    and slot_coordinates[0] == slot_coordinates[1] == slot_coordinates[2]
    and score_contracts[0] == score_contracts[1] == score_contracts[2]
    and len(set(tokenizers)) == 1
):
    raise SystemExit("base/one-epoch/two-epoch arms are not exactly paired")

def metric_block(score):
    return {
        key: score[key]
        for key in ("pass_at_1", "pass_at_k", "compile_at_k")
    }

one_by_task = {row["task_id"]: row for row in scores[1]["task_results"]}
two_by_task = {row["task_id"]: row for row in scores[2]["task_results"]}
paired = {}
for metric in ("pass_at_1", "pass_at_k", "compile_at_k"):
    wins = losses = ties = 0
    for task_id in score_orders[1]:
        left = bool(two_by_task[task_id][metric])
        right = bool(one_by_task[task_id][metric])
        wins += left and not right
        losses += right and not left
        ties += left == right
    paired[metric] = {
        "two_epoch_above_one_epoch_tasks": wins,
        "one_epoch_above_two_epoch_tasks": losses,
        "equal_tasks": ties,
    }

report = {
    "schema": "t5gemma2-sft-epoch-ablation-comparison-v1",
    "status": "complete",
    "heldout_tasks": 175,
    "k": 10,
    "exact_pairing_validated": True,
    "same_inference_code": True,
    "same_task_order_and_sources": True,
    "same_sampling_and_slot_seeds": True,
    "same_scoring_contract": True,
    "no_frontier_api": True,
    "tests_exposed_to_model": False,
    "arms": {
        label: {
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": sha256_file(prediction_path),
            "score": str(score_path.resolve()),
            "score_sha256": sha256_file(score_path),
            "metrics": metric_block(score),
        }
        for label, prediction_path, score_path, score in zip(
            labels, prediction_paths, score_paths, scores, strict=True
        )
    },
    "paired_two_epoch_vs_one_epoch": paired,
}
require_exact_or_write(output, report)
print(json.dumps(report, sort_keys=True), flush=True)
PY

echo "T5GEMMA_2E_EVAL_COMPLETE output=${OUTPUT_DIR}/comparison.json"
