#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
PRE_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
MIXED_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_mixed_rs_sft_final_v1"
MIXED_CHECKPOINT="${MIXED_DIR}/checkpoint-optstep-000426"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_mixed_rs_sft_passk_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -s "${MIXED_DIR}/result.json" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-mixed-rs-sft-run-v1"
     and .status == "complete"
     and .architecture == "native_encoder_decoder"
     and .updates == 426
     and .planned_updates == 426
     and .rows == 1132
     and .latest_checkpoint == "checkpoint-optstep-000426"
     and .production_floor_eligible == true' \
    "${MIXED_DIR}/result.json" >/dev/null; then
  echo "T5GEMMA_MIXED_EVAL_BLOCKED completed mixed result is absent or changed" >&2
  exit 78
fi
if [[ ! -d "${MIXED_CHECKPOINT}" ]] \
  || [[ ! -s "${MIXED_CHECKPOINT}/adapter/adapter_model.safetensors" ]] \
  || [[ ! -s "${MIXED_CHECKPOINT}/adapter/adapter_config.json" ]] \
  || [[ ! -s "${MIXED_CHECKPOINT}/tokenizer/tokenizer.json" ]] \
  || [[ ! -s "${MIXED_CHECKPOINT}/run_contract.json" ]] \
  || ! cmp -s "${MIXED_DIR}/run_contract.json" \
    "${MIXED_CHECKPOINT}/run_contract.json" \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-mixed-rs-sft-run-v1"
     and .status == "training"
     and .architecture == "native_encoder_decoder"
     and .optimization.planned_updates == 426
     and .optimization.epochs == 3
     and .optimization.batch_size == 1
     and .optimization.gradient_accumulation == 8
     and .optimization.seed == 42
     and .dataset.rows == 1132
     and .dataset.heldout_overlap == 0
     and .privacy.heldout_content_model_visible == false
     and .privacy.tests_model_visible == false
     and .privacy.private_feedback_model_visible == false
     and .production_floor_eligible == true
     and .lora.encoder_and_decoder_trainable == true' \
    "${MIXED_CHECKPOINT}/run_contract.json" >/dev/null; then
  echo "T5GEMMA_MIXED_EVAL_BLOCKED checkpoint contract failed" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_MIXED_EVAL_BLOCKED Dart 3.12.2 is not executable" >&2
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

PRE_PREDICTIONS="${PRE_DIR}/two_epoch_k10_predictions.json"
PRE_SCORE="${PRE_DIR}/two_epoch_k10_score.json"
POST_PREDICTIONS="${OUTPUT_DIR}/post_mixed_k10_predictions.json"
POST_SCORE="${OUTPUT_DIR}/post_mixed_k10_score.json"
POST_COMPAT="${POST_PREDICTIONS}.checkpoint-loader-compat.json"
for path in \
  "${PRE_PREDICTIONS}" \
  "${PRE_PREDICTIONS}.provenance.json" \
  "${PRE_PREDICTIONS}.generation.journal.jsonl" \
  "${PRE_SCORE}"; do
  if [[ ! -s "${path}" ]]; then
    echo "T5GEMMA_MIXED_EVAL_BLOCKED paired pre-RS-SFT artifact is absent: ${path}" >&2
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
  --compat_checkpoint "${MIXED_CHECKPOINT}" \
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

/venv/main/bin/python - \
  "${PRE_PREDICTIONS}" \
  "${POST_PREDICTIONS}" \
  "${PRE_SCORE}" \
  "${POST_SCORE}" \
  "${POST_COMPAT}" \
  "${OUTPUT_DIR}/comparison.json" <<'PY'
import json
import sys
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import (
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)

prediction_paths = list(map(Path, sys.argv[1:3]))
score_paths = list(map(Path, sys.argv[3:5]))
compat_path = Path(sys.argv[5])
output = Path(sys.argv[6])
labels = ("pre_mixed_rs_sft", "post_mixed_rs_sft")
provenances = []
journals = []
scores = []
predictions = []
for label, prediction_path, score_path in zip(
    labels, prediction_paths, score_paths, strict=True
):
    provenance_path = Path(str(prediction_path) + ".provenance.json")
    journal_path = Path(str(prediction_path) + ".generation.journal.jsonl")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    journal = load_journal(journal_path)
    if (
        provenance.get("schema") != "direct-compact-inference-v1"
        or provenance.get("arm") != "sft"
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
compat = json.loads(compat_path.read_text(encoding="utf-8"))
compat_wrapper = Path(compat.get("wrapper_path", ""))
if (
    compat.get("schema") != "t5gemma2-mixed-passk-loader-compat-v1"
    or compat.get("scope") != "checkpoint_contract_loader_only"
    or compat.get("sampling_code_changed") is not False
    or compat.get("generation_code_changed") is not False
    or compat.get("scoring_code_changed") is not False
    or compat.get("core_inference_sha256") != script_hashes[1]
    or not compat_wrapper.is_file()
    or compat.get("wrapper_sha256") != sha256_file(compat_wrapper)
    or compat.get("checkpoint_run_contract_sha256")
    != provenances[1]["model"]["adapter"]["run_contract_sha256"]
):
    raise SystemExit("mixed checkpoint-loader compatibility binding failed")
if not (
    sampling[0] == sampling[1]
    and heldout[0] == heldout[1]
    and script_hashes[0] == script_hashes[1]
    and task_orders[0] == task_orders[1]
    and score_orders[0] == score_orders[1]
    and slot_coordinates[0] == slot_coordinates[1]
    and score_contracts[0] == score_contracts[1]
    and tokenizers[0] == tokenizers[1]
):
    raise SystemExit("pre/post mixed RS-SFT arms are not exactly paired")

def metric_block(score):
    return {
        key: score[key]
        for key in ("pass_at_1", "pass_at_k", "compile_at_k")
    }

pre_by_task = {row["task_id"]: row for row in scores[0]["task_results"]}
post_by_task = {row["task_id"]: row for row in scores[1]["task_results"]}
paired = {}
for metric in ("pass_at_1", "pass_at_k", "compile_at_k"):
    wins = losses = ties = 0
    for task_id in score_orders[0]:
        before = bool(pre_by_task[task_id][metric])
        after = bool(post_by_task[task_id][metric])
        wins += after and not before
        losses += before and not after
        ties += before == after
    paired[metric] = {
        "post_above_pre_tasks": wins,
        "pre_above_post_tasks": losses,
        "equal_tasks": ties,
    }

report = {
    "schema": "t5gemma2-mixed-rs-sft-comparison-v1",
    "status": "complete",
    "heldout_tasks": 175,
    "k": 10,
    "exact_pairing_validated": True,
    "same_inference_code": True,
    "same_task_order_and_sources": True,
    "same_sampling_and_slot_seeds": True,
    "same_scoring_contract": True,
    "post_checkpoint_loader_compat": {
        "path": str(compat_path.resolve()),
        "sha256": sha256_file(compat_path),
        "wrapper_sha256": compat["wrapper_sha256"],
        "scope": compat["scope"],
    },
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
    "paired_post_vs_pre": paired,
}
require_exact_or_write(output, report)
print(json.dumps(report, sort_keys=True), flush=True)
PY

echo "T5GEMMA_MIXED_EVAL_COMPLETE output=${OUTPUT_DIR}/comparison.json"
