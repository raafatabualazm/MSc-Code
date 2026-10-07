#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
PILOT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_compiler_verpo_pilot16_2epoch_v1"
PILOT_CHECKPOINT="${PILOT_DIR}/checkpoint-optstep-000016"
BASELINE_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_compiler_verpo_pilot16_passk_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
TRAIN_PROGRAM="${T5GEMMA_VERPO_PILOT16_PROGRAM:-t5gemma-verpo-pilot16-2epoch}"
SUPERVISORCTL="${T5GEMMA_VERPO_EVAL_SUPERVISORCTL:-/usr/local/bin/supervisorctl}"
EXPECTED_RUN_CONTRACT_SHA256=5d2f91531938079dfa032741aeef3161607d378274d6c92f60d81c184f8a7c86
HISTORICAL_INFERENCE_SHA256=564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9
HISTORICAL_SCORER_SHA256=2d2d0d40eac8061290427c585be6385f147d002d82def912af88bca3a3a8fe19
HISTORICAL_DART_EVALUATOR_SHA256=249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6

set +e
status_line="$("${SUPERVISORCTL}" status "${TRAIN_PROGRAM}" 2>&1)"
status_rc=$?
set -e
if [[ -z "${status_line}" ]]; then
  echo "T5GEMMA_VERPO_EVAL_BLOCKED empty Supervisor response (rc=${status_rc})" >&2
  exit 78
fi
train_state="$(printf '%s\n' "${status_line}" | /usr/bin/awk '{print $2}')"
if [[ "${train_state}" != EXITED ]]; then
  echo "T5GEMMA_VERPO_EVAL_BLOCKED pilot is not terminal: ${status_line}" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_VERPO_EVAL_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi

# The baseline and evaluation implementations are immutable inputs.  In
# particular, this prevents a loader refactor from silently changing generation
# while claiming an exact paired comparison.
printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a \
  "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab \
  "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 \
  "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  16f27a9d96df73e4e5c3e4f43ced4cd3b46574bf3dc9cceb5beadb382c76e14d \
  "${BASELINE_DIR}/two_epoch_k10_predictions.json" \
  9e204f0d5caad8bf1e99af04f04b51773bed0e2bf89deeba201bae0fb344a6e0 \
  "${BASELINE_DIR}/two_epoch_k10_predictions.json.provenance.json" \
  260c52887faedadf4f46ffc616ae3489910289de03f64667381b7b3ef42c2552 \
  "${BASELINE_DIR}/two_epoch_k10_predictions.json.generation.journal.jsonl" \
  cf235747ba24cb37dd08764406d1a12278e0f2fbd9dc56753d0e5a5032cab2c3 \
  "${BASELINE_DIR}/two_epoch_k10_predictions.json.generation.journal.jsonl.chain-head.json" \
  e98d2f7dea3d12a17a4287d77ba324b48e50bff0ba3ca62c765bd85349b43334 \
  "${BASELINE_DIR}/two_epoch_k10_score.json" \
  954e3dd79290ba93187820093de3774fc64450b7f7b4d56f92ee98ccda4cd012 \
  "${BASELINE_DIR}/comparison.json" \
  "${HISTORICAL_INFERENCE_SHA256}" \
  "${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py" \
  "${HISTORICAL_SCORER_SHA256}" \
  "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  "${HISTORICAL_DART_EVALUATOR_SHA256}" \
  "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  | sha256sum -c -

/venv/main/bin/python - \
  "${PROJECT}" "${PILOT_DIR}" "${PILOT_CHECKPOINT}" \
  "${EXPECTED_RUN_CONTRACT_SHA256}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

project, pilot_dir, checkpoint = map(Path, sys.argv[1:4])
expected_contract_sha = sys.argv[4]


def read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"T5GEMMA_VERPO_EVAL_BLOCKED invalid {label}: {error}")
    if not isinstance(value, dict):
        raise SystemExit(f"T5GEMMA_VERPO_EVAL_BLOCKED {label} is not an object")
    return value


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root_contract = read_object(pilot_dir / "run_contract.json", "root contract")
checkpoint_contract = read_object(
    checkpoint / "run_contract.json", "checkpoint contract"
)
pointer = read_object(pilot_dir / "latest_checkpoint.json", "checkpoint pointer")
result = read_object(pilot_dir / "result.json", "pilot result")
contract_sha = canonical_sha256(root_contract)
runtime = root_contract.get("runtime_provenance") or {}
code = runtime.get("code") or {}
warm = root_contract.get("warmstart") or {}
if (
    contract_sha != expected_contract_sha
    or checkpoint_contract != root_contract
    or root_contract.get("schema")
    != "t5gemma2-compiler-feedback-verpo-run-v1"
    or root_contract.get("architecture")
    != "native_t5gemma2_encoder_decoder"
    or root_contract.get("no_frontier_api") is not True
    or root_contract.get("acceptance_tests_exposed") is not False
    or root_contract.get("private_holdback_exposed") is not False
    or (root_contract.get("optimization") or {}).get("max_updates") != 16
    or (root_contract.get("optimization") or {}).get("tasks_per_update") != 1
    or (root_contract.get("sampling") or {}).get("max_new_tokens") != 4096
    or (root_contract.get("sampling") or {}).get("max_source_tokens") != 32768
    or runtime.get("schema")
    != "t5gemma2-compiler-feedback-verpo-runtime-provenance-v1"
    or runtime.get("code_bundle_sha256") != canonical_sha256(code)
    or warm.get("run_contract_sha256")
    != "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3"
    or pointer.get("schema")
    != "t5gemma2-compiler-feedback-verpo-checkpoint-v1"
    or pointer.get("update") != 16
    or pointer.get("run_contract_sha256") != contract_sha
    or Path(str(pointer.get("path") or "")).resolve() != checkpoint.resolve()
    or result.get("schema") != "t5gemma2-compiler-feedback-verpo-run-v1"
    or result.get("status") != "complete"
    or result.get("updates") != 16
    or result.get("latest_checkpoint") != "checkpoint-optstep-000016"
    or result.get("run_contract_sha256") != contract_sha
    or result.get("no_frontier_api") is not True
):
    raise SystemExit("T5GEMMA_VERPO_EVAL_BLOCKED pilot completion contract differs")

for relative in (
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "tokenizer/tokenizer.json",
    "training_state.pt",
):
    path = checkpoint / relative
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(
            f"T5GEMMA_VERPO_EVAL_BLOCKED checkpoint member absent: {relative}"
        )

expected_runtime_paths = {
    "trainer": "scripts/training/t5gemma2_compiler_feedback_verpo.py",
    "seq2seq_core": "scripts/training/seq2seq_verpo_core.py",
    "dart_evaluator": "scripts/evaluation/graph_compile_at_k_antigravity.py",
    "feedback_boundary_builder": "scripts/preprocessing/build_verpo_feedback_view.py",
    "enriched_sft_helper": "scripts/training/t5gemma2_enriched_sft.py",
}
if set(code) != set(expected_runtime_paths):
    raise SystemExit("T5GEMMA_VERPO_EVAL_BLOCKED runtime code set differs")
for name, relative in expected_runtime_paths.items():
    record = code.get(name) or {}
    path = (project / relative).resolve()
    if (
        not path.is_relative_to(project.resolve())
        or record.get("relative_path") != relative
        or not path.is_file()
        or record.get("sha256") != file_sha256(path)
    ):
        raise SystemExit(
            f"T5GEMMA_VERPO_EVAL_BLOCKED training runtime changed: {name}"
        )

lines = (pilot_dir / "rollout_metrics.jsonl").read_text(
    encoding="utf-8"
).splitlines()
if len(lines) != 16 or any(not line for line in lines):
    raise SystemExit("T5GEMMA_VERPO_EVAL_BLOCKED rollout metric count differs")
for update, line in enumerate(lines, 1):
    row = json.loads(line)
    if (
        row.get("schema") != "t5gemma2-compiler-feedback-verpo-rollout-v1"
        or row.get("update") != update
        or row.get("run_contract_sha256") != contract_sha
        or row.get("optimizer_step") is not True
        or row.get("no_frontier_api") is not True
        or any(
            not math.isfinite(float(row.get(key, float("nan"))))
            for key in (
                "policy_loss",
                "sft_replay_loss",
                "max_on_policy_logprob_drift",
                "grad_norm",
            )
        )
        or float(row.get("max_on_policy_logprob_drift", float("inf"))) > 2e-4
    ):
        raise SystemExit(
            f"T5GEMMA_VERPO_EVAL_BLOCKED rollout metric {update} differs"
        )
PY

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "${DART_BIN}"):${PATH}"

BASELINE_PREDICTIONS="${BASELINE_DIR}/two_epoch_k10_predictions.json"
BASELINE_SCORE="${BASELINE_DIR}/two_epoch_k10_score.json"
POST_PREDICTIONS="${OUTPUT_DIR}/post_verpo16_k10_predictions.json"
POST_SCORE="${OUTPUT_DIR}/post_verpo16_k10_score.json"
POST_COMPAT="${POST_PREDICTIONS}.checkpoint-loader-compat.json"

cd "${PROJECT}"
/venv/main/bin/python \
  scripts/evaluation/t5gemma2_f2_passk_verpo_compat.py \
  --compat_record "${POST_COMPAT}" \
  --compat_checkpoint "${PILOT_CHECKPOINT}" \
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
  "${BASELINE_PREDICTIONS}" "${POST_PREDICTIONS}" \
  "${BASELINE_SCORE}" "${POST_SCORE}" "${POST_COMPAT}" \
  "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  "${OUTPUT_DIR}/comparison.json" "${EXPECTED_RUN_CONTRACT_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.evaluation.durable_evaluation_journal import (
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)

pre_prediction, post_prediction, pre_score, post_score, compat, scorer, output = map(
    Path, sys.argv[1:8]
)
expected_contract_sha = sys.argv[8]
labels = ("pre_verpo", "post_verpo16")
prediction_paths = (pre_prediction, post_prediction)
score_paths = (pre_score, post_score)
provenances = []
journals = []
predictions = []
scores = []
for label, prediction_path, score_path in zip(
    labels, prediction_paths, score_paths, strict=True
):
    provenance = json.loads(
        Path(str(prediction_path) + ".provenance.json").read_text(encoding="utf-8")
    )
    journal_path = Path(str(prediction_path) + ".generation.journal.jsonl")
    journal = load_journal(journal_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
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
        or score.get("predictions", {}).get("sha256")
        != sha256_file(prediction_path)
        or len(prediction) != 175
        or any(len(row.get("predictions") or []) != 10 for row in prediction)
    ):
        raise SystemExit(f"{label}: sealed evaluation contract failed")
    provenances.append(provenance)
    journals.append(journal)
    predictions.append(prediction)
    scores.append(score)

compatibility = json.loads(compat.read_text(encoding="utf-8"))
if (
    compatibility.get("schema") != "t5gemma2-verpo-passk-loader-compat-v1"
    or compatibility.get("checkpoint_run_contract_sha256")
    != expected_contract_sha
    or compatibility.get("core_inference_sha256")
    != "564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9"
    or compatibility.get("scope") != "checkpoint_contract_loader_only"
    or compatibility.get("sampling_code_changed") is not False
    or compatibility.get("generation_code_changed") is not False
    or compatibility.get("scoring_code_changed") is not False
    or provenances[1].get("sft_checkpoint_contract_sha256")
    != expected_contract_sha
    or provenances[1].get("model", {}).get("training_stage_schema")
    != "t5gemma2-compiler-feedback-verpo-run-v1"
):
    raise SystemExit("VeRPO checkpoint-loader compatibility record differs")

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
    sampling[0] == sampling[1]
    and heldout[0] == heldout[1]
    and script_hashes
    == [
        "564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9"
    ]
    * 2
    and task_orders[0] == task_orders[1]
    and score_orders[0] == score_orders[1]
    and slot_coordinates[0] == slot_coordinates[1]
    and score_contracts[0] == score_contracts[1]
    and tokenizers[0] == tokenizers[1]
):
    raise SystemExit("SFT/VeRPO arms are not exactly paired")


def metric_block(score: dict) -> dict:
    return {
        key: score[key]
        for key in ("pass_at_1", "pass_at_k", "compile_at_k")
    }


pre_by_task = {row["task_id"]: row for row in scores[0]["task_results"]}
post_by_task = {row["task_id"]: row for row in scores[1]["task_results"]}
paired = {}
for metric in ("pass_at_1", "pass_at_k", "compile_at_k"):
    gains = losses = ties = 0
    for task_id in score_orders[0]:
        before = bool(pre_by_task[task_id][metric])
        after = bool(post_by_task[task_id][metric])
        gains += after and not before
        losses += before and not after
        ties += after == before
    paired[metric] = {
        "post_above_pre_tasks": gains,
        "pre_above_post_tasks": losses,
        "equal_tasks": ties,
    }

report = {
    "schema": "t5gemma2-verpo-pilot16-comparison-v1",
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
    "checkpoint_loader_compatibility": {
        "path": str(compat.resolve()),
        "sha256": sha256_file(compat),
    },
    "scorer": {
        "path": str(scorer.resolve()),
        "sha256": sha256_file(scorer),
    },
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

echo "T5GEMMA_VERPO_EVAL_COMPLETE output=${OUTPUT_DIR}/comparison.json"
