#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
WARMSTART="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_4b4b_compiler_verpo_pilot16_2epoch_v1"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
MAX_UPDATES=16

if [[ ! -d "${WARMSTART}" ]] || [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_VERPO_PILOT16_BLOCKED warm-start or Dart runtime is absent" >&2
  exit 78
fi
printf '%s  %s\n' \
  562c3da5f89428e6a7263ad8ec79dde9c8b6eb25c77949606277d7d80aecea4f \
  "${WARMSTART}/run_contract.json" \
  83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc \
  "${WARMSTART}/adapter/adapter_model.safetensors" \
  c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3 \
  "${WARMSTART}/adapter/adapter_config.json" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d \
  "${WARMSTART}/tokenizer/tokenizer.json" \
  11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614 \
  "${FEEDBACK_DIR}/verpo_feedback_view.public.json" \
  | sha256sum -c -

/venv/main/bin/python - "${WARMSTART}/run_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
contract = json.loads(path.read_text(encoding="utf-8"))
canonical = hashlib.sha256(
    json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
if (
    canonical
    != "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3"
    or contract.get("schema") != "t5gemma2-enriched-sft-run-v1"
    or contract.get("architecture") != "native_encoder_decoder"
    or contract.get("dataset", {}).get("rows") != 2776
    or contract.get("lora", {}).get("encoder_and_decoder_trainable") is not True
    or len(contract.get("lora", {}).get("targets") or []) != 476
    or contract.get("tokenization", {}).get("truncated_rows") != 0
    or contract.get("tokenization", {}).get("source_tokens", {}).get("max") != 13253
    or contract.get("tokenization", {}).get("target_tokens", {}).get("max") != 3343
):
    raise SystemExit("T5GEMMA_VERPO_PILOT16_BLOCKED warm-start contract differs")
PY

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"

validate_pilot_state() {
  /venv/main/bin/python - \
    "${OUTPUT_DIR}" "${PROJECT}" "${WARMSTART}" "${MAX_UPDATES}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()
project = Path(sys.argv[2]).resolve()
warmstart = Path(sys.argv[3]).resolve()
max_updates = int(sys.argv[4])
run_schema = "t5gemma2-compiler-feedback-verpo-run-v1"
checkpoint_schema = "t5gemma2-compiler-feedback-verpo-checkpoint-v1"
rollout_schema = "t5gemma2-compiler-feedback-verpo-rollout-v1"


def read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"T5GEMMA_VERPO_PILOT16_BLOCKED invalid {label}: {error}")
    if not isinstance(value, dict):
        raise SystemExit(f"T5GEMMA_VERPO_PILOT16_BLOCKED {label} is not an object")
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


def validate_contract(contract: dict) -> str:
    optimization = contract.get("optimization") or {}
    sampling = contract.get("sampling") or {}
    warm = contract.get("warmstart") or {}
    runtime = contract.get("runtime_provenance") or {}
    code = runtime.get("code") or {}
    expected_code = {
        "trainer": "scripts/training/t5gemma2_compiler_feedback_verpo.py",
        "seq2seq_core": "scripts/training/seq2seq_verpo_core.py",
        "dart_evaluator": "scripts/evaluation/graph_compile_at_k_antigravity.py",
        "feedback_boundary_builder": "scripts/preprocessing/build_verpo_feedback_view.py",
        "enriched_sft_helper": "scripts/training/t5gemma2_enriched_sft.py",
    }
    expected = (
        contract.get("schema") == run_schema
        and contract.get("architecture") == "native_t5gemma2_encoder_decoder"
        and contract.get("objective")
        == "on_policy_visible_execution_verpo_plus_local_compiler_repair"
        and contract.get("no_frontier_api") is True
        and contract.get("llm_judge") is False
        and contract.get("acceptance_tests_exposed") is False
        and contract.get("private_holdback_exposed") is False
        and warm.get("path") == str(warmstart)
        and warm.get("run_contract_sha256")
        == "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3"
        and optimization.get("max_updates") == max_updates
        and optimization.get("tasks_per_update") == 1
        and optimization.get("learning_rate") == 1e-6
        and optimization.get("max_grad_norm") == 1.0
        and optimization.get("ppo_clip") == 0.0
        and optimization.get("sft_replay_weight") == 0.02
        and optimization.get("on_policy_logprob_tolerance") == 2e-4
        and optimization.get("declared_trajectory_slots_per_task") == 12
        and sampling.get("group_size") == 4
        and sampling.get("repair_group_size") == 4
        and sampling.get("max_repair_parents") == 2
        and sampling.get("temperature") == 0.8
        and sampling.get("top_p") == 1.0
        and sampling.get("top_k") == 0
        and sampling.get("max_new_tokens") == 4096
        and sampling.get("max_source_tokens") == 32768
        and sampling.get("pad_token_id") == 0
        and sampling.get("eos_token_ids") == [1]
        and sampling.get("suppressed_token_ids") == [0]
        and sampling.get("pad_removed_from_sampling_support") is True
        and sampling.get("sampling_support_constraint_exactly_recomputed") is True
        and sampling.get("pad_before_eos_fail_closed") is True
        and sampling.get("distribution_truncated") is False
        and contract.get("seed") == 42
        and runtime.get("schema")
        == "t5gemma2-compiler-feedback-verpo-runtime-provenance-v1"
        and set(code) == set(expected_code)
    )
    if not expected:
        raise SystemExit("T5GEMMA_VERPO_PILOT16_BLOCKED run contract differs")
    for name, relative in expected_code.items():
        record = code.get(name) or {}
        path = (project / relative).resolve()
        if (
            not path.is_relative_to(project)
            or record.get("relative_path") != relative
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
        ):
            raise SystemExit(
                f"T5GEMMA_VERPO_PILOT16_BLOCKED runtime source differs: {name}"
            )
    if runtime.get("code_bundle_sha256") != canonical_sha256(code):
        raise SystemExit(
            "T5GEMMA_VERPO_PILOT16_BLOCKED runtime code bundle differs"
        )
    return canonical_sha256(contract)


def validate_metrics(contract_sha256: str, expected_updates: int) -> None:
    path = output_dir / "rollout_metrics.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SystemExit(
            f"T5GEMMA_VERPO_PILOT16_BLOCKED missing rollout metrics: {error}"
        )
    if any(not line.strip() for line in lines) or len(lines) != expected_updates:
        raise SystemExit(
            "T5GEMMA_VERPO_PILOT16_BLOCKED metrics/checkpoint update mismatch"
        )
    rows = []
    for index, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"T5GEMMA_VERPO_PILOT16_BLOCKED invalid metric {index}: {error}"
            )
        if (
            not isinstance(row, dict)
            or row.get("schema") != rollout_schema
            or row.get("update") != index
            or row.get("run_contract_sha256") != contract_sha256
            or row.get("no_frontier_api") is not True
            or row.get("declared_trajectory_slots") != 12
            or not 4 <= int(row.get("trajectory_count", 0)) <= 12
            or not 0 <= int(row.get("active_policy_trajectories", -1))
            <= int(row.get("trajectory_count", 0))
            or row.get("optimizer_step") is not True
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
                f"T5GEMMA_VERPO_PILOT16_BLOCKED invalid metric {index}"
            )
        records = row.get("task_records")
        if not isinstance(records, list) or len(records) != 1:
            raise SystemExit(
                f"T5GEMMA_VERPO_PILOT16_BLOCKED invalid task record {index}"
            )
        trajectories = records[0].get("trajectories")
        if (
            not isinstance(trajectories, list)
            or len(trajectories) != row["trajectory_count"]
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("actions"), list)
                or not 1 <= int(item.get("action_tokens", 0)) <= 4096
                or len(item["actions"]) != int(item["action_tokens"])
                or item["actions"][-1] != 1
                for item in trajectories
            )
        ):
            raise SystemExit(
                f"T5GEMMA_VERPO_PILOT16_BLOCKED invalid trajectories {index}"
            )
        rows.append(row)


root_contract_path = output_dir / "run_contract.json"
pointer_path = output_dir / "latest_checkpoint.json"
result_path = output_dir / "result.json"

if not pointer_path.is_file():
    unexpected = [
        path
        for path in output_dir.iterdir()
        if path.name != "run_contract.json"
    ]
    if unexpected:
        raise SystemExit(
            "T5GEMMA_VERPO_PILOT16_BLOCKED output has artifacts without a checkpoint"
        )
    if root_contract_path.is_file():
        validate_contract(read_object(root_contract_path, "root run contract"))
    print("fresh")
    raise SystemExit(0)

if not root_contract_path.is_file():
    raise SystemExit("T5GEMMA_VERPO_PILOT16_BLOCKED checkpoint has no root contract")
root_contract = read_object(root_contract_path, "root run contract")
contract_sha256 = validate_contract(root_contract)
pointer = read_object(pointer_path, "checkpoint pointer")
update = pointer.get("update")
if (
    pointer.get("schema") != checkpoint_schema
    or type(update) is not int
    or not 1 <= update <= max_updates
    or pointer.get("run_contract_sha256") != contract_sha256
):
    raise SystemExit("T5GEMMA_VERPO_PILOT16_BLOCKED invalid checkpoint pointer")
checkpoint = Path(str(pointer.get("path") or "")).resolve()
if (
    checkpoint.parent != output_dir
    or checkpoint.name != f"checkpoint-optstep-{update:06d}"
    or not checkpoint.is_dir()
    or not (checkpoint / "training_state.pt").is_file()
    or not (checkpoint / "adapter").is_dir()
    or not (checkpoint / "tokenizer").is_dir()
):
    raise SystemExit("T5GEMMA_VERPO_PILOT16_BLOCKED incomplete checkpoint")
checkpoint_contract = read_object(
    checkpoint / "run_contract.json", "checkpoint run contract"
)
if canonical_sha256(checkpoint_contract) != contract_sha256:
    raise SystemExit("T5GEMMA_VERPO_PILOT16_BLOCKED checkpoint contract differs")
validate_metrics(contract_sha256, update)

if result_path.is_file():
    result = read_object(result_path, "result")
    if (
        result.get("schema") != run_schema
        or result.get("status") != "complete"
        or result.get("updates") != max_updates
        or result.get("latest_checkpoint")
        != f"checkpoint-optstep-{max_updates:06d}"
        or result.get("run_contract_sha256") != contract_sha256
        or result.get("no_frontier_api") is not True
        or update != max_updates
    ):
        raise SystemExit("T5GEMMA_VERPO_PILOT16_BLOCKED invalid completed result")
    print("complete")
elif update < max_updates:
    print(f"resume:{checkpoint}")
else:
    raise SystemExit(
        "T5GEMMA_VERPO_PILOT16_BLOCKED final checkpoint has no completed result"
    )
PY
}

set +e
state="$(validate_pilot_state)"
validation_status=$?
set -e
if [[ "${validation_status}" -ne 0 ]]; then
  echo "T5GEMMA_VERPO_PILOT16_BLOCKED preflight validation failed" >&2
  exit 78
fi
if [[ "${state}" == "complete" ]]; then
  echo "T5GEMMA_VERPO_PILOT16_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

resume_args=()
if [[ "${state}" == resume:* ]]; then
  resume_checkpoint="${state#resume:}"
  resume_args=(--resume_checkpoint "${resume_checkpoint}")
  echo "T5GEMMA_VERPO_PILOT16_RESUME checkpoint=${resume_checkpoint}"
elif [[ "${state}" != "fresh" ]]; then
  echo "T5GEMMA_VERPO_PILOT16_BLOCKED invalid preflight state" >&2
  exit 78
fi

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
/venv/main/bin/python scripts/training/t5gemma2_compiler_feedback_verpo.py \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --rollout_seal "${FEEDBACK_DIR}/verpo_rollout_feedback.seal.json" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --f2_manifest "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl.manifest.json" \
  --feedback_public_manifest "${FEEDBACK_DIR}/verpo_feedback_view.public.json" \
  --expected_feedback_public_manifest_sha256 11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614 \
  --compact_contract "${WORKSPACE}/multifunction_v1/expanded2776/executable_target24k/compact_contract.json" \
  --warmstart_checkpoint "${WARMSTART}" \
  --output_dir "${OUTPUT_DIR}" \
  --group_size 4 \
  --repair_group_size 4 \
  --max_repair_parents 2 \
  --tasks_per_update 1 \
  --max_updates "${MAX_UPDATES}" \
  --temperature 0.8 \
  --max_new_tokens 4096 \
  --max_source_tokens 32768 \
  --max_target_tokens 32768 \
  --verpo_alpha 2.0 \
  --local_weight 1.0 \
  --compile_weight 0.25 \
  --learning_rate 1e-6 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --ppo_clip 0.0 \
  --sft_replay_weight 0.02 \
  --on_policy_logprob_tolerance 2e-4 \
  --reward_workers 4 \
  --reward_timeout 30 \
  --reward_stability_runs 1 \
  --checkpoint_interval 1 \
  --keep_last_checkpoints 2 \
  --seed 42 \
  --attn_implementation sdpa \
  --bf16 \
  "${resume_args[@]}"

set +e
final_state="$(validate_pilot_state)"
validation_status=$?
set -e
if [[ "${validation_status}" -ne 0 || "${final_state}" != "complete" ]]; then
  echo "T5GEMMA_VERPO_PILOT16_BLOCKED post-run validation failed" >&2
  exit 78
fi
echo "T5GEMMA_VERPO_PILOT16_COMPLETE output=${OUTPUT_DIR}"
