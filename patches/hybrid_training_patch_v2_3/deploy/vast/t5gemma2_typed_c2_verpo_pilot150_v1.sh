#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="${T5GEMMA_C2_VERPO_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_C2_VERPO_PYTHON:-/venv/main/bin/python}"
SECRET_FILE="${T5GEMMA_HF_ENV:-${WORKSPACE}/secrets/HF.env}"
FEEDBACK_ROOT="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
COMPACT_ROOT="${WORKSPACE}/multifunction_v1/expanded2776/executable_target24k"
PROXY_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_proxy_reward_audit_v1"
HOLDBACK_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_holdback_alignment_v1"
WARMSTART_ROOT="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_fold_gold_replay_v2"
WARMSTART="${WARMSTART_ROOT}/checkpoint-optstep-000058"
TASK_VIEW_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_c2_verpo_task_view_v1"
TASK_VIEW="${TASK_VIEW_ROOT}/task_view.public.jsonl"
TASK_VIEW_MANIFEST="${TASK_VIEW_ROOT}/task_view.manifest.json"
OUTPUT_DIR="${T5GEMMA_C2_VERPO_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_c2_verpo_pilot150_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
PREREG="${WORKSPACE}/analysis_rs_sft_fold/T5GEMMA2_TYPED_C2_VERPO_150_PREREGISTRATION.md"
PREREG_SEAL="${WORKSPACE}/analysis_rs_sft_fold/T5GEMMA2_TYPED_C2_VERPO_150_PREREGISTRATION.seal.json"
TRAINER="${PROJECT}/scripts/training/t5gemma2_typed_c2_verpo_pilot150.py"
VIEW_BUILDER="${PROJECT}/scripts/preprocessing/build_t5gemma2_typed_c2_verpo_task_view.py"
MIN_FRESH_KIB=12582912
MIN_RESUME_KIB=8388608

blocked() { echo "T5GEMMA_TYPED_C2_VERPO150_BLOCKED $*" >&2; exit 78; }
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" && -s "${SECRET_FILE}" ]] \
  || blocked "Python, Dart, or the HF secret is absent"

# Hashes below are deliberately literal.  A supervisor restart may resume only
# the exact code, preregistration, public aggregate GO gate, and Arm-C2 parent.
printf '%s  %s\n' \
  0a88076cda0f6c981e7d07e402b5917966551f7b17f1efb05cc9eb833368fe31 "${TRAINER}" \
  c82af916b229991d56961ce356e0430f752ab03988d5804491a4087dbfe9a89c "${VIEW_BUILDER}" \
  232880791b108df96b4f01bc44a613c595cf4edaa738f6cb9a624412da5e50e4 "${PROJECT}/scripts/training/t5gemma2_compiler_feedback_verpo.py" \
  c4c72410333669f78d109d8848c70a79321ef42dba6e1a8344b138e8bfdbdb51 "${PROJECT}/scripts/training/seq2seq_verpo_core.py" \
  5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  6decd1ed1ecd3ce8e8a0bd6d861c30a26063c9d913957e361584413705f28a3b "${PROJECT}/scripts/preprocessing/build_verpo_feedback_view.py" \
  dd4026b2e86a8c3280af5e4379f1cd8a07615e69f9d1959fd1e5ee7dc4f245e2 "${PROJECT}/scripts/training/t5gemma2_enriched_sft.py" \
  bee03f83b4b86baaf60110e8b7d387e80550c43f07d675bc71710a17fba9fc66 "${PROJECT}/scripts/training/t5gemma2_typed_contract_sft.py" \
  6436838ffaed0d9c6350c0df58ff9950e5ecb08fc7899af431ee11c0cd5204bb "${PROJECT}/scripts/training/t5gemma2_typed_fold_gold_replay_v1.py" \
  38a003ae2d5b1fc19bf5c065d5c2577962dde0c5a4e14bc3ca8e3992efce6438 "${PROJECT}/scripts/evaluation/t5gemma2_typed_fold_gold_replay_inference_v1.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  bc515b7dd7efb4d2458da3af407028eca572cf2a7a1af6616e0a8f8797c134a9 "${PREREG}" \
  5bfbe6359d0b84ecabe43542eaebadd77bde0f9e959a210682ebcfa453445c80 "${PREREG_SEAL}" \
  2bd9aa7a0c4ce5740e670a7ab7a702a6522790f4dc60ec39355a4a7647f13117 "${HOLDBACK_ROOT}/holdback_alignment.summary.json" \
  b0d73acb0391adea3844afa6f36589f4035e5fa4e73751f25836b318f43d9435 "${PROXY_ROOT}/reward_audit.summary.json" \
  b63250b1db1ca53fdf033cd3935824b4a96a76c37ef4f1f390dabd72370be1f4 "${PROXY_ROOT}/reward_audit.journal.jsonl" \
  76b4bcc98ef7f16fd57a76d0501a7c91617c9ac80d1506b2beeb8763a2ab8172 "${PROXY_ROOT}/reward_audit.journal.jsonl.chain-head.json" \
  31b7f4fee0ea991a9b5ad6e9a9e14157b1f91ce0dbddf9e401fdbc57de8ebe7e "${WARMSTART_ROOT}/result.json" \
  06f49fc798537aa73bcd56e36d88d58a6255f7b09fd4da8b5bcd6319cf6bd301 "${WARMSTART}/run_contract.json" \
  80b50fab88e076d3e14771d09b5d1706baffeb2fd6c0c9d51b8841dd4135a004 "${WARMSTART}/adapter/adapter_model.safetensors" \
  6991ae89d05153ee47192bda89fbf97a7fcc3f00f3db6fcc5455cfb79660b708 "${WARMSTART}/adapter/adapter_config.json" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d "${WARMSTART}/tokenizer/tokenizer.json" \
  | sha256sum -c - || blocked "sealed code/evidence/parent differs"

"${PYTHON_BIN}" - "${HOLDBACK_ROOT}/holdback_alignment.summary.json" \
  "${WARMSTART}/run_contract.json" <<'PY' || exit 78
import hashlib
import json
import sys
from pathlib import Path

def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

holdback = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metrics = ((holdback.get("metrics") or {}).get("preregistered_61_p_new") or {})
uplift = metrics.get("tie_averaged_visible_local_argmax_uplift") or {}
rank = metrics.get("pairwise_rank_accuracy") or {}
if not (
    holdback.get("schema") == "t5gemma2-typed-holdback-alignment-summary-v1"
    and holdback.get("status") == "complete"
    and holdback.get("decision") == "GO"
    and metrics.get("overall_decision") == "GO"
    and uplift.get("decision") == "GO"
    and float((uplift.get("interval") or {}).get("lower", -1.0)) > 0.0
    and rank.get("decision") == "GO"
    and float((rank.get("interval") or {}).get("lower", -1.0)) > 0.5
    and (holdback.get("privacy") or {}).get("aggregate_only") is True
    and (holdback.get("one_shot_policy") or {}).get(
        "future_reward_weight_tuning_on_this_holdback_forbidden"
    ) is True
):
    raise SystemExit("T5GEMMA_TYPED_C2_VERPO150_BLOCKED holdback GO differs")
warm = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if canonical(warm) != "a3d325af70ac9cd0a0c55cb7e66f4df2b390f78fab3ca6a70a930093ac989a00":
    raise SystemExit("T5GEMMA_TYPED_C2_VERPO150_BLOCKED parent canonical contract differs")
PY

mkdir -p "${TASK_VIEW_ROOT}" "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
export CUDA_VISIBLE_DEVICES=-1
cd "${PROJECT}"
"${PYTHON_BIN}" "${VIEW_BUILDER}" \
  --rollout_file "${FEEDBACK_ROOT}/verpo_rollout_feedback.jsonl" \
  --rollout_seal "${FEEDBACK_ROOT}/verpo_rollout_feedback.seal.json" \
  --f2_jsonl "${FEEDBACK_ROOT}/verpo_teacher_f2.jsonl" \
  --f2_manifest "${FEEDBACK_ROOT}/verpo_teacher_f2.jsonl.manifest.json" \
  --feedback_public_manifest "${FEEDBACK_ROOT}/verpo_feedback_view.public.json" \
  --compact_contract "${COMPACT_ROOT}/compact_contract.json" \
  --proxy_audit_summary "${PROXY_ROOT}/reward_audit.summary.json" \
  --proxy_audit_journal "${PROXY_ROOT}/reward_audit.journal.jsonl" \
  --proxy_audit_chain_head "${PROXY_ROOT}/reward_audit.journal.jsonl.chain-head.json" \
  --output_view "${TASK_VIEW}" \
  --output_manifest "${TASK_VIEW_MANIFEST}"

TASK_VIEW_MANIFEST_SHA256="$(sha256sum "${TASK_VIEW_MANIFEST}" | awk '{print $1}')"
[[ "${TASK_VIEW_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || blocked "typed task-view manifest hash is malformed"

# Print exactly one restart mode.  This independently verifies the immutable
# contract, checkpoint/RNG/optimizer state, metrics, and all crossed gates.
validate_state() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${PROJECT}" "${WARMSTART}" \
    "${TASK_VIEW_MANIFEST}" "${TASK_VIEW_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from scripts.training import t5gemma2_typed_c2_verpo_pilot150 as pilot

out = Path(sys.argv[1]).resolve()
project = Path(sys.argv[2]).resolve()
workspace = project.parent.resolve()
warm = Path(sys.argv[3]).resolve()
view_manifest = Path(sys.argv[4]).resolve()
view_manifest_sha = sys.argv[5]

def die(message):
    raise SystemExit(f"T5GEMMA_TYPED_C2_VERPO150_BLOCKED {message}")

def read(path, label):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        die(f"{label} is not an object")
    return value

def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def file_sha(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        die(f"cannot hash {path}: {exc}")
    return digest.hexdigest()

allowed_names = {
    "run_contract.json", "rollout_metrics.jsonl", "latest_checkpoint.json",
    "phase_status.json", "result.json",
}
entries = list(out.iterdir())
foreign = [p.name for p in entries if p.name not in allowed_names
           and not p.name.startswith("checkpoint-optstep-")
           and not p.name.startswith("pilot_gate_update")]
if foreign:
    die(f"foreign output entries: {foreign[:3]}")
pointer_path = out / "latest_checkpoint.json"
if not pointer_path.is_file():
    if entries:
        die("nonempty output has no checkpoint pointer")
    print("fresh")
    raise SystemExit(0)

contract = read(out / "run_contract.json", "root run contract")
contract_sha = canonical(contract)
opt = contract.get("optimization") or {}
sampling = contract.get("sampling") or {}
pilot_contract = contract.get("pilot") or {}
selection = contract.get("selection") or {}
source_preflight = contract.get("source_token_preflight") or {}
if not (
    contract.get("schema") == "t5gemma2-typed-c2-verpo-pilot150-run-v1"
    and contract.get("status") == "training"
    and contract.get("architecture") == "native_encoder_decoder"
    and contract.get("policy_architecture") == "native_t5gemma2_encoder_decoder"
    and contract.get("automatic_promotion_permitted") is False
    and contract.get("production_floor_eligible") is False
    and contract.get("private_holdback_exposed") is False
    and (contract.get("warmstart") or {}).get("path") == str(warm)
    and opt.get("max_updates") == 150
    and opt.get("tasks_per_update") == 1
    and opt.get("learning_rate") == 1e-6
    and opt.get("weight_decay") == 0.0
    and opt.get("max_grad_norm") == 1.0
    and opt.get("ppo_clip") == 0.0
    and opt.get("sft_replay_weight") == 0.0
    and opt.get("gold_or_sft_replay_gradient") is False
    and sampling.get("group_size") == 4
    and sampling.get("repair_group_size") == 4
    and sampling.get("max_repair_parents") == 2
    and sampling.get("temperature") == 0.8
    and sampling.get("top_p") == 1.0
    and sampling.get("top_k") == 0
    and sampling.get("max_new_tokens") == 8192
    and sampling.get("max_source_tokens") == 32768
    and sampling.get("eos_token_ids") == [1]
    and sampling.get("suppressed_token_ids") == [0]
    and sampling.get("pad_before_eos_fail_closed") is True
    and pilot_contract.get("maximum_updates") == 150
    and pilot_contract.get("gate_interval") == 16
    and pilot_contract.get("mandatory_pause_after_first_gate") is True
    and pilot_contract.get("later_gates_automatic") is True
    and selection.get("tasks") == 150
    and selection.get("task_view_manifest_sha256") == view_manifest_sha
    and selection.get("trainer_opened_gold_or_target_source") is False
    and source_preflight.get("rows") == 150
    and source_preflight.get("truncated_rows") == 0
    and int(source_preflight.get("maximum", 32769)) <= 32768
    and contract.get("seed") == 42
):
    die("root run contract differs")

runtime = contract.get("runtime_provenance") or {}
code = runtime.get("code") or {}
for name, record in code.items():
    if not isinstance(record, dict) or set(record) != {"relative_path", "sha256"}:
        die(f"runtime record differs: {name}")
    path = (project / str(record["relative_path"])).resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        die(f"runtime path escapes workspace: {name}")
    if file_sha(path) != record["sha256"]:
        die(f"runtime source differs: {name}")
if runtime.get("code_bundle_sha256") != canonical(code):
    die("runtime code bundle differs")

pointer = read(pointer_path, "checkpoint pointer")
update = pointer.get("update")
if (pointer.get("schema") != "t5gemma2-typed-c2-verpo-pilot150-checkpoint-v1"
        or type(update) is not int or not 1 <= update <= 150
        or pointer.get("run_contract_sha256") != contract_sha):
    die("checkpoint pointer differs")
checkpoint = Path(str(pointer.get("path") or "")).resolve()
if (checkpoint.parent != out
        or checkpoint.name != f"checkpoint-optstep-{update:06d}"
        or not (checkpoint / "training_state.pt").is_file()
        or not (checkpoint / "adapter" / "adapter_model.safetensors").is_file()
        or not (checkpoint / "tokenizer" / "tokenizer.json").is_file()
        or canonical(read(checkpoint / "run_contract.json", "checkpoint contract"))
           != contract_sha):
    die("checkpoint is incomplete or escapes output root")
state = torch.load(checkpoint / "training_state.pt", map_location="cpu",
                   weights_only=False)
if not (isinstance(state, dict)
        and state.get("schema") == pointer["schema"]
        and state.get("update") == update
        and state.get("run_contract_sha256") == contract_sha
        and isinstance(state.get("optimizer"), dict)
        and isinstance(state.get("rng"), dict)
        and {"python", "torch_cpu"}.issubset(state["rng"])):
    die("checkpoint optimizer/RNG state differs")

metrics_path = out / "rollout_metrics.jsonl"
try:
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
except OSError as exc:
    die(f"rollout metrics missing: {exc}")
if len(lines) != update or any(not line.strip() for line in lines):
    die("metrics/checkpoint update mismatch")
rows = []
for index, line in enumerate(lines, 1):
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        die(f"invalid metric {index}: {exc}")
    trajectories = ((row.get("task_records") or [{}])[0]).get("trajectories")
    if not (
        isinstance(row, dict)
        and row.get("schema") == "t5gemma2-typed-c2-verpo-pilot150-rollout-v1"
        and row.get("update") == index
        and row.get("run_contract_sha256") == contract_sha
        and row.get("no_frontier_api") is True
        and row.get("declared_trajectory_slots") == 12
        and 4 <= int(row.get("trajectory_count", 0)) <= 12
        and 0 <= int(row.get("active_policy_trajectories", -1))
            <= int(row.get("trajectory_count", 0))
        and float(row.get("sft_replay_loss", float("nan"))) == 0.0
        and math.isfinite(float(row.get("policy_loss", float("nan"))))
        and math.isfinite(float(row.get("grad_norm", float("nan"))))
        and math.isfinite(float(row.get("max_on_policy_logprob_drift", float("nan"))))
        and float(row["max_on_policy_logprob_drift"]) <= 2e-4
        and int(row.get("sampled_pad_before_eos", -1)) == 0
        and isinstance(trajectories, list)
        and len(trajectories) == int(row["trajectory_count"])
    ):
        die(f"metric {index} violates integrity")
    active = int(row["active_policy_trajectories"])
    if ((active == 0) != (row.get("optimizer_step") is False)
            or (active == 0 and (float(row["policy_loss"]) != 0.0
                                 or float(row["grad_norm"]) != 0.0))
            or (active > 0 and (row.get("optimizer_step") is not True
                                or float(row["grad_norm"]) <= 0.0))):
        die(f"metric {index} optimizer semantics differ")
    for trajectory in trajectories:
        actions = trajectory.get("actions") if isinstance(trajectory, dict) else None
        if not (isinstance(actions, list) and 1 <= len(actions) <= 8192
                and len(actions) == int(trajectory.get("action_tokens", -1))
                and actions[-1] == 1
                and int(trajectory.get("sampled_pad_before_eos", -1)) == 0):
            die(f"metric {index} EOS/action invariants differ")
    rows.append(row)

gate_by_end = {}
for boundary in range(16, min(update, 144) + 1, 16):
    gate_path = out / f"pilot_gate_update{boundary:06d}.json"
    if not gate_path.is_file():
        if boundary < update:
            die(f"crossed gate {boundary} is absent")
        continue
    gate = read(gate_path, f"gate {boundary}")
    expected = pilot.evaluate_mechanics_gate(
        rows[boundary - 16:boundary], run_contract_sha256=contract_sha,
        window_start=boundary - 15, window_end=boundary,
    )
    if gate != expected:
        die(f"gate {boundary} differs from recomputation")
    gate_by_end[boundary] = gate
    if boundary < update and gate.get("decision") != "GO":
        die(f"training crossed STOP gate {boundary}")

result_path = out / "result.json"
if result_path.is_file():
    result = read(result_path, "result")
    if result.get("run_contract_sha256") != contract_sha:
        die("result contract differs")
    if result.get("status") == "complete":
        if not (update == 150 and result.get("updates") == 150
                and result.get("latest_checkpoint") == "checkpoint-optstep-000150"
                and result.get("automatic_promotion_performed") is False
                and result.get("production_floor_eligible") is False
                and all(gate_by_end.get(boundary, {}).get("decision") == "GO"
                        for boundary in range(16, 145, 16))):
            die("completed result differs")
        print("complete")
        raise SystemExit(0)
    if result.get("status") == "stopped_at_window_gate":
        if not (update in gate_by_end
                and gate_by_end[update].get("decision") == "STOP"
                and result.get("updates") == update
                and result.get("gate_update") == update
                and result.get("automatic_promotion_performed") is False):
            die("STOP result differs")
        print("stopped")
        raise SystemExit(0)
    die("unknown result state")

phase_path = out / "phase_status.json"
if update == 16 and phase_path.is_file():
    phase = read(phase_path, "phase status")
    if not (phase.get("schema") == "t5gemma2-typed-c2-verpo-phase-status-v1"
            and phase.get("status") == "awaiting_explicit_resume_after_gate16"
            and phase.get("completed_update") == 16
            and phase.get("latest_checkpoint") == "checkpoint-optstep-000016"
            and phase.get("gate_decision") == "GO"
            and phase.get("run_contract_sha256") == contract_sha
            and gate_by_end.get(16, {}).get("decision") == "GO"):
        die("phase-16 status differs")
    print(f"phase16-go:{checkpoint}")
elif update <= 16 and 16 not in gate_by_end:
    print(f"resume-pre16:{checkpoint}")
elif gate_by_end.get(16, {}).get("decision") == "GO":
    print(f"resume-continue:{checkpoint}")
elif gate_by_end.get(update, {}).get("decision") == "STOP":
    print(f"resume-continue:{checkpoint}")
else:
    die("unrecognized resumable state")
PY
}

secret_line="$("${PYTHON_BIN}" - "${SECRET_FILE}" <<'PY'
import sys
from pathlib import Path
raw = Path(sys.argv[1]).read_bytes()
try: text = raw.decode("utf-8-sig")
except UnicodeDecodeError: text = raw.decode("utf-16")
lines = [line.strip() for line in text.splitlines()
         if line.strip() and not line.lstrip().startswith("#")]
if len(lines) != 1: raise SystemExit("HF secret must have one non-comment line")
value = lines[0]
if "=" in value and value.split("=", 1)[0].replace("export ", "").strip() == "HF_TOKEN":
    value = value.split("=", 1)[1].strip()
if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'": value = value[1:-1]
if not value or any(ch.isspace() for ch in value): raise SystemExit("HF token malformed")
print(value, end="")
PY
)" || blocked "HF secret is malformed"
export HF_TOKEN="${secret_line}"
unset secret_line

export HF_HOME="${WORKSPACE}/.hf_home"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

run_trainer() {
  local continuation="$1"
  local checkpoint="${2:-}"
  local task_manifest_sha
  task_manifest_sha="$(sha256sum "${TASK_VIEW_MANIFEST}" | awk '{print $1}')"
  local args=(
    --typed_task_view "${TASK_VIEW}"
    --typed_task_view_manifest "${TASK_VIEW_MANIFEST}"
    --rollout_file "${TASK_VIEW}"
    --rollout_seal "${TASK_VIEW_MANIFEST}"
    --f2_jsonl "${TASK_VIEW}"
    --f2_manifest "${TASK_VIEW_MANIFEST}"
    --feedback_public_manifest "${TASK_VIEW_MANIFEST}"
    --expected_feedback_public_manifest_sha256 "${task_manifest_sha}"
    --compact_contract "${TASK_VIEW_MANIFEST}"
    --warmstart_checkpoint "${WARMSTART}"
    --output_dir "${OUTPUT_DIR}"
    --group_size 4
    --repair_group_size 4
    --max_repair_parents 2
    --tasks_per_update 1
    --max_updates 150
    --temperature 0.8
    --max_new_tokens 8192
    --max_source_tokens 32768
    --max_target_tokens 32768
    --verpo_alpha 2.0
    --local_weight 1.0
    --compile_weight 0.25
    --learning_rate 1e-6
    --weight_decay 0.0
    --max_grad_norm 1.0
    --ppo_clip 0.0
    --sft_replay_weight 0.0
    --on_policy_logprob_tolerance 2e-4
    --reward_workers 4
    --reward_timeout 30
    --reward_stability_runs 1
    --checkpoint_interval 1
    --keep_last_checkpoints 2
    --seed 42
    --attn_implementation sdpa
    --bf16
  )
  [[ -z "${checkpoint}" ]] || args+=(--resume_checkpoint "${checkpoint}")
  [[ "${continuation}" != "yes" ]] || args+=(--continue_after_gate16)
  "${PYTHON_BIN}" "${TRAINER}" "${args[@]}"
}

state="$(validate_state)" || blocked "state validation failed"
case "${state}" in
  complete)
    echo "T5GEMMA_TYPED_C2_VERPO150_ALREADY_COMPLETE output=${OUTPUT_DIR}"
    exit 0 ;;
  stopped)
    echo "T5GEMMA_TYPED_C2_VERPO150_ALREADY_STOPPED output=${OUTPUT_DIR}"
    exit 0 ;;
  fresh)
    minimum_kib="${MIN_FRESH_KIB}"
    first_checkpoint="" ;;
  resume-pre16:*)
    minimum_kib="${MIN_RESUME_KIB}"
    first_checkpoint="${state#resume-pre16:}" ;;
  phase16-go:*|resume-continue:*)
    minimum_kib="${MIN_RESUME_KIB}"
    first_checkpoint="" ;;
  *) blocked "unknown pre-run state: ${state}" ;;
esac

available_kib="$(df -Pk "${OUTPUT_DIR}" | awk 'NR==2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge "${minimum_kib}" ]] \
  || blocked "insufficient storage available_kib=${available_kib:-unknown} required_kib=${minimum_kib}; no launcher cleanup is permitted"

if [[ "${state}" == "fresh" || "${state}" == resume-pre16:* ]]; then
  run_trainer no "${first_checkpoint}"
  state="$(validate_state)" || blocked "phase-16 post-run validation failed"
  if [[ "${state}" == "stopped" ]]; then
    echo "T5GEMMA_TYPED_C2_VERPO150_STOP gate=16 output=${OUTPUT_DIR}"
    exit 0
  fi
  [[ "${state}" == phase16-go:* ]] \
    || blocked "trainer did not stop at the mandatory update-16 GO boundary: ${state}"
fi

if [[ "${state}" == phase16-go:* || "${state}" == resume-continue:* ]]; then
  resume_checkpoint="${state#*:}"
  run_trainer yes "${resume_checkpoint}"
  state="$(validate_state)" || blocked "continuation post-run validation failed"
fi

case "${state}" in
  complete) echo "T5GEMMA_TYPED_C2_VERPO150_COMPLETE output=${OUTPUT_DIR}" ;;
  stopped) echo "T5GEMMA_TYPED_C2_VERPO150_STOPPED_AT_WINDOW_GATE output=${OUTPUT_DIR}" ;;
  *) blocked "continuation ended in a nonterminal state: ${state}" ;;
esac
