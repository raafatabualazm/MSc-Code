#!/usr/bin/env bash
# Sparse bounded-teacher VeRPO over the exact sealed TRAIN feedback view, followed
# only after training by one matched four-arm evaluation on heldout175.
set -Eeuo pipefail

PATCH_ROOT="${PATCH_ROOT:-/workspace/hybrid_training_patch_v2_3}"
RS_ROOT="${RS_ROOT:-/workspace/artifacts/direct_compact_rs_sft_union2776_target24k}"
QWEN_ROOT="${QWEN_ROOT:-/workspace/artifacts/direct_compact_qwen38_union2776}"
EXEC_ROOT="${MULTIFUNCTION_EXECUTABLE_ROOT:-/workspace/multifunction_v1/expanded2776/executable_target24k}"
VERPO_FEEDBACK_ROOT="${VERPO_FEEDBACK_ROOT:-/workspace/multifunction_v1/expanded2776/verpo_feedback_target24k_seed42}"
MULTIFUNCTION_ROOT="${MULTIFUNCTION_ROOT:-/workspace/multifunction_v1/expanded2776}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
PYTHON="${PYTHON:-/venv/main/bin/python}"

ROLLOUT_FILE="${ROLLOUT_FILE:-${VERPO_FEEDBACK_ROOT}/verpo_rollout_feedback.jsonl}"
ROLLOUT_SEAL="${ROLLOUT_SEAL:-${VERPO_FEEDBACK_ROOT}/verpo_rollout_feedback.seal.json}"
TEACHER_PROMPT_JSONL="${TEACHER_PROMPT_JSONL:-${VERPO_FEEDBACK_ROOT}/verpo_teacher_f2.jsonl}"
TEACHER_PROMPT_MANIFEST="${TEACHER_PROMPT_MANIFEST:-${TEACHER_PROMPT_JSONL}.manifest.json}"
VERPO_FEEDBACK_REPORT="${VERPO_FEEDBACK_REPORT:-${VERPO_FEEDBACK_ROOT}/verpo_feedback_view.build.json}"
VERPO_FEEDBACK_PUBLIC="${VERPO_FEEDBACK_PUBLIC:-${VERPO_FEEDBACK_ROOT}/verpo_feedback_view.public.json}"
EXECUTABLE_VIEW_REPORT="${EXECUTABLE_VIEW_REPORT:-${EXEC_ROOT}/executable_view.build.json}"
CONTRACT="${CONTRACT:-${EXEC_ROOT}/compact_contract.json}"
CODEBOOK="${CODEBOOK:-${MULTIFUNCTION_ROOT}/multifunction_inline_cfg_v2_codebook.json}"
CODEC="${CODEC:-/workspace/scripts/data/build_multifunction_compact_v2.py}"
EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256="${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256:-}"
EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256:-f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f}"
EXPECTED_PARENT_FIT_ROWS="${EXPECTED_PARENT_FIT_ROWS:-2776}"
: "${EXPECTED_VERPO_FEEDBACK_REPORT_SHA256:?set the exact private feedback-view report SHA-256}"
: "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256:?set the exact trainer-safe feedback manifest SHA-256}"

QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-${QWEN_ROOT}/direct_compact_qwen_cot_sft}"
QWEN_BUILD_MANIFEST="${QWEN_BUILD_MANIFEST:-${QWEN_ROOT}/qwen_mc_sequence_train.build.json}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-${RS_ROOT}/02_gold_control}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-${RS_ROOT}/03_rs_sft}"
RS_BUILD_REPORT="${RS_BUILD_REPORT:-${RS_ROOT}/00_matched_data/build_report.json}"
RS_TRAIN_SIDE_HANDOFF="${RS_TRAIN_SIDE_HANDOFF:-${RS_ROOT}/train_side_handoff.json}"
CHAIN_CONTRACT="${CHAIN_CONTRACT:-/workspace/artifacts/post_qwen_union2776_target24k_chain.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/artifacts/direct_compact_verpo_union2776_target24k}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
VERPO_STORAGE_PREFLIGHT_ONLY="${VERPO_STORAGE_PREFLIGHT_ONLY:-false}"

HELDOUT="${HELDOUT:-}"
HELDOUT_SEAL="${HELDOUT_SEAL:-}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${OUTPUT_ROOT}/heldout175_evaluation}"

TRAIN_SEED="${TRAIN_SEED:-42}"
VERPO_GROUP_SIZE="${VERPO_GROUP_SIZE:-8}"
VERPO_ROLLOUT_BATCH_SIZE="${VERPO_ROLLOUT_BATCH_SIZE:-1}"
VERPO_TEMPERATURE="${VERPO_TEMPERATURE:-0.8}"
VERPO_MAX_NEW_TOKENS="${VERPO_MAX_NEW_TOKENS:-2048}"
MAX_UPDATES="${MAX_UPDATES:-}"
VERPO_CHECKPOINT_INTERVAL="${VERPO_CHECKPOINT_INTERVAL:-154}"
VERPO_LEARNING_RATE="${VERPO_LEARNING_RATE:-1e-6}"
VERPO_WEIGHT_DECAY="${VERPO_WEIGHT_DECAY:-0.0}"
VERPO_MAX_GRAD_NORM="${VERPO_MAX_GRAD_NORM:-1.0}"
VERPO_PPO_CLIP="${VERPO_PPO_CLIP:-0.0}"
VERPO_SFT_REPLAY_WEIGHT="${VERPO_SFT_REPLAY_WEIGHT:-0.05}"
VERPO_ON_POLICY_LOGPROB_TOLERANCE="${VERPO_ON_POLICY_LOGPROB_TOLERANCE:-0.0002}"
VERPO_ALPHA="${VERPO_ALPHA:-2.0}"
VERPO_BETA="${VERPO_BETA:-1.0}"
VERPO_REWARD_WORKERS="${VERPO_REWARD_WORKERS:-16}"
VERPO_REWARD_TIMEOUT="${VERPO_REWARD_TIMEOUT:-30}"
VERPO_REWARD_STABILITY_RUNS="${VERPO_REWARD_STABILITY_RUNS:-1}"
VERPO_JUDGE_WEIGHT="${VERPO_JUDGE_WEIGHT:-0.25}"
VERPO_ADVANTAGE_CONTRACT="${VERPO_ADVANTAGE_CONTRACT:-fnorm_1_separate_centering}"
VERPO_JUDGE_MODE="${VERPO_JUDGE_MODE:-sparse_inline}"
VERPO_JUDGE_MODEL="${VERPO_JUDGE_MODEL:-gpt-5.6-terra}"
VERPO_JUDGE_API_STYLE="${VERPO_JUDGE_API_STYLE:-openai_responses}"
VERPO_JUDGE_BASE_URL="${VERPO_JUDGE_BASE_URL:-https://api.openai.com/v1}"
VERPO_JUDGE_BASE_URL="${VERPO_JUDGE_BASE_URL%/}"
VERPO_JUDGE_CONCURRENCY="${VERPO_JUDGE_CONCURRENCY:-1}"
VERPO_JUDGE_MAX_TOKENS="${VERPO_JUDGE_MAX_TOKENS:-12288}"
VERPO_JUDGE_COMPLETION_RETRIES="${VERPO_JUDGE_COMPLETION_RETRIES:-0}"
VERPO_JUDGE_RETRY_MAX_TOKENS="${VERPO_JUDGE_RETRY_MAX_TOKENS:-12288}"
VERPO_JUDGE_THINKING_MODE="${VERPO_JUDGE_THINKING_MODE:-provider_default}"
VERPO_JUDGE_REASONING_MODE="${VERPO_JUDGE_REASONING_MODE:-standard}"
VERPO_JUDGE_REASONING_EFFORT="${VERPO_JUDGE_REASONING_EFFORT:-high}"
VERPO_JUDGE_TIMEOUT_SECONDS="${VERPO_JUDGE_TIMEOUT_SECONDS:-60}"
VERPO_JUDGE_MAX_RETRIES="${VERPO_JUDGE_MAX_RETRIES:-0}"
VERPO_JUDGE_INTERVAL="${VERPO_JUDGE_INTERVAL:-8}"
VERPO_JUDGE_GROUP_TOP_N="${VERPO_JUDGE_GROUP_TOP_N:-2}"
VERPO_JUDGE_DEADLINE_SECONDS="${VERPO_JUDGE_DEADLINE_SECONDS:-60}"
VERPO_JUDGE_FAILURE_POLICY="${VERPO_JUDGE_FAILURE_POLICY:-local_only}"
VERPO_JUDGE_MAX_CALLS="${VERPO_JUDGE_MAX_CALLS:-}"
VERPO_JUDGE_ESCALATION_QUEUE="${VERPO_JUDGE_ESCALATION_QUEUE:-${OUTPUT_ROOT}/offline_teacher_escalations.jsonl}"

K="${K:-10}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-24576}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.8}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_TOP_K="${EVAL_TOP_K:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-48}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}"
EVAL_STABILITY_RUNS="${EVAL_STABILITY_RUNS:-2}"

if (( $# != 0 )); then
  printf 'This sealed VeRPO launcher accepts no positional overrides\n' >&2
  exit 2
fi
if [[ "${VERPO_STORAGE_PREFLIGHT_ONLY}" != true \
   && "${VERPO_STORAGE_PREFLIGHT_ONLY}" != false ]]; then
  printf 'VERPO_STORAGE_PREFLIGHT_ONLY must be true or false\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256 is still an unsealed placeholder\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_CONTRACT_SHA256 is unsealed\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_VERPO_FEEDBACK_REPORT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_VERPO_FEEDBACK_REPORT_SHA256 is unsealed\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256 is unsealed\n' >&2
  exit 2
fi
if (( EVAL_MAX_NEW_TOKENS != 24576 || EVAL_BATCH_SIZE != 1 )); then
  printf 'Heldout generation must use target24k max_new_tokens=24576 and batch_size=1\n' >&2
  exit 2
fi
if [[ -f /workspace/data.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/data.env
  set +a
fi
if [[ -f /workspace/OpenAI.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/OpenAI.env
  set +a
fi
VERPO_JUDGE_BASE_URL="${VERPO_JUDGE_BASE_URL%/}"
if [[ "${VERPO_ADVANTAGE_CONTRACT}" != "fnorm_1_separate_centering" ]]; then
  printf 'VERPO_ADVANTAGE_CONTRACT must be fnorm_1_separate_centering\n' >&2
  exit 2
fi
if [[ "${VERPO_JUDGE_MODE}" == sparse_inline \
   && "${VERPO_JUDGE_REASONING_MODE}" != standard ]]; then
  printf 'GPT-5.6 Pro is offline-only; sparse inline VeRPO requires standard mode\n' >&2
  exit 2
fi
if [[ "${VERPO_JUDGE_FAILURE_POLICY}" != local_only \
   || "${VERPO_JUDGE_COMPLETION_RETRIES}" != 0 \
   || "${VERPO_JUDGE_MAX_RETRIES}" != 0 ]]; then
  printf 'Sparse inline teacher must fail to local rewards with zero blocking retries\n' >&2
  exit 2
fi
if [[ "${VERPO_JUDGE_TIMEOUT_SECONDS}" != "${VERPO_JUDGE_DEADLINE_SECONDS}" ]]; then
  printf 'VERPO_JUDGE_TIMEOUT_SECONDS and VERPO_JUDGE_DEADLINE_SECONDS must match\n' >&2
  exit 2
fi
for required in \
  "${PATCH_ROOT}/scripts/training/direct_compact_verpo.py" \
  "${PATCH_ROOT}/scripts/training/direct_compact_qwen_decompiler.py" \
  "${PATCH_ROOT}/scripts/training/verpo_judge_antigravity.py" \
  "${PATCH_ROOT}/scripts/evaluation/prepare_direct_compact_eval.py" \
  "${PATCH_ROOT}/scripts/evaluation/direct_compact_qwen_inference.py" \
  "${PATCH_ROOT}/scripts/evaluation/score_direct_compact_passk.py" \
  "${PATCH_ROOT}/scripts/evaluation/seal_post_qwen_evaluation_suite.py" \
  "${PATCH_ROOT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  "${PATCH_ROOT}/models/direct_compact_causal.py" \
  "${ROLLOUT_FILE}" "${ROLLOUT_SEAL}" "${CONTRACT}" "${CODEBOOK}" \
  "${CODEC}" "${TOKENIZER_JSON}" "${QWEN_BUILD_MANIFEST}" \
  "${TEACHER_PROMPT_JSONL}" "${TEACHER_PROMPT_MANIFEST}" \
  "${EXECUTABLE_VIEW_REPORT}" "${VERPO_FEEDBACK_REPORT}" \
  "${VERPO_FEEDBACK_PUBLIC}" "${RS_BUILD_REPORT}" \
  "${RS_TRAIN_SIDE_HANDOFF}" "${CHAIN_CONTRACT}" \
  "${QWEN_CHECKPOINT}/decoder_adapter/adapter_config.json" \
  "${QWEN_CHECKPOINT}/source_embedding_overlay.pt" \
  "${QWEN_CHECKPOINT}/run_provenance.json" \
  "${CONTROL_CHECKPOINT}/decoder_adapter/adapter_config.json" \
  "${CONTROL_CHECKPOINT}/source_embedding_overlay.pt" \
  "${CONTROL_CHECKPOINT}/run_provenance.json" \
  "${WARMSTART_CHECKPOINT}/decoder_adapter/adapter_config.json" \
  "${WARMSTART_CHECKPOINT}/source_embedding_overlay.pt" \
  "${WARMSTART_CHECKPOINT}/compact_contract.json" \
  "${WARMSTART_CHECKPOINT}/run_provenance.json"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Required multi-function VeRPO input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "${VERPO_FEEDBACK_REPORT}" | awk '{print $1}')" \
   != "${EXPECTED_VERPO_FEEDBACK_REPORT_SHA256}" \
   || "$(sha256sum "${VERPO_FEEDBACK_PUBLIC}" | awk '{print $1}')" \
      != "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}" ]]; then
  printf 'VeRPO feedback manifests differ from their pinned hashes\n' >&2
  exit 2
fi
mapfile -t SEALED_FEEDBACK_ACCOUNTING < <(
  "${PYTHON}" - "${VERPO_FEEDBACK_PUBLIC}" "${CHAIN_CONTRACT}" \
    "${EXPECTED_PARENT_FIT_ROWS}" \
    "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}" <<'PY'
import json
import pathlib
import sys
public = json.loads(pathlib.Path(sys.argv[1]).read_text())
chain = json.loads(pathlib.Path(sys.argv[2]).read_text())
payload = chain.get("payload") or {}
view = payload.get("executable_train") or {}
verpo = payload.get("verpo") or {}
accounting = public.get("accounting") or {}
digests = public.get("digests") or {}
if (
    public.get("schema") != "verpo-train-feedback-public-manifest-v1"
    or public.get("status") != "complete"
    or int(view.get("parent_rows", -1)) != int(sys.argv[3])
    or int(view.get("heldout_rows", -1)) != 175
    or int(accounting.get("parent_rows", -1)) != int(view.get("rows", -2))
    or (verpo.get("feedback_public_manifest") or {}).get("sha256")
       != sys.argv[4]
):
    raise SystemExit("sealed feedback/chain accounting is incoherent")
for key in (
    "parent_rows",
    "eligible_rows",
    "excluded_rows",
    "source_expect_cases",
    "visible_expect_cases",
    "holdback_expect_cases",
    "odd_case_tasks",
):
    print(int(accounting[key]))
print(digests["eligible_task_ids_sha256"])
print(digests["excluded_task_ids_sha256"])
print((view.get("heldout") or {})["path"])
print((view.get("heldout_seal") or {})["path"])
PY
)
EXPECTED_VERPO_PARENT_ROWS="${SEALED_FEEDBACK_ACCOUNTING[0]}"
EXPECTED_VERPO_ELIGIBLE_ROWS="${SEALED_FEEDBACK_ACCOUNTING[1]}"
EXPECTED_VERPO_EXCLUDED_ROWS="${SEALED_FEEDBACK_ACCOUNTING[2]}"
EXPECTED_VERPO_SOURCE_EXPECT_CASES="${SEALED_FEEDBACK_ACCOUNTING[3]}"
EXPECTED_VERPO_VISIBLE_EXPECT_CASES="${SEALED_FEEDBACK_ACCOUNTING[4]}"
EXPECTED_VERPO_HOLDBACK_EXPECT_CASES="${SEALED_FEEDBACK_ACCOUNTING[5]}"
EXPECTED_VERPO_ODD_CASE_TASKS="${SEALED_FEEDBACK_ACCOUNTING[6]}"
EXPECTED_VERPO_ELIGIBLE_TASK_IDS_SHA256="${SEALED_FEEDBACK_ACCOUNTING[7]}"
EXPECTED_VERPO_EXCLUDED_TASK_IDS_SHA256="${SEALED_FEEDBACK_ACCOUNTING[8]}"
SEALED_HELDOUT="${SEALED_FEEDBACK_ACCOUNTING[9]}"
SEALED_HELDOUT_SEAL="${SEALED_FEEDBACK_ACCOUNTING[10]}"
if [[ -n "${HELDOUT}" && "$(realpath -m -- "${HELDOUT}")" != "$(realpath -m -- "${SEALED_HELDOUT}")" ]]; then
  printf 'HELDOUT differs from the predeclared chain\n' >&2
  exit 2
fi
if [[ -n "${HELDOUT_SEAL}" && "$(realpath -m -- "${HELDOUT_SEAL}")" != "$(realpath -m -- "${SEALED_HELDOUT_SEAL}")" ]]; then
  printf 'HELDOUT_SEAL differs from the predeclared chain\n' >&2
  exit 2
fi
HELDOUT="${SEALED_HELDOUT}"
HELDOUT_SEAL="${SEALED_HELDOUT_SEAL}"
MAX_UPDATES="${MAX_UPDATES:-${EXPECTED_VERPO_ELIGIBLE_ROWS}}"
VERPO_JUDGE_MAX_CALLS="${VERPO_JUDGE_MAX_CALLS:-$((((MAX_UPDATES * VERPO_ROLLOUT_BATCH_SIZE) + VERPO_JUDGE_INTERVAL - 1) / VERPO_JUDGE_INTERVAL))}"
if (( MAX_UPDATES * VERPO_ROLLOUT_BATCH_SIZE < EXPECTED_VERPO_ELIGIBLE_ROWS )); then
  printf 'Production VeRPO must cover all %s eligible tasks at least once; planned groups=%s\n' \
    "${EXPECTED_VERPO_ELIGIBLE_ROWS}" \
    "$(( MAX_UPDATES * VERPO_ROLLOUT_BATCH_SIZE ))" >&2
  exit 2
fi
if [[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" \
   != "${EXPECTED_CONTRACT_SHA256}" ]]; then
  printf 'Target24k compact contract hash mismatch: %s\n' "${CONTRACT}" >&2
  exit 2
fi

# This is a train-side integrity check. It opens neither heldout data nor scores.
"${PYTHON}" - "${CHAIN_CONTRACT}" "${RS_TRAIN_SIDE_HANDOFF}" \
  "${RS_BUILD_REPORT}" "${WARMSTART_CHECKPOINT}" "${CONTROL_CHECKPOINT}" \
  "${OUTPUT_ROOT}" "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
  "${VERPO_GROUP_SIZE}" "${VERPO_ROLLOUT_BATCH_SIZE}" \
  "${VERPO_TEMPERATURE}" "${VERPO_MAX_NEW_TOKENS}" "${MAX_UPDATES}" \
  "${VERPO_LEARNING_RATE}" "${VERPO_WEIGHT_DECAY}" \
  "${VERPO_MAX_GRAD_NORM}" "${VERPO_PPO_CLIP}" \
  "${VERPO_SFT_REPLAY_WEIGHT}" "${VERPO_ON_POLICY_LOGPROB_TOLERANCE}" \
  "${VERPO_ALPHA}" "${VERPO_BETA}" \
  "${VERPO_REWARD_WORKERS}" "${VERPO_REWARD_TIMEOUT}" \
  "${VERPO_REWARD_STABILITY_RUNS}" "${VERPO_JUDGE_WEIGHT}" \
  "${VERPO_ADVANTAGE_CONTRACT}" "${VERPO_JUDGE_BASE_URL}" \
  "${VERPO_JUDGE_CONCURRENCY}" "${VERPO_JUDGE_MAX_TOKENS}" \
  "${VERPO_JUDGE_COMPLETION_RETRIES}" \
  "${VERPO_JUDGE_RETRY_MAX_TOKENS}" \
  "${VERPO_JUDGE_THINKING_MODE}" "${VERPO_JUDGE_TIMEOUT_SECONDS}" \
  "${VERPO_JUDGE_MAX_RETRIES}" "${TRAIN_SEED}" \
  "${K}" "${EVAL_MAX_NEW_TOKENS}" "${EVAL_TEMPERATURE}" \
  "${EVAL_TOP_P}" "${EVAL_TOP_K}" "${EVAL_BATCH_SIZE}" \
  "${EVAL_WORKERS}" "${EVAL_TIMEOUT}" "${EVAL_STABILITY_RUNS}" \
  "${VERPO_CHECKPOINT_INTERVAL}" "${RESUME_CHECKPOINT}" \
  "${EXPECTED_VERPO_FEEDBACK_REPORT_SHA256}" \
  "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}" \
  "${EXPECTED_VERPO_ELIGIBLE_ROWS}" "${EXPECTED_VERPO_EXCLUDED_ROWS}" \
  "${EXPECTED_VERPO_SOURCE_EXPECT_CASES}" \
  "${EXPECTED_VERPO_VISIBLE_EXPECT_CASES}" \
  "${EXPECTED_VERPO_HOLDBACK_EXPECT_CASES}" \
  "${EXPECTED_VERPO_ODD_CASE_TASKS}" \
  "${EXPECTED_VERPO_ELIGIBLE_TASK_IDS_SHA256}" \
  "${EXPECTED_VERPO_EXCLUDED_TASK_IDS_SHA256}" \
  "${EXPECTED_PARENT_FIT_ROWS}" \
  "${VERPO_JUDGE_MODEL}" "${VERPO_JUDGE_API_STYLE}" \
  "${VERPO_JUDGE_MODE}" "${VERPO_JUDGE_REASONING_MODE}" \
  "${VERPO_JUDGE_REASONING_EFFORT}" "${VERPO_JUDGE_INTERVAL}" \
  "${VERPO_JUDGE_GROUP_TOP_N}" "${VERPO_JUDGE_DEADLINE_SECONDS}" \
  "${VERPO_JUDGE_FAILURE_POLICY}" "${VERPO_JUDGE_MAX_CALLS}" \
  "${VERPO_JUDGE_ESCALATION_QUEUE}" <<'PY'
import hashlib, json, pathlib, sys
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
chain=json.loads(pathlib.Path(sys.argv[1]).read_text())
handoff=json.loads(pathlib.Path(sys.argv[2]).read_text())
build=json.loads(pathlib.Path(sys.argv[3]).read_text())
rs_root=pathlib.Path(sys.argv[4]).resolve()
control_root=pathlib.Path(sys.argv[5]).resolve()
payload=chain.get("payload") or {}
declared_rs=payload.get("rs_sft") or {}
declared_verpo=payload.get("verpo") or {}
declared_eval=payload.get("evaluation") or {}
view=payload.get("executable_train") or {}
expected_verpo={
    "group_size":int(sys.argv[8]),
    "rollout_batch_size":int(sys.argv[9]),
    "temperature":float(sys.argv[10]),
    "top_p":1.0,
    "top_k":0,
    "max_new_tokens":int(sys.argv[11]),
    "max_updates":int(sys.argv[12]),
    "checkpoint_interval":int(sys.argv[44]),
    "learning_rate":float(sys.argv[13]),
    "weight_decay":float(sys.argv[14]),
    "max_grad_norm":float(sys.argv[15]),
    "ppo_clip":float(sys.argv[16]),
    "sft_replay_weight":float(sys.argv[17]),
    "on_policy_logprob_tolerance":float(sys.argv[18]),
    "verpo_alpha":float(sys.argv[19]),
    "verpo_beta":float(sys.argv[20]),
    "reward_workers":int(sys.argv[21]),
    "reward_timeout":int(sys.argv[22]),
    "reward_stability_runs":int(sys.argv[23]),
    "judge_weight":float(sys.argv[24]),
    "advantage_contract":{
        "global":"full_pass_minus_group_mean",
        "local":"density_calibrated_reward_minus_group_mean",
        "teacher":"observed_selected_signal_minus_selected_mean_unobserved_zero",
        "unified":"A_global + verpo_beta*A_local + judge_weight*A_teacher",
        "normalization_factor":1,
        "population_std_division":False,
        "teacher_is_separately_centered_paper_extension":True,
        "missing_teacher_signal_is_neutral":True,
    },
    "judge_mode":sys.argv[59],
    "judge_api_style":sys.argv[58],
    "judge_base_url":sys.argv[26],
    "judge_concurrency":int(sys.argv[27]),
    "judge_max_tokens":int(sys.argv[28]),
    "judge_completion_retries":int(sys.argv[29]),
    "judge_retry_max_tokens":int(sys.argv[30]),
    "judge_thinking_mode":sys.argv[31],
    "judge_reasoning_mode":sys.argv[60],
    "judge_reasoning_effort":sys.argv[61],
    "judge_timeout_seconds":float(sys.argv[32]),
    "judge_max_retries":int(sys.argv[33]),
    "judge_interval":int(sys.argv[62]),
    "judge_group_top_n":int(sys.argv[63]),
    "judge_deadline_seconds":float(sys.argv[64]),
    "judge_failure_policy":sys.argv[65],
    "judge_max_calls":int(sys.argv[66]),
    "judge_escalation_queue":str(pathlib.Path(sys.argv[67]).resolve()),
    "decoder_model":"Qwen/Qwen3-8B",
    "decoder_revision":"b968826d9c46dd6066d109eabc6255188de91218",
    "attn_implementation":"flash_attention_2",
    "bf16":True,
    "fp16":False,
    "load_4bit":False,
    "resume_within_declared_output_only":True,
    "seed":int(sys.argv[34]),
    "heldout_loaded_during_training":False,
    "task_sampling_policy":"stable_sha256_epoch_permutation_without_replacement_then_cycle",
    "planned_rollout_groups":int(sys.argv[12]) * int(sys.argv[9]),
    "planned_unique_tasks":min(
        int(sys.argv[48]), int(sys.argv[12]) * int(sys.argv[9])
    ),
    "planned_unique_fraction":min(
        int(sys.argv[48]), int(sys.argv[12]) * int(sys.argv[9])
    ) / int(sys.argv[48]),
    "complete_dataset_cycles":(
        int(sys.argv[12]) * int(sys.argv[9]) // int(sys.argv[48])
    ),
    "partial_cycle_groups":(
        int(sys.argv[12]) * int(sys.argv[9]) % int(sys.argv[48])
    ),
}
if sys.argv[25] != "fnorm_1_separate_centering":
    raise SystemExit("VeRPO advantage contract sentinel mismatch")
expected_eval={
    "k":int(sys.argv[35]),
    "num_samples":int(sys.argv[35]),
    "max_new_tokens":int(sys.argv[36]),
    "temperature":float(sys.argv[37]),
    "top_p":float(sys.argv[38]),
    "top_k":int(sys.argv[39]),
    "batch_size":int(sys.argv[40]),
    "limit":0,
    "score_workers":int(sys.argv[41]),
    "timeout":int(sys.argv[42]),
    "stability_runs":int(sys.argv[43]),
    "decoder_model":"Qwen/Qwen3-8B",
    "decoder_revision":"b968826d9c46dd6066d109eabc6255188de91218",
    "attn_implementation":"flash_attention_2",
    "bf16":True,
    "seed":int(sys.argv[34]),
    "direct_prompt_modes":{
        "qwen_sequence_kd":"qwen_cot_v1",
        "matched_gold_control":"code_only_v1",
        "gpt_5_6_sol_rs_sft":"code_only_v1",
        "sparse_teacher_verpo":"code_only_v1",
    },
}
resume=sys.argv[45].strip()
if resume:
    resume_path=pathlib.Path(resume).resolve()
    if resume_path.parent != pathlib.Path(sys.argv[6]).resolve():
        raise SystemExit("resume checkpoint is outside the predeclared VeRPO output")
if (
    chain.get("schema") != "post-qwen-predeclared-training-chain-v1"
    or payload.get("stage_order_predeclared") is not True
    or payload.get("launch_decisions_from_heldout") is not False
    or handoff.get("schema") != "post-qwen-rs-train-side-handoff-v1"
    or handoff.get("passed") is not True
    or handoff.get("predeclared_chain_sha256") != sha(sys.argv[1])
    or handoff.get("matched_data_build_sha256") != sha(sys.argv[3])
    or build.get("low_coverage_smoke_override") is not False
    or pathlib.Path(declared_rs.get("intervention_output", "")).resolve() != rs_root
    or pathlib.Path(declared_rs.get("matched_control_output", "")).resolve() != control_root
    or pathlib.Path(declared_verpo.get("output", "")).resolve() != pathlib.Path(sys.argv[6]).resolve()
    or declared_verpo.get("teacher") != sys.argv[57]
    or declared_verpo.get("heldout_loaded_during_training") is not False
    or (view.get("report") or {}).get("sha256") != sys.argv[7]
    or int(view.get("parent_rows", -1)) != int(sys.argv[56])
    or int(view.get("heldout_rows", -1)) != 175
    or (declared_verpo.get("feedback_view_report") or {}).get("sha256")
       != sys.argv[46]
    or (declared_verpo.get("feedback_public_manifest") or {}).get("sha256")
       != sys.argv[47]
    or declared_verpo.get("rollout_rows") != int(sys.argv[48])
    or declared_verpo.get("parent_safe_rows")
       != int(sys.argv[48]) + int(sys.argv[49])
    or declared_verpo.get("parent_fit_rows") != int(sys.argv[56])
    or declared_verpo.get("feedback_accounting") != {
        "parent_rows":int(sys.argv[48]) + int(sys.argv[49]),
        "eligible_rows":int(sys.argv[48]),
        "excluded_rows":int(sys.argv[49]),
        "source_expect_cases":int(sys.argv[50]),
        "visible_expect_cases":int(sys.argv[51]),
        "holdback_expect_cases":int(sys.argv[52]),
        "odd_case_tasks":int(sys.argv[53]),
    }
    or declared_verpo.get("feedback_eligible_task_ids_sha256")
       != sys.argv[54]
    or declared_verpo.get("feedback_excluded_task_ids_sha256")
       != sys.argv[55]
    or any(declared_verpo.get(key) != item for key,item in expected_verpo.items())
    or any(declared_eval.get(key) != item for key,item in expected_eval.items())
):
    raise SystemExit("VeRPO invocation differs from predeclared train-side chain")
for checkpoint in (rs_root, control_root):
    prov=json.loads((checkpoint/"run_provenance.json").read_text())
    if prov.get("heldout_loaded_during_training") is not False:
        raise SystemExit("RS/control checkpoint loaded heldout during fitting")
print("PREDECLARED_VERPO_CHAIN_VERIFIED", flush=True)
PY

validate_completed_run() {
  "${PYTHON}" - "${OUTPUT_ROOT}" "${CONTRACT}" "${MAX_UPDATES}" \
    "${VERPO_CHECKPOINT_INTERVAL}" <<'PY'
import hashlib
import json
import pathlib
import sys

from scripts.training.direct_compact_verpo import validate_resume_checkpoint


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = pathlib.Path(sys.argv[1]).resolve()
contract = pathlib.Path(sys.argv[2]).resolve()
max_updates = int(sys.argv[3])
checkpoint_interval = int(sys.argv[4])
completed_path = root / "completed.json"
run_contract_path = root / "run_contract.json"
completed = json.loads(completed_path.read_text())
run_contract_sha256 = sha(run_contract_path)
latest = pathlib.Path(str(completed.get("latest_checkpoint") or "")).resolve()
provenance, paths = validate_resume_checkpoint(
    latest,
    output_dir=root,
    run_contract_sha256=run_contract_sha256,
    compact_contract_path=contract,
)
telemetry = json.loads(paths["judge_telemetry"].read_text())
if (
    completed.get("schema") != "direct-compact-verpo-completed-v1"
    or int(completed.get("optimizer_steps", -1)) != max_updates
    or int(completed.get("checkpoint_interval", -1)) != checkpoint_interval
    or latest.parent != root
    or int(provenance.get("optimizer_step", -1)) != max_updates
    or completed.get("latest_checkpoint_provenance_sha256")
       != sha(paths["checkpoint_provenance"])
    or completed.get("run_contract_sha256") != run_contract_sha256
    or completed.get("rollout_journal_chain_sha256")
       != provenance.get("rollout_journal_chain_sha256")
    or completed.get("latest_step_journal_sha256")
       != provenance.get("latest_step_journal_sha256")
    or int(completed.get("deepseek_receipt_count", -1))
       != int(telemetry.get("receipt_count", -2))
    or completed.get("deepseek_receipt_chain_sha256")
       != telemetry.get("receipt_chain_sha256")
    or completed.get("judge_response_ids_sha256")
       != provenance.get("judge_response_ids_sha256")
    or completed.get("judge_response_receipts_sha256")
       != provenance.get("judge_response_receipts_sha256")
    or completed.get("judge_telemetry") != telemetry
    or completed.get("judge_telemetry_cumulative") != telemetry
    or not isinstance(completed.get("finalize_only_recovery"), bool)
):
    raise SystemExit("completed VeRPO run failed immutable reuse validation")
print(f"VERPO_COMPLETED_RUN_VERIFIED checkpoint={latest}", flush=True)
PY
}

SKIP_TRAINING=false
RESUME_UNSTARTED=false
if [[ -f "${OUTPUT_ROOT}/completed.json" ]]; then
  validate_completed_run
  SKIP_TRAINING=true
elif [[ -d "${OUTPUT_ROOT}" ]]; then
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
      printf 'Requested RESUME_CHECKPOINT is missing: %s\n' "${RESUME_CHECKPOINT}" >&2
      exit 2
    fi
  else
    mapfile -t existing_checkpoints < <(
      find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
        -name 'checkpoint-optstep-[0-9][0-9][0-9][0-9][0-9][0-9]' \
        -print | sort
    )
    if (( ${#existing_checkpoints[@]} > 0 )); then
      RESUME_CHECKPOINT="${existing_checkpoints[-1]}"
      printf 'AUTO_RESUME_VERPO checkpoint=%s\n' "${RESUME_CHECKPOINT}"
    elif [[ -f "${OUTPUT_ROOT}/run_contract.json" ]]; then
      RESUME_UNSTARTED=true
      printf 'AUTO_RECOVER_UNSTARTED_VERPO output=%s\n' "${OUTPUT_ROOT}"
    else
      printf 'Existing VeRPO root has no sealed run contract: %s\n' "${OUTPUT_ROOT}" >&2
      exit 2
    fi
  fi
elif [[ -n "${RESUME_CHECKPOINT}" ]]; then
  printf 'Resume checkpoint was supplied but OUTPUT_ROOT does not exist\n' >&2
  exit 2
fi

if [[ "${SKIP_TRAINING}" != true ]] \
  && [[ "${VERPO_STORAGE_PREFLIGHT_ONLY}" != true ]] \
  && [[ "${VERPO_JUDGE_MODE}" == sparse_inline ]] \
  && [[ -z "${VERPO_JUDGE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'VERPO_JUDGE_API_KEY or OPENAI_API_KEY is required for sparse inline training\n' >&2
  exit 2
fi

mkdir -p /workspace/locks
exec 9>/workspace/locks/direct_compact_verpo.lock
if ! flock -n 9; then
  printf 'Another direct-compact VeRPO run holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[direct_compact_verpo] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

export PYTHONPATH="${PATCH_ROOT}:/workspace"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER_PROMPT_SHA256="$(sha256sum "${TEACHER_PROMPT_JSONL}" | awk '{print $1}')"
TEACHER_PROMPT_MANIFEST_SHA256="$(sha256sum "${TEACHER_PROMPT_MANIFEST}" | awk '{print $1}')"
CHAIN_CONTRACT_SHA256="$(sha256sum "${CHAIN_CONTRACT}" | awk '{print $1}')"

arguments=(
  --rollout_file "${ROLLOUT_FILE}"
  --rollout_seal "${ROLLOUT_SEAL}"
  --output_dir "${OUTPUT_ROOT}"
  --contract "${CONTRACT}"
  --codebook "${CODEBOOK}"
  --codec_artifact "${CODEC}"
  --tokenizer_json "${TOKENIZER_JSON}"
  --warmstart_checkpoint "${WARMSTART_CHECKPOINT}"
  --teacher_prompt_jsonl "${TEACHER_PROMPT_JSONL}"
  --expected_teacher_prompt_sha256 "${TEACHER_PROMPT_SHA256}"
  --teacher_prompt_manifest "${TEACHER_PROMPT_MANIFEST}"
  --expected_teacher_prompt_manifest_sha256 "${TEACHER_PROMPT_MANIFEST_SHA256}"
  --executable_view_report "${EXECUTABLE_VIEW_REPORT}"
  --expected_executable_view_report_sha256 "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}"
  --feedback_view_public_manifest "${VERPO_FEEDBACK_PUBLIC}"
  --expected_feedback_view_public_manifest_sha256 "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}"
  --expected_feedback_eligible_rows "${EXPECTED_VERPO_ELIGIBLE_ROWS}"
  --expected_feedback_excluded_rows "${EXPECTED_VERPO_EXCLUDED_ROWS}"
  --expected_feedback_source_expect_cases "${EXPECTED_VERPO_SOURCE_EXPECT_CASES}"
  --expected_feedback_visible_expect_cases "${EXPECTED_VERPO_VISIBLE_EXPECT_CASES}"
  --expected_feedback_holdback_expect_cases "${EXPECTED_VERPO_HOLDBACK_EXPECT_CASES}"
  --expected_feedback_odd_case_tasks "${EXPECTED_VERPO_ODD_CASE_TASKS}"
  --expected_feedback_eligible_task_ids_sha256 "${EXPECTED_VERPO_ELIGIBLE_TASK_IDS_SHA256}"
  --expected_feedback_excluded_task_ids_sha256 "${EXPECTED_VERPO_EXCLUDED_TASK_IDS_SHA256}"
  --derive_feedback_accounting_from_sealed_manifest
  --expected_parent_fit_rows "${EXPECTED_PARENT_FIT_ROWS}"
  --predeclared_chain_contract "${CHAIN_CONTRACT}"
  --expected_predeclared_chain_sha256 "${CHAIN_CONTRACT_SHA256}"
  --decoder_model Qwen/Qwen3-8B
  --decoder_revision b968826d9c46dd6066d109eabc6255188de91218
  --attn_implementation flash_attention_2
  --group_size "${VERPO_GROUP_SIZE:-8}"
  --rollout_batch_size "${VERPO_ROLLOUT_BATCH_SIZE}"
  --temperature "${VERPO_TEMPERATURE}"
  --top_p 1.0
  --top_k 0
  --max_new_tokens "${VERPO_MAX_NEW_TOKENS}"
  --max_updates "${MAX_UPDATES}"
  --checkpoint_interval "${VERPO_CHECKPOINT_INTERVAL}"
  --learning_rate "${VERPO_LEARNING_RATE}"
  --weight_decay "${VERPO_WEIGHT_DECAY}"
  --max_grad_norm "${VERPO_MAX_GRAD_NORM}"
  --ppo_clip "${VERPO_PPO_CLIP}"
  --sft_replay_weight "${VERPO_SFT_REPLAY_WEIGHT}"
  --on_policy_logprob_tolerance "${VERPO_ON_POLICY_LOGPROB_TOLERANCE}"
  --verpo_alpha "${VERPO_ALPHA}"
  --verpo_beta "${VERPO_BETA}"
  --reward_workers "${VERPO_REWARD_WORKERS}"
  --reward_timeout "${VERPO_REWARD_TIMEOUT}"
  --reward_stability_runs "${VERPO_REWARD_STABILITY_RUNS}"
  --judge_weight "${VERPO_JUDGE_WEIGHT}"
  --judge_mode "${VERPO_JUDGE_MODE}"
  --judge_model "${VERPO_JUDGE_MODEL}"
  --judge_api_style "${VERPO_JUDGE_API_STYLE}"
  --judge_base_url "${VERPO_JUDGE_BASE_URL}"
  --judge_concurrency "${VERPO_JUDGE_CONCURRENCY}"
  --judge_max_tokens "${VERPO_JUDGE_MAX_TOKENS}"
  --judge_completion_retries "${VERPO_JUDGE_COMPLETION_RETRIES}"
  --judge_retry_max_tokens "${VERPO_JUDGE_RETRY_MAX_TOKENS}"
  --judge_thinking_mode "${VERPO_JUDGE_THINKING_MODE}"
  --judge_reasoning_mode "${VERPO_JUDGE_REASONING_MODE}"
  --judge_reasoning_effort "${VERPO_JUDGE_REASONING_EFFORT}"
  --judge_timeout_seconds "${VERPO_JUDGE_TIMEOUT_SECONDS}"
  --judge_max_retries "${VERPO_JUDGE_MAX_RETRIES}"
  --judge_interval "${VERPO_JUDGE_INTERVAL}"
  --judge_group_top_n "${VERPO_JUDGE_GROUP_TOP_N}"
  --judge_deadline_seconds "${VERPO_JUDGE_DEADLINE_SECONDS}"
  --judge_failure_policy "${VERPO_JUDGE_FAILURE_POLICY}"
  --judge_max_calls "${VERPO_JUDGE_MAX_CALLS}"
  --judge_escalation_queue "${VERPO_JUDGE_ESCALATION_QUEUE}"
  --seed "${TRAIN_SEED}"
  --bf16
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  arguments+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
elif [[ "${RESUME_UNSTARTED}" == true ]]; then
  arguments+=(--resume_unstarted)
fi
if [[ "${VERPO_STORAGE_PREFLIGHT_ONLY}" == true ]]; then
  arguments+=(--storage_preflight_only)
fi

if [[ "${SKIP_TRAINING}" != true ]]; then
  "${PYTHON}" -m scripts.training.direct_compact_verpo "${arguments[@]}"
else
  printf 'VERPO_TRAINING_ALREADY_COMPLETE output=%s\n' "${OUTPUT_ROOT}"
fi
if [[ "${VERPO_STORAGE_PREFLIGHT_ONLY}" == true ]]; then
  printf 'VERPO_TRAIN_ONLY_STORAGE_PREFLIGHT_COMPLETE output=%s heldout_opened=false\n' \
    "${OUTPUT_ROOT}"
  exit 0
fi
validate_completed_run

# All fitting is now complete. Only now may the measure-only split be opened.
if [[ ! -f "${HELDOUT}" || ! -f "${HELDOUT_SEAL}" ]]; then
  printf 'Heldout175 artifacts are missing after training: %s %s\n' \
    "${HELDOUT}" "${HELDOUT_SEAL}" >&2
  exit 2
fi
COMPLETED="${OUTPUT_ROOT}/completed.json"
if [[ ! -f "${COMPLETED}" ]]; then
  printf 'VeRPO did not publish completed.json\n' >&2
  exit 2
fi
VERPO_CHECKPOINT="$("${PYTHON}" - "${COMPLETED}" <<'PY'
import json, pathlib, sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("schema") != "direct-compact-verpo-completed-v1":
    raise SystemExit("invalid VeRPO completion")
print(pathlib.Path(value["latest_checkpoint"]).resolve())
PY
)"

mkdir -p "${EVALUATION_ROOT}/predictions" "${EVALUATION_ROOT}/scores"
"${PYTHON}" -m scripts.evaluation.prepare_direct_compact_eval \
  --input "${HELDOUT}" \
  --output_dir "${EVALUATION_ROOT}/views" \
  --role measure

generate_arm() {
  local checkpoint="$1"
  local output="$2"
  local direct_prompt_mode="$3"
  "${PYTHON}" -m scripts.evaluation.direct_compact_qwen_inference \
    --dataset "${EVALUATION_ROOT}/views/public.jsonl" \
    --alignment "${EVALUATION_ROOT}/views/alignment.jsonl" \
    --role measure \
    --output "${output}" \
    --journal "${output}.generation.journal.jsonl" \
    --contract "${CONTRACT}" \
    --codebook "${CODEBOOK}" \
    --codec_artifact "${CODEC}" \
    --tokenizer_json "${TOKENIZER_JSON}" \
    --source_overlay "${checkpoint}/source_embedding_overlay.pt" \
    --decoder_adapter "${checkpoint}/decoder_adapter" \
    --decoder_model Qwen/Qwen3-8B \
    --decoder_revision b968826d9c46dd6066d109eabc6255188de91218 \
    --num_samples "${K}" \
    --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
    --direct_prompt_mode "${direct_prompt_mode}" \
    --temperature "${EVAL_TEMPERATURE}" \
    --top_p "${EVAL_TOP_P}" \
    --top_k "${EVAL_TOP_K}" \
    --seed "${TRAIN_SEED}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --bf16 \
    --attn_implementation flash_attention_2
}

score_arm() {
  local predictions="$1"
  local output="$2"
  "${PYTHON}" -m scripts.evaluation.score_direct_compact_passk \
    --predictions "${predictions}" \
    --evaluation_file "${EVALUATION_ROOT}/views/tests.jsonl" \
    --output "${output}" \
    --journal "${output}.evaluation.journal.jsonl" \
    --k "${K}" \
    --workers "${EVAL_WORKERS}" \
    --timeout "${EVAL_TIMEOUT}" \
    --stability_runs "${EVAL_STABILITY_RUNS}"
}

generate_arm "${QWEN_CHECKPOINT}" \
  "${EVALUATION_ROOT}/predictions/qwen.json" qwen_cot_v1
generate_arm "${CONTROL_CHECKPOINT}" \
  "${EVALUATION_ROOT}/predictions/control.json" code_only_v1
generate_arm "${WARMSTART_CHECKPOINT}" \
  "${EVALUATION_ROOT}/predictions/rs_sft.json" code_only_v1
generate_arm "${VERPO_CHECKPOINT}" \
  "${EVALUATION_ROOT}/predictions/verpo.json" code_only_v1

score_arm "${EVALUATION_ROOT}/predictions/qwen.json" "${EVALUATION_ROOT}/scores/qwen.json"
score_arm "${EVALUATION_ROOT}/predictions/control.json" "${EVALUATION_ROOT}/scores/control.json"
score_arm "${EVALUATION_ROOT}/predictions/rs_sft.json" "${EVALUATION_ROOT}/scores/rs_sft.json"
score_arm "${EVALUATION_ROOT}/predictions/verpo.json" "${EVALUATION_ROOT}/scores/verpo.json"

"${PYTHON}" -m scripts.evaluation.seal_post_qwen_evaluation_suite \
  --chain-contract "${CHAIN_CONTRACT}" \
  --heldout "${HELDOUT}" \
  --heldout-seal "${HELDOUT_SEAL}" \
  --eval-views-report "${EVALUATION_ROOT}/views/report.json" \
  --eval-public "${EVALUATION_ROOT}/views/public.jsonl" \
  --eval-alignment "${EVALUATION_ROOT}/views/alignment.jsonl" \
  --eval-tests "${EVALUATION_ROOT}/views/tests.jsonl" \
  --qwen-score "${EVALUATION_ROOT}/scores/qwen.json" \
  --qwen-checkpoint "${QWEN_CHECKPOINT}" \
  --control-score "${EVALUATION_ROOT}/scores/control.json" \
  --control-checkpoint "${CONTROL_CHECKPOINT}" \
  --rs-score "${EVALUATION_ROOT}/scores/rs_sft.json" \
  --rs-checkpoint "${WARMSTART_CHECKPOINT}" \
  --verpo-score "${EVALUATION_ROOT}/scores/verpo.json" \
  --verpo-checkpoint "${VERPO_CHECKPOINT}" \
  --verpo-completed "${COMPLETED}" \
  --output "${EVALUATION_ROOT}/evaluation_suite.json"

printf 'Predeclared Qwen/control/RS/VeRPO heldout175 suite complete: %s\n' \
  "${EVALUATION_ROOT}/evaluation_suite.json"
