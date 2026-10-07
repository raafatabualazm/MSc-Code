#!/usr/bin/env bash
# Predeclare the entire post-Qwen chain before either fitted stage starts.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_ROOT="${PATCH_ROOT:-/workspace/hybrid_training_patch_v2_3}"
PYTHON="${PYTHON:-/venv/main/bin/python}"

EXEC_ROOT="${MULTIFUNCTION_EXECUTABLE_ROOT:-/workspace/multifunction_v1/expanded2776/executable_target24k}"
MULTIFUNCTION_BUILD="${MULTIFUNCTION_BUILD:-/workspace/multifunction_v1/expanded2776}"
CONTRACT="${CONTRACT:-${EXEC_ROOT}/compact_contract.json}"
CODEBOOK="${CODEBOOK:-${MULTIFUNCTION_BUILD}/multifunction_inline_cfg_v2_codebook.json}"
CODEC="${CODEC:-/workspace/scripts/data/build_multifunction_compact_v2.py}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
VERPO_FEEDBACK_ROOT="${VERPO_FEEDBACK_ROOT:-/workspace/multifunction_v1/expanded2776/verpo_feedback_target24k_seed42}"
QWEN_ROOT="${QWEN_ROOT:-/workspace/artifacts/direct_compact_qwen38_union2776}"
CHATGPT_ROOT="${CHATGPT_ROOT:-/workspace/artifacts/chatgpt_rs_qwen38_union2776_target24k_gpt56}"
RS_OUTPUT_ROOT="${RS_OUTPUT_ROOT:-/workspace/artifacts/direct_compact_rs_sft_union2776_target24k}"
VERPO_OUTPUT_ROOT="${VERPO_OUTPUT_ROOT:-/workspace/artifacts/direct_compact_verpo_union2776_target24k}"
CHAIN_CONTRACT="${CHAIN_CONTRACT:-/workspace/artifacts/post_qwen_union2776_target24k_chain.json}"
POST_QWEN_DRY_RUN="${POST_QWEN_DRY_RUN:-false}"
HELDOUT="${HELDOUT:-/workspace/multifunction_v1/build/dev_multifunction_binary.jsonl}"
HELDOUT_SEAL="${HELDOUT_SEAL:-/workspace/multifunction_v1/build/dev_multifunction_binary.seal.json}"

QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-${QWEN_ROOT}/direct_compact_qwen_cot_sft}"
QWEN_BUILD_MANIFEST="${QWEN_BUILD_MANIFEST:-${QWEN_ROOT}/qwen_mc_sequence_train.build.json}"
CHATGPT_REPAIRS="${CHATGPT_REPAIRS:-${CHATGPT_ROOT}/verified_repairs.jsonl}"
CHATGPT_REPORT="${CHATGPT_REPORT:-${CHATGPT_ROOT}/report.json}"
EXECUTABLE_VIEW_REPORT="${EXECUTABLE_VIEW_REPORT:-${EXEC_ROOT}/executable_view.build.json}"
EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256="${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256:-}"
EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256:-f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f}"
EXPECTED_PARENT_FIT_ROWS="${EXPECTED_PARENT_FIT_ROWS:-2776}"
RS_HARVEST_VERIFIED_TARGET="${RS_HARVEST_VERIFIED_TARGET:-450}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-24576}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

if (( $# != 0 )); then
  printf 'This sealed pipeline accepts configuration through environment variables only\n' >&2
  exit 2
fi
if [[ "${POST_QWEN_DRY_RUN}" != true && "${POST_QWEN_DRY_RUN}" != false ]]; then
  printf 'POST_QWEN_DRY_RUN must be true or false\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256 is still an unsealed placeholder\n' >&2
  exit 2
fi
if [[ ! "${EXPECTED_CONTRACT_SHA256}" =~ ^[0-9a-f]{64}$ \
   || ! -f "${CONTRACT}" \
   || "$(sha256sum "${CONTRACT}" | awk '{print $1}')" \
      != "${EXPECTED_CONTRACT_SHA256}" ]]; then
  printf 'Target24k compact contract is missing or has the wrong hash: %s\n' \
    "${CONTRACT}" >&2
  exit 2
fi
if (( EVAL_MAX_NEW_TOKENS != 24576 || EVAL_BATCH_SIZE != 1 )); then
  printf 'Heldout generation must use target24k max_new_tokens=24576 and batch_size=1\n' >&2
  exit 2
fi
export PYTHONPATH="${PATCH_ROOT}:/workspace"

if [[ ! -e "${VERPO_FEEDBACK_ROOT}" ]]; then
  "${PYTHON}" -m scripts.preprocessing.build_verpo_feedback_view \
    --executable-dataset "${EXEC_ROOT}/train_multifunction_binary_executable.jsonl" \
    --executable-seal "${EXEC_ROOT}/train_multifunction_binary_executable.seal.json" \
    --executable-f2 "${EXEC_ROOT}/train_multifunction_binary_executable_f2.jsonl" \
    --executable-f2-manifest "${EXEC_ROOT}/train_multifunction_binary_executable_f2.jsonl.manifest.json" \
    --executable-view-report "${EXECUTABLE_VIEW_REPORT}" \
    --expected-executable-view-report-sha256 \
      "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
    --contract "${CONTRACT}" \
    --output-dir "${VERPO_FEEDBACK_ROOT}" \
    --seed "${TRAIN_SEED:-42}" \
    --expected-parent-fit-rows "${EXPECTED_PARENT_FIT_ROWS}" \
    --derive-sealed-accounting
fi
VERPO_FEEDBACK_REPORT="${VERPO_FEEDBACK_ROOT}/verpo_feedback_view.build.json"
VERPO_FEEDBACK_PUBLIC="${VERPO_FEEDBACK_ROOT}/verpo_feedback_view.public.json"
if [[ ! -f "${VERPO_FEEDBACK_REPORT}" || ! -f "${VERPO_FEEDBACK_PUBLIC}" ]]; then
  printf 'VeRPO feedback-view build is incomplete: %s\n' "${VERPO_FEEDBACK_ROOT}" >&2
  exit 2
fi
EXPECTED_VERPO_FEEDBACK_REPORT_SHA256="$(sha256sum "${VERPO_FEEDBACK_REPORT}" | awk '{print $1}')"
EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256="$(sha256sum "${VERPO_FEEDBACK_PUBLIC}" | awk '{print $1}')"
"${PYTHON}" - "${VERPO_FEEDBACK_ROOT}" "${EXEC_ROOT}" \
  "${EXPECTED_VERPO_FEEDBACK_REPORT_SHA256}" \
  "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
  "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}" \
  "${CONTRACT}" "${EXPECTED_PARENT_FIT_ROWS}" <<'PY'
import sys
from scripts.preprocessing.build_verpo_feedback_view import validate_feedback_view
feedback, executable, feedback_sha, executable_sha, public_sha = sys.argv[1:6]
result = validate_feedback_view(
    rollout=f"{feedback}/verpo_rollout_feedback.jsonl",
    seal=f"{feedback}/verpo_rollout_feedback.seal.json",
    f2=f"{feedback}/verpo_teacher_f2.jsonl",
    f2_manifest=f"{feedback}/verpo_teacher_f2.jsonl.manifest.json",
    build_report=f"{feedback}/verpo_feedback_view.build.json",
    expected_build_report_sha256=feedback_sha,
    public_manifest=f"{feedback}/verpo_feedback_view.public.json",
    expected_public_manifest_sha256=public_sha,
    executable_dataset=f"{executable}/train_multifunction_binary_executable.jsonl",
    executable_seal=f"{executable}/train_multifunction_binary_executable.seal.json",
    executable_f2=f"{executable}/train_multifunction_binary_executable_f2.jsonl",
    executable_f2_manifest=f"{executable}/train_multifunction_binary_executable_f2.jsonl.manifest.json",
    executable_view_report=f"{executable}/executable_view.build.json",
    expected_executable_view_report_sha256=executable_sha,
    contract=sys.argv[6],
    expected_parent_fit_rows=int(sys.argv[7]),
)
if result["acceptance_tests_exposed"] or result["reward_holdback_exposed"]:
    raise SystemExit("VeRPO feedback boundary failed")
print(f"VERPO_FEEDBACK_VIEW_VERIFIED eligible={result['rows']}", flush=True)
PY

mapfile -t SEALED_FEEDBACK_ACCOUNTING < <(
  "${PYTHON}" - "${VERPO_FEEDBACK_PUBLIC}" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
accounting = value["accounting"]
digests = value["digests"]
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
PY
)
EXECUTABLE_ROWS="${SEALED_FEEDBACK_ACCOUNTING[0]}"
EXPECTED_VERPO_ELIGIBLE_ROWS="${SEALED_FEEDBACK_ACCOUNTING[1]}"
EXPECTED_VERPO_EXCLUDED_ROWS="${SEALED_FEEDBACK_ACCOUNTING[2]}"
EXPECTED_VERPO_SOURCE_EXPECT_CASES="${SEALED_FEEDBACK_ACCOUNTING[3]}"
EXPECTED_VERPO_VISIBLE_EXPECT_CASES="${SEALED_FEEDBACK_ACCOUNTING[4]}"
EXPECTED_VERPO_HOLDBACK_EXPECT_CASES="${SEALED_FEEDBACK_ACCOUNTING[5]}"
EXPECTED_VERPO_ODD_CASE_TASKS="${SEALED_FEEDBACK_ACCOUNTING[6]}"
EXPECTED_VERPO_ELIGIBLE_TASK_IDS_SHA256="${SEALED_FEEDBACK_ACCOUNTING[7]}"
EXPECTED_VERPO_EXCLUDED_TASK_IDS_SHA256="${SEALED_FEEDBACK_ACCOUNTING[8]}"
ROWS_PER_ARM="${ROWS_PER_ARM:-$((2 * EXECUTABLE_ROWS))}"
MAX_UPDATES="${MAX_UPDATES:-${EXPECTED_VERPO_ELIGIBLE_ROWS}}"
VERPO_ROLLOUT_BATCH_SIZE="${VERPO_ROLLOUT_BATCH_SIZE:-1}"
VERPO_JUDGE_INTERVAL="${VERPO_JUDGE_INTERVAL:-8}"
VERPO_JUDGE_MAX_CALLS="${VERPO_JUDGE_MAX_CALLS:-$((((MAX_UPDATES * VERPO_ROLLOUT_BATCH_SIZE) + VERPO_JUDGE_INTERVAL - 1) / VERPO_JUDGE_INTERVAL))}"
VERPO_JUDGE_ESCALATION_QUEUE="${VERPO_JUDGE_ESCALATION_QUEUE:-${VERPO_OUTPUT_ROOT}/offline_teacher_escalations.jsonl}"

"${PYTHON}" -m scripts.training.seal_post_qwen_chain \
  --qwen-checkpoint "${QWEN_CHECKPOINT}" \
  --qwen-build-manifest "${QWEN_BUILD_MANIFEST}" \
  --executable-dataset "${EXEC_ROOT}/train_multifunction_binary_executable.jsonl" \
  --executable-seal "${EXEC_ROOT}/train_multifunction_binary_executable.seal.json" \
  --executable-f2 "${EXEC_ROOT}/train_multifunction_binary_executable_f2.jsonl" \
  --executable-f2-manifest "${EXEC_ROOT}/train_multifunction_binary_executable_f2.jsonl.manifest.json" \
  --executable-view-report "${EXECUTABLE_VIEW_REPORT}" \
  --expected-executable-view-report-sha256 \
    "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
  --expected-parent-fit-rows "${EXPECTED_PARENT_FIT_ROWS}" \
  --contract "${CONTRACT}" \
  --verpo-rollout "${VERPO_FEEDBACK_ROOT}/verpo_rollout_feedback.jsonl" \
  --verpo-rollout-seal "${VERPO_FEEDBACK_ROOT}/verpo_rollout_feedback.seal.json" \
  --verpo-teacher-f2 "${VERPO_FEEDBACK_ROOT}/verpo_teacher_f2.jsonl" \
  --verpo-teacher-f2-manifest "${VERPO_FEEDBACK_ROOT}/verpo_teacher_f2.jsonl.manifest.json" \
  --verpo-feedback-view-report "${VERPO_FEEDBACK_REPORT}" \
  --expected-verpo-feedback-view-report-sha256 \
    "${EXPECTED_VERPO_FEEDBACK_REPORT_SHA256}" \
  --verpo-feedback-public-manifest "${VERPO_FEEDBACK_PUBLIC}" \
  --expected-verpo-feedback-public-manifest-sha256 \
    "${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}" \
  --expected-verpo-eligible-rows "${EXPECTED_VERPO_ELIGIBLE_ROWS}" \
  --expected-verpo-excluded-rows "${EXPECTED_VERPO_EXCLUDED_ROWS}" \
  --expected-verpo-source-expect-cases \
    "${EXPECTED_VERPO_SOURCE_EXPECT_CASES}" \
  --expected-verpo-visible-expect-cases \
    "${EXPECTED_VERPO_VISIBLE_EXPECT_CASES}" \
  --expected-verpo-holdback-expect-cases \
    "${EXPECTED_VERPO_HOLDBACK_EXPECT_CASES}" \
  --expected-verpo-odd-case-tasks "${EXPECTED_VERPO_ODD_CASE_TASKS}" \
  --expected-verpo-eligible-task-ids-sha256 \
    "${EXPECTED_VERPO_ELIGIBLE_TASK_IDS_SHA256}" \
  --expected-verpo-excluded-task-ids-sha256 \
    "${EXPECTED_VERPO_EXCLUDED_TASK_IDS_SHA256}" \
  --derive-verpo-accounting-from-sealed-manifest \
  --repair-artifact "${CHATGPT_REPAIRS}" \
  --repair-report "${CHATGPT_REPORT}" \
  --rs-output-root "${RS_OUTPUT_ROOT}/03_rs_sft" \
  --control-output-root "${RS_OUTPUT_ROOT}/02_gold_control" \
  --verpo-output-root "${VERPO_OUTPUT_ROOT}" \
  --output "${CHAIN_CONTRACT}" \
  --seed "${TRAIN_SEED:-42}" \
  --rows-per-rs-arm "${ROWS_PER_ARM}" \
  --rows-per-rs-arm-from-sealed-parent \
  --rs-min-unique-repairs "${MIN_UNIQUE_REPAIRS:-400}" \
  --rs-harvest-verified-target "${RS_HARVEST_VERIFIED_TARGET}" \
  --rs-recertify-timeout "${RECERTIFY_TIMEOUT:-30}" \
  --rs-recertify-stability-runs "${RECERTIFY_STABILITY_RUNS:-2}" \
  --rs-learning-rate "${LEARNING_RATE:-2e-5}" \
  --rs-epochs "${EPOCHS:-1}" \
  --rs-max-steps "${MAX_STEPS:--1}" \
  --rs-batch-size 1 \
  --rs-grad-accum 16 \
  --rs-lora-r 64 \
  --rs-lora-alpha 128 \
  --rs-lora-dropout "${RS_LORA_DROPOUT:-0.05}" \
  --verpo-group-size "${VERPO_GROUP_SIZE:-8}" \
  --verpo-rollout-batch-size "${VERPO_ROLLOUT_BATCH_SIZE:-1}" \
  --verpo-temperature "${VERPO_TEMPERATURE:-0.8}" \
  --verpo-max-updates "${MAX_UPDATES}" \
  --verpo-checkpoint-interval "${VERPO_CHECKPOINT_INTERVAL:-154}" \
  --verpo-learning-rate "${VERPO_LEARNING_RATE:-1e-6}" \
  --verpo-weight-decay "${VERPO_WEIGHT_DECAY:-0.0}" \
  --verpo-max-grad-norm "${VERPO_MAX_GRAD_NORM:-1.0}" \
  --verpo-ppo-clip "${VERPO_PPO_CLIP:-0.0}" \
  --verpo-sft-replay-weight "${VERPO_SFT_REPLAY_WEIGHT:-0.05}" \
  --verpo-on-policy-logprob-tolerance \
    "${VERPO_ON_POLICY_LOGPROB_TOLERANCE:-0.0002}" \
  --verpo-alpha "${VERPO_ALPHA:-2.0}" \
  --verpo-beta "${VERPO_BETA:-1.0}" \
  --verpo-max-new-tokens "${VERPO_MAX_NEW_TOKENS:-2048}" \
  --verpo-reward-workers "${VERPO_REWARD_WORKERS:-16}" \
  --verpo-reward-timeout "${VERPO_REWARD_TIMEOUT:-30}" \
  --verpo-reward-stability-runs "${VERPO_REWARD_STABILITY_RUNS:-1}" \
  --verpo-judge-weight "${VERPO_JUDGE_WEIGHT:-0.25}" \
  --verpo-judge-mode "${VERPO_JUDGE_MODE:-sparse_inline}" \
  --verpo-judge-model "${VERPO_JUDGE_MODEL:-gpt-5.6-terra}" \
  --verpo-judge-api-style \
    "${VERPO_JUDGE_API_STYLE:-openai_responses}" \
  --verpo-judge-base-url \
    "${VERPO_JUDGE_BASE_URL:-https://api.openai.com/v1}" \
  --verpo-judge-concurrency "${VERPO_JUDGE_CONCURRENCY:-1}" \
  --verpo-judge-max-tokens "${VERPO_JUDGE_MAX_TOKENS:-12288}" \
  --verpo-judge-completion-retries \
    "${VERPO_JUDGE_COMPLETION_RETRIES:-0}" \
  --verpo-judge-retry-max-tokens \
    "${VERPO_JUDGE_RETRY_MAX_TOKENS:-12288}" \
  --verpo-judge-thinking-mode \
    "${VERPO_JUDGE_THINKING_MODE:-provider_default}" \
  --verpo-judge-reasoning-mode \
    "${VERPO_JUDGE_REASONING_MODE:-standard}" \
  --verpo-judge-reasoning-effort \
    "${VERPO_JUDGE_REASONING_EFFORT:-high}" \
  --verpo-judge-timeout-seconds \
    "${VERPO_JUDGE_TIMEOUT_SECONDS:-60}" \
  --verpo-judge-max-retries "${VERPO_JUDGE_MAX_RETRIES:-0}" \
  --verpo-judge-interval "${VERPO_JUDGE_INTERVAL}" \
  --verpo-judge-group-top-n "${VERPO_JUDGE_GROUP_TOP_N:-2}" \
  --verpo-judge-deadline-seconds \
    "${VERPO_JUDGE_DEADLINE_SECONDS:-60}" \
  --verpo-judge-failure-policy \
    "${VERPO_JUDGE_FAILURE_POLICY:-local_only}" \
  --verpo-judge-max-calls "${VERPO_JUDGE_MAX_CALLS}" \
  --verpo-judge-escalation-queue "${VERPO_JUDGE_ESCALATION_QUEUE}" \
  --evaluation-k "${K:-10}" \
  --evaluation-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --evaluation-temperature "${EVAL_TEMPERATURE:-0.8}" \
  --evaluation-top-p "${EVAL_TOP_P:-0.95}" \
  --evaluation-top-k "${EVAL_TOP_K:-0}" \
  --evaluation-batch-size "${EVAL_BATCH_SIZE}" \
  --qwen-evaluation-direct-prompt-mode qwen_cot_v1 \
  --post-rs-evaluation-direct-prompt-mode code_only_v1 \
  --evaluation-workers "${EVAL_WORKERS:-48}" \
  --evaluation-timeout "${EVAL_TIMEOUT:-30}" \
  --evaluation-stability-runs "${EVAL_STABILITY_RUNS:-2}"

if [[ "${POST_QWEN_DRY_RUN}" == true ]]; then
  printf 'POST_QWEN_TRAIN_ONLY_DRY_RUN_COMPLETE chain=%s feedback_public=%s heldout_opened=false\n' \
    "${CHAIN_CONTRACT}" "${VERPO_FEEDBACK_PUBLIC}"
  exit 0
fi

PATCH_ROOT="${PATCH_ROOT}" \
MULTIFUNCTION_EXECUTABLE_ROOT="${EXEC_ROOT}" \
MULTIFUNCTION_BUILD="${MULTIFUNCTION_BUILD}" \
CONTRACT="${CONTRACT}" \
CODEBOOK="${CODEBOOK}" \
CODEC="${CODEC}" \
TOKENIZER_JSON="${TOKENIZER_JSON}" \
QWEN_BUILD_MANIFEST="${QWEN_BUILD_MANIFEST}" \
WARMSTART_CHECKPOINT="${QWEN_CHECKPOINT}" \
CHATGPT_ROOT="${CHATGPT_ROOT}" \
CHATGPT_REPAIRS="${CHATGPT_REPAIRS}" \
CHATGPT_REPORT="${CHATGPT_REPORT}" \
CHAIN_CONTRACT="${CHAIN_CONTRACT}" \
EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256="${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256}" \
EXPECTED_PARENT_FIT_ROWS="${EXPECTED_PARENT_FIT_ROWS}" \
OUTPUT_ROOT="${RS_OUTPUT_ROOT}" \
MIN_UNIQUE_REPAIRS="${MIN_UNIQUE_REPAIRS:-400}" \
RECERTIFY_TIMEOUT="${RECERTIFY_TIMEOUT:-30}" \
RECERTIFY_STABILITY_RUNS="${RECERTIFY_STABILITY_RUNS:-2}" \
LEARNING_RATE="${LEARNING_RATE:-2e-5}" \
EPOCHS="${EPOCHS:-1}" \
MAX_STEPS="${MAX_STEPS:--1}" \
ROWS_PER_ARM="${ROWS_PER_ARM}" \
TRAIN_SEED="${TRAIN_SEED:-42}" \
RS_LORA_DROPOUT="${RS_LORA_DROPOUT:-0.05}" \
  bash "${SCRIPT_DIR}/run_finish_rs_sft.sh"

PATCH_ROOT="${PATCH_ROOT}" \
MULTIFUNCTION_EXECUTABLE_ROOT="${EXEC_ROOT}" \
MULTIFUNCTION_ROOT="${MULTIFUNCTION_BUILD}" \
CONTRACT="${CONTRACT}" \
CODEBOOK="${CODEBOOK}" \
CODEC="${CODEC}" \
TOKENIZER_JSON="${TOKENIZER_JSON}" \
QWEN_ROOT="${QWEN_ROOT}" \
QWEN_CHECKPOINT="${QWEN_CHECKPOINT}" \
QWEN_BUILD_MANIFEST="${QWEN_BUILD_MANIFEST}" \
RS_ROOT="${RS_OUTPUT_ROOT}" \
CHAIN_CONTRACT="${CHAIN_CONTRACT}" \
EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256="${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256}" \
EXPECTED_PARENT_FIT_ROWS="${EXPECTED_PARENT_FIT_ROWS}" \
VERPO_FEEDBACK_ROOT="${VERPO_FEEDBACK_ROOT}" \
EXPECTED_VERPO_FEEDBACK_REPORT_SHA256="${EXPECTED_VERPO_FEEDBACK_REPORT_SHA256}" \
EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256="${EXPECTED_VERPO_FEEDBACK_PUBLIC_SHA256}" \
EXPECTED_VERPO_ELIGIBLE_ROWS="${EXPECTED_VERPO_ELIGIBLE_ROWS}" \
EXPECTED_VERPO_EXCLUDED_ROWS="${EXPECTED_VERPO_EXCLUDED_ROWS}" \
EXPECTED_VERPO_SOURCE_EXPECT_CASES="${EXPECTED_VERPO_SOURCE_EXPECT_CASES}" \
EXPECTED_VERPO_VISIBLE_EXPECT_CASES="${EXPECTED_VERPO_VISIBLE_EXPECT_CASES}" \
EXPECTED_VERPO_HOLDBACK_EXPECT_CASES="${EXPECTED_VERPO_HOLDBACK_EXPECT_CASES}" \
EXPECTED_VERPO_ODD_CASE_TASKS="${EXPECTED_VERPO_ODD_CASE_TASKS}" \
EXPECTED_VERPO_ELIGIBLE_TASK_IDS_SHA256="${EXPECTED_VERPO_ELIGIBLE_TASK_IDS_SHA256}" \
EXPECTED_VERPO_EXCLUDED_TASK_IDS_SHA256="${EXPECTED_VERPO_EXCLUDED_TASK_IDS_SHA256}" \
HELDOUT="${HELDOUT}" \
HELDOUT_SEAL="${HELDOUT_SEAL}" \
OUTPUT_ROOT="${VERPO_OUTPUT_ROOT}" \
TRAIN_SEED="${TRAIN_SEED:-42}" \
VERPO_GROUP_SIZE="${VERPO_GROUP_SIZE:-8}" \
VERPO_ROLLOUT_BATCH_SIZE="${VERPO_ROLLOUT_BATCH_SIZE:-1}" \
VERPO_TEMPERATURE="${VERPO_TEMPERATURE:-0.8}" \
VERPO_MAX_NEW_TOKENS="${VERPO_MAX_NEW_TOKENS:-2048}" \
MAX_UPDATES="${MAX_UPDATES}" \
VERPO_CHECKPOINT_INTERVAL="${VERPO_CHECKPOINT_INTERVAL:-154}" \
VERPO_LEARNING_RATE="${VERPO_LEARNING_RATE:-1e-6}" \
VERPO_WEIGHT_DECAY="${VERPO_WEIGHT_DECAY:-0.0}" \
VERPO_MAX_GRAD_NORM="${VERPO_MAX_GRAD_NORM:-1.0}" \
VERPO_PPO_CLIP="${VERPO_PPO_CLIP:-0.0}" \
VERPO_SFT_REPLAY_WEIGHT="${VERPO_SFT_REPLAY_WEIGHT:-0.05}" \
VERPO_ON_POLICY_LOGPROB_TOLERANCE="${VERPO_ON_POLICY_LOGPROB_TOLERANCE:-0.0002}" \
VERPO_ALPHA="${VERPO_ALPHA:-2.0}" \
VERPO_BETA="${VERPO_BETA:-1.0}" \
VERPO_REWARD_WORKERS="${VERPO_REWARD_WORKERS:-16}" \
VERPO_REWARD_TIMEOUT="${VERPO_REWARD_TIMEOUT:-30}" \
VERPO_REWARD_STABILITY_RUNS="${VERPO_REWARD_STABILITY_RUNS:-1}" \
VERPO_JUDGE_WEIGHT="${VERPO_JUDGE_WEIGHT:-0.25}" \
VERPO_ADVANTAGE_CONTRACT="fnorm_1_separate_centering" \
VERPO_JUDGE_MODE="${VERPO_JUDGE_MODE:-sparse_inline}" \
VERPO_JUDGE_MODEL="${VERPO_JUDGE_MODEL:-gpt-5.6-terra}" \
VERPO_JUDGE_API_STYLE="${VERPO_JUDGE_API_STYLE:-openai_responses}" \
VERPO_JUDGE_BASE_URL="${VERPO_JUDGE_BASE_URL:-https://api.openai.com/v1}" \
VERPO_JUDGE_CONCURRENCY="${VERPO_JUDGE_CONCURRENCY:-1}" \
VERPO_JUDGE_MAX_TOKENS="${VERPO_JUDGE_MAX_TOKENS:-12288}" \
VERPO_JUDGE_COMPLETION_RETRIES="${VERPO_JUDGE_COMPLETION_RETRIES:-0}" \
VERPO_JUDGE_RETRY_MAX_TOKENS="${VERPO_JUDGE_RETRY_MAX_TOKENS:-12288}" \
VERPO_JUDGE_THINKING_MODE="${VERPO_JUDGE_THINKING_MODE:-provider_default}" \
VERPO_JUDGE_REASONING_MODE="${VERPO_JUDGE_REASONING_MODE:-standard}" \
VERPO_JUDGE_REASONING_EFFORT="${VERPO_JUDGE_REASONING_EFFORT:-high}" \
VERPO_JUDGE_TIMEOUT_SECONDS="${VERPO_JUDGE_TIMEOUT_SECONDS:-60}" \
VERPO_JUDGE_MAX_RETRIES="${VERPO_JUDGE_MAX_RETRIES:-0}" \
VERPO_JUDGE_INTERVAL="${VERPO_JUDGE_INTERVAL}" \
VERPO_JUDGE_GROUP_TOP_N="${VERPO_JUDGE_GROUP_TOP_N:-2}" \
VERPO_JUDGE_DEADLINE_SECONDS="${VERPO_JUDGE_DEADLINE_SECONDS:-60}" \
VERPO_JUDGE_FAILURE_POLICY="${VERPO_JUDGE_FAILURE_POLICY:-local_only}" \
VERPO_JUDGE_MAX_CALLS="${VERPO_JUDGE_MAX_CALLS}" \
VERPO_JUDGE_ESCALATION_QUEUE="${VERPO_JUDGE_ESCALATION_QUEUE}" \
K="${K:-10}" \
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}" \
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.8}" \
EVAL_TOP_P="${EVAL_TOP_P:-0.95}" \
EVAL_TOP_K="${EVAL_TOP_K:-0}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
EVAL_WORKERS="${EVAL_WORKERS:-48}" \
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30}" \
EVAL_STABILITY_RUNS="${EVAL_STABILITY_RUNS:-2}" \
  bash "${SCRIPT_DIR}/run_verpo_v2.sh"
