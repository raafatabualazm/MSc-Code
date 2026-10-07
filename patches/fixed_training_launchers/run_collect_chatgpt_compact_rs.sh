#!/usr/bin/env bash
# Fresh executable-fit failures from the sealed Qwen sequence-KD + CoT checkpoint,
# followed by synchronous official OpenAI Responses API repair harvesting.
set -Eeuo pipefail

PATCH_ROOT="${PATCH_ROOT:-/workspace/hybrid_training_patch_v2_3}"
EXEC_ROOT="${MULTIFUNCTION_EXECUTABLE_ROOT:-/workspace/multifunction_v1/expanded2776/executable_target24k}"
MULTIFUNCTION_BUILD="${MULTIFUNCTION_BUILD:-/workspace/multifunction_v1/expanded2776}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-/workspace/artifacts/direct_compact_qwen38_union2776/direct_compact_qwen_cot_sft}"
QWEN_BUILD_MANIFEST="${QWEN_BUILD_MANIFEST:-$(dirname "${STUDENT_CHECKPOINT}")/qwen_mc_sequence_train.build.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/artifacts/chatgpt_rs_qwen38_union2776_target24k_gpt56}"
PYTHON="${PYTHON:-/venv/main/bin/python}"

TRAIN_FILE="${TRAIN_FILE:-${EXEC_ROOT}/train_multifunction_binary_executable.jsonl}"
TRAIN_SEAL="${TRAIN_SEAL:-${EXEC_ROOT}/train_multifunction_binary_executable.seal.json}"
SERIALIZED="${SERIALIZED:-${EXEC_ROOT}/train_multifunction_binary_executable_f2.jsonl}"
SERIALIZED_MANIFEST="${SERIALIZED_MANIFEST:-${SERIALIZED}.manifest.json}"
EXECUTABLE_VIEW_REPORT="${EXECUTABLE_VIEW_REPORT:-${EXEC_ROOT}/executable_view.build.json}"
CONTRACT="${CONTRACT:-${EXEC_ROOT}/compact_contract.json}"
CODEBOOK="${CODEBOOK:-${MULTIFUNCTION_BUILD}/multifunction_inline_cfg_v2_codebook.json}"
CODEC="${CODEC:-/workspace/scripts/data/build_multifunction_compact_v2.py}"

# Filled only after the exact multi-function executable view is built.  A
# placeholder deliberately aborts before GPU or API work.
EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256="${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256:-}"
EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256:-f51583b5020c0989c7d20e28cb270d4701b8b8d4fc7955296204959b940fd69f}"
EXPECTED_PARENT_FIT_ROWS="${EXPECTED_PARENT_FIT_ROWS:-2776}"

STUDENT_K="${STUDENT_K:-1}"
STUDENT_MAX_NEW_TOKENS="${STUDENT_MAX_NEW_TOKENS:-24576}"
CHATGPT_MODEL="${CHATGPT_MODEL:-gpt-5.6-sol}"
CHATGPT_REASONING_EFFORT="${CHATGPT_REASONING_EFFORT:-high}"
CHATGPT_SAMPLES_PER_TASK="${CHATGPT_SAMPLES_PER_TASK:-4}"
RS_RECERTIFICATION_FLOOR="${RS_RECERTIFICATION_FLOOR:-400}"
HARVEST_VERIFIED_TASKS="${HARVEST_VERIFIED_TASKS:-450}"
LIMIT="${LIMIT:-0}"

require_sha256() {
  local value="$1"
  local label="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s must be replaced with the exact lowercase SHA-256; got %s\n' \
      "${label}" "${value}" >&2
    exit 2
  fi
}
require_sha256 "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
  EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256
require_sha256 "${EXPECTED_CONTRACT_SHA256}" EXPECTED_CONTRACT_SHA256
if (( STUDENT_MAX_NEW_TOKENS != 24576 )); then
  printf 'STUDENT_MAX_NEW_TOKENS must remain 24576 for the target24k failure-generation contract\n' >&2
  exit 2
fi
if (( HARVEST_VERIFIED_TASKS <= RS_RECERTIFICATION_FLOOR )); then
  printf 'HARVEST_VERIFIED_TASKS must exceed RS_RECERTIFICATION_FLOOR to preserve a recertification buffer\n' >&2
  exit 2
fi

for required in \
  "${PATCH_ROOT}/scripts/preprocessing/build_multifunction_executable_view.py" \
  "${PATCH_ROOT}/scripts/evaluation/prepare_direct_compact_eval.py" \
  "${PATCH_ROOT}/scripts/evaluation/direct_compact_qwen_inference.py" \
  "${PATCH_ROOT}/scripts/evaluation/score_direct_compact_passk.py" \
  "${PATCH_ROOT}/scripts/training/collect_chatgpt_compact_rs.py" \
  "${TRAIN_FILE}" "${TRAIN_SEAL}" "${CONTRACT}" "${CODEBOOK}" "${CODEC}" \
  "${TOKENIZER_JSON}" "${SERIALIZED}" "${SERIALIZED_MANIFEST}" \
  "${EXECUTABLE_VIEW_REPORT}" \
  "${STUDENT_CHECKPOINT}/decoder_adapter/adapter_config.json" \
  "${STUDENT_CHECKPOINT}/source_embedding_overlay.pt" \
  "${STUDENT_CHECKPOINT}/compact_contract.json" \
  "${STUDENT_CHECKPOINT}/run_provenance.json" \
  "${QWEN_BUILD_MANIFEST}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Required post-Qwen RS input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "${CONTRACT}" | awk '{print $1}')" \
   != "${EXPECTED_CONTRACT_SHA256}" ]]; then
  printf 'Target24k compact contract hash mismatch: %s\n' "${CONTRACT}" >&2
  exit 2
fi

mkdir -p /workspace/locks
exec 9>/workspace/locks/chatgpt_compact_rs.lock
if ! flock -n 9; then
  printf 'Another ChatGPT compact RS harvest holds the lock\n' >&2
  exit 3
fi
trap 'status=$?; printf "[chatgpt_compact_rs] %s exit=%s\n" "$(date -u +%FT%TZ)" "${status}" >&2' EXIT

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PATCH_ROOT}:/workspace"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Train-side only: this opens no heldout bytes.
"${PYTHON}" - "${TRAIN_FILE}" "${TRAIN_SEAL}" "${SERIALIZED}" \
  "${SERIALIZED_MANIFEST}" "${EXECUTABLE_VIEW_REPORT}" \
  "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" "${CONTRACT}" \
  "${EXPECTED_PARENT_FIT_ROWS}" <<'PY'
import sys
from scripts.preprocessing.build_multifunction_executable_view import validate_executable_view
result = validate_executable_view(
    dataset=sys.argv[1],
    seal=sys.argv[2],
    f2=sys.argv[3],
    f2_manifest=sys.argv[4],
    build_report=sys.argv[5],
    expected_build_report_sha256=sys.argv[6],
    contract=sys.argv[7],
    verify_heldout=False,
    expected_parent_rows=int(sys.argv[8]),
)
if (
    result["parent_rows"] != int(sys.argv[8])
    or result["heldout_rows"] != 175
    or result["heldout_bytes_opened_during_validation"]
):
    raise SystemExit("expanded executable train-only validation failed")
print(
    f"MULTIFUNCTION_EXECUTABLE_VIEW_VERIFIED "
    f"parent={result['parent_rows']} executable={result['rows']}",
    flush=True,
)
PY

"${PYTHON}" -c \
  'import sys; from scripts.training.collect_chatgpt_compact_rs import validate_qwen_student_checkpoint; validate_qwen_student_checkpoint(sys.argv[1], qwen_build_manifest=sys.argv[2]); print("QWEN_STUDENT_CHECKPOINT_VERIFIED", flush=True)' \
  "${STUDENT_CHECKPOINT}" "${QWEN_BUILD_MANIFEST}"

EVAL_VIEWS="${OUTPUT_ROOT}/00_train_views"
PREDICTIONS="${OUTPUT_ROOT}/01_fresh_qwen_student_predictions.json"
SCORE="${OUTPUT_ROOT}/02_fresh_qwen_student_scores.json"
"${PYTHON}" -m scripts.evaluation.prepare_direct_compact_eval \
  --input "${TRAIN_FILE}" \
  --output_dir "${EVAL_VIEWS}" \
  --role fit

# These commands validate and reuse complete immutable outputs, resume their
# sealed journals when incomplete, and fail closed on any contract drift.
"${PYTHON}" -m scripts.evaluation.direct_compact_qwen_inference \
  --dataset "${EVAL_VIEWS}/public.jsonl" \
  --alignment "${EVAL_VIEWS}/alignment.jsonl" \
  --role fit \
  --output "${PREDICTIONS}" \
  --journal "${PREDICTIONS}.generation.journal.jsonl" \
  --contract "${CONTRACT}" \
  --codebook "${CODEBOOK}" \
  --codec_artifact "${CODEC}" \
  --tokenizer_json "${TOKENIZER_JSON}" \
  --source_overlay "${STUDENT_CHECKPOINT}/source_embedding_overlay.pt" \
  --decoder_adapter "${STUDENT_CHECKPOINT}/decoder_adapter" \
  --decoder_model Qwen/Qwen3-8B \
  --decoder_revision b968826d9c46dd6066d109eabc6255188de91218 \
  --num_samples "${STUDENT_K}" \
  --max_new_tokens "${STUDENT_MAX_NEW_TOKENS}" \
  --direct_prompt_mode qwen_cot_v1 \
  --temperature 0.8 \
  --top_p 0.95 \
  --top_k 0 \
  --seed 42 \
  --batch_size 1 \
  --bf16 \
  --attn_implementation flash_attention_2

"${PYTHON}" -m scripts.evaluation.score_direct_compact_passk \
  --predictions "${PREDICTIONS}" \
  --evaluation_file "${TRAIN_FILE}" \
  --output "${SCORE}" \
  --journal "${SCORE}.evaluation.journal.jsonl" \
  --k "${STUDENT_K}" \
  --workers "${EVAL_WORKERS:-48}" \
  --timeout "${EVAL_TIMEOUT:-30}" \
  --stability_runs "${EVAL_STABILITY_RUNS:-2}"

if [[ ! -f "${OPENAI_ENV_FILE:-/workspace/OpenAI.env}" && -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'Fresh Qwen failures are ready; set OPENAI_API_KEY or create %s\n' \
    "${OPENAI_ENV_FILE:-/workspace/OpenAI.env}" >&2
  exit 2
fi

"${PYTHON}" -m scripts.training.collect_chatgpt_compact_rs \
  --serialized_inputs "${SERIALIZED}" \
  --serialized_manifest "${SERIALIZED_MANIFEST}" \
  --tokenizer_json "${TOKENIZER_JSON}" \
  --train_file "${TRAIN_FILE}" \
  --train_seal "${TRAIN_SEAL}" \
  --score_report "${SCORE}" \
  --predictions "${PREDICTIONS}" \
  --student_checkpoint "${STUDENT_CHECKPOINT}" \
  --qwen_build_manifest "${QWEN_BUILD_MANIFEST}" \
  --executable_view_report "${EXECUTABLE_VIEW_REPORT}" \
  --expected_executable_view_report_sha256 \
    "${EXPECTED_EXECUTABLE_VIEW_REPORT_SHA256}" \
  --output_dir "${OUTPUT_ROOT}" \
  --env_file "${OPENAI_ENV_FILE:-/workspace/OpenAI.env}" \
  --base_url https://api.openai.com/v1 \
  --model "${CHATGPT_MODEL}" \
  --reasoning_effort "${CHATGPT_REASONING_EFFORT}" \
  --samples_per_task "${CHATGPT_SAMPLES_PER_TASK}" \
  --max_output_tokens "${CHATGPT_MAX_OUTPUT_TOKENS:-8192}" \
  --max_output_tokens_ceiling \
    "${CHATGPT_MAX_OUTPUT_TOKENS_CEILING:-12288}" \
  --max_prompt_tokens "${CHATGPT_MAX_PROMPT_TOKENS:-12000}" \
  --workers "${CHATGPT_WORKERS:-24}" \
  --api_timeout "${CHATGPT_TIMEOUT:-600}" \
  --api_retries "${CHATGPT_API_RETRIES:-4}" \
  --stability_runs "${CHATGPT_VERIFY_STABILITY_RUNS:-2}" \
  --min_verified_tasks "${HARVEST_VERIFIED_TASKS}" \
  --include_student_candidate \
  --limit "${LIMIT}"

printf 'Official synchronous GPT-5.6-sol executable-fit RS corpus ready: %s harvest_target=%s downstream_floor=%s\n' \
  "${OUTPUT_ROOT}/verified_repairs.jsonl" "${HARVEST_VERIFIED_TASKS}" \
  "${RS_RECERTIFICATION_FLOOR}"
