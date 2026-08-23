#!/usr/bin/env bash
# Offline-harvested Qwen sequence distillation -> direct-compact warm-start SFT.
#
# The only network phase is collect_qwen_direct_compact_teacher.py. The teacher
# is never invoked from the gradient loop. Do not put API keys on this command
# line; export QWEN_API_KEY (or set API_KEY_ENV) in the process environment.
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
QWEN_ENV_FILE="${QWEN_ENV_FILE:-/workspace/Qwen.env}"
if [[ -f "${QWEN_ENV_FILE}" ]]; then
  # Parse credentials as data, never shell source. This accepts CRLF-safe
  # KEY=VALUE lines for an explicit allowlist and rejects command
  # substitutions, exports, and unrelated configuration.
  declare -A qwen_env_seen=()
  while IFS= read -r qwen_env_line || [[ -n "${qwen_env_line}" ]]; do
    qwen_env_line="${qwen_env_line%$'\r'}"
    if [[ -z "${qwen_env_line}" || "${qwen_env_line}" == \#* ]]; then
      continue
    fi
    if [[ ! "${qwen_env_line}" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
      printf 'Qwen.env contains a non-KEY=VALUE line\n' >&2
      exit 2
    fi
    qwen_env_key="${BASH_REMATCH[1]}"
    qwen_env_value="${BASH_REMATCH[2]}"
    case "${qwen_env_key}" in
      API_KEY|QWEN_API_KEY|DASHSCOPE_ENDPOINT|QWEN_BASE_URL|QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED)
        ;;
      *)
        printf 'Qwen.env contains disallowed key: %s\n' \
          "${qwen_env_key}" >&2
        exit 2
        ;;
    esac
    if [[ -n "${qwen_env_seen[${qwen_env_key}]:-}" ]]; then
      printf 'Qwen.env contains duplicate key: %s\n' \
        "${qwen_env_key}" >&2
      exit 2
    fi
    qwen_env_seen["${qwen_env_key}"]=1
    if [[ "${qwen_env_value}" == *'$('* \
       || "${qwen_env_value}" == *'`'* \
       || "${qwen_env_value}" == *';'* ]]; then
      printf 'Qwen.env value for %s contains shell syntax\n' \
        "${qwen_env_key}" >&2
      exit 2
    fi
    printf -v "${qwen_env_key}" '%s' "${qwen_env_value}"
    export "${qwen_env_key}"
  done < "${QWEN_ENV_FILE}"
fi
if [[ -z "${QWEN_API_KEY:-}" && -n "${API_KEY:-}" ]]; then
  export QWEN_API_KEY="${API_KEY}"
fi
QWEN_TEACHER_MODEL="${QWEN_TEACHER_MODEL:-qwen3.8-max-preview}"
QWEN_OBJECTIVE_MODE="${QWEN_OBJECTIVE_MODE:-sequence_only}"
QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED="${QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED:-0}"
QWEN_ORPHAN_REISSUE_AUTHORIZED="${QWEN_ORPHAN_REISSUE_AUTHORIZED:-0}"
QWEN_GOLD_ONLY="${QWEN_GOLD_ONLY:-0}"
QWEN_DEFER_STUDENT_PREP="${QWEN_DEFER_STUDENT_PREP:-0}"
case "${QWEN_GOLD_ONLY,,}" in
  0|false|no) QWEN_GOLD_ONLY=0 ;;
  1|true|yes) QWEN_GOLD_ONLY=1 ;;
  *)
    printf 'QWEN_GOLD_ONLY must be 0/1, false/true, or no/yes\n' >&2
    exit 2
    ;;
esac
case "${QWEN_DEFER_STUDENT_PREP,,}" in
  0|false|no) QWEN_DEFER_STUDENT_PREP=0 ;;
  1|true|yes) QWEN_DEFER_STUDENT_PREP=1 ;;
  *)
    printf 'QWEN_DEFER_STUDENT_PREP must be 0/1, false/true, or no/yes\n' >&2
    exit 2
    ;;
esac
case "${QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED,,}" in
  1|true|yes)
    TOKEN_PLAN_AUTH_ARGS=(--token-plan-automation-authorized)
    ;;
  0|false|no)
    TOKEN_PLAN_AUTH_ARGS=()
    ;;
  *)
    printf 'QWEN_TOKEN_PLAN_AUTOMATION_AUTHORIZED must be 0/1, false/true, or no/yes\n' >&2
    exit 2
    ;;
esac
case "${QWEN_ORPHAN_REISSUE_AUTHORIZED,,}" in
  1|true|yes)
    if (( ${#TOKEN_PLAN_AUTH_ARGS[@]} == 0 )); then
      printf 'QWEN_ORPHAN_REISSUE_AUTHORIZED requires explicit Token Plan automation authorization\n' >&2
      exit 2
    fi
    ORPHAN_REISSUE_ARGS=(
      --authorize-orphan-reissue-with-duplicate-billing-risk
    )
    ;;
  0|false|no)
    ORPHAN_REISSUE_ARGS=()
    ;;
  *)
    printf 'QWEN_ORPHAN_REISSUE_AUTHORIZED must be 0/1, false/true, or no/yes\n' >&2
    exit 2
    ;;
esac
if [[ -z "${QWEN_ENABLE_THINKING:-}" ]]; then
  if [[ "${QWEN_OBJECTIVE_MODE}" == "require_top5" ]]; then
    QWEN_ENABLE_THINKING=0
  else
    QWEN_ENABLE_THINKING=1
  fi
fi
case "${QWEN_ENABLE_THINKING,,}" in
  1|true|yes)
    THINKING_ARGS=(--enable-thinking)
    ;;
  0|false|no)
    THINKING_ARGS=(--no-enable-thinking)
    ;;
  *)
    printf 'QWEN_ENABLE_THINKING must be 0/1, false/true, or no/yes\n' >&2
    exit 2
    ;;
esac
API_KEY_ENV="QWEN_API_KEY"
QWEN_BASE_URL="${QWEN_BASE_URL:-${DASHSCOPE_ENDPOINT:-}}"
COLLECTION_BASE_URL="${QWEN_BASE_URL}"
if [[ "${QWEN_GOLD_ONLY}" == "1" ]]; then
  # The collector runs only its no-network artifact preflight in this mode.
  # Use an approved syntactic placeholder so a Token Plan credential is never
  # contacted while the independent GPU gold initialization proceeds.
  COLLECTION_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
fi
REQUIRED_FUNCTION="${REQUIRED_FUNCTION:-fn0}"
GOLD_REPLAY_FRACTION="${GOLD_REPLAY_FRACTION:-0.0}"
SEED="${SEED:-44}"

: "${PROMPT_JSONL:?set PROMPT_JSONL}"
: "${PROMPT_SHA256:?set PROMPT_SHA256}"
: "${PROMPT_ROWS:?set PROMPT_ROWS}"
: "${PROMPT_MANIFEST:?set PROMPT_MANIFEST}"
: "${PROMPT_MANIFEST_SHA256:?set PROMPT_MANIFEST_SHA256}"
: "${VERIFIER_JSONL:?set VERIFIER_JSONL}"
: "${VERIFIER_SHA256:?set VERIFIER_SHA256}"
: "${STUDENT_TOKENIZER_JSON:?set STUDENT_TOKENIZER_JSON}"
: "${STUDENT_TOKENIZER_SHA256:?set STUDENT_TOKENIZER_SHA256}"
: "${STUDENT_EOS_TOKEN_ID:?set STUDENT_EOS_TOKEN_ID}"
: "${COMPACT_TRAIN_JSONL:?set COMPACT_TRAIN_JSONL}"
: "${COMPACT_TRAIN_SEAL:?set COMPACT_TRAIN_SEAL}"
: "${COMPACT_CONTRACT:?set COMPACT_CONTRACT}"
: "${COMPACT_CODEBOOK:?set COMPACT_CODEBOOK}"
: "${COMPACT_CODEC_ARTIFACT:?set COMPACT_CODEC_ARTIFACT}"
: "${DIRECT_COMPACT_WARMSTART:?set DIRECT_COMPACT_WARMSTART}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${QWEN_BASE_URL:?set QWEN_BASE_URL or DASHSCOPE_ENDPOINT}"
if [[ "${QWEN_OBJECTIVE_MODE}" == "sequence_only" \
   && "${GOLD_REPLAY_FRACTION}" != "0" \
   && "${GOLD_REPLAY_FRACTION}" != "0.0" ]]; then
  printf 'sequence_only requires GOLD_REPLAY_FRACTION=0.0; gold adaptation is already a separate initialization stage\n' >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
JOURNAL="${OUTPUT_ROOT}/qwen_teacher.journal.jsonl"
PARSEABLE="${OUTPUT_ROOT}/qwen_teacher.parseable.mc_sequence.jsonl"
VERIFIED_RS="${OUTPUT_ROOT}/qwen_teacher.verified_only.rs_sft.jsonl"
AUDIT="${OUTPUT_ROOT}/qwen_teacher.audit.json"
CONTRACT_PROBE="${OUTPUT_ROOT}/qwen_teacher.contract_probe.json"
TRAIN_JSONL="${OUTPUT_ROOT}/qwen_mc_sequence_train.jsonl"
TRAIN_SEAL="${OUTPUT_ROOT}/qwen_mc_sequence_train.seal.json"
SCHEDULE="${OUTPUT_ROOT}/qwen_mc_sequence_train.schedule.jsonl"
BUILD_MANIFEST="${OUTPUT_ROOT}/qwen_mc_sequence_train.build.json"
TRAIN_OUTPUT="${OUTPUT_ROOT}/direct_compact_qwen_sequence_warmstart"
PRE_V2_DIRECT_COMPACT_WARMSTART="${DIRECT_COMPACT_WARMSTART}"
MIGRATED_DIRECT_COMPACT_WARMSTART="${MIGRATED_DIRECT_COMPACT_WARMSTART:-${OUTPUT_ROOT}/direct_compact_inline_cfg_v2_migrated_base}"
BASE_DIRECT_COMPACT_WARMSTART="${MIGRATED_DIRECT_COMPACT_WARMSTART}"
GOLD_ADAPT_OUTPUT="${GOLD_ADAPT_OUTPUT:-${OUTPUT_ROOT}/direct_compact_multifunction_gold_sft}"
GOLD_ADAPT_LOG="${OUTPUT_ROOT}/multifunction_gold_sft.log"
GOLD_ADAPT_TRAIN_ROWS="${GOLD_ADAPT_TRAIN_ROWS:-${PROMPT_ROWS}}"
CAPACITY_MIGRATED_GOLD_WARMSTART="${CAPACITY_MIGRATED_GOLD_WARMSTART:-}"
CAPACITY_MIGRATION_SOURCE_GOLD="${CAPACITY_MIGRATION_SOURCE_GOLD:-}"

cd "${PATCH_ROOT}"

GOLD_ADAPT_PID=""

overlay_migration_common_args() {
  OVERLAY_MIGRATION_ARGS=(
    --output_dir "${MIGRATED_DIRECT_COMPACT_WARMSTART}"
    --contract "${COMPACT_CONTRACT}"
    --codebook "${COMPACT_CODEBOOK}"
    --codec_artifact "${COMPACT_CODEC_ARTIFACT}"
    --tokenizer_json "${STUDENT_TOKENIZER_JSON}"
    --warmstart_checkpoint "${PRE_V2_DIRECT_COMPACT_WARMSTART}"
  )
  if [[ -n "${DECODER_MODEL:-}" ]]; then
    OVERLAY_MIGRATION_ARGS+=(--decoder_model "${DECODER_MODEL}")
  fi
  if [[ -n "${DECODER_REVISION:-}" ]]; then
    OVERLAY_MIGRATION_ARGS+=(--decoder_revision "${DECODER_REVISION}")
  fi
}

validate_overlay_migration() {
  overlay_migration_common_args
  "${PYTHON_BIN}" -m scripts.training.direct_compact_qwen_decompiler \
    "${OVERLAY_MIGRATION_ARGS[@]}" \
    --validate_migrated_warmstart_only
}

validate_capacity_migrated_gold() {
  : "${CAPACITY_MIGRATED_GOLD_WARMSTART:?set capacity-migrated gold checkpoint}"
  : "${CAPACITY_MIGRATION_SOURCE_GOLD:?set source gold checkpoint}"
  "${PYTHON_BIN}" -m scripts.training.direct_compact_qwen_decompiler \
    --output_dir "${CAPACITY_MIGRATED_GOLD_WARMSTART}" \
    --contract "${COMPACT_CONTRACT}" \
    --codebook "${COMPACT_CODEBOOK}" \
    --codec_artifact "${COMPACT_CODEC_ARTIFACT}" \
    --tokenizer_json "${STUDENT_TOKENIZER_JSON}" \
    --warmstart_checkpoint "${CAPACITY_MIGRATION_SOURCE_GOLD}" \
    --validate_migrated_warmstart_only
  export DIRECT_COMPACT_WARMSTART="${CAPACITY_MIGRATED_GOLD_WARMSTART}"
  printf 'CAPACITY_MIGRATED_GOLD_READY checkpoint=%s source=%s\n' \
    "${DIRECT_COMPACT_WARMSTART}" "${CAPACITY_MIGRATION_SOURCE_GOLD}"
}

prepare_overlay_migration() {
  if [[ -e "${MIGRATED_DIRECT_COMPACT_WARMSTART}" ]]; then
    validate_overlay_migration
    printf 'DIRECT_COMPACT_INLINE_CFG_V2_MIGRATION_REUSE checkpoint=%s\n' \
      "${MIGRATED_DIRECT_COMPACT_WARMSTART}"
    return
  fi
  overlay_migration_common_args
  "${PYTHON_BIN}" -m scripts.training.direct_compact_qwen_decompiler \
    "${OVERLAY_MIGRATION_ARGS[@]}" \
    --migrate_warmstart_only \
    --attn_implementation eager \
    --bf16
  validate_overlay_migration
  printf 'DIRECT_COMPACT_INLINE_CFG_V2_MIGRATION_READY source=%s checkpoint=%s receipt=%s\n' \
    "${PRE_V2_DIRECT_COMPACT_WARMSTART}" \
    "${MIGRATED_DIRECT_COMPACT_WARMSTART}" \
    "${MIGRATED_DIRECT_COMPACT_WARMSTART}/overlay_migration_receipt.json"
}

validate_gold_adapt() {
  "${PYTHON_BIN}" \
    -m scripts.evaluation.validate_direct_compact_training_stage \
    --checkpoint "${GOLD_ADAPT_OUTPUT}" \
    --contract "${COMPACT_CONTRACT}" \
    --train-file "${COMPACT_TRAIN_JSONL}" \
    --train-seal "${COMPACT_TRAIN_SEAL}" \
    --expected-train-rows "${GOLD_ADAPT_TRAIN_ROWS}" \
    --no-eval-during-training \
    --loss-mode token_mean \
    --base-warmstart "${BASE_DIRECT_COMPACT_WARMSTART}"
}

gold_exit_handler() {
  local status=$?
  local gold_status=0
  trap - EXIT
  if [[ -n "${GOLD_ADAPT_PID}" ]]; then
    printf 'Waiting for useful multi-function gold adaptation after pipeline exit=%s\n' \
      "${status}" >&2
    set +e
    wait "${GOLD_ADAPT_PID}"
    gold_status=$?
    set -e
    GOLD_ADAPT_PID=""
    if (( status == 0 && gold_status != 0 )); then
      status=5
    fi
  fi
  exit "${status}"
}

terminate_gold_adapt() {
  trap - INT TERM
  if [[ -n "${GOLD_ADAPT_PID}" ]] && kill -0 "${GOLD_ADAPT_PID}" 2>/dev/null; then
    kill "${GOLD_ADAPT_PID}" 2>/dev/null || true
    wait "${GOLD_ADAPT_PID}" 2>/dev/null || true
    GOLD_ADAPT_PID=""
  fi
  exit 130
}

trap gold_exit_handler EXIT
trap terminate_gold_adapt INT TERM

start_gold_adapt() {
  if [[ -e "${GOLD_ADAPT_OUTPUT}" ]]; then
    if validate_gold_adapt; then
      printf 'MULTIFUNCTION_GOLD_SFT_REUSE checkpoint=%s\n' \
        "${GOLD_ADAPT_OUTPUT}"
      return
    fi
    printf 'MULTIFUNCTION_GOLD_SFT_RESUME output=%s\n' \
      "${GOLD_ADAPT_OUTPUT}"
  fi
  local gold_args=(
    --train_file "${COMPACT_TRAIN_JSONL}"
    --train_seal "${COMPACT_TRAIN_SEAL}"
    --no_eval_during_training
    --output_dir "${GOLD_ADAPT_OUTPUT}"
    --contract "${COMPACT_CONTRACT}"
    --codebook "${COMPACT_CODEBOOK}"
    --codec_artifact "${COMPACT_CODEC_ARTIFACT}"
    --tokenizer_json "${STUDENT_TOKENIZER_JSON}"
    --warmstart_checkpoint "${BASE_DIRECT_COMPACT_WARMSTART}"
    --learning_rate "${GOLD_ADAPT_LEARNING_RATE:-2e-5}"
    --epochs "${GOLD_ADAPT_EPOCHS:-1.0}"
    --batch_size "${GOLD_ADAPT_BATCH_SIZE:-1}"
    --grad_accum "${GOLD_ADAPT_GRAD_ACCUM:-16}"
    --eval_strategy no
    --seed "${SEED}"
    --gradient_checkpointing
    --save_steps "${GOLD_ADAPT_SAVE_STEPS:-25}"
    --bf16
  )
  if [[ -d "${GOLD_ADAPT_OUTPUT}" ]]; then
    gold_args+=(--resume_from_checkpoint auto)
  fi
  if [[ -n "${DECODER_MODEL:-}" ]]; then
    gold_args+=(--decoder_model "${DECODER_MODEL}")
  fi
  if [[ -n "${DECODER_REVISION:-}" ]]; then
    gold_args+=(--decoder_revision "${DECODER_REVISION}")
  fi
  "${PYTHON_BIN}" -m scripts.training.direct_compact_qwen_decompiler \
    "${gold_args[@]}" >"${GOLD_ADAPT_LOG}" 2>&1 &
  GOLD_ADAPT_PID=$!
  printf 'MULTIFUNCTION_GOLD_SFT_STARTED pid=%s rows=%s log=%s\n' \
    "${GOLD_ADAPT_PID}" "${GOLD_ADAPT_TRAIN_ROWS}" "${GOLD_ADAPT_LOG}"
}

validate_sequence_train() {
  local expected_rows
  expected_rows="$(jq -r '.counts.output_rows' "${BUILD_MANIFEST}")"
  "${PYTHON_BIN}" \
    -m scripts.evaluation.validate_direct_compact_training_stage \
    --checkpoint "${TRAIN_OUTPUT}" \
    --contract "${COMPACT_CONTRACT}" \
    --train-file "${TRAIN_JSONL}" \
    --train-seal "${TRAIN_SEAL}" \
    --expected-train-rows "${expected_rows}" \
    --no-eval-during-training \
    --loss-mode sequence_sum \
    --base-warmstart "${DIRECT_COMPACT_WARMSTART}"
}

finish_gold_adapt() {
  local gold_status=0
  if [[ -n "${GOLD_ADAPT_PID}" ]]; then
    set +e
    wait "${GOLD_ADAPT_PID}"
    gold_status=$?
    set -e
    GOLD_ADAPT_PID=""
    if (( gold_status != 0 )); then
      printf 'Multi-function gold adaptation failed; see %s\n' \
        "${GOLD_ADAPT_LOG}" >&2
      exit 5
    fi
  fi
  validate_gold_adapt
  export DIRECT_COMPACT_WARMSTART="${GOLD_ADAPT_OUTPUT}"
  printf 'MULTIFUNCTION_GOLD_SFT_READY checkpoint=%s\n' \
    "${DIRECT_COMPACT_WARMSTART}"
}

check_gold_adapt_before_full_harvest() {
  local gold_status=0
  if [[ -n "${GOLD_ADAPT_PID}" ]] \
    && ! kill -0 "${GOLD_ADAPT_PID}" 2>/dev/null; then
    set +e
    wait "${GOLD_ADAPT_PID}"
    gold_status=$?
    set -e
    GOLD_ADAPT_PID=""
    if (( gold_status != 0 )); then
      printf 'Gold adaptation failed before the full paid Qwen harvest; see %s\n' \
        "${GOLD_ADAPT_LOG}" >&2
      exit 5
    fi
    validate_gold_adapt
    export DIRECT_COMPACT_WARMSTART="${GOLD_ADAPT_OUTPUT}"
  fi
}

COLLECT_COMMON_ARGS=(
  "${TOKEN_PLAN_AUTH_ARGS[@]}"
  "${ORPHAN_REISSUE_ARGS[@]}"
  --prompt-jsonl "${PROMPT_JSONL}"
  --expected-prompt-sha256 "${PROMPT_SHA256}"
  --prompt-manifest "${PROMPT_MANIFEST}"
  --expected-prompt-manifest-sha256 "${PROMPT_MANIFEST_SHA256}"
  --verifier-jsonl "${VERIFIER_JSONL}"
  --expected-verifier-sha256 "${VERIFIER_SHA256}"
  --student-tokenizer-json "${STUDENT_TOKENIZER_JSON}"
  --expected-student-tokenizer-sha256 "${STUDENT_TOKENIZER_SHA256}"
  --student-eos-token-id "${STUDENT_EOS_TOKEN_ID}"
  --target-contract "${COMPACT_CONTRACT}"
  --expected-target-contract-sha256 "$(sha256sum "${COMPACT_CONTRACT}" | awk '{print $1}')"
  --model "${QWEN_TEACHER_MODEL}"
  --objective-mode "${QWEN_OBJECTIVE_MODE}"
  "${THINKING_ARGS[@]}"
  --required-function "${REQUIRED_FUNCTION}"
  --base-url "${COLLECTION_BASE_URL}"
  --api-key-env "${API_KEY_ENV}"
  --temperature "${TEACHER_TEMPERATURE:-1.0}"
  --top-p "${TEACHER_TOP_P:-1.0}"
  --top-k "${TEACHER_TOP_K:-101}"
  --max-tokens "${TEACHER_MAX_TOKENS:-12288}"
  --length-max-token-escalation 16384 24576
  --thinking-budget "${TEACHER_THINKING_BUDGET:-8192}"
  --seed-base "${TEACHER_SEED_BASE:-44}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-12000}"
  --chat-overhead-reserve "${CHAT_OVERHEAD_RESERVE:-256}"
  --timeout-seconds "${TEACHER_TIMEOUT_SECONDS:-600}"
  --verifier-timeout-seconds "${VERIFIER_TIMEOUT_SECONDS:-45}"
  --max-retries "${TEACHER_MAX_RETRIES:-8}"
  --workers "${QWEN_TEACHER_WORKERS:-16}"
  --verifier-workers "${QWEN_VERIFIER_WORKERS:-16}"
  --progress-every "${QWEN_PROGRESS_EVERY:-50}"
)
if [[ -n "${PROMPT_ROWS:-}" ]]; then
  COLLECT_COMMON_ARGS+=(--expected-prompt-rows "${PROMPT_ROWS}")
fi
COLLECT_ARGS=(
  "${COLLECT_COMMON_ARGS[@]}"
  --journal "${JOURNAL}"
  --parseable-output "${PARSEABLE}"
  --rs-sft-output "${VERIFIED_RS}"
  --audit-output "${AUDIT}"
)
PREFLIGHT_ARGS=("${COLLECT_ARGS[@]}" --dry-run)
"${PYTHON_BIN}" -m scripts.training.collect_qwen_direct_compact_teacher \
  "${PREFLIGHT_ARGS[@]}"
if [[ "${QWEN_COLLECT_DRY_RUN:-0}" == "1" ]]; then
  printf 'QWEN_SEQUENCE_PREFLIGHT_COMPLETE model=%s objective_mode=%s prompts=%s\n' \
    "${QWEN_TEACHER_MODEL}" "${QWEN_OBJECTIVE_MODE}" "${PROMPT_JSONL}"
  exit 0
fi

# Gold-only adaptation is train1580-only and independent of all Qwen API
# responses.  Start it immediately after the no-network contract preflight so
# the GPU overlaps the paid probe, pilot, and full K=8 harvest.  The final
# sequence stage still joins only after both artifacts are complete.
if [[ "${QWEN_DEFER_STUDENT_PREP}" == "0" \
   && -z "${CAPACITY_MIGRATED_GOLD_WARMSTART}" ]]; then
  prepare_overlay_migration
  start_gold_adapt
fi
if [[ "${QWEN_GOLD_ONLY}" == "1" ]]; then
  finish_gold_adapt
  printf 'QWEN_GOLD_ONLY_COMPLETE checkpoint=%s network_calls=0\n' \
    "${DIRECT_COMPACT_WARMSTART}"
  exit 0
fi

# One bounded synchronous call proves that this exact model+endpoint accepts
# the requested contract before a new K=8 fan-out can spend meaningful quota.
# A nonempty production journal is itself stronger evidence: it is hash-bound
# to the exact endpoint/model/prompt contract and contains durable provider
# responses.  Re-probing on every crash-recovery would waste a paid draw and
# can itself fail transiently, so journal resumes skip the probe.
if [[ -s "${JOURNAL}" ]]; then
  printf 'QWEN_CONTRACT_PROBE_SKIP reason=production_journal_resume journal=%s\n' \
    "${JOURNAL}"
else
  PROBE_TASK_ARGS=()
  if [[ -n "${QWEN_PROBE_TASK_ID:-}" ]]; then
    PROBE_TASK_ARGS+=(--task-id "${QWEN_PROBE_TASK_ID}")
  fi
  "${PYTHON_BIN}" -m scripts.training.probe_qwen_teacher_contract \
    --prompt-jsonl "${PROMPT_JSONL}" \
    --expected-prompt-sha256 "${PROMPT_SHA256}" \
    --expected-prompt-rows "${PROMPT_ROWS}" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --expected-prompt-manifest-sha256 "${PROMPT_MANIFEST_SHA256}" \
    --output "${CONTRACT_PROBE}" \
    --model "${QWEN_TEACHER_MODEL}" \
    --objective-mode "${QWEN_OBJECTIVE_MODE}" \
    "${THINKING_ARGS[@]}" \
    "${TOKEN_PLAN_AUTH_ARGS[@]}" \
    --base-url "${QWEN_BASE_URL}" \
    --api-key-env "${API_KEY_ENV}" \
    --student-tokenizer-json "${STUDENT_TOKENIZER_JSON}" \
    --expected-student-tokenizer-sha256 "${STUDENT_TOKENIZER_SHA256}" \
    --student-eos-token-id "${STUDENT_EOS_TOKEN_ID}" \
    --timeout-seconds "${QWEN_PROBE_TIMEOUT_SECONDS:-120}" \
    --max-tokens "${QWEN_PROBE_MAX_TOKENS:-256}" \
    "${PROBE_TASK_ARGS[@]}"
fi

# A small, production-parameter K=8 pilot must demonstrate that the teacher
# can recover this representation before the 1,580-task paid fan-out starts.
# It uses a separate journal so a failed pilot cannot contaminate production.
QWEN_PILOT_TASKS="${QWEN_PILOT_TASKS:-16}"
QWEN_PILOT_MIN_VERIFIED_TASKS="${QWEN_PILOT_MIN_VERIFIED_TASKS:-1}"
QWEN_PILOT_MIN_PARSEABLE_FRACTION="${QWEN_PILOT_MIN_PARSEABLE_FRACTION:-0.50}"
if ((
  QWEN_PILOT_TASKS < 1
  || QWEN_PILOT_TASKS > PROMPT_ROWS
  || QWEN_PILOT_MIN_VERIFIED_TASKS < 1
  || QWEN_PILOT_MIN_VERIFIED_TASKS > QWEN_PILOT_TASKS
)); then
  printf 'Invalid Qwen pilot task/verified-task thresholds\n' >&2
  exit 2
fi
if ! [[ "${QWEN_PILOT_MIN_PARSEABLE_FRACTION}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
  printf 'QWEN_PILOT_MIN_PARSEABLE_FRACTION must be in [0,1]\n' >&2
  exit 2
fi
PILOT_ROOT="${OUTPUT_ROOT}/quality_pilot"
PILOT_JOURNAL="${PILOT_ROOT}/qwen_teacher.pilot.journal.jsonl"
PILOT_SEQUENCE="${PILOT_ROOT}/qwen_teacher.pilot.sequence.jsonl"
PILOT_VERIFIED="${PILOT_ROOT}/qwen_teacher.pilot.verified_only.jsonl"
PILOT_AUDIT="${PILOT_ROOT}/qwen_teacher.pilot.audit.json"
PILOT_GATE="${PILOT_ROOT}/qwen_teacher.pilot.quality_gate.json"
mkdir -p "${PILOT_ROOT}"
PILOT_ARGS=(
  "${COLLECT_COMMON_ARGS[@]}"
  --journal "${PILOT_JOURNAL}"
  --parseable-output "${PILOT_SEQUENCE}"
  --rs-sft-output "${PILOT_VERIFIED}"
  --audit-output "${PILOT_AUDIT}"
  --max-tasks "${QWEN_PILOT_TASKS}"
  --task-selection-strategy deterministic_hash
)
if [[ -s "${JOURNAL}" ]]; then
  # The production run header sealed the exact pilot gate that authorized this
  # paid fan-out.  Re-materializing the completed pilot changes its audit
  # creation timestamp and therefore its hash, even though every response and
  # metric is unchanged.  Reuse and verify the originally sealed gate instead
  # of mutating it during crash recovery.
  PILOT_GATE_SHA256="$(
    "${PYTHON_BIN}" - "${JOURNAL}" "${PILOT_GATE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

journal = Path(sys.argv[1])
gate = Path(sys.argv[2])
with journal.open("r", encoding="utf-8") as handle:
    header = json.loads(next(handle))
record = (header.get("payload") or {}).get("pilot_quality_gate") or {}
expected_path = str(record.get("path") or "")
expected_sha256 = str(record.get("sha256") or "")
expected_size = int(record.get("size_bytes", -1))
if expected_path != str(gate):
    raise SystemExit("production journal pilot-gate path differs")
payload = gate.read_bytes()
if len(payload) != expected_size:
    raise SystemExit("production journal pilot-gate size differs")
if hashlib.sha256(payload).hexdigest() != expected_sha256:
    raise SystemExit("production journal pilot-gate hash differs")
print(expected_sha256)
PY
  )"
  PILOT_CANDIDATES="$(jq -er '.candidates' "${PILOT_GATE}")"
  PILOT_PARSEABLE="$(jq -er '.parseable_candidates' "${PILOT_GATE}")"
  PILOT_VERIFIED_TASKS="$(jq -er '.verified_tasks' "${PILOT_GATE}")"
  PILOT_MIN_UNIQUE="$(
    jq -er '.sampling_diversity.minimum_unique_final_sequences_per_task' \
      "${PILOT_GATE}"
  )"
  PILOT_MAX_UNIQUE="$(
    jq -er '.sampling_diversity.maximum_unique_final_sequences_per_task' \
      "${PILOT_GATE}"
  )"
  PILOT_IDENTICAL_TASKS="$(
    jq -er '.sampling_diversity.tasks_with_all_k8_draws_identical' \
      "${PILOT_GATE}"
  )"
  printf 'QWEN_QUALITY_PILOT_REUSE gate_sha256=%s reason=production_journal_resume\n' \
    "${PILOT_GATE_SHA256}"
else
  "${PYTHON_BIN}" -m scripts.training.collect_qwen_direct_compact_teacher \
    "${PILOT_ARGS[@]}"

  PILOT_VERIFIED_TASKS="$(
    jq -s '[.[].task_id] | unique | length' "${PILOT_VERIFIED}"
  )"
  if (( PILOT_VERIFIED_TASKS < QWEN_PILOT_MIN_VERIFIED_TASKS )); then
    printf 'Qwen quality pilot failed: verified tasks %s < %s; full harvest blocked\n' \
      "${PILOT_VERIFIED_TASKS}" "${QWEN_PILOT_MIN_VERIFIED_TASKS}" >&2
    exit 4
  fi
  if ! jq -e \
  --argjson minimum "${QWEN_PILOT_MIN_PARSEABLE_FRACTION}" \
  '.coverage.candidates > 0
   and (.coverage.parseable_candidates / .coverage.candidates) >= $minimum
   and .target_length_gate.passed == true
   and .target_length_gate.overflow_count == 0
   and .target_length_gate.targets_checked == .coverage.candidates' \
    "${PILOT_AUDIT}" >/dev/null; then
    printf 'Qwen quality pilot failed parseable-fraction/target-length gate; full harvest blocked\n' >&2
    exit 4
  fi
  if ! jq -e \
  --argjson pilot_tasks "${QWEN_PILOT_TASKS}" \
  '.sampling.pathological_all_tasks_have_identical_k8_draws == false
   and (.sampling.unique_final_sequences_per_task | length) == $pilot_tasks
   and ([
     .sampling.unique_final_sequences_per_task[]
     | select(. < 1 or . > 8)
   ] | length) == 0' \
    "${PILOT_AUDIT}" >/dev/null; then
    printf 'Qwen quality pilot failed sampled-sequence diversity contract; full harvest blocked\n' >&2
    exit 4
  fi
  PILOT_CANDIDATES="$(jq -r '.coverage.candidates' "${PILOT_AUDIT}")"
  PILOT_PARSEABLE="$(jq -r '.coverage.parseable_candidates' "${PILOT_AUDIT}")"
  PILOT_PARSEABLE_FRACTION="$(
    jq -r '.coverage.parseable_candidates / .coverage.candidates' \
      "${PILOT_AUDIT}"
  )"
  PILOT_NON_CODE_TARGETS="$(
    jq -r '.target_length_gate.non_code_target_count' "${PILOT_AUDIT}"
  )"
  PILOT_UNIQUE_BY_TASK="$(
    jq -c '.sampling.unique_final_sequences_per_task' "${PILOT_AUDIT}"
  )"
  PILOT_IDENTICAL_TASKS="$(
    jq -r '.sampling.tasks_with_all_k8_draws_identical | length' "${PILOT_AUDIT}"
  )"
  PILOT_MIN_UNIQUE="$(
    jq -r '.sampling.minimum_unique_final_sequences_per_task' "${PILOT_AUDIT}"
  )"
  PILOT_MAX_UNIQUE="$(
    jq -r '.sampling.maximum_unique_final_sequences_per_task' "${PILOT_AUDIT}"
  )"
  PILOT_TARGET_EVIDENCE_SHA256="$(
    jq -r '.target_length_gate.evidence_sha256' "${PILOT_AUDIT}"
  )"
  PILOT_TARGET_CONTRACT_SHA256="$(
    jq -r '.target_length_gate.target_contract.trainer_contract.sha256' \
      "${PILOT_AUDIT}"
  )"
  PILOT_MAX_TARGET_TOKENS="$(
    jq -r '.target_length_gate.target_contract.max_target_tokens' \
      "${PILOT_AUDIT}"
  )"
  jq -n \
  --arg audit_sha256 "$(sha256sum "${PILOT_AUDIT}" | awk '{print $1}')" \
  --arg verified_sha256 "$(sha256sum "${PILOT_VERIFIED}" | awk '{print $1}')" \
  --argjson pilot_tasks "${QWEN_PILOT_TASKS}" \
  --argjson candidates "${PILOT_CANDIDATES}" \
  --argjson parseable_candidates "${PILOT_PARSEABLE}" \
  --argjson parseable_fraction "${PILOT_PARSEABLE_FRACTION}" \
  --argjson non_code_target_count "${PILOT_NON_CODE_TARGETS}" \
  --argjson verified_tasks "${PILOT_VERIFIED_TASKS}" \
  --argjson minimum_verified_tasks "${QWEN_PILOT_MIN_VERIFIED_TASKS}" \
  --argjson minimum_parseable_fraction \
    "${QWEN_PILOT_MIN_PARSEABLE_FRACTION}" \
  --argjson unique_final_sequences_per_task "${PILOT_UNIQUE_BY_TASK}" \
  --argjson tasks_with_all_k8_draws_identical "${PILOT_IDENTICAL_TASKS}" \
  --argjson minimum_unique_final_sequences_per_task "${PILOT_MIN_UNIQUE}" \
  --argjson maximum_unique_final_sequences_per_task "${PILOT_MAX_UNIQUE}" \
  --arg target_length_evidence_sha256 "${PILOT_TARGET_EVIDENCE_SHA256}" \
  --arg target_contract_sha256 "${PILOT_TARGET_CONTRACT_SHA256}" \
  --argjson max_target_tokens "${PILOT_MAX_TARGET_TOKENS}" \
  '{
    schema: "qwen-teacher-quality-gate-v1",
    passed: true,
    pilot_tasks: $pilot_tasks,
    candidates: $candidates,
    parseable_candidates: $parseable_candidates,
    parseable_fraction: $parseable_fraction,
    verified_tasks: $verified_tasks,
    minimum_verified_tasks: $minimum_verified_tasks,
    minimum_parseable_fraction: $minimum_parseable_fraction,
    sampling_diversity: {
      unique_final_sequences_per_task: $unique_final_sequences_per_task,
      tasks_with_all_k8_draws_identical: $tasks_with_all_k8_draws_identical,
      minimum_unique_final_sequences_per_task:
        $minimum_unique_final_sequences_per_task,
      maximum_unique_final_sequences_per_task:
        $maximum_unique_final_sequences_per_task,
      pathological_all_tasks_have_identical_k8_draws: false,
      duplicate_draws_filtered: false
    },
    target_length_gate: {
      passed: true,
      target_length_evidence_sha256: $target_length_evidence_sha256,
      target_contract_sha256: $target_contract_sha256,
      max_target_tokens: $max_target_tokens,
      overflow_count: 0,
      non_code_target_count: $non_code_target_count,
      final_dart_code_only_required: false,
      truncate: false,
      filter_draw: false,
      resample_draw: false
    },
    pilot_audit_sha256: $audit_sha256,
    pilot_verified_only_sha256: $verified_sha256
    }' > "${PILOT_GATE}"
  PILOT_GATE_SHA256="$(sha256sum "${PILOT_GATE}" | awk '{print $1}')"
fi
COLLECT_ARGS+=(
  --quality-gate-json "${PILOT_GATE}"
  --expected-quality-gate-sha256 "${PILOT_GATE_SHA256}"
)
printf 'QWEN_QUALITY_PILOT_PASS tasks=%s candidates=%s parseable=%s verified_tasks=%s\n' \
  "${QWEN_PILOT_TASKS}" "${PILOT_CANDIDATES}" "${PILOT_PARSEABLE}" \
  "${PILOT_VERIFIED_TASKS}"
printf 'QWEN_QUALITY_PILOT_DIVERSITY min_unique=%s max_unique=%s all_identical_tasks=%s provider_seed_honor_not_assumed=true\n' \
  "${PILOT_MIN_UNIQUE}" "${PILOT_MAX_UNIQUE}" "${PILOT_IDENTICAL_TASKS}"

# Do not fan out 12,640 paid calls behind a dead GPU initialization stage.
check_gold_adapt_before_full_harvest

"${PYTHON_BIN}" -m scripts.training.collect_qwen_direct_compact_teacher \
  "${COLLECT_ARGS[@]}"

# Re-materialize from the durable journal with no API client involved.
"${PYTHON_BIN}" -m scripts.evaluation.audit_qwen_direct_compact_teacher \
  --journal "${JOURNAL}" \
  --student-tokenizer-json "${STUDENT_TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "${STUDENT_TOKENIZER_SHA256}" \
  --student-eos-token-id "${STUDENT_EOS_TOKEN_ID}" \
  --parseable-output "${PARSEABLE}" \
  --rs-sft-output "${VERIFIED_RS}" \
  --audit-output "${AUDIT}"

JOURNAL_SHA256="$(sha256sum "${JOURNAL}" | awk '{print $1}')"
PARSEABLE_SHA256="$(sha256sum "${PARSEABLE}" | awk '{print $1}')"
AUDIT_SHA256="$(sha256sum "${AUDIT}" | awk '{print $1}')"

"${PYTHON_BIN}" -m scripts.training.build_qwen_sequence_kd \
  --compact-train-jsonl "${COMPACT_TRAIN_JSONL}" \
  --compact-train-seal "${COMPACT_TRAIN_SEAL}" \
  --contract "${COMPACT_CONTRACT}" \
  --prompt-jsonl "${PROMPT_JSONL}" \
  --expected-prompt-sha256 "${PROMPT_SHA256}" \
  --teacher-parseable-jsonl "${PARSEABLE}" \
  --expected-teacher-parseable-sha256 "${PARSEABLE_SHA256}" \
  --teacher-journal "${JOURNAL}" \
  --expected-teacher-journal-sha256 "${JOURNAL_SHA256}" \
  --teacher-audit-json "${AUDIT}" \
  --expected-teacher-audit-sha256 "${AUDIT_SHA256}" \
  --student-tokenizer-json "${STUDENT_TOKENIZER_JSON}" \
  --expected-student-tokenizer-sha256 "${STUDENT_TOKENIZER_SHA256}" \
  --output-jsonl "${TRAIN_JSONL}" \
  --output-seal "${TRAIN_SEAL}" \
  --schedule-output "${SCHEDULE}" \
  --build-manifest "${BUILD_MANIFEST}" \
  --gold-replay-fraction "${GOLD_REPLAY_FRACTION}" \
  --seed "${SEED}"

# The original 1,580-task harvest is now an immutable parent of the expanded
# 2,776-task fit run.  A durable sentinel lets the already-running collector
# finish and materialize its complete offline artifacts, but prevents it from
# spending GPU time on the obsolete parent-only sequence stage.  The expanded
# continuation launcher consumes this exact parent artifact later.
EXPANDED_FIT_SENTINEL="${OUTPUT_ROOT}/EXPAND_TO_FIT2776"
if [[ -f "${EXPANDED_FIT_SENTINEL}" ]]; then
  printf 'QWEN_SEQUENCE_PARENT_ONLY_TRAIN_SKIP sentinel=%s parent_tasks=%s expanded_fit_tasks=2776\n' \
    "${EXPANDED_FIT_SENTINEL}" "${PROMPT_ROWS}"
  exit 0
fi

# The final sequence stage cannot begin until the independently useful
# train1580-only gold adaptation has also completed and validated.
if [[ -n "${CAPACITY_MIGRATED_GOLD_WARMSTART}" ]]; then
  validate_capacity_migrated_gold
elif [[ "${QWEN_DEFER_STUDENT_PREP}" == "1" ]]; then
  prepare_overlay_migration
  start_gold_adapt
fi
if [[ -z "${CAPACITY_MIGRATED_GOLD_WARMSTART}" ]]; then
  finish_gold_adapt
fi

if [[ "${QWEN_OBJECTIVE_MODE}" == "require_top5" ]]; then
  # The audited top-5 path trains once from the original compact warm start on
  # primary sequence MC-NLL plus the coarsened top5+tail KL auxiliary. It is
  # never called dense/full-vocabulary KD.
  export SEQUENCE_TRAIN_JSONL="${TRAIN_JSONL}"
  export SEQUENCE_TRAIN_SEAL="${TRAIN_SEAL}"
  export SEQUENCE_SCHEDULE_JSONL="${SCHEDULE}"
  export SEQUENCE_BUILD_MANIFEST="${BUILD_MANIFEST}"
  export TEACHER_PARSEABLE_JSONL="${PARSEABLE}"
  export TEACHER_AUDIT_JSON="${AUDIT}"
  export QWEN_TRAIN_OUTPUT="${TRAIN_OUTPUT}"
  bash "${PATCH_ROOT}/scripts/run_qwen_sparse_topk_tail_warmstart.sh"
  printf 'QWEN_TOP5_SEQUENCE_WARMSTART_COMPLETE model=%s objective_mode=%s dense_kl=false output=%s\n' \
    "${QWEN_TEACHER_MODEL}" "${QWEN_OBJECTIVE_MODE}" "${TRAIN_OUTPUT}"
  exit 0
elif [[ "${QWEN_OBJECTIVE_MODE}" != "sequence_only" ]]; then
  printf 'Unsupported QWEN_OBJECTIVE_MODE: %s\n' "${QWEN_OBJECTIVE_MODE}" >&2
  exit 2
fi

# This is equal-draw, EOS-inclusive summed sequence NLL over the precomputed
# teacher samples. DIRECT_COMPACT_WARMSTART is the just-validated full-gold
# multi-function adaptation, never the old fn0-only checkpoint.
TRAIN_ARGS=(
  --train_file "${TRAIN_JSONL}"
  --train_seal "${TRAIN_SEAL}"
  --no_eval_during_training
  --output_dir "${TRAIN_OUTPUT}"
  --contract "${COMPACT_CONTRACT}"
  --codebook "${COMPACT_CODEBOOK}"
  --codec_artifact "${COMPACT_CODEC_ARTIFACT}"
  --tokenizer_json "${STUDENT_TOKENIZER_JSON}"
  --warmstart_checkpoint "${DIRECT_COMPACT_WARMSTART}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --epochs "${EPOCHS:-1.0}"
  --batch_size "${BATCH_SIZE:-1}"
  --grad_accum "${GRAD_ACCUM:-16}"
  --eval_strategy no
  --seed "${SEED}"
  --sequence_distribution_nll
  --gradient_checkpointing
  --save_steps "${SEQUENCE_SAVE_STEPS:-25}"
  --bf16
)
if [[ -n "${DECODER_MODEL:-}" ]]; then
  TRAIN_ARGS+=(--decoder_model "${DECODER_MODEL}")
fi
if [[ -n "${DECODER_REVISION:-}" ]]; then
  TRAIN_ARGS+=(--decoder_revision "${DECODER_REVISION}")
fi
if [[ -e "${TRAIN_OUTPUT}" ]] && validate_sequence_train; then
  printf 'QWEN_SEQUENCE_WARMSTART_REUSE output=%s\n' "${TRAIN_OUTPUT}"
else
  if [[ -d "${TRAIN_OUTPUT}" ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint auto)
  fi
  "${PYTHON_BIN}" -m scripts.training.direct_compact_qwen_decompiler \
    "${TRAIN_ARGS[@]}"
  validate_sequence_train
fi

printf 'QWEN_SEQUENCE_WARMSTART_COMPLETE model=%s objective_mode=%s dense_kl=false train=%s rs_sft_verified_only=%s output=%s\n' \
  "${QWEN_TEACHER_MODEL}" "${QWEN_OBJECTIVE_MODE}" "${TRAIN_JSONL}" \
  "${VERIFIED_RS}" "${TRAIN_OUTPUT}"
