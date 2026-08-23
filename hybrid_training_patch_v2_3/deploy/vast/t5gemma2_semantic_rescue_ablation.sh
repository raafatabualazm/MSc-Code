#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
SECRET_FILE="${T5GEMMA_ANTHROPIC_ENV:-${WORKSPACE}/secrets/Anthropic.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
F2_DECODER="${WORKSPACE}/frontier_ceiling_patch_v1/frontier_f2.py"
RESCUE_GROUP_COUNT="${T5GEMMA_SEMANTIC_RESCUE_GROUPS:-100}"
RUN_TAG="${T5GEMMA_SEMANTIC_RESCUE_RUN_TAG:-production100}"
OUTPUT_DIR="${T5GEMMA_SEMANTIC_RESCUE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_semantic_rescue_${RUN_TAG}_v1}"

PILOT_DIR="${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1"
MIDDLE_DIR="${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_1000x4_v1"
REMAINING_DIR="${WORKSPACE}/artifacts/t5gemma2_local_base_harvest_2epoch_remaining1186x4_v1"

if [[ ! "${RESCUE_GROUP_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "T5GEMMA_SEMANTIC_RESCUE_BLOCKED invalid group count" >&2
  exit 78
fi
if [[ ! -s "${SECRET_FILE}" ]] || [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_SEMANTIC_RESCUE_BLOCKED secret or Dart runtime is absent" >&2
  exit 78
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "T5GEMMA_SEMANTIC_RESCUE_BLOCKED frozen SFT checkpoint is absent" >&2
  exit 78
fi

# Keep the credential out of argv, logs, and persisted experiment artifacts.
anthropic_key="$(
  /venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import re
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if stat.S_IMODE(path.stat().st_mode) & 0o077:
    raise SystemExit("Anthropic.env must not be group/world accessible")
raw = path.read_bytes()
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("utf-16")
lines = [
    line.strip()
    for line in text.splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(lines) != 1:
    raise SystemExit("Anthropic.env must contain exactly one key line")
line = lines[0]
match = re.fullmatch(
    r"(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.*)", line, re.IGNORECASE
)
value = match.group(1).strip() if match else line
if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
    value = value[1:-1]
if not value or any(character.isspace() for character in value):
    raise SystemExit("Anthropic API key is empty or malformed")
print(value, end="")
PY
)"
export ANTHROPIC_API_KEY="${anthropic_key}"
unset anthropic_key

printf '%s  %s\n' \
  5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58 \
  "${PILOT_DIR}/harvest.journal.jsonl" \
  b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab \
  "${PILOT_DIR}/harvest_report.json" \
  80a326b6b2b2c8bdb0cd745f9884ace91baf411971023b1fed2d98192a022024 \
  "${MIDDLE_DIR}/harvest.journal.jsonl" \
  8783af49e7081d012ef6d3a6b3424818252cb6b8177a941873eb23874f9f8d50 \
  "${MIDDLE_DIR}/harvest_report.json" \
  680e9df0e05b39d1a7c41d9ebd50332d8ec59e87ce932d470853bc5c8eb6ace2 \
  "${REMAINING_DIR}/harvest.journal.jsonl" \
  883aeeab6d6a62d4bab41c62f77c8578866cf1a3b9355c4fd74e1de06c048cae \
  "${REMAINING_DIR}/harvest_report.json" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  54d0a1d1eab55a0165fd1a20b99d29dfcc9df7b4e5621d4362781d52ae2e7419 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl.manifest.json" \
  11a82c87432a26fff1a0290d48dedb19d0777a833d05e15685f9ba03ad78f614 \
  "${FEEDBACK_DIR}/verpo_feedback_view.public.json" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc \
  "${CHECKPOINT}/adapter/adapter_model.safetensors" \
  c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3 \
  "${CHECKPOINT}/adapter/adapter_config.json" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d \
  "${CHECKPOINT}/tokenizer/tokenizer.json" \
  097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce \
  "${F2_DECODER}" \
  | sha256sum -c -

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/logs" "${WORKSPACE}/.hf_home"
export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

RUNNER="${PROJECT}/scripts/training/t5gemma2_semantic_rescue_ablation.py"
PROJECTION="${OUTPUT_DIR}/base_projection.json"
PLAN="${OUTPUT_DIR}/plan.json"
DIAGNOSES="${OUTPUT_DIR}/diagnoses.json"
DIAGNOSIS_JOURNAL="${OUTPUT_DIR}/diagnoses.journal.jsonl"
GENERATION="${OUTPUT_DIR}/generation.json"
GENERATION_JOURNAL="${OUTPUT_DIR}/generation.journal.jsonl"
SCORE="${OUTPUT_DIR}/score.json"
VISIBLE_SCORE="${OUTPUT_DIR}/visible_selection.json"

cd "${PROJECT}"

/venv/main/bin/python "${RUNNER}" project \
  --base-journal "${PILOT_DIR}/harvest.journal.jsonl" \
  --base-report "${PILOT_DIR}/harvest_report.json" \
  --base-journal "${MIDDLE_DIR}/harvest.journal.jsonl" \
  --base-report "${MIDDLE_DIR}/harvest_report.json" \
  --base-journal "${REMAINING_DIR}/harvest.journal.jsonl" \
  --base-report "${REMAINING_DIR}/harvest_report.json" \
  --output "${PROJECTION}"

/venv/main/bin/python "${RUNNER}" plan \
  --projection "${PROJECTION}" \
  --rollout-file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2-jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --f2-manifest "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl.manifest.json" \
  --public-manifest "${FEEDBACK_DIR}/verpo_feedback_view.public.json" \
  --groups "${RESCUE_GROUP_COUNT}" \
  --seed 260103525 \
  --reward-timeout 30 \
  --stability-runs 2 \
  --output "${PLAN}"

/venv/main/bin/python "${RUNNER}" diagnose \
  --plan "${PLAN}" \
  --model claude-sonnet-5 \
  --base-url https://api.anthropic.com \
  --api-style anthropic_messages \
  --max-tokens 4096 \
  --timeout-seconds 300 \
  --thinking-mode adaptive \
  --reasoning-effort high \
  --reasoning-mode standard \
  --no-chat-json-schema \
  --journal "${DIAGNOSIS_JOURNAL}" \
  --output "${DIAGNOSES}"

accepted_diagnoses="$(/usr/bin/jq -r '.funnel.accepted_parent_diagnoses // -1' "${DIAGNOSES}")"
receipt_count="$(/usr/bin/jq -r '.receipt_chain.count // -1' "${DIAGNOSES}")"
if [[ "${receipt_count}" -ne "${RESCUE_GROUP_COUNT}" ]] \
  || [[ "${accepted_diagnoses}" -le 0 ]] \
  || { [[ "${RESCUE_GROUP_COUNT}" -eq 1 ]] && [[ "${accepted_diagnoses}" -ne 2 ]]; }; then
  echo "T5GEMMA_SEMANTIC_RESCUE_BLOCKED diagnosis health gate failed: receipts=${receipt_count}/${RESCUE_GROUP_COUNT} accepted_parents=${accepted_diagnoses}" >&2
  exit 78
fi

/venv/main/bin/python "${RUNNER}" generate \
  --plan "${PLAN}" \
  --diagnoses "${DIAGNOSES}" \
  --sft-checkpoint "${CHECKPOINT}" \
  --seed 9102026 \
  --max-source-tokens 32768 \
  --max-new-tokens 4096 \
  --temperature 0.8 \
  --top-p 0.95 \
  --attn-implementation sdpa \
  --bf16 \
  --journal "${GENERATION_JOURNAL}" \
  --output "${GENERATION}"

/venv/main/bin/python "${RUNNER}" score \
  --plan "${PLAN}" \
  --generation "${GENERATION}" \
  --private-holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected-private-holdback-sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --visible-output "${VISIBLE_SCORE}" \
  --reward-timeout 30 \
  --stability-runs 2 \
  --workers 8 \
  --output "${SCORE}"

echo "T5GEMMA_SEMANTIC_RESCUE_COMPLETE groups=${RESCUE_GROUP_COUNT} output=${OUTPUT_DIR}"
