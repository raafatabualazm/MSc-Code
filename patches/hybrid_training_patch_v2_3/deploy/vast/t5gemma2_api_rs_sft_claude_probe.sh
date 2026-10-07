#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
LOCAL_PILOT_DIR="${T5GEMMA_RS_SFT_PILOT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_v1}"
LOCAL_PILOT_JOURNAL="${LOCAL_PILOT_DIR}/harvest.journal.jsonl"
OUTPUT_DIR="${T5GEMMA_CLAUDE_PROBE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_probe_prefix10_v1}"
SECRET_FILE="${T5GEMMA_ANTHROPIC_ENV:-${WORKSPACE}/secrets/Anthropic.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_CLAUDE_PROBE_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi
if [[ ! -s "${LOCAL_PILOT_JOURNAL}" ]] \
  || [[ ! -s "${LOCAL_PILOT_JOURNAL}.chain-head.json" ]]; then
  echo "T5GEMMA_CLAUDE_PROBE_BLOCKED local pilot journal/chain head is absent" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_CLAUDE_PROBE_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi

# Read one raw key or one ANTHROPIC_API_KEY assignment without sourcing shell
# syntax. The value is exported only to the child process and is never logged.
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
    raise SystemExit(
        "Anthropic.env must contain exactly one non-comment key line"
    )
line = lines[0]
match = re.fullmatch(
    r"(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.*)", line, re.IGNORECASE
)
value = match.group(1).strip() if match else line
if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
    value = value[1:-1]
if not value or any(char.isspace() for char in value):
    raise SystemExit("Anthropic API key is empty or malformed")
print(value, end="")
PY
)"
export ANTHROPIC_API_KEY="${anthropic_key}"
unset anthropic_key

printf '%s  %s\n' \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

mkdir -p "${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/api_rescue_report.json" ]] \
  && [[ "$(/usr/bin/jq -r '.status // empty' "${OUTPUT_DIR}/api_rescue_report.json")" == complete ]]; then
  echo "T5GEMMA_CLAUDE_PROBE_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
exec /venv/main/bin/python scripts/training/t5gemma2_api_rs_sft_rescue.py \
  --pilot_journal "${LOCAL_PILOT_JOURNAL}" \
  --exploratory_terminal_prefix 10 \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected_rollout_sha256 14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  --expected_f2_sha256 c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  --expected_private_holdback_sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --output_dir "${OUTPUT_DIR}" \
  --provider anthropic \
  --model claude-sonnet-5 \
  --base_url https://api.anthropic.com \
  --api_key_env ANTHROPIC_API_KEY \
  --anthropic_thinking adaptive \
  --anthropic_effort high \
  --seed 42 \
  --max_tasks 5 \
  --max_parents_per_task 1 \
  --samples_per_parent 1 \
  --max_calls 5 \
  --max_input_tokens_per_call 49152 \
  --max_output_tokens 8192 \
  --max_input_tokens_total 245760 \
  --max_output_tokens_total 40960 \
  --max_total_tokens 286720 \
  --max_usd 0.95 \
  --input_usd_per_million 2 \
  --output_usd_per_million 10 \
  --timeout_seconds 600 \
  --timeout 30 \
  --stability_runs 2
