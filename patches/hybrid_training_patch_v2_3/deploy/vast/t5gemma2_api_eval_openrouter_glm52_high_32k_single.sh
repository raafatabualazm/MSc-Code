#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
LOCAL_PILOT_DIR="${T5GEMMA_RS_SFT_2EPOCH_PILOT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1}"
LOCAL_PILOT_JOURNAL="${LOCAL_PILOT_DIR}/harvest.journal.jsonl"
LOCAL_PILOT_REPORT="${LOCAL_PILOT_DIR}/harvest_report.json"
TRANCHE1_DIR="${T5GEMMA_CLAUDE_PRODUCTION_TRANCHE1_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1}"
TRANCHE1_REPORT="${TRANCHE1_DIR}/api_rescue_report.json"
TRANCHE2_DIR="${T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1}"
TRANCHE2_REPORT="${TRANCHE2_DIR}/api_rescue_report.json"
OUTPUT_DIR="${T5GEMMA_OPENROUTER_GLM52_HIGH_32K_SINGLE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_eval_openrouter_glm52_high_32k_single_2epoch_v1}"
SECRET_FILE="${T5GEMMA_OPENROUTER_ENV:-${WORKSPACE}/secrets/Azure.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
RESCUE_SCRIPT="${PROJECT}/scripts/training/t5gemma2_api_rs_sft_rescue.py"

# Seal the executable, source harvest, prior-success reports, and all three
# evaluation views. The runner independently reopens and validates each
# prior report's journal, chain head, manifests, and JSONL outputs.
printf '%s  %s\n' \
  f655a8cb0253033f5674845b83f71f44df35ba7c962448b349e449aa04b79dda \
  "${RESCUE_SCRIPT}" \
  5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58 \
  "${LOCAL_PILOT_JOURNAL}" \
  9f670cf606f7fc68e157508e2e064a1954280fef9063c6ec5239e3e6ca63be1d \
  "${LOCAL_PILOT_JOURNAL}.chain-head.json" \
  b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab \
  "${LOCAL_PILOT_REPORT}" \
  fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad \
  "${TRANCHE1_REPORT}" \
  99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4 \
  "${TRANCHE2_REPORT}" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

expected_system_prompt_sha256=0118f9f452ff23093b11abab0076a6297f6d91be7d0777e7a91b1720f183bb5e
observed_system_prompt_sha256="$(
  /venv/main/bin/python - "${PROJECT}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from scripts.training.seq2seq_verpo_core import sha256_text
from scripts.training.t5gemma2_api_rs_sft_rescue import SYSTEM_PROMPT
print(sha256_text(SYSTEM_PROMPT), end="")
PY
)"
if [[ "${observed_system_prompt_sha256}" != "${expected_system_prompt_sha256}" ]]; then
  echo "T5GEMMA_OPENROUTER_GLM52_HIGH_32K_SINGLE_BLOCKED system prompt digest differs" >&2
  exit 78
fi
unset observed_system_prompt_sha256 expected_system_prompt_sha256

if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_OPENROUTER_GLM52_HIGH_32K_SINGLE_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi
if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_OPENROUTER_GLM52_HIGH_32K_SINGLE_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi

# Parse only OPENROUTER_API_KEY. Do not source a multi-provider secret bundle.
OPENROUTER_API_KEY="$(
  /venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import re
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if stat.S_IMODE(path.stat().st_mode) & 0o077:
    raise SystemExit("OpenRouter secret bundle must not be group/world accessible")
raw = path.read_bytes()
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("utf-16")
assignment = re.compile(
    r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)"
)
values = []
for line_number, raw_line in enumerate(text.splitlines(), 1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    match = assignment.fullmatch(line)
    if not match:
        if "OPENROUTER_API_KEY" in line:
            raise SystemExit(
                f"malformed OPENROUTER_API_KEY assignment at line {line_number}"
            )
        continue
    name, value = match.groups()
    if name != "OPENROUTER_API_KEY":
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value or any(char.isspace() for char in value):
        raise SystemExit("OPENROUTER_API_KEY is empty or malformed")
    values.append(value)
if len(values) != 1:
    raise SystemExit("OPENROUTER_API_KEY must occur exactly once")
print(values[0], end="")
PY
)"
if [[ -z "${OPENROUTER_API_KEY}" ]]; then
  echo "T5GEMMA_OPENROUTER_GLM52_HIGH_32K_SINGLE_BLOCKED key parse failed" >&2
  exit 78
fi
export OPENROUTER_API_KEY

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
# One call reserves at most $0.106496:
# 49,152 * $0.70/M input + 32,768 * $2.20/M output, below $0.12.
# This is evaluation-only: the runner emits no trainable target artifacts.
exec /venv/main/bin/python scripts/training/t5gemma2_api_rs_sft_rescue.py \
  --pilot_journal "${LOCAL_PILOT_JOURNAL}" \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected_rollout_sha256 14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  --expected_f2_sha256 c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  --expected_private_holdback_sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --prior_success_report "${TRANCHE1_REPORT}" \
  --expected_prior_success_report_sha256 fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad \
  --prior_success_report "${TRANCHE2_REPORT}" \
  --expected_prior_success_report_sha256 99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4 \
  --require_prior_schedules_disjoint \
  --require_prior_schedule_complete_coverage \
  --expected_prior_scheduled_tasks 188 \
  --expected_prior_verified_tasks 65 \
  --expected_residual_tasks 123 \
  --expected_prior_scheduled_task_ids_sha256 a52cacfb325a927dc758fca3e03608dfdb479b108f26791f2b916983f1de6994 \
  --expected_prior_verified_task_ids_sha256 2797d436c1b596a6771b9c1494c15ce95cf91046c17a04f081c0847dad36bc7a \
  --expected_residual_task_ids_sha256 262532ddcc7ee9b1b03125d7600d96e0ded5886971038985061523c0fcf1c4e6 \
  --expected_scheduled_task_ids_sha256 1751df08e563a3b07c0c9b968e7a5723cf4d3227402596485306c8f3d5595247 \
  --output_dir "${OUTPUT_DIR}" \
  --provider openrouter_chat \
  --model z-ai/glm-5.2 \
  --base_url https://openrouter.ai/api/v1 \
  --api_key_env OPENROUTER_API_KEY \
  --chat_token_parameter max_tokens \
  --openrouter_provider_only novita/fp8 \
  --openrouter_require_parameters \
  --evaluation_only \
  --openrouter_reasoning enabled \
  --openrouter_reasoning_effort high \
  --openrouter_include_reasoning \
  --seed 42 \
  --eligible_task_offset 101 \
  --max_tasks 1 \
  --max_parents_per_task 1 \
  --samples_per_parent 1 \
  --max_calls 1 \
  --max_input_tokens_per_call 49152 \
  --max_output_tokens 32768 \
  --max_input_tokens_total 49152 \
  --max_output_tokens_total 32768 \
  --max_total_tokens 81920 \
  --max_usd 0.12 \
  --input_usd_per_million 0.70 \
  --output_usd_per_million 2.20 \
  --timeout_seconds 600 \
  --timeout 30 \
  --stability_runs 2
