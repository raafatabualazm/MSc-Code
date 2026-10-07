#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
LOCAL_PILOT_DIR="${T5GEMMA_RS_SFT_2EPOCH_PILOT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1}"
LOCAL_PILOT_JOURNAL="${LOCAL_PILOT_DIR}/harvest.journal.jsonl"
LOCAL_PILOT_REPORT="${LOCAL_PILOT_DIR}/harvest_report.json"
TRANCHE1_DIR="${T5GEMMA_CLAUDE_PRODUCTION_TRANCHE1_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_v1}"
TRANCHE1_JOURNAL="${TRANCHE1_DIR}/api_rescue.journal.jsonl"
TRANCHE1_REPORT="${TRANCHE1_DIR}/api_rescue_report.json"
OUTPUT_DIR="${T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1}"
SECRET_FILE="${T5GEMMA_ANTHROPIC_ENV:-${WORKSPACE}/secrets/Anthropic.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi
if [[ ! -s "${LOCAL_PILOT_JOURNAL}" ]] \
  || [[ ! -s "${LOCAL_PILOT_JOURNAL}.chain-head.json" ]] \
  || [[ ! -s "${LOCAL_PILOT_REPORT}" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-local-rs-sft-pilot-report-v1"
     and .status == "complete"
     and .pilot.tasks == 200
     and .pilot.accepted_unique_targets == 9
     and .checkpoint.warmstart_contract_sha256
       == "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3"
     and .privacy_invariants.frontier_api_calls == false
     and .privacy_invariants.heldout_175_opened == false
     and .privacy_invariants.private_holdback_text_in_model_input == false' \
    "${LOCAL_PILOT_REPORT}" >/dev/null; then
  echo "T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_BLOCKED completed two-epoch pilot is absent" >&2
  exit 78
fi
if [[ ! -s "${TRANCHE1_JOURNAL}" ]] \
  || [[ ! -s "${TRANCHE1_JOURNAL}.chain-head.json" ]] \
  || [[ ! -s "${TRANCHE1_REPORT}" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-api-rs-sft-rescue-report-v1"
     and .status == "complete"
     and .run_contract_sha256
       == "056834fb23af50bc14222254baec5c985b3223179698e34429a408c989a7ccf7"
     and .production_floor_eligible == true
     and .schedule.eligible_all_zero_tasks_before_caps == 188
     and .schedule.scheduled_tasks == 73
     and .schedule.scheduled_calls == 73
     and .verification.verified_unique_hard_targets == 23
     and .budget_charged.estimated_usd == "3.914444000"
     and .provider.provider == "anthropic"
     and .provider.model == "claude-sonnet-5"
     and .provider.thinking == "adaptive"
     and .provider.effort == "high"
     and .privacy_invariants.private_holdback_sent_to_provider == false
     and .privacy_invariants.gold_sent_to_provider == false' \
    "${TRANCHE1_REPORT}" >/dev/null \
  || ! /usr/bin/head -n 1 "${TRANCHE1_JOURNAL}" \
    | /usr/bin/jq -e \
      '.event == "header"
       and .contract_sha256
         == "056834fb23af50bc14222254baec5c985b3223179698e34429a408c989a7ccf7"
       and .contract.selection.seed == 42
       and .contract.selection.max_tasks_before_budget == 188
       and .contract.selection.max_parents_per_task == 1
       and .contract.selection.samples_per_parent == 1
       and .contract.selection.scheduled_tasks == 73
       and .contract.selection.scheduled_slots == 73
       and .contract.selection.task_ids_sha256
         == "b142fc681a538e2d7356caba0a3a7ce5fc0f4edf435f06b8ccef789e9cd1cf0e"' \
      >/dev/null; then
  echo "T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_BLOCKED exact completed tranche 1 is absent" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_BLOCKED Dart 3.12.2 is not executable" >&2
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
  5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58 \
  "${LOCAL_PILOT_JOURNAL}" \
  9f670cf606f7fc68e157508e2e064a1954280fef9063c6ec5239e3e6ca63be1d \
  "${LOCAL_PILOT_JOURNAL}.chain-head.json" \
  b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab \
  "${LOCAL_PILOT_REPORT}" \
  b2b6dfbb3d0a3efd5cbadee09e134c24fa7594f6df1238833d25a7b671c9af10 \
  "${TRANCHE1_JOURNAL}" \
  61108ea2c34fc7776c5b5797d103429a6ac979fae5f3d0385d74ce807f69afae \
  "${TRANCHE1_JOURNAL}.chain-head.json" \
  fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad \
  "${TRANCHE1_REPORT}" \
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
  echo "T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_ALREADY_COMPLETE output=${OUTPUT_DIR}"
  exit 0
fi

export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
# Offset 73 skips exactly the first completed tranche's deterministic eligible
# prefix. At the sealed $2/$10 rates, each call reserves $0.180224 and all 115
# remaining calls reserve $20.725760. Prior actual probe/tranche-1 spend plus
# this full reservation is $0.310832 + $3.914444 + $20.725760 = $24.951036,
# below the user-authorized $30 total Claude ceiling.
exec /venv/main/bin/python scripts/training/t5gemma2_api_rs_sft_rescue.py \
  --pilot_journal "${LOCAL_PILOT_JOURNAL}" \
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
  --eligible_task_offset 73 \
  --max_tasks 115 \
  --max_parents_per_task 1 \
  --samples_per_parent 1 \
  --max_calls 115 \
  --max_input_tokens_per_call 49152 \
  --max_output_tokens 8192 \
  --max_input_tokens_total 5652480 \
  --max_output_tokens_total 942080 \
  --max_total_tokens 6594560 \
  --max_usd 21.00 \
  --input_usd_per_million 2 \
  --output_usd_per_million 10 \
  --timeout_seconds 600 \
  --timeout 30 \
  --stability_runs 2
