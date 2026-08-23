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
TRANCHE2_DIR="${T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1}"
TRANCHE2_JOURNAL="${TRANCHE2_DIR}/api_rescue.journal.jsonl"
TRANCHE2_REPORT="${TRANCHE2_DIR}/api_rescue_report.json"
OUTPUT_DIR="${T5GEMMA_CLAUDE_OPUS_RESIDUAL_PROBE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1}"
SECRET_FILE="${T5GEMMA_ANTHROPIC_ENV:-${WORKSPACE}/secrets/Anthropic.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

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
  echo "T5GEMMA_CLAUDE_OPUS_RESIDUAL_BLOCKED exact completed pilot is absent" >&2
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
     and .heldout_175_opened == false
     and .schedule.scheduled_tasks == 73
     and .schedule.scheduled_calls == 73
     and .verification.verified_unique_hard_targets == 23
     and .budget_charged.estimated_usd == "3.914444000"
     and .provider.provider == "anthropic"
     and .provider.model == "claude-sonnet-5"
     and .provider.thinking == "adaptive"
     and .provider.effort == "high"
     and .outputs.direct_targets.sha256
       == "2bbd8ccc486734a7aed738e9cb705105e79778162f2fbb99798895e9142611d3"
     and .outputs.direct_f2.sha256
       == "b8a0d0c6af81499e56d64d5fe66e0b82b40e791602e8063f843456f276489614"
     and .outputs.repair_targets.sha256
       == "5f5d19a84eb897591de9b57b72f8cfb3e86449a9abe4ad16dfaca2a6cd8e58f0"
     and .outputs.repair_sources.sha256
       == "b05f7236d272be6aa04caec27c2b401938173812c8df149c327679c6f8159068"
     and .privacy_invariants.private_holdback_sent_to_provider == false
     and .privacy_invariants.gold_sent_to_provider == false' \
    "${TRANCHE1_REPORT}" >/dev/null \
  || ! /usr/bin/head -n 1 "${TRANCHE1_JOURNAL}" \
    | /usr/bin/jq -e \
      '.event == "header"
       and .contract_sha256
         == "056834fb23af50bc14222254baec5c985b3223179698e34429a408c989a7ccf7"
       and .contract.selection.seed == 42
       and .contract.selection.scheduled_tasks == 73
       and .contract.selection.scheduled_slots == 73
       and .contract.selection.task_ids_sha256
         == "b142fc681a538e2d7356caba0a3a7ce5fc0f4edf435f06b8ccef789e9cd1cf0e"' \
      >/dev/null; then
  echo "T5GEMMA_CLAUDE_OPUS_RESIDUAL_BLOCKED exact Sonnet tranche 1 is absent" >&2
  exit 78
fi

if [[ ! -s "${TRANCHE2_JOURNAL}" ]] \
  || [[ ! -s "${TRANCHE2_JOURNAL}.chain-head.json" ]] \
  || [[ ! -s "${TRANCHE2_REPORT}" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-api-rs-sft-rescue-report-v1"
     and .status == "complete"
     and .run_contract_sha256
       == "2bb6a249bf3bfbd7949611dca16f8d7b7cc76ce07c2aee74eb141f9fd48625bc"
     and .production_floor_eligible == true
     and .heldout_175_opened == false
     and .schedule.eligible_all_zero_tasks_before_offset == 188
     and .schedule.eligible_task_offset == 73
     and .schedule.scheduled_tasks == 115
     and .schedule.scheduled_calls == 115
     and .schedule.task_ids_sha256
       == "beff513901e3fe30808a1665f4ac71b98106ac833b6f5ce6f4da3dbd93a67851"
     and .verification.verified_unique_hard_targets == 42
     and .budget_charged.estimated_usd == "5.756584000"
     and .provider.provider == "anthropic"
     and .provider.model == "claude-sonnet-5"
     and .provider.thinking == "adaptive"
     and .provider.effort == "high"
     and .outputs.direct_targets.sha256
       == "e31d438ade29469b5a742c16f4dc4708b6b8491a6aa8843fad29ee20d8114b1b"
     and .outputs.direct_f2.sha256
       == "32693a4cba090008c11dff47ca916f8c4c9f1209cbb90c2825f80f13a52758f0"
     and .outputs.repair_targets.sha256
       == "4063d7ef4cd0daa34097dac9bf4d7205ad5ec9674bea4466b2b74ce82b6c6c84"
     and .outputs.repair_sources.sha256
       == "d41096e2b1a6257bc1ea0695dcdd02a31b0bbf0c3c9757df9479c45ea44f688d"
     and .privacy_invariants.private_holdback_sent_to_provider == false
     and .privacy_invariants.gold_sent_to_provider == false' \
    "${TRANCHE2_REPORT}" >/dev/null \
  || ! /usr/bin/head -n 1 "${TRANCHE2_JOURNAL}" \
    | /usr/bin/jq -e \
      '.event == "header"
       and .contract_sha256
         == "2bb6a249bf3bfbd7949611dca16f8d7b7cc76ce07c2aee74eb141f9fd48625bc"
       and .contract.selection.seed == 42
       and .contract.selection.eligible_task_offset == 73
       and .contract.selection.scheduled_tasks == 115
       and .contract.selection.scheduled_slots == 115
       and .contract.selection.task_ids_sha256
         == "beff513901e3fe30808a1665f4ac71b98106ac833b6f5ce6f4da3dbd93a67851"' \
      >/dev/null; then
  echo "T5GEMMA_CLAUDE_OPUS_RESIDUAL_BLOCKED exact Sonnet tranche 2 is absent" >&2
  exit 78
fi

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
  2bbd8ccc486734a7aed738e9cb705105e79778162f2fbb99798895e9142611d3 \
  "${TRANCHE1_DIR}/direct_hard_targets.jsonl" \
  b8a0d0c6af81499e56d64d5fe66e0b82b40e791602e8063f843456f276489614 \
  "${TRANCHE1_DIR}/direct_hard_targets_f2.jsonl" \
  ba6a79bbc52df4ee1ca9ec8b6dfc5da6f55d2b89636735d5fc5ae5ea9146ce59 \
  "${TRANCHE1_DIR}/direct_manifest.json" \
  7eebca14461c2ff9bb6c05bfcdbbb410f2fa276ddf1d592c23d90006719f9253 \
  "${TRANCHE1_DIR}/repair_policy_manifest.json" \
  b05f7236d272be6aa04caec27c2b401938173812c8df149c327679c6f8159068 \
  "${TRANCHE1_DIR}/repair_policy_sources.jsonl" \
  5f5d19a84eb897591de9b57b72f8cfb3e86449a9abe4ad16dfaca2a6cd8e58f0 \
  "${TRANCHE1_DIR}/repair_policy_targets.jsonl" \
  4bdeb9e6f5a0d3063b6d454d91bde65596ef788a7edd08d67045fa545b6481d6 \
  "${TRANCHE2_JOURNAL}" \
  1acc0053ccf0f03553dbbe5477fed09e079261c112984eb4cb35a673032a4ba2 \
  "${TRANCHE2_JOURNAL}.chain-head.json" \
  99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4 \
  "${TRANCHE2_REPORT}" \
  e31d438ade29469b5a742c16f4dc4708b6b8491a6aa8843fad29ee20d8114b1b \
  "${TRANCHE2_DIR}/direct_hard_targets.jsonl" \
  32693a4cba090008c11dff47ca916f8c4c9f1209cbb90c2825f80f13a52758f0 \
  "${TRANCHE2_DIR}/direct_hard_targets_f2.jsonl" \
  7b7ace2229afed6de7948fc3e0dfe3a0a490e408b6c9552550d4b1a3f3eb54c8 \
  "${TRANCHE2_DIR}/direct_manifest.json" \
  13928f719d1c341949b9a3d064e85f813065985b7cc46e7d73b900cbe7c65c31 \
  "${TRANCHE2_DIR}/repair_policy_manifest.json" \
  d41096e2b1a6257bc1ea0695dcdd02a31b0bbf0c3c9757df9479c45ea44f688d \
  "${TRANCHE2_DIR}/repair_policy_sources.jsonl" \
  4063d7ef4cd0daa34097dac9bf4d7205ad5ec9674bea4466b2b74ce82b6c6c84 \
  "${TRANCHE2_DIR}/repair_policy_targets.jsonl" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_CLAUDE_OPUS_RESIDUAL_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi
if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_CLAUDE_OPUS_RESIDUAL_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi

# Read one raw key or one ANTHROPIC_API_KEY assignment without sourcing shell.
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

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
# One Opus call is paired with each of the first 20 deterministic residual
# tasks. At $5/M input and $25/M output, a full 49,152-in/8,192-out call
# reserves $0.450560; 20 calls reserve the exact $9.011200 cap. Adding the
# sealed prior actual spend gives $9.981860 + $9.011200 = $18.993060, below
# the user-authorized $30 total Claude ceiling.
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
  --expected_scheduled_task_ids_sha256 9ddf1a24954de70810a94cf44ae0b1ebc8fc13ca50703a2b1715325855451f8f \
  --output_dir "${OUTPUT_DIR}" \
  --provider anthropic \
  --model claude-opus-5 \
  --base_url https://api.anthropic.com \
  --api_key_env ANTHROPIC_API_KEY \
  --anthropic_thinking adaptive \
  --anthropic_effort high \
  --seed 42 \
  --eligible_task_offset 0 \
  --max_tasks 20 \
  --max_parents_per_task 1 \
  --samples_per_parent 1 \
  --max_calls 20 \
  --max_input_tokens_per_call 49152 \
  --max_output_tokens 8192 \
  --max_input_tokens_total 983040 \
  --max_output_tokens_total 163840 \
  --max_total_tokens 1146880 \
  --max_usd 9.0112 \
  --input_usd_per_million 5 \
  --output_usd_per_million 25 \
  --timeout_seconds 600 \
  --timeout 30 \
  --stability_runs 2
