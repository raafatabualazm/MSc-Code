#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
FEEDBACK_DIR="${WORKSPACE}/multifunction_v1/expanded2776/verpo_feedback_t5gemma2_v1"
LOCAL_DIR="${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_mixed_kimi_probe_v1"
LOCAL_JOURNAL="${LOCAL_DIR}/harvest.journal.jsonl"
LOCAL_REPORT="${LOCAL_DIR}/harvest_report.json"
SOURCE_DIR="${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_kimi_k3_mixed_paired50_v12"
SOURCE_REPORT="${SOURCE_DIR}/api_rescue_report.json"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_kimi_k3_retry17_8k_v1"
SECRET_FILE="${T5GEMMA_OPENROUTER_ENV:-${WORKSPACE}/secrets/Azure.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
RESCUE_SCRIPT="${PROJECT}/scripts/training/t5gemma2_api_rs_sft_rescue.py"

printf '%s  %s\n' \
  ad681aaa68db63dbc64ce847f32f18e2740e4db2050d1211e5d5457fdc6dff69 \
  "${RESCUE_SCRIPT}" \
  fe9bcd00c6774432b7911129246c8b2837523d85b1c94efb29c03f85ae860205 \
  "${SOURCE_REPORT}" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-local-rs-sft-pilot-report-v1"
   and .status == "complete"
   and .pilot.tasks == 100
   and .privacy_invariants.heldout_175_opened == false
   and .privacy_invariants.frontier_api_calls == false' \
  "${LOCAL_REPORT}" >/dev/null; then
  echo "T5GEMMA_KIMI_RETRY_BLOCKED local harvest report differs" >&2
  exit 78
fi
if ! head -n 1 "${LOCAL_JOURNAL}" | /usr/bin/jq -e \
  '.schema == "t5gemma2-local-rs-sft-pilot-journal-v1"
   and .contract.script_sha256
     == "0a6134c1753e69b75aa46eb4e762ab463b61c411db0c4c3ba7b18fe2f8e96f1d"
   and .contract.checkpoint_contract_sha256
     == "90d95239e17a524c2541dbe98cd728c16b1fbdd891baad27c8a488f5dc111369"
   and .contract.checkpoint_loader_compatibility.wrapper_sha256
     == "8ddf567c42124e6d549c69eb3748ecce256c8ec20522a8aae7f32732167a6477"
   and .contract.checkpoint_loader_compatibility.inference_core_sha256
     == "564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9"
   and .contract.checkpoint_loader_compatibility.mixed_loader_sha256
     == "758b0cf37475cacf8789ce9db62d3e6e8f88fe344c6d616c53b0c1d221921972"
   and .contract.schedule.seed == 20260730
   and .contract.schedule.pilot_tasks == 100
   and .contract.sampling.base_samples == 4
   and .contract.sampling.repair_samples == 0
   and .contract.sampling.max_repair_parents == 0
   and .contract.training_build.gold_replay_ratio == 0
   and .contract.no_frontier_api == true
   and .contract.heldout_175_opened == false' >/dev/null; then
  echo "T5GEMMA_KIMI_RETRY_BLOCKED local journal contract differs" >&2
  exit 78
fi
if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_KIMI_RETRY_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi
if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_KIMI_RETRY_BLOCKED OpenRouter secret bundle is absent" >&2
  exit 78
fi

ENDPOINTS_JSON="$(mktemp)"
trap 'rm -f -- "${ENDPOINTS_JSON}"' EXIT
curl -fsS --max-time 30 \
  https://openrouter.ai/api/v1/models/moonshotai/kimi-k3-20260715/endpoints \
  >"${ENDPOINTS_JSON}"
if ! /usr/bin/jq -e \
  '.data.id == "moonshotai/kimi-k3"
   and any(.data.endpoints[];
     (.tag == "baseten/fp8"
      or .tag == "modal/mxfp4"
      or .tag == "digitalocean"
      or .tag == "together"
      or .tag == "fireworks"
      or .tag == "moonshotai/mxfp4")
     and .status == 0
     and .context_length >= 57344
     and (.pricing.prompt | tonumber) <= 0.000003
     and (.pricing.completion | tonumber) <= 0.000015
     and (.supported_parameters | index("reasoning"))
     and (.supported_parameters | index("reasoning_effort"))
     and (.supported_parameters | index("max_tokens")))' \
  "${ENDPOINTS_JSON}" >/dev/null; then
  echo "T5GEMMA_KIMI_RETRY_BLOCKED no sealed route is currently suitable" >&2
  exit 78
fi

# Parse only OPENROUTER_API_KEY; never source the multi-provider bundle.
OPENROUTER_API_KEY="$(
  /venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import re
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if stat.S_IMODE(path.stat().st_mode) & 0o077:
    raise SystemExit("OpenRouter secret bundle must be mode 0600")
raw = path.read_bytes()
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("utf-16")
pattern = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")
values = []
for line_number, raw_line in enumerate(text.splitlines(), 1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    match = pattern.fullmatch(line)
    if not match:
        if "OPENROUTER_API_KEY" in line:
            raise SystemExit(f"malformed key assignment at line {line_number}")
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
export OPENROUTER_API_KEY
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT}"

# 17 * (13,312 input * $3/M + 8,192 output * $15/M) = $2.767872.
# Only source parse failures or length-truncated responses are regenerated.
set +e
/venv/main/bin/python "${RESCUE_SCRIPT}" \
  --pilot_journal "${LOCAL_JOURNAL}" \
  --rollout_file "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  --f2_jsonl "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  --private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  --expected_rollout_sha256 14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  --expected_f2_sha256 c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  --expected_private_holdback_sha256 dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  --retry_parse_failures_or_truncations_report "${SOURCE_REPORT}" \
  --expected_retry_parse_failures_or_truncations_report_sha256 fe9bcd00c6774432b7911129246c8b2837523d85b1c94efb29c03f85ae860205 \
  --expected_retry_parse_failures_or_truncations_tasks 17 \
  --expected_retry_parse_failures_or_truncations_task_ids_sha256 15a66c43b97fa72fd47702689babc3b6ab33ee48f24813a34a7b62f2d9ccc00a \
  --expected_scheduled_task_ids_sha256 15a66c43b97fa72fd47702689babc3b6ab33ee48f24813a34a7b62f2d9ccc00a \
  --output_dir "${OUTPUT_DIR}" \
  --provider openrouter_chat \
  --model moonshotai/kimi-k3 \
  --base_url https://openrouter.ai/api/v1 \
  --api_key_env OPENROUTER_API_KEY \
  --chat_token_parameter max_tokens \
  --openrouter_provider_only baseten/fp8 \
  --openrouter_provider_only modal/mxfp4 \
  --openrouter_provider_only digitalocean \
  --openrouter_provider_only together \
  --openrouter_provider_only fireworks \
  --openrouter_provider_only moonshotai/mxfp4 \
  --openrouter_provider_order baseten/fp8 \
  --openrouter_provider_order modal/mxfp4 \
  --openrouter_provider_order digitalocean \
  --openrouter_provider_order together \
  --openrouter_provider_order fireworks \
  --openrouter_provider_order moonshotai/mxfp4 \
  --openrouter_allow_fallbacks \
  --openrouter_require_parameters \
  --openrouter_enforce_distillable_text \
  --openrouter_reasoning enabled \
  --openrouter_reasoning_effort low \
  --openrouter_include_reasoning \
  --seed 20260730 \
  --eligible_task_offset 0 \
  --max_tasks 17 \
  --max_parents_per_task 1 \
  --samples_per_parent 1 \
  --max_calls 17 \
  --max_input_tokens_per_call 13312 \
  --max_output_tokens 8192 \
  --max_input_tokens_total 226304 \
  --max_output_tokens_total 139264 \
  --max_total_tokens 365568 \
  --max_usd 3 \
  --input_usd_per_million 3 \
  --output_usd_per_million 15 \
  --timeout_seconds 900 \
  --inter_call_delay_seconds 10 \
  --abort_on_provider_error \
  --provider_max_attempts 8 \
  --provider_retry_base_seconds 2 \
  --provider_retry_max_seconds 30 \
  --timeout 30 \
  --stability_runs 2
run_status=$?
set -e
if (( run_status != 0 )); then
  echo "T5GEMMA_KIMI_RETRY_FAIL_FAST status=${run_status}" >&2
  exit 78
fi
