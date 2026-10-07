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
FIRST_OPUS_DIR="${T5GEMMA_CLAUDE_OPUS_RESIDUAL_PROBE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1}"
FIRST_OPUS_JOURNAL="${FIRST_OPUS_DIR}/api_rescue.journal.jsonl"
FIRST_OPUS_REPORT="${FIRST_OPUS_DIR}/api_rescue_report.json"
SECOND_OPUS_DIR="${T5GEMMA_CLAUDE_OPUS_RESIDUAL_TRANCHE2_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_claude_opus_production_residual_tranche2_2epoch_v1}"
SECOND_OPUS_JOURNAL="${SECOND_OPUS_DIR}/api_rescue.journal.jsonl"
SECOND_OPUS_REPORT="${SECOND_OPUS_DIR}/api_rescue_report.json"
AZURE_DIR="${T5GEMMA_AZURE_RESIDUAL_PROBE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1}"
AZURE_JOURNAL="${AZURE_DIR}/api_rescue.journal.jsonl"
AZURE_REPORT="${AZURE_DIR}/api_rescue_report.json"
M3_DIR="${T5GEMMA_OPENROUTER_MINIMAX_M3_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_minimax_m3_residual_probe_2epoch_v1}"
M3_JOURNAL="${M3_DIR}/api_rescue.journal.jsonl"
M3_REPORT="${M3_DIR}/api_rescue_report.json"
OUTPUT_DIR="${T5GEMMA_OPENROUTER_GLM52_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_openrouter_glm52_residual_probe_2epoch_v1}"
# The already-deployed 0600 secret bundle contains the single targeted
# OPENROUTER_API_KEY assignment.  The parser below ignores every other entry.
SECRET_FILE="${T5GEMMA_OPENROUTER_ENV:-${WORKSPACE}/secrets/Azure.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
RESCUE_SCRIPT="${PROJECT}/scripts/training/t5gemma2_api_rs_sft_rescue.py"

# These exact SHA-256 pins cover the source pilot, all earlier API rescue
# journals/reports/training artifacts, and the three sealed evaluation views.
# The Azure and rejected M3 arms are evidence only: neither is supplied as a
# prior-success report. Only the 65 verified Sonnet successes are excluded, so
# this probe is positions 80..99 in the stable Sonnet-only residual ordering.
printf '%s  %s\n' \
  f655a8cb0253033f5674845b83f71f44df35ba7c962448b349e449aa04b79dda \
  "${RESCUE_SCRIPT}" \
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
  49b97de386b759955497e3f9ab7b4358ca5e74ebf3a877fb6c7f3d98e39275b6 \
  "${FIRST_OPUS_JOURNAL}" \
  79b2d7a95dadabc5b6701d063192df684054bfbbd322ee1faf4d9171ed6e186c \
  "${FIRST_OPUS_JOURNAL}.chain-head.json" \
  f42e0fc17cf317ede9d7d562549938e0068c91dc780dfa089d9fc844a791570b \
  "${FIRST_OPUS_REPORT}" \
  15ef808838ed01347e646e9b4462f48ae88d4afcb467d144f6c6283576abf180 \
  "${FIRST_OPUS_DIR}/direct_hard_targets.jsonl" \
  e0837783e4939ace53b794e0a5654a1781c7cd0fe5133e70556bc27135cf6209 \
  "${FIRST_OPUS_DIR}/direct_hard_targets_f2.jsonl" \
  c98a5e60542e88e5fcdd4a5868f4b6d8939fd879f86ae1211f3f1ca1a1d814a7 \
  "${FIRST_OPUS_DIR}/direct_manifest.json" \
  ceed91c71f5f3e9c22f9fca90b1a694c9da0f513f29c861c1c49d5f5b41cd04e \
  "${FIRST_OPUS_DIR}/repair_policy_manifest.json" \
  b23662bcecc442d722c3d15bc261e95cd7266e6a449cd1586238e713dcc9e409 \
  "${FIRST_OPUS_DIR}/repair_policy_sources.jsonl" \
  1ccad6105a24a590ffebae0ec34a534fbbfddf0615bccfc142f404f79ea2f632 \
  "${FIRST_OPUS_DIR}/repair_policy_targets.jsonl" \
  5c610a4073122e209e26af8e689a683258405c00e58a23c6e9a109c76f9c4c6c \
  "${SECOND_OPUS_JOURNAL}" \
  d5da7c8ed6ec045239602f08d290dac7e123d44c9edc17b1bf121763db1b1511 \
  "${SECOND_OPUS_JOURNAL}.chain-head.json" \
  fa0c70c73767a525f2ca710fd822cb2bdca60140f133696ad15b87e71d2751d1 \
  "${SECOND_OPUS_REPORT}" \
  2e02a7db60d0baf9d64afdc9b5bb211fcd0253186b490e9af911de8d49b87bf7 \
  "${SECOND_OPUS_DIR}/direct_hard_targets.jsonl" \
  5fd2d1cd56b0c2de0ed79fd4cfdb3017244774f038517e574a7089a39ea51a91 \
  "${SECOND_OPUS_DIR}/direct_hard_targets_f2.jsonl" \
  059e80aed630301ec10049de8e3b1342f61b9e4a96706a8dbb4342982b49b069 \
  "${SECOND_OPUS_DIR}/direct_manifest.json" \
  f3c84ed416722481f74cb448bf70b2a94e206c04654255c33ef535c7eb8b47ed \
  "${SECOND_OPUS_DIR}/repair_policy_manifest.json" \
  bdbe4aba30cb0b861bc40d5dc6c89e4d35e63b374742aba4c0bc42f627950da7 \
  "${SECOND_OPUS_DIR}/repair_policy_sources.jsonl" \
  a1e98f2928373c70034c69f37ba366a612c7f7741126ed67506d457aad369673 \
  "${SECOND_OPUS_DIR}/repair_policy_targets.jsonl" \
  33bf539f37beb285459511ee5349f8eec34b8335ff4c07339ce8a95467379cf0 \
  "${AZURE_JOURNAL}" \
  06af6f49ea45d485e6c61b0e4a8b783894ffb4a1491235c56fb2c0428cf0e683 \
  "${AZURE_JOURNAL}.chain-head.json" \
  336874a72569f6a82bbc844260b772e7f3dc631c399e23c979d54502713ea727 \
  "${AZURE_REPORT}" \
  aa22e905037222a34eb01964eb2f6b6a9826ffbb19376490ff1c130a2d8bf18b \
  "${AZURE_DIR}/direct_hard_targets.jsonl" \
  a8c9bc693a27d46c5d83d7b2beb4dddcdae6e1d46d64916d163688de3a3ba557 \
  "${AZURE_DIR}/direct_hard_targets_f2.jsonl" \
  2ff8c7cfef215075fa8e3d2a867084b422ee1e3ed0a04c7059c0b13d3ef5d75c \
  "${AZURE_DIR}/direct_manifest.json" \
  6b2840eb0e270b1bc246350130d9cc1c2de671a66beecc2f1ae506a580dbebdd \
  "${AZURE_DIR}/repair_policy_manifest.json" \
  77cae6c03ca0dd1e80e303afedf2fb551fd1e8ea7ceee0844ecf8448877b423e \
  "${AZURE_DIR}/repair_policy_sources.jsonl" \
  903fd33974f37fb6144267eac84e39f7d5d8ffcf437bf96db79920fd1f9b6924 \
  "${AZURE_DIR}/repair_policy_targets.jsonl" \
  c474ac844acd027a02bf48015447b76a0955052c85c44a9cd698a020e88caef4 \
  "${M3_JOURNAL}" \
  b4794de43b6fd74073754671579a049ecd0e6caa6f3ca5b03c9707f51eb670e8 \
  "${M3_JOURNAL}.chain-head.json" \
  97329655958e0cde43c328990c2d115b749a8f4cb17647b314f819bb0c3fb137 \
  "${M3_REPORT}" \
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
  "${M3_DIR}/direct_hard_targets.jsonl" \
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
  "${M3_DIR}/direct_hard_targets_f2.jsonl" \
  5369a707643d953ef4b13542ad95f9fcd3d606e2d4a1d77eb051c1e7ae2d8b9d \
  "${M3_DIR}/direct_manifest.json" \
  745226bea7c3cddda88c45d48b2efd72709fc828cffb49644723f210b35c6570 \
  "${M3_DIR}/repair_policy_manifest.json" \
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
  "${M3_DIR}/repair_policy_sources.jsonl" \
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
  "${M3_DIR}/repair_policy_targets.jsonl" \
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-api-rs-sft-rescue-report-v1"
   and .status == "complete"
   and .run_contract_sha256
     == "57b1eb70d5c589e4801545b43f9a4a6d26f659ee2e0857d343c09a11b356a2db"
   and .production_floor_eligible == true
   and .heldout_175_opened == false
   and .schedule.eligible_task_offset == 40
   and .schedule.scheduled_tasks == 20
   and .schedule.scheduled_calls == 20
   and .schedule.task_ids_sha256
     == "05c8f8052b820113dfa881c2181982fbca7f007de4df86af2ba2f0d96c0c30c7"
   and .verification.verified_unique_hard_targets == 1
   and .provider.provider == "azure_v1_chat"
   and .provider.model == "gpt-chat-latest"
   and .outputs.direct_targets.sha256
     == "aa22e905037222a34eb01964eb2f6b6a9826ffbb19376490ff1c130a2d8bf18b"
   and .privacy_invariants.private_holdback_sent_to_provider == false
   and .privacy_invariants.gold_sent_to_provider == false' \
  "${AZURE_REPORT}" >/dev/null; then
  echo "T5GEMMA_OPENROUTER_GLM52_BLOCKED exact Azure probe is absent" >&2
  exit 78
fi

# The rejected M3 arm is pinned as immutable negative routing evidence only.
# It is never supplied as a prior-success report and excludes no tasks.
if ! /usr/bin/jq -e \
  '.schema == "t5gemma2-api-rs-sft-rescue-report-v1"
   and .status == "complete"
   and .run_contract_sha256
     == "f1e177ba049a7915a67025f088ab2b268e9985356792271eba42a6679edf7f00"
   and .schedule.eligible_task_offset == 60
   and .schedule.scheduled_tasks == 20
   and .schedule.scheduled_calls == 20
   and .schedule.task_ids_sha256
     == "5bac934b9e0e56a81f19c22aca69d1f0dccdf2a1f7038d443171dfa087f5e14e"
   and .schedule.provider_responses == 0
   and .schedule.code_only_responses == 0
   and .verification.verified_unique_hard_targets == 0
   and .provider.provider == "openrouter_chat"
   and .provider.model == "minimax/minimax-m3"
   and .provider.openrouter_routing.enforce_distillable_text == true
   and .heldout_175_opened == false
   and .privacy_invariants.private_holdback_sent_to_provider == false
   and .privacy_invariants.gold_sent_to_provider == false' \
  "${M3_REPORT}" >/dev/null; then
  echo "T5GEMMA_OPENROUTER_GLM52_BLOCKED exact rejected M3 evidence is absent" >&2
  exit 78
fi

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
  echo "T5GEMMA_OPENROUTER_GLM52_BLOCKED system prompt digest differs" >&2
  exit 78
fi
unset observed_system_prompt_sha256 expected_system_prompt_sha256

if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_OPENROUTER_GLM52_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi
if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_OPENROUTER_GLM52_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi

# Parse only OPENROUTER_API_KEY. Never source the bundle because it may contain
# unrelated provider credentials that must not enter this process.
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
values = []
assignment = re.compile(
    r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)"
)
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
  echo "T5GEMMA_OPENROUTER_GLM52_BLOCKED key parse failed" >&2
  exit 78
fi
export OPENROUTER_API_KEY

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
# At the sealed conservative caps each call reserves at most $0.0704512:
# 49,152 * $0.70/M input + 16,384 * $2.20/M output. Twenty calls reserve
# $1.409024, below the strict $2 arm cap.
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
  --expected_scheduled_task_ids_sha256 380b7dc9603a5da7367859d897ebf312c8660374326cdd187e8d1df0dc7b0f51 \
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
  --openrouter_reasoning_effort xhigh \
  --openrouter_include_reasoning \
  --seed 42 \
  --eligible_task_offset 80 \
  --max_tasks 20 \
  --max_parents_per_task 1 \
  --samples_per_parent 1 \
  --max_calls 20 \
  --max_input_tokens_per_call 49152 \
  --max_output_tokens 16384 \
  --max_input_tokens_total 983040 \
  --max_output_tokens_total 327680 \
  --max_total_tokens 1310720 \
  --max_usd 2 \
  --input_usd_per_million 0.70 \
  --output_usd_per_million 2.20 \
  --timeout_seconds 600 \
  --timeout 30 \
  --stability_runs 2
