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
OUTPUT_DIR="${T5GEMMA_AZURE_RESIDUAL_PROBE_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_api_rs_sft_azure_production_residual_probe_2epoch_v1}"
SECRET_FILE="${T5GEMMA_AZURE_ENV:-${WORKSPACE}/secrets/Azure.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
RESCUE_SCRIPT="${PROJECT}/scripts/training/t5gemma2_api_rs_sft_rescue.py"

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

# The first Opus tranche is evidence for both sequentiality and the
# $11.758550 actual-spend baseline. Validate it without feeding it into the
# generic success exclusion: offset 20 skips its entire schedule, including
# all three successes, so the second tranche remains exactly positions 20..39
# of the Sonnet-only residual.
if [[ ! -s "${FIRST_OPUS_JOURNAL}" ]] \
  || [[ ! -s "${FIRST_OPUS_JOURNAL}.chain-head.json" ]] \
  || [[ ! -s "${FIRST_OPUS_REPORT}" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-api-rs-sft-rescue-report-v1"
     and .status == "complete"
     and .run_contract_sha256
       == "04fe2c2841574b16dfe412ce0732d4eada52fdc9cf25cf0528f2891e5660d206"
     and .production_floor_eligible == true
     and .heldout_175_opened == false
     and .schedule.eligible_task_offset == 0
     and .schedule.scheduled_tasks == 20
     and .schedule.scheduled_calls == 20
     and .schedule.task_ids_sha256
       == "9ddf1a24954de70810a94cf44ae0b1ebc8fc13ca50703a2b1715325855451f8f"
     and .verification.verified_unique_hard_targets == 3
     and .budget_charged.calls == 20
     and .budget_charged.estimated_usd == "1.776690000"
     and .provider.provider == "anthropic"
     and .provider.model == "claude-opus-5"
     and .provider.thinking == "adaptive"
     and .provider.effort == "high"
     and .provider.max_output_tokens == 8192
     and .outputs.direct_targets.sha256
       == "15ef808838ed01347e646e9b4462f48ae88d4afcb467d144f6c6283576abf180"
     and .privacy_invariants.private_holdback_sent_to_provider == false
     and .privacy_invariants.gold_sent_to_provider == false' \
    "${FIRST_OPUS_REPORT}" >/dev/null \
  || ! /usr/bin/head -n 1 "${FIRST_OPUS_JOURNAL}" \
    | /usr/bin/jq -e \
      '.event == "header"
       and .contract_sha256
         == "04fe2c2841574b16dfe412ce0732d4eada52fdc9cf25cf0528f2891e5660d206"
       and .contract.selection.seed == 42
       and .contract.selection.eligible_task_offset == 0
       and .contract.selection.scheduled_tasks == 20
       and .contract.selection.scheduled_slots == 20
       and .contract.selection.task_ids_sha256
         == "9ddf1a24954de70810a94cf44ae0b1ebc8fc13ca50703a2b1715325855451f8f"
       and .contract.verification.all_api_calls_before_any_private_gate == true
       and .contract.verification.private_failure_triggers_api_call == false' \
      >/dev/null; then
  echo "T5GEMMA_CLAUDE_OPUS_RESIDUAL_TRANCHE2_BLOCKED exact first Opus tranche is absent" >&2
  exit 78
fi

if [[ ! -s "${SECOND_OPUS_JOURNAL}" ]] \
  || [[ ! -s "${SECOND_OPUS_JOURNAL}.chain-head.json" ]] \
  || [[ ! -s "${SECOND_OPUS_REPORT}" ]] \
  || ! /usr/bin/jq -e \
    '.schema == "t5gemma2-api-rs-sft-rescue-report-v1"
     and .status == "complete"
     and .run_contract_sha256
       == "cb369617ec52f8b120b3852fceed326ef50b0a389d15d2f1a4b9d6d30999b2b3"
     and .production_floor_eligible == true
     and .heldout_175_opened == false
     and .schedule.eligible_task_offset == 20
     and .schedule.scheduled_tasks == 20
     and .schedule.scheduled_calls == 20
     and .schedule.task_ids_sha256
       == "f6d83bab2b4ff9dcb4a8f0ba1c1935b6dd36a79b044ca576bb6940aaa10e8655"
     and .verification.verified_unique_hard_targets == 2
     and .budget_charged.calls == 20
     and .budget_charged.estimated_usd == "5.823850000"
     and .provider.provider == "anthropic"
     and .provider.model == "claude-opus-5"
     and .provider.thinking == "adaptive"
     and .provider.effort == "high"
     and .provider.max_output_tokens == 16384
     and .outputs.direct_targets.sha256
       == "2e02a7db60d0baf9d64afdc9b5bb211fcd0253186b490e9af911de8d49b87bf7"
     and .outputs.direct_f2.sha256
       == "5fd2d1cd56b0c2de0ed79fd4cfdb3017244774f038517e574a7089a39ea51a91"
     and .privacy_invariants.private_holdback_sent_to_provider == false
     and .privacy_invariants.gold_sent_to_provider == false' \
    "${SECOND_OPUS_REPORT}" >/dev/null \
  || ! /usr/bin/head -n 1 "${SECOND_OPUS_JOURNAL}" \
    | /usr/bin/jq -e \
      '.event == "header"
       and .contract_sha256
         == "cb369617ec52f8b120b3852fceed326ef50b0a389d15d2f1a4b9d6d30999b2b3"
       and .contract.selection.seed == 42
       and .contract.selection.eligible_task_offset == 20
       and .contract.selection.scheduled_tasks == 20
       and .contract.selection.scheduled_slots == 20
       and .contract.selection.task_ids_sha256
         == "f6d83bab2b4ff9dcb4a8f0ba1c1935b6dd36a79b044ca576bb6940aaa10e8655"
       and .contract.verification.all_api_calls_before_any_private_gate == true
       and .contract.verification.private_failure_triggers_api_call == false' \
      >/dev/null; then
  echo "T5GEMMA_AZURE_RESIDUAL_BLOCKED exact second Opus tranche is absent" >&2
  exit 78
fi

printf '%s  %s\n' \
  4900c5704f1488a55369d2149f637ce8c2346443330524f0a5c58b0a16820ad8 \
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
  14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c \
  "${FEEDBACK_DIR}/verpo_rollout_feedback.jsonl" \
  c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3 \
  "${FEEDBACK_DIR}/verpo_teacher_f2.jsonl" \
  dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f \
  "${FEEDBACK_DIR}/reward_holdback.private.jsonl" \
  | sha256sum -c -

# Seal the user-authorized synthetic-benchmark system prompt independently
# from the containing script so prompt drift cannot silently change this arm.
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
  echo "T5GEMMA_AZURE_RESIDUAL_BLOCKED system prompt digest differs" >&2
  exit 78
fi
unset observed_system_prompt_sha256 expected_system_prompt_sha256

if [[ ! -x "${DART_BIN}" ]]; then
  echo "T5GEMMA_AZURE_RESIDUAL_BLOCKED Dart 3.12.2 is not executable" >&2
  exit 78
fi
if [[ ! -s "${SECRET_FILE}" ]]; then
  echo "T5GEMMA_AZURE_RESIDUAL_BLOCKED missing ${SECRET_FILE}" >&2
  exit 78
fi

# Parse only the two required Azure assignments. Never source the file: it may
# contain unrelated credentials, and those must not enter this process.
mapfile -t azure_secret_values < <(
  /venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import re
import stat
import sys
from urllib.parse import urlparse
from pathlib import Path

path = Path(sys.argv[1])
if stat.S_IMODE(path.stat().st_mode) & 0o077:
    raise SystemExit("Azure.env must not be group/world accessible")
raw = path.read_bytes()
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("utf-16")
targets = {"AZURE_OPENAI_ENDPOINT": [], "AZURE_OPENAI_API_KEY": []}
assignment = re.compile(
    r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)"
)
for line_number, raw_line in enumerate(text.splitlines(), 1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    match = assignment.fullmatch(line)
    if not match:
        if any(name in line for name in targets):
            raise SystemExit(
                f"malformed targeted Azure assignment at line {line_number}"
            )
        continue
    name, value = match.groups()
    if name not in targets:
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value or "\n" in value or "\r" in value:
        raise SystemExit(f"{name} is empty or malformed")
    targets[name].append(value)
for name, values in targets.items():
    if len(values) != 1:
        raise SystemExit(f"{name} must occur exactly once")

endpoint = targets["AZURE_OPENAI_ENDPOINT"][0].rstrip("/")
parsed = urlparse(endpoint)
if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
    raise SystemExit("AZURE_OPENAI_ENDPOINT must be an HTTPS endpoint")
if any(char.isspace() for char in endpoint):
    raise SystemExit("AZURE_OPENAI_ENDPOINT contains whitespace")
if not endpoint.endswith("/openai/v1"):
    endpoint += "/openai/v1"
key = targets["AZURE_OPENAI_API_KEY"][0]
if any(char.isspace() for char in key):
    raise SystemExit("AZURE_OPENAI_API_KEY contains whitespace")
print(endpoint)
print(key)
PY
)
if [[ "${#azure_secret_values[@]}" -ne 2 ]]; then
  echo "T5GEMMA_AZURE_RESIDUAL_BLOCKED Azure secret parse failed" >&2
  exit 78
fi
azure_base_url="${azure_secret_values[0]}"
export AZURE_OPENAI_API_KEY="${azure_secret_values[1]}"
unset azure_secret_values

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"

cd "${PROJECT}"
# One Azure OpenAI call is paired with residual positions 40..59 after
# excluding the exact 65 Sonnet successes. At the conservative $5/M input and
# $25/M output bounds, 20 full 49,152-in/8,192-out calls reserve $9.011200.
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
  --expected_scheduled_task_ids_sha256 05c8f8052b820113dfa881c2181982fbca7f007de4df86af2ba2f0d96c0c30c7 \
  --output_dir "${OUTPUT_DIR}" \
  --provider azure_v1_chat \
  --model gpt-chat-latest \
  --base_url "${azure_base_url}" \
  --api_key_env AZURE_OPENAI_API_KEY \
  --chat_token_parameter max_completion_tokens \
  --seed 42 \
  --eligible_task_offset 40 \
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
