#!/usr/bin/env bash
set -euo pipefail

# A credential-free plan is sealed before the Anthropic key is read.  The
# live run then has to reproduce the exact 16-task schedule digest.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SCRIPT="${PROJECT}/scripts/training/t5gemma2_typed_opus_retry_harvest.py"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
LOCAL_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1"
EXISTING_MANIFEST="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1/dataset_manifest.json"
SPLIT_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_api_visible_split_v1"
PROJECTION_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_visible_failure_projection_v1"
CASCADE_DIR="${WORKSPACE}/artifacts/t5gemma2_typed_dual_api_rescue_v1"
OUTPUT_DIR="${T5GEMMA_TYPED_OPUS_RETRY_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_opus_retry_16_v2}"
PLAN_PATH="${OUTPUT_DIR}/opus_retry_plan.json"
SECRET_FILE="${T5GEMMA_ANTHROPIC_ENV:-${WORKSPACE}/secrets/Anthropic.env}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
EXPECTED_SCHEDULE_SHA=839ac9fa414f0434b68561f76acb3f3a0ca3a0f4a8588a476e0d09684a5428fe

blocked() {
  echo "T5GEMMA_TYPED_OPUS_RETRY_BLOCKED $*" >&2
  exit 78
}

printf '%s  %s\n' \
  c6694b9c44d43432082d3476878df1fb34bdf8daffb0c1956cc39c155d23f476 "${SCRIPT}" \
  7a03af003e998497012706361f5cbf0734d8defa82c7e458aa5f87f796e01143 "${PROJECT}/scripts/training/t5gemma2_typed_api_rescue_cascade.py" \
  ad681aaa68db63dbc64ce847f32f18e2740e4db2050d1211e5d5457fdc6dff69 "${PROJECT}/scripts/training/t5gemma2_api_rs_sft_rescue.py" \
  | sha256sum -c - || blocked "harvest code differs"

printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1 "${LOCAL_DIR}/harvest_report.json" \
  ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc "${LOCAL_DIR}/harvest.journal.jsonl" \
  c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4 "${LOCAL_DIR}/direct_targets.jsonl" \
  1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3 "${EXISTING_MANIFEST}" \
  d9694f084c694c6d1f3cc85ffa00b06d91bb953e97f3e5f8b8f74f5708e0afcc "${SPLIT_DIR}/split_manifest.json" \
  0f6054d688d1fdd9b7f332cef703ae7ff0c00956f57c47944223c9883055ad23 "${SPLIT_DIR}/visible_train.jsonl" \
  419917fceb8fd98849333309270277b412c877f4b2d7205976532bd532d1494b "${SPLIT_DIR}/holdback.private.jsonl" \
  2f2ef4a288da49f47fdd659576a1dda67836bdd069bd9546c8b5b1e479a3426c "${PROJECTION_DIR}/visible_projection_report.json" \
  1359c20028418a1a678c70364b6bf522338ac95e9169a8156626bd32af9b8502 "${PROJECTION_DIR}/visible_projection.journal.jsonl" \
  992d8565e8975a0802df8980e7e252a35203466ed58da69d5d358dd0edebb58b "${CASCADE_DIR}/kimi_initial_c000/typed_api_rescue_report.json" \
  ec222eb97f1993b87e06163ca027b5bc6040243f112e4b284b9042c0c4085f75 "${CASCADE_DIR}/kimi_retry_c000/typed_api_rescue_report.json" \
  0b0eabe2ba10d9da0012bc08faec112f747fc84cbd440f309c8af672b5ae7620 "${CASCADE_DIR}/sonnet_residual_c000/typed_api_rescue_report.json" \
  | sha256sum -c - || blocked "sealed input lineage differs"

[[ -x "${DART_BIN}" ]] || blocked "Dart 3.12.2 is absent"
[[ -x /venv/main/bin/python ]] || blocked "Python runtime is absent"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}"
export CUDA_VISIBLE_DEVICES=""
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"

common_args=(
  --local_harvest_report "${LOCAL_DIR}/harvest_report.json"
  --expected_local_harvest_report_sha256 1d2daa386ba20b2a86f6585719d23fadce7a0af1754a4f33e0a3f9ed324eb2b1
  --pilot_journal "${LOCAL_DIR}/harvest.journal.jsonl"
  --expected_local_harvest_journal_sha256 ed876d6ddf1cc624f8f1ab7b0de8e739b7d40578e95f10a200a890535fdfaebc
  --local_harvest_targets "${LOCAL_DIR}/direct_targets.jsonl"
  --expected_local_harvest_targets_sha256 c7c9df735370c99a2cb305f466c18b2bd947d6152538abb022b5f76b2046cfc4
  --existing_direct_manifest "${EXISTING_MANIFEST}"
  --expected_existing_direct_manifest_sha256 1a6c660f8d7f08ab21d963537386c166cd69b9191b6f6231198174cf5354b9c3
  --gold_train_jsonl "${GOLD_TRAIN}"
  --expected_gold_train_sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904
  --gold_f2_jsonl "${GOLD_F2}" --f2_jsonl "${GOLD_F2}"
  --expected_gold_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe
  --expected_f2_sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe
  --heldout_jsonl "${HELDOUT}"
  --expected_heldout_sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7
  --visible_split_manifest "${SPLIT_DIR}/split_manifest.json"
  --expected_visible_split_manifest_sha256 d9694f084c694c6d1f3cc85ffa00b06d91bb953e97f3e5f8b8f74f5708e0afcc
  --visible_train "${SPLIT_DIR}/visible_train.jsonl" --rollout_file "${SPLIT_DIR}/visible_train.jsonl"
  --expected_visible_train_sha256 0f6054d688d1fdd9b7f332cef703ae7ff0c00956f57c47944223c9883055ad23
  --expected_rollout_sha256 0f6054d688d1fdd9b7f332cef703ae7ff0c00956f57c47944223c9883055ad23
  --private_split_holdback "${SPLIT_DIR}/holdback.private.jsonl"
  --private_holdback "${SPLIT_DIR}/holdback.private.jsonl"
  --expected_private_split_holdback_sha256 419917fceb8fd98849333309270277b412c877f4b2d7205976532bd532d1494b
  --expected_private_holdback_sha256 419917fceb8fd98849333309270277b412c877f4b2d7205976532bd532d1494b
  --visible_projection_report "${PROJECTION_DIR}/visible_projection_report.json"
  --expected_visible_projection_report_sha256 2f2ef4a288da49f47fdd659576a1dda67836bdd069bd9546c8b5b1e479a3426c
  --visible_projection_journal "${PROJECTION_DIR}/visible_projection.journal.jsonl"
  --expected_visible_projection_journal_sha256 1359c20028418a1a678c70364b6bf522338ac95e9169a8156626bd32af9b8502
  --prior_success_report "${CASCADE_DIR}/kimi_initial_c000/typed_api_rescue_report.json"
  --expected_prior_success_report_sha256 992d8565e8975a0802df8980e7e252a35203466ed58da69d5d358dd0edebb58b
  --prior_success_report "${CASCADE_DIR}/kimi_retry_c000/typed_api_rescue_report.json"
  --expected_prior_success_report_sha256 ec222eb97f1993b87e06163ca027b5bc6040243f112e4b284b9042c0c4085f75
  --prior_success_report "${CASCADE_DIR}/sonnet_residual_c000/typed_api_rescue_report.json"
  --expected_prior_success_report_sha256 0b0eabe2ba10d9da0012bc08faec112f747fc84cbd440f309c8af672b5ae7620
  --output_dir "${OUTPUT_DIR}"
  --provider anthropic --model claude-opus-5
  --base_url https://api.anthropic.com --api_key_env ANTHROPIC_API_KEY
  --anthropic_thinking adaptive --anthropic_effort high
  --seed 20260801 --max_tasks 16
  --max_parents_per_task 1 --samples_per_parent 1 --max_calls 16
  --max_input_tokens_per_call 32768 --max_output_tokens 8192
  --max_input_tokens_total 524288 --max_output_tokens_total 131072
  --max_total_tokens 655360 --max_usd 5.89824
  --input_usd_per_million 5 --output_usd_per_million 25
  --timeout_seconds 900 --inter_call_delay_seconds 2
  --abort_on_provider_error --provider_max_attempts 8
  --provider_retry_base_seconds 2 --provider_retry_max_seconds 30
  --timeout 30 --stability_runs 2
)

# Planning imports and validates all lineage but never reads a credential and
# never calls Anthropic.
/venv/main/bin/python "${SCRIPT}" "${common_args[@]}" \
  --plan_only_output "${PLAN_PATH}"
/usr/bin/jq -e \
  --arg digest "${EXPECTED_SCHEDULE_SHA}" \
  '.schema == "t5gemma2-typed-api-rescue-cascade-plan-v1"
   and .status == "complete"
   and .phase == "opus_retry"
   and .selection.scheduled_tasks == 16
   and .selection.scheduled_calls == 16
   and .selection.retry_tasks == 16
   and .selection.retry_task_ids_sha256 == $digest
   and .selection.task_ids_sha256 == $digest
   and .selection.targeted_non_code_or_length_only == true
   and .selection.prior_verified_excluded == true
   and .selection.selection_uses_heldout_175 == false
   and .selection.max_prompt_byte_upper_bound <= 32768
   and .budget.max_input_tokens_per_call == 32768
   and .budget.max_output_tokens_per_call == 8192
   and .budget.max_usd == "5.89824"
   and .provider_credentials_read == false
   and .frontier_api_calls == false' \
  "${PLAN_PATH}" >/dev/null || blocked "credential-free Opus plan differs"

[[ -s "${SECRET_FILE}" ]] || blocked "Anthropic secret file is absent"
anthropic_key="$(/venv/main/bin/python - "${SECRET_FILE}" <<'PY'
import re, stat, sys
from pathlib import Path
p=Path(sys.argv[1])
if stat.S_IMODE(p.stat().st_mode) & 0o077:
    raise SystemExit("Anthropic.env must be mode 0600")
raw=p.read_bytes()
try: text=raw.decode("utf-8-sig")
except UnicodeDecodeError: text=raw.decode("utf-16")
values=[]
for line in text.splitlines():
    line=line.strip()
    if not line or line.startswith("#"): continue
    m=re.fullmatch(r"(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.*)", line, re.I)
    if m:
        value=m.group(1).strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in "\"'": value=value[1:-1]
        values.append(value)
if len(values)!=1 or not values[0] or any(c.isspace() for c in values[0]):
    raise SystemExit("ANTHROPIC_API_KEY must occur exactly once and be well formed")
print(values[0], end="")
PY
)"
export ANTHROPIC_API_KEY="${anthropic_key}"
unset anthropic_key

exec nice -n 10 /venv/main/bin/python "${SCRIPT}" "${common_args[@]}" \
  --expected_scheduled_task_ids_sha256 "${EXPECTED_SCHEDULE_SHA}"
