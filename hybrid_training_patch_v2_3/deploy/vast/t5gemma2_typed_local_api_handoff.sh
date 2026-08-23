#!/usr/bin/env bash
set -euo pipefail

# Atomic Supervisor handoff:
#   completed typed local K=4 harvest
#     -> deep independent audit
#     -> CPU-only visible split/projection
#     -> deep projection/schedule audit
#     -> durable Kimi cohort 0 / targeted retry / Sonnet residual controller
#
# No provider credential is read until the final exec reaches the already
# fail-closed API launcher.  Any incomplete, mutable, or inconsistent local
# artifact exits 78 before a provider process can be created.

WORKSPACE="${T5GEMMA_TYPED_HANDOFF_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
GOLD_DIR="${WORKSPACE}/multifunction_v1/expanded2776/build"
GOLD_TRAIN="${GOLD_DIR}/train_multifunction_binary_expanded_2776.jsonl"
GOLD_F2="${GOLD_DIR}/train_multifunction_binary_expanded_2776_f2.jsonl"
HELDOUT="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
LOCAL_DIR="${T5GEMMA_TYPED_LOCAL_HARVEST_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_local_direct_harvest_rs58_k4_v1}"
EXISTING_MANIFEST="${T5GEMMA_TYPED_225_MANIFEST:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1/dataset_manifest.json}"
SPLIT_DIR="${T5GEMMA_TYPED_API_SPLIT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_api_visible_split_v1}"
PROJECTION_DIR="${T5GEMMA_TYPED_API_PROJECTION_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_visible_failure_projection_v1}"
AUDIT_DIR="${T5GEMMA_TYPED_API_HANDOFF_AUDIT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_local_api_handoff_audit_v1}"
DUAL_API_OUTPUT_ROOT="${T5GEMMA_TYPED_DUAL_API_OUTPUT_ROOT:-${WORKSPACE}/artifacts/t5gemma2_typed_dual_api_rescue_v1}"

LOCAL_REPORT="${LOCAL_DIR}/harvest_report.json"
LOCAL_JOURNAL="${LOCAL_DIR}/harvest.journal.jsonl"
LOCAL_CHAIN_HEAD="${LOCAL_JOURNAL}.chain-head.json"
LOCAL_TARGETS="${LOCAL_DIR}/direct_targets.jsonl"
LOCAL_F2="${LOCAL_DIR}/direct_f2.jsonl"
LOCAL_SCHEDULE="${LOCAL_DIR}/schedule_manifest.jsonl"
SPLIT_MANIFEST="${SPLIT_DIR}/split_manifest.json"
VISIBLE_TRAIN="${SPLIT_DIR}/visible_train.jsonl"
PRIVATE_SPLIT="${SPLIT_DIR}/holdback.private.jsonl"
PROJECTION_REPORT="${PROJECTION_DIR}/visible_projection_report.json"
PROJECTION_JOURNAL="${PROJECTION_DIR}/visible_projection.journal.jsonl"
HARVEST_AUDIT="${AUDIT_DIR}/typed_local_harvest_audit.json"
PROJECTION_AUDIT="${AUDIT_DIR}/typed_visible_projection_audit.json"

AUDITOR="${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_api_handoff.py"
PREPARE_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_api_rescue_prepare.sh"
CASCADE_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_api_rescue_cascade.sh"
DUAL_API_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_dual_api_orchestrator.sh"
PYTHON_BIN="${T5GEMMA_TYPED_API_HANDOFF_PYTHON:-/venv/main/bin/python}"
SUPERVISORCTL="${T5GEMMA_TYPED_API_HANDOFF_SUPERVISORCTL:-supervisorctl}"
HARVEST_PROGRAM="${T5GEMMA_TYPED_API_HANDOFF_HARVEST_PROGRAM:-t5gemma2-typed-local-direct-harvest}"
POLL_SECONDS="${T5GEMMA_TYPED_API_HANDOFF_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_TYPED_API_HANDOFF_STABILITY_SECONDS:-2}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

blocked() {
  echo "T5GEMMA_TYPED_LOCAL_API_HANDOFF_BLOCKED $*" >&2
  exit 78
}

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  blocked "poll/stability interval is invalid"
fi
if ! [[ "${HARVEST_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  blocked "harvest Supervisor program name is invalid"
fi
for executable in "${PYTHON_BIN}" "${DART_BIN}" "${PREPARE_LAUNCHER}" "${CASCADE_LAUNCHER}" "${DUAL_API_LAUNCHER}"; do
  if [[ ! -x "${executable}" ]]; then
    blocked "required executable is absent: ${executable}"
  fi
done
if [[ ! -s "${AUDITOR}" ]]; then
  blocked "audit program is absent: ${AUDITOR}"
fi
printf '%s  %s\n' \
  3714c845574bf3eae8250d79078c34ac009a8f07460ddf767ee0fa6d5f0add33 "${AUDITOR}" \
  a2c2c3adaa96467a8de1025697222c11146ef94fe63c0f85e9cfbf8beebd753c "${PREPARE_LAUNCHER}" \
  c69e845cfefcd91555171813a66492dba0b2b5c9d44bbd8efd21175f5f7f2e14 "${CASCADE_LAUNCHER}" \
  83fb363aa04f9f8993d44d8b085707897699f0eeebdd0c4539b279184b8b2796 "${DUAL_API_LAUNCHER}" \
  875517222f2aa3a1cd823d476b44cd51f49fb2a7dff8f2e4a5cb18466622264a "${PROJECT}/scripts/training/t5gemma2_typed_local_direct_harvest.py" \
  | sha256sum -c - || blocked "handoff/audit/phase launcher code differs"

# These inputs are immutable study constants; the late-bound values below are
# limited to the output artifacts whose bytes cannot be known before harvest.
printf '%s  %s\n' \
  fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 "${GOLD_TRAIN}" \
  94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe "${GOLD_F2}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${HELDOUT}" \
  | sha256sum -c - || blocked "sealed TRAIN/heldout input differs"

while true; do
  status_line="$("${SUPERVISORCTL}" status "${HARVEST_PROGRAM}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING)
      sleep "${POLL_SECONDS}"
      ;;
    EXITED)
      break
      ;;
    STOPPED)
      blocked "harvest was stopped rather than completed"
      ;;
    FATAL|BACKOFF|UNKNOWN|"")
      blocked "harvest Supervisor state=${state:-missing}"
      ;;
    *)
      blocked "unexpected harvest Supervisor state=${state}"
      ;;
  esac
done

sealed_files=(
  "${LOCAL_REPORT}"
  "${LOCAL_JOURNAL}"
  "${LOCAL_CHAIN_HEAD}"
  "${LOCAL_TARGETS}"
  "${LOCAL_F2}"
  "${LOCAL_SCHEDULE}"
  "${EXISTING_MANIFEST}"
)
for required in "${sealed_files[@]}"; do
  if [[ ! -f "${required}" ]]; then
    blocked "harvest exited without sealed artifact ${required}"
  fi
done

# Require two byte-identical observations after EXITED.  This rejects a stale
# report paired with a journal or target file that is still being modified.
snapshot_one="$(sha256sum "${sealed_files[@]}")"
sleep "${STABILITY_SECONDS}"
snapshot_two="$(sha256sum "${sealed_files[@]}")"
if [[ "${snapshot_one}" != "${snapshot_two}" ]]; then
  blocked "harvest artifact digests changed after Supervisor EXITED"
fi
unset snapshot_one snapshot_two

local_report_sha="$(sha256sum "${LOCAL_REPORT}" | awk '{print $1}')"
local_journal_sha="$(sha256sum "${LOCAL_JOURNAL}" | awk '{print $1}')"
local_targets_sha="$(sha256sum "${LOCAL_TARGETS}" | awk '{print $1}')"
manifest_sha="$(sha256sum "${EXISTING_MANIFEST}" | awk '{print $1}')"
for digest in "${local_report_sha}" "${local_journal_sha}" "${local_targets_sha}" "${manifest_sha}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || blocked "late-bound harvest digest is malformed"
done

mkdir -p "${AUDIT_DIR}"
export PYTHONPATH="${PROJECT}"
export DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
if ! "${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_api_handoff.py harvest \
  --local-harvest-report "${LOCAL_REPORT}" \
  --expected-local-harvest-report-sha256 "${local_report_sha}" \
  --local-harvest-journal "${LOCAL_JOURNAL}" \
  --expected-local-harvest-journal-sha256 "${local_journal_sha}" \
  --local-harvest-targets "${LOCAL_TARGETS}" \
  --expected-local-harvest-targets-sha256 "${local_targets_sha}" \
  --existing-direct-manifest "${EXISTING_MANIFEST}" \
  --expected-existing-direct-manifest-sha256 "${manifest_sha}" \
  --gold-train-jsonl "${GOLD_TRAIN}" \
  --expected-gold-train-sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --gold-f2-jsonl "${GOLD_F2}" \
  --expected-gold-f2-sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --heldout-jsonl "${HELDOUT}" \
  --expected-heldout-sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --timeout 30 --evaluation-workers 16 --output "${HARVEST_AUDIT}"; then
  blocked "completed harvest failed deep audit; no API phase launched"
fi
if ! /usr/bin/jq -e '
  .schema == "t5gemma2-typed-local-harvest-handoff-audit-v1"
  and .status == "pass"
  and .lineage.checkpoint_stage == "typed_direct"
  and .lineage.checkpoint_update == 58
  and .schedule.clean_train_tasks == 2775
  and .schedule.excluded_previous_direct_tasks == 225
  and .schedule.scheduled_tasks == 2550
  and .schedule.samples_per_task == 4
  and .schedule.terminal_events == 2550
  and .accepted.independently_reverified == .accepted.direct_targets
  and .accepted.independent_failures == 0
  and .privacy.provider_credentials_read == false
  and .privacy.frontier_api_calls == false' "${HARVEST_AUDIT}" >/dev/null; then
  blocked "harvest audit gate differs; no API phase launched"
fi

export T5GEMMA_TYPED_LOCAL_REPORT_SHA256="${local_report_sha}"
export T5GEMMA_TYPED_LOCAL_JOURNAL_SHA256="${local_journal_sha}"
export T5GEMMA_TYPED_LOCAL_TARGETS_SHA256="${local_targets_sha}"
export T5GEMMA_TYPED_225_MANIFEST_SHA256="${manifest_sha}"
export T5GEMMA_TYPED_LOCAL_HARVEST_OUTPUT_DIR="${LOCAL_DIR}"
export T5GEMMA_TYPED_225_MANIFEST="${EXISTING_MANIFEST}"
export T5GEMMA_TYPED_API_SPLIT_DIR="${SPLIT_DIR}"
export T5GEMMA_TYPED_API_PROJECTION_DIR="${PROJECTION_DIR}"

# CPU-only.  A complete existing projection is audited in place and never
# regenerated.  An absent/partial projection is resumed by the credential-free
# prepare launcher.
projection_files=(
  "${SPLIT_MANIFEST}"
  "${VISIBLE_TRAIN}"
  "${PRIVATE_SPLIT}"
  "${PROJECTION_REPORT}"
  "${PROJECTION_JOURNAL}"
  "${PROJECTION_JOURNAL}.chain-head.json"
)
projection_complete=true
for required in "${projection_files[@]}"; do
  [[ -f "${required}" ]] || projection_complete=false
done
if [[ "${projection_complete}" != true ]] && ! "${PREPARE_LAUNCHER}"; then
  blocked "visible split/projection preparation failed; no API phase launched"
fi

for required in "${projection_files[@]}"; do
  if [[ ! -f "${required}" ]]; then
    blocked "prepare exited without sealed artifact ${required}"
  fi
done
split_manifest_sha="$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')"
visible_sha="$(sha256sum "${VISIBLE_TRAIN}" | awk '{print $1}')"
private_sha="$(sha256sum "${PRIVATE_SPLIT}" | awk '{print $1}')"
projection_report_sha="$(sha256sum "${PROJECTION_REPORT}" | awk '{print $1}')"
projection_journal_sha="$(sha256sum "${PROJECTION_JOURNAL}" | awk '{print $1}')"

if ! "${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_api_handoff.py projection \
  --local-harvest-report "${LOCAL_REPORT}" \
  --expected-local-harvest-report-sha256 "${local_report_sha}" \
  --local-harvest-journal "${LOCAL_JOURNAL}" \
  --expected-local-harvest-journal-sha256 "${local_journal_sha}" \
  --local-harvest-targets "${LOCAL_TARGETS}" \
  --expected-local-harvest-targets-sha256 "${local_targets_sha}" \
  --existing-direct-manifest "${EXISTING_MANIFEST}" \
  --expected-existing-direct-manifest-sha256 "${manifest_sha}" \
  --gold-train-jsonl "${GOLD_TRAIN}" \
  --expected-gold-train-sha256 fbfa6bc2a26e9d062352e9fcd508262b07af7fbe019cd1876cf6dd0875f4e904 \
  --gold-f2-jsonl "${GOLD_F2}" \
  --expected-gold-f2-sha256 94bea0ce81db113b346375568ead3cebe34f7a4d33e6c33fce4e994b7e0919fe \
  --heldout-jsonl "${HELDOUT}" \
  --expected-heldout-sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  --visible-split-manifest "${SPLIT_MANIFEST}" \
  --expected-visible-split-manifest-sha256 "${split_manifest_sha}" \
  --visible-train "${VISIBLE_TRAIN}" \
  --expected-visible-train-sha256 "${visible_sha}" \
  --private-split-holdback "${PRIVATE_SPLIT}" \
  --expected-private-split-holdback-sha256 "${private_sha}" \
  --visible-projection-report "${PROJECTION_REPORT}" \
  --expected-visible-projection-report-sha256 "${projection_report_sha}" \
  --visible-projection-journal "${PROJECTION_JOURNAL}" \
  --expected-visible-projection-journal-sha256 "${projection_journal_sha}" \
  --output "${PROJECTION_AUDIT}"; then
  blocked "visible projection failed deep audit; no API phase launched"
fi
if ! /usr/bin/jq -e '
  .schema == "t5gemma2-typed-api-projection-handoff-audit-v1"
  and .status == "pass"
  and .phase == "kimi_initial"
  and .cohort_index == 0
  and .projection.tasks == 2550
  and .first_cohort.tasks == 50
  and .first_cohort.one_parent_per_task == true
  and .first_cohort.calls_reserved == 50
  and .privacy.eligibility_uses_visible_train_split_only == true
  and .privacy.private_complete_outcome_used_for_eligibility == false
  and .privacy.private_complete_diagnostic_used == false
  and .privacy.provider_credentials_read == false
  and .privacy.frontier_api_calls == false' "${PROJECTION_AUDIT}" >/dev/null; then
  blocked "projection audit gate differs; no API phase launched"
fi

schedule_sha="$(/usr/bin/jq -r '.first_cohort.task_ids_sha256 // empty' "${PROJECTION_AUDIT}")"
[[ "${schedule_sha}" =~ ^[0-9a-f]{64}$ ]] \
  || blocked "projection audit lacks exact cohort schedule digest"

export T5GEMMA_TYPED_API_SPLIT_MANIFEST_SHA256="${split_manifest_sha}"
export T5GEMMA_TYPED_API_VISIBLE_SHA256="${visible_sha}"
export T5GEMMA_TYPED_API_PRIVATE_SPLIT_SHA256="${private_sha}"
export T5GEMMA_TYPED_API_PROJECTION_REPORT_SHA256="${projection_report_sha}"
export T5GEMMA_TYPED_API_PROJECTION_JOURNAL_SHA256="${projection_journal_sha}"
export T5GEMMA_TYPED_API_SCHEDULE_SHA256="${schedule_sha}"
export T5GEMMA_TYPED_DUAL_API_OUTPUT_ROOT="${DUAL_API_OUTPUT_ROOT}"

echo "T5GEMMA_TYPED_LOCAL_API_HANDOFF_SEALED harvest_audit=${HARVEST_AUDIT} projection_audit=${PROJECTION_AUDIT} schedule_sha256=${schedule_sha}"
echo "T5GEMMA_TYPED_LOCAL_API_HANDOFF_STARTING controller=dual_api kimi_tasks=50 sonnet_max_tasks=38"
exec "${DUAL_API_LAUNCHER}"
