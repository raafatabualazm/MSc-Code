#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_PASS3_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SUPERVISORCTL="${T5GEMMA_TYPED_PASS3_SUPERVISORCTL:-supervisorctl}"
C002_PROGRAM="${T5GEMMA_TYPED_C002_PROGRAM:-t5gemma2-typed-kimi-c002-resume47}"
C002_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_c002_resume47_v2"
PREFIX_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_c002_prefix3_verification_v1"
PREFIX_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_c002_prefix3_verify.sh"
TRAIN_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_direct_rs_sft_pass3.sh"
EVAL_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_direct_rs_sft_pass3_eval_gate.sh"
CHECKPOINT_MANIFEST="${T5GEMMA_TYPED_PASS3_CHECKPOINT_MANIFEST:-${WORKSPACE}/artifacts/t5gemma2_typed_pass3_checkpoint_manifest_v1.json}"
POLL_SECONDS="${T5GEMMA_TYPED_PASS3_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_TYPED_PASS3_STABILITY_SECONDS:-2}"

blocked() { echo "T5GEMMA_TYPED_PASS3_HANDOFF_BLOCKED $*" >&2; exit 78; }
[[ "${C002_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ && "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ && "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || blocked "handoff configuration is malformed"
for launcher in "${PREFIX_LAUNCHER}" "${TRAIN_LAUNCHER}" "${EVAL_LAUNCHER}"; do
  [[ -x "${launcher}" ]] || blocked "launcher is absent: ${launcher}"
done
printf '%s  %s\n' \
  51bb9c45d74a96e8b4971e80cadd5eb8f2e031233088a8e8958fc40ce3cba53d "${PREFIX_LAUNCHER}" \
  ab1b0bd042e12c03455853fa0a5cc92c29d823e5a0f8e53a9c804896126d6c48 "${TRAIN_LAUNCHER}" \
  b2f20692d97f7678528620513b98af2d9ef9a7d4643de571b6aa9010040eef87 "${EVAL_LAUNCHER}" \
  | sha256sum -c - || blocked "pass-3 launcher code differs"

while true; do
  status_line="$("${SUPERVISORCTL}" status "${C002_PROGRAM}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING) sleep "${POLL_SECONDS}" ;;
    EXITED) break ;;
    STOPPED) blocked "c002 was stopped rather than completed" ;;
    FATAL|BACKOFF|UNKNOWN|"") blocked "c002 state=${state:-missing}" ;;
    *) blocked "unexpected c002 state=${state}" ;;
  esac
done

c002_files=("${C002_ROOT}/resume_report.json" "${C002_ROOT}/direct_manifest.json" "${C002_ROOT}/direct_targets.jsonl" "${C002_ROOT}/resume.journal.jsonl")
for required in "${c002_files[@]}"; do [[ -f "${required}" ]] || blocked "completed c002 lacks ${required}"; done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-kimi-c002-resume47-report-v1"
  and .status == "complete"
  and .heldout_175_opened == false
  and .prefix_disposition.training_used_in_this_stage == false
  and .budget.within_contract == true
' "${C002_ROOT}/resume_report.json" >/dev/null || blocked "c002 completion report differs"
c002_snapshot_one="$(sha256sum "${c002_files[@]}")"; sleep "${STABILITY_SECONDS}"; c002_snapshot_two="$(sha256sum "${c002_files[@]}")"
[[ "${c002_snapshot_one}" == "${c002_snapshot_two}" ]] || blocked "c002 artifacts are not stable"
export T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256="$(sha256sum "${C002_ROOT}/resume_report.json" | awk '{print $1}')"
export T5GEMMA_TYPED_C002_RESUME_MANIFEST_SHA256="$(sha256sum "${C002_ROOT}/direct_manifest.json" | awk '{print $1}')"
export T5GEMMA_TYPED_C002_RESUME_TARGETS_SHA256="$(sha256sum "${C002_ROOT}/direct_targets.jsonl" | awk '{print $1}')"

"${PREFIX_LAUNCHER}"
prefix_files=("${PREFIX_ROOT}/prefix_verification_report.json" "${PREFIX_ROOT}/direct_manifest.json" "${PREFIX_ROOT}/direct_targets.jsonl" "${PREFIX_ROOT}/prefix_verification.journal.jsonl")
for required in "${prefix_files[@]}"; do [[ -s "${required}" ]] || blocked "prefix verifier lacks ${required}"; done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-c002-prefix3-verification-report-v1"
  and .status == "complete"
  and .paid_prefix_tasks == 3
  and .verified_targets == 1
  and .verified_task_ids == ["fresh-eval-dba1fc9af285"]
  and .provider_calls == 0
  and .provider_credentials_read == false
  and .heldout_175_opened == false
' "${PREFIX_ROOT}/prefix_verification_report.json" >/dev/null || blocked "prefix verification result differs"
prefix_snapshot_one="$(sha256sum "${prefix_files[@]}")"; sleep "${STABILITY_SECONDS}"; prefix_snapshot_two="$(sha256sum "${prefix_files[@]}")"
[[ "${prefix_snapshot_one}" == "${prefix_snapshot_two}" ]] || blocked "prefix outputs are not stable"
export T5GEMMA_TYPED_PREFIX3_REPORT_SHA256="$(sha256sum "${PREFIX_ROOT}/prefix_verification_report.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PREFIX3_MANIFEST_SHA256="$(sha256sum "${PREFIX_ROOT}/direct_manifest.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PREFIX3_TARGETS_SHA256="$(sha256sum "${PREFIX_ROOT}/direct_targets.jsonl" | awk '{print $1}')"

[[ "$(sha256sum "${c002_files[@]}")" == "${c002_snapshot_one}" ]] || blocked "c002 evidence changed after prefix verification"
"${TRAIN_LAUNCHER}"
[[ "$(sha256sum "${c002_files[@]}")" == "${c002_snapshot_one}" ]] || blocked "c002 evidence changed during training"
[[ "$(sha256sum "${prefix_files[@]}")" == "${prefix_snapshot_one}" ]] || blocked "prefix evidence changed during training"
"${EVAL_LAUNCHER}"
[[ -s "${CHECKPOINT_MANIFEST}" ]] || blocked "pass-3 checkpoint manifest was not emitted"
checkpoint_manifest_sha256="$(sha256sum "${CHECKPOINT_MANIFEST}" | awk '{print $1}')"

echo "T5GEMMA_TYPED_PASS3_HANDOFF_COMPLETE checkpoint_manifest=${CHECKPOINT_MANIFEST} checkpoint_manifest_sha256=${checkpoint_manifest_sha256} promotion=HOLD_REQUIRES_3PLUS_SEEDS verpo=HOLD"
