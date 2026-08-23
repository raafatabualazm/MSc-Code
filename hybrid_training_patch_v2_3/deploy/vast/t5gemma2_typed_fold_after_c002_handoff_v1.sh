#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_FOLD_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SUPERVISORCTL="${T5GEMMA_TYPED_FOLD_SUPERVISORCTL:-supervisorctl}"
C002_PROGRAM="${T5GEMMA_TYPED_C002_PROGRAM:-t5gemma2-typed-kimi-c002-resume47}"
PASS3_PROGRAM="${T5GEMMA_TYPED_PASS3_PROGRAM:-t5gemma2-typed-pass3-after-c002-handoff}"
PASS3_EVAL_RETRY_PROGRAM="${T5GEMMA_TYPED_PASS3_EVAL_RETRY_PROGRAM:-t5gemma2-typed-pass3-eval-retry-v1}"
C002_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_c002_resume47_v2"
PASS3_DECISION="${T5GEMMA_TYPED_PASS3_DECISION:-${WORKSPACE}/artifacts/t5gemma2_typed_direct_rs_sft_pass3_eval_v1/pass3_single_seed_promotion_gate.json}"
PASS3_CHECKPOINT_MANIFEST="${T5GEMMA_TYPED_PASS3_CHECKPOINT_MANIFEST:-${WORKSPACE}/artifacts/t5gemma2_typed_pass3_checkpoint_manifest_v1.json}"
PREFIX_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_c002_prefix3_verification_v1"
PREFIX_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_c002_prefix3_verify.sh"
TRAIN_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_fold_rs_sft_union_v1.sh"
EVAL_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_fold_rs_sft_union_v1_eval_gate.sh"
POLL_SECONDS="${T5GEMMA_TYPED_FOLD_POLL_SECONDS:-20}"
STABILITY_SECONDS="${T5GEMMA_TYPED_FOLD_STABILITY_SECONDS:-2}"

blocked() { echo "T5GEMMA_TYPED_FOLD_HANDOFF_BLOCKED $*" >&2; exit 78; }
[[ "${C002_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ && "${PASS3_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ && "${PASS3_EVAL_RETRY_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ && "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ && "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || blocked "handoff configuration is malformed"
for launcher in "${PREFIX_LAUNCHER}" "${TRAIN_LAUNCHER}" "${EVAL_LAUNCHER}"; do
  [[ -x "${launcher}" ]] || blocked "launcher is absent: ${launcher}"
done
printf '%s  %s\n' \
  51bb9c45d74a96e8b4971e80cadd5eb8f2e031233088a8e8958fc40ce3cba53d "${PREFIX_LAUNCHER}" \
  f0e336c3e28cf97525e1ef4d01e12672b10b6f7905c8d3b07522a5e057d6376a "${TRAIN_LAUNCHER}" \
  19ffea1186671eb7c9390bf1e3b4a977c1b8695f7744317f29b1821d8df4532b "${EVAL_LAUNCHER}" \
  | sha256sum -c - || blocked "fold launcher code differs"

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

# Arm B is deliberately serialized behind the preregistered pass-3 train/eval
# handoff.  Starting both Supervisor programs is safe: this arm cannot reach
# prefix verification or the GPU until pass 3 has completed its sealed audit.
while true; do
  status_line="$("${SUPERVISORCTL}" status "${PASS3_PROGRAM}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING) sleep "${POLL_SECONDS}" ;;
    EXITED)
      if [[ -s "${PASS3_DECISION}" && -s "${PASS3_CHECKPOINT_MANIFEST}" ]]; then
        break
      fi
      retry_status_line="$("${SUPERVISORCTL}" status "${PASS3_EVAL_RETRY_PROGRAM}" 2>/dev/null || true)"
      retry_state="$(awk '{print $2}' <<<"${retry_status_line}")"
      case "${retry_state}" in
        RUNNING|STARTING) sleep "${POLL_SECONDS}" ;;
        EXITED) blocked "pass-3 evaluation retry exited without sealed decision artifacts" ;;
        STOPPED) blocked "pass-3 handoff exited without sealed decision artifacts" ;;
        FATAL|BACKOFF|UNKNOWN|"") blocked "pass-3 evaluation retry state=${retry_state:-missing}" ;;
        *) blocked "unexpected pass-3 evaluation retry state=${retry_state}" ;;
      esac
      ;;
    STOPPED) blocked "pass-3 handoff was stopped rather than completed" ;;
    FATAL|BACKOFF|UNKNOWN|"") blocked "pass-3 handoff state=${state:-missing}" ;;
    *) blocked "unexpected pass-3 handoff state=${state}" ;;
  esac
done
[[ -s "${PASS3_DECISION}" && -s "${PASS3_CHECKPOINT_MANIFEST}" ]] \
  || blocked "pass-3 handoff exited without sealed decision artifacts"
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-pass3-single-seed-promotion-gate-v1"
  and .status == "pass"
  and .decision.promotion_status == "HOLD_REQUIRES_3PLUS_SEEDS"
  and .decision.promoted_checkpoint == null
  and .decision.verpo_status == "HOLD"
  and .replication_status.validated_seed_count == 1
  and .replication_status.minimum_required_for_promotion == 3
' "${PASS3_DECISION}" >/dev/null || blocked "pass-3 sealed decision differs"
pass3_snapshot_one="$(sha256sum "${PASS3_DECISION}" "${PASS3_CHECKPOINT_MANIFEST}")"
sleep "${STABILITY_SECONDS}"
pass3_snapshot_two="$(sha256sum "${PASS3_DECISION}" "${PASS3_CHECKPOINT_MANIFEST}")"
[[ "${pass3_snapshot_one}" == "${pass3_snapshot_two}" ]] || blocked "pass-3 evidence is not stable"

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
[[ "$(sha256sum "${PASS3_DECISION}" "${PASS3_CHECKPOINT_MANIFEST}")" == "${pass3_snapshot_one}" ]] \
  || blocked "pass-3 evidence changed during folded-arm training/evaluation"

echo "T5GEMMA_TYPED_FOLD_HANDOFF_COMPLETE arm=B_fold_only promotion=HOLD_REQUIRES_3PLUS_MATCHED_SEEDS verpo=HOLD"
