#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_ARM_C_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
SUPERVISORCTL="${T5GEMMA_TYPED_ARM_C_SUPERVISORCTL:-supervisorctl}"
ARM_B_PROGRAM="${T5GEMMA_TYPED_ARM_B_PROGRAM:-t5gemma2-typed-fold-after-c002-handoff-v1}"
ARM_B_STAGE="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_fold_rs_sft_union_v1"
ARM_B_GENERATION="${WORKSPACE}/artifacts/t5gemma2_typed_fold_rs_sft_union_eval_v1"
ARM_B_EVAL="${WORKSPACE}/artifacts/t5gemma2_typed_extractor_v2_rescore_v1"
ARM_B_DECISION="${ARM_B_EVAL}/arm_b_extractor_v2_single_seed_promotion_gate.json"
ARM_C_STAGE="${T5GEMMA_TYPED_ARM_C_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_fold_gold_replay_v2}"
C002_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_kimi_c002_resume47_v2"
PREFIX_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_c002_prefix3_verification_v1"
TRAIN_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_fold_gold_replay_v1.sh"
EVAL_LAUNCHER="${PROJECT}/deploy/vast/t5gemma2_typed_fold_gold_replay_v1_eval_gate.sh"
PREREG="${WORKSPACE}/analysis_rs_sft_fold/ARM_C_GOLD_REPLAY_PREREGISTRATION.md"
PREREG_SEAL="${WORKSPACE}/analysis_rs_sft_fold/ARM_C_GOLD_REPLAY_PREREGISTRATION.seal.json"
AMENDMENT="${WORKSPACE}/analysis_rs_sft_fold/ARM_C2_PRODUCTION_ELIGIBILITY_AMENDMENT.md"
AMENDMENT_SEAL="${WORKSPACE}/analysis_rs_sft_fold/ARM_C2_PRODUCTION_ELIGIBILITY_AMENDMENT.seal.json"
RESCORE_SEAL="${WORKSPACE}/analysis_rs_sft_fold/TYPED_EXTRACTOR_V2_RESCORE_SEAL.json"
STABILITY_SECONDS="${T5GEMMA_TYPED_ARM_C_STABILITY_SECONDS:-2}"
MIN_INITIAL_KIB=26214400
MIN_RESUME_KIB=12582912

blocked() { echo "T5GEMMA_TYPED_ARM_C_HANDOFF_BLOCKED $*" >&2; exit 78; }
[[ "${ARM_B_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ && "${STABILITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || blocked "handoff configuration is malformed"
for launcher in "${TRAIN_LAUNCHER}" "${EVAL_LAUNCHER}"; do
  [[ -x "${launcher}" ]] || blocked "launcher is absent: ${launcher}"
done
printf '%s  %s\n' \
  5358999872fad573d08023fa2b655f7692b905ff1a8f6cabfb364f56e72321d6 "${TRAIN_LAUNCHER}" \
  5296e5d80c11a1725df30ba3db3a8b65f91683776ae66a7b7f0270122c77e4d7 "${EVAL_LAUNCHER}" \
  aa8fb9b3ba258a0ee117e8c7f98acb55d92fba2d79ef4b0df7b093d57135dcf6 "${PREREG}" \
  c0aa14cecb80681515c05a9db6bedaa89eebfb84660fb5b8932e2ee9ec977f96 "${PREREG_SEAL}" \
  8226d0ebd55476088d2e2a5cbfb06e573e92539012c4dc4ba551417158e261ed "${AMENDMENT}" \
  a15d5b9f42a4df410dadda677e3ceba262bc7ff5f743c566e1370f41cedb2cb7 "${AMENDMENT_SEAL}" \
  c02793912f998dc8c2a85a45a3fcaf5d221561a9bd919256bcf6510bf2caf542 "${RESCORE_SEAL}" \
  | sha256sum -c - || blocked "Arm C launchers/preregistration differ"

status_line="$("${SUPERVISORCTL}" status "${ARM_B_PROGRAM}" 2>/dev/null || true)"
state="$(awk '{print $2}' <<<"${status_line}")"
[[ "${state}" == "EXITED" ]] || blocked "Arm B supervisor must be EXITED, observed=${state:-missing}"

arm_b_files=(
  "${ARM_B_STAGE}/result.json"
  "${ARM_B_STAGE}/run_contract.json"
  "${ARM_B_STAGE}/dataset_manifest.json"
  "${ARM_B_STAGE}/latest_checkpoint.json"
  "${ARM_B_EVAL}/arm_b_opt58_seed42_k10_score_full175.json"
  "${ARM_B_GENERATION}/typed_fold_union_seed42_k10_predictions.json"
  "${ARM_B_GENERATION}/typed_fold_union_seed42_k10_predictions.json.provenance.json"
  "${ARM_B_DECISION}"
)
for required in "${arm_b_files[@]}"; do [[ -s "${required}" ]] || blocked "Arm B evidence is absent: ${required}"; done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-fold-single-seed-promotion-gate-v1"
  and .status == "pass"
  and .decision.promotion_status == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
  and .decision.promoted_checkpoint == null
  and .decision.verpo_status == "HOLD"
  and .automatic_promotion_performed == false
  and .verpo_launched == false
' "${ARM_B_DECISION}" >/dev/null || blocked "Arm B sealed decision differs"
printf '%s  %s\n' \
  f1accbf1db6ab326583b8bdc789250c021db34028690b8bab6d014b69437ac05 "${ARM_B_STAGE}/dataset_manifest.json" \
  8544e778a4e077eb0e62016b6b849f67dcda6c55fef3dc161a0fba573174a248 "${ARM_B_EVAL}/arm_b_opt58_seed42_k10_score_full175.json" \
  88c4d9e121d8734728c9fa0a340fecb8667c300bc72b935fd22eca0b56dad939 "${ARM_B_GENERATION}/typed_fold_union_seed42_k10_predictions.json" \
  ff367d79d03ea08f17171c3ba99f6bc0fc205ce53c29efd40637e1c742cf4db8 "${ARM_B_GENERATION}/typed_fold_union_seed42_k10_predictions.json.provenance.json" \
  eff6296caa560641e7e67f916197354e1b4f89aeb14e387049d1570949093b2c "${ARM_B_DECISION}" \
  | sha256sum -c - || blocked "Arm B direct-union manifest differs"
arm_b_snapshot_one="$(sha256sum "${arm_b_files[@]}")"
sleep "${STABILITY_SECONDS}"
arm_b_snapshot_two="$(sha256sum "${arm_b_files[@]}")"
[[ "${arm_b_snapshot_one}" == "${arm_b_snapshot_two}" ]] || blocked "Arm B evidence is not stable"

available_kib="$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')"
minimum_kib="${MIN_INITIAL_KIB}"
if [[ -s "${ARM_C_STAGE}/latest_checkpoint.json" ]]; then
  /usr/bin/jq -e '.schema=="t5gemma2-typed-fold-gold-replay-checkpoint-v2" and (.update==20 or .update==40)' "${ARM_C_STAGE}/latest_checkpoint.json" >/dev/null \
    || blocked "Arm C resume pointer is malformed"
  minimum_kib="${MIN_RESUME_KIB}"
fi
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge "${minimum_kib}" ]] \
  || blocked "Arm C storage gate failed; available_kib=${available_kib:-unknown} required_kib=${minimum_kib}; cleanup is never automatic"

c002_files=("${C002_ROOT}/resume_report.json" "${C002_ROOT}/direct_manifest.json" "${C002_ROOT}/direct_targets.jsonl")
prefix_files=("${PREFIX_ROOT}/prefix_verification_report.json" "${PREFIX_ROOT}/direct_manifest.json" "${PREFIX_ROOT}/direct_targets.jsonl")
for required in "${c002_files[@]}" "${prefix_files[@]}"; do [[ -s "${required}" ]] || blocked "fold source evidence is absent: ${required}"; done
export T5GEMMA_TYPED_C002_RESUME_REPORT_SHA256="$(sha256sum "${C002_ROOT}/resume_report.json" | awk '{print $1}')"
export T5GEMMA_TYPED_C002_RESUME_MANIFEST_SHA256="$(sha256sum "${C002_ROOT}/direct_manifest.json" | awk '{print $1}')"
export T5GEMMA_TYPED_C002_RESUME_TARGETS_SHA256="$(sha256sum "${C002_ROOT}/direct_targets.jsonl" | awk '{print $1}')"
export T5GEMMA_TYPED_PREFIX3_REPORT_SHA256="$(sha256sum "${PREFIX_ROOT}/prefix_verification_report.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PREFIX3_MANIFEST_SHA256="$(sha256sum "${PREFIX_ROOT}/direct_manifest.json" | awk '{print $1}')"
export T5GEMMA_TYPED_PREFIX3_TARGETS_SHA256="$(sha256sum "${PREFIX_ROOT}/direct_targets.jsonl" | awk '{print $1}')"
source_snapshot="$(sha256sum "${c002_files[@]}" "${prefix_files[@]}")"

"${TRAIN_LAUNCHER}"
[[ "$(sha256sum "${arm_b_files[@]}")" == "${arm_b_snapshot_one}" ]] || blocked "Arm B evidence changed during Arm C training"
[[ "$(sha256sum "${c002_files[@]}" "${prefix_files[@]}")" == "${source_snapshot}" ]] || blocked "fold source evidence changed during Arm C training"
"${EVAL_LAUNCHER}"
[[ "$(sha256sum "${arm_b_files[@]}")" == "${arm_b_snapshot_one}" ]] || blocked "Arm B evidence changed during Arm C evaluation"

echo "T5GEMMA_TYPED_ARM_C_HANDOFF_COMPLETE promotion=HOLD_REQUIRES_3PLUS_MATCHED_SEEDS verpo=HOLD"
