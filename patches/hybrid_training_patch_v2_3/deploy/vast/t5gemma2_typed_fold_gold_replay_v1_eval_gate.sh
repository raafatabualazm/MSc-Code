#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_ARM_C_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_ARM_C_PYTHON:-/venv/main/bin/python}"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
STAGE_DIR="${T5GEMMA_TYPED_ARM_C_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_fold_gold_replay_v2}"
OUTPUT_DIR="${T5GEMMA_TYPED_ARM_C_EVAL_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_fold_gold_replay_eval_v2}"
ARM_B_EVAL="${WORKSPACE}/artifacts/t5gemma2_typed_extractor_v2_rescore_v1"
ARM_B_GENERATION="${WORKSPACE}/artifacts/t5gemma2_typed_fold_rs_sft_union_eval_v1"
ARM_B_SCORE="${ARM_B_EVAL}/arm_b_opt58_seed42_k10_score_full175.json"
ARM_B_PREDICTIONS="${ARM_B_GENERATION}/typed_fold_union_seed42_k10_predictions.json"
ARM_B_PROVENANCE="${ARM_B_PREDICTIONS}.provenance.json"
ARM_B_DECISION="${ARM_B_EVAL}/arm_b_extractor_v2_single_seed_promotion_gate.json"
BASELINE_ROOT="${WORKSPACE}/artifacts/t5gemma2_typed_pass2_update58_current_stack_matched_v1"
BASELINE_SCORE="${ARM_B_EVAL}/pass1_update58_seed42_k10_score_full175.json"
BASELINE_PREDICTIONS="${BASELINE_ROOT}/update58_current_stack_seed42_k10_predictions.json"
BASELINE_PROVENANCE="${BASELINE_PREDICTIONS}.provenance.json"
COLLAPSE_CHECKER="${WORKSPACE}/analysis_rs_sft_fold/check_collapse.py"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
RESCORE_SEAL="${WORKSPACE}/analysis_rs_sft_fold/TYPED_EXTRACTOR_V2_RESCORE_SEAL.json"
PREDICTIONS="${OUTPUT_DIR}/typed_fold_gold_replay_c2_seed42_k10_predictions.json"
PROVENANCE="${PREDICTIONS}.provenance.json"
FULL_SCORE="${OUTPUT_DIR}/typed_fold_gold_replay_c2_seed42_k10_score_full175.json"
CLEAN_SCORE="${OUTPUT_DIR}/typed_fold_gold_replay_c2_seed42_k10_score_clean174.json"
DECISION="${OUTPUT_DIR}/typed_fold_gold_replay_c2_single_seed_promotion_gate.json"

blocked() { echo "T5GEMMA_TYPED_ARM_C_EVAL_GATE_BLOCKED $*" >&2; exit 78; }
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] || blocked "evaluation runtime is absent"
printf '%s  %s\n' \
  6436838ffaed0d9c6350c0df58ff9950e5ecb08fc7899af431ee11c0cd5204bb "${PROJECT}/scripts/training/t5gemma2_typed_fold_gold_replay_v1.py" \
  38a003ae2d5b1fc19bf5c065d5c2577962dde0c5a4e14bc3ca8e3992efce6438 "${PROJECT}/scripts/evaluation/t5gemma2_typed_fold_gold_replay_inference_v1.py" \
  7d49c670a58191f3fd7784e39669e7867bc028cb1ccdc8383abce914f8a3893b "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_fold_gold_replay_promotion.py" \
  411e1e6c5adf3a23d5cd741c1dc5280a8ea768bfe546f9b4d8f770bcabbb021b "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_pass3_promotion.py" \
  7311cba90e5cbdf0abae17d7051728021140e83e9c61b43edb976e4eaf6a7fa9 "${COLLAPSE_CHECKER}" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py" \
  30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d "${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py" \
  a425b5669f62e7b259a648b97097213f7738c0e7cd2905547011e2c968d0466b "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  eb418ff372b8f20a5aad3f4eb232b1a66a397c88704cdb109f591f4f2deabede "${PROJECT}/scripts/evaluation/derive_passk_exclusion_sensitivity.py" \
  1cf19c3c107f6289732465b67087f56be96736e7c45d9eabf0ac9346b7d74217 "${BASELINE_SCORE}" \
  74d8c944185e736565088c593ceb20f6d52740bde0ef293118ed98d94785278a "${BASELINE_PREDICTIONS}" \
  cdeed55d8246b61f068c8bc7eaf9cb27daf13ffb871f2b2e6c804e9371b7d316 "${BASELINE_PROVENANCE}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  c02793912f998dc8c2a85a45a3fcaf5d221561a9bd919256bcf6510bf2caf542 "${RESCORE_SEAL}" \
  | sha256sum -c - || blocked "pinned Arm C evaluation code/input differs"

for required in "${ARM_B_SCORE}" "${ARM_B_PREDICTIONS}" "${ARM_B_PROVENANCE}" "${ARM_B_DECISION}"; do
  [[ -s "${required}" ]] || blocked "Arm B sealed evaluation artifact is absent: ${required}"
done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-fold-single-seed-promotion-gate-v1"
  and .status == "pass"
  and .decision.promotion_status == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
  and .decision.verpo_status == "HOLD"
  and .automatic_promotion_performed == false
  and .verpo_launched == false
' "${ARM_B_DECISION}" >/dev/null || blocked "Arm B decision differs"
ARM_B_SCORE_SHA="$(/usr/bin/jq -r '.inputs.fold.sha256 // empty' "${ARM_B_DECISION}")"
ARM_B_PREDICTIONS_SHA="$(/usr/bin/jq -r '.inputs.fold.generation.predictions.sha256 // empty' "${ARM_B_DECISION}")"
ARM_B_PROVENANCE_SHA="$(/usr/bin/jq -r '.inputs.fold.generation.provenance.sha256 // empty' "${ARM_B_DECISION}")"
[[ "${ARM_B_SCORE_SHA}" =~ ^[0-9a-f]{64}$ && "${ARM_B_PREDICTIONS_SHA}" =~ ^[0-9a-f]{64}$ && "${ARM_B_PROVENANCE_SHA}" =~ ^[0-9a-f]{64}$ ]] \
  || blocked "Arm B decision lacks pinned evaluation hashes"
printf '%s  %s\n' \
  "${ARM_B_SCORE_SHA}" "${ARM_B_SCORE}" \
  "${ARM_B_PREDICTIONS_SHA}" "${ARM_B_PREDICTIONS}" \
  "${ARM_B_PROVENANCE_SHA}" "${ARM_B_PROVENANCE}" \
  | sha256sum -c - || blocked "Arm B evaluation bytes differ from its decision"
arm_b_snapshot_one="$(sha256sum "${ARM_B_SCORE}" "${ARM_B_PREDICTIONS}" "${ARM_B_PROVENANCE}" "${ARM_B_DECISION}")"
sleep 2
arm_b_snapshot_two="$(sha256sum "${ARM_B_SCORE}" "${ARM_B_PREDICTIONS}" "${ARM_B_PROVENANCE}" "${ARM_B_DECISION}")"
[[ "${arm_b_snapshot_one}" == "${arm_b_snapshot_two}" ]] || blocked "Arm B evidence is not stable"

for required in "${STAGE_DIR}/result.json" "${STAGE_DIR}/run_contract.json" "${STAGE_DIR}/dataset_manifest.json" "${STAGE_DIR}/latest_checkpoint.json"; do
  [[ -s "${required}" ]] || blocked "Arm C training output missing ${required}"
done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-fold-gold-replay-run-v2"
  and .status == "training"
  and .architecture == "native_encoder_decoder"
  and .warmstart.update == 348
  and .optimization.epochs == 1
  and .optimization.batch_size == 1
  and .optimization.gradient_accumulation == 16
  and .optimization.learning_rate == 0.000005
  and .optimization.warmup_updates == 0
  and .optimization.planned_updates == 58
  and .optimization.updates_per_epoch == 58
  and .optimization.seed == 42
  and .checkpointing.interval == 20
  and .dataset.schema == "t5gemma2-typed-fold-gold-replay-dataset-v2"
  and .dataset.arm == "C2_fold_plus_production_eligible_typed_gold_replay_1to1"
  and .dataset.estimand == "practical_B_plus_1to1_production_eligible_typed_gold_replay_recipe"
  and .dataset.original_arm_c_status == "TERMINATED_PREFLIGHT_INFEASIBLE"
  and .dataset.arm_c2_amendment.document_sha256 == "8226d0ebd55476088d2e2a5cbfb06e573e92539012c4dc4ba551417158e261ed"
  and .dataset.arm_c2_amendment.seal_sha256 == "a15d5b9f42a4df410dadda677e3ceba262bc7ff5f743c566e1370f41cedb2cb7"
  and .dataset.pure_gold_content_causal_claim_permitted == false
  and .dataset.rows == 916
  and .dataset.composition.verified_direct == 458
  and .dataset.composition.gold_replay == 458
  and .dataset.composition.repair_conditioned == 0
  and .dataset.composition.reasoning_rows == 0
  and .dataset.direct_union.rows == 458
  and .dataset.direct_union.dataset_manifest.sha256 == "f1accbf1db6ab326583b8bdc789250c021db34028690b8bab6d014b69437ac05"
  and .dataset.gold_replay.selected_rows == 458
  and .dataset.gold_replay.direct_tasks_excluded == true
  and .dataset.gold_replay.direct_typed_source_sha256s_excluded == true
  and .dataset.gold_replay.unique_typed_source_sha256_within_replay == true
  and .dataset.gold_replay.selected_task_ids_sha256 == "6da49d120c902fde194c09fa14f7718bb379d8e676da8071211e1ac95da8e9df"
  and .dataset.gold_replay.selected_source_sha256s_sha256 == "1c818f33808c4142eb7b148733ce6879a779f795f1855e66533832baa99b31d6"
  and .dataset.gold_replay.selected_target_sha256s_sha256 == "c7031487c72a2edba0baca1d8fe9eadc76232136c439a0f1fc95b25e2044e8f6"
  and .dataset.gold_replay.production_verifier_eligible_unique_typed_sources == 2312
  and .dataset.gold_replay.production_admissibility.candidates_checked == 2314
  and .dataset.gold_replay.production_admissibility.eligible == 2312
  and .dataset.gold_replay.production_admissibility.rejected == 2
  and .dataset.gold_replay.production_admissibility.rejected_task_ids_sha256 == "4907b672737c0b2886b80dd4a4f8f9136b1e8b02f1a66793123f423b843c7b28"
  and .dataset.heldout_overlap == 0
  and .dataset.heldout_175_model_visible == false
  and .dataset.tests_model_visible == false
  and .dataset.private_feedback_model_visible == false
  and .dataset.automatic_promotion_permitted == false
  and .dataset.promotion_status == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
  and .dataset.verpo_status == "HOLD"
  and .privacy.heldout_overlap == 0
  and .privacy.tests_model_visible == false
  and .privacy.private_feedback_model_visible == false
' "${STAGE_DIR}/run_contract.json" >/dev/null || blocked "Arm C run contract differs"

checkpoint_name="$(/usr/bin/jq -r '.latest_checkpoint // empty' "${STAGE_DIR}/result.json")"
[[ "${checkpoint_name}" == "checkpoint-optstep-000058" ]] || blocked "Arm C final checkpoint name differs"
CHECKPOINT="$(realpath -e "${STAGE_DIR}/${checkpoint_name}" 2>/dev/null || true)"
[[ -n "${CHECKPOINT}" && "$(dirname "${CHECKPOINT}")" == "$(realpath -e "${STAGE_DIR}")" ]] || blocked "Arm C checkpoint escapes stage root"
/usr/bin/jq -e --arg checkpoint "${checkpoint_name}" '
  .schema == "t5gemma2-typed-fold-gold-replay-run-v2"
  and .status == "complete"
  and .rows == 916
  and .updates == 58
  and .planned_updates == 58
  and .latest_checkpoint == $checkpoint
' "${STAGE_DIR}/result.json" >/dev/null || blocked "Arm C result is incomplete"
files=("${STAGE_DIR}/result.json" "${STAGE_DIR}/run_contract.json" "${STAGE_DIR}/dataset_manifest.json" "${STAGE_DIR}/latest_checkpoint.json" \
  "${CHECKPOINT}/run_contract.json" "${CHECKPOINT}/training_state.pt" "${CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${CHECKPOINT}/adapter/adapter_config.json" "${CHECKPOINT}/tokenizer/tokenizer.json")
for required in "${files[@]}"; do [[ -s "${required}" ]] || blocked "Arm C checkpoint output missing ${required}"; done
snapshot_one="$(sha256sum "${files[@]}")"; sleep 2; snapshot_two="$(sha256sum "${files[@]}")"
[[ "${snapshot_one}" == "${snapshot_two}" ]] || blocked "Arm C checkpoint is not stable"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
export PYTHONPATH="${PROJECT}" HF_HOME="${WORKSPACE}/.hf_home" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
"${PYTHON_BIN}" scripts/evaluation/t5gemma2_typed_fold_gold_replay_inference_v1.py \
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  --sft_checkpoint "${CHECKPOINT}" --arm sft --input_view typed_opaque_contract --num_samples 10 --generation_batch_size 10 \
  --max_source_tokens 32768 --max_new_tokens 4096 --temperature 0.8 --top_p 0.95 --seed 42 --attn_implementation sdpa --bf16 \
  --output "${PREDICTIONS}"
"${PYTHON_BIN}" scripts/evaluation/score_direct_compact_passk.py \
  --predictions "${PREDICTIONS}" --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" --output "${FULL_SCORE}" \
  --k 10 --workers 32 --timeout 30 --stability_runs 2
"${PYTHON_BIN}" scripts/evaluation/derive_passk_exclusion_sensitivity.py \
  --score "${FULL_SCORE}" --output "${CLEAN_SCORE}" --exclude_task_id sigless_8bf7f40ca356
[[ "$(sha256sum "${files[@]}")" == "${snapshot_one}" ]] || blocked "Arm C checkpoint changed during evaluation"
[[ "$(sha256sum "${ARM_B_SCORE}" "${ARM_B_PREDICTIONS}" "${ARM_B_PROVENANCE}" "${ARM_B_DECISION}")" == "${arm_b_snapshot_one}" ]] \
  || blocked "Arm B evidence changed during Arm C evaluation"

ARM_C_SCORE_SHA="$(sha256sum "${FULL_SCORE}" | awk '{print $1}')"
ARM_C_PREDICTIONS_SHA="$(sha256sum "${PREDICTIONS}" | awk '{print $1}')"
ARM_C_PROVENANCE_SHA="$(sha256sum "${PROVENANCE}" | awk '{print $1}')"
ARM_B_DECISION_SHA="$(sha256sum "${ARM_B_DECISION}" | awk '{print $1}')"
"${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_fold_gold_replay_promotion.py \
  --arm-c-score "${FULL_SCORE}" --expected-arm-c-score-sha256 "${ARM_C_SCORE_SHA}" \
  --arm-c-predictions "${PREDICTIONS}" --expected-arm-c-predictions-sha256 "${ARM_C_PREDICTIONS_SHA}" \
  --arm-c-provenance "${PROVENANCE}" --expected-arm-c-provenance-sha256 "${ARM_C_PROVENANCE_SHA}" \
  --arm-b-score "${ARM_B_SCORE}" --expected-arm-b-score-sha256 "${ARM_B_SCORE_SHA}" \
  --arm-b-predictions "${ARM_B_PREDICTIONS}" --expected-arm-b-predictions-sha256 "${ARM_B_PREDICTIONS_SHA}" \
  --arm-b-provenance "${ARM_B_PROVENANCE}" --expected-arm-b-provenance-sha256 "${ARM_B_PROVENANCE_SHA}" \
  --arm-b-decision "${ARM_B_DECISION}" --expected-arm-b-decision-sha256 "${ARM_B_DECISION_SHA}" \
  --update58-score "${BASELINE_SCORE}" --expected-update58-score-sha256 1cf19c3c107f6289732465b67087f56be96736e7c45d9eabf0ac9346b7d74217 \
  --update58-predictions "${BASELINE_PREDICTIONS}" --expected-update58-predictions-sha256 74d8c944185e736565088c593ceb20f6d52740bde0ef293118ed98d94785278a \
  --update58-provenance "${BASELINE_PROVENANCE}" --expected-update58-provenance-sha256 cdeed55d8246b61f068c8bc7eaf9cb27daf13ffb871f2b2e6c804e9371b7d316 \
  --collapse-checker "${COLLAPSE_CHECKER}" --expected-collapse-checker-sha256 7311cba90e5cbdf0abae17d7051728021140e83e9c61b43edb976e4eaf6a7fa9 \
  --output "${DECISION}"
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-fold-gold-replay-single-seed-promotion-gate-v2"
  and .arm == "typed_fold_production_eligible_gold_replay_arm_c2_v2"
  and .status == "pass"
  and .decision.promotion_status == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
  and .decision.promoted_checkpoint == null
  and .decision.verpo_status == "HOLD"
  and .decision.arm_b_comparison_is_single_seed_diagnostic_only == true
  and .automatic_promotion_performed == false
  and .verpo_launched == false
  and .replication_status.minimum_required_for_promotion == 3
  and .replication_status.arm_b_same_seed_comparators_required == true
' "${DECISION}" >/dev/null || blocked "Arm C promotion hold differs"

echo "T5GEMMA_TYPED_ARM_C_EVAL_COMPLETE decision=${DECISION} promotion=HOLD_REQUIRES_3PLUS_MATCHED_SEEDS verpo=HOLD"
