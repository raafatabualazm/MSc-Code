#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_FOLD_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_TYPED_FOLD_PYTHON:-/venv/main/bin/python}"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
STAGE_DIR="${T5GEMMA_TYPED_FOLD_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_fold_rs_sft_union_v1}"
OUTPUT_DIR="${T5GEMMA_TYPED_FOLD_EVAL_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_fold_rs_sft_union_eval_v1}"
BASELINE_SCORE="${WORKSPACE}/artifacts/t5gemma2_typed_pass2_update58_current_stack_matched_v1/update58_current_stack_seed42_k10_score_full175.json"
BASELINE_PREDICTIONS="${WORKSPACE}/artifacts/t5gemma2_typed_pass2_update58_current_stack_matched_v1/update58_current_stack_seed42_k10_predictions.json"
BASELINE_PROVENANCE="${BASELINE_PREDICTIONS}.provenance.json"
COLLAPSE_CHECKER="${WORKSPACE}/analysis_rs_sft_fold/check_collapse.py"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
PREDICTIONS="${OUTPUT_DIR}/typed_fold_union_seed42_k10_predictions.json"
PROVENANCE="${PREDICTIONS}.provenance.json"
FULL_SCORE="${OUTPUT_DIR}/typed_fold_union_seed42_k10_score_full175.json"
CLEAN_SCORE="${OUTPUT_DIR}/typed_fold_union_seed42_k10_score_clean174.json"
DECISION="${OUTPUT_DIR}/typed_fold_union_single_seed_promotion_gate.json"

blocked() { echo "T5GEMMA_TYPED_FOLD_EVAL_GATE_BLOCKED $*" >&2; exit 78; }
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] || blocked "evaluation runtime is absent"
printf '%s  %s\n' \
  2ae23d69f5dffe816d6b88d0356dc16d88bec16964a1d5dbe66db19c72afdd3c "${PROJECT}/scripts/training/t5gemma2_typed_fold_rs_sft_union_v1.py" \
  4bd44c53548a1449d28cf4efadc93f439a00c392887f2fefab7b55745b79e889 "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_fold_promotion.py" \
  411e1e6c5adf3a23d5cd741c1dc5280a8ea768bfe546f9b4d8f770bcabbb021b "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_pass3_promotion.py" \
  7311cba90e5cbdf0abae17d7051728021140e83e9c61b43edb976e4eaf6a7fa9 "${COLLAPSE_CHECKER}" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py" \
  30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d "${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py" \
  2c543c54a0ee5e55b4df708e8fd088cb772e62d012ddd41550c784c20e617cf0 "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  eb418ff372b8f20a5aad3f4eb232b1a66a397c88704cdb109f591f4f2deabede "${PROJECT}/scripts/evaluation/derive_passk_exclusion_sensitivity.py" \
  4f6dca7eec07731f9b7e80a294ecff931f93e88f5cbd5d433ff56495148ab8a9 "${BASELINE_SCORE}" \
  74d8c944185e736565088c593ceb20f6d52740bde0ef293118ed98d94785278a "${BASELINE_PREDICTIONS}" \
  cdeed55d8246b61f068c8bc7eaf9cb27daf13ffb871f2b2e6c804e9371b7d316 "${BASELINE_PROVENANCE}" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c - || blocked "pinned fold evaluation code/input differs"

for required in "${STAGE_DIR}/result.json" "${STAGE_DIR}/run_contract.json" "${STAGE_DIR}/dataset_manifest.json" "${STAGE_DIR}/latest_checkpoint.json"; do
  [[ -s "${required}" ]] || blocked "training output missing ${required}"
done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-fold-rs-sft-union-run-v1"
  and .status == "training"
  and .architecture == "native_encoder_decoder"
  and .warmstart.update == 348
  and .optimization.epochs == 1
  and .optimization.batch_size == 1
  and .optimization.gradient_accumulation == 8
  and .optimization.learning_rate == 0.000005
  and .optimization.warmup_updates == 0
  and .optimization.seed == 42
  and .dataset.schema == "t5gemma2-typed-fold-rs-sft-union-dataset-v1"
  and .dataset.arm == "B_fold_only"
  and (.dataset.rows >= 447 and .dataset.rows <= 494)
  and .dataset.composition.pass1_225 == 225
  and .dataset.composition.pass2_209 == 209
  and .dataset.composition.kimi_c001 == 12
  and .dataset.composition.kimi_c002_prefix == 1
  and .dataset.composition.kimi_c002_tail == (.dataset.rows - 447)
  and .dataset.composition.gold_replay == 0
  and .dataset.composition.repair_conditioned == 0
  and .dataset.composition.reasoning_rows == 0
  and .dataset.heldout_overlap == 0
  and .dataset.heldout_175_model_visible == false
  and .dataset.task_id_deduplication == "reject_any_duplicate_across_all_source_cohorts"
  and .dataset.target_code_deduplication == "none_retain_same_code_for_distinct_tasks"
  and .dataset.equivalent_typed_source_groups == 1
  and .dataset.equivalent_typed_source_rows == 2
  and .dataset.equivalent_typed_source_cross_acceptance.rows == 4
  and .dataset.equivalent_typed_source_cross_acceptance.passed == 4
  and .dataset.equivalent_typed_source_cross_acceptance.tests_model_visible == false
  and .dataset.tests_model_visible == false
  and .dataset.private_feedback_model_visible == false
  and .dataset.automatic_promotion_permitted == false
  and .dataset.promotion_status == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
  and .privacy.heldout_overlap == 0
  and .privacy.tests_model_visible == false
  and .privacy.private_feedback_model_visible == false
' "${STAGE_DIR}/run_contract.json" >/dev/null || blocked "fold run contract differs"

checkpoint_name="$(/usr/bin/jq -r '.latest_checkpoint // empty' "${STAGE_DIR}/result.json")"
[[ "${checkpoint_name}" =~ ^checkpoint-optstep-[0-9]{6}$ ]] || blocked "fold final checkpoint name is malformed"
CHECKPOINT="$(realpath -e "${STAGE_DIR}/${checkpoint_name}" 2>/dev/null || true)"
[[ -n "${CHECKPOINT}" && "$(dirname "${CHECKPOINT}")" == "$(realpath -e "${STAGE_DIR}")" ]] || blocked "fold checkpoint escapes stage root"
/usr/bin/jq -e --arg checkpoint "${checkpoint_name}" '
  .schema == "t5gemma2-typed-fold-rs-sft-union-run-v1"
  and .status == "complete"
  and (.rows >= 447 and .rows <= 494)
  and .updates == .planned_updates
  and .latest_checkpoint == $checkpoint
' "${STAGE_DIR}/result.json" >/dev/null || blocked "fold result is incomplete"
files=("${STAGE_DIR}/result.json" "${STAGE_DIR}/run_contract.json" "${STAGE_DIR}/dataset_manifest.json" "${STAGE_DIR}/latest_checkpoint.json" \
  "${CHECKPOINT}/run_contract.json" "${CHECKPOINT}/training_state.pt" "${CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${CHECKPOINT}/adapter/adapter_config.json" "${CHECKPOINT}/tokenizer/tokenizer.json")
for required in "${files[@]}"; do [[ -s "${required}" ]] || blocked "checkpoint output missing ${required}"; done
snapshot_one="$(sha256sum "${files[@]}")"; sleep 2; snapshot_two="$(sha256sum "${files[@]}")"
[[ "${snapshot_one}" == "${snapshot_two}" ]] || blocked "fold checkpoint is not stable"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
export PYTHONPATH="${PROJECT}" HF_HOME="${WORKSPACE}/.hf_home" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 DART_BIN
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"
"${PYTHON_BIN}" scripts/evaluation/t5gemma2_measurement_audit_inference.py \
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
[[ "$(sha256sum "${files[@]}")" == "${snapshot_one}" ]] || blocked "fold checkpoint changed during evaluation"

FOLD_SCORE_SHA="$(sha256sum "${FULL_SCORE}" | awk '{print $1}')"
FOLD_PREDICTIONS_SHA="$(sha256sum "${PREDICTIONS}" | awk '{print $1}')"
FOLD_PROVENANCE_SHA="$(sha256sum "${PROVENANCE}" | awk '{print $1}')"
"${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_fold_promotion.py \
  --fold-score "${FULL_SCORE}" --expected-fold-score-sha256 "${FOLD_SCORE_SHA}" \
  --fold-predictions "${PREDICTIONS}" --expected-fold-predictions-sha256 "${FOLD_PREDICTIONS_SHA}" \
  --fold-provenance "${PROVENANCE}" --expected-fold-provenance-sha256 "${FOLD_PROVENANCE_SHA}" \
  --update58-score "${BASELINE_SCORE}" --expected-update58-score-sha256 4f6dca7eec07731f9b7e80a294ecff931f93e88f5cbd5d433ff56495148ab8a9 \
  --update58-predictions "${BASELINE_PREDICTIONS}" --expected-update58-predictions-sha256 74d8c944185e736565088c593ceb20f6d52740bde0ef293118ed98d94785278a \
  --update58-provenance "${BASELINE_PROVENANCE}" --expected-update58-provenance-sha256 cdeed55d8246b61f068c8bc7eaf9cb27daf13ffb871f2b2e6c804e9371b7d316 \
  --collapse-checker "${COLLAPSE_CHECKER}" --expected-collapse-checker-sha256 7311cba90e5cbdf0abae17d7051728021140e83e9c61b43edb976e4eaf6a7fa9 \
  --output "${DECISION}"
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-fold-single-seed-promotion-gate-v1"
  and .arm == "typed_fold_rs_sft_union_v1"
  and .status == "pass"
  and .decision.promotion_status == "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
  and .decision.promoted_checkpoint == null
  and .decision.verpo_status == "HOLD"
  and .automatic_promotion_performed == false
  and .verpo_launched == false
  and .replication_status.minimum_required_for_promotion == 3
' "${DECISION}" >/dev/null || blocked "fold promotion hold differs"

echo "T5GEMMA_TYPED_FOLD_EVAL_COMPLETE decision=${DECISION} promotion=HOLD_REQUIRES_3PLUS_MATCHED_SEEDS verpo=HOLD"
