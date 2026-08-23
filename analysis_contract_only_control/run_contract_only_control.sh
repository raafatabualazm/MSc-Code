#!/usr/bin/env bash
set -euo pipefail

# Two-seed opaque typed-contract-only oracle control. This launcher is staged
# for explicit post-review execution; it performs inference/scoring only.
WORKSPACE="${T5GEMMA_CONTRACT_ONLY_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
CONTROL_DIR="${WORKSPACE}/analysis_contract_only_control"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
REFERENCE_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
AUDIT_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_measurement_audit_v1"
INTERVENTION_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_intervention_multiseed_v1"
OUTPUT_DIR="${T5GEMMA_CONTRACT_ONLY_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_contract_only_control_v1}"
PYTHON_BIN="${T5GEMMA_CONTRACT_ONLY_PYTHON:-/venv/main/bin/python}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
GPU_LOCK="${T5GEMMA_CONTRACT_ONLY_GPU_LOCK:-${WORKSPACE}/artifacts/.t5gemma2_typed_seed_replication_gpu.lock}"
GPU_WAIT_SECONDS="${T5GEMMA_CONTRACT_ONLY_GPU_WAIT_SECONDS:-86400}"
GPU_POLL_SECONDS="${T5GEMMA_CONTRACT_ONLY_GPU_POLL_SECONDS:-30}"
MIN_FREE_KIB=5242880
MIN_GPU_FREE_MIB=5120

VIEW_BUILDER="${CONTROL_DIR}/contract_only_view.py"
INFERENCE="${CONTROL_DIR}/contract_only_inference.py"
SCORER_WRAPPER="${CONTROL_DIR}/score_contract_only.py"
SMOKE_VERIFIER="${CONTROL_DIR}/verify_smoke_replay.py"
REPORTER="${CONTROL_DIR}/contract_only_report.py"
HANDOFF_ATTESTER="${CONTROL_DIR}/handoff_attestation.py"
PREREG="${CONTROL_DIR}/preregistration.json"
BUNDLE_MANIFEST="${CONTROL_DIR}/bundle.sha256"
STOCK_SCORER="${PROJECT}/scripts/evaluation/score_direct_compact_passk.py"
EVALUATION="${DATA_DIR}/dev_multifunction_binary.jsonl"
F2="${DATA_DIR}/dev_multifunction_binary_f2.jsonl"
GOLD_SCORE="${AUDIT_DIR}/gold_roundtrip/gold_k1_score.json"
CURRENT_REPORT="${INTERVENTION_DIR}/intervention_multiseed_report.json"
RUNTIME_COMPAT="${INTERVENTION_DIR}/runtime_compatibility/runtime_compatibility.json"
HANDOFF_ATTESTATION="${OUTPUT_DIR}/handoff_attestation.json"

blocked() {
  echo "T5GEMMA_CONTRACT_ONLY_BLOCKED $*" >&2
  exit 78
}

[[ "${GPU_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ \
   && "${GPU_POLL_SECONDS}" =~ ^[1-9][0-9]*$ \
   && "${MIN_FREE_KIB}" =~ ^[1-9][0-9]*$ \
   && "${MIN_GPU_FREE_MIB}" =~ ^[1-9][0-9]*$ ]] \
  || blocked "wait/storage/GPU configuration is invalid"
(( MIN_FREE_KIB >= 5242880 && MIN_GPU_FREE_MIB >= 5120 )) \
  || blocked "registered 5-GiB resource floors cannot be lowered"
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] \
  || blocked "pinned Python or Dart runtime is absent"
for command in sha256sum jq flock nvidia-smi; do
  command -v "${command}" >/dev/null 2>&1 || blocked "required command absent: ${command}"
done

# An explicit human-reviewed bundle digest is mandatory. Merely copying or
# invoking this file cannot authorize GPU execution.
[[ -s "${BUNDLE_MANIFEST}" ]] || blocked "bundle manifest is absent"
bundle_manifest_sha256="$(sha256sum "${BUNDLE_MANIFEST}" | /usr/bin/awk '{print $1}')"
[[ "${T5GEMMA_CONTRACT_ONLY_REVIEWED_BUNDLE_SHA256:-}" == "${bundle_manifest_sha256}" ]] \
  || blocked "review approval does not bind this exact bundle"
(
  cd "${CONTROL_DIR}"
  sha256sum -c "${BUNDLE_MANIFEST}"
) || blocked "reviewed control bundle differs"

# Pin the complete imported generation/scoring path.
printf '%s  %s\n' \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py" \
  800f276275cb583dfed27ac815443d02443c48683e38f99e9f1f2e6797bc34f9 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inputs.py" \
  30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d "${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py" \
  a425b5669f62e7b259a648b97097213f7738c0e7cd2905547011e2c968d0466b "${STOCK_SCORER}" \
  10098cfac9a6475b1c54320d93b2bd989d7efc2298972d5030c7ff567e06c9db "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_report.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  232880791b108df96b4f01bc44a613c595cf4edaa738f6cb9a624412da5e50e4 "${PROJECT}/scripts/training/t5gemma2_compiler_feedback_verpo.py" \
  c4c72410333669f78d109d8848c70a79321ef42dba6e1a8344b138e8bfdbdb51 "${PROJECT}/scripts/training/seq2seq_verpo_core.py" \
  bee03f83b4b86baaf60110e8b7d387e80550c43f07d675bc71710a17fba9fc66 "${PROJECT}/scripts/training/t5gemma2_typed_contract_sft.py" \
  dd4026b2e86a8c3280af5e4379f1cd8a07615e69f9d1959fd1e5ee7dc4f245e2 "${PROJECT}/scripts/training/t5gemma2_enriched_sft.py" \
  a2a34a3e6013556c5958a2cfae637939140e54c21e73929f8b0ee44f7a711bba "${PROJECT}/scripts/training/hybrid_data_controls.py" \
  2ae9c3b012d11baa0f65224b4ab8e18b05807e16fa750850b4a25b7e2790c72a "${PROJECT}/scripts/training/teacher_repair_dataset_antigravity.py" \
  6decd1ed1ecd3ce8e8a0bd6d861c30a26063c9d913957e361584413705f28a3b "${PROJECT}/scripts/preprocessing/build_verpo_feedback_view.py" \
  ece07083b31f8333fb05e9d3d74fbbfdae03304ca798cfc2a0805be2d3dfca45 "${PROJECT}/scripts/preprocessing/build_multifunction_executable_view.py" \
  | sha256sum -c - || blocked "pinned repository code differs"

printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${EVALUATION}" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${F2}" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c - || blocked "sealed full175 input differs"

checkpoint_files=(
  "${CHECKPOINT}/run_contract.json"
  "${CHECKPOINT}/adapter/adapter_model.safetensors"
  "${CHECKPOINT}/adapter/adapter_config.json"
  "${CHECKPOINT}/tokenizer/tokenizer.json"
)
for required in "${checkpoint_files[@]}"; do
  [[ -s "${required}" ]] || blocked "missing frozen checkpoint file ${required}"
done
printf '%s  %s\n' \
  562c3da5f89428e6a7263ad8ec79dde9c8b6eb25c77949606277d7d80aecea4f "${CHECKPOINT}/run_contract.json" \
  83d8152edc7236a144fcb7b321f03c4dc5fcf90a1e866fa334338938ee0bdcdc "${CHECKPOINT}/adapter/adapter_model.safetensors" \
  c21ee4458e7c9fe1321337ce22409ee2a03dfe37299c25cfc7c468a490ffb4c3 "${CHECKPOINT}/adapter/adapter_config.json" \
  f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d "${CHECKPOINT}/tokenizer/tokenizer.json" \
  | sha256sum -c - || blocked "frozen original enriched-SFT checkpoint differs"
/usr/bin/jq -e \
  '.schema == "t5gemma2-enriched-sft-run-v1"
   and .architecture == "native_encoder_decoder"
   and .optimization.epochs == 2
   and .optimization.planned_updates == 348
   and .optimization.seed == 42
   and .lora.rank == 64
   and .lora.alpha == 128' \
  "${CHECKPOINT}/run_contract.json" >/dev/null \
  || blocked "frozen checkpoint contract differs"
checkpoint_snapshot="$(sha256sum "${checkpoint_files[@]}")"

# Static Rank-0 and seed-42 context gates are checked before the GPU wait.
printf '%s  %s\n' \
  3bc3415fb74a613b2da51199f96818f966b4de716b002aa076ec6884f1ab4d1a "${GOLD_SCORE}" \
  16f27a9d96df73e4e5c3e4f43ced4cd3b46574bf3dc9cceb5beadb382c76e14d "${REFERENCE_DIR}/two_epoch_k10_predictions.json" \
  e98d2f7dea3d12a17a4287d77ba324b48e50bff0ba3ca62c765bd85349b43334 "${REFERENCE_DIR}/two_epoch_k10_score.json" \
  b6f44467270b00f4af2bac924485bd6d66ab3e6f5af1e0482924a3007f5c95d1 "${AUDIT_DIR}/typed_opaque_contract_seed42_k10_predictions.json" \
  e5f4e5ad6aeec730aac4ebfa593efe1ea735fac71c157ed8c73887332bf89f83 "${AUDIT_DIR}/typed_opaque_contract_seed42_k10_score.json" \
  | sha256sum -c - || blocked "published Rank-0/seed-42 context differs"
/usr/bin/jq -e \
  '.schema == "direct-compact-attested-passk-v1"
   and .tasks == 175 and .k == 1
   and .pass_at_1.count == 175
   and .pass_at_k.count == 175
   and .compile_at_k.count == 175' \
  "${GOLD_SCORE}" >/dev/null || blocked "Rank-0 gold round-trip failed"

available_kib="$(df -Pk "${WORKSPACE}" | /usr/bin/awk 'NR==2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge "${MIN_FREE_KIB}" ]] \
  || blocked "less than ${MIN_FREE_KIB} KiB is free"

mkdir -p "${OUTPUT_DIR}/smoke" "${OUTPUT_DIR}/full" "${WORKSPACE}/.hf_home" "$(dirname "${GPU_LOCK}")"
export PYTHONPATH="${WORKSPACE}:${PROJECT}"

# CPU-only source/privacy preflight. It also creates the exact first-five
# scoring JSONL and refuses to overwrite any non-identical prior artifact.
"${PYTHON_BIN}" "${VIEW_BUILDER}" \
  --dataset "${EVALUATION}" --f2_jsonl "${F2}" \
  --output "${OUTPUT_DIR}/input_view.preflight.json" \
  --smoke_dataset_output "${OUTPUT_DIR}/smoke/evaluation_first5.jsonl" \
  --smoke_manifest_output "${OUTPUT_DIR}/smoke/evaluation_first5.manifest.json" \
  --smoke_rows 5 >/dev/null || blocked "contract-only privacy preflight failed"
/usr/bin/jq -e \
  '.schema == "t5gemma2-f2-measurement-input-view-v1"
   and .view == "typed_contract_only" and .rows == 175
   and .ordered_task_ids_sha256 == "9b93767fd4d0b4057bc752113faeb1efda9faa609e537e189350a6d874d6e38e"
   and .ordered_source_sha256s_sha256 == "5da3f58c3d9d2c936fd5c02dbb54618a36aed493daa52315098b2e461f39708f"
   and .row_transformations_sha256 == "b563744ca311992983d8c244a41c50fde38befbf6c09f0e8f8cd19fea30d719c"
   and .tests_exposed_to_model == false
   and .full_gold_targets_exposed_to_model == false
   and .gold_interface_types_and_arity_exposed_to_model == true
   and .f2_exposed_to_model == false
   and .summary.gold_derived_oracle_control == true
   and .summary.deployable_type_recovery_frontend_evaluated == false
   and .summary.binary_placeholder.text == ""
   and .summary.binary_placeholder.utf8_hex == ""
   and .summary.binary_placeholder.utf8_bytes == 0
   and .summary.binary_placeholder.task_invariant == true
   and .summary.f2_text_serialized_to_model == false
   and .summary.f2_utf8_bytes_serialized_to_model == 0
   and .summary.recovered_constants_serialized_to_model == false
   and .summary.f2_structure_serialized_to_model == false
   and .summary.external_call_identities_serialized_to_model == false
   and .summary.arity_histogram == {"0":33,"1":82,"2":39,"3":13,"4":5,"5":2,"6":1}' \
  "${OUTPUT_DIR}/input_view.preflight.json" >/dev/null \
  || blocked "contract-only model-visible byte/privacy gate failed"

# The same lock used by the currently active multiseed evaluation makes this a
# true post-current handoff. The predecessor report is checked only after lock
# acquisition, so an early invocation waits rather than racing or failing.
exec 9>"${GPU_LOCK}"
/usr/bin/flock -w "${GPU_WAIT_SECONDS}" 9 \
  || blocked "timed out waiting for shared evaluation GPU lock"

# Close the potentially long flock-wait TOCTOU window.
available_kib="$(df -Pk "${WORKSPACE}" | /usr/bin/awk 'NR==2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge "${MIN_FREE_KIB}" ]] \
  || blocked "less than ${MIN_FREE_KIB} KiB is free after GPU-lock acquisition"

[[ -s "${CURRENT_REPORT}" && -s "${RUNTIME_COMPAT}" ]] \
  || blocked "predecessor multiseed evaluation did not publish final attestations"
printf '%s  %s\n' \
  17645716115052bb48a906a4c7231c76ec28a9d7b66dc55bc53e669e990bee63 "${CURRENT_REPORT}" \
  | sha256sum -c - || blocked "corrected sealed predecessor report differs"
/usr/bin/jq -e \
  '.schema == "t5gemma2-f2-intervention-multiseed-report-v1"
   and .status == "complete"
   and .design.seeds == [42,43,44,45,46]
   and .design.tasks_per_run == 175 and .design.k == 10
   and .script_sha256 == "89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a"
   and .model_visible_source_bytes_identical_across_seeds == true
   and .full_input_view_records_identical_across_seeds == false
   and .allowed_input_view_metadata_drift.field == "row_transformations_sha256"
   and .allowed_input_view_metadata_drift.full_record_identity_not_claimed == true
   and .rank0_gold_roundtrip.passed == 175' \
  "${CURRENT_REPORT}" >/dev/null \
  || blocked "predecessor multiseed report gate failed"
/usr/bin/jq -e \
  '.schema == "t5gemma2-measurement-runtime-compat-v1"
   and .status == "pass"
   and .current_generation_replay.exact_prefix_reproduction == true
   and .current_generation_replay.model_identity_projection_identical == true
   and .current_scoring_replay.candidate_compile_pass_decisions_identical == true
   and .current_scoring_replay.task_metrics_identical == true' \
  "${RUNTIME_COMPAT}" >/dev/null \
  || blocked "predecessor runtime compatibility gate failed"

# A direct start of this downstream program is unauthorized unless the
# separate handoff sealed exact upstream EXITED + report reproduction/stability
# against this same reviewed bundle and predecessor report.
current_report_sha256="$(sha256sum "${CURRENT_REPORT}" | /usr/bin/awk '{print $1}')"
"${PYTHON_BIN}" "${HANDOFF_ATTESTER}" \
  --mode verify \
  --predecessor_report "${CURRENT_REPORT}" \
  --predecessor_reporter_sha256 89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a \
  --stable_report_sha256 "${current_report_sha256}" \
  --bundle_manifest "${BUNDLE_MANIFEST}" \
  --reviewed_bundle_sha256 "${bundle_manifest_sha256}" \
  --upstream_program t5gemma2-measurement-intervention-multiseed-v1 \
  --upstream_state EXITED --stability_seconds 5 \
  --minimum_free_kib "${MIN_FREE_KIB}" --minimum_gpu_free_mib "${MIN_GPU_FREE_MIB}" \
  --output "${HANDOFF_ATTESTATION}" >/dev/null \
  || blocked "exact-EXITED handoff attestation is absent or stale"

waited=0
while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  if [[ -z "${gpu_pids}" && "${gpu_free_mib}" =~ ^[0-9]+$ \
        && "${gpu_free_mib}" -ge "${MIN_GPU_FREE_MIB}" ]]; then
    break
  fi
  (( waited >= GPU_WAIT_SECONDS )) \
    && blocked "GPU unavailable: pids=${gpu_pids:-none} free_mib=${gpu_free_mib:-unknown}"
  sleep "${GPU_POLL_SECONDS}"
  waited=$((waited + GPU_POLL_SECONDS))
done

export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${WORKSPACE}"

common_data=(
  --dataset "${EVALUATION}"
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json"
  --f2_jsonl "${F2}"
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json"
  --sft_checkpoint "${CHECKPOINT}" --arm sft
  --input_view typed_contract_only
  --num_samples 10 --generation_batch_size 10
  --max_source_tokens 32768 --max_new_tokens 4096
  --temperature 0.8 --top_p 0.95
  --attn_implementation sdpa --bf16
)
common_score=(
  --k 10 --workers 32 --timeout 30 --stability_runs 2
)

require_chain_artifacts() {
  local predictions="$1"
  local score="$2"
  [[ -s "${predictions}" \
     && -s "${predictions}.provenance.json" \
     && -s "${predictions}.generation.journal.jsonl" \
     && -s "${predictions}.generation.journal.jsonl.chain-head.json" \
     && -s "${score}" \
     && -s "${score}.evaluation.journal.jsonl" \
     && -s "${score}.evaluation.journal.jsonl.chain-head.json" ]] \
    || blocked "incomplete hash-chain artifacts: ${predictions} / ${score}"
}

# Mandatory 5-task x K=10 smoke, including scorer admission and Dart execution.
SMOKE_PRED="${OUTPUT_DIR}/smoke/typed_contract_only_seed42_first5_k10_predictions.json"
SMOKE_SCORE="${OUTPUT_DIR}/smoke/typed_contract_only_seed42_first5_k10_score.json"
"${PYTHON_BIN}" "${INFERENCE}" "${common_data[@]}" \
  --seed 42 --limit 5 --output "${SMOKE_PRED}"
"${PYTHON_BIN}" "${SCORER_WRAPPER}" \
  --predictions "${SMOKE_PRED}" \
  --evaluation_file "${OUTPUT_DIR}/smoke/evaluation_first5.jsonl" \
  --output "${SMOKE_SCORE}" "${common_score[@]}"
require_chain_artifacts "${SMOKE_PRED}" "${SMOKE_SCORE}"
/usr/bin/jq -e \
  '.input_view == "typed_contract_only" and .num_rows == 5 and .num_samples == 10
   and .f2_exposed_to_model == false
   and .gold_derived_oracle_control == true
   and .deployable_type_recovery_frontend_evaluated == false
   and .heldout.selected_rows == 5
   and .heldout.binary_payload.utf8_bytes == 0' \
  "${SMOKE_PRED}.provenance.json" >/dev/null \
  || blocked "five-task smoke provenance/privacy gate failed"
echo "T5GEMMA_CONTRACT_ONLY_SMOKE_PASS predictions=${SMOKE_PRED} score=${SMOKE_SCORE}"

# Full seed 42, then require exact replay of all 50 smoke generation/scoring
# slots before spending on seed 43.
FULL42_PRED="${OUTPUT_DIR}/full/typed_contract_only_seed42_k10_predictions.json"
FULL42_SCORE="${OUTPUT_DIR}/full/typed_contract_only_seed42_k10_score.json"
"${PYTHON_BIN}" "${INFERENCE}" "${common_data[@]}" \
  --seed 42 --output "${FULL42_PRED}"
"${PYTHON_BIN}" "${SCORER_WRAPPER}" \
  --predictions "${FULL42_PRED}" --evaluation_file "${EVALUATION}" \
  --output "${FULL42_SCORE}" "${common_score[@]}"
require_chain_artifacts "${FULL42_PRED}" "${FULL42_SCORE}"
SMOKE_GATE="${OUTPUT_DIR}/smoke/smoke_to_full_seed42_replay_gate.json"
"${PYTHON_BIN}" "${SMOKE_VERIFIER}" \
  --smoke_predictions "${SMOKE_PRED}" --smoke_score "${SMOKE_SCORE}" \
  --smoke_evaluation "${OUTPUT_DIR}/smoke/evaluation_first5.jsonl" \
  --full_predictions "${FULL42_PRED}" --full_score "${FULL42_SCORE}" \
  --full_evaluation "${EVALUATION}" --output "${SMOKE_GATE}" \
  --smoke_tasks 5 --full_tasks 175 --k 10 --timeout 30 --stability_runs 2
/usr/bin/jq -e \
  '.schema == "t5gemma2-contract-only-smoke-replay-gate-v1"
   and .status == "pass" and .candidate_slots == 50
   and .smoke_evaluation_exact_first_n_bytes == true
   and .predictions_exact_prefix_reproduction == true
   and .candidate_compile_pass_decisions_identical == true' \
  "${SMOKE_GATE}" >/dev/null || blocked "smoke/full seed42 replay gate failed"

FULL43_PRED="${OUTPUT_DIR}/full/typed_contract_only_seed43_k10_predictions.json"
FULL43_SCORE="${OUTPUT_DIR}/full/typed_contract_only_seed43_k10_score.json"
"${PYTHON_BIN}" "${INFERENCE}" "${common_data[@]}" \
  --seed 43 --output "${FULL43_PRED}"
"${PYTHON_BIN}" "${SCORER_WRAPPER}" \
  --predictions "${FULL43_PRED}" --evaluation_file "${EVALUATION}" \
  --output "${FULL43_SCORE}" "${common_score[@]}"
require_chain_artifacts "${FULL43_PRED}" "${FULL43_SCORE}"

[[ -s "${AUDIT_DIR}/baseline_seed43_k10_predictions.json" \
   && -s "${AUDIT_DIR}/baseline_seed43_k10_score.json" \
   && -s "${INTERVENTION_DIR}/typed_opaque_contract_seed43_k10_predictions.json" \
   && -s "${INTERVENTION_DIR}/typed_opaque_contract_seed43_k10_score.json" ]] \
  || blocked "same-seed baseline/typed+F2 comparison artifacts are absent"

FINAL_REPORT="${OUTPUT_DIR}/contract_only_two_seed_report.json"
"${PYTHON_BIN}" "${REPORTER}" \
  --baseline "42|${REFERENCE_DIR}/two_epoch_k10_predictions.json|${REFERENCE_DIR}/two_epoch_k10_score.json" \
  --baseline "43|${AUDIT_DIR}/baseline_seed43_k10_predictions.json|${AUDIT_DIR}/baseline_seed43_k10_score.json" \
  --typed "42|${AUDIT_DIR}/typed_opaque_contract_seed42_k10_predictions.json|${AUDIT_DIR}/typed_opaque_contract_seed42_k10_score.json" \
  --typed "43|${INTERVENTION_DIR}/typed_opaque_contract_seed43_k10_predictions.json|${INTERVENTION_DIR}/typed_opaque_contract_seed43_k10_score.json" \
  --control "42|${FULL42_PRED}|${FULL42_SCORE}" \
  --control "43|${FULL43_PRED}|${FULL43_SCORE}" \
  --evaluation "${EVALUATION}" \
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  --f2_jsonl "${F2}" \
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  --checkpoint "${CHECKPOINT}" --gold_score "${GOLD_SCORE}" \
  --runtime_compatibility "${RUNTIME_COMPAT}" \
  --current_multiseed_report "${CURRENT_REPORT}" \
  --preregistration "${PREREG}" \
  --handoff_attestation "${HANDOFF_ATTESTATION}" \
  --smoke_predictions "${SMOKE_PRED}" --smoke_score "${SMOKE_SCORE}" \
  --smoke_evaluation "${OUTPUT_DIR}/smoke/evaluation_first5.jsonl" \
  --smoke_gate "${SMOKE_GATE}" \
  --expected_tasks 175 --k 10 --output "${FINAL_REPORT}"

[[ "$(sha256sum "${checkpoint_files[@]}")" == "${checkpoint_snapshot}" ]] \
  || blocked "frozen checkpoint changed during control"
/usr/bin/jq -e \
  '.schema == "t5gemma2-contract-only-two-seed-report-v1"
   and .status == "complete"
   and .design.fresh_full_seeds == [42,43]
   and .design.fresh_full_seed_count == 2
   and .design.f2_exposed_to_model == false
   and .design.no_training_or_checkpoint_write == true
   and .limitations.semantic_decoding_proven_by_this_control_alone == false
   and .preregistered_interpretation_gate.compile_equivalence_claimed == false
   and .preregistered_interpretation_gate.semantic_decoding_claimed == false' \
  "${FINAL_REPORT}" >/dev/null || blocked "final report contract failed"
echo "T5GEMMA_CONTRACT_ONLY_COMPLETE output=${FINAL_REPORT}"
