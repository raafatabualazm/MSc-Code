#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_C2_VERPO_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${T5GEMMA_C2_VERPO_PYTHON:-/venv/main/bin/python}"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
C2_ROOT="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_fold_gold_replay_v2"
C2_CHECKPOINT="${C2_ROOT}/checkpoint-optstep-000058"
VERPO_ROOT="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_c2_verpo_pilot150_v1"
VERPO_CHECKPOINT="${VERPO_ROOT}/checkpoint-optstep-000150"
OUTPUT_DIR="${T5GEMMA_C2_VERPO_EVAL_OUTPUT:-${WORKSPACE}/artifacts/t5gemma2_typed_c2_verpo_matched_eval8192_v1}"
PREREG="${WORKSPACE}/analysis_rs_sft_fold/T5GEMMA2_TYPED_C2_VERPO_150_PREREGISTRATION.md"
PREREG_SEAL="${WORKSPACE}/analysis_rs_sft_fold/T5GEMMA2_TYPED_C2_VERPO_150_PREREGISTRATION.seal.json"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
REPORT="${OUTPUT_DIR}/typed_c2_verpo_seeds42_44_k10_8192_cluster_report.json"

blocked() { echo "T5GEMMA_TYPED_C2_VERPO_MATCHED_EVAL_BLOCKED $*" >&2; exit 78; }
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] || blocked "evaluation runtime is absent"

printf '%s  %s\n' \
  bc515b7dd7efb4d2458da3af407028eca572cf2a7a1af6616e0a8f8797c134a9 "${PREREG}" \
  5bfbe6359d0b84ecabe43542eaebadd77bde0f9e959a210682ebcfa453445c80 "${PREREG_SEAL}" \
  0a88076cda0f6c981e7d07e402b5917966551f7b17f1efb05cc9eb833368fe31 "${PROJECT}/scripts/training/t5gemma2_typed_c2_verpo_pilot150.py" \
  f30bcb283a47255122a01b6bc69d3c855a2d651f22270953ab96031b04faf5c9 "${PROJECT}/scripts/evaluation/t5gemma2_typed_c2_verpo_inference_v1.py" \
  71eb4e512d97f19ae85a139a17188164afa670acd379d9ccfae1c7ae427b642f "${PROJECT}/scripts/evaluation/audit_t5gemma2_typed_c2_verpo_multiseed.py" \
  38a003ae2d5b1fc19bf5c065d5c2577962dde0c5a4e14bc3ca8e3992efce6438 "${PROJECT}/scripts/evaluation/t5gemma2_typed_fold_gold_replay_inference_v1.py" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py" \
  30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d "${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py" \
  a425b5669f62e7b259a648b97097213f7738c0e7cd2905547011e2c968d0466b "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  6436838ffaed0d9c6350c0df58ff9950e5ecb08fc7899af431ee11c0cd5204bb "${PROJECT}/scripts/training/t5gemma2_typed_fold_gold_replay_v1.py" \
  2ae23d69f5dffe816d6b88d0356dc16d88bec16964a1d5dbe66db19c72afdd3c "${PROJECT}/scripts/training/t5gemma2_typed_fold_rs_sft_union_v1.py" \
  bee03f83b4b86baaf60110e8b7d387e80550c43f07d675bc71710a17fba9fc66 "${PROJECT}/scripts/training/t5gemma2_typed_contract_sft.py" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c - || blocked "pinned matched-evaluation code/input differs"

/usr/bin/jq -e '
  .schema == "t5gemma2-typed-c2-verpo-150-preregistration-seal-v1"
  and .status == "sealed_before_rollout_or_training"
  and .artifact_sha256 == "bc515b7dd7efb4d2458da3af407028eca572cf2a7a1af6616e0a8f8797c134a9"
  and .optimizer_updates_visible_at_fixation == 0
' "${PREREG_SEAL}" >/dev/null || blocked "VeRPO preregistration seal differs"

for required in \
  "${C2_CHECKPOINT}/run_contract.json" \
  "${C2_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${VERPO_ROOT}/run_contract.json" \
  "${VERPO_ROOT}/result.json" \
  "${VERPO_ROOT}/latest_checkpoint.json" \
  "${VERPO_CHECKPOINT}/run_contract.json" \
  "${VERPO_CHECKPOINT}/training_state.pt" \
  "${VERPO_CHECKPOINT}/adapter/adapter_model.safetensors"; do
  [[ -s "${required}" ]] || blocked "required sealed checkpoint artifact is absent: ${required}"
done
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-c2-verpo-pilot150-run-v1"
  and .status == "complete"
  and .updates == 150
  and .latest_checkpoint == "checkpoint-optstep-000150"
  and .mechanics_gate == "GO"
  and .automatic_promotion_performed == false
  and .production_floor_eligible == false
  and .pilot_disposition == "discardable_not_for_automatic_promotion"
  and .window_gates_passed == [16,32,48,64,80,96,112,128,144]
' "${VERPO_ROOT}/result.json" >/dev/null || blocked "VeRPO did not reach its sealed successful final state"

gate_files=()
for gate_update in 16 32 48 64 80 96 112 128 144; do
  gate_file="${VERPO_ROOT}/pilot_gate_update$(printf '%06d' "${gate_update}").json"
  gate_files+=("${gate_file}")
  [[ -s "${gate_file}" ]] || blocked "sealed GO gate is absent: ${gate_file}"
  /usr/bin/jq -e --argjson gate "${gate_update}" '
    .schema == "t5gemma2-typed-c2-verpo-window-gate-v1"
    and .status == "pass"
    and .gate_update == $gate
    and .window_end_update == $gate
    and .window_start_update == ($gate - 15)
    and .decision == "GO"
    and .criteria.integrity.pass == true
    and .automatic_promotion_performed == false
    and .private_holdback_read == false
  ' "${gate_file}" >/dev/null || blocked "VeRPO gate ${gate_update} is not a sealed GO"
done

checkpoint_snapshot_one="$(sha256sum \
  "${C2_CHECKPOINT}/run_contract.json" \
  "${C2_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${VERPO_ROOT}/run_contract.json" \
  "${VERPO_ROOT}/result.json" \
  "${VERPO_ROOT}/latest_checkpoint.json" \
  "${VERPO_CHECKPOINT}/run_contract.json" \
  "${VERPO_CHECKPOINT}/training_state.pt" \
  "${VERPO_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${gate_files[@]}")"
sleep 2
checkpoint_snapshot_two="$(sha256sum \
  "${C2_CHECKPOINT}/run_contract.json" \
  "${C2_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${VERPO_ROOT}/run_contract.json" \
  "${VERPO_ROOT}/result.json" \
  "${VERPO_ROOT}/latest_checkpoint.json" \
  "${VERPO_CHECKPOINT}/run_contract.json" \
  "${VERPO_CHECKPOINT}/training_state.pt" \
  "${VERPO_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${gate_files[@]}")"
[[ "${checkpoint_snapshot_one}" == "${checkpoint_snapshot_two}" ]] \
  || blocked "checkpoint evidence is not stable"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home"
export PYTHONPATH="${PROJECT}" HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0 DART_BIN PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"

artifact_args=()
for seed in 42 43 44; do
  baseline_predictions="${OUTPUT_DIR}/c2_baseline_seed${seed}_k10_8192_predictions.json"
  baseline_score="${OUTPUT_DIR}/c2_baseline_seed${seed}_k10_8192_score_full175.json"
  verpo_predictions="${OUTPUT_DIR}/c2_verpo_u150_seed${seed}_k10_8192_predictions.json"
  verpo_score="${OUTPUT_DIR}/c2_verpo_u150_seed${seed}_k10_8192_score_full175.json"

  "${PYTHON_BIN}" scripts/evaluation/t5gemma2_typed_fold_gold_replay_inference_v1.py \
    --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
    --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
    --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
    --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
    --sft_checkpoint "${C2_CHECKPOINT}" --arm sft \
    --input_view typed_opaque_contract --num_samples 10 --generation_batch_size 10 \
    --max_source_tokens 32768 --max_new_tokens 8192 \
    --temperature 0.8 --top_p 0.95 --seed "${seed}" \
    --attn_implementation sdpa --bf16 --output "${baseline_predictions}"
  "${PYTHON_BIN}" scripts/evaluation/score_direct_compact_passk.py \
    --predictions "${baseline_predictions}" \
    --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
    --output "${baseline_score}" --k 10 --workers 32 --timeout 30 --stability_runs 2

  "${PYTHON_BIN}" scripts/evaluation/t5gemma2_typed_c2_verpo_inference_v1.py \
    --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
    --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
    --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
    --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
    --sft_checkpoint "${VERPO_CHECKPOINT}" --arm sft \
    --input_view typed_opaque_contract --num_samples 10 --generation_batch_size 10 \
    --max_source_tokens 32768 --max_new_tokens 8192 \
    --temperature 0.8 --top_p 0.95 --seed "${seed}" \
    --attn_implementation sdpa --bf16 --output "${verpo_predictions}"
  "${PYTHON_BIN}" scripts/evaluation/score_direct_compact_passk.py \
    --predictions "${verpo_predictions}" \
    --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
    --output "${verpo_score}" --k 10 --workers 32 --timeout 30 --stability_runs 2

  artifact_args+=(
    --artifact "c2_baseline|${seed}|${baseline_predictions}|${baseline_score}"
    --artifact "c2_verpo|${seed}|${verpo_predictions}|${verpo_score}"
  )
done

[[ "$(sha256sum \
  "${C2_CHECKPOINT}/run_contract.json" \
  "${C2_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${VERPO_ROOT}/run_contract.json" \
  "${VERPO_ROOT}/result.json" \
  "${VERPO_ROOT}/latest_checkpoint.json" \
  "${VERPO_CHECKPOINT}/run_contract.json" \
  "${VERPO_CHECKPOINT}/training_state.pt" \
  "${VERPO_CHECKPOINT}/adapter/adapter_model.safetensors" \
  "${gate_files[@]}")" == "${checkpoint_snapshot_one}" ]] \
  || blocked "checkpoint evidence changed during matched evaluation"

"${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_c2_verpo_multiseed.py \
  "${artifact_args[@]}" --output "${REPORT}"
/usr/bin/jq -e '
  .schema == "t5gemma2-typed-c2-verpo-multiseed-eval-v1"
  and .status == "complete"
  and .contract.seeds == [42,43,44]
  and .contract.sampling.max_new_tokens == 8192
  and .decision.status == "STOP_AFTER_MATCHED_EVALUATION"
  and .decision.automatic_promotion_performed == false
  and .decision.promoted_checkpoint == null
  and .decision.promotion_permitted_from_this_report == false
' "${REPORT}" >/dev/null || blocked "matched evaluation disposition differs"

echo "T5GEMMA_TYPED_C2_VERPO_MATCHED_EVAL_COMPLETE report=${REPORT} automatic_promotion=false"
