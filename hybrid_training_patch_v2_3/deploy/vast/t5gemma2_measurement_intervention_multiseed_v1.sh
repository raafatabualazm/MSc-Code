#!/usr/bin/env bash
set -euo pipefail

# Frozen x86 multi-seed replication for the three input interventions already
# evaluated at seed 42.  This is inference/scoring only: no optimizer, API, or
# checkpoint write is allowed.
WORKSPACE="${T5GEMMA_INTERVENTION_REPL_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
CHECKPOINT="${WORKSPACE}/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348"
REFERENCE_DIR="${WORKSPACE}/artifacts/t5gemma2_sft_epoch_ablation_passk_v1"
AUDIT_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_measurement_audit_v1"
OUTPUT_DIR="${T5GEMMA_INTERVENTION_REPL_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_f2_intervention_multiseed_v1}"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
PYTHON_BIN="${T5GEMMA_INTERVENTION_REPL_PYTHON:-/venv/main/bin/python}"
GPU_LOCK="${T5GEMMA_INTERVENTION_REPL_GPU_LOCK:-${WORKSPACE}/artifacts/.t5gemma2_typed_seed_replication_gpu.lock}"
GPU_WAIT_SECONDS="${T5GEMMA_INTERVENTION_REPL_GPU_WAIT_SECONDS:-86400}"
GPU_POLL_SECONDS="${T5GEMMA_INTERVENTION_REPL_GPU_POLL_SECONDS:-30}"
MIN_FREE_KIB="${T5GEMMA_INTERVENTION_REPL_MIN_FREE_KIB:-5242880}"

INFERENCE="${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py"
INPUTS="${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inputs.py"
BASE_INFERENCE="${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py"
SCORER="${PROJECT}/scripts/evaluation/score_direct_compact_passk.py"
REPORTER="${PROJECT}/scripts/evaluation/t5gemma2_measurement_intervention_multiseed_report_v1.py"
COMPAT_CHECKER="${PROJECT}/scripts/evaluation/verify_t5gemma2_measurement_runtime_compat_v1.py"
EVALUATION="${DATA_DIR}/dev_multifunction_binary.jsonl"
GOLD_SCORE="${AUDIT_DIR}/gold_roundtrip/gold_k1_score.json"
PRIOR_REPORT="${AUDIT_DIR}/measurement_report.json"

blocked() {
  echo "T5GEMMA_INTERVENTION_MULTISEED_BLOCKED $*" >&2
  exit 78
}

[[ "${GPU_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ \
   && "${GPU_POLL_SECONDS}" =~ ^[1-9][0-9]*$ \
   && "${MIN_FREE_KIB}" =~ ^[1-9][0-9]*$ ]] \
  || blocked "wait/storage configuration is invalid"
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] \
  || blocked "pinned Python or Dart runtime is absent"

# Pin every repository file on the generation/scoring/report import paths.
printf '%s  %s\n' \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${INFERENCE}" \
  800f276275cb583dfed27ac815443d02443c48683e38f99e9f1f2e6797bc34f9 "${INPUTS}" \
  30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d "${BASE_INFERENCE}" \
  a425b5669f62e7b259a648b97097213f7738c0e7cd2905547011e2c968d0466b "${SCORER}" \
  89b2b9e6f03dc3c08db072a107cc858bf0506aaeaed6fb90bca2f44600864d8a "${REPORTER}" \
  07d14fd62ffd52d361b11e1ef0eb1d816ad89b78edb7a6b62fddfcb52b5a8895 "${COMPAT_CHECKER}" \
  10098cfac9a6475b1c54320d93b2bd989d7efc2298972d5030c7ff567e06c9db "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_report.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  5a76523647c8bef54cf0beba611c5c29611c02cdf9053273ca5e531afe14d23d "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  232880791b108df96b4f01bc44a613c595cf4edaa738f6cb9a624412da5e50e4 "${PROJECT}/scripts/training/t5gemma2_compiler_feedback_verpo.py" \
  c4c72410333669f78d109d8848c70a79321ef42dba6e1a8344b138e8bfdbdb51 "${PROJECT}/scripts/training/seq2seq_verpo_core.py" \
  bee03f83b4b86baaf60110e8b7d387e80550c43f07d675bc71710a17fba9fc66 "${PROJECT}/scripts/training/t5gemma2_typed_contract_sft.py" \
  dd4026b2e86a8c3280af5e4379f1cd8a07615e69f9d1959fd1e5ee7dc4f245e2 "${PROJECT}/scripts/training/t5gemma2_enriched_sft.py" \
  a2a34a3e6013556c5958a2cfae637939140e54c21e73929f8b0ee44f7a711bba "${PROJECT}/scripts/training/hybrid_data_controls.py" \
  | sha256sum -c - || blocked "replication code differs"

printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${EVALUATION}" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
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
  | sha256sum -c - || blocked "frozen checkpoint differs"
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

# Bind reused context to the published measurement audit and Rank-0 harness
# gate.  The report also revalidates every hash-chained seed-42 arm.
printf '%s  %s\n' \
  140efcfb69eb9fd9c73e076d672f4384f2531e7e787228f71a3508e174d647a2 "${PRIOR_REPORT}" \
  3bc3415fb74a613b2da51199f96818f966b4de716b002aa076ec6884f1ab4d1a "${GOLD_SCORE}" \
  16f27a9d96df73e4e5c3e4f43ced4cd3b46574bf3dc9cceb5beadb382c76e14d "${REFERENCE_DIR}/two_epoch_k10_predictions.json" \
  e98d2f7dea3d12a17a4287d77ba324b48e50bff0ba3ca62c765bd85349b43334 "${REFERENCE_DIR}/two_epoch_k10_score.json" \
  f459c44a4ba0d7a73af59e005801c5eaf79f2f207a4b129177ebedb2246f523d "${AUDIT_DIR}/semantic_body_swap_seed42_k10_predictions.json" \
  700246b517b4cfd408b7dccd96476542f5e79a27023655c6c089220da0427958 "${AUDIT_DIR}/semantic_body_swap_seed42_k10_score.json" \
  9bf07f3a455ee81f9e1798d4aea3dfb8f9d3f7b130ad09721c355175ae146a36 "${AUDIT_DIR}/constants_stripped_seed42_k10_predictions.json" \
  238144453620476bb71cae1aad1d365a21c8588e807529e3d9559247bf12f304 "${AUDIT_DIR}/constants_stripped_seed42_k10_score.json" \
  b6f44467270b00f4af2bac924485bd6d66ab3e6f5af1e0482924a3007f5c95d1 "${AUDIT_DIR}/typed_opaque_contract_seed42_k10_predictions.json" \
  e5f4e5ad6aeec730aac4ebfa593efe1ea735fac71c157ed8c73887332bf89f83 "${AUDIT_DIR}/typed_opaque_contract_seed42_k10_score.json" \
  | sha256sum -c - || blocked "published seed-42 context differs"
/usr/bin/jq -e \
  '.schema == "t5gemma2-f2-measurement-audit-report-v1"
   and .status == "complete"
   and .heldout_tasks == 175
   and .k == 10
   and .interpretation_gate.triggered == true' \
  "${PRIOR_REPORT}" >/dev/null || blocked "prior measurement report gate failed"
/usr/bin/jq -e \
  '.schema == "direct-compact-attested-passk-v1"
   and .tasks == 175 and .k == 1
   and .pass_at_1.count == 175
   and .pass_at_k.count == 175
   and .compile_at_k.count == 175' \
  "${GOLD_SCORE}" >/dev/null || blocked "Rank-0 gold round-trip failed"

# Recompute the complete historical audit before acquiring the GPU.  This
# catches a missing/corrupt baseline seed 43--46 now, not after twelve runs.
historical_report_args=(
  --baseline "42|${REFERENCE_DIR}/two_epoch_k10_predictions.json|${REFERENCE_DIR}/two_epoch_k10_score.json"
)
for seed in 43 44 45 46; do
  historical_report_args+=(
    --baseline "${seed}|${AUDIT_DIR}/baseline_seed${seed}_k10_predictions.json|${AUDIT_DIR}/baseline_seed${seed}_k10_score.json"
  )
done
for view in semantic_body_swap constants_stripped typed_opaque_contract; do
  historical_report_args+=(
    --ablation "${view}|${AUDIT_DIR}/${view}_seed42_k10_predictions.json|${AUDIT_DIR}/${view}_seed42_k10_score.json"
  )
done
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
  "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_report.py" \
  "${historical_report_args[@]}" --gold_score "${GOLD_SCORE}" \
  --expected_tasks 175 --k 10 --output "${PRIOR_REPORT}" \
  >/dev/null || blocked "historical five-seed audit no longer reproduces"

available_kib="$(df -Pk "${WORKSPACE}" | /usr/bin/awk 'NR==2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ && "${available_kib}" -ge "${MIN_FREE_KIB}" ]] \
  || blocked "less than ${MIN_FREE_KIB} KiB is free"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home" "$(dirname "${GPU_LOCK}")"
exec 9>"${GPU_LOCK}"
/usr/bin/flock -w "${GPU_WAIT_SECONDS}" 9 \
  || blocked "timed out waiting for shared evaluation GPU lock"

waited=0
while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "${gpu_pids}" ]] && break
  (( waited >= GPU_WAIT_SECONDS )) \
    && blocked "GPU remained occupied by pid(s): $(tr '\n' ',' <<<"${gpu_pids}")"
  sleep "${GPU_POLL_SECONDS}"
  waited=$((waited + GPU_POLL_SECONDS))
done

export PYTHONPATH="${PROJECT}"
export HF_HOME="${WORKSPACE}/.hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "${DART_BIN}"):${PATH}"
cd "${PROJECT}"

common_data=(
  --dataset "${EVALUATION}"
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json"
  --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl"
  --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json"
  --sft_checkpoint "${CHECKPOINT}" --arm sft
  --num_samples 10 --generation_batch_size 10
  --max_source_tokens 32768 --max_new_tokens 4096
  --temperature 0.8 --top_p 0.95
  --attn_implementation sdpa --bf16
)
common_score=(
  --evaluation_file "${EVALUATION}" --k 10 --workers 32
  --timeout 30 --stability_runs 2
)

# The seed-42 artifacts used an earlier core inference/evaluator revision.
# Before spending on twelve runs, replay 50 generation slots exactly and
# rescore all 1,750 sealed seed-42 slots.  Only loader/provenance extensions are
# acceptable; any changed generated text or compile/pass decision stops here.
PREFLIGHT_DIR="${OUTPUT_DIR}/runtime_compatibility"
PREFLIGHT_PREDICTIONS="${PREFLIGHT_DIR}/typed_seed42_first5_predictions.json"
PREFLIGHT_RESCORE="${PREFLIGHT_DIR}/typed_seed42_current_rescore.json"
RUNTIME_COMPAT="${PREFLIGHT_DIR}/runtime_compatibility.json"
mkdir -p "${PREFLIGHT_DIR}"
"${PYTHON_BIN}" "${INFERENCE}" \
  "${common_data[@]}" --input_view typed_opaque_contract --seed 42 --limit 5 \
  --output "${PREFLIGHT_PREDICTIONS}"
"${PYTHON_BIN}" "${SCORER}" \
  --predictions "${AUDIT_DIR}/typed_opaque_contract_seed42_k10_predictions.json" \
  --output "${PREFLIGHT_RESCORE}" "${common_score[@]}"
"${PYTHON_BIN}" "${COMPAT_CHECKER}" \
  --historical_predictions "${AUDIT_DIR}/typed_opaque_contract_seed42_k10_predictions.json" \
  --replay_predictions "${PREFLIGHT_PREDICTIONS}" \
  --historical_score "${AUDIT_DIR}/typed_opaque_contract_seed42_k10_score.json" \
  --rescored "${PREFLIGHT_RESCORE}" --historical_rows 175 --replay_rows 5 \
  --input_view typed_opaque_contract --output "${RUNTIME_COMPAT}"
/usr/bin/jq -e \
  '.schema == "t5gemma2-measurement-runtime-compat-v1"
   and .status == "pass"
   and .current_generation_replay.candidates == 50
   and .current_generation_replay.exact_prefix_reproduction == true
   and .current_generation_replay.model_identity_projection_identical == true
   and .current_scoring_replay.candidate_compile_pass_decisions_identical == true
   and .current_scoring_replay.task_metrics_identical == true' \
  "${RUNTIME_COMPAT}" >/dev/null || blocked "runtime compatibility replay failed"
echo "T5GEMMA_INTERVENTION_MULTISEED_RUNTIME_COMPAT_PASS record=${RUNTIME_COMPAT}"

# View-major priority produces complete five-seed arms if rental time ends:
# headline typed contract, security-relevant constants, then body-swap control.
for view in typed_opaque_contract constants_stripped semantic_body_swap; do
  for seed in 43 44 45 46; do
    predictions="${OUTPUT_DIR}/${view}_seed${seed}_k10_predictions.json"
    score="${OUTPUT_DIR}/${view}_seed${seed}_k10_score.json"
    "${PYTHON_BIN}" "${INFERENCE}" \
      "${common_data[@]}" --input_view "${view}" --seed "${seed}" \
      --output "${predictions}"
    "${PYTHON_BIN}" "${SCORER}" \
      --predictions "${predictions}" --output "${score}" "${common_score[@]}"
    [[ -s "${predictions}.provenance.json" \
       && -s "${predictions}.generation.journal.jsonl.chain-head.json" \
       && -s "${score}.evaluation.journal.jsonl.chain-head.json" ]] \
      || blocked "${view} seed ${seed} did not publish complete hash-chain artifacts"
    echo "T5GEMMA_INTERVENTION_MULTISEED_RUN_COMPLETE view=${view} seed=${seed} score=${score}"
  done
done

report_args=(
  --baseline "42|${REFERENCE_DIR}/two_epoch_k10_predictions.json|${REFERENCE_DIR}/two_epoch_k10_score.json"
)
for seed in 43 44 45 46; do
  report_args+=(
    --baseline "${seed}|${AUDIT_DIR}/baseline_seed${seed}_k10_predictions.json|${AUDIT_DIR}/baseline_seed${seed}_k10_score.json"
  )
done
for view in typed_opaque_contract constants_stripped semantic_body_swap; do
  report_args+=(
    --arm "${view}|42|${AUDIT_DIR}/${view}_seed42_k10_predictions.json|${AUDIT_DIR}/${view}_seed42_k10_score.json"
  )
  for seed in 43 44 45 46; do
    report_args+=(
      --arm "${view}|${seed}|${OUTPUT_DIR}/${view}_seed${seed}_k10_predictions.json|${OUTPUT_DIR}/${view}_seed${seed}_k10_score.json"
    )
  done
done
"${PYTHON_BIN}" "${REPORTER}" \
  "${report_args[@]}" \
  --seed42_measurement_report "${PRIOR_REPORT}" \
  --runtime_compatibility "${RUNTIME_COMPAT}" \
  --gold_score "${GOLD_SCORE}" --expected_tasks 175 --k 10 \
  --output "${OUTPUT_DIR}/intervention_multiseed_report.json"

[[ "$(sha256sum "${checkpoint_files[@]}")" == "${checkpoint_snapshot}" ]] \
  || blocked "frozen checkpoint changed during replication"
echo "T5GEMMA_INTERVENTION_MULTISEED_COMPLETE output=${OUTPUT_DIR}/intervention_multiseed_report.json"
