#!/usr/bin/env bash
set -euo pipefail

# Exact-resumable current-stack typed replication.  The three arms use separate
# roots but one advisory GPU lock; a second arm waits rather than contending.
ARM="${1:-}"
WORKSPACE="${T5GEMMA_TYPED_SEED_REPL_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"
PYTHON_BIN="${T5GEMMA_TYPED_SEED_REPL_PYTHON:-/venv/main/bin/python}"
GPU_LOCK="${T5GEMMA_TYPED_SEED_REPL_GPU_LOCK:-${WORKSPACE}/artifacts/.t5gemma2_typed_seed_replication_gpu.lock}"
GPU_WAIT_SECONDS="${T5GEMMA_TYPED_SEED_REPL_GPU_WAIT_SECONDS:-86400}"
GPU_POLL_SECONDS="${T5GEMMA_TYPED_SEED_REPL_GPU_POLL_SECONDS:-30}"
INFERENCE_ADAPTER="${PROJECT}/scripts/evaluation/t5gemma2_typed_seed_replication_inference_v1.py"
WRAPPER="${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py"
BASE_INFERENCE="${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py"
PASS3_SEALER="${PROJECT}/scripts/evaluation/seal_t5gemma2_typed_pass3_checkpoint.py"
SCORER="${PROJECT}/scripts/evaluation/score_direct_compact_passk.py"
EVALUATOR="${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py"
EVALUATION="${DATA_DIR}/dev_multifunction_binary.jsonl"

blocked() {
  echo "T5GEMMA_TYPED_SEED_REPLICATION_BLOCKED arm=${ARM:-missing} $*" >&2
  exit 78
}

[[ "${ARM}" =~ ^(typed_sft|incumbent|pass3)$ ]] \
  || blocked "arm must be typed_sft, incumbent, or pass3"
[[ "${GPU_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ && "${GPU_POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || blocked "GPU wait configuration is invalid"
[[ -x "${PYTHON_BIN}" && -x "${DART_BIN}" ]] \
  || blocked "pinned Python or Dart runtime is absent"

# These hashes are both checked here and embedded in each generation journal.
printf '%s  %s\n' \
  85a7e8b5bc2519233051121228e4dcafd287598e8f9644360f49393fcaf182bf "${INFERENCE_ADAPTER}" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${WRAPPER}" \
  30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d "${BASE_INFERENCE}" \
  f33a11aea6337a612fa664fffe5a3eb70b11d92f7b773d2f9b8c2b134334b6e1 "${PASS3_SEALER}" \
  800f276275cb583dfed27ac815443d02443c48683e38f99e9f1f2e6797bc34f9 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inputs.py" \
  2c543c54a0ee5e55b4df708e8fd088cb772e62d012ddd41550c784c20e617cf0 "${SCORER}" \
  249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 "${EVALUATOR}" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  232880791b108df96b4f01bc44a613c595cf4edaa738f6cb9a624412da5e50e4 "${PROJECT}/scripts/training/t5gemma2_compiler_feedback_verpo.py" \
  c4c72410333669f78d109d8848c70a79321ef42dba6e1a8344b138e8bfdbdb51 "${PROJECT}/scripts/training/seq2seq_verpo_core.py" \
  | sha256sum -c - || blocked "replication code differs"
printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${EVALUATION}" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a "${DATA_DIR}/dev_multifunction_binary.seal.json" \
  6ba98eb496af2ef36ca1a0d460bf6e64b715c42f0b9216c64b4a8fc300ccffab "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  777078c9ba759f45db8908b44990306e4fa403c0bd3b825546029ea7bd49ef44 "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
  | sha256sum -c - || blocked "sealed full175 input differs"

manifest_args=()
case "${ARM}" in
  typed_sft)
    STAGE="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_contract_sft_2epoch_v1"
    CHECKPOINT="${STAGE}/checkpoint-optstep-000348"
    OUTPUT_DIR="${T5GEMMA_TYPED_SEED_REPL_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_seed_replication_sft_opt348_v1}"
    EXPECTED_CONTRACT_CANONICAL=3cb25d54f12743ed43572b219e119667f264abab94ec4cbfac72a94407fbdfc7
    EXPECTED_ADAPTER=71078435105dc29aff1aba5942abd5c272e78ef817896081f6e994938da9d77a
    EXPECTED_ADAPTER_CONFIG=f3701f13cb66b6b5952cd1dd2a71b17206e77c1c646ec806f6dd43d7e059a92d
    EXPECTED_TOKENIZER=f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d
    ;;
  incumbent)
    STAGE="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1"
    CHECKPOINT="${STAGE}/checkpoint-optstep-000058"
    OUTPUT_DIR="${T5GEMMA_TYPED_SEED_REPL_INCUMBENT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_seed_replication_update58_v1}"
    EXPECTED_CONTRACT_CANONICAL=0b979384ff0f87a4331792bbfee73d0df6944259f14a371c8f09fa5ab98ca53f
    EXPECTED_ADAPTER=62377c4c4a7d883a3ea1f0ac55a64d23a303c1cf4c41cdd14530f021163a4bec
    EXPECTED_ADAPTER_CONFIG=b7637ef38530d4d4a936a6b5280d4c5fe761288a7eb06a76d3e67293b4f0fd1b
    EXPECTED_TOKENIZER=f5b325224482ec441ec5fbe2a5ac08c3758e0f9605f6e54368e31f736fcfb01d
    ;;
  pass3)
    PASS3_MANIFEST="${T5GEMMA_TYPED_SEED_REPL_PASS3_MANIFEST:-}"
    PASS3_MANIFEST_SHA256="${T5GEMMA_TYPED_SEED_REPL_PASS3_MANIFEST_SHA256:-}"
    [[ -s "${PASS3_MANIFEST}" && "${PASS3_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
      || blocked "pass3 checkpoint manifest and handoff SHA are required"
    printf '%s  %s\n' "${PASS3_MANIFEST_SHA256}" "${PASS3_MANIFEST}" \
      | sha256sum -c - || blocked "pass3 checkpoint manifest differs"
    CHECKPOINT="$(/usr/bin/jq -er '.checkpoint' "${PASS3_MANIFEST}")" \
      || blocked "pass3 manifest checkpoint path is absent"
    OUTPUT_DIR="${T5GEMMA_TYPED_SEED_REPL_PASS3_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_seed_replication_pass3_v1}"
    manifest_args=(
      --checkpoint-manifest "${PASS3_MANIFEST}"
      --expected-checkpoint-manifest-sha256 "${PASS3_MANIFEST_SHA256}"
    )
    ;;
esac

checkpoint_files=(
  "${CHECKPOINT}/run_contract.json"
  "${CHECKPOINT}/adapter/adapter_model.safetensors"
  "${CHECKPOINT}/adapter/adapter_config.json"
  "${CHECKPOINT}/tokenizer/tokenizer.json"
)
for required in "${checkpoint_files[@]}"; do
  [[ -s "${required}" ]] || blocked "missing checkpoint file ${required}"
done
if [[ "${ARM}" != pass3 ]]; then
  [[ -s "${STAGE}/result.json" ]] || blocked "training result is absent"
  printf '%s  %s\n' \
    "${EXPECTED_ADAPTER}" "${CHECKPOINT}/adapter/adapter_model.safetensors" \
    "${EXPECTED_ADAPTER_CONFIG}" "${CHECKPOINT}/adapter/adapter_config.json" \
    "${EXPECTED_TOKENIZER}" "${CHECKPOINT}/tokenizer/tokenizer.json" \
    | sha256sum -c - || blocked "sealed ${ARM} checkpoint differs"
  observed_contract_canonical="$("${PYTHON_BIN}" - "${CHECKPOINT}/run_contract.json" <<'PY'
import hashlib, json, sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
print(hashlib.sha256(payload).hexdigest())
PY
)"
  [[ "${observed_contract_canonical}" == "${EXPECTED_CONTRACT_CANONICAL}" ]] \
    || blocked "sealed ${ARM} run contract differs"
fi
checkpoint_snapshot="$(sha256sum "${checkpoint_files[@]}")"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE}/.hf_home" "$(dirname "${GPU_LOCK}")"
exec 9>"${GPU_LOCK}"
/usr/bin/flock -w "${GPU_WAIT_SECONDS}" 9 \
  || blocked "timed out waiting for the shared replication GPU lock"

# The lock covers this stack.  The idle check also catches an unrelated
# training/evaluation process that does not yet participate in the lock.
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

for seed in 43 44 45 46; do
  predictions="${OUTPUT_DIR}/${ARM}_seed${seed}_k10_predictions.json"
  score="${OUTPUT_DIR}/${ARM}_seed${seed}_k10_score_full175.json"
  "${PYTHON_BIN}" "${INFERENCE_ADAPTER}" \
    --replication-arm "${ARM}" "${manifest_args[@]}" \
    --dataset "${EVALUATION}" \
    --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json" \
    --f2_jsonl "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
    --f2_manifest "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json" \
    --sft_checkpoint "${CHECKPOINT}" --arm sft \
    --input_view typed_opaque_contract \
    --num_samples 10 --generation_batch_size 10 \
    --max_source_tokens 32768 --max_new_tokens 4096 \
    --temperature 0.8 --top_p 0.95 --seed "${seed}" \
    --attn_implementation sdpa --bf16 --output "${predictions}"
  "${PYTHON_BIN}" "${SCORER}" \
    --predictions "${predictions}" --evaluation_file "${EVALUATION}" \
    --output "${score}" --k 10 --workers 32 --timeout 30 --stability_runs 2
  [[ -s "${predictions}.typed_seed_replication.json" \
     && -s "${predictions}.generation.journal.jsonl.chain-head.json" \
     && -s "${score}.evaluation.journal.jsonl.chain-head.json" ]] \
    || blocked "seed ${seed} did not publish complete hash-chain artifacts"
  echo "T5GEMMA_TYPED_SEED_REPLICATION_SEED_COMPLETE arm=${ARM} seed=${seed} score=${score}"
done

[[ "$(sha256sum "${checkpoint_files[@]}")" == "${checkpoint_snapshot}" ]] \
  || blocked "checkpoint changed during replication"
echo "T5GEMMA_TYPED_SEED_REPLICATION_COMPLETE arm=${ARM} output=${OUTPUT_DIR}"
