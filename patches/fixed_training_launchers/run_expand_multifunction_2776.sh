#!/bin/bash
set -Eeuo pipefail

# Build only the missing 1,196 Phase-0 fit rows, then byte-append them to the
# immutable 1,580-row parent.  This script never stops or restarts qwen38_kd.

cd /workspace
export PYTHONPATH=/workspace
export DART_BIN=/workspace/dart-3.12.2/usr/bin/dart

root=/workspace/multifunction_v1/expanded2776
selection="${root}/selection"
private="${root}/private"
extraction="${root}/extraction"
constants="${root}/constants"
sanitized="${root}/sanitized"
build="${root}/build"
logs="${root}/logs"
mkdir -p "${root}" "${logs}"

python_bin=/venv/main/bin/python
patch=/workspace/hybrid_training_patch_v2_3
release=/workspace/releases/direct_compact_phase0_s44_pool_v3_dart3122_cfgtypes_v2_release
parent=/workspace/multifunction_v1/build
tokenizer=/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json
attestation_key=/workspace/multifunction_v1/private_v2/user_symbol_attestation.key

sha() {
  sha256sum "$1" | cut -d ' ' -f1
}

require_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local observed
  test -f "${path}" || {
    printf 'Missing %s: %s\n' "${label}" "${path}" >&2
    return 2
  }
  observed="$(sha "${path}")"
  test "${observed}" = "${expected}" || {
    printf '%s SHA mismatch: %s != %s\n' \
      "${label}" "${observed}" "${expected}" >&2
    return 2
  }
}

require_rows() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local observed
  observed="$(wc -l < "${path}")"
  test "${observed}" -eq "${expected}" || {
    printf '%s row mismatch: %s != %s\n' \
      "${label}" "${observed}" "${expected}" >&2
    return 2
  }
}

prepare_script="${patch}/scripts/preprocessing/prepare_phase0_2776_expansion.py"
sanitize_script="${patch}/scripts/preprocessing/sanitize_phase0_supplemental_targets.py"
build_script="${patch}/scripts/preprocessing/build_phase0_2776_multifunction_expansion.py"

# These hashes intentionally bind the deployed implementation, not merely its
# filename.  Update them only together with reviewed script changes.
require_sha "${prepare_script}" \
  3d7a8c1c3e0c81a273e4d949f275fa5560de2a26dc4bd1658dcbd1d13b83c969 \
  expansion-preparation-script
require_sha "${sanitize_script}" \
  4faa69180b77b4b57f6cbb7e07bb7a97bf0bd0a6041e8242efb2d28806e919ba \
  supplemental-sanitation-script
require_sha "${build_script}" \
  c68f7ae1b278ed65ae2ff7585f95fbcf85dd6028738623f10db0f163af940b51 \
  expansion-build-script

require_sha "${release}/source_preparation/prepared/train_private_labels.jsonl" \
  cbeec4154b21604ea9e6989552035b006a112ffb0a4079b2a44adef75997b123 \
  phase0-train-labels
require_sha "${release}/source_preparation/private_build_inputs/train.jsonl" \
  a56b8fb9d8a26c872b4e26f28823cecef76e00cdb9dec08c87c72678c012ad1a \
  phase0-private-train
require_sha "${release}/source_preparation/private_build_inputs/dev.jsonl" \
  1464004d3970b5b4eb5aecfe5a74a9f40f74de35e9a246af3521e33cb1ad17a8 \
  phase0-private-dev
require_sha "${release}/binary_build/aot_manifest.jsonl" \
  e5bdb05eaf08281113298b9f37f26960a6f630b258de51eee92e672af722e633 \
  phase0-aot-manifest
require_sha /workspace/data/training/combined_fresh_s44_train_input.clean.jsonl \
  312d5a7cfc9a5866c38479a3384bd49f47b55b26b1dd46200bb70539945e9b65 \
  phase0-source-corpus

require_sha "${parent}/train_multifunction_binary.jsonl" \
  3cf3dd5b0fc950f7516434b2c90ce2bd6334178ace962e4442026bd33f476e18 \
  frozen-parent-dataset
require_sha "${parent}/train_multifunction_binary.seal.json" \
  14f2cc5ddcb63c8ad25f6a5c9bb6209af111dafbb0827523a24d8089585a94a5 \
  frozen-parent-seal
require_sha "${parent}/train_multifunction_binary_f2.jsonl" \
  f86032533f7f3e1bade9280e5bf6e28858daf25e34167837fec160a34b2370ba \
  frozen-parent-f2
require_sha "${parent}/train_multifunction_binary_f2.jsonl.manifest.json" \
  231227e9aa6f31c793ff097e8167972975e9f94871c9ea682d6a0dc996ed4af8 \
  frozen-parent-f2-manifest
require_sha "${parent}/dev_multifunction_binary.jsonl" \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  historical-heldout-dataset
require_sha "${parent}/dev_multifunction_binary.seal.json" \
  5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a \
  historical-heldout-seal
require_sha "${parent}/multifunction_inline_cfg_v2_contract.json" \
  14ece4b23b8b2193c4a5b05716f86b7ca0fa05449baece0cf00153c91ebaa7cb \
  frozen-contract
require_sha "${parent}/multifunction_inline_cfg_v2_codebook.json" \
  c36f9aadc3308c9916070c2088704d4b5bf6410be84d8df6c7831ed448ed0e14 \
  frozen-codebook
require_sha "${tokenizer}" \
  aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
  tokenizer

require_sha /workspace/scripts/data/build_dart_user_symbol_attestation.py \
  d1888944bf9000082504cf00ee562d8261c4ae8e213bf9fbc7e97d22ca2c8bad \
  symbol-attestation-builder
require_sha /workspace/scripts/data/extract_dart_aot_user_function_bundle.py \
  ff3cd323eb3045da0a9bf8b3489f8e867cbf3ae3dfd12181fc6c2004423af2a5 \
  multifunction-extractor
require_sha /workspace/scripts/data/extract_attested_binary_pool_constants.py \
  a05238b0fbdb0cb8e0576d757e4d531fbaf67e146d30c73785e289d257db54f1 \
  constants-extractor
require_sha /workspace/scripts/data/gdb_dump_attested_pool_offsets.py \
  7df848639591b8646c800c46bf9babe07dce93223509203188fa2a5ab90af09c \
  constants-gdb-script
require_sha /workspace/scripts/data/build_multifunction_compact_v2.py \
  b2e0e33a56c470ac54257a0fc2124bcc2b9d58639a58416d33c7fbbf74d2ca52 \
  inline-cfg-codec
require_sha "${patch}/scripts/preprocessing/build_multifunction_binary_compact.py" \
  bb954a0b5aafe5fa51c97cce40d25c80dd6f65f3fa2696d289f31dcbdf4fae66 \
  frozen-adapter-script
require_sha "${patch}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 \
  production-evaluator
require_sha /workspace/frontier_ceiling_patch_v1/frontier_f2.py \
  097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce \
  frontier-f2
test "$(stat -c '%a' "${attestation_key}")" = 600
test "$(wc -c < "${attestation_key}")" -ge 32

if [[ ! -f "${selection}/expansion_selection.seal.json" ]]; then
  if [[ -e "${selection}" ]]; then
    printf 'Incomplete selection directory exists; inspect before retry: %s\n' \
      "${selection}" >&2
    exit 2
  fi
  "${python_bin}" "${prepare_script}" \
    --phase0-train-labels "${release}/source_preparation/prepared/train_private_labels.jsonl" \
    --expected-phase0-train-labels-sha256 cbeec4154b21604ea9e6989552035b006a112ffb0a4079b2a44adef75997b123 \
    --phase0-private-build-train "${release}/source_preparation/private_build_inputs/train.jsonl" \
    --expected-phase0-private-build-train-sha256 a56b8fb9d8a26c872b4e26f28823cecef76e00cdb9dec08c87c72678c012ad1a \
    --phase0-aot-manifest "${release}/binary_build/aot_manifest.jsonl" \
    --expected-phase0-aot-manifest-sha256 e5bdb05eaf08281113298b9f37f26960a6f630b258de51eee92e672af722e633 \
    --phase0-source-corpus /workspace/data/training/combined_fresh_s44_train_input.clean.jsonl \
    --expected-phase0-source-corpus-sha256 312d5a7cfc9a5866c38479a3384bd49f47b55b26b1dd46200bb70539945e9b65 \
    --parent-fit "${parent}/train_multifunction_binary.jsonl" \
    --expected-parent-fit-sha256 3cf3dd5b0fc950f7516434b2c90ce2bd6334178ace962e4442026bd33f476e18 \
    --parent-fit-seal "${parent}/train_multifunction_binary.seal.json" \
    --expected-parent-fit-seal-sha256 14f2cc5ddcb63c8ad25f6a5c9bb6209af111dafbb0827523a24d8089585a94a5 \
    --heldout "${parent}/dev_multifunction_binary.jsonl" \
    --expected-heldout-sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
    --heldout-seal "${parent}/dev_multifunction_binary.seal.json" \
    --expected-heldout-seal-sha256 5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a \
    --frozen-contract "${parent}/multifunction_inline_cfg_v2_contract.json" \
    --expected-frozen-contract-sha256 14ece4b23b8b2193c4a5b05716f86b7ca0fa05449baece0cf00153c91ebaa7cb \
    --output-dir "${selection}" \
    2>&1 | tee "${logs}/selection.log"
fi
require_rows "${selection}/fit_task_manifest_2776.jsonl" 2776 fit-membership
require_rows "${selection}/supplemental_task_manifest_1196.jsonl" 1196 supplemental-membership
require_rows "${selection}/supplemental_aot_manifest_1196.jsonl" 1196 supplemental-aot-manifest
require_rows "${selection}/supplemental_targets_unsanitized_1196.jsonl" 1196 supplemental-targets

selection_sha="$(sha "${selection}/expansion_selection.seal.json")"
supplemental_manifest_sha="$(sha "${selection}/supplemental_task_manifest_1196.jsonl")"
supplemental_aot_sha="$(sha "${selection}/supplemental_aot_manifest_1196.jsonl")"
supplemental_targets_sha="$(sha "${selection}/supplemental_targets_unsanitized_1196.jsonl")"
supplemental_targets_seal_sha="$(sha "${selection}/supplemental_targets_unsanitized_1196.seal.json")"

mkdir -p "${private}"
attestation="${private}/user_symbol_attestation_1196.jsonl"
attestation_report="${private}/attestation_build_report.json"
if [[ ! -f "${attestation}" || ! -f "${attestation_report}" ]]; then
  test ! -e "${attestation}" && test ! -e "${attestation_report}"
  "${python_bin}" /workspace/scripts/data/build_dart_user_symbol_attestation.py \
    --aot-manifest "${selection}/supplemental_aot_manifest_1196.jsonl" \
    --aot-manifest-sha256 "${supplemental_aot_sha}" \
    --train-build-input "${release}/source_preparation/private_build_inputs/train.jsonl" \
    --train-build-input-sha256 a56b8fb9d8a26c872b4e26f28823cecef76e00cdb9dec08c87c72678c012ad1a \
    --dev-build-input "${release}/source_preparation/private_build_inputs/dev.jsonl" \
    --dev-build-input-sha256 1464004d3970b5b4eb5aecfe5a74a9f40f74de35e9a246af3521e33cb1ad17a8 \
    --key-file "${attestation_key}" \
    --expected-rows 1196 \
    --output-jsonl "${attestation}" \
    --report "${attestation_report}" \
    2>&1 | tee "${logs}/attestation.log"
fi
require_rows "${attestation}" 1196 supplemental-symbol-attestation
attestation_sha="$(sha "${attestation}")"
jq -e \
  --arg aot "${supplemental_aot_sha}" \
  --arg train a56b8fb9d8a26c872b4e26f28823cecef76e00cdb9dec08c87c72678c012ad1a \
  --arg dev 1464004d3970b5b4eb5aecfe5a74a9f40f74de35e9a246af3521e33cb1ad17a8 \
  --arg output "${attestation_sha}" \
  '.schema == "dart-user-symbol-attestation-build-report-v1"
   and .complete == true
   and .rows == 1196
   and .expected_rows == 1196
   and .input_hashes.aot_manifest_sha256 == $aot
   and .input_hashes.train_build_input_sha256 == $train
   and .input_hashes.dev_build_input_sha256 == $dev
   and .output_jsonl_sha256 == $output' \
  "${attestation_report}" >/dev/null

# Target recertification is independent of GDB extraction.  Run it in parallel
# so no GPU time is lost waiting for serial CPU-only preprocessing.
sanitize_pid=
if [[ ! -f "${sanitized}/supplemental_train_imitation_1196.seal.json" ]]; then
  if [[ -e "${sanitized}" ]]; then
    printf 'Incomplete sanitation directory exists; inspect before retry: %s\n' \
      "${sanitized}" >&2
    exit 2
  fi
  (
    "${python_bin}" "${sanitize_script}" \
      --input "${selection}/supplemental_targets_unsanitized_1196.jsonl" \
      --expected-input-sha256 "${supplemental_targets_sha}" \
      --input-seal "${selection}/supplemental_targets_unsanitized_1196.seal.json" \
      --expected-input-seal-sha256 "${supplemental_targets_seal_sha}" \
      --supplemental-manifest "${selection}/supplemental_task_manifest_1196.jsonl" \
      --expected-supplemental-manifest-sha256 "${supplemental_manifest_sha}" \
      --selection-seal "${selection}/expansion_selection.seal.json" \
      --expected-selection-seal-sha256 "${selection_sha}" \
      --contract "${parent}/multifunction_inline_cfg_v2_contract.json" \
      --expected-contract-sha256 14ece4b23b8b2193c4a5b05716f86b7ca0fa05449baece0cf00153c91ebaa7cb \
      --evaluator "${patch}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
      --expected-evaluator-sha256 249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 \
      --expected-dart-version 'Dart SDK version: 3.12.2 (stable) (Tue Jun 9 01:11:39 2026 -0700) on "linux_x64"' \
      --output-dir "${sanitized}" \
      --workers "${SUPPLEMENTAL_SANITIZE_WORKERS:-24}" \
      --timeout 30 \
      --stability-runs 2 \
      2>&1 | tee "${logs}/sanitation.log"
  ) &
  sanitize_pid=$!
fi

mkdir -p "${extraction}/receipts"
bundles="${extraction}/user_function_bundles_1196.jsonl"
extraction_report="${extraction}/preflight_1196.json"
if [[ ! -f "${bundles}" || ! -f "${extraction_report}" ]]; then
  test ! -e "${bundles}" && test ! -e "${extraction_report}"
  "${python_bin}" /workspace/scripts/data/extract_dart_aot_user_function_bundle.py \
    --aot-manifest "${selection}/supplemental_aot_manifest_1196.jsonl" \
    --aot-manifest-sha256 "${supplemental_aot_sha}" \
    --aot-root /workspace/compact_qwen_pool_v3_full_dart3122_v1 \
    --symbol-attestation "${attestation}" \
    --symbol-attestation-sha256 "${attestation_sha}" \
    --symbol-attestation-key-file "${attestation_key}" \
    --expected-rows 1196 \
    --output-jsonl "${bundles}" \
    --receipt-dir "${extraction}/receipts" \
    --report "${extraction_report}" \
    --failures-jsonl "${extraction}/failures_1196.jsonl" \
    --gdb /usr/bin/gdb \
    --root-symbol candidate \
    --workers "${MULTIFUNCTION_EXPANSION_EXTRACT_WORKERS:-32}" \
    2>&1 | tee "${logs}/extraction.log"
fi
require_rows "${bundles}" 1196 supplemental-function-bundles
bundles_sha="$(sha "${bundles}")"
jq -e \
  --arg aot "${supplemental_aot_sha}" \
  --arg attestation "${attestation_sha}" \
  '.schema == "dart-aot-user-function-corpus-preflight-v1"
   and .passed == true
   and .manifest_rows == 1196
   and .extracted_rows == 1196
   and .failed_rows == 0
   and .aot_manifest_sha256 == $aot
   and .symbol_attestation.file_sha256 == $attestation
   and .symbol_attestation.selected_rows == 1196' \
  "${extraction_report}" >/dev/null

constants_jsonl="${constants}/attested_pool_constants_1196.jsonl"
constants_report="${constants}/report_1196.json"
if [[ ! -f "${constants_jsonl}" || ! -f "${constants_report}" ]]; then
  if [[ -e "${constants}" ]]; then
    printf 'Incomplete constants directory exists; inspect before retry: %s\n' \
      "${constants}" >&2
    exit 2
  fi
  "${python_bin}" /workspace/scripts/data/extract_attested_binary_pool_constants.py \
    --aot-manifest "${selection}/supplemental_aot_manifest_1196.jsonl" \
    --aot-manifest-sha256 "${supplemental_aot_sha}" \
    --aot-root /workspace/compact_qwen_pool_v3_full_dart3122_v1 \
    --function-bundles "${bundles}" \
    --function-bundles-sha256 "${bundles_sha}" \
    --gdb-script /workspace/scripts/data/gdb_dump_attested_pool_offsets.py \
    --gdb-script-sha256 7df848639591b8646c800c46bf9babe07dce93223509203188fa2a5ab90af09c \
    --gdb /usr/bin/gdb \
    --runtime /workspace/dart-3.12.2/usr/bin/dartaotruntime \
    --expected-rows 1196 \
    --workers "${MULTIFUNCTION_EXPANSION_CONSTANT_WORKERS:-32}" \
    --timeout-seconds 120 \
    --output-jsonl "${constants_jsonl}" \
    --report "${constants_report}" \
    2>&1 | tee "${logs}/constants.log"
fi
require_rows "${constants_jsonl}" 1196 supplemental-binary-constants
constants_sha="$(sha "${constants_jsonl}")"
jq -e \
  --arg aot "${supplemental_aot_sha}" \
  --arg bundles "${bundles_sha}" \
  --arg gdb 7df848639591b8646c800c46bf9babe07dce93223509203188fa2a5ab90af09c \
  --arg output "${constants_sha}" \
  '.schema == "dart-aot-attested-pool-constants-report-v1"
   and .passed == true
   and .rows == 1196
   and .expected_rows == 1196
   and .input_hashes.aot_manifest_sha256 == $aot
   and .input_hashes.function_bundles_sha256 == $bundles
   and .input_hashes.gdb_script_sha256 == $gdb
   and .output_jsonl_sha256 == $output' \
  "${constants_report}" >/dev/null

if [[ -n "${sanitize_pid}" ]]; then
  wait "${sanitize_pid}"
fi
require_rows "${sanitized}/supplemental_train_imitation_1196.jsonl" 1196 sanitized-supplemental-imitation
sanitized_sha="$(sha "${sanitized}/supplemental_train_imitation_1196.jsonl")"
sanitized_seal_sha="$(sha "${sanitized}/supplemental_train_imitation_1196.seal.json")"
test "$(jq -r '.passed' "${sanitized}/sanitation_report.json")" = true

if [[ ! -f "${build}/expansion_build.seal.json" ]]; then
  if [[ -e "${build}" ]]; then
    printf 'Incomplete expansion build exists; inspect before retry: %s\n' \
      "${build}" >&2
    exit 2
  fi
  "${python_bin}" "${build_script}" \
    --supplemental-base "${sanitized}/supplemental_train_imitation_1196.jsonl" \
    --expected-supplemental-base-sha256 "${sanitized_sha}" \
    --supplemental-base-seal "${sanitized}/supplemental_train_imitation_1196.seal.json" \
    --expected-supplemental-base-seal-sha256 "${sanitized_seal_sha}" \
    --supplemental-manifest "${selection}/supplemental_task_manifest_1196.jsonl" \
    --expected-supplemental-manifest-sha256 "${supplemental_manifest_sha}" \
    --selection-seal "${selection}/expansion_selection.seal.json" \
    --expected-selection-seal-sha256 "${selection_sha}" \
    --function-bundles "${bundles}" \
    --expected-function-bundles-sha256 "${bundles_sha}" \
    --constants "${constants_jsonl}" \
    --expected-constants-sha256 "${constants_sha}" \
    --extractor-script /workspace/scripts/data/extract_dart_aot_user_function_bundle.py \
    --expected-extractor-script-sha256 ff3cd323eb3045da0a9bf8b3489f8e867cbf3ae3dfd12181fc6c2004423af2a5 \
    --adapter-script "${patch}/scripts/preprocessing/build_multifunction_binary_compact.py" \
    --expected-adapter-script-sha256 bb954a0b5aafe5fa51c97cce40d25c80dd6f65f3fa2696d289f31dcbdf4fae66 \
    --frozen-contract "${parent}/multifunction_inline_cfg_v2_contract.json" \
    --expected-frozen-contract-sha256 14ece4b23b8b2193c4a5b05716f86b7ca0fa05449baece0cf00153c91ebaa7cb \
    --frozen-codebook "${parent}/multifunction_inline_cfg_v2_codebook.json" \
    --expected-frozen-codebook-sha256 c36f9aadc3308c9916070c2088704d4b5bf6410be84d8df6c7831ed448ed0e14 \
    --tokenizer-json "${tokenizer}" \
    --expected-tokenizer-sha256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    --inline-cfg-codec /workspace/scripts/data/build_multifunction_compact_v2.py \
    --expected-inline-cfg-codec-sha256 b2e0e33a56c470ac54257a0fc2124bcc2b9d58639a58416d33c7fbbf74d2ca52 \
    --frontier-f2 /workspace/frontier_ceiling_patch_v1/frontier_f2.py \
    --expected-frontier-f2-sha256 097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce \
    --parent-dataset "${parent}/train_multifunction_binary.jsonl" \
    --expected-parent-dataset-sha256 3cf3dd5b0fc950f7516434b2c90ce2bd6334178ace962e4442026bd33f476e18 \
    --parent-seal "${parent}/train_multifunction_binary.seal.json" \
    --expected-parent-seal-sha256 14f2cc5ddcb63c8ad25f6a5c9bb6209af111dafbb0827523a24d8089585a94a5 \
    --parent-f2 "${parent}/train_multifunction_binary_f2.jsonl" \
    --expected-parent-f2-sha256 f86032533f7f3e1bade9280e5bf6e28858daf25e34167837fec160a34b2370ba \
    --parent-f2-manifest "${parent}/train_multifunction_binary_f2.jsonl.manifest.json" \
    --expected-parent-f2-manifest-sha256 231227e9aa6f31c793ff097e8167972975e9f94871c9ea682d6a0dc996ed4af8 \
    --heldout-dataset "${parent}/dev_multifunction_binary.jsonl" \
    --expected-heldout-dataset-sha256 abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
    --heldout-seal "${parent}/dev_multifunction_binary.seal.json" \
    --expected-heldout-seal-sha256 5c3497a9de1d6a478c3d3f104c3942ba4cec03272f82dc12ff8b1e99ed7c1e4a \
    --output-dir "${build}" \
    --api-prompt-token-limit 12000 \
    --chat-overhead-reserve 256 \
    2>&1 | tee "${logs}/build.log"
fi

require_rows "${build}/train_multifunction_binary_supplemental_1196.jsonl" 1196 supplemental-representation
require_rows "${build}/train_multifunction_binary_supplemental_1196_f2.jsonl" 1196 supplemental-f2
require_rows "${build}/train_multifunction_binary_expanded_2776.jsonl" 2776 expanded-representation
require_rows "${build}/train_multifunction_binary_expanded_2776_f2.jsonl" 2776 expanded-f2
test "$(jq -r '.passed' "${build}/expansion_build.seal.json")" = true
test "$(jq -r '.invariants.parent_dataset_bytes_exact_prefix' "${build}/expansion_build.seal.json")" = true
test "$(jq -r '.invariants.parent_f2_bytes_exact_prefix' "${build}/expansion_build.seal.json")" = true

printf 'EXPANDED_2776_READY root=%s dataset_sha256=%s f2_sha256=%s\n' \
  "${root}" \
  "$(sha "${build}/train_multifunction_binary_expanded_2776.jsonl")" \
  "$(sha "${build}/train_multifunction_binary_expanded_2776_f2.jsonl")"
