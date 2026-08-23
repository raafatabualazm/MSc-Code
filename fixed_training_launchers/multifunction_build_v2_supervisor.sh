#!/bin/bash
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1091
. "${utils}/logging.sh" ""
# shellcheck disable=SC1091
. "${utils}/environment.sh"

source /venv/main/bin/activate
cd /workspace
mkdir -p /workspace/logs

output=/workspace/multifunction_v1/build
if [[ -e "${output}" ]]; then
  printf 'Refusing to overwrite existing multi-function build: %s\n' "${output}" >&2
  exit 2
fi
test "$(
  sha256sum \
    /workspace/hybrid_training_patch_v2_3/scripts/preprocessing/build_multifunction_binary_compact.py \
    | cut -d ' ' -f1
)" = "bb954a0b5aafe5fa51c97cce40d25c80dd6f65f3fa2696d289f31dcbdf4fae66"

pty /venv/main/bin/python \
  /workspace/hybrid_training_patch_v2_3/scripts/preprocessing/build_multifunction_binary_compact.py \
  --base-train /workspace/temp/codex_harness_sanitize_dual_20260723/output/train_fn0_whole_real_imitation.jsonl \
  --expected-base-train-sha256 0a2d94e279ae04cb1e9ea656ff5222401e6970f41a519e9234ea74f8053c3db4 \
  --base-train-seal /workspace/temp/codex_harness_sanitize_dual_20260723/output/train_fn0_whole_real_imitation.seal.json \
  --expected-base-train-seal-sha256 0bfd2007657a7523f2754a42d9527501dd1fa76ca5c3f6b876b63768e2b18c32 \
  --base-dev /workspace/temp/codex_harness_sanitize_dual_20260723/output/dev_fn0_whole_real_executable.jsonl \
  --expected-base-dev-sha256 eb39335541dc4efc385a69a138d774bd1777c6743d7e423bf265db8ce7c7bc88 \
  --base-dev-seal /workspace/temp/codex_harness_sanitize_dual_20260723/output/dev_fn0_whole_real_executable.seal.json \
  --expected-base-dev-seal-sha256 8531dc6e1ed871b0a879ea50f296d703cbb309bcac1093c73b8f234c3493abf1 \
  --function-bundles /workspace/multifunction_v1/extraction_v2/user_function_bundles_1755.jsonl \
  --expected-function-bundles-sha256 d2a019fe14e500bf1d242367e3b52b644f3e166bb8e3b5ad47e980e6ccb688d2 \
  --constants /workspace/multifunction_v1/constants_v5/attested_pool_constants_1755.jsonl \
  --expected-constants-sha256 2b5dc0d353e5f7cb70bb79cb398406b16a92b524e4252fdbca01bd48a7c7b857 \
  --extractor-script /workspace/scripts/data/extract_dart_aot_user_function_bundle.py \
  --expected-extractor-script-sha256 ff3cd323eb3045da0a9bf8b3489f8e867cbf3ae3dfd12181fc6c2004423af2a5 \
  --contract /workspace/artifacts/compact_fn0_rebuild/fn0_contract.json \
  --expected-contract-sha256 4801767387e4312cced559166c2fbf7145242ab21b2b35883a54c9f99f367e02 \
  --codebook /workspace/direct_compact_stage/scrubbed_master_v2_release/direct_compact_split_v1/compact_qwen_confirmatory_v1/codebook.json \
  --expected-codebook-sha256 d44f9be95debe6e7d8766bf434cf9aeabd89a3d6ca5b09a06e3c50272543e76c \
  --tokenizer-json /workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json \
  --expected-tokenizer-sha256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
  --codec /workspace/direct_compact_stage/scripts/data/build_compact_qwen_v1.py \
  --expected-codec-sha256 b9f439b366f60fa646efc0363d483ca216879f95ee960fbf343da2e7e93f2cf4 \
  --inline-cfg-codec /workspace/scripts/data/build_multifunction_compact_v2.py \
  --expected-inline-cfg-codec-sha256 b2e0e33a56c470ac54257a0fc2124bcc2b9d58639a58416d33c7fbbf74d2ca52 \
  --frontier-f2 /workspace/frontier_ceiling_patch_v1/frontier_f2.py \
  --expected-frontier-f2-sha256 097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce \
  --output-dir "${output}" \
  --expected-train-rows 1580 \
  --expected-dev-rows 175 \
  --student-token-limit 9000 \
  --api-prompt-token-limit 12000 \
  --chat-overhead-reserve 256 \
  2>&1 | tee -a /workspace/logs/multifunction_build_v2.log
