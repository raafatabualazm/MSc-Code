#!/bin/bash
set -Eeuo pipefail

source /venv/main/bin/activate
cd /workspace

audit_root=/workspace/tmp/f2_roundtrip_audit_20260723_1648
audit_script=/workspace/tmp/exhaustive_f2_sweep_20260723.py

test "$(
  sha256sum "${audit_script}" | cut -d ' ' -f1
)" = "51f2792f50f67d3f5408bddb0627e5fc689fef32457ca3267f9b895c843de3f7"

if [[ -e "${audit_root}" ]]; then
  printf 'Refusing to overwrite existing audit root: %s\n' "${audit_root}" >&2
  exit 2
fi
mkdir -p "${audit_root}"

exec /venv/main/bin/python "${audit_script}" \
  --builder /workspace/hybrid_training_patch_v2_3/scripts/preprocessing/build_multifunction_binary_compact.py \
  --extractor /workspace/scripts/data/extract_dart_aot_user_function_bundle.py \
  --f2 /workspace/frontier_ceiling_patch_v1/frontier_f2.py \
  --tokenizer /workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json \
  --bundles /workspace/multifunction_v1/extraction_v2/user_function_bundles_1755.jsonl \
  --constants /workspace/multifunction_v1/constants_v5/attested_pool_constants_1755.jsonl \
  --journal "${audit_root}/task_results.jsonl" \
  --summary "${audit_root}/summary.json" \
  --progress-every 25
