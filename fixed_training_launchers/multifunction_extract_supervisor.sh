#!/bin/bash
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1091
. "${utils}/logging.sh" ""
# shellcheck disable=SC1091
. "${utils}/environment.sh"

source /venv/main/bin/activate
cd /workspace

manifest=/workspace/multifunction_v1/input/aot_manifest_1755.jsonl
expected_manifest_sha=22721f6da58ae04c83b049b34877009e132455996197e1cbf763f4f4341cbf5d
observed_manifest_sha="$(sha256sum "${manifest}" | cut -d ' ' -f1)"
if [[ "${observed_manifest_sha}" != "${expected_manifest_sha}" ]]; then
  printf 'AOT subset manifest hash mismatch: %s != %s\n' \
    "${observed_manifest_sha}" "${expected_manifest_sha}" >&2
  exit 2
fi

output=/workspace/multifunction_v1/extraction
mkdir -p "${output}/receipts"
mkdir -p /workspace/logs

pty /venv/main/bin/python \
  /workspace/scripts/data/extract_dart_aot_user_function_bundle.py \
  --aot-manifest "${manifest}" \
  --aot-manifest-sha256 "${expected_manifest_sha}" \
  --aot-root /workspace/compact_qwen_pool_v3_full_dart3122_v1 \
  --expected-rows 1755 \
  --output-jsonl "${output}/user_function_bundles_1755.jsonl" \
  --receipt-dir "${output}/receipts" \
  --report "${output}/preflight_1755.json" \
  --failures-jsonl "${output}/failures_1755.jsonl" \
  --gdb /usr/bin/gdb \
  --root-symbol candidate \
  --workers "${MULTIFUNCTION_EXTRACT_WORKERS:-32}" \
  2>&1 | tee -a /workspace/logs/multifunction_extract.log
