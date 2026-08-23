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

output=/workspace/multifunction_v1/constants_v5
if [[ -e "${output}" ]]; then
  printf 'Refusing to overwrite existing constants output: %s\n' "${output}" >&2
  exit 2
fi

test "$(
  sha256sum /workspace/scripts/data/extract_attested_binary_pool_constants.py \
    | cut -d ' ' -f1
)" = "a05238b0fbdb0cb8e0576d757e4d531fbaf67e146d30c73785e289d257db54f1"
test "$(
  sha256sum /workspace/scripts/data/gdb_dump_attested_pool_offsets.py \
    | cut -d ' ' -f1
)" = "7df848639591b8646c800c46bf9babe07dce93223509203188fa2a5ab90af09c"

pty /venv/main/bin/python \
  /workspace/scripts/data/extract_attested_binary_pool_constants.py \
  --aot-manifest /workspace/multifunction_v1/input/aot_manifest_1755.jsonl \
  --aot-manifest-sha256 22721f6da58ae04c83b049b34877009e132455996197e1cbf763f4f4341cbf5d \
  --aot-root /workspace/compact_qwen_pool_v3_full_dart3122_v1 \
  --function-bundles /workspace/multifunction_v1/extraction_v2/user_function_bundles_1755.jsonl \
  --function-bundles-sha256 d2a019fe14e500bf1d242367e3b52b644f3e166bb8e3b5ad47e980e6ccb688d2 \
  --gdb-script /workspace/scripts/data/gdb_dump_attested_pool_offsets.py \
  --gdb-script-sha256 7df848639591b8646c800c46bf9babe07dce93223509203188fa2a5ab90af09c \
  --gdb /usr/bin/gdb \
  --runtime /workspace/dart-3.12.2/usr/bin/dartaotruntime \
  --expected-rows 1755 \
  --workers "${MULTIFUNCTION_CONSTANT_WORKERS:-32}" \
  --timeout-seconds 120 \
  --output-jsonl "${output}/attested_pool_constants_1755.jsonl" \
  --report "${output}/report_1755.json" \
  2>&1 | tee -a /workspace/logs/multifunction_constants_v2.log
