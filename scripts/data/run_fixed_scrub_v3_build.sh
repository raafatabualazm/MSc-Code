#!/usr/bin/env bash
# Build both v3 opaque-signature arms from the frozen Regions16 benchmark.
set -Eeuo pipefail

ROOT="${1:-/root/fixed_scrub_build}"
PYTHON_BIN="${PYTHON_BIN:-/root/experiment_workspace/.venv/bin/python}"
DATA="$ROOT/data/testing"
LOGS="$ROOT/logs/fixed_scrub_v3"
PRIVATE_STATE="$ROOT/private"
STATUS="$ROOT/fixed_scrub_v3_build.status"
INPUT="$DATA/grpo_data_graphv2.jsonl"
BUILDER="$ROOT/scripts/data/build_signature_scrubbed_eval.py"
EXPECTED_INPUT_SHA256="8453876a40d2279684a190a5bf1430a62897c84e063a78e25c57198287bc6928"
EXPECTED_EXTRACTOR_SHA256="3b522afc7ea9d24440c4ed0e1bafd2c4a047bb76f0f592560451b61e10c2613d"
TARGET_NAME="fn0"
SHUFFLE_SEED=4343

mkdir -p "$LOGS" "$PRIVATE_STATE"
chmod 700 "$PRIVATE_STATE"

fail_status() {
  local rc=$?
  printf 'FAILED stage=%s rc=%s time=%s\n' "${STAGE:-unknown}" "$rc" "$(date -u +%FT%TZ)" > "$STATUS"
  exit "$rc"
}
trap fail_status ERR

STAGE=preflight
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
test -s "$INPUT"
test -s "$BUILDER"
test -x "$PYTHON_BIN"
test "$(sha256sum "$INPUT" | awk '{print $1}')" = "$EXPECTED_INPUT_SHA256"

DART_VERSION="$(/usr/local/bin/dart --version 2>&1)"
GDB_VERSION="$(/usr/local/bin/gdb --version | head -n 1)"
[[ "$DART_VERSION" == "Dart SDK version: 3.11.5 (stable)"* ]]
[[ "$GDB_VERSION" == *"17.1"* ]]

SALT_FILE="$PRIVATE_STATE/fixed_scrub_v3_id_salt"
if [[ ! -s "$SALT_FILE" ]]; then
  umask 077
  openssl rand -hex 32 > "$SALT_FILE"
fi
chmod 600 "$SALT_FILE"
ID_SALT="$(tr -d '\r\n' < "$SALT_FILE")"
[[ "$ID_SALT" =~ ^[0-9a-f]{64}$ ]]

build_arm() {
  local arm="$1"
  local mode="$2"
  local private="$DATA/grpo_data_graphv2_sigscrub_v3_${arm}_private.jsonl"
  local public="$DATA/grpo_data_graphv2_sigscrub_v3_${arm}_public.jsonl"
  local rejects="$DATA/grpo_data_graphv2_sigscrub_v3_${arm}_rejects.json"

  STAGE="build_${arm}"
  printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
  "$PYTHON_BIN" "$BUILDER" \
    --input "$INPUT" \
    --output "$private" \
    --public_output "$public" \
    --rejects "$rejects" \
    --benchmark_kind existing_ablation \
    --protocol_schema dart-signature-scrubbed-v3 \
    --expected_rows 154 \
    --workers 4 \
    --timeout 120 \
    --id_salt "$ID_SALT" \
    --target_name "$TARGET_NAME" \
    --redact_effective_id_salt \
    --public_signature_mode "$mode" \
    --shuffle_public_seed "$SHUFFLE_SEED" \
    --assembly_mode rename_frozen \
    --frozen_dart_version "$DART_VERSION" \
    --frozen_gdb_version "$GDB_VERSION" \
    --frozen_toolchain_version "dart-3.11.5|gdb-17.1|extractor-$EXPECTED_EXTRACTOR_SHA256" \
    --max_block_instrs 20 \
    --max_dataflow_edges 0 \
    --dart_bin /usr/local/bin/dart \
    --gdb_bin /usr/local/bin/gdb \
    > "$LOGS/build_${arm}.log" 2>&1
}

build_arm opaque_nameonly name_only
build_arm opaque_neutralexact neutral_exact

STAGE=manifest
printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
sha256sum \
  "$INPUT" \
  "$BUILDER" \
  "$ROOT/scripts/data/test_signature_scrubbed_eval.py" \
  "$DATA"/grpo_data_graphv2_sigscrub_v3_opaque_nameonly_* \
  "$DATA"/grpo_data_graphv2_sigscrub_v3_opaque_neutralexact_* \
  "$LOGS"/build_*.log \
  > "$ROOT/fixed_scrub_v3_build_sha256.txt"

printf 'COMPLETE time=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
