#!/usr/bin/env bash
# Re-score the frozen exact-signature comparator with the v3 JIT classifier.
# This script intentionally performs no inference and never mutates the input.
set -Eeuo pipefail

ROOT="${1:-/root/fixed_scrub_build}"
PREDICTIONS="${2:-$ROOT/results/v3_comparator_rescore/comparator_predictions.json}"
OUT="${3:-$ROOT/results/v3_comparator_rescore}"
PYTHON_BIN="${PYTHON_BIN:-/root/experiment_workspace/.venv/bin/python}"
WORKERS="${EVAL_DART_WORKERS:-4}"
EXPECTED_DART_PREFIX="Dart SDK version: 3.11.5 (stable)"
STATUS="$OUT/comparator_rescore.status"
LOG="$OUT/comparator_rescore.log"

mkdir -p "$OUT"

fail_status() {
  local rc=$?
  printf 'FAILED stage=%s rc=%s time=%s\n' "${STAGE:-unknown}" "$rc" "$(date -u +%FT%TZ)" > "$STATUS"
  exit "$rc"
}
trap fail_status ERR

run() {
  STAGE=preflight
  printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
  test -s "$PREDICTIONS"
  test -x "$PYTHON_BIN"
  local dart_version
  dart_version="$(dart --version 2>&1)"
  [[ "$dart_version" == "$EXPECTED_DART_PREFIX"* ]]
  printf '%s\n' "$dart_version" > "$OUT/dart_version.txt"
  "$PYTHON_BIN" -m py_compile \
    "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
    "$ROOT/scripts/evaluation/graph_pass_at_k_antigravity.py" \
    "$ROOT/scripts/evaluation/compile_statistical_results_antigravity.py"

  STAGE=compile_at_k
  printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
  "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
    --predictions "$PREDICTIONS" --k_values 1,5 --workers "$WORKERS" \
    --timeout 30 --compile_mode jit_tests > "$OUT/comparator_compile_at_k.txt"

  STAGE=pass_at_k
  printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
  "$PYTHON_BIN" "$ROOT/scripts/evaluation/graph_pass_at_k_antigravity.py" \
    --predictions "$PREDICTIONS" --k_values 1,5,10 --workers "$WORKERS" \
    --timeout 30 > "$OUT/comparator_pass_at_k.txt"

  STAGE=stats
  printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
  "$PYTHON_BIN" "$ROOT/scripts/evaluation/compile_statistical_results_antigravity.py" \
    --predictions "$PREDICTIONS" --output "$OUT/comparator_stats.csv" \
    --workers "$WORKERS" --timeout 30 --compile_mode jit_tests

  STAGE=manifest
  printf 'RUNNING stage=%s time=%s\n' "$STAGE" "$(date -u +%FT%TZ)" > "$STATUS"
  sha256sum \
    "$PREDICTIONS" \
    "$ROOT/scripts/evaluation/graph_compile_at_k_antigravity.py" \
    "$ROOT/scripts/evaluation/graph_pass_at_k_antigravity.py" \
    "$ROOT/scripts/evaluation/compile_statistical_results_antigravity.py" \
    "$OUT/dart_version.txt" \
    "$OUT/comparator_compile_at_k.txt" \
    "$OUT/comparator_pass_at_k.txt" \
    "$OUT/comparator_stats.csv" > "$OUT/comparator_rescore_sha256.txt"
  printf 'COMPLETE time=%s\n' "$(date -u +%FT%TZ)" > "$STATUS"
}

run > "$LOG" 2>&1
