#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:$PATH"

SOURCE="data/testing/fresh_eval_low_topup_chatgpt_s46.jsonl"
OUT="data/testing/fresh_eval_low_topup_chatgpt_s46_assembly.jsonl"
STATUS="logs/fresh_eval_low_topup_chatgpt_s46_assembly.status"
PID_FILE="logs/fresh_eval_low_topup_chatgpt_s46_assembly.pid"

if [[ ! -f "$SOURCE" ]]; then
  printf 'REFUSED\nreason=source_missing\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

if [[ $(wc -l < "$SOURCE") -ne 200 ]]; then
  printf 'REFUSED\nreason=source_row_count_mismatch\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

if [[ -e "$OUT" ]]; then
  printf 'REFUSED\nreason=output_exists\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

cp --reflink=auto "$SOURCE" "$OUT"
printf '%s\n' "$$" > "$PID_FILE"
printf 'RUNNING\nstarted_at=%s\ninput_rows=200\ndart=3.11.5\ngdb=17.1\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

set +e
"$ROOT/.venv/bin/python" "$ROOT/generate_fresh_eval_tasks.py" \
  --resume_assembly \
  --out "$ROOT/$OUT" \
  --benchmark "$ROOT/data/testing/grpo_data_graphv2.jsonl" \
  --synthetic "$ROOT/data/datasets/synthetic_pool_graphv2.jsonl" \
  --dart_bin /usr/local/bin/dart
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  printf 'DONE\nexit_code=0\nended_at=%s\ninput_rows=200\ndart=3.11.5\ngdb=17.1\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
else
  printf 'FAILED\nexit_code=%s\nended_at=%s\ninput_rows=200\ndart=3.11.5\ngdb=17.1\n' \
    "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
fi

exit "$rc"
