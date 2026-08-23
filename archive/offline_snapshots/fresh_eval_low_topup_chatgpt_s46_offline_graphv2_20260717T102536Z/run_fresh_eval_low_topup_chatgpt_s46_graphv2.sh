#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="data/testing/fresh_eval_low_topup_chatgpt_s46_assembly.jsonl"
OUTPUT="data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.jsonl"
REJECTED="data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.rejected.jsonl"
SUMMARY="data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.summary.json"
STATUS="logs/fresh_eval_low_topup_chatgpt_s46_graphv2.status"
PID_FILE="logs/fresh_eval_low_topup_chatgpt_s46_graphv2.pid"

if [[ ! -f "$INPUT" || $(wc -l < "$INPUT") -ne 200 ]]; then
  printf 'REFUSED\nreason=input_missing_or_row_count_mismatch\nended_at=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

if [[ -e "$OUTPUT" || -e "$REJECTED" || -e "$SUMMARY" ]]; then
  printf 'REFUSED\nreason=output_exists\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

networkx_version="$("$ROOT/.venv/bin/python" -c 'import networkx; print(networkx.__version__)')"
if [[ "$networkx_version" != "3.6.1" ]]; then
  printf 'REFUSED\nreason=networkx_version_mismatch\nobserved=%s\nended_at=%s\n' \
    "$networkx_version" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

printf '%s\n' "$$" > "$PID_FILE"
printf 'RUNNING\nstarted_at=%s\ninput_rows=200\nnetworkx=3.6.1\nmax_block_instrs=20\ndrop_invalid=true\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

set +e
"$ROOT/.venv/bin/python" "$ROOT/scripts/data/build_graph_v2_jsonl.py" \
  --input "$ROOT/$INPUT" \
  --output "$ROOT/$OUTPUT" \
  --rejected "$ROOT/$REJECTED" \
  --summary "$ROOT/$SUMMARY" \
  --drop_invalid \
  --expected_input_rows 200 \
  --max_block_instrs 20
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  printf 'DONE\nexit_code=0\nended_at=%s\ninput_rows=200\nnetworkx=3.6.1\nmax_block_instrs=20\ndrop_invalid=true\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
else
  printf 'FAILED\nexit_code=%s\nended_at=%s\ninput_rows=200\nnetworkx=3.6.1\nmax_block_instrs=20\ndrop_invalid=true\n' \
    "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
fi

exit "$rc"
