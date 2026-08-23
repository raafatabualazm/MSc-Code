#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="/tmp/fresh_graphv2_holdout_s44_chatgpt_s46_determinism.jsonl"
MANIFEST="/tmp/fresh_graphv2_holdout_s44_chatgpt_s46_determinism.manifest.json"
STATUS="logs/fresh_graphv2_holdout_s44_chatgpt_s46_determinism.status"
PID_FILE="logs/fresh_graphv2_holdout_s44_chatgpt_s46_determinism.pid"

if [[ ! -f data/testing/fresh_graphv2_holdout_s44.jsonl ||
      ! -f data/testing/fresh_graphv2_holdout_s44.manifest.json ]]; then
  printf 'REFUSED\nreason=canonical_seal_missing\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

if [[ -e "$OUTPUT" || -e "$MANIFEST" ]]; then
  printf 'REFUSED\nreason=determinism_output_exists\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

printf '%s\n' "$$" > "$PID_FILE"
printf 'RUNNING\nstarted_at=%s\nseed=44\nquotas=low:170,mid:170,high:160\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

set +e
"$ROOT/.venv/bin/python" "$ROOT/scripts/data/seal_fresh_graphv2_holdout.py" \
  --pool base_llm=data/testing/fresh_eval_llm_graphv2.jsonl \
  --pool topup_s44=data/testing/fresh_eval_lowmid_topup_s44_graphv2.jsonl \
  --pool topup_s45=data/testing/fresh_eval_low_topup_deepseek_s45_graphv2.jsonl \
  --pool topup_chatgpt_s46=data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.jsonl \
  --exclude data/testing/grpo_data_graphv2.jsonl \
  --exclude data/datasets/synthetic_pool_graphv2.jsonl \
  --output "$OUTPUT" \
  --manifest "$MANIFEST" \
  --seed 44 \
  --low 170 \
  --mid 170 \
  --high 160 \
  --low-max 14 \
  --mid-max 25 \
  --jac-threshold 0.55 \
  --sequence-threshold 0.70
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  printf 'DONE\nexit_code=0\nended_at=%s\nseed=44\nquotas=low:170,mid:170,high:160\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
else
  printf 'FAILED\nexit_code=%s\nended_at=%s\nseed=44\nquotas=low:170,mid:170,high:160\n' \
    "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
fi

exit "$rc"
