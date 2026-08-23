#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="data/testing/fresh_eval_low_topup_chatgpt_s46.jsonl"
LOG="logs/fresh_eval_low_topup_chatgpt_s46.log"
STATUS="logs/fresh_eval_low_topup_chatgpt_s46.status"
PID_FILE="logs/fresh_eval_low_topup_chatgpt_s46.pid"

if [[ -e "$OUT" || -e "$OUT.manifest.json" || -e "$OUT.rejects.jsonl" ]]; then
  printf 'REFUSED\nreason=output_exists\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

printf '%s\n' "$$" > "$PID_FILE"
printf 'RUNNING\nstarted_at=%s\nprovider=azure\nmodel=gpt-chat-latest\ntarget=200\nseed=46\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

set +e
"$ROOT/.venv/bin/python" "$ROOT/generate_fresh_eval_tasks.py" \
  --num_tasks 200 \
  --oversample 8 \
  --providers azure \
  --azure_models gpt-chat-latest \
  --benchmark "$ROOT/data/testing/grpo_data_graphv2.jsonl" \
  --synthetic "$ROOT/data/datasets/synthetic_pool_graphv2.jsonl" \
  --decontam_jsonl "$ROOT/data/testing/fresh_eval_llm.jsonl" \
  --decontam_jsonl "$ROOT/data/testing/fresh_eval_lowmid_topup_s44.jsonl" \
  --decontam_jsonl "$ROOT/data/testing/fresh_eval_low_topup_deepseek_s45.jsonl" \
  --out "$ROOT/$OUT" \
  --workers 4 \
  --jac_thr 0.55 \
  --seq_thr 0.70 \
  --strata_mix low:1,mid:0,high:0 \
  --stability_runs 2 \
  --mutation_max 8 \
  --mutation_min_kill 0.5 \
  --shape_gate 1 \
  --loc_tol 1.0 \
  --branch_tol 1.0 \
  --rng_seed 46 \
  --dart_bin /usr/local/bin/dart
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  printf 'DONE\nexit_code=0\nended_at=%s\nprovider=azure\nmodel=gpt-chat-latest\ntarget=200\nseed=46\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
else
  printf 'FAILED\nexit_code=%s\nended_at=%s\nprovider=azure\nmodel=gpt-chat-latest\ntarget=200\nseed=46\n' \
    "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
fi

exit "$rc"
