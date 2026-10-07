#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="data/testing/fresh_graphv2_holdout_s44.jsonl"
MANIFEST="data/testing/fresh_graphv2_holdout_s44.manifest.json"
STATUS="logs/fresh_graphv2_holdout_s44_chatgpt_s46.status"
PID_FILE="logs/fresh_graphv2_holdout_s44_chatgpt_s46.pid"

if [[ -e "$OUTPUT" || -e "$MANIFEST" ]]; then
  printf 'REFUSED\nreason=output_exists\nended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
  exit 2
fi

declare -A expected_rows=(
  [data/testing/fresh_eval_llm_graphv2.jsonl]=588
  [data/testing/fresh_eval_lowmid_topup_s44_graphv2.jsonl]=324
  [data/testing/fresh_eval_low_topup_deepseek_s45_graphv2.jsonl]=558
  [data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.jsonl]=196
)
for path in "${!expected_rows[@]}"; do
  if [[ ! -f "$path" || $(wc -l < "$path") -ne ${expected_rows[$path]} ]]; then
    printf 'REFUSED\nreason=pool_missing_or_row_count_mismatch\npath=%s\nended_at=%s\n' \
      "$path" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
    exit 2
  fi
done

printf '%s\n' "$$" > "$PID_FILE"
printf 'RUNNING\nstarted_at=%s\ncandidate_rows=1666\nseed=44\nquotas=low:170,mid:170,high:160\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

set +e
"$ROOT/.venv/bin/python" "$ROOT/scripts/data/seal_fresh_graphv2_holdout.py" \
  --pool base_llm=data/testing/fresh_eval_llm_graphv2.jsonl \
  --pool topup_s44=data/testing/fresh_eval_lowmid_topup_s44_graphv2.jsonl \
  --pool topup_s45=data/testing/fresh_eval_low_topup_deepseek_s45_graphv2.jsonl \
  --pool topup_chatgpt_s46=data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.jsonl \
  --exclude data/testing/grpo_data_graphv2.jsonl \
  --exclude data/datasets/synthetic_pool_graphv2.jsonl \
  --provenance data/testing/fresh_eval_llm.jsonl \
  --provenance data/testing/fresh_eval_llm_graphv2.summary.json \
  --provenance data/testing/fresh_eval_llm_graphv2.rejected.jsonl \
  --provenance data/testing/fresh_eval_lowmid_topup_s44.jsonl \
  --provenance data/testing/fresh_eval_lowmid_topup_s44.jsonl.manifest.json \
  --provenance data/testing/fresh_eval_lowmid_topup_s44.jsonl.distribution.json \
  --provenance data/testing/fresh_eval_lowmid_topup_s44_graphv2.summary.json \
  --provenance data/testing/fresh_eval_lowmid_topup_s44_graphv2.rejected.jsonl \
  --provenance data/testing/fresh_eval_low_topup_deepseek_s45.jsonl \
  --provenance data/testing/fresh_eval_low_topup_deepseek_s45.jsonl.manifest.json \
  --provenance data/testing/fresh_eval_low_topup_deepseek_s45.jsonl.rejects.jsonl \
  --provenance logs/fresh_eval_low_topup_deepseek_s45.log \
  --provenance logs/fresh_eval_low_topup_deepseek_s45.status \
  --provenance archive/provenance/fresh_eval_low_topup_deepseek_s45.authorized_early_cutoff.json \
  --provenance data/testing/fresh_eval_low_topup_deepseek_s45_assembly.jsonl \
  --provenance logs/fresh_eval_low_topup_deepseek_s45_assembly.log \
  --provenance logs/fresh_eval_low_topup_deepseek_s45_assembly.status \
  --provenance data/testing/fresh_eval_low_topup_deepseek_s45_graphv2.summary.json \
  --provenance data/testing/fresh_eval_low_topup_deepseek_s45_graphv2.rejected.jsonl \
  --provenance logs/fresh_eval_low_topup_deepseek_s45_graphv2.log \
  --provenance logs/fresh_eval_low_topup_deepseek_s45_graphv2.status \
  --provenance data/testing/fresh_eval_low_topup_chatgpt_s46.jsonl \
  --provenance data/testing/fresh_eval_low_topup_chatgpt_s46.jsonl.manifest.json \
  --provenance data/testing/fresh_eval_low_topup_chatgpt_s46.jsonl.distribution.json \
  --provenance data/testing/fresh_eval_low_topup_chatgpt_s46.jsonl.rejects.jsonl \
  --provenance logs/fresh_eval_low_topup_chatgpt_s46.log \
  --provenance logs/fresh_eval_low_topup_chatgpt_s46.status \
  --provenance archive/provenance/fresh_eval_low_topup_chatgpt_s46.authorization.json \
  --provenance archive/provenance/fresh_eval_low_topup_chatgpt_s46.source_completion.json \
  --provenance archive/provenance/fresh_eval_low_topup_chatgpt_s46.offline_completion.json \
  --provenance data/testing/fresh_eval_low_topup_chatgpt_s46_assembly.jsonl \
  --provenance logs/fresh_eval_low_topup_chatgpt_s46_assembly.log \
  --provenance logs/fresh_eval_low_topup_chatgpt_s46_assembly.status \
  --provenance data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.summary.json \
  --provenance data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.rejected.jsonl \
  --provenance logs/fresh_eval_low_topup_chatgpt_s46_graphv2.log \
  --provenance logs/fresh_eval_low_topup_chatgpt_s46_graphv2.status \
  --provenance scripts/run_fresh_eval_low_topup_chatgpt_s46.ps1 \
  --provenance scripts/run_fresh_eval_low_topup_chatgpt_s46_assembly.sh \
  --provenance scripts/run_fresh_eval_low_topup_chatgpt_s46_graphv2.sh \
  --provenance archive/provenance/fresh_graphv2_holdout_s44.stopped_insufficient_supply.json \
  --provenance archive/provenance/fresh_graphv2_holdout_toolchain_20260717.json \
  --provenance scripts/data/build_graph_v2_jsonl.py \
  --provenance generate_fresh_eval_tasks.py \
  --provenance scripts/data/seal_fresh_graphv2_holdout.py \
  --tool dart=/usr/local/bin/dart \
  --tool gdb=/usr/local/bin/gdb \
  --tool python=.venv/bin/python \
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
  printf 'DONE\nexit_code=0\nended_at=%s\ncandidate_rows=1666\nseed=44\nquotas=low:170,mid:170,high:160\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
else
  printf 'FAILED\nexit_code=%s\nended_at=%s\ncandidate_rows=1666\nseed=44\nquotas=low:170,mid:170,high:160\n' \
    "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
fi

exit "$rc"
