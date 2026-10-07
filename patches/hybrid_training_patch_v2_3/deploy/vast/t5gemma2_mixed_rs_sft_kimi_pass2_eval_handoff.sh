#!/usr/bin/env bash
set -euo pipefail

GENERATION_SERVICE=t5gemma-mixed-rs-sft-kimi-pass2-generation
EVAL_LAUNCHER=/opt/supervisor-scripts/t5gemma2_mixed_rs_sft_kimi_pass2_eval.sh

while true; do
  status_line="$(supervisorctl status "${GENERATION_SERVICE}" 2>/dev/null || true)"
  state="$(awk '{print $2}' <<<"${status_line}")"
  case "${state}" in
    RUNNING|STARTING|STOPPING)
      sleep 30
      ;;
    EXITED|STOPPED)
      break
      ;;
    *)
      echo "T5GEMMA_KIMI_PASS2_EVAL_HANDOFF_BLOCKED generation state=${state:-missing}" >&2
      exit 78
      ;;
  esac
done

# The evaluator revalidates the complete generation journal and exact output,
# then scores and writes the paired regression report.  If generation exited
# early, its durable journal is resumed rather than duplicated.
exec "${EVAL_LAUNCHER}"
