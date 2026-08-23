#!/bin/bash
# Durable post-Qwen chain: CoT SFT -> synchronous GPT RS -> RS-SFT -> VeRPO.
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1091
. "${utils}/logging.sh" ""
# shellcheck disable=SC1091
. "${utils}/environment.sh"

source /venv/main/bin/activate
mkdir -p /workspace/logs
cd /workspace

sequence_checkpoint=/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2/direct_compact_qwen_sequence_warmstart
sequence_build=/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2/qwen_mc_sequence_train.build.json
qwen_restart_count=0
qwen_restart_limit="${QWEN_RESTART_LIMIT:-3}"
if [[ ! "${qwen_restart_limit}" =~ ^[0-9]+$ ]] \
  || (( qwen_restart_limit < 1 )); then
  printf 'QWEN_RESTART_LIMIT must be a positive integer\n' >&2
  exit 2
fi
while [[ ! -f "${sequence_checkpoint}/run_provenance.json" \
      || ! -f "${sequence_build}" ]]; do
  qwen_status="$(supervisorctl status qwen38_kd 2>/dev/null || true)"
  if [[ "${qwen_status}" == *"FATAL"* ]]; then
    printf 'Qwen sequence stage entered FATAL before publishing: %s\n' \
      "${qwen_status}" >&2
    exit 2
  fi
  if [[ "${qwen_status}" == *"EXITED"* \
     || "${qwen_status}" == *"STOPPED"* ]]; then
    if (( qwen_restart_count >= qwen_restart_limit )); then
      printf 'Qwen sequence stage exhausted %s journal-resume attempts: %s\n' \
        "${qwen_restart_limit}" "${qwen_status}" >&2
      exit 2
    fi
    qwen_restart_count=$((qwen_restart_count + 1))
    printf 'POST_QWEN_RESUMING_QWEN_FROM_JOURNAL attempt=%s/%s status=%s\n' \
      "${qwen_restart_count}" "${qwen_restart_limit}" "${qwen_status}"
    supervisorctl start qwen38_kd
    sleep 5
    continue
  fi
  printf 'POST_QWEN_WAITING_FOR_SEQUENCE checkpoint=%s status=%s\n' \
    "${sequence_checkpoint}" "${qwen_status:-unknown}"
  sleep 60
done

pty bash -lc \
  'bash /workspace/run_qwen_cot_sft.sh && bash /workspace/run_collect_chatgpt_compact_rs.sh && bash /workspace/run_rs_sft_then_verpo.sh' \
  2>&1 | tee -a /workspace/logs/post_qwen_rs_verpo.log
