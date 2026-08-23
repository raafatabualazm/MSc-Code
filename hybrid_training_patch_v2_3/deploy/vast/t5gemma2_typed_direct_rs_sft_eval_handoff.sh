#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR=/workspace/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1
RESULT="${STAGE_DIR}/result.json"
CHECKPOINT="${STAGE_DIR}/checkpoint-optstep-000058"
TRAIN_PROGRAM=t5gemma2-typed-direct-rs-sft-225
EVAL_PROGRAM=t5gemma2-typed-direct-rs-sft-225-eval

while true; do
  if [[ -s "${RESULT}" ]]; then
    if ! /usr/bin/jq -e '
      .schema == "t5gemma2-typed-direct-rs-sft-run-v1"
      and .status == "complete"
      and .updates == 58
      and .planned_updates == 58
      and .rows == 225
      and .latest_checkpoint == "checkpoint-optstep-000058"' \
      "${RESULT}" >/dev/null; then
      echo "T5GEMMA_TYPED_DIRECT_RS_SFT_HANDOFF_BLOCKED result contract differs" >&2
      exit 78
    fi
    if [[ ! -s "${CHECKPOINT}/run_contract.json" ]]; then
      echo "T5GEMMA_TYPED_DIRECT_RS_SFT_HANDOFF_BLOCKED final checkpoint absent" >&2
      exit 78
    fi
    eval_state="$(supervisorctl status "${EVAL_PROGRAM}" 2>/dev/null || true)"
    if [[ "${eval_state}" == *" RUNNING "* || "${eval_state}" == *" STARTING "* ]]; then
      echo "T5GEMMA_TYPED_DIRECT_RS_SFT_HANDOFF_EVAL_ALREADY_RUNNING"
      exit 0
    fi
    supervisorctl start "${EVAL_PROGRAM}"
    echo "T5GEMMA_TYPED_DIRECT_RS_SFT_HANDOFF_EVAL_STARTED"
    exit 0
  fi

  train_state="$(supervisorctl status "${TRAIN_PROGRAM}" 2>/dev/null || true)"
  if [[ "${train_state}" == *" FATAL "* || "${train_state}" == *" BACKOFF "* || "${train_state}" == *" UNKNOWN "* ]]; then
    echo "T5GEMMA_TYPED_DIRECT_RS_SFT_HANDOFF_BLOCKED training=${train_state}" >&2
    exit 78
  fi
  if [[ "${train_state}" == *" EXITED "* || "${train_state}" == *" STOPPED "* ]]; then
    echo "T5GEMMA_TYPED_DIRECT_RS_SFT_HANDOFF_BLOCKED training ended without sealed result" >&2
    exit 78
  fi
  sleep 20
done
