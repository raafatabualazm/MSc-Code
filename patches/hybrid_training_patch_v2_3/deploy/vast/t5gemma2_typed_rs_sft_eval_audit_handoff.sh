#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_RS_SFT_AUDIT_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
BASELINE_STAGE="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_contract_sft_2epoch_v1"
BASELINE_EVAL="${WORKSPACE}/artifacts/t5gemma2_typed_contract_sft_2epoch_eval_v1"
UPDATE_STAGE="${WORKSPACE}/artifacts/t5gemma2_4b4b_typed_direct_rs_sft_225_v1"
UPDATE_EVAL="${T5GEMMA_TYPED_RS_SFT_EVAL_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_direct_rs_sft_225_eval_v1}"
AUDIT_DIR="${T5GEMMA_TYPED_RS_SFT_AUDIT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_rs_sft_matched_audit_v1}"
AUDIT_OUTPUT="${AUDIT_DIR}/typed_sft_vs_typed_direct_rs_sft_update58_audit.json"
BASELINE_SEAL="${PROJECT}/configs/t5gemma2_typed_rs_sft_eval_baseline_seal.json"
EVALUATOR="${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py"
PYTHON_BIN="${T5GEMMA_TYPED_RS_SFT_AUDIT_PYTHON:-/venv/main/bin/python}"
EVAL_PROGRAM="${T5GEMMA_TYPED_RS_SFT_EVAL_PROGRAM:-t5gemma2-typed-direct-rs-sft-225-eval}"
NEXT_PROGRAM="${T5GEMMA_TYPED_RS_SFT_AUDIT_NEXT_PROGRAM:-t5gemma2-typed-local-direct-harvest}"
POLL_SECONDS="${T5GEMMA_TYPED_RS_SFT_AUDIT_POLL_SECONDS:-20}"

BASELINE_PREDICTIONS="${BASELINE_EVAL}/typed_contract_seed42_k10_predictions.json"
BASELINE_FULL_SCORE="${BASELINE_EVAL}/typed_contract_seed42_k10_score_full175.json"
BASELINE_CLEAN_SCORE="${BASELINE_EVAL}/typed_contract_seed42_k10_score_clean174.json"
UPDATE_PREDICTIONS="${UPDATE_EVAL}/typed_direct_rs_sft_seed42_k10_predictions.json"
UPDATE_FULL_SCORE="${UPDATE_EVAL}/typed_direct_rs_sft_seed42_k10_score_full175.json"
UPDATE_CLEAN_SCORE="${UPDATE_EVAL}/typed_direct_rs_sft_seed42_k10_score_clean174.json"
EVALUATION_FILE="${DATA_DIR}/dev_multifunction_binary.jsonl"

blocked() {
  echo "T5GEMMA_TYPED_RS_SFT_AUDIT_HANDOFF_BLOCKED $*" >&2
  exit 78
}

if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  blocked "invalid poll interval"
fi
if [[ ! "${EVAL_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  blocked "invalid evaluation supervisor program"
fi
if [[ ! "${NEXT_PROGRAM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  blocked "invalid next supervisor program"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  blocked "audit Python is absent: ${PYTHON_BIN}"
fi

for required in \
  "${BASELINE_PREDICTIONS}" \
  "${BASELINE_PREDICTIONS}.provenance.json" \
  "${BASELINE_PREDICTIONS}.generation.journal.jsonl" \
  "${BASELINE_PREDICTIONS}.generation.journal.jsonl.chain-head.json" \
  "${BASELINE_FULL_SCORE}" \
  "${BASELINE_FULL_SCORE}.evaluation.journal.jsonl" \
  "${BASELINE_FULL_SCORE}.evaluation.journal.jsonl.chain-head.json" \
  "${BASELINE_CLEAN_SCORE}" \
  "${BASELINE_STAGE}/checkpoint-optstep-000348/run_contract.json" \
  "${BASELINE_STAGE}/result.json" \
  "${BASELINE_SEAL}" \
  "${EVALUATION_FILE}" \
  "${EVALUATOR}"; do
  if [[ ! -s "${required}" ]]; then
    blocked "missing baseline/audit prerequisite ${required}"
  fi
done

update_required=(
  "${UPDATE_PREDICTIONS}"
  "${UPDATE_PREDICTIONS}.provenance.json"
  "${UPDATE_PREDICTIONS}.generation.journal.jsonl"
  "${UPDATE_PREDICTIONS}.generation.journal.jsonl.chain-head.json"
  "${UPDATE_FULL_SCORE}"
  "${UPDATE_FULL_SCORE}.evaluation.journal.jsonl"
  "${UPDATE_FULL_SCORE}.evaluation.journal.jsonl.chain-head.json"
  "${UPDATE_CLEAN_SCORE}"
  "${UPDATE_STAGE}/checkpoint-optstep-000058/run_contract.json"
  "${UPDATE_STAGE}/result.json"
)

while true; do
  missing=()
  for required in "${update_required[@]}"; do
    if [[ ! -s "${required}" ]]; then
      missing+=("${required}")
    fi
  done
  eval_state="$(supervisorctl status "${EVAL_PROGRAM}" 2>/dev/null || true)"
  if (( ${#missing[@]} == 0 )); then
    case " ${eval_state} " in
      *" RUNNING "*|*" STARTING "*)
        sleep "${POLL_SECONDS}"
        continue
        ;;
      *" EXITED "*|*" STOPPED "*)
        break
        ;;
      *" FATAL "*|*" BACKOFF "*|*" UNKNOWN "*)
        blocked "evaluation state=${eval_state}"
        ;;
      *)
        sleep "${POLL_SECONDS}"
        continue
        ;;
    esac
  fi
  case " ${eval_state} " in
    *" FATAL "*|*" BACKOFF "*|*" UNKNOWN "*)
      blocked "evaluation failed before sealed artifacts: ${eval_state}"
      ;;
    *" EXITED "*|*" STOPPED "*)
      blocked "evaluation ended with missing artifact ${missing[0]}"
      ;;
  esac
  sleep "${POLL_SECONDS}"
done

mkdir -p "${AUDIT_DIR}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}"
if ! "${PYTHON_BIN}" scripts/evaluation/audit_t5gemma2_typed_rs_sft.py \
  --baseline-predictions "${BASELINE_PREDICTIONS}" \
  --baseline-full-score "${BASELINE_FULL_SCORE}" \
  --baseline-clean-score "${BASELINE_CLEAN_SCORE}" \
  --baseline-checkpoint-contract "${BASELINE_STAGE}/checkpoint-optstep-000348/run_contract.json" \
  --baseline-training-result "${BASELINE_STAGE}/result.json" \
  --baseline-seal "${BASELINE_SEAL}" \
  --update-predictions "${UPDATE_PREDICTIONS}" \
  --update-full-score "${UPDATE_FULL_SCORE}" \
  --update-clean-score "${UPDATE_CLEAN_SCORE}" \
  --update-checkpoint-contract "${UPDATE_STAGE}/checkpoint-optstep-000058/run_contract.json" \
  --update-training-result "${UPDATE_STAGE}/result.json" \
  --evaluation-file "${EVALUATION_FILE}" \
  --evaluator-file "${EVALUATOR}" \
  --output "${AUDIT_OUTPUT}"; then
  blocked "matched audit failed; next program was not started"
fi

if ! /usr/bin/jq -e '
  .schema == "t5gemma2-typed-rs-sft-matched-eval-audit-v1"
  and .status == "pass"
  and .exact_pairing_validated == true
  and .contract.tasks == 175
  and .contract.k == 10
  and .contract.clean_tasks == 174
  and .checks.no_source_truncation == true
  and .checks.same_scorer_and_scoring_settings == true
  and .paired.full175.tasks == 175
  and .paired.clean174.tasks == 174' "${AUDIT_OUTPUT}" >/dev/null; then
  blocked "audit output gate differs; next program was not started"
fi

next_state="$(supervisorctl status "${NEXT_PROGRAM}" 2>/dev/null || true)"
case " ${next_state} " in
  *" RUNNING "*|*" STARTING "*)
    echo "T5GEMMA_TYPED_RS_SFT_AUDIT_HANDOFF_ALREADY_RUNNING audit=${AUDIT_OUTPUT} next=${NEXT_PROGRAM}"
    exit 0
    ;;
  *" FATAL "*|*" BACKOFF "*|*" UNKNOWN "*)
    blocked "next program cannot be started: ${next_state}"
    ;;
esac

if ! supervisorctl start "${NEXT_PROGRAM}"; then
  blocked "supervisor rejected next program ${NEXT_PROGRAM}"
fi
echo "T5GEMMA_TYPED_RS_SFT_AUDIT_HANDOFF_STARTED audit=${AUDIT_OUTPUT} next=${NEXT_PROGRAM}"
