#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${T5GEMMA_TYPED_SEED_REPL_WORKSPACE:-/workspace}"
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
PYTHON_BIN="${T5GEMMA_TYPED_SEED_REPL_PYTHON:-/venv/main/bin/python}"
SFT_ROOT="${T5GEMMA_TYPED_SEED_REPL_SFT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_seed_replication_sft_opt348_v1}"
INCUMBENT_ROOT="${T5GEMMA_TYPED_SEED_REPL_INCUMBENT_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_seed_replication_update58_v1}"
PASS3_ROOT="${T5GEMMA_TYPED_SEED_REPL_PASS3_OUTPUT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_seed_replication_pass3_v1}"
REPORT_ROOT="${T5GEMMA_TYPED_SEED_REPL_REPORT_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_seed_replication_report_v1}"
REPORTER="${PROJECT}/scripts/evaluation/t5gemma2_typed_seed_replication_report_v1.py"
SFT42_ROOT="${T5GEMMA_TYPED_SEED_REPL_SFT42_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_contract_sft_2epoch_eval_v1}"
INCUMBENT42_ROOT="${T5GEMMA_TYPED_SEED_REPL_INCUMBENT42_DIR:-${WORKSPACE}/artifacts/t5gemma2_typed_pass2_update58_current_stack_matched_v1}"
PASS3_42_PREDICTIONS="${T5GEMMA_TYPED_SEED_REPL_PASS3_42_PREDICTIONS:-}"
PASS3_42_SCORE="${T5GEMMA_TYPED_SEED_REPL_PASS3_42_SCORE:-}"
PASS2_ROOT="${T5GEMMA_TYPED_SEED_REPL_OPTIONAL_PASS2_ROOT:-}"

blocked() {
  echo "T5GEMMA_TYPED_SEED_REPLICATION_REPORT_BLOCKED $*" >&2
  exit 78
}

[[ -x "${PYTHON_BIN}" && -s "${PASS3_42_PREDICTIONS}" && -s "${PASS3_42_SCORE}" ]] \
  || blocked "Python and explicit pass3 seed-42 diagnostic artifacts are required"
printf '%s  %s\n' \
  caba84fcc2e97474fad00c4ad964c6375d39dae968449855c661fe765040d370 "${REPORTER}" \
  85a7e8b5bc2519233051121228e4dcafd287598e8f9644360f49393fcaf182bf "${PROJECT}/scripts/evaluation/t5gemma2_typed_seed_replication_inference_v1.py" \
  27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 "${PROJECT}/scripts/evaluation/t5gemma2_measurement_audit_inference.py" \
  30afdd256ccd2c5dd1c1482bbabf5f99f13029a68da70aeff75a57897167be4d "${PROJECT}/scripts/evaluation/t5gemma2_f2_passk_inference.py" \
  f33a11aea6337a612fa664fffe5a3eb70b11d92f7b773d2f9b8c2b134334b6e1 "${PROJECT}/scripts/evaluation/seal_t5gemma2_typed_pass3_checkpoint.py" \
  2c543c54a0ee5e55b4df708e8fd088cb772e62d012ddd41550c784c20e617cf0 "${PROJECT}/scripts/evaluation/score_direct_compact_passk.py" \
  249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 "${PROJECT}/scripts/evaluation/graph_compile_at_k_antigravity.py" \
  551403e8bd018c91acce2d3df5bfc690ea268437ec71c71a34d66a2547e35432 "${PROJECT}/scripts/evaluation/durable_evaluation_journal.py" \
  | sha256sum -c - || blocked "report/current evaluation code differs"
printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  | sha256sum -c - || blocked "full175 evaluation input differs"

args=(
  --arm "typed_sft|42|${SFT42_ROOT}/typed_contract_seed42_k10_predictions.json|${SFT42_ROOT}/typed_contract_seed42_k10_score_full175.json"
  --arm "incumbent|42|${INCUMBENT42_ROOT}/update58_current_stack_seed42_k10_predictions.json|${INCUMBENT42_ROOT}/update58_current_stack_seed42_k10_score_full175.json"
  --arm "pass3|42|${PASS3_42_PREDICTIONS}|${PASS3_42_SCORE}"
)
for seed in 43 44 45 46; do
  args+=(
    --arm "typed_sft|${seed}|${SFT_ROOT}/typed_sft_seed${seed}_k10_predictions.json|${SFT_ROOT}/typed_sft_seed${seed}_k10_score_full175.json"
    --arm "incumbent|${seed}|${INCUMBENT_ROOT}/incumbent_seed${seed}_k10_predictions.json|${INCUMBENT_ROOT}/incumbent_seed${seed}_k10_score_full175.json"
    --arm "pass3|${seed}|${PASS3_ROOT}/pass3_seed${seed}_k10_predictions.json|${PASS3_ROOT}/pass3_seed${seed}_k10_score_full175.json"
  )
done
if [[ -n "${PASS2_ROOT}" ]]; then
  [[ -d "${PASS2_ROOT}" ]] || blocked "optional pass2 root is absent"
  for seed in 42 43 44 45 46; do
    args+=(
      --arm "pass2|${seed}|${PASS2_ROOT}/pass2_seed${seed}_k10_predictions.json|${PASS2_ROOT}/pass2_seed${seed}_k10_score_full175.json"
    )
  done
fi

mkdir -p "${REPORT_ROOT}"
export PYTHONPATH="${PROJECT}"
cd "${PROJECT}"
"${PYTHON_BIN}" "${REPORTER}" "${args[@]}" \
  --evaluation-file "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --expected-wrapper-sha256 27fe6c11d487a88cd42e6330629ae470c7888c8a271c4c856b39b45208eeeb60 \
  --expected-base-inference-sha256 2b3c8803307fb8c51304e52d5eb2d81112a5f5ab4f4cf3eaca54afb7eeed02d4 \
  --expected-evaluator-sha256 249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6 \
  --expected-adapter-sha256 85a7e8b5bc2519233051121228e4dcafd287598e8f9644360f49393fcaf182bf \
  --output "${REPORT_ROOT}/typed_sft_vs_incumbent_vs_pass3_seeds42_46_report.json"

echo "T5GEMMA_TYPED_SEED_REPLICATION_REPORT_COMPLETE output=${REPORT_ROOT}/typed_sft_vs_incumbent_vs_pass3_seeds42_46_report.json promotion=not_performed"
