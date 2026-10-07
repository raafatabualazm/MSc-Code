#!/usr/bin/env bash
set -Eeuo pipefail

PATCH_ROOT="${TRUE_KD_PATCH_ROOT:-/workspace/true_kd_patch_v1}"
COMMAND="${1:-}"
if [[ -z "${COMMAND}" ]]; then
  echo "usage: $0 audit-legacy | validate-sparse | checkpoint-preflight | dense" >&2
  exit 2
fi
shift

case "${COMMAND}" in
  audit-legacy)
    REBUILD="${REBUILD:-/workspace/artifacts/compact_fn0_rebuild}"
    TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
    exec /venv/main/bin/python \
      "${PATCH_ROOT}/scripts/data/validate_qwen_kd_artifacts.py" \
      legacy-qwen-audit \
      --input "${LEGACY_QWEN_LOGPROBS:-${REBUILD}/verified_repairs_lp.jsonl}" \
      --expected_training_rows "${EXPECTED_TRAINING_ROWS:-1580}" \
      --student_tokenizer_json "${TOKENIZER_JSON}" \
      --report "${AUDIT_REPORT:-${REBUILD}/verified_repairs_lp.audit.json}" \
      "$@"
    ;;
  validate-sparse)
    : "${SPARSE_DATA:?Set SPARSE_DATA to the sparse top-k+tail JSONL}"
    : "${SPARSE_MANIFEST:?Set SPARSE_MANIFEST to its sealed manifest}"
    : "${TOKENIZER_JSON:?Set TOKENIZER_JSON to the exact student tokenizer JSON}"
    : "${CONTRACT:?Set CONTRACT to the sealed compact contract}"
    exec /venv/main/bin/python \
      "${PATCH_ROOT}/scripts/data/validate_qwen_kd_artifacts.py" \
      sparse-validate \
      --input "${SPARSE_DATA}" \
      --manifest "${SPARSE_MANIFEST}" \
      --student_tokenizer_json "${TOKENIZER_JSON}" \
      --contract "${CONTRACT}" \
      "$@"
    ;;
  checkpoint-preflight)
    : "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the local teacher checkpoint}"
    : "${STUDENT_WARMSTART_CHECKPOINT:?Set STUDENT_WARMSTART_CHECKPOINT to the student checkpoint}"
    REBUILD="${REBUILD:-/workspace/artifacts/compact_fn0_rebuild}"
    TOKENIZER_JSON="${TOKENIZER_JSON:-/workspace/.hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json}"
    exec /venv/main/bin/python \
      "${PATCH_ROOT}/scripts/data/check_kd_checkpoint_bindings.py" \
      --contract "${CONTRACT:-${REBUILD}/fn0_contract.json}" \
      --tokenizer_json "${TOKENIZER_JSON}" \
      --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
      --student_checkpoint "${STUDENT_WARMSTART_CHECKPOINT}" \
      "$@"
    ;;
  dense)
    # The focused CPU tests run before either large model is allocated.
    /venv/main/bin/python -m unittest discover \
      -s "${PATCH_ROOT}/tests" -v
    exec bash "${PATCH_ROOT}/run_dense_full_kd.sh" "$@"
    ;;
  *)
    echo "unknown command ${COMMAND@Q}; expected audit-legacy, validate-sparse, checkpoint-preflight, or dense" >&2
    exit 2
    ;;
esac
