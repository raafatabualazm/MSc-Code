#!/usr/bin/env bash
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

for required in frontier_core.py frontier_f2.py frontier_passk.py serialize_compact_inputs.py; do
  test -f "${PATCH_DIR}/${required}"
done

EVALUATOR="${WORKSPACE_DIR}/hybrid_training_patch_v2_3/scripts/evaluation/graph_compile_at_k_antigravity.py"
test -f "${EVALUATOR}"
test -x "${WORKSPACE_DIR}/dart-3.12.2/usr/bin/dart"

if test -f "${WORKSPACE_DIR}/frontier_passk.py"; then
  cp -a \
    "${WORKSPACE_DIR}/frontier_passk.py" \
    "${WORKSPACE_DIR}/frontier_passk.py.pre_audited_${STAMP}"
fi

install -m 0644 "${PATCH_DIR}/frontier_core.py" "${WORKSPACE_DIR}/frontier_core.py"
install -m 0644 "${PATCH_DIR}/frontier_f2.py" "${WORKSPACE_DIR}/frontier_f2.py"
install -m 0755 "${PATCH_DIR}/frontier_passk.py" "${WORKSPACE_DIR}/frontier_passk.py"
install -m 0755 \
  "${PATCH_DIR}/serialize_compact_inputs.py" \
  "${WORKSPACE_DIR}/serialize_compact_inputs.py"

/venv/main/bin/python -m py_compile \
  "${WORKSPACE_DIR}/frontier_core.py" \
  "${WORKSPACE_DIR}/frontier_f2.py" \
  "${WORKSPACE_DIR}/frontier_passk.py" \
  "${WORKSPACE_DIR}/serialize_compact_inputs.py" \
  "${EVALUATOR}"

sha256sum \
  "${WORKSPACE_DIR}/frontier_core.py" \
  "${WORKSPACE_DIR}/frontier_f2.py" \
  "${WORKSPACE_DIR}/frontier_passk.py" \
  "${WORKSPACE_DIR}/serialize_compact_inputs.py" \
  "${EVALUATOR}" \
  "${WORKSPACE_DIR}/dart-3.12.2/usr/bin/dart"

echo "Installed audited frontier runner. No evaluation was launched."
