#!/usr/bin/env bash
set -euo pipefail

# Rank-0 evaluator integrity check: feed each held-out gold Dart function
# directly to the unchanged sealed evaluator.  This is never model-visible and
# never used for training; it verifies only the source/test/evaluator join.
WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
DATASET="${WORKSPACE}/multifunction_v1/build/dev_multifunction_binary.jsonl"
OUTPUT_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_measurement_audit_v1/gold_roundtrip"
PREDICTIONS="${OUTPUT_DIR}/gold_k1_predictions.json"
PROVENANCE="${PREDICTIONS}.provenance.json"
SCORE="${OUTPUT_DIR}/gold_k1_score.json"
DART_BIN="${WORKSPACE}/tools/dart-3.12.2/usr/lib/dart/bin/dart"

if [[ ! -x "${DART_BIN}" ]]; then
  echo "GOLD_ROUNDTRIP_BLOCKED pinned Dart binary is absent" >&2
  exit 78
fi
printf '%s  %s\n' \
  abc8499f6984d8503fa71855021893bb1aba0c655fb744e55e6c41708b8edce7 \
  "${DATASET}" | sha256sum -c -
mkdir -p "${OUTPUT_DIR}"

/venv/main/bin/python - "${DATASET}" "${PREDICTIONS}" "${PROVENANCE}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1]).resolve()
predictions_path = Path(sys.argv[2]).resolve()
provenance_path = Path(sys.argv[3]).resolve()

rows = []
seen = set()
with dataset_path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        row = json.loads(line)
        task_id = str(row.get("task_id") or "")
        source = str(row.get("dart_source") or "")
        tests = str(
            row.get("acceptance_tests")
            or row.get("tests")
            or row.get("feedback_tests")
            or ""
        )
        if not task_id or task_id in seen or not source or not tests:
            raise ValueError(f"invalid held-out row {line_number}: {task_id!r}")
        if "fn0" not in source:
            raise ValueError(f"gold source lacks fn0: {task_id}")
        seen.add(task_id)
        rows.append({"id": task_id, "predictions": [source]})
if len(rows) != 175:
    raise ValueError(f"expected 175 gold rows, got {len(rows)}")

payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
payload_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
provenance = {
    "schema": "direct-compact-inference-v1",
    "arm": "gold_evaluator_roundtrip",
    "purpose": "evaluator_integrity_only",
    "num_rows": len(rows),
    "num_samples": 1,
    "output_sha256": payload_sha,
    "evaluation_dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    "gold_targets_serialized_to_model": False,
    "model_generation_used": False,
    "training_allowed": False,
}

if predictions_path.exists():
    if predictions_path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing gold predictions differ")
else:
    predictions_path.write_text(payload, encoding="utf-8")
provenance_text = json.dumps(provenance, sort_keys=True, indent=2) + "\n"
if provenance_path.exists():
    if provenance_path.read_text(encoding="utf-8") != provenance_text:
        raise ValueError("existing gold provenance differs")
else:
    provenance_path.write_text(provenance_text, encoding="utf-8")
print(f"GOLD_ROUNDTRIP_BUILT rows={len(rows)} sha256={payload_sha}")
PY

export PYTHONPATH="${PROJECT}"
export PATH="$(dirname "${DART_BIN}"):${PATH}"
export CUDA_VISIBLE_DEVICES=""
cd "${PROJECT}"
/venv/main/bin/python scripts/evaluation/score_direct_compact_passk.py \
  --predictions "${PREDICTIONS}" \
  --evaluation_file "${DATASET}" \
  --k 1 \
  --workers 32 \
  --timeout 30 \
  --stability_runs 2 \
  --output "${SCORE}"

/venv/main/bin/python - "${SCORE}" <<'PY'
import json
import sys
from pathlib import Path

score = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tasks = int(score.get("tasks", -1))
passed = int((score.get("pass_at_k") or {}).get("count", -1))
compiled = int((score.get("compile_at_k") or {}).get("count", -1))
print(f"GOLD_ROUNDTRIP_RESULT pass={passed}/{tasks} compile={compiled}/{tasks}")
if tasks != 175 or passed != 175 or compiled != 175:
    raise SystemExit(2)
PY

