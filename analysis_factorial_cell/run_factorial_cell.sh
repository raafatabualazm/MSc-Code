#!/usr/bin/env bash
# Missing 2x2 cell: typed opaque contract WITH constants stripped.
#
# Mirrors deploy/vast/t5gemma2_measurement_intervention_multiseed_v1.sh exactly
# — same inference wrapper, same scorer, same seals, same scoring settings — so
# the resulting numbers are directly comparable to the three one-way arms.
#
# NOTHING under hybrid_training_patch_v2_3/ is modified. The new view is
# registered at runtime by a shim that wraps build_input_view; the sealed
# module itself is imported unchanged.
#
# Seeds 42,43,44 (~75 min each). View-major so a truncated run still yields a
# usable arm. Run AFTER the body-swap seed 46 completes — the GPU lock is
# honoured either way.
set -euo pipefail

WORKSPACE=/workspace
PROJECT="${WORKSPACE}/hybrid_training_patch_v2_3"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
DATA_DIR="${WORKSPACE}/multifunction_v1/build"
AUDIT_DIR="${WORKSPACE}/artifacts/t5gemma2_f2_measurement_audit_v1"
OUT="${WORKSPACE}/artifacts/t5gemma2_f2_factorial_cell_v1"
SFT_CKPT="${T5GEMMA_SFT_CHECKPOINT:?set T5GEMMA_SFT_CHECKPOINT to the same checkpoint the multiseed arms used}"

SCORER="${PROJECT}/scripts/evaluation/score_direct_compact_passk.py"
SHIM="${OUT}/inference_shim.py"

blocked() { echo "FACTORIAL_CELL_BLOCKED $*" >&2; exit 78; }

mkdir -p "${OUT}"

# ---------------------------------------------------------------- build view
PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" \
  "${WORKSPACE}/build_typed_minus_constants_view.py" \
  --dataset "${DATA_DIR}/dev_multifunction_binary.jsonl" \
  --f2      "${DATA_DIR}/dev_multifunction_binary_f2.jsonl" \
  --out     "${OUT}" || blocked "view build failed"

# Control: the composition must reproduce the single-transform counts exactly.
/usr/bin/jq -e '
  .summary.changed_rows == 96
  and .summary.unchanged_no_literal_rows == 79
  and .summary.removed_literal_lines == 109
  and .summary.removed_literal_bytes == 10915
  and .summary.arity_histogram["1"] == 82
  and .rows == 175
  and .tests_exposed_to_model == false
  and .full_gold_targets_exposed_to_model == false
' "${OUT}/input_view.json" >/dev/null \
  || blocked "composed view does not reproduce the single-transform controls"
echo "FACTORIAL_CELL_VIEW_CONTROLS_PASS"

# ---------------------------------------------------------------- shim
cat > "${SHIM}" <<'PY'
"""Register the composed view at runtime, then delegate to the sealed wrapper.

The audit inference validates --input_view against SUPPORTED_INPUT_VIEWS. We
extend that set and route the new name to the composed builder, leaving every
other view and the whole downstream path byte-identical.
"""
import runpy
import sys

from scripts.evaluation import t5gemma2_measurement_audit_inputs as inputs

sys.path.insert(0, "/workspace")
from build_typed_minus_constants_view import VIEW, build  # noqa: E402

_original = inputs.build_input_view
inputs.SUPPORTED_INPUT_VIEWS = frozenset(set(inputs.SUPPORTED_INPUT_VIEWS) | {VIEW})


def _dispatch(*, dataset_rows, f2_rows, view):
    if view == VIEW:
        return build(list(dataset_rows), list(f2_rows))
    return _original(dataset_rows=dataset_rows, f2_rows=f2_rows, view=view)


inputs.build_input_view = _dispatch
runpy.run_path(
    "/workspace/hybrid_training_patch_v2_3/scripts/evaluation/"
    "t5gemma2_measurement_audit_inference.py",
    run_name="__main__",
)
PY

# ---------------------------------------------------------------- run
common_data=(
  --dataset      "${DATA_DIR}/dev_multifunction_binary.jsonl"
  --dataset_seal "${DATA_DIR}/dev_multifunction_binary.seal.json"
  --f2_jsonl     "${DATA_DIR}/dev_multifunction_binary_f2.jsonl"
  --f2_manifest  "${DATA_DIR}/dev_multifunction_binary_f2.jsonl.manifest.json"
  --sft_checkpoint "${SFT_CKPT}"
  --arm sft --num_samples 10 --generation_batch_size 10
  --max_source_tokens 32768 --max_new_tokens 4096
  --temperature 0.8 --top_p 0.95 --bf16
)
common_score=(
  --evaluation_file "${DATA_DIR}/dev_multifunction_binary.jsonl"
  --k 10 --workers 32 --timeout 30 --stability_runs 2
)

for seed in 42 43 44; do
  pred="${OUT}/typed_contract_minus_constants_seed${seed}_k10_predictions.json"
  score="${OUT}/typed_contract_minus_constants_seed${seed}_k10_score.json"
  [[ -s "${score}" ]] && { echo "FACTORIAL_CELL_SKIP seed=${seed} (already scored)"; continue; }

  PYTHONPATH="${PROJECT}" "${PYTHON_BIN}" "${SHIM}" \
    "${common_data[@]}" --input_view typed_contract_minus_constants \
    --seed "${seed}" --output "${pred}" || blocked "inference failed seed ${seed}"

  "${PYTHON_BIN}" "${SCORER}" \
    --predictions "${pred}" --output "${score}" "${common_score[@]}" \
    || blocked "scoring failed seed ${seed}"

  [[ -s "${pred}.provenance.json" \
     && -s "${pred}.generation.journal.jsonl.chain-head.json" \
     && -s "${score}.evaluation.journal.jsonl.chain-head.json" ]] \
    || blocked "seed ${seed} did not publish complete hash-chain artifacts"
  echo "FACTORIAL_CELL_RUN_COMPLETE seed=${seed} score=${score}"
done

# ---------------------------------------------------------------- aggregate
"${PYTHON_BIN}" - "${OUT}" <<'PY'
import json, glob, os, statistics as st, sys
rows = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_k10_score.json"))):
    seed = int(f.split("seed")[1].split("_")[0])
    d = json.load(open(f))
    per = {}
    for c in d.get("candidate_results", []):
        per.setdefault(c["task_id"], []).append(c["code_sha256"])
    dis = [len(set(v)) for v in per.values()]
    rows[seed] = (d["pass_at_1"]["count"], d["pass_at_k"]["count"],
                  d["compile_at_k"]["count"], sum(dis) / len(dis))
print()
print("  seed  pass@1  pass@10  compile@10  distinct")
for s in sorted(rows):
    v = rows[s]
    print("  %-5d %-7d %-8d %-11d %.2f" % (s, v[0], v[1], v[2], v[3]))
if len(rows) > 1:
    pk = [rows[s][1] for s in sorted(rows)]
    ck = [rows[s][2] for s in sorted(rows)]
    print("  pass@10 mean %.2f SD %.2f | compile@10 mean %.2f SD %.2f | n=%d"
          % (st.mean(pk), st.stdev(pk), st.mean(ck), st.stdev(ck), len(pk)))
print()
print("  PRE-REGISTERED PREDICTION (fixed 2026-08-03, before the run):")
print("    compile@10 ~ 167 +/- 5   (types drive compilation; constants do not)")
print("    pass@10    ~ 8           (ADDITIVE: 2.6 + 5.2)")
print("    near 11  -> types substitute for constants; rewrite the dissociation")
print("    near 2.6 -> types are conditional on constants; qualify the abstract")
PY
echo "FACTORIAL_CELL_DONE"
