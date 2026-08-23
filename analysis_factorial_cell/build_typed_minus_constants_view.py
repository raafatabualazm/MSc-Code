# -*- coding: utf-8 -*-
"""Build the missing 2x2 cell: typed opaque contract WITH constants stripped.

The three one-way ablations already run give:

                     | constants present | constants stripped
    ---------------- + ----------------- + ------------------
    no types         |  5.8 / 127.0      |  2.6 / 125.2
    typed contract   | 11.0 / 169.4      |  ??? / ???        <- this file

Filling that cell converts three one-way ablations into a factorial with a
measured interaction term, and answers whether declared types and recovered
constants are independent channels or redundant ones.

Composition is exact: `constants_stripped` rewrites the F2 constants prefix and
leaves the structural payload byte-identical; `typed_opaque_contract` injects a
typed `fn0` signature into the encoder preamble and appends whatever F2 text it
is given. The two transforms touch disjoint regions, so applying the typed
wrapper to the already-stripped F2 composes them without interference.

Nothing under hybrid_training_patch_v2_3/ is modified; this imports from it.

Usage (on the pod):
    PYTHONPATH=/workspace/hybrid_training_patch_v2_3 \
    python3 build_typed_minus_constants_view.py \
        --dataset /workspace/multifunction_v1/build/dev_multifunction_binary.jsonl \
        --f2      /workspace/multifunction_v1/build/dev_multifunction_binary_f2.jsonl \
        --out     /workspace/artifacts/t5gemma2_f2_factorial_cell_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.t5gemma2_measurement_audit_inputs import (
    SOURCE_PREAMBLE,
    SOURCE_SUFFIX,
    _literal_stripped_prefix,
    _sha256_text,
    _typed_encoder_source,
    canonical_sha256,
    opaque_contract_signature,
    parse_f2,
    render_f2,
)

VIEW = "typed_contract_minus_constants"
SCHEMA = "t5gemma2-f2-measurement-input-view-v1"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build(dataset_rows, f2_rows):
    if len(dataset_rows) != len(f2_rows):
        raise ValueError("dataset/F2 row counts differ")

    sources, row_records = [], []
    task_ids, arities, return_types = [], [], []
    changed_rows = removed_lines = removed_bytes = 0

    for index, (drow, frow) in enumerate(zip(dataset_rows, f2_rows)):
        task_id = str(drow.get("task_id") or "").strip()
        if not task_id or task_id != str(frow.get("task_id") or "").strip():
            raise ValueError(f"row {index}: dataset/F2 identity mismatch")
        text = frow.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{task_id}: F2 text is absent")
        declared = str(frow.get("text_sha256") or "").lower()
        if declared != _sha256_text(text):
            raise ValueError(f"{task_id}: F2 text digest mismatch")

        parsed = parse_f2(text)

        # --- transform 1: strip recovered literals from the constants prefix
        stripped_prefix, counts = _literal_stripped_prefix(parsed.prefix)
        stripped_f2 = render_f2(prefix=stripped_prefix, structure=parsed.structure)
        check = parse_f2(stripped_f2)
        if check.structure_sha256 != parsed.structure_sha256:
            raise ValueError(f"{task_id}: constants transform altered F2 structure")
        if "// externals" in parsed.prefix and "// externals" not in check.prefix:
            raise ValueError(f"{task_id}: constants transform removed external identities")

        # --- transform 2: typed opaque contract over the STRIPPED F2
        gold_source = drow.get("dart_source")
        signature, signature_record = opaque_contract_signature(gold_source)
        source = _typed_encoder_source(stripped_f2, signature)
        if str(gold_source).strip() in source:
            raise ValueError(f"{task_id}: gold implementation body leaked into source")

        changed_rows += int(stripped_f2 != text)
        removed_lines += counts["removed_literal_lines"]
        removed_bytes += counts["removed_literal_bytes"]
        task_ids.append(task_id)
        arities.append(int(signature_record["arity"]))
        return_types.append(str(signature_record["return_type"]))
        sources.append(source)
        row_records.append({
            "task_id": task_id,
            "original_prefix_sha256": parsed.prefix_sha256,
            "stripped_prefix_sha256": check.prefix_sha256,
            "structure_sha256": parsed.structure_sha256,
            **counts,
            **signature_record,
            "source_sha256": _sha256_text(source),
        })

    from collections import Counter
    record = {
        "schema": SCHEMA,
        "view": VIEW,
        "rows": len(task_ids),
        "ordered_task_ids_sha256": canonical_sha256(task_ids),
        "ordered_source_sha256s_sha256": canonical_sha256(
            [_sha256_text(s) for s in sources]),
        "row_transformations_sha256": canonical_sha256(row_records),
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "summary": {
            "intervention": "typed_contract_over_literal_stripped_prefix",
            "composed_from": ["constants_stripped", "typed_opaque_contract"],
            "composition_order": "strip literals from F2 prefix, then wrap with typed fn0 contract",
            "f2_structure_byte_identical": True,
            "external_call_identities_preserved": True,
            "gold_implementation_body_exposed_to_model": False,
            "gold_semantic_parameter_names_exposed_to_model": False,
            "function_name": "fn0",
            "parameter_name_policy": "p{zero_based_index}",
            "changed_rows": changed_rows,
            "unchanged_no_literal_rows": len(task_ids) - changed_rows,
            "removed_literal_lines": removed_lines,
            "removed_literal_bytes": removed_bytes,
            "minimum_arity": min(arities),
            "maximum_arity": max(arities),
            "arity_histogram": {str(k): v for k, v in sorted(Counter(arities).items())},
            "return_type_histogram": dict(sorted(Counter(return_types).items())),
        },
    }
    return sources, record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--f2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sources, record = build(read_jsonl(args.dataset), read_jsonl(args.f2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_sources.json").write_text(
        json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    (out / "input_view.json").write_text(
        json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")

    s = record["summary"]
    print("view                : %s" % record["view"])
    print("rows                : %d" % record["rows"])
    print("literal-changed rows: %d   unchanged (no literals): %d"
          % (s["changed_rows"], s["unchanged_no_literal_rows"]))
    print("removed             : %d literal lines, %d bytes"
          % (s["removed_literal_lines"], s["removed_literal_bytes"]))
    print("arity histogram     : %s" % s["arity_histogram"])
    print("sources sha256      : %s" % record["ordered_source_sha256s_sha256"][:16])
    print()
    print("CONTROL: expect literal-changed rows == 96 and unchanged == 79,")
    print("         matching the constants_stripped arm exactly. If they differ,")
    print("         the composition is not applying the same literal transform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
