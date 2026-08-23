#!/usr/bin/env python3
"""Strict admission wrapper around the unchanged direct-compact scorer.

Only provenance admission is extended for ``typed_contract_only``. Candidate
extraction, Dart execution, stability checks, aggregation, and hash-chained
resume all execute in ``score_direct_compact_passk`` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from analysis_contract_only_control.contract_only_view import (
    EMPTY_SHA256,
    OOD_CAVEAT,
    ORACLE_CAVEAT,
    VIEW,
)
from scripts.evaluation import score_direct_compact_passk as scorer


PROVENANCE_SCHEMA = "t5gemma2-f2-measurement-ablation-provenance-v1"


def validate_contract_only_provenance(
    prediction_provenance: Mapping[str, Any],
    *,
    prediction_sha256: str,
    k: int,
) -> None:
    heldout = prediction_provenance.get("heldout")
    view_record = heldout.get("input_view") if isinstance(heldout, Mapping) else None
    heldout_binary = heldout.get("binary_payload") if isinstance(heldout, Mapping) else None
    summary = view_record.get("summary") if isinstance(view_record, Mapping) else None
    placeholder = summary.get("binary_placeholder") if isinstance(summary, Mapping) else None
    if (
        prediction_provenance.get("schema") != PROVENANCE_SCHEMA
        or prediction_provenance.get("input_view") != VIEW
        or prediction_provenance.get("output_sha256") != prediction_sha256
        or int(prediction_provenance.get("num_samples", -1)) != int(k)
        or prediction_provenance.get("no_frontier_api") is not True
        or prediction_provenance.get("tests_exposed_to_model") is not False
        or prediction_provenance.get("full_gold_targets_exposed_to_model") is not False
        or prediction_provenance.get("gold_interface_types_and_arity_exposed_to_model")
        is not True
        or prediction_provenance.get("f2_exposed_to_model") is not False
        or prediction_provenance.get("recovered_constants_exposed_to_model")
        is not False
        or prediction_provenance.get("f2_structure_exposed_to_model") is not False
        or prediction_provenance.get("external_call_identities_exposed_to_model")
        is not False
        or prediction_provenance.get("gold_derived_oracle_control") is not True
        or prediction_provenance.get("deployable_type_recovery_frontend_evaluated")
        is not False
        or prediction_provenance.get("oracle_caveat") != ORACLE_CAVEAT
        or prediction_provenance.get("out_of_distribution_caveat") != OOD_CAVEAT
        or not isinstance(heldout, Mapping)
        or heldout.get("f2_serialized_to_model") is not False
        or heldout.get("recovered_constants_serialized_to_model") is not False
        or heldout.get("f2_structure_serialized_to_model") is not False
        or heldout.get("external_call_identities_serialized_to_model") is not False
        or heldout.get("model_visible_fields")
        != [
            "gold_derived_opaque_types_and_arity",
            "task_invariant_empty_binary_payload",
        ]
        or heldout.get("gold_derived_oracle_control") is not True
        or heldout.get("deployable_type_recovery_frontend_evaluated") is not False
        or heldout.get("oracle_caveat") != ORACLE_CAVEAT
        or heldout.get("out_of_distribution_caveat") != OOD_CAVEAT
        or not isinstance(heldout_binary, Mapping)
        or heldout_binary.get("text") != ""
        or heldout_binary.get("utf8_hex") != ""
        or heldout_binary.get("utf8_bytes") != 0
        or heldout_binary.get("sha256") != EMPTY_SHA256
        or heldout_binary.get("task_invariant") is not True
        or not isinstance(view_record, Mapping)
        or view_record.get("schema")
        != "t5gemma2-f2-measurement-input-view-v1"
        or view_record.get("view") != VIEW
        or view_record.get("f2_exposed_to_model") is not False
        or not isinstance(summary, Mapping)
        or summary.get("f2_text_serialized_to_model") is not False
        or summary.get("f2_utf8_bytes_serialized_to_model") != 0
        or summary.get("recovered_constants_serialized_to_model") is not False
        or summary.get("f2_structure_serialized_to_model") is not False
        or summary.get("external_call_identities_serialized_to_model") is not False
        or summary.get("gold_derived_oracle_control") is not True
        or summary.get("oracle_caveat") != ORACLE_CAVEAT
        or summary.get("out_of_distribution_caveat") != OOD_CAVEAT
        or not isinstance(placeholder, Mapping)
        or placeholder.get("text") != ""
        or placeholder.get("utf8_hex") != ""
        or placeholder.get("utf8_bytes") != 0
        or placeholder.get("sha256") != EMPTY_SHA256
        or placeholder.get("task_invariant") is not True
    ):
        raise ValueError("contract-only prediction provenance contract failed")


def validate_existing_contract_only_score(
    *,
    predictions: str | Path,
    evaluation_file: str | Path,
    output: str | Path,
    k: int,
    timeout: int,
    stability_runs: int,
    journal: str | Path | None = None,
) -> dict[str, Any]:
    """Use the stock scorer's complete resume/hash validation with our admission."""

    original = scorer.validate_prediction_provenance
    scorer.validate_prediction_provenance = validate_contract_only_provenance
    try:
        return scorer.validate_existing_score(
            predictions=predictions,
            evaluation_file=evaluation_file,
            output=output,
            k=k,
            timeout=timeout,
            stability_runs=stability_runs,
            journal=journal,
        )
    finally:
        scorer.validate_prediction_provenance = original


def main() -> None:
    original = scorer.validate_prediction_provenance
    scorer.validate_prediction_provenance = validate_contract_only_provenance
    try:
        scorer.main()
    finally:
        scorer.validate_prediction_provenance = original


if __name__ == "__main__":
    main()
