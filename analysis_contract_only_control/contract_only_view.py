#!/usr/bin/env python3
"""Build the gold-derived opaque typed-contract-only held-out input view.

The model-visible source keeps the exact historical typed-contract prompt and
the original encoder framing, but places exactly zero bytes between the
``<enriched_binary>`` tags.  F2 is read only to revalidate sealed row identity
and digests; no F2 text, recovered constants, or structural payload is passed
to the source constructor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation import t5gemma2_measurement_audit_inputs as audit_inputs
from scripts.training.t5gemma2_enriched_sft import SOURCE_PREAMBLE, SOURCE_SUFFIX


VIEW = "typed_contract_only"
INPUT_VIEW_SCHEMA = "t5gemma2-f2-measurement-input-view-v1"
ORACLE_CAVEAT = (
    "This is a gold-derived oracle control: it bounds what a perfect type-and-"
    "arity recovery front-end could provide, and is not a deployable binary-"
    "decompilation result because no type-recovery front-end is evaluated."
)
OOD_CAVEAT = (
    "The empty-binary condition is out of distribution for this frozen policy. "
    "A performance drop establishes dependence on the task-specific binary-"
    "channel condition; by itself it does not prove semantic decoding of F2."
)
HISTORICAL_TYPED_INTERFACE_TEMPLATE = (
    "Use this exact opaque top-level Dart interface (types and arity only; "
    "parameter names are neutral): {signature}.\n"
)
EMPTY_BINARY_PAYLOAD = ""
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
OPAQUE_INTERFACE_MARKER = "<enriched_binary>\n"
HEX_SHA256 = frozenset("0123456789abcdef")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and set(text) <= HEX_SHA256


def _extract_binary_payload(source: str) -> str:
    if source.count(OPAQUE_INTERFACE_MARKER) != 1 or not source.endswith(
        SOURCE_SUFFIX
    ):
        raise ValueError("typed source framing differs from the historical prompt")
    payload_start = source.index(OPAQUE_INTERFACE_MARKER) + len(
        OPAQUE_INTERFACE_MARKER
    )
    payload_end = len(source) - len(SOURCE_SUFFIX)
    if payload_end < payload_start:
        raise ValueError("typed source framing overlaps")
    return source[payload_start:payload_end]


def _render_historical_typed_source(*, signature: str, binary_payload: str) -> str:
    """Render one typed source while preserving the historical prompt bytes.

    The control builder below always passes the literal empty payload.  Keeping
    the renderer's payload boundary explicit also makes a separately reviewed
    future donor-permutation control possible without changing prompt grammar.
    """

    instruction = HISTORICAL_TYPED_INTERFACE_TEMPLATE.format(signature=signature)
    expected_preamble = SOURCE_PREAMBLE.replace(
        OPAQUE_INTERFACE_MARKER,
        instruction + OPAQUE_INTERFACE_MARKER,
    )
    source = audit_inputs._typed_encoder_source(binary_payload, signature)  # noqa: SLF001
    if source != expected_preamble + binary_payload + SOURCE_SUFFIX:
        raise ValueError("historical typed source construction drifted")
    return source


def build_input_view(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    f2_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Return exact typed-contract-only sources and aggregate provenance."""

    if not dataset_rows or len(dataset_rows) != len(f2_rows):
        raise ValueError("dataset/F2 row counts differ for contract-only control")
    if (
        getattr(audit_inputs, "_OPAQUE_INTERFACE_TEMPLATE", None)
        != HISTORICAL_TYPED_INTERFACE_TEMPLATE
    ):
        raise ValueError("historical typed instruction bytes changed")
    if SOURCE_PREAMBLE.count(OPAQUE_INTERFACE_MARKER) != 1:
        raise ValueError("historical encoder preamble marker changed")

    sources: list[str] = []
    row_records: list[dict[str, Any]] = []
    task_ids: list[str] = []
    arities: list[int] = []
    return_types: list[str] = []
    source_byte_lengths: list[int] = []
    for index, (dataset_row, f2_row) in enumerate(
        zip(dataset_rows, f2_rows, strict=True)
    ):
        task_id = str(dataset_row.get("task_id") or "").strip()
        if not task_id or task_id != str(f2_row.get("task_id") or "").strip():
            raise ValueError(f"row {index}: dataset/F2 identity mismatch")
        if task_id in task_ids:
            raise ValueError(f"duplicate task ID: {task_id}")
        f2_text = f2_row.get("text")
        declared_f2_sha256 = str(f2_row.get("text_sha256") or "").lower()
        if (
            not isinstance(f2_text, str)
            or not f2_text
            or not _is_sha256(declared_f2_sha256)
            or declared_f2_sha256 != _sha256_text(f2_text)
        ):
            raise ValueError(f"{task_id}: sealed F2 text/digest is invalid")

        gold_source = dataset_row.get("dart_source")
        if not isinstance(gold_source, str) or not gold_source.strip():
            raise ValueError(f"{task_id}: gold source needed for oracle signature is absent")
        signature, signature_record = audit_inputs.opaque_contract_signature(
            gold_source
        )
        # EMPTY_BINARY_PAYLOAD is a literal module constant.  F2 text is never
        # an argument to this constructor; it is used above only for seal checks.
        source = _render_historical_typed_source(
            signature=signature,
            binary_payload=EMPTY_BINARY_PAYLOAD,
        )
        payload = _extract_binary_payload(source)
        if payload != EMPTY_BINARY_PAYLOAD or payload.encode("utf-8") != b"":
            raise ValueError(f"{task_id}: non-empty binary payload reached the model")
        if gold_source.strip() in source:
            raise ValueError(f"{task_id}: gold implementation body leaked into source")
        if f2_text in source:
            raise ValueError(f"{task_id}: complete F2 text leaked into source")

        task_ids.append(task_id)
        arities.append(int(signature_record["arity"]))
        return_types.append(str(signature_record["return_type"]))
        sources.append(source)
        source_byte_lengths.append(len(source.encode("utf-8")))
        row_records.append(
            {
                "task_id": task_id,
                "sealed_f2_text_sha256": declared_f2_sha256,
                **signature_record,
                "binary_payload_utf8_bytes": 0,
                "binary_payload_sha256": EMPTY_SHA256,
                "source_sha256": _sha256_text(source),
            }
        )

    placeholder_record = {
        "text": EMPTY_BINARY_PAYLOAD,
        "utf8_hex": EMPTY_BINARY_PAYLOAD.encode("utf-8").hex(),
        "utf8_bytes": 0,
        "sha256": EMPTY_SHA256,
        "task_invariant": True,
        "placement": "exactly_between_enriched_binary_open_and_close_tags",
    }
    summary: dict[str, Any] = {
        "intervention": "withhold_entire_f2_retain_gold_derived_types_and_arity",
        "gold_derived_oracle_control": True,
        "oracle_caveat": ORACLE_CAVEAT,
        "out_of_distribution_caveat": OOD_CAVEAT,
        "deployable_type_recovery_frontend_evaluated": False,
        "claim_scope": "oracle_upper_bound_for_perfect_type_and_arity_recovery",
        "historical_typed_instruction_template": HISTORICAL_TYPED_INTERFACE_TEMPLATE,
        "historical_typed_instruction_template_utf8_hex": (
            HISTORICAL_TYPED_INTERFACE_TEMPLATE.encode("utf-8").hex()
        ),
        "historical_typed_instruction_template_sha256": _sha256_text(
            HISTORICAL_TYPED_INTERFACE_TEMPLATE
        ),
        "source_preamble": SOURCE_PREAMBLE,
        "source_preamble_utf8_hex": SOURCE_PREAMBLE.encode("utf-8").hex(),
        "source_preamble_sha256": _sha256_text(SOURCE_PREAMBLE),
        "source_suffix": SOURCE_SUFFIX,
        "source_suffix_utf8_hex": SOURCE_SUFFIX.encode("utf-8").hex(),
        "source_suffix_sha256": _sha256_text(SOURCE_SUFFIX),
        "binary_placeholder": placeholder_record,
        "f2_text_serialized_to_model": False,
        "f2_utf8_bytes_serialized_to_model": 0,
        "recovered_constants_serialized_to_model": False,
        "f2_structure_serialized_to_model": False,
        "external_call_identities_serialized_to_model": False,
        "tests_serialized_to_model": False,
        "gold_implementation_body_serialized_to_model": False,
        "gold_semantic_parameter_names_serialized_to_model": False,
        "gold_interface_types_and_arity_serialized_to_model": True,
        "function_name": "fn0",
        "parameter_name_policy": "p{zero_based_index}",
        "minimum_arity": min(arities),
        "maximum_arity": max(arities),
        "arity_histogram": {
            str(key): value for key, value in sorted(Counter(arities).items())
        },
        "return_type_histogram": dict(sorted(Counter(return_types).items())),
        "minimum_source_utf8_bytes": min(source_byte_lengths),
        "maximum_source_utf8_bytes": max(source_byte_lengths),
    }
    record = {
        "schema": INPUT_VIEW_SCHEMA,
        "view": VIEW,
        "rows": len(task_ids),
        "ordered_task_ids_sha256": canonical_sha256(task_ids),
        "ordered_source_sha256s_sha256": canonical_sha256(
            [_sha256_text(source) for source in sources]
        ),
        "row_transformations_sha256": canonical_sha256(row_records),
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "gold_interface_types_and_arity_exposed_to_model": True,
        "f2_exposed_to_model": False,
        "summary": summary,
    }
    if len(sources) != len(task_ids) or any(
        _extract_binary_payload(source) != "" for source in sources
    ):
        raise ValueError("contract-only source set is incomplete or non-empty")
    return sources, record


def _require_exact_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing file differs from exact payload: {path}")
        return
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"concurrent file differs from exact payload: {path}")


def materialize_smoke_dataset(
    *, source_dataset: str | Path, output: str | Path, rows: int = 5
) -> dict[str, Any]:
    """Materialize an exact first-N JSONL scoring subset for the smoke run."""

    if rows <= 0:
        raise ValueError("smoke row count must be positive")
    source = Path(source_dataset).expanduser().resolve()
    raw_lines = source.read_bytes().splitlines(keepends=True)
    if len(raw_lines) < rows or any(not line.strip() for line in raw_lines):
        raise ValueError("source evaluation JSONL is incomplete or contains blanks")
    selected = raw_lines[:rows]
    task_ids: list[str] = []
    for index, line in enumerate(selected):
        value = json.loads(line.decode("utf-8"))
        task_id = str(value.get("task_id") or "") if isinstance(value, dict) else ""
        if not task_id or task_id in task_ids:
            raise ValueError(f"smoke row {index}: missing/duplicate task ID")
        task_ids.append(task_id)
    payload = b"".join(selected)
    if payload and not payload.endswith((b"\n", b"\r")):
        payload += b"\n"
    target = Path(output).expanduser().resolve()
    _require_exact_bytes(target, payload)
    return {
        "schema": "t5gemma2-contract-only-smoke-dataset-v1",
        "selection": "first_n_sealed_order",
        "rows": rows,
        "source_dataset": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "output": {"path": str(target), "sha256": sha256_file(target)},
        "ordered_task_ids": task_ids,
        "ordered_task_ids_sha256": canonical_sha256(task_ids),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke_dataset_output", default="")
    parser.add_argument("--smoke_manifest_output", default="")
    parser.add_argument("--smoke_rows", type=int, default=5)
    args = parser.parse_args(argv)
    if bool(args.smoke_dataset_output) != bool(args.smoke_manifest_output):
        parser.error("smoke dataset and manifest outputs must be supplied together")
    if args.smoke_rows <= 0:
        parser.error("smoke_rows must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_rows = _read_jsonl(args.dataset, "held-out dataset")
    f2_rows = _read_jsonl(args.f2_jsonl, "held-out F2")
    _sources, record = build_input_view(
        dataset_rows=dataset_rows,
        f2_rows=f2_rows,
    )
    require_exact_or_write(Path(args.output).expanduser().resolve(), record)
    result: dict[str, Any] = {"input_view": record}
    if args.smoke_dataset_output:
        smoke = materialize_smoke_dataset(
            source_dataset=args.dataset,
            output=args.smoke_dataset_output,
            rows=args.smoke_rows,
        )
        require_exact_or_write(
            Path(args.smoke_manifest_output).expanduser().resolve(), smoke
        )
        result["smoke_dataset"] = smoke
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
