#!/usr/bin/env python3
"""Deterministic, fail-closed input interventions for the F2 measurement audit.

The functions in this module are deliberately CPU-only.  They operate on the
already sealed held-out F2 rows and return model-visible strings plus a compact
provenance record.  No test text or gold implementation body is ever copied to
an encoder source.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import canonical_sha256
from scripts.training import hybrid_data_controls as controls
from scripts.training.t5gemma2_enriched_sft import (
    SOURCE_PREAMBLE,
    SOURCE_SUFFIX,
)
from scripts.training.t5gemma2_typed_contract_sft import (
    opaque_contract_signature as training_opaque_contract_signature,
)


INPUT_VIEW_SCHEMA = "t5gemma2-f2-measurement-input-view-v1"
SUPPORTED_INPUT_VIEWS = frozenset(
    {
        "semantic_body_swap",
        "constants_stripped",
        "typed_opaque_contract",
    }
)
_F2_HEADER = b"F2\nC"
_OPAQUE_INTERFACE_MARKER = "<enriched_binary>\n"
# This wording is a sealed part of the original seed-42 measurement arm.  The
# trainer later adopted a clearer six-token-longer instruction, but changing it
# here would make seeds 43--46 a different intervention.  Keep the historical
# model-visible bytes for the paired multi-seed replication.
_OPAQUE_INTERFACE_TEMPLATE = (
    "Use this exact opaque top-level Dart interface (types and arity only; "
    "parameter names are neutral): {signature}.\n"
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TYPE_TEXT = re.compile(r"[A-Za-z_][A-Za-z0-9_<>,? ]*\Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParsedF2:
    """Exact byte partition of one F2 string."""

    prefix: str
    structure: str
    prefix_sha256: str
    structure_sha256: str
    prefix_bytes: int
    structure_bytes: int


def parse_f2(text: str) -> ParsedF2:
    """Split an F2 stream without decoding or normalising either payload."""

    if not isinstance(text, str) or not text:
        raise ValueError("F2 text is empty")
    payload = text.encode("utf-8")
    if not payload.startswith(_F2_HEADER):
        raise ValueError("F2 text has an invalid header")
    length_end = payload.find(b"\n", len(_F2_HEADER))
    if length_end < 0:
        raise ValueError("F2 constants length is unterminated")
    length_text = payload[len(_F2_HEADER) : length_end]
    if not length_text or not length_text.isdigit():
        raise ValueError("F2 constants length is invalid")
    prefix_bytes = int(length_text.decode("ascii"))
    prefix_start = length_end + 1
    prefix_end = prefix_start + prefix_bytes
    if prefix_end >= len(payload):
        raise ValueError("F2 constants payload is truncated")
    if payload[prefix_end : prefix_end + 1] != b"\n":
        raise ValueError("F2 constants payload has no structural separator")
    prefix_payload = payload[prefix_start:prefix_end]
    structure_payload = payload[prefix_end + 1 :]
    try:
        prefix = prefix_payload.decode("utf-8")
        structure = structure_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("F2 payload is not valid UTF-8") from exc
    if not structure.startswith("A") or not structure.endswith("\nX\n"):
        raise ValueError("F2 structural payload lacks its sealed boundaries")
    return ParsedF2(
        prefix=prefix,
        structure=structure,
        prefix_sha256=_sha256_text(prefix),
        structure_sha256=_sha256_text(structure),
        prefix_bytes=len(prefix_payload),
        structure_bytes=len(structure_payload),
    )


def render_f2(*, prefix: str, structure: str) -> str:
    """Recompose an exact F2 stream with a corrected byte-length header."""

    prefix_payload = prefix.encode("utf-8")
    structure_payload = structure.encode("utf-8")
    if not structure_payload.startswith(b"A") or not structure_payload.endswith(
        b"\nX\n"
    ):
        raise ValueError("refusing to render an invalid F2 structure")
    output = (
        _F2_HEADER
        + str(len(prefix_payload)).encode("ascii")
        + b"\n"
        + prefix_payload
        + b"\n"
        + structure_payload
    )
    rendered = output.decode("utf-8")
    parsed = parse_f2(rendered)
    if parsed.prefix != prefix or parsed.structure != structure:
        raise ValueError("F2 recomposition did not round-trip exactly")
    return rendered


def _literal_stripped_prefix(prefix: str) -> tuple[str, dict[str, int]]:
    """Remove recovered string/number literals, retaining external identities."""

    kept: list[str] = []
    removed_lines = 0
    removed_bytes = 0
    for line in prefix.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        removable = (
            content == "// constant pool recovered from binary"
            or content.startswith("// strings")
            or content.startswith("// numbers")
        )
        if removable:
            removed_lines += 1
            removed_bytes += len(line.encode("utf-8"))
            continue
        if content and not content.startswith("// externals"):
            raise ValueError(
                "unknown F2 prefix line prevents a literal-only intervention"
            )
        kept.append(line)
    stripped = "".join(kept)
    if any(
        token in stripped
        for token in ("// strings", "// numbers", "constant pool recovered")
    ):
        raise ValueError("recovered literal material survived prefix stripping")
    return stripped, {
        "removed_literal_lines": removed_lines,
        "removed_literal_bytes": removed_bytes,
        "retained_prefix_bytes": len(stripped.encode("utf-8")),
    }


def opaque_contract_signature(gold_source: str) -> tuple[str, dict[str, Any]]:
    """Recover only fn0's types/arity and rename parameters to p0, p1, ... ."""

    # Training and held-out inference must never drift onto different prompt
    # grammars.  Keep this public evaluation helper as a compatibility alias
    # while making the trainer implementation the single source of truth.
    return training_opaque_contract_signature(gold_source)


def _typed_encoder_source(f2_text: str, signature: str) -> str:
    if SOURCE_PREAMBLE.count(_OPAQUE_INTERFACE_MARKER) != 1:
        raise ValueError("SFT encoder preamble marker changed")
    instruction = _OPAQUE_INTERFACE_TEMPLATE.format(signature=signature)
    preamble = SOURCE_PREAMBLE.replace(
        _OPAQUE_INTERFACE_MARKER,
        instruction + _OPAQUE_INTERFACE_MARKER,
    )
    return preamble + f2_text + SOURCE_SUFFIX


def _matched_structure_derangement(
    task_ids: Sequence[str], parsed: Sequence[ParsedF2]
) -> dict[int, int]:
    """Stable nearest-length permutation: adjacent pairs plus one 3-cycle."""

    if len(task_ids) != len(parsed) or len(task_ids) < 2:
        raise ValueError("semantic-body intervention requires at least two rows")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("semantic-body intervention received duplicate task IDs")
    ordered = sorted(
        range(len(task_ids)),
        key=lambda index: (parsed[index].structure_bytes, task_ids[index]),
    )
    mapping: dict[int, int] = {}
    pair_limit = len(ordered)
    if len(ordered) % 2:
        if len(ordered) < 3:
            raise ValueError("odd semantic-body intervention needs three rows")
        pair_limit -= 3
    for position in range(0, pair_limit, 2):
        left, right = ordered[position : position + 2]
        mapping[left] = right
        mapping[right] = left
    if pair_limit != len(ordered):
        first, second, third = ordered[pair_limit:]
        mapping[first] = second
        mapping[second] = third
        mapping[third] = first
    if (
        set(mapping) != set(range(len(task_ids)))
        or set(mapping.values()) != set(range(len(task_ids)))
        or any(target == donor for target, donor in mapping.items())
    ):
        raise ValueError("semantic-body mapping is not an exact derangement")
    return mapping


def build_input_view(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    f2_rows: Sequence[Mapping[str, Any]],
    view: str,
) -> tuple[list[str], dict[str, Any]]:
    """Build one intervention and its sealed aggregate provenance."""

    if view not in SUPPORTED_INPUT_VIEWS:
        raise ValueError(f"unsupported measurement input view: {view!r}")
    if not dataset_rows or len(dataset_rows) != len(f2_rows):
        raise ValueError("dataset/F2 row counts differ for input intervention")
    task_ids: list[str] = []
    parsed: list[ParsedF2] = []
    f2_texts: list[str] = []
    for index, (dataset_row, f2_row) in enumerate(
        zip(dataset_rows, f2_rows, strict=True)
    ):
        task_id = str(dataset_row.get("task_id") or "").strip()
        if not task_id or task_id != str(f2_row.get("task_id") or "").strip():
            raise ValueError(f"row {index}: dataset/F2 identity mismatch")
        text = f2_row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{task_id}: F2 text is absent")
        declared = str(f2_row.get("text_sha256") or "").lower()
        if not _HEX_SHA256.fullmatch(declared) or declared != _sha256_text(text):
            raise ValueError(f"{task_id}: F2 text digest mismatch")
        task_ids.append(task_id)
        f2_texts.append(text)
        parsed.append(parse_f2(text))

    sources: list[str] = []
    row_records: list[dict[str, Any]] = []
    summary: dict[str, Any]
    if view == "semantic_body_swap":
        mapping = _matched_structure_derangement(task_ids, parsed)
        deltas: list[int] = []
        for target_index, task_id in enumerate(task_ids):
            donor_index = mapping[target_index]
            target = parsed[target_index]
            donor = parsed[donor_index]
            transformed = render_f2(
                prefix=target.prefix,
                structure=donor.structure,
            )
            check = parse_f2(transformed)
            if (
                check.prefix_sha256 != target.prefix_sha256
                or check.structure_sha256 != donor.structure_sha256
            ):
                raise ValueError("semantic-body intervention violated its splice")
            delta = donor.structure_bytes - target.structure_bytes
            deltas.append(abs(delta))
            sources.append(SOURCE_PREAMBLE + transformed + SOURCE_SUFFIX)
            row_records.append(
                {
                    "task_id": task_id,
                    "donor_task_id": task_ids[donor_index],
                    "target_prefix_sha256": target.prefix_sha256,
                    "target_structure_sha256": target.structure_sha256,
                    "donor_structure_sha256": donor.structure_sha256,
                    "structure_byte_delta": delta,
                    "source_sha256": _sha256_text(sources[-1]),
                }
            )
        summary = {
            "intervention": "cross_task_structure_permutation",
            "mapping_policy": (
                "structure_utf8_length_then_task_id; adjacent swaps; "
                "largest-three cycle for odd cardinality"
            ),
            "exact_derangement": True,
            "target_recovered_prefix_byte_identical": True,
            "donor_structure_byte_identical": True,
            "median_absolute_structure_byte_delta": median(deltas),
            "maximum_absolute_structure_byte_delta": max(deltas),
        }
    elif view == "constants_stripped":
        changed_rows = 0
        removed_lines = 0
        removed_bytes = 0
        changed_task_ids: list[str] = []
        unchanged_task_ids: list[str] = []
        for task_id, original, value in zip(
            task_ids, f2_texts, parsed, strict=True
        ):
            stripped_prefix, counts = _literal_stripped_prefix(value.prefix)
            transformed = render_f2(
                prefix=stripped_prefix,
                structure=value.structure,
            )
            check = parse_f2(transformed)
            if check.structure_sha256 != value.structure_sha256:
                raise ValueError("constants intervention changed F2 structure")
            if "// externals" in value.prefix and "// externals" not in check.prefix:
                raise ValueError("constants intervention removed external identities")
            changed_rows += int(transformed != original)
            (changed_task_ids if transformed != original else unchanged_task_ids).append(
                task_id
            )
            removed_lines += counts["removed_literal_lines"]
            removed_bytes += counts["removed_literal_bytes"]
            sources.append(SOURCE_PREAMBLE + transformed + SOURCE_SUFFIX)
            row_records.append(
                {
                    "task_id": task_id,
                    "original_prefix_sha256": value.prefix_sha256,
                    "stripped_prefix_sha256": check.prefix_sha256,
                    "structure_sha256": value.structure_sha256,
                    **counts,
                    "source_sha256": _sha256_text(sources[-1]),
                }
            )
        summary = {
            "intervention": "remove_recovered_string_and_number_literals",
            "external_call_identities_preserved": True,
            "f2_structure_byte_identical": True,
            "changed_rows": changed_rows,
            "unchanged_no_literal_rows": len(task_ids) - changed_rows,
            "removed_literal_lines": removed_lines,
            "removed_literal_bytes": removed_bytes,
            "changed_task_ids": changed_task_ids,
            "unchanged_task_ids": unchanged_task_ids,
            "changed_task_ids_sha256": canonical_sha256(changed_task_ids),
            "unchanged_task_ids_sha256": canonical_sha256(unchanged_task_ids),
        }
    else:
        arities: list[int] = []
        return_types: list[str] = []
        for task_id, dataset_row, f2_text in zip(
            task_ids, dataset_rows, f2_texts, strict=True
        ):
            gold_source = dataset_row.get("dart_source")
            signature, signature_record = opaque_contract_signature(gold_source)
            source = _typed_encoder_source(f2_text, signature)
            if str(gold_source).strip() in source:
                raise ValueError("gold implementation body leaked into typed source")
            arities.append(int(signature_record["arity"]))
            return_types.append(str(signature_record["return_type"]))
            sources.append(source)
            row_records.append(
                {
                    "task_id": task_id,
                    **signature_record,
                    "source_sha256": _sha256_text(source),
                }
            )
        summary = {
            "intervention": "gold_derived_types_and_arity_only",
            "gold_implementation_body_exposed_to_model": False,
            "gold_semantic_parameter_names_exposed_to_model": False,
            "function_name": "fn0",
            "parameter_name_policy": "p{zero_based_index}",
            "minimum_arity": min(arities),
            "maximum_arity": max(arities),
            "arity_histogram": {
                str(key): value for key, value in sorted(Counter(arities).items())
            },
            "return_type_histogram": dict(sorted(Counter(return_types).items())),
        }

    if len(sources) != len(task_ids) or any(not source for source in sources):
        raise ValueError("input intervention produced an incomplete source set")
    record = {
        "schema": INPUT_VIEW_SCHEMA,
        "view": view,
        "rows": len(task_ids),
        "ordered_task_ids_sha256": canonical_sha256(task_ids),
        "ordered_source_sha256s_sha256": canonical_sha256(
            [_sha256_text(source) for source in sources]
        ),
        "row_transformations_sha256": canonical_sha256(row_records),
        "tests_exposed_to_model": False,
        "full_gold_targets_exposed_to_model": False,
        "summary": summary,
    }
    return sources, record
