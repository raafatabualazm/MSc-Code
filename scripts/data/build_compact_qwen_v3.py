#!/usr/bin/env python3
"""Lossless compact-Qwen v3 graph + Dart AOT pool-value codec.

V3 deliberately reuses the sealed v2 graph canonicalisation rules.  It adds a
target-scoped binary pool projection to the compact text stream; it
does not inspect Dart source or labels.  The pool is encoded as canonical
ASCII JSON between reserved markers, so its payload uses only the base Qwen
tokenizer while instructions, blocks, routes, and control edges retain v2's
one-token atoms.

The represented domain is exact for:

* the v2 scrubbed canonical graph (DFG is regenerated with the pinned route);
* the ordered list of supported primitive or complete recursive Array/Map
  storage pool records referenced by instructions retained in that canonical
  graph, including strict source-blind nonliteral descriptors nested inside a
  composite; and
* every ordered use-site in those records.

It is intentionally *not* a lossless encoding of the raw target disassembly.
The pinned CFG extractor can deterministically prune unreachable instruction
islands.  Pool xrefs in those islands are outside this codec's domain and must
be enumerated in the separately hash-bound private reconciliation manifest.
No graph-retained supported pool value may be omitted by that reconciliation.

The codec is intentionally fail closed.  Unsupported literal kinds, malformed
payloads, incomplete/unresolved/reference/cyclic composites, non-canonical
JSON, marker injection, and pool use-sites that do not name the matching fixed
``r15`` displacement are rejected.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import build_compact_qwen_v2 as graph_v2


CONTRACT_SCHEMA = "direct-compact-causal-v3"
CODEBOOK_SCHEMA = "compact-qwen-v3-codebook"
PREFLIGHT_SCHEMA = "compact-qwen-v3-preflight"
POOL_SCHEMA = "dart-aot-target-literal-pool-v1"
POOL_ENCODING = "canonical-positional-json-delta-v2"
POOL_SCOPE = "canonical-graph-retained-fixed-r15-uses-v1"
TARGET_FUNCTION = graph_v2.TARGET_FUNCTION
MAX_COMPOSITE_DEPTH = 64
MAX_COMPOSITE_NODES = 100_000

# Keep all v2 route and edge atoms byte-for-byte.  Only the stream/version and
# pool markers are new.
ROUTE_LEGACY = graph_v2.ROUTE_LEGACY
ROUTE_CURRENT = graph_v2.ROUTE_CURRENT
ROUTE_SPECS = graph_v2.ROUTE_SPECS
ROUTE_BY_COMBINED = graph_v2.ROUTE_BY_COMBINED
ROUTE_BY_ATOM = graph_v2.ROUTE_BY_ATOM
EDGE_TOKEN = graph_v2.EDGE_TOKEN
TOKEN_EDGE = graph_v2.TOKEN_EDGE
RUNTIME_POLICY = graph_v2.RUNTIME_POLICY

STREAM_START = "<G2C3>"
POOL_START = "<PX0>"
POOL_END = "<PEND>"
STREAM_END = "<END>"

CONTROL = [
    STREAM_START,
    "<AX64>",
    "<ENTRY>",
    "<BLOCKS>",
    "<CFG>",
    POOL_START,
    POOL_END,
    STREAM_END,
    "<R>",
    "<E>",
] + [spec.atom for spec in ROUTE_SPECS.values()] + list(TOKEN_EDGE)

TAG_RE = graph_v2.TAG_RE
_INTEGER_TEXT_RE = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_DOUBLE_BITS_RE = re.compile(r"[0-9a-f]{16}\Z")
_FIXED_R15_RE = re.compile(
    r"\[\s*r15\s*(?:(?P<sign>[+-])\s*(?P<amount>0x[0-9a-f]+|[0-9]+))?\s*\]",
    re.IGNORECASE,
)

_USE_KEYS = frozenset({"block", "instruction"})
_RECORD_KEYS = frozenset({"pp_offset", "kind", "payload", "use_sites"})
_POOL_KEYS = frozenset({"schema", "target_function", "uses"})
_COMPOSITE_ELEMENT_KEYS = frozenset({"index", "value"})
_NESTED_VALUE_KEYS = frozenset({"kind", "payload"})
_COMPOSITE_TYPES = frozenset({"array_storage", "map_storage"})
_NONLITERAL_PAYLOAD_KEYS = frozenset({"nonliteral_kind", "profile_type"})
# Exact source-blind pairs emitted by the pinned static receipt decoder.  This
# is deliberately a finite pair allowlist: adding a new Dart/V8 profile type is
# a versioned code review, never a permissive regex fallback.  In particular,
# annotations, names, symbols, addresses, node offsets, CIDs, and hashes have no
# representable field.
NESTED_NONLITERAL_PROFILE_KIND = {
    "Type": "type_metadata",
    "TypeArguments": "type_metadata",
    "FunctionType": "type_metadata",
    "TypeParameter": "type_metadata",
    "TypeParameters": "type_metadata",
    "Code": "code",
    "Function": "callable",
    "Closure": "callable",
    "ClosureData": "callable",
    "Field": "declaration_metadata",
    "PatchClass": "declaration_metadata",
    "Class": "declaration_metadata",
    "ArgumentsDescriptor": "calling_convention_metadata",
    "SubtypeTestCache": "calling_convention_metadata",
    "UnlinkedCall": "calling_convention_metadata",
    "Instance": "runtime_object",
    "Record": "runtime_object",
}
_POOL_KIND_TO_TAG = {
    "string": 0,
    "int": 1,
    "double": 2,
    "null": 3,
    "bool": 4,
    "composite": 5,
    "nonliteral": 6,
}
_POOL_TAG_TO_KIND = {value: key for key, value in _POOL_KIND_TO_TAG.items()}
_COMPOSITE_TYPE_TO_TAG = {"array_storage": 0, "map_storage": 1}
_COMPOSITE_TAG_TO_TYPE = {
    value: key for key, value in _COMPOSITE_TYPE_TO_TAG.items()
}
_NONLITERAL_PAIRS = tuple(sorted(NESTED_NONLITERAL_PROFILE_KIND.items()))
_NONLITERAL_PAIR_TO_TAG = {
    pair: index for index, pair in enumerate(_NONLITERAL_PAIRS)
}
_PAYLOAD_KEYS = {
    "string": frozenset({"code_units"}),
    "int": frozenset({"decimal"}),
    "double": frozenset({"bits_hex"}),
    "null": frozenset(),
    "bool": frozenset({"value"}),
    "composite": frozenset(
        {"complete", "composite_type", "elements", "omitted_edge_counts"}
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def graph_codec_sha256() -> str:
    """Return the transitive graph-codec dependency pinned by v3 contracts."""
    return sha256_bytes(Path(graph_v2.__file__).read_bytes())


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label}_keys:missing={missing}:extra={extra}")


def _plain_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label}_must_be_integer")
    return value


def fixed_r15_offsets(instruction: str) -> list[int]:
    """Return fixed signed PP displacements in textual instruction order."""
    offsets: list[int] = []
    for match in _FIXED_R15_RE.finditer(str(instruction)):
        amount = match.group("amount")
        if amount is None:
            offsets.append(0)
            continue
        parsed = int(amount, 0)
        offsets.append(-parsed if match.group("sign") == "-" else parsed)
    return offsets


def _canonical_payload(
    kind: str,
    value: Any,
    *,
    depth: int = 0,
    nodes_seen: list[int] | None = None,
    ancestors: frozenset[int] = frozenset(),
    nested: bool = False,
) -> dict[str, Any]:
    if kind == "nonliteral":
        if not nested:
            raise ValueError("top_level_nonliteral_pool_record_not_supported")
        if not isinstance(value, dict):
            raise ValueError("pool_nonliteral_payload_must_be_object")
        _strict_keys(value, _NONLITERAL_PAYLOAD_KEYS, "pool_nonliteral_payload")
        nonliteral_kind = value["nonliteral_kind"]
        profile_type = value["profile_type"]
        if not isinstance(nonliteral_kind, str) or not isinstance(profile_type, str):
            raise ValueError("pool_nonliteral_descriptor_values_must_be_strings")
        expected_kind = NESTED_NONLITERAL_PROFILE_KIND.get(profile_type)
        if expected_kind is None or nonliteral_kind != expected_kind:
            raise ValueError(
                "unsupported_pool_nonliteral_descriptor_pair:"
                f"{nonliteral_kind}:{profile_type}"
            )
        if nodes_seen is None:
            nodes_seen = [0]
        nodes_seen[0] += 1
        if nodes_seen[0] > MAX_COMPOSITE_NODES:
            raise ValueError("pool_composite_node_limit_exceeded")
        if depth > MAX_COMPOSITE_DEPTH:
            raise ValueError("pool_composite_depth_limit_exceeded")
        return {
            "nonliteral_kind": nonliteral_kind,
            "profile_type": profile_type,
        }
    if kind not in _PAYLOAD_KEYS:
        raise ValueError(f"unsupported_pool_literal_kind:{kind or 'missing'}")
    if not isinstance(value, dict):
        raise ValueError(f"pool_{kind}_payload_must_be_object")
    if nodes_seen is None:
        nodes_seen = [0]
    nodes_seen[0] += 1
    if nodes_seen[0] > MAX_COMPOSITE_NODES:
        raise ValueError("pool_composite_node_limit_exceeded")
    if depth > MAX_COMPOSITE_DEPTH:
        raise ValueError("pool_composite_depth_limit_exceeded")
    _strict_keys(value, _PAYLOAD_KEYS[kind], f"pool_{kind}_payload")

    if kind == "string":
        code_units = value["code_units"]
        if not isinstance(code_units, list):
            raise ValueError("pool_string_code_units_must_be_list")
        canonical_units: list[int] = []
        for index, unit in enumerate(code_units):
            unit = _plain_int(unit, f"pool_string_code_unit_{index}")
            if not 0 <= unit <= 0xFFFF:
                raise ValueError(f"pool_string_code_unit_out_of_range:{index}")
            canonical_units.append(unit)
        return {"code_units": canonical_units}

    if kind == "int":
        decimal = value["decimal"]
        if not isinstance(decimal, str) or not _INTEGER_TEXT_RE.fullmatch(decimal):
            raise ValueError("pool_int_decimal_not_canonical")
        return {"decimal": decimal}

    if kind == "double":
        bits = value["bits_hex"]
        if not isinstance(bits, str) or not _DOUBLE_BITS_RE.fullmatch(bits):
            raise ValueError("pool_double_bits_not_canonical_lower_hex64")
        return {"bits_hex": bits}

    if kind == "null":
        return {}

    if kind == "bool":
        boolean = value["value"]
        if not isinstance(boolean, bool):
            raise ValueError("pool_bool_value_must_be_boolean")
        return {"value": boolean}

    # Composite nodes are compiler-emitted Array/Map storage projections, not
    # inferred source collections.  Element order and duplicate indices are
    # preserved exactly.  Only complete acyclic trees of the exact primitive
    # domain (or further complete composites) are representable.
    identity = id(value)
    if identity in ancestors:
        raise ValueError("pool_composite_cycle_detected")
    if value["complete"] is not True:
        raise ValueError("pool_composite_must_be_complete")
    composite_type = value["composite_type"]
    if composite_type not in _COMPOSITE_TYPES:
        raise ValueError(f"unsupported_pool_composite_type:{composite_type!r}")
    raw_elements = value["elements"]
    if not isinstance(raw_elements, list):
        raise ValueError("pool_composite_elements_must_be_list")
    next_ancestors = ancestors | {identity}
    elements: list[dict[str, Any]] = []
    for element_index, raw_element in enumerate(raw_elements):
        if not isinstance(raw_element, dict):
            raise ValueError(f"pool_composite_element_must_be_object:{element_index}")
        _strict_keys(
            raw_element,
            _COMPOSITE_ELEMENT_KEYS,
            f"pool_composite_element_{element_index}",
        )
        index = _plain_int(
            raw_element["index"], f"pool_composite_element_index_{element_index}"
        )
        if index < 0:
            raise ValueError(f"pool_composite_element_index_negative:{element_index}")
        raw_nested = raw_element["value"]
        if not isinstance(raw_nested, dict):
            raise ValueError(f"pool_composite_value_must_be_object:{element_index}")
        _strict_keys(
            raw_nested,
            _NESTED_VALUE_KEYS,
            f"pool_composite_value_{element_index}",
        )
        nested_kind = raw_nested["kind"]
        if not isinstance(nested_kind, str):
            raise ValueError(f"pool_composite_value_kind_must_be_string:{element_index}")
        nested_payload = _canonical_payload(
            nested_kind,
            raw_nested["payload"],
            depth=depth + 1,
            nodes_seen=nodes_seen,
            ancestors=next_ancestors,
            nested=True,
        )
        elements.append(
            {"index": index, "value": {"kind": nested_kind, "payload": nested_payload}}
        )

    raw_omitted = value["omitted_edge_counts"]
    if not isinstance(raw_omitted, dict):
        raise ValueError("pool_composite_omitted_edge_counts_must_be_object")
    omitted: dict[str, int] = {}
    for edge_type, raw_count in raw_omitted.items():
        if not isinstance(edge_type, str) or not edge_type or any(
            character in edge_type for character in "<>&"
        ):
            raise ValueError("pool_composite_omitted_edge_type_invalid")
        count = _plain_int(
            raw_count, f"pool_composite_omitted_edge_count_{edge_type}"
        )
        if count < 0:
            raise ValueError(
                f"pool_composite_omitted_edge_count_negative:{edge_type}"
            )
        omitted[edge_type] = count
    return {
        "complete": True,
        "composite_type": composite_type,
        "elements": elements,
        "omitted_edge_counts": dict(sorted(omitted.items())),
    }


def canonicalize_pool_uses(
    value: Any,
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate and copy the ordered pool-value projection without sorting.

    Duplicate records and duplicate use-sites are meaningful and preserved.
    The projection is scoped to canonical-graph-retained instructions.  When
    ``blocks`` are supplied, every use-site must point to an extant instruction
    containing the record's exact signed fixed ``r15`` offset.  Raw AOT xrefs
    absent from this canonical graph belong only in the private reconciliation
    manifest and cannot be inserted into this stream with a fabricated site.
    """
    if not isinstance(value, list):
        raise ValueError("binary_pool_uses_must_be_list")
    result: list[dict[str, Any]] = []
    for record_index, raw_record in enumerate(value):
        if not isinstance(raw_record, dict):
            raise ValueError(f"binary_pool_record_must_be_object:{record_index}")
        _strict_keys(raw_record, _RECORD_KEYS, f"binary_pool_record_{record_index}")
        pp_offset = _plain_int(
            raw_record["pp_offset"], f"binary_pool_pp_offset_{record_index}"
        )
        kind = raw_record["kind"]
        if not isinstance(kind, str):
            raise ValueError(f"binary_pool_kind_must_be_string:{record_index}")
        payload = _canonical_payload(kind, raw_record["payload"])
        raw_sites = raw_record["use_sites"]
        if not isinstance(raw_sites, list) or not raw_sites:
            raise ValueError(f"binary_pool_use_sites_must_be_nonempty_list:{record_index}")
        sites: list[dict[str, int]] = []
        for site_index, raw_site in enumerate(raw_sites):
            if not isinstance(raw_site, dict):
                raise ValueError(
                    f"binary_pool_use_site_must_be_object:{record_index}:{site_index}"
                )
            _strict_keys(
                raw_site,
                _USE_KEYS,
                f"binary_pool_use_site_{record_index}_{site_index}",
            )
            block_id = _plain_int(
                raw_site["block"],
                f"binary_pool_use_block_{record_index}_{site_index}",
            )
            instruction_id = _plain_int(
                raw_site["instruction"],
                f"binary_pool_use_instruction_{record_index}_{site_index}",
            )
            if block_id < 0 or instruction_id < 0:
                raise ValueError(
                    f"binary_pool_use_site_negative:{record_index}:{site_index}"
                )
            if blocks is not None:
                if block_id >= len(blocks):
                    raise ValueError(
                        f"binary_pool_use_block_out_of_range:{record_index}:{site_index}"
                    )
                block = blocks[block_id]
                if int(block.get("id", -1)) != block_id:
                    raise ValueError("binary_pool_blocks_not_position_aligned")
                instructions = block.get("instructions")
                if not isinstance(instructions, list) or instruction_id >= len(instructions):
                    raise ValueError(
                        f"binary_pool_use_instruction_out_of_range:{record_index}:{site_index}"
                    )
                offsets = fixed_r15_offsets(str(instructions[instruction_id]))
                if pp_offset not in offsets:
                    raise ValueError(
                        "binary_pool_use_offset_not_present_at_site:"
                        f"{record_index}:{site_index}:{pp_offset}"
                    )
            sites.append({"block": block_id, "instruction": instruction_id})
        result.append(
            {
                "pp_offset": pp_offset,
                "kind": kind,
                "payload": payload,
                "use_sites": sites,
            }
        )
    return result


def pool_envelope(
    uses: Any,
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": POOL_SCHEMA,
        "target_function": TARGET_FUNCTION,
        "uses": canonicalize_pool_uses(uses, blocks=blocks),
    }


def _validate_pool_envelope(
    value: Any,
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("binary_pool_envelope_must_be_object")
    _strict_keys(value, _POOL_KEYS, "binary_pool_envelope")
    if value["schema"] != POOL_SCHEMA:
        raise ValueError("binary_pool_schema_mismatch")
    if value["target_function"] != TARGET_FUNCTION:
        raise ValueError("binary_pool_target_function_must_be_candidate")
    return pool_envelope(value["uses"], blocks=blocks)


def _utf16_units_to_text(units: Sequence[int]) -> str:
    raw = bytearray()
    for unit in units:
        raw.extend((unit & 0xFF, unit >> 8))
    return bytes(raw).decode("utf-16-le", errors="surrogatepass")


def _text_to_utf16_units(value: str) -> list[int]:
    raw = value.encode("utf-16-le", errors="surrogatepass")
    return [raw[index] | (raw[index + 1] << 8) for index in range(0, len(raw), 2)]


def _encode_pool_value(kind: str, payload: Mapping[str, Any]) -> list[Any]:
    tag = _POOL_KIND_TO_TAG[kind]
    if kind == "string":
        return [tag, _utf16_units_to_text(payload["code_units"])]
    if kind == "int":
        return [tag, payload["decimal"]]
    if kind == "double":
        return [tag, payload["bits_hex"]]
    if kind == "null":
        return [tag]
    if kind == "bool":
        return [tag, payload["value"]]
    if kind == "nonliteral":
        pair = (payload["profile_type"], payload["nonliteral_kind"])
        return [tag, _NONLITERAL_PAIR_TO_TAG[pair]]

    elements: list[Any] = []
    previous_index = 0
    for element in payload["elements"]:
        index = element["index"]
        nested = element["value"]
        elements.extend(
            [index - previous_index, _encode_pool_value(nested["kind"], nested["payload"])]
        )
        previous_index = index
    omitted: list[Any] = []
    for edge_type, count in sorted(payload["omitted_edge_counts"].items()):
        omitted.extend([edge_type, count])
    return [
        tag,
        _COMPOSITE_TYPE_TO_TAG[payload["composite_type"]],
        elements,
        omitted,
    ]


def _decode_pool_value(value: Any, *, nested: bool) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("pool_positional_value_must_be_nonempty_array")
    tag = _plain_int(value[0], "pool_positional_kind_tag")
    kind = _POOL_TAG_TO_KIND.get(tag)
    if kind is None or (kind == "nonliteral" and not nested):
        raise ValueError(f"pool_positional_kind_tag_invalid:{tag}")
    if kind == "string":
        if len(value) != 2 or not isinstance(value[1], str):
            raise ValueError("pool_positional_string_shape")
        return kind, {"code_units": _text_to_utf16_units(value[1])}
    if kind == "int":
        if len(value) != 2:
            raise ValueError("pool_positional_int_shape")
        return kind, {"decimal": value[1]}
    if kind == "double":
        if len(value) != 2:
            raise ValueError("pool_positional_double_shape")
        return kind, {"bits_hex": value[1]}
    if kind == "null":
        if len(value) != 1:
            raise ValueError("pool_positional_null_shape")
        return kind, {}
    if kind == "bool":
        if len(value) != 2:
            raise ValueError("pool_positional_bool_shape")
        return kind, {"value": value[1]}
    if kind == "nonliteral":
        if len(value) != 2:
            raise ValueError("pool_positional_nonliteral_shape")
        pair_tag = _plain_int(value[1], "pool_positional_nonliteral_tag")
        if not 0 <= pair_tag < len(_NONLITERAL_PAIRS):
            raise ValueError("pool_positional_nonliteral_tag_out_of_range")
        profile_type, nonliteral_kind = _NONLITERAL_PAIRS[pair_tag]
        return kind, {
            "nonliteral_kind": nonliteral_kind,
            "profile_type": profile_type,
        }

    if len(value) != 4:
        raise ValueError("pool_positional_composite_shape")
    type_tag = _plain_int(value[1], "pool_positional_composite_type_tag")
    composite_type = _COMPOSITE_TAG_TO_TYPE.get(type_tag)
    if composite_type is None:
        raise ValueError("pool_positional_composite_type_tag_invalid")
    raw_elements = value[2]
    if not isinstance(raw_elements, list) or len(raw_elements) % 2:
        raise ValueError("pool_positional_composite_elements_shape")
    elements: list[dict[str, Any]] = []
    previous_index = 0
    for position in range(0, len(raw_elements), 2):
        delta = _plain_int(
            raw_elements[position], "pool_positional_composite_index_delta"
        )
        index = previous_index + delta
        if index < 0:
            raise ValueError("pool_positional_composite_index_negative")
        nested_kind, nested_payload = _decode_pool_value(
            raw_elements[position + 1], nested=True
        )
        elements.append(
            {
                "index": index,
                "value": {"kind": nested_kind, "payload": nested_payload},
            }
        )
        previous_index = index
    raw_omitted = value[3]
    if not isinstance(raw_omitted, list) or len(raw_omitted) % 2:
        raise ValueError("pool_positional_composite_omitted_shape")
    omitted: dict[str, int] = {}
    for position in range(0, len(raw_omitted), 2):
        edge_type = raw_omitted[position]
        if not isinstance(edge_type, str) or edge_type in omitted:
            raise ValueError("pool_positional_composite_omitted_key_invalid")
        omitted[edge_type] = _plain_int(
            raw_omitted[position + 1], "pool_positional_composite_omitted_count"
        )
    return kind, {
        "complete": True,
        "composite_type": composite_type,
        "elements": elements,
        "omitted_edge_counts": omitted,
    }


def _encode_pool_positional(canonical: Mapping[str, Any]) -> list[list[Any]]:
    records: list[list[Any]] = []
    previous_pp = 0
    for record in canonical["uses"]:
        sites: list[int] = []
        previous_block = 0
        previous_instruction = 0
        for site in record["use_sites"]:
            block = site["block"]
            instruction = site["instruction"]
            block_delta = block - previous_block
            instruction_delta = (
                instruction - previous_instruction
                if block_delta == 0
                else instruction
            )
            sites.extend([block_delta, instruction_delta])
            previous_block = block
            previous_instruction = instruction
        records.append(
            [
                record["pp_offset"] - previous_pp,
                *_encode_pool_value(record["kind"], record["payload"]),
                sites,
            ]
        )
        previous_pp = record["pp_offset"]
    return records


def _decode_pool_positional(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        raise ValueError("pool_positional_root_must_be_array")
    records: list[dict[str, Any]] = []
    previous_pp = 0
    for record_index, raw_record in enumerate(value):
        if not isinstance(raw_record, list) or len(raw_record) < 3:
            raise ValueError(f"pool_positional_record_shape:{record_index}")
        pp_delta = _plain_int(raw_record[0], "pool_positional_pp_delta")
        tag = _plain_int(raw_record[1], "pool_positional_record_kind_tag")
        kind = _POOL_TAG_TO_KIND.get(tag)
        if kind is None or kind == "nonliteral":
            raise ValueError(f"pool_positional_record_kind_invalid:{record_index}")
        value_length = 4 if kind == "composite" else (1 if kind == "null" else 2)
        expected_length = 2 + (value_length - 1) + 1
        if len(raw_record) != expected_length:
            raise ValueError(f"pool_positional_record_length:{record_index}")
        encoded_value = [tag, *raw_record[2:-1]]
        decoded_kind, payload = _decode_pool_value(encoded_value, nested=False)
        raw_sites = raw_record[-1]
        if not isinstance(raw_sites, list) or not raw_sites or len(raw_sites) % 2:
            raise ValueError(f"pool_positional_sites_shape:{record_index}")
        sites: list[dict[str, int]] = []
        previous_block = 0
        previous_instruction = 0
        for position in range(0, len(raw_sites), 2):
            block_delta = _plain_int(
                raw_sites[position], "pool_positional_site_block_delta"
            )
            instruction_delta = _plain_int(
                raw_sites[position + 1], "pool_positional_site_instruction_delta"
            )
            block = previous_block + block_delta
            instruction = (
                previous_instruction + instruction_delta
                if block_delta == 0
                else instruction_delta
            )
            if block < 0 or instruction < 0:
                raise ValueError(f"pool_positional_site_negative:{record_index}")
            sites.append({"block": block, "instruction": instruction})
            previous_block = block
            previous_instruction = instruction
        pp_offset = previous_pp + pp_delta
        records.append(
            {
                "pp_offset": pp_offset,
                "kind": decoded_kind,
                "payload": payload,
                "use_sites": sites,
            }
        )
        previous_pp = pp_offset
    return pool_envelope(records)


def canonical_pool_json(value: Mapping[str, Any]) -> str:
    """Serialize the exact pool as compact, positional, canonical ASCII JSON.

    Schema/target are contract constants.  Kind/type/nonliteral enums are
    finite contract tables; record offsets, element indices, and use sites are
    delta coded.  Dart strings are represented as JSON strings reconstructed
    from their exact UTF-16 code units, including unpaired surrogates.
    """
    canonical = _validate_pool_envelope(value)
    text = json.dumps(
        _encode_pool_positional(canonical),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def canonicalize(
    row: dict[str, Any],
    symbol_policy: str = "runtime_aware",
    route_override: str | None = None,
) -> dict[str, Any]:
    canonical = graph_v2.canonicalize(row, symbol_policy, route_override)
    if "binary_pool_uses" not in row:
        raise ValueError("missing_binary_pool_uses")
    canonical["binary_pool"] = pool_envelope(
        row["binary_pool_uses"], blocks=canonical["blocks"]
    )
    return canonical


def source_token_contract(
    tokenizer_path: Path,
    model_vocab_size: int,
    expansions: list[str],
    max_blocks: int,
) -> tuple[int, dict[str, list[int]], dict[str, int]]:
    """Create v3 custom-token IDs while leaving pool JSON base-tokenized."""
    base = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_size = base.get_vocab_size(with_added_tokens=True)
    human = {
        STREAM_START: " compact graph and literal pool version three",
        "<AX64>": " x86 64",
        "<DX0>": " frozen legacy data flow extractor",
        "<DX1>": " current combined data flow extractor",
        "<ENTRY>": " entry blocks",
        "<BLOCKS>": " basic blocks",
        "<CFG>": " control flow edges",
        POOL_START: " target binary literal pool",
        POOL_END: " end target binary literal pool",
        STREAM_END: " end compact representation",
        "<R>": " raw instruction",
        "<E>": " end raw instruction",
        "<CT>": " conditional true",
        "<CF>": " conditional false",
        "<CN>": " linear fallthrough",
        "<CL>": " loop backedge",
        "<CU>": " unconditional",
        "<CJ>": " unconditional jump",
        "<CC>": " internal call edge",
    }
    atoms = (
        [f"<I{index}>" for index in range(len(expansions))]
        + [f"<B{index}>" for index in range(max_blocks)]
        + CONTROL
    )
    if len(atoms) != len(set(atoms)):
        raise ValueError("duplicate_compact_atoms")
    atom_ids = {token: model_vocab_size + index for index, token in enumerate(atoms)}
    mapping: dict[str, list[int]] = {}
    for index, line in enumerate(expansions):
        mapping[str(atom_ids[f"<I{index}>"])] = base.encode(
            line, add_special_tokens=False
        ).ids
    for index in range(max_blocks):
        mapping[str(atom_ids[f"<B{index}>"])] = base.encode(
            f" block {index}", add_special_tokens=False
        ).ids
    for token, text in human.items():
        mapping[str(atom_ids[token])] = base.encode(text, add_special_tokens=False).ids
    if any(not value for value in mapping.values()):
        raise ValueError("invalid_source_token_expansion")
    return tokenizer_size, mapping, atom_ids


def compact_ids(text: str, base: Tokenizer, atom_ids: dict[str, int]) -> list[int]:
    return graph_v2.compact_ids(text, base, atom_ids)


def encode(canonical: dict[str, Any], code: dict[str, int]) -> str:
    if "binary_pool" not in canonical:
        raise ValueError("missing_canonical_binary_pool")
    graph_text = graph_v2.encode(canonical, code)
    if not graph_text.startswith("<G2C2>") or not graph_text.endswith(STREAM_END):
        raise AssertionError("v2_graph_codec_stream_shape_drift")
    graph_body = graph_text[len("<G2C2>") : -len(STREAM_END)]
    pool = _validate_pool_envelope(
        canonical["binary_pool"], blocks=canonical.get("blocks")
    )
    pool_text = canonical_pool_json(pool)
    return STREAM_START + graph_body + POOL_START + pool_text + POOL_END + STREAM_END


def decode(text: str, expansions: list[str]) -> dict[str, Any]:
    if not isinstance(text, str) or not text.startswith(STREAM_START):
        raise ValueError("missing_v3_stream_start")
    if not text.endswith(POOL_END + STREAM_END):
        raise ValueError("missing_v3_pool_or_stream_end")
    if text.count(POOL_START) != 1 or text.count(POOL_END) != 1:
        raise ValueError("pool_marker_count_mismatch")
    pool_start = text.find(POOL_START)
    pool_end = text.find(POOL_END, pool_start + len(POOL_START))
    if pool_start < len(STREAM_START) or pool_end < pool_start:
        raise ValueError("invalid_pool_marker_order")

    graph_body = text[len(STREAM_START) : pool_start]
    graph_text = "<G2C2>" + graph_body + STREAM_END
    decoded = graph_v2.decode(graph_text, expansions)

    pool_text = text[pool_start + len(POOL_START) : pool_end]
    try:
        raw_pool = json.loads(pool_text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid_binary_pool_json") from error
    canonical_pool = _validate_pool_envelope(
        _decode_pool_positional(raw_pool), blocks=decoded["blocks"]
    )
    if canonical_pool_json(canonical_pool) != pool_text:
        raise ValueError("binary_pool_json_not_canonical")
    decoded["binary_pool"] = canonical_pool
    return decoded


def regenerate_dfg(
    decoded: dict[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Regenerate v2 route-specific DFG and return a complete canonical copy."""
    route = str(decoded.get("dfg_route") or "")
    if route not in registry:
        raise ValueError(f"missing_dfg_registry_route:{route or 'missing'}")
    result = dict(decoded)
    regenerated = registry[route]["build_dfg"](
        result["blocks"], result["cfg_edges"], max_edges=100000
    )
    result["dfg_edges"] = graph_v2._sort_dfg(
        graph_v2._canonical_dfg_edge(edge, route) for edge in regenerated
    )
    return result


def codec_contract(
    *,
    codec_sha256: str,
    codebook_sha256: str,
    tokenizer_json_sha256: str,
    pool_extractor_sha256: str,
    aot_manifest_sha256: str,
    dart_toolchain_manifest_sha256: str,
    pool_reconciliation_manifest_sha256: str,
) -> dict[str, Any]:
    """Return the mandatory v3 lossless/provenance portion of a release seal."""
    values = {
        "codec_sha256": codec_sha256,
        "codebook_sha256": codebook_sha256,
        "tokenizer_json_sha256": tokenizer_json_sha256,
        "pool_extractor_sha256": pool_extractor_sha256,
        "aot_manifest_sha256": aot_manifest_sha256,
        "dart_toolchain_manifest_sha256": dart_toolchain_manifest_sha256,
        "pool_reconciliation_manifest_sha256": pool_reconciliation_manifest_sha256,
    }
    for label, value in values.items():
        graph_v2.digest(value, label)
    return {
        "schema": CONTRACT_SCHEMA,
        **values,
        "graph_codec_dependency_sha256": graph_codec_sha256(),
        "target_function": TARGET_FUNCTION,
        "target_architecture": "x86_64",
        "pool_schema": POOL_SCHEMA,
        "pool_encoding": POOL_ENCODING,
        "pool_positional_encoding": {
            "kind_to_tag": dict(_POOL_KIND_TO_TAG),
            "composite_type_to_tag": dict(_COMPOSITE_TYPE_TO_TAG),
            "nonliteral_pairs": [
                {"profile_type": profile_type, "nonliteral_kind": nonliteral_kind}
                for profile_type, nonliteral_kind in _NONLITERAL_PAIRS
            ],
            "record_pp_offsets": "signed-delta-from-previous-starting-zero",
            "composite_indices": "signed-delta-from-previous-starting-zero",
            "use_sites": (
                "flat-block-delta-and-same-block-instruction-delta-"
                "otherwise-absolute-instruction"
            ),
            "schema_and_target": "implicit-contract-constants",
            "utf16_json_strings": "surrogatepass-exact-code-unit-roundtrip",
        },
        "pool_scope": POOL_SCOPE,
        "pool_projection": (
            "canonical-graph-retained-target-fixed-r15-exact-primitives-and-"
            "complete-recursive-array-map-storage-with-source-blind-nested-"
            "nonliteral-descriptors-and-exact-use-sites"
        ),
        "all_encoded_pool_uses_reference_canonical_graph_instructions": True,
        "raw_disassembly_unreachable_islands_in_lossless_domain": False,
        "non_graph_aot_xrefs": (
            "excluded-by-projection-and-exhaustively-accounted-in-private-"
            "reconciliation-manifest"
        ),
        "graph_retained_literal_use_omission_policy": (
            "reject-via-hash-bound-private-reconciliation"
        ),
        "pool_order_and_duplicates_preserved": True,
        "string_representation": "ordered-dart-utf16-code-units",
        "integer_representation": "canonical-signed-decimal",
        "double_representation": "exact-ieee754-binary64-bits-lower-hex",
        "composite_representation": {
            "types": sorted(_COMPOSITE_TYPES),
            "ordered_element_indices_preserved": True,
            "duplicate_element_indices_preserved": True,
            "omitted_edge_counts_preserved": True,
            "incomplete_unresolved_reference_or_cyclic_nodes": "reject",
            "max_depth": MAX_COMPOSITE_DEPTH,
            "max_nodes_per_pool_record": MAX_COMPOSITE_NODES,
        },
        "nested_nonliteral_descriptors": {
            "top_level_records": "reject",
            "nested_composite_values": "allow-strict-source-blind-pairs",
            "payload_keys": sorted(_NONLITERAL_PAYLOAD_KEYS),
            "profile_type_to_nonliteral_kind": dict(
                sorted(NESTED_NONLITERAL_PROFILE_KIND.items())
            ),
            "names_symbols_addresses_offsets_cids_and_hashes": "unrepresentable",
            "unresolved_truncated_or_reference_nodes": "reject",
        },
        "lossless_domain": (
            "scrubbed-canonical-graph-v2-plus-complete-source-blind-pool-"
            "values-at-"
            "canonical-graph-retained-fixed-r15-uses-v1"
        ),
    }


def iter_pool_uses(canonical_rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield pool records without changing row/record order (audit helper)."""
    for row in canonical_rows:
        pool = _validate_pool_envelope(row.get("binary_pool"))
        yield from pool["uses"]


__all__ = [
    "CODEBOOK_SCHEMA",
    "CONTRACT_SCHEMA",
    "CONTROL",
    "EDGE_TOKEN",
    "POOL_ENCODING",
    "POOL_SCHEMA",
    "POOL_SCOPE",
    "PREFLIGHT_SCHEMA",
    "MAX_COMPOSITE_DEPTH",
    "MAX_COMPOSITE_NODES",
    "NESTED_NONLITERAL_PROFILE_KIND",
    "ROUTE_CURRENT",
    "ROUTE_LEGACY",
    "ROUTE_SPECS",
    "RUNTIME_POLICY",
    "TARGET_FUNCTION",
    "canonical_pool_json",
    "canonicalize",
    "canonicalize_pool_uses",
    "codec_contract",
    "compact_ids",
    "decode",
    "encode",
    "fixed_r15_offsets",
    "graph_codec_sha256",
    "pool_envelope",
    "regenerate_dfg",
    "source_token_contract",
]
