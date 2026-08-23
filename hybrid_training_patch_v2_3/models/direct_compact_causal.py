"""Encoder-free utilities for compact-binary causal language modeling.

This module intentionally depends only on PyTorch.  In particular, it must not
import GraphCodeBERT, an encoder ``AutoModel``, PyG, a CFG tensor builder, or any
soft-prefix component.  Compact binary records are ordinary decoder token IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTRACT_SCHEMA_V1 = "direct-compact-causal-v1"
CONTRACT_SCHEMA_V2 = "direct-compact-causal-v2"
CONTRACT_SCHEMA_V3 = "direct-compact-causal-v3"
# Backward-compatible public alias used by the existing v1 training/tests.
CONTRACT_SCHEMA = CONTRACT_SCHEMA_V1
SUPPORTED_CONTRACT_SCHEMAS = frozenset(
    (CONTRACT_SCHEMA_V1, CONTRACT_SCHEMA_V2, CONTRACT_SCHEMA_V3)
)

CODEBOOK_SCHEMA_V3 = "compact-qwen-v3-codebook"
LOSSLESS_DOMAIN_V3 = (
    "scrubbed-canonical-graph-v2-plus-complete-source-blind-pool-values-at-"
    "canonical-graph-retained-fixed-r15-uses-v1"
)
POOL_PAYLOAD_SCHEMA_V1 = "dart-aot-target-literal-pool-v1"
POOL_ENCODING_V1 = "canonical-positional-json-delta-v2"
POOL_PROJECTION_V1 = (
    "canonical-graph-retained-target-fixed-r15-exact-primitives-and-"
    "complete-recursive-array-map-storage-with-source-blind-nested-"
    "nonliteral-descriptors-and-exact-use-sites"
)
POOL_SCOPE_V1 = "canonical-graph-retained-fixed-r15-uses-v1"
NON_GRAPH_AOT_XREF_POLICY_V1 = (
    "excluded-by-projection-and-exhaustively-accounted-in-private-"
    "reconciliation-manifest"
)
GRAPH_LITERAL_OMISSION_POLICY_V1 = (
    "reject-via-hash-bound-private-reconciliation"
)
TARGET_ARCHITECTURE_V3 = "x86_64"
STRING_REPRESENTATION_V1 = "ordered-dart-utf16-code-units"
INTEGER_REPRESENTATION_V1 = "canonical-signed-decimal"
DOUBLE_REPRESENTATION_V1 = "exact-ieee754-binary64-bits-lower-hex"
MAX_COMPOSITE_DEPTH_V1 = 64
MAX_COMPOSITE_NODES_V1 = 100_000
COMPOSITE_REPRESENTATION_V1 = {
    "types": ["array_storage", "map_storage"],
    "ordered_element_indices_preserved": True,
    "duplicate_element_indices_preserved": True,
    "omitted_edge_counts_preserved": True,
    "incomplete_unresolved_reference_or_cyclic_nodes": "reject",
    "max_depth": MAX_COMPOSITE_DEPTH_V1,
    "max_nodes_per_pool_record": MAX_COMPOSITE_NODES_V1,
}
NESTED_NONLITERAL_PROFILE_KIND_V1 = {
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
_POOL_KIND_TO_TAG_V2 = {
    "string": 0,
    "int": 1,
    "double": 2,
    "null": 3,
    "bool": 4,
    "composite": 5,
    "nonliteral": 6,
}
_POOL_TAG_TO_KIND_V2 = {value: key for key, value in _POOL_KIND_TO_TAG_V2.items()}
_COMPOSITE_TYPE_TO_TAG_V2 = {"array_storage": 0, "map_storage": 1}
_COMPOSITE_TAG_TO_TYPE_V2 = {
    value: key for key, value in _COMPOSITE_TYPE_TO_TAG_V2.items()
}
_NONLITERAL_PAIRS_V2 = tuple(sorted(NESTED_NONLITERAL_PROFILE_KIND_V1.items()))
_NONLITERAL_PAIR_TO_TAG_V2 = {
    pair: index for index, pair in enumerate(_NONLITERAL_PAIRS_V2)
}
POOL_POSITIONAL_ENCODING_V2 = {
    "kind_to_tag": dict(_POOL_KIND_TO_TAG_V2),
    "composite_type_to_tag": dict(_COMPOSITE_TYPE_TO_TAG_V2),
    "nonliteral_pairs": [
        {"profile_type": profile_type, "nonliteral_kind": nonliteral_kind}
        for profile_type, nonliteral_kind in _NONLITERAL_PAIRS_V2
    ],
    "record_pp_offsets": "signed-delta-from-previous-starting-zero",
    "composite_indices": "signed-delta-from-previous-starting-zero",
    "use_sites": (
        "flat-block-delta-and-same-block-instruction-delta-"
        "otherwise-absolute-instruction"
    ),
    "schema_and_target": "implicit-contract-constants",
    "utf16_json_strings": "surrogatepass-exact-code-unit-roundtrip",
}
NESTED_NONLITERAL_DESCRIPTORS_V1 = {
    "top_level_records": "reject",
    "nested_composite_values": "allow-strict-source-blind-pairs",
    "payload_keys": ["nonliteral_kind", "profile_type"],
    "profile_type_to_nonliteral_kind": dict(
        sorted(NESTED_NONLITERAL_PROFILE_KIND_V1.items())
    ),
    "names_symbols_addresses_offsets_cids_and_hashes": "unrepresentable",
    "unresolved_truncated_or_reference_nodes": "reject",
}
POOL_ALIGNMENT_SCHEMA_V1 = "dart-aot-target-pool-alignment-v1"
JOIN_SEAL_SCHEMA_V1 = "compact-public-private-join-seal-v1"
JOIN_SEAL_SCHEMA_V2 = "compact-public-private-join-seal-v2"

# V3's structural grammar is contract-bound by token ID, not rediscovered by
# decoding untrusted row content.  The pool payload between <PX0> and <PEND> is
# ordinary base-tokenizer JSON; all five markers themselves are source atoms.
V3_STREAM_MARKERS = ("<G2C3>", "<CFG>", "<PX0>", "<PEND>", "<END>")

_POOL_ALIGNMENT_FIELDS = frozenset(
    {
        "schema",
        "receipt_sha256",
        "projection_sha256",
        "use_count",
        "source_blind",
        "target_function",
    }
)

_EXTRACTOR_ROUTE_FIELDS = frozenset(
    {
        "allow_call_edges",
        "cfg_extractor_sha256",
        "combined_hash_algorithm",
        "dfg_extractor_sha256",
        "dfg_metadata",
        "graph_extractor_sha256",
        "route_atom",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: str | Path) -> str:
    """Hash one file or a directory tree without binding absolute paths.

    Directory hashes cover every regular file's POSIX relative path, byte size,
    and content digest in lexical order.  Symlinks are rejected so an adapter
    provenance record cannot silently depend on mutable data outside the tree.
    """

    artifact = Path(path)
    if artifact.is_file():
        return sha256_file(artifact)
    if not artifact.is_dir():
        raise FileNotFoundError(f"artifact does not exist: {artifact}")
    digest = hashlib.sha256(b"direct-compact-artifact-tree-v1\0")
    files = sorted(
        (item for item in artifact.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(artifact).as_posix(),
    )
    if not files:
        raise ValueError(f"artifact directory is empty: {artifact}")
    for item in files:
        if item.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {item}")
        relative = item.relative_to(artifact).as_posix().encode("utf-8")
        size = item.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def validate_join_seal(
    dataset_path: str | Path,
    seal_path: str | Path,
    contract_path: str | Path,
    *,
    expected_role: str,
) -> dict[str, Any]:
    """Fail closed unless a supervised JSONL matches its sealed join record."""

    dataset = Path(dataset_path)
    seal_file = Path(seal_path)
    with seal_file.open("r", encoding="utf-8") as handle:
        seal = json.load(handle)
    if not isinstance(seal, dict):
        raise ValueError(f"invalid compact join seal: {seal_file}")
    with Path(contract_path).open("r", encoding="utf-8") as handle:
        raw_contract = json.load(handle)
    contract: DirectCompactContract | None
    if isinstance(raw_contract, Mapping) and raw_contract.get("schema"):
        contract = DirectCompactContract.from_mapping(raw_contract)
    elif seal.get("schema") == JOIN_SEAL_SCHEMA_V1:
        # Preserve the original v1 seal verifier's ability to bind an opaque
        # contract artifact by hash.  V2/V3 contracts emitted by this pipeline
        # always carry an explicit schema; a v3 seal can never take this path.
        contract = None
    else:
        raise ValueError("schema-bearing compact contract required by join seal")
    expected_seal_schema = (
        JOIN_SEAL_SCHEMA_V2
        if contract is not None and contract.schema == CONTRACT_SCHEMA_V3
        else JOIN_SEAL_SCHEMA_V1
    )
    if seal.get("schema") != expected_seal_schema:
        contract_schema = contract.schema if contract is not None else "legacy-v1"
        raise ValueError(
            f"invalid compact join seal schema for {contract_schema}: "
            f"expected {expected_seal_schema!r}, observed {seal.get('schema')!r}"
        )
    if seal.get("selected_role") != expected_role:
        raise ValueError(
            f"join seal role mismatch: expected {expected_role!r}, "
            f"observed {seal.get('selected_role')!r}"
        )
    expected_output = _canonical_sha256(seal.get("output_sha256"), "seal.output_sha256")
    observed_output = sha256_file(dataset)
    if observed_output != expected_output:
        raise ValueError(
            f"sealed dataset SHA-256 mismatch: expected {expected_output}, "
            f"observed {observed_output}"
        )
    expected_contract = _canonical_sha256(
        seal.get("contract_sha256"), "seal.contract_sha256"
    )
    observed_contract = sha256_file(contract_path)
    if observed_contract != expected_contract:
        raise ValueError(
            f"join seal contract mismatch: expected {expected_contract}, "
            f"observed {observed_contract}"
        )
    row_count = 0
    with dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"sealed dataset contains a blank row: {dataset}")
            row_count += 1
    if row_count != int(seal.get("rows", -1)):
        raise ValueError(
            f"sealed dataset row-count mismatch: expected {seal.get('rows')}, "
            f"observed {row_count}"
        )
    if contract is not None and contract.schema == CONTRACT_SCHEMA_V3:
        if seal.get("contract_schema") != CONTRACT_SCHEMA_V3:
            raise ValueError("v3 join seal contract_schema mismatch")
        pool = seal.get("pool_metadata")
        if not isinstance(pool, Mapping):
            raise ValueError("v3 join seal has no pool_metadata summary")
        expected_pool_fields = {
            "schema",
            "rows",
            "source_blind_rows",
            "target_function",
            "projection_sha256",
            "total_use_count",
        }
        if set(pool) != expected_pool_fields:
            raise ValueError("v3 join seal pool_metadata field mismatch")
        if pool.get("schema") != POOL_ALIGNMENT_SCHEMA_V1:
            raise ValueError("v3 join seal pool_metadata schema mismatch")
        pool_rows = pool.get("rows")
        if isinstance(pool_rows, bool) or not isinstance(pool_rows, int):
            raise ValueError("v3 join seal pool_metadata rows is invalid")
        if pool_rows != row_count:
            raise ValueError("v3 join seal pool_metadata row-count mismatch")
        source_blind_rows = pool.get("source_blind_rows")
        if isinstance(source_blind_rows, bool) or not isinstance(
            source_blind_rows, int
        ):
            raise ValueError("v3 join seal pool source_blind_rows is invalid")
        if source_blind_rows != row_count:
            raise ValueError("v3 join seal contains a non-source-blind pool row")
        if pool.get("target_function") != contract.target_function:
            raise ValueError("v3 join seal pool target_function mismatch")
        _canonical_sha256(
            pool.get("projection_sha256"),
            "seal.pool_metadata.projection_sha256",
        )
        total_use_count = pool.get("total_use_count")
        if (
            isinstance(total_use_count, bool)
            or not isinstance(total_use_count, int)
            or total_use_count < 0
        ):
            raise ValueError("v3 join seal pool total_use_count is invalid")
    return seal


def resolve_decoder_config_path(
    decoder_model: str, decoder_revision: str
) -> Path:
    """Resolve the exact decoder config bound by the compact contract."""

    local = Path(decoder_model).expanduser()
    if local.is_dir():
        config = local / "config.json"
        if not config.is_file():
            raise FileNotFoundError(f"decoder directory has no config.json: {local}")
        return config.resolve()
    if not decoder_revision:
        raise ValueError("an immutable decoder revision is required")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=decoder_model,
            filename="config.json",
            revision=decoder_revision,
        )
    ).resolve()


def _canonical_sha256(value: Any, field: str) -> str:
    result = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _canonical_extractor_routes(
    value: Any,
) -> tuple[tuple[str, str], ...]:
    """Freeze the v2 route map as sorted, canonical JSON payloads."""

    if value in (None, "", (), []):
        return ()
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        raise ValueError("extractor_routes must be a route-name mapping")

    normalized: list[tuple[str, str]] = []
    route_atoms: set[str] = set()
    for raw_name, raw_payload in items:
        name = str(raw_name or "").strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"invalid extractor route name: {name!r}")
        if isinstance(raw_payload, str):
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError as error:
                raise ValueError(f"{name}: invalid extractor-route JSON") from error
        else:
            payload = raw_payload
        if not isinstance(payload, Mapping):
            raise ValueError(f"{name}: extractor route must be an object")
        if set(payload) != _EXTRACTOR_ROUTE_FIELDS:
            missing = sorted(_EXTRACTOR_ROUTE_FIELDS - set(payload))
            extra = sorted(set(payload) - _EXTRACTOR_ROUTE_FIELDS)
            raise ValueError(
                f"{name}: extractor route field mismatch; missing={missing}, extra={extra}"
            )
        canonical = dict(payload)
        for field in (
            "cfg_extractor_sha256",
            "dfg_extractor_sha256",
            "graph_extractor_sha256",
        ):
            canonical[field] = _canonical_sha256(
                payload.get(field), f"extractor_routes.{name}.{field}"
            )
        if not isinstance(payload.get("allow_call_edges"), bool):
            raise ValueError(f"{name}: allow_call_edges must be boolean")
        atom = str(payload.get("route_atom") or "").strip()
        if not re.fullmatch(r"<DX[0-9]+>", atom):
            raise ValueError(f"{name}: invalid extractor route atom {atom!r}")
        if atom in route_atoms:
            raise ValueError(f"duplicate extractor route atom: {atom}")
        route_atoms.add(atom)
        canonical["route_atom"] = atom
        for field in ("combined_hash_algorithm", "dfg_metadata"):
            canonical[field] = str(payload.get(field) or "").strip()
            if not canonical[field]:
                raise ValueError(f"{name}: {field} may not be empty")
        normalized.append(
            (
                name,
                json.dumps(
                    canonical,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    names = [name for name, _ in normalized]
    if len(names) != len(set(names)):
        raise ValueError("duplicate extractor route name")
    return tuple(sorted(normalized))


def _canonical_stream_marker_ids(value: Any) -> tuple[tuple[str, int], ...]:
    """Validate and freeze the exact v3 structural-marker token IDs."""

    if not isinstance(value, Mapping):
        raise ValueError("stream_marker_ids must be a marker-to-token-ID mapping")
    observed = {str(marker): token_id for marker, token_id in value.items()}
    expected = set(V3_STREAM_MARKERS)
    if set(observed) != expected:
        raise ValueError(
            "stream_marker_ids field mismatch; "
            f"missing={sorted(expected - set(observed))}, "
            f"extra={sorted(set(observed) - expected)}"
        )
    normalized: list[tuple[str, int]] = []
    token_ids: list[int] = []
    for marker in V3_STREAM_MARKERS:
        token_id = observed[marker]
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"stream_marker_ids.{marker} must be a non-negative integer")
        normalized.append((marker, token_id))
        token_ids.append(token_id)
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("stream_marker_ids must assign distinct token IDs")
    return tuple(normalized)


def _canonical_composite_representation(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("composite_representation must be an object")
    observed = dict(value)
    if observed != COMPOSITE_REPRESENTATION_V1:
        raise ValueError(
            "composite_representation must exactly bind the recursive v3 domain"
        )
    return json.dumps(
        observed, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _canonical_nested_nonliteral_descriptors(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("nested_nonliteral_descriptors must be an object")
    observed = dict(value)
    if observed != NESTED_NONLITERAL_DESCRIPTORS_V1:
        raise ValueError(
            "nested_nonliteral_descriptors must exactly bind the finite "
            "source-blind descriptor domain"
        )
    return json.dumps(
        observed, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _canonical_pool_positional_encoding(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("pool_positional_encoding must be an object")
    observed = dict(value)
    if observed != POOL_POSITIONAL_ENCODING_V2:
        raise ValueError(
            "pool_positional_encoding must exactly bind the v3 positional grammar"
        )
    return json.dumps(
        observed, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def validate_v3_pool_alignment_metadata(
    alignment: Mapping[str, Any], identity: str
) -> dict[str, Any]:
    """Validate the audit-only binary-pool receipt projection for one v3 row.

    Literal payloads remain exclusively in ``compact_input_ids``.  This
    sidecar projection carries only integrity hashes, a count, and the two
    invariants needed to prove that the extractor consumed the target AOT
    function rather than the supervised Dart source.
    """

    raw = alignment.get("pool_metadata")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{identity}: v3 alignment has no pool_metadata object")
    if set(raw) != _POOL_ALIGNMENT_FIELDS:
        raise ValueError(
            f"{identity}: pool_metadata field mismatch; "
            f"missing={sorted(_POOL_ALIGNMENT_FIELDS - set(raw))}, "
            f"extra={sorted(set(raw) - _POOL_ALIGNMENT_FIELDS)}"
        )
    schema = str(raw.get("schema") or "").strip()
    if schema != POOL_ALIGNMENT_SCHEMA_V1:
        raise ValueError(
            f"{identity}: unsupported pool_metadata schema {schema!r}; "
            f"expected {POOL_ALIGNMENT_SCHEMA_V1!r}"
        )
    receipt_sha256 = _canonical_sha256(
        raw.get("receipt_sha256"), f"{identity}.pool_metadata.receipt_sha256"
    )
    projection_sha256 = _canonical_sha256(
        raw.get("projection_sha256"),
        f"{identity}.pool_metadata.projection_sha256",
    )
    use_count = raw.get("use_count")
    if isinstance(use_count, bool) or not isinstance(use_count, int) or use_count < 0:
        raise ValueError(
            f"{identity}: pool_metadata.use_count must be a non-negative integer"
        )
    if raw.get("source_blind") is not True:
        raise ValueError(f"{identity}: pool_metadata.source_blind must be true")
    target_function = str(raw.get("target_function") or "").strip()
    if target_function != "candidate":
        raise ValueError(
            f"{identity}: pool_metadata.target_function must be 'candidate'"
        )
    return {
        "schema": schema,
        "receipt_sha256": receipt_sha256,
        "projection_sha256": projection_sha256,
        "use_count": use_count,
        "source_blind": True,
        "target_function": target_function,
    }


def _strict_json_value(text: str, identity: str) -> Any:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{identity}: duplicate pool JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{identity}: non-finite pool JSON number {value!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{identity}: malformed v3 pool JSON payload") from error


def _canonical_v3_pool_payload(
    kind: Any,
    payload: Any,
    label: str,
    *,
    depth: int = 0,
    nodes_seen: list[int] | None = None,
    ancestors: frozenset[int] = frozenset(),
    nested: bool = False,
) -> dict[str, Any]:
    payload_fields = {
        "string": {"code_units"},
        "int": {"decimal"},
        "double": {"bits_hex"},
        "null": set(),
        "bool": {"value"},
        "composite": {
            "complete",
            "composite_type",
            "elements",
            "omitted_edge_counts",
        },
    }
    if kind == "nonliteral":
        if not nested:
            raise ValueError(f"{label} may not be a top-level nonliteral record")
        if not isinstance(payload, Mapping) or set(payload) != {
            "nonliteral_kind",
            "profile_type",
        }:
            raise ValueError(f"{label} has invalid nonliteral payload fields")
        nonliteral_kind = payload.get("nonliteral_kind")
        profile_type = payload.get("profile_type")
        if not isinstance(nonliteral_kind, str) or not isinstance(profile_type, str):
            raise ValueError(f"{label} nonliteral descriptors must be strings")
        expected_kind = NESTED_NONLITERAL_PROFILE_KIND_V1.get(profile_type)
        if expected_kind is None or nonliteral_kind != expected_kind:
            raise ValueError(f"{label} has an unsupported nonliteral descriptor pair")
        if nodes_seen is None:
            nodes_seen = [0]
        nodes_seen[0] += 1
        if nodes_seen[0] > MAX_COMPOSITE_NODES_V1:
            raise ValueError(f"{label} exceeds the composite node limit")
        if depth > MAX_COMPOSITE_DEPTH_V1:
            raise ValueError(f"{label} exceeds the composite depth limit")
        return {
            "nonliteral_kind": nonliteral_kind,
            "profile_type": profile_type,
        }
    if not isinstance(kind, str) or kind not in payload_fields:
        raise ValueError(f"{label} has unsupported literal kind {kind!r}")
    if not isinstance(payload, Mapping) or set(payload) != payload_fields[kind]:
        raise ValueError(f"{label} has invalid {kind} payload fields")
    if nodes_seen is None:
        nodes_seen = [0]
    nodes_seen[0] += 1
    if nodes_seen[0] > MAX_COMPOSITE_NODES_V1:
        raise ValueError(f"{label} exceeds the composite node limit")
    if depth > MAX_COMPOSITE_DEPTH_V1:
        raise ValueError(f"{label} exceeds the composite depth limit")

    if kind == "string":
        code_units = payload.get("code_units")
        if not isinstance(code_units, list) or any(
            isinstance(unit, bool)
            or not isinstance(unit, int)
            or not 0 <= unit <= 0xFFFF
            for unit in code_units
        ):
            raise ValueError(f"{label} has invalid UTF-16 code units")
        return {"code_units": list(code_units)}
    if kind == "int":
        decimal = payload.get("decimal")
        if not isinstance(decimal, str) or not re.fullmatch(
            r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)", decimal
        ):
            raise ValueError(f"{label} has non-canonical integer text")
        return {"decimal": decimal}
    if kind == "double":
        bits_hex = payload.get("bits_hex")
        if not isinstance(bits_hex, str) or not re.fullmatch(r"[0-9a-f]{16}", bits_hex):
            raise ValueError(f"{label} has non-canonical binary64 bits")
        return {"bits_hex": bits_hex}
    if kind == "null":
        return {}
    if kind == "bool":
        boolean = payload.get("value")
        if not isinstance(boolean, bool):
            raise ValueError(f"{label} has invalid bool payload")
        return {"value": boolean}

    payload_identity = id(payload)
    if payload_identity in ancestors:
        raise ValueError(f"{label} contains a cyclic composite")
    if payload.get("complete") is not True:
        raise ValueError(f"{label} composite must be complete")
    composite_type = payload.get("composite_type")
    if composite_type not in {"array_storage", "map_storage"}:
        raise ValueError(f"{label} has unsupported composite_type {composite_type!r}")
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError(f"{label} composite elements must be a list")
    next_ancestors = ancestors | {payload_identity}
    canonical_elements: list[dict[str, Any]] = []
    for element_index, element in enumerate(elements):
        element_label = f"{label} composite element {element_index}"
        if not isinstance(element, Mapping) or set(element) != {"index", "value"}:
            raise ValueError(f"{element_label} has invalid fields")
        index = element.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{element_label} index must be a non-negative integer")
        nested = element.get("value")
        if not isinstance(nested, Mapping) or set(nested) != {"kind", "payload"}:
            raise ValueError(f"{element_label} value has invalid fields")
        nested_kind = nested.get("kind")
        nested_payload = _canonical_v3_pool_payload(
            nested_kind,
            nested.get("payload"),
            element_label,
            depth=depth + 1,
            nodes_seen=nodes_seen,
            ancestors=next_ancestors,
            nested=True,
        )
        canonical_elements.append(
            {
                "index": index,
                "value": {"kind": nested_kind, "payload": nested_payload},
            }
        )
    omitted = payload.get("omitted_edge_counts")
    if not isinstance(omitted, Mapping):
        raise ValueError(f"{label} omitted_edge_counts must be an object")
    canonical_omitted: dict[str, int] = {}
    for edge_type, count in omitted.items():
        if (
            not isinstance(edge_type, str)
            or not edge_type
            or any(character in edge_type for character in "<>&")
        ):
            raise ValueError(f"{label} has an invalid omitted-edge type")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label} omitted-edge count must be non-negative")
        canonical_omitted[edge_type] = count
    return {
        "complete": True,
        "composite_type": composite_type,
        "elements": canonical_elements,
        "omitted_edge_counts": dict(sorted(canonical_omitted.items())),
    }


def _canonical_v3_pool_envelope(value: Any, identity: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "target_function",
        "uses",
    }:
        raise ValueError(f"{identity}: invalid v3 pool envelope fields")
    if value.get("schema") != POOL_PAYLOAD_SCHEMA_V1:
        raise ValueError(f"{identity}: v3 pool envelope schema mismatch")
    if value.get("target_function") != "candidate":
        raise ValueError(f"{identity}: v3 pool target_function must be 'candidate'")
    uses = value.get("uses")
    if not isinstance(uses, list):
        raise ValueError(f"{identity}: v3 pool uses must be a list")
    canonical_uses: list[dict[str, Any]] = []
    for record_index, record in enumerate(uses):
        label = f"{identity}: v3 pool use {record_index}"
        if not isinstance(record, Mapping) or set(record) != {
            "pp_offset",
            "kind",
            "payload",
            "use_sites",
        }:
            raise ValueError(f"{label} has invalid fields")
        pp_offset = record.get("pp_offset")
        if isinstance(pp_offset, bool) or not isinstance(pp_offset, int):
            raise ValueError(f"{label} pp_offset must be an integer")
        kind = record.get("kind")
        canonical_payload = _canonical_v3_pool_payload(
            kind, record.get("payload"), label
        )
        use_sites = record.get("use_sites")
        if not isinstance(use_sites, list) or not use_sites:
            raise ValueError(f"{label} use_sites must be a non-empty list")
        canonical_sites: list[dict[str, int]] = []
        for site_index, site in enumerate(use_sites):
            if not isinstance(site, Mapping) or set(site) != {
                "block",
                "instruction",
            }:
                raise ValueError(f"{label} use site {site_index} has invalid fields")
            block = site.get("block")
            instruction = site.get("instruction")
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (block, instruction)
            ):
                raise ValueError(f"{label} use site {site_index} is invalid")
            canonical_sites.append({"block": block, "instruction": instruction})
        canonical_uses.append(
            {
                "pp_offset": pp_offset,
                "kind": kind,
                "payload": canonical_payload,
                "use_sites": canonical_sites,
            }
        )
    return {
        "schema": POOL_PAYLOAD_SCHEMA_V1,
        "target_function": "candidate",
        "uses": canonical_uses,
    }


def _pool_plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _pool_utf16_units_to_text(units: Sequence[int]) -> str:
    raw = bytearray()
    for unit in units:
        raw.extend((unit & 0xFF, unit >> 8))
    return bytes(raw).decode("utf-16-le", errors="surrogatepass")


def _pool_text_to_utf16_units(value: str) -> list[int]:
    raw = value.encode("utf-16-le", errors="surrogatepass")
    return [raw[index] | (raw[index + 1] << 8) for index in range(0, len(raw), 2)]


def _encode_v3_pool_value(kind: str, payload: Mapping[str, Any]) -> list[Any]:
    tag = _POOL_KIND_TO_TAG_V2[kind]
    if kind == "string":
        return [tag, _pool_utf16_units_to_text(payload["code_units"])]
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
        return [tag, _NONLITERAL_PAIR_TO_TAG_V2[pair]]
    elements: list[Any] = []
    previous_index = 0
    for element in payload["elements"]:
        index = element["index"]
        nested = element["value"]
        elements.extend(
            [
                index - previous_index,
                _encode_v3_pool_value(nested["kind"], nested["payload"]),
            ]
        )
        previous_index = index
    omitted: list[Any] = []
    for edge_type, count in sorted(payload["omitted_edge_counts"].items()):
        omitted.extend([edge_type, count])
    return [
        tag,
        _COMPOSITE_TYPE_TO_TAG_V2[payload["composite_type"]],
        elements,
        omitted,
    ]


def _decode_v3_pool_value(
    value: Any, identity: str, *, nested: bool
) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{identity}: positional value must be a non-empty array")
    tag = _pool_plain_int(value[0], f"{identity}: positional kind tag")
    kind = _POOL_TAG_TO_KIND_V2.get(tag)
    if kind is None or (kind == "nonliteral" and not nested):
        raise ValueError(f"{identity}: invalid positional kind tag {tag}")
    if kind == "string":
        if len(value) != 2 or not isinstance(value[1], str):
            raise ValueError(f"{identity}: invalid positional string shape")
        return kind, {"code_units": _pool_text_to_utf16_units(value[1])}
    if kind == "int":
        if len(value) != 2:
            raise ValueError(f"{identity}: invalid positional int shape")
        return kind, {"decimal": value[1]}
    if kind == "double":
        if len(value) != 2:
            raise ValueError(f"{identity}: invalid positional double shape")
        return kind, {"bits_hex": value[1]}
    if kind == "null":
        if len(value) != 1:
            raise ValueError(f"{identity}: invalid positional null shape")
        return kind, {}
    if kind == "bool":
        if len(value) != 2:
            raise ValueError(f"{identity}: invalid positional bool shape")
        return kind, {"value": value[1]}
    if kind == "nonliteral":
        if len(value) != 2:
            raise ValueError(f"{identity}: invalid positional nonliteral shape")
        pair_tag = _pool_plain_int(
            value[1], f"{identity}: positional nonliteral tag"
        )
        if not 0 <= pair_tag < len(_NONLITERAL_PAIRS_V2):
            raise ValueError(f"{identity}: positional nonliteral tag out of range")
        profile_type, nonliteral_kind = _NONLITERAL_PAIRS_V2[pair_tag]
        return kind, {
            "nonliteral_kind": nonliteral_kind,
            "profile_type": profile_type,
        }
    if len(value) != 4:
        raise ValueError(f"{identity}: invalid positional composite shape")
    type_tag = _pool_plain_int(
        value[1], f"{identity}: positional composite type tag"
    )
    composite_type = _COMPOSITE_TAG_TO_TYPE_V2.get(type_tag)
    if composite_type is None:
        raise ValueError(f"{identity}: invalid positional composite type tag")
    raw_elements = value[2]
    if not isinstance(raw_elements, list) or len(raw_elements) % 2:
        raise ValueError(f"{identity}: invalid positional composite elements")
    elements: list[dict[str, Any]] = []
    previous_index = 0
    for position in range(0, len(raw_elements), 2):
        index = previous_index + _pool_plain_int(
            raw_elements[position], f"{identity}: composite index delta"
        )
        if index < 0:
            raise ValueError(f"{identity}: positional composite index is negative")
        nested_kind, nested_payload = _decode_v3_pool_value(
            raw_elements[position + 1], identity, nested=True
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
        raise ValueError(f"{identity}: invalid positional omitted-edge list")
    omitted: dict[str, int] = {}
    for position in range(0, len(raw_omitted), 2):
        edge_type = raw_omitted[position]
        if not isinstance(edge_type, str) or edge_type in omitted:
            raise ValueError(f"{identity}: invalid positional omitted-edge key")
        omitted[edge_type] = _pool_plain_int(
            raw_omitted[position + 1], f"{identity}: omitted-edge count"
        )
    return kind, {
        "complete": True,
        "composite_type": composite_type,
        "elements": elements,
        "omitted_edge_counts": omitted,
    }


def _encode_v3_pool_positional(canonical: Mapping[str, Any]) -> list[list[Any]]:
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
                *_encode_v3_pool_value(record["kind"], record["payload"]),
                sites,
            ]
        )
        previous_pp = record["pp_offset"]
    return records


def _decode_v3_pool_positional(value: Any, identity: str) -> dict[str, Any]:
    if not isinstance(value, list):
        raise ValueError(f"{identity}: positional pool root must be an array")
    uses: list[dict[str, Any]] = []
    previous_pp = 0
    for record_index, raw_record in enumerate(value):
        label = f"{identity}: positional pool record {record_index}"
        if not isinstance(raw_record, list) or len(raw_record) < 3:
            raise ValueError(f"{label} has invalid shape")
        pp_delta = _pool_plain_int(raw_record[0], f"{label} pp delta")
        tag = _pool_plain_int(raw_record[1], f"{label} kind tag")
        kind = _POOL_TAG_TO_KIND_V2.get(tag)
        if kind is None or kind == "nonliteral":
            raise ValueError(f"{label} has invalid top-level kind tag")
        value_length = 4 if kind == "composite" else (1 if kind == "null" else 2)
        if len(raw_record) != 2 + (value_length - 1) + 1:
            raise ValueError(f"{label} has invalid length")
        decoded_kind, payload = _decode_v3_pool_value(
            [tag, *raw_record[2:-1]], label, nested=False
        )
        raw_sites = raw_record[-1]
        if not isinstance(raw_sites, list) or not raw_sites or len(raw_sites) % 2:
            raise ValueError(f"{label} has invalid site list")
        sites: list[dict[str, int]] = []
        previous_block = 0
        previous_instruction = 0
        for position in range(0, len(raw_sites), 2):
            block_delta = _pool_plain_int(
                raw_sites[position], f"{label} block delta"
            )
            instruction_delta = _pool_plain_int(
                raw_sites[position + 1], f"{label} instruction delta"
            )
            block = previous_block + block_delta
            instruction = (
                previous_instruction + instruction_delta
                if block_delta == 0
                else instruction_delta
            )
            if block < 0 or instruction < 0:
                raise ValueError(f"{label} reconstructs a negative site")
            sites.append({"block": block, "instruction": instruction})
            previous_block = block
            previous_instruction = instruction
        pp_offset = previous_pp + pp_delta
        uses.append(
            {
                "pp_offset": pp_offset,
                "kind": decoded_kind,
                "payload": payload,
                "use_sites": sites,
            }
        )
        previous_pp = pp_offset
    return {
        "schema": POOL_PAYLOAD_SCHEMA_V1,
        "target_function": "candidate",
        "uses": uses,
    }


def canonical_v3_pool_json(value: Any) -> str:
    """Contract-bound positional ASCII JSON between <PX0>/<PEND>."""

    canonical = _canonical_v3_pool_envelope(value, "pool JSON")
    encoded = json.dumps(
        _encode_v3_pool_positional(canonical),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    # Prevent a future tag scanner or audit renderer from interpreting a
    # literal '<...>' string as compact control syntax.
    return (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the effective vocabulary and special-token contract.

    This deliberately avoids paths and tokenizer class implementation details,
    so the codec builder and trainer can reproduce the fingerprint on different
    machines.  Token-to-ID assignments remain fully covered.
    """

    if not hasattr(tokenizer, "get_vocab"):
        raise TypeError("tokenizer must expose get_vocab()")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping):
        raise TypeError("tokenizer.get_vocab() must return a mapping")
    normalized_vocab = sorted(
        ((str(token), int(token_id)) for token, token_id in vocab.items()),
        key=lambda item: (item[1], item[0]),
    )
    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    special_map = json_safe(getattr(tokenizer, "special_tokens_map", {}) or {})
    payload = {
        "vocab": normalized_vocab,
        "special_tokens_map": special_map,
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DirectCompactContract:
    codec_sha256: str
    codebook_sha256: str
    tokenizer_json_sha256: str
    tokenizer_fingerprint_sha256: str
    model_config_sha256: str
    decoder_model: str
    decoder_revision: str
    target_function: str
    target_language: str
    dfg_extractor_sha256: str | None
    lossless_domain: str
    max_source_tokens: int = 9000
    max_target_tokens: int = 3072
    max_total_tokens: int = 12288
    base_vocab_size: int | None = None
    source_token_ids: tuple[int, ...] = ()
    source_token_expansions: tuple[tuple[int, tuple[int, ...]], ...] = ()
    source_embedding_init: str = "codebook_mean"
    extractor_routes: tuple[tuple[str, str], ...] | Mapping[str, Any] = ()
    runtime_symbol_policy_sha256: str | None = None
    pool_extractor_sha256: str | None = None
    dart_toolchain_manifest_sha256: str | None = None
    aot_manifest_sha256: str | None = None
    pool_reconciliation_manifest_sha256: str | None = None
    graph_codec_dependency_sha256: str | None = None
    target_architecture: str | None = None
    pool_schema: str | None = None
    pool_encoding: str | None = None
    pool_positional_encoding: str | Mapping[str, Any] | None = None
    pool_scope: str | None = None
    pool_projection: str | None = None
    all_encoded_pool_uses_reference_canonical_graph_instructions: bool | None = None
    raw_disassembly_unreachable_islands_in_lossless_domain: bool | None = None
    non_graph_aot_xrefs: str | None = None
    graph_retained_literal_use_omission_policy: str | None = None
    pool_order_and_duplicates_preserved: bool | None = None
    string_representation: str | None = None
    integer_representation: str | None = None
    double_representation: str | None = None
    composite_representation: str | Mapping[str, Any] | None = None
    nested_nonliteral_descriptors: str | Mapping[str, Any] | None = None
    stream_marker_ids: tuple[tuple[str, int], ...] | Mapping[str, int] = ()
    schema: str = CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in SUPPORTED_CONTRACT_SCHEMAS:
            raise ValueError(
                f"unsupported compact contract schema {self.schema!r}; "
                f"expected one of {sorted(SUPPORTED_CONTRACT_SCHEMAS)!r}"
            )
        object.__setattr__(
            self, "codec_sha256", _canonical_sha256(self.codec_sha256, "codec_sha256")
        )
        object.__setattr__(
            self,
            "codebook_sha256",
            _canonical_sha256(self.codebook_sha256, "codebook_sha256"),
        )
        object.__setattr__(
            self, "tokenizer_json_sha256",
            _canonical_sha256(self.tokenizer_json_sha256, "tokenizer_json_sha256"),
        )
        object.__setattr__(
            self, "tokenizer_fingerprint_sha256",
            _canonical_sha256(
                self.tokenizer_fingerprint_sha256, "tokenizer_fingerprint_sha256"
            ),
        )
        object.__setattr__(
            self, "model_config_sha256",
            _canonical_sha256(self.model_config_sha256, "model_config_sha256"),
        )
        object.__setattr__(self, "decoder_model", str(self.decoder_model or "").strip())
        object.__setattr__(self, "decoder_revision", str(self.decoder_revision or "").strip())
        if not self.decoder_model or not self.decoder_revision:
            raise ValueError("decoder_model and immutable decoder_revision are required")
        object.__setattr__(self, "target_function", str(self.target_function or "").strip())
        if not re.fullmatch(r"(?:candidate|fn\d+)", self.target_function):
            raise ValueError("target_function must be a neutral name such as candidate or fn0")
        object.__setattr__(self, "target_language", str(self.target_language or "").strip())
        if self.target_language != "Dart":
            raise ValueError("target_language must be the canonical spelling 'Dart'")
        routes = _canonical_extractor_routes(self.extractor_routes)
        object.__setattr__(self, "extractor_routes", routes)
        if self.schema == CONTRACT_SCHEMA_V1:
            object.__setattr__(
                self, "dfg_extractor_sha256",
                _canonical_sha256(self.dfg_extractor_sha256, "dfg_extractor_sha256"),
            )
            if routes:
                raise ValueError("v1 contracts may not carry extractor_routes")
            if self.runtime_symbol_policy_sha256 not in (None, ""):
                raise ValueError(
                    "v1 contracts may not carry runtime_symbol_policy_sha256"
                )
            object.__setattr__(self, "runtime_symbol_policy_sha256", None)
        elif self.schema in (CONTRACT_SCHEMA_V2, CONTRACT_SCHEMA_V3):
            if self.dfg_extractor_sha256 not in (None, ""):
                raise ValueError(
                    f"{self.schema} contracts use extractor_routes, not "
                    "dfg_extractor_sha256"
                )
            object.__setattr__(self, "dfg_extractor_sha256", None)
            if len(routes) < 2:
                raise ValueError(
                    f"{self.schema} contracts must bind both extractor routes"
                )
            object.__setattr__(
                self,
                "runtime_symbol_policy_sha256",
                _canonical_sha256(
                    self.runtime_symbol_policy_sha256,
                    "runtime_symbol_policy_sha256",
                ),
            )
        else:  # defensive: SUPPORTED_CONTRACT_SCHEMAS is checked above
            raise AssertionError(f"unhandled compact contract schema {self.schema!r}")

        if self.schema == CONTRACT_SCHEMA_V3:
            if self.target_function != "candidate":
                raise ValueError("v3 contracts require target_function='candidate'")
            object.__setattr__(
                self,
                "pool_extractor_sha256",
                _canonical_sha256(
                    self.pool_extractor_sha256, "pool_extractor_sha256"
                ),
            )
            object.__setattr__(
                self,
                "dart_toolchain_manifest_sha256",
                _canonical_sha256(
                    self.dart_toolchain_manifest_sha256,
                    "dart_toolchain_manifest_sha256",
                ),
            )
            object.__setattr__(
                self,
                "aot_manifest_sha256",
                _canonical_sha256(
                    self.aot_manifest_sha256, "aot_manifest_sha256"
                ),
            )
            object.__setattr__(
                self,
                "pool_reconciliation_manifest_sha256",
                _canonical_sha256(
                    self.pool_reconciliation_manifest_sha256,
                    "pool_reconciliation_manifest_sha256",
                ),
            )
            object.__setattr__(
                self,
                "graph_codec_dependency_sha256",
                _canonical_sha256(
                    self.graph_codec_dependency_sha256,
                    "graph_codec_dependency_sha256",
                ),
            )
            expected_v3_scalars = {
                "target_architecture": TARGET_ARCHITECTURE_V3,
                "pool_schema": POOL_PAYLOAD_SCHEMA_V1,
                "pool_encoding": POOL_ENCODING_V1,
                "pool_scope": POOL_SCOPE_V1,
                "pool_projection": POOL_PROJECTION_V1,
                "non_graph_aot_xrefs": NON_GRAPH_AOT_XREF_POLICY_V1,
                "graph_retained_literal_use_omission_policy": (
                    GRAPH_LITERAL_OMISSION_POLICY_V1
                ),
                "string_representation": STRING_REPRESENTATION_V1,
                "integer_representation": INTEGER_REPRESENTATION_V1,
                "double_representation": DOUBLE_REPRESENTATION_V1,
            }
            for field, expected in expected_v3_scalars.items():
                observed = str(getattr(self, field) or "").strip()
                if observed != expected:
                    raise ValueError(f"{field} must be {expected!r}")
                object.__setattr__(self, field, observed)
            if (
                self.all_encoded_pool_uses_reference_canonical_graph_instructions
                is not True
            ):
                raise ValueError(
                    "all_encoded_pool_uses_reference_canonical_graph_instructions "
                    "must be true"
                )
            object.__setattr__(
                self,
                "all_encoded_pool_uses_reference_canonical_graph_instructions",
                True,
            )
            if (
                self.raw_disassembly_unreachable_islands_in_lossless_domain
                is not False
            ):
                raise ValueError(
                    "raw_disassembly_unreachable_islands_in_lossless_domain "
                    "must be false"
                )
            object.__setattr__(
                self,
                "raw_disassembly_unreachable_islands_in_lossless_domain",
                False,
            )
            if self.pool_order_and_duplicates_preserved is not True:
                raise ValueError(
                    "pool_order_and_duplicates_preserved must be true"
                )
            object.__setattr__(self, "pool_order_and_duplicates_preserved", True)
            object.__setattr__(
                self,
                "composite_representation",
                _canonical_composite_representation(
                    self.composite_representation
                ),
            )
            object.__setattr__(
                self,
                "nested_nonliteral_descriptors",
                _canonical_nested_nonliteral_descriptors(
                    self.nested_nonliteral_descriptors
                ),
            )
            object.__setattr__(
                self,
                "pool_positional_encoding",
                _canonical_pool_positional_encoding(
                    self.pool_positional_encoding
                ),
            )
            object.__setattr__(
                self,
                "stream_marker_ids",
                _canonical_stream_marker_ids(self.stream_marker_ids),
            )
        else:
            forbidden_pool_fields = {
                "pool_extractor_sha256": self.pool_extractor_sha256,
                "dart_toolchain_manifest_sha256": self.dart_toolchain_manifest_sha256,
                "aot_manifest_sha256": self.aot_manifest_sha256,
                "pool_reconciliation_manifest_sha256": (
                    self.pool_reconciliation_manifest_sha256
                ),
                "graph_codec_dependency_sha256": self.graph_codec_dependency_sha256,
                "target_architecture": self.target_architecture,
                "pool_schema": self.pool_schema,
                "pool_encoding": self.pool_encoding,
                "pool_positional_encoding": self.pool_positional_encoding,
                "pool_scope": self.pool_scope,
                "pool_projection": self.pool_projection,
                "all_encoded_pool_uses_reference_canonical_graph_instructions": (
                    self.all_encoded_pool_uses_reference_canonical_graph_instructions
                ),
                "raw_disassembly_unreachable_islands_in_lossless_domain": (
                    self.raw_disassembly_unreachable_islands_in_lossless_domain
                ),
                "non_graph_aot_xrefs": self.non_graph_aot_xrefs,
                "graph_retained_literal_use_omission_policy": (
                    self.graph_retained_literal_use_omission_policy
                ),
                "pool_order_and_duplicates_preserved": (
                    self.pool_order_and_duplicates_preserved
                ),
                "string_representation": self.string_representation,
                "integer_representation": self.integer_representation,
                "double_representation": self.double_representation,
                "composite_representation": self.composite_representation,
                "nested_nonliteral_descriptors": (
                    self.nested_nonliteral_descriptors
                ),
                "stream_marker_ids": self.stream_marker_ids,
            }
            populated = sorted(
                field
                for field, value in forbidden_pool_fields.items()
                if value not in (None, "", (), {})
            )
            if populated:
                raise ValueError(
                    f"{self.schema} contracts may not carry v3 pool fields: {populated}"
                )
            object.__setattr__(self, "pool_extractor_sha256", None)
            object.__setattr__(self, "dart_toolchain_manifest_sha256", None)
            object.__setattr__(self, "aot_manifest_sha256", None)
            object.__setattr__(self, "pool_reconciliation_manifest_sha256", None)
            object.__setattr__(self, "graph_codec_dependency_sha256", None)
            object.__setattr__(self, "target_architecture", None)
            object.__setattr__(self, "pool_schema", None)
            object.__setattr__(self, "pool_encoding", None)
            object.__setattr__(self, "pool_positional_encoding", None)
            object.__setattr__(self, "pool_scope", None)
            object.__setattr__(self, "pool_projection", None)
            object.__setattr__(
                self,
                "all_encoded_pool_uses_reference_canonical_graph_instructions",
                None,
            )
            object.__setattr__(
                self,
                "raw_disassembly_unreachable_islands_in_lossless_domain",
                None,
            )
            object.__setattr__(self, "non_graph_aot_xrefs", None)
            object.__setattr__(
                self, "graph_retained_literal_use_omission_policy", None
            )
            object.__setattr__(self, "pool_order_and_duplicates_preserved", None)
            object.__setattr__(self, "string_representation", None)
            object.__setattr__(self, "integer_representation", None)
            object.__setattr__(self, "double_representation", None)
            object.__setattr__(self, "composite_representation", None)
            object.__setattr__(self, "nested_nonliteral_descriptors", None)
            object.__setattr__(self, "stream_marker_ids", ())
        object.__setattr__(self, "lossless_domain", str(self.lossless_domain or "").strip())
        expected_lossless_domain = {
            CONTRACT_SCHEMA_V1: "scrubbed_canonical_graph",
            CONTRACT_SCHEMA_V2: "scrubbed_canonical_graph_v2",
            CONTRACT_SCHEMA_V3: LOSSLESS_DOMAIN_V3,
        }[self.schema]
        if self.lossless_domain != expected_lossless_domain:
            raise ValueError(
                f"lossless_domain must be {expected_lossless_domain!r} for {self.schema}"
            )
        for field in ("max_source_tokens", "max_target_tokens", "max_total_tokens"):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.max_source_tokens > 9000:
            raise ValueError("confirmatory compact input may not exceed 9000 tokens")
        if self.max_total_tokens < self.max_source_tokens:
            raise ValueError("max_total_tokens must fit max_source_tokens")
        if self.source_embedding_init != "codebook_mean":
            raise ValueError("source_embedding_init must be 'codebook_mean'")
        token_ids = tuple(int(value) for value in self.source_token_ids)
        if len(token_ids) != len(set(token_ids)) or any(value < 0 for value in token_ids):
            raise ValueError("source_token_ids must be unique non-negative integers")
        object.__setattr__(self, "source_token_ids", token_ids)
        if token_ids and self.base_vocab_size is None:
            raise ValueError("base_vocab_size is required when source_token_ids are appended")
        if self.base_vocab_size is not None and int(self.base_vocab_size) <= 0:
            raise ValueError("base_vocab_size must be positive")
        expansions = tuple(
            sorted(
                (
                    int(source_id),
                    tuple(int(base_id) for base_id in base_ids),
                )
                for source_id, base_ids in self.source_token_expansions
            )
        )
        if token_ids and {source_id for source_id, _ in expansions} != set(token_ids):
            raise ValueError(
                "source_token_expansions must contain exactly one expansion for each source token"
            )
        for source_id, base_ids in expansions:
            if source_id < int(self.base_vocab_size or 0):
                raise ValueError("source token IDs must be outside the frozen base vocabulary")
            if not base_ids:
                raise ValueError("every source token requires a non-empty base-token expansion")
            if any(base_id < 0 or base_id >= int(self.base_vocab_size or 0) for base_id in base_ids):
                raise ValueError("source token expansion IDs must belong to the base vocabulary")
        object.__setattr__(self, "source_token_expansions", expansions)
        if self.schema == CONTRACT_SCHEMA_V3:
            marker_ids = dict(self.stream_marker_ids)
            if not token_ids:
                raise ValueError("v3 contracts require registered source_token_ids")
            unknown_markers = sorted(set(marker_ids.values()) - set(token_ids))
            if unknown_markers:
                raise ValueError(
                    "stream_marker_ids must be registered source tokens; "
                    f"unknown={unknown_markers}"
                )
            if self.base_vocab_size is None or any(
                token_id < self.base_vocab_size for token_id in marker_ids.values()
            ):
                raise ValueError(
                    "v3 stream markers must be outside the frozen base vocabulary"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DirectCompactContract":
        return cls(
            schema=str(value.get("schema") or ""),
            codec_sha256=value.get("codec_sha256"),
            codebook_sha256=value.get("codebook_sha256"),
            tokenizer_json_sha256=value.get("tokenizer_json_sha256"),
            tokenizer_fingerprint_sha256=value.get("tokenizer_fingerprint_sha256"),
            model_config_sha256=value.get("model_config_sha256"),
            decoder_model=value.get("decoder_model"),
            decoder_revision=value.get("decoder_revision"),
            target_function=value.get("target_function"),
            target_language=value.get("target_language"),
            dfg_extractor_sha256=value.get("dfg_extractor_sha256"),
            lossless_domain=value.get("lossless_domain"),
            max_source_tokens=int(value.get("max_source_tokens", 9000)),
            max_target_tokens=int(value.get("max_target_tokens", 3072)),
            max_total_tokens=int(value.get("max_total_tokens", 12288)),
            base_vocab_size=(
                None
                if value.get("base_vocab_size") is None
                else int(value["base_vocab_size"])
            ),
            source_token_ids=tuple(value.get("source_token_ids") or ()),
            source_token_expansions=tuple(
                (int(source_id), tuple(base_ids))
                for source_id, base_ids in (
                    value.get("source_token_expansions") or {}
                ).items()
            ),
            source_embedding_init=str(
                value.get("source_embedding_init") or "codebook_mean"
            ),
            extractor_routes=value.get("extractor_routes") or (),
            runtime_symbol_policy_sha256=value.get(
                "runtime_symbol_policy_sha256"
            ),
            pool_extractor_sha256=value.get("pool_extractor_sha256"),
            dart_toolchain_manifest_sha256=value.get(
                "dart_toolchain_manifest_sha256"
            ),
            aot_manifest_sha256=value.get("aot_manifest_sha256"),
            pool_reconciliation_manifest_sha256=value.get(
                "pool_reconciliation_manifest_sha256"
            ),
            graph_codec_dependency_sha256=value.get(
                "graph_codec_dependency_sha256"
            ),
            target_architecture=value.get("target_architecture"),
            pool_schema=value.get("pool_schema"),
            pool_encoding=value.get("pool_encoding"),
            pool_positional_encoding=value.get("pool_positional_encoding"),
            pool_scope=value.get("pool_scope"),
            pool_projection=value.get("pool_projection"),
            all_encoded_pool_uses_reference_canonical_graph_instructions=value.get(
                "all_encoded_pool_uses_reference_canonical_graph_instructions"
            ),
            raw_disassembly_unreachable_islands_in_lossless_domain=value.get(
                "raw_disassembly_unreachable_islands_in_lossless_domain"
            ),
            non_graph_aot_xrefs=value.get("non_graph_aot_xrefs"),
            graph_retained_literal_use_omission_policy=value.get(
                "graph_retained_literal_use_omission_policy"
            ),
            pool_order_and_duplicates_preserved=value.get(
                "pool_order_and_duplicates_preserved"
            ),
            string_representation=value.get("string_representation"),
            integer_representation=value.get("integer_representation"),
            double_representation=value.get("double_representation"),
            composite_representation=value.get("composite_representation"),
            nested_nonliteral_descriptors=value.get(
                "nested_nonliteral_descriptors"
            ),
            stream_marker_ids=value.get("stream_marker_ids") or (),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DirectCompactContract":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("compact contract must be a JSON object")
        return cls.from_mapping(value)

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "codec_sha256": self.codec_sha256,
            "codebook_sha256": self.codebook_sha256,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "tokenizer_fingerprint_sha256": self.tokenizer_fingerprint_sha256,
            "model_config_sha256": self.model_config_sha256,
            "decoder_model": self.decoder_model,
            "decoder_revision": self.decoder_revision,
            "target_function": self.target_function,
            "target_language": self.target_language,
            "lossless_domain": self.lossless_domain,
            "max_source_tokens": self.max_source_tokens,
            "max_target_tokens": self.max_target_tokens,
            "max_total_tokens": self.max_total_tokens,
            "base_vocab_size": self.base_vocab_size,
            "source_token_ids": list(self.source_token_ids),
            "source_token_expansions": {
                str(source_id): list(base_ids)
                for source_id, base_ids in self.source_token_expansions
            },
            "source_embedding_init": self.source_embedding_init,
        }
        if self.schema == CONTRACT_SCHEMA_V1:
            value["dfg_extractor_sha256"] = self.dfg_extractor_sha256
        else:
            value["extractor_routes"] = {
                name: json.loads(payload)
                for name, payload in self.extractor_routes
            }
            value["runtime_symbol_policy_sha256"] = (
                self.runtime_symbol_policy_sha256
            )
        if self.schema == CONTRACT_SCHEMA_V3:
            value.update(
                {
                    "pool_extractor_sha256": self.pool_extractor_sha256,
                    "dart_toolchain_manifest_sha256": (
                        self.dart_toolchain_manifest_sha256
                    ),
                    "aot_manifest_sha256": self.aot_manifest_sha256,
                    "pool_reconciliation_manifest_sha256": (
                        self.pool_reconciliation_manifest_sha256
                    ),
                    "graph_codec_dependency_sha256": (
                        self.graph_codec_dependency_sha256
                    ),
                    "target_architecture": self.target_architecture,
                    "pool_schema": self.pool_schema,
                    "pool_encoding": self.pool_encoding,
                    "pool_positional_encoding": json.loads(
                        str(self.pool_positional_encoding)
                    ),
                    "pool_scope": self.pool_scope,
                    "pool_projection": self.pool_projection,
                    "all_encoded_pool_uses_reference_canonical_graph_instructions": (
                        self.all_encoded_pool_uses_reference_canonical_graph_instructions
                    ),
                    "raw_disassembly_unreachable_islands_in_lossless_domain": (
                        self.raw_disassembly_unreachable_islands_in_lossless_domain
                    ),
                    "non_graph_aot_xrefs": self.non_graph_aot_xrefs,
                    "graph_retained_literal_use_omission_policy": (
                        self.graph_retained_literal_use_omission_policy
                    ),
                    "pool_order_and_duplicates_preserved": (
                        self.pool_order_and_duplicates_preserved
                    ),
                    "string_representation": self.string_representation,
                    "integer_representation": self.integer_representation,
                    "double_representation": self.double_representation,
                    "composite_representation": json.loads(
                        str(self.composite_representation)
                    ),
                    "nested_nonliteral_descriptors": json.loads(
                        str(self.nested_nonliteral_descriptors)
                    ),
                    "stream_marker_ids": dict(self.stream_marker_ids),
                }
            )
        return value

    def validate_artifacts(
        self,
        *,
        tokenizer: Any,
        tokenizer_json_path: str | Path | None = None,
        codec_path: str | Path | None = None,
        codebook_path: str | Path | None = None,
    ) -> None:
        observed_tokenizer = tokenizer_fingerprint(tokenizer)
        if observed_tokenizer != self.tokenizer_fingerprint_sha256:
            raise ValueError(
                "tokenizer fingerprint mismatch: "
                f"expected {self.tokenizer_fingerprint_sha256}, observed {observed_tokenizer}"
            )
        if tokenizer_json_path is not None:
            observed_json = sha256_file(tokenizer_json_path)
            if observed_json != self.tokenizer_json_sha256:
                raise ValueError(
                    "tokenizer.json SHA-256 mismatch: "
                    f"expected {self.tokenizer_json_sha256}, observed {observed_json}"
                )
        for label, path, expected in (
            ("codec", codec_path, self.codec_sha256),
            ("codebook", codebook_path, self.codebook_sha256),
        ):
            if path is None:
                continue
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(
                    f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
                )
        if codebook_path is not None:
            with Path(codebook_path).open("r", encoding="utf-8") as handle:
                codebook = json.load(handle)
            if not isinstance(codebook, Mapping):
                raise ValueError("compact codebook must be a JSON object")
            expected_scalars = {
                "tokenizer_json_sha256": self.tokenizer_json_sha256,
                "model_config_sha256": self.model_config_sha256,
                "decoder_model": self.decoder_model,
                "decoder_revision": self.decoder_revision,
                "base_vocab_size": self.base_vocab_size,
            }
            if self.schema == CONTRACT_SCHEMA_V1:
                expected_scalars["dfg_extractor_sha256"] = (
                    self.dfg_extractor_sha256
                )
            else:
                expected_scalars["runtime_symbol_policy_sha256"] = (
                    self.runtime_symbol_policy_sha256
                )
            for field, expected in expected_scalars.items():
                if codebook.get(field) != expected:
                    raise ValueError(
                        f"codebook {field} mismatch: expected {expected!r}, "
                        f"observed {codebook.get(field)!r}"
                    )
            if self.schema in (CONTRACT_SCHEMA_V2, CONTRACT_SCHEMA_V3):
                expected_routes = {
                    name: json.loads(payload)
                    for name, payload in self.extractor_routes
                }
                if codebook.get("extractor_routes") != expected_routes:
                    raise ValueError("codebook extractor_routes mismatch")
            if self.schema == CONTRACT_SCHEMA_V3:
                if codebook.get("schema") != CODEBOOK_SCHEMA_V3:
                    raise ValueError(
                        f"v3 codebook schema must be {CODEBOOK_SCHEMA_V3!r}"
                    )
                expected_pool_scalars = {
                    "pool_extractor_sha256": self.pool_extractor_sha256,
                    "dart_toolchain_manifest_sha256": (
                        self.dart_toolchain_manifest_sha256
                    ),
                    "aot_manifest_sha256": self.aot_manifest_sha256,
                    "pool_reconciliation_manifest_sha256": (
                        self.pool_reconciliation_manifest_sha256
                    ),
                    "graph_codec_dependency_sha256": (
                        self.graph_codec_dependency_sha256
                    ),
                    "target_architecture": self.target_architecture,
                    "pool_schema": self.pool_schema,
                    "pool_encoding": self.pool_encoding,
                    "pool_positional_encoding": json.loads(
                        str(self.pool_positional_encoding)
                    ),
                    "pool_scope": self.pool_scope,
                    "pool_projection": self.pool_projection,
                    "all_encoded_pool_uses_reference_canonical_graph_instructions": (
                        self.all_encoded_pool_uses_reference_canonical_graph_instructions
                    ),
                    "raw_disassembly_unreachable_islands_in_lossless_domain": (
                        self.raw_disassembly_unreachable_islands_in_lossless_domain
                    ),
                    "non_graph_aot_xrefs": self.non_graph_aot_xrefs,
                    "graph_retained_literal_use_omission_policy": (
                        self.graph_retained_literal_use_omission_policy
                    ),
                    "pool_order_and_duplicates_preserved": (
                        self.pool_order_and_duplicates_preserved
                    ),
                    "string_representation": self.string_representation,
                    "integer_representation": self.integer_representation,
                    "double_representation": self.double_representation,
                    "composite_representation": json.loads(
                        str(self.composite_representation)
                    ),
                    "nested_nonliteral_descriptors": json.loads(
                        str(self.nested_nonliteral_descriptors)
                    ),
                }
                for field, expected in expected_pool_scalars.items():
                    if codebook.get(field) != expected:
                        raise ValueError(
                            f"codebook {field} mismatch: expected {expected!r}, "
                            f"observed {codebook.get(field)!r}"
                        )
                codebook_atoms = codebook.get("source_atom_ids")
                if not isinstance(codebook_atoms, Mapping):
                    raise ValueError("v3 codebook source_atom_ids must be an object")
                observed_markers = {
                    marker: codebook_atoms.get(marker) for marker in V3_STREAM_MARKERS
                }
                if observed_markers != dict(self.stream_marker_ids):
                    raise ValueError("codebook v3 stream marker IDs mismatch")
            observed_expansions = {
                int(source_id): tuple(int(value) for value in base_ids)
                for source_id, base_ids in (
                    codebook.get("source_token_expansions") or {}
                ).items()
            }
            if observed_expansions != dict(self.source_token_expansions):
                raise ValueError("codebook source_token_expansions mismatch")

    def validate_decoder_binding(
        self,
        *,
        decoder_model: str,
        decoder_revision: str,
        model_config_path: str | Path | None = None,
    ) -> None:
        observed_model = str(decoder_model or "").strip()
        observed_revision = str(decoder_revision or "").strip()
        if observed_model != self.decoder_model:
            raise ValueError(
                f"decoder model mismatch: expected {self.decoder_model!r}, "
                f"observed {observed_model!r}"
            )
        if observed_revision != self.decoder_revision:
            raise ValueError(
                f"decoder revision mismatch: expected {self.decoder_revision!r}, "
                f"observed {observed_revision!r}"
            )
        if model_config_path is not None:
            observed_config = sha256_file(model_config_path)
            if observed_config != self.model_config_sha256:
                raise ValueError(
                    "decoder config.json SHA-256 mismatch: "
                    f"expected {self.model_config_sha256}, observed {observed_config}"
                )

    def validate_row(self, row: Mapping[str, Any], identity: str) -> list[int]:
        for field, expected in (
            ("compact_codec_sha256", self.codec_sha256),
            ("compact_codebook_sha256", self.codebook_sha256),
            ("compact_tokenizer_sha256", self.tokenizer_json_sha256),
        ):
            observed = _canonical_sha256(row.get(field), f"{identity}.{field}")
            if observed != expected:
                raise ValueError(
                    f"{identity}: {field} mismatch; expected {expected}, observed {observed}"
                )
        raw_ids = row.get("compact_input_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"{identity}: compact_input_ids must be a non-empty list")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
            raise ValueError(f"{identity}: compact_input_ids must contain only integers")
        if any(value < 0 for value in raw_ids):
            raise ValueError(f"{identity}: compact_input_ids contains a negative token ID")
        if self.base_vocab_size is not None:
            source_ids = set(self.source_token_ids)
            unknown = [
                value
                for value in raw_ids
                if value >= self.base_vocab_size and value not in source_ids
            ]
            if unknown:
                raise ValueError(
                    f"{identity}: compact_input_ids contains unregistered source token "
                    f"IDs: {sorted(set(unknown))[:8]}"
                )
        if len(raw_ids) > self.max_source_tokens:
            raise ValueError(
                f"{identity}: compact source needs {len(raw_ids)} tokens, exceeding "
                f"the no-truncation limit {self.max_source_tokens}"
            )
        if self.schema == CONTRACT_SCHEMA_V3:
            self._validate_v3_stream(raw_ids, identity)
        return list(raw_ids)

    def _validate_v3_stream(self, raw_ids: Sequence[int], identity: str) -> None:
        """Fail closed on v3 marker drift before a row reaches the decoder."""

        marker_ids = dict(self.stream_marker_ids)
        positions: dict[str, int] = {}
        for marker in V3_STREAM_MARKERS:
            token_id = marker_ids[marker]
            matches = [index for index, value in enumerate(raw_ids) if value == token_id]
            if len(matches) != 1:
                raise ValueError(
                    f"{identity}: v3 marker {marker} must occur exactly once; "
                    f"observed {len(matches)}"
                )
            positions[marker] = matches[0]
        ordered = [positions[marker] for marker in V3_STREAM_MARKERS]
        if ordered != sorted(ordered):
            raise ValueError(
                f"{identity}: invalid v3 marker order; expected "
                "<G2C3> ... <CFG> ... <PX0> JSON <PEND><END>"
            )
        if positions["<G2C3>"] != 0:
            raise ValueError(f"{identity}: <G2C3> must be the first compact token")
        if positions["<PEND>"] + 1 != positions["<END>"]:
            raise ValueError(f"{identity}: <PEND> must be immediately followed by <END>")
        if positions["<END>"] != len(raw_ids) - 1:
            raise ValueError(f"{identity}: <END> must be the final compact token")
        payload_start = positions["<PX0>"] + 1
        payload_end = positions["<PEND>"]
        if payload_start >= payload_end:
            raise ValueError(f"{identity}: v3 pool JSON payload may not be empty")
        if self.base_vocab_size is None:
            raise AssertionError("validated v3 contract has no base_vocab_size")
        nonbase_payload = sorted(
            {
                value
                for value in raw_ids[payload_start:payload_end]
                if value >= self.base_vocab_size
            }
        )
        if nonbase_payload:
            raise ValueError(
                f"{identity}: v3 pool JSON must use only frozen base-tokenizer IDs; "
                f"observed source IDs {nonbase_payload[:8]}"
            )

    def validate_v3_pool_payload(
        self, raw_ids: Sequence[int], tokenizer: Any, identity: str
    ) -> dict[str, Any]:
        """Decode and canonicalize-check a v3 pool payload with the bound tokenizer."""

        if self.schema != CONTRACT_SCHEMA_V3:
            return {}
        self._validate_v3_stream(raw_ids, identity)
        markers = dict(self.stream_marker_ids)
        start = list(raw_ids).index(markers["<PX0>"]) + 1
        end = list(raw_ids).index(markers["<PEND>"])
        payload_ids = list(raw_ids[start:end])
        if not hasattr(tokenizer, "decode"):
            raise TypeError("tokenizer must expose decode() for v3 pool validation")
        try:
            decoded = tokenizer.decode(
                payload_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = tokenizer.decode(payload_ids, skip_special_tokens=False)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer.decode() must return text")
        try:
            decoded.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(f"{identity}: v3 pool JSON must be ASCII") from error
        value = _strict_json_value(decoded, identity)
        canonical_value = _canonical_v3_pool_envelope(
            _decode_v3_pool_positional(value, identity), identity
        )
        canonical = canonical_v3_pool_json(canonical_value)
        if decoded != canonical:
            raise ValueError(f"{identity}: v3 pool JSON is not canonical ASCII JSON")
        return canonical_value


class SourceTokenEmbeddingOverlay(torch.nn.Module):
    """Input-only embeddings for compact source tokens.

    The frozen Qwen embedding table remains unchanged and the untied LM head is
    never resized.  Compact token IDs are legal only in the conditioning prefix;
    each is represented by a small trainable overlay row.
    """

    def __init__(
        self,
        base_embedding: torch.nn.Module,
        source_token_expansions: Mapping[int, Sequence[int]],
        *,
        base_vocab_size: int,
        initialize_from_expansions: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(base_embedding, "weight"):
            raise TypeError("base embedding module must expose weight")
        if int(base_embedding.weight.size(0)) != int(base_vocab_size):
            raise ValueError(
                "base_vocab_size must equal the unexpanded decoder embedding size"
            )
        expansions = {
            int(source_id): tuple(int(base_id) for base_id in base_ids)
            for source_id, base_ids in source_token_expansions.items()
        }
        if not expansions:
            raise ValueError("source_token_expansions may not be empty")
        source_ids = sorted(expansions)
        if min(source_ids) < base_vocab_size:
            raise ValueError("source token IDs must lie outside the base vocabulary")
        for base_ids in expansions.values():
            if not base_ids or any(value < 0 or value >= base_vocab_size for value in base_ids):
                raise ValueError("invalid base-token expansion")

        self.base_embedding = base_embedding
        self.base_embedding.weight.requires_grad_(False)
        self.base_vocab_size = int(base_vocab_size)
        self.source_token_ids = tuple(source_ids)
        self.source_embeddings = torch.nn.Embedding(
            len(source_ids), int(base_embedding.weight.size(1))
        )
        # In inference the base decoder may already be on CUDA before the
        # overlay is installed (notably after wrapping it with PEFT).  Build
        # the persistent lookup beside the base embedding so a freshly
        # restored overlay is immediately usable on that device.  Training
        # used to hide this bug because the accelerator moved the complete
        # model after overlay installation.
        lookup = torch.full(
            (max(source_ids) + 1,),
            -1,
            dtype=torch.long,
            device=base_embedding.weight.device,
        )
        for index, source_id in enumerate(source_ids):
            lookup[source_id] = index
        self.register_buffer("source_id_to_row", lookup, persistent=True)

        self.source_embeddings = self.source_embeddings.to(
            device=base_embedding.weight.device, dtype=base_embedding.weight.dtype
        )
        if initialize_from_expansions:
            with torch.no_grad():
                # Only the few rows in each codebook expansion are promoted to
                # float32. Never materialize the full Qwen table in fp32. Restore
                # bypasses this 20K-row loop because checkpoint rows overwrite it.
                base_weight = base_embedding.weight.detach()
                for index, source_id in enumerate(source_ids):
                    expansion = torch.tensor(
                        expansions[source_id], device=base_weight.device, dtype=torch.long
                    )
                    mean = base_weight.index_select(0, expansion).float().mean(dim=0)
                    self.source_embeddings.weight[index].copy_(
                        mean.to(self.source_embeddings.weight.dtype)
                    )

    @property
    def weight(self) -> torch.Tensor:
        # Compatibility for utilities that inspect the base vocabulary table.
        return self.base_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            return self.base_embedding(input_ids)
        base_mask = input_ids < self.base_vocab_size
        source_candidates = input_ids.masked_fill(base_mask, 0)
        if int(source_candidates.max().detach().cpu()) >= self.source_id_to_row.numel():
            raise ValueError("input contains a token outside base and source vocabularies")
        source_rows = self.source_id_to_row[source_candidates]
        if torch.any((~base_mask) & (source_rows < 0)):
            raise ValueError("input contains an unregistered compact source token")
        safe_base_ids = input_ids.masked_fill(~base_mask, 0)
        result = self.base_embedding(safe_base_ids)
        if torch.any(~base_mask):
            result = result.clone()
            result[~base_mask] = self.source_embeddings(source_rows[~base_mask])
        return result

    def overlay_state(self) -> dict[str, Any]:
        return {
            "schema": "source-token-embedding-overlay-v1",
            "base_vocab_size": self.base_vocab_size,
            "source_token_ids": list(self.source_token_ids),
            "source_embeddings": self.source_embeddings.weight.detach().cpu(),
        }


def install_source_embedding_overlay(
    model: torch.nn.Module,
    source_token_expansions: Mapping[int, Sequence[int]],
    *,
    base_vocab_size: int,
    initialize_from_expansions: bool = True,
) -> SourceTokenEmbeddingOverlay:
    """Install the input-only overlay without resizing Qwen's output vocabulary."""

    base_embedding = model.get_input_embeddings()
    overlay = SourceTokenEmbeddingOverlay(
        base_embedding,
        source_token_expansions,
        base_vocab_size=base_vocab_size,
        initialize_from_expansions=initialize_from_expansions,
    )
    if not hasattr(model, "set_input_embeddings"):
        raise TypeError("causal LM must expose set_input_embeddings()")
    model.set_input_embeddings(overlay)
    return overlay


def restore_source_embedding_overlay(
    model: torch.nn.Module,
    source_token_expansions: Mapping[int, Sequence[int]],
    checkpoint: str | Path,
    *,
    base_vocab_size: int,
) -> SourceTokenEmbeddingOverlay:
    """Install and restore a previously saved input-only source overlay."""

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or state.get("schema") != "source-token-embedding-overlay-v1":
        raise ValueError("invalid source embedding overlay checkpoint")
    overlay = install_source_embedding_overlay(
        model,
        source_token_expansions,
        base_vocab_size=base_vocab_size,
        initialize_from_expansions=False,
    )
    if int(state.get("base_vocab_size", -1)) != base_vocab_size:
        raise ValueError("source overlay base vocabulary mismatch")
    if tuple(state.get("source_token_ids") or ()) != overlay.source_token_ids:
        raise ValueError("source overlay token-ID contract mismatch")
    weights = state.get("source_embeddings")
    if not isinstance(weights, torch.Tensor) or weights.shape != overlay.source_embeddings.weight.shape:
        raise ValueError("source overlay tensor shape mismatch")
    with torch.no_grad():
        overlay.source_embeddings.weight.copy_(
            weights.to(
                device=overlay.source_embeddings.weight.device,
                dtype=overlay.source_embeddings.weight.dtype,
            )
        )
    return overlay


def migrate_source_embedding_overlay(
    model: torch.nn.Module,
    *,
    old_source_token_expansions: Mapping[int, Sequence[int]],
    new_source_token_expansions: Mapping[int, Sequence[int]],
    checkpoint: str | Path,
    base_vocab_size: int,
) -> tuple[SourceTokenEmbeddingOverlay, dict[str, Any]]:
    """Migrate an overlay across a train-only codebook refit.

    Source-token IDs are a stable ABI, but a refit may assign a different
    base-token expansion (and therefore a different meaning) to an existing
    ID.  Learned rows are safe to reuse *only* when the complete ordered
    expansion tuple is unchanged.  Every changed row is initialized from the
    new expansion's frozen base-embedding mean exactly as a fresh overlay
    would be.

    The returned report contains no tensors and is suitable for inclusion in
    a hash-sealed migration receipt.
    """

    def normalize(
        value: Mapping[int, Sequence[int]], label: str
    ) -> dict[int, tuple[int, ...]]:
        result = {
            int(source_id): tuple(int(base_id) for base_id in base_ids)
            for source_id, base_ids in value.items()
        }
        if not result:
            raise ValueError(f"{label} source-token expansions may not be empty")
        if any(
            not base_ids
            or any(
                base_id < 0 or base_id >= int(base_vocab_size)
                for base_id in base_ids
            )
            for base_ids in result.values()
        ):
            raise ValueError(f"{label} contains an invalid base-token expansion")
        return result

    old_expansions = normalize(
        old_source_token_expansions, "old compact contract"
    )
    new_expansions = normalize(
        new_source_token_expansions, "new compact contract"
    )
    old_ids = tuple(sorted(old_expansions))
    new_ids = tuple(sorted(new_expansions))
    if old_ids != new_ids:
        raise ValueError(
            "overlay migration requires an identical stable source-token ID set"
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, Mapping)
        or state.get("schema") != "source-token-embedding-overlay-v1"
    ):
        raise ValueError("invalid source embedding overlay checkpoint")
    if int(state.get("base_vocab_size", -1)) != int(base_vocab_size):
        raise ValueError("source overlay base vocabulary mismatch")
    if tuple(state.get("source_token_ids") or ()) != old_ids:
        raise ValueError(
            "source overlay token IDs differ from its old compact contract"
        )
    old_weights = state.get("source_embeddings")
    base_embedding = model.get_input_embeddings()
    expected_shape = (
        len(old_ids),
        int(base_embedding.weight.size(1)),
    )
    if (
        not isinstance(old_weights, torch.Tensor)
        or tuple(old_weights.shape) != expected_shape
    ):
        raise ValueError("source overlay tensor shape mismatch")

    # This performs the required new-codebook-mean initialization first.
    overlay = install_source_embedding_overlay(
        model,
        new_expansions,
        base_vocab_size=int(base_vocab_size),
        initialize_from_expansions=True,
    )
    old_row = {source_id: index for index, source_id in enumerate(old_ids)}
    new_row = {
        source_id: index
        for index, source_id in enumerate(overlay.source_token_ids)
    }
    reused_ids = [
        source_id
        for source_id in new_ids
        if old_expansions[source_id] == new_expansions[source_id]
    ]
    reused_id_set = set(reused_ids)
    reinitialized_ids = [
        source_id for source_id in new_ids if source_id not in reused_id_set
    ]
    with torch.no_grad():
        for source_id in reused_ids:
            overlay.source_embeddings.weight[new_row[source_id]].copy_(
                old_weights[old_row[source_id]].to(
                    device=overlay.source_embeddings.weight.device,
                    dtype=overlay.source_embeddings.weight.dtype,
                )
            )

    def expansion_digest(
        expansions: Mapping[int, Sequence[int]],
    ) -> str:
        encoded = json.dumps(
            {
                str(source_id): list(expansions[source_id])
                for source_id in sorted(expansions)
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    report = {
        "schema": "source-token-overlay-expansion-migration-v1",
        "policy": (
            "reuse_learned_row_iff_source_id_and_ordered_base_token_"
            "expansion_are_identical_else_new_codebook_mean"
        ),
        "base_vocab_size": int(base_vocab_size),
        "source_token_ids": list(new_ids),
        "source_token_ids_sha256": hashlib.sha256(
            json.dumps(
                list(new_ids), separators=(",", ":")
            ).encode("ascii")
        ).hexdigest(),
        "old_source_token_expansions_sha256": expansion_digest(
            old_expansions
        ),
        "new_source_token_expansions_sha256": expansion_digest(
            new_expansions
        ),
        "rows": {
            "total": len(new_ids),
            "reused_identical_expansion": len(reused_ids),
            "reinitialized_new_codebook_mean": len(reinitialized_ids),
        },
        "reused_source_token_ids": reused_ids,
        "reinitialized_source_token_ids": reinitialized_ids,
        "invariants": {
            "stable_source_token_id_set_identical": True,
            "changed_expansion_rows_copied_from_old_overlay": False,
            "changed_expansion_rows_initialized_from_new_codebook_mean": True,
            "base_embedding_and_lm_head_not_resized": True,
        },
    }
    return overlay, report


def validate_base_model_vocab(
    model: torch.nn.Module, contract: DirectCompactContract
) -> int:
    """Bind the contract to Qwen's model/config vocab, not tokenizer length.

    Qwen3 reserves model embedding rows that are absent from tokenizer.get_vocab(),
    so tokenizer cardinality must not be used as the source-ID boundary.
    """

    expected = int(contract.base_vocab_size or 0)
    config = getattr(model, "config", None)
    config_vocab = getattr(config, "vocab_size", None)
    embedding = model.get_input_embeddings()
    embedding_vocab = int(embedding.weight.size(0))
    if not isinstance(config_vocab, int) or config_vocab <= 0:
        raise ValueError("decoder config does not expose a positive vocab_size")
    if config_vocab != expected or embedding_vocab != expected:
        raise ValueError(
            "base model vocabulary mismatch: "
            f"contract={expected}, config={config_vocab}, embedding={embedding_vocab}"
        )
    if contract.source_token_ids and min(contract.source_token_ids) < expected:
        raise ValueError("source token IDs overlap the model vocabulary")
    return expected


class DirectCompactBatchCollator:
    """Construct ordinary right-padded causal-LM batches without truncation."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        max_source_tokens: int = 9000,
        max_target_tokens: int = 3072,
        max_total_tokens: int = 12288,
        pad_to_multiple_of: int | None = None,
        source_token_ids: Iterable[int] = (),
        allow_empty_source: bool = False,
    ) -> None:
        if max_source_tokens <= 0 or max_source_tokens > 9000:
            raise ValueError("max_source_tokens must be in [1, 9000]")
        if max_target_tokens <= 0 or max_total_tokens <= 0:
            raise ValueError("target and total token limits must be positive")
        self.pad_token_id = int(pad_token_id)
        self.max_source_tokens = int(max_source_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.max_total_tokens = int(max_total_tokens)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.source_token_ids = frozenset(int(value) for value in source_token_ids)
        self.allow_empty_source = bool(allow_empty_source)

    @staticmethod
    def _ids(feature: Mapping[str, Any], key: str) -> list[int]:
        value = feature.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{key} must be a token-ID sequence")
        result = list(value)
        if any(isinstance(token, bool) or not isinstance(token, int) for token in result):
            raise ValueError(f"{key} must contain only integer token IDs")
        if any(token < 0 for token in result):
            raise ValueError(f"{key} contains a negative token ID")
        return result

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("DirectCompactBatchCollator received an empty batch")
        rows: list[tuple[list[int], list[int], int, int]] = []
        for index, feature in enumerate(features):
            prompt = self._ids(feature, "decoder_prompt_input_ids")
            source = self._ids(feature, "compact_input_ids")
            target_key = "target_input_ids" if "target_input_ids" in feature else "labels"
            target = self._ids(feature, target_key)
            if not source and not self.allow_empty_source:
                raise ValueError(f"batch row {index}: compact source may not be empty")
            if not target:
                raise ValueError(f"batch row {index}: target may not be empty")
            forbidden_targets = self.source_token_ids.intersection(target)
            if forbidden_targets:
                raise ValueError(
                    f"batch row {index}: compact source token IDs occur in target labels: "
                    f"{sorted(forbidden_targets)[:8]}"
                )
            if len(source) > self.max_source_tokens:
                raise ValueError(
                    f"batch row {index}: source has {len(source)} tokens; "
                    f"limit is {self.max_source_tokens}; refusing truncation"
                )
            if len(target) > self.max_target_tokens:
                raise ValueError(
                    f"batch row {index}: target has {len(target)} tokens; "
                    f"limit is {self.max_target_tokens}; refusing truncation"
                )
            total = len(prompt) + len(source) + len(target)
            if total > self.max_total_tokens:
                raise ValueError(
                    f"batch row {index}: prompt+source+target has {total} tokens; "
                    f"limit is {self.max_total_tokens}; refusing truncation"
                )
            input_ids = prompt + source + target
            labels = [-100] * (len(prompt) + len(source)) + target
            rows.append((input_ids, labels, len(prompt), len(source)))

        padded_length = max(len(row[0]) for row in rows)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            if multiple <= 0:
                raise ValueError("pad_to_multiple_of must be positive")
            padded_length = int(math.ceil(padded_length / multiple) * multiple)

        input_batch, mask_batch, label_batch = [], [], []
        prompt_lengths, source_lengths = [], []
        for input_ids, labels, prompt_length, source_length in rows:
            padding = padded_length - len(input_ids)
            input_batch.append(input_ids + [self.pad_token_id] * padding)
            mask_batch.append([1] * len(input_ids) + [0] * padding)
            label_batch.append(labels + [-100] * padding)
            prompt_lengths.append(prompt_length)
            source_lengths.append(source_length)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "source_lengths": torch.tensor(source_lengths, dtype=torch.long),
        }


class DirectCompactCausalLM(torch.nn.Module):
    """Thin decoder-only wrapper used by tests and the direct training entry."""

    def __init__(
        self,
        causal_lm: torch.nn.Module,
        *,
        sequence_sum_nll: bool = False,
        sequence_nll_position_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        self.sequence_sum_nll = bool(sequence_sum_nll)
        self.sequence_nll_position_chunk_size = int(
            sequence_nll_position_chunk_size
        )
        if self.sequence_nll_position_chunk_size <= 0:
            raise ValueError("sequence NLL position chunk size must be positive")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        zero_source_embeddings: bool = False,
        **kwargs: Any,
    ) -> Any:
        # Length metadata is for auditing and is not a decoder argument.
        prompt_lengths = kwargs.pop("prompt_lengths", None)
        source_lengths = kwargs.pop("source_lengths", None)
        if self.sequence_sum_nll:
            if labels is None:
                raise ValueError("sequence-sum NLL requires causal target labels")
            supervised = labels.ne(-100)
            if torch.any(supervised.sum(dim=1) == 0):
                raise ValueError(
                    "every sequence needs at least one supervised target token"
                )
            positions = torch.arange(
                labels.size(1), device=labels.device
            ).unsqueeze(0).expand_as(labels)
            first_targets = positions.masked_fill(~supervised, labels.size(1)).min(
                dim=1
            ).values
            first_prediction = max(int(first_targets.min().item()) - 1, 0)
            # Qwen3 accepts an integer ``logits_to_keep`` and emits only the
            # final positions. This avoids materializing vocabulary logits for
            # the prompt and compact source, which carry no labels.
            kwargs.setdefault(
                "logits_to_keep", int(labels.size(1) - first_prediction)
            )
        if zero_source_embeddings:
            if prompt_lengths is None or source_lengths is None:
                raise ValueError(
                    "zero_source_embeddings requires prompt/source length metadata"
                )
            inputs_embeds = self.causal_lm.get_input_embeddings()(input_ids)
            inputs_embeds = inputs_embeds.clone()
            for row, (prompt_length, source_length) in enumerate(
                zip(prompt_lengths.tolist(), source_lengths.tolist())
            ):
                start = int(prompt_length)
                end = start + int(source_length)
                inputs_embeds[row, start:end] = 0
            outputs = self.causal_lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=(None if self.sequence_sum_nll else labels),
                **kwargs,
            )
        else:
            outputs = self.causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=(None if self.sequence_sum_nll else labels),
                **kwargs,
            )
        if self.sequence_sum_nll:
            logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
            sequence_values = per_sequence_causal_nll_sum(
                logits,
                labels,
                position_chunk_size=self.sequence_nll_position_chunk_size,
            )
            primary = sequence_values.mean()
            if isinstance(outputs, Mapping):
                outputs["loss"] = primary
                outputs["per_sequence_sum_nll"] = sequence_values.detach()
            else:
                outputs.loss = primary
                setattr(
                    outputs,
                    "per_sequence_sum_nll",
                    sequence_values.detach(),
                )
        return outputs

    def get_input_embeddings(self) -> Any:
        return self.causal_lm.get_input_embeddings()

    def get_output_embeddings(self) -> Any:
        if hasattr(self.causal_lm, "get_output_embeddings"):
            return self.causal_lm.get_output_embeddings()
        return None


def per_sequence_causal_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Length-normalized next-token NLL for a standard causal-LM batch."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels must have aligned [batch, sequence] dimensions")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    losses = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    valid = shifted_labels.ne(-100)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def per_sequence_causal_nll_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    position_chunk_size: int = 512,
) -> torch.Tensor:
    """EOS-inclusive sequence NLL sums for Monte Carlo forward-KL training.

    Each sampled teacher sequence is one Monte Carlo draw.  Forward KL differs
    from the base causal LM's token-mean objective: it requires
    ``-log p_student(sequence)`` (the sum of next-token NLLs, including EOS)
    followed by an equal-weight mean across sampled sequences.
    """

    if (
        logits.ndim != 3
        or labels.ndim != 2
        or logits.size(0) != labels.size(0)
        or logits.size(1) > labels.size(1)
        or logits.size(1) < 2
    ):
        raise ValueError(
            "logits must be a nonempty suffix aligned to [batch, sequence] labels"
        )
    if int(position_chunk_size) <= 0:
        raise ValueError("position_chunk_size must be positive")
    label_offset = labels.size(1) - logits.size(1)
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, label_offset + 1 :]
    valid = shifted_labels.ne(-100)
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("every sequence needs at least one supervised target token")
    totals = torch.zeros(
        labels.size(0), device=logits.device, dtype=torch.float32
    )

    def chunk_loss(
        chunk_logits: torch.Tensor, chunk_labels: torch.Tensor
    ) -> torch.Tensor:
        losses = torch.nn.functional.cross_entropy(
            chunk_logits.float().reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(chunk_labels)
        return (losses * chunk_labels.ne(-100)).sum(dim=1)

    for start in range(0, shifted_logits.size(1), int(position_chunk_size)):
        end = min(start + int(position_chunk_size), shifted_logits.size(1))
        chunk_logits = shifted_logits[:, start:end]
        chunk_labels = shifted_labels[:, start:end]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            values = checkpoint(
                chunk_loss,
                chunk_logits,
                chunk_labels,
                use_reentrant=False,
            )
        else:
            values = chunk_loss(chunk_logits, chunk_labels)
        totals = totals + values
    return totals


def matched_permutation_indices(lengths: Sequence[int], seed: int = 42) -> list[int]:
    """Return a deterministic minimum-length-cost derangement for source swaps.

    This uses the Hungarian assignment algorithm rather than cyclic rotations,
    which can pair the longest source with the shortest even when a much closer
    derangement exists.  Seeded tie costs make equal-length solutions stable.
    """

    if len(lengths) < 2:
        raise ValueError("matched permutation requires at least two rows")
    normalized = [int(value) for value in lengths]
    if any(value < 0 for value in normalized):
        raise ValueError("source lengths must be non-negative")
    count = len(normalized)
    rng = random.Random(int(seed))
    tie_order = list(range(count * count))
    rng.shuffle(tie_order)
    tie_rank = {
        (flat // count, flat % count): rank
        for rank, flat in enumerate(tie_order)
    }
    # One unit of aggregate length cost must dominate every possible aggregate
    # tie-rank difference across the complete assignment.
    tie_scale = count**3 + 1
    maximum_delta = max(normalized) - min(normalized)
    forbidden = (maximum_delta + 1) * tie_scale * (count + 1)
    costs = [
        [
            forbidden
            if row == column
            else abs(normalized[row] - normalized[column]) * tie_scale
            + tie_rank[(row, column)]
            for column in range(count)
        ]
        for row in range(count)
    ]

    # Hungarian algorithm for a square integer cost matrix. assignment[row]
    # is the selected donor column.
    u = [0] * (count + 1)
    v = [0] * (count + 1)
    p = [0] * (count + 1)
    way = [0] * (count + 1)
    for row in range(1, count + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (count + 1)
        used = [False] * (count + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, count + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            if delta == float("inf"):
                raise RuntimeError("matched permutation assignment is infeasible")
            integer_delta = int(delta)
            for column in range(count + 1):
                if used[column]:
                    u[p[column]] += integer_delta
                    v[column] -= integer_delta
                else:
                    minimum[column] -= integer_delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [0] * count
    for column in range(1, count + 1):
        assignment[p[column] - 1] = column - 1
    if any(index == donor for index, donor in enumerate(assignment)):
        raise RuntimeError("matched permutation construction produced a fixed point")
    return assignment
