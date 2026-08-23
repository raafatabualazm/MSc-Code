#!/usr/bin/env python3
"""Build a source-blind receipt for Dart AOT object-pool values used by a target.

The parser consumes two outputs from the *same* pinned ``gen_snapshot`` run:

* stderr produced with ``--disassemble --disassemble-optimized``; and
* the V8 snapshot profile produced with ``--write-v8-snapshot-profile-to``.

It never accepts or reads Dart source.  The disassembly supplies target-scoped
pool uses and printable numeric payloads.  The V8 profile independently binds
global-pool indices to typed heap nodes and supplies exact string payloads.

This module intentionally describes non-literal pool objects instead of
serializing their names.  That keeps function/type/debug symbols out of the
model-side receipt while still making every target pool use auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "dart-aot-target-pool-receipt-v1"
PP_FIRST_OFFSET = 0xF
PP_ENTRY_STRIDE = 8
DEFAULT_MAX_COMPOSITE_DEPTH = 8
DEFAULT_MAX_COMPOSITE_NODES = 1024


class PoolReceiptError(ValueError):
    """Raised when the two compiler artifacts cannot be reconciled safely."""


FUNCTION_HEADER_RE = re.compile(
    r"(?m)^Code for optimized function '(?P<name>[^']+)' "
    r"\((?P<kind>[^)]+)\) \{$"
)
FUNCTION_END_RE = re.compile(r"(?m)^}\s*$")
POOL_USE_RE = re.compile(
    r"(?m)^(?P<pc>0x[0-9a-fA-F]+)\s+"
    r"(?P<bytes>[0-9a-fA-F]+)\s+"
    r"(?P<instruction>[^\n]*?\[pp\+(?P<offset>0x[0-9a-fA-F]+)\][^\n]*)$"
)
GLOBAL_POOL_HEADER_RE = re.compile(
    r"(?m)^Global object pool:\s*\nObjectPool len:(?P<length>[0-9]+) \{$"
)
GLOBAL_POOL_ENTRY_RE = re.compile(
    r"(?m)^  \[pp\+(?P<offset>0x[0-9a-fA-F]+)\][ \t]?"
)
GLOBAL_POOL_STORAGE_RE = re.compile(
    r"(?s)^(?P<annotation>.*?)\s+\((?P<storage>obj|native function|raw)\)\s*$"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pool_index_from_offset(offset: int) -> int:
    delta = offset - PP_FIRST_OFFSET
    if delta < 0 or delta % PP_ENTRY_STRIDE:
        raise PoolReceiptError(f"invalid_pp_offset:0x{offset:x}")
    return delta // PP_ENTRY_STRIDE


def pool_offset_from_index(index: int) -> int:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise PoolReceiptError(f"invalid_pool_index:{index!r}")
    return PP_FIRST_OFFSET + index * PP_ENTRY_STRIDE


def _target_scope_role(raw_name: str, target: str) -> str | None:
    tail = raw_name.rsplit("::", 1)[-1]
    exact_names = (target, f"_{target}")
    if tail in exact_names:
        return "exact"
    if any(
        tail.startswith(prefix + separator)
        for prefix in exact_names
        for separator in ("_", ".", "<")
    ):
        return "descendant"
    return None


def _parse_target_uses(
    disassembly: str, target: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    raw_uses: list[dict[str, Any]] = []
    for header in FUNCTION_HEADER_RE.finditer(disassembly):
        raw_name = header.group("name")
        role = _target_scope_role(raw_name, target)
        if role is None:
            continue
        end_match = FUNCTION_END_RE.search(disassembly, header.end())
        if end_match is None:
            raise PoolReceiptError("unterminated_target_function_block")
        block_text = disassembly[header.end() : end_match.start()]
        block = {
            "raw_name": raw_name,
            "kind": header.group("kind"),
            "scope_role": role,
        }
        blocks.append(block)
        for use in POOL_USE_RE.finditer(block_text):
            offset = int(use.group("offset"), 16)
            raw_uses.append(
                {
                    "raw_name": raw_name,
                    "pc": int(use.group("pc"), 16),
                    "offset": offset,
                }
            )

    if not blocks:
        raise PoolReceiptError(f"target_function_not_found:{target}")

    # Raw compiler identities include the private source URI.  Assign stable,
    # neutral receipt IDs and never emit those identities.
    unique_blocks = {
        (block["raw_name"], block["kind"], block["scope_role"])
        for block in blocks
    }
    ordered_blocks = sorted(
        unique_blocks,
        key=lambda item: (0 if item[2] == "exact" else 1, item[0], item[1]),
    )
    function_id_by_name: dict[str, str] = {}
    public_blocks: list[dict[str, Any]] = []
    exact_count = 0
    descendant_count = 0
    for raw_name, kind, role in ordered_blocks:
        if role == "exact":
            function_id = target if exact_count == 0 else f"{target}_exact_{exact_count}"
            exact_count += 1
        else:
            function_id = f"{target}_descendant_{descendant_count}"
            descendant_count += 1
        function_id_by_name[raw_name] = function_id
        public_blocks.append(
            {
                "function_id": function_id,
                "function_kind": kind,
                "scope_role": role,
            }
        )

    uses = [
        {
            "function_id": function_id_by_name[use["raw_name"]],
            "pc": f"0x{use['pc']:x}",
            "pool_offset": use["offset"],
        }
        for use in raw_uses
    ]
    uses.sort(key=lambda use: (use["pool_offset"], use["function_id"], int(use["pc"], 16)))
    return public_blocks, uses


def _parse_global_pool(disassembly: str) -> tuple[int, dict[int, dict[str, str]]]:
    headers = list(GLOBAL_POOL_HEADER_RE.finditer(disassembly))
    if len(headers) != 1:
        raise PoolReceiptError(f"global_pool_header_count:{len(headers)}")
    header = headers[0]
    declared_length = int(header.group("length"))

    # The global pool is the final disassembly section.  Entries can contain
    # literal newlines, including a line containing only `}`; use the last
    # closing brace and entry-start delimiters instead of line parsing.
    pool_end = disassembly.rfind("\n}")
    if pool_end < header.end():
        raise PoolReceiptError("global_pool_end_not_found")
    pool_text = disassembly[header.end() : pool_end]
    starts = list(GLOBAL_POOL_ENTRY_RE.finditer(pool_text))
    if len(starts) != declared_length:
        raise PoolReceiptError(
            f"global_pool_entry_count:{len(starts)}!={declared_length}"
        )

    entries: dict[int, dict[str, str]] = {}
    for position, start in enumerate(starts):
        segment_end = starts[position + 1].start() if position + 1 < len(starts) else len(pool_text)
        segment = pool_text[start.end() : segment_end]
        storage_match = GLOBAL_POOL_STORAGE_RE.match(segment)
        if storage_match is None:
            raise PoolReceiptError(f"unparseable_global_pool_entry:{start.group('offset')}")
        offset = int(start.group("offset"), 16)
        index = pool_index_from_offset(offset)
        if index != position:
            raise PoolReceiptError(
                f"global_pool_position_mismatch:{position}:0x{offset:x}:{index}"
            )
        if offset in entries:
            raise PoolReceiptError(f"duplicate_global_pool_offset:0x{offset:x}")
        entries[offset] = {
            "annotation": storage_match.group("annotation"),
            "storage": storage_match.group("storage").replace(" ", "_"),
        }
    return declared_length, entries


class V8SnapshotGraph:
    """Strict reader for Dart's V8 heap-snapshot profile encoding."""

    def __init__(self, profile: dict[str, Any]) -> None:
        try:
            snapshot = profile["snapshot"]
            meta = snapshot["meta"]
            node_fields = meta["node_fields"]
            edge_fields = meta["edge_fields"]
            node_types = meta["node_types"][node_fields.index("type")]
            edge_types = meta["edge_types"][edge_fields.index("type")]
            strings = profile["strings"]
            flat_nodes = profile["nodes"]
            flat_edges = profile["edges"]
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise PoolReceiptError("invalid_v8_profile_schema") from error

        if not node_fields or not edge_fields:
            raise PoolReceiptError("empty_v8_profile_fields")
        node_width = len(node_fields)
        edge_width = len(edge_fields)
        if len(flat_nodes) % node_width or len(flat_edges) % edge_width:
            raise PoolReceiptError("misaligned_v8_profile_arrays")

        self.nodes: dict[int, dict[str, Any]] = {}
        for offset in range(0, len(flat_nodes), node_width):
            values = flat_nodes[offset : offset + node_width]
            node = dict(zip(node_fields, values))
            try:
                node["type_name"] = node_types[node["type"]]
                node["name_text"] = strings[node["name"]]
            except (IndexError, TypeError) as error:
                raise PoolReceiptError(f"invalid_v8_node:{offset}") from error
            node["offset"] = offset
            node["edges"] = []
            self.nodes[offset] = node

        edge_cursor = 0
        for offset in range(0, len(flat_nodes), node_width):
            node = self.nodes[offset]
            edge_count = node.get("edge_count")
            if not isinstance(edge_count, int) or isinstance(edge_count, bool) or edge_count < 0:
                raise PoolReceiptError(f"invalid_v8_edge_count:{offset}")
            for _ in range(edge_count):
                values = flat_edges[edge_cursor : edge_cursor + edge_width]
                if len(values) != edge_width:
                    raise PoolReceiptError("truncated_v8_edges")
                edge_cursor += edge_width
                edge = dict(zip(edge_fields, values))
                try:
                    edge["type_name"] = edge_types[edge["type"]]
                    if edge["type_name"] == "element":
                        edge["label"] = edge["name_or_index"]
                    else:
                        edge["label"] = strings[edge["name_or_index"]]
                    edge["target"] = self.nodes[edge["to_node"]]
                except (KeyError, IndexError, TypeError) as error:
                    raise PoolReceiptError(f"invalid_v8_edge:{edge_cursor // edge_width - 1}") from error
                node["edges"].append(edge)
        if edge_cursor != len(flat_edges):
            raise PoolReceiptError(f"unused_v8_edges:{len(flat_edges) - edge_cursor}")

        expected_nodes = snapshot.get("node_count")
        expected_edges = snapshot.get("edge_count")
        if expected_nodes is not None and expected_nodes != len(self.nodes):
            raise PoolReceiptError("v8_node_count_mismatch")
        if expected_edges is not None and expected_edges != len(flat_edges) // edge_width:
            raise PoolReceiptError("v8_edge_count_mismatch")

        pools = [node for node in self.nodes.values() if node["type_name"] == "ObjectPool"]
        nonempty_size_pools = [node for node in pools if node.get("self_size", 0) > 0]
        if len(nonempty_size_pools) == 1:
            self.global_pool = nonempty_size_pools[0]
        else:
            if not pools:
                raise PoolReceiptError("v8_object_pool_not_found")
            largest_edge_count = max(node["edge_count"] for node in pools)
            largest = [node for node in pools if node["edge_count"] == largest_edge_count]
            if len(largest) != 1:
                raise PoolReceiptError("v8_global_pool_ambiguous")
            self.global_pool = largest[0]

        self.pool_targets: dict[int, dict[str, Any]] = {}
        for edge in self.global_pool["edges"]:
            if edge["type_name"] != "element":
                continue
            index = edge["label"]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise PoolReceiptError("invalid_v8_pool_edge_index")
            if index in self.pool_targets:
                raise PoolReceiptError(f"duplicate_v8_pool_edge:{index}")
            self.pool_targets[index] = edge["target"]


def _annotation_digest(annotation: str) -> str:
    return sha256_bytes(annotation.encode("utf-8"))


def _parse_double(annotation: str) -> dict[str, Any]:
    text = annotation.strip()
    special = {
        "nan": "NaN",
        "+nan": "NaN",
        "-nan": "NaN",
        "inf": "Infinity",
        "+inf": "Infinity",
        "infinity": "Infinity",
        "+infinity": "Infinity",
        "-inf": "-Infinity",
        "-infinity": "-Infinity",
    }
    if text.lower() in special:
        return {
            "type": "double",
            "value": special[text.lower()],
            "text": text,
            "exact_bits_available": False,
        }
    try:
        value = float(text)
    except ValueError as error:
        raise PoolReceiptError(f"unparseable_double_annotation:{text!r}") from error
    if not math.isfinite(value):
        raise PoolReceiptError(f"noncanonical_double_annotation:{text!r}")
    return {
        "type": "double",
        "value": value,
        "text": text,
        "exact_bits_available": False,
    }


def _parse_integer(annotation: str, fallback: str) -> dict[str, Any]:
    text = annotation.strip()
    candidates = (text, fallback.strip())
    for candidate in candidates:
        try:
            value = int(candidate, 0)
            return {"type": "int", "decimal": str(value)}
        except ValueError:
            continue
    raise PoolReceiptError(f"unparseable_integer_annotation:{text!r}")


def _nonliteral_kind(profile_type: str) -> str:
    if profile_type in {"Type", "TypeArguments", "FunctionType", "TypeParameter", "TypeParameters"}:
        return "type_metadata"
    if profile_type == "Code":
        return "code"
    if profile_type in {"Function", "Closure", "ClosureData"}:
        return "callable"
    if profile_type in {"Field", "PatchClass", "Class"}:
        return "declaration_metadata"
    if profile_type in {"ArgumentsDescriptor", "SubtypeTestCache", "UnlinkedCall"}:
        return "calling_convention_metadata"
    return "runtime_object"


class _NodeDecoder:
    def __init__(
        self,
        graph: V8SnapshotGraph,
        footer: dict[int, dict[str, str]],
        *,
        max_depth: int,
        max_nodes: int,
    ) -> None:
        self.graph = graph
        self.footer = footer
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.visited_nodes: set[int] = set()
        self.annotations_by_node: dict[int, list[str]] = {}
        for index, node in graph.pool_targets.items():
            offset = pool_offset_from_index(index)
            if offset in footer:
                self.annotations_by_node.setdefault(node["offset"], []).append(
                    footer[offset]["annotation"]
                )

    def _annotation_for_node(self, node: dict[str, Any], preferred: str | None) -> str:
        if preferred is not None:
            return preferred
        annotations = self.annotations_by_node.get(node["offset"], [])
        unique = sorted(set(annotations))
        if len(unique) == 1:
            return unique[0]
        if not unique:
            return ""
        raise PoolReceiptError(f"conflicting_annotations_for_v8_node:{node['offset']}")

    def decode(
        self,
        node: dict[str, Any],
        *,
        preferred_annotation: str | None = None,
        depth: int = 0,
        ancestors: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        node_offset = node["offset"]
        if node_offset in ancestors:
            return {"category": "reference", "profile_node_offset": node_offset}
        if depth > self.max_depth:
            return {
                "category": "truncated",
                "reason": "max_composite_depth",
                "profile_node_offset": node_offset,
            }
        self.visited_nodes.add(node_offset)
        if len(self.visited_nodes) > self.max_nodes:
            return {
                "category": "truncated",
                "reason": "max_composite_nodes",
                "profile_node_offset": node_offset,
            }

        profile_type = node["type_name"]
        annotation = self._annotation_for_node(node, preferred_annotation)
        if profile_type in {"CanonicalString", "(RO) String"}:
            encoded = node["name_text"].encode("utf-16-le", errors="surrogatepass")
            code_units = [
                encoded[index] | (encoded[index + 1] << 8)
                for index in range(0, len(encoded), 2)
            ]
            return {
                "category": "literal",
                "literal": {
                    "type": "string",
                    "code_units": code_units,
                    "value": node["name_text"],
                },
            }
        if profile_type == "double":
            if not annotation:
                return {
                    "category": "unresolved_literal",
                    "literal_type": "double",
                    "reason": "numeric_payload_absent_from_profile",
                }
            return {"category": "literal", "literal": _parse_double(annotation)}
        if profile_type in {"int", "Smi"}:
            return {
                "category": "literal",
                "literal": _parse_integer(annotation, node["name_text"]),
            }
        if profile_type == "bool":
            value_text = node["name_text"] if node["name_text"] in {"true", "false"} else annotation.strip()
            if value_text not in {"true", "false"}:
                raise PoolReceiptError(f"unparseable_bool_node:{node_offset}")
            return {
                "category": "literal",
                "literal": {"type": "bool", "value": value_text == "true"},
            }
        if profile_type == "Null":
            return {"category": "literal", "literal": {"type": "null", "value": None}}

        if profile_type in {"Array", "Map"}:
            next_ancestors = ancestors | {node_offset}
            element_edges = [edge for edge in node["edges"] if edge["type_name"] == "element"]
            element_edges.sort(key=lambda edge: (edge["label"], edge["to_node"]))
            omitted = Counter(
                edge["type_name"] for edge in node["edges"] if edge["type_name"] != "element"
            )
            elements = [
                {
                    "index": edge["label"],
                    "value": self.decode(
                        edge["target"],
                        depth=depth + 1,
                        ancestors=next_ancestors,
                    ),
                }
                for edge in element_edges
            ]
            complete = all(
                element["value"]["category"] not in {"truncated", "unresolved_literal"}
                for element in elements
            )
            return {
                "category": "composite",
                "composite_type": "array_storage" if profile_type == "Array" else "map_storage",
                "elements": elements,
                "omitted_edge_counts": dict(sorted(omitted.items())),
                "complete": complete,
            }

        return {
            "category": "nonliteral",
            "nonliteral_kind": _nonliteral_kind(profile_type),
            "profile_type": profile_type,
        }


def build_pool_receipt(
    disassembly: str,
    profile: dict[str, Any],
    *,
    target: str = "candidate",
    disassembly_sha256: str | None = None,
    profile_sha256: str | None = None,
    max_composite_depth: int = DEFAULT_MAX_COMPOSITE_DEPTH,
    max_composite_nodes: int = DEFAULT_MAX_COMPOSITE_NODES,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
        raise PoolReceiptError(f"invalid_target_name:{target!r}")
    if max_composite_depth < 0 or max_composite_nodes < 1:
        raise PoolReceiptError("invalid_composite_limits")

    functions, uses = _parse_target_uses(disassembly, target)
    declared_length, footer = _parse_global_pool(disassembly)
    graph = V8SnapshotGraph(profile)
    if graph.pool_targets and max(graph.pool_targets) >= declared_length:
        raise PoolReceiptError("v8_pool_index_exceeds_declared_length")
    decoder = _NodeDecoder(
        graph,
        footer,
        max_depth=max_composite_depth,
        max_nodes=max_composite_nodes,
    )

    uses_by_offset: dict[int, list[dict[str, str]]] = {}
    for use in uses:
        uses_by_offset.setdefault(use["pool_offset"], []).append(
            {"function_id": use["function_id"], "pc": use["pc"]}
        )

    entries: list[dict[str, Any]] = []
    for offset in sorted(uses_by_offset):
        index = pool_index_from_offset(offset)
        if offset not in footer:
            raise PoolReceiptError(f"target_pool_offset_missing_from_footer:0x{offset:x}")
        footer_entry = footer[offset]
        storage = footer_entry["storage"]
        entry: dict[str, Any] = {
            "pool_offset": f"0x{offset:x}",
            "pp_offset": offset,
            "pool_index": index,
            "storage": storage,
            "annotation_sha256": _annotation_digest(footer_entry["annotation"]),
            "uses": sorted(
                uses_by_offset[offset],
                key=lambda use: (use["function_id"], int(use["pc"], 16)),
            ),
        }
        profile_node = graph.pool_targets.get(index)
        if storage == "obj":
            if profile_node is None:
                entry.update(
                    {
                        "category": "unresolved_object",
                        "reason": "v8_global_pool_edge_missing",
                    }
                )
            else:
                entry.update(
                    decoder.decode(
                        profile_node,
                        preferred_annotation=footer_entry["annotation"],
                    )
                )
                entry["profile_node_type"] = profile_node["type_name"]
                entry["profile_node_offset"] = profile_node["offset"]
        elif storage == "native_function":
            entry.update({"category": "nonliteral", "nonliteral_kind": "native_function"})
        elif storage == "raw":
            entry.update({"category": "nonliteral", "nonliteral_kind": "raw_pool_value"})
        else:  # pragma: no cover - guarded by parser regex
            raise PoolReceiptError(f"unknown_pool_storage:{storage}")
        entries.append(entry)

    disassembly_hash = disassembly_sha256 or sha256_bytes(disassembly.encode("utf-8"))
    profile_hash = profile_sha256 or sha256_bytes(
        json.dumps(profile, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "source_blind": True,
        "target_function": target,
        "inputs": {
            "disassembly_sha256": disassembly_hash,
            "v8_profile_sha256": profile_hash,
        },
        "pool_contract": {
            "first_offset": f"0x{PP_FIRST_OFFSET:x}",
            "entry_stride_bytes": PP_ENTRY_STRIDE,
            "declared_entries": declared_length,
            "index_formula": "(pp_offset-0xf)/8",
        },
        "target_scope": functions,
        "entries": entries,
        "summary": {
            "target_functions": len(functions),
            "unique_pool_entries": len(entries),
            "pool_uses": sum(len(entry["uses"]) for entry in entries),
            "literal_entries": sum(entry["category"] == "literal" for entry in entries),
            "composite_entries": sum(entry["category"] == "composite" for entry in entries),
            "nonliteral_entries": sum(entry["category"] == "nonliteral" for entry in entries),
            "unresolved_entries": sum(entry["category"].startswith("unresolved") for entry in entries),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="candidate")
    parser.add_argument("--max-composite-depth", type=int, default=DEFAULT_MAX_COMPOSITE_DEPTH)
    parser.add_argument("--max-composite-nodes", type=int, default=DEFAULT_MAX_COMPOSITE_NODES)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    disassembly_bytes = args.disassembly.read_bytes()
    profile_bytes = args.profile.read_bytes()
    try:
        disassembly = disassembly_bytes.decode("utf-8")
        profile = json.loads(profile_bytes)
        receipt = build_pool_receipt(
            disassembly,
            profile,
            target=args.target,
            disassembly_sha256=sha256_bytes(disassembly_bytes),
            profile_sha256=sha256_bytes(profile_bytes),
            max_composite_depth=args.max_composite_depth,
            max_composite_nodes=args.max_composite_nodes,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, PoolReceiptError) as error:
        raise SystemExit(f"pool_receipt_failed:{error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
