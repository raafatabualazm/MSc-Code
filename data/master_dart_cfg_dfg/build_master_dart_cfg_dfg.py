#!/usr/bin/env python3
"""Build a deduplicated Dart/Flutter assembly -> source dataset with unified CFG+DFG.

The builder intentionally keeps the held-out test files out of the training pool.
It converts all graph rows to one fixed block/edge schema, generates behavioral
oracle tests for records without a harness, performs deterministic static checks,
and records runtime-test status explicitly instead of claiming tests were run when
no Dart SDK is available.

Default inputs are the five files uploaded with this task.  The output is designed
for graph-conditioned decompilation training while retaining enough provenance to
rebuild or audit every row.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA = "antigravity-master-dart-cfg-dfg-v1"
GRAPH_SCHEMA = "antigravity-unified-cfg-dfg-v1"
TEST_SCHEMA = "antigravity-behavior-tests-v1"
BUILD_VERSION = "2026-07-16.1"

CONTROL_TYPES = {
    "linear_fallthrough",
    "conditional_true",
    "conditional_false",
    "unconditional_jump",
    "loop_backedge",
    "switch_case",
    "exception",
    "call",
    "return",
}

# Optional import of the user's previous register def-use extractor.  The
# generated test-set DFG still works without it, but register coverage is richer
# when the module is available.
_DFG_MODULE = None
try:
    _PRIOR = Path(__file__).with_name("prior_extractors")
    if _PRIOR.is_dir():
        sys.path.insert(0, str(_PRIOR))
    import dfg_extractor as _DFG_MODULE  # type: ignore
except Exception:
    _DFG_MODULE = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_no}, got {type(row).__name__}")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def gzip_file(src: Path, dst: Path) -> None:
    with src.open("rb") as f_in, dst.open("wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw_out, mtime=0) as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)


# ---------------------------------------------------------------------------
# Canonical fingerprints
# ---------------------------------------------------------------------------

def canonicalize_dart_source(source: str) -> str:
    """Remove comments and insignificant whitespace while preserving strings.

    This is a lexical fingerprint, not a formatter. It supports ordinary/raw and
    triple-quoted strings plus nested block comments, which is sufficient for
    duplicate/leakage detection without changing string literal contents.
    """
    out: list[str] = []
    i, n = 0, len(source)
    block_depth = 0
    line_comment = False
    quote: str | None = None
    triple = False
    raw = False

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if line_comment:
            if ch in "\r\n":
                line_comment = False
            i += 1
            continue

        if block_depth:
            if ch == "/" and nxt == "*":
                block_depth += 1
                i += 2
            elif ch == "*" and nxt == "/":
                block_depth -= 1
                i += 2
            else:
                i += 1
            continue

        if quote is not None:
            if triple:
                marker = quote * 3
                if source.startswith(marker, i):
                    out.append(marker)
                    i += 3
                    quote = None
                    triple = False
                    raw = False
                    continue
            else:
                if ch == quote:
                    out.append(ch)
                    i += 1
                    quote = None
                    raw = False
                    continue
            if ch == "\\" and not raw and i + 1 < n:
                out.append(source[i : i + 2])
                i += 2
            else:
                out.append(ch)
                i += 1
            continue

        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_depth = 1
            i += 2
            continue

        # Raw string prefix belongs to the literal.
        if ch in "rR" and i + 1 < n and source[i + 1] in "'\"":
            q = source[i + 1]
            marker = q * 3
            if source.startswith(marker, i + 1):
                out.extend([ch, marker])
                i += 4
                quote, triple, raw = q, True, True
            else:
                out.extend([ch, q])
                i += 2
                quote, triple, raw = q, False, True
            continue

        if ch in "'\"":
            marker = ch * 3
            if source.startswith(marker, i):
                out.append(marker)
                i += 3
                quote, triple, raw = ch, True, False
            else:
                out.append(ch)
                i += 1
                quote, triple, raw = ch, False, False
            continue

        if ch.isspace():
            i += 1
            continue
        out.append(ch)
        i += 1

    return "".join(out)


_INSN_LINE = re.compile(r"^\s*0x[0-9a-fA-F]+\s+<\+\d+>:\s*(.*?)\s*$")
_SOURCE_SYMBOL_LINE = re.compile(r"^\s*\d+:\s+(?:static\s+)?(?:void|int|double|num|bool|String|dynamic|Object|Future|List|Map|Set|Iterable|[A-Za-z_$])")
_BRANCH_OR_CALL = re.compile(r"^(?:j\w+|callq?|b(?:\.[a-z]+)?|bl|blr|br|cbz|cbnz|tbz|tbnz)\b", re.I)


def normalize_instruction(instruction: str) -> str:
    text = re.sub(r"\s+", " ", instruction.strip())
    text = re.sub(r"\s*,\s*", ",", text)
    # Absolute code addresses vary between builds. Normalize only direct targets,
    # not data constants or offsets inside memory operands.
    if _BRANCH_OR_CALL.match(text):
        text = re.sub(r"\b0x[0-9a-fA-F]+\b(?=\s*(?:<|$))", "<addr>", text)
    return text


def canonicalize_assembly(assembly: str, cfg: Sequence[dict[str, Any]] | None = None) -> str:
    instructions: list[str] = []
    for raw_line in assembly.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _INSN_LINE.match(raw_line)
        if match:
            instructions.append(normalize_instruction(match.group(1)))
            continue
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Dump of assembler", "End of assembler", "All functions matching", "File file:")):
            continue
        if _SOURCE_SYMBOL_LINE.match(stripped):
            continue
        # Some cleaned dumps already contain bare instructions.
        if re.match(r"^[a-zA-Z][a-zA-Z0-9.]*\s", stripped) and not stripped.startswith(("Dart SDK", "GNU gdb")):
            instructions.append(normalize_instruction(stripped))

    if not instructions and cfg:
        for block in cfg:
            for insn in block.get("instructions") or []:
                instructions.append(normalize_instruction(str(insn)))

    return "\n".join(instructions)


def source_fingerprint(source: str) -> str:
    return sha256_text(canonicalize_dart_source(source))


def assembly_fingerprint(assembly: str, cfg: Sequence[dict[str, Any]] | None = None) -> str:
    return sha256_text(canonicalize_assembly(assembly, cfg))


# ---------------------------------------------------------------------------
# Dart source and signature helpers
# ---------------------------------------------------------------------------

def source_of(row: dict[str, Any]) -> str:
    value = row.get("dart_source")
    if not isinstance(value, str):
        value = row.get("source")
    return value if isinstance(value, str) else ""


def function_of(row: dict[str, Any]) -> str:
    value = row.get("function") or row.get("name") or "main"
    return str(value).strip() or "main"


def simple_function_name(name: str) -> str:
    return name.split(".")[-1].strip()


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    result: list[str] = []
    current: list[str] = []
    stack: list[str] = []
    quote: str | None = None
    escape = False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    for ch in text:
        if quote:
            current.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch in pairs:
            stack.append(pairs[ch])
            current.append(ch)
        elif stack and ch == stack[-1]:
            stack.pop()
            current.append(ch)
        elif ch == delimiter and not stack:
            item = "".join(current).strip()
            if item:
                result.append(item)
            current = []
        else:
            current.append(ch)
    item = "".join(current).strip()
    if item:
        result.append(item)
    return result


def infer_signature(source: str, target: str) -> str:
    explicit = simple_function_name(target)
    # Remove annotations only for matching; retain declaration text.
    pattern = re.compile(
        rf"(?m)^[ \t]*(?:@[A-Za-z_$][\w$.:]*(?:\([^\n]*\))?\s*)*"
        rf"(?P<decl>(?:(?:external|static|late|final|const|factory|operator|get|set|async|sync\*)\s+)*"
        rf"(?:[A-Za-z_$][\w$<>,?.\[\] ]*\s+)?{re.escape(explicit)}\s*\((?P<params>[^)]*)\))\s*"
        rf"(?:async\*?|sync\*?)?\s*(?:=>|\{{)",
    )
    match = pattern.search(source)
    if match:
        return re.sub(r"\s+", " ", match.group("decl").strip())
    if explicit == "main":
        return "void main()"
    return f"dynamic {explicit}()"


def framework_of(source: str, target: str) -> str:
    if re.search(r"package:flutter/|\bdart:ui\b|\bWidget\b|runApp\s*\(", source):
        return "flutter"
    if simple_function_name(target) == "main" or re.search(r"\bmain\s*\(", source):
        return "dart_cli"
    return "dart_library"


def contains_declaration(source: str, target: str) -> bool:
    name = re.escape(simple_function_name(target))
    return bool(re.search(rf"(?m)^\s*(?:@[\w$.:]+(?:\([^\n]*\))?\s*)*(?:[\w$<>,?.\[\] ]+\s+)?{name}\s*\([^;]*\)\s*(?:=>|\{{)", source))


def has_main(source: str) -> bool:
    return bool(re.search(r"(?m)^\s*(?:@[\w$.:]+(?:\([^\n]*\))?\s*)*(?:(?:Future(?:<[^>]+>)?|void|dynamic|int)\s+)?main\s*\(", source))


def source_characteristics(source: str) -> dict[str, bool]:
    return {
        "uses_stdin": bool(re.search(r"\bstdin\b|readLineSync\s*\(", source)),
        "uses_random": bool(re.search(r"\bRandom\s*\(\s*\)|Random\.secure\s*\(", source)),
        "uses_time": bool(re.search(r"DateTime\.now\s*\(|Stopwatch\s*\(", source)),
        "uses_ffi": bool(re.search(r"dart:ffi|DynamicLibrary", source)),
        "uses_isolates": bool(re.search(r"dart:isolate|Isolate\.", source)),
        "uses_file_or_network": bool(re.search(r"\bFile\s*\(|\bDirectory\s*\(|HttpClient|Socket\.", source)),
    }


# ---------------------------------------------------------------------------
# Unified graph representation
# ---------------------------------------------------------------------------

def _control_edges(raw_edges: Sequence[dict[str, Any]], node_count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for edge in raw_edges or []:
        if edge.get("edge_type") == "dataflow" or edge.get("edge_family") == "data":
            continue
        try:
            source, target = int(edge.get("source")), int(edge.get("target"))
        except (TypeError, ValueError):
            continue
        if not (0 <= source < node_count and 0 <= target < node_count):
            continue
        edge_type = str(edge.get("edge_type") or "linear_fallthrough")
        key = (source, target, edge_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "source": source,
                "target": target,
                "edge_family": "control",
                "edge_type": edge_type,
                "locations": [],
                "dependency_count": 0,
            }
        )
    result.sort(key=lambda e: (e["source"], e["target"], e["edge_type"]))
    return result


_STACK_LOCATION = re.compile(r"\[(?P<base>r(?:bp|sp))\s*(?P<sign>[+-])\s*(?P<imm>0x[0-9a-fA-F]+|\d+)\]", re.I)
_FLAGS_WRITE = {
    "add", "sub", "xor", "or", "and", "imul", "mul", "div", "idiv", "adc", "sbb",
    "shl", "shr", "sar", "rol", "ror", "not", "neg", "inc", "dec", "cmp", "test",
}
_RMW = {"add", "sub", "xor", "or", "and", "imul", "adc", "sbb", "shl", "shr", "sar", "rol", "ror", "not", "neg", "inc", "dec"}
_MOV = {"mov", "movzx", "movsx", "movsxd", "movabs", "movbe"}


def _stack_locations(operand: str) -> list[str]:
    locations: list[str] = []
    for match in _STACK_LOCATION.finditer(operand):
        base = match.group("base").lower()
        value = int(match.group("imm"), 0)
        if match.group("sign") == "-":
            value = -value
        locations.append(f"stack:{base}:{value}")
    return locations


def instruction_location_def_use(instruction: str) -> tuple[set[str], set[str]]:
    """Return (reads, writes) for tracked registers, flags, and rbp/rsp slots."""
    text = re.sub(r"<[^>]*>", "", instruction.strip())
    text = re.sub(r"//.*$", "", text).strip()
    if not text:
        return set(), set()
    parts = text.split(None, 1)
    mnemonic = parts[0].lower()
    operand_text = parts[1] if len(parts) > 1 else ""
    operands = split_top_level(operand_text)

    reads: set[str] = set()
    writes: set[str] = set()
    if _DFG_MODULE is not None:
        try:
            r, w = _DFG_MODULE.instruction_def_use(instruction)
            reads.update(str(x) for x in r)
            writes.update(str(x) for x in w)
        except Exception:
            pass

    memory = [_stack_locations(op) for op in operands]
    if mnemonic == "lea":
        pass
    elif mnemonic in _MOV:
        if memory:
            writes.update(memory[0])
        for values in memory[1:]:
            reads.update(values)
    elif mnemonic in _RMW:
        if memory:
            reads.update(memory[0])
            writes.update(memory[0])
        for values in memory[1:]:
            reads.update(values)
    elif mnemonic in {"cmp", "test", "push"}:
        for values in memory:
            reads.update(values)
    elif mnemonic == "pop":
        if memory:
            writes.update(memory[0])
    else:
        # Conservative default: explicit memory operands are reads.
        for values in memory:
            reads.update(values)

    if mnemonic in _FLAGS_WRITE:
        writes.add("flags")
    if (mnemonic.startswith("j") and mnemonic not in {"jmp", "jmpq"}) or mnemonic.startswith(("cmov", "set")):
        reads.add("flags")
    if mnemonic in {"adc", "sbb"}:
        reads.add("flags")

    return reads, writes


def block_location_def_use(instructions: Sequence[str]) -> tuple[set[str], set[str]]:
    definitions: set[str] = set()
    upward_uses: set[str] = set()
    for instruction in instructions:
        reads, writes = instruction_location_def_use(instruction)
        upward_uses.update(location for location in reads if location not in definitions)
        definitions.update(writes)
    return definitions, upward_uses


def build_enriched_dfg(blocks: Sequence[dict[str, Any]], control_edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Block-level reaching definitions with locations aggregated per pair."""
    n = len(blocks)
    if n <= 1:
        return []
    generated: list[set[str]] = []
    upward: list[set[str]] = []
    for block in blocks:
        defs, uses = block_location_def_use(block.get("instructions") or [])
        generated.append(defs)
        upward.append(uses)

    predecessors: list[list[int]] = [[] for _ in range(n)]
    for edge in control_edges:
        s, t = edge["source"], edge["target"]
        predecessors[t].append(s)

    reaching_out: list[dict[str, set[int]]] = [dict() for _ in range(n)]

    def compute_in(block_id: int) -> dict[str, set[int]]:
        incoming: dict[str, set[int]] = {}
        for pred in predecessors[block_id]:
            for location, sources in reaching_out[pred].items():
                incoming.setdefault(location, set()).update(sources)
        return incoming

    for _ in range(min(2 * n + 4, 128)):
        changed = False
        for block_id in range(n):
            incoming = compute_in(block_id)
            outgoing: dict[str, set[int]] = {}
            for location in set(incoming) | generated[block_id]:
                outgoing[location] = {block_id} if location in generated[block_id] else set(incoming[location])
            if outgoing != reaching_out[block_id]:
                reaching_out[block_id] = outgoing
                changed = True
        if not changed:
            break

    pair_locations: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    for block_id in range(n):
        incoming = compute_in(block_id)
        for location in upward[block_id]:
            for definition_block in incoming.get(location, ()):
                pair_locations[(definition_block, block_id)].add(location)

    result: list[dict[str, Any]] = []
    for (source, target), locations in sorted(pair_locations.items()):
        sorted_locations = sorted(locations)
        result.append(
            {
                "source": source,
                "target": target,
                "edge_family": "data",
                "edge_type": "dataflow",
                "locations": sorted_locations,
                "dependency_count": len(sorted_locations),
            }
        )
    return result


def _normalize_existing_data_edges(raw_edges: Sequence[dict[str, Any]], node_count: int) -> list[dict[str, Any]]:
    pair_locations: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    pair_declared_counts: dict[tuple[int, int], int] = collections.defaultdict(int)
    for edge in raw_edges or []:
        if edge.get("edge_type") != "dataflow" and edge.get("edge_family") != "data":
            continue
        try:
            source, target = int(edge.get("source")), int(edge.get("target"))
        except (TypeError, ValueError):
            continue
        if not (0 <= source < node_count and 0 <= target < node_count):
            continue
        key = (source, target)
        values = edge.get("locations") or []
        if isinstance(values, str):
            values = [values]
        pair_locations[key].update(str(value) for value in values if str(value))
        try:
            pair_declared_counts[key] = max(pair_declared_counts[key], int(edge.get("dependency_count") or 0))
        except (TypeError, ValueError):
            pass
    result: list[dict[str, Any]] = []
    for key in sorted(set(pair_locations) | set(pair_declared_counts)):
        locations = sorted(pair_locations[key])
        count = len(locations) if locations else pair_declared_counts[key]
        result.append(
            {
                "source": key[0],
                "target": key[1],
                "edge_family": "data",
                "edge_type": "dataflow",
                "locations": locations,
                "dependency_count": count,
            }
        )
    return result


def normalize_graph(row: dict[str, Any], *, generate_dfg_if_missing: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    raw_cfg = row.get("cfg") or []
    if not isinstance(raw_cfg, list):
        raw_cfg = []
    blocks: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_cfg):
        raw = raw if isinstance(raw, dict) else {}
        instructions = [normalize_instruction(str(x)) for x in (raw.get("instructions") or []) if str(x).strip()]
        blocks.append(
            {
                "id": index,
                "label": str(raw.get("label") or f"block_{index}"),
                "start_address": str(raw.get("start_address") or ""),
                "instructions": instructions,
                "instruction_count": len(instructions),
                "block_type": str(raw.get("block_type") or "basic_block"),
                "predecessors": [],
                "successors": [],
                "edge_types": [],
            }
        )

    if not blocks:
        canonical = canonicalize_assembly(str(row.get("assembly") or ""))
        instructions = [line for line in canonical.splitlines() if line]
        blocks = [
            {
                "id": 0,
                "label": "block_0",
                "start_address": "",
                "instructions": instructions,
                "instruction_count": len(instructions),
                "block_type": "single_block_fallback",
                "predecessors": [],
                "successors": [],
                "edge_types": [],
            }
        ]

    raw_edges = row.get("edges") or []
    if not isinstance(raw_edges, list):
        raw_edges = []
    control = _control_edges(raw_edges, len(blocks))
    data = _normalize_existing_data_edges(raw_edges, len(blocks))
    dfg_origin = "source_graph_v2" if data else "generated_reaching_definitions"
    if not data and generate_dfg_if_missing:
        data = build_enriched_dfg(blocks, control)

    for edge in control:
        s, t = edge["source"], edge["target"]
        blocks[s]["successors"].append(t)
        blocks[s]["edge_types"].append(edge["edge_type"])
        blocks[t]["predecessors"].append(s)
    for block in blocks:
        block["predecessors"] = sorted(set(block["predecessors"]))
        block["successors"] = sorted(set(block["successors"]))
        block["edge_types"] = sorted(set(block["edge_types"]))

    edges = control + data
    edges.sort(key=lambda e: (0 if e["edge_family"] == "control" else 1, e["source"], e["target"], e["edge_type"], e["locations"]))

    all_in_range = all(0 <= e["source"] < len(blocks) and 0 <= e["target"] < len(blocks) for e in edges)
    all_nonempty = all(bool(block["instructions"]) for block in blocks)
    integrity = {
        "valid": bool(blocks) and all_in_range and all_nonempty,
        "entry_block": 0,
        "node_count": len(blocks),
        "control_edge_count": len(control),
        "dataflow_edge_count": len(data),
        "all_edges_in_range": all_in_range,
        "all_blocks_nonempty": all_nonempty,
        "source_integrity": row.get("integrity") if isinstance(row.get("integrity"), dict) else {},
    }
    return blocks, edges, integrity, dfg_origin


# ---------------------------------------------------------------------------
# Test generation and static validation
# ---------------------------------------------------------------------------

def balanced_dart(text: str) -> tuple[bool, str | None]:
    stack: list[tuple[str, int]] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    quote: str | None = None
    triple = False
    raw = False
    line_comment = False
    block_depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch in "\r\n":
                line_comment = False
            i += 1
            continue
        if block_depth:
            if ch == "/" and nxt == "*":
                block_depth += 1
                i += 2
            elif ch == "*" and nxt == "/":
                block_depth -= 1
                i += 2
            else:
                i += 1
            continue
        if quote:
            if triple and text.startswith(quote * 3, i):
                i += 3
                quote = None
                triple = False
                raw = False
                continue
            if not triple and ch == quote:
                i += 1
                quote = None
                raw = False
                continue
            if ch == "\\" and not raw:
                i += 2
            else:
                i += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_depth = 1
            i += 2
            continue
        if ch in "rR" and i + 1 < len(text) and text[i + 1] in "'\"":
            q = text[i + 1]
            if text.startswith(q * 3, i + 1):
                quote, triple, raw = q, True, True
                i += 4
            else:
                quote, triple, raw = q, False, True
                i += 2
            continue
        if ch in "'\"":
            quote = ch
            triple = text.startswith(ch * 3, i)
            raw = False
            i += 3 if triple else 1
            continue
        if ch in pairs:
            stack.append((pairs[ch], i))
        elif ch in ")]}":
            if not stack or stack[-1][0] != ch:
                return False, f"unmatched {ch} at offset {i}"
            stack.pop()
        i += 1
    if quote:
        return False, "unterminated string"
    if block_depth:
        return False, "unterminated block comment"
    if stack:
        return False, f"unclosed delimiter expecting {stack[-1][0]}"
    return True, None


def count_harness_cases(harness: str) -> int:
    # Exclude helper declaration `void expect(` and count invocation sites.
    return len(re.findall(r"(?<!\bvoid\s)(?<!\bFuture\s)\bexpect\s*\(", harness))


def extract_calls_from_prints(source: str, target: str, limit: int = 8) -> list[str]:
    name = simple_function_name(target)
    calls: list[str] = []
    # Capture balanced target(...) expressions, then retain those close to a
    # print/assert context. Recursive calls inside the function are excluded.
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    for match in pattern.finditer(source):
        prefix = source[max(0, match.start() - 120) : match.start()]
        if not re.search(r"(?:print|assert|expect)\s*\([^)]*$", prefix, re.S):
            continue
        start = match.start()
        open_pos = source.find("(", start)
        depth = 0
        quote: str | None = None
        escape = False
        end = None
        for i in range(open_pos, len(source)):
            ch = source[i]
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in "'\"":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            call = re.sub(r"\s+", " ", source[start:end].strip())
            if call not in calls:
                calls.append(call)
        if len(calls) >= limit:
            break
    return calls


def parse_signature_parameters(signature: str, target: str) -> list[dict[str, Any]]:
    open_pos = signature.find("(")
    close_pos = signature.rfind(")")
    if open_pos < 0 or close_pos < open_pos:
        return []
    text = signature[open_pos + 1 : close_pos].strip()
    params: list[dict[str, Any]] = []
    named = False
    optional = False
    for raw_part in split_top_level(text):
        part = raw_part.strip()
        if part.startswith("{"):
            named = True
            part = part[1:].strip()
        if part.startswith("["):
            optional = True
            part = part[1:].strip()
        if part.endswith("}"):
            part = part[:-1].strip()
        if part.endswith("]"):
            part = part[:-1].strip()
        if not part:
            continue
        part = re.sub(r"^(?:required|covariant|final|var)\s+", "", part)
        default = None
        pieces = split_top_level(part, delimiter="=")
        if len(pieces) >= 2:
            part = pieces[0].strip()
            default = "=".join(pieces[1:]).strip()
        tokens = part.split()
        if not tokens:
            continue
        name = tokens[-1].rstrip("?")
        type_text = " ".join(tokens[:-1]).strip() or "dynamic"
        # Function-typed parameter can leave the name inside a declaration.
        if "(" in name or ")" in name:
            name = f"arg{len(params)}"
        params.append(
            {
                "name": name,
                "type": type_text,
                "named": named,
                "optional": optional or named,
                "default": default,
            }
        )
    return params


def dart_values_for_type(type_text: str, name: str) -> list[str]:
    t = re.sub(r"\s+", "", type_text).replace("?", "")
    low_name = name.lower()
    if "Function" in t or re.search(r"\bcallback|\bfunc|^f$", low_name):
        return ["(num t, num y) => y - t * t + 1", "(dynamic x) => x"]
    if t in {"int", "dynamic", "var", "Object", "Object?"}:
        if any(key in low_name for key in ("string", "text", "word", "name", "isin")):
            return ["''", "'a'", "'abc'", "'racecar'"]
        return ["0", "1", "2", "5", "10", "-1"]
    if "BigInt" in t:
        return ["BigInt.zero", "BigInt.one", "BigInt.from(2)", "BigInt.from(10)"]
    if t in {"num", "double"} or "double" in t or "num" in t:
        return ["0", "1", "-1", "2.5", "10"]
    if "bool" in t:
        return ["false", "true"]
    if "String" in t:
        return ["''", "'a'", "'abc'", "'racecar'", "'hello world'"]
    if "List<String>" in t or "Iterable<String>" in t:
        return ["<String>[]", "['a']", "['a','b','c']", "['10','2','3']"]
    if "List<int>" in t or "Iterable<int>" in t:
        return ["<int>[]", "[0]", "[1,2,3]", "[-1,0,1]", "[5,3,5,2]"]
    if "List<num>" in t or "Iterable<num>" in t or t.startswith("List<") or t.startswith("Iterable<"):
        return ["<num>[]", "[0]", "[1,2,3]", "[-1,0,1]"]
    if t.startswith("Set<"):
        return ["{}", "{0}", "{1,2,3}"]
    if t.startswith("Map<") or t == "Map":
        return ["{}", "{0: 1}", "{1: 2, 2: 3}"]
    if "Duration" in t or "duration" in low_name:
        return ["Duration.zero", "const Duration(milliseconds: 1)", "const Duration(seconds: 1)"]
    return ["0", "1"]


def generated_function_calls(signature: str, target: str, source: str, limit: int = 8) -> list[dict[str, str]]:
    calls = [{"call": call, "origin": "source_example"} for call in extract_calls_from_prints(source, target, limit)]
    params = parse_signature_parameters(signature, target)
    name = simple_function_name(target)
    if params:
        values = [dart_values_for_type(p["type"], p["name"]) for p in params]
        max_cases = min(limit, max(len(v) for v in values))
        for case_index in range(max_cases):
            args: list[str] = []
            for p, candidates in zip(params, values):
                # Leave optional parameters at their defaults in the first case.
                if case_index == 0 and p["optional"] and p["default"] is not None:
                    continue
                value = candidates[case_index % len(candidates)]
                args.append(f"{p['name']}: {value}" if p["named"] else value)
            call = f"{name}({', '.join(args)})"
            if all(existing["call"] != call for existing in calls):
                calls.append({"call": call, "origin": "signature_boundary_generation"})
            if len(calls) >= limit:
                break
    return calls[:limit]


def generated_stdin_cases(source: str) -> list[dict[str, Any]]:
    read_count = max(
        len(re.findall(r"readLineSync\s*\(", source)),
        len(re.findall(r"\bstdin\.readLineSync\s*\(", source)),
    )
    if read_count == 0:
        return [{"stdin": "", "origin": "no_stdin", "timeout_ms": 5000}]
    # Candidate cases are filtered by the runtime validator: a case is accepted
    # only if the reference terminates reproducibly.
    line_count = max(16, read_count * 4)
    return [
        {"stdin": "\n".join(["0"] * line_count) + "\n", "origin": "zero_boundary", "timeout_ms": 5000},
        {"stdin": "\n".join(["1"] * line_count) + "\n", "origin": "one_boundary", "timeout_ms": 5000},
        {"stdin": "\n".join(str((i % 9) + 1) for i in range(line_count)) + "\n", "origin": "small_positive_sequence", "timeout_ms": 5000},
    ]


def validate_existing_harness(harness: str, source: str, target: str) -> dict[str, Any]:
    balanced, error = balanced_dart(harness)
    simple = simple_function_name(target)
    checks = {
        "nonempty": bool(harness.strip()),
        "balanced_delimiters": balanced,
        "has_test_main": bool(re.search(r"\bmain\s*\(", harness)),
        "references_target": bool(re.search(rf"\b{re.escape(simple)}\b", harness)),
        "has_assertion_mechanism": bool(re.search(r"\b(?:expect|assert)\s*\(", harness)),
        "source_defines_target": contains_declaration(source, target),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "error": error,
    }


def build_tests(row: dict[str, Any], source: str, target: str, signature: str) -> dict[str, Any]:
    existing = row.get("tests")
    characteristics = source_characteristics(source)
    runtime = {
        "status": "not_run",
        "reason": "dart_sdk_unavailable_in_build_environment",
        "validator": "validate_master_dart_tests.py",
    }
    if isinstance(existing, str) and existing.strip():
        static = validate_existing_harness(existing, source, target)
        return {
            "schema": TEST_SCHEMA,
            "kind": "dart_harness",
            "origin": "provided",
            "harness": existing,
            "cases": [],
            "case_count": count_harness_cases(existing),
            "comparison": {"mode": "assertion_harness"},
            "deterministic": not any(characteristics.values()),
            "rewrite_rules": [],
            "validation": {"static": static, "runtime": runtime},
        }

    simple = simple_function_name(target)
    if simple != "main" and contains_declaration(source, target):
        cases = generated_function_calls(signature, target, source)
        if cases:
            checks = {
                "source_defines_target": True,
                "case_count_positive": len(cases) > 0,
                "calls_reference_target": all(re.search(rf"\b{re.escape(simple)}\s*\(", c["call"]) for c in cases),
                "calls_balanced": all(balanced_dart(c["call"])[0] for c in cases),
            }
            return {
                "schema": TEST_SCHEMA,
                "kind": "differential_function",
                "origin": "generated",
                "harness": None,
                "cases": cases,
                "case_count": len(cases),
                "comparison": {
                    "mode": "reference_oracle",
                    "return_value": "deep_json_or_string",
                    "exceptions": "runtime_type_and_message",
                },
                "deterministic": not (characteristics["uses_random"] or characteristics["uses_time"]),
                "rewrite_rules": ([{"pattern": "Random()", "replacement": "Random(0)", "scope": "reference_and_candidate"}] if characteristics["uses_random"] else []),
                "validation": {
                    "static": {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "error": None},
                    "runtime": runtime,
                },
            }

    # Whole-program oracle is the safe fallback for main targets, unusual
    # signatures, class methods, and sources whose behavior is expressed through
    # their own main function.
    cases = generated_stdin_cases(source)
    checks = {
        "source_has_main": has_main(source),
        "case_count_positive": len(cases) > 0,
        "stdin_cases_are_strings": all(isinstance(c.get("stdin"), str) for c in cases),
        "timeouts_positive": all(int(c.get("timeout_ms") or 0) > 0 for c in cases),
    }
    return {
        "schema": TEST_SCHEMA,
        "kind": "differential_program",
        "origin": "generated",
        "harness": None,
        "cases": cases,
        "case_count": len(cases),
        "comparison": {
            "mode": "reference_oracle",
            "stdout": "exact_after_crlf_normalization",
            "stderr": "exact_after_crlf_normalization",
            "exit_code": "exact",
        },
        "deterministic": not (characteristics["uses_random"] or characteristics["uses_time"]),
        "rewrite_rules": ([{"pattern": "Random()", "replacement": "Random(0)", "scope": "reference_and_candidate"}] if characteristics["uses_random"] else []),
        "validation": {
            "static": {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "error": None},
            "runtime": runtime,
        },
    }


# ---------------------------------------------------------------------------
# Row conversion, deduplication, and audit
# ---------------------------------------------------------------------------

def small_metadata(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "assembly", "source", "dart_source", "cfg", "edges", "integrity", "tests", "graph_v2",
        "assembly_rebuild", "language", "lang", "function", "name", "dart_function_signature",
    }
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        result[key] = value
    if isinstance(row.get("assembly_rebuild"), dict):
        result["assembly_rebuild"] = row["assembly_rebuild"]
    return result


def canonical_record(row: dict[str, Any], *, split: str, source_dataset: str, source_index: int) -> dict[str, Any]:
    source = source_of(row)
    target = function_of(row)
    signature = str(row.get("dart_function_signature") or infer_signature(source, target))
    blocks, edges, integrity, dfg_origin = normalize_graph(row)
    assembly = str(row.get("assembly") or "")
    source_hash = source_fingerprint(source)
    assembly_hash = assembly_fingerprint(assembly, blocks)
    graph_hash = sha256_text(json_compact({"cfg": blocks, "edges": edges}))
    tests = build_tests(row, source, target, signature)
    framework = framework_of(source, target)
    record_id = f"{split}_{assembly_hash[:16]}_{source_hash[:10]}"

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "id": record_id,
        "split": split,
        "language": "Dart",
        "framework": framework,
        "function": target,
        "dart_function_signature": signature,
        "dart_source": source,
        "assembly": assembly,
        "cfg": blocks,
        "edges": edges,
        "graph_v2": {
            "schema": GRAPH_SCHEMA,
            "node_granularity": "basic_block",
            "edge_families": ["control", "data"],
            "entry_block": 0,
            "dataflow_origin": dfg_origin,
            "build_version": BUILD_VERSION,
        },
        "integrity": integrity,
        "tests": tests,
        "fingerprints": {
            "source_sha256": source_hash,
            "assembly_sha256": assembly_hash,
            "graph_sha256": graph_hash,
        },
        "provenance": {
            "source_dataset": source_dataset,
            "source_index": source_index,
            "source_filename": row.get("filename"),
            "source_task_id": row.get("task_id"),
            "source_graph_schema": (row.get("graph_v2") or {}).get("schema") if isinstance(row.get("graph_v2"), dict) else None,
            "source_compile_evidence": "assembly_present_and_graph_extracted" if assembly.strip() else "missing_assembly",
        },
        "metadata": small_metadata(row),
    }
    if isinstance(row.get("reasoning"), str) and row["reasoning"].strip():
        record["reasoning"] = row["reasoning"]
    return record


def quality_score(record: dict[str, Any]) -> tuple[int, int, int, int, str]:
    tests = record["tests"]
    characteristics = source_characteristics(record["dart_source"])
    return (
        100 if tests["validation"]["static"]["status"] == "passed" else 0,
        50 if tests["origin"] == "provided" else 0,
        20 if record["integrity"]["valid"] else 0,
        10 if not (characteristics["uses_random"] or characteristics["uses_time"] or characteristics["uses_ffi"]) else 0,
        record["fingerprints"]["source_sha256"],
    )


def deduplicate_records(records: Sequence[dict[str, Any]], *, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_assembly: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_assembly[record["fingerprints"]["assembly_sha256"]].append(record)

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    duplicate_groups = 0
    conflicting_groups = 0
    for assembly_hash, group in sorted(by_assembly.items()):
        if len(group) == 1:
            kept.append(group[0])
            continue
        duplicate_groups += 1
        source_hashes = {r["fingerprints"]["source_sha256"] for r in group}
        if len(source_hashes) > 1:
            conflicting_groups += 1
        winner = max(group, key=quality_score)
        kept.append(winner)
        for loser in group:
            if loser is winner:
                continue
            rejected.append(
                {
                    "schema": "antigravity-master-rejection-v1",
                    "split": split,
                    "reason": "duplicate_assembly_input_conflicting_target" if len(source_hashes) > 1 else "duplicate_assembly_input",
                    "assembly_sha256": assembly_hash,
                    "kept_id": winner["id"],
                    "rejected_id": loser["id"],
                    "kept_source_sha256": winner["fingerprints"]["source_sha256"],
                    "rejected_source_sha256": loser["fingerprints"]["source_sha256"],
                    "kept_provenance": winner["provenance"],
                    "rejected_provenance": loser["provenance"],
                }
            )

    # Second guard against exact source duplicates that somehow compiled to
    # different dump text. Keep the stronger record deterministically.
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in kept:
        by_source[record["fingerprints"]["source_sha256"]].append(record)
    final: list[dict[str, Any]] = []
    source_duplicate_groups = 0
    for source_hash, group in sorted(by_source.items()):
        if len(group) == 1:
            final.append(group[0])
            continue
        source_duplicate_groups += 1
        winner = max(group, key=quality_score)
        final.append(winner)
        for loser in group:
            if loser is winner:
                continue
            rejected.append(
                {
                    "schema": "antigravity-master-rejection-v1",
                    "split": split,
                    "reason": "duplicate_source_target",
                    "source_sha256": source_hash,
                    "kept_id": winner["id"],
                    "rejected_id": loser["id"],
                    "kept_assembly_sha256": winner["fingerprints"]["assembly_sha256"],
                    "rejected_assembly_sha256": loser["fingerprints"]["assembly_sha256"],
                    "kept_provenance": winner["provenance"],
                    "rejected_provenance": loser["provenance"],
                }
            )

    final.sort(key=lambda r: (r["provenance"]["source_dataset"], r["provenance"]["source_index"], r["id"]))
    stats = {
        "input_rows": len(records),
        "output_rows": len(final),
        "rejected_rows": len(rejected),
        "assembly_duplicate_groups": duplicate_groups,
        "assembly_conflicting_target_groups": conflicting_groups,
        "source_duplicate_groups": source_duplicate_groups,
    }
    return final, rejected, stats


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema") != SCHEMA:
        errors.append("wrong schema")
    cfg = record.get("cfg")
    edges = record.get("edges")
    if not isinstance(cfg, list) or not cfg:
        errors.append("empty cfg")
        return errors
    if not isinstance(edges, list):
        errors.append("edges is not a list")
        return errors
    n = len(cfg)
    for i, block in enumerate(cfg):
        if block.get("id") != i:
            errors.append(f"block id mismatch at {i}")
        if not isinstance(block.get("instructions"), list) or not block.get("instructions"):
            errors.append(f"empty block {i}")
    for i, edge in enumerate(edges):
        if set(edge) != {"source", "target", "edge_family", "edge_type", "locations", "dependency_count"}:
            errors.append(f"edge {i} schema mismatch")
        if not (0 <= edge.get("source", -1) < n and 0 <= edge.get("target", -1) < n):
            errors.append(f"edge {i} out of range")
        if edge.get("edge_family") not in {"control", "data"}:
            errors.append(f"edge {i} invalid family")
        if not isinstance(edge.get("locations"), list):
            errors.append(f"edge {i} locations not list")
    tests = record.get("tests")
    if not isinstance(tests, dict) or tests.get("schema") != TEST_SCHEMA:
        errors.append("test schema mismatch")
    elif tests.get("validation", {}).get("static", {}).get("status") != "passed":
        errors.append("static test validation failed")
    return errors


def aggregate_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    graph_nodes = [len(r["cfg"]) for r in records]
    control = [sum(e["edge_family"] == "control" for e in r["edges"]) for r in records]
    data = [sum(e["edge_family"] == "data" for e in r["edges"]) for r in records]
    test_kinds = collections.Counter(r["tests"]["kind"] for r in records)
    test_origins = collections.Counter(r["tests"]["origin"] for r in records)
    static_status = collections.Counter(r["tests"]["validation"]["static"]["status"] for r in records)
    runtime_status = collections.Counter(r["tests"]["validation"]["runtime"]["status"] for r in records)
    frameworks = collections.Counter(r["framework"] for r in records)
    datasets = collections.Counter(r["provenance"]["source_dataset"] for r in records)

    def describe(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"min": 0, "median": 0, "max": 0, "mean": 0}
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }

    return {
        "rows": len(records),
        "source_datasets": dict(datasets),
        "frameworks": dict(frameworks),
        "graph": {
            "nodes": describe(graph_nodes),
            "control_edges": describe(control),
            "dataflow_edges": describe(data),
            "total_nodes": sum(graph_nodes),
            "total_control_edges": sum(control),
            "total_dataflow_edges": sum(data),
        },
        "tests": {
            "kinds": dict(test_kinds),
            "origins": dict(test_origins),
            "static_status": dict(static_status),
            "runtime_status": dict(runtime_status),
            "total_cases": sum(int(r["tests"].get("case_count") or 0) for r in records),
        },
    }


def output_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "dart_all_graphv2_train": args.dart_train,
        "synthetic_pool_reward_clean_graphv2": args.synthetic_train,
        "test_set": args.test_set,
        "test_set_cfg": args.test_cfg,
        "test_set_cfg_clean": args.test_cfg_clean,
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    train_raw: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(args.dart_train)):
        train_raw.append(canonical_record(row, split="train", source_dataset="dart_all_graphv2_train", source_index=index))
    for index, row in enumerate(iter_jsonl(args.synthetic_train)):
        train_raw.append(canonical_record(row, split="train", source_dataset="synthetic_pool_reward_clean_graphv2", source_index=index))

    # The three test files are alternate representations of the same 165 rows.
    # cfg_clean has exactly the same graph topology as cfg but normalized dump
    # text, so it is the canonical held-out source.
    heldout_raw = [
        canonical_record(row, split="test", source_dataset="test_set_cfg_clean", source_index=index)
        for index, row in enumerate(iter_jsonl(args.test_cfg_clean))
    ]

    train, rejected_train, train_dedup = deduplicate_records(train_raw, split="train")
    heldout, rejected_test, test_dedup = deduplicate_records(heldout_raw, split="test")

    # Strict inclusion gate: a record must have a valid canonical graph and a
    # statically valid test suite. Unsupported isolated snippets are quarantined
    # rather than silently entering training/evaluation.
    validation_errors: list[dict[str, Any]] = []
    quality_rejections: list[dict[str, Any]] = []

    def apply_quality_gate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        for record in records:
            errors = validate_record(record)
            if not errors:
                accepted.append(record)
                continue
            item = {"id": record["id"], "split": record["split"], "errors": errors}
            validation_errors.append(item)
            quality_rejections.append({
                "schema": "antigravity-master-rejection-v1",
                "split": record["split"],
                "reason": "canonical_graph_or_test_static_validation_failed",
                "rejected_id": record["id"],
                "errors": errors,
                "provenance": record["provenance"],
                "fingerprints": record["fingerprints"],
                "test_kind": record.get("tests", {}).get("kind"),
                "test_static_validation": record.get("tests", {}).get("validation", {}).get("static"),
            })
        return accepted

    train_before_quality = len(train)
    heldout_before_quality = len(heldout)
    train = apply_quality_gate(train)
    heldout = apply_quality_gate(heldout)
    quality_filter = {
        "train_input_rows": train_before_quality,
        "train_output_rows": len(train),
        "train_rejected_rows": train_before_quality - len(train),
        "heldout_input_rows": heldout_before_quality,
        "heldout_output_rows": len(heldout),
        "heldout_rejected_rows": heldout_before_quality - len(heldout),
    }

    train_source = {r["fingerprints"]["source_sha256"] for r in train}
    test_source = {r["fingerprints"]["source_sha256"] for r in heldout}
    train_assembly = {r["fingerprints"]["assembly_sha256"] for r in train}
    test_assembly = {r["fingerprints"]["assembly_sha256"] for r in heldout}
    leakage = {
        "source_overlap_count": len(train_source & test_source),
        "assembly_overlap_count": len(train_assembly & test_assembly),
        "source_overlap_sha256": sorted(train_source & test_source),
        "assembly_overlap_sha256": sorted(train_assembly & test_assembly),
    }
    if leakage["source_overlap_count"] or leakage["assembly_overlap_count"]:
        raise RuntimeError(f"Train/test leakage detected: {leakage}")

    train_path = output_dir / "master_dart_cfg_dfg_train.jsonl"
    heldout_path = output_dir / "master_dart_cfg_dfg_heldout.jsonl"
    rejected_path = output_dir / "master_dart_cfg_dfg_rejected.jsonl"
    sample_path = output_dir / "master_dart_cfg_dfg_sample.jsonl"
    manifest_path = output_dir / "master_dart_cfg_dfg_manifest.json"
    readme_path = output_dir / "README_master_dart_cfg_dfg.md"

    write_jsonl(train_path, train)
    write_jsonl(heldout_path, heldout)
    write_jsonl(rejected_path, rejected_train + rejected_test + quality_rejections)
    sample_rows = train[:5] + heldout[:2]
    write_jsonl(sample_path, sample_rows)

    train_gz = train_path.with_suffix(train_path.suffix + ".gz")
    heldout_gz = heldout_path.with_suffix(heldout_path.suffix + ".gz")
    gzip_file(train_path, train_gz)
    gzip_file(heldout_path, heldout_gz)

    dart_path = shutil.which("dart")
    flutter_path = shutil.which("flutter")
    manifest: dict[str, Any] = {
        "schema": "antigravity-master-dataset-manifest-v1",
        "dataset_schema": SCHEMA,
        "graph_schema": GRAPH_SCHEMA,
        "test_schema": TEST_SCHEMA,
        "build_version": BUILD_VERSION,
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            label: {
                "path": str(path),
                "rows": sum(1 for _ in iter_jsonl(path)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for label, path in input_paths.items()
        },
        "split_policy": {
            "training_inputs": ["dart_all_graphv2_train", "synthetic_pool_reward_clean_graphv2"],
            "heldout_inputs": ["test_set", "test_set_cfg", "test_set_cfg_clean"],
            "test_variants_collapsed": True,
            "test_variant_selected": "test_set_cfg_clean",
            "test_rows_are_not_in_training": True,
        },
        "deduplication": {
            "primary_key": "normalized_assembly_sha256",
            "secondary_key": "comment_and_whitespace_insensitive_source_sha256",
            "train": train_dedup,
            "heldout": test_dedup,
        },
        "leakage": leakage,
        "quality_filter": quality_filter,
        "train": aggregate_stats(train),
        "heldout": aggregate_stats(heldout),
        "validation": {
            "canonical_record_errors_quarantined": len(validation_errors),
            "dart_executable": dart_path,
            "flutter_executable": flutter_path,
            "runtime_tests_executed": False,
            "runtime_tests_reason": "No dart/flutter executable was present in this build environment.",
            "static_tests_required_for_inclusion": True,
            "source_compile_evidence": "Each retained input includes assembly and a parsed graph derived from its source program.",
        },
        "outputs": {},
        "notes": [
            "The files named test-set are held out; they are never merged into master_dart_cfg_dfg_train.jsonl.",
            "Provided Dart harnesses are preserved inside tests.harness.",
            "Missing tests are represented as generated differential function/program oracle specifications.",
            "Runtime status remains not_run until validate_master_dart_tests.py is executed where the Dart SDK is installed.",
        ],
    }

    readme = f"""# Master Dart/Flutter CFG+DFG Dataset

## Outputs

- `master_dart_cfg_dfg_train.jsonl`: deduplicated training split.
- `master_dart_cfg_dfg_heldout.jsonl`: deduplicated held-out split built from the three test-set variants.
- `master_dart_cfg_dfg_rejected.jsonl`: duplicate/conflict decisions with kept/rejected IDs.
- `master_dart_cfg_dfg_manifest.json`: input/output hashes, counts, leakage checks, graph and test statistics.
- `master_dart_cfg_dfg_sample.jsonl`: seven-row inspection sample.

## Canonical row schema

Every row uses `{SCHEMA}` and contains:

- `assembly`: original AOT disassembly text.
- `dart_source`: target Dart/Flutter source.
- `cfg`: normalized basic blocks with stable positional IDs.
- `edges`: one fixed edge schema. `edge_family=control` represents CFG edges; `edge_family=data` and `edge_type=dataflow` represent reaching-definition dependencies. Data edges retain `locations` and `dependency_count`.
- `tests`: one fixed `{TEST_SCHEMA}` object. Existing harnesses use `kind=dart_harness`; generated suites use differential reference oracles.
- `fingerprints`: normalized source, assembly, and graph SHA-256 values.
- `provenance`: original dataset and row identity.

## Deduplication and split safety

Training rows are deduplicated first by normalized assembly input and then by normalized source. Same-assembly/different-source groups are treated as conflicting labels: one deterministic best-quality representative is retained and every removed row is recorded in the rejection file. The held-out test files are treated as three representations of the same tasks, not three additional training datasets.

## Test validation

The build performs structural/static validation for every suite. The current execution environment does not contain a Dart or Flutter executable, so runtime validation is deliberately recorded as `not_run`, never as passed. Run:

```bash
python validate_master_dart_tests.py \\
  --input master_dart_cfg_dfg_train.jsonl \\
  --output master_dart_cfg_dfg_train.runtime_validated.jsonl \\
  --dart /path/to/dart
```

The validator executes provided harnesses and differential reference oracles, checks determinism, filters invalid generated stdin cases, and writes per-row runtime results.
"""
    readme_path.write_text(readme, encoding="utf-8")

    output_files = [train_path, train_gz, heldout_path, heldout_gz, rejected_path, sample_path, readme_path]
    manifest["outputs"] = {path.name: output_descriptor(path) for path in output_files}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dart-train", type=Path, default=Path("/mnt/data/dart_all_graphv2_train.jsonl"))
    parser.add_argument("--synthetic-train", type=Path, default=Path("/mnt/data/synthetic_pool_reward_clean_graphv2.jsonl"))
    parser.add_argument("--test-set", type=Path, default=Path("/mnt/data/test-set.jsonl"))
    parser.add_argument("--test-cfg", type=Path, default=Path("/mnt/data/test-set_cfg.jsonl"))
    parser.add_argument("--test-cfg-clean", type=Path, default=Path("/mnt/data/test-set_cfg_clean.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/master_dart_cfg_dfg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(json.dumps({
        "train_rows": manifest["train"]["rows"],
        "heldout_rows": manifest["heldout"]["rows"],
        "train_dedup": manifest["deduplication"]["train"],
        "heldout_dedup": manifest["deduplication"]["heldout"],
        "leakage": manifest["leakage"],
        "test_static_status": manifest["train"]["tests"]["static_status"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
