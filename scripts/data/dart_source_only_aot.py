#!/usr/bin/env python3
"""Prepare a neutral, test-free Dart source for AOT binary extraction.

This module is private-side build infrastructure.  It may read a reference
source in order to compile it, but it never extracts literals or constructs
model inputs from source text.  Public pool metadata must be recovered later
from the resulting AOT artifact alone.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Iterable

from tree_sitter import Language, Parser
import tree_sitter_dart


DART_LANGUAGE = Language(tree_sitter_dart.language())
TARGET = "candidate"
REQUIRED_IMPORTS = ("dart:async", "dart:convert")


@dataclasses.dataclass(frozen=True)
class FunctionDecl:
    name: str
    full_start: int
    signature_start: int
    body_end: int


def _parser() -> Parser:
    return Parser(DART_LANGUAGE)


def _walk(node) -> Iterable:
    yield node
    for child in node.children:
        yield from _walk(child)


def _top_level_functions(source: str, parser: Parser) -> tuple[list[FunctionDecl], object]:
    blob = source.encode("utf-8")
    tree = parser.parse(blob)
    root = tree.root_node
    if root.has_error:
        raise ValueError("dart_source_parse_error")
    children = root.children
    result: list[FunctionDecl] = []
    for index, node in enumerate(children):
        if node.type != "function_signature":
            continue
        body = (
            children[index + 1]
            if index + 1 < len(children) and children[index + 1].type == "function_body"
            else None
        )
        if body is None:
            continue
        identifiers = [child for child in node.children if child.type == "identifier"]
        if not identifiers:
            continue
        identifier = identifiers[-1]
        name = blob[identifier.start_byte : identifier.end_byte].decode("utf-8")
        full_start = node.start_byte
        previous = index - 1
        while previous >= 0 and children[previous].type == "annotation":
            full_start = children[previous].start_byte
            previous -= 1
        result.append(
            FunctionDecl(
                name=name,
                full_start=full_start,
                signature_start=node.start_byte,
                body_end=body.end_byte,
            )
        )
    return result, tree


def _apply_edits(source: str, edits: list[tuple[int, int, str]]) -> str:
    blob = source.encode("utf-8")
    unique = sorted(set(edits), key=lambda item: (item[0], item[1]), reverse=True)
    last_start = len(blob) + 1
    for start, end, replacement in unique:
        if end > last_start:
            continue
        blob = blob[:start] + replacement.encode("utf-8") + blob[end:]
        last_start = start
    return blob.decode("utf-8")


def function_only_candidate(source: str) -> tuple[str, dict[str, int]]:
    """Remove top-level demos/tests and enforce both retention pragmas."""

    if not str(source or "").strip():
        raise ValueError("empty_dart_source")
    parser = _parser()
    functions, tree = _top_level_functions(source, parser)
    candidates = [item for item in functions if item.name == TARGET]
    if len(candidates) != 1:
        raise ValueError(f"expected_one_candidate_function:observed={len(candidates)}")
    mains = [item for item in functions if item.name == "main"]
    removal_ranges = [(item.full_start, item.body_end) for item in mains]

    def inside_removed(start: int, end: int) -> bool:
        return any(start >= left and end <= right for left, right in removal_ranges)

    edits: list[tuple[int, int, str]] = [
        (left, right, "") for left, right in removal_ranges
    ]
    source_blob = source.encode("utf-8")
    for node in _walk(tree.root_node):
        if node.type == "comment" and not inside_removed(node.start_byte, node.end_byte):
            edits.append((node.start_byte, node.end_byte, ""))
    candidate = candidates[0]
    # Replace, rather than stack on top of, existing retention annotations.
    # Keep unrelated annotations intact.
    for node in tree.root_node.children:
        if (
            node.type == "annotation"
            and candidate.full_start <= node.start_byte < candidate.signature_start
        ):
            text = source_blob[node.start_byte : node.end_byte].decode("utf-8")
            if "vm:never-inline" in text or "vm:entry-point" in text:
                edits.append((node.start_byte, node.end_byte, ""))
    edits.append(
        (
            candidate.full_start,
            candidate.full_start,
            "@pragma('vm:never-inline')\n@pragma('vm:entry-point')\n",
        )
    )
    result = _apply_edits(source, edits)
    for pragma in ("vm:never-inline", "vm:entry-point"):
        pattern = re.compile(
            rf"(?:@pragma\(\s*(['\"])%s\1\s*\)\s*){{2,}}" % re.escape(pragma)
        )
        result = pattern.sub(f"@pragma('{pragma}')\n", result)
    result = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", result).strip() + "\n"

    observed, parsed = _top_level_functions(result, parser)
    if parsed.root_node.has_error:
        raise ValueError("transformed_dart_source_parse_error")
    if [item.name for item in observed].count(TARGET) != 1:
        raise ValueError("transformed_candidate_count_mismatch")
    if any(item.name == "main" for item in observed):
        raise ValueError("top_level_main_survived_transform")
    transformed_candidate = next(item for item in observed if item.name == TARGET)
    transformed_blob = result.encode("utf-8")
    candidate_annotations = transformed_blob[
        transformed_candidate.full_start : transformed_candidate.signature_start
    ].decode("utf-8")
    if (
        candidate_annotations.count("vm:never-inline") != 1
        or candidate_annotations.count("vm:entry-point") != 1
    ):
        raise ValueError("retention_pragma_count_mismatch")
    return result, {
        "removed_top_level_main": len(mains),
        "retained_top_level_functions": len(observed),
    }


def _add_imports(source: str) -> str:
    missing = [
        uri
        for uri in REQUIRED_IMPORTS
        if not re.search(rf"^\s*import\s+['\"]{re.escape(uri)}['\"]\s*;", source, re.MULTILINE)
    ]
    if not missing:
        return source
    block = "".join(f"import '{uri}';\n" for uri in missing)
    library = re.match(r"\s*library\s+[^;]+;\s*", source)
    position = library.end() if library else 0
    return source[:position] + block + source[position:]


def declared_symbol_names(source: str) -> dict[str, list[str]]:
    """Collect private-side names used only to scrub disassembler labels."""

    blob = source.encode("utf-8")
    tree = _parser().parse(blob)
    if tree.root_node.has_error:
        raise ValueError("symbol_source_parse_error")
    functions: list[str] = []
    types: list[str] = []

    def append_unique(items: list[str], value: str) -> None:
        if value and value not in items:
            items.append(value)

    for node in _walk(tree.root_node):
        if node.type in ("function_signature", "method_signature"):
            identifiers = [child for child in node.children if child.type == "identifier"]
            if identifiers:
                identifier = identifiers[-1]
                append_unique(
                    functions,
                    blob[identifier.start_byte : identifier.end_byte].decode("utf-8"),
                )
        elif node.type in {
            "class_definition",
            "enum_declaration",
            "extension_declaration",
            "mixin_declaration",
            "type_alias",
        }:
            named = node.child_by_field_name("name")
            candidates = [named] if named is not None else [
                child
                for child in node.children
                if child.type in ("identifier", "type_identifier")
            ]
            if candidates:
                candidate = candidates[0]
                append_unique(
                    types,
                    blob[candidate.start_byte : candidate.end_byte].decode("utf-8"),
                )
    return {"functions": functions, "types": types}


def source_only_program(source: str) -> tuple[str, str, dict[str, Any]]:
    """Return (function-only label, compilable analysis program, metadata)."""

    function_source, metadata = function_only_candidate(source)
    program = _add_imports(function_source.rstrip())
    program = program.rstrip() + "\n\nvoid main() {}\n"
    functions, tree = _top_level_functions(program, _parser())
    if tree.root_node.has_error:
        raise ValueError("analysis_program_parse_error")
    if [item.name for item in functions].count(TARGET) != 1:
        raise ValueError("analysis_program_candidate_count_mismatch")
    if [item.name for item in functions].count("main") != 1:
        raise ValueError("analysis_program_main_count_mismatch")
    metadata = dict(metadata)
    metadata["source_symbols"] = declared_symbol_names(function_source)
    return function_source, program, metadata
