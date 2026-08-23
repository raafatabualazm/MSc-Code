"""Acceptance gates for the paired fixed signature-scrub v3 experiment.

The gate is deliberately independent of inference.  It proves that the two
public policy inputs differ only in the advertised signature contract, that
their frozen CFG evidence still matches the comparator, and that the private
sidecars remain executable under the project's patched aligned-JIT harness.

The four default stub failures and the twelve default reference failures are
pre-existing HumanEval-Dart defects.  They are checked as *exact sets*, not
silently waived.  This preserves comparator semantics and makes the valid-150
sensitivity denominator explicit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPT_SOURCE = (
    ROOT
    / "scripts"
    / "training"
    / "graph_encoder_decoder_decompiler_v2_antigravity.py"
)
EXPECTED_PROMPT_SCHEMA = "antigravity-v3-matched-function-contract"
EXPECTED_DATASET_SCHEMA = "dart-signature-scrubbed-v3"

# These IDs are the original benchmark task_id values, not shuffled row indices.
DEFAULT_STUB_COMPILE_FAILURES = frozenset({"121", "127", "153", "161"})
DEFAULT_REFERENCE_COMPILE_FAILURES = frozenset({"127", "153", "161"})
DEFAULT_REFERENCE_PASS_FAILURES = frozenset(
    {"8", "20", "54", "87", "90", "107", "112", "127", "136", "153", "155", "161"}
)
KNOWN_STALE_TARGET_BINDINGS = {
    "121": "solution",
    "153": "Strongest_Extension",
}

PRIVATE_ONLY_FIELDS = frozenset(
    {"dart_source", "evaluation_only_dart_function_signature", "tests"}
)
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "original_source_sha256",
        "semantic_function_name_sha256",
        "frozen_assembly_sha256",
        "effective_id_salt",
        "id_salt",
    }
)
REQUIRED_NAMEONLY_WITHHOLDS = frozenset(
    {
        "return_type",
        "arity",
        "parameter_count",
        "parameter_types",
        "parameter_names",
        "reference_source",
        "tests",
        "semantic_function_name",
    }
)
SHARED_PROMPT_CONSTRAINTS = (
    "Return only valid source code.",
    "Do not include explanations, markdown fences, test code, or placeholder demos.",
    "Do not replace it with only a void main() demo.",
    "Do not define the required function inside main(); define it at top level.",
)


class GateError(AssertionError):
    """One or more acceptance invariants did not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSONL {path}: {exc}") from exc


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"summary is not a JSON object: {path}")
    return value


def strip_main_and_imports(code: str) -> str:
    """Mirror the builder's source-fingerprint normalization."""
    code = re.sub(r"^import\s+.*;\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^@pragma\(.*\)\s*$", "", code, flags=re.MULTILINE)
    main_match = re.search(r"void\s+main\s*\([^)]*\)\s*\{", code)
    if main_match:
        start = main_match.start()
        depth = 0
        index = main_match.end() - 1
        while index < len(code):
            if code[index] == "{":
                depth += 1
            elif code[index] == "}":
                depth -= 1
                if depth == 0:
                    code = code[:start] + code[index + 1 :]
                    break
            index += 1
    return code.strip()


def original_source_sha256(row: dict[str, Any]) -> str:
    source = str(row.get("dart_source") or row.get("source") or "")
    _require(bool(source.strip()), f"benchmark row {row.get('task_id')} has no Dart source")
    normalized = strip_main_and_imports(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _task_id(row: dict[str, Any]) -> str:
    value = row.get("task_id", row.get("id", row.get("filename", "")))
    return str(value).removesuffix(".dart")


def _semantic_name(row: dict[str, Any]) -> str:
    return str(
        row.get("function")
        or row.get("camel_case_function_name")
        or row.get("name")
        or ""
    )


_DART_NON_FUNCTION_IDENTIFIERS = frozenset(
    {"main", "if", "for", "while", "switch", "catch"}
)
_DART_FUNCTION_DECLARATION_RE = re.compile(
    r"(?m)^\s*"
    r"(?:[A-Za-z_][\w<>,\[\]?]*\s+)+"
    r"([A-Za-z_]\w*)\s*"
    r"\([^;{}]*\)\s*"
    r"(?:async\s*\*?\s*)?"
    r"(?:\{|=>)"
)


def extract_user_function_identifiers(row: dict[str, Any]) -> set[str]:
    """User-defined target/helper names recoverable from private evidence.

    Source declarations catch top-level and nested Dart helpers. The frozen
    target-qualified symbol table is an independent backstop for declaration
    shapes the conservative source regex may miss (for example
    ``target.localHelper.<anonymous closure>``).
    """
    source = str(row.get("dart_source") or row.get("source") or "")
    target = _semantic_name(row)
    names = {
        match.group(1)
        for match in _DART_FUNCTION_DECLARATION_RE.finditer(source)
        if match.group(1) not in _DART_NON_FUNCTION_IDENTIFIERS
    }
    if target:
        names.add(target)
        assembly = str(row.get("assembly") or "")
        names.update(
            match.group(1)
            for match in re.finditer(
                rf"\b{re.escape(target)}\.([A-Za-z_]\w*)\b", assembly
            )
            if match.group(1) not in _DART_NON_FUNCTION_IDENTIFIERS
        )
    return names


def _private_source_hash(row: dict[str, Any]) -> str:
    protocol = row.get("benchmark_protocol")
    _require(isinstance(protocol, dict), f"{_task_id(row)} lacks benchmark_protocol")
    value = str(protocol.get("original_source_sha256") or "")
    _require(
        bool(re.fullmatch(r"[0-9a-f]{64}", value)),
        f"private row {_task_id(row)} lacks a valid original_source_sha256",
    )
    return value


@dataclass(frozen=True)
class Arm:
    label: str
    mode: str
    public: tuple[dict[str, Any], ...]
    private: tuple[dict[str, Any], ...]
    public_by_id: dict[str, dict[str, Any]]
    private_by_id: dict[str, dict[str, Any]]
    public_source_sequence: tuple[str, ...]
    public_by_source: dict[str, dict[str, Any]]
    private_by_source: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Bundle:
    benchmark: tuple[dict[str, Any], ...]
    benchmark_by_source: dict[str, dict[str, Any]]
    nameonly: Arm
    neutralexact: Arm
    expected_rows: int
    target_name: str


def _unique_by(
    rows: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str], label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = key(row)
        _require(bool(value), f"{label} contains an empty key")
        _require(value not in output, f"{label} contains duplicate key {value}")
        output[value] = row
    return output


def _public_projection(private_row: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the builder's v3 private-to-public redaction boundary."""
    projected = json.loads(json.dumps(private_row))
    for field in PRIVATE_ONLY_FIELDS:
        projected.pop(field, None)
    protocol = projected.get("benchmark_protocol")
    if isinstance(protocol, dict):
        protocol.pop("original_source_sha256", None)
        protocol.pop("semantic_function_name_sha256", None)
        build = protocol.get("assembly_build")
        if isinstance(build, dict):
            build.pop("frozen_assembly_sha256", None)
    return projected


def _build_arm(
    label: str,
    mode: str,
    public: Sequence[dict[str, Any]],
    private: Sequence[dict[str, Any]],
    expected_rows: int,
    benchmark_sources: set[str],
) -> Arm:
    _require(
        len(public) == expected_rows,
        f"{label} public rows: expected {expected_rows}, found {len(public)}",
    )
    _require(
        len(private) == expected_rows,
        f"{label} private rows: expected {expected_rows}, found {len(private)}",
    )
    public_by_id = _unique_by(public, _task_id, f"{label} public")
    private_by_id = _unique_by(private, _task_id, f"{label} private")
    _require(
        public_by_id.keys() == private_by_id.keys(),
        f"{label} public/private task-id sets differ",
    )
    for task_id, public_row in public_by_id.items():
        _require(
            public_row == _public_projection(private_by_id[task_id]),
            f"{label} public row {task_id} is not the exact redacted private projection",
        )
    private_by_source = _unique_by(
        private, _private_source_hash, f"{label} private source mapping"
    )
    _require(
        set(private_by_source) == benchmark_sources,
        f"{label} private source mapping is not a permutation of the benchmark",
    )
    public_sequence = tuple(
        _private_source_hash(private_by_id[_task_id(row)]) for row in public
    )
    public_by_source = {
        _private_source_hash(private_by_id[task_id]): row
        for task_id, row in public_by_id.items()
    }
    return Arm(
        label=label,
        mode=mode,
        public=tuple(public),
        private=tuple(private),
        public_by_id=public_by_id,
        private_by_id=private_by_id,
        public_source_sequence=public_sequence,
        public_by_source=public_by_source,
        private_by_source=private_by_source,
    )


def build_bundle(
    *,
    nameonly_public: Sequence[dict[str, Any]],
    nameonly_private: Sequence[dict[str, Any]],
    neutralexact_public: Sequence[dict[str, Any]],
    neutralexact_private: Sequence[dict[str, Any]],
    benchmark: Sequence[dict[str, Any]],
    expected_rows: int = 154,
    target_name: str = "fn0",
) -> Bundle:
    _require(
        bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target_name)),
        f"invalid Dart target name: {target_name!r}",
    )
    _require(
        len(benchmark) == expected_rows,
        f"benchmark rows: expected {expected_rows}, found {len(benchmark)}",
    )
    benchmark_by_source = _unique_by(
        benchmark, original_source_sha256, "benchmark source mapping"
    )
    sources = set(benchmark_by_source)
    nameonly = _build_arm(
        "name-only", "name_only", nameonly_public, nameonly_private, expected_rows, sources
    )
    neutralexact = _build_arm(
        "neutral-exact",
        "neutral_exact",
        neutralexact_public,
        neutralexact_private,
        expected_rows,
        sources,
    )
    _require(
        nameonly.public_source_sequence == neutralexact.public_source_sequence,
        "paired public row order maps to different original-source sequences",
    )
    return Bundle(
        benchmark=tuple(benchmark),
        benchmark_by_source=benchmark_by_source,
        nameonly=nameonly,
        neutralexact=neutralexact,
        expected_rows=expected_rows,
        target_name=target_name,
    )


def _walk_items(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            yield child_path, child
            yield from _walk_items(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_items(child, path + (str(index),))


def _walk_strings(value: Any) -> Iterator[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        yield (), value
    for path, child in _walk_items(value):
        if isinstance(child, str):
            yield path, child


def _parse_signature(signature: str, target_name: str) -> tuple[str, list[str]]:
    match = re.fullmatch(
        rf"\s*(.+?)\s+{re.escape(target_name)}\s*\((.*)\)\s*", signature, re.S
    )
    _require(bool(match), f"invalid typed {target_name} signature: {signature!r}")
    return_type = match.group(1).strip()
    raw_params = match.group(2).strip()
    if not raw_params:
        return return_type, []
    chunks: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(raw_params):
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth -= 1
        elif char == "," and depth == 0:
            chunks.append(raw_params[start:index].strip())
            start = index + 1
    chunks.append(raw_params[start:].strip())
    names: list[str] = []
    for chunk in chunks:
        name_match = re.search(r"([A-Za-z_]\w*)\s*(?:=.*)?$", chunk)
        _require(bool(name_match), f"parameter has no identifier in {signature!r}")
        names.append(name_match.group(1))
    return return_type, names


def _contains_target_symbol(assembly: str, target_name: str) -> bool:
    escaped = re.escape(target_name)
    patterns = (
        rf"<{escaped}(?=[+>.\-])",
        rf"\bfunction\s+{escaped}\b",
        rf'"{escaped}"',
        rf"\b{escaped}(?=\s*[.(])",
    )
    return any(re.search(pattern, assembly) for pattern in patterns)


def _containing_line(text: str, start: int) -> tuple[str, int]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    return text[line_start : len(text) if line_end < 0 else line_end], start - line_start


def _is_opcode_occurrence(
    name: str, text: str, start: int, path: tuple[str, ...]
) -> bool:
    line, relative_start = _containing_line(text, start)
    if "instructions" in path:
        # Graph-v2 stores one normalized instruction per string, with the
        # opcode as its first token.
        match = re.match(r"^\s*(?P<opcode>[A-Za-z][A-Za-z0-9.]*)\b", line)
    elif "assembly" in path:
        # Frozen x86 evidence uses GDB's canonical shape:
        #   0x0000000000090ace <+30>:\tadd    rcx,rdx
        # Requiring a hexadecimal address and offset annotation excludes the
        # preceding `info functions` table (`283: static void List.add`).
        match = re.match(
            r"^\s*0x[0-9A-Fa-f]+\s+<[^>\r\n]+>:\s*"
            r"(?P<opcode>[A-Za-z][A-Za-z0-9.]*)\b",
            line,
        )
    else:
        return False
    return bool(
        match is not None
        and match.group("opcode") == name
        and match.start("opcode") == relative_start
    )


def _is_qualified_sdk_member_occurrence(line: str, relative_start: int) -> bool:
    # Method/member spellings (`List.add`, `JsonCodec.encode`) are qualified.
    if relative_start > 0 and line[relative_start - 1] == ".":
        return True
    # A library URI can also contain the token, but only inside a `dart:` URI;
    # file/source paths are intentionally not exempted.
    for match in re.finditer(r"\bdart:[^\s:;]+", line):
        if match.start() <= relative_start < match.end():
            return True
    return False


def _allowed_semantic_collision(
    name: str,
    text: str,
    start: int,
    path: tuple[str, ...],
    *,
    comparator_assembly: str | None = None,
) -> bool:
    """Allow only documented, comparator-bound identifier collisions.

    Some task/helper names are also opcodes (``add``, ``xor``), while the GDB
    preamble can contain SDK members such as ``List.add`` or
    ``JsonCodec.encode``. Opcodes are parsed positionally. Qualified SDK
    members/library URIs are benign only when the complete line at the same
    line ordinal is byte-identical in the corresponding frozen comparator.
    Bare symbols, target annotations/synopses, file paths, operands, and prose
    remain leaks.
    """
    if _is_opcode_occurrence(name, text, start, path):
        return True
    if "assembly" not in path or comparator_assembly is None:
        return False
    line, relative_start = _containing_line(text, start)
    line_number = text.count("\n", 0, start)
    comparator_lines = comparator_assembly.splitlines()
    return bool(
        _is_qualified_sdk_member_occurrence(line, relative_start)
        and line_number < len(comparator_lines)
        and comparator_lines[line_number] == line
    )


def _semantic_leaks(
    row: dict[str, Any],
    semantic_pattern: re.Pattern[str],
    *,
    comparator: dict[str, Any] | None = None,
) -> list[str]:
    leaks: list[str] = []
    comparator_assembly = str((comparator or {}).get("assembly") or "")
    for path, text in _walk_strings(row):
        for match in semantic_pattern.finditer(text):
            name = match.group(0)
            if _is_opcode_occurrence(name, text, match.start(), path):
                continue
            if _allowed_semantic_collision(
                name,
                text,
                match.start(),
                path,
                comparator_assembly=comparator_assembly,
            ):
                continue
            leaks.append(f"{'.'.join(path)}:{name}")
            if len(leaks) >= 5:
                return leaks
    return leaks


def _validate_public_row(
    row: dict[str, Any], arm: Arm, bundle: Bundle, semantic_pattern: re.Pattern[str]
) -> None:
    row_id = _task_id(row)
    for field in PRIVATE_ONLY_FIELDS:
        _require(field not in row, f"{arm.label} public {row_id} exposes {field}")
    for path, _value in _walk_items(row):
        if path and path[-1] in FORBIDDEN_PUBLIC_KEYS:
            raise GateError(
                f"{arm.label} public {row_id} exposes forbidden key {'.'.join(path)}"
            )
    for field in ("function", "camel_case_function_name"):
        _require(
            row.get(field) == bundle.target_name,
            f"{arm.label} public {row_id} has {field}={row.get(field)!r}",
        )
    protocol = row.get("benchmark_protocol")
    _require(isinstance(protocol, dict), f"{arm.label} public {row_id} lacks protocol")
    _require(
        protocol.get("schema") == EXPECTED_DATASET_SCHEMA,
        f"{arm.label} public {row_id} schema is {protocol.get('schema')!r}, "
        f"expected {EXPECTED_DATASET_SCHEMA!r}",
    )
    _require(
        protocol.get("neutral_target_name") == bundle.target_name,
        f"{arm.label} public {row_id} has wrong protocol target",
    )
    _require(
        protocol.get("public_signature_mode") == arm.mode,
        f"{arm.label} public {row_id} has wrong signature mode",
    )
    assembly = str(row.get("assembly") or "")
    _require(
        _contains_target_symbol(assembly, bundle.target_name),
        f"{arm.label} public {row_id} assembly lacks target symbol {bundle.target_name}",
    )
    source_hash = _private_source_hash(arm.private_by_id[row_id])
    comparator = bundle.benchmark_by_source[source_hash]
    leaks = _semantic_leaks(row, semantic_pattern, comparator=comparator)
    helper_names = extract_user_function_identifiers(comparator) - {
        _semantic_name(comparator)
    }
    if helper_names:
        helper_pattern = re.compile(
            r"\b(?:"
            + "|".join(
                re.escape(name)
                for name in sorted(helper_names, key=len, reverse=True)
            )
            + r")\b"
        )
        for leak in _semantic_leaks(
            row, helper_pattern, comparator=comparator
        ):
            if leak not in leaks:
                leaks.append(leak)
    _require(not leaks, f"{arm.label} public {row_id} semantic-name leak(s): {leaks}")

    if arm.mode == "name_only":
        _require(row.get("prompt_signature_mode") == "name_only", f"{row_id}: not name_only")
        _require(not str(row.get("dart_function_signature") or ""), f"{row_id}: public signature exposed")
        _require(not str(row.get("public_prompt_signature") or ""), f"{row_id}: prompt signature exposed")
        withheld = set(protocol.get("prompt_withholds") or [])
        _require(
            REQUIRED_NAMEONLY_WITHHOLDS <= withheld,
            f"{row_id}: name-only withholds missing {sorted(REQUIRED_NAMEONLY_WITHHOLDS - withheld)}",
        )
        exposes = set(protocol.get("prompt_exposes") or [])
        _require(
            not exposes.intersection({"typed_signature", "arity", "parameter_types"}),
            f"{row_id}: name-only prompt_exposes leaks interface metadata",
        )
    else:
        _require(row.get("prompt_signature_mode") == "exact", f"{row_id}: not exact")
        signature = str(row.get("dart_function_signature") or "")
        _require(signature == str(row.get("public_prompt_signature") or ""), f"{row_id}: signature fields differ")
        _return_type, names = _parse_signature(signature, bundle.target_name)
        expected_names = list("abcdefghijklmnop"[: len(names)])
        _require(names == expected_names, f"{row_id}: parameters are not neutral: {names}")


def _validate_private_row(
    row: dict[str, Any], arm: Arm, bundle: Bundle, source_hash: str
) -> None:
    row_id = _task_id(row)
    benchmark = bundle.benchmark_by_source[source_hash]
    benchmark_id = _task_id(benchmark)
    original_name = _semantic_name(benchmark)
    for field in ("function", "camel_case_function_name"):
        _require(row.get(field) == bundle.target_name, f"{arm.label} private {row_id}: wrong {field}")
    source = str(row.get("dart_source") or "")
    tests = str(row.get("tests") or "")
    evaluator_signature = str(row.get("evaluation_only_dart_function_signature") or "")
    protocol = row.get("benchmark_protocol") or {}
    _require(
        protocol.get("schema") == EXPECTED_DATASET_SCHEMA,
        f"{arm.label} private {row_id} schema is not {EXPECTED_DATASET_SCHEMA}",
    )
    _require(
        bool(re.search(rf"(?m)^\s*(?:@pragma\([^\n]*\)\s*)*[^\n]*\b{re.escape(bundle.target_name)}\s*\(", source)),
        f"{arm.label} private {row_id}: source does not declare {bundle.target_name}",
    )
    _parse_signature(evaluator_signature, bundle.target_name)
    if benchmark_id in KNOWN_STALE_TARGET_BINDINGS:
        stale = KNOWN_STALE_TARGET_BINDINGS[benchmark_id]
        _require(
            bool(re.search(rf"\bfinal\s+\w+\s*=\s*{re.escape(stale)}\s*;", tests)),
            f"known stale binding {benchmark_id} no longer matches expected {stale}",
        )
    else:
        _require(
            bool(
                re.search(
                    rf"(?:\bfinal\s+\w+\s*=\s*{re.escape(bundle.target_name)}\s*;|\b{re.escape(bundle.target_name)}\s*\()",
                    tests,
                )
            ),
            f"{arm.label} private {row_id}: tests do not call {bundle.target_name}",
        )
    if original_name and original_name != bundle.target_name:
        _require(
            not re.search(rf"\b{re.escape(original_name)}\b", tests),
            f"{arm.label} private {row_id}: original target remains in tests",
        )


def validate_contracts_and_hygiene(bundle: Bundle) -> str:
    semantic_names = sorted(
        {_semantic_name(row) for row in bundle.benchmark if _semantic_name(row)},
        key=len,
        reverse=True,
    )
    semantic_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(name) for name in semantic_names) + r")\b"
    )
    for arm in (bundle.nameonly, bundle.neutralexact):
        for public_row in arm.public:
            _validate_public_row(public_row, arm, bundle, semantic_pattern)
        for source_hash, private_row in arm.private_by_source.items():
            _validate_private_row(private_row, arm, bundle, source_hash)

    for source_hash in bundle.benchmark_by_source:
        left = bundle.nameonly.private_by_source[source_hash]
        right = bundle.neutralexact.private_by_source[source_hash]
        for field in ("dart_source", "tests", "evaluation_only_dart_function_signature"):
            _require(
                left.get(field) == right.get(field),
                f"private arms differ for source {source_hash[:12]} field {field}",
            )
    return (
        f"{bundle.expected_rows}x2 public rows are private-field/fingerprint clean; "
        f"target={bundle.target_name}; known stale bindings={sorted(KNOWN_STALE_TARGET_BINDINGS)}"
    )


def _canonical_instruction(instruction: str) -> str:
    # Symbol annotations are the only permitted instruction-level difference.
    instruction = re.sub(r"<[^>]*>", "<SYMBOL>", instruction)
    return " ".join(instruction.split())


def _scalar_or_sequence(value: Any) -> tuple[Any, ...]:
    """Treat legacy scalar CFG fields as one value, never as characters."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def canonical_cfg(row: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    cfg = row.get("cfg")
    _require(isinstance(cfg, list), f"{_task_id(row)} has no CFG list")
    projection: list[tuple[Any, ...]] = []
    for block in cfg:
        _require(isinstance(block, dict), f"{_task_id(row)} has a non-object CFG block")
        projection.append(
            (
                block.get("id"),
                block.get("start_address"),
                block.get("end_address"),
                tuple(
                    _canonical_instruction(str(item))
                    for item in _scalar_or_sequence(block.get("instructions"))
                ),
                _scalar_or_sequence(block.get("predecessors")),
                _scalar_or_sequence(block.get("successors")),
                _scalar_or_sequence(block.get("edge_types")),
                block.get("instruction_count"),
                block.get("block_type"),
            )
        )
    return tuple(projection)


def validate_frozen_cfg_parity(bundle: Bundle) -> str:
    for source_hash, benchmark in bundle.benchmark_by_source.items():
        expected_cfg = canonical_cfg(benchmark)
        expected_edges = benchmark.get("edges")
        left = bundle.nameonly.public_by_source[source_hash]
        right = bundle.neutralexact.public_by_source[source_hash]
        left_cfg = canonical_cfg(left)
        right_cfg = canonical_cfg(right)
        _require(left_cfg == expected_cfg, f"name-only CFG drift at benchmark task {_task_id(benchmark)}")
        _require(right_cfg == expected_cfg, f"neutral-exact CFG drift at benchmark task {_task_id(benchmark)}")
        _require(left_cfg == right_cfg, f"cross-arm CFG drift at benchmark task {_task_id(benchmark)}")
        _require(left.get("edges") == expected_edges, f"name-only edge drift at task {_task_id(benchmark)}")
        _require(right.get("edges") == expected_edges, f"neutral-exact edge drift at task {_task_id(benchmark)}")
        _require(left.get("edges") == right.get("edges"), f"cross-arm edge drift at task {_task_id(benchmark)}")
    return f"{bundle.expected_rows} paired CFGs preserve instructions modulo symbols and exact edges"


def load_prompt_renderer(source_path: Path = PROMPT_SOURCE) -> tuple[str, tuple[str, ...], Callable[..., str]]:
    """Load the real prompt function without importing the heavyweight trainer.

    The selected AST nodes are executed directly from the trainer source.  This
    keeps the build-box gate independent of torch/transformers while still
    testing the exact prompt implementation that inference will import.
    """
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except OSError as exc:
        raise GateError(f"cannot read prompt source {source_path}: {exc}") from exc
    wanted_functions = {
        "_compact_prompt_block",
        "_build_test_call_hint",
        "_fit_assembly_to_prompt_budget",
        "_fit_assembly_tokens",
        "_clean_asm_for_prompt",
        "build_decoder_prompt",
    }
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names: set[str] = set()
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            elif isinstance(node.target, ast.Name):
                names = {node.target.id}
            if names.intersection({"PROMPT_SCHEMA_VERSION", "_TOP_LEVEL_FUNCTION_CONSTRAINTS"}):
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            selected.append(node)
    namespace: dict[str, Any] = {"os": os, "re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    schema = namespace.get("PROMPT_SCHEMA_VERSION")
    constraints = namespace.get("_TOP_LEVEL_FUNCTION_CONSTRAINTS")
    renderer = namespace.get("build_decoder_prompt")
    _require(isinstance(schema, str), "prompt source lacks PROMPT_SCHEMA_VERSION")
    _require(isinstance(constraints, tuple), "prompt source lacks top-level constraints")
    _require(callable(renderer), "prompt source lacks build_decoder_prompt")
    return schema, constraints, renderer


def _prompt_skeleton(prompt: str) -> tuple[str, ...]:
    contract_prefixes = (
        "Implement this exact top-level Dart signature:",
        "Implement a top-level Dart function named exactly",
        "Infer the return type and complete parameter list",
        "from the binary representation.",
    )
    return tuple(
        line
        for line in prompt.splitlines()
        if not line.strip().startswith(contract_prefixes)
    )


def validate_rendered_prompts(
    bundle: Bundle, source_path: Path = PROMPT_SOURCE
) -> str:
    schema, top_level_constraints, renderer = load_prompt_renderer(source_path)
    _require(schema == EXPECTED_PROMPT_SCHEMA, f"prompt schema is {schema!r}, expected {EXPECTED_PROMPT_SCHEMA!r}")
    _require(
        set(top_level_constraints) <= set(SHARED_PROMPT_CONSTRAINTS),
        "prompt top-level constraints do not match the v3 gate contract",
    )
    old_values = {
        key: os.environ.get(key)
        for key in ("GRAPH_PROMPT_ASSEMBLY_MODE", "GRAPH_QWEN_PREFIX_TOKENS", "GRAPH_USE_REASONING")
    }
    os.environ.update(
        {
            "GRAPH_PROMPT_ASSEMBLY_MODE": "none",
            "GRAPH_QWEN_PREFIX_TOKENS": "64",
            "GRAPH_USE_REASONING": "0",
        }
    )
    try:
        for source_hash in bundle.nameonly.public_source_sequence:
            name_row = bundle.nameonly.public_by_source[source_hash]
            exact_row = bundle.neutralexact.public_by_source[source_hash]
            name_prompt = renderer(name_row)
            exact_prompt = renderer(exact_row)
            for constraint in SHARED_PROMPT_CONSTRAINTS:
                _require(name_prompt.count(constraint) == 1, f"name-only prompt missing/duplicates: {constraint}")
                _require(exact_prompt.count(constraint) == 1, f"neutral-exact prompt missing/duplicates: {constraint}")
            _require(_prompt_skeleton(name_prompt) == _prompt_skeleton(exact_prompt), "prompt structure differs beyond contract line")
            _require("assembly provided via graph channel" in name_prompt, "name-only prompt is not assembly_mode=none")
            _require("assembly provided via graph channel" in exact_prompt, "neutral-exact prompt is not assembly_mode=none")
            _require(str(name_row.get("assembly") or "") not in name_prompt, "name-only prompt contains raw assembly")
            _require(str(exact_row.get("assembly") or "") not in exact_prompt, "neutral-exact prompt contains raw assembly")
            _require("expect(" not in name_prompt and "ORACLE DIAGNOSTIC" not in name_prompt, "name-only prompt exposes tests")
            _require("expect(" not in exact_prompt and "ORACLE DIAGNOSTIC" not in exact_prompt, "neutral-exact prompt exposes tests")
            hidden = bundle.nameonly.private_by_source[source_hash]
            hidden_signature = str(hidden.get("evaluation_only_dart_function_signature") or "")
            _require(hidden_signature not in name_prompt, "name-only prompt exposes hidden evaluator signature")
            _require("Implement this exact top-level Dart signature" not in name_prompt, "name-only prompt uses typed contract")
            _require("Infer the return type and complete parameter list" in name_prompt, "name-only prompt omits ABI-inference instruction")
            exact_signature = str(exact_row.get("dart_function_signature") or "")
            _require(exact_signature in exact_prompt, "neutral-exact prompt omits public typed signature")
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return f"prompt schema {schema}; {bundle.expected_rows} matched assembly_mode=none prompt pairs"


def _toolchain_stamp(row: dict[str, Any]) -> tuple[str, str]:
    protocol = row.get("benchmark_protocol") or {}
    build = protocol.get("assembly_build") or {}
    _require(build.get("assembly_derivation") == "text_rename_of_frozen_benchmark_assembly", f"{_task_id(row)} lacks frozen-rename derivation")
    dart = str(build.get("asserted_frozen_dart_version") or "")
    gdb = str(build.get("asserted_frozen_gdb_version") or "")
    _require(bool(dart), f"{_task_id(row)} lacks asserted frozen Dart version")
    _require(bool(gdb), f"{_task_id(row)} lacks asserted frozen GDB version")
    _require(bool(build.get("extractor_sha256")), f"{_task_id(row)} lacks extractor hash")
    return dart, gdb


def validate_toolchain_and_summaries(
    bundle: Bundle,
    *,
    expected_dart_version: str | None = None,
    expected_gdb_version: str | None = None,
    nameonly_summary: dict[str, Any] | None = None,
    neutralexact_summary: dict[str, Any] | None = None,
) -> str:
    stamps = {
        _toolchain_stamp(row)
        for arm in (bundle.nameonly, bundle.neutralexact)
        for row in (*arm.public, *arm.private)
    }
    _require(len(stamps) == 1, f"toolchain stamps differ across rows/arms: {sorted(stamps)}")
    dart, gdb = next(iter(stamps))
    if expected_dart_version is not None:
        _require(dart == expected_dart_version, f"Dart stamp {dart!r} != expected {expected_dart_version!r}")
    if expected_gdb_version is not None:
        _require(gdb == expected_gdb_version, f"GDB stamp {gdb!r} != expected {expected_gdb_version!r}")

    summaries = ((bundle.nameonly, nameonly_summary), (bundle.neutralexact, neutralexact_summary))
    shuffle_seeds: list[Any] = []
    for arm, summary in summaries:
        if summary is None:
            continue
        _require("effective_id_salt" not in summary and "id_salt" not in summary, f"{arm.label} summary exposes effective salt")
        salt_hash = str(summary.get("effective_id_salt_sha256") or "")
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", salt_hash)), f"{arm.label} summary lacks redacted salt hash")
        _require(summary.get("accepted_rows") == bundle.expected_rows, f"{arm.label} summary accepted_rows mismatch")
        _require(summary.get("rejected_rows") == 0, f"{arm.label} summary has rejects")
        _require(summary.get("schema") == EXPECTED_DATASET_SCHEMA, f"{arm.label} summary schema mismatch")
        _require(summary.get("target_name") == bundle.target_name, f"{arm.label} summary target mismatch")
        _require(summary.get("public_signature_mode") == arm.mode, f"{arm.label} summary mode mismatch")
        summary_stamp = summary.get("toolchain") or {}
        _require(summary_stamp.get("asserted_frozen_dart_version") == dart, f"{arm.label} summary Dart stamp mismatch")
        _require(summary_stamp.get("asserted_frozen_gdb_version") == gdb, f"{arm.label} summary GDB stamp mismatch")
        shuffle_seeds.append(summary.get("shuffle_public_seed"))
    if len(shuffle_seeds) == 2:
        _require(shuffle_seeds[0] == shuffle_seeds[1], "public shuffle seeds differ across arms")
    return f"frozen toolchain stamps are complete and uniform: Dart={dart}; GDB={gdb}"


def validate_harness_patch() -> str:
    from scripts.evaluation.graph_compile_at_k_antigravity import _is_dart_jit_static_error

    _require(
        _is_dart_jit_static_error("Crash when compiling: RangeError (index): Invalid value"),
        "aligned-JIT classifier does not reject Dart front-end crashes",
    )
    return "aligned-JIT classifier includes front-end crash diagnostics"


DartEvaluator = Callable[[str, str, str, int, int], tuple[bool, bool, str, str]]


def run_dart_gates(
    bundle: Bundle,
    *,
    evaluator: DartEvaluator | None = None,
    workers: int = 4,
    timeout: int = 30,
    expected_stub_compile_failures: Iterable[str] = DEFAULT_STUB_COMPILE_FAILURES,
    expected_reference_compile_failures: Iterable[str] = DEFAULT_REFERENCE_COMPILE_FAILURES,
    expected_reference_pass_failures: Iterable[str] = DEFAULT_REFERENCE_PASS_FAILURES,
) -> str:
    if evaluator is None:
        from scripts.evaluation.graph_compile_at_k_antigravity import (
            evaluate_dart_jit_tests_detail,
        )

        evaluator = evaluate_dart_jit_tests_detail
    expected_stub = {str(value) for value in expected_stub_compile_failures}
    expected_ref_compile = {str(value) for value in expected_reference_compile_failures}
    expected_ref_pass = {str(value) for value in expected_reference_pass_failures}
    jobs: list[tuple[str, str, str, str]] = []
    for source_hash, private in bundle.neutralexact.private_by_source.items():
        benchmark_id = _task_id(bundle.benchmark_by_source[source_hash])
        public = bundle.neutralexact.public_by_source[source_hash]
        signature = str(public.get("dart_function_signature") or "")
        stub = f"{signature} {{ throw UnimplementedError(); }}"
        tests = str(private.get("tests") or "")
        reference = str(private.get("dart_source") or "")
        jobs.append(("stub", benchmark_id, stub, tests))
        jobs.append(("reference", benchmark_id, reference, tests))

    results: dict[tuple[str, str], tuple[bool, bool, str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(evaluator, code, tests, f"v3_{kind}_{task_id}", timeout, 1): (kind, task_id)
            for kind, task_id, code, tests in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                raise GateError(f"Dart evaluator crashed for {key}: {exc}") from exc

    stub_compile_failures = {
        task_id for (kind, task_id), result in results.items() if kind == "stub" and not result[0]
    }
    reference_compile_failures = {
        task_id for (kind, task_id), result in results.items() if kind == "reference" and not result[0]
    }
    reference_pass_failures = {
        task_id for (kind, task_id), result in results.items() if kind == "reference" and not result[1]
    }
    _require(
        stub_compile_failures == expected_stub,
        f"stub compile-failure set {sorted(stub_compile_failures)} != expected {sorted(expected_stub)}",
    )
    _require(
        reference_compile_failures == expected_ref_compile,
        f"reference compile-failure set {sorted(reference_compile_failures)} != expected {sorted(expected_ref_compile)}",
    )
    _require(
        reference_pass_failures == expected_ref_pass,
        f"reference pass-failure set {sorted(reference_pass_failures)} != expected {sorted(expected_ref_pass)}",
    )
    return (
        f"JIT exact sets: stubs compile {bundle.expected_rows-len(expected_stub)}/{bundle.expected_rows}; "
        f"references compile {bundle.expected_rows-len(expected_ref_compile)}/{bundle.expected_rows}, "
        f"pass {bundle.expected_rows-len(expected_ref_pass)}/{bundle.expected_rows}"
    )


def _parse_id_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nameonly_public", required=True, type=Path)
    parser.add_argument("--nameonly_private", required=True, type=Path)
    parser.add_argument("--neutralexact_public", required=True, type=Path)
    parser.add_argument("--neutralexact_private", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--nameonly_summary", type=Path)
    parser.add_argument("--neutralexact_summary", type=Path)
    parser.add_argument("--target_name", default="fn0")
    parser.add_argument("--expected_rows", type=int, default=154)
    parser.add_argument("--expected_dart_version")
    parser.add_argument("--expected_gdb_version")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--skip_dart",
        action="store_true",
        help="Run all static/prompt/toolchain gates but skip executable Dart checks",
    )
    parser.add_argument(
        "--expected_stub_compile_failures",
        default=",".join(sorted(DEFAULT_STUB_COMPILE_FAILURES)),
    )
    parser.add_argument(
        "--expected_reference_compile_failures",
        default=",".join(sorted(DEFAULT_REFERENCE_COMPILE_FAILURES)),
    )
    parser.add_argument(
        "--expected_reference_pass_failures",
        default=",".join(sorted(DEFAULT_REFERENCE_PASS_FAILURES)),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = build_bundle(
            nameonly_public=read_jsonl(args.nameonly_public),
            nameonly_private=read_jsonl(args.nameonly_private),
            neutralexact_public=read_jsonl(args.neutralexact_public),
            neutralexact_private=read_jsonl(args.neutralexact_private),
            benchmark=read_jsonl(args.benchmark),
            expected_rows=args.expected_rows,
            target_name=args.target_name,
        )
        evidence = [
            f"paired row order maps to one {args.expected_rows}-source permutation",
            validate_contracts_and_hygiene(bundle),
            validate_frozen_cfg_parity(bundle),
            validate_rendered_prompts(bundle),
            validate_toolchain_and_summaries(
                bundle,
                expected_dart_version=args.expected_dart_version,
                expected_gdb_version=args.expected_gdb_version,
                nameonly_summary=read_json(args.nameonly_summary) if args.nameonly_summary else None,
                neutralexact_summary=read_json(args.neutralexact_summary) if args.neutralexact_summary else None,
            ),
            validate_harness_patch(),
        ]
        if args.skip_dart:
            evidence.append("executable Dart gates explicitly skipped")
        else:
            evidence.append(
                run_dart_gates(
                    bundle,
                    workers=args.workers,
                    timeout=args.timeout,
                    expected_stub_compile_failures=_parse_id_set(args.expected_stub_compile_failures),
                    expected_reference_compile_failures=_parse_id_set(args.expected_reference_compile_failures),
                    expected_reference_pass_failures=_parse_id_set(args.expected_reference_pass_failures),
                )
            )
    except GateError as exc:
        print(f"FAIL fixed-scrub-v3: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Fail closed on unexpected gate bugs/environment faults.
        print(f"FAIL fixed-scrub-v3: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for item in evidence:
        print(f"PASS {item}")
    print("PASS fixed-scrub-v3 acceptance gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
