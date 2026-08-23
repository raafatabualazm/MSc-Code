#!/usr/bin/env python3
"""Fail-closed data, test-partition, and factual-grounding controls.

The helpers in this module are deliberately model-independent.  They are shared
by Phase-0 preparation, frontier repair harvesting, functional graph-use gates,
and the verified-only GRPO anchor loader.

The implementation favours conservative, auditable checks over fuzzy semantic
judgements.  A frontier model may propose a repair, but only executable tests
and mechanically extracted facts can admit that repair into training.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_VERSION = 2
NEUTRAL_FUNCTION_NAME = "fn0"

_SOURCE_KEYS = ("source", "dart_source", "swift_source")
_FUNCTION_KEYS = (
    "function",
    "name",
    "camel_case_function_name",
    "python_function_name",
)
_SIGNATURE_KEYS = ("dart_function_signature", "function_signature", "signature")

_DART_KEYWORDS = {
    "abstract", "as", "assert", "async", "await", "base", "break", "case",
    "catch", "class", "const", "continue", "covariant", "default", "deferred",
    "do", "dynamic", "else", "enum", "export", "extends", "extension", "external",
    "factory", "false", "final", "finally", "for", "Function", "get", "hide",
    "if", "implements", "import", "in", "interface", "is", "late", "library",
    "mixin", "new", "null", "of", "on", "operator", "part", "required", "rethrow",
    "return", "sealed", "set", "show", "static", "super", "switch", "sync", "this",
    "throw", "true", "try", "typedef", "var", "void", "when", "while", "with",
    "yield", "print", "main", "int", "double", "num", "bool", "String", "List",
    "Map", "Set", "Iterable", "Future", "Never", "Object", "Record", "RuneIterator",
}

_FRAME_RUNTIME_REGISTERS = {
    # ARM64 Dart VM / frame / object-pool registers.
    "sp", "fp", "x15", "w15", "x16", "w16", "x22", "w22", "x26", "w26",
    "x27", "w27", "x28", "w28", "x29", "w29", "x30", "w30",
    # x86-64 frame/stack/instruction pointers and common VM-reserved bases.
    "rsp", "esp", "rbp", "ebp", "rip", "eip", "r15", "r14", "r13",
}

_DATA_IMMEDIATE_MNEMONICS = {
    "add", "adds", "sub", "subs", "mul", "madd", "msub", "smull", "umull",
    "sdiv", "udiv", "div", "idiv", "imul", "and", "ands", "orr", "or", "eor",
    "xor", "bic", "lsl", "lsr", "asr", "ror", "shl", "shr", "sar", "inc", "dec",
    "neg", "not", "min", "max", "fadd", "fsub", "fmul", "fdiv",
}

_ASM_BRANCH_MNEMONICS = {
    # AArch64
    "b", "b.eq", "b.ne", "b.gt", "b.ge", "b.lt", "b.le", "b.hi", "b.hs",
    "b.lo", "b.ls", "cbz", "cbnz", "tbz", "tbnz", "br", "blr", "bl",
    # x86/x86-64
    "jmp", "je", "jne", "jg", "jge", "jl", "jle", "ja", "jae", "jb", "jbe",
    "jz", "jnz", "js", "jns", "jo", "jno", "jp", "jnp", "jecxz", "jrcxz",
    "call", "callq",
}
_ASM_CALL_MNEMONICS = {"bl", "blr", "call", "callq"}
_ASM_RETURN_MNEMONICS = {"ret", "retq"}
_ASM_COMPARE_MNEMONICS = {
    "cmp", "cmn", "tst", "test", "subs", "ucomisd", "comisd", "fcmp", "ccmp",
}

_RUNTIME_CALLEE_FRAGMENTS = (
    "stub", "runtime", "dart", "writebarrier", "safepoint", "stackoverflow",
    "allocate", "thread", "object", "typecheck", "monomorphic", "megamorphic",
    "iccall", "deopt", "patch", "slowpath", "nullerror", "rangeerror",
)


def read_jsonl_many(spec: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read one or more comma-separated JSONL files."""
    raw_parts: list[str | Path]
    if isinstance(spec, (str, Path)):
        raw_parts = [part.strip() for part in str(spec).split(",") if part.strip()]
    else:
        raw_parts = list(spec)
    rows: list[dict[str, Any]] = []
    for raw in raw_parts:
        path = Path(raw).expanduser().resolve()
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def source_text(record: dict[str, Any]) -> str:
    for key in _SOURCE_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def set_source_text(record: dict[str, Any], text: str) -> None:
    present = [key for key in _SOURCE_KEYS if key in record]
    if not present:
        record["source"] = text
        return
    for key in present:
        record[key] = text


def infer_function_name(record: dict[str, Any]) -> str:
    for key in _FUNCTION_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    signature = signature_text(record)
    match = re.search(r"\b([A-Za-z_]\w*)\s*\(", signature)
    if match:
        return match.group(1)
    source = source_text(record)
    match = re.search(
        r"(?m)^\s*(?:[A-Za-z_]\w*(?:<[^;{}()]+>)?[?]?\s+)+([A-Za-z_]\w*)\s*\(",
        source,
    )
    return match.group(1) if match else ""


def signature_text(record: dict[str, Any]) -> str:
    for key in _SIGNATURE_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip().rstrip(";{").strip()
    return ""


def task_identity(record: dict[str, Any], index: int | None = None) -> str:
    for key in ("task_id", "id", "problem_id", "filename", "file"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value).replace("\\", "/").rsplit("/", 1)[-1]
    name = infer_function_name(record)
    if name:
        return name
    return f"row-{index if index is not None else 'unknown'}"


def semantic_pair_identity(record: dict[str, Any]) -> str:
    """Return the opaque cross-architecture semantic grouping key, if present."""
    value = record.get("semantic_pair_id")
    if value in (None, ""):
        metadata = record.get("hybrid_metadata") or {}
        if isinstance(metadata, dict):
            value = metadata.get("semantic_pair_id")
    return str(value).strip() if value not in (None, "") else ""


def _strip_comments(text: str) -> str:
    """Remove comments while preserving quoted strings."""
    out: list[str] = []
    i = 0
    quote: str | None = None
    escaped = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if quote is not None:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            out.append("\n")
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(len(text), i + 2)
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalized_source(record_or_source: dict[str, Any] | str) -> str:
    """Canonical text used for within-pool and frozen-eval overlap checks.

    The known function name is replaced by a neutral token so a renamed copy of
    a benchmark solution still collides.  String literals are not modified.
    """
    if isinstance(record_or_source, dict):
        text = source_text(record_or_source)
        function_name = infer_function_name(record_or_source)
    else:
        text = str(record_or_source)
        function_name = ""
    text = _strip_comments(text)
    if function_name:
        text = replace_identifier_outside_strings(text, function_name, "__FUNCTION__")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_source_hash(record_or_source: dict[str, Any] | str) -> str:
    return hashlib.sha256(normalized_source(record_or_source).encode("utf-8")).hexdigest()


def _dart_lexical_tokens(text: str) -> list[str]:
    """Tokenise enough Dart syntax for a conservative alpha-equivalence hash.

    This is not a compiler parser.  It deliberately preserves literals, operators,
    member names, and external call names while making local/parameter identifiers
    insensitive to renaming.  Comments and whitespace never affect the result.
    """
    text = _strip_comments(text)
    tokens: list[str] = []
    operators = (
        ">>>=", "<<=", ">>=", "??=", "?..", "...", "=>", "==", "!=", "<=", ">=",
        "&&", "||", "??", "?.", "..", "++", "--", "+=", "-=", "*=", "/=", "%=",
        "&=", "|=", "^=", "<<", ">>>", ">>", "~/", "~/=",
    )
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        raw = False
        if ch in {"r", "R"} and i + 1 < len(text) and text[i + 1] in {"'", '"'}:
            raw = True
            start = i
            i += 1
            ch = text[i]
        elif ch in {"'", '"'}:
            start = i
        else:
            start = i
        if ch in {"'", '"'}:
            quote = ch
            triple = text.startswith(quote * 3, i)
            delimiter = quote * (3 if triple else 1)
            i += len(delimiter)
            while i < len(text):
                if text.startswith(delimiter, i):
                    i += len(delimiter)
                    break
                if not raw and text[i] == "\\" and i + 1 < len(text):
                    i += 2
                else:
                    i += 1
            tokens.append(text[start:i])
            continue
        if ch.isalpha() or ch == "_":
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(text[start:i])
            continue
        if ch.isdigit():
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] in {"_", ".", "+", "-"}):
                # Stop + / - unless they belong to an exponent.
                if text[i] in {"+", "-"} and text[i - 1] not in {"e", "E"}:
                    break
                i += 1
            tokens.append(text[start:i])
            continue
        matched = next((operator for operator in operators if text.startswith(operator, i)), None)
        if matched:
            tokens.append(matched)
            i += len(matched)
            continue
        tokens.append(ch)
        i += 1
    return tokens


def _declared_dart_identifiers(text: str, function_name: str) -> set[str]:
    clean = _strip_comments(text)
    names = {function_name} if function_name else set()
    # Local declarations with var/final/const/late.
    for match in re.finditer(
        r"\b(?:var|final|const|late)\s+(?:(?:[A-Za-z_]\w*(?:\s*<[^;={}]+>)?[?]?)\s+)?([A-Za-z_]\w*)",
        clean,
    ):
        names.add(match.group(1))
    # Function declarations and their parameters, including local helpers.
    function_pattern = re.compile(
        r"(?m)(?:^|[;{}])\s*(?:(?:external|static|late|final|const|factory)\s+)*"
        r"(?:[A-Za-z_]\w*(?:\s*<[^;{}()]+>)?[?]?\s+)+([A-Za-z_]\w*)\s*"
        r"\(([^;{}]*)\)\s*(?:async\s*)?(?==>|\{)"
    )
    for match in function_pattern.finditer(clean):
        name = match.group(1)
        if name not in _DART_KEYWORDS:
            names.add(name)
        params = match.group(2).replace("{", "").replace("}", "").replace("[", "").replace("]", "")
        for parameter in _split_parameters(params):
            parameter = parameter.split("=", 1)[0].strip()
            found = re.search(r"([A-Za-z_]\w*)\s*$", parameter)
            if found and found.group(1) not in _DART_KEYWORDS:
                names.add(found.group(1))
    # catch (error, stack), for-loop variables, and typed local declarations.
    for match in re.finditer(r"\bcatch\s*\(([^)]*)\)", clean):
        for item in _split_parameters(match.group(1)):
            found = re.search(r"([A-Za-z_]\w*)\s*$", item)
            if found:
                names.add(found.group(1))
    for match in re.finditer(
        r"\bfor\s*\(\s*(?:(?:var|final|const)\s+|(?:[A-Za-z_]\w*(?:<[^;()]+>)?[?]?)\s+)([A-Za-z_]\w*)",
        clean,
    ):
        names.add(match.group(1))
    return {name for name in names if name and name not in _DART_KEYWORDS}


def alpha_normalized_source(record_or_source: dict[str, Any] | str) -> str:
    """Canonicalise local identifier names while preserving program evidence.

    The exact-neutral hash catches byte-level copies with a renamed top-level
    function.  This second hash also catches copies whose parameters and local
    variables were renamed.  Literals, operators, types, member names, and
    external call names remain intact to limit false collisions.
    """
    if isinstance(record_or_source, dict):
        text = source_text(record_or_source)
        function_name = infer_function_name(record_or_source)
    else:
        text = str(record_or_source)
        function_name = ""
    declared = _declared_dart_identifiers(text, function_name)
    tokens = _dart_lexical_tokens(text)
    mapping: dict[str, str] = {}
    result: list[str] = []
    for index, token in enumerate(tokens):
        if not re.fullmatch(r"[A-Za-z_]\w*", token):
            result.append(token)
            continue
        previous = tokens[index - 1] if index else ""
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if function_name and token == function_name:
            result.append("__FUNCTION__")
            continue
        if token in mapping:
            result.append(mapping[token])
            continue
        if token in _DART_KEYWORDS or token[:1].isupper() or previous in {".", "?."}:
            result.append(token)
            continue
        # Preserve unknown free call symbols (sqrt(...), max(...), etc.).  Helper
        # declarations and known local functions are in ``declared`` and are
        # therefore alpha-renamed together with their call sites.
        if following == "(" and token not in declared:
            result.append(token)
            continue
        mapping[token] = f"__id{len(mapping)}__"
        result.append(mapping[token])
    return " ".join(result)


def source_fingerprints(record_or_source: dict[str, Any] | str) -> dict[str, str]:
    return {
        "neutral_sha256": normalized_source_hash(record_or_source),
        "alpha_structural_sha256": hashlib.sha256(
            alpha_normalized_source(record_or_source).encode("utf-8")
        ).hexdigest(),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def replace_identifier_outside_strings(text: str, old: str, new: str) -> str:
    """Replace one identifier without rewriting comments/string literals."""
    if not old or old == new:
        return text
    out: list[str] = []
    i = 0
    quote: str | None = None
    escaped = False
    in_line_comment = False
    in_block_comment = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line_comment:
            out.append(ch)
            if ch in "\r\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            out.append(ch)
            if ch == "*" and nxt == "/":
                out.append(nxt)
                i += 2
                in_block_comment = False
            else:
                i += 1
            continue
        if quote is not None:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            out.extend((ch, nxt))
            i += 2
            in_line_comment = True
            continue
        if ch == "/" and nxt == "*":
            out.extend((ch, nxt))
            i += 2
            in_block_comment = True
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if (ch.isalpha() or ch == "_"):
            start = i
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] == "_"):
                i += 1
            token = text[start:i]
            out.append(new if token == old else token)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_parameters(text: str) -> list[str]:
    params: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            params.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        params.append(tail)
    return params


def parameter_arity(signature: str) -> int | None:
    match = re.search(r"\((.*)\)", signature or "", flags=re.S)
    if not match:
        return None
    inner = match.group(1).strip()
    if not inner:
        return 0
    # Strip optional/named grouping delimiters but retain their contents.
    inner = inner.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    return len([value for value in _split_parameters(inner) if value])


def signature_return_type(signature: str) -> str | None:
    match = re.match(
        r"^\s*(?:external\s+|static\s+|late\s+|final\s+|const\s+)*"
        r"(.+?)\s+[A-Za-z_]\w*\s*\(",
        signature or "",
        flags=re.S,
    )
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1).strip())


def _neutral_parameter_name(index: int) -> str:
    """Return short opaque parameter names: a..z, then a26, a27, ..."""
    if 0 <= index < 26:
        return chr(ord("a") + index)
    return f"a{index}"


def _parameter_type(parameter: str, index: int) -> tuple[str, str]:
    value = parameter.strip()
    value = re.sub(r"^(?:required\s+)", "", value)
    value = value.split("=", 1)[0].strip()
    opaque_name = _neutral_parameter_name(index)
    # Function-typed parameters and destructuring are deliberately left as-is.
    match = re.match(r"(.+?)\s+([A-Za-z_]\w*)$", value, flags=re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip()), opaque_name
    return "dynamic", opaque_name


def neutral_signature(signature: str, old_name: str, new_name: str = NEUTRAL_FUNCTION_NAME) -> str:
    """Rename a signature and neutralise parameter identifiers while retaining types."""
    signature = signature.strip().rstrip(";{").strip()
    match = re.match(r"^(.*?)\b([A-Za-z_]\w*)\s*\((.*)\)\s*(?:async\s*)?$", signature, flags=re.S)
    if not match:
        raise ValueError(f"could not parse function signature: {signature!r}")
    prefix, parsed_name, params_text = match.groups()
    if old_name and parsed_name != old_name:
        # A stale metadata name is safer to reject than silently neutralise the wrong function.
        raise ValueError(
            f"signature function {parsed_name!r} does not match record function {old_name!r}"
        )
    params = _split_parameters(params_text)
    rendered: list[str] = []
    for index, parameter in enumerate(params):
        if not parameter.strip():
            continue
        type_text, arg_name = _parameter_type(parameter, index)
        rendered.append(f"{type_text} {arg_name}")
    return f"{prefix.strip()} {new_name}({', '.join(rendered)})".strip()


def extract_source_signature(source: str, function_name: str) -> str:
    if not function_name:
        return ""
    pattern = re.compile(
        rf"(?m)^\s*((?:(?:external|static|late|final|const)\s+)*"
        rf"[A-Za-z_][\w<>,?\[\]\s]*(?:\s+))({re.escape(function_name)})\s*\(([^)]*)\)",
    )
    match = pattern.search(source)
    if not match:
        return ""
    return f"{match.group(1).strip()} {function_name}({match.group(3).strip()})"


def _rename_assembly_symbol_contexts(text: str, name: str, replacement: str) -> str:
    """Rename one symbol in GDB/objdump contexts without touching opcodes.

    A blanket identifier replacement corrupts real instructions when a target
    has an architecture mnemonic for a name (for example ``add``).  The
    contexts below are limited to disassembler headers, symbol annotations and
    labels, declaration synopses, and branch/call operands.
    """
    if not text or not name or name == replacement:
        return text
    escaped = re.escape(name)

    # GDB/objdump symbol annotations and dump/symbol-search headers.
    text = re.sub(rf"<{escaped}(?=[+>.@\-])", f"<{replacement}", text)
    text = re.sub(
        rf"(?i)(\b(?:function|symbol)\s+){escaped}(?![A-Za-z0-9_])",
        rf"\g<1>{replacement}",
        text,
    )
    text = re.sub(rf'"{escaped}"', f'"{replacement}"', text)

    # Static declaration synopses such as ``1: static int target(void);`` and
    # nested Dart symbols such as ``target.<anonymous closure>``.
    text = re.sub(
        rf"(?<![A-Za-z0-9_<.]){escaped}(?=\s*[.(])",
        replacement,
        text,
    )

    # Bare and address-prefixed labels.  Anchoring at the start of a line is
    # what keeps an instruction like ``add rax, rbx`` unchanged.
    text = re.sub(
        rf"(?m)^(\s*(?:(?:0x)?[0-9A-Fa-f]+\s+)?)({escaped})(?=\s*:)",
        rf"\g<1>{replacement}",
        text,
    )

    # Direct symbolic branch/call operands not wrapped in ``<...>``.  The
    # prefix may include registers (cbz/tbz) or an address column.
    single_target_branch = (
        r"(?:b(?:\.[A-Za-z][A-Za-z0-9_.]*)?|bl|callq?|jmp|j[a-z][a-z0-9]*)"
    )
    text = re.sub(
        rf"(?im)^([^\r\n]*?\b{single_target_branch}\b\s+(?:[#=$*]\s*)?)"
        rf"{escaped}(?=(?:[+@.]|\s|$))",
        rf"\g<1>{replacement}",
        text,
    )
    text = re.sub(
        rf"(?im)^([^\r\n]*?\b(?:cbz|cbnz|tbz|tbnz)\b[^\r\n]*,\s*)"
        rf"{escaped}(?=(?:[+@.]|\s|$))",
        rf"\g<1>{replacement}",
        text,
    )
    return text


def scrub_assembly_symbol(
    assembly: str,
    original_name: str,
    neutral_name: str = NEUTRAL_FUNCTION_NAME,
) -> str:
    """Scrub a target symbol from GDB/objdump text and fail on residue."""
    renamed = _rename_assembly_symbol_contexts(assembly, original_name, neutral_name)
    # Source paths are disassembler provenance, not binary evidence, and may
    # themselves carry the semantic function name.
    renamed = re.sub(r"file:///\S+", "file:///scrubbed/program.dart", renamed)
    renamed = re.sub(
        r"(?im)(?:[A-Za-z]:[\\/]|/(?:tmp|root)/)[^\s:]+\.dart",
        "/scrubbed/program.dart",
        renamed,
    )

    escaped = re.escape(original_name)
    single_target_branch = (
        r"(?:b(?:\.[A-Za-z][A-Za-z0-9_.]*)?|bl|callq?|jmp|j[a-z][a-z0-9]*)"
    )
    residue_patterns = (
        rf"<{escaped}(?=[+>.@\-])",
        rf"(?i)\b(?:function|symbol)\s+{escaped}(?![A-Za-z0-9_])",
        rf'"{escaped}"',
        rf"(?<![A-Za-z0-9_<.]){escaped}(?=\s*[.(])",
        rf"(?m)^\s*(?:(?:0x)?[0-9A-Fa-f]+\s+)?{escaped}(?=\s*:)",
        rf"(?im)^[^\r\n]*?\b{single_target_branch}\b\s+(?:[#=$*]\s*)?"
        rf"{escaped}(?=(?:[+@.]|\s|$))",
        rf"(?im)^[^\r\n]*?\b(?:cbz|cbnz|tbz|tbnz)\b[^\r\n]*,\s*"
        rf"{escaped}(?=(?:[+@.]|\s|$))",
    )
    for pattern in residue_patterns:
        residue = re.search(pattern, renamed)
        if residue:
            raise ValueError(
                f"assembly symbol residue {residue.group(0)!r} survived neutralization"
            )
    return renamed


def neutralize_training_row(
    record: dict[str, Any],
    neutral_name: str = NEUTRAL_FUNCTION_NAME,
) -> dict[str, Any]:
    """Create a typed, opaque-contract copy suitable for training or evaluation.

    Function identifiers are replaced only outside quoted literals/comments.
    Parameter *names* are neutralised in the prompt signature, while source and
    test parameter identifiers are preserved to avoid rewriting executable code.
    """
    row = copy.deepcopy(record)
    old_name = infer_function_name(row)
    if not old_name:
        raise ValueError(f"{task_identity(row)}: missing function name")
    source = source_text(row)
    if not source:
        raise ValueError(f"{task_identity(row)}: missing source")
    signature = signature_text(row) or extract_source_signature(source, old_name)
    if not signature:
        raise ValueError(f"{task_identity(row)}: could not recover typed signature")
    neutral_sig = neutral_signature(signature, old_name, neutral_name)

    set_source_text(row, replace_identifier_outside_strings(source, old_name, neutral_name))

    # The binary channel must be neutral as well.  Otherwise a graph-only arm
    # can recover the semantic task label from an assembly symbol even though
    # the decoder contract was renamed to fn0.  Use disassembler contexts
    # rather than blanket identifier replacement so mnemonic-shaped names do
    # not corrupt real instructions.
    original_assembly = str(row.get("assembly") or "")
    if original_assembly:
        neutral_assembly = scrub_assembly_symbol(
            original_assembly, old_name, neutral_name
        )
        row["assembly"] = neutral_assembly
        if "assembly_sha256" in row:
            row["assembly_sha256"] = sha256_text(neutral_assembly)
        graph_v2 = row.get("graph_v2")
        if isinstance(graph_v2, dict) and "assembly_sha256" in graph_v2:
            graph_v2 = copy.deepcopy(graph_v2)
            graph_v2["assembly_sha256"] = sha256_text(neutral_assembly)
            row["graph_v2"] = graph_v2

    cfg = row.get("cfg")
    if isinstance(cfg, list):
        neutral_cfg = copy.deepcopy(cfg)
        for block in neutral_cfg:
            if not isinstance(block, dict):
                continue
            instructions = block.get("instructions")
            if isinstance(instructions, list):
                block["instructions"] = [
                    scrub_assembly_symbol(str(value), old_name, neutral_name)
                    for value in instructions
                ]
            elif isinstance(instructions, str):
                block["instructions"] = scrub_assembly_symbol(
                    instructions, old_name, neutral_name
                )
        row["cfg"] = neutral_cfg

    if row.get("tests"):
        row["tests"] = replace_identifier_outside_strings(str(row["tests"]), old_name, neutral_name)
    if row.get("feedback_tests"):
        row["feedback_tests"] = replace_identifier_outside_strings(
            str(row["feedback_tests"]), old_name, neutral_name
        )
    if row.get("acceptance_tests"):
        row["acceptance_tests"] = replace_identifier_outside_strings(
            str(row["acceptance_tests"]), old_name, neutral_name
        )

    for key in _FUNCTION_KEYS:
        if key in row or key == "function":
            row[key] = neutral_name
    for key in _SIGNATURE_KEYS:
        if key in row or key == "signature":
            row[key] = neutral_sig
    row["prompt_signature_mode"] = "exact"
    metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
    # ``hybrid_metadata`` travels with model-facing rows.  Never retain the
    # semantic identifier there; private manifests can bind the source through
    # the existing hashes instead.
    metadata.pop("original_function", None)
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "neutralized": True,
            "neutral_contract": True,
            "neutral_function": neutral_name,
            "original_signature_sha256": sha256_text(signature),
            "original_assembly_sha256": sha256_text(original_assembly),
            "neutralized_assembly_sha256": sha256_text(str(row.get("assembly") or "")),
            "binary_symbol_neutralized": True,
        }
    )
    row["hybrid_metadata"] = metadata
    return row


def _find_main_body(test_code: str) -> tuple[int, int, int, int] | None:
    match = re.search(r"\bvoid\s+main\s*\([^)]*\)\s*\{", test_code)
    if not match:
        return None
    open_brace = test_code.find("{", match.start(), match.end())
    if open_brace < 0:
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    in_line_comment = False
    in_block_comment = False
    i = open_brace
    while i < len(test_code):
        ch = test_code[i]
        nxt = test_code[i + 1] if i + 1 < len(test_code) else ""
        if in_line_comment:
            if ch in "\r\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                i += 2
                in_block_comment = False
            else:
                i += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return match.start(), open_brace, open_brace + 1, i
        i += 1
    return None


def _candidate_expect_spans(body: str) -> list[tuple[int, int, str]]:
    """Locate executable ``expect(...)`` statements in a main-body fragment.

    The earlier line-anchored regex missed compact one-line harnesses.  This
    scanner accepts either layout while skipping strings and comments so an
    assertion shown in documentation cannot enter the feedback/acceptance split.
    """
    starts: list[int] = []
    i = 0
    quote: str | None = None
    escaped = False
    in_line_comment = False
    in_block_comment = False
    while i < len(body):
        ch = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if in_line_comment:
            if ch in "\r\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if body.startswith("expect", i):
            before_ok = i == 0 or not (body[i - 1].isalnum() or body[i - 1] == "_")
            j = i + len("expect")
            after_ok = j >= len(body) or not (body[j].isalnum() or body[j] == "_")
            while j < len(body) and body[j].isspace():
                j += 1
            if before_ok and after_ok and j < len(body) and body[j] == "(":
                starts.append(i)
                i = j + 1
                continue
        i += 1

    spans: list[tuple[int, int, str]] = []
    for start in starts:
        open_paren = body.find("(", start + len("expect"))
        if open_paren < 0:
            continue
        i = open_paren
        depth = 0
        quote = None
        escaped = False
        in_line_comment = False
        in_block_comment = False
        while i < len(body):
            ch = body[i]
            nxt = body[i + 1] if i + 1 < len(body) else ""
            if in_line_comment:
                if ch in "\r\n":
                    in_line_comment = False
                i += 1
                continue
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue
            if quote is not None:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
                i += 1
                continue
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue
            if ch in {"'", '"'}:
                quote = ch
                i += 1
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == ";" and depth == 0:
                end = i + 1
                statement = body[start:end]
                if re.search(r"\bcandidate\s*\(", statement):
                    spans.append((start, end, statement.strip()))
                break
            i += 1

    result: list[tuple[int, int, str]] = []
    last_end = -1
    for item in sorted(spans):
        if item[0] >= last_end:
            result.append(item)
            last_end = item[1]
    return result


def candidate_assertions(test_code: str) -> list[str]:
    main = _find_main_body(test_code or "")
    if main is None:
        return []
    _, _, body_start, body_end = main
    return [text for _, _, text in _candidate_expect_spans(test_code[body_start:body_end])]


def candidate_assertion_count(test_code: str) -> int:
    return len(candidate_assertions(test_code))


def _rebuild_test_harness(test_code: str, keep_indices: set[int]) -> str:
    main = _find_main_body(test_code)
    if main is None:
        raise ValueError("test harness has no parseable void main()")
    _, _, body_start, body_end = main
    body = test_code[body_start:body_end]
    spans = _candidate_expect_spans(body)
    if not spans:
        raise ValueError("test harness has no expect(candidate(...)) assertions")
    pieces: list[str] = []
    cursor = 0
    selected_statements: list[str] = []
    for index, (start, end, statement) in enumerate(spans):
        pieces.append(body[cursor:start])
        if index in keep_indices:
            selected_statements.append(statement)
            pieces.append(body[start:end])
        cursor = end
    pieces.append(body[cursor:])
    rebuilt = test_code[:body_start] + "".join(pieces) + test_code[body_end:]
    # Assert fail-closed partition integrity.
    if candidate_assertion_count(rebuilt) != len(selected_statements):
        raise RuntimeError("test partition reconstruction changed assertion count")
    return rebuilt


def split_test_harness(
    test_code: str,
    *,
    feedback_fraction: float = 0.6,
    seed: int = 42,
    identity: str = "",
) -> tuple[str, str, dict[str, Any]]:
    """Deterministically partition candidate assertions into visible and hidden sets.

    For a single-assertion task, the assertion is hidden and the visible split
    is empty.  Such tasks remain usable for final acceptance, but are not sent
    to a teacher with test-derived diagnostics.
    """
    assertions = candidate_assertions(test_code)
    count = len(assertions)
    if count == 0:
        raise ValueError("no expect(candidate(...)) assertions found")
    if not 0.0 < feedback_fraction < 1.0:
        raise ValueError("feedback_fraction must be strictly between 0 and 1")
    if count == 1:
        return "", test_code, {
            "assertions_total": 1,
            "feedback_assertions": 0,
            "acceptance_assertions": 1,
            "single_assertion_hidden": True,
        }

    material = f"{seed}:{identity}:{sha256_text(test_code)}"
    rng = random.Random(int(hashlib.sha256(material.encode()).hexdigest()[:16], 16))
    order = list(range(count))
    rng.shuffle(order)
    feedback_count = int(round(count * feedback_fraction))
    feedback_count = min(count - 1, max(1, feedback_count))
    feedback_indices = set(order[:feedback_count])
    acceptance_indices = set(order[feedback_count:])
    feedback = _rebuild_test_harness(test_code, feedback_indices)
    acceptance = _rebuild_test_harness(test_code, acceptance_indices)
    return feedback, acceptance, {
        "assertions_total": count,
        "feedback_assertions": len(feedback_indices),
        "acceptance_assertions": len(acceptance_indices),
        "single_assertion_hidden": False,
        "partition_sha256": sha256_text(
            json.dumps(
                {
                    "feedback": sorted(feedback_indices),
                    "acceptance": sorted(acceptance_indices),
                },
                separators=(",", ":"),
            )
        ),
    }


def attach_test_partition(
    record: dict[str, Any],
    *,
    feedback_fraction: float = 0.6,
    seed: int = 42,
) -> dict[str, Any]:
    row = copy.deepcopy(record)
    tests = str(row.get("tests") or "")
    if candidate_assertion_count(tests) == 0:
        # Differential / stdout-capture oracle (e.g. the scrubbed-master
        # runZoned print-capture harness) carries no discrete
        # expect(candidate(...)) assertions to partition. Treat the whole
        # runnable harness as a single hidden acceptance oracle: the row still
        # validates through full-harness reference replay and feeds the SFT
        # stages, but has no visible feedback tests, so it is excluded from the
        # teacher-feedback / visible-test repair stages (which require discrete
        # assertions). Rows with neither assertions nor a runnable harness are
        # still rejected.
        if not tests.strip():
            raise ValueError("no expect(candidate(...)) assertions and no runnable harness")
        # The scrubbed-master differential harness uses runZoned/ZoneSpecification
        # (dart:async) and often dart:convert, which the master build supplied
        # only at compile time. The pass-aligned evaluator hoists imports from
        # the candidate SOURCE, so ensure the reference carries them. These
        # imports never appear in the precomputed assembly (imports are compile-
        # time), so the model input is unchanged.
        for _field in ("dart_source", "source"):
            _src = row.get(_field)
            if isinstance(_src, str) and _src.strip():
                _needed = [
                    _imp for _imp in ("import 'dart:async';", "import 'dart:convert';")
                    if _imp not in _src
                ]
                if _needed:
                    row[_field] = "\n".join(_needed) + "\n" + _src
        row["feedback_tests"] = ""
        row["acceptance_tests"] = tests
        metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
        metadata["test_partition"] = {
            "assertions_total": 0,
            "feedback_assertions": 0,
            "acceptance_assertions": 0,
            "single_assertion_hidden": True,
            "oracle_kind": "differential_whole_harness",
        }
        metadata["test_partition_hidden_from_teacher"] = True
        metadata["differential_oracle"] = True
        row["hybrid_metadata"] = metadata
        return row
    feedback, acceptance, info = split_test_harness(
        tests,
        feedback_fraction=feedback_fraction,
        seed=seed,
        identity=task_identity(row),
    )
    row["feedback_tests"] = feedback
    row["acceptance_tests"] = acceptance
    metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
    metadata["test_partition"] = info
    metadata["test_partition_hidden_from_teacher"] = True
    row["hybrid_metadata"] = metadata
    return row


def _strip_asm_comment(line: str) -> str:
    # Preserve '#imm' on ARM64 while stripping semicolon and // annotations.
    line = re.sub(r"//.*$", "", line)
    line = re.sub(r";.*$", "", line)
    return line.strip()


def _parse_asm_line(line: str) -> tuple[str, str] | None:
    clean = _strip_asm_comment(line)
    if not clean or clean.startswith((".", "[", "{")):
        return None
    lowered = clean.lower()
    if lowered.startswith(
        (
            "all functions matching regular expression",
            "dump of assembler code for function",
            "end of assembler dump",
        )
    ):
        return None
    if re.match(
        r"^\s*\d+:\s*(?:(?:static|external|native|abstract)\s+)*(?:void|int|double|bool|string|dynamic)\b",
        clean,
        flags=re.I,
    ):
        return None
    # Remove address/bytes prefixes and labels.
    clean = re.sub(r"^\s*(?:0x)?[0-9a-fA-F]+:\s*", "", clean)
    if re.match(r"^[A-Za-z_.$][\w.$@<>+-]*:\s*$", clean):
        return None
    match = re.search(r"\b([A-Za-z][A-Za-z0-9.]*)\b(?:\s+(.*))?$", clean)
    if not match:
        return None
    mnemonic = match.group(1).lower()
    operands = (match.group(2) or "").strip()
    return mnemonic, operands


def _canonical_number(token: str) -> int | None:
    value = token.strip().lower().replace("_", "")
    value = value.lstrip("#=$")
    value = value.rstrip(",]")
    if not value or value in {"-", "+"}:
        return None
    try:
        if value.startswith("-0x"):
            return -int(value[3:], 16)
        if value.startswith("+0x"):
            return int(value[3:], 16)
        return int(value, 0)
    except ValueError:
        return None


def _extract_asm_numbers(operands: str) -> list[int]:
    tokens = re.findall(r"(?<![A-Za-z_])[-+]?(?:0x[0-9a-fA-F]+|\d+)(?![A-Za-z_])", operands)
    values: list[int] = []
    for token in tokens:
        value = _canonical_number(token)
        if value is not None:
            values.append(value)
    return values


def _semantic_asm_numbers(mnemonic: str, operands: str) -> list[int]:
    """Return immediates likely to describe source semantics, not VM layout.

    Branch destinations, frame sizes, object-pool offsets, memory displacements,
    and Smi-unboxing bitfield widths dominate AOT dumps but are not constants a
    decompiler should reproduce.  The FACTS gate must not reward emitting those
    implementation artefacts.  We retain data-register compares, arithmetic,
    shifts, and immediate moves while conservatively dropping ambiguous address
    calculations through known runtime/frame registers.
    """
    mnemonic = (mnemonic or "").lower()
    if (
        mnemonic in _ASM_BRANCH_MNEMONICS
        or mnemonic in _ASM_CALL_MNEMONICS
        or mnemonic in _ASM_RETURN_MNEMONICS
        or mnemonic in {"adr", "adrp", "lea", "nop", "hint"}
        or mnemonic.startswith(("ld", "st"))
    ):
        return []
    # Remove angle-bracket disassembler annotations and all memory addressing.
    cleaned = re.sub(r"<[^>]*>", "", operands or "")
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(
        r"(?<![A-Za-z_])[-+]?(?:0x[0-9a-fA-F]+|\d+)?\s*\([^)]*\)",
        "",
        cleaned,
    )
    values = _extract_asm_numbers(cleaned)
    if not values:
        return []
    registers = {
        token.lower()
        for token in re.findall(
            r"\b(?:[xw][0-9]+|[er]?(?:ax|bx|cx|dx|si|di|sp|bp)|r(?:[0-9]+|ip)|eip|rip|sp|fp)\b",
            cleaned,
            flags=re.I,
        )
    }
    if mnemonic in _ASM_COMPARE_MNEMONICS or mnemonic in {"cmp", "cmn", "test", "tst"}:
        data_registers = registers - _FRAME_RUNTIME_REGISTERS
        return values if data_registers or not registers else []
    if mnemonic.startswith(("mov", "mvn")):
        destination = (cleaned.split(",", 1)[0] if cleaned else "").strip().lower()
        if destination in _FRAME_RUNTIME_REGISTERS:
            return []
        return values
    if mnemonic in _DATA_IMMEDIATE_MNEMONICS or mnemonic.startswith(("add", "sub", "and", "orr", "eor")):
        if registers & _FRAME_RUNTIME_REGISTERS:
            return []
        return values
    # Unknown instructions are intentionally excluded: a false-negative fact is
    # safer than forcing source code to reproduce a runtime-layout constant.
    return []


def _salient_numbers(values: Iterable[int], limit: int = 24) -> list[int]:
    # Exclude ubiquitous offsets/booleans. Keep negatives, powers not commonly
    # generated as frame offsets, and larger semantic constants.
    selected = {
        value
        for value in values
        if value not in {0, 1, 2, 4, 8, 16}
        and (value < 0 or abs(value) >= 3)
        and abs(value) < 2**31
    }
    return sorted(selected, key=lambda value: (abs(value), value))[:limit]


def _extract_strings(text: str, limit: int = 24) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"(['\"])(.*?)(?<!\\)\1", text or "", flags=re.S):
        value = match.group(2)
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _architecture(assembly: str) -> str:
    lower = assembly.lower()
    if re.search(r"\b(?:x[0-9]+|w[0-9]+|sp|xzr|wzr)\b", lower) and re.search(
        r"\b(?:adrp|stp|ldp|cbz|tbnz|blr)\b", lower
    ):
        return "arm64"
    if re.search(r"\b(?:rax|rbx|rcx|rdx|rsp|rbp|r\d+)\b", lower) or re.search(
        r"\b(?:movq|pushq|callq)\b", lower
    ):
        return "x86_64"
    return "unknown"


def normalize_architecture(value: Any) -> str:
    """Canonicalise common ISA labels used by the unified binary datasets."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "x86__64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "arm64_v8a": "arm64",
    }
    return aliases.get(normalized, normalized or "unknown")


def record_architecture(record: dict[str, Any]) -> str:
    """Return explicit architecture provenance, falling back to assembly."""
    candidates: list[Any] = [
        record.get("architecture"),
        record.get("target_architecture"),
        record.get("target_arch"),
        record.get("isa"),
    ]
    for container_name in ("graph_v2", "assembly_build", "hybrid_metadata"):
        container = record.get(container_name) or {}
        if isinstance(container, dict):
            candidates.extend(
                container.get(key)
                for key in ("architecture", "target_architecture", "target_arch", "isa")
            )
    for value in candidates:
        if value not in (None, ""):
            return normalize_architecture(value)
    return _architecture(str(record.get("assembly") or ""))


def mechanical_facts(record: dict[str, Any]) -> dict[str, Any]:
    """Extract a deterministic, bounded fact sheet from the binary record."""
    assembly = str(record.get("assembly") or "")
    parsed = [item for line in assembly.splitlines() if (item := _parse_asm_line(line))]
    mnemonics = [item[0] for item in parsed]
    numbers = [
        value
        for mnemonic, operands in parsed
        for value in _semantic_asm_numbers(mnemonic, operands)
    ]
    callees: list[str] = []
    for mnemonic, operands in parsed:
        if mnemonic not in _ASM_CALL_MNEMONICS:
            continue
        target = operands.split(",")[-1].strip()
        target = re.sub(r"^[#=$*]+", "", target)
        symbol = re.search(r"<([^>]+)>", target)
        if symbol:
            target = symbol.group(1)
        target = re.sub(r"[+@].*$", "", target).strip()
        if not target or re.fullmatch(r"(?:0x)?[0-9a-fA-F]+", target):
            continue
        if any(fragment in target.lower() for fragment in _RUNTIME_CALLEE_FRAGMENTS):
            continue
        if target not in callees:
            callees.append(target)
        if len(callees) >= 16:
            break

    cfg = record.get("cfg") or []
    edges = record.get("edges") or []
    signature = signature_text(record)
    call_count = sum(mnemonic in _ASM_CALL_MNEMONICS for mnemonic in mnemonics)
    branch_count = sum(mnemonic in _ASM_BRANCH_MNEMONICS for mnemonic in mnemonics)
    return_count = sum(mnemonic in _ASM_RETURN_MNEMONICS for mnemonic in mnemonics)
    memory_ops = sum(
        mnemonic.startswith(("ldr", "str", "ldp", "stp", "mov", "load", "store"))
        for mnemonic in mnemonics
    )
    salient = _salient_numbers(numbers)
    # Only parsed operands are eligible.  Header/disassembler comments often
    # contain paths, symbols, and numeric metadata that are not program literals.
    strings = _extract_strings("\n".join(operands for _, operands in parsed))
    facts = {
        "schema_version": SCHEMA_VERSION,
        "facts_extractor_version": 3,
        "architecture": _architecture(assembly),
        "instruction_count": len(parsed),
        "block_count": len(cfg) if isinstance(cfg, list) and cfg else max(1, sum(1 for line in assembly.splitlines() if line.strip().endswith(":"))),
        "edge_count": len(edges) if isinstance(edges, list) else 0,
        "arity": parameter_arity(signature),
        "return_type": signature_return_type(signature),
        "branch_count": branch_count,
        "call_count": call_count,
        "return_count": return_count,
        "comparisons": sum(mnemonic in _ASM_COMPARE_MNEMONICS for mnemonic in mnemonics),
        "memory_ops": memory_ops,
        "salient_numeric_constants": salient,
        "string_literals": strings,
        "direct_callees": callees,
        # Compatibility aliases used by the compact structured-text prompt.
        "calls": call_count,
        "conditional_branches": max(0, branch_count - call_count),
        "returns": return_count,
        "available_constants": [str(value) for value in salient],
        "available_strings": strings,
    }
    return facts


def facts_comment(facts: dict[str, Any]) -> str:
    compact = json.dumps(normalize_fact_claims(facts), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"// FACTS_JSON: {compact}"


def parse_facts_comment(code: str) -> dict[str, Any] | None:
    match = re.search(r"(?m)^\s*//\s*FACTS_JSON\s*:\s*(\{.*\})\s*$", code or "")
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalize_fact_claims(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = value or {}
    scalar_ints = (
        "instruction_count", "block_count", "edge_count", "arity", "branch_count",
        "call_count", "return_count", "comparisons",
    )
    normalized: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    normalized["architecture"] = str(raw.get("architecture") or "unknown").lower()
    for key in scalar_ints:
        item = raw.get(key)
        normalized[key] = int(item) if item is not None and str(item).strip() not in {"", "None"} else None
    return_type = raw.get("return_type")
    normalized["return_type"] = re.sub(r"\s+", " ", str(return_type).strip()) if return_type else None
    for key in ("salient_numeric_constants", "string_literals", "direct_callees"):
        values = raw.get(key) or []
        if not isinstance(values, list):
            values = [values]
        if key == "salient_numeric_constants":
            canonical: list[int] = []
            for item in values:
                try:
                    number = int(item)
                except (TypeError, ValueError):
                    continue
                if number not in canonical:
                    canonical.append(number)
            normalized[key] = sorted(canonical)
        else:
            normalized[key] = sorted({str(item).strip() for item in values if str(item).strip()})
    return normalized


def _code_signature(code: str, expected_name: str) -> str:
    clean = _strip_comments(code)
    name_pattern = re.escape(expected_name) if expected_name else r"[A-Za-z_]\w*"
    match = re.search(
        rf"(?m)^\s*((?:(?:external|static|late|final|const)\s+)*"
        rf"[A-Za-z_][\w<>,?\[\]\s]*(?:\s+))({name_pattern})\s*\(([^)]*)\)",
        clean,
    )
    if not match:
        return ""
    return f"{match.group(1).strip()} {match.group(2)}({match.group(3).strip()})"


def code_facts(code: str, expected_name: str = NEUTRAL_FUNCTION_NAME) -> dict[str, Any]:
    signature = _code_signature(code, expected_name)
    clean = _strip_comments(code)
    numeric_values: list[int] = []
    for token in re.findall(r"(?<![A-Za-z_])[-+]?(?:0x[0-9a-fA-F]+|\d+)(?![A-Za-z_])", clean):
        value = _canonical_number(token)
        if value is not None:
            numeric_values.append(value)
    calls = []
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", clean):
        name = match.group(1)
        if name == expected_name or name in _DART_KEYWORDS:
            continue
        if name not in calls:
            calls.append(name)
    return {
        "arity": parameter_arity(signature),
        "return_type": signature_return_type(signature),
        "salient_numeric_constants": _salient_numbers(numeric_values),
        "string_literals": _extract_strings(clean),
        "direct_callees": calls[:16],
    }


def fact_claim_match(
    expected: dict[str, Any],
    claimed: dict[str, Any] | None,
    *,
    require_complete: bool = True,
) -> tuple[bool, list[str]]:
    expected_n = normalize_fact_claims(expected)
    claimed_n = normalize_fact_claims(claimed)
    reasons: list[str] = []
    required_keys = (
        "architecture", "instruction_count", "block_count", "edge_count", "arity",
        "return_type", "branch_count", "call_count", "return_count", "comparisons",
        "salient_numeric_constants", "string_literals", "direct_callees",
    )
    raw_claimed = claimed or {}
    if require_complete:
        for key in required_keys:
            if key not in raw_claimed:
                reasons.append(f"teacher fact claims omitted {key}")
    for key in required_keys:
        if key in raw_claimed and claimed_n.get(key) != expected_n.get(key):
            reasons.append(
                f"teacher fact claim mismatch for {key}: expected={expected_n.get(key)!r} "
                f"claimed={claimed_n.get(key)!r}"
            )
    return not reasons, reasons


def candidate_fact_match(
    task: dict[str, Any],
    code: str,
    *,
    mode: str = "conservative",
    teacher_claim: dict[str, Any] | None = None,
    claims: dict[str, Any] | None = None,
    require_claims: bool = False,
) -> tuple[bool, list[str]]:
    """Conservative static gate complementing hidden executable tests.

    ``signature`` checks the neutral typed contract only. ``conservative`` also
    requires any assembly-visible string literal and a bounded subset of
    salient constants to survive into the proposed source. The teacher's facts
    must be a complete exact copy of mechanical facts when ``require_claims`` is
    true; this prevents unstructured post-hoc rationales from being accepted.
    """
    if mode not in {"off", "signature", "conservative", "strict"}:
        raise ValueError(f"unknown facts gate mode: {mode}")
    if mode == "off":
        return True, []
    if teacher_claim is None and claims is not None:
        teacher_claim = claims
    expected_name = infer_function_name(task) or NEUTRAL_FUNCTION_NAME
    expected = normalize_fact_claims(task.get("binary_facts") or mechanical_facts(task))
    # GDB/objdump headers frequently quote the selected symbol in text such as
    # ``All functions matching regular expression "fn0"``. Version-2 fact
    # sheets accidentally preserved that tool metadata as a program string.
    # Filter it at verification time as well as in the v3 extractor so legacy
    # harvests can be re-certified without weakening genuine string checks.
    expected["string_literals"] = [
        value
        for value in expected.get("string_literals") or []
        if value != expected_name
    ]
    observed = code_facts(code, expected_name)
    reasons: list[str] = []
    if observed.get("arity") != expected.get("arity"):
        reasons.append(
            f"arity mismatch: expected={expected.get('arity')} observed={observed.get('arity')}"
        )
    expected_return = expected.get("return_type")
    observed_return = observed.get("return_type")
    if expected_return and observed_return and expected_return != observed_return:
        reasons.append(
            f"return type mismatch: expected={expected_return!r} observed={observed_return!r}"
        )
    if not _code_signature(code, expected_name):
        reasons.append(f"required top-level function {expected_name!r} was not found")

    if mode in {"conservative", "strict"}:
        missing_strings = sorted(
            set(expected.get("string_literals") or []) - set(observed.get("string_literals") or [])
        )
        if missing_strings:
            reasons.append(f"missing assembly-visible string literals: {missing_strings[:6]}")
        expected_numbers = set(expected.get("salient_numeric_constants") or [])
        observed_numbers = set(observed.get("salient_numeric_constants") or [])
        if expected_numbers:
            overlap = len(expected_numbers & observed_numbers) / len(expected_numbers)
            threshold = 1.0 if mode == "strict" else 0.34
            if overlap < threshold:
                reasons.append(
                    f"salient numeric-constant overlap {overlap:.3f} below {threshold:.3f}"
                )

    if require_claims or teacher_claim is not None:
        audited_claim = dict(teacher_claim or {})
        if "string_literals" in audited_claim:
            audited_claim["string_literals"] = [
                value
                for value in audited_claim.get("string_literals") or []
                if value != expected_name
            ]
        claims_ok, claim_reasons = fact_claim_match(
            expected,
            audited_claim,
            require_complete=require_claims,
        )
        if not claims_ok:
            reasons.extend(claim_reasons)
    return not reasons, reasons


def facts_match_score(expected: dict[str, Any], predicted: dict[str, Any] | None) -> float:
    """Bounded diagnostic score for free-running FACTS_JSON generations."""
    if not predicted:
        return 0.0
    exp = normalize_fact_claims(expected)
    pred = normalize_fact_claims(predicted)
    keys = (
        "architecture", "arity", "return_type", "branch_count", "call_count",
        "return_count", "comparisons", "salient_numeric_constants", "string_literals",
        "direct_callees",
    )
    scores: list[float] = []
    for key in keys:
        left, right = exp.get(key), pred.get(key)
        if isinstance(left, list):
            left_set, right_set = set(left), set(right or [])
            if not left_set and not right_set:
                scores.append(1.0)
            elif not left_set:
                scores.append(0.0)
            else:
                scores.append(len(left_set & right_set) / len(left_set | right_set or {None}))
        else:
            scores.append(1.0 if left == right else 0.0)
    return sum(scores) / len(scores)


def instruction_count(record: dict[str, Any]) -> int:
    facts = record.get("binary_facts") or mechanical_facts(record)
    return int(facts.get("instruction_count") or 0)


def length_bin(value: int) -> str:
    if value < 60:
        return "lt60"
    if value < 90:
        return "60_89"
    if value < 120:
        return "90_119"
    if value < 150:
        return "120_149"
    if value < 200:
        return "150_199"
    if value < 300:
        return "200_299"
    return "ge300"


def sanitize_verifier_diagnostic(text: str, max_chars: int = 2400) -> str:
    """Remove test-oracle values while retaining compiler/runtime categories."""
    text = (text or "").replace("\x00", "")
    text = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s:\n]+[/\\]", "<path>/", text)
    output: list[str] = []
    redacted_assertion = False
    sensitive = re.compile(
        r"\b(expected|actual|input|output|received|wanted|got|value|values)\b\s*[:=]",
        flags=re.I,
    )
    assertion_line = re.compile(r"\b(expect|assert|matcher|comparison failed)\b", flags=re.I)
    comparison_operand = (
        r"(?:[-+]?(?:\d+(?:\.\d+)?|infinity|nan)|true|false|null|"
        r"<[^>\r\n]{0,200}>|\[[^\]\r\n]{0,200}\]|\{[^}\r\n]{0,200}\}|"
        r"['\"][^'\"\r\n]{0,200}['\"]|[A-Za-z_]\w*)"
    )
    bare_assertion_comparison = re.compile(
        rf"^(?:unhandled exception:\s*)?{comparison_operand}\s*(?:!=|==)\s*"
        rf"{comparison_operand}\s*$",
        flags=re.I,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if (
            sensitive.search(line)
            or assertion_line.search(line)
            or bare_assertion_comparison.fullmatch(line)
        ):
            redacted_assertion = True
            continue
        # Remove source snippets/caret diagnostics likely to reveal assertions.
        if "candidate(" in line or re.match(r"^[\s^~|+-]+$", line):
            redacted_assertion = True
            continue
        # Retain compiler error type/location, but redact quoted literal values.
        line = re.sub(r"(['\"])(?:\\.|(?!\1).)*\1", "<literal>", line)
        output.append(line)
    if redacted_assertion:
        output.append("[assertion mismatch values redacted]")
    compact = "\n".join(output)
    if len(compact) <= max_chars:
        return compact
    head = max_chars * 3 // 5
    tail = max_chars - head
    return compact[:head] + "\n... <diagnostic middle omitted> ...\n" + compact[-tail:]


def verified_origin(record: dict[str, Any]) -> bool:
    """Return true only for independently replayed non-gold positives.

    A row is not a verified alternative merely because its final full harness
    passed.  Its visible feedback, hidden acceptance suite, and mechanical
    assembly-fact gate must all have been replayed and recorded explicitly.
    In particular, ``facts_gate_passed=True`` is not sufficient on its own:
    older ``mode=off`` harvests incorrectly wrote that value even though no
    factual check ran.  New verified rows must bind both the applied mode and
    its result.
    """
    metadata = record.get("hybrid_metadata") or {}
    origin = str(metadata.get("origin") or "")
    if origin not in {"teacher_repair", "model_rollout", "verified_rollout"}:
        return False
    facts_mode = str(metadata.get("facts_gate_mode") or "").strip().lower()
    facts_gate_certified = bool(
        metadata.get("facts_gate_applied") is True
        and facts_mode in {"signature", "conservative", "strict"}
        and metadata.get("facts_gate_passed") is True
    )
    return bool(
        metadata.get("verifier_replayed")
        and metadata.get("feedback_replayed")
        and metadata.get("feedback_tests_passed")
        and metadata.get("verifier_full_pass")
        and metadata.get("hidden_acceptance_replayed")
        and metadata.get("acceptance_tests_passed")
        and facts_gate_certified
    )


def assert_training_approved(record: dict[str, Any]) -> None:
    """Fail closed unless a row is bound to the Phase-0 training manifest."""
    metadata = record.get("hybrid_metadata") or {}
    identity = task_identity(record)
    if not metadata.get("phase0_approved"):
        raise ValueError(f"{identity} is not Phase-0 approved")
    if metadata.get("evaluation_only"):
        raise ValueError(f"{identity} is marked evaluation-only")
    if not (metadata.get("neutralized") or metadata.get("neutral_contract")):
        raise ValueError(f"{identity} lacks a neutral contract")
    replay = metadata.get("reference_test_replay") or {}
    if replay.get("passed") is not True:
        raise ValueError(f"{identity} lacks successful Phase-0 reference replay")
    if not metadata.get("source_overlap_hash"):
        raise ValueError(f"{identity} lacks a frozen-evaluation overlap hash")
    role = str(metadata.get("data_role") or "").lower()
    if role not in {"train", "development", "dev"}:
        raise ValueError(f"{identity} has invalid training data role {role!r}")
    if not record.get("binary_facts"):
        raise ValueError(f"{identity} lacks deterministic assembly facts")
    if not record.get("acceptance_tests"):
        raise ValueError(f"{identity} lacks hidden acceptance tests")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": file_sha256(value),
    }

# Compatibility names used by the hybrid training scripts. They remain public
# here so all gate logic has one implementation rather than diverging copies.
FACT_FIELDS = (
    "architecture",
    "instruction_count",
    "block_count",
    "edge_count",
    "arity",
    "return_type",
    "branch_count",
    "call_count",
    "return_count",
    "comparisons",
    "salient_numeric_constants",
    "string_literals",
    "direct_callees",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_many(path)


def task_key(record: dict[str, Any], index: int | None = None) -> str:
    return task_identity(record, index)


def normalize_fact_sheet(value: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_fact_claims(value)
    return {key: normalized.get(key) for key in FACT_FIELDS}


def candidate_expect_lines(test_code: str) -> list[str]:
    return candidate_assertions(test_code)


def _subset_test_harness(test_code: str, keep_assertions: set[str]) -> str:
    assertions = candidate_assertions(test_code)
    keep_indices = {index for index, assertion in enumerate(assertions) if assertion in keep_assertions}
    if not keep_indices:
        raise ValueError("requested test subset does not contain a candidate assertion")
    return _rebuild_test_harness(test_code, keep_indices)
