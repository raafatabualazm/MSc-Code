#!/usr/bin/env python3
"""Extract every user-file function from a Dart AOT ELF into a name-free form.

The old graph builder disassembled only the exact ``candidate`` symbol.  That
silently loses top-level helpers, named local functions, and closure bodies.
This extractor uses a keyed private source-symbol attestation (never source
text or raw attested names) and:

1. asks GDB for its debug ``File ...:`` function table;
2. finds the *single* file section containing the exact root symbol;
3. disassembles every function in that same file section;
4. assigns source-neutral IDs (``F0`` is the root, then address order);
5. emits a lossless per-function instruction/CFG projection and all call sites.

No Dart source path, declaration, or raw user symbol is serialized.  SDK and
runtime call labels are retained through an ``X#`` dictionary because erasing
those labels destroys useful semantics.

The corpus CLI is intentionally two phase.  Extraction never truncates a row.
An optional encoded-measurement JSONL, bound to each canonical model-projection
SHA-256, enforces the production 9K student / 12K API gates after a shared
multi-function codebook has been fitted.  ``--require-budget-measurements``
prevents an extraction-only corpus from being mistaken for production-ready
training input.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.build_dart_user_symbol_attestation import (
    SCHEMA as SYMBOL_ATTESTATION_SCHEMA,
    key_id_sha256 as attestation_key_id_sha256,
    load_key as load_attestation_key,
    ordered_commitment as attestation_ordered_commitment,
    row_salt as attestation_row_salt,
    symbol_digest as attested_symbol_digest,
)

SCHEMA = "dart-aot-user-function-bundle-v1"
MODEL_SCHEMA = "dart-aot-multifunction-graph-canonical-v1"
REPORT_SCHEMA = "dart-aot-user-function-corpus-preflight-v1"
MEASUREMENT_SCHEMA = "dart-aot-multifunction-encoded-measurement-v1"
SOURCE_ONLY_AOT_ROW_SCHEMA = "phase0-s44-source-only-aot-row-v1"
SOURCE_ONLY_BUILD_INPUT_SCHEMA = "dart-source-only-aot-build-input-v1"
SCAFFOLD_CONTRACT = "source-only-program-empty-main-v1"
DEFAULT_ROOT_SYMBOL = "candidate"
DEFAULT_STUDENT_BUDGET = 9000
DEFAULT_API_BUDGET = 12000

RUNTIME_SYMBOL_POLICY = {
    "version": "dart-aot-external-symbol-policy-v2",
    "retain_exact": ["print"],
    "retain_prefixes": ["dart:", "stub _iso_stub_", "_kDart"],
    "retain_gdb_file_prefixes": ["dart:"],
    "neutralize_exact": ["unknown"],
    "neutralize_prefixes": ["runtime ", "stub "],
    "otherwise": (
        "neutralize_after_complete_per_task_symbol_attestation"
        "_and_dart_only_import_attestation"
    ),
}

X86_CONDITIONAL = {
    "ja",
    "jae",
    "jb",
    "jbe",
    "je",
    "jecxz",
    "jg",
    "jge",
    "jl",
    "jle",
    "jne",
    "jno",
    "jnp",
    "jns",
    "jnz",
    "jo",
    "jp",
    "jpe",
    "jpo",
    "jrcxz",
    "js",
    "jz",
    "loop",
    "loope",
    "loopne",
    "loopnz",
    "loopz",
}
X86_UNCONDITIONAL = {"jmp", "jmpq"}
X86_CALLS = {"call", "callq"}
X86_RETURNS = {"ret", "retq"}
X86_TRAPS = {"hlt", "int3", "ud2"}
CONTROL_PREFIXES = {"bnd", "notrack"}

FILE_HEADER_RE = re.compile(r"^File (?P<file>.+):\s*$")
DEBUG_DECLARATION_RE = re.compile(
    r"^\s*(?P<line>[0-9]+):\s*"
    r"(?:(?:static|instance)\s+)?void\s+"
    r"(?P<symbol>.+?)\(void\);\s*$"
)
DISASSEMBLY_HEADER_RE = re.compile(
    r"^Dump of assembler code for function (?P<symbol>.+):\s*$"
)
DISASSEMBLY_LINE_RE = re.compile(
    r"^\s*(?:=>\s*)?"
    r"(?P<address>0x[0-9a-fA-F]+)\s+"
    r"<(?P<label>.+)>:\s*(?P<body>.*?)\s*$"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TASK_ID_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")


class UserFunctionExtractionError(ValueError):
    """Raised when complete, public-name-free extraction cannot be proven."""


class AttestedSymbols:
    """Verify and query one private, name-free symbol attestation row."""

    def __init__(self, row: Mapping[str, Any], key: bytes) -> None:
        if row.get("schema") != SYMBOL_ATTESTATION_SCHEMA:
            raise UserFunctionExtractionError(
                "symbol_attestation_schema_mismatch"
            )
        self.row = dict(row)
        self.key = key
        self.task_id = str(row.get("task_id") or "")
        self.salt_hex = str(row.get("salt_hex") or "")
        if (
            not TASK_ID_RE.fullmatch(self.task_id)
            or str(row.get("key_id_sha256") or "")
            != attestation_key_id_sha256(key)
        ):
            raise UserFunctionExtractionError(
                "symbol_attestation_key_binding_mismatch"
            )
        for field in (
            "analysis_program_sha256",
            "function_source_sha256",
            "producer_script_sha256",
        ):
            if not SHA256_RE.fullmatch(
                str(row.get(field) or "").lower()
            ):
                raise UserFunctionExtractionError(
                    f"symbol_attestation_invalid_{field}"
                )
        if not re.fullmatch(r"[0-9a-f]{32}", self.salt_hex):
            raise UserFunctionExtractionError(
                "symbol_attestation_salt_malformed"
            )
        expected_salt = attestation_row_salt(
            key,
            task_id=self.task_id,
            analysis_program_sha256=str(
                row["analysis_program_sha256"]
            ).lower(),
        )
        if not hmac.compare_digest(self.salt_hex, expected_salt):
            raise UserFunctionExtractionError(
                "symbol_attestation_salt_binding_mismatch"
            )
        completeness = row.get("completeness")
        if not isinstance(completeness, Mapping) or (
            completeness.get("complete_source_symbols_projection") is not True
            or completeness.get("source_symbols_bound_to_transform_metadata")
            is not True
            or completeness.get("only_dart_scheme_imports") is not True
        ):
            raise UserFunctionExtractionError(
                "symbol_attestation_incomplete"
            )
        self.function_entries = self._validate_entries(
            row.get("function_symbols"), "function", "AF"
        )
        self.type_entries = self._validate_entries(
            row.get("type_symbols"), "type", "T"
        )
        if int(completeness.get("ordered_function_count", -1)) != len(
            self.function_entries
        ) or int(completeness.get("ordered_type_count", -1)) != len(
            self.type_entries
        ):
            raise UserFunctionExtractionError(
                "symbol_attestation_count_mismatch"
            )
        expected_commitment = attestation_ordered_commitment(
            key,
            task_id=self.task_id,
            salt_hex=self.salt_hex,
            function_digests=[
                entry["digest"] for entry in self.function_entries
            ],
            type_digests=[entry["digest"] for entry in self.type_entries],
        )
        if completeness.get("ordered_commitment") != expected_commitment:
            raise UserFunctionExtractionError(
                "symbol_attestation_commitment_mismatch"
            )
        self.row_sha256 = canonical_sha256(row)

    @staticmethod
    def _validate_entries(
        raw: Any, kind: str, alias_prefix: str
    ) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            raise UserFunctionExtractionError(
                f"symbol_attestation_{kind}_list_missing"
            )
        result: list[dict[str, str]] = []
        digests: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise UserFunctionExtractionError(
                    f"symbol_attestation_{kind}_entry_invalid"
                )
            alias = str(item.get("alias") or "")
            digest = str(item.get("digest") or "").lower()
            if alias != f"{alias_prefix}{index}" or not SHA256_RE.fullmatch(
                digest
            ):
                raise UserFunctionExtractionError(
                    f"symbol_attestation_{kind}_entry_malformed"
                )
            if digest in digests:
                raise UserFunctionExtractionError(
                    f"symbol_attestation_{kind}_digest_duplicate"
                )
            digests.add(digest)
            result.append({"alias": alias, "digest": digest})
        return result

    def _match(self, kind: str, symbol: str) -> str | None:
        symbol = unicodedata.normalize("NFC", str(symbol))
        entries = (
            self.function_entries if kind == "function" else self.type_entries
        )
        matches: list[str] = []
        for index, entry in enumerate(entries):
            digest = attested_symbol_digest(
                self.key,
                task_id=self.task_id,
                salt_hex=self.salt_hex,
                kind=kind,
                index=index,
                symbol=symbol,
            )
            if hmac.compare_digest(digest, entry["digest"]):
                matches.append(entry["alias"])
        if len(matches) > 1:
            raise UserFunctionExtractionError(
                f"ambiguous_attested_{kind}_symbol"
            )
        return matches[0] if matches else None

    def match_function(self, symbol: str) -> str | None:
        return self._match("function", symbol)

    def match_type(self, symbol: str) -> str | None:
        return self._match("type", symbol)

    @property
    def public_type_aliases(self) -> list[str]:
        return [entry["alias"] for entry in self.type_entries]

    @property
    def public_function_aliases(self) -> list[str]:
        return [entry["alias"] for entry in self.function_entries]

    def verify_source_contract(
        self, source_only_contract: Mapping[str, Any]
    ) -> None:
        expected = {
            "analysis_program_sha256": self.row.get(
                "analysis_program_sha256"
            ),
            "function_source_sha256": self.row.get(
                "function_source_sha256"
            ),
            "producer_script_sha256": self.row.get(
                "producer_script_sha256"
            ),
        }
        observed = {
            key: str(source_only_contract.get(key) or "").lower()
            for key in expected
        }
        if observed != expected:
            raise UserFunctionExtractionError(
                "symbol_attestation_source_contract_mismatch"
            )

    def public_binding(self, attestation_file_sha256: str) -> dict[str, Any]:
        return {
            "schema": SYMBOL_ATTESTATION_SCHEMA,
            "attestation_file_sha256": attestation_file_sha256,
            "attestation_row_sha256": self.row_sha256,
            "key_id_sha256": str(self.row["key_id_sha256"]),
            "function_symbol_count": len(self.function_entries),
            "type_symbol_count": len(self.type_entries),
            "complete": True,
            "raw_names_present": False,
        }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise UserFunctionExtractionError(
                    f"blank_jsonl_line:{path}:{line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise UserFunctionExtractionError(
                    f"invalid_jsonl:{path}:{line_number}:{error}"
                ) from error
            if not isinstance(value, dict):
                raise UserFunctionExtractionError(
                    f"non_object_jsonl_row:{path}:{line_number}"
                )
            values.append(value)
    return values


def _parse_debug_sections(info_output: str) -> list[dict[str, Any]]:
    """Parse GDB's debug function table, retaining paths only in memory."""

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in info_output.splitlines():
        header = FILE_HEADER_RE.match(raw_line.strip())
        if header:
            current = {
                "raw_file": header.group("file"),
                "symbols": [],
                "unparsed_declarations": [],
            }
            sections.append(current)
            continue
        if raw_line.strip() == "Non-debugging symbols:":
            current = None
            continue
        if current is None:
            continue
        declaration = DEBUG_DECLARATION_RE.match(raw_line)
        if declaration:
            symbol = declaration.group("symbol").strip()
            if not symbol or "\n" in symbol or "\r" in symbol:
                raise UserFunctionExtractionError("invalid_gdb_debug_symbol")
            current["symbols"].append(
                {
                    "raw_symbol": symbol,
                    "source_line": int(declaration.group("line")),
                }
            )
        elif re.match(r"^\s*[0-9]+:", raw_line):
            # If the selected file contains a declaration syntax this parser
            # does not understand, proceeding would silently omit a function.
            current["unparsed_declarations"].append(raw_line.strip())
    return sections


def select_same_file_symbols(
    info_output: str, root_symbol: str = DEFAULT_ROOT_SYMBOL
) -> tuple[str, list[str]]:
    """Return the private file identity and every symbol in the root's section.

    The caller must never serialize the returned file identity or symbols.
    """

    if not root_symbol or any(character in root_symbol for character in "\r\n"):
        raise UserFunctionExtractionError("invalid_root_symbol")
    sections = _parse_debug_sections(info_output)
    matches: list[dict[str, Any]] = []
    for section in sections:
        symbols = [entry["raw_symbol"] for entry in section["symbols"]]
        if root_symbol in symbols:
            matches.append(section)
    if len(matches) != 1:
        raise UserFunctionExtractionError(
            f"root_file_section_count:{len(matches)}:{root_symbol}"
        )
    selected = matches[0]
    if selected["unparsed_declarations"]:
        raise UserFunctionExtractionError(
            "unparsed_declarations_in_root_file:"
            + str(len(selected["unparsed_declarations"]))
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for entry in selected["symbols"]:
        symbol = entry["raw_symbol"]
        if symbol in seen:
            raise UserFunctionExtractionError(
                f"duplicate_symbol_in_root_file:{symbol}"
            )
        seen.add(symbol)
        ordered.append(symbol)
    if root_symbol not in seen:
        raise UserFunctionExtractionError("root_symbol_missing_after_selection")
    if not ordered:
        raise UserFunctionExtractionError("empty_root_file_section")
    return str(selected["raw_file"]), ordered


def external_symbol_evidence(
    info_output: str, selected_file_identity: str
) -> tuple[set[str], set[str]]:
    """Return private trusted-runtime and non-runtime debug symbol sets.

    Only symbols whose GDB File identity begins with ``dart:`` are eligible to
    retain a label. The second set is private diagnostic evidence for any
    other non-selected debug file. The keyed attestation separately proves
    complete source declarations and dart-only imports. Neither set is
    serialized.
    """

    trusted: set[str] = set()
    nonruntime: set[str] = set()
    for section in _parse_debug_sections(info_output):
        raw_file = str(section["raw_file"])
        symbols = {
            str(entry["raw_symbol"]) for entry in section["symbols"]
        }
        if raw_file == selected_file_identity:
            continue
        if any(
            raw_file.startswith(prefix)
            for prefix in RUNTIME_SYMBOL_POLICY["retain_gdb_file_prefixes"]
        ):
            trusted.update(symbols)
        else:
            nonruntime.update(symbols)
    return trusted, nonruntime


def _parse_offset_label(label: str) -> int:
    match = re.search(r"\+(?P<offset>0x[0-9a-fA-F]+|[0-9]+)\Z", label)
    if not match:
        raise UserFunctionExtractionError(f"missing_gdb_function_offset:{label}")
    value = match.group("offset")
    return int(value, 16 if value.lower().startswith("0x") else 10)


def _split_machine_bytes(body: str) -> tuple[str, str]:
    tokens = body.split()
    byte_tokens: list[str] = []
    while tokens and re.fullmatch(r"[0-9a-fA-F]{2}", tokens[0]):
        byte_tokens.append(tokens.pop(0).lower())
    if not byte_tokens:
        raise UserFunctionExtractionError("gdb_raw_bytes_missing")
    instruction = " ".join(tokens).strip()
    if not instruction:
        raise UserFunctionExtractionError("gdb_instruction_missing")
    return "".join(byte_tokens), instruction


def parse_gdb_disassembly(raw: str, expected_symbol: str) -> dict[str, Any]:
    """Parse one ``disassemble /r`` block and prove byte/offset accounting."""

    lines = raw.splitlines()
    header_positions = [
        position
        for position, line in enumerate(lines)
        if DISASSEMBLY_HEADER_RE.match(line.strip())
    ]
    if len(header_positions) != 1:
        raise UserFunctionExtractionError(
            f"disassembly_header_count:{expected_symbol}:{len(header_positions)}"
        )
    start = header_positions[0]
    header = DISASSEMBLY_HEADER_RE.match(lines[start].strip())
    assert header is not None
    if header.group("symbol") != expected_symbol:
        raise UserFunctionExtractionError(
            f"disassembly_symbol_mismatch:{header.group('symbol')}!={expected_symbol}"
        )
    end_positions = [
        position
        for position in range(start + 1, len(lines))
        if lines[position].strip() == "End of assembler dump."
    ]
    if not end_positions:
        raise UserFunctionExtractionError(
            f"unterminated_disassembly:{expected_symbol}"
        )
    end = end_positions[0]
    instructions: list[dict[str, Any]] = []
    for line in lines[start + 1 : end]:
        match = DISASSEMBLY_LINE_RE.match(line)
        if not match:
            if line.strip():
                raise UserFunctionExtractionError(
                    f"unparsed_disassembly_line:{expected_symbol}:{line.strip()[:160]}"
                )
            continue
        address = int(match.group("address"), 16)
        offset = _parse_offset_label(match.group("label"))
        machine_bytes, text = _split_machine_bytes(match.group("body"))
        instructions.append(
            {
                "address": address,
                "offset": offset,
                "machine_bytes": machine_bytes,
                "raw_text": text,
            }
        )
    if not instructions:
        raise UserFunctionExtractionError(f"empty_disassembly:{expected_symbol}")
    zero = [instruction for instruction in instructions if instruction["offset"] == 0]
    if len(zero) != 1:
        raise UserFunctionExtractionError(
            f"function_entry_count:{expected_symbol}:{len(zero)}"
        )
    entry_address = zero[0]["address"]
    instructions.sort(key=lambda instruction: instruction["offset"])
    seen_offsets: set[int] = set()
    seen_addresses: set[int] = set()
    for instruction in instructions:
        if instruction["offset"] in seen_offsets:
            raise UserFunctionExtractionError(
                f"duplicate_instruction_offset:{expected_symbol}:{instruction['offset']}"
            )
        if instruction["address"] in seen_addresses:
            raise UserFunctionExtractionError(
                f"duplicate_instruction_address:{expected_symbol}:"
                f"0x{instruction['address']:x}"
            )
        seen_offsets.add(instruction["offset"])
        seen_addresses.add(instruction["address"])
        expected_address = entry_address + instruction["offset"]
        if instruction["address"] != expected_address:
            raise UserFunctionExtractionError(
                f"address_offset_mismatch:{expected_symbol}:"
                f"0x{instruction['address']:x}!=0x{expected_address:x}"
            )
    code_size = max(
        instruction["offset"] + len(instruction["machine_bytes"]) // 2
        for instruction in instructions
    )
    return {
        "raw_symbol": expected_symbol,
        "entry_address": entry_address,
        "code_size_bytes": code_size,
        "instructions": instructions,
        "raw_sha256": sha256_bytes(raw.encode("utf-8")),
    }


def parse_combined_gdb_disassemblies(
    output: str, expected_symbols: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Split a batch GDB output into exact, one-per-request disassemblies."""

    lines = output.splitlines()
    starts: list[tuple[int, str]] = []
    for position, line in enumerate(lines):
        match = DISASSEMBLY_HEADER_RE.match(line.strip())
        if match:
            starts.append((position, match.group("symbol")))
    chunks: dict[str, str] = {}
    for position, (start, symbol) in enumerate(starts):
        stop = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        chunk_lines = lines[start:stop]
        # Keep only through this dump's terminator; GDB diagnostics between
        # commands must not be accepted as assembly.
        terminator = next(
            (
                index
                for index, line in enumerate(chunk_lines)
                if line.strip() == "End of assembler dump."
            ),
            None,
        )
        if terminator is None:
            raise UserFunctionExtractionError(
                f"unterminated_batch_disassembly:{symbol}"
            )
        if symbol in chunks:
            raise UserFunctionExtractionError(f"duplicate_disassembly:{symbol}")
        chunks[symbol] = "\n".join(chunk_lines[: terminator + 1]) + "\n"
    expected = set(expected_symbols)
    actual = set(chunks)
    if len(expected) != len(expected_symbols):
        raise UserFunctionExtractionError("duplicate_expected_symbols")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise UserFunctionExtractionError(
            f"disassembly_set_mismatch:missing={missing}:extra={extra}"
        )
    return {
        symbol: parse_gdb_disassembly(chunks[symbol], symbol)
        for symbol in expected_symbols
    }


def parse_combined_gdb_disassemblies_by_address(
    output: str, expected_targets: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Parse address-requested dumps and prove each exact entry address.

    GDB can expose duplicate function names (observed for ``new StateError``).
    Therefore recursive implicit-function recovery must request the direct call
    operand address, preserve command order, and reject any returned function
    whose entry is not exactly that address.
    """

    lines = output.splitlines()
    starts: list[tuple[int, str]] = []
    for position, line in enumerate(lines):
        match = DISASSEMBLY_HEADER_RE.match(line.strip())
        if match:
            starts.append((position, match.group("symbol")))
    if len(starts) != len(expected_targets):
        raise UserFunctionExtractionError(
            "address_disassembly_count_mismatch:"
            f"{len(starts)}!={len(expected_targets)}"
        )

    parsed_by_identity: dict[str, dict[str, Any]] = {}
    seen_addresses: set[int] = set()
    for index, ((start, header_symbol), target) in enumerate(
        zip(starts, expected_targets, strict=True)
    ):
        expected_address = int(target["address"])
        if expected_address in seen_addresses:
            raise UserFunctionExtractionError(
                f"duplicate_address_disassembly_target:"
                f"0x{expected_address:x}"
            )
        seen_addresses.add(expected_address)
        stop = starts[index + 1][0] if index + 1 < len(starts) else len(lines)
        chunk_lines = lines[start:stop]
        terminator = next(
            (
                offset
                for offset, line in enumerate(chunk_lines)
                if line.strip() == "End of assembler dump."
            ),
            None,
        )
        if terminator is None:
            raise UserFunctionExtractionError(
                f"unterminated_address_disassembly:"
                f"0x{expected_address:x}"
            )
        parsed = parse_gdb_disassembly(
            "\n".join(chunk_lines[: terminator + 1]) + "\n",
            header_symbol,
        )
        if int(parsed["entry_address"]) != expected_address:
            raise UserFunctionExtractionError(
                f"address_disassembly_entry_mismatch:"
                f"0x{int(parsed['entry_address']):x}"
                f"!=0x{expected_address:x}"
            )
        identity = f"__attested_entry_0x{expected_address:x}"
        parsed["attested_recovery_kind"] = str(target["recovery_kind"])
        parsed_by_identity[identity] = parsed
    return parsed_by_identity


def discover_attested_direct_callees(
    parsed_by_symbol: Mapping[str, Mapping[str, Any]],
    attestation: AttestedSymbols,
) -> list[dict[str, Any]]:
    """Find directly reachable user functions omitted from GDB's File table."""

    discovered: dict[int, dict[str, Any]] = {}
    existing_ranges = [
        (
            int(function["entry_address"]),
            int(function["entry_address"])
            + int(function["code_size_bytes"]),
        )
        for function in parsed_by_symbol.values()
    ]
    for function in parsed_by_symbol.values():
        for instruction in function["instructions"]:
            raw_text = str(instruction["raw_text"])
            if _opcode(raw_text) not in X86_CALLS | X86_UNCONDITIONAL:
                continue
            annotation = _annotation(raw_text)
            semantics = _attested_annotation_semantics(
                annotation, attestation
            )
            if not semantics or not semantics["user_callable"]:
                continue
            address = _direct_address(raw_text)
            if address is None:
                raise UserFunctionExtractionError(
                    "attested_user_callable_without_direct_address"
                )
            if any(start <= address < end for start, end in existing_ranges):
                continue
            target = {
                "address": address,
                "raw_annotation": str(
                    semantics["raw_callable_symbol"]
                ),
                "recovery_kind": "constructor",
            }
            prior = discovered.get(address)
            if prior is not None and prior != target:
                raise UserFunctionExtractionError(
                    f"conflicting_attested_target_at_address:0x{address:x}"
                )
            discovered[address] = target
    return [discovered[address] for address in sorted(discovered)]


def _opcode(text: str) -> str:
    tokens = text.lower().split()
    while tokens and tokens[0] in CONTROL_PREFIXES:
        tokens.pop(0)
    return tokens[0] if tokens else ""


def _direct_address(text: str) -> int | None:
    tokens = text.split(None, 1)
    operand = tokens[1] if len(tokens) > 1 else ""
    if "[" in operand:
        return None
    match = re.search(r"(?<![\w+])0x[0-9a-fA-F]+", operand)
    return int(match.group(0), 16) if match else None


def _annotation(text: str) -> str | None:
    start = text.find("<")
    end = text.rfind(">")
    if start < 0 or end <= start:
        return None
    return text[start + 1 : end].strip()


def _internal_annotation(
    annotation: str | None, raw_symbol_to_id: Mapping[str, str]
) -> tuple[str, int] | None:
    if annotation is None:
        return None
    for raw_symbol in sorted(raw_symbol_to_id, key=len, reverse=True):
        if annotation == raw_symbol:
            return raw_symbol_to_id[raw_symbol], 0
        if annotation.startswith(raw_symbol + "+"):
            suffix = annotation[len(raw_symbol) + 1 :]
            try:
                return raw_symbol_to_id[raw_symbol], int(
                    suffix, 16 if suffix.lower().startswith("0x") else 10
                )
            except ValueError:
                continue
    return None


DART_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
CONSTRUCTOR_OWNER_RE = re.compile(
    r"^new\s+(?P<owner>[A-Za-z_$][A-Za-z0-9_$]*)"
)


def _strip_annotation_offset(annotation: str) -> str:
    return re.sub(
        r"\+(?:0x[0-9a-fA-F]+|[0-9]+)\Z", "", annotation
    ).strip()


def _attested_annotation_semantics(
    annotation: str | None, attestation: AttestedSymbols
) -> dict[str, Any] | None:
    """Classify a raw GDB annotation using only keyed name comparisons."""

    if annotation is None:
        return None
    base = _strip_annotation_offset(annotation)
    tokens = list(DART_IDENTIFIER_RE.finditer(base))
    type_hits: list[tuple[int, int, str]] = []
    for token in tokens:
        value = token.group(0)
        type_alias = attestation.match_type(value)
        if type_alias is not None:
            type_hits.append((token.start(), token.end(), type_alias))

    lowered = base.lower()
    is_type_assertion = (
        lowered.startswith("assert type is ")
        or " type is " in lowered
        or lowered.startswith("type test ")
    )
    constructor_match = CONSTRUCTOR_OWNER_RE.match(base)
    constructor_owner_alias = (
        attestation.match_type(constructor_match.group("owner"))
        if constructor_match is not None
        else None
    )
    # Only the immediate owner after `new` proves a user-owned constructor.
    # A source type used merely as an SDK generic argument (`new List<Dog>`)
    # is not ownership evidence. Explicit functions/methods/closures are
    # covered by GDB's complete same-file File section and are never recovered
    # from collision-prone labels such as `_List.map` or even bare `map`.
    is_constructor = constructor_owner_alias is not None
    user_callable = is_constructor

    replacements: list[tuple[int, int, str]] = []
    replacements.extend(
        (start, end, "@" + alias) for start, end, alias in type_hits
    )
    public_label = base
    for start, end, replacement in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        public_label = public_label[:start] + replacement + public_label[end:]
    if not replacements:
        public_label = None
    return {
        "raw_callable_symbol": base if user_callable else None,
        "user_callable": user_callable,
        "constructor": is_constructor,
        "type_assertion": is_type_assertion and bool(type_hits),
        "public_label": public_label if is_type_assertion else None,
        "public_reference_label": public_label,
        "type_aliases": sorted(
            {alias for _, _, alias in type_hits},
            key=lambda alias: int(alias[1:]),
        ),
        "has_attested_symbol": bool(type_hits),
    }


def _matches_symbol_or_offset(label: str, symbols: Iterable[str]) -> bool:
    return any(
        label == symbol
        or (
            label.startswith(symbol + "+")
            and re.fullmatch(
                r"(?:0x[0-9a-fA-F]+|[0-9]+)", label[len(symbol) + 1 :]
            )
            is not None
        )
        for symbol in symbols
    )


def _classify_external_annotation(
    annotation: str | None,
    *,
    attestation: AttestedSymbols,
    direct_control_transfer: bool,
    trusted_runtime_symbols: set[str],
    known_nonruntime_symbols: set[str],
) -> tuple[str | None, str] | None:
    """Classify an annotation without ever publishing an untrusted label."""

    if annotation is None:
        return None
    label = re.sub(r"\s+", " ", annotation).strip()
    if not label:
        return None
    lowered = label.lower()
    if (
        "file://" in lowered
        or ".dart" in lowered
        or "\n" in label
        or "\r" in label
    ):
        raise UserFunctionExtractionError(
            "source_identity_in_external_annotation"
        )
    # Strong GDB/runtime evidence wins before source-name comparisons. This
    # prevents a source method named `map` from capturing `_List.map`.
    if (
        label in RUNTIME_SYMBOL_POLICY["retain_exact"]
        or any(
            label.startswith(prefix)
            for prefix in RUNTIME_SYMBOL_POLICY["retain_prefixes"]
        )
        or _matches_symbol_or_offset(label, trusted_runtime_symbols)
    ):
        return label, "trusted_runtime"
    if (
        label in RUNTIME_SYMBOL_POLICY["neutralize_exact"]
        or any(
            label.startswith(prefix)
            for prefix in RUNTIME_SYMBOL_POLICY["neutralize_prefixes"]
        )
    ):
        return None, "neutralized_untrusted_runtime"
    semantics = _attested_annotation_semantics(label, attestation)
    if semantics and semantics["user_callable"]:
        if direct_control_transfer:
            raise UserFunctionExtractionError(
                "attested_user_function_not_disassembled"
            )
        return (
            str(semantics["public_reference_label"]),
            "trusted_runtime",
        )
    if semantics and semantics["type_assertion"]:
        return str(semantics["public_label"]), "trusted_runtime"
    # The complete keyed source-symbol attestation proves this label is not a
    # declared user function/type. Keep the distinct call identity but publish
    # no raw text. This covers SDK implicit constructors omitted from GDB's
    # File table (for example `new _Set`) without leaking private names.
    return None, "neutralized_untrusted_runtime"


def _classify_function(raw_symbol: str, root_symbol: str) -> tuple[str, str]:
    """Return an honest kind and the evidence available in GDB fallback mode."""

    lowered = raw_symbol.lower()
    if raw_symbol == root_symbol:
        return "RegularFunction", "exact_root_symbol"
    if raw_symbol.startswith("new "):
        return "Constructor", "attested_recursive_constructor"
    if (
        "implicit closure" in lowered
        or raw_symbol in {f"{root_symbol}_{root_symbol}", f"{root_symbol}.{root_symbol}"}
    ):
        return "ImplicitClosureFunction", "gdb_symbol_shape"
    if (
        "anonymous closure" in lowered
        or raw_symbol.startswith(root_symbol + ".")
        or raw_symbol.startswith(root_symbol + "<")
    ):
        return "ClosureFunction", "gdb_symbol_shape"
    if "." not in raw_symbol and "<" not in raw_symbol and ">" not in raw_symbol:
        return "RegularFunction", "top_level_gdb_symbol_shape"
    # GDB's File table does not expose the Dart VM Code::Kind.  Unknown is
    # serialized explicitly rather than fabricating a regular/closure label.
    return "UnknownFunction", "gdb_kind_unavailable"


def _function_ranges(
    parsed_functions: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, str]]:
    return [
        (
            int(function["entry_address"]),
            int(function["entry_address"]) + int(function["code_size_bytes"]),
            str(function["function_id"]),
        )
        for function in parsed_functions
    ]


def _address_target(
    address: int | None,
    ranges: Sequence[tuple[int, int, str]],
) -> tuple[str, int] | None:
    if address is None:
        return None
    matches = [
        (function_id, address - start)
        for start, end, function_id in ranges
        if start <= address < end
    ]
    if len(matches) > 1:
        raise UserFunctionExtractionError(
            f"ambiguous_internal_target_address:0x{address:x}"
        )
    return matches[0] if matches else None


def _normalize_instruction(
    *,
    raw_text: str,
    current_function_id: str,
    raw_symbol_to_id: Mapping[str, str],
    ranges: Sequence[tuple[int, int, str]],
    external_ids_by_raw_annotation: Mapping[str, str],
) -> str:
    opcode = _opcode(raw_text)
    annotation = _annotation(raw_text)
    internal = _internal_annotation(annotation, raw_symbol_to_id)
    direct = _direct_address(raw_text)
    if internal is None:
        internal = _address_target(direct, ranges)
    external_id = (
        None
        if internal is not None or annotation is None
        else external_ids_by_raw_annotation.get(annotation)
    )
    if internal is None and annotation is not None and external_id is None:
        raise UserFunctionExtractionError(
            "external_annotation_missing_policy_record"
        )

    normalized = re.sub(r"\s+", " ", raw_text).strip()
    if opcode in X86_CONDITIONAL | X86_UNCONDITIONAL | X86_CALLS:
        tokens = normalized.split(None, 1)
        prefix = tokens[0]
        if internal is not None:
            target_id, target_offset = internal
            if target_id == current_function_id and opcode not in X86_CALLS:
                target = f"@L+0x{target_offset:x}"
            else:
                target = f"@{target_id}"
                if target_offset:
                    target += f"+0x{target_offset:x}"
            return f"{prefix} {target}"
        if external_id is not None:
            return f"{prefix} @{external_id}"

    if annotation is not None:
        if internal is not None:
            target_id, target_offset = internal
            replacement = f"@{target_id}"
            if target_offset:
                replacement += f"+0x{target_offset:x}"
        elif external_id is not None:
            replacement = f"@{external_id}"
        else:
            replacement = "@X?"
        start = normalized.find("<")
        end = normalized.rfind(">")
        normalized = normalized[:start] + "<" + replacement + ">" + normalized[end + 1 :]

    # GDB puts symbolic identities in angle annotations.  Do not perform a
    # blind substring replacement here: a legal one-letter helper such as
    # ``f`` must not rewrite every ``f`` appearing in an opcode or register.
    return normalized


def _build_lossless_cfg(
    instructions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a non-pruning CFG and account for every parsed instruction."""

    if not instructions:
        raise UserFunctionExtractionError("cfg_empty_instruction_stream")
    by_address = {int(item["address"]): index for index, item in enumerate(instructions)}
    if len(by_address) != len(instructions):
        raise UserFunctionExtractionError("cfg_duplicate_instruction_address")
    leaders = {0}
    unknown_branch_mnemonics: set[str] = set()
    external_direct_branches = 0
    indirect_branches = 0
    for index, item in enumerate(instructions):
        opcode = _opcode(str(item["raw_text"]))
        target = _direct_address(str(item["raw_text"]))
        if opcode in X86_CONDITIONAL | X86_UNCONDITIONAL:
            if target in by_address:
                leaders.add(by_address[target])
            elif target is None:
                indirect_branches += 1
            else:
                external_direct_branches += 1
            if index + 1 < len(instructions):
                leaders.add(index + 1)
        elif opcode in X86_CALLS | X86_RETURNS | X86_TRAPS:
            if index + 1 < len(instructions):
                leaders.add(index + 1)
        elif opcode.startswith("j"):
            unknown_branch_mnemonics.add(opcode)
            if index + 1 < len(instructions):
                leaders.add(index + 1)

    ordered_leaders = sorted(leaders)
    blocks: list[dict[str, Any]] = []
    instruction_index_to_block: dict[int, int] = {}
    for block_id, start in enumerate(ordered_leaders):
        stop = (
            ordered_leaders[block_id + 1]
            if block_id + 1 < len(ordered_leaders)
            else len(instructions)
        )
        indices = list(range(start, stop))
        if not indices:
            raise UserFunctionExtractionError("cfg_empty_block")
        for instruction_index in indices:
            instruction_index_to_block[instruction_index] = block_id
        last_opcode = _opcode(str(instructions[indices[-1]]["raw_text"]))
        if last_opcode in X86_CONDITIONAL:
            block_type = "conditional"
        elif last_opcode in X86_UNCONDITIONAL:
            block_type = "jump"
        elif last_opcode in X86_CALLS:
            block_type = "call"
        elif last_opcode in X86_RETURNS:
            block_type = "return"
        elif last_opcode in X86_TRAPS:
            block_type = "trap"
        elif last_opcode.startswith("j"):
            block_type = "unknown_branch"
        else:
            block_type = "linear"
        blocks.append(
            {
                "id": block_id,
                "start_offset": int(instructions[start]["offset"]),
                "instruction_offsets": [
                    int(instructions[index]["offset"]) for index in indices
                ],
                "block_type": block_type,
            }
        )

    edges: list[dict[str, Any]] = []

    def add_edge(source: int, target: int, edge_type: str) -> None:
        edge = {"source": source, "target": target, "edge_type": edge_type}
        if edge not in edges:
            edges.append(edge)

    for block in blocks:
        source = int(block["id"])
        last_offset = int(block["instruction_offsets"][-1])
        last_index = next(
            index
            for index, instruction in enumerate(instructions)
            if int(instruction["offset"]) == last_offset
        )
        item = instructions[last_index]
        opcode = _opcode(str(item["raw_text"]))
        target = _direct_address(str(item["raw_text"]))
        if opcode in X86_CONDITIONAL:
            if target in by_address:
                add_edge(
                    source,
                    instruction_index_to_block[by_address[target]],
                    "conditional_true",
                )
            if last_index + 1 < len(instructions):
                add_edge(
                    source,
                    instruction_index_to_block[last_index + 1],
                    "conditional_false",
                )
        elif opcode in X86_UNCONDITIONAL:
            if target in by_address:
                add_edge(
                    source,
                    instruction_index_to_block[by_address[target]],
                    "unconditional_jump",
                )
        elif opcode in X86_RETURNS | X86_TRAPS or opcode.startswith("j"):
            pass
        elif last_index + 1 < len(instructions):
            add_edge(
                source,
                instruction_index_to_block[last_index + 1],
                "linear_fallthrough",
            )

    reachable = {0}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge["source"] in reachable and edge["target"] not in reachable:
                reachable.add(edge["target"])
                changed = True
    all_instruction_offsets = [
        int(offset)
        for block in blocks
        for offset in block["instruction_offsets"]
    ]
    expected_offsets = [int(instruction["offset"]) for instruction in instructions]
    all_edges_in_range = all(
        0 <= int(edge["source"]) < len(blocks)
        and 0 <= int(edge["target"]) < len(blocks)
        for edge in edges
    )
    integrity = {
        "valid": bool(
            all_edges_in_range
            and all_instruction_offsets == expected_offsets
            and all(block["instruction_offsets"] for block in blocks)
        ),
        "entry_block": 0,
        "entry_blocks": [0],
        "block_count": len(blocks),
        "parsed_instruction_count": len(instructions),
        "represented_instruction_count": len(all_instruction_offsets),
        "excluded_instruction_count": 0,
        "all_edges_in_range": all_edges_in_range,
        "all_blocks_nonempty": all(
            bool(block["instruction_offsets"]) for block in blocks
        ),
        "unreachable_blocks_retained": sorted(set(range(len(blocks))) - reachable),
        "pruned_unreachable_block_count": 0,
        "external_direct_branch_count": external_direct_branches,
        "indirect_branch_count": indirect_branches,
        "unknown_branch_mnemonics": sorted(unknown_branch_mnemonics),
        "control_transfers_are_accounted_not_invented": True,
    }
    if not integrity["valid"]:
        raise UserFunctionExtractionError("lossless_cfg_integrity_failed")
    return blocks, edges, integrity


def model_projection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the source-side object a multi-function codec must encode."""

    functions = []
    for function in bundle["functions"]:
        text_by_offset = {
            int(instruction["offset"]): str(instruction["text"])
            for instruction in function["instructions"]
        }
        blocks = [
            {
                "id": int(block["id"]),
                "instructions": [
                    text_by_offset[int(offset)]
                    for offset in block["instruction_offsets"]
                ],
            }
            for block in function["cfg"]
        ]
        functions.append(
            {
                "function_id": str(function["function_id"]),
                "function_kind": str(function["function_kind"]),
                "entry_blocks": [0],
                "blocks": blocks,
                "cfg_edges": list(function["cfg_edges"]),
            }
        )
    return {
        "schema": MODEL_SCHEMA,
        "architecture": "x86_64",
        "root_function_id": "F0",
        "functions": functions,
        "interfunction_transfers": list(bundle["interfunction_transfers"]),
        "external_symbols": list(bundle["external_symbols"]),
        "type_aliases": list(bundle["type_aliases"]),
        "function_attestation_aliases": list(
            bundle["function_attestation_aliases"]
        ),
        "symbol_attestation_binding_sha256": canonical_sha256(
            bundle["symbol_attestation_binding"]
        ),
        "runtime_symbol_policy_sha256": str(
            bundle["runtime_symbol_policy_sha256"]
        ),
    }


def build_user_function_bundle(
    *,
    task_id: str,
    root_symbol: str,
    private_file_identity: str,
    selected_symbols: Sequence[str],
    parsed_by_symbol: Mapping[str, Mapping[str, Any]],
    info_output_sha256: str,
    aot_sha256: str,
    aot_size_bytes: int,
    source_only_contract: Mapping[str, Any],
    symbol_attestation: AttestedSymbols,
    symbol_attestation_file_sha256: str,
    gdb_file_symbols: Sequence[str] | None = None,
    trusted_runtime_symbols: set[str] | None = None,
    known_nonruntime_symbols: set[str] | None = None,
    split: str | None = None,
    split_row: int | None = None,
) -> dict[str, Any]:
    """Neutralize and combine complete same-file GDB disassemblies."""

    if set(parsed_by_symbol) != set(selected_symbols):
        raise UserFunctionExtractionError("parsed_symbol_set_mismatch")
    if root_symbol not in parsed_by_symbol:
        raise UserFunctionExtractionError("parsed_root_missing")
    if source_only_contract.get("contract") != SCAFFOLD_CONTRACT:
        raise UserFunctionExtractionError("source_only_scaffold_contract_mismatch")
    for field in (
        "analysis_program_sha256",
        "function_source_sha256",
        "producer_script_sha256",
    ):
        if not SHA256_RE.fullmatch(
            str(source_only_contract.get(field) or "").lower()
        ):
            raise UserFunctionExtractionError(
                f"invalid_source_only_contract_{field}"
            )
    if source_only_contract.get("aot_row_schema") != SOURCE_ONLY_AOT_ROW_SCHEMA:
        raise UserFunctionExtractionError("source_only_aot_row_schema_mismatch")
    if not isinstance(symbol_attestation, AttestedSymbols):
        raise UserFunctionExtractionError("symbol_attestation_required")
    if symbol_attestation.task_id != task_id:
        raise UserFunctionExtractionError("symbol_attestation_task_mismatch")
    if not SHA256_RE.fullmatch(
        str(symbol_attestation_file_sha256 or "").lower()
    ):
        raise UserFunctionExtractionError(
            "symbol_attestation_file_sha256_invalid"
        )
    symbol_attestation.verify_source_contract(source_only_contract)
    gdb_file_symbols = list(gdb_file_symbols or selected_symbols)
    if not set(gdb_file_symbols).issubset(set(selected_symbols)):
        raise UserFunctionExtractionError(
            "gdb_file_symbols_not_in_disassembly_set"
        )

    # `source_only_program` removes every gold top-level main and appends this
    # exact producer-owned compilation scaffold:
    #
    #     void main() {}
    #
    # The AOT manifest is accepted only when it is bound to the corresponding
    # source-only producer contract.  Therefore exact `main`, if GDB exposes
    # it, is not user behavior.  It is still disassembled and accounted before
    # being excluded from the model projection.  No prefixed/nested symbol is
    # excluded by this rule.
    scaffold_symbols = [symbol for symbol in gdb_file_symbols if symbol == "main"]
    if len(scaffold_symbols) > 1:
        raise UserFunctionExtractionError("duplicate_producer_main_scaffold")
    user_symbols = [symbol for symbol in selected_symbols if symbol != "main"]
    if root_symbol not in user_symbols:
        raise UserFunctionExtractionError("root_was_scaffold")
    ordered_raw = [root_symbol] + sorted(
        (symbol for symbol in user_symbols if symbol != root_symbol),
        key=lambda symbol: (
            int(parsed_by_symbol[symbol]["entry_address"]),
            symbol,
        ),
    )
    raw_symbol_to_id = {
        raw_symbol: f"F{index}" for index, raw_symbol in enumerate(ordered_raw)
    }
    internal_functions: list[dict[str, Any]] = []
    for raw_symbol in ordered_raw:
        parsed = dict(parsed_by_symbol[raw_symbol])
        parsed["function_id"] = raw_symbol_to_id[raw_symbol]
        internal_functions.append(parsed)
    ranges = _function_ranges(internal_functions)

    trusted_runtime_symbols = set(trusted_runtime_symbols or ())
    known_nonruntime_symbols = set(known_nonruntime_symbols or ())
    external_ids: dict[str, str] = {}
    external_records: list[dict[str, Any]] = []
    for function in internal_functions:
        for instruction in function["instructions"]:
            raw_text = str(instruction["raw_text"])
            annotation = _annotation(raw_text)
            internal = _internal_annotation(
                annotation, raw_symbol_to_id
            )
            if internal is None:
                internal = _address_target(
                    _direct_address(raw_text), ranges
                )
            if internal is not None:
                continue
            classified = _classify_external_annotation(
                annotation,
                attestation=symbol_attestation,
                direct_control_transfer=(
                    _opcode(raw_text)
                    in X86_CALLS | X86_UNCONDITIONAL
                ),
                trusted_runtime_symbols=trusted_runtime_symbols,
                known_nonruntime_symbols=known_nonruntime_symbols,
            )
            if classified is not None and annotation not in external_ids:
                public_label, symbol_class = classified
                external_id = f"X{len(external_ids)}"
                external_ids[annotation] = external_id
                external_records.append(
                    {
                        "external_id": external_id,
                        "symbol": public_label,
                        "symbol_class": symbol_class,
                    }
                )

    public_functions: list[dict[str, Any]] = []
    interfunction_transfers: list[dict[str, Any]] = []
    call_site_count = 0
    represented_call_site_count = 0
    direct_internal_calls = 0
    direct_external_calls = 0
    indirect_calls = 0
    for function in internal_functions:
        raw_symbol = str(function["raw_symbol"])
        function_id = str(function["function_id"])
        blocks, cfg_edges, integrity = _build_lossless_cfg(function["instructions"])
        public_instructions: list[dict[str, Any]] = []
        for instruction in function["instructions"]:
            raw_text = str(instruction["raw_text"])
            text = _normalize_instruction(
                raw_text=raw_text,
                current_function_id=function_id,
                raw_symbol_to_id=raw_symbol_to_id,
                ranges=ranges,
                external_ids_by_raw_annotation=external_ids,
            )
            public_instructions.append(
                {
                    "offset": int(instruction["offset"]),
                    "machine_bytes": str(instruction["machine_bytes"]),
                    "text": text,
                }
            )
            opcode = _opcode(raw_text)
            if opcode not in X86_CALLS:
                continue
            call_site_count += 1
            annotation = _annotation(raw_text)
            internal = _internal_annotation(annotation, raw_symbol_to_id)
            direct = _direct_address(raw_text)
            if internal is None:
                internal = _address_target(direct, ranges)
            if internal is not None:
                target_id, target_offset = internal
                transfer = {
                    "caller_function_id": function_id,
                    "caller_offset": int(instruction["offset"]),
                    "transfer_kind": "direct_internal_call",
                    "target_function_id": target_id,
                    "target_offset": int(target_offset),
                }
                direct_internal_calls += 1
            else:
                external_id = (
                    external_ids.get(annotation)
                    if annotation is not None
                    else None
                )
                if external_id is not None:
                    transfer = {
                        "caller_function_id": function_id,
                        "caller_offset": int(instruction["offset"]),
                        "transfer_kind": "direct_external_call",
                        "external_id": external_id,
                    }
                    direct_external_calls += 1
                else:
                    transfer = {
                        "caller_function_id": function_id,
                        "caller_offset": int(instruction["offset"]),
                        "transfer_kind": (
                            "indirect_call"
                            if direct is None
                            else "unlabelled_direct_external_call"
                        ),
                        "operand": text.split(None, 1)[1] if " " in text else "",
                    }
                    if direct is None:
                        indirect_calls += 1
                    else:
                        direct_external_calls += 1
            interfunction_transfers.append(transfer)
            represented_call_site_count += 1

        recovery_kind = function.get("attested_recovery_kind")
        if recovery_kind == "constructor":
            function_kind = "Constructor"
            kind_evidence = "keyed_attestation_and_exact_call_target_address"
        else:
            function_kind, kind_evidence = _classify_function(
                raw_symbol, root_symbol
            )
        public_functions.append(
            {
                "function_id": function_id,
                "function_kind": function_kind,
                "function_kind_evidence": kind_evidence,
                "code_size_bytes": int(function["code_size_bytes"]),
                "instructions": public_instructions,
                "cfg": blocks,
                "cfg_edges": cfg_edges,
                "integrity": integrity,
            }
        )

    external_symbols = external_records
    raw_instruction_count = sum(
        len(function["instructions"]) for function in internal_functions
    )
    scaffold_instruction_count = sum(
        len(parsed_by_symbol[symbol]["instructions"])
        for symbol in scaffold_symbols
    )
    scaffold_call_site_count = sum(
        _opcode(str(instruction["raw_text"])) in X86_CALLS
        for symbol in scaffold_symbols
        for instruction in parsed_by_symbol[symbol]["instructions"]
    )
    emitted_instruction_count = sum(
        len(function["instructions"]) for function in public_functions
    )
    bundle: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": task_id,
        "split": split,
        "split_row": split_row,
        "source_text_read": False,
        "raw_source_names_serialized": False,
        "raw_source_paths_serialized": False,
        "source_symbol_attestation_used": True,
        "source_symbol_attestation_is_keyed": True,
        "architecture": "x86_64",
        "root_function_id": "F0",
        "functions": public_functions,
        "interfunction_transfers": interfunction_transfers,
        "external_symbols": external_symbols,
        "type_aliases": [
            {"type_alias": alias}
            for alias in symbol_attestation.public_type_aliases
        ],
        "function_attestation_aliases": [
            {"function_attestation_alias": alias}
            for alias in symbol_attestation.public_function_aliases
        ],
        "symbol_attestation_binding": symbol_attestation.public_binding(
            str(symbol_attestation_file_sha256).lower()
        ),
        "runtime_symbol_policy": RUNTIME_SYMBOL_POLICY,
        "runtime_symbol_policy_sha256": canonical_sha256(
            RUNTIME_SYMBOL_POLICY
        ),
        "accounting": {
            "selected_function_count": len(selected_symbols),
            "gdb_file_function_count": len(gdb_file_symbols),
            "attested_recursive_function_count": len(
                set(selected_symbols) - set(gdb_file_symbols)
            ),
            "successfully_disassembled_function_count": len(parsed_by_symbol),
            "producer_scaffold_function_count": len(scaffold_symbols),
            "producer_scaffold_instruction_count": scaffold_instruction_count,
            "producer_scaffold_call_site_count": scaffold_call_site_count,
            "user_function_count": len(user_symbols),
            "emitted_function_count": len(public_functions),
            "excluded_user_function_count": 0,
            "raw_user_instruction_count": raw_instruction_count,
            "emitted_instruction_count": emitted_instruction_count,
            "excluded_user_instruction_count": 0,
            "call_site_count": call_site_count,
            "represented_call_site_count": represented_call_site_count,
            "excluded_user_call_site_count": 0,
            "direct_internal_call_count": direct_internal_calls,
            "direct_external_call_count": direct_external_calls,
            "indirect_call_count": indirect_calls,
            "unknown_function_kind_count": sum(
                function["function_kind"] == "UnknownFunction"
                for function in public_functions
            ),
            "trusted_runtime_external_symbol_count": sum(
                record["symbol_class"] == "trusted_runtime"
                for record in external_symbols
            ),
            "neutralized_external_symbol_count": sum(
                record["symbol_class"]
                in {
                    "neutralized_untrusted_runtime",
                }
                for record in external_symbols
            ),
            "attested_type_assertion_count": sum(
                record["symbol_class"] == "trusted_runtime"
                and isinstance(record.get("symbol"), str)
                and "@T" in record["symbol"]
                and "type is" in record["symbol"]
                for record in external_symbols
            ),
        },
        "lossless_contract": {
            "domain": MODEL_SCHEMA,
            "all_same_gdb_file_functions_disassembled": True,
            "all_attested_user_constructor_targets_disassembled": True,
            "recursive_disassembly_uses_direct_operand_address": True,
            "recursive_entry_must_equal_direct_operand_address": True,
            "type_assertion_names_replaced_by_attested_aliases": True,
            "indirect_calls_preserved_as_unresolved_dynamic_dispatch": True,
            "producer_owned_empty_main_scaffold_excluded": True,
            "all_user_functions_required": True,
            "all_user_machine_instructions_required": True,
            "all_user_call_sites_required": True,
            "unreachable_blocks_retained": True,
            "truncation_allowed": False,
            "student_token_limit": DEFAULT_STUDENT_BUDGET,
            "api_token_limit": DEFAULT_API_BUDGET,
            "budget_measurements_bind_model_projection_sha256": True,
            "runtime_symbol_policy_sha256": canonical_sha256(
                RUNTIME_SYMBOL_POLICY
            ),
        },
        "inputs": {
            "aot_sha256": aot_sha256,
            "aot_size_bytes": int(aot_size_bytes),
            "gdb_info_output_sha256": info_output_sha256,
            "gdb_file_section_sha256": sha256_bytes(
                private_file_identity.encode("utf-8")
            ),
            "raw_disassembly_sha256": canonical_sha256(
                {
                    raw_symbol_to_id[symbol]: parsed_by_symbol[symbol]["raw_sha256"]
                    for symbol in ordered_raw
                }
                | {
                    f"SCAFFOLD{index}": parsed_by_symbol[symbol]["raw_sha256"]
                    for index, symbol in enumerate(scaffold_symbols)
                }
            ),
        },
        "source_only_producer_contract": {
            "contract": SCAFFOLD_CONTRACT,
            "build_input_schema": SOURCE_ONLY_BUILD_INPUT_SCHEMA,
            "aot_row_schema": SOURCE_ONLY_AOT_ROW_SCHEMA,
            "analysis_program_sha256": str(
                source_only_contract["analysis_program_sha256"]
            ).lower(),
            "function_source_sha256": str(
                source_only_contract["function_source_sha256"]
            ).lower(),
            "producer_script_sha256": str(
                source_only_contract["producer_script_sha256"]
            ).lower(),
            "excluded_scaffold_symbol": "main" if scaffold_symbols else None,
        },
    }
    projection = model_projection(bundle)
    bundle["model_projection_sha256"] = canonical_sha256(projection)

    serialized = canonical_bytes(bundle).decode("ascii")
    if "file://" in serialized.lower() or ".dart" in serialized.lower():
        raise UserFunctionExtractionError("source_path_leaked_into_public_bundle")
    for raw_symbol in selected_symbols:
        # Check symbol-bearing contexts, not arbitrary substrings such as the
        # common root name "main" inside an unrelated SDK identifier.
        residue_patterns = (
            f"<{raw_symbol}",
            f">{raw_symbol}<",
            f'"raw_symbol":"{raw_symbol}"',
        )
        if any(pattern in serialized for pattern in residue_patterns):
            raise UserFunctionExtractionError(
                f"raw_user_symbol_leaked:{raw_symbol}"
            )
    accounting = bundle["accounting"]
    if (
        accounting["selected_function_count"]
        != accounting["successfully_disassembled_function_count"]
        or accounting["user_function_count"]
        != accounting["emitted_function_count"]
        or accounting["gdb_file_function_count"]
        + accounting["attested_recursive_function_count"]
        != accounting["successfully_disassembled_function_count"]
        or accounting["gdb_file_function_count"]
        + accounting["attested_recursive_function_count"]
        != accounting["user_function_count"]
        + accounting["producer_scaffold_function_count"]
        or accounting["raw_user_instruction_count"]
        != accounting["emitted_instruction_count"]
        or accounting["call_site_count"]
        != accounting["represented_call_site_count"]
        or accounting["excluded_user_function_count"]
        or accounting["excluded_user_instruction_count"]
        or accounting["excluded_user_call_site_count"]
    ):
        raise UserFunctionExtractionError(
            "zero_user_exclusion_accounting_failed"
        )
    return bundle


def _gdb_quote(value: str) -> str:
    if any(character in value for character in "\r\n"):
        raise UserFunctionExtractionError("newline_in_gdb_argument")
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _run(
    command: Sequence[str], *, timeout: float, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise UserFunctionExtractionError(
            f"command_failed_to_run:{command[0]}:{error}"
        ) from error
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout or "")[:1000]
        raise UserFunctionExtractionError(
            f"command_failed:{command[0]}:{result.returncode}:{diagnostic}"
        )
    return result


def extract_aot(
    *,
    task_id: str,
    aot_path: Path,
    gdb: str,
    root_symbol: str,
    timeout: float,
    split: str | None,
    split_row: int | None,
    source_only_contract: Mapping[str, Any],
    symbol_attestation: AttestedSymbols,
    symbol_attestation_file_sha256: str,
) -> dict[str, Any]:
    """Run the two GDB passes for one existing AOT and return a public bundle."""

    if not aot_path.is_file():
        raise UserFunctionExtractionError(f"aot_missing:{aot_path}")
    aot_hash = sha256_file(aot_path)
    common = [
        gdb,
        "-q",
        "-nx",
        "-batch",
        "-ex",
        "set pagination off",
        "-ex",
        "set width 0",
        "-ex",
        "set disassembly-flavor intel",
        "-ex",
        f"file {_gdb_quote(str(aot_path.resolve()))}",
    ]
    info_result = _run(
        [*common, "-ex", "info functions"], timeout=timeout, cwd=aot_path.parent
    )
    private_file_identity, symbols = select_same_file_symbols(
        info_result.stdout, root_symbol
    )
    trusted_runtime_symbols, known_nonruntime_symbols = (
        external_symbol_evidence(info_result.stdout, private_file_identity)
    )
    def disassemble_batch(batch: Sequence[str]) -> dict[str, dict[str, Any]]:
        disassembly_commands = list(common)
        for symbol in batch:
            disassembly_commands.extend(
                ["-ex", f"disassemble /r {_gdb_quote(symbol)}"]
            )
        disassembly_result = _run(
            disassembly_commands, timeout=timeout, cwd=aot_path.parent
        )
        return parse_combined_gdb_disassemblies(
            disassembly_result.stdout, batch
        )

    def disassemble_address_batch(
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        disassembly_commands = list(common)
        for target in targets:
            disassembly_commands.extend(
                [
                    "-ex",
                    f"disassemble /r 0x{int(target['address']):x}",
                ]
            )
        disassembly_result = _run(
            disassembly_commands, timeout=timeout, cwd=aot_path.parent
        )
        return parse_combined_gdb_disassemblies_by_address(
            disassembly_result.stdout, targets
        )

    gdb_file_symbols = list(symbols)
    parsed = disassemble_batch(gdb_file_symbols)
    selected_symbols = list(gdb_file_symbols)
    for _round in range(128):
        newly_discovered_targets = discover_attested_direct_callees(
            parsed, symbol_attestation
        )
        if not newly_discovered_targets:
            break
        if len(parsed) + len(newly_discovered_targets) > 4096:
            raise UserFunctionExtractionError(
                "attested_recursive_function_limit_exceeded"
            )
        new_parsed = disassemble_address_batch(
            newly_discovered_targets
        )
        if set(new_parsed) & set(parsed):
            raise UserFunctionExtractionError(
                "attested_recursive_identity_collision"
            )
        parsed.update(new_parsed)
        selected_symbols.extend(new_parsed)
    else:
        raise UserFunctionExtractionError(
            "attested_recursive_disassembly_did_not_converge"
        )
    return build_user_function_bundle(
        task_id=task_id,
        root_symbol=root_symbol,
        private_file_identity=private_file_identity,
        selected_symbols=selected_symbols,
        parsed_by_symbol=parsed,
        info_output_sha256=sha256_bytes(info_result.stdout.encode("utf-8")),
        aot_sha256=aot_hash,
        aot_size_bytes=aot_path.stat().st_size,
        source_only_contract=source_only_contract,
        symbol_attestation=symbol_attestation,
        symbol_attestation_file_sha256=symbol_attestation_file_sha256,
        gdb_file_symbols=gdb_file_symbols,
        trusted_runtime_symbols=trusted_runtime_symbols,
        known_nonruntime_symbols=known_nonruntime_symbols,
        split=split,
        split_row=split_row,
    )


def _manifest_aot_path(row: Mapping[str, Any], aot_root: Path) -> Path:
    raw = str(row.get("aot_path") or "").strip()
    if not raw:
        raise UserFunctionExtractionError("manifest_missing_aot_path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = aot_root / candidate
    return candidate.resolve()


def _validate_manifest(
    rows: Sequence[Mapping[str, Any]], aot_root: Path
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    split_positions: set[tuple[str, int]] = set()
    for position, row in enumerate(rows):
        if row.get("schema") != SOURCE_ONLY_AOT_ROW_SCHEMA:
            raise UserFunctionExtractionError(
                f"aot_manifest_row_schema_mismatch:{position}"
            )
        task_id = str(row.get("task_id") or "").strip()
        if not TASK_ID_RE.fullmatch(task_id):
            raise UserFunctionExtractionError(
                f"invalid_task_id:{position}:{task_id!r}"
            )
        if task_id in task_ids:
            raise UserFunctionExtractionError(f"duplicate_task_id:{task_id}")
        task_ids.add(task_id)
        split = str(row.get("split") or "")
        split_row = int(row.get("split_row", position))
        key = (split, split_row)
        if key in split_positions:
            raise UserFunctionExtractionError(
                f"duplicate_split_position:{split}:{split_row}"
            )
        split_positions.add(key)
        aot_path = _manifest_aot_path(row, aot_root)
        if not aot_path.is_file():
            raise UserFunctionExtractionError(f"aot_missing:{aot_path}")
        expected_sha = str(row.get("aot_sha256") or "").lower()
        if expected_sha:
            if not SHA256_RE.fullmatch(expected_sha):
                raise UserFunctionExtractionError(
                    f"invalid_manifest_aot_sha256:{task_id}"
                )
            actual_sha = sha256_file(aot_path)
            if actual_sha != expected_sha:
                raise UserFunctionExtractionError(
                    f"manifest_aot_sha256_mismatch:{task_id}"
                )
        expected_size = row.get("aot_size_bytes")
        if expected_size is not None and aot_path.stat().st_size != int(expected_size):
            raise UserFunctionExtractionError(
                f"manifest_aot_size_mismatch:{task_id}"
            )
        analysis_program_sha256 = str(
            row.get("analysis_program_sha256") or ""
        ).lower()
        function_source_sha256 = str(
            row.get("function_source_sha256") or ""
        ).lower()
        producer = row.get("producer")
        producer_script_sha256 = (
            str(producer.get("script_sha256") or "").lower()
            if isinstance(producer, Mapping)
            else ""
        )
        for label, digest in (
            ("analysis_program", analysis_program_sha256),
            ("function_source", function_source_sha256),
            ("producer_script", producer_script_sha256),
        ):
            if not SHA256_RE.fullmatch(digest):
                raise UserFunctionExtractionError(
                    f"invalid_manifest_{label}_sha256:{task_id}"
                )
        validated.append(
            {
                "task_id": task_id,
                "split": split,
                "split_row": split_row,
                "aot_path": aot_path,
                "aot_sha256": expected_sha or sha256_file(aot_path),
                "source_only_contract": {
                    "contract": SCAFFOLD_CONTRACT,
                    "aot_row_schema": SOURCE_ONLY_AOT_ROW_SCHEMA,
                    "analysis_program_sha256": analysis_program_sha256,
                    "function_source_sha256": function_source_sha256,
                    "producer_script_sha256": producer_script_sha256,
                },
            }
        )
    return validated


def _load_measurements(
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    measurements: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("schema") != MEASUREMENT_SCHEMA:
            raise UserFunctionExtractionError("measurement_schema_mismatch")
        task_id = str(row.get("task_id") or "")
        if task_id in measurements:
            raise UserFunctionExtractionError(
                f"duplicate_encoded_measurement:{task_id}"
            )
        projection_sha = str(row.get("model_projection_sha256") or "").lower()
        if not SHA256_RE.fullmatch(projection_sha):
            raise UserFunctionExtractionError(
                f"invalid_measurement_projection_sha256:{task_id}"
            )
        student_tokens = int(row.get("student_tokens", -1))
        api_tokens = int(row.get("api_tokens", -1))
        if student_tokens < 0 or api_tokens < 0:
            raise UserFunctionExtractionError(
                f"negative_encoded_measurement:{task_id}"
            )
        measurements[task_id] = {
            "model_projection_sha256": projection_sha,
            "student_tokens": student_tokens,
            "api_tokens": api_tokens,
        }
    return measurements


def _load_symbol_attestations(
    path: Path,
    key: bytes,
    work: Sequence[Mapping[str, Any]],
    *,
    allow_unselected_rows: bool,
) -> dict[str, AttestedSymbols]:
    """Load a hash-pinned private attestation and bind it to manifest work."""

    attestations: dict[str, AttestedSymbols] = {}
    for position, row in enumerate(read_jsonl(path)):
        attestation = AttestedSymbols(row, key)
        if attestation.task_id in attestations:
            raise UserFunctionExtractionError(
                f"duplicate_symbol_attestation_task:"
                f"{attestation.task_id}:{position}"
            )
        attestations[attestation.task_id] = attestation

    expected = {str(item["task_id"]) for item in work}
    actual = set(attestations)
    missing = sorted(expected - actual)
    if missing:
        raise UserFunctionExtractionError(
            f"symbol_attestation_tasks_missing:{missing[:10]}"
        )
    extra = sorted(actual - expected)
    if extra and not allow_unselected_rows:
        raise UserFunctionExtractionError(
            f"symbol_attestation_tasks_not_in_manifest:{extra[:10]}"
        )

    for item in work:
        task_id = str(item["task_id"])
        attestation = attestations[task_id]
        if (
            str(attestation.row.get("split") or "") != str(item["split"])
            or int(attestation.row.get("split_row", -1))
            != int(item["split_row"])
        ):
            raise UserFunctionExtractionError(
                f"symbol_attestation_split_mismatch:{task_id}"
            )
        attestation.verify_source_contract(item["source_only_contract"])
    return {task_id: attestations[task_id] for task_id in expected}


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False
    )
    parser.add_argument("--aot-manifest", type=Path, required=True)
    parser.add_argument(
        "--aot-manifest-sha256",
        required=True,
        help="Expected lowercase SHA-256 of the exact selected/full manifest file.",
    )
    parser.add_argument(
        "--aot-root",
        type=Path,
        required=True,
        help="Base directory for relative aot_path values in the manifest.",
    )
    parser.add_argument(
        "--symbol-attestation",
        type=Path,
        required=True,
        help=(
            "Private keyed, name-free attestation JSONL produced by "
            "build_dart_user_symbol_attestation.py."
        ),
    )
    parser.add_argument(
        "--symbol-attestation-sha256",
        required=True,
        help="Expected lowercase SHA-256 of the exact attestation JSONL.",
    )
    parser.add_argument(
        "--symbol-attestation-key-file",
        type=Path,
        required=True,
        help=(
            "Private >=32-byte HMAC key used to build the symbol "
            "attestation. The key is never serialized."
        ),
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path, required=True)
    parser.add_argument("--gdb", default="/usr/bin/gdb")
    parser.add_argument("--root-symbol", default=DEFAULT_ROOT_SYMBOL)
    parser.add_argument(
        "--task-id",
        help=(
            "Extract one exact task from a larger sealed manifest (smoke test)."
        ),
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        required=True,
        help="Fail unless the selected manifest scope has exactly this many rows.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--encoded-measurements",
        type=Path,
        help=(
            "Optional post-codebook token measurements bound to each "
            "model_projection_sha256."
        ),
    )
    parser.add_argument(
        "--require-budget-measurements",
        action="store_true",
        help="Fail unless every row has a hash-bound 9K/12K measurement.",
    )
    parser.add_argument(
        "--student-token-budget", type=int, default=DEFAULT_STUDENT_BUDGET
    )
    parser.add_argument(
        "--api-token-budget", type=int, default=DEFAULT_API_BUDGET
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.workers <= 0:
        raise UserFunctionExtractionError("workers_must_be_positive")
    if args.timeout <= 0:
        raise UserFunctionExtractionError("timeout_must_be_positive")
    if args.student_token_budget != DEFAULT_STUDENT_BUDGET:
        raise UserFunctionExtractionError("student_budget_must_equal_9000")
    if args.api_token_budget != DEFAULT_API_BUDGET:
        raise UserFunctionExtractionError("api_budget_must_equal_12000")
    if args.require_budget_measurements and args.encoded_measurements is None:
        raise UserFunctionExtractionError(
            "required_encoded_measurements_missing"
        )

    aot_manifest = args.aot_manifest.resolve()
    expected_manifest_sha = str(args.aot_manifest_sha256).lower()
    if not SHA256_RE.fullmatch(expected_manifest_sha):
        raise UserFunctionExtractionError("invalid_aot_manifest_sha256")
    observed_manifest_sha = sha256_file(aot_manifest)
    if observed_manifest_sha != expected_manifest_sha:
        raise UserFunctionExtractionError(
            f"aot_manifest_sha256_mismatch:{observed_manifest_sha}"
            f"!={expected_manifest_sha}"
        )
    manifest_rows = read_jsonl(aot_manifest)
    if args.task_id is not None:
        manifest_rows = [
            row
            for row in manifest_rows
            if str(row.get("task_id") or "") == args.task_id
        ]
    if args.expected_rows <= 0:
        raise UserFunctionExtractionError("expected_rows_must_be_positive")
    if len(manifest_rows) != args.expected_rows:
        raise UserFunctionExtractionError(
            f"manifest_scope_row_count:{len(manifest_rows)}"
            f"!={args.expected_rows}"
        )
    work = _validate_manifest(manifest_rows, args.aot_root.resolve())
    expected_attestation_sha = str(
        args.symbol_attestation_sha256
    ).lower()
    if not SHA256_RE.fullmatch(expected_attestation_sha):
        raise UserFunctionExtractionError(
            "invalid_symbol_attestation_sha256"
        )
    symbol_attestation_path = args.symbol_attestation.resolve()
    observed_attestation_sha = sha256_file(symbol_attestation_path)
    if observed_attestation_sha != expected_attestation_sha:
        raise UserFunctionExtractionError(
            f"symbol_attestation_sha256_mismatch:"
            f"{observed_attestation_sha}!={expected_attestation_sha}"
        )
    attestation_key = load_attestation_key(
        args.symbol_attestation_key_file.resolve()
    )
    attestations = _load_symbol_attestations(
        symbol_attestation_path,
        attestation_key,
        work,
        allow_unselected_rows=args.task_id is not None,
    )
    attestation_key_id = attestation_key_id_sha256(attestation_key)
    measurements = _load_measurements(
        args.encoded_measurements.resolve()
        if args.encoded_measurements is not None
        else None
    )
    manifest_task_ids = {row["task_id"] for row in work}
    extra_measurements = sorted(set(measurements) - manifest_task_ids)
    if extra_measurements:
        raise UserFunctionExtractionError(
            f"measurement_tasks_not_in_manifest:{extra_measurements[:10]}"
        )

    script_sha = sha256_file(Path(__file__))
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(work)
    failures: list[dict[str, Any]] = []

    def build(position: int, item: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        receipt_name = (
            f"{position:06d}_"
            f"{hashlib.sha256(item['task_id'].encode()).hexdigest()[:16]}.json"
        )
        receipt = args.receipt_dir / receipt_name
        if receipt.is_file():
            existing = json.loads(receipt.read_text(encoding="utf-8"))
            expected_binding = attestations[
                str(item["task_id"])
            ].public_binding(observed_attestation_sha)
            if (
                existing.get("schema") == SCHEMA
                and existing.get("task_id") == item["task_id"]
                and existing.get("inputs", {}).get("aot_sha256")
                == item["aot_sha256"]
                and existing.get("producer", {}).get("script_sha256")
                == script_sha
                and existing.get("symbol_attestation_binding")
                == expected_binding
            ):
                return position, existing
            raise UserFunctionExtractionError(
                f"stale_existing_receipt:{receipt}"
            )
        bundle = extract_aot(
            task_id=str(item["task_id"]),
            aot_path=Path(item["aot_path"]),
            gdb=str(args.gdb),
            root_symbol=str(args.root_symbol),
            timeout=float(args.timeout),
            split=str(item["split"]),
            split_row=int(item["split_row"]),
            source_only_contract=item["source_only_contract"],
            symbol_attestation=attestations[str(item["task_id"])],
            symbol_attestation_file_sha256=observed_attestation_sha,
        )
        bundle["producer"] = {
            "script_sha256": script_sha,
            "gdb": str(args.gdb),
        }
        measurement = measurements.get(str(item["task_id"]))
        if measurement is not None:
            if (
                measurement["model_projection_sha256"]
                != bundle["model_projection_sha256"]
            ):
                raise UserFunctionExtractionError(
                    f"measurement_projection_mismatch:{item['task_id']}"
                )
            bundle["encoded_budget_measurement"] = measurement
        elif args.require_budget_measurements:
            raise UserFunctionExtractionError(
                f"missing_encoded_measurement:{item['task_id']}"
            )
        write_json_atomic(receipt, bundle)
        return position, bundle

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(build, position, item): (position, item)
            for position, item in enumerate(work)
        }
        for future in concurrent.futures.as_completed(futures):
            position, item = futures[future]
            try:
                result_position, bundle = future.result()
                results[result_position] = bundle
            except Exception as error:
                failures.append(
                    {
                        "position": position,
                        "task_id": item["task_id"],
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )

    complete = [result for result in results if result is not None]
    budget_rows = [
        result["encoded_budget_measurement"]
        for result in complete
        if "encoded_budget_measurement" in result
    ]
    budget_failures = [
        {
            "task_id": result["task_id"],
            "student_tokens": result["encoded_budget_measurement"]["student_tokens"],
            "api_tokens": result["encoded_budget_measurement"]["api_tokens"],
        }
        for result in complete
        if "encoded_budget_measurement" in result
        and (
            result["encoded_budget_measurement"]["student_tokens"]
            > args.student_token_budget
            or result["encoded_budget_measurement"]["api_tokens"]
            > args.api_token_budget
        )
    ]
    for failure in budget_failures:
        failures.append(
            {
                "position": None,
                "task_id": failure["task_id"],
                "reason": "encoded_token_budget_overflow",
                **failure,
            }
        )

    function_counts = [
        int(result["accounting"]["emitted_function_count"])
        for result in complete
    ]
    instruction_counts = [
        int(result["accounting"]["emitted_instruction_count"])
        for result in complete
    ]
    student_lengths = [
        int(row["student_tokens"]) for row in budget_rows
    ]
    api_lengths = [int(row["api_tokens"]) for row in budget_rows]
    all_rows_measured = len(budget_rows) == len(work)
    passed = (
        not failures
        and len(complete) == len(work)
        and (
            all_rows_measured
            if args.require_budget_measurements
            else True
        )
    )
    report = {
        "schema": REPORT_SCHEMA,
        "manifest_rows": len(work),
        "aot_manifest_sha256": observed_manifest_sha,
        "symbol_attestation": {
            "schema": SYMBOL_ATTESTATION_SCHEMA,
            "file_sha256": observed_attestation_sha,
            "key_id_sha256": attestation_key_id,
            "selected_rows": len(attestations),
            "key_serialized": False,
            "raw_names_serialized": False,
        },
        "extracted_rows": len(complete),
        "failed_rows": len(failures),
        "failure_reasons": dict(
            Counter(failure["reason"].split(":", 1)[0] for failure in failures)
        ),
        "zero_exclusion": {
            "functions": all(
                result["accounting"]["excluded_user_function_count"] == 0
                for result in complete
            ),
            "instructions": all(
                result["accounting"]["excluded_user_instruction_count"] == 0
                for result in complete
            ),
            "call_sites": all(
                result["accounting"]["excluded_user_call_site_count"] == 0
                for result in complete
            ),
            "producer_scaffold_functions": sum(
                result["accounting"]["producer_scaffold_function_count"]
                for result in complete
            ),
            "truncated_rows": 0,
        },
        "attested_recovery": {
            "recursive_functions": sum(
                result["accounting"][
                    "attested_recursive_function_count"
                ]
                for result in complete
            ),
            "type_assertions": sum(
                result["accounting"]["attested_type_assertion_count"]
                for result in complete
            ),
            "direct_operand_address_authoritative": True,
            "returned_entry_address_verified": True,
        },
        "functions": {
            "total": sum(function_counts),
            "min": min(function_counts) if function_counts else 0,
            "p50": _percentile(function_counts, 0.50),
            "p95": _percentile(function_counts, 0.95),
            "max": max(function_counts) if function_counts else 0,
        },
        "instructions": {
            "total": sum(instruction_counts),
            "min": min(instruction_counts) if instruction_counts else 0,
            "p50": _percentile(instruction_counts, 0.50),
            "p95": _percentile(instruction_counts, 0.95),
            "max": max(instruction_counts) if instruction_counts else 0,
        },
        "encoded_budget_gate": {
            "required": bool(args.require_budget_measurements),
            "measured_rows": len(budget_rows),
            "all_rows_measured": all_rows_measured,
            "student_limit": args.student_token_budget,
            "api_limit": args.api_token_budget,
            "overflow_rows": len(budget_failures),
            "student_tokens": {
                "p50": _percentile(student_lengths, 0.50),
                "p95": _percentile(student_lengths, 0.95),
                "max": max(student_lengths) if student_lengths else None,
            },
            "api_tokens": {
                "p50": _percentile(api_lengths, 0.50),
                "p95": _percentile(api_lengths, 0.95),
                "max": max(api_lengths) if api_lengths else None,
            },
        },
        "production_ready": bool(
            passed and all_rows_measured and not budget_failures
        ),
        "passed": passed,
    }
    write_jsonl_atomic(args.failures_jsonl, failures)
    write_json_atomic(args.report, report)
    if failures or len(complete) != len(work):
        # Do not publish a partial corpus under the requested final path.
        return 1
    write_jsonl_atomic(
        args.output_jsonl, (result for result in results if result is not None)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
