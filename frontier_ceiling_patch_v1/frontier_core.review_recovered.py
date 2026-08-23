#!/usr/bin/env python3
"""Fail-closed primitives for the audited frontier pass@k runner.

This module deliberately contains no API client and no benchmark harness import.
That keeps artifact decoding, prompt construction, response validation, and
provenance logic independently testable.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from frontier_f2 import (
        F2_SCHEMA,
        F2_SYSTEM_PROMPT,
        F2CodecError,
        decode_f2,
        serialize_f2,
        visible_one_token_symbols,
    )
except ImportError:  # Package import, used by frontier_ceiling_patch_v1.__init__.
    from .frontier_f2 import (
        F2_SCHEMA,
        F2_SYSTEM_PROMPT,
        F2CodecError,
        decode_f2,
        serialize_f2,
        visible_one_token_symbols,
    )

SCHEMA_VERSION = "audited-frontier-passk-v1"
RAW_SCHEMA_VERSION = "complete-fn0-family-gdb-v1"
GRAPH_MARKER = "<G2C1>"
EDGE_LABELS = {
    "conditional_true": "conditional_true",
    "conditional_false": "conditional_false",
    "linear_fallthrough": "linear_fallthrough",
    "loop_backedge": "loop_backedge",
    "unconditional": "unconditional",
    "unconditional_jump": "unconditional_jump",
}


class PreflightError(RuntimeError):
    """The run is scientifically invalid and must not issue API requests."""


class InvalidCompletion(RuntimeError):
    """An API response is not one of the K valid samples."""


class ResponseContractError(RuntimeError):
    """A returned provider response cannot support an auditable slot."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return sha256_bytes(stable_json_bytes(value))


def require_digest(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PreflightError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PreflightError(f"{label} does not exist: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must contain one JSON object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except Exception as exc:
        raise PreflightError(
            f"cannot parse {label} {path} at or before line {line_number}: {exc}"
        ) from exc
    if not rows:
        raise PreflightError(f"{label} is empty: {path}")
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class JsonlJournal:
    """Thread-safe, flush-on-record JSONL append journal."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, value: Mapping[str, Any]) -> None:
        line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def import_frozen_module(path: Path, expected_sha256: str, name: str) -> Any:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise PreflightError(
            f"{name} hash mismatch: expected {expected_sha256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location(
        f"frontier_frozen_{name}_{actual[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot import frozen {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constant_preamble(strings: Sequence[Any], numbers: Sequence[Any]) -> str:
    """Reproduce build_real_enriched.py exactly."""
    normalized_strings = sorted(
        {str(value) for value in strings}, key=lambda value: (len(value), value)
    )[:48]
    normalized_numbers = sorted(
        {str(value) for value in numbers}, key=lambda value: (len(value), value)
    )[:48]
    parts: list[str] = []
    if normalized_strings:
        parts.append(
            "strings: " + " ".join(f'"{value}"' for value in normalized_strings)
        )
    if normalized_numbers:
        parts.append("numbers: " + " ".join(normalized_numbers))
    if not parts:
        return ""
    return "// constant pool recovered from binary\n// " + " | ".join(parts) + "\n"


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return [int(value) for value in ids]


def _tokenizer_decode(tokenizer: Any, ids: Sequence[int]) -> str:
    if not ids:
        return ""
    return str(tokenizer.decode(list(ids), skip_special_tokens=False))


@dataclass(frozen=True)
class PreparedInput:
    task_id: str
    compact_ids_sha256: str
    compact_text_sha256: str
    canonical_sha256: str
    constants_record_sha256: str
    constants_extraction_error: str | None
    constant_prefix_ids: tuple[int, ...]
    constant_prefix_text: str
    graph_ids: tuple[int, ...]
    canonical: dict[str, Any]
    readable_compact: str


class CompactArtifactBundle:
    """Load and prove the exact student source before making it API-readable."""

    def __init__(
        self,
        *,
        contract_path: Path,
        codebook_path: Path,
        tokenizer_path: Path,
        codec_path: Path,
        constants_path: Path,
        expected_constants_sha256: str,
        max_constant_prefix_tokens: int | None = 256,
    ) -> None:
        self.contract_path = require_file(contract_path, "compact contract")
        self.codebook_path = require_file(codebook_path, "compact codebook")
        self.tokenizer_path = require_file(tokenizer_path, "base tokenizer")
        self.codec_path = require_file(codec_path, "compact codec")
        self.constants_path = require_file(constants_path, "binary constants")
        if (
            max_constant_prefix_tokens is not None
            and int(max_constant_prefix_tokens) <= 0
        ):
            raise PreflightError(
                "max_constant_prefix_tokens must be positive or None"
            )
        self.max_constant_prefix_tokens = (
            int(max_constant_prefix_tokens)
            if max_constant_prefix_tokens is not None
            else None
        )
        self.contract = load_json(self.contract_path, "compact contract")
        self.codebook = load_json(self.codebook_path, "compact codebook")

        if self.contract.get("schema") != "direct-compact-causal-v1":
            raise PreflightError(
                f"unexpected compact contract schema: {self.contract.get('schema')!r}"
            )
        if self.codebook.get("schema") != "compact-qwen-v1-codebook":
            raise PreflightError(
                f"unexpected compact codebook schema: {self.codebook.get('schema')!r}"
            )

        self.codec_sha256 = require_digest(
            self.contract.get("codec_sha256"), "contract codec_sha256"
        )
        self.codebook_sha256 = require_digest(
            self.contract.get("codebook_sha256"), "contract codebook_sha256"
        )
        self.tokenizer_sha256 = require_digest(
            self.contract.get("tokenizer_json_sha256"),
            "contract tokenizer_json_sha256",
        )
        expected_constants_sha256 = require_digest(
            expected_constants_sha256, "expected constants SHA-256"
        )
        actual = {
            "codec": sha256_file(self.codec_path),
            "codebook": sha256_file(self.codebook_path),
            "tokenizer": sha256_file(self.tokenizer_path),
            "constants": sha256_file(self.constants_path),
        }
        expected = {
            "codec": self.codec_sha256,
            "codebook": self.codebook_sha256,
            "tokenizer": self.tokenizer_sha256,
            "constants": expected_constants_sha256,
        }
        mismatches = [
            f"{name}: expected {expected[name]}, got {actual[name]}"
            for name in expected
            if expected[name] != actual[name]
        ]
        if mismatches:
            raise PreflightError("artifact hash mismatch: " + "; ".join(mismatches))

        if self.codebook.get("tokenizer_json_sha256") != self.tokenizer_sha256:
            raise PreflightError("codebook tokenizer hash disagrees with contract")
        if int(self.codebook.get("model_vocab_size", -1)) != int(
            self.contract.get("base_vocab_size", -2)
        ):
            raise PreflightError("base/model vocabulary size mismatch")
        if self.codebook.get("source_token_expansions") != self.contract.get(
            "source_token_expansions"
        ):
            raise PreflightError(
                "contract source-token expansions disagree with codebook"
            )

        source_atom_ids = self.codebook.get("source_atom_ids")
        if not isinstance(source_atom_ids, dict) or not source_atom_ids:
            raise PreflightError("codebook has no source_atom_ids")
        self.atom_ids = {str(key): int(value) for key, value in source_atom_ids.items()}
        if len(set(self.atom_ids.values())) != len(self.atom_ids):
            raise PreflightError("source_atom_ids are not one-to-one")
        contract_ids = [int(value) for value in self.contract.get("source_token_ids", [])]
        if sorted(contract_ids) != sorted(self.atom_ids.values()):
            raise PreflightError("contract source_token_ids disagree with codebook")
        if GRAPH_MARKER not in self.atom_ids:
            raise PreflightError(f"codebook has no {GRAPH_MARKER} atom")
        self.id_atoms = {value: key for key, value in self.atom_ids.items()}
        self.graph_marker_id = self.atom_ids[GRAPH_MARKER]
        self.base_vocab_size = int(self.contract["base_vocab_size"])
        self.max_source_tokens = int(self.contract["max_source_tokens"])
        self.expansions = [str(value) for value in self.codebook.get("expansions", [])]
        if not self.expansions:
            raise PreflightError("codebook has no instruction expansions")
        self.code = {value: index for index, value in enumerate(self.expansions)}

        try:
            from tokenizers import Tokenizer
        except Exception as exc:
            raise PreflightError(
                "the 'tokenizers' package is required for sealed-input verification"
            ) from exc
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.visible_symbols = visible_one_token_symbols(self.tokenizer)
        self.codec = import_frozen_module(
            self.codec_path, self.codec_sha256, "compact_codec"
        )

        constant_rows = load_jsonl(self.constants_path, "binary constants")
        self.constants: dict[str, dict[str, Any]] = {}
        for row in constant_rows:
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise PreflightError("binary-constants row has no task_id")
            if task_id in self.constants:
                raise PreflightError(f"duplicate binary-constants task_id: {task_id}")
            self.constants[task_id] = row

    def artifact_records(self) -> dict[str, dict[str, Any]]:
        return {
            "contract": file_record(self.contract_path),
            "codebook": file_record(self.codebook_path),
            "tokenizer": file_record(self.tokenizer_path),
            "codec": file_record(self.codec_path),
            "constants": file_record(self.constants_path),
        }

    def _graph_text_from_ids(self, graph_ids: Sequence[int]) -> str:
        pieces: list[str] = []
        base_run: list[int] = []

        def flush() -> None:
            if base_run:
                pieces.append(_tokenizer_decode(self.tokenizer, base_run))
                base_run.clear()

        for token_id in graph_ids:
            atom = self.id_atoms.get(int(token_id))
            if atom is not None:
                flush()
                pieces.append(atom)
            elif 0 <= int(token_id) < self.base_vocab_size:
                base_run.append(int(token_id))
            else:
                raise PreflightError(
                    f"compact source contains unknown token id {token_id}"
                )
        flush()
        return "".join(pieces)

    def _expected_prefix(
        self, task_id: str, graph_length: int
    ) -> tuple[list[int], str, str, str | None]:
        record = self.constants.get(task_id)
        if record is None:
            raise PreflightError(
                f"task {task_id} has no row in the pinned binary-constants artifact"
            )
        extraction_error = (
            str(record.get("err")) if record.get("err") not in (None, "") else None
        )
        preamble = constant_preamble(
            record.get("strings") or [], record.get("numbers") or []
        )
        # build_real_enriched.py called Tokenizer.encode(pre) without overriding
        # add_special_tokens, so preserve that exact default here.
        encoded_value = self.tokenizer.encode(preamble)
        encoded_ids = (
            encoded_value.ids if hasattr(encoded_value, "ids") else encoded_value
        )
        encoded = [
            int(token_id)
            for token_id in encoded_ids
            if int(token_id) < self.base_vocab_size
        ]
        if self.max_constant_prefix_tokens is not None:
            encoded = encoded[: self.max_constant_prefix_tokens]
        room = self.max_source_tokens - graph_length
        if room < 0:
            raise PreflightError(
                f"task {task_id} compact graph exceeds contract source budget"
            )
        expected = encoded[:room] if preamble and room > 0 else []
        return (
            expected,
            _tokenizer_decode(self.tokenizer, expected),
            stable_sha256(record),
            extraction_error,
        )

    def prepare(self, row: Mapping[str, Any]) -> PreparedInput:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise PreflightError("dataset row has no task_id")
        expected_row_hashes = {
            "compact_codec_sha256": self.codec_sha256,
            "compact_codebook_sha256": self.codebook_sha256,
            "compact_tokenizer_sha256": self.tokenizer_sha256,
        }
        for key, expected in expected_row_hashes.items():
            actual = str(row.get(key) or "")
            if actual != expected:
                raise PreflightError(
                    f"task {task_id} {key} mismatch: expected {expected}, got {actual!r}"
                )

        raw_ids = row.get("compact_input_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise PreflightError(f"task {task_id} has no compact_input_ids")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
            raise PreflightError(f"task {task_id} has non-integer compact_input_ids")
        ids = [int(value) for value in raw_ids]
        if len(ids) > self.max_source_tokens:
            raise PreflightError(
                f"task {task_id} has {len(ids)} source tokens, contract allows "
                f"{self.max_source_tokens}"
            )
        marker_positions = [
            index for index, token_id in enumerate(ids) if token_id == self.graph_marker_id
        ]
        if len(marker_positions) != 1:
            raise PreflightError(
                f"task {task_id} must contain exactly one {GRAPH_MARKER} marker"
            )
        marker = marker_positions[0]
        prefix_ids, graph_ids = ids[:marker], ids[marker:]
        if any(token_id >= self.base_vocab_size for token_id in prefix_ids):
            raise PreflightError(
                f"task {task_id} constant prefix contains a custom graph token"
            )
        (
            expected_prefix,
            prefix_text,
            constants_record_sha,
            constants_extraction_error,
        ) = self._expected_prefix(task_id, len(graph_ids))
        if prefix_ids != expected_prefix:
            raise PreflightError(
                f"task {task_id} constant prefix does not match pinned "
                "binary-constants enrichment"
            )

        graph_text = self._graph_text_from_ids(graph_ids)
        if not graph_text.startswith(GRAPH_MARKER):
            raise PreflightError(f"task {task_id} graph text has no leading marker")
        try:
            canonical = self.codec.decode(graph_text, self.expansions)
            reencoded = self.codec.encode(canonical, self.code)
            roundtrip_ids = self.codec.compact_ids(
                reencoded, self.tokenizer, self.atom_ids
            )
        except Exception as exc:
            raise PreflightError(
                f"task {task_id} compact graph cannot be decoded losslessly: {exc}"
            ) from exc
        if reencoded != graph_text:
            raise PreflightError(
                f"task {task_id} compact text failed exact encode/decode round trip"
            )
        if [int(value) for value in roundtrip_ids] != graph_ids:
            raise PreflightError(
                f"task {task_id} compact ids failed exact codec/tokenizer round trip"
            )
        canonical_hash = stable_sha256(canonical)
        readable = serialize_compact_graph(
            prefix_text,
            canonical,
            self.code,
            tokenizer=self.tokenizer,
            visible_symbols=self.visible_symbols,
        )
        return PreparedInput(
            task_id=task_id,
            compact_ids_sha256=stable_sha256(ids),
            compact_text_sha256=sha256_text(graph_text),
            canonical_sha256=canonical_hash,
            constants_record_sha256=constants_record_sha,
            constants_extraction_error=constants_extraction_error,
            constant_prefix_ids=tuple(prefix_ids),
            constant_prefix_text=prefix_text,
            graph_ids=tuple(graph_ids),
            canonical=canonical,
            readable_compact=readable,
        )


def prepare_api_readable_compact(
    bundle: CompactArtifactBundle,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Public adapter for frontier evaluation or black-box sequence-KL harvest.

    The returned ``text`` contains no opaque ``<I…>``/``<B…>`` IDs.  It is
    emitted only after contract/file/row hash checks, exact compact
    encode/decode/token-ID round trips, and reconstruction of the exact binary
    constant prefix supplied to the student.
    """
    prepared = bundle.prepare(row)
    return {
        "schema": SCHEMA_VERSION,
        "representation_schema": F2_SCHEMA,
        "system_prompt_sha256": sha256_text(COMPACT_F2_SYSTEM_PROMPT),
        "task_id": prepared.task_id,
        "text": prepared.readable_compact,
        "text_sha256": sha256_text(prepared.readable_compact),
        "compact_ids_sha256": prepared.compact_ids_sha256,
        "compact_text_sha256": prepared.compact_text_sha256,
        "canonical_sha256": prepared.canonical_sha256,
        "constants_record_sha256": prepared.constants_record_sha256,
        "constants_extraction_error": prepared.constants_extraction_error,
        "constant_prefix_tokens": len(prepared.constant_prefix_ids),
        "graph_tokens": len(prepared.graph_ids),
        "verified": {
            "artifact_hashes": True,
            "row_contract_hashes": True,
            "codec_text_roundtrip": True,
            "codec_token_id_roundtrip": True,
            "student_constant_prefix": True,
            "per_task_instruction_dictionary_roundtrip": True,
            "compact_semantic_f2_roundtrip": True,
            "branch_targets_reconstructed_from_cfg": True,
            "visible_task_symbols_one_token": True,
            "opaque_custom_ids_in_text": False,
        },
    }


def _serialize_compact_graph_f1_legacy(
    prefix_text: str,
    canonical: Mapping[str, Any],
    codebook_index: Mapping[str, int],
) -> str:
    """Render a lossless API-readable per-task instruction dictionary.

    Instructions used more than once receive a short task-local integer
    I-reference and are defined exactly once. A single-use instruction is
    expanded inline (``!instruction``), because defining and then referencing
    it would make the black-box representation longer without preserving any
    additional information. Every block stream is expanded and checked against
    the canonical instruction list before any text is returned.
    """
    instruction_counts: dict[str, int] = {}
    canonical_blocks = canonical.get("blocks") or []
    for block in canonical_blocks:
        for instruction_value in block.get("instructions") or []:
            instruction = str(instruction_value)
            if any(character in instruction for character in "\r\n\t"):
                raise PreflightError(
                    "normalized instruction contains a dictionary delimiter"
                )
            if ";" in instruction:
                raise PreflightError(
                    "normalized instruction contains the block-stream delimiter"
                )
            instruction_counts[instruction] = (
                instruction_counts.get(instruction, 0) + 1
            )

    definitions: dict[str, str] = {}
    instruction_refs: dict[str, str] = {}
    for block in canonical_blocks:
        for instruction_value in block.get("instructions") or []:
            instruction = str(instruction_value)
            if instruction_counts[instruction] < 2 or instruction in instruction_refs:
                continue
            # A codebook instruction is proven by the compact round trip. Raw
            # fallbacks are equally lossless and may also benefit from one local
            # definition when repeated.
            reference = str(len(instruction_refs))
            instruction_refs[instruction] = reference
            definitions[reference] = instruction

    block_streams: list[tuple[int, list[str]]] = []
    for block in canonical_blocks:
        entries: list[str] = []
        for instruction_value in block.get("instructions") or []:
            instruction = str(instruction_value)
            entries.append(instruction_refs.get(instruction, "!" + instruction))
        block_streams.append((int(block["id"]), entries))

    if len(block_streams) != len(canonical_blocks):
        raise PreflightError("instruction-dictionary block count mismatch")
    for block, (block_id, entries) in zip(canonical_blocks, block_streams):
        if int(block["id"]) != block_id:
            raise PreflightError("instruction-dictionary block identity mismatch")
        reconstructed = [
            entry[1:] if entry.startswith("!") else definitions[entry]
            for entry in entries
        ]
        expected = [str(value) for value in block.get("instructions") or []]
        if reconstructed != expected:
            raise PreflightError(
                f"instruction-dictionary round trip failed for block B{block_id}"
            )

    lines = [
        "F1",
        "CONSTANTS",
    ]
    if prefix_text:
        lines.extend(prefix_text.rstrip("\n").splitlines())
    else:
        lines.append("(none)")
    lines.append("ARCH " + str(canonical.get("architecture") or "x86_64"))
    entry_blocks = canonical.get("entry_blocks") or []
    lines.append("ENTRY " + ",".join(str(int(value)) for value in entry_blocks))
    lines.append("IREFS n=normalized_instruction")
    if definitions:
        for reference, instruction in definitions.items():
            lines.append(f"{reference}={instruction}")
    else:
        lines.append("(none)")
    lines.append("BLOCKS block:ordered_;_entries n=IREF !x=inline")
    for block_id, entries in block_streams:
        lines.append(f"{block_id}:" + ";".join(entries))
    edge_codes = {
        "conditional_true": "T",
        "conditional_false": "F",
        "linear_fallthrough": "N",
        "loop_backedge": "L",
        "unconditional": "U",
        "unconditional_jump": "J",
    }
    lines.append(
        "EDGE_TYPES T=true F=false N=fallthrough L=loopback "
        "U=unconditional J=jump"
    )
    lines.append("CFG sourceLetterTarget")
    edges = canonical.get("cfg_edges") or []
    if edges:
        for edge in edges:
            edge_type = str(edge["edge_type"])
            if edge_type not in EDGE_LABELS:
                raise PreflightError(f"unknown CFG edge type {edge_type!r}")
            lines.append(
                f"{int(edge['source'])}{edge_codes[edge_type]}"
                f"{int(edge['target'])}"
            )
    else:
        lines.append("(none)")
    lines.append("END")
    return "\n".join(lines) + "\n"


def serialize_compact_graph(
    prefix_text: str,
    canonical: Mapping[str, Any],
    codebook_index: Mapping[str, int],
    *,
    tokenizer: Any,
    visible_symbols: Sequence[str] | None = None,
) -> str:
    """Render and prove the lossless F2 API-readable semantic graph."""
    del codebook_index  # The sealed codec round trip is proved before F2.
    try:
        return serialize_f2(
            prefix_text,
            canonical,
            tokenizer=tokenizer,
            visible_symbols=visible_symbols,
        )
    except F2CodecError as exc:
        raise PreflightError(f"lossless F2 serialization failed: {exc}") from exc


# Stable collector integration point. The exact same string must be used for
# prompt-token preflight and for every F2 API request.
COMPACT_F2_SYSTEM_PROMPT = F2_SYSTEM_PROMPT
SYSTEM_PROMPT = COMPACT_F2_SYSTEM_PROMPT

RAW_SYSTEM_PROMPT = (
    "You are an expert reverse engineer. Reconstruct a correct Dart top-level "
    "function named `fn0` with the same observable behavior as the supplied "
    "complete x86-64 Dart AOT disassembly. Return one self-contained Dart "
    "compilation-unit fragment containing fn0 plus any required imports and "
    "helper declarations. Do not return main, tests, markdown fences, or prose "
    "outside Dart comments."
)


def build_messages(
    *,
    arm: str,
    prepared: PreparedInput,
    raw_disassembly: str | None = None,
) -> list[dict[str, str]]:
    if arm == "compact":
        user = prepared.readable_compact
        system = SYSTEM_PROMPT
    elif arm in {"raw", "raw_constants"}:
        if not raw_disassembly or not raw_disassembly.strip():
            raise PreflightError(
                f"task {prepared.task_id} has no complete raw disassembly"
            )
        sections = [
            "REPRESENTATION: complete fn0-family GDB disassembly",
            "RAW DISASSEMBLY:",
            raw_disassembly.rstrip(),
        ]
        if arm == "raw_constants":
            sections.extend(
                [
                    "BINARY-RECOVERED CONSTANT PREFIX AS SEEN BY STUDENT:",
                    prepared.constant_prefix_text.rstrip()
                    if prepared.constant_prefix_text
                    else "(none)",
                ]
            )
        sections.append("Return Dart fn0 now.")
        user = "\n".join(sections)
        system = RAW_SYSTEM_PROMPT
    else:
        raise PreflightError(f"unsupported frontier arm: {arm}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def count_prompt_tokens(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    chat_overhead_reserve: int,
) -> dict[str, int]:
    if chat_overhead_reserve < 0:
        raise PreflightError("chat_overhead_reserve must be non-negative")
    system_tokens = 0
    user_tokens = 0
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        count = len(_tokenizer_encode(tokenizer, content))
        if role == "system":
            system_tokens += count
        elif role == "user":
            user_tokens += count
        else:
            raise PreflightError(f"unsupported prompt role {role!r}")
    return {
        "system_tokens": system_tokens,
        "user_tokens": user_tokens,
        "chat_overhead_reserve": chat_overhead_reserve,
        "estimated_prompt_tokens": system_tokens
        + user_tokens
        + chat_overhead_reserve,
    }


def flatten_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    raise InvalidCompletion("response content contains a non-text block")
            else:
                raise InvalidCompletion("response content has an unsupported block")
        return "".join(parts)
    if value is None:
        return ""
    raise InvalidCompletion("response content is not text")


def extract_dart_code(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    fenced = re.findall(
        r"```(?:dart)?\s*\n?(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        stripped = max(fenced, key=len).strip()
    stripped = re.sub(r"^\s*(?:Dart(?:\s+source)?|Answer)\s*:\s*", "", stripped)
    return stripped.strip()


def _mask_comments_and_strings(code: str) -> str:
    """Mask Dart comments/strings while preserving token boundaries."""
    out: list[str] = []
    index = 0
    length = len(code)
    while index < length:
        if code.startswith("//", index):
            end = code.find("\n", index)
            if end < 0:
                out.append(" " * (length - index))
                break
            out.append(" " * (end - index))
            out.append("\n")
            index = end + 1
            continue
        if code.startswith("/*", index):
            end = code.find("*/", index + 2)
            end = length if end < 0 else end + 2
            segment = code[index:end]
            out.append("".join("\n" if char == "\n" else " " for char in segment))
            index = end
            continue
        prefix_length = 0
        quote = ""
        if code.startswith("r'''", index) or code.startswith("R'''", index):
            prefix_length, quote = 1, "'''"
        elif code.startswith('r"""', index) or code.startswith('R"""', index):
            prefix_length, quote = 1, '"""'
        elif code.startswith("'''", index):
            quote = "'''"
        elif code.startswith('"""', index):
            quote = '"""'
        elif code[index] in {"'", '"'}:
            quote = code[index]
        elif (
            code[index] in {"r", "R"}
            and index + 1 < length
            and code[index + 1] in {"'", '"'}
        ):
            prefix_length, quote = 1, code[index + 1]
        if quote:
            start = index
            cursor = index + prefix_length + len(quote)
            raw = prefix_length == 1
            while cursor < length:
                if code.startswith(quote, cursor):
                    cursor += len(quote)
                    break
                if not raw and code[cursor] == "\\":
                    cursor += 2
                else:
                    cursor += 1
            segment = code[start:cursor]
            out.append("".join("\n" if char == "\n" else " " for char in segment))
            index = cursor
            continue
        out.append(code[index])
        index += 1
    return "".join(out)


def candidate_safety_reasons(code: str) -> list[str]:
    masked = _mask_comments_and_strings(code)
    reasons: list[str] = []
    checks = [
        (r"\bmain\s*\(", "defines main"),
        (r"(?<![\w.])exit\s*\(", "calls process exit"),
        (r"\bIsolate\s*\.\s*exit\b", "calls Isolate.exit"),
        (r"\bProcess\s*\.\s*killPid\b", "calls Process.killPid"),
        (r"\bProcessSignal\b[\s\S]{0,120}\.\s*send\s*\(", "sends a process signal"),
    ]
    for pattern, reason in checks:
        if re.search(pattern, masked):
            reasons.append(reason)
    if not re.search(r"\bfn0\s*\(", masked):
        reasons.append("does not define fn0")
    return reasons


def _object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return dumped
    raise InvalidCompletion("SDK response cannot be serialized")


@dataclass(frozen=True)
class ValidatedCompletion:
    response_id: str
    response_model: str
    response_created: int | None
    finish_reason: str
    content: str
    reasoning_content: str
    code: str
    code_sha256: str
    usage: dict[str, int]
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class TerminalProviderResponse:
    """One successfully returned provider response occupying one K slot."""

    response_id: str
    response_model: str
    response_created: int | None
    finish_reason: str
    content: str
    reasoning_content: str
    refusal_present: bool
    candidate_valid: bool
    terminal_reason: str
    code: str
    code_sha256: str | None
    usage: dict[str, int]
    raw_response: dict[str, Any]


def classify_terminal_provider_response(
    response: Any,
    *,
    expected_model: str,
    max_prompt_tokens: int,
    requested_max_tokens: int,
) -> TerminalProviderResponse:
    """Validate a response envelope and classify its terminal candidate.

    Every returned response with a valid identity and internally consistent
    usage occupies exactly one sample slot. ``finish_reason`` is recorded but
    does not by itself invalidate extractable safe Dart code.
    """

    if not expected_model:
        raise ResponseContractError("expected resolved model is empty")
    if max_prompt_tokens <= 0 or requested_max_tokens <= 0:
        raise ResponseContractError("token caps must be positive")
    try:
        raw = _object_dict(response)
    except InvalidCompletion as exc:
        raise ResponseContractError(str(exc)) from exc

    response_id = str(raw.get("id") or "")
    if not response_id:
        raise ResponseContractError("response id is missing")
    response_model = str(raw.get("model") or "")
    if not response_model:
        raise ResponseContractError("resolved response model is missing")
    if response_model != expected_model:
        raise ResponseContractError(
            f"resolved response model {response_model!r} does not equal "
            f"requested model {expected_model!r}"
        )

    usage_raw = raw.get("usage")
    if not isinstance(usage_raw, dict):
        raise ResponseContractError("response has no token usage")
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage_raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ResponseContractError(f"usage.{key} is missing")
        if value < 0:
            raise ResponseContractError(f"usage.{key} is negative")
        usage[key] = value
    if usage["prompt_tokens"] <= 0:
        raise ResponseContractError("usage.prompt_tokens is not positive")
    if usage["total_tokens"] <= 0:
        raise ResponseContractError("usage.total_tokens is not positive")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ResponseContractError("usage.total_tokens is internally inconsistent")
    if usage["prompt_tokens"] > max_prompt_tokens:
        raise ResponseContractError(
            f"provider counted {usage['prompt_tokens']} prompt tokens, cap is "
            f"{max_prompt_tokens}"
        )
    if usage["completion_tokens"] > requested_max_tokens:
        raise ResponseContractError(
            f"provider counted {usage['completion_tokens']} completion tokens, "
            f"requested cap is {requested_max_tokens}"
        )

    created_raw = raw.get("created")
    created = int(created_raw) if isinstance(created_raw, (int, float)) else None
    choices = raw.get("choices")
    finish_reason = ""
    content = ""
    reasoning = ""
    refusal_present = False
    code = ""
    terminal_reason = ""

    if not isinstance(choices, list) or len(choices) != 1:
        terminal_reason = "response_choice_count_is_not_one"
    else:
        choice = choices[0]
        if not isinstance(choice, dict):
            terminal_reason = "response_choice_is_malformed"
        else:
            finish_reason = str(choice.get("finish_reason") or "")
            message = choice.get("message")
            if not isinstance(message, dict):
                terminal_reason = "response_has_no_assistant_message"
            else:
                refusal = message.get("refusal")
                refusal_present = refusal not in (None, "", [])
                try:
                    content = flatten_content(message.get("content")).strip()
                except InvalidCompletion:
                    terminal_reason = "response_content_is_not_text"
                try:
                    reasoning = flatten_content(
                        message.get("reasoning_content")
                    ).strip()
                except InvalidCompletion:
                    reasoning = ""
                if not terminal_reason and refusal_present:
                    terminal_reason = "response_contains_refusal"
                if not terminal_reason and not content:
                    terminal_reason = "response_content_is_empty"
                if not terminal_reason:
                    embedded_reasoning = re.findall(
                        r"<think>(.*?)</think>",
                        content,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    code_content = re.sub(
                        r"<think>.*?</think>",
                        "",
                        content,
                        flags=re.IGNORECASE | re.DOTALL,
                    ).strip()
                    if embedded_reasoning and not reasoning:
                        reasoning = "\n\n".join(
                            value.strip() for value in embedded_reasoning
                        )
                    code = extract_dart_code(code_content)
                    if not code:
                        terminal_reason = "no_dart_candidate_could_be_extracted"
                    else:
                        safety_reasons = candidate_safety_reasons(code)
                        if safety_reasons:
                            terminal_reason = (
                                "unsafe_or_invalid_candidate:"
                                + "; ".join(safety_reasons)
                            )

    candidate_valid = not terminal_reason
    if candidate_valid:
        terminal_reason = "candidate_valid"
    return TerminalProviderResponse(
        response_id=response_id,
        response_model=response_model,
        response_created=created,
        finish_reason=finish_reason,
        content=content,
        reasoning_content=reasoning,
        refusal_present=refusal_present,
        candidate_valid=candidate_valid,
        terminal_reason=terminal_reason,
        code=code,
        code_sha256=sha256_text(code) if code else None,
        usage=usage,
        raw_response=raw,
    )


def validate_completion(
    response: Any,
    *,
    max_prompt_tokens: int,
    max_output_tokens: int,
) -> ValidatedCompletion:
    raw = _object_dict(response)
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise InvalidCompletion("response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise InvalidCompletion("response choice is malformed")
    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason != "stop":
        raise InvalidCompletion(f"finish_reason is {finish_reason!r}, not 'stop'")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise InvalidCompletion("response has no assistant message")
    refusal = message.get("refusal")
    if refusal not in (None, "", []):
        raise InvalidCompletion("response contains a refusal")
    content = flatten_content(message.get("content")).strip()
    if not content:
        raise InvalidCompletion("response content is empty")
    reasoning = flatten_content(message.get("reasoning_content")).strip()
    embedded_reasoning = re.findall(
        r"<think>(.*?)</think>", content, flags=re.IGNORECASE | re.DOTALL
    )
    code_content = re.sub(
        r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    if embedded_reasoning and not reasoning:
        reasoning = "\n\n".join(value.strip() for value in embedded_reasoning)
    code = extract_dart_code(code_content)
    if not code:
        raise InvalidCompletion("no Dart candidate could be extracted")
    safety_reasons = candidate_safety_reasons(code)
    if safety_reasons:
        raise InvalidCompletion("unsafe/invalid candidate: " + "; ".join(safety_reasons))

    usage_raw = raw.get("usage")
    if not isinstance(usage_raw, dict):
        raise InvalidCompletion("response has no token usage")
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage_raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidCompletion(f"usage.{key} is missing")
        usage[key] = value
        if usage[key] < 0:
            raise InvalidCompletion(f"usage.{key} is negative")
    if any(usage[key] <= 0 for key in usage):
        raise InvalidCompletion("token usage is zero")
    if usage["prompt_tokens"] > max_prompt_tokens:
        raise InvalidCompletion(
            f"provider counted {usage['prompt_tokens']} prompt tokens, cap is "
            f"{max_prompt_tokens}"
        )
    if usage["completion_tokens"] > max_output_tokens:
        raise InvalidCompletion(
            f"provider counted {usage['completion_tokens']} completion tokens, "
            f"cap is {max_output_tokens}"
        )
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise InvalidCompletion("usage.total_tokens is internally inconsistent")
    response_id = str(raw.get("id") or "")
    response_model = str(raw.get("model") or "")
    if not response_id:
        raise InvalidCompletion("response id is missing")
    if not response_model:
        raise InvalidCompletion("resolved response model is missing")
    created_raw = raw.get("created")
    created = int(created_raw) if isinstance(created_raw, (int, float)) else None
    return ValidatedCompletion(
        response_id=response_id,
        response_model=response_model,
        response_created=created,
        finish_reason=finish_reason,
        content=content,
        reasoning_content=reasoning,
        code=code,
        code_sha256=sha256_text(code),
        usage=usage,
        raw_response=raw,
    )


class TokenBudget:
    """Reserve worst-case tokens before API calls so concurrency cannot overshoot."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("token budget cannot be negative")
        self.limit = limit
        self.spent = 0
        self.reserved = 0
        self._lock = threading.Lock()

    def reserve(self, worst_case: int) -> bool:
        if worst_case <= 0:
            raise ValueError("token reservation must be positive")
        if self.limit == 0:
            return True
        with self._lock:
            if self.spent + self.reserved + worst_case > self.limit:
                return False
            self.reserved += worst_case
            return True

    def settle(self, worst_case: int, actual: int) -> None:
        if actual < 0:
            raise RuntimeError("actual token charge cannot be negative")
        if self.limit == 0:
            with self._lock:
                self.spent += actual
            return
        with self._lock:
            self.reserved -= worst_case
            if self.reserved < 0:
                raise RuntimeError("token budget reservation underflow")
            # A provider can theoretically report more tokens than requested.
            # Record that charge honestly; the caller treats it as fatal.
            self.spent += actual

    def cancel(self, worst_case: int) -> None:
        if self.limit == 0:
            return
        with self._lock:
            self.reserved -= worst_case
            if self.reserved < 0:
                raise RuntimeError("token budget reservation underflow")

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit": self.limit,
                "spent": self.spent,
                "reserved": self.reserved,
            }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * (
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _tool_version(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=30, check=False
        )
    except Exception as exc:
        raise PreflightError(f"cannot run {' '.join(command)}: {exc}") from exc
    text = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0 or not text:
        raise PreflightError(
            f"{' '.join(command)} failed with status {result.returncode}: {text[:300]}"
        )
    return text


def _gdb_function_names(info_output: str) -> list[str]:
    names: list[str] = []
    for line in info_output.splitlines():
        match = re.match(r"^\s*0x[0-9a-fA-F]+\s+(.+?)\s*$", line)
        if not match:
            continue
        symbol = match.group(1).strip().rstrip(";")
        symbol = re.sub(r"\(\)\s*$", "", symbol).strip()
        fn_position = symbol.find("fn0")
        if fn_position < 0:
            continue
        symbol = symbol[fn_position:]
        if symbol == "fn0" or symbol.startswith("fn0."):
            names.append(symbol)
    return sorted(set(names), key=lambda value: (value != "fn0", value))


def complete_raw_disassembly(
    *,
    task_id: str,
    dart_source: str,
    dart_binary: Path,
    cache_dir: Path,
    main_stub: str,
) -> tuple[str, dict[str, Any]]:
    """Compile and disassemble every fn0-family symbol without truncation."""
    dart_binary = require_file(dart_binary, "Dart compiler")
    try:
        gdb_version = _tool_version(["gdb", "--version"]).splitlines()[0]
        dart_version = _tool_version([str(dart_binary), "--version"]).splitlines()[0]
    except PreflightError:
        raise
    source_hash = sha256_text(dart_source)
    key_payload = {
        "schema": RAW_SCHEMA_VERSION,
        "task_id": task_id,
        "source_sha256": source_hash,
        "dart_binary_sha256": sha256_file(dart_binary),
        "dart_version": dart_version,
        "gdb_version": gdb_version,
        "main_stub_sha256": sha256_text(main_stub),
    }
    cache_key = stable_sha256(key_payload)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        cached = load_json(cache_path, "raw disassembly cache")
        if cached.get("key") != key_payload:
            raise PreflightError(f"raw cache provenance mismatch: {cache_path}")
        disassembly = str(cached.get("disassembly") or "")
        if not disassembly:
            raise PreflightError(f"raw cache has empty disassembly: {cache_path}")
        if sha256_text(disassembly) != cached.get("disassembly_sha256"):
            raise PreflightError(f"raw cache payload hash mismatch: {cache_path}")
        return disassembly, {
            "cache_key": cache_key,
            "cache_path": str(cache_path.resolve()),
            "cache_hit": True,
            "symbols": cached.get("symbols") or [],
            "key": key_payload,
            "disassembly_sha256": cached["disassembly_sha256"],
        }

    with tempfile.TemporaryDirectory(prefix="frontier_raw_") as temporary:
        root = Path(temporary)
        source_path = root / "program.dart"
        snapshot_path = root / "program.aot"
        source = dart_source
        if not re.search(r"\bmain\s*\(", _mask_comments_and_strings(source)):
            source += main_stub
        source_path.write_text(source, encoding="utf-8")
        compile_result = subprocess.run(
            [
                str(dart_binary),
                "compile",
                "aot-snapshot",
                str(source_path),
                "-o",
                str(snapshot_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if compile_result.returncode != 0 or not snapshot_path.is_file():
            diagnostic = (compile_result.stderr or compile_result.stdout or "")[:1000]
            raise PreflightError(
                f"task {task_id} raw-control compilation failed: {diagnostic}"
            )
        info_result = subprocess.run(
            [
                "gdb",
                "-batch",
                "-ex",
                "set disassembly-flavor intel",
                "-ex",
                f'file "{snapshot_path}"',
                "-ex",
                "info functions ^fn0",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if info_result.returncode != 0:
            raise PreflightError(
                f"task {task_id} gdb symbol query failed: "
                f"{(info_result.stderr or '')[:1000]}"
            )
        symbols = _gdb_function_names(info_result.stdout)
        if "fn0" not in symbols:
            raise PreflightError(
                f"task {task_id} raw control found no root fn0 symbol"
            )
        chunks: list[str] = []
        for symbol in symbols:
            disassemble_result = subprocess.run(
                [
                    "gdb",
                    "-batch",
                    "-ex",
                    "set disassembly-flavor intel",
                    "-ex",
                    f'file "{snapshot_path}"',
                    "-ex",
                    f"disassemble /r '{symbol}'",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if (
                disassemble_result.returncode != 0
                or "Dump of assembler" not in disassemble_result.stdout
            ):
                raise PreflightError(
                    f"task {task_id} failed to disassemble symbol {symbol!r}"
                )
            chunks.append(
                f"===== SYMBOL {symbol} =====\n{disassemble_result.stdout.rstrip()}"
            )
        disassembly = "\n\n".join(chunks) + "\n"
    cached = {
        "schema": RAW_SCHEMA_VERSION,
        "key": key_payload,
        "symbols": symbols,
        "disassembly": disassembly,
        "disassembly_sha256": sha256_text(disassembly),
        "created_at": utc_now(),
    }
    atomic_write_json(cache_path, cached)
    return disassembly, {
        "cache_key": cache_key,
        "cache_path": str(cache_path.resolve()),
        "cache_hit": False,
        "symbols": symbols,
        "key": key_payload,
        "disassembly_sha256": cached["disassembly_sha256"],
    }


def public_dataclass(value: Any) -> Any:
    return asdict(value) if hasattr(value, "__dataclass_fields__") else value
