"""Closed, fail-closed references for feedback-conditioned VeRPO rescue.

The judge must not invent labels for F2 blocks, instructions, CFG edges, or
candidate source locations.  This module decodes the *exact* F2 payload with
the local/pinned ``frontier_f2.py`` implementation and assigns deterministic
task-local aliases:

* ``F2B000`` -- canonical block 0
* ``F2B000:I000`` -- instruction 0 in canonical block 0
* ``F2E000`` -- canonical CFG edge 0
* ``C000:L0001`` -- line 1 in candidate 0
* ``C000:BOF`` / ``C000:EOF`` -- candidate insertion boundaries

The compact prompt catalogue deliberately does not repeat expanded assembly
instructions.  The original F2 payload remains the semantic source shown to
the judge; the catalogue supplies only a closed reference vocabulary and CFG
relations.  Its digest binds the exact F2 bytes, decoder bytes, candidates,
and optional diagnostics.

Invalid judge items return item-local :class:`ValidationResult` objects.  A
bad item therefore cannot suppress valid sibling items, and no invalid or
unknown reference is accepted by substring matching.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


GROUNDING_SCHEMA = "verpo-rescue-grounding-v1"
EXPECTED_F2_SCHEMA = "lossless-semantic-f2"

EVIDENCE_KINDS = frozenset(
    {
        "f2_block",
        "f2_instruction",
        "f2_edge",
        "candidate_line",
        "diagnostic",
    }
)
EDIT_OPERATIONS = frozenset(
    {
        "insert_before",
        "insert_after",
        "replace_range",
        "delete_range",
        "unknown",
    }
)

MAX_F2_BYTES = 2_000_000
MAX_CANDIDATES = 128
MAX_CANDIDATE_BYTES = 1_000_000
MAX_CANDIDATE_LINES = 9_999
MAX_DIAGNOSTIC_BYTES = 256_000
# Reference widths are part of the closed grammar consumed by the judge.
# Refuse graphs that would spill into a wider, ambiguously parsed namespace.
MAX_BLOCKS = 1_000
MAX_INSTRUCTIONS_PER_BLOCK = 1_000
MAX_INSTRUCTIONS = MAX_BLOCKS * MAX_INSTRUCTIONS_PER_BLOCK
MAX_EDGES = 1_000
MAX_EVIDENCE_ITEMS = 32
MAX_CLAIM_CHARACTERS = 4_000

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_DECODER_CACHE: dict[tuple[str, str], types.ModuleType] = {}
_DECODER_CACHE_LOCK = threading.Lock()
_MISSING = object()


class GroundingError(ValueError):
    """The F2 source or grounding catalogue cannot be trusted."""


@dataclass(frozen=True)
class F2InstructionReference:
    ref: str
    block_ref: str
    block_id: int
    instruction_index: int
    text: str


@dataclass(frozen=True)
class F2BlockReference:
    ref: str
    block_id: int
    is_entry: bool
    instruction_refs: tuple[str, ...]


@dataclass(frozen=True)
class F2EdgeReference:
    ref: str
    edge_index: int
    source_ref: str
    target_ref: str
    edge_type: str


@dataclass(frozen=True)
class CandidateLineReference:
    ref: str
    line_number: int
    text: str


@dataclass(frozen=True)
class CandidateReference:
    candidate_index: int
    source_sha256: str
    bof_ref: str
    lines: tuple[CandidateLineReference, ...]
    eof_ref: str
    diagnostic_ref: str | None
    diagnostic: str | None


@dataclass(frozen=True)
class ValidationIssue:
    """A stable telemetry-ready reason for rejecting one diagnosis item."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Item-local grounding validation result.

    ``normalized_*`` values contain only the grounding fields that passed
    syntactic normalization.  Callers must use ``valid`` before consuming
    them; partially normalized data is retained only for debugging receipts.
    """

    valid: bool
    candidate_index: int | None
    issues: tuple[ValidationIssue, ...]
    normalized_evidence: tuple[dict[str, str], ...] = ()
    normalized_edit_location: dict[str, Any] | None = None

    @property
    def rejection_causes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(issue.code for issue in self.issues))


@dataclass(frozen=True)
class GroundingCatalog:
    """Exact decoded F2 structure plus closed candidate references."""

    frontier_f2_sha256: str
    f2_source_sha256: str
    constant_prefix_sha256: str
    f2_schema: str
    architecture: str
    entry_block_refs: tuple[str, ...]
    blocks: tuple[F2BlockReference, ...]
    instructions: tuple[F2InstructionReference, ...]
    edges: tuple[F2EdgeReference, ...]
    candidates: tuple[CandidateReference, ...]

    @property
    def block_refs(self) -> frozenset[str]:
        return frozenset(value.ref for value in self.blocks)

    @property
    def instruction_refs(self) -> frozenset[str]:
        return frozenset(value.ref for value in self.instructions)

    @property
    def edge_refs(self) -> frozenset[str]:
        return frozenset(value.ref for value in self.edges)

    @property
    def candidate_line_refs(self) -> frozenset[str]:
        return frozenset(
            line.ref
            for candidate in self.candidates
            for line in candidate.lines
        )

    @property
    def candidate_anchor_refs(self) -> frozenset[str]:
        return frozenset(
            ref
            for candidate in self.candidates
            for ref in (
                candidate.bof_ref,
                *(line.ref for line in candidate.lines),
                candidate.eof_ref,
            )
        )

    @property
    def diagnostic_refs(self) -> frozenset[str]:
        return frozenset(
            candidate.diagnostic_ref
            for candidate in self.candidates
            if candidate.diagnostic_ref is not None
        )

    def candidate(self, candidate_index: int) -> CandidateReference:
        if (
            type(candidate_index) is not int
            or candidate_index < 0
            or candidate_index >= len(self.candidates)
        ):
            raise GroundingError(
                f"candidate index {candidate_index!r} is outside the catalogue"
            )
        candidate = self.candidates[candidate_index]
        if candidate.candidate_index != candidate_index:
            raise GroundingError("candidate catalogue ordering is corrupt")
        return candidate

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-serializable closed vocabulary.

        ``catalog_sha256`` is the canonical SHA-256 of this mapping with that
        one field removed.  Expanded instruction text is intentionally absent
        to avoid undoing F2 compression.
        """

        payload: dict[str, Any] = {
            "schema": GROUNDING_SCHEMA,
            "frontier_f2_sha256": self.frontier_f2_sha256,
            "f2_source_sha256": self.f2_source_sha256,
            "constant_prefix_sha256": self.constant_prefix_sha256,
            "f2_schema": self.f2_schema,
            "architecture": self.architecture,
            "entry_block_refs": list(self.entry_block_refs),
            "blocks": [
                {
                    "ref": block.ref,
                    "entry": block.is_entry,
                    "instruction_refs": list(block.instruction_refs),
                }
                for block in self.blocks
            ],
            "edges": [
                {
                    "ref": edge.ref,
                    "source_ref": edge.source_ref,
                    "target_ref": edge.target_ref,
                    "edge_type": edge.edge_type,
                }
                for edge in self.edges
            ],
            "candidates": [
                {
                    "candidate_index": candidate.candidate_index,
                    "source_sha256": candidate.source_sha256,
                    "bof_ref": candidate.bof_ref,
                    "lines": [
                        {"ref": line.ref, "text": line.text}
                        for line in candidate.lines
                    ],
                    "eof_ref": candidate.eof_ref,
                    "diagnostic": (
                        None
                        if candidate.diagnostic_ref is None
                        else {
                            "ref": candidate.diagnostic_ref,
                            "text": candidate.diagnostic,
                        }
                    ),
                }
                for candidate in self.candidates
            ],
            "allowed_refs": {
                "f2_block": sorted(self.block_refs),
                "f2_instruction": sorted(self.instruction_refs),
                "f2_edge": sorted(self.edge_refs),
                "candidate_line": sorted(self.candidate_line_refs),
                "candidate_anchor": sorted(self.candidate_anchor_refs),
                "diagnostic": sorted(self.diagnostic_refs),
            },
        }
        payload["catalog_sha256"] = canonical_payload_sha256(payload)
        return copy.deepcopy(payload)

    @property
    def catalog_sha256(self) -> str:
        return self.to_prompt_dict()["catalog_sha256"]

    @property
    def prompt_text(self) -> str:
        """Compact prompt rendering; it can replace an unnumbered candidate."""

        return (
            "CLOSED_REFERENCE_CATALOG "
            "(use only exact refs; candidate text is line-numbered here)\n"
            + json.dumps(
                self.to_prompt_dict(),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a prompt payload after excluding its self-referential digest."""

    if not isinstance(payload, Mapping):
        raise GroundingError("catalog payload must be a mapping")
    body = {
        str(key): value
        for key, value in payload.items()
        if str(key) != "catalog_sha256"
    }
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise GroundingError(
            "catalog payload is not canonical UTF-8 JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def default_frontier_f2_path() -> Path:
    """Resolve the workspace-local decoder without consulting ``sys.path``."""

    return (
        Path(__file__).resolve().parents[3]
        / "frontier_ceiling_patch_v1"
        / "frontier_f2.py"
    )


def _utf8_bytes(value: str, label: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeError as exc:
        raise GroundingError(f"{label} is not valid UTF-8 text") from exc


def _checked_expected_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GroundingError(
            "expected_frontier_f2_sha256 must be 64 hexadecimal characters"
        )
    return value.lower()


def _load_frontier_f2(
    path: str | Path | None,
    expected_sha256: str | None,
) -> tuple[types.ModuleType, str]:
    resolved = Path(path) if path is not None else default_frontier_f2_path()
    try:
        resolved = resolved.resolve(strict=True)
        source = resolved.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise GroundingError(
            f"cannot read the pinned frontier_f2 decoder: {resolved}"
        ) from exc

    digest = hashlib.sha256(source).hexdigest()
    expected = _checked_expected_sha256(expected_sha256)
    if expected is not None and digest != expected:
        raise GroundingError(
            "frontier_f2 decoder SHA-256 does not match the pinned digest"
        )

    cache_key = (str(resolved), digest)
    with _DECODER_CACHE_LOCK:
        cached = _DECODER_CACHE.get(cache_key)
        if cached is not None:
            return cached, digest

    try:
        source_text = source.decode("utf-8")
        code = compile(source_text, str(resolved), "exec")
        module = types.ModuleType(
            f"_verpo_rescue_frontier_f2_{digest[:16]}"
        )
        module.__file__ = str(resolved)
        module.__package__ = ""
        exec(code, module.__dict__)
    except Exception as exc:
        raise GroundingError(
            "the pinned frontier_f2 decoder could not be loaded"
        ) from exc

    if getattr(module, "F2_SCHEMA", None) != EXPECTED_F2_SCHEMA:
        raise GroundingError(
            "frontier_f2 decoder exposes an unexpected F2 schema"
        )
    if not callable(getattr(module, "decode_f2", None)):
        raise GroundingError("frontier_f2 decoder has no decode_f2 function")
    edge_types = getattr(module, "EDGE_TYPE_TO_CODE", None)
    if not isinstance(edge_types, Mapping) or not edge_types:
        raise GroundingError(
            "frontier_f2 decoder has no closed CFG edge vocabulary"
        )

    with _DECODER_CACHE_LOCK:
        existing = _DECODER_CACHE.setdefault(cache_key, module)
    return existing, digest


def _plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise GroundingError(f"{label} must be an integer")
    return value


def _safe_structural_text(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise GroundingError(f"{label} must be text")
    if not value and not allow_empty:
        raise GroundingError(f"{label} must not be empty")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise GroundingError(f"{label} contains a structural control character")
    _utf8_bytes(value, label)
    return value


def _candidate_lines(source: str) -> tuple[str, ...]:
    # Use source-code line semantics, not str.splitlines(), which also splits
    # on several Unicode control characters that are legal inside JSON text.
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return ()
    values = normalized.split("\n")
    if normalized.endswith("\n"):
        values.pop()
    return tuple(values)


def _block_ref(block_id: int) -> str:
    return f"F2B{block_id:03d}"


def _instruction_ref(block_id: int, instruction_index: int) -> str:
    return f"{_block_ref(block_id)}:I{instruction_index:03d}"


def _edge_ref(edge_index: int) -> str:
    return f"F2E{edge_index:03d}"


def _candidate_prefix(candidate_index: int) -> str:
    return f"C{candidate_index:03d}"


def build_grounding_catalog(
    f2_text: str,
    candidates: Sequence[str],
    *,
    diagnostics: Sequence[str | None] | None = None,
    frontier_f2_path: str | Path | None = None,
    expected_frontier_f2_sha256: str | None = None,
) -> GroundingCatalog:
    """Decode exact F2 and build deterministic task-local references.

    Decoder/source failures raise :class:`GroundingError`; continuing with an
    approximate graph would make evidence grounding unsound.
    """

    if not isinstance(f2_text, str):
        raise GroundingError("f2_text must be text")
    f2_bytes = _utf8_bytes(f2_text, "f2_text")
    if len(f2_bytes) > MAX_F2_BYTES:
        raise GroundingError("f2_text exceeds the fail-closed byte limit")
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise GroundingError("candidates must be a sequence of source strings")
    if not candidates or len(candidates) > MAX_CANDIDATES:
        raise GroundingError(
            f"candidate count must be between 1 and {MAX_CANDIDATES}"
        )
    if diagnostics is None:
        diagnostic_values: tuple[str | None, ...] = (None,) * len(candidates)
    else:
        if isinstance(diagnostics, (str, bytes)) or not isinstance(
            diagnostics, Sequence
        ):
            raise GroundingError(
                "diagnostics must be a sequence aligned with candidates"
            )
        if len(diagnostics) != len(candidates):
            raise GroundingError(
                "diagnostics and candidates must have identical lengths"
            )
        diagnostic_values = tuple(diagnostics)

    decoder, decoder_sha256 = _load_frontier_f2(
        frontier_f2_path,
        expected_frontier_f2_sha256,
    )
    try:
        constant_prefix, canonical = decoder.decode_f2(f2_text)
    except Exception as exc:
        raise GroundingError(
            "f2_text failed exact frontier_f2 decoding"
        ) from exc
    if not isinstance(constant_prefix, str) or not isinstance(
        canonical, Mapping
    ):
        raise GroundingError("frontier_f2 returned an invalid decoded shape")

    architecture = _safe_structural_text(
        canonical.get("architecture"), "F2 architecture"
    )
    raw_blocks = canonical.get("blocks")
    raw_entries = canonical.get("entry_blocks")
    raw_edges = canonical.get("cfg_edges")
    if not isinstance(raw_blocks, list):
        raise GroundingError("decoded F2 blocks must be a list")
    if not isinstance(raw_entries, list):
        raise GroundingError("decoded F2 entry blocks must be a list")
    if not isinstance(raw_edges, list):
        raise GroundingError("decoded F2 CFG edges must be a list")
    if not raw_blocks or len(raw_blocks) > MAX_BLOCKS:
        raise GroundingError(
            f"decoded F2 block count must be between 1 and {MAX_BLOCKS}"
        )
    if len(raw_edges) > MAX_EDGES:
        raise GroundingError("decoded F2 edge count exceeds the safety limit")

    entry_ids = tuple(
        _plain_int(value, "F2 entry block") for value in raw_entries
    )
    if len(set(entry_ids)) != len(entry_ids):
        raise GroundingError("decoded F2 entry block list contains duplicates")
    if any(value < 0 or value >= len(raw_blocks) for value in entry_ids):
        raise GroundingError("decoded F2 entry block is outside the graph")
    entry_id_set = frozenset(entry_ids)

    block_references: list[F2BlockReference] = []
    instruction_references: list[F2InstructionReference] = []
    for expected_id, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, Mapping):
            raise GroundingError("decoded F2 block must be an object")
        block_id = _plain_int(raw_block.get("id"), "F2 block id")
        if block_id != expected_id:
            raise GroundingError(
                "decoded F2 blocks are not ordered contiguous canonical IDs"
            )
        raw_instructions = raw_block.get("instructions")
        if not isinstance(raw_instructions, list):
            raise GroundingError("decoded F2 instructions must be a list")
        if len(raw_instructions) > MAX_INSTRUCTIONS_PER_BLOCK:
            raise GroundingError(
                "decoded F2 block exceeds the instruction-reference limit"
            )
        if len(instruction_references) + len(raw_instructions) > MAX_INSTRUCTIONS:
            raise GroundingError(
                "decoded F2 instruction count exceeds the safety limit"
            )
        refs: list[str] = []
        for instruction_index, raw_instruction in enumerate(raw_instructions):
            instruction = _safe_structural_text(
                raw_instruction,
                "decoded F2 instruction",
            )
            ref = _instruction_ref(block_id, instruction_index)
            refs.append(ref)
            instruction_references.append(
                F2InstructionReference(
                    ref=ref,
                    block_ref=_block_ref(block_id),
                    block_id=block_id,
                    instruction_index=instruction_index,
                    text=instruction,
                )
            )
        block_references.append(
            F2BlockReference(
                ref=_block_ref(block_id),
                block_id=block_id,
                is_entry=block_id in entry_id_set,
                instruction_refs=tuple(refs),
            )
        )

    allowed_edge_types = frozenset(decoder.EDGE_TYPE_TO_CODE)
    edge_references: list[F2EdgeReference] = []
    for edge_index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, Mapping):
            raise GroundingError("decoded F2 CFG edge must be an object")
        source = _plain_int(raw_edge.get("source"), "F2 edge source")
        target = _plain_int(raw_edge.get("target"), "F2 edge target")
        edge_type = _safe_structural_text(
            raw_edge.get("edge_type"), "F2 edge type"
        )
        if source < 0 or source >= len(raw_blocks):
            raise GroundingError("decoded F2 edge has an unknown source block")
        if target < 0 or target >= len(raw_blocks):
            raise GroundingError("decoded F2 edge has an unknown target block")
        if edge_type not in allowed_edge_types:
            raise GroundingError("decoded F2 edge has an unknown type")
        edge_references.append(
            F2EdgeReference(
                ref=_edge_ref(edge_index),
                edge_index=edge_index,
                source_ref=_block_ref(source),
                target_ref=_block_ref(target),
                edge_type=edge_type,
            )
        )

    candidate_references: list[CandidateReference] = []
    for candidate_index, (source, diagnostic_value) in enumerate(
        zip(candidates, diagnostic_values)
    ):
        if not isinstance(source, str):
            raise GroundingError(
                f"candidate {candidate_index} source must be text"
            )
        source_bytes = _utf8_bytes(source, f"candidate {candidate_index}")
        if len(source_bytes) > MAX_CANDIDATE_BYTES:
            raise GroundingError(
                f"candidate {candidate_index} exceeds the byte limit"
            )
        if "\x00" in source:
            raise GroundingError(
                f"candidate {candidate_index} contains a NUL character"
            )
        lines = _candidate_lines(source)
        if len(lines) > MAX_CANDIDATE_LINES:
            raise GroundingError(
                f"candidate {candidate_index} exceeds the line limit"
            )
        prefix = _candidate_prefix(candidate_index)
        line_references = tuple(
            CandidateLineReference(
                ref=f"{prefix}:L{line_number:04d}",
                line_number=line_number,
                text=line,
            )
            for line_number, line in enumerate(lines, start=1)
        )

        diagnostic: str | None
        diagnostic_ref: str | None
        if diagnostic_value is None or diagnostic_value == "":
            diagnostic = None
            diagnostic_ref = None
        else:
            if not isinstance(diagnostic_value, str):
                raise GroundingError(
                    f"diagnostic {candidate_index} must be text or null"
                )
            diagnostic_bytes = _utf8_bytes(
                diagnostic_value, f"diagnostic {candidate_index}"
            )
            if len(diagnostic_bytes) > MAX_DIAGNOSTIC_BYTES:
                raise GroundingError(
                    f"diagnostic {candidate_index} exceeds the byte limit"
                )
            if "\x00" in diagnostic_value:
                raise GroundingError(
                    f"diagnostic {candidate_index} contains a NUL character"
                )
            diagnostic = diagnostic_value
            diagnostic_ref = f"{prefix}:DIAGNOSTIC"

        candidate_references.append(
            CandidateReference(
                candidate_index=candidate_index,
                source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                bof_ref=f"{prefix}:BOF",
                lines=line_references,
                eof_ref=f"{prefix}:EOF",
                diagnostic_ref=diagnostic_ref,
                diagnostic=diagnostic,
            )
        )

    return GroundingCatalog(
        frontier_f2_sha256=decoder_sha256,
        f2_source_sha256=hashlib.sha256(f2_bytes).hexdigest(),
        constant_prefix_sha256=hashlib.sha256(
            _utf8_bytes(constant_prefix, "decoded F2 constant prefix")
        ).hexdigest(),
        f2_schema=EXPECTED_F2_SCHEMA,
        architecture=architecture,
        entry_block_refs=tuple(_block_ref(value) for value in entry_ids),
        blocks=tuple(block_references),
        instructions=tuple(instruction_references),
        edges=tuple(edge_references),
        candidates=tuple(candidate_references),
    )


def _issue(
    issues: list[ValidationIssue],
    code: str,
    path: str,
    message: str,
) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message))


def _item_candidate_index(
    item: Mapping[str, Any],
    catalog: GroundingCatalog,
    expected_candidate_index: int | None,
    issues: list[ValidationIssue],
) -> int | None:
    if expected_candidate_index is not None and (
        type(expected_candidate_index) is not int
        or expected_candidate_index < 0
        or expected_candidate_index >= len(catalog.candidates)
    ):
        _issue(
            issues,
            "expected_candidate_index_invalid",
            "$",
            "the caller supplied an invalid expected candidate index",
        )
        return None

    raw = item.get("candidate_index", _MISSING)
    if raw is _MISSING:
        if expected_candidate_index is None:
            _issue(
                issues,
                "candidate_index_missing",
                "$.candidate_index",
                "candidate identity is neither explicit nor caller-bound",
            )
            return None
        return expected_candidate_index
    if type(raw) is not int:
        _issue(
            issues,
            "candidate_index_type",
            "$.candidate_index",
            "candidate_index must be an integer",
        )
        return None
    if raw < 0 or raw >= len(catalog.candidates):
        _issue(
            issues,
            "candidate_index_unknown",
            "$.candidate_index",
            "candidate_index is outside the closed catalogue",
        )
        return None
    if (
        expected_candidate_index is not None
        and raw != expected_candidate_index
    ):
        _issue(
            issues,
            "candidate_index_mismatch",
            "$.candidate_index",
            "candidate_index does not match the caller-bound item position",
        )
        return None
    return raw


def _candidate_ref_maps(
    catalog: GroundingCatalog,
) -> tuple[
    dict[str, tuple[int, str]],
    dict[str, tuple[int, int, str]],
    dict[str, int],
]:
    anchors: dict[str, tuple[int, str]] = {}
    lines: dict[str, tuple[int, int, str]] = {}
    diagnostics: dict[str, int] = {}
    for candidate in catalog.candidates:
        anchors[candidate.bof_ref] = (candidate.candidate_index, "")
        for line in candidate.lines:
            anchors[line.ref] = (candidate.candidate_index, line.text)
            lines[line.ref] = (
                candidate.candidate_index,
                line.line_number,
                line.text,
            )
        anchors[candidate.eof_ref] = (candidate.candidate_index, "")
        if candidate.diagnostic_ref is not None:
            diagnostics[candidate.diagnostic_ref] = candidate.candidate_index
    return anchors, lines, diagnostics


def _validate_evidence(
    raw_evidence: Any,
    catalog: GroundingCatalog,
    candidate_index: int | None,
    issues: list[ValidationIssue],
) -> tuple[dict[str, str], ...]:
    if not isinstance(raw_evidence, list):
        _issue(
            issues,
            "evidence_not_array",
            "$.evidence",
            "evidence must be a JSON array",
        )
        return ()
    if not raw_evidence:
        _issue(
            issues,
            "evidence_empty",
            "$.evidence",
            "at least one grounded evidence item is required",
        )
        return ()
    if len(raw_evidence) > MAX_EVIDENCE_ITEMS:
        _issue(
            issues,
            "evidence_too_many",
            "$.evidence",
            f"evidence has more than {MAX_EVIDENCE_ITEMS} items",
        )
        return ()

    anchors, lines, diagnostics = _candidate_ref_maps(catalog)
    del anchors
    allowed_by_kind: dict[str, frozenset[str]] = {
        "f2_block": catalog.block_refs,
        "f2_instruction": catalog.instruction_refs,
        "f2_edge": catalog.edge_refs,
        "candidate_line": frozenset(lines),
        "diagnostic": frozenset(diagnostics),
    }
    normalized: list[dict[str, str]] = []
    for index, raw_item in enumerate(raw_evidence):
        path = f"$.evidence[{index}]"
        if not isinstance(raw_item, Mapping):
            _issue(
                issues,
                "evidence_item_not_object",
                path,
                "evidence item must be an object",
            )
            continue
        if set(raw_item) != {"kind", "ref", "claim"}:
            _issue(
                issues,
                "evidence_shape",
                path,
                "evidence item must contain exactly kind, ref, and claim",
            )
            continue
        kind = raw_item.get("kind")
        ref = raw_item.get("ref")
        claim = raw_item.get("claim")
        if not isinstance(kind, str) or kind not in EVIDENCE_KINDS:
            _issue(
                issues,
                "evidence_kind_unknown",
                f"{path}.kind",
                "evidence kind is outside the closed vocabulary",
            )
            continue
        if not isinstance(ref, str) or ref not in allowed_by_kind[kind]:
            _issue(
                issues,
                "evidence_ref_unknown",
                f"{path}.ref",
                "evidence ref is not a member of the declared kind",
            )
            continue
        if (
            not isinstance(claim, str)
            or not claim.strip()
            or len(claim) > MAX_CLAIM_CHARACTERS
            or "\x00" in claim
        ):
            _issue(
                issues,
                "evidence_claim_invalid",
                f"{path}.claim",
                "evidence claim must be non-empty bounded text",
            )
            continue
        try:
            claim.encode("utf-8")
        except UnicodeError:
            _issue(
                issues,
                "evidence_claim_invalid",
                f"{path}.claim",
                "evidence claim is not valid UTF-8 text",
            )
            continue

        referenced_candidate: int | None = None
        if kind == "candidate_line":
            referenced_candidate = lines[ref][0]
        elif kind == "diagnostic":
            referenced_candidate = diagnostics[ref]
        if (
            referenced_candidate is not None
            and candidate_index is not None
            and referenced_candidate != candidate_index
        ):
            _issue(
                issues,
                "evidence_candidate_mismatch",
                f"{path}.ref",
                "candidate-local evidence points at a sibling candidate",
            )
            continue
        normalized.append({"kind": kind, "ref": ref, "claim": claim})
    return tuple(normalized)


def _validate_edit_location(
    raw_edit: Any,
    catalog: GroundingCatalog,
    candidate_index: int | None,
    issues: list[ValidationIssue],
) -> dict[str, Any] | None:
    if not isinstance(raw_edit, Mapping):
        _issue(
            issues,
            "edit_location_not_object",
            "$.edit_location",
            "edit_location must be an object",
        )
        return None

    operation = raw_edit.get("operation")
    if not isinstance(operation, str) or operation not in EDIT_OPERATIONS:
        _issue(
            issues,
            "edit_operation_unknown",
            "$.edit_location.operation",
            "edit operation is outside the closed vocabulary",
        )
        return None

    anchors, lines, _diagnostics = _candidate_ref_maps(catalog)
    if operation == "unknown":
        allowed = {
            "operation",
            "anchor_ref",
            "anchor_text",
            "start_ref",
            "end_ref",
        }
        if any(key not in allowed for key in raw_edit):
            _issue(
                issues,
                "edit_shape",
                "$.edit_location",
                "unknown edit location has unsupported fields",
            )
            return None
        if any(
            raw_edit.get(key) is not None
            for key in ("anchor_ref", "anchor_text", "start_ref", "end_ref")
            if key in raw_edit
        ):
            _issue(
                issues,
                "edit_unknown_has_anchor",
                "$.edit_location",
                "unknown edit location must not fabricate an anchor",
            )
            return None
        return {"operation": "unknown"}

    if operation in {"insert_before", "insert_after"}:
        required = {"operation", "anchor_ref", "anchor_text"}
        if set(raw_edit) != required:
            _issue(
                issues,
                "edit_shape",
                "$.edit_location",
                "insertion requires exactly operation, anchor_ref, anchor_text",
            )
            return None
        anchor_ref = raw_edit.get("anchor_ref")
        anchor_text = raw_edit.get("anchor_text")
        if not isinstance(anchor_ref, str) or anchor_ref not in anchors:
            _issue(
                issues,
                "edit_anchor_ref_unknown",
                "$.edit_location.anchor_ref",
                "insertion anchor is outside the candidate vocabulary",
            )
            return None
        owner, expected_text = anchors[anchor_ref]
        if candidate_index is not None and owner != candidate_index:
            _issue(
                issues,
                "edit_candidate_mismatch",
                "$.edit_location.anchor_ref",
                "insertion anchor points at a sibling candidate",
            )
            return None
        if not isinstance(anchor_text, str) or anchor_text != expected_text:
            _issue(
                issues,
                "edit_anchor_text_mismatch",
                "$.edit_location.anchor_text",
                "anchor_text is not the exact referenced candidate line",
            )
            return None
        return {
            "operation": operation,
            "anchor_ref": anchor_ref,
            "anchor_text": anchor_text,
        }

    required = {"operation", "start_ref", "end_ref", "anchor_text"}
    if set(raw_edit) != required:
        _issue(
            issues,
            "edit_shape",
            "$.edit_location",
            "range edit requires operation, start_ref, end_ref, anchor_text",
        )
        return None
    start_ref = raw_edit.get("start_ref")
    end_ref = raw_edit.get("end_ref")
    anchor_text = raw_edit.get("anchor_text")
    if (
        not isinstance(start_ref, str)
        or not isinstance(end_ref, str)
        or start_ref not in lines
        or end_ref not in lines
    ):
        _issue(
            issues,
            "edit_range_ref_unknown",
            "$.edit_location",
            "range endpoints must be exact candidate line refs",
        )
        return None
    start_owner, start_line, _ = lines[start_ref]
    end_owner, end_line, _ = lines[end_ref]
    if (
        start_owner != end_owner
        or (
            candidate_index is not None
            and start_owner != candidate_index
        )
    ):
        _issue(
            issues,
            "edit_candidate_mismatch",
            "$.edit_location",
            "range endpoints do not belong to the bound candidate",
        )
        return None
    if start_line > end_line:
        _issue(
            issues,
            "edit_range_order",
            "$.edit_location",
            "range start follows range end",
        )
        return None
    candidate = catalog.candidate(start_owner)
    expected_text = "\n".join(
        line.text
        for line in candidate.lines[start_line - 1 : end_line]
    )
    if not isinstance(anchor_text, str) or anchor_text != expected_text:
        _issue(
            issues,
            "edit_anchor_text_mismatch",
            "$.edit_location.anchor_text",
            "anchor_text is not the exact referenced candidate range",
        )
        return None
    return {
        "operation": operation,
        "start_ref": start_ref,
        "end_ref": end_ref,
        "anchor_text": anchor_text,
    }


def validate_diagnosis_item(
    item: Mapping[str, Any] | Any,
    catalog: GroundingCatalog,
    *,
    expected_candidate_index: int | None = None,
) -> ValidationResult:
    """Validate one judge diagnosis without raising on model output.

    Top-level diagnosis fields outside ``evidence``, ``edit_location``, and
    optional ``candidate_index`` are intentionally left to the response
    contract validator.  Nested grounding objects are strict and reject
    unknown fields.
    """

    issues: list[ValidationIssue] = []
    if not isinstance(catalog, GroundingCatalog):
        return ValidationResult(
            valid=False,
            candidate_index=None,
            issues=(
                ValidationIssue(
                    code="catalog_invalid",
                    path="$",
                    message="grounding catalogue has the wrong type",
                ),
            ),
        )
    if not isinstance(item, Mapping):
        return ValidationResult(
            valid=False,
            candidate_index=None,
            issues=(
                ValidationIssue(
                    code="item_not_object",
                    path="$",
                    message="diagnosis item must be an object",
                ),
            ),
        )

    try:
        candidate_index = _item_candidate_index(
            item,
            catalog,
            expected_candidate_index,
            issues,
        )
        evidence = _validate_evidence(
            item.get("evidence", _MISSING),
            catalog,
            candidate_index,
            issues,
        )
        edit_location = _validate_edit_location(
            item.get("edit_location", _MISSING),
            catalog,
            candidate_index,
            issues,
        )
    except Exception:
        # Model output must never terminate or partially admit a group.  Keep
        # this deliberately generic; detailed unexpected exceptions belong in
        # private application logs, not model-visible receipts.
        _issue(
            issues,
            "grounding_internal_error",
            "$",
            "grounding validation failed closed",
        )
        evidence = ()
        edit_location = None
        candidate_index = None

    return ValidationResult(
        valid=not issues,
        candidate_index=candidate_index,
        issues=tuple(issues),
        normalized_evidence=evidence,
        normalized_edit_location=edit_location,
    )


__all__ = [
    "EDIT_OPERATIONS",
    "EVIDENCE_KINDS",
    "EXPECTED_F2_SCHEMA",
    "GROUNDING_SCHEMA",
    "CandidateLineReference",
    "CandidateReference",
    "F2BlockReference",
    "F2EdgeReference",
    "F2InstructionReference",
    "GroundingCatalog",
    "GroundingError",
    "ValidationIssue",
    "ValidationResult",
    "build_grounding_catalog",
    "canonical_payload_sha256",
    "default_frontier_f2_path",
    "validate_diagnosis_item",
]
