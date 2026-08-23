#!/usr/bin/env python3
"""Build matched train/dev compact rows from complete AOT user-function bundles.

The direct-compact-v1 source-token ID set is kept byte-for-byte stable while
its instruction dictionary is refit from the 1,580 training graphs only.  Old
instruction slots used by train retain their indices; previously unused slots
are deterministically assigned to train-only new instructions.  Block and
control IDs never move.  A source-implicit inline-CFG codec then removes only
the redundant source-block token from each edge while retaining every edge.

Representation adapter:

* extractor F0 becomes ``@SELF`` and F1..Fn become ``@U0``..``@U(n-1)``;
* all per-function blocks are concatenated with global, contiguous block IDs;
* ``fn @SELF`` / ``fn @U<n>`` is prepended to every function entry block;
* local branch offsets become global ``@B<n>`` targets;
* external ``@X<n>`` calls stay distinct from user helpers and their exact,
  source-blind dictionary is included in the student/API-visible prefix;
* all internal CFG edges are offset and retained.

No source text or source symbol name is read to construct the representation.
The already sanitized imitation-train and executable held-out base rows are
copied only after all binary-side work and gates complete.
The extractor model projection, AOT digest, extractor digest, v1 artifacts and
F2 serializer are all hash-bound.  A build is published only if all 1,580 train
and 175 held-out rows are present, no extractor item was excluded/truncated,
the student representation fits 9K, and the exact F2 prompt fits 12K.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUILD_SCHEMA = "binary-multifunction-compact-build-v2"
ADAPTER_SCHEMA = "binary-multifunction-v1-semantic-adapter-v1"
# Keep the established split-seal envelope so downstream training validates
# the already sanitized target/harness role and the new representation.
SPLIT_SEAL_SCHEMA = "compact-public-private-join-seal-v1"
SANITATION_SCHEMA = "compact-target-harness-sanitation-v1"
IMITATION_TRAIN_SCOPE = "sequence_imitation_all_train"
F2_MANIFEST_SCHEMA = "verified-api-readable-compact-v2"
F2_ROW_SCHEMA = "audited-frontier-passk-v1"
EXTRACTOR_SCHEMA = "dart-aot-user-function-bundle-v1"
EXTRACTOR_MODEL_SCHEMA = "dart-aot-multifunction-graph-canonical-v1"
COMPACT_CONTRACT_SCHEMA = "direct-compact-causal-v1"
CODEBOOK_SCHEMA = "compact-qwen-v1-codebook"
OUTPUT_CODEBOOK_SCHEMA = "compact-qwen-inline-cfg-v2-codebook"
F2_REPRESENTATION_SCHEMA = "lossless-semantic-f2"
CODEBOOK_REFIT_ALGORITHM = (
    "train-only-stable-index-frequency-lexical-v1"
)

EXPECTED_TRAIN_ROWS = 1580
EXPECTED_DEV_ROWS = 175
STUDENT_TOKEN_LIMIT = 9000
API_PROMPT_TOKEN_LIMIT = 12000
CHAT_OVERHEAD_RESERVE = 256
GRAPH_MARKER = "<G2C1>"
ADAPTER_CONTRACT = {
    "schema": ADAPTER_SCHEMA,
    "function_aliases": "F0=@SELF; F(n>0)=@U(n-1)",
    "function_boundaries": "exact fn alias pseudo-instruction at each entry",
    "blocks": "function-order concatenation with contiguous global offsets",
    "cfg": "all per-function edges retained with the same global offsets",
    "external_aliases": (
        "exact extractor @X dictionary retained in binary prefix; never @U"
    ),
    "source_projection_metadata": {
        "function_kind": (
            "extractor provenance only; boundary and complete body retained"
        ),
        "instruction_byte_offsets": (
            "omitted after branch targets and every call transfer are proven"
        ),
        "interfunction_transfer_rows": (
            "omitted after one-to-one call operand/function-boundary proof"
        ),
    },
    "source_or_user_names_allowed": False,
    "keyed_private_source_symbol_attestation_required": True,
    "raw_attested_names_serialized": False,
    "truncation_allowed": False,
}
ADAPTER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        ADAPTER_CONTRACT,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
FUNCTION_ID_RE = re.compile(r"F([0-9]+)\Z")
EXTERNAL_ID_RE = re.compile(r"X([0-9]+)\Z")
TYPE_ATTESTATION_ALIAS_RE = re.compile(r"T([0-9]+)\Z")
FUNCTION_ATTESTATION_ALIAS_RE = re.compile(r"AF([0-9]+)\Z")
VISIBLE_TYPE_ATTESTATION_REF_RE = re.compile(r"@T([0-9]+)")
VISIBLE_FUNCTION_ATTESTATION_REF_RE = re.compile(r"@AF([0-9]+)")
FUNCTION_REF_RE = re.compile(r"@F([0-9]+)(\+0x[0-9a-fA-F]+)?")
LOCAL_REF_RE = re.compile(r"@L\+(0x[0-9a-fA-F]+|[0-9]+)")
EXTERNAL_REF_RE = re.compile(r"@X([0-9]+)")
ANNOTATED_ALIAS_RE = re.compile(
    r"0x[0-9a-fA-F]+\s*<"
    r"(@(?:SELF|U[0-9]+|X[0-9]+)(?:\+0x[0-9a-fA-F]+)?)>"
)
REMAINING_ALIAS_ANNOTATION_RE = re.compile(
    r"<(@(?:SELF|U[0-9]+|X[0-9]+)(?:\+0x[0-9a-fA-F]+)?)>"
)
CALL_PREFIX_RE = re.compile(
    r"^(?:(?:bnd|notrack)\s+)*(?:call|callq)\s+(.+)$",
    re.IGNORECASE,
)


class MultiFunctionBuildError(ValueError):
    """The complete matched representation cannot be proven."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_artifact_bytes(value: Any) -> bytes:
    """Return the exact bytes written by :func:`atomic_write_json`."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise MultiFunctionBuildError(f"{label} is not a lowercase SHA-256")
    return digest


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise MultiFunctionBuildError(f"missing required file: {resolved}")
    size = resolved.stat().st_size
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": size,
        "size_bytes": size,
    }


def require_file_hash(
    path: str | Path, expected_sha256: str, label: str
) -> dict[str, Any]:
    record = file_record(path)
    expected = require_digest(expected_sha256, f"expected {label} digest")
    if record["sha256"] != expected:
        raise MultiFunctionBuildError(
            f"{label} hash mismatch: expected {expected}, "
            f"got {record['sha256']}"
        )
    return record


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MultiFunctionBuildError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise MultiFunctionBuildError(f"{label} is not a JSON object")
    return value


def validate_sanitized_base_seal(
    *,
    dataset_path: Path,
    seal_path: Path,
    dataset_record: Mapping[str, Any],
    contract_sha256: str,
    role: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Bind the builder to the sanitizer's intended train/dev role views."""

    seal = load_json(seal_path, f"sanitized base {role} seal")
    if seal.get("schema") != SPLIT_SEAL_SCHEMA:
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal schema mismatch"
        )
    if seal.get("selected_role") != role:
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal role mismatch"
        )
    if seal.get("sanitation_schema") != SANITATION_SCHEMA:
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal sanitation schema mismatch"
        )
    for field in (
        "sanitizer_sha256",
        "evaluator_sha256",
        "quarantine_sha256",
    ):
        require_digest(seal.get(field), f"sanitized base {role} {field}")
    if not str(seal.get("completion_attestation_id") or "").strip():
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal has no completion attestation"
        )
    if not str(seal.get("dart_version") or "").strip():
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal has no Dart runtime version"
        )
    if not isinstance(seal.get("stability_runs"), int) or int(
        seal["stability_runs"]
    ) < 2:
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal has insufficient stability runs"
        )
    if seal.get("output_sha256") != dataset_record["sha256"]:
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal dataset mismatch"
        )
    if seal.get("contract_sha256") != contract_sha256:
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal contract mismatch"
        )
    if int(seal.get("rows", -1)) != int(expected_rows):
        raise MultiFunctionBuildError(
            f"sanitized base {role} seal row-count mismatch"
        )
    observed_rows = sum(
        1 for line in dataset_path.open(encoding="utf-8") if line.strip()
    )
    if observed_rows != int(expected_rows):
        raise MultiFunctionBuildError(
            f"sanitized base {role} dataset row-count mismatch"
        )

    if role == "fit":
        if seal.get("training_objective_scope") != IMITATION_TRAIN_SCOPE:
            raise MultiFunctionBuildError(
                "base train must be the sanitizer's all-1580 sequence-"
                "imitation view, not its executable-reward subset"
            )
        ineligible = seal.get("execution_ineligible_task_ids")
        if not isinstance(ineligible, list) or any(
            not isinstance(task_id, str) or not task_id
            for task_id in ineligible
        ):
            raise MultiFunctionBuildError(
                "sanitized base fit seal has invalid execution-ineligible IDs"
            )
        if len(set(ineligible)) != len(ineligible):
            raise MultiFunctionBuildError(
                "sanitized base fit seal repeats execution-ineligible IDs"
            )
        eligible_rows = seal.get("executable_reward_eligible_rows")
        if not isinstance(eligible_rows, int):
            raise MultiFunctionBuildError(
                "sanitized base fit seal has no executable eligibility count"
            )
        if eligible_rows + len(ineligible) != int(expected_rows):
            raise MultiFunctionBuildError(
                "sanitized base fit seal eligibility accounting mismatch"
            )
    elif role == "measure":
        if seal.get("training_objective_scope") == "executable_reward_only":
            raise MultiFunctionBuildError(
                "held-out development seal unexpectedly authorizes training"
            )
    else:
        raise MultiFunctionBuildError(f"unsupported sanitized base role {role!r}")

    return seal


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise MultiFunctionBuildError(
                    f"{label} has a blank line at {line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MultiFunctionBuildError(
                    f"{label} line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise MultiFunctionBuildError(
                    f"{label} line {line_number} is not an object"
                )
            values.append(value)
    if not values:
        raise MultiFunctionBuildError(f"{label} is empty")
    return values


def import_pinned_module(
    path: Path, expected_sha256: str, label: str
) -> Any:
    record = require_file_hash(path, expected_sha256, label)
    spec = importlib.util.spec_from_file_location(
        f"pinned_{label.replace(' ', '_')}_{record['sha256'][:12]}", path
    )
    if spec is None or spec.loader is None:
        raise MultiFunctionBuildError(f"cannot import {label}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, json_artifact_bytes(value))


def atomic_write_jsonl(
    path: Path, values: Iterable[Mapping[str, Any]]
) -> None:
    payload = b"".join(
        canonical_json_bytes(dict(value)) + b"\n" for value in values
    )
    _atomic_write_bytes(path, payload)


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    # Constants are semantic values, not source text.  Sorting makes the prefix
    # stable across extractor scheduling without imposing a lossy item cap.
    return sorted({str(value) for value in values}, key=lambda item: (len(item), item))


def binary_enrichment_preamble(
    strings: Sequence[Any],
    numbers: Sequence[Any],
    external_symbols: Sequence[Mapping[str, Any]],
) -> str:
    """Return the uncapped binary-only prefix seen by student and API teacher.

    External rows use a lossless positional encoding because repeating the
    three JSON field names for every alias needlessly consumes the student's
    fixed 9K source window.  Position ``n`` is exactly ``X<n>``; the parallel
    class stream uses ``T`` for ``trusted_runtime`` and ``N`` for
    ``neutralized_untrusted_runtime``.  The parser below reconstructs and
    verifies the exact extractor dictionary before a row can be published.
    """

    normalized_strings = _ordered_unique(strings)
    normalized_numbers = _ordered_unique(numbers)
    lines: list[str] = []
    if normalized_strings:
        lines.append(
            "// strings "
            + json.dumps(
                normalized_strings,
                # The pinned tokenizer normalizes some canonically equivalent
                # Unicode spellings while decoding (for example ``y`` followed
                # by U+0302 becomes U+0177).  JSON ASCII escapes preserve the
                # extractor's exact code-point sequence through token
                # encode/decode without dropping or normalizing information.
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    if normalized_numbers:
        lines.append(
            "// numbers "
            + json.dumps(
                normalized_numbers,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    if external_symbols:
        class_codes = {
            "trusted_runtime": "T",
            "neutralized_untrusted_runtime": "N",
        }
        classes: list[str] = []
        symbols: list[str | None] = []
        for index, value in enumerate(external_symbols):
            if str(value.get("external_id") or "") != f"X{index}":
                raise MultiFunctionBuildError(
                    "external dictionary is not contiguous X-index order"
                )
            symbol_class = str(value.get("symbol_class") or "")
            if symbol_class not in class_codes:
                raise MultiFunctionBuildError(
                    f"unsupported external symbol class {symbol_class!r}"
                )
            classes.append(class_codes[symbol_class])
            symbols.append(
                str(value["symbol"])
                if value.get("symbol") is not None
                else None
            )
        lines.append(
            "// externals[X=index,T=runtime,N=neutralized]:"
            + "".join(classes)
            + "|"
            + json.dumps(
                symbols,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def parse_external_dictionary_from_preamble(
    preamble: str,
) -> list[dict[str, Any]]:
    marker = "// externals[X=index,T=runtime,N=neutralized]:"
    matching_lines = [
        line for line in preamble.splitlines() if line.startswith(marker)
    ]
    if not matching_lines:
        return []
    if len(matching_lines) != 1:
        raise MultiFunctionBuildError(
            "external dictionary prefix occurs more than once"
        )
    payload = matching_lines[0][len(marker) :]
    classes, separator, encoded_symbols = payload.partition("|")
    if not separator:
        raise MultiFunctionBuildError(
            "external dictionary prefix has no class/symbol separator"
        )
    try:
        symbols = json.loads(encoded_symbols)
    except json.JSONDecodeError as exc:
        raise MultiFunctionBuildError("external symbol array is not JSON") from exc
    if (
        not isinstance(symbols, list)
        or len(symbols) != len(classes)
        or any(symbol is not None and not isinstance(symbol, str) for symbol in symbols)
        or any(code not in {"T", "N"} for code in classes)
    ):
        raise MultiFunctionBuildError(
            "external dictionary prefix has an invalid shape"
        )
    class_names = {
        "T": "trusted_runtime",
        "N": "neutralized_untrusted_runtime",
    }
    return [
        {
            "external_id": f"X{index}",
            "symbol": symbol,
            "symbol_class": class_names[classes[index]],
        }
        for index, symbol in enumerate(symbols)
    ]


def _function_alias(function_id: str, function_count: int) -> str:
    match = FUNCTION_ID_RE.fullmatch(function_id)
    if match is None:
        raise MultiFunctionBuildError(f"invalid function ID {function_id!r}")
    index = int(match.group(1))
    if not 0 <= index < function_count:
        raise MultiFunctionBuildError(
            f"function reference {function_id} is out of range"
        )
    return "@SELF" if index == 0 else f"@U{index - 1}"


def _call_operand(instruction: str) -> str | None:
    match = CALL_PREFIX_RE.fullmatch(instruction.strip())
    return match.group(1) if match is not None else None


def prove_transfer_table_redundant(
    bundle: Mapping[str, Any],
) -> dict[str, int]:
    """Prove every transfer row is exactly represented by a call instruction.

    The compact graph intentionally omits byte-coordinate bookkeeping.  This
    proof establishes that the semantically relevant transfer target/operand is
    retained one-for-one in the ordered instruction streams.
    """

    calls: dict[tuple[str, int], str] = {}
    for function in bundle.get("functions") or []:
        function_id = str(function.get("function_id") or "")
        for instruction in function.get("instructions") or []:
            text = str(instruction.get("text") or "")
            operand = _call_operand(text)
            if operand is None:
                continue
            key = (function_id, int(instruction["offset"]))
            if key in calls:
                raise MultiFunctionBuildError(
                    f"duplicate call coordinate {key}"
                )
            calls[key] = operand

    transfers = bundle.get("interfunction_transfers") or []
    if not isinstance(transfers, list):
        raise MultiFunctionBuildError("interfunction_transfers is not an array")
    represented: set[tuple[str, int]] = set()
    kind_counts: Counter[str] = Counter()
    for transfer in transfers:
        if not isinstance(transfer, dict):
            raise MultiFunctionBuildError("transfer row is not an object")
        key = (
            str(transfer.get("caller_function_id") or ""),
            int(transfer.get("caller_offset", -1)),
        )
        if key in represented or key not in calls:
            raise MultiFunctionBuildError(
                f"transfer/call coordinate mismatch at {key}"
            )
        kind = str(transfer.get("transfer_kind") or "")
        if kind == "direct_internal_call":
            target = str(transfer.get("target_function_id") or "")
            target_offset = int(transfer.get("target_offset", -1))
            if target_offset < 0:
                raise MultiFunctionBuildError("negative internal target offset")
            expected = f"@{target}"
            if target_offset:
                expected += f"+0x{target_offset:x}"
        elif kind == "direct_external_call":
            expected = "@" + str(transfer.get("external_id") or "")
        elif kind in {"indirect_call", "unlabelled_direct_external_call"}:
            expected = str(transfer.get("operand") or "")
        else:
            raise MultiFunctionBuildError(
                f"unsupported transfer kind {kind!r}"
            )
        if calls[key] != expected:
            raise MultiFunctionBuildError(
                f"transfer target mismatch at {key}: "
                f"{calls[key]!r} != {expected!r}"
            )
        represented.add(key)
        kind_counts[kind] += 1
    if represented != set(calls):
        missing = sorted(set(calls) - represented)
        raise MultiFunctionBuildError(
            f"{len(missing)} call instructions have no transfer row: "
            f"{missing[:5]}"
        )
    return {
        "call_instruction_count": len(calls),
        "transfer_row_count": len(transfers),
        "direct_internal_call_count": kind_counts["direct_internal_call"],
        "direct_external_call_count": kind_counts["direct_external_call"],
        "indirect_call_count": kind_counts["indirect_call"],
        "unlabelled_direct_external_call_count": kind_counts[
            "unlabelled_direct_external_call"
        ],
    }


def _validate_attestation_aliases(
    values: Any,
    *,
    row_field: str,
    alias_re: re.Pattern[str],
) -> list[str]:
    if not isinstance(values, list):
        raise MultiFunctionBuildError(
            f"source-symbol attestation {row_field} is not an array"
        )
    result: list[str] = []
    for expected_index, raw in enumerate(values):
        if not isinstance(raw, dict) or set(raw) != {row_field}:
            raise MultiFunctionBuildError(
                f"source-symbol attestation {row_field} row is not name-free"
            )
        alias = str(raw.get(row_field) or "")
        match = alias_re.fullmatch(alias)
        if match is None or int(match.group(1)) != expected_index:
            raise MultiFunctionBuildError(
                f"source-symbol attestation {row_field} aliases must be "
                "ordered and contiguous"
            )
        result.append(alias)
    return result


def _validate_source_symbol_attestation(
    bundle: Mapping[str, Any],
    source_projection: Mapping[str, Any],
    extractor_module: Any,
) -> dict[str, Any]:
    """Require the extractor's keyed, complete, name-free public binding."""

    if (
        bundle.get("source_text_read") is not False
        or bundle.get("raw_source_names_serialized") is not False
        or bundle.get("raw_source_paths_serialized") is not False
    ):
        raise MultiFunctionBuildError(
            "extractor source/raw-name serialization truth fields failed"
        )
    if (
        bundle.get("source_symbol_attestation_used") is not True
        or bundle.get("source_symbol_attestation_is_keyed") is not True
    ):
        raise MultiFunctionBuildError(
            "complete keyed source-symbol attestation is required"
        )

    type_aliases = _validate_attestation_aliases(
        bundle.get("type_aliases"),
        row_field="type_alias",
        alias_re=TYPE_ATTESTATION_ALIAS_RE,
    )
    function_aliases = _validate_attestation_aliases(
        bundle.get("function_attestation_aliases"),
        row_field="function_attestation_alias",
        alias_re=FUNCTION_ATTESTATION_ALIAS_RE,
    )
    if source_projection.get("type_aliases") != bundle.get("type_aliases"):
        raise MultiFunctionBuildError(
            "extractor type-attestation aliases changed in model projection"
        )
    if source_projection.get("function_attestation_aliases") != bundle.get(
        "function_attestation_aliases"
    ):
        raise MultiFunctionBuildError(
            "extractor function-attestation aliases changed in model projection"
        )

    binding = bundle.get("symbol_attestation_binding")
    expected_binding_fields = {
        "schema",
        "attestation_file_sha256",
        "attestation_row_sha256",
        "key_id_sha256",
        "function_symbol_count",
        "type_symbol_count",
        "complete",
        "raw_names_present",
    }
    if not isinstance(binding, dict) or set(binding) != expected_binding_fields:
        raise MultiFunctionBuildError(
            "source-symbol attestation binding shape is not exact/name-free"
        )
    if binding.get("schema") != extractor_module.SYMBOL_ATTESTATION_SCHEMA:
        raise MultiFunctionBuildError(
            "source-symbol attestation binding schema mismatch"
        )
    for field in (
        "attestation_file_sha256",
        "attestation_row_sha256",
        "key_id_sha256",
    ):
        require_digest(binding.get(field), f"source-symbol attestation {field}")
    if binding.get("complete") is not True or binding.get(
        "raw_names_present"
    ) is not False:
        raise MultiFunctionBuildError(
            "source-symbol attestation binding is incomplete or exposes names"
        )
    if (
        not isinstance(binding.get("function_symbol_count"), int)
        or isinstance(binding.get("function_symbol_count"), bool)
        or int(binding["function_symbol_count"]) != len(function_aliases)
        or not isinstance(binding.get("type_symbol_count"), int)
        or isinstance(binding.get("type_symbol_count"), bool)
        or int(binding["type_symbol_count"]) != len(type_aliases)
    ):
        raise MultiFunctionBuildError(
            "source-symbol attestation alias/count binding mismatch"
        )
    binding_sha = extractor_module.canonical_sha256(binding)
    if source_projection.get("symbol_attestation_binding_sha256") != binding_sha:
        raise MultiFunctionBuildError(
            "source-symbol attestation projection binding mismatch"
        )
    return {
        "used": True,
        "is_keyed": True,
        "binding": dict(binding),
        "binding_sha256": binding_sha,
        "type_aliases": type_aliases,
        "function_attestation_aliases": function_aliases,
        "raw_names_serialized": False,
    }


def _validate_external_symbols(
    values: Any,
    *,
    type_attestation_aliases: Sequence[str],
    function_attestation_aliases: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise MultiFunctionBuildError("external_symbols is not an array")
    result: list[dict[str, Any]] = []
    known_type_aliases = set(type_attestation_aliases)
    known_function_aliases = set(function_attestation_aliases)
    for expected_index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise MultiFunctionBuildError("external symbol row is not an object")
        external_id = str(raw.get("external_id") or "")
        raw_symbol = raw.get("symbol")
        symbol = str(raw_symbol) if raw_symbol is not None else None
        symbol_class = str(raw.get("symbol_class") or "")
        match = EXTERNAL_ID_RE.fullmatch(external_id)
        if match is None or int(match.group(1)) != expected_index:
            raise MultiFunctionBuildError(
                "external IDs must be ordered contiguous X0..Xn"
            )
        if symbol_class not in {
            "trusted_runtime",
            "neutralized_untrusted_runtime",
        }:
            raise MultiFunctionBuildError(
                f"external alias {external_id} has an invalid symbol class"
            )
        if symbol_class == "trusted_runtime" and not symbol:
            raise MultiFunctionBuildError(
                f"trusted external alias {external_id} has no label"
            )
        if symbol_class == "neutralized_untrusted_runtime" and symbol is not None:
            raise MultiFunctionBuildError(
                f"neutralized external alias {external_id} leaked a label"
            )
        lowered = (symbol or "").lower()
        if (
            (symbol is not None and ("\n" in symbol or "\r" in symbol))
            or "file://" in lowered
            or ".dart" in lowered
        ):
            raise MultiFunctionBuildError(
                f"external alias {external_id} contains a source identity"
            )
        for reference in VISIBLE_TYPE_ATTESTATION_REF_RE.finditer(symbol or ""):
            if f"T{int(reference.group(1))}" not in known_type_aliases:
                raise MultiFunctionBuildError(
                    f"external alias {external_id} references an unattested type"
                )
        for reference in VISIBLE_FUNCTION_ATTESTATION_REF_RE.finditer(
            symbol or ""
        ):
            if f"AF{int(reference.group(1))}" not in known_function_aliases:
                raise MultiFunctionBuildError(
                    f"external alias {external_id} references an unattested "
                    "function"
                )
        if VISIBLE_FUNCTION_ATTESTATION_REF_RE.search(symbol or "") and (
            re.fullmatch(r"@AF[0-9]+", symbol or "") is None
        ):
            raise MultiFunctionBuildError(
                f"external alias {external_id} has an invalid attested "
                "function reference"
            )
        if VISIBLE_TYPE_ATTESTATION_REF_RE.search(symbol or ""):
            is_type_assertion = (
                lowered.startswith("assert type is ")
                or " type is " in lowered
                or lowered.startswith("type test ")
            )
            if not is_type_assertion:
                raise MultiFunctionBuildError(
                    f"external alias {external_id} has an attested type alias "
                    "outside a type assertion"
                )
        result.append(
            {
                "external_id": external_id,
                "symbol": symbol,
                "symbol_class": symbol_class,
            }
        )
    return result


def _rewrite_instruction(
    instruction: str,
    *,
    function_count: int,
    local_start_to_global_block: Mapping[int, int],
    external_count: int,
) -> str:
    def replace_function(match: re.Match[str]) -> str:
        return _function_alias(
            f"F{int(match.group(1))}", function_count
        ) + (match.group(2) or "")

    def replace_local(match: re.Match[str]) -> str:
        raw = match.group(1)
        offset = int(raw, 16 if raw.lower().startswith("0x") else 10)
        if offset not in local_start_to_global_block:
            raise MultiFunctionBuildError(
                f"local branch offset 0x{offset:x} is not a block entry"
            )
        return f"@B{local_start_to_global_block[offset]}"

    value = FUNCTION_REF_RE.sub(replace_function, instruction)
    value = LOCAL_REF_RE.sub(replace_local, value)
    value = ANNOTATED_ALIAS_RE.sub(lambda match: match.group(1), value)
    value = REMAINING_ALIAS_ANNOTATION_RE.sub(
        lambda match: "[" + match.group(1) + "]", value
    )
    if FUNCTION_REF_RE.search(value) or LOCAL_REF_RE.search(value):
        raise MultiFunctionBuildError(
            f"unresolved function/local alias in {instruction!r}"
        )
    for match in EXTERNAL_REF_RE.finditer(value):
        if int(match.group(1)) >= external_count:
            raise MultiFunctionBuildError(
                f"instruction references unknown external alias {match.group(0)}"
            )
    if any(character in value for character in "<>"):
        # v1 uses <...> as atom framing.  Passing raw angle annotations through
        # would be ambiguous and could silently change the token stream.
        raise MultiFunctionBuildError(
            f"v1-incompatible angle annotation remains in {value!r}"
        )
    if any(character in value for character in "\r\n\t{}|"):
        raise MultiFunctionBuildError(
            f"instruction contains a compact/F2 delimiter: {value!r}"
        )
    return value


def combine_user_function_bundle(
    bundle: Mapping[str, Any],
    extractor_module: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one complete extractor bundle to a v1 canonical graph."""

    if bundle.get("schema") != EXTRACTOR_SCHEMA:
        raise MultiFunctionBuildError("user-function bundle schema mismatch")
    if bundle.get("architecture") != "x86_64":
        raise MultiFunctionBuildError("only x86_64 extractor bundles are supported")
    if bundle.get("root_function_id") != "F0":
        raise MultiFunctionBuildError("extractor root must be F0")
    accounting = bundle.get("accounting")
    if not isinstance(accounting, dict):
        raise MultiFunctionBuildError("extractor accounting is missing")
    zero_fields = (
        "excluded_user_function_count",
        "excluded_user_instruction_count",
        "excluded_user_call_site_count",
    )
    if any(int(accounting.get(field, -1)) != 0 for field in zero_fields):
        raise MultiFunctionBuildError("extractor zero-exclusion gate failed")
    gdb_function_count = int(accounting.get("gdb_file_function_count", -1))
    recursive_function_count = int(
        accounting.get("attested_recursive_function_count", -1)
    )
    selected_function_count = int(
        accounting.get("selected_function_count", -1)
    )
    disassembled_function_count = int(
        accounting.get("successfully_disassembled_function_count", -1)
    )
    if (
        min(
            gdb_function_count,
            recursive_function_count,
            selected_function_count,
            disassembled_function_count,
        )
        < 0
        or selected_function_count
        != gdb_function_count + recursive_function_count
        or selected_function_count != disassembled_function_count
        or int(accounting.get("user_function_count", -1))
        != int(accounting.get("emitted_function_count", -2))
        or selected_function_count
        != int(accounting.get("user_function_count", -2))
        + int(accounting.get("producer_scaffold_function_count", -3))
        or int(accounting.get("raw_user_instruction_count", -1))
        != int(accounting.get("emitted_instruction_count", -2))
        or int(accounting.get("call_site_count", -1))
        != int(accounting.get("represented_call_site_count", -2))
    ):
        raise MultiFunctionBuildError("extractor accounting equality failed")
    lossless = bundle.get("lossless_contract")
    if (
        not isinstance(lossless, dict)
        or lossless.get("domain") != EXTRACTOR_MODEL_SCHEMA
        or lossless.get("all_same_gdb_file_functions_disassembled") is not True
        or lossless.get("all_attested_user_constructor_targets_disassembled")
        is not True
        or lossless.get("recursive_disassembly_uses_direct_operand_address")
        is not True
        or lossless.get("recursive_entry_must_equal_direct_operand_address")
        is not True
        or lossless.get("type_assertion_names_replaced_by_attested_aliases")
        is not True
        or lossless.get("indirect_calls_preserved_as_unresolved_dynamic_dispatch")
        is not True
        or lossless.get("producer_owned_empty_main_scaffold_excluded") is not True
        or lossless.get("all_user_functions_required") is not True
        or lossless.get("all_user_machine_instructions_required") is not True
        or lossless.get("all_user_call_sites_required") is not True
        or lossless.get("unreachable_blocks_retained") is not True
        or lossless.get("truncation_allowed") is not False
    ):
        raise MultiFunctionBuildError("extractor lossless contract failed")
    source_only_contract = bundle.get("source_only_producer_contract")
    if (
        not isinstance(source_only_contract, dict)
        or source_only_contract.get("contract")
        != extractor_module.SCAFFOLD_CONTRACT
        or source_only_contract.get("build_input_schema")
        != extractor_module.SOURCE_ONLY_BUILD_INPUT_SCHEMA
        or source_only_contract.get("aot_row_schema")
        != extractor_module.SOURCE_ONLY_AOT_ROW_SCHEMA
    ):
        raise MultiFunctionBuildError(
            "source-only producer scaffold contract mismatch"
        )
    for field in (
        "analysis_program_sha256",
        "function_source_sha256",
        "producer_script_sha256",
    ):
        require_digest(
            source_only_contract.get(field), f"source-only contract {field}"
        )
    runtime_policy_sha = require_digest(
        bundle.get("runtime_symbol_policy_sha256"),
        "runtime symbol policy SHA-256",
    )
    if (
        extractor_module.canonical_sha256(bundle.get("runtime_symbol_policy"))
        != runtime_policy_sha
        or lossless.get("runtime_symbol_policy_sha256") != runtime_policy_sha
    ):
        raise MultiFunctionBuildError("runtime symbol policy binding mismatch")

    source_projection = extractor_module.model_projection(bundle)
    source_projection_sha = extractor_module.canonical_sha256(source_projection)
    if source_projection.get("schema") != EXTRACTOR_MODEL_SCHEMA:
        raise MultiFunctionBuildError("extractor model projection schema mismatch")
    if source_projection_sha != bundle.get("model_projection_sha256"):
        raise MultiFunctionBuildError(
            "extractor model projection SHA-256 mismatch"
        )
    attestation = _validate_source_symbol_attestation(
        bundle, source_projection, extractor_module
    )
    transfer_proof = prove_transfer_table_redundant(bundle)
    external_symbols = _validate_external_symbols(
        source_projection.get("external_symbols"),
        type_attestation_aliases=attestation["type_aliases"],
        function_attestation_aliases=attestation[
            "function_attestation_aliases"
        ],
    )
    trusted_external_count = sum(
        record["symbol_class"] == "trusted_runtime"
        for record in external_symbols
    )
    neutralized_external_count = sum(
        record["symbol_class"] == "neutralized_untrusted_runtime"
        for record in external_symbols
    )
    attested_type_assertion_count = sum(
        record["symbol_class"] == "trusted_runtime"
        and isinstance(record.get("symbol"), str)
        and VISIBLE_TYPE_ATTESTATION_REF_RE.search(record["symbol"])
        is not None
        and "type is" in record["symbol"]
        for record in external_symbols
    )
    if (
        trusted_external_count + neutralized_external_count
        != len(external_symbols)
        or int(
            accounting.get("trusted_runtime_external_symbol_count", -1)
        )
        != trusted_external_count
        or int(accounting.get("neutralized_external_symbol_count", -1))
        != neutralized_external_count
        or int(accounting.get("attested_type_assertion_count", -1))
        != attested_type_assertion_count
    ):
        raise MultiFunctionBuildError(
            "extractor external-symbol class accounting failed"
        )

    projection_functions = source_projection.get("functions")
    raw_functions = bundle.get("functions")
    if (
        not isinstance(projection_functions, list)
        or not isinstance(raw_functions, list)
        or not projection_functions
        or len(projection_functions) != len(raw_functions)
    ):
        raise MultiFunctionBuildError("extractor functions are missing/misaligned")
    function_count = len(projection_functions)
    if function_count != int(accounting["emitted_function_count"]):
        raise MultiFunctionBuildError(
            "extractor function-array/accounting mismatch"
        )
    expected_function_ids = [f"F{index}" for index in range(function_count)]
    if [
        str(function.get("function_id") or "")
        for function in projection_functions
    ] != expected_function_ids:
        raise MultiFunctionBuildError(
            "extractor functions must be ordered contiguous F0..Fn"
        )

    global_blocks: list[dict[str, Any]] = []
    global_edges: list[dict[str, Any]] = []
    root_entries: list[int] = []
    function_summaries: list[dict[str, Any]] = []
    global_offset = 0
    represented_machine_instructions = 0
    for function_index, (function, raw_function) in enumerate(
        zip(projection_functions, raw_functions)
    ):
        function_id = expected_function_ids[function_index]
        if (
            str(raw_function.get("function_id") or "") != function_id
            or str(function.get("function_id") or "") != function_id
        ):
            raise MultiFunctionBuildError("raw/projection function order drift")
        blocks = function.get("blocks")
        cfg_rows = raw_function.get("cfg")
        if (
            not isinstance(blocks, list)
            or not blocks
            or not isinstance(cfg_rows, list)
            or len(blocks) != len(cfg_rows)
        ):
            raise MultiFunctionBuildError(f"{function_id}: invalid blocks")
        if [int(block.get("id", -1)) for block in blocks] != list(
            range(len(blocks))
        ):
            raise MultiFunctionBuildError(
                f"{function_id}: local block IDs are not contiguous"
            )
        if [int(row.get("id", -1)) for row in cfg_rows] != list(
            range(len(blocks))
        ):
            raise MultiFunctionBuildError(
                f"{function_id}: raw CFG IDs are not contiguous"
            )
        start_to_global: dict[int, int] = {}
        for local_id, cfg_row in enumerate(cfg_rows):
            start = int(cfg_row.get("start_offset", -1))
            if start < 0 or start in start_to_global:
                raise MultiFunctionBuildError(
                    f"{function_id}: invalid/duplicate block start offset"
                )
            start_to_global[start] = global_offset + local_id

        entries = [int(value) for value in function.get("entry_blocks") or []]
        if not entries or len(entries) != len(set(entries)):
            raise MultiFunctionBuildError(
                f"{function_id}: invalid entry block list"
            )
        if any(not 0 <= value < len(blocks) for value in entries):
            raise MultiFunctionBuildError(
                f"{function_id}: entry block is out of range"
            )
        marker = (
            "fn @SELF" if function_index == 0 else f"fn @U{function_index - 1}"
        )
        for local_id, block in enumerate(blocks):
            instructions = [
                _rewrite_instruction(
                    str(instruction),
                    function_count=function_count,
                    local_start_to_global_block=start_to_global,
                    external_count=len(external_symbols),
                )
                for instruction in (block.get("instructions") or [])
            ]
            represented_machine_instructions += len(instructions)
            if local_id in entries:
                instructions.insert(0, marker)
            global_blocks.append(
                {
                    "id": global_offset + local_id,
                    "instructions": instructions,
                }
            )
        edges = function.get("cfg_edges")
        if not isinstance(edges, list):
            raise MultiFunctionBuildError(f"{function_id}: cfg_edges is invalid")
        for edge in edges:
            source = int(edge["source"])
            target = int(edge["target"])
            if (
                not 0 <= source < len(blocks)
                or not 0 <= target < len(blocks)
            ):
                raise MultiFunctionBuildError(
                    f"{function_id}: CFG edge is out of range"
                )
            global_edges.append(
                {
                    "source": global_offset + source,
                    "target": global_offset + target,
                    "edge_type": str(edge["edge_type"]),
                }
            )
        global_entries = [global_offset + value for value in entries]
        if function_index == 0:
            root_entries = global_entries
        function_summaries.append(
            {
                "source_function_id": function_id,
                "model_alias": marker.split(" ", 1)[1],
                "global_entry_blocks": global_entries,
                "global_block_start": global_offset,
                "block_count": len(blocks),
                "machine_instruction_count": sum(
                    len(block.get("instructions") or []) for block in blocks
                ),
            }
        )
        global_offset += len(blocks)

    if represented_machine_instructions != int(
        accounting["emitted_instruction_count"]
    ):
        raise MultiFunctionBuildError(
            "combined graph machine-instruction accounting mismatch"
        )
    if len(global_blocks) > 4096:
        raise MultiFunctionBuildError(
            "combined graph exceeds the pinned v1 block vocabulary"
        )
    canonical = {
        "architecture": "x86_64",
        "entry_blocks": root_entries,
        "blocks": global_blocks,
        "cfg_edges": global_edges,
    }
    semantic_projection = {
        "schema": ADAPTER_SCHEMA,
        "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
        "architecture": "x86_64",
        "root_alias": "@SELF",
        "functions": function_summaries,
        "external_symbols": external_symbols,
        "source_symbol_attestation": attestation,
        "runtime_symbol_policy_sha256": runtime_policy_sha,
        "canonical_graph": canonical,
        "source_model_projection_sha256": source_projection_sha,
        "source_only_producer_contract_sha256": stable_sha256(
            source_only_contract
        ),
        "transfer_semantics": {
            **transfer_proof,
            "byte_coordinates_omitted_after_one_to_one_proof": True,
            "call_operands_function_boundaries_and_target_offsets_retained": True,
        },
    }
    return canonical, semantic_projection


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return [int(value) for value in ids]


def _tokenizer_decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return str(tokenizer.decode(list(ids), skip_special_tokens=False))


def _instruction_counts(
    canonicals: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for canonical in canonicals:
        blocks = canonical.get("blocks")
        if not isinstance(blocks, list):
            raise MultiFunctionBuildError(
                "canonical graph has no block array during codebook fit"
            )
        for block in blocks:
            instructions = block.get("instructions")
            if not isinstance(instructions, list):
                raise MultiFunctionBuildError(
                    "canonical block has no instruction array during "
                    "codebook fit"
                )
            counts.update(str(value) for value in instructions)
    if not counts:
        raise MultiFunctionBuildError("training graphs contain no instructions")
    return counts


def build_train_only_stable_codebook(
    *,
    parent_codebook: Mapping[str, Any],
    parent_contract: Mapping[str, Any],
    train_canonicals: Sequence[Mapping[str, Any]],
    train_task_ids: Sequence[str],
    tokenizer: Any,
    tokenizer_sha256: str,
    parent_codebook_sha256: str,
    parent_contract_sha256: str,
    inline_cfg_codec_sha256: str,
    function_bundles_sha256: str,
    builder_script_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit instruction slots without consulting held-out graphs.

    The complete source-token ID set is inherited.  An old instruction that
    occurs in train stays at its old index.  Train instructions absent from the
    parent dictionary occupy the lowest free old-only indices in descending
    frequency and lexical tie-break order.  Remaining old-only entries stay in
    place, maximizing exact warm-start overlay reuse.
    """

    parent_expansions = [
        str(value) for value in parent_codebook.get("expansions") or []
    ]
    if not parent_expansions:
        raise MultiFunctionBuildError("parent codebook has no expansions")
    if len(parent_expansions) != len(set(parent_expansions)):
        raise MultiFunctionBuildError(
            "parent instruction codebook contains duplicate expansions"
        )
    counts = _instruction_counts(train_canonicals)
    if len(counts) > len(parent_expansions):
        raise MultiFunctionBuildError(
            "train-only instruction vocabulary exceeds the fixed overlay "
            f"capacity: {len(counts)} > {len(parent_expansions)}"
        )

    parent_index = {
        instruction: index
        for index, instruction in enumerate(parent_expansions)
    }
    retained = set(counts).intersection(parent_index)
    train_new = sorted(
        set(counts).difference(parent_index),
        key=lambda instruction: (-counts[instruction], instruction),
    )
    free_indices = [
        index
        for index, instruction in enumerate(parent_expansions)
        if instruction not in retained
    ]
    if len(train_new) > len(free_indices):
        raise MultiFunctionBuildError(
            "fixed instruction overlay has insufficient refit slots"
        )
    expansions = list(parent_expansions)
    for index, instruction in zip(free_indices, train_new):
        expansions[index] = instruction
    if not set(counts).issubset(expansions):
        raise MultiFunctionBuildError(
            "train-only refit failed to cover every train instruction"
        )
    changed_indices = [
        index
        for index, (old, new) in enumerate(
            zip(parent_expansions, expansions)
        )
        if old != new
    ]

    source_atom_ids = {
        str(key): int(value)
        for key, value in (
            parent_codebook.get("source_atom_ids") or {}
        ).items()
    }
    parent_source_expansions = {
        str(key): [int(value) for value in values]
        for key, values in (
            parent_codebook.get("source_token_expansions") or {}
        ).items()
    }
    if parent_source_expansions != {
        str(key): [int(value) for value in values]
        for key, values in (
            parent_contract.get("source_token_expansions") or {}
        ).items()
    }:
        raise MultiFunctionBuildError(
            "parent contract/codebook source expansions differ"
        )
    expected_source_ids = {
        int(value) for value in parent_contract.get("source_token_ids") or []
    }
    if expected_source_ids != {
        int(value) for value in source_atom_ids.values()
    }:
        raise MultiFunctionBuildError(
            "parent codebook atom IDs differ from contract source IDs"
        )
    if set(parent_source_expansions) != {
        str(value) for value in expected_source_ids
    }:
        raise MultiFunctionBuildError(
            "parent source expansion keys differ from source IDs"
        )
    base_vocab_size = int(parent_contract.get("base_vocab_size", -1))
    if base_vocab_size <= 0:
        raise MultiFunctionBuildError(
            "parent contract has no positive base vocabulary size"
        )
    instruction_token_ids: list[int] = []
    for index, old_instruction in enumerate(parent_expansions):
        atom = f"<I{index}>"
        if atom not in source_atom_ids:
            raise MultiFunctionBuildError(
                f"parent codebook lacks instruction atom {atom}"
            )
        source_id = source_atom_ids[atom]
        instruction_token_ids.append(source_id)
        expected_old = _tokenizer_encode(tokenizer, old_instruction)
        if not expected_old or any(
            value < 0 or value >= base_vocab_size for value in expected_old
        ):
            raise MultiFunctionBuildError(
                f"parent instruction {index} has an invalid base expansion"
            )
        if parent_source_expansions[str(source_id)] != expected_old:
            raise MultiFunctionBuildError(
                f"parent instruction {index} expansion is not tokenizer-exact"
            )
    if len(instruction_token_ids) != len(set(instruction_token_ids)):
        raise MultiFunctionBuildError(
            "parent instruction atoms reuse a source-token ID"
        )

    source_token_expansions = dict(parent_source_expansions)
    for index in changed_indices:
        source_id = source_atom_ids[f"<I{index}>"]
        encoded = _tokenizer_encode(tokenizer, expansions[index])
        if not encoded or any(
            value < 0 or value >= base_vocab_size for value in encoded
        ):
            raise MultiFunctionBuildError(
                f"refit instruction {index} has an invalid base expansion"
            )
        source_token_expansions[str(source_id)] = encoded

    codebook = dict(parent_codebook)
    for stale in ("fit_public_sha256", "fit_retained", "fit_quarantined"):
        codebook.pop(stale, None)
    codebook.update(
        {
            "schema": OUTPUT_CODEBOOK_SCHEMA,
            "parent_codebook_sha256": parent_codebook_sha256,
            "parent_contract_sha256": parent_contract_sha256,
            "codec_sha256": inline_cfg_codec_sha256,
            "builder_script_sha256": builder_script_sha256,
            "fit_function_bundles_sha256": function_bundles_sha256,
            "fit_task_set_sha256": stable_sha256(list(train_task_ids)),
            "fit_rows": len(train_task_ids),
            "heldout_rows_used_for_fit": 0,
            "fit_instruction_occurrences": sum(counts.values()),
            "fit_unique_instructions": len(counts),
            "codebook_size": len(expansions),
            "expansions": expansions,
            "source_token_expansions": source_token_expansions,
            "tokenizer_json_sha256": tokenizer_sha256,
            "refit": {
                "algorithm": CODEBOOK_REFIT_ALGORITHM,
                "new_instruction_order": (
                    "descending_train_frequency_then_unicode_lexical"
                ),
                "free_slot_order": "ascending_parent_index",
                "train_existing_slots_retained": len(retained),
                "train_new_slots_assigned": len(train_new),
                "changed_instruction_slots": len(changed_indices),
                "unchanged_instruction_slots": (
                    len(expansions) - len(changed_indices)
                ),
                "unused_capacity_after_complete_train_coverage": (
                    len(expansions) - len(counts)
                ),
                "source_token_id_set_preserved": True,
                "block_and_control_token_ids_preserved": True,
                "dev_graphs_or_targets_consulted": False,
            },
        }
    )
    if len(codebook["expansions"]) != int(
        parent_codebook.get("codebook_size", -1)
    ):
        raise MultiFunctionBuildError(
            "refit changed the fixed instruction codebook capacity"
        )
    if codebook.get("source_atom_ids") != parent_codebook.get(
        "source_atom_ids"
    ):
        raise MultiFunctionBuildError(
            "refit changed the source atom ID mapping"
        )

    stats = {
        "algorithm": CODEBOOK_REFIT_ALGORITHM,
        "train_rows": len(train_task_ids),
        "heldout_rows_used_for_fit": 0,
        "instruction_occurrences": sum(counts.values()),
        "unique_instructions": len(counts),
        "capacity": len(expansions),
        "train_existing_slots_retained": len(retained),
        "train_new_slots_assigned": len(train_new),
        "changed_instruction_slots": len(changed_indices),
        "unchanged_instruction_slots": (
            len(expansions) - len(changed_indices)
        ),
        "changed_instruction_source_token_ids_sha256": stable_sha256(
            [instruction_token_ids[index] for index in changed_indices]
        ),
    }
    return codebook, stats


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    return ordered[round((len(ordered) - 1) * quantile)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--expected-base-train-sha256", required=True)
    parser.add_argument("--base-train-seal", required=True, type=Path)
    parser.add_argument("--expected-base-train-seal-sha256", required=True)
    parser.add_argument("--base-dev", required=True, type=Path)
    parser.add_argument("--expected-base-dev-sha256", required=True)
    parser.add_argument("--base-dev-seal", required=True, type=Path)
    parser.add_argument("--expected-base-dev-seal-sha256", required=True)
    parser.add_argument("--function-bundles", required=True, type=Path)
    parser.add_argument("--expected-function-bundles-sha256", required=True)
    parser.add_argument("--constants", required=True, type=Path)
    parser.add_argument("--expected-constants-sha256", required=True)
    parser.add_argument("--extractor-script", required=True, type=Path)
    parser.add_argument("--expected-extractor-script-sha256", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--expected-codebook-sha256", required=True)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--codec", required=True, type=Path)
    parser.add_argument("--expected-codec-sha256", required=True)
    parser.add_argument("--inline-cfg-codec", required=True, type=Path)
    parser.add_argument(
        "--expected-inline-cfg-codec-sha256", required=True
    )
    parser.add_argument("--frontier-f2", required=True, type=Path)
    parser.add_argument("--expected-frontier-f2-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-train-rows", type=int, default=EXPECTED_TRAIN_ROWS
    )
    parser.add_argument(
        "--expected-dev-rows", type=int, default=EXPECTED_DEV_ROWS
    )
    parser.add_argument(
        "--student-token-limit", type=int, default=STUDENT_TOKEN_LIMIT
    )
    parser.add_argument(
        "--api-prompt-token-limit", type=int, default=API_PROMPT_TOKEN_LIMIT
    )
    parser.add_argument(
        "--chat-overhead-reserve", type=int, default=CHAT_OVERHEAD_RESERVE
    )
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    if (
        int(args.expected_train_rows) != EXPECTED_TRAIN_ROWS
        or int(args.expected_dev_rows) != EXPECTED_DEV_ROWS
    ):
        raise MultiFunctionBuildError(
            "production split sizes are fixed at train=1580/dev=175"
        )
    if int(args.student_token_limit) != STUDENT_TOKEN_LIMIT:
        raise MultiFunctionBuildError("student token limit must equal 9000")
    if int(args.api_prompt_token_limit) != API_PROMPT_TOKEN_LIMIT:
        raise MultiFunctionBuildError("API prompt token limit must equal 12000")
    if int(args.chat_overhead_reserve) != CHAT_OVERHEAD_RESERVE:
        raise MultiFunctionBuildError("chat overhead reserve must equal 256")

    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "base_train": args.base_train,
            "base_train_seal": args.base_train_seal,
            "base_dev": args.base_dev,
            "base_dev_seal": args.base_dev_seal,
            "function_bundles": args.function_bundles,
            "constants": args.constants,
            "extractor_script": args.extractor_script,
            "contract": args.contract,
            "codebook": args.codebook,
            "tokenizer": args.tokenizer_json,
            "codec": args.codec,
            "inline_cfg_codec": args.inline_cfg_codec,
            "frontier_f2": args.frontier_f2,
        }.items()
    }
    expected_hashes = {
        "base_train": args.expected_base_train_sha256,
        "base_train_seal": args.expected_base_train_seal_sha256,
        "base_dev": args.expected_base_dev_sha256,
        "base_dev_seal": args.expected_base_dev_seal_sha256,
        "function_bundles": args.expected_function_bundles_sha256,
        "constants": args.expected_constants_sha256,
        "extractor_script": args.expected_extractor_script_sha256,
        "contract": args.expected_contract_sha256,
        "codebook": args.expected_codebook_sha256,
        "tokenizer": args.expected_tokenizer_sha256,
        "codec": args.expected_codec_sha256,
        "inline_cfg_codec": args.expected_inline_cfg_codec_sha256,
        "frontier_f2": args.expected_frontier_f2_sha256,
    }
    input_records = {
        name: require_file_hash(paths[name], expected, name)
        for name, expected in expected_hashes.items()
    }
    input_records["adapter_script"] = file_record(Path(__file__))
    contract = load_json(paths["contract"], "compact contract")
    codebook = load_json(paths["codebook"], "compact codebook")
    if contract.get("schema") != COMPACT_CONTRACT_SCHEMA:
        raise MultiFunctionBuildError("compact contract schema mismatch")
    if codebook.get("schema") != CODEBOOK_SCHEMA:
        raise MultiFunctionBuildError("compact codebook schema mismatch")
    parent_artifact_bindings = {
        "codec_sha256": input_records["codec"]["sha256"],
        "codebook_sha256": input_records["codebook"]["sha256"],
        "tokenizer_json_sha256": input_records["tokenizer"]["sha256"],
    }
    for field, observed in parent_artifact_bindings.items():
        if contract.get(field) != observed:
            raise MultiFunctionBuildError(f"contract {field} binding mismatch")
    if codebook.get("tokenizer_json_sha256") != parent_artifact_bindings[
        "tokenizer_json_sha256"
    ]:
        raise MultiFunctionBuildError("codebook/tokenizer binding mismatch")
    if int(contract.get("max_source_tokens", -1)) != STUDENT_TOKEN_LIMIT:
        raise MultiFunctionBuildError("compact contract is not the pinned 9K contract")
    if int(codebook.get("max_blocks", -1)) != 4096:
        raise MultiFunctionBuildError("pinned codebook max_blocks is not 4096")
    if codebook.get("source_token_expansions") != contract.get(
        "source_token_expansions"
    ):
        raise MultiFunctionBuildError("contract/codebook expansion mismatch")

    base_train_seal = validate_sanitized_base_seal(
        dataset_path=paths["base_train"],
        seal_path=paths["base_train_seal"],
        dataset_record=input_records["base_train"],
        contract_sha256=input_records["contract"]["sha256"],
        role="fit",
        expected_rows=EXPECTED_TRAIN_ROWS,
    )
    base_dev_seal = validate_sanitized_base_seal(
        dataset_path=paths["base_dev"],
        seal_path=paths["base_dev_seal"],
        dataset_record=input_records["base_dev"],
        contract_sha256=input_records["contract"]["sha256"],
        role="measure",
        expected_rows=EXPECTED_DEV_ROWS,
    )
    for field in (
        "sanitation_schema",
        "sanitizer_sha256",
        "evaluator_sha256",
        "completion_attestation_id",
        "dart_version",
        "stability_runs",
        "quarantine_sha256",
    ):
        if base_train_seal.get(field) != base_dev_seal.get(field):
            raise MultiFunctionBuildError(
                f"sanitized train/dev seal {field} mismatch"
            )

    extractor = import_pinned_module(
        paths["extractor_script"],
        input_records["extractor_script"]["sha256"],
        "user function extractor",
    )
    parent_codec = import_pinned_module(
        paths["codec"], input_records["codec"]["sha256"], "compact v1 codec"
    )
    codec = import_pinned_module(
        paths["inline_cfg_codec"],
        input_records["inline_cfg_codec"]["sha256"],
        "inline CFG v2 codec",
    )
    frontier_f2 = import_pinned_module(
        paths["frontier_f2"],
        input_records["frontier_f2"]["sha256"],
        "frontier f2",
    )
    if frontier_f2.F2_SCHEMA != F2_REPRESENTATION_SCHEMA:
        raise MultiFunctionBuildError("frontier F2 schema mismatch")
    system_prompt = str(frontier_f2.F2_SYSTEM_PROMPT)
    system_prompt_sha = sha256_text(system_prompt)

    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise MultiFunctionBuildError("tokenizers package is required") from exc
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    base_vocab_size = int(contract["base_vocab_size"])
    parent_atom_ids = {
        str(key): int(value)
        for key, value in (codebook.get("source_atom_ids") or {}).items()
    }
    if GRAPH_MARKER not in parent_atom_ids:
        raise MultiFunctionBuildError("pinned codebook has no v1 graph marker")
    parent_expansions = [
        str(value) for value in codebook.get("expansions") or []
    ]
    if not parent_expansions:
        raise MultiFunctionBuildError("pinned codebook has no expansions")
    parent_code = {
        value: index for index, value in enumerate(parent_expansions)
    }
    if "fn @SELF" in parent_code or any(
        re.fullmatch(r"fn @U[0-9]+", value) for value in parent_code
    ):
        raise MultiFunctionBuildError(
            "function markers unexpectedly collide with the frozen codebook"
        )
    if (
        tuple(getattr(codec, "FORMAT_SENTINEL", ()))
        != ("<G2C1>", "<CFG>")
    ):
        raise MultiFunctionBuildError(
            "inline CFG codec format sentinel mismatch"
        )
    # Importing the parent codec is itself a hash/interface check; encoding is
    # performed only by the explicitly distinct inline-CFG codec below.
    if not all(
        hasattr(parent_codec, name)
        for name in ("encode", "decode", "compact_ids")
    ):
        raise MultiFunctionBuildError("parent compact codec API mismatch")

    train_rows = load_jsonl(paths["base_train"], "base train")
    dev_rows = load_jsonl(paths["base_dev"], "base dev")
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise MultiFunctionBuildError(
            f"base train has {len(train_rows)} rows, expected 1580"
        )
    if len(dev_rows) != EXPECTED_DEV_ROWS:
        raise MultiFunctionBuildError(
            f"base dev has {len(dev_rows)} rows, expected 175"
        )
    split_rows = {"train": train_rows, "dev": dev_rows}
    all_base = train_rows + dev_rows
    task_ids = [str(row.get("task_id") or "") for row in all_base]
    if any(not task_id for task_id in task_ids):
        raise MultiFunctionBuildError("base row has no task_id")
    if len(set(task_ids)) != len(task_ids):
        raise MultiFunctionBuildError(
            "train/dev task IDs are not globally disjoint and unique"
        )
    train_task_ids = {str(row["task_id"]) for row in train_rows}
    execution_ineligible = set(
        base_train_seal["execution_ineligible_task_ids"]
    )
    if not execution_ineligible.issubset(train_task_ids):
        raise MultiFunctionBuildError(
            "sanitized fit seal execution-ineligible IDs are not a subset "
            "of the 1,580-row imitation train set"
        )

    bundle_rows = load_jsonl(paths["function_bundles"], "function bundles")
    bundles: dict[str, dict[str, Any]] = {}
    for bundle in bundle_rows:
        task_id = str(bundle.get("task_id") or "")
        if not task_id or task_id in bundles:
            raise MultiFunctionBuildError(
                f"invalid/duplicate function bundle task_id {task_id!r}"
            )
        producer = bundle.get("producer")
        if (
            not isinstance(producer, dict)
            or producer.get("script_sha256")
            != input_records["extractor_script"]["sha256"]
        ):
            raise MultiFunctionBuildError(
                f"{task_id}: extractor script binding mismatch"
            )
        inputs = bundle.get("inputs")
        if not isinstance(inputs, dict):
            raise MultiFunctionBuildError(f"{task_id}: binary inputs missing")
        for field in (
            "aot_sha256",
            "gdb_info_output_sha256",
            "gdb_file_section_sha256",
            "raw_disassembly_sha256",
        ):
            require_digest(inputs.get(field), f"{task_id} {field}")
        bundles[task_id] = bundle
    if set(bundles) != set(task_ids):
        raise MultiFunctionBuildError(
            "function-bundle task set differs from train+dev"
        )

    constants_rows = load_jsonl(paths["constants"], "binary constants")
    constants: dict[str, dict[str, Any]] = {}
    for record in constants_rows:
        task_id = str(record.get("task_id") or "")
        if not task_id or task_id in constants:
            raise MultiFunctionBuildError(
                f"invalid/duplicate constants task_id {task_id!r}"
            )
        if record.get("schema") != "dart-aot-attested-pool-constants-v1":
            raise MultiFunctionBuildError(
                f"{task_id}: binary constants are not from the attested "
                "all-function pool extractor"
            )
        if record.get("err") not in (None, ""):
            raise MultiFunctionBuildError(
                f"{task_id}: binary constant extraction is incomplete: "
                f"{record.get('err')}"
            )
        if not isinstance(record.get("strings"), list) or not isinstance(
            record.get("numbers"), list
        ):
            raise MultiFunctionBuildError(
                f"{task_id}: constants strings/numbers are not arrays"
            )
        if not all(
            isinstance(value, str) for value in record["strings"]
        ) or not all(isinstance(value, str) for value in record["numbers"]):
            raise MultiFunctionBuildError(
                f"{task_id}: binary constants contain a non-string value"
            )
        noff = record.get("noff")
        if isinstance(noff, bool) or not isinstance(noff, int) or noff < 0:
            raise MultiFunctionBuildError(
                f"{task_id}: invalid object-pool offset count"
            )
        require_digest(
            record.get("pool_offsets_sha256"),
            f"{task_id} pool-offset projection",
        )
        constant_accounting = record.get("accounting")
        if not isinstance(constant_accounting, dict):
            raise MultiFunctionBuildError(
                f"{task_id}: missing binary-constant accounting"
            )
        if constant_accounting.get("unreadable_entries") != 0:
            raise MultiFunctionBuildError(
                f"{task_id}: binary constants contain unreadable entries"
            )
        for field in (
            "supported_string_objects",
            "supported_number_objects",
            "inline_float32_entries",
            "inline_float64_entries",
            "inline_float32x4_entries",
            "tagged_sentinel_entries",
            "metadata_strings_rejected",
            "unsupported_or_immediate_entries",
        ):
            value = constant_accounting.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise MultiFunctionBuildError(
                    f"{task_id}: invalid binary-constant accounting {field}"
                )
        constants[task_id] = record
    if set(constants) != set(task_ids):
        raise MultiFunctionBuildError(
            "binary-constants task set differs from train+dev"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_paths = {
        "contract": output_dir / "multifunction_inline_cfg_v2_contract.json",
        "codebook": output_dir / "multifunction_inline_cfg_v2_codebook.json",
        "train": output_dir / "train_multifunction_binary.jsonl",
        "train_seal": output_dir / "train_multifunction_binary.seal.json",
        "train_f2": output_dir / "train_multifunction_binary_f2.jsonl",
        "train_f2_manifest": output_dir
        / "train_multifunction_binary_f2.jsonl.manifest.json",
        "dev": output_dir / "dev_multifunction_binary.jsonl",
        "dev_seal": output_dir / "dev_multifunction_binary.seal.json",
        "dev_f2": output_dir / "dev_multifunction_binary_f2.jsonl",
        "dev_f2_manifest": output_dir
        / "dev_multifunction_binary_f2.jsonl.manifest.json",
        "report": output_dir / "build_report.json",
    }
    if any(path.exists() for path in output_paths.values()):
        existing = [
            str(path) for path in output_paths.values() if path.exists()
        ]
        raise FileExistsError(
            "refusing to overwrite existing outputs: " + ", ".join(existing)
        )

    # Canonicalize train first, freeze the instruction dictionary from only
    # those graphs, and only then canonicalize held-out graphs.  This ordering
    # makes accidental dev influence on the refit structurally impossible.
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for base_row in train_rows:
        task_id = str(base_row["task_id"])
        prepared[task_id] = combine_user_function_bundle(
            bundles[task_id], extractor
        )
    train_ids_in_order = [str(row["task_id"]) for row in train_rows]
    derived_codebook, codebook_refit = (
        build_train_only_stable_codebook(
            parent_codebook=codebook,
            parent_contract=contract,
            train_canonicals=[
                prepared[task_id][0] for task_id in train_ids_in_order
            ],
            train_task_ids=train_ids_in_order,
            tokenizer=tokenizer,
            tokenizer_sha256=input_records["tokenizer"]["sha256"],
            parent_codebook_sha256=input_records["codebook"]["sha256"],
            parent_contract_sha256=input_records["contract"]["sha256"],
            inline_cfg_codec_sha256=input_records[
                "inline_cfg_codec"
            ]["sha256"],
            function_bundles_sha256=input_records[
                "function_bundles"
            ]["sha256"],
            builder_script_sha256=input_records[
                "adapter_script"
            ]["sha256"],
        )
    )
    derived_codebook_bytes = json_artifact_bytes(derived_codebook)
    derived_codebook_sha = sha256_bytes(derived_codebook_bytes)
    derived_contract = dict(contract)
    derived_contract.update(
        {
            "codec_sha256": input_records["inline_cfg_codec"]["sha256"],
            "codebook_sha256": derived_codebook_sha,
            "source_token_expansions": derived_codebook[
                "source_token_expansions"
            ],
        }
    )
    derived_contract_bytes = json_artifact_bytes(derived_contract)
    derived_contract_sha = sha256_bytes(derived_contract_bytes)
    artifact_bindings = {
        "codec_sha256": input_records["inline_cfg_codec"]["sha256"],
        "codebook_sha256": derived_codebook_sha,
        "tokenizer_json_sha256": input_records["tokenizer"]["sha256"],
    }
    if derived_contract["source_token_expansions"] != derived_codebook[
        "source_token_expansions"
    ]:
        raise MultiFunctionBuildError(
            "derived contract/codebook expansion mismatch"
        )
    if set(derived_contract.get("source_token_ids") or []) != {
        int(value)
        for value in derived_codebook["source_atom_ids"].values()
    }:
        raise MultiFunctionBuildError(
            "derived contract/codebook source-token ID mismatch"
        )
    atom_ids = {
        str(key): int(value)
        for key, value in derived_codebook["source_atom_ids"].items()
    }
    expansions = [
        str(value) for value in derived_codebook["expansions"]
    ]
    code = {
        value: index for index, value in enumerate(expansions)
    }
    for base_row in dev_rows:
        task_id = str(base_row["task_id"])
        prepared[task_id] = combine_user_function_bundle(
            bundles[task_id], extractor
        )

    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    f2_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    student_lengths: list[int] = []
    api_lengths: list[int] = []
    graph_lengths: list[int] = []
    prefix_lengths: list[int] = []
    function_counts: list[int] = []
    block_counts: list[int] = []
    instruction_counts: list[int] = []
    transfer_counts: list[int] = []
    aot_digests: list[str] = []
    semantic_projection_digests: list[str] = []
    source_projection_digests: list[str] = []
    attestation_binding_digests: list[str] = []
    attestation_binding_digests_by_split: dict[str, list[str]] = {
        "train": [],
        "dev": [],
    }
    attestation_file_digests: set[str] = set()
    attestation_key_ids: set[str] = set()
    attested_type_symbol_counts: list[int] = []
    attested_function_symbol_counts: list[int] = []
    maximum_student: tuple[str, int] = ("", -1)
    maximum_api: tuple[str, int] = ("", -1)

    for split, rows in split_rows.items():
        for split_row, base_row in enumerate(rows):
            task_id = str(base_row["task_id"])
            bundle = bundles[task_id]
            # The AOT pool predates the direct-compact 1,580/175 partition, so
            # both model roles legitimately carry the pool's original
            # ``split=train`` provenance.  Membership and role come only from
            # the hash-pinned, disjoint base train/dev task-ID sets.
            canonical, semantic_projection = prepared[task_id]
            external_symbols = semantic_projection["external_symbols"]
            source_attestation = semantic_projection[
                "source_symbol_attestation"
            ]
            attestation_binding = source_attestation["binding"]
            attestation_binding_sha = source_attestation["binding_sha256"]
            constant_record = constants[task_id]
            prefix_source = binary_enrichment_preamble(
                constant_record["strings"],
                constant_record["numbers"],
                external_symbols,
            )
            prefix_ids = _tokenizer_encode(tokenizer, prefix_source)
            if any(
                token_id < 0 or token_id >= base_vocab_size
                for token_id in prefix_ids
            ):
                raise MultiFunctionBuildError(
                    f"{task_id}: prefix uses a non-base token"
                )
            prefix_text = _tokenizer_decode(tokenizer, prefix_ids)
            if prefix_text != prefix_source:
                raise MultiFunctionBuildError(
                    f"{task_id}: prefix tokenizer byte roundtrip failed"
                )
            if _tokenizer_encode(tokenizer, prefix_text) != prefix_ids:
                raise MultiFunctionBuildError(
                    f"{task_id}: prefix token roundtrip failed"
                )
            if parse_external_dictionary_from_preamble(
                prefix_text
            ) != external_symbols:
                raise MultiFunctionBuildError(
                    f"{task_id}: external dictionary prefix roundtrip failed"
                )

            try:
                graph_text = codec.encode(canonical, code)
                decoded = codec.decode(graph_text, expansions)
                graph_ids = [
                    int(value)
                    for value in codec.compact_ids(
                        graph_text, tokenizer, atom_ids
                    )
                ]
                reencoded = codec.encode(decoded, code)
                reencoded_ids = [
                    int(value)
                    for value in codec.compact_ids(
                        reencoded, tokenizer, atom_ids
                    )
                ]
            except Exception as exc:
                raise MultiFunctionBuildError(
                    f"{task_id}: inline CFG v2 codec failed: {exc}"
                ) from exc
            if decoded != canonical or reencoded != graph_text:
                raise MultiFunctionBuildError(
                    f"{task_id}: inline CFG canonical/text roundtrip failed"
                )
            if reencoded_ids != graph_ids:
                raise MultiFunctionBuildError(
                    f"{task_id}: inline CFG token roundtrip failed"
                )
            if graph_ids.count(atom_ids[GRAPH_MARKER]) != 1:
                raise MultiFunctionBuildError(
                    f"{task_id}: graph marker count is not exactly one"
                )
            if graph_ids.count(atom_ids["<CFG>"]) != 1:
                raise MultiFunctionBuildError(
                    f"{task_id}: inline CFG sentinel count is not exactly one"
                )
            compact_ids = prefix_ids + graph_ids
            if len(compact_ids) > STUDENT_TOKEN_LIMIT:
                raise MultiFunctionBuildError(
                    f"{task_id}: complete student source has "
                    f"{len(compact_ids)} tokens, exceeds 9000"
                )

            try:
                f2_text = frontier_f2.serialize_f2(
                    prefix_text, canonical, tokenizer=tokenizer
                )
                decoded_prefix, decoded_f2 = frontier_f2.decode_f2(f2_text)
            except Exception as exc:
                raise MultiFunctionBuildError(
                    f"{task_id}: F2 serialization failed: {exc}"
                ) from exc
            if decoded_prefix != prefix_text or decoded_f2 != canonical:
                raise MultiFunctionBuildError(
                    f"{task_id}: F2 semantic roundtrip failed"
                )
            system_tokens = len(_tokenizer_encode(tokenizer, system_prompt))
            user_tokens = len(_tokenizer_encode(tokenizer, f2_text))
            api_tokens = (
                system_tokens + user_tokens + CHAT_OVERHEAD_RESERVE
            )
            if api_tokens > API_PROMPT_TOKEN_LIMIT:
                raise MultiFunctionBuildError(
                    f"{task_id}: complete F2 prompt has {api_tokens} tokens, "
                    "exceeds 12000"
                )

            parent_row_hashes = {
                "compact_codec_sha256": parent_artifact_bindings[
                    "codec_sha256"
                ],
                "compact_codebook_sha256": parent_artifact_bindings[
                    "codebook_sha256"
                ],
                "compact_tokenizer_sha256": parent_artifact_bindings[
                    "tokenizer_json_sha256"
                ],
            }
            output_row_hashes = {
                "compact_codec_sha256": artifact_bindings["codec_sha256"],
                "compact_codebook_sha256": artifact_bindings["codebook_sha256"],
                "compact_tokenizer_sha256": artifact_bindings[
                    "tokenizer_json_sha256"
                ],
            }
            for field, expected in parent_row_hashes.items():
                if str(base_row.get(field) or "") != expected:
                    raise MultiFunctionBuildError(
                        f"{task_id}: base row {field} binding mismatch"
                    )
            source_projection_sha = str(bundle["model_projection_sha256"])
            semantic_projection_sha = stable_sha256(semantic_projection)
            output = dict(base_row)
            output["compact_input_ids"] = compact_ids
            output.update(output_row_hashes)
            output.update(
                {
                    "binary_multifunction_schema": ADAPTER_SCHEMA,
                    "binary_adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
                    "binary_adapter_script_sha256": input_records[
                        "adapter_script"
                    ]["sha256"],
                    "binary_aot_sha256": bundle["inputs"]["aot_sha256"],
                    "binary_source_model_projection_sha256": source_projection_sha,
                    "binary_semantic_projection_sha256": semantic_projection_sha,
                    "binary_function_count": len(
                        semantic_projection["functions"]
                    ),
                    "binary_external_symbol_count": len(external_symbols),
                    "binary_transfer_count": semantic_projection[
                        "transfer_semantics"
                    ]["transfer_row_count"],
                    "binary_source_symbol_attestation_used": True,
                    "binary_source_symbol_attestation_is_keyed": True,
                    "binary_source_symbol_attestation_binding": (
                        attestation_binding
                    ),
                    "binary_source_symbol_attestation_binding_sha256": (
                        attestation_binding_sha
                    ),
                }
            )
            output_rows[split].append(output)

            constants_binding = {
                "constants_record": constant_record,
                "external_symbols": external_symbols,
            }
            f2_rows[split].append(
                {
                    "schema": F2_ROW_SCHEMA,
                    "representation_schema": F2_REPRESENTATION_SCHEMA,
                    "system_prompt_sha256": system_prompt_sha,
                    "task_id": task_id,
                    "text": f2_text,
                    "text_sha256": sha256_text(f2_text),
                    "compact_ids_sha256": stable_sha256(compact_ids),
                    "compact_text_sha256": sha256_text(graph_text),
                    "canonical_sha256": stable_sha256(canonical),
                    "constants_record_sha256": stable_sha256(
                        constants_binding
                    ),
                    "constants_extraction_error": None,
                    "constant_prefix_tokens": len(prefix_ids),
                    "graph_tokens": len(graph_ids),
                    "source_model_projection_sha256": source_projection_sha,
                    "semantic_projection_sha256": semantic_projection_sha,
                    "binary_aot_sha256": bundle["inputs"]["aot_sha256"],
                    "source_symbol_attestation_used": True,
                    "source_symbol_attestation_is_keyed": True,
                    "source_symbol_attestation_binding": attestation_binding,
                    "source_symbol_attestation_binding_sha256": (
                        attestation_binding_sha
                    ),
                    "prompt_preflight": {
                        "system_tokens": system_tokens,
                        "user_tokens": user_tokens,
                        "chat_overhead_reserve": CHAT_OVERHEAD_RESERVE,
                        "estimated_prompt_tokens": api_tokens,
                    },
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
                        "all_user_functions_retained": True,
                        "all_external_symbols_retained": True,
                        "transfer_table_redundancy_proven": True,
                        "keyed_source_symbol_attestation_bound": True,
                        "raw_source_names_not_serialized": True,
                    },
                }
            )

            student_lengths.append(len(compact_ids))
            api_lengths.append(api_tokens)
            graph_lengths.append(len(graph_ids))
            prefix_lengths.append(len(prefix_ids))
            function_counts.append(len(semantic_projection["functions"]))
            block_counts.append(len(canonical["blocks"]))
            instruction_counts.append(
                sum(
                    int(item["machine_instruction_count"])
                    for item in semantic_projection["functions"]
                )
            )
            transfer_counts.append(
                int(
                    semantic_projection["transfer_semantics"][
                        "transfer_row_count"
                    ]
                )
            )
            aot_digests.append(str(bundle["inputs"]["aot_sha256"]))
            semantic_projection_digests.append(semantic_projection_sha)
            source_projection_digests.append(source_projection_sha)
            attestation_binding_digests.append(attestation_binding_sha)
            attestation_binding_digests_by_split[split].append(
                attestation_binding_sha
            )
            attestation_file_digests.add(
                attestation_binding["attestation_file_sha256"]
            )
            attestation_key_ids.add(attestation_binding["key_id_sha256"])
            attested_function_symbol_counts.append(
                int(attestation_binding["function_symbol_count"])
            )
            attested_type_symbol_counts.append(
                int(attestation_binding["type_symbol_count"])
            )
            if len(compact_ids) > maximum_student[1]:
                maximum_student = (task_id, len(compact_ids))
            if api_tokens > maximum_api[1]:
                maximum_api = (task_id, api_tokens)

    if (
        len(output_rows["train"]) != EXPECTED_TRAIN_ROWS
        or len(output_rows["dev"]) != EXPECTED_DEV_ROWS
        or len(f2_rows["train"]) != EXPECTED_TRAIN_ROWS
        or len(f2_rows["dev"]) != EXPECTED_DEV_ROWS
    ):
        raise MultiFunctionBuildError("zero-exclusion split accounting failed")
    if len(attestation_file_digests) != 1 or len(attestation_key_ids) != 1:
        raise MultiFunctionBuildError(
            "function bundles do not share one attestation file/key binding"
        )
    attestation_file_sha = next(iter(attestation_file_digests))
    attestation_key_id = next(iter(attestation_key_ids))

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(output_paths["codebook"], derived_codebook_bytes)
    _atomic_write_bytes(output_paths["contract"], derived_contract_bytes)
    atomic_write_jsonl(output_paths["train"], output_rows["train"])
    atomic_write_jsonl(output_paths["dev"], output_rows["dev"])
    atomic_write_jsonl(output_paths["train_f2"], f2_rows["train"])
    atomic_write_jsonl(output_paths["dev_f2"], f2_rows["dev"])

    split_task_ids = {
        split: [str(row["task_id"]) for row in output_rows[split]]
        for split in ("train", "dev")
    }
    artifact_records = {
        key: input_records[key]
        for key in (
            "base_train",
            "base_train_seal",
            "base_dev",
            "base_dev_seal",
            "function_bundles",
            "constants",
            "extractor_script",
            "contract",
            "codebook",
            "tokenizer",
            "codec",
            "inline_cfg_codec",
            "frontier_f2",
        )
    }
    artifact_records.update(
        {
            "representation_contract": file_record(
                output_paths["contract"]
            ),
            "representation_codebook": file_record(
                output_paths["codebook"]
            ),
        }
    )
    for split in ("train", "dev"):
        f2_path = output_paths[f"{split}_f2"]
        max_row = max(
            f2_rows[split],
            key=lambda value: value["prompt_preflight"][
                "estimated_prompt_tokens"
            ],
        )
        f2_manifest = {
            "schema": F2_MANIFEST_SCHEMA,
            "created_at": utc_now(),
            "dataset": file_record(output_paths[split]),
            "task_set_sha256": stable_sha256(split_task_ids[split]),
            "rows": len(f2_rows[split]),
            "binary_constant_extraction_errors": {
                "count": 0,
                "task_ids": [],
            },
            "source_symbol_attestation": {
                "used": True,
                "is_keyed": True,
                "attestation_file_sha256": attestation_file_sha,
                "key_id_sha256": attestation_key_id,
                "binding_sha256_sequence": stable_sha256(
                    attestation_binding_digests_by_split[split]
                ),
                "raw_names_serialized": False,
            },
            "artifacts": artifact_records,
            "f2_prompt_contract": {
                "representation_schema": F2_REPRESENTATION_SCHEMA,
                "system_prompt": system_prompt,
                "system_prompt_sha256": system_prompt_sha,
                "tokenizer_sha256": artifact_bindings[
                    "tokenizer_json_sha256"
                ],
                "constant_prefix_token_cap": None,
                "max_prompt_tokens": API_PROMPT_TOKEN_LIMIT,
                "chat_overhead_reserve": CHAT_OVERHEAD_RESERVE,
                "maximum_estimated_prompt_tokens": max_row[
                    "prompt_preflight"
                ]["estimated_prompt_tokens"],
                "maximum_task_id": max_row["task_id"],
                "all_rows_within_limit": True,
            },
            "output": file_record(f2_path),
            "invariants": {
                "all_artifact_hashes_verified": True,
                "all_row_contract_hashes_verified": True,
                "all_codec_roundtrips_verified": True,
                "all_student_constant_prefixes_verified": True,
                "all_f2_semantic_roundtrips_verified": True,
                "f2_system_prompt_self_contained_and_hashed": True,
                "all_complete_prompts_within_limit": True,
                "opaque_source_ids_expanded": True,
                "cfg_explicit": True,
                "all_user_functions_retained": True,
                "all_external_symbols_retained": True,
                "transfer_table_redundancy_proven": True,
                "train_dev_representation_contract_identical": True,
                "keyed_private_source_symbol_attestation_used": True,
                "raw_source_names_not_serialized": True,
            },
        }
        atomic_write_json(output_paths[f"{split}_f2_manifest"], f2_manifest)

    contract_sha = derived_contract_sha
    for split, role in (("train", "fit"), ("dev", "measure")):
        source_seal = base_train_seal if split == "train" else base_dev_seal
        seal = {
            "schema": SPLIT_SEAL_SCHEMA,
            "selected_role": role,
            "training_allowed": split == "train",
            "heldout_measure_only": split == "dev",
            "rows": len(output_rows[split]),
            "task_set_sha256": stable_sha256(split_task_ids[split]),
            "output_sha256": sha256_file(output_paths[split]),
            "output": file_record(output_paths[split]),
            "f2_output": file_record(output_paths[f"{split}_f2"]),
            "f2_manifest": file_record(
                output_paths[f"{split}_f2_manifest"]
            ),
            "contract_sha256": contract_sha,
            "representation_schema": ADAPTER_SCHEMA,
            "frontier_f2_schema": F2_REPRESENTATION_SCHEMA,
            "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
            "adapter_script_sha256": input_records["adapter_script"]["sha256"],
            "source_function_bundles_sha256": input_records[
                "function_bundles"
            ]["sha256"],
            "source_symbol_attestation_used": True,
            "source_symbol_attestation_is_keyed": True,
            "source_symbol_attestation_file_sha256": attestation_file_sha,
            "source_symbol_attestation_key_id_sha256": attestation_key_id,
            "source_symbol_attestation_binding_sha256_sequence": stable_sha256(
                attestation_binding_digests_by_split[split]
            ),
            "raw_source_names_serialized": False,
            "sanitized_base_sha256": input_records[
                f"base_{split}"
            ]["sha256"],
            "sanitized_base_seal_sha256": input_records[
                f"base_{split}_seal"
            ]["sha256"],
            "sanitation_schema": source_seal["sanitation_schema"],
            "sanitizer_sha256": source_seal["sanitizer_sha256"],
            "evaluator_sha256": source_seal["evaluator_sha256"],
            "completion_attestation_id": source_seal[
                "completion_attestation_id"
            ],
            "dart_version": source_seal["dart_version"],
            "stability_runs": source_seal["stability_runs"],
            "quarantine_sha256": source_seal["quarantine_sha256"],
        }
        if split == "train":
            seal.update(
                {
                    "training_objective_scope": IMITATION_TRAIN_SCOPE,
                    "executable_reward_eligible_rows": source_seal[
                        "executable_reward_eligible_rows"
                    ],
                    "execution_ineligible_task_ids": source_seal[
                        "execution_ineligible_task_ids"
                    ],
                }
            )
        atomic_write_json(output_paths[f"{split}_seal"], seal)

    counts = {
        "rows": len(all_base),
        "train_rows": len(output_rows["train"]),
        "dev_rows": len(output_rows["dev"]),
        "excluded_rows": 0,
        "truncated_rows": 0,
        "constant_extraction_error_rows": 0,
        "functions": sum(function_counts),
        "blocks": sum(block_counts),
        "machine_instructions": sum(instruction_counts),
        "interfunction_transfers": sum(transfer_counts),
        "attested_function_symbols": sum(attested_function_symbol_counts),
        "attested_type_symbols": sum(attested_type_symbol_counts),
    }
    token_stats = {
        "student": {
            "limit": STUDENT_TOKEN_LIMIT,
            "min": min(student_lengths),
            "p50": _percentile(student_lengths, 0.50),
            "p95": _percentile(student_lengths, 0.95),
            "p99": _percentile(student_lengths, 0.99),
            "max": maximum_student[1],
            "max_task_id": maximum_student[0],
        },
        "api_f2_prompt": {
            "limit": API_PROMPT_TOKEN_LIMIT,
            "chat_overhead_reserve": CHAT_OVERHEAD_RESERVE,
            "min": min(api_lengths),
            "p50": _percentile(api_lengths, 0.50),
            "p95": _percentile(api_lengths, 0.95),
            "p99": _percentile(api_lengths, 0.99),
            "max": maximum_api[1],
            "max_task_id": maximum_api[0],
        },
        "graph": {
            "max": max(graph_lengths),
            "p95": _percentile(graph_lengths, 0.95),
        },
        "binary_prefix": {
            "max": max(prefix_lengths),
            "p95": _percentile(prefix_lengths, 0.95),
            "cap": None,
        },
    }
    report = {
        "schema": BUILD_SCHEMA,
        "created_at": utc_now(),
        "representation_schema": ADAPTER_SCHEMA,
        "adapter_contract": ADAPTER_CONTRACT,
        "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
        "inputs": input_records,
        "derived_representation": {
            "contract": file_record(output_paths["contract"]),
            "codebook": file_record(output_paths["codebook"]),
            "codec": input_records["inline_cfg_codec"],
            "parent_contract_sha256": input_records["contract"]["sha256"],
            "parent_codebook_sha256": input_records["codebook"]["sha256"],
            "parent_codec_sha256": input_records["codec"]["sha256"],
            "codebook_refit": codebook_refit,
        },
        "counts": counts,
        "tokens": token_stats,
        "digests": {
            "all_task_ids_sha256": stable_sha256(task_ids),
            "train_task_ids_sha256": stable_sha256(split_task_ids["train"]),
            "dev_task_ids_sha256": stable_sha256(split_task_ids["dev"]),
            "aot_digest_sequence_sha256": stable_sha256(aot_digests),
            "source_model_projection_digest_sequence_sha256": stable_sha256(
                source_projection_digests
            ),
            "semantic_projection_digest_sequence_sha256": stable_sha256(
                semantic_projection_digests
            ),
            "source_symbol_attestation_binding_digest_sequence_sha256": (
                stable_sha256(attestation_binding_digests)
            ),
            "f2_system_prompt_sha256": system_prompt_sha,
        },
        "source_symbol_attestation": {
            "used": True,
            "is_keyed": True,
            "attestation_file_sha256": attestation_file_sha,
            "key_id_sha256": attestation_key_id,
            "binding_rows": len(attestation_binding_digests),
            "binding_sha256_sequence": stable_sha256(
                attestation_binding_digests
            ),
            "raw_names_serialized": False,
        },
        "invariants": {
            "source_text_read_to_build_representation": False,
            "keyed_private_source_symbol_attestation_used_to_build_representation": True,
            "raw_source_or_user_names_serialized_in_model_inputs": False,
            "only_name_free_attestation_aliases_and_bindings_propagated": True,
            "base_target_rows_inspected_for_representation": False,
            "all_aot_and_extractor_hashes_bound": True,
            "all_user_functions_retained": True,
            "producer_scaffold_disassembled_accounted_and_attested": True,
            "all_machine_instructions_retained": True,
            "all_cfg_edges_retained_with_global_offsets": True,
            "all_global_user_call_aliases_retained": True,
            "all_external_aliases_and_exact_definitions_retained": True,
            "all_transfer_semantics_proven_redundant": True,
            "source_token_id_set_preserved_from_parent": True,
            "block_and_control_token_ids_preserved_from_parent": True,
            "instruction_codebook_refit_from_train_only": True,
            "heldout_rows_used_for_instruction_codebook_fit": 0,
            "warmstart_overlay_rows_reusable_only_when_expansions_match": True,
            "inline_cfg_source_is_current_containing_block": True,
            "inline_cfg_omits_only_redundant_edge_source_tokens": True,
            "all_inline_cfg_text_and_token_roundtrips_verified": True,
            "all_f2_semantic_roundtrips_verified": True,
            "all_student_rows_within_9000": True,
            "all_api_prompts_within_12000": True,
            "constant_prefix_has_no_item_or_token_cap": True,
            "zero_excluded_rows": True,
            "zero_truncated_rows": True,
            "train_dev_task_sets_disjoint": True,
            "dev_is_measure_only_and_not_training": True,
            "train_dev_representation_contract_identical": True,
        },
        "outputs": {
            key: file_record(path)
            for key, path in output_paths.items()
            if key != "report"
        },
        "passed": True,
    }
    atomic_write_json(output_paths["report"], report)
    report["outputs"]["report"] = file_record(output_paths["report"])
    print(
        "MULTIFUNCTION_BINARY_COMPACT_BUILD "
        f"train={counts['train_rows']} dev={counts['dev_rows']} "
        f"functions={counts['functions']} "
        f"student_max={token_stats['student']['max']} "
        f"api_max={token_stats['api_f2_prompt']['max']} "
        f"train_sha256={report['outputs']['train']['sha256']} "
        f"dev_sha256={report['outputs']['dev']['sha256']}",
        flush=True,
    )
    return report


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
