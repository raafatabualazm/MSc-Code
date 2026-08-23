#!/usr/bin/env python3
"""Build keyed, source-name-free attestations for Dart AOT user symbols.

This is a private preparation step. It reads the hash-pinned source-only build
inputs, but emits only ordered, domain-separated HMAC digests. The extraction
stage can therefore test whether a GDB annotation denotes a declared user
function/type without receiving or publishing any source name.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "dart-user-symbol-attestation-v1"
REPORT_SCHEMA = "dart-user-symbol-attestation-build-report-v1"
SOURCE_INPUT_SCHEMA = "dart-source-only-aot-build-input-v1"
AOT_ROW_SCHEMA = "phase0-s44-source-only-aot-row-v1"
DOMAIN = b"openai-codex:dart-user-symbol-attestation:v1\x00"
MINIMUM_KEY_BYTES = 32
SHA256_RE = __import__("re").compile(r"[0-9a-f]{64}\Z")
DIRECTIVE_START_RE = __import__("re").compile(
    r"(?m)^[ \t]*(?:import|export|part)\b"
)
DIRECTIVE_RE = __import__("re").compile(
    r"(?ms)^[ \t]*(?P<kind>import|export|part)\b(?P<body>.*?);"
)
QUOTED_URI_RE = __import__("re").compile(r"(['\"])(?P<uri>[^'\"]+)\1")


class SymbolAttestationError(ValueError):
    pass


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_symbol(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value))
    if (
        not normalized
        or normalized != normalized.strip()
        or "\x00" in normalized
        or "\r" in normalized
        or "\n" in normalized
    ):
        raise SymbolAttestationError("invalid_source_symbol")
    return normalized


def require_dart_only_imports(analysis_program: str, task_id: str) -> None:
    """Seal the corpus audit invariant: no package/file/part code is linked."""

    starts = list(DIRECTIVE_START_RE.finditer(analysis_program))
    directives = list(DIRECTIVE_RE.finditer(analysis_program))
    if len(starts) != len(directives):
        raise SymbolAttestationError(
            f"unparsed_library_directive:{task_id}"
        )
    for directive in directives:
        kind = directive.group("kind")
        uris = [
            match.group("uri")
            for match in QUOTED_URI_RE.finditer(
                directive.group("body")
            )
        ]
        if (
            kind != "import"
            or not uris
            or any(not uri.startswith("dart:") for uri in uris)
        ):
            raise SymbolAttestationError(
                f"non_dart_library_directive:{task_id}"
            )


def load_key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) < MINIMUM_KEY_BYTES:
        raise SymbolAttestationError(
            f"attestation_key_too_short:{len(key)}<{MINIMUM_KEY_BYTES}"
        )
    return key


def key_id_sha256(key: bytes) -> str:
    return hashlib.sha256(
        DOMAIN + b"key-id\x00" + key
    ).hexdigest()


def row_salt(
    key: bytes, *, task_id: str, analysis_program_sha256: str
) -> str:
    payload = (
        DOMAIN
        + b"row-salt\x00"
        + task_id.encode("utf-8")
        + b"\x00"
        + analysis_program_sha256.encode("ascii")
    )
    return hmac.new(key, payload, hashlib.sha256).hexdigest()[:32]


def symbol_digest(
    key: bytes,
    *,
    task_id: str,
    salt_hex: str,
    kind: str,
    index: int,
    symbol: str,
) -> str:
    if kind not in {"function", "type"}:
        raise SymbolAttestationError(f"invalid_symbol_kind:{kind}")
    if index < 0:
        raise SymbolAttestationError("negative_symbol_index")
    symbol = normalize_symbol(symbol)
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError as error:
        raise SymbolAttestationError("invalid_attestation_salt") from error
    if len(salt) != 16:
        raise SymbolAttestationError("attestation_salt_must_be_16_bytes")
    payload = (
        DOMAIN
        + b"symbol\x00"
        + task_id.encode("utf-8")
        + b"\x00"
        + salt
        + b"\x00"
        + kind.encode("ascii")
        + b"\x00"
        + str(index).encode("ascii")
        + b"\x00"
        + symbol.encode("utf-8")
    )
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def ordered_commitment(
    key: bytes,
    *,
    task_id: str,
    salt_hex: str,
    function_digests: Sequence[str],
    type_digests: Sequence[str],
) -> str:
    payload = (
        DOMAIN
        + b"ordered-completeness\x00"
        + task_id.encode("utf-8")
        + b"\x00"
        + bytes.fromhex(salt_hex)
        + b"\x00"
        + canonical_bytes(
            {
                "function_digests": list(function_digests),
                "type_digests": list(type_digests),
            }
        )
    )
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def build_attestation_row(
    *,
    build_row: Mapping[str, Any],
    aot_row: Mapping[str, Any],
    key: bytes,
) -> dict[str, Any]:
    task_id = str(build_row.get("task_id") or "")
    if not task_id or task_id != str(aot_row.get("task_id") or ""):
        raise SymbolAttestationError("task_id_alignment_mismatch")
    if build_row.get("schema") != SOURCE_INPUT_SCHEMA:
        raise SymbolAttestationError("source_input_schema_mismatch")
    if aot_row.get("schema") != AOT_ROW_SCHEMA:
        raise SymbolAttestationError("aot_row_schema_mismatch")
    if (
        build_row.get("split") != aot_row.get("split")
        or int(build_row.get("split_row", -1))
        != int(aot_row.get("split_row", -2))
    ):
        raise SymbolAttestationError(f"split_alignment_mismatch:{task_id}")
    analysis_sha = str(build_row.get("analysis_program_sha256") or "").lower()
    function_sha = str(build_row.get("function_source_sha256") or "").lower()
    if not SHA256_RE.fullmatch(analysis_sha) or not SHA256_RE.fullmatch(
        function_sha
    ):
        raise SymbolAttestationError(f"invalid_source_hash:{task_id}")
    if (
        analysis_sha != str(aot_row.get("analysis_program_sha256") or "").lower()
        or function_sha
        != str(aot_row.get("function_source_sha256") or "").lower()
    ):
        raise SymbolAttestationError(f"source_aot_hash_mismatch:{task_id}")
    analysis_program = str(build_row.get("analysis_program") or "")
    function_source = str(build_row.get("function_source") or "")
    if sha256_text(analysis_program) != analysis_sha:
        raise SymbolAttestationError(f"analysis_program_hash_mismatch:{task_id}")
    if sha256_text(function_source) != function_sha:
        raise SymbolAttestationError(f"function_source_hash_mismatch:{task_id}")
    if not analysis_program.rstrip().endswith("void main() {}"):
        raise SymbolAttestationError(f"source_only_main_contract_mismatch:{task_id}")
    require_dart_only_imports(analysis_program, task_id)
    producer = aot_row.get("producer")
    producer_sha = (
        str(producer.get("script_sha256") or "").lower()
        if isinstance(producer, Mapping)
        else ""
    )
    if not SHA256_RE.fullmatch(producer_sha):
        raise SymbolAttestationError(f"invalid_producer_hash:{task_id}")
    symbols = build_row.get("source_symbols")
    transform = build_row.get("transform_metadata")
    if not isinstance(symbols, Mapping) or not isinstance(transform, Mapping):
        raise SymbolAttestationError(f"missing_source_symbol_contract:{task_id}")
    if transform.get("source_symbols") != symbols:
        raise SymbolAttestationError(
            f"source_symbol_transform_mismatch:{task_id}"
        )
    functions_raw = symbols.get("functions")
    types_raw = symbols.get("types")
    if not isinstance(functions_raw, list) or not isinstance(types_raw, list):
        raise SymbolAttestationError(f"invalid_source_symbol_lists:{task_id}")
    functions = [normalize_symbol(value) for value in functions_raw]
    types = [normalize_symbol(value) for value in types_raw]
    if len(functions) != len(set(functions)):
        raise SymbolAttestationError(f"duplicate_function_symbol:{task_id}")
    if len(types) != len(set(types)):
        raise SymbolAttestationError(f"duplicate_type_symbol:{task_id}")
    if functions.count("candidate") != 1:
        raise SymbolAttestationError(f"candidate_symbol_count:{task_id}")

    salt = row_salt(
        key, task_id=task_id, analysis_program_sha256=analysis_sha
    )
    function_entries = [
        {
            "alias": f"AF{index}",
            "digest": symbol_digest(
                key,
                task_id=task_id,
                salt_hex=salt,
                kind="function",
                index=index,
                symbol=symbol,
            ),
        }
        for index, symbol in enumerate(functions)
    ]
    type_entries = [
        {
            "alias": f"T{index}",
            "digest": symbol_digest(
                key,
                task_id=task_id,
                salt_hex=salt,
                kind="type",
                index=index,
                symbol=symbol,
            ),
        }
        for index, symbol in enumerate(types)
    ]
    commitment = ordered_commitment(
        key,
        task_id=task_id,
        salt_hex=salt,
        function_digests=[entry["digest"] for entry in function_entries],
        type_digests=[entry["digest"] for entry in type_entries],
    )
    return {
        "schema": SCHEMA,
        "task_id": task_id,
        "split": str(build_row["split"]),
        "split_row": int(build_row["split_row"]),
        "analysis_program_sha256": analysis_sha,
        "function_source_sha256": function_sha,
        "producer_script_sha256": producer_sha,
        "key_id_sha256": key_id_sha256(key),
        "salt_hex": salt,
        "function_symbols": function_entries,
        "type_symbols": type_entries,
        "completeness": {
            "complete_source_symbols_projection": True,
            "source_symbols_bound_to_transform_metadata": True,
            "only_dart_scheme_imports": True,
            "ordered_function_count": len(function_entries),
            "ordered_type_count": len(type_entries),
            "ordered_commitment": commitment,
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SymbolAttestationError(
                    f"blank_jsonl_line:{path}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SymbolAttestationError(
                    f"non_object_jsonl_row:{path}:{line_number}"
                )
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


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
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--aot-manifest", type=Path, required=True)
    parser.add_argument("--aot-manifest-sha256", required=True)
    parser.add_argument("--train-build-input", type=Path, required=True)
    parser.add_argument("--train-build-input-sha256", required=True)
    parser.add_argument("--dev-build-input", type=Path, required=True)
    parser.add_argument("--dev-build-input-sha256", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1755)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    expected = str(expected).lower()
    if not SHA256_RE.fullmatch(expected):
        raise SymbolAttestationError(f"invalid_{label}_sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise SymbolAttestationError(
            f"{label}_sha256_mismatch:{observed}!={expected}"
        )
    return observed


def main() -> int:
    args = _args()
    manifest_sha = _require_file_hash(
        args.aot_manifest.resolve(),
        args.aot_manifest_sha256,
        "aot_manifest",
    )
    train_sha = _require_file_hash(
        args.train_build_input.resolve(),
        args.train_build_input_sha256,
        "train_build_input",
    )
    dev_sha = _require_file_hash(
        args.dev_build_input.resolve(),
        args.dev_build_input_sha256,
        "dev_build_input",
    )
    key = load_key(args.key_file.resolve())
    aot_rows = read_jsonl(args.aot_manifest.resolve())
    if args.expected_rows <= 0 or len(aot_rows) != args.expected_rows:
        raise SymbolAttestationError(
            f"aot_manifest_row_count:{len(aot_rows)}!={args.expected_rows}"
        )
    source_rows = read_jsonl(args.train_build_input.resolve()) + read_jsonl(
        args.dev_build_input.resolve()
    )
    source_by_task: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in source_by_task:
            raise SymbolAttestationError(
                f"invalid_or_duplicate_source_task:{task_id}"
            )
        source_by_task[task_id] = row
    output: list[dict[str, Any]] = []
    selected_task_ids: set[str] = set()
    selected_split_positions: set[tuple[str, int]] = set()
    for aot_row in aot_rows:
        task_id = str(aot_row.get("task_id") or "")
        if not task_id or task_id in selected_task_ids:
            raise SymbolAttestationError(
                f"invalid_or_duplicate_aot_task:{task_id}"
            )
        selected_task_ids.add(task_id)
        split_position = (
            str(aot_row.get("split") or ""),
            int(aot_row.get("split_row", -1)),
        )
        if split_position in selected_split_positions:
            raise SymbolAttestationError(
                "duplicate_aot_split_position:"
                f"{split_position[0]}:{split_position[1]}"
            )
        selected_split_positions.add(split_position)
        source_row = source_by_task.get(task_id)
        if source_row is None:
            raise SymbolAttestationError(
                f"aot_task_missing_source_input:{task_id}"
            )
        output.append(
            build_attestation_row(
                build_row=source_row,
                aot_row=aot_row,
                key=key,
            )
        )
    write_jsonl_atomic(args.output_jsonl.resolve(), output)
    output_sha = sha256_file(args.output_jsonl.resolve())
    report = {
        "schema": REPORT_SCHEMA,
        "rows": len(output),
        "expected_rows": args.expected_rows,
        "input_hashes": {
            "aot_manifest_sha256": manifest_sha,
            "train_build_input_sha256": train_sha,
            "dev_build_input_sha256": dev_sha,
        },
        "output_jsonl_sha256": output_sha,
        "key_id_sha256": key_id_sha256(key),
        "function_symbols": sum(
            len(row["function_symbols"]) for row in output
        ),
        "type_symbols": sum(len(row["type_symbols"]) for row in output),
        "raw_source_names_emitted": 0,
        "raw_source_paths_emitted": 0,
        "complete": True,
    }
    write_json_atomic(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
