#!/usr/bin/env python3
"""Extract lossless supported constants referenced by attested AOT functions.

The function bundle is the authority for scope.  Every immediate ``r15``
object-pool reference in every retained function is sent to a small GDB reader;
no source text or source-visible function-name lookup is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping


POOL_MARKER = re.compile(r"POOLJSON(.*)POOLEND", re.DOTALL)
POOL_OPERAND = re.compile(
    r"\[r15(?:(?P<sign>[+-])0x(?P<hex>[0-9a-f]+))?\]",
    re.IGNORECASE,
)
ANY_R15_OPERAND = re.compile(r"\[r15[^\]]*\]")
SUPPORTED_STRING_CLASS_IDS = {94: "latin-1", 95: "utf-16-le"}
BOXED_MINT_CLASS_ID = 61
BOXED_DOUBLE_CLASS_ID = 62


class ConstantExtractionError(RuntimeError):
    """The output cannot be claimed complete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_hash(path: Path, expected: str, label: str) -> str:
    expected = expected.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ConstantExtractionError(f"{label} expected SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise ConstantExtractionError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConstantExtractionError(
                    f"{label} line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ConstantExtractionError(
                    f"{label} line {line_number} is not an object"
                )
            values.append(value)
    if not values:
        raise ConstantExtractionError(f"{label} is empty")
    return values


def stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(stable_json(dict(row)) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ConstantExtractionError(f"refusing to overwrite {path}")
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


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = stable_json(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ConstantExtractionError(f"refusing to overwrite {path}")
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


def collect_pool_offsets(bundle: Mapping[str, Any]) -> list[int]:
    return sorted(collect_pool_accesses(bundle))


def collect_pool_accesses(
    bundle: Mapping[str, Any],
) -> dict[int, list[str]]:
    task_id = str(bundle.get("task_id") or "<missing>")
    functions = bundle.get("functions")
    if not isinstance(functions, list) or not functions:
        raise ConstantExtractionError(f"{task_id}: no retained functions")
    accounting = bundle.get("accounting")
    if not isinstance(accounting, dict):
        raise ConstantExtractionError(f"{task_id}: missing accounting")
    expected_functions = int(accounting.get("emitted_function_count", -1))
    if len(functions) != expected_functions:
        raise ConstantExtractionError(
            f"{task_id}: function/accounting count mismatch"
        )

    accesses: dict[int, set[str]] = {}
    instruction_count = 0
    for function in functions:
        instructions = function.get("instructions")
        if not isinstance(instructions, list):
            raise ConstantExtractionError(
                f"{task_id}: function has no instruction array"
            )
        instruction_count += len(instructions)
        for instruction in instructions:
            text = str(instruction.get("text") or "")
            operands = ANY_R15_OPERAND.findall(text)
            parsed = list(POOL_OPERAND.finditer(text))
            if len(operands) != len(parsed):
                raise ConstantExtractionError(
                    f"{task_id}: unsupported r15 operand in {text!r}"
                )
            for match in parsed:
                if match.group("hex") is None:
                    offset = 0
                else:
                    offset = int(match.group("hex"), 16)
                    if match.group("sign") == "-":
                        offset = -offset
                mnemonic = text.split(None, 1)[0].lower()
                if mnemonic == "movsd":
                    access = "inline_float64"
                elif mnemonic == "movss":
                    access = "inline_float32"
                elif mnemonic == "movups":
                    access = "inline_float32x4"
                else:
                    access = "tagged_word"
                accesses.setdefault(offset, set()).add(access)
    if instruction_count != int(
        accounting.get("emitted_instruction_count", -1)
    ):
        raise ConstantExtractionError(
            f"{task_id}: instruction/accounting count mismatch"
        )
    normalized: dict[int, list[str]] = {}
    for offset, kinds in accesses.items():
        numeric = [kind for kind in kinds if kind.startswith("inline_")]
        if len(numeric) > 1:
            raise ConstantExtractionError(
                f"{task_id}: conflicting inline access widths at "
                f"{offset:#x}: {sorted(kinds)}"
            )
        normalized[offset] = sorted(kinds)
    return normalized


def format_double(value: float) -> str:
    if math.isnan(value):
        return "double.nan"
    if math.isinf(value):
        return (
            "double.infinity"
            if value > 0
            else "double.negativeInfinity"
        )
    # repr is shortest-roundtrip and preserves the type and signed zero.
    return repr(value)


def format_float32(value: float) -> str:
    if math.isnan(value):
        return "double.nan"
    if math.isinf(value):
        return (
            "double.infinity"
            if value > 0
            else "double.negativeInfinity"
        )
    rendered = format(value, ".9g")
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0.0"
    return rendered


def keep_string(value: str) -> tuple[str | None, str | None]:
    """Keep exact program-like strings; reject only binary metadata."""

    if not value or not any(character.isalnum() for character in value):
        return None, "empty_or_punctuation"
    lowered = value.lower()
    if lowered.startswith(("dart:", "package:", "file:", "http:", "https:", "vm:")):
        return None, "uri_metadata"
    if lowered.endswith(".dart"):
        return None, "source_path"
    if ("\\" in value or "/" in value) and "." in value:
        return None, "path_metadata"
    return value, None


def decode_entries(
    entries: list[Mapping[str, Any]],
) -> tuple[list[str], list[str], dict[str, int]]:
    strings: set[str] = set()
    numbers: set[str] = set()
    counters = {
        "supported_string_objects": 0,
        "supported_number_objects": 0,
        "inline_float32_entries": 0,
        "inline_float64_entries": 0,
        "inline_float32x4_entries": 0,
        "metadata_strings_rejected": 0,
        "unsupported_or_immediate_entries": 0,
        "tagged_sentinel_entries": 0,
        "unreadable_entries": 0,
    }
    for entry in entries:
        if entry.get("inline_float64_raw") is not None:
            raw = bytes.fromhex(str(entry["inline_float64_raw"]))
            if len(raw) != 8:
                raise ConstantExtractionError(
                    "inline Float64 pool entry is not 8 bytes"
                )
            numbers.add(format_double(struct.unpack("<d", raw)[0]))
            counters["inline_float64_entries"] += 1
            counters["supported_number_objects"] += 1
            continue
        if entry.get("inline_float32_raw") is not None:
            raw = bytes.fromhex(str(entry["inline_float32_raw"]))
            if len(raw) != 4:
                raise ConstantExtractionError(
                    "inline Float32 pool entry is not 4 bytes"
                )
            numbers.add(format_float32(struct.unpack("<f", raw)[0]))
            counters["inline_float32_entries"] += 1
            counters["supported_number_objects"] += 1
            continue
        if entry.get("inline_float32x4_raw") is not None:
            raw = bytes.fromhex(str(entry["inline_float32x4_raw"]))
            if len(raw) != 16:
                raise ConstantExtractionError(
                    "inline Float32x4 pool entry is not 16 bytes"
                )
            values = ", ".join(
                format_float32(value)
                for value in struct.unpack("<4f", raw)
            )
            numbers.add(f"Float32x4({values})")
            counters["inline_float32x4_entries"] += 1
            counters["supported_number_objects"] += 1
            continue
        if entry.get("read_error"):
            counters["unreadable_entries"] += 1
            raise ConstantExtractionError(
                "unreadable tagged object-pool word "
                f"{int(entry.get('word') or 0):#x} at offset "
                f"{int(entry.get('offset') or 0):#x}: "
                f"{entry.get('read_error')}"
            )
        if entry.get("word_class") in {
            "non_object_tagged_sentinel",
            "non_object_noncanonical_word",
        }:
            counters["tagged_sentinel_entries"] += 1
            counters["unsupported_or_immediate_entries"] += 1
            continue
        raw_hex = entry.get("raw")
        if not raw_hex:
            counters["unsupported_or_immediate_entries"] += 1
            continue
        try:
            raw = bytes.fromhex(str(raw_hex))
        except ValueError as exc:
            raise ConstantExtractionError("GDB returned malformed raw hex") from exc
        if len(raw) < 16:
            raise ConstantExtractionError("GDB returned a short heap object")
        word0 = int.from_bytes(raw[0:8], "little")
        class_id = (word0 >> 12) & 0xFFFFF
        word1 = int.from_bytes(raw[8:16], "little")
        length = word1 >> 1
        if class_id in SUPPORTED_STRING_CLASS_IDS:
            byte_length = length * (2 if class_id == 95 else 1)
            expected = 16 + byte_length
            if len(raw) != expected:
                raise ConstantExtractionError(
                    f"incomplete string object: cid={class_id} "
                    f"length={length} bytes={len(raw)}"
                )
            value = raw[16:].decode(
                SUPPORTED_STRING_CLASS_IDS[class_id], "strict"
            )
            kept, rejection = keep_string(value)
            if kept is not None:
                strings.add(kept)
                counters["supported_string_objects"] += 1
            elif rejection is not None:
                counters["metadata_strings_rejected"] += 1
        elif class_id == BOXED_DOUBLE_CLASS_ID:
            if len(raw) != 16:
                raise ConstantExtractionError("boxed Double is not 16 bytes")
            numbers.add(format_double(struct.unpack("<d", raw[8:16])[0]))
            counters["supported_number_objects"] += 1
        elif class_id == BOXED_MINT_CLASS_ID:
            if len(raw) != 16:
                raise ConstantExtractionError("boxed Mint is not 16 bytes")
            numbers.add(
                str(int.from_bytes(raw[8:16], "little", signed=True))
            )
            counters["supported_number_objects"] += 1
        else:
            counters["unsupported_or_immediate_entries"] += 1
    return (
        sorted(strings, key=lambda item: (len(item), item)),
        sorted(numbers, key=lambda item: (len(item), item)),
        counters,
    )


def run_gdb(
    *,
    task_id: str,
    aot_path: Path,
    expected_aot_sha256: str,
    expected_aot_size: int,
    offsets: list[int],
    accesses: dict[int, list[str]],
    gdb: Path,
    gdb_script: Path,
    runtime: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if aot_path.stat().st_size != expected_aot_size:
        raise ConstantExtractionError(f"{task_id}: AOT size mismatch")
    require_file_hash(aot_path, expected_aot_sha256, f"{task_id} AOT")
    environment = dict(
        os.environ,
        POOL_AOT=str(aot_path),
        POOL_RUNTIME=str(runtime),
        POOL_OFFSETS_JSON=json.dumps(offsets, separators=(",", ":")),
        POOL_ACCESS_JSON=json.dumps(
            {str(key): value for key, value in accesses.items()},
            separators=(",", ":"),
            sort_keys=True,
        ),
        POOL_ENTRY_API="Dart_EnterIsolate",
        POOL_THREAD_POOL_OFF="0x690",
        POOL_MAX_OBJECT_BYTES=str(1024 * 1024),
    )
    try:
        completed = subprocess.run(
            [str(gdb), "-batch", "-x", str(gdb_script)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConstantExtractionError(f"{task_id}: GDB timeout") from exc
    match = POOL_MARKER.search(completed.stdout)
    if match is None:
        raise ConstantExtractionError(
            f"{task_id}: GDB emitted no pool payload "
            f"(exit={completed.returncode})"
        )
    payload = json.loads(match.group(1))
    if payload.get("error"):
        raise ConstantExtractionError(
            f"{task_id}: GDB pool error: {payload['error']}; "
            f"break_symbol={payload.get('break_symbol')!r}; "
            f"break_address={payload.get('break_address')!r}"
        )
    if payload.get("schema") != "dart-aot-attested-pool-dump-v1":
        raise ConstantExtractionError(f"{task_id}: GDB schema mismatch")
    if payload.get("offsets") != offsets:
        raise ConstantExtractionError(f"{task_id}: GDB offset drift")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(offsets):
        raise ConstantExtractionError(f"{task_id}: GDB entry count mismatch")
    strings, numbers, counters = decode_entries(entries)
    return {
        "schema": "dart-aot-attested-pool-constants-v1",
        "task_id": task_id,
        "strings": strings,
        "numbers": numbers,
        "err": None,
        "noff": len(offsets),
        "pool_offsets_sha256": hashlib.sha256(stable_json(offsets)).hexdigest(),
        "accounting": counters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--aot-manifest", type=Path, required=True)
    parser.add_argument("--aot-manifest-sha256", required=True)
    parser.add_argument("--aot-root", type=Path, required=True)
    parser.add_argument("--function-bundles", type=Path, required=True)
    parser.add_argument("--function-bundles-sha256", required=True)
    parser.add_argument("--gdb-script", type=Path, required=True)
    parser.add_argument("--gdb-script-sha256", required=True)
    parser.add_argument("--gdb", type=Path, default=Path("/usr/bin/gdb"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--task-id")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_rows <= 0 or args.workers <= 0:
        raise ConstantExtractionError("row and worker counts must be positive")
    manifest_sha = require_file_hash(
        args.aot_manifest, args.aot_manifest_sha256, "AOT manifest"
    )
    bundles_sha = require_file_hash(
        args.function_bundles,
        args.function_bundles_sha256,
        "function bundles",
    )
    gdb_script_sha = require_file_hash(
        args.gdb_script, args.gdb_script_sha256, "GDB reader"
    )
    manifest = load_jsonl(args.aot_manifest, "AOT manifest")
    bundles = load_jsonl(args.function_bundles, "function bundles")
    if len(manifest) != len(bundles):
        raise ConstantExtractionError(
            f"manifest/bundle row counts differ: "
            f"{len(manifest)}/{len(bundles)}"
        )
    manifest_by_task = {str(row.get("task_id") or ""): row for row in manifest}
    bundle_by_task = {str(row.get("task_id") or ""): row for row in bundles}
    if (
        "" in manifest_by_task
        or "" in bundle_by_task
        or len(manifest_by_task) != len(manifest)
        or len(bundle_by_task) != len(bundles)
    ):
        raise ConstantExtractionError("missing or duplicate task IDs")
    if set(manifest_by_task) != set(bundle_by_task):
        raise ConstantExtractionError("manifest/bundle task sets differ")

    ordered_tasks = [str(row["task_id"]) for row in manifest]
    if args.task_id:
        if args.task_id not in manifest_by_task:
            raise ConstantExtractionError(f"unknown task ID {args.task_id}")
        ordered_tasks = [args.task_id]
    if len(ordered_tasks) != args.expected_rows:
        raise ConstantExtractionError(
            f"selected {len(ordered_tasks)} rows, expected {args.expected_rows}"
        )

    jobs: list[dict[str, Any]] = []
    for task_id in ordered_tasks:
        manifest_row = manifest_by_task[task_id]
        bundle = bundle_by_task[task_id]
        if bundle.get("inputs", {}).get("aot_sha256") != manifest_row.get(
            "aot_sha256"
        ):
            raise ConstantExtractionError(
                f"{task_id}: bundle/manifest AOT hash mismatch"
            )
        relative_path = Path(str(manifest_row.get("aot_path") or ""))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ConstantExtractionError(f"{task_id}: unsafe AOT path")
        jobs.append(
            {
                "task_id": task_id,
                "aot_path": args.aot_root / relative_path,
                "expected_aot_sha256": str(manifest_row["aot_sha256"]),
                "expected_aot_size": int(manifest_row["aot_size_bytes"]),
                "offsets": collect_pool_offsets(bundle),
                "accesses": collect_pool_accesses(bundle),
                "gdb": args.gdb,
                "gdb_script": args.gdb_script,
                "runtime": args.runtime,
                "timeout_seconds": args.timeout_seconds,
            }
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda kwargs: run_gdb(**kwargs), jobs))
    if [str(row["task_id"]) for row in rows] != ordered_tasks:
        raise ConstantExtractionError("parallel extraction changed row order")
    atomic_write_jsonl(args.output_jsonl, rows)
    output_sha = sha256_file(args.output_jsonl)
    totals = {
        key: sum(int(row["accounting"][key]) for row in rows)
        for key in rows[0]["accounting"]
    }
    report = {
        "schema": "dart-aot-attested-pool-constants-report-v1",
        "passed": True,
        "rows": len(rows),
        "expected_rows": args.expected_rows,
        "source_text_read": False,
        "source_symbol_names_read": False,
        "all_attested_functions_scanned": True,
        "all_immediate_r15_offsets_accounted": True,
        "no_item_or_length_cap": True,
        "nonfinite_doubles_serialized_as_dart_expressions": True,
        "input_hashes": {
            "aot_manifest_sha256": manifest_sha,
            "function_bundles_sha256": bundles_sha,
            "gdb_script_sha256": gdb_script_sha,
        },
        "counts": {
            "pool_offsets": sum(int(row["noff"]) for row in rows),
            "strings": sum(len(row["strings"]) for row in rows),
            "numbers": sum(len(row["numbers"]) for row in rows),
            **totals,
        },
        "output_jsonl_sha256": output_sha,
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
