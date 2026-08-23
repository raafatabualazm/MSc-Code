"""Read exact Dart AOT object-pool offsets inside GDB.

This file is executed by ``gdb -batch -x``.  The caller supplies the complete
set of ``r15`` offsets found in the already-attested user-function bundle, so
this script never discovers functions by source-visible symbol name.

Environment:
  POOL_AOT              exact AOT ELF path
  POOL_RUNTIME          matching dartaotruntime
  POOL_OFFSETS_JSON     JSON array of non-negative integer byte offsets
  POOL_ACCESS_JSON      JSON object mapping offsets to assembly access classes
  POOL_ENTRY_API        isolate-entry API stop (default Dart_EnterIsolate)
  POOL_THREAD_POOL_OFF  Dart Thread object-pool-pointer offset (default 0x690)
  POOL_MAX_OBJECT_BYTES maximum bytes read from one referenced heap object
"""
from __future__ import annotations

import json
import os
import re

import gdb  # type: ignore


AOT = os.environ["POOL_AOT"]
RUNTIME = os.environ["POOL_RUNTIME"]
OFFSETS = json.loads(os.environ["POOL_OFFSETS_JSON"])
ACCESSES = json.loads(os.environ["POOL_ACCESS_JSON"])
ENTRY_API = os.environ.get(
    "POOL_ENTRY_API", "Dart_EnterIsolate"
).strip()
THREAD_POOL_OFFSET = int(
    os.environ.get("POOL_THREAD_POOL_OFF", "0x690"), 0
)
MAX_OBJECT_BYTES = int(os.environ.get("POOL_MAX_OBJECT_BYTES", "1048576"))

out = {
    "schema": "dart-aot-attested-pool-dump-v1",
    "r15": None,
    "offsets": [],
    "entries": [],
    "break_symbol": None,
    "break_address": None,
    "thread_address": None,
    "thread_pool_offset": THREAD_POOL_OFFSET,
    "error": None,
}


def _validated_offsets() -> list[int]:
    if not isinstance(OFFSETS, list):
        raise RuntimeError("POOL_OFFSETS_JSON is not an array")
    values: list[int] = []
    for value in OFFSETS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("POOL_OFFSETS_JSON contains a non-integer")
        if value < -0x7FFFFFFF or value > 0x7FFFFFFF:
            raise RuntimeError(f"invalid object-pool offset {value}")
        values.append(value)
    if values != sorted(set(values)):
        raise RuntimeError("POOL_OFFSETS_JSON must be sorted and unique")
    return values


def _validated_accesses(offsets: list[int]) -> dict[int, list[str]]:
    if not isinstance(ACCESSES, dict):
        raise RuntimeError("POOL_ACCESS_JSON is not an object")
    allowed = {
        "tagged_word",
        "inline_float32",
        "inline_float64",
        "inline_float32x4",
    }
    values: dict[int, list[str]] = {}
    for raw_offset, raw_kinds in ACCESSES.items():
        try:
            offset = int(raw_offset)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("POOL_ACCESS_JSON has a bad offset") from exc
        if (
            not isinstance(raw_kinds, list)
            or not raw_kinds
            or any(kind not in allowed for kind in raw_kinds)
        ):
            raise RuntimeError(
                f"POOL_ACCESS_JSON has bad access classes at {offset}"
            )
        kinds = sorted(set(raw_kinds))
        numeric = [
            kind for kind in kinds if kind.startswith("inline_")
        ]
        if len(numeric) > 1:
            raise RuntimeError(
                f"conflicting inline access widths at offset {offset}"
            )
        values[offset] = kinds
    if sorted(values) != offsets:
        raise RuntimeError("POOL_ACCESS_JSON keys differ from offsets")
    return values


def _heap_object_read_size(header: bytes) -> int:
    """Return the exact useful payload size for supported Dart 3.12 objects."""

    if len(header) < 16:
        return 16
    word0 = int.from_bytes(header[0:8], "little")
    class_id = (word0 >> 12) & 0xFFFFF
    word1 = int.from_bytes(header[8:16], "little")
    length = word1 >> 1
    if class_id == 94:
        return 16 + length
    if class_id == 95:
        return 16 + (2 * length)
    if class_id in (61, 62):
        return 16
    return 32


def _lookup_symbol_address(objfile, name: str) -> int | None:
    symbol = (
        objfile.lookup_static_symbol(name)
        or objfile.lookup_global_symbol(name)
        or gdb.lookup_static_symbol(name)
        or gdb.lookup_global_symbol(name)
    )
    if symbol is not None:
        return int(symbol.value().address)
    try:
        info = gdb.execute(
            "info address " + json.dumps(name), to_string=True
        )
    except gdb.error:
        info = ""
    match = re.search(r"\bis at (0x[0-9a-fA-F]+)\b", info)
    if match:
        return int(match.group(1), 16)
    # A fuzzy ``info functions main`` also matches many runtime helpers.  An
    # exact AOT ``main`` must have been found through objfile lookup above.
    if name == "main":
        return None
    try:
        functions = gdb.execute(
            "info functions " + name.rsplit(" ", 1)[-1],
            to_string=True,
        )
    except gdb.error:
        return None
    for line in functions.splitlines():
        if name not in line:
            continue
        address = re.match(r"^\s*(0x[0-9a-fA-F]+)\s+", line)
        if address:
            return int(address.group(1), 16)
    return None


try:
    offsets = _validated_offsets()
    accesses = _validated_accesses(offsets)
    if MAX_OBJECT_BYTES < 32:
        raise RuntimeError("POOL_MAX_OBJECT_BYTES must be at least 32")

    gdb.execute("set confirm off")
    gdb.execute("set pagination off")
    gdb.execute("set breakpoint pending on")
    gdb.execute("set auto-solib-add on")
    gdb.execute(f'file "{RUNTIME}"')
    gdb.execute(f'set args "{AOT}"')

    target = os.path.realpath(AOT)
    gdb.execute("set stop-on-solib-events 1")
    gdb.execute("run")
    aot_obj = None
    for _ in range(40):
        for objfile in gdb.current_progspace().objfiles():
            if (
                objfile.filename
                and os.path.realpath(objfile.filename) == target
            ):
                aot_obj = objfile
                break
        if aot_obj is not None:
            break
        gdb.execute("continue", to_string=True)
    if aot_obj is None:
        raise RuntimeError("AOT object was not mapped after 40 solib events")
    gdb.execute("set stop-on-solib-events 0")

    break_address = _lookup_symbol_address(aot_obj, ENTRY_API)
    if break_address is None:
        raise RuntimeError(f"runtime entry API not found: {ENTRY_API}")
    out["break_symbol"] = ENTRY_API
    out["break_address"] = break_address

    breakpoint = gdb.Breakpoint(f"*{break_address:#x}", internal=True)
    gdb.execute("continue")
    if breakpoint.hit_count < 1:
        raise RuntimeError(f"runtime entry API was not reached: {ENTRY_API}")
    # At function return, TLS contains the newly entered isolate's Thread and
    # its object-pool field is initialized.  Stopping at entry is too early.
    gdb.execute("finish")
    fs_base = int(gdb.parse_and_eval("$fs_base")) & 0xFFFFFFFFFFFFFFFF
    inferior = gdb.selected_inferior()
    thread_pointer = int.from_bytes(
        bytes(inferior.read_memory(fs_base - 0x30, 8)), "little"
    )
    if thread_pointer == 0:
        raise RuntimeError("Dart current-thread TLS pointer is null")
    pool_pointer = int.from_bytes(
        bytes(
            inferior.read_memory(
                thread_pointer + THREAD_POOL_OFFSET, 8
            )
        ),
        "little",
    )
    # Dart heap/object-pool pointers are tagged and therefore end in bit 0=1.
    if pool_pointer <= 1 or not (pool_pointer & 1):
        raise RuntimeError(
            f"invalid Dart object-pool pointer {pool_pointer:#x}"
        )
    out["thread_address"] = thread_pointer
    out["r15"] = pool_pointer
    out["offsets"] = offsets

    for offset in offsets:
        entry = {
            "offset": offset,
            "word": None,
            "raw": None,
            "word_class": None,
            "access_kinds": accesses[offset],
        }
        try:
            pool_address = pool_pointer + offset
            kinds = accesses[offset]
            if "inline_float32x4" in kinds:
                entry["inline_float32x4_raw"] = bytes(
                    inferior.read_memory(pool_address, 16)
                ).hex()
                out["entries"].append(entry)
                continue
            if "inline_float64" in kinds:
                entry["inline_float64_raw"] = bytes(
                    inferior.read_memory(pool_address, 8)
                ).hex()
                out["entries"].append(entry)
                continue
            if "inline_float32" in kinds:
                entry["inline_float32_raw"] = bytes(
                    inferior.read_memory(pool_address, 4)
                ).hex()
                out["entries"].append(entry)
                continue
            word = int.from_bytes(
                bytes(inferior.read_memory(pool_address, 8)), "little"
            )
            entry["word"] = word
            if (word & 1) and word > 0x10000:
                address = word - 1
                if word == 0x8000000000000001:
                    # Dart's unlinked-call/non-object object-pool sentinel.
                    entry["word_class"] = "non_object_tagged_sentinel"
                elif address >= 0x0000800000000000:
                    # Not a canonical userspace address on x86-64, therefore
                    # this is a raw tagged sentinel/zap word, not an omitted
                    # heap object.
                    entry["word_class"] = "non_object_noncanonical_word"
                else:
                    entry["word_class"] = "tagged_heap_object"
                    header = bytes(inferior.read_memory(address, 32))
                    read_size = _heap_object_read_size(header)
                    if read_size > MAX_OBJECT_BYTES:
                        raise RuntimeError(
                            f"pool object at offset {offset:#x} requires "
                            f"{read_size} bytes, above limit "
                            f"{MAX_OBJECT_BYTES}"
                        )
                    entry["raw"] = bytes(
                        inferior.read_memory(address, read_size)
                    ).hex()
        except gdb.MemoryError:
            entry["read_error"] = "memory"
        out["entries"].append(entry)
    gdb.execute("kill")
except Exception as exc:  # noqa: BLE001 - must serialize GDB failures
    out["error"] = str(exc)[:1000]

print("POOLJSON" + json.dumps(out, sort_keys=True) + "POOLEND")
