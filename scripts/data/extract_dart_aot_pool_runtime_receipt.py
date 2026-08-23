#!/usr/bin/env python3
"""Resolve target-referenced Dart AOT pool literals from an x64 AOT ELF alone.

This independent audit path consumes the AOT snapshot and its matching pinned
``dartaotruntime``.  It obtains pool offsets from target symbol disassembly,
stops the snapshot at the always-executed Dart ``_startMainIsolate`` entry after
pool initialization (so an empty ``main`` is supported), and decodes only pinned
Dart 3.12.2 primitive object layouts.  No Dart source, tests, labels, compiler
disassembly log, or snapshot profile is accepted.

The scalable production extractor is ``extract_dart_aot_pool_receipt.py``.  This
runtime resolver is deliberately separate so a sample of static receipts can be
cross-checked against evidence derived from the shipped AOT bytes alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


RUNTIME_RECEIPT_SCHEMA = "dart-aot-runtime-pool-receipt-v1"
SDK_LAYOUT_CONTRACT = "dart-3.12.2-linux-x64-object-layout-v1"
SUPPORTED_LITERAL_CIDS = {
    61: "Mint",
    62: "Double",
    94: "OneByteString",
    95: "TwoByteString",
}


class RuntimePoolReceiptError(ValueError):
    """Raised when source-blind runtime extraction cannot be proven safely."""


NM_SYMBOL_RE = re.compile(
    r"^(?P<address>[0-9a-fA-F]+)\s+(?P<size>[0-9a-fA-F]+)\s+\w\s+(?P<name>.+)$"
)
OBJDUMP_INSTRUCTION_RE = re.compile(
    r"^\s*(?P<pc>[0-9a-fA-F]+):\s+(?P<instruction>.*)$"
)
X64_POOL_RE = re.compile(r"\[r15\+(?P<offset>0x[0-9a-fA-F]+)\]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _scope_role(symbol_name: str, target: str) -> str | None:
    if symbol_name == target:
        return "exact"
    if any(symbol_name.startswith(target + separator) for separator in (".", "_", "<")):
        return "descendant"
    return None


def parse_target_symbols(nm_output: str, target: str) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for line in nm_output.splitlines():
        match = NM_SYMBOL_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        role = _scope_role(name, target)
        if role is None:
            continue
        size = int(match.group("size"), 16)
        if size <= 0:
            continue
        symbols.append(
            {
                "raw_name": name,
                "address": int(match.group("address"), 16),
                "size": size,
                "scope_role": role,
            }
        )
    if not symbols:
        raise RuntimePoolReceiptError(f"target_symbols_not_found:{target}")
    symbols.sort(
        key=lambda symbol: (
            0 if symbol["scope_role"] == "exact" else 1,
            symbol["raw_name"],
            symbol["address"],
        )
    )
    exact_counter = 0
    descendant_counter = 0
    for symbol in symbols:
        if symbol["scope_role"] == "exact":
            symbol["function_id"] = (
                target if exact_counter == 0 else f"{target}_exact_{exact_counter}"
            )
            exact_counter += 1
        else:
            symbol["function_id"] = f"{target}_descendant_{descendant_counter}"
            descendant_counter += 1
    return symbols


def parse_objdump_pool_xrefs(
    objdump_output: str, *, function_id: str
) -> list[dict[str, Any]]:
    xrefs: list[dict[str, Any]] = []
    for line in objdump_output.splitlines():
        instruction = OBJDUMP_INSTRUCTION_RE.match(line)
        if instruction is None:
            continue
        for pool in X64_POOL_RE.finditer(instruction.group("instruction")):
            xrefs.append(
                {
                    "function_id": function_id,
                    "pc": int(instruction.group("pc"), 16),
                    "pool_offset": int(pool.group("offset"), 16),
                }
            )
    return xrefs


def collect_aot_xrefs(
    aot: Path,
    *,
    target: str,
    nm: str,
    objdump: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nm_output = _run([nm, "-S", "-an", "--defined-only", str(aot)]).stdout
    symbols = parse_target_symbols(nm_output, target)
    xrefs: list[dict[str, Any]] = []
    for symbol in symbols:
        start = symbol["address"]
        stop = start + symbol["size"]
        output = _run(
            [
                objdump,
                "-d",
                "-Mintel",
                "--no-show-raw-insn",
                f"--start-address=0x{start:x}",
                f"--stop-address=0x{stop:x}",
                str(aot),
            ]
        ).stdout
        xrefs.extend(
            parse_objdump_pool_xrefs(output, function_id=symbol["function_id"])
        )
    if not xrefs:
        raise RuntimePoolReceiptError("target_has_no_x64_pool_xrefs")
    xrefs.sort(key=lambda item: (item["pool_offset"], item["function_id"], item["pc"]))
    public_symbols = [
        {
            "function_id": symbol["function_id"],
            "scope_role": symbol["scope_role"],
            "aot_address": f"0x{symbol['address']:x}",
            "size_bytes": symbol["size"],
        }
        for symbol in symbols
    ]
    return public_symbols, xrefs


def _gdb_program(offsets: list[int]) -> str:
    # GDB's embedded Python reads the live, relocated object pool.  ASLR
    # addresses are intentionally omitted from the sentinel JSON.
    return f"""set pagination off
set breakpoint pending on
break _startMainIsolate
run
python
import gdb, json, struct
inferior = gdb.selected_inferior()
pp = int(gdb.parse_and_eval('$r15'))
thr = int(gdb.parse_and_eval('$r14'))
def read_u64(address):
    return struct.unpack('<Q', inferior.read_memory(address, 8).tobytes())[0]
canonical_null = read_u64(thr + 0x80)
canonical_true = read_u64(thr + 0x90)
canonical_false = read_u64(thr + 0x98)
entries = []
for offset in {offsets!r}:
    slot = pp + offset
    try:
        tagged = read_u64(slot)
    except Exception as error:
        entries.append({{'pool_offset': hex(offset), 'category': 'unresolved', 'reason': 'slot_read_failed'}})
        continue
    if tagged == canonical_null:
        entries.append({{'pool_offset': hex(offset), 'category': 'literal', 'literal': {{'type': 'null', 'value': None}}, 'runtime_type': 'Null'}})
        continue
    if tagged == canonical_true or tagged == canonical_false:
        entries.append({{'pool_offset': hex(offset), 'category': 'literal', 'literal': {{'type': 'bool', 'value': tagged == canonical_true}}, 'runtime_type': 'bool'}})
        continue
    if tagged & 1 == 0:
        entries.append({{
            'pool_offset': hex(offset),
            'category': 'nonliteral',
            'nonliteral_kind': 'untagged_or_smi',
        }})
        continue
    raw = tagged - 1
    try:
        header = struct.unpack('<Q', inferior.read_memory(raw, 8).tobytes())[0]
        class_id = (header & 0xffffffff) >> 12
        word1 = struct.unpack('<Q', inferior.read_memory(raw + 8, 8).tobytes())[0]
    except Exception as error:
        entries.append({{'pool_offset': hex(offset), 'category': 'unresolved', 'reason': 'object_read_failed'}})
        continue
    entry = {{'pool_offset': hex(offset), 'class_id': class_id}}
    if class_id in (94, 95):
        length = word1 >> 1
        width = 1 if class_id == 94 else 2
        try:
            payload = inferior.read_memory(raw + 16, length * width).tobytes() if length else b''
            if width == 1:
                units = list(payload)
                entry.update({{'category': 'literal', 'literal': {{'type': 'string', 'code_units': units}}, 'runtime_type': 'OneByteString'}})
            else:
                units = list(struct.unpack('<' + 'H' * length, payload)) if length else []
                entry.update({{'category': 'literal', 'literal': {{'type': 'string', 'code_units': units}}, 'runtime_type': 'TwoByteString'}})
        except Exception:
            entry.update({{'category': 'unresolved', 'reason': 'string_payload_read_failed'}})
    elif class_id == 62:
        entry.update({{'category': 'literal', 'literal': {{'type': 'double', 'bits_hex': format(word1, '016x')}}, 'runtime_type': 'Double'}})
    elif class_id == 61:
        value = struct.unpack('<q', struct.pack('<Q', word1))[0]
        entry.update({{'category': 'literal', 'literal': {{'type': 'int', 'decimal': str(value)}}, 'runtime_type': 'Mint'}})
    else:
        entry.update({{'category': 'nonliteral', 'nonliteral_kind': 'runtime_object', 'runtime_type': 'cid_' + str(class_id)}})
    entries.append(entry)
print('DART_POOL_RUNTIME_JSON=' + json.dumps({{'entries': entries}}, ensure_ascii=True, allow_nan=False, sort_keys=True))
end
quit
"""


def resolve_pool_with_gdb(
    aot: Path,
    runtime: Path,
    *,
    offsets: list[int],
    target: str,
    gdb: str,
) -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".gdb", delete=False, encoding="utf-8", newline="\n"
    ) as command_file:
        command_file.write(_gdb_program(offsets))
        command_path = Path(command_file.name)
    try:
        completed = subprocess.run(
            [
                gdb,
                "-q",
                "-batch",
                "-x",
                str(command_path),
                "--args",
                str(runtime),
                str(aot),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        command_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        diagnostic = (completed.stdout + "\n" + completed.stderr)[-4000:]
        raise RuntimePoolReceiptError(
            f"gdb_failed:{completed.returncode}:{diagnostic}"
        )
    prefix = "DART_POOL_RUNTIME_JSON="
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            try:
                result = json.loads(line.removeprefix(prefix))
            except json.JSONDecodeError as error:
                raise RuntimePoolReceiptError("invalid_gdb_sentinel_json") from error
            return result["entries"]
    raise RuntimePoolReceiptError("gdb_sentinel_missing")


def assemble_runtime_receipt(
    *,
    target: str,
    aot_sha256: str,
    runtime_sha256: str,
    functions: list[dict[str, Any]],
    xrefs: list[dict[str, Any]],
    resolved_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    by_offset: dict[int, dict[str, Any]] = {}
    for entry in resolved_entries:
        offset = int(entry["pool_offset"], 16)
        if offset in by_offset:
            raise RuntimePoolReceiptError(f"duplicate_resolved_offset:0x{offset:x}")
        by_offset[offset] = dict(entry)
    uses_by_offset: dict[int, list[dict[str, str]]] = {}
    for xref in xrefs:
        uses_by_offset.setdefault(xref["pool_offset"], []).append(
            {"function_id": xref["function_id"], "pc": f"0x{xref['pc']:x}"}
        )
    if set(by_offset) != set(uses_by_offset):
        raise RuntimePoolReceiptError("resolved_xref_offset_set_mismatch")

    entries: list[dict[str, Any]] = []
    for offset in sorted(by_offset):
        entry = by_offset[offset]
        entry["pool_offset"] = f"0x{offset:x}"
        entry["pp_offset"] = offset
        entry["uses"] = sorted(
            uses_by_offset[offset],
            key=lambda use: (use["function_id"], int(use["pc"], 16)),
        )
        entries.append(entry)
    bool_pool_entries = sum(
        entry.get("literal", {}).get("type") == "bool" for entry in entries
    )
    null_pool_entries = sum(
        entry.get("literal", {}).get("type") == "null" for entry in entries
    )
    return {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "source_blind": True,
        "target_function": target,
        "layout_contract": SDK_LAYOUT_CONTRACT,
        "inputs": {
            "aot_sha256": aot_sha256,
            "dartaotruntime_sha256": runtime_sha256,
        },
        "target_scope": functions,
        "entries": entries,
        "pool_literal_presence": {
            "bool": bool_pool_entries > 0,
            "null": null_pool_entries > 0,
        },
        "summary": {
            "target_functions": len(functions),
            "unique_pool_entries": len(entries),
            "pool_uses": sum(len(entry["uses"]) for entry in entries),
            "literal_entries": sum(entry["category"] == "literal" for entry in entries),
            "bool_pool_entries": bool_pool_entries,
            "null_pool_entries": null_pool_entries,
            "nonliteral_entries": sum(entry["category"] == "nonliteral" for entry in entries),
            "unresolved_entries": sum(entry["category"] == "unresolved" for entry in entries),
        },
    }


def build_runtime_receipt(
    aot: Path,
    runtime: Path,
    *,
    target: str = "candidate",
    nm: str = "nm",
    objdump: str = "x86_64-linux-gnu-objdump",
    gdb: str = "gdb",
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
        raise RuntimePoolReceiptError(f"invalid_target_name:{target!r}")
    if not aot.is_file() or not runtime.is_file():
        raise RuntimePoolReceiptError("aot_or_runtime_missing")
    architecture = _run([objdump, "-f", str(aot)]).stdout.lower()
    if "x86-64" not in architecture and "i386:x86-64" not in architecture:
        raise RuntimePoolReceiptError("unsupported_aot_architecture")
    functions, xrefs = collect_aot_xrefs(
        aot,
        target=target,
        nm=nm,
        objdump=objdump,
    )
    offsets = sorted({xref["pool_offset"] for xref in xrefs})
    resolved = resolve_pool_with_gdb(
        aot,
        runtime,
        offsets=offsets,
        target=target,
        gdb=gdb,
    )
    return assemble_runtime_receipt(
        target=target,
        aot_sha256=_sha256_file(aot),
        runtime_sha256=_sha256_file(runtime),
        functions=functions,
        xrefs=xrefs,
        resolved_entries=resolved,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="candidate")
    parser.add_argument("--nm", default="nm")
    parser.add_argument("--objdump", default="x86_64-linux-gnu-objdump")
    parser.add_argument("--gdb", default="gdb")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        receipt = build_runtime_receipt(
            args.aot,
            args.runtime,
            target=args.target,
            nm=args.nm,
            objdump=args.objdump,
            gdb=args.gdb,
        )
    except (OSError, subprocess.CalledProcessError, RuntimePoolReceiptError) as error:
        raise SystemExit(f"runtime_pool_receipt_failed:{error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
