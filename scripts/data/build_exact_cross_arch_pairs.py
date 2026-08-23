#!/usr/bin/env python3
"""Build exact neutral-source x86-64/AArch64 Dart AOT pairs.

The input is the *private* neutralized release.  Each selected ``candidate``
source and its test harness form one immutable Dart program.  That exact byte
stream is compiled for Linux x64 and Linux arm64 under one Dart SDK.  Raw AOT
artifacts, source, tests, signatures, build commands and hashes remain private.
The public graph rows contain no source, tests, signature, paths or task IDs.

This script deliberately does not consume the historical Flutter ARM64 pool:
those binaries were built from the original named sources and therefore cannot
establish a same-neutral-source architecture pair.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.build_graph_v2_jsonl import (  # noqa: E402
    GRAPH_SCHEMA_VERSION,
    build_record,
    extractor_sha256,
)


SCHEMA = "exact-neutral-cross-arch-pairs-v1"
PUBLIC_SCHEMA = "exact-neutral-cross-arch-public-v1"
PRIVATE_SCHEMA = "exact-neutral-cross-arch-private-v1"
MODEL_SCHEMA = "direct-source-graph-public-v1"
PAIR_RECEIPT_SCHEMA = "exact-neutral-cross-arch-pair-receipt-v1"
PAIR_ID_POLICY = "hmac-sha256-neutral-source-v1"
STANDARD_PROGRAM_IMPORTS = ("dart:async", "dart:convert")
SYMBOL_POLICY = {
    "version": "cross-arch-symbol-scrub-v1",
    "self": "candidate / candidate+offset",
    "self_nested": "candidate.local_N",
    "trusted_runtime": "stub _iso_stub_* and dart:*",
    "snapshot_image": "runtime_image",
    "untrusted": "symbol_N",
}
FIXED_PUBLIC_BASE = 0x100000
TARGET_ANNOTATION = re.compile(r"(?<![A-Za-z0-9_])(?:0x)?([0-9a-fA-F]+)\s*<([^>]+)>")
ANNOTATION = re.compile(r"<([^>]+)>")
INSTRUCTION_LINE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.+?)\s*$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: float = 180,
    check: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update(
        {"DART_SUPPRESS_ANALYTICS": "1", "PUB_ENVIRONMENT": "exact_cross_arch_builder"}
    )
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"command_timeout:{command}:{error}") from error
    result = {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }
    if check and process.returncode != 0:
        raise RuntimeError(
            f"command_failed:{command}:rc={process.returncode}:"
            f"{(process.stderr or process.stdout)[-4000:]}"
        )
    return result


def executable(path_or_name: str) -> Path:
    found = shutil.which(path_or_name)
    if found is None:
        raise FileNotFoundError(f"required executable not found: {path_or_name}")
    return Path(found).resolve()


def rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"blank_jsonl_line:{path}:{index + 1}")
            yield index, json.loads(line)


def load_selected(path: Path, selected: set[int]) -> list[tuple[int, dict[str, Any]]]:
    result = [(index, row) for index, row in rows(path) if index in selected]
    found = {index for index, _ in result}
    if found != selected:
        raise ValueError(f"missing requested indices: {sorted(selected - found)}")
    return result


def input_row_count(path: Path) -> int:
    return sum(1 for _ in rows(path))


def load_indices_file(path: Path, file_format: str) -> list[int]:
    """Load an explicit zero-based list or the frozen split alignment JSONL."""
    values: list[int] = []
    split_lines: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank_indices_file_line:{path}:{line_number}")
            if file_format == "zero-based":
                try:
                    value = int(line.strip())
                except ValueError as error:
                    raise ValueError(
                        f"invalid_zero_based_index:{path}:{line_number}"
                    ) from error
            elif file_format == "alignment-jsonl":
                try:
                    record = json.loads(line)
                    original_line = int(record["original_line"])
                    split_line = int(record["split_line"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid_alignment_record:{path}:{line_number}"
                    ) from error
                if original_line < 1:
                    raise ValueError(f"invalid_original_line:{path}:{line_number}")
                if split_line < 1 or split_line in split_lines:
                    raise ValueError(f"invalid_split_line:{path}:{line_number}")
                split_lines.add(split_line)
                value = original_line - 1
            else:  # argparse constrains this; retain a fail-closed library API.
                raise ValueError(f"unknown_indices_file_format:{file_format}")
            if value < 0:
                raise ValueError(f"negative_index:{path}:{line_number}")
            values.append(value)
    if file_format == "alignment-jsonl" and split_lines != set(
        range(1, len(values) + 1)
    ):
        raise ValueError(f"non_contiguous_split_lines:{path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate_indices:{path}")
    if not values:
        raise ValueError(f"empty_indices_file:{path}")
    return sorted(values)


def select_shard(
    indices: Iterable[int], shard_size: int | None, shard_index: int | None
) -> tuple[list[int], dict[str, int | None]]:
    ordered = sorted(indices)
    if (shard_size is None) != (shard_index is None):
        raise ValueError("shard_size_and_shard_index_must_be_used_together")
    if shard_size is None:
        return ordered, {
            "shard_size": None,
            "shard_index": None,
            "shard_count": 1,
        }
    if shard_size < 1 or shard_index is None or shard_index < 0:
        raise ValueError("invalid_shard_parameters")
    shard_count = (len(ordered) + shard_size - 1) // shard_size
    if shard_index >= shard_count:
        raise ValueError(
            f"shard_index_out_of_range:{shard_index}:shard_count={shard_count}"
        )
    start = shard_index * shard_size
    return ordered[start : start + shard_size], {
        "shard_size": shard_size,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }


def neutral_program(row: dict[str, Any]) -> tuple[str, str, str]:
    source = str(row.get("dart_source") or "")
    tests = str(row.get("tests") or "")
    if not source.strip() or not tests.strip():
        raise ValueError("source_and_tests_required")
    if "@pragma('vm:entry-point')" not in source:
        raise ValueError("candidate_missing_entry_point_pragma")
    if "@pragma('vm:never-inline')" not in source:
        raise ValueError("candidate_missing_never_inline_pragma")
    if not re.search(r"\bcandidate\s*\(", source):
        raise ValueError("candidate_definition_missing")
    if re.search(r"^\s*(?:Future<\s*void\s*>|void)\s+main\s*\(", source, re.MULTILINE):
        raise ValueError("private_source_must_not_contain_main")
    if not re.search(r"\bmain\s*\(", tests):
        raise ValueError("test_harness_main_missing")
    # The scrubbed release stores its neutral source and unique harness without
    # the standard harness imports. Reconstruct exactly the same deterministic
    # compile envelope used by build_scrubbed_dataset.py. These library names
    # are fixed infrastructure, not task-specific input.
    program_body = source.rstrip() + "\n\n" + tests.lstrip()
    missing_imports = [
        uri
        for uri in STANDARD_PROGRAM_IMPORTS
        if not re.search(
            rf"^\s*import\s+['\"]{re.escape(uri)}['\"]\s*;", program_body, re.MULTILINE
        )
    ]
    import_block = "".join(f"import '{uri}';\n" for uri in missing_imports)
    library_match = re.match(r"\s*library\s+[^;]+;\s*", program_body)
    if library_match:
        insert = library_match.end()
        program = program_body[:insert] + import_block + program_body[insert:]
    else:
        program = import_block + program_body
    if not program.endswith("\n"):
        program += "\n"
    return source, tests, program


def semantic_pair_id(source: str, salt: bytes) -> str:
    digest = hmac.new(salt, source.encode(), hashlib.sha256).hexdigest()
    return "sp_" + digest[:24]


def symbol_rows(aot: Path, symbol: str, readelf: Path) -> list[dict[str, int]]:
    result = run([str(readelf), "-sW", str(aot)], cwd=aot.parent)
    symbols: list[dict[str, int]] = []
    for line in result["stdout"].splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[3] != "FUNC" or parts[-1] != symbol:
            continue
        try:
            symbols.append({"address": int(parts[1], 16), "size": int(parts[2])})
        except ValueError:
            continue
    if not symbols:
        raise RuntimeError(f"symbol_not_found:{symbol}:{aot}")
    return sorted(symbols, key=lambda item: (-item["size"], item["address"]))


def scrub_symbol(symbol: str, aliases: dict[str, str], nested: dict[str, str]) -> str:
    # Target-address replacement and the general annotation pass can observe
    # the same annotation.  Neutral forms therefore must be idempotent.
    if symbol == "runtime_image" or re.fullmatch(r"symbol_\d+", symbol):
        return symbol
    if re.fullmatch(r"candidate\.local_\d+", symbol):
        return symbol
    if symbol == "candidate" or symbol.startswith("candidate+"):
        return symbol
    if symbol.startswith("candidate."):
        if symbol not in nested:
            nested[symbol] = f"candidate.local_{len(nested)}"
        return nested[symbol]
    stub = re.match(r"^stub _iso_stub_([A-Za-z0-9_]+)$", symbol)
    if stub:
        return f"stub _iso_stub_{stub.group(1)}"
    if symbol.startswith("dart:"):
        return "dart:" + re.sub(r"[^A-Za-z0-9_:.+-]", "_", symbol[5:])
    if symbol.startswith("_kDartIsolateSnapshotInstructions"):
        return "runtime_image"
    if symbol not in aliases:
        aliases[symbol] = f"symbol_{len(aliases)}"
    return aliases[symbol]


def scrub_and_rebase_instruction(
    instruction: str,
    *,
    true_base: int,
    true_stop: int,
    aliases: dict[str, str],
    nested: dict[str, str],
) -> str:
    def replace_target(match: re.Match[str]) -> str:
        address = int(match.group(1), 16)
        symbol = scrub_symbol(match.group(2), aliases, nested)
        if true_base <= address < true_stop:
            public_address = FIXED_PUBLIC_BASE + address - true_base
        else:
            # External absolute locations are build fingerprints, not source semantics.
            public_address = 0
        return f"0x{public_address:x} <{symbol}>"

    scrubbed = TARGET_ANNOTATION.sub(replace_target, instruction)
    scrubbed = ANNOTATION.sub(
        lambda match: f"<{scrub_symbol(match.group(1), aliases, nested)}>", scrubbed
    )
    return re.sub(r"\s+", " ", scrubbed.strip())


def extract_candidate(
    aot: Path,
    *,
    architecture: str,
    readelf: Path,
    objdump_x64: Path,
    objdump_arm64: Path,
) -> tuple[str, dict[str, Any]]:
    symbols = symbol_rows(aot, "candidate", readelf)
    selected = symbols[0]
    address = selected["address"]
    size = max(1, selected["size"])
    stop = address + size
    objdump = objdump_x64 if architecture == "x86_64" else objdump_arm64
    command = [
        str(objdump),
        "-d",
        "--no-show-raw-insn",
        f"--start-address=0x{address:x}",
        f"--stop-address=0x{stop:x}",
    ]
    if architecture == "x86_64":
        command.insert(2, "-Mintel")
    command.append(str(aot))
    disassembly = run(command, cwd=aot.parent)
    aliases: dict[str, str] = {}
    nested: dict[str, str] = {}
    instructions: list[tuple[int, str]] = []
    for line in disassembly["stdout"].splitlines():
        match = INSTRUCTION_LINE.match(line)
        if match is None:
            continue
        true_address = int(match.group(1), 16)
        if not (address <= true_address < stop):
            continue
        instruction = match.group(2)
        if instruction.startswith((".byte", "(bad)")):
            continue
        public_address = FIXED_PUBLIC_BASE + true_address - address
        instructions.append(
            (
                public_address,
                scrub_and_rebase_instruction(
                    instruction,
                    true_base=address,
                    true_stop=stop,
                    aliases=aliases,
                    nested=nested,
                ),
            )
        )
    if not instructions:
        raise RuntimeError(f"no_candidate_instructions:{aot}")
    lines = [
        'All functions matching regular expression "candidate":',
        "",
        "Dump of assembler code for function candidate:",
    ]
    base = instructions[0][0]
    for public_address, instruction in instructions:
        lines.append(
            f"   0x{public_address:016x} <+{public_address-base}>:\t{instruction}"
        )
    lines.append("End of assembler dump.")
    assembly = "\n".join(lines) + "\n"
    allowed_annotation = re.compile(
        r"^(?:candidate(?:\+[^>]*)?|candidate\.local_\d+|"
        r"stub _iso_stub_[A-Za-z0-9_]+|dart:[A-Za-z0-9_:.+\-]+|"
        r"runtime_image|symbol_\d+|\+\d+)$"
    )
    unexpected = sorted(
        {
            match.group(1)
            for match in ANNOTATION.finditer(assembly)
            if not allowed_annotation.match(match.group(1))
        }
    )
    if unexpected:
        raise RuntimeError(f"unexpected_public_symbol_annotations:{unexpected}")
    return assembly, {
        "candidate_symbol_count": len(symbols),
        "selected_true_address": f"0x{address:x}",
        "selected_size": selected["size"],
        "selected_rule": "largest_size_then_lowest_address",
        "public_rebased_entry_address": f"0x{FIXED_PUBLIC_BASE:x}",
        "instruction_count": len(instructions),
        "untrusted_symbol_aliases": len(aliases),
        "nested_candidate_aliases": len(nested),
        "objdump_command": command,
    }


def public_graph(assembly: str, architecture: str) -> dict[str, Any]:
    graph, _ = build_record(
        {
            "architecture": architecture,
            "assembly": assembly,
            "function": "candidate",
            "flutter_function_symbol_ranges": [
                {"name": "candidate", "address": f"0x{FIXED_PUBLIC_BASE:x}", "size": 0}
            ],
        },
        max_block_instrs=20,
        max_dataflow_edges=0,
        extractor_hash=extractor_sha256(),
    )
    if not graph.get("integrity", {}).get("valid"):
        raise RuntimeError(
            f"graph_integrity_invalid:{architecture}:{graph.get('integrity')}"
        )
    return graph


def required_package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"required_python_package_missing:{name}") from error
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
        if not version:
            raise RuntimeError(f"required_python_package_version_unknown:{name}")
        return str(version)


def tool_record(path: Path, version_command: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": file_sha256(path.resolve()),
        "size_bytes": path.resolve().stat().st_size,
    }
    if version_command:
        version = run(version_command, cwd=ROOT, check=False)
        result["version_stdout"] = version["stdout"].strip()
        result["version_stderr"] = version["stderr"].strip()
        result["version_returncode"] = version["returncode"]
    return result


def jsonl_text(values: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        for value in values
    )


def atomic_write_text(path: Path, value: str) -> None:
    """Replace one publication file atomically; COMPLETE commits the file set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, jsonl_text(values))


def _quarantine_artifact_directory(
    path: Path, quarantine_root: Path, label: str
) -> None:
    if not path.exists():
        return
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{label}-{uuid.uuid4().hex}"
    os.replace(path, destination)


def validate_pair_receipt(
    receipt_path: Path,
    *,
    source_index: int,
    pair_slot: int,
    source_sha256: str,
    program_sha256: str,
    pair_id: str,
    contract_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema": PAIR_RECEIPT_SCHEMA,
        "source_release_index": source_index,
        "pair_slot": pair_slot,
        "source_sha256": source_sha256,
        "program_sha256": program_sha256,
        "semantic_pair_id": pair_id,
        "build_contract_sha256": contract_sha256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"receipt_contract_mismatch:{key}")
    public_rows = receipt.get("public_rows")
    model_rows = receipt.get("model_rows")
    private_rows = receipt.get("private_rows")
    if not all(
        isinstance(value, list) and len(value) == 2
        for value in (public_rows, model_rows, private_rows)
    ):
        raise ValueError("receipt_requires_two_architecture_rows")
    expected_architectures = ["x86_64", "aarch64"]
    if [row.get("architecture") for row in public_rows] != expected_architectures:
        raise ValueError("receipt_public_architecture_order")
    if [row.get("architecture") for row in private_rows] != expected_architectures:
        raise ValueError("receipt_private_architecture_order")
    if [row.get("architecture") for row in model_rows] != expected_architectures:
        raise ValueError("receipt_model_architecture_order")
    for public_row, model_row, private_row in zip(
        public_rows, model_rows, private_rows
    ):
        if set(model_row) != {"architecture", "assembly", "cfg", "edges"}:
            raise ValueError("receipt_model_allowlist_changed")
        if not public_row.get("integrity", {}).get("valid"):
            raise ValueError("receipt_graph_invalid")
        if private_row.get("source_sha256") != source_sha256:
            raise ValueError("receipt_source_hash_mismatch")
        if private_row.get("program_sha256") != program_sha256:
            raise ValueError("receipt_program_hash_mismatch")
        if public_row.get("semantic_pair_id") != pair_id:
            raise ValueError("receipt_public_pair_id_mismatch")
        if private_row.get("semantic_pair_id") != pair_id:
            raise ValueError("receipt_private_pair_id_mismatch")
        artifact = (output_dir / str(private_row.get("aot_private_path"))).resolve()
        try:
            artifact.relative_to(output_dir.resolve())
        except ValueError as error:
            raise ValueError("receipt_artifact_outside_output") from error
        if not artifact.is_file() or file_sha256(artifact) != private_row.get(
            "aot_sha256"
        ):
            raise ValueError("receipt_artifact_hash_mismatch")
        if sha256_text(public_row["assembly"]) != private_row.get(
            "public_assembly_sha256"
        ):
            raise ValueError("receipt_public_assembly_hash_mismatch")
        if stable_sha256(
            {"cfg": public_row["cfg"], "edges": public_row["edges"]}
        ) != private_row.get("public_graph_sha256"):
            raise ValueError("receipt_public_graph_hash_mismatch")
    program_path = receipt_path.parent / "program.dart"
    if not program_path.is_file() or file_sha256(program_path) != program_sha256:
        raise ValueError("receipt_program_artifact_hash_mismatch")
    return receipt


def build_pair_receipt(
    *,
    row: dict[str, Any],
    source_index: int,
    pair_slot: int,
    salt: bytes,
    contract_sha256: str,
    output_dir: Path,
    artifact_dir: Path,
    quarantine_artifact_dir: Path,
    dart: Path,
    dartaotruntime: Path,
    readelf: Path,
    objdump_x64: Path,
    objdump_arm64: Path,
    arm64_runtime: Path | None,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    source, tests, program = neutral_program(row)
    pair_id = semantic_pair_id(source, salt)
    source_sha256 = sha256_text(source)
    program_sha256 = sha256_text(program)
    pair_work = artifact_dir / pair_id
    receipt_path = pair_work / "pair_receipt.json"
    if resume and receipt_path.is_file():
        try:
            return (
                validate_pair_receipt(
                    receipt_path,
                    source_index=source_index,
                    pair_slot=pair_slot,
                    source_sha256=source_sha256,
                    program_sha256=program_sha256,
                    pair_id=pair_id,
                    contract_sha256=contract_sha256,
                    output_dir=output_dir,
                ),
                True,
            )
        except Exception:
            _quarantine_artifact_directory(
                pair_work, quarantine_artifact_dir, f"stale-{pair_id}"
            )
    elif pair_work.exists():
        _quarantine_artifact_directory(
            pair_work, quarantine_artifact_dir, f"stale-{pair_id}"
        )

    temporary = artifact_dir / f".{pair_id}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        program_path = temporary / "program.dart"
        program_path.write_text(program, encoding="utf-8", newline="\n")
        run([str(dart), "run", str(program_path)], cwd=temporary, timeout=60)
        program_file_sha256 = file_sha256(program_path)
        if program_file_sha256 != program_sha256:
            raise RuntimeError("program_write_hash_mismatch")
        public_rows: list[dict[str, Any]] = []
        model_rows: list[dict[str, Any]] = []
        private_rows: list[dict[str, Any]] = []
        for arch, dart_arch in (("x86_64", "x64"), ("aarch64", "arm64")):
            aot_path = temporary / f"candidate.{dart_arch}.aot"
            compile_command = [
                str(dart),
                "compile",
                "aot-snapshot",
                "--target-os=linux",
                f"--target-arch={dart_arch}",
                "-o",
                str(aot_path),
                str(program_path),
            ]
            compile_result = run(compile_command, cwd=temporary, timeout=240)
            if arch == "x86_64":
                runtime = run(
                    [str(dartaotruntime), str(aot_path)], cwd=temporary, timeout=60
                )
                runtime_status = "passed"
            elif arm64_runtime is not None:
                runtime = run(
                    [str(arm64_runtime), str(aot_path)], cwd=temporary, timeout=60
                )
                runtime_status = "passed"
            else:
                runtime = None
                runtime_status = "not_run_no_native_or_qemu_runtime"
            assembly, extraction = extract_candidate(
                aot_path,
                architecture=arch,
                readelf=readelf,
                objdump_x64=objdump_x64,
                objdump_arm64=objdump_arm64,
            )
            graph = public_graph(assembly, arch)
            public_row = {
                "schema": PUBLIC_SCHEMA,
                "semantic_pair_id": pair_id,
                "architecture": arch,
                "function": "candidate",
                "assembly": graph["assembly"],
                "cfg": graph["cfg"],
                "edges": graph["edges"],
                "graph_v2": graph["graph_v2"],
                "integrity": graph["integrity"],
            }
            model_row = {
                "architecture": arch,
                "assembly": graph["assembly"],
                "cfg": graph["cfg"],
                "edges": graph["edges"],
            }
            if set(model_row) != {"architecture", "assembly", "cfg", "edges"}:
                raise AssertionError("model row allowlist changed")
            final_aot_path = pair_work / aot_path.name
            private_row = {
                "schema": PRIVATE_SCHEMA,
                "model_row": None,
                "semantic_pair_id": pair_id,
                "pair_slot": pair_slot,
                "architecture": arch,
                "source_release_index": source_index,
                "source_release_task_id": row.get("task_id"),
                "evaluation_only_dart_function_signature": row.get(
                    "evaluation_only_dart_function_signature"
                ),
                "dart_source": source,
                "tests": tests,
                "source_sha256": source_sha256,
                "program_sha256": program_sha256,
                "aot_sha256": file_sha256(aot_path),
                "aot_size_bytes": aot_path.stat().st_size,
                "aot_private_path": str(final_aot_path.relative_to(output_dir)),
                "compile_command": compile_command,
                "compile_elapsed_seconds": compile_result["elapsed_seconds"],
                "runtime_status": runtime_status,
                "runtime_elapsed_seconds": (
                    None if runtime is None else runtime["elapsed_seconds"]
                ),
                "extraction": extraction,
                "public_assembly_sha256": sha256_text(graph["assembly"]),
                "public_graph_sha256": stable_sha256(
                    {"cfg": graph["cfg"], "edges": graph["edges"]}
                ),
            }
            public_rows.append(public_row)
            model_rows.append(model_row)
            private_rows.append(private_row)
        if len({item["source_sha256"] for item in private_rows}) != 1:
            raise RuntimeError("architecture_source_hash_mismatch")
        if len({item["program_sha256"] for item in private_rows}) != 1:
            raise RuntimeError("architecture_program_hash_mismatch")
        receipt = {
            "schema": PAIR_RECEIPT_SCHEMA,
            "build_contract_sha256": contract_sha256,
            "semantic_pair_id": pair_id,
            "pair_slot": pair_slot,
            "source_release_index": source_index,
            "source_sha256": source_sha256,
            "program_sha256": program_sha256,
            "public_rows": public_rows,
            "model_rows": model_rows,
            "private_rows": private_rows,
            "pair_summary": {
                "semantic_pair_id": pair_id,
                "pair_slot": pair_slot,
                "source_release_index_private": source_index,
                "source_sha256_private": source_sha256,
                "program_sha256_private": program_sha256,
                "jit_tests": "passed",
                "architectures": ["x86_64", "aarch64"],
                "aot_sha256_private": {
                    item["architecture"]: item["aot_sha256"] for item in private_rows
                },
            },
        }
        atomic_write_text(
            temporary / "pair_receipt.json",
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        )
        if pair_work.exists():
            _quarantine_artifact_directory(
                pair_work, quarantine_artifact_dir, f"stale-{pair_id}"
            )
        os.replace(temporary, pair_work)
        return receipt, False
    except Exception:
        _quarantine_artifact_directory(
            temporary, quarantine_artifact_dir, f"failed-{pair_id}"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--private-input", type=Path, required=True)
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument(
        "--indices",
        help="pilot mode: 2-8 distinct zero-based comma-separated row indices",
    )
    selectors.add_argument("--indices-file", type=Path)
    selectors.add_argument("--all", action="store_true", dest="all_rows")
    parser.add_argument(
        "--indices-file-format",
        choices=("zero-based", "alignment-jsonl"),
        default="zero-based",
    )
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dart", default="dart")
    parser.add_argument("--dartaotruntime", default="dartaotruntime")
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--objdump-x64", default="objdump")
    parser.add_argument("--objdump-arm64", default="aarch64-linux-gnu-objdump")
    parser.add_argument("--expected-dart-version", default="3.11.5")
    parser.add_argument("--pair-salt", type=Path, default=None)
    parser.add_argument("--arm64-runtime", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    networkx_version = required_package_version("networkx")
    args.private_input = args.private_input.resolve()
    args.output_dir = args.output_dir.resolve()
    row_count = input_row_count(args.private_input)
    if args.indices is not None:
        raw_values = [
            value.strip() for value in args.indices.split(",") if value.strip()
        ]
        try:
            all_indices = [int(value) for value in raw_values]
        except ValueError as error:
            raise ValueError("invalid_inline_indices") from error
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("duplicate_inline_indices")
        if not 2 <= len(all_indices) <= 8:
            raise ValueError("pilot requires 2-8 distinct rows")
        selector = {"mode": "inline", "path": None, "sha256": None}
    elif args.indices_file is not None:
        args.indices_file = args.indices_file.resolve()
        all_indices = load_indices_file(args.indices_file, args.indices_file_format)
        selector = {
            "mode": args.indices_file_format,
            "path": str(args.indices_file),
            "sha256": file_sha256(args.indices_file),
        }
    else:
        all_indices = list(range(row_count))
        selector = {"mode": "all", "path": None, "sha256": None}
    if any(index < 0 or index >= row_count for index in all_indices):
        raise ValueError(f"selected_index_out_of_range:row_count={row_count}")
    all_indices = sorted(all_indices)
    shard_indices, shard = select_shard(all_indices, args.shard_size, args.shard_index)
    selected_indices = set(shard_indices)
    global_pair_slots = {value: slot for slot, value in enumerate(all_indices)}

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("nonempty_output_requires_resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.output_dir / "private"
    artifact_dir = private_dir / "artifacts"
    quarantine_artifact_dir = private_dir / "quarantine_artifacts"
    private_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Always retain a private in-output salt copy so checksums and resume are
    # self-contained even when a shared external salt is supplied.
    salt_path = private_dir / "semantic_pair_salt"
    supplied_salt = (
        None if args.pair_salt is None else args.pair_salt.resolve().read_bytes()
    )
    if salt_path.exists():
        salt = salt_path.read_bytes()
        if supplied_salt is not None and supplied_salt != salt:
            raise ValueError("supplied_pair_salt_mismatch")
    elif supplied_salt is not None:
        salt = supplied_salt
        atomic_write_bytes(salt_path, salt)
    else:
        salt = secrets.token_bytes(32)
        atomic_write_bytes(salt_path, salt)
    if len(salt) < 32:
        raise ValueError("pair salt must contain at least 32 bytes")

    dart = executable(args.dart)
    dartaotruntime = executable(args.dartaotruntime)
    readelf = executable(args.readelf)
    objdump_x64 = executable(args.objdump_x64)
    objdump_arm64 = executable(args.objdump_arm64)
    dart_version = run([str(dart), "--version"], cwd=args.output_dir, check=False)
    dart_version_text = (dart_version["stderr"] or dart_version["stdout"]).strip()
    if args.expected_dart_version not in dart_version_text:
        raise RuntimeError(
            f"wrong_dart_version:expected {args.expected_dart_version!r}:got {dart_version_text!r}"
        )
    arm64_runtime = None
    if args.arm64_runtime is not None:
        arm64_runtime = executable(str(args.arm64_runtime))

    sdk_root = dart.parent.parent
    possible_sdk_tools = [
        sdk_root / "bin/utils/gen_snapshot",
        sdk_root / "bin/dartaotruntime",
        Path.home()
        / f".dart/dartdev/sdk_cache/{args.expected_dart_version}/gen_snapshot_linux_x64_linux_arm64",
        Path.home()
        / f".dart/dartdev/sdk_cache/{args.expected_dart_version}/dartaotruntime_linux_arm64",
    ]
    toolchain = {
        "dart_version": dart_version_text,
        "dart": tool_record(dart, [str(dart), "--version"]),
        "dartaotruntime_x64": tool_record(dartaotruntime),
        "readelf": tool_record(readelf, [str(readelf), "--version"]),
        "objdump_x64": tool_record(objdump_x64, [str(objdump_x64), "--version"]),
        "objdump_arm64": tool_record(objdump_arm64, [str(objdump_arm64), "--version"]),
        "sdk_target_tools": {
            str(path): tool_record(path) for path in possible_sdk_tools if path.exists()
        },
        "arm64_runtime": None if arm64_runtime is None else tool_record(arm64_runtime),
    }
    private_input_sha256 = file_sha256(args.private_input)
    build_contract = {
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "graph_builder_sha256": file_sha256(
            ROOT / "scripts/data/build_graph_v2_jsonl.py"
        ),
        "extractor_sha256": extractor_sha256(),
        "private_input_sha256": private_input_sha256,
        "pair_salt_sha256": sha256_bytes(salt),
        "symbol_policy_sha256": stable_sha256(SYMBOL_POLICY),
        "toolchain_sha256": stable_sha256(toolchain),
    }
    contract_sha256 = stable_sha256(build_contract)

    selected = load_selected(args.private_input, selected_indices)
    raw_pair_ids: dict[str, int] = {}
    for source_index, row in selected:
        raw_source = str(row.get("dart_source") or "")
        pair_id = semantic_pair_id(raw_source, salt)
        if pair_id in raw_pair_ids:
            raise RuntimeError(
                f"duplicate_semantic_pair_id:{raw_pair_ids[pair_id]}:{source_index}"
            )
        raw_pair_ids[pair_id] = source_index

    receipts: list[dict[str, Any]] = []
    quarantines: list[dict[str, Any]] = []
    resumed_pairs = 0
    built_pairs = 0
    for source_index, row in selected:
        pair_slot = global_pair_slots[source_index]
        try:
            receipt, resumed = build_pair_receipt(
                row=row,
                source_index=source_index,
                pair_slot=pair_slot,
                salt=salt,
                contract_sha256=contract_sha256,
                output_dir=args.output_dir,
                artifact_dir=artifact_dir,
                quarantine_artifact_dir=quarantine_artifact_dir,
                dart=dart,
                dartaotruntime=dartaotruntime,
                readelf=readelf,
                objdump_x64=objdump_x64,
                objdump_arm64=objdump_arm64,
                arm64_runtime=arm64_runtime,
                resume=args.resume,
            )
            receipts.append(receipt)
            resumed_pairs += int(resumed)
            built_pairs += int(not resumed)
        except Exception as error:
            message = str(error)
            quarantines.append(
                {
                    "source_release_index": source_index,
                    "pair_slot": pair_slot,
                    "source_release_task_id": row.get("task_id"),
                    "error_type": type(error).__name__,
                    "error_message": message[-8000:],
                    "error_sha256": sha256_text(message),
                }
            )

    receipts.sort(key=lambda item: item["pair_slot"])
    public_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for receipt in receipts:
        for public_row, model_row, private_row in zip(
            receipt["public_rows"], receipt["model_rows"], receipt["private_rows"]
        ):
            private_row = dict(private_row)
            private_row["model_row"] = len(model_rows)
            public_rows.append(public_row)
            model_rows.append(model_row)
            private_rows.append(private_row)
        pair_summaries.append(receipt["pair_summary"])

    public_path = args.output_dir / "paired_public.jsonl"
    model_path = args.output_dir / "model_public.jsonl"
    private_path = private_dir / "paired_private.jsonl"
    quarantine_path = private_dir / "quarantine.jsonl"
    public_text = jsonl_text(public_rows)
    model_text = jsonl_text(model_rows)
    private_text = jsonl_text(private_rows)
    quarantine_text = jsonl_text(
        sorted(quarantines, key=lambda item: item["pair_slot"])
    )
    public_blob = public_text + model_text
    forbidden_values = [
        str(row.get("task_id") or "") for _, row in selected if row.get("task_id")
    ]
    forbidden_values += [
        str(row.get("evaluation_only_dart_function_signature") or "")
        for _, row in selected
        if row.get("evaluation_only_dart_function_signature")
    ]
    residues = sorted(
        value for value in forbidden_values if value and value in public_blob
    )
    forbidden_terms = sorted(
        term
        for term in (
            "dart_source",
            '"tests"',
            "evaluation_only_dart_function_signature",
            "file://",
            "sigless_",
        )
        if term in public_blob
    )
    if residues or forbidden_terms:
        raise RuntimeError(f"public_leakage:values={residues}:terms={forbidden_terms}")
    manifest = {
        "schema": SCHEMA,
        "created_utc_epoch": int(time.time()),
        "builder": {
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "graph_builder_sha256": file_sha256(
                ROOT / "scripts/data/build_graph_v2_jsonl.py"
            ),
            "python": sys.version,
            "networkx_version": networkx_version,
        },
        "source_policy": "same byte-exact neutral candidate program compiled once per ISA",
        "pair_id_policy": PAIR_ID_POLICY,
        "pair_id_salt": "private/semantic_pair_salt (withheld from public artifacts)",
        "symbol_policy": SYMBOL_POLICY,
        "symbol_policy_sha256": stable_sha256(SYMBOL_POLICY),
        "public_address_policy": f"candidate entry rebased to 0x{FIXED_PUBLIC_BASE:x}; external absolute targets zeroed",
        "private_input": {
            "path": str(args.private_input),
            "sha256": private_input_sha256,
            "row_count": row_count,
            "selector": selector,
            "selection_sha256": stable_sha256(all_indices),
            "selection_count": len(all_indices),
            "shard": shard,
            "selected_zero_based_indices": shard_indices,
        },
        "build_contract": build_contract,
        "build_contract_sha256": contract_sha256,
        "extractor": {
            "schema": GRAPH_SCHEMA_VERSION,
            "combined_cfg_dfg_sha256": extractor_sha256(),
            "cfg_sha256": file_sha256(ROOT / "scripts/data/cfg_extractor.py"),
            "dfg_sha256": file_sha256(ROOT / "scripts/data/dfg_extractor.py"),
            "max_block_instructions": 20,
            "max_dfg_edges": 0,
        },
        "toolchain": toolchain,
        "runtime_validation": {
            "jit_x64_all_pairs": (
                "passed" if not quarantines else "incomplete_quarantined"
            ),
            "aot_x64_all_pairs": (
                "passed" if not quarantines else "incomplete_quarantined"
            ),
            "jit_x64_all_completed_pairs": "passed",
            "aot_x64_all_completed_pairs": "passed",
            "aot_arm64": (
                "passed"
                if arm64_runtime is not None
                else "not_run_no_native_or_qemu_runtime"
            ),
        },
        "counts": {
            "selected_semantic_pairs": len(selected),
            "semantic_pairs": len(pair_summaries),
            "built_pairs": built_pairs,
            "resumed_pairs": resumed_pairs,
            "quarantined_pairs": len(quarantines),
            "public_architecture_rows": len(public_rows),
            "private_architecture_rows": len(private_rows),
            "x86_64_rows": sum(row["architecture"] == "x86_64" for row in public_rows),
            "aarch64_rows": sum(
                row["architecture"] == "aarch64" for row in public_rows
            ),
        },
        "pair_summaries_private": pair_summaries,
        "quarantines_private": quarantines,
        "public_leakage_gate": {
            "forbidden_value_residues": residues,
            "forbidden_term_residues": forbidden_terms,
            "passed": not residues and not forbidden_terms,
        },
    }
    manifest_path = private_dir / "build_manifest.json"
    public_manifest = {
        "schema": SCHEMA,
        "builder": manifest["builder"],
        "counts": manifest["counts"],
        "extractor": manifest["extractor"],
        "toolchain_sha256": stable_sha256(toolchain),
        "symbol_policy": SYMBOL_POLICY,
        "symbol_policy_sha256": manifest["symbol_policy_sha256"],
        "public_address_policy": manifest["public_address_policy"],
        "runtime_validation": manifest["runtime_validation"],
        "public_leakage_gate": manifest["public_leakage_gate"],
        "publication": {
            "complete": not quarantines,
            "selected_semantic_pairs": len(selected),
            "completed_semantic_pairs": len(pair_summaries),
            "quarantined_semantic_pairs": len(quarantines),
        },
    }
    public_manifest_path = args.output_dir / "public_manifest.json"
    complete_path = args.output_dir / "COMPLETE"
    if complete_path.exists():
        complete_path.unlink()
    atomic_write_text(public_path, public_text)
    atomic_write_text(model_path, model_text)
    atomic_write_text(private_path, private_text)
    atomic_write_text(quarantine_path, quarantine_text)
    atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    atomic_write_text(
        public_manifest_path,
        json.dumps(public_manifest, indent=2, sort_keys=True) + "\n",
    )
    checksum_paths = [
        public_path,
        model_path,
        public_manifest_path,
        private_path,
        manifest_path,
        quarantine_path,
        salt_path,
    ] + sorted(artifact_dir.rglob("*"))
    checksum_paths = [path for path in checksum_paths if path.is_file()]
    checksums_path = private_dir / "SHA256SUMS.txt"
    checksums_text = "".join(
        f"{file_sha256(path)}  {path.relative_to(args.output_dir).as_posix()}\n"
        for path in checksum_paths
    )
    atomic_write_text(checksums_path, checksums_text)
    if len(public_path.read_text(encoding="utf-8").splitlines()) != 2 * len(receipts):
        raise RuntimeError("published_public_row_count_mismatch")
    if len(model_path.read_text(encoding="utf-8").splitlines()) != 2 * len(receipts):
        raise RuntimeError("published_model_row_count_mismatch")
    for line in checksums_text.splitlines():
        expected_hash, relative_path = line.split("  ", 1)
        if file_sha256(args.output_dir / relative_path) != expected_hash:
            raise RuntimeError(f"published_checksum_mismatch:{relative_path}")
    if not quarantines:
        atomic_write_text(
            complete_path,
            json.dumps(
                {
                    "schema": SCHEMA,
                    "build_contract_sha256": contract_sha256,
                    "public_manifest_sha256": file_sha256(public_manifest_path),
                    "checksums_sha256": file_sha256(checksums_path),
                },
                sort_keys=True,
            )
            + "\n",
        )
    print(json.dumps(public_manifest, indent=2, sort_keys=True))
    if quarantines:
        raise RuntimeError(f"quarantined_pairs:{len(quarantines)}")


if __name__ == "__main__":
    main()
