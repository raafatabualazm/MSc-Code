"""Recompile source rows and recover complete Dart AOT GDB disassembly.

Historical SFT assembly was captured through a helper that retained only the
last 4,000 subprocess-output characters. Large functions therefore kept an
`End of assembler dump.` marker while losing their entry and branch targets.
This tool rebuilds every row from source, retains the complete GDB stream,
validates CFG closure, and writes an atomic JSONL plus a provenance report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.cfg_extractor import AssemblyCFGExtractor


SCHEMA = "antigravity-sft-assembly-rebuild-v1"
STANDARD_IMPORTS = (
    "import 'dart:async';",
    "import 'dart:collection';",
    "import 'dart:convert';",
    "import 'dart:io';",
    "import 'dart:isolate';",
    "import 'dart:math';",
    "import 'dart:typed_data';",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_source_sha256(row: dict[str, Any]) -> str:
    source = str(row.get("source") or row.get("dart_source") or "")
    return sha256_text(re.sub(r"\s+", " ", source).strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def private_dart_env(workdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = workdir / ".dart_home"
    appdata = home / "AppData" / "Roaming"
    localappdata = home / "AppData" / "Local"
    pub_cache = home / ".pub-cache"
    for path in (home, appdata, localappdata, pub_cache):
        path.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(appdata),
            "LOCALAPPDATA": str(localappdata),
            "PUB_CACHE": str(pub_cache),
            "CI": "true",
            "DART_SUPPRESS_ANALYTICS": "1",
        }
    )
    return env


def run_full(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def run_with_retries(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str],
    retries: int,
) -> subprocess.CompletedProcess[str]:
    last_timeout: subprocess.TimeoutExpired | None = None
    for _ in range(max(0, retries) + 1):
        try:
            return run_full(command, cwd=cwd, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
    assert last_timeout is not None
    raise last_timeout


def function_name(row: dict[str, Any]) -> str:
    value = row.get("name") or row.get("function")
    if value:
        return str(value)
    source = str(row.get("source") or row.get("dart_source") or "")
    match = re.search(r"\b(?:void|[A-Za-z_]\w*(?:<[^>]+>)?)\s+([A-Za-z_]\w*)\s*\(", source)
    if not match:
        raise ValueError("cannot infer function name")
    return match.group(1)


def prepare_compilable_source(source: str, target_function: str) -> str:
    """Preserve a library fragment while making it AOT-compilable.

    The historical SFT corpus mixes complete programs and bare top-level
    functions whose original import/driver context was discarded. Standard
    imports are harmless when unused. The entry-point pragma prevents the
    target from being tree-shaken; an empty `main` satisfies the AOT compiler.
    """
    existing_imports = set(
        line.strip()
        for line in re.findall(r"^\s*import\s+[^;]+;\s*$", source, flags=re.MULTILINE)
    )
    imports = [line for line in STANDARD_IMPORTS if line not in existing_imports]
    body = source.strip()
    target_pattern = re.compile(
        rf"^(?P<indent>\s*)(?P<decl>"
        rf"(?:(?:[A-Za-z_]\w*(?:<[^>]+>)?|void)\s+)?"
        rf"{re.escape(target_function)}\s*\()",
        flags=re.MULTILINE,
    )
    match = target_pattern.search(body)
    preceding = body[: match.start()] if match else ""
    already_annotated = bool(
        re.search(r"@pragma\(['\"]vm:entry-point['\"]\)\s*$", preceding)
    )
    if match and not already_annotated:
        body = (
            body[: match.start()]
            + match.group("indent")
            + "@pragma('vm:entry-point')\n"
            + match.group("indent")
            + body[match.start("decl") :]
        )
    if target_function != "main" and not re.search(r"^\s*void\s+main\s*\(", body, re.MULTILINE):
        body += "\n\nvoid main() {}"
    return "\n".join(imports) + ("\n\n" if imports else "") + body + "\n"


def compile_and_disassemble(
    row: dict[str, Any],
    *,
    index: int,
    dart_bin: str,
    gdb_bin: str,
    timeout: int,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    source = str(row.get("source") or row.get("dart_source") or "")
    if not source.strip():
        raise ValueError("missing source/dart_source")
    target_function = function_name(row)
    compilable_source = prepare_compilable_source(source, target_function)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_function)[:48] or "function"

    with tempfile.TemporaryDirectory(prefix=f"ag_rebuild_{index:05d}_{safe_name}_") as raw_tmp:
        tmp = Path(raw_tmp).resolve()
        source_path = tmp / "program.dart"
        aot_path = tmp / "program.aot"
        source_path.write_text(compilable_source, encoding="utf-8")
        env = private_dart_env(tmp)

        compiled = run_with_retries(
            [dart_bin, "compile", "aot-snapshot", str(source_path), "-o", str(aot_path)],
            cwd=tmp,
            timeout=timeout,
            env=env,
            retries=retries,
        )
        if compiled.returncode != 0:
            diagnostic = (compiled.stderr or compiled.stdout or "dart compile failed").strip()
            raise RuntimeError(f"dart compile failed: {diagnostic[:1000]}")

        gdb_target = aot_path.as_posix().replace('"', '\\"')
        disassembled = run_with_retries(
            [
                gdb_bin,
                "-batch",
                "-q",
                "-nx",
                "-ex",
                "set pagination off",
                "-ex",
                "set disassembly-flavor intel",
                "-ex",
                f'file "{gdb_target}"',
                "-ex",
                f"info functions {target_function}",
                "-ex",
                f"disassemble {target_function}",
            ],
            cwd=tmp,
            timeout=timeout,
            env=env,
            retries=retries,
        )
        assembly = (disassembled.stdout or "") + (disassembled.stderr or "")
        if disassembled.returncode != 0:
            raise RuntimeError(f"gdb failed: {assembly[:1000]}")
        marker = assembly.find("All functions matching")
        if marker < 0:
            marker = assembly.find("Dump of assembler code")
        if marker < 0 or "End of assembler dump." not in assembly:
            raise RuntimeError(f"gdb did not return a complete disassembly: {assembly[:1000]}")
        assembly = assembly[marker:].strip()

        blocks, edges, integrity = AssemblyCFGExtractor(assembly).build_blocks()
        if not integrity.get("valid"):
            raise RuntimeError(f"rebuilt CFG integrity failed: {integrity}")
        return assembly, {
            "function": target_function,
            "blocks": len(blocks),
            "cfg_edges": len(edges),
            "parsed_instructions": int(integrity.get("parsed_instruction_count", 0)),
            "external_direct_branches": int(
                integrity.get("external_direct_branch_count") or 0
            ),
            "pruned_unreachable_blocks": int(
                integrity.get("pruned_unreachable_block_count") or 0
            ),
        }


def rebuild_one(
    item: tuple[int, dict[str, Any]],
    *,
    dart_bin: str,
    gdb_bin: str,
    timeout: int,
    dart_version: str,
    gdb_version: str,
    retries: int,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    index, row = item
    try:
        assembly, graph = compile_and_disassemble(
            row,
            index=index,
            dart_bin=dart_bin,
            gdb_bin=gdb_bin,
            timeout=timeout,
            retries=retries,
        )
        rebuilt = dict(row)
        rebuilt["assembly"] = assembly
        rebuilt["assembly_rebuild"] = {
            "schema": SCHEMA,
            "source_sha256": sha256_text(str(row.get("source") or row.get("dart_source") or "")),
            "input_line_number": index + 1,
            "assembly_sha256": sha256_text(assembly),
            "dart_version": dart_version,
            "gdb_version": gdb_version,
            **graph,
        }
        return index, rebuilt, None
    except Exception as exc:
        return index, None, {
            "line_number": index + 1,
            "name": row.get("name") or row.get("function"),
            "error": str(exc),
        }


def tool_version(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return ((result.stdout or "") + (result.stderr or "")).strip().splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dart_bin", default="dart")
    parser.add_argument("--gdb_bin", default="gdb")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected_input_rows", type=int, default=0)
    parser.add_argument("--expected_output_rows", type=int, default=0)
    parser.add_argument("--allow_rejects", action="store_true")
    parser.add_argument("--dedupe_source", action="store_true")
    parser.add_argument(
        "--include_lines_from",
        type=Path,
        default=None,
        help="Optional JSON/JSONL rejection manifest containing 1-based line_number values",
    )
    args = parser.parse_args()

    dart_bin = shutil.which(args.dart_bin) or args.dart_bin
    gdb_bin = shutil.which(args.gdb_bin) or args.gdb_bin
    if not Path(dart_bin).exists() and not shutil.which(dart_bin):
        raise SystemExit(f"dart not found: {args.dart_bin}")
    if not Path(gdb_bin).exists() and not shutil.which(gdb_bin):
        raise SystemExit(f"gdb not found: {args.gdb_bin}")

    rows = read_jsonl(args.input)
    total_input_rows = len(rows)
    if args.expected_input_rows and total_input_rows != args.expected_input_rows:
        raise SystemExit(f"input row mismatch: {total_input_rows} != {args.expected_input_rows}")
    items = list(enumerate(rows))
    if args.include_lines_from:
        raw_manifest = args.include_lines_from.read_text(encoding="utf-8-sig").strip()
        if raw_manifest.startswith("["):
            manifest_rows = json.loads(raw_manifest)
        else:
            manifest_rows = [json.loads(line) for line in raw_manifest.splitlines() if line.strip()]
        selected_lines = {int(item["line_number"]) for item in manifest_rows}
        items = [item for item in items if item[0] + 1 in selected_lines]
        missing_lines = selected_lines - {item[0] + 1 for item in items}
        if missing_lines:
            raise SystemExit(f"rejection manifest lines not found in input: {sorted(missing_lines)}")
    duplicate_source_rows = 0
    if args.dedupe_source:
        seen_sources: set[str] = set()
        unique_items: list[tuple[int, dict[str, Any]]] = []
        for item in items:
            source_hash = normalized_source_sha256(item[1])
            if source_hash in seen_sources:
                duplicate_source_rows += 1
                continue
            seen_sources.add(source_hash)
            unique_items.append(item)
        items = unique_items
    if args.limit > 0:
        items = items[: args.limit]

    os.environ["GRAPH_MAX_BLOCK_INSTRS"] = "24"
    dart_version = tool_version([dart_bin, "--version"])
    gdb_version = tool_version([gdb_bin, "--version"])
    accepted: list[tuple[int, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                rebuild_one,
                item,
                dart_bin=dart_bin,
                gdb_bin=gdb_bin,
                timeout=args.timeout,
                dart_version=dart_version,
                gdb_version=gdb_version,
                retries=args.retries,
            )
            for item in items
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, rebuilt, failure = future.result()
            if rebuilt is not None:
                accepted.append((index, rebuilt))
            if failure is not None:
                rejected.append(failure)
                print(
                    f"REJECT line={failure['line_number']} name={failure.get('name')}: "
                    f"{failure['error'][:500]}",
                    flush=True,
                )
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"[{completed}/{len(futures)}] accepted={len(accepted)} rejected={len(rejected)}",
                    flush=True,
                )

    accepted.sort(key=lambda pair: pair[0])
    rejected.sort(key=lambda item: int(item["line_number"]))
    output_rows = [row for _, row in accepted]
    if rejected and not args.allow_rejects:
        args.rejected.parent.mkdir(parents=True, exist_ok=True)
        args.rejected.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(rejected)} rows failed; see {args.rejected}; output was not replaced")
    if args.expected_output_rows and len(output_rows) != args.expected_output_rows:
        raise SystemExit(f"output row mismatch: {len(output_rows)} != {args.expected_output_rows}")

    write_jsonl_atomic(args.output, output_rows)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    rebuilt_lines = [len(str(row["assembly"]).splitlines()) for row in output_rows]
    report = {
        "schema": SCHEMA,
        "input": str(args.input),
        "input_rows": total_input_rows,
        "selected_rows": len(items),
        "dedupe_source": args.dedupe_source,
        "duplicate_source_rows": duplicate_source_rows,
        "input_sha256": file_sha256(args.input),
        "output": str(args.output),
        "output_rows": len(output_rows),
        "output_sha256": file_sha256(args.output),
        "rejected": str(args.rejected),
        "rejected_rows": len(rejected),
        "dart_version": dart_version,
        "gdb_version": gdb_version,
        "workers": args.workers,
        "retries": args.retries,
        "assembly_lines_min": min(rebuilt_lines) if rebuilt_lines else 0,
        "assembly_lines_median": median(rebuilt_lines) if rebuilt_lines else 0,
        "assembly_lines_max": max(rebuilt_lines) if rebuilt_lines else 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
