"""Build a HumanEval-style synthetic Antigravity pool from raw Dart tasks.

Input rows should contain at least:
  function, dart_function_signature, dart_source, tests

This script:
  1. normalizes each row to the SFT/GRPO schema,
  2. verifies the reference source passes its own tests using the evaluator
     harness,
  3. compiles the reference with Dart,
  4. extracts real GDB assembly for the target function,
  5. writes a JSONL pool ready for validate_synthetic_pool.py and CFG build.

Run on Linux where both `dart` and `gdb` are available.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from rerank_predictions_antigravity import (  # noqa: E402
    _resolve_dart_binary,
    run_dart_tests,
    strip_main_and_imports,
    validate_dart_binary,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def infer_function(row: dict[str, Any]) -> str:
    if row.get("function"):
        return str(row["function"])
    tests = row.get("tests", "") or ""
    match = re.search(r"\bfinal\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;", tests)
    if match:
        return match.group(1)
    signature = row.get("dart_function_signature", "") or row.get("signature", "")
    match = re.search(r"\b(?:[A-Za-z_]\w*<[^>]+>|[A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(", signature)
    if match:
        return match.group(1)
    source = row.get("dart_source", "") or row.get("source", "")
    match = re.search(r"\b(?:[A-Za-z_]\w*<[^>]+>|[A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(", source)
    if match:
        return match.group(1)
    raise ValueError("cannot infer function name")


def infer_signature(source: str, function: str) -> str:
    pattern = re.compile(
        rf"^\s*(?:@pragma\([^\n]+\)\s*)?"
        rf"((?:[A-Za-z_]\w*(?:<[^>]+>)?|void)\s+{re.escape(function)}\s*\([^)]*\))",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if not match:
        raise ValueError(f"cannot infer Dart signature for {function}")
    return re.sub(r"\s+", " ", match.group(1)).strip()


def ensure_entrypoint(source: str, function: str) -> str:
    if "@pragma('vm:entry-point')" in source or '@pragma("vm:entry-point")' in source:
        return source.strip()
    match = re.search(
        rf"^(\s*)(?:[A-Za-z_]\w*(?:<[^;\n{{}}()]+>)?\??|void)\s+"
        rf"{re.escape(function)}\s*\(",
        source,
        flags=re.MULTILINE,
    )
    if not match:
        return source.strip()
    return (source[: match.start()] + "@pragma('vm:entry-point')\n" + source[match.start() :]).strip()


def imports_from(*texts: str) -> str:
    imports = sorted(
        set(
            line.strip()
            for text in texts
            for line in re.findall(r"^import\s+.*;\s*$", text or "", flags=re.MULTILINE)
        )
    )
    return "\n".join(imports)


def compile_and_disassemble(
    row: dict[str, Any],
    *,
    dart_bin: str,
    gdb_bin: str,
    timeout: int,
    keep_temps: bool,
) -> str:
    function = row["function"]
    source_body = strip_main_and_imports(row["dart_source"])
    imports = imports_from(row.get("dart_source", ""), row.get("tests", ""))
    tests = row.get("tests", "")
    full_source = (imports + "\n\n" if imports else "") + source_body + "\n\n" + tests

    tmp_obj = tempfile.TemporaryDirectory(prefix=f"ag_synth_{function}_", ignore_cleanup_errors=True)
    tmp = Path(tmp_obj.name)
    dart_path = tmp / "main.dart"
    exe_path = tmp / "main.exe"
    dart_path.write_text(full_source, encoding="utf-8")

    try:
        compile_result = subprocess.run(
            [dart_bin, "compile", "exe", str(dart_path), "-o", str(exe_path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if compile_result.returncode != 0:
            raise RuntimeError((compile_result.stderr or compile_result.stdout or "dart compile failed").strip())

        gdb_result = subprocess.run(
            [
                gdb_bin,
                "-batch",
                "-nx",
                "-ex",
                "set pagination off",
                "-ex",
                f"file {exe_path}",
                "-ex",
                f"info functions {function}",
                "-ex",
                f"disassemble {function}",
            ],
            cwd=tmp,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        assembly = (gdb_result.stdout or "") + ("\n" + gdb_result.stderr if gdb_result.stderr else "")
        if "Dump of assembler code" not in assembly or "End of assembler dump" not in assembly:
            raise RuntimeError(f"gdb did not disassemble {function}: {assembly[:500]}")
        return f'All functions matching regular expression "{function}":\n\n{assembly.strip()}\n'
    finally:
        if keep_temps:
            print(f"kept temp dir: {tmp}", file=sys.stderr)
        else:
            tmp_obj.cleanup()


def normalize_raw_row(row: dict[str, Any], index: int, prefix: str) -> dict[str, Any]:
    source = row.get("dart_source") or row.get("source") or row.get("reference") or ""
    if not source.strip():
        raise ValueError("missing dart_source/source/reference")
    tests = row.get("tests") or ""
    if not tests.strip():
        raise ValueError("missing tests")

    function = infer_function({**row, "dart_source": source, "tests": tests})
    source = ensure_entrypoint(source, function)
    signature = row.get("dart_function_signature") or row.get("signature") or infer_signature(source, function)
    task_id = str(row.get("task_id") or f"{prefix}_{index:06d}")

    return {
        **row,
        "filename": str(row.get("filename") or f"{task_id}.dart"),
        "function": function,
        "dart_function_signature": signature,
        "dart_source": source,
        "assembly": "",
        "lang": str(row.get("lang") or row.get("language") or "Dart"),
        "task_id": task_id,
        "tests": tests,
    }


def build_one(
    item: tuple[int, dict[str, Any]],
    *,
    prefix: str,
    dart_bin: str,
    gdb_bin: str,
    timeout: int,
    keep_temps: bool,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    index, raw = item
    try:
        row = normalize_raw_row(raw, index, prefix)
        ok, diagnostic = run_dart_tests(row["dart_source"], row["tests"], row["task_id"], dart_bin, timeout)
        if not ok:
            raise RuntimeError(f"reference tests failed: {diagnostic[:500]}")
        row["assembly"] = compile_and_disassemble(
            row,
            dart_bin=dart_bin,
            gdb_bin=gdb_bin,
            timeout=timeout,
            keep_temps=keep_temps,
        )
        return index, row, None
    except Exception as exc:
        return index, None, {
            "index": index,
            "task_id": raw.get("task_id"),
            "function": raw.get("function"),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, help="Raw HumanEval-style source/tests JSONL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejects", type=Path, default=None)
    parser.add_argument("--prefix", default="synthetic_he")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dart_bin", default=None)
    parser.add_argument("--gdb_bin", default="gdb")
    parser.add_argument("--keep_temps", action="store_true")
    args = parser.parse_args()

    dart_bin = _resolve_dart_binary(args.dart_bin)
    validate_dart_binary(dart_bin)
    if not shutil.which(args.gdb_bin):
        raise SystemExit(f"gdb not found: {args.gdb_bin}")

    raw_rows = read_jsonl(args.raw)
    if args.limit and args.limit > 0:
        raw_rows = raw_rows[: args.limit]

    rows: list[tuple[int, dict[str, Any]]] = []
    rejects: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                build_one,
                (idx, row),
                prefix=args.prefix,
                dart_bin=dart_bin,
                gdb_bin=args.gdb_bin,
                timeout=args.timeout,
                keep_temps=args.keep_temps,
            )
            for idx, row in enumerate(raw_rows)
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            idx, row, reject = future.result()
            if row is not None:
                rows.append((idx, row))
            if reject is not None:
                rejects.append(reject)
            if done % 25 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] accepted={len(rows)} rejected={len(rejects)}")

    rows.sort(key=lambda pair: pair[0])
    write_jsonl(args.output, [row for _, row in rows])
    print(json.dumps({"raw": len(raw_rows), "accepted": len(rows), "rejected": len(rejects), "output": str(args.output)}, indent=2))

    if args.rejects:
        args.rejects.parent.mkdir(parents=True, exist_ok=True)
        args.rejects.write_text(json.dumps(rejects, indent=2), encoding="utf-8")
        print(f"rejects: {args.rejects}")


if __name__ == "__main__":
    main()
