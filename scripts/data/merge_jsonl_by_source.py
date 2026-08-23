"""Merge JSONL files deterministically while removing repeated source programs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any


def source_hash(row: dict[str, Any]) -> str:
    source = str(row.get("source") or row.get("dart_source") or "")
    normalized = re.sub(r"\s+", " ", source).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected_output_rows", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    input_rows = 0
    duplicate_rows = 0
    inputs = []
    for path in args.input:
        path_rows = 0
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                path_rows += 1
                input_rows += 1
                row = json.loads(line)
                digest = source_hash(row)
                if digest in seen:
                    duplicate_rows += 1
                    continue
                seen.add(digest)
                rows.append(row)
        inputs.append({"path": str(path), "rows": path_rows, "sha256": file_hash(path)})

    if args.expected_output_rows and len(rows) != args.expected_output_rows:
        raise SystemExit(f"output row mismatch: {len(rows)} != {args.expected_output_rows}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=args.output.parent,
        prefix=args.output.name + ".",
        suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)

    report = {
        "schema": "antigravity-source-deduplicated-merge-v1",
        "inputs": inputs,
        "input_rows": input_rows,
        "duplicate_source_rows": duplicate_rows,
        "output": str(args.output),
        "output_rows": len(rows),
        "output_sha256": file_hash(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
