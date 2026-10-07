"""Strip interleaved source lines from the `assembly` field of a JSONL dataset.

Some corpora (notably dart_all.jsonl and test-set.jsonl) append the numbered
Dart SOURCE after the disassembly, e.g. lines like:

    2\tvoid main() {
    3\t  // ...

That leaks the target into the model's decoder prompt (the model can copy
instead of decompile). This removes those numbered-source lines while keeping
real disassembly and the gdb header lines.

Discriminator: a polluted source line is "<n>\\t<code>" (digits then a TAB).
Real disassembly starts with "0x..."; gdb declaration headers use "<n>:\\t"
(digits then a COLON), so both are preserved. Only the `assembly` field is
modified, so any existing cfg/edges (which were already built from the
disassembly only) remain valid.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SOURCE_LINE = re.compile(r"^\s*\d+\t")  # "<n>\t<source>"; NOT "<n>:\t" (gdb decl) or "0x..." (asm)


def clean_assembly(assembly: str) -> str:
    return "\n".join(
        line for line in assembly.splitlines() if not SOURCE_LINE.match(line)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = cleaned = removed_lines = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as f_in, args.output.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            assembly = record.get("assembly")
            if isinstance(assembly, str):
                new_assembly = clean_assembly(assembly)
                if new_assembly != assembly:
                    cleaned += 1
                    removed_lines += len(assembly.splitlines()) - len(new_assembly.splitlines())
                record["assembly"] = new_assembly
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "rows": rows,
        "rows_cleaned": cleaned,
        "source_lines_removed": removed_lines,
    }, indent=2))


if __name__ == "__main__":
    main()
