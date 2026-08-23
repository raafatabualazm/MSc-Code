"""Strip GDB per-line address columns from the `assembly` field of a JSONL.

Produces the CLEAN representation for the 02a3 long-budget re-test: each
instruction line loses ONLY its leading `0x... <+N>:` prefix; every other
byte (headers with the function name, `<callee>` comments, GDB spacing) is
preserved so the runner's usual GRAPH_PROMPT_CLEAN_ASM pass sees the same
content it saw in training, minus the address column that dominates the
token budget (~28 of ~40 tok/instr on x86-64 GDB dumps).

All other row fields (tests, cfg, hybrid_metadata.evaluation_only, ...) are
copied through untouched.

Usage:
  python strip_asm_addresses_antigravity.py --input in.jsonl --output out.jsonl \
      [--report report.json]
"""
import argparse
import json
import re
import sys

_ADDR_PREFIX = re.compile(r"^\s*0x[0-9a-fA-F]+\s+<\+\d+>:\s*(.*)$")


def strip_addresses(assembly: str):
    """Return (clean_text, instruction_lines, stripped_lines)."""
    out = []
    instr = 0
    stripped = 0
    for ln in assembly.splitlines():
        m = _ADDR_PREFIX.match(ln)
        if m:
            out.append(m.group(1))
            instr += 1
            stripped += 1
        else:
            out.append(ln)
    return "\n".join(out), instr, stripped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))

    stats = []
    with open(args.output, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            asm = row.get("assembly") or ""
            clean, instr, stripped = strip_addresses(asm)
            if instr == 0:
                print(f"WARNING row {i}: no GDB instruction lines matched", file=sys.stderr)
            row = dict(row)
            row["assembly"] = clean
            marked = bool(((row.get("hybrid_metadata") or {}).get("evaluation_only")) is True)
            stats.append({
                "index": i,
                "task_id": row.get("task_id"),
                "instr_lines": instr,
                "raw_chars": len(asm),
                "clean_chars": len(clean),
                "evaluation_only": marked,
            })
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(stats)
    marked = sum(1 for s in stats if s["evaluation_only"])
    raw_c = sum(s["raw_chars"] for s in stats)
    clean_c = sum(s["clean_chars"] for s in stats)
    print(f"rows={n} evaluation_only_marked={marked}")
    print(f"chars: raw={raw_c} clean={clean_c} ratio={clean_c / max(1, raw_c):.3f}")
    print(f"min instr_lines={min(s['instr_lines'] for s in stats)} "
          f"max={max(s['instr_lines'] for s in stats)}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"rows": n, "evaluation_only_marked": marked,
                       "raw_chars": raw_c, "clean_chars": clean_c,
                       "per_row": stats}, fh, indent=2)
    # fail loudly if any row lost its evaluation_only marking or matched nothing
    if marked != n:
        print("ERROR: not every row is marked evaluation_only", file=sys.stderr)
        return 1
    if any(s["instr_lines"] == 0 for s in stats):
        print("ERROR: some rows had zero GDB instruction lines", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
