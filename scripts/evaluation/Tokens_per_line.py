#!/usr/bin/env python3
"""
Compute tokens-per-line statistics for Dart AOT assembly evaluation datasets.

Reads the same JSONL format used by compile_at_k.py and measures actual
token counts using a HuggingFace tokenizer (default: Qwen/Qwen3-8B).

Supports both .jsonl (one JSON object per line) and .json (array of objects).

Fields expected per entry (matching compile_at_k.py):
    language, source, assembly, function, filename

Usage:
    python tokens_per_line.py data/datasets/test-set2.jsonl
    python tokens_per_line.py data/datasets/test-set2.jsonl --model Qwen/Qwen3-8B
    python tokens_per_line.py data/datasets/test-set2.jsonl --strip-addr
    python tokens_per_line.py data/datasets/test-set2.jsonl --prompt          # include system+user prompt overhead
    python tokens_per_line.py data/datasets/test-set2.jsonl --per-line        # verbose per-line dump
    python tokens_per_line.py data/datasets/test-set2.jsonl -o stats.json     # JSON export
"""

import argparse
import json
import sys
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Prompts  (identical to compile_at_k.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a reverse engineering expert with advanced knowledge in assembly and {lang}. "
    "Please convert the following assembly code to idiomatic and clear {lang} code. "
    "Think carefully about the task and create a step-by-step chain of thoughts to ensure "
    "a logical and accurate response. "
    "Return your final answer inside a ```{lang_lower}``` code block."
)

USER_PROMPT = "Convert the following assembly to idiomatic {lang} code:\n\n```asm\n{assembly}\n```"

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_tokenizer = None


def get_tokenizer(model_name: str):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        print(f"Loaded tokenizer: {model_name}  (vocab={_tokenizer.vocab_size})")
    return _tokenizer


def count_tokens(text: str, tok) -> int:
    return len(tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Data loading  (JSON array  or  JSONL — one dict per line)
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    """Load .jsonl (one JSON per line) or .json (array)."""
    text = path.read_text(encoding="utf-8")

    # JSONL first (matches compile_at_k.py)
    if path.suffix == ".jsonl" or "\n{" in text[:4096]:
        entries = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {i}: {e}", file=sys.stderr)
        return entries

    # Fall back to JSON array / single object
    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    return data


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

# GDB format: "   0x00000000000914cd <+1333>:\tsqrtsd xmm1,xmm0"
ADDR_RE = re.compile(r"^\s*0x[0-9a-f]+\s+<\+\d+>:\t")


def strip_address_prefix(line: str) -> str:
    """Remove the hex-address + offset prefix, keep only the instruction."""
    return ADDR_RE.sub("", line)


def classify_line(line: str) -> str:
    """Rough classification of an assembly line for per-category stats."""
    stripped = line.strip()
    if stripped.startswith("Dump of assembler") or stripped.startswith("End of assembler"):
        return "header/footer"
    if not ADDR_RE.search(line):
        return "other"
    instr = ADDR_RE.sub("", line).strip()
    mnemonic = instr.split()[0].rstrip(",") if instr.split() else ""

    if mnemonic in ("push", "pop", "ret", "int3", "cdq", "cqo", "nop"):
        return "simple"
    if mnemonic.startswith("j") or mnemonic == "call":
        return "call_stub" if "stub" in instr else "branch/call"
    if "PTR" in instr:
        return "mem_access"
    if mnemonic == "lea":
        return "mem_access"
    if any(mnemonic.startswith(p) for p in (
        "movs", "movup", "movap", "muls", "adds", "subs", "divs",
        "sqrt", "comis", "cvts", "xorp",
    )):
        return "sse/fp"
    if any(mnemonic.startswith(p) for p in (
        "mov", "xor", "and", "or", "not", "test", "cmp",
        "add", "sub", "shr", "shl", "sar", "sal", "imul", "idiv", "neg",
    )):
        return "alu/reg"
    return "other"


# ---------------------------------------------------------------------------
# Per-function stats
# ---------------------------------------------------------------------------

@dataclass
class FunctionStats:
    function: str = ""
    filename: str = ""
    language: str = ""
    total_lines: int = 0
    total_tokens: int = 0
    min_tokens: int = 999_999
    max_tokens: int = 0
    prompt_tokens: int = 0          # system + user prompt (assembly included)
    source_tokens: int = 0          # reference Dart source
    source_lines: int = 0
    category_lines: dict = field(default_factory=lambda: defaultdict(int))
    category_tokens: dict = field(default_factory=lambda: defaultdict(int))
    per_line: list = field(default_factory=list)   # token count per asm line

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.total_lines if self.total_lines else 0.0


def analyse_entry(entry: dict, tok, strip_addr: bool, measure_prompt: bool) -> FunctionStats:
    lang     = entry.get("language", entry.get("lang", "Dart"))
    asm_text = entry.get("assembly", "")
    source   = entry.get("source", "")
    func     = entry.get("function", "?")
    fname    = entry.get("filename", "?")

    stats = FunctionStats(function=func, filename=fname, language=lang)

    # --- Assembly lines ---
    for raw_line in asm_text.splitlines():
        if not raw_line.strip():
            continue
        line = strip_address_prefix(raw_line) if strip_addr else raw_line
        n = count_tokens(line, tok)
        cat = classify_line(raw_line)

        stats.total_lines += 1
        stats.total_tokens += n
        stats.min_tokens = min(stats.min_tokens, n)
        stats.max_tokens = max(stats.max_tokens, n)
        stats.category_lines[cat] += 1
        stats.category_tokens[cat] += n
        stats.per_line.append(n)

    if stats.min_tokens == 999_999:
        stats.min_tokens = 0

    # --- Source tokens ---
    src_lines = [l for l in source.splitlines() if l.strip()]
    stats.source_lines = len(src_lines)
    if src_lines:
        stats.source_tokens = sum(count_tokens(l, tok) for l in src_lines)

    # --- Full prompt tokens (mirrors compile_at_k.py) ---
    if measure_prompt and asm_text:
        sys_msg = SYSTEM_PROMPT.format(lang=lang, lang_lower=lang.lower())
        usr_msg = USER_PROMPT.format(lang=lang, assembly=asm_text)
        stats.prompt_tokens = count_tokens(sys_msg, tok) + count_tokens(usr_msg, tok)

    return stats


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def percentile(data: list, p: float) -> float:
    """Simple nearest-rank percentile."""
    if not data:
        return 0.0
    s = sorted(data)
    k = int(len(s) * p / 100)
    return s[min(k, len(s) - 1)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Measure tokens-per-line for asm evaluation datasets"
    )
    ap.add_argument("dataset", type=Path,
                    help="Input .jsonl or .json (same format as compile_at_k.py --test-set)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HF tokenizer name (default: Qwen/Qwen3-8B)")
    ap.add_argument("--strip-addr", action="store_true",
                    help="Strip hex-address prefix before tokenizing")
    ap.add_argument("--prompt", action="store_true",
                    help="Also measure full prompt token count (system + user)")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Write detailed JSON stats to this file")
    ap.add_argument("--top", type=int, default=10,
                    help="Show top-N most expensive functions")
    ap.add_argument("--per-line", action="store_true",
                    help="Print every line's token count (verbose)")
    args = ap.parse_args()

    # ---- Load dataset ----
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} entries from {args.dataset}")

    tok = get_tokenizer(args.model)

    # ---- Analyse ----
    all_stats: list[FunctionStats] = []
    all_category_lines  = defaultdict(int)
    all_category_tokens = defaultdict(int)

    grand_asm_lines   = 0
    grand_asm_tokens  = 0
    grand_src_lines   = 0
    grand_src_tokens  = 0
    grand_prompt_tokens = 0
    all_line_tokens: list[int] = []       # every single line across all functions

    for i, entry in enumerate(data):
        fs = analyse_entry(entry, tok, args.strip_addr, args.prompt)
        all_stats.append(fs)

        grand_asm_lines   += fs.total_lines
        grand_asm_tokens  += fs.total_tokens
        grand_src_lines   += fs.source_lines
        grand_src_tokens  += fs.source_tokens
        grand_prompt_tokens += fs.prompt_tokens
        all_line_tokens.extend(fs.per_line)

        for cat in fs.category_lines:
            all_category_lines[cat]  += fs.category_lines[cat]
            all_category_tokens[cat] += fs.category_tokens[cat]

        if (i + 1) % 25 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] processed…")

    # ==================================================================
    # Report
    # ==================================================================
    mode = "stripped" if args.strip_addr else "raw (with address prefix)"
    avg  = grand_asm_tokens / grand_asm_lines if grand_asm_lines else 0

    print(f"\n{'='*65}")
    print(f"  ASSEMBLY — {mode}")
    print(f"{'='*65}")
    print(f"  Tokenizer          : {args.model}")
    print(f"  Functions analysed : {len(all_stats)}")
    print(f"  Total asm lines    : {grand_asm_lines:,}")
    print(f"  Total asm tokens   : {grand_asm_tokens:,}")
    if grand_asm_lines:
        print(f"  Avg tokens/line    : {avg:.2f}")
        print(f"  Median tokens/line : {percentile(all_line_tokens, 50):.0f}")
        print(f"  P10  tokens/line   : {percentile(all_line_tokens, 10):.0f}")
        print(f"  P90  tokens/line   : {percentile(all_line_tokens, 90):.0f}")
        print(f"  Min  tokens/line   : {min(all_line_tokens)}")
        print(f"  Max  tokens/line   : {max(all_line_tokens)}")
    print()

    # Histogram of tokens-per-line distribution
    if all_line_tokens:
        print(f"  Tokens/line distribution:")
        print()
        # Build buckets
        lo = min(all_line_tokens)
        hi = max(all_line_tokens)
        # Auto-size buckets: aim for ~12-15 rows
        bucket_size = max(1, (hi - lo + 1) // 14)
        if bucket_size <= 2:
            bucket_size = 1
        elif bucket_size <= 6:
            bucket_size = 5
        else:
            bucket_size = 10

        buckets: dict[int, int] = defaultdict(int)
        for t in all_line_tokens:
            b = (t // bucket_size) * bucket_size
            buckets[b] += 1

        max_count = max(buckets.values())
        BAR_WIDTH = 40

        for b in sorted(buckets):
            count = buckets[b]
            bar_len = int(count / max_count * BAR_WIDTH)
            pct = count / len(all_line_tokens) * 100
            if bucket_size == 1:
                label = f"{b:>4}"
            else:
                label = f"{b:>3}-{b + bucket_size - 1:<3}"
            print(f"  {label} │{'█' * bar_len}{'░' * (BAR_WIDTH - bar_len)}│ {count:>5} ({pct:5.1f}%)")

        print()

        # Cumulative: what % of lines fit in N tokens?
        sorted_tokens = sorted(all_line_tokens)
        print(f"  Cumulative:")
        for threshold in [10, 15, 20, 25, 30, 35, 40, 50, 60]:
            if threshold < lo:
                continue
            if threshold > hi + 10:
                break
            count = sum(1 for t in sorted_tokens if t <= threshold)
            pct = count / len(sorted_tokens) * 100
            print(f"    ≤{threshold:>2} tokens: {pct:6.1f}%  ({count:,}/{len(sorted_tokens):,})")
        print()

    # Per-category table
    print(f"  {'Category':<16} {'Lines':>7} {'Tokens':>8} {'Avg':>6}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*6}")
    for cat in sorted(all_category_lines, key=lambda c: -all_category_tokens[c]):
        nl = all_category_lines[cat]
        nt = all_category_tokens[cat]
        print(f"  {cat:<16} {nl:>7,} {nt:>8,} {nt/nl:>6.1f}")
    print()

    # Top-N most expensive functions (by total tokens)
    sorted_stats = sorted(all_stats, key=lambda s: -s.total_tokens)
    top_max = sorted_stats[0].total_tokens if sorted_stats else 1
    print(f"  Top-{args.top} functions by total asm tokens:")
    print(f"  {'Function':<30} {'Lines':>5} {'Tokens':>7} {'Avg':>5} {'Min':>4} {'Max':>4}  Distribution")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*5} {'-'*4} {'-'*4}  {'-'*24}")
    for fs in sorted_stats[:args.top]:
        bar_len = int(fs.total_tokens / top_max * 20)
        print(f"  {fs.function:<30} {fs.total_lines:>5,} {fs.total_tokens:>7,} "
              f"{fs.avg_tokens:>5.1f} {fs.min_tokens:>4} {fs.max_tokens:>4}  "
              f"{'█' * bar_len}{'░' * (20 - bar_len)}")
    print()

    # Per-function line-length profile (all functions, sorted by avg tokens/line)
    print(f"  All functions by avg tokens/line:")
    fn_by_avg = sorted(all_stats, key=lambda s: -s.avg_tokens)
    avg_max = fn_by_avg[0].avg_tokens if fn_by_avg else 1
    print(f"  {'Function':<30} {'Lines':>5} {'Avg':>5} {'Med':>4} {'P90':>4}  Profile")
    print(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*4} {'-'*4}  {'-'*24}")
    for fs in fn_by_avg:
        med = percentile(fs.per_line, 50) if fs.per_line else 0
        p90 = percentile(fs.per_line, 90) if fs.per_line else 0
        bar_len = int(fs.avg_tokens / avg_max * 20) if avg_max else 0
        print(f"  {fs.function:<30} {fs.total_lines:>5,} {fs.avg_tokens:>5.1f} "
              f"{med:>4.0f} {p90:>4.0f}  {'█' * bar_len}{'░' * (20 - bar_len)}")
    print()

    # Source side
    print(f"{'='*65}")
    print(f"  SOURCE (Dart)")
    print(f"{'='*65}")
    print(f"  Total source lines : {grand_src_lines:,}")
    print(f"  Total source tokens: {grand_src_tokens:,}")
    if grand_src_lines:
        print(f"  Avg tokens/line    : {grand_src_tokens/grand_src_lines:.2f}")
    print()

    if grand_src_tokens:
        print(f"  Asm/Source token ratio: {grand_asm_tokens/grand_src_tokens:.1f}x")
        print(f"  Asm/Source line ratio : {grand_asm_lines/grand_src_lines:.1f}x")
        print()

    # Full-prompt stats
    if args.prompt:
        prompt_list = [fs.prompt_tokens for fs in all_stats if fs.prompt_tokens > 0]
        print(f"{'='*65}")
        print(f"  FULL PROMPT (system + user, per compile_at_k.py)")
        print(f"{'='*65}")
        if prompt_list:
            print(f"  Samples measured   : {len(prompt_list)}")
            print(f"  Total prompt tokens: {grand_prompt_tokens:,}")
            print(f"  Avg prompt tokens  : {grand_prompt_tokens/len(prompt_list):,.0f}")
            print(f"  Min prompt tokens  : {min(prompt_list):,}")
            print(f"  Max prompt tokens  : {max(prompt_list):,}")
            print(f"  Median             : {percentile(prompt_list, 50):,.0f}")
            print(f"  P90                : {percentile(prompt_list, 90):,.0f}")
            print(f"  P95                : {percentile(prompt_list, 95):,.0f}")
        print()

    # Context budget table
    print(f"{'='*65}")
    if grand_asm_lines:
        print(f"  CONTEXT BUDGET (assembly only, avg tokens/line = {avg:.1f})")
    else:
        print(f"  CONTEXT BUDGET")
    print(f"{'='*65}")
    if grand_asm_lines:
        for ctx in [4096, 8192, 16384, 32768, 65536, 131072]:
            fit = int(ctx / avg)
            print(f"  {ctx//1024:>4}K context → ~{fit:,} asm lines")
    print()

    # Verbose per-line dump
    if args.per_line:
        print(f"{'='*65}")
        print(f"  PER-LINE TOKEN COUNTS")
        print(f"{'='*65}")
        for idx, (fs, entry) in enumerate(zip(all_stats, data)):
            print(f"\n  --- {fs.function} ({fs.filename}) ---")
            asm_lines = [l for l in entry.get("assembly", "").splitlines() if l.strip()]
            for n_tok, raw in zip(fs.per_line, asm_lines):
                tag = classify_line(raw)
                disp = strip_address_prefix(raw).strip() if args.strip_addr else raw.strip()
                print(f"  {n_tok:4d}  [{tag:<14}]  {disp[:80]}")

    # JSON export
    if args.output:
        out = {
            "model": args.model,
            "strip_addr": args.strip_addr,
            "measure_prompt": args.prompt,
            "summary": {
                "num_functions": len(all_stats),
                "total_asm_lines": grand_asm_lines,
                "total_asm_tokens": grand_asm_tokens,
                "avg_tokens_per_asm_line": round(avg, 3),
                "median_tokens_per_asm_line": percentile(all_line_tokens, 50),
                "p10_tokens_per_asm_line": percentile(all_line_tokens, 10),
                "p90_tokens_per_asm_line": percentile(all_line_tokens, 90),
                "total_src_lines": grand_src_lines,
                "total_src_tokens": grand_src_tokens,
                "total_prompt_tokens": grand_prompt_tokens,
            },
            "per_category": {
                cat: {
                    "lines": all_category_lines[cat],
                    "tokens": all_category_tokens[cat],
                    "avg": round(all_category_tokens[cat] / all_category_lines[cat], 3),
                }
                for cat in sorted(all_category_lines)
            },
            "per_function": [
                {
                    "function": fs.function,
                    "filename": fs.filename,
                    "language": fs.language,
                    "asm_lines": fs.total_lines,
                    "asm_tokens": fs.total_tokens,
                    "avg_tokens_per_line": round(fs.avg_tokens, 3),
                    "min_tokens": fs.min_tokens,
                    "max_tokens": fs.max_tokens,
                    "source_lines": fs.source_lines,
                    "source_tokens": fs.source_tokens,
                    "prompt_tokens": fs.prompt_tokens,
                }
                for fs in all_stats
            ],
        }
        args.output.write_text(json.dumps(out, indent=2))
        print(f"Detailed stats written to {args.output}")


if __name__ == "__main__":
    main()
