"""Measure prompt/target sizes for grpo_data.jsonl to size the GRPO padding gap
and the 256-token rollout truncation risk. Mirrors build_decoder_prompt() from
graph_encoder_decoder_decompiler_v2_antigravity.py without importing torch.
"""

import json
import re
import statistics
import sys
from pathlib import Path


def _build_test_call_hint(test_code: str, max_lines: int = 10) -> str:
    lines = []
    for line in (test_code or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("void expect") or stripped.startswith("void expectList") or stripped.startswith("void expectMap"):
            break
        if "candidate" in stripped or stripped.startswith("final candidate"):
            lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def build_decoder_prompt(record: dict) -> str:
    language = record.get("language", record.get("lang", "Dart"))
    assembly = (record.get("assembly") or "").strip()
    function_name = (
        record.get("function")
        or record.get("name")
        or record.get("camel_case_function_name")
        or record.get("python_function_name")
        or ""
    )
    signature = (
        record.get("dart_function_signature")
        or record.get("function_signature")
        or record.get("signature")
        or ""
    )
    tests = _build_test_call_hint(record.get("tests", ""), max_lines=10)

    parts = [
        f"Convert the following assembly to {language} source code.",
        "Return only valid source code.",
        "Do not include explanations, markdown fences, test code, or placeholder demos.",
        "Include any required imports at the top, for example dart:io or dart:math.",
        "If you call a helper function, define that helper in the same output.",
    ]
    if signature:
        parts.append(f"Implement this exact top-level Dart signature: {signature}.")
        parts.append("Do not replace it with only a void main() demo.")
        parts.append("Do not define the required function inside main(); define it at top level.")
    elif function_name:
        parts.append(f"Target function name: {function_name}.")
    if tests:
        parts.extend(["", "Unit-test harness excerpt, for signature/call-shape only:", tests])
    parts.extend(["", "Assembly:", assembly, "", f"{language} code:"])
    return "\n".join(parts)


def _remove_top_level_main(code: str) -> str:
    main_match = re.search(
        r"^\s*(?:\w+(?:<[^>]*>)?\s+)?main\s*\([^)]*\)\s*(?:async\s*)?\{",
        code,
        flags=re.MULTILINE,
    )
    if not main_match:
        return code.strip()
    start = main_match.start()
    depth = 0
    i = main_match.end() - 1
    end = None
    while i < len(code):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    if end is None:
        return code[:start].strip()
    return (code[:start] + code[end:]).strip()


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/testing/grpo_data.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"rows: {len(rows)}")

    prompt_chars = []
    target_chars = []
    test_counts = []
    has_tests = 0
    has_signature = 0
    for r in rows:
        prompt_chars.append(len(build_decoder_prompt(r)))
        src = r.get("source") or r.get("dart_source") or r.get("swift_source") or ""
        if r.get("tests"):
            has_tests += 1
            src = _remove_top_level_main(src)
            src = re.sub(r"^@pragma\(.*\)\s*$", "", src, flags=re.MULTILINE)
            n_expects = sum(
                1 for ln in r["tests"].splitlines()
                if ln.strip().startswith("expect(") and "candidate(" in ln
            )
            test_counts.append(n_expects)
        if r.get("dart_function_signature") or r.get("function_signature") or r.get("signature"):
            has_signature += 1
        target_chars.append(len(src.strip()))

    def describe(name, vals):
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        pct = lambda p: vals_sorted[min(n - 1, int(p * n))]
        print(
            f"{name}: min={vals_sorted[0]} p25={pct(0.25)} median={pct(0.5)} "
            f"p75={pct(0.75)} p90={pct(0.9)} p95={pct(0.95)} max={vals_sorted[-1]} "
            f"mean={statistics.mean(vals_sorted):.0f}"
        )

    describe("prompt_chars", prompt_chars)
    describe("target_chars", target_chars)
    if test_counts:
        describe("expect_calls_per_task", test_counts)
    print(f"rows_with_tests: {has_tests}/{len(rows)}")
    print(f"rows_with_signature: {has_signature}/{len(rows)}")

    # Rough token estimates (Qwen BPE on asm/code ~ 3.0-3.5 chars/token).
    for cpt in (3.0, 3.5):
        over_768 = sum(1 for c in prompt_chars if c / cpt > 768)
        med_tok = statistics.median(prompt_chars) / cpt
        print(
            f"assuming {cpt} chars/token: median prompt ~{med_tok:.0f} tok; "
            f"prompts over 768 tok: {over_768}/{len(rows)}; "
            f"median pad gap ~{max(0, 768 - med_tok):.0f} positions"
        )
        over_256 = sum(1 for c in target_chars if c / cpt > 256)
        over_768t = sum(1 for c in target_chars if c / cpt > 768)
        med_t = statistics.median(target_chars) / cpt
        print(
            f"assuming {cpt} chars/token: median target ~{med_t:.0f} tok; "
            f"targets over 256 tok: {over_256}/{len(rows)}; over 768 tok: {over_768t}/{len(rows)}"
        )


if __name__ == "__main__":
    main()
