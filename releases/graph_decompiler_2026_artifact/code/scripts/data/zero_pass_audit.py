"""Audit the zero-pass wall: classify why ~2/3 of tasks never pass.

Takes every archived *_pass_stats.csv (per-candidate pass/compile/codebleu
flags for the same 154-task eval order) and grpo_data.jsonl, finds the
HARD-CORE zero-pass set (tasks with zero passing candidates across the
UNION of all pools), and classifies each into:

  near_miss     best candidate compiles and max CodeBLEU >= 0.65 across all
                pools: the model is close; harness-strict or subtle
                semantics. RS-SFT/selection plausibly winnable.
  literal_gap   the reference contains string literals (or mostly numeric
                literals) that do NOT appear in the assembly text (checked
                raw and hex for numbers): the tested behavior depends on
                constants the input may not legibly carry -> evidence of
                the INFORMATION ceiling.
  far_miss      everything else: the model is far away and the input seems
                sufficient -> KNOWLEDGE gap (SFT data regime).

Also reports the union any-pass count (the cross-checkpoint ensemble
oracle). All offline; no Dart, no GPU.

Usage:
  python scripts/data/zero_pass_audit.py --report results/analysis/zero_pass_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_POOLS = [
    ("base", "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_pass_stats.csv"),
    ("ut", "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_pass_stats.csv"),
    ("ut_gentle", "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle_grpo_pass_stats.csv"),
    ("ut_gentle2", "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_gentle2_grpo_pass_stats.csv"),
    ("ut_grpo", "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_grpo_pass_stats.csv"),
    ("ut_rewardfix", "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardfix_grpo_pass_stats.csv"),
    ("ut_rewardsoft", "results-qwen-9b-latest-3/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_ut_rewardsoft_grpo_pass_stats.csv"),
    ("fitut", "results-qwen-9b-stageC/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut_pass_stats.csv"),
    ("fitut_grpo", "results-qwen-9b-stageC/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitut_grpo_pass_stats.csv"),
    ("pk5", "results-qwen-9b-stageD/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5_grpo_pass_stats.csv"),
    ("pk5h", "results-qwen-9b-stageE/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5h_grpo_pass_stats.csv"),
    ("pk5g16", "results-qwen-9b-stageF/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5g16_grpo_pass_stats.csv"),
    ("pk5dapo", "results-qwen-9b-stageG/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5dapo_grpo_pass_stats.csv"),
    ("pk5gspo", "results-qwen-9b-stageG/sweeps_antigravity/qwen-9b-base_lora_enc_dec_r64_5e6_gcb_a128_fitutpk5gspo_grpo_pass_stats.csv"),
]

_STRING_RE = re.compile(r"""(?:'([^'\\]{2,})')|(?:"([^"\\]{2,})")""")
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def _flag(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    try:
        return 1 if float(text) >= 0.5 else 0
    except ValueError:
        return 0


def _float(value) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def load_pool(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    n_cands = max(
        int(match.group(1))
        for match in (re.match(r"cand_(\d+)_pass$", col) for col in rows[0])
        if match
    )
    return rows, n_cands


def reference_literals(source: str):
    strings = [m.group(1) or m.group(2) for m in _STRING_RE.finditer(source)]
    numbers = []
    for m in _NUMBER_RE.finditer(source):
        text = m.group(1)
        try:
            value = float(text)
        except ValueError:
            continue
        if abs(value) >= 10:  # small ints are ubiquitous immediates; skip
            numbers.append(text)
    return strings, numbers


def tests_main_strings(tests: str):
    """String literals asserted in the tests' main() body (the expected
    outputs), excluding the shared expect/expectList/expectMap helper
    boilerplate that appears after main()."""
    main_body = tests.split("void expect", 1)[0]
    return [m.group(1) or m.group(2) for m in _STRING_RE.finditer(main_body)]


def literal_in_assembly(literal: str, assembly: str) -> bool:
    if literal in assembly:
        return True
    try:
        value = float(literal)
        if value == int(value):
            hex_form = hex(int(value))
            if hex_form in assembly or hex_form.upper() in assembly:
                return True
    except ValueError:
        pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/testing/grpo_data.jsonl", type=Path)
    parser.add_argument("--report", default="results/analysis/zero_pass_audit.json", type=Path)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]

    pools = []
    for label, path in DEFAULT_POOLS:
        p = Path(path)
        if not p.is_file():
            print(f"WARNING: missing pool {label}: {path} - skipped")
            continue
        rows, n_cands = load_pool(p)
        if len(rows) != len(tasks):
            print(f"WARNING: pool {label} has {len(rows)} rows vs {len(tasks)} tasks - skipped")
            continue
        pools.append((label, rows, n_cands))
    print(f"pools loaded: {len(pools)} -> {[p[0] for p in pools]}")

    per_task = []
    for idx, task in enumerate(tasks):
        passes = 0
        compiles = 0
        max_cb = 0.0
        for _, rows, n_cands in pools:
            row = rows[idx]
            for cand in range(1, n_cands + 1):
                passes += _flag(row.get(f"cand_{cand}_pass", "0"))
                compiles += _flag(row.get(f"cand_{cand}_compile", "0"))
                max_cb = max(max_cb, _float(row.get(f"cand_{cand}_codebleu", "0")))
        strings, numbers = reference_literals(task["dart_source"])
        assembly = task.get("assembly", "")
        missing_strings = [s for s in strings if not literal_in_assembly(s, assembly)]
        missing_numbers = [n for n in numbers if not literal_in_assembly(n, assembly)]
        test_strings = tests_main_strings(task.get("tests", ""))
        test_strings_missing = [s for s in test_strings if not literal_in_assembly(s, assembly)]
        per_task.append({
            "index": idx,
            "task_id": task.get("task_id", idx),
            "function": task.get("function", ""),
            "union_passes": passes,
            "union_compiles": compiles,
            "max_codebleu": round(max_cb, 4),
            "ref_strings": len(strings),
            "ref_strings_missing": len(missing_strings),
            "ref_numbers": len(numbers),
            "ref_numbers_missing": len(missing_numbers),
            "test_strings": len(test_strings),
            "test_strings_missing": len(test_strings_missing),
            "test_string_examples": test_strings_missing[:3],
        })

    union_any_pass = sum(1 for t in per_task if t["union_passes"] > 0)
    zero = [t for t in per_task if t["union_passes"] == 0]
    solved = [t for t in per_task if t["union_passes"] > 0]

    # Classification: tests asserting STRING outputs absent from the assembly
    # are underivable (information gap); numeric/bool expectations are
    # computable from recovered logic, so the rest is a knowledge gap. The
    # same flag is computed on SOLVED tasks as the control group.
    def has_string_gap(t) -> bool:
        return t["test_strings_missing"] > 0

    for t in zero:
        t["bucket"] = "info_gap_string_outputs" if has_string_gap(t) else "knowledge_gap"

    buckets = {}
    for t in zero:
        buckets[t["bucket"]] = buckets.get(t["bucket"], 0) + 1

    summary = {
        "tasks": len(tasks),
        "pools": [p[0] for p in pools],
        "union_any_pass": union_any_pass,
        "hard_core_zero_pass": len(zero),
        "zero_pass_buckets": buckets,
        "control_solved_with_string_gap": sum(1 for t in solved if has_string_gap(t)),
        "control_solved_total": len(solved),
        "zero_pass_all_compile_somewhere": sum(1 for t in zero if t["union_compiles"] > 0),
        "median_max_codebleu_zero": sorted(t["max_codebleu"] for t in zero)[len(zero) // 2],
        "median_max_codebleu_solved": sorted(t["max_codebleu"] for t in solved)[len(solved) // 2],
        "knowledge_gap_task_ids": [t["task_id"] for t in zero if t["bucket"] == "knowledge_gap"],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "zero_pass_tasks": zero, "all_tasks": per_task}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
