"""Compute paper-level CIs and paired tests from Antigravity result CSVs.

This script is intentionally offline: it consumes the archived
*_compile_stats.csv and *_pass_stats.csv files produced by the evaluator and
does not rerun Dart. It reports:

* Wilson intervals for binary task coverage (compile@5 and pass@10).
* Bootstrap intervals for mean estimator metrics (CodeBLEU, pass@1/pass@5).
* Exact McNemar tests for paired coverage comparisons against clean G3.
* The multi-arm union coverage and paired comparison against clean G3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable


TABLE_ARMS = [
    ("binary GRPO", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_binary_pk10_g32_grpo_grpo"),
    ("RS-SFT all-arms", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_rs_sft_allarms"),
    ("binary RS-SFT ref", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_binary_rs_sft_ref"),
    ("clean G3 graph-only", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_graphonly"),
    ("style repair ultralite", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_style_repair_ultralite"),
    ("G3 p128r", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_p128r"),
    ("style1036 SFT", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_style1036_sft"),
    ("CFG-only", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g2c_cfgonly"),
    ("SimKO-style GRPO", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_simko_eval_grpo"),
    ("graph-text", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g2_graphtext"),
    ("base reference", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_ref_base"),
    ("null graph", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g0_null"),
    ("text-only", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g1_textonly"),
    ("p128 without RMS", "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_p128"),
]


def flag(value: str | int | float | None) -> int:
    try:
        return 1 if float(value or 0) > 0 else 0
    except (TypeError, ValueError):
        return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_indices(row: dict[str, str], suffix: str) -> list[int]:
    out: list[int] = []
    for key in row:
        match = re.fullmatch(rf"cand_(\d+)_{suffix}", key)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total == 0:
        return {"lo": 0.0, "hi": 0.0}
    phat = successes / total
    denom = 1.0 + (z * z) / total
    centre = phat + (z * z) / (2 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4 * total)) / total)
    return {"lo": (centre - margin) / denom, "hi": (centre + margin) / denom}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def bootstrap_ci(values: list[float], reps: int, rng: random.Random) -> dict[str, float]:
    if not values:
        return {"lo": 0.0, "hi": 0.0}
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return {"lo": percentile(means, 0.025), "hi": percentile(means, 0.975)}


def bootstrap_diff_ci(a: list[float], b: list[float], reps: int, rng: random.Random) -> dict[str, float]:
    assert len(a) == len(b)
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    samples = []
    for _ in range(reps):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    return {"lo": percentile(samples, 0.025), "hi": percentile(samples, 0.975)}


def binom_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k + 1)) / (2**n)


def binom_sf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def mcnemar_exact(arm: list[int], base: list[int]) -> dict[str, float | int]:
    assert len(arm) == len(base)
    gains = sum(1 for a, b in zip(arm, base) if a == 1 and b == 0)
    losses = sum(1 for a, b in zip(arm, base) if a == 0 and b == 1)
    discordant = gains + losses
    if discordant == 0:
        p = 1.0
    else:
        p = min(1.0, 2.0 * min(binom_cdf(min(gains, losses), discordant), binom_sf(max(gains, losses), discordant)))
    return {"gains": gains, "losses": losses, "discordant": discordant, "p_two_sided": p}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, returned in original key order."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted_raw: dict[str, float] = {}
    running_max = 0.0
    for rank, (key, p_value) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_raw[key] = running_max
    return {key: adjusted_raw[key] for key in p_values}


def summarize_stats(rows: list[dict[str, str]], mode: str) -> dict:
    if not rows:
        return {}
    suffix = "pass" if mode == "pass" else "compile"
    idxs = candidate_indices(rows[0], suffix)
    codebleu_idxs = candidate_indices(rows[0], "codebleu")

    counts: list[int] = []
    coverage: list[int] = []
    estimator_values: dict[int, list[float]] = {1: [], 5: [], 10: []}
    codebleu_best: list[float] = []

    for row in rows:
        c = sum(flag(row.get(f"cand_{idx}_{suffix}")) for idx in idxs)
        n = len(idxs)
        counts.append(c)
        coverage.append(1 if c > 0 else 0)
        for k in estimator_values:
            if k <= n:
                estimator_values[k].append(pass_at_k(n, c, k))

        scores = []
        for idx in codebleu_idxs:
            try:
                scores.append(float(row.get(f"cand_{idx}_codebleu") or 0.0))
            except ValueError:
                scores.append(0.0)
        codebleu_best.append(max(scores) if scores else 0.0)

    return {
        "rows": len(rows),
        "candidate_count": len(idxs),
        "candidate_successes": sum(counts),
        "coverage_successes": sum(coverage),
        "coverage_rate": sum(coverage) / len(rows),
        "coverage": coverage,
        "counts": counts,
        "estimator_values": estimator_values,
        "estimator_means": {f"{mode}_at_{k}": sum(vals) / len(vals) for k, vals in estimator_values.items() if vals},
        "codebleu_values": codebleu_best,
        "codebleu_mean": sum(codebleu_best) / len(codebleu_best) if codebleu_best else 0.0,
    }


def maybe_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_json_error": f"could not parse {path}"}


def format_ci(mean: float, ci: dict[str, float]) -> str:
    return f"{mean:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]"


def union_from_report(report_path: Path) -> dict:
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pools = report.get("pools", [])
    coverage: list[int] | None = None
    for pool in pools:
        stats_path = Path(pool["stats"])
        if not stats_path.is_file():
            continue
        rows = read_csv(stats_path)
        summary = summarize_stats(rows, "pass")
        cov = summary["coverage"]
        if coverage is None:
            coverage = [0] * len(cov)
        for i, ok in enumerate(cov):
            coverage[i] = 1 if coverage[i] or ok else 0
    if coverage is None:
        return {}
    return {
        "prediction_files_used": report.get("prediction_files_used"),
        "coverage": coverage,
        "coverage_successes": sum(coverage),
        "coverage_rate": sum(coverage) / len(coverage),
        "reported_union_tasks_with_pass": report.get("union_tasks_with_pass"),
        "reported_zero_pass_tasks": report.get("zero_pass_tasks"),
        "reported_passing_candidates_total": report.get("passing_candidates_total"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results-20260707/results")
    parser.add_argument("--union_report", default="data/testing/rs_sft_x86_8b_allarms_with_h100_report.json")
    parser.add_argument("--output_json", default="results-20260707/results/sweeps_antigravity/x86_8b_paper_statistics.json")
    parser.add_argument("--output_csv", default="results-20260707/results/sweeps_antigravity/x86_8b_paper_statistics.csv")
    parser.add_argument("--bootstrap_reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sweeps = results_dir / "sweeps_antigravity"
    rng = random.Random(args.seed)

    arm_results: dict[str, dict] = {}
    baseline_key = "clean G3 graph-only"

    for label, stem in TABLE_ARMS:
        compile_stats_path = sweeps / f"{stem}_compile_stats.csv"
        pass_stats_path = sweeps / f"{stem}_pass_stats.csv"
        summary_path = sweeps / f"{stem}.json"
        compile_json_path = sweeps / f"{stem}_compile_at_k.json"
        pass_json_path = sweeps / f"{stem}_pass_at_k.json"
        codebleu_json_path = sweeps / f"{stem}_codebleu.json"

        compile_summary = summarize_stats(read_csv(compile_stats_path), "compile") if compile_stats_path.is_file() else {}
        pass_summary = summarize_stats(read_csv(pass_stats_path), "pass") if pass_stats_path.is_file() else {}

        summary_json = maybe_json(summary_path)
        if not summary_json:
            summary_json = {
                "compile_at_k": maybe_json(compile_json_path),
                "pass_at_k": maybe_json(pass_json_path),
                "codebleu": maybe_json(codebleu_json_path),
            }

        row: dict = {
            "stem": stem,
            "summary_json": summary_json,
            "has_compile_stats": bool(compile_summary),
            "has_pass_stats": bool(pass_summary),
            "compile": compile_summary,
            "pass": pass_summary,
        }

        if compile_summary:
            row["compile"]["coverage_ci_wilson"] = wilson(compile_summary["coverage_successes"], compile_summary["rows"])
            row["compile"]["codebleu_ci_bootstrap"] = bootstrap_ci(compile_summary["codebleu_values"], args.bootstrap_reps, rng)
        if pass_summary:
            row["pass"]["coverage_ci_wilson"] = wilson(pass_summary["coverage_successes"], pass_summary["rows"])
            for metric, vals in pass_summary["estimator_values"].items():
                row["pass"][f"pass_at_{metric}_ci_bootstrap"] = bootstrap_ci(vals, args.bootstrap_reps, rng)

        arm_results[label] = row

    base_compile = arm_results[baseline_key]["compile"]
    base_pass = arm_results[baseline_key]["pass"]
    for label, row in arm_results.items():
        if row.get("compile") and base_compile and len(row["compile"]["coverage"]) == len(base_compile["coverage"]):
            row["compile"]["mcnemar_vs_clean_g3"] = mcnemar_exact(row["compile"]["coverage"], base_compile["coverage"])
            row["compile"]["codebleu_diff_vs_clean_g3_ci_bootstrap"] = bootstrap_diff_ci(
                row["compile"]["codebleu_values"], base_compile["codebleu_values"], args.bootstrap_reps, rng
            )
        if row.get("pass") and base_pass and len(row["pass"]["coverage"]) == len(base_pass["coverage"]):
            row["pass"]["mcnemar_vs_clean_g3"] = mcnemar_exact(row["pass"]["coverage"], base_pass["coverage"])
            row["pass"]["pass_at_5_diff_vs_clean_g3_ci_bootstrap"] = bootstrap_diff_ci(
                row["pass"]["estimator_values"][5], base_pass["estimator_values"][5], args.bootstrap_reps, rng
            )

    union = union_from_report(Path(args.union_report))
    if union and base_pass:
        union["coverage_ci_wilson"] = wilson(union["coverage_successes"], len(union["coverage"]))
        union["mcnemar_vs_clean_g3"] = mcnemar_exact(union["coverage"], base_pass["coverage"])

    selected_p_values: dict[str, float] = {}
    for label in ["binary GRPO", "RS-SFT all-arms", "CFG-only", "text-only", "p128 without RMS"]:
        row = arm_results[label]
        selected_p_values[f"{label} compile@5"] = row["compile"]["mcnemar_vs_clean_g3"]["p_two_sided"]
        selected_p_values[f"{label} pass@10"] = row["pass"]["mcnemar_vs_clean_g3"]["p_two_sided"]
    if union:
        selected_p_values["all-arm union pass coverage"] = union["mcnemar_vs_clean_g3"]["p_two_sided"]

    selected_holm = holm_adjust(selected_p_values)

    payload = {
        "method": {
            "coverage_ci": "Wilson 95% interval",
            "mean_metric_ci": f"nonparametric bootstrap over tasks, reps={args.bootstrap_reps}, seed={args.seed}",
            "paired_binary_test": "exact two-sided McNemar/binomial sign test on discordant task coverage pairs",
            "multiple_comparison_adjustment": "Holm-Bonferroni over the selected paired tests reported in the paper table",
            "baseline": baseline_key,
            "note": "Statistics are recomputed from archived candidate-level CSVs when available. Some repair arms lack compile-set CSVs and therefore have no paired compile test.",
        },
        "arms": arm_results,
        "union": union,
        "selected_p_values_raw": selected_p_values,
        "selected_p_values_holm": selected_holm,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_csv = Path(args.output_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "arm",
            "codebleu_mean_ci",
            "compile_at_5_ci",
            "compile_mcnemar_gains",
            "compile_mcnemar_losses",
            "compile_mcnemar_p",
            "compile_mcnemar_holm_p",
            "pass_at_10_ci",
            "pass_mcnemar_gains",
            "pass_mcnemar_losses",
            "pass_mcnemar_p",
            "pass_mcnemar_holm_p",
            "pass_at_5_diff_vs_clean_g3_ci",
        ])
        for label, row in arm_results.items():
            compile_s = row.get("compile") or {}
            pass_s = row.get("pass") or {}
            compile_m = compile_s.get("mcnemar_vs_clean_g3", {})
            pass_m = pass_s.get("mcnemar_vs_clean_g3", {})
            writer.writerow([
                label,
                format_ci(compile_s.get("codebleu_mean", 0.0), compile_s.get("codebleu_ci_bootstrap", {"lo": 0.0, "hi": 0.0})) if compile_s else "",
                format_ci(compile_s.get("coverage_rate", 0.0), compile_s.get("coverage_ci_wilson", {"lo": 0.0, "hi": 0.0})) if compile_s else "",
                compile_m.get("gains", ""),
                compile_m.get("losses", ""),
                f"{compile_m.get('p_two_sided', ''):.6g}" if compile_m else "",
                f"{selected_holm.get(f'{label} compile@5', ''):.6g}" if f"{label} compile@5" in selected_holm else "",
                format_ci(pass_s.get("coverage_rate", 0.0), pass_s.get("coverage_ci_wilson", {"lo": 0.0, "hi": 0.0})) if pass_s else "",
                pass_m.get("gains", ""),
                pass_m.get("losses", ""),
                f"{pass_m.get('p_two_sided', ''):.6g}" if pass_m else "",
                f"{selected_holm.get(f'{label} pass@10', ''):.6g}" if f"{label} pass@10" in selected_holm else "",
                format_ci(
                    pass_s.get("estimator_means", {}).get("pass_at_5", 0.0) - base_pass.get("estimator_means", {}).get("pass_at_5", 0.0),
                    pass_s.get("pass_at_5_diff_vs_clean_g3_ci_bootstrap", {"lo": 0.0, "hi": 0.0}),
                ) if pass_s and "pass_at_5_diff_vs_clean_g3_ci_bootstrap" in pass_s else "",
            ])

    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    print("\nSelected comparisons vs clean G3:")
    for label in ["binary GRPO", "RS-SFT all-arms", "CFG-only", "text-only", "p128 without RMS"]:
        row = arm_results[label]
        comp = row["compile"]
        pas = row["pass"]
        print(
            f"{label}: compile@5 {format_ci(comp['coverage_rate'], comp['coverage_ci_wilson'])} "
            f"McNemar +{comp['mcnemar_vs_clean_g3']['gains']}/-{comp['mcnemar_vs_clean_g3']['losses']} "
            f"p={comp['mcnemar_vs_clean_g3']['p_two_sided']:.4g} "
            f"(Holm {selected_holm[f'{label} compile@5']:.4g}); "
            f"pass@10 {format_ci(pas['coverage_rate'], pas['coverage_ci_wilson'])} "
            f"McNemar +{pas['mcnemar_vs_clean_g3']['gains']}/-{pas['mcnemar_vs_clean_g3']['losses']} "
            f"p={pas['mcnemar_vs_clean_g3']['p_two_sided']:.4g} "
            f"(Holm {selected_holm[f'{label} pass@10']:.4g})"
        )
    if union:
        print(
            f"all-arm union: pass coverage {format_ci(union['coverage_rate'], union['coverage_ci_wilson'])} "
            f"McNemar +{union['mcnemar_vs_clean_g3']['gains']}/-{union['mcnemar_vs_clean_g3']['losses']} "
            f"p={union['mcnemar_vs_clean_g3']['p_two_sided']:.4g} "
            f"(Holm {selected_holm['all-arm union pass coverage']:.4g})"
        )


if __name__ == "__main__":
    main()
