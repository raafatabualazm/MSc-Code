"""Analyze the leakage-free Graph-v2 study from archived result files.

The script is offline: it reads sweep summaries, candidate-level pass CSVs,
prediction pools, provenance manifests, and the Graph-v2 benchmark. It does not
invoke Dart or load a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


MODEL_PREFIXES = (
    "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_",
    "qwen3-8b-base_lora_enc_dec_r64_5e6_clap_a128_graphv2_clean_",
)
CORE_VARIANTS = ("prefix_no_edges", "prefix_cfg", "prefix_cfg_dfg")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        return sorted(rows, key=lambda row: int(row["problem_id"]))
    except (KeyError, TypeError, ValueError):
        return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flag(value: str | int | float | None) -> int:
    try:
        return int(float(value or 0) > 0)
    except (TypeError, ValueError):
        return 0


def candidate_indices(row: dict[str, str], suffix: str) -> list[int]:
    indices: list[int] = []
    for key in row:
        match = re.fullmatch(rf"cand_(\d+)_{suffix}", key)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    miss = 1.0
    for index in range(k):
        miss *= (n - c - index) / (n - index)
    return 1.0 - miss


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def bootstrap_paired_diff(
    arm: list[float], baseline: list[float], reps: int, rng: random.Random
) -> dict[str, float]:
    if len(arm) != len(baseline):
        raise ValueError("Paired vectors have different lengths")
    diffs = [a - b for a, b in zip(arm, baseline)]
    n = len(diffs)
    estimates = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)]
    return {
        "difference": statistics.fmean(diffs),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def hierarchical_bootstrap_diff(
    arm_by_seed: dict[int, list[float]],
    baseline_by_seed: dict[int, list[float]],
    reps: int,
    rng: random.Random,
) -> dict[str, float]:
    seeds = sorted(set(arm_by_seed) & set(baseline_by_seed))
    if not seeds:
        return {"difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    seed_means = [
        statistics.fmean(a - b for a, b in zip(arm_by_seed[seed], baseline_by_seed[seed]))
        for seed in seeds
    ]
    estimates: list[float] = []
    for _ in range(reps):
        sampled_seed_means: list[float] = []
        for _ in seeds:
            seed = seeds[rng.randrange(len(seeds))]
            arm = arm_by_seed[seed]
            base = baseline_by_seed[seed]
            n = len(arm)
            sampled_seed_means.append(
                sum(arm[index] - base[index] for index in (rng.randrange(n) for _ in range(n))) / n
            )
        estimates.append(statistics.fmean(sampled_seed_means))
    return {
        "difference": statistics.fmean(seed_means),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def binom_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, index) for index in range(k + 1)) / (2**n)


def binom_sf(k: int, n: int) -> float:
    return sum(math.comb(n, index) for index in range(k, n + 1)) / (2**n)


def mcnemar_exact(arm: list[int], baseline: list[int]) -> dict[str, float | int]:
    if len(arm) != len(baseline):
        raise ValueError("Paired coverage vectors have different lengths")
    gains = sum(a == 1 and b == 0 for a, b in zip(arm, baseline))
    losses = sum(a == 0 and b == 1 for a, b in zip(arm, baseline))
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(
            1.0,
            2.0
            * min(
                binom_cdf(min(gains, losses), discordant),
                binom_sf(max(gains, losses), discordant),
            ),
        )
    return {
        "gains": gains,
        "losses": losses,
        "discordant": discordant,
        "p_two_sided": p_value,
    }


def canonical_candidate(text: str) -> str:
    text = re.sub(r"```(?:dart)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return re.sub(r"\s+", " ", text).strip()


def summarize_stats(rows: list[dict[str, str]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    pass_indices = candidate_indices(rows[0], "pass")
    compile_indices = candidate_indices(rows[0], "compile")
    codebleu_indices = candidate_indices(rows[0], "codebleu")
    if pass_indices != compile_indices:
        raise ValueError("Pass and compile candidate columns do not align")

    task_metrics: dict[str, list[float]] = {
        "pass_at_1": [],
        "pass_at_5": [],
        "pass_at_10": [],
        "compile_at_1": [],
        "compile_at_5": [],
        "compile_at_10": [],
        "best_codebleu": [],
    }
    pass_counts: list[int] = []
    compile_counts: list[int] = []
    for row in rows:
        n = len(pass_indices)
        passed = sum(flag(row.get(f"cand_{index}_pass")) for index in pass_indices)
        compiled = sum(flag(row.get(f"cand_{index}_compile")) for index in compile_indices)
        pass_counts.append(passed)
        compile_counts.append(compiled)
        for k in (1, 5, 10):
            task_metrics[f"pass_at_{k}"].append(pass_at_k(n, passed, k))
            task_metrics[f"compile_at_{k}"].append(pass_at_k(n, compiled, k))
        scores = [float(row.get(f"cand_{index}_codebleu") or 0.0) for index in codebleu_indices]
        task_metrics["best_codebleu"].append(max(scores) if scores else 0.0)

    unique_counts: list[int] = []
    if len(predictions) == len(rows):
        for row in predictions:
            candidates = row.get("predictions") or [row.get("prediction", "")]
            unique_counts.append(len({canonical_candidate(str(candidate)) for candidate in candidates}))

    return {
        "rows": len(rows),
        "candidates_per_task": len(pass_indices),
        "metrics": {key: statistics.fmean(values) for key, values in task_metrics.items()},
        "task_metrics": task_metrics,
        "pass_counts": pass_counts,
        "compile_counts": compile_counts,
        "pass_coverage": [int(count > 0) for count in pass_counts],
        "compile_coverage": [int(count > 0) for count in compile_counts],
        "tasks_with_pass": sum(count > 0 for count in pass_counts),
        "tasks_with_compile": sum(count > 0 for count in compile_counts),
        "candidate_pass_rate": sum(pass_counts) / (len(rows) * len(pass_indices)),
        "candidate_compile_rate": sum(compile_counts) / (len(rows) * len(compile_indices)),
        "mean_unique_candidates": statistics.fmean(unique_counts) if unique_counts else None,
        "median_unique_candidates": statistics.median(unique_counts) if unique_counts else None,
        "all_identical_task_count": sum(count == 1 for count in unique_counts),
    }


def summary_consistency(summary: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Compare archived aggregate JSON with its later candidate-level replay.

    Aggregate pass/compile and candidate CSVs were historically produced by
    separate Dart executions. A timeout can therefore move one candidate, and
    the old CodeBLEU script used a less capable code extractor than the CSV
    compiler. Keep those disagreements visible instead of silently mixing the
    two sources in one table.
    """
    comparisons: dict[str, dict[str, Any]] = {}

    def add(name: str, archived: Any, replayed: Any, tolerance: float) -> None:
        if archived is None or replayed is None:
            return
        archived_value = float(archived)
        replayed_value = float(replayed)
        difference = replayed_value - archived_value
        comparisons[name] = {
            "archived_summary": archived_value,
            "candidate_replay": replayed_value,
            "difference": difference,
            "tolerance": tolerance,
            "matches": abs(difference) <= tolerance,
        }

    summary_pass = summary.get("pass_at_k") or {}
    summary_compile = summary.get("compile_at_k") or {}
    replayed = stats.get("metrics") or {}
    for k in (1, 5, 10):
        add(f"pass_at_{k}", summary_pass.get(f"pass_at_{k}"), replayed.get(f"pass_at_{k}"), 1e-9)
        add(
            f"compile_at_{k}",
            summary_compile.get(f"compile_at_{k}"),
            replayed.get(f"compile_at_{k}"),
            1e-9,
        )
    add(
        "codebleu",
        (summary.get("codebleu") or {}).get("mean_codebleu"),
        replayed.get("best_codebleu"),
        1e-4,
    )
    mismatches = {name: item for name, item in comparisons.items() if not item["matches"]}
    return {
        "passed": not mismatches,
        "comparisons": comparisons,
        "mismatches": mismatches,
    }


def load_stability_corrections(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "logs" / "leakage_free_graphv2" / "stability_corrections.json"
    if not path.is_file():
        return {"schema": None, "corrections": [], "diagnostics": {}, "path": str(path)}
    payload = read_json(path)
    payload["path"] = str(path)
    return payload


def apply_stability_corrections(
    rows: list[dict[str, str]], stem: str, corrections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for correction in corrections:
        if correction.get("stem") != stem:
            continue
        row_index = int(correction["row_one_based"]) - 1
        candidate = int(correction["candidate_one_based"])
        field = str(correction["field"])
        if row_index < 0 or row_index >= len(rows):
            raise ValueError(f"Stability correction row out of range for {stem}: {row_index + 1}")
        key = f"cand_{candidate}_{field}"
        archived = flag(rows[row_index].get(key))
        expected = int(correction["archived_value"])
        if archived != expected:
            raise ValueError(
                f"Stability correction source drift for {stem} row {row_index + 1} {key}: "
                f"expected {expected}, found {archived}"
            )
        rows[row_index][key] = str(int(correction["corrected_value"]))
        applied.append(correction)
    return applied


def discover_runs(results_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    sweeps = results_dir / "sweeps_antigravity"
    stability = load_stability_corrections(results_dir)
    patterns = [
        re.compile(rf"^{re.escape(prefix)}s(\d+)_(.+)\.json$")
        for prefix in MODEL_PREFIXES
    ]
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for summary_path in sorted(sweeps.glob("*graphv2_clean_s*.json")):
        match = next(
            (pattern.match(summary_path.name) for pattern in patterns if pattern.match(summary_path.name)),
            None,
        )
        if not match:
            continue
        seed = int(match.group(1))
        variant = match.group(2)
        stem = summary_path.stem
        stats_path = sweeps / f"{stem}_pass_stats.csv"
        prediction_path = results_dir / f"{stem}_pass_predictions.json"
        provenance_path = results_dir / f"{stem}_pass_predictions.json.provenance.json"
        if not (stats_path.is_file() and prediction_path.is_file() and provenance_path.is_file()):
            continue
        summary = read_json(summary_path)
        predictions = read_json(prediction_path)
        provenance = read_json(provenance_path)
        stats_rows = read_csv(stats_path)
        applied_corrections = apply_stability_corrections(
            stats_rows, stem, stability.get("corrections", [])
        )
        stats = summarize_stats(stats_rows, predictions)
        runs[(seed, variant)] = {
            "seed": seed,
            "variant": variant,
            "stem": stem,
            "summary": summary,
            "stats": stats,
            "summary_consistency": summary_consistency(summary, stats),
            "stability_corrections": applied_corrections,
            "provenance": provenance,
            "paths": {
                "summary": str(summary_path),
                "stats": str(stats_path),
                "predictions": str(prediction_path),
                "provenance": str(provenance_path),
            },
        }
    return runs


def seed_summary(runs: dict[tuple[int, str], dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [runs[(seed, variant)] for seed in sorted(seed for seed, name in runs if name == variant)]
    metrics = sorted(selected[0]["stats"]["metrics"]) if selected else []
    out: dict[str, Any] = {"seeds": [run["seed"] for run in selected], "seed_count": len(selected)}
    out["metrics"] = {}
    for metric in metrics:
        values = [run["stats"]["metrics"][metric] for run in selected]
        out["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
            "per_seed": {str(run["seed"]): value for run, value in zip(selected, values)},
        }
    for field in ("tasks_with_pass", "tasks_with_compile", "mean_unique_candidates", "all_identical_task_count"):
        values = [run["stats"][field] for run in selected]
        out[field] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "per_seed": {str(run["seed"]): value for run, value in zip(selected, values)},
        }
    return out


def compare_variants(
    runs: dict[tuple[int, str], dict[str, Any]],
    arm_variant: str,
    baseline_variant: str,
    reps: int,
    rng: random.Random,
) -> dict[str, Any]:
    seeds = sorted(
        seed
        for seed, variant in runs
        if variant == arm_variant and (seed, baseline_variant) in runs
    )
    result: dict[str, Any] = {
        "arm": arm_variant,
        "baseline": baseline_variant,
        "seeds": seeds,
        "per_seed": {},
        "hierarchical_bootstrap": {},
    }
    metrics = ("pass_at_1", "pass_at_5", "pass_at_10", "compile_at_1", "compile_at_5", "compile_at_10", "best_codebleu")
    for seed in seeds:
        arm = runs[(seed, arm_variant)]["stats"]
        base = runs[(seed, baseline_variant)]["stats"]
        seed_result: dict[str, Any] = {"metric_differences": {}}
        for metric in metrics:
            seed_result["metric_differences"][metric] = bootstrap_paired_diff(
                arm["task_metrics"][metric], base["task_metrics"][metric], reps, rng
            )
        seed_result["pass_at_10_mcnemar"] = mcnemar_exact(arm["pass_coverage"], base["pass_coverage"])
        seed_result["compile_at_10_mcnemar"] = mcnemar_exact(
            arm["compile_coverage"], base["compile_coverage"]
        )
        result["per_seed"][str(seed)] = seed_result

    for metric in metrics:
        arm_by_seed = {
            seed: runs[(seed, arm_variant)]["stats"]["task_metrics"][metric] for seed in seeds
        }
        base_by_seed = {
            seed: runs[(seed, baseline_variant)]["stats"]["task_metrics"][metric] for seed in seeds
        }
        result["hierarchical_bootstrap"][metric] = hierarchical_bootstrap_diff(
            arm_by_seed, base_by_seed, reps, rng
        )
    return result


def coverage_union(runs: dict[tuple[int, str], dict[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    selected = [run for (seed, variant), run in sorted(runs.items()) if variant in variants]
    if not selected:
        return {}
    row_count = selected[0]["stats"]["rows"]
    union = [0] * row_count
    frequency = [0] * row_count
    for run in selected:
        for index, covered in enumerate(run["stats"]["pass_coverage"]):
            union[index] = int(union[index] or covered)
            frequency[index] += covered
    return {
        "run_count": len(selected),
        "runs": [run["stem"] for run in selected],
        "tasks_solved": sum(union),
        "coverage_rate": statistics.fmean(union),
        "tasks_never_solved": row_count - sum(union),
        "tasks_solved_by_every_run": sum(count == len(selected) for count in frequency),
        "coverage": union,
    }


def complexity_strata(
    benchmark_path: Path, runs: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, Any]:
    rows = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    complexity: list[dict[str, int]] = []
    for row in rows:
        edges = row.get("edges") or []
        complexity.append(
            {
                "blocks": len(row.get("cfg") or []),
                "cfg_edges": sum(edge.get("edge_type") != "dataflow" for edge in edges),
                "dfg_edges": sum(edge.get("edge_type") == "dataflow" for edge in edges),
            }
        )

    result: dict[str, Any] = {}
    for measure in ("blocks", "cfg_edges", "dfg_edges"):
        ordered_indices = sorted(range(len(rows)), key=lambda index: complexity[index][measure])
        groups = {
            "low": ordered_indices[: len(rows) // 3],
            "mid": ordered_indices[len(rows) // 3 : 2 * len(rows) // 3],
            "high": ordered_indices[2 * len(rows) // 3 :],
        }
        measure_result: dict[str, Any] = {}
        for label, indices in groups.items():
            entry: dict[str, Any] = {
                "tasks": len(indices),
                "range": [
                    min(complexity[index][measure] for index in indices),
                    max(complexity[index][measure] for index in indices),
                ],
                "pass_at_10": {},
            }
            for variant in CORE_VARIANTS:
                selected = [run for (seed, name), run in runs.items() if name == variant]
                values = [
                    run["stats"]["pass_coverage"][index]
                    for run in selected
                    for index in indices
                ]
                entry["pass_at_10"][variant] = statistics.fmean(values) if values else None
            measure_result[label] = entry
        result[measure] = measure_result
    return result


def provenance_audit(runs: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for run in runs.values():
        provenance = run["provenance"]
        pass_path = Path(run["paths"]["predictions"])
        compile_path = pass_path.with_name(pass_path.name.replace("_pass_predictions.json", "_compile_predictions.json"))
        records.append(
            {
                "stem": run["stem"],
                "row_count": provenance.get("row_count"),
                "prompt_schema_version": provenance.get("prompt_schema_version"),
                "scoring_tests_visible_to_policy": provenance.get("scoring_tests_visible_to_policy"),
                "dataset_sha256": provenance.get("dataset", {}).get("sha256"),
                "decoder_revision": provenance.get("models", {}).get("decoder", {}).get("resolved_commit"),
                "encoder_revision": provenance.get("models", {}).get("encoder", {}).get("resolved_commit"),
                "compile_pass_pool_identical": (
                    compile_path.is_file() and sha256(compile_path) == sha256(pass_path)
                ),
            }
        )
    expected = {
        "row_count": 154,
        "prompt_schema_version": "antigravity-v2-no-test-hints",
        "scoring_tests_visible_to_policy": False,
        "dataset_sha256": "8453876a40d2279684a190a5bf1430a62897c84e063a78e25c57198287bc6928",
        "decoder_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "encoder_revision": "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d",
        "compile_pass_pool_identical": True,
    }
    failures: list[dict[str, Any]] = []
    for record in records:
        mismatches = {
            key: {"expected": value, "actual": record.get(key)}
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            failures.append({"stem": record["stem"], "mismatches": mismatches})
    return {
        "passed": not failures,
        "run_count": len(records),
        "expected": expected,
        "failures": failures,
        "records": records,
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(payload: dict[str, Any]) -> str:
    stability = payload["stability_corrections"]
    bulk_audit = stability.get("bulk_audit", {})
    correction_count = len(stability.get("corrections", []))
    correction_noun = "false positive is" if correction_count == 1 else "false positives are"
    lines = [
        "# Graph-v2 Leakage-Free Study Analysis",
        "",
        "This report recomputes metrics from archived candidate pools; no model inference was rerun. Dart was rerun only for the explicit multi-run pass-stability audit described below.",
        "",
        "## Protocol",
        "",
        f"- Provenance audit: **{'PASS' if payload['provenance_audit']['passed'] else 'FAIL'}** across {payload['provenance_audit']['run_count']} complete runs.",
        "- All audited runs use 154 tasks, 10 candidates per task, the pinned Qwen3-8B and GraphCodeBERT revisions, and `antigravity-v2-no-test-hints`.",
        "- Compile and pass metrics reuse the same candidate pool and use the pass-aligned JIT/test harness.",
        "- Candidate-level CSV replay is canonical for paired effects, task coverage, and the tables below.",
        f"- Pass stability requires {stability.get('stability_runs', 1)} successful executions. The raw archives are preserved; {correction_count} documented stochastic {correction_noun} corrected through an overlay.",
        f"- The bulk stability audit replayed {bulk_audit.get('archived_passing_candidates_checked', 0)} archived positives across {bulk_audit.get('run_count', 0)} runs; {bulk_audit.get('stable_passing_candidates', 0)} remained passing and {bulk_audit.get('invalidated_candidates', 0)} was invalidated.",
        "- CodeBLEU uses extracted Dart code. The evaluator has been unified with the compile/statistics extractor; archived aggregate JSONs produced by the older raw-text extractor are audited below rather than mixed into the analysis.",
        "- The legacy standalone-AOT compiled-only CodeBLEU count is retained only as a diagnostic and is not treated as the aligned compile metric.",
        "",
        "## Three-Seed Core Ablation",
        "",
        "| Variant | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | compile@10 | CodeBLEU | Solved tasks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "prefix_no_edges": "Prefix, no edges",
        "prefix_cfg": "Prefix + CFG",
        "prefix_cfg_dfg": "Prefix + CFG + DFG",
    }
    for variant in CORE_VARIANTS:
        summary = payload["core_seed_summaries"][variant]
        metrics = summary["metrics"]
        def cell(name: str) -> str:
            entry = metrics[name]
            return f"{entry['mean']:.4f} +/- {entry['sample_std']:.4f}"
        solved = summary["tasks_with_pass"]
        lines.append(
            f"| {labels[variant]} | {cell('pass_at_1')} | {cell('pass_at_5')} | {cell('pass_at_10')} | "
            f"{cell('compile_at_1')} | {cell('compile_at_5')} | {cell('compile_at_10')} | "
            f"{cell('best_codebleu')} | {solved['mean']:.1f} +/- {solved['sample_std']:.1f} |"
        )

    lines.extend(["", "## Edge Effects", ""])
    for key in ("cfg_vs_no_edges", "cfg_dfg_vs_cfg", "cfg_dfg_vs_no_edges"):
        comparison = payload["comparisons"][key]
        effect = comparison["hierarchical_bootstrap"]["pass_at_10"]
        lines.append(
            f"- **{comparison['arm']} vs {comparison['baseline']}**: pass@10 difference "
            f"{effect['difference']:+.4f}, hierarchical bootstrap 95% CI "
            f"[{effect['ci95_low']:+.4f}, {effect['ci95_high']:+.4f}]."
        )
    seed44_cfg = payload["comparisons"]["cfg_vs_no_edges"]["per_seed"].get("44", {})
    seed44_dfg = payload["comparisons"]["cfg_dfg_vs_cfg"]["per_seed"].get("44", {})
    if seed44_cfg and seed44_dfg:
        cfg_test = seed44_cfg["pass_at_10_mcnemar"]
        dfg_test = seed44_dfg["pass_at_10_mcnemar"]
        lines.append(
            "- Seed 44 illustrates the instability: CFG versus no-edge has "
            f"{cfg_test['gains']} gains/{cfg_test['losses']} losses "
            f"(exact p={cfg_test['p_two_sided']:.4f}), while adding DFG to CFG has "
            f"{dfg_test['gains']} gains/{dfg_test['losses']} losses "
            f"(exact p={dfg_test['p_two_sided']:.4f}). The directions reverse within one seed rather than reproducing across seeds."
        )
    lines.extend(
        [
            "",
            "The intervals all cross zero. The current evidence supports the learned block-prefix representation, but does not support a causal benefit from CFG or DFG edges.",
            "",
            "## Signature-Only Control",
            "",
        ]
    )
    for key, label in (
        ("text_vs_signature_s42", "Trained raw assembly vs signature-only base"),
        ("no_edges_vs_signature_s42", "No-edge prefix vs signature-only base"),
        ("cfg_vs_signature_s42", "CFG prefix vs signature-only base"),
    ):
        comparison = payload["comparisons"].get(key)
        if not comparison:
            continue
        effect = comparison["per_seed"]["42"]["metric_differences"]["pass_at_10"]
        test = comparison["per_seed"]["42"]["pass_at_10_mcnemar"]
        lines.append(
            f"- {label}: pass@10 difference {effect['difference']:+.4f}, paired task 95% CI "
            f"[{effect['ci95_low']:+.4f}, {effect['ci95_high']:+.4f}], "
            f"{test['gains']} gains/{test['losses']} losses (exact p={test['p_two_sided']:.4f})."
        )
    lines.extend(
        [
            "",
            "The exact HumanEval-style signature is itself a strong task-recognition cue. Prefix arms improve over raw-assembly SFT in this fixed seed, but their small pass@10 advantage over the signature-only base is not distinguishable from zero. Claims must therefore be about this benchmark and representation pipeline, not general binary-semantic recovery.",
            "",
            "## Complexity Stratification",
            "",
            "| Block-count stratum | Range | Prefix, no edges | Prefix + CFG | Prefix + CFG + DFG |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stratum in ("low", "mid", "high"):
        entry = payload["complexity_strata"]["blocks"][stratum]
        rates = entry["pass_at_10"]
        lines.append(
            f"| {stratum} | {entry['range'][0]}-{entry['range'][1]} blocks | "
            f"{rates['prefix_no_edges']:.4f} | {rates['prefix_cfg']:.4f} | "
            f"{rates['prefix_cfg_dfg']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All variants degrade sharply on larger graphs, and explicit edges do not rescue the high-complexity stratum. This is evidence of a remaining representation/capacity ceiling, not evidence that topology is unnecessary in principle.",
            "",
            "## Seed-42 Diagnostics",
            "",
            "| Variant | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | CodeBLEU | Solved | Mean unique/10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    diagnostic_order = (
        "signature_only_base",
        "untuned",
        "text",
        "prefix_no_edges",
        "prefix_cfg",
        "prefix_cfg_dfg",
        "prefix_no_gine",
        "prefix_no_gine_clap",
        "prefix_no_gine_multivector2",
        "prefix_no_gine_multivector4",
        "prefix_no_gine_multivector8",
        "prefix_no_gine_regions4",
        "prefix_no_gine_regions",
        "prefix_no_gine_regions16",
        "prefix_no_gine_no_attention",
        "prefix_no_gine_no_positions",
        "prefix_no_gine_frozen_encoder",
        "prefix_no_edges_gine2",
        "prefix_no_gine_tokens_per_log2_2",
        "prefix_no_gine_tokens_per_log2_6",
        "prefix_no_gine_eval_gate_zero",
        "prefix_no_gine_eval_prefix_permuted",
        "prefix_no_gine_eval_block_order_shuffled",
        "prefix_no_edges_eval_gate_zero",
        "prefix_no_edges_eval_prefix_permuted",
        "prefix_no_edges_eval_block_order_shuffled",
        "prefix_shuffled",
        "prefix_cfg_dfg_text",
        "prefix_cfg_dfg_expanded",
    )
    dynamic_grid_variants = sorted(
        key.split(":", 1)[1]
        for key in payload["runs"]
        if key.startswith("42:")
        and "_ppl" in key
        and "_gate" in key
        and "_eval_" not in key
    )
    rendered_diagnostics: set[str] = set()
    for variant in (*diagnostic_order, *dynamic_grid_variants):
        if variant in rendered_diagnostics:
            continue
        rendered_diagnostics.add(variant)
        run = payload["runs"].get(f"42:{variant}")
        if not run:
            continue
        stats = run["stats"]
        metrics = stats["metrics"]
        lines.append(
            f"| {variant} | {fmt(metrics['pass_at_1'])} | {fmt(metrics['pass_at_5'])} | "
            f"{fmt(metrics['pass_at_10'])} | {fmt(metrics['compile_at_1'])} | "
            f"{fmt(metrics['compile_at_5'])} | {fmt(metrics['best_codebleu'])} | "
            f"{stats['tasks_with_pass']} | {stats['mean_unique_candidates']:.2f} |"
        )

    union = payload["coverage_unions"]
    lines.extend(
        [
            "",
            "## Coverage Across Seeds",
            "",
            f"- No-edge family union: {union['prefix_no_edges']['tasks_solved']}/154 tasks.",
            f"- CFG family union: {union['prefix_cfg']['tasks_solved']}/154 tasks.",
            f"- CFG+DFG family union: {union['prefix_cfg_dfg']['tasks_solved']}/154 tasks.",
            f"- All nine core pools: {union['all_core']['tasks_solved']}/154 tasks; {union['all_core']['tasks_never_solved']} tasks remain unsolved.",
            "",
            "These are oracle candidate-pool ceilings, not deployable pass@1 results.",
            "",
            "## Expanded SFT and GRPO Readiness",
            "",
        ]
    )
    expanded = payload["comparisons"].get("expanded_vs_cfg_dfg_s42")
    if expanded:
        pass10 = expanded["per_seed"]["42"]["metric_differences"]["pass_at_10"]
        lines.append(
            f"- Expanded SFT changes pass@10 by {pass10['difference']:+.4f} "
            f"(95% paired task bootstrap [{pass10['ci95_low']:+.4f}, {pass10['ci95_high']:+.4f}])."
        )
    reward = payload.get("reward_preflight") or {}
    if reward:
        lines.append(
            f"- GRPO reward preflight found signal in {reward.get('signal_group_rate_mean', 0):.1%} of groups, "
            f"with {reward.get('perfect_sample_rate_mean', 0):.1%} perfect samples; this is sufficient to run GRPO, not evidence that GRPO improves held-out performance."
        )

    followup_labels = {
        "no_gine_vs_no_edges_s42": "No-GINE vs no-edge GINE",
        "no_gine_vs_cfg_dfg_s42": "No-GINE vs CFG+DFG GINE",
        "clap_vs_no_gine_s42": "CLAP-ASM vs GraphCodeBERT no-GINE",
        "multivector_vs_no_gine_s42": "Four block vectors vs one CLS vector",
        "multivector2_vs_no_gine_s42": "Two block vectors vs one CLS vector",
        "multivector8_vs_no_gine_s42": "Eight block vectors vs one CLS vector",
        "regions_vs_no_gine_s42": "Hierarchical regions vs no-GINE block prefix",
        "regions4_vs_no_gine_s42": "Region maximum 4 vs no regions",
        "regions16_vs_no_gine_s42": "Region maximum 16 vs no regions",
        "no_attention_vs_no_gine_s42": "No global attention vs no-GINE",
        "no_positions_vs_no_gine_s42": "No block positions vs sinusoidal positions",
        "frozen_encoder_vs_no_gine_s42": "Frozen local encoder vs encoder LoRA",
        "gine2_vs_no_edges_s42": "Two-layer GINE vs four-layer no-edge GINE",
        "gate_zero_vs_no_gine_s42": "Zero prefix gate vs intact no-GINE prefix",
        "permuted_prefix_vs_no_gine_s42": "Cross-task permuted graph vs intact graph",
        "shuffled_blocks_vs_no_gine_s42": "Shuffled block order vs intact block order",
        "capacity2_vs_no_gine_s42": "Dynamic prefix scale 2 vs scale 4",
        "capacity6_vs_no_gine_s42": "Dynamic prefix scale 6 vs scale 4",
        "gate_zero_vs_no_edges_s42": "Zero prefix gate vs intact no-edge prefix",
        "permuted_prefix_vs_no_edges_s42": "Cross-task permuted blocks vs intact no-edge blocks",
        "shuffled_blocks_vs_no_edges_s42": "Shuffled block order vs intact no-edge block order",
    }
    available_followups = [
        (key, followup_labels[key])
        for key in followup_labels
        if key in payload["comparisons"]
    ]
    if available_followups:
        lines.extend(["", "## GINE and Prefix Causal Follow-ups", ""])
        for key, label in available_followups:
            comparison = payload["comparisons"][key]
            effect = comparison["per_seed"]["42"]["metric_differences"]["pass_at_10"]
            test = comparison["per_seed"]["42"]["pass_at_10_mcnemar"]
            lines.append(
                f"- {label}: pass@10 difference {effect['difference']:+.4f}, paired task "
                f"95% CI [{effect['ci95_low']:+.4f}, {effect['ci95_high']:+.4f}], "
                f"{test['gains']} gains/{test['losses']} losses "
                f"(exact p={test['p_two_sided']:.4f})."
            )
        lines.append(
            "These are seed-42 causal diagnostics. The winning architecture must be repeated "
            "at seeds 43-46 before it becomes the five-seed confirmatory configuration."
        )

    grid_comparisons = sorted(
        (key, value)
        for key, value in payload["comparisons"].items()
        if key.startswith("grid_")
    )
    if grid_comparisons:
        lines.extend(["", "## Prefix Density and Gate Grid", ""])
        for _, comparison in grid_comparisons:
            effect = comparison["per_seed"]["42"]["metric_differences"]["pass_at_10"]
            test = comparison["per_seed"]["42"]["pass_at_10_mcnemar"]
            lines.append(
                f"- {comparison['arm']} vs {comparison['baseline']}: pass@10 "
                f"{effect['difference']:+.4f}, paired 95% CI "
                f"[{effect['ci95_low']:+.4f}, {effect['ci95_high']:+.4f}], "
                f"{test['gains']} gains/{test['losses']} losses."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. The leakage-free graph-prefix family has the strongest fixed-run local results, particularly relative to raw full-assembly prompting, but the advantage over the signature-only base is modest at k=10.",
            "2. Explicit topology is not validated: no-edge has the best three-seed mean pass@10, shuffled edges do not hurt at seed 42, and CFG+DFG does not consistently beat CFG.",
            "3. The likely useful component is learned assembly-to-prefix compression through the block encoder and adapter. Calling the observed gain a graph-topology gain would overstate the evidence.",
            "4. Adding raw assembly back to CFG+DFG raises CodeBLEU but does not improve pass@10, consistent with context overload or competing conditioning channels.",
            "5. Expanded synthetic SFT is essentially neutral on this benchmark. The pending GRPO run is justified by the reward-signal audit, but must be accepted only on held-out leakage-free metrics.",
            "",
            "## Reporting Caveats",
            "",
            "- The legacy core has three seeds; the frozen selected configuration requires five matched seeds. Report all per-seed values, mean +/- sample SD, and hierarchical bootstrap intervals.",
            "- Seed-42 controls, shuffled edges, graph+text, and expanded SFT remain single-run diagnostics.",
            "- CodeBLEU and aligned JIT compile measure different properties from functional pass@k; functional conclusions should lead with pass@k.",
        ]
    )

    consistency = payload["artifact_consistency"]
    lines.extend(
        [
            "",
            "## Artifact Consistency",
            "",
            f"- {consistency['mismatch_run_count']} of {consistency['run_count']} runs contain at least one aggregate-summary/candidate-replay disagreement.",
            f"- CodeBLEU extraction differences occur in {consistency['codebleu_mismatch_run_count']} runs and are resolved by the shared code extractor.",
            f"- Functional metric differences occur in {consistency['functional_mismatch_run_count']} runs.",
        ]
    )
    stability = payload["stability_corrections"].get("diagnostics", {})
    if stability:
        single = stability.get("single_run_task158", {})
        stable = stability.get("stable3_task158", {})
        lines.append(
            f"- Task 158 diagnosed the cause: one-run evaluation passed in {single.get('positive_repeats', 0)}/{single.get('repeats', 0)} replays, while three-run stability passed in {stable.get('positive_repeats', 0)}/{stable.get('repeats', 0)} replays. The candidate used random tie-breaking on a deterministic task."
        )
    for mismatch in consistency["functional_mismatches"]:
        lines.append(
            f"- `{mismatch['stem']}` `{mismatch['metric']}` differs by "
            f"{mismatch['difference']:+.6f}; the archived aggregate is retained, while the corrected candidate replay is used for analysis."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results-20260713")
    parser.add_argument("--benchmark", default="data/testing/grpo_data_graphv2.jsonl")
    parser.add_argument("--output_json", default="results-20260713/graphv2_clean_study_analysis.json")
    parser.add_argument("--output_md", default="results-20260713/GRAPHV2_CLEAN_STUDY_ANALYSIS.md")
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260713)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    benchmark_path = Path(args.benchmark)
    rng = random.Random(args.bootstrap_seed)
    runs = discover_runs(results_dir)
    if not runs:
        raise SystemExit(f"No complete Graph-v2 runs found under {results_dir}")

    comparisons = {
        "cfg_vs_no_edges": compare_variants(runs, "prefix_cfg", "prefix_no_edges", args.bootstrap_reps, rng),
        "cfg_dfg_vs_cfg": compare_variants(runs, "prefix_cfg_dfg", "prefix_cfg", args.bootstrap_reps, rng),
        "cfg_dfg_vs_no_edges": compare_variants(runs, "prefix_cfg_dfg", "prefix_no_edges", args.bootstrap_reps, rng),
    }
    single_seed_pairs = {
        "shuffled_vs_cfg_dfg_s42": ("prefix_shuffled", "prefix_cfg_dfg"),
        "expanded_vs_cfg_dfg_s42": ("prefix_cfg_dfg_expanded", "prefix_cfg_dfg"),
        "graph_text_vs_cfg_dfg_s42": ("prefix_cfg_dfg_text", "prefix_cfg_dfg"),
        "text_vs_signature_s42": ("text", "signature_only_base"),
        "no_edges_vs_text_s42": ("prefix_no_edges", "text"),
        "cfg_vs_text_s42": ("prefix_cfg", "text"),
        "no_edges_vs_signature_s42": ("prefix_no_edges", "signature_only_base"),
        "cfg_vs_signature_s42": ("prefix_cfg", "signature_only_base"),
        "no_gine_vs_no_edges_s42": ("prefix_no_gine", "prefix_no_edges"),
        "no_gine_vs_cfg_dfg_s42": ("prefix_no_gine", "prefix_cfg_dfg"),
        "clap_vs_no_gine_s42": (
            "prefix_no_gine_clap",
            "prefix_no_gine",
        ),
        "multivector_vs_no_gine_s42": (
            "prefix_no_gine_multivector4",
            "prefix_no_gine",
        ),
        "multivector2_vs_no_gine_s42": (
            "prefix_no_gine_multivector2",
            "prefix_no_gine",
        ),
        "multivector8_vs_no_gine_s42": (
            "prefix_no_gine_multivector8",
            "prefix_no_gine",
        ),
        "regions4_vs_no_gine_s42": (
            "prefix_no_gine_regions4",
            "prefix_no_gine",
        ),
        "regions_vs_no_gine_s42": (
            "prefix_no_gine_regions",
            "prefix_no_gine",
        ),
        "regions16_vs_no_gine_s42": (
            "prefix_no_gine_regions16",
            "prefix_no_gine",
        ),
        "no_attention_vs_no_gine_s42": (
            "prefix_no_gine_no_attention",
            "prefix_no_gine",
        ),
        "no_positions_vs_no_gine_s42": (
            "prefix_no_gine_no_positions",
            "prefix_no_gine",
        ),
        "frozen_encoder_vs_no_gine_s42": (
            "prefix_no_gine_frozen_encoder",
            "prefix_no_gine",
        ),
        "gine2_vs_no_edges_s42": ("prefix_no_edges_gine2", "prefix_no_edges"),
        "gate_zero_vs_no_gine_s42": (
            "prefix_no_gine_eval_gate_zero",
            "prefix_no_gine",
        ),
        "permuted_prefix_vs_no_gine_s42": (
            "prefix_no_gine_eval_prefix_permuted",
            "prefix_no_gine",
        ),
        "shuffled_blocks_vs_no_gine_s42": (
            "prefix_no_gine_eval_block_order_shuffled",
            "prefix_no_gine",
        ),
        "capacity2_vs_no_gine_s42": (
            "prefix_no_gine_tokens_per_log2_2",
            "prefix_no_gine",
        ),
        "capacity6_vs_no_gine_s42": (
            "prefix_no_gine_tokens_per_log2_6",
            "prefix_no_gine",
        ),
        "gate_zero_vs_no_edges_s42": (
            "prefix_no_edges_eval_gate_zero",
            "prefix_no_edges",
        ),
        "permuted_prefix_vs_no_edges_s42": (
            "prefix_no_edges_eval_prefix_permuted",
            "prefix_no_edges",
        ),
        "shuffled_blocks_vs_no_edges_s42": (
            "prefix_no_edges_eval_block_order_shuffled",
            "prefix_no_edges",
        ),
    }
    for key, (arm, baseline) in single_seed_pairs.items():
        if (42, arm) in runs and (42, baseline) in runs:
            comparisons[key] = compare_variants(runs, arm, baseline, args.bootstrap_reps, rng)

    for seed, variant in sorted(runs):
        if seed != 42 or "_eval_" in variant:
            continue
        match = re.fullmatch(r"(.+)_ppl\d+_gate[0-9mp]+", variant)
        if not match:
            continue
        baseline = match.group(1)
        if (42, baseline) not in runs:
            continue
        comparisons[f"grid_{variant}_vs_{baseline}_s42"] = compare_variants(
            runs,
            variant,
            baseline,
            args.bootstrap_reps,
            rng,
        )

    reward_path = (
        results_dir
        / "artifacts"
        / "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_cfgdfg_expanded_synrl_reward_preflight_grpo"
        / "reward_preflight.json"
    )
    mismatch_runs = [run for run in runs.values() if not run["summary_consistency"]["passed"]]
    functional_mismatches: list[dict[str, Any]] = []
    codebleu_mismatch_run_count = 0
    for run in mismatch_runs:
        mismatches = run["summary_consistency"]["mismatches"]
        if "codebleu" in mismatches:
            codebleu_mismatch_run_count += 1
        for metric, mismatch in mismatches.items():
            if metric == "codebleu":
                continue
            functional_mismatches.append(
                {"stem": run["stem"], "metric": metric, **mismatch}
            )
    payload: dict[str, Any] = {
        "schema": "antigravity-graphv2-clean-study-analysis-v1",
        "method": {
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "paired_task_ci": "nonparametric paired bootstrap over benchmark tasks",
            "multi_seed_ci": "hierarchical bootstrap resampling seeds then paired tasks",
            "coverage_test": "exact two-sided McNemar/binomial sign test on pass@10 task coverage",
            "warning": "Legacy three-seed intervals and seed-42 screens are descriptive; confirmatory claims require the frozen five-seed matched comparison.",
        },
        "provenance_audit": provenance_audit(runs),
        "stability_corrections": load_stability_corrections(results_dir),
        "artifact_consistency": {
            "run_count": len(runs),
            "mismatch_run_count": len(mismatch_runs),
            "codebleu_mismatch_run_count": codebleu_mismatch_run_count,
            "functional_mismatch_run_count": len(
                {item["stem"] for item in functional_mismatches}
            ),
            "functional_mismatches": functional_mismatches,
        },
        "core_seed_summaries": {variant: seed_summary(runs, variant) for variant in CORE_VARIANTS},
        "comparisons": comparisons,
        "coverage_unions": {
            variant: coverage_union(runs, (variant,)) for variant in CORE_VARIANTS
        },
        "complexity_strata": complexity_strata(benchmark_path, runs),
        "reward_preflight": read_json(reward_path) if reward_path.is_file() else None,
        "runs": {
            f"{seed}:{variant}": {
                "stem": run["stem"],
                "stats": run["stats"],
                "summary_consistency": run["summary_consistency"],
                "stability_corrections": run["stability_corrections"],
                "paths": run["paths"],
            }
            for (seed, variant), run in sorted(runs.items())
        },
    }
    payload["coverage_unions"]["all_core"] = coverage_union(runs, CORE_VARIANTS)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


if __name__ == "__main__":
    main()
