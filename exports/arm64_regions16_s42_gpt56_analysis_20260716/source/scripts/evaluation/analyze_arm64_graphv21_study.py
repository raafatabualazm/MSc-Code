"""Offline analysis for the leakage-free ARM64 Graph-v2.1 replication."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.analyze_graphv2_clean_study import (
    compare_variants,
    coverage_union,
    read_csv,
    read_json,
    seed_summary,
    summarize_stats,
    summary_consistency,
)


MODEL_PREFIX = (
    "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_arm64v21_"
)
EXPECTED_ROWS = 343
EXPECTED_DATASET_SHA256 = (
    "864dc0bb7e9ee305ba0fc4be6e5d5ecbbeb7c17fd15bb3e41cfcc6d9aaf65fac"
)
CORE_VARIANTS = ("prefix_no_edges", "prefix_cfg", "prefix_cfg_dfg")


def discover_runs(results_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    sweeps = results_dir / "sweeps_antigravity"
    pattern = re.compile(rf"^{re.escape(MODEL_PREFIX)}s(\d+)_(.+)\.json$")
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for summary_path in sorted(sweeps.glob(f"{MODEL_PREFIX}s*.json")):
        match = pattern.match(summary_path.name)
        if not match:
            continue
        seed = int(match.group(1))
        variant = match.group(2)
        stem = summary_path.stem
        stats_path = sweeps / f"{stem}_pass_stats.csv"
        predictions_path = results_dir / f"{stem}_pass_predictions.json"
        provenance_path = Path(str(predictions_path) + ".provenance.json")
        if not all(path.is_file() for path in (stats_path, predictions_path, provenance_path)):
            continue
        rows = read_csv(stats_path)
        predictions = read_json(predictions_path)
        provenance = read_json(provenance_path)
        if (
            len(predictions) != EXPECTED_ROWS
            or provenance.get("row_count") != EXPECTED_ROWS
            or provenance.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
            or provenance.get("prompt_schema_version") != "antigravity-v2-no-test-hints"
            or provenance.get("scoring_tests_visible_to_policy") is not False
        ):
            raise SystemExit(f"ARM64 provenance mismatch: {stem}")
        stats = summarize_stats(rows, predictions)
        summary = read_json(summary_path)
        runs[(seed, variant)] = {
            "seed": seed,
            "variant": variant,
            "stem": stem,
            "stats": stats,
            "summary": summary,
            "summary_consistency": summary_consistency(summary, stats),
            "provenance": provenance,
            "paths": {
                "summary": str(summary_path),
                "stats": str(stats_path),
                "predictions": str(predictions_path),
                "provenance": str(provenance_path),
            },
        }
    return runs


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ARM64 Graph-v2.1 Study Analysis",
        "",
        "All metrics are recomputed from the same 343-task, 10-candidate pools. "
        "The evaluation split has no exact or near-source overlap with ARM64 training "
        "or the x86 HumanEval-Dart benchmark.",
        "",
        "## Results",
        "",
        "| Variant | Seeds | pass@1 | pass@5 | pass@10 | compile@1 | compile@10 | Solved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        "signature_only",
        "text",
        "prefix_no_edges",
        "prefix_cfg",
        "prefix_cfg_dfg",
        "prefix_shuffled",
        "prefix_no_gine",
    )
    for variant in order:
        summary = payload["variant_summaries"].get(variant)
        if not summary or not summary["seed_count"]:
            continue
        metrics = summary["metrics"]

        def cell(metric: str) -> str:
            item = metrics[metric]
            return f"{item['mean']:.4f} +/- {item['sample_std']:.4f}"

        solved = summary["tasks_with_pass"]
        lines.append(
            f"| {variant} | {summary['seed_count']} | {cell('pass_at_1')} | "
            f"{cell('pass_at_5')} | {cell('pass_at_10')} | {cell('compile_at_1')} | "
            f"{cell('compile_at_10')} | {solved['mean']:.1f} +/- {solved['sample_std']:.1f} |"
        )

    if payload["comparisons"]:
        lines.extend(["", "## Paired Effects", ""])
        for name, comparison in payload["comparisons"].items():
            effect = comparison["hierarchical_bootstrap"]["pass_at_10"]
            lines.append(
                f"- **{name}**: pass@10 difference {effect['difference']:+.4f}, "
                f"hierarchical bootstrap 95% CI [{effect['ci95_low']:+.4f}, "
                f"{effect['ci95_high']:+.4f}] across seeds {comparison['seeds']}."
            )

    union = payload.get("coverage_union") or {}
    if union:
        lines.extend([
            "",
            "## Candidate Coverage",
            "",
            f"The available core ARM64 pools solve {union['tasks_solved']}/{EXPECTED_ROWS} "
            "tasks in oracle union coverage. This is a pool ceiling, not deployable pass@1.",
        ])
    lines.extend([
        "",
        "## Scope",
        "",
        "This is a cross-ISA and longer-function replication on real Flutter ARM64 "
        "release-binary slices with synthetic algorithmic semantics. It does not by "
        "itself establish performance on organic application business logic.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results-20260713")
    parser.add_argument(
        "--output_json",
        default="results-20260713/arm64_graphv21_study_analysis.json",
    )
    parser.add_argument(
        "--output_md",
        default="results-20260713/ARM64_GRAPHV21_STUDY_ANALYSIS.md",
    )
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260713)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = discover_runs(results_dir)
    if not runs:
        raise SystemExit(f"No complete ARM64 Graph-v2.1 runs under {results_dir}")
    rng = random.Random(args.bootstrap_seed)
    pairs = {
        "CFG vs no edges": ("prefix_cfg", "prefix_no_edges"),
        "CFG+DFG vs CFG": ("prefix_cfg_dfg", "prefix_cfg"),
        "no-GINE vs no-edge GINE": ("prefix_no_gine", "prefix_no_edges"),
        "shuffled vs CFG+DFG": ("prefix_shuffled", "prefix_cfg_dfg"),
        "no-edge prefix vs text": ("prefix_no_edges", "text"),
    }
    comparisons = {
        label: compare_variants(runs, arm, baseline, args.bootstrap_reps, rng)
        for label, (arm, baseline) in pairs.items()
        if any(variant == arm for _, variant in runs)
        and any(variant == baseline for _, variant in runs)
    }
    variants = sorted({variant for _, variant in runs})
    payload = {
        "schema": "antigravity-arm64-graphv21-analysis-v1",
        "expected_rows": EXPECTED_ROWS,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "run_count": len(runs),
        "variant_summaries": {
            variant: seed_summary(runs, variant) for variant in variants
        },
        "comparisons": comparisons,
        "coverage_union": coverage_union(runs, CORE_VARIANTS),
        "artifact_consistency": {
            "passed": all(run["summary_consistency"]["passed"] for run in runs.values()),
            "mismatches": {
                run["stem"]: run["summary_consistency"]["mismatches"]
                for run in runs.values()
                if not run["summary_consistency"]["passed"]
            },
        },
        "runs": {
            f"{seed}:{variant}": {
                "stem": run["stem"],
                "stats": run["stats"],
                "paths": run["paths"],
            }
            for (seed, variant), run in sorted(runs.items())
        },
    }
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
