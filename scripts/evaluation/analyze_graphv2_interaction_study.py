"""Analyze the seed-42 Graph-v2 architecture and encoder interaction cells.

This is an offline replay over archived candidate pools. It exists separately
from the legacy clean-study analyzer because frozen-encoder artifact stems use a
different model prefix that the legacy discovery routine does not recognize.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from analyze_graphv2_clean_study import (
    apply_stability_corrections,
    bootstrap_paired_diff,
    load_stability_corrections,
    mcnemar_exact,
    percentile,
    read_csv,
    read_json,
    sha256,
    summarize_stats,
)


DATASET_SHA256 = "8453876a40d2279684a190a5bf1430a62897c84e063a78e25c57198287bc6928"
DECODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GCB_REVISION = "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"
CLAP_REVISION = "620f4beba2edce172e8f35e263399716494950c9"

GCB_PREFIX = "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_graphv2_clean_s42_"
CLAP_PREFIX = "qwen3-8b-base_lora_enc_dec_r64_5e6_clap_a128_graphv2_clean_s42_"
FROZEN_GCB_PREFIX = "qwen3-8b-base_freeze_enc_lora_dec_5e6_gcb_a128_graphv2_clean_s42_"
FROZEN_CLAP_PREFIX = "qwen3-8b-base_freeze_enc_lora_dec_5e6_clap_a128_graphv2_clean_s42_"


CELLS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "GCB no-GINE (trainable)",
        "stem": GCB_PREFIX + "prefix_no_gine",
    },
    "gcb_cfg_gine": {
        "label": "GCB CFG-GINE (trainable)",
        "stem": GCB_PREFIX + "prefix_cfg",
    },
    "regions4": {
        "label": "Regions 4",
        "stem": GCB_PREFIX + "prefix_no_gine_regions4",
    },
    "regions8": {
        "label": "Regions 8",
        "stem": GCB_PREFIX + "prefix_no_gine_regions",
    },
    "regions16": {
        "label": "Regions 16",
        "stem": GCB_PREFIX + "prefix_no_gine_regions16",
    },
    "clap_trainable": {
        "label": "CLAP no-GINE (trainable)",
        "stem": CLAP_PREFIX + "prefix_no_gine_clap",
    },
    "clap_frozen": {
        "label": "CLAP no-GINE (frozen)",
        "stem": FROZEN_CLAP_PREFIX + "prefix_no_gine_clap_frozen_encoder",
    },
    "clap_cfg_gine": {
        "label": "CLAP CFG-GINE (trainable)",
        "stem": CLAP_PREFIX + "prefix_cfg_clap",
    },
    "multivector2": {
        "label": "Multivector 2",
        "stem": GCB_PREFIX + "prefix_no_gine_multivector2",
    },
    "multivector4": {
        "label": "Multivector 4",
        "stem": GCB_PREFIX + "prefix_no_gine_multivector4",
    },
    "multivector8": {
        "label": "Multivector 8",
        "stem": GCB_PREFIX + "prefix_no_gine_multivector8",
    },
    "no_attention": {
        "label": "No global attention",
        "stem": GCB_PREFIX + "prefix_no_gine_no_attention",
    },
    "gine2": {
        "label": "GINE 2 layers",
        "stem": GCB_PREFIX + "prefix_no_edges_gine2",
    },
    "no_position": {
        "label": "No block position",
        "stem": GCB_PREFIX + "prefix_no_gine_no_positions",
    },
    "gcb_frozen": {
        "label": "GCB no-GINE (frozen)",
        "stem": FROZEN_GCB_PREFIX + "prefix_no_gine_frozen_encoder",
    },
}

FULL_SCREEN = (
    "baseline",
    "regions4",
    "regions8",
    "regions16",
    "clap_trainable",
    "clap_frozen",
    "clap_cfg_gine",
    "multivector2",
    "multivector4",
    "multivector8",
    "no_attention",
    "gine2",
    "no_position",
    "gcb_frozen",
)

INTERACTIONS = {
    "encoder_family_x_trainability": (
        "baseline",
        "gcb_frozen",
        "clap_trainable",
        "clap_frozen",
    ),
    "encoder_family_x_graph_propagation": (
        "baseline",
        "gcb_cfg_gine",
        "clap_trainable",
        "clap_cfg_gine",
    ),
}

FACTOR_CONTRASTS = {
    "encoder_family_x_trainability": {
        "freeze_effect_gcb": {"gcb_frozen": 1, "baseline": -1},
        "freeze_effect_clap": {"clap_frozen": 1, "clap_trainable": -1},
        "clap_effect_trainable": {"clap_trainable": 1, "baseline": -1},
        "clap_effect_frozen": {"clap_frozen": 1, "gcb_frozen": -1},
        "interaction": {
            "clap_frozen": 1,
            "clap_trainable": -1,
            "gcb_frozen": -1,
            "baseline": 1,
        },
    },
    "encoder_family_x_graph_propagation": {
        "cfg_gine_effect_gcb": {"gcb_cfg_gine": 1, "baseline": -1},
        "cfg_gine_effect_clap": {"clap_cfg_gine": 1, "clap_trainable": -1},
        "clap_effect_no_gine": {"clap_trainable": 1, "baseline": -1},
        "clap_effect_cfg_gine": {"clap_cfg_gine": 1, "gcb_cfg_gine": -1},
        "interaction": {
            "clap_cfg_gine": 1,
            "clap_trainable": -1,
            "gcb_cfg_gine": -1,
            "baseline": 1,
        },
    },
}

ARCHITECTURE_EXPECTATIONS = {
    "baseline": ("microsoft/graphcodebert-base", GCB_REVISION, "lora", "0", "full", "identity"),
    "gcb_frozen": ("microsoft/graphcodebert-base", GCB_REVISION, "none", "1", "full", "identity"),
    "gcb_cfg_gine": ("microsoft/graphcodebert-base", GCB_REVISION, "lora", "0", "cfg", "full"),
    "clap_trainable": ("hustcw/clap-asm", CLAP_REVISION, "lora", "0", "full", "identity"),
    "clap_frozen": ("hustcw/clap-asm", CLAP_REVISION, "none", "1", "full", "identity"),
    "clap_cfg_gine": ("hustcw/clap-asm", CLAP_REVISION, "lora", "0", "cfg", "full"),
}

# These values are archived in graphv2_interaction_cells_s42_20260715T194554Z.log.
QUEUE_RUNTIME = {
    "clap_frozen": {"train_runtime_seconds": 5133.0, "final_eval_loss": 0.5931},
    "clap_cfg_gine": {"train_runtime_seconds": 6128.0, "final_eval_loss": 0.5841},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def output_sha_matches(path: Path, provenance: dict[str, Any]) -> bool:
    return provenance.get("output", {}).get("sha256") == sha256(path)


def load_cell(
    key: str,
    spec: dict[str, Any],
    results_dir: Path,
    corrections: dict[str, Any],
) -> dict[str, Any]:
    stem = spec["stem"]
    sweeps = results_dir / "sweeps_antigravity"
    paths = {
        "summary": sweeps / f"{stem}.json",
        "compile_stats": sweeps / f"{stem}_compile_stats.csv",
        "pass_stats": sweeps / f"{stem}_pass_stats.csv",
        "compile_predictions": results_dir / f"{stem}_compile_predictions.json",
        "compile_provenance": results_dir / f"{stem}_compile_predictions.json.provenance.json",
        "pass_predictions": results_dir / f"{stem}_pass_predictions.json",
        "pass_provenance": results_dir / f"{stem}_pass_predictions.json.provenance.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    require(not missing, f"{key}: missing artifacts: {missing}")

    summary = read_json(paths["summary"])
    compile_provenance = read_json(paths["compile_provenance"])
    pass_provenance = read_json(paths["pass_provenance"])
    predictions = read_json(paths["pass_predictions"])
    stats_rows = read_csv(paths["pass_stats"])
    applied = apply_stability_corrections(
        stats_rows, stem, corrections.get("corrections", [])
    )
    stats = summarize_stats(stats_rows, predictions)

    require(stats["rows"] == 154, f"{key}: expected 154 rows, found {stats['rows']}")
    require(
        sha256(paths["compile_predictions"]) == sha256(paths["pass_predictions"]),
        f"{key}: compile and pass candidate pools differ",
    )
    for phase, provenance, prediction_path in (
        ("compile", compile_provenance, paths["compile_predictions"]),
        ("pass", pass_provenance, paths["pass_predictions"]),
    ):
        require(provenance.get("row_count") == 154, f"{key}: {phase} row_count drift")
        require(provenance.get("seed") == 42, f"{key}: {phase} seed drift")
        require(
            provenance.get("dataset", {}).get("sha256") == DATASET_SHA256,
            f"{key}: {phase} dataset drift",
        )
        require(
            provenance.get("prompt_schema_version") == "antigravity-v2-no-test-hints",
            f"{key}: {phase} prompt schema drift",
        )
        require(
            provenance.get("scoring_tests_visible_to_policy") is False,
            f"{key}: {phase} exposes scoring tests",
        )
        require(
            provenance.get("models", {}).get("decoder", {}).get("resolved_commit")
            == DECODER_REVISION,
            f"{key}: {phase} decoder revision drift",
        )
        require(output_sha_matches(prediction_path, provenance), f"{key}: {phase} output hash drift")

    reused_from = Path(str(pass_provenance.get("reused_candidate_pool_from", ""))).name
    require(
        reused_from == paths["compile_predictions"].name,
        f"{key}: pass evaluation did not reuse the compile candidate pool",
    )

    env = pass_provenance.get("graph_environment", {})
    model = pass_provenance.get("models", {}).get("encoder", {})
    if key in ARCHITECTURE_EXPECTATIONS:
        expected = ARCHITECTURE_EXPECTATIONS[key]
        actual = (
            model.get("requested_id"),
            model.get("resolved_commit"),
            env.get("GRAPH_ENCODER_PEFT"),
            env.get("GRAPH_FREEZE_ENCODER"),
            env.get("GRAPH_EDGE_ABLATION"),
            env.get("GRAPH_GNN_ABLATION"),
        )
        require(actual == expected, f"{key}: architecture drift: expected {expected}, found {actual}")
        require(env.get("GRAPH_DFG_MODE") == "edges", f"{key}: DFG source drift")

    pooling = env.get("GRAPH_BLOCK_POOLING", "cls")
    vectors = int(env.get("GRAPH_BLOCK_VECTORS_PER_BLOCK", "4"))
    effective_vectors = vectors if pooling == "multi_query" else 1
    checkpoint_bytes = int(pass_provenance.get("checkpoint", {}).get("size_bytes") or 0)
    representation = {
        "pooling": pooling,
        "effective_vectors_per_block": effective_vectors,
        "region_compression": env.get("GRAPH_REGION_COMPRESSION", "off"),
        "region_max_blocks": int(env.get("GRAPH_REGION_MAX_BLOCKS", "8")),
        "global_attention": env.get("GRAPH_GLOBAL_ATTENTION_ABLATION", "full"),
        "block_positions": env.get("GRAPH_BLOCK_POSITION_MODE", "sinusoidal"),
        "edge_ablation": env.get("GRAPH_EDGE_ABLATION", "full"),
        "gnn_ablation": env.get("GRAPH_GNN_ABLATION", "identity"),
        "gnn_layers": int(env.get("GRAPH_GNN_LAYERS", "4")),
        "encoder_model": model.get("requested_id"),
        "encoder_trainable": env.get("GRAPH_FREEZE_ENCODER") != "1",
        "encoder_peft": env.get("GRAPH_ENCODER_PEFT"),
    }
    cost = {
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mib": checkpoint_bytes / (1024 * 1024),
        "candidate_pool_bytes": paths["pass_predictions"].stat().st_size,
        **QUEUE_RUNTIME.get(key, {}),
    }
    return {
        "key": key,
        "label": spec["label"],
        "stem": stem,
        "summary": summary,
        "stats": stats,
        "problem_ids": [str(row["problem_id"]) for row in stats_rows],
        "representation": representation,
        "cost": cost,
        "stability_corrections": applied,
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": {name: sha256(path) for name, path in paths.items()},
    }


def paired_effects(
    cell: dict[str, Any], baseline: dict[str, Any], reps: int, seed: int
) -> dict[str, Any]:
    require(cell["problem_ids"] == baseline["problem_ids"], f"{cell['key']}: task order drift")
    effects: dict[str, Any] = {}
    for metric in (
        "pass_at_1",
        "pass_at_5",
        "pass_at_10",
        "compile_at_1",
        "compile_at_5",
        "compile_at_10",
        "best_codebleu",
    ):
        effects[metric] = bootstrap_paired_diff(
            cell["stats"]["task_metrics"][metric],
            baseline["stats"]["task_metrics"][metric],
            reps,
            random.Random(f"{seed}:{cell['key']}:{metric}"),
        )
    effects["pass_at_10_coverage"] = mcnemar_exact(
        cell["stats"]["pass_coverage"], baseline["stats"]["pass_coverage"]
    )
    return effects


def bootstrap_factor_contrast(
    cells: dict[str, Any],
    coefficients: dict[str, int],
    metric: str,
    reps: int,
    seed: int,
    study: str,
    contrast: str,
) -> dict[str, float]:
    vectors = {
        key: cells[key]["stats"]["task_metrics"][metric]
        for key in coefficients
    }
    lengths = {len(values) for values in vectors.values()}
    require(len(lengths) == 1, f"{study}/{contrast}: paired vector length drift")
    n = lengths.pop()
    per_task = [
        sum(coefficient * vectors[key][index] for key, coefficient in coefficients.items())
        for index in range(n)
    ]
    rng = random.Random(f"{seed}:{study}:{contrast}:{metric}")
    estimates = [
        sum(per_task[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(reps)
    ]
    return {
        "difference": statistics.fmean(per_task),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def factor_contrasts(
    cells: dict[str, Any], reps: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for study, contrasts in FACTOR_CONTRASTS.items():
        result[study] = {}
        for contrast, coefficients in contrasts.items():
            result[study][contrast] = {
                metric: bootstrap_factor_contrast(
                    cells,
                    coefficients,
                    metric,
                    reps,
                    seed,
                    study,
                    contrast,
                )
                for metric in (
                    "pass_at_1",
                    "pass_at_5",
                    "pass_at_10",
                    "compile_at_1",
                    "compile_at_5",
                )
            }
    return result


def block_strata(benchmark_path: Path, problem_ids: list[str]) -> dict[str, Any]:
    rows = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["task_id"]): len(row.get("cfg") or []) for row in rows}
    require(set(problem_ids) == set(by_id), "Benchmark and candidate pools contain different task IDs")
    ordered = sorted(range(len(problem_ids)), key=lambda index: by_id[problem_ids[index]])
    groups = {
        "low": ordered[: len(ordered) // 3],
        "mid": ordered[len(ordered) // 3 : 2 * len(ordered) // 3],
        "high": ordered[2 * len(ordered) // 3 :],
    }
    return {
        label: {
            "indices": indices,
            "tasks": len(indices),
            "range": [
                min(by_id[problem_ids[index]] for index in indices),
                max(by_id[problem_ids[index]] for index in indices),
            ],
        }
        for label, indices in groups.items()
    }


def complexity_screen(cells: dict[str, Any], strata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, entry in strata.items():
        indices = entry["indices"]
        rates = {
            key: statistics.fmean(cell["stats"]["task_metrics"]["pass_at_10"][index] for index in indices)
            for key, cell in cells.items()
        }
        baseline_rate = rates["baseline"]
        result[label] = {
            "tasks": entry["tasks"],
            "range": entry["range"],
            "pass_at_10": rates,
            "difference_vs_baseline": {key: value - baseline_rate for key, value in rates.items()},
        }
    return result


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_effect(effect: dict[str, float]) -> str:
    return f"{effect['difference']:+.4f} [{effect['ci95_low']:+.4f}, {effect['ci95_high']:+.4f}]"


def representation_label(cell: dict[str, Any]) -> str:
    rep = cell["representation"]
    if rep["pooling"] == "multi_query":
        label = f"{rep['effective_vectors_per_block']} query/block"
    else:
        label = "CLS/block"
    if rep["region_compression"] != "off":
        label += f" + regions/{rep['region_max_blocks']}"
    if rep["global_attention"] == "identity":
        label += " + no-attn"
    if rep["block_positions"] == "off":
        label += " + no-pos"
    if rep["gnn_ablation"] == "full":
        label += f" + {rep['gnn_layers']}L-GINE/{rep['edge_ablation']}"
    return label


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Graph-v2 Seed-42 Architecture Interaction Study",
        "",
        "This report replays archived candidate pools only. No model inference or Dart evaluation was run.",
        "",
        f"- Strict artifact/provenance validation: **PASS** for {payload['validated_cell_count']} cells.",
        "- All effects are paired against seed-42 GCB no-GINE and show 95% task-bootstrap intervals.",
        "- These N=154 cells are descriptive table-completion runs, below the decision floor.",
        "",
        "## Encoder Family x Trainability",
        "",
        "| Cell | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | checkpoint MiB | train runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in INTERACTIONS["encoder_family_x_trainability"]:
        cell = payload["cells"][key]
        metrics = cell["stats"]["metrics"]
        runtime = cell["cost"].get("train_runtime_seconds")
        runtime_text = f"{runtime / 60:.1f} min" if runtime is not None else "n/a"
        lines.append(
            f"| {cell['label']} | {fmt(metrics['pass_at_1'])} | {fmt(metrics['pass_at_5'])} | "
            f"{fmt(metrics['pass_at_10'])} | {fmt(metrics['compile_at_1'])} | "
            f"{fmt(metrics['compile_at_5'])} | {cell['cost']['checkpoint_mib']:.1f} | {runtime_text} |"
        )
    lines.extend(
        [
            "",
            "## Encoder Family x Graph Propagation",
            "",
            "| Cell | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | checkpoint MiB | train runtime |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in INTERACTIONS["encoder_family_x_graph_propagation"]:
        cell = payload["cells"][key]
        metrics = cell["stats"]["metrics"]
        runtime = cell["cost"].get("train_runtime_seconds")
        runtime_text = f"{runtime / 60:.1f} min" if runtime is not None else "n/a"
        lines.append(
            f"| {cell['label']} | {fmt(metrics['pass_at_1'])} | {fmt(metrics['pass_at_5'])} | "
            f"{fmt(metrics['pass_at_10'])} | {fmt(metrics['compile_at_1'])} | "
            f"{fmt(metrics['compile_at_5'])} | {cell['cost']['checkpoint_mib']:.1f} | {runtime_text} |"
        )
    lines.extend(
        [
            "",
            "## Factor Contrasts",
            "",
            "The interaction row is the paired difference-in-differences. Positive trainability interaction means freezing hurts CLAP less than GCB; negative propagation interaction means CFG-GINE hurts CLAP more than GCB.",
            "",
            "| Study | Contrast | delta pass@1 | delta pass@5 | delta pass@10 | delta compile@1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    contrast_labels = {
        "freeze_effect_gcb": "Frozen - trainable, GCB",
        "freeze_effect_clap": "Frozen - trainable, CLAP",
        "clap_effect_trainable": "CLAP - GCB, trainable",
        "clap_effect_frozen": "CLAP - GCB, frozen",
        "cfg_gine_effect_gcb": "CFG-GINE - no-GINE, GCB",
        "cfg_gine_effect_clap": "CFG-GINE - no-GINE, CLAP",
        "clap_effect_no_gine": "CLAP - GCB, no-GINE",
        "clap_effect_cfg_gine": "CLAP - GCB, CFG-GINE",
        "interaction": "Interaction",
    }
    study_labels = {
        "encoder_family_x_trainability": "Family x trainability",
        "encoder_family_x_graph_propagation": "Family x propagation",
    }
    for study, contrasts in payload["factor_contrasts"].items():
        for contrast, metrics in contrasts.items():
            lines.append(
                f"| {study_labels[study]} | {contrast_labels[contrast]} | "
                f"{fmt_effect(metrics['pass_at_1'])} | {fmt_effect(metrics['pass_at_5'])} | "
                f"{fmt_effect(metrics['pass_at_10'])} | {fmt_effect(metrics['compile_at_1'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired Effects vs Baseline",
            "",
            "| Variant | delta pass@1 | delta pass@5 | delta pass@10 | delta compile@1 | gains/losses at k=10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in FULL_SCREEN:
        cell = payload["cells"][key]
        effects = payload["paired_vs_baseline"][key]
        coverage = effects["pass_at_10_coverage"]
        lines.append(
            f"| {cell['label']} | {fmt_effect(effects['pass_at_1'])} | "
            f"{fmt_effect(effects['pass_at_5'])} | {fmt_effect(effects['pass_at_10'])} | "
            f"{fmt_effect(effects['compile_at_1'])} | {coverage['gains']}/{coverage['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Full Architecture Screen",
            "",
            "| Variant | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | CodeBLEU | solved |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in FULL_SCREEN:
        cell = payload["cells"][key]
        metrics = cell["stats"]["metrics"]
        lines.append(
            f"| {cell['label']} | {fmt(metrics['pass_at_1'])} | {fmt(metrics['pass_at_5'])} | "
            f"{fmt(metrics['pass_at_10'])} | {fmt(metrics['compile_at_1'])} | "
            f"{fmt(metrics['compile_at_5'])} | {fmt(metrics['best_codebleu'])} | "
            f"{cell['stats']['tasks_with_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Block-Count Strata (pass@10)",
            "",
            "| Variant | low | mid | high |",
            "|---|---:|---:|---:|",
        ]
    )
    complexity = payload["complexity_strata"]
    for key in FULL_SCREEN:
        cell = payload["cells"][key]
        lines.append(
            f"| {cell['label']} | {fmt(complexity['low']['pass_at_10'][key])} | "
            f"{fmt(complexity['mid']['pass_at_10'][key])} | "
            f"{fmt(complexity['high']['pass_at_10'][key])} |"
        )
    lines.extend(
        [
            "",
            "## Representation and Artifact Cost",
            "",
            "| Variant | representation | encoder | trainable | checkpoint MiB | candidate pool MiB |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for key in FULL_SCREEN:
        cell = payload["cells"][key]
        rep = cell["representation"]
        encoder = "CLAP" if rep["encoder_model"] == "hustcw/clap-asm" else "GCB"
        lines.append(
            f"| {cell['label']} | {representation_label(cell)} | {encoder} | "
            f"{'yes' if rep['encoder_trainable'] else 'no'} | {cell['cost']['checkpoint_mib']:.1f} | "
            f"{cell['cost']['candidate_pool_bytes'] / (1024 * 1024):.2f} |"
        )
    lines.extend(
        [
            "",
            "Runtime note: exact trainer runtime was available in the just-completed queue for frozen CLAP "
            "(85.5 min) and CLAP CFG-GINE (102.1 min). Older queue logs are not encoded in the seven result artifacts, "
            "so their runtime cells are intentionally reported as unavailable rather than inferred.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results-20260713")
    parser.add_argument("--benchmark", default="data/testing/grpo_data_graphv2.jsonl")
    parser.add_argument(
        "--output_json", default="results-20260713/graphv2_interaction_study_analysis.json"
    )
    parser.add_argument(
        "--output_md", default="results-20260713/GRAPHV2_INTERACTION_STUDY_ANALYSIS.md"
    )
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260716)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    corrections = load_stability_corrections(results_dir)
    cells = {
        key: load_cell(key, spec, results_dir, corrections)
        for key, spec in CELLS.items()
    }
    baseline = cells["baseline"]
    paired = {
        key: paired_effects(cell, baseline, args.bootstrap_reps, args.bootstrap_seed)
        for key, cell in cells.items()
    }
    contrasts = factor_contrasts(cells, args.bootstrap_reps, args.bootstrap_seed)
    strata = block_strata(Path(args.benchmark), baseline["problem_ids"])
    complexity = complexity_screen(cells, strata)
    for entry in complexity.values():
        entry.pop("indices", None)

    payload = {
        "schema": "antigravity-graphv2-interaction-study-v1",
        "method": {
            "seed": 42,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "baseline": "baseline",
            "candidate_replay": True,
            "model_inference_rerun": False,
            "dart_rerun": False,
        },
        "validated_cell_count": len(cells),
        "interaction_tables": INTERACTIONS,
        "factor_contrasts": contrasts,
        "full_screen": FULL_SCREEN,
        "paired_vs_baseline": paired,
        "complexity_strata": complexity,
        "cells": cells,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


if __name__ == "__main__":
    main()
