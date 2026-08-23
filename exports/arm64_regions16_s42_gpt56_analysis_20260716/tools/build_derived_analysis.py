#!/usr/bin/env python3
"""Build analysis-friendly ARM64 tables from the immutable eval split and stats."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable


def pass_at_k(successes: int, k: int, n: int = 10) -> float:
    if successes <= 0:
        return 0.0
    if k >= n:
        return 1.0
    return 1.0 - math.comb(n - successes, k) / math.comb(n, k)


def text_length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, sort_keys=True, ensure_ascii=False))


def sequence_length(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, dict)) else 0


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def aggregate(
    dimension: str,
    label: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    lengths = [integer(row["reference_length"]) for row in rows]
    return {
        "dimension": dimension,
        "bin": label,
        "rows": len(rows),
        "reference_length_min": min(lengths) if lengths else 0,
        "reference_length_mean": round(mean(lengths), 3),
        "reference_length_max": max(lengths) if lengths else 0,
        "compile_at_1": round(mean(float(row["compile_at_1"]) for row in rows), 9),
        "compile_at_5": round(mean(float(row["compile_at_5"]) for row in rows), 9),
        "pass_at_1": round(mean(float(row["pass_at_1"]) for row in rows), 9),
        "pass_at_5": round(mean(float(row["pass_at_5"]) for row in rows), 9),
        "pass_at_10": round(mean(float(row["pass_at_10"]) for row in rows), 9),
        "tasks_with_any_compile": sum(integer(row["any_compile"]) for row in rows),
        "tasks_with_any_pass": sum(integer(row["any_pass"]) for row in rows),
        "any_compile_rate": round(mean(float(row["any_compile"]) for row in rows), 9),
        "any_pass_rate": round(mean(float(row["any_pass"]) for row in rows), 9),
        "mean_codebleu_all_candidates": round(
            mean(float(row["mean_codebleu_all_candidates"]) for row in rows), 9
        ),
    }


def add_binned_groups(
    output: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    dimension: str,
    bins: list[tuple[str, Callable[[dict[str, Any]], bool]]],
) -> None:
    for label, predicate in bins:
        group = [row for row in rows if predicate(row)]
        if group:
            output.append(aggregate(dimension, label, group))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.workspace.resolve()
    output = args.output.resolve()
    stem = (
        "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_"
        "arm64v21_s42_prefix_no_gine_regions16"
    )
    result_dir = root / "results-20260716" / "arm64_regions16_s42"
    stats_path = result_dir / f"{stem}_pass_stats.csv"
    summary_path = result_dir / f"{stem}.json"
    provenance_path = result_dir / "run_provenance.json"
    eval_path = (
        root
        / "data"
        / "datasets"
        / "arm64_graphv2"
        / "flutter_eval_graphv2.jsonl"
    )

    eval_rows = load_jsonl(eval_path)
    eval_by_id = {str(row["task_id"]): row for row in eval_rows}
    if len(eval_by_id) != len(eval_rows):
        raise ValueError("The eval split contains duplicate task_id values")

    with stats_path.open("r", encoding="utf-8", newline="") as handle:
        stats_rows = list(csv.DictReader(handle))

    if len(stats_rows) != 343 or len(eval_rows) != 343:
        raise ValueError(
            f"Expected 343 stats and eval rows, got {len(stats_rows)} and {len(eval_rows)}"
        )

    per_task: list[dict[str, Any]] = []
    for stats in stats_rows:
        task_id = str(stats["problem_id"])
        if task_id not in eval_by_id:
            raise ValueError(f"Stats task is absent from eval split: {task_id}")
        source = eval_by_id[task_id]
        integrity = source.get("integrity") or {}

        compile_flags = [integer(stats[f"cand_{i}_compile"]) for i in range(1, 11)]
        pass_flags = [integer(stats[f"cand_{i}_pass"]) for i in range(1, 11)]
        codebleu = [float(stats[f"cand_{i}_codebleu"]) for i in range(1, 11)]
        compile_count = sum(compile_flags)
        pass_count = sum(pass_flags)
        compiled_codebleu = [
            score for score, compiled in zip(codebleu, compile_flags) if compiled
        ]

        cfg_blocks = sequence_length(source.get("cfg"))
        parsed_instructions = integer(integrity.get("parsed_instruction_count"))
        tests = source.get("tests")
        tests_count = sequence_length(tests) if not isinstance(tests, str) else 1
        row = {
            "task_id": task_id,
            "filename": source.get("filename", ""),
            "category": source.get("category", ""),
            "difficulty": source.get("difficulty", ""),
            "generator_provider": source.get("generator_provider", ""),
            "generator_model": source.get("generator_model", ""),
            "reference_length": integer(stats["reference_length"]),
            "dart_source_chars": text_length(source.get("dart_source")),
            "function_chars": text_length(source.get("function")),
            "tests_count": tests_count,
            "tests_chars": text_length(tests),
            "assembly_chars": text_length(source.get("assembly")),
            "assembly_lines": str(source.get("assembly") or "").count("\n") + 1,
            "cfg_blocks": cfg_blocks,
            "region_count_ceil_blocks_div_16": math.ceil(cfg_blocks / 16) if cfg_blocks else 0,
            "blocks_over_first_region": max(cfg_blocks - 16, 0),
            "edge_records": sequence_length(source.get("edges")),
            "cfg_edge_count": integer(integrity.get("cfg_edge_count")),
            "dataflow_edge_count": integer(integrity.get("dataflow_edge_count")),
            "parsed_instruction_count": parsed_instructions,
            "isolated_nodes": integer(integrity.get("isolated_nodes")),
            "unreachable_nodes": integer(integrity.get("unreachable_nodes")),
            "compile_successes_of_10": compile_count,
            "pass_successes_of_10": pass_count,
            "any_compile": int(compile_count > 0),
            "any_pass": int(pass_count > 0),
            "compile_at_1": compile_count / 10.0,
            "compile_at_5": pass_at_k(compile_count, 5),
            "pass_at_1": pass_count / 10.0,
            "pass_at_5": pass_at_k(pass_count, 5),
            "pass_at_10": pass_at_k(pass_count, 10),
            "mean_codebleu_all_candidates": mean(codebleu),
            "mean_codebleu_compiled_candidates": (
                mean(compiled_codebleu) if compiled_codebleu else ""
            ),
            "long_reference_ge_1000": int(integer(stats["reference_length"]) >= 1000),
            "semantic_failure_all_10": int(pass_count == 0),
        }
        per_task.append(row)

    if set(eval_by_id) != {row["task_id"] for row in per_task}:
        raise ValueError("Eval and stats task ID sets do not match exactly")

    per_task.sort(key=lambda row: row["task_id"])
    fields = list(per_task[0])
    write_csv(output / "per_task_analysis.csv", per_task, fields)
    write_csv(
        output / "long_reference_semantic_failures.csv",
        [
            row
            for row in per_task
            if integer(row["long_reference_ge_1000"])
            and integer(row["semantic_failure_all_10"])
        ],
        fields,
    )
    write_csv(
        output / "semantic_successes.csv",
        [row for row in per_task if integer(row["any_pass"])],
        fields,
    )

    groups: list[dict[str, Any]] = []
    add_binned_groups(
        groups,
        per_task,
        "reference_length",
        [
            ("lt700", lambda row: integer(row["reference_length"]) < 700),
            ("700_999", lambda row: 700 <= integer(row["reference_length"]) < 1000),
            ("1000_1299", lambda row: 1000 <= integer(row["reference_length"]) < 1300),
            ("ge1300", lambda row: integer(row["reference_length"]) >= 1300),
        ],
    )
    add_binned_groups(
        groups,
        per_task,
        "cfg_blocks",
        [
            ("le16", lambda row: integer(row["cfg_blocks"]) <= 16),
            ("17_32", lambda row: 17 <= integer(row["cfg_blocks"]) <= 32),
            ("33_64", lambda row: 33 <= integer(row["cfg_blocks"]) <= 64),
            ("ge65", lambda row: integer(row["cfg_blocks"]) >= 65),
        ],
    )
    add_binned_groups(
        groups,
        per_task,
        "parsed_instruction_count",
        [
            ("le64", lambda row: integer(row["parsed_instruction_count"]) <= 64),
            ("65_128", lambda row: 65 <= integer(row["parsed_instruction_count"]) <= 128),
            ("129_256", lambda row: 129 <= integer(row["parsed_instruction_count"]) <= 256),
            ("ge257", lambda row: integer(row["parsed_instruction_count"]) >= 257),
        ],
    )
    for difficulty in sorted({str(row["difficulty"]) for row in per_task}):
        groups.append(
            aggregate(
                "difficulty",
                difficulty or "blank",
                [row for row in per_task if str(row["difficulty"]) == difficulty],
            )
        )
    for region_count in sorted(
        {integer(row["region_count_ceil_blocks_div_16"]) for row in per_task}
    ):
        groups.append(
            aggregate(
                "region_count_ceil_blocks_div_16",
                str(region_count),
                [
                    row
                    for row in per_task
                    if integer(row["region_count_ceil_blocks_div_16"]) == region_count
                ],
            )
        )

    group_fields = list(groups[0])
    write_csv(output / "stratified_metrics.csv", groups, group_fields)
    write_csv(
        output / "length_stratified_metrics.csv",
        [row for row in groups if row["dimension"] == "reference_length"],
        group_fields,
    )

    with summary_path.open("r", encoding="utf-8") as handle:
        official_summary = json.load(handle)
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)

    overall = aggregate("overall", "all", per_task)
    long_rows = [row for row in per_task if integer(row["reference_length"]) >= 1000]
    short_rows = [row for row in per_task if integer(row["reference_length"]) < 1000]
    derived_summary = {
        "schema_version": 1,
        "task_count": len(per_task),
        "candidate_count_per_task": 10,
        "candidate_count_total": len(per_task) * 10,
        "overall_derived_metrics": overall,
        "official_metrics": {
            "codebleu": official_summary["codebleu"],
            "codebleu_compiled_only": official_summary["codebleu_compiled_only"],
            "compile_at_k": official_summary["compile_at_k"],
            "pass_at_k": official_summary["pass_at_k"],
        },
        "metric_reproduction_absolute_error": {
            "compile_at_1": abs(
                overall["compile_at_1"]
                - float(official_summary["compile_at_k"]["compile_at_1"])
            ),
            "compile_at_5": abs(
                overall["compile_at_5"]
                - float(official_summary["compile_at_k"]["compile_at_5"])
            ),
            "pass_at_1": abs(
                overall["pass_at_1"]
                - float(official_summary["pass_at_k"]["pass_at_1"])
            ),
            "pass_at_5": abs(
                overall["pass_at_5"]
                - float(official_summary["pass_at_k"]["pass_at_5"])
            ),
            "pass_at_10": abs(
                overall["pass_at_10"]
                - float(official_summary["pass_at_k"]["pass_at_10"])
            ),
        },
        "length_break_observation": {
            "reference_length_is_a_proxy_not_a_causal_variable": True,
            "rows_reference_length_lt_1000": len(short_rows),
            "tasks_with_any_pass_reference_length_lt_1000": sum(
                integer(row["any_pass"]) for row in short_rows
            ),
            "rows_reference_length_ge_1000": len(long_rows),
            "tasks_with_any_pass_reference_length_ge_1000": sum(
                integer(row["any_pass"]) for row in long_rows
            ),
        },
        "provenance_checks": {
            "seed": provenance.get("seed"),
            "scoring_tests_visible_to_policy": provenance.get(
                "scoring_tests_visible_to_policy"
            ),
            "prompt_schema_version": provenance.get("prompt_schema_version"),
            "eval_dataset_sha256": next(
                item["sha256"]
                for item in provenance["datasets"]
                if item["path"].endswith("flutter_eval_graphv2.jsonl")
            ),
        },
    }
    max_error = max(derived_summary["metric_reproduction_absolute_error"].values())
    if max_error > 1e-8:
        raise ValueError(f"Derived pass/compile metrics disagree with official summary: {max_error}")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(derived_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({"status": "ok", "rows": len(per_task), "output": str(output)}))


if __name__ == "__main__":
    main()
