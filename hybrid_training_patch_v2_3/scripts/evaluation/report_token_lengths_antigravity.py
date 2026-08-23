#!/usr/bin/env python3
"""Measure prompt/target token distributions before any GPU training stage.

The report quantifies how many examples would have been silently truncated by
historical limits and fails closed when the configured training or generation
budget cannot represent a complete supervised target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (  # noqa: E402
    build_decoder_prompt,
    target_source_from_record,
)
from scripts.training.hybrid_data_controls import (  # noqa: E402
    instruction_count,
    read_jsonl_many,
    source_text,
    task_identity,
)


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def stratum(count: int, short_max: int, bridge_max: int) -> str:
    if count <= short_max:
        return "short"
    if count <= bridge_max:
        return "bridge"
    return "long"


def summary(values: list[int], historical_limits: list[int]) -> dict[str, Any]:
    total = len(values)
    return {
        "count": total,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": (sum(values) / total) if total else 0.0,
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "over_historical_limits": {
            str(limit): {
                "count": sum(value > limit for value in values),
                "fraction": (sum(value > limit for value in values) / total) if total else 0.0,
            }
            for limit in historical_limits
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder_model", required=True)
    parser.add_argument("--decoder_revision", default="")
    parser.add_argument("--max_target_tokens", type=int, required=True)
    parser.add_argument("--max_prompt_tokens", type=int, required=True)
    parser.add_argument("--max_generation_tokens", type=int, required=True)
    parser.add_argument("--short_max", type=int, default=150)
    parser.add_argument("--bridge_max", type=int, default=199)
    parser.add_argument("--historical_limits", default="768,1024,2048")
    parser.add_argument("--prompt_mode", choices=["full", "graph_only", "none"], default="full")
    parser.add_argument("--fail_on_overflow", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    for name in ("max_target_tokens", "max_prompt_tokens", "max_generation_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.bridge_max < args.short_max:
        parser.error("--bridge_max must be >= --short_max")

    historical_limits = sorted(
        {int(value.strip()) for value in args.historical_limits.split(",") if value.strip()}
    )
    if any(value <= 0 for value in historical_limits):
        parser.error("historical limits must be positive")

    os.environ["GRAPH_PROMPT_ASSEMBLY_MODE"] = args.prompt_mode
    tokenizer = AutoTokenizer.from_pretrained(
        args.decoder_model,
        revision=args.decoder_revision or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_rows: list[tuple[str, dict[str, Any]]] = []
    for raw_path in args.dataset:
        resolved = str(Path(raw_path).expanduser().resolve())
        dataset_rows.extend((resolved, row) for row in read_jsonl_many([raw_path]))
    if not dataset_rows:
        raise SystemExit("token-length preflight received an empty dataset")

    records: list[dict[str, Any]] = []
    by_stratum: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"target": [], "prompt": []}
    )
    by_task: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"target": [], "prompt": []}
    )
    overflow: list[dict[str, Any]] = []

    for index, (dataset_path, row) in enumerate(dataset_rows):
        raw_source = source_text(row)
        target = target_source_from_record(row, raw_source)
        target_ids = tokenizer(
            target, add_special_tokens=True, truncation=False, padding=False
        )["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_ids = list(target_ids) + [eos_id]
        target_tokens = len(target_ids)

        prompt = build_decoder_prompt(row, tokenizer, args.max_prompt_tokens)
        prompt_tokens = len(
            tokenizer(prompt, add_special_tokens=True, truncation=False, padding=False)[
                "input_ids"
            ]
        )
        instructions = instruction_count(row)
        bucket = stratum(instructions, args.short_max, args.bridge_max)
        task = str(row.get("training_task") or "code")
        by_stratum[bucket]["target"].append(target_tokens)
        by_stratum[bucket]["prompt"].append(prompt_tokens)
        by_task[task]["target"].append(target_tokens)
        by_task[task]["prompt"].append(prompt_tokens)

        reasons = []
        if target_tokens > args.max_target_tokens:
            reasons.append("target_budget")
        if target_tokens > args.max_generation_tokens and task != "region_plan":
            reasons.append("generation_budget")
        if prompt_tokens > args.max_prompt_tokens:
            reasons.append("prompt_budget")
        if reasons:
            overflow.append(
                {
                    "index": index,
                    "dataset": dataset_path,
                    "task_id": task_identity(row),
                    "training_task": task,
                    "instruction_count": instructions,
                    "stratum": bucket,
                    "target_tokens": target_tokens,
                    "prompt_tokens": prompt_tokens,
                    "reasons": reasons,
                }
            )
        records.append(
            {
                "index": index,
                "dataset": dataset_path,
                "task_id": task_identity(row),
                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "training_task": task,
                "instruction_count": instructions,
                "stratum": bucket,
                "target_tokens": target_tokens,
                "prompt_tokens": prompt_tokens,
            }
        )

    target_values = [record["target_tokens"] for record in records]
    prompt_values = [record["prompt_tokens"] for record in records]

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)

    unique_code: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record["training_task"] not in {"code", "code_from_region_plan"}:
            continue
        key = (record["task_id"], record["target_sha256"])
        unique_code.setdefault(key, record)
    unique_code_records = list(unique_code.values())

    report = {
        "schema_version": 1,
        "datasets": [str(Path(path).expanduser().resolve()) for path in args.dataset],
        "decoder_model": args.decoder_model,
        "decoder_revision": args.decoder_revision or None,
        "rows": len(records),
        "budgets": {
            "target_tokens": args.max_target_tokens,
            "prompt_tokens": args.max_prompt_tokens,
            "generation_tokens": args.max_generation_tokens,
        },
        "historical_limits": historical_limits,
        "overall": {
            "target": summary(target_values, historical_limits),
            "prompt": summary(prompt_values, historical_limits),
        },
        "unique_code_targets": {
            "rows": len(unique_code_records),
            "target": summary(
                [record["target_tokens"] for record in unique_code_records],
                historical_limits,
            ),
            "by_instruction_stratum": {
                bucket: summary(
                    [
                        record["target_tokens"]
                        for record in unique_code_records
                        if record["stratum"] == bucket
                    ],
                    historical_limits,
                )
                for bucket in ("short", "bridge", "long")
            },
        },
        "by_dataset": {
            dataset: {
                "rows": len(dataset_records),
                "target": summary(
                    [record["target_tokens"] for record in dataset_records],
                    historical_limits,
                ),
                "prompt": summary(
                    [record["prompt_tokens"] for record in dataset_records],
                    historical_limits,
                ),
            }
            for dataset, dataset_records in sorted(by_dataset.items())
        },
        "by_instruction_stratum": {
            bucket: {
                "target": summary(values["target"], historical_limits),
                "prompt": summary(values["prompt"], historical_limits),
            }
            for bucket, values in sorted(by_stratum.items())
        },
        "by_training_task": {
            task: {
                "target": summary(values["target"], historical_limits),
                "prompt": summary(values["prompt"], historical_limits),
            }
            for task, values in sorted(by_task.items())
        },
        "stratum_counts": dict(Counter(record["stratum"] for record in records)),
        "training_task_counts": dict(Counter(record["training_task"] for record in records)),
        "overflow_count": len(overflow),
        "overflow": overflow,
        "largest_targets": sorted(
            records, key=lambda value: value["target_tokens"], reverse=True
        )[:25],
        "largest_prompts": sorted(
            records, key=lambda value: value["prompt_tokens"], reverse=True
        )[:25],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if overflow and args.fail_on_overflow:
        raise SystemExit(
            f"token-length preflight found {len(overflow)} rows outside the configured "
            f"train/generation budgets; see {args.output}"
        )


if __name__ == "__main__":
    main()
