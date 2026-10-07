#!/usr/bin/env python3
"""Build all-length hierarchical SFT datasets for neural decompilation.

The builder keeps every Phase-0-approved training row, including functions with
200+ instructions. Long functions receive three supervised views:

1. direct whole-function code generation;
2. deterministic CFG-region planning from the same binary representation;
3. plan-conditioned whole-function reconstruction.

The final recovery dataset contains code-only targets, so the deployed model is
never required to emit or consume a teacher-only plan. Short rows are replayed
in the hierarchical stage to prevent catastrophic length-specialisation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.cfg_extractor import ensure_cfg_blocks  # type: ignore  # noqa: E402
from models.pyg_cfg_dataset import build_linear_region_ids  # type: ignore  # noqa: E402
from scripts.training.hybrid_data_controls import (  # noqa: E402
    SCHEMA_VERSION,
    instruction_count,
    length_bin,
    mechanical_facts,
    read_jsonl_many,
    source_text,
    task_identity,
    write_jsonl,
)

_CALL_RE = re.compile(r"\b(?:bl|blr|callq?|jal)\s+([^,\s]+)", re.I)
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:0x[0-9a-fA-F]+|\d+)(?![A-Za-z_])")
_STRING_RE = re.compile(r"(?:(?:\"(?:\\.|[^\"\\])*\")|(?:'(?:\\.|[^'\\])*'))")
_COMPARE_RE = re.compile(r"\b(?:cmp|cmn|tst|test|subs|fcmp|ccmp)\b", re.I)
_BRANCH_RE = re.compile(r"\b(?:b\.[a-z]+|cbz|cbnz|tbz|tbnz|j[a-z]+)\b", re.I)
_RETURN_RE = re.compile(r"\b(?:ret|retq)\b", re.I)


def _copy_with_task(
    row: dict[str, Any],
    task: str,
    target: str | None = None,
    *,
    short_max: int = 150,
    bridge_max: int = 199,
) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    copied["training_task"] = task
    if target is not None:
        copied["supervised_target"] = target
    metadata = copy.deepcopy(copied.get("hybrid_metadata") or {})
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "hierarchical_training_task": task,
            "instruction_stratum": _stratum(
                instruction_count(copied), short_max=short_max, bridge_max=bridge_max
            ),
            "length_bin": length_bin(instruction_count(copied)),
            "trained_in_all_length_curriculum": True,
        }
    )
    copied["hybrid_metadata"] = metadata
    return copied


def _stratum(count: int, short_max: int = 150, bridge_max: int = 199) -> str:
    if count <= short_max:
        return "short"
    if count <= bridge_max:
        return "bridge"
    return "long"


def _instruction_lines(block: dict[str, Any]) -> list[str]:
    instructions = block.get("instructions") or []
    if isinstance(instructions, str):
        return [line.strip() for line in instructions.splitlines() if line.strip()]
    return [str(line).strip() for line in instructions if str(line).strip()]


def _normalise_edge(edge: dict[str, Any]) -> dict[str, Any] | None:
    source = edge.get("source")
    target = edge.get("target")
    if not isinstance(source, int) or not isinstance(target, int):
        return None
    return {
        "source": source,
        "target": target,
        "type": str(edge.get("edge_type") or "unknown"),
    }


# Bounds that keep the deterministic region-plan JSON compact regardless of
# function size. ``max_region_blocks`` bounds region size, not region count, so
# these are the actual guardrails against unbounded plan growth.
_MAX_CALLS_PER_REGION = 8          # distinct call targets kept per region
_MAX_FACT_LIST = 16                # entries kept per list-valued function fact
_PLAN_CHAR_BUDGET = 6000           # serialized-plan char ceiling (~1.7k tokens)


def _region_plan(row: dict[str, Any], max_region_blocks: int) -> dict[str, Any]:
    cfg, edges = ensure_cfg_blocks(row)
    if not cfg:
        raise ValueError(f"{task_identity(row)}: no CFG blocks")
    clean_edges = [value for edge in edges if (value := _normalise_edge(edge)) is not None]
    block_types = [str(block.get("block_type") or "unknown") for block in cfg]
    region_tensor = build_linear_region_ids(
        len(cfg),
        edges,
        max_region_blocks=max_region_blocks,
        block_types=block_types,
    )
    region_ids = [int(value) for value in region_tensor.tolist()]
    grouped: dict[int, list[int]] = defaultdict(list)
    for block_index, region_id in enumerate(region_ids):
        grouped[region_id].append(block_index)

    regions: list[dict[str, Any]] = []
    for region_id in sorted(grouped):
        block_indices = grouped[region_id]
        lines = [line for index in block_indices for line in _instruction_lines(cfg[index])]
        calls = sorted({match.group(1) for line in lines for match in _CALL_RE.finditer(line)})
        constants = []
        for line in lines:
            for token in _NUMBER_RE.findall(line):
                if token not in constants:
                    constants.append(token)
                if len(constants) >= 12:
                    break
            if len(constants) >= 12:
                break
        strings = []
        for line in lines:
            for token in _STRING_RE.findall(line):
                if token not in strings:
                    strings.append(token)
                if len(strings) >= 6:
                    break
            if len(strings) >= 6:
                break
        region_edges = [
            edge
            for edge in clean_edges
            if edge["source"] in block_indices or edge["target"] in block_indices
        ]
        predecessors = sorted(
            {
                region_ids[edge["source"]]
                for edge in clean_edges
                if edge["target"] in block_indices
                and region_ids[edge["source"]] != region_id
            }
        )
        successors = sorted(
            {
                region_ids[edge["target"]]
                for edge in clean_edges
                if edge["source"] in block_indices
                and region_ids[edge["target"]] != region_id
            }
        )
        regions.append(
            {
                "region": region_id,
                # ``block_count`` + a bounded block-type histogram replace the raw
                # per-block index/type lists, which grew O(blocks) and made plans
                # for long functions explode past the prompt/target budget.
                "block_count": len(block_indices),
                "block_types": dict(
                    sorted(Counter(block_types[index] for index in block_indices).items())
                ),
                "instruction_count": len(lines),
                "predecessors": predecessors,
                "successors": successors,
                "edge_types": sorted({edge["type"] for edge in region_edges}),
                "calls": calls[:_MAX_CALLS_PER_REGION],
                "constants": constants,
                "string_literals": strings,
                "comparisons": sum(bool(_COMPARE_RE.search(line)) for line in lines),
                "conditional_branches": sum(bool(_BRANCH_RE.search(line)) for line in lines),
                "returns": sum(bool(_RETURN_RE.search(line)) for line in lines),
            }
        )

    facts = row.get("binary_facts") or mechanical_facts(row)
    function_facts: dict[str, Any] = {}
    for key in (
        "calls",
        "conditional_branches",
        "returns",
        "comparisons",
        "memory_ops",
        "salient_numeric_constants",
        "available_strings",
    ):
        if key in facts:
            value = facts[key]
            function_facts[key] = value[:_MAX_FACT_LIST] if isinstance(value, list) else value

    # Deduplicate inter-region control edges to distinct (source, target, type)
    # triples. The raw list emitted one entry per underlying block-level edge, so
    # it scaled O(edges) and drove region-plan JSON for long functions to tens of
    # thousands of tokens. Distinct region-level triples are the control skeleton.
    inter_seen: set[tuple[int, int, str]] = set()
    inter_region_edges: list[dict[str, Any]] = []
    for edge in clean_edges:
        src_region = region_ids[edge["source"]]
        tgt_region = region_ids[edge["target"]]
        if src_region == tgt_region:
            continue
        key = (src_region, tgt_region, edge["type"])
        if key in inter_seen:
            continue
        inter_seen.add(key)
        inter_region_edges.append(
            {"source_region": src_region, "target_region": tgt_region, "type": edge["type"]}
        )
    inter_region_edges.sort(
        key=lambda item: (item["source_region"], item["target_region"], item["type"])
    )

    plan: dict[str, Any] = {
        "schema": "antigravity-region-plan-v1",
        "function": "fn0",
        "instruction_count": instruction_count(row),
        "region_count": len(regions),
        "regions_emitted": len(regions),
        "entry_region": region_ids[0] if region_ids else 0,
        "function_facts": function_facts,
        "regions": list(regions),
        "inter_region_edges": inter_region_edges,
    }

    # ``max_region_blocks`` bounds region SIZE, not region COUNT, so region_count
    # is unbounded and a fully-detailed plan can still exceed the prompt/target
    # budget for very long functions. Trim trailing regions (and edges touching
    # them) until the serialized plan fits a hard character budget. The graph
    # soft-prefix carries the complete CFG; this text plan is a bounded scaffold.
    while len(_plan_text(plan)) > _PLAN_CHAR_BUDGET and len(plan["regions"]) > 1:
        dropped = plan["regions"].pop()["region"]
        plan["regions_truncated"] = True
        plan["inter_region_edges"] = [
            edge
            for edge in plan["inter_region_edges"]
            if edge["source_region"] != dropped and edge["target_region"] != dropped
        ]
    plan["regions_emitted"] = len(plan["regions"])
    return plan


def _plan_text(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_rank(row: dict[str, Any], seed: int, salt: str) -> str:
    material = f"{seed}|{salt}|{task_identity(row)}|{instruction_count(row)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _balanced_recovery_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    long_repeat: int,
    bridge_repeat: int,
    short_repeat: int,
    short_max: int = 150,
    bridge_max: int = 199,
) -> list[dict[str, Any]]:
    repeats = {"short": short_repeat, "bridge": bridge_repeat, "long": long_repeat}
    out: list[dict[str, Any]] = []
    for row in rows:
        stratum = _stratum(
            instruction_count(row), short_max=short_max, bridge_max=bridge_max
        )
        for replica in range(repeats[stratum]):
            copied = _copy_with_task(
                row, "code", short_max=short_max, bridge_max=bridge_max
            )
            metadata = copy.deepcopy(copied.get("hybrid_metadata") or {})
            metadata["curriculum_replica"] = replica
            metadata["curriculum_stage"] = "code_recovery"
            copied["hybrid_metadata"] = metadata
            out.append(copied)
    out.sort(key=lambda row: _stable_rank(row, seed, "recovery"))
    return out


def build_datasets(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    short_max: int,
    bridge_max: int,
    max_region_blocks: int,
    short_replay_fraction: float,
    long_repeat: int,
    bridge_repeat: int,
    short_repeat: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not rows:
        raise ValueError("approved all-length training input is empty")
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    plans: dict[int, dict[str, Any]] = {}
    for row in rows:
        count = instruction_count(row)
        stratum = "short" if count <= short_max else "bridge" if count <= bridge_max else "long"
        strata[stratum].append(row)
        if stratum in {"bridge", "long"}:
            plans[id(row)] = _region_plan(row, max_region_blocks)

    direct = [
        _copy_with_task(row, "code", short_max=short_max, bridge_max=bridge_max)
        for row in rows
    ]
    direct.sort(key=lambda row: _stable_rank(row, seed, "direct"))

    hierarchical: list[dict[str, Any]] = []
    matched_control: list[dict[str, Any]] = []
    for stratum in ("bridge", "long"):
        for row in strata[stratum]:
            plan = plans[id(row)]
            plan_text = _plan_text(plan)
            plan_row = _copy_with_task(
                row, "region_plan", plan_text, short_max=short_max, bridge_max=bridge_max
            )
            plan_row["hierarchical_region_plan"] = plan
            plan_row["hybrid_metadata"]["curriculum_stage"] = "region_plan"
            code_row = _copy_with_task(
                row, "code_from_region_plan", short_max=short_max, bridge_max=bridge_max
            )
            code_row["hierarchical_region_plan"] = plan
            code_row["hybrid_metadata"]["curriculum_stage"] = "plan_conditioned_code"
            direct_row = _copy_with_task(
                row, "code", short_max=short_max, bridge_max=bridge_max
            )
            direct_row["hierarchical_region_plan"] = plan
            direct_row["hybrid_metadata"]["curriculum_stage"] = "direct_long_code"
            hierarchical.extend((plan_row, code_row, direct_row))
            for replica in range(3):
                control_row = _copy_with_task(
                    row, "code", short_max=short_max, bridge_max=bridge_max
                )
                control_row["hybrid_metadata"]["curriculum_stage"] = "matched_direct_control"
                control_row["hybrid_metadata"]["curriculum_replica"] = replica
                matched_control.append(control_row)

    short_rows = sorted(strata["short"], key=lambda row: _stable_rank(row, seed, "short_replay"))
    desired_short = min(
        len(short_rows),
        max(0, int(round(len(hierarchical) * short_replay_fraction))),
    )
    for row in short_rows[:desired_short]:
        replay = _copy_with_task(
            row, "code", short_max=short_max, bridge_max=bridge_max
        )
        replay["hybrid_metadata"]["curriculum_stage"] = "short_replay"
        hierarchical.append(replay)
        control = _copy_with_task(
            row, "code", short_max=short_max, bridge_max=bridge_max
        )
        control["hybrid_metadata"]["curriculum_stage"] = "matched_direct_control"
        matched_control.append(control)
    hierarchical.sort(key=lambda row: _stable_rank(row, seed, "hierarchical"))
    matched_control.sort(key=lambda row: _stable_rank(row, seed, "matched_control"))
    if len(matched_control) != len(hierarchical):
        raise RuntimeError(
            f"matched-control cardinality drift: control={len(matched_control)} "
            f"hierarchical={len(hierarchical)}"
        )

    recovery = []
    for row in rows:
        count = instruction_count(row)
        stratum = "short" if count <= short_max else "bridge" if count <= bridge_max else "long"
        repeat = {"short": short_repeat, "bridge": bridge_repeat, "long": long_repeat}[stratum]
        for replica in range(repeat):
            copied = _copy_with_task(
                row, "code", short_max=short_max, bridge_max=bridge_max
            )
            copied["hybrid_metadata"]["instruction_stratum"] = stratum
            copied["hybrid_metadata"]["curriculum_stage"] = "code_recovery"
            copied["hybrid_metadata"]["curriculum_replica"] = replica
            recovery.append(copied)
    recovery.sort(key=lambda row: _stable_rank(row, seed, "recovery"))

    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "build_hierarchical_all_length_training",
        "input_rows": len(rows),
        "strata": {name: len(strata[name]) for name in ("short", "bridge", "long")},
        "direct_sft_rows": len(direct),
        "hierarchical_multitask_rows": len(hierarchical),
        "matched_direct_control_rows": len(matched_control),
        "code_recovery_rows": len(recovery),
        "hierarchical_task_counts": dict(Counter(row.get("training_task") for row in hierarchical)),
        "recovery_stratum_counts": dict(
            Counter((row.get("hybrid_metadata") or {}).get("instruction_stratum") for row in recovery)
        ),
        "region_statistics": {
            "planned_functions": len(plans),
            "total_regions": sum(plan["region_count"] for plan in plans.values()),
            "max_regions": max((plan["region_count"] for plan in plans.values()), default=0),
            "mean_regions": (
                sum(plan["region_count"] for plan in plans.values()) / len(plans)
                if plans else 0.0
            ),
        },
        "configuration": {
            "short_max": short_max,
            "bridge_max": bridge_max,
            "max_region_blocks": max_region_blocks,
            "short_replay_fraction": short_replay_fraction,
            "short_repeat": short_repeat,
            "bridge_repeat": bridge_repeat,
            "long_repeat": long_repeat,
        },
    }
    return direct, hierarchical, matched_control, recovery, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--direct_output", required=True)
    parser.add_argument("--hierarchical_output", required=True)
    parser.add_argument("--matched_control_output", required=True)
    parser.add_argument("--recovery_output", required=True)
    parser.add_argument("--short_subset_output", default="")
    parser.add_argument("--bridge_subset_output", default="")
    parser.add_argument("--long_subset_output", default="")
    parser.add_argument("--report", required=True)
    parser.add_argument("--short_max", type=int, default=150)
    parser.add_argument("--bridge_max", type=int, default=199)
    parser.add_argument("--max_region_blocks", type=int, default=8)
    parser.add_argument("--short_replay_fraction", type=float, default=0.33)
    parser.add_argument("--short_repeat", type=int, default=1)
    parser.add_argument("--bridge_repeat", type=int, default=2)
    parser.add_argument("--long_repeat", type=int, default=3)
    parser.add_argument("--min_long_rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.short_max <= 0 or args.bridge_max < args.short_max:
        parser.error("invalid short/bridge instruction thresholds")
    if args.max_region_blocks <= 0:
        parser.error("--max_region_blocks must be positive")
    if not 0.0 <= args.short_replay_fraction <= 1.0:
        parser.error("--short_replay_fraction must be in [0,1]")
    if min(args.short_repeat, args.bridge_repeat, args.long_repeat) <= 0:
        parser.error("curriculum repeat counts must be positive")

    rows = read_jsonl_many(args.input)
    direct, hierarchical, matched_control, recovery, report = build_datasets(
        rows,
        seed=args.seed,
        short_max=args.short_max,
        bridge_max=args.bridge_max,
        max_region_blocks=args.max_region_blocks,
        short_replay_fraction=args.short_replay_fraction,
        short_repeat=args.short_repeat,
        bridge_repeat=args.bridge_repeat,
        long_repeat=args.long_repeat,
    )
    if report["strata"]["long"] < args.min_long_rows:
        raise SystemExit(
            f"only {report['strata']['long']} long rows; minimum is {args.min_long_rows}"
        )
    write_jsonl(args.direct_output, direct)
    write_jsonl(args.hierarchical_output, hierarchical)
    write_jsonl(args.matched_control_output, matched_control)
    write_jsonl(args.recovery_output, recovery)
    # The short/bridge/long dev subsets are consumed by the functional gates,
    # which fail closed unless every row carries hybrid_metadata.evaluation_only.
    # These rows are held out from supervised training (dev split), so flagging
    # them evaluation-only is accurate. Deep-copy so the shared row objects that
    # also feed the Trainer eval views are not mutated.
    def _eval_only(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        marked: list[dict[str, Any]] = []
        for row in subset:
            row = copy.deepcopy(row)
            metadata = row.get("hybrid_metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata["evaluation_only"] = True
            row["hybrid_metadata"] = metadata
            marked.append(row)
        return marked
    if args.short_subset_output:
        write_jsonl(args.short_subset_output, _eval_only([row for row in rows if instruction_count(row) <= args.short_max]))
    if args.bridge_subset_output:
        write_jsonl(
            args.bridge_subset_output,
            _eval_only([row for row in rows if args.short_max < instruction_count(row) <= args.bridge_max]),
        )
    if args.long_subset_output:
        write_jsonl(args.long_subset_output, _eval_only([row for row in rows if instruction_count(row) > args.bridge_max]))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
