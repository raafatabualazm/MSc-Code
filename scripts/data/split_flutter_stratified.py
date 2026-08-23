#!/usr/bin/env python3
"""Create a leakage-safe, category/length-stratified Flutter ARM64 split.

Rows are grouped by normalized source and by high-similarity source components
before assignment, so exact or near-clone programs cannot straddle train/eval.
The deterministic split preserves both task category and assembly-length bands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.data.audit_dataset_overlap_antigravity import (
        normalize_source,
        shingles,
        source_of,
    )
except ModuleNotFoundError:
    from audit_dataset_overlap_antigravity import (  # type: ignore
        normalize_source,
        shingles,
        source_of,
    )


BINS = [
    ("<50", 0, 50),
    ("50-100", 50, 100),
    ("100-200", 100, 200),
    ("200-500", 200, 500),
    (">500", 500, 10**9),
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instr_count(row: dict[str, Any]) -> int:
    cfg = row.get("cfg")
    if cfg:
        return sum(
            int(block.get("instruction_count") or len(block.get("instructions") or []))
            for block in cfg
        )
    assembly = row.get("assembly") or ""
    return sum(
        1
        for line in assembly.splitlines()
        if re.match(r"\s*(?:0x)?[0-9a-fA-F]+:\s", line)
    )


def bin_of(count: int) -> str:
    for name, lower, upper in BINS:
        if lower <= count < upper:
            return name
    return ">500"


def normalized_source_key(row: dict[str, Any]) -> str:
    normalized = normalize_source(source_of(row))
    if not normalized:
        raise ValueError("row has no Dart source")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def source_components(
    rows: list[dict[str, Any]], threshold: float
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    sources = [source_of(row) for row in rows]
    source_keys = [normalized_source_key(row) for row in rows]
    source_groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(source_keys):
        source_groups[key].append(index)

    dsu = DisjointSet(len(rows))
    for indices in source_groups.values():
        for index in indices[1:]:
            dsu.union(indices[0], index)

    row_shingles = [shingles(source) for source in sources]
    inverted: dict[str, list[int]] = defaultdict(list)
    near_pairs: list[dict[str, Any]] = []
    for right, values in enumerate(row_shingles):
        intersections: dict[int, int] = defaultdict(int)
        for value in values:
            for left in inverted.get(value, []):
                intersections[left] += 1
        for left, intersection in intersections.items():
            union = len(values) + len(row_shingles[left]) - intersection
            score = intersection / union if union else 0.0
            if score >= threshold:
                dsu.union(left, right)
                near_pairs.append(
                    {
                        "left_index": left,
                        "left_task_id": rows[left].get("task_id", left),
                        "right_index": right,
                        "right_task_id": rows[right].get("task_id", right),
                        "jaccard_7gram": score,
                    }
                )
        for value in values:
            inverted[value].append(right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[dsu.find(index)].append(index)
    components = sorted(grouped.values(), key=lambda indices: min(indices))
    near_pairs.sort(key=lambda item: item["jaccard_7gram"], reverse=True)
    return components, near_pairs


def component_stratum(rows: list[dict[str, Any]], indices: list[int]) -> tuple[str, str]:
    categories = Counter(str(rows[index].get("category") or "unknown") for index in indices)
    category = sorted(categories.items(), key=lambda item: (-item[1], item[0]))[0][0]
    count = round(median(instr_count(rows[index]) for index in indices))
    return category, bin_of(count)


def allocate_components(
    rows: list[dict[str, Any]],
    components: list[list[int]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[set[int], set[int], dict[tuple[str, str], dict[str, int]]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    component_strata: dict[int, tuple[str, str]] = {}
    for component_id, indices in enumerate(components):
        stratum = component_stratum(rows, indices)
        component_strata[component_id] = stratum
        strata[stratum].append(component_id)

    eval_components: set[int] = set()
    stratum_stats: dict[tuple[str, str], dict[str, int]] = {}
    for stratum in sorted(strata):
        component_ids = sorted(strata[stratum])
        rng.shuffle(component_ids)
        total_rows = sum(len(components[component_id]) for component_id in component_ids)
        target_rows = round(total_rows * eval_fraction)
        selected_rows = 0
        for component_id in component_ids:
            size = len(components[component_id])
            current_distance = abs(selected_rows - target_rows)
            candidate_distance = abs(selected_rows + size - target_rows)
            if selected_rows < target_rows and candidate_distance <= current_distance:
                eval_components.add(component_id)
                selected_rows += size
        stratum_stats[stratum] = {
            "total": total_rows,
            "target_eval": target_rows,
            "eval": selected_rows,
        }

    target_total = round(len(rows) * eval_fraction)
    current_total = sum(len(components[item]) for item in eval_components)

    # Singleton components make exact global allocation possible without breaking
    # any exact/near-source component. Prioritize the most under/over-filled strata.
    if current_total < target_total:
        candidates = [
            item
            for item, indices in enumerate(components)
            if item not in eval_components and len(indices) == 1
        ]
        candidates.sort(
            key=lambda item: (
                stratum_stats[component_strata[item]]["eval"]
                - stratum_stats[component_strata[item]]["target_eval"],
                rng.random(),
            )
        )
        for component_id in candidates[: target_total - current_total]:
            eval_components.add(component_id)
            stratum_stats[component_strata[component_id]]["eval"] += 1
    elif current_total > target_total:
        candidates = [item for item in eval_components if len(components[item]) == 1]
        candidates.sort(
            key=lambda item: (
                stratum_stats[component_strata[item]]["target_eval"]
                - stratum_stats[component_strata[item]]["eval"],
                rng.random(),
            )
        )
        for component_id in candidates[: current_total - target_total]:
            eval_components.remove(component_id)
            stratum_stats[component_strata[component_id]]["eval"] -= 1

    eval_indices = {
        index
        for component_id in eval_components
        for index in components[component_id]
    }
    train_indices = set(range(len(rows))) - eval_indices
    if len(eval_indices) != target_total:
        raise RuntimeError(
            f"could not allocate exact eval target {target_total}; got {len(eval_indices)}"
        )
    return train_indices, eval_indices, stratum_stats


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        band = bin_of(instr_count(row))
        output.setdefault(category, {name: 0 for name, _, _ in BINS})[band] += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--eval-out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=float, default=0.8)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    if not 0.0 < args.eval_frac < 1.0:
        raise SystemExit("--eval-frac must be between 0 and 1")

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")

    components, near_pairs = source_components(rows, args.near_threshold)
    train_indices, eval_indices, stratum_stats = allocate_components(
        rows,
        components,
        eval_fraction=args.eval_frac,
        seed=args.seed,
    )
    train_rows = [row for index, row in enumerate(rows) if index in train_indices]
    eval_rows = [row for index, row in enumerate(rows) if index in eval_indices]

    component_side: dict[int, str] = {}
    for component_id, indices in enumerate(components):
        sides = {"eval" if index in eval_indices else "train" for index in indices}
        if len(sides) != 1:
            raise RuntimeError(f"component {component_id} crosses the split")
        component_side[component_id] = sides.pop()

    write_jsonl_atomic(args.train_out, train_rows)
    write_jsonl_atomic(args.eval_out, eval_rows)
    manifest = {
        "schema": "antigravity-arm64-split-v2",
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "input_rows": len(rows),
        "train": str(args.train_out),
        "train_sha256": file_sha256(args.train_out),
        "train_rows": len(train_rows),
        "eval": str(args.eval_out),
        "eval_sha256": file_sha256(args.eval_out),
        "eval_rows": len(eval_rows),
        "seed": args.seed,
        "eval_fraction": args.eval_frac,
        "near_threshold": args.near_threshold,
        "source_components": len(components),
        "multirow_components": sum(len(indices) > 1 for indices in components),
        "near_pairs": near_pairs,
        "cross_split_component_count": 0,
        "train_distribution": distribution(train_rows),
        "eval_distribution": distribution(eval_rows),
        "strata": {
            f"{category} | {band}": stats
            for (category, band), stats in sorted(stratum_stats.items())
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "source_components": len(components),
                "multirow_components": manifest["multirow_components"],
                "near_pairs": len(near_pairs),
                "cross_split_component_count": 0,
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
