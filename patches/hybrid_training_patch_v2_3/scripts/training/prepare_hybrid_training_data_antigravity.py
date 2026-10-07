#!/usr/bin/env python3
"""Prepare an immutable, fail-closed all-length training/development pool.

Phase 0 runs before GPU work. It rejects frozen-evaluation overlap, deduplicates
by normalized source and semantic-pair/ISA key, neutralizes the executable contract to typed ``fn0``,
partitions feedback from hidden acceptance assertions, extracts deterministic
binary facts, and replays every reference through the same Dart evaluator used
by pass@k. Every approved length stratum participates in supervised training;
only a deterministic, length-stratified development slice is withheld.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.hybrid_data_controls import (  # noqa: E402
    SCHEMA_VERSION,
    attach_test_partition,
    candidate_assertion_count,
    facts_comment,
    file_record,
    instruction_count,
    length_bin,
    mechanical_facts,
    neutralize_training_row,
    normalized_source_hash,
    record_architecture,
    semantic_pair_identity,
    source_fingerprints,
    read_jsonl_many,
    sanitize_verifier_diagnostic,
    sha256_text,
    source_text,
    task_identity,
    write_jsonl,
)


def _load_evaluator():
    try:
        from scripts.evaluation.graph_compile_at_k_antigravity import (  # type: ignore
            evaluate_dart_jit_tests_detail,
        )
    except Exception as exc:  # pragma: no cover - project integration guard
        raise RuntimeError(
            "Could not import the pass-aligned Dart evaluator. Overlay this patch "
            "inside the project repository before running Phase 0."
        ) from exc
    return evaluate_dart_jit_tests_detail


def _evaluate_reference(row: dict[str, Any], timeout: int) -> dict[str, Any]:
    evaluator = _load_evaluator()
    source = source_text(row)
    task_id = task_identity(row)
    harnesses = {
        "full": str(row.get("tests") or ""),
        "feedback": str(row.get("feedback_tests") or ""),
        "acceptance": str(row.get("acceptance_tests") or ""),
    }
    results: dict[str, Any] = {}
    for split, tests in harnesses.items():
        if split == "feedback" and not tests:
            results[split] = {
                "compiled": None,
                "passed": None,
                "diagnostic": "",
                "skipped_no_visible_assertions": True,
            }
            continue
        if not tests:
            results[split] = {
                "compiled": False,
                "passed": False,
                "diagnostic": "missing test harness",
            }
            continue
        compiled, passed, diagnostic, _full_source = evaluator(
            source,
            tests,
            f"phase0_{task_id}_{split}",
            timeout=timeout,
        )
        results[split] = {
            "compiled": bool(compiled),
            "passed": bool(passed),
            "diagnostic": sanitize_verifier_diagnostic(str(diagnostic or "")),
        }
    required = [results["full"].get("passed"), results["acceptance"].get("passed")]
    if results["feedback"].get("passed") is not None:
        required.append(results["feedback"].get("passed"))
    return {"passed": all(bool(value) for value in required), "splits": results}


def _prepare_one(
    raw: dict[str, Any],
    *,
    seed: int,
    feedback_fraction: float,
    neutral_name: str,
) -> dict[str, Any]:
    original_hashes = source_fingerprints(raw)
    row = neutralize_training_row(raw, neutral_name=neutral_name)
    row = attach_test_partition(row, feedback_fraction=feedback_fraction, seed=seed)
    facts = mechanical_facts(row)
    row["binary_facts"] = facts
    row["facts_target_comment"] = facts_comment(facts)
    count = instruction_count(row)
    metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "phase0_approved": False,
            "evaluation_only": False,
            # Keep the historical field for downstream compatibility while
            # recording the stronger dual-fingerprint control explicitly.
            "source_overlap_hash": original_hashes["neutral_sha256"],
            "source_overlap_hashes": original_hashes,
            "prepared_source_sha256": sha256_text(source_text(row)),
            "instruction_count": count,
            "length_bin": length_bin(count),
            "facts_first_target": True,
            "hidden_acceptance_required": True,
            "data_role": "unassigned",
        }
    )
    row["hybrid_metadata"] = metadata
    return row


def _rank(row: dict[str, Any], seed: int) -> str:
    material = (
        f"{seed}|{(row.get('hybrid_metadata') or {}).get('length_bin')}|"
        f"{task_identity(row)}|{normalized_source_hash(row)}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _set_role(row: dict[str, Any], role: str) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    metadata = copy.deepcopy(copied.get("hybrid_metadata") or {})
    metadata["data_role"] = role
    copied["hybrid_metadata"] = metadata
    return copied


def _deduplicate_nonoverlap(
    raw_rows: list[dict[str, Any]],
    forbidden_hashes: dict[str, dict[str, list[str]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Filter frozen overlap and deduplicate with cross-ISA pair awareness.

    Legacy rows without ``semantic_pair_id`` retain global source-fingerprint
    deduplication.  Paired rows may share a source only when their pair ID is
    identical and their canonical architectures differ.  A pair/ISA key can
    occur once; divergent rows under the same key are reported as conflicts.
    """
    unique_raw: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    pair_conflicts: list[dict[str, Any]] = []
    seen: dict[str, dict[str, list[dict[str, str]]]] = {
        "neutral_sha256": defaultdict(list),
        "alpha_structural_sha256": defaultdict(list),
    }
    pair_arch_seen: dict[tuple[str, str], dict[str, Any]] = {}
    pair_reference: dict[str, dict[str, Any]] = {}

    for index, row in enumerate(raw_rows):
        identity = task_identity(row, index)
        fingerprints = source_fingerprints(row)
        matched_eval = {
            kind: forbidden_hashes[kind][digest]
            for kind, digest in fingerprints.items()
            if digest in forbidden_hashes[kind]
        }
        if matched_eval:
            overlaps.append({
                "task": identity,
                "source_fingerprints": fingerprints,
                "matched_fingerprint_kinds": sorted(matched_eval),
                "frozen_eval_tasks": sorted(
                    {task for tasks in matched_eval.values() for task in tasks}
                ),
            })
            continue

        pair_id = semantic_pair_identity(row)
        architecture = record_architecture(row)
        if pair_id and architecture == "unknown":
            pair_conflicts.append({
                "task": identity,
                "semantic_pair_id": pair_id,
                "architecture": architecture,
                "reason": "paired row has no usable architecture provenance",
                "source_fingerprints": fingerprints,
            })
            continue

        pair_arch_key = (pair_id, architecture)
        previous_same_isa = pair_arch_seen.get(pair_arch_key) if pair_id else None
        if previous_same_isa is not None:
            if fingerprints == previous_same_isa["source_fingerprints"]:
                duplicates.append({
                    "task": identity,
                    "duplicate_of": [previous_same_isa["task"]],
                    "semantic_pair_id": pair_id,
                    "architecture": architecture,
                    "reason": "duplicate semantic_pair_id/architecture",
                    "source_fingerprints": fingerprints,
                    "matched_fingerprint_kinds": sorted(fingerprints),
                })
            else:
                pair_conflicts.append({
                    "task": identity,
                    "conflicts_with": previous_same_isa["task"],
                    "semantic_pair_id": pair_id,
                    "architecture": architecture,
                    "reason": "conflicting rows share semantic_pair_id/architecture",
                    "source_fingerprints": fingerprints,
                    "conflicting_source_fingerprints": previous_same_isa["source_fingerprints"],
                })
            continue

        previous_pair = pair_reference.get(pair_id) if pair_id else None
        if previous_pair is not None and fingerprints != previous_pair["source_fingerprints"]:
            pair_conflicts.append({
                "task": identity,
                "conflicts_with": previous_pair["task"],
                "semantic_pair_id": pair_id,
                "architecture": architecture,
                "conflicting_architecture": previous_pair["architecture"],
                "reason": "cross-architecture semantic pair has divergent source",
                "source_fingerprints": fingerprints,
                "conflicting_source_fingerprints": previous_pair["source_fingerprints"],
            })
            continue

        matched_occurrences: dict[str, dict[str, str]] = {}
        matched_kinds: set[str] = set()
        for kind, digest in fingerprints.items():
            for occurrence in seen[kind].get(digest, []):
                matched_occurrences[occurrence["occurrence_id"]] = occurrence
                matched_kinds.add(kind)
        allowed_cross_isa_pair = bool(matched_occurrences) and bool(pair_id) and all(
            occurrence["semantic_pair_id"] == pair_id
            and occurrence["architecture"] != architecture
            for occurrence in matched_occurrences.values()
        )
        if matched_occurrences and not allowed_cross_isa_pair:
            duplicates.append({
                "task": identity,
                "duplicate_of": sorted(
                    {occurrence["task"] for occurrence in matched_occurrences.values()}
                ),
                "semantic_pair_id": pair_id or None,
                "architecture": architecture,
                "source_fingerprints": fingerprints,
                "matched_fingerprint_kinds": sorted(matched_kinds),
            })
            continue

        occurrence = {
            "occurrence_id": str(index),
            "task": identity,
            "semantic_pair_id": pair_id,
            "architecture": architecture,
        }
        for kind, digest in fingerprints.items():
            seen[kind][digest].append(occurrence)
        if pair_id:
            pair_arch_seen[pair_arch_key] = {
                **occurrence,
                "source_fingerprints": fingerprints,
            }
            pair_reference.setdefault(
                pair_id,
                {
                    **occurrence,
                    "source_fingerprints": fingerprints,
                },
            )
        unique_raw.append(row)

    return unique_raw, duplicates, overlaps, pair_conflicts


def _split_train_dev_legacy(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    dev_fraction: float,
    min_train: int,
    min_dev: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split every represented length bin proportionally.

    The previous implementation guaranteed one row per bin and then filled the
    remainder globally. With a 46% long-function population this could still
    under-sample the >=200 development stratum by chance. Hamilton-style
    apportionment keeps the development distribution close to the approved pool
    while retaining at least one training row in every non-singleton bin.
    """
    if min_dev <= 0 and dev_fraction <= 0:
        return [_set_role(row, "train") for row in rows], []
    target_dev = max(min_dev, int(round(len(rows) * dev_fraction)))
    target_dev = min(target_dev, max(0, len(rows) - min_train))
    if target_dev < min_dev:
        raise ValueError(
            f"cannot allocate {min_dev} development rows while retaining {min_train} train rows "
            f"from {len(rows)} approved all-length rows"
        )

    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = str((row.get("hybrid_metadata") or {}).get("length_bin") or "unknown")
        by_bin[name].append(row)
    for bucket in by_bin.values():
        bucket.sort(key=lambda row: _rank(row, seed))

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    total_rows = max(1, len(rows))
    for name, bucket in by_bin.items():
        exact = target_dev * len(bucket) / total_rows
        maximum = max(0, len(bucket) - 1) if len(bucket) > 1 else 0
        base = min(maximum, int(exact))
        if target_dev > 0 and len(bucket) >= 2:
            base = max(1, base)
        quotas[name] = base
        remainders.append((exact - int(exact), name))

    allocated = sum(quotas.values())
    if allocated > target_dev:
        for _fraction, name in sorted(remainders, key=lambda item: (item[0], item[1])):
            minimum = 1 if len(by_bin[name]) >= 2 and target_dev >= len(by_bin) else 0
            while allocated > target_dev and quotas[name] > minimum:
                quotas[name] -= 1
                allocated -= 1
    elif allocated < target_dev:
        for _fraction, name in sorted(remainders, key=lambda item: (-item[0], item[1])):
            maximum = max(0, len(by_bin[name]) - 1) if len(by_bin[name]) > 1 else 0
            while allocated < target_dev and quotas[name] < maximum:
                quotas[name] += 1
                allocated += 1
        if allocated < target_dev:
            # A second deterministic pass handles saturation in high-remainder bins.
            for name in sorted(by_bin):
                maximum = max(0, len(by_bin[name]) - 1) if len(by_bin[name]) > 1 else 0
                while allocated < target_dev and quotas[name] < maximum:
                    quotas[name] += 1
                    allocated += 1

    chosen_ids: set[int] = set()
    for name, bucket in by_bin.items():
        chosen_ids.update(id(row) for row in bucket[: quotas[name]])
    dev = [_set_role(row, "development") for row in rows if id(row) in chosen_ids]
    train = [_set_role(row, "train") for row in rows if id(row) not in chosen_ids]
    train.sort(key=lambda row: _rank(row, seed))
    dev.sort(key=lambda row: _rank(row, seed))
    return train, dev


def _semantic_split_rank(
    key: tuple[str, str],
    unit_rows: list[dict[str, Any]],
    seed: int,
) -> str:
    material = "|".join(
        [
            str(seed),
            key[0],
            key[1],
            *sorted(
                f"{task_identity(row)}:{record_architecture(row)}:{normalized_source_hash(row)}"
                for row in unit_rows
            ),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _split_train_dev_grouped(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    dev_fraction: float,
    min_train: int,
    min_dev: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split atomic semantic units while approximately preserving length bins."""
    if min_dev <= 0 and dev_fraction <= 0:
        return [_set_role(row, "train") for row in rows], []
    target_dev = max(min_dev, int(round(len(rows) * dev_fraction)))
    capacity = max(0, len(rows) - min_train)
    required_dev = max(min_dev, 1 if target_dev > 0 else 0)
    if capacity < required_dev:
        raise ValueError(
            f"cannot allocate {required_dev} development rows while retaining {min_train} train rows "
            f"from {len(rows)} approved all-length rows"
        )
    target_dev = min(target_dev, capacity)

    units: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair_id = semantic_pair_identity(row)
        key = ("pair", pair_id) if pair_id else ("row", f"{index}:{task_identity(row, index)}")
        units[key].append(row)

    desired_bins = Counter(
        str((row.get("hybrid_metadata") or {}).get("length_bin") or "unknown")
        for row in rows
    )
    desired_bins = Counter({
        name: target_dev * count / max(1, len(rows))
        for name, count in desired_bins.items()
    })
    unit_records: list[dict[str, Any]] = []
    for key, unit_rows in units.items():
        unit_records.append({
            "key": key,
            "rows": unit_rows,
            "size": len(unit_rows),
            "bins": Counter(
                str((row.get("hybrid_metadata") or {}).get("length_bin") or "unknown")
                for row in unit_rows
            ),
            "rank": _semantic_split_rank(key, unit_rows, seed),
        })

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    selected_bins: Counter[str] = Counter()
    selected_rows = 0
    remaining = list(unit_records)
    while remaining and selected_rows < target_dev:
        candidates = [
            unit for unit in remaining
            if selected_rows + int(unit["size"]) <= capacity
        ]
        if not candidates:
            break

        def candidate_order(unit: dict[str, Any]) -> tuple[float, int, int, str]:
            before = sum(
                abs(float(desired_bins[name]) - selected_bins[name])
                for name in desired_bins
            )
            after = sum(
                abs(
                    float(desired_bins[name])
                    - selected_bins[name]
                    - int(unit["bins"].get(name, 0))
                )
                for name in desired_bins
            )
            new_total = selected_rows + int(unit["size"])
            return (
                -(before - after),
                abs(target_dev - new_total),
                max(0, new_total - target_dev),
                str(unit["rank"]),
            )

        chosen = min(candidates, key=candidate_order)
        selected.append(chosen)
        selected_keys.add(chosen["key"])
        selected_bins.update(chosen["bins"])
        selected_rows += int(chosen["size"])
        remaining.remove(chosen)

    if selected_rows < required_dev:
        for unit in sorted(remaining, key=lambda value: str(value["rank"])):
            if selected_rows + int(unit["size"]) > capacity:
                continue
            selected.append(unit)
            selected_keys.add(unit["key"])
            selected_bins.update(unit["bins"])
            selected_rows += int(unit["size"])
            if selected_rows >= required_dev:
                break
    if selected_rows < required_dev:
        raise ValueError(
            f"semantic grouping permits only {selected_rows} development rows; minimum is {required_dev}"
        )

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        pair_id = semantic_pair_identity(row)
        key = ("pair", pair_id) if pair_id else ("row", f"{index}:{task_identity(row, index)}")
        destination = dev if key in selected_keys else train
        destination.append(_set_role(row, "development" if key in selected_keys else "train"))
    train.sort(key=lambda row: _rank(row, seed))
    dev.sort(key=lambda row: _rank(row, seed))
    return train, dev


def _split_train_dev(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    dev_fraction: float,
    min_train: int,
    min_dev: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split train/dev, keeping every cross-ISA semantic pair atomic.

    The legacy implementation remains byte-for-byte authoritative when no row
    carries a semantic pair ID.  In mixed datasets, unpaired rows are singleton
    units while all rows sharing a pair ID move together.
    """
    if not any(semantic_pair_identity(row) for row in rows):
        return _split_train_dev_legacy(
            rows,
            seed=seed,
            dev_fraction=dev_fraction,
            min_train=min_train,
            min_dev=min_dev,
        )
    return _split_train_dev_grouped(
        rows,
        seed=seed,
        dev_fraction=dev_fraction,
        min_train=min_train,
        min_dev=min_dev,
    )


def _architecture_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record_architecture(row) for row in rows).items()))


def _semantic_pair_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unpaired = 0
    for row in rows:
        pair_id = semantic_pair_identity(row)
        if pair_id:
            pairs[pair_id].append(row)
        else:
            unpaired += 1
    architecture_sets = Counter(
        "+".join(sorted({record_architecture(row) for row in pair_rows}))
        for pair_rows in pairs.values()
    )
    return {
        "rows_with_pair_id": sum(len(pair_rows) for pair_rows in pairs.values()),
        "rows_without_pair_id": unpaired,
        "unique_pair_ids": len(pairs),
        "multi_isa_pair_ids": sum(
            len({record_architecture(row) for row in pair_rows}) > 1
            for pair_rows in pairs.values()
        ),
        "pair_ids_by_isa_set": dict(sorted(architecture_sets.items())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", "--train_file", dest="input", required=True,
                        help="Comma-separated raw training JSONL files")
    parser.add_argument(
        "--forbidden_eval", "--frozen_eval", dest="forbidden_eval",
        action="append", default=[],
        help="Frozen evaluation JSONL file(s); repeat or use comma-separated specs",
    )
    parser.add_argument("--output", "--out_approved", dest="output", required=True,
                        help="All-length approved training rows after the stratified dev split")
    parser.add_argument("--dev_output", "--out_dev", dest="dev_output", default="")
    parser.add_argument("--short_output", default="",
                        help="Optional approved training subset at or below --max_instructions")
    parser.add_argument("--bridge_output", default="",
                        help="Approved training subset above --max_instructions through --max_bridge_instructions")
    parser.add_argument("--long_output", "--out_long", dest="long_output", required=True,
                        help="Approved training subset above --max_bridge_instructions; these rows are trained")
    parser.add_argument("--rejected_output", "--out_rejected", dest="rejected_output", default="")
    parser.add_argument("--report", required=True)
    parser.add_argument("--neutral_name", default="fn0")
    parser.add_argument(
        "--data_role", choices=["train", "development", "dev"], default="train",
        help="Role assigned when no explicit --dev_output split is requested.",
    )
    parser.add_argument("--feedback_fraction", type=float, default=0.70)
    parser.add_argument("--dev_fraction", type=float, default=0.10)
    parser.add_argument("--max_instructions", "--short_max_instructions", dest="max_instructions", type=int, default=120)
    parser.add_argument("--max_bridge_instructions", type=int, default=199,
                        help="Upper bound of the bridge training stratum; rows above it are long")
    parser.add_argument("--min_short_rows", "--min_approved_rows", dest="min_short_rows", type=int, default=1)
    parser.add_argument("--min_long_rows", type=int, default=1,
                        help="Fail if fewer approved >=200-instruction training rows remain")
    parser.add_argument("--min_dev_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 1)))
    parser.add_argument(
        "--drop_invalid_references",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Development-only: drop failed references instead of aborting Phase 0",
    )
    parser.add_argument(
        "--skip_reference_execution",
        action="store_true",
        help="Development-only; no row is approved and the minimum-row gate fails",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.max_instructions <= 0 or args.workers <= 0:
        parser.error("instruction and worker counts must be positive")
    if args.max_bridge_instructions < args.max_instructions:
        parser.error("--max_bridge_instructions must be >= --max_instructions")
    data_role = "development" if args.data_role == "dev" else args.data_role
    if args.dev_output and data_role == "development":
        parser.error("--dev_output cannot be combined with --data_role development")
    if not 0.0 < args.feedback_fraction < 1.0:
        parser.error("--feedback_fraction must be strictly between 0 and 1")
    if not 0.0 <= args.dev_fraction < 1.0:
        parser.error("--dev_fraction must be in [0, 1)")
    if args.min_short_rows < 0 or args.min_long_rows < 0 or args.min_dev_rows < 0:
        parser.error("minimum row counts must be non-negative")

    raw_rows = read_jsonl_many(args.input)
    if not raw_rows:
        raise SystemExit("training input is empty")
    forbidden_specs: list[str] = []
    for value in args.forbidden_eval:
        forbidden_specs.extend(part.strip() for part in value.split(",") if part.strip())
    if not forbidden_specs:
        raise SystemExit(
            "Phase 0 requires at least one frozen evaluation file; an operator-supplied data_role "
            "string is not a leakage control"
        )
    forbidden_rows = read_jsonl_many(forbidden_specs)
    forbidden_hashes: dict[str, dict[str, list[str]]] = {
        "neutral_sha256": defaultdict(list),
        "alpha_structural_sha256": defaultdict(list),
    }
    for index, row in enumerate(forbidden_rows):
        identity = task_identity(row, index)
        for kind, digest in source_fingerprints(row).items():
            forbidden_hashes[kind][digest].append(identity)

    unique_raw, duplicates, overlaps, pair_conflicts = _deduplicate_nonoverlap(
        raw_rows,
        forbidden_hashes,
    )

    preparation_failures: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(unique_raw):
        try:
            prepared.append(_prepare_one(
                row,
                seed=args.seed,
                feedback_fraction=args.feedback_fraction,
                neutral_name=args.neutral_name,
            ))
        except Exception as exc:
            preparation_failures.append({
                "task": task_identity(row, index),
                "error": f"{type(exc).__name__}: {exc}",
            })

    reference_results: list[dict[str, Any]] = []
    if not args.skip_reference_execution:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_evaluate_reference, row, args.timeout) for row in prepared]
            for index, (row, future) in enumerate(zip(prepared, futures), 1):
                result = future.result()
                reference_results.append(result)
                metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
                metadata["reference_test_replay"] = result
                metadata["phase0_approved"] = bool(result["passed"])
                row["hybrid_metadata"] = metadata
                if index % 50 == 0 or index == len(prepared):
                    print(f"reference replay {index}/{len(prepared)}")
    else:
        reference_results = [{"passed": False, "skipped": True} for _ in prepared]

    invalid_references: list[dict[str, Any]] = []
    approved: list[dict[str, Any]] = []
    for row, result in zip(prepared, reference_results):
        if result.get("passed"):
            approved.append(row)
        else:
            invalid_references.append({"task": task_identity(row), "result": result})

    # Split the complete approved population, stratified by fine-grained length
    # bins. The instruction thresholds define training curricula, not exclusion.
    try:
        train_rows, dev_rows = _split_train_dev(
            approved,
            seed=args.seed,
            dev_fraction=args.dev_fraction if args.dev_output else 0.0,
            min_train=args.min_short_rows + args.min_long_rows,
            min_dev=args.min_dev_rows if args.dev_output else 0,
        )
        if not args.dev_output and data_role != "train":
            train_rows = [_set_role(row, data_role) for row in train_rows]
    except Exception as exc:
        train_rows, dev_rows = [], []
        preparation_failures.append({"task": "<train-dev-split>", "error": f"{type(exc).__name__}: {exc}"})

    short_rows = [
        row for row in train_rows
        if instruction_count(row) <= args.max_instructions
    ]
    bridge_rows = [
        row for row in train_rows
        if args.max_instructions < instruction_count(row) <= args.max_bridge_instructions
    ]
    long_rows = [
        row for row in train_rows
        if instruction_count(row) > args.max_bridge_instructions
    ]
    for row in train_rows:
        count = instruction_count(row)
        metadata = copy.deepcopy(row.get("hybrid_metadata") or {})
        metadata["instruction_stratum"] = (
            "short" if count <= args.max_instructions
            else "bridge" if count <= args.max_bridge_instructions
            else "long"
        )
        metadata["trained_in_all_length_curriculum"] = True
        row["hybrid_metadata"] = metadata

    rejected_rows: list[dict[str, Any]] = []
    for kind, values in (
        ("duplicate", duplicates),
        ("semantic_pair_conflict", pair_conflicts),
        ("frozen_eval_overlap", overlaps),
        ("preparation_failure", preparation_failures),
        ("reference_failure", invalid_references),
    ):
        rejected_rows.extend({"rejection_type": kind, **value} for value in values)

    write_jsonl(args.output, train_rows)
    if args.dev_output:
        write_jsonl(args.dev_output, dev_rows)
    if args.short_output:
        write_jsonl(args.short_output, short_rows)
    if args.bridge_output:
        write_jsonl(args.bridge_output, bridge_rows)
    write_jsonl(args.long_output, long_rows)
    if args.rejected_output:
        write_jsonl(args.rejected_output, rejected_rows)

    outputs = {
        "all_length_train": file_record(args.output),
        "development": file_record(args.dev_output) if args.dev_output else None,
        "short": file_record(args.short_output) if args.short_output else None,
        "bridge": file_record(args.bridge_output) if args.bridge_output else None,
        "long": file_record(args.long_output),
        "rejected": file_record(args.rejected_output) if args.rejected_output else None,
    }
    fatal: list[str] = []
    if overlaps:
        fatal.append(
            f"{len(overlaps)} exact-neutral or alpha-structural overlaps with frozen evaluation data"
        )
    if pair_conflicts:
        fatal.append(f"{len(pair_conflicts)} semantic-pair architecture conflicts")
    if preparation_failures:
        fatal.append(f"{len(preparation_failures)} rows/splits failed preparation")
    if invalid_references and not args.drop_invalid_references:
        fatal.append(f"{len(invalid_references)} references failed full/feedback/acceptance replay")
    if len(short_rows) < args.min_short_rows:
        fatal.append(f"only {len(short_rows)} approved short train rows; minimum is {args.min_short_rows}")
    if len(long_rows) < args.min_long_rows:
        fatal.append(f"only {len(long_rows)} approved long train rows; minimum is {args.min_long_rows}")
    if args.dev_output and len(dev_rows) < args.min_dev_rows:
        fatal.append(f"only {len(dev_rows)} development rows; minimum is {args.min_dev_rows}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "phase0_prepare",
        "status": "failed" if fatal else "passed",
        "fatal_issues": fatal,
        "input_rows": len(raw_rows),
        "unique_nonoverlap_rows": len(unique_raw),
        "prepared_rows": len(prepared),
        "approved_rows": len(approved),
        "approved_all_length_total": len(approved),
        "all_length_train_rows": len(train_rows),
        "short_rows": len(short_rows),
        "development_rows": len(dev_rows),
        "bridge_rows": len(bridge_rows),
        "long_rows": len(long_rows),
        "long_rows_are_trained": True,
        "duplicates": duplicates,
        "duplicate_rows": len(duplicates),
        "semantic_pair_conflicts": pair_conflicts,
        "semantic_pair_conflict_rows": len(pair_conflicts),
        "frozen_eval_overlaps": overlaps,
        "frozen_eval_overlap_rows": len(overlaps),
        "preparation_failure_rows": len(preparation_failures),
        "invalid_reference_rows": len(invalid_references),
        "overlap_controls": {
            "fingerprints": ["neutral_sha256", "alpha_structural_sha256"],
            "alpha_normalizes_local_and_parameter_identifiers": True,
        },
        "preparation_failures": preparation_failures,
        "invalid_references": invalid_references,
        "length_bins": dict(Counter((row.get("hybrid_metadata") or {}).get("length_bin") for row in approved)),
        "architecture_counts": {
            "input": _architecture_counts(raw_rows),
            "unique_nonoverlap": _architecture_counts(unique_raw),
            "approved": _architecture_counts(approved),
            "train": _architecture_counts(train_rows),
            "development": _architecture_counts(dev_rows),
        },
        "semantic_pair_counts": {
            "input": _semantic_pair_counts(raw_rows),
            "unique_nonoverlap": _semantic_pair_counts(unique_raw),
            "approved": _semantic_pair_counts(approved),
            "train": _semantic_pair_counts(train_rows),
            "development": _semantic_pair_counts(dev_rows),
        },
        "test_assertions": {
            "full": sum(candidate_assertion_count(str(row.get("tests") or "")) for row in approved),
            "feedback": sum(candidate_assertion_count(str(row.get("feedback_tests") or "")) for row in approved),
            "acceptance": sum(candidate_assertion_count(str(row.get("acceptance_tests") or "")) for row in approved),
        },
        "arguments": vars(args),
        "inputs": {
            "train": [file_record(path.strip()) for path in str(args.input).split(",") if Path(path.strip()).is_file()],
            "forbidden_eval": [file_record(path) for path in forbidden_specs],
        },
        "outputs": outputs,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "input_rows": len(raw_rows),
        "all_length_train_rows": len(train_rows),
        "short_rows": len(short_rows),
        "development_rows": len(dev_rows),
        "bridge_rows": len(bridge_rows),
        "long_rows": len(long_rows),
        "long_rows_are_trained": True,
    }, indent=2))

    if fatal:
        raise SystemExit("Phase-0 gate failed: " + "; ".join(fatal))


if __name__ == "__main__":
    main()
