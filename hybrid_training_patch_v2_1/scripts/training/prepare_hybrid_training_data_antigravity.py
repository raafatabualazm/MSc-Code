#!/usr/bin/env python3
"""Prepare an immutable, fail-closed short-function training/development pool.

Phase 0 runs before GPU work. It rejects frozen-evaluation overlap, deduplicates
by normalized source, neutralizes the executable contract to typed ``fn0``,
partitions feedback from hidden acceptance assertions, extracts deterministic
binary facts, and replays every reference through the same Dart evaluator used
by pass@k. Approved short rows are deterministically split into train/dev; longer
rows are quarantined for a separate hierarchical-decoding research track.
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


def _split_train_dev(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    dev_fraction: float,
    min_train: int,
    min_dev: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if min_dev <= 0 and dev_fraction <= 0:
        return [_set_role(row, "train") for row in rows], []
    target_dev = max(min_dev, int(round(len(rows) * dev_fraction)))
    target_dev = min(target_dev, max(0, len(rows) - min_train))
    if target_dev < min_dev:
        raise ValueError(
            f"cannot allocate {min_dev} development rows while retaining {min_train} train rows "
            f"from {len(rows)} approved short rows"
        )

    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bin[str((row.get("hybrid_metadata") or {}).get("length_bin") or "unknown")].append(row)
    for bucket in by_bin.values():
        bucket.sort(key=lambda row: _rank(row, seed))

    chosen_ids: set[int] = set()
    # Give represented bins one deterministic dev row where possible.
    for bin_name in sorted(by_bin):
        if len(chosen_ids) >= target_dev:
            break
        bucket = by_bin[bin_name]
        if len(bucket) >= 2:
            chosen_ids.add(id(bucket[0]))

    remaining = sorted(
        (row for row in rows if id(row) not in chosen_ids),
        key=lambda row: _rank(row, seed),
    )
    for row in remaining:
        if len(chosen_ids) >= target_dev:
            break
        chosen_ids.add(id(row))

    dev = [_set_role(row, "development") for row in rows if id(row) in chosen_ids]
    train = [_set_role(row, "train") for row in rows if id(row) not in chosen_ids]
    train.sort(key=lambda row: _rank(row, seed))
    dev.sort(key=lambda row: _rank(row, seed))
    return train, dev


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", "--train_file", dest="input", required=True,
                        help="Comma-separated raw training JSONL files")
    parser.add_argument(
        "--forbidden_eval", "--frozen_eval", dest="forbidden_eval",
        action="append", default=[],
        help="Frozen evaluation JSONL file(s); repeat or use comma-separated specs",
    )
    parser.add_argument("--output", "--out_approved", dest="output", required=True)
    parser.add_argument("--dev_output", "--out_dev", dest="dev_output", default="")
    parser.add_argument("--bridge_output", default="",
                        help="Optional rows above --max_instructions through --max_bridge_instructions")
    parser.add_argument("--long_output", "--out_quarantine", dest="long_output", required=True,
                        help="Hierarchical-decoding holdout above --max_bridge_instructions")
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
                        help="Upper bound of the bridge holdout; rows above it enter the long holdout")
    parser.add_argument("--min_short_rows", "--min_approved_rows", dest="min_short_rows", type=int, default=1)
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
    if args.min_short_rows < 0 or args.min_dev_rows < 0:
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

    unique_raw: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    seen: dict[str, dict[str, str]] = {
        "neutral_sha256": {},
        "alpha_structural_sha256": {},
    }
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
        matched_duplicate = {
            kind: seen[kind][digest]
            for kind, digest in fingerprints.items()
            if digest in seen[kind]
        }
        if matched_duplicate:
            duplicates.append({
                "task": identity,
                "duplicate_of": sorted(set(matched_duplicate.values())),
                "source_fingerprints": fingerprints,
                "matched_fingerprint_kinds": sorted(matched_duplicate),
            })
            continue
        for kind, digest in fingerprints.items():
            seen[kind][digest] = identity
        unique_raw.append(row)

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

    short_all = [row for row in approved if instruction_count(row) <= args.max_instructions]
    bridge_rows = [
        _set_role(row, "bridge_quarantine")
        for row in approved
        if args.max_instructions < instruction_count(row) <= args.max_bridge_instructions
    ]
    long_rows = [
        _set_role(row, "long_quarantine")
        for row in approved
        if instruction_count(row) > args.max_bridge_instructions
    ]
    try:
        train_rows, dev_rows = _split_train_dev(
            short_all,
            seed=args.seed,
            dev_fraction=args.dev_fraction if args.dev_output else 0.0,
            min_train=args.min_short_rows,
            min_dev=args.min_dev_rows if args.dev_output else 0,
        )
        if not args.dev_output and data_role != "train":
            train_rows = [_set_role(row, data_role) for row in train_rows]
    except Exception as exc:
        train_rows, dev_rows = [], []
        preparation_failures.append({"task": "<train-dev-split>", "error": f"{type(exc).__name__}: {exc}"})

    rejected_rows: list[dict[str, Any]] = []
    for kind, values in (
        ("duplicate", duplicates),
        ("frozen_eval_overlap", overlaps),
        ("preparation_failure", preparation_failures),
        ("reference_failure", invalid_references),
    ):
        rejected_rows.extend({"rejection_type": kind, **value} for value in values)

    write_jsonl(args.output, train_rows)
    if args.dev_output:
        write_jsonl(args.dev_output, dev_rows)
    if args.bridge_output:
        write_jsonl(args.bridge_output, bridge_rows)
    else:
        # Backward-compatible single quarantine output. The report still records
        # bridge versus long counts so these populations cannot be conflated.
        long_rows = bridge_rows + long_rows
    write_jsonl(args.long_output, long_rows)
    if args.rejected_output:
        write_jsonl(args.rejected_output, rejected_rows)

    outputs = {
        "short": file_record(args.output),
        "development": file_record(args.dev_output) if args.dev_output else None,
        "bridge": file_record(args.bridge_output) if args.bridge_output else None,
        "long": file_record(args.long_output),
        "rejected": file_record(args.rejected_output) if args.rejected_output else None,
    }
    fatal: list[str] = []
    if overlaps:
        fatal.append(
            f"{len(overlaps)} exact-neutral or alpha-structural overlaps with frozen evaluation data"
        )
    if preparation_failures:
        fatal.append(f"{len(preparation_failures)} rows/splits failed preparation")
    if invalid_references and not args.drop_invalid_references:
        fatal.append(f"{len(invalid_references)} references failed full/feedback/acceptance replay")
    if len(train_rows) < args.min_short_rows:
        fatal.append(f"only {len(train_rows)} approved train rows; minimum is {args.min_short_rows}")
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
        "approved_short_total": len(short_all),
        "short_rows": len(train_rows),
        "development_rows": len(dev_rows),
        "bridge_rows": len(bridge_rows),
        "long_rows": len(long_rows) if args.bridge_output else max(0, len(long_rows) - len(bridge_rows)),
        "duplicates": duplicates,
        "duplicate_rows": len(duplicates),
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
        "train_rows": len(train_rows),
        "development_rows": len(dev_rows),
        "bridge_rows": len(bridge_rows),
        "long_rows": len(long_rows) if args.bridge_output else max(0, len(long_rows) - len(bridge_rows)),
    }, indent=2))

    if fatal:
        raise SystemExit("Phase-0 gate failed: " + "; ".join(fatal))


if __name__ == "__main__":
    main()
