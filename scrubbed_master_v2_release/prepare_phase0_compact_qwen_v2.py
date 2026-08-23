#!/usr/bin/env python3
"""Reconcile and materialize the exact Phase-0 s44 corpus for compact-Qwen v2.

Membership and train/dev assignment come only from the supplied Phase-0
manifest.  This stage neutralizes top-up targets to ``candidate``, applies the
frozen compact-v1 leakage fingerprints, quarantines corrupt mnemonics, and
emits codec-private graph rows plus separately sealed private labels.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(HERE))

from hybrid_training_patch_v2_3.scripts.training import hybrid_data_controls as controls
from scripts.data import build_compact_qwen_v2 as codec
from prepare_direct_compact_split import (
    JACCARD_THRESHOLD,
    SEQUENCE_THRESHOLD,
    Entry,
    SimilarityIndex,
    fingerprint_source,
    make_entry,
)


SCHEMA = "phase0-compact-qwen-v2-preparation"
INPUT_SHA256 = "312d5a7cfc9a5866c38479a3384bd49f47b55b26b1dd46200bb70539945e9b65"
SPLIT_SHA256 = "d69d7110a63d768207ec4eaf5bf03ce1afc0cb431326358c102fef8c40093258"
SOURCE_POOL_MANIFEST_SHA256 = "4e3c728d0250b9875b72810cd0aefb926119011efdc5fa1cbc88045a7a288b64"
EXPECTED_INPUT_ROWS = 3306
EXPECTED_MANIFEST_ROWS = 3305
EXPECTED_STATUS_COUNTS = {
    "included-train": 2951,
    "included-dev": 326,
    "quarantined": 14,
    "excluded": 15,
}
EXPECTED_NEAR_CLONES = 14
EXPECTED_UNLISTED = {"sigless_4901067c13b9"}
EXPECTED_CORRUPT = {
    "sigless_de21a24ed292",
    "sigless_997b3f344158",
    "sigless_73473598ab1b",
    "sigless_031f51486eea",
    "sigless_0bf7ae03ddfe",
    "sigless_e0c5a7b6ec4b",
    "sigless_ce43724c502b",
    "sigless_1b3267e7dd3f",
    "sigless_37d21415f60e",
    "sigless_bd4cf0d1d7db",
    "sigless_a8b709e424be",
    "sigless_bb79fb181ca4",
    "sigless_179318026249",
    "sigless_b51cb0fef9eb",
}
EXPECTED_NAMED_LONG_CORRUPT = {
    "sigless_179318026249",
    "sigless_b51cb0fef9eb",
}

SOURCE_POOLS = {
    "base_llm": (
        WORKSPACE / "data/testing/fresh_eval_llm_graphv2.jsonl",
        "9d1fec926ce5721bdc13e3293720f7e21d8057fe57236bc58f569fb8bb88d9f2",
    ),
    "topup_s44": (
        WORKSPACE / "data/testing/fresh_eval_lowmid_topup_s44_graphv2.jsonl",
        "6bdef7b566fc5c083fb09f4329d359139a5e028f2b643d33c9e7ec2b1c1d1cb4",
    ),
    "topup_s45": (
        WORKSPACE / "data/testing/fresh_eval_low_topup_deepseek_s45_graphv2.jsonl",
        "162a02882798c99166d65e31344a0ca3cb4d9374a04e7589b1643604c3941020",
    ),
    "topup_s46": (
        WORKSPACE / "data/testing/fresh_eval_low_topup_chatgpt_s46_graphv2.jsonl",
        "1f16c3e5060cc15be7e24f879978a40171a79b29e91054eecbeb751df3299101",
    ),
}
FAMILY_POLICY = {
    "version": "pre-s46-phase-umbrella-v1",
    "master": "master",
    "base_llm": "topup_s45",
    "topup_s44": "topup_s45",
    "topup_s45": "topup_s45",
    "topup_s46": "topup_s46",
    "note": (
        "topup_s45 is a coarse pre-s46 experimental bucket, not literal source-pool "
        "provenance; source_pool is retained privately"
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def stable_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_jsonl_raw(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    result: list[tuple[dict[str, Any], bytes]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            result.append((value, raw.rstrip(b"\r\n")))
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for row, _ in read_jsonl_raw(path)]


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def describe(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def infer_source_pool_map() -> tuple[dict[str, str], list[dict[str, Any]]]:
    mapping: dict[str, str] = {}
    bindings: list[dict[str, Any]] = []
    for pool, (path, expected_sha) in SOURCE_POOLS.items():
        observed = sha256_file(path)
        if observed != expected_sha:
            raise ValueError(f"source pool {pool} SHA mismatch: {observed} != {expected_sha}")
        pool_rows = read_jsonl(path)
        for line_number, row in enumerate(pool_rows, 1):
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise ValueError(f"{path}:{line_number}: missing task_id")
            if task_id in mapping:
                raise ValueError(f"top-up task appears in multiple source pools: {task_id}")
            mapping[task_id] = pool
        bindings.append({"source_pool": pool, **describe(path, len(pool_rows))})
    return mapping, bindings


def normalize_family(task_id: str, source_pool: str | None) -> str:
    if task_id.startswith("sigless_"):
        if source_pool is not None:
            raise ValueError(f"master task unexpectedly belongs to source pool: {task_id}")
        return "master"
    if not task_id.startswith("fresh-eval-"):
        raise ValueError(f"unknown task family prefix: {task_id}")
    if source_pool not in SOURCE_POOLS:
        raise ValueError(f"top-up task has no authoritative source pool: {task_id}")
    return str(FAMILY_POLICY[source_pool])


def neutralize_row(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    original_target = str(row.get("function") or "")
    if original_target == "candidate":
        if not str(row.get("task_id") or "").startswith("sigless_"):
            raise ValueError("top-up unexpectedly already uses candidate")
        neutral = json.loads(json.dumps(row))
    else:
        neutral = controls.neutralize_training_row(row, neutral_name="candidate")
    if neutral.get("function") != "candidate":
        raise ValueError("target neutralization failed")
    source = controls.source_text(neutral)
    if not source.strip():
        raise ValueError("neutralized row has no target source")
    observed_signature = controls.extract_source_signature(source, "candidate")
    if not observed_signature:
        raise ValueError("neutralized target source has no candidate declaration")
    return neutral, original_target


def forbidden_entries(paths: list[Path]) -> tuple[list[Entry], list[dict[str, Any]]]:
    entries: list[Entry] = []
    bindings: list[dict[str, Any]] = []
    for path in paths:
        values = read_jsonl(path)
        bindings.append(describe(path, len(values)))
        for line_number, row in enumerate(values, 1):
            entries.append(make_entry(row, line_number, relative(path)))
    return entries, bindings


def best_forbidden_match(
    entry: Entry,
    entries: list[Entry],
    exact: dict[str, list[int]],
    alpha: dict[str, list[int]],
    index: SimilarityIndex,
) -> tuple[dict[str, Any] | None, int]:
    if entry.neutral_sha256 in exact:
        candidate = entries[exact[entry.neutral_sha256][0]]
        return {
            "reason": "forbidden_exact_neutral",
            "matched_forbidden_path": candidate.corpus,
            "matched_forbidden_line": candidate.original_line,
        }, 0
    if entry.alpha_sha256 in alpha:
        candidate = entries[alpha[entry.alpha_sha256][0]]
        return {
            "reason": "forbidden_alpha_structural",
            "matched_forbidden_path": candidate.corpus,
            "matched_forbidden_line": candidate.original_line,
        }, 0
    matches, comparisons = index.query(entry)
    if not matches:
        return None, comparisons
    candidate_index, jaccard, sequence = matches[0]
    candidate = entries[candidate_index]
    return {
        "reason": "forbidden_near_clone",
        "matched_forbidden_path": candidate.corpus,
        "matched_forbidden_line": candidate.original_line,
        "jaccard": round(jaccard, 6),
        "sequence": round(sequence, 6),
        "match_count": len(matches),
    }, comparisons


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE / "data/training/combined_fresh_s44_train_input.clean.jsonl",
    )
    parser.add_argument(
        "--phase0-manifest",
        type=Path,
        default=WORKSPACE
        / "phase0_split_manifest_s44/phase0_split_manifest_s44_20260720.jsonl",
    )
    parser.add_argument("--forbidden", action="append", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "direct_compact_phase0_s44_v2/prepared",
    )
    args = parser.parse_args()
    forbidden_paths = args.forbidden or [
        WORKSPACE / "data/testing/grpo_data_graphv2_signature_scrubbed_private.jsonl",
        WORKSPACE / "data/testing/grpo_data_graphv2_sigscrub_v2_nameonly_private.jsonl",
        WORKSPACE / "data/testing/grpo_data_graphv2_sigscrub_v2_neutralexact_private.jsonl",
        WORKSPACE / "data/testing/grpo_data_graphv2_sigscrub_v3_opaque_nameonly_private.jsonl",
        WORKSPACE / "data/testing/grpo_data_graphv2_sigscrub_v3_opaque_neutralexact_private.jsonl",
        WORKSPACE
        / "data/testing/humaneval_v2_nameonly_dfg_harmonized/"
        "humaneval_v2_nameonly_dfg_harmonized_private.jsonl",
        WORKSPACE / "data/testing/fresh_graphv2_holdout_s44.jsonl",
        WORKSPACE / "data/testing/neutral_functional_eval_s44_rebuilt_490.jsonl",
    ]
    required = [args.input, args.phase0_manifest, *forbidden_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    if sha256_file(args.input) != INPUT_SHA256:
        raise ValueError("canonical input SHA mismatch")
    if sha256_file(args.phase0_manifest) != SPLIT_SHA256:
        raise ValueError("Phase-0 split manifest SHA mismatch")
    source_pool_manifest = (
        WORKSPACE / "data/training/fresh_graphv2_train_inputs_s44.manifest.json"
    )
    observed_source_pool_manifest_sha = sha256_file(source_pool_manifest)
    if observed_source_pool_manifest_sha != SOURCE_POOL_MANIFEST_SHA256:
        raise ValueError(
            "source-pool manifest SHA mismatch: "
            f"{observed_source_pool_manifest_sha} != {SOURCE_POOL_MANIFEST_SHA256}"
        )

    raw_rows = read_jsonl_raw(args.input)
    split_rows = read_jsonl(args.phase0_manifest)
    if len(raw_rows) != EXPECTED_INPUT_ROWS or len(split_rows) != EXPECTED_MANIFEST_ROWS:
        raise ValueError("canonical corpus/manifest row count mismatch")
    input_by_id: dict[str, tuple[int, dict[str, Any], bytes]] = {}
    for input_line, (row, raw) in enumerate(raw_rows, 1):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in input_by_id:
            raise ValueError(f"missing or duplicate input task_id at line {input_line}")
        input_by_id[task_id] = (input_line, row, raw)
    manifest_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for manifest_line, row in enumerate(split_rows, 1):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in manifest_by_id:
            raise ValueError(f"missing or duplicate manifest task_id at line {manifest_line}")
        if row.get("split") not in {"train", "dev"}:
            raise ValueError(f"invalid Phase-0 split for {task_id}")
        manifest_by_id[task_id] = (manifest_line, row)
    if set(manifest_by_id) - set(input_by_id):
        raise ValueError("Phase-0 manifest contains tasks absent from canonical input")
    unlisted = set(input_by_id) - set(manifest_by_id)
    if unlisted != EXPECTED_UNLISTED:
        raise ValueError(f"unexpected unlisted input tasks: {sorted(unlisted)}")

    pool_by_id, pool_bindings = infer_source_pool_map()
    topup_ids = {task_id for task_id in input_by_id if task_id.startswith("fresh-eval-")}
    if collections.Counter(pool_by_id[task_id] for task_id in topup_ids) != {
        "base_llm": 449,
        "topup_s44": 226,
        "topup_s45": 324,
        "topup_s46": 118,
    }:
        raise ValueError("canonical top-up source-pool reconciliation drift")

    forbidden, forbidden_bindings = forbidden_entries(forbidden_paths)
    exact: dict[str, list[int]] = collections.defaultdict(list)
    alpha: dict[str, list[int]] = collections.defaultdict(list)
    for index, entry in enumerate(forbidden):
        exact[entry.neutral_sha256].append(index)
        alpha[entry.alpha_sha256].append(index)
    similarity = SimilarityIndex(forbidden)

    prepared: dict[str, dict[str, Any]] = {}
    reconciliation: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    overlap_audit: list[dict[str, Any]] = []
    comparisons = 0
    for task_id, (input_line, raw_row, raw_bytes) in input_by_id.items():
        manifest_item = manifest_by_id.get(task_id)
        manifest_line = manifest_item[0] if manifest_item else None
        assignment = manifest_item[1] if manifest_item else None
        source_pool = pool_by_id.get(task_id)
        family = normalize_family(task_id, source_pool)
        base_reconciliation = {
            "input_line": input_line,
            "task_id": task_id,
            "input_row_sha256": sha256_bytes(raw_bytes),
            "phase0_manifest_line": manifest_line,
            "phase0_manifest_present": assignment is not None,
            "phase0_split": assignment.get("split") if assignment else None,
            "in_long_dev_ge200": bool(assignment and assignment.get("in_long_dev_ge200")),
            "family": family,
            "source_pool": source_pool,
            "generator_provider": raw_row.get("generator_provider"),
            "generator_model": raw_row.get("generator_model"),
            "phase0_family_raw": assignment.get("family") if assignment else None,
            "source_graph_extractor_sha256": str(
                (raw_row.get("graph_v2") or {}).get("extractor_sha256") or ""
            ),
            "original_target": str(raw_row.get("function") or ""),
            "target_function": "candidate",
            "model_row": None,
        }
        if assignment is None:
            reconciliation.append(
                {
                    **base_reconciliation,
                    "status": "excluded",
                    "reason": (
                        "not_in_phase0_manifest:"
                        "invalid_reference_nondeterministic_datetime_stdout"
                    ),
                }
            )
            continue

        neutral, original_target = neutralize_row(raw_row)
        try:
            canonical = codec.canonicalize(neutral)
        except ValueError as error:
            reason = str(error)
            if not reason.startswith("unknown_or_corrupt_mnemonic:local_"):
                raise
            item = {
                **base_reconciliation,
                "original_target": original_target,
                "status": "quarantined",
                "reason": reason,
            }
            reconciliation.append(item)
            quarantine.append(item)
            continue

        entry = make_entry(raw_row, input_line, relative(args.input))
        match, compared = best_forbidden_match(entry, forbidden, exact, alpha, similarity)
        comparisons += compared
        if match:
            forbidden_entry = next(
                (
                    value
                    for value in forbidden
                    if value.corpus == match["matched_forbidden_path"]
                    and value.original_line == match["matched_forbidden_line"]
                ),
                None,
            )
            if forbidden_entry is not None:
                forbidden_rows = read_jsonl(Path(WORKSPACE / forbidden_entry.corpus))
                match["matched_forbidden_task_id"] = forbidden_rows[
                    forbidden_entry.original_line - 1
                ].get("task_id")
            item = {
                **base_reconciliation,
                "original_target": original_target,
                "status": "excluded",
                **match,
            }
            reconciliation.append(item)
            overlap_audit.append(item)
            continue

        metadata = {
            "input_line": input_line,
            "input_row_sha256": sha256_bytes(raw_bytes),
            "phase0_manifest_line": manifest_line,
            "phase0_split": assignment["split"],
            "in_long_dev_ge200": bool(assignment.get("in_long_dev_ge200")),
            "family": family,
            "phase0_family_raw": assignment.get("family"),
            "source_pool": source_pool,
            "generator_provider": raw_row.get("generator_provider"),
            "generator_model": raw_row.get("generator_model"),
            "original_target_sha256": sha256_bytes(original_target.encode("utf-8")),
            "target_function": "candidate",
            "dfg_route_source": "row_graph_v2",
        }
        codec_row = {
            "task_id": task_id,
            "lang": str(neutral.get("lang") or "Dart"),
            "function": "candidate",
            "cfg": neutral["cfg"],
            "edges": neutral["edges"],
            "integrity": neutral["integrity"],
            "graph_v2": neutral["graph_v2"],
            "compact_private_metadata": metadata,
        }
        label = {
            "task_id": task_id,
            "lang": str(neutral.get("lang") or "Dart"),
            "function": "candidate",
            "dart_source": controls.source_text(neutral),
            "family": family,
        }
        prepared[task_id] = {
            "codec": codec_row,
            "label": label,
            "canonical_sha256": stable_sha256(canonical),
            "reconciliation": base_reconciliation,
        }

    if {item["task_id"] for item in quarantine} != EXPECTED_CORRUPT:
        raise ValueError("corrupt mnemonic quarantine set drift")
    named_long = {
        item["task_id"]
        for item in quarantine
        if item["in_long_dev_ge200"]
    }
    if named_long != EXPECTED_NAMED_LONG_CORRUPT:
        raise ValueError("long-dev corrupt quarantine set drift")
    near_count = sum(item["reason"] == "forbidden_near_clone" for item in overlap_audit)
    if near_count != EXPECTED_NEAR_CLONES:
        raise ValueError(f"forbidden near-clone count drift: {near_count}")
    if any(item["reason"] != "forbidden_near_clone" for item in overlap_audit):
        raise ValueError("unexpected exact or alpha forbidden collision")

    train_codec: list[dict[str, Any]] = []
    dev_codec: list[dict[str, Any]] = []
    train_labels: list[dict[str, Any]] = []
    dev_labels: list[dict[str, Any]] = []
    model_row_by_id: dict[str, int] = {}
    for manifest_line, assignment in enumerate(split_rows, 1):
        task_id = assignment["task_id"]
        if task_id not in prepared:
            continue
        item = prepared[task_id]
        if assignment["split"] == "train":
            model_row_by_id[task_id] = len(train_codec)
            train_codec.append(item["codec"])
            train_labels.append(item["label"])
        else:
            dev_codec.append(item["codec"])
            dev_labels.append(item["label"])
    dev_offset = len(train_codec)
    for index, row in enumerate(dev_codec):
        model_row_by_id[row["task_id"]] = dev_offset + index

    completed_reconciliation: list[dict[str, Any]] = []
    status_by_id = {
        item["task_id"]: item for item in reconciliation
    }
    for task_id, (input_line, raw_row, raw_bytes) in input_by_id.items():
        if task_id in status_by_id:
            completed_reconciliation.append(status_by_id[task_id])
            continue
        item = prepared[task_id]
        split = item["reconciliation"]["phase0_split"]
        completed_reconciliation.append(
            {
                **item["reconciliation"],
                "status": f"included-{split}",
                "reason": "phase0_assignment_after_quarantine_and_forbidden_gates",
                "canonical_sha256": item["canonical_sha256"],
                "model_row": model_row_by_id[task_id],
            }
        )
    reconciliation = completed_reconciliation
    status_counts = dict(collections.Counter(item["status"] for item in reconciliation))
    if status_counts != EXPECTED_STATUS_COUNTS:
        raise ValueError(f"final reconciliation counts drift: {status_counts}")
    if len(reconciliation) != EXPECTED_INPUT_ROWS:
        raise ValueError("reconciliation does not cover every input row")
    if [row["task_id"] for row in train_codec] != [row["task_id"] for row in train_labels]:
        raise ValueError("train codec/private label alignment mismatch")
    if [row["task_id"] for row in dev_codec] != [row["task_id"] for row in dev_labels]:
        raise ValueError("dev codec/private label alignment mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_codec_private.jsonl": train_codec,
        "dev_codec_private.jsonl": dev_codec,
        "train_private_labels.jsonl": train_labels,
        "dev_private_labels.jsonl": dev_labels,
        "reconciliation.jsonl": reconciliation,
        "quarantine.jsonl": quarantine,
        "forbidden_overlap_audit.jsonl": overlap_audit,
    }
    counts: dict[str, int] = {}
    for name, values in outputs.items():
        counts[name] = write_jsonl(args.output_dir / name, values)

    manifest = {
        "schema": SCHEMA,
        "canonical_membership": "phase0_split_manifest_s44_20260720 task IDs only",
        "ordering": {
            "codec_rows": "Phase-0 manifest order, train then dev",
            "reconciliation": "canonical input line order",
        },
        "family_policy": FAMILY_POLICY,
        "runtime_symbol_policy": codec.RUNTIME_POLICY,
        "runtime_symbol_policy_sha256": stable_sha256(codec.RUNTIME_POLICY),
        "target_function": "candidate",
        "counts": {
            "input": len(raw_rows),
            "phase0_manifest": len(split_rows),
            "statuses": status_counts,
            "train_codec": len(train_codec),
            "dev_codec": len(dev_codec),
            "long_dev_included": sum(
                item["status"] == "included-dev" and item["in_long_dev_ge200"]
                for item in reconciliation
            ),
            "long_dev_quarantined": sum(
                item["status"] == "quarantined" and item["in_long_dev_ge200"]
                for item in reconciliation
            ),
            "forbidden_near_clones": near_count,
        },
        "leakage_policy": {
            "fingerprints": ["exact normalized", "alpha structural", "alpha-token 5-gram"],
            "jaccard_threshold": JACCARD_THRESHOLD,
            "sequence_matcher_threshold": SEQUENCE_THRESHOLD,
            "jaccard_comparisons": comparisons,
            "demo_main_removed_before_fingerprint": True,
            "compiler_pragmas_removed_before_fingerprint": True,
        },
        "inputs": {
            "canonical_corpus": describe(args.input, len(raw_rows)),
            "phase0_split_manifest": describe(args.phase0_manifest, len(split_rows)),
            "source_pool_manifest": describe(
                source_pool_manifest
            ),
            "source_pools": pool_bindings,
            "forbidden": forbidden_bindings,
        },
        "scripts": {
            "preparer": describe(Path(__file__)),
            "compact_codec": describe(WORKSPACE / "scripts/data/build_compact_qwen_v2.py"),
            "hybrid_data_controls": describe(
                WORKSPACE
                / "hybrid_training_patch_v2_3/scripts/training/hybrid_data_controls.py"
            ),
            "fingerprint_implementation": describe(
                HERE / "prepare_direct_compact_split.py"
            ),
        },
        "gates": {
            "input_and_manifest_hashes_exact": True,
            "manifest_assignment_preserved": True,
            "every_input_row_reconciled_once": True,
            "codebook_fit_input_is_train_only": True,
            "private_targets_uniform_candidate": True,
            "all_corrupt_mnemonics_quarantined": True,
            "all_forbidden_near_clones_excluded": True,
            "source_pool_provenance_retained": True,
            "passed": True,
        },
        "outputs": {
            name: describe(args.output_dir / name, counts[name]) for name in outputs
        },
    }
    manifest_path = args.output_dir / "preparation_manifest.json"
    write_json(manifest_path, manifest)
    sealed_names = [*outputs, "preparation_manifest.json"]
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256_file(args.output_dir / name)}  {name}\n"
            for name in sealed_names
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
