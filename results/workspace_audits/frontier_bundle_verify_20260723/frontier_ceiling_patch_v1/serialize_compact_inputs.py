#!/usr/bin/env python3
"""Materialize verified API-readable compact inputs for frontier/KL sampling."""
from __future__ import annotations

import argparse
from pathlib import Path

from frontier_core import (
    CompactArtifactBundle,
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    load_jsonl,
    prepare_api_readable_compact,
    sha256_file,
    stable_sha256,
    utc_now,
)

WORKSPACE = Path("/workspace")
RB = WORKSPACE / "artifacts" / "compact_fn0_rebuild"
CODEBOOK = (
    WORKSPACE
    / "direct_compact_stage"
    / "scrubbed_master_v2_release"
    / "direct_compact_split_v1"
    / "compact_qwen_confirmatory_v1"
    / "codebook.json"
)
TOKENIZER = (
    WORKSPACE
    / ".hf_home"
    / "hub"
    / "models--Qwen--Qwen3-8B"
    / "snapshots"
    / "b968826d9c46dd6066d109eabc6255188de91218"
    / "tokenizer.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=RB / "fn0_contract.json")
    parser.add_argument("--codebook", type=Path, default=CODEBOOK)
    parser.add_argument("--tokenizer-json", type=Path, default=TOKENIZER)
    parser.add_argument(
        "--codec",
        type=Path,
        default=WORKSPACE
        / "direct_compact_stage"
        / "scripts"
        / "data"
        / "build_compact_qwen_v1.py",
    )
    parser.add_argument("--constants", type=Path, default=RB / "real_constants.jsonl")
    parser.add_argument("--expected-constants-sha256", required=True)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    actual_dataset_sha = sha256_file(dataset)
    if actual_dataset_sha != args.expected_dataset_sha256.strip().lower():
        raise SystemExit(
            "dataset hash mismatch: "
            f"expected {args.expected_dataset_sha256}, got {actual_dataset_sha}"
        )
    rows = load_jsonl(dataset, "compact dataset")
    if len(rows) != args.expected_rows:
        raise SystemExit(
            f"dataset has {len(rows)} rows, expected {args.expected_rows}"
        )
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if any(not task_id for task_id in task_ids):
        raise SystemExit("one or more dataset rows has no task_id")
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("dataset has duplicate task IDs")
    bundle = CompactArtifactBundle(
        contract_path=args.contract,
        codebook_path=args.codebook,
        tokenizer_path=args.tokenizer_json,
        codec_path=args.codec,
        constants_path=args.constants,
        expected_constants_sha256=args.expected_constants_sha256,
    )
    serialized = [prepare_api_readable_compact(bundle, row) for row in rows]
    atomic_write_jsonl(args.out, serialized)
    manifest = {
        "schema": "verified-api-readable-compact-v1",
        "created_at": utc_now(),
        "dataset": file_record(dataset),
        "task_set_sha256": stable_sha256(task_ids),
        "rows": len(serialized),
        "binary_constant_extraction_errors": {
            "count": sum(
                value["constants_extraction_error"] is not None
                for value in serialized
            ),
            "task_ids": [
                value["task_id"]
                for value in serialized
                if value["constants_extraction_error"] is not None
            ],
        },
        "artifacts": bundle.artifact_records(),
        "output": file_record(args.out),
        "invariants": {
            "all_artifact_hashes_verified": True,
            "all_row_contract_hashes_verified": True,
            "all_codec_roundtrips_verified": True,
            "all_student_constant_prefixes_verified": True,
            "opaque_source_ids_expanded": True,
            "cfg_explicit": True,
        },
    }
    atomic_write_json(args.out.with_suffix(args.out.suffix + ".manifest.json"), manifest)
    print(
        f"SERIALIZED_COMPACT_INPUTS rows={len(serialized)} "
        f"sha256={manifest['output']['sha256']} out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
