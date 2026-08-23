#!/usr/bin/env python3
"""Materialize verified API-readable compact inputs for frontier/KL sampling."""
from __future__ import annotations

import argparse
from pathlib import Path

from frontier_core import (
    COMPACT_F2_SYSTEM_PROMPT,
    F2_SCHEMA,
    CompactArtifactBundle,
    atomic_write_json,
    atomic_write_jsonl,
    count_prompt_tokens,
    file_record,
    load_jsonl,
    prepare_api_readable_compact,
    sha256_file,
    sha256_text,
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
    parser.add_argument("--max-prompt-tokens", type=int, default=12000)
    parser.add_argument("--chat-overhead-reserve", type=int, default=256)
    parser.add_argument(
        "--max-constant-prefix-tokens",
        type=int,
        default=256,
        help=(
            "Student-side binary-constant prefix cap. Use 0 only for an "
            "explicitly rebuilt lossless-prefix dataset."
        ),
    )
    args = parser.parse_args()
    if args.max_prompt_tokens <= 0:
        raise SystemExit("--max-prompt-tokens must be positive")
    if args.chat_overhead_reserve < 0:
        raise SystemExit("--chat-overhead-reserve must be non-negative")
    if args.max_constant_prefix_tokens < 0:
        raise SystemExit("--max-constant-prefix-tokens cannot be negative")
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
        max_constant_prefix_tokens=(
            None
            if args.max_constant_prefix_tokens == 0
            else args.max_constant_prefix_tokens
        ),
    )
    serialized = [prepare_api_readable_compact(bundle, row) for row in rows]
    prompt_counts: list[tuple[str, dict[str, int]]] = []
    for value in serialized:
        count = count_prompt_tokens(
            [
                {"role": "system", "content": COMPACT_F2_SYSTEM_PROMPT},
                {"role": "user", "content": value["text"]},
            ],
            bundle.tokenizer,
            chat_overhead_reserve=args.chat_overhead_reserve,
        )
        value["prompt_preflight"] = count
        prompt_counts.append((value["task_id"], count))
    over_limit = [
        (task_id, count["estimated_prompt_tokens"])
        for task_id, count in prompt_counts
        if count["estimated_prompt_tokens"] > args.max_prompt_tokens
    ]
    if over_limit:
        preview = ", ".join(
            f"{task_id}={tokens}" for task_id, tokens in over_limit[:20]
        )
        raise SystemExit(
            f"{len(over_limit)} F2 prompts exceed --max-prompt-tokens "
            f"{args.max_prompt_tokens}: {preview}"
        )
    atomic_write_jsonl(args.out, serialized)
    max_task_id, max_count = max(
        prompt_counts,
        key=lambda item: item[1]["estimated_prompt_tokens"],
    )
    manifest = {
        "schema": "verified-api-readable-compact-v2",
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
        "f2_prompt_contract": {
            "representation_schema": F2_SCHEMA,
            "system_prompt": COMPACT_F2_SYSTEM_PROMPT,
            "system_prompt_sha256": sha256_text(COMPACT_F2_SYSTEM_PROMPT),
            "tokenizer_sha256": bundle.tokenizer_sha256,
            "constant_prefix_token_cap": bundle.max_constant_prefix_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "chat_overhead_reserve": args.chat_overhead_reserve,
            "maximum_estimated_prompt_tokens": max_count[
                "estimated_prompt_tokens"
            ],
            "maximum_task_id": max_task_id,
            "all_rows_within_limit": True,
        },
        "output": file_record(args.out),
        "invariants": {
            "all_artifact_hashes_verified": True,
            "all_row_contract_hashes_verified": True,
            "all_codec_roundtrips_verified": True,
            "all_student_constant_prefixes_verified": True,
            "all_f2_semantic_roundtrips_verified": True,
            "f2_system_prompt_self_contained_and_hashed": True,
            "all_complete_prompts_within_limit": True,
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
