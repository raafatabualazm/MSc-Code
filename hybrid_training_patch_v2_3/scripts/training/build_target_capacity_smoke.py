#!/usr/bin/env python3
"""Build a one-row near-limit direct-compact training smoke artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from models.direct_compact_causal import DirectCompactContract, sha256_file


TARGET_FIELDS = ("supervised_target", "dart_source", "source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--source-seal", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-seal", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--eos-token-id", required=True, type=int)
    parser.add_argument("--minimum-target-tokens", type=int, default=24560)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    for path in (args.output_jsonl, args.output_seal, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite smoke artifact: {path}")
    contract = DirectCompactContract.load(args.contract)
    if args.minimum_target_tokens >= contract.max_target_tokens:
        raise ValueError("minimum target tokens must leave one EOS position")
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tokenizers is required") from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    if sha256_file(args.tokenizer_json) != contract.tokenizer_json_sha256:
        raise ValueError("tokenizer JSON differs from the compact contract")
    if args.eos_token_id < 0:
        raise ValueError("EOS token ID must be non-negative")

    rows: list[dict[str, Any]] = []
    with args.source_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank source row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"source row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("source dataset is empty")
    row = min(
        rows,
        key=lambda value: len(value.get("compact_input_ids") or []),
    )
    identity = str(row.get("task_id") or "capacity-smoke")
    compact_ids = contract.validate_row(row, identity)

    prefix = "int fn0() {\n  /*"
    suffix = " */\n  return 0;\n}"

    def target_for(repetitions: int) -> tuple[str, list[int]]:
        text = prefix + (" x" * repetitions) + suffix
        return text, list(tokenizer.encode(text, add_special_tokens=False).ids)

    low = 1
    high = contract.max_target_tokens * 2
    best_text = ""
    best_ids: list[int] = []
    while low <= high:
        middle = (low + high) // 2
        text, token_ids = target_for(middle)
        eos_inclusive = len(token_ids) + (
            0
            if token_ids and token_ids[-1] == args.eos_token_id
            else 1
        )
        if eos_inclusive <= contract.max_target_tokens:
            best_text = text
            best_ids = token_ids
            low = middle + 1
        else:
            high = middle - 1
    eos_inclusive_target_tokens = len(best_ids) + (
        0 if best_ids and best_ids[-1] == args.eos_token_id else 1
    )
    if eos_inclusive_target_tokens < args.minimum_target_tokens:
        raise RuntimeError(
            "could not construct a sufficiently long target: "
            f"{eos_inclusive_target_tokens}"
        )

    output_row = dict(row)
    output_row["task_id"] = f"{identity}__target_capacity_smoke"
    output_row["dart_source"] = best_text
    for field in TARGET_FIELDS:
        if field in output_row:
            output_row[field] = best_text
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    encoded_row = json.dumps(
        output_row,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    args.output_jsonl.write_text(encoded_row + "\n", encoding="utf-8")

    source_seal = _load_json(args.source_seal)
    smoke_seal = dict(source_seal)
    smoke_seal["rows"] = 1
    if "source_rows" in smoke_seal:
        smoke_seal["source_rows"] = 1
    smoke_seal["output_sha256"] = sha256_file(args.output_jsonl)
    smoke_seal["output_size_bytes"] = args.output_jsonl.stat().st_size
    smoke_seal["task_set_sha256"] = _stable_sha256(
        [output_row["task_id"]]
    )
    if "output" in smoke_seal:
        smoke_seal["output"] = {
            "path": str(args.output_jsonl.resolve()),
            "sha256": sha256_file(args.output_jsonl),
            "bytes": args.output_jsonl.stat().st_size,
            "size_bytes": args.output_jsonl.stat().st_size,
        }
    pool = smoke_seal.get("pool_metadata")
    if isinstance(pool, dict):
        raise ValueError(
            "v3 pool seals need a dedicated projection builder; refusing smoke"
        )
    args.output_seal.write_text(
        json.dumps(
            smoke_seal,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    total_without_prompt = len(compact_ids) + eos_inclusive_target_tokens
    report = {
        "schema": "direct-compact-target-capacity-smoke-v1",
        "source_task_id": identity,
        "smoke_task_id": output_row["task_id"],
        "compact_source_tokens": len(compact_ids),
        "eos_inclusive_target_tokens": eos_inclusive_target_tokens,
        "total_without_prompt_tokens": total_without_prompt,
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256_file(args.contract),
            "max_target_tokens": contract.max_target_tokens,
            "max_total_tokens": contract.max_total_tokens,
            "eos_token_id": args.eos_token_id,
        },
        "output": {
            "path": str(args.output_jsonl.resolve()),
            "sha256": sha256_file(args.output_jsonl),
        },
        "seal": {
            "path": str(args.output_seal.resolve()),
            "sha256": sha256_file(args.output_seal),
        },
        "invariants": {
            "no_source_truncation": True,
            "target_within_contract": True,
            "near_target_capacity": True,
            "training_semantics": "memory_smoke_only",
        },
    }
    args.report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "DIRECT_COMPACT_TARGET_CAPACITY_SMOKE_BUILT "
        f"source={len(compact_ids)} target={eos_inclusive_target_tokens} "
        f"max_target={contract.max_target_tokens}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
