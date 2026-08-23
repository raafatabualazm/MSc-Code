#!/usr/bin/env python3
"""Build compact train/dev rows with whole-user-binary constant enrichment.

This consumes the un-enriched sealed compact rows and a hash-pinned JSONL made
by the binary object-pool extractor in ``whole`` scope (fn0 plus user-source
helpers/closures).  It never reads source literals to construct the prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEAL_SCHEMA = "compact-public-private-join-seal-v1"
BUILD_SCHEMA = "whole-binary-enriched-compact-build-v1"


class BuildError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser().resolve()
    return {
        "path": str(value),
        "sha256": sha256_file(value),
        "bytes": value.stat().st_size,
    }


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != expected.strip().lower():
        raise BuildError(
            f"{label} hash mismatch: expected {expected}, "
            f"got {record['sha256']}"
        )
    return record


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BuildError(f"{label} has a blank row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BuildError(
                    f"{label} line {line_number} is not a JSON object"
                )
            rows.append(value)
    if not rows:
        raise BuildError(f"{label} is empty")
    return rows


def constant_preamble(
    strings: Sequence[Any], numbers: Sequence[Any]
) -> str:
    """Match the deployed compact binary-enrichment format exactly."""

    normalized_strings = sorted(
        {str(value) for value in strings}, key=lambda value: (len(value), value)
    )[:48]
    normalized_numbers = sorted(
        {str(value) for value in numbers}, key=lambda value: (len(value), value)
    )[:48]
    parts: list[str] = []
    if normalized_strings:
        parts.append(
            "strings: "
            + " ".join(f'"{value}"' for value in normalized_strings)
        )
    if normalized_numbers:
        parts.append("numbers: " + " ".join(normalized_numbers))
    if not parts:
        return ""
    return (
        "// constant pool recovered from binary\n// "
        + " | ".join(parts)
        + "\n"
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--expected-base-train-sha256", required=True)
    parser.add_argument("--base-dev", required=True, type=Path)
    parser.add_argument("--expected-base-dev-sha256", required=True)
    parser.add_argument("--constants", required=True, type=Path)
    parser.add_argument("--expected-constants-sha256", required=True)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-train-seal", required=True, type=Path)
    parser.add_argument("--output-dev", required=True, type=Path)
    parser.add_argument("--output-dev-seal", required=True, type=Path)
    parser.add_argument("--build-report", required=True, type=Path)
    parser.add_argument("--expected-train-rows", type=int, default=1580)
    parser.add_argument("--expected-dev-rows", type=int, default=175)
    parser.add_argument(
        "--constants-scope",
        choices=("whole_user_source_functions",),
        default="whole_user_source_functions",
    )
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_train = args.base_train.expanduser().resolve()
    base_dev = args.base_dev.expanduser().resolve()
    constants_path = args.constants.expanduser().resolve()
    tokenizer_path = args.tokenizer_json.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    inputs = {
        "base_train": require_hash(
            base_train, args.expected_base_train_sha256, "base train"
        ),
        "base_dev": require_hash(
            base_dev, args.expected_base_dev_sha256, "base dev"
        ),
        "whole_binary_constants": require_hash(
            constants_path,
            args.expected_constants_sha256,
            "whole-binary constants",
        ),
        "tokenizer": require_hash(
            tokenizer_path,
            args.expected_tokenizer_sha256,
            "student tokenizer",
        ),
        "contract": require_hash(
            contract_path, args.expected_contract_sha256, "compact contract"
        ),
    }
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "direct-compact-causal-v1":
        raise BuildError("compact contract schema mismatch")
    if (
        contract.get("tokenizer_json_sha256")
        != inputs["tokenizer"]["sha256"]
    ):
        raise BuildError("contract tokenizer binding mismatch")
    base_vocab_size = int(contract["base_vocab_size"])
    max_source_tokens = int(contract["max_source_tokens"])
    expected_row_hashes = {
        "compact_codebook_sha256": str(contract["codebook_sha256"]),
        "compact_codec_sha256": str(contract["codec_sha256"]),
        "compact_tokenizer_sha256": str(contract["tokenizer_json_sha256"]),
    }
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise BuildError("the tokenizers package is required") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    train_rows = read_jsonl(base_train, "base train")
    dev_rows = read_jsonl(base_dev, "base dev")
    if len(train_rows) != args.expected_train_rows:
        raise BuildError(
            f"base train has {len(train_rows)} rows, "
            f"expected {args.expected_train_rows}"
        )
    if len(dev_rows) != args.expected_dev_rows:
        raise BuildError(
            f"base dev has {len(dev_rows)} rows, "
            f"expected {args.expected_dev_rows}"
        )
    all_rows = train_rows + dev_rows
    task_ids = [str(row.get("task_id") or "") for row in all_rows]
    if any(not task_id for task_id in task_ids):
        raise BuildError("a base row has no task_id")
    if len(set(task_ids)) != len(task_ids):
        raise BuildError("base train/dev task IDs are not globally unique")

    constant_rows = read_jsonl(constants_path, "whole-binary constants")
    constants: dict[str, dict[str, Any]] = {}
    for row in constant_rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in constants:
            raise BuildError(
                f"invalid/duplicate constants task_id {task_id!r}"
            )
        if not isinstance(row.get("strings"), list) or not isinstance(
            row.get("numbers"), list
        ):
            raise BuildError(
                f"{task_id}: constants strings/numbers are not arrays"
            )
        constants[task_id] = row
    if set(constants) != set(task_ids):
        missing = sorted(set(task_ids) - set(constants))
        extra = sorted(set(constants) - set(task_ids))
        raise BuildError(
            "constants/base task sets differ: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    stats = {
        "rows": 0,
        "enriched_rows": 0,
        "constant_empty_rows": 0,
        "constant_extraction_error_rows": 0,
        "constant_prefix_tokens": 0,
        "maximum_constant_prefix_tokens": 0,
        "maximum_constant_prefix_task_id": "",
        "maximum_output_source_tokens": 0,
    }

    def enrich(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for raw in rows:
            row = dict(raw)
            task_id = str(row["task_id"])
            for field, expected in expected_row_hashes.items():
                if row.get(field) != expected:
                    raise BuildError(f"{task_id}: {field} mismatch")
            graph_ids = row.get("compact_input_ids")
            if (
                not isinstance(graph_ids, list)
                or not graph_ids
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in graph_ids
                )
            ):
                raise BuildError(f"{task_id}: invalid compact_input_ids")
            record = constants[task_id]
            preamble = constant_preamble(
                record.get("strings") or [], record.get("numbers") or []
            )
            encoded = tokenizer.encode(preamble)
            prefix_ids = [
                int(value)
                for value in (
                    encoded.ids if hasattr(encoded, "ids") else encoded
                )
                if int(value) < base_vocab_size
            ]
            if preamble and not prefix_ids:
                raise BuildError(
                    f"{task_id}: nonempty constants encoded to no base tokens"
                )
            if len(prefix_ids) + len(graph_ids) > max_source_tokens:
                raise BuildError(
                    f"{task_id}: lossless whole-binary constant prefix would "
                    "exceed the compact source contract"
                )
            row["compact_input_ids"] = prefix_ids + list(graph_ids)
            stats["rows"] += 1
            stats["enriched_rows"] += int(bool(prefix_ids))
            stats["constant_empty_rows"] += int(not prefix_ids)
            stats["constant_extraction_error_rows"] += int(
                record.get("err") not in (None, "")
            )
            stats["constant_prefix_tokens"] += len(prefix_ids)
            if len(prefix_ids) > stats["maximum_constant_prefix_tokens"]:
                stats["maximum_constant_prefix_tokens"] = len(prefix_ids)
                stats["maximum_constant_prefix_task_id"] = task_id
            stats["maximum_output_source_tokens"] = max(
                stats["maximum_output_source_tokens"],
                len(row["compact_input_ids"]),
            )
            output.append(row)
        return output

    enriched_train = enrich(train_rows)
    enriched_dev = enrich(dev_rows)
    output_train = args.output_train.expanduser().resolve()
    output_dev = args.output_dev.expanduser().resolve()
    output_train_seal = args.output_train_seal.expanduser().resolve()
    output_dev_seal = args.output_dev_seal.expanduser().resolve()
    report_path = args.build_report.expanduser().resolve()
    for path in (
        output_train,
        output_dev,
        output_train_seal,
        output_dev_seal,
        report_path,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    atomic_write_jsonl(output_train, enriched_train)
    atomic_write_jsonl(output_dev, enriched_dev)
    contract_sha = inputs["contract"]["sha256"]
    atomic_write_json(
        output_train_seal,
        {
            "schema": SEAL_SCHEMA,
            "selected_role": "fit",
            "output_sha256": sha256_file(output_train),
            "contract_sha256": contract_sha,
            "rows": len(enriched_train),
        },
    )
    atomic_write_json(
        output_dev_seal,
        {
            "schema": SEAL_SCHEMA,
            "selected_role": "measure",
            "output_sha256": sha256_file(output_dev),
            "contract_sha256": contract_sha,
            "rows": len(enriched_dev),
        },
    )
    report = {
        "schema": BUILD_SCHEMA,
        "constants_scope": args.constants_scope,
        "source_literals_read_for_enrichment": False,
        "lossless_constant_prefix": True,
        "constant_prefix_token_cap": None,
        "inputs": inputs,
        "counts": stats,
        "task_set_sha256": hashlib.sha256(
            json.dumps(
                sorted(task_ids), separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "outputs": {
            "train": file_record(output_train),
            "train_seal": file_record(output_train_seal),
            "dev": file_record(output_dev),
            "dev_seal": file_record(output_dev_seal),
        },
    }
    atomic_write_json(report_path, report)
    return report


def main() -> int:
    report = build(parse_args())
    print(
        "WHOLE_BINARY_COMPACT_BUILD "
        f"rows={report['counts']['rows']} "
        f"enriched={report['counts']['enriched_rows']} "
        f"max_prefix={report['counts']['maximum_constant_prefix_tokens']} "
        f"train_sha256={report['outputs']['train']['sha256']} "
        f"dev_sha256={report['outputs']['dev']['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
