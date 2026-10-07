#!/usr/bin/env python3
"""Read-only exhaustive F2 round-trip audit for the sealed 1,755-task pool.

The audit imports the deployed builder, extractor, F2 codec, and tokenizer,
then exercises every bundle without invoking the publishing build entry point.
Only an audit journal and summary are written.  Build inputs and outputs are
opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED = {
    "builder": "bb954a0b5aafe5fa51c97cce40d25c80dd6f65f3fa2696d289f31dcbdf4fae66",
    "extractor": "ff3cd323eb3045da0a9bf8b3489f8e867cbf3ae3dfd12181fc6c2004423af2a5",
    "f2": "097a7fac3fcc8b07106c7ea326efd0ee9f880622c781f113e57cf8657e2241ce",
    "tokenizer": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "bundles": "d2a019fe14e500bf1d242367e3b52b644f3e166bb8e3b5ad47e980e6ccb688d2",
    "constants": "2b5dc0d353e5f7cb70bb79cb398406b16a92b524e4252fdbca01bd48a7c7b857",
}
EXPECTED_ROWS = 1755
API_PROMPT_TOKEN_LIMIT = 12000
CHAT_OVERHEAD_RESERVE = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def import_file(path: Path, label: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"f2_audit_{label}_{sha256_file(path)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {label}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"{path}:{line_number}: row is not an object"
                )
            rows.append(value)
    return rows


def value_fingerprint(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "type": "str",
            "unicode_scalars": len(value),
            "utf8_bytes": len(value.encode("utf-8")),
            "sha256": sha256_text(value),
        }
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "keys": len(value),
            "canonical_sha256": canonical_sha256(value),
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return {
            "type": "array",
            "items": len(value),
            "canonical_sha256": canonical_sha256(value),
        }
    return {"type": type(value).__name__, "value": value}


def string_mismatch(expected: str, actual: str) -> dict[str, Any]:
    prefix = 0
    common = min(len(expected), len(actual))
    while prefix < common and expected[prefix] == actual[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < common - prefix
        and expected[len(expected) - 1 - suffix]
        == actual[len(actual) - 1 - suffix]
    ):
        suffix += 1
    expected_scalar = (
        None if prefix >= len(expected) else ord(expected[prefix])
    )
    actual_scalar = None if prefix >= len(actual) else ord(actual[prefix])
    return {
        "first_differing_scalar_index": prefix,
        "first_differing_utf8_byte_offset": len(
            expected[:prefix].encode("utf-8")
        ),
        "common_suffix_scalars": suffix,
        "expected_scalar_codepoint": expected_scalar,
        "actual_scalar_codepoint": actual_scalar,
        "expected": value_fingerprint(expected),
        "actual": value_fingerprint(actual),
    }


def deep_differences(
    expected: Any, actual: Any, path: str = "$"
) -> list[dict[str, Any]]:
    if type(expected) is not type(actual):
        return [
            {
                "path": path,
                "kind": "type",
                "expected": value_fingerprint(expected),
                "actual": value_fingerprint(actual),
            }
        ]
    if isinstance(expected, Mapping):
        differences: list[dict[str, Any]] = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            differences.append(
                {
                    "path": f"{path}.{key}",
                    "kind": "missing_actual_key",
                    "expected": value_fingerprint(expected[key]),
                }
            )
        for key in sorted(actual_keys - expected_keys):
            differences.append(
                {
                    "path": f"{path}.{key}",
                    "kind": "unexpected_actual_key",
                    "actual": value_fingerprint(actual[key]),
                }
            )
        for key in sorted(expected_keys & actual_keys):
            differences.extend(
                deep_differences(
                    expected[key], actual[key], f"{path}.{key}"
                )
            )
        return differences
    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        differences = []
        if len(expected) != len(actual):
            differences.append(
                {
                    "path": path,
                    "kind": "array_length",
                    "expected_items": len(expected),
                    "actual_items": len(actual),
                }
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            differences.extend(
                deep_differences(
                    expected_item, actual_item, f"{path}[{index}]"
                )
            )
        return differences
    if isinstance(expected, str) and expected != actual:
        return [
            {
                "path": path,
                "kind": "string",
                **string_mismatch(expected, actual),
            }
        ]
    if expected != actual:
        return [
            {
                "path": path,
                "kind": "scalar",
                "expected": value_fingerprint(expected),
                "actual": value_fingerprint(actual),
            }
        ]
    return []


def normalized_f2_expected(canonical: Mapping[str, Any]) -> dict[str, Any]:
    blocks = list(canonical.get("blocks") or [])
    return {
        "architecture": str(canonical.get("architecture") or ""),
        "entry_blocks": [
            int(value) for value in canonical.get("entry_blocks") or []
        ],
        "blocks": [
            {
                "id": block_id,
                "instructions": [
                    str(value) for value in block.get("instructions") or []
                ],
            }
            for block_id, block in enumerate(blocks)
        ],
        "cfg_edges": [
            {
                "source": int(edge["source"]),
                "target": int(edge["target"]),
                "edge_type": str(edge["edge_type"]),
            }
            for edge in canonical.get("cfg_edges") or []
        ],
    }


def add_failure(
    failures: list[dict[str, Any]], stage: str, **details: Any
) -> None:
    failures.append({"stage": stage, **details})


def write_journal_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--f2", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "builder": args.builder.resolve(),
        "extractor": args.extractor.resolve(),
        "f2": args.f2.resolve(),
        "tokenizer": args.tokenizer.resolve(),
        "bundles": args.bundles.resolve(),
        "constants": args.constants.resolve(),
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(
            "deployed input hash mismatch: "
            + json.dumps(
                {
                    key: {"expected": EXPECTED[key], "actual": observed[key]}
                    for key in EXPECTED
                    if observed.get(key) != EXPECTED[key]
                },
                sort_keys=True,
            )
        )

    builder = import_file(paths["builder"], "builder")
    extractor = import_file(paths["extractor"], "extractor")
    f2 = import_file(paths["f2"], "f2")
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError("tokenizers package is required") from exc
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))

    bundle_rows = load_jsonl(paths["bundles"])
    constant_rows = load_jsonl(paths["constants"])
    if len(bundle_rows) != EXPECTED_ROWS or len(constant_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_ROWS} bundles/constants, got "
            f"{len(bundle_rows)}/{len(constant_rows)}"
        )
    bundle_ids = [str(row.get("task_id") or "") for row in bundle_rows]
    constant_by_id = {
        str(row.get("task_id") or ""): row for row in constant_rows
    }
    if (
        any(not task_id for task_id in bundle_ids)
        or len(set(bundle_ids)) != EXPECTED_ROWS
        or len(constant_by_id) != EXPECTED_ROWS
        or set(bundle_ids) != set(constant_by_id)
    ):
        raise RuntimeError("bundle/constants task-set equality failed")

    # This is semantically identical to serialize_f2's default discovery but
    # avoids repeating a full tokenizer-vocabulary scan 1,755 times.
    visible_symbols = f2.visible_one_token_symbols(tokenizer)

    original_decode_f2 = f2.decode_f2
    decode_capture: dict[str, Any] = {}

    def capturing_decode(text: str) -> tuple[str, dict[str, Any]]:
        decode_capture.clear()
        decode_capture["text_sha256"] = sha256_text(text)
        decode_capture["text_utf8_bytes"] = len(text.encode("utf-8"))
        try:
            result = original_decode_f2(text)
        except Exception as exc:
            decode_capture["exception_type"] = type(exc).__name__
            decode_capture["exception"] = str(exc)
            raise
        decode_capture["result"] = result
        return result

    f2.decode_f2 = capturing_decode
    system_prompt_tokens = len(
        builder._tokenizer_encode(tokenizer, str(f2.F2_SYSTEM_PROMPT))
    )

    if args.journal.exists() or args.summary.exists():
        raise FileExistsError(
            "refusing to overwrite an existing audit journal/summary"
        )
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    failed_task_ids: list[str] = []
    failure_stage_counts: dict[str, int] = {}
    maximum_api_tokens = -1
    maximum_api_task_id = ""
    passed = 0
    with args.journal.open("x", encoding="utf-8", newline="\n") as journal:
        for index, bundle in enumerate(bundle_rows, 1):
            task_id = str(bundle["task_id"])
            failures: list[dict[str, Any]] = []
            canonical: dict[str, Any] | None = None
            semantic_projection: dict[str, Any] | None = None
            try:
                canonical, semantic_projection = (
                    builder.combine_user_function_bundle(bundle, extractor)
                )
            except Exception as exc:
                add_failure(
                    failures,
                    "combine_user_function_bundle",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                )

            if canonical is not None and semantic_projection is not None:
                constants = constant_by_id[task_id]
                external_symbols = semantic_projection["external_symbols"]
                try:
                    prefix_source = builder.binary_enrichment_preamble(
                        constants["strings"],
                        constants["numbers"],
                        external_symbols,
                    )
                    prefix_ids = builder._tokenizer_encode(
                        tokenizer, prefix_source
                    )
                    prefix_text = builder._tokenizer_decode(
                        tokenizer, prefix_ids
                    )
                    if prefix_text != prefix_source:
                        add_failure(
                            failures,
                            "prefix_tokenizer_byte_roundtrip",
                            **string_mismatch(prefix_source, prefix_text),
                        )
                    reencoded_prefix_ids = builder._tokenizer_encode(
                        tokenizer, prefix_text
                    )
                    if reencoded_prefix_ids != prefix_ids:
                        add_failure(
                            failures,
                            "prefix_token_id_roundtrip",
                            expected_count=len(prefix_ids),
                            actual_count=len(reencoded_prefix_ids),
                            expected_sha256=canonical_sha256(prefix_ids),
                            actual_sha256=canonical_sha256(
                                reencoded_prefix_ids
                            ),
                        )
                    try:
                        parsed_external = (
                            builder.parse_external_dictionary_from_preamble(
                                prefix_text
                            )
                        )
                    except Exception as exc:
                        add_failure(
                            failures,
                            "external_dictionary_parse",
                            exception_type=type(exc).__name__,
                            exception=str(exc),
                        )
                    else:
                        if parsed_external != external_symbols:
                            add_failure(
                                failures,
                                "external_dictionary_roundtrip",
                                differences=deep_differences(
                                    external_symbols,
                                    parsed_external,
                                ),
                            )

                    decode_capture.clear()
                    f2_text: str | None = None
                    try:
                        f2_text = f2.serialize_f2(
                            prefix_text,
                            canonical,
                            tokenizer=tokenizer,
                            visible_symbols=visible_symbols,
                        )
                    except Exception as exc:
                        details: dict[str, Any] = {
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                        }
                        captured_result = decode_capture.get("result")
                        if captured_result is not None:
                            decoded_prefix, decoded_canonical = captured_result
                            expected_canonical = normalized_f2_expected(
                                canonical
                            )
                            if decoded_prefix != prefix_text:
                                details["prefix_difference"] = (
                                    string_mismatch(
                                        prefix_text, decoded_prefix
                                    )
                                )
                            if decoded_canonical != expected_canonical:
                                details["canonical_differences"] = (
                                    deep_differences(
                                        expected_canonical,
                                        decoded_canonical,
                                    )
                                )
                            details["f2_text_sha256"] = decode_capture.get(
                                "text_sha256"
                            )
                            details["f2_text_utf8_bytes"] = (
                                decode_capture.get("text_utf8_bytes")
                            )
                        elif "exception" in decode_capture:
                            details["decode_exception_type"] = (
                                decode_capture.get("exception_type")
                            )
                            details["decode_exception"] = decode_capture.get(
                                "exception"
                            )
                        add_failure(
                            failures,
                            "f2_serialize_internal_roundtrip",
                            **details,
                        )
                    if f2_text is not None:
                        try:
                            decoded_prefix, decoded_canonical = (
                                original_decode_f2(f2_text)
                            )
                        except Exception as exc:
                            add_failure(
                                failures,
                                "f2_external_decode",
                                exception_type=type(exc).__name__,
                                exception=str(exc),
                            )
                        else:
                            if decoded_prefix != prefix_text:
                                add_failure(
                                    failures,
                                    "f2_prefix_external_roundtrip",
                                    **string_mismatch(
                                        prefix_text, decoded_prefix
                                    ),
                                )
                            expected_canonical = normalized_f2_expected(
                                canonical
                            )
                            if decoded_canonical != expected_canonical:
                                add_failure(
                                    failures,
                                    "f2_canonical_external_roundtrip",
                                    differences=deep_differences(
                                        expected_canonical,
                                        decoded_canonical,
                                    ),
                                )
                        user_tokens = len(
                            builder._tokenizer_encode(tokenizer, f2_text)
                        )
                        api_tokens = (
                            system_prompt_tokens
                            + user_tokens
                            + CHAT_OVERHEAD_RESERVE
                        )
                        if api_tokens > maximum_api_tokens:
                            maximum_api_tokens = api_tokens
                            maximum_api_task_id = task_id
                        if api_tokens > API_PROMPT_TOKEN_LIMIT:
                            add_failure(
                                failures,
                                "f2_api_prompt_budget",
                                api_tokens=api_tokens,
                                limit=API_PROMPT_TOKEN_LIMIT,
                                system_tokens=system_prompt_tokens,
                                user_tokens=user_tokens,
                                overhead_reserve=CHAT_OVERHEAD_RESERVE,
                            )
                except Exception as exc:
                    add_failure(
                        failures,
                        "unexpected_task_audit_exception",
                        exception_type=type(exc).__name__,
                        exception=str(exc),
                    )

            row = {
                "schema": "f2-roundtrip-audit-task-v1",
                "index": index,
                "task_id": task_id,
                "passed": not failures,
                "failures": failures,
            }
            write_journal_line(journal, row)
            if failures:
                failed_task_ids.append(task_id)
                for failure in failures:
                    stage = str(failure["stage"])
                    failure_stage_counts[stage] = (
                        failure_stage_counts.get(stage, 0) + 1
                    )
            else:
                passed += 1
            if index % args.progress_every == 0 or index == EXPECTED_ROWS:
                print(
                    "F2_AUDIT_PROGRESS "
                    f"processed={index}/{EXPECTED_ROWS} "
                    f"passed={passed} failed={len(failed_task_ids)}",
                    flush=True,
                )

    summary = {
        "schema": "f2-roundtrip-audit-summary-v1",
        "read_only_build_artifact_audit": True,
        "inputs": {
            name: {"path": str(paths[name]), "sha256": observed[name]}
            for name in paths
        },
        "counts": {
            "expected_tasks": EXPECTED_ROWS,
            "processed_tasks": EXPECTED_ROWS,
            "passed_tasks": passed,
            "failed_tasks": len(failed_task_ids),
            "failure_stage_counts": failure_stage_counts,
        },
        "failed_task_ids": failed_task_ids,
        "token_budget": {
            "limit": API_PROMPT_TOKEN_LIMIT,
            "chat_overhead_reserve": CHAT_OVERHEAD_RESERVE,
            "system_prompt_tokens": system_prompt_tokens,
            "maximum_api_tokens": maximum_api_tokens,
            "maximum_api_task_id": maximum_api_task_id,
        },
        "method": {
            "all_failures_collected": True,
            "f2_visible_symbol_pool_precomputed_once": True,
            "f2_visible_symbol_pool_count": len(visible_symbols),
            "full_canonical_differences_redacted_to_hashes_and_codepoints": True,
        },
        "journal": {
            "path": str(args.journal.resolve()),
            "sha256": sha256_file(args.journal),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.summary.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "F2_AUDIT_COMPLETE "
        f"passed={passed} failed={len(failed_task_ids)} "
        f"summary={args.summary}",
        flush=True,
    )
    return 0 if not failed_task_ids else 3


if __name__ == "__main__":
    raise SystemExit(main())
