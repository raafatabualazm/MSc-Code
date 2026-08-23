#!/usr/bin/env python3
"""Private diagnostic for source-neutral AOT external-symbol policy.

This tool deliberately records raw GDB annotations and therefore MUST NOT feed
model inputs, release artifacts, API prompts, or public reports.  It is used to
distinguish missing user-function bodies from Dart SDK/runtime annotations when
hardening the production extractor's frozen policy.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import threading
from pathlib import Path
from typing import Any

from scripts.data import extract_dart_aot_user_function_bundle as extractor


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--aot-manifest", type=Path, required=True)
    parser.add_argument("--aot-root", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path, required=True)
    parser.add_argument("--output-private-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gdb", default="/usr/bin/gdb")
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = _args()
    manifest_rows = extractor.read_jsonl(args.aot_manifest.resolve())
    requested = {
        str(row["task_id"])
        for row in extractor.read_jsonl(args.failures_jsonl.resolve())
    }
    selected = [
        row
        for row in manifest_rows
        if str(row.get("task_id") or "") in requested
    ]
    if len(selected) != len(requested):
        raise SystemExit(
            f"failure_task_manifest_mismatch:{len(selected)}!={len(requested)}"
        )
    work = extractor._validate_manifest(selected, args.aot_root.resolve())
    original = extractor._classify_external_annotation
    lock = threading.Lock()
    unknown: collections.Counter[str] = collections.Counter()
    tasks_by_label: dict[str, set[str]] = collections.defaultdict(set)
    local = threading.local()

    def auditing_classifier(
        annotation: str | None,
        *,
        trusted_runtime_symbols: set[str],
        known_nonruntime_symbols: set[str],
    ) -> tuple[str | None, str] | None:
        try:
            return original(
                annotation,
                trusted_runtime_symbols=trusted_runtime_symbols,
                known_nonruntime_symbols=known_nonruntime_symbols,
            )
        except extractor.UserFunctionExtractionError as error:
            if str(error) != "possible_same_program_external_function_not_selected":
                raise
            label = str(annotation)
            with lock:
                unknown[label] += 1
                tasks_by_label[label].add(str(local.task_id))
            return None, "private_audit_neutralized_unknown"

    extractor._classify_external_annotation = auditing_classifier
    errors: list[dict[str, str]] = []

    def inspect(row: dict[str, Any]) -> None:
        local.task_id = row["task_id"]
        extractor.extract_aot(
            task_id=str(row["task_id"]),
            aot_path=Path(row["aot_path"]),
            gdb=str(args.gdb),
            root_symbol="candidate",
            timeout=float(args.timeout),
            split=str(row["split"]),
            split_row=int(row["split_row"]),
            source_only_contract=row["source_only_contract"],
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(inspect, row): row["task_id"] for row in work
        }
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                future.result()
            except Exception as error:
                errors.append(
                    {
                        "task_id": str(task_id),
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )
    extractor._classify_external_annotation = original
    payload = {
        "private": True,
        "must_not_feed_models_or_apis": True,
        "requested_tasks": len(requested),
        "audited_tasks": len(work) - len(errors),
        "audit_errors": errors,
        "unknown_labels": [
            {
                "label": label,
                "occurrences": count,
                "task_count": len(tasks_by_label[label]),
            }
            for label, count in sorted(
                unknown.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }
    args.output_private_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_private_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_private_json.chmod(0o600)
    print(
        json.dumps(
            {
                "requested_tasks": payload["requested_tasks"],
                "audited_tasks": payload["audited_tasks"],
                "audit_errors": len(errors),
                "unknown_label_count": len(payload["unknown_labels"]),
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
