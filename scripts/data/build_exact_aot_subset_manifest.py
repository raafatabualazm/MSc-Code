#!/usr/bin/env python3
"""Build an exact AOT-manifest projection for sealed train/dev task IDs.

The selected AOT rows are copied without changing any field.  Model membership
is recorded separately because the 175 held-out tasks were carved from the
upstream pool's ``train`` split; rewriting the pool split would falsify the
binary-build provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AOT_ROW_SCHEMA = "phase0-s44-source-only-aot-row-v1"
MEMBERSHIP_SCHEMA = "dart-aot-multifunction-membership-v1"
REPORT_SCHEMA = "dart-aot-exact-subset-manifest-report-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TASK_ID_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")


class SubsetManifestError(ValueError):
    """Raised when an exact, leakage-safe projection cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SubsetManifestError(
                    f"blank_jsonl_line:{path}:{line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SubsetManifestError(
                    f"invalid_jsonl:{path}:{line_number}:{error}"
                ) from error
            if not isinstance(value, dict):
                raise SubsetManifestError(
                    f"non_object_jsonl_row:{path}:{line_number}"
                )
            rows.append(value)
    return rows


def write_jsonl_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(canonical_bytes(row).decode("ascii") + "\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    os.replace(temporary, path)


def require_sha256(path: Path, expected: str, label: str) -> str:
    expected = expected.lower()
    if not SHA256_RE.fullmatch(expected):
        raise SubsetManifestError(f"invalid_expected_sha256:{label}")
    observed = sha256_file(path)
    if observed != expected:
        raise SubsetManifestError(
            f"sha256_mismatch:{label}:{observed}!={expected}"
        )
    return observed


def _task_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_rows: int,
) -> list[str]:
    if len(rows) != expected_rows:
        raise SubsetManifestError(
            f"row_count_mismatch:{label}:{len(rows)}!={expected_rows}"
        )
    task_ids: list[str] = []
    seen: set[str] = set()
    for position, row in enumerate(rows):
        task_id = str(row.get("task_id") or "")
        if not TASK_ID_RE.fullmatch(task_id):
            raise SubsetManifestError(
                f"invalid_task_id:{label}:{position}:{task_id!r}"
            )
        if task_id in seen:
            raise SubsetManifestError(
                f"duplicate_task_id:{label}:{task_id}"
            )
        seen.add(task_id)
        task_ids.append(task_id)
    return task_ids


def build_projection(
    *,
    full_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    expected_train_rows: int,
    expected_dev_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return unchanged AOT rows and a separate model-membership ledger."""

    full_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    source_positions: set[tuple[str, int]] = set()
    for source_row, raw in enumerate(full_rows):
        row = dict(raw)
        if row.get("schema") != AOT_ROW_SCHEMA:
            raise SubsetManifestError(
                f"full_manifest_schema_mismatch:{source_row}"
            )
        task_id = str(row.get("task_id") or "")
        if not TASK_ID_RE.fullmatch(task_id):
            raise SubsetManifestError(
                f"invalid_full_manifest_task_id:{source_row}:{task_id!r}"
            )
        if task_id in full_by_id:
            raise SubsetManifestError(
                f"duplicate_full_manifest_task_id:{task_id}"
            )
        source_key = (str(row.get("split") or ""), int(row["split_row"]))
        if source_key in source_positions:
            raise SubsetManifestError(
                f"duplicate_full_manifest_split_position:{source_key}"
            )
        source_positions.add(source_key)
        full_by_id[task_id] = (source_row, row)

    train_ids = _task_ids(
        train_rows, label="train", expected_rows=expected_train_rows
    )
    dev_ids = _task_ids(
        dev_rows, label="dev", expected_rows=expected_dev_rows
    )
    overlap = sorted(set(train_ids) & set(dev_ids))
    if overlap:
        raise SubsetManifestError(
            f"train_dev_task_overlap:{overlap[:10]}"
        )
    missing = [
        task_id
        for task_id in [*train_ids, *dev_ids]
        if task_id not in full_by_id
    ]
    if missing:
        raise SubsetManifestError(
            f"tasks_missing_from_full_manifest:{missing[:10]}"
        )

    selected: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    for model_role, task_ids in (("train", train_ids), ("dev", dev_ids)):
        for model_row, task_id in enumerate(task_ids):
            source_row, aot_row = full_by_id[task_id]
            selected.append(dict(aot_row))
            membership.append(
                {
                    "schema": MEMBERSHIP_SCHEMA,
                    "task_id": task_id,
                    "model_role": model_role,
                    "model_row": model_row,
                    "source_manifest_row": source_row,
                    "source_split": str(aot_row["split"]),
                    "source_split_row": int(aot_row["split_row"]),
                    "source_aot_row_sha256": canonical_sha256(aot_row),
                }
            )
    if len(selected) != expected_train_rows + expected_dev_rows:
        raise SubsetManifestError("internal_selected_row_count_mismatch")
    if any(
        canonical_sha256(row)
        != membership_row["source_aot_row_sha256"]
        for row, membership_row in zip(selected, membership)
    ):
        raise SubsetManifestError("selected_row_mutation_detected")
    return selected, membership


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--full-manifest-sha256", required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl-sha256", required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl-sha256", required=True)
    parser.add_argument("--expected-train-rows", type=int, default=1580)
    parser.add_argument("--expected-dev-rows", type=int, default=175)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--membership-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.expected_train_rows <= 0 or args.expected_dev_rows <= 0:
        raise SubsetManifestError("expected_rows_must_be_positive")
    full_manifest = args.full_manifest.resolve()
    train_jsonl = args.train_jsonl.resolve()
    dev_jsonl = args.dev_jsonl.resolve()
    input_hashes = {
        "full_manifest": require_sha256(
            full_manifest,
            args.full_manifest_sha256,
            "full_manifest",
        ),
        "train": require_sha256(
            train_jsonl, args.train_jsonl_sha256, "train"
        ),
        "dev": require_sha256(dev_jsonl, args.dev_jsonl_sha256, "dev"),
    }
    selected, membership = build_projection(
        full_rows=read_jsonl(full_manifest),
        train_rows=read_jsonl(train_jsonl),
        dev_rows=read_jsonl(dev_jsonl),
        expected_train_rows=args.expected_train_rows,
        expected_dev_rows=args.expected_dev_rows,
    )
    output_jsonl = args.output_jsonl.resolve()
    membership_jsonl = args.membership_jsonl.resolve()
    write_jsonl_atomic(output_jsonl, selected)
    write_jsonl_atomic(membership_jsonl, membership)
    report = {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "ordering": "model_train_then_model_dev",
        "aot_rows_copied_without_field_changes": True,
        "model_membership_recorded_separately": True,
        "train_rows": args.expected_train_rows,
        "dev_rows": args.expected_dev_rows,
        "total_rows": len(selected),
        "train_dev_disjoint": True,
        "input": {
            "full_manifest": {
                "path": str(full_manifest),
                "sha256": input_hashes["full_manifest"],
            },
            "train": {
                "path": str(train_jsonl),
                "sha256": input_hashes["train"],
            },
            "dev": {
                "path": str(dev_jsonl),
                "sha256": input_hashes["dev"],
            },
        },
        "output": {
            "manifest": {
                "path": str(output_jsonl),
                "sha256": sha256_file(output_jsonl),
            },
            "membership": {
                "path": str(membership_jsonl),
                "sha256": sha256_file(membership_jsonl),
            },
        },
        "membership_sequence_sha256": canonical_sha256(membership),
    }
    write_json_atomic(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
