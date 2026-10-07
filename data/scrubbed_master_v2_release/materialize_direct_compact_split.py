#!/usr/bin/env python3
"""Materialize leakage-minimal codec inputs and labels from a frozen split.

The split sidecars contain original 1-based release line numbers.  This tool
verifies every binding before emitting graph-only public codec inputs and
four-field private labels.  It deliberately does not perform compression or
fit a codebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_SPLIT_DIR = HERE / "direct_compact_split_v1"
SCHEMA = "direct-compact-split-materialization-v1"

MASTER_PUBLIC = HERE / "master_dart_graphv2_signature_scrubbed_public.jsonl"
MASTER_PRIVATE = HERE / "master_dart_graphv2_signature_scrubbed_private.jsonl"
MASTER_LEDGER = HERE / "master_dart_graphv2_compile_ledger.jsonl"

PUBLIC_KEYS = ("task_id", "lang", "function", "cfg", "edges", "integrity")
PRIVATE_KEYS = ("task_id", "lang", "function", "dart_source")
FORBIDDEN_PUBLIC_KEYS = {
    "dart_source",
    "tests",
    "evaluation_only_dart_function_signature",
    "dart_function_signature",
    "benchmark_protocol",
}
FORBIDDEN_PRIVATE_KEYS = {
    "assembly",
    "cfg",
    "edges",
    "integrity",
    "graph_v2",
    "tests",
    "dart_function_signature",
    "evaluation_only_dart_function_signature",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            result.append(value)
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
            count += 1
    return count


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_bound_file(
    path: Path, binding: dict[str, Any], expected_rows: int | None = None
) -> None:
    expected_sha = str(binding.get("sha256", ""))
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"SHA mismatch for {path}: expected {expected_sha}, got {actual_sha}"
        )
    if expected_rows is not None and int(binding.get("rows", -1)) != expected_rows:
        raise ValueError(
            f"row binding mismatch for {path}: manifest={binding.get('rows')} "
            f"actual={expected_rows}"
        )


def verify_master_rows(
    public: list[dict[str, Any]],
    private: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> None:
    if not (len(public) == len(private) == len(ledger)):
        raise ValueError(
            "master row count mismatch: "
            f"public={len(public)} private={len(private)} ledger={len(ledger)}"
        )
    seen: set[str] = set()
    for line_number, (pub, priv, led) in enumerate(zip(public, private, ledger), 1):
        pub_id = pub.get("task_id")
        priv_id = priv.get("task_id")
        ledger_id = led.get("neutral_id")
        if not isinstance(pub_id, str) or not pub_id:
            raise ValueError(f"master line {line_number}: missing public task_id")
        if pub_id != priv_id or pub_id != ledger_id:
            raise ValueError(
                f"master line {line_number}: task identity mismatch "
                f"public={pub_id!r} private={priv_id!r} ledger={ledger_id!r}"
            )
        if pub_id in seen:
            raise ValueError(f"master line {line_number}: duplicate task_id {pub_id}")
        seen.add(pub_id)
        for label, row in (("public", pub), ("private", priv)):
            if row.get("function") != "candidate":
                raise ValueError(
                    f"master line {line_number}: non-neutral {label} function"
                )
            if row.get("dart_function_signature") != "":
                raise ValueError(
                    f"master line {line_number}: exposed {label} signature"
                )
            if row.get("prompt_signature_mode") != "name_only":
                raise ValueError(
                    f"master line {line_number}: unexpected {label} prompt mode"
                )
        if pub.get("lang") != priv.get("lang"):
            raise ValueError(f"master line {line_number}: language mismatch")
        for key in ("assembly", "cfg", "edges", "integrity", "graph_v2"):
            if pub.get(key) != priv.get(key):
                raise ValueError(
                    f"master line {line_number}: public/private {key} mismatch"
                )
        if not isinstance(priv.get("dart_source"), str) or not priv["dart_source"].strip():
            raise ValueError(f"master line {line_number}: empty dart_source")


def verify_alignment(
    name: str, rows: list[dict[str, Any]], master_count: int
) -> list[int]:
    original_lines: list[int] = []
    allowed = {"original_line", "semantic_group", "split_line"}
    for expected_split_line, row in enumerate(rows, 1):
        if set(row) != allowed:
            raise ValueError(
                f"{name} split line {expected_split_line}: unexpected keys {sorted(row)}"
            )
        if row.get("split_line") != expected_split_line:
            raise ValueError(
                f"{name} split line {expected_split_line}: non-contiguous split_line"
            )
        original_line = row.get("original_line")
        semantic_group = row.get("semantic_group")
        if not isinstance(original_line, int) or not 1 <= original_line <= master_count:
            raise ValueError(
                f"{name} split line {expected_split_line}: invalid original_line"
            )
        if not isinstance(semantic_group, int) or semantic_group < 1:
            raise ValueError(
                f"{name} split line {expected_split_line}: invalid semantic_group"
            )
        original_lines.append(original_line)
    if len(original_lines) != len(set(original_lines)):
        raise ValueError(f"{name}: duplicate original_line")
    return original_lines


def public_codec_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: row[key] for key in PUBLIC_KEYS}
    if set(result) & FORBIDDEN_PUBLIC_KEYS:
        raise AssertionError("public codec projection contains a forbidden key")
    if set(result) != set(PUBLIC_KEYS):
        raise AssertionError("public codec projection schema drift")
    return result


def private_label_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: row[key] for key in PRIVATE_KEYS}
    if set(result) & FORBIDDEN_PRIVATE_KEYS:
        raise AssertionError("private label projection contains a forbidden key")
    if set(result) != set(PRIVATE_KEYS):
        raise AssertionError("private label projection schema drift")
    return result


def describe(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": relative(path),
        "rows": rows,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--public", type=Path, default=MASTER_PUBLIC)
    parser.add_argument("--private", type=Path, default=MASTER_PRIVATE)
    parser.add_argument("--ledger", type=Path, default=MASTER_LEDGER)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_dir = args.split_dir.resolve()
    split_manifest_path = split_dir / "split_manifest.json"
    train_alignment_path = split_dir / "train_private_alignment.jsonl"
    dev_alignment_path = split_dir / "dev_private_alignment.jsonl"
    destinations = {
        "train_public_codec.jsonl": split_dir / "train_public_codec.jsonl",
        "dev_public_codec.jsonl": split_dir / "dev_public_codec.jsonl",
        "train_private_labels.jsonl": split_dir / "train_private_labels.jsonl",
        "dev_private_labels.jsonl": split_dir / "dev_private_labels.jsonl",
        "materialization_manifest.json": split_dir / "materialization_manifest.json",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing materialization; pass --force: "
            + ", ".join(existing)
        )

    split_manifest = load_json(split_manifest_path)
    if split_manifest.get("schema") != "direct-compact-split-preparation-v1":
        raise ValueError("unsupported split manifest schema")
    public = load_jsonl(args.public)
    private = load_jsonl(args.private)
    ledger = load_jsonl(args.ledger)

    inputs = split_manifest.get("inputs", {})
    verify_bound_file(args.public, inputs.get("public", {}), len(public))
    verify_bound_file(args.private, inputs.get("private", {}), len(private))
    verify_bound_file(args.ledger, inputs.get("ledger", {}), len(ledger))
    verify_master_rows(public, private, ledger)

    train_alignment = load_jsonl(train_alignment_path)
    dev_alignment = load_jsonl(dev_alignment_path)
    split_outputs = split_manifest.get("outputs", {})
    verify_bound_file(
        train_alignment_path,
        split_outputs.get("train_private_alignment.jsonl", {}),
        len(train_alignment),
    )
    verify_bound_file(
        dev_alignment_path,
        split_outputs.get("dev_private_alignment.jsonl", {}),
        len(dev_alignment),
    )
    train_lines = verify_alignment("train", train_alignment, len(public))
    dev_lines = verify_alignment("dev", dev_alignment, len(public))
    if set(train_lines) & set(dev_lines):
        raise ValueError("train/dev original-line overlap")
    train_groups = {int(row["semantic_group"]) for row in train_alignment}
    dev_groups = {int(row["semantic_group"]) for row in dev_alignment}
    if train_groups & dev_groups:
        raise ValueError("train/dev semantic-group overlap")

    projected: dict[str, list[dict[str, Any]]] = {
        "train_public_codec.jsonl": [public_codec_row(public[n - 1]) for n in train_lines],
        "dev_public_codec.jsonl": [public_codec_row(public[n - 1]) for n in dev_lines],
        "train_private_labels.jsonl": [private_label_row(private[n - 1]) for n in train_lines],
        "dev_private_labels.jsonl": [private_label_row(private[n - 1]) for n in dev_lines],
    }
    for split in ("train", "dev"):
        pub_rows = projected[f"{split}_public_codec.jsonl"]
        label_rows = projected[f"{split}_private_labels.jsonl"]
        if [row["task_id"] for row in pub_rows] != [row["task_id"] for row in label_rows]:
            raise ValueError(f"{split}: materialized public/private identity mismatch")

    split_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".materialize-", dir=split_dir))
    try:
        counts: dict[str, int] = {}
        for name, rows_out in projected.items():
            counts[name] = write_jsonl(temp_dir / name, rows_out)
        manifest = {
            "schema": SCHEMA,
            "deterministic": True,
            "projection": {
                "public_codec_keys": list(PUBLIC_KEYS),
                "private_label_keys": list(PRIVATE_KEYS),
                "public_contains_target_source": False,
                "private_contains_tests_signatures_or_assembly": False,
                "compression_or_codebook_fit_performed": False,
            },
            "verification": {
                "master_rows": len(public),
                "public_private_ledger_ids_equal": True,
                "candidate_neutrality_verified": True,
                "public_private_binary_fields_equal": True,
                "train_dev_original_lines_disjoint": True,
                "train_dev_semantic_groups_disjoint": True,
            },
            "inputs": {
                "split_manifest": {
                    "path": relative(split_manifest_path),
                    "sha256": sha256_file(split_manifest_path),
                },
                "public": describe(args.public, len(public)),
                "private": describe(args.private, len(private)),
                "ledger": describe(args.ledger, len(ledger)),
                "train_alignment": describe(train_alignment_path, len(train_alignment)),
                "dev_alignment": describe(dev_alignment_path, len(dev_alignment)),
            },
            "script": {
                "path": relative(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "outputs": {
                name: describe(temp_dir / name, counts[name])
                for name in sorted(projected)
            },
        }
        # Paths in the manifest describe their final, not temporary, locations.
        for name, item in manifest["outputs"].items():
            item["path"] = relative(destinations[name])
        write_json(temp_dir / "materialization_manifest.json", manifest)
        for name, destination in destinations.items():
            os.replace(temp_dir / name, destination)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
