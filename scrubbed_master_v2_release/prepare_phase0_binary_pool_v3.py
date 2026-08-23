#!/usr/bin/env python3
"""Prepare sealed, private source-only AOT build inputs for compact-Qwen v3."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.dart_source_only_aot import source_only_program


DEFAULT_V2 = ROOT / "scrubbed_master_v2_release/direct_compact_phase0_s44_v2"
DEFAULT_OUT = ROOT / "scrubbed_master_v2_release/direct_compact_phase0_s44_pool_v3"
SCHEMA = "phase0-s44-binary-pool-v3-source-preparation-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank_jsonl_row:{path}:{number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non_object_jsonl_row:{path}:{number}")
            result.append(value)
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def prepare_split(v2: Path, out: Path, split: str) -> dict[str, Any]:
    codec_path = v2 / f"prepared/{split}_codec_private.jsonl"
    labels_path = v2 / f"prepared/{split}_private_labels.jsonl"
    codecs = load_jsonl(codec_path)
    labels = load_jsonl(labels_path)
    if len(codecs) != len(labels):
        raise ValueError(f"{split}:codec_label_count_mismatch")

    build_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    removed_main = 0
    for index, (codec, label) in enumerate(zip(codecs, labels, strict=True)):
        task_id = str(codec.get("task_id") or "")
        if not task_id or task_id != str(label.get("task_id") or ""):
            raise ValueError(f"{split}:{index}:task_id_alignment_mismatch")
        if str(label.get("function") or "") != "candidate":
            raise ValueError(f"{split}:{index}:target_not_candidate")
        function_source, program, metadata = source_only_program(
            str(label.get("dart_source") or "")
        )
        removed_main += metadata["removed_top_level_main"]
        private_metadata = codec.get("compact_private_metadata") or {}
        if not isinstance(private_metadata, dict):
            raise ValueError(f"{split}:{index}:invalid_compact_private_metadata")
        build_rows.append(
            {
                "schema": "dart-source-only-aot-build-input-v1",
                "role": "fit" if split == "train" else "measure",
                "split": split,
                "split_row": index,
                "task_id": task_id,
                "function": "candidate",
                "function_source": function_source,
                "function_source_sha256": sha256_text(function_source),
                "analysis_program": program,
                "analysis_program_sha256": sha256_text(program),
                "transform_metadata": metadata,
                "source_symbols": metadata["source_symbols"],
                "family": private_metadata.get("family"),
                "source_pool": private_metadata.get("source_pool"),
                "phase0_manifest_line": private_metadata.get("phase0_manifest_line"),
                "compact_private_metadata": private_metadata,
            }
        )
        label_rows.append(
            {
                "dart_source": function_source,
                "family": label.get("family"),
                "function": "candidate",
                "lang": "dart",
                "task_id": task_id,
            }
        )

    build_path = out / f"private_build_inputs/{split}.jsonl"
    output_labels = out / f"prepared/{split}_private_labels.jsonl"
    write_jsonl(build_path, build_rows)
    write_jsonl(output_labels, label_rows)
    return {
        "split": split,
        "rows": len(build_rows),
        "removed_top_level_main": removed_main,
        "input_codec": {"path": str(codec_path), "sha256": sha256_file(codec_path)},
        "input_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        "private_build_inputs": {
            "path": str(build_path.relative_to(out)),
            "sha256": sha256_file(build_path),
        },
        "private_labels": {
            "path": str(output_labels.relative_to(out)),
            "sha256": sha256_file(output_labels),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--v2-release", type=Path, default=DEFAULT_V2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    v2 = args.v2_release.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    splits = [prepare_split(v2, out, split) for split in ("train", "dev")]
    manifest = {
        "schema": SCHEMA,
        "transformer": {
            "path": "scripts/data/dart_source_only_aot.py",
            "sha256": sha256_file(ROOT / "scripts/data/dart_source_only_aot.py"),
        },
        "producer": {
            "path": "scrubbed_master_v2_release/prepare_phase0_binary_pool_v3.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_release_manifest_sha256": sha256_file(v2 / "release_manifest.json"),
        "splits": splits,
        "gates": {
            "all_rows_preserved": sum(item["rows"] for item in splits) == 3277,
            "phase0_train_rows": splits[0]["rows"] == 2951,
            "phase0_dev_rows": splits[1]["rows"] == 326,
            "all_topup_demo_mains_removed": sum(
                item["removed_top_level_main"] for item in splits
            )
            == 1105,
        },
    }
    if not all(manifest["gates"].values()):
        raise SystemExit(json.dumps(manifest, indent=2))
    manifest_path = out / "source_preparation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    names = [
        "private_build_inputs/train.jsonl",
        "private_build_inputs/dev.jsonl",
        "prepared/train_private_labels.jsonl",
        "prepared/dev_private_labels.jsonl",
        "source_preparation_manifest.json",
    ]
    (out / "SOURCE_SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(out / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
