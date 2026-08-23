#!/usr/bin/env python3
"""Validate the packaged ARM64 analysis bundle and emit a JSON report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()

    expected_hashes = {
        "datasets/flutter_train_graphv2.jsonl": "f21782dd60edc11988867659dd2d16a5f6b6d2f550594cae09ad7cf92b68dcb7",
        "datasets/flutter_eval_graphv2.jsonl": "864dc0bb7e9ee305ba0fc4be6e5d5ecbbeb7c17fd15bb3e41cfcc6d9aaf65fac",
        "source/scripts/run_arm64_graphv21_study.py": "31e378614ce07c01dfef24db3f4f3f077ce0d4a1c0165fb7777d17ce3a9a3ff6",
        "source/scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py": "6234f82bd4c64c29a561160374888d1bc6916af8d7d7368e30ba97cb5f237e13",
        "source/models/hierarchical_graph_encoder_antigravity.py": "0d9ed811fd3e2793d8d21a003ced9cff67cecfe65cf03a33bb611ec44af6f7db",
        "source/models/graphcodebert_tensor_builder.py": "a324bba2c3a642176404fba513c386e6952b4c6f04d26e9be01732ed36900ed5",
        "source/models/pyg_cfg_dataset.py": "5cf35b3c2d446e3e4444a833d6d0a39c6ddd366f2375a8b866706b4cd322edb3",
        "source/scripts/data/cfg_extractor.py": "daebbbfa7ac53fed9104e66396bc861bc837a8cea5a948548204d34439ee553c",
        "source/scripts/data/dfg_extractor.py": "603c052e8a79e7f6f689e97acdfc9c87245505b4fbf497bc2c49c2343fb0ed12",
    }
    hash_checks: dict[str, dict[str, Any]] = {}
    for relative, expected in expected_hashes.items():
        actual = sha256(bundle / relative)
        hash_checks[relative] = {
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
        }

    result_manifest = bundle / "results" / "arm64_regions16_s42.sha256"
    result_checks: dict[str, dict[str, Any]] = {}
    with result_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            expected, filename = line.strip().split(maxsplit=1)
            path = bundle / "results" / filename
            actual = sha256(path)
            result_checks[filename] = {
                "expected": expected,
                "actual": actual,
                "match": actual == expected,
            }

    json_files = sorted(bundle.rglob("*.json"))
    for path in json_files:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)

    jsonl_counts = {
        str(path.relative_to(bundle)).replace("\\", "/"): read_jsonl(path)
        for path in sorted(bundle.rglob("*.jsonl"))
    }
    if jsonl_counts["datasets/flutter_train_graphv2.jsonl"] != 1371:
        raise ValueError("Train JSONL row count is not 1,371")
    if jsonl_counts["datasets/flutter_eval_graphv2.jsonl"] != 343:
        raise ValueError("Eval JSONL row count is not 343")

    stem = (
        "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_"
        "arm64v21_s42_prefix_no_gine_regions16"
    )
    prediction_checks: dict[str, Any] = {}
    for kind in ("compile", "pass"):
        path = bundle / "results" / f"{stem}_{kind}_predictions.json"
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        prediction_checks[kind] = {
            "rows": len(rows),
            "all_have_10_candidates": all(
                len(row.get("predictions", [])) == 10 for row in rows
            ),
            "unique_ids": len({str(row.get("id")) for row in rows}),
        }
        if prediction_checks[kind] != {
            "rows": 343,
            "all_have_10_candidates": True,
            "unique_ids": 343,
        }:
            raise ValueError(f"Unexpected {kind} prediction shape: {prediction_checks[kind]}")

    csv_counts: dict[str, int] = {}
    for path in sorted(bundle.rglob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            csv_counts[str(path.relative_to(bundle)).replace("\\", "/")] = sum(
                1 for _ in csv.DictReader(handle)
            )

    all_hashes_match = all(item["match"] for item in hash_checks.values())
    all_results_match = all(item["match"] for item in result_checks.values())
    if not all_hashes_match or not all_results_match:
        raise ValueError("One or more packaged files do not match recorded SHA-256 values")

    report = {
        "schema_version": 1,
        "status": "passed",
        "bundle": bundle.name,
        "file_count_before_sha256_manifest": sum(
            1 for path in bundle.rglob("*") if path.is_file()
        ),
        "json_files_validated": len(json_files),
        "jsonl_counts": jsonl_counts,
        "csv_row_counts": csv_counts,
        "prediction_checks": prediction_checks,
        "recorded_hash_checks": hash_checks,
        "original_result_manifest_checks": result_checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "passed", "report": str(args.report)}))


if __name__ == "__main__":
    main()
