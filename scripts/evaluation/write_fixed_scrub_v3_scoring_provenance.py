"""Seal matched scorer/toolchain provenance for fixed-scrub-v3 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing provenance input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def nested(doc: dict[str, Any], *parts: str) -> Any:
    value: Any = doc
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["comparator", "neutral_exact", "name_only"])
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--inference_provenance", required=True, type=Path)
    parser.add_argument("--join_provenance", type=Path, default=None)
    parser.add_argument("--public_dataset", type=Path, default=None)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--dart_version_file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--row_count", type=int, default=154)
    parser.add_argument("--samples_per_row", type=int, default=10)
    args = parser.parse_args()

    inference = load_json(args.inference_provenance)
    checkpoint = file_record(args.checkpoint)
    recorded_checkpoint = nested(inference, "checkpoint", "sha256")
    if recorded_checkpoint != checkpoint["sha256"]:
        raise SystemExit("inference/checkpoint SHA-256 mismatch")
    if inference.get("scoring_tests_visible_to_policy") is not False:
        raise SystemExit("inference provenance does not prove hidden scoring tests")
    seed = inference.get("seed")
    prompt_schema = inference.get("prompt_schema_version")
    if seed is None or not isinstance(prompt_schema, str) or not prompt_schema:
        raise SystemExit("inference provenance lacks seed or prompt schema")

    dart_version = args.dart_version_file.read_text(encoding="utf-8").strip()
    if not dart_version.startswith("Dart SDK version: 3.11.5 (stable)"):
        raise SystemExit(f"unexpected Dart SDK: {dart_version!r}")
    scorer = file_record(args.scorer)
    if not re.fullmatch(r"[0-9a-f]{64}", scorer["sha256"]):
        raise SystemExit("malformed scorer hash")

    join_record = None
    public_record = None
    public_only = args.arm != "comparator"
    if public_only:
        if args.join_provenance is None or args.public_dataset is None:
            raise SystemExit("v3 arms require join provenance and public dataset")
        join = load_json(args.join_provenance)
        if join.get("policy_input_verified_public_only") is not True:
            raise SystemExit("join provenance does not verify public-only policy input")
        raw_dataset_hash = nested(inference, "dataset", "sha256")
        public_record = file_record(args.public_dataset)
        if raw_dataset_hash != public_record["sha256"]:
            raise SystemExit("inference/public dataset SHA-256 mismatch")
        join_record = file_record(args.join_provenance)

    payload = {
        "schema_version": "fixed-scrub-v3-scoring-provenance-v1",
        "arm": args.arm,
        "checkpoint": checkpoint,
        "generation_seed": seed,
        "compile_mode": "jit_tests",
        "scorer_sha256": scorer["sha256"],
        "dart_sdk_version": dart_version,
        "prompt_schema_version": prompt_schema,
        "scoring_tests_visible_to_policy": False,
        "policy_input_verified_public_only": public_only,
        "row_count": args.row_count,
        "samples_per_row": args.samples_per_row,
        "inputs": {
            "predictions": file_record(args.predictions),
            "stats": file_record(args.stats),
            "inference_provenance": file_record(args.inference_provenance),
            "join_provenance": join_record,
            "public_dataset": public_record,
            "scorer": scorer,
            "dart_version": file_record(args.dart_version_file),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
