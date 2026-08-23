"""Create a deterministic, source-grouped SFT train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def normalized_source(row: dict) -> str:
    source = str(row.get("source", row.get("dart_source", "")) or "")
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"\s+", " ", source).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train_output", required=True, type=Path)
    parser.add_argument("--validation_output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation_percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 1 <= args.validation_percent <= 50:
        raise SystemExit("--validation_percent must be between 1 and 50")

    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        fingerprint = hashlib.sha256(normalized_source(row).encode("utf-8")).hexdigest()
        groups[fingerprint].append(index)

    validation_groups = {
        fingerprint
        for fingerprint in groups
        if int(hashlib.sha256(f"{args.seed}|{fingerprint}".encode()).hexdigest()[:16], 16) % 100
        < args.validation_percent
    }
    train_indices = [i for fp, indices in groups.items() if fp not in validation_groups for i in indices]
    validation_indices = [i for fp, indices in groups.items() if fp in validation_groups for i in indices]

    # A hash partition can land a few rows away from the requested ratio. Keep
    # it deterministic and report the realized size rather than moving groups.
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text("\n".join(lines[i] for i in train_indices) + "\n", encoding="utf-8")
    args.validation_output.write_text("\n".join(lines[i] for i in validation_indices) + "\n", encoding="utf-8")
    manifest = {
        "source": str(args.input),
        "seed": args.seed,
        "rule": "SHA256(seed|normalized-source-SHA256) mod 100 < validation_percent",
        "validation_percent_requested": args.validation_percent,
        "rows_total": len(rows),
        "source_groups": len(groups),
        "duplicate_rows": len(rows) - len(groups),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "train_indices": sorted(train_indices),
        "validation_indices": sorted(validation_indices),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "rows_total", "source_groups", "duplicate_rows", "train_rows", "validation_rows"
    )}, indent=2))


if __name__ == "__main__":
    main()
