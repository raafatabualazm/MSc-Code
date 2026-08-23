#!/usr/bin/env python3
"""Project a fixed-scrub-v3 prediction artifact to a candidate-only pool.

The standalone compile diagnostic must not receive tests, references, typed
signatures, filenames, graph metadata, or semantic task IDs.  This projection
retains the 154x10 candidate stream verbatim and replaces each row identity
with a deterministic opaque identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "fixed-scrub-v3-standalone-pool-projection-v1"
EXPECTED_ROWS = 154
EXPECTED_SAMPLES = 10
POOL_ROW_KEYS = {"id", "predictions"}
OPAQUE_ID_RE = re.compile(r"standalone_row_[0-9]{4}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def candidate_stream_sha256(candidate_rows: Sequence[Sequence[str]]) -> str:
    """Hash ordered candidate strings without task metadata."""
    canonical = json.dumps(
        [list(row) for row in candidate_rows],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_prediction_candidate_rows(path: Path) -> list[list[str]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read predictions JSON {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("predictions must be a JSON array")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"predictions: expected {EXPECTED_ROWS} rows, found {len(rows)}"
        )

    candidate_rows: list[list[str]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"prediction row {index}: expected an object")
        candidates = row.get("predictions")
        if not isinstance(candidates, list) or len(candidates) != EXPECTED_SAMPLES:
            found = len(candidates) if isinstance(candidates, list) else "invalid"
            raise ValueError(
                f"prediction row {index}: expected {EXPECTED_SAMPLES} candidates, "
                f"found {found}"
            )
        if not all(isinstance(candidate, str) for candidate in candidates):
            raise ValueError(f"prediction row {index}: every candidate must be a string")
        candidate_rows.append(list(candidates))
    return candidate_rows


def load_candidate_pool(path: Path) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidate pool {path}: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        found = len(rows) if isinstance(rows, list) else "invalid"
        raise ValueError(
            f"candidate pool: expected {EXPECTED_ROWS} rows, found {found}"
        )
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != POOL_ROW_KEYS:
            keys = sorted(row) if isinstance(row, dict) else "invalid"
            raise ValueError(
                f"candidate pool row {index}: expected only {sorted(POOL_ROW_KEYS)}, "
                f"found {keys}"
            )
        expected_id = f"standalone_row_{index:04d}"
        if row.get("id") != expected_id or not OPAQUE_ID_RE.fullmatch(str(row.get("id"))):
            raise ValueError(f"candidate pool row {index}: non-opaque or reordered ID")
        candidates = row.get("predictions")
        if not isinstance(candidates, list) or len(candidates) != EXPECTED_SAMPLES:
            raise ValueError(
                f"candidate pool row {index}: expected {EXPECTED_SAMPLES} candidates"
            )
        if not all(isinstance(candidate, str) for candidate in candidates):
            raise ValueError(f"candidate pool row {index}: non-string candidate")
    return rows


def project_predictions(
    predictions: Path,
    output: Path,
    provenance_output: Path,
) -> dict[str, Any]:
    resolved = {
        predictions.resolve(),
        output.resolve(),
        provenance_output.resolve(),
    }
    if len(resolved) != 3:
        raise ValueError("predictions, output, and provenance output must be distinct files")

    source_candidates = load_prediction_candidate_rows(predictions)
    stream_digest = candidate_stream_sha256(source_candidates)
    pool = [
        {
            "id": f"standalone_row_{index:04d}",
            "predictions": list(candidates),
        }
        for index, candidates in enumerate(source_candidates, 1)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Re-read the exact artifact that the scorer will consume.  Equality and a
    # second digest prevent serialization or accidental projection drift.
    written_pool = load_candidate_pool(output)
    written_candidates = [row["predictions"] for row in written_pool]
    if written_candidates != source_candidates:
        raise RuntimeError("candidate stream changed during standalone projection")
    if candidate_stream_sha256(written_candidates) != stream_digest:
        raise RuntimeError("candidate stream digest changed during standalone projection")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_role": "secondary_standalone_compile",
        "row_count": EXPECTED_ROWS,
        "samples_per_row": EXPECTED_SAMPLES,
        "candidate_count": EXPECTED_ROWS * EXPECTED_SAMPLES,
        "candidate_stream_sha256": stream_digest,
        "candidate_stream_preserved": True,
        "pool_is_candidate_only": True,
        "pool_row_keys": sorted(POOL_ROW_KEYS),
        "opaque_id_scheme": "standalone_row_{one_based_index:04d}",
        "semantic_identifiers_preserved": False,
        "hidden_evaluation_metadata_preserved": False,
        "inputs": {"predictions": file_record(predictions)},
        "output": {"candidate_pool": file_record(output)},
        "projector": file_record(Path(__file__)),
    }
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance_output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance_output = args.provenance_output or Path(str(args.output) + ".provenance.json")
    payload = project_predictions(args.predictions, args.output, provenance_output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
