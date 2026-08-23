#!/usr/bin/env python3
"""Attach hidden labels after signature-scrubbed inference.

Inference must run on the public dataset, which contains neither reference
source nor tests. This tool verifies that separation and joins the private
labels by task ID only after candidate generation has finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "signature-scrubbed-post-inference-join-v3"
PRIVATE_FIELDS = {
    "dart_source",
    "evaluation_only_dart_function_signature",
    "source",
    "tests",
}
SIGNATURE_MODES = ("name_only", "neutral_exact")
# Fingerprints of withheld secrets that must never ship in public rows built by
# sanitized protocol schemas (unsalted hashes are dictionary-recoverable).
PUBLIC_PROTOCOL_SECRET_KEYS = {"semantic_function_name_sha256", "original_source_sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def row_id(row: dict[str, Any], *, source: str) -> str:
    value = row.get("task_id", row.get("id"))
    if value is None or str(value).strip() == "":
        raise ValueError(f"{source}: missing task_id/id")
    return str(value)


def index_unique(rows: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row_id(row, source=source)
        if task_id in indexed:
            raise ValueError(f"{source}: duplicate task ID {task_id!r}")
        indexed[task_id] = row
    return indexed


def candidate_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = [str(row.get("id", "")), row.get("predictions", [])]
        digest.update(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def validate_public_row(
    row: dict[str, Any],
    task_id: str,
    *,
    signature_mode: str = "name_only",
    expected_target_name: str = "candidate",
    private_row: dict[str, Any] | None = None,
) -> None:
    if signature_mode not in SIGNATURE_MODES:
        raise ValueError(f"unsupported signature mode: {signature_mode!r}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expected_target_name):
        raise ValueError(
            f"expected target name is not a valid Dart identifier: "
            f"{expected_target_name!r}"
        )
    if expected_target_name == "main":
        raise ValueError("expected target name must not be main")
    leaked = sorted(PRIVATE_FIELDS.intersection(row))
    if leaked:
        raise ValueError(f"public dataset row {task_id!r} exposes private fields: {leaked}")
    if row.get("function") != expected_target_name:
        raise ValueError(
            f"public dataset row {task_id!r} function does not match expected "
            f"target {expected_target_name!r}"
        )
    if row.get("camel_case_function_name") != expected_target_name:
        raise ValueError(
            f"public dataset row {task_id!r} prompt name does not match expected "
            f"target {expected_target_name!r}"
        )
    target_call = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(expected_target_name)}\s*\("
    )
    if private_row is not None:
        for field in ("function", "camel_case_function_name"):
            if private_row.get(field) != expected_target_name:
                raise ValueError(
                    f"private dataset row {task_id!r} {field} does not match expected "
                    f"target {expected_target_name!r}"
                )
        hidden_signature = str(
            private_row.get("evaluation_only_dart_function_signature", "")
        ).strip()
        if not hidden_signature or not target_call.search(hidden_signature):
            raise ValueError(
                f"private dataset row {task_id!r} hidden signature does not target "
                f"{expected_target_name!r}"
            )
    signature = str(row.get("dart_function_signature", "")).strip()
    public_prompt_signature = str(row.get("public_prompt_signature", "")).strip()
    if signature_mode == "name_only":
        if row.get("prompt_signature_mode") != "name_only":
            raise ValueError(f"public dataset row {task_id!r} is not name-only")
        if signature or public_prompt_signature:
            raise ValueError(f"public dataset row {task_id!r} exposes an exact signature")
    else:  # neutral_exact
        if row.get("prompt_signature_mode") != "exact":
            raise ValueError(f"public dataset row {task_id!r} is not exact-mode")
        if not signature or not target_call.search(signature):
            raise ValueError(
                f"public dataset row {task_id!r} lacks a typed signature for "
                f"expected target {expected_target_name!r}"
            )
        expected = str((private_row or {}).get("public_prompt_signature", "")).strip()
        if not expected:
            raise ValueError(
                f"private dataset row {task_id!r} is missing public_prompt_signature"
            )
        if signature != expected:
            raise ValueError(
                f"public dataset row {task_id!r} signature does not match the sealed "
                f"public_prompt_signature"
            )
        if public_prompt_signature and public_prompt_signature != signature:
            raise ValueError(
                f"public dataset row {task_id!r} public_prompt_signature does not "
                f"match dart_function_signature"
            )
    protocol = row.get("benchmark_protocol")
    if isinstance(protocol, dict) and str(protocol.get("schema", "")).endswith(
        ("-v2", "-v3")
    ):
        exposed = sorted(PUBLIC_PROTOCOL_SECRET_KEYS.intersection(protocol))
        if exposed:
            raise ValueError(
                f"public dataset row {task_id!r} exposes fingerprint hashes: {exposed}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--public_dataset", type=Path, required=True)
    parser.add_argument("--private_dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_rows", type=int, default=154)
    parser.add_argument("--expected_samples", type=int, default=10)
    parser.add_argument(
        "--expected_signature_mode",
        default="name_only",
        choices=list(SIGNATURE_MODES),
    )
    parser.add_argument(
        "--expected_target_name",
        default="candidate",
        help="neutral Dart target name expected in public rows (default: candidate)",
    )
    args = parser.parse_args()

    raw_rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError("predictions must be a JSON array of objects")
    public_rows = load_jsonl(args.public_dataset)
    private_rows = load_jsonl(args.private_dataset)

    for label, rows in (
        ("predictions", raw_rows),
        ("public dataset", public_rows),
        ("private dataset", private_rows),
    ):
        if len(rows) != args.expected_rows:
            raise ValueError(f"{label}: expected {args.expected_rows} rows, found {len(rows)}")

    public_by_id = index_unique(public_rows, source="public dataset")
    private_by_id = index_unique(private_rows, source="private dataset")
    prediction_by_id = index_unique(raw_rows, source="predictions")
    expected_ids = set(public_by_id)
    if set(private_by_id) != expected_ids or set(prediction_by_id) != expected_ids:
        raise ValueError("public, private, and prediction task-ID sets are not identical")

    provenance_path = Path(str(args.predictions) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("row_count") != args.expected_rows:
        raise ValueError("inference provenance row count does not match")
    if provenance.get("scoring_tests_visible_to_policy") is not False:
        raise ValueError("inference provenance does not prove hidden-test separation")
    recorded_dataset = provenance.get("dataset", {}).get("sha256")
    if recorded_dataset != sha256_file(args.public_dataset):
        raise ValueError("inference provenance does not identify the public dataset")
    if provenance.get("generation", {}).get("num_samples") != args.expected_samples:
        raise ValueError("inference provenance sample count does not match")

    joined: list[dict[str, Any]] = []
    for raw in raw_rows:
        task_id = row_id(raw, source="predictions")
        public = public_by_id[task_id]
        private = private_by_id[task_id]
        validate_public_row(
            public,
            task_id,
            signature_mode=args.expected_signature_mode,
            expected_target_name=args.expected_target_name,
            private_row=private,
        )
        if raw.get("reference") not in (None, "") or raw.get("tests") not in (None, ""):
            raise ValueError(f"prediction row {task_id!r} already contains hidden labels")
        predictions = raw.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != args.expected_samples:
            raise ValueError(
                f"prediction row {task_id!r}: expected {args.expected_samples} candidates"
            )
        if str(raw.get("filename", "")) != str(public.get("filename", "")):
            raise ValueError(f"prediction row {task_id!r} filename does not match public data")
        reference = str(private.get("dart_source", ""))
        tests = str(private.get("tests", ""))
        if not reference.strip() or not tests.strip():
            raise ValueError(f"private dataset row {task_id!r} is missing source or tests")
        joined_row = dict(raw)
        joined_row["reference"] = reference
        joined_row["tests"] = tests
        joined.append(joined_row)

    before_digest = candidate_digest(raw_rows)
    after_digest = candidate_digest(joined)
    if before_digest != after_digest:
        raise RuntimeError("candidate stream changed during hidden-label join")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(joined, indent=2), encoding="utf-8")
    join_provenance = {
        "schema_version": SCHEMA_VERSION,
        "post_inference_join": True,
        "expected_signature_mode": args.expected_signature_mode,
        "expected_target_name": args.expected_target_name,
        "row_count": len(joined),
        "samples_per_row": args.expected_samples,
        "candidate_stream_sha256": before_digest,
        "policy_input_verified_public_only": True,
        "inputs": {
            "raw_predictions": file_record(args.predictions),
            "raw_inference_provenance": file_record(provenance_path),
            "public_dataset": file_record(args.public_dataset),
            "private_dataset": file_record(args.private_dataset),
        },
        "output": file_record(args.output),
        "source": file_record(Path(__file__)),
    }
    output_provenance = Path(str(args.output) + ".provenance.json")
    output_provenance.write_text(json.dumps(join_provenance, indent=2), encoding="utf-8")
    print(json.dumps(join_provenance, indent=2))


if __name__ == "__main__":
    main()
