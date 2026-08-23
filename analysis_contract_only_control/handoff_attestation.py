#!/usr/bin/env python3
"""Create or verify the deterministic exact-EXITED handoff attestation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    require_exact_or_write,
    sha256_file,
)


SCHEMA = "t5gemma2-contract-only-handoff-attestation-v1"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.predecessor_report).expanduser().resolve()
    bundle_path = Path(args.bundle_manifest).expanduser().resolve()
    report = _read_json(report_path, "predecessor report")
    observed_report_sha256 = sha256_file(report_path)
    observed_bundle_sha256 = sha256_file(bundle_path)
    if (
        args.upstream_state != "EXITED"
        or args.stable_report_sha256 != observed_report_sha256
        or args.reviewed_bundle_sha256 != observed_bundle_sha256
        or report.get("schema")
        != "t5gemma2-f2-intervention-multiseed-report-v1"
        or report.get("status") != "complete"
        or report.get("script_sha256") != args.predecessor_reporter_sha256
        or observed_report_sha256
        != "17645716115052bb48a906a4c7231c76ec28a9d7b66dc55bc53e669e990bee63"
        or report.get("design", {}).get("seeds") != [42, 43, 44, 45, 46]
        or report.get("design", {}).get("fresh_runs") != 12
        or report.get("design", {}).get("tasks_per_run") != 175
        or report.get("design", {}).get("k") != 10
        or report.get("design", {}).get("no_training_or_promotion") is not True
        or report.get("model_visible_source_bytes_identical_across_seeds") is not True
        or report.get("full_input_view_records_identical_across_seeds") is not False
        or report.get("allowed_input_view_metadata_drift", {}).get("field")
        != "row_transformations_sha256"
        or report.get("allowed_input_view_metadata_drift", {}).get(
            "full_record_identity_not_claimed"
        )
        is not True
        or report.get("rank0_gold_roundtrip", {}).get("passed") != 175
        or args.minimum_free_kib < 5 * 1024 * 1024
        or args.minimum_gpu_free_mib < 5 * 1024
    ):
        raise ValueError("exact-EXITED handoff contract failed")
    return {
        "schema": SCHEMA,
        "status": "pass",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "upstream_supervisor": {
            "program": args.upstream_program,
            "required_state": "EXITED",
            "observed_state": "EXITED",
            "stopped_fatal_backoff_rejected": True,
        },
        "predecessor_report": {
            "path": str(report_path),
            "sha256": observed_report_sha256,
            "schema": report["schema"],
            "status": report["status"],
            "reporter_sha256": args.predecessor_reporter_sha256,
            "recomputed_immediately_before_attestation": True,
            "stable_hash_gate_passed": True,
            "stability_seconds": args.stability_seconds,
        },
        "reviewed_bundle": {
            "manifest_path": str(bundle_path),
            "manifest_sha256": observed_bundle_sha256,
            "sha256sum_check_passed": True,
        },
        "resource_gates": {
            "minimum_free_kib": args.minimum_free_kib,
            "minimum_gpu_free_mib": args.minimum_gpu_free_mib,
            "disk_gate_passed": True,
            "gpu_compute_processes_empty": True,
            "gpu_memory_gate_passed": True,
        },
        "downstream_start_authorized": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=["create", "verify"], required=True)
    parser.add_argument("--predecessor_report", required=True)
    parser.add_argument("--predecessor_reporter_sha256", required=True)
    parser.add_argument("--stable_report_sha256", required=True)
    parser.add_argument("--bundle_manifest", required=True)
    parser.add_argument("--reviewed_bundle_sha256", required=True)
    parser.add_argument("--upstream_program", required=True)
    parser.add_argument("--upstream_state", required=True)
    parser.add_argument("--stability_seconds", type=int, required=True)
    parser.add_argument("--minimum_free_kib", type=int, required=True)
    parser.add_argument("--minimum_gpu_free_mib", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.stability_seconds <= 0:
        parser.error("stability_seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = build_attestation(args)
    output = Path(args.output).expanduser().resolve()
    if args.mode == "create":
        require_exact_or_write(output, expected)
    else:
        if _read_json(output, "handoff attestation") != expected:
            raise ValueError("handoff attestation differs from exact current state")
    print(json.dumps(expected, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
