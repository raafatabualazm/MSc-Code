#!/usr/bin/env python3
"""Provider-free monitor for adopted Qwen capacity outcome gaps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import frontier_passk as runner
import qwen37_capacity_adopted_outcome_reconciliation_v1 as reconciliation


SCHEMA = "qwen37-capacity-adopted-outcome-monitor-v1"
EXTENSION_NAME = "qwen37_capacity_adopted_outcome_extension_v1.json"
EXPECTED_RECONCILIATION_ENTRY_SHA256 = (
    "a5bc531c71652d0fb881d748863ad91c08f8eb155905250226dc0507346e7457"
)
EXPECTED_STATUS_ENTRY_SHA256 = (
    "0b972aee9000af129394f7c02162d6e704e7bd2c3b138ccf809a0f124b5f7734"
)


def validate_dependencies(workspace: Path) -> None:
    patch = workspace.resolve() / "frontier_ceiling_patch_v1"
    entry_path = (
        patch / "qwen37_capacity_adopted_outcome_reconciliation_v1.py"
    )
    contract_path = patch / reconciliation.CONTRACT_NAME
    status_path = patch / "qwen37_primary_effective_status_v8.py"
    extension_path = patch / EXTENSION_NAME
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    if not isinstance(extension, dict):
        raise reconciliation.AuditError(
            "adopted-outcome extension is not an object"
        )
    monitor = extension.get("monitor")
    reconciler = extension.get("reconciliation")
    effective_status = extension.get("effective_status")
    if (
        runner.sha256_file(entry_path)
        != EXPECTED_RECONCILIATION_ENTRY_SHA256
        or runner.sha256_file(contract_path)
        != reconciliation.EXPECTED_CONTRACT_SHA256
        or not isinstance(monitor, dict)
        or not isinstance(reconciler, dict)
        or not isinstance(effective_status, dict)
        or extension.get("schema")
        != "qwen37-capacity-adopted-outcome-extension-v1"
        or reconciler.get("entry_sha256")
        != EXPECTED_RECONCILIATION_ENTRY_SHA256
        or reconciler.get("contract_sha256")
        != reconciliation.EXPECTED_CONTRACT_SHA256
        or monitor.get("entry_sha256")
        != runner.sha256_file(Path(__file__).resolve())
        or effective_status.get("entry_sha256")
        != EXPECTED_STATUS_ENTRY_SHA256
        or runner.sha256_file(status_path)
        != EXPECTED_STATUS_ENTRY_SHA256
    ):
        raise reconciliation.AuditError(
            "adopted-outcome monitor dependency/extension mismatch"
        )


def run_once(workspace: Path) -> dict[str, Any]:
    validate_dependencies(workspace)
    report = reconciliation.reconcile_all(workspace.resolve())
    status = {
        "schema": SCHEMA,
        "status": "healthy",
        "provider_imports": False,
        "provider_calls": 0,
        "last_reconciliation": report,
        "updated_at": runner.utc_now(),
    }
    output = (
        workspace.resolve()
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "qwen37_capacity_adopted_outcome_reconciliation_v1"
        / "monitor_status.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    runner.atomic_write_json(output, status)
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 5 or args.interval_seconds > 300:
        parser.error("--interval-seconds must be in [5, 300]")
    return args


def main() -> int:
    args = parse_args()
    try:
        while True:
            run_once(args.workspace.resolve())
            if args.once:
                return 0
            time.sleep(args.interval_seconds)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider_imports": False,
                    "provider_calls": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
