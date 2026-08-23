#!/usr/bin/env python3
"""Merged Qwen status with adopted-capacity outcome reconciliation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import frontier_passk as runner
import qwen37_capacity_adopted_outcome_reconciliation_v1 as adopted
import qwen37_capacity_fallback_v6 as capacity
import qwen37_capacity_length_repair_v7 as capacity_length
import qwen37_primary_effective_status_v7 as parent


SCHEMA = "qwen37-primary-effective-status-v8"
EXPECTED_PARENT_STATUS_SHA256 = (
    "f8efd980737e4efc9a41c521c12ec62a69fa098f8de698021a840861c03cb1b2"
)
EXPECTED_RECONCILIATION_CONTRACT_SHA256 = adopted.EXPECTED_CONTRACT_SHA256
EXPECTED_RECONCILIATION_ENTRY_SHA256 = (
    "a5bc531c71652d0fb881d748863ad91c08f8eb155905250226dc0507346e7457"
)


class AuditError(RuntimeError):
    pass


def validate_reconciliation_dependencies(patch: Path) -> None:
    if (
        runner.sha256_file(
            patch / "qwen37_capacity_adopted_outcome_reconciliation_v1.py"
        )
        != EXPECTED_RECONCILIATION_ENTRY_SHA256
        or runner.sha256_file(patch / adopted.CONTRACT_NAME)
        != EXPECTED_RECONCILIATION_CONTRACT_SHA256
    ):
        raise AuditError("v8 reconciliation dependency hash mismatch")


def _binding_sha(
    *,
    effective: Mapping[str, Any],
    outcome: Mapping[str, Any],
    reconciled: bool,
) -> str:
    if not reconciled:
        return capacity.canonical_sha(outcome)
    source = outcome.get("source_outcome_provenance")
    if (
        effective.get("origin") != "adopted_clean_diagnostic"
        or outcome.get("outcome_origin")
        != "adopted_clean_diagnostic_reconciliation_v1"
        or not isinstance(source, Mapping)
        or outcome.get("legacy_effective_outcome_binding_sha256")
        != source.get("canonical_row_sha256")
    ):
        raise AuditError("invalid reconciled adopted-outcome binding")
    return str(source.get("canonical_row_sha256") or "")


def capacity_outputs(
    workspace: Path,
    *,
    expected_contract_sha256: str,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    set[str],
    dict[str, int],
    list[dict[str, Any]],
]:
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    terminals: dict[tuple[str, str, int], dict[str, Any]] = {}
    outcomes: dict[tuple[str, str, int], dict[str, Any]] = {}
    response_ids: set[str] = set()
    usage_totals: dict[str, int] = defaultdict(int)
    progress: list[dict[str, Any]] = []
    for partition, arm, directory in parent.CAPACITY_OUTPUTS:
        root = run_root / directory
        effective_rows = capacity._load_effective(
            root / "effective_terminals.jsonl"
        )
        feed_rows = capacity._load_terminal_feed(
            root / "effective_terminal_feed.jsonl"
        )
        raw_outcomes = capacity.read_jsonl(root / "outcomes.jsonl")
        base_by_selection: dict[str, dict[str, Any]] = {}
        for row in raw_outcomes:
            selection_id = str(row.get("selection_id") or "")
            if not selection_id or selection_id in base_by_selection:
                raise AuditError("duplicate capacity outcome selection")
            base_by_selection[selection_id] = row
        reconciled = adopted.load_effective_outcomes(root)
        missing = set(effective_rows).difference(base_by_selection)
        if set(reconciled) != missing:
            raise AuditError(
                "adopted reconciliation does not exactly cover missing "
                f"capacity outcomes for {partition}/{arm}"
            )
        if set(base_by_selection).intersection(reconciled):
            raise AuditError("base/reconciled capacity outcomes overlap")
        outcome_by_selection = {**base_by_selection, **reconciled}
        if set(effective_rows) != set(feed_rows):
            raise AuditError("capacity effective/feed selection mismatch")
        if set(effective_rows) != set(outcome_by_selection):
            raise AuditError("capacity effective/outcome selection mismatch")
        for selection_id, effective in effective_rows.items():
            feed = feed_rows[selection_id]
            outcome = outcome_by_selection[selection_id]
            is_reconciled = selection_id in reconciled
            if (
                effective.get("schema") != capacity.SCHEMA
                or effective.get("record_type")
                != "effective_capacity_terminal"
                or feed.get("schema") != capacity.SCHEMA
                or feed.get("record_type")
                != "capacity_effective_terminal_feed"
                or outcome.get("schema") != capacity.SCHEMA
                or outcome.get("record_type")
                != "capacity_candidate_outcome"
                or effective.get("overlay_contract_sha256")
                != expected_contract_sha256
                or feed.get("overlay_contract_sha256")
                != expected_contract_sha256
                or effective.get("arm") != arm
                or outcome.get("arm") != arm
                or effective.get("task_id") != feed.get("task_id")
                or effective.get("task_id") != outcome.get("task_id")
                or effective.get("global_sample_index")
                != feed.get("global_sample_index")
                or effective.get("global_sample_index")
                != outcome.get("global_sample_index")
                or effective.get("response_id") != feed.get("response_id")
                or effective.get("response_id") != outcome.get("response_id")
                or effective.get("effective_attempt_id")
                != outcome.get("attempt_id")
                or effective.get("finish_reason") != feed.get("finish_reason")
                or effective.get("finish_reason")
                != outcome.get("finish_reason")
                or effective.get("candidate_valid")
                != feed.get("candidate_valid")
                or effective.get("candidate_valid")
                != outcome.get("candidate_valid")
                or effective.get("code_sha256") != feed.get("code_sha256")
                or effective.get("code_sha256")
                != outcome.get("code_sha256")
                or effective.get("usage") != feed.get("validated_usage")
                or effective.get("canonical_terminal_row_sha256")
                != feed.get("effective_terminal_canonical_row_sha256")
                or effective.get("canonical_outcome_row_sha256")
                != _binding_sha(
                    effective=effective,
                    outcome=outcome,
                    reconciled=is_reconciled,
                )
                or effective.get("compiled") != outcome.get("compiled")
                or effective.get("passed") != outcome.get("passed")
            ):
                raise AuditError("capacity effective/feed/outcome mismatch")
            capacity_length.validate_feed_terminal(
                feed,
                expected_capacity_contract_sha256=(
                    expected_contract_sha256
                ),
            )
            parent.validate_outcome(
                outcome,
                candidate_valid=bool(effective["candidate_valid"]),
                evaluator_sha256=parent.primary.EXPECTED_EVALUATOR_SHA256,
            )
            response_id = str(effective.get("response_id") or "")
            if not response_id or response_id in response_ids:
                raise AuditError("duplicate capacity response ID")
            response_ids.add(response_id)
            usage = effective.get("usage")
            if not isinstance(usage, Mapping):
                raise AuditError("capacity effective terminal lacks usage")
            parent.add_usage(usage_totals, usage, cap=12_298)
            key = (
                arm,
                str(effective["task_id"]),
                int(effective["global_sample_index"]),
            )
            if key in terminals:
                raise AuditError("duplicate capacity global terminal")
            terminals[key] = effective
            outcomes[key] = outcome
        status_path = root / "status.json"
        status_row = (
            capacity.read_json(status_path) if status_path.is_file() else {}
        )
        progress.append(
            {
                "source": "capacity_v6_plus_adopted_reconciliation_v1",
                "partition": partition,
                "arm": arm,
                "root": str(root),
                "status": status_row.get("status", "not_started"),
                "effective_terminals": len(effective_rows),
                "base_outcomes": len(base_by_selection),
                "reconciled_adopted_outcomes": len(reconciled),
                "outcomes": len(outcome_by_selection),
                "length_terminals": sum(
                    row.get("finish_reason") == "length"
                    for row in feed_rows.values()
                ),
                "remaining": status_row.get("remaining"),
            }
        )
    return (
        terminals,
        outcomes,
        response_ids,
        dict(usage_totals),
        progress,
    )


def aggregate(
    workspace: Path,
    *,
    expected_capacity_contract_sha256: str,
    expected_capacity_script_sha256: str,
    expected_capacity_length_script_sha256: str,
) -> dict[str, Any]:
    patch = workspace / "frontier_ceiling_patch_v1"
    validate_reconciliation_dependencies(patch)
    if (
        runner.sha256_file(patch / "qwen37_primary_effective_status_v7.py")
        != EXPECTED_PARENT_STATUS_SHA256
    ):
        raise AuditError("v8 status parent/reconciliation hash mismatch")
    original_capacity_outputs = parent.capacity_outputs
    parent.capacity_outputs = capacity_outputs
    try:
        report = parent.aggregate(
            workspace,
            expected_capacity_contract_sha256=(
                expected_capacity_contract_sha256
            ),
            expected_capacity_script_sha256=(
                expected_capacity_script_sha256
            ),
            expected_capacity_length_script_sha256=(
                expected_capacity_length_script_sha256
            ),
        )
    finally:
        parent.capacity_outputs = original_capacity_outputs
    report["schema"] = SCHEMA
    report["adopted_capacity_outcome_reconciliation"] = {
        "contract_sha256": EXPECTED_RECONCILIATION_CONTRACT_SHA256,
        "provider_imports": False,
        "provider_calls": 0,
        "source_journals_modified": False,
        "base_capacity_journals_modified": False,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--expected-capacity-contract-sha256", required=True)
    parser.add_argument("--expected-capacity-script-sha256", required=True)
    parser.add_argument("--expected-capacity-length-script-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    for name in (
        "expected_capacity_contract_sha256",
        "expected_capacity_script_sha256",
        "expected_capacity_length_script_sha256",
    ):
        value = str(getattr(args, name)).strip().lower()
        if len(value) != 64 or any(
            ch not in "0123456789abcdef" for ch in value
        ):
            parser.error(f"--{name.replace('_', '-')} must be SHA-256")
        setattr(args, name, value)
    return args


def main() -> int:
    args = parse_args()
    try:
        report = aggregate(
            args.workspace.resolve(),
            expected_capacity_contract_sha256=(
                args.expected_capacity_contract_sha256
            ),
            expected_capacity_script_sha256=(
                args.expected_capacity_script_sha256
            ),
            expected_capacity_length_script_sha256=(
                args.expected_capacity_length_script_sha256
            ),
        )
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if args.out:
            runner.atomic_write_json(args.out.resolve(), report)
        print(json.dumps(report, sort_keys=True))
        return 2
    if args.out:
        runner.atomic_write_json(args.out.resolve(), report)
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
