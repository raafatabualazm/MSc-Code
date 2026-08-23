#!/usr/bin/env python3
"""Provider-free reconciliation for adopted Qwen capacity outcomes.

The sealed v6 capacity runner selected clean-diagnostic terminals without
looking at their execution outcomes, but its adopted branch wrote the
effective terminal without projecting the already-existing diagnostic
outcome into the capacity outcome schema.  This program repairs only that
bookkeeping omission in a separate append-only overlay.  It deliberately
does not import an API SDK or any provider entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import frontier_passk as runner


SCHEMA = "qwen37-capacity-adopted-outcome-reconciliation-v1"
CAPACITY_SCHEMA = "qwen37-capacity-fallback-v6"
CONTRACT_NAME = (
    "qwen37_capacity_adopted_outcome_reconciliation_contract_v1.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "344ac01c949f76e351cb1040d018d075875dcfbc2b08bc16483f9a28661c2b74"
)
EXPECTED_CAPACITY_ENTRY_SHA256 = (
    "6b2d642be25bb7b2e97daddf70e9a2245a8ff09e5f0c6e5e32c09afc92159521"
)
EXPECTED_CAPACITY_CONTRACT_SHA256 = (
    "cea8acaa785ddc2685a5da8b4426dce41837a25af2e7dd9639dd70f632d59631"
)
EXPECTED_EVALUATOR_SHA256 = (
    "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6"
)
EXPECTED_DIAGNOSTIC_SCHEMA = "audited-frontier-passk-v2"
ARMS = ("opus", "codex")
PARTITIONS: dict[str, dict[str, Any]] = {
    "0520": {
        "capacity_indices": (0, 1, 2, 3, 4),
        "diagnostic_indices": (0, 1, 2),
        "diagnostic_model": "qwen3.7-max-2026-05-20",
        "diagnostic_directory": (
            "qwen37_clean_v4_0520_{arm}_k3_mc12k_tol10_tb8k"
        ),
        "local_index_offset": 0,
    },
    "0608": {
        "capacity_indices": (5, 6, 7, 8, 9),
        "diagnostic_indices": (5, 6),
        "diagnostic_model": "qwen3.7-max-2026-06-08",
        "diagnostic_directory": (
            "qwen37_clean_v4_0608_{arm}_k2_mc12k_tol10_tb8k"
        ),
        "local_index_offset": 5,
    },
}
CAPACITY_DIRECTORIES = tuple(
    (
        partition,
        arm,
        f"qwen37_capacity_v6_{partition}_{arm}_mc12k_tb8k",
    )
    for partition in ("0520", "0608")
    for arm in ARMS
)
OVERLAY_DIRECTORY = "adopted_outcome_reconciliation_v1"
OUTCOME_FILE = "effective_outcomes.jsonl"


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise AuditError(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Mapping[str, Any]) -> str:
    return runner.stable_sha256(dict(value))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _unique_by(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = str(row.get(field) or "")
        if not key or key in result:
            raise AuditError(f"duplicate/missing {label} {field}")
        result[key] = row
    return result


def _validate_target(
    row: Mapping[str, Any],
    *,
    arm: str,
    partition: str,
) -> None:
    immutable = dict(row)
    observed = immutable.pop("selection_record_sha256", None)
    if (
        row.get("schema") != CAPACITY_SCHEMA
        or row.get("record_type") != "capacity_target"
        or row.get("arm") != arm
        or int(row.get("global_sample_index", -1))
        not in PARTITIONS[partition]["capacity_indices"]
        or observed != canonical_sha(immutable)
        or row.get("selection_reads_outcomes") is not False
    ):
        raise AuditError("capacity target identity/integrity mismatch")


def _load_capacity_context(
    capacity_root: Path,
    *,
    arm: str,
    partition: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    provenance = read_json(capacity_root / "provenance.json")
    config = provenance.get("config")
    if (
        not isinstance(config, dict)
        or provenance.get("config_sha256") != canonical_sha(config)
        or config.get("schema") != CAPACITY_SCHEMA
        or config.get("arm") != arm
        or config.get("partition") != partition
        or Path(str(config.get("out") or "")).resolve()
        != capacity_root.resolve()
        or config.get("contract_sha256")
        != EXPECTED_CAPACITY_CONTRACT_SHA256
        or config.get("runtime_identity", {}).get(
            "capacity_runner_sha256"
        )
        != EXPECTED_CAPACITY_ENTRY_SHA256
    ):
        raise AuditError("capacity provenance/config binding mismatch")
    target_path = capacity_root / "targets.jsonl"
    if config.get("targets_sha256") != sha256_file(target_path):
        raise AuditError("capacity targets SHA mismatch")
    targets = _unique_by(
        read_jsonl(target_path), "selection_id", label="capacity target"
    )
    for target in targets.values():
        _validate_target(target, arm=arm, partition=partition)
    effective = _unique_by(
        read_jsonl(capacity_root / "effective_terminals.jsonl"),
        "selection_id",
        label="effective terminal",
    )
    base_outcomes = _unique_by(
        read_jsonl(capacity_root / "outcomes.jsonl"),
        "selection_id",
        label="capacity outcome",
    )
    if not set(effective).issubset(targets):
        raise AuditError("effective terminal is not target-backed")
    if not set(base_outcomes).issubset(effective):
        raise AuditError("base capacity outcome is not effective-backed")
    return targets, effective, base_outcomes


def _diagnostic_source(
    workspace: Path,
    *,
    target: Mapping[str, Any],
    effective: Mapping[str, Any],
    arm: str,
    partition: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    spec = PARTITIONS[partition]
    global_index = int(target["global_sample_index"])
    if global_index not in spec["diagnostic_indices"]:
        raise AuditError("adopted target is outside diagnostic reuse indices")
    local_index = global_index - int(spec["local_index_offset"])
    run_root = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
    )
    diagnostic_root = run_root / str(spec["diagnostic_directory"]).format(
        arm=arm
    )
    provenance = read_json(diagnostic_root / "provenance.json")
    config = provenance.get("config")
    if (
        not isinstance(config, dict)
        or provenance.get("config_sha256") != canonical_sha(config)
    ):
        raise AuditError("diagnostic provenance/config SHA mismatch")
    slot_policy = config.get("slot_policy")
    if (
        not isinstance(slot_policy, dict)
        or config.get("slot_policy_sha256") != canonical_sha(slot_policy)
        or Path(str(effective.get("effective_source_directory") or "")).resolve()
        != diagnostic_root.resolve()
        or effective.get("effective_source_config_sha256")
        != provenance.get("config_sha256")
        or effective.get("effective_source_slot_policy_sha256")
        != config.get("slot_policy_sha256")
        or effective.get("effective_endpoint_sha256")
        != config.get("api_base_url_sha256")
    ):
        raise AuditError("diagnostic/effective provenance mismatch")
    attempts = [
        row
        for row in read_jsonl(diagnostic_root / "attempts.jsonl")
        if row.get("task_id") == target.get("task_id")
        and row.get("sample_index") == local_index
        and row.get("response_received") is True
        and row.get("slot_terminal") is True
    ]
    if len(attempts) != 1:
        raise AuditError("diagnostic terminal evidence is missing/ambiguous")
    terminal = attempts[0]
    outcomes = [
        row
        for row in read_jsonl(diagnostic_root / "outcomes.jsonl")
        if row.get("task_id") == terminal.get("task_id")
        and row.get("sample_index") == terminal.get("sample_index")
        and row.get("attempt_id") == terminal.get("attempt_id")
        and row.get("response_id") == terminal.get("response_id")
    ]
    if len(outcomes) != 1:
        raise AuditError("diagnostic outcome evidence is missing/ambiguous")
    return diagnostic_root, terminal, outcomes[0]


def _validate_terminal_and_outcome(
    *,
    target: Mapping[str, Any],
    effective: Mapping[str, Any],
    terminal: Mapping[str, Any],
    outcome: Mapping[str, Any],
    model: str,
    diagnostic_config_sha256: str,
) -> None:
    terminal_fields = {
        "task_id": target["task_id"],
        "sample_index": (
            int(target["global_sample_index"])
            - (
                0
                if model == "qwen3.7-max-2026-05-20"
                else 5
            )
        ),
        "attempt_id": effective["effective_attempt_id"],
        "response_id": effective["response_id"],
        "requested_model": model,
        "resolved_model": model,
        "finish_reason": effective["finish_reason"],
        "candidate_valid": effective["candidate_valid"],
        "terminal_reason": effective["terminal_reason"],
        "code_sha256": effective["code_sha256"],
        "prompt_sha256": target["prompt_sha256"],
        "config_sha256": diagnostic_config_sha256,
        "response_received": True,
        "slot_terminal": True,
    }
    for key, value in terminal_fields.items():
        if terminal.get(key) != value:
            raise AuditError(f"diagnostic terminal mismatch for {key}")
    if (
        terminal.get("schema") != EXPECTED_DIAGNOSTIC_SCHEMA
        or terminal.get("usage") != effective.get("usage")
        or canonical_sha(terminal)
        != effective.get("canonical_terminal_row_sha256")
    ):
        raise AuditError("diagnostic terminal canonical/effective mismatch")
    outcome_fields = {
        "task_id": terminal["task_id"],
        "sample_index": terminal["sample_index"],
        "attempt_id": terminal["attempt_id"],
        "response_id": terminal["response_id"],
        "finish_reason": terminal["finish_reason"],
        "candidate_valid": terminal["candidate_valid"],
        "terminal_reason": terminal["terminal_reason"],
        "code_sha256": terminal["code_sha256"],
        "config_sha256": diagnostic_config_sha256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
    }
    for key, value in outcome_fields.items():
        if outcome.get(key) != value:
            raise AuditError(f"diagnostic outcome mismatch for {key}")
    candidate_valid = terminal.get("candidate_valid") is True
    runs = outcome.get("stability_runs")
    if (
        outcome.get("schema") != EXPECTED_DIAGNOSTIC_SCHEMA
        or outcome.get("record_type") != "candidate_outcome"
        or type(outcome.get("compiled")) is not bool
        or type(outcome.get("passed")) is not bool
        or type(outcome.get("evaluation_performed")) is not bool
        or not isinstance(runs, list)
        or len(runs) != (2 if candidate_valid else 0)
        or (
            outcome.get("passed") is True
            and outcome.get("compiled") is not True
        )
    ):
        raise AuditError("diagnostic outcome result contract mismatch")
    if candidate_valid:
        attestation_satisfied = outcome.get(
            "completion_attestation_satisfied_all_runs"
        )
        if (
            outcome.get("evaluation_performed") is not True
            or outcome.get("completion_attestation_id")
            != runner.REQUIRED_ATTESTATION_ID
            or outcome.get("completion_attestation_enforced") is not True
            or type(attestation_satisfied) is not bool
        ):
            raise AuditError("diagnostic outcome attestation mismatch")
        run_attestations: list[bool] = []
        for run in runs:
            if (
                not isinstance(run, dict)
                or type(run.get("compiled")) is not bool
                or type(run.get("passed")) is not bool
                or (
                    run.get("passed") is True
                    and run.get("compiled") is not True
                )
                or run.get("completion_attestation_id")
                != runner.REQUIRED_ATTESTATION_ID
                or run.get("completion_attestation_required") is not True
                or type(run.get("completion_attestation_satisfied"))
                is not bool
            ):
                raise AuditError("diagnostic stability evidence mismatch")
            run_attestations.append(
                bool(run["completion_attestation_satisfied"])
            )
        if attestation_satisfied != all(run_attestations):
            raise AuditError("diagnostic aggregate attestation mismatch")
    elif (
        outcome.get("evaluation_performed") is not False
        or outcome.get("compiled") is not False
        or outcome.get("passed") is not False
        or outcome.get("completion_attestation_enforced") is not False
        or outcome.get(
            "completion_attestation_satisfied_all_runs"
        )
        is not False
    ):
        raise AuditError("invalid diagnostic candidate is not fail-closed")


def transform_diagnostic_outcome(
    *,
    workspace: Path,
    capacity_root: Path,
    target: Mapping[str, Any],
    effective: Mapping[str, Any],
    arm: str,
    partition: str,
) -> dict[str, Any]:
    if (
        effective.get("schema") != CAPACITY_SCHEMA
        or effective.get("record_type")
        != "effective_capacity_terminal"
        or effective.get("origin") != "adopted_clean_diagnostic"
        or effective.get("selection_id") != target.get("selection_id")
        or effective.get("arm") != arm
        or effective.get("task_id") != target.get("task_id")
        or effective.get("global_sample_index")
        != target.get("global_sample_index")
    ):
        raise AuditError("effective adopted-terminal identity mismatch")
    diagnostic_root, terminal, source_outcome = _diagnostic_source(
        workspace,
        target=target,
        effective=effective,
        arm=arm,
        partition=partition,
    )
    diagnostic_provenance = read_json(diagnostic_root / "provenance.json")
    diagnostic_config = diagnostic_provenance["config"]
    model = str(PARTITIONS[partition]["diagnostic_model"])
    _validate_terminal_and_outcome(
        target=target,
        effective=effective,
        terminal=terminal,
        outcome=source_outcome,
        model=model,
        diagnostic_config_sha256=str(
            diagnostic_provenance["config_sha256"]
        ),
    )
    source_outcome_sha = canonical_sha(source_outcome)
    if effective.get("canonical_outcome_row_sha256") != source_outcome_sha:
        raise AuditError(
            "legacy effective outcome binding does not match diagnostic source"
        )
    source_provenance = {
        "kind": "clean_diagnostic_candidate_outcome",
        "directory": str(diagnostic_root.resolve()),
        "schema": source_outcome["schema"],
        "record_type": source_outcome["record_type"],
        "config_sha256": diagnostic_provenance["config_sha256"],
        "slot_policy_sha256": diagnostic_config["slot_policy_sha256"],
        "endpoint_sha256": diagnostic_config["api_base_url_sha256"],
        "task_id": source_outcome["task_id"],
        "sample_index": source_outcome["sample_index"],
        "attempt_id": source_outcome["attempt_id"],
        "response_id": source_outcome["response_id"],
        "canonical_row_sha256": source_outcome_sha,
    }
    row = {
        "schema": CAPACITY_SCHEMA,
        "record_type": "capacity_candidate_outcome",
        "outcome_origin": "adopted_clean_diagnostic_reconciliation_v1",
        "selection_id": target["selection_id"],
        "task_id": target["task_id"],
        "global_sample_index": target["global_sample_index"],
        "arm": arm,
        "attempt_id": terminal["attempt_id"],
        "response_id": terminal["response_id"],
        "model": terminal["resolved_model"],
        "finish_reason": terminal["finish_reason"],
        "candidate_valid": terminal["candidate_valid"],
        "terminal_reason": terminal["terminal_reason"],
        "code_sha256": terminal["code_sha256"],
        "evaluator_entrypoint": source_outcome.get(
            "evaluator_entrypoint"
        ),
        "evaluator_sha256": source_outcome["evaluator_sha256"],
        "evaluation_performed": source_outcome["evaluation_performed"],
        "compiled": source_outcome["compiled"],
        "passed": source_outcome["passed"],
        "completion_attestation_id": source_outcome[
            "completion_attestation_id"
        ],
        "completion_attestation_enforced": source_outcome[
            "completion_attestation_enforced"
        ],
        "completion_attestation_satisfied_all_runs": source_outcome[
            "completion_attestation_satisfied_all_runs"
        ],
        "stability_runs": source_outcome["stability_runs"],
        "evaluated_at": source_outcome["evaluated_at"],
        "source_outcome_provenance": source_provenance,
        "legacy_effective_outcome_binding_sha256": source_outcome_sha,
        "reconciliation_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "provider_calls": 0,
        "source_journals_modified": False,
        "base_capacity_journals_modified": False,
    }
    row["reconciliation_payload_sha256"] = canonical_sha(row)
    return row


def load_effective_outcomes(
    capacity_root: Path,
) -> dict[str, dict[str, Any]]:
    path = (
        capacity_root.resolve()
        / OVERLAY_DIRECTORY
        / OUTCOME_FILE
    )
    rows = _unique_by(
        read_jsonl(path), "selection_id", label="reconciled outcome"
    )
    for row in rows.values():
        immutable = dict(row)
        observed = immutable.pop("reconciliation_payload_sha256", None)
        source = row.get("source_outcome_provenance")
        if (
            row.get("schema") != CAPACITY_SCHEMA
            or row.get("record_type") != "capacity_candidate_outcome"
            or row.get("outcome_origin")
            != "adopted_clean_diagnostic_reconciliation_v1"
            or row.get("reconciliation_contract_sha256")
            != EXPECTED_CONTRACT_SHA256
            or row.get("provider_calls") != 0
            or row.get("source_journals_modified") is not False
            or row.get("base_capacity_journals_modified") is not False
            or not isinstance(source, dict)
            or row.get("legacy_effective_outcome_binding_sha256")
            != source.get("canonical_row_sha256")
            or observed != canonical_sha(immutable)
        ):
            raise AuditError("reconciled outcome integrity mismatch")
    return rows


def _prepare_overlay(
    workspace: Path,
    capacity_root: Path,
    *,
    arm: str,
    partition: str,
) -> tuple[Path, dict[str, Any]]:
    patch = workspace / "frontier_ceiling_patch_v1"
    if (
        sha256_file(patch / CONTRACT_NAME) != EXPECTED_CONTRACT_SHA256
        or sha256_file(patch / "qwen37_capacity_fallback_v6.py")
        != EXPECTED_CAPACITY_ENTRY_SHA256
        or sha256_file(
            patch / "qwen37_capacity_fallback_contract_v6.json"
        )
        != EXPECTED_CAPACITY_CONTRACT_SHA256
    ):
        raise AuditError("reconciliation dependency hash mismatch")
    overlay = capacity_root / OVERLAY_DIRECTORY
    overlay.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": SCHEMA,
        "capacity_root": str(capacity_root.resolve()),
        "arm": arm,
        "partition": partition,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "capacity_entry_sha256": EXPECTED_CAPACITY_ENTRY_SHA256,
        "capacity_contract_sha256": EXPECTED_CAPACITY_CONTRACT_SHA256,
        "provider_imports": False,
        "provider_calls": 0,
    }
    provenance = {
        "schema": SCHEMA,
        "status": "preflight_complete",
        "config": config,
        "config_sha256": canonical_sha(config),
        "contract": runner.file_record(patch / CONTRACT_NAME),
        "entry": runner.file_record(Path(__file__).resolve()),
    }
    provenance_path = overlay / "provenance.json"
    if provenance_path.is_file():
        existing = read_json(provenance_path)
        if (
            existing.get("config_sha256") != provenance["config_sha256"]
            or existing.get("entry", {}).get("sha256")
            != provenance["entry"]["sha256"]
        ):
            raise AuditError("reconciliation provenance changed on resume")
    else:
        runner.atomic_write_json(provenance_path, provenance)
    return overlay, provenance


def reconcile_one(
    workspace: Path,
    capacity_root: Path,
    *,
    arm: str,
    partition: str,
) -> dict[str, Any]:
    capacity_root = capacity_root.resolve()
    overlay, provenance = _prepare_overlay(
        workspace.resolve(),
        capacity_root,
        arm=arm,
        partition=partition,
    )
    targets, effective, base_outcomes = _load_capacity_context(
        capacity_root, arm=arm, partition=partition
    )
    existing = load_effective_outcomes(capacity_root)
    missing = set(effective).difference(base_outcomes)
    ineligible = sorted(
        sid
        for sid in missing
        if effective[sid].get("origin") != "adopted_clean_diagnostic"
    )
    if ineligible:
        raise AuditError(
            "non-adopted effective terminals lack base outcomes: "
            + ",".join(ineligible)
        )
    if set(existing) != missing:
        unexpected = sorted(set(existing).difference(missing))
        absent = sorted(missing.difference(existing))
        if unexpected:
            raise AuditError(
                "reconciliation contains no-longer-missing selections: "
                + ",".join(unexpected)
            )
    else:
        absent = []
    journal = runner.JsonlJournal(overlay / OUTCOME_FILE)
    appended: list[dict[str, Any]] = []
    verified: dict[str, dict[str, Any]] = {}
    for sid in sorted(missing):
        expected = transform_diagnostic_outcome(
            workspace=workspace.resolve(),
            capacity_root=capacity_root,
            target=targets[sid],
            effective=effective[sid],
            arm=arm,
            partition=partition,
        )
        current = existing.get(sid)
        if current is None:
            journal.append(expected)
            existing[sid] = expected
            appended.append(expected)
            current = expected
        elif current != expected:
            raise AuditError("reconciled outcome changed on resume")
        verified[sid] = current
    if set(verified) != missing:
        raise AuditError("reconciliation does not exactly cover missing rows")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "capacity_root": str(capacity_root),
        "arm": arm,
        "partition": partition,
        "effective_terminals": len(effective),
        "base_outcomes": len(base_outcomes),
        "missing_adopted_outcomes": len(missing),
        "reconciled_outcomes": len(verified),
        "newly_appended": len(appended),
        "newly_appended_compiled": sum(
            row["compiled"] is True for row in appended
        ),
        "newly_appended_passed": sum(
            row["passed"] is True for row in appended
        ),
        "total_reconciled_compiled": sum(
            row["compiled"] is True for row in verified.values()
        ),
        "total_reconciled_passed": sum(
            row["passed"] is True for row in verified.values()
        ),
        "provider_imports": False,
        "provider_calls": 0,
        "source_journals_modified": False,
        "base_capacity_journals_modified": False,
        "config_sha256": provenance["config_sha256"],
    }
    runner.atomic_write_json(overlay / "summary.json", report)
    return report


def reconcile_all(workspace: Path) -> dict[str, Any]:
    run_root = (
        workspace.resolve()
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
    )
    reports: list[dict[str, Any]] = []
    for partition, arm, directory in CAPACITY_DIRECTORIES:
        capacity_root = run_root / directory
        if not capacity_root.is_dir():
            continue
        reports.append(
            reconcile_one(
                workspace.resolve(),
                capacity_root,
                arm=arm,
                partition=partition,
            )
        )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "capacity_outputs": len(reports),
        "newly_appended": sum(row["newly_appended"] for row in reports),
        "newly_appended_compiled": sum(
            row["newly_appended_compiled"] for row in reports
        ),
        "newly_appended_passed": sum(
            row["newly_appended_passed"] for row in reports
        ),
        "reconciled_outcomes": sum(
            row["reconciled_outcomes"] for row in reports
        ),
        "provider_imports": False,
        "provider_calls": 0,
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reconcile", "status"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--capacity-out", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--partition", choices=tuple(PARTITIONS))
    args = parser.parse_args()
    if args.capacity_out is not None and (
        args.arm is None or args.partition is None
    ):
        parser.error("--capacity-out requires --arm and --partition")
    if args.capacity_out is None and (
        args.arm is not None or args.partition is not None
    ):
        parser.error("--arm/--partition require --capacity-out")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.capacity_out is None:
            report = reconcile_all(args.workspace.resolve())
        else:
            report = reconcile_one(
                args.workspace.resolve(),
                args.capacity_out.resolve(),
                arm=args.arm,
                partition=args.partition,
            )
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
