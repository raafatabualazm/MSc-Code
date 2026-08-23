#!/usr/bin/env python3
"""Fail-closed automatic capacity-epoch scheduler for the v6 Qwen overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "qwen37-capacity-epoch-scheduler-v6"
OVERLAY_ENTRY_SHA256 = (
    "6b2d642be25bb7b2e97daddf70e9a2245a8ff09e5f0c6e5e32c09afc92159521"
)
LAUNCHER_SHA256 = (
    "88b84b3a1b9138b89d83ac86c0fc5a1f5e1766d4b1ce8091be93636f9e3235ed"
)
CONTRACT_SHA256 = (
    "cea8acaa785ddc2685a5da8b4426dce41837a25af2e7dd9639dd70f632d59631"
)
EXTENSION_CONTRACT_SHA256 = (
    "db4ee883e5073f08a0dad160e7e5ea594c54a3a3dd0c31a68fe4479aea1536ef"
)
ARMS = ("opus", "codex")
PARTITIONS = ("0520", "0608")
QUOTA_CODES = ("AllocationQuota.FreeTierOnly", "insufficient_quota")


class SchedulerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stage:
    name: str
    action_suffix: str
    epoch: str
    unit_stage: str
    required_exact_aliases: tuple[str, ...]


STAGES = (
    Stage(
        name="current_dated",
        action_suffix="",
        epoch="secondary-workspace-20260726-epoch1",
        unit_stage="",
        required_exact_aliases=(
            "qwen3.7-max-2026-05-20",
            "qwen3.7-max-2026-06-08",
        ),
    ),
    Stage(
        name="current_preview",
        action_suffix="-current-preview",
        epoch="secondary-workspace-20260726-preview-epoch1",
        unit_stage="-preview",
        required_exact_aliases=(
            "qwen3.7-max-2026-05-20",
            "qwen3.7-max-2026-06-08",
            "qwen3.7-max-preview",
        ),
    ),
    Stage(
        name="current_0517",
        action_suffix="-current-0517",
        epoch="secondary-workspace-20260726-0517-epoch1",
        unit_stage="-0517",
        required_exact_aliases=(
            "qwen3.7-max-2026-05-20",
            "qwen3.7-max-2026-06-08",
            "qwen3.7-max-preview",
            "qwen3.7-max-2026-05-17",
        ),
    ),
    Stage(
        name="current_generic",
        action_suffix="-current-generic",
        epoch="secondary-workspace-20260726-generic-epoch1",
        unit_stage="-generic",
        required_exact_aliases=("qwen3.7-max",),
    ),
    Stage(
        name="fallback_all_five",
        action_suffix="-fallback",
        epoch="fallback-workspace-20260726-epoch1",
        unit_stage="-fallback",
        required_exact_aliases=(
            "qwen3.7-max-2026-05-20",
            "qwen3.7-max-2026-06-08",
            "qwen3.7-max-preview",
            "qwen3.7-max",
            "qwen3.7-max-2026-05-17",
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchedulerError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SchedulerError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload["record_sha256"] = stable_sha(payload)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exact_quota_403(row: dict[str, Any]) -> bool:
    error = str(row.get("error") or "")
    return (
        row.get("record_type") == "capacity_route_attempt"
        and row.get("route_result") == "quota_exhausted_403"
        and row.get("response_received") is False
        and row.get("http_status") == 403
        and "Error code: 403" in error
        and "free quota has been exhausted" in error.lower()
        and any(code in error for code in QUOTA_CODES)
    )


def output_dirs(workspace: Path) -> list[Path]:
    root = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
    )
    return [
        root / f"qwen37_capacity_v6_{partition}_{arm}_mc12k_tb8k"
        for partition in PARTITIONS
        for arm in ARMS
    ]


def unit_names(stage: Stage) -> list[str]:
    return [
        (
            f"frontier-qwen37-capacity-v6-{partition}-{arm}"
            f"{stage.unit_stage}-mc12k-tb8k.service"
        )
        for partition in PARTITIONS
        for arm in ARMS
    ]


def all_complete(workspace: Path) -> bool:
    for out in output_dirs(workspace):
        targets = read_jsonl(out / "targets.jsonl")
        effective = read_jsonl(out / "effective_terminals.jsonl")
        if not targets or len(effective) != len(targets):
            return False
        target_ids = {str(row.get("selection_id") or "") for row in targets}
        effective_ids = {
            str(row.get("selection_id") or "") for row in effective
        }
        if (
            "" in target_ids
            or "" in effective_ids
            or len(target_ids) != len(targets)
            or len(effective_ids) != len(effective)
            or target_ids != effective_ids
        ):
            raise SchedulerError("effective/target selection identity mismatch")
    return True


def epoch_exact_evidence(
    workspace: Path,
    stage: Stage,
) -> dict[str, list[dict[str, str]]]:
    evidence: dict[str, list[dict[str, str]]] = {
        alias: [] for alias in stage.required_exact_aliases
    }
    dispatch_ids: set[str] = set()
    terminal_ids: set[str] = set()
    for out in output_dirs(workspace):
        for row in read_jsonl(out / "dispatches.jsonl"):
            if row.get("capacity_epoch") != stage.epoch:
                continue
            attempt_id = str(row.get("attempt_id") or "")
            if not attempt_id or attempt_id in dispatch_ids:
                raise SchedulerError("duplicate/missing stage dispatch attempt")
            dispatch_ids.add(attempt_id)
        route_path = out / "route_attempts.jsonl"
        for row in read_jsonl(route_path):
            if row.get("capacity_epoch") != stage.epoch:
                continue
            attempt_id = str(row.get("attempt_id") or "")
            if not attempt_id:
                if (
                    row.get("source") == "clean_diagnostic_journal"
                    and row.get("route_result") == "data_inspection_failed"
                ):
                    continue
                raise SchedulerError("stage route row lacks attempt identity")
            if attempt_id in terminal_ids:
                raise SchedulerError("duplicate/missing stage route attempt")
            terminal_ids.add(attempt_id)
            model = str(row.get("requested_model") or "")
            if model in evidence and exact_quota_403(row):
                evidence[model].append(
                    {
                        "path": str(route_path),
                        "attempt_id": attempt_id,
                        "row_sha256": stable_sha(row),
                    }
                )
    unmatched = sorted(dispatch_ids - terminal_ids)
    if unmatched:
        raise SchedulerError(
            f"stage has {len(unmatched)} dispatches without terminal routes"
        )
    return evidence


def units_inactive(stage: Stage) -> tuple[bool, list[dict[str, str]]]:
    states: list[dict[str, str]] = []
    inactive = True
    for unit in unit_names(stage):
        completed = subprocess.run(
            [
                "systemctl",
                "show",
                unit,
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=Result",
                "--property=ExecMainStatus",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        values = {"unit": unit}
        for line in completed.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        states.append(values)
        if values.get("LoadState") != "loaded":
            inactive = False
        if values.get("ActiveState") in {"active", "activating", "reloading"}:
            inactive = False
    return inactive, states


def launcher_action(
    workspace: Path,
    *,
    operation: str,
    stage: Stage,
) -> None:
    launcher = (
        workspace
        / "frontier_ceiling_patch_v1"
        / "run_qwen37_capacity_fallback_v6.sh"
    )
    action = operation + stage.action_suffix
    env = dict(os.environ)
    env["CAPACITY_EPOCH"] = stage.epoch
    completed = subprocess.run(
        ["bash", str(launcher), action],
        cwd=str(launcher.parent),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise SchedulerError(
            f"launcher action failed: {action} rc={completed.returncode}"
        )


def wait_stage(
    workspace: Path,
    stage: Stage,
    *,
    poll_seconds: int,
    timeout_seconds: int,
) -> tuple[str, dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all_complete(workspace):
            inactive, states = units_inactive(stage)
            if inactive:
                return "complete", {}, states
        inactive, states = units_inactive(stage)
        if inactive:
            evidence = epoch_exact_evidence(workspace, stage)
            if all(evidence.values()):
                return "exact_quota_exhausted", evidence, states
            raise SchedulerError(
                f"{stage.name} stopped without every exact quota boundary"
            )
        time.sleep(poll_seconds)
    raise SchedulerError(f"{stage.name} exceeded scheduler timeout")


def run_scheduler(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    patch = workspace / "frontier_ceiling_patch_v1"
    exact_hashes = {
        patch / "qwen37_capacity_fallback_v6.py": OVERLAY_ENTRY_SHA256,
        patch / "run_qwen37_capacity_fallback_v6.sh": LAUNCHER_SHA256,
        patch / "qwen37_capacity_fallback_contract_v6.json": CONTRACT_SHA256,
        patch
        / "qwen37_current_five_capacity_extension_v1.json": (
            EXTENSION_CONTRACT_SHA256
        ),
    }
    for path, expected in exact_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise SchedulerError(
                f"scheduler dependency SHA mismatch: {path} {actual}"
            )
    journal = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "qwen37_capacity_v6_scheduler"
        / "transitions.jsonl"
    )
    for stage in STAGES:
        if all_complete(workspace):
            append_journal(
                journal,
                {
                    "schema": SCHEMA,
                    "record_type": "scheduler_complete",
                    "status": "complete",
                    "before_stage": stage.name,
                    "recorded_at": time.time(),
                },
            )
            return 0
        launcher_action(workspace, operation="preflight", stage=stage)
        append_journal(
            journal,
            {
                "schema": SCHEMA,
                "record_type": "stage_preflight",
                "stage": stage.name,
                "capacity_epoch": stage.epoch,
                "selection_reads_outcomes": False,
                "recorded_at": time.time(),
            },
        )
        launcher_action(workspace, operation="start", stage=stage)
        status, evidence, states = wait_stage(
            workspace,
            stage,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.stage_timeout_seconds,
        )
        append_journal(
            journal,
            {
                "schema": SCHEMA,
                "record_type": "stage_terminal",
                "stage": stage.name,
                "capacity_epoch": stage.epoch,
                "status": status,
                "required_exact_aliases": list(stage.required_exact_aliases),
                "exact_boundary_evidence": evidence,
                "unit_states": states,
                "selection_reads_outcomes": False,
                "recorded_at": time.time(),
            },
        )
        if status == "complete":
            return 0
    if all_complete(workspace):
        return 0
    return 76


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--stage-timeout-seconds", type=int, default=259200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_scheduler(args)
    except SchedulerError as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "failed_closed",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
