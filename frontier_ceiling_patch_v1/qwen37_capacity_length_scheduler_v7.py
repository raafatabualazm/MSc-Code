#!/usr/bin/env python3
"""Automatic endpoint-epoch scheduler for capacity length repairs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import frontier_passk as runner
import qwen37_capacity_fallback_v6 as capacity


SCHEMA = "qwen37-capacity-length-scheduler-v7"
ENTRY_SHA256 = (
    "a74cb5595032c5d2b1d7fc325e38903ccf4f8fa844e8caf0f15f30a853ec19a8"
)
CONTRACT_SHA256 = (
    "b69b5cf91f33e785f78e965eb67c814372fa774ded4d19114eddf49ba9149809"
)
LAUNCHER_SHA256 = (
    "bee20556307ebe4b8fe3b8cd9f016e1417d1f8ca70ff34ee1ca81418078aec67"
)
CAPACITY_ENTRY_SHA256 = (
    "6b2d642be25bb7b2e97daddf70e9a2245a8ff09e5f0c6e5e32c09afc92159521"
)
CAPACITY_CONTRACT_SHA256 = (
    "cea8acaa785ddc2685a5da8b4426dce41837a25af2e7dd9639dd70f632d59631"
)
CAPACITY_SCHEDULER_SHA256 = (
    "ac87e57264dd341b19b5d80e8862642c06e91f5d62f6cb4dcb2a725f34a0c792"
)
EXTENSION_CONTRACT_SHA256 = (
    "db4ee883e5073f08a0dad160e7e5ea594c54a3a3dd0c31a68fe4479aea1536ef"
)
STATUS_RECONCILIATION_ADDENDUM_SHA256 = (
    "df2364fb93d80be5b53fba1d9f01f0dc437ddbe18c7694d574c2b02a9ad3b881"
)
STATUS_SHA256 = (
    "f8efd980737e4efc9a41c521c12ec62a69fa098f8de698021a840861c03cb1b2"
)
RECONCILIATION_ENTRY_SHA256 = (
    "b8244a8092c76da6cfa1d849b64c64ad9e7a56f8d0016c04672e20e1b3721441"
)
RECONCILIATION_CONTRACT_SHA256 = (
    "ebbb10cfa939c1fa4fbdd26e46f058960a7b1841a56d53cb6c476184f1644825"
)
PARTITIONS = ("0520", "0608")
ARMS = ("opus", "codex")
STAGES = ("dated", "preview", "source", "generic", "fallback")
V6_STAGE_TO_REPAIR_STAGE = {
    "current_dated": "dated",
    "current_preview": "preview",
    "current_0517": "source",
    "current_generic": "generic",
    "fallback_all_five": "fallback",
}


class SchedulerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    value = runner.sha256_file(path)
    if not value:
        raise SchedulerError(f"cannot hash required file: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchedulerError(f"JSON artifact is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SchedulerError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def append_journal(path: Path, row: dict[str, Any]) -> None:
    runner.JsonlJournal(path).append(row)


def capacity_complete(workspace: Path) -> bool:
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    for partition in PARTITIONS:
        for arm in ARMS:
            status = read_json(
                run_root
                / f"qwen37_capacity_v6_{partition}_{arm}_mc12k_tb8k"
                / "status.json"
            )
            if status.get("status") != "complete":
                return False
    return True


def v6_stage(workspace: Path) -> str:
    journal = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "qwen37_capacity_v6_scheduler"
        / "transitions.jsonl"
    )
    rows = read_jsonl(journal)
    observed_stages = [
        str(row.get("stage") or "")
        for row in rows
        if row.get("record_type") == "stage_preflight"
        and str(row.get("stage") or "") in V6_STAGE_TO_REPAIR_STAGE
    ]
    if not observed_stages:
        return "dated"
    return V6_STAGE_TO_REPAIR_STAGE[observed_stages[-1]]


def required_repair_stages(stage: str) -> tuple[str, ...]:
    if stage not in STAGES:
        raise SchedulerError(f"unknown repair stage: {stage}")
    return STAGES[: STAGES.index(stage) + 1]


def repair_root(
    workspace: Path,
    *,
    partition: str,
    arm: str,
    stage: str,
) -> Path:
    suffix = {
        "dated": "secondary_dated",
        "preview": "secondary_preview",
        "source": "secondary_0517",
        "generic": "secondary_generic",
        "fallback": "fallback_all_five",
    }[stage]
    return (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
        / f"qwen37_capacity_length_v7_{partition}_{arm}_"
        f"repair_epoch_{suffix}"
    )


def exact_quota_failure(root: Path) -> bool:
    failure = read_json(root / "failure.json")
    if not failure:
        return False
    rows = read_jsonl(root / "repair_attempts.jsonl")
    boundaries = [
        row
        for row in rows
        if row.get("response_received") is False
        and row.get("retryable_transport") is False
        and capacity.exact_quota_403(str(row.get("transport_error") or ""))
    ]
    return bool(boundaries)


def hard_failure(root: Path) -> str | None:
    failure = read_json(root / "failure.json")
    if not failure or exact_quota_failure(root):
        return None
    return str(failure.get("error") or "unclassified v7 failure")


def call_launcher(
    workspace: Path,
    *,
    operation: str,
    stage: str,
    partition: str,
    arm: str,
) -> int:
    launcher = (
        workspace
        / "frontier_ceiling_patch_v1"
        / "run_qwen37_capacity_length_v7.sh"
    )
    env = dict(os.environ)
    env["WORKSPACE"] = str(workspace)
    env["V7_ONLY_PARTITION"] = partition
    env["V7_ONLY_ARM"] = arm
    completed = subprocess.run(
        ["bash", str(launcher), f"{operation}-{stage}"],
        cwd=str(launcher.parent),
        env=env,
        check=False,
    )
    return completed.returncode


def ensure_preflight(
    workspace: Path,
    *,
    stage: str,
    partition: str,
    arm: str,
) -> None:
    root = repair_root(
        workspace,
        partition=partition,
        arm=arm,
        stage=stage,
    )
    if (root / "preflight.json").is_file():
        return
    return_code = call_launcher(
        workspace,
        operation="preflight",
        stage=stage,
        partition=partition,
        arm=arm,
    )
    if return_code != 0:
        raise SchedulerError(
            f"v7 preflight failed: {stage} {partition} {arm} rc={return_code}"
        )


def ensure_prior_preflights(
    workspace: Path,
    *,
    stage: str,
    partition: str,
    arm: str,
) -> None:
    for required_stage in required_repair_stages(stage):
        ensure_preflight(
            workspace,
            stage=required_stage,
            partition=partition,
            arm=arm,
        )


def all_required_repairs_complete(workspace: Path, *, stage: str) -> bool:
    for required_stage in required_repair_stages(stage):
        for partition in PARTITIONS:
            for arm in ARMS:
                summary = read_json(
                    repair_root(
                        workspace,
                        partition=partition,
                        arm=arm,
                        stage=required_stage,
                    )
                    / "summary.json"
                )
                if summary.get("status") != "complete":
                    return False
    return True


def validate_dependencies(workspace: Path) -> None:
    patch = workspace / "frontier_ceiling_patch_v1"
    expected = {
        patch / "qwen37_capacity_length_repair_v7.py": ENTRY_SHA256,
        patch / "qwen37_capacity_length_contract_v7.json": CONTRACT_SHA256,
        patch / "run_qwen37_capacity_length_v7.sh": LAUNCHER_SHA256,
        patch / "qwen37_capacity_fallback_v6.py": CAPACITY_ENTRY_SHA256,
        patch
        / "qwen37_capacity_fallback_contract_v6.json": (
            CAPACITY_CONTRACT_SHA256
        ),
        patch / "qwen37_capacity_scheduler_v6.py": CAPACITY_SCHEDULER_SHA256,
        patch
        / "qwen37_current_five_capacity_extension_v1.json": (
            EXTENSION_CONTRACT_SHA256
        ),
        patch / "qwen37_primary_effective_status_v7.py": STATUS_SHA256,
        patch
        / "qwen37_effective_status_reconciliation_extension_v1.json": (
            STATUS_RECONCILIATION_ADDENDUM_SHA256
        ),
        patch
        / "qwen37_original_outcome_reconciliation_v1.py": (
            RECONCILIATION_ENTRY_SHA256
        ),
        patch
        / "qwen37_original_outcome_reconciliation_contract_v1.json": (
            RECONCILIATION_CONTRACT_SHA256
        ),
    }
    for path, required in expected.items():
        actual = sha256_file(path)
        if actual != required:
            raise SchedulerError(
                f"v7 scheduler dependency hash mismatch: {path} {actual}"
            )


def run_scheduler(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    validate_dependencies(workspace)
    journal = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "qwen37_capacity_length_v7_scheduler"
        / "transitions.jsonl"
    )
    deadline = time.monotonic() + args.timeout_seconds
    last_stage = ""
    while time.monotonic() < deadline:
        stage = v6_stage(workspace)
        if stage != last_stage:
            append_journal(
                journal,
                {
                    "schema": SCHEMA,
                    "record_type": "repair_stage_observed",
                    "stage": stage,
                    "capacity_complete": capacity_complete(workspace),
                    "recorded_at": runner.utc_now(),
                },
            )
            last_stage = stage
        stages_to_run = (
            required_repair_stages(stage)
            if capacity_complete(workspace)
            else (stage,)
        )
        for repair_stage in stages_to_run:
            for partition in PARTITIONS:
                for arm in ARMS:
                    ensure_prior_preflights(
                        workspace,
                        stage=repair_stage,
                        partition=partition,
                        arm=arm,
                    )
                    root = repair_root(
                        workspace,
                        partition=partition,
                        arm=arm,
                        stage=repair_stage,
                    )
                    error = hard_failure(root)
                    if error is not None:
                        raise SchedulerError(
                            f"v7 hard failure "
                            f"{repair_stage}/{partition}/{arm}: {error}"
                        )
                    if exact_quota_failure(root):
                        continue
                    return_code = call_launcher(
                        workspace,
                        operation="once",
                        stage=repair_stage,
                        partition=partition,
                        arm=arm,
                    )
                    if return_code != 0:
                        if exact_quota_failure(root):
                            append_journal(
                                journal,
                                {
                                    "schema": SCHEMA,
                                    "record_type": (
                                        "repair_exact_quota_boundary"
                                    ),
                                    "stage": repair_stage,
                                    "partition": partition,
                                    "arm": arm,
                                    "failure_sha256": sha256_file(
                                        root / "failure.json"
                                    ),
                                    "recorded_at": runner.utc_now(),
                                },
                            )
                            continue
                        error = hard_failure(root)
                        raise SchedulerError(
                            f"v7 repair invocation failed "
                            f"{repair_stage}/{partition}/{arm}: "
                            f"{error or return_code}"
                        )
        if capacity_complete(workspace) and all_required_repairs_complete(
            workspace,
            stage=stage,
        ):
            append_journal(
                journal,
                {
                    "schema": SCHEMA,
                    "record_type": "scheduler_complete",
                    "status": "complete",
                    "terminal_capacity_stage": stage,
                    "required_repair_stages": list(
                        required_repair_stages(stage)
                    ),
                    "recorded_at": runner.utc_now(),
                },
            )
            return 0
        time.sleep(args.poll_seconds)
    raise SchedulerError("v7 capacity-length scheduler timed out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=604800)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("poll/timeout seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        return run_scheduler(args)
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
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
