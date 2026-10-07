#!/usr/bin/env python3
"""Build the collector's sealed quality gate from an exact K=8 pilot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.build_qwen_sequence_kd import strict_json  # noqa: E402
from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    ArtifactError,
    SAMPLES_PER_TASK,
    atomic_write_json,
    file_record,
    read_jsonl,
)


GATE_SCHEMA = "qwen-teacher-quality-gate-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pilot-audit", required=True, type=Path)
    parser.add_argument("--pilot-verified-only", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot-tasks", type=int, default=16)
    parser.add_argument("--minimum-verified-tasks", type=int, default=0)
    parser.add_argument("--minimum-parseable-fraction", type=float, default=0.5)
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.pilot_audit.expanduser().resolve()
    verified_path = args.pilot_verified_only.expanduser().resolve()
    audit = strict_json(audit_path)
    verified_rows = read_jsonl(verified_path)
    pilot_tasks = int(args.pilot_tasks)
    minimum_verified = int(args.minimum_verified_tasks)
    minimum_parseable = float(args.minimum_parseable_fraction)
    if (
        pilot_tasks != 16
        or not 0 <= minimum_verified <= pilot_tasks
        or not 0.0 <= minimum_parseable <= 1.0
    ):
        raise ArtifactError("invalid fixed Qwen pilot thresholds")
    coverage = audit.get("coverage")
    sampling = audit.get("sampling")
    target_gate = audit.get("target_length_gate")
    if not all(
        isinstance(value, Mapping)
        for value in (coverage, sampling, target_gate)
    ):
        raise ArtifactError("pilot audit lacks coverage/sampling/target gate")
    candidates = int(coverage.get("candidates", -1))
    parseable = int(coverage.get("parseable_candidates", -1))
    if candidates != pilot_tasks * SAMPLES_PER_TASK:
        raise ArtifactError(
            f"pilot candidates={candidates}, expected={pilot_tasks * SAMPLES_PER_TASK}"
        )
    parseable_fraction = parseable / candidates
    verified_task_ids = {
        str(row.get("task_id") or "") for row in verified_rows
    }
    verified_task_ids.discard("")
    verified_tasks = len(verified_task_ids)
    unique_by_task = sampling.get("unique_final_sequences_per_task")
    if (
        parseable_fraction < minimum_parseable
        or verified_tasks < minimum_verified
        or not isinstance(unique_by_task, Mapping)
        or len(unique_by_task) != pilot_tasks
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= SAMPLES_PER_TASK
            for value in unique_by_task.values()
        )
        or sampling.get("pathological_all_tasks_have_identical_k8_draws")
        is not False
        or target_gate.get("passed") is not True
        or int(target_gate.get("targets_checked", -1)) != candidates
        or int(target_gate.get("overflow_count", -1)) != 0
    ):
        raise ArtifactError("pilot did not pass the production quality gate")
    unique_values = [int(value) for value in unique_by_task.values()]
    identical_tasks = sum(value == 1 for value in unique_values)
    target_contract = target_gate.get("target_contract")
    if not isinstance(target_contract, Mapping):
        raise ArtifactError("pilot target gate lacks its trainer contract")
    trainer_contract = target_contract.get("trainer_contract")
    if not isinstance(trainer_contract, Mapping):
        raise ArtifactError("pilot target gate lacks trainer contract record")
    result = {
        "schema": GATE_SCHEMA,
        "passed": True,
        "pilot_tasks": pilot_tasks,
        "candidates": candidates,
        "parseable_candidates": parseable,
        "parseable_fraction": parseable_fraction,
        "verified_tasks": verified_tasks,
        "minimum_verified_tasks": minimum_verified,
        "verified_correctness_is_diagnostic_only": minimum_verified == 0,
        "minimum_parseable_fraction": minimum_parseable,
        "sampling_diversity": {
            "unique_final_sequences_per_task": dict(unique_by_task),
            "tasks_with_all_k8_draws_identical": identical_tasks,
            "minimum_unique_final_sequences_per_task": min(unique_values),
            "maximum_unique_final_sequences_per_task": max(unique_values),
            "pathological_all_tasks_have_identical_k8_draws": False,
            "duplicate_draws_filtered": False,
        },
        "target_length_gate": {
            "passed": True,
            "target_length_evidence_sha256": target_gate["evidence_sha256"],
            "target_contract_sha256": trainer_contract["sha256"],
            "max_target_tokens": int(target_contract["max_target_tokens"]),
            "overflow_count": 0,
            "non_code_target_count": int(
                target_gate.get("non_code_target_count", 0)
            ),
            "final_dart_code_only_required": False,
            "truncate": False,
            "filter_draw": False,
            "resample_draw": False,
        },
        "pilot_audit_sha256": file_record(audit_path)["sha256"],
        "pilot_verified_only_sha256": file_record(verified_path)["sha256"],
    }
    atomic_write_json(args.output.expanduser().resolve(), result)
    return result


def main() -> int:
    gate = build(parse_args())
    print(
        "QWEN_QUALITY_GATE "
        f"tasks={gate['pilot_tasks']} "
        f"candidates={gate['candidates']} "
        f"parseable={gate['parseable_candidates']} "
        f"verified_tasks={gate['verified_tasks']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
