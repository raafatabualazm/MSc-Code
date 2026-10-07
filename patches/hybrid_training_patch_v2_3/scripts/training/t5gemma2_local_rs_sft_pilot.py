#!/usr/bin/env python3
"""Harvest TRAIN-only local T5Gemma 2 RS-SFT targets with a private gate.

For each task in a deterministic pilot schedule this program:

1. samples an ordinary group from the original, sealed F2 encoder source;
2. when compiler repair is enabled and that group has no visible-test pass,
   selects diverse non-compiling candidates and samples
   compiler-diagnostic-conditioned repairs;
3. finishes *all* generation and visible scoring for the task; and only then
4. uses the complementary private train holdback as a binary transfer gate.

Private holdback text and diagnostics are never serialized to the model,
repair context, journal, or training outputs.  A private-gate failure cannot
cause another generation.  Accepted code is paired with the task's original
F2 source, and matched intervention/control files include configurable
original-gold replay.  There are no frontier API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.graph_compile_at_k_antigravity import (
    evaluate_dart_jit_tests_detail,
    validate_dart_binary,
)
from scripts.evaluation.t5gemma2_f2_passk_inference import (
    _checkpoint_record,
    generate_candidate_batch,
    load_policy,
)
from scripts.preprocessing.build_verpo_feedback_view import (
    SPLIT_SCHEMA,
    extract_expect_spans,
)
from scripts.training.seq2seq_verpo_core import (
    build_compiler_repair_context,
    max_min_diverse_indices,
    sanitize_compiler_diagnostic,
    sha256_text,
)
from scripts.training.t5gemma2_compiler_feedback_verpo import (
    _decoder_special_ids,
    _encode_source,
)
from scripts.training.t5gemma2_enriched_sft import build_encoder_source


RUN_SCHEMA = "t5gemma2-local-rs-sft-pilot-v1"
JOURNAL_SCHEMA = "t5gemma2-local-rs-sft-pilot-journal-v1"
TARGET_SCHEMA = "t5gemma2-local-rs-sft-target-v1"
REPORT_SCHEMA = "t5gemma2-local-rs-sft-pilot-report-v1"
SCHEDULE_SCHEMA = "t5gemma2-local-rs-sft-matched-schedule-v1"
DEFAULT_PRODUCTION_TARGET_FLOOR = 200
CHECKPOINT_LOADER_COMPATIBILITY: Mapping[str, Any] | None = None
_HEX_SHA256 = frozenset("0123456789abcdef")
_FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "tests",
        "acceptance_tests",
        "hidden_tests",
        "holdback_tests",
        "reward_holdback_tests",
    }
)
_PRIVATE_PAYLOAD_KEYS = frozenset(
    {
        "diagnostic",
        "feedback_tests",
        "reward_holdback_tests",
        "holdback_tests",
        "private_tests",
        "private_diagnostic",
        "holdback_diagnostic",
    }
)


@dataclass(frozen=True)
class PilotTask:
    task_id: str
    source: str
    source_sha256: str
    visible_tests: str
    gold_target: str
    gold_target_sha256: str
    f2_row: dict[str, Any]
    split_binding_sha256: str


@dataclass(frozen=True)
class PrivateGate:
    task_id: str
    tests: str
    split_binding_sha256: str


@dataclass(frozen=True)
class Evaluation:
    compiled: bool
    passed: bool
    diagnostic: str


GenerateFn = Callable[[str, int, int], list[dict[str, Any]]]
EvaluateFn = Callable[[str, str, str], Evaluation]


def _read_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label}: no rows")
    return rows


def _pin_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
    allow_unpinned: bool,
) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    expected = str(expected_sha256 or "").strip().lower()
    if not expected and not allow_unpinned:
        raise ValueError(f"{label} requires an expected SHA-256")
    if expected and (
        len(expected) != 64
        or any(char not in _HEX_SHA256 for char in expected)
        or observed != expected
    ):
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return resolved, observed


def _target_source(row: Mapping[str, Any], task_id: str) -> str:
    target = str(
        row.get("supervised_target")
        or row.get("dart_source")
        or row.get("source")
        or ""
    ).strip()
    if not target:
        raise ValueError(f"{task_id}: missing original gold target")
    return target


def _split_binding(row: Mapping[str, Any], task_id: str) -> str:
    if str(row.get("schema") or "") != SPLIT_SCHEMA:
        raise ValueError(f"{task_id}: private split schema mismatch")
    integer_fields = ("case_count", "visible_count", "holdback_count")
    if any(type(row.get(field)) is not int for field in integer_fields):
        raise ValueError(f"{task_id}: private split counts are malformed")
    case_count = int(row["case_count"])
    visible_count = int(row["visible_count"])
    holdback_count = int(row["holdback_count"])
    visible = row.get("visible_case_indices")
    holdback = row.get("holdback_case_indices")
    if (
        case_count < 2
        or visible_count <= 0
        or holdback_count <= 0
        or visible_count + holdback_count != case_count
        or not isinstance(visible, list)
        or not isinstance(holdback, list)
        or any(type(index) is not int for index in [*visible, *holdback])
        or len(visible) != visible_count
        or len(holdback) != holdback_count
        or set(visible) & set(holdback)
        or set(visible) | set(holdback) != set(range(case_count))
    ):
        raise ValueError(f"{task_id}: private split membership is incoherent")
    tests_sha = str(row.get("tests_sha256") or "").strip().lower()
    if len(tests_sha) != 64 or any(char not in _HEX_SHA256 for char in tests_sha):
        raise ValueError(f"{task_id}: source tests digest is malformed")
    return canonical_sha256(
        {
            "tests_sha256": tests_sha,
            "case_count": case_count,
            "visible_count": visible_count,
            "holdback_count": holdback_count,
            "visible_case_indices": visible,
            "holdback_case_indices": holdback,
        }
    )


def load_pilot_inputs(
    *,
    rollout_file: str | Path,
    f2_jsonl: str | Path,
    private_holdback: str | Path,
    expected_rollout_sha256: str = "",
    expected_f2_sha256: str = "",
    expected_private_holdback_sha256: str = "",
    allow_unpinned_inputs: bool = False,
) -> tuple[list[PilotTask], dict[str, PrivateGate], dict[str, Any]]:
    """Load and bind public generation inputs to the complementary private gate."""

    rollout_path, rollout_sha = _pin_file(
        rollout_file,
        expected_rollout_sha256,
        label="visible rollout",
        allow_unpinned=allow_unpinned_inputs,
    )
    f2_path, f2_sha = _pin_file(
        f2_jsonl,
        expected_f2_sha256,
        label="F2 source",
        allow_unpinned=allow_unpinned_inputs,
    )
    holdback_path, holdback_sha = _pin_file(
        private_holdback,
        expected_private_holdback_sha256,
        label="private train holdback",
        allow_unpinned=allow_unpinned_inputs,
    )
    rollout_rows = _read_jsonl(rollout_path, "visible rollout")
    f2_rows = _read_jsonl(f2_path, "F2 source")
    private_rows = _read_jsonl(holdback_path, "private train holdback")
    if not (len(rollout_rows) == len(f2_rows) == len(private_rows)):
        raise ValueError("rollout, F2, and private holdback row counts differ")

    rollout_ids = [str(row.get("task_id") or "").strip() for row in rollout_rows]
    f2_ids = [str(row.get("task_id") or "").strip() for row in f2_rows]
    private_ids = [str(row.get("task_id") or "").strip() for row in private_rows]
    if (
        not all(rollout_ids)
        or len(set(rollout_ids)) != len(rollout_ids)
        or rollout_ids != f2_ids
        or rollout_ids != private_ids
    ):
        raise ValueError("rollout, F2, and private holdback identity/order differ")

    tasks: list[PilotTask] = []
    gates: dict[str, PrivateGate] = {}
    for rollout, f2, private in zip(
        rollout_rows, f2_rows, private_rows, strict=True
    ):
        task_id = str(rollout["task_id"])
        leaked = sorted(field for field in _FORBIDDEN_PUBLIC_FIELDS if field in rollout)
        if leaked:
            raise ValueError(f"{task_id}: private/test fields leaked to rollout: {leaked}")
        visible_tests = rollout.get("feedback_tests")
        private_visible = private.get("feedback_tests")
        holdback_tests = private.get("reward_holdback_tests")
        if (
            not isinstance(visible_tests, str)
            or not visible_tests.strip()
            or private_visible != visible_tests
            or not isinstance(holdback_tests, str)
            or not holdback_tests.strip()
        ):
            raise ValueError(f"{task_id}: visible/private harness binding failed")
        binding = _split_binding(private, task_id)
        if (
            rollout.get("verpo_feedback_split_schema") != SPLIT_SCHEMA
            or rollout.get("verpo_feedback_split_binding_sha256") != binding
            or len(extract_expect_spans(visible_tests))
            != int(private["visible_count"])
            or len(extract_expect_spans(holdback_tests))
            != int(private["holdback_count"])
        ):
            raise ValueError(f"{task_id}: visible/private split attestation differs")
        source = build_encoder_source(f2, task_id)
        gold = _target_source(rollout, task_id)
        tasks.append(
            PilotTask(
                task_id=task_id,
                source=source,
                source_sha256=sha256_text(source),
                visible_tests=visible_tests,
                gold_target=gold,
                gold_target_sha256=sha256_text(gold),
                f2_row=dict(f2),
                split_binding_sha256=binding,
            )
        )
        gates[task_id] = PrivateGate(
            task_id=task_id,
            tests=holdback_tests,
            split_binding_sha256=binding,
        )
    return tasks, gates, {
        "rollout": {"sha256": rollout_sha, "rows": len(tasks)},
        "f2": {"sha256": f2_sha, "rows": len(tasks)},
        # The private path is deliberately not returned or serialized.
        "private_holdback": {"sha256": holdback_sha, "rows": len(tasks)},
        "task_ids_sha256": canonical_sha256(rollout_ids),
        "private_path_serialized": False,
    }


def deterministic_pilot_indices(
    tasks: Sequence[PilotTask], *, seed: int, limit: int, offset: int = 0
) -> list[int]:
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("pilot tasks must be nonempty with unique identities")
    if seed < 0 or limit < 0 or offset < 0:
        raise ValueError("pilot seed/limit/offset must be non-negative")
    if offset >= len(tasks):
        raise ValueError("pilot offset must leave at least one scheduled task")
    ordered = sorted(
        range(len(tasks)),
        key=lambda index: canonical_sha256(
            {
                "schema": RUN_SCHEMA,
                "seed": seed,
                "task_id": tasks[index].task_id,
            }
        ),
    )
    remaining = ordered[offset:]
    count = len(remaining) if limit == 0 else min(limit, len(remaining))
    return remaining[:count]


def deterministic_residual_indices(
    tasks: Sequence[PilotTask],
    *,
    excluded_task_ids: set[str],
    seed: int,
    limit: int = 0,
) -> list[int]:
    """Schedule only source tasks not already privately accepted.

    This is deliberately a fresh ordering rather than an offset into a prior
    ordering.  A residual harvest may combine several non-contiguous earlier
    stages, so offsets cannot prove that every accepted source task is absent.
    ``limit=0`` means every unresolved task.
    """

    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("residual tasks must be nonempty with unique identities")
    if seed < 0 or limit < 0:
        raise ValueError("residual seed/limit must be non-negative")
    known = {task.task_id for task in tasks}
    unknown = sorted(excluded_task_ids - known)
    if unknown:
        raise ValueError(
            "excluded accepted task is absent from current pool: " + unknown[0]
        )
    unresolved = [
        index for index, task in enumerate(tasks) if task.task_id not in excluded_task_ids
    ]
    if not unresolved:
        raise ValueError("residual schedule has no unresolved tasks")
    ordered = sorted(
        unresolved,
        key=lambda index: canonical_sha256(
            {
                "schema": "t5gemma2-local-rs-sft-residual-schedule-v1",
                "seed": seed,
                "task_id": tasks[index].task_id,
            }
        ),
    )
    return ordered if limit == 0 else ordered[:limit]


def load_excluded_verified_tasks(
    *,
    reports: Sequence[str | Path],
    journals: Sequence[str | Path],
    expected_report_sha256: Sequence[str],
    expected_journal_sha256: Sequence[str],
    tasks: Sequence[PilotTask],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Load accepted source identities from pinned, completed local harvests.

    The report binds a durable journal hash and chain head.  We then fully
    validate the journal against the current public task identities before
    treating a ``selected_target`` as privately verified.  This keeps the
    residual stage from trusting a mutable JSONL or accidentally excluding a
    visible-only success.
    """

    sizes = {
        len(reports),
        len(journals),
        len(expected_report_sha256),
        len(expected_journal_sha256),
    }
    if not reports:
        return set(), []
    if len(sizes) != 1:
        raise ValueError("excluded harvest report/journal argument counts differ")
    by_id = {task.task_id: task for task in tasks}
    accepted_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    seen_report_paths: set[Path] = set()
    for position, (report_value, journal_value, report_digest, journal_digest) in enumerate(
        zip(reports, journals, expected_report_sha256, expected_journal_sha256, strict=True)
    ):
        report_path, observed_report = _pin_file(
            report_value,
            report_digest,
            label=f"excluded harvest report {position}",
            allow_unpinned=False,
        )
        journal_path, observed_journal = _pin_file(
            journal_value,
            journal_digest,
            label=f"excluded harvest journal {position}",
            allow_unpinned=False,
        )
        if report_path in seen_report_paths:
            raise ValueError("excluded harvest reports must be distinct")
        seen_report_paths.add(report_path)
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"excluded harvest report {position} is malformed") from exc
        if (
            not isinstance(report, Mapping)
            or report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or not isinstance(report.get("journal"), Mapping)
        ):
            raise ValueError(f"excluded harvest report {position} is not a completed local harvest")
        report_journal = report["journal"]
        chain_path = Path(str(journal_path) + ".chain-head.json")
        if (
            report_journal.get("sha256") != observed_journal
            or not chain_path.is_file()
            or report_journal.get("chain_head_sha256") != sha256_file(chain_path)
        ):
            raise ValueError(f"excluded harvest report {position} does not bind its journal")
        events = load_journal(journal_path)
        if not events or not isinstance(events[0].get("contract"), Mapping):
            raise ValueError(f"excluded harvest journal {position} has no run contract")
        prior_contract = events[0]["contract"]
        task_ids = [
            str(event.get("task_id") or "")
            for event in events
            if event.get("event") == "task_terminal"
        ]
        if not task_ids or len(task_ids) != len(set(task_ids)) or any(
            task_id not in by_id for task_id in task_ids
        ):
            raise ValueError(f"excluded harvest journal {position} task identities are invalid")
        prior_tasks = [by_id[task_id] for task_id in task_ids]
        terminals, complete = validate_journal_state(
            events, contract=prior_contract, scheduled_tasks=prior_tasks
        )
        if not complete:
            raise ValueError(f"excluded harvest journal {position} is incomplete")
        pilot = report.get("pilot")
        if (
            not isinstance(pilot, Mapping)
            or int(pilot.get("tasks", -1)) != len(terminals)
        ):
            raise ValueError(f"excluded harvest report {position} task accounting differs")
        accepted_here = {
            str(terminal["task_id"])
            for terminal in terminals
            if terminal.get("selected_target") is not None
        }
        if int(pilot.get("accepted_unique_targets", -1)) != len(accepted_here):
            raise ValueError(f"excluded harvest report {position} acceptance accounting differs")
        overlap = accepted_ids & accepted_here
        if overlap:
            raise ValueError(
                "accepted source task appears in multiple excluded harvests: "
                + sorted(overlap)[0]
            )
        accepted_ids.update(accepted_here)
        records.append(
            {
                "report_sha256": observed_report,
                "journal_sha256": observed_journal,
                "journal_chain_head_sha256": sha256_file(chain_path),
                "tasks": len(terminals),
                "accepted_unique_targets": len(accepted_here),
                "accepted_task_ids_sha256": canonical_sha256(sorted(accepted_here)),
            }
        )
    return accepted_ids, records


def load_excluded_api_verified_tasks(
    *,
    reports: Sequence[str | Path],
    journals: Sequence[str | Path],
    target_files: Sequence[str | Path],
    expected_report_sha256: Sequence[str],
    expected_journal_sha256: Sequence[str],
    expected_target_sha256: Sequence[str],
    tasks: Sequence[PilotTask],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Load a sealed API hard-target ledger without trusting provider output alone.

    An API stage is accepted only when its immutable completed report binds a
    hash-chained journal *and* binds the separately pinned direct target file.
    The target rows themselves carry the binary private-gate result.  This is
    intentionally optional so a later sealed Opus report can be appended
    without changing how already-pinned Sonnet stages are interpreted.
    """

    sizes = {
        len(reports), len(journals), len(target_files), len(expected_report_sha256),
        len(expected_journal_sha256), len(expected_target_sha256),
    }
    if not reports:
        return set(), []
    if len(sizes) != 1:
        raise ValueError("excluded API report/journal/target argument counts differ")
    known = {task.task_id for task in tasks}
    accepted_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for position, values in enumerate(
        zip(
            reports, journals, target_files, expected_report_sha256,
            expected_journal_sha256, expected_target_sha256, strict=True,
        )
    ):
        report_value, journal_value, target_value, report_digest, journal_digest, target_digest = values
        report_path, observed_report = _pin_file(
            report_value, report_digest, label=f"excluded API report {position}", allow_unpinned=False
        )
        journal_path, observed_journal = _pin_file(
            journal_value, journal_digest, label=f"excluded API journal {position}", allow_unpinned=False
        )
        target_path, observed_target = _pin_file(
            target_value, target_digest, label=f"excluded API targets {position}", allow_unpinned=False
        )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"excluded API report {position} is malformed") from exc
        journal = report.get("journal") if isinstance(report, Mapping) else None
        outputs = report.get("outputs") if isinstance(report, Mapping) else None
        verification = report.get("verification") if isinstance(report, Mapping) else None
        direct = outputs.get("direct_targets") if isinstance(outputs, Mapping) else None
        head_path = Path(str(journal_path) + ".chain-head.json")
        if (
            not isinstance(report, Mapping)
            or report.get("schema") != "t5gemma2-api-rs-sft-rescue-report-v1"
            or report.get("status") != "complete"
            or not isinstance(journal, Mapping)
            or journal.get("sha256") != observed_journal
            or not head_path.is_file()
            or journal.get("chain_head_sha256") != sha256_file(head_path)
            or not isinstance(direct, Mapping)
            or direct.get("sha256") != observed_target
            or not isinstance(verification, Mapping)
        ):
            raise ValueError(f"excluded API report {position} does not bind sealed evidence")
        # Validates the durable chain and avoids using a target file whose
        # companion provider transaction is partial or rewritten.
        if not load_journal(journal_path):
            raise ValueError(f"excluded API journal {position} is empty")
        rows = _read_jsonl(target_path, f"excluded API targets {position}")
        task_ids: list[str] = []
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if (
                row.get("schema") != "t5gemma2-api-rs-sft-direct-target-v1"
                or not task_id
                or task_id not in known
                or row.get("private_gate_passed") is not True
                or row.get("production_floor_eligible") is not True
            ):
                raise ValueError(f"excluded API targets {position} contains invalid target evidence")
            task_ids.append(task_id)
        if (
            len(task_ids) != len(set(task_ids))
            or direct.get("rows") != len(rows)
            or verification.get("verified_unique_hard_targets") != len(rows)
        ):
            raise ValueError(f"excluded API report {position} target accounting differs")
        overlap = accepted_ids & set(task_ids)
        if overlap:
            raise ValueError("accepted source task appears in multiple API ledgers: " + sorted(overlap)[0])
        accepted_ids.update(task_ids)
        records.append(
            {
                "report_sha256": observed_report,
                "journal_sha256": observed_journal,
                "journal_chain_head_sha256": sha256_file(head_path),
                "targets_sha256": observed_target,
                "accepted_unique_targets": len(task_ids),
                "accepted_task_ids_sha256": canonical_sha256(sorted(task_ids)),
            }
        )
    return accepted_ids, records


def derived_seed(
    seed: int, *, task_position: int, phase: str, parent_position: int = -1
) -> int:
    if seed < 0 or task_position < 0 or not phase:
        raise ValueError("invalid deterministic seed coordinates")
    digest = canonical_sha256(
        {
            "schema": RUN_SCHEMA,
            "seed": seed,
            "task_position": task_position,
            "phase": phase,
            "parent_position": parent_position,
        }
    )
    return int(digest[:16], 16) % (2**63 - 1)


def _normalize_generated(
    values: Sequence[Mapping[str, Any]], *, count: int, origin: str
) -> list[dict[str, Any]]:
    if len(values) != count:
        raise ValueError(f"{origin}: generator returned {len(values)} rows, wanted {count}")
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        code = str(value.get("text") or value.get("code") or "").strip()
        record = {
            key: item
            for key, item in dict(value).items()
            if key not in {"text", "code"}
        }
        normalized.append(
            {
                "origin": origin,
                "sample_index": index,
                "code": code,
                "code_sha256": sha256_text(code),
                "generation": record,
            }
        )
    return normalized


def _evaluate_many(
    candidates: Sequence[Mapping[str, Any]],
    *,
    tests: str,
    task_id: str,
    phase: str,
    evaluate: EvaluateFn,
    workers: int,
    retain_compiler_feedback: bool,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("evaluation workers must be positive")

    def one(position: int) -> tuple[int, Evaluation]:
        candidate = candidates[position]
        return position, evaluate(
            str(candidate["code"]),
            tests,
            f"{task_id}-{phase}-{position}",
        )

    scored: list[Evaluation | None] = [None] * len(candidates)
    if workers == 1:
        results = [one(index) for index in range(len(candidates))]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates) or 1)) as pool:
            results = list(pool.map(one, range(len(candidates))))
    for position, result in results:
        if not isinstance(result, Evaluation):
            raise TypeError("candidate evaluator must return Evaluation")
        scored[position] = result

    output: list[dict[str, Any]] = []
    for candidate, result in zip(candidates, scored, strict=True):
        assert result is not None
        record = {
            **dict(candidate),
            "visible": {
                "compiled": bool(result.compiled),
                "passed": bool(result.compiled and result.passed),
            },
        }
        if retain_compiler_feedback and not result.compiled and str(candidate["code"]):
            safe = sanitize_compiler_diagnostic(result.diagnostic)
            record["safe_compiler_feedback"] = safe
            record["safe_compiler_feedback_sha256"] = sha256_text(safe)
        output.append(record)
    return output


def _private_gate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    gate: PrivateGate,
    task_id: str,
    evaluate: EvaluateFn,
    workers: int,
) -> list[dict[str, Any]]:
    """Return binary gate outcomes and intentionally discard every diagnostic."""

    def one(position: int) -> tuple[int, bool]:
        result = evaluate(
            str(candidates[position]["code"]),
            gate.tests,
            f"{task_id}-private-gate-{position}",
        )
        return position, bool(result.compiled and result.passed)

    outcomes = [False] * len(candidates)
    if workers == 1:
        results = [one(index) for index in range(len(candidates))]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates) or 1)) as pool:
            results = list(pool.map(one, range(len(candidates))))
    for position, passed in results:
        outcomes[position] = passed
    return [
        {
            "candidate_sha256": str(candidate["code_sha256"]),
            "origin": str(candidate["origin"]),
            "private_gate_passed": outcomes[index],
        }
        for index, candidate in enumerate(candidates)
    ]


def process_task(
    *,
    task: PilotTask,
    gate: PrivateGate,
    task_position: int,
    seed: int,
    base_samples: int,
    repair_samples: int,
    max_repair_parents: int,
    evaluation_workers: int,
    generate: GenerateFn,
    evaluate: EvaluateFn,
) -> dict[str, Any]:
    """Produce one terminal event with strict generation-before-holdback ordering."""

    if gate.task_id != task.task_id or gate.split_binding_sha256 != task.split_binding_sha256:
        raise ValueError(f"{task.task_id}: private gate identity/binding mismatch")
    if base_samples <= 0:
        raise ValueError("base sample count must be positive")
    if repair_samples < 0 or max_repair_parents < 0:
        raise ValueError("repair counts must be non-negative")
    repair_enabled = repair_samples > 0 or max_repair_parents > 0
    if repair_enabled and min(repair_samples, max_repair_parents) <= 0:
        raise ValueError(
            "repair_samples and max_repair_parents must both be zero or both positive"
        )

    base = _normalize_generated(
        generate(
            task.source,
            base_samples,
            derived_seed(seed, task_position=task_position, phase="base"),
        ),
        count=base_samples,
        origin="base",
    )
    base = _evaluate_many(
        base,
        tests=task.visible_tests,
        task_id=task.task_id,
        phase="base-visible",
        evaluate=evaluate,
        workers=evaluation_workers,
        retain_compiler_feedback=True,
    )

    repair_groups: list[dict[str, Any]] = []
    # Compiler repair is a rescue for a flat all-zero visible group.  A
    # holdback result is not available yet and therefore cannot influence it.
    if repair_enabled and not any(
        candidate["visible"]["passed"] for candidate in base
    ):
        repairable = [
            candidate
            for candidate in base
            if not candidate["visible"]["compiled"]
            and str(candidate["code"]).strip()
            and "safe_compiler_feedback" in candidate
        ]
        if repairable:
            count = min(max_repair_parents, len(repairable))
            selected = max_min_diverse_indices(
                [str(candidate["code"]) for candidate in repairable], count
            )
            for parent_position, repairable_index in enumerate(selected):
                parent = repairable[repairable_index]
                context = build_compiler_repair_context(
                    task_id=task.task_id,
                    source_sha256=task.source_sha256,
                    candidate=str(parent["code"]),
                    diagnostic=str(parent["safe_compiler_feedback"]),
                    compiled=False,
                )
                repair_source = task.source + "\n" + str(context["text"])
                repaired = _normalize_generated(
                    generate(
                        repair_source,
                        repair_samples,
                        derived_seed(
                            seed,
                            task_position=task_position,
                            phase="repair",
                            parent_position=parent_position,
                        ),
                    ),
                    count=repair_samples,
                    origin="compiler_repair",
                )
                repaired = _evaluate_many(
                    repaired,
                    tests=task.visible_tests,
                    task_id=task.task_id,
                    phase=f"repair-{parent_position}-visible",
                    evaluate=evaluate,
                    workers=evaluation_workers,
                    retain_compiler_feedback=False,
                )
                repair_groups.append(
                    {
                        "parent_candidate_sha256": parent["code_sha256"],
                        "repair_context_sha256": context["text_sha256"],
                        "compiler_feedback_sha256": context["payload"][
                            "compiler_feedback_sha256"
                        ],
                        "candidates": repaired,
                    }
                )

    # This line is the privacy boundary: all model calls for this task are
    # complete.  From here on, holdback information may only reject transfer.
    all_candidates = [*base]
    for group in repair_groups:
        all_candidates.extend(group["candidates"])
    visible_passes: list[dict[str, Any]] = []
    seen_code: set[str] = set()
    for candidate in all_candidates:
        digest = str(candidate["code_sha256"])
        if candidate["visible"]["passed"] and digest not in seen_code:
            seen_code.add(digest)
            visible_passes.append(candidate)
    gate_results = _private_gate(
        visible_passes,
        gate=gate,
        task_id=task.task_id,
        evaluate=evaluate,
        workers=evaluation_workers,
    )
    selected_target: dict[str, Any] | None = None
    for candidate, result in zip(visible_passes, gate_results, strict=True):
        if result["private_gate_passed"]:
            selected_target = {
                "schema": TARGET_SCHEMA,
                "task_id": task.task_id,
                "code": candidate["code"],
                "code_sha256": candidate["code_sha256"],
                "origin": candidate["origin"],
                "source_sha256": task.source_sha256,
                "visible_passed": True,
                "private_gate_passed": True,
            }
            break
    event = {
        "event": "task_terminal",
        "schema": JOURNAL_SCHEMA,
        "task_position": task_position,
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "split_binding_sha256": task.split_binding_sha256,
        "base_candidates": base,
        "repair_groups": repair_groups,
        "visible_unique_passes": len(visible_passes),
        "private_gate_results": gate_results,
        "selected_target": selected_target,
        "all_generation_completed_before_private_gate": True,
        "private_feedback_serialized_to_model": False,
        "holdback_failure_triggers_generation": False,
    }
    _assert_no_private_payload(event)
    return event


def _assert_no_private_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        leaked = _PRIVATE_PAYLOAD_KEYS & set(map(str, value))
        if leaked:
            raise ValueError(f"private payload key serialized: {sorted(leaked)}")
        for child in value.values():
            _assert_no_private_payload(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _assert_no_private_payload(child)


def validate_journal_state(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    scheduled_tasks: Sequence[PilotTask],
) -> tuple[list[dict[str, Any]], bool]:
    if not events:
        return [], False
    sampling = contract.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("RS-SFT run contract sampling is missing")
    try:
        expected_base_samples = int(sampling["base_samples"])
        expected_repair_samples = int(sampling["repair_samples"])
        expected_repair_parents = int(sampling["max_repair_parents"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RS-SFT run contract sampling is malformed") from exc
    if (
        expected_base_samples <= 0
        or expected_repair_samples < 0
        or expected_repair_parents < 0
        or ((expected_repair_samples == 0) != (expected_repair_parents == 0))
    ):
        raise ValueError("RS-SFT run contract sampling is incoherent")
    repair_enabled = expected_repair_samples > 0
    header = events[0]
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("RS-SFT journal header differs from the exact run")
    terminals: list[dict[str, Any]] = []
    complete = False
    for event in events[1:]:
        _assert_no_private_payload(event)
        if event.get("event") == "complete":
            if complete or len(terminals) != len(scheduled_tasks):
                raise ValueError("RS-SFT journal completion is early/duplicate")
            if (
                event.get("schema") != JOURNAL_SCHEMA
                or int(event.get("tasks", -1)) != len(scheduled_tasks)
                or event.get("terminal_task_ids_sha256")
                != canonical_sha256([row["task_id"] for row in terminals])
            ):
                raise ValueError("RS-SFT journal completion digest differs")
            complete = True
            continue
        if complete or event.get("event") != "task_terminal":
            raise ValueError("RS-SFT journal contains an invalid event ordering")
        position = len(terminals)
        expected = scheduled_tasks[position]
        if (
            event.get("schema") != JOURNAL_SCHEMA
            or event.get("task_position") != position
            or event.get("task_id") != expected.task_id
            or event.get("source_sha256") != expected.source_sha256
            or event.get("split_binding_sha256")
            != expected.split_binding_sha256
            or event.get("all_generation_completed_before_private_gate") is not True
            or event.get("private_feedback_serialized_to_model") is not False
            or event.get("holdback_failure_triggers_generation") is not False
        ):
            raise ValueError(f"RS-SFT terminal {position} differs from schedule")
        candidate_by_sha: dict[str, Mapping[str, Any]] = {}
        visible_pass_sha: list[str] = []
        base_candidates = event.get("base_candidates")
        if (
            not isinstance(base_candidates, list)
            or len(base_candidates) != expected_base_samples
            or any(
                not isinstance(candidate, Mapping)
                or candidate.get("origin") != "base"
                for candidate in base_candidates
            )
        ):
            raise ValueError(
                f"RS-SFT terminal {position} base candidate count/origin differs"
            )
        candidate_groups = [base_candidates]
        repair_groups = event.get("repair_groups")
        if not isinstance(repair_groups, list):
            raise ValueError(f"RS-SFT terminal {position} repair groups are invalid")
        if (
            (not repair_enabled and repair_groups)
            or len(repair_groups) > expected_repair_parents
        ):
            raise ValueError(
                f"RS-SFT terminal {position} repair mode/count differs"
            )
        for group in repair_groups:
            if not isinstance(group, Mapping):
                raise ValueError(
                    f"RS-SFT terminal {position} repair group is invalid"
                )
            repair_candidates = group.get("candidates")
            if (
                not isinstance(repair_candidates, list)
                or len(repair_candidates) != expected_repair_samples
                or any(
                    not isinstance(candidate, Mapping)
                    or candidate.get("origin") != "compiler_repair"
                    for candidate in repair_candidates
                )
            ):
                raise ValueError(
                    f"RS-SFT terminal {position} repair candidate count/origin differs"
                )
            candidate_groups.append(repair_candidates)
        for group in candidate_groups:
            if not isinstance(group, list):
                raise ValueError(
                    f"RS-SFT terminal {position} candidate group is invalid"
                )
            for candidate in group:
                if not isinstance(candidate, Mapping):
                    raise ValueError(
                        f"RS-SFT terminal {position} candidate is invalid"
                    )
                code = str(candidate.get("code") or "")
                digest = str(candidate.get("code_sha256") or "")
                visible = candidate.get("visible")
                if (
                    sha256_text(code) != digest
                    or not isinstance(visible, Mapping)
                    or type(visible.get("compiled")) is not bool
                    or type(visible.get("passed")) is not bool
                    or (visible.get("passed") and not visible.get("compiled"))
                ):
                    raise ValueError(
                        f"RS-SFT terminal {position} candidate evidence is invalid"
                    )
                candidate_by_sha.setdefault(digest, candidate)
                if visible.get("passed") and digest not in visible_pass_sha:
                    visible_pass_sha.append(digest)
        gate_results = event.get("private_gate_results")
        if (
            not isinstance(gate_results, list)
            or int(event.get("visible_unique_passes", -1))
            != len(visible_pass_sha)
            or len(gate_results) != len(visible_pass_sha)
        ):
            raise ValueError(
                f"RS-SFT terminal {position} private gate accounting differs"
            )
        for digest, result in zip(visible_pass_sha, gate_results, strict=True):
            if (
                not isinstance(result, Mapping)
                or result.get("candidate_sha256") != digest
                or type(result.get("private_gate_passed")) is not bool
            ):
                raise ValueError(
                    f"RS-SFT terminal {position} private gate result is invalid"
                )
        selected = event.get("selected_target")
        if selected is not None and (
            not isinstance(selected, Mapping)
            or selected.get("schema") != TARGET_SCHEMA
            or selected.get("task_id") != expected.task_id
            or selected.get("source_sha256") != expected.source_sha256
            or selected.get("visible_passed") is not True
            or selected.get("private_gate_passed") is not True
            or sha256_text(str(selected.get("code") or ""))
            != selected.get("code_sha256")
            or selected.get("code_sha256") not in candidate_by_sha
            or not any(
                result.get("candidate_sha256") == selected.get("code_sha256")
                and result.get("private_gate_passed") is True
                for result in gate_results
            )
        ):
            raise ValueError(f"RS-SFT terminal {position} target is not verified")
        terminals.append(dict(event))
    return terminals, complete


def _atomic_write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    payload = b"".join(
        (
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_matched_training_rows(
    *,
    all_tasks: Sequence[PilotTask],
    terminals: Sequence[Mapping[str, Any]],
    gold_replay_ratio: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Build one source-matched intervention/control with unique task IDs."""

    if gold_replay_ratio < 0:
        raise ValueError("gold replay ratio must be non-negative")
    tasks_by_id = {task.task_id: task for task in all_tasks}
    accepted: dict[str, Mapping[str, Any]] = {}
    for terminal in terminals:
        selected = terminal.get("selected_target")
        if selected is None:
            continue
        task_id = str(terminal.get("task_id") or "")
        if task_id in accepted or task_id not in tasks_by_id:
            raise ValueError("accepted targets contain a duplicate/unknown task")
        accepted[task_id] = selected

    gold_pool = [task for task in all_tasks if task.task_id not in accepted]
    gold_pool.sort(
        key=lambda task: canonical_sha256(
            {
                "schema": SCHEDULE_SCHEMA,
                "seed": seed,
                "kind": "gold_replay",
                "task_id": task.task_id,
            }
        )
    )
    requested_gold = gold_replay_ratio * len(accepted)
    selected_gold = gold_pool[:requested_gold]
    scheduled: list[tuple[str, PilotTask]] = [
        ("repair", tasks_by_id[task_id]) for task_id in accepted
    ] + [("gold_replay", task) for task in selected_gold]
    scheduled.sort(
        key=lambda item: canonical_sha256(
            {
                "schema": SCHEDULE_SCHEMA,
                "seed": seed,
                "kind": item[0],
                "task_id": item[1].task_id,
            }
        )
    )

    intervention: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    f2: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    repairs_f2: list[dict[str, Any]] = []
    for position, (kind, task) in enumerate(scheduled):
        target = (
            str(accepted[task.task_id]["code"])
            if kind == "repair"
            else task.gold_target
        )
        intervention.append({"task_id": task.task_id, "dart_source": target})
        control.append({"task_id": task.task_id, "dart_source": task.gold_target})
        f2.append(dict(task.f2_row))
        schedule.append(
            {
                "schema": SCHEDULE_SCHEMA,
                "position": position,
                "task_id": task.task_id,
                "kind": kind,
                "source_sha256": task.source_sha256,
                "intervention_target_sha256": sha256_text(target),
                "control_target_sha256": task.gold_target_sha256,
            }
        )
        if kind == "repair":
            repairs.append({"task_id": task.task_id, "dart_source": target})
            repairs_f2.append(dict(task.f2_row))
    if [row["task_id"] for row in intervention] != [row["task_id"] for row in f2]:
        raise AssertionError("matched intervention/F2 order drifted")
    if [row["task_id"] for row in intervention] != [row["task_id"] for row in control]:
        raise AssertionError("matched intervention/control order drifted")
    return {
        "repairs": repairs,
        "repairs_f2": repairs_f2,
        "intervention": intervention,
        "control": control,
        "matched_f2": f2,
        "schedule": schedule,
    }


def _runtime_evaluator(
    *, timeout: int, stability_runs: int
) -> EvaluateFn:
    def evaluate(code: str, tests: str, slot: str) -> Evaluation:
        compiled, passed, diagnostic, _ = evaluate_dart_jit_tests_detail(
            code,
            tests,
            slot,
            timeout=timeout,
            stability_runs=stability_runs,
        )
        return Evaluation(bool(compiled), bool(passed), str(diagnostic or ""))

    return evaluate


def _runtime_generator(
    *,
    model: Any,
    tokenizer: Any,
    max_source_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
) -> GenerateFn:
    decoder_start, pad_id, eos_ids = _decoder_special_ids(model, tokenizer)
    device = torch.device("cuda")

    def generate(source: str, count: int, seed: int) -> list[dict[str, Any]]:
        input_ids, attention_mask = _encode_source(
            tokenizer,
            source,
            max_source_tokens=max_source_tokens,
            device=device,
        )
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        output: list[dict[str, Any]] = []
        for batch_start in range(0, count, generation_batch_size):
            batch_count = min(generation_batch_size, count - batch_start)
            batch = generate_candidate_batch(
                model=model,
                tokenizer=tokenizer,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_start=decoder_start,
                pad_id=pad_id,
                eos_ids=eos_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed + batch_start,
                count=batch_count,
            )
            for item in batch:
                output.append(
                    {
                        **item,
                        "group_sample_index": len(output),
                        "encoder_tokens": int(input_ids.size(1)),
                    }
                )
        return output

    return generate


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("local T5Gemma 2 RS-SFT harvesting requires CUDA")
    validate_dart_binary()
    tasks, gates, input_record = load_pilot_inputs(
        rollout_file=args.rollout_file,
        f2_jsonl=args.f2_jsonl,
        private_holdback=args.private_holdback,
        expected_rollout_sha256=args.expected_rollout_sha256,
        expected_f2_sha256=args.expected_f2_sha256,
        expected_private_holdback_sha256=args.expected_private_holdback_sha256,
        allow_unpinned_inputs=args.allow_unpinned_inputs,
    )
    local_excluded_ids, local_exclusion_records = load_excluded_verified_tasks(
        reports=args.exclude_verified_report,
        journals=args.exclude_verified_journal,
        expected_report_sha256=args.expected_exclude_verified_report_sha256,
        expected_journal_sha256=args.expected_exclude_verified_journal_sha256,
        tasks=tasks,
    )
    api_excluded_ids, api_exclusion_records = load_excluded_api_verified_tasks(
        reports=args.exclude_verified_api_report,
        journals=args.exclude_verified_api_journal,
        target_files=args.exclude_verified_api_targets,
        expected_report_sha256=args.expected_exclude_verified_api_report_sha256,
        expected_journal_sha256=args.expected_exclude_verified_api_journal_sha256,
        expected_target_sha256=args.expected_exclude_verified_api_targets_sha256,
        tasks=tasks,
    )
    overlap = local_excluded_ids & api_excluded_ids
    if overlap:
        raise ValueError(
            "accepted source task appears in both local and API exclusion ledgers: "
            + sorted(overlap)[0]
        )
    excluded_ids = local_excluded_ids | api_excluded_ids
    exclusion_records = {
        "local_harvests": local_exclusion_records,
        "api_harvests": api_exclusion_records,
    }
    if excluded_ids:
        if args.pilot_offset != 0:
            raise ValueError("residual harvest must not use pilot_offset")
        indices = deterministic_residual_indices(
            tasks,
            excluded_task_ids=excluded_ids,
            seed=args.seed,
            limit=args.pilot_tasks,
        )
        schedule_mode = "residual_unresolved_only"
    else:
        indices = deterministic_pilot_indices(
            tasks,
            seed=args.seed,
            limit=args.pilot_tasks,
            offset=args.pilot_offset,
        )
        schedule_mode = "pilot_offset"
    scheduled_tasks = [tasks[index] for index in indices]
    checkpoint = Path(args.sft_checkpoint).expanduser().resolve()
    checkpoint_contract, model_record = _checkpoint_record(checkpoint, "sft")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "harvest.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": input_record,
        "excluded_verified_harvests": exclusion_records,
        "checkpoint": model_record,
        "checkpoint_contract_sha256": canonical_sha256(checkpoint_contract),
        "schedule": {
            "mode": schedule_mode,
            "seed": args.seed,
            "pilot_offset": args.pilot_offset,
            "pilot_tasks": len(scheduled_tasks),
            "task_ids_sha256": canonical_sha256(
                [task.task_id for task in scheduled_tasks]
            ),
            "excluded_accepted_task_count": len(excluded_ids),
            "excluded_accepted_task_ids_sha256": canonical_sha256(
                sorted(excluded_ids)
            ),
        },
        "sampling": {
            "base_samples": args.base_samples,
            "repair_samples": args.repair_samples,
            "max_repair_parents": args.max_repair_parents,
            "repair_enabled": bool(
                args.repair_samples and args.max_repair_parents
            ),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_source_tokens": args.max_source_tokens,
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": args.generation_batch_size,
        },
        "verification": {
            "timeout": args.timeout,
            "stability_runs": args.stability_runs,
            "visible_before_private": True,
            "private_gate_binary_only": True,
            "private_failure_triggers_generation": False,
        },
        "training_build": {
            "gold_replay_ratio": args.gold_replay_ratio,
            "production_min_unique_targets": args.production_min_unique_targets,
            "matched_gold_control": True,
            "original_f2_sources": True,
        },
        "no_frontier_api": True,
        "heldout_175_opened": False,
    }
    if CHECKPOINT_LOADER_COMPATIBILITY is not None:
        contract["checkpoint_loader_compatibility"] = dict(
            CHECKPOINT_LOADER_COMPATIBILITY
        )
    events = load_journal(journal_path)
    if not events:
        append_event(
            journal_path,
            {
                "event": "header",
                "schema": JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
        events = load_journal(journal_path)
    terminals, complete = validate_journal_state(
        events, contract=contract, scheduled_tasks=scheduled_tasks
    )

    if not complete:
        model, tokenizer, loaded_record = load_policy(
            checkpoint=checkpoint,
            arm="sft",
            bf16=args.bf16,
            attn_implementation=args.attn_implementation,
        )
        if loaded_record != model_record:
            raise ValueError("loaded model record differs from preflight")
        generate = _runtime_generator(
            model=model,
            tokenizer=tokenizer,
            max_source_tokens=args.max_source_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            generation_batch_size=args.generation_batch_size,
        )
        evaluate = _runtime_evaluator(
            timeout=args.timeout, stability_runs=args.stability_runs
        )
        for position in range(len(terminals), len(scheduled_tasks)):
            task = scheduled_tasks[position]
            event = process_task(
                task=task,
                gate=gates[task.task_id],
                task_position=position,
                seed=args.seed,
                base_samples=args.base_samples,
                repair_samples=args.repair_samples,
                max_repair_parents=args.max_repair_parents,
                evaluation_workers=args.evaluation_workers,
                generate=generate,
                evaluate=evaluate,
            )
            terminal = append_event(journal_path, event)
            terminals.append(terminal)
            print(
                json.dumps(
                    {
                        "task": position + 1,
                        "tasks": len(scheduled_tasks),
                        "task_id": task.task_id,
                        "visible_unique_passes": event["visible_unique_passes"],
                        "accepted": event["selected_target"] is not None,
                        "accepted_total": sum(
                            row.get("selected_target") is not None
                            for row in terminals
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        append_event(
            journal_path,
            {
                "event": "complete",
                "schema": JOURNAL_SCHEMA,
                "tasks": len(scheduled_tasks),
                "terminal_task_ids_sha256": canonical_sha256(
                    [row["task_id"] for row in terminals]
                ),
            },
        )
        events = load_journal(journal_path)
        terminals, complete = validate_journal_state(
            events, contract=contract, scheduled_tasks=scheduled_tasks
        )
    if not complete:
        raise RuntimeError("RS-SFT harvest journal did not complete")

    rows = build_matched_training_rows(
        all_tasks=tasks,
        terminals=terminals,
        gold_replay_ratio=args.gold_replay_ratio,
        seed=args.seed,
    )
    output_paths = {
        "repairs": output_dir / "rs_sft_repairs.jsonl",
        "repairs_f2": output_dir / "rs_sft_repairs_f2.jsonl",
        "intervention": output_dir / "rs_sft_intervention.jsonl",
        "control": output_dir / "gold_only_matched.jsonl",
        "matched_f2": output_dir / "matched_f2.jsonl",
        "schedule": output_dir / "matched_schedule.jsonl",
    }
    for name, path in output_paths.items():
        _atomic_write_jsonl(path, rows[name])
    accepted = len(rows["repairs"])
    repair_origins: dict[str, int] = {}
    for terminal in terminals:
        selected = terminal.get("selected_target")
        if selected is not None:
            origin = str(selected["origin"])
            repair_origins[origin] = repair_origins.get(origin, 0) + 1
    gold_hash_by_id = {
        task.task_id: task.gold_target_sha256 for task in tasks
    }
    exact_gold_matches = sum(
        sha256_text(row["dart_source"]) == gold_hash_by_id[row["task_id"]]
        for row in rows["repairs"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": canonical_sha256(contract),
        "inputs": input_record,
        "checkpoint": model_record,
        "pilot": {
            "schedule_mode": schedule_mode,
            "tasks": len(scheduled_tasks),
            "accepted_unique_targets": accepted,
            "acceptance_rate": accepted / len(scheduled_tasks),
            "production_min_unique_targets": args.production_min_unique_targets,
            "production_floor_met": accepted >= args.production_min_unique_targets,
            "accepted_by_origin": dict(sorted(repair_origins.items())),
            "accepted_exact_gold_matches": exact_gold_matches,
            "excluded_previously_verified_source_tasks": len(excluded_ids),
        },
        "matched_training": {
            "rows": len(rows["intervention"]),
            "repair_rows": accepted,
            "gold_replay_rows": len(rows["intervention"]) - accepted,
            "requested_gold_replay_ratio": args.gold_replay_ratio,
            "source_sequence_exactly_matched": True,
            "original_f2_sources": True,
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows[name]),
            }
            for name, path in output_paths.items()
        },
        "journal": journal_record(journal_path),
        "privacy_invariants": {
            "heldout_175_opened": False,
            "frontier_api_calls": False,
            "private_holdback_text_in_model_input": False,
            "private_holdback_text_in_outputs": False,
            "private_diagnostics_persisted": False,
            "all_generation_precedes_private_gate_per_task": True,
            "private_gate_can_only_reject_transfer": True,
        },
    }
    require_exact_or_write(output_dir / "harvest_report.json", report)
    print(
        json.dumps(
            {
                "tasks": len(scheduled_tasks),
                "accepted": accepted,
                "production_floor_met": report["pilot"]["production_floor_met"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--rollout_file", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--private_holdback", required=True)
    parser.add_argument("--expected_rollout_sha256", default="")
    parser.add_argument("--expected_f2_sha256", default="")
    parser.add_argument("--expected_private_holdback_sha256", default="")
    parser.add_argument("--allow_unpinned_inputs", action="store_true")
    parser.add_argument(
        "--exclude_verified_report",
        action="append",
        default=[],
        help="Completed local-harvest report to exclude its privately accepted tasks.",
    )
    parser.add_argument(
        "--exclude_verified_journal",
        action="append",
        default=[],
        help="Durable journal paired positionally with --exclude_verified_report.",
    )
    parser.add_argument(
        "--expected_exclude_verified_report_sha256",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--expected_exclude_verified_journal_sha256",
        action="append",
        default=[],
    )
    parser.add_argument("--exclude_verified_api_report", action="append", default=[])
    parser.add_argument("--exclude_verified_api_journal", action="append", default=[])
    parser.add_argument("--exclude_verified_api_targets", action="append", default=[])
    parser.add_argument(
        "--expected_exclude_verified_api_report_sha256", action="append", default=[]
    )
    parser.add_argument(
        "--expected_exclude_verified_api_journal_sha256", action="append", default=[]
    )
    parser.add_argument(
        "--expected_exclude_verified_api_targets_sha256", action="append", default=[]
    )
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pilot_tasks", type=int, default=200)
    parser.add_argument("--pilot_offset", type=int, default=0)
    parser.add_argument("--base_samples", type=int, default=4)
    parser.add_argument("--repair_samples", type=int, default=4)
    parser.add_argument("--max_repair_parents", type=int, default=2)
    parser.add_argument("--gold_replay_ratio", type=int, default=3)
    parser.add_argument(
        "--production_min_unique_targets",
        type=int,
        default=DEFAULT_PRODUCTION_TARGET_FLOOR,
    )
    parser.add_argument("--max_source_tokens", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--evaluation_workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn_implementation", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    positive = (
        "base_samples",
        "max_source_tokens",
        "max_new_tokens",
        "generation_batch_size",
        "evaluation_workers",
        "timeout",
        "stability_runs",
        "production_min_unique_targets",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error(
            "base sample, token, worker, timeout, stability, and floor values "
            "must be positive"
        )
    if any(
        getattr(args, name) < 0
        for name in (
            "pilot_tasks",
            "pilot_offset",
            "repair_samples",
            "max_repair_parents",
            "gold_replay_ratio",
            "seed",
        )
    ):
        parser.error(
            "pilot, repair, gold_replay_ratio, and seed values must be non-negative"
        )
    if (args.repair_samples == 0) != (args.max_repair_parents == 0):
        parser.error(
            "repair_samples and max_repair_parents must both be zero or both positive"
        )
    exclusion_lengths = {
        len(args.exclude_verified_report),
        len(args.exclude_verified_journal),
        len(args.expected_exclude_verified_report_sha256),
        len(args.expected_exclude_verified_journal_sha256),
    }
    if len(exclusion_lengths) != 1:
        parser.error(
            "each excluded harvest requires exactly one report, journal, and both SHA-256 values"
        )
    api_exclusion_lengths = {
        len(args.exclude_verified_api_report),
        len(args.exclude_verified_api_journal),
        len(args.exclude_verified_api_targets),
        len(args.expected_exclude_verified_api_report_sha256),
        len(args.expected_exclude_verified_api_journal_sha256),
        len(args.expected_exclude_verified_api_targets_sha256),
    }
    if len(api_exclusion_lengths) != 1:
        parser.error(
            "each excluded API harvest requires report, journal, direct targets, and all SHA-256 values"
        )
    if (args.exclude_verified_report or args.exclude_verified_api_report) and args.pilot_offset != 0:
        parser.error("residual exclusion mode requires --pilot_offset 0")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top_p must lie in (0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
