#!/usr/bin/env python3
"""Hash-sealed, direct-only API rescue for the typed T5Gemma local harvest.

This module deliberately reuses the already-tested provider transaction and
verification machinery in :mod:`t5gemma2_api_rs_sft_rescue`, but gives the
typed experiment its own schemas and output contract.  API models may see only
the opaque typed contract plus compressed enriched F2, one failed local
candidate, a sanitized compile diagnostic, and the separately pinned visible
TRAIN split.  Complete acceptance tests, the complementary holdback, gold
source, and held-out evaluation tasks are never provider inputs.

The three fail-closed phases are:

* ``kimi_initial``: OpenRouter Kimi K3, low reasoning, 2,048 output tokens,
  at most 50 tasks;
* ``kimi_retry``: the exact non-code/length-truncated subset of one completed
  initial cohort, 8,192 output tokens; and
* ``sonnet_residual``: Anthropic Sonnet 5, adaptive/high, 16,384 output tokens,
  after excluding all hash-pinned earlier successes.

Only direct, visible-and-private verified code targets are published.  Failed
programs, diagnostics, visible tests, reasoning, repair-conditioned rows, and
gold replay are not training outputs.  The journal is append-only and
hash-chained, with a call intent persisted before every provider request.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.graph_compile_at_k_antigravity import validate_dart_binary
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training.t5gemma2_local_rs_sft_pilot import (
    PilotTask,
    PrivateGate,
)
from scripts.preprocessing import build_t5gemma2_typed_api_visible_split as visible_split
from scripts.training import t5gemma2_enriched_sft as base_sft


RUN_SCHEMA = "t5gemma2-typed-api-rescue-cascade-run-v1"
JOURNAL_SCHEMA = "t5gemma2-typed-api-rescue-cascade-journal-v1"
DIRECT_TARGET_SCHEMA = "t5gemma2-typed-api-rescue-cascade-direct-target-v1"
DIRECT_MANIFEST_SCHEMA = "t5gemma2-typed-api-rescue-cascade-direct-manifest-v1"
REPORT_SCHEMA = "t5gemma2-typed-api-rescue-cascade-report-v1"
PLAN_SCHEMA = "t5gemma2-typed-api-rescue-cascade-plan-v1"
PRIOR_INDEX_SCHEMA = "t5gemma2-typed-api-rescue-cascade-prior-index-v1"
COHORT_DECISION_SCHEMA = "t5gemma2-typed-api-rescue-cascade-decision-v1"
EXISTING_DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-dataset-v1"
LOCAL_SOURCE_MODE = "completed_typed_local_direct_harvest"

PHASE_KIMI_INITIAL = "kimi_initial"
PHASE_KIMI_RETRY = "kimi_retry"
PHASE_SONNET_RESIDUAL = "sonnet_residual"
PHASES = (PHASE_KIMI_INITIAL, PHASE_KIMI_RETRY, PHASE_SONNET_RESIDUAL)

KIMI_MODEL = "moonshotai/kimi-k3"
SONNET_MODEL = "claude-sonnet-5"
KIMI_COHORT_SIZE = 50
KIMI_CONTINUE_MIN_YIELD = 8
KIMI_INITIAL_MAX_OUTPUT = 2048
KIMI_RETRY_MAX_OUTPUT = 8192
SONNET_MAX_OUTPUT = 16384
STABILITY_RUNS = 2
KNOWN_CONTAMINANT = "sigless_6b1dd0c6b6fc"

SYSTEM_PROMPT = """\
You repair one Dart function from an isolated synthetic programming benchmark
for authorized model-training research. Return exactly one complete Dart
source program and nothing else. Do not explain your reasoning. Do not emit
JSON. You may use one ```dart code fence, with no prose before or after it.
Preserve the opaque typed fn0 contract. Visible TRAIN-only checks may be used
for diagnosis; no reference source, complete private acceptance suite, private
holdback, or held-out evaluation task is provided.
"""

_HEX64 = re.compile(r"[0-9a-f]{64}")


class TypedSourceContext(tuple):
    """Validated source bundle returned by :func:`load_typed_source_context`."""

    __slots__ = ()

    def __new__(
        cls,
        tasks: Sequence[PilotTask],
        gates: Mapping[str, PrivateGate],
        scheduled_tasks: Sequence[PilotTask],
        terminals: Sequence[Mapping[str, Any]],
        input_record: Mapping[str, Any],
        source_journal_record: Mapping[str, Any],
        visible_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "TypedSourceContext":
        return tuple.__new__(
            cls,
            (
                list(tasks),
                dict(gates),
                list(scheduled_tasks),
                [dict(row) for row in terminals],
                dict(input_record),
                dict(source_journal_record),
                {
                    str(task_id): dict(record)
                    for task_id, record in (visible_metadata or {}).items()
                },
            ),
        )

    tasks = property(lambda self: self[0])
    gates = property(lambda self: self[1])
    scheduled_tasks = property(lambda self: self[2])
    terminals = property(lambda self: self[3])
    input_record = property(lambda self: self[4])
    source_journal_record = property(lambda self: self[5])
    visible_metadata = property(lambda self: self[6])


def _require_digest(value: str, label: str) -> str:
    value = str(value or "").strip().lower()
    if not _HEX64.fullmatch(value):
        raise ValueError(f"{label} requires an exact lowercase SHA-256")
    return value


def _pin(path_value: str | Path, expected: str, label: str) -> tuple[Path, str]:
    path = Path(path_value).expanduser().resolve()
    expected = _require_digest(expected, f"expected {label}")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return path, observed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}:{line_number}: malformed JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row is not an object")
            rows.append(value)
    return rows


@contextlib.contextmanager
def _typed_base_schemas() -> Iterator[None]:
    """Temporarily bind imported durable helpers to this profile's schemas."""

    names = {
        "RUN_SCHEMA": RUN_SCHEMA,
        "JOURNAL_SCHEMA": JOURNAL_SCHEMA,
        "DIRECT_TARGET_SCHEMA": DIRECT_TARGET_SCHEMA,
        "REPORT_SCHEMA": REPORT_SCHEMA,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
    }
    originals = {name: getattr(base, name) for name in names}
    try:
        for name, value in names.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def load_existing_225_exclusions(
    manifest_path: str | Path, expected_sha256: str
) -> tuple[frozenset[str], dict[str, Any]]:
    """Validate and return the task identities in the sealed 225-row corpus."""

    path, observed = _pin(
        manifest_path, expected_sha256, "existing typed direct dataset manifest"
    )
    manifest = _read_json(path, "existing typed direct dataset manifest")
    composition = manifest.get("composition")
    schedule = manifest.get("schedule")
    verification = manifest.get("full_acceptance_reverification")
    if (
        manifest.get("schema") != EXISTING_DATASET_SCHEMA
        or manifest.get("rows") != 225
        or not isinstance(composition, Mapping)
        or composition.get("verified_direct") != 225
        or composition.get("local_student_direct") != 141
        or composition.get("external_teacher_direct") != 84
        or composition.get("repair_conditioned") != 0
        or composition.get("gold_replay") != 0
        or manifest.get("heldout_overlap") != 0
        or manifest.get("known_contaminant_excluded") != KNOWN_CONTAMINANT
        or manifest.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or manifest.get("tests_model_visible") is not False
        or manifest.get("private_feedback_model_visible") is not False
        or manifest.get("repair_conditioned_prefixes_visible") is not False
        or manifest.get("production_floor_eligible") is not True
        or not isinstance(schedule, list)
        or len(schedule) != 225
        or not isinstance(verification, Mapping)
        or verification.get("rows") != 225
        or verification.get("passed") != 225
        or verification.get("stability_runs") != STABILITY_RUNS
    ):
        raise ValueError("existing typed direct dataset manifest contract differs")
    if canonical_sha256(schedule) != manifest.get("schedule_sha256"):
        raise ValueError("existing typed direct dataset schedule digest differs")
    task_ids = [str(row.get("source_task_id") or "") for row in schedule]
    if (
        any(not task_id for task_id in task_ids)
        or len(set(task_ids)) != 225
        or KNOWN_CONTAMINANT in task_ids
    ):
        raise ValueError("existing typed direct dataset identities are invalid")
    return frozenset(task_ids), {
        "schema": EXISTING_DATASET_SCHEMA,
        "sha256": observed,
        "rows": 225,
        "task_ids_sha256": canonical_sha256(sorted(task_ids)),
        "excluded_before_api_scheduling": True,
    }


def _load_completed_local_artifacts(
    args: argparse.Namespace,
    *,
    excluded_ids: frozenset[str],
    exclusion_record: Mapping[str, Any],
) -> tuple[
    list[Any],
    dict[str, Any],
    list[Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Reconstruct and validate a completed local ledger without a checkpoint.

    The checkpoint and the eleven predecessor ledgers were needed to *create*
    the local schedule.  They are not needed to consume a completed journal:
    the journal contract already seals their identities.  This loader rebuilds
    every typed source and private acceptance gate from the pinned gold inputs,
    rebuilds the exact 2,550-task schedule from the independently pinned 225-ID
    exclusion, validates every terminal against that schedule, and finally
    validates the report and published local targets against the journal.
    """

    from scripts.training import t5gemma2_typed_local_direct_harvest as local

    report_path, report_sha = _pin(
        args.local_harvest_report,
        args.expected_local_harvest_report_sha256,
        "typed local harvest report",
    )
    journal_path, journal_sha = _pin(
        args.pilot_journal,
        args.expected_local_harvest_journal_sha256,
        "typed local harvest journal",
    )
    targets_path, targets_sha = _pin(
        args.local_harvest_targets,
        args.expected_local_harvest_targets_sha256,
        "typed local harvest targets",
    )
    tasks, gates, rebuilt_input = local.load_harvest_inputs(
        gold_train_jsonl=args.gold_train_jsonl,
        gold_f2_jsonl=args.gold_f2_jsonl,
        expected_gold_train_sha256=args.expected_gold_train_sha256,
        expected_gold_f2_sha256=args.expected_gold_f2_sha256,
        expected_gold_rows=args.expected_gold_rows,
        heldout_jsonl=args.heldout_jsonl,
        expected_heldout_sha256=args.expected_heldout_sha256,
        expected_heldout_rows=args.expected_heldout_rows,
    )
    events = load_journal(journal_path)
    if not events or not isinstance(events[0].get("contract"), Mapping):
        raise ValueError("typed local harvest journal lacks a contract")
    contract = dict(events[0]["contract"])
    schedule_record = contract.get("schedule")
    previous = contract.get("previous_direct_exclusion")
    if (
        contract.get("schema") != local.RUN_SCHEMA
        or contract.get("input") != rebuilt_input
        or contract.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or contract.get("complete_acceptance_model_visible") is not False
        or contract.get("heldout_175_opened") is not False
        or contract.get("frontier_api_calls") is not False
        or not isinstance(schedule_record, Mapping)
        or schedule_record.get("schema") != local.SCHEDULE_SCHEMA
        or not isinstance(previous, Mapping)
        or previous.get("schema") != EXISTING_DATASET_SCHEMA
        or previous.get("rows") != 225
        or previous.get("task_ids_sha256")
        != exclusion_record["task_ids_sha256"]
    ):
        raise ValueError("typed local harvest run contract differs")
    scheduled = local.build_schedule(
        tasks,
        excluded_task_ids=set(excluded_ids),
        seed=int(schedule_record.get("seed", -1)),
    )
    if (
        schedule_record.get("scheduled_tasks") != len(scheduled)
        or schedule_record.get("task_ids_sha256")
        != canonical_sha256([task.task_id for task in scheduled])
        or schedule_record.get("source_sha256s_sha256")
        != canonical_sha256([task.source_sha256 for task in scheduled])
    ):
        raise ValueError("typed local harvest schedule binding differs")
    terminals, complete = local.validate_journal_state(
        events, contract=contract, scheduled_tasks=scheduled, gates=gates
    )
    if not complete:
        raise ValueError("typed local harvest journal is incomplete")

    report = _read_json(report_path, "typed local harvest report")
    outputs = report.get("outputs")
    direct_record = outputs.get("direct_targets") if isinstance(outputs, Mapping) else None
    report_journal = report.get("journal")
    if (
        report.get("schema") != local.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("production_floor_eligible") is not True
        or report.get("run_contract_sha256") != canonical_sha256(contract)
        or not isinstance(report_journal, Mapping)
        or report_journal.get("sha256") != journal_sha
        or not isinstance(direct_record, Mapping)
        or direct_record.get("sha256") != targets_sha
        or report.get("schedule", {}).get("tasks") != len(scheduled)
        or report.get("schedule", {}).get("task_ids_sha256")
        != canonical_sha256([task.task_id for task in scheduled])
        or report.get("privacy", {}).get("heldout_175_opened") is not False
        or report.get("privacy", {}).get("frontier_api_calls") is not False
    ):
        raise ValueError("typed local harvest report/artifact binding differs")
    target_rows = _read_jsonl(targets_path, "typed local harvest targets")
    selected_by_id = {
        str(row["task_id"]): row["selected_target"]
        for row in terminals
        if isinstance(row.get("selected_target"), Mapping)
    }
    target_ids: list[str] = []
    task_by_id = {task.task_id: task for task in scheduled}
    for row in target_rows:
        task_id = str(row.get("task_id") or "")
        selected = selected_by_id.get(task_id)
        code = str(row.get("dart_source") or "")
        if (
            row.get("schema") != local.TARGET_SCHEMA
            or task_id not in task_by_id
            or not isinstance(selected, Mapping)
            or row.get("source_sha256") != task_by_id[task_id].source_sha256
            or row.get("origin") != "local_student_direct"
            or row.get("full_acceptance_passed") is not True
            or row.get("stability_runs") != STABILITY_RUNS
            or row.get("repair_conditioned") is not False
            or row.get("gold_replay") is not False
            or local.sha256_text(code) != row.get("dart_source_sha256")
            or selected.get("code_sha256") != row.get("dart_source_sha256")
        ):
            raise ValueError("typed local direct target evidence differs")
        target_ids.append(task_id)
    if (
        len(target_ids) != len(set(target_ids))
        or direct_record.get("rows") != len(target_rows)
        or report.get("accepted", {}).get("unique_direct_targets")
        != len(target_rows)
        or report.get("accepted", {}).get("task_ids_sha256")
        != canonical_sha256(target_ids)
        or set(target_ids) != set(selected_by_id)
    ):
        raise ValueError("typed local direct target accounting differs")
    source_record = journal_record(journal_path)
    source_record.pop("path", None)
    source_record.pop("chain_head_path", None)
    source_record.update(
        {
            "schema": local.JOURNAL_SCHEMA,
            "run_contract_sha256": canonical_sha256(contract),
            "report_sha256": report_sha,
            "targets_sha256": targets_sha,
            "production_floor_eligible": True,
            "source_journal_modified": False,
        }
    )
    local_record = {
        **rebuilt_input,
        "report_sha256": report_sha,
        "journal_sha256": journal_sha,
        "targets_sha256": targets_sha,
        "accepted_direct_task_ids_sha256": canonical_sha256(target_ids),
    }
    return tasks, gates, scheduled, terminals, local_record, source_record


def load_typed_source_context(args: argparse.Namespace) -> TypedSourceContext:
    """Merge a sealed typed K=4 ledger with the permitted visible TRAIN split.

    The typed local loader owns validation of its complete private acceptance
    suite and local journal.  A full-clean 2,775-task split artifact supplies
    the separately attested visible provider checks.  Its private complement
    remains local and is included only by hash, never in a prompt.
    """

    public_by_id, public_record = load_full_visible_split(args)
    excluded_ids, exclusion_record = load_existing_225_exclusions(
        args.existing_direct_manifest,
        args.expected_existing_direct_manifest_sha256,
    )
    from scripts.training import t5gemma2_typed_local_direct_harvest as local

    loaded = local.load_completed_harvest_artifacts(
        report_path=args.local_harvest_report,
        expected_report_sha256=args.expected_local_harvest_report_sha256,
        journal_path=args.pilot_journal,
        expected_journal_sha256=args.expected_local_harvest_journal_sha256,
        targets_path=args.local_harvest_targets,
        expected_targets_sha256=args.expected_local_harvest_targets_sha256,
        gold_train_jsonl=args.gold_train_jsonl,
        expected_gold_train_sha256=args.expected_gold_train_sha256,
        gold_f2_jsonl=args.gold_f2_jsonl,
        expected_gold_f2_sha256=args.expected_gold_f2_sha256,
        heldout_jsonl=args.heldout_jsonl,
        expected_heldout_sha256=args.expected_heldout_sha256,
        expected_gold_rows=args.expected_gold_rows,
        expected_heldout_rows=args.expected_heldout_rows,
    )
    (
        typed_tasks,
        private_gates,
        scheduled_typed,
        terminals,
        local_record,
        source_journal_record,
    ) = loaded
    typed_by_id = {task.task_id: task for task in typed_tasks}
    if len(typed_by_id) != len(typed_tasks):
        raise ValueError("typed/public task identities are not unique")
    if set(typed_by_id) != set(public_by_id):
        raise ValueError("typed local and permitted visible split identities differ")
    scheduled_id_set = {task.task_id for task in scheduled_typed}
    if set(typed_by_id) - scheduled_id_set != set(excluded_ids):
        raise ValueError("typed local predecessor complement differs from sealed 225")
    local_record = {
        **dict(local_record),
        "existing_225_manifest": exclusion_record,
    }
    merged_by_id: dict[str, PilotTask] = {}
    merged_gates: dict[str, PrivateGate] = {}
    visible_metadata: dict[str, dict[str, Any]] = {}
    for task_id, typed in typed_by_id.items():
        public = public_by_id[task_id]
        gate = private_gates.get(task_id)
        if gate is None or not str(gate.tests).strip():
            raise ValueError(f"{task_id}: typed private complete-acceptance gate is absent")
        split_binding = canonical_sha256(
            {
                "typed_private_gate_binding": str(gate.tests_sha256),
                "permitted_visible_split_binding": public["split_binding_sha256"],
            }
        )
        merged_by_id[task_id] = PilotTask(
            task_id=task_id,
            source=str(typed.source),
            source_sha256=str(typed.source_sha256),
            visible_tests=str(public["visible_tests"]),
            gold_target=str(public["gold_target"]),
            gold_target_sha256=base.sha256_text(str(public["gold_target"])),
            f2_row=dict(typed.f2_row),
            split_binding_sha256=split_binding,
        )
        merged_gates[task_id] = PrivateGate(
            task_id=task_id,
            tests=str(gate.tests),
            split_binding_sha256=split_binding,
        )
        visible_metadata[task_id] = {
            "strategy": public["strategy"],
            "semantic_visible_cases": public["visible_count"],
            "split_binding_sha256": public["split_binding_sha256"],
        }
    scheduled = [merged_by_id[task.task_id] for task in scheduled_typed]
    scheduled_ids = [task.task_id for task in scheduled]
    terminal_ids = [str(row.get("task_id") or "") for row in terminals]
    if scheduled_ids != terminal_ids:
        raise ValueError("typed local schedule and terminal order differ")
    source_record = dict(source_journal_record)
    source_record.update(
        {
            "mode": LOCAL_SOURCE_MODE,
            "exploratory_prefix": False,
            "production_floor_eligible": True,
            "terminal_prefix_length": None,
            "source_journal_modified": False,
        }
    )
    input_record = {
        "typed_local_harvest": dict(local_record),
        "permitted_visible_train_split": public_record,
        "task_ids_sha256": canonical_sha256(list(public_by_id)),
        "complete_acceptance_text_serialized": False,
        "private_holdback_text_serialized": False,
    }
    return TypedSourceContext(
        list(merged_by_id.values()),
        merged_gates,
        scheduled,
        terminals,
        input_record,
        source_record,
        visible_metadata,
    )


def load_full_visible_split(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate the clean 2,775-task public/complement split and gold binding."""

    manifest_path, manifest_sha = _pin(
        args.visible_split_manifest,
        args.expected_visible_split_manifest_sha256,
        "typed API visible split manifest",
    )
    public_path, public_sha = _pin(
        args.visible_train,
        args.expected_visible_train_sha256,
        "typed API visible TRAIN file",
    )
    private_path, private_sha = _pin(
        args.private_split_holdback,
        args.expected_private_split_holdback_sha256,
        "typed API private complement file",
    )
    manifest = _read_json(manifest_path, "typed API visible split manifest")
    outputs = manifest.get("outputs")
    visible_record = outputs.get("visible_train") if isinstance(outputs, Mapping) else None
    private_record = outputs.get("private_holdback") if isinstance(outputs, Mapping) else None
    if (
        manifest.get("schema") != visible_split.MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("clean_rows") != 2775
        or manifest.get("known_contaminant_excluded") != KNOWN_CONTAMINANT
        or manifest.get("inputs", {}).get("gold_train_sha256")
        != args.expected_gold_train_sha256
        or not isinstance(visible_record, Mapping)
        or visible_record.get("sha256") != public_sha
        or visible_record.get("rows") != 2775
        or not isinstance(private_record, Mapping)
        or private_record.get("sha256") != private_sha
        or private_record.get("rows") != 2775
        or manifest.get("privacy", {}).get("visible_file_contains_private_complement")
        is not False
        or manifest.get("privacy", {}).get("gold_source_present") is not False
        or manifest.get("privacy", {}).get("heldout_175_opened") is not False
        or manifest.get("privacy", {}).get("singleton_stdout_answer_visible")
        is not False
    ):
        raise ValueError("typed API visible split manifest differs")
    public_rows = _read_jsonl(public_path, "typed API visible TRAIN")
    private_rows = _read_jsonl(private_path, "typed API private complement")
    gold_path, _gold_sha = _pin(
        args.gold_train_jsonl, args.expected_gold_train_sha256, "gold TRAIN"
    )
    gold_rows = base_sft._read_jsonl(gold_path)  # noqa: SLF001
    gold_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(gold_rows):
        task_id = base_sft._identity(row, index)  # noqa: SLF001
        if task_id != KNOWN_CONTAMINANT:
            gold_by_id[task_id] = row
    if len(gold_rows) != 2776 or len(gold_by_id) != 2775:
        raise ValueError("typed API visible split gold universe differs")
    if len(public_rows) != 2775 or len(private_rows) != 2775:
        raise ValueError("typed API visible/complement row count differs")
    result: dict[str, dict[str, Any]] = {}
    bindings: list[str] = []
    for public, private in zip(public_rows, private_rows, strict=True):
        task_id = str(public.get("task_id") or "")
        gold = gold_by_id.get(task_id)
        visible_tests = public.get("visible_tests")
        holdback_tests = private.get("holdback_tests")
        binding = str(public.get("split_binding_sha256") or "")
        if (
            public.get("schema") != visible_split.PUBLIC_SCHEMA
            or private.get("schema") != visible_split.PRIVATE_SCHEMA
            or private.get("task_id") != task_id
            or gold is None
            or task_id in result
            or not isinstance(visible_tests, str)
            or not visible_tests.strip()
            or not isinstance(holdback_tests, str)
            or not holdback_tests.strip()
            or private.get("split_binding_sha256") != binding
            or not _HEX64.fullmatch(binding)
            or public.get("private_complement_present") is not False
            or private.get("visible_tests_present") is not False
            or public.get("gold_present") is not False
            or private.get("gold_present") is not False
            or not isinstance(public.get("strategy"), str)
            or type(public.get("visible_count")) is not int
            or int(public["visible_count"]) < 0
        ):
            raise ValueError("typed API visible/complement row binding differs")
        original_tests = gold.get("acceptance_tests") or gold.get("tests")
        if (
            not isinstance(original_tests, str)
            or hashlib_sha256(original_tests) != public.get("tests_sha256")
            or public.get("tests_sha256") != private.get("tests_sha256")
        ):
            raise ValueError(f"{task_id}: visible split source tests differ")
        gold_target = base_sft._target_source(gold, task_id)  # noqa: SLF001
        # The split builder's public schema is intentionally narrow.  Refuse
        # the obvious private/gold field classes even if a pinned file drifts.
        forbidden = {
            "holdback_tests",
            "reward_holdback_tests",
            "acceptance_tests",
            "tests",
            "dart_source",
            "supervised_target",
            "gold_target",
        }
        if forbidden & set(public):
            raise ValueError(f"{task_id}: visible split contains forbidden fields")
        result[task_id] = {
            "visible_tests": visible_tests,
            "split_binding_sha256": binding,
            "gold_target": gold_target,
            "strategy": str(public.get("strategy") or ""),
            "visible_count": int(public.get("visible_count", -1)),
        }
        bindings.append(binding)
    if (
        set(result) != set(gold_by_id)
        or manifest.get("task_ids_sha256") != canonical_sha256(list(result))
        or manifest.get("split_bindings_sha256") != canonical_sha256(bindings)
    ):
        raise ValueError("typed API visible split identity/digest differs")
    return result, {
        "schema": visible_split.MANIFEST_SCHEMA,
        "manifest_sha256": manifest_sha,
        "visible_train_sha256": public_sha,
        "private_complement_sha256": private_sha,
        "rows": len(result),
        "task_ids_sha256": canonical_sha256(list(result)),
        "split_bindings_sha256": canonical_sha256(bindings),
        "private_complement_text_serialized": False,
        "gold_text_serialized": False,
    }


def hashlib_sha256(text: str) -> str:
    # Local helper avoids ever placing raw private harness text in a record.
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_typed_provider_prompt(
    plan: base.RescuePlan, parent: base.RescueParent
) -> str:
    prompt = f"""\
Repair the failed Dart reconstruction below.

<OPAQUE_TYPED_CONTRACT_AND_COMPRESSED_ENRICHED_F2>
{plan.task.source}
</OPAQUE_TYPED_CONTRACT_AND_COMPRESSED_ENRICHED_F2>

<FAILED_STUDENT_PROGRAM>
{parent.code}
</FAILED_STUDENT_PROGRAM>

<SANITIZED_COMPILER_DIAGNOSTIC>
{parent.diagnostic}
</SANITIZED_COMPILER_DIAGNOSTIC>

<VISIBLE_TRAIN_CHECKS_PROVIDER_ONLY>
{plan.task.visible_tests}
</VISIBLE_TRAIN_CHECKS_PROVIDER_ONLY>

Return only the complete repaired Dart source.
"""
    if plan.task.source not in prompt or plan.task.visible_tests not in prompt:
        raise ValueError(f"{plan.task.task_id}: provider prompt lost its public inputs")
    for label, forbidden in (
        ("private complete acceptance", plan.gate.tests),
        ("gold target", plan.task.gold_target),
    ):
        if str(forbidden).strip() and str(forbidden) in prompt:
            raise ValueError(f"{plan.task.task_id}: {label} leaked to API prompt")
    return prompt


def build_typed_slots(
    plans: Sequence[base.RescuePlan], *, samples_per_parent: int
) -> list[base.ApiSlot]:
    if samples_per_parent != 1:
        raise ValueError("typed rescue fixes one candidate per parent")
    slots: list[base.ApiSlot] = []
    for task_position, plan in enumerate(plans):
        if plan.task_position != task_position:
            raise ValueError("typed rescue plan positions are not contiguous")
        for parent_position, parent in enumerate(plan.parents):
            prompt = build_typed_provider_prompt(plan, parent)
            slots.append(
                base.ApiSlot(
                    slot_position=len(slots),
                    task_position=task_position,
                    task_id=plan.task.task_id,
                    parent_position=parent_position,
                    sample_index=0,
                    parent=parent,
                    prompt=prompt,
                    prompt_sha256=base.sha256_text(prompt),
                )
            )
    return slots


def _reindex(plans: Sequence[base.RescuePlan]) -> list[base.RescuePlan]:
    return [replace(plan, task_position=index) for index, plan in enumerate(plans)]


def load_prior_cascade_reports(
    *,
    report_paths: Sequence[str | Path],
    expected_sha256s: Sequence[str],
    input_record: Mapping[str, Any],
    source_journal_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate exact earlier typed-cascade reports and their durable ledgers."""

    if len(report_paths) != len(expected_sha256s):
        raise ValueError("prior cascade report/digest counts differ")
    records: list[dict[str, Any]] = []
    seen_reports: set[Path] = set()
    for position, (path_value, expected) in enumerate(
        zip(report_paths, expected_sha256s, strict=True)
    ):
        report_path, observed = _pin(
            path_value, expected, f"prior cascade report {position}"
        )
        if report_path in seen_reports:
            raise ValueError("prior cascade reports must be distinct")
        seen_reports.add(report_path)
        report = _read_json(report_path, f"prior cascade report {position}")
        journal_path = report_path.parent / "typed_api_rescue.journal.jsonl"
        targets_path = report_path.parent / "direct_targets.jsonl"
        manifest_path = report_path.parent / "direct_manifest.json"
        if not all(path.is_file() for path in (journal_path, targets_path, manifest_path)):
            raise ValueError("prior cascade report is missing sealed sibling artifacts")
        events = load_journal(journal_path)
        if not events or not isinstance(events[0].get("contract"), Mapping):
            raise ValueError("prior cascade journal lacks a contract")
        contract = events[0]["contract"]
        report_journal = report.get("journal")
        outputs = report.get("outputs")
        target_record = outputs.get("direct_targets") if isinstance(outputs, Mapping) else None
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("run_contract_sha256") != canonical_sha256(contract)
            or contract.get("schema") != RUN_SCHEMA
            or contract.get("inputs") != input_record
            or contract.get("source_local_harvest_journal")
            != source_journal_record
            or not isinstance(report_journal, Mapping)
            or report_journal.get("sha256") != sha256_file(journal_path)
            or not isinstance(target_record, Mapping)
            or target_record.get("sha256") != sha256_file(targets_path)
        ):
            raise ValueError("prior cascade report contract/artifact binding differs")
        actual_journal = journal_record(journal_path)
        for key in ("sha256", "chain_head_sha256", "event_count", "head_event_sha256"):
            if report_journal.get(key) != actual_journal.get(key):
                raise ValueError(f"prior cascade journal {key} differs")
        manifest = _read_json(manifest_path, "prior cascade direct manifest")
        target_rows = _read_jsonl(targets_path, "prior cascade direct targets")
        task_ids: list[str] = []
        for row in target_rows:
            code = str(row.get("dart_source") or "")
            task_id = str(row.get("task_id") or "")
            if (
                row.get("schema") != DIRECT_TARGET_SCHEMA
                or not task_id
                or row.get("visible_train_passed") is not True
                or row.get("private_full_acceptance_passed") is not True
                or row.get("stability_runs") != STABILITY_RUNS
                or row.get("reasoning_present") is not False
                or row.get("repair_conditioned_training_source_present") is not False
                or row.get("gold_replay") is not False
                or base.sha256_text(code) != row.get("dart_source_sha256")
            ):
                raise ValueError("prior cascade direct target evidence differs")
            task_ids.append(task_id)
        if (
            len(task_ids) != len(set(task_ids))
            or target_record.get("rows") != len(target_rows)
            or manifest.get("schema") != DIRECT_MANIFEST_SCHEMA
            or manifest.get("run_contract_sha256") != canonical_sha256(contract)
            or manifest.get("targets") != target_record
            or manifest.get("rows") != len(target_rows)
            or manifest.get("task_ids_sha256") != canonical_sha256(task_ids)
            or manifest.get("direct_only") is not True
        ):
            raise ValueError("prior cascade direct manifest accounting differs")
        intents = [row for row in events if row.get("event") == "call_intent"]
        results = [row for row in events if row.get("event") == "call_result"]
        verifications = [
            row for row in events if row.get("event") == "task_verification"
        ]
        if len(intents) != len(results) or len(verifications) != len(intents):
            raise ValueError("prior cascade journal phase accounting differs")
        scheduled_ids = [str(row.get("task_id") or "") for row in intents]
        if (
            any(not task_id for task_id in scheduled_ids)
            or len(scheduled_ids) != len(set(scheduled_ids))
            or report.get("schedule", {}).get("scheduled_tasks")
            != len(scheduled_ids)
            or report.get("schedule", {}).get("task_ids_sha256")
            != canonical_sha256(scheduled_ids)
        ):
            raise ValueError("prior cascade schedule differs")
        selected_ids = [
            str(row["selected_target"]["task_id"])
            for row in verifications
            if isinstance(row.get("selected_target"), Mapping)
        ]
        if selected_ids != task_ids:
            raise ValueError("prior cascade journal/direct targets differ")
        verified = set(task_ids)
        retry_ids: list[str] = []
        for result in results:
            task_id = str(result.get("task_id") or "")
            response = result.get("response")
            finish = str(response.get("finish_reason") or "") if isinstance(response, Mapping) else ""
            if task_id not in verified and (
                result.get("parse_accepted") is not True or finish == "length"
            ):
                retry_ids.append(task_id)
        if len(retry_ids) != len(set(retry_ids)):
            raise ValueError("prior cascade retry cohort contains duplicates")
        records.append(
            {
                "path": str(report_path),
                "report_sha256": observed,
                "phase": str(report.get("phase") or ""),
                "cohort_index": int(report.get("cohort_index", -1)),
                "scheduled_task_ids": scheduled_ids,
                "verified_task_ids": task_ids,
                "retry_eligible_task_ids": retry_ids,
                "journal_sha256": sha256_file(journal_path),
                "targets_sha256": sha256_file(targets_path),
            }
        )
    return records


def _cohort_outcome(
    prior_records: Sequence[Mapping[str, Any]],
    cohort_index: int,
    *,
    budget_skipped_retry_tasks: int = 0,
    budget_skipped_retry_task_ids_sha256: str = "",
) -> tuple[set[str], set[str], bool]:
    initial = [
        row
        for row in prior_records
        if row.get("phase") == PHASE_KIMI_INITIAL
        and row.get("cohort_index") == cohort_index
    ]
    retries = [
        row
        for row in prior_records
        if row.get("phase") == PHASE_KIMI_RETRY
        and row.get("cohort_index") == cohort_index
    ]
    if len(initial) != 1 or len(retries) > 1:
        raise ValueError(f"Kimi cohort {cohort_index} prior evidence is incomplete/duplicate")
    initial_row = initial[0]
    retry_needed = set(initial_row["retry_eligible_task_ids"])
    retry_was_budget_skipped = (
        retry_needed
        and not retries
        and budget_skipped_retry_tasks == len(retry_needed)
        and budget_skipped_retry_task_ids_sha256
        == canonical_sha256(initial_row["retry_eligible_task_ids"])
    )
    if retry_needed and len(retries) != 1 and not retry_was_budget_skipped:
        raise ValueError(f"Kimi cohort {cohort_index} requires its targeted retry")
    if retries and (budget_skipped_retry_tasks or budget_skipped_retry_task_ids_sha256):
        raise ValueError("completed Kimi retry conflicts with a budget-skip attestation")
    retry_scheduled = set(retries[0]["scheduled_task_ids"]) if retries else set()
    if retry_scheduled != retry_needed:
        raise ValueError(f"Kimi cohort {cohort_index} retry schedule differs")
    verified = set(initial_row["verified_task_ids"])
    if retries:
        overlap = verified & set(retries[0]["verified_task_ids"])
        if overlap:
            raise ValueError("Kimi initial/retry verified identities overlap")
        verified.update(retries[0]["verified_task_ids"])
    scheduled = set(initial_row["scheduled_task_ids"])
    return scheduled, verified, len(verified) >= KIMI_CONTINUE_MIN_YIELD


def load_visible_projection(
    args: argparse.Namespace,
    *,
    context: TypedSourceContext,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.training import t5gemma2_typed_visible_failure_projection as projection

    report_path, report_sha = _pin(
        args.visible_projection_report,
        args.expected_visible_projection_report_sha256,
        "typed visible projection report",
    )
    journal_path, journal_sha = _pin(
        args.visible_projection_journal,
        args.expected_visible_projection_journal_sha256,
        "typed visible projection journal",
    )
    report = _read_json(report_path, "typed visible projection report")
    events = load_journal(journal_path)
    if not events or not isinstance(events[0].get("contract"), Mapping):
        raise ValueError("typed visible projection journal lacks a contract")
    contract = events[0]["contract"]
    if (
        report.get("schema") != projection.REPORT_SCHEMA
        or report.get("status") != "complete"
        or contract.get("schema") != projection.RUN_SCHEMA
        or contract.get("inputs") != context.input_record
        or contract.get("source_local_harvest_journal")
        != context.source_journal_record
        or report.get("run_contract_sha256") != canonical_sha256(contract)
        or report.get("journal", {}).get("sha256") != journal_sha
        or report.get("privacy", {}).get(
            "private_complete_outcome_consumed_for_eligibility"
        )
        is not False
        or report.get("privacy", {}).get("private_complete_diagnostic_consumed")
        is not False
        or report.get("privacy", {}).get("heldout_175_opened") is not False
        or report.get("privacy", {}).get("frontier_api_calls") is not False
    ):
        raise ValueError("typed visible projection report binding differs")
    actual = journal_record(journal_path)
    for key in ("sha256", "chain_head_sha256", "event_count", "head_event_sha256"):
        if report.get("journal", {}).get(key) != actual.get(key):
            raise ValueError(f"typed visible projection journal {key} differs")
    terminals, complete = projection.validate_journal(
        events, contract=contract, context=context
    )
    if not complete:
        raise ValueError("typed visible projection is incomplete")
    zero_ids = [
        str(row["task_id"])
        for row in terminals
        if row["visible_unique_passes"] == 0
    ]
    eligible_ids = [
        str(row["task_id"])
        for row in terminals
        if row.get("api_eligible") is True
    ]
    singleton_ids = [
        str(row["task_id"])
        for row in terminals
        if row.get("api_eligibility_stratum")
        == "singleton_stdout_compile_call_only"
    ]
    if (
        report.get("tasks") != len(terminals)
        or report.get("candidate_executions") != len(terminals) * 4
        or report.get("visible_all_zero_tasks") != len(zero_ids)
        or report.get("visible_all_zero_task_ids_sha256")
        != canonical_sha256(zero_ids)
        or report.get("api_eligible_tasks") != len(eligible_ids)
        or report.get("api_eligible_task_ids_sha256")
        != canonical_sha256(eligible_ids)
        or report.get("singleton_stdout_compile_call_only_tasks")
        != len(singleton_ids)
        or report.get("singleton_stdout_task_ids_sha256")
        != canonical_sha256(singleton_ids)
    ):
        raise ValueError("typed visible projection accounting differs")
    return terminals, {
        "schema": projection.REPORT_SCHEMA,
        "report_sha256": report_sha,
        "journal_sha256": journal_sha,
        "run_contract_sha256": canonical_sha256(contract),
        "tasks": len(terminals),
        "visible_all_zero_tasks": len(zero_ids),
        "visible_all_zero_task_ids_sha256": canonical_sha256(zero_ids),
        "api_eligible_tasks": len(eligible_ids),
        "api_eligible_task_ids_sha256": canonical_sha256(eligible_ids),
        "singleton_stdout_compile_call_only_tasks": len(singleton_ids),
        "singleton_stdout_task_ids_sha256": canonical_sha256(singleton_ids),
        "private_complete_outcome_consumed_for_eligibility": False,
        "private_complete_diagnostic_consumed": False,
    }


def select_visible_zero_tasks(
    *,
    context: TypedSourceContext,
    projection_terminals: Sequence[Mapping[str, Any]],
    seed: int,
    excluded_ids: set[str],
) -> list[tuple[int, PilotTask, Mapping[str, Any]]]:
    values: list[tuple[int, PilotTask, Mapping[str, Any]]] = []
    for position, (task, terminal) in enumerate(
        zip(context.scheduled_tasks, projection_terminals, strict=True)
    ):
        if terminal.get("task_id") != task.task_id:
            raise ValueError("typed local task/terminal identity differs")
        visible_passes = terminal.get("visible_unique_passes")
        if type(visible_passes) is not int or visible_passes < 0:
            raise ValueError("typed visible projection pass count is malformed")
        if terminal.get("api_eligible") is True and task.task_id not in excluded_ids:
            values.append((position, task, terminal))
    values.sort(
        key=lambda item: (
            1
            if item[2].get("api_eligibility_stratum")
            == "singleton_stdout_compile_call_only"
            else 0,
            canonical_sha256(
                {
                    "schema": RUN_SCHEMA,
                    "seed": seed,
                    "task_id": item[1].task_id,
                    "local_task_position": item[0],
                    "eligibility_stratum": item[2].get(
                        "api_eligibility_stratum"
                    ),
                }
            ),
        )
    )
    return values


def build_visible_only_plans(
    *,
    selected: Sequence[tuple[int, PilotTask, Mapping[str, Any]]],
    gates: Mapping[str, PrivateGate],
) -> tuple[list[base.RescuePlan], dict[str, Any]]:
    """Build parents exclusively from the sealed visible-only projection."""

    plans: list[base.RescuePlan] = []
    evidence: list[dict[str, Any]] = []
    for _local_position, task, terminal in selected:
        gate = gates[task.task_id]
        singleton_compile_only = (
            terminal.get("api_eligibility_stratum")
            == "singleton_stdout_compile_call_only"
        )
        if terminal.get("api_eligible") is not True:
            raise ValueError("API plan source is not in a sealed eligible stratum")
        parents: list[base.RescueParent] = []
        seen: set[str] = set()
        candidate_rows = base._all_terminal_candidates(terminal)  # noqa: SLF001
        for candidate_index, original in enumerate(candidate_rows):
            code = str(original.get("code") or "").strip()
            digest = str(original.get("code_sha256") or "")
            if not code or base.sha256_text(code) != digest or digest in seen:
                continue
            seen.add(digest)
            visible = original.get("visible")
            safe = str(original.get("safe_visible_diagnostic") or "")
            if (
                not isinstance(visible, Mapping)
                or type(visible.get("compiled")) is not bool
                or type(visible.get("passed")) is not bool
                or (not singleton_compile_only and visible.get("passed") is not False)
                or not safe
                or base.sha256_text(safe)
                != original.get("safe_visible_diagnostic_sha256")
                or original.get("diagnostic_source")
                != "sealed_visible_TRAIN_split"
                or original.get("private_complete_diagnostic_consumed") is not False
            ):
                raise ValueError("visible-only parent evidence differs")
            compiled = bool(visible["compiled"])
            feedback_source, safe = base._feedback_source(  # noqa: SLF001
                task=task,
                code=code,
                diagnostic=safe,
                compiled=compiled,
            )
            parents.append(
                base.RescueParent(
                    task_id=task.task_id,
                    parent_index=candidate_index,
                    code=code,
                    code_sha256=digest,
                    compiled=compiled,
                    diagnostic=safe,
                    diagnostic_sha256=base.sha256_text(safe),
                    origin=str(original.get("origin") or "local_student_direct"),
                    feedback_source=feedback_source,
                    feedback_source_sha256=base.sha256_text(feedback_source),
                )
            )
            evidence.append(
                {
                    "task_id": task.task_id,
                    "candidate_code_sha256": digest,
                    "compiled_on_permitted_visible_split": compiled,
                    "passed_permitted_visible_split": False,
                    "diagnostic_sha256": base.sha256_text(safe),
                    "diagnostic_text_persisted_in_contract": False,
                    "private_complete_diagnostic_consumed": False,
                    "eligibility_stratum": terminal.get(
                        "api_eligibility_stratum"
                    ),
                    "singleton_private_answer_used_for_eligibility": False,
                }
            )
        informative = [
            row
            for row in parents
            if not row.compiled and row.diagnostic != base.MISSING_SAFE_DIAGNOSTIC
        ]
        other_noncompiling = [
            row for row in parents if not row.compiled and row not in informative
        ]
        compiled = [row for row in parents if row.compiled]
        chosen = (informative or other_noncompiling or compiled)[:1]
        if not chosen:
            continue
        plans.append(
            base.RescuePlan(
                task_position=len(plans),
                task=task,
                gate=gate,
                local_terminal_sha256=str(
                    terminal.get("journal_event_sha256") or ""
                ),
                parents=tuple(chosen),
            )
        )
    return plans, {
        "schema": "t5gemma2-typed-api-visible-diagnostic-provenance-v1",
        "candidates_reexecuted": len(evidence),
        "evidence_sha256": canonical_sha256(evidence),
        "diagnostic_source": "separately_pinned_permitted_visible_TRAIN_split",
        "complete_private_suite_used_for_diagnostic": False,
        "complete_private_suite_diagnostic_persisted": False,
        "singleton_private_answer_used_for_eligibility": False,
    }


def publish_direct_only(
    *,
    output_dir: Path,
    plans: Sequence[base.RescuePlan],
    verifications: Sequence[Mapping[str, Any]],
    contract_sha256: str,
    provider_phase: str,
    provider_model: str,
    stability_runs: int,
) -> dict[str, Any]:
    plan_by_id = {plan.task.task_id: plan for plan in plans}
    rows: list[dict[str, Any]] = []
    for event in verifications:
        selected = event.get("selected_target")
        if not isinstance(selected, Mapping):
            continue
        task_id = str(selected.get("task_id") or "")
        plan = plan_by_id.get(task_id)
        code = str(selected.get("code") or "")
        if (
            plan is None
            or not code.strip()
            or selected.get("visible_passed") is not True
            or selected.get("private_gate_passed") is not True
            or base.sha256_text(code) != selected.get("code_sha256")
        ):
            raise ValueError("selected direct target lacks sealed verification")
        rows.append(
            {
                "schema": DIRECT_TARGET_SCHEMA,
                "task_id": task_id,
                "source_sha256": plan.task.source_sha256,
                "dart_source": code,
                "dart_source_sha256": base.sha256_text(code),
                "origin": "external_teacher_direct_verified",
                "provider_phase": provider_phase,
                "provider_model": provider_model,
                "visible_train_passed": True,
                "private_full_acceptance_passed": True,
                "stability_runs": stability_runs,
                "reasoning_present": False,
                "repair_conditioned_training_source_present": False,
                "gold_replay": False,
                "provenance": {
                    "run_contract_sha256": contract_sha256,
                    "slot_position": selected["slot_position"],
                    "parent_code_sha256": selected["parent_code_sha256"],
                    "diagnostic_sha256": selected["diagnostic_sha256"],
                },
            }
        )
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("direct-only output contains duplicate task identities")
    target_path = output_dir / "direct_targets.jsonl"
    base._exact_write_jsonl(target_path, rows)  # noqa: SLF001
    target_record = {
        "path": str(target_path),
        "sha256": sha256_file(target_path),
        "rows": len(rows),
    }
    manifest = {
        "schema": DIRECT_MANIFEST_SCHEMA,
        "run_contract_sha256": contract_sha256,
        "rows": len(rows),
        "targets": target_record,
        "mapping": "typed_opaque_contract_plus_compressed_enriched_F2_to_verified_Dart",
        "direct_only": True,
        "repair_conditioned_rows": 0,
        "gold_replay_rows": 0,
        "reasoning_rows": 0,
        "tests_in_training_output": False,
        "private_feedback_in_training_output": False,
        "full_acceptance_reverified": True,
        "stability_runs": stability_runs,
        "task_ids_sha256": canonical_sha256(task_ids),
        "production_floor_eligible": True,
    }
    require_exact_or_write(output_dir / "direct_manifest.json", manifest)
    return {
        "rows": rows,
        "files": {"direct_targets": target_record},
        "manifest": manifest,
    }


def cohort_decision(
    *,
    initial_verified_ids: Sequence[str],
    retry_verified_ids: Sequence[str] = (),
) -> dict[str, Any]:
    initial = list(dict.fromkeys(initial_verified_ids))
    retry = list(dict.fromkeys(retry_verified_ids))
    if set(initial) & set(retry):
        raise ValueError("initial and retry verified target identities overlap")
    combined = initial + retry
    yield_count = len(combined)
    return {
        "schema": COHORT_DECISION_SCHEMA,
        "initial_verified": len(initial),
        "retry_verified": len(retry),
        "verified_unique_targets": yield_count,
        "verified_task_ids_sha256": canonical_sha256(combined),
        "continue_kimi": yield_count >= KIMI_CONTINUE_MIN_YIELD,
        "minimum_yield_to_continue": KIMI_CONTINUE_MIN_YIELD,
    }


def validate_phase_profile(args: argparse.Namespace) -> None:
    fixed_kimi_cohort_limit = int(
        getattr(args, "fixed_kimi_cohort_limit", 0)
    )
    if fixed_kimi_cohort_limit < 0:
        raise ValueError("fixed Kimi cohort limit must be non-negative")
    budget_skipped_retry_tasks = int(
        getattr(args, "budget_skipped_kimi_retry_tasks", 0)
    )
    budget_skipped_retry_digest = str(
        getattr(args, "budget_skipped_kimi_retry_task_ids_sha256", "") or ""
    )
    if budget_skipped_retry_tasks < 0:
        raise ValueError("budget-skipped Kimi retry count must be non-negative")
    if bool(budget_skipped_retry_tasks) != bool(budget_skipped_retry_digest):
        raise ValueError("budget-skipped Kimi retry count/digest must be paired")
    if budget_skipped_retry_digest:
        _require_digest(
            budget_skipped_retry_digest,
            "budget-skipped Kimi retry task IDs",
        )
    if budget_skipped_retry_tasks and args.phase != PHASE_SONNET_RESIDUAL:
        raise ValueError("only Sonnet may consume a budget-skipped Kimi retry attestation")
    if (
        fixed_kimi_cohort_limit
        and args.phase == PHASE_KIMI_INITIAL
        and args.cohort_index >= fixed_kimi_cohort_limit
    ):
        raise ValueError("Kimi initial cohort exceeds the fixed cohort limit")
    common = {
        "max_parents_per_task": 1,
        "samples_per_parent": 1,
        "stability_runs": STABILITY_RUNS,
    }
    for name, expected in common.items():
        if getattr(args, name) != expected:
            raise ValueError(f"typed rescue fixes --{name}={expected}")
    if args.evaluation_only or args.exploratory_terminal_prefix:
        raise ValueError("typed direct rescue forbids evaluation-only/exploratory mode")
    if args.allow_unpinned_inputs:
        raise ValueError("typed direct rescue forbids unpinned inputs")
    if args.phase == PHASE_KIMI_INITIAL:
        expected = {
            "provider": "openrouter_chat",
            "model": KIMI_MODEL,
            "max_output_tokens": KIMI_INITIAL_MAX_OUTPUT,
            "openrouter_reasoning": "enabled",
            "openrouter_reasoning_effort": "low",
            "chat_token_parameter": "max_tokens",
        }
        if args.max_tasks != KIMI_COHORT_SIZE:
            raise ValueError("Kimi initial cohorts fix --max_tasks=50")
        if args.retry_parse_failures_or_truncations_report:
            raise ValueError("Kimi initial phase cannot consume a retry source")
    elif args.phase == PHASE_KIMI_RETRY:
        expected = {
            "provider": "openrouter_chat",
            "model": KIMI_MODEL,
            "max_output_tokens": KIMI_RETRY_MAX_OUTPUT,
            "openrouter_reasoning": "enabled",
            "openrouter_reasoning_effort": "low",
            "chat_token_parameter": "max_tokens",
        }
        if not args.retry_parse_failures_or_truncations_report:
            raise ValueError("Kimi retry requires its pinned initial report")
    else:
        expected = {
            "provider": "anthropic",
            "model": SONNET_MODEL,
            "max_output_tokens": SONNET_MAX_OUTPUT,
            "anthropic_thinking": "adaptive",
            "anthropic_effort": "high",
        }
        if args.retry_parse_failures_or_truncations_report:
            raise ValueError("Sonnet residual phase cannot consume a retry source")
    for name, wanted in expected.items():
        if getattr(args, name) != wanted:
            raise ValueError(f"{args.phase} fixes --{name}={wanted}")
    if args.phase.startswith("kimi"):
        if (
            not args.openrouter_require_parameters
            or not args.openrouter_enforce_distillable_text
            or not args.openrouter_provider_only
        ):
            raise ValueError("Kimi requires sealed distillable OpenRouter routing")
    if args.phase == PHASE_KIMI_RETRY:
        if args.max_tasks != args.expected_retry_parse_failures_or_truncations_tasks:
            raise ValueError("Kimi retry max_tasks must equal its exact retry count")
        if args.max_calls != args.max_tasks:
            raise ValueError("Kimi retry fixes one reserved call per retry task")
    elif args.max_calls != args.max_tasks:
        raise ValueError("typed initial/residual phase fixes one reserved call per task")
    for name in (
        "max_calls",
        "max_input_tokens_per_call",
        "max_input_tokens_total",
        "max_output_tokens_total",
        "max_total_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"typed rescue requires a positive --{name} reservation")
    prices = (
        Decimal(str(args.max_usd)),
        Decimal(str(args.input_usd_per_million)),
        Decimal(str(args.output_usd_per_million)),
    )
    if any(not value.is_finite() or value <= 0 for value in prices):
        raise ValueError("typed rescue budget values must be finite and positive")

    max_usd = prices[0]
    if args.phase == PHASE_KIMI_INITIAL:
        if max_usd > Decimal("12"):
            raise ValueError("Kimi initial per-cohort spend cap may not exceed $12")
        expected_prices = (Decimal("3"), Decimal("15"))
    elif args.phase == PHASE_KIMI_RETRY:
        if max_usd > Decimal("12"):
            raise ValueError("Kimi retry reservation may not exceed $12")
        expected_prices = (Decimal("3"), Decimal("15"))
    else:
        if max_usd > Decimal("11.5"):
            raise ValueError("Sonnet residual reservation may not exceed $11.50")
        expected_prices = (Decimal("2"), Decimal("10"))
    if prices[1:] != expected_prices:
        raise ValueError(f"{args.phase} list-price reservation differs")


def _phase_selection(
    *,
    args: argparse.Namespace,
    all_visible_zero: Sequence[tuple[int, PilotTask, Mapping[str, Any]]],
    prior_records: Sequence[Mapping[str, Any]],
) -> tuple[
    list[tuple[int, PilotTask, Mapping[str, Any]]],
    dict[str, Any],
]:
    by_id = {task.task_id: row for row in all_visible_zero for task in [row[1]]}
    all_ids = [row[1].task_id for row in all_visible_zero]
    fixed_kimi_cohort_limit = int(
        getattr(args, "fixed_kimi_cohort_limit", 0)
    )
    budget_skipped_retry_tasks = int(
        getattr(args, "budget_skipped_kimi_retry_tasks", 0)
    )
    budget_skipped_retry_digest = str(
        getattr(args, "budget_skipped_kimi_retry_task_ids_sha256", "") or ""
    )
    if args.phase == PHASE_KIMI_RETRY:
        if len(prior_records) != 1:
            raise ValueError("Kimi retry requires exactly one pinned initial report")
        source = prior_records[0]
        if (
            source.get("phase") != PHASE_KIMI_INITIAL
            or source.get("cohort_index") != args.cohort_index
        ):
            raise ValueError("Kimi retry source is not its matching initial cohort")
        retry_ids = list(source["retry_eligible_task_ids"])
        if (
            len(retry_ids) != args.expected_retry_parse_failures_or_truncations_tasks
            or canonical_sha256(retry_ids)
            != args.expected_retry_parse_failures_or_truncations_task_ids_sha256
            or any(task_id not in by_id for task_id in retry_ids)
        ):
            raise ValueError("Kimi targeted retry cohort differs")
        return [by_id[task_id] for task_id in retry_ids], {
            "mode": PHASE_KIMI_RETRY,
            "source_report_sha256": source["report_sha256"],
            "targeted_non_code_or_length_only": True,
            "accepted_nontruncated_source_responses_regenerated": False,
        }

    initial_records = [
        row for row in prior_records if row.get("phase") == PHASE_KIMI_INITIAL
    ]
    if args.phase == PHASE_KIMI_INITIAL:
        if fixed_kimi_cohort_limit and args.cohort_index >= fixed_kimi_cohort_limit:
            raise ValueError("fixed Kimi cohort limit forbids another initial cohort")
        expected_indices = list(range(args.cohort_index))
        observed_indices = sorted(int(row["cohort_index"]) for row in initial_records)
        if observed_indices != expected_indices:
            raise ValueError("prior Kimi initial cohorts are not an exact prefix")
        prior_scheduled: set[str] = set()
        outcomes: list[dict[str, Any]] = []
        for cohort_index in expected_indices:
            scheduled, verified, can_continue = _cohort_outcome(
                prior_records, cohort_index
            )
            if not can_continue:
                raise ValueError(
                    f"Kimi stop rule forbids cohort {args.cohort_index}; "
                    f"cohort {cohort_index} yielded fewer than 8"
                )
            if prior_scheduled & scheduled:
                raise ValueError("Kimi initial cohort schedules overlap")
            prior_scheduled.update(scheduled)
            outcomes.append(
                {
                    "cohort_index": cohort_index,
                    "scheduled": len(scheduled),
                    "verified": len(verified),
                    "continued": True,
                }
            )
        residual = [row for row in all_visible_zero if row[1].task_id not in prior_scheduled]
        selected = residual[: args.max_tasks]
        return selected, {
            "mode": PHASE_KIMI_INITIAL,
            "cohort_index": args.cohort_index,
            "cohort_size": KIMI_COHORT_SIZE,
            "previous_cohort_outcomes": outcomes,
            "prior_scheduled_tasks_excluded": len(prior_scheduled),
            "prior_scheduled_task_ids_sha256": canonical_sha256(
                [task_id for task_id in all_ids if task_id in prior_scheduled]
            ),
            "minimum_verified_yield_to_continue": KIMI_CONTINUE_MIN_YIELD,
            "fixed_kimi_cohort_limit": fixed_kimi_cohort_limit,
        }

    if not initial_records:
        raise ValueError("Sonnet residual requires at least one completed Kimi cohort")
    last_index = max(int(row["cohort_index"]) for row in initial_records)
    expected_indices = list(range(last_index + 1))
    if sorted(int(row["cohort_index"]) for row in initial_records) != expected_indices:
        raise ValueError("Kimi evidence before Sonnet is not an exact cohort prefix")
    all_verified: set[str] = set()
    last_continue = False
    outcomes = []
    for cohort_index in expected_indices:
        _scheduled, verified, can_continue = _cohort_outcome(
            prior_records,
            cohort_index,
            budget_skipped_retry_tasks=(
                budget_skipped_retry_tasks if cohort_index == last_index else 0
            ),
            budget_skipped_retry_task_ids_sha256=(
                budget_skipped_retry_digest if cohort_index == last_index else ""
            ),
        )
        overlap = all_verified & verified
        if overlap:
            raise ValueError("verified task appears in multiple Kimi cohorts")
        all_verified.update(verified)
        last_continue = can_continue
        outcomes.append(
            {
                "cohort_index": cohort_index,
                "verified": len(verified),
                "continued": can_continue,
            }
        )
    fixed_limit_reached = (
        fixed_kimi_cohort_limit > 0
        and len(expected_indices) == fixed_kimi_cohort_limit
    )
    if fixed_kimi_cohort_limit and len(expected_indices) > fixed_kimi_cohort_limit:
        raise ValueError("Kimi evidence exceeds the fixed cohort limit")
    if last_continue and not fixed_limit_reached:
        raise ValueError("Sonnet cannot start while the Kimi >=8 yield rule says continue")
    all_verified.update(
        task_id
        for row in prior_records
        if row.get("phase") == PHASE_SONNET_RESIDUAL
        for task_id in row["verified_task_ids"]
    )
    residual = [row for row in all_visible_zero if row[1].task_id not in all_verified]
    return residual[: args.max_tasks], {
        "mode": PHASE_SONNET_RESIDUAL,
        "kimi_cohort_outcomes": outcomes,
        "kimi_stopped_for_yield_below_8": not last_continue,
        "kimi_stopped_for_fixed_cohort_limit": fixed_limit_reached,
        "fixed_kimi_cohort_limit": fixed_kimi_cohort_limit,
        "budget_skipped_kimi_retry_tasks": budget_skipped_retry_tasks,
        "budget_skipped_kimi_retry_task_ids_sha256": budget_skipped_retry_digest,
        "prior_verified_tasks_excluded": len(all_verified),
        "prior_verified_task_ids_sha256": canonical_sha256(
            [task_id for task_id in all_ids if task_id in all_verified]
        ),
    }


def run(
    args: argparse.Namespace,
    *,
    transport: base.ProviderTransport | None = None,
    evaluate: base.EvaluateFn | None = None,
) -> dict[str, Any]:
    validate_phase_profile(args)
    context = load_typed_source_context(args)
    projection_terminals, projection_record = load_visible_projection(
        args, context=context
    )
    existing_ids, existing_record = load_existing_225_exclusions(
        args.existing_direct_manifest,
        args.expected_existing_direct_manifest_sha256,
    )
    local_ids = {task.task_id for task in context.scheduled_tasks}
    if local_ids & set(existing_ids) or KNOWN_CONTAMINANT in local_ids:
        raise ValueError("typed local/API source schedule includes an excluded identity")
    input_record = {
        "source": context.input_record,
        "visible_failure_projection": projection_record,
        "existing_225_exclusion": existing_record,
    }

    prior_paths: list[str | Path]
    prior_digests: list[str]
    if args.phase == PHASE_KIMI_RETRY:
        prior_paths = [args.retry_parse_failures_or_truncations_report]
        prior_digests = [
            args.expected_retry_parse_failures_or_truncations_report_sha256
        ]
    else:
        prior_paths = list(args.prior_success_report)
        prior_digests = list(args.expected_prior_success_report_sha256)
    prior_records = load_prior_cascade_reports(
        report_paths=prior_paths,
        expected_sha256s=prior_digests,
        input_record=input_record,
        source_journal_record=context.source_journal_record,
    )
    all_api_eligible = select_visible_zero_tasks(
        context=context,
        projection_terminals=projection_terminals,
        seed=args.seed,
        excluded_ids=set(existing_ids),
    )
    selected, cascade_selection = _phase_selection(
        args=args,
        all_visible_zero=all_api_eligible,
        prior_records=prior_records,
    )
    if not selected and not (
        args.plan_only_output and args.phase == PHASE_SONNET_RESIDUAL
    ):
        raise ValueError("typed API rescue phase has no scheduled residual tasks")
    if args.phase == PHASE_KIMI_INITIAL and len(selected) != min(
        KIMI_COHORT_SIZE, len(all_api_eligible)
    ):
        # Later cohorts normally still have >=50 residual rows.  This check is
        # intentionally conservative only for cohort zero; later exactness is
        # bound by the prior-schedule record below.
        if args.cohort_index == 0:
            raise ValueError("first Kimi cohort is not exactly 50 tasks")
    plans, diagnostic_record = build_visible_only_plans(
        selected=selected,
        gates=context.gates,
    )
    if len(plans) != len(selected):
        raise ValueError("visible-zero task lacked a usable K=4 parent")

    input_price = Decimal(str(args.input_usd_per_million))
    output_price = Decimal(str(args.output_usd_per_million))
    max_usd = Decimal(str(args.max_usd))
    capacity, budget_contract = base.schedule_capacity(
        max_calls=args.max_calls,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        max_output_tokens_per_call=args.max_output_tokens,
        max_input_tokens_total=args.max_input_tokens_total,
        max_output_tokens_total=args.max_output_tokens_total,
        max_total_tokens=args.max_total_tokens,
        max_usd=max_usd,
        input_usd_per_million=input_price,
        output_usd_per_million=output_price,
    )
    capped = base.cap_plans_to_budget(
        plans, samples_per_parent=1, call_capacity=capacity
    )
    if len(capped) != len(plans):
        raise ValueError("budget reservation does not cover the fixed typed cohort")
    plans = capped
    slots = build_typed_slots(plans, samples_per_parent=1)
    if len(slots) != len(plans) or len(slots) > args.max_calls:
        raise ValueError("typed API one-call-per-task accounting differs")
    schedule_ids = [plan.task.task_id for plan in plans]
    if args.expected_scheduled_task_ids_sha256 and (
        canonical_sha256(schedule_ids) != args.expected_scheduled_task_ids_sha256
    ):
        raise ValueError("typed API scheduled task digest differs")

    if args.plan_only_output:
        plan_record = {
            "schema": PLAN_SCHEMA,
            "status": "complete",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "phase": args.phase,
            "cohort_index": args.cohort_index,
            "fixed_kimi_cohort_limit": int(args.fixed_kimi_cohort_limit),
            "inputs_sha256": canonical_sha256(input_record),
            "source_journal_sha256": context.source_journal_record.get("sha256"),
            "prior_reports": [
                {
                    "report_sha256": row["report_sha256"],
                    "phase": row["phase"],
                    "cohort_index": row["cohort_index"],
                    "journal_sha256": row["journal_sha256"],
                    "targets_sha256": row["targets_sha256"],
                }
                for row in prior_records
            ],
            "selection": {
                **cascade_selection,
                "scheduled_tasks": len(plans),
                "scheduled_calls": len(slots),
                "task_ids_sha256": canonical_sha256(schedule_ids),
            },
            "budget": budget_contract,
            "provider_credentials_read": False,
            "frontier_api_calls": False,
        }
        plan_path = Path(args.plan_only_output).expanduser().resolve()
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        require_exact_or_write(plan_path, plan_record)
        print(json.dumps(plan_record, sort_keys=True), flush=True)
        return plan_record

    base_url = base.validate_provider_endpoint(
        provider=args.provider, base_url=args.base_url, api_version=args.api_version
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = (
        "repair_policy_targets.jsonl",
        "repair_policy_sources.jsonl",
        "repair_policy_manifest.json",
        "direct_hard_targets_f2.jsonl",
    )
    stale = [name for name in forbidden if (output_dir / name).exists()]
    if stale:
        raise ValueError("typed direct-only output directory contains forbidden artifacts")
    journal_path = output_dir / "typed_api_rescue.journal.jsonl"
    provider_contract = base._provider_contract(args, base_url)  # noqa: SLF001
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "phase": args.phase,
        "cohort_index": args.cohort_index,
        "source_local_harvest_journal": context.source_journal_record,
        "inputs": input_record,
        "prior_reports": [
            {
                key: row[key]
                for key in (
                    "report_sha256",
                    "phase",
                    "cohort_index",
                    "journal_sha256",
                    "targets_sha256",
                )
            }
            for row in prior_records
        ],
        "selection": {
            **cascade_selection,
            "seed": args.seed,
            "eligibility": "sealed_visible_TRAIN_all_zero_K4_only",
            "private_complete_outcome_used_for_eligibility": False,
            "api_eligible_tasks": len(all_api_eligible),
            "semantic_all_zero_precedes_singleton_compile_call_only": True,
            "scheduled_tasks": len(plans),
            "scheduled_slots": len(slots),
            "task_ids_sha256": canonical_sha256(schedule_ids),
            "slot_bindings_sha256": canonical_sha256(
                [base._slot_binding(slot) for slot in slots]  # noqa: SLF001
            ),
            "max_parents_per_task": 1,
            "samples_per_parent": 1,
        },
        "visible_diagnostic_provenance": diagnostic_record,
        "provider": provider_contract,
        "budget": budget_contract,
        "verification": {
            "visible_before_private": True,
            "all_api_calls_before_any_private_gate": True,
            "private_gate": "complete_TRAIN_acceptance",
            "stability_runs": STABILITY_RUNS,
            "private_failure_triggers_api_call": False,
            "private_gate_can_only_reject_transfer": True,
        },
        "privacy": {
            "api_input_fields": [
                "opaque_typed_contract",
                "compressed_enriched_F2",
                "failed_local_candidate",
                "visible_split_derived_diagnostic",
                "visible_TRAIN_checks",
            ],
            "private_complete_acceptance_sent_to_provider": False,
            "private_split_holdback_sent_to_provider": False,
            "gold_sent_to_provider": False,
            "heldout_175_opened": False,
            "api_credentials_persisted": False,
            "plaintext_reasoning_persisted": False,
        },
        "training_outputs": {
            "direct_verified_code_targets": True,
            "repair_conditioned_rows": 0,
            "gold_replay_rows": 0,
            "reasoning_rows": 0,
            "tests_in_training_outputs": False,
            "production_floor_eligible": True,
            "training_use_forbidden": False,
        },
        "heldout_175_opened": False,
    }
    base._assert_secret_free(contract)  # noqa: SLF001

    with _typed_base_schemas():
        events = load_journal(journal_path)
        if not events:
            base._append_safe(  # noqa: SLF001
                journal_path,
                {
                    "event": "header",
                    "schema": JOURNAL_SCHEMA,
                    "contract": contract,
                    "contract_sha256": canonical_sha256(contract),
                },
            )
        else:
            base.validate_rescue_journal(
                events, contract=contract, plans=plans, slots=slots
            )
        api_key = str(os.environ.get(args.api_key_env) or "")
        if not api_key:
            raise RuntimeError(
                f"provider credential environment variable {args.api_key_env!r} is empty"
            )
        if transport is None:
            transport = base._build_transport(  # noqa: SLF001
                args, api_key=api_key, base_url=base_url
            )
        slot_results = base.execute_api_phase(
            journal_path=journal_path,
            contract=contract,
            plans=plans,
            slots=slots,
            transport=transport,
            api_key=api_key,
            max_input_tokens=args.max_input_tokens_per_call,
            max_output_tokens=args.max_output_tokens,
            input_usd_per_million=input_price,
            output_usd_per_million=output_price,
            inter_call_delay_seconds=args.inter_call_delay_seconds,
            abort_on_provider_error=args.abort_on_provider_error,
            provider_max_attempts=args.provider_max_attempts,
            provider_retry_base_seconds=args.provider_retry_base_seconds,
            provider_retry_max_seconds=args.provider_retry_max_seconds,
        )
        if evaluate is None:
            validate_dart_binary()
            evaluate = base._runtime_evaluator(  # noqa: SLF001
                timeout=args.timeout, stability_runs=STABILITY_RUNS
            )
        verifications = base.execute_verification_phase(
            journal_path=journal_path,
            contract=contract,
            plans=plans,
            slots=slots,
            evaluate=evaluate,
            api_key=api_key,
        )
        final_state = base.validate_rescue_journal(
            load_journal(journal_path),
            contract=contract,
            plans=plans,
            slots=slots,
        )
    if not final_state["complete"]:
        raise RuntimeError("typed API rescue journal did not complete")
    contract_sha = canonical_sha256(contract)
    outputs = publish_direct_only(
        output_dir=output_dir,
        plans=plans,
        verifications=verifications,
        contract_sha256=contract_sha,
        provider_phase=args.phase,
        provider_model=args.model,
        stability_runs=STABILITY_RUNS,
    )
    charged_input = sum(row["usage"]["charged_input_tokens"] for row in slot_results)
    charged_output = sum(row["usage"]["charged_output_tokens"] for row in slot_results)
    charged_nanos = sum(row["usage"]["charged_usd_nanos"] for row in slot_results)
    retry_eligible = []
    verified_ids = [row["task_id"] for row in outputs["rows"]]
    verified_set = set(verified_ids)
    for row in slot_results:
        response = row.get("response")
        finish = str(response.get("finish_reason") or "") if isinstance(response, Mapping) else ""
        if str(row.get("task_id")) not in verified_set and (
            row.get("parse_accepted") is not True or finish == "length"
        ):
            retry_eligible.append(str(row["task_id"]))
    decision = None
    if args.phase == PHASE_KIMI_RETRY:
        source_initial = prior_records[0]
        decision = cohort_decision(
            initial_verified_ids=source_initial["verified_task_ids"],
            retry_verified_ids=verified_ids,
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "phase": args.phase,
        "cohort_index": args.cohort_index,
        "run_contract_sha256": contract_sha,
        "provider": provider_contract,
        "schedule": {
            "api_eligible_tasks": len(all_api_eligible),
            "scheduled_tasks": len(plans),
            "scheduled_calls": len(slots),
            "task_ids_sha256": canonical_sha256(schedule_ids),
            "provider_responses": sum(
                row.get("status") == "response" for row in slot_results
            ),
            "code_only_responses": sum(
                row.get("parse_accepted") is True for row in slot_results
            ),
            "retry_eligible_non_code_or_length_tasks": len(retry_eligible),
            "retry_eligible_task_ids_sha256": canonical_sha256(retry_eligible),
        },
        "verification": {
            "visible_passes": sum(
                candidate["passed"]
                for event in verifications
                for candidate in event["visible_results"]
            ),
            "private_full_acceptance_passes": len(verified_ids),
            "verified_unique_hard_targets": len(verified_ids),
            "verified_task_ids_sha256": canonical_sha256(verified_ids),
        },
        "budget_charged": {
            "calls": len(slot_results),
            "input_tokens": charged_input,
            "output_tokens": charged_output,
            "total_tokens": charged_input + charged_output,
            "estimated_usd_nanos": charged_nanos,
            "estimated_usd": f"{Decimal(charged_nanos) / Decimal(1_000_000_000):.9f}",
            "unknown_usage_failures_charged_at_full_reservation": True,
            "within_contract": charged_nanos
            <= int(
                (max_usd * Decimal(1_000_000_000)).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        },
        "outputs": outputs["files"],
        "direct_manifest": outputs["manifest"],
        "repair_policy_manifest": None,
        "cohort_decision": decision,
        "journal": journal_record(journal_path),
        "privacy_invariants": contract["privacy"],
        "heldout_175_opened": False,
    }
    base._assert_secret_free(report, api_key=api_key)  # noqa: SLF001
    require_exact_or_write(output_dir / "typed_api_rescue_report.json", report)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "tasks": len(plans),
                "verified_targets": len(verified_ids),
                "retry_eligible": len(retry_eligible),
                "estimated_usd": report["budget_charged"]["estimated_usd"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--phase", choices=PHASES, required=True)
    pre.add_argument("--local_harvest_report", required=True)
    pre.add_argument("--expected_local_harvest_report_sha256", required=True)
    pre.add_argument("--expected_local_harvest_journal_sha256", required=True)
    pre.add_argument("--local_harvest_targets", required=True)
    pre.add_argument("--expected_local_harvest_targets_sha256", required=True)
    pre.add_argument("--existing_direct_manifest", required=True)
    pre.add_argument("--expected_existing_direct_manifest_sha256", required=True)
    pre.add_argument("--gold_train_jsonl", required=True)
    pre.add_argument("--expected_gold_train_sha256", required=True)
    pre.add_argument("--gold_f2_jsonl", required=True)
    pre.add_argument("--expected_gold_f2_sha256", required=True)
    pre.add_argument("--gold_train_rows", dest="expected_gold_rows", type=int, default=2776)
    pre.add_argument("--heldout_jsonl", required=True)
    pre.add_argument("--expected_heldout_sha256", required=True)
    pre.add_argument("--expected_heldout_rows", type=int, default=175)
    pre.add_argument("--visible_split_manifest", required=True)
    pre.add_argument("--expected_visible_split_manifest_sha256", required=True)
    pre.add_argument("--visible_train", required=True)
    pre.add_argument("--expected_visible_train_sha256", required=True)
    pre.add_argument("--private_split_holdback", required=True)
    pre.add_argument("--expected_private_split_holdback_sha256", required=True)
    pre.add_argument("--visible_projection_report", required=True)
    pre.add_argument("--expected_visible_projection_report_sha256", required=True)
    pre.add_argument("--visible_projection_journal", required=True)
    pre.add_argument("--expected_visible_projection_journal_sha256", required=True)
    pre.add_argument("--cohort_index", type=int, default=0)
    pre.add_argument("--fixed_kimi_cohort_limit", type=int, default=0)
    pre.add_argument("--budget_skipped_kimi_retry_tasks", type=int, default=0)
    pre.add_argument("--budget_skipped_kimi_retry_task_ids_sha256", default="")
    pre.add_argument("--plan_only_output", default="")
    extras, remaining = pre.parse_known_args(argv)
    args = base.parse_args(remaining)
    for name, value in vars(extras).items():
        setattr(args, name, value)
    for name in (
        "expected_local_harvest_report_sha256",
        "expected_local_harvest_journal_sha256",
        "expected_local_harvest_targets_sha256",
        "expected_existing_direct_manifest_sha256",
        "expected_gold_train_sha256",
        "expected_gold_f2_sha256",
        "expected_heldout_sha256",
        "expected_visible_split_manifest_sha256",
        "expected_visible_train_sha256",
        "expected_private_split_holdback_sha256",
        "expected_visible_projection_report_sha256",
        "expected_visible_projection_journal_sha256",
    ):
        _require_digest(getattr(args, name), name)
    if args.cohort_index < 0:
        raise ValueError("cohort index must be non-negative")
    path_aliases = (
        (args.rollout_file, args.visible_train, "rollout_file/visible_train"),
        (args.f2_jsonl, args.gold_f2_jsonl, "f2_jsonl/gold_f2_jsonl"),
        (
            args.private_holdback,
            args.private_split_holdback,
            "private_holdback/private_split_holdback",
        ),
    )
    for left, right, label in path_aliases:
        if Path(left).expanduser().resolve() != Path(right).expanduser().resolve():
            raise ValueError(f"{label} aliases must identify the same pinned file")
    if (
        args.expected_rollout_sha256 != args.expected_visible_train_sha256
        or args.expected_f2_sha256 != args.expected_gold_f2_sha256
        or args.expected_private_holdback_sha256
        != args.expected_private_split_holdback_sha256
    ):
        raise ValueError("base compatibility digest aliases differ")
    validate_phase_profile(args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
