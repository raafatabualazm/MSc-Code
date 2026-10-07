#!/usr/bin/env python3
"""Full-TRAIN local rejection-sampling harvest for typed T5Gemma 2.

This stage samples exactly four independent direct candidates from the sealed
typed opaque-contract + F2 source for every clean TRAIN task that was not used
by the preceding 225-row direct RS-SFT stage.  All four candidates are produced
before the task's complete TRAIN acceptance suite is consulted.  The suite is
then used only as a binary transfer gate (two stability runs); it cannot cause
another generation and its text/diagnostics are never persisted.

The output contains at most one verified direct target per source task.  There
are no repair-conditioned rows, gold replay rows, frontier calls, or development
tasks.  An append-only hash-chained journal provides exact resume semantics.
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

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
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
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_local_rs_sft_pilot as local_harvest
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_contract_sft as typed_sft
from scripts.training import t5gemma2_typed_direct_rs_sft as typed_direct
from scripts.training.seq2seq_verpo_core import sha256_text


RUN_SCHEMA = "t5gemma2-typed-local-direct-harvest-run-v1"
JOURNAL_SCHEMA = "t5gemma2-typed-local-direct-harvest-journal-v1"
TARGET_SCHEMA = "t5gemma2-typed-local-direct-harvest-target-v1"
REPORT_SCHEMA = "t5gemma2-typed-local-direct-harvest-report-v1"
SCHEDULE_SCHEMA = "t5gemma2-typed-local-direct-harvest-schedule-v1"

CONTAMINATED_TRAIN_TASK_ID = typed_direct.CONTAMINATED_TRAIN_TASK_ID
EXPECTED_INPUT_ROWS = 2776
EXPECTED_CLEAN_ROWS = 2775
EXPECTED_PREVIOUS_DIRECT_TASKS = 225
EXPECTED_SCHEDULED_TASKS = 2550
EXPECTED_SAMPLES = 4
EXPECTED_MAX_SOURCE_TOKENS = 32768
EXPECTED_MAX_NEW_TOKENS = 4096
EXPECTED_STABILITY_RUNS = 2
EXPECTED_CHECKPOINT_UPDATE = 58
TYPED_SFT_CHECKPOINT_UPDATE = 348
CHECKPOINT_STAGES = {
    "typed_direct": {
        "run_schema": typed_direct.RUN_SCHEMA,
        "checkpoint_schema": typed_direct.CHECKPOINT_SCHEMA,
        "update": EXPECTED_CHECKPOINT_UPDATE,
    },
    "typed_sft": {
        "run_schema": typed_sft.RUN_SCHEMA,
        "checkpoint_schema": typed_sft.CHECKPOINT_SCHEMA,
        "update": TYPED_SFT_CHECKPOINT_UPDATE,
    },
}

_HEX_SHA256 = frozenset("0123456789abcdef")
_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "tests",
        "acceptance_tests",
        "feedback_tests",
        "reward_holdback_tests",
        "private_tests",
        "private_diagnostic",
        "holdback_diagnostic",
    }
)


@dataclass(frozen=True)
class HarvestTask:
    task_id: str
    source: str
    source_sha256: str
    f2_row: dict[str, Any]
    gold_target_sha256: str
    typed_contract_sha256: str


@dataclass(frozen=True)
class PrivateAcceptanceGate:
    task_id: str
    tests: str
    tests_sha256: str


@dataclass(frozen=True)
class Evaluation:
    compiled: bool
    passed: bool
    diagnostic: str


GenerateFn = Callable[[str, int, int], list[dict[str, Any]]]
EvaluateFn = Callable[[str, str, str], Evaluation]
VerifyFn = Callable[[str, str, str], bool]

_PREFLIGHT_ONLY_CHECKPOINT_KEYS = frozenset(
    {"checkpoint_stage", "checkpoint_update", "training_state_sha256"}
)


def _require_sha256(value: str, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in _HEX_SHA256 for char in digest):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return digest


def _assert_no_private_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        leaked = _FORBIDDEN_OUTPUT_KEYS & set(map(str, value))
        if leaked:
            raise ValueError(f"private payload key serialized: {sorted(leaked)}")
        for child in value.values():
            _assert_no_private_payload(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            _assert_no_private_payload(child)


def _complete_tests(row: Mapping[str, Any], task_id: str) -> str:
    tests = row.get("acceptance_tests") or row.get("tests")
    if not isinstance(tests, str) or not tests.strip():
        raise ValueError(f"{task_id}: complete TRAIN acceptance tests are absent")
    return tests


def validate_checkpoint(
    checkpoint: str | Path,
    *,
    checkpoint_stage: str,
    expected_update: int,
    expected_run_contract_sha256: str,
    expected_run_contract_file_sha256: str,
    expected_training_state_sha256: str,
    expected_adapter_weights_sha256: str,
    expected_adapter_config_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Bind the harvest to the exact completed typed-direct RS-SFT adapter."""

    stage = CHECKPOINT_STAGES.get(checkpoint_stage)
    if stage is None:
        raise ValueError("checkpoint_stage must be typed_direct or typed_sft")
    if expected_update != stage["update"]:
        raise ValueError(
            f"{checkpoint_stage} requires checkpoint update {stage['update']}"
        )
    path = Path(checkpoint).expanduser().resolve()
    contract, record = inference._checkpoint_record(path, "sft")  # noqa: SLF001
    state_path = path / "training_state.pt"
    if not state_path.is_file():
        raise ValueError("typed-direct harvest checkpoint lacks training_state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    adapter = record.get("adapter")
    contract_sha = canonical_sha256(contract)
    if (
        contract.get("schema") != stage["run_schema"]
        or not isinstance(state, Mapping)
        or state.get("schema") != stage["checkpoint_schema"]
        or int(state.get("update", -1)) != expected_update
        or state.get("run_contract_sha256") != contract_sha
        or path.name != base_sft._checkpoint_name(expected_update)  # noqa: SLF001
        or contract_sha
        != _require_sha256(
            expected_run_contract_sha256, "expected checkpoint run contract"
        )
        or sha256_file(path / "run_contract.json")
        != _require_sha256(
            expected_run_contract_file_sha256,
            "expected checkpoint run-contract file",
        )
        or sha256_file(state_path)
        != _require_sha256(
            expected_training_state_sha256,
            "expected checkpoint training state",
        )
        or not isinstance(adapter, Mapping)
        or adapter.get("adapter_weights_sha256")
        != _require_sha256(
            expected_adapter_weights_sha256, "expected adapter weights"
        )
        or adapter.get("adapter_config_sha256")
        != _require_sha256(
            expected_adapter_config_sha256, "expected adapter config"
        )
    ):
        raise ValueError(f"selected {checkpoint_stage} parent checkpoint differs")
    record["checkpoint_stage"] = checkpoint_stage
    record["checkpoint_update"] = expected_update
    record["training_state_sha256"] = sha256_file(state_path)
    return path, contract, record


def assert_loaded_checkpoint_matches_preflight(
    loaded_record: Mapping[str, Any], preflight_record: Mapping[str, Any]
) -> None:
    """Match the reloaded adapter while excluding preflight-only provenance.

    ``inference.load_policy`` reconstructs the checkpoint record from the run
    contract, tokenizer, and adapter files.  ``validate_checkpoint`` augments
    that record with the stage/update/training-state provenance that it checks
    separately; those three fields are therefore not returned by the loader.
    All loader-visible identity fields must still match exactly.
    """

    missing = _PREFLIGHT_ONLY_CHECKPOINT_KEYS.difference(preflight_record)
    if missing:
        raise ValueError(
            "typed-direct preflight record lacks provenance: "
            f"{sorted(missing)}"
        )
    expected_loaded_record = {
        key: value
        for key, value in preflight_record.items()
        if key not in _PREFLIGHT_ONLY_CHECKPOINT_KEYS
    }
    if dict(loaded_record) != expected_loaded_record:
        raise ValueError("loaded typed-direct adapter differs from preflight")


def _runtime_verify(code: str, tests: str, slot: str) -> bool:
    compiled, passed, _diagnostic, _details = evaluate_dart_jit_tests_detail(
        code,
        tests,
        slot,
        timeout=30,
        stability_runs=EXPECTED_STABILITY_RUNS,
    )
    return bool(compiled and passed)


def load_harvest_inputs(
    *,
    gold_train_jsonl: str | Path,
    gold_f2_jsonl: str | Path,
    expected_gold_train_sha256: str,
    expected_gold_f2_sha256: str,
    expected_gold_rows: int,
    heldout_jsonl: str | Path,
    expected_heldout_sha256: str,
    expected_heldout_rows: int,
) -> tuple[list[HarvestTask], dict[str, PrivateAcceptanceGate], dict[str, Any]]:
    """Build the clean typed source universe while keeping tests private."""

    if expected_gold_rows != EXPECTED_INPUT_ROWS:
        raise ValueError("typed local harvest requires the exact 2776-row input")
    typed_pairs, typed_manifest = typed_sft.load_typed_text_pairs(
        gold_train_jsonl,
        gold_f2_jsonl,
        expected_dataset_sha256=expected_gold_train_sha256,
        expected_f2_sha256=expected_gold_f2_sha256,
        expected_rows=expected_gold_rows,
        heldout_path=heldout_jsonl,
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
        exclude_train_task_ids=[CONTAMINATED_TRAIN_TASK_ID],
        allow_unpinned_inputs=False,
    )
    train_rows = base_sft._read_jsonl(gold_train_jsonl)  # noqa: SLF001
    f2_rows = base_sft._read_jsonl(gold_f2_jsonl)  # noqa: SLF001
    if len(train_rows) != expected_gold_rows or len(f2_rows) != expected_gold_rows:
        raise ValueError("typed local harvest input row counts differ")
    train_ids = [
        base_sft._identity(row, index)  # noqa: SLF001
        for index, row in enumerate(train_rows)
    ]
    f2_ids = [
        base_sft._identity(row, index)  # noqa: SLF001
        for index, row in enumerate(f2_rows)
    ]
    if train_ids != f2_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("gold TRAIN/F2 identities or order differ")

    filtered = [
        (task_id, train, f2)
        for task_id, train, f2 in zip(train_ids, train_rows, f2_rows, strict=True)
        if task_id != CONTAMINATED_TRAIN_TASK_ID
    ]
    if len(filtered) != EXPECTED_CLEAN_ROWS or len(typed_pairs) != EXPECTED_CLEAN_ROWS:
        raise ValueError("clean typed TRAIN must contain exactly 2775 rows")
    if [pair.task_id for pair in typed_pairs] != [item[0] for item in filtered]:
        raise ValueError("typed source order differs from clean TRAIN")

    tasks: list[HarvestTask] = []
    gates: dict[str, PrivateAcceptanceGate] = {}
    contract_records: list[dict[str, Any]] = []
    for pair, (task_id, train_row, f2_row) in zip(
        typed_pairs, filtered, strict=True
    ):
        target = base_sft._target_source(train_row, task_id)  # noqa: SLF001
        rebuilt_source, contract = typed_sft.build_typed_encoder_source(
            f2_row, task_id, target
        )
        if rebuilt_source != pair.source or sha256_text(rebuilt_source) != pair.source_sha256:
            raise ValueError(f"{task_id}: typed source builder drifted")
        tests = _complete_tests(train_row, task_id)
        tests_sha = sha256_text(tests)
        tasks.append(
            HarvestTask(
                task_id=task_id,
                source=pair.source,
                source_sha256=pair.source_sha256,
                f2_row=dict(f2_row),
                gold_target_sha256=sha256_text(target),
                typed_contract_sha256=str(contract["opaque_signature_sha256"]),
            )
        )
        gates[task_id] = PrivateAcceptanceGate(
            task_id=task_id,
            tests=tests,
            tests_sha256=tests_sha,
        )
        contract_records.append(contract)
    input_record = {
        "gold_train": {
            "sha256": _require_sha256(
                expected_gold_train_sha256, "gold TRAIN digest"
            ),
            "rows": expected_gold_rows,
        },
        "gold_f2": {
            "sha256": _require_sha256(expected_gold_f2_sha256, "gold F2 digest"),
            "rows": expected_gold_rows,
        },
        "heldout": typed_manifest["heldout"],
        "typed_manifest_sha256": canonical_sha256(typed_manifest),
        "typed_contracts_sha256": canonical_sha256(contract_records),
        "clean_rows": len(tasks),
        "clean_task_ids_sha256": canonical_sha256([task.task_id for task in tasks]),
        "clean_source_sha256s_sha256": canonical_sha256(
            [task.source_sha256 for task in tasks]
        ),
        "complete_acceptance_sha256s_sha256": canonical_sha256(
            [gates[task.task_id].tests_sha256 for task in tasks]
        ),
        "known_contaminant_excluded": CONTAMINATED_TRAIN_TASK_ID,
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "complete_acceptance_model_visible": False,
        "heldout_175_opened": False,
    }
    return tasks, gates, input_record


def load_previous_direct_exclusions(
    *,
    gold_train_jsonl: str | Path,
    gold_f2_jsonl: str | Path,
    expected_gold_train_sha256: str,
    expected_gold_f2_sha256: str,
    expected_gold_rows: int,
    heldout_jsonl: str | Path,
    expected_heldout_sha256: str,
    expected_heldout_rows: int,
    local_reports: Sequence[tuple[Path, str]],
    api_reports: Sequence[tuple[Path, str]],
    verify: VerifyFn | None = None,
    verification_workers: int = typed_direct.FULL_VERIFY_WORKERS,
) -> tuple[set[str], dict[str, Any]]:
    """Validate the exact 225-row predecessor evidence before exclusion."""

    if len(local_reports) != 4 or len(api_reports) != 7:
        raise ValueError("typed local harvest requires exactly 4 local and 7 API ledgers")
    pairs, manifest = typed_direct.build_typed_direct_pairs(
        gold_train_jsonl=Path(gold_train_jsonl),
        gold_f2_jsonl=Path(gold_f2_jsonl),
        expected_gold_train_sha256=expected_gold_train_sha256,
        expected_gold_f2_sha256=expected_gold_f2_sha256,
        expected_gold_rows=expected_gold_rows,
        heldout_jsonl=Path(heldout_jsonl),
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
        local_reports=local_reports,
        api_reports=api_reports,
        warmstart=typed_direct._source_sft_identity(),  # noqa: SLF001
        gold_replay_ratio=0.0,
        gold_replay_rows=0,
        min_verified_direct_targets=EXPECTED_PREVIOUS_DIRECT_TASKS,
        min_repair_conditioned_targets=0,
        allow_exploratory_inputs=False,
        require_local_production_floor=False,
        seed=42,
        verify=verify,
        verification_workers=verification_workers,
    )
    task_ids = {pair.source_task_id for pair in pairs}
    if len(task_ids) != EXPECTED_PREVIOUS_DIRECT_TASKS:
        raise ValueError("predecessor direct task set is not exactly 225 unique tasks")
    record = {
        "schema": typed_direct.DATASET_SCHEMA,
        "rows": len(pairs),
        "task_ids_sha256": canonical_sha256(sorted(task_ids)),
        "manifest_sha256": canonical_sha256(manifest),
        "composition": manifest["composition"],
        "full_acceptance_reverification": manifest[
            "full_acceptance_reverification"
        ],
        "reports_sha256": canonical_sha256(manifest["reports"]),
    }
    return task_ids, record


def build_schedule(
    tasks: Sequence[HarvestTask], *, excluded_task_ids: set[str], seed: int
) -> list[HarvestTask]:
    if len(tasks) != EXPECTED_CLEAN_ROWS or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("typed local harvest requires 2775 unique clean tasks")
    if seed < 0:
        raise ValueError("schedule seed must be non-negative")
    known = {task.task_id for task in tasks}
    unknown = sorted(excluded_task_ids - known)
    if unknown:
        raise ValueError("excluded direct task is outside clean TRAIN: " + unknown[0])
    if len(excluded_task_ids) != EXPECTED_PREVIOUS_DIRECT_TASKS:
        raise ValueError("typed local harvest must exclude exactly 225 source tasks")
    scheduled = [task for task in tasks if task.task_id not in excluded_task_ids]
    scheduled.sort(
        key=lambda task: canonical_sha256(
            {"schema": SCHEDULE_SCHEMA, "seed": seed, "task_id": task.task_id}
        )
    )
    if len(scheduled) != EXPECTED_SCHEDULED_TASKS:
        raise ValueError("typed local residual schedule must contain exactly 2550 tasks")
    return scheduled


def derived_seed(seed: int, *, task_position: int) -> int:
    if seed < 0 or task_position < 0:
        raise ValueError("invalid deterministic seed coordinates")
    digest = canonical_sha256(
        {
            "schema": RUN_SCHEMA,
            "seed": seed,
            "task_position": task_position,
            "phase": "direct_base_k4",
        }
    )
    return int(digest[:16], 16) % (2**63 - 1)


def _normalize_generated(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(values) != EXPECTED_SAMPLES:
        raise ValueError(
            f"generator returned {len(values)} rows, wanted {EXPECTED_SAMPLES}"
        )
    output: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        code = str(value.get("text") or value.get("code") or "").strip()
        generation = {
            key: item
            for key, item in dict(value).items()
            if key not in {"text", "code"}
        }
        output.append(
            {
                "origin": "local_student_direct",
                "sample_index": index,
                "code": code,
                "code_sha256": sha256_text(code),
                "generation": generation,
            }
        )
    return output


def _evaluate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    gate: PrivateAcceptanceGate,
    evaluate: EvaluateFn,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("evaluation workers must be positive")

    def one(position: int) -> tuple[int, Evaluation]:
        result = evaluate(
            str(candidates[position]["code"]),
            gate.tests,
            f"typed-local-{gate.task_id}-{position}",
        )
        if not isinstance(result, Evaluation):
            raise TypeError("candidate evaluator must return Evaluation")
        return position, result

    if workers == 1:
        outcomes = [one(index) for index in range(len(candidates))]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as pool:
            outcomes = list(pool.map(one, range(len(candidates))))
    by_position = {position: result for position, result in outcomes}
    output: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        result = by_position[position]
        row = {
            **dict(candidate),
            # Kept under the established binary field name so the typed API
            # cascade can select all-zero terminals.  The run contract states
            # that this is the complete private acceptance result.
            "visible": {
                "compiled": bool(result.compiled),
                "passed": bool(result.compiled and result.passed),
            },
        }
        output.append(row)
    return output


def process_task(
    *,
    task: HarvestTask,
    gate: PrivateAcceptanceGate,
    task_position: int,
    seed: int,
    generate: GenerateFn,
    evaluate: EvaluateFn,
    evaluation_workers: int,
) -> dict[str, Any]:
    if gate.task_id != task.task_id:
        raise ValueError(f"{task.task_id}: private gate identity differs")
    # The privacy boundary is deliberate: all K generations finish before any
    # complete acceptance test runs, and its outcome cannot trigger more work.
    candidates = _normalize_generated(
        generate(task.source, EXPECTED_SAMPLES, derived_seed(seed, task_position=task_position))
    )
    candidates = _evaluate_candidates(
        candidates,
        gate=gate,
        evaluate=evaluate,
        workers=evaluation_workers,
    )
    passing = [candidate for candidate in candidates if candidate["visible"]["passed"]]
    selected: dict[str, Any] | None = None
    if passing:
        candidate = passing[0]
        selected = {
            "schema": TARGET_SCHEMA,
            "task_id": task.task_id,
            "code": candidate["code"],
            "code_sha256": candidate["code_sha256"],
            "source_sha256": task.source_sha256,
            "origin": "local_student_direct",
            "full_acceptance_passed": True,
            "stability_runs": EXPECTED_STABILITY_RUNS,
        }
    event = {
        "event": "task_terminal",
        "schema": JOURNAL_SCHEMA,
        "task_position": task_position,
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "typed_contract_sha256": task.typed_contract_sha256,
        "complete_acceptance_sha256": gate.tests_sha256,
        "base_candidates": candidates,
        "repair_groups": [],
        "visible_unique_passes": len({
            candidate["code_sha256"]
            for candidate in passing
        }),
        "selected_target": selected,
        "all_generation_completed_before_private_gate": True,
        "private_feedback_serialized_to_model": False,
        "private_failure_triggers_generation": False,
        "binary_field_semantics": "complete_train_acceptance_private_gate",
    }
    _assert_no_private_payload(event)
    return event


def validate_journal_state(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    scheduled_tasks: Sequence[HarvestTask],
    gates: Mapping[str, PrivateAcceptanceGate],
) -> tuple[list[dict[str, Any]], bool]:
    if not events:
        return [], False
    header = events[0]
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("typed local harvest journal header differs")
    terminals: list[dict[str, Any]] = []
    complete = False
    for event in events[1:]:
        _assert_no_private_payload(event)
        if event.get("event") == "complete":
            if complete or len(terminals) != len(scheduled_tasks):
                raise ValueError("typed local harvest completion is early/duplicate")
            if (
                event.get("schema") != JOURNAL_SCHEMA
                or event.get("tasks") != len(scheduled_tasks)
                or event.get("terminal_task_ids_sha256")
                != canonical_sha256([row["task_id"] for row in terminals])
            ):
                raise ValueError("typed local harvest completion digest differs")
            complete = True
            continue
        if complete or event.get("event") != "task_terminal":
            raise ValueError("typed local harvest event ordering differs")
        position = len(terminals)
        expected = scheduled_tasks[position]
        gate = gates[expected.task_id]
        if (
            event.get("schema") != JOURNAL_SCHEMA
            or event.get("task_position") != position
            or event.get("task_id") != expected.task_id
            or event.get("source_sha256") != expected.source_sha256
            or event.get("typed_contract_sha256") != expected.typed_contract_sha256
            or event.get("complete_acceptance_sha256") != gate.tests_sha256
            or event.get("repair_groups") != []
            or event.get("all_generation_completed_before_private_gate") is not True
            or event.get("private_feedback_serialized_to_model") is not False
            or event.get("private_failure_triggers_generation") is not False
            or event.get("binary_field_semantics")
            != "complete_train_acceptance_private_gate"
        ):
            raise ValueError(f"typed local harvest terminal {position} differs")
        candidates = event.get("base_candidates")
        if not isinstance(candidates, list) or len(candidates) != EXPECTED_SAMPLES:
            raise ValueError(f"typed local terminal {position} candidate count differs")
        passing: list[Mapping[str, Any]] = []
        unique_passing: set[str] = set()
        for sample_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"typed local terminal {position} candidate malformed")
            code = str(candidate.get("code") or "")
            digest = str(candidate.get("code_sha256") or "")
            binary = candidate.get("visible")
            if (
                candidate.get("origin") != "local_student_direct"
                or candidate.get("sample_index") != sample_index
                or sha256_text(code) != digest
                or not isinstance(binary, Mapping)
                or type(binary.get("compiled")) is not bool
                or type(binary.get("passed")) is not bool
                or (binary.get("passed") and not binary.get("compiled"))
            ):
                raise ValueError(f"typed local terminal {position} evidence malformed")
            if (
                "safe_compiler_feedback" in candidate
                or "safe_compiler_feedback_sha256" in candidate
                or "diagnostic" in candidate
            ):
                raise ValueError("typed local private-gate diagnostic was persisted")
            if binary.get("passed"):
                passing.append(candidate)
                unique_passing.add(digest)
        if event.get("visible_unique_passes") != len(unique_passing):
            raise ValueError(f"typed local terminal {position} pass accounting differs")
        selected = event.get("selected_target")
        first = passing[0] if passing else None
        if first is None:
            if selected is not None:
                raise ValueError(f"typed local terminal {position} selected a failure")
        elif (
            not isinstance(selected, Mapping)
            or selected.get("schema") != TARGET_SCHEMA
            or selected.get("task_id") != expected.task_id
            or selected.get("source_sha256") != expected.source_sha256
            or selected.get("code") != first.get("code")
            or selected.get("code_sha256") != first.get("code_sha256")
            or selected.get("origin") != "local_student_direct"
            or selected.get("full_acceptance_passed") is not True
            or selected.get("stability_runs") != EXPECTED_STABILITY_RUNS
        ):
            raise ValueError(f"typed local terminal {position} target differs")
        terminals.append(dict(event))
    return terminals, complete


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _runtime_evaluator(*, timeout: int) -> EvaluateFn:
    def evaluate(code: str, tests: str, slot: str) -> Evaluation:
        compiled, passed, diagnostic, _details = evaluate_dart_jit_tests_detail(
            code,
            tests,
            slot,
            timeout=timeout,
            stability_runs=EXPECTED_STABILITY_RUNS,
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
    delegate = local_harvest._runtime_generator(  # noqa: SLF001
        model=model,
        tokenizer=tokenizer,
        max_source_tokens=max_source_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        generation_batch_size=generation_batch_size,
    )

    def generate(source: str, count: int, seed: int) -> list[dict[str, Any]]:
        # Tokenize without truncation once and fail closed rather than silently
        # changing the typed+F2 view.  The delegated generation then uses the
        # exact same tokenizer and cap.
        raw = tokenizer(source, add_special_tokens=True, truncation=False)
        length = len(raw["input_ids"])
        if length > max_source_tokens:
            raise ValueError(
                f"typed source requires {length} tokens, cap={max_source_tokens}"
            )
        values = delegate(source, count, seed)
        if any(int(value.get("encoder_tokens", -1)) != length for value in values):
            raise ValueError("typed source token count changed during generation")
        return values

    return generate


def _build_context(args: argparse.Namespace) -> tuple[
    list[HarvestTask],
    dict[str, PrivateAcceptanceGate],
    list[HarvestTask],
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    tasks, gates, input_record = load_harvest_inputs(
        gold_train_jsonl=args.gold_train_jsonl,
        gold_f2_jsonl=args.gold_f2_jsonl,
        expected_gold_train_sha256=args.expected_gold_train_sha256,
        expected_gold_f2_sha256=args.expected_gold_f2_sha256,
        expected_gold_rows=args.expected_gold_rows,
        heldout_jsonl=args.heldout_jsonl,
        expected_heldout_sha256=args.expected_heldout_sha256,
        expected_heldout_rows=args.expected_heldout_rows,
    )
    local_reports = mixed._parse_pinned_specs(  # noqa: SLF001
        args.local_report, label="--local_report"
    )
    api_reports = mixed._parse_pinned_specs(  # noqa: SLF001
        args.api_report, label="--api_report"
    )
    excluded, exclusion_record = load_previous_direct_exclusions(
        gold_train_jsonl=args.gold_train_jsonl,
        gold_f2_jsonl=args.gold_f2_jsonl,
        expected_gold_train_sha256=args.expected_gold_train_sha256,
        expected_gold_f2_sha256=args.expected_gold_f2_sha256,
        expected_gold_rows=args.expected_gold_rows,
        heldout_jsonl=args.heldout_jsonl,
        expected_heldout_sha256=args.expected_heldout_sha256,
        expected_heldout_rows=args.expected_heldout_rows,
        local_reports=local_reports,
        api_reports=api_reports,
    )
    scheduled = build_schedule(tasks, excluded_task_ids=excluded, seed=args.seed)
    checkpoint, checkpoint_contract, checkpoint_record = validate_checkpoint(
        args.checkpoint,
        checkpoint_stage=args.checkpoint_stage,
        expected_update=args.expected_checkpoint_update,
        expected_run_contract_sha256=args.expected_checkpoint_run_contract_sha256,
        expected_run_contract_file_sha256=args.expected_checkpoint_run_contract_file_sha256,
        expected_training_state_sha256=args.expected_checkpoint_training_state_sha256,
        expected_adapter_weights_sha256=args.expected_checkpoint_adapter_weights_sha256,
        expected_adapter_config_sha256=args.expected_checkpoint_adapter_config_sha256,
    )
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "typed_source_builder_sha256": sha256_file(Path(typed_sft.__file__).resolve()),
        "input": input_record,
        "previous_direct_exclusion": exclusion_record,
        "checkpoint": checkpoint_record,
        "checkpoint_stage": args.checkpoint_stage,
        "checkpoint_contract_sha256": canonical_sha256(checkpoint_contract),
        "schedule": {
            "schema": SCHEDULE_SCHEMA,
            "seed": args.seed,
            "clean_train_rows": len(tasks),
            "excluded_previous_direct_tasks": len(excluded),
            "scheduled_tasks": len(scheduled),
            "task_ids_sha256": canonical_sha256(
                [task.task_id for task in scheduled]
            ),
            "source_sha256s_sha256": canonical_sha256(
                [task.source_sha256 for task in scheduled]
            ),
        },
        "sampling": {
            "samples_per_task": EXPECTED_SAMPLES,
            "repair_samples": 0,
            "max_repair_parents": 0,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_source_tokens": args.max_source_tokens,
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": args.generation_batch_size,
            "silent_source_truncation": False,
        },
        "verification": {
            "suite": "complete_train_acceptance",
            "stability_runs": EXPECTED_STABILITY_RUNS,
            "timeout_seconds": args.timeout,
            "workers": args.evaluation_workers,
            "all_k_generated_before_gate": True,
            "gate_can_only_reject_transfer": True,
            "diagnostics_persisted": False,
        },
        "outputs": {
            "at_most_one_direct_target_per_task": True,
            "repair_conditioned_rows": 0,
            "gold_replay_rows": 0,
        },
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "complete_acceptance_model_visible": False,
        "heldout_175_opened": False,
        "frontier_api_calls": False,
    }
    return tasks, gates, scheduled, contract, checkpoint, checkpoint_record


def load_completed_harvest_context(args: argparse.Namespace) -> tuple[
    list[HarvestTask],
    dict[str, PrivateAcceptanceGate],
    list[HarvestTask],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Strict adapter for a later typed API rescue cascade.

    Complete acceptance tests exist only in the returned private gate mapping.
    Callers must construct any provider-visible TRAIN checks independently.
    """

    tasks, gates, scheduled, contract, _checkpoint, _record = _build_context(args)
    journal_path = Path(args.output_dir).expanduser().resolve() / "harvest.journal.jsonl"
    events = load_journal(journal_path)
    terminals, complete = validate_journal_state(
        events, contract=contract, scheduled_tasks=scheduled, gates=gates
    )
    if not complete:
        raise ValueError("typed local harvest is not complete")
    source_record = journal_record(journal_path)
    source_record.pop("path", None)
    source_record.pop("chain_head_path", None)
    source_record.update(
        {
            "schema": JOURNAL_SCHEMA,
            "run_contract_sha256": canonical_sha256(contract),
            "production_floor_eligible": True,
            "source_journal_modified": False,
        }
    )
    return tasks, gates, scheduled, terminals, contract["input"], source_record


def _read_jsonl_allow_empty(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _pin_existing(path: str | Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    expected = _require_sha256(expected_sha256, f"{label} expected digest")
    if sha256_file(resolved) != expected:
        raise ValueError(f"{label} digest differs")
    return resolved


def load_completed_harvest_artifacts(
    *,
    report_path: str | Path,
    expected_report_sha256: str,
    journal_path: str | Path,
    expected_journal_sha256: str,
    targets_path: str | Path,
    expected_targets_sha256: str,
    gold_train_jsonl: str | Path,
    expected_gold_train_sha256: str,
    gold_f2_jsonl: str | Path,
    expected_gold_f2_sha256: str,
    heldout_jsonl: str | Path,
    expected_heldout_sha256: str,
    expected_gold_rows: int = EXPECTED_INPUT_ROWS,
    expected_heldout_rows: int = 175,
) -> tuple[
    list[HarvestTask],
    dict[str, PrivateAcceptanceGate],
    list[HarvestTask],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate a completed harvest without a GPU or predecessor checkpoints.

    The pinned report binds the immutable run contract, chain head, schedule,
    direct targets, and aligned F2 rows.  The typed source universe and private
    complete-acceptance gates are independently reconstructed from the pinned
    TRAIN inputs; neither a mutable checkpoint nor the predecessor ledgers are
    trusted during this handoff.
    """

    report_file = _pin_existing(
        report_path, expected_report_sha256, "typed local harvest report"
    )
    journal_file = _pin_existing(
        journal_path, expected_journal_sha256, "typed local harvest journal"
    )
    target_file = _pin_existing(
        targets_path, expected_targets_sha256, "typed local direct targets"
    )
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("typed local harvest report is malformed") from exc
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("production_floor_eligible") is not True
    ):
        raise ValueError("typed local harvest report is not completed production")
    outputs = report.get("outputs")
    report_journal = report.get("journal")
    if not isinstance(outputs, Mapping) or not isinstance(report_journal, Mapping):
        raise ValueError("typed local harvest report inventory is absent")
    target_record = outputs.get("direct_targets")
    f2_record = outputs.get("direct_f2")
    schedule_record = outputs.get("schedule_manifest")
    if not all(isinstance(item, Mapping) for item in (target_record, f2_record, schedule_record)):
        raise ValueError("typed local harvest output inventory is malformed")
    if (
        target_record.get("sha256") != sha256_file(target_file)
        or target_record.get("sha256") != expected_targets_sha256
        or Path(str(target_record.get("path") or "")).name != target_file.name
    ):
        raise ValueError("typed local direct-target report binding differs")

    sibling_dir = report_file.parent
    f2_file = sibling_dir / "direct_f2.jsonl"
    schedule_file = sibling_dir / "schedule_manifest.jsonl"
    for path, record, label in (
        (f2_file, f2_record, "direct F2"),
        (schedule_file, schedule_record, "schedule manifest"),
    ):
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or Path(str(record.get("path") or "")).name != path.name
        ):
            raise ValueError(f"typed local {label} report binding differs")

    events = load_journal(journal_file)
    actual_journal = journal_record(journal_file)
    for key in ("sha256", "chain_head_sha256", "event_count", "head_event_sha256"):
        if actual_journal.get(key) != report_journal.get(key):
            raise ValueError(f"typed local harvest journal {key} differs")
    if Path(str(report_journal.get("path") or "")).name != journal_file.name:
        raise ValueError("typed local harvest journal basename differs")
    if not events:
        raise ValueError("typed local harvest journal is empty")
    header = events[0]
    contract = header.get("contract")
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or not isinstance(contract, Mapping)
        or contract.get("schema") != RUN_SCHEMA
        or header.get("contract_sha256") != canonical_sha256(contract)
        or report.get("run_contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("typed local harvest report/journal contract differs")

    tasks, gates, input_record = load_harvest_inputs(
        gold_train_jsonl=gold_train_jsonl,
        gold_f2_jsonl=gold_f2_jsonl,
        expected_gold_train_sha256=expected_gold_train_sha256,
        expected_gold_f2_sha256=expected_gold_f2_sha256,
        expected_gold_rows=expected_gold_rows,
        heldout_jsonl=heldout_jsonl,
        expected_heldout_sha256=expected_heldout_sha256,
        expected_heldout_rows=expected_heldout_rows,
    )
    if contract.get("input") != input_record:
        raise ValueError("typed local harvest reconstructed input binding differs")
    schedule_rows = _read_jsonl_allow_empty(schedule_file, "schedule manifest")
    if len(schedule_rows) != EXPECTED_SCHEDULED_TASKS:
        raise ValueError("typed local harvest schedule artifact is not 2550 rows")
    if schedule_record.get("rows") != len(schedule_rows):
        raise ValueError("typed local harvest schedule row count differs")
    task_by_id = {task.task_id: task for task in tasks}
    scheduled: list[HarvestTask] = []
    seen: set[str] = set()
    for position, row in enumerate(schedule_rows):
        task_id = str(row.get("task_id") or "")
        task = task_by_id.get(task_id)
        if (
            row.get("schema") != SCHEDULE_SCHEMA
            or row.get("position") != position
            or task is None
            or task_id in seen
            or row.get("source_sha256") != task.source_sha256
            or row.get("typed_contract_sha256") != task.typed_contract_sha256
            or row.get("complete_acceptance_sha256") != gates[task_id].tests_sha256
        ):
            raise ValueError(f"typed local schedule row {position} differs")
        seen.add(task_id)
        scheduled.append(task)
    contract_schedule = contract.get("schedule")
    if not isinstance(contract_schedule, Mapping):
        raise ValueError("typed local contract schedule is absent")
    seed = int(contract_schedule.get("seed", -1))
    if (
        contract_schedule.get("schema") != SCHEDULE_SCHEMA
        or contract_schedule.get("clean_train_rows") != EXPECTED_CLEAN_ROWS
        or contract_schedule.get("excluded_previous_direct_tasks")
        != EXPECTED_PREVIOUS_DIRECT_TASKS
        or contract_schedule.get("scheduled_tasks") != EXPECTED_SCHEDULED_TASKS
        or contract_schedule.get("task_ids_sha256")
        != canonical_sha256([task.task_id for task in scheduled])
        or contract_schedule.get("source_sha256s_sha256")
        != canonical_sha256([task.source_sha256 for task in scheduled])
    ):
        raise ValueError("typed local contract schedule binding differs")
    excluded = set(task_by_id) - seen
    if build_schedule(tasks, excluded_task_ids=excluded, seed=seed) != scheduled:
        raise ValueError("typed local schedule cannot be reconstructed")
    exclusion = contract.get("previous_direct_exclusion")
    if (
        not isinstance(exclusion, Mapping)
        or exclusion.get("rows") != EXPECTED_PREVIOUS_DIRECT_TASKS
        or exclusion.get("task_ids_sha256") != canonical_sha256(sorted(excluded))
    ):
        raise ValueError("typed local predecessor exclusion binding differs")

    terminals, complete = validate_journal_state(
        events, contract=contract, scheduled_tasks=scheduled, gates=gates
    )
    if not complete:
        raise ValueError("typed local harvest journal is incomplete")
    for position, (row, terminal) in enumerate(
        zip(schedule_rows, terminals, strict=True)
    ):
        selected = terminal.get("selected_target")
        if (
            row.get("candidate_code_sha256s")
            != [
                candidate["code_sha256"]
                for candidate in terminal["base_candidates"]
            ]
            or row.get("unique_full_passes")
            != terminal.get("visible_unique_passes")
            or row.get("selected_target_sha256")
            != (
                selected.get("code_sha256")
                if isinstance(selected, Mapping)
                else None
            )
        ):
            raise ValueError(
                f"typed local schedule evidence row {position} differs"
            )
    targets = _read_jsonl_allow_empty(target_file, "direct targets")
    direct_f2 = _read_jsonl_allow_empty(f2_file, "direct F2")
    expected_targets: list[dict[str, Any]] = []
    expected_f2: list[dict[str, Any]] = []
    for task, terminal in zip(scheduled, terminals, strict=True):
        selected = terminal.get("selected_target")
        if not isinstance(selected, Mapping):
            continue
        code = str(selected["code"])
        expected_targets.append(
            {
                "schema": TARGET_SCHEMA,
                "task_id": task.task_id,
                "dart_source": code,
                "dart_source_sha256": sha256_text(code),
                "source_sha256": task.source_sha256,
                "origin": "local_student_direct",
                "full_acceptance_passed": True,
                "stability_runs": EXPECTED_STABILITY_RUNS,
                "repair_conditioned": False,
                "gold_replay": False,
            }
        )
        expected_f2.append(dict(task.f2_row))
    if targets != expected_targets or direct_f2 != expected_f2:
        raise ValueError("typed local direct outputs differ from verified terminals")
    report_schedule = report.get("schedule")
    report_accepted = report.get("accepted")
    report_verification = report.get("verification")
    report_privacy = report.get("privacy")
    report_checkpoint = report.get("checkpoint")
    contract_checkpoint = contract.get("checkpoint")
    exact_gold = sum(
        row["dart_source_sha256"] == task_by_id[row["task_id"]].gold_target_sha256
        for row in targets
    )
    if (
        not isinstance(report_checkpoint, Mapping)
        or not isinstance(contract_checkpoint, Mapping)
        or report_checkpoint != contract_checkpoint
        or not isinstance(report_schedule, Mapping)
        or report_schedule.get("clean_train_rows") != EXPECTED_CLEAN_ROWS
        or report_schedule.get("excluded_previous_direct_tasks")
        != EXPECTED_PREVIOUS_DIRECT_TASKS
        or report_schedule.get("tasks") != EXPECTED_SCHEDULED_TASKS
        or report_schedule.get("samples_per_task") != EXPECTED_SAMPLES
        or report_schedule.get("candidate_generations")
        != EXPECTED_SCHEDULED_TASKS * EXPECTED_SAMPLES
        or report_schedule.get("task_ids_sha256")
        != canonical_sha256([task.task_id for task in scheduled])
        or not isinstance(report_accepted, Mapping)
        or report_accepted.get("unique_direct_targets") != len(targets)
        or report_accepted.get("task_ids_sha256")
        != canonical_sha256([row["task_id"] for row in targets])
        or report_accepted.get("exact_gold_targets") != exact_gold
        or report_accepted.get("at_most_one_per_task") is not True
        or target_record.get("rows") != len(targets)
        or f2_record.get("rows") != len(direct_f2)
        or report.get("composition", {}).get("local_student_direct") != len(targets)
        or report.get("composition", {}).get("repair_conditioned") != 0
        or report.get("composition", {}).get("gold_replay") != 0
        or not isinstance(report_verification, Mapping)
        or report_verification.get("suite") != "complete_train_acceptance"
        or report_verification.get("stability_runs") != EXPECTED_STABILITY_RUNS
        or report_verification.get("tests_model_visible") is not False
        or report_verification.get("tests_persisted") is not False
        or report_verification.get("diagnostics_persisted") is not False
        or not isinstance(report_privacy, Mapping)
        or report_privacy.get("model_visible_fields")
        != ["opaque_typed_contract", "F2.text"]
        or report_privacy.get("complete_acceptance_model_visible") is not False
        or report_privacy.get("heldout_175_opened") is not False
        or report_privacy.get("frontier_api_calls") is not False
    ):
        raise ValueError("typed local report direct-output accounting differs")
    source_record = {
        key: value
        for key, value in actual_journal.items()
        if key not in {"path", "chain_head_path"}
    }
    source_record.update(
        {
            "schema": JOURNAL_SCHEMA,
            "run_contract_sha256": canonical_sha256(contract),
            "report_sha256": expected_report_sha256,
            "direct_targets_sha256": expected_targets_sha256,
            "production_floor_eligible": True,
            "source_journal_modified": False,
        }
    )
    return tasks, gates, scheduled, terminals, input_record, source_record


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("typed local rejection-sampling harvest requires CUDA")
    validate_dart_binary()
    tasks, gates, scheduled, contract, checkpoint, checkpoint_record = _build_context(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "harvest.journal.jsonl"
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
        events, contract=contract, scheduled_tasks=scheduled, gates=gates
    )
    if not complete:
        model, tokenizer, loaded_record = inference.load_policy(
            checkpoint=checkpoint,
            arm="sft",
            bf16=args.bf16,
            attn_implementation=args.attn_implementation,
        )
        assert_loaded_checkpoint_matches_preflight(loaded_record, checkpoint_record)
        generate = _runtime_generator(
            model=model,
            tokenizer=tokenizer,
            max_source_tokens=args.max_source_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            generation_batch_size=args.generation_batch_size,
        )
        evaluate = _runtime_evaluator(timeout=args.timeout)
        for position in range(len(terminals), len(scheduled)):
            task = scheduled[position]
            event = process_task(
                task=task,
                gate=gates[task.task_id],
                task_position=position,
                seed=args.seed,
                generate=generate,
                evaluate=evaluate,
                evaluation_workers=args.evaluation_workers,
            )
            terminal = append_event(journal_path, event)
            terminals.append(terminal)
            print(
                json.dumps(
                    {
                        "task": position + 1,
                        "tasks": len(scheduled),
                        "task_id": task.task_id,
                        "full_passes": event["visible_unique_passes"],
                        "accepted": event["selected_target"] is not None,
                        "accepted_total": sum(
                            row.get("selected_target") is not None for row in terminals
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
                "tasks": len(scheduled),
                "terminal_task_ids_sha256": canonical_sha256(
                    [row["task_id"] for row in terminals]
                ),
            },
        )
        events = load_journal(journal_path)
        terminals, complete = validate_journal_state(
            events, contract=contract, scheduled_tasks=scheduled, gates=gates
        )
    if not complete:
        raise RuntimeError("typed local harvest journal did not complete")

    task_by_id = {task.task_id: task for task in scheduled}
    targets: list[dict[str, Any]] = []
    direct_f2: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    exact_gold = 0
    for terminal in terminals:
        task = task_by_id[str(terminal["task_id"])]
        selected = terminal.get("selected_target")
        if isinstance(selected, Mapping):
            code = str(selected["code"])
            targets.append(
                {
                    "schema": TARGET_SCHEMA,
                    "task_id": task.task_id,
                    "dart_source": code,
                    "dart_source_sha256": sha256_text(code),
                    "source_sha256": task.source_sha256,
                    "origin": "local_student_direct",
                    "full_acceptance_passed": True,
                    "stability_runs": EXPECTED_STABILITY_RUNS,
                    "repair_conditioned": False,
                    "gold_replay": False,
                }
            )
            direct_f2.append(dict(task.f2_row))
            exact_gold += sha256_text(code) == task.gold_target_sha256
        schedule_rows.append(
            {
                "schema": SCHEDULE_SCHEMA,
                "position": int(terminal["task_position"]),
                "task_id": task.task_id,
                "source_sha256": task.source_sha256,
                "typed_contract_sha256": task.typed_contract_sha256,
                "complete_acceptance_sha256": gates[task.task_id].tests_sha256,
                "candidate_code_sha256s": [
                    candidate["code_sha256"] for candidate in terminal["base_candidates"]
                ],
                "unique_full_passes": terminal["visible_unique_passes"],
                "selected_target_sha256": (
                    selected["code_sha256"] if isinstance(selected, Mapping) else None
                ),
            }
        )
    outputs = {
        "direct_targets": output_dir / "direct_targets.jsonl",
        "direct_f2": output_dir / "direct_f2.jsonl",
        "schedule_manifest": output_dir / "schedule_manifest.jsonl",
    }
    rows_by_name = {
        "direct_targets": targets,
        "direct_f2": direct_f2,
        "schedule_manifest": schedule_rows,
    }
    for name, path in outputs.items():
        _atomic_write_jsonl(path, rows_by_name[name])
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "production_floor_eligible": True,
        "run_contract_sha256": canonical_sha256(contract),
        "checkpoint": checkpoint_record,
        "schedule": {
            "clean_train_rows": len(tasks),
            "excluded_previous_direct_tasks": EXPECTED_PREVIOUS_DIRECT_TASKS,
            "tasks": len(scheduled),
            "samples_per_task": EXPECTED_SAMPLES,
            "candidate_generations": len(scheduled) * EXPECTED_SAMPLES,
            "task_ids_sha256": canonical_sha256(
                [task.task_id for task in scheduled]
            ),
        },
        "accepted": {
            "unique_direct_targets": len(targets),
            "task_ids_sha256": canonical_sha256(
                [row["task_id"] for row in targets]
            ),
            "exact_gold_targets": exact_gold,
            "at_most_one_per_task": True,
        },
        "composition": {
            "local_student_direct": len(targets),
            "repair_conditioned": 0,
            "gold_replay": 0,
        },
        "verification": {
            "suite": "complete_train_acceptance",
            "stability_runs": EXPECTED_STABILITY_RUNS,
            "tests_model_visible": False,
            "tests_persisted": False,
            "diagnostics_persisted": False,
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows_by_name[name]),
            }
            for name, path in outputs.items()
        },
        "journal": journal_record(journal_path),
        "privacy": {
            "model_visible_fields": ["opaque_typed_contract", "F2.text"],
            "complete_acceptance_model_visible": False,
            "heldout_175_opened": False,
            "frontier_api_calls": False,
        },
    }
    _assert_no_private_payload(report)
    require_exact_or_write(output_dir / "harvest_report.json", report)
    print(
        json.dumps(
            {
                "tasks": len(scheduled),
                "accepted": len(targets),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gold_train_jsonl", required=True)
    parser.add_argument("--gold_f2_jsonl", required=True)
    parser.add_argument("--expected_gold_train_sha256", required=True)
    parser.add_argument("--expected_gold_f2_sha256", required=True)
    parser.add_argument("--expected_gold_rows", type=int, default=EXPECTED_INPUT_ROWS)
    parser.add_argument("--heldout_jsonl", required=True)
    parser.add_argument("--expected_heldout_sha256", required=True)
    parser.add_argument("--expected_heldout_rows", type=int, default=175)
    parser.add_argument("--local_report", action="append", default=[])
    parser.add_argument("--api_report", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint_stage",
        choices=sorted(CHECKPOINT_STAGES),
        default="typed_direct",
    )
    parser.add_argument(
        "--expected_checkpoint_update", type=int, default=EXPECTED_CHECKPOINT_UPDATE
    )
    parser.add_argument("--expected_checkpoint_run_contract_sha256", required=True)
    parser.add_argument("--expected_checkpoint_run_contract_file_sha256", required=True)
    parser.add_argument("--expected_checkpoint_training_state_sha256", required=True)
    parser.add_argument("--expected_checkpoint_adapter_weights_sha256", required=True)
    parser.add_argument("--expected_checkpoint_adapter_config_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_task", type=int, default=EXPECTED_SAMPLES)
    parser.add_argument("--max_source_tokens", type=int, default=EXPECTED_MAX_SOURCE_TOKENS)
    parser.add_argument("--max_new_tokens", type=int, default=EXPECTED_MAX_NEW_TOKENS)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--evaluation_workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=EXPECTED_STABILITY_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    fixed = {
        "expected_gold_rows": EXPECTED_INPUT_ROWS,
        "samples_per_task": EXPECTED_SAMPLES,
        "max_source_tokens": EXPECTED_MAX_SOURCE_TOKENS,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "stability_runs": EXPECTED_STABILITY_RUNS,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            parser.error(f"--{name} is fixed at {expected}")
    required_update = int(CHECKPOINT_STAGES[args.checkpoint_stage]["update"])
    if args.expected_checkpoint_update != required_update:
        parser.error(
            f"--checkpoint_stage {args.checkpoint_stage} requires "
            f"--expected_checkpoint_update {required_update}"
        )
    if len(args.local_report) != 4 or len(args.api_report) != 7:
        parser.error("exactly 4 local and 7 API predecessor reports are required")
    if (
        args.expected_heldout_rows <= 0
        or args.generation_batch_size <= 0
        or args.evaluation_workers <= 0
        or args.timeout <= 0
        or args.seed < 0
    ):
        parser.error("row, batch, worker, timeout, and seed controls are invalid")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top_p must lie in (0,1]")
    for name in (
        "expected_checkpoint_run_contract_sha256",
        "expected_checkpoint_run_contract_file_sha256",
        "expected_checkpoint_training_state_sha256",
        "expected_checkpoint_adapter_weights_sha256",
        "expected_checkpoint_adapter_config_sha256",
    ):
        try:
            _require_sha256(getattr(args, name), name)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
