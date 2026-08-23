#!/usr/bin/env python3
"""Union legacy and supplemental Qwen training artifacts over exactly 2,776 tasks.

The two parent harvests remain separate and independently auditable.  This
tool joins their already-built sequence-KL and CoT datasets, requires exact
K=8/Kcot=2 task grids, verifies source rows against the sealed 2,776-task fit
view, and publishes deterministic union datasets with cryptographic bindings
to every parent artifact and the heldout-exclusion derivation.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.direct_compact_causal import (  # noqa: E402
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.training.build_qwen_cot_sft import (  # noqa: E402
    BUILD_SCHEMA as COT_BUILD_SCHEMA,
    K_COT,
    SCHEDULE_SCHEMA as COT_SCHEDULE_SCHEMA,
    cot_coverage_gate,
)
from scripts.training.build_qwen_sequence_kd import (  # noqa: E402
    BUILD_SCHEMA as SEQUENCE_BUILD_SCHEMA,
    SCHEDULE_SCHEMA as SEQUENCE_SCHEDULE_SCHEMA,
    compact_ids_sha256,
    exact_output_seal,
    load_student_tokenizer,
    require_file_hash,
    strict_json,
    target_text,
)
from scripts.training.direct_compact_qwen_decompiler import (  # noqa: E402
    DIRECT_PROMPT_MODE_QWEN_COT_V1,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    ArtifactError,
    SAMPLES_PER_TASK,
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    read_jsonl,
    sha256_text,
    stable_sha256,
)
from scripts.training.prepare_qwen_2776_supplement import (  # noqa: E402
    EXPECTED_FIT_TASKS,
    EXPECTED_HOLDOUT_TASKS,
    EXPECTED_LEGACY_TASKS,
    EXPECTED_SUPPLEMENT_TASKS,
    MANIFEST_SCHEMA as DERIVATION_SCHEMA,
)


UNION_SCHEMA = "qwen-2776-training-artifact-union-v1"
EXPECTED_SEQUENCE_ROWS = EXPECTED_FIT_TASKS * SAMPLES_PER_TASK
EXPECTED_COT_ROWS = EXPECTED_FIT_TASKS * K_COT
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ParentPaths:
    label: str
    dataset: Path
    seal: Path
    schedule: Path
    manifest: Path
    expected_manifest_sha256: str


@dataclass
class ParentArtifact:
    paths: ParentPaths
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    schedule: list[dict[str, Any]]
    records: dict[str, dict[str, Any]]


def _add_parent_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    option = prefix.replace("_", "-")
    parser.add_argument(f"--{option}-jsonl", required=True, type=Path)
    parser.add_argument(f"--{option}-seal", required=True, type=Path)
    parser.add_argument(f"--{option}-schedule", required=True, type=Path)
    parser.add_argument(f"--{option}-manifest", required=True, type=Path)
    parser.add_argument(f"--expected-{option}-manifest-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--derivation-manifest", required=True, type=Path)
    parser.add_argument("--expected-derivation-manifest-sha256", required=True)
    parser.add_argument("--fit-compact-jsonl", required=True, type=Path)
    parser.add_argument("--fit-compact-seal", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    _add_parent_args(parser, "legacy_sequence")
    _add_parent_args(parser, "supplement_sequence")
    _add_parent_args(parser, "legacy_cot")
    _add_parent_args(parser, "supplement_cot")
    parser.add_argument("--sequence-output-jsonl", required=True, type=Path)
    parser.add_argument("--sequence-output-seal", required=True, type=Path)
    parser.add_argument("--sequence-output-schedule", required=True, type=Path)
    parser.add_argument("--sequence-output-manifest", required=True, type=Path)
    parser.add_argument("--cot-output-jsonl", required=True, type=Path)
    parser.add_argument("--cot-output-seal", required=True, type=Path)
    parser.add_argument("--cot-output-schedule", required=True, type=Path)
    parser.add_argument("--cot-output-manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def _parent_paths(args: argparse.Namespace, prefix: str) -> ParentPaths:
    return ParentPaths(
        label=prefix,
        dataset=getattr(args, f"{prefix}_jsonl").expanduser().resolve(),
        seal=getattr(args, f"{prefix}_seal").expanduser().resolve(),
        schedule=getattr(args, f"{prefix}_schedule").expanduser().resolve(),
        manifest=getattr(args, f"{prefix}_manifest").expanduser().resolve(),
        expected_manifest_sha256=getattr(
            args, f"expected_{prefix}_manifest_sha256"
        ),
    )


def _task_id(row: Mapping[str, Any], label: str) -> str:
    value = str(row.get("task_id") or row.get("id") or "")
    if not value:
        raise ArtifactError(f"{label} has no task_id")
    return value


def _record_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if (
        actual.get("sha256") != expected.get("sha256")
        or int(actual.get("size_bytes", -1))
        != int(expected.get("size_bytes", expected.get("bytes", -2)))
    ):
        raise ArtifactError(f"{label} differs from its parent build manifest")


def _load_parent(
    paths: ParentPaths,
    *,
    contract_path: Path,
    expected_schema: str,
) -> ParentArtifact:
    manifest_record = require_file_hash(
        paths.manifest,
        paths.expected_manifest_sha256,
        f"{paths.label} manifest",
    )
    manifest = strict_json(paths.manifest)
    if manifest.get("schema") != expected_schema:
        raise ArtifactError(
            f"{paths.label} manifest schema={manifest.get('schema')!r}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ArtifactError(f"{paths.label} manifest has no outputs")
    records = {
        "dataset": file_record(paths.dataset),
        "seal": file_record(paths.seal),
        "schedule": file_record(paths.schedule),
        "manifest": manifest_record,
    }
    _record_matches(records["dataset"], outputs.get("dataset") or {}, paths.label)
    _record_matches(
        records["seal"],
        outputs.get("standard_direct_compact_seal") or {},
        f"{paths.label} seal",
    )
    _record_matches(
        records["schedule"],
        outputs.get("schedule") or {},
        f"{paths.label} schedule",
    )
    validate_join_seal(
        paths.dataset, paths.seal, contract_path, expected_role="fit"
    )
    rows = read_jsonl(paths.dataset)
    schedule = read_jsonl(paths.schedule)
    if len(rows) != len(schedule):
        raise ArtifactError(f"{paths.label} dataset/schedule row count differs")
    if stable_sha256(schedule) != manifest.get("schedule_sha256"):
        raise ArtifactError(f"{paths.label} schedule semantic hash differs")
    return ParentArtifact(paths, manifest, rows, schedule, records)


def validate_exact_grid(
    *,
    label: str,
    rows: Sequence[dict[str, Any]],
    schedule: Sequence[dict[str, Any]],
    expected_task_ids: set[str],
    samples_per_task: int,
    schedule_schema: str,
    contract: DirectCompactContract,
    fit_by_task: Mapping[str, tuple[int, dict[str, Any]]],
    cot: bool,
) -> list[dict[str, Any]]:
    """Validate one parent and return row/schedule pairs for unioning."""

    expected_grid = {
        (task_id, sample_index)
        for task_id in expected_task_ids
        for sample_index in range(samples_per_task)
    }
    seen_grid: set[tuple[str, int]] = set()
    seen_candidates: set[str] = set()
    paired: list[dict[str, Any]] = []
    for position, (row, item) in enumerate(zip(rows, schedule, strict=True)):
        if (
            item.get("schema") != schedule_schema
            or int(item.get("position", -1)) != position
        ):
            raise ArtifactError(f"{label} schedule breaks at position {position}")
        task_id = str(item.get("task_id") or "")
        sample_index = int(item.get("sample_index", -1))
        candidate_id = str(item.get("candidate_id") or "")
        key = (task_id, sample_index)
        if key in seen_grid or not candidate_id or candidate_id in seen_candidates:
            raise ArtifactError(
                f"{label} duplicate/missing slot or candidate at {position}"
            )
        if task_id not in expected_task_ids:
            raise ArtifactError(f"{label} task {task_id} is outside parent scope")
        seen_grid.add(key)
        seen_candidates.add(candidate_id)
        contract.validate_row(row, f"{label}-row-{position}")
        if _task_id(row, f"{label}-row-{position}") != task_id:
            raise ArtifactError(f"{label} row/schedule task differs")
        full_index, fit_row = fit_by_task[task_id]
        expected_compact = compact_ids_sha256(fit_row, f"fit-{task_id}")
        observed_compact = compact_ids_sha256(row, f"{label}-{task_id}")
        if (
            observed_compact != expected_compact
            or str(item.get("compact_ids_sha256") or "") != expected_compact
        ):
            raise ArtifactError(
                f"{label} source differs from sealed fit row for {task_id}"
            )
        target = target_text(row, f"{label}-{task_id}-{sample_index}")
        target_field = "cot_target_sha256" if cot else "target_sha256"
        if sha256_text(target) != str(item.get(target_field) or ""):
            raise ArtifactError(
                f"{label} target hash differs for {task_id}/{sample_index}"
            )
        if cot:
            if (
                row.get("direct_prompt_mode")
                != DIRECT_PROMPT_MODE_QWEN_COT_V1
                or sample_index not in range(K_COT)
            ):
                raise ArtifactError(f"{label} has an invalid CoT mode/slot")
        elif item.get("kind") != "teacher_draw":
            raise ArtifactError(f"{label} contains a non-teacher sequence row")
        paired.append(
            {
                "task_id": task_id,
                "sample_index": sample_index,
                "candidate_id": candidate_id,
                "full_base_row_index": full_index,
                "row": row,
                "schedule": item,
                "parent_label": label,
            }
        )
    if seen_grid != expected_grid:
        missing = sorted(expected_grid.difference(seen_grid))
        extra = sorted(seen_grid.difference(expected_grid))
        raise ArtifactError(
            f"{label} is not exact K={samples_per_task}: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    return paired


def _deterministic_union_order(
    paired: Sequence[dict[str, Any]], *, seed: int, objective: str
) -> list[dict[str, Any]]:
    return sorted(
        paired,
        key=lambda item: (
            stable_sha256(
                {
                    "algorithm": "qwen-2776-union-order-v2",
                    "seed": int(seed),
                    "objective": objective,
                    "task_id": item["task_id"],
                    "sample_index": item["sample_index"],
                }
            ),
            str(item["task_id"]),
            int(item["sample_index"]),
        ),
    )


def _parent_bindings(parent: ParentArtifact) -> dict[str, Any]:
    return {
        "label": parent.paths.label,
        "dataset": parent.records["dataset"],
        "seal": parent.records["seal"],
        "schedule": parent.records["schedule"],
        "build_manifest": parent.records["manifest"],
        "task_count": len(
            {str(item.get("task_id") or "") for item in parent.schedule}
        ),
        "task_set_sha256": stable_sha256(
            sorted(
                {str(item.get("task_id") or "") for item in parent.schedule}
            )
        ),
    }


def _validate_parent_objectives(
    *,
    sequence_parent: ParentArtifact,
    cot_parent: ParentArtifact,
    expected_tasks: int,
) -> None:
    sequence_objective = sequence_parent.manifest.get("objective")
    sequence_counts = sequence_parent.manifest.get("counts")
    sequence_gold = sequence_parent.manifest.get("gold_replay")
    sampling = (
        sequence_objective.get("teacher_sampling")
        if isinstance(sequence_objective, Mapping)
        else None
    )
    cot_objective = cot_parent.manifest.get("objective")
    cot_counts = cot_parent.manifest.get("counts")
    if (
        not isinstance(sequence_objective, Mapping)
        or sequence_objective.get("name")
        != "monte_carlo_sequence_forward_kl_nll"
        or sequence_objective.get("objective_mode") != "sequence_only"
        or sequence_objective.get("all_k8_draws_required_and_emitted")
        is not True
        or sequence_objective.get("every_teacher_draw_emitted_exactly_once")
        is not True
        or sequence_objective.get("parseability_filtering") is not False
        or sequence_objective.get("correctness_filtering") is not False
        or sequence_objective.get("dense_full_vocabulary_kl") is not False
        or not isinstance(sampling, Mapping)
        or float(sampling.get("temperature", -1.0)) != 1.0
        or float(sampling.get("top_p", -1.0)) != 1.0
        or int(sampling.get("top_k", -1)) != 101
        or sampling.get("tempered") is not False
        or sampling.get("truncated") is not False
        or not isinstance(sequence_counts, Mapping)
        or int(sequence_counts.get("teacher_draw_rows", -1))
        != expected_tasks * SAMPLES_PER_TASK
        or int(sequence_counts.get("gold_replay_rows", -1)) != 0
        or int(sequence_counts.get("output_rows", -1))
        != expected_tasks * SAMPLES_PER_TASK
        or not isinstance(sequence_gold, Mapping)
        or float(sequence_gold.get("requested_final_fraction", -1.0)) != 0.0
        or int(sequence_gold.get("rows", -1)) != 0
        or not isinstance(cot_objective, Mapping)
        or cot_objective.get("name") != "qwen_cot_hard_sft"
        or cot_objective.get("ordinary_hard_sft") is not True
        or cot_objective.get("direct_prompt_mode")
        != DIRECT_PROMPT_MODE_QWEN_COT_V1
        or cot_objective.get("samples_per_task") != K_COT
        or cot_objective.get("correctness_filtering") is not False
        or cot_objective.get("parseability_filtering") is not False
        or cot_objective.get("resampling") is not False
        or not isinstance(cot_counts, Mapping)
        or int(cot_counts.get("tasks", -1)) != expected_tasks
        or int(cot_counts.get("rows", -1)) != expected_tasks * K_COT
        or int(cot_counts.get("rows_per_task", -1)) != K_COT
    ):
        raise ArtifactError(
            f"{sequence_parent.paths.label}/{cot_parent.paths.label} "
            "objective contract failed"
        )
    sequence_inputs = sequence_parent.manifest.get("inputs") or {}
    cot_inputs = cot_parent.manifest.get("inputs") or {}
    for field in (
        "compact_train",
        "compact_train_seal",
        "contract",
        "prompt_artifact",
        "prompt_manifest",
        "f2_prompt_contract",
        "teacher_audit",
        "student_tokenizer",
    ):
        if sequence_inputs.get(field) != cot_inputs.get(field):
            raise ArtifactError(
                f"{sequence_parent.paths.label}/{cot_parent.paths.label} "
                f"input {field} differs"
            )


def _write_union(
    *,
    paired: Sequence[dict[str, Any]],
    objective: str,
    schedule_schema: str,
    dataset_output: Path,
    seal_output: Path,
    schedule_output: Path,
    contract_path: Path,
    contract: DirectCompactContract,
    tokenizer: Any,
    seed: int,
    cot: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ordered = _deterministic_union_order(
        paired, seed=seed, objective=objective
    )
    rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    for position, item in enumerate(ordered):
        row = dict(item["row"])
        source_schedule = dict(item["schedule"])
        schedule = {
            **source_schedule,
            "schema": schedule_schema,
            "position": position,
            "base_row_index": int(item["full_base_row_index"]),
            "union_parent": item["parent_label"],
            "union_parent_position": int(source_schedule["position"]),
        }
        rows.append(row)
        schedules.append(schedule)
    atomic_write_jsonl(dataset_output, rows)
    atomic_write_jsonl(schedule_output, schedules)
    seal = exact_output_seal(
        output_path=dataset_output,
        contract_path=contract_path,
        contract=contract,
        rows=rows,
        tokenizer=tokenizer,
    )
    atomic_write_json(seal_output, seal)
    validate_join_seal(
        dataset_output, seal_output, contract_path, expected_role="fit"
    )
    expected_rows = EXPECTED_COT_ROWS if cot else EXPECTED_SEQUENCE_ROWS
    if len(rows) != expected_rows:
        raise AssertionError(
            f"{objective} union rows={len(rows)}, expected={expected_rows}"
        )
    return rows, schedules, seal


def build(args: argparse.Namespace) -> dict[str, Any]:
    derivation_path = args.derivation_manifest.expanduser().resolve()
    derivation_record = require_file_hash(
        derivation_path,
        args.expected_derivation_manifest_sha256,
        "2,776-task derivation manifest",
    )
    derivation = strict_json(derivation_path)
    counts = derivation.get("counts") or {}
    if (
        derivation.get("schema") != DERIVATION_SCHEMA
        or derivation.get("fit_scope")
        != "phase0_train_minus_heldout175"
        or int(counts.get("fit_tasks", -1)) != EXPECTED_FIT_TASKS
        or int(counts.get("legacy_parent_tasks", -1))
        != EXPECTED_LEGACY_TASKS
        or int(counts.get("supplement_tasks", -1))
        != EXPECTED_SUPPLEMENT_TASKS
        or int(counts.get("heldout_tasks", -1)) != EXPECTED_HOLDOUT_TASKS
        or int(derivation.get("heldout_intersection_count", -1)) != 0
        or (derivation.get("set_equations") or {}).get(
            "fit_equals_legacy_disjoint_union_supplement"
        )
        is not True
        or (derivation.get("invariants") or {}).get(
            "live_journal_modified"
        )
        is not False
    ):
        raise ArtifactError("2,776-task derivation contract failed")
    fit_order = [str(value) for value in derivation.get("ordered_task_ids") or []]
    supplement_order = [
        str(value)
        for value in derivation.get("supplement_ordered_task_ids") or []
    ]
    if (
        len(fit_order) != EXPECTED_FIT_TASKS
        or len(set(fit_order)) != len(fit_order)
        or stable_sha256(fit_order)
        != derivation.get("ordered_task_ids_sha256")
        or len(supplement_order) != EXPECTED_SUPPLEMENT_TASKS
        or stable_sha256(supplement_order)
        != derivation.get("supplement_ordered_task_ids_sha256")
    ):
        raise ArtifactError("derivation ordered task IDs failed")
    fit_ids = set(fit_order)
    supplement_ids = set(supplement_order)
    legacy_ids = fit_ids.difference(supplement_ids)
    if (
        len(legacy_ids) != EXPECTED_LEGACY_TASKS
        or legacy_ids.intersection(supplement_ids)
    ):
        raise ArtifactError("derivation task partition failed")

    fit_path = args.fit_compact_jsonl.expanduser().resolve()
    fit_seal_path = args.fit_compact_seal.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    expected_outputs = derivation.get("outputs") or {}
    for path, key in (
        (fit_path, "fit_compact"),
        (fit_seal_path, "fit_compact_seal"),
    ):
        actual = file_record(path)
        _record_matches(actual, expected_outputs.get(key) or {}, key)
    contract_record = file_record(contract_path)
    _record_matches(
        contract_record,
        (derivation.get("inputs") or {}).get("contract") or {},
        "compact contract",
    )
    contract = DirectCompactContract.load(contract_path)
    tokenizer, tokenizer_record = load_student_tokenizer(
        contract,
        args.student_tokenizer_json.expanduser().resolve(),
        args.expected_student_tokenizer_sha256,
    )
    validate_join_seal(
        fit_path, fit_seal_path, contract_path, expected_role="fit"
    )
    fit_rows = read_jsonl(fit_path)
    if len(fit_rows) != EXPECTED_FIT_TASKS:
        raise ArtifactError("fit compact artifact is not 2,776 rows")
    fit_by_task: dict[str, tuple[int, dict[str, Any]]] = {}
    observed_fit_order: list[str] = []
    for index, row in enumerate(fit_rows):
        task_id = _task_id(row, f"fit row {index}")
        if task_id in fit_by_task:
            raise ArtifactError(f"duplicate fit task {task_id}")
        contract.validate_row(row, f"fit row {index}")
        target_text(row, f"fit row {index}")
        fit_by_task[task_id] = (index, row)
        observed_fit_order.append(task_id)
    if observed_fit_order != fit_order:
        raise ArtifactError("fit compact order differs from derivation")

    legacy_sequence = _load_parent(
        _parent_paths(args, "legacy_sequence"),
        contract_path=contract_path,
        expected_schema=SEQUENCE_BUILD_SCHEMA,
    )
    supplement_sequence = _load_parent(
        _parent_paths(args, "supplement_sequence"),
        contract_path=contract_path,
        expected_schema=SEQUENCE_BUILD_SCHEMA,
    )
    legacy_cot = _load_parent(
        _parent_paths(args, "legacy_cot"),
        contract_path=contract_path,
        expected_schema=COT_BUILD_SCHEMA,
    )
    supplement_cot = _load_parent(
        _parent_paths(args, "supplement_cot"),
        contract_path=contract_path,
        expected_schema=COT_BUILD_SCHEMA,
    )
    _validate_parent_objectives(
        sequence_parent=legacy_sequence,
        cot_parent=legacy_cot,
        expected_tasks=EXPECTED_LEGACY_TASKS,
    )
    _validate_parent_objectives(
        sequence_parent=supplement_sequence,
        cot_parent=supplement_cot,
        expected_tasks=EXPECTED_SUPPLEMENT_TASKS,
    )

    sequence_pairs = [
        *validate_exact_grid(
            label="legacy_sequence",
            rows=legacy_sequence.rows,
            schedule=legacy_sequence.schedule,
            expected_task_ids=legacy_ids,
            samples_per_task=SAMPLES_PER_TASK,
            schedule_schema=SEQUENCE_SCHEDULE_SCHEMA,
            contract=contract,
            fit_by_task=fit_by_task,
            cot=False,
        ),
        *validate_exact_grid(
            label="supplement_sequence",
            rows=supplement_sequence.rows,
            schedule=supplement_sequence.schedule,
            expected_task_ids=supplement_ids,
            samples_per_task=SAMPLES_PER_TASK,
            schedule_schema=SEQUENCE_SCHEDULE_SCHEMA,
            contract=contract,
            fit_by_task=fit_by_task,
            cot=False,
        ),
    ]
    cot_pairs = [
        *validate_exact_grid(
            label="legacy_cot",
            rows=legacy_cot.rows,
            schedule=legacy_cot.schedule,
            expected_task_ids=legacy_ids,
            samples_per_task=K_COT,
            schedule_schema=COT_SCHEDULE_SCHEMA,
            contract=contract,
            fit_by_task=fit_by_task,
            cot=True,
        ),
        *validate_exact_grid(
            label="supplement_cot",
            rows=supplement_cot.rows,
            schedule=supplement_cot.schedule,
            expected_task_ids=supplement_ids,
            samples_per_task=K_COT,
            schedule_schema=COT_SCHEDULE_SCHEMA,
            contract=contract,
            fit_by_task=fit_by_task,
            cot=True,
        ),
    ]
    if len({item["candidate_id"] for item in sequence_pairs}) != len(
        sequence_pairs
    ):
        raise ArtifactError("sequence candidate IDs collide across parents")
    if len({item["candidate_id"] for item in cot_pairs}) != len(cot_pairs):
        raise ArtifactError("CoT candidate IDs collide across parents")
    sequence_candidate_by_slot = {
        (item["task_id"], int(item["sample_index"])): item["candidate_id"]
        for item in sequence_pairs
    }
    if any(
        sequence_candidate_by_slot.get(
            (item["task_id"], int(item["sample_index"]))
        )
        != item["candidate_id"]
        for item in cot_pairs
    ):
        raise ArtifactError(
            "CoT sample indices 0/1 do not bind the same teacher candidates "
            "as the sequence objective"
        )

    sequence_output = args.sequence_output_jsonl.expanduser().resolve()
    sequence_seal_output = args.sequence_output_seal.expanduser().resolve()
    sequence_schedule_output = (
        args.sequence_output_schedule.expanduser().resolve()
    )
    sequence_manifest_output = (
        args.sequence_output_manifest.expanduser().resolve()
    )
    cot_output = args.cot_output_jsonl.expanduser().resolve()
    cot_seal_output = args.cot_output_seal.expanduser().resolve()
    cot_schedule_output = args.cot_output_schedule.expanduser().resolve()
    cot_manifest_output = args.cot_output_manifest.expanduser().resolve()
    sequence_rows, sequence_schedule, _ = _write_union(
        paired=sequence_pairs,
        objective="sequence_forward_kl",
        schedule_schema=SEQUENCE_SCHEDULE_SCHEMA,
        dataset_output=sequence_output,
        seal_output=sequence_seal_output,
        schedule_output=sequence_schedule_output,
        contract_path=contract_path,
        contract=contract,
        tokenizer=tokenizer,
        seed=int(args.seed),
        cot=False,
    )
    cot_rows, cot_schedule, _ = _write_union(
        paired=cot_pairs,
        objective="cot_hard_sft",
        schedule_schema=COT_SCHEDULE_SCHEMA,
        dataset_output=cot_output,
        seal_output=cot_seal_output,
        schedule_output=cot_schedule_output,
        contract_path=contract_path,
        contract=contract,
        tokenizer=tokenizer,
        seed=int(args.seed),
        cot=True,
    )

    parent_sequence_bindings = [
        _parent_bindings(legacy_sequence),
        _parent_bindings(supplement_sequence),
    ]
    parent_cot_bindings = [
        _parent_bindings(legacy_cot),
        _parent_bindings(supplement_cot),
    ]
    common_union = {
        "schema": UNION_SCHEMA,
        "fit_scope": "phase0_train_minus_heldout175",
        "task_count": EXPECTED_FIT_TASKS,
        "ordered_task_ids": fit_order,
        "ordered_task_ids_sha256": stable_sha256(fit_order),
        "task_set_sha256": stable_sha256(sorted(fit_order)),
        "legacy_task_count": EXPECTED_LEGACY_TASKS,
        "supplement_task_count": EXPECTED_SUPPLEMENT_TASKS,
        "heldout_task_count": EXPECTED_HOLDOUT_TASKS,
        "heldout_task_set_sha256": derivation["heldout_task_set_sha256"],
        "heldout_intersection_count": 0,
        "derivation_manifest": derivation_record,
        "contract": contract_record,
        "student_tokenizer": tokenizer_record,
        "ordering": {
            "algorithm": "qwen-2776-union-order-v2",
            "seed": int(args.seed),
            "outcome_independent": True,
            "request_identity_fields": ["task_id", "sample_index"],
            "teacher_candidate_id_used": False,
        },
        "invariants": {
            "parent_journals_modified": False,
            "parent_rows_modified": False,
            "teacher_targets_filtered": False,
            "teacher_targets_resampled": False,
            "heldout_used_for_fit": False,
            "heldout_used_for_teacher_collection": False,
            "exact_task_partition": True,
        },
    }

    sequence_manifest = copy.deepcopy(legacy_sequence.manifest)
    sequence_manifest["schema"] = SEQUENCE_BUILD_SCHEMA
    sequence_manifest["seed"] = int(args.seed)
    sequence_manifest["gold_replay"] = {
        **dict(sequence_manifest.get("gold_replay") or {}),
        "requested_final_fraction": 0.0,
        "realized_final_fraction": 0.0,
        "rows": 0,
        "required_zero_for_sequence_only": True,
    }
    sequence_manifest["counts"] = {
        "teacher_draw_rows": len(sequence_rows),
        "gold_replay_rows": 0,
        "output_rows": len(sequence_rows),
        "unique_teacher_candidate_ids": len(sequence_rows),
        "tasks": EXPECTED_FIT_TASKS,
        "samples_per_task": SAMPLES_PER_TASK,
    }
    sequence_manifest["inputs"] = {
        "compact_train": file_record(fit_path),
        "compact_train_seal": file_record(fit_seal_path),
        "contract": contract_record,
        "student_tokenizer": tokenizer_record,
        "union_derivation": derivation_record,
        "parent_builds": parent_sequence_bindings,
    }
    sequence_manifest["outputs"] = {
        "dataset": file_record(sequence_output),
        "standard_direct_compact_seal": file_record(sequence_seal_output),
        "schedule": file_record(sequence_schedule_output),
    }
    sequence_manifest["schedule_sha256"] = stable_sha256(sequence_schedule)
    sequence_manifest["union_2776"] = {
        **common_union,
        "samples_per_task": SAMPLES_PER_TASK,
        "expected_grid_rows": EXPECTED_SEQUENCE_ROWS,
        "observed_grid_rows": len(sequence_rows),
        "grid_sha256": stable_sha256(
            sorted(
                (
                    item["task_id"],
                    int(item["sample_index"]),
                    item["candidate_id"],
                )
                for item in sequence_pairs
            )
        ),
        "parents": parent_sequence_bindings,
    }
    sequence_manifest["objective"] = {
        **dict(sequence_manifest.get("objective") or {}),
        "all_k8_draws_required_and_emitted": True,
        "every_teacher_draw_emitted_exactly_once": True,
        "parseability_filtering": False,
        "correctness_filtering": False,
        "dense_token_kl": False,
        "dense_full_vocabulary_kl": False,
    }
    legacy_sampling = legacy_sequence.manifest["objective"][
        "teacher_sampling"
    ]
    supplement_sampling = supplement_sequence.manifest["objective"][
        "teacher_sampling"
    ]
    legacy_unique = dict(
        legacy_sampling.get("unique_final_sequences_per_task") or {}
    )
    supplement_unique = dict(
        supplement_sampling.get("unique_final_sequences_per_task") or {}
    )
    if set(legacy_unique).intersection(supplement_unique):
        raise ArtifactError("parent sampling-diversity task maps overlap")
    union_unique = {**legacy_unique, **supplement_unique}
    if set(union_unique) != fit_ids:
        raise ArtifactError(
            "parent sampling-diversity maps do not cover fit2776"
        )
    sequence_manifest["objective"]["teacher_sampling"] = {
        **dict(legacy_sampling),
        "unique_final_sequences_per_task": union_unique,
        "provider_seed_honor_attested": bool(
            legacy_sampling.get("provider_seed_honor_attested")
            and supplement_sampling.get("provider_seed_honor_attested")
        ),
        "provider_seed_honor_assumed": False,
        "duplicate_teacher_draws_retained": True,
        "pathological_all_tasks_have_identical_k8_draws": False,
    }
    parent_target_gates = [
        dict(legacy_sequence.manifest.get("target_length_gate") or {}),
        dict(supplement_sequence.manifest.get("target_length_gate") or {}),
    ]
    sequence_manifest["target_length_gate"] = {
        **parent_target_gates[0],
        "passed": True,
        "teacher_rows_revalidated": len(sequence_rows),
        "audit_evidence_sha256": stable_sha256(
            [
                gate.get("audit_evidence_sha256")
                for gate in parent_target_gates
            ]
        ),
        "parent_target_length_gates_sha256": stable_sha256(
            parent_target_gates
        ),
        "gpu_launch_authorized_only_after_this_manifest": True,
    }
    atomic_write_json(sequence_manifest_output, sequence_manifest)

    empty_reasoning = [
        {
            key: item["schedule"].get(key)
            for key in (
                "task_id",
                "sample_index",
                "candidate_id",
                "reasoning_content_sha256",
                "raw_final_content_sha256",
                "cot_target_sha256",
                "target_length_evidence",
            )
        }
        for item in cot_pairs
        if item["schedule"].get("reasoning_content_empty") is True
    ]
    overflow = [
        {
            key: item["schedule"].get(key)
            for key in (
                "task_id",
                "sample_index",
                "candidate_id",
                "target_length_evidence",
            )
        }
        for item in cot_pairs
        if (
            (item["schedule"].get("target_length_evidence") or {}).get(
                "within_contract"
            )
            is not True
            or (item["schedule"].get("target_length_evidence") or {}).get(
                "within_total_contract"
            )
            is not True
        )
    ]
    min_reasoning = min(
        float(
            legacy_cot.manifest["coverage_gate"][
                "minimum_nonempty_reasoning_fraction"
            ]
        ),
        float(
            supplement_cot.manifest["coverage_gate"][
                "minimum_nonempty_reasoning_fraction"
            ]
        ),
    )
    cot_gate = cot_coverage_gate(
        task_count=EXPECTED_FIT_TASKS,
        schedule_rows=cot_schedule,
        empty_reasoning=empty_reasoning,
        overflow=overflow,
        min_nonempty_reasoning_fraction=min_reasoning,
        max_target_tokens=contract.max_target_tokens,
        max_total_tokens=contract.max_total_tokens,
    )
    if cot_gate.get("passed") is not True:
        raise ArtifactError("union CoT coverage gate failed")
    cot_manifest = copy.deepcopy(legacy_cot.manifest)
    cot_manifest["schema"] = COT_BUILD_SCHEMA
    cot_manifest["build_completed"] = True
    cot_manifest["coverage_gate"] = cot_gate
    cot_manifest["counts"] = {
        "tasks": EXPECTED_FIT_TASKS,
        "rows": len(cot_rows),
        "rows_per_task": K_COT,
        "unique_candidate_ids": len(cot_rows),
        "empty_reasoning_rows_retained": len(empty_reasoning),
    }
    cot_manifest["inputs"] = {
        "compact_train": file_record(fit_path),
        "compact_train_seal": file_record(fit_seal_path),
        "contract": contract_record,
        "student_tokenizer": tokenizer_record,
        "union_derivation": derivation_record,
        "parent_builds": parent_cot_bindings,
    }
    cot_manifest["outputs"] = {
        "dataset": file_record(cot_output),
        "standard_direct_compact_seal": file_record(cot_seal_output),
        "schedule": file_record(cot_schedule_output),
    }
    cot_manifest["schedule_sha256"] = stable_sha256(cot_schedule)
    cot_manifest["union_2776"] = {
        **common_union,
        "samples_per_task": K_COT,
        "expected_grid_rows": EXPECTED_COT_ROWS,
        "observed_grid_rows": len(cot_rows),
        "grid_sha256": stable_sha256(
            sorted(
                (
                    item["task_id"],
                    int(item["sample_index"]),
                    item["candidate_id"],
                )
                for item in cot_pairs
            )
        ),
        "parents": parent_cot_bindings,
    }
    atomic_write_json(cot_manifest_output, cot_manifest)
    return {
        "schema": UNION_SCHEMA,
        "sequence_manifest": file_record(sequence_manifest_output),
        "cot_manifest": file_record(cot_manifest_output),
        "counts": {
            "tasks": EXPECTED_FIT_TASKS,
            "sequence_rows": len(sequence_rows),
            "cot_rows": len(cot_rows),
        },
    }


def main() -> int:
    result = build(parse_args())
    print(
        "QWEN_2776_UNION "
        f"tasks={result['counts']['tasks']} "
        f"sequence_rows={result['counts']['sequence_rows']} "
        f"cot_rows={result['counts']['cot_rows']} "
        "heldout_intersection=0 parent_journals_modified=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
