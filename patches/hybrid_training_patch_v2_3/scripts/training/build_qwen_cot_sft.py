#!/usr/bin/env python3
"""Build a separate, mode-conditioned Qwen reasoning hard-SFT corpus.

This builder consumes the sealed Qwen sequence-only production journal
directly.  For every task it selects sample indices 0 and 1, independent of
correctness, compilation, parseability, confidence, logprobs, or reasoning
presence.  It never asks the provider for another draw.

The exact supervised target is::

    <think>\n + raw_reasoning_content + \n</think>\n + raw final content

The resulting rows carry ``direct_prompt_mode=qwen_cot_v1``.  They are an
ordinary hard-SFT corpus and are deliberately separate from the pure sampled
sequence forward-KL/NLL artifact.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
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
from scripts.training.build_qwen_sequence_kd import (  # noqa: E402
    compact_ids_sha256,
    exact_output_seal,
    load_student_tokenizer,
    require_file_hash,
    strict_json,
    target_text,
)
from scripts.training.direct_compact_qwen_decompiler import (  # noqa: E402
    DIRECT_PROMPT_MODE_QWEN_COT_V1,
    direct_prompt,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    AUDIT_SCHEMA,
    DEFAULT_MODEL,
    OBJECTIVE_MODE_SEQUENCE_ONLY,
    SAMPLES_PER_TASK,
    ArtifactError,
    JournalState,
    StudentTokenizerBinding,
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    load_verified_prompt_rows,
    read_jsonl,
    sha256_text,
    stable_sha256,
    validate_mc_teacher_sampling,
    validate_qwen38_sequence_sampling,
    validate_target_length_contract,
    target_length_evidence,
)


BUILD_SCHEMA = "direct-compact-qwen-cot-hard-sft-build-v1"
SCHEDULE_SCHEMA = "direct-compact-qwen-cot-hard-sft-schedule-v1"
K_COT = 2
SELECTED_SAMPLE_INDICES = tuple(range(K_COT))
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
QWEN3_THINK_OPEN_ID = 151667
QWEN3_THINK_CLOSE_ID = 151668
DEFAULT_MIN_NONEMPTY_REASONING_FRACTION = 0.90
PILOT_NONEMPTY_REASONING_ROWS = 128
PILOT_SELECTED_ROWS = 128
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TARGET_FIELDS = ("supervised_target", "dart_source", "source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--compact-train-jsonl", required=True, type=Path)
    parser.add_argument("--compact-train-seal", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--teacher-journal", required=True, type=Path)
    parser.add_argument("--expected-teacher-journal-sha256", required=True)
    parser.add_argument("--teacher-audit-json", required=True, type=Path)
    parser.add_argument("--expected-teacher-audit-sha256", required=True)
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-seal", required=True, type=Path)
    parser.add_argument("--schedule-output", required=True, type=Path)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument(
        "--min-nonempty-reasoning-fraction",
        type=float,
        default=DEFAULT_MIN_NONEMPTY_REASONING_FRACTION,
        help=(
            "Fail-closed corpus usability floor. Empty-reasoning rows remain "
            "selected and counted; they are never replaced or filtered."
        ),
    )
    return parser.parse_args()


def compose_cot_target(raw_reasoning: str, raw_final: str) -> str:
    """Compose the byte-exact mode target without trimming either provider field."""

    if not isinstance(raw_reasoning, str):
        raise ArtifactError("raw reasoning content is not a string")
    if not isinstance(raw_final, str) or not raw_final.strip():
        raise ArtifactError("raw final content is empty")
    return f"{THINK_OPEN}\n{raw_reasoning}\n{THINK_CLOSE}\n{raw_final}"


def selected_candidates(
    *,
    state: JournalState,
    task_ids: Sequence[str],
    prompt_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Select sealed slots 0 and 1 only; content never influences selection."""

    by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in state.candidates.values():
        task_id = str(candidate.get("task_id") or "")
        sample_index = int(candidate.get("sample_index", -1))
        slot = (task_id, sample_index)
        if slot in by_slot:
            raise ArtifactError(f"duplicate teacher candidate slot: {slot}")
        by_slot[slot] = candidate

    expected_all = {
        (task_id, sample_index)
        for task_id in task_ids
        for sample_index in range(SAMPLES_PER_TASK)
    }
    observed_all = set(by_slot)
    if observed_all != expected_all:
        missing = sorted(expected_all.difference(observed_all))
        extra = sorted(observed_all.difference(expected_all))
        raise ArtifactError(
            "teacher journal is not the complete sealed K=8 task grid: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    selected: list[dict[str, Any]] = []
    for task_id in task_ids:
        for sample_index in SELECTED_SAMPLE_INDICES:
            candidate = by_slot[(task_id, sample_index)]
            if candidate.get("completion_attested") is not True or str(
                candidate.get("prompt_sha256") or ""
            ) != prompt_hashes.get(task_id):
                raise ArtifactError(
                    "selected Qwen CoT slot lacks completion/prompt attestation: "
                    f"task={task_id} sample={sample_index}"
                )
            selected.append(candidate)
    return selected


def _replace_target_with_cot(row: Mapping[str, Any], target: str) -> dict[str, Any]:
    result = dict(row)
    present = [field for field in TARGET_FIELDS if field in result]
    result["dart_source"] = target
    for field in present:
        result[field] = target
    result["direct_prompt_mode"] = DIRECT_PROMPT_MODE_QWEN_COT_V1
    return result


def _token_ids(tokenizer: Any, text: str, *, special: bool) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=special)
    return [
        int(value) for value in (encoded.ids if hasattr(encoded, "ids") else encoded)
    ]


def _require_native_think_tokens(tokenizer: Any) -> dict[str, int]:
    open_ids = _token_ids(tokenizer, THINK_OPEN, special=False)
    close_ids = _token_ids(tokenizer, THINK_CLOSE, special=False)
    if open_ids != [QWEN3_THINK_OPEN_ID] or close_ids != [QWEN3_THINK_CLOSE_ID]:
        raise ArtifactError(
            "student tokenizer does not expose Qwen3 native think tokens: "
            f"open={open_ids} close={close_ids}"
        )
    return {
        "open_token_id": QWEN3_THINK_OPEN_ID,
        "close_token_id": QWEN3_THINK_CLOSE_ID,
    }


def _join_train_rows(
    *,
    train_rows: Sequence[dict[str, Any]],
    contract: DirectCompactContract,
    raw_prompts: Sequence[dict[str, Any]],
) -> dict[str, tuple[int, dict[str, Any]]]:
    by_compact_hash: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(train_rows):
        identity = f"compact-train-row-{index}"
        contract.validate_row(row, identity)
        target_text(row, identity)
        by_compact_hash.setdefault(compact_ids_sha256(row, identity), []).append(
            (index, row)
        )

    result: dict[str, tuple[int, dict[str, Any]]] = {}
    for raw_prompt in raw_prompts:
        task_id = str(raw_prompt.get("task_id") or "")
        expected_compact = str(raw_prompt.get("compact_ids_sha256") or "")
        if not task_id or not SHA256_RE.fullmatch(expected_compact):
            raise ArtifactError("verified prompt lacks its compact-row join key")
        candidates = list(by_compact_hash.get(expected_compact, []))
        if len(candidates) > 1:
            candidates = [
                item
                for item in candidates
                if str(item[1].get("task_id") or item[1].get("id") or "") == task_id
            ]
        if len(candidates) != 1:
            raise ArtifactError(
                f"prompt {task_id} does not join bijectively to compact train "
                f"(matches={len(candidates)})"
            )
        if task_id in result:
            raise ArtifactError(f"duplicate joined prompt task: {task_id}")
        result[task_id] = candidates[0]
    return result


def _base_manifest(
    *,
    min_nonempty_reasoning_fraction: float,
    inputs: Mapping[str, Any],
    coverage_gate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": BUILD_SCHEMA,
        "build_completed": False,
        "objective": {
            "name": "qwen_cot_hard_sft",
            "ordinary_hard_sft": True,
            "dense_token_kl": False,
            "sequence_forward_kl": False,
            "reasoning_logprobs_available": False,
            "pure_sequence_kl_artifact_modified": False,
            "direct_prompt_mode": DIRECT_PROMPT_MODE_QWEN_COT_V1,
            "target_template": (
                "<think>\\n + raw_reasoning_content + "
                "\\n</think>\\n + raw_final_content"
            ),
            "target_transform": "none_byte_exact_provider_strings",
            "samples_per_task": K_COT,
            "selected_sample_indices": list(SELECTED_SAMPLE_INDICES),
            "selection_depends_only_on": ["task_id", "sample_index"],
            "correctness_filtering": False,
            "compile_filtering": False,
            "parseability_filtering": False,
            "confidence_filtering": False,
            "logprob_filtering": False,
            "empty_reasoning_filtering": False,
            "resampling": False,
            "provider_calls": False,
        },
        "coverage_gate": {
            **dict(coverage_gate),
            "minimum_nonempty_reasoning_fraction": (min_nonempty_reasoning_fraction),
            "pilot_prior": {
                "selected_rows": PILOT_SELECTED_ROWS,
                "nonempty_reasoning_rows": PILOT_NONEMPTY_REASONING_ROWS,
                "nonempty_reasoning_fraction": (
                    PILOT_NONEMPTY_REASONING_ROWS / PILOT_SELECTED_ROWS
                ),
                "binding": (
                    "informational expectation only; production coverage is "
                    "independently measured from the sealed production journal"
                ),
            },
            "empty_rows_retained_if_gate_passes": True,
            "overflow_rows_retained_or_replaced": False,
            "overflow_policy": "abort_build_without_filtering_or_resampling",
        },
        "inputs": dict(inputs),
    }


def cot_coverage_gate(
    *,
    task_count: int,
    schedule_rows: Sequence[Mapping[str, Any]],
    empty_reasoning: Sequence[Mapping[str, Any]],
    overflow: Sequence[Mapping[str, Any]],
    min_nonempty_reasoning_fraction: float,
    max_target_tokens: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    """Compute the fail-closed gate without changing the selected row set."""

    selected_count = len(schedule_rows)
    expected_selected = int(task_count) * K_COT
    nonempty_count = selected_count - len(empty_reasoning)
    nonempty_fraction = nonempty_count / selected_count if selected_count else 0.0
    passed = bool(
        selected_count == expected_selected
        and len({str(row.get("task_id") or "") for row in schedule_rows})
        == int(task_count)
        and nonempty_fraction >= float(min_nonempty_reasoning_fraction)
        and not overflow
    )
    return {
        "passed": passed,
        "expected_tasks": int(task_count),
        "selected_tasks": len({str(row.get("task_id") or "") for row in schedule_rows}),
        "expected_rows": expected_selected,
        "selected_rows": selected_count,
        "exact_kcot_coverage_fraction": (
            selected_count / expected_selected if expected_selected else 0.0
        ),
        "nonempty_reasoning_rows": nonempty_count,
        "empty_reasoning_rows": len(empty_reasoning),
        "nonempty_reasoning_fraction": nonempty_fraction,
        "empty_reasoning_diagnostics": list(empty_reasoning),
        "max_target_tokens": int(max_target_tokens),
        "max_total_tokens": int(max_total_tokens),
        "overflow_rows": len(overflow),
        "overflow_diagnostics": list(overflow),
        "target_length_evidence_sha256": stable_sha256(
            [row.get("target_length_evidence") for row in schedule_rows]
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    min_reasoning = float(args.min_nonempty_reasoning_fraction)
    if not 0.0 < min_reasoning <= 1.0:
        raise ArtifactError("--min-nonempty-reasoning-fraction must be in (0, 1]")

    train_path = args.compact_train_jsonl.expanduser().resolve()
    train_seal_path = args.compact_train_seal.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    prompt_path = args.prompt_jsonl.expanduser().resolve()
    journal_path = args.teacher_journal.expanduser().resolve()
    audit_path = args.teacher_audit_json.expanduser().resolve()
    tokenizer_path = args.student_tokenizer_json.expanduser().resolve()
    for path in (
        train_path,
        train_seal_path,
        contract_path,
        prompt_path,
        journal_path,
        audit_path,
        tokenizer_path,
    ):
        if not path.is_file():
            raise ArtifactError(f"required artifact does not exist: {path}")

    contract = DirectCompactContract.load(contract_path)
    tokenizer, tokenizer_record = load_student_tokenizer(
        contract,
        tokenizer_path,
        args.expected_student_tokenizer_sha256,
    )
    think_tokens = _require_native_think_tokens(tokenizer)
    source_seal = validate_join_seal(
        train_path, train_seal_path, contract_path, expected_role="fit"
    )
    train_rows = read_jsonl(train_path)
    if len(train_rows) != int(source_seal["rows"]):
        raise ArtifactError("compact train row count changed after seal validation")

    prompts, prompt_record = load_verified_prompt_rows(
        prompt_path, expected_sha256=args.expected_prompt_sha256
    )
    raw_prompts = read_jsonl(prompt_path)
    if len(prompts) != len(raw_prompts):
        raise AssertionError("verified prompt loader changed row count")
    compact_by_task = _join_train_rows(
        train_rows=train_rows,
        contract=contract,
        raw_prompts=raw_prompts,
    )
    if set(compact_by_task) != {prompt.task_id for prompt in prompts}:
        raise ArtifactError("compact/prompt task sets differ")

    journal_record = require_file_hash(
        journal_path,
        args.expected_teacher_journal_sha256,
        "teacher journal",
    )
    state = JournalState.load(journal_path)
    if state.header is None:
        raise ArtifactError("teacher journal has no run header")
    if state.rejections:
        first = next(iter(state.rejections.values()))
        raise ArtifactError(
            "teacher journal contains a consumed rejected draw; CoT selection "
            "cannot resample it: "
            f"task={first.get('task_id')} sample={first.get('sample_index')}"
        )
    payload = state.header.get("payload") or {}
    task_ids = [str(value) for value in payload.get("task_ids") or []]
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or int(payload.get("samples_per_task", -1)) != SAMPLES_PER_TASK
        or set(task_ids) != set(compact_by_task)
    ):
        raise ArtifactError("teacher header does not seal the complete train task set")
    if (
        payload.get("requested_model") != DEFAULT_MODEL
        or payload.get("returned_model_must_equal_requested") is not True
        or str(payload.get("objective_mode") or "") != OBJECTIVE_MODE_SEQUENCE_ONLY
    ):
        raise ArtifactError(
            "CoT SFT requires the exact Qwen3.8 sequence-only production journal"
        )
    generation = payload.get("generation_parameters")
    if not isinstance(generation, Mapping):
        raise ArtifactError("teacher header lacks generation parameters")
    validate_mc_teacher_sampling(generation)
    validate_qwen38_sequence_sampling(DEFAULT_MODEL, generation)

    header_target_contract = payload.get("target_length_contract")
    if not isinstance(header_target_contract, Mapping):
        raise ArtifactError("teacher header lacks the target-length contract")
    binding = StudentTokenizerBinding(
        tokenizer,
        eos_token_id=int(header_target_contract.get("student_eos_token_id", -1)),
        tokenizer_record=tokenizer_record,
    )
    validate_target_length_contract(
        header_target_contract,
        binding=binding,
        objective_mode=OBJECTIVE_MODE_SEQUENCE_ONLY,
    )
    if (header_target_contract.get("trainer_contract") or {}).get(
        "sha256"
    ) != sha256_file(contract_path) or int(
        header_target_contract.get("max_target_tokens", -1)
    ) != contract.max_target_tokens:
        raise ArtifactError(
            "teacher journal target capacity differs from the supplied contract"
        )

    prompt_bindings = payload.get("prompt_bindings")
    if not isinstance(prompt_bindings, list):
        raise ArtifactError("teacher header lacks prompt bindings")
    prompt_hashes: dict[str, str] = {}
    for item in prompt_bindings:
        if not isinstance(item, Mapping):
            raise ArtifactError("teacher prompt binding is not an object")
        task_id = str(item.get("task_id") or "")
        digest = str(item.get("request_messages_sha256") or "")
        if (
            task_id not in task_ids
            or task_id in prompt_hashes
            or not SHA256_RE.fullmatch(digest)
        ):
            raise ArtifactError("teacher header contains an invalid prompt binding")
        prompt_hashes[task_id] = digest
    if set(prompt_hashes) != set(task_ids):
        raise ArtifactError("teacher prompt bindings do not cover every task")
    if (payload.get("prompt_artifact") or {}).get("sha256") != prompt_record["sha256"]:
        raise ArtifactError("teacher journal was collected from another prompt file")

    audit_record = require_file_hash(
        audit_path,
        args.expected_teacher_audit_sha256,
        "teacher audit",
    )
    audit = strict_json(audit_path)
    chain_head = file_record(Path(str(journal_path) + ".chain-head.json"))
    coverage = audit.get("coverage") or {}
    expected_candidate_count = len(task_ids) * SAMPLES_PER_TASK
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("objective_mode") != OBJECTIVE_MODE_SEQUENCE_ONLY
        or (audit.get("journal") or {}).get("sha256") != journal_record["sha256"]
        or audit.get("journal_chain_head") != chain_head
        or audit.get("production_ready") is not True
        or (audit.get("production_readiness") or {}).get("mc_sequence_forward_kl_nll")
        is not True
        or int(audit.get("expected_tasks", -1)) != len(task_ids)
        or int(audit.get("samples_per_task", -1)) != SAMPLES_PER_TASK
        or audit.get("incomplete_task_sample_counts") not in ({}, None)
        or audit.get("invalid_task_sample_indices") not in ({}, None)
        or audit.get("all_candidates_independently_verified") is not True
        or int(coverage.get("candidates", -1)) != expected_candidate_count
        or int(coverage.get("sequence_candidates", -1)) != expected_candidate_count
        or int(coverage.get("completion_attested_candidates", -1))
        != expected_candidate_count
        or (audit.get("target_length_gate") or {}).get("passed") is not True
        or audit.get("student_tokenizer") != tokenizer_record
        or audit.get("prompt_artifact") != payload.get("prompt_artifact")
        or audit.get("prompt_manifest") != payload.get("prompt_manifest")
        or audit.get("f2_prompt_contract") != payload.get("f2_prompt_contract")
    ):
        raise ArtifactError(
            "teacher audit does not attest a complete production K=8 journal"
        )

    candidates = selected_candidates(
        state=state,
        task_ids=task_ids,
        prompt_hashes=prompt_hashes,
    )
    output_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    empty_reasoning: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        task_id = str(candidate["task_id"])
        sample_index = int(candidate["sample_index"])
        response = candidate.get("response") or {}
        raw_reasoning = response.get("raw_reasoning_content")
        raw_final = response.get("raw_content")
        if not isinstance(raw_reasoning, str) or not isinstance(raw_final, str):
            raise ArtifactError(
                f"selected candidate has non-string provider fields: {task_id}"
            )
        if sha256_text(raw_reasoning) != str(
            response.get("raw_reasoning_content_sha256") or ""
        ) or sha256_text(raw_final) != str(response.get("raw_content_sha256") or ""):
            raise ArtifactError(
                f"selected candidate provider-field hash mismatch: {task_id}"
            )
        target = compose_cot_target(raw_reasoning, raw_final)
        length = target_length_evidence(
            target,
            binding=binding,
            max_target_tokens=contract.max_target_tokens,
        )
        base_index, base_row = compact_by_task[task_id]
        output_row = _replace_target_with_cot(base_row, target)
        contract.validate_row(output_row, f"cot-output-row-{position}")
        prompt_ids = _token_ids(
            tokenizer,
            direct_prompt(
                output_row,
                target_function=contract.target_function,
                target_language=contract.target_language,
            ),
            special=True,
        )
        total_tokens = (
            len(prompt_ids)
            + len(output_row["compact_input_ids"])
            + int(length["eos_inclusive_target_token_count"])
        )
        length = {
            **length,
            "prompt_token_count": len(prompt_ids),
            "compact_source_token_count": len(output_row["compact_input_ids"]),
            "prompt_source_target_token_count": total_tokens,
            "max_total_tokens": contract.max_total_tokens,
            "within_total_contract": total_tokens <= contract.max_total_tokens,
        }
        diagnostic = {
            "task_id": task_id,
            "sample_index": sample_index,
            "candidate_id": str(candidate["candidate_id"]),
            "reasoning_content_sha256": sha256_text(raw_reasoning),
            "raw_final_content_sha256": sha256_text(raw_final),
            "cot_target_sha256": sha256_text(target),
            "target_length_evidence": length,
        }
        if not raw_reasoning:
            empty_reasoning.append(diagnostic)
        if (
            length["within_contract"] is not True
            or length["within_total_contract"] is not True
        ):
            overflow.append(diagnostic)
        output_rows.append(output_row)
        schedule_rows.append(
            {
                "schema": SCHEDULE_SCHEMA,
                "position": position,
                "task_id": task_id,
                "sample_index": sample_index,
                "candidate_id": str(candidate["candidate_id"]),
                "base_row_index": base_index,
                "compact_ids_sha256": compact_ids_sha256(base_row, task_id),
                "reasoning_content_sha256": sha256_text(raw_reasoning),
                "raw_final_content_sha256": sha256_text(raw_final),
                "cot_target_sha256": sha256_text(target),
                "reasoning_content_empty": not bool(raw_reasoning),
                "target_length_evidence": length,
                "selection_rule": "sealed_sample_index_in_[0,1]",
                "selected_without_outcome_inspection": True,
            }
        )

    selected_count = len(candidates)
    expected_selected = len(task_ids) * K_COT
    coverage_gate = cot_coverage_gate(
        task_count=len(task_ids),
        schedule_rows=schedule_rows,
        empty_reasoning=empty_reasoning,
        overflow=overflow,
        min_nonempty_reasoning_fraction=min_reasoning,
        max_target_tokens=contract.max_target_tokens,
        max_total_tokens=contract.max_total_tokens,
    )
    nonempty_fraction = float(coverage_gate["nonempty_reasoning_fraction"])
    gate_passed = bool(coverage_gate["passed"])
    inputs = {
        "compact_train": file_record(train_path),
        "compact_train_seal": file_record(train_seal_path),
        "contract": file_record(contract_path),
        "prompt_artifact": prompt_record,
        "prompt_manifest": payload.get("prompt_manifest"),
        "f2_prompt_contract": payload.get("f2_prompt_contract"),
        "teacher_journal": journal_record,
        "teacher_journal_chain_head": chain_head,
        "teacher_audit": audit_record,
        "student_tokenizer": tokenizer_record,
        "native_think_tokens": think_tokens,
    }
    manifest = _base_manifest(
        min_nonempty_reasoning_fraction=min_reasoning,
        inputs=inputs,
        coverage_gate=coverage_gate,
    )
    manifest_path = args.build_manifest.expanduser().resolve()
    if not gate_passed:
        manifest["failure"] = {
            "reason": (
                "CoT coverage/length gate failed; no dataset, seal, or schedule "
                "was published"
            ),
            "no_filtering_or_resampling_performed": True,
        }
        atomic_write_json(manifest_path, manifest)
        raise ArtifactError(
            "Qwen CoT fail-closed gate failed: "
            f"selected={selected_count}/{expected_selected} "
            f"nonempty_reasoning={nonempty_fraction:.6f} "
            f"required={min_reasoning:.6f} overflow={len(overflow)}; "
            f"diagnostics={manifest_path}"
        )

    output_path = args.output_jsonl.expanduser().resolve()
    output_seal_path = args.output_seal.expanduser().resolve()
    schedule_path = args.schedule_output.expanduser().resolve()
    atomic_write_jsonl(output_path, output_rows)
    atomic_write_jsonl(schedule_path, schedule_rows)
    output_seal = exact_output_seal(
        output_path=output_path,
        contract_path=contract_path,
        contract=contract,
        rows=output_rows,
        tokenizer=tokenizer,
    )
    atomic_write_json(output_seal_path, output_seal)
    validate_join_seal(
        output_path, output_seal_path, contract_path, expected_role="fit"
    )

    manifest["build_completed"] = True
    manifest["counts"] = {
        "tasks": len(task_ids),
        "rows": len(output_rows),
        "rows_per_task": K_COT,
        "unique_candidate_ids": len({row["candidate_id"] for row in schedule_rows}),
        "empty_reasoning_rows_retained": len(empty_reasoning),
    }
    manifest["outputs"] = {
        "dataset": file_record(output_path),
        "standard_direct_compact_seal": file_record(output_seal_path),
        "schedule": file_record(schedule_path),
    }
    manifest["schedule_sha256"] = stable_sha256(schedule_rows)
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build(args)
    print(
        "QWEN_COT_SFT_BUILD "
        f"tasks={manifest['counts']['tasks']} "
        f"rows={manifest['counts']['rows']} "
        "kcot=2 "
        f"nonempty_reasoning="
        f"{manifest['coverage_gate']['nonempty_reasoning_fraction']:.6f} "
        "dense_token_kl=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
