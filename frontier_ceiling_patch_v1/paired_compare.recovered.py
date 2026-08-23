#!/usr/bin/env python3
"""Fail-closed paired comparison of two completed frontier pass@K runs.

The comparison direction is always arm B minus arm A.  Pairing is by the
*ordered* task result sequence, not by a permissive task-ID join: a reordered
or different cohort is rejected.  Directory inputs are verified against the
runner's final manifest; a standalone file input must be a complete
``summary.json`` object.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from frontier_core import (
        ResponseContractError,
        classify_terminal_provider_response,
        sha256_text,
    )
except ImportError:
    from .frontier_core import (
        ResponseContractError,
        classify_terminal_provider_response,
        sha256_text,
    )

RUN_SCHEMA = "audited-frontier-passk-v2"
COMPARISON_SCHEMA = "audited-frontier-paired-comparison-v2"
SLOT_POLICY_SCHEMA = "fixed-cap-exact-response-slot-v1"
DEFAULT_BOOTSTRAP_REPLICATES = 50_000
DEFAULT_BOOTSTRAP_SEED = 20260725
REQUIRED_RUN_FILES = (
    "provenance.json",
    "tasks.jsonl",
    "prompts.jsonl",
    "attempts.jsonl",
    "outcomes.jsonl",
    "summary.json",
)


class ComparisonError(ValueError):
    """Raised when inputs cannot support an auditable paired comparison."""


@dataclass(frozen=True)
class CompletedRun:
    source: Path
    source_kind: str
    summary_path: Path
    summary_sha256: str
    summary: dict[str, Any]
    provenance: dict[str, Any]
    task_ids: tuple[str, ...]
    passed: tuple[bool, ...]
    k: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ComparisonError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must contain one JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ComparisonError(
                    f"{label} line {line_number} is not an object"
                )
            rows.append(value)
    except ComparisonError:
        raise
    except Exception as exc:
        raise ComparisonError(f"cannot parse {label} {path}: {exc}") from exc
    return rows


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{label} must be an object")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ComparisonError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ComparisonError(f"{label} must be an integer >= {minimum}")
    return value


def _require_digest(value: Any, label: str) -> str:
    digest = str(value or "")
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ComparisonError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _verify_file_record(
    path: Path,
    record: Any,
    label: str,
) -> None:
    record = _require_mapping(record, label)
    if not path.is_file():
        raise ComparisonError(f"{label} is missing: {path}")
    expected_sha = _require_digest(record.get("sha256"), f"{label}.sha256")
    expected_bytes = _require_int(record.get("bytes"), f"{label}.bytes")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ComparisonError(
            f"{label} byte-size mismatch: expected {expected_bytes}, "
            f"got {actual_bytes}"
        )
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ComparisonError(
            f"{label} hash mismatch: expected {expected_sha}, got {actual_sha}"
        )


def _resolve_and_verify_source(source: Path) -> tuple[Path, str, Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        raise ComparisonError(
            "v2 exact-response comparison requires a completed run directory "
            "so attempts and outcomes can be rederived"
        )
    if not source.is_dir():
        raise ComparisonError(f"run input does not exist: {source}")

    manifest_path = source / "manifest.json"
    provenance_path = source / "provenance.json"
    summary_path = source / "summary.json"
    if not manifest_path.is_file():
        raise ComparisonError(
            f"completed run directory has no final manifest: {manifest_path}"
        )
    manifest = _load_json(manifest_path, "run manifest")
    if manifest.get("schema") != RUN_SCHEMA:
        raise ComparisonError(
            f"unexpected run manifest schema: {manifest.get('schema')!r}"
        )
    files = _require_mapping(manifest.get("files"), "run manifest files")
    for filename in REQUIRED_RUN_FILES:
        if filename not in files:
            raise ComparisonError(
                f"run manifest does not attest required file {filename!r}"
            )
        _verify_file_record(
            source / filename,
            files[filename],
            f"manifest file {filename}",
        )

    provenance = _load_json(provenance_path, "run provenance")
    if provenance.get("schema") != RUN_SCHEMA:
        raise ComparisonError(
            f"unexpected run provenance schema: {provenance.get('schema')!r}"
        )
    if provenance.get("status") != "complete":
        raise ComparisonError(
            f"run provenance is not complete: {provenance.get('status')!r}"
        )
    expected_summary_sha = _require_digest(
        provenance.get("summary_sha256"),
        "run provenance summary_sha256",
    )
    actual_summary_sha = _sha256_file(summary_path)
    if actual_summary_sha != expected_summary_sha:
        raise ComparisonError(
            "run provenance summary hash disagrees with summary.json: "
            f"{expected_summary_sha} != {actual_summary_sha}"
        )
    return source, "verified_run_directory", summary_path


def _validate_rate_block(
    summary: Mapping[str, Any],
    field: str,
    expected_flags: Sequence[bool],
) -> None:
    block = _require_mapping(summary.get(field), field)
    successes = _require_int(block.get("successes"), f"{field}.successes")
    total = _require_int(block.get("total"), f"{field}.total", minimum=1)
    expected_successes = sum(expected_flags)
    expected_total = len(expected_flags)
    if successes != expected_successes or total != expected_total:
        raise ComparisonError(
            f"{field} aggregate disagrees with task_results: "
            f"({successes}/{total}) != "
            f"({expected_successes}/{expected_total})"
        )
    rate = block.get("rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ComparisonError(f"{field}.rate must be numeric")
    expected_rate = expected_successes / expected_total
    if not math.isfinite(float(rate)) or not math.isclose(
        float(rate),
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ComparisonError(
            f"{field}.rate disagrees with task_results: "
            f"{rate!r} != {expected_rate}"
        )


def load_completed_run(source: str | Path) -> CompletedRun:
    """Load and validate one complete runner result.

    Passing a directory invokes full final-manifest and provenance verification.
    Passing a file validates the self-contained summary but cannot attest
    sibling artifacts.
    """

    resolved_source, source_kind, summary_path = _resolve_and_verify_source(
        Path(source)
    )
    summary = _load_json(summary_path, "run summary")
    if summary.get("schema") != RUN_SCHEMA:
        raise ComparisonError(
            f"unexpected run summary schema: {summary.get('schema')!r}"
        )
    if summary.get("status") != "complete":
        raise ComparisonError(
            f"run summary is not complete: {summary.get('status')!r}"
        )
    provenance = _load_json(
        resolved_source / "provenance.json",
        "run provenance",
    )
    config = _require_mapping(provenance.get("config"), "provenance.config")
    if config.get("schema") != RUN_SCHEMA:
        raise ComparisonError("provenance config uses a legacy schema")
    config_sha = _require_digest(
        provenance.get("config_sha256"),
        "provenance.config_sha256",
    )
    if _stable_sha256(config) != config_sha:
        raise ComparisonError("provenance config hash is inconsistent")
    fixed_cap = _require_int(
        summary.get("fixed_max_output_tokens"),
        "summary.fixed_max_output_tokens",
        minimum=1,
    )
    slot_policy = _require_mapping(
        summary.get("slot_policy"),
        "summary.slot_policy",
    )
    if slot_policy.get("schema") != SLOT_POLICY_SCHEMA:
        raise ComparisonError(
            f"unsupported slot policy: {slot_policy.get('schema')!r}"
        )
    slot_policy_sha = _require_digest(
        summary.get("slot_policy_sha256"),
        "summary.slot_policy_sha256",
    )
    if _stable_sha256(slot_policy) != slot_policy_sha:
        raise ComparisonError("summary slot-policy hash is inconsistent")
    if slot_policy.get("fixed_max_output_tokens") != fixed_cap:
        raise ComparisonError("summary fixed cap disagrees with slot policy")
    if config.get("slot_policy_sha256") != slot_policy_sha:
        raise ComparisonError("provenance config slot-policy hash mismatch")
    if config.get("slot_policy") != slot_policy:
        raise ComparisonError("provenance config slot policy mismatch")
    if config.get("max_output_tokens") != fixed_cap:
        raise ComparisonError("provenance config fixed cap mismatch")
    for field in (
        "every_returned_response_consumes_one_slot",
        "retry_only_when_no_provider_response",
        "finish_reason_length_consumes_slot",
        "no_candidate_resampling",
        "duplicate_response_id_is_fatal",
    ):
        if slot_policy.get(field) is not True:
            raise ComparisonError(f"slot policy does not enforce {field}")

    k = _require_int(summary.get("k"), "summary.k", minimum=1)
    task_count = _require_int(summary.get("tasks"), "summary.tasks", minimum=1)
    results = summary.get("task_results")
    if not isinstance(results, list):
        raise ComparisonError("summary.task_results must be an array")
    if len(results) != task_count:
        raise ComparisonError(
            f"summary.task_results length {len(results)} != tasks {task_count}"
        )

    task_ids: list[str] = []
    passed: list[bool] = []
    compiled: list[bool] = []
    for task_index, raw_result in enumerate(results):
        result = _require_mapping(
            raw_result,
            f"summary.task_results[{task_index}]",
        )
        task_id = result.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ComparisonError(
                f"summary.task_results[{task_index}].task_id "
                "must be a nonempty string"
            )
        terminal_responses = _require_int(
            result.get("terminal_responses"),
            f"task {task_id} terminal_responses",
        )
        if terminal_responses != k:
            raise ComparisonError(
                f"task {task_id} has {terminal_responses} terminal responses, "
                f"expected K={k}"
            )
        outcomes = result.get("candidate_outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != k:
            actual = len(outcomes) if isinstance(outcomes, list) else "non-array"
            raise ComparisonError(
                f"task {task_id} has {actual} candidate outcomes, expected K={k}"
            )

        outcome_passed: list[bool] = []
        outcome_compiled: list[bool] = []
        sample_indices: list[int] = []
        for outcome_index, raw_outcome in enumerate(outcomes):
            outcome = _require_mapping(
                raw_outcome,
                f"task {task_id} candidate_outcomes[{outcome_index}]",
            )
            if outcome.get("task_id") != task_id:
                raise ComparisonError(
                    f"task {task_id} candidate outcome {outcome_index} "
                    "has a different task_id"
                )
            sample_indices.append(
                _require_int(
                    outcome.get("sample_index"),
                    f"task {task_id} outcome sample_index",
                )
            )
            _require_bool(
                outcome.get("candidate_valid"),
                f"task {task_id} outcome candidate_valid",
            )
            if not str(outcome.get("response_id") or ""):
                raise ComparisonError(
                    f"task {task_id} candidate outcome lacks response_id"
                )
            if not str(outcome.get("terminal_reason") or ""):
                raise ComparisonError(
                    f"task {task_id} candidate outcome lacks terminal_reason"
                )
            outcome_passed.append(
                _require_bool(
                    outcome.get("passed"),
                    f"task {task_id} outcome passed",
                )
            )
            outcome_compiled.append(
                _require_bool(
                    outcome.get("compiled"),
                    f"task {task_id} outcome compiled",
                )
            )
        if sorted(sample_indices) != list(range(k)):
            raise ComparisonError(
                f"task {task_id} candidate sample indices are not exactly 0..K-1"
            )

        task_passed = _require_bool(
            result.get("passed"),
            f"task {task_id} passed",
        )
        task_compiled = _require_bool(
            result.get("compiled"),
            f"task {task_id} compiled",
        )
        if task_passed != any(outcome_passed):
            raise ComparisonError(
                f"task {task_id} pass@K flag disagrees with candidate outcomes"
            )
        if task_compiled != any(outcome_compiled):
            raise ComparisonError(
                f"task {task_id} compile@K flag disagrees with candidate outcomes"
            )
        task_ids.append(task_id)
        passed.append(task_passed)
        compiled.append(task_compiled)

    if len(set(task_ids)) != len(task_ids):
        raise ComparisonError("summary.task_results contains duplicate task IDs")
    expected_task_set_sha = _stable_sha256(task_ids)
    recorded_task_set_sha = _require_digest(
        summary.get("task_set_sha256"),
        "summary.task_set_sha256",
    )
    if recorded_task_set_sha != expected_task_set_sha:
        raise ComparisonError(
            "summary.task_set_sha256 disagrees with ordered task_results: "
            f"{recorded_task_set_sha} != {expected_task_set_sha}"
        )

    terminal_responses = _require_int(
        summary.get("terminal_responses"),
        "summary.terminal_responses",
    )
    if terminal_responses != task_count * k:
        raise ComparisonError(
            f"summary.terminal_responses {terminal_responses} != tasks*K "
            f"{task_count * k}"
        )
    if (
        summary.get("all_tasks_have_exactly_k_terminal_provider_responses")
        is not True
    ):
        raise ComparisonError(
            "summary does not attest exactly K terminal responses for every task"
        )
    if summary.get("every_terminal_provider_response_has_exactly_one_outcome") is not True:
        raise ComparisonError("summary does not attest one outcome per response")
    if summary.get("returned_responses_resampled") is not False:
        raise ComparisonError("summary permits returned-response resampling")
    if summary.get("transport_failures_only_retried") is not True:
        raise ComparisonError("summary does not restrict retries to transport failures")
    if summary.get("early_stopping_used") is not False:
        raise ComparisonError("summary does not attest that early stopping was unused")
    if summary.get("prompt_truncation_used") is not False:
        raise ComparisonError(
            "summary does not attest that prompt truncation was unused"
        )

    task_id_set = set(task_ids)
    prompt_sha_by_task: dict[str, str] = {}
    for row in _load_jsonl(resolved_source / "prompts.jsonl", "prompt journal"):
        if row.get("schema") != RUN_SCHEMA:
            raise ComparisonError("prompt journal contains a legacy schema")
        task_id = str(row.get("task_id") or "")
        prompt_sha = _require_digest(
            row.get("prompt_sha256"),
            f"prompt {task_id} sha256",
        )
        if task_id not in task_id_set or task_id in prompt_sha_by_task:
            raise ComparisonError("prompt journal task identity is invalid")
        prompt_sha_by_task[task_id] = prompt_sha
    if set(prompt_sha_by_task) != task_id_set:
        raise ComparisonError("prompt journal does not cover the exact task set")

    grouped_attempts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    transport_retries = 0
    recorded_budget_charge = 0
    derived_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for row in _load_jsonl(resolved_source / "attempts.jsonl", "attempt journal"):
        if row.get("schema") != RUN_SCHEMA:
            raise ComparisonError("attempt journal contains a legacy schema")
        task_id = str(row.get("task_id") or "")
        sample_index = _require_int(
            row.get("sample_index"),
            "attempt sample_index",
        )
        _require_int(row.get("attempt_index"), "attempt attempt_index")
        if task_id not in task_id_set or sample_index >= k:
            raise ComparisonError("attempt journal contains a foreign slot")
        if row.get("requested_max_tokens") != fixed_cap:
            raise ComparisonError("attempt fixed completion cap mismatch")
        if row.get("slot_policy_sha256") != slot_policy_sha:
            raise ComparisonError("attempt slot-policy hash mismatch")
        if row.get("config_sha256") != config_sha:
            raise ComparisonError("attempt config hash mismatch")
        if row.get("prompt_sha256") != prompt_sha_by_task[task_id]:
            raise ComparisonError("attempt prompt hash mismatch")
        if not str(row.get("attempt_id") or ""):
            raise ComparisonError("attempt journal lacks an attempt id")
        charge = _require_int(
            row.get("budget_charge_tokens"),
            "attempt budget_charge_tokens",
        )
        recorded_budget_charge += charge
        grouped_attempts.setdefault((task_id, sample_index), []).append(row)
        if row.get("response_received") is False:
            transport_retries += 1

    terminal_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    response_ids: set[str] = set()
    max_prompt_tokens = _require_int(
        slot_policy.get("max_prompt_tokens"),
        "slot_policy.max_prompt_tokens",
        minimum=1,
    )
    worst_case_reservation = max_prompt_tokens + fixed_cap
    for slot, rows in grouped_attempts.items():
        ordered = sorted(rows, key=lambda row: int(row["attempt_index"]))
        indices = [int(row["attempt_index"]) for row in ordered]
        if indices != list(range(len(ordered))):
            raise ComparisonError(f"attempt indices are not contiguous for {slot}")
        saw_terminal = False
        for row in ordered:
            response_received = _require_bool(
                row.get("response_received"),
                "attempt response_received",
            )
            slot_terminal = _require_bool(
                row.get("slot_terminal"),
                "attempt slot_terminal",
            )
            if saw_terminal:
                raise ComparisonError(f"post-terminal attempt found for {slot}")
            if response_received:
                if not slot_terminal:
                    raise ComparisonError("returned response is not terminal")
                raw = _require_mapping(
                    row.get("response"),
                    "terminal raw response",
                )
                try:
                    classified = classify_terminal_provider_response(
                        dict(raw),
                        expected_model=str(summary.get("requested_model") or ""),
                        max_prompt_tokens=max_prompt_tokens,
                        requested_max_tokens=fixed_cap,
                    )
                except ResponseContractError as exc:
                    raise ComparisonError(
                        f"terminal raw response violates contract: {exc}"
                    ) from exc
                expected_fields = {
                    "response_id": classified.response_id,
                    "resolved_model": classified.response_model,
                    "response_created": classified.response_created,
                    "finish_reason": classified.finish_reason,
                    "candidate_valid": classified.candidate_valid,
                    "terminal_reason": classified.terminal_reason,
                    "content": classified.content,
                    "reasoning_content": classified.reasoning_content,
                    "code": classified.code,
                    "code_sha256": classified.code_sha256,
                    "usage": classified.usage,
                }
                for field, expected in expected_fields.items():
                    if row.get(field) != expected:
                        raise ComparisonError(
                            f"terminal classified field {field} mismatch"
                        )
                response_id = classified.response_id
                if response_id in response_ids:
                    raise ComparisonError(
                        f"duplicate terminal response id: {response_id}"
                    )
                response_ids.add(response_id)
                if row.get("transport_retry") is not False:
                    raise ComparisonError("terminal response is a transport retry")
                if row.get("fatal_response_contract") is not False:
                    raise ComparisonError("terminal response is contract-fatal")
                if row.get("budget_charge_tokens") != classified.usage[
                    "total_tokens"
                ]:
                    raise ComparisonError("terminal budget charge mismatch")
                for key_name in derived_usage:
                    derived_usage[key_name] += classified.usage[key_name]
                terminal_by_slot[slot] = row
                saw_terminal = True
            else:
                if slot_terminal:
                    raise ComparisonError("response-less attempt is terminal")
                if row.get("transport_retry") is not True:
                    raise ComparisonError("response-less attempt is not a transport retry")
                if row.get("retryable_transport") is not True:
                    raise ComparisonError("completed run contains non-retryable API error")
                if row.get("candidate_valid") is not None:
                    raise ComparisonError("transport attempt has candidate validity")
                if row.get("response") is not None or row.get("usage") is not None:
                    raise ComparisonError("transport attempt contains response data")
                if row.get("budget_charge_tokens") != worst_case_reservation:
                    raise ComparisonError("transport budget charge mismatch")

    expected_slots = {
        (task_id, sample_index)
        for task_id in task_ids
        for sample_index in range(k)
    }
    if set(terminal_by_slot) != expected_slots:
        raise ComparisonError(
            "attempt journal does not have exactly one terminal receipt per K slot"
        )

    evaluator = _require_mapping(summary.get("evaluator"), "summary.evaluator")
    evaluator_sha = _require_digest(
        evaluator.get("sha256"),
        "summary.evaluator.sha256",
    )
    attestation_id = str(summary.get("completion_attestation_id") or "")
    if not attestation_id:
        raise ComparisonError("summary lacks completion-attestation identity")
    expected_stability_runs = _require_int(
        config.get("eval_stability_runs"),
        "config.eval_stability_runs",
        minimum=1,
    )
    outcomes_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _load_jsonl(resolved_source / "outcomes.jsonl", "outcome journal"):
        if row.get("schema") != RUN_SCHEMA:
            raise ComparisonError("outcome journal contains a legacy schema")
        slot = (
            str(row.get("task_id") or ""),
            _require_int(row.get("sample_index"), "outcome sample_index"),
        )
        if slot in outcomes_by_slot:
            raise ComparisonError(f"duplicate outcome for slot {slot}")
        terminal = terminal_by_slot.get(slot)
        if terminal is None:
            raise ComparisonError("outcome journal contains a foreign slot")
        for field in (
            "attempt_id",
            "response_id",
            "finish_reason",
            "candidate_valid",
            "terminal_reason",
            "code_sha256",
        ):
            if row.get(field) != terminal.get(field):
                raise ComparisonError(f"outcome/terminal field {field} mismatch")
        if row.get("config_sha256") != config_sha:
            raise ComparisonError("outcome config hash mismatch")
        if row.get("evaluator_sha256") != evaluator_sha:
            raise ComparisonError("outcome evaluator hash mismatch")
        if row.get("completion_attestation_id") != attestation_id:
            raise ComparisonError("outcome attestation identity mismatch")
        compiled_value = _require_bool(row.get("compiled"), "outcome compiled")
        passed_value = _require_bool(row.get("passed"), "outcome passed")
        if passed_value and not compiled_value:
            raise ComparisonError("outcome passed without compiling")
        runs = row.get("stability_runs")
        if not isinstance(runs, list):
            raise ComparisonError("outcome stability runs are malformed")
        if terminal.get("candidate_valid") is False:
            if (
                compiled_value
                or passed_value
                or row.get("evaluation_performed") is not False
                or runs
                or row.get("completion_attestation_enforced") is not False
            ):
                raise ComparisonError(
                    "invalid candidate has a successful/evaluated outcome"
                )
        else:
            if (
                row.get("evaluation_performed") is not True
                or len(runs) != expected_stability_runs
            ):
                raise ComparisonError("evaluable candidate was not evaluated")
            if row.get("completion_attestation_enforced") is not True:
                raise ComparisonError("evaluable outcome lacks attestation")
            if type(
                row.get("completion_attestation_satisfied_all_runs")
            ) is not bool:
                raise ComparisonError("outcome attestation result is not boolean")
            for run in runs:
                run = _require_mapping(run, "stability run")
                if run.get("completion_attestation_id") != attestation_id:
                    raise ComparisonError(
                        "stability-run attestation identity mismatch"
                    )
                if run.get("completion_attestation_required") is not True:
                    raise ComparisonError("stability run did not require attestation")
                run_compiled = _require_bool(
                    run.get("compiled"), "stability run compiled"
                )
                run_passed = _require_bool(
                    run.get("passed"), "stability run passed"
                )
                attested = _require_bool(
                    run.get("completion_attestation_satisfied"),
                    "stability run attestation",
                )
                if run_passed and not run_compiled:
                    raise ComparisonError("stability run passed without compiling")
                if run_passed != attested:
                    raise ComparisonError("stability run attestation mismatch")
            if compiled_value != all(bool(run["compiled"]) for run in runs):
                raise ComparisonError("outcome compile result is not stable")
            if passed_value != all(bool(run["passed"]) for run in runs):
                raise ComparisonError("outcome pass result is not stable")
            if (
                row.get("completion_attestation_satisfied_all_runs")
                != passed_value
            ):
                raise ComparisonError("outcome attestation aggregate mismatch")
        outcomes_by_slot[slot] = row
    if set(outcomes_by_slot) != expected_slots:
        raise ComparisonError("outcome journal does not have exactly K outcomes per task")

    summary_outcomes: dict[tuple[str, int], Mapping[str, Any]] = {}
    for result in results:
        task_id = str(result["task_id"])
        for outcome in result["candidate_outcomes"]:
            slot = (task_id, int(outcome["sample_index"]))
            summary_outcomes[slot] = _require_mapping(
                outcome,
                f"summary outcome {slot}",
            )
    for slot, persisted in outcomes_by_slot.items():
        summarized = summary_outcomes.get(slot)
        if summarized is None:
            raise ComparisonError(f"summary omits outcome slot {slot}")
        for field in (
            "attempt_id",
            "response_id",
            "finish_reason",
            "candidate_valid",
            "terminal_reason",
            "code_sha256",
            "compiled",
            "passed",
        ):
            if summarized.get(field) != persisted.get(field):
                raise ComparisonError(f"summary/outcome field {field} mismatch")

    derived_invalid = sum(
        row.get("candidate_valid") is False for row in terminal_by_slot.values()
    )
    derived_evaluable = len(terminal_by_slot) - derived_invalid
    derived_length = sum(
        row.get("finish_reason") == "length" for row in terminal_by_slot.values()
    )
    for field, expected in (
        ("evaluable_candidates", derived_evaluable),
        ("invalid_candidates", derived_invalid),
        ("transport_retries", transport_retries),
        ("length_slots", derived_length),
        ("model_invalid_responses", derived_invalid),
    ):
        if _require_int(summary.get(field), f"summary.{field}") != expected:
            raise ComparisonError(f"summary {field} disagrees with journals")
    if _require_int(
        summary.get("discarded_terminal_responses"),
        "summary.discarded_terminal_responses",
    ) != 0:
        raise ComparisonError("completed run discarded terminal provider responses")
    summary_usage = _require_mapping(summary.get("usage"), "summary.usage")
    if dict(summary_usage) != derived_usage:
        raise ComparisonError("summary usage disagrees with terminal receipts")
    if _require_int(
        summary.get("recorded_budget_charge_tokens"),
        "summary.recorded_budget_charge_tokens",
    ) != recorded_budget_charge:
        raise ComparisonError("summary budget charge disagrees with attempts")
    budget = _require_mapping(summary.get("budget"), "summary.budget")
    if budget.get("spent") != recorded_budget_charge:
        raise ComparisonError("summary budget ledger disagrees with attempts")

    _validate_rate_block(summary, "pass_at_k", passed)
    _validate_rate_block(summary, "compile_at_k", compiled)
    return CompletedRun(
        source=resolved_source,
        source_kind=source_kind,
        summary_path=summary_path,
        summary_sha256=_sha256_file(summary_path),
        summary=summary,
        provenance=provenance,
        task_ids=tuple(task_ids),
        passed=tuple(passed),
        k=k,
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ComparisonError("cannot calculate a percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def paired_bootstrap_interval(
    deltas: Sequence[int],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Return a seeded paired percentile-bootstrap 95% interval."""

    if not deltas:
        raise ComparisonError("cannot bootstrap an empty paired cohort")
    if any(value not in (-1, 0, 1) for value in deltas):
        raise ComparisonError("paired pass@K deltas must be -1, 0, or 1")
    _require_int(replicates, "bootstrap replicates", minimum=1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ComparisonError("bootstrap seed must be an integer")

    generator = random.Random(seed)
    task_count = len(deltas)
    estimates = [
        sum(deltas[generator.randrange(task_count)] for _ in range(task_count))
        / task_count
        for _ in range(replicates)
    ]
    estimates.sort()
    return (
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    )


def exact_mcnemar(only_a: int, only_b: int) -> dict[str, Any]:
    """Calculate the conventional two-sided exact McNemar binomial test."""

    only_a = _require_int(only_a, "McNemar only_a")
    only_b = _require_int(only_b, "McNemar only_b")
    discordant = only_a + only_b
    if discordant == 0:
        probability = Fraction(1, 1)
    else:
        smaller = min(only_a, only_b)
        lower_tail_numerator = sum(
            math.comb(discordant, successes)
            for successes in range(smaller + 1)
        )
        denominator = 2**discordant
        probability = Fraction(
            min(2 * lower_tail_numerator, denominator),
            denominator,
        )
    return {
        "method": "two-sided exact binomial McNemar test",
        "null_hypothesis": (
            "the two arms have equal marginal task-level pass@K probability"
        ),
        "only_arm_a_passed": only_a,
        "only_arm_b_passed": only_b,
        "discordant_pairs": discordant,
        "p_value": float(probability),
        "p_value_exact_fraction": (
            f"{probability.numerator}/{probability.denominator}"
        ),
    }


def compare_completed_runs(
    arm_a: CompletedRun,
    arm_b: CompletedRun,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Create an auditable paired pass@K comparison (B minus A)."""

    for field in ("provider", "requested_model", "resolved_models"):
        value_a = arm_a.summary.get(field)
        value_b = arm_b.summary.get(field)
        if value_a != value_b:
            raise ComparisonError(
                f"matched-run setting {field!r} differs: "
                f"arm A={value_a!r}, arm B={value_b!r}"
            )
    if arm_a.k != arm_b.k:
        raise ComparisonError(
            f"K mismatch: arm A has K={arm_a.k}, arm B has K={arm_b.k}"
        )
    for field in ("fixed_max_output_tokens", "slot_policy_sha256"):
        value_a = arm_a.summary.get(field)
        value_b = arm_b.summary.get(field)
        if value_a != value_b:
            raise ComparisonError(
                f"matched fixed-slot setting {field!r} differs: "
                f"arm A={value_a!r}, arm B={value_b!r}"
            )
    config_a = _require_mapping(
        arm_a.provenance.get("config"), "arm A provenance config"
    )
    config_b = _require_mapping(
        arm_b.provenance.get("config"), "arm B provenance config"
    )
    for field in (
        "api_base_url_sha256",
        "runtime_identity",
        "expected_evaluator_sha256",
        "expected_dart_sha256",
        "dart_binary",
        "timeout_seconds",
        "eval_timeout_seconds",
        "eval_stability_runs",
    ):
        if config_a.get(field) != config_b.get(field):
            raise ComparisonError(
                f"matched runtime setting {field!r} differs"
            )
    evaluator_a = arm_a.summary.get("evaluator")
    evaluator_b = arm_b.summary.get("evaluator")
    evaluator_sha_a = (
        evaluator_a.get("sha256")
        if isinstance(evaluator_a, Mapping)
        else None
    )
    evaluator_sha_b = (
        evaluator_b.get("sha256")
        if isinstance(evaluator_b, Mapping)
        else None
    )
    if evaluator_sha_a != evaluator_sha_b:
        raise ComparisonError(
            "matched runs use different evaluator identities"
        )
    if arm_a.summary.get("completion_attestation_id") != arm_b.summary.get(
        "completion_attestation_id"
    ):
        raise ComparisonError(
            "matched runs use different completion-attestation contracts"
        )
    input_mode_a = arm_a.summary.get("input_mode")
    input_mode_b = arm_b.summary.get("input_mode")
    if input_mode_a != input_mode_b:
        raise ComparisonError(
            f"input-mode mismatch: arm A={input_mode_a!r}, "
            f"arm B={input_mode_b!r}"
        )
    pair_manifest_sha: str | None = None
    acceptance_sequence_sha: str | None = None
    if input_mode_a == "prematerialized_f2":
        pair_manifest_a = _require_digest(
            arm_a.summary.get("pair_manifest_sha256"),
            "arm A pair_manifest_sha256",
        )
        pair_manifest_b = _require_digest(
            arm_b.summary.get("pair_manifest_sha256"),
            "arm B pair_manifest_sha256",
        )
        if pair_manifest_a != pair_manifest_b:
            raise ComparisonError(
                "paired runs reference different pair manifests"
            )
        pair_manifest_sha = pair_manifest_a
        expected_pair_keys = {
            "opus_real_fn0_cfg",
            "codex_multifunction_cfg",
        }
        observed_pair_keys = {
            str(arm_a.summary.get("pair_arm_key") or ""),
            str(arm_b.summary.get("pair_arm_key") or ""),
        }
        if observed_pair_keys != expected_pair_keys:
            raise ComparisonError(
                "paired runs do not contain exactly one Opus and one Codex "
                f"pair arm: {sorted(observed_pair_keys)}"
            )
        acceptance_a = _require_digest(
            arm_a.summary.get("acceptance_test_sequence_sha256"),
            "arm A acceptance_test_sequence_sha256",
        )
        acceptance_b = _require_digest(
            arm_b.summary.get("acceptance_test_sequence_sha256"),
            "arm B acceptance_test_sequence_sha256",
        )
        if acceptance_a != acceptance_b:
            raise ComparisonError(
                "paired runs use different acceptance-test sequences"
            )
        acceptance_sequence_sha = acceptance_a

    if arm_a.task_ids != arm_b.task_ids:
        mismatch_index = next(
            (
                index
                for index, values in enumerate(zip(arm_a.task_ids, arm_b.task_ids))
                if values[0] != values[1]
            ),
            min(len(arm_a.task_ids), len(arm_b.task_ids)),
        )
        value_a = (
            arm_a.task_ids[mismatch_index]
            if mismatch_index < len(arm_a.task_ids)
            else "<missing>"
        )
        value_b = (
            arm_b.task_ids[mismatch_index]
            if mismatch_index < len(arm_b.task_ids)
            else "<missing>"
        )
        raise ComparisonError(
            "ordered task IDs mismatch at index "
            f"{mismatch_index}: arm A={value_a!r}, arm B={value_b!r}"
        )

    task_count = len(arm_a.task_ids)
    deltas = [
        int(passed_b) - int(passed_a)
        for passed_a, passed_b in zip(arm_a.passed, arm_b.passed)
    ]
    both_failed = sum(
        not passed_a and not passed_b
        for passed_a, passed_b in zip(arm_a.passed, arm_b.passed)
    )
    only_a = sum(
        passed_a and not passed_b
        for passed_a, passed_b in zip(arm_a.passed, arm_b.passed)
    )
    only_b = sum(
        not passed_a and passed_b
        for passed_a, passed_b in zip(arm_a.passed, arm_b.passed)
    )
    both_passed = sum(
        passed_a and passed_b
        for passed_a, passed_b in zip(arm_a.passed, arm_b.passed)
    )
    successes_a = sum(arm_a.passed)
    successes_b = sum(arm_b.passed)
    rate_a = successes_a / task_count
    rate_b = successes_b / task_count
    delta = rate_b - rate_a
    bootstrap_low, bootstrap_high = paired_bootstrap_interval(
        deltas,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    chosen_label_a = label_a or str(
        arm_a.summary.get("dataset_label")
        or arm_a.summary.get("arm")
        or arm_a.summary.get("run_id")
        or "arm_a"
    )
    chosen_label_b = label_b or str(
        arm_b.summary.get("dataset_label")
        or arm_b.summary.get("arm")
        or arm_b.summary.get("run_id")
        or "arm_b"
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "complete",
        "created_at": _utc_now(),
        "direction": "arm_b_minus_arm_a",
        "tasks": task_count,
        "k": arm_a.k,
        "fixed_max_output_tokens": arm_a.summary.get(
            "fixed_max_output_tokens"
        ),
        "slot_policy_sha256": arm_a.summary.get("slot_policy_sha256"),
        "task_set_sha256": _stable_sha256(list(arm_a.task_ids)),
        "pair_manifest_sha256": pair_manifest_sha,
        "acceptance_test_sequence_sha256": acceptance_sequence_sha,
        "inputs": {
            "arm_a": {
                "label": chosen_label_a,
                "source": str(arm_a.source),
                "source_kind": arm_a.source_kind,
                "summary_path": str(arm_a.summary_path),
                "summary_sha256": arm_a.summary_sha256,
                "run_id": arm_a.summary.get("run_id"),
                "dataset_label": arm_a.summary.get("dataset_label"),
                "arm": arm_a.summary.get("arm"),
                "provider": arm_a.summary.get("provider"),
                "requested_model": arm_a.summary.get("requested_model"),
                "resolved_models": arm_a.summary.get("resolved_models"),
                "fixed_max_output_tokens": arm_a.summary.get(
                    "fixed_max_output_tokens"
                ),
                "slot_policy_sha256": arm_a.summary.get(
                    "slot_policy_sha256"
                ),
            },
            "arm_b": {
                "label": chosen_label_b,
                "source": str(arm_b.source),
                "source_kind": arm_b.source_kind,
                "summary_path": str(arm_b.summary_path),
                "summary_sha256": arm_b.summary_sha256,
                "run_id": arm_b.summary.get("run_id"),
                "dataset_label": arm_b.summary.get("dataset_label"),
                "arm": arm_b.summary.get("arm"),
                "provider": arm_b.summary.get("provider"),
                "requested_model": arm_b.summary.get("requested_model"),
                "resolved_models": arm_b.summary.get("resolved_models"),
                "fixed_max_output_tokens": arm_b.summary.get(
                    "fixed_max_output_tokens"
                ),
                "slot_policy_sha256": arm_b.summary.get(
                    "slot_policy_sha256"
                ),
            },
        },
        "pairing_invariants": {
            "ordered_task_ids_identical": True,
            "task_set_sha256_identical": True,
            "k_identical": True,
            "all_input_aggregates_rederived_from_candidate_outcomes": True,
            "provider_and_model_identity_identical": True,
            "fixed_completion_cap_identical": True,
            "exact_response_slot_policy_identical": True,
            "all_k_terminal_receipts_and_outcomes_rederived": True,
            "input_mode_identical": True,
            "shared_pair_manifest_verified": (
                input_mode_a == "prematerialized_f2"
            ),
            "acceptance_test_sequence_identical": (
                input_mode_a == "prematerialized_f2"
            ),
            "arm_a_source_fully_manifest_verified": (
                arm_a.source_kind == "verified_run_directory"
            ),
            "arm_b_source_fully_manifest_verified": (
                arm_b.source_kind == "verified_run_directory"
            ),
        },
        "pass_at_k": {
            "arm_a": {
                "successes": successes_a,
                "total": task_count,
                "rate": rate_a,
            },
            "arm_b": {
                "successes": successes_b,
                "total": task_count,
                "rate": rate_b,
            },
            "delta_arm_b_minus_arm_a": {
                "rate": delta,
                "percentage_points": 100.0 * delta,
            },
            "paired_bootstrap_95": {
                "method": "paired task resampling with percentile interval",
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "lower_rate": bootstrap_low,
                "upper_rate": bootstrap_high,
                "lower_percentage_points": 100.0 * bootstrap_low,
                "upper_percentage_points": 100.0 * bootstrap_high,
            },
        },
        "paired_contingency": {
            "neither_passed": both_failed,
            "only_arm_a_passed": only_a,
            "only_arm_b_passed": only_b,
            "both_passed": both_passed,
        },
        "mcnemar_exact": exact_mcnemar(only_a, only_b),
        "task_results": [
            {
                "task_index": task_index,
                "task_id": task_id,
                "arm_a_passed": passed_a,
                "arm_b_passed": passed_b,
                "delta_arm_b_minus_arm_a": int(passed_b) - int(passed_a),
            }
            for task_index, (task_id, passed_a, passed_b) in enumerate(
                zip(arm_a.task_ids, arm_a.passed, arm_b.passed)
            )
        ],
    }


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise ComparisonError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two completed frontier pass@K runs using paired "
            "task-level statistics. The reported direction is B minus A."
        )
    )
    parser.add_argument("arm_a", type=Path, help="arm A run directory or summary")
    parser.add_argument("arm_b", type=Path, help="arm B run directory or summary")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write new JSON file instead of stdout (existing files are refused)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        arm_a = load_completed_run(args.arm_a)
        arm_b = load_completed_run(args.arm_b)
        comparison = compare_completed_runs(
            arm_a,
            arm_b,
            label_a=args.label_a,
            label_b=args.label_b,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        if args.output is None:
            print(
                json.dumps(
                    comparison,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            _atomic_write_new_json(args.output, comparison)
            print(str(args.output.expanduser().resolve()))
    except ComparisonError as exc:
        print(f"PAIRED_COMPARISON_FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
