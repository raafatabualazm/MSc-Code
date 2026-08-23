#!/usr/bin/env python3
"""Seal a stopped, rejection-sampled 32K frontier pilot as an invalid audit.

This utility is deliberately read-only with respect to the run directory.  It
hashes a stable snapshot, validates the append-only journals, recomputes the
partial selected-candidate metrics, and writes one *sibling* audit file.  It
never imports an API client, evaluates code, resumes a runner, or edits the
pilot.

The old runner discarded terminal model responses that did not satisfy its
candidate contract and retried the same (task, sample) slot.  In particular,
``finish_reason == "length"`` responses at the 32K output cap were replaced.
Consequently, even the exactly recomputed partial metrics below are metrics of
the rejection-selected candidates, not an unbiased pass@K ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "stopped-frontier-32k-pilot-audit-v1"
INVALID_STATUS = "invalid_for_definitive_ceiling"
INVALID_REASONS = [
    "output_cap_censoring",
    "model_output_rejection_sampling",
]
REQUIRED_ATTESTATION_ID = "per-run-256-bit-marker-exactly-once-v1"
REQUIRED_INPUT_MODE = "prematerialized_f2"
REQUIRED_ARM = "compact"
REQUIRED_TASK_COUNT = 175
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SealError(RuntimeError):
    """The pilot cannot be sealed without weakening the audit contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _require_plain_file(path: Path, run_dir: Path) -> None:
    if path.is_symlink():
        raise SealError(f"run snapshot contains a symlink: {path}")
    if not path.is_file():
        raise SealError(f"run snapshot contains a non-regular entry: {path}")
    try:
        path.resolve().relative_to(run_dir)
    except ValueError as exc:
        raise SealError(f"run entry escapes the run directory: {path}") from exc


def snapshot_files(run_dir: Path) -> list[dict[str, Any]]:
    """Hash every file below ``run_dir`` and reject ambiguous filesystem entries."""

    records: list[dict[str, Any]] = []
    entries = sorted(run_dir.rglob("*"), key=lambda item: item.as_posix())
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            continue
        _require_plain_file(entry, run_dir)
        before = entry.stat()
        digest = sha256_file(entry)
        after = entry.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            raise SealError(f"file changed while it was hashed: {entry}")
        records.append(
            {
                "relative_path": entry.relative_to(run_dir).as_posix(),
                "bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": digest,
            }
        )
    if not records:
        raise SealError(f"run directory has no files: {run_dir}")
    return records


def snapshot_identity(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(records)))


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SealError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SealError(f"{label} must be one JSON object: {path}")
    return value


def load_jsonl(
    path: Path,
    label: str,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except Exception as exc:
        raise SealError(
            f"cannot parse {label} {path} at or before line {line_number}: {exc}"
        ) from exc
    if not rows and not allow_empty:
        raise SealError(f"{label} is empty: {path}")
    return rows


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SealError(f"{label} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SealError(f"{label} must be boolean")
    return value


def _require_sha(value: Any, label: str) -> str:
    digest = str(value or "")
    if not SHA256_RE.fullmatch(digest):
        raise SealError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "total": total,
        "rate": (successes / total) if total else None,
        "rate_fraction": f"{successes}/{total}",
    }


def _response_finish_reason(response: Any) -> str | None:
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    value = choice.get("finish_reason")
    return str(value) if value is not None else ""


def _response_id(response: Any) -> str:
    if not isinstance(response, Mapping):
        return ""
    return str(response.get("id") or "")


def _validate_response_usage(
    row: Mapping[str, Any],
    *,
    label: str,
    response_present: bool,
) -> Mapping[str, Any] | None:
    """Require byte-equivalent row/raw usage and internally consistent totals."""

    row_usage = row.get("usage")
    response = row.get("response")
    raw_usage = (
        response.get("usage")
        if isinstance(response, Mapping)
        and isinstance(response.get("usage"), Mapping)
        else None
    )
    if not response_present:
        if row_usage is not None or raw_usage is not None:
            raise SealError(f"{label} has usage without a provider response")
        return None
    if not isinstance(row_usage, Mapping) or not isinstance(raw_usage, Mapping):
        raise SealError(f"{label} provider response lacks duplicated usage evidence")
    if canonical_json_bytes(dict(row_usage)) != canonical_json_bytes(dict(raw_usage)):
        raise SealError(f"{label} row usage disagrees with raw-response usage")
    values: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values[name] = _require_int(
            row_usage.get(name), f"{label}.usage.{name}", minimum=1
        )
    if values["total_tokens"] != (
        values["prompt_tokens"] + values["completion_tokens"]
    ):
        raise SealError(f"{label} usage total is internally inconsistent")
    return row_usage


def _is_cap_censored(row: Mapping[str, Any], output_cap: int) -> bool:
    finish_reason = _response_finish_reason(row.get("response"))
    if finish_reason == "length":
        return True
    reason = str(row.get("invalid_reason") or "").lower()
    # Retain a journal-level fallback for SDK/provider envelopes that could not
    # be serialized in full, while keeping token-at-cap as a separate diagnostic.
    return "finish_reason" in reason and "length" in reason


def _provider_response_present(row: Mapping[str, Any]) -> bool:
    # The legacy runner stored None only when it had no terminal provider
    # response.  Even a malformed serialized object is a returned response and
    # therefore should have consumed a fixed sample slot.
    return isinstance(row.get("response"), Mapping)


def _validate_stability_evidence(
    row: Mapping[str, Any],
    label: str,
    *,
    expected_evaluator_sha256: str,
) -> None:
    if row.get("evaluator_sha256") != expected_evaluator_sha256:
        raise SealError(f"{label} evaluator fingerprint mismatch")
    if row.get("evaluator_entrypoint") != "evaluate_dart_jit_tests_detail":
        raise SealError(f"{label} evaluator entrypoint mismatch")
    if row.get("completion_attestation_id") != REQUIRED_ATTESTATION_ID:
        raise SealError(f"{label} completion-attestation identity mismatch")
    if row.get("completion_attestation_enforced") is not True:
        raise SealError(f"{label} completion attestation was not enforced")
    runs = row.get("stability_runs")
    if not isinstance(runs, list) or not runs:
        raise SealError(f"{label} has no stability-run evidence")
    all_compiled = True
    all_passed = True
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise SealError(f"{label}.stability_runs[{index}] is malformed")
        compiled = _require_bool(
            run.get("compiled"), f"{label}.stability_runs[{index}].compiled"
        )
        passed = _require_bool(
            run.get("passed"), f"{label}.stability_runs[{index}].passed"
        )
        if run.get("completion_attestation_id") != REQUIRED_ATTESTATION_ID:
            raise SealError(
                f"{label}.stability_runs[{index}] attestation identity mismatch"
            )
        if run.get("completion_attestation_required") is not True:
            raise SealError(
                f"{label}.stability_runs[{index}] did not require attestation"
            )
        if (
            _require_bool(
                run.get("completion_attestation_satisfied"),
                (
                    f"{label}.stability_runs[{index}]"
                    ".completion_attestation_satisfied"
                ),
            )
            != passed
        ):
            raise SealError(
                f"{label}.stability_runs[{index}] attestation/pass mismatch"
            )
        _require_sha(
            run.get("evaluated_source_sha256"),
            f"{label}.stability_runs[{index}].evaluated_source_sha256",
        )
        if passed and not compiled:
            raise SealError(f"{label}.stability_runs[{index}] passed without compile")
        all_compiled = all_compiled and compiled
        all_passed = all_passed and passed
    if _require_bool(row.get("compiled"), f"{label}.compiled") != all_compiled:
        raise SealError(f"{label} compiled flag disagrees with stability runs")
    if _require_bool(row.get("passed"), f"{label}.passed") != all_passed:
        raise SealError(f"{label} passed flag disagrees with stability runs")
    if (
        _require_bool(
            row.get("completion_attestation_satisfied_all_runs"),
            f"{label}.completion_attestation_satisfied_all_runs",
        )
        != all_passed
    ):
        raise SealError(f"{label} aggregate attestation/pass mismatch")


def recompute_partial_diagnostics(
    *,
    attempts: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    planned_task_ids: Sequence[str],
    prompt_sha256_by_task: Mapping[str, str],
    config_sha256: str,
    k: int,
    output_cap: int,
    expected_model: str,
    expected_provider: str,
    expected_evaluator_sha256: str,
) -> dict[str, Any]:
    """Validate journals and compute exact, explicitly non-definitive metrics."""

    planned = set(planned_task_ids)
    if len(planned) != len(planned_task_ids):
        raise SealError("tasks journal contains duplicate task ids")

    slots: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    attempts_by_id: dict[str, dict[str, Any]] = {}
    seen_attempt_positions: set[tuple[str, int, int]] = set()
    seen_response_ids: set[str] = set()
    invalid_reasons: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    valid_attempts = 0
    invalid_attempts = 0
    provider_responses = 0
    rejected_provider_responses = 0
    cap_censored = 0
    cap_censored_rejected = 0
    completion_tokens_at_or_above_cap = 0
    transport_or_no_response_attempts = 0

    for row_number, original in enumerate(attempts, 1):
        row = dict(original)
        label = f"attempts.jsonl row {row_number}"
        if row.get("config_sha256") != config_sha256:
            raise SealError(f"{label} has a foreign config fingerprint")
        task_id = str(row.get("task_id") or "")
        sample_index = _require_int(
            row.get("sample_index"), f"{label}.sample_index"
        )
        attempt_index = _require_int(
            row.get("attempt_index"), f"{label}.attempt_index"
        )
        if task_id not in planned:
            raise SealError(f"{label} references an unplanned task: {task_id!r}")
        if sample_index >= k:
            raise SealError(f"{label} sample_index {sample_index} is outside K={k}")
        if row.get("prompt_sha256") != prompt_sha256_by_task.get(task_id):
            raise SealError(f"{label} prompt fingerprint mismatch")
        if row.get("requested_model") != expected_model:
            raise SealError(f"{label} requested-model mismatch")
        if row.get("provider") != expected_provider:
            raise SealError(f"{label} provider mismatch")
        position = (task_id, sample_index, attempt_index)
        if position in seen_attempt_positions:
            raise SealError(f"duplicate attempt position: {position}")
        seen_attempt_positions.add(position)
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id or attempt_id in attempts_by_id:
            raise SealError(f"{label} has a missing or duplicate attempt_id")
        attempts_by_id[attempt_id] = row
        slots[(task_id, sample_index)].append(row)

        is_valid = _require_bool(row.get("valid"), f"{label}.valid")
        response_present = _provider_response_present(row)
        usage = _validate_response_usage(
            row,
            label=label,
            response_present=response_present,
        )
        if is_valid:
            valid_attempts += 1
            if not response_present:
                raise SealError(f"{label} is valid but has no stored provider response")
            code = str(row.get("code") or "")
            if not code or sha256_text(code) != _require_sha(
                row.get("code_sha256"), f"{label}.code_sha256"
            ):
                raise SealError(f"{label} candidate hash mismatch")
        else:
            invalid_attempts += 1
            invalid_reasons[str(row.get("invalid_reason") or "unknown")] += 1

        if response_present:
            provider_responses += 1
            response_id = _response_id(row.get("response"))
            if not response_id:
                raise SealError(f"{label} provider response id is empty")
            if response_id in seen_response_ids:
                raise SealError(f"duplicate provider response id: {response_id}")
            seen_response_ids.add(response_id)
            raw_response = row["response"]
            if str(raw_response.get("model") or "") != expected_model:
                raise SealError(f"{label} raw-response model mismatch")
            if is_valid:
                if row.get("response_id") != response_id:
                    raise SealError(f"{label} selected response-id binding mismatch")
                if row.get("resolved_model") != expected_model:
                    raise SealError(f"{label} selected resolved-model mismatch")
                if row.get("finish_reason") != _response_finish_reason(raw_response):
                    raise SealError(f"{label} selected finish-reason binding mismatch")
            finish_reason = _response_finish_reason(row.get("response"))
            finish_reasons[
                finish_reason if finish_reason is not None else "(unparseable)"
            ] += 1
            if not is_valid:
                rejected_provider_responses += 1
        else:
            transport_or_no_response_attempts += 1

        capped = _is_cap_censored(row, output_cap)
        if capped:
            cap_censored += 1
            if not is_valid:
                cap_censored_rejected += 1
        if usage is not None:
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_totals[name] += int(usage[name])
            completion = usage.get("completion_tokens")
            if isinstance(completion, int) and completion >= output_cap:
                completion_tokens_at_or_above_cap += 1

    if not attempts_by_id:
        raise SealError("attempt journal is empty")

    slots_with_multiple_attempts = 0
    slots_with_response_replacement = 0
    selected_after_rejected_response = 0
    accepted_slots = 0
    attempts_per_slot: list[int] = []
    provider_responses_per_slot: list[int] = []
    first_response_cap_censored = 0
    first_response_rejected = 0
    first_provider_attempt_ids: set[str] = set()
    valid_attempt_by_slot: dict[tuple[str, int], dict[str, Any]] = {}

    for slot, rows in slots.items():
        ordered = sorted(rows, key=lambda item: int(item["attempt_index"]))
        indices = [int(item["attempt_index"]) for item in ordered]
        if indices != list(range(0, indices[-1] + 1)):
            raise SealError(
                f"attempt indexes must start at 0 and be contiguous for slot {slot}"
            )
        attempts_per_slot.append(len(ordered))
        if len(ordered) > 1:
            slots_with_multiple_attempts += 1
        returned = [item for item in ordered if _provider_response_present(item)]
        provider_responses_per_slot.append(len(returned))
        if len(returned) > 1:
            slots_with_response_replacement += 1
        if returned:
            first = returned[0]
            first_provider_attempt_ids.add(str(first["attempt_id"]))
            if not bool(first["valid"]):
                first_response_rejected += 1
            if _is_cap_censored(first, output_cap):
                first_response_cap_censored += 1
        valid = [item for item in ordered if bool(item["valid"])]
        if len(valid) > 1:
            raise SealError(f"multiple selected valid candidates occupy slot {slot}")
        if valid:
            accepted_slots += 1
            selected = valid[0]
            valid_attempt_by_slot[slot] = selected
            earlier_rejected = any(
                _provider_response_present(item) and not bool(item["valid"])
                for item in ordered
                if int(item["attempt_index"]) < int(selected["attempt_index"])
            )
            if earlier_rejected:
                selected_after_rejected_response += 1

    outcome_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    seen_outcome_keys: set[tuple[str, int, str]] = set()
    for row_number, original in enumerate(outcomes, 1):
        row = dict(original)
        label = f"outcomes.jsonl row {row_number}"
        if row.get("config_sha256") != config_sha256:
            raise SealError(f"{label} has a foreign config fingerprint")
        task_id = str(row.get("task_id") or "")
        sample_index = _require_int(
            row.get("sample_index"), f"{label}.sample_index"
        )
        attempt_id = str(row.get("attempt_id") or "")
        key = (task_id, sample_index, attempt_id)
        if not attempt_id or key in seen_outcome_keys:
            raise SealError(f"{label} has a missing or duplicate outcome identity")
        seen_outcome_keys.add(key)
        attempt = attempts_by_id.get(attempt_id)
        if attempt is None or not bool(attempt.get("valid")):
            raise SealError(f"{label} does not reference a selected valid attempt")
        if (
            task_id != str(attempt.get("task_id") or "")
            or sample_index != int(attempt.get("sample_index", -1))
        ):
            raise SealError(f"{label} identity disagrees with its attempt")
        code_sha = _require_sha(row.get("code_sha256"), f"{label}.code_sha256")
        if code_sha != attempt.get("code_sha256"):
            raise SealError(f"{label} code hash disagrees with its attempt")
        _validate_stability_evidence(
            row,
            label,
            expected_evaluator_sha256=expected_evaluator_sha256,
        )
        slot = (task_id, sample_index)
        if slot in outcome_by_slot:
            raise SealError(f"multiple outcomes occupy selected slot {slot}")
        outcome_by_slot[slot] = row

    candidate_compiled = sum(bool(row["compiled"]) for row in outcome_by_slot.values())
    candidate_passed = sum(bool(row["passed"]) for row in outcome_by_slot.values())
    outcomes_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (task_id, _sample_index), row in outcome_by_slot.items():
        outcomes_by_task[task_id].append(row)
    complete_tasks = {
        task_id: rows
        for task_id, rows in outcomes_by_task.items()
        if len(rows) == k
        and {
            sample
            for (candidate_task, sample) in outcome_by_slot
            if candidate_task == task_id
        }
        == set(range(k))
    }
    complete_task_compiled = sum(
        any(bool(row["compiled"]) for row in rows)
        for rows in complete_tasks.values()
    )
    complete_task_passed = sum(
        any(bool(row["passed"]) for row in rows)
        for rows in complete_tasks.values()
    )
    observed_task_compiled = sum(
        any(bool(row["compiled"]) for row in rows)
        for rows in outcomes_by_task.values()
    )
    observed_task_passed = sum(
        any(bool(row["passed"]) for row in rows)
        for rows in outcomes_by_task.values()
    )
    first_response_outcomes = sum(
        str(row.get("attempt_id") or "") in first_provider_attempt_ids
        for row in outcome_by_slot.values()
    )
    planned_slots = len(planned_task_ids) * k
    fixed_slot_metric_reconstructable = (
        len(first_provider_attempt_ids) == planned_slots
        and first_response_outcomes == planned_slots
    )

    if cap_censored <= 0:
        raise SealError("no output-cap-censored response was found in the pilot")
    if rejected_provider_responses <= 0:
        raise SealError("no rejected terminal model response was found in the pilot")
    if slots_with_response_replacement <= 0:
        raise SealError(
            "no same-slot provider-response replacement was found; "
            "model-output rejection sampling is not evidenced"
        )

    attempts_per_slot_mean = (
        math.fsum(attempts_per_slot) / len(attempts_per_slot)
        if attempts_per_slot
        else None
    )
    return {
        "journal_rows": {
            "attempts": len(attempts),
            "outcomes": len(outcomes),
        },
        "progress": {
            "planned_tasks": len(planned_task_ids),
            "planned_slots": planned_slots,
            "tasks_with_any_attempt": len({task for task, _ in slots}),
            "slots_with_any_attempt": len(slots),
            "selected_valid_slots": accepted_slots,
            "evaluated_selected_slots": len(outcome_by_slot),
            "tasks_with_any_selected_outcome": len(outcomes_by_task),
            "tasks_with_exactly_k_selected_outcomes": len(complete_tasks),
        },
        "selected_candidate_partial_metrics": {
            "warning": (
                "Exact for recorded rejection-selected candidates only; invalid "
                "for estimating the fixed-slot definitive ceiling."
            ),
            "candidate_compile": _rate(candidate_compiled, len(outcome_by_slot)),
            "candidate_pass": _rate(candidate_passed, len(outcome_by_slot)),
            "observed_task_any_compile": _rate(
                observed_task_compiled, len(outcomes_by_task)
            ),
            "observed_task_any_pass": _rate(
                observed_task_passed, len(outcomes_by_task)
            ),
            "complete_selected_task_compile_at_k": _rate(
                complete_task_compiled, len(complete_tasks)
            ),
            "complete_selected_task_pass_at_k": _rate(
                complete_task_passed, len(complete_tasks)
            ),
        },
        "rejection_sampling": {
            "api_attempts": len(attempts),
            "valid_selected_attempts": valid_attempts,
            "invalid_attempts": invalid_attempts,
            "terminal_provider_responses": provider_responses,
            "rejected_terminal_provider_responses": rejected_provider_responses,
            "transport_or_no_response_attempts": transport_or_no_response_attempts,
            "slots_with_multiple_attempts": slots_with_multiple_attempts,
            "slots_with_multiple_provider_responses": slots_with_response_replacement,
            "selected_valid_slots_after_rejected_provider_response": (
                selected_after_rejected_response
            ),
            "invalid_attempt_reasons": dict(sorted(invalid_reasons.items())),
            "response_finish_reasons": dict(sorted(finish_reasons.items())),
            "attempts_per_touched_slot": {
                "minimum": min(attempts_per_slot) if attempts_per_slot else None,
                "maximum": max(attempts_per_slot) if attempts_per_slot else None,
                "mean": attempts_per_slot_mean,
            },
            "provider_responses_per_touched_slot": {
                "minimum": (
                    min(provider_responses_per_slot)
                    if provider_responses_per_slot
                    else None
                ),
                "maximum": (
                    max(provider_responses_per_slot)
                    if provider_responses_per_slot
                    else None
                ),
            },
        },
        "output_cap_censoring": {
            "configured_output_cap_tokens": output_cap,
            "censored_terminal_provider_responses": cap_censored,
            "censored_responses_rejected_by_runner": cap_censored_rejected,
            "first_provider_responses_censored": first_response_cap_censored,
            "provider_responses_with_completion_tokens_at_or_above_cap": (
                completion_tokens_at_or_above_cap
            ),
        },
        "fixed_slot_reconstruction": {
            "first_terminal_provider_responses": len(first_provider_attempt_ids),
            "first_terminal_provider_responses_rejected": first_response_rejected,
            "first_terminal_provider_responses_with_recorded_outcome": (
                first_response_outcomes
            ),
            "definitive_fixed_slot_metric_reconstructable": (
                fixed_slot_metric_reconstructable
            ),
            "why_not": (
                None
                if fixed_slot_metric_reconstructable
                else (
                    "Rejected terminal model outputs were not evaluated, and/or "
                    "the stopped pilot does not contain every planned fixed slot."
                )
            ),
        },
        "usage_summed_over_attempt_journal": {
            "prompt_tokens": usage_totals["prompt_tokens"],
            "completion_tokens": usage_totals["completion_tokens"],
            "total_tokens": usage_totals["total_tokens"],
        },
    }


def query_systemd_unit(unit: str) -> dict[str, str]:
    if not unit or "\x00" in unit or "\n" in unit or "\r" in unit:
        raise SealError(f"invalid systemd unit name: {unit!r}")
    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                unit,
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        raise SealError(f"cannot query systemd unit {unit!r}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise SealError(
            f"systemd unit query failed for {unit!r} "
            f"(exit {result.returncode}): {detail}"
        )
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    if properties.get("LoadState") != "loaded":
        raise SealError(
            f"systemd unit {unit!r} is not loaded "
            f"(LoadState={properties.get('LoadState')!r})"
        )
    if properties.get("ActiveState") not in {"inactive", "failed"}:
        raise SealError(
            f"systemd unit {unit!r} is not inactive "
            f"(ActiveState={properties.get('ActiveState')!r})"
        )
    return properties


def verify_services_inactive(
    service_units: Sequence[str],
    *,
    unit_query: Any = query_systemd_unit,
) -> list[dict[str, str]]:
    if not service_units:
        raise SealError("at least one --service unit is required")
    if len(set(service_units)) != len(service_units):
        raise SealError("duplicate --service unit")
    records: list[dict[str, str]] = []
    for unit in service_units:
        properties = unit_query(unit)
        if str(properties.get("LoadState") or "") != "loaded":
            raise SealError(
                f"systemd unit {unit!r} is not loaded "
                f"(LoadState={properties.get('LoadState')!r})"
            )
        if str(properties.get("ActiveState") or "") not in {
            "inactive",
            "failed",
        }:
            raise SealError(
                f"systemd unit {unit!r} is not inactive "
                f"(ActiveState={properties.get('ActiveState')!r})"
            )
        records.append(
            {
                "unit": unit,
                "load_state": str(properties.get("LoadState") or ""),
                "active_state": str(properties.get("ActiveState") or ""),
                "sub_state": str(properties.get("SubState") or ""),
            }
        )
    return records


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_run_lock(run_dir: Path) -> dict[str, Any]:
    path = run_dir / ".run.lock"
    if not path.exists():
        return {"present": False, "owner_process_alive": False}
    if path.is_symlink() or not path.is_file():
        raise SealError(".run.lock is not a regular file")
    payload = load_json_object(path, "run lock")
    pid = _require_int(payload.get("pid"), ".run.lock.pid", minimum=1)
    host = str(payload.get("host") or "")
    local_host = socket.gethostname()
    same_host = host in {local_host, socket.getfqdn(), "localhost"}
    if not same_host:
        raise SealError(
            f".run.lock claims foreign host {host!r}; local host is {local_host!r}"
        )
    alive = _pid_exists(pid)
    if alive:
        raise SealError(f".run.lock owner process is still alive: pid={pid}")
    return {
        "present": True,
        "pid": pid,
        "host": host,
        "same_host": same_host,
        "owner_process_alive": alive,
        "note": "stale lock retained unchanged in sealed source snapshot",
    }


def build_audit(
    run_dir: Path,
    *,
    service_units: Sequence[str],
    expected_config_sha256: str,
    expected_pair_manifest_sha256: str,
    expected_pair_arm_key: str,
    expected_evaluator_sha256: str,
    expected_output_cap: int = 32768,
    expected_k: int = 10,
    expected_model: str = "deepseek-v4-pro",
    expected_provider: str = "deepseek",
    expected_task_count: int = REQUIRED_TASK_COUNT,
    unit_query: Any = query_systemd_unit,
) -> dict[str, Any]:
    """Build an audit in memory; do not write or mutate anything."""

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SealError(f"run directory does not exist: {run_dir}")
    expected_output_cap = _require_int(
        expected_output_cap, "expected_output_cap", minimum=1
    )
    expected_k = _require_int(expected_k, "expected_k", minimum=1)
    expected_task_count = _require_int(
        expected_task_count, "expected_task_count", minimum=1
    )
    expected_config_sha256 = _require_sha(
        expected_config_sha256, "expected_config_sha256"
    )
    expected_pair_manifest_sha256 = _require_sha(
        expected_pair_manifest_sha256, "expected_pair_manifest_sha256"
    )
    expected_evaluator_sha256 = _require_sha(
        expected_evaluator_sha256, "expected_evaluator_sha256"
    )
    if not expected_pair_arm_key:
        raise SealError("expected_pair_arm_key is empty")

    first_snapshot = snapshot_files(run_dir)
    provenance_path = run_dir / "provenance.json"
    attempts_path = run_dir / "attempts.jsonl"
    outcomes_path = run_dir / "outcomes.jsonl"
    tasks_path = run_dir / "tasks.jsonl"
    prompts_path = run_dir / "prompts.jsonl"
    for required in (
        provenance_path,
        attempts_path,
        outcomes_path,
        tasks_path,
        prompts_path,
    ):
        if not required.is_file() or required.is_symlink():
            raise SealError(f"required pilot artifact is missing: {required}")

    provenance = load_json_object(provenance_path, "provenance")
    attempts = load_jsonl(attempts_path, "attempt journal")
    outcomes = load_jsonl(outcomes_path, "outcome journal", allow_empty=True)
    tasks = load_jsonl(tasks_path, "tasks journal")
    prompts = load_jsonl(prompts_path, "prompts journal")
    config = provenance.get("config")
    if not isinstance(config, dict):
        raise SealError("provenance.config is missing or malformed")
    config_sha = _require_sha(
        provenance.get("config_sha256"), "provenance.config_sha256"
    )
    if config_sha != expected_config_sha256:
        raise SealError(
            f"pilot config hash is {config_sha}, expected {expected_config_sha256}"
        )
    recomputed_config_sha = sha256_bytes(canonical_json_bytes(config))
    if config_sha != recomputed_config_sha:
        raise SealError(
            "provenance config hash mismatch: "
            f"{config_sha} != {recomputed_config_sha}"
        )
    output_cap = _require_int(
        config.get("max_output_tokens"), "config.max_output_tokens", minimum=1
    )
    k = _require_int(config.get("k"), "config.k", minimum=1)
    if output_cap != expected_output_cap:
        raise SealError(
            f"pilot output cap is {output_cap}, expected {expected_output_cap}"
        )
    if k != expected_k:
        raise SealError(f"pilot K is {k}, expected {expected_k}")
    requested_model = str(config.get("model_requested") or "")
    if requested_model != expected_model:
        raise SealError(
            f"pilot requested model is {requested_model!r}, "
            f"expected {expected_model!r}"
        )
    if config.get("provider") != expected_provider:
        raise SealError(
            f"pilot provider is {config.get('provider')!r}, "
            f"expected {expected_provider!r}"
        )
    if config.get("input_mode") != REQUIRED_INPUT_MODE:
        raise SealError(
            f"pilot input_mode is {config.get('input_mode')!r}, "
            f"expected {REQUIRED_INPUT_MODE!r}"
        )
    if config.get("arm") != REQUIRED_ARM:
        raise SealError(
            f"pilot arm is {config.get('arm')!r}, expected {REQUIRED_ARM!r}"
        )
    if config.get("pair_arm_key") != expected_pair_arm_key:
        raise SealError("pilot pair-arm key does not match the caller pin")
    sealed_inputs = config.get("sealed_inputs")
    if not isinstance(sealed_inputs, Mapping):
        raise SealError("config.sealed_inputs is missing")
    if (
        sealed_inputs.get("pair_manifest_sha256")
        != expected_pair_manifest_sha256
    ):
        raise SealError("config pair-manifest hash does not match the caller pin")
    if sealed_inputs.get("pair_arm_key") != expected_pair_arm_key:
        raise SealError("sealed-input pair-arm key does not match the caller pin")
    if config.get("expected_evaluator_sha256") != expected_evaluator_sha256:
        raise SealError("config evaluator hash does not match the caller pin")
    evaluator_record = provenance.get("evaluator")
    if not isinstance(evaluator_record, Mapping):
        raise SealError("provenance evaluator record is missing")
    if evaluator_record.get("sha256") != expected_evaluator_sha256:
        raise SealError("provenance evaluator hash does not match the caller pin")
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SealError("provenance artifacts record is missing")
    pair_record = artifacts.get("pair_manifest")
    if not isinstance(pair_record, Mapping):
        raise SealError("provenance pair-manifest artifact is missing")
    if pair_record.get("sha256") != expected_pair_manifest_sha256:
        raise SealError("provenance pair-manifest hash does not match the caller pin")
    pair_claims = provenance.get("source_pair_manifest_claims")
    if not isinstance(pair_claims, Mapping):
        raise SealError("provenance pair-manifest claims are missing")
    if pair_claims.get("sha256") != expected_pair_manifest_sha256:
        raise SealError("provenance pair-manifest claim hash mismatch")
    if pair_claims.get("pair_arm_key") != expected_pair_arm_key:
        raise SealError("provenance pair-arm claim mismatch")

    task_ids = [str(row.get("task_id") or "") for row in tasks]
    if any(not value for value in task_ids):
        raise SealError("tasks journal contains an empty task_id")
    if len(task_ids) != expected_task_count:
        raise SealError(
            f"tasks journal has {len(task_ids)} rows, "
            f"caller-pinned expected count is {expected_task_count}"
        )
    prompt_ids = [str(row.get("task_id") or "") for row in prompts]
    if prompt_ids != task_ids:
        raise SealError("prompt and task journal task order differs")
    prompt_sha256_by_task: dict[str, str] = {}
    for index, prompt in enumerate(prompts):
        label = f"prompts.jsonl row {index + 1}"
        task_id = prompt_ids[index]
        if prompt.get("arm") != REQUIRED_ARM:
            raise SealError(f"{label} arm mismatch")
        if prompt.get("input_mode") != REQUIRED_INPUT_MODE:
            raise SealError(f"{label} input-mode mismatch")
        messages = prompt.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SealError(f"{label} messages are missing")
        prompt_sha = _require_sha(
            prompt.get("prompt_sha256"), f"{label}.prompt_sha256"
        )
        if prompt_sha != sha256_bytes(canonical_json_bytes(messages)):
            raise SealError(f"{label} messages/prompt hash mismatch")
        prompt_sha256_by_task[task_id] = prompt_sha
    configured_task_count = config.get("expected_task_count")
    if isinstance(configured_task_count, int) and not isinstance(
        configured_task_count, bool
    ):
        if len(task_ids) != configured_task_count:
            raise SealError(
                f"tasks journal has {len(task_ids)} rows, "
                f"expected {configured_task_count}"
            )
    else:
        raise SealError("config.expected_task_count is missing or malformed")

    service_records_before = verify_services_inactive(
        service_units, unit_query=unit_query
    )
    lock_record_before = inspect_run_lock(run_dir)
    diagnostics = recompute_partial_diagnostics(
        attempts=attempts,
        outcomes=outcomes,
        planned_task_ids=task_ids,
        prompt_sha256_by_task=prompt_sha256_by_task,
        config_sha256=config_sha,
        k=k,
        output_cap=output_cap,
        expected_model=expected_model,
        expected_provider=expected_provider,
        expected_evaluator_sha256=expected_evaluator_sha256,
    )

    # Hash again after parsing and process-state checks.  Any append, replacement,
    # or metadata change makes the seal fail instead of producing a mixed-time
    # snapshot.
    second_snapshot = snapshot_files(run_dir)
    if first_snapshot != second_snapshot:
        raise SealError("run directory changed while the audit was built")
    service_records_after = verify_services_inactive(
        service_units, unit_query=unit_query
    )
    lock_record_after = inspect_run_lock(run_dir)
    if service_records_before != service_records_after:
        raise SealError("service state changed during the sealing audit")
    if lock_record_before != lock_record_after:
        raise SealError("run-lock state changed during the sealing audit")
    third_snapshot = snapshot_files(run_dir)
    if second_snapshot != third_snapshot:
        raise SealError("run directory changed during the final inactivity recheck")

    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "status": INVALID_STATUS,
        "invalid_for_definitive_ceiling_reasons": list(INVALID_REASONS),
        "sealed_at": utc_now(),
        "source_run": {
            "path": str(run_dir),
            "run_id": run_dir.name,
            "source_directory_mutated": False,
            "snapshot_semantics": "stable_read_only_point_in_time",
            "snapshot_sha256": snapshot_identity(first_snapshot),
            "file_count": len(first_snapshot),
            "files": first_snapshot,
        },
        "runner_state": {
            "services_verified_inactive_before_snapshot_confirmation": (
                service_records_before
            ),
            "services_verified_inactive_after_snapshot_confirmation": (
                service_records_after
            ),
            "run_lock_before_snapshot_confirmation": lock_record_before,
            "run_lock_after_snapshot_confirmation": lock_record_after,
            "provenance_status_at_seal": provenance.get("status"),
        },
        "contract": {
            "config_sha256": config_sha,
            "requested_model": requested_model,
            "k": k,
            "max_output_tokens": output_cap,
            "planned_tasks": len(task_ids),
            "input_mode": REQUIRED_INPUT_MODE,
            "arm": REQUIRED_ARM,
            "pair_manifest_sha256": expected_pair_manifest_sha256,
            "pair_arm_key": expected_pair_arm_key,
            "evaluator_sha256": expected_evaluator_sha256,
        },
        "diagnostics": diagnostics,
        "interpretation": {
            "definitive_ceiling_result": None,
            "partial_metrics_publishable_as_ceiling": False,
            "pilot_reusable_for_cost_and_failure_mode_diagnostics": True,
            "required_successor_design": (
                "Use one terminal provider response per predeclared slot; "
                "finish_reason='length' must consume that slot, with any safe "
                "extractable Dart evaluated and unusable output scored as failure."
            ),
        },
    }
    audit["audit_payload_sha256"] = sha256_bytes(canonical_json_bytes(audit))
    return audit


def default_output_path(run_dir: Path) -> Path:
    resolved = run_dir.expanduser().resolve()
    return resolved.with_name(resolved.name + ".stopped_32k_pilot.audit.json")


def write_sibling_audit_exclusive(
    run_dir: Path,
    output_path: Path,
    audit: Mapping[str, Any],
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if output.parent != run_dir.parent:
        raise SealError(
            "audit output must be a sibling of the source run directory"
        )
    if output == run_dir or output.is_relative_to(run_dir):
        raise SealError("audit output must not be inside the source run directory")
    payload = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise SealError(f"refusing to overwrite existing audit: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seal one stopped 32K pilot as invalid for a definitive ceiling. "
            "No API calls, evaluations, resumes, or source-directory writes occur."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--service",
        action="append",
        required=True,
        help="systemd service that must be loaded and inactive (repeatable)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-output-cap", type=int, default=32768)
    parser.add_argument("--expected-k", type=int, default=10)
    parser.add_argument("--expected-model", default="deepseek-v4-pro")
    parser.add_argument("--expected-provider", default="deepseek")
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
    parser.add_argument("--expected-pair-arm-key", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(run_dir)
    )
    try:
        audit = build_audit(
            run_dir,
            service_units=args.service,
            expected_config_sha256=args.expected_config_sha256,
            expected_pair_manifest_sha256=args.expected_pair_manifest_sha256,
            expected_pair_arm_key=args.expected_pair_arm_key,
            expected_evaluator_sha256=args.expected_evaluator_sha256,
            expected_output_cap=args.expected_output_cap,
            expected_k=args.expected_k,
            expected_model=args.expected_model,
            expected_provider=args.expected_provider,
        )
        written = write_sibling_audit_exclusive(run_dir, output, audit)
    except Exception as exc:
        print(
            f"PILOT_SEAL_FAILED_CLOSED error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        "PILOT_SEALED_INVALID "
        f"status={INVALID_STATUS} "
        f"snapshot_sha256={audit['source_run']['snapshot_sha256']} "
        f"audit_sha256={sha256_file(written)} "
        f"output={written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
