#!/usr/bin/env python3
"""Second typed direct-only RS-SFT pass over newly verified targets.

This profile continues the sealed 225-target typed-direct checkpoint, but it
does not replay any of those 225 rows (or any gold row).  Its only training
rows are the independently audited 190-target local harvest and the final
direct-code aggregate produced by the Kimi/Sonnet dual-provider controller.

The API corpus size is intentionally not a source-code constant.  It is read
only after the completed orchestration report, every non-empty phase report,
the orchestration journal, and the aggregate artifacts have been SHA-pinned
and cross-validated.  Complete TRAIN acceptance tests are used only as a
private gate and are never serialized into a model-visible row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluation.durable_evaluation_journal import (
    canonical_sha256,
    journal_record,
    load_journal,
    sha256_file,
)
from scripts.evaluation.graph_compile_at_k_antigravity import validate_dart_binary
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade
from scripts.training import t5gemma2_typed_direct_rs_sft as pass1
from scripts.training import t5gemma2_typed_dual_api_orchestrator as dual
from scripts.training import t5gemma2_typed_local_direct_harvest as local_harvest


RUN_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-checkpoint-v1"
DATASET_SCHEMA = "t5gemma2-typed-direct-rs-sft-pass2-dataset-v1"

EXPECTED_LOCAL_NEW_ROWS = 190
MAX_DUAL_API_ROWS = 88  # 50 unique Kimi tasks + at most 38 Sonnet residuals.
EXPECTED_PRIOR_ROWS = pass1.EXPECTED_DIRECT_ROWS

# The paid harvest is late-bound, so its output digests cannot be source-code
# constants.  The code that was allowed to produce those bytes can and must be
# pinned, however.  These are the reviewed producer versions deployed for this
# one Kimi-then-Sonnet harvest.
EXPECTED_DUAL_PRODUCER_SHA256 = (
    "15020fefa5e617029abdf62832a349a968ac23837c8e244073568ccde0b0d30e"
)
EXPECTED_CASCADE_PRODUCER_SHA256 = (
    "7a03af003e998497012706361f5cbf0734d8defa82c7e458aa5f87f796e01143"
)
EXPECTED_PHASE_LAUNCHER_SHA256 = (
    "c69e845cfefcd91555171813a66492dba0b2b5c9d44bbd8efd21175f5f7f2e14"
)

LOCAL_SPEC_BASENAMES = (
    "harvest_report.json",
    "harvest.journal.jsonl",
    "direct_targets.jsonl",
    "dataset_manifest.json",
)
API_SPEC_BASENAMES = (
    "orchestration_report.json",
    "direct_manifest.json",
    "direct_targets.jsonl",
)

_MIXED_VALIDATE_WARMSTART = mixed.validate_warmstart
_MIXED_RUNTIME_CONTRACT = mixed._runtime_contract  # noqa: SLF001


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is absent or malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(
    path: Path, label: str, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} is absent") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ValueError(f"{label}:{line_number}: blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number}: malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number}: row is not an object")
        rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"{label} is empty")
    return rows


def _require_specs(
    specs: Sequence[tuple[Path, str]],
    basenames: Sequence[str],
    label: str,
) -> list[tuple[Path, str]]:
    if len(specs) != len(basenames):
        raise ValueError(
            f"{label} requires exactly {len(basenames)} SHA-pinned artifacts"
        )
    result = list(specs)
    for index, ((path, digest), basename) in enumerate(
        zip(result, basenames, strict=True)
    ):
        if path.name != basename or sha256_file(path) != digest:
            raise ValueError(f"{label} artifact {index} binding differs")
    return result


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _phase_provider_expectation(phase: str) -> dict[str, Any]:
    if phase == cascade.PHASE_KIMI_INITIAL:
        return {
            "provider": "openrouter_chat",
            "model": cascade.KIMI_MODEL,
            "base_url": "https://openrouter.ai/api/v1",
            "max_output_tokens": cascade.KIMI_INITIAL_MAX_OUTPUT,
            "reasoning_effort": None,
            "chat_token_parameter": "max_tokens",
            "openrouter_reasoning": {
                "enabled": True,
                "included_in_response": True,
                "effort": "low",
            },
        }
    if phase == cascade.PHASE_KIMI_RETRY:
        return {
            "provider": "openrouter_chat",
            "model": cascade.KIMI_MODEL,
            "base_url": "https://openrouter.ai/api/v1",
            "max_output_tokens": cascade.KIMI_RETRY_MAX_OUTPUT,
            "reasoning_effort": None,
            "chat_token_parameter": "max_tokens",
            "openrouter_reasoning": {
                "enabled": True,
                "included_in_response": True,
                "effort": "low",
            },
        }
    if phase == cascade.PHASE_SONNET_RESIDUAL:
        return {
            "provider": "anthropic",
            "model": cascade.SONNET_MODEL,
            "base_url": "https://api.anthropic.com",
            "max_output_tokens": cascade.SONNET_MAX_OUTPUT,
            "thinking": "adaptive",
            "effort": "high",
        }
    raise ValueError(f"unexpected dual-provider phase {phase!r}")


def _audit_phase_evidence(
    report_path: Path,
    *,
    expected_phase: str,
    expected_cohort: int,
    expected_input_record: Mapping[str, Any],
    expected_source_journal_record: Mapping[str, Any],
    expected_prior_records: Sequence[Mapping[str, Any]],
    expected_cascade_input_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every published API target back to its paid call and private gate."""

    output_dir = report_path.parent.resolve()
    report = _read_object(report_path, "phase report")
    manifest = _read_object(
        output_dir / "direct_manifest.json", "phase direct manifest"
    )
    target_path = output_dir / "direct_targets.jsonl"
    journal_path = output_dir / "typed_api_rescue.journal.jsonl"
    rows = _read_jsonl(target_path, "phase direct targets", allow_empty=True)
    events = load_journal(journal_path)
    if not events:
        raise ValueError("phase journal is empty")
    header = events[0]
    contract = header.get("contract")
    cascade_input_record = (
        contract.get("inputs") if isinstance(contract, Mapping) else None
    )
    cascade_source_record = (
        cascade_input_record.get("source")
        if isinstance(cascade_input_record, Mapping)
        else None
    )
    typed_local_input = (
        cascade_source_record.get("typed_local_harvest")
        if isinstance(cascade_source_record, Mapping)
        else None
    )
    expected_phase_source_journal = dict(expected_source_journal_record)
    expected_phase_source_journal.update(
        {
            "mode": cascade.LOCAL_SOURCE_MODE,
            "exploratory_prefix": False,
            "production_floor_eligible": True,
            "terminal_prefix_length": None,
            "source_journal_modified": False,
        }
    )
    contract_sha256 = (
        canonical_sha256(contract) if isinstance(contract, Mapping) else ""
    )
    if (
        header.get("event") != "header"
        or header.get("schema") != cascade.JOURNAL_SCHEMA
        or not isinstance(contract, Mapping)
        or header.get("contract_sha256") != contract_sha256
        or contract.get("schema") != cascade.RUN_SCHEMA
        or contract.get("script_sha256") != EXPECTED_CASCADE_PRODUCER_SHA256
        or contract.get("phase") != expected_phase
        or contract.get("cohort_index") != expected_cohort
        or not isinstance(cascade_input_record, Mapping)
        or not isinstance(cascade_source_record, Mapping)
        or not isinstance(typed_local_input, Mapping)
        or {
            key: value
            for key, value in typed_local_input.items()
            if key != "existing_225_manifest"
        }
        != dict(expected_input_record)
        or not isinstance(typed_local_input.get("existing_225_manifest"), Mapping)
        or not isinstance(
            cascade_source_record.get("permitted_visible_train_split"), Mapping
        )
        or cascade_source_record.get("complete_acceptance_text_serialized") is not False
        or cascade_source_record.get("private_holdback_text_serialized") is not False
        or not isinstance(
            cascade_input_record.get("visible_failure_projection"), Mapping
        )
        or not isinstance(cascade_input_record.get("existing_225_exclusion"), Mapping)
        or (
            expected_cascade_input_record is not None
            and cascade_input_record != expected_cascade_input_record
        )
        or contract.get("source_local_harvest_journal") != expected_phase_source_journal
        or contract.get("prior_reports") != list(expected_prior_records)
        or report.get("run_contract_sha256") != contract_sha256
    ):
        raise ValueError("phase journal producer/input contract differs")

    privacy = contract.get("privacy")
    training_outputs = contract.get("training_outputs")
    verification_contract = contract.get("verification")
    if (
        not isinstance(privacy, Mapping)
        or privacy.get("private_complete_acceptance_sent_to_provider") is not False
        or privacy.get("private_split_holdback_sent_to_provider") is not False
        or privacy.get("gold_sent_to_provider") is not False
        or privacy.get("heldout_175_opened") is not False
        or not isinstance(training_outputs, Mapping)
        or training_outputs.get("direct_verified_code_targets") is not True
        or training_outputs.get("repair_conditioned_rows") != 0
        or training_outputs.get("gold_replay_rows") != 0
        or training_outputs.get("reasoning_rows") != 0
        or training_outputs.get("tests_in_training_outputs") is not False
        or training_outputs.get("production_floor_eligible") is not True
        or not isinstance(verification_contract, Mapping)
        or verification_contract.get("all_api_calls_before_any_private_gate")
        is not True
        or verification_contract.get("private_gate") != "complete_TRAIN_acceptance"
        or verification_contract.get("stability_runs") != cascade.STABILITY_RUNS
        or verification_contract.get("private_failure_triggers_api_call") is not False
        or verification_contract.get("private_gate_can_only_reject_transfer")
        is not True
        or contract.get("heldout_175_opened") is not False
        or report.get("privacy_invariants") != privacy
        or report.get("heldout_175_opened") is not False
    ):
        raise ValueError("phase privacy/no-gold-source contract differs")

    provider = contract.get("provider")
    expected_provider = _phase_provider_expectation(expected_phase)
    if (
        not isinstance(provider, Mapping)
        or report.get("provider") != provider
        or provider.get("credential_source") != "environment_value_not_persisted"
        or provider.get("one_candidate_per_call") is not True
        or any(provider.get(key) != value for key, value in expected_provider.items())
    ):
        raise ValueError("phase provider/model contract differs")

    selection = contract.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("phase selection contract is absent")
    scheduled_tasks = selection.get("scheduled_tasks")
    scheduled_slots = selection.get("scheduled_slots")
    if (
        type(scheduled_tasks) is not int
        or scheduled_tasks <= 0
        or scheduled_tasks != scheduled_slots
        or selection.get("max_parents_per_task") != 1
        or selection.get("samples_per_parent") != 1
        or not _is_sha256(selection.get("task_ids_sha256"))
    ):
        raise ValueError("phase one-call-per-task selection differs")

    binding_keys = (
        "slot_position",
        "task_position",
        "task_id",
        "parent_position",
        "sample_index",
        "parent_code_sha256",
        "diagnostic_sha256",
        "feedback_source_sha256",
        "prompt_sha256",
    )
    cursor = 1
    intents: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for slot_position in range(scheduled_slots):
        if cursor + 1 >= len(events):
            raise ValueError("phase journal ended inside the API call ledger")
        intent = events[cursor]
        result = events[cursor + 1]
        usage = result.get("usage")
        code = str(result.get("code") or "")
        if (
            intent.get("event") != "call_intent"
            or intent.get("schema") != cascade.JOURNAL_SCHEMA
            or intent.get("slot_position") != slot_position
            or result.get("event") != "call_result"
            or result.get("schema") != cascade.JOURNAL_SCHEMA
            or any(result.get(key) != intent.get(key) for key in binding_keys)
            or result.get("status")
            not in {"response", "provider_error", "contract_error"}
            or not isinstance(usage, Mapping)
            or any(
                type(usage.get(key)) is not int or usage[key] < 0
                for key in (
                    "charged_input_tokens",
                    "charged_output_tokens",
                    "charged_usd_nanos",
                )
            )
            or (code and mixed.sha256_text(code) != result.get("code_sha256"))
            or (not code and result.get("code_sha256") is not None)
        ):
            raise ValueError("phase API intent/result binding differs")
        intents.append(dict(intent))
        results.append(dict(result))
        cursor += 2

    task_ids = [str(row.get("task_id") or "") for row in intents]
    if (
        any(not task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
        or selection.get("task_ids_sha256") != canonical_sha256(task_ids)
    ):
        raise ValueError("phase scheduled task identity accounting differs")

    verifications: list[dict[str, Any]] = []
    expected_rows: list[dict[str, Any]] = []
    selected_slots: set[int] = set()
    for task_position, task_id in enumerate(task_ids):
        if cursor >= len(events):
            raise ValueError(
                "phase journal ended inside the private verification ledger"
            )
        event = events[cursor]
        selected = event.get("selected_target")
        if (
            event.get("event") != "task_verification"
            or event.get("schema") != cascade.JOURNAL_SCHEMA
            or event.get("task_position") != task_position
            or event.get("task_id") != task_id
            or not _is_sha256(event.get("source_sha256"))
            or event.get("all_api_generation_completed_before_private_gate") is not True
            or event.get("private_feedback_serialized_to_model") is not False
            or event.get("holdback_failure_triggers_generation") is not False
            or event.get("private_diagnostics_persisted") is not False
        ):
            raise ValueError("phase verification order/source binding differs")
        if selected is not None:
            if not isinstance(selected, Mapping):
                raise ValueError("phase selected target is malformed")
            code = str(selected.get("code") or "")
            slot_position = selected.get("slot_position")
            if type(slot_position) is not int or not 0 <= slot_position < len(results):
                raise ValueError("phase selected target has an invalid API slot")
            result = results[slot_position]
            visible = [
                row
                for row in event.get("visible_results", [])
                if isinstance(row, Mapping)
                and row.get("slot_position") == slot_position
            ]
            private = [
                row
                for row in event.get("private_gate_results", [])
                if isinstance(row, Mapping)
                and row.get("slot_position") == slot_position
            ]
            if (
                selected.get("schema") != cascade.DIRECT_TARGET_SCHEMA
                or selected.get("task_id") != task_id
                or selected.get("source_sha256") != event.get("source_sha256")
                or not code.strip()
                or mixed.sha256_text(code) != selected.get("code_sha256")
                or selected.get("visible_passed") is not True
                or selected.get("private_gate_passed") is not True
                or selected.get("exploratory_prefix") is not False
                or selected.get("production_floor_eligible") is not True
                or selected.get("training_use_forbidden") is not False
                or result.get("task_id") != task_id
                or result.get("status") != "response"
                or result.get("parse_accepted") is not True
                or result.get("code") != code
                or result.get("code_sha256") != selected.get("code_sha256")
                or len(visible) != 1
                or visible[0].get("code") != code
                or visible[0].get("code_sha256") != selected.get("code_sha256")
                or visible[0].get("passed") is not True
                or len(private) != 1
                or private[0].get("code_sha256") != selected.get("code_sha256")
                or private[0].get("private_gate_passed") is not True
                or slot_position in selected_slots
            ):
                raise ValueError(
                    "phase target is not bound to its API result and private gate"
                )
            selected_slots.add(slot_position)
            expected_rows.append(
                {
                    "schema": cascade.DIRECT_TARGET_SCHEMA,
                    "task_id": task_id,
                    "source_sha256": selected["source_sha256"],
                    "dart_source": code,
                    "dart_source_sha256": selected["code_sha256"],
                    "origin": "external_teacher_direct_verified",
                    "provider_phase": expected_phase,
                    "provider_model": provider["model"],
                    "visible_train_passed": True,
                    "private_full_acceptance_passed": True,
                    "stability_runs": cascade.STABILITY_RUNS,
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
        verifications.append(dict(event))
        cursor += 1

    if cursor >= len(events):
        raise ValueError("phase journal lacks a completion event")
    complete = events[cursor]
    if (
        complete.get("event") != "complete"
        or complete.get("schema") != cascade.JOURNAL_SCHEMA
        or complete.get("tasks") != scheduled_tasks
        or complete.get("slots") != scheduled_slots
        or complete.get("verified_targets") != len(expected_rows)
        or complete.get("exploratory_prefix") is not False
        or complete.get("production_floor_eligible") is not True
        or cursor + 1 != len(events)
        or rows != expected_rows
    ):
        raise ValueError("phase completion/direct-target reconstruction differs")

    actual_journal = journal_record(journal_path)
    report_journal = report.get("journal")
    output_record = report.get("outputs", {}).get("direct_targets")
    verification_report = report.get("verification")
    schedule_report = report.get("schedule")
    budget = report.get("budget_charged")
    charged_input = sum(row["usage"]["charged_input_tokens"] for row in results)
    charged_output = sum(row["usage"]["charged_output_tokens"] for row in results)
    charged_nanos = sum(row["usage"]["charged_usd_nanos"] for row in results)
    verified_ids = [row["task_id"] for row in rows]
    retry_ids: list[str] = []
    verified_set = set(verified_ids)
    for result in results:
        response = result.get("response")
        finish_reason = (
            str(response.get("finish_reason") or "")
            if isinstance(response, Mapping)
            else ""
        )
        if str(result.get("task_id") or "") not in verified_set and (
            result.get("parse_accepted") is not True or finish_reason == "length"
        ):
            retry_ids.append(str(result.get("task_id") or ""))
    if (
        not isinstance(report_journal, Mapping)
        or any(
            report_journal.get(key) != actual_journal.get(key)
            for key in (
                "sha256",
                "chain_head_sha256",
                "event_count",
                "head_event_sha256",
            )
        )
        or not isinstance(output_record, Mapping)
        or output_record.get("sha256") != sha256_file(target_path)
        or output_record.get("rows") != len(rows)
        or report.get("direct_manifest") != manifest
        or manifest.get("schema") != cascade.DIRECT_MANIFEST_SCHEMA
        or manifest.get("run_contract_sha256") != contract_sha256
        or manifest.get("targets") != output_record
        or manifest.get("rows") != len(rows)
        or manifest.get("task_ids_sha256") != canonical_sha256(verified_ids)
        or manifest.get("direct_only") is not True
        or manifest.get("full_acceptance_reverified") is not True
        or manifest.get("stability_runs") != cascade.STABILITY_RUNS
        or manifest.get("repair_conditioned_rows") != 0
        or manifest.get("gold_replay_rows") != 0
        or manifest.get("reasoning_rows") != 0
        or manifest.get("tests_in_training_output") is not False
        or manifest.get("private_feedback_in_training_output") is not False
        or manifest.get("production_floor_eligible") is not True
        or not isinstance(schedule_report, Mapping)
        or schedule_report.get("scheduled_tasks") != scheduled_tasks
        or schedule_report.get("scheduled_calls") != scheduled_slots
        or schedule_report.get("task_ids_sha256") != canonical_sha256(task_ids)
        or schedule_report.get("provider_responses")
        != sum(row.get("status") == "response" for row in results)
        or schedule_report.get("code_only_responses")
        != sum(row.get("parse_accepted") is True for row in results)
        or schedule_report.get("retry_eligible_non_code_or_length_tasks")
        != len(retry_ids)
        or schedule_report.get("retry_eligible_task_ids_sha256")
        != canonical_sha256(retry_ids)
        or not isinstance(verification_report, Mapping)
        or verification_report.get("private_full_acceptance_passes") != len(rows)
        or verification_report.get("verified_unique_hard_targets") != len(rows)
        or verification_report.get("verified_task_ids_sha256")
        != canonical_sha256(verified_ids)
        or not isinstance(budget, Mapping)
        or budget.get("calls") != len(results)
        or budget.get("input_tokens") != charged_input
        or budget.get("output_tokens") != charged_output
        or budget.get("total_tokens") != charged_input + charged_output
        or budget.get("estimated_usd_nanos") != charged_nanos
        or Decimal(str(budget.get("estimated_usd")))
        != Decimal(charged_nanos) / Decimal(1_000_000_000)
        or budget.get("within_contract") is not True
    ):
        raise ValueError("phase report/manifest/journal accounting differs")
    return {
        "phase": expected_phase,
        "cohort_index": expected_cohort,
        "run_contract_sha256": contract_sha256,
        "cascade_input_record": dict(cascade_input_record),
        "provider": dict(provider),
        "scheduled_task_ids": task_ids,
        "scheduled_tasks": scheduled_tasks,
        "schedule_sha256": canonical_sha256(task_ids),
        "verified_task_ids": verified_ids,
        "retry_eligible_task_ids": retry_ids,
        "rows_bound_to_call_result_and_private_gate": True,
        "gold_source_replay": False,
    }


def _load_prior_225_manifest(
    path: Path, digest: str
) -> tuple[set[str], dict[str, Any]]:
    if sha256_file(path) != digest:
        raise ValueError("prior-225 dataset manifest digest differs")
    manifest = _read_object(path, "prior-225 dataset manifest")
    schedule = manifest.get("schedule")
    composition = manifest.get("composition")
    if (
        manifest.get("schema") != pass1.DATASET_SCHEMA
        or manifest.get("rows") != EXPECTED_PRIOR_ROWS
        or manifest.get("heldout_overlap") != 0
        or manifest.get("tests_model_visible") is not False
        or manifest.get("private_feedback_model_visible") is not False
        or manifest.get("repair_conditioned_prefixes_visible") is not False
        or not isinstance(composition, Mapping)
        or composition.get("verified_direct") != EXPECTED_PRIOR_ROWS
        or composition.get("gold_replay") != 0
        or composition.get("repair_conditioned") != 0
        or not isinstance(schedule, list)
        or len(schedule) != EXPECTED_PRIOR_ROWS
        or manifest.get("schedule_sha256") != canonical_sha256(schedule)
    ):
        raise ValueError("prior-225 dataset safety contract differs")
    task_ids = [
        str(row.get("source_task_id") or "")
        for row in schedule
        if isinstance(row, Mapping)
    ]
    if (
        len(task_ids) != EXPECTED_PRIOR_ROWS
        or any(not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
        or pass1.CONTAMINATED_TRAIN_TASK_ID in task_ids
    ):
        raise ValueError("prior-225 task identity accounting differs")
    return set(task_ids), {
        "path": str(path.resolve()),
        "sha256": digest,
        "rows": len(task_ids),
        "task_ids_sha256": canonical_sha256(sorted(task_ids)),
        "used_for_training_in_this_stage": False,
        "used_for_exclusion_only": True,
    }


def audit_completed_dual_orchestration(
    *,
    report_path: Path,
    report_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    targets_path: Path,
    targets_sha256: str,
    expected_input_record: Mapping[str, Any],
    expected_source_journal_record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deep-audit the sealed dual-provider result before observing its row count."""

    if not (report_path.parent == manifest_path.parent == targets_path.parent):
        raise ValueError("dual-provider aggregate artifacts are not siblings")
    for path, expected, label in (
        (report_path, report_sha256, "orchestration report"),
        (manifest_path, manifest_sha256, "dual direct manifest"),
        (targets_path, targets_sha256, "dual direct targets"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} digest differs")

    root = report_path.parent.resolve()
    report = _read_object(report_path, "orchestration report")
    manifest = _read_object(manifest_path, "dual direct manifest")
    rows = _read_jsonl(targets_path, "dual direct targets", allow_empty=True)
    target_record = manifest.get("targets")
    source_reports = manifest.get("source_reports")
    phases = report.get("phases")
    final_index = report.get("prior_report_index")
    provider = report.get("providers")
    if (
        report.get("schema") != dual.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("heldout_175_opened_for_exclusion_audit") is not True
        or report.get("heldout_175_model_visible") is not False
        or report.get("heldout_175_used_for_generation_or_selection") is not False
        or report.get("direct_manifest") != manifest
        or not isinstance(phases, list)
        or not isinstance(final_index, Mapping)
        or not isinstance(provider, Mapping)
        or manifest.get("schema") != dual.AGGREGATE_SCHEMA
        or manifest.get("rows") != len(rows)
        or manifest.get("direct_only") is not True
        or manifest.get("visible_and_private_verified") is not True
        or manifest.get("reasoning_rows") != 0
        or manifest.get("repair_conditioned_rows") != 0
        or manifest.get("gold_replay_rows") != 0
        or manifest.get("tests_in_training_output") is not False
        or manifest.get("diagnostics_in_training_output") is not False
        or manifest.get("production_floor_eligible") is not True
        or not isinstance(source_reports, list)
        or not isinstance(target_record, Mapping)
        or Path(str(target_record.get("path") or "")).name != targets_path.name
        or target_record.get("sha256") != targets_sha256
        or target_record.get("rows") != len(rows)
        or manifest.get("task_ids_sha256")
        != canonical_sha256([row.get("task_id") for row in rows])
    ):
        raise ValueError("completed dual-provider aggregate contract differs")

    expected_phase_names = [cascade.PHASE_KIMI_INITIAL]
    if (
        len(phases) == 3
        and isinstance(phases[1], Mapping)
        and phases[1].get("phase") == cascade.PHASE_KIMI_RETRY
    ):
        expected_phase_names.append(cascade.PHASE_KIMI_RETRY)
    expected_phase_names.append(cascade.PHASE_SONNET_RESIDUAL)
    phase_names = [
        str(row.get("phase") or "") if isinstance(row, Mapping) else ""
        for row in phases
    ]
    if (
        len(phases) not in {2, 3}
        or phase_names != expected_phase_names
        or any(
            not isinstance(row, Mapping) or row.get("cohort_index") != 0
            for row in phases
        )
        or any(
            not str(row.get(key) or "")
            for row in phases[:-1]
            for key in ("report_sha256", "journal_sha256", "targets_sha256")
        )
        or (
            bool(phases[-1].get("report_sha256"))
            != bool(phases[-1].get("journal_sha256"))
        )
        or (
            bool(phases[-1].get("report_sha256"))
            != bool(phases[-1].get("targets_sha256"))
        )
    ):
        raise ValueError("dual-provider phase order/inventory differs")

    openrouter = provider.get("openrouter")
    anthropic = provider.get("anthropic")
    try:
        openrouter_charged = Decimal(str(openrouter.get("charged_usd")))
        anthropic_charged = Decimal(str(anthropic.get("charged_usd")))
    except (AttributeError, ArithmeticError) as exc:
        raise ValueError("dual-provider spend accounting is malformed") from exc
    if (
        not isinstance(openrouter, Mapping)
        or not isinstance(anthropic, Mapping)
        or openrouter.get("max_usd") != "12.0"
        or anthropic.get("max_usd") != "11.5"
        or not openrouter_charged.is_finite()
        or not anthropic_charged.is_finite()
        or not Decimal(0) <= openrouter_charged <= dual.OPENROUTER_CAP
        or not Decimal(0) <= anthropic_charged <= dual.ANTHROPIC_CAP
    ):
        raise ValueError("dual-provider spend exceeds its sealed contract")

    entries = final_index.get("entries")
    tsv = final_index.get("tsv")
    if (
        final_index.get("schema") != dual.INDEX_SCHEMA
        or final_index.get("status") != "complete"
        or not isinstance(entries, list)
        or not entries
        or final_index.get("entries_sha256") != canonical_sha256(entries)
        or not isinstance(tsv, Mapping)
    ):
        raise ValueError("dual-provider final phase index differs")
    tsv_path = root / Path(str(tsv.get("path") or "")).name
    expected_tsv = "".join(
        f"{entry.get('report_sha256')}\t{entry.get('report_path')}\n"
        for entry in entries
    )
    if (
        not tsv_path.is_file()
        or sha256_file(tsv_path) != tsv.get("sha256")
        or tsv_path.read_text(encoding="utf-8") != expected_tsv
    ):
        raise ValueError("dual-provider final phase TSV differs")

    audited_phases: list[dict[str, Any]] = []
    strict_phase_evidence: list[dict[str, Any]] = []
    reconstructed_rows: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("position") != position:
            raise ValueError("dual-provider phase index order differs")
        phase_report_path = (
            Path(str(entry.get("report_path") or "")).expanduser().resolve()
        )
        if root not in phase_report_path.parents:
            raise ValueError(
                "dual-provider phase report escaped its orchestration root"
            )
        phase = str(entry.get("phase") or "")
        cohort = int(entry.get("cohort_index", -1))
        audited = dual.inspect_phase_report(
            phase_report_path,
            expected_phase=phase,
            expected_cohort=cohort,
        )
        for key in ("report_sha256", "journal_sha256", "targets_sha256"):
            if audited[key] != entry.get(key):
                raise ValueError(f"dual-provider phase {key} differs")
        expected_prior_records = [
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
            for row in audited_phases
        ]
        strict = _audit_phase_evidence(
            phase_report_path,
            expected_phase=phase,
            expected_cohort=cohort,
            expected_input_record=expected_input_record,
            expected_source_journal_record=expected_source_journal_record,
            expected_prior_records=expected_prior_records,
            expected_cascade_input_record=(
                strict_phase_evidence[0]["cascade_input_record"]
                if strict_phase_evidence
                else None
            ),
        )
        if (
            strict["verified_task_ids"] != audited["verified_task_ids"]
            or strict["retry_eligible_task_ids"] != audited["retry_eligible_task_ids"]
        ):
            raise ValueError("dual-provider strict phase evidence differs")
        audited_phases.append(audited)
        strict_phase_evidence.append(strict)
        reconstructed_rows.extend(
            _read_jsonl(Path(audited["targets_path"]), "audited phase targets")
        )
    if reconstructed_rows != rows:
        raise ValueError("dual-provider aggregate is not the ordered phase union")

    nonempty_phases = [
        row
        for row in phases
        if isinstance(row, Mapping) and str(row.get("report_sha256") or "")
    ]
    if [
        {
            key: row.get(key)
            for key in (
                "phase",
                "cohort_index",
                "report_sha256",
                "journal_sha256",
                "targets_sha256",
            )
        }
        for row in nonempty_phases
    ] != [
        {
            key: row.get(key)
            for key in (
                "phase",
                "cohort_index",
                "report_sha256",
                "journal_sha256",
                "targets_sha256",
            )
        }
        for row in audited_phases
    ]:
        raise ValueError("orchestration report phase inventory differs")
    expected_source_reports = [
        {
            "phase": row.get("phase"),
            "report_sha256": row.get("report_sha256"),
            "targets_sha256": row.get("targets_sha256"),
        }
        for row in phases
        if isinstance(row, Mapping)
    ]
    if source_reports != expected_source_reports:
        raise ValueError("dual-provider aggregate source-report inventory differs")

    orchestration_journal = root / "orchestration.journal.jsonl"
    actual_journal = journal_record(orchestration_journal)
    reported_journal = report.get("journal")
    events = load_journal(orchestration_journal)
    contract = events[0].get("contract") if events else None
    phase_launcher = (
        contract.get("phase_launcher") if isinstance(contract, Mapping) else None
    )
    if (
        not isinstance(reported_journal, Mapping)
        or any(
            actual_journal.get(key) != reported_journal.get(key)
            for key in (
                "sha256",
                "chain_head_sha256",
                "event_count",
                "head_event_sha256",
            )
        )
        or Path(str(reported_journal.get("path") or "")).name
        != orchestration_journal.name
        or not isinstance(contract, Mapping)
        or contract.get("schema") != dual.RUN_SCHEMA
        or events[0].get("event") != "header"
        or events[0].get("contract_sha256") != canonical_sha256(contract)
        or report.get("run_contract_sha256") != canonical_sha256(contract)
        or contract.get("script_sha256") != EXPECTED_DUAL_PRODUCER_SHA256
        or not isinstance(phase_launcher, Mapping)
        or Path(str(phase_launcher.get("path") or "")).name
        != "t5gemma2_typed_api_rescue_cascade.sh"
        or phase_launcher.get("sha256") != EXPECTED_PHASE_LAUNCHER_SHA256
        or not _is_sha256(contract.get("initial_schedule_sha256"))
        or contract.get("kimi")
        != {
            "cohorts": 1,
            "openrouter_max_usd": "12.0",
            "retry": "non_code_or_length_only",
        }
        or contract.get("sonnet")
        != {
            "model": cascade.SONNET_MODEL,
            "anthropic_max_usd": "11.5",
            "max_output_tokens": cascade.SONNET_MAX_OUTPUT,
        }
        or contract.get("training_output") != "direct_verified_code_only"
    ):
        raise ValueError("dual-provider orchestration journal differs")
    completions = [row for row in events if row.get("event") == "phase_complete"]
    if len(completions) != len(audited_phases):
        raise ValueError("dual-provider phase completion count differs")
    for audited in audited_phases:
        matching = [
            row
            for row in completions
            if row.get("phase") == audited["phase"]
            and row.get("cohort_index") == audited["cohort_index"]
        ]
        if (
            len(matching) != 1
            or any(
                matching[0].get(key) != audited[key]
                for key in ("report_sha256", "journal_sha256", "targets_sha256")
            )
            or Decimal(str(matching[0].get("spent"))) != audited["spent"]
        ):
            raise ValueError("dual-provider completion evidence differs")

    if openrouter_charged != sum(
        (
            row["spent"]
            for row in audited_phases
            if str(row["phase"]).startswith("kimi")
        ),
        Decimal(0),
    ) or anthropic_charged != sum(
        (
            row["spent"]
            for row in audited_phases
            if row["phase"] == cascade.PHASE_SONNET_RESIDUAL
        ),
        Decimal(0),
    ):
        raise ValueError("dual-provider cumulative spend accounting differs")

    starts = [row for row in events if row.get("event") == "phase_start"]
    if len(starts) != len(audited_phases):
        raise ValueError("dual-provider phase-start count differs")
    for audited, strict in zip(audited_phases, strict_phase_evidence, strict=True):
        matching = [
            row
            for row in starts
            if row.get("phase") == audited["phase"]
            and row.get("cohort_index") == audited["cohort_index"]
        ]
        plan_path = root / f"plan_{audited['phase']}_c000.json"
        plan = _read_object(plan_path, "credential-free phase plan")
        if (
            len(matching) != 1
            or matching[0].get("plan_sha256") != sha256_file(plan_path)
            or matching[0].get("schedule_sha256") != strict["schedule_sha256"]
            or plan.get("schema") != cascade.PLAN_SCHEMA
            or plan.get("status") != "complete"
            or plan.get("phase") != audited["phase"]
            or plan.get("cohort_index") != 0
            or plan.get("fixed_kimi_cohort_limit") != 1
            or plan.get("provider_credentials_read") is not False
            or plan.get("frontier_api_calls") is not False
            or plan.get("selection", {}).get("scheduled_tasks")
            != strict["scheduled_tasks"]
            or plan.get("selection", {}).get("task_ids_sha256")
            != strict["schedule_sha256"]
        ):
            raise ValueError("dual-provider plan/start/paid schedule binding differs")

    sonnet_present = any(
        row["phase"] == cascade.PHASE_SONNET_RESIDUAL for row in strict_phase_evidence
    )
    sonnet_skips = [
        row
        for row in events
        if row.get("event") == "phase_skipped_no_residual"
        and row.get("phase") == cascade.PHASE_SONNET_RESIDUAL
    ]
    if (sonnet_present and sonnet_skips) or (
        not sonnet_present
        and (
            len(sonnet_skips) != 1
            or sonnet_skips[0].get("cohort_index") != 0
            or not _is_sha256(sonnet_skips[0].get("plan_sha256"))
            or not _is_sha256(sonnet_skips[0].get("schedule_sha256"))
        )
    ):
        raise ValueError("dual-provider Sonnet residual/skip evidence differs")
    if not sonnet_present:
        sonnet_plan_path = root / f"plan_{cascade.PHASE_SONNET_RESIDUAL}_c000.json"
        sonnet_plan = _read_object(sonnet_plan_path, "empty Sonnet residual plan")
        if (
            sonnet_skips[0].get("plan_sha256") != sha256_file(sonnet_plan_path)
            or sonnet_plan.get("schema") != cascade.PLAN_SCHEMA
            or sonnet_plan.get("status") != "complete"
            or sonnet_plan.get("phase") != cascade.PHASE_SONNET_RESIDUAL
            or sonnet_plan.get("cohort_index") != 0
            or sonnet_plan.get("provider_credentials_read") is not False
            or sonnet_plan.get("frontier_api_calls") is not False
            or sonnet_plan.get("selection", {}).get("scheduled_tasks") != 0
            or sonnet_plan.get("selection", {}).get("task_ids_sha256")
            != sonnet_skips[0].get("schedule_sha256")
        ):
            raise ValueError("empty Sonnet residual plan/skip binding differs")

    initial = strict_phase_evidence[0]
    retry_record = report.get("kimi_retry")
    has_retry_phase = any(
        row["phase"] == cascade.PHASE_KIMI_RETRY for row in strict_phase_evidence
    )
    retry_eligible = initial["retry_eligible_task_ids"]
    if (
        not isinstance(retry_record, Mapping)
        or retry_record.get("eligible_tasks") != len(retry_eligible)
        or (
            has_retry_phase
            and (
                retry_record.get("budget_skipped_tasks") != 0
                or retry_record.get("budget_skipped_task_ids_sha256")
                != canonical_sha256([])
                or retry_record.get("skip_reason") is not None
            )
        )
        or (
            not has_retry_phase
            and retry_eligible
            and (
                retry_record.get("budget_skipped_tasks") != len(retry_eligible)
                or retry_record.get("budget_skipped_task_ids_sha256")
                != canonical_sha256(retry_eligible)
                or retry_record.get("skip_reason")
                != "exact_targeted_retry_set_does_not_fit_remaining_provider_cap"
            )
        )
        or (
            not retry_eligible
            and (
                retry_record.get("budget_skipped_tasks") != 0
                or retry_record.get("budget_skipped_task_ids_sha256")
                != canonical_sha256([])
                or retry_record.get("skip_reason") is not None
            )
        )
    ):
        raise ValueError("dual-provider Kimi retry decision evidence differs")

    if not 0 <= len(rows) <= MAX_DUAL_API_ROWS:
        raise ValueError("completed dual-provider harvest yielded too many rows")
    task_ids: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        code = str(row.get("dart_source") or "")
        if (
            row.get("schema") != cascade.DIRECT_TARGET_SCHEMA
            or not task_id
            or not code.strip()
            or row.get("origin") != "external_teacher_direct_verified"
            or row.get("visible_train_passed") is not True
            or row.get("private_full_acceptance_passed") is not True
            or row.get("stability_runs") != cascade.STABILITY_RUNS
            or row.get("reasoning_present") is not False
            or row.get("repair_conditioned_training_source_present") is not False
            or row.get("gold_replay") is not False
            or mixed.sha256_text(code) != row.get("dart_source_sha256")
        ):
            raise ValueError(
                "dual-provider aggregate contains a non-direct or unsafe row"
            )
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("dual-provider aggregate contains duplicate task IDs")

    audit = {
        "schema": "t5gemma2-typed-dual-api-pass2-input-audit-v1",
        "status": "complete",
        "report": {"path": str(report_path), "sha256": report_sha256},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "targets": {
            "path": str(targets_path),
            "sha256": targets_sha256,
            "rows": len(rows),
        },
        "phase_reports": [
            {
                key: row[key]
                for key in (
                    "phase",
                    "cohort_index",
                    "report_sha256",
                    "journal_sha256",
                    "targets_sha256",
                )
            }
            for row in audited_phases
        ],
        "phase_evidence": strict_phase_evidence,
        "task_ids_sha256": canonical_sha256(task_ids),
        "row_count_late_bound_after_complete_audit": True,
        "direct_code_only": True,
        "targets_bound_to_paid_call_results_and_private_gates": True,
        "gold_source_replay": False,
        "heldout_175_model_visible": False,
        "heldout_175_used_for_generation_or_selection": False,
    }
    return rows, audit


def validate_pass2_warmstart(
    checkpoint: Path,
    *,
    expected_update: int,
    expected_run_contract_sha256: str,
    expected_adapter_weights_sha256: str,
    expected_adapter_config_sha256: str,
    model: str,
    model_revision: str,
) -> tuple[mixed.WarmstartIdentity, dict[str, Any]]:
    """Reuse the mixed loader under the exact pass-1 schema it must validate."""

    current_run = mixed.RUN_SCHEMA
    current_checkpoint = mixed.CHECKPOINT_SCHEMA
    current_dataset = mixed.DATASET_SCHEMA
    mixed.RUN_SCHEMA = pass1.RUN_SCHEMA
    mixed.CHECKPOINT_SCHEMA = pass1.CHECKPOINT_SCHEMA
    mixed.DATASET_SCHEMA = pass1.DATASET_SCHEMA
    try:
        identity, contract = _MIXED_VALIDATE_WARMSTART(
            checkpoint,
            expected_update=expected_update,
            expected_run_contract_sha256=expected_run_contract_sha256,
            expected_adapter_weights_sha256=expected_adapter_weights_sha256,
            expected_adapter_config_sha256=expected_adapter_config_sha256,
            model=model,
            model_revision=model_revision,
        )
    finally:
        mixed.RUN_SCHEMA = current_run
        mixed.CHECKPOINT_SCHEMA = current_checkpoint
        mixed.DATASET_SCHEMA = current_dataset
    if contract.get("schema") != pass1.RUN_SCHEMA or identity.update != 58:
        raise ValueError("pass-2 requires the sealed typed-direct update58 parent")
    return identity, contract


def build_typed_direct_pass2_pairs(
    *,
    gold_train_jsonl: Path,
    gold_f2_jsonl: Path,
    expected_gold_train_sha256: str,
    expected_gold_f2_sha256: str,
    expected_gold_rows: int,
    heldout_jsonl: Path,
    expected_heldout_sha256: str,
    expected_heldout_rows: int,
    local_reports: Sequence[tuple[Path, str]],
    api_reports: Sequence[tuple[Path, str]],
    warmstart: mixed.WarmstartIdentity,
    gold_replay_ratio: float,
    gold_replay_rows: int,
    min_verified_direct_targets: int,
    min_repair_conditioned_targets: int,
    allow_exploratory_inputs: bool,
    require_local_production_floor: bool,
    seed: int,
) -> tuple[list[mixed.MixedPair], dict[str, Any]]:
    del warmstart
    if (
        gold_replay_ratio != 0.0
        or gold_replay_rows != 0
        or min_verified_direct_targets != EXPECTED_LOCAL_NEW_ROWS
        or min_repair_conditioned_targets != 0
        or allow_exploratory_inputs
        or require_local_production_floor
    ):
        raise ValueError("pass-2 accepts only its sealed direct-only no-replay profile")
    local_specs = _require_specs(
        local_reports, LOCAL_SPEC_BASENAMES, "pass-2 local input"
    )
    api_specs = _require_specs(api_reports, API_SPEC_BASENAMES, "pass-2 API input")
    if len({path.parent for path, _digest in local_specs[:3]}) != 1:
        raise ValueError("local harvest artifacts are not siblings")

    tasks, gates, _scheduled, _terminals, input_record, local_source_record = (
        local_harvest.load_completed_harvest_artifacts(
            report_path=local_specs[0][0],
            expected_report_sha256=local_specs[0][1],
            journal_path=local_specs[1][0],
            expected_journal_sha256=local_specs[1][1],
            targets_path=local_specs[2][0],
            expected_targets_sha256=local_specs[2][1],
            gold_train_jsonl=gold_train_jsonl,
            expected_gold_train_sha256=expected_gold_train_sha256,
            gold_f2_jsonl=gold_f2_jsonl,
            expected_gold_f2_sha256=expected_gold_f2_sha256,
            heldout_jsonl=heldout_jsonl,
            expected_heldout_sha256=expected_heldout_sha256,
            expected_gold_rows=expected_gold_rows,
            expected_heldout_rows=expected_heldout_rows,
        )
    )
    local_rows = _read_jsonl(local_specs[2][0], "typed local direct targets")
    if len(local_rows) != EXPECTED_LOCAL_NEW_ROWS:
        raise ValueError(
            "pass-2 requires the exact independently audited 190 local targets"
        )
    prior_ids, prior_record = _load_prior_225_manifest(*local_specs[3])
    api_rows, api_audit = audit_completed_dual_orchestration(
        report_path=api_specs[0][0],
        report_sha256=api_specs[0][1],
        manifest_path=api_specs[1][0],
        manifest_sha256=api_specs[1][1],
        targets_path=api_specs[2][0],
        targets_sha256=api_specs[2][1],
        expected_input_record=input_record,
        expected_source_journal_record=local_source_record,
    )

    typed_by_id = {task.task_id: task for task in tasks}
    if len(typed_by_id) != len(tasks):
        raise ValueError("clean typed TRAIN universe contains duplicate task IDs")
    heldout_ids, heldout_record = mixed._load_heldout_ids(  # noqa: SLF001
        heldout_jsonl,
        expected_sha256=expected_heldout_sha256,
        expected_rows=expected_heldout_rows,
    )

    collected = [("local_student_new", row) for row in local_rows] + [
        ("external_teacher_new", row) for row in api_rows
    ]
    task_ids = [str(row.get("task_id") or "") for _category, row in collected]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(
            "pass-2 inputs overlap by task_id; no source may replace another"
        )
    if set(task_ids) & prior_ids:
        raise ValueError(
            "pass-2 attempted to replay an already-consumed prior-225 task"
        )
    if set(task_ids) & heldout_ids or pass1.CONTAMINATED_TRAIN_TASK_ID in task_ids:
        raise ValueError("held-out or known contaminated task entered pass-2")
    unknown = sorted(set(task_ids) - set(typed_by_id))
    if unknown:
        raise ValueError("pass-2 target is outside clean typed TRAIN: " + unknown[0])

    pairs: list[mixed.MixedPair] = []
    category_by_id: dict[str, str] = {}
    for category, row in collected:
        task_id = str(row["task_id"])
        target = str(row["dart_source"])
        task = typed_by_id[task_id]
        if category == "local_student_new":
            if (
                row.get("schema") != local_harvest.TARGET_SCHEMA
                or row.get("origin") != "local_student_direct"
                or row.get("full_acceptance_passed") is not True
                or row.get("stability_runs") != local_harvest.EXPECTED_STABILITY_RUNS
                or row.get("repair_conditioned") is not False
                or row.get("gold_replay") is not False
                or row.get("source_sha256") != task.source_sha256
                or row.get("dart_source_sha256") != mixed.sha256_text(target)
            ):
                raise ValueError("local pass-2 target safety binding differs")
            source_digest = local_specs[0][1]
        else:
            if row.get("source_sha256") != task.source_sha256:
                raise ValueError(
                    "API pass-2 target is bound to a different typed source"
                )
            source_digest = api_specs[0][1]
        category_by_id[task_id] = category
        pairs.append(
            mixed._make_pair(  # noqa: SLF001
                pair_id=f"{task_id}::typed-direct-pass2::{category}",
                source_task_id=task_id,
                kind="verified_direct",
                source=task.source,
                target=target,
                provenance=(
                    ("dataset_schema", DATASET_SCHEMA),
                    ("source_category", category),
                    ("source_report_sha256", source_digest),
                    ("typed_source_sha256", task.source_sha256),
                ),
            )
        )

    verification = pass1._verify_all(  # noqa: SLF001
        pairs,
        tests_by_id={task_id: gate.tests for task_id, gate in gates.items()},
        verify=pass1._runtime_verify,  # noqa: SLF001
        workers=pass1.FULL_VERIFY_WORKERS,
    )
    pairs.sort(
        key=lambda pair: canonical_sha256(
            {
                "schema": DATASET_SCHEMA,
                "seed": seed,
                "pair_id": pair.pair_id,
                "source_sha256": pair.source_sha256,
                "target_sha256": pair.target_sha256,
            }
        )
    )
    exact_gold = sum(
        pair.target_sha256 == typed_by_id[pair.source_task_id].gold_target_sha256
        for pair in pairs
    )
    schedule = [
        {
            "position": position,
            "pair_id": pair.pair_id,
            "source_task_id": pair.source_task_id,
            "kind": "verified_direct",
            "source_category": category_by_id[pair.source_task_id],
            "source_sha256": pair.source_sha256,
            "target_sha256": pair.target_sha256,
            "provenance": dict(pair.provenance),
        }
        for position, pair in enumerate(pairs)
    ]
    manifest = {
        "schema": DATASET_SCHEMA,
        "rows": len(pairs),
        "architecture": "native_encoder_decoder",
        "composition": {
            "verified_direct": len(pairs),
            "local_student_new": len(local_rows),
            "external_teacher_new": len(api_rows),
            "prior_225_replay": 0,
            "gold_replay": 0,
            "repair_conditioned": 0,
            "reasoning_rows": 0,
            "independently_generated_exact_gold_matches": exact_gold,
            "gold_source_replay": 0,
        },
        "local_harvest": {
            "report_sha256": local_specs[0][1],
            "journal_sha256": local_specs[1][1],
            "targets_sha256": local_specs[2][1],
            "rows": len(local_rows),
            "source_audit": local_source_record,
        },
        "dual_api_harvest": api_audit,
        "api_row_count_policy": "late_bound_after_audited_complete_orchestration",
        "typed_train_input": input_record,
        "prior_225_exclusion": prior_record,
        "heldout_identity_audit": heldout_record,
        "heldout_overlap": 0,
        "known_contaminant_excluded": pass1.CONTAMINATED_TRAIN_TASK_ID,
        "task_id_deduplication": "reject_any_duplicate_or_cross-source_overlap",
        "exact_gold_match_policy": (
            "retain_only_when_cryptographically_bound_to_independent_"
            "journal_candidate_and_private_verification"
        ),
        "all_targets_bound_to_generation_journals": True,
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "tests_model_visible": False,
        "private_feedback_model_visible": False,
        "repair_conditioned_prefixes_visible": False,
        "reasoning_model_visible": False,
        "full_acceptance_reverification": verification,
        "schedule": schedule,
        "schedule_sha256": canonical_sha256(schedule),
        "task_ids_sha256": canonical_sha256([pair.source_task_id for pair in pairs]),
        "source_sha256s_sha256": canonical_sha256(
            [pair.source_sha256 for pair in pairs]
        ),
        "target_sha256s_sha256": canonical_sha256(
            [pair.target_sha256 for pair in pairs]
        ),
        "production_floor_eligible": True,
    }
    return pairs, manifest


def _profile_runtime_contract() -> dict[str, str]:
    record = dict(_MIXED_RUNTIME_CONTRACT())
    record["mixed_training_engine_sha256"] = record["trainer_sha256"]
    record["trainer_sha256"] = base_sft.sha256_file(Path(__file__).resolve())
    record["pass1_profile_sha256"] = base_sft.sha256_file(
        Path(pass1.__file__).resolve()
    )
    record["local_harvest_validator_sha256"] = base_sft.sha256_file(
        Path(local_harvest.__file__).resolve()
    )
    record["dual_orchestrator_validator_sha256"] = base_sft.sha256_file(
        Path(dual.__file__).resolve()
    )
    record["api_cascade_validator_sha256"] = base_sft.sha256_file(
        Path(cascade.__file__).resolve()
    )
    record["trainer_profile"] = "typed_direct_only_rs_sft_pass2_local190_plus_dual_api"
    return record


def _validate_profile_args(args: argparse.Namespace) -> None:
    expected = {
        "gold_replay_ratio": 0.0,
        "gold_replay_rows": 0,
        "min_verified_direct_targets": EXPECTED_LOCAL_NEW_ROWS,
        "min_repair_conditioned_targets": 0,
        "expected_warmstart_update": 58,
        "epochs": 2,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_updates": 0,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.0,
        "seed": 42,
    }
    for name, wanted in expected.items():
        observed = getattr(args, name)
        matches = (
            math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float)
            else observed == wanted
        )
        if not matches:
            raise ValueError(
                f"typed direct pass-2 fixes --{name}={wanted}, observed={observed}"
            )
    if args.allow_exploratory_inputs or args.require_local_production_floor:
        raise ValueError("typed direct pass-2 requires sealed aggregate inputs")
    if len(args.local_report) != len(LOCAL_SPEC_BASENAMES) or len(
        args.api_report
    ) != len(API_SPEC_BASENAMES):
        raise ValueError(
            "typed direct pass-2 requires 4 local and 3 dual-API pinned artifacts"
        )


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_profile_args(args)
    validate_dart_binary()
    originals = {
        "run_schema": mixed.RUN_SCHEMA,
        "checkpoint_schema": mixed.CHECKPOINT_SCHEMA,
        "dataset_schema": mixed.DATASET_SCHEMA,
        "builder": mixed.build_mixed_pairs,
        "warmstart": mixed.validate_warmstart,
        "runtime": mixed._runtime_contract,  # noqa: SLF001
    }
    mixed.RUN_SCHEMA = RUN_SCHEMA
    mixed.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    mixed.DATASET_SCHEMA = DATASET_SCHEMA
    mixed.build_mixed_pairs = build_typed_direct_pass2_pairs
    mixed.validate_warmstart = validate_pass2_warmstart
    mixed._runtime_contract = _profile_runtime_contract  # noqa: SLF001
    try:
        return mixed.train(args)
    finally:
        mixed.RUN_SCHEMA = originals["run_schema"]
        mixed.CHECKPOINT_SCHEMA = originals["checkpoint_schema"]
        mixed.DATASET_SCHEMA = originals["dataset_schema"]
        mixed.build_mixed_pairs = originals["builder"]
        mixed.validate_warmstart = originals["warmstart"]
        mixed._runtime_contract = originals["runtime"]  # noqa: SLF001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = mixed.parse_args(argv)
    try:
        _validate_profile_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
