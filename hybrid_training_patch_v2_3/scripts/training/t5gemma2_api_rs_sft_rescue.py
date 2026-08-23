#!/usr/bin/env python3
"""Use a small, capped frontier budget to rescue TRAIN-only RS-SFT failures.

The normal input is the completed, hash-chained journal produced by
``t5gemma2_local_rs_sft_pilot.py``.  A separately marked exploratory mode can
freeze an explicit exact terminal prefix from an incomplete journal; those
artifacts can never count toward the production target floor.  Only tasks
whose complete local rollout group has exactly zero visible passes are
eligible.  For each eligible task, this program deterministically selects
diverse failed parents, preferring non-compiling parents with sanitized
compiler diagnostics.

Every provider receives only:

* the original test-free F2 encoder source;
* one failed student program; and
* a sanitized compiler diagnostic (or an explicit no-diagnostic sentinel);
* the visible TRAIN-only checks, for semantic diagnosis.

The private train holdback, gold Dart, and held-out evaluation data are never
sent to a provider.  Visible tests are provider-only context and are never
copied into either training view.  The full API schedule is fixed before the
first call and all calls finish before any private holdback result is opened.
Provider output is accepted only when it is code-only, passes the visible
training checks, and then passes the complementary binary private holdback.

Two distinct training artifacts are emitted:

1. original-F2 -> verified code direct hard targets; and
2. exact F2+failed-code+diagnostic encoder source -> the same verified code,
   for repair-policy SFT before compiler-feedback VeRPO.

The transport supports native Anthropic Messages, OpenAI Responses/chat,
OpenRouter chat, Azure OpenAI's current ``.../openai/v1``
standard-OpenAI-client endpoint, and the older dated Azure SDK endpoint as
explicitly separate modes.  Calls, input tokens, output tokens, total tokens,
and estimated list dollars are all fail-closed by worst-case reservations.
The journal is append-only and hash-chained.  Credentials are read only from an
environment variable and are never serialized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

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
from scripts.training.seq2seq_verpo_core import (
    build_compiler_repair_context,
    max_min_diverse_indices,
    sanitize_compiler_diagnostic,
    sha256_text,
)
from scripts.training.t5gemma2_local_rs_sft_pilot import (
    JOURNAL_SCHEMA as LOCAL_JOURNAL_SCHEMA,
    RUN_SCHEMA as LOCAL_RUN_SCHEMA,
    Evaluation,
    PilotTask,
    PrivateGate,
    deterministic_pilot_indices,
    load_pilot_inputs,
    validate_journal_state as validate_local_journal_state,
)


RUN_SCHEMA = "t5gemma2-api-rs-sft-rescue-v1"
JOURNAL_SCHEMA = "t5gemma2-api-rs-sft-rescue-journal-v1"
DIRECT_TARGET_SCHEMA = "t5gemma2-api-rs-sft-direct-target-v1"
REPAIR_PAIR_SCHEMA = "t5gemma2-api-rs-sft-repair-policy-pair-v1"
DIRECT_MANIFEST_SCHEMA = "t5gemma2-api-rs-sft-direct-manifest-v1"
REPAIR_MANIFEST_SCHEMA = "t5gemma2-api-rs-sft-repair-policy-manifest-v1"
REPORT_SCHEMA = "t5gemma2-api-rs-sft-rescue-report-v1"
PRIOR_SUCCESS_SOURCE_SCHEMA = "t5gemma2-api-rs-sft-prior-success-source-v1"
PRIOR_SUCCESS_EXCLUSION_SCHEMA = "t5gemma2-api-rs-sft-prior-success-exclusion-v1"
RETRY_PARSE_FAILURES_OR_TRUNCATIONS_SOURCE_SCHEMA = (
    "t5gemma2-api-rs-sft-retry-parse-failures-or-truncations-source-v1"
)
COMPLETED_LOCAL_RUN_MODE = "completed_local_pilot"
EXPLORATORY_PREFIX_RUN_MODE = "exploratory_terminal_prefix"
COMPILED_NO_DIAGNOSTIC = "candidate_compiled_but_no_compiler_diagnostic_is_available"
MISSING_SAFE_DIAGNOSTIC = "compiler_failed_without_safe_diagnostic"
SYSTEM_PROMPT = """\
You repair one Dart function from an isolated synthetic programming benchmark
and visible TRAIN-only checks. The benchmark artifacts are user-supplied and
authorized for model-training research; this task does not access, exploit, or
modify any external system.
Return exactly one complete Dart source program and nothing else.
Do not explain your reasoning. Do not emit JSON. You may use one ```dart
code fence, but no prose may appear before or after it. Preserve the requested
public function surface and use the failed program and compiler diagnostic as
repair evidence. Visible checks may be used for diagnosis, but you are not
given any private holdback or reference code.
"""
_SECRET_LIKE_RE = re.compile(r"(?i)(?:sk|api[_-]?key|bearer)[-_ .]?[a-z0-9+/=_-]{20,}")
_FENCED_CODE_RE = re.compile(
    r"\A```(?:dart)?[ \t]*\r?\n(?P<code>[\s\S]*?)\r?\n```[ \t]*\Z",
    re.IGNORECASE,
)
_PROSE_PREFIX_RE = re.compile(
    r"(?i)\A(?:here(?:'s| is)|sure[,!:]|the (?:fixed|repaired) code|analysis:)"
)
_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "x-api-key",
        "credential",
        "credentials",
    }
)
_PRIOR_OUTPUT_FILES = {
    "direct_targets": "direct_hard_targets.jsonl",
    "direct_f2": "direct_hard_targets_f2.jsonl",
    "repair_targets": "repair_policy_targets.jsonl",
    "repair_sources": "repair_policy_sources.jsonl",
}
@dataclass(frozen=True)
class RescueParent:
    task_id: str
    parent_index: int
    code: str
    code_sha256: str
    compiled: bool
    diagnostic: str
    diagnostic_sha256: str
    origin: str
    feedback_source: str
    feedback_source_sha256: str


@dataclass(frozen=True)
class RescuePlan:
    task_position: int
    task: PilotTask
    gate: PrivateGate
    local_terminal_sha256: str
    parents: tuple[RescueParent, ...]


@dataclass(frozen=True)
class ApiSlot:
    slot_position: int
    task_position: int
    task_id: str
    parent_position: int
    sample_index: int
    parent: RescueParent
    prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    response_id: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class PriorSuccessExclusions:
    """Verified task identities sealed by completed earlier rescue runs."""

    scheduled_task_ids: frozenset[str]
    verified_task_ids: frozenset[str]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RetryParseFailuresOrTruncationsSource:
    """Exact parse-failed/truncated cohort from a completed production rescue."""

    plans: tuple[RescuePlan, ...]
    record: dict[str, Any]


class ProviderTransport(Protocol):
    def create(
        self, *, system: str, user: str, max_output_tokens: int
    ) -> ProviderResponse: ...


EvaluateFn = Callable[[str, str, str], Evaluation]


class PendingProviderCall(RuntimeError):
    """A crash left an intent whose billing/result cannot be inferred safely."""


class RetryableProviderPayloadError(RuntimeError):
    """A provider returned a transient, HTTP-success-shaped unusable payload."""

    status_code = 503


def _field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
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


def _exact_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = _canonical_jsonl_bytes(rows)
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


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} must be an exact lowercase SHA-256")
    return digest


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} is absent: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(
                        f"{label} contains a blank row at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{label} row {line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSONL: {path}") from exc
    return rows


def _declared_basename(value: Any) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def _prior_sibling(
    report_path: Path, declared_path: Any, expected_name: str, label: str
) -> Path:
    if _declared_basename(declared_path) != expected_name:
        raise ValueError(f"{label} declared path is not {expected_name}")
    sibling = report_path.parent / expected_name
    if not sibling.is_file():
        raise ValueError(f"{label} sibling is absent: {sibling}")
    return sibling


def _validate_prior_journal_structure(
    events: Sequence[Mapping[str, Any]],
    *,
    report: Mapping[str, Any],
    input_record: Mapping[str, Any],
    source_journal_record: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]], Mapping[str, Any]]:
    """Validate a completed prior rescue without trusting its summary report."""

    if not events:
        raise ValueError("prior rescue journal is empty")
    header = events[0]
    contract = header.get("contract")
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or not isinstance(contract, Mapping)
        or contract.get("schema") != RUN_SCHEMA
        or header.get("contract_sha256") != canonical_sha256(contract)
        or header.get("contract_sha256") != report.get("run_contract_sha256")
        or contract.get("inputs") != input_record
        or contract.get("source_local_pilot_journal") != source_journal_record
        or contract.get("heldout_175_opened") is not False
    ):
        raise ValueError(
            "prior rescue journal header differs from the current sealed inputs"
        )
    local_source = contract.get("local_source")
    privacy = contract.get("privacy")
    training_outputs = contract.get("training_outputs")
    if (
        not isinstance(local_source, Mapping)
        or local_source.get("mode") != COMPLETED_LOCAL_RUN_MODE
        or local_source.get("exploratory_prefix") is not False
        or local_source.get("production_floor_eligible") is not True
        or not isinstance(privacy, Mapping)
        or privacy.get("private_holdback_sent_to_provider") is not False
        or privacy.get("gold_sent_to_provider") is not False
        or not isinstance(training_outputs, Mapping)
        or training_outputs.get("production_floor_eligible") is not True
    ):
        raise ValueError("prior rescue is not a private production-eligible run")
    selection = contract.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("prior rescue selection contract is absent")
    scheduled_tasks = selection.get("scheduled_tasks")
    scheduled_slots = selection.get("scheduled_slots")
    if (
        type(scheduled_tasks) is not int
        or scheduled_tasks <= 0
        or type(scheduled_slots) is not int
        or scheduled_slots <= 0
    ):
        raise ValueError("prior rescue scheduled counts are malformed")

    cursor = 1
    slot_task_ids: list[str] = []
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
    for slot_position in range(scheduled_slots):
        if cursor + 1 >= len(events):
            raise ValueError("prior rescue journal has an incomplete API phase")
        intent = events[cursor]
        result = events[cursor + 1]
        if (
            intent.get("event") != "call_intent"
            or intent.get("schema") != JOURNAL_SCHEMA
            or intent.get("slot_position") != slot_position
            or result.get("event") != "call_result"
            or result.get("schema") != JOURNAL_SCHEMA
            or any(result.get(key) != intent.get(key) for key in binding_keys)
        ):
            raise ValueError("prior rescue API event order/binding differs")
        task_id = str(intent.get("task_id") or "")
        if not task_id:
            raise ValueError("prior rescue API slot has an empty task identity")
        if result.get("status") not in {
            "response",
            "provider_error",
            "contract_error",
        }:
            raise ValueError("prior rescue API result status is invalid")
        usage = result.get("usage")
        if not isinstance(usage, Mapping) or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in (
                "charged_input_tokens",
                "charged_output_tokens",
                "charged_usd_nanos",
            )
        ):
            raise ValueError("prior rescue API charge is malformed")
        slot_task_ids.append(task_id)
        cursor += 2

    scheduled_task_ids: list[str] = []
    verifications: list[dict[str, Any]] = []
    for task_position in range(scheduled_tasks):
        if cursor >= len(events):
            raise ValueError("prior rescue journal has an incomplete private phase")
        event = events[cursor]
        task_id = str(event.get("task_id") or "")
        if (
            event.get("event") != "task_verification"
            or event.get("schema") != JOURNAL_SCHEMA
            or event.get("task_position") != task_position
            or not task_id
            or event.get("all_api_generation_completed_before_private_gate") is not True
            or event.get("private_feedback_serialized_to_model") is not False
            or event.get("holdback_failure_triggers_generation") is not False
        ):
            raise ValueError("prior rescue verification order/binding differs")
        selected = event.get("selected_target")
        if selected is not None and (
            not isinstance(selected, Mapping)
            or selected.get("schema") != DIRECT_TARGET_SCHEMA
            or selected.get("task_id") != task_id
            or selected.get("visible_passed") is not True
            or selected.get("private_gate_passed") is not True
            or selected.get("exploratory_prefix") is not False
            or selected.get("production_floor_eligible") is not True
            or sha256_text(str(selected.get("code") or ""))
            != selected.get("code_sha256")
        ):
            raise ValueError("prior rescue selected target is not verified")
        scheduled_task_ids.append(task_id)
        verifications.append(dict(event))
        cursor += 1

    if cursor >= len(events):
        raise ValueError("prior rescue journal has no completion event")
    complete = events[cursor]
    verified_count = sum(
        row.get("selected_target") is not None for row in verifications
    )
    if (
        complete.get("event") != "complete"
        or complete.get("schema") != JOURNAL_SCHEMA
        or complete.get("tasks") != scheduled_tasks
        or complete.get("slots") != scheduled_slots
        or complete.get("verified_targets") != verified_count
        or complete.get("exploratory_prefix") is not False
        or complete.get("production_floor_eligible") is not True
        or cursor + 1 != len(events)
    ):
        raise ValueError("prior rescue completion event differs")
    if (
        len(scheduled_task_ids) != len(set(scheduled_task_ids))
        or set(slot_task_ids) != set(scheduled_task_ids)
        or canonical_sha256(scheduled_task_ids) != selection.get("task_ids_sha256")
    ):
        raise ValueError("prior rescue task schedule identity differs")

    report_schedule = report.get("schedule")
    report_verification = report.get("verification")
    report_budget = report.get("budget_charged")
    if (
        not isinstance(report_schedule, Mapping)
        or report_schedule.get("scheduled_tasks") != scheduled_tasks
        or report_schedule.get("scheduled_calls") != scheduled_slots
        or (
            report_schedule.get("task_ids_sha256") is not None
            and report_schedule.get("task_ids_sha256")
            != canonical_sha256(scheduled_task_ids)
        )
        or not isinstance(report_verification, Mapping)
        or report_verification.get("verified_unique_hard_targets") != verified_count
        or not isinstance(report_budget, Mapping)
        or report_budget.get("calls") != scheduled_slots
        or report_budget.get("within_contract") is not True
    ):
        raise ValueError("prior rescue report summary differs from its journal")
    return scheduled_task_ids, verifications, contract


def _validate_prior_output_record(
    *,
    report_path: Path,
    name: str,
    record: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"prior rescue output record {name} is absent")
    expected_name = _PRIOR_OUTPUT_FILES[name]
    path = _prior_sibling(
        report_path,
        record.get("path"),
        expected_name,
        f"prior rescue {name}",
    )
    declared_sha256 = _require_sha256(
        record.get("sha256"), f"prior rescue {name} digest"
    )
    rows = record.get("rows")
    if type(rows) is not int or rows < 0:
        raise ValueError(f"prior rescue {name} row count is malformed")
    if sha256_file(path) != declared_sha256:
        raise ValueError(f"prior rescue {name} digest differs")
    values = _read_jsonl_objects(path, f"prior rescue {name}")
    if len(values) != rows:
        raise ValueError(f"prior rescue {name} row count differs")
    return (
        {
            "path": str(path.resolve()),
            "sha256": declared_sha256,
            "rows": rows,
        },
        values,
    )


def _validate_prior_outputs(
    *,
    report_path: Path,
    report: Mapping[str, Any],
    verifications: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(_PRIOR_OUTPUT_FILES):
        raise ValueError("prior rescue output inventory differs")
    records: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in _PRIOR_OUTPUT_FILES:
        records[name], rows[name] = _validate_prior_output_record(
            report_path=report_path,
            name=name,
            record=outputs.get(name),
        )

    verified_rows = [
        row for row in verifications if row.get("selected_target") is not None
    ]
    verified_task_ids = [str(row["task_id"]) for row in verified_rows]
    direct_task_ids = [str(row.get("task_id") or "") for row in rows["direct_targets"]]
    if (
        not all(direct_task_ids)
        or direct_task_ids != verified_task_ids
        or [str(row.get("task_id") or "") for row in rows["direct_f2"]]
        != verified_task_ids
        or [str(row.get("source_task_id") or "") for row in rows["repair_targets"]]
        != verified_task_ids
        or [str(row.get("source_task_id") or "") for row in rows["repair_sources"]]
        != verified_task_ids
    ):
        raise ValueError("prior rescue output task identities differ from journal")
    for direct, verification in zip(rows["direct_targets"], verified_rows, strict=True):
        selected = verification["selected_target"]
        code = str(direct.get("dart_source") or "")
        if (
            direct.get("schema") != DIRECT_TARGET_SCHEMA
            or direct.get("visible_passed") is not True
            or direct.get("private_gate_passed") is not True
            or direct.get("exploratory_prefix") is not False
            or direct.get("production_floor_eligible") is not True
            or code != selected.get("code")
            or sha256_text(code) != direct.get("dart_source_sha256")
        ):
            raise ValueError("prior rescue direct target differs from verification")
    for targets, sources in zip(
        rows["repair_targets"], rows["repair_sources"], strict=True
    ):
        if (
            targets.get("task_id") != sources.get("task_id")
            or targets.get("dart_source_sha256")
            != next(
                row["dart_source_sha256"]
                for row in rows["direct_targets"]
                if row["task_id"] == targets["source_task_id"]
            )
            or targets.get("production_floor_eligible") is not True
            or sources.get("production_floor_eligible") is not True
            or sources.get("private_feedback_present") is not False
            or sources.get("tests_present") is not False
            or sources.get("gold_present") is not False
        ):
            raise ValueError("prior rescue repair output binding differs")

    manifest_records: dict[str, Any] = {}
    for report_key, filename in (
        ("direct_manifest", "direct_manifest.json"),
        ("repair_policy_manifest", "repair_policy_manifest.json"),
    ):
        declared = report.get(report_key)
        if not isinstance(declared, Mapping):
            raise ValueError(f"prior rescue {report_key} is absent")
        manifest_path = report_path.parent / filename
        observed = _read_json_object(manifest_path, f"prior rescue {report_key}")
        if observed != declared:
            raise ValueError(f"prior rescue {report_key} differs from report")
        manifest_records[report_key] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        }
    if (
        report["direct_manifest"].get("targets") != outputs["direct_targets"]
        or report["direct_manifest"].get("f2") != outputs["direct_f2"]
        or report["repair_policy_manifest"].get("targets") != outputs["repair_targets"]
        or report["repair_policy_manifest"].get("prebuilt_encoder_sources")
        != outputs["repair_sources"]
    ):
        raise ValueError("prior rescue manifests differ from output inventory")
    return {**records, **manifest_records}, verified_task_ids


def load_prior_success_exclusions(
    *,
    report_paths: Sequence[str | Path],
    expected_report_sha256s: Sequence[str],
    current_eligible_task_ids: Sequence[str],
    input_record: Mapping[str, Any],
    source_journal_record: Mapping[str, Any],
    require_disjoint_schedules: bool = False,
    require_complete_coverage: bool = False,
) -> PriorSuccessExclusions:
    """Load exact prior reports and exclude only privately verified successes.

    Each expected report digest pins the report's embedded journal and output
    digests.  The actual sibling journal, chain head, four JSONL outputs, and
    two manifests are then independently opened and validated before any task
    identity can enter the exclusion set.
    """

    if len(report_paths) != len(expected_report_sha256s):
        raise ValueError(
            "each --prior_success_report requires one matching expected digest"
        )
    if not report_paths:
        if require_disjoint_schedules or require_complete_coverage:
            raise ValueError("prior schedule requirements need prior reports")
        return PriorSuccessExclusions(frozenset(), frozenset(), ())
    eligible_order = [str(task_id) for task_id in current_eligible_task_ids]
    if (
        not eligible_order
        or any(not task_id for task_id in eligible_order)
        or len(eligible_order) != len(set(eligible_order))
    ):
        raise ValueError("current eligible task identities are not unique")
    eligible_set = set(eligible_order)
    resolved_paths = [Path(path).expanduser().resolve() for path in report_paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("prior success report paths must be unique")

    schedule_sets: list[set[str]] = []
    verified_set: set[str] = set()
    records: list[dict[str, Any]] = []
    for report_index, (report_path, expected_sha_value) in enumerate(
        zip(resolved_paths, expected_report_sha256s, strict=True)
    ):
        expected_sha256 = _require_sha256(
            expected_sha_value,
            f"prior success report {report_index} expected digest",
        )
        if not report_path.is_file() or sha256_file(report_path) != expected_sha256:
            raise ValueError(f"prior success report {report_index} digest differs")
        report = _read_json_object(report_path, f"prior success report {report_index}")
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("exploratory_prefix") is not False
            or report.get("production_floor_eligible") is not True
            or report.get("may_count_toward_production_min_unique_targets") is not True
            or report.get("heldout_175_opened") is not False
        ):
            raise ValueError(
                f"prior success report {report_index} is not completed production"
            )
        journal = report.get("journal")
        if not isinstance(journal, Mapping):
            raise ValueError(
                f"prior success report {report_index} has no journal record"
            )
        journal_path = _prior_sibling(
            report_path,
            journal.get("path"),
            "api_rescue.journal.jsonl",
            f"prior success report {report_index} journal",
        )
        chain_head_path = _prior_sibling(
            report_path,
            journal.get("chain_head_path"),
            "api_rescue.journal.jsonl.chain-head.json",
            f"prior success report {report_index} chain head",
        )
        actual_journal = journal_record(journal_path)
        for key in (
            "sha256",
            "chain_head_sha256",
            "event_count",
            "head_event_sha256",
        ):
            if actual_journal.get(key) != journal.get(key):
                raise ValueError(
                    f"prior success report {report_index} journal {key} differs"
                )
        if sha256_file(chain_head_path) != _require_sha256(
            journal.get("chain_head_sha256"),
            f"prior success report {report_index} chain-head digest",
        ):
            raise ValueError(
                f"prior success report {report_index} chain-head digest differs"
            )
        events = load_journal(journal_path)
        scheduled_ids, verifications, contract = _validate_prior_journal_structure(
            events,
            report=report,
            input_record=input_record,
            source_journal_record=source_journal_record,
        )
        output_records, verified_ids = _validate_prior_outputs(
            report_path=report_path,
            report=report,
            verifications=verifications,
        )
        scheduled = set(scheduled_ids)
        if not scheduled.issubset(eligible_set):
            raise ValueError(
                f"prior success report {report_index} scheduled a non-eligible task"
            )
        if not set(verified_ids).issubset(scheduled):
            raise ValueError(
                f"prior success report {report_index} verified an unscheduled task"
            )
        if require_disjoint_schedules and any(
            scheduled.intersection(previous) for previous in schedule_sets
        ):
            raise ValueError("prior rescue schedules are not disjoint")
        schedule_sets.append(scheduled)
        verified_set.update(verified_ids)
        records.append(
            {
                "schema": PRIOR_SUCCESS_SOURCE_SCHEMA,
                "report_index": report_index,
                "report_path": str(report_path),
                "report_sha256": expected_sha256,
                "run_contract_sha256": report["run_contract_sha256"],
                "provider": contract.get("provider"),
                "scheduled_tasks": len(scheduled_ids),
                "scheduled_task_ids_sha256": canonical_sha256(scheduled_ids),
                "verified_tasks": len(verified_ids),
                "verified_task_ids_sha256": canonical_sha256(verified_ids),
                "journal": {
                    "path": str(journal_path.resolve()),
                    "sha256": actual_journal["sha256"],
                    "chain_head_path": str(chain_head_path.resolve()),
                    "chain_head_sha256": actual_journal["chain_head_sha256"],
                    "event_count": actual_journal["event_count"],
                    "head_event_sha256": actual_journal["head_event_sha256"],
                },
                "outputs": output_records,
                "heldout_175_opened": False,
            }
        )
    scheduled_union = set().union(*schedule_sets)
    if require_complete_coverage and scheduled_union != eligible_set:
        raise ValueError(
            "prior rescue schedules do not exactly cover the eligible cohort"
        )
    return PriorSuccessExclusions(
        scheduled_task_ids=frozenset(scheduled_union),
        verified_task_ids=frozenset(verified_set),
        records=tuple(records),
    )


def exclude_prior_verified_plans(
    plans: Sequence[RescuePlan], verified_task_ids: Sequence[str] | set[str]
) -> list[RescuePlan]:
    excluded = set(verified_task_ids)
    available = {plan.task.task_id for plan in plans}
    if not excluded.issubset(available):
        raise ValueError("prior verified exclusion contains a non-eligible task")
    result: list[RescuePlan] = []
    for plan in plans:
        if plan.task.task_id in excluded:
            continue
        result.append(
            RescuePlan(
                task_position=len(result),
                task=plan.task,
                gate=plan.gate,
                local_terminal_sha256=plan.local_terminal_sha256,
                parents=plan.parents,
            )
        )
    return result


def slice_rescue_plans(
    plans: Sequence[RescuePlan], *, offset: int, max_tasks: int
) -> list[RescuePlan]:
    if (
        type(offset) is not int
        or offset < 0
        or type(max_tasks) is not int
        or max_tasks < 0
    ):
        raise ValueError("invalid residual task slice")
    if offset > len(plans):
        raise ValueError("eligible task offset exceeds the residual cohort")
    selected = list(plans[offset:])
    if max_tasks:
        selected = selected[:max_tasks]
    return [
        RescuePlan(
            task_position=position,
            task=plan.task,
            gate=plan.gate,
            local_terminal_sha256=plan.local_terminal_sha256,
            parents=plan.parents,
        )
        for position, plan in enumerate(selected)
    ]


def _safe_endpoint(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "provider base URL must be an http(s) URL without userinfo, query, "
            "or fragment"
        )
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("unencrypted non-local provider endpoints are forbidden")
    return value.rstrip("/")


def validate_provider_endpoint(
    *, provider: str, base_url: str, api_version: str
) -> str:
    """Bind each Azure transport name to its actual SDK/URL contract."""

    endpoint = _safe_endpoint(base_url)
    dated_azure = {"azure_responses", "azure_chat"}
    azure_v1 = {"azure_v1_responses", "azure_v1_chat"}
    if provider in dated_azure and not str(api_version).strip():
        raise ValueError(
            f"{provider} is the dated AzureOpenAI-client mode and requires "
            "--api_version"
        )
    if provider in azure_v1:
        if str(api_version).strip():
            raise ValueError(
                f"{provider} uses the standard OpenAI client at /openai/v1 "
                "and must not receive --api_version"
            )
        path = urlsplit(endpoint).path.rstrip("/").lower()
        if not path.endswith("/openai/v1"):
            raise ValueError(f"{provider} base URL must end in /openai/v1")
    if provider == "openrouter_chat":
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "openrouter.ai"
            or parsed.port is not None
            or parsed.path.rstrip("/") != "/api/v1"
        ):
            raise ValueError(
                "openrouter_chat base URL must be exactly "
                "https://openrouter.ai/api/v1"
            )
        if str(api_version).strip():
            raise ValueError("openrouter_chat must not receive --api_version")
    return endpoint


def _redact_text(value: str, api_key: str) -> str:
    text = str(value).replace("\x00", "")[:2000]
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return _SECRET_LIKE_RE.sub("[REDACTED]", text)


def _assert_secret_free(value: Any, *, api_key: str = "") -> None:
    if isinstance(value, Mapping):
        forbidden = {
            str(key).strip().lower()
            for key in value
            if str(key).strip().lower() in _FORBIDDEN_ARTIFACT_KEYS
        }
        if forbidden:
            raise ValueError(
                f"credential-bearing artifact fields are forbidden: {sorted(forbidden)}"
            )
        for child in value.values():
            _assert_secret_free(child, api_key=api_key)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _assert_secret_free(child, api_key=api_key)
    elif isinstance(value, str):
        if api_key and api_key in value:
            raise ValueError("API credential value would be serialized")
        if _SECRET_LIKE_RE.search(value):
            raise ValueError("secret-like value would be serialized")


def _append_safe(
    path: Path, event: Mapping[str, Any], *, api_key: str = ""
) -> dict[str, Any]:
    _assert_secret_free(event, api_key=api_key)
    return append_event(path, event)


def parse_code_only(text: str, *, api_key: str = "") -> tuple[str, str | None]:
    """Accept raw Dart or exactly one fenced Dart block, never prose+code."""

    if not isinstance(text, str):
        return "", "non_text_response"
    if api_key and api_key in text:
        return "", "response_contains_credential"
    if _SECRET_LIKE_RE.search(text):
        return "", "response_contains_secret_like_text"
    stripped = text.strip()
    if not stripped:
        return "", "empty_response"
    fenced = _FENCED_CODE_RE.fullmatch(stripped)
    if fenced is not None:
        code = fenced.group("code").strip()
        if not code or "```" in code:
            return "", "invalid_fenced_code"
        return code, None
    if "```" in stripped or _PROSE_PREFIX_RE.search(stripped):
        return "", "response_is_not_code_only"
    if "<analysis" in stripped.lower() or "<thinking" in stripped.lower():
        return "", "response_contains_reasoning_markup"
    return stripped, None


def _feedback_source(
    *, task: PilotTask, code: str, diagnostic: str, compiled: bool
) -> tuple[str, str]:
    safe = sanitize_compiler_diagnostic(diagnostic)
    if not compiled:
        context = build_compiler_repair_context(
            task_id=task.task_id,
            source_sha256=task.source_sha256,
            candidate=code,
            diagnostic=safe,
            compiled=False,
        )
        return task.source + "\n" + str(context["text"]), safe
    payload = {
        "schema": "t5gemma2-compiled-failure-repair-context-v1",
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "candidate": code,
        "candidate_sha256": sha256_text(code),
        "compiler_feedback": COMPILED_NO_DIAGNOSTIC,
        "compiler_feedback_sha256": sha256_text(COMPILED_NO_DIAGNOSTIC),
        "compiled": True,
        "tests_visible": False,
        "private_holdback_visible": False,
    }
    text = (
        "<COMPILER_REPAIR_CONTEXT_JSON>\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n</COMPILER_REPAIR_CONTEXT_JSON>"
    )
    return task.source + "\n" + text, COMPILED_NO_DIAGNOSTIC


def _all_terminal_candidates(
    terminal: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    groups: list[Any] = [terminal.get("base_candidates")]
    repairs = terminal.get("repair_groups")
    if not isinstance(repairs, list):
        raise ValueError("local terminal has malformed repair groups")
    for group in repairs:
        if not isinstance(group, Mapping):
            raise ValueError("local terminal has malformed repair group")
        groups.append(group.get("candidates"))
    candidates: list[Mapping[str, Any]] = []
    for group in groups:
        if not isinstance(group, list):
            raise ValueError("local terminal has malformed candidate group")
        for candidate in group:
            if not isinstance(candidate, Mapping):
                raise ValueError("local terminal candidate is malformed")
            candidates.append(candidate)
    return candidates


def _candidate_parent(
    task: PilotTask, candidate: Mapping[str, Any], parent_index: int
) -> RescueParent | None:
    visible = candidate.get("visible")
    if not isinstance(visible, Mapping) or visible.get("passed") is not False:
        return None
    compiled = visible.get("compiled")
    if type(compiled) is not bool:
        raise ValueError("local candidate compile status is malformed")
    code = str(candidate.get("code") or "").strip()
    digest = str(candidate.get("code_sha256") or "")
    if not code or sha256_text(code) != digest:
        return None
    if not compiled:
        declared = str(candidate.get("safe_compiler_feedback") or "")
        declared_sha = str(candidate.get("safe_compiler_feedback_sha256") or "")
        if declared and sha256_text(declared) != declared_sha:
            raise ValueError("local sanitized compiler diagnostic digest differs")
        diagnostic = (
            sanitize_compiler_diagnostic(declared)
            if declared
            else MISSING_SAFE_DIAGNOSTIC
        )
    else:
        diagnostic = COMPILED_NO_DIAGNOSTIC
    feedback_source, safe = _feedback_source(
        task=task,
        code=code,
        diagnostic=diagnostic,
        compiled=compiled,
    )
    return RescueParent(
        task_id=task.task_id,
        parent_index=parent_index,
        code=code,
        code_sha256=digest,
        compiled=compiled,
        diagnostic=safe,
        diagnostic_sha256=sha256_text(safe),
        origin=str(candidate.get("origin") or "unknown"),
        feedback_source=feedback_source,
        feedback_source_sha256=sha256_text(feedback_source),
    )


def _diverse_take(values: Sequence[RescueParent], count: int) -> list[RescueParent]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    indices = max_min_diverse_indices([value.code for value in values], count)
    return [values[index] for index in indices]


def select_rescue_plans(
    *,
    scheduled_tasks: Sequence[PilotTask],
    gates: Mapping[str, PrivateGate],
    terminals: Sequence[Mapping[str, Any]],
    seed: int,
    max_tasks: int,
    max_parents_per_task: int,
    eligible_task_offset: int = 0,
) -> list[RescuePlan]:
    """Select a deterministic all-zero schedule with non-compilers first.

    ``eligible_task_offset`` is applied only after the complete all-zero
    cohort has been deterministically ordered and before ``max_tasks``.  This
    permits independently journaled, non-overlapping production tranches
    without making prior provider outcomes part of later eligibility.
    """

    if len(scheduled_tasks) != len(terminals):
        raise ValueError("local task and terminal counts differ")
    if (
        type(seed) is not int
        or seed < 0
        or type(max_tasks) is not int
        or max_tasks < 0
        or type(max_parents_per_task) is not int
        or max_parents_per_task <= 0
        or type(eligible_task_offset) is not int
        or eligible_task_offset < 0
    ):
        raise ValueError("invalid rescue selection controls")
    eligible: list[tuple[int, PilotTask, Mapping[str, Any]]] = []
    for position, (task, terminal) in enumerate(
        zip(scheduled_tasks, terminals, strict=True)
    ):
        if terminal.get("task_id") != task.task_id:
            raise ValueError(
                f"local terminal {position} identity differs from its task"
            )
        visible_unique_passes = terminal.get("visible_unique_passes")
        if type(visible_unique_passes) is not int or visible_unique_passes < 0:
            raise ValueError(
                f"{task.task_id}: visible pass count is not a non-negative integer"
            )
        if visible_unique_passes == 0:
            eligible.append((position, task, terminal))
    eligible.sort(
        key=lambda item: canonical_sha256(
            {
                "schema": RUN_SCHEMA,
                "seed": seed,
                "task_id": item[1].task_id,
                "local_task_position": item[0],
            }
        )
    )
    if eligible_task_offset > len(eligible):
        raise ValueError(
            "eligible task offset exceeds the deterministic all-zero cohort"
        )
    eligible = eligible[eligible_task_offset:]
    if max_tasks:
        eligible = eligible[:max_tasks]

    plans: list[RescuePlan] = []
    for local_position, task, terminal in eligible:
        gate = gates.get(task.task_id)
        if gate is None:
            raise ValueError(f"{task.task_id}: private gate is missing")
        deduplicated: list[RescueParent] = []
        seen: set[str] = set()
        for candidate_index, candidate in enumerate(_all_terminal_candidates(terminal)):
            parent = _candidate_parent(task, candidate, candidate_index)
            if parent is None or parent.code_sha256 in seen:
                continue
            seen.add(parent.code_sha256)
            deduplicated.append(parent)
        informative = [
            value
            for value in deduplicated
            if not value.compiled and value.diagnostic != MISSING_SAFE_DIAGNOSTIC
        ]
        other_noncompiling = [
            value
            for value in deduplicated
            if not value.compiled and value not in informative
        ]
        compiled = [value for value in deduplicated if value.compiled]
        selected = _diverse_take(informative, max_parents_per_task)
        for pool in (other_noncompiling, compiled):
            remaining = max_parents_per_task - len(selected)
            if remaining <= 0:
                break
            selected.extend(_diverse_take(pool, remaining))
        if not selected:
            continue
        plans.append(
            RescuePlan(
                task_position=len(plans),
                task=task,
                gate=gate,
                local_terminal_sha256=str(terminal.get("journal_event_sha256") or ""),
                parents=tuple(selected),
            )
        )
    return plans


def build_provider_prompt(plan: RescuePlan, parent: RescueParent) -> str:
    prompt = """\
Repair the failed Dart reconstruction below.

<TEST_FREE_FEEDBACK_CONDITIONED_ENCODER_SOURCE>
{source}
</TEST_FREE_FEEDBACK_CONDITIONED_ENCODER_SOURCE>

<VISIBLE_TRAINING_CHECKS_PROVIDER_ONLY>
{visible_tests}
</VISIBLE_TRAINING_CHECKS_PROVIDER_ONLY>

Return only the complete repaired Dart source.
""".format(
        source=parent.feedback_source,
        visible_tests=plan.task.visible_tests,
    )
    for label, forbidden in (
        ("private holdback", plan.gate.tests),
        ("gold target", plan.task.gold_target),
    ):
        if forbidden.strip() and forbidden in prompt:
            raise ValueError(f"{plan.task.task_id}: {label} leaked to API prompt")
    return prompt


def build_slots(
    plans: Sequence[RescuePlan], *, samples_per_parent: int
) -> list[ApiSlot]:
    if samples_per_parent <= 0:
        raise ValueError("samples_per_parent must be positive")
    slots: list[ApiSlot] = []
    for task_position, plan in enumerate(plans):
        if plan.task_position != task_position:
            raise ValueError("rescue plan positions are not contiguous")
        for parent_position, parent in enumerate(plan.parents):
            prompt = build_provider_prompt(plan, parent)
            for sample_index in range(samples_per_parent):
                slots.append(
                    ApiSlot(
                        slot_position=len(slots),
                        task_position=task_position,
                        task_id=plan.task.task_id,
                        parent_position=parent_position,
                        sample_index=sample_index,
                        parent=parent,
                        prompt=prompt,
                        prompt_sha256=sha256_text(prompt),
                    )
                )
    return slots


def load_retry_parse_failures_or_truncations_source(
    *,
    report_path: str | Path,
    expected_report_sha256: str,
    current_eligible_plans: Sequence[RescuePlan],
    input_record: Mapping[str, Any],
    source_journal_record: Mapping[str, Any],
) -> RetryParseFailuresOrTruncationsSource:
    """Select parse-rejected or length-truncated responses from a sealed run.

    The report digest pins the sibling hash-chained journal and all published
    outputs.  The source run must have exactly one API slot per unique task.
    Its complete journal is revalidated against plans reconstructed from the
    current sealed local pilot, so a task, parent, diagnostic, or prompt drift
    cannot silently enter the continuation.
    """

    resolved_report = Path(report_path).expanduser().resolve()
    expected_sha256 = _require_sha256(
        expected_report_sha256, "retry non-code report expected digest"
    )
    if (
        not resolved_report.is_file()
        or sha256_file(resolved_report) != expected_sha256
    ):
        raise ValueError("retry non-code report digest differs")
    report = _read_json_object(resolved_report, "retry non-code report")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("execution_mode") != "rs_sft_rescue"
        or report.get("exploratory_prefix") is not False
        or report.get("evaluation_only") is not False
        or report.get("training_use_forbidden") is not False
        or report.get("production_floor_eligible") is not True
        or report.get("may_count_toward_production_min_unique_targets") is not True
        or report.get("heldout_175_opened") is not False
    ):
        raise ValueError(
            "retry non-code source is not a completed production rescue report"
        )
    journal = report.get("journal")
    if not isinstance(journal, Mapping):
        raise ValueError("retry non-code report has no journal record")
    journal_path = _prior_sibling(
        resolved_report,
        journal.get("path"),
        "api_rescue.journal.jsonl",
        "retry non-code journal",
    )
    chain_head_path = _prior_sibling(
        resolved_report,
        journal.get("chain_head_path"),
        "api_rescue.journal.jsonl.chain-head.json",
        "retry non-code chain head",
    )
    actual_journal = journal_record(journal_path)
    for key in (
        "sha256",
        "chain_head_sha256",
        "event_count",
        "head_event_sha256",
    ):
        if actual_journal.get(key) != journal.get(key):
            raise ValueError(f"retry non-code journal {key} differs")
    if sha256_file(chain_head_path) != _require_sha256(
        journal.get("chain_head_sha256"),
        "retry non-code chain-head digest",
    ):
        raise ValueError("retry non-code chain-head digest differs")

    events = load_journal(journal_path)
    scheduled_ids, verifications, source_contract = (
        _validate_prior_journal_structure(
            events,
            report=report,
            input_record=input_record,
            source_journal_record=source_journal_record,
        )
    )
    output_records, _verified_ids = _validate_prior_outputs(
        report_path=resolved_report,
        report=report,
        verifications=verifications,
    )
    selection = source_contract["selection"]
    scheduled_tasks = int(selection["scheduled_tasks"])
    scheduled_slots = int(selection["scheduled_slots"])
    if (
        scheduled_slots != scheduled_tasks
        or len(scheduled_ids) != scheduled_tasks
        or len(set(scheduled_ids)) != scheduled_tasks
    ):
        raise ValueError(
            "retry non-code source requires exactly one slot per unique task"
        )

    results = [dict(events[2 + 2 * index]) for index in range(scheduled_slots)]
    result_task_ids = [str(row.get("task_id") or "") for row in results]
    if result_task_ids != scheduled_ids:
        raise ValueError("retry non-code source slot order differs from task schedule")
    response_count = 0
    code_only_count = 0
    for row in results:
        parse_accepted = row.get("parse_accepted")
        if type(parse_accepted) is not bool:
            raise ValueError("retry non-code source parse status is malformed")
        code = str(row.get("code") or "")
        if parse_accepted != bool(code):
            raise ValueError("retry non-code source parse/code binding differs")
        if row.get("status") == "response":
            response_count += 1
            response = row.get("response")
            if (
                not isinstance(response, Mapping)
                or type(response.get("finish_reason")) is not str
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(response.get("raw_text_sha256") or "")
                )
            ):
                raise ValueError("retry non-code source response metadata is malformed")
            if parse_accepted:
                code_only_count += 1
            elif not str(row.get("parse_rejection") or ""):
                raise ValueError(
                    "retry non-code source rejected response has no parse reason"
                )
    report_schedule = report["schedule"]
    if (
        report_schedule.get("provider_responses") != response_count
        or report_schedule.get("code_only_responses") != code_only_count
    ):
        raise ValueError(
            "retry non-code report response counts differ from its journal"
        )

    eligible_by_id = {
        plan.task.task_id: plan for plan in current_eligible_plans
    }
    if len(eligible_by_id) != len(current_eligible_plans):
        raise ValueError("current eligible rescue task identities are not unique")
    reconstructed: list[RescuePlan] = []
    for task_position, (task_id, result) in enumerate(
        zip(scheduled_ids, results, strict=True)
    ):
        source_plan = eligible_by_id.get(task_id)
        if source_plan is None:
            raise ValueError(
                "retry non-code source scheduled a task outside the current "
                "all-zero eligible cohort"
            )
        matching_parents = []
        for parent in source_plan.parents:
            prompt = build_provider_prompt(source_plan, parent)
            if (
                parent.code_sha256 == result.get("parent_code_sha256")
                and parent.diagnostic_sha256 == result.get("diagnostic_sha256")
                and parent.feedback_source_sha256
                == result.get("feedback_source_sha256")
                and sha256_text(prompt) == result.get("prompt_sha256")
            ):
                matching_parents.append(parent)
        if len(matching_parents) != 1:
            raise ValueError(
                f"{task_id}: retry non-code source parent/prompt binding differs"
            )
        reconstructed.append(
            RescuePlan(
                task_position=task_position,
                task=source_plan.task,
                gate=source_plan.gate,
                local_terminal_sha256=source_plan.local_terminal_sha256,
                parents=(matching_parents[0],),
            )
        )
    source_slots = build_slots(reconstructed, samples_per_parent=1)
    source_state = validate_rescue_journal(
        events,
        contract=source_contract,
        plans=reconstructed,
        slots=source_slots,
    )
    if not source_state["complete"]:
        raise ValueError("retry non-code source journal is not complete")

    retry_plans: list[RescuePlan] = []
    qualifying_results: list[dict[str, Any]] = []
    finish_reason_counts: dict[str, int] = {}
    parse_failure_count = 0
    truncation_count = 0
    parse_failure_and_truncation_count = 0
    for plan, result in zip(reconstructed, results, strict=True):
        response = result.get("response")
        parse_failed = result.get("parse_accepted") is False
        truncated = (
            isinstance(response, Mapping)
            and response.get("finish_reason") == "length"
        )
        if result.get("status") != "response" or not (parse_failed or truncated):
            continue
        retry_plans.append(
            RescuePlan(
                task_position=len(retry_plans),
                task=plan.task,
                gate=plan.gate,
                local_terminal_sha256=plan.local_terminal_sha256,
                parents=plan.parents,
            )
        )
        qualifying_results.append(result)
        parse_failure_count += int(parse_failed)
        truncation_count += int(truncated)
        parse_failure_and_truncation_count += int(parse_failed and truncated)
        finish_reason = str(response["finish_reason"])
        finish_reason_counts[finish_reason] = (
            finish_reason_counts.get(finish_reason, 0) + 1
        )
    if not retry_plans:
        raise ValueError("retry non-code source contains no qualifying responses")
    retry_task_ids = [plan.task.task_id for plan in retry_plans]
    result_event_sha256s = [
        _require_sha256(
            row.get("journal_event_sha256"),
            "retry non-code result event digest",
        )
        for row in qualifying_results
    ]
    record = {
        "schema": RETRY_PARSE_FAILURES_OR_TRUNCATIONS_SOURCE_SCHEMA,
        "enabled": True,
        "report_path": str(resolved_report),
        "report_sha256": expected_sha256,
        "run_contract_sha256": report["run_contract_sha256"],
        "journal": {
            "path": str(journal_path.resolve()),
            "sha256": actual_journal["sha256"],
            "chain_head_path": str(chain_head_path.resolve()),
            "chain_head_sha256": actual_journal["chain_head_sha256"],
            "event_count": actual_journal["event_count"],
            "head_event_sha256": actual_journal["head_event_sha256"],
        },
        "outputs": output_records,
        "source_scheduled_tasks": scheduled_tasks,
        "source_scheduled_task_ids_sha256": canonical_sha256(scheduled_ids),
        "source_one_slot_per_unique_task": True,
        "predicate": (
            "status=response_and_"
            "(parse_accepted=false_or_finish_reason=length)"
        ),
        "retry_tasks": len(retry_plans),
        "retry_task_ids_sha256": canonical_sha256(retry_task_ids),
        "qualifying_result_event_count": len(qualifying_results),
        "qualifying_result_events_sha256": canonical_sha256(qualifying_results),
        "qualifying_result_event_sha256s_sha256": canonical_sha256(
            result_event_sha256s
        ),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "parse_failure_count": parse_failure_count,
        "truncation_count": truncation_count,
        "parse_failure_and_truncation_count": (
            parse_failure_and_truncation_count
        ),
        "source_schedule_order_preserved": True,
        "accepted_nontruncated_responses_regenerated": False,
        "heldout_175_opened": False,
    }
    return RetryParseFailuresOrTruncationsSource(tuple(retry_plans), record)


def _usd_nanos(
    input_tokens: int,
    output_tokens: int,
    *,
    input_usd_per_million: Decimal,
    output_usd_per_million: Decimal,
) -> int:
    value = (
        Decimal(input_tokens) * input_usd_per_million
        + Decimal(output_tokens) * output_usd_per_million
    ) * Decimal(1000)
    # (tokens * USD/1M) * 1e9 = tokens * USD/M * 1e3
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def schedule_capacity(
    *,
    max_calls: int,
    max_input_tokens_per_call: int,
    max_output_tokens_per_call: int,
    max_input_tokens_total: int,
    max_output_tokens_total: int,
    max_total_tokens: int,
    max_usd: Decimal,
    input_usd_per_million: Decimal,
    output_usd_per_million: Decimal,
) -> tuple[int, dict[str, Any]]:
    values = (
        max_calls,
        max_input_tokens_per_call,
        max_output_tokens_per_call,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("call and per-call token caps must be positive integers")
    if any(
        type(value) is not int or value < 0
        for value in (
            max_input_tokens_total,
            max_output_tokens_total,
            max_total_tokens,
        )
    ):
        raise ValueError("aggregate token caps must be non-negative integers")
    if max_usd <= 0 or input_usd_per_million <= 0 or output_usd_per_million <= 0:
        raise ValueError("strict dollar cap and provider prices must be positive")
    candidates = [max_calls]
    if max_input_tokens_total:
        candidates.append(max_input_tokens_total // max_input_tokens_per_call)
    if max_output_tokens_total:
        candidates.append(max_output_tokens_total // max_output_tokens_per_call)
    if max_total_tokens:
        candidates.append(
            max_total_tokens // (max_input_tokens_per_call + max_output_tokens_per_call)
        )
    per_call_nanos = _usd_nanos(
        max_input_tokens_per_call,
        max_output_tokens_per_call,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    max_nanos = int(
        (max_usd * Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_CEILING)
    )
    candidates.append(max_nanos // per_call_nanos)
    capacity = min(candidates)
    return capacity, {
        "max_calls": max_calls,
        "max_input_tokens_per_call": max_input_tokens_per_call,
        "max_output_tokens_per_call": max_output_tokens_per_call,
        "max_input_tokens_total": max_input_tokens_total,
        "max_output_tokens_total": max_output_tokens_total,
        "max_total_tokens": max_total_tokens,
        "max_usd": str(max_usd),
        "input_usd_per_million": str(input_usd_per_million),
        "output_usd_per_million": str(output_usd_per_million),
        "worst_case_usd_nanos_per_call": per_call_nanos,
        "worst_case_complete_call_capacity": capacity,
        "reservation_policy": "full_per_call_caps_for_every_scheduled_call",
    }


def cap_plans_to_budget(
    plans: Sequence[RescuePlan],
    *,
    samples_per_parent: int,
    call_capacity: int,
) -> list[RescuePlan]:
    if call_capacity <= 0:
        raise ValueError("strict caps cannot fund one complete API call")
    selected: list[RescuePlan] = []
    used = 0
    for plan in plans:
        calls = len(plan.parents) * samples_per_parent
        if used + calls > call_capacity:
            break
        selected.append(
            RescuePlan(
                task_position=len(selected),
                task=plan.task,
                gate=plan.gate,
                local_terminal_sha256=plan.local_terminal_sha256,
                parents=plan.parents,
            )
        )
        used += calls
    if not selected:
        raise ValueError("strict caps cannot fund one complete rescue task")
    return selected


def _slot_binding(slot: ApiSlot) -> dict[str, Any]:
    return {
        "slot_position": slot.slot_position,
        "task_position": slot.task_position,
        "task_id": slot.task_id,
        "parent_position": slot.parent_position,
        "sample_index": slot.sample_index,
        "parent_code_sha256": slot.parent.code_sha256,
        "diagnostic_sha256": slot.parent.diagnostic_sha256,
        "feedback_source_sha256": slot.parent.feedback_source_sha256,
        "prompt_sha256": slot.prompt_sha256,
    }


def _event_matches_slot(event: Mapping[str, Any], slot: ApiSlot) -> bool:
    return all(event.get(key) == value for key, value in _slot_binding(slot).items())


def _contract_is_exploratory_prefix(contract: Mapping[str, Any]) -> bool:
    source = contract.get("local_source")
    return bool(
        isinstance(source, Mapping)
        and source.get("mode") == EXPLORATORY_PREFIX_RUN_MODE
        and source.get("exploratory_prefix") is True
        and source.get("production_floor_eligible") is False
        and type(source.get("terminal_prefix_length")) is int
        and source["terminal_prefix_length"] > 0
    )


def _contract_training_use_forbidden(contract: Mapping[str, Any]) -> bool:
    training_outputs = contract.get("training_outputs")
    return bool(
        isinstance(training_outputs, Mapping)
        and training_outputs.get("training_use_forbidden") is True
        and training_outputs.get("production_floor_eligible") is False
        and training_outputs.get("may_count_toward_production_min_unique_targets")
        is False
    )


def validate_rescue_journal(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    plans: Sequence[RescuePlan],
    slots: Sequence[ApiSlot],
) -> dict[str, Any]:
    """Validate exact event order and return resumable cursors/results."""

    if not events:
        return {
            "initialized": False,
            "slot_results": [],
            "verification_events": [],
            "complete": False,
        }
    header = events[0]
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("API rescue journal header differs from the exact run")
    cursor = 1
    results: list[dict[str, Any]] = []
    for slot in slots:
        if cursor >= len(events):
            break
        intent = events[cursor]
        if intent.get("event") != "call_intent" or not _event_matches_slot(
            intent, slot
        ):
            raise ValueError("API rescue journal call intent order differs")
        if intent.get("schema") != JOURNAL_SCHEMA or intent.get(
            "request_sha256"
        ) != canonical_sha256(
            {
                "system_sha256": sha256_text(SYSTEM_PROMPT),
                **_slot_binding(slot),
            }
        ):
            raise ValueError("API rescue journal call intent binding differs")
        cursor += 1
        if cursor >= len(events):
            raise PendingProviderCall(
                "the journal ends after call_intent; billing/result is ambiguous "
                "and the slot will not be called again automatically"
            )
        result = events[cursor]
        if result.get("event") != "call_result" or not _event_matches_slot(
            result, slot
        ):
            raise PendingProviderCall(
                "call_intent is not followed by its exact call_result"
            )
        if result.get("schema") != JOURNAL_SCHEMA:
            raise ValueError("API rescue call result schema differs")
        status = result.get("status")
        if status not in {"response", "provider_error", "contract_error"}:
            raise ValueError("API rescue call result status is invalid")
        code = str(result.get("code") or "")
        if code:
            if (
                sha256_text(code) != result.get("code_sha256")
                or result.get("parse_accepted") is not True
            ):
                raise ValueError("API rescue parsed-code digest differs")
        elif result.get("code_sha256") is not None:
            raise ValueError("API rescue empty code has a digest")
        usage = result.get("usage")
        if (
            not isinstance(usage, Mapping)
            or type(usage.get("charged_input_tokens")) is not int
            or type(usage.get("charged_output_tokens")) is not int
            or type(usage.get("charged_usd_nanos")) is not int
            or min(
                usage["charged_input_tokens"],
                usage["charged_output_tokens"],
                usage["charged_usd_nanos"],
            )
            < 0
        ):
            raise ValueError("API rescue budget charge is malformed")
        results.append(dict(result))
        cursor += 1
    if len(results) < len(slots):
        if cursor != len(events):
            raise ValueError("API rescue journal has events after a partial call phase")
        return {
            "initialized": True,
            "slot_results": results,
            "verification_events": [],
            "complete": False,
        }

    verifications: list[dict[str, Any]] = []
    for task_position, plan in enumerate(plans):
        if cursor >= len(events):
            break
        event = events[cursor]
        if (
            event.get("event") != "task_verification"
            or event.get("schema") != JOURNAL_SCHEMA
            or event.get("task_position") != task_position
            or event.get("task_id") != plan.task.task_id
            or event.get("source_sha256") != plan.task.source_sha256
            or event.get("split_binding_sha256") != plan.task.split_binding_sha256
            or event.get("all_api_generation_completed_before_private_gate") is not True
            or event.get("private_feedback_serialized_to_model") is not False
            or event.get("holdback_failure_triggers_generation") is not False
        ):
            raise ValueError("API rescue task verification order/binding differs")
        selected = event.get("selected_target")
        exploratory_prefix = _contract_is_exploratory_prefix(contract)
        training_use_forbidden = _contract_training_use_forbidden(contract)
        production_floor_eligible = (
            not exploratory_prefix and not training_use_forbidden
        )
        if selected is not None and (
            not isinstance(selected, Mapping)
            or selected.get("schema") != DIRECT_TARGET_SCHEMA
            or selected.get("task_id") != plan.task.task_id
            or selected.get("source_sha256") != plan.task.source_sha256
            or selected.get("visible_passed") is not True
            or selected.get("private_gate_passed") is not True
            or selected.get("exploratory_prefix") is not exploratory_prefix
            or selected.get("production_floor_eligible")
            is not production_floor_eligible
            or (
                training_use_forbidden
                and selected.get("training_use_forbidden") is not True
            )
            or sha256_text(str(selected.get("code") or ""))
            != selected.get("code_sha256")
        ):
            raise ValueError("API rescue selected target is not verified")
        verifications.append(dict(event))
        cursor += 1
    complete = False
    if len(verifications) == len(plans) and cursor < len(events):
        event = events[cursor]
        if (
            event.get("event") != "complete"
            or event.get("schema") != JOURNAL_SCHEMA
            or event.get("tasks") != len(plans)
            or event.get("slots") != len(slots)
            or event.get("verified_targets")
            != sum(row.get("selected_target") is not None for row in verifications)
            or event.get("exploratory_prefix")
            is not _contract_is_exploratory_prefix(contract)
            or event.get("production_floor_eligible")
            is not (
                not _contract_is_exploratory_prefix(contract)
                and not _contract_training_use_forbidden(contract)
            )
            or (
                _contract_training_use_forbidden(contract)
                and event.get("training_use_forbidden") is not True
            )
        ):
            raise ValueError("API rescue completion event differs")
        complete = True
        cursor += 1
    if cursor != len(events):
        raise ValueError("API rescue journal has trailing events")
    return {
        "initialized": True,
        "slot_results": results,
        "verification_events": verifications,
        "complete": complete,
    }


def _usage_charge(
    *,
    response: ProviderResponse | None,
    max_input_tokens: int,
    max_output_tokens: int,
    input_usd_per_million: Decimal,
    output_usd_per_million: Decimal,
) -> dict[str, int]:
    if response is None:
        charged_input = max_input_tokens
        charged_output = max_output_tokens
    else:
        charged_input = _positive_int(response.input_tokens, "provider input usage")
        charged_output = _positive_int(response.output_tokens, "provider output usage")
        if charged_input > max_input_tokens or charged_output > max_output_tokens:
            raise ValueError("provider usage exceeded its strict per-call reservation")
    return {
        "charged_input_tokens": charged_input,
        "charged_output_tokens": charged_output,
        "charged_usd_nanos": _usd_nanos(
            charged_input,
            charged_output,
            input_usd_per_million=input_usd_per_million,
            output_usd_per_million=output_usd_per_million,
        ),
    }


def _provider_retry_delay(
    exc: Exception,
    *,
    attempt: int,
    base_seconds: float,
    max_seconds: float,
) -> float | None:
    status_code = getattr(exc, "status_code", None)
    if status_code not in {429, 503}:
        return None
    header_delay = 0.0
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after")
            if raw is not None:
                header_delay = float(raw)
        except (TypeError, ValueError):
            header_delay = 0.0
    exponential = base_seconds * (2 ** max(0, attempt - 1))
    return min(max_seconds, max(header_delay, exponential))


def execute_api_phase(
    *,
    journal_path: Path,
    contract: Mapping[str, Any],
    plans: Sequence[RescuePlan],
    slots: Sequence[ApiSlot],
    transport: ProviderTransport,
    api_key: str,
    max_input_tokens: int,
    max_output_tokens: int,
    input_usd_per_million: Decimal,
    output_usd_per_million: Decimal,
    inter_call_delay_seconds: float = 0.0,
    abort_on_provider_error: bool = False,
    provider_max_attempts: int = 1,
    provider_retry_base_seconds: float = 1.0,
    provider_retry_max_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    state = validate_rescue_journal(
        load_journal(journal_path),
        contract=contract,
        plans=plans,
        slots=slots,
    )
    results = list(state["slot_results"])
    if state["verification_events"]:
        return results
    for slot in slots[len(results) :]:
        if results and inter_call_delay_seconds > 0:
            time.sleep(inter_call_delay_seconds)
        byte_upper_bound = (
            len(slot.prompt.encode("utf-8")) + len(SYSTEM_PROMPT.encode("utf-8")) + 1024
        )
        if byte_upper_bound > max_input_tokens:
            raise ValueError(
                f"{slot.task_id}: conservative prompt-token upper bound "
                f"{byte_upper_bound} exceeds reserved {max_input_tokens}"
            )
        binding = _slot_binding(slot)
        request_sha = canonical_sha256(
            {"system_sha256": sha256_text(SYSTEM_PROMPT), **binding}
        )
        _append_safe(
            journal_path,
            {
                "event": "call_intent",
                "schema": JOURNAL_SCHEMA,
                **binding,
                "request_sha256": request_sha,
                "reserved_input_tokens": max_input_tokens,
                "reserved_output_tokens": max_output_tokens,
                "plaintext_prompt_persisted": False,
                "credential_persisted": False,
            },
            api_key=api_key,
        )
        response: ProviderResponse | None = None
        status = "provider_error"
        error: str | None = None
        raw_text_sha: str | None = None
        code = ""
        parse_error: str | None = None
        attempts = 0
        retry_delays: list[float] = []
        for provider_attempt in range(1, provider_max_attempts + 1):
            attempts = provider_attempt
            try:
                response = transport.create(
                    system=SYSTEM_PROMPT,
                    user=slot.prompt,
                    max_output_tokens=max_output_tokens,
                )
                status = "response"
                raw_text_sha = sha256_text(response.text)
                code, parse_error = parse_code_only(response.text, api_key=api_key)
                break
            except Exception as exc:
                error = _redact_text(f"{type(exc).__name__}: {exc}", api_key)
                response = None
                retry_delay = _provider_retry_delay(
                    exc,
                    attempt=provider_attempt,
                    base_seconds=provider_retry_base_seconds,
                    max_seconds=provider_retry_max_seconds,
                )
                if retry_delay is None or provider_attempt == provider_max_attempts:
                    break
                retry_delays.append(retry_delay)
                print(
                    json.dumps(
                        {
                            "api_slot": slot.slot_position + 1,
                            "api_slots": len(slots),
                            "provider_attempt": provider_attempt,
                            "retry_in_seconds": retry_delay,
                            "status": "provider_retry",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                time.sleep(retry_delay)
        try:
            charge = _usage_charge(
                response=response,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                input_usd_per_million=input_usd_per_million,
                output_usd_per_million=output_usd_per_million,
            )
        except Exception as exc:
            status = "contract_error"
            error = _redact_text(f"{type(exc).__name__}: {exc}", api_key)
            response = None
            code = ""
            parse_error = "provider_usage_contract_error"
            charge = _usage_charge(
                response=None,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                input_usd_per_million=input_usd_per_million,
                output_usd_per_million=output_usd_per_million,
            )
        event = {
            "event": "call_result",
            "schema": JOURNAL_SCHEMA,
            **binding,
            "request_sha256": request_sha,
            "status": status,
            "response": (
                {
                    "response_id_sha256": (
                        sha256_text(response.response_id)
                        if response.response_id
                        else None
                    ),
                    "resolved_model": response.model,
                    "finish_reason": response.finish_reason,
                    "raw_text_sha256": raw_text_sha,
                }
                if response is not None
                else None
            ),
            "parse_accepted": bool(code),
            "parse_rejection": parse_error,
            "code": code,
            "code_sha256": sha256_text(code) if code else None,
            "error": error,
            "usage": charge,
            "provider_attempts": attempts,
            "provider_retry_delays_seconds": retry_delays,
            "plaintext_reasoning_persisted": False,
            "credential_persisted": False,
        }
        result = _append_safe(journal_path, event, api_key=api_key)
        results.append(result)
        print(
            json.dumps(
                {
                    "api_slot": slot.slot_position + 1,
                    "api_slots": len(slots),
                    "task_id": slot.task_id,
                    "status": status,
                    "code_only": bool(code),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if abort_on_provider_error and status in {
            "provider_error",
            "contract_error",
        }:
            raise RuntimeError(
                f"{slot.task_id}: fail-fast provider status {status}"
            )
    return results


def _runtime_evaluator(*, timeout: int, stability_runs: int) -> EvaluateFn:
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


def execute_verification_phase(
    *,
    journal_path: Path,
    contract: Mapping[str, Any],
    plans: Sequence[RescuePlan],
    slots: Sequence[ApiSlot],
    evaluate: EvaluateFn,
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Verify only after the entire fixed API schedule is terminal."""

    state = validate_rescue_journal(
        load_journal(journal_path),
        contract=contract,
        plans=plans,
        slots=slots,
    )
    results = list(state["slot_results"])
    if len(results) != len(slots):
        raise ValueError("all API generation must finish before verification")
    verifications = list(state["verification_events"])
    if state["complete"]:
        return verifications
    exploratory_prefix = _contract_is_exploratory_prefix(contract)
    training_use_forbidden = _contract_training_use_forbidden(contract)
    production_floor_eligible = (
        not exploratory_prefix and not training_use_forbidden
    )
    by_task: dict[int, list[tuple[ApiSlot, Mapping[str, Any]]]] = {}
    for slot, result in zip(slots, results, strict=True):
        by_task.setdefault(slot.task_position, []).append((slot, result))

    for task_position in range(len(verifications), len(plans)):
        plan = plans[task_position]
        visible: list[dict[str, Any]] = []
        for slot, result in by_task.get(task_position, []):
            code = str(result.get("code") or "")
            if not code:
                continue
            outcome = evaluate(
                code,
                plan.task.visible_tests,
                f"{plan.task.task_id}-api-visible-{slot.slot_position}",
            )
            visible.append(
                {
                    "slot_position": slot.slot_position,
                    "parent_position": slot.parent_position,
                    "sample_index": slot.sample_index,
                    "parent_code_sha256": slot.parent.code_sha256,
                    "diagnostic_sha256": slot.parent.diagnostic_sha256,
                    "feedback_source_sha256": slot.parent.feedback_source_sha256,
                    "code": code,
                    "code_sha256": sha256_text(code),
                    "compiled": bool(outcome.compiled),
                    "passed": bool(outcome.compiled and outcome.passed),
                }
            )
        # Private checks can only reject already-visible-passing code.  No API
        # request follows any private check anywhere in the run.
        private_results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in visible:
            digest = candidate["code_sha256"]
            if not candidate["passed"] or digest in seen:
                continue
            seen.add(digest)
            outcome = evaluate(
                candidate["code"],
                plan.gate.tests,
                f"{plan.task.task_id}-api-private-{candidate['slot_position']}",
            )
            private_results.append(
                {
                    "slot_position": candidate["slot_position"],
                    "code_sha256": digest,
                    "private_gate_passed": bool(outcome.compiled and outcome.passed),
                }
            )
        selected: dict[str, Any] | None = None
        for gate_result in private_results:
            if not gate_result["private_gate_passed"]:
                continue
            candidate = next(
                row
                for row in visible
                if row["slot_position"] == gate_result["slot_position"]
            )
            selected = {
                "schema": DIRECT_TARGET_SCHEMA,
                "task_id": plan.task.task_id,
                "source_sha256": plan.task.source_sha256,
                "code": candidate["code"],
                "code_sha256": candidate["code_sha256"],
                "slot_position": candidate["slot_position"],
                "parent_position": candidate["parent_position"],
                "parent_code_sha256": candidate["parent_code_sha256"],
                "diagnostic_sha256": candidate["diagnostic_sha256"],
                "feedback_source_sha256": candidate["feedback_source_sha256"],
                "visible_passed": True,
                "private_gate_passed": True,
                "exploratory_prefix": exploratory_prefix,
                "production_floor_eligible": production_floor_eligible,
                "training_use_forbidden": training_use_forbidden,
            }
            break
        event = _append_safe(
            journal_path,
            {
                "event": "task_verification",
                "schema": JOURNAL_SCHEMA,
                "task_position": task_position,
                "task_id": plan.task.task_id,
                "source_sha256": plan.task.source_sha256,
                "split_binding_sha256": plan.task.split_binding_sha256,
                "visible_results": visible,
                "private_gate_results": private_results,
                "selected_target": selected,
                "all_api_generation_completed_before_private_gate": True,
                "private_feedback_serialized_to_model": False,
                "holdback_failure_triggers_generation": False,
                "private_diagnostics_persisted": False,
            },
            api_key=api_key,
        )
        verifications.append(event)
        print(
            json.dumps(
                {
                    "verification_task": task_position + 1,
                    "verification_tasks": len(plans),
                    "task_id": plan.task.task_id,
                    "visible_passes": sum(row["passed"] for row in visible),
                    "accepted": selected is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _append_safe(
        journal_path,
        {
            "event": "complete",
            "schema": JOURNAL_SCHEMA,
            "tasks": len(plans),
            "slots": len(slots),
            "verified_targets": sum(
                row.get("selected_target") is not None for row in verifications
            ),
            "exploratory_prefix": exploratory_prefix,
            "production_floor_eligible": production_floor_eligible,
            "training_use_forbidden": training_use_forbidden,
        },
        api_key=api_key,
    )
    return verifications


class AnthropicTransport:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        thinking: str,
        effort: str,
    ) -> None:
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("install the anthropic Python package") from exc
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.thinking = thinking
        self.effort = effort

    def create(
        self, *, system: str, user: str, max_output_tokens: int
    ) -> ProviderResponse:
        options: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.thinking == "adaptive":
            options["thinking"] = {"type": "adaptive"}
            options["output_config"] = {"effort": self.effort}
        response = self.client.messages.create(**options)
        blocks = _field(response, "content")
        if not isinstance(blocks, list):
            raise ValueError("Anthropic response content is malformed")
        text = "\n\n".join(
            str(_field(block, "text"))
            for block in blocks
            if _field(block, "type") == "text"
            and isinstance(_field(block, "text"), str)
        )
        usage = _field(response, "usage")
        input_tokens = int(_field(usage, "input_tokens") or 0)
        input_tokens += int(_field(usage, "cache_creation_input_tokens") or 0)
        input_tokens += int(_field(usage, "cache_read_input_tokens") or 0)
        return ProviderResponse(
            text=text,
            response_id=str(_field(response, "id") or ""),
            model=str(_field(response, "model") or ""),
            input_tokens=input_tokens,
            output_tokens=int(_field(usage, "output_tokens") or 0),
            finish_reason=str(_field(response, "stop_reason") or ""),
        )


class OpenAITransport:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        provider: str,
        api_version: str,
        reasoning_effort: str,
        chat_token_parameter: str,
        openrouter_reasoning: str = "disabled",
        openrouter_provider_only: Sequence[str] = (),
        openrouter_provider_order: Sequence[str] = (),
        openrouter_allow_fallbacks: bool = False,
        openrouter_require_parameters: bool = False,
        openrouter_include_reasoning: bool = False,
        openrouter_enforce_distillable_text: bool = False,
        openrouter_reasoning_effort: str = "",
    ) -> None:
        try:
            from openai import AzureOpenAI, OpenAI
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("install the openai Python package") from exc
        if provider in {"azure_responses", "azure_chat"}:
            if not api_version:
                raise ValueError("Azure OpenAI requires --api_version")
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                max_retries=0,
            )
        else:
            # Azure OpenAI v1 deliberately uses the ordinary OpenAI client
            # with base_url=https://.../openai/v1.  It has no api-version
            # query parameter; ``model`` remains the Azure deployment name.
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
            )
        self.model = model
        self.responses_style = provider.endswith("_responses")
        self.reasoning_effort = reasoning_effort
        self.chat_token_parameter = chat_token_parameter
        self.openrouter_reasoning = openrouter_reasoning
        self.openrouter_chat = provider == "openrouter_chat"
        self.openrouter_provider_only = tuple(openrouter_provider_only)
        self.openrouter_provider_order = tuple(openrouter_provider_order)
        self.openrouter_allow_fallbacks = bool(openrouter_allow_fallbacks)
        self.openrouter_require_parameters = bool(openrouter_require_parameters)
        self.openrouter_include_reasoning = bool(openrouter_include_reasoning)
        self.openrouter_enforce_distillable_text = bool(
            openrouter_enforce_distillable_text
        )
        self.openrouter_reasoning_effort = openrouter_reasoning_effort

    @staticmethod
    def _responses_text(response: Any) -> str:
        direct = _field(response, "output_text")
        if isinstance(direct, str):
            return direct
        chunks: list[str] = []
        output = _field(response, "output")
        if not isinstance(output, list):
            return ""
        for item in output:
            content = _field(item, "content")
            if not isinstance(content, list):
                continue
            for part in content:
                if _field(part, "type") in {"output_text", "text"} and isinstance(
                    _field(part, "text"), str
                ):
                    chunks.append(_field(part, "text"))
        return "".join(chunks)

    def create(
        self, *, system: str, user: str, max_output_tokens: int
    ) -> ProviderResponse:
        if self.responses_style:
            options: dict[str, Any] = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_output_tokens": max_output_tokens,
            }
            if self.reasoning_effort:
                options["reasoning"] = {"effort": self.reasoning_effort}
            response = self.client.responses.create(**options)
            usage = _field(response, "usage")
            incomplete = _field(response, "incomplete_details")
            return ProviderResponse(
                text=self._responses_text(response),
                response_id=str(_field(response, "id") or ""),
                model=str(_field(response, "model") or ""),
                input_tokens=int(_field(usage, "input_tokens") or 0),
                output_tokens=int(_field(usage, "output_tokens") or 0),
                finish_reason=(
                    str(_field(incomplete, "reason") or "")
                    if str(_field(response, "status") or "") != "completed"
                    else "stop"
                ),
            )
        options = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        options[self.chat_token_parameter] = max_output_tokens
        if self.reasoning_effort:
            options["reasoning_effort"] = self.reasoning_effort
        if self.openrouter_chat:
            provider_options: dict[str, Any] = {
                "only": list(self.openrouter_provider_only),
                "allow_fallbacks": self.openrouter_allow_fallbacks,
                "require_parameters": self.openrouter_require_parameters,
                "enforce_distillable_text": (
                    self.openrouter_enforce_distillable_text
                ),
            }
            if self.openrouter_provider_order:
                provider_options["order"] = list(self.openrouter_provider_order)
            extra_body: dict[str, Any] = {
                "provider": provider_options
            }
            if self.openrouter_reasoning == "enabled":
                # Reasoning is returned separately from ``message.content``.
                # The sealed journal persists only the final code text, while
                # authoritative completion usage still charges reasoning.
                reasoning: dict[str, Any] = {
                    "enabled": True,
                    "exclude": False,
                }
                if self.openrouter_reasoning_effort:
                    reasoning["effort"] = self.openrouter_reasoning_effort
                extra_body["reasoning"] = reasoning
                extra_body["include_reasoning"] = self.openrouter_include_reasoning
            options["extra_body"] = extra_body
        response = self.client.chat.completions.create(**options)
        choices = _field(response, "choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RetryableProviderPayloadError(
                "OpenAI-compatible response must contain exactly one choice"
            )
        choice = choices[0]
        message = _field(choice, "message")
        usage = _field(response, "usage")
        return ProviderResponse(
            text=str(_field(message, "content") or ""),
            response_id=str(_field(response, "id") or ""),
            model=str(_field(response, "model") or ""),
            input_tokens=int(_field(usage, "prompt_tokens") or 0),
            output_tokens=int(_field(usage, "completion_tokens") or 0),
            finish_reason=str(_field(choice, "finish_reason") or ""),
        )


def freeze_local_terminal_prefix(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    scheduled_tasks: Sequence[PilotTask],
    terminal_prefix_length: int,
) -> tuple[list[PilotTask], list[dict[str, Any]], dict[str, Any]]:
    """Validate and bind an immutable prefix of an append-only local pilot.

    ``events`` must come from :func:`load_journal`, which verifies the durable
    hash chain and its chain-head sidecar before this semantic validation is
    reached.  The returned source record binds only the header and requested
    terminal prefix.  Consequently, appending later terminals to the source
    journal does not alter an already-started exploratory rescue contract.
    """

    if (
        isinstance(terminal_prefix_length, bool)
        or not isinstance(terminal_prefix_length, int)
        or terminal_prefix_length <= 0
    ):
        raise ValueError("exploratory terminal prefix length must be positive")
    if terminal_prefix_length > len(scheduled_tasks):
        raise ValueError(
            "exploratory terminal prefix exceeds the sealed local schedule"
        )
    if not events:
        raise ValueError("local RS-SFT pilot journal is empty")

    # Validate the header and every terminal currently present.  This catches
    # malformed tails as well as malformed prefix rows; a probe never proceeds
    # from a semantically invalid source journal.
    all_terminals, _complete = validate_local_journal_state(
        events, contract=contract, scheduled_tasks=scheduled_tasks
    )
    if len(all_terminals) < terminal_prefix_length:
        raise ValueError(
            "local RS-SFT pilot has fewer validated terminals than the "
            "requested exploratory prefix"
        )

    prefix_events = list(events[: terminal_prefix_length + 1])
    prefix_tasks = list(scheduled_tasks[:terminal_prefix_length])
    prefix_terminals, prefix_complete = validate_local_journal_state(
        prefix_events,
        contract=contract,
        scheduled_tasks=scheduled_tasks,
    )
    if prefix_complete or len(prefix_terminals) != terminal_prefix_length:
        raise ValueError("exploratory local terminal prefix is not exact")
    terminal_event_sha256s = [
        str(row.get("journal_event_sha256") or "") for row in prefix_terminals
    ]
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in terminal_event_sha256s
    ):
        raise ValueError("exploratory terminal prefix lacks hash-chain bindings")
    header_sha256 = str(events[0].get("journal_event_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", header_sha256):
        raise ValueError("local RS-SFT pilot header lacks a hash-chain binding")
    source_record = {
        "mode": EXPLORATORY_PREFIX_RUN_MODE,
        "exploratory_prefix": True,
        "production_floor_eligible": False,
        "terminal_prefix_length": terminal_prefix_length,
        "header_event_sha256": header_sha256,
        "terminal_prefix_head_event_sha256": terminal_event_sha256s[-1],
        "terminal_event_sha256s_sha256": canonical_sha256(terminal_event_sha256s),
        "terminal_task_ids_sha256": canonical_sha256(
            [row["task_id"] for row in prefix_terminals]
        ),
        "prefix_events_sha256": canonical_sha256(prefix_events),
        "source_journal_modified": False,
    }
    return prefix_tasks, prefix_terminals, source_record


def _load_completed_local_run(
    args: argparse.Namespace,
) -> tuple[
    list[PilotTask],
    dict[str, PrivateGate],
    list[PilotTask],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    tasks, gates, input_record = load_pilot_inputs(
        rollout_file=args.rollout_file,
        f2_jsonl=args.f2_jsonl,
        private_holdback=args.private_holdback,
        expected_rollout_sha256=args.expected_rollout_sha256,
        expected_f2_sha256=args.expected_f2_sha256,
        expected_private_holdback_sha256=args.expected_private_holdback_sha256,
        allow_unpinned_inputs=args.allow_unpinned_inputs,
    )
    events = load_journal(args.pilot_journal)
    if not events:
        raise ValueError("local RS-SFT pilot journal is empty")
    header = events[0]
    contract = header.get("contract")
    if (
        header.get("schema") != LOCAL_JOURNAL_SCHEMA
        or not isinstance(contract, Mapping)
        or contract.get("schema") != LOCAL_RUN_SCHEMA
        or contract.get("inputs") != input_record
    ):
        raise ValueError("local RS-SFT pilot header/input binding differs")
    schedule = contract.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("local RS-SFT pilot schedule is missing")
    indices = deterministic_pilot_indices(
        tasks,
        seed=int(schedule.get("seed", -1)),
        limit=int(schedule.get("pilot_tasks", -1)),
        offset=int(schedule.get("pilot_offset", 0)),
    )
    scheduled = [tasks[index] for index in indices]
    if canonical_sha256([task.task_id for task in scheduled]) != schedule.get(
        "task_ids_sha256"
    ):
        raise ValueError("local RS-SFT pilot task schedule digest differs")
    if args.exploratory_terminal_prefix:
        scheduled, terminals, source_record = freeze_local_terminal_prefix(
            events,
            contract=contract,
            scheduled_tasks=scheduled,
            terminal_prefix_length=args.exploratory_terminal_prefix,
        )
    else:
        terminals, complete = validate_local_journal_state(
            events, contract=contract, scheduled_tasks=scheduled
        )
        if not complete:
            raise ValueError(
                "API rescue requires a completed local RS-SFT pilot journal "
                "unless --exploratory_terminal_prefix is explicit"
            )
        source_record = journal_record(args.pilot_journal)
        source_record.pop("path", None)
        source_record.pop("chain_head_path", None)
        source_record.update(
            {
                "mode": COMPLETED_LOCAL_RUN_MODE,
                "exploratory_prefix": False,
                "production_floor_eligible": True,
                "terminal_prefix_length": None,
                "source_journal_modified": False,
            }
        )
    return (
        tasks,
        gates,
        scheduled,
        terminals,
        input_record,
        source_record,
    )


def _provider_contract(args: argparse.Namespace, base_url: str) -> dict[str, Any]:
    dated_azure = args.provider in {"azure_responses", "azure_chat"}
    azure_v1 = args.provider in {"azure_v1_responses", "azure_v1_chat"}
    result = {
        "provider": args.provider,
        "model": args.model,
        "base_url": base_url,
        "api_version": args.api_version if dated_azure else None,
        "max_output_tokens": args.max_output_tokens,
        "timeout_seconds": args.timeout_seconds,
        "automatic_retries": max(
            0, getattr(args, "provider_max_attempts", 1) - 1
        ),
        "retryable_http_statuses": [429, 503],
        "provider_retry_base_seconds": getattr(
            args, "provider_retry_base_seconds", 1.0
        ),
        "provider_retry_max_seconds": getattr(
            args, "provider_retry_max_seconds", 30.0
        ),
        "inter_call_delay_seconds": getattr(
            args, "inter_call_delay_seconds", 0.0
        ),
        "abort_on_provider_error": getattr(
            args, "abort_on_provider_error", False
        ),
        "one_candidate_per_call": True,
        "credential_source": "environment_value_not_persisted",
        "azure": dated_azure or azure_v1,
        "client": (
            "AzureOpenAI"
            if dated_azure
            else "OpenAI" if args.provider != "anthropic" else "Anthropic"
        ),
        "azure_endpoint_mode": (
            "dated_azure_sdk_with_api_version"
            if dated_azure
            else "openai_v1_standard_client_without_api_version" if azure_v1 else None
        ),
        "model_semantics": (
            "azure_deployment_name" if dated_azure or azure_v1 else "model_id"
        ),
    }
    if args.provider == "anthropic":
        result["thinking"] = args.anthropic_thinking
        result["effort"] = args.anthropic_effort
    else:
        result["reasoning_effort"] = args.reasoning_effort or None
        if args.provider.endswith("_chat"):
            result["chat_token_parameter"] = args.chat_token_parameter
        if args.provider == "openrouter_chat":
            result["openrouter_reasoning"] = {
                "enabled": args.openrouter_reasoning == "enabled",
                "included_in_response": args.openrouter_include_reasoning,
            }
            if args.openrouter_reasoning_effort:
                result["openrouter_reasoning"]["effort"] = (
                    args.openrouter_reasoning_effort
                )
            openrouter_routing = {
                "only": list(args.openrouter_provider_only),
                "allow_fallbacks": args.openrouter_allow_fallbacks,
                "require_parameters": args.openrouter_require_parameters,
                "enforce_distillable_text": (
                    args.openrouter_enforce_distillable_text
                ),
            }
            openrouter_provider_order = getattr(
                args, "openrouter_provider_order", ()
            )
            if openrouter_provider_order:
                openrouter_routing["order"] = list(openrouter_provider_order)
            result["openrouter_routing"] = openrouter_routing
    return result


def _build_transport(
    args: argparse.Namespace, *, api_key: str, base_url: str
) -> ProviderTransport:
    if args.provider == "anthropic":
        return AnthropicTransport(
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            timeout=args.timeout_seconds,
            thinking=args.anthropic_thinking,
            effort=args.anthropic_effort,
        )
    return OpenAITransport(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        timeout=args.timeout_seconds,
        provider=args.provider,
        api_version=args.api_version,
        reasoning_effort=args.reasoning_effort,
        chat_token_parameter=args.chat_token_parameter,
        openrouter_reasoning=args.openrouter_reasoning,
        openrouter_provider_only=args.openrouter_provider_only,
        openrouter_provider_order=args.openrouter_provider_order,
        openrouter_allow_fallbacks=args.openrouter_allow_fallbacks,
        openrouter_require_parameters=args.openrouter_require_parameters,
        openrouter_include_reasoning=args.openrouter_include_reasoning,
        openrouter_enforce_distillable_text=(
            args.openrouter_enforce_distillable_text
        ),
        openrouter_reasoning_effort=args.openrouter_reasoning_effort,
    )


def _publish_training_outputs(
    *,
    output_dir: Path,
    plans: Sequence[RescuePlan],
    verifications: Sequence[Mapping[str, Any]],
    contract_sha256: str,
    exploratory_prefix: bool = False,
) -> dict[str, Any]:
    plan_by_id = {plan.task.task_id: plan for plan in plans}
    direct_targets: list[dict[str, Any]] = []
    direct_f2: list[dict[str, Any]] = []
    repair_targets: list[dict[str, Any]] = []
    repair_sources: list[dict[str, Any]] = []
    for verification in verifications:
        selected = verification.get("selected_target")
        if not isinstance(selected, Mapping):
            continue
        task_id = str(selected["task_id"])
        plan = plan_by_id[task_id]
        parent_position = int(selected["parent_position"])
        parent = plan.parents[parent_position]
        code = str(selected["code"])
        direct_targets.append(
            {
                "schema": DIRECT_TARGET_SCHEMA,
                "task_id": task_id,
                "dart_source": code,
                "dart_source_sha256": sha256_text(code),
                "source_sha256": plan.task.source_sha256,
                "visible_passed": True,
                "private_gate_passed": True,
                "exploratory_prefix": exploratory_prefix,
                "production_floor_eligible": not exploratory_prefix,
                "provenance": {
                    "run_contract_sha256": contract_sha256,
                    "slot_position": selected["slot_position"],
                    "parent_code_sha256": parent.code_sha256,
                    "diagnostic_sha256": parent.diagnostic_sha256,
                },
            }
        )
        direct_f2.append(dict(plan.task.f2_row))
        repair_id = f"{task_id}::api-rescue::{selected['slot_position']:06d}"
        repair_targets.append(
            {
                "schema": REPAIR_PAIR_SCHEMA,
                "task_id": repair_id,
                "source_task_id": task_id,
                "dart_source": code,
                "dart_source_sha256": sha256_text(code),
                "exploratory_prefix": exploratory_prefix,
                "production_floor_eligible": not exploratory_prefix,
            }
        )
        repair_sources.append(
            {
                "schema": REPAIR_PAIR_SCHEMA,
                "task_id": repair_id,
                "source_task_id": task_id,
                "encoder_source": parent.feedback_source,
                "encoder_source_sha256": parent.feedback_source_sha256,
                "original_f2_source_sha256": plan.task.source_sha256,
                "parent_code_sha256": parent.code_sha256,
                "compiler_diagnostic": parent.diagnostic,
                "compiler_diagnostic_sha256": parent.diagnostic_sha256,
                "private_feedback_present": False,
                "tests_present": False,
                "gold_present": False,
                "exploratory_prefix": exploratory_prefix,
                "production_floor_eligible": not exploratory_prefix,
            }
        )
    paths = {
        "direct_targets": output_dir / "direct_hard_targets.jsonl",
        "direct_f2": output_dir / "direct_hard_targets_f2.jsonl",
        "repair_targets": output_dir / "repair_policy_targets.jsonl",
        "repair_sources": output_dir / "repair_policy_sources.jsonl",
    }
    rows = {
        "direct_targets": direct_targets,
        "direct_f2": direct_f2,
        "repair_targets": repair_targets,
        "repair_sources": repair_sources,
    }
    for name, path in paths.items():
        _exact_write_jsonl(path, rows[name])
    records = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows[name]),
        }
        for name, path in paths.items()
    }
    direct_manifest = {
        "schema": DIRECT_MANIFEST_SCHEMA,
        "run_contract_sha256": contract_sha256,
        "rows": len(direct_targets),
        "targets": records["direct_targets"],
        "f2": records["direct_f2"],
        "mapping": "original_sealed_F2_to_visible_and_private_verified_Dart",
        "compatible_trainer": "t5gemma2_enriched_sft.py",
        "unique_source_tasks": True,
        "exploratory_prefix": exploratory_prefix,
        "production_floor_eligible": not exploratory_prefix,
        "may_count_toward_production_min_unique_targets": not exploratory_prefix,
    }
    repair_manifest = {
        "schema": REPAIR_MANIFEST_SCHEMA,
        "run_contract_sha256": contract_sha256,
        "rows": len(repair_targets),
        "targets": records["repair_targets"],
        "prebuilt_encoder_sources": records["repair_sources"],
        "mapping": (
            "exact_original_F2_plus_failed_candidate_plus_sanitized_compiler_"
            "diagnostic_to_the_same_visible_and_private_verified_Dart"
        ),
        "source_is_exact_model_input": True,
        "requires_prebuilt_encoder_source_loader": True,
        "private_feedback_present": False,
        "exploratory_prefix": exploratory_prefix,
        "production_floor_eligible": not exploratory_prefix,
        "may_count_toward_production_min_unique_targets": not exploratory_prefix,
    }
    require_exact_or_write(output_dir / "direct_manifest.json", direct_manifest)
    require_exact_or_write(output_dir / "repair_policy_manifest.json", repair_manifest)
    return {
        "rows": rows,
        "files": records,
        "direct_manifest": direct_manifest,
        "repair_manifest": repair_manifest,
    }


def run(
    args: argparse.Namespace,
    *,
    transport: ProviderTransport | None = None,
    evaluate: EvaluateFn | None = None,
) -> dict[str, Any]:
    (
        _all_tasks,
        gates,
        scheduled_tasks,
        terminals,
        input_record,
        source_journal_record,
    ) = _load_completed_local_run(args)
    exploratory_prefix = (
        source_journal_record.get("mode") == EXPLORATORY_PREFIX_RUN_MODE
    )
    evaluation_only = bool(args.evaluation_only)
    production_floor_eligible = not exploratory_prefix and not evaluation_only
    all_eligible_plans = select_rescue_plans(
        scheduled_tasks=scheduled_tasks,
        gates=gates,
        terminals=terminals,
        seed=args.seed,
        max_tasks=0,
        max_parents_per_task=args.max_parents_per_task,
        eligible_task_offset=0,
    )
    all_eligible_task_ids = [plan.task.task_id for plan in all_eligible_plans]
    retry_source: RetryParseFailuresOrTruncationsSource | None = None
    if args.retry_parse_failures_or_truncations_report:
        retry_source = load_retry_parse_failures_or_truncations_source(
            report_path=args.retry_parse_failures_or_truncations_report,
            expected_report_sha256=(
                args.expected_retry_parse_failures_or_truncations_report_sha256
            ),
            current_eligible_plans=all_eligible_plans,
            input_record=input_record,
            source_journal_record=source_journal_record,
        )
        retry_task_ids = [plan.task.task_id for plan in retry_source.plans]
        if (
            args.expected_retry_parse_failures_or_truncations_tasks
            != len(retry_task_ids)
        ):
            raise ValueError(
                "retry parse-failures-or-truncations task count differs: "
                f"expected "
                f"{args.expected_retry_parse_failures_or_truncations_tasks}, "
                f"got {len(retry_task_ids)}"
            )
        if (
            args.expected_retry_parse_failures_or_truncations_task_ids_sha256
            != canonical_sha256(retry_task_ids)
        ):
            raise ValueError(
                "retry parse-failures-or-truncations task digest differs"
            )
        prior_exclusions = PriorSuccessExclusions(
            frozenset(), frozenset(), ()
        )
        residual_plans = list(retry_source.plans)
    else:
        prior_exclusions = load_prior_success_exclusions(
            report_paths=args.prior_success_report,
            expected_report_sha256s=args.expected_prior_success_report_sha256,
            current_eligible_task_ids=all_eligible_task_ids,
            input_record=input_record,
            source_journal_record=source_journal_record,
            require_disjoint_schedules=args.require_prior_schedules_disjoint,
            require_complete_coverage=(
                args.require_prior_schedule_complete_coverage
            ),
        )
        residual_plans = exclude_prior_verified_plans(
            all_eligible_plans, prior_exclusions.verified_task_ids
        )
    prior_scheduled_order = [
        task_id
        for task_id in all_eligible_task_ids
        if task_id in prior_exclusions.scheduled_task_ids
    ]
    prior_verified_order = [
        task_id
        for task_id in all_eligible_task_ids
        if task_id in prior_exclusions.verified_task_ids
    ]
    residual_task_ids = [plan.task.task_id for plan in residual_plans]
    expected_counts = (
        (
            args.expected_prior_scheduled_tasks,
            len(prior_scheduled_order),
            "prior scheduled task count",
        ),
        (
            args.expected_prior_verified_tasks,
            len(prior_verified_order),
            "prior verified task count",
        ),
        (
            args.expected_residual_tasks,
            len(residual_task_ids),
            "residual task count",
        ),
    )
    for expected, observed, label in expected_counts:
        if expected >= 0 and expected != observed:
            raise ValueError(f"{label} differs: expected {expected}, got {observed}")
    expected_digests = (
        (
            args.expected_prior_scheduled_task_ids_sha256,
            canonical_sha256(prior_scheduled_order),
            "prior scheduled task digest",
        ),
        (
            args.expected_prior_verified_task_ids_sha256,
            canonical_sha256(prior_verified_order),
            "prior verified task digest",
        ),
        (
            args.expected_residual_task_ids_sha256,
            canonical_sha256(residual_task_ids),
            "residual task digest",
        ),
    )
    for expected, observed, label in expected_digests:
        if expected:
            _require_sha256(expected, f"expected {label}")
            if expected != observed:
                raise ValueError(f"{label} differs")
    prior_exclusion_contract = {
        "schema": PRIOR_SUCCESS_EXCLUSION_SCHEMA,
        "enabled": bool(prior_exclusions.records),
        "sources": list(prior_exclusions.records),
        "source_reports": len(prior_exclusions.records),
        "schedules_required_disjoint": (args.require_prior_schedules_disjoint),
        "schedule_complete_coverage_required": (
            args.require_prior_schedule_complete_coverage
        ),
        "scheduled_tasks": len(prior_scheduled_order),
        "scheduled_task_ids_sha256": canonical_sha256(prior_scheduled_order),
        "verified_tasks_excluded": len(prior_verified_order),
        "verified_task_ids_sha256": canonical_sha256(prior_verified_order),
        "residual_tasks": len(residual_task_ids),
        "residual_task_ids_sha256": canonical_sha256(residual_task_ids),
        "exclusion_requires_visible_and_private_verified_target": True,
        "prior_private_feedback_used_as_model_input": False,
    }
    eligible_after_offset = slice_rescue_plans(
        residual_plans, offset=args.eligible_task_offset, max_tasks=0
    )
    plans_before_budget = slice_rescue_plans(
        residual_plans,
        offset=args.eligible_task_offset,
        max_tasks=args.max_tasks,
    )
    if not plans_before_budget:
        raise ValueError("completed local pilot contains no rescuable residual task")
    input_price = Decimal(str(args.input_usd_per_million))
    output_price = Decimal(str(args.output_usd_per_million))
    max_usd = Decimal(str(args.max_usd))
    capacity, budget_contract = schedule_capacity(
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
    plans = cap_plans_to_budget(
        plans_before_budget,
        samples_per_parent=args.samples_per_parent,
        call_capacity=capacity,
    )
    if retry_source is not None and len(plans) != len(residual_plans):
        raise ValueError(
            "normal task/budget caps do not cover the exact retry "
            "parse-failures-or-truncations cohort"
        )
    slots = build_slots(plans, samples_per_parent=args.samples_per_parent)
    scheduled_task_ids_sha256 = canonical_sha256([plan.task.task_id for plan in plans])
    if args.expected_scheduled_task_ids_sha256:
        _require_sha256(
            args.expected_scheduled_task_ids_sha256,
            "expected scheduled task digest",
        )
        if args.expected_scheduled_task_ids_sha256 != scheduled_task_ids_sha256:
            raise ValueError("scheduled task digest differs")
    base_url = validate_provider_endpoint(
        provider=args.provider,
        base_url=args.base_url,
        api_version=args.api_version,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if evaluation_only:
        forbidden_artifacts = (
            *tuple(_PRIOR_OUTPUT_FILES.values()),
            "direct_manifest.json",
            "repair_policy_manifest.json",
        )
        stale = [
            name for name in forbidden_artifacts if (output_dir / name).exists()
        ]
        if stale:
            raise ValueError(
                "evaluation-only output directory contains trainable artifacts: "
                + ", ".join(sorted(stale))
            )
    journal_path = output_dir / "api_rescue.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "execution_mode": (
            "evaluation_only"
            if evaluation_only
            else (
                "rs_sft_retry_parse_failures_or_truncations"
                if retry_source is not None
                else "rs_sft_rescue"
            )
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_local_pilot_journal": source_journal_record,
        "local_source": {
            "mode": source_journal_record["mode"],
            "exploratory_prefix": exploratory_prefix,
            "production_floor_eligible": not exploratory_prefix,
            "terminal_prefix_length": source_journal_record["terminal_prefix_length"],
            "terminal_prefix_head_event_sha256": source_journal_record.get(
                "terminal_prefix_head_event_sha256"
            ),
            "terminal_event_sha256s_sha256": source_journal_record.get(
                "terminal_event_sha256s_sha256"
            ),
            "source_journal_modified": False,
        },
        "inputs": input_record,
        "prior_success_exclusion": prior_exclusion_contract,
        "retry_parse_failures_or_truncations_source": (
            {"enabled": False}
            if retry_source is None
            else retry_source.record
        ),
        "selection": {
            "seed": args.seed,
            "eligible_task_offset": args.eligible_task_offset,
            "eligible_task_offset_applied_after_deterministic_sort": True,
            "eligible_task_offset_applied_after_deterministic_sort_and_prior_success_exclusion": True,
            "eligible_tasks_before_offset": len(all_eligible_plans),
            "eligible_tasks_after_prior_success_exclusion": len(residual_plans),
            "eligible_task_ids_before_prior_success_exclusion_sha256": (
                canonical_sha256(all_eligible_task_ids)
            ),
            "residual_task_ids_before_offset_sha256": canonical_sha256(
                residual_task_ids
            ),
            "eligible_tasks_after_offset_before_task_cap": len(eligible_after_offset),
            "max_tasks_before_budget": args.max_tasks,
            "eligible_tasks_after_task_cap_before_budget": len(plans_before_budget),
            "max_parents_per_task": args.max_parents_per_task,
            "samples_per_parent": args.samples_per_parent,
            "scheduled_tasks": len(plans),
            "scheduled_slots": len(slots),
            "task_ids_sha256": scheduled_task_ids_sha256,
            "slot_bindings_sha256": canonical_sha256(
                [_slot_binding(slot) for slot in slots]
            ),
            "all_zero_visible_only": True,
            "noncompiling_with_diagnostic_preferred": True,
            "code_diversity": "deterministic_max_min_normalized_token_ngram",
            "source_terminal_prefix_is_explicit": exploratory_prefix,
            "retry_parse_failures_or_truncations_only": (
                retry_source is not None
            ),
            "retry_source_schedule_order_preserved": (
                retry_source is not None
            ),
            "accepted_nontruncated_source_responses_regenerated": False,
        },
        "provider": _provider_contract(args, base_url),
        "budget": budget_contract,
        "verification": {
            "timeout": args.timeout,
            "stability_runs": args.stability_runs,
            "all_api_calls_before_any_private_gate": True,
            "visible_before_private": True,
            "private_gate_binary_only": True,
            "private_failure_triggers_api_call": False,
        },
        "privacy": {
            "api_input_fields": [
                "original_test_free_F2",
                "failed_student_code",
                "sanitized_compiler_diagnostic",
                "visible_training_tests_provider_only",
            ],
            "visible_training_tests_sent_to_provider": True,
            "visible_training_tests_in_training_outputs": False,
            "private_holdback_sent_to_provider": False,
            "gold_sent_to_provider": False,
            "api_credentials_persisted": False,
            "plaintext_reasoning_persisted": False,
        },
        "training_outputs": {
            "direct_original_f2_hard_targets": not evaluation_only,
            "exact_feedback_conditioned_repair_policy_targets": (
                not evaluation_only
            ),
            "same_verified_code_in_both_views": not evaluation_only,
            "exploratory_prefix": exploratory_prefix,
            "evaluation_only": evaluation_only,
            "training_use_forbidden": evaluation_only,
            "production_floor_eligible": production_floor_eligible,
            "may_count_toward_production_min_unique_targets": (
                production_floor_eligible
            ),
        },
        "heldout_175_opened": False,
    }
    _assert_secret_free(contract)
    events = load_journal(journal_path)
    if not events:
        _append_safe(
            journal_path,
            {
                "event": "header",
                "schema": JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
    else:
        validate_rescue_journal(events, contract=contract, plans=plans, slots=slots)
    api_key = str(os.environ.get(args.api_key_env) or "")
    if not api_key:
        raise RuntimeError(
            f"provider credential environment variable {args.api_key_env!r} is empty"
        )
    if transport is None:
        transport = _build_transport(args, api_key=api_key, base_url=base_url)
    slot_results = execute_api_phase(
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
        evaluate = _runtime_evaluator(
            timeout=args.timeout, stability_runs=args.stability_runs
        )
    verifications = execute_verification_phase(
        journal_path=journal_path,
        contract=contract,
        plans=plans,
        slots=slots,
        evaluate=evaluate,
        api_key=api_key,
    )
    final_state = validate_rescue_journal(
        load_journal(journal_path),
        contract=contract,
        plans=plans,
        slots=slots,
    )
    if not final_state["complete"]:
        raise RuntimeError("API rescue journal did not complete")
    contract_sha = canonical_sha256(contract)
    private_verified_candidates = sum(
        row.get("selected_target") is not None for row in verifications
    )
    outputs: dict[str, Any] | None = None
    if not evaluation_only:
        outputs = _publish_training_outputs(
            output_dir=output_dir,
            plans=plans,
            verifications=verifications,
            contract_sha256=contract_sha,
            exploratory_prefix=exploratory_prefix,
        )
    charged_input = sum(row["usage"]["charged_input_tokens"] for row in slot_results)
    charged_output = sum(row["usage"]["charged_output_tokens"] for row in slot_results)
    charged_nanos = sum(row["usage"]["charged_usd_nanos"] for row in slot_results)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": contract_sha,
        "execution_mode": contract["execution_mode"],
        "exploratory_prefix": exploratory_prefix,
        "evaluation_only": evaluation_only,
        "training_use_forbidden": evaluation_only,
        "production_floor_eligible": production_floor_eligible,
        "may_count_toward_production_min_unique_targets": (
            production_floor_eligible
        ),
        "local_source": contract["local_source"],
        "provider": contract["provider"],
        "schedule": {
            "eligible_all_zero_tasks_before_offset": len(all_eligible_plans),
            "eligible_all_zero_tasks_before_prior_success_exclusion": len(
                all_eligible_plans
            ),
            "prior_verified_tasks_excluded": len(prior_verified_order),
            "eligible_residual_tasks_before_offset": len(residual_plans),
            "residual_task_ids_sha256": canonical_sha256(residual_task_ids),
            "eligible_task_offset": args.eligible_task_offset,
            "eligible_task_offset_applied_after_deterministic_sort": True,
            "eligible_task_offset_applied_after_deterministic_sort_and_prior_success_exclusion": True,
            "eligible_all_zero_tasks_after_offset_before_task_cap": len(
                eligible_after_offset
            ),
            "eligible_all_zero_tasks_before_caps": len(plans_before_budget),
            "scheduled_tasks": len(plans),
            "scheduled_calls": len(slots),
            "task_ids_sha256": contract["selection"]["task_ids_sha256"],
            "provider_responses": sum(
                row.get("status") == "response" for row in slot_results
            ),
            "code_only_responses": sum(
                row.get("parse_accepted") is True for row in slot_results
            ),
        },
        "verification": {
            "visible_passes": sum(
                candidate["passed"]
                for row in verifications
                for candidate in row["visible_results"]
            ),
            "private_holdback_passes": private_verified_candidates,
            "verified_unique_hard_targets": (
                0
                if evaluation_only
                else len(outputs["rows"]["direct_targets"])
            ),
        },
        "budget_charged": {
            "calls": len(slot_results),
            "input_tokens": charged_input,
            "output_tokens": charged_output,
            "total_tokens": charged_input + charged_output,
            "estimated_usd_nanos": charged_nanos,
            "estimated_usd": f"{Decimal(charged_nanos) / Decimal(1_000_000_000):.9f}",
            "unknown_usage_failures_charged_at_full_reservation": True,
            "within_contract": (
                len(slot_results) <= args.max_calls
                and charged_nanos
                <= int(
                    (max_usd * Decimal(1_000_000_000)).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                )
            ),
        },
        "outputs": {} if evaluation_only else outputs["files"],
        "prior_success_exclusion": prior_exclusion_contract,
        "retry_parse_failures_or_truncations_source": (
            contract["retry_parse_failures_or_truncations_source"]
        ),
        "direct_manifest": None if evaluation_only else outputs["direct_manifest"],
        "repair_policy_manifest": (
            None if evaluation_only else outputs["repair_manifest"]
        ),
        "journal": journal_record(journal_path),
        "privacy_invariants": contract["privacy"],
        "heldout_175_opened": False,
    }
    _assert_secret_free(report, api_key=api_key)
    require_exact_or_write(output_dir / "api_rescue_report.json", report)
    print(
        json.dumps(
            {
                "tasks": len(plans),
                "calls": len(slots),
                "private_holdback_passes": private_verified_candidates,
                "verified_targets": (
                    0
                    if evaluation_only
                    else len(outputs["rows"]["direct_targets"])
                ),
                "estimated_usd": report["budget_charged"]["estimated_usd"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pilot_journal", required=True)
    parser.add_argument("--rollout_file", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--private_holdback", required=True)
    parser.add_argument("--expected_rollout_sha256", default="")
    parser.add_argument("--expected_f2_sha256", default="")
    parser.add_argument("--expected_private_holdback_sha256", default="")
    parser.add_argument("--allow_unpinned_inputs", action="store_true")
    parser.add_argument(
        "--evaluation_only",
        action="store_true",
        help=(
            "score a fixed provider arm without emitting trainable artifacts; "
            "the journal and report are sealed as forbidden for training use"
        ),
    )
    parser.add_argument(
        "--exploratory_terminal_prefix",
        type=int,
        default=0,
        help=(
            "explicit positive count of hash-chained local task terminals to "
            "freeze from an incomplete pilot; outputs are exploratory and "
            "can never satisfy the production target floor"
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--provider",
        choices=(
            "anthropic",
            "openai_responses",
            "openai_chat",
            "azure_responses",
            "azure_chat",
            "azure_v1_responses",
            "azure_v1_chat",
            "openrouter_chat",
        ),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_version", default="")
    parser.add_argument("--api_key_env", default="RS_SFT_API_KEY")
    parser.add_argument("--reasoning_effort", default="")
    parser.add_argument(
        "--openrouter_reasoning",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument(
        "--openrouter_provider_only",
        action="append",
        default=[],
        help="exact OpenRouter provider endpoint slug; repeat to allow more than one",
    )
    parser.add_argument(
        "--openrouter_provider_order",
        action="append",
        default=[],
        help=(
            "exact lowercase OpenRouter provider or provider/variant slug in "
            "routing priority order; repeat once per provider"
        ),
    )
    parser.add_argument(
        "--openrouter_allow_fallbacks",
        action="store_true",
        help=(
            "allow provider failover; --openrouter_provider_only remains the "
            "hard endpoint allowlist (off by default)"
        ),
    )
    parser.add_argument(
        "--openrouter_require_parameters",
        action="store_true",
        help="require the routed endpoint to support every request parameter",
    )
    parser.add_argument(
        "--openrouter_include_reasoning",
        action="store_true",
        help="request separate reasoning content; it is never journaled",
    )
    parser.add_argument(
        "--openrouter_reasoning_effort",
        choices=("low", "medium", "high", "xhigh"),
        default="",
        help="pin OpenRouter reasoning effort when reasoning is enabled",
    )
    parser.add_argument(
        "--openrouter_enforce_distillable_text",
        action="store_true",
        help="route only to endpoints whose outputs permit text distillation",
    )
    parser.add_argument(
        "--chat_token_parameter",
        choices=("max_completion_tokens", "max_tokens"),
        default="max_completion_tokens",
    )
    parser.add_argument(
        "--anthropic_thinking",
        choices=("adaptive", "disabled"),
        default="adaptive",
    )
    parser.add_argument(
        "--anthropic_effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retry_parse_failures_or_truncations_report",
        default="",
        help=(
            "completed production rescue report whose response parse failures "
            "and finish_reason=length slots must be retried in source order"
        ),
    )
    parser.add_argument(
        "--expected_retry_parse_failures_or_truncations_report_sha256",
        default="",
        help="exact SHA-256 of --retry_parse_failures_or_truncations_report",
    )
    parser.add_argument(
        "--expected_retry_parse_failures_or_truncations_tasks",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--expected_retry_parse_failures_or_truncations_task_ids_sha256",
        default="",
    )
    parser.add_argument(
        "--prior_success_report",
        action="append",
        default=[],
        help=(
            "completed production API rescue report whose exact verified "
            "targets must be excluded; repeat once per prior run"
        ),
    )
    parser.add_argument(
        "--expected_prior_success_report_sha256",
        action="append",
        default=[],
        help=(
            "exact SHA-256 paired by position with --prior_success_report; "
            "the report pins and the loader verifies its journal and outputs"
        ),
    )
    parser.add_argument(
        "--require_prior_schedules_disjoint",
        action="store_true",
        help="fail unless supplied prior rescue schedules are pairwise disjoint",
    )
    parser.add_argument(
        "--require_prior_schedule_complete_coverage",
        action="store_true",
        help=(
            "fail unless prior schedules exactly cover the current "
            "deterministic all-zero eligible cohort"
        ),
    )
    parser.add_argument("--expected_prior_scheduled_tasks", type=int, default=-1)
    parser.add_argument("--expected_prior_verified_tasks", type=int, default=-1)
    parser.add_argument("--expected_residual_tasks", type=int, default=-1)
    parser.add_argument("--expected_prior_scheduled_task_ids_sha256", default="")
    parser.add_argument("--expected_prior_verified_task_ids_sha256", default="")
    parser.add_argument("--expected_residual_task_ids_sha256", default="")
    parser.add_argument("--expected_scheduled_task_ids_sha256", default="")
    parser.add_argument(
        "--eligible_task_offset",
        type=int,
        default=0,
        help=(
            "skip this exact prefix after deterministic all-zero ordering "
            "and pinned prior-success exclusion, before --max_tasks"
        ),
    )
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--max_parents_per_task", type=int, default=2)
    parser.add_argument("--samples_per_parent", type=int, default=1)
    parser.add_argument("--max_calls", type=int, required=True)
    parser.add_argument("--max_input_tokens_per_call", type=int, default=65536)
    parser.add_argument("--max_output_tokens", type=int, default=16384)
    parser.add_argument("--max_input_tokens_total", type=int, default=0)
    parser.add_argument("--max_output_tokens_total", type=int, default=0)
    parser.add_argument("--max_total_tokens", type=int, default=0)
    parser.add_argument("--max_usd", type=str, required=True)
    parser.add_argument("--input_usd_per_million", type=str, required=True)
    parser.add_argument("--output_usd_per_million", type=str, required=True)
    parser.add_argument("--timeout_seconds", type=float, default=300.0)
    parser.add_argument("--inter_call_delay_seconds", type=float, default=0.0)
    parser.add_argument("--abort_on_provider_error", action="store_true")
    parser.add_argument("--provider_max_attempts", type=int, default=1)
    parser.add_argument("--provider_retry_base_seconds", type=float, default=1.0)
    parser.add_argument("--provider_retry_max_seconds", type=float, default=30.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=2)
    args = parser.parse_args(argv)
    if len(args.prior_success_report) != len(args.expected_prior_success_report_sha256):
        parser.error(
            "each --prior_success_report requires one matching "
            "--expected_prior_success_report_sha256"
        )
    if (
        args.seed < 0
        or args.eligible_task_offset < 0
        or args.max_tasks < 0
        or args.exploratory_terminal_prefix < 0
        or args.expected_prior_scheduled_tasks < -1
        or args.expected_prior_verified_tasks < -1
        or args.expected_residual_tasks < -1
        or args.expected_retry_parse_failures_or_truncations_tasks < -1
    ):
        parser.error(
            "seed, offsets, caps, and expected counts must be non-negative "
            "(expected counts may use -1 for unspecified)"
        )
    if args.prior_success_report and args.exploratory_terminal_prefix:
        parser.error(
            "prior production success exclusion cannot consume an exploratory "
            "local terminal prefix"
        )
    retry_mode = bool(args.retry_parse_failures_or_truncations_report)
    retry_assertions = bool(
        args.expected_retry_parse_failures_or_truncations_report_sha256
        or args.expected_retry_parse_failures_or_truncations_tasks >= 0
        or args.expected_retry_parse_failures_or_truncations_task_ids_sha256
    )
    if retry_mode:
        if (
            not args.expected_retry_parse_failures_or_truncations_report_sha256
            or args.expected_retry_parse_failures_or_truncations_tasks <= 0
            or not args.expected_retry_parse_failures_or_truncations_task_ids_sha256
        ):
            parser.error(
                "retry parse-failures-or-truncations mode requires exact "
                "source report SHA-256, positive task count, and task digest"
            )
        if args.prior_success_report or args.exploratory_terminal_prefix:
            parser.error(
                "retry parse-failures-or-truncations mode cannot be combined "
                "with prior-success exclusion or an exploratory local prefix"
            )
    elif retry_assertions:
        parser.error(
            "retry parse-failures-or-truncations assertions require the "
            "source report"
        )
    if (
        args.require_prior_schedules_disjoint
        or args.require_prior_schedule_complete_coverage
        or args.expected_prior_scheduled_tasks >= 0
        or args.expected_prior_verified_tasks >= 0
        or args.expected_residual_tasks >= 0
        or args.expected_prior_scheduled_task_ids_sha256
        or args.expected_prior_verified_task_ids_sha256
        or args.expected_residual_task_ids_sha256
    ) and not args.prior_success_report:
        parser.error("prior-success assertions require prior success reports")
    for name in (
        "expected_prior_success_report_sha256",
        "expected_prior_scheduled_task_ids_sha256",
        "expected_prior_verified_task_ids_sha256",
        "expected_residual_task_ids_sha256",
        "expected_scheduled_task_ids_sha256",
        "expected_retry_parse_failures_or_truncations_report_sha256",
        "expected_retry_parse_failures_or_truncations_task_ids_sha256",
    ):
        values = getattr(args, name)
        if isinstance(values, str):
            values = [values] if values else []
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
            parser.error(f"--{name} requires exact lowercase SHA-256 values")
    if args.max_parents_per_task <= 0 or args.samples_per_parent <= 0:
        parser.error("parent and sample counts must be positive")
    if retry_mode and (
        args.max_parents_per_task != 1 or args.samples_per_parent != 1
    ):
        parser.error(
            "retry parse-failures-or-truncations mode requires exactly one "
            "parent and one sample per task"
        )
    if (
        args.timeout_seconds <= 0
        or not math.isfinite(args.inter_call_delay_seconds)
        or args.inter_call_delay_seconds < 0
        or args.provider_max_attempts <= 0
        or not math.isfinite(args.provider_retry_base_seconds)
        or args.provider_retry_base_seconds <= 0
        or not math.isfinite(args.provider_retry_max_seconds)
        or args.provider_retry_max_seconds <= 0
        or args.provider_retry_max_seconds < args.provider_retry_base_seconds
        or args.timeout <= 0
        or args.stability_runs <= 0
    ):
        parser.error("timeouts and stability runs must be positive")
    if args.provider in {"azure_responses", "azure_chat"} and not args.api_version:
        parser.error(
            "dated AzureOpenAI-client modes require --api_version; use "
            "azure_v1_responses/azure_v1_chat for an /openai/v1 endpoint"
        )
    if args.provider in {"azure_v1_responses", "azure_v1_chat"}:
        if args.api_version:
            parser.error("Azure /openai/v1 modes do not accept --api_version")
        try:
            validate_provider_endpoint(
                provider=args.provider,
                base_url=args.base_url,
                api_version=args.api_version,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.provider == "openrouter_chat":
        if args.reasoning_effort:
            parser.error(
                "openrouter_chat uses --openrouter_reasoning, not "
                "--reasoning_effort"
            )
        if not args.openrouter_provider_only:
            parser.error(
                "openrouter_chat requires at least one "
                "--openrouter_provider_only"
            )
        if any(
            not re.fullmatch(
                r"[a-z0-9_.-]+(?:/[a-z0-9_.-]+)?", value
            )
            for value in args.openrouter_provider_only
        ):
            parser.error(
                "--openrouter_provider_only values must be exact lowercase "
                "provider or provider/variant slugs"
            )
        if len(set(args.openrouter_provider_only)) != len(
            args.openrouter_provider_only
        ):
            parser.error("--openrouter_provider_only values must be unique")
        if any(
            not re.fullmatch(
                r"[a-z0-9_.-]+(?:/[a-z0-9_.-]+)?", value
            )
            for value in args.openrouter_provider_order
        ):
            parser.error(
                "--openrouter_provider_order values must be exact lowercase "
                "provider or provider/variant slugs"
            )
        if len(set(args.openrouter_provider_order)) != len(
            args.openrouter_provider_order
        ):
            parser.error("--openrouter_provider_order values must be unique")
        if not set(args.openrouter_provider_order).issubset(
            args.openrouter_provider_only
        ):
            parser.error(
                "--openrouter_provider_order values must also appear in "
                "--openrouter_provider_only"
            )
        if not args.openrouter_require_parameters:
            parser.error(
                "openrouter_chat requires --openrouter_require_parameters"
            )
        if (
            not args.openrouter_enforce_distillable_text
            and not args.evaluation_only
        ):
            parser.error(
                "openrouter_chat requires "
                "--openrouter_enforce_distillable_text for RS-SFT outputs; "
                "non-distillable endpoints are valid only with "
                "--evaluation_only"
            )
        if args.openrouter_include_reasoning and args.openrouter_reasoning != "enabled":
            parser.error(
                "--openrouter_include_reasoning requires "
                "--openrouter_reasoning enabled"
            )
        if (
            args.openrouter_reasoning_effort
            and args.openrouter_reasoning != "enabled"
        ):
            parser.error(
                "--openrouter_reasoning_effort requires "
                "--openrouter_reasoning enabled"
            )
        try:
            validate_provider_endpoint(
                provider=args.provider,
                base_url=args.base_url,
                api_version=args.api_version,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif (
        args.openrouter_reasoning != "disabled"
        or args.openrouter_provider_only
        or args.openrouter_provider_order
        or args.openrouter_allow_fallbacks
        or args.openrouter_require_parameters
        or args.openrouter_include_reasoning
        or args.openrouter_reasoning_effort
        or args.openrouter_enforce_distillable_text
    ):
        parser.error(
            "OpenRouter-specific options are valid only with --provider "
            "openrouter_chat"
        )
    try:
        for value in (
            Decimal(args.max_usd),
            Decimal(args.input_usd_per_million),
            Decimal(args.output_usd_per_million),
        ):
            if not value.is_finite() or value <= 0:
                raise ValueError
    except Exception:
        parser.error("dollar cap and prices must be finite positive decimals")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
