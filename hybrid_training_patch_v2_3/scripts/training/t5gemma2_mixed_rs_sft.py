#!/usr/bin/env python3
"""Fit a mixed, provenance-sealed T5Gemma 2 RS-SFT adapter.

The stage starts from an already-trained native T5Gemma 2 encoder-decoder
LoRA checkpoint and mixes three kinds of supervision:

* privately verified direct hard targets on the original sealed F2 source;
* privately verified repairs on the exact compiler-feedback-conditioned
  encoder source used to produce the repair; and
* a configurable deterministic replay of original TRAIN-only gold pairs.

Only completed local/API rescue reports with known schemas are accepted.
Every referenced file is hash-bound, the 175-task development identity set is
used only as a deny-list, and no test, private diagnostic, or gold source is
serialized into an encoder input.  API prefix probes are rejected unless the
caller explicitly marks the resulting run exploratory.

``--warmstart_checkpoint`` initializes adapter weights only, from either the
sealed enriched-SFT checkpoint or a sealed checkpoint produced by an earlier
mixed RS-SFT pass.  A separate ``--resume_checkpoint`` resumes this exact
stage's optimizer/scheduler/RNG state.  This distinction prevents a parent
stage's optimizer and epoch position from being mistaken for a same-contract
resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from scripts.evaluation.durable_evaluation_journal import (
    journal_record,
    load_journal,
)
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training.seq2seq_verpo_core import (
    COMPILER_REPAIR_MARKER,
    COMPILER_REPAIR_SCHEMA,
    canonical_json,
    sha256_text,
)


RUN_SCHEMA = "t5gemma2-mixed-rs-sft-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-mixed-rs-sft-checkpoint-v1"
DATASET_SCHEMA = "t5gemma2-mixed-rs-sft-dataset-v1"
LOCAL_REPORT_SCHEMA = "t5gemma2-local-rs-sft-pilot-report-v1"
API_REPORT_SCHEMA = "t5gemma2-api-rs-sft-rescue-report-v1"
API_DIRECT_MANIFEST_SCHEMA = "t5gemma2-api-rs-sft-direct-manifest-v1"
API_REPAIR_MANIFEST_SCHEMA = "t5gemma2-api-rs-sft-repair-policy-manifest-v1"
API_DIRECT_TARGET_SCHEMA = "t5gemma2-api-rs-sft-direct-target-v1"
API_REPAIR_PAIR_SCHEMA = "t5gemma2-api-rs-sft-repair-policy-pair-v1"
COMPILED_REPAIR_SCHEMA = "t5gemma2-compiled-failure-repair-context-v1"
COMPILED_REPAIR_OPEN = "<COMPILER_REPAIR_CONTEXT_JSON>\n"
COMPILED_REPAIR_CLOSE = "\n</COMPILER_REPAIR_CONTEXT_JSON>"
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_RE = re.compile(r"checkpoint-optstep-([0-9]{6,})\Z")


@dataclass(frozen=True)
class WarmstartIdentity:
    checkpoint_name: str
    update: int
    run_contract_sha256: str
    adapter_weights_sha256: str
    adapter_config_sha256: str
    model: str
    model_revision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    exact_lora_targets: tuple[str, ...]


@dataclass(frozen=True)
class MixedPair:
    pair_id: str
    source_task_id: str
    kind: str
    source: str
    target: str
    source_sha256: str
    target_sha256: str
    provenance: tuple[tuple[str, str], ...]

    def as_text_pair(self) -> base_sft.TextPair:
        return base_sft.TextPair(
            task_id=self.pair_id,
            source=self.source,
            target=self.target,
            source_sha256=self.source_sha256,
            target_sha256=self.target_sha256,
        )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if not _HEX_SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root is not an object")
    return value


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank rows are forbidden")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"{path}: cannot read JSONL") from exc
    if not rows and not allow_empty:
        raise ValueError(f"{path}: no rows")
    return rows


def _parse_pinned_specs(values: Sequence[str], *, label: str) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{label} must be SHA256=PATH, got {raw!r}")
        expected, raw_path = raw.split("=", 1)
        expected = _require_sha256(expected, f"{label} expected digest")
        path = Path(raw_path).expanduser().resolve()
        if path in seen:
            raise ValueError(f"{label} contains duplicate path {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        observed = base_sft.sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"{label} digest mismatch for {path}: "
                f"expected={expected}, observed={observed}"
            )
        seen.add(path)
        result.append((path, observed))
    return result


def _resolve_output_record(
    report_dir: Path,
    record: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label}: file record is missing")
    expected_sha = _require_sha256(record.get("sha256"), f"{label}.sha256")
    try:
        expected_rows = int(record.get("rows", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.rows is malformed") from exc
    if expected_rows < 0 or (expected_rows == 0 and not allow_empty):
        raise ValueError(f"{label}.rows is invalid")
    recorded = Path(str(record.get("path") or ""))
    candidates: list[Path] = []
    if recorded.is_absolute() and recorded.is_file():
        candidates.append(recorded.resolve())
    if recorded.name:
        local = (report_dir / recorded.name).resolve()
        if local.is_file() and local not in candidates:
            candidates.append(local)
    if len(candidates) != 1:
        raise ValueError(
            f"{label}: expected exactly one local artifact for recorded "
            f"path {recorded!s}, found={candidates}"
        )
    path = candidates[0]
    if base_sft.sha256_file(path) != expected_sha:
        raise ValueError(f"{label}: referenced artifact digest mismatch")
    rows = _read_jsonl(path, allow_empty=allow_empty)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{label}: row count mismatch; manifest={expected_rows}, "
            f"file={len(rows)}"
        )
    return path, rows


def _validate_journal_record(
    *,
    report_dir: Path,
    record: Any,
    expected_schema: str,
    expected_contract_sha256: str,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label}: journal record is absent")
    recorded_path = Path(str(record.get("path") or ""))
    local_path = (report_dir / recorded_path.name).resolve()
    if not recorded_path.name or not local_path.is_file():
        raise ValueError(f"{label}: local journal artifact is absent")
    actual = journal_record(local_path)
    for field in (
        "sha256",
        "chain_head_sha256",
        "event_count",
        "head_event_sha256",
    ):
        if actual.get(field) != record.get(field):
            raise ValueError(f"{label}: journal record field {field} differs")
    recorded_head = Path(str(record.get("chain_head_path") or ""))
    if recorded_head.name != Path(str(actual["chain_head_path"])).name:
        raise ValueError(f"{label}: chain-head filename differs")
    events = load_journal(local_path)
    if (
        not events
        or events[0].get("event") != "header"
        or events[0].get("schema") != expected_schema
        or events[0].get("contract_sha256") != expected_contract_sha256
        or events[-1].get("event") != "complete"
        or events[-1].get("schema") != expected_schema
    ):
        raise ValueError(f"{label}: journal header/completion binding failed")
    return events


def _task_id(row: Mapping[str, Any], label: str) -> str:
    value = str(row.get("task_id") or "").strip()
    if not value:
        raise ValueError(f"{label}: task_id is absent")
    return value


def _target(row: Mapping[str, Any], label: str) -> str:
    value = str(row.get("dart_source") or "").strip()
    if not value:
        raise ValueError(f"{label}: Dart target is absent")
    return value


def _provenance_tuple(**items: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in items.items()))


def _make_pair(
    *,
    pair_id: str,
    source_task_id: str,
    kind: str,
    source: str,
    target: str,
    provenance: tuple[tuple[str, str], ...],
) -> MixedPair:
    if (
        not pair_id
        or not source_task_id
        or kind
        not in {
            "verified_direct",
            "repair_conditioned",
            "gold_replay",
        }
    ):
        raise ValueError("mixed pair identity/kind is malformed")
    if not source.strip() or not target.strip():
        raise ValueError(f"{pair_id}: source or target is empty")
    return MixedPair(
        pair_id=pair_id,
        source_task_id=source_task_id,
        kind=kind,
        source=source,
        target=target,
        source_sha256=sha256_text(source),
        target_sha256=sha256_text(target),
        provenance=provenance,
    )


def _validate_local_report(
    *,
    report_path: Path,
    report_sha256: str,
    warmstart: WarmstartIdentity,
) -> tuple[list[MixedPair], dict[str, Any]]:
    report = _read_json(report_path)
    privacy = report.get("privacy_invariants")
    pilot = report.get("pilot")
    checkpoint = report.get("checkpoint")
    adapter = checkpoint.get("adapter") if isinstance(checkpoint, Mapping) else None
    report_contract_sha = _require_sha256(
        report.get("run_contract_sha256"), "local report run contract"
    )
    if (
        report.get("schema") != LOCAL_REPORT_SCHEMA
        or report.get("status") != "complete"
        or not isinstance(privacy, Mapping)
        or privacy.get("heldout_175_opened") is not False
        or privacy.get("frontier_api_calls") is not False
        or privacy.get("private_holdback_text_in_model_input") is not False
        or privacy.get("private_holdback_text_in_outputs") is not False
        or privacy.get("private_diagnostics_persisted") is not False
        or privacy.get("all_generation_precedes_private_gate_per_task") is not True
        or privacy.get("private_gate_can_only_reject_transfer") is not True
        or not isinstance(pilot, Mapping)
        or not isinstance(adapter, Mapping)
        or adapter.get("run_contract_sha256") != warmstart.run_contract_sha256
        or adapter.get("adapter_weights_sha256") != warmstart.adapter_weights_sha256
        or adapter.get("adapter_config_sha256") != warmstart.adapter_config_sha256
        or checkpoint.get("name") != warmstart.model
        or checkpoint.get("revision") != warmstart.model_revision
    ):
        raise ValueError(
            f"{report_path}: local rescue report failed completion, privacy, "
            "or warm-start binding"
        )
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{report_path}: local outputs are absent")
    _target_path, target_rows = _resolve_output_record(
        report_path.parent,
        outputs.get("repairs"),
        label="local.outputs.repairs",
        allow_empty=True,
    )
    _f2_path, f2_rows = _resolve_output_record(
        report_path.parent,
        outputs.get("repairs_f2"),
        label="local.outputs.repairs_f2",
        allow_empty=True,
    )
    if len(target_rows) != len(f2_rows):
        raise ValueError(f"{report_path}: local direct target/F2 counts differ")
    expected_accepted = int(pilot.get("accepted_unique_targets", -1))
    if expected_accepted != len(target_rows):
        raise ValueError(f"{report_path}: local accepted count differs")
    events = _validate_journal_record(
        report_dir=report_path.parent,
        record=report.get("journal"),
        expected_schema="t5gemma2-local-rs-sft-pilot-journal-v1",
        expected_contract_sha256=report_contract_sha,
        label="local.journal",
    )
    expected_tasks = int(pilot.get("tasks", -1))
    terminals = [event for event in events if event.get("event") == "task_terminal"]
    if (
        expected_tasks <= 0
        or len(terminals) != expected_tasks
        or int(events[-1].get("tasks", -1)) != expected_tasks
    ):
        raise ValueError(f"{report_path}: local journal task accounting differs")
    journal_targets: dict[str, tuple[str, str]] = {}
    for terminal in terminals:
        selected = terminal.get("selected_target")
        if selected is None:
            continue
        if (
            not isinstance(selected, Mapping)
            or selected.get("schema") != "t5gemma2-local-rs-sft-target-v1"
            or selected.get("visible_passed") is not True
            or selected.get("private_gate_passed") is not True
        ):
            raise ValueError(f"{report_path}: local journal target is unverified")
        task_id = _task_id(selected, "local journal target")
        code = str(selected.get("code") or "")
        selected_source_sha = _require_sha256(
            selected.get("source_sha256"), "local selected source"
        )
        if (
            not code.strip()
            or sha256_text(code) != selected.get("code_sha256")
            or task_id in journal_targets
        ):
            raise ValueError(f"{report_path}: local journal target hash differs")
        journal_targets[task_id] = (code, selected_source_sha)
    if len(journal_targets) != expected_accepted:
        raise ValueError(f"{report_path}: local journal accepted count differs")

    pairs: list[MixedPair] = []
    seen: set[str] = set()
    for index, (target_row, f2_row) in enumerate(
        zip(target_rows, f2_rows, strict=True)
    ):
        task_id = _task_id(target_row, f"local target {index}")
        if task_id in seen or _task_id(f2_row, f"local F2 {index}") != task_id:
            raise ValueError(f"{report_path}: local task IDs duplicate or drift")
        seen.add(task_id)
        source = base_sft.build_encoder_source(f2_row, task_id)
        target = _target(target_row, f"local target {index}")
        if journal_targets.get(task_id) != (target, sha256_text(source)):
            raise ValueError(
                f"{report_path}: local output is not the journal-selected target"
            )
        target_sha = sha256_text(target)
        pairs.append(
            _make_pair(
                pair_id=(
                    f"{task_id}::direct::local::"
                    f"{report_sha256[:12]}::{target_sha[:12]}"
                ),
                source_task_id=task_id,
                kind="verified_direct",
                source=source,
                target=target,
                provenance=_provenance_tuple(
                    report_schema=LOCAL_REPORT_SCHEMA,
                    report_sha256=report_sha256,
                ),
            )
        )
    record = {
        "schema": LOCAL_REPORT_SCHEMA,
        "sha256": report_sha256,
        "accepted_direct_rows": len(pairs),
        "production_floor_met": pilot.get("production_floor_met") is True,
        "private_gate_bound": True,
        "warmstart_bound": True,
    }
    return pairs, record


def _parse_feedback_context(
    *,
    source_task_id: str,
    original_source: str,
    encoder_source: str,
    source_row: Mapping[str, Any],
) -> None:
    if not encoder_source.startswith(original_source + "\n"):
        raise ValueError(
            f"{source_task_id}: repair source does not extend the exact F2 input"
        )
    suffix = encoder_source[len(original_source) + 1 :]
    if suffix.startswith(COMPILER_REPAIR_MARKER + "\n"):
        payload_text = suffix[len(COMPILER_REPAIR_MARKER) + 1 :]
        if not payload_text.endswith("\n") or payload_text.count("\n") != 1:
            raise ValueError(
                f"{source_task_id}: compiler repair context is not canonical"
            )
        payload_text = payload_text[:-1]
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source_task_id}: compiler repair payload is malformed"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != COMPILER_REPAIR_SCHEMA
            or payload.get("feedback_kind") != "compiler_only"
            or payload.get("task_id") != source_task_id
            or payload.get("source_sha256")
            != source_row.get("original_f2_source_sha256")
            or payload.get("compiler_failed") is not True
            or canonical_json(payload) != payload_text
        ):
            raise ValueError(
                f"{source_task_id}: compiler repair payload contract failed"
            )
        candidate = str(payload.get("candidate") or "")
        feedback = str(payload.get("compiler_feedback") or "")
        if (
            not candidate.strip()
            or sha256_text(candidate) != payload.get("candidate_sha256")
            or sha256_text(candidate) != source_row.get("parent_code_sha256")
            or sha256_text(feedback) != payload.get("compiler_feedback_sha256")
            or feedback != source_row.get("compiler_diagnostic")
            or sha256_text(feedback) != source_row.get("compiler_diagnostic_sha256")
        ):
            raise ValueError(f"{source_task_id}: compiler repair hashes/content differ")
        return

    if suffix.startswith(COMPILED_REPAIR_OPEN) and suffix.endswith(
        COMPILED_REPAIR_CLOSE
    ):
        payload_text = suffix[len(COMPILED_REPAIR_OPEN) : -len(COMPILED_REPAIR_CLOSE)]
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source_task_id}: compiled repair payload is malformed"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != COMPILED_REPAIR_SCHEMA
            or payload.get("task_id") != source_task_id
            or payload.get("source_sha256")
            != source_row.get("original_f2_source_sha256")
            or payload.get("compiled") is not True
            or payload.get("tests_visible") is not False
            or payload.get("private_holdback_visible") is not False
            or canonical_json(payload) != payload_text
        ):
            raise ValueError(
                f"{source_task_id}: compiled repair payload contract failed"
            )
        candidate = str(payload.get("candidate") or "")
        feedback = str(payload.get("compiler_feedback") or "")
        if (
            not candidate.strip()
            or sha256_text(candidate) != payload.get("candidate_sha256")
            or sha256_text(candidate) != source_row.get("parent_code_sha256")
            or sha256_text(feedback) != payload.get("compiler_feedback_sha256")
            or feedback != source_row.get("compiler_diagnostic")
            or sha256_text(feedback) != source_row.get("compiler_diagnostic_sha256")
        ):
            raise ValueError(f"{source_task_id}: compiled repair hashes/content differ")
        return
    raise ValueError(f"{source_task_id}: unknown repair-conditioning schema")


def _validate_api_report(
    *,
    report_path: Path,
    report_sha256: str,
    allow_exploratory: bool,
) -> tuple[list[MixedPair], dict[str, Any]]:
    report = _read_json(report_path)
    privacy = report.get("privacy_invariants")
    source_record = report.get("local_source")
    direct_manifest = report.get("direct_manifest")
    repair_manifest = report.get("repair_policy_manifest")
    exploratory = report.get("exploratory_prefix") is True
    if (
        report.get("schema") != API_REPORT_SCHEMA
        or report.get("status") != "complete"
        or not isinstance(privacy, Mapping)
        or privacy.get("api_credentials_persisted") is not False
        or privacy.get("gold_sent_to_provider") is not False
        or privacy.get("plaintext_reasoning_persisted") is not False
        or privacy.get("private_holdback_sent_to_provider") is not False
        or privacy.get("visible_training_tests_in_training_outputs") is not False
        or privacy.get("visible_training_tests_sent_to_provider") is not True
        or privacy.get("api_input_fields")
        != [
            "original_test_free_F2",
            "failed_student_code",
            "sanitized_compiler_diagnostic",
            "visible_training_tests_provider_only",
        ]
        or report.get("heldout_175_opened") is not False
        or not isinstance(source_record, Mapping)
        or source_record.get("source_journal_modified") is not False
        or source_record.get("exploratory_prefix") is not exploratory
        or source_record.get("production_floor_eligible") is not (not exploratory)
        or report.get("production_floor_eligible") is not (not exploratory)
        or report.get("may_count_toward_production_min_unique_targets")
        is not (not exploratory)
        or not isinstance(direct_manifest, Mapping)
        or not isinstance(repair_manifest, Mapping)
    ):
        raise ValueError(
            f"{report_path}: API rescue completion/privacy/source contract failed"
        )
    if exploratory and not allow_exploratory:
        raise ValueError(
            f"{report_path}: exploratory API prefix is forbidden; use "
            "--allow_exploratory_inputs only for a clearly labelled pilot"
        )

    standalone_direct = _read_json(report_path.parent / "direct_manifest.json")
    standalone_repair = _read_json(report_path.parent / "repair_policy_manifest.json")
    if standalone_direct != direct_manifest or standalone_repair != repair_manifest:
        raise ValueError(f"{report_path}: standalone/embedded API manifests differ")
    run_contract_sha = _require_sha256(
        report.get("run_contract_sha256"), "API run contract"
    )
    api_events = _validate_journal_record(
        report_dir=report_path.parent,
        record=report.get("journal"),
        expected_schema="t5gemma2-api-rs-sft-rescue-journal-v1",
        expected_contract_sha256=run_contract_sha,
        label="api.journal",
    )
    journal_selected: dict[str, Mapping[str, Any]] = {}
    for event in api_events:
        if event.get("event") != "task_verification":
            continue
        if (
            event.get("all_api_generation_completed_before_private_gate") is not True
            or event.get("holdback_failure_triggers_generation") is not False
            or event.get("private_diagnostics_persisted") is not False
            or event.get("private_feedback_serialized_to_model") is not False
        ):
            raise ValueError(f"{report_path}: API verification ordering leaked")
        selected = event.get("selected_target")
        if selected is None:
            continue
        if (
            not isinstance(selected, Mapping)
            or selected.get("schema") != API_DIRECT_TARGET_SCHEMA
            or selected.get("visible_passed") is not True
            or selected.get("private_gate_passed") is not True
            or selected.get("exploratory_prefix") is not exploratory
            or selected.get("production_floor_eligible") is not (not exploratory)
        ):
            raise ValueError(f"{report_path}: API journal target is unverified")
        task_id = _task_id(selected, "API journal selected target")
        code = str(selected.get("code") or "")
        if (
            task_id in journal_selected
            or not code.strip()
            or sha256_text(code) != selected.get("code_sha256")
            or selected.get("task_id") != event.get("task_id")
            or selected.get("source_sha256") != event.get("source_sha256")
        ):
            raise ValueError(f"{report_path}: API journal target binding differs")
        journal_selected[task_id] = selected
    if (
        direct_manifest.get("schema") != API_DIRECT_MANIFEST_SCHEMA
        or direct_manifest.get("run_contract_sha256") != run_contract_sha
        or direct_manifest.get("mapping")
        != "original_sealed_F2_to_visible_and_private_verified_Dart"
        or direct_manifest.get("compatible_trainer") != "t5gemma2_enriched_sft.py"
        or direct_manifest.get("unique_source_tasks") is not True
        or direct_manifest.get("exploratory_prefix") is not exploratory
        or direct_manifest.get("production_floor_eligible") is not (not exploratory)
        or direct_manifest.get("may_count_toward_production_min_unique_targets")
        is not (not exploratory)
        or repair_manifest.get("schema") != API_REPAIR_MANIFEST_SCHEMA
        or repair_manifest.get("run_contract_sha256") != run_contract_sha
        or repair_manifest.get("source_is_exact_model_input") is not True
        or repair_manifest.get("requires_prebuilt_encoder_source_loader") is not True
        or repair_manifest.get("private_feedback_present") is not False
        or repair_manifest.get("exploratory_prefix") is not exploratory
        or repair_manifest.get("production_floor_eligible") is not (not exploratory)
        or repair_manifest.get("may_count_toward_production_min_unique_targets")
        is not (not exploratory)
    ):
        raise ValueError(f"{report_path}: API training manifests failed")

    _direct_path, direct_rows = _resolve_output_record(
        report_path.parent,
        direct_manifest.get("targets"),
        label="api.direct.targets",
        allow_empty=True,
    )
    _f2_path, f2_rows = _resolve_output_record(
        report_path.parent,
        direct_manifest.get("f2"),
        label="api.direct.f2",
        allow_empty=True,
    )
    _repair_source_path, repair_source_rows = _resolve_output_record(
        report_path.parent,
        repair_manifest.get("prebuilt_encoder_sources"),
        label="api.repair.sources",
        allow_empty=True,
    )
    _repair_target_path, repair_target_rows = _resolve_output_record(
        report_path.parent,
        repair_manifest.get("targets"),
        label="api.repair.targets",
        allow_empty=True,
    )
    if (
        len(direct_rows) != len(f2_rows)
        or len(repair_source_rows) != len(repair_target_rows)
        or len(direct_rows) != len(repair_target_rows)
        or int(direct_manifest.get("rows", -1)) != len(direct_rows)
        or int(repair_manifest.get("rows", -1)) != len(repair_target_rows)
    ):
        raise ValueError(f"{report_path}: API view row counts differ")

    pairs: list[MixedPair] = []
    direct_by_task: dict[str, tuple[str, str]] = {}
    for index, (target_row, f2_row) in enumerate(
        zip(direct_rows, f2_rows, strict=True)
    ):
        task_id = _task_id(target_row, f"API direct target {index}")
        if task_id in direct_by_task or _task_id(f2_row, f"API F2 {index}") != task_id:
            raise ValueError(f"{report_path}: API direct task IDs duplicate/drift")
        source = base_sft.build_encoder_source(f2_row, task_id)
        target = _target(target_row, f"API direct target {index}")
        journal_target = journal_selected.get(task_id)
        if (
            target_row.get("schema") != API_DIRECT_TARGET_SCHEMA
            or target_row.get("run_contract_sha256") is not None
            or target_row.get("source_sha256") != sha256_text(source)
            or target_row.get("dart_source_sha256") != sha256_text(target)
            or target_row.get("visible_passed") is not True
            or target_row.get("private_gate_passed") is not True
            or target_row.get("exploratory_prefix") is not exploratory
            or target_row.get("production_floor_eligible") is not (not exploratory)
            or not isinstance(target_row.get("provenance"), Mapping)
            or target_row["provenance"].get("run_contract_sha256") != run_contract_sha
            or not isinstance(journal_target, Mapping)
            or journal_target.get("code") != target
            or journal_target.get("code_sha256") != sha256_text(target)
            or journal_target.get("source_sha256") != sha256_text(source)
        ):
            raise ValueError(f"{report_path}: API direct row {index} failed")
        direct_by_task[task_id] = (source, target)
        pairs.append(
            _make_pair(
                pair_id=(
                    f"{task_id}::direct::api::"
                    f"{report_sha256[:12]}::{sha256_text(target)[:12]}"
                ),
                source_task_id=task_id,
                kind="verified_direct",
                source=source,
                target=target,
                provenance=_provenance_tuple(
                    report_schema=API_REPORT_SCHEMA,
                    report_sha256=report_sha256,
                    view="direct",
                ),
            )
        )

    seen_repair_ids: set[str] = set()
    for index, (source_row, target_row) in enumerate(
        zip(repair_source_rows, repair_target_rows, strict=True)
    ):
        pair_id = _task_id(source_row, f"API repair source {index}")
        source_task_id = str(source_row.get("source_task_id") or "").strip()
        if (
            pair_id in seen_repair_ids
            or target_row.get("task_id") != pair_id
            or target_row.get("source_task_id") != source_task_id
            or source_task_id not in direct_by_task
            or source_row.get("schema") != API_REPAIR_PAIR_SCHEMA
            or target_row.get("schema") != API_REPAIR_PAIR_SCHEMA
            or source_row.get("private_feedback_present") is not False
            or source_row.get("tests_present") is not False
            or source_row.get("gold_present") is not False
            or source_row.get("exploratory_prefix") is not exploratory
            or target_row.get("exploratory_prefix") is not exploratory
            or source_row.get("production_floor_eligible") is not (not exploratory)
            or target_row.get("production_floor_eligible") is not (not exploratory)
        ):
            raise ValueError(f"{report_path}: API repair row {index} failed")
        seen_repair_ids.add(pair_id)
        encoder_source = str(source_row.get("encoder_source") or "")
        target = _target(target_row, f"API repair target {index}")
        original_source, direct_target = direct_by_task[source_task_id]
        if (
            sha256_text(encoder_source) != source_row.get("encoder_source_sha256")
            or sha256_text(original_source)
            != source_row.get("original_f2_source_sha256")
            or sha256_text(target) != target_row.get("dart_source_sha256")
            or target != direct_target
            or journal_selected[source_task_id].get("feedback_source_sha256")
            != sha256_text(encoder_source)
            or journal_selected[source_task_id].get("parent_code_sha256")
            != source_row.get("parent_code_sha256")
            or journal_selected[source_task_id].get("diagnostic_sha256")
            != source_row.get("compiler_diagnostic_sha256")
        ):
            raise ValueError(
                f"{report_path}: API repair source/target hash binding failed"
            )
        _parse_feedback_context(
            source_task_id=source_task_id,
            original_source=original_source,
            encoder_source=encoder_source,
            source_row=source_row,
        )
        pairs.append(
            _make_pair(
                pair_id=pair_id,
                source_task_id=source_task_id,
                kind="repair_conditioned",
                source=encoder_source,
                target=target,
                provenance=_provenance_tuple(
                    report_schema=API_REPORT_SCHEMA,
                    report_sha256=report_sha256,
                    view="repair_conditioned",
                ),
            )
        )

    verified = report.get("verification")
    if (
        not isinstance(verified, Mapping)
        or int(verified.get("verified_unique_hard_targets", -1)) != len(direct_rows)
        or len(journal_selected) != len(direct_rows)
        or int(api_events[-1].get("verified_targets", -1)) != len(direct_rows)
        or api_events[-1].get("exploratory_prefix") is not exploratory
        or api_events[-1].get("production_floor_eligible") is not (not exploratory)
    ):
        raise ValueError(f"{report_path}: API verified target count differs")
    record = {
        "schema": API_REPORT_SCHEMA,
        "sha256": report_sha256,
        "provider": dict(report.get("provider") or {}),
        "direct_rows": len(direct_rows),
        "repair_conditioned_rows": len(repair_target_rows),
        "exploratory_prefix": exploratory,
        "production_floor_eligible": not exploratory,
        "private_gate_bound": True,
    }
    # Provider metadata can contain endpoint/model configuration but must not
    # contain a credential. The producer promises this; enforce it recursively.
    serialized_record = json.dumps(record, sort_keys=True).lower()
    if "api_key" in serialized_record or "secret" in serialized_record:
        raise ValueError(f"{report_path}: provider record appears to contain a secret")
    return pairs, record


def _load_heldout_ids(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[set[str], dict[str, Any]]:
    observed = base_sft._pin_file(
        path,
        expected_sha256,
        allow_unpinned=False,
    )
    rows = _read_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(
            f"held-out identity audit expected {expected_rows} rows, "
            f"observed {len(rows)}"
        )
    ids = [_task_id(row, f"held-out row {index}") for index, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError("held-out identity audit contains duplicate task IDs")
    return set(ids), {
        "sha256": observed,
        "rows": len(ids),
        "task_ids_sha256": _canonical_sha256(ids),
        "identity_fields_read": ["task_id"],
        "content_model_visible": False,
    }


def _deduplicate_pairs(pairs: Sequence[MixedPair]) -> list[MixedPair]:
    by_content: dict[tuple[str, str, str], MixedPair] = {}
    # An exact encoder input must have exactly one accepted target.  Merging
    # byte-identical copies from independently sealed reports is fine, but
    # silently teaching two different programs for one source is not.  This
    # matters most for direct F2 targets when a later rescue is accidentally
    # run over an already-covered task.
    target_by_input: dict[tuple[str, str], str] = {}
    pair_ids: set[str] = set()
    for pair in pairs:
        if pair.pair_id in pair_ids:
            raise ValueError(f"duplicate mixed pair ID: {pair.pair_id}")
        pair_ids.add(pair.pair_id)
        input_key = (pair.kind, pair.source_sha256)
        prior_target = target_by_input.get(input_key)
        if prior_target is not None and prior_target != pair.target_sha256:
            raise ValueError(
                "one mixed encoder input has conflicting verified targets; "
                "select a single source before final training"
            )
        target_by_input[input_key] = pair.target_sha256
        key = (pair.kind, pair.source_sha256, pair.target_sha256)
        existing = by_content.get(key)
        if existing is None:
            by_content[key] = pair
            continue
        if existing.source_task_id != pair.source_task_id:
            raise ValueError(
                "identical mixed source/target hashes claim different tasks"
            )
        merged_provenance = tuple(
            sorted(set(existing.provenance) | set(pair.provenance))
        )
        by_content[key] = MixedPair(
            pair_id=min(existing.pair_id, pair.pair_id),
            source_task_id=existing.source_task_id,
            kind=existing.kind,
            source=existing.source,
            target=existing.target,
            source_sha256=existing.source_sha256,
            target_sha256=existing.target_sha256,
            provenance=merged_provenance,
        )
    return list(by_content.values())


def build_mixed_pairs(
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
    warmstart: WarmstartIdentity,
    gold_replay_ratio: float,
    gold_replay_rows: int,
    min_verified_direct_targets: int,
    min_repair_conditioned_targets: int,
    allow_exploratory_inputs: bool,
    require_local_production_floor: bool,
    seed: int,
) -> tuple[list[MixedPair], dict[str, Any]]:
    """Validate all reports and build one deterministic, test-free schedule."""

    if not local_reports and not api_reports:
        raise ValueError("at least one completed local/API rescue report is required")
    if gold_replay_ratio < 0.0 or not math.isfinite(gold_replay_ratio):
        raise ValueError("gold replay ratio must be finite and non-negative")
    if gold_replay_rows < -1:
        raise ValueError("gold replay rows must be -1 or non-negative")
    if min_verified_direct_targets < 0 or min_repair_conditioned_targets < 0:
        raise ValueError("minimum target counts must be non-negative")

    gold_pairs, gold_manifest = base_sft.load_text_pairs(
        gold_train_jsonl,
        gold_f2_jsonl,
        expected_dataset_sha256=expected_gold_train_sha256,
        expected_f2_sha256=expected_gold_f2_sha256,
        expected_rows=expected_gold_rows,
        allow_unpinned_inputs=False,
    )
    heldout_ids, heldout_record = _load_heldout_ids(
        heldout_jsonl,
        expected_sha256=expected_heldout_sha256,
        expected_rows=expected_heldout_rows,
    )
    gold_ids = [pair.task_id for pair in gold_pairs]
    leaked_gold = sorted(set(gold_ids) & heldout_ids)
    if leaked_gold:
        raise ValueError(
            f"gold TRAIN universe overlaps held-out identities: {leaked_gold[:10]}"
        )

    collected: list[MixedPair] = []
    report_records: list[dict[str, Any]] = []
    for path, digest in local_reports:
        pairs, record = _validate_local_report(
            report_path=path,
            report_sha256=digest,
            warmstart=warmstart,
        )
        if require_local_production_floor and not record["production_floor_met"]:
            raise ValueError(f"{path}: local production target floor was not met")
        collected.extend(pairs)
        report_records.append(record)
    for path, digest in api_reports:
        pairs, record = _validate_api_report(
            report_path=path,
            report_sha256=digest,
            allow_exploratory=allow_exploratory_inputs,
        )
        collected.extend(pairs)
        report_records.append(record)

    rescue_pairs = _deduplicate_pairs(collected)
    gold_id_set = set(gold_ids)
    unknown = sorted(
        {
            pair.source_task_id
            for pair in rescue_pairs
            if pair.source_task_id not in gold_id_set
        }
    )
    heldout_leaks = sorted(
        {
            pair.source_task_id
            for pair in rescue_pairs
            if pair.source_task_id in heldout_ids
        }
    )
    if unknown:
        raise ValueError(
            f"rescue targets are outside the sealed TRAIN universe: {unknown[:10]}"
        )
    if heldout_leaks:
        raise ValueError(
            f"rescue targets overlap held-out identities: {heldout_leaks[:10]}"
        )
    direct_count = sum(pair.kind == "verified_direct" for pair in rescue_pairs)
    repair_count = sum(pair.kind == "repair_conditioned" for pair in rescue_pairs)
    if direct_count < min_verified_direct_targets:
        raise ValueError(
            f"verified direct target floor not met: {direct_count} < "
            f"{min_verified_direct_targets}"
        )
    if repair_count < min_repair_conditioned_targets:
        raise ValueError(
            f"repair-conditioned target floor not met: {repair_count} < "
            f"{min_repair_conditioned_targets}"
        )

    rescue_task_ids = {pair.source_task_id for pair in rescue_pairs}
    replay_pool = [pair for pair in gold_pairs if pair.task_id not in rescue_task_ids]
    replay_pool.sort(
        key=lambda pair: _canonical_sha256(
            {
                "schema": DATASET_SCHEMA,
                "seed": seed,
                "kind": "gold_replay",
                "task_id": pair.task_id,
            }
        )
    )
    replay_count = (
        gold_replay_rows
        if gold_replay_rows >= 0
        else math.ceil(gold_replay_ratio * len(rescue_pairs))
    )
    if replay_count > len(replay_pool):
        raise ValueError(
            f"requested {replay_count} gold replay rows, only "
            f"{len(replay_pool)} non-rescue TRAIN rows exist"
        )
    replay_pairs = [
        _make_pair(
            pair_id=f"{pair.task_id}::gold-replay",
            source_task_id=pair.task_id,
            kind="gold_replay",
            source=pair.source,
            target=pair.target,
            provenance=_provenance_tuple(
                gold_train_sha256=gold_manifest["dataset"]["sha256"],
                gold_f2_sha256=gold_manifest["f2"]["sha256"],
            ),
        )
        for pair in replay_pool[:replay_count]
    ]
    mixed = rescue_pairs + replay_pairs
    mixed.sort(
        key=lambda pair: _canonical_sha256(
            {
                "schema": DATASET_SCHEMA,
                "seed": seed,
                "pair_id": pair.pair_id,
                "kind": pair.kind,
                "source_sha256": pair.source_sha256,
                "target_sha256": pair.target_sha256,
            }
        )
    )
    if len({pair.pair_id for pair in mixed}) != len(mixed):
        raise AssertionError("mixed schedule contains duplicate pair IDs")
    if any(pair.source_task_id in heldout_ids for pair in mixed):
        raise AssertionError("held-out identity entered the final mixed schedule")

    exploratory = any(
        record.get("exploratory_prefix") is True for record in report_records
    )
    schedule_records = [
        {
            "position": position,
            "pair_id": pair.pair_id,
            "source_task_id": pair.source_task_id,
            "kind": pair.kind,
            "source_sha256": pair.source_sha256,
            "target_sha256": pair.target_sha256,
            "provenance": dict(pair.provenance),
        }
        for position, pair in enumerate(mixed)
    ]
    manifest = {
        "schema": DATASET_SCHEMA,
        "rows": len(mixed),
        "composition": {
            "verified_direct": direct_count,
            "repair_conditioned": repair_count,
            "gold_replay": len(replay_pairs),
        },
        "gold_replay": {
            "requested_rows": gold_replay_rows,
            "ratio_when_rows_is_minus_one": gold_replay_ratio,
            "selected_rows": len(replay_pairs),
            "rescue_tasks_excluded": True,
            "selection": "deterministic_hash_order",
            "seed": seed,
        },
        "gold_train": gold_manifest,
        "heldout_denylist": heldout_record,
        "reports": report_records,
        "pair_ids_sha256": _canonical_sha256([pair.pair_id for pair in mixed]),
        "source_sha256s_sha256": _canonical_sha256(
            [pair.source_sha256 for pair in mixed]
        ),
        "target_sha256s_sha256": _canonical_sha256(
            [pair.target_sha256 for pair in mixed]
        ),
        "schedule_sha256": _canonical_sha256(schedule_records),
        "schedule": schedule_records,
        "exploratory_inputs": exploratory,
        "production_floor_eligible": not exploratory,
        "architecture": "native_encoder_decoder",
        "model_visible_fields": [
            "sealed_F2_encoder_source",
            "sanitized_compiler_feedback_context_when_repair_conditioned",
        ],
        "tests_model_visible": False,
        "private_feedback_model_visible": False,
        "heldout_overlap": 0,
    }
    return mixed, manifest


def _adapter_weight_path(checkpoint: Path) -> Path:
    candidates = [
        checkpoint / "adapter" / "adapter_model.safetensors",
        checkpoint / "adapter" / "adapter_model.bin",
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError(f"{checkpoint}: expected exactly one adapter weight artifact")
    return present[0]


def validate_warmstart(
    checkpoint: Path,
    *,
    expected_update: int,
    expected_run_contract_sha256: str,
    expected_adapter_weights_sha256: str,
    expected_adapter_config_sha256: str,
    model: str,
    model_revision: str,
) -> tuple[WarmstartIdentity, dict[str, Any]]:
    checkpoint = checkpoint.expanduser().resolve()
    raw_contract = _read_json(checkpoint / "run_contract.json")
    raw_targets = raw_contract.get("lora", {}).get("targets")
    if (
        not isinstance(raw_targets, list)
        or not raw_targets
        or any(not isinstance(name, str) or not name for name in raw_targets)
        or len(set(raw_targets)) != len(raw_targets)
    ):
        raise ValueError("warm-start contract lacks exact LoRA targets")
    exact_targets = sorted(raw_targets)
    source_schema = raw_contract.get("schema")
    if source_schema == base_sft.RUN_SCHEMA:
        saved_contract, state = base_sft._load_resume_artifacts(
            checkpoint,
            exact_targets=exact_targets,
        )
    elif source_schema == RUN_SCHEMA:
        # A later RS-SFT pass is a new optimization contract, not a resume.
        # Validate the parent with the mixed stage's own checkpoint loader but
        # deliberately carry forward only its adapter and tokenizer below.
        saved_contract = raw_contract
        state = _load_stage_checkpoint(
            checkpoint,
            run_contract=saved_contract,
            exact_targets=exact_targets,
        )
        parent_dataset = saved_contract.get("dataset")
        parent_privacy = saved_contract.get("privacy")
        parent_lora = saved_contract.get("lora")
        if (
            not isinstance(parent_dataset, Mapping)
            or parent_dataset.get("schema") != DATASET_SCHEMA
            or parent_dataset.get("heldout_overlap") != 0
            or not isinstance(parent_privacy, Mapping)
            or parent_privacy.get("heldout_overlap") != 0
            or parent_privacy.get("heldout_content_model_visible") is not False
            or parent_privacy.get("tests_model_visible") is not False
            or parent_privacy.get("private_feedback_model_visible") is not False
            or not isinstance(parent_lora, Mapping)
            or parent_lora.get("new_adapter_attached") is not False
            or parent_lora.get("warmstart_weights_continued") is not True
        ):
            raise ValueError("mixed RS-SFT warm-start safety contract differs")
    else:
        raise ValueError("unsupported warm-start run contract schema")
    update = int(state.get("update", -1))
    contract_sha = base_sft.canonical_sha256(saved_contract)
    weights_path = _adapter_weight_path(checkpoint)
    weights_sha = base_sft.sha256_file(weights_path)
    config_sha = base_sft.sha256_file(checkpoint / "adapter" / "adapter_config.json")
    if (
        update != expected_update
        or checkpoint.name != base_sft._checkpoint_name(expected_update)
        or contract_sha
        != _require_sha256(
            expected_run_contract_sha256,
            "expected warm-start run contract",
        )
        or weights_sha
        != _require_sha256(
            expected_adapter_weights_sha256,
            "expected warm-start adapter weights",
        )
        or config_sha
        != _require_sha256(
            expected_adapter_config_sha256,
            "expected warm-start adapter config",
        )
        or saved_contract.get("architecture") != "native_encoder_decoder"
        or saved_contract.get("model") != model
        or saved_contract.get("model_revision") != model_revision
        or saved_contract.get("base_model", {}).get("is_encoder_decoder") is not True
        or saved_contract.get("lora", {}).get("encoder_and_decoder_trainable")
        is not True
        or saved_contract.get("lora", {}).get("vision_trainable") is not False
    ):
        raise ValueError("selected warm-start checkpoint contract differs")
    lora = saved_contract.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("warm-start LoRA contract is missing")
    identity = WarmstartIdentity(
        checkpoint_name=checkpoint.name,
        update=update,
        run_contract_sha256=contract_sha,
        adapter_weights_sha256=weights_sha,
        adapter_config_sha256=config_sha,
        model=model,
        model_revision=model_revision,
        lora_rank=int(lora.get("rank", -1)),
        lora_alpha=int(lora.get("alpha", -1)),
        lora_dropout=float(lora.get("dropout", -1.0)),
        exact_lora_targets=tuple(sorted(map(str, exact_targets))),
    )
    if (
        identity.lora_rank <= 0
        or identity.lora_alpha <= 0
        or not 0.0 <= identity.lora_dropout < 1.0
    ):
        raise ValueError("warm-start LoRA hyperparameters are invalid")
    adapter_config = _read_json(checkpoint / "adapter" / "adapter_config.json")
    try:
        config_rank = int(adapter_config.get("r", -1))
        config_alpha = int(adapter_config.get("lora_alpha", -1))
        config_dropout = float(adapter_config.get("lora_dropout", -1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("warm-start adapter hyperparameters are malformed") from exc
    if (
        config_rank != identity.lora_rank
        or config_alpha != identity.lora_alpha
        or config_dropout != identity.lora_dropout
        or str(adapter_config.get("task_type") or "") != "SEQ_2_SEQ_LM"
    ):
        raise ValueError("warm-start adapter config differs from the SFT run contract")
    return identity, saved_contract


def _atomic_exact_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing sealed JSON differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checkpoint_name(update: int) -> str:
    return f"checkpoint-optstep-{update:06d}"


def _save_checkpoint(
    *,
    output_dir: Path,
    update: int,
    epoch: int,
    next_row: int,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    run_contract: Mapping[str, Any],
) -> Path:
    destination = output_dir / _checkpoint_name(update)
    if destination.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {destination}")
    temporary = output_dir / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise ValueError(f"stale checkpoint temporary directory exists: {temporary}")
    temporary.mkdir(parents=False)
    try:
        (temporary / "adapter").mkdir()
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "update": update,
                "epoch": epoch,
                "next_row": next_row,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": base_sft._rng_state(),
                "run_contract_sha256": _canonical_sha256(run_contract),
            },
            temporary / "training_state.pt",
        )
        _atomic_exact_json(temporary / "run_contract.json", run_contract)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    latest = {
        "schema": CHECKPOINT_SCHEMA,
        "path": str(destination.resolve()),
        "update": update,
        "run_contract_sha256": _canonical_sha256(run_contract),
    }
    pointer = output_dir / "latest_checkpoint.json"
    temp_pointer = pointer.with_suffix(pointer.suffix + f".tmp-{os.getpid()}")
    with temp_pointer.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(latest, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_pointer, pointer)
    return destination


def _load_stage_checkpoint(
    checkpoint: Path,
    *,
    run_contract: Mapping[str, Any],
    exact_targets: Sequence[str],
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    required = [
        checkpoint / "run_contract.json",
        checkpoint / "training_state.pt",
        checkpoint / "adapter" / "adapter_config.json",
        checkpoint / "tokenizer" / "tokenizer_config.json",
    ]
    if (
        match is None
        or not checkpoint.is_dir()
        or any(not path.is_file() for path in required)
    ):
        raise ValueError("mixed RS-SFT resume checkpoint is incomplete")
    saved_contract = _read_json(checkpoint / "run_contract.json")
    # A freshly constructed contract can contain tuples (for example the
    # frozen LoRA target tuple in WarmstartIdentity), while its exact JSON
    # round-trip necessarily contains lists. Compare the canonical serialized
    # representation that also binds training_state.pt instead of Python
    # container types.
    if _canonical_sha256(saved_contract) != _canonical_sha256(run_contract):
        raise ValueError("mixed RS-SFT resume run contract differs")
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(state, Mapping)
        or state.get("schema") != CHECKPOINT_SCHEMA
        or state.get("run_contract_sha256") != _canonical_sha256(run_contract)
        or int(state.get("update", -1)) != int(match.group(1))
        or base_sft._adapter_weight_target_modules(checkpoint)
        != set(map(str, exact_targets))
    ):
        raise ValueError("mixed RS-SFT resume checkpoint binding failed")
    for key in ("optimizer", "scheduler", "rng"):
        if key not in state:
            raise ValueError(f"mixed RS-SFT resume lacks {key}")
    return dict(state)


def _load_policy(
    args: argparse.Namespace,
    *,
    token: str | None,
    warmstart: WarmstartIdentity,
    run_contract: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    from peft import PeftModel

    model = base_sft._load_base_model(args, token)
    exact_targets = base_sft._resolve_lora_targets(model, args)
    if tuple(sorted(exact_targets)) != warmstart.exact_lora_targets:
        raise ValueError("runtime LoRA targets differ from warm-start targets")
    resume_state: dict[str, Any] | None = None
    if args.resume_checkpoint:
        checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
        resume_state = _load_stage_checkpoint(
            checkpoint,
            run_contract=run_contract,
            exact_targets=exact_targets,
        )
        adapter_path = checkpoint / "adapter"
    else:
        adapter_path = (
            Path(args.warmstart_checkpoint).expanduser().resolve() / "adapter"
        )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "decoder"):
        model.config.decoder.use_cache = False
    base_sft._validate_trainable_adapter_parameters(model)
    return model, resume_state


def _runtime_contract() -> dict[str, str]:
    import peft
    import transformers

    return {
        "trainer_sha256": base_sft.sha256_file(Path(__file__).resolve()),
        "base_sft_helpers_sha256": base_sft.sha256_file(
            Path(base_sft.__file__).resolve()
        ),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "cuda": str(torch.version.cuda or ""),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.preflight_only:
        raise RuntimeError("mixed T5Gemma 2 RS-SFT training requires CUDA")
    if args.bf16 and not args.preflight_only and not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support BF16")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    token = os.environ.get("HF_TOKEN") or None
    # Data-only preflight callers can inject a sealed identity without loading
    # the 9B model. Real training validates the checkpoint from its own exact
    # adapter keys here and resolves the base model's module program once later.
    if args.preflight_only and args.warmstart_identity_json:
        raw_identity = _read_json(Path(args.warmstart_identity_json).resolve())
        warmstart = WarmstartIdentity(
            checkpoint_name=str(raw_identity["checkpoint_name"]),
            update=int(raw_identity["update"]),
            run_contract_sha256=_require_sha256(
                raw_identity["run_contract_sha256"], "warmstart identity"
            ),
            adapter_weights_sha256=_require_sha256(
                raw_identity["adapter_weights_sha256"], "warmstart weights"
            ),
            adapter_config_sha256=_require_sha256(
                raw_identity["adapter_config_sha256"], "warmstart config"
            ),
            model=str(raw_identity["model"]),
            model_revision=str(raw_identity["model_revision"]),
            lora_rank=int(raw_identity["lora_rank"]),
            lora_alpha=int(raw_identity["lora_alpha"]),
            lora_dropout=float(raw_identity["lora_dropout"]),
            exact_lora_targets=tuple(
                sorted(map(str, raw_identity["exact_lora_targets"]))
            ),
        )
        warmstart_contract: dict[str, Any] = {}
    else:
        warmstart, warmstart_contract = validate_warmstart(
            Path(args.warmstart_checkpoint),
            expected_update=args.expected_warmstart_update,
            expected_run_contract_sha256=args.expected_warmstart_run_contract_sha256,
            expected_adapter_weights_sha256=args.expected_warmstart_adapter_weights_sha256,
            expected_adapter_config_sha256=args.expected_warmstart_adapter_config_sha256,
            model=args.model,
            model_revision=args.model_revision,
        )

    local_reports = _parse_pinned_specs(args.local_report, label="--local_report")
    api_reports = _parse_pinned_specs(args.api_report, label="--api_report")
    pairs, dataset_manifest = build_mixed_pairs(
        gold_train_jsonl=Path(args.gold_train_jsonl).resolve(),
        gold_f2_jsonl=Path(args.gold_f2_jsonl).resolve(),
        expected_gold_train_sha256=args.expected_gold_train_sha256,
        expected_gold_f2_sha256=args.expected_gold_f2_sha256,
        expected_gold_rows=args.expected_gold_rows,
        heldout_jsonl=Path(args.heldout_jsonl).resolve(),
        expected_heldout_sha256=args.expected_heldout_sha256,
        expected_heldout_rows=args.expected_heldout_rows,
        local_reports=local_reports,
        api_reports=api_reports,
        warmstart=warmstart,
        gold_replay_ratio=args.gold_replay_ratio,
        gold_replay_rows=args.gold_replay_rows,
        min_verified_direct_targets=args.min_verified_direct_targets,
        min_repair_conditioned_targets=args.min_repair_conditioned_targets,
        allow_exploratory_inputs=args.allow_exploratory_inputs,
        require_local_production_floor=args.require_local_production_floor,
        seed=args.seed,
    )
    _atomic_exact_json(output_dir / "dataset_manifest.json", dataset_manifest)
    if args.preflight_only:
        result = {
            "schema": RUN_SCHEMA,
            "status": "preflight_complete",
            "rows": len(pairs),
            "composition": dataset_manifest["composition"],
            "warmstart": asdict(warmstart),
            "production_floor_eligible": dataset_manifest["production_floor_eligible"],
        }
        _atomic_exact_json(output_dir / "preflight.json", result)
        return result

    tokenizer = base_sft._load_tokenizer(
        str(Path(args.warmstart_checkpoint).resolve() / "tokenizer"),
        "",
        None,
    )
    if base_sft._tokenizer_contract(tokenizer) != warmstart_contract.get("tokenizer"):
        raise ValueError("warm-start tokenizer differs from its run contract")
    tokenized, token_report = base_sft.tokenize_pairs(
        tokenizer,
        [pair.as_text_pair() for pair in pairs],
        max_source_tokens=args.max_source_tokens,
        max_target_tokens=args.max_target_tokens,
    )
    schedule = base_sft.calculate_training_schedule(
        rows=len(tokenized),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_updates=args.max_updates,
        warmup_ratio=args.warmup_ratio,
    )
    run_contract = {
        "schema": RUN_SCHEMA,
        "status": "training",
        "architecture": "native_encoder_decoder",
        "model": args.model,
        "model_revision": args.model_revision,
        "base_model": dict(warmstart_contract["base_model"]),
        "runtime": _runtime_contract(),
        "warmstart": asdict(warmstart),
        "warmstart_contract_schema": str(warmstart_contract["schema"]),
        "dataset": dataset_manifest,
        "tokenization": token_report,
        "tokenizer": base_sft._tokenizer_contract(tokenizer),
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_updates": schedule["warmup_updates"],
            "planned_updates": schedule["planned_updates"],
            "updates_per_epoch": schedule["updates_per_epoch"],
            "seed": args.seed,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "attn_implementation": args.attn_implementation,
        },
        "lora": {
            "rank": warmstart.lora_rank,
            "alpha": warmstart.lora_alpha,
            "dropout": warmstart.lora_dropout,
            "targets": list(warmstart.exact_lora_targets),
            "encoder_and_decoder_trainable": True,
            "vision_trainable": False,
            "new_adapter_attached": False,
            "warmstart_weights_continued": True,
        },
        "privacy": {
            "heldout_overlap": 0,
            "heldout_content_model_visible": False,
            "tests_model_visible": False,
            "private_feedback_model_visible": False,
            "reasoning_persisted": False,
        },
        "production_floor_eligible": dataset_manifest["production_floor_eligible"],
        "checkpointing": {
            "interval": args.checkpoint_interval,
            "immutable_adapter_only": True,
            "resume_state": ["optimizer", "scheduler", "rng", "epoch", "next_row"],
        },
    }
    _atomic_exact_json(output_dir / "run_contract.json", run_contract)
    latest_pointer = output_dir / "latest_checkpoint.json"
    if latest_pointer.is_file():
        latest = _read_json(latest_pointer)
        recorded = Path(str(latest.get("path") or "")).expanduser().resolve()
        if (
            latest.get("schema") != CHECKPOINT_SCHEMA
            or not args.resume_checkpoint
            or recorded != Path(args.resume_checkpoint).expanduser().resolve()
            or latest.get("run_contract_sha256") != _canonical_sha256(run_contract)
            or int(latest.get("update", -1))
            != int(
                _CHECKPOINT_RE.fullmatch(recorded.name).group(1)
                if _CHECKPOINT_RE.fullmatch(recorded.name)
                else -1
            )
        ):
            raise ValueError(
                "existing mixed RS-SFT checkpoint requires an exact "
                "--resume_checkpoint matching latest_checkpoint.json"
            )
    elif args.resume_checkpoint:
        raise ValueError(
            "--resume_checkpoint requires this output directory's sealed "
            "latest_checkpoint.json"
        )
    model, resume_state = _load_policy(
        args,
        token=token,
        warmstart=warmstart,
        run_contract=run_contract,
    )
    source_capacity, target_capacity = base_sft._config_position_capacities(model)
    if (
        int(token_report["source_tokens"]["max"]) > source_capacity
        or int(token_report["target_tokens"]["max"]) > target_capacity
    ):
        raise ValueError("mixed data exceeds native encoder/decoder capacity")
    device = torch.device("cuda")
    model.to(device)
    optimizer, scheduler = base_sft._optimizer_and_scheduler(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_updates=schedule["warmup_updates"],
        total_updates=schedule["planned_updates"],
    )
    start_epoch = 0
    start_row = 0
    update = 0
    if resume_state is not None:
        start_epoch = int(resume_state.get("epoch", -1))
        start_row = int(resume_state.get("next_row", -1))
        update = int(resume_state.get("update", -1))
        completed_microbatches = (
            math.ceil(start_row / args.batch_size) if start_row else 0
        )
        expected_update = (
            start_epoch * schedule["updates_per_epoch"]
            + completed_microbatches // args.gradient_accumulation
        )
        if (
            start_epoch < 0
            or start_epoch > args.epochs
            or start_row < 0
            or start_row > len(tokenized)
            or start_row % args.batch_size != 0
            or completed_microbatches % args.gradient_accumulation != 0
            or update != expected_update
            or update > schedule["planned_updates"]
        ):
            raise ValueError("mixed RS-SFT resume position is invalid")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        base_sft._restore_rng_state(resume_state["rng"])

    if update == schedule["planned_updates"]:
        result = {
            "schema": RUN_SCHEMA,
            "status": "complete",
            "updates": update,
            "planned_updates": schedule["planned_updates"],
            "rows": len(tokenized),
            "latest_checkpoint": _checkpoint_name(update),
            "resumed_complete_checkpoint": True,
            "production_floor_eligible": dataset_manifest["production_floor_eligible"],
        }
        _atomic_exact_json(output_dir / "result.json", result)
        return result

    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    accumulated = 0
    running_loss = 0.0
    stop = False
    for epoch in range(start_epoch, args.epochs):
        order = base_sft.deterministic_epoch_order(
            tokenized, seed=args.seed, epoch=epoch
        )
        position = start_row if epoch == start_epoch else 0
        while position < len(order):
            indices = order[position : position + args.batch_size]
            batch = base_sft.collate_pairs(
                [tokenized[index] for index in indices],
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite mixed RS-SFT loss at epoch={epoch}, " f"row={position}"
                )
            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.detach().cpu())
            accumulated += 1
            position += len(indices)
            epoch_finished = position >= len(order)
            if accumulated < args.gradient_accumulation and not epoch_finished:
                continue
            if accumulated < args.gradient_accumulation:
                correction = args.gradient_accumulation / accumulated
                for parameter in model.parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        parameter.grad.mul_(correction)
            grad_norm = clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            metric = {
                "schema": RUN_SCHEMA,
                "update": update,
                "epoch": epoch,
                "next_row": position,
                "loss": running_loss / accumulated,
                "grad_norm": float(grad_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "microbatches": accumulated,
                "source_tokens": int(batch["attention_mask"].sum().item()),
                "target_tokens": int((batch["labels"] != -100).sum().item()),
            }
            with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        metric,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(metric, sort_keys=True), flush=True)
            accumulated = 0
            running_loss = 0.0
            next_epoch = epoch
            next_row = position
            if next_row >= len(order):
                next_epoch = epoch + 1
                next_row = 0
            if (
                update % args.checkpoint_interval == 0
                or update >= schedule["planned_updates"]
            ):
                _save_checkpoint(
                    output_dir=output_dir,
                    update=update,
                    epoch=next_epoch,
                    next_row=next_row,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    run_contract=run_contract,
                )
            if update >= schedule["planned_updates"]:
                stop = True
                break
        if stop:
            break
        start_row = 0

    result = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "updates": update,
        "planned_updates": schedule["planned_updates"],
        "rows": len(tokenized),
        "latest_checkpoint": _checkpoint_name(update),
        "production_floor_eligible": dataset_manifest["production_floor_eligible"],
        "architecture": "native_encoder_decoder",
    }
    _atomic_exact_json(output_dir / "result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gold_train_jsonl", required=True)
    parser.add_argument("--gold_f2_jsonl", required=True)
    parser.add_argument("--expected_gold_train_sha256", required=True)
    parser.add_argument("--expected_gold_f2_sha256", required=True)
    parser.add_argument("--expected_gold_rows", type=int, default=2776)
    parser.add_argument("--heldout_jsonl", required=True)
    parser.add_argument("--expected_heldout_sha256", required=True)
    parser.add_argument("--expected_heldout_rows", type=int, default=175)
    parser.add_argument(
        "--local_report",
        action="append",
        default=[],
        metavar="SHA256=PATH",
    )
    parser.add_argument(
        "--api_report",
        action="append",
        default=[],
        metavar="SHA256=PATH",
    )
    parser.add_argument("--allow_exploratory_inputs", action="store_true")
    parser.add_argument("--require_local_production_floor", action="store_true")
    parser.add_argument("--gold_replay_ratio", type=float, default=3.0)
    parser.add_argument(
        "--gold_replay_rows",
        type=int,
        default=-1,
        help="-1 derives the count as ceil(ratio * rescue rows)",
    )
    parser.add_argument("--min_verified_direct_targets", type=int, default=1)
    parser.add_argument("--min_repair_conditioned_targets", type=int, default=1)
    parser.add_argument("--warmstart_checkpoint", required=True)
    parser.add_argument("--expected_warmstart_update", type=int, default=348)
    parser.add_argument("--expected_warmstart_run_contract_sha256", required=True)
    parser.add_argument("--expected_warmstart_adapter_weights_sha256", required=True)
    parser.add_argument("--expected_warmstart_adapter_config_sha256", required=True)
    parser.add_argument(
        "--warmstart_identity_json",
        default="",
        help="Tests/preflight only: sealed identity without loading the 9B model",
    )
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=base_sft.DEFAULT_MODEL)
    parser.add_argument("--model_revision", required=True)
    parser.add_argument("--max_source_tokens", type=int, default=32768)
    parser.add_argument("--max_target_tokens", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--max_updates", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa"],
        default="sdpa",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args(argv)
    for field in (
        "expected_gold_train_sha256",
        "expected_gold_f2_sha256",
        "expected_heldout_sha256",
        "expected_warmstart_run_contract_sha256",
        "expected_warmstart_adapter_weights_sha256",
        "expected_warmstart_adapter_config_sha256",
    ):
        try:
            _require_sha256(getattr(args, field), f"--{field}")
        except ValueError as exc:
            parser.error(str(exc))
    if args.expected_gold_rows <= 0 or args.expected_heldout_rows <= 0:
        parser.error("expected row counts must be positive")
    if args.expected_warmstart_update <= 0:
        parser.error("--expected_warmstart_update must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if args.gradient_accumulation <= 0 or args.checkpoint_interval <= 0:
        parser.error("accumulation and checkpoint interval must be positive")
    if args.max_updates < 0:
        parser.error("--max_updates cannot be negative")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        parser.error("learning rate and max grad norm must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must lie in [0,1)")
    if args.gold_replay_rows < -1:
        parser.error("--gold_replay_rows must be -1 or non-negative")
    if args.gold_replay_ratio < 0 or not math.isfinite(args.gold_replay_ratio):
        parser.error("--gold_replay_ratio must be finite and non-negative")
    if args.min_verified_direct_targets < 0 or args.min_repair_conditioned_targets < 0:
        parser.error("minimum target counts must be non-negative")
    if args.warmstart_identity_json and not args.preflight_only:
        parser.error("--warmstart_identity_json is allowed only with --preflight_only")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
