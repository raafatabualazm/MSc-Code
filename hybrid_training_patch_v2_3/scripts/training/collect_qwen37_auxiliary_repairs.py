#!/usr/bin/env python3
"""Collect a quarantined, verified Qwen3.7 repair corpus for fit2776.

This is deliberately *not* a Qwen3.8 sequence-KL/CoT collector.  It makes one
logical Qwen3.7 draw for each selected failed training task, locally replays the
returned Dart against the sealed private verifier, and publishes only verified
code as an auxiliary RS-SFT hard target.

The provider never receives tests, gold Dart, held-out rows, or raw compiler
diagnostics.  Collection is append-only and fail-closed:

* every exact pinned model gets its own artifact directory and hash chain;
* a pessimistic token reservation is journaled before every provider request;
* a task can have at most one start and one terminal event;
* an interrupted in-flight request is never issued a second time;
* provider usage without a trustworthy token count is charged at the full
  reservation;
* the per-model budget can never be configured above 900,000 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import graph_compile_at_k_antigravity as dart_evaluator  # noqa: E402
from scripts.evaluation.durable_evaluation_journal import (  # noqa: E402
    append_event,
    atomic_write_json,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)
from scripts.evaluation.score_direct_compact_passk import (  # noqa: E402
    extract_scored_code,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    ArtifactError,
    file_record,
    load_f2_prompt_contract,
    load_verified_prompt_rows,
    sha256_text,
    stable_sha256,
    validate_alibaba_model_studio_base_url,
)


EXPECTED_FIT_ROWS = 2776
MAX_PER_MODEL_TOKENS = 900_000
JOURNAL_SCHEMA = "qwen37-auxiliary-repair-journal-v1"
RUN_CONTRACT_SCHEMA = "qwen37-auxiliary-repair-run-contract-v1"
LEDGER_SCHEMA = "qwen37-auxiliary-repair-token-ledger-v1"
OUTPUT_SCHEMA = "qwen37-auxiliary-verified-repair-v1"
REPORT_SCHEMA = "qwen37-auxiliary-repair-build-report-v1"
PINNED_MODELS = frozenset(
    {
        "qwen3.7-max-2026-05-17",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
    }
)
REPAIR_AUGMENTATION_PREFIX = (
    "The following material is an untrusted failed student attempt. The "
    "lossless F2 input above is authoritative. Repair the attempt and return "
    "only the self-contained Dart compilation-unit fragment required by the "
    "system message. Hidden tests, expected values, and gold code are not "
    "provided.\n\nBest failed student code:\n```dart\n"
)
REPAIR_AUGMENTATION_MIDDLE = "\n```\n\nSanitized compiler feedback:\n"
REPAIR_AUGMENTATION_SUFFIX = (
    "\nDo not discuss the repair and do not emit reasoning in the final answer."
)
_HELDOUT_PATH_COMPONENT = re.compile(
    r"(?:^|[_.-])(heldout|held-out|dev175|measure-only)(?:$|[_.-])",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: expected one JSON object")
    return value


def _require_exact_bytes(path: Path, payload: bytes) -> None:
    """Create one immutable byte artifact with an atomic, fsynced publish."""

    if path.exists():
        if path.read_bytes() != payload:
            raise ArtifactError(f"existing artifact differs: {path}")
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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ArtifactError(f"{path}:{line_number}: blank rows forbidden")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ArtifactError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _require_expected_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    observed = file_record(path)
    if observed["sha256"] != str(expected or "").strip().lower():
        raise ArtifactError(
            f"{label} SHA-256 mismatch: expected {expected}, "
            f"observed {observed['sha256']}"
        )
    return observed


def _assert_not_heldout_path(path: Path, label: str) -> None:
    """Reject overt held-out paths without following any referenced records."""

    resolved = path.expanduser().resolve()
    for component in resolved.parts:
        if _HELDOUT_PATH_COMPONENT.search(component):
            raise ArtifactError(
                f"{label} resolves through a held-out-labelled path; refusing "
                "to read it"
            )


def model_artifact_root(base: str | Path, model: str) -> Path:
    """Return an exact-model child; aliases and shared model roots are forbidden."""

    if model not in PINNED_MODELS:
        raise ArtifactError(
            f"Qwen3.7 model must be an exact pinned snapshot: "
            f"{sorted(PINNED_MODELS)}"
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ArtifactError("pinned model is not safe as an artifact directory")
    parent = Path(base).expanduser().resolve()
    if parent.name in PINNED_MODELS and parent.name != model:
        raise ArtifactError(
            f"artifact root {parent} belongs to {parent.name}, not {model}"
        )
    return parent if parent.name == model else parent / model


def sanitize_compiler_feedback(
    diagnostic: str,
    *,
    compiled: bool,
) -> tuple[str, str]:
    """Map a private-harness diagnostic to a fixed, non-leaking message."""

    if compiled:
        return (
            "compiled_behavior_failed",
            "The candidate compiled, but its behavior did not pass the local "
            "acceptance verifier. No test inputs, outputs, or expected values "
            "are disclosed.",
        )
    lowered = str(diagnostic or "").lower()
    categories: tuple[tuple[tuple[str, ...], str, str], ...] = (
        (
            ("syntax error", "expected", "unexpected token", "parser"),
            "dart_syntax_or_parse_error",
            "Dart reported a syntax or parse error in the candidate.",
        ),
        (
            (
                "undefined name",
                "undefined identifier",
                "isn't defined",
                "not found",
            ),
            "undefined_identifier",
            "Dart reported an undefined identifier in the candidate.",
        ),
        (
            (
                "argument",
                "positional",
                "named parameter",
                "too many",
                "too few",
            ),
            "call_arity_or_parameter_error",
            "Dart reported an argument or parameter mismatch in the candidate.",
        ),
        (
            (
                "type ",
                "assignable",
                "subtype",
                "return type",
                "can't be assigned",
            ),
            "dart_type_error",
            "Dart reported a static type error in the candidate.",
        ),
        (
            ("import", "library", "uri"),
            "import_or_library_error",
            "Dart reported an import or library-resolution error.",
        ),
        (
            ("already declared", "duplicate"),
            "duplicate_declaration",
            "Dart reported a duplicate declaration in the candidate.",
        ),
    )
    for needles, category, message in categories:
        if any(needle in lowered for needle in needles):
            return category, message
    return (
        "dart_compile_error_unspecified",
        "The candidate did not compile under the local Dart verifier.",
    )


@dataclass(frozen=True)
class FailedCandidate:
    task_id: str
    sample_index: int
    priority: int
    priority_name: str
    code: str
    code_sha256: str
    raw_sha256: str
    compiled: bool
    feedback_category: str
    feedback: str
    diagnostic_sha256: str
    compiling_samples: int

    def seal(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "priority": self.priority,
            "priority_name": self.priority_name,
            "code_sha256": self.code_sha256,
            "raw_sha256": self.raw_sha256,
            "compiled": self.compiled,
            "feedback_category": self.feedback_category,
            "feedback_sha256": sha256_text(self.feedback),
            "diagnostic_sha256": self.diagnostic_sha256,
            "compiling_samples": self.compiling_samples,
        }


def _candidate_rank(
    *,
    compiled: bool,
    feedback_category: str,
    code: str,
    sample_index: int,
) -> tuple[Any, ...]:
    category_rank = {
        "undefined_identifier": 0,
        "call_arity_or_parameter_error": 1,
        "dart_type_error": 2,
        "duplicate_declaration": 3,
        "import_or_library_error": 4,
        "dart_syntax_or_parse_error": 5,
        "dart_compile_error_unspecified": 6,
    }.get(feedback_category, 7)
    if compiled:
        category_rank = 0
    return (
        category_rank,
        len(code),
        sha256_text(code),
        int(sample_index),
    )


def select_failed_candidates(
    *,
    predictions: Mapping[str, Sequence[str]],
    task_results: Mapping[str, Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
) -> tuple[list[FailedCandidate], dict[str, int]]:
    """Recompute and order fit failures from the attested candidate-level score."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidate_results:
        task_id = str(row.get("task_id") or "")
        if task_id not in predictions:
            raise ArtifactError(f"score candidate belongs to unknown task {task_id!r}")
        by_task.setdefault(task_id, []).append(row)

    chosen: list[FailedCandidate] = []
    stats = {
        "tasks": len(predictions),
        "already_passed_skipped": 0,
        "compiled_failed": 0,
        "parseable_noncompile": 0,
        "unparseable_skipped": 0,
    }
    for task_id in sorted(predictions):
        aggregate = task_results.get(task_id)
        if aggregate is None:
            raise ArtifactError(f"score aggregate missing task {task_id}")
        rows = sorted(
            by_task.get(task_id, []),
            key=lambda row: int(row.get("sample_index", -1)),
        )
        samples = list(predictions[task_id])
        if [int(row.get("sample_index", -1)) for row in rows] != list(
            range(len(samples))
        ):
            raise ArtifactError(f"{task_id}: score candidate coverage differs")

        candidates: list[dict[str, Any]] = []
        for index, (raw, score_row) in enumerate(zip(samples, rows)):
            raw_text = str(raw or "")
            code = extract_scored_code(raw_text)
            raw_hash = sha256_text(raw_text)
            code_hash = sha256_text(code)
            if (
                score_row.get("raw_sha256") != raw_hash
                or score_row.get("code_sha256") != code_hash
            ):
                raise ArtifactError(
                    f"{task_id}:{index}: score hashes do not bind predictions"
                )
            passed = bool(score_row.get("passed"))
            compiled = bool(score_row.get("compiled"))
            if passed and not compiled:
                raise ArtifactError(f"{task_id}:{index}: passed without compiling")
            diagnostic = str(score_row.get("diagnostic") or "")
            category, feedback = sanitize_compiler_feedback(
                diagnostic, compiled=compiled
            )
            candidates.append(
                {
                    "sample_index": index,
                    "raw_sha256": raw_hash,
                    "code": code,
                    "code_sha256": code_hash,
                    "compiled": compiled,
                    "passed": passed,
                    "feedback_category": category,
                    "feedback": feedback,
                    "diagnostic_sha256": sha256_text(diagnostic),
                }
            )

        recomputed_pass = any(row["passed"] for row in candidates)
        recomputed_compile = any(row["compiled"] for row in candidates)
        if (
            recomputed_pass != bool(aggregate.get("pass_at_k"))
            or recomputed_compile != bool(aggregate.get("compile_at_k"))
            or sum(row["passed"] for row in candidates)
            != int(aggregate.get("passing_samples", -1))
            or sum(row["compiled"] for row in candidates)
            != int(aggregate.get("compiling_samples", -1))
        ):
            raise ArtifactError(f"{task_id}: score aggregate is inconsistent")
        if recomputed_pass:
            stats["already_passed_skipped"] += 1
            continue

        compiled_failures = [
            row for row in candidates if row["compiled"] and not row["passed"]
        ]
        parseable_noncompile = [
            row
            for row in candidates
            if not row["compiled"] and bool(str(row["code"]).strip())
        ]
        if compiled_failures:
            pool = compiled_failures
            priority = 0
            priority_name = "compiled_failed"
            stats["compiled_failed"] += 1
        elif parseable_noncompile:
            pool = parseable_noncompile
            priority = 1
            priority_name = "parseable_noncompile"
            stats["parseable_noncompile"] += 1
        else:
            stats["unparseable_skipped"] += 1
            continue
        best = min(
            pool,
            key=lambda row: _candidate_rank(
                compiled=bool(row["compiled"]),
                feedback_category=str(row["feedback_category"]),
                code=str(row["code"]),
                sample_index=int(row["sample_index"]),
            ),
        )
        chosen.append(
            FailedCandidate(
                task_id=task_id,
                sample_index=int(best["sample_index"]),
                priority=priority,
                priority_name=priority_name,
                code=str(best["code"]),
                code_sha256=str(best["code_sha256"]),
                raw_sha256=str(best["raw_sha256"]),
                compiled=bool(best["compiled"]),
                feedback_category=str(best["feedback_category"]),
                feedback=str(best["feedback"]),
                diagnostic_sha256=str(best["diagnostic_sha256"]),
                compiling_samples=sum(row["compiled"] for row in candidates),
            )
        )

    chosen.sort(
        key=lambda row: (
            row.priority,
            -row.compiling_samples,
            _candidate_rank(
                compiled=row.compiled,
                feedback_category=row.feedback_category,
                code=row.code,
                sample_index=row.sample_index,
            ),
            row.task_id,
        )
    )
    return chosen, stats


def _load_fit2776(
    *,
    fit_path: Path,
    expected_fit_sha256: str,
    fit_seal_path: Path,
    expected_fit_seal_sha256: str,
    frozen_contract_path: Path,
    expected_frozen_contract_sha256: str,
    prompt_record: Mapping[str, Any],
    prompt_manifest_record: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    for path, label in (
        (fit_path, "fit2776"),
        (fit_seal_path, "fit2776 seal"),
        (frozen_contract_path, "frozen split contract"),
    ):
        _assert_not_heldout_path(path, label)
    fit_record = _require_expected_sha(
        fit_path, expected_fit_sha256, "fit2776"
    )
    seal_record = _require_expected_sha(
        fit_seal_path, expected_fit_seal_sha256, "fit2776 seal"
    )
    contract_record = _require_expected_sha(
        frozen_contract_path,
        expected_frozen_contract_sha256,
        "frozen split contract",
    )
    seal = _json_object(fit_seal_path)
    task_rows = _jsonl(fit_path)
    if len(task_rows) != EXPECTED_FIT_ROWS:
        raise ArtifactError(
            f"fit artifact has {len(task_rows)} rows, expected {EXPECTED_FIT_ROWS}"
        )
    tests: dict[str, str] = {}
    ordered_ids: list[str] = []
    for index, row in enumerate(task_rows):
        task_id = str(row.get("task_id") or "")
        tests_text = row.get("acceptance_tests") or row.get("tests")
        if (
            not task_id
            or task_id in tests
            or not isinstance(tests_text, str)
            or not tests_text.strip()
        ):
            raise ArtifactError(
                f"fit row {index} has missing/duplicate task or verifier"
            )
        ordered_ids.append(task_id)
        tests[task_id] = tests_text

    seal_f2 = seal.get("f2_output") or {}
    seal_manifest = seal.get("f2_manifest") or {}
    if (
        seal.get("selected_role") != "fit"
        or seal.get("training_allowed") is not True
        or seal.get("heldout_measure_only") is not False
        or int(seal.get("rows", -1)) != EXPECTED_FIT_ROWS
        or seal.get("output_sha256") != fit_record["sha256"]
        or seal.get("contract_sha256") != contract_record["sha256"]
        or not isinstance(seal_f2, Mapping)
        or seal_f2.get("sha256") != prompt_record.get("sha256")
        or not isinstance(seal_manifest, Mapping)
        or seal_manifest.get("sha256")
        != prompt_manifest_record.get("sha256")
        or seal.get("task_set_sha256") != stable_sha256(ordered_ids)
        or seal.get("ordered_task_ids_sha256") != stable_sha256(ordered_ids)
        or seal.get("sorted_task_set_sha256")
        != stable_sha256(sorted(ordered_ids))
    ):
        raise ArtifactError(
            "fit2776 seal does not bind the exact training-only dataset, F2 "
            "artifact, manifest, membership, and frozen split contract"
        )
    # Do not dereference anything under heldout_commitment.  It is copied only
    # as a canonical digest, proving the fit seal carried a split commitment.
    heldout_commitment = seal.get("heldout_commitment")
    if not isinstance(heldout_commitment, Mapping):
        raise ArtifactError("fit2776 seal lacks a held-out split commitment")
    return tests, {
        "fit": fit_record,
        "fit_seal": seal_record,
        "frozen_contract": contract_record,
        "heldout_commitment_sha256": stable_sha256(heldout_commitment),
        "heldout_artifact_opened": False,
    }, {
        "ordered_ids": ordered_ids,
        "ordered_task_ids_sha256": stable_sha256(ordered_ids),
    }


def _load_scored_predictions(
    *,
    fit_record: Mapping[str, Any],
    expected_task_ids: set[str],
    prediction_path: Path,
    expected_prediction_sha256: str,
    score_path: Path,
    expected_score_sha256: str,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    for path, label in (
        (prediction_path, "fit predictions"),
        (score_path, "fit score"),
    ):
        _assert_not_heldout_path(path, label)
    prediction_record = _require_expected_sha(
        prediction_path, expected_prediction_sha256, "fit predictions"
    )
    score_record = _require_expected_sha(
        score_path, expected_score_sha256, "fit score"
    )
    provenance_path = Path(str(prediction_path) + ".provenance.json")
    _assert_not_heldout_path(provenance_path, "prediction provenance")
    provenance_record = file_record(provenance_path)
    provenance = _json_object(provenance_path)
    score = _json_object(score_path)
    raw_predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(raw_predictions, list):
        raise ArtifactError("predictions must be one JSON array")
    k = int(score.get("k", -1))
    predictions: dict[str, list[str]] = {}
    for row in raw_predictions:
        if not isinstance(row, Mapping):
            raise ArtifactError("prediction row is not an object")
        task_id = str(row.get("id") or "")
        samples = row.get("predictions")
        if (
            not task_id
            or task_id in predictions
            or not isinstance(samples, list)
            or len(samples) != k
        ):
            raise ArtifactError(f"invalid prediction row {task_id!r}")
        predictions[task_id] = [str(value or "") for value in samples]
    task_rows = score.get("task_results")
    candidate_rows = score.get("candidate_results")
    if not isinstance(task_rows, list) or not isinstance(candidate_rows, list):
        raise ArtifactError("score lacks task/candidate detail")
    task_results: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        if not isinstance(row, Mapping):
            raise ArtifactError("score task result is not an object")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in task_results:
            raise ArtifactError("score has missing/duplicate task result")
        task_results[task_id] = dict(row)
    if (
        set(predictions) != expected_task_ids
        or set(task_results) != expected_task_ids
        or len(candidate_rows) != len(expected_task_ids) * k
        or score.get("schema") != "direct-compact-attested-passk-v1"
        or score.get("evaluation", {}).get("sha256")
        != fit_record.get("sha256")
        or score.get("predictions", {}).get("sha256")
        != prediction_record["sha256"]
        or score.get("predictions", {}).get("provenance_sha256")
        != provenance_record["sha256"]
        or score.get("evaluator", {}).get("completion_attestation")
        != "per-run-256-bit-marker-exactly-once-v1"
        or int(score.get("tasks", -1)) != EXPECTED_FIT_ROWS
        or provenance.get("schema") != "direct-compact-inference-v1"
        or provenance.get("output_sha256") != prediction_record["sha256"]
        or int(provenance.get("num_samples", -1)) != k
    ):
        raise ArtifactError(
            "score/prediction provenance does not bind the exact fit2776 "
            "training universe"
        )
    return (
        predictions,
        task_results,
        [dict(row) for row in candidate_rows],
        {
            "predictions": prediction_record,
            "prediction_provenance": provenance_record,
            "score": score_record,
            "k": k,
        },
    )


def load_tokenizer(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    _assert_not_heldout_path(path, "student tokenizer")
    record = _require_expected_sha(path, expected_sha256, "student tokenizer")
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - environment dependency
        raise ArtifactError("the tokenizers package is required") from exc
    return Tokenizer.from_file(str(path.resolve())), record


def count_message_tokens(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    overhead_reserve: int,
) -> int:
    total = int(overhead_reserve)
    for message in messages:
        encoded = tokenizer.encode(
            str(message.get("content") or ""), add_special_tokens=False
        )
        total += len(encoded.ids if hasattr(encoded, "ids") else encoded)
    return total


def repair_messages(
    *,
    system_prompt: str,
    prompt_text: str,
    failed: FailedCandidate,
) -> list[dict[str, str]]:
    augmentation = (
        REPAIR_AUGMENTATION_PREFIX
        + failed.code.rstrip()
        + REPAIR_AUGMENTATION_MIDDLE
        + failed.feedback
        + REPAIR_AUGMENTATION_SUFFIX
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": augmentation},
    ]


@dataclass(frozen=True)
class JournalState:
    starts: dict[str, dict[str, Any]]
    terminals: dict[str, dict[str, Any]]
    budget_debits: int
    actual_usage_tokens: int
    unknown_usage_slots: int
    complete: dict[str, Any] | None
    orphan_task_id: str | None


def validate_journal_state(
    events: Sequence[Mapping[str, Any]],
    *,
    contract_sha256: str,
    budget_cap: int,
) -> JournalState:
    if not events:
        return JournalState({}, {}, 0, 0, 0, None, None)
    header = events[0]
    if (
        header.get("event") != "repair_header"
        or header.get("schema") != JOURNAL_SCHEMA
        or header.get("run_contract_sha256") != contract_sha256
    ):
        raise ArtifactError("repair journal header differs from run contract")
    starts: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    complete: dict[str, Any] | None = None
    debit_total = 0
    actual_total = 0
    unknown = 0
    for event in events[1:]:
        event_type = str(event.get("event") or "")
        if complete is not None:
            raise ArtifactError("journal contains events after collection_complete")
        if event_type == "repair_slot_started":
            task_id = str(event.get("task_id") or "")
            reservation = event.get("reservation_tokens")
            if (
                not task_id
                or task_id in starts
                or task_id in terminals
                or isinstance(reservation, bool)
                or not isinstance(reservation, int)
                or reservation <= 0
                or debit_total + reservation > budget_cap
            ):
                raise ArtifactError("invalid or over-budget repair reservation")
            if set(starts).difference(terminals):
                raise ArtifactError("parallel/open repair reservations are forbidden")
            starts[task_id] = dict(event)
        elif event_type == "repair_slot_terminal":
            task_id = str(event.get("task_id") or "")
            if (
                task_id not in starts
                or task_id in terminals
                or event.get("start_event_sha256")
                != starts[task_id].get("journal_event_sha256")
            ):
                raise ArtifactError("repair terminal does not match one start")
            reservation = int(starts[task_id]["reservation_tokens"])
            debit = event.get("budget_debit_tokens")
            usage = event.get("provider_usage")
            if (
                isinstance(debit, bool)
                or not isinstance(debit, int)
                or debit <= 0
                or debit > reservation
                or debit_total + debit > budget_cap
            ):
                raise ArtifactError("repair terminal has invalid budget debit")
            if usage is None:
                if debit != reservation:
                    raise ArtifactError(
                        "unknown provider usage must debit the full reservation"
                    )
                unknown += 1
            else:
                if not isinstance(usage, Mapping):
                    raise ArtifactError("provider usage is not an object")
                actual = usage.get("total_tokens")
                if (
                    isinstance(actual, bool)
                    or not isinstance(actual, int)
                    or actual <= 0
                    or actual > reservation
                    or debit != actual
                ):
                    raise ArtifactError(
                        "provider usage exceeds reservation or differs from debit"
                    )
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if (
                    isinstance(prompt_tokens, bool)
                    or not isinstance(prompt_tokens, int)
                    or prompt_tokens < 0
                    or isinstance(completion_tokens, bool)
                    or not isinstance(completion_tokens, int)
                    or completion_tokens < 0
                    or actual < prompt_tokens + completion_tokens
                ):
                    raise ArtifactError("provider token usage is internally invalid")
                actual_total += actual
            debit_total += debit
            terminals[task_id] = dict(event)
        elif event_type == "collection_complete":
            if set(starts).difference(terminals):
                raise ArtifactError("collection completed with an open request")
            if int(event.get("budget_debit_tokens", -1)) != debit_total:
                raise ArtifactError("completion budget total is inconsistent")
            complete = dict(event)
        else:
            raise ArtifactError(f"unknown repair journal event {event_type!r}")
    open_tasks = sorted(set(starts).difference(terminals))
    return JournalState(
        starts=starts,
        terminals=terminals,
        budget_debits=debit_total,
        actual_usage_tokens=actual_total,
        unknown_usage_slots=unknown,
        complete=complete,
        orphan_task_id=open_tasks[0] if open_tasks else None,
    )


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_from_response(response: Any) -> dict[str, int] | None:
    usage = _field(response, "usage")
    if usage is None:
        return None
    result: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _field(usage, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        result[name] = int(value)
    if (
        result["total_tokens"] <= 0
        or result["total_tokens"]
        < result["prompt_tokens"] + result["completion_tokens"]
    ):
        return None
    return result


def _error_receipt(exc: BaseException) -> dict[str, Any]:
    # Exception strings can echo endpoints or request bodies.  Persist only a
    # class and hash, never the potentially sensitive raw message.
    raw = f"{type(exc).__module__}.{type(exc).__name__}:{exc}"
    return {
        "error_type": f"{type(exc).__module__}.{type(exc).__name__}",
        "error_message_sha256": sha256_text(raw),
    }


def invoke_one(
    *,
    client: Any,
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    generation: Mapping[str, Any],
    failed: FailedCandidate,
    tests: str,
    reservation_tokens: int,
    verifier_timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    """Make exactly one provider request, then locally attest its final code."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **dict(generation),
        )
    except BaseException as exc:
        return {
            "status": "provider_error",
            "provider_usage": None,
            "budget_debit_tokens": reservation_tokens,
            "provider_error": _error_receipt(exc),
            "requested_model": model,
            "endpoint": base_url,
        }

    raw_response = _plain(response)
    response_hash = stable_sha256(raw_response)
    usage = _usage_from_response(response)
    if usage is not None and usage["total_tokens"] > reservation_tokens:
        # Do not accept or train from a response that violated the pre-request
        # budget envelope.  The caller will fail closed before another request.
        return {
            "status": "provider_usage_exceeded_reservation",
            "provider_usage": None,
            "budget_debit_tokens": reservation_tokens,
            "observed_usage_sha256": stable_sha256(usage),
            "provider_response_sha256": response_hash,
            "raw_provider_response": raw_response,
            "requested_model": model,
            "returned_model": str(_field(response, "model") or ""),
            "system_fingerprint": _field(response, "system_fingerprint"),
            "provider_request_id": str(_field(response, "id") or ""),
            "endpoint": base_url,
        }
    debit = reservation_tokens if usage is None else usage["total_tokens"]
    choices = _field(response, "choices", [])
    choice = choices[0] if isinstance(choices, Sequence) and len(choices) == 1 else None
    message = _field(choice, "message") if choice is not None else None
    content = str(_field(message, "content") or "")
    reasoning = str(
        _field(message, "reasoning_content")
        or _field(message, "reasoning")
        or ""
    )
    finish_reason = str(_field(choice, "finish_reason") or "")
    returned_model = str(_field(response, "model") or "")
    common = {
        "provider_usage": usage,
        "budget_debit_tokens": debit,
        "requested_model": model,
        "returned_model": returned_model,
        "returned_model_matches_requested": returned_model == model,
        "system_fingerprint": _field(response, "system_fingerprint"),
        "provider_request_id": str(_field(response, "id") or ""),
        "endpoint": base_url,
        "finish_reason": finish_reason,
        "provider_response_sha256": response_hash,
        "raw_provider_response": raw_response,
        "raw_content_sha256": sha256_text(content),
        "raw_reasoning_sha256": sha256_text(reasoning),
        "raw_reasoning_characters": len(reasoning),
    }
    if returned_model != model:
        return {**common, "status": "returned_model_mismatch"}
    if finish_reason != "stop":
        return {
            **common,
            "status": (
                "provider_output_truncated"
                if finish_reason == "length"
                else "provider_nonterminal_finish"
            ),
        }
    code = extract_scored_code(content)
    if not code.strip():
        return {**common, "status": "unparseable_final_answer"}
    compiled, passed, diagnostic, _source = (
        dart_evaluator.evaluate_dart_jit_tests_detail(
            code,
            tests,
            f"qwen37_aux_{failed.task_id}",
            timeout=verifier_timeout,
            stability_runs=stability_runs,
        )
    )
    verifier_path = Path(inspect.getsourcefile(dart_evaluator) or "").resolve()
    verification = {
        "compiled": bool(compiled),
        "passed": bool(passed),
        "harness_completion_attested": bool(passed),
        "completion_attestation": "per-run-256-bit-marker-exactly-once-v1",
        "diagnostic_sha256": sha256_text(str(diagnostic or "")),
        "tests_sha256": sha256_text(tests),
        "verifier_path": str(verifier_path),
        "verifier_sha256": sha256_file(verifier_path),
        "timeout_seconds": int(verifier_timeout),
        "stability_runs": int(stability_runs),
    }
    return {
        **common,
        "status": "verified_pass" if passed else "verification_failed",
        "code": code,
        "code_sha256": sha256_text(code),
        "verification": verification,
    }


def _materialize(
    *,
    artifact_root: Path,
    model: str,
    base_url: str,
    journal_path: Path,
    contract: Mapping[str, Any],
    state: JournalState,
    eligible: Sequence[FailedCandidate],
) -> dict[str, Path]:
    if state.complete is None:
        raise ArtifactError("cannot publish a nonterminal repair collection")
    candidate_by_task = {row.task_id: row for row in eligible}
    verified_rows: list[dict[str, Any]] = []
    for task_id in [row.task_id for row in eligible]:
        terminal = state.terminals.get(task_id)
        if terminal is None or terminal.get("status") != "verified_pass":
            continue
        code = str(terminal.get("code") or "")
        verification = terminal.get("verification") or {}
        if (
            not code
            or sha256_text(code) != terminal.get("code_sha256")
            or verification.get("passed") is not True
            or verification.get("harness_completion_attested") is not True
            or terminal.get("returned_model") != model
        ):
            raise ArtifactError(f"{task_id}: verified terminal is inconsistent")
        source = candidate_by_task[task_id]
        verified_rows.append(
            {
                "schema": OUTPUT_SCHEMA,
                "task_id": task_id,
                "target": code,
                "target_sha256": sha256_text(code),
                "target_mode": "final_dart_code_only",
                "reasoning_in_target": False,
                "training_use": "auxiliary_verified_rs_sft_hard_target_only",
                "source": {
                    "model": model,
                    "endpoint": base_url,
                    "system_fingerprint": terminal.get("system_fingerprint"),
                    "provider_request_id": terminal.get("provider_request_id"),
                    "provider_response_sha256": terminal.get(
                        "provider_response_sha256"
                    ),
                    "raw_content_sha256": terminal.get("raw_content_sha256"),
                    "raw_reasoning_sha256": terminal.get(
                        "raw_reasoning_sha256"
                    ),
                    "failed_candidate_code_sha256": source.code_sha256,
                    "priority": source.priority_name,
                },
                "attestation": {
                    "tests_sha256": verification.get("tests_sha256"),
                    "verifier_sha256": verification.get("verifier_sha256"),
                    "completion_attestation": verification.get(
                        "completion_attestation"
                    ),
                    "stability_runs": verification.get("stability_runs"),
                    "passed": True,
                },
            }
        )
    verified_payload = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in verified_rows
    )
    verified_path = artifact_root / "verified_repairs.jsonl"
    _require_exact_bytes(verified_path, verified_payload)

    statuses: dict[str, int] = {}
    for terminal in state.terminals.values():
        status = str(terminal.get("status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    journal = journal_record(journal_path)
    ledger = {
        "schema": LEDGER_SCHEMA,
        "model": model,
        "endpoint": base_url,
        "budget_cap_tokens": int(contract["budget"]["cap_tokens"]),
        "budget_debit_tokens": state.budget_debits,
        "provider_reported_actual_tokens": state.actual_usage_tokens,
        "unknown_usage_slots_charged_at_full_reservation": (
            state.unknown_usage_slots
        ),
        "remaining_tokens": int(contract["budget"]["cap_tokens"])
        - state.budget_debits,
        "logical_draws": len(state.terminals),
        "journal": journal,
    }
    ledger_path = artifact_root / "token_ledger.json"
    require_exact_or_write(ledger_path, ledger)
    report = {
        "schema": REPORT_SCHEMA,
        "model": model,
        "endpoint": base_url,
        "run_contract_sha256": canonical_sha256(contract),
        "eligible_failures": len(eligible),
        "logical_draws": len(state.terminals),
        "verified_repairs": len(verified_rows),
        "terminal_statuses": statuses,
        "verified_repairs_artifact": file_record(verified_path),
        "token_ledger": file_record(ledger_path),
        "journal": journal,
        "contamination_controls": {
            "fit_rows": EXPECTED_FIT_ROWS,
            "heldout_artifact_opened": False,
            "provider_received_tests": False,
            "provider_received_gold": False,
            "provider_received_raw_compiler_diagnostics": False,
            "provider_received_compressed_enriched_assembly": True,
            "provider_received_compressed_cfg": True,
        },
        "compatibility": {
            "qwen38_sequence_kl_import_allowed": False,
            "qwen38_cot_import_allowed": False,
            "qwen38_union_import_allowed": False,
            "auxiliary_verified_rs_sft_hard_target_import_allowed": True,
            "reason": (
                "different exact teacher snapshot/objective; only locally "
                "attested final code may cross into a separate RS-SFT stage"
            ),
        },
    }
    report_path = artifact_root / "build_report.json"
    require_exact_or_write(report_path, report)
    return {
        "verified_repairs": verified_path,
        "token_ledger": ledger_path,
        "build_report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fit-jsonl", required=True, type=Path)
    parser.add_argument("--expected-fit-sha256", required=True)
    parser.add_argument("--fit-seal", required=True, type=Path)
    parser.add_argument("--expected-fit-seal-sha256", required=True)
    parser.add_argument("--frozen-contract", required=True, type=Path)
    parser.add_argument("--expected-frozen-contract-sha256", required=True)
    parser.add_argument("--prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--prompt-manifest", required=True, type=Path)
    parser.add_argument("--expected-prompt-manifest-sha256", required=True)
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--expected-predictions-sha256", required=True)
    parser.add_argument("--score", required=True, type=Path)
    parser.add_argument("--expected-score-sha256", required=True)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(PINNED_MODELS))
    parser.add_argument(
        "--mode",
        choices=("auxiliary_verified_rs_sft_hard_targets_only",),
        default="auxiliary_verified_rs_sft_hard_targets_only",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Must be 1: sequential reservations make the per-model hard budget "
            "and one-logical-draw contract auditable."
        ),
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--token-plan-automation-authorized", action="store_true")
    parser.add_argument("--budget-tokens", type=int, default=MAX_PER_MODEL_TOKENS)
    parser.add_argument("--max-prompt-tokens", type=int, default=12_288)
    parser.add_argument("--max-output-tokens", type=int, default=12_288)
    parser.add_argument("--thinking-budget", type=int, default=8_192)
    parser.add_argument("--chat-overhead-reserve", type=int, default=512)
    parser.add_argument("--provider-usage-safety-tokens", type=int, default=2_048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--verifier-timeout-seconds", type=int, default=30)
    parser.add_argument("--stability-runs", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the sealed plan without writing or calling Qwen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.budget_tokens <= 0
        or args.budget_tokens > MAX_PER_MODEL_TOKENS
    ):
        raise ArtifactError(
            f"--budget-tokens must be in [1, {MAX_PER_MODEL_TOKENS}]"
        )
    if (
        args.max_prompt_tokens <= 0
        or args.max_output_tokens <= 0
        or args.thinking_budget <= 0
        or args.thinking_budget > args.max_output_tokens
        or args.chat_overhead_reserve < 0
        or args.provider_usage_safety_tokens < 0
        or args.stability_runs < 2
        or args.workers != 1
    ):
        raise ArtifactError(
            "invalid token/verifier settings; --workers must be exactly 1"
        )
    base_url = validate_alibaba_model_studio_base_url(
        args.base_url,
        token_plan_automation_authorized=args.token_plan_automation_authorized,
    )
    artifact_root = model_artifact_root(args.artifact_root, args.model)

    prompt_path = args.prompt_jsonl.expanduser().resolve()
    prompt_manifest_path = args.prompt_manifest.expanduser().resolve()
    _assert_not_heldout_path(prompt_path, "F2 prompt artifact")
    _assert_not_heldout_path(prompt_manifest_path, "F2 prompt manifest")
    prompts, prompt_record = load_verified_prompt_rows(
        prompt_path,
        expected_sha256=args.expected_prompt_sha256,
        expected_rows=EXPECTED_FIT_ROWS,
    )
    tokenizer, tokenizer_record = load_tokenizer(
        args.student_tokenizer_json,
        expected_sha256=args.expected_student_tokenizer_sha256,
    )
    system_prompt, prompt_manifest_record, prompt_manifest = (
        load_f2_prompt_contract(
            prompt_manifest_path,
            expected_sha256=args.expected_prompt_manifest_sha256,
            prompt_record=prompt_record,
            expected_rows=EXPECTED_FIT_ROWS,
            student_tokenizer_sha256=tokenizer_record["sha256"],
        )
    )
    prompt_by_task = {row.task_id: row for row in prompts}
    if (
        len(prompt_by_task) != EXPECTED_FIT_ROWS
        or any(
            row.system_prompt_sha256 != sha256_text(system_prompt)
            for row in prompts
        )
    ):
        raise ArtifactError("F2 prompt rows differ from the manifest contract")

    tests, fit_inputs, fit_membership = _load_fit2776(
        fit_path=args.fit_jsonl.expanduser().resolve(),
        expected_fit_sha256=args.expected_fit_sha256,
        fit_seal_path=args.fit_seal.expanduser().resolve(),
        expected_fit_seal_sha256=args.expected_fit_seal_sha256,
        frozen_contract_path=args.frozen_contract.expanduser().resolve(),
        expected_frozen_contract_sha256=args.expected_frozen_contract_sha256,
        prompt_record=prompt_record,
        prompt_manifest_record=prompt_manifest_record,
    )
    expected_ids = set(tests)
    if (
        set(prompt_by_task) != expected_ids
        or fit_membership["ordered_ids"] != [row.task_id for row in prompts]
    ):
        raise ArtifactError(
            "fit2776 and F2 prompt task membership/order are not identical"
        )
    predictions, task_results, candidate_results, score_inputs = (
        _load_scored_predictions(
            fit_record=fit_inputs["fit"],
            expected_task_ids=expected_ids,
            prediction_path=args.predictions.expanduser().resolve(),
            expected_prediction_sha256=args.expected_predictions_sha256,
            score_path=args.score.expanduser().resolve(),
            expected_score_sha256=args.expected_score_sha256,
        )
    )
    eligible, selection_stats = select_failed_candidates(
        predictions=predictions,
        task_results=task_results,
        candidate_results=candidate_results,
    )

    prompt_bindings: list[dict[str, Any]] = []
    runnable: list[FailedCandidate] = []
    for failed in eligible:
        messages = repair_messages(
            system_prompt=system_prompt,
            prompt_text=prompt_by_task[failed.task_id].text,
            failed=failed,
        )
        prompt_tokens = count_message_tokens(
            tokenizer,
            messages,
            overhead_reserve=args.chat_overhead_reserve,
        )
        reservation = (
            prompt_tokens
            + args.max_output_tokens
            + args.provider_usage_safety_tokens
        )
        within_prompt_cap = prompt_tokens <= args.max_prompt_tokens
        within_single_request_budget = reservation <= args.budget_tokens
        prompt_bindings.append(
            {
                **failed.seal(),
                "f2_text_sha256": prompt_by_task[failed.task_id].text_sha256,
                "request_messages_sha256": stable_sha256(messages),
                "estimated_prompt_tokens": prompt_tokens,
                "reservation_tokens": reservation,
                "within_prompt_cap": within_prompt_cap,
                "within_single_request_budget": within_single_request_budget,
            }
        )
        if within_prompt_cap and within_single_request_budget:
            runnable.append(failed)
    binding_by_task = {
        str(row["task_id"]): row for row in prompt_bindings
    }
    generation = {
        "n": 1,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_output_tokens),
        "extra_body": {
            "enable_thinking": True,
            "thinking_budget": int(args.thinking_budget),
            "top_k": int(args.top_k),
        },
    }
    contract = {
        "schema": RUN_CONTRACT_SCHEMA,
        "exact_pinned_model": args.model,
        "returned_model_must_equal_requested": True,
        "endpoint": base_url,
        "system_fingerprint_capture": "exact_value_including_null",
        "inputs": {
            **fit_inputs,
            **score_inputs,
            "prompt": prompt_record,
            "prompt_manifest": prompt_manifest_record,
            "student_tokenizer": tokenizer_record,
        },
        "f2_prompt_contract_sha256": stable_sha256(
            prompt_manifest["f2_prompt_contract"]
        ),
        "selection": {
            "algorithm": (
                "skip-pass_then_compiled-failed_then-parseable-noncompile-v1"
            ),
            "stats": selection_stats,
            "eligible_task_ids_sha256": stable_sha256(
                [row.task_id for row in eligible]
            ),
            "runnable_task_ids_sha256": stable_sha256(
                [row.task_id for row in runnable]
            ),
            "prompt_bindings": prompt_bindings,
            "prompt_bindings_sha256": stable_sha256(prompt_bindings),
        },
        "budget": {
            "scope": "per_exact_model",
            "cap_tokens": int(args.budget_tokens),
            "absolute_max_tokens": MAX_PER_MODEL_TOKENS,
            "reservation": (
                "estimated_prompt_plus_max_output_plus_provider_safety"
            ),
            "unknown_usage_policy": "debit_full_reservation",
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_output_tokens": int(args.max_output_tokens),
            "chat_overhead_reserve": int(args.chat_overhead_reserve),
            "provider_usage_safety_tokens": int(
                args.provider_usage_safety_tokens
            ),
        },
        "generation": generation,
        "transport": {
            "api": "synchronous_chat_completions",
            "n": 1,
            "workers": int(args.workers),
            "sdk_max_retries": 0,
            "application_retries": 0,
            "one_terminal_logical_draw_per_task": True,
            "started_without_terminal_policy": "fail_closed_never_reissue",
            "timeout_seconds": float(args.timeout_seconds),
        },
        "verifier": {
            "implementation_path": str(Path(dart_evaluator.__file__).resolve()),
            "implementation_sha256": sha256_file(
                Path(dart_evaluator.__file__).resolve()
            ),
            "timeout_seconds": int(args.verifier_timeout_seconds),
            "stability_runs": int(args.stability_runs),
            "completion_attestation": (
                "per-run-256-bit-marker-exactly-once-v1"
            ),
        },
        "contamination_contract": {
            "fit_rows": EXPECTED_FIT_ROWS,
            "heldout_artifact_opened": False,
            "tests_in_provider_messages": False,
            "gold_in_provider_messages": False,
            "raw_diagnostic_in_provider_messages": False,
            "best_failed_code_in_provider_messages": True,
            "sanitized_compiler_feedback_in_provider_messages": True,
            "compressed_enriched_assembly_in_provider_messages": True,
            "compressed_cfg_in_provider_messages": True,
        },
        "training_compatibility": {
            "qwen38_sequence_kl": False,
            "qwen38_cot": False,
            "qwen38_union": False,
            "auxiliary_verified_rs_sft_hard_targets_only": True,
        },
        "mode": args.mode,
    }
    contract_sha = canonical_sha256(contract)
    plan_summary = {
        "model": args.model,
        "endpoint": base_url,
        "fit_rows": EXPECTED_FIT_ROWS,
        "eligible_failures": len(eligible),
        "runnable_failures": len(runnable),
        "selection_stats": selection_stats,
        "budget_tokens": args.budget_tokens,
        "run_contract_sha256": contract_sha,
        "artifact_root": str(artifact_root),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(plan_summary, sort_keys=True))
        return

    api_key = str(os.environ.get(args.api_key_env) or "").strip()
    if not api_key:
        raise ArtifactError(f"missing API key environment variable {args.api_key_env}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    contract_path = artifact_root / "run_contract.json"
    require_exact_or_write(contract_path, contract)
    journal_path = artifact_root / "attempts.journal.jsonl"
    events = load_journal(journal_path)
    if not events:
        append_event(
            journal_path,
            {
                "event": "repair_header",
                "schema": JOURNAL_SCHEMA,
                "created_at": utc_now(),
                "run_contract": file_record(contract_path),
                "run_contract_sha256": contract_sha,
                "model": args.model,
                "endpoint": base_url,
            },
        )
        events = load_journal(journal_path)
    state = validate_journal_state(
        events,
        contract_sha256=contract_sha,
        budget_cap=args.budget_tokens,
    )
    if state.orphan_task_id is not None:
        raise ArtifactError(
            f"journal has an interrupted request for {state.orphan_task_id}; "
            "fail-closed one-draw policy forbids automatic reissue"
        )
    if state.complete is None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment dependency
            raise ArtifactError("the openai package is required") from exc
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(args.timeout_seconds),
            max_retries=0,
        )
        for failed in runnable:
            if failed.task_id in state.terminals:
                continue
            binding = binding_by_task[failed.task_id]
            reservation = int(binding["reservation_tokens"])
            if state.budget_debits + reservation > args.budget_tokens:
                continue
            messages = repair_messages(
                system_prompt=system_prompt,
                prompt_text=prompt_by_task[failed.task_id].text,
                failed=failed,
            )
            started = append_event(
                journal_path,
                {
                    "event": "repair_slot_started",
                    "schema": JOURNAL_SCHEMA,
                    "started_at": utc_now(),
                    "task_id": failed.task_id,
                    "logical_draw_index": len(state.terminals),
                    "reservation_tokens": reservation,
                    "budget_debits_before": state.budget_debits,
                    "candidate": failed.seal(),
                    "request_messages_sha256": binding[
                        "request_messages_sha256"
                    ],
                    "estimated_prompt_tokens": binding[
                        "estimated_prompt_tokens"
                    ],
                    "requested_model": args.model,
                    "endpoint": base_url,
                },
            )
            result = invoke_one(
                client=client,
                model=args.model,
                base_url=base_url,
                messages=messages,
                generation=generation,
                failed=failed,
                tests=tests[failed.task_id],
                reservation_tokens=reservation,
                verifier_timeout=args.verifier_timeout_seconds,
                stability_runs=args.stability_runs,
            )
            terminal_payload = {
                key: value for key, value in result.items() if key != "code"
            }
            if "code" in result:
                terminal_payload["code"] = result["code"]
            append_event(
                journal_path,
                {
                    "event": "repair_slot_terminal",
                    "schema": JOURNAL_SCHEMA,
                    "terminal_at": utc_now(),
                    "task_id": failed.task_id,
                    "start_event_sha256": started["journal_event_sha256"],
                    "reservation_tokens": reservation,
                    **terminal_payload,
                },
            )
            state = validate_journal_state(
                load_journal(journal_path),
                contract_sha256=contract_sha,
                budget_cap=args.budget_tokens,
            )
            if result["status"] == "provider_usage_exceeded_reservation":
                raise ArtifactError(
                    "provider reported usage outside the pessimistic reservation; "
                    "collection stopped before any further request"
                )

        append_event(
            journal_path,
            {
                "event": "collection_complete",
                "schema": JOURNAL_SCHEMA,
                "completed_at": utc_now(),
                "budget_debit_tokens": state.budget_debits,
                "logical_draws": len(state.terminals),
                "verified_passes": sum(
                    event.get("status") == "verified_pass"
                    for event in state.terminals.values()
                ),
                "completion_reason": (
                    "eligible_tasks_exhausted_or_remaining_budget_cannot_fit_"
                    "another_pessimistic_reservation"
                ),
            },
        )
        state = validate_journal_state(
            load_journal(journal_path),
            contract_sha256=contract_sha,
            budget_cap=args.budget_tokens,
        )

    outputs = _materialize(
        artifact_root=artifact_root,
        model=args.model,
        base_url=base_url,
        journal_path=journal_path,
        contract=contract,
        state=state,
        eligible=eligible,
    )
    print(
        json.dumps(
            {
                **plan_summary,
                "logical_draws": len(state.terminals),
                "verified_repairs": sum(
                    event.get("status") == "verified_pass"
                    for event in state.terminals.values()
                ),
                "budget_debit_tokens": state.budget_debits,
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
