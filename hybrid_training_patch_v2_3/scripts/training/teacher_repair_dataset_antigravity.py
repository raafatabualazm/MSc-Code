#!/usr/bin/env python3
"""Verifier-bounded frontier repair with hidden acceptance and FACTS grounding.

Stages are resumable and deliberately offline from policy optimization:

``collect``
    Align train/development rollouts, evaluate only deterministic feedback tests,
    record per-test pass vectors, retain bounded positives, and choose informative
    failures.  Frozen/evaluation-only or non-Phase0 rows are rejected.
``teacher``
    Ask a frontier model for a strict JSON diagnosis/repair.  Raw tests and
    Expected/Actual values are never included.  The model must copy the supplied
    mechanical FACTS object exactly.
``build``
    Treat every rollout/repair as untrusted.  Replay hidden acceptance tests and
    the complete harness, enforce exact FACTS/contract consistency, and emit only
    independently verified alternatives for RS-SFT and the GRPO anchor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.hybrid_data_controls import (  # noqa: E402
    FACT_FIELDS,
    SCHEMA_VERSION,
    assert_training_approved,
    _subset_test_harness,
    candidate_expect_lines,
    candidate_fact_match,
    facts_comment,
    mechanical_facts,
    normalize_fact_sheet,
    parse_facts_comment,
    read_jsonl,
    sanitize_verifier_diagnostic,
    signature_text,
    source_text,
    task_key,
    write_jsonl,
)

ALLOWED_DATA_ROLES = {"train", "development", "dev"}
FAILURE_CLASSES = {
    "syntax",
    "type_contract",
    "missing_import",
    "control_flow",
    "data_flow",
    "arithmetic",
    "boundary_case",
    "state_effect",
    "unsupported",
    "unknown",
}

FACT_SCHEMA_TEXT = json.dumps({field: f"COPY {field} EXACTLY" for field in FACT_FIELDS}, indent=2)
SYSTEM_PROMPT = f"""You are a senior compiler and reverse-engineering critic.
Repair a candidate Dart decompilation using only the supplied assembly, required
contract, current candidate, mechanical FACTS object, and redacted verifier
summary. Do not infer from hidden tests; they are not supplied. Do not reveal a
hidden chain of thought. Return exactly one JSON object with these keys:

{{
  "failure_class": "syntax | type_contract | missing_import | control_flow | data_flow | arithmetic | boundary_case | state_effect | unsupported | unknown",
  "confidence": 0.0,
  "fact_claims": {FACT_SCHEMA_TEXT},
  "failure_evidence": ["short evidence-grounded observation"],
  "repair_actions": ["concrete correction"],
  "repaired_code": "complete top-level Dart implementation"
}}

fact_claims must be copied byte-for-value from MECHANICAL FACTS in the user
message; do not reinterpret, omit, or add fields. repaired_code must contain the
required top-level function, no main(), no tests, and no Markdown fence. The
hidden acceptance verifier and exact FACTS gate, not confidence, decide entry.
"""


@dataclass(frozen=True)
class EvalResult:
    compiled: bool
    passed: bool
    diagnostic: str
    passed_count: int
    total: int
    test_passes: tuple[bool, ...]

    @property
    def pass_ratio(self) -> float:
        return self.passed_count / self.total if self.total else 0.0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)


def load_prediction_pool(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig").lstrip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"{source} must contain a JSON list of objects")
        return value
    return read_jsonl(source)


def canonical_code(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_code(text: str) -> str:
    value = (text or "").strip()
    match = re.search(r"```(?:dart)?\s*(.*?)```", value, re.I | re.S)
    return (match.group(1) if match else value).strip()


def compact_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 3 // 5
    tail = max_chars - head
    return text[:head] + "\n... <middle omitted> ...\n" + text[-tail:]


def load_exact_evaluator():
    try:
        from scripts.evaluation.graph_compile_at_k_antigravity import (  # type: ignore
            evaluate_dart_jit_tests_detail,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import the project-aligned Dart evaluator. Overlay this patch into the repository root."
        ) from exc
    return evaluate_dart_jit_tests_detail


def evaluate_candidate(
    candidate: str,
    tests: str,
    identifier: str,
    timeout: int,
    *,
    collect_per_test: bool = False,
) -> EvalResult:
    evaluator = load_exact_evaluator()
    compiled, passed, diagnostic, _source = evaluator(candidate, tests, identifier, timeout=timeout)
    assertions = candidate_expect_lines(tests)
    passes: list[bool] = []
    if collect_per_test and bool(compiled) and assertions:
        for index, assertion in enumerate(assertions):
            one = _subset_test_harness(tests, {assertion})
            one_compiled, one_passed, _diag, _src = evaluator(
                candidate,
                one,
                f"{identifier}_case{index}",
                timeout=timeout,
            )
            passes.append(bool(one_compiled and one_passed))
    elif assertions:
        passes = [bool(passed)] * len(assertions)
    total = len(assertions)
    passed_count = sum(passes) if passes else (total if passed else 0)
    return EvalResult(
        compiled=bool(compiled),
        passed=bool(passed),
        diagnostic=sanitize_verifier_diagnostic(str(diagnostic or "")),
        passed_count=passed_count,
        total=total,
        test_passes=tuple(passes),
    )


def validate_phase0_artifact(dataset_path: str | Path, report_path: str | Path | None) -> dict[str, Any]:
    """Bind teacher harvesting to the exact immutable Phase-0 output file.

    The curriculum always supplies the signed/hash-bound report. A missing
    report is accepted only for small standalone fixtures whose rows already
    carry embedded Phase-0 approval, neutral-contract, and hidden-test metadata.
    Set HYBRID_REQUIRE_PHASE0_REPORT=1 to make the report mandatory for every
    direct invocation as well.
    """
    if not report_path:
        if os.environ.get("HYBRID_REQUIRE_PHASE0_REPORT", "0") == "1":
            raise ValueError("--phase0_report is required by HYBRID_REQUIRE_PHASE0_REPORT=1")
        rows = read_jsonl(dataset_path)
        for index, row in enumerate(rows):
            metadata = row.get("hybrid_metadata") or {}
            if metadata.get("phase0_approved") is not True:
                raise ValueError(f"dataset row {index} lacks embedded Phase-0 approval")
            if not (metadata.get("neutralized") or metadata.get("neutral_contract")):
                raise ValueError(f"dataset row {index} lacks embedded neutral-contract provenance")
            if not row.get("acceptance_tests"):
                raise ValueError(f"dataset row {index} lacks hidden acceptance tests")
        return {
            "stage": "embedded_phase0_fixture",
            "short_rows": len(rows),
            "outputs": {"short": file_record(dataset_path)},
            "frozen_eval_overlaps": [],
            "preparation_failures": [],
        }
    report_file = Path(report_path).expanduser().resolve()
    if not report_file.is_file():
        raise FileNotFoundError(report_file)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("stage") != "phase0_prepare":
        raise ValueError("--phase0_report is not a Phase-0 preparation report")
    outputs = report.get("outputs") or {}
    observed = file_record(dataset_path)
    approved = {
        key: (rec or {}).get("sha256")
        for key, rec in outputs.items()
        if (rec or {}).get("sha256")
    }
    if observed.get("sha256") not in set(approved.values()):
        raise ValueError(
            "dataset hash does not match any approved Phase-0 output artifact: "
            f"observed={observed.get('sha256')} approved_keys={sorted(approved)} "
            f"approved={approved}"
        )
    if report.get("frozen_eval_overlaps"):
        raise ValueError("Phase-0 report contains frozen-evaluation overlap")
    if report.get("preparation_failures"):
        raise ValueError("Phase-0 report contains preparation failures")
    return report



def validate_prediction_artifact(
    dataset_path: str | Path,
    predictions_path: str | Path,
    predictions: list[dict[str, Any]],
    *,
    provenance_path: str | Path | None = None,
    expected_checkpoint: str | Path | None = None,
    required: bool = False,
) -> dict[str, Any]:
    """Bind a rollout pool to the exact dataset/checkpoint that produced it.

    The inference sidecar is not a cryptographic signature, but hashing the
    dataset, output, and checkpoint turns accidental/stale pool reuse into a
    fail-closed error.  Phase-0 curriculum runs require the sidecar; tiny direct
    unit-test fixtures may omit it when ``required`` is false.
    """
    prediction_file = Path(predictions_path).expanduser().resolve()
    sidecar = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path
        else Path(str(prediction_file) + ".provenance.json")
    )
    if not sidecar.is_file():
        if required or os.environ.get("HYBRID_REQUIRE_PREDICTION_PROVENANCE", "0") == "1":
            raise FileNotFoundError(
                "rollout provenance is required for Phase-0 harvesting: " + str(sidecar)
            )
        return {
            "status": "not_required",
            "predictions": file_record(prediction_file),
            "provenance": None,
        }

    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    dataset_record = file_record(dataset_path)
    predictions_record = file_record(prediction_file)
    failures: list[str] = []

    recorded_dataset = provenance.get("dataset") or {}
    if recorded_dataset.get("sha256") != dataset_record["sha256"]:
        failures.append(
            "dataset SHA-256 mismatch "
            f"({recorded_dataset.get('sha256')} != {dataset_record['sha256']})"
        )
    recorded_output = provenance.get("output") or {}
    if recorded_output.get("sha256") != predictions_record["sha256"]:
        failures.append(
            "prediction output SHA-256 mismatch "
            f"({recorded_output.get('sha256')} != {predictions_record['sha256']})"
        )
    try:
        recorded_row_count = int(provenance.get("row_count", -1))
    except (TypeError, ValueError):
        recorded_row_count = -1
    if recorded_row_count != len(predictions):
        failures.append(
            f"row_count mismatch ({provenance.get('row_count')} != {len(predictions)})"
        )
    if provenance.get("scoring_tests_visible_to_policy") is not False:
        failures.append("provenance does not explicitly assert scoring tests were hidden")
    if not str(provenance.get("prompt_schema_version") or "").strip():
        failures.append("missing prompt_schema_version")

    ablation = provenance.get("graph_input_ablation") or {}
    if str(ablation.get("mode") or "").lower() != "none":
        failures.append(
            f"teacher harvesting requires unablated rollouts, got {ablation.get('mode')!r}"
        )
    if bool(ablation.get("final_context_zeroed")):
        failures.append("teacher harvesting cannot use a zeroed graph context")
    if ((provenance.get("graph_prefix_gate") or {}).get("override_requested")) not in (None, ""):
        failures.append("teacher harvesting cannot use a manual graph-prefix gate override")

    generation = provenance.get("generation") or {}
    try:
        expected_samples = int(generation.get("num_samples", 0))
    except (TypeError, ValueError):
        expected_samples = 0
    if expected_samples <= 0:
        failures.append("generation.num_samples must be positive")

    source_lines: list[int] = []
    for index, row in enumerate(predictions):
        try:
            source_line = int(row.get("source_line"))
        except (TypeError, ValueError):
            failures.append(f"prediction row {index} has no valid source_line")
            continue
        source_lines.append(source_line)
        candidates = row.get("predictions") or []
        if not isinstance(candidates, list):
            failures.append(f"prediction row {index} predictions is not a list")
        elif expected_samples > 0 and len(candidates) != expected_samples:
            failures.append(
                f"prediction row {index} has {len(candidates)} candidates; expected {expected_samples}"
            )
    dataset_rows = len(read_jsonl(dataset_path))
    out_of_range = [value for value in source_lines if not 1 <= value <= dataset_rows]
    if out_of_range:
        failures.append(f"source_line values out of dataset range: {out_of_range[:8]}")
    if len(source_lines) != len(set(source_lines)):
        failures.append("source_line values are not unique")

    checkpoint_record = provenance.get("checkpoint") or {}
    checkpoint_load = provenance.get("checkpoint_load") or {}
    if checkpoint_record:
        if checkpoint_load.get("status") != "passed":
            failures.append(
                "checkpoint load contract was not passed "
                f"(status={checkpoint_load.get('status')!r})"
            )
    if expected_checkpoint:
        expected_checkpoint_record = file_record(expected_checkpoint)
        if checkpoint_record.get("sha256") != expected_checkpoint_record["sha256"]:
            failures.append(
                "rollout checkpoint SHA-256 mismatch "
                f"({checkpoint_record.get('sha256')} != {expected_checkpoint_record['sha256']})"
            )
        if checkpoint_load.get("status") != "passed":
            failures.append("expected checkpoint was not loaded under a passed architecture contract")
    elif required and not checkpoint_record:
        failures.append("Phase-0 rollout provenance has no checkpoint record")

    report = {
        "status": "passed" if not failures else "failed",
        "dataset": dataset_record,
        "predictions": predictions_record,
        "provenance": file_record(sidecar),
        "checkpoint": checkpoint_record or None,
        "checkpoint_load_status": checkpoint_load.get("status"),
        "prompt_schema_version": provenance.get("prompt_schema_version"),
        "generation": generation,
        "graph_input_ablation": ablation,
        "source_line_count": len(source_lines),
        "unique_source_line_count": len(set(source_lines)),
        "failures": failures,
    }
    if failures:
        raise ValueError("invalid rollout provenance: " + "; ".join(failures))
    return report


def validate_collected_artifact(collected_path: str | Path) -> dict[str, Any]:
    """Validate the collector manifest and the exact selected-record bytes."""
    collected_file = Path(collected_path).expanduser().resolve()
    manifest_path = Path(str(collected_file) + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing collect manifest required for provenance: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = file_record(collected_file)
    recorded = manifest.get("output") or {}
    if recorded.get("sha256") != observed["sha256"]:
        raise ValueError(
            "collected JSONL does not match its manifest: "
            f"{recorded.get('sha256')} != {observed['sha256']}"
        )
    prediction_validation = manifest.get("prediction_validation") or {}
    validation_status = prediction_validation.get("status")
    provenance_required = bool(manifest.get("phase0_report")) or (
        os.environ.get("HYBRID_REQUIRE_PREDICTION_PROVENANCE", "0") == "1"
    )
    if validation_status != "passed" and (
        provenance_required or validation_status != "not_required"
    ):
        raise ValueError(
            "collect manifest lacks a passed rollout-provenance contract"
        )
    return manifest


def validate_data_role(value: str) -> str:
    role = value.strip().lower()
    if role not in ALLOWED_DATA_ROLES:
        raise ValueError(f"forbidden data role: {value!r}")
    return "development" if role == "dev" else role


def validate_task_row(
    row: dict[str, Any],
    expected_role: str,
    index: int,
    *,
    require_manifest_fields: bool = True,
) -> None:
    metadata = row.get("hybrid_metadata") or {}
    assert_training_approved(row)
    if require_manifest_fields:
        replay = metadata.get("reference_test_replay") or {}
        if replay.get("passed") is not True:
            raise ValueError(f"row {index} lacks successful Phase-0 reference replay")
        if not metadata.get("source_overlap_hash"):
            raise ValueError(f"row {index} lacks the Phase-0 normalized-source overlap hash")
    observed_role = str(metadata.get("data_role") or "").lower()
    if observed_role == "dev":
        observed_role = "development"
    # The cryptographic Phase-0 manifest is authoritative.  A role, when
    # present, must agree; a missing legacy role is filled by the collector.
    if observed_role and observed_role != expected_role:
        raise ValueError(
            f"row {index} role mismatch: metadata={observed_role!r}, requested={expected_role!r}"
        )
    if not row.get("binary_facts"):
        raise ValueError(f"row {index} has no Phase-0 binary_facts")
    # Single-assertion rows may intentionally have no visible feedback split.
    # They remain eligible for hidden verification but are not teacher-guided.
    if not str(row.get("acceptance_tests") or "").strip():
        raise ValueError(f"row {index} has no hidden acceptance_tests")


def align_task(
    dataset: list[dict[str, Any]],
    prediction: dict[str, Any],
    pred_index: int,
    by_key: dict[str, list[int]],
) -> tuple[int, dict[str, Any]]:
    source_line = prediction.get("source_line")
    if source_line not in (None, ""):
        index = int(source_line) - 1
        if not 0 <= index < len(dataset):
            raise IndexError(f"prediction row {pred_index + 1} source_line out of range")
        return index, dataset[index]
    key = task_key(prediction, pred_index)
    matches = by_key.get(key, [])
    if len(matches) != 1:
        raise ValueError(f"prediction row {pred_index + 1} key={key!r} matched {len(matches)} rows")
    return matches[0], dataset[matches[0]]


def candidate_record(
    task: dict[str, Any],
    dataset_index: int,
    prediction: dict[str, Any],
    candidate: str,
    candidate_index: int,
    result: EvalResult,
    data_role: str,
) -> dict[str, Any]:
    key = task_key(task, dataset_index)
    candidate_id = sha256_text(f"{key}\0{canonical_code(candidate)}")[:24]
    return {
        "schema_version": SCHEMA_VERSION,
        "data_role": data_role,
        "task_key": key,
        "dataset_index": dataset_index,
        "source_line": dataset_index + 1,
        "candidate_index": candidate_index,
        "failure_id": candidate_id,
        "candidate_sha256": sha256_text(candidate),
        "candidate": candidate,
        "verifier": {
            "compiled": result.compiled,
            "passed_feedback_suite": result.passed,
            "pass_ratio": result.pass_ratio,
            "passed_count": result.passed_count,
            "total": result.total,
            "test_passes": list(result.test_passes),
            "diagnostic": result.diagnostic,
        },
        "task": copy.deepcopy(task),
        "prediction_metadata": {
            "id": prediction.get("id"),
            "filename": prediction.get("filename"),
            "graph_input_ablation": prediction.get("graph_input_ablation"),
        },
    }


def collect_command(args: argparse.Namespace) -> int:
    role = validate_data_role(args.data_role)
    phase0_report = validate_phase0_artifact(args.dataset, args.phase0_report)
    dataset = read_jsonl(args.dataset)
    predictions = load_prediction_pool(args.predictions)
    if not dataset or not predictions:
        raise SystemExit("dataset and prediction pool must both be non-empty")
    prediction_validation = validate_prediction_artifact(
        args.dataset,
        args.predictions,
        predictions,
        provenance_path=args.prediction_provenance or None,
        expected_checkpoint=args.expected_checkpoint or None,
        required=bool(args.phase0_report),
    )
    for index, row in enumerate(dataset):
        validate_task_row(
            row, role, index, require_manifest_fields=bool(args.phase0_report)
        )

    by_key: dict[str, list[int]] = {}
    for index, row in enumerate(dataset):
        by_key.setdefault(task_key(row, index), []).append(index)
    jobs: list[tuple[dict[str, Any], int, dict[str, Any], str, int]] = []
    seen: set[tuple[str, str]] = set()
    for pred_index, prediction in enumerate(predictions):
        dataset_index, task = align_task(dataset, prediction, pred_index, by_key)
        if not str(task.get("feedback_tests") or "").strip():
            continue
        for candidate_index, raw in enumerate(prediction.get("predictions") or []):
            candidate = extract_code(str(raw or ""))
            if len(candidate) < args.min_candidate_chars:
                continue
            dedupe = (task_key(task, dataset_index), canonical_code(candidate))
            if dedupe in seen:
                continue
            seen.add(dedupe)
            jobs.append((task, dataset_index, prediction, candidate, candidate_index))
    if args.limit_candidates:
        jobs = jobs[: args.limit_candidates]
    print(f"Evaluating {len(jobs)} unique candidates on feedback tests with {args.workers} workers")

    def run(job):
        task, dataset_index, prediction, candidate, candidate_index = job
        visible_tests = str(task.get("feedback_tests") or "")
        if not visible_tests:
            # No visible assertion means no teacher-safe reward signal.  Return
            # a synthetic no-signal result; the task is skipped below.
            result = EvalResult(False, False, "no visible feedback assertions", 0, 0, ())
        else:
            result = evaluate_candidate(
                candidate,
                visible_tests,
                task_key(task, dataset_index),
                args.timeout,
                collect_per_test=True,
            )
        return candidate_record(task, dataset_index, prediction, candidate, candidate_index, result, role)

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done, record in enumerate(pool.map(run, jobs), 1):
            records.append(record)
            if done % 25 == 0 or done == len(jobs):
                print(f"  feedback verifier {done}/{len(jobs)}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if int(record["verifier"].get("total") or 0) <= 0:
            continue
        grouped.setdefault(record["task_key"], []).append(record)
    selected: list[dict[str, Any]] = []
    for group in grouped.values():
        passing = [row for row in group if row["verifier"]["passed_feedback_suite"]]
        passing.sort(key=lambda row: row["candidate_index"])
        for row in passing[: args.max_positives_per_task]:
            row["disposition"] = "feedback_positive"
            selected.append(row)
        failures = [row for row in group if not row["verifier"]["passed_feedback_suite"]]
        failures.sort(
            key=lambda row: (
                0 if row["verifier"]["compiled"] else 1,
                -float(row["verifier"]["pass_ratio"]),
                row["candidate_index"],
            )
        )
        for row in failures[: args.max_failures_per_task]:
            row["disposition"] = "needs_teacher"
            selected.append(row)

    rollout_binding = {
        "provenance_sha256": (prediction_validation.get("provenance") or {}).get("sha256"),
        "dataset_sha256": (prediction_validation.get("dataset") or {}).get("sha256"),
        "predictions_sha256": (prediction_validation.get("predictions") or {}).get("sha256"),
        "checkpoint_sha256": (prediction_validation.get("checkpoint") or {}).get("sha256"),
        "prompt_schema_version": prediction_validation.get("prompt_schema_version"),
    }
    for row in selected:
        row["rollout_binding"] = copy.deepcopy(rollout_binding)
    count = write_jsonl(args.out, selected)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "collect",
        "data_role": role,
        "dataset": file_record(args.dataset),
        "phase0_report": file_record(args.phase0_report) if args.phase0_report else None,
        "phase0_approved_rows": int(phase0_report.get("short_rows", 0)),
        "predictions": file_record(args.predictions),
        "prediction_provenance": (
            file_record(args.prediction_provenance)
            if args.prediction_provenance
            else file_record(str(args.predictions) + ".provenance.json")
            if Path(str(args.predictions) + ".provenance.json").is_file()
            else None
        ),
        "prediction_validation": prediction_validation,
        "evaluated_candidates": len(records),
        "selected_records": count,
        "feedback_positives": sum(row["disposition"] == "feedback_positive" for row in selected),
        "teacher_failures": sum(row["disposition"] == "needs_teacher" for row in selected),
        "tasks": len(grouped),
        "hidden_acceptance_tests_exposed": False,
        "output": file_record(args.out),
    }
    Path(str(args.out) + ".manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def make_teacher_prompt(
    record: dict[str, Any],
    visibility: str,
    max_assembly_chars: int,
    max_candidate_chars: int,
) -> str:
    task = record["task"]
    verifier = record["verifier"]
    status: dict[str, Any] = {"compiled": bool(verifier["compiled"])}
    if visibility in {"summary", "diagnostics"}:
        status.update(
            {
                "feedback_pass_ratio": float(verifier["pass_ratio"]),
                "feedback_passed": int(verifier["passed_count"]),
                "feedback_total": int(verifier.get("total", verifier.get("total_count", 0))),
            }
        )
    if visibility == "diagnostics":
        status["feedback_pass_vector"] = list(verifier.get("test_passes") or [])
        status["diagnostic"] = sanitize_verifier_diagnostic(str(verifier.get("diagnostic") or ""))
    facts = normalize_fact_sheet(task.get("binary_facts") or mechanical_facts(task))
    sections = [
        f"TASK ID: {record['task_key']}",
        f"REQUIRED CONTRACT:\n{signature_text(task)}",
        "MECHANICAL FACTS — COPY THIS OBJECT EXACTLY:\n" + json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True),
        "ASSEMBLY / BINARY EVIDENCE:\n" + compact_middle(str(task.get("assembly") or ""), max_assembly_chars),
        "CURRENT CANDIDATE:\n" + compact_middle(str(record.get("candidate") or ""), max_candidate_chars),
        "REDACTED FEEDBACK STATUS:\n" + json.dumps(status, indent=2, ensure_ascii=False),
        "No test source, assertion inputs, expected values, actual values, or hidden acceptance results are supplied.",
        "Return the strict JSON repair object now.",
    ]
    return "\n\n".join(sections)


def openai_batch_row(record: dict[str, Any], model: str, prompt: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "custom_id": record["failure_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": max_output_tokens,
            "store": False,
        },
    }


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        newline = cleaned.find("\n")
        if newline >= 0:
            cleaned = cleaned[newline + 1 : -3].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher response is not a JSON object")
    failure_class = str(value.get("failure_class", "unknown")).lower().strip()
    value["failure_class"] = failure_class if failure_class in FAILURE_CLASSES else "unknown"
    value["confidence"] = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    for key in ("failure_evidence", "repair_actions"):
        items = value.get(key, [])
        if not isinstance(items, list):
            items = [items]
        value[key] = [str(item).strip() for item in items if str(item).strip()][:12]
    raw_claims = value.get("fact_claims")
    if not isinstance(raw_claims, dict):
        # Backward compatibility with v1 teacher outputs. New requests use
        # fact_claims so the field name reflects that these values are audited.
        raw_claims = value.get("assembly_facts")
    if not isinstance(raw_claims, dict):
        raise ValueError("teacher response has no fact_claims object")
    value["fact_claims"] = normalize_fact_sheet(raw_claims)
    value.pop("assembly_facts", None)
    value["repaired_code"] = extract_code(str(value.get("repaired_code") or ""))
    if not value["repaired_code"]:
        raise ValueError("teacher response has no repaired_code")
    return value


def response_output_text(body: dict[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    chunks: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
            elif isinstance(text, dict) and isinstance(text.get("value"), str):
                chunks.append(text["value"])
    return "\n".join(chunks)


def teacher_command(args: argparse.Namespace) -> int:
    validate_collected_artifact(args.collected)
    records = [row for row in read_jsonl(args.collected) if row.get("disposition") == "needs_teacher"]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no needs_teacher records found")
    prompts = {
        row["failure_id"]: make_teacher_prompt(
            row,
            args.test_visibility,
            args.max_assembly_chars,
            args.max_candidate_chars,
        )
        for row in records
    }
    if args.mode == "batch":
        count = write_jsonl(
            args.out,
            [openai_batch_row(row, args.model, prompts[row["failure_id"]], args.max_output_tokens) for row in records],
        )
        print(f"wrote {count} /v1/responses Batch API requests to {args.out}")
        return 0

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the official SDK: pip install -U openai") from exc
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable {args.api_key_env}")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    client = OpenAI(**kwargs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if out_path.exists() and args.resume:
        completed = {str(row["failure_id"]) for row in read_jsonl(out_path) if row.get("failure_id")}
    pending = [row for row in records if row["failure_id"] not in completed]

    def call(record: dict[str, Any]) -> dict[str, Any]:
        failure_id = record["failure_id"]
        last_error = ""
        for attempt in range(args.retries):
            try:
                response = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "developer", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompts[failure_id]},
                    ],
                    max_output_tokens=args.max_output_tokens,
                    store=False,
                )
                raw_text = str(getattr(response, "output_text", "") or "")
                return {
                    "schema_version": SCHEMA_VERSION,
                    "failure_id": failure_id,
                    "task_key": record["task_key"],
                    "model": args.model,
                    "teacher_test_visibility": args.test_visibility,
                    "teacher_prompt_sha256": sha256_text(prompts[failure_id]),
                    "response_id": getattr(response, "id", None),
                    "raw_text": raw_text,
                    "parsed": parse_json_object(raw_text),
                    "error": None,
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < args.retries:
                    time.sleep(min(args.retry_max_seconds, args.retry_base_seconds * (2**attempt)))
        return {
            "schema_version": SCHEMA_VERSION,
            "failure_id": failure_id,
            "task_key": record["task_key"],
            "model": args.model,
            "teacher_test_visibility": args.test_visibility,
            "teacher_prompt_sha256": sha256_text(prompts[failure_id]),
            "raw_text": "",
            "parsed": None,
            "error": last_error,
        }

    mode = "a" if out_path.exists() and args.resume else "w"
    with out_path.open(mode, encoding="utf-8", newline="\n") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(call, row): row for row in pending}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"  teacher {done}/{len(pending)} {result['failure_id']} {'ok' if result.get('parsed') else 'error'}")
    return 0


def load_teacher_map(path: str | Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("failure_id"):
            parsed = row.get("parsed")
            if parsed is None and row.get("raw_text"):
                try:
                    parsed = parse_json_object(str(row["raw_text"]))
                except Exception:
                    parsed = None
            mapping[str(row["failure_id"])] = {"parsed": parsed, "raw": row}
            continue
        custom_id = row.get("custom_id")
        if not custom_id:
            continue
        body = ((row.get("response") or {}).get("body") or {})
        raw_text = response_output_text(body)
        try:
            parsed = parse_json_object(raw_text)
        except Exception:
            parsed = None
        mapping[str(custom_id)] = {"parsed": parsed, "raw": row, "raw_text": raw_text}
    return mapping


def teacher_model_name(entry: dict[str, Any] | None) -> str | None:
    if not entry:
        return None
    raw = entry.get("raw") or {}
    if raw.get("model"):
        return str(raw["model"])
    body = ((raw.get("response") or {}).get("body") or {})
    return str(body.get("model")) if body.get("model") else None


def teacher_test_visibility(
    entry: dict[str, Any] | None,
    declared_visibility: str,
) -> str:
    """Resolve and validate the teacher's verifier-feedback visibility."""
    raw = (entry or {}).get("raw") or {}
    recorded = str(raw.get("teacher_test_visibility") or "").strip().lower()
    if recorded and recorded not in {"none", "summary", "diagnostics"}:
        raise ValueError(f"invalid recorded teacher_test_visibility={recorded!r}")
    declared = str(declared_visibility or "").strip().lower()
    if declared not in {"unknown", "none", "summary", "diagnostics"}:
        raise ValueError(f"invalid declared teacher_test_visibility={declared!r}")
    if recorded and declared != "unknown" and recorded != declared:
        raise ValueError(
            "teacher visibility provenance mismatch: "
            f"response={recorded!r}, build={declared!r}"
        )
    resolved = recorded or declared
    if resolved == "unknown":
        raise ValueError(
            "teacher repair response lacks feedback-visibility provenance; pass "
            "--teacher_test_visibility matching the request-generation setting"
        )
    return resolved


def positive_training_row(task: dict[str, Any], code: str, metadata: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(task)
    target = code
    if parse_facts_comment(target) is None:
        target = facts_comment(task.get("binary_facts") or mechanical_facts(task)) + "\n" + target
    if "source" in row or "dart_source" not in row:
        row["source"] = target
    if "dart_source" in row:
        row["dart_source"] = target
    inherited = dict(row.get("hybrid_metadata") or {})
    inherited.update(metadata)
    inherited["phase0_approved"] = True
    row["hybrid_metadata"] = inherited
    return row


def build_command(args: argparse.Namespace) -> int:
    role = validate_data_role(args.data_role)
    if args.facts_gate_mode == "off":
        raise SystemExit(
            "--facts_gate_mode=off cannot certify verified RS-SFT rows. "
            "Use signature, conservative, or strict so facts_gate_passed records "
            "a check that actually ran."
        )
    collected_manifest = validate_collected_artifact(args.collected)
    dataset_path = ((collected_manifest.get("dataset") or {}).get("path"))
    if not dataset_path:
        raise ValueError("collect manifest does not identify its Phase-0 dataset")
    validate_phase0_artifact(dataset_path, args.phase0_report)
    collected = read_jsonl(args.collected)
    for index, record in enumerate(collected):
        validate_task_row(
            record["task"], role, index, require_manifest_fields=bool(args.phase0_report)
        )
        if str(record.get("data_role") or "") not in {role, "dev" if role == "development" else role}:
            raise ValueError(f"collected row {index} role mismatch")
    teacher = load_teacher_map(args.teacher_responses) if args.teacher_responses else {}

    proposals: list[tuple[dict[str, Any], str, str, dict[str, Any]]] = []
    for record in collected:
        if record.get("disposition") == "feedback_positive":
            proposals.append((record, str(record["candidate"]), "model_rollout", {}))
        elif record.get("disposition") == "needs_teacher":
            entry = teacher.get(str(record["failure_id"]))
            parsed = entry.get("parsed") if entry else None
            if isinstance(parsed, dict):
                teacher_test_visibility(entry, args.teacher_test_visibility)
                proposals.append((record, str(parsed.get("repaired_code") or ""), "teacher_repair", parsed))
    if args.limit:
        proposals = proposals[: args.limit]
    print(f"Hidden build-stage verification of {len(proposals)} proposals")

    def verify(proposal):
        record, raw_code, origin, parsed = proposal
        code = extract_code(raw_code)
        task = record["task"]
        feedback = evaluate_candidate(
            code,
            str(task.get("feedback_tests") or task["tests"]),
            f"{record['task_key']}_feedback_replay",
            args.timeout,
        )
        acceptance = evaluate_candidate(
            code,
            str(task["acceptance_tests"]),
            f"{record['task_key']}_hidden",
            args.timeout,
        )
        full = evaluate_candidate(
            code,
            str(task["tests"]),
            f"{record['task_key']}_full",
            args.timeout,
        )
        claims = (
            (parsed.get("fact_claims") or parsed.get("assembly_facts"))
            if origin == "teacher_repair"
            else parse_facts_comment(code)
        )
        fact_ok, fact_reasons = candidate_fact_match(
            task,
            code,
            teacher_claim=claims,
            mode=args.facts_gate_mode,
            require_claims=(origin == "teacher_repair"),
        )
        return proposal, code, feedback, acceptance, full, fact_ok, fact_reasons

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done, value in enumerate(pool.map(verify, proposals), 1):
            results.append(value)
            if done % 25 == 0 or done == len(proposals):
                print(f"  hidden verifier {done}/{len(proposals)}")

    sft_rows: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    origins: dict[str, int] = {}
    rejection_counts = {"feedback_replay": 0, "hidden_acceptance": 0, "full_harness": 0, "facts_gate": 0}
    for proposal, code, feedback, acceptance, full, fact_ok, fact_reasons in results:
        record, _raw, origin, parsed = proposal
        reasons: list[str] = []
        if not feedback.passed:
            rejection_counts["feedback_replay"] += 1
            reasons.append("feedback replay failed")
        if not acceptance.passed:
            rejection_counts["hidden_acceptance"] += 1
            reasons.append("hidden acceptance failed")
        if not full.passed:
            rejection_counts["full_harness"] += 1
            reasons.append("complete harness failed")
        if not fact_ok:
            rejection_counts["facts_gate"] += 1
            reasons.extend(fact_reasons)
        if reasons:
            rejections.append(
                {
                    "task_key": record["task_key"],
                    "failure_id": record["failure_id"],
                    "origin": origin,
                    "reasons": reasons,
                    "feedback": {
                        "compiled": feedback.compiled,
                        "passed": feedback.passed,
                        "diagnostic": feedback.diagnostic,
                    },
                    "acceptance": {
                        "compiled": acceptance.compiled,
                        "passed": acceptance.passed,
                        "diagnostic": acceptance.diagnostic,
                    },
                    "full": {"compiled": full.compiled, "passed": full.passed, "diagnostic": full.diagnostic},
                }
            )
            continue
        dedupe = (str(record["task_key"]), canonical_code(code))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        origins[origin] = origins.get(origin, 0) + 1
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "origin": origin,
            "failure_id": record["failure_id"],
            "candidate_sha256": record["candidate_sha256"],
            "target_sha256": sha256_text(code),
            "teacher_model": teacher_model_name(teacher.get(str(record["failure_id"]))) if origin == "teacher_repair" else None,
            "teacher_test_visibility": (
                teacher_test_visibility(
                    teacher.get(str(record["failure_id"])),
                    args.teacher_test_visibility,
                )
                if origin == "teacher_repair"
                else None
            ),
            "teacher_failure_class": parsed.get("failure_class") if parsed else None,
            "teacher_confidence": parsed.get("confidence") if parsed else None,
            "repair_actions": parsed.get("repair_actions", []) if parsed else [],
            "verifier_replayed": True,
            "feedback_replayed": True,
            "feedback_tests_passed": True,
            "verifier_full_pass": True,
            "hidden_acceptance_replayed": True,
            "acceptance_tests_passed": True,
            "facts_gate_mode": args.facts_gate_mode,
            "facts_gate_applied": True,
            "facts_gate_passed": True,
            "data_role": role,
        }
        sft_rows.append(positive_training_row(record["task"], code, metadata))
        if canonical_code(record["candidate"]) != canonical_code(code):
            preferences.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_key": record["task_key"],
                    "assembly": record["task"].get("assembly", ""),
                    "signature": signature_text(record["task"]),
                    "chosen": code,
                    "rejected": record["candidate"],
                    "chosen_origin": origin,
                    "failure_id": record["failure_id"],
                    "failure_class": parsed.get("failure_class") if parsed else None,
                    "repair_actions": parsed.get("repair_actions", []) if parsed else [],
                }
            )

    sft_count = write_jsonl(args.out_sft, sft_rows)
    pref_count = write_jsonl(args.out_preferences, preferences)
    rejection_counts = {
        "hidden_acceptance": sum(
            any("hidden acceptance" in reason for reason in item["reasons"])
            for item in rejections
        ),
        "complete_harness": sum(
            any("complete harness" in reason for reason in item["reasons"])
            for item in rejections
        ),
        "facts_gate": sum(
            any(
                "fact" in reason.lower() or "arity" in reason.lower()
                or "return type" in reason.lower() or "constant" in reason.lower()
                for reason in item["reasons"]
            )
            for item in rejections
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "build",
        "data_role": role,
        "collected": file_record(args.collected),
        "teacher_responses": file_record(args.teacher_responses) if args.teacher_responses else None,
        "proposals": len(proposals),
        "rs_sft_rows": sft_count,
        "unique_tasks": len({task_key(row, index) for index, row in enumerate(sft_rows)}),
        "length_bins": sorted({
            str((row.get("hybrid_metadata") or {}).get("length_bin") or "unknown")
            for row in sft_rows
        }),
        "preference_pairs": pref_count,
        "origins": origins,
        "rejected": len(rejections),
        "rejections": rejection_counts,
        "rejection_examples": rejections[:50],
        "feedback_test_source_exposed_to_teacher": False,
        "teacher_test_visibility": args.teacher_test_visibility,
        "feedback_outcome_summary_exposed_to_teacher": (
            args.teacher_test_visibility in {"summary", "diagnostics"}
        ),
        "feedback_diagnostics_exposed_to_teacher": (
            args.teacher_test_visibility == "diagnostics"
        ),
        "hidden_acceptance_tests_exposed_to_teacher": False,
        "facts_gate_mode": args.facts_gate_mode,
        "facts_gate_applied": True,
        "outputs": {"rs_sft": file_record(args.out_sft), "preferences": file_record(args.out_preferences)},
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    unique_task_count = int(report["unique_tasks"])
    length_bin_count = len(set(report["length_bins"]) - {"unknown"})
    failures: list[str] = []
    if sft_count < args.min_verified_rows:
        failures.append(f"verified rows {sft_count} < {args.min_verified_rows}")
    if unique_task_count < args.min_verified_unique_tasks:
        failures.append(
            f"verified unique tasks {unique_task_count} < {args.min_verified_unique_tasks}"
        )
    if length_bin_count < args.min_verified_length_bins:
        failures.append(
            f"verified length bins {length_bin_count} < {args.min_verified_length_bins}"
        )
    if failures:
        raise SystemExit("verified RS-SFT kill switch failed: " + "; ".join(failures))
    return 0


def recertify_command(args: argparse.Namespace) -> int:
    """Re-certify legacy harvests without trusting their old gate booleans.

    The July text-finish run wrote ``facts_gate_passed=True`` even when the gate
    mode was ``off``.  This migration replays every executable verifier and a
    real facts gate, then writes a distinct output artifact with explicit
    applied-mode provenance.  Rows that fail any check are omitted.
    """
    if args.facts_gate_mode == "off":
        raise SystemExit("recertification requires a real facts gate")
    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("recertification input is empty")

    def verify(index_and_row: tuple[int, dict[str, Any]]):
        index, row = index_and_row
        assert_training_approved(row)
        metadata = row.get("hybrid_metadata") or {}
        origin = str(metadata.get("origin") or "")
        if origin not in {"teacher_repair", "model_rollout", "verified_rollout"}:
            return index, row, None, [f"non-verifiable origin {origin!r}"]
        feedback_tests = str(row.get("feedback_tests") or "")
        acceptance_tests = str(row.get("acceptance_tests") or "")
        full_tests = str(row.get("tests") or "")
        missing = [
            name
            for name, value in (
                ("feedback_tests", feedback_tests),
                ("acceptance_tests", acceptance_tests),
                ("tests", full_tests),
            )
            if not value.strip()
        ]
        if missing:
            return index, row, None, ["missing verifier suites: " + ", ".join(missing)]
        code = extract_code(source_text(row))
        identity = task_key(row, index)
        feedback = evaluate_candidate(
            code, feedback_tests, f"{identity}_feedback_recertify", args.timeout
        )
        acceptance = evaluate_candidate(
            code, acceptance_tests, f"{identity}_hidden_recertify", args.timeout
        )
        full = evaluate_candidate(
            code, full_tests, f"{identity}_full_recertify", args.timeout
        )
        claims = parse_facts_comment(code)
        fact_ok, fact_reasons = candidate_fact_match(
            row,
            code,
            teacher_claim=claims,
            mode=args.facts_gate_mode,
            require_claims=True,
        )
        reasons: list[str] = []
        if not feedback.passed:
            reasons.append("feedback replay failed")
        if not acceptance.passed:
            reasons.append("hidden acceptance replay failed")
        if not full.passed:
            reasons.append("complete harness replay failed")
        if not fact_ok:
            reasons.extend(fact_reasons)
        details = {
            "feedback": {"compiled": feedback.compiled, "passed": feedback.passed},
            "acceptance": {"compiled": acceptance.compiled, "passed": acceptance.passed},
            "full": {"compiled": full.compiled, "passed": full.passed},
            "facts_gate_passed": bool(fact_ok),
        }
        return index, row, details, reasons

    checked: list[tuple[int, dict[str, Any], dict[str, Any] | None, list[str]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done, result in enumerate(pool.map(verify, enumerate(rows)), 1):
            checked.append(result)
            if done % 25 == 0 or done == len(rows):
                print(f"  re-certified {done}/{len(rows)}")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, row, details, reasons in checked:
        identity = task_key(row, index)
        if reasons:
            rejected.append(
                {
                    "index": index,
                    "task_key": identity,
                    "reasons": reasons,
                    "verification": details,
                }
            )
            continue
        copied = copy.deepcopy(row)
        metadata = dict(copied.get("hybrid_metadata") or {})
        prior = {
            "facts_gate_mode": metadata.get("facts_gate_mode"),
            "facts_gate_applied": metadata.get("facts_gate_applied"),
            "facts_gate_passed": metadata.get("facts_gate_passed"),
        }
        metadata.update(
            {
                "verifier_replayed": True,
                "feedback_replayed": True,
                "feedback_tests_passed": True,
                "verifier_full_pass": True,
                "hidden_acceptance_replayed": True,
                "acceptance_tests_passed": True,
                "facts_gate_mode": args.facts_gate_mode,
                "facts_gate_applied": True,
                "facts_gate_passed": True,
                "verification_recertified": True,
                "verification_recertification_schema": 1,
                "prior_facts_gate_provenance": prior,
            }
        )
        copied["hybrid_metadata"] = metadata
        accepted.append(copied)

    output_count = write_jsonl(args.out, accepted)
    unique_tasks = len({task_key(row, index) for index, row in enumerate(accepted)})
    length_bins = sorted(
        {
            str((row.get("hybrid_metadata") or {}).get("length_bin") or "unknown")
            for row in accepted
        }
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "recertify_verified_rs_sft",
        "facts_gate_mode": args.facts_gate_mode,
        "input": file_record(args.input),
        "output": file_record(args.out),
        "input_rows": len(rows),
        "certified_rows": output_count,
        "rejected_rows": len(rejected),
        "unique_tasks": unique_tasks,
        "length_bins": length_bins,
        "all_executable_suites_replayed": True,
        "original_artifact_mutated": False,
        "rejection_examples": rejected[:100],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failures: list[str] = []
    if output_count < args.min_verified_rows:
        failures.append(f"verified rows {output_count} < {args.min_verified_rows}")
    if unique_tasks < args.min_verified_unique_tasks:
        failures.append(
            f"verified unique tasks {unique_tasks} < {args.min_verified_unique_tasks}"
        )
    known_bins = len(set(length_bins) - {"unknown"})
    if known_bins < args.min_verified_length_bins:
        failures.append(
            f"verified length bins {known_bins} < {args.min_verified_length_bins}"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit("RS-SFT re-certification kill switch failed: " + "; ".join(failures))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--dataset", required=True)
    collect.add_argument("--predictions", required=True)
    collect.add_argument(
        "--prediction_provenance",
        default="",
        help="Inference provenance sidecar; defaults to <predictions>.provenance.json",
    )
    collect.add_argument(
        "--expected_checkpoint",
        default="",
        help="Optional checkpoint whose SHA-256 must match rollout provenance",
    )
    collect.add_argument("--phase0_report", default="")
    collect.add_argument("--data_role", required=True, choices=sorted(ALLOWED_DATA_ROLES))
    collect.add_argument("--out", required=True)
    collect.add_argument("--timeout", type=int, default=15)
    collect.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 1)))
    collect.add_argument("--max_failures_per_task", type=int, default=2)
    collect.add_argument("--max_positives_per_task", type=int, default=3)
    collect.add_argument("--min_candidate_chars", type=int, default=8)
    collect.add_argument("--limit_candidates", type=int, default=0)
    collect.set_defaults(func=collect_command)

    teacher = sub.add_parser("teacher")
    teacher.add_argument("--collected", required=True)
    teacher.add_argument("--out", required=True)
    teacher.add_argument("--mode", choices=["sync", "batch"], default="sync")
    teacher.add_argument("--model", required=True)
    teacher.add_argument("--test_visibility", choices=["none", "summary", "diagnostics"], default="summary")
    teacher.add_argument("--max_assembly_chars", type=int, default=18000)
    teacher.add_argument("--max_candidate_chars", type=int, default=8000)
    teacher.add_argument("--max_output_tokens", type=int, default=3000)
    teacher.add_argument("--concurrency", type=int, default=4)
    teacher.add_argument("--retries", type=int, default=4)
    teacher.add_argument("--retry_base_seconds", type=float, default=1.0)
    teacher.add_argument("--retry_max_seconds", type=float, default=20.0)
    teacher.add_argument("--api_key_env", default="OPENAI_API_KEY")
    teacher.add_argument("--base_url", default="")
    teacher.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    teacher.add_argument("--limit", type=int, default=0)
    teacher.set_defaults(func=teacher_command)

    build = sub.add_parser("build")
    build.add_argument("--collected", required=True)
    build.add_argument("--phase0_report", default="")
    build.add_argument("--teacher_responses", default="")
    build.add_argument("--data_role", required=True, choices=sorted(ALLOWED_DATA_ROLES))
    build.add_argument("--out_sft", required=True)
    build.add_argument("--out_preferences", required=True)
    build.add_argument("--report", required=True)
    build.add_argument("--timeout", type=int, default=15)
    build.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 1)))
    build.add_argument("--facts_gate_mode", choices=["off", "signature", "conservative", "strict"], default="conservative")
    build.add_argument(
        "--teacher_test_visibility",
        choices=["unknown", "none", "summary", "diagnostics"],
        default="unknown",
        help=(
            "Feedback visibility used when teacher requests were generated. "
            "Required for legacy/batch responses that do not record it themselves."
        ),
    )
    build.add_argument("--min_verified_rows", type=int, default=1)
    build.add_argument("--min_verified_unique_tasks", type=int, default=1)
    build.add_argument("--min_verified_length_bins", type=int, default=0)
    build.add_argument("--limit", type=int, default=0)
    build.set_defaults(func=build_command)

    recertify = sub.add_parser(
        "recertify",
        help="Replay all gates on a legacy verified-RS-SFT artifact.",
    )
    recertify.add_argument("--input", required=True)
    recertify.add_argument("--out", required=True)
    recertify.add_argument("--report", required=True)
    recertify.add_argument("--timeout", type=int, default=15)
    recertify.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4) - 1)),
    )
    recertify.add_argument(
        "--facts_gate_mode",
        choices=["signature", "conservative", "strict"],
        default="signature",
    )
    recertify.add_argument("--min_verified_rows", type=int, default=1)
    recertify.add_argument("--min_verified_unique_tasks", type=int, default=1)
    recertify.add_argument("--min_verified_length_bins", type=int, default=0)
    recertify.set_defaults(func=recertify_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "workers", 1) <= 0 or getattr(args, "concurrency", 1) <= 0:
        parser.error("worker/concurrency values must be positive")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
