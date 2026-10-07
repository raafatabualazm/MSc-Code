#!/usr/bin/env python3
"""Execute direct-compact candidates with attested Dart tests and persist detail."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluation import graph_compile_at_k_antigravity as dart_evaluator
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    require_unique_slots,
)
from scripts.training.teacher_repair_dataset_antigravity import extract_code

evaluate_dart_jit_tests_detail = dart_evaluator.evaluate_dart_jit_tests_detail
SCORE_JOURNAL_SCHEMA = "direct-compact-attested-passk-journal-v1"


def extract_scored_code(text: str) -> str:
    """Remove CoT/prose wrappers before hashing and executing the Dart body."""

    value = str(text or "").strip()
    if value.startswith("<think>"):
        close = value.find("</think>")
        if close < 0:
            # A capped/incomplete reasoning trace has no attested final answer.
            return ""
        value = value[close + len("</think>") :].lstrip()
    # ``extract_code`` handles Markdown fences. The evaluator's deterministic
    # extractor additionally skips a leading <think> block or decoded reasoning
    # prose by selecting the first plausible Dart declaration. Applying it here
    # makes the persisted code hash describe the source that is actually run.
    return dart_evaluator._extract_code(extract_code(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--evaluation_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--journal",
        default="",
        help="Append-only exact-resume journal (defaults beside --output).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--stability_runs", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def validate_existing_score(
    *,
    predictions: str | Path,
    evaluation_file: str | Path,
    output: str | Path,
    k: int,
    timeout: int,
    stability_runs: int,
    journal: str | Path | None = None,
) -> dict[str, Any]:
    prediction_path = Path(predictions).expanduser().resolve()
    provenance_path = Path(str(prediction_path) + ".provenance.json")
    evaluation_path = Path(evaluation_file).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(provenance, dict):
        raise ValueError("existing score/provenance is not a JSON object")
    validate_prediction_provenance(
        provenance,
        prediction_sha256=sha256_file(prediction_path),
        k=k,
    )
    prediction_rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    evaluation_rows = read_jsonl(evaluation_path)
    task_ids = {
        str(row.get("task_id") or "") for row in evaluation_rows
    }
    candidate_rows = report.get("candidate_results")
    task_rows = report.get("task_results")
    evaluator_path = Path(dart_evaluator.__file__).resolve()
    if (
        report.get("schema") != "direct-compact-attested-passk-v1"
        or (report.get("predictions") or {}).get("sha256")
        != sha256_file(prediction_path)
        or (report.get("predictions") or {}).get("provenance_sha256")
        != sha256_file(provenance_path)
        or (report.get("evaluation") or {}).get("sha256")
        != sha256_file(evaluation_path)
        or (report.get("evaluator") or {}).get("sha256")
        != sha256_file(evaluator_path)
        or (report.get("evaluator") or {}).get("completion_attestation")
        != dart_evaluator.COMPLETION_ATTESTATION_ID
        or int(report.get("k", -1)) != int(k)
        or int(report.get("timeout", -1)) != int(timeout)
        or int(report.get("stability_runs", -1)) != int(stability_runs)
        or int(report.get("tasks", -1)) != len(task_ids)
        or provenance.get("output_sha256") != sha256_file(prediction_path)
        or int(provenance.get("num_samples", -1)) != int(k)
        or not isinstance(prediction_rows, list)
        or len(prediction_rows) != len(task_ids)
        or not isinstance(candidate_rows, list)
        or len(candidate_rows) != len(task_ids) * int(k)
        or not isinstance(task_rows, list)
        or len(task_rows) != len(task_ids)
        or {str(row.get("task_id") or "") for row in task_rows}
        != task_ids
        or report.get("evaluation_journal")
        != journal_record(
            journal or (str(output_path) + ".evaluation.journal.jsonl")
        )
    ):
        raise ValueError("existing score report differs from exact inputs/settings")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        task_id = str(row.get("task_id") or "")
        by_task.setdefault(task_id, []).append(row)
    recomputed: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        rows = sorted(
            by_task.get(task_id, []), key=lambda row: row.get("sample_index", -1)
        )
        if [row.get("sample_index") for row in rows] != list(range(k)):
            raise ValueError(f"{task_id}: existing score sample coverage differs")
        recomputed.append(
            {
                "task_id": task_id,
                "pass_at_1": bool(rows[0].get("passed")),
                "pass_at_k": any(bool(row.get("passed")) for row in rows),
                "compile_at_k": any(bool(row.get("compiled")) for row in rows),
                "passing_samples": sum(bool(row.get("passed")) for row in rows),
                "compiling_samples": sum(
                    bool(row.get("compiled")) for row in rows
                ),
            }
        )
    if recomputed != task_rows:
        raise ValueError("existing score task aggregates are inconsistent")
    counts = {
        "pass_at_1": sum(row["pass_at_1"] for row in recomputed),
        "pass_at_k": sum(row["pass_at_k"] for row in recomputed),
        "compile_at_k": sum(row["compile_at_k"] for row in recomputed),
    }
    if any(
        (report.get(name) or {}).get("count") != value
        for name, value in counts.items()
    ):
        raise ValueError("existing score global aggregates are inconsistent")
    return report


def _score_batch_seal(
    *,
    batch_index: int,
    slot_ids: list[str],
    jobs_canonical_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "direct-compact-score-sealed-batch-v1",
        "batch_index": int(batch_index),
        "slot_ids": list(slot_ids),
        "jobs_canonical_sha256": str(jobs_canonical_sha256),
    }


def make_score_orphan_retry_event(
    orphan: dict[str, Any],
) -> dict[str, Any]:
    """Authorize one exact rerun of a durably started local score batch."""

    started = orphan["started"]
    retries = list(orphan["retries"])
    batch_seal = dict(orphan["batch_seal"])
    previous_attempt_sha256 = (
        retries[-1]["journal_event_sha256"]
        if retries
        else started["journal_event_sha256"]
    )
    return {
        "event": "score_batch_orphan_retry",
        "schema": SCORE_JOURNAL_SCHEMA,
        "batch_index": int(started["batch_index"]),
        "retry_index": len(retries) + 1,
        "started_event_sha256": started["journal_event_sha256"],
        "previous_attempt_event_sha256": previous_attempt_sha256,
        "sealed_batch": batch_seal,
        "sealed_batch_sha256": canonical_sha256(batch_seal),
        "completed_terminal_batches_preserved": int(
            orphan["completed_terminal_batches"]
        ),
        "recovery_reason": "process_interrupted_after_durable_batch_start",
        "rerun_identical_sealed_batch": True,
        "change_evaluation_slots": False,
    }


def _score_journal_state(
    events: list[dict[str, Any]],
    *,
    contract_payload: dict[str, Any],
    jobs: list[dict[str, Any]],
    batch_size: int,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    bool,
    dict[str, Any] | None,
]:
    """Validate completed batches and expose only a trailing sealed orphan."""

    if not events:
        return {}, False, None
    if (
        events[0].get("event") != "score_header"
        or events[0].get("schema") != SCORE_JOURNAL_SCHEMA
        or events[0].get("contract") != contract_payload
        or events[0].get("contract_sha256")
        != canonical_sha256(contract_payload)
    ):
        raise ValueError("score journal header differs from exact run contract")
    terminals: dict[int, list[dict[str, Any]]] = {}
    retry_event_count = 0
    retry_slot_count = 0
    cursor = 1
    batch_count = (len(jobs) + batch_size - 1) // batch_size
    while cursor < len(events):
        event = events[cursor]
        if event.get("event") == "score_complete":
            if (
                event.get("schema") != SCORE_JOURNAL_SCHEMA
                or cursor != len(events) - 1
                or len(terminals) != batch_count
            ):
                raise ValueError("score completion event is premature")
            ordered = [
                result
                for index in range(batch_count)
                for result in terminals[index]
            ]
            if (
                event.get("candidate_results_canonical_sha256")
                != canonical_sha256(ordered)
                or int(event.get("slots", -1)) != len(jobs)
                or int(event.get("rerun_slots", -1)) != 0
                or int(event.get("orphan_retry_events", -1))
                != retry_event_count
                or int(event.get("orphan_rerun_slots", -1))
                != retry_slot_count
            ):
                raise ValueError("score completion digest/count differs")
            return terminals, True, None
        batch_index = len(terminals)
        expected_jobs = jobs[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        expected_slots = [
            f"{job['task_id']}:{job['sample_index']}" for job in expected_jobs
        ]
        job_projection = [
            {
                key: value
                for key, value in job.items()
                if key != "code"
            }
            for job in expected_jobs
        ]
        expected_jobs_sha256 = canonical_sha256(job_projection)
        expected_seal = _score_batch_seal(
            batch_index=batch_index,
            slot_ids=expected_slots,
            jobs_canonical_sha256=expected_jobs_sha256,
        )
        if (
            event.get("event") != "score_batch_started"
            or event.get("schema") != SCORE_JOURNAL_SCHEMA
            or int(event.get("batch_index", -1)) != batch_index
            or event.get("slot_ids") != expected_slots
            or event.get("jobs_canonical_sha256")
            != expected_jobs_sha256
        ):
            raise ValueError("score journal batch-start schedule differs")
        started = event
        cursor += 1
        retries: list[dict[str, Any]] = []
        previous_attempt_sha256 = str(
            started.get("journal_event_sha256") or ""
        )
        while (
            cursor < len(events)
            and events[cursor].get("event") == "score_batch_orphan_retry"
        ):
            retry = events[cursor]
            if (
                retry.get("schema") != SCORE_JOURNAL_SCHEMA
                or int(retry.get("batch_index", -1)) != batch_index
                or int(retry.get("retry_index", -1)) != len(retries) + 1
                or retry.get("started_event_sha256")
                != started.get("journal_event_sha256")
                or retry.get("previous_attempt_event_sha256")
                != previous_attempt_sha256
                or retry.get("sealed_batch") != expected_seal
                or retry.get("sealed_batch_sha256")
                != canonical_sha256(expected_seal)
                or int(
                    retry.get(
                        "completed_terminal_batches_preserved", -1
                    )
                )
                != len(terminals)
                or retry.get("recovery_reason")
                != "process_interrupted_after_durable_batch_start"
                or retry.get("rerun_identical_sealed_batch") is not True
                or retry.get("change_evaluation_slots") is not False
            ):
                raise ValueError("score orphan-retry receipt is inconsistent")
            previous_attempt_sha256 = str(
                retry.get("journal_event_sha256") or ""
            )
            retries.append(retry)
            retry_event_count += 1
            retry_slot_count += len(expected_slots)
            cursor += 1
        if cursor >= len(events):
            return (
                terminals,
                False,
                {
                    "batch_index": batch_index,
                    "started": started,
                    "retries": retries,
                    "batch_seal": expected_seal,
                    "completed_terminal_batches": len(terminals),
                },
            )
        terminal = events[cursor]
        results = terminal.get("candidate_results")
        expected_latest_retry = (
            retries[-1].get("journal_event_sha256") if retries else None
        )
        if (
            terminal.get("event") != "score_batch_terminal"
            or terminal.get("schema") != SCORE_JOURNAL_SCHEMA
            or int(terminal.get("batch_index", -1)) != batch_index
            or terminal.get("started_event_sha256")
            != started.get("journal_event_sha256")
            or int(terminal.get("retry_count", 0)) != len(retries)
            or terminal.get("latest_retry_event_sha256")
            != expected_latest_retry
            or not isinstance(results, list)
            or [
                f"{row.get('task_id')}:{row.get('sample_index')}"
                for row in results
            ]
            != expected_slots
            or terminal.get("candidate_results_canonical_sha256")
            != canonical_sha256(results)
        ):
            raise ValueError("score journal batch terminal is inconsistent")
        terminals[batch_index] = results
        cursor += 1
    return terminals, False, None


def validate_prediction_provenance(
    prediction_provenance: Mapping[str, Any],
    *,
    prediction_sha256: str,
    k: int,
) -> None:
    """Validate the exact inference family admitted by this scorer."""

    provenance_schema = prediction_provenance.get("schema")
    allowed_provenance_schemas = {
        "direct-compact-inference-v1",
        "t5gemma2-f2-measurement-ablation-provenance-v1",
    }
    if provenance_schema not in allowed_provenance_schemas:
        raise ValueError("prediction provenance has an unknown schema")
    if provenance_schema == "t5gemma2-f2-measurement-ablation-provenance-v1":
        input_view = prediction_provenance.get("input_view")
        heldout_view = (prediction_provenance.get("heldout") or {}).get(
            "input_view"
        )
        if (
            input_view
            not in {
                "semantic_body_swap",
                "constants_stripped",
                "typed_opaque_contract",
            }
            or not isinstance(heldout_view, dict)
            or heldout_view.get("schema")
            != "t5gemma2-f2-measurement-input-view-v1"
            or heldout_view.get("view") != input_view
            or prediction_provenance.get("no_frontier_api") is not True
            or prediction_provenance.get("tests_exposed_to_model") is not False
            or prediction_provenance.get("full_gold_targets_exposed_to_model")
            is not False
        ):
            raise ValueError("measurement-ablation provenance contract failed")
    if prediction_provenance.get("output_sha256") != prediction_sha256:
        raise ValueError("predictions do not match their inference provenance")
    if int(prediction_provenance.get("num_samples", -1)) != k:
        raise ValueError("inference provenance does not match requested k")


def main() -> None:
    args = parse_args()
    if args.k <= 0 or args.workers <= 0:
        raise ValueError("k and workers must be positive")
    if args.timeout <= 0 or args.stability_runs <= 0:
        raise ValueError("timeout and stability_runs must be positive")
    prediction_path = Path(args.predictions).expanduser().resolve()
    prediction_provenance_path = Path(
        str(prediction_path) + ".provenance.json"
    )
    evaluation_path = Path(args.evaluation_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    journal_path = Path(
        args.journal or (str(output_path) + ".evaluation.journal.jsonl")
    ).expanduser().resolve()
    args.journal = str(journal_path)
    for required in (
        prediction_path,
        prediction_provenance_path,
        evaluation_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_path.exists():
        validate_existing_score(
            predictions=prediction_path,
            evaluation_file=evaluation_path,
            output=output_path,
            k=args.k,
            timeout=args.timeout,
            stability_runs=args.stability_runs,
            journal=journal_path,
        )
        print(f"DIRECT_COMPACT_SCORE_REUSED output={output_path}", flush=True)
        return

    prediction_sha256 = sha256_file(prediction_path)
    prediction_provenance = json.loads(
        prediction_provenance_path.read_text(encoding="utf-8")
    )
    if not isinstance(prediction_provenance, dict):
        raise ValueError("prediction provenance must be a JSON object")
    validate_prediction_provenance(
        prediction_provenance,
        prediction_sha256=prediction_sha256,
        k=args.k,
    )

    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predictions must be a non-empty JSON array")
    evaluation_rows = read_jsonl(evaluation_path)
    if int(prediction_provenance.get("num_rows", -1)) != len(predictions):
        raise ValueError("prediction row count does not match inference provenance")
    tests_by_id: dict[str, str] = {}
    for row in evaluation_rows:
        task_id = str(row.get("task_id") or "")
        tests = str(
            row.get("acceptance_tests")
            or row.get("tests")
            or row.get("feedback_tests")
            or ""
        )
        if not task_id or task_id in tests_by_id or not tests:
            raise ValueError(
                f"evaluation file has missing/duplicate task or tests: {task_id!r}"
            )
        tests_by_id[task_id] = tests

    prediction_ids: set[str] = set()
    jobs: list[dict[str, Any]] = []
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("prediction row is not an object")
        task_id = str(row.get("id") or "")
        samples = row.get("predictions")
        if not task_id or task_id in prediction_ids:
            raise ValueError(f"missing/duplicate prediction id {task_id!r}")
        if task_id not in tests_by_id:
            raise ValueError(f"prediction task has no tests: {task_id}")
        if not isinstance(samples, list) or len(samples) != args.k:
            raise ValueError(
                f"{task_id}: expected exactly {args.k} candidates, "
                f"observed {len(samples) if isinstance(samples, list) else 'invalid'}"
            )
        prediction_ids.add(task_id)
        for sample_index, raw in enumerate(samples):
            raw_text = str(raw or "")
            code = extract_scored_code(raw_text)
            jobs.append(
                {
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "raw_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                    "code": code,
                    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                }
            )
    if prediction_ids != set(tests_by_id):
        missing = sorted(set(tests_by_id) - prediction_ids)
        extra = sorted(prediction_ids - set(tests_by_id))
        raise ValueError(
            f"prediction/evaluation task set mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    slot_ids = [
        f"{job['task_id']}:{job['sample_index']}" for job in jobs
    ]
    require_unique_slots(slot_ids)
    evaluator_path = Path(dart_evaluator.__file__).resolve()
    journal_contract = {
        "schema": SCORE_JOURNAL_SCHEMA,
        "predictions_sha256": prediction_sha256,
        "prediction_provenance_sha256": sha256_file(
            prediction_provenance_path
        ),
        "evaluation_sha256": sha256_file(evaluation_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "completion_attestation": (
            "per-run-256-bit-marker-exactly-once-v1"
        ),
        "k": args.k,
        "workers": args.workers,
        "batch_size": args.workers,
        "timeout": args.timeout,
        "stability_runs": args.stability_runs,
        "ordered_slot_ids_sha256": canonical_sha256(slot_ids),
        "slots": len(slot_ids),
        "started_without_terminal_policy": (
            "retry_identical_sealed_batch_with_hash_chained_receipt"
        ),
    }
    events = load_journal(journal_path)
    if not events:
        append_event(
            journal_path,
            {
                "event": "score_header",
                "schema": SCORE_JOURNAL_SCHEMA,
                "contract": journal_contract,
                "contract_sha256": canonical_sha256(journal_contract),
            },
        )
        events = load_journal(journal_path)

    def evaluate(job: dict[str, Any]) -> dict[str, Any]:
        if not job["code"].strip():
            return {
                **{key: value for key, value in job.items() if key != "code"},
                "compiled": False,
                "passed": False,
                "diagnostic": "empty_extracted_code",
            }
        compiled, passed, diagnostic, _source = evaluate_dart_jit_tests_detail(
            job["code"],
            tests_by_id[job["task_id"]],
            f"{job['task_id']}_s{job['sample_index']}",
            timeout=args.timeout,
            stability_runs=args.stability_runs,
        )
        return {
            **{key: value for key, value in job.items() if key != "code"},
            "compiled": bool(compiled),
            "passed": bool(passed),
            "diagnostic": str(diagnostic)[:2000],
        }

    result_batches, complete, orphan = _score_journal_state(
        events,
        contract_payload=journal_contract,
        jobs=jobs,
        batch_size=args.workers,
    )
    batch_count = (len(jobs) + args.workers - 1) // args.workers
    for batch_index in range(len(result_batches), batch_count):
        batch = jobs[
            batch_index * args.workers : (batch_index + 1) * args.workers
        ]
        slot_batch = [
            f"{job['task_id']}:{job['sample_index']}" for job in batch
        ]
        projection = [
            {key: value for key, value in job.items() if key != "code"}
            for job in batch
        ]
        latest_retry: dict[str, Any] | None = None
        if orphan is not None:
            if int(orphan.get("batch_index", -1)) != batch_index:
                raise RuntimeError(
                    "score orphan is not the next incomplete batch"
                )
            started = dict(orphan["started"])
            latest_retry = append_event(
                journal_path,
                make_score_orphan_retry_event(orphan),
            )
            retry_count = len(orphan["retries"]) + 1
            orphan = None
        else:
            started = append_event(
                journal_path,
                {
                    "event": "score_batch_started",
                    "schema": SCORE_JOURNAL_SCHEMA,
                    "batch_index": batch_index,
                    "slot_ids": slot_batch,
                    "jobs_canonical_sha256": canonical_sha256(projection),
                },
            )
            retry_count = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(batch))
        ) as pool:
            futures = [pool.submit(evaluate, job) for job in batch]
            results = [future.result() for future in futures]
        append_event(
            journal_path,
            {
                "event": "score_batch_terminal",
                "schema": SCORE_JOURNAL_SCHEMA,
                "batch_index": batch_index,
                "started_event_sha256": started["journal_event_sha256"],
                "retry_count": retry_count,
                "latest_retry_event_sha256": (
                    None
                    if latest_retry is None
                    else latest_retry["journal_event_sha256"]
                ),
                "candidate_results": results,
                "candidate_results_canonical_sha256": canonical_sha256(
                    results
                ),
            },
        )
        result_batches[batch_index] = results
    candidate_results = [
        result
        for batch_index in range(batch_count)
        for result in result_batches[batch_index]
    ]
    if not complete:
        events_before_complete = load_journal(journal_path)
        orphan_retry_events = [
            event
            for event in events_before_complete
            if event.get("event") == "score_batch_orphan_retry"
        ]
        append_event(
            journal_path,
            {
                "event": "score_complete",
                "schema": SCORE_JOURNAL_SCHEMA,
                "slots": len(jobs),
                "candidate_results_canonical_sha256": canonical_sha256(
                    candidate_results
                ),
                "rerun_slots": 0,
                "orphan_retry_events": len(orphan_retry_events),
                "orphan_rerun_slots": sum(
                    len((event.get("sealed_batch") or {}).get("slot_ids") or [])
                    for event in orphan_retry_events
                ),
            },
        )
    final_events = load_journal(journal_path)
    result_batches, complete, orphan = _score_journal_state(
        final_events,
        contract_payload=journal_contract,
        jobs=jobs,
        batch_size=args.workers,
    )
    if not complete or orphan is not None:
        raise RuntimeError("score journal did not reach completion")
    completion_event = final_events[-1]
    candidate_results = [
        result
        for batch_index in range(batch_count)
        for result in result_batches[batch_index]
    ]

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_results:
        by_task.setdefault(row["task_id"], []).append(row)
    task_results = []
    for task_id in sorted(by_task):
        candidates = by_task[task_id]
        if [row["sample_index"] for row in candidates] != list(range(args.k)):
            raise AssertionError(f"{task_id}: candidate index coverage changed")
        task_results.append(
            {
                "task_id": task_id,
                "pass_at_1": bool(candidates[0]["passed"]),
                "pass_at_k": any(row["passed"] for row in candidates),
                "compile_at_k": any(row["compiled"] for row in candidates),
                "passing_samples": sum(row["passed"] for row in candidates),
                "compiling_samples": sum(row["compiled"] for row in candidates),
            }
        )

    tasks = len(task_results)
    pass1 = sum(row["pass_at_1"] for row in task_results)
    passk = sum(row["pass_at_k"] for row in task_results)
    compilek = sum(row["compile_at_k"] for row in task_results)
    report = {
        "schema": "direct-compact-attested-passk-v1",
        "predictions": {
            "path": str(prediction_path),
            "sha256": prediction_sha256,
            "provenance_sha256": sha256_file(prediction_provenance_path),
        },
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": sha256_file(evaluation_path),
        },
        "k": args.k,
        "timeout": args.timeout,
        "stability_runs": args.stability_runs,
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": sha256_file(evaluator_path),
            "completion_attestation": "per-run-256-bit-marker-exactly-once-v1",
        },
        "evaluation_journal": journal_record(journal_path),
        "started_without_terminal_policy": (
            "retry_identical_sealed_batch_with_hash_chained_receipt"
        ),
        "rerun_slots": 0,
        "orphan_retry_events": int(
            completion_event["orphan_retry_events"]
        ),
        "orphan_rerun_slots": int(
            completion_event["orphan_rerun_slots"]
        ),
        "tasks": tasks,
        "pass_at_1": {"count": pass1, "rate": pass1 / tasks},
        "pass_at_k": {"count": passk, "rate": passk / tasks},
        "compile_at_k": {"count": compilek, "rate": compilek / tasks},
        "task_results": task_results,
        "candidate_results": candidate_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require_exact_or_write(output_path, report)
    validate_existing_score(
        predictions=prediction_path,
        evaluation_file=evaluation_path,
        output=output_path,
        k=args.k,
        timeout=args.timeout,
        stability_runs=args.stability_runs,
        journal=journal_path,
    )
    print(
        f"PASSK_RESULT tasks={tasks} pass@1={pass1}/{tasks}={pass1/tasks:.4f} "
        f"pass@{args.k}={passk}/{tasks}={passk/tasks:.4f} "
        f"compile@{args.k}={compilek}/{tasks}={compilek/tasks:.4f}"
    )


if __name__ == "__main__":
    main()
