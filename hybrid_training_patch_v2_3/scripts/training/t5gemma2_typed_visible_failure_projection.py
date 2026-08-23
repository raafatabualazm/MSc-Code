#!/usr/bin/env python3
"""Project every typed local K=4 candidate onto the sealed visible TRAIN split.

This is the sole eligibility/diagnostic source for typed API rescue.  It never
uses the complete-suite pass bit or complete-suite diagnostics.  Every local
candidate is re-executed on the separately sealed visible split, with two
stability runs, and a resumable hash-chained journal records only those public
outcomes.  API cohorts can then be scheduled without reopening either the GPU
checkpoint or a private outcome.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from scripts.training import t5gemma2_api_rs_sft_rescue as base
from scripts.training import t5gemma2_typed_api_rescue_cascade as cascade


RUN_SCHEMA = "t5gemma2-typed-visible-failure-projection-run-v1"
JOURNAL_SCHEMA = "t5gemma2-typed-visible-failure-projection-journal-v1"
REPORT_SCHEMA = "t5gemma2-typed-visible-failure-projection-report-v1"
STABILITY_RUNS = 2
EXPECTED_TASKS = 2550
EXPECTED_CANDIDATES = 4


def _evaluate(code: str, tests: str, slot: str, *, timeout: int) -> base.Evaluation:
    compiled, passed, diagnostic, _details = evaluate_dart_jit_tests_detail(
        code,
        tests,
        slot,
        timeout=timeout,
        stability_runs=STABILITY_RUNS,
    )
    return base.Evaluation(bool(compiled), bool(compiled and passed), str(diagnostic or ""))


def _candidate_rows(terminal: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = terminal.get("base_candidates")
    if not isinstance(values, list) or len(values) != EXPECTED_CANDIDATES:
        raise ValueError("typed local terminal does not contain exactly K=4")
    output: list[Mapping[str, Any]] = []
    for candidate in values:
        if not isinstance(candidate, Mapping):
            raise ValueError("typed local candidate is malformed")
        # A complete-suite diagnostic is forbidden even if an old local file
        # happens to be supplied.  Its binary outcome is ignored below.
        if {
            "diagnostic",
            "safe_compiler_feedback",
            "safe_compiler_feedback_sha256",
        } & set(candidate):
            raise ValueError("typed local candidate contains private-suite diagnostic")
        output.append(candidate)
    return output


def project_task(
    *,
    task: Any,
    terminal: Mapping[str, Any],
    task_position: int,
    timeout: int,
    workers: int,
    visible_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = _candidate_rows(terminal)

    def one(index: int) -> tuple[int, base.Evaluation]:
        code = str(candidates[index].get("code") or "")
        return index, _evaluate(
            code,
            task.visible_tests,
            f"typed-visible-{task.task_id}-{index}",
            timeout=timeout,
        )

    if workers == 1:
        outcomes = [one(index) for index in range(len(candidates))]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as pool:
            outcomes = list(pool.map(one, range(len(candidates))))
    by_index = {index: outcome for index, outcome in outcomes}
    rows: list[dict[str, Any]] = []
    passing: set[str] = set()
    for index, candidate in enumerate(candidates):
        code = str(candidate.get("code") or "")
        code_sha = str(candidate.get("code_sha256") or "")
        if base.sha256_text(code) != code_sha:
            raise ValueError("typed local candidate code digest differs")
        outcome = by_index[index]
        safe = (
            base.sanitize_compiler_diagnostic(outcome.diagnostic)
            if not outcome.compiled
            else base.COMPILED_NO_DIAGNOSTIC
        )
        if not safe:
            safe = base.MISSING_SAFE_DIAGNOSTIC
        passed = bool(outcome.compiled and outcome.passed)
        if passed:
            passing.add(code_sha)
        rows.append(
            {
                "sample_index": index,
                "origin": str(candidate.get("origin") or "local_student_direct"),
                "code": code,
                "code_sha256": code_sha,
                "visible": {
                    "compiled": bool(outcome.compiled),
                    "passed": passed,
                },
                "safe_visible_diagnostic": safe,
                "safe_visible_diagnostic_sha256": base.sha256_text(safe),
                "diagnostic_source": "sealed_visible_TRAIN_split",
                "private_complete_diagnostic_consumed": False,
            }
        )
    metadata = dict(visible_metadata or {})
    strategy = str(metadata.get("strategy") or "unknown")
    singleton_compile_only = (
        strategy == "stdout_singleton_compile_and_call_visible"
    )
    semantic_visible_cases = int(metadata.get("semantic_visible_cases", -1))
    if singleton_compile_only:
        if semantic_visible_cases != 0:
            raise ValueError("singleton stdout stratum must expose zero semantic cases")
        eligibility_stratum = "singleton_stdout_compile_call_only"
    else:
        if semantic_visible_cases <= 0:
            raise ValueError("semantic visible stratum must expose at least one case")
        eligibility_stratum = "semantic_visible_all_zero"
    return {
        "event": "task_terminal",
        "schema": JOURNAL_SCHEMA,
        "task_position": task_position,
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "split_binding_sha256": task.split_binding_sha256,
        "base_candidates": rows,
        "repair_groups": [],
        "visible_unique_passes": len(passing),
        "visible_split_strategy": strategy,
        "semantic_visible_cases": semantic_visible_cases,
        "api_eligibility_stratum": eligibility_stratum,
        "api_eligible": singleton_compile_only or not passing,
        "singleton_private_answer_used_for_eligibility": False,
        "all_k_reexecuted_on_visible_split": True,
        "private_complete_outcome_consumed_for_eligibility": False,
        "private_complete_diagnostic_consumed": False,
    }


def validate_journal(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    context: cascade.TypedSourceContext,
) -> tuple[list[dict[str, Any]], bool]:
    if not events:
        return [], False
    header = events[0]
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("typed visible projection journal header differs")
    terminals: list[dict[str, Any]] = []
    complete = False
    for event in events[1:]:
        if event.get("event") == "complete":
            if complete or len(terminals) != len(context.scheduled_tasks):
                raise ValueError("typed visible projection completion is early/duplicate")
            if (
                event.get("schema") != JOURNAL_SCHEMA
                or event.get("tasks") != len(terminals)
                or event.get("task_ids_sha256")
                != canonical_sha256([row["task_id"] for row in terminals])
            ):
                raise ValueError("typed visible projection completion differs")
            complete = True
            continue
        if complete or event.get("event") != "task_terminal":
            raise ValueError("typed visible projection event order differs")
        position = len(terminals)
        task = context.scheduled_tasks[position]
        if (
            event.get("schema") != JOURNAL_SCHEMA
            or event.get("task_position") != position
            or event.get("task_id") != task.task_id
            or event.get("source_sha256") != task.source_sha256
            or event.get("split_binding_sha256") != task.split_binding_sha256
            or event.get("repair_groups") != []
            or event.get("all_k_reexecuted_on_visible_split") is not True
            or event.get("private_complete_outcome_consumed_for_eligibility") is not False
            or event.get("private_complete_diagnostic_consumed") is not False
            or event.get("singleton_private_answer_used_for_eligibility") is not False
        ):
            raise ValueError("typed visible projection terminal binding differs")
        candidates = event.get("base_candidates")
        if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATES:
            raise ValueError("typed visible projection K differs")
        passing: set[str] = set()
        for index, row in enumerate(candidates):
            binary = row.get("visible") if isinstance(row, Mapping) else None
            code = str(row.get("code") or "") if isinstance(row, Mapping) else ""
            diagnostic = str(row.get("safe_visible_diagnostic") or "") if isinstance(row, Mapping) else ""
            if (
                not isinstance(row, Mapping)
                or row.get("sample_index") != index
                or base.sha256_text(code) != row.get("code_sha256")
                or not isinstance(binary, Mapping)
                or type(binary.get("compiled")) is not bool
                or type(binary.get("passed")) is not bool
                or (binary.get("passed") and not binary.get("compiled"))
                or not diagnostic
                or base.sha256_text(diagnostic)
                != row.get("safe_visible_diagnostic_sha256")
                or row.get("diagnostic_source") != "sealed_visible_TRAIN_split"
                or row.get("private_complete_diagnostic_consumed") is not False
            ):
                raise ValueError("typed visible projection candidate differs")
            if binary.get("passed"):
                passing.add(str(row["code_sha256"]))
        if event.get("visible_unique_passes") != len(passing):
            raise ValueError("typed visible projection pass accounting differs")
        singleton = (
            event.get("api_eligibility_stratum")
            == "singleton_stdout_compile_call_only"
        )
        if (
            type(event.get("semantic_visible_cases")) is not int
            or (singleton and event.get("semantic_visible_cases") != 0)
            or (
                not singleton
                and event.get("api_eligibility_stratum")
                != "semantic_visible_all_zero"
            )
            or (not singleton and event.get("semantic_visible_cases", 0) <= 0)
            or event.get("api_eligible") is not (singleton or not passing)
        ):
            raise ValueError("typed visible projection eligibility stratum differs")
        terminals.append(dict(event))
    return terminals, complete


def run(
    args: argparse.Namespace,
    *,
    context_loader: Any = cascade.load_typed_source_context,
) -> dict[str, Any]:
    context = context_loader(args)
    if len(context.scheduled_tasks) != EXPECTED_TASKS:
        raise ValueError("typed visible projection requires 2,550 local residual tasks")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "visible_projection.journal.jsonl"
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_local_harvest_journal": context.source_journal_record,
        "inputs": context.input_record,
        "schedule": {
            "tasks": len(context.scheduled_tasks),
            "candidates_per_task": EXPECTED_CANDIDATES,
            "task_ids_sha256": canonical_sha256(
                [task.task_id for task in context.scheduled_tasks]
            ),
            "source_sha256s_sha256": canonical_sha256(
                [task.source_sha256 for task in context.scheduled_tasks]
            ),
        },
        "verification": {
            "view": "sealed_visible_TRAIN_split_only",
            "stability_runs": STABILITY_RUNS,
            "timeout": args.timeout,
            "workers": args.evaluation_workers,
        },
        "eligibility": {
            "semantic_tasks": "K4_visible_all_zero",
            "singleton_stdout_compile_call_only": (
                "always_eligible_without_private_answer"
            ),
            "priority": [
                "semantic_visible_all_zero",
                "singleton_stdout_compile_call_only",
            ],
        },
        "private_complete_outcome_consumed_for_eligibility": False,
        "private_complete_diagnostic_consumed": False,
        "heldout_175_opened": False,
        "frontier_api_calls": False,
    }
    events = load_journal(journal_path)
    if not events:
        append_event(
            journal_path,
            {
                "event": "header",
                "schema": JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
        events = load_journal(journal_path)
    terminals, complete = validate_journal(events, contract=contract, context=context)
    if not complete:
        validate_dart_binary()
        for position in range(len(terminals), len(context.scheduled_tasks)):
            task = context.scheduled_tasks[position]
            event = project_task(
                task=task,
                terminal=context.terminals[position],
                task_position=position,
                timeout=args.timeout,
                workers=args.evaluation_workers,
                visible_metadata=context.visible_metadata[task.task_id],
            )
            terminals.append(append_event(journal_path, event))
            print(
                json.dumps(
                    {
                        "task": position + 1,
                        "tasks": len(context.scheduled_tasks),
                        "task_id": event["task_id"],
                        "visible_unique_passes": event["visible_unique_passes"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        append_event(
            journal_path,
            {
                "event": "complete",
                "schema": JOURNAL_SCHEMA,
                "tasks": len(terminals),
                "task_ids_sha256": canonical_sha256(
                    [row["task_id"] for row in terminals]
                ),
            },
        )
        terminals, complete = validate_journal(
            load_journal(journal_path), contract=contract, context=context
        )
    if not complete:
        raise RuntimeError("typed visible projection did not complete")
    zero_ids = [
        str(row["task_id"])
        for row in terminals
        if row["visible_unique_passes"] == 0
    ]
    eligible_ids = [
        str(row["task_id"]) for row in terminals if row["api_eligible"] is True
    ]
    singleton_ids = [
        str(row["task_id"])
        for row in terminals
        if row["api_eligibility_stratum"]
        == "singleton_stdout_compile_call_only"
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": canonical_sha256(contract),
        "tasks": len(terminals),
        "candidate_executions": len(terminals) * EXPECTED_CANDIDATES,
        "visible_all_zero_tasks": len(zero_ids),
        "visible_all_zero_task_ids_sha256": canonical_sha256(zero_ids),
        "api_eligible_tasks": len(eligible_ids),
        "api_eligible_task_ids_sha256": canonical_sha256(eligible_ids),
        "singleton_stdout_compile_call_only_tasks": len(singleton_ids),
        "singleton_stdout_task_ids_sha256": canonical_sha256(singleton_ids),
        "journal": journal_record(journal_path),
        "privacy": {
            "eligibility_source": "sealed_visible_TRAIN_split_only",
            "private_complete_outcome_consumed_for_eligibility": False,
            "private_complete_diagnostic_consumed": False,
            "singleton_private_answer_used_for_eligibility": False,
            "heldout_175_opened": False,
            "frontier_api_calls": False,
        },
    }
    require_exact_or_write(output_dir / "visible_projection_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    # Sealed local source.
    parser.add_argument("--local_harvest_report", required=True)
    parser.add_argument("--expected_local_harvest_report_sha256", required=True)
    parser.add_argument("--pilot_journal", required=True)
    parser.add_argument("--expected_local_harvest_journal_sha256", required=True)
    parser.add_argument("--local_harvest_targets", required=True)
    parser.add_argument("--expected_local_harvest_targets_sha256", required=True)
    parser.add_argument("--existing_direct_manifest", required=True)
    parser.add_argument("--expected_existing_direct_manifest_sha256", required=True)
    # Gold typed-source reconstruction and heldout exclusion seal.
    parser.add_argument("--gold_train_jsonl", required=True)
    parser.add_argument("--expected_gold_train_sha256", required=True)
    parser.add_argument("--gold_f2_jsonl", required=True)
    parser.add_argument("--expected_gold_f2_sha256", required=True)
    parser.add_argument("--expected_gold_rows", type=int, default=2776)
    parser.add_argument("--heldout_jsonl", required=True)
    parser.add_argument("--expected_heldout_sha256", required=True)
    parser.add_argument("--expected_heldout_rows", type=int, default=175)
    # Full-clean visible/complement split.
    parser.add_argument("--visible_split_manifest", required=True)
    parser.add_argument("--expected_visible_split_manifest_sha256", required=True)
    parser.add_argument("--visible_train", required=True)
    parser.add_argument("--expected_visible_train_sha256", required=True)
    parser.add_argument("--private_split_holdback", required=True)
    parser.add_argument("--expected_private_split_holdback_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--evaluation_workers", type=int, default=16)
    args = parser.parse_args(argv)
    if (
        args.expected_gold_rows != 2776
        or args.expected_heldout_rows != 175
        or args.timeout <= 0
        or args.evaluation_workers <= 0
    ):
        parser.error("typed visible projection fixes row counts and positive runtime caps")
    for name, value in vars(args).items():
        if name.startswith("expected_") and name.endswith("_sha256"):
            cascade._require_digest(value, name)  # noqa: SLF001
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
