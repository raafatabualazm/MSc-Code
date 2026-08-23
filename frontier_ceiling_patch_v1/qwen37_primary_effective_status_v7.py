#!/usr/bin/env python3
"""Fail-closed merged status for original, capacity, and length overlays."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import frontier_passk as runner
import frontier_passk_qwen_completion as qwen_entry
import qwen37_capacity_fallback_v6 as capacity
import qwen37_capacity_length_repair_v7 as capacity_length
import qwen37_length_repair_v5 as original_length
import qwen37_original_outcome_reconciliation_v1 as reconciliation
import qwen37_primary_alias_status_v5 as primary


SCHEMA = "qwen37-primary-effective-status-v7"
EXPECTED_TASKS = 175
EXPECTED_K = 10
ARMS = ("opus", "codex")
CAPACITY_OUTPUTS = tuple(
    (partition, arm, f"qwen37_capacity_v6_{partition}_{arm}_mc12k_tb8k")
    for partition in ("0520", "0608")
    for arm in ARMS
)
CAPACITY_LENGTH_PATTERN = (
    "qwen37_capacity_length_v7_{partition}_{arm}_repair_epoch_*"
)


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return runner.sha256_file(path)


def add_usage(
    totals: dict[str, int],
    usage: Mapping[str, Any],
    *,
    cap: int,
) -> None:
    required = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "answer_tokens",
    )
    normalized: dict[str, int] = {}
    for field in required:
        value = usage.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise AuditError(f"usage field {field!r} is malformed")
        normalized[field] = value
    if (
        normalized["prompt_tokens"] > 12_000
        or normalized["completion_tokens"] > cap
        or normalized["reasoning_tokens"] <= 0
        or normalized["reasoning_tokens"] > 8_192
        or normalized["answer_tokens"]
        != normalized["completion_tokens"] - normalized["reasoning_tokens"]
        or normalized["total_tokens"]
        != normalized["prompt_tokens"] + normalized["completion_tokens"]
    ):
        raise AuditError("normalized usage arithmetic/cap mismatch")
    for field, value in normalized.items():
        totals[field] += value


def validate_outcome(
    row: Mapping[str, Any],
    *,
    candidate_valid: bool,
    evaluator_sha256: str,
) -> None:
    if (
        row.get("evaluator_sha256") != evaluator_sha256
        or type(row.get("compiled")) is not bool
        or type(row.get("passed")) is not bool
        or type(row.get("evaluation_performed")) is not bool
        or not isinstance(row.get("stability_runs"), list)
    ):
        raise AuditError("outcome evaluator/result contract mismatch")
    if candidate_valid:
        if (
            row.get("evaluation_performed") is not True
            or not row.get("stability_runs")
            or row.get("completion_attestation_id")
            != runner.REQUIRED_ATTESTATION_ID
            or row.get("completion_attestation_enforced") is not True
        ):
            raise AuditError("evaluable outcome lacks attested stability evidence")
    elif (
        row.get("evaluation_performed") is not False
        or row.get("compiled") is not False
        or row.get("passed") is not False
    ):
        raise AuditError("invalid candidate outcome is not fail-closed")


def load_originals(
    workspace: Path,
) -> tuple[
    tuple[str, ...],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    set[str],
    dict[str, int],
    list[dict[str, Any]],
]:
    qwen_entry.install_qwen_completion_policy()
    patch_root = workspace / "frontier_ceiling_patch_v1"
    primary.validate_meta_contract(patch_root)
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    task_sequence: tuple[str, ...] | None = None
    terminals: dict[tuple[str, str, int], dict[str, Any]] = {}
    outcomes: dict[tuple[str, str, int], dict[str, Any]] = {}
    response_ids: set[str] = set()
    usage_totals: dict[str, int] = defaultdict(int)
    progress: list[dict[str, Any]] = []
    for shard in primary.SHARDS:
        for arm in ARMS:
            root = run_root / shard.directory_template.format(arm=arm)
            if sha256_file(root / shard.copied_contract) != (
                shard.copied_contract_sha256
            ):
                raise AuditError(f"source copied contract mismatch: {root}")
            provenance = primary.read_json(root / "provenance.json")
            config_sha, endpoint_sha = primary.validate_config_and_provenance(
                provenance,
                shard=shard,
                arm=arm,
            )
            tasks = primary.read_jsonl(root / "tasks.jsonl")
            prompts = primary.read_jsonl(root / "prompts.jsonl")
            ids = tuple(str(row.get("task_id") or "") for row in tasks)
            prompt_ids = tuple(
                str(row.get("task_id") or "") for row in prompts
            )
            if (
                len(ids) != EXPECTED_TASKS
                or len(set(ids)) != EXPECTED_TASKS
                or ids != prompt_ids
            ):
                raise AuditError("source task/prompt order mismatch")
            if task_sequence is None:
                task_sequence = ids
            elif ids != task_sequence:
                raise AuditError("source shards have different task order")
            prompt_map = {str(row["task_id"]): row for row in prompts}
            terminal = original_length.load_source_terminals_outcome_blind(
                root / "attempts.jsonl",
                config_sha256=config_sha,
                prompt_map=prompt_map,
                requested_model=shard.model,
                local_k=shard.local_k,
                slot_policy_sha256=str(
                    provenance["config"]["slot_policy_sha256"]
                ),
                response_ids=response_ids,
            )
            outcome = runner.load_resume_outcomes(
                root / "outcomes.jsonl",
                config_sha=config_sha,
                evaluator_sha256=primary.EXPECTED_EVALUATOR_SHA256,
            )
            for (task_id, local_index), terminal_row in terminal.items():
                global_key = (
                    arm,
                    task_id,
                    shard.global_indices[local_index],
                )
                if global_key in terminals:
                    raise AuditError("duplicate original global terminal")
                usage = terminal_row.get("usage")
                if not isinstance(usage, Mapping):
                    raise AuditError("original terminal has no usage")
                add_usage(usage_totals, usage, cap=12_298)
                terminals[global_key] = terminal_row
            for (task_id, local_index, attempt_id), outcome_row in outcome.items():
                terminal_row = terminal.get((task_id, local_index))
                if (
                    terminal_row is None
                    or terminal_row.get("attempt_id") != attempt_id
                    or terminal_row.get("response_id")
                    != outcome_row.get("response_id")
                    or terminal_row.get("code_sha256")
                    != outcome_row.get("code_sha256")
                ):
                    raise AuditError("original outcome is not terminal-backed")
                validate_outcome(
                    outcome_row,
                    candidate_valid=bool(terminal_row["candidate_valid"]),
                    evaluator_sha256=primary.EXPECTED_EVALUATOR_SHA256,
                )
                global_key = (
                    arm,
                    task_id,
                    shard.global_indices[local_index],
                )
                if global_key in outcomes:
                    raise AuditError("duplicate original global outcome")
                outcomes[global_key] = outcome_row
            attempts = primary.read_jsonl(root / "attempts.jsonl")
            progress.append(
                {
                    "source": "original",
                    "shard": shard.key,
                    "arm": arm,
                    "model": shard.model,
                    "endpoint_sha256": endpoint_sha,
                    "attempt_rows": len(attempts),
                    "terminal_responses": len(terminal),
                    "outcomes": len(outcome),
                    "response_less_nonretryable": sum(
                        row.get("response_received") is False
                        and row.get("retryable_transport") is False
                        for row in attempts
                    ),
                    "failure_receipt_sha256": sha256_file(
                        root / "failure.json"
                    ),
                }
            )
    if task_sequence is None:
        raise AuditError("no original task sequence")
    missing_outcomes = set(terminals).difference(outcomes)
    if missing_outcomes:
        try:
            reconciled, reconciliation_progress = (
                reconciliation.load_effective_outcomes(
                    workspace,
                    original_terminals=terminals,
                )
            )
        except Exception as exc:
            raise AuditError(
                "original terminal outcomes require the sealed local "
                f"reconciliation overlay: {exc}"
            ) from exc
        if set(reconciled) != missing_outcomes:
            raise AuditError(
                "reconciliation overlay does not exactly cover original "
                "returned-but-unscored terminals"
            )
        for key, outcome in reconciled.items():
            validate_outcome(
                outcome,
                candidate_valid=bool(terminals[key]["candidate_valid"]),
                evaluator_sha256=primary.EXPECTED_EVALUATOR_SHA256,
            )
            outcomes[key] = outcome
        progress.append(reconciliation_progress)
    if set(outcomes) != set(terminals):
        raise AuditError(
            "original terminals and source-plus-reconciled outcomes differ"
        )
    return (
        task_sequence,
        terminals,
        outcomes,
        response_ids,
        dict(usage_totals),
        progress,
    )


def capacity_outputs(
    workspace: Path,
    *,
    expected_contract_sha256: str,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    set[str],
    dict[str, int],
    list[dict[str, Any]],
]:
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    terminals: dict[tuple[str, str, int], dict[str, Any]] = {}
    outcomes: dict[tuple[str, str, int], dict[str, Any]] = {}
    response_ids: set[str] = set()
    usage_totals: dict[str, int] = defaultdict(int)
    progress: list[dict[str, Any]] = []
    for partition, arm, directory in CAPACITY_OUTPUTS:
        root = run_root / directory
        effective_rows = capacity._load_effective(
            root / "effective_terminals.jsonl"
        )
        feed_rows = capacity._load_terminal_feed(
            root / "effective_terminal_feed.jsonl"
        )
        raw_outcomes = capacity.read_jsonl(root / "outcomes.jsonl")
        outcome_by_selection: dict[str, dict[str, Any]] = {}
        for row in raw_outcomes:
            selection_id = str(row.get("selection_id") or "")
            if not selection_id or selection_id in outcome_by_selection:
                raise AuditError("duplicate capacity outcome selection")
            outcome_by_selection[selection_id] = row
        if set(effective_rows) != set(feed_rows):
            raise AuditError("capacity effective/feed selection mismatch")
        if not set(effective_rows).issubset(outcome_by_selection):
            raise AuditError("capacity effective terminal lacks outcome")
        for selection_id, effective in effective_rows.items():
            feed = feed_rows[selection_id]
            outcome = outcome_by_selection[selection_id]
            if (
                effective.get("schema") != capacity.SCHEMA
                or effective.get("record_type")
                != "effective_capacity_terminal"
                or feed.get("schema") != capacity.SCHEMA
                or feed.get("record_type")
                != "capacity_effective_terminal_feed"
                or outcome.get("schema") != capacity.SCHEMA
                or effective.get("overlay_contract_sha256")
                != expected_contract_sha256
                or feed.get("overlay_contract_sha256")
                != expected_contract_sha256
                or effective.get("arm") != arm
                or feed.get("arm") != arm
                or effective.get("task_id") != feed.get("task_id")
                or effective.get("global_sample_index")
                != feed.get("global_sample_index")
                or effective.get("response_id") != feed.get("response_id")
                or effective.get("finish_reason") != feed.get("finish_reason")
                or effective.get("candidate_valid")
                != feed.get("candidate_valid")
                or effective.get("code_sha256") != feed.get("code_sha256")
                or effective.get("usage") != feed.get("validated_usage")
                or effective.get("canonical_terminal_row_sha256")
                != feed.get("effective_terminal_canonical_row_sha256")
                or effective.get("canonical_outcome_row_sha256")
                != capacity.canonical_sha(outcome)
                or effective.get("compiled") != outcome.get("compiled")
                or effective.get("passed") != outcome.get("passed")
            ):
                raise AuditError("capacity effective/feed/outcome mismatch")
            capacity_length.validate_feed_terminal(
                feed,
                expected_capacity_contract_sha256=(
                    expected_contract_sha256
                ),
            )
            validate_outcome(
                outcome,
                candidate_valid=bool(effective["candidate_valid"]),
                evaluator_sha256=primary.EXPECTED_EVALUATOR_SHA256,
            )
            response_id = str(effective.get("response_id") or "")
            if not response_id or response_id in response_ids:
                raise AuditError("duplicate capacity response ID")
            response_ids.add(response_id)
            usage = effective.get("usage")
            if not isinstance(usage, Mapping):
                raise AuditError("capacity effective terminal lacks usage")
            add_usage(usage_totals, usage, cap=12_298)
            key = (
                arm,
                str(effective["task_id"]),
                int(effective["global_sample_index"]),
            )
            if key in terminals:
                raise AuditError("duplicate capacity global terminal")
            terminals[key] = effective
            outcomes[key] = outcome
        status_path = root / "status.json"
        status_row = (
            capacity.read_json(status_path) if status_path.is_file() else {}
        )
        progress.append(
            {
                "source": "capacity_v6",
                "partition": partition,
                "arm": arm,
                "root": str(root),
                "status": status_row.get("status", "not_started"),
                "effective_terminals": len(effective_rows),
                "outcomes": len(
                    set(effective_rows).intersection(outcome_by_selection)
                ),
                "length_terminals": sum(
                    row.get("finish_reason") == "length"
                    for row in feed_rows.values()
                ),
                "remaining": status_row.get("remaining"),
            }
        )
    return (
        terminals,
        outcomes,
        response_ids,
        dict(usage_totals),
        progress,
    )


def original_length_outputs(
    workspace: Path,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    set[str],
    dict[str, int],
    dict[str, Any],
]:
    root = (
        workspace
        / "artifacts"
        / "frontier_ceiling_two_enrichments"
        / "runs"
        / "qwen37_primary_alias_length_repairs_v5"
    )
    sources, _all_final = original_length.scan_length_sources(workspace)
    source_by_key = {
        str(source["source_slot_key"]): source for source in sources
    }
    if not root.is_dir():
        return {}, set(), {}, {
            "source": "original_length_v5",
            "observed_length_slots": len(source_by_key),
            "repairs": 0,
        }
    provenance = primary.read_json(root / "provenance.json")
    config_sha = str(provenance.get("config_sha256") or "")
    terminals, outcomes = original_length.load_existing_repairs(
        root / "repair_attempts.jsonl",
        root / "repair_outcomes.jsonl",
        config_sha256=config_sha,
    )
    if not set(terminals).issubset(source_by_key):
        raise AuditError("v5 length repair contains a foreign source")
    mapped: dict[tuple[str, str, int], dict[str, Any]] = {}
    response_ids: set[str] = set()
    usage_totals: dict[str, int] = defaultdict(int)
    for source_key, outcome in outcomes.items():
        source = source_by_key[source_key]
        key = (
            str(source["arm"]),
            str(source["task_id"]),
            int(source["global_sample_index"]),
        )
        if key in mapped:
            raise AuditError("duplicate v5 length global replacement")
        mapped[key] = outcome
        terminal = terminals[source_key]
        response_id = str(terminal.get("response_id") or "")
        if not response_id or response_id in response_ids:
            raise AuditError("duplicate v5 repair response ID")
        response_ids.add(response_id)
        usage = terminal.get("usage")
        if not isinstance(usage, Mapping):
            raise AuditError("v5 repair lacks usage")
        add_usage(usage_totals, usage, cap=24_586)
        validate_outcome(
            outcome,
            candidate_valid=bool(terminal["candidate_valid"]),
            evaluator_sha256=primary.EXPECTED_EVALUATOR_SHA256,
        )
    return (
        mapped,
        response_ids,
        dict(usage_totals),
        {
            "source": "original_length_v5",
            "observed_length_slots": len(source_by_key),
            "terminal_repairs": len(terminals),
            "evaluated_repairs": len(outcomes),
        },
    )


def capacity_length_outputs(
    workspace: Path,
    *,
    expected_capacity_contract_sha256: str,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    set[str],
    dict[str, int],
    list[dict[str, Any]],
]:
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    mapped: dict[tuple[str, str, int], dict[str, Any]] = {}
    response_ids: set[str] = set()
    usage_totals: dict[str, int] = defaultdict(int)
    progress: list[dict[str, Any]] = []
    for partition in ("0520", "0608"):
        for arm in ARMS:
            capacity_root = (
                run_root
                / f"qwen37_capacity_v6_{partition}_{arm}_mc12k_tb8k"
            )
            sources = capacity_length.scan_capacity_length_sources(
                capacity_root / "effective_terminal_feed.jsonl",
                expected_capacity_contract_sha256=(
                    expected_capacity_contract_sha256
                ),
            )
            source_by_key = {
                str(source["capacity_source_key"]): source
                for source in sources
            }
            epoch_roots = sorted(
                run_root.glob(
                    CAPACITY_LENGTH_PATTERN.format(
                        partition=partition,
                        arm=arm,
                    )
                )
            )
            known_roots = {str(root.resolve()) for root in epoch_roots}
            aggregate_terminal: dict[str, dict[str, Any]] = {}
            aggregate_outcomes: dict[str, dict[str, Any]] = {}
            epoch_progress: list[dict[str, Any]] = []
            dependency_graph: dict[str, set[str]] = {}
            for root in epoch_roots:
                provenance = capacity.read_json(root / "provenance.json")
                config = provenance.get("config")
                if (
                    not isinstance(config, Mapping)
                    or config.get("capacity_out")
                    != str(capacity_root.resolve())
                ):
                    raise AuditError(
                        "v7 epoch capacity-source binding mismatch"
                    )
                current_root = str(root.resolve())
                priors = {
                    str(Path(value).resolve())
                    for value in config.get("prior_repair_outputs", [])
                }
                if current_root in priors or not priors.issubset(known_roots):
                    raise AuditError("v7 epoch prior-output graph is foreign")
                dependency_graph[current_root] = priors
                config_sha = str(provenance.get("config_sha256") or "")
                terminals, outcomes = capacity_length.load_existing(
                    root / "repair_attempts.jsonl",
                    root / "repair_outcomes.jsonl",
                    config_sha256=config_sha,
                    allow_sealed_quota_boundary=True,
                )
                if not set(terminals).issubset(source_by_key):
                    raise AuditError(
                        "v7 length repair contains a foreign source"
                    )
                for source_key, terminal in terminals.items():
                    if source_key in aggregate_terminal:
                        raise AuditError(
                            "v7 endpoint epochs duplicated a terminal repair"
                        )
                    if source_key not in outcomes:
                        raise AuditError(
                            "v7 epoch terminal response lacks an outcome"
                        )
                    aggregate_terminal[source_key] = terminal
                    aggregate_outcomes[source_key] = outcomes[source_key]
                epoch_progress.append(
                    {
                        "repair_epoch": config.get("repair_endpoint_epoch"),
                        "repair_endpoint_sha256": config.get(
                            "api_endpoint_sha256"
                        ),
                        "terminal_repairs": len(terminals),
                        "evaluated_repairs": len(outcomes),
                        "failed_receipt_sha256": sha256_file(
                            root / "failure.json"
                        ),
                    }
                )
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(node: str) -> None:
                if node in visiting:
                    raise AuditError("v7 prior-output graph has a cycle")
                if node in visited:
                    return
                visiting.add(node)
                for prior in dependency_graph.get(node, set()):
                    visit(prior)
                visiting.remove(node)
                visited.add(node)

            for node in dependency_graph:
                visit(node)
            for source_key, outcome in aggregate_outcomes.items():
                source = source_by_key[source_key]
                feed = source["feed"]
                key = (
                    str(feed["arm"]),
                    str(feed["task_id"]),
                    int(feed["global_sample_index"]),
                )
                if key in mapped:
                    raise AuditError(
                        "duplicate v7 length global replacement"
                    )
                mapped[key] = outcome
                terminal = aggregate_terminal[source_key]
                response_id = str(terminal.get("response_id") or "")
                if not response_id or response_id in response_ids:
                    raise AuditError("duplicate v7 repair response ID")
                response_ids.add(response_id)
                usage = terminal.get("usage")
                if not isinstance(usage, Mapping):
                    raise AuditError("v7 repair lacks usage")
                add_usage(usage_totals, usage, cap=24_586)
                validate_outcome(
                    outcome,
                    candidate_valid=bool(terminal["candidate_valid"]),
                    evaluator_sha256=primary.EXPECTED_EVALUATOR_SHA256,
                )
            progress.append(
                {
                    "source": "capacity_length_v7",
                    "partition": partition,
                    "arm": arm,
                    "observed_length_slots": len(source_by_key),
                    "terminal_repairs": len(aggregate_terminal),
                    "evaluated_repairs": len(aggregate_outcomes),
                    "endpoint_epochs": epoch_progress,
                }
            )
    return mapped, response_ids, dict(usage_totals), progress


def metrics_if_complete(
    outcomes: Mapping[tuple[str, str, int], Mapping[str, Any]],
    task_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    return primary.metrics_if_complete(outcomes, task_ids)


def build_adaptive_outcomes(
    *,
    fixed_terminals: Mapping[tuple[str, str, int], Mapping[str, Any]],
    fixed_outcomes: Mapping[tuple[str, str, int], dict[str, Any]],
    original_keys: set[tuple[str, str, int]],
    capacity_keys: set[tuple[str, str, int]],
    original_replacements: Mapping[
        tuple[str, str, int], dict[str, Any]
    ],
    capacity_replacements: Mapping[
        tuple[str, str, int], dict[str, Any]
    ],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    list[tuple[str, str, int]],
]:
    if original_keys.intersection(capacity_keys):
        raise AuditError("source-kind key sets overlap")
    if set(fixed_terminals) != original_keys.union(capacity_keys):
        raise AuditError("source-kind keys do not cover fixed terminals")
    if set(fixed_outcomes) != set(fixed_terminals):
        raise AuditError("fixed terminal/outcome keys differ")
    if not set(original_replacements).issubset(original_keys):
        raise AuditError("v5 replacement is not for an original terminal")
    if not set(capacity_replacements).issubset(capacity_keys):
        raise AuditError("v7 replacement is not for a capacity terminal")
    for key in original_replacements:
        if fixed_terminals[key].get("finish_reason") != "length":
            raise AuditError("v5 replacement targets a non-length terminal")
    for key in capacity_replacements:
        if fixed_terminals[key].get("finish_reason") != "length":
            raise AuditError("v7 replacement targets a non-length terminal")
    adaptive: dict[tuple[str, str, int], dict[str, Any]] = {}
    waiting: list[tuple[str, str, int]] = []
    for key, terminal in fixed_terminals.items():
        if terminal.get("finish_reason") != "length":
            adaptive[key] = fixed_outcomes[key]
            continue
        replacements = (
            original_replacements
            if key in original_keys
            else capacity_replacements
        )
        replacement = replacements.get(key)
        if replacement is None:
            waiting.append(key)
        else:
            adaptive[key] = replacement
    return adaptive, waiting


def aggregate(
    workspace: Path,
    *,
    expected_capacity_contract_sha256: str,
    expected_capacity_script_sha256: str,
    expected_capacity_length_script_sha256: str,
) -> dict[str, Any]:
    patch_root = workspace / "frontier_ceiling_patch_v1"
    if (
        sha256_file(patch_root / "qwen37_capacity_fallback_v6.py")
        != expected_capacity_script_sha256
        or sha256_file(
            patch_root / "qwen37_capacity_fallback_contract_v6.json"
        )
        != expected_capacity_contract_sha256
        or sha256_file(patch_root / "qwen37_capacity_length_repair_v7.py")
        != expected_capacity_length_script_sha256
    ):
        raise AuditError("merged-status dependency hash mismatch")
    (
        task_ids,
        original_terminals,
        original_outcomes,
        original_ids,
        original_usage,
        original_progress,
    ) = load_originals(workspace)
    (
        capacity_terminals,
        capacity_outcome_map,
        capacity_ids,
        capacity_usage,
        capacity_progress,
    ) = capacity_outputs(
        workspace,
        expected_contract_sha256=expected_capacity_contract_sha256,
    )
    if set(original_terminals).intersection(capacity_terminals):
        raise AuditError("capacity overlay resampled an original terminal slot")
    fixed_terminals = {**original_terminals, **capacity_terminals}
    fixed_outcomes = {**original_outcomes, **capacity_outcome_map}
    expected_keys = {
        (arm, task_id, sample_index)
        for arm in ARMS
        for task_id in task_ids
        for sample_index in range(EXPECTED_K)
    }
    if (
        not set(fixed_terminals).issubset(expected_keys)
        or set(fixed_outcomes) != set(fixed_terminals)
    ):
        raise AuditError("fixed effective terminal/outcome key mismatch")
    (
        original_replacements,
        original_repair_ids,
        original_repair_usage,
        original_repair_progress,
    ) = original_length_outputs(workspace)
    (
        capacity_replacements,
        capacity_repair_ids,
        capacity_repair_usage,
        capacity_repair_progress,
    ) = capacity_length_outputs(
        workspace,
        expected_capacity_contract_sha256=expected_capacity_contract_sha256,
    )
    all_id_sets = (
        original_ids,
        capacity_ids,
        original_repair_ids,
        capacity_repair_ids,
    )
    if sum(len(values) for values in all_id_sets) != len(
        set().union(*all_id_sets)
    ):
        raise AuditError("response ID reused across source/overlay layers")
    adaptive, waiting_length = build_adaptive_outcomes(
        fixed_terminals=fixed_terminals,
        fixed_outcomes=fixed_outcomes,
        original_keys=set(original_terminals),
        capacity_keys=set(capacity_terminals),
        original_replacements=original_replacements,
        capacity_replacements=capacity_replacements,
    )
    fixed_metrics = metrics_if_complete(fixed_outcomes, task_ids)
    adaptive_metrics = metrics_if_complete(adaptive, task_ids)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "expected": {
            "tasks_per_arm": EXPECTED_TASKS,
            "k_per_arm": EXPECTED_K,
            "global_slots": len(expected_keys),
        },
        "progress": {
            "original_effective_slots": len(original_terminals),
            "capacity_effective_slots": len(capacity_terminals),
            "fixed_effective_slots": len(fixed_terminals),
            "remaining_fixed_slots": len(expected_keys)
            - len(fixed_terminals),
            "fixed_length_slots": sum(
                row.get("finish_reason") == "length"
                for row in fixed_terminals.values()
            ),
            "adaptive_outcomes": len(adaptive),
            "waiting_length_repairs": len(waiting_length),
        },
        "layers": [
            *original_progress,
            *capacity_progress,
            original_repair_progress,
            *capacity_repair_progress,
        ],
        "usage": {
            "original_12k": original_usage,
            "capacity_v6_12k": capacity_usage,
            "original_length_v5_24k": original_repair_usage,
            "capacity_length_v7_24k": capacity_repair_usage,
        },
        "metrics_withheld_until_complete": True,
    }
    if fixed_metrics is not None:
        report["fixed_12k_metrics"] = fixed_metrics
    if adaptive_metrics is not None:
        report["adaptive_24k_metrics"] = adaptive_metrics
    if fixed_metrics is None:
        report["status"] = "in_progress"
    elif waiting_length or adaptive_metrics is None:
        report["status"] = "awaiting_length_repairs"
    else:
        report["status"] = "complete"
        report["metrics_withheld_until_complete"] = False
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--expected-capacity-contract-sha256", required=True)
    parser.add_argument("--expected-capacity-script-sha256", required=True)
    parser.add_argument("--expected-capacity-length-script-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    for name in (
        "expected_capacity_contract_sha256",
        "expected_capacity_script_sha256",
        "expected_capacity_length_script_sha256",
    ):
        value = str(getattr(args, name)).strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            parser.error(f"--{name.replace('_', '-')} must be SHA-256")
        setattr(args, name, value)
    return args


def main() -> int:
    args = parse_args()
    try:
        report = aggregate(
            args.workspace.resolve(),
            expected_capacity_contract_sha256=(
                args.expected_capacity_contract_sha256
            ),
            expected_capacity_script_sha256=(
                args.expected_capacity_script_sha256
            ),
            expected_capacity_length_script_sha256=(
                args.expected_capacity_length_script_sha256
            ),
        )
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output = json.dumps(report, sort_keys=True)
        if args.out:
            runner.atomic_write_json(args.out.resolve(), report)
        print(output)
        return 2
    if args.out:
        runner.atomic_write_json(args.out.resolve(), report)
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
