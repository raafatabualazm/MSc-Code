#!/usr/bin/env python3
"""Provider-free reconciliation of six returned Qwen primary responses.

The original journals are read-only inputs.  This overlay first seals the exact
terminal slots without opening pass/compile results, then evaluates only those
sealed candidates with the already-pinned local Dart evaluator.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import frontier_passk as runner
import qwen37_length_repair_v5 as original_length
import qwen37_primary_alias_status_v5 as primary


SCHEMA = "qwen37-original-outcome-reconciliation-v1"
CONTRACT_NAME = "qwen37_original_outcome_reconciliation_contract_v1.json"
EXPECTED_CONTRACT_SHA256 = (
    "ebbb10cfa939c1fa4fbdd26e46f058960a7b1841a56d53cb6c476184f1644825"
)
EXPECTED_EVALUATOR_SHA256 = primary.EXPECTED_EVALUATOR_SHA256
EXPECTED_DART_SHA256 = (
    "c03ad868b5c53e31461b0fef22dc6eb6aeb56b7567efff6ca488ce9c4a6f8a6a"
)
EXPECTED_ORPHANS = 6
EXPECTED_SHARDS = 8
DEFAULT_OUTPUT = (
    "artifacts/frontier_ceiling_two_enrichments/runs/"
    "qwen37_primary_original_outcome_reconciliation_v1"
)
SOURCE_SNAPSHOT_FILES = (
    "provenance.json",
    "tasks.jsonl",
    "prompts.jsonl",
    "attempts.jsonl",
    "outcomes.jsonl",
)


class ReconciliationError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReconciliationError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconciliationError(f"JSON artifact is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReconciliationError(
                    f"{path}:{line_number} is not an object"
                )
            rows.append(value)
    return rows


def canonical_sha(value: Any) -> str:
    return runner.stable_sha256(value)


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ReconciliationError(f"missing sealed file: {path}")
    return {
        "path": str(path),
        "sha256": runner.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def assert_provider_free_source(script_path: Path | None = None) -> None:
    """Statically reject provider client imports/call expressions in this file."""
    path = (script_path or Path(__file__)).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_roots = {"openai", "anthropic", "dashscope"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in forbidden_roots for alias in node.names):
                raise ReconciliationError("provider client import is forbidden")
        elif isinstance(node, ast.ImportFrom):
            if str(node.module or "").split(".", 1)[0] in forbidden_roots:
                raise ReconciliationError("provider client import is forbidden")
        elif isinstance(node, ast.Call):
            rendered = ast.unparse(node.func)
            if (
                "chat.completions" in rendered
                or "responses.create" in rendered
                or "api_credentials" in rendered
            ):
                raise ReconciliationError(
                    "provider/API call expression is forbidden"
                )


def load_contract(patch_root: Path) -> dict[str, Any]:
    path = patch_root / CONTRACT_NAME
    if runner.sha256_file(path) != EXPECTED_CONTRACT_SHA256:
        raise ReconciliationError("reconciliation contract SHA mismatch")
    contract = read_json(path)
    if (
        contract.get("schema")
        != "qwen37-original-outcome-reconciliation-contract-v1"
        or contract.get("parent_primary_contract_sha256")
        != primary.EXPECTED_META_CONTRACT_SHA256
    ):
        raise ReconciliationError("reconciliation parent contract mismatch")
    network = contract.get("network_policy")
    if not isinstance(network, Mapping) or any(network.values()):
        raise ReconciliationError("reconciliation network policy is not all false")
    expected = contract.get("expected_orphans")
    if not isinstance(expected, list) or len(expected) != EXPECTED_ORPHANS:
        raise ReconciliationError("contract does not pin exactly six orphans")
    identities: set[tuple[str, str, int, str]] = set()
    response_ids: set[str] = set()
    for row in expected:
        if not isinstance(row, Mapping):
            raise ReconciliationError("malformed expected orphan")
        identity = (
            str(row.get("arm") or ""),
            str(row.get("task_id") or ""),
            int(row.get("global_sample_index", -1)),
            str(row.get("attempt_id") or ""),
        )
        response_id = str(row.get("response_id") or "")
        if (
            not identity[0]
            or not identity[1]
            or not identity[3]
            or identity[2] < 0
            or identity in identities
            or not response_id
            or response_id in response_ids
        ):
            raise ReconciliationError("duplicate/malformed expected orphan identity")
        identities.add(identity)
        response_ids.add(response_id)
    return contract


def validate_sealed_artifacts(provenance: Mapping[str, Any]) -> None:
    config = provenance.get("config")
    artifacts = provenance.get("artifacts")
    if not isinstance(config, Mapping) or not isinstance(artifacts, Mapping):
        raise ReconciliationError("source provenance lacks sealed artifacts")
    sealed = config.get("sealed_inputs")
    if not isinstance(sealed, Mapping):
        raise ReconciliationError("source provenance lacks sealed_inputs")
    keys = (
        "prompt_jsonl",
        "prompt_manifest",
        "eval_jsonl",
        "eval_seal",
        "pair_manifest",
    )
    for key in keys:
        raw_path = sealed.get(key)
        expected_sha = str(sealed.get(f"{key}_sha256") or "")
        record = artifacts.get(key)
        if (
            not isinstance(raw_path, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not isinstance(record, Mapping)
            or record.get("path") != raw_path
            or record.get("sha256") != expected_sha
            or runner.sha256_file(Path(raw_path).resolve()) != expected_sha
        ):
            raise ReconciliationError(f"sealed artifact mismatch for {key}")
    frontier = artifacts.get("frontier_f2")
    if (
        not isinstance(frontier, Mapping)
        or frontier.get("sha256") != primary.EXPECTED_F2_SHA256
        or runner.sha256_file(Path(str(frontier.get("path") or "")).resolve())
        != primary.EXPECTED_F2_SHA256
    ):
        raise ReconciliationError("sealed frontier_f2 artifact mismatch")


def load_test_bundle(
    source_root: Path,
    tasks: list[dict[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    config = provenance["config"]
    sealed = config["sealed_inputs"]
    eval_path = Path(str(sealed["eval_jsonl"])).resolve()
    if runner.sha256_file(eval_path) != sealed["eval_jsonl_sha256"]:
        raise ReconciliationError("sealed eval dataset SHA mismatch")
    eval_rows = read_jsonl(eval_path)
    if len(tasks) != primary.EXPECTED_TASKS or len(eval_rows) != len(tasks):
        raise ReconciliationError("sealed eval/task count mismatch")
    task_ids = [str(row.get("task_id") or "") for row in tasks]
    eval_ids = [str(row.get("task_id") or "") for row in eval_rows]
    if (
        task_ids != eval_ids
        or len(set(task_ids)) != primary.EXPECTED_TASKS
        or any(not value for value in task_ids)
    ):
        raise ReconciliationError("sealed eval/task order mismatch")
    tests: dict[str, str] = {}
    for task, eval_row in zip(tasks, eval_rows, strict=True):
        task_id = str(task["task_id"])
        acceptance = eval_row.get("acceptance_tests")
        if (
            not isinstance(acceptance, str)
            or not acceptance.strip()
            or eval_row.get("tests") != acceptance
            or task.get("tests_equal_acceptance_tests") is not True
            or task.get("tests_sha256") != runner.sha256_text(acceptance)
            or task.get("acceptance_tests_sha256")
            != runner.sha256_text(acceptance)
        ):
            raise ReconciliationError(
                f"sealed private-test binding mismatch for {task_id}"
            )
        tests[task_id] = acceptance
    return tests


def validate_evaluator_and_dart(
    provenance: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    record = provenance.get("evaluator")
    if not isinstance(record, Mapping):
        raise ReconciliationError("source evaluator provenance missing")
    dart = record.get("dart_binary")
    if (
        record.get("sha256") != EXPECTED_EVALUATOR_SHA256
        or not isinstance(dart, Mapping)
        or dart.get("sha256") != EXPECTED_DART_SHA256
    ):
        raise ReconciliationError("source evaluator/Dart pin mismatch")
    module, actual = runner.import_evaluator(
        Path(str(record.get("path") or "")),
        EXPECTED_EVALUATOR_SHA256,
        dart_binary=Path(str(dart.get("path") or "")),
        expected_dart_hash=EXPECTED_DART_SHA256,
        validate_dart=True,
    )
    if (
        actual.get("sha256") != record.get("sha256")
        or actual.get("entrypoint") != record.get("entrypoint")
        or actual["dart_binary"].get("sha256") != dart.get("sha256")
    ):
        raise ReconciliationError("live evaluator/Dart differs from provenance")
    return module, actual


def selection_id(row: Mapping[str, Any]) -> str:
    return canonical_sha(
        {
            "schema": SCHEMA,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "arm": row["arm"],
            "source_shard_key": row["source_shard_key"],
            "task_id": row["task_id"],
            "local_sample_index": row["local_sample_index"],
            "global_sample_index": row["global_sample_index"],
            "attempt_id": row["attempt_id"],
            "response_id": row["response_id"],
            "terminal_row_sha256": row["terminal_row_sha256"],
        }
    )


def _validate_existing_outcome(
    outcome: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> None:
    exact = (
        "task_id",
        "sample_index",
        "attempt_id",
        "response_id",
        "finish_reason",
        "candidate_valid",
        "terminal_reason",
        "code_sha256",
    )
    if any(outcome.get(key) != terminal.get(key) for key in exact):
        raise ReconciliationError("source outcome is not exact-terminal-backed")


def audit_sources(
    workspace: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all eight primary shards and select exact outcome orphans."""
    primary.qwen_entry.install_qwen_completion_policy()
    patch_root = workspace / "frontier_ceiling_patch_v1"
    primary.validate_meta_contract(patch_root)
    run_root = (
        workspace / "artifacts" / "frontier_ceiling_two_enrichments" / "runs"
    )
    expected_by_directory: dict[str, list[dict[str, Any]]] = {}
    for item in contract["expected_orphans"]:
        expected_by_directory.setdefault(
            str(item["source_directory"]), []
        ).append(dict(item))
    snapshots: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    global_response_ids: set[str] = set()
    evaluator_cache: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
    attested_eval_datasets: set[
        tuple[tuple[str, str], str]
    ] = set()
    test_bundles: dict[str, dict[str, str]] = {}
    source_count = 0
    for shard in primary.SHARDS:
        for arm in primary.ARMS:
            source_count += 1
            directory = shard.directory_template.format(arm=arm)
            root = run_root / directory
            copied_contract = root / shard.copied_contract
            if runner.sha256_file(copied_contract) != shard.copied_contract_sha256:
                raise ReconciliationError(
                    f"source copied contract mismatch: {root}"
                )
            provenance = read_json(root / "provenance.json")
            config_sha, _endpoint_sha = primary.validate_config_and_provenance(
                provenance,
                shard=shard,
                arm=arm,
            )
            validate_sealed_artifacts(provenance)
            tasks = read_jsonl(root / "tasks.jsonl")
            prompts = read_jsonl(root / "prompts.jsonl")
            task_ids = [str(row.get("task_id") or "") for row in tasks]
            prompt_ids = [str(row.get("task_id") or "") for row in prompts]
            if (
                len(tasks) != primary.EXPECTED_TASKS
                or task_ids != prompt_ids
                or len(set(task_ids)) != primary.EXPECTED_TASKS
            ):
                raise ReconciliationError("source task/prompt order mismatch")
            for prompt in prompts:
                messages = prompt.get("messages")
                if (
                    not isinstance(messages, list)
                    or prompt.get("prompt_sha256") != canonical_sha(messages)
                    or prompt.get("never_truncated") is not True
                    or prompt.get("tests_exposed") is not False
                ):
                    raise ReconciliationError("source prompt receipt mismatch")
            tests = load_test_bundle(root, tasks, provenance)
            test_bundles[arm] = tests
            evaluator_record = provenance["evaluator"]
            dart_record = evaluator_record["dart_binary"]
            evaluator_key = (
                str(evaluator_record["path"]),
                str(dart_record["path"]),
            )
            if evaluator_key not in evaluator_cache:
                evaluator_cache[evaluator_key] = validate_evaluator_and_dart(
                    provenance
                )
            eval_dataset_key = (
                evaluator_key,
                str(
                    provenance["config"]["sealed_inputs"][
                        "eval_jsonl_sha256"
                    ]
                ),
            )
            if eval_dataset_key not in attested_eval_datasets:
                module = evaluator_cache[evaluator_key][0]
                for task_id, acceptance in tests.items():
                    ok, diagnostic, _source, marker = (
                        module.prepare_dart_test_completion_attestation(
                            acceptance
                        )
                    )
                    if not ok or not marker:
                        raise ReconciliationError(
                            f"private tests cannot be attested for {task_id}: "
                            f"{diagnostic}"
                        )
                attested_eval_datasets.add(eval_dataset_key)
            prompt_map = {
                str(row["task_id"]): row for row in prompts
            }
            terminals = original_length.load_source_terminals_outcome_blind(
                root / "attempts.jsonl",
                config_sha256=config_sha,
                prompt_map=prompt_map,
                requested_model=shard.model,
                local_k=shard.local_k,
                slot_policy_sha256=str(
                    provenance["config"]["slot_policy_sha256"]
                ),
                response_ids=global_response_ids,
            )
            outcomes = runner.load_resume_outcomes(
                root / "outcomes.jsonl",
                config_sha=config_sha,
                evaluator_sha256=EXPECTED_EVALUATOR_SHA256,
            )
            terminal_by_attempt = {
                (task_id, local_index, str(row["attempt_id"])): row
                for (task_id, local_index), row in terminals.items()
            }
            for key, outcome in outcomes.items():
                terminal = terminal_by_attempt.get(key)
                if terminal is None:
                    raise ReconciliationError("source has an orphan outcome")
                _validate_existing_outcome(outcome, terminal)
            missing = [
                (task_id, local_index, row)
                for (task_id, local_index), row in terminals.items()
                if (
                    task_id,
                    local_index,
                    str(row["attempt_id"]),
                )
                not in outcomes
            ]
            expected_here = expected_by_directory.get(directory, [])
            expected_keys = {
                (
                    str(row["task_id"]),
                    int(row["local_sample_index"]),
                    str(row["attempt_id"]),
                )
                for row in expected_here
            }
            missing_keys = {
                (task_id, local_index, str(row["attempt_id"]))
                for task_id, local_index, row in missing
            }
            if missing_keys != expected_keys:
                raise ReconciliationError(
                    f"unscored-terminal set differs from contract in {directory}"
                )
            expected_lookup = {
                (
                    str(row["task_id"]),
                    int(row["local_sample_index"]),
                    str(row["attempt_id"]),
                ): row
                for row in expected_here
            }
            for task_id, local_index, terminal in missing:
                expected = expected_lookup[
                    (task_id, local_index, str(terminal["attempt_id"]))
                ]
                exact = {
                    "arm": arm,
                    "source_shard_key": shard.key,
                    "source_directory": directory,
                    "task_id": task_id,
                    "local_sample_index": local_index,
                    "global_sample_index": shard.global_indices[local_index],
                    "attempt_id": terminal["attempt_id"],
                    "response_id": terminal["response_id"],
                    "code_sha256": terminal["code_sha256"],
                    "terminal_row_sha256": canonical_sha(terminal),
                }
                if any(expected.get(key) != value for key, value in exact.items()):
                    raise ReconciliationError(
                        "terminal identity/hash differs from pinned orphan"
                    )
                selected = {
                    "schema": SCHEMA,
                    "record_type": "outcome_blind_selection",
                    "contract_sha256": EXPECTED_CONTRACT_SHA256,
                    **exact,
                    "source_config_sha256": config_sha,
                    "source_prompt_sha256": terminal["prompt_sha256"],
                    "candidate_valid": terminal["candidate_valid"],
                    "finish_reason": terminal["finish_reason"],
                    "terminal_reason": terminal["terminal_reason"],
                    "selection_uses_pass_or_compile": False,
                }
                selected["selection_id"] = selection_id(selected)
                observed.append(
                    {
                        "selection": selected,
                        "terminal": terminal,
                        "tests": tests[task_id],
                        "source_root": root,
                        "evaluator_key": evaluator_key,
                    }
                )
            source_files = [
                *SOURCE_SNAPSHOT_FILES,
                shard.copied_contract,
            ]
            for optional in ("failure.json", "summary.json", "manifest.json"):
                if (root / optional).is_file():
                    source_files.append(optional)
            snapshots.append(
                {
                    "arm": arm,
                    "source_shard_key": shard.key,
                    "source_directory": directory,
                    "config_sha256": config_sha,
                    "files": {
                        name: file_record(root / name)
                        for name in source_files
                    },
                }
            )
    if source_count != EXPECTED_SHARDS or len(observed) != EXPECTED_ORPHANS:
        raise ReconciliationError("source/orphan count differs from contract")
    observed.sort(
        key=lambda row: (
            row["selection"]["arm"],
            row["selection"]["global_sample_index"],
            row["selection"]["task_id"],
        )
    )
    selection_ids = [
        str(row["selection"]["selection_id"]) for row in observed
    ]
    if len(set(selection_ids)) != EXPECTED_ORPHANS:
        raise ReconciliationError("duplicate selection identity")
    snapshot = {
        "schema": SCHEMA,
        "record_type": "immutable_source_snapshot",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "source_shards": sorted(
            snapshots,
            key=lambda row: (row["source_shard_key"], row["arm"]),
        ),
        "source_shards_count": source_count,
        "selected_orphans": EXPECTED_ORPHANS,
        "source_journals_modified": False,
    }
    snapshot["snapshot_sha256"] = canonical_sha(snapshot)
    return {
        "snapshot": snapshot,
        "observed": observed,
        "evaluators": evaluator_cache,
        "test_bundles": test_bundles,
    }


def _overlay_outcome_key(row: Mapping[str, Any]) -> str:
    return str(row.get("selection_id") or "")


def validate_reconciled_outcome(
    row: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    if (
        row.get("schema") != SCHEMA
        or row.get("record_type") != "reconciled_candidate_outcome"
        or row.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
        or row.get("selection_id") != selection.get("selection_id")
        or row.get("arm") != selection.get("arm")
        or row.get("task_id") != selection.get("task_id")
        or row.get("local_sample_index")
        != selection.get("local_sample_index")
        or row.get("global_sample_index")
        != selection.get("global_sample_index")
        or row.get("attempt_id") != selection.get("attempt_id")
        or row.get("response_id") != selection.get("response_id")
        or row.get("finish_reason") != selection.get("finish_reason")
        or row.get("candidate_valid") != selection.get("candidate_valid")
        or row.get("terminal_reason") != selection.get("terminal_reason")
        or row.get("code_sha256") != selection.get("code_sha256")
        or row.get("evaluator_sha256") != EXPECTED_EVALUATOR_SHA256
        or type(row.get("compiled")) is not bool
        or type(row.get("passed")) is not bool
        or type(row.get("evaluation_performed")) is not bool
        or not isinstance(row.get("stability_runs"), list)
    ):
        raise ReconciliationError("reconciled outcome identity/result mismatch")
    if selection["candidate_valid"]:
        runs = row["stability_runs"]
        runs_valid = all(
            isinstance(run, Mapping)
            and type(run.get("compiled")) is bool
            and type(run.get("passed")) is bool
            and run.get("completion_attestation_id")
            == runner.REQUIRED_ATTESTATION_ID
            and run.get("completion_attestation_required") is True
            and run.get("completion_attestation_satisfied")
            is run.get("passed")
            for run in runs
        )
        all_passed = all(run.get("passed") is True for run in runs)
        if (
            row["evaluation_performed"] is not True
            or len(runs) != 2
            or not runs_valid
            or row.get("completion_attestation_id")
            != runner.REQUIRED_ATTESTATION_ID
            or row.get("completion_attestation_enforced") is not True
            or row["compiled"] is not all(run.get("compiled") is True for run in runs)
            or row["passed"] is not all_passed
            or row.get("completion_attestation_satisfied_all_runs")
            is not all_passed
        ):
            raise ReconciliationError("reconciled evaluator evidence mismatch")
    elif (
        row["evaluation_performed"] is not False
        or row["stability_runs"]
        or row["compiled"] is not False
        or row["passed"] is not False
        or row.get("completion_attestation_enforced") is not False
    ):
        raise ReconciliationError("invalid candidate was not failed closed")


def load_reconciled_outcomes(
    path: Path,
    *,
    selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = _overlay_outcome_key(row)
        if not key or key in outcomes or key not in selections:
            raise ReconciliationError(
                "duplicate/foreign reconciled outcome selection"
            )
        validate_reconciled_outcome(row, selections[key])
        outcomes[key] = row
    return outcomes


def load_effective_outcomes(
    workspace: Path,
    *,
    original_terminals: Mapping[
        tuple[str, str, int], Mapping[str, Any]
    ],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, Any],
]:
    """Load the completed overlay and bind it to current immutable terminals."""
    patch_root = workspace / "frontier_ceiling_patch_v1"
    contract = load_contract(patch_root)
    out = (workspace / DEFAULT_OUTPUT).resolve()
    if runner.sha256_file(out / CONTRACT_NAME) != EXPECTED_CONTRACT_SHA256:
        raise ReconciliationError("completed overlay contract copy mismatch")
    snapshot = read_json(out / "source_snapshot.json")
    claimed_snapshot_sha = str(snapshot.get("snapshot_sha256") or "")
    snapshot_payload = dict(snapshot)
    snapshot_payload.pop("snapshot_sha256", None)
    if (
        snapshot.get("schema") != SCHEMA
        or snapshot.get("source_shards_count") != EXPECTED_SHARDS
        or snapshot.get("selected_orphans") != EXPECTED_ORPHANS
        or claimed_snapshot_sha != canonical_sha(snapshot_payload)
    ):
        raise ReconciliationError("completed overlay snapshot is malformed")
    source_shards = snapshot.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != EXPECTED_SHARDS:
        raise ReconciliationError("completed overlay snapshot shard count mismatch")
    for shard in source_shards:
        files = shard.get("files") if isinstance(shard, Mapping) else None
        if not isinstance(files, Mapping):
            raise ReconciliationError("completed overlay snapshot lacks file map")
        for record in files.values():
            if (
                not isinstance(record, Mapping)
                or runner.sha256_file(
                    Path(str(record.get("path") or "")).resolve()
                )
                != record.get("sha256")
            ):
                raise ReconciliationError(
                    "a source file changed after reconciliation"
                )
    selections_list = read_jsonl(out / "selections.jsonl")
    if len(selections_list) != EXPECTED_ORPHANS:
        raise ReconciliationError("completed overlay selection count mismatch")
    selections: dict[str, dict[str, Any]] = {}
    contract_rows = {
        (
            str(row["arm"]),
            str(row["task_id"]),
            int(row["global_sample_index"]),
            str(row["attempt_id"]),
        ): row
        for row in contract["expected_orphans"]
    }
    for selection in selections_list:
        sid = str(selection.get("selection_id") or "")
        identity = (
            str(selection.get("arm") or ""),
            str(selection.get("task_id") or ""),
            int(selection.get("global_sample_index", -1)),
            str(selection.get("attempt_id") or ""),
        )
        expected = contract_rows.get(identity)
        if (
            not sid
            or sid in selections
            or sid != selection_id(selection)
            or expected is None
            or selection.get("selection_uses_pass_or_compile") is not False
            or any(
                selection.get(key) != expected.get(key)
                for key in (
                    "arm",
                    "source_shard_key",
                    "source_directory",
                    "task_id",
                    "local_sample_index",
                    "global_sample_index",
                    "attempt_id",
                    "response_id",
                    "code_sha256",
                    "terminal_row_sha256",
                )
            )
        ):
            raise ReconciliationError("completed overlay selection mismatch")
        selections[sid] = selection
    outcomes = load_reconciled_outcomes(
        out / "reconciled_outcomes.jsonl",
        selections=selections,
    )
    if set(outcomes) != set(selections):
        raise ReconciliationError("completed overlay lacks a selected outcome")
    status = read_json(out / "status.json")
    provenance = read_json(out / "provenance.json")
    if (
        status.get("schema") != SCHEMA
        or status.get("status") != "complete"
        or status.get("provider_calls") != 0
        or status.get("source_journals_modified") is not False
        or status.get("source_snapshot_unchanged_after_evaluation") is not True
        or status.get("source_snapshot_sha256") != claimed_snapshot_sha
        or status.get("outcomes_sha256")
        != runner.sha256_file(out / "reconciled_outcomes.jsonl")
        or provenance.get("status") != "complete"
        or provenance.get("provider_calls") != 0
        or provenance.get("source_journals_modified") is not False
    ):
        raise ReconciliationError("completed overlay status/provenance mismatch")
    mapped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for sid, outcome in outcomes.items():
        selection = selections[sid]
        key = (
            str(selection["arm"]),
            str(selection["task_id"]),
            int(selection["global_sample_index"]),
        )
        terminal = original_terminals.get(key)
        if (
            terminal is None
            or canonical_sha(terminal) != selection["terminal_row_sha256"]
            or terminal.get("attempt_id") != selection["attempt_id"]
            or terminal.get("response_id") != selection["response_id"]
            or terminal.get("code_sha256") != selection["code_sha256"]
            or key in mapped
        ):
            raise ReconciliationError(
                "completed overlay is not bound to the original terminal"
            )
        mapped[key] = outcome
    return mapped, {
        "source": "original_outcome_reconciliation_v1",
        "status": "complete",
        "selected_orphans": EXPECTED_ORPHANS,
        "reconciled_outcomes": len(mapped),
        "provider_calls": 0,
        "source_journals_modified": False,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "source_snapshot_sha256": claimed_snapshot_sha,
    }


def preflight(
    workspace: Path,
    out: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_provider_free_source()
    patch_root = workspace / "frontier_ceiling_patch_v1"
    contract = load_contract(patch_root)
    audit = audit_sources(workspace, contract)
    selections = [row["selection"] for row in audit["observed"]]
    out.mkdir(parents=True, exist_ok=True)
    copied_contract = out / CONTRACT_NAME
    if copied_contract.is_file():
        if runner.sha256_file(copied_contract) != EXPECTED_CONTRACT_SHA256:
            raise ReconciliationError("copied overlay contract changed")
    else:
        copied_contract.write_bytes((patch_root / CONTRACT_NAME).read_bytes())
    snapshot_path = out / "source_snapshot.json"
    selections_path = out / "selections.jsonl"
    if snapshot_path.is_file():
        if read_json(snapshot_path) != audit["snapshot"]:
            raise ReconciliationError("immutable source snapshot changed")
    else:
        runner.atomic_write_json(snapshot_path, audit["snapshot"])
    if selections_path.is_file():
        if read_jsonl(selections_path) != selections:
            raise ReconciliationError("outcome-blind selections changed")
    else:
        runner.atomic_write_jsonl(selections_path, selections)
    provenance = {
        "schema": SCHEMA,
        "status": "preflight_complete",
        "contract": file_record(copied_contract),
        "source_snapshot": file_record(snapshot_path),
        "selections": file_record(selections_path),
        "selection_count": len(selections),
        "selection_outcome_blind": True,
        "source_journals_modified": False,
        "provider_imports": False,
        "provider_clients": False,
        "provider_calls": 0,
        "api_credentials_read": False,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "dart_sha256": EXPECTED_DART_SHA256,
    }
    provenance["provenance_payload_sha256"] = canonical_sha(provenance)
    existing_path = out / "provenance.json"
    if existing_path.is_file():
        existing = read_json(existing_path)
        for key, value in provenance.items():
            if key == "status":
                continue
            if existing.get(key) != value:
                raise ReconciliationError("overlay provenance changed")
    else:
        runner.atomic_write_json(existing_path, provenance)
    return audit, provenance


def reconcile(workspace: Path, out: Path) -> dict[str, Any]:
    with runner.RunLock(out / ".reconcile.lock"):
        before, provenance = preflight(workspace, out)
        selection_by_id = {
            str(row["selection"]["selection_id"]): row["selection"]
            for row in before["observed"]
        }
        existing = load_reconciled_outcomes(
            out / "reconciled_outcomes.jsonl",
            selections=selection_by_id,
        )
        journal = runner.JsonlJournal(out / "reconciled_outcomes.jsonl")
        for item in before["observed"]:
            selection = item["selection"]
            sid = str(selection["selection_id"])
            if sid in existing:
                continue
            terminal = item["terminal"]
            if selection["candidate_valid"]:
                module, evaluator_record = before["evaluators"][
                    item["evaluator_key"]
                ]
                evaluation = runner.evaluate_candidate_stably(
                    module.evaluate_dart_jit_tests_detail,
                    code=str(terminal["code"]),
                    tests=str(item["tests"]),
                    task_id=str(selection["task_id"]),
                    sample_index=int(selection["global_sample_index"]),
                    stability_runs=2,
                    timeout=30,
                )
                evaluation_performed = True
            else:
                evaluator_record = {
                    "sha256": EXPECTED_EVALUATOR_SHA256,
                    "entrypoint": "evaluate_dart_jit_tests_detail",
                }
                evaluation = {
                    "compiled": False,
                    "passed": False,
                    "completion_attestation_id": (
                        runner.REQUIRED_ATTESTATION_ID
                    ),
                    "completion_attestation_enforced": False,
                    "completion_attestation_satisfied_all_runs": False,
                    "stability_runs": [],
                }
                evaluation_performed = False
            outcome = {
                "schema": SCHEMA,
                "record_type": "reconciled_candidate_outcome",
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
                "selection_id": sid,
                "source_terminal_row_sha256": selection[
                    "terminal_row_sha256"
                ],
                "arm": selection["arm"],
                "source_shard_key": selection["source_shard_key"],
                "source_directory": selection["source_directory"],
                "task_id": selection["task_id"],
                "local_sample_index": selection["local_sample_index"],
                "global_sample_index": selection["global_sample_index"],
                "attempt_id": selection["attempt_id"],
                "response_id": selection["response_id"],
                "finish_reason": selection["finish_reason"],
                "candidate_valid": selection["candidate_valid"],
                "terminal_reason": selection["terminal_reason"],
                "code_sha256": selection["code_sha256"],
                "evaluator_sha256": evaluator_record["sha256"],
                "evaluator_entrypoint": evaluator_record["entrypoint"],
                "evaluation_performed": evaluation_performed,
                "completion_attestation_id": evaluation[
                    "completion_attestation_id"
                ],
                "completion_attestation_enforced": evaluation[
                    "completion_attestation_enforced"
                ],
                "completion_attestation_satisfied_all_runs": evaluation[
                    "completion_attestation_satisfied_all_runs"
                ],
                "compiled": evaluation["compiled"],
                "passed": evaluation["passed"],
                "stability_runs": evaluation["stability_runs"],
                "evaluated_at": runner.utc_now(),
                "provider_calls": 0,
            }
            validate_reconciled_outcome(outcome, selection)
            journal.append(outcome)
            existing[sid] = outcome
        after = audit_sources(workspace, load_contract(
            workspace / "frontier_ceiling_patch_v1"
        ))
        if after["snapshot"] != before["snapshot"]:
            raise ReconciliationError(
                "source artifacts changed during local reconciliation"
            )
        completed = load_reconciled_outcomes(
            out / "reconciled_outcomes.jsonl",
            selections=selection_by_id,
        )
        if set(completed) != set(selection_by_id):
            raise ReconciliationError("not every selected orphan was reconciled")
        status = {
            "schema": SCHEMA,
            "status": "complete",
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "selected_orphans": EXPECTED_ORPHANS,
            "reconciled_outcomes": len(completed),
            "source_snapshot_sha256": before["snapshot"]["snapshot_sha256"],
            "source_snapshot_unchanged_after_evaluation": True,
            "selection_outcome_blind": True,
            "source_journals_modified": False,
            "provider_imports": False,
            "provider_clients": False,
            "provider_calls": 0,
            "api_credentials_read": False,
            "compiled": sum(row["compiled"] is True for row in completed.values()),
            "passed": sum(row["passed"] is True for row in completed.values()),
            "outcomes_sha256": runner.sha256_file(
                out / "reconciled_outcomes.jsonl"
            ),
        }
        runner.atomic_write_json(out / "status.json", status)
        final_provenance = dict(provenance)
        final_provenance["status"] = "complete"
        final_provenance["reconciled_outcomes"] = file_record(
            out / "reconciled_outcomes.jsonl"
        )
        final_provenance["status_record"] = file_record(out / "status.json")
        final_provenance["provider_calls"] = 0
        runner.atomic_write_json(out / "provenance.json", final_provenance)
        return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--out", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--reconcile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    out = (
        args.out.expanduser().resolve()
        if args.out
        else (workspace / DEFAULT_OUTPUT).resolve()
    )
    try:
        if args.preflight_only:
            audit, _provenance = preflight(workspace, out)
            report = {
                "schema": SCHEMA,
                "status": "preflight_complete",
                "selected_orphans": len(audit["observed"]),
                "source_shards": audit["snapshot"]["source_shards_count"],
                "provider_calls": 0,
                "source_journals_modified": False,
            }
        else:
            report = reconcile(workspace, out)
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "provider_calls": 0,
            "source_journals_modified": False,
        }
        print(json.dumps(report, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
