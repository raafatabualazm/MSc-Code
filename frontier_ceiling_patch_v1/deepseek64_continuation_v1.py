#!/usr/bin/env python3
"""Fail-closed 128K -> 64K continuation for the paired DeepSeek frontier run.

The stopped 128K journals are immutable inputs.  A slot is complete if and
only if the source attempt journal contains a validated terminal provider
response.  Outcomes are deliberately not consulted when selecting slots.
Terminal source responses that were stopped before evaluation are evaluated
locally before an API client is constructed.

Paid calls require ``--mode run`` and a separate confirmation string.  The
default mode is read-only preflight.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import frontier_passk as runner


CONTRACT_SCHEMA = "deepseek-64k-continuation-contract-v1"
OVERLAY_SCHEMA = "deepseek-64k-continuation-v1"
DISPATCH_SCHEMA = "deepseek-64k-dispatch-v1"
INDEX_SCHEMA = "deepseek-64k-effective-slot-index-v1"
PAID_CONFIRMATION = "YES_64K_CONTINUATION"
DEFAULT_CONTRACT = Path(__file__).with_name(
    "deepseek64_continuation_contract_v1.json"
)


class ContinuationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawRow:
    value: dict[str, Any]
    raw: bytes


@dataclass
class SourceSnapshot:
    arm: str
    source_root: Path
    out_root: Path
    provenance: dict[str, Any]
    prompts: list[RawRow]
    tasks: list[RawRow]
    prompt_map: dict[str, dict[str, Any]]
    eval_map: dict[str, dict[str, Any]]
    expected_slots: set[tuple[str, int]]
    source_terminal: dict[tuple[str, int], dict[str, Any]]
    source_terminal_raw: dict[tuple[str, int], bytes]
    source_outcomes: dict[tuple[str, int, str], dict[str, Any]]
    source_outcome_raw: dict[tuple[str, int, str], bytes]
    source_attempt_rows: list[RawRow]
    source_outcome_rows: list[RawRow]
    overlay_config_sha256: str
    overlay_slot_policy: dict[str, Any]
    overlay_slot_policy_sha256: str
    arm_contract: dict[str, Any]
    policy: dict[str, Any]
    contract_sha256: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--arm", choices=("opus", "codex"), required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "adopt", "run", "status"),
        default="preflight",
        help=(
            "preflight/status make no API calls; adopt evaluates already-returned "
            "terminal responses only; run permits unresolved-slot provider calls"
        ),
    )
    parser.add_argument(
        "--paid-confirmation",
        default="",
        help="Required exact confirmation for --mode run.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--deepseek-env-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.mode == "run" and args.paid_confirmation != PAID_CONFIRMATION:
        parser.error(
            "--mode run requires --paid-confirmation "
            f"{PAID_CONFIRMATION!r}"
        )
    if args.mode != "run" and args.paid_confirmation:
        parser.error("--paid-confirmation is accepted only with --mode run")
    return args


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContinuationError(f"cannot read contract {resolved}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != CONTRACT_SCHEMA:
        raise ContinuationError("continuation contract has an incompatible schema")
    if not isinstance(value.get("arms"), dict) or not isinstance(
        value.get("policy"), dict
    ):
        raise ContinuationError("continuation contract is incomplete")
    return value, runner.sha256_file(resolved)


def resolve_under(workspace: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return candidate.resolve()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ContinuationError(f"missing {label}: {path}")
    actual = runner.sha256_file(path)
    if actual != str(expected).strip().lower():
        raise ContinuationError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_raw_jsonl(path: Path, label: str, *, allow_empty: bool = False) -> list[RawRow]:
    try:
        payload = path.read_bytes()
    except Exception as exc:
        raise ContinuationError(f"cannot read {label} {path}: {exc}") from exc
    if not payload:
        if allow_empty:
            return []
        raise ContinuationError(f"{label} is empty")
    if not payload.endswith(b"\n"):
        raise ContinuationError(f"{label} lacks a final newline")
    rows: list[RawRow] = []
    for line_number, raw in enumerate(payload.splitlines(keepends=True), 1):
        if not raw.strip():
            raise ContinuationError(f"{label} has a blank line at {line_number}")
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise ContinuationError(
                f"cannot parse {label} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ContinuationError(
                f"{label} line {line_number} is not an object"
            )
        rows.append(RawRow(value=value, raw=raw))
    return rows


def raw_map(
    rows: Iterable[RawRow],
    key_fn: Any,
    label: str,
) -> dict[Any, bytes]:
    result: dict[Any, bytes] = {}
    for row in rows:
        key = key_fn(row.value)
        if key in result:
            raise ContinuationError(f"{label} contains duplicate key {key!r}")
        result[key] = row.raw
    return result


def atomic_write_raw(path: Path, lines: Iterable[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("wb") as handle:
        for raw in lines:
            if not raw.endswith(b"\n"):
                raise ContinuationError("raw JSONL row lacks a final newline")
            handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def overlay_policy(
    source_provenance: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    source_slot = dict((source_provenance.get("config") or {}).get("slot_policy") or {})
    required = {
        "schema": runner.FIXED_SLOT_POLICY_SCHEMA,
        "requested_model": policy["requested_model"],
        "k": policy["k"],
        "fixed_max_output_tokens": policy["source_max_output_tokens"],
        "max_prompt_tokens": policy["max_prompt_tokens"],
        "temperature": policy["temperature"],
        "top_p": policy["top_p"],
        "request_timeout_seconds": policy["timeout_seconds"],
        "max_transport_attempts_per_slot": policy["max_attempts_per_slot"],
    }
    for key, expected in required.items():
        if source_slot.get(key) != expected:
            raise ContinuationError(
                f"source slot policy {key!r} mismatch: "
                f"{source_slot.get(key)!r} != {expected!r}"
            )
    source_slot["fixed_max_output_tokens"] = policy[
        "continuation_max_output_tokens"
    ]
    source_slot["continuation_source_max_output_tokens"] = policy[
        "source_max_output_tokens"
    ]
    source_slot["continuation_selection"] = "source_terminal_presence_only"
    return source_slot, runner.stable_sha256(source_slot)


def make_overlay_config(
    *,
    arm: str,
    arm_contract: Mapping[str, Any],
    policy: Mapping[str, Any],
    contract_sha256: str,
    slot_policy: Mapping[str, Any],
    slot_policy_sha256: str,
) -> tuple[dict[str, Any], str]:
    value = {
        "schema": OVERLAY_SCHEMA,
        "arm": arm,
        "pair_arm_key": arm_contract["pair_arm_key"],
        "dataset_label": arm_contract["dataset_label"],
        "source_run": arm_contract["source_run"],
        "source_config_sha256": arm_contract["source_config_sha256"],
        "source_files": arm_contract["files"],
        "contract_sha256": contract_sha256,
        "provider": policy["provider"],
        "requested_model": policy["requested_model"],
        "k": policy["k"],
        "workers": policy["workers"],
        "source_max_output_tokens": policy["source_max_output_tokens"],
        "max_output_tokens": policy["continuation_max_output_tokens"],
        "max_prompt_tokens": policy["max_prompt_tokens"],
        "temperature": policy["temperature"],
        "top_p": policy["top_p"],
        "timeout_seconds": policy["timeout_seconds"],
        "max_attempts_per_slot": policy["max_attempts_per_slot"],
        "retry_base_seconds": policy["retry_base_seconds"],
        "retry_max_seconds": policy["retry_max_seconds"],
        "eval_timeout_seconds": policy["eval_timeout_seconds"],
        "eval_stability_runs": policy["eval_stability_runs"],
        "evaluator_sha256": policy["evaluator_sha256"],
        "dart_sha256": policy["dart_sha256"],
        "slot_policy": slot_policy,
        "slot_policy_sha256": slot_policy_sha256,
    }
    return value, runner.stable_sha256(value)


def request_args(
    snapshot: SourceSnapshot,
    deepseek_env_file: Path,
    workers: int,
) -> SimpleNamespace:
    p = snapshot.policy
    provenance_config = snapshot.provenance["config"]
    return SimpleNamespace(
        provider=p["provider"],
        model=p["requested_model"],
        k=int(p["k"]),
        workers=workers,
        max_output_tokens=int(p["continuation_max_output_tokens"]),
        max_prompt_tokens=int(p["max_prompt_tokens"]),
        temperature=float(p["temperature"]),
        top_p=float(p["top_p"]),
        timeout_seconds=int(p["timeout_seconds"]),
        max_attempts_per_sample=int(p["max_attempts_per_slot"]),
        retry_base_seconds=float(p["retry_base_seconds"]),
        retry_max_seconds=float(p["retry_max_seconds"]),
        eval_timeout_seconds=int(p["eval_timeout_seconds"]),
        eval_stability_runs=int(p["eval_stability_runs"]),
        extra_body=dict(provenance_config.get("extra_body") or {}),
        deepseek_env_file=deepseek_env_file,
        api_key="",
        base_url="",
    )


def validate_source(
    *,
    workspace: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    arm: str,
    out_override: Path | None,
) -> SourceSnapshot:
    policy = dict(contract["policy"])
    arm_contract = dict(contract["arms"][arm])
    runner_spec = contract["runner"]
    require_hash(
        resolve_under(workspace, runner_spec["path"]),
        runner_spec["sha256"],
        "pinned frontier runner",
    )
    require_hash(
        resolve_under(workspace, runner_spec["core_path"]),
        runner_spec["core_sha256"],
        "pinned frontier core",
    )
    run_root = resolve_under(workspace, contract["run_root"])
    source_root = (run_root / arm_contract["source_run"]).resolve()
    out_root = (
        out_override.expanduser().resolve()
        if out_override is not None
        else (run_root / arm_contract["continuation_run"]).resolve()
    )
    if source_root == out_root:
        raise ContinuationError("continuation output must differ from source root")
    for name, digest in arm_contract["files"].items():
        require_hash(source_root / name, digest, f"source {arm} {name}")

    provenance = json.loads(
        (source_root / "provenance.json").read_text(encoding="utf-8")
    )
    if provenance.get("schema") != runner.RUN_SCHEMA_VERSION:
        raise ContinuationError("source provenance schema is incompatible")
    if provenance.get("config_sha256") != arm_contract["source_config_sha256"]:
        raise ContinuationError("source config fingerprint mismatch")
    if provenance.get("task_set_sha256") != arm_contract["task_set_sha256"]:
        raise ContinuationError("source task-set fingerprint mismatch")
    config = provenance.get("config") or {}
    exact_config = {
        "provider": policy["provider"],
        "model_requested": policy["requested_model"],
        "k": policy["k"],
        "max_output_tokens": policy["source_max_output_tokens"],
        "max_prompt_tokens": policy["max_prompt_tokens"],
        "temperature": policy["temperature"],
        "top_p": policy["top_p"],
        "timeout_seconds": policy["timeout_seconds"],
        "max_attempts_per_sample": policy["max_attempts_per_slot"],
        "eval_timeout_seconds": policy["eval_timeout_seconds"],
        "eval_stability_runs": policy["eval_stability_runs"],
        "expected_task_count": policy["expected_task_count"],
        "pair_arm_key": arm_contract["pair_arm_key"],
        "dataset_label": arm_contract["dataset_label"],
    }
    for key, expected in exact_config.items():
        if config.get(key) != expected:
            raise ContinuationError(
                f"source config {key!r} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    if provenance.get("evaluator", {}).get("sha256") != policy["evaluator_sha256"]:
        raise ContinuationError("source evaluator identity mismatch")
    if (
        provenance.get("evaluator", {}).get("dart_binary", {}).get("sha256")
        != policy["dart_sha256"]
    ):
        raise ContinuationError("source Dart identity mismatch")

    evaluator_path = Path(provenance["evaluator"]["path"]).resolve()
    dart_path = Path(provenance["evaluator"]["dart_binary"]["path"]).resolve()
    require_hash(evaluator_path, policy["evaluator_sha256"], "evaluator")
    require_hash(dart_path, policy["dart_sha256"], "Dart binary")
    dataset_record = provenance["dataset"]
    eval_path = Path(dataset_record["path"]).resolve()
    require_hash(eval_path, dataset_record["sha256"], "source evaluator dataset")

    tasks = load_raw_jsonl(source_root / "tasks.jsonl", "source tasks")
    prompts = load_raw_jsonl(source_root / "prompts.jsonl", "source prompts")
    eval_rows = runner.load_jsonl(eval_path, "source evaluator dataset")
    expected_count = int(policy["expected_task_count"])
    if len(tasks) != expected_count or len(prompts) != expected_count:
        raise ContinuationError("source task/prompt row count mismatch")
    if len(eval_rows) != expected_count:
        raise ContinuationError("source evaluator row count mismatch")
    task_ids = [str(row.value.get("task_id") or "") for row in tasks]
    prompt_ids = [str(row.value.get("task_id") or "") for row in prompts]
    eval_ids = [str(row.get("task_id") or "") for row in eval_rows]
    if (
        not all(task_ids)
        or len(set(task_ids)) != expected_count
        or task_ids != prompt_ids
        or task_ids != eval_ids
    ):
        raise ContinuationError("source task/prompt/evaluator order or identity differs")
    prompt_map = {str(row.value["task_id"]): row.value for row in prompts}
    eval_map = {str(row["task_id"]): row for row in eval_rows}
    for task in tasks:
        task_id = str(task.value["task_id"])
        eval_row = eval_map[task_id]
        tests = eval_row.get("acceptance_tests")
        if not isinstance(tests, str) or not tests:
            raise ContinuationError(f"task {task_id} has no acceptance tests")
        if eval_row.get("tests") != tests:
            raise ContinuationError(f"task {task_id} tests disagree")
        if runner.sha256_text(tests) != task.value.get("acceptance_tests_sha256"):
            raise ContinuationError(f"task {task_id} acceptance-test hash mismatch")
        prompt = prompt_map[task_id]
        if prompt.get("never_truncated") is not True:
            raise ContinuationError(f"task {task_id} prompt is not sealed untruncated")
        if not isinstance(prompt.get("messages"), list):
            raise ContinuationError(f"task {task_id} prompt messages are malformed")

    source_slot_policy = config["slot_policy"]
    source_slot_sha = config["slot_policy_sha256"]
    if runner.stable_sha256(source_slot_policy) != source_slot_sha:
        raise ContinuationError("source slot policy hash mismatch")
    source_attempt_rows = load_raw_jsonl(
        source_root / "attempts.jsonl", "source attempts"
    )
    source_outcome_rows = load_raw_jsonl(
        source_root / "outcomes.jsonl", "source outcomes", allow_empty=True
    )
    budget = runner.TokenBudget(0)
    source_terminal, _next = runner.load_resume_attempts(
        source_root / "attempts.jsonl",
        config_sha=arm_contract["source_config_sha256"],
        prompt_map=prompt_map,
        budget=budget,
        requested_model=policy["requested_model"],
        k=int(policy["k"]),
        max_prompt_tokens=int(policy["max_prompt_tokens"]),
        requested_max_tokens=int(policy["source_max_output_tokens"]),
        max_transport_attempts_per_slot=int(policy["max_attempts_per_slot"]),
        slot_policy_sha256=source_slot_sha,
    )
    source_outcomes = runner.load_resume_outcomes(
        source_root / "outcomes.jsonl",
        config_sha=arm_contract["source_config_sha256"],
        evaluator_sha256=policy["evaluator_sha256"],
    )
    terminal_attempt_keys = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in source_terminal.items()
    }
    if not set(source_outcomes).issubset(terminal_attempt_keys):
        raise ContinuationError("source outcomes contain an orphan record")
    for outcome_key, outcome in source_outcomes.items():
        slot = outcome_key[:2]
        terminal = source_terminal[slot]
        receipt = {
            "attempt_id": terminal["attempt_id"],
            "response_id": terminal["response_id"],
            "finish_reason": terminal["finish_reason"],
            "candidate_valid": terminal["candidate_valid"],
            "terminal_reason": terminal["terminal_reason"],
            "code_sha256": terminal["code_sha256"],
        }
        for field, expected in receipt.items():
            if outcome.get(field) != expected:
                raise ContinuationError(
                    f"source outcome receipt mismatch for {outcome_key}: {field}"
                )
    expected_slots = {
        (task_id, sample_index)
        for task_id in task_ids
        for sample_index in range(int(policy["k"]))
    }
    if not set(source_terminal).issubset(expected_slots):
        raise ContinuationError("source terminals contain a foreign slot")
    counts = arm_contract["expected_snapshot_counts"]
    observed = {
        "terminal_slots": len(source_terminal),
        "source_outcomes": len(source_outcomes),
        "terminal_without_outcome": len(source_terminal) - len(source_outcomes),
        "missing_slots": len(expected_slots) - len(source_terminal),
    }
    if observed != counts:
        raise ContinuationError(
            f"source snapshot counts changed: expected {counts}, got {observed}"
        )
    source_attempt_raw_by_id = raw_map(
        source_attempt_rows,
        lambda row: str(row.get("attempt_id") or ""),
        "source attempts",
    )
    source_terminal_raw = {
        key: source_attempt_raw_by_id[str(value["attempt_id"])]
        for key, value in source_terminal.items()
    }
    source_outcome_raw = raw_map(
        source_outcome_rows,
        lambda row: (
            str(row.get("task_id") or ""),
            int(row.get("sample_index", -1)),
            str(row.get("attempt_id") or ""),
        ),
        "source outcomes",
    )
    slot_policy, slot_policy_sha = overlay_policy(provenance, policy)
    _overlay_config, overlay_config_sha = make_overlay_config(
        arm=arm,
        arm_contract=arm_contract,
        policy=policy,
        contract_sha256=contract_sha256,
        slot_policy=slot_policy,
        slot_policy_sha256=slot_policy_sha,
    )
    return SourceSnapshot(
        arm=arm,
        source_root=source_root,
        out_root=out_root,
        provenance=provenance,
        prompts=prompts,
        tasks=tasks,
        prompt_map=prompt_map,
        eval_map=eval_map,
        expected_slots=expected_slots,
        source_terminal=source_terminal,
        source_terminal_raw=source_terminal_raw,
        source_outcomes=source_outcomes,
        source_outcome_raw=source_outcome_raw,
        source_attempt_rows=source_attempt_rows,
        source_outcome_rows=source_outcome_rows,
        overlay_config_sha256=overlay_config_sha,
        overlay_slot_policy=slot_policy,
        overlay_slot_policy_sha256=slot_policy_sha,
        arm_contract=arm_contract,
        policy=policy,
        contract_sha256=contract_sha256,
    )


def overlay_attempt_path(snapshot: SourceSnapshot) -> Path:
    return snapshot.out_root / "attempts64.jsonl"


def overlay_outcome_path(snapshot: SourceSnapshot) -> Path:
    return snapshot.out_root / "outcomes64.jsonl"


def adopted_outcome_path(snapshot: SourceSnapshot) -> Path:
    return snapshot.out_root / "adopted_source_outcomes.jsonl"


def load_overlay_state(
    snapshot: SourceSnapshot,
    args: SimpleNamespace,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], int],
    dict[tuple[str, int, str], dict[str, Any]],
]:
    budget = runner.TokenBudget(0)
    terminals, next_attempt = runner.load_resume_attempts(
        overlay_attempt_path(snapshot),
        config_sha=snapshot.overlay_config_sha256,
        prompt_map=snapshot.prompt_map,
        budget=budget,
        requested_model=args.model,
        k=args.k,
        max_prompt_tokens=args.max_prompt_tokens,
        requested_max_tokens=args.max_output_tokens,
        max_transport_attempts_per_slot=args.max_attempts_per_sample,
        slot_policy_sha256=snapshot.overlay_slot_policy_sha256,
    )
    if set(terminals).intersection(snapshot.source_terminal):
        overlap = sorted(set(terminals).intersection(snapshot.source_terminal))
        raise ContinuationError(
            f"64K overlay contains source-complete slot {overlap[0]}"
        )
    outcomes = runner.load_resume_outcomes(
        overlay_outcome_path(snapshot),
        config_sha=snapshot.overlay_config_sha256,
        evaluator_sha256=snapshot.policy["evaluator_sha256"],
    )
    expected = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in terminals.items()
    }
    if not set(outcomes).issubset(expected):
        raise ContinuationError("64K outcomes contain an orphan record")
    for outcome_key, outcome in outcomes.items():
        terminal = terminals[outcome_key[:2]]
        for field in (
            "attempt_id",
            "response_id",
            "finish_reason",
            "candidate_valid",
            "terminal_reason",
            "code_sha256",
        ):
            if outcome.get(field) != terminal.get(field):
                raise ContinuationError(
                    f"64K outcome receipt mismatch for {outcome_key}: {field}"
                )
    return terminals, next_attempt, outcomes


def load_adopted_outcomes(
    snapshot: SourceSnapshot,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    path = adopted_outcome_path(snapshot)
    outcomes = runner.load_resume_outcomes(
        path,
        config_sha=snapshot.arm_contract["source_config_sha256"],
        evaluator_sha256=snapshot.policy["evaluator_sha256"],
    )
    original = set(snapshot.source_outcomes)
    for key, value in outcomes.items():
        slot = key[:2]
        terminal = snapshot.source_terminal.get(slot)
        if terminal is None:
            raise ContinuationError("adopted outcome refers to a nonterminal source slot")
        expected_key = (slot[0], slot[1], str(terminal["attempt_id"]))
        if key != expected_key or key in original:
            raise ContinuationError("adopted outcome is not an exact missing source outcome")
        for field in (
            "response_id",
            "finish_reason",
            "candidate_valid",
            "terminal_reason",
            "code_sha256",
        ):
            if value.get(field) != terminal.get(field):
                raise ContinuationError(
                    f"adopted outcome receipt mismatch for {key}: {field}"
                )
    return outcomes


def outcome_record(
    *,
    terminal: Mapping[str, Any],
    task_id: str,
    sample_index: int,
    config_sha256: str,
    evaluator_record: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evaluation_performed: bool,
) -> dict[str, Any]:
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "record_type": "candidate_outcome",
        "config_sha256": config_sha256,
        "task_id": task_id,
        "sample_index": sample_index,
        "attempt_id": terminal["attempt_id"],
        "response_id": terminal["response_id"],
        "finish_reason": terminal["finish_reason"],
        "candidate_valid": terminal["candidate_valid"],
        "terminal_reason": terminal["terminal_reason"],
        "code_sha256": terminal["code_sha256"],
        "evaluator_sha256": evaluator_record["sha256"],
        "evaluator_entrypoint": evaluator_record["entrypoint"],
        "evaluation_performed": evaluation_performed,
        "completion_attestation_id": evaluation["completion_attestation_id"],
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
    }


def evaluate_terminal(
    *,
    terminal: Mapping[str, Any],
    task_id: str,
    sample_index: int,
    tests: str,
    config_sha256: str,
    evaluator: Any,
    evaluator_record: Mapping[str, Any],
    args: SimpleNamespace,
) -> dict[str, Any]:
    if terminal["candidate_valid"]:
        evaluation = runner.evaluate_candidate_stably(
            evaluator,
            code=str(terminal["code"]),
            tests=tests,
            task_id=task_id,
            sample_index=sample_index,
            stability_runs=args.eval_stability_runs,
            timeout=args.eval_timeout_seconds,
        )
        performed = True
    else:
        evaluation = {
            "compiled": False,
            "passed": False,
            "completion_attestation_id": runner.REQUIRED_ATTESTATION_ID,
            "completion_attestation_enforced": False,
            "completion_attestation_satisfied_all_runs": False,
            "stability_runs": [],
        }
        performed = False
    return outcome_record(
        terminal=terminal,
        task_id=task_id,
        sample_index=sample_index,
        config_sha256=config_sha256,
        evaluator_record=evaluator_record,
        evaluation=evaluation,
        evaluation_performed=performed,
    )


def import_evaluator(
    snapshot: SourceSnapshot,
    *,
    validate_dart: bool,
) -> tuple[Any, dict[str, Any]]:
    provenance = snapshot.provenance["evaluator"]
    module, record = runner.import_evaluator(
        Path(provenance["path"]),
        snapshot.policy["evaluator_sha256"],
        dart_binary=Path(provenance["dart_binary"]["path"]),
        expected_dart_hash=snapshot.policy["dart_sha256"],
        validate_dart=validate_dart,
    )
    if record["entrypoint"] != provenance["entrypoint"]:
        raise ContinuationError("evaluator entrypoint changed")
    return module.evaluate_dart_jit_tests_detail, record


def reconcile_outcomes(
    snapshot: SourceSnapshot,
    args: SimpleNamespace,
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    dict[tuple[str, int, str], dict[str, Any]],
]:
    evaluator, evaluator_record = import_evaluator(snapshot, validate_dart=True)
    adopted = load_adopted_outcomes(snapshot)
    overlay_terminal, _next, overlay_outcomes = load_overlay_state(snapshot, args)
    adopted_journal = runner.JsonlJournal(adopted_outcome_path(snapshot))
    overlay_journal = runner.JsonlJournal(overlay_outcome_path(snapshot))

    original_source_keys = set(snapshot.source_outcomes)
    for slot in sorted(snapshot.source_terminal):
        terminal = snapshot.source_terminal[slot]
        key = (slot[0], slot[1], str(terminal["attempt_id"]))
        if key in original_source_keys or key in adopted:
            continue
        record = evaluate_terminal(
            terminal=terminal,
            task_id=slot[0],
            sample_index=slot[1],
            tests=snapshot.eval_map[slot[0]]["acceptance_tests"],
            config_sha256=snapshot.arm_contract["source_config_sha256"],
            evaluator=evaluator,
            evaluator_record=evaluator_record,
            args=args,
        )
        adopted_journal.append(record)
        adopted[key] = record

    for slot in sorted(overlay_terminal):
        terminal = overlay_terminal[slot]
        key = (slot[0], slot[1], str(terminal["attempt_id"]))
        if key in overlay_outcomes:
            continue
        record = evaluate_terminal(
            terminal=terminal,
            task_id=slot[0],
            sample_index=slot[1],
            tests=snapshot.eval_map[slot[0]]["acceptance_tests"],
            config_sha256=snapshot.overlay_config_sha256,
            evaluator=evaluator,
            evaluator_record=evaluator_record,
            args=args,
        )
        overlay_journal.append(record)
        overlay_outcomes[key] = record
    return adopted, overlay_outcomes


def load_dispatches(
    path: Path,
    *,
    config_sha256: str,
) -> set[str]:
    if not path.is_file():
        return set()
    rows = runner.load_jsonl(path, "64K dispatch journal")
    intents: dict[str, dict[str, Any]] = {}
    settlements: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != DISPATCH_SCHEMA:
            raise ContinuationError("dispatch journal schema mismatch")
        if row.get("config_sha256") != config_sha256:
            raise ContinuationError("dispatch journal config mismatch")
        dispatch_id = str(row.get("dispatch_id") or "")
        if not dispatch_id:
            raise ContinuationError("dispatch journal has an empty dispatch id")
        kind = row.get("record_type")
        target = intents if kind == "dispatch_intent" else settlements
        if kind not in {"dispatch_intent", "dispatch_settlement"}:
            raise ContinuationError("dispatch journal has an invalid record type")
        if dispatch_id in target:
            raise ContinuationError("dispatch journal contains a duplicate record")
        target[dispatch_id] = row
    dangling = sorted(set(intents) - set(settlements))
    if dangling:
        raise ContinuationError(
            "dispatch journal has an unresolved intent; refusing a possible "
            f"duplicate provider call: {dangling[0]}"
        )
    if set(settlements) - set(intents):
        raise ContinuationError("dispatch journal contains an orphan settlement")
    attempt_ids: set[str] = set()
    for dispatch_id, intent in intents.items():
        settlement = settlements[dispatch_id]
        for field in (
            "task_id",
            "sample_index",
            "attempt_index",
            "attempt_id",
        ):
            if settlement.get(field) != intent.get(field):
                raise ContinuationError("dispatch settlement identity mismatch")
        if settlement.get("attempt_recorded") is not True:
            raise ContinuationError("dispatch settlement lacks an attempt receipt")
        attempt_id = str(intent.get("attempt_id") or "")
        if not attempt_id or attempt_id in attempt_ids:
            raise ContinuationError("dispatch journal attempt identity is invalid")
        attempt_ids.add(attempt_id)
    return attempt_ids


def delay_seconds(args: SimpleNamespace, attempt_index: int) -> float:
    base = min(
        args.retry_max_seconds,
        args.retry_base_seconds * (2 ** min(attempt_index, 8)),
    )
    return min(args.retry_max_seconds, base * random.uniform(0.8, 1.2))


def run_provider_calls(
    snapshot: SourceSnapshot,
    args: SimpleNamespace,
) -> None:
    # Reconciliation is intentionally complete before this function imports
    # the SDK or constructs a client.
    overlay_terminal, next_attempt, overlay_outcomes = load_overlay_state(
        snapshot, args
    )
    dispatch_path = snapshot.out_root / "dispatches64.jsonl"
    dispatched_attempt_ids = load_dispatches(
        dispatch_path, config_sha256=snapshot.overlay_config_sha256
    )
    attempt_rows = load_raw_jsonl(
        overlay_attempt_path(snapshot),
        "64K attempts",
        allow_empty=True,
    ) if overlay_attempt_path(snapshot).is_file() else []
    recorded_attempt_ids = {
        str(row.value.get("attempt_id") or "") for row in attempt_rows
    }
    if dispatched_attempt_ids != recorded_attempt_ids:
        raise ContinuationError(
            "dispatch journal and attempt journal do not cover identical attempts"
        )
    missing = sorted(
        snapshot.expected_slots
        - set(snapshot.source_terminal)
        - set(overlay_terminal)
    )
    if not missing:
        return

    try:
        from openai import OpenAI
    except Exception as exc:
        raise ContinuationError("the openai Python package is required") from exc
    key, base_url = runner.api_credentials(args)
    client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
    attempts = runner.JsonlJournal(overlay_attempt_path(snapshot))
    outcomes = runner.JsonlJournal(overlay_outcome_path(snapshot))
    dispatches = runner.JsonlJournal(dispatch_path)
    stop = threading.Event()
    response_lock = threading.Lock()
    response_ids = {
        str(row["response_id"]) for row in snapshot.source_terminal.values()
    }
    response_ids.update(
        str(row["response_id"]) for row in overlay_terminal.values()
    )
    reservation = args.max_prompt_tokens + args.max_output_tokens

    evaluator, evaluator_record = import_evaluator(snapshot, validate_dart=True)

    def one_slot(slot: tuple[str, int]) -> None:
        task_id, sample_index = slot
        first = next_attempt.get(slot, 0)
        terminal_record: dict[str, Any] | None = None
        for attempt_index in range(first, args.max_attempts_per_sample):
            if stop.is_set():
                raise ContinuationError("stopped after another fatal error")
            attempt_id = (
                f"{runner.safe_label(task_id)}.s{sample_index}."
                f"c64.a{attempt_index}.{uuid.uuid4().hex[:10]}"
            )
            dispatch_id = uuid.uuid4().hex
            identity = {
                "schema": DISPATCH_SCHEMA,
                "config_sha256": snapshot.overlay_config_sha256,
                "dispatch_id": dispatch_id,
                "task_id": task_id,
                "sample_index": sample_index,
                "attempt_index": attempt_index,
                "attempt_id": attempt_id,
            }
            dispatches.append(
                {
                    **identity,
                    "record_type": "dispatch_intent",
                    "requested_max_tokens": args.max_output_tokens,
                    "created_at": runner.utc_now(),
                }
            )
            base_record = {
                "schema": runner.RUN_SCHEMA_VERSION,
                "record_type": "api_attempt",
                "attempt_id": attempt_id,
                "config_sha256": snapshot.overlay_config_sha256,
                "task_id": task_id,
                "sample_index": sample_index,
                "attempt_index": attempt_index,
                "prompt_sha256": snapshot.prompt_map[task_id]["prompt_sha256"],
                "requested_model": args.model,
                "requested_max_tokens": args.max_output_tokens,
                "provider": args.provider,
                "slot_policy_sha256": snapshot.overlay_slot_policy_sha256,
                "started_at": runner.utc_now(),
            }
            try:
                response = runner.make_request(
                    client,
                    args,
                    snapshot.prompt_map[task_id]["messages"],
                    requested_max_tokens=args.max_output_tokens,
                )
                raw_response = runner.response_to_dict(response)
                settled = runner.usage_total(raw_response, reservation)
                try:
                    terminal = runner.classify_terminal_provider_response(
                        response,
                        expected_model=args.model,
                        max_prompt_tokens=args.max_prompt_tokens,
                        requested_max_tokens=args.max_output_tokens,
                    )
                except runner.ResponseContractError as exc:
                    record = {
                        **base_record,
                        "finished_at": runner.utc_now(),
                        "response_received": True,
                        "slot_terminal": True,
                        "candidate_valid": False,
                        "terminal_reason": f"fatal_response_contract:{exc}",
                        "transport_retry": False,
                        "transport_error": None,
                        "fatal_response_contract": True,
                        "budget_charge_tokens": settled,
                        "usage": (
                            raw_response.get("usage")
                            if isinstance(raw_response.get("usage"), Mapping)
                            else None
                        ),
                        "response": raw_response,
                    }
                    attempts.append(record)
                    dispatches.append(
                        {
                            **identity,
                            "record_type": "dispatch_settlement",
                            "attempt_recorded": True,
                            "settled_at": runner.utc_now(),
                        }
                    )
                    raise ContinuationError(
                        f"fatal response contract for {slot}: {exc}"
                    ) from exc
                with response_lock:
                    duplicate = terminal.response_id in response_ids
                    if not duplicate:
                        response_ids.add(terminal.response_id)
                record = {
                    **base_record,
                    "finished_at": runner.utc_now(),
                    "response_received": True,
                    "slot_terminal": True,
                    "candidate_valid": terminal.candidate_valid,
                    "terminal_reason": terminal.terminal_reason,
                    "transport_retry": False,
                    "transport_error": None,
                    "fatal_response_contract": False,
                    "response_id": terminal.response_id,
                    "resolved_model": terminal.response_model,
                    "response_created": terminal.response_created,
                    "finish_reason": terminal.finish_reason,
                    "budget_charge_tokens": settled,
                    "usage": terminal.usage,
                    "content": terminal.content,
                    "reasoning_content": terminal.reasoning_content,
                    "code": terminal.code,
                    "code_sha256": terminal.code_sha256,
                    "response": terminal.raw_response,
                }
                attempts.append(record)
                dispatches.append(
                    {
                        **identity,
                        "record_type": "dispatch_settlement",
                        "attempt_recorded": True,
                        "settled_at": runner.utc_now(),
                    }
                )
                if duplicate:
                    raise ContinuationError(
                        f"duplicate terminal response id: {terminal.response_id}"
                    )
                terminal_record = record
                break
            except ContinuationError:
                raise
            except Exception as exc:
                retryable = runner.is_retryable_api_exception(exc)
                record = {
                    **base_record,
                    "finished_at": runner.utc_now(),
                    "response_received": False,
                    "slot_terminal": False,
                    "candidate_valid": None,
                    "terminal_reason": None,
                    "transport_retry": True,
                    "retryable_transport": retryable,
                    "transport_error": (
                        f"api_exception:{type(exc).__name__}:{str(exc)[:1000]}"
                    ),
                    "fatal_response_contract": False,
                    "budget_charge_tokens": reservation,
                    "usage": None,
                    "response": None,
                }
                attempts.append(record)
                dispatches.append(
                    {
                        **identity,
                        "record_type": "dispatch_settlement",
                        "attempt_recorded": True,
                        "settled_at": runner.utc_now(),
                    }
                )
                if not retryable:
                    raise ContinuationError(
                        f"non-retryable API exception for {slot}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            if attempt_index + 1 < args.max_attempts_per_sample:
                if stop.wait(delay_seconds(args, attempt_index)):
                    raise ContinuationError("stopped during retry backoff")
        if terminal_record is None:
            raise ContinuationError(
                f"slot {slot} exhausted response-less transport attempts"
            )
        outcome = evaluate_terminal(
            terminal=terminal_record,
            task_id=task_id,
            sample_index=sample_index,
            tests=snapshot.eval_map[task_id]["acceptance_tests"],
            config_sha256=snapshot.overlay_config_sha256,
            evaluator=evaluator,
            evaluator_record=evaluator_record,
            args=args,
        )
        outcomes.append(outcome)
        overlay_outcomes[(task_id, sample_index, attempt_id)] = outcome

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(one_slot, slot): slot for slot in missing}
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_map), 1
        ):
            slot = future_map[future]
            try:
                future.result()
            except Exception as exc:
                stop.set()
                failures.append(f"{slot}:{type(exc).__name__}:{exc}")
            if completed % 10 == 0 or completed == len(missing):
                print(
                    f"DEEPSEEK64_PROGRESS arm={snapshot.arm} "
                    f"{completed}/{len(missing)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise ContinuationError(
            f"{len(failures)} slot(s) failed; first={failures[0]}"
        )


def build_effective(
    snapshot: SourceSnapshot,
    args: SimpleNamespace,
) -> dict[str, Any]:
    adopted = load_adopted_outcomes(snapshot)
    overlay_terminal, _next, overlay_outcomes = load_overlay_state(snapshot, args)
    effective_terminal = dict(snapshot.source_terminal)
    for key, value in overlay_terminal.items():
        if key in effective_terminal:
            raise ContinuationError("effective terminal collision")
        effective_terminal[key] = value
    effective_outcome: dict[tuple[str, int, str], dict[str, Any]] = dict(
        snapshot.source_outcomes
    )
    for collection in (adopted, overlay_outcomes):
        for key, value in collection.items():
            if key in effective_outcome:
                raise ContinuationError("effective outcome collision")
            effective_outcome[key] = value

    adopted_raw_rows = (
        load_raw_jsonl(
            adopted_outcome_path(snapshot), "adopted outcomes", allow_empty=True
        )
        if adopted_outcome_path(snapshot).is_file()
        else []
    )
    adopted_raw = raw_map(
        adopted_raw_rows,
        lambda row: (
            str(row.get("task_id") or ""),
            int(row.get("sample_index", -1)),
            str(row.get("attempt_id") or ""),
        ),
        "adopted outcomes",
    )
    overlay_attempt_rows = (
        load_raw_jsonl(
            overlay_attempt_path(snapshot), "64K attempts", allow_empty=True
        )
        if overlay_attempt_path(snapshot).is_file()
        else []
    )
    overlay_attempt_raw_by_id = raw_map(
        overlay_attempt_rows,
        lambda row: str(row.get("attempt_id") or ""),
        "64K attempts",
    )
    overlay_outcome_rows = (
        load_raw_jsonl(
            overlay_outcome_path(snapshot), "64K outcomes", allow_empty=True
        )
        if overlay_outcome_path(snapshot).is_file()
        else []
    )
    overlay_outcome_raw = raw_map(
        overlay_outcome_rows,
        lambda row: (
            str(row.get("task_id") or ""),
            int(row.get("sample_index", -1)),
            str(row.get("attempt_id") or ""),
        ),
        "64K outcomes",
    )
    source_attempt_line = {
        str(row.value.get("attempt_id") or ""): index
        for index, row in enumerate(snapshot.source_attempt_rows, 1)
    }
    source_outcome_line = {
        (
            str(row.value.get("task_id") or ""),
            int(row.value.get("sample_index", -1)),
            str(row.value.get("attempt_id") or ""),
        ): index
        for index, row in enumerate(snapshot.source_outcome_rows, 1)
    }
    adopted_outcome_line = {
        (
            str(row.value.get("task_id") or ""),
            int(row.value.get("sample_index", -1)),
            str(row.value.get("attempt_id") or ""),
        ): index
        for index, row in enumerate(adopted_raw_rows, 1)
    }
    overlay_attempt_line = {
        str(row.value.get("attempt_id") or ""): index
        for index, row in enumerate(overlay_attempt_rows, 1)
    }
    overlay_outcome_line = {
        (
            str(row.value.get("task_id") or ""),
            int(row.value.get("sample_index", -1)),
            str(row.value.get("attempt_id") or ""),
        ): index
        for index, row in enumerate(overlay_outcome_rows, 1)
    }

    task_order = {
        str(row.value["task_id"]): index for index, row in enumerate(snapshot.tasks)
    }
    sorted_slots = sorted(
        effective_terminal, key=lambda key: (task_order[key[0]], key[1])
    )
    attempt_lines: list[bytes] = []
    outcome_lines: list[bytes] = []
    index_rows: list[dict[str, Any]] = []
    source_outcome_keys = set(snapshot.source_outcomes)
    adopted_keys = set(adopted)
    for slot in sorted_slots:
        terminal = effective_terminal[slot]
        outcome_key = (slot[0], slot[1], str(terminal["attempt_id"]))
        if slot in snapshot.source_terminal:
            cap = int(snapshot.policy["source_max_output_tokens"])
            stratum = "source_128k"
            attempt_raw = snapshot.source_terminal_raw[slot]
            attempt_origin_file = str(snapshot.source_root / "attempts.jsonl")
            attempt_origin_line = source_attempt_line[str(terminal["attempt_id"])]
            if outcome_key in source_outcome_keys:
                outcome_raw = snapshot.source_outcome_raw[outcome_key]
                outcome_origin = "source_original"
                outcome_origin_file = str(snapshot.source_root / "outcomes.jsonl")
                outcome_origin_line_number = source_outcome_line[outcome_key]
            elif outcome_key in adopted_keys:
                outcome_raw = adopted_raw[outcome_key]
                outcome_origin = "source_adopted_local_evaluation"
                outcome_origin_file = str(adopted_outcome_path(snapshot))
                outcome_origin_line_number = adopted_outcome_line[outcome_key]
            else:
                outcome_raw = None
                outcome_origin = "pending_local_evaluation"
                outcome_origin_file = None
                outcome_origin_line_number = None
        else:
            cap = int(snapshot.policy["continuation_max_output_tokens"])
            stratum = "continuation_64k"
            attempt_raw = overlay_attempt_raw_by_id[str(terminal["attempt_id"])]
            attempt_origin_file = str(overlay_attempt_path(snapshot))
            attempt_origin_line = overlay_attempt_line[str(terminal["attempt_id"])]
            outcome_raw = overlay_outcome_raw.get(outcome_key)
            outcome_origin = (
                "continuation_evaluation"
                if outcome_raw is not None
                else "pending_local_evaluation"
            )
            outcome_origin_file = (
                str(overlay_outcome_path(snapshot))
                if outcome_raw is not None
                else None
            )
            outcome_origin_line_number = (
                overlay_outcome_line[outcome_key]
                if outcome_raw is not None
                else None
            )
        attempt_lines.append(attempt_raw)
        if outcome_raw is not None:
            outcome_lines.append(outcome_raw)
        index_rows.append(
            {
                "schema": INDEX_SCHEMA,
                "task_id": slot[0],
                "sample_index": slot[1],
                "attempt_id": terminal["attempt_id"],
                "response_id": terminal["response_id"],
                "max_output_tokens_stratum": cap,
                "stratum": stratum,
                "attempt_origin_file": attempt_origin_file,
                "attempt_origin_line": attempt_origin_line,
                "attempt_raw_line_sha256": runner.sha256_text(
                    attempt_raw.decode("utf-8")
                ),
                "outcome_origin": outcome_origin,
                "outcome_origin_file": outcome_origin_file,
                "outcome_origin_line": outcome_origin_line_number,
                "outcome_raw_line_sha256": (
                    runner.sha256_text(outcome_raw.decode("utf-8"))
                    if outcome_raw is not None
                    else None
                ),
                "has_outcome": outcome_raw is not None,
            }
        )
    atomic_write_raw(snapshot.out_root / "effective_attempts.jsonl", attempt_lines)
    atomic_write_raw(snapshot.out_root / "effective_outcomes.jsonl", outcome_lines)
    runner.atomic_write_jsonl(snapshot.out_root / "effective_slot_index.jsonl", index_rows)

    counts = {
        "expected_slots": len(snapshot.expected_slots),
        "source_128k_terminals": len(snapshot.source_terminal),
        "continuation_64k_terminals": len(overlay_terminal),
        "effective_terminals": len(effective_terminal),
        "effective_outcomes": len(effective_outcome),
        "missing_terminals": len(snapshot.expected_slots) - len(effective_terminal),
        "terminal_without_outcome": len(effective_terminal) - len(effective_outcome),
    }
    if counts["effective_outcomes"] > counts["effective_terminals"]:
        raise ContinuationError("effective outcomes exceed terminal responses")
    status = {
        "schema": OVERLAY_SCHEMA,
        "status": (
            "complete"
            if counts["effective_terminals"] == len(snapshot.expected_slots)
            and counts["effective_outcomes"] == len(snapshot.expected_slots)
            else "incomplete"
        ),
        "updated_at": runner.utc_now(),
        "arm": snapshot.arm,
        "contract_sha256": snapshot.contract_sha256,
        "overlay_config_sha256": snapshot.overlay_config_sha256,
        "selection_basis": "terminal_presence_only",
        "outcomes_consulted_for_selection": False,
        "counts": counts,
        "partial_pass_at_k_withheld": True,
        "source_journals_modified": False,
        "effective_files": {
            name: runner.file_record(snapshot.out_root / name)
            for name in (
                "effective_attempts.jsonl",
                "effective_outcomes.jsonl",
                "effective_slot_index.jsonl",
            )
        },
    }
    runner.atomic_write_json(snapshot.out_root / "status.json", status)
    if status["status"] == "complete":
        write_summary(snapshot, effective_terminal, effective_outcome, index_rows)
    return status


def write_summary(
    snapshot: SourceSnapshot,
    terminals: Mapping[tuple[str, int], Mapping[str, Any]],
    outcomes: Mapping[tuple[str, int, str], Mapping[str, Any]],
    index_rows: list[dict[str, Any]],
) -> None:
    if set(terminals) != snapshot.expected_slots:
        raise ContinuationError("cannot summarize an incomplete terminal set")
    if len(outcomes) != len(snapshot.expected_slots):
        raise ContinuationError("cannot summarize an incomplete outcome set")
    outcome_by_slot: dict[tuple[str, int], Mapping[str, Any]] = {}
    for key, value in outcomes.items():
        slot = key[:2]
        terminal = terminals.get(slot)
        if terminal is None or key[2] != str(terminal["attempt_id"]):
            raise ContinuationError("summary outcome/terminal identity mismatch")
        if slot in outcome_by_slot:
            raise ContinuationError("summary contains duplicate outcome slot")
        outcome_by_slot[slot] = value

    task_order = [str(row.value["task_id"]) for row in snapshot.tasks]
    passed_tasks = 0
    compiled_tasks = 0
    task_rows: list[dict[str, Any]] = []
    composition: dict[str, dict[str, int]] = {
        "all_128k": {"tasks": 0, "passed": 0, "compiled": 0},
        "mixed_128k_64k": {"tasks": 0, "passed": 0, "compiled": 0},
        "all_64k": {"tasks": 0, "passed": 0, "compiled": 0},
    }
    cap_by_slot = {
        (str(row["task_id"]), int(row["sample_index"])): int(
            row["max_output_tokens_stratum"]
        )
        for row in index_rows
    }
    for task_id in task_order:
        slots = [(task_id, index) for index in range(int(snapshot.policy["k"]))]
        task_outcomes = [outcome_by_slot[slot] for slot in slots]
        passed = any(bool(row["passed"]) for row in task_outcomes)
        compiled = any(bool(row["compiled"]) for row in task_outcomes)
        passed_tasks += int(passed)
        compiled_tasks += int(compiled)
        caps = {cap_by_slot[slot] for slot in slots}
        if caps == {int(snapshot.policy["source_max_output_tokens"])}:
            group = "all_128k"
        elif caps == {int(snapshot.policy["continuation_max_output_tokens"])}:
            group = "all_64k"
        else:
            group = "mixed_128k_64k"
        composition[group]["tasks"] += 1
        composition[group]["passed"] += int(passed)
        composition[group]["compiled"] += int(compiled)
        task_rows.append(
            {
                "task_id": task_id,
                "capacity_composition": group,
                "source_128k_slots": sum(
                    cap_by_slot[slot]
                    == int(snapshot.policy["source_max_output_tokens"])
                    for slot in slots
                ),
                "continuation_64k_slots": sum(
                    cap_by_slot[slot]
                    == int(snapshot.policy["continuation_max_output_tokens"])
                    for slot in slots
                ),
                "compiled_at_10": compiled,
                "passed_at_10": passed,
            }
        )

    capacity: dict[str, dict[str, Any]] = {}
    for label, cap in (
        ("source_128k", int(snapshot.policy["source_max_output_tokens"])),
        ("continuation_64k", int(snapshot.policy["continuation_max_output_tokens"])),
    ):
        selected_slots = [slot for slot, value in cap_by_slot.items() if value == cap]
        selected_outcomes = [outcome_by_slot[slot] for slot in selected_slots]
        attempt_rows = [terminals[slot] for slot in selected_slots]
        total_tokens = sum(
            int((row.get("usage") or {}).get("total_tokens") or 0)
            for row in attempt_rows
        )
        capacity[label] = {
            "max_output_tokens": cap,
            "terminal_slots": len(selected_slots),
            "candidate_valid": sum(
                bool(row["candidate_valid"]) for row in attempt_rows
            ),
            "compiled_candidates": sum(
                bool(row["compiled"]) for row in selected_outcomes
            ),
            "passed_candidates": sum(
                bool(row["passed"]) for row in selected_outcomes
            ),
            "usage_total_tokens": total_tokens,
            "observational_not_randomized": True,
        }
    total_tasks = len(task_order)
    summary = {
        "schema": OVERLAY_SCHEMA,
        "status": "complete",
        "completed_at": runner.utc_now(),
        "arm": snapshot.arm,
        "requested_model": snapshot.policy["requested_model"],
        "k": snapshot.policy["k"],
        "tasks": total_tasks,
        "terminal_responses": len(terminals),
        "outcomes": len(outcomes),
        "pass_at_10": {
            "successes": passed_tasks,
            "total": total_tasks,
            "rate": passed_tasks / total_tasks,
            "wilson_95": runner.wilson_interval(passed_tasks, total_tasks),
        },
        "compile_at_10": {
            "successes": compiled_tasks,
            "total": total_tasks,
            "rate": compiled_tasks / total_tasks,
            "wilson_95": runner.wilson_interval(compiled_tasks, total_tasks),
        },
        "capacity_strata": capacity,
        "task_capacity_composition": composition,
        "task_results": task_rows,
        "selection_basis": "terminal_presence_only",
        "outcomes_consulted_for_selection": False,
        "source_rows_copied_byte_for_byte": True,
        "source_journals_modified": False,
        "contract_sha256": snapshot.contract_sha256,
        "source_file_hashes": snapshot.arm_contract["files"],
        "artifacts": {
            name: runner.file_record(snapshot.out_root / name)
            for name in (
                "effective_attempts.jsonl",
                "effective_outcomes.jsonl",
                "effective_slot_index.jsonl",
            )
        },
    }
    runner.atomic_write_json(snapshot.out_root / "summary.json", summary)


def write_provenance(
    snapshot: SourceSnapshot,
    *,
    mode: str,
) -> None:
    prior_path = snapshot.out_root / "provenance.json"
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if (
            prior.get("schema") != OVERLAY_SCHEMA
            or prior.get("overlay_config_sha256")
            != snapshot.overlay_config_sha256
        ):
            raise ContinuationError("existing overlay provenance is incompatible")
        created_at = prior["created_at"]
    else:
        created_at = runner.utc_now()
    runner.atomic_write_json(
        prior_path,
        {
            "schema": OVERLAY_SCHEMA,
            "status": "running" if mode in {"adopt", "run"} else mode,
            "created_at": created_at,
            "updated_at": runner.utc_now(),
            "mode": mode,
            "arm": snapshot.arm,
            "source_root": str(snapshot.source_root),
            "source_config_sha256": snapshot.arm_contract[
                "source_config_sha256"
            ],
            "source_files": snapshot.arm_contract["files"],
            "source_snapshot_counts": snapshot.arm_contract[
                "expected_snapshot_counts"
            ],
            "source_journals_modified": False,
            "contract_sha256": snapshot.contract_sha256,
            "overlay_config_sha256": snapshot.overlay_config_sha256,
            "overlay_slot_policy": snapshot.overlay_slot_policy,
            "overlay_slot_policy_sha256": snapshot.overlay_slot_policy_sha256,
            "capacity_transition": {
                "source": snapshot.policy["source_max_output_tokens"],
                "continuation": snapshot.policy[
                    "continuation_max_output_tokens"
                ],
            },
            "selection_basis": "terminal_presence_only",
            "outcomes_consulted_for_selection": False,
        },
    )


def verify_source_still_frozen(snapshot: SourceSnapshot) -> None:
    for name, expected in snapshot.arm_contract["files"].items():
        require_hash(
            snapshot.source_root / name,
            expected,
            f"frozen source {snapshot.arm} {name}",
        )


def main(argv: list[str] | None = None) -> int:
    args_cli = parse_args(argv)
    try:
        contract, contract_sha = load_contract(args_cli.contract)
        workspace = args_cli.workspace.expanduser().resolve()
        arm_spec = contract["arms"].get(args_cli.arm)
        if not isinstance(arm_spec, dict):
            raise ContinuationError(f"contract has no arm {args_cli.arm!r}")
        run_root = resolve_under(workspace, contract["run_root"])
        source_root = (run_root / arm_spec["source_run"]).resolve()
        out_root = (
            args_cli.out.expanduser().resolve()
            if args_cli.out is not None
            else (run_root / arm_spec["continuation_run"]).resolve()
        )
        out_root.mkdir(parents=True, exist_ok=True)
        # Holding both advisory locks makes accidental old-run restart and a
        # second continuation process fail before any journal or provider call.
        with runner.RunLock(source_root / ".run.lock"):
            with runner.RunLock(out_root / ".run.lock"):
                snapshot = validate_source(
                    workspace=workspace,
                    contract=contract,
                    contract_sha256=contract_sha,
                    arm=args_cli.arm,
                    out_override=out_root,
                )
                workers = args_cli.workers or int(snapshot.policy["workers"])
                env_file = (
                    args_cli.deepseek_env_file.expanduser().resolve()
                    if args_cli.deepseek_env_file is not None
                    else resolve_under(
                        workspace, snapshot.policy["deepseek_env_file"]
                    )
                )
                req_args = request_args(snapshot, env_file, workers)
                write_provenance(snapshot, mode=args_cli.mode)
                # These validations do not call a provider.
                load_overlay_state(snapshot, req_args)
                load_adopted_outcomes(snapshot)
                verify_source_still_frozen(snapshot)
                if args_cli.mode in {"adopt", "run"}:
                    reconcile_outcomes(snapshot, req_args)
                    # This second full hash check is the paid-call boundary.
                    verify_source_still_frozen(snapshot)
                if args_cli.mode == "run":
                    run_provider_calls(snapshot, req_args)
                    # Reconcile a terminal written just before any interruption.
                    reconcile_outcomes(snapshot, req_args)
                status = build_effective(snapshot, req_args)
                verify_source_still_frozen(snapshot)
                print(
                    f"DEEPSEEK64_{args_cli.mode.upper()}_OK arm={args_cli.arm} "
                    f"terminal={status['counts']['effective_terminals']}/"
                    f"{status['counts']['expected_slots']} "
                    f"outcomes={status['counts']['effective_outcomes']} "
                    f"missing={status['counts']['missing_terminals']} "
                    f"out={snapshot.out_root}",
                    flush=True,
                )
                return 0
    except Exception as exc:
        failure = {
            "schema": OVERLAY_SCHEMA,
            "status": "failed_closed",
            "failed_at": runner.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            if "out_root" in locals():
                runner.atomic_write_json(out_root / "failure.json", failure)
        except Exception:
            pass
        print(
            f"DEEPSEEK64_FAILED_CLOSED error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
