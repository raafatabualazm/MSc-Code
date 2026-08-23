#!/usr/bin/env python3
"""Recover only dispatches stranded by the sealed DeepSeek64 Codex OOM.

This program is deliberately provider-free.  It converts each exact,
contract-listed dangling dispatch into a conservative response-less transport
attempt charged at the full prompt+completion reservation, then appends the
matching dispatch settlement.  The original request may have reached the
provider, so every receipt records both the possible charge and duplicate-call
risk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import deepseek64_continuation_v1 as continuation
import frontier_passk as runner


SCHEMA = "deepseek64-oom-recovery-contract-v1"
AUDIT_SCHEMA = "deepseek64-oom-recovery-audit-v1"
CONFIRMATION = "YES_RECOVER_CODEX_OOM_UNKNOWN_RESPONSES"
RECOVERY_REASON = "process_oom_unknown_response"
CHARGE_STATUS = "unknown_may_have_been_billed"
DUPLICATE_WARNING = (
    "The pre-OOM request may have reached the provider and may have been "
    "billed; a later retry may duplicate that generation and charge."
)
DEFAULT_CONTRACT = Path(__file__).with_name(
    "deepseek64_oom_recovery_contract_v1.json"
)


class RecoveryError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--mode",
        choices=("preflight", "recover"),
        default="preflight",
    )
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--journal-evidence",
        type=Path,
        default=None,
        help=(
            "Optional captured journal text for offline verification. "
            "When omitted, the fixed contract time window is read with "
            "journalctl."
        ),
    )
    args = parser.parse_args(argv)
    if args.mode == "recover" and args.confirmation != CONFIRMATION:
        parser.error(
            f"--mode recover requires --confirmation {CONFIRMATION!r}"
        )
    if args.mode != "recover" and args.confirmation:
        parser.error("--confirmation is accepted only with --mode recover")
    return args


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    try:
        payload = resolved.read_bytes()
        value = json.loads(payload)
    except Exception as exc:
        raise RecoveryError(f"cannot read recovery contract: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise RecoveryError("recovery contract schema mismatch")
    if value.get("arm") != "codex":
        raise RecoveryError("recovery is sealed for the Codex arm only")
    return value, sha256_bytes(payload)


def resolve_under(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def read_journal_evidence(
    evidence: Mapping[str, Any],
    supplied_path: Path | None,
) -> str:
    if supplied_path is not None:
        try:
            return supplied_path.expanduser().resolve().read_text(
                encoding="utf-8"
            )
        except Exception as exc:
            raise RecoveryError(f"cannot read journal evidence: {exc}") from exc
    command = [
        "journalctl",
        "--since",
        str(evidence["since"]),
        "--until",
        str(evidence["until"]),
        "--no-pager",
        "-o",
        "short-iso-precise",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        raise RecoveryError(f"cannot read OOM journal evidence: {exc}") from exc
    return completed.stdout


def default_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def verify_oom_evidence(
    contract: Mapping[str, Any],
    *,
    workspace: Path,
    journal_text: str,
    process_alive: Callable[[int], bool],
    hostname: str,
) -> dict[str, Any]:
    owner = contract["dead_owner"]
    pid = int(owner["pid"])
    if hostname != owner["host"]:
        raise RecoveryError(
            f"host mismatch: recovery belongs to {owner['host']!r}"
        )
    if process_alive(pid):
        raise RecoveryError(f"refusing recovery while owner PID {pid} is alive")

    evidence = contract["oom_evidence"]
    missing = [
        text
        for text in evidence["required_journal_substrings"]
        if text not in journal_text
    ]
    if missing:
        raise RecoveryError(
            f"OOM journal evidence is incomplete; missing {missing[0]!r}"
        )

    archive = resolve_under(workspace, evidence["archived_lock_dir"])
    expected_names = set(evidence["archived_locks"])
    observed_names = {
        path.name for path in archive.iterdir() if path.is_file()
    } if archive.is_dir() else set()
    if observed_names != expected_names:
        raise RecoveryError(
            "archived lock set differs from the sealed OOM evidence"
        )
    lock_records: dict[str, dict[str, Any]] = {}
    for name, expected_sha in evidence["archived_locks"].items():
        path = archive / name
        payload = path.read_bytes()
        if sha256_bytes(payload) != expected_sha:
            raise RecoveryError(f"archived lock hash mismatch: {name}")
        try:
            lock = json.loads(payload)
        except Exception as exc:
            raise RecoveryError(f"archived lock is malformed: {name}") from exc
        if lock.get("pid") != pid or lock.get("host") != hostname:
            raise RecoveryError(f"archived lock owner mismatch: {name}")
        lock_records[name] = lock

    for value in evidence["live_locks_must_be_absent"]:
        if resolve_under(workspace, value).exists():
            raise RecoveryError(f"live lock still exists: {value}")
    return {
        "dead_pid": pid,
        "host": hostname,
        "journal_sha256": sha256_bytes(journal_text.encode("utf-8")),
        "archived_locks": lock_records,
    }


def _raw_rows(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    if not path.is_file():
        return b"", []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise RecoveryError(f"journal lacks final newline: {path}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(payload.splitlines(), 1):
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise RecoveryError(
                f"malformed JSONL row {number}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise RecoveryError(f"non-object JSONL row {number}: {path}")
        rows.append(value)
    return payload, rows


def _verify_prefix(
    payload: bytes,
    rows: list[dict[str, Any]],
    spec: Mapping[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    line_count = int(spec["lines"])
    if len(rows) < line_count:
        raise RecoveryError(f"{label} lost sealed baseline rows")
    raw_lines = payload.splitlines(keepends=True)
    prefix = b"".join(raw_lines[:line_count])
    if len(prefix) != int(spec["bytes"]):
        raise RecoveryError(f"{label} baseline byte count changed")
    if sha256_bytes(prefix) != spec["sha256"]:
        raise RecoveryError(f"{label} baseline hash changed")
    return rows[line_count:]


IDENTITY_FIELDS = (
    "dispatch_id",
    "task_id",
    "sample_index",
    "attempt_index",
    "attempt_id",
)


def _identities(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    attempt_ids: set[str] = set()
    for raw in contract["expected_dangling_intents"]:
        row = dict(raw)
        dispatch_id = str(row.get("dispatch_id") or "")
        attempt_id = str(row.get("attempt_id") or "")
        if not dispatch_id or dispatch_id in result:
            raise RecoveryError("contract has a duplicate/empty dispatch id")
        if not attempt_id or attempt_id in attempt_ids:
            raise RecoveryError("contract has a duplicate/empty attempt id")
        result[dispatch_id] = row
        attempt_ids.add(attempt_id)
    return result


def _validate_dispatch_baseline(
    rows: list[dict[str, Any]],
    *,
    config_sha256: str,
    expected: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    intents: dict[str, dict[str, Any]] = {}
    settlements: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != continuation.DISPATCH_SCHEMA:
            raise RecoveryError("dispatch schema mismatch")
        if row.get("config_sha256") != config_sha256:
            raise RecoveryError("foreign dispatch config")
        dispatch_id = str(row.get("dispatch_id") or "")
        kind = row.get("record_type")
        if kind == "dispatch_intent":
            target = intents
        elif kind == "dispatch_settlement":
            target = settlements
        else:
            raise RecoveryError("invalid dispatch record type")
        if not dispatch_id or dispatch_id in target:
            raise RecoveryError("duplicate/empty dispatch record")
        target[dispatch_id] = row
    if set(settlements) - set(intents):
        raise RecoveryError("orphan dispatch settlement")
    for dispatch_id, settlement in settlements.items():
        intent = intents[dispatch_id]
        if any(settlement.get(field) != intent.get(field) for field in IDENTITY_FIELDS):
            raise RecoveryError("dispatch settlement identity mismatch")
        if settlement.get("attempt_recorded") is not True:
            raise RecoveryError("settlement lacks an attempt receipt")
    dangling = set(intents) - set(settlements)
    if dangling != set(expected):
        foreign = sorted(dangling - set(expected))
        missing = sorted(set(expected) - dangling)
        detail = foreign[0] if foreign else missing[0]
        raise RecoveryError(
            f"dangling intent set is not the exact sealed recovery set: {detail}"
        )
    for dispatch_id, sealed in expected.items():
        intent = intents[dispatch_id]
        for field in IDENTITY_FIELDS:
            if intent.get(field) != sealed.get(field):
                raise RecoveryError(
                    f"sealed dangling intent identity mismatch: {dispatch_id}"
                )
        if intent.get("requested_max_tokens") != sealed["requested_max_tokens"]:
            raise RecoveryError("dangling intent completion cap mismatch")
    return intents, settlements


def _expected_attempt(
    intent: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    prompt_sha256: str,
    evidence_sha256: str,
    contract_sha256: str,
    recorded_at: str,
) -> dict[str, Any]:
    policy = contract["fixed_policy"]
    return {
        "schema": runner.RUN_SCHEMA_VERSION,
        "record_type": "api_attempt",
        "attempt_id": intent["attempt_id"],
        "config_sha256": policy["config_sha256"],
        "task_id": intent["task_id"],
        "sample_index": intent["sample_index"],
        "attempt_index": intent["attempt_index"],
        "prompt_sha256": prompt_sha256,
        "requested_model": policy["requested_model"],
        "requested_max_tokens": policy["requested_max_tokens"],
        "provider": policy["provider"],
        "slot_policy_sha256": policy["slot_policy_sha256"],
        "started_at": intent["created_at"],
        "finished_at": recorded_at,
        "response_received": False,
        "slot_terminal": False,
        "candidate_valid": None,
        "terminal_reason": None,
        "transport_retry": True,
        "retryable_transport": True,
        "transport_error": RECOVERY_REASON,
        "fatal_response_contract": False,
        "budget_charge_tokens": (
            policy["max_prompt_tokens"] + policy["requested_max_tokens"]
        ),
        "usage": None,
        "response": None,
        "recovery_reason": RECOVERY_REASON,
        "provider_charge_status": CHARGE_STATUS,
        "duplicate_call_warning": DUPLICATE_WARNING,
        "oom_recovery_contract_sha256": contract_sha256,
        "oom_journal_evidence_sha256": evidence_sha256,
    }


def _expected_settlement(
    intent: Mapping[str, Any],
    *,
    evidence_sha256: str,
    contract_sha256: str,
    recorded_at: str,
) -> dict[str, Any]:
    return {
        "schema": continuation.DISPATCH_SCHEMA,
        "record_type": "dispatch_settlement",
        "config_sha256": intent["config_sha256"],
        **{field: intent[field] for field in IDENTITY_FIELDS},
        "attempt_recorded": True,
        "settled_at": recorded_at,
        "recovery_reason": RECOVERY_REASON,
        "provider_response_state": "unknown_after_process_oom",
        "provider_charge_status": CHARGE_STATUS,
        "duplicate_call_warning": DUPLICATE_WARNING,
        "oom_recovery_contract_sha256": contract_sha256,
        "oom_journal_evidence_sha256": evidence_sha256,
    }


def _same_static_recovery(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    timestamp_field: str,
) -> bool:
    if not str(actual.get(timestamp_field) or ""):
        return False
    return all(
        actual.get(key) == value
        for key, value in expected.items()
        if key != timestamp_field
    ) and set(actual) == set(expected)


def recover_exact_dangling(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    out_root: Path,
    prompt_map: Mapping[str, Mapping[str, Any]],
    source_terminal: Mapping[tuple[str, int], Mapping[str, Any]],
    overlay_terminal: Mapping[tuple[str, int], Mapping[str, Any]],
    next_attempt: Mapping[tuple[str, int], int],
    evidence_sha256: str,
    apply: bool,
) -> dict[str, Any]:
    specs = contract["baseline_journals"]
    attempts_path = out_root / "attempts64.jsonl"
    outcomes_path = out_root / "outcomes64.jsonl"
    dispatches_path = out_root / "dispatches64.jsonl"
    attempts_raw, attempts = _raw_rows(attempts_path)
    outcomes_raw, outcomes = _raw_rows(outcomes_path)
    dispatches_raw, dispatches = _raw_rows(dispatches_path)
    attempt_extras = _verify_prefix(
        attempts_raw, attempts, specs["attempts64.jsonl"], "attempt journal"
    )
    dispatch_extras = _verify_prefix(
        dispatches_raw,
        dispatches,
        specs["dispatches64.jsonl"],
        "dispatch journal",
    )
    if (
        len(outcomes) != specs["outcomes64.jsonl"]["lines"]
        or len(outcomes_raw) != specs["outcomes64.jsonl"]["bytes"]
        or sha256_bytes(outcomes_raw) != specs["outcomes64.jsonl"]["sha256"]
    ):
        raise RecoveryError("outcome journal changed before recovery")

    expected = _identities(contract)
    baseline_dispatches = dispatches[: specs["dispatches64.jsonl"]["lines"]]
    intents, _settlements = _validate_dispatch_baseline(
        baseline_dispatches,
        config_sha256=contract["fixed_policy"]["config_sha256"],
        expected=expected,
    )

    extras_by_attempt: dict[str, dict[str, Any]] = {}
    for row in attempt_extras:
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id in extras_by_attempt:
            raise RecoveryError("duplicate recovery attempt receipt")
        extras_by_attempt[attempt_id] = row
    extras_by_dispatch: dict[str, dict[str, Any]] = {}
    for row in dispatch_extras:
        if row.get("record_type") != "dispatch_settlement":
            raise RecoveryError("non-settlement row appended during recovery")
        dispatch_id = str(row.get("dispatch_id") or "")
        if dispatch_id in extras_by_dispatch:
            raise RecoveryError("duplicate recovery settlement")
        extras_by_dispatch[dispatch_id] = row
    expected_attempt_ids = {
        str(value["attempt_id"]) for value in expected.values()
    }
    if set(extras_by_attempt) - expected_attempt_ids:
        raise RecoveryError("foreign attempt appended during recovery")
    if set(extras_by_dispatch) - set(expected):
        raise RecoveryError("foreign settlement appended during recovery")

    reservation = (
        contract["fixed_policy"]["max_prompt_tokens"]
        + contract["fixed_policy"]["requested_max_tokens"]
    )
    recovered: list[dict[str, Any]] = []
    attempts_journal = runner.JsonlJournal(attempts_path)
    dispatch_journal = runner.JsonlJournal(dispatches_path)
    for dispatch_id in sorted(expected):
        intent = intents[dispatch_id]
        slot = (str(intent["task_id"]), int(intent["sample_index"]))
        attempt_id = str(intent["attempt_id"])
        if slot in source_terminal or slot in overlay_terminal:
            raise RecoveryError(f"recovery slot is already terminal: {slot}")
        existing_attempt = extras_by_attempt.get(attempt_id)
        existing_settlement = extras_by_dispatch.get(dispatch_id)
        expected_next_attempt = int(intent["attempt_index"]) + (
            1 if existing_attempt is not None else 0
        )
        if int(next_attempt.get(slot, 0)) != expected_next_attempt:
            raise RecoveryError(f"recovery attempt index is not next for {slot}")
        prompt = prompt_map.get(slot[0])
        if not isinstance(prompt, Mapping):
            raise RecoveryError(f"recovery task has no sealed prompt: {slot[0]}")
        prompt_sha = str(prompt.get("prompt_sha256") or "")
        if prompt_sha != contract["fixed_policy"]["prompt_sha256"]:
            raise RecoveryError("recovery prompt fingerprint mismatch")

        recorded_at = (
            str(existing_attempt.get("finished_at"))
            if existing_attempt is not None
            else (
                str(existing_settlement.get("settled_at"))
                if existing_settlement is not None
                else runner.utc_now()
            )
        )
        expected_attempt = _expected_attempt(
            intent,
            contract=contract,
            prompt_sha256=prompt_sha,
            evidence_sha256=evidence_sha256,
            contract_sha256=contract_sha256,
            recorded_at=recorded_at,
        )
        expected_settlement = _expected_settlement(
            intent,
            evidence_sha256=evidence_sha256,
            contract_sha256=contract_sha256,
            recorded_at=recorded_at,
        )
        if existing_attempt is not None and not _same_static_recovery(
            existing_attempt,
            expected_attempt,
            timestamp_field="finished_at",
        ):
            raise RecoveryError(f"existing recovery attempt was tampered: {attempt_id}")
        if existing_settlement is not None and not _same_static_recovery(
            existing_settlement,
            expected_settlement,
            timestamp_field="settled_at",
        ):
            raise RecoveryError(
                f"existing recovery settlement was tampered: {dispatch_id}"
            )
        if existing_settlement is not None and existing_attempt is None:
            raise RecoveryError("settlement exists without recovery attempt receipt")
        if apply and existing_attempt is None:
            attempts_journal.append(expected_attempt)
            extras_by_attempt[attempt_id] = expected_attempt
        if apply and existing_settlement is None:
            dispatch_journal.append(expected_settlement)
            extras_by_dispatch[dispatch_id] = expected_settlement
        recovered.append(
            {
                **{field: intent[field] for field in IDENTITY_FIELDS},
                "prompt_sha256": prompt_sha,
                "full_reservation_tokens": reservation,
                "attempt_receipt_present": existing_attempt is not None or apply,
                "settlement_present": existing_settlement is not None or apply,
            }
        )
    return {
        "recovered": recovered,
        "recovered_count": len(recovered),
        "full_reservation_tokens_per_attempt": reservation,
        "total_conservative_charge_tokens": reservation * len(recovered),
        "duplicate_call_warning": DUPLICATE_WARNING,
        "would_write": not apply,
    }


def _write_audit_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "schema",
            "status",
            "arm",
            "contract_sha256",
            "oom_journal_evidence_sha256",
            "recovered_dispatch_ids",
            "recovered_attempt_ids",
            "recovered_count",
            "duplicate_call_warning",
        ):
            if existing.get(field) != value.get(field):
                raise RecoveryError(f"existing recovery audit mismatch: {field}")
        return
    runner.atomic_write_json(path, value)


def execute(
    *,
    contract_path: Path,
    workspace: Path,
    mode: str,
    supplied_journal: Path | None,
    process_alive: Callable[[int], bool] = default_process_alive,
    hostname: str | None = None,
) -> dict[str, Any]:
    contract, contract_sha = load_contract(contract_path)
    workspace = workspace.expanduser().resolve()
    continuation_path = resolve_under(
        workspace, contract["continuation_contract"]["path"]
    )
    if runner.sha256_file(continuation_path) != contract[
        "continuation_contract"
    ]["sha256"]:
        raise RecoveryError("continuation contract hash mismatch")
    continuation_contract, continuation_sha = continuation.load_contract(
        continuation_path
    )
    if continuation_sha != contract["continuation_contract"]["sha256"]:
        raise RecoveryError("continuation contract digest mismatch")
    arm_spec = continuation_contract["arms"]["codex"]
    run_root = resolve_under(workspace, continuation_contract["run_root"])
    source_root = (run_root / arm_spec["source_run"]).resolve()
    out_root = (run_root / arm_spec["continuation_run"]).resolve()
    if str(out_root) != str(
        resolve_under(workspace, contract["continuation_out_root"])
    ):
        raise RecoveryError("continuation output root mismatch")

    journal_text = read_journal_evidence(
        contract["oom_evidence"], supplied_journal
    )
    evidence = verify_oom_evidence(
        contract,
        workspace=workspace,
        journal_text=journal_text,
        process_alive=process_alive,
        hostname=hostname or socket.gethostname(),
    )

    with runner.RunLock(source_root / ".run.lock"):
        with runner.RunLock(out_root / ".run.lock"):
            snapshot = continuation.validate_source(
                workspace=workspace,
                contract=continuation_contract,
                contract_sha256=continuation_sha,
                arm="codex",
                out_override=out_root,
            )
            req_args = continuation.request_args(
                snapshot,
                resolve_under(workspace, snapshot.policy["deepseek_env_file"]),
                1,
            )
            overlay_terminal, next_attempt, _outcomes = (
                continuation.load_overlay_state(snapshot, req_args)
            )
            continuation.verify_source_still_frozen(snapshot)
            result = recover_exact_dangling(
                contract=contract,
                contract_sha256=contract_sha,
                out_root=out_root,
                prompt_map=snapshot.prompt_map,
                source_terminal=snapshot.source_terminal,
                overlay_terminal=overlay_terminal,
                next_attempt=next_attempt,
                evidence_sha256=evidence["journal_sha256"],
                apply=mode == "recover",
            )
            if mode == "recover":
                # The ordinary loader must now accept both journals and prove
                # exact one-to-one dispatch/attempt coverage.
                continuation.load_overlay_state(snapshot, req_args)
                dispatched = continuation.load_dispatches(
                    out_root / "dispatches64.jsonl",
                    config_sha256=snapshot.overlay_config_sha256,
                )
                recorded = {
                    str(row["attempt_id"])
                    for row in runner.load_jsonl(
                        out_root / "attempts64.jsonl", "64K attempts"
                    )
                }
                if dispatched != recorded:
                    raise RecoveryError(
                        "post-recovery dispatch/attempt coverage mismatch"
                    )
                continuation.verify_source_still_frozen(snapshot)
                recovered = result["recovered"]
                audit = {
                    "schema": AUDIT_SCHEMA,
                    "status": "completed",
                    "arm": "codex",
                    "recorded_at": runner.utc_now(),
                    "contract_sha256": contract_sha,
                    "oom_journal_evidence_sha256": evidence["journal_sha256"],
                    "dead_owner_pid": evidence["dead_pid"],
                    "dead_owner_host": evidence["host"],
                    "archived_locks": evidence["archived_locks"],
                    "recovered_count": result["recovered_count"],
                    "recovered_dispatch_ids": sorted(
                        row["dispatch_id"] for row in recovered
                    ),
                    "recovered_attempt_ids": sorted(
                        row["attempt_id"] for row in recovered
                    ),
                    "full_reservation_tokens_per_attempt": result[
                        "full_reservation_tokens_per_attempt"
                    ],
                    "total_conservative_charge_tokens": result[
                        "total_conservative_charge_tokens"
                    ],
                    "recovery_reason": RECOVERY_REASON,
                    "provider_charge_status": CHARGE_STATUS,
                    "duplicate_call_warning": DUPLICATE_WARNING,
                    "provider_imported": False,
                    "provider_called": False,
                    "resume_runtime_requirements": contract[
                        "resume_runtime_requirements"
                    ],
                    "attempts64_after": runner.file_record(
                        out_root / "attempts64.jsonl"
                    ),
                    "dispatches64_after": runner.file_record(
                        out_root / "dispatches64.jsonl"
                    ),
                    "outcomes64_unchanged": runner.file_record(
                        out_root / "outcomes64.jsonl"
                    ),
                }
                _write_audit_once(
                    out_root / "oom_recovery_audit_v1.json", audit
                )
            return {
                **result,
                "mode": mode,
                "out_root": str(out_root),
                "evidence": evidence,
                "provider_imported": False,
                "provider_called": False,
            }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = execute(
            contract_path=args.contract,
            workspace=args.workspace,
            mode=args.mode,
            supplied_journal=args.journal_evidence,
        )
        print(
            f"DEEPSEEK64_OOM_{args.mode.upper()}_OK "
            f"recovered={result['recovered_count']} "
            f"reservation_each={result['full_reservation_tokens_per_attempt']} "
            f"provider_called=false out={result['out_root']}",
            flush=True,
        )
        if args.mode == "recover":
            print(f"WARNING: {DUPLICATE_WARNING}", flush=True)
        return 0
    except Exception as exc:
        print(
            f"DEEPSEEK64_OOM_FAILED_CLOSED "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
