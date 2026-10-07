#!/usr/bin/env python3
"""Audit and authorize checkpoint resume after a response-less balance error.

The exact-response runner intentionally fails fast on non-retryable HTTP
errors.  A later, user-authorized resume after replenishing the provider
balance is a new invocation boundary, however.  This tool changes only the
``retryable_transport`` field on strictly validated, response-less HTTP 402
``Insufficient Balance`` attempt rows and adds an explicit resume-override
attestation.  Returned provider responses and evaluator outcomes are never
changed.

Before applying the override, the complete original attempt journal and the
affected original rows are archived with SHA-256 receipts.  The rewritten
journal is then validated by the unchanged pinned runner's own resume loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import frontier_core as core
import frontier_passk as runner


MIGRATION_SCHEMA = "frontier-balance-resume-authorization-v1"
OVERRIDE_SCHEMA = "response-less-balance-resume-override-v1"


class MigrationError(RuntimeError):
    """Raised when a run cannot be safely authorized for resume."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MigrationError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"{label} is not a JSON object: {path}")
    return value


def load_raw_jsonl(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MigrationError(f"cannot read attempt journal {path}: {exc}") from exc
    lines = payload.splitlines(keepends=True)
    if b"".join(lines) != payload:
        raise MigrationError("attempt journal cannot be losslessly split")
    parsed: list[tuple[bytes, dict[str, Any]]] = []
    for line_number, raw in enumerate(lines, 1):
        if not raw.endswith(b"\n"):
            raise MigrationError(
                f"attempt journal line {line_number} lacks a final newline"
            )
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise MigrationError(
                f"cannot parse attempt journal line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise MigrationError(
                f"attempt journal line {line_number} is not an object"
            )
        parsed.append((raw, value))
    return parsed


def require_digest(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise MigrationError(f"{label} is not a SHA-256 digest")
    return digest


def require_expected_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != require_digest(expected, f"expected {label} SHA-256"):
        raise MigrationError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def is_exact_balance_boundary(row: Mapping[str, Any]) -> bool:
    error = str(row.get("transport_error") or "")
    return (
        row.get("schema") == runner.RUN_SCHEMA_VERSION
        and row.get("record_type") == "api_attempt"
        and row.get("response_received") is False
        and row.get("slot_terminal") is False
        and row.get("candidate_valid") is None
        and row.get("terminal_reason") is None
        and row.get("transport_retry") is True
        and row.get("retryable_transport") is False
        and row.get("fatal_response_contract") is False
        and row.get("usage") is None
        and row.get("response") is None
        and "APIStatusError:Error code: 402" in error
        and "Insufficient Balance" in error
    )


def terminal_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("task_id"),
        row.get("sample_index"),
        row.get("attempt_index"),
        row.get("attempt_id"),
        row.get("response_id"),
        row.get("code_sha256"),
        row.get("finish_reason"),
        row.get("usage"),
    )


@dataclass(frozen=True)
class MigrationPlan:
    run_dir: Path
    attempts_path: Path
    outcomes_path: Path
    original_attempts: bytes
    rewritten_attempts: bytes
    affected_original_rows: bytes
    affected: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def build_plan(
    run_dir: Path,
    *,
    expected_attempts_sha256: str,
    expected_outcomes_sha256: str,
    expected_runner_sha256: str,
    expected_core_sha256: str,
    expected_affected_rows: int,
) -> MigrationPlan:
    run_dir = run_dir.expanduser().resolve()
    attempts_path = run_dir / "attempts.jsonl"
    outcomes_path = run_dir / "outcomes.jsonl"
    prompts_path = run_dir / "prompts.jsonl"
    provenance_path = run_dir / "provenance.json"
    failure_path = run_dir / "failure.json"
    for path in (
        attempts_path,
        outcomes_path,
        prompts_path,
        provenance_path,
        failure_path,
    ):
        if not path.is_file():
            raise MigrationError(f"required run artifact is missing: {path}")
    if (run_dir / "summary.json").exists() or (run_dir / "manifest.json").exists():
        raise MigrationError("refusing to migrate a finalized run")

    attempts_before_sha = require_expected_hash(
        attempts_path, expected_attempts_sha256, "attempt journal"
    )
    outcomes_sha = require_expected_hash(
        outcomes_path, expected_outcomes_sha256, "outcome journal"
    )
    actual_runner_sha = require_expected_hash(
        Path(runner.__file__).resolve(),
        expected_runner_sha256,
        "runner source",
    )
    actual_core_sha = require_expected_hash(
        Path(core.__file__).resolve(),
        expected_core_sha256,
        "core source",
    )

    provenance = load_object(provenance_path, "provenance")
    failure = load_object(failure_path, "failure record")
    if provenance.get("schema") != runner.RUN_SCHEMA_VERSION:
        raise MigrationError("provenance uses an incompatible schema")
    if failure.get("schema") != runner.RUN_SCHEMA_VERSION:
        raise MigrationError("failure record uses an incompatible schema")
    if failure.get("status") != "failed_closed":
        raise MigrationError("run does not have a failed-closed failure record")
    failure_text = str(failure.get("error") or "")
    if "402" not in failure_text or "Insufficient Balance" not in failure_text:
        raise MigrationError("run failure is not an insufficient-balance boundary")

    config = provenance.get("config")
    if not isinstance(config, Mapping):
        raise MigrationError("provenance config is missing")
    config_sha = require_digest(provenance.get("config_sha256"), "config SHA-256")
    if core.stable_sha256(config) != config_sha:
        raise MigrationError("provenance config fingerprint is inconsistent")
    identity = config.get("runtime_identity")
    if not isinstance(identity, Mapping):
        raise MigrationError("runtime identity is missing")
    if identity.get("runner_sha256") != actual_runner_sha:
        raise MigrationError("pinned runner identity disagrees with installed source")
    if identity.get("core_sha256") != actual_core_sha:
        raise MigrationError("pinned core identity disagrees with installed source")

    raw_rows = load_raw_jsonl(attempts_path)
    selected_indices: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    grouped_indices: dict[tuple[str, int], list[int]] = {}
    for index, (_, row) in enumerate(raw_rows):
        task_id = str(row.get("task_id") or "")
        sample_index = row.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise MigrationError("attempt journal has an invalid sample index")
        grouped_indices.setdefault((task_id, sample_index), []).append(index)
        if row.get("response_received") is False and row.get(
            "retryable_transport"
        ) is False:
            if not is_exact_balance_boundary(row):
                raise MigrationError(
                    "found a non-retryable response-less row that is not an "
                    "exact HTTP 402 Insufficient Balance boundary"
                )
            selected_indices.append(index)
            selected_rows.append(row)

    if len(selected_indices) != expected_affected_rows:
        raise MigrationError(
            f"expected {expected_affected_rows} balance rows, found "
            f"{len(selected_indices)}"
        )
    if expected_affected_rows <= 0:
        raise MigrationError("expected affected-row count must be positive")

    worst_case = int(config["max_prompt_tokens"]) + int(
        config["max_output_tokens"]
    )
    selected_set = set(selected_indices)
    for index, row in zip(selected_indices, selected_rows):
        key = (str(row["task_id"]), int(row["sample_index"]))
        if grouped_indices[key][-1] != index:
            raise MigrationError(
                f"balance boundary is not the latest attempt for slot {key}"
            )
        if row.get("budget_charge_tokens") != worst_case:
            raise MigrationError(
                f"balance boundary has a wrong reservation charge for slot {key}"
            )

    authorized_at = utc_now()
    rewritten_lines: list[bytes] = []
    affected_raw: list[bytes] = []
    affected_receipts: list[dict[str, Any]] = []
    terminal_raw_before: list[bytes] = []
    terminal_raw_after: list[bytes] = []
    for index, (raw, original) in enumerate(raw_rows):
        if original.get("response_received") is True:
            terminal_raw_before.append(raw)
        if index not in selected_set:
            rewritten = raw
        else:
            affected_raw.append(raw)
            source_row_sha = sha256_bytes(raw)
            updated = dict(original)
            updated["retryable_transport"] = True
            updated["resume_override"] = {
                "schema": OVERRIDE_SCHEMA,
                "authorized_at": authorized_at,
                "reason": "provider_balance_replenished_for_new_invocation",
                "original_retryable_transport": False,
                "original_row_sha256": source_row_sha,
                "returned_provider_response": False,
            }
            rewritten = (
                json.dumps(
                    updated,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            affected_receipts.append(
                {
                    "task_id": original["task_id"],
                    "sample_index": original["sample_index"],
                    "attempt_index": original["attempt_index"],
                    "attempt_id": original["attempt_id"],
                    "original_row_sha256": source_row_sha,
                    "authorized_row_sha256": sha256_bytes(rewritten),
                    "response_received": False,
                }
            )
        rewritten_lines.append(rewritten)
        if original.get("response_received") is True:
            terminal_raw_after.append(rewritten)

    if terminal_raw_before != terminal_raw_after:
        raise MigrationError("a returned-provider-response row would be changed")
    original_attempts = b"".join(raw for raw, _ in raw_rows)
    rewritten_attempts = b"".join(rewritten_lines)
    affected_original_rows = b"".join(affected_raw)
    if len(rewritten_lines) != len(raw_rows):
        raise MigrationError("attempt row count changed")

    prompts = runner.load_jsonl(prompts_path, "prompt journal")
    prompt_map = {
        str(row["task_id"]): {"prompt_sha256": row["prompt_sha256"]}
        for row in prompts
    }
    slot_policy_sha = require_digest(
        config.get("slot_policy_sha256"), "slot-policy SHA-256"
    )
    with tempfile.TemporaryDirectory(prefix="frontier_balance_resume_") as tmp:
        candidate = Path(tmp) / "attempts.jsonl"
        candidate.write_bytes(rewritten_attempts)
        terminal, next_attempt = runner.load_resume_attempts(
            candidate,
            config_sha=config_sha,
            prompt_map=prompt_map,
            budget=core.TokenBudget(0),
            requested_model=str(config["model_requested"]),
            k=int(config["k"]),
            max_prompt_tokens=int(config["max_prompt_tokens"]),
            requested_max_tokens=int(config["max_output_tokens"]),
            max_transport_attempts_per_slot=int(
                config["max_attempts_per_sample"]
            ),
            slot_policy_sha256=slot_policy_sha,
        )
    outcomes = runner.load_resume_outcomes(
        outcomes_path,
        config_sha=config_sha,
        evaluator_sha256=str(config["expected_evaluator_sha256"]),
    )
    terminal_attempt_keys = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in terminal.items()
    }
    orphan_outcomes = sorted(set(outcomes) - terminal_attempt_keys)
    if orphan_outcomes:
        raise MigrationError(
            f"outcome journal has orphan records after authorization: "
            f"{orphan_outcomes[0]}"
        )
    affected_keys = {
        (str(row["task_id"]), int(row["sample_index"])) for row in selected_rows
    }
    for key in affected_keys:
        if key in terminal:
            raise MigrationError(f"balance-boundary slot is already terminal: {key}")
        if next_attempt.get(key, 0) <= int(
            next(
                row["attempt_index"]
                for row in selected_rows
                if (str(row["task_id"]), int(row["sample_index"])) == key
            )
        ):
            raise MigrationError(f"resume cursor did not advance for slot {key}")

    report = {
        "schema": MIGRATION_SCHEMA,
        "status": "validated",
        "authorized_at": authorized_at,
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "config_sha256": config_sha,
        "runner_sha256": actual_runner_sha,
        "core_sha256": actual_core_sha,
        "attempts_before_sha256": attempts_before_sha,
        "attempts_after_sha256": sha256_bytes(rewritten_attempts),
        "attempt_rows_before": len(raw_rows),
        "attempt_rows_after": len(rewritten_lines),
        "terminal_response_rows_before": len(terminal_raw_before),
        "terminal_response_rows_after": len(terminal_raw_after),
        "terminal_response_rows_byte_identical": True,
        "affected_response_less_balance_rows": len(selected_rows),
        "affected_rows": affected_receipts,
        "affected_original_rows_sha256": sha256_bytes(affected_original_rows),
        "outcomes_sha256": outcomes_sha,
        "outcomes_unchanged": True,
        "validated_terminal_slots": len(terminal),
        "validated_outcomes": len(outcomes),
        "provider_calls_made": False,
    }
    return MigrationPlan(
        run_dir=run_dir,
        attempts_path=attempts_path,
        outcomes_path=outcomes_path,
        original_attempts=original_attempts,
        rewritten_attempts=rewritten_attempts,
        affected_original_rows=affected_original_rows,
        affected=tuple(selected_rows),
        report=report,
    )


def write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_replace_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.balance-resume.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_plan(plan: MigrationPlan, archive_root: Path | None) -> dict[str, Any]:
    run_dir = plan.run_dir
    if archive_root is None:
        stamp = plan.report["authorized_at"].replace("-", "").replace(":", "")
        archive_dir = (
            run_dir
            / "resume_migrations"
            / f"{stamp}_{plan.report['attempts_before_sha256'][:12]}"
        )
    else:
        archive_dir = archive_root.expanduser().resolve()
    if archive_dir.exists():
        raise MigrationError(f"archive directory already exists: {archive_dir}")

    with runner.RunLock(run_dir / ".run.lock"):
        if sha256_file(plan.attempts_path) != plan.report["attempts_before_sha256"]:
            raise MigrationError("attempt journal changed after validation")
        if sha256_file(plan.outcomes_path) != plan.report["outcomes_sha256"]:
            raise MigrationError("outcome journal changed after validation")
        archive_dir.mkdir(parents=True, exist_ok=False)
        write_new_bytes(
            archive_dir / "attempts.before.jsonl", plan.original_attempts
        )
        write_new_bytes(
            archive_dir / "affected_response_less_402.before.jsonl",
            plan.affected_original_rows,
        )
        shutil.copyfile(
            run_dir / "provenance.json", archive_dir / "provenance.before.json"
        )
        shutil.copyfile(
            run_dir / "failure.json", archive_dir / "failure.before.json"
        )
        atomic_replace_bytes(plan.attempts_path, plan.rewritten_attempts)
        if sha256_file(plan.attempts_path) != plan.report["attempts_after_sha256"]:
            raise MigrationError("authorized attempt journal hash mismatch")
        if sha256_file(plan.outcomes_path) != plan.report["outcomes_sha256"]:
            raise MigrationError("outcome journal changed during authorization")
        completed = dict(plan.report)
        completed["status"] = "complete"
        completed["archive_dir"] = str(archive_dir)
        runner.atomic_write_json(archive_dir / "migration.json", completed)
        return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-attempts-sha256", required=True)
    parser.add_argument("--expected-outcomes-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--expected-affected-rows", type=int, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(
            args.run_dir,
            expected_attempts_sha256=args.expected_attempts_sha256,
            expected_outcomes_sha256=args.expected_outcomes_sha256,
            expected_runner_sha256=args.expected_runner_sha256,
            expected_core_sha256=args.expected_core_sha256,
            expected_affected_rows=args.expected_affected_rows,
        )
        if args.apply:
            report = apply_plan(plan, args.archive_root)
        else:
            report = dict(plan.report)
            report["status"] = "dry_run_validated"
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": MIGRATION_SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
