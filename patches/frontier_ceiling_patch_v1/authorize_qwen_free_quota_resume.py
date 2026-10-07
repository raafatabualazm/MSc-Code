#!/usr/bin/env python3
"""Audit and authorize Qwen frontier resume after exact free-quota HTTP 403s.

This is deliberately a journal migration, not an API retry policy.  It accepts
only the two exact response-less Alibaba payload shapes observed when the
model-specific free quota was exhausted:

* HTTP 403 ``AllocationQuota.FreeTierOnly``; and
* HTTP 403 ``insufficient_quota`` whose message explicitly says that the free
  quota has been exhausted.

In particular, a generic HTTP 429 ``insufficient_quota`` remains a transient
rate-limit response and is never authorized by this tool.

Before changing anything, the tool validates the sealed Qwen request contract,
the pinned runner/core/Qwen-entry hashes, every attempt row, and every outcome.
It archives the complete original journal and the exact affected rows.  The
only semantic change is ``retryable_transport: false -> true`` on exact 403
boundary rows, accompanied by a cryptographic resume-override attestation.
All returned provider responses, outcome rows, and unaffected attempt bytes
remain byte-identical.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import authorize_balance_resume as common
import frontier_core as core
import frontier_passk as runner
import frontier_passk_qwen_completion as qwen_entry


MIGRATION_SCHEMA = "qwen-free-quota-resume-authorization-v1"
OVERRIDE_SCHEMA = "response-less-qwen-free-quota-resume-override-v1"
FREE_QUOTA_SENTENCE = "The free quota has been exhausted."
ERROR_PREFIX = (
    "api_exception:PermissionDeniedError:Error code: 403 - "
)

# Exact run/model mapping for the viable clean-v4/v5 services.  The two
# moderation-rejected 05-20/06-08 Opus shards are intentionally absent.
ALLOWED_RUN_MODELS: Mapping[str, str] = {
    "qwen37_clean_v4_0517_opus_k3_mc12k_tol10_tb8k":
        "qwen3.7-max-2026-05-17",
    "qwen37_clean_v4_0517_codex_k3_mc12k_tol10_tb8k":
        "qwen3.7-max-2026-05-17",
    "qwen37_clean_v4_preview_opus_k2_mc12k_tol10_tb8k":
        "qwen3.7-max-preview",
    "qwen37_clean_v4_preview_codex_k2_mc12k_tol10_tb8k":
        "qwen3.7-max-preview",
    "qwen37_clean_v5_supplement_0517_opus_k2_mc12k_tol10_tb8k":
        "qwen3.7-max-2026-05-17",
    "qwen37_clean_v5_supplement_0517_codex_k2_mc12k_tol10_tb8k":
        "qwen3.7-max-2026-05-17",
    "qwen37_clean_v5_supplement_preview_opus_k3_mc12k_tol10_tb8k":
        "qwen3.7-max-preview",
    "qwen37_clean_v5_supplement_preview_codex_k3_mc12k_tol10_tb8k":
        "qwen3.7-max-preview",
    "qwen37_clean_v4_0520_codex_k3_mc12k_tol10_tb8k":
        "qwen3.7-max-2026-05-20",
    "qwen37_clean_v4_0608_codex_k2_mc12k_tol10_tb8k":
        "qwen3.7-max-2026-06-08",
}


class MigrationError(RuntimeError):
    """Raised when a run cannot be safely authorized for resume."""


@dataclass(frozen=True)
class MigrationPlan:
    run_dir: Path
    attempts_path: Path
    outcomes_path: Path
    original_attempts: bytes
    rewritten_attempts: bytes
    affected_original_rows: bytes
    report: dict[str, Any]


def _exact_provider_payload(error_text: str) -> tuple[str, Mapping[str, Any]] | None:
    """Return the exact observed free-quota variant and payload, or ``None``."""

    if not error_text.startswith(ERROR_PREFIX):
        return None
    encoded = error_text[len(ERROR_PREFIX):]
    try:
        payload = ast.literal_eval(encoded)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return None
    message = error.get("message")
    if not isinstance(message, str) or not message.startswith(
        FREE_QUOTA_SENTENCE
    ):
        return None

    code = error.get("code")
    error_type = error.get("type")
    if (
        code == "AllocationQuota.FreeTierOnly"
        and error_type == "AllocationQuota.FreeTierOnly"
        and set(error) == {"message", "type", "param", "code"}
        and error.get("param") is None
        and set(payload) == {"error", "id", "request_id"}
        and isinstance(payload.get("id"), str)
        and isinstance(payload.get("request_id"), str)
    ):
        return "AllocationQuota.FreeTierOnly", payload
    if (
        code == "insufficient_quota"
        and error_type == "insufficient_quota"
        and set(error) == {"message", "id", "type", "code"}
        and isinstance(error.get("id"), str)
        and set(payload) == {"error"}
    ):
        return "insufficient_quota", payload
    return None


def qwen_free_quota_variant(row: Mapping[str, Any]) -> str | None:
    """Classify only an exact, response-less, nonretryable Qwen 403 boundary."""

    if (
        row.get("schema") != runner.RUN_SCHEMA_VERSION
        or row.get("record_type") != "api_attempt"
        or row.get("provider") != "qwen"
        or row.get("requested_model") not in set(ALLOWED_RUN_MODELS.values())
        or row.get("response_received") is not False
        or row.get("slot_terminal") is not False
        or row.get("candidate_valid") is not None
        or row.get("terminal_reason") is not None
        or row.get("transport_retry") is not True
        or row.get("retryable_transport") is not False
        or row.get("fatal_response_contract") is not False
        or row.get("usage") is not None
        or row.get("response") is not None
        or "resume_override" in row
    ):
        return None
    classified = _exact_provider_payload(str(row.get("transport_error") or ""))
    return classified[0] if classified is not None else None


def is_exact_qwen_free_quota_boundary(row: Mapping[str, Any]) -> bool:
    return qwen_free_quota_variant(row) is not None


def _require_qwen_config(
    provenance: Mapping[str, Any],
    *,
    expected_model: str,
    expected_runner_sha256: str,
    expected_core_sha256: str,
    expected_qwen_entry_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    if provenance.get("schema") != runner.RUN_SCHEMA_VERSION:
        raise MigrationError("provenance uses an incompatible schema")
    config = provenance.get("config")
    if not isinstance(config, Mapping):
        raise MigrationError("provenance config is missing")
    config_sha = common.require_digest(
        provenance.get("config_sha256"), "config SHA-256"
    )
    if core.stable_sha256(config) != config_sha:
        raise MigrationError("provenance config fingerprint is inconsistent")
    if config.get("provider") != "qwen":
        raise MigrationError("run provider is not Qwen")
    if config.get("model_requested") != expected_model:
        raise MigrationError("requested model disagrees with the sealed run mapping")
    if config.get("budget") != 0:
        raise MigrationError("Qwen clean run must use the unlimited runner budget")
    if config.get("max_output_tokens") != qwen_entry.TOTAL_COMPLETION_CAP:
        raise MigrationError("Qwen total completion cap is not sealed at 12,288")
    if config.get("max_prompt_tokens") != 12_000:
        raise MigrationError("Qwen prompt cap is not sealed at 12,000")
    if config.get("max_attempts_per_sample") != 6:
        raise MigrationError("Qwen transport-attempt cap is not sealed at six")

    identity = config.get("runtime_identity")
    if not isinstance(identity, Mapping):
        raise MigrationError("runtime identity is missing")
    expected_identity = {
        "runner_sha256": common.require_digest(
            expected_runner_sha256, "expected runner SHA-256"
        ),
        "core_sha256": common.require_digest(
            expected_core_sha256, "expected core SHA-256"
        ),
        "qwen_completion_entry_sha256": common.require_digest(
            expected_qwen_entry_sha256,
            "expected Qwen completion entry SHA-256",
        ),
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise MigrationError(f"runtime identity mismatch for {key}")

    contract = config.get("qwen_request_contract")
    if not isinstance(contract, Mapping):
        raise MigrationError("Qwen request contract is missing")
    exact_contract = {
        "schema": qwen_entry.REQUEST_CONTRACT_SCHEMA,
        "provider": "qwen",
        "request_cap_parameter": "max_completion_tokens",
        "forbidden_request_cap_parameter": "max_tokens",
        "total_completion_cap": qwen_entry.TOTAL_COMPLETION_CAP,
        "provider_completion_tolerance":
            qwen_entry.PROVIDER_COMPLETION_TOLERANCE,
        "completion_usage_validation_cap":
            qwen_entry.COMPLETION_USAGE_VALIDATION_CAP,
        "thinking_budget": qwen_entry.THINKING_BUDGET,
        "finite_runner_budget_forbidden": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            raise MigrationError(f"Qwen request contract mismatch for {key}")
    if expected_model not in contract.get("allowed_models", []):
        raise MigrationError("requested model is absent from Qwen contract allowlist")
    return config, config_sha


def build_plan(
    run_dir: Path,
    *,
    expected_attempts_sha256: str,
    expected_outcomes_sha256: str,
    expected_runner_sha256: str,
    expected_core_sha256: str,
    expected_qwen_entry_sha256: str,
    expected_affected_rows: int,
) -> MigrationPlan:
    run_dir = run_dir.expanduser().resolve()
    expected_model = ALLOWED_RUN_MODELS.get(run_dir.name)
    if expected_model is None:
        raise MigrationError(
            f"run directory is not in the exact clean-v4/v5 allowlist: "
            f"{run_dir.name}"
        )
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

    attempts_before_sha = common.require_expected_hash(
        attempts_path, expected_attempts_sha256, "attempt journal"
    )
    outcomes_sha = common.require_expected_hash(
        outcomes_path, expected_outcomes_sha256, "outcome journal"
    )
    actual_runner_sha = common.require_expected_hash(
        Path(runner.__file__).resolve(),
        expected_runner_sha256,
        "runner source",
    )
    actual_core_sha = common.require_expected_hash(
        Path(core.__file__).resolve(),
        expected_core_sha256,
        "core source",
    )
    actual_qwen_entry_sha = common.require_expected_hash(
        Path(qwen_entry.__file__).resolve(),
        expected_qwen_entry_sha256,
        "Qwen completion entry source",
    )

    provenance = common.load_object(provenance_path, "provenance")
    config, config_sha = _require_qwen_config(
        provenance,
        expected_model=expected_model,
        expected_runner_sha256=actual_runner_sha,
        expected_core_sha256=actual_core_sha,
        expected_qwen_entry_sha256=actual_qwen_entry_sha,
    )
    failure = common.load_object(failure_path, "failure record")
    if failure.get("schema") != runner.RUN_SCHEMA_VERSION:
        raise MigrationError("failure record uses an incompatible schema")
    if failure.get("status") != "failed_closed":
        raise MigrationError("run does not have a failed-closed failure record")
    failure_text = str(failure.get("error") or "")
    if (
        "PermissionDeniedError" not in failure_text
        or "Error code: 403" not in failure_text
        or FREE_QUOTA_SENTENCE not in failure_text
        or (
            "AllocationQuota.FreeTierOnly" not in failure_text
            and "insufficient_quota" not in failure_text
        )
    ):
        raise MigrationError("run failure is not an exact Qwen free-quota boundary")

    raw_rows = common.load_raw_jsonl(attempts_path)
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
            if not is_exact_qwen_free_quota_boundary(row):
                raise MigrationError(
                    "found a non-retryable response-less row that is not an "
                    "exact Qwen HTTP 403 free-quota boundary"
                )
            if row.get("requested_model") != expected_model:
                raise MigrationError("free-quota row has a foreign requested model")
            selected_indices.append(index)
            selected_rows.append(row)

    if expected_affected_rows <= 0:
        raise MigrationError("expected affected-row count must be positive")
    if len(selected_indices) != expected_affected_rows:
        raise MigrationError(
            f"expected {expected_affected_rows} free-quota rows, found "
            f"{len(selected_indices)}"
        )

    worst_case = int(config["max_prompt_tokens"]) + int(
        config["max_output_tokens"]
    )
    for index, row in zip(selected_indices, selected_rows):
        key = (str(row["task_id"]), int(row["sample_index"]))
        if grouped_indices[key][-1] != index:
            raise MigrationError(
                f"free-quota boundary is not the latest attempt for slot {key}"
            )
        if row.get("budget_charge_tokens") != worst_case:
            raise MigrationError(
                f"free-quota boundary has a wrong reservation charge for slot {key}"
            )

    authorized_at = common.utc_now()
    selected_set = set(selected_indices)
    rewritten_lines: list[bytes] = []
    affected_raw: list[bytes] = []
    affected_receipts: list[dict[str, Any]] = []
    terminal_raw_before: list[bytes] = []
    terminal_raw_after: list[bytes] = []
    variant_counts: collections.Counter[str] = collections.Counter()
    for index, (raw, original) in enumerate(raw_rows):
        if original.get("response_received") is True:
            terminal_raw_before.append(raw)
        if index not in selected_set:
            rewritten = raw
        else:
            affected_raw.append(raw)
            source_row_sha = common.sha256_bytes(raw)
            variant = qwen_free_quota_variant(original)
            if variant is None:
                raise MigrationError("selected row lost its exact boundary identity")
            variant_counts[variant] += 1
            updated = dict(original)
            updated["retryable_transport"] = True
            updated["resume_override"] = {
                "schema": OVERRIDE_SCHEMA,
                "authorized_at": authorized_at,
                "reason": "model_free_quota_replenished_for_new_invocation",
                "provider_error_variant": variant,
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
                    "provider_error_variant": variant,
                    "original_row_sha256": source_row_sha,
                    "authorized_row_sha256": common.sha256_bytes(rewritten),
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
    slot_policy_sha = common.require_digest(
        config.get("slot_policy_sha256"), "slot-policy SHA-256"
    )
    qwen_entry.install_qwen_completion_policy()
    with tempfile.TemporaryDirectory(prefix="qwen_free_quota_resume_") as tmp:
        candidate = Path(tmp) / "attempts.jsonl"
        candidate.write_bytes(rewritten_attempts)
        terminal, next_attempt = runner.load_resume_attempts(
            candidate,
            config_sha=config_sha,
            prompt_map=prompt_map,
            budget=core.TokenBudget(0),
            requested_model=expected_model,
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
    for row in selected_rows:
        key = (str(row["task_id"]), int(row["sample_index"]))
        if key in terminal:
            raise MigrationError(f"free-quota slot is already terminal: {key}")
        if next_attempt.get(key, 0) <= int(row["attempt_index"]):
            raise MigrationError(f"resume cursor did not advance for slot {key}")

    report = {
        "schema": MIGRATION_SCHEMA,
        "status": "validated",
        "authorized_at": authorized_at,
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "requested_model": expected_model,
        "config_sha256": config_sha,
        "runner_sha256": actual_runner_sha,
        "core_sha256": actual_core_sha,
        "qwen_completion_entry_sha256": actual_qwen_entry_sha,
        "attempts_before_sha256": attempts_before_sha,
        "attempts_after_sha256": common.sha256_bytes(rewritten_attempts),
        "attempt_rows_before": len(raw_rows),
        "attempt_rows_after": len(rewritten_lines),
        "terminal_response_rows_before": len(terminal_raw_before),
        "terminal_response_rows_after": len(terminal_raw_after),
        "terminal_response_rows_byte_identical": True,
        "unaffected_attempt_rows_byte_identical": True,
        "affected_response_less_qwen_free_quota_rows": len(selected_rows),
        "provider_error_variant_counts": dict(sorted(variant_counts.items())),
        "affected_rows": affected_receipts,
        "affected_original_rows_sha256": common.sha256_bytes(
            affected_original_rows
        ),
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
        report=report,
    )


def apply_plan(
    plan: MigrationPlan,
    archive_root: Path | None,
) -> dict[str, Any]:
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
        if (
            common.sha256_file(plan.attempts_path)
            != plan.report["attempts_before_sha256"]
        ):
            raise MigrationError("attempt journal changed after validation")
        if (
            common.sha256_file(plan.outcomes_path)
            != plan.report["outcomes_sha256"]
        ):
            raise MigrationError("outcome journal changed after validation")
        archive_dir.mkdir(parents=True, exist_ok=False)
        common.write_new_bytes(
            archive_dir / "attempts.before.jsonl", plan.original_attempts
        )
        common.write_new_bytes(
            archive_dir / "affected_response_less_qwen_403.before.jsonl",
            plan.affected_original_rows,
        )
        shutil.copyfile(
            run_dir / "provenance.json", archive_dir / "provenance.before.json"
        )
        shutil.copyfile(
            run_dir / "failure.json", archive_dir / "failure.before.json"
        )
        common.atomic_replace_bytes(plan.attempts_path, plan.rewritten_attempts)
        if (
            common.sha256_file(plan.attempts_path)
            != plan.report["attempts_after_sha256"]
        ):
            raise MigrationError("authorized attempt journal hash mismatch")
        if (
            common.sha256_file(plan.outcomes_path)
            != plan.report["outcomes_sha256"]
        ):
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
    parser.add_argument("--expected-qwen-entry-sha256", required=True)
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
            expected_qwen_entry_sha256=args.expected_qwen_entry_sha256,
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
