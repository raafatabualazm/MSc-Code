#!/usr/bin/env python3
"""Adopt a completed benign-F2 8K screen into the $100 continuation.

The source journals are never modified.  Every adopted row is rewritten into
the continuation's distinct schema/config namespace and carries a commitment
to its immutable source row.  The continuation runner can then schedule only
the logical slots whose source response ended at ``max_tokens``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

import frontier_passk_anthropic_batch as source_batch
import frontier_passk_anthropic_batch_budget100_v2 as target_batch
from frontier_core import (
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    load_json,
    load_jsonl,
    sha256_file,
    stable_sha256,
    utc_now,
)


SCHEMA = "anthropic-benign-8k-budget100-adoption-v1"
EXPECTED_SLOTS = 350
SOURCE_CAP = 8192


class AdoptionError(RuntimeError):
    pass


def _slot_key(row: Mapping[str, Any]) -> tuple[str, int]:
    task_id = str(row.get("task_id") or "")
    sample_index = row.get("sample_index")
    if (
        not task_id
        or isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
    ):
        raise AdoptionError("row has a malformed logical-slot identity")
    return task_id, sample_index


def _unique_by_slot(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        key = _slot_key(row)
        if key in result:
            raise AdoptionError(f"{label} contains duplicate logical slot {key}")
        result[key] = row
    return result


def _assert_prompt_identity(source: Path, target: Path) -> None:
    source_tasks = load_jsonl(source / "tasks.jsonl", "source tasks")
    target_tasks = load_jsonl(target / "tasks.jsonl", "target tasks")
    if stable_sha256(source_tasks) != stable_sha256(target_tasks):
        raise AdoptionError("source and continuation task schedules differ")

    source_prompts = load_jsonl(source / "prompts.jsonl", "source prompts")
    target_prompts = load_jsonl(target / "prompts.jsonl", "target prompts")
    if len(source_prompts) != len(target_prompts):
        raise AdoptionError("source and continuation prompt counts differ")
    for source_row, target_row in zip(source_prompts, target_prompts):
        for field in ("task_id", "prompt_sha256", "messages"):
            if source_row.get(field) != target_row.get(field):
                raise AdoptionError(
                    f"source and continuation prompts differ at {field}"
                )


def _adopt_rows(
    rows: list[dict[str, Any]],
    *,
    source_config_sha256: str,
    target_config_sha256: str,
    source_path: Path,
) -> list[dict[str, Any]]:
    adopted: list[dict[str, Any]] = []
    for row_index, source_row in enumerate(rows):
        if source_row.get("config_sha256") != source_config_sha256:
            raise AdoptionError(f"foreign config row in {source_path.name}")
        row = dict(source_row)
        row["schema"] = target_batch.SCHEMA
        row["config_sha256"] = target_config_sha256
        row["adoption"] = {
            "schema": SCHEMA,
            "source_file": str(source_path),
            "source_file_sha256": sha256_file(source_path),
            "source_row_index": row_index,
            "source_row_sha256": stable_sha256(source_row),
            "source_config_sha256": source_config_sha256,
        }
        adopted.append(row)
    return adopted


def _assert_clean_target(target: Path) -> None:
    forbidden = (
        "batch_events.jsonl",
        "batch_slot_attempts.jsonl",
        "terminal_slots.jsonl",
        "outcomes.jsonl",
        "primary_8192_outcomes.jsonl",
        "batch_submission_intent.json",
        "failure.json",
        "progress.json",
        "summary.json",
        "continuation_adoption_manifest.json",
    )
    present = [name for name in forbidden if (target / name).exists()]
    if present:
        raise AdoptionError(
            "continuation target already has dynamic state: " + ", ".join(present)
        )


def adopt(source: Path, target: Path) -> dict[str, Any]:
    source = source.resolve()
    target = target.resolve()
    if not source.is_dir() or not target.is_dir():
        raise AdoptionError("source and target preflight directories must exist")
    _assert_clean_target(target)
    _assert_prompt_identity(source, target)

    source_provenance = load_json(source / "provenance.json", "source provenance")
    target_provenance = load_json(target / "provenance.json", "target provenance")
    source_config = str(source_provenance.get("config_sha256") or "")
    target_config = str(target_provenance.get("config_sha256") or "")
    if not source_config or not target_config or source_config == target_config:
        raise AdoptionError("source/target config identities are malformed")

    source_attempt_path = source / "batch_slot_attempts.jsonl"
    source_terminal_path = source / "terminal_slots.jsonl"
    source_outcome_path = source / "outcomes.jsonl"
    source_primary_path = source / "primary_8192_outcomes.jsonl"
    attempts = load_jsonl(source_attempt_path, "source slot attempts")
    terminals = load_jsonl(source_terminal_path, "source terminal slots")
    outcomes = load_jsonl(source_outcome_path, "source adaptive outcomes")
    primary = load_jsonl(source_primary_path, "source primary outcomes")

    attempt_map = _unique_by_slot(attempts, label="source attempts")
    terminal_map = _unique_by_slot(terminals, label="source terminals")
    outcome_map = _unique_by_slot(outcomes, label="source outcomes")
    primary_map = _unique_by_slot(primary, label="source primary outcomes")
    if len(attempt_map) != EXPECTED_SLOTS or len(primary_map) != EXPECTED_SLOTS:
        raise AdoptionError("source 8K stage is not complete over 350 slots")

    length_keys: set[tuple[str, int]] = set()
    for key, row in attempt_map.items():
        if row.get("schema") != source_batch.SCHEMA:
            raise AdoptionError("source attempt schema is unexpected")
        if int(row.get("requested_max_tokens", -1)) != SOURCE_CAP:
            raise AdoptionError("source attempt is not an 8K primary response")
        if row.get("result_type") != "succeeded":
            raise AdoptionError("source provider batch contains a failed request")
        if row.get("finish_reason") == "length":
            length_keys.add(key)
    if set(primary_map) != set(attempt_map):
        raise AdoptionError("source primary outcomes do not cover every attempt")
    non_length_keys = set(attempt_map) - length_keys
    if set(terminal_map) != non_length_keys or set(outcome_map) != non_length_keys:
        raise AdoptionError(
            "source terminal/adaptive journals disagree with 8K length slots"
        )

    progress = load_json(source / "progress.json", "source progress")
    if int(progress.get("remaining_logical_slots", -1)) != len(length_keys):
        raise AdoptionError("source progress disagrees with derived length slots")
    source_cost = progress.get("usage_and_list_cost")
    if not isinstance(source_cost, Mapping):
        raise AdoptionError("source progress has no provider-reported cost record")

    adopted_attempts = _adopt_rows(
        attempts,
        source_config_sha256=source_config,
        target_config_sha256=target_config,
        source_path=source_attempt_path,
    )
    adopted_terminals = _adopt_rows(
        terminals,
        source_config_sha256=source_config,
        target_config_sha256=target_config,
        source_path=source_terminal_path,
    )
    adopted_outcomes = _adopt_rows(
        outcomes,
        source_config_sha256=source_config,
        target_config_sha256=target_config,
        source_path=source_outcome_path,
    )
    adopted_primary = _adopt_rows(
        primary,
        source_config_sha256=source_config,
        target_config_sha256=target_config,
        source_path=source_primary_path,
    )

    manifest = {
        "schema": SCHEMA,
        "adopted_at": utc_now(),
        "source_root": str(source),
        "target_root": str(target),
        "source_config_sha256": source_config,
        "target_config_sha256": target_config,
        "source_schema": source_batch.SCHEMA,
        "target_schema": target_batch.SCHEMA,
        "source_cap": SOURCE_CAP,
        "source_slots": len(attempt_map),
        "adopted_terminal_slots": len(terminal_map),
        "pending_16k_slots": len(length_keys),
        "source_actual_list_cost": dict(source_cost),
        "source_artifacts": {
            "provenance": file_record(source / "provenance.json"),
            "tasks": file_record(source / "tasks.jsonl"),
            "prompts": file_record(source / "prompts.jsonl"),
            "batch_events": file_record(source / "batch_events.jsonl"),
            "batch_slot_attempts": file_record(source_attempt_path),
            "terminal_slots": file_record(source_terminal_path),
            "outcomes": file_record(source_outcome_path),
            "primary_8192_outcomes": file_record(source_primary_path),
            "progress": file_record(source / "progress.json"),
        },
        "target_preflight_artifacts": {
            "provenance": file_record(target / "provenance.json"),
            "tasks": file_record(target / "tasks.jsonl"),
            "prompts": file_record(target / "prompts.jsonl"),
        },
        "adoption_script": file_record(Path(__file__).resolve()),
        "source_files_modified": False,
        "only_8k_length_slots_pending_after_adoption": True,
    }
    manifest["manifest_sha256_excluding_self"] = stable_sha256(manifest)
    manifest_path = target / "continuation_adoption_manifest.json"
    atomic_write_json(manifest_path, manifest)

    atomic_write_jsonl(target / "batch_slot_attempts.jsonl", adopted_attempts)
    atomic_write_jsonl(target / "terminal_slots.jsonl", adopted_terminals)
    atomic_write_jsonl(target / "outcomes.jsonl", adopted_outcomes)
    atomic_write_jsonl(target / "primary_8192_outcomes.jsonl", adopted_primary)
    event = {
        "schema": target_batch.SCHEMA,
        "event_type": "source_stage_adopted",
        "recorded_at": utc_now(),
        "config_sha256": target_config,
        "source_config_sha256": source_config,
        "source_cap": SOURCE_CAP,
        "source_slots": len(attempt_map),
        "adopted_terminal_slots": len(terminal_map),
        "pending_16k_slots": len(length_keys),
        "adoption_manifest": file_record(manifest_path),
    }
    atomic_write_jsonl(target / "batch_events.jsonl", [event])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = adopt(args.source, args.target)
    print(
        "ANTHROPIC_8K_ADOPTION_OK "
        f"source_slots={result['source_slots']} "
        f"pending_16k_slots={result['pending_16k_slots']} "
        f"target={result['target_root']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
