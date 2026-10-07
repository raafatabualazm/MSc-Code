#!/usr/bin/env python3
"""Rebase a failed Qwen journal while preserving every completed draw.

This recovery is intentionally narrow: it may reopen only terminal rejected
draws whose provider response ended with ``finish_reason=length``. Successful
candidate bytes, request IDs, verification results, task/sample coordinates,
and requested seeds are copied unchanged into a new hash-chained journal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from scripts.training.qwen_direct_compact_teacher_artifact import (
    JOURNAL_CHAIN_FIELDS,
    ArtifactError,
    JournalState,
    append_event,
    atomic_write_json,
    file_record,
    load_hash_chained_journal,
    stable_sha256,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--target-contract",
        required=True,
        type=Path,
        help="Expanded target contract to bind into the recovered run header.",
    )
    parser.add_argument("--max-retries", required=True, type=int)
    parser.add_argument(
        "--length-max-token-capacities",
        required=True,
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--allow-no-rejections",
        action="store_true",
        help=(
            "Permit a completed journal reheader when only sealed runtime "
            "metadata changed and no failed slots need reopening."
        ),
    )
    return parser.parse_args()


def _slot(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("task_id") or ""),
        str(row.get("prompt_sha256") or ""),
        int(row.get("sample_index", -1)),
    )


def _without_chain(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in JOURNAL_CHAIN_FIELDS
    }


def _finish_reason(rejection: Mapping[str, Any]) -> str:
    response = rejection.get("provider_response") or {}
    choices = response.get("choices") or []
    if not isinstance(choices, list) or len(choices) != 1:
        return ""
    return str((choices[0] or {}).get("finish_reason") or "")


def main() -> int:
    args = parse_args()
    if args.output.exists() or Path(str(args.output) + ".chain-head.json").exists():
        raise ArtifactError("recovered output already exists")
    if args.receipt.exists():
        raise ArtifactError("recovery receipt already exists")
    if args.max_retries < 0:
        raise ArtifactError("--max-retries cannot be negative")
    capacities = list(args.length_max_token_capacities)
    if capacities != sorted(set(capacities)) or any(value <= 0 for value in capacities):
        raise ArtifactError("length capacities must be unique and increasing")

    source_state = JournalState.load(args.source)
    if source_state.header is None:
        raise ArtifactError("source journal has no header")
    if not source_state.rejections and not args.allow_no_rejections:
        raise ArtifactError("source journal has no rejected draws to recover")
    non_length = [
        row
        for row in source_state.rejections.values()
        if _finish_reason(row) != "length"
    ]
    if non_length:
        raise ArtifactError("recovery refuses a non-length rejected draw")

    target_contract = file_record(args.target_contract)
    header = _without_chain(source_state.header)
    payload = dict(header.get("payload") or {})
    old_target_contract = dict(payload.get("target_length_contract") or {})
    old_trainer_contract = dict(old_target_contract.get("trainer_contract") or {})
    old_trainer_contract.update(target_contract)
    old_target_contract["trainer_contract"] = old_trainer_contract
    import json

    contract_payload = json.loads(args.target_contract.read_text(encoding="utf-8"))
    old_target_contract["max_target_tokens"] = int(
        contract_payload["max_target_tokens"]
    )
    payload["target_length_contract"] = old_target_contract
    transport = dict(payload.get("transport") or {})
    transport["application_max_retries_per_slot"] = int(args.max_retries)
    transport["length_capped_response_policy"] = {
        "same_task_draw_only": True,
        "completed_draws_reissued": False,
        "max_token_capacities": capacities,
        "capped_responses_retained_by_hash": True,
    }
    payload["transport"] = transport
    implementation = dict(payload.get("implementation") or {})
    implementation["collector"] = file_record(
        Path(__file__).resolve().with_name(
            "collect_qwen_direct_compact_teacher.py"
        )
    )
    implementation["artifact_core"] = file_record(
        Path(__file__).resolve().with_name(
            "qwen_direct_compact_teacher_artifact.py"
        )
    )
    payload["implementation"] = implementation
    header["payload"] = payload
    header["header_sha256"] = stable_sha256(payload)

    successful_slots = {
        _slot(candidate) for candidate in source_state.candidates.values()
    }
    candidate_ids = set(source_state.candidates)
    copied = 0
    append_event(args.output, header)
    for row in load_hash_chained_journal(args.source):
        event = row.get("event")
        keep = (
            event != "run_header"
            and (
                (
                    event
                    in {
                        "teacher_slot_started",
                        "teacher_slot_terminal",
                        "teacher_error",
                    }
                    and _slot(row) in successful_slots
                )
                or (
                    event == "teacher_candidate"
                    and str(row.get("candidate_id") or "") in candidate_ids
                )
                or (
                    event == "verification"
                    and str(row.get("candidate_id") or "") in candidate_ids
                )
            )
        )
        if keep:
            append_event(args.output, _without_chain(row))
            copied += 1

    recovered_state = JournalState.load(args.output)
    if (
        len(recovered_state.candidates) != len(source_state.candidates)
        or recovered_state.rejections
        or set(recovered_state.verifications)
        != set(source_state.verifications)
    ):
        raise ArtifactError("recovered journal did not preserve completed slots")
    receipt = {
        "schema": "qwen-journal-length-recovery-v1",
        "created_at": utc_now(),
        "source": file_record(args.source),
        "source_chain_head": file_record(
            Path(str(args.source) + ".chain-head.json")
        ),
        "output": file_record(args.output),
        "output_chain_head": file_record(
            Path(str(args.output) + ".chain-head.json")
        ),
        "expanded_target_contract": target_contract,
        "retained": {
            "candidate_draws": len(recovered_state.candidates),
            "verifications": len(recovered_state.verifications),
            "events_after_header": copied,
        },
        "reopened_length_slots": [
            {
                "task_id": row["task_id"],
                "sample_index": row["sample_index"],
                "provider_request_id": row["provider_request_id"],
                "provider_response_sha256": row["provider_response_sha256"],
                "finish_reason": "length",
            }
            for row in source_state.rejections.values()
        ],
        "policy": {
            "completed_draws_reissued": False,
            "only_length_capped_slots_reopened": True,
            "max_token_capacities": capacities,
            "application_max_retries_per_slot": int(args.max_retries),
        },
    }
    atomic_write_json(args.receipt, receipt)
    print(
        "QWEN_JOURNAL_RECOVERED "
        f"retained={len(recovered_state.candidates)} "
        f"reopened={len(source_state.rejections)} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
