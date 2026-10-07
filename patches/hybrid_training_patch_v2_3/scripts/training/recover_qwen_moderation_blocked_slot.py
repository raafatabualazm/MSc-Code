#!/usr/bin/env python3
"""Seal one losslessly transported Qwen draw for a moderation-blocked F2 slot.

Alibaba's input moderation can false-positive on the arbitrary CJK characters
used by the frozen F2 instruction codebook.  This recovery is deliberately
manual and narrow: it accepts a provider response already obtained from the
same model, seed, and generation parameters after replacing each non-ASCII F2
character with ``~UHHHH;`` and appending a fixed decoder instruction to the
system message.

The canonical student/API prompt is not rewritten.  The candidate remains
keyed to that canonical prompt, but its payload explicitly binds the exact
transport messages, the reversible transform, the saved raw provider response,
and the operator attestation.  This is sequence distillation from a losslessly
equivalent teacher input, not a claim that the blocked raw-byte request ran.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .qwen_direct_compact_teacher_artifact import (
    ArtifactError,
    JournalState,
    PromptRow,
    append_event,
    atomic_write_json,
    build_messages,
    file_record,
    make_slot_terminal_event,
    normalize_response,
    sha256_text,
    stable_sha256,
)


TRANSPORT_SCHEMA = "qwen-f2-unicode-escape-transport-v1"
RECOVERY_SCHEMA = "qwen-moderation-blocked-slot-recovery-v1"
TRANSPORT_SYSTEM_SUFFIX = (
    "Transport layer: in the user message, every ASCII escape ~UHHHH; "
    "denotes exactly the single Unicode code point U+HHHH. Expand every "
    "escape exactly before decoding F2."
)
ESCAPE_RE = re.compile(r"~U([0-9A-F]{4,6});")
EVENT_METADATA = {
    "schema",
    "event",
    "created_at",
    "candidate_id",
    "candidate_payload_sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--response-json", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--attestation-output", required=True, type=Path)
    parser.add_argument(
        "--moderation-error-code",
        choices=("data_inspection_failed",),
        required=True,
    )
    parser.add_argument(
        "--operator-attests-exact-request",
        action="store_true",
        help=(
            "Required: attest that the saved response used the sealed model, "
            "seed, generation parameters, and deterministic transport below"
        ),
    )
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} is not a JSON object")
    return value


def find_prompt(path: Path, task_id: str) -> tuple[PromptRow, dict[str, Any]]:
    found: tuple[PromptRow, dict[str, Any]] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"prompt JSONL is invalid at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict) or row.get("task_id") != task_id:
                continue
            if found is not None:
                raise ArtifactError(f"duplicate prompt task ID: {task_id}")
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise ArtifactError("moderation-recovery prompt has no text")
            if row.get("text_sha256") != sha256_text(text):
                raise ArtifactError("moderation-recovery prompt hash mismatch")
            found = (
                PromptRow(
                    task_id=task_id,
                    text=text,
                    text_sha256=sha256_text(text),
                    source_record_sha256=stable_sha256(row),
                    source_schema=str(row.get("schema") or ""),
                    representation_schema=str(
                        row.get("representation_schema") or ""
                    ),
                    system_prompt_sha256=str(
                        row.get("system_prompt_sha256") or ""
                    ),
                ),
                row,
            )
    if found is None:
        raise ArtifactError(f"task is absent from prompt JSONL: {task_id}")
    return found


def escape_non_ascii(text: str) -> str:
    if ESCAPE_RE.search(text) or "~" in text:
        raise ArtifactError(
            "canonical F2 text collides with the moderation transport escape"
        )
    return "".join(
        character
        if ord(character) < 128
        else f"~U{ord(character):04X};"
        for character in text
    )


def unescape_non_ascii(text: str) -> str:
    return ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 16)), text)


def candidate_with_transport(
    candidate: Mapping[str, Any],
    transport: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        key: value for key, value in candidate.items() if key not in EVENT_METADATA
    }
    payload["request_transport"] = dict(transport)
    payload_sha256 = stable_sha256(payload)
    basis = {
        "task_id": payload.get("task_id"),
        "sample_index": payload.get("sample_index"),
        "prompt_sha256": payload.get("prompt_sha256"),
        "candidate_payload_sha256": payload_sha256,
    }
    return {
        "schema": candidate["schema"],
        "event": candidate["event"],
        "created_at": candidate["created_at"],
        "candidate_id": stable_sha256(basis),
        "candidate_payload_sha256": payload_sha256,
        **payload,
    }


def main() -> int:
    args = parse_args()
    if not args.operator_attests_exact_request:
        raise ArtifactError("--operator-attests-exact-request is required")
    if not 0 <= args.sample_index < 8:
        raise ArtifactError("sample index must be in [0, 8)")
    prompt_record = file_record(args.prompt_jsonl)
    if prompt_record["sha256"] != args.expected_prompt_sha256.strip().lower():
        raise ArtifactError("prompt artifact hash differs from its expected hash")

    state = JournalState.load(
        args.journal,
        allow_indeterminate_slots=True,
    )
    if state.header is None:
        raise ArtifactError("teacher journal has no run header")
    header = state.header
    payload = header.get("payload") or {}
    if (
        prompt_record["sha256"]
        != (payload.get("prompt_artifact") or {}).get("sha256")
        or (payload.get("provider_authorization") or {}).get(
            "token_plan_automation_authorized"
        )
        is not True
        or payload.get("objective_mode") != "sequence_only"
    ):
        raise ArtifactError(
            "journal is not the authorized sequence-only prompt parent"
        )

    prompt, _raw_prompt = find_prompt(args.prompt_jsonl, args.task_id)
    system_prompt = str(
        (payload.get("f2_prompt_contract") or {}).get("system_prompt") or ""
    )
    canonical_messages = build_messages(system_prompt, prompt)
    canonical_prompt_sha256 = stable_sha256(canonical_messages)
    slot = (args.task_id, canonical_prompt_sha256, args.sample_index)
    started = state.starts.get(slot)
    if started is None or slot in state.slots or slot in state.terminals:
        raise ArtifactError(
            "moderation recovery requires one started, nonterminal logical slot"
        )
    attempts = state.reissue_attempts.get(slot, [])
    if not attempts:
        raise ArtifactError(
            "moderation recovery requires a durable orphan-reissue attempt"
        )
    if state.error_counts.get(slot, 0) < 1:
        raise ArtifactError("moderation recovery slot has no provider error receipt")

    transported_text = escape_non_ascii(prompt.text)
    if unescape_non_ascii(transported_text) != prompt.text:
        raise ArtifactError("moderation transport does not round-trip")
    transport_messages = [
        {
            "role": "system",
            "content": system_prompt + "\n" + TRANSPORT_SYSTEM_SUFFIX,
        },
        {"role": "user", "content": transported_text},
    ]
    response = load_json(args.response_json, "saved provider response")
    requested_model = str(payload.get("requested_model") or "")
    request_parameters = dict(started.get("request_parameters") or {})
    candidate = normalize_response(
        response,
        task_id=args.task_id,
        sample_index=args.sample_index,
        prompt_sha256=canonical_prompt_sha256,
        requested_model=requested_model,
        request_parameters=request_parameters,
        required_function=str(payload.get("required_function") or "fn0"),
    )
    if (
        candidate.get("completion_attested") is not True
        or (candidate.get("response") or {}).get("returned_model")
        != requested_model
    ):
        raise ArtifactError("saved moderation-recovery response is not complete")
    response_id = str((candidate.get("response") or {}).get("request_id") or "")
    existing_response_ids = {
        str((row.get("response") or {}).get("request_id") or "")
        for row in state.candidates.values()
    }
    if response_id in existing_response_ids:
        raise ArtifactError("saved response ID already exists in the journal")

    attestation = {
        "schema": RECOVERY_SCHEMA,
        "task_id": args.task_id,
        "sample_index": args.sample_index,
        "canonical_prompt_sha256": canonical_prompt_sha256,
        "transport_schema": TRANSPORT_SCHEMA,
        "transport_messages_sha256": stable_sha256(transport_messages),
        "transport_system_suffix": TRANSPORT_SYSTEM_SUFFIX,
        "transport_system_suffix_sha256": sha256_text(
            TRANSPORT_SYSTEM_SUFFIX
        ),
        "canonical_text_sha256": sha256_text(prompt.text),
        "transported_text_sha256": sha256_text(transported_text),
        "roundtrip_sha256": sha256_text(
            unescape_non_ascii(transported_text)
        ),
        "non_ascii_codepoints_escaped": sum(
            ord(character) >= 128 for character in prompt.text
        ),
        "moderation_error_code_operator_attestation": (
            args.moderation_error_code
        ),
        "same_requested_model": requested_model,
        "same_request_parameters": request_parameters,
        "same_request_parameters_sha256": stable_sha256(request_parameters),
        "provider_response": file_record(args.response_json),
        "provider_response_id": response_id,
        "journal_header_sha256": header.get("header_sha256"),
        "slot_started_id": started.get("slot_started_id"),
        "latest_orphan_reissue_attempt_id": attempts[-1].get(
            "orphan_reissue_attempt_id"
        ),
        "operator_attests_exact_request": True,
        "claims": {
            "canonical_raw_byte_request_executed": False,
            "losslessly_equivalent_transport_request_executed": True,
            "dense_full_vocabulary_kl": False,
            "monte_carlo_sequence_distillation_draw": True,
        },
    }
    if args.attestation_output.exists():
        raise ArtifactError("refusing to overwrite moderation attestation")
    atomic_write_json(args.attestation_output, attestation)
    attestation_record = file_record(args.attestation_output)
    transport = {
        "schema": TRANSPORT_SCHEMA,
        "reason": "provider_input_moderation_false_positive",
        "canonical_messages_sha256": canonical_prompt_sha256,
        "transport_messages_sha256": stable_sha256(transport_messages),
        "reversible_non_ascii_escape": "~UHHHH;",
        "roundtrip_proven": True,
        "canonical_raw_byte_request_executed": False,
        "moderation_error_code_operator_attestation": (
            args.moderation_error_code
        ),
        "attestation": attestation_record,
    }
    transported_candidate = candidate_with_transport(candidate, transport)
    append_event(args.journal, transported_candidate)
    terminal = make_slot_terminal_event(
        started,
        transported_candidate,
        outcome_type="candidate",
        orphan_reissue_attempt_id=str(
            attempts[-1].get("orphan_reissue_attempt_id") or ""
        ),
    )
    append_event(args.journal, terminal)

    repaired = JournalState.load(
        args.journal,
        allow_indeterminate_slots=True,
    )
    if (
        slot not in repaired.terminals
        or repaired.slots.get(slot) != transported_candidate["candidate_id"]
    ):
        raise ArtifactError("moderation-recovered slot did not seal")
    print(
        "QWEN_MODERATION_BLOCKED_SLOT_RECOVERED "
        f"task={args.task_id} sample={args.sample_index} "
        f"candidate_id={transported_candidate['candidate_id']} "
        f"attestation_sha256={attestation_record['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
