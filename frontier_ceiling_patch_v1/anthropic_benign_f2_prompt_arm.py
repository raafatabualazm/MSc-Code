#!/usr/bin/env python3
"""Sealed, context-explicit Anthropic prompt overlay for the F2 ceiling.

This module does not call an API.  It defines a separate experimental prompt
arm which leaves each sealed F2 payload byte-for-byte unchanged, while adding
plain-language benchmark context around the opaque representation.

The source F2 artifacts remain governed by their original manifests.  In
particular, this overlay never rewrites an F2 row or presents itself as the
original prompt arm.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import frontier_passk as audited
from frontier_core import (
    PreflightError,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_text,
    stable_sha256,
)
from frontier_f2 import F2_SCHEMA, F2_SYSTEM_PROMPT, decode_f2


SCHEMA = "anthropic-benign-research-f2-prompt-arm-v1"
ARM_LABEL = "anthropic_benign_research_context_f2_v1"

# Authorization is deliberately attributed to the operator.  The sealed data
# manifests establish the benchmark/evaluator facts below, but they do not
# establish legal ownership or authorization independently.
BENIGN_CONTEXT = (
    "CONTROLLED PROGRAM-EQUIVALENCE BENCHMARK — "
    "OPERATOR-ATTESTED AUTHORIZATION\n"
    "The benchmark operator states that this is an authorized, controlled "
    "research evaluation of supplied compiler artifacts. Work only from the "
    "static benchmark payload in the user message. The requested result is "
    "Dart source for the benchmark's local evaluator; the payload does not "
    "identify or request interaction with an external target. The payload is "
    "a sealed lossless semantic representation with an explicit control-flow "
    "graph. Reconstruct a semantically equivalent fn0.\n\n"
    "ORIGINAL F2 OUTPUT AND DECODING CONTRACT (verbatim):\n"
)

BENIGN_SYSTEM_PROMPT = BENIGN_CONTEXT + F2_SYSTEM_PROMPT

USER_HEADER = (
    "CONTROLLED BENCHMARK ROW\n"
    f"Prompt arm: {ARM_LABEL}\n"
    f"Representation: {F2_SCHEMA}\n"
    "Target language/function: Dart fn0\n"
    "Evaluation role: measure\n"
    "Gold source and acceptance tests: withheld from the model\n"
    "The exact sealed F2 payload begins after this marker and is unchanged.\n"
    "--- BEGIN EXACT SEALED F2 PAYLOAD ---\n"
)


def arm_contract() -> dict[str, Any]:
    """Return the immutable public contract for this separate prompt arm."""

    contract = {
        "schema": SCHEMA,
        "arm_label": ARM_LABEL,
        "relationship_to_original": (
            "separate experimental prompt arm; does not replace or mutate "
            "the original sealed F2 prompt arm"
        ),
        "source_representation_schema": F2_SCHEMA,
        "source_system_prompt_sha256": sha256_text(F2_SYSTEM_PROMPT),
        "runtime_system_prompt": BENIGN_SYSTEM_PROMPT,
        "runtime_system_prompt_sha256": sha256_text(BENIGN_SYSTEM_PROMPT),
        "user_header": USER_HEADER,
        "user_header_sha256": sha256_text(USER_HEADER),
        "f2_payload_placement": (
            "byte-identical UTF-8 suffix immediately following user_header"
        ),
        "authorization_evidence": {
            "kind": "operator_attestation_only",
            "artifact_verified": False,
            "prompt_wording_attributes_authorization_to_operator": True,
            "runtime_requires_explicit_operator_attestation_flag": True,
        },
        "artifact_provenance_limits": [
            (
                "The sealed manifests do not independently establish legal "
                "ownership, licensing, or authorization for every underlying "
                "source artifact."
            ),
            (
                "The sealed manifests do not establish how a provider safety "
                "classifier will treat the task."
            ),
            (
                "The Opus-named arm's evaluator seal identifies role=measure "
                "but does not itself contain heldout_measure_only=true."
            ),
        ],
        "required_refusal_reporting": {
            "native_anthropic_stop_reason_refusal_reported_separately": True,
            "normalized_content_filter_reported_separately": True,
            "refusals_not_combined_with_incorrect_nonrefusal_candidates": True,
            "slot_refusal_rate": True,
            "task_any_refusal_rate": True,
            "task_all_slots_refused_rate": True,
            "unconditional_pass_at_k_remains_primary": True,
            "conditional_nonrefusal_pass_rate_is_descriptive_only": True,
        },
    }
    contract["contract_sha256"] = stable_sha256(contract)
    return contract


def extract_exact_f2_payload(user_content: str) -> str:
    """Extract and validate the byte-identical F2 suffix from an arm prompt."""

    if not isinstance(user_content, str) or not user_content.startswith(USER_HEADER):
        raise PreflightError("benign-arm user message has no exact sealed header")
    payload = user_content[len(USER_HEADER) :]
    if not payload:
        raise PreflightError("benign-arm user message has no F2 payload")
    try:
        decode_f2(payload)
    except Exception as exc:
        raise PreflightError("benign-arm F2 suffix is malformed") from exc
    return payload


def _message_pair(messages: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if len(messages) != 2:
        raise PreflightError("F2 prompt must contain exactly system+user messages")
    first, second = messages
    if first.get("role") != "system" or second.get("role") != "user":
        raise PreflightError("F2 prompt roles must be exactly system then user")
    system = first.get("content")
    user = second.get("content")
    if not isinstance(system, str) or not isinstance(user, str):
        raise PreflightError("F2 prompt message content must be text")
    return system, user


def build_benign_messages(
    original_messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Create the separate prompt arm and prove F2 information preservation."""

    source_system, source_payload = _message_pair(original_messages)
    if source_system != F2_SYSTEM_PROMPT:
        raise PreflightError(
            "source prompt system message is not the sealed original F2 contract"
        )
    try:
        source_prefix, source_canonical = decode_f2(source_payload)
    except Exception as exc:
        raise PreflightError("source prompt user message is not valid F2") from exc

    transformed = [
        {"role": "system", "content": BENIGN_SYSTEM_PROMPT},
        {"role": "user", "content": USER_HEADER + source_payload},
    ]
    runtime_system, runtime_user = _message_pair(transformed)
    recovered_payload = extract_exact_f2_payload(runtime_user)
    recovered_prefix, recovered_canonical = decode_f2(recovered_payload)
    if recovered_payload.encode("utf-8") != source_payload.encode("utf-8"):
        raise PreflightError("benign prompt overlay changed the sealed F2 bytes")
    if recovered_prefix != source_prefix or recovered_canonical != source_canonical:
        raise PreflightError("benign prompt overlay changed decoded F2 semantics")
    if not runtime_system.endswith(F2_SYSTEM_PROMPT):
        raise PreflightError("runtime system prompt changed the F2 grammar")

    proof = {
        "schema": SCHEMA,
        "arm_label": ARM_LABEL,
        "source_system_prompt_sha256": sha256_text(source_system),
        "runtime_system_prompt_sha256": sha256_text(runtime_system),
        "user_header_sha256": sha256_text(USER_HEADER),
        "source_f2_payload_sha256": sha256_text(source_payload),
        "recovered_f2_payload_sha256": sha256_text(recovered_payload),
        "source_f2_utf8_bytes": len(source_payload.encode("utf-8")),
        "recovered_f2_utf8_bytes": len(recovered_payload.encode("utf-8")),
        "decoded_constant_prefix_sha256": sha256_text(source_prefix),
        "decoded_canonical_sha256": stable_sha256(source_canonical),
        "f2_payload_utf8_bytes_identical": True,
        "decoded_constant_prefix_identical": True,
        "decoded_canonical_semantics_identical": True,
        "original_f2_grammar_verbatim_suffix_of_runtime_system": True,
    }
    proof["proof_sha256"] = stable_sha256(proof)
    return transformed, proof


def verify_source_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the common claims used in the readable header hold."""

    source_pair = provenance.get("source_pair_manifest_claims")
    source_seal = provenance.get("source_eval_seal_claims")
    invariants = provenance.get("preflight_invariants")
    if not isinstance(source_pair, Mapping):
        raise PreflightError("source pair-manifest claims are absent")
    if not isinstance(source_seal, Mapping):
        raise PreflightError("source evaluator-seal claims are absent")
    if not isinstance(invariants, Mapping):
        raise PreflightError("source preflight invariants are absent")

    if source_pair.get("schema") != audited.PAIR_MANIFEST_SCHEMA:
        raise PreflightError("source pair-manifest schema is unexpected")
    if source_pair.get("rows") != 175:
        raise PreflightError("benign research arm is sealed to exactly 175 tasks")
    if source_seal.get("selected_role") != "measure":
        raise PreflightError("source evaluator role is not measure")

    required = (
        "input_mode_is_prematerialized_f2",
        "selected_pair_arm_artifact_bindings_verified",
        "paired_acceptance_test_sequence_sha256_verified",
        "ordered_prompt_eval_task_ids_identical",
        "per_row_f2_syntax_verified",
        "per_row_f2_verification_map_verified",
        "prompts_never_truncated",
        "tests_not_exposed_to_teacher",
        "source_not_exposed_to_teacher",
        "exact_private_source_and_tests_absent_from_f2_text",
    )
    missing = [name for name in required if invariants.get(name) is not True]
    if missing:
        raise PreflightError(
            "source provenance does not prove required invariant(s): "
            + ", ".join(missing)
        )
    return {
        "verified_from_sealed_artifacts": {
            "rows": 175,
            "task_set_sha256": provenance.get("task_set_sha256"),
            "acceptance_test_sequence_sha256": provenance.get(
                "acceptance_test_sequence_sha256"
            ),
            "pair_manifest_sha256": source_pair.get("sha256"),
            "pair_arm_key": source_pair.get("pair_arm_key"),
            "evaluation_role": "measure",
            "representation_schema": provenance.get(
                "source_f2_manifest_claims", {}
            ).get("representation_schema"),
            "tests_withheld": True,
            "gold_source_withheld": True,
            "f2_syntax_and_row_verification_checked": True,
            "no_prompt_truncation": True,
        },
        "operator_attested_not_artifact_verified": {
            "analysis_is_authorized": True,
        },
        "artifact_provenance_limits": arm_contract()[
            "artifact_provenance_limits"
        ],
    }


def apply_prompt_arm(
    *,
    tokenizer: Any,
    plans: list[dict[str, Any]],
    prompt_map: MutableMapping[str, dict[str, Any]],
    config_sha256: str,
    provenance: MutableMapping[str, Any],
    args: Any,
    out: Path,
) -> tuple[list[dict[str, Any]], MutableMapping[str, dict[str, Any]], dict[str, Any]]:
    """Apply the overlay to an already verified/prepared F2 run.

    This function is deterministic and performs no provider calls.
    """

    source_evidence = verify_source_provenance(provenance)
    if getattr(args, "operator_attests_authorized_benchmark", False) is not True:
        raise PreflightError(
            "benign prompt arm requires an explicit operator authorization "
            "attestation"
        )
    if len(plans) != 175 or len(prompt_map) != 175:
        raise PreflightError("benign prompt arm requires the complete 175-task arm")

    row_commitments: list[dict[str, Any]] = []
    ordered_payload_hashes: list[str] = []
    ordered_runtime_prompt_hashes: list[str] = []
    maximum_tokens = -1
    maximum_task_id = ""
    for plan in plans:
        task_id = str(plan.get("task_id") or "")
        prompt_record = prompt_map.get(task_id)
        if not task_id or not isinstance(prompt_record, MutableMapping):
            raise PreflightError("prepared prompt map is incomplete")
        source_messages = plan.get("messages")
        if not isinstance(source_messages, Sequence):
            raise PreflightError(f"prepared task {task_id} has no messages")
        transformed, proof = build_benign_messages(source_messages)
        runtime_prompt_sha = stable_sha256(transformed)
        token_count = audited.count_prompt_tokens(
            transformed,
            tokenizer,
            chat_overhead_reserve=args.chat_overhead_reserve,
        )
        estimated = int(token_count["estimated_prompt_tokens"])
        if estimated > args.max_prompt_tokens:
            raise PreflightError(
                f"benign-arm prompt {task_id} has {estimated} estimated "
                f"tokens, cap is {args.max_prompt_tokens}; refusing to truncate"
            )

        source_prompt_sha = str(plan.get("prompt_sha256") or "")
        plan["messages"] = transformed
        plan["prompt_sha256"] = runtime_prompt_sha
        plan["estimated_prompt_tokens"] = estimated
        prompt_record["messages"] = transformed
        prompt_record["prompt_sha256"] = runtime_prompt_sha
        prompt_record["token_count"] = token_count
        prompt_record["token_count_basis"] = (
            "sealed_qwen_tokenizer_estimate_recomputed_for_benign_overlay"
        )
        prompt_record["prompt_arm"] = {
            "schema": SCHEMA,
            "arm_label": ARM_LABEL,
            "source_prompt_sha256": source_prompt_sha,
            "runtime_prompt_sha256": runtime_prompt_sha,
            "f2_payload_sha256": proof["source_f2_payload_sha256"],
            "preservation_proof": proof,
        }
        ordered_payload_hashes.append(proof["source_f2_payload_sha256"])
        ordered_runtime_prompt_hashes.append(runtime_prompt_sha)
        row_commitments.append(
            {
                "task_id": task_id,
                "source_prompt_sha256": source_prompt_sha,
                "runtime_prompt_sha256": runtime_prompt_sha,
                "f2_payload_sha256": proof["source_f2_payload_sha256"],
                "f2_utf8_bytes": proof["source_f2_utf8_bytes"],
                "preservation_proof_sha256": proof["proof_sha256"],
                "estimated_prompt_tokens": estimated,
            }
        )
        if estimated > maximum_tokens:
            maximum_tokens = estimated
            maximum_task_id = task_id

    contract = arm_contract()
    arm_manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "arm_label": ARM_LABEL,
        "status": "preflight_complete_no_api_calls",
        "config_sha256": config_sha256,
        "source_evidence": source_evidence,
        "contract": contract,
        "tasks": len(plans),
        "task_set_sha256": provenance.get("task_set_sha256"),
        "acceptance_test_sequence_sha256": provenance.get(
            "acceptance_test_sequence_sha256"
        ),
        "pair_manifest_sha256": provenance.get("artifacts", {})
        .get("pair_manifest", {})
        .get("sha256"),
        "pair_arm_key": provenance.get("source_pair_manifest_claims", {}).get(
            "pair_arm_key"
        ),
        "ordered_f2_payload_hashes_sha256": stable_sha256(
            ordered_payload_hashes
        ),
        "ordered_runtime_prompt_hashes_sha256": stable_sha256(
            ordered_runtime_prompt_hashes
        ),
        "maximum_estimated_prompt_tokens": maximum_tokens,
        "maximum_estimated_prompt_task_id": maximum_task_id,
        "max_prompt_tokens": args.max_prompt_tokens,
        "all_prompts_within_limit_without_truncation": True,
        "all_f2_payload_utf8_bytes_identical": True,
        "all_decoded_f2_semantics_identical": True,
        "row_commitments": row_commitments,
        "expected_refusal_report_path": "benign_refusal_report.json",
    }
    arm_manifest["manifest_sha256_excluding_self"] = stable_sha256(arm_manifest)

    provenance["prompt_arm"] = {
        "schema": SCHEMA,
        "arm_label": ARM_LABEL,
        "contract_sha256": contract["contract_sha256"],
        "manifest_path": str(out / "benign_prompt_arm_manifest.json"),
        "manifest_sha256_excluding_self": arm_manifest[
            "manifest_sha256_excluding_self"
        ],
        "source_system_prompt_sha256": contract[
            "source_system_prompt_sha256"
        ],
        "runtime_system_prompt_sha256": contract[
            "runtime_system_prompt_sha256"
        ],
        "f2_payload_bytes_preserved": True,
        "decoded_f2_semantics_preserved": True,
        "authorization_evidence_kind": "operator_attestation_only",
        "required_refusal_report_path": "benign_refusal_report.json",
    }
    provenance.setdefault("preflight_invariants", {})[
        "separate_benign_prompt_arm_explicitly_labeled"
    ] = True
    provenance["preflight_invariants"][
        "original_sealed_f2_payload_bytes_preserved"
    ] = True
    provenance["preflight_invariants"][
        "original_f2_decoded_semantics_preserved"
    ] = True
    provenance["preflight_invariants"][
        "runtime_prompt_tokens_recomputed_for_overlay"
    ] = True

    atomic_write_jsonl(
        out / "prompts.jsonl",
        [prompt_map[str(plan["task_id"])] for plan in plans],
    )
    atomic_write_json(out / "benign_prompt_arm_manifest.json", arm_manifest)
    atomic_write_json(out / "provenance.json", dict(provenance))
    return plans, prompt_map, arm_manifest


def _native_stop_reason(attempt: Mapping[str, Any]) -> str | None:
    raw = attempt.get("native_batch_result")
    if not isinstance(raw, Mapping):
        return None
    result = raw.get("result")
    if not isinstance(result, Mapping):
        return None
    message = result.get("message")
    if not isinstance(message, Mapping):
        return None
    value = message.get("stop_reason")
    return str(value) if value is not None else None


def build_refusal_report(
    *,
    terminals: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    k: int,
    config_sha256: str,
) -> dict[str, Any]:
    """Build a refusal-specific report without redefining pass@k."""

    if k <= 0 or len(set(task_ids)) != len(task_ids):
        raise PreflightError("invalid refusal-report task schedule")
    attempts_by_custom_id = {
        str(row.get("custom_id") or ""): row for row in attempts
    }
    if "" in attempts_by_custom_id or len(attempts_by_custom_id) != len(attempts):
        raise PreflightError("attempt rows have missing/duplicate custom IDs")

    allowed_tasks = set(task_ids)
    seen_slots: set[tuple[str, int]] = set()
    refusal_slots: set[tuple[str, int]] = set()
    normalized_filter_slots: set[tuple[str, int]] = set()
    native_reason_counts: dict[str, int] = {}
    for terminal in terminals:
        task_id = str(terminal.get("task_id") or "")
        sample_index = terminal.get("sample_index")
        if (
            task_id not in allowed_tasks
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < k
        ):
            raise PreflightError("terminal row is outside the sealed schedule")
        slot = (task_id, sample_index)
        if slot in seen_slots:
            raise PreflightError("duplicate terminal logical slot")
        seen_slots.add(slot)
        custom_id = str(terminal.get("custom_id") or "")
        attempt = attempts_by_custom_id.get(custom_id)
        if attempt is None:
            raise PreflightError("terminal row has no matching native attempt")
        native_reason = _native_stop_reason(attempt)
        native_key = native_reason if native_reason is not None else "(missing)"
        native_reason_counts[native_key] = (
            native_reason_counts.get(native_key, 0) + 1
        )
        normalized_reason = str(terminal.get("finish_reason") or "")
        if native_reason == "refusal":
            refusal_slots.add(slot)
            if normalized_reason != "content_filter":
                raise PreflightError(
                    "native Anthropic refusal is not normalized as content_filter"
                )
        if normalized_reason == "content_filter":
            normalized_filter_slots.add(slot)

    task_any = {
        task_id
        for task_id in task_ids
        if any((task_id, index) in refusal_slots for index in range(k))
    }
    task_all = {
        task_id
        for task_id in task_ids
        if all((task_id, index) in refusal_slots for index in range(k))
    }
    observed = len(seen_slots)
    expected = len(task_ids) * k
    report = {
        "schema": SCHEMA,
        "arm_label": ARM_LABEL,
        "config_sha256": config_sha256,
        "status": "complete" if observed == expected else "incomplete",
        "tasks": len(task_ids),
        "k": k,
        "expected_logical_slots": expected,
        "terminal_logical_slots": observed,
        "native_stop_reason_counts": native_reason_counts,
        "native_refusal_slots": len(refusal_slots),
        "native_refusal_rate_among_terminal_slots": (
            len(refusal_slots) / observed if observed else None
        ),
        "native_refusal_rate_among_expected_slots": (
            len(refusal_slots) / expected if expected else None
        ),
        "normalized_content_filter_slots": len(normalized_filter_slots),
        "native_refusal_and_normalized_content_filter_sets_identical": (
            refusal_slots == normalized_filter_slots
        ),
        "tasks_with_any_native_refusal": len(task_any),
        "task_any_native_refusal_rate": len(task_any) / len(task_ids),
        "tasks_with_all_k_slots_native_refusal": len(task_all),
        "task_all_slots_native_refusal_rate": len(task_all) / len(task_ids),
        "metric_interpretation": {
            "unconditional_pass_at_k_remains_primary": True,
            "conditional_nonrefusal_pass_rate_is_descriptive_only": True,
            "refusals_must_not_be_reported_as_capability_failures_without_qualification": True,
        },
    }
    report["report_sha256_excluding_self"] = stable_sha256(report)
    return report


def write_refusal_report(
    *,
    out: Path,
    task_ids: Sequence[str],
    k: int,
    config_sha256: str,
) -> dict[str, Any] | None:
    """Write the report when batch journals exist; make no provider calls."""

    terminals_path = out / "terminal_slots.jsonl"
    attempts_path = out / "batch_slot_attempts.jsonl"
    if not terminals_path.is_file() or not attempts_path.is_file():
        return None
    terminals = audited.load_jsonl(terminals_path, "terminal logical slots")
    attempts = audited.load_jsonl(attempts_path, "batch slot attempts")
    report = build_refusal_report(
        terminals=terminals,
        attempts=attempts,
        task_ids=task_ids,
        k=k,
        config_sha256=config_sha256,
    )
    atomic_write_json(out / "benign_refusal_report.json", report)
    return report
