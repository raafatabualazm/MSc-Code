#!/usr/bin/env python3
"""Fail-closed OpenAI Responses Batch ceiling run for the sealed F2 pair.

This is an additive protocol.  It never mutates or resumes an Anthropic,
Qwen, or DeepSeek run.  One invocation represents one model/arm job.  The
active default is Terra over either sealed enrichment arm:

    gpt-5.6-terra x {opus_real_fn0_cfg, codex_multifunction_cfg}

Sol remains available only when selected explicitly; its availability here
does not imply that a Sol Batch is authorized or being submitted.

Each job has 175 sealed held-out tasks and K=2 logical response slots.  The
request protocol is OpenAI Batch -> /v1/responses, max_output_tokens=32768,
reasoning.effort=max, with no sampling overrides and no prompt truncation.

The original F2 payload bytes are preserved as a suffix of a separately
labelled, operator-attested research-context overlay.  The overlay, its
preservation proof, the exact Batch input JSONL, provider-native Batch output,
and every evaluator result are retained.

``preflight`` is API-free.  Before ``submit`` creates any paid Batch, it:

* obtains authoritative per-prompt counts from responses.input_tokens;
* proves all inputs fit the provider audit cap;
* computes an exact worst-case charge using the 50%-discounted Batch rates;
* compares that Decimal-valued projection with an explicit per-job cap; and
* requires an explicit paid-Batch authorization flag.

Upload and Batch creation use deterministic idempotency keys and a durable
intent record.  Re-running submit/status/harvest is therefore resume-safe.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import frontier_passk as audited
from frontier_core import (
    JsonlJournal,
    PreflightError,
    ResponseContractError,
    atomic_write_json,
    atomic_write_jsonl,
    file_record,
    load_json,
    load_jsonl,
    sha256_file,
    sha256_text,
    stable_sha256,
    utc_now,
    wilson_interval,
)
from frontier_f2 import F2_SCHEMA, F2_SYSTEM_PROMPT, decode_f2


SCHEMA = "openai56-responses-batch-k2-fasttrack-v1"
TRANSPORT_SCHEMA = "openai-batch-v1-responses-v1"
PROMPT_ARM_SCHEMA = "openai56-authorized-research-f2-overlay-v1"
PROMPT_ARM_LABEL = "openai56_authorized_research_f2_overlay_v1"
ALLOWED_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra")
K = 2
MAX_OUTPUT_TOKENS = 32_768
REASONING_EFFORT = "max"
PROVIDER_INPUT_TOKEN_CAP = 16_384
COMPLETION_WINDOW = "24h"
BATCH_ENDPOINT = "/v1/responses"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ENV_FILE = Path("/workspace/OpenAI.env")
INPUT_TOKEN_MAX_ATTEMPTS = 5
INPUT_TOKEN_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)
OPENAI_HTTP_TIMEOUT_SECONDS = 120.0
DEFAULT_SHARD_INPUT_TOKEN_CAP = 700_000
PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICE_BASIS = "OpenAI Batch API 50% discount applied to standard input and output rates"
CACHE_WRITE_WORST_MULTIPLIER = Decimal("1.25")

# Decimal USD per one million tokens.  The values below are already the Batch
# rates (50% of $5/$30 for Sol and $2.50/$15 for Terra).
BATCH_PRICES: dict[str, dict[str, Decimal]] = {
    "gpt-5.6-sol": {
        "input_usd_per_million": Decimal("2.50"),
        "output_usd_per_million": Decimal("15.00"),
    },
    "gpt-5.6-terra": {
        "input_usd_per_million": Decimal("1.25"),
        "output_usd_per_million": Decimal("7.50"),
    },
}

_BASE_PARSE_ARGS = audited.parse_args
_BASE_CONFIG_FOR_HASH = audited.config_for_hash
_BASE_PREPARE_RUN = audited.prepare_run


RESEARCH_CONTEXT = (
    "CONTROLLED PROGRAM-EQUIVALENCE RESEARCH BENCHMARK - "
    "OPERATOR-ATTESTED AUTHORIZATION\n"
    "The benchmark operator attests that this is an authorized, controlled "
    "evaluation of supplied compiler artifacts. Work only from the static "
    "benchmark payload in this request. Do not access, probe, or interact "
    "with any external target. The requested result is self-contained Dart "
    "source for a local held-out evaluator. The payload is a sealed lossless "
    "semantic representation with an explicit control-flow graph; reconstruct "
    "a semantically equivalent fn0.\n\n"
    "ORIGINAL F2 OUTPUT AND DECODING CONTRACT (verbatim):\n"
)
RUNTIME_SYSTEM_PROMPT = RESEARCH_CONTEXT + F2_SYSTEM_PROMPT
USER_HEADER = (
    "CONTROLLED HELD-OUT BENCHMARK ROW\n"
    f"Prompt arm: {PROMPT_ARM_LABEL}\n"
    f"Representation: {F2_SCHEMA}\n"
    "Target: self-contained Dart fn0 plus only required imports/helpers\n"
    "Evaluation role: held-out measure\n"
    "Gold source and acceptance tests: withheld from the model\n"
    "The exact sealed F2 payload begins after this marker and is unchanged.\n"
    "--- BEGIN EXACT SEALED F2 PAYLOAD ---\n"
)


def prompt_arm_contract() -> dict[str, Any]:
    """Return the immutable overlay contract bound into the run hash."""

    contract: dict[str, Any] = {
        "schema": PROMPT_ARM_SCHEMA,
        "arm_label": PROMPT_ARM_LABEL,
        "relationship_to_source": (
            "separate experimental prompt overlay; original sealed F2 "
            "artifacts are neither replaced nor mutated"
        ),
        "authorization_evidence": {
            "kind": "operator_attestation_only",
            "artifact_verified": False,
            "explicit_runtime_attestation_required": True,
        },
        "source_representation_schema": F2_SCHEMA,
        "source_system_prompt_sha256": sha256_text(F2_SYSTEM_PROMPT),
        "runtime_system_prompt_sha256": sha256_text(RUNTIME_SYSTEM_PROMPT),
        "user_header_sha256": sha256_text(USER_HEADER),
        "f2_payload_placement": (
            "byte-identical UTF-8 suffix immediately following user_header"
        ),
        "gold_source_exposed": False,
        "acceptance_tests_exposed": False,
        "provider_safety_behavior_not_assumed": True,
    }
    contract["contract_sha256"] = stable_sha256(contract)
    return contract


def _flag_present(values: Sequence[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in values)


@contextlib.contextmanager
def _argv(values: Sequence[str]) -> Iterable[None]:
    prior = sys.argv
    try:
        sys.argv = [prior[0], *values]
        yield
    finally:
        sys.argv = prior


def _decimal_cap(value: str) -> Decimal | None:
    if not value.strip():
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("cost cap must be a Decimal number") from exc
    if not result.is_finite() or result <= 0:
        raise argparse.ArgumentTypeError("cost cap must be finite and positive")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse OpenAI controls, then reuse the audited sealed-input parser."""

    raw = list(sys.argv[1:] if argv is None else argv)
    front = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    front.add_argument(
        "--action",
        choices=("preflight", "submit", "status", "harvest", "auto"),
        default=os.environ.get("OPENAI56_BATCH_ACTION", "preflight"),
    )
    front.add_argument(
        "--model",
        choices=ALLOWED_MODELS,
        default=os.environ.get("OPENAI56_MODEL", "gpt-5.6-terra"),
    )
    front.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
    )
    front.add_argument(
        "--openai-env-file",
        type=Path,
        default=Path(os.environ.get("OPENAI_ENV_FILE", DEFAULT_ENV_FILE)),
    )
    front.add_argument(
        "--job-cost-cap-usd",
        default=os.environ.get("OPENAI56_JOB_COST_CAP_USD", ""),
        help=(
            "Explicit maximum worst-case Batch charge for this one model/arm "
            "job. Required for submit; excluded from the scientific config hash."
        ),
    )
    front.add_argument(
        "--authorize-paid-batch",
        action="store_true",
        default=os.environ.get("OPENAI56_AUTHORIZE_PAID_BATCH", "") == "1",
    )
    front.add_argument(
        "--attest-authorized-benchmark",
        action="store_true",
        default=(
            os.environ.get("OPENAI_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK", "") == "1"
        ),
    )
    front.add_argument(
        "--input-token-workers",
        type=int,
        default=int(os.environ.get("OPENAI56_INPUT_TOKEN_WORKERS", "8")),
    )
    front.add_argument(
        "--shard-input-token-cap",
        type=int,
        default=int(
            os.environ.get(
                "OPENAI56_SHARD_INPUT_TOKEN_CAP",
                str(DEFAULT_SHARD_INPUT_TOKEN_CAP),
            )
        ),
        help=(
            "Maximum provider-counted input tokens in one Batch shard. "
            "Shards run sequentially inside the same model/arm job."
        ),
    )
    specific, remaining = front.parse_known_args(raw)

    forbidden = (
        "--provider",
        "--model",
        "--k",
        "--max-output-tokens",
        "--temperature",
        "--top-p",
        "--budget",
        "--api-key",
        "--base-url",
        "--extra-body-json",
        "--preflight-only",
    )
    for name in forbidden:
        if _flag_present(remaining, name):
            front.error(f"{name} is fixed or disabled by the OpenAI56 Batch contract")
    if not specific.attest_authorized_benchmark:
        front.error(
            "--attest-authorized-benchmark is required; sealed artifacts do "
            "not independently prove operator authorization"
        )
    if specific.input_token_workers <= 0 or specific.input_token_workers > 32:
        front.error("--input-token-workers must be in [1,32]")
    if (
        specific.shard_input_token_cap <= 0
        or specific.shard_input_token_cap > PROVIDER_INPUT_TOKEN_CAP * 175 * K
    ):
        front.error("--shard-input-token-cap is outside the supported range")
    try:
        cap = _decimal_cap(str(specific.job_cost_cap_usd))
    except argparse.ArgumentTypeError as exc:
        front.error(str(exc))
    if specific.action == "submit":
        if not specific.authorize_paid_batch:
            front.error("--authorize-paid-batch is required for submit")
        if cap is None:
            front.error("--job-cost-cap-usd is required for submit")

    fixed = [
        "--provider",
        "qwen",  # transport is replaced process-locally below
        "--model",
        specific.model,
        "--arm",
        "compact",
        "--k",
        str(K),
        "--max-output-tokens",
        str(MAX_OUTPUT_TOKENS),
        "--temperature",
        "1",
        "--top-p",
        "1",
        "--budget",
        "0",
    ]
    with _argv([*remaining, *fixed]):
        args = _BASE_PARSE_ARGS()
    if args.input_mode != "prematerialized_f2":
        front.error("OpenAI56 Batch requires --input-mode prematerialized_f2")
    if args.limit:
        front.error("OpenAI56 Batch forbids --limit")
    if args.expected_task_count != 175:
        front.error("OpenAI56 Batch is sealed to exactly 175 tasks")
    args.provider = "openai"
    args.action = specific.action
    args.model = specific.model
    args.k = K
    args.max_output_tokens = MAX_OUTPUT_TOKENS
    args.openai_base_url = str(specific.openai_base_url).rstrip("/")
    args.openai_env_file = specific.openai_env_file
    args.job_cost_cap_usd = cap
    args.authorize_paid_batch = bool(specific.authorize_paid_batch)
    args.operator_attests_authorized_benchmark = True
    args.input_token_workers = specific.input_token_workers
    args.shard_input_token_cap = specific.shard_input_token_cap
    return args


def _load_openai_key(args: argparse.Namespace) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        values = audited.read_env_file(args.openai_env_file.expanduser())
        key = values.get("OPENAI_API_KEY", "").strip()
    return key


def resolve_api_configuration(args: argparse.Namespace) -> tuple[str, str]:
    return _load_openai_key(args), str(args.openai_base_url).rstrip("/")


def fixed_slot_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "openai56-fixed-responses-batch-slot-v1",
        "transport_schema": TRANSPORT_SCHEMA,
        "requested_model": args.model,
        "resolved_model_must_equal_requested": True,
        "tasks": 175,
        "k": K,
        "logical_slots": 175 * K,
        "fixed_max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": REASONING_EFFORT},
        "sampling_parameters_on_wire": "omitted",
        "truncation": "disabled",
        "store": False,
        "one_returned_response_consumes_one_slot": True,
        "finish_reason_length_consumes_slot": True,
        "safe_extractable_fn0_is_evaluated_even_if_incomplete": True,
        "provider_request_errors_are_not_capability_failures": True,
        "provider_request_errors_make_the_run_incomplete": True,
        "response_resampling": False,
        "early_stopping": False,
        "dispatch_order": "sample_index_then_sealed_task_order",
        "completion_window": COMPLETION_WINDOW,
        "batch_endpoint": BATCH_ENDPOINT,
        "provider_input_token_audit_cap": PROVIDER_INPUT_TOKEN_CAP,
        "batch_shard_input_token_cap": args.shard_input_token_cap,
        "at_most_one_unharvested_shard_per_job": True,
    }


def config_for_hash(args: argparse.Namespace) -> dict[str, Any]:
    config = _BASE_CONFIG_FOR_HASH(args)
    config["provider"] = "openai"
    config["model_requested"] = args.model
    config["k"] = K
    config["max_output_tokens"] = MAX_OUTPUT_TOKENS
    config["temperature"] = None
    config["top_p"] = None
    config["budget"] = 0
    policy = fixed_slot_policy(args)
    config["slot_policy"] = policy
    config["slot_policy_sha256"] = stable_sha256(policy)
    runtime = dict(config.get("runtime_identity") or {})
    runtime.update(
        {
            "audited_runner_sha256": runtime.pop("runner_sha256", None),
            "openai56_batch_runner_sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    config["runtime_identity"] = runtime
    prices = BATCH_PRICES[args.model]
    config["openai56_responses_batch"] = {
        "schema": SCHEMA,
        "transport_schema": TRANSPORT_SCHEMA,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "reasoning_effort": REASONING_EFFORT,
        "batch_discount_fraction": "0.50",
        "batch_input_usd_per_million": str(prices["input_usd_per_million"]),
        "batch_output_usd_per_million": str(prices["output_usd_per_million"]),
        "cache_write_worst_multiplier": str(CACHE_WRITE_WORST_MULTIPLIER),
        "price_basis": PRICE_BASIS,
        "price_source": PRICE_SOURCE,
        "prompt_arm_contract": prompt_arm_contract(),
        "operator_attests_authorized_benchmark": True,
        "shard_input_token_cap": args.shard_input_token_cap,
    }
    # action, current budget authorization, and the user's cost ceiling are
    # operational controls; none changes model inputs or metric semantics.
    return config


def install_hooks() -> None:
    audited.resolve_api_configuration = resolve_api_configuration
    audited.fixed_slot_policy = fixed_slot_policy
    audited.config_for_hash = config_for_hash


def _message_pair(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    if len(messages) != 2:
        raise PreflightError("source F2 prompt must be exactly system+user")
    system, user = messages
    if system.get("role") != "system" or user.get("role") != "user":
        raise PreflightError("source F2 prompt roles are not system then user")
    system_text = system.get("content")
    user_text = user.get("content")
    if not isinstance(system_text, str) or not isinstance(user_text, str):
        raise PreflightError("source F2 messages must contain text")
    return system_text, user_text


def build_overlay_messages(
    original_messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Add context while proving that the F2 payload bytes did not change."""

    source_system, source_payload = _message_pair(original_messages)
    if source_system != F2_SYSTEM_PROMPT:
        raise PreflightError("source system prompt is not the sealed F2 contract")
    try:
        source_prefix, source_graph = decode_f2(source_payload)
    except Exception as exc:
        raise PreflightError("source user message is not valid F2") from exc
    transformed = [
        {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
        {"role": "user", "content": USER_HEADER + source_payload},
    ]
    runtime_system, runtime_user = _message_pair(transformed)
    if not runtime_user.startswith(USER_HEADER):
        raise AssertionError("overlay construction lost its header")
    recovered = runtime_user[len(USER_HEADER) :]
    try:
        recovered_prefix, recovered_graph = decode_f2(recovered)
    except Exception as exc:
        raise PreflightError("overlay F2 suffix is malformed") from exc
    if recovered.encode("utf-8") != source_payload.encode("utf-8"):
        raise PreflightError("overlay changed the sealed F2 payload bytes")
    if recovered_prefix != source_prefix or recovered_graph != source_graph:
        raise PreflightError("overlay changed decoded F2 semantics")
    if not runtime_system.endswith(F2_SYSTEM_PROMPT):
        raise PreflightError("overlay changed the original F2 grammar")
    proof: dict[str, Any] = {
        "schema": PROMPT_ARM_SCHEMA,
        "source_system_prompt_sha256": sha256_text(source_system),
        "runtime_system_prompt_sha256": sha256_text(runtime_system),
        "user_header_sha256": sha256_text(USER_HEADER),
        "source_f2_payload_sha256": sha256_text(source_payload),
        "recovered_f2_payload_sha256": sha256_text(recovered),
        "source_f2_utf8_bytes": len(source_payload.encode("utf-8")),
        "recovered_f2_utf8_bytes": len(recovered.encode("utf-8")),
        "decoded_constant_prefix_sha256": sha256_text(source_prefix),
        "decoded_canonical_sha256": stable_sha256(source_graph),
        "f2_payload_utf8_bytes_identical": True,
        "decoded_f2_semantics_identical": True,
        "original_f2_grammar_is_verbatim_runtime_suffix": True,
    }
    proof["proof_sha256"] = stable_sha256(proof)
    return transformed, proof


def _verify_overlay_source(provenance: Mapping[str, Any]) -> None:
    pair = provenance.get("source_pair_manifest_claims")
    seal = provenance.get("source_eval_seal_claims")
    invariants = provenance.get("preflight_invariants")
    if not isinstance(pair, Mapping) or pair.get("rows") != 175:
        raise PreflightError("source pair manifest does not seal 175 tasks")
    if not isinstance(seal, Mapping) or seal.get("selected_role") != "measure":
        raise PreflightError("source evaluator seal is not role=measure")
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
    if not isinstance(invariants, Mapping):
        raise PreflightError("source preflight invariants are missing")
    missing = [name for name in required if invariants.get(name) is not True]
    if missing:
        raise PreflightError(
            "source preflight lacks required invariant(s): " + ", ".join(missing)
        )


def apply_prompt_overlay(
    *,
    tokenizer: Any,
    plans: list[dict[str, Any]],
    prompt_map: MutableMapping[str, dict[str, Any]],
    config_sha: str,
    provenance: MutableMapping[str, Any],
    args: argparse.Namespace,
    out: Path,
) -> None:
    _verify_overlay_source(provenance)
    if not args.operator_attests_authorized_benchmark:
        raise PreflightError("operator authorization attestation is absent")
    if len(plans) != 175 or len(prompt_map) != 175:
        raise PreflightError("overlay requires the full sealed 175-task arm")

    rows: list[dict[str, Any]] = []
    payload_hashes: list[str] = []
    runtime_hashes: list[str] = []
    for plan in plans:
        task_id = str(plan.get("task_id") or "")
        record = prompt_map.get(task_id)
        if not task_id or not isinstance(record, MutableMapping):
            raise PreflightError("prepared prompt map is incomplete")
        messages, proof = build_overlay_messages(plan["messages"])
        runtime_sha = stable_sha256(messages)
        count = audited.count_prompt_tokens(
            messages,
            tokenizer,
            chat_overhead_reserve=args.chat_overhead_reserve,
        )
        estimated = int(count["estimated_prompt_tokens"])
        if estimated > args.max_prompt_tokens:
            raise PreflightError(
                f"overlay prompt {task_id} has {estimated} sealed-Qwen tokens, "
                f"cap is {args.max_prompt_tokens}; refusing to truncate"
            )
        source_sha = str(plan.get("prompt_sha256") or "")
        plan["messages"] = messages
        plan["prompt_sha256"] = runtime_sha
        plan["estimated_prompt_tokens"] = estimated
        record["messages"] = messages
        record["prompt_sha256"] = runtime_sha
        record["token_count"] = count
        record["token_count_basis"] = (
            "sealed_qwen_tokenizer_estimate_recomputed_for_openai_overlay"
        )
        record["prompt_arm"] = {
            "schema": PROMPT_ARM_SCHEMA,
            "arm_label": PROMPT_ARM_LABEL,
            "source_prompt_sha256": source_sha,
            "runtime_prompt_sha256": runtime_sha,
            "preservation_proof": proof,
        }
        payload_hashes.append(proof["source_f2_payload_sha256"])
        runtime_hashes.append(runtime_sha)
        rows.append(
            {
                "task_id": task_id,
                "source_prompt_sha256": source_sha,
                "runtime_prompt_sha256": runtime_sha,
                "source_f2_payload_sha256": proof["source_f2_payload_sha256"],
                "source_f2_utf8_bytes": proof["source_f2_utf8_bytes"],
                "preservation_proof_sha256": proof["proof_sha256"],
                "estimated_prompt_tokens": estimated,
            }
        )
    manifest: dict[str, Any] = {
        "schema": PROMPT_ARM_SCHEMA,
        "arm_label": PROMPT_ARM_LABEL,
        "config_sha256": config_sha,
        "tasks": len(plans),
        "task_set_sha256": provenance.get("task_set_sha256"),
        "acceptance_test_sequence_sha256": provenance.get(
            "acceptance_test_sequence_sha256"
        ),
        "pair_arm_key": provenance.get("source_pair_manifest_claims", {}).get(
            "pair_arm_key"
        ),
        "pair_manifest_sha256": provenance.get("source_pair_manifest_claims", {}).get(
            "sha256"
        ),
        "contract": prompt_arm_contract(),
        "ordered_f2_payload_hashes_sha256": stable_sha256(payload_hashes),
        "ordered_runtime_prompt_hashes_sha256": stable_sha256(runtime_hashes),
        "all_f2_payload_utf8_bytes_identical": True,
        "all_decoded_f2_semantics_identical": True,
        "all_prompts_within_limit_without_truncation": True,
        "authorization_evidence": "operator_attestation_only",
        "row_commitments": rows,
    }
    manifest["manifest_sha256_excluding_self"] = stable_sha256(manifest)
    atomic_write_jsonl(
        out / "prompts.jsonl",
        [prompt_map[str(plan["task_id"])] for plan in plans],
    )
    atomic_write_json(out / "openai56_prompt_overlay_manifest.json", manifest)
    provenance["prompt_arm"] = {
        "schema": PROMPT_ARM_SCHEMA,
        "arm_label": PROMPT_ARM_LABEL,
        "contract_sha256": manifest["contract"]["contract_sha256"],
        "manifest_sha256_excluding_self": manifest["manifest_sha256_excluding_self"],
        "f2_payload_bytes_preserved": True,
        "decoded_f2_semantics_preserved": True,
        "operator_attestation_only": True,
    }
    invariants = provenance.setdefault("preflight_invariants", {})
    invariants["separate_openai56_prompt_overlay_explicitly_labeled"] = True
    invariants["original_sealed_f2_payload_bytes_preserved"] = True
    invariants["original_f2_decoded_semantics_preserved"] = True
    invariants["runtime_prompt_tokens_recomputed_for_overlay"] = True
    atomic_write_json(out / "provenance.json", dict(provenance))


def prepare_run(
    args: argparse.Namespace, out: Path
) -> tuple[Any, list[dict[str, Any]], dict[str, dict[str, Any]], str, dict[str, Any]]:
    tokenizer, plans, prompt_map, config_sha, provenance = _BASE_PREPARE_RUN(args, out)
    apply_prompt_overlay(
        tokenizer=tokenizer,
        plans=plans,
        prompt_map=prompt_map,
        config_sha=config_sha,
        provenance=provenance,
        args=args,
        out=out,
    )
    return tokenizer, plans, prompt_map, config_sha, provenance


def _response_input(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"system", "developer", "user"} or not isinstance(content, str):
            raise PreflightError("OpenAI Responses input contains an invalid message")
        result.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            }
        )
    return result


def _custom_id(task_index: int, sample_index: int) -> str:
    value = f"oai56_s{sample_index:02d}_t{task_index:03d}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise audited.RunFailure(f"invalid custom_id: {value!r}")
    return value


def request_specs(plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(plans) != 175:
        raise audited.RunFailure("request schedule requires exactly 175 plans")
    specs: list[dict[str, Any]] = []
    for sample_index in range(K):
        for task_index, plan in enumerate(plans):
            specs.append(
                {
                    "task_id": str(plan["task_id"]),
                    "task_index": task_index,
                    "sample_index": sample_index,
                    "custom_id": _custom_id(task_index, sample_index),
                }
            )
    if len(specs) != 175 * K:
        raise AssertionError("fixed request schedule is incomplete")
    return specs


def response_body(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": args.model,
        "input": _response_input(plan["messages"]),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": REASONING_EFFORT},
        "store": False,
        "truncation": "disabled",
    }


def batch_requests(
    args: argparse.Namespace,
    plans: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_task = {str(plan["task_id"]): plan for plan in plans}
    requests: list[dict[str, Any]] = []
    for spec in specs:
        plan = by_task.get(str(spec["task_id"]))
        if plan is None:
            raise audited.RunFailure("request spec references an unknown task")
        requests.append(
            {
                "custom_id": str(spec["custom_id"]),
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": response_body(args, plan),
            }
        )
    return requests


def _write_or_verify_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if path.is_file():
        prior = load_jsonl(path, path.name)
        if stable_sha256(prior) != stable_sha256(values):
            raise audited.RunFailure(
                f"existing {path.name} differs from sealed content"
            )
        return
    atomic_write_jsonl(path, values)


def _jsonable(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return json.loads(json.dumps(value))
    if hasattr(value, "model_dump"):
        try:
            result = value.model_dump(mode="json")
        except TypeError:
            result = value.model_dump()
        if isinstance(result, dict):
            return json.loads(json.dumps(result, default=str))
    if hasattr(value, "dict"):
        result = value.dict()
        if isinstance(result, dict):
            return json.loads(json.dumps(result, default=str))
    raise audited.RunFailure(f"cannot serialize {type(value).__name__}")


def _client(args: argparse.Namespace) -> tuple[Any, str]:
    key, base_url = resolve_api_configuration(args)
    if not key:
        raise PreflightError("OPENAI_API_KEY is required for this action")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise PreflightError("openai Python package 2.x is required") from exc
    return (
        OpenAI(
            api_key=key,
            base_url=base_url,
            max_retries=0,
            timeout=OPENAI_HTTP_TIMEOUT_SECONDS,
        ),
        key,
    )


def _redact_exception(exc: BaseException, secret: str) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text[:2000]


def _slot_key(task_id: str, sample_index: int) -> str:
    return f"{task_id}\x1f{sample_index}"


def _journal_rows(
    out: Path, name: str, config_sha: str, *, unique: str | None = None
) -> list[dict[str, Any]]:
    path = out / name
    rows = load_jsonl(path, name) if path.is_file() else []
    seen: set[str] = set()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure(f"foreign row in {name}")
        if unique is not None:
            key = str(row.get(unique) or "")
            if not key or key in seen:
                raise audited.RunFailure(f"duplicate/missing {unique} in {name}")
            seen.add(key)
    return rows


def _batch_events(out: Path, config_sha: str) -> list[dict[str, Any]]:
    return _journal_rows(out, "openai_batch_events.jsonl", config_sha)


def _active_submission(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    submitted = [row for row in events if row.get("event_type") == "batch_submitted"]
    harvested = {
        str(row.get("batch_id") or "")
        for row in events
        if row.get("event_type") == "batch_harvested"
    }
    active = [
        dict(row)
        for row in submitted
        if str(row.get("batch_id") or "") not in harvested
    ]
    if len(active) > 1:
        raise audited.RunFailure("more than one unharvested Batch exists")
    return active[0] if active else None


def _money(tokens: int, rate_per_million: Decimal) -> Decimal:
    if tokens < 0:
        raise audited.RunFailure("token cost cannot be negative")
    return Decimal(tokens) * rate_per_million / Decimal(1_000_000)


def _money_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001")), "f")


def cost_projection(
    *,
    model: str,
    exact_input_tokens: int,
    requests: int,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    if model not in BATCH_PRICES:
        raise audited.RunFailure(f"no Batch price for {model!r}")
    if exact_input_tokens < 0 or requests <= 0 or max_output_tokens <= 0:
        raise audited.RunFailure("invalid cost-projection inputs")
    output_tokens = requests * max_output_tokens
    prices = BATCH_PRICES[model]
    input_usd = _money(exact_input_tokens, prices["input_usd_per_million"])
    all_cache_write_input_usd = input_usd * CACHE_WRITE_WORST_MULTIPLIER
    output_usd = _money(output_tokens, prices["output_usd_per_million"])
    total = input_usd + output_usd
    return {
        "schema": "openai56-batch-worst-case-cost-v1",
        "model": model,
        "requests": requests,
        "provider_exact_input_tokens": exact_input_tokens,
        "maximum_output_tokens_per_request": max_output_tokens,
        "maximum_output_tokens_all_requests": output_tokens,
        "batch_discount_fraction": "0.50",
        "input_usd_per_million": str(prices["input_usd_per_million"]),
        "output_usd_per_million": str(prices["output_usd_per_million"]),
        "exact_input_usd": _money_text(input_usd),
        "all_cache_write_input_usd": _money_text(all_cache_write_input_usd),
        "cache_write_worst_multiplier": str(CACHE_WRITE_WORST_MULTIPLIER),
        "maximum_output_usd": _money_text(output_usd),
        "exact_worst_case_total_usd": _money_text(total),
        "price_basis": PRICE_BASIS,
        "price_source": PRICE_SOURCE,
        "reasoning_tokens_included_in_output_cap": True,
        "all_cache_write_total_usd": _money_text(
            all_cache_write_input_usd + output_usd
        ),
        "cost_gate_uses_published_batch_input_rate": True,
        "estimate_not_invoice": True,
    }


def absolute_preflight_cost(args: argparse.Namespace, requests: int) -> dict[str, Any]:
    result = cost_projection(
        model=args.model,
        exact_input_tokens=requests * PROVIDER_INPUT_TOKEN_CAP,
        requests=requests,
    )
    result["input_count_basis"] = (
        "absolute provider-audit cap; submit replaces with authoritative "
        "responses.input_tokens counts before paid Batch creation"
    )
    result["provider_exact_at_preflight"] = False
    return result


def _count_one_prompt(
    client: Any, args: argparse.Namespace, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "model": args.model,
        "input": _response_input(plan["messages"]),
        "reasoning": {"effort": REASONING_EFFORT},
        "truncation": "disabled",
    }
    last: BaseException | None = None
    for attempt in range(INPUT_TOKEN_MAX_ATTEMPTS):
        try:
            response = client.responses.input_tokens.count(**request)
            raw = _jsonable(response)
            count = raw.get("input_tokens")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise audited.RunFailure(
                    "responses.input_tokens returned an invalid count"
                )
            return raw, {
                "task_id": str(plan["task_id"]),
                "request_sha256": stable_sha256(request),
                "runtime_prompt_sha256": str(plan["prompt_sha256"]),
                "input_tokens": count,
                "native_count_response": raw,
            }
        except audited.RunFailure:
            raise
        except Exception as exc:
            last = exc
            if attempt + 1 >= INPUT_TOKEN_MAX_ATTEMPTS:
                break
            time.sleep(INPUT_TOKEN_RETRY_DELAYS[attempt])
    assert last is not None
    raise last


def authoritative_input_token_audit(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: Sequence[Mapping[str, Any]],
    config_sha: str,
    client: Any,
    api_key: str,
) -> dict[str, Any]:
    path = out / "openai_input_token_counts.jsonl"
    prior = load_jsonl(path, "OpenAI input-token counts") if path.is_file() else []
    by_task: dict[str, dict[str, Any]] = {}
    for row in prior:
        if row.get("schema") != SCHEMA or row.get("config_sha256") != config_sha:
            raise audited.RunFailure("foreign OpenAI input-token count row")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in by_task:
            raise audited.RunFailure("duplicate/malformed input-token count row")
        by_task[task_id] = row

    missing = [plan for plan in plans if str(plan["task_id"]) not in by_task]
    journal = JsonlJournal(path)
    if missing:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.input_token_workers
        ) as pool:
            futures = {
                pool.submit(_count_one_prompt, client, args, plan): plan
                for plan in missing
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    plan = futures[future]
                    try:
                        _raw, value = future.result()
                    except Exception as exc:
                        raise audited.RunFailure(
                            "OpenAI input-token count failed for "
                            f"{plan['task_id']}: {_redact_exception(exc, api_key)}"
                        ) from exc
                    row = {
                        "schema": SCHEMA,
                        "record_type": "provider_input_token_count",
                        "recorded_at": utc_now(),
                        "config_sha256": config_sha,
                        **value,
                    }
                    journal.append(row)
                    by_task[str(plan["task_id"])] = row
            finally:
                for future in futures:
                    future.cancel()

    ordered: list[dict[str, Any]] = []
    for plan in plans:
        task_id = str(plan["task_id"])
        row = by_task.get(task_id)
        if row is None:
            raise audited.RunFailure("input-token audit remains incomplete")
        request = {
            "model": args.model,
            "input": _response_input(plan["messages"]),
            "reasoning": {"effort": REASONING_EFFORT},
            "truncation": "disabled",
        }
        if row.get("request_sha256") != stable_sha256(request):
            raise audited.RunFailure("persisted token count request hash mismatch")
        if row.get("runtime_prompt_sha256") != plan["prompt_sha256"]:
            raise audited.RunFailure("persisted token count prompt hash mismatch")
        count = row.get("input_tokens")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise audited.RunFailure("persisted input-token count is invalid")
        ordered.append(row)
    maximum = max(int(row["input_tokens"]) for row in ordered)
    if maximum > PROVIDER_INPUT_TOKEN_CAP:
        task = max(ordered, key=lambda row: int(row["input_tokens"]))
        raise audited.RunFailure(
            f"provider counted {maximum} input tokens for {task['task_id']}, "
            f"cap is {PROVIDER_INPUT_TOKEN_CAP}; refusing to truncate or submit"
        )
    per_sample = sum(int(row["input_tokens"]) for row in ordered)
    audit: dict[str, Any] = {
        "schema": "openai56-provider-input-token-audit-v1",
        "config_sha256": config_sha,
        "model": args.model,
        "unique_prompts_counted": len(ordered),
        "logical_requests_covered": len(ordered) * K,
        "input_tokens_one_sample_over_tasks": per_sample,
        "exact_input_tokens_all_k_requests": per_sample * K,
        "minimum_input_tokens": min(int(row["input_tokens"]) for row in ordered),
        "maximum_input_tokens": maximum,
        "provider_input_token_cap": PROVIDER_INPUT_TOKEN_CAP,
        "all_counts_within_cap": True,
        "count_endpoint": "/v1/responses/input_tokens",
        "counting_is_provider_authoritative": True,
        "ordered_task_ids_sha256": stable_sha256(
            [str(plan["task_id"]) for plan in plans]
        ),
        "ordered_counts_sha256": stable_sha256(
            [
                {
                    "task_id": row["task_id"],
                    "input_tokens": row["input_tokens"],
                    "request_sha256": row["request_sha256"],
                }
                for row in ordered
            ]
        ),
        "count_rows": file_record(path),
    }
    audit["audit_sha256_excluding_self"] = stable_sha256(audit)
    audit_path = out / "openai_input_token_audit.json"
    if audit_path.is_file():
        old = load_json(audit_path, "OpenAI input-token audit")
        comparable = dict(old)
        comparable.pop("count_rows", None)
        expected = dict(audit)
        expected.pop("count_rows", None)
        if comparable != expected:
            raise audited.RunFailure("persisted input-token audit changed")
    atomic_write_json(audit_path, audit)
    return audit


def deterministic_shards(
    specs: Sequence[Mapping[str, Any]],
    token_count_rows: Sequence[Mapping[str, Any]],
    *,
    input_token_cap: int,
) -> list[dict[str, Any]]:
    """Pack the fixed slot order into deterministic sequential Batch shards."""

    if input_token_cap <= 0:
        raise audited.RunFailure("Batch shard input-token cap must be positive")
    by_task: dict[str, int] = {}
    for row in token_count_rows:
        task_id = str(row.get("task_id") or "")
        count = row.get("input_tokens")
        if (
            not task_id
            or task_id in by_task
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise audited.RunFailure("input-token rows cannot define Batch shards")
        by_task[task_id] = count
    shards: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for raw_spec in specs:
        spec = dict(raw_spec)
        count = by_task.get(str(spec.get("task_id") or ""))
        if count is None:
            raise audited.RunFailure("request spec has no provider input-token count")
        if count > input_token_cap:
            raise audited.RunFailure(
                f"one request needs {count} input tokens, above shard cap "
                f"{input_token_cap}"
            )
        if current and current_tokens + count > input_token_cap:
            shards.append(
                {
                    "shard_index": len(shards),
                    "input_tokens": current_tokens,
                    "request_count": len(current),
                    "request_specs": current,
                }
            )
            current = []
            current_tokens = 0
        current.append(spec)
        current_tokens += count
    if current:
        shards.append(
            {
                "shard_index": len(shards),
                "input_tokens": current_tokens,
                "request_count": len(current),
                "request_specs": current,
            }
        )
    if not shards or sum(int(row["request_count"]) for row in shards) != len(specs):
        raise audited.RunFailure("deterministic Batch sharding lost request slots")
    for shard in shards:
        shard["shard_count"] = len(shards)
        shard["custom_ids_sha256"] = stable_sha256(
            [str(row["custom_id"]) for row in shard["request_specs"]]
        )
    return shards


def _idempotency_key(config_sha: str, purpose: str, shard_index: int) -> str:
    return f"oai56-{purpose}-s{shard_index:03d}-{config_sha[:24]}"


def _submission_intent_path(out: Path) -> Path:
    return out / "openai_batch_submission_intent.json"


def _submit(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
) -> dict[str, Any]:
    events = _batch_events(out, config_sha)
    active = _active_submission(events)
    if active is not None:
        stale_intent = _submission_intent_path(out)
        if stale_intent.is_file():
            value = load_json(stale_intent, "OpenAI Batch submission intent")
            if (
                value.get("schema") != SCHEMA
                or value.get("config_sha256") != config_sha
                or str(value.get("batch_id") or "") != str(active.get("batch_id") or "")
            ):
                raise audited.RunFailure(
                    "active Batch has a conflicting unresolved submission intent"
                )
            stale_intent.unlink()
        return {
            "status": "already_submitted",
            "batch_id": active["batch_id"],
            "requests": active["request_count"],
        }
    if not args.authorize_paid_batch or args.job_cost_cap_usd is None:
        raise audited.RunFailure(
            "submit requires explicit paid-Batch authorization and job cost cap"
        )

    specs = request_specs(plans)
    requests = batch_requests(args, plans, specs)
    request_path = out / "openai_batch_requests.jsonl"
    _write_or_verify_jsonl(request_path, requests)
    request_sha = sha256_file(request_path)
    client, key = _client(args)
    audit = authoritative_input_token_audit(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
        client=client,
        api_key=key,
    )
    count_rows = load_jsonl(
        out / "openai_input_token_counts.jsonl",
        "OpenAI input-token counts",
    )
    shards = deterministic_shards(
        specs,
        count_rows,
        input_token_cap=args.shard_input_token_cap,
    )
    shard_manifest: dict[str, Any] = {
        "schema": "openai56-deterministic-batch-shards-v1",
        "config_sha256": config_sha,
        "model": args.model,
        "pair_arm_key": args.pair_arm_key,
        "input_token_cap_per_shard": args.shard_input_token_cap,
        "shard_count": len(shards),
        "request_count": len(specs),
        "exact_input_tokens": sum(int(row["input_tokens"]) for row in shards),
        "shards": shards,
    }
    shard_manifest["manifest_sha256_excluding_self"] = stable_sha256(shard_manifest)
    atomic_write_json(out / "openai_batch_shards.json", shard_manifest)
    submitted_indices = {
        int(row["shard_index"])
        for row in events
        if row.get("event_type") == "batch_submitted"
    }
    if not submitted_indices.issubset(set(range(len(shards)))):
        raise audited.RunFailure("event journal references an unknown Batch shard")
    pending = [
        shard for shard in shards if int(shard["shard_index"]) not in submitted_indices
    ]
    if not pending:
        harvested = {
            int(row["shard_index"])
            for row in events
            if row.get("event_type") == "batch_harvested"
        }
        if harvested == set(range(len(shards))):
            return {"status": "all_shards_submitted_and_harvested"}
        raise audited.RunFailure(
            "all shards were submitted but an active/unharvested shard is missing"
        )
    shard = pending[0]
    shard_index = int(shard["shard_index"])
    if shard_index != len(submitted_indices):
        raise audited.RunFailure("Batch shards were not submitted in order")
    shard_specs = list(shard["request_specs"])
    by_custom_id = {str(row["custom_id"]): row for row in requests}
    shard_requests = [by_custom_id[str(spec["custom_id"])] for spec in shard_specs]
    request_path = out / f"openai_batch_requests_shard_{shard_index:03d}.jsonl"
    _write_or_verify_jsonl(request_path, shard_requests)
    request_sha = sha256_file(request_path)
    projection = cost_projection(
        model=args.model,
        exact_input_tokens=int(audit["exact_input_tokens_all_k_requests"]),
        requests=len(requests),
    )
    projected = Decimal(str(projection["exact_worst_case_total_usd"]))
    if projected > args.job_cost_cap_usd:
        raise audited.RunFailure(
            "Batch submission blocked by exact cost gate: projected "
            f"${projection['exact_worst_case_total_usd']} > explicit job cap "
            f"${args.job_cost_cap_usd}"
        )
    gate: dict[str, Any] = {
        **projection,
        "config_sha256": config_sha,
        "explicit_job_cost_cap_usd": str(args.job_cost_cap_usd),
        "gate_passed": True,
        "provider_input_token_audit": file_record(
            out / "openai_input_token_audit.json"
        ),
        "exact_batch_requests": file_record(out / "openai_batch_requests.jsonl"),
        "deterministic_shards": file_record(out / "openai_batch_shards.json"),
    }
    gate_path = out / "openai_batch_cost_gate.json"
    if gate_path.is_file():
        if load_json(gate_path, "OpenAI Batch cost gate") != gate:
            raise audited.RunFailure(
                "persisted OpenAI Batch cost gate changed between shards"
            )
    else:
        atomic_write_json(gate_path, gate)

    commitment = stable_sha256(shard_requests)
    intent_path = _submission_intent_path(out)
    if intent_path.is_file():
        intent = load_json(intent_path, "OpenAI Batch submission intent")
        if (
            intent.get("schema") != SCHEMA
            or intent.get("config_sha256") != config_sha
            or intent.get("request_commitment_sha256") != commitment
            or intent.get("request_file_sha256") != request_sha
            or intent.get("shard_index") != shard_index
        ):
            raise audited.RunFailure("unresolved submission intent is foreign")
    else:
        intent = {
            "schema": SCHEMA,
            "status": "prepared",
            "created_at": utc_now(),
            "config_sha256": config_sha,
            "shard_index": shard_index,
            "shard_count": len(shards),
            "shard_input_tokens": int(shard["input_tokens"]),
            "request_count": len(shard_requests),
            "request_commitment_sha256": commitment,
            "request_file_sha256": request_sha,
            "upload_idempotency_key": _idempotency_key(
                config_sha, "upload", shard_index
            ),
            "batch_idempotency_key": _idempotency_key(config_sha, "batch", shard_index),
            "cost_gate": file_record(out / "openai_batch_cost_gate.json"),
        }
        atomic_write_json(intent_path, intent)

    input_file_id = str(intent.get("input_file_id") or "")
    if not input_file_id:
        try:
            with request_path.open("rb") as handle:
                receipt_obj = client.files.create(
                    file=handle,
                    purpose="batch",
                    extra_headers={
                        "Idempotency-Key": str(intent["upload_idempotency_key"])
                    },
                )
        except Exception as exc:
            raise audited.RunFailure(
                "Batch input upload returned no receipt; retry submit with the "
                "same durable idempotency key: " + _redact_exception(exc, key)
            ) from exc
        receipt = _jsonable(receipt_obj)
        input_file_id = str(receipt.get("id") or "")
        if not input_file_id:
            raise audited.RunFailure("Batch input upload receipt has no file id")
        intent["status"] = "input_file_uploaded"
        intent["input_file_id"] = input_file_id
        intent["input_file_receipt"] = receipt
        intent["input_file_uploaded_at"] = utc_now()
        atomic_write_json(intent_path, intent)

    batch_id = str(intent.get("batch_id") or "")
    if not batch_id:
        metadata = {
            "protocol": "openai56-f2-k2-v1",
            "config": config_sha[:32],
            "pair_arm": str(args.pair_arm_key),
            "model": args.model,
            "shard": str(shard_index),
        }
        try:
            receipt_obj = client.batches.create(
                input_file_id=input_file_id,
                endpoint=BATCH_ENDPOINT,
                completion_window=COMPLETION_WINDOW,
                metadata=metadata,
                extra_headers={"Idempotency-Key": str(intent["batch_idempotency_key"])},
            )
        except Exception as exc:
            raise audited.RunFailure(
                "Batch creation returned no receipt; retry submit with the same "
                "durable idempotency key: " + _redact_exception(exc, key)
            ) from exc
        receipt = _jsonable(receipt_obj)
        batch_id = str(receipt.get("id") or "")
        if not batch_id:
            raise audited.RunFailure("Batch creation receipt has no batch id")
        intent["status"] = "batch_created"
        intent["batch_id"] = batch_id
        intent["batch_receipt"] = receipt
        intent["batch_created_at"] = utc_now()
        atomic_write_json(intent_path, intent)

    event = {
        "schema": SCHEMA,
        "event_type": "batch_submitted",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "shard_index": shard_index,
        "shard_count": len(shards),
        "shard_input_tokens": int(shard["input_tokens"]),
        "request_count": len(shard_requests),
        "request_commitment_sha256": commitment,
        "request_file": file_record(request_path),
        "request_specs": shard_specs,
        "cost_gate": file_record(out / "openai_batch_cost_gate.json"),
        "projected_worst_case_usd": projection["exact_worst_case_total_usd"],
        "batch": intent["batch_receipt"],
    }
    JsonlJournal(out / "openai_batch_events.jsonl").append(event)
    intent_path.unlink()
    return {
        "status": "submitted",
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "shard_index": shard_index,
        "shard_count": len(shards),
        "shard_input_tokens": int(shard["input_tokens"]),
        "requests": len(shard_requests),
        "projected_worst_case_usd": projection["exact_worst_case_total_usd"],
    }


def _retrieve_active(
    args: argparse.Namespace, *, out: Path, config_sha: str
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    active = _active_submission(_batch_events(out, config_sha))
    if active is None:
        raise audited.RunFailure("there is no unharvested Batch")
    client, key = _client(args)
    try:
        value = client.batches.retrieve(str(active["batch_id"]))
    except Exception as exc:
        raise audited.RunFailure(
            "Batch status retrieval failed: " + _redact_exception(exc, key)
        ) from exc
    batch = _jsonable(value)
    if str(batch.get("id") or "") != str(active["batch_id"]):
        raise audited.RunFailure("retrieved Batch id differs from submitted id")
    event = {
        "schema": SCHEMA,
        "event_type": "batch_status",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "batch_id": active["batch_id"],
        "shard_index": active.get("shard_index"),
        "shard_count": active.get("shard_count"),
        "status": batch.get("status"),
        "request_counts": batch.get("request_counts"),
        "output_file_id": batch.get("output_file_id"),
        "error_file_id": batch.get("error_file_id"),
        "batch": batch,
    }
    JsonlJournal(out / "openai_batch_events.jsonl").append(event)
    return active, batch, client


def _status(args: argparse.Namespace, *, out: Path, config_sha: str) -> dict[str, Any]:
    active, batch, _client_value = _retrieve_active(
        args, out=out, config_sha=config_sha
    )
    return {
        "status": str(batch.get("status") or "unknown"),
        "batch_id": active["batch_id"],
        "shard_index": active.get("shard_index"),
        "shard_count": active.get("shard_count"),
        "request_counts": batch.get("request_counts"),
        "output_file_id": batch.get("output_file_id"),
        "error_file_id": batch.get("error_file_id"),
    }


def _download_bytes(client: Any, file_id: str) -> bytes:
    value = client.files.content(file_id)
    if isinstance(value, bytes):
        return value
    if hasattr(value, "read"):
        data = value.read()
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode("utf-8")
    content = getattr(value, "content", None)
    if isinstance(content, bytes):
        return content
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    raise audited.RunFailure("downloaded OpenAI file has no readable bytes")


def _download_or_verify(client: Any, *, file_id: str, path: Path) -> bytes:
    receipt_path = path.with_name(path.name + ".download.json")
    if path.is_file():
        if not receipt_path.is_file():
            raise audited.RunFailure(
                f"persisted raw Batch file {path.name} has no download receipt"
            )
        receipt = load_json(receipt_path, "raw Batch download receipt")
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("file_id") != file_id
            or receipt.get("sha256") != sha256_file(path)
        ):
            raise audited.RunFailure(
                f"persisted raw Batch file {path.name} failed receipt verification"
            )
        return path.read_bytes()
    data = _download_bytes(client, file_id)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    atomic_write_json(
        receipt_path,
        {
            "schema": SCHEMA,
            "downloaded_at": utc_now(),
            "file_id": file_id,
            "sha256": sha256_file(path),
            "bytes": len(data),
        },
    )
    return data


def _parse_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise audited.RunFailure(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise audited.RunFailure(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise audited.RunFailure(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _text_parts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            result.append(text)
    return result


def native_response_metadata(body: Mapping[str, Any]) -> dict[str, Any]:
    status = str(body.get("status") or "")
    details = body.get("incomplete_details")
    incomplete_reason = (
        str(details.get("reason") or "") if isinstance(details, Mapping) else ""
    )
    error = body.get("error")
    error_code = (
        str(error.get("code") or error.get("type") or "")
        if isinstance(error, Mapping)
        else ""
    )
    refusal_texts: list[str] = []
    output_types: list[str] = []
    message_statuses: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, Mapping):
            continue
        output_types.append(str(item.get("type") or "missing"))
        if item.get("type") == "message":
            message_statuses.append(str(item.get("status") or "missing"))
            for content in item.get("content") or []:
                if not isinstance(content, Mapping):
                    continue
                if content.get("type") == "refusal":
                    value = content.get("refusal")
                    if isinstance(value, str) and value:
                        refusal_texts.append(value)
    content_filter = incomplete_reason in {
        "content_filter",
        "safety",
    } or error_code in {"content_filter", "safety"}
    return {
        "native_status": status,
        "native_incomplete_reason": incomplete_reason or None,
        "native_error_code": error_code or None,
        "native_output_type_counts": dict(Counter(output_types)),
        "native_message_status_counts": dict(Counter(message_statuses)),
        "native_refusal_present": bool(refusal_texts),
        "native_refusal_count": len(refusal_texts),
        "native_content_filter": content_filter,
    }


def normalize_responses_body(body: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a native Responses object for the audited candidate parser."""

    metadata = native_response_metadata(body)
    output_texts: list[str] = []
    refusal_texts: list[str] = []
    reasoning_texts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content") or []:
                if not isinstance(content, Mapping):
                    continue
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        output_texts.append(text)
                elif content.get("type") == "refusal":
                    refusal = content.get("refusal")
                    if isinstance(refusal, str):
                        refusal_texts.append(refusal)
        elif item_type == "reasoning":
            reasoning_texts.extend(_text_parts(item.get("summary")))
            reasoning_texts.extend(_text_parts(item.get("content")))
    status = str(body.get("status") or "")
    incomplete_reason = metadata["native_incomplete_reason"]
    if metadata["native_content_filter"] or refusal_texts:
        finish_reason = "content_filter"
    elif incomplete_reason == "max_output_tokens":
        finish_reason = "length"
    elif status == "completed":
        finish_reason = "stop"
    elif status:
        finish_reason = status
    else:
        finish_reason = "unknown"

    usage_raw = body.get("usage")
    if not isinstance(usage_raw, Mapping):
        raise ResponseContractError("native Responses object has no usage")
    input_tokens = usage_raw.get("input_tokens")
    output_tokens = usage_raw.get("output_tokens")
    total_tokens = usage_raw.get("total_tokens")
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("total_tokens", total_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResponseContractError(f"native usage.{name} is invalid")
    if total_tokens != input_tokens + output_tokens:
        raise ResponseContractError("native Responses token usage is inconsistent")
    created = body.get("created_at")
    return {
        "id": body.get("id"),
        "model": body.get("model"),
        "created": created,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "\n".join(output_texts),
                    "reasoning_content": "\n".join(reasoning_texts),
                    "refusal": "\n".join(refusal_texts) or None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "_native_responses_metadata": metadata,
    }


def _failed_evaluation() -> dict[str, Any]:
    return {
        "compiled": False,
        "passed": False,
        "completion_attestation_id": audited.REQUIRED_ATTESTATION_ID,
        "completion_attestation_enforced": False,
        "completion_attestation_satisfied_all_runs": False,
        "stability_runs": [],
    }


def _evaluate_terminal(
    args: argparse.Namespace,
    *,
    evaluator: Any,
    plan: Mapping[str, Any],
    terminal: Any,
    task_id: str,
    sample_index: int,
) -> tuple[dict[str, Any], bool]:
    if not terminal.candidate_valid:
        return _failed_evaluation(), False
    return (
        audited.evaluate_candidate_stably(
            evaluator,
            code=terminal.code,
            tests=plan["row"]["acceptance_tests"],
            task_id=task_id,
            sample_index=sample_index,
            stability_runs=args.eval_stability_runs,
            timeout=args.eval_timeout_seconds,
        ),
        True,
    )


def _result_by_custom_id(
    output_rows: Sequence[Mapping[str, Any]],
    error_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source, rows in (("output", output_rows), ("error", error_rows)):
        for row in rows:
            custom_id = str(row.get("custom_id") or "")
            if not custom_id or custom_id in result:
                raise audited.RunFailure(
                    "Batch output/error files have missing or duplicate custom_id"
                )
            value = dict(row)
            value["_source_file_kind"] = source
            result[custom_id] = value
    return result


def _actual_cost(model: str, terminals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    for row in terminals:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            raise audited.RunFailure("terminal response has no usage")
        input_tokens += int(usage["prompt_tokens"])
        output_tokens += int(usage["completion_tokens"])
    prices = BATCH_PRICES[model]
    input_usd = _money(input_tokens, prices["input_usd_per_million"])
    output_usd = _money(output_tokens, prices["output_usd_per_million"])
    return {
        "schema": "openai56-batch-actual-cost-estimate-v1",
        "model": model,
        "terminal_responses": len(terminals),
        "provider_reported_input_tokens": input_tokens,
        "provider_reported_output_tokens": output_tokens,
        "input_usd_per_million": str(prices["input_usd_per_million"]),
        "output_usd_per_million": str(prices["output_usd_per_million"]),
        "estimated_input_usd": _money_text(input_usd),
        "estimated_output_usd": _money_text(output_usd),
        "estimated_total_usd": _money_text(input_usd + output_usd),
        "batch_discount_fraction": "0.50",
        "estimate_not_invoice": True,
        "price_source": PRICE_SOURCE,
    }


def _transparency(
    terminals: Sequence[Mapping[str, Any]],
    *,
    expected_slots: int,
    request_error_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(row.get("native_status") or "missing") for row in terminals)
    incomplete = Counter(
        str(row.get("native_incomplete_reason") or "none") for row in terminals
    )
    finish = Counter(str(row.get("finish_reason") or "missing") for row in terminals)
    terminal_reasons = Counter(
        str(row.get("terminal_reason") or "missing") for row in terminals
    )
    refusals = sum(bool(row.get("native_refusal_present")) for row in terminals)
    filters = sum(bool(row.get("native_content_filter")) for row in terminals)
    policy_slots = sum(
        bool(row.get("native_refusal_present"))
        or bool(row.get("native_content_filter"))
        for row in terminals
    )
    length = sum(row.get("finish_reason") == "length" for row in terminals)
    request_error_codes = Counter(
        str(row.get("native_error_code") or "missing") for row in request_error_rows
    )
    policy_request_errors = sum(
        bool(row.get("native_content_filter")) for row in request_error_rows
    )
    observed = len(terminals)
    policy_affected_expected_slots = policy_slots + policy_request_errors
    dominated = (
        expected_slots > 0 and policy_affected_expected_slots / expected_slots > 0.5
    )
    complete = observed == expected_slots and not request_error_rows
    if not complete:
        assessment = "invalid_incomplete_provider_slot_coverage"
    elif dominated:
        assessment = "invalid_refusal_or_filter_dominated"
    elif refusals or filters or policy_request_errors:
        assessment = "policy_contaminated_not_unqualified_capability_ceiling"
    else:
        assessment = "complete_no_observed_policy_refusals"
    return {
        "schema": "openai56-native-response-transparency-v1",
        "expected_logical_slots": expected_slots,
        "terminal_provider_responses": observed,
        "provider_request_errors": len(request_error_rows),
        "provider_request_error_code_counts": dict(request_error_codes),
        "provider_request_content_filter_errors": policy_request_errors,
        "complete_slot_coverage": complete,
        "native_response_status_counts": dict(statuses),
        "native_incomplete_reason_counts": dict(incomplete),
        "normalized_finish_reason_counts": dict(finish),
        "terminal_reason_counts": dict(terminal_reasons),
        "native_refusal_slots": refusals,
        "native_content_filter_slots": filters,
        "native_refusal_or_content_filter_slots": policy_slots,
        "length_slots": length,
        "refusal_or_filter_dominated": dominated,
        "capability_metric_assessment": assessment,
        "ceiling_claim_allowed": (
            complete and refusals == 0 and filters == 0 and policy_request_errors == 0
        ),
        "pass_denominator_never_conditioned_on_nonrefusal": True,
        "conditional_nonrefusal_metric_not_used": True,
    }


def _write_progress_or_summary(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
    evaluator_record: Mapping[str, Any],
    batch_id: str,
    request_error_rows: Sequence[Mapping[str, Any]],
    all_shards_harvested: bool,
) -> dict[str, Any]:
    terminals = _journal_rows(
        out, "terminal_slots.jsonl", config_sha, unique="custom_id"
    )
    outcomes = _journal_rows(out, "outcomes.jsonl", config_sha, unique="custom_id")
    terminal_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"])) for row in terminals
    }
    outcome_keys = {
        _slot_key(str(row["task_id"]), int(row["sample_index"])) for row in outcomes
    }
    if terminal_keys != outcome_keys:
        raise audited.RunFailure("terminal and outcome logical-slot sets differ")
    expected = len(plans) * K
    transparency = _transparency(
        terminals,
        expected_slots=expected,
        request_error_rows=request_error_rows,
    )
    atomic_write_json(out / "openai_native_response_report.json", transparency)
    cost = _actual_cost(args.model, terminals)
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "config_sha256": config_sha,
        "batch_id": batch_id,
        "model": args.model,
        "pair_arm_key": args.pair_arm_key,
        "tasks": len(plans),
        "k": K,
        "expected_logical_slots": expected,
        "terminal_provider_responses": len(terminals),
        "provider_request_errors": len(request_error_rows),
        "fixed_max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
        "transparency": transparency,
        "actual_cost_estimate": cost,
        "evaluator": dict(evaluator_record),
    }
    if len(terminals) != expected or request_error_rows:
        progress_status = (
            "invalid_incomplete_provider_slot_coverage"
            if all_shards_harvested
            else "in_progress_shards"
        )
        progress = {
            **base,
            "status": progress_status,
            "updated_at": utc_now(),
            "pass_at_2": None,
            "compile_at_2": None,
            "pass_at_k": None,
            "compile_at_k": None,
            "missing_terminal_slots": expected - len(terminals),
        }
        atomic_write_json(out / "progress.json", progress)
        provenance["status"] = progress["status"]
        provenance["updated_at"] = progress["updated_at"]
        atomic_write_json(out / "provenance.json", provenance)
        return progress

    by_slot = {(str(row["task_id"]), int(row["sample_index"])): row for row in outcomes}
    expected_keys = {
        (str(plan["task_id"]), sample) for plan in plans for sample in range(K)
    }
    if set(by_slot) != expected_keys:
        raise audited.RunFailure("complete outcome schedule does not match sealed K")
    task_results: list[dict[str, Any]] = []
    for plan in plans:
        task_id = str(plan["task_id"])
        candidates = [by_slot[(task_id, sample)] for sample in range(K)]
        task_results.append(
            {
                "task_id": task_id,
                "compiled": any(bool(row["compiled"]) for row in candidates),
                "passed": any(bool(row["passed"]) for row in candidates),
                "candidate_outcomes": candidates,
            }
        )
    passed = sum(bool(row["passed"]) for row in task_results)
    compiled = sum(bool(row["compiled"]) for row in task_results)
    usage = {
        "input_tokens": sum(int(row["usage"]["prompt_tokens"]) for row in terminals),
        "output_tokens": sum(
            int(row["usage"]["completion_tokens"]) for row in terminals
        ),
        "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in terminals),
    }
    pass_metric = {
        "successes": passed,
        "total": len(task_results),
        "rate": passed / len(task_results),
        "wilson_95": wilson_interval(passed, len(task_results)),
    }
    compile_metric = {
        "successes": compiled,
        "total": len(task_results),
        "rate": compiled / len(task_results),
        "wilson_95": wilson_interval(compiled, len(task_results)),
    }
    summary: dict[str, Any] = {
        **base,
        "status": "complete",
        "completed_at": utc_now(),
        "input_mode": "prematerialized_f2",
        "prompt_arm": PROMPT_ARM_LABEL,
        "pair_manifest_sha256": provenance["artifacts"]["pair_manifest"]["sha256"],
        "task_set_sha256": provenance["task_set_sha256"],
        "acceptance_test_sequence_sha256": provenance[
            "acceptance_test_sequence_sha256"
        ],
        "resolved_models": sorted(
            {str(row.get("resolved_model") or "") for row in terminals}
        ),
        "pass_at_2": pass_metric,
        "compile_at_2": compile_metric,
        "pass_at_k": pass_metric,
        "compile_at_k": compile_metric,
        "usage": usage,
        "task_results": task_results,
        "all_tasks_have_exactly_k_terminal_provider_responses": True,
        "every_terminal_response_has_exactly_one_outcome": True,
        "returned_responses_resampled": False,
        "prompt_truncation_used": False,
        "raw_native_batch_output_retained": True,
        "artifacts": {
            name: file_record(out / name)
            for name in (
                "tasks.jsonl",
                "prompts.jsonl",
                "openai56_prompt_overlay_manifest.json",
                "openai_batch_requests.jsonl",
                "openai_input_token_counts.jsonl",
                "openai_input_token_audit.json",
                "openai_batch_cost_gate.json",
                "openai_batch_events.jsonl",
                "batch_slot_attempts.jsonl",
                "terminal_slots.jsonl",
                "outcomes.jsonl",
                "openai_native_response_report.json",
            )
            if (out / name).is_file()
        },
    }
    atomic_write_json(out / "summary.json", summary)
    provenance["status"] = "complete"
    provenance["completed_at"] = summary["completed_at"]
    provenance["summary_sha256"] = sha256_file(out / "summary.json")
    atomic_write_json(out / "provenance.json", provenance)
    manifest_files = [
        "provenance.json",
        "tasks.jsonl",
        "prompts.jsonl",
        "openai56_prompt_overlay_manifest.json",
        "openai_batch_requests.jsonl",
        "openai_input_token_counts.jsonl",
        "openai_input_token_audit.json",
        "openai_batch_cost_gate.json",
        "openai_batch_events.jsonl",
        "batch_slot_attempts.jsonl",
        "terminal_slots.jsonl",
        "outcomes.jsonl",
        "openai_native_response_report.json",
        "summary.json",
    ]
    atomic_write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "created_at": utc_now(),
            "files": {
                name: file_record(out / name)
                for name in manifest_files
                if (out / name).is_file()
            },
        },
    )
    return summary


def _harvest(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    active, batch, client = _retrieve_active(args, out=out, config_sha=config_sha)
    status = str(batch.get("status") or "")
    if status not in {"completed", "failed", "expired", "cancelled"}:
        return {
            "status": status or "unknown",
            "batch_id": active["batch_id"],
            "request_counts": batch.get("request_counts"),
        }
    batch_id = str(active["batch_id"])
    output_file_id = str(batch.get("output_file_id") or "")
    error_file_id = str(batch.get("error_file_id") or "")
    output_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    raw_artifacts: dict[str, Any] = {}
    if output_file_id:
        path = out / f"openai_batch_output_{batch_id}.jsonl"
        data = _download_or_verify(client, file_id=output_file_id, path=path)
        output_rows = _parse_jsonl_bytes(data, "OpenAI Batch output")
        raw_artifacts["output"] = file_record(path)
    if error_file_id:
        path = out / f"openai_batch_errors_{batch_id}.jsonl"
        data = _download_or_verify(client, file_id=error_file_id, path=path)
        error_rows = _parse_jsonl_bytes(data, "OpenAI Batch error file")
        raw_artifacts["errors"] = file_record(path)
    if not output_rows and not error_rows:
        raise audited.RunFailure("terminal Batch has no output or error rows")

    all_results = _result_by_custom_id(output_rows, error_rows)
    specs = active.get("request_specs")
    if not isinstance(specs, list) or not specs:
        raise audited.RunFailure("submitted event has no shard request schedule")
    if len(specs) != int(active.get("request_count") or -1):
        raise audited.RunFailure("submitted shard request count is inconsistent")
    expected = {str(spec["custom_id"]): spec for spec in specs}
    unknown = set(all_results) - set(expected)
    if unknown:
        raise audited.RunFailure("Batch returned unknown custom IDs")
    missing_records = set(expected) - set(all_results)
    if missing_records:
        raise audited.RunFailure(
            f"terminal Batch omitted {len(missing_records)} request records"
        )

    if not args.expected_evaluator_sha256.strip():
        raise PreflightError("--expected-evaluator-sha256 is required")
    if not args.expected_dart_sha256.strip():
        raise PreflightError("--expected-dart-sha256 is required")
    evaluator_module, evaluator_record = audited.import_evaluator(
        args.evaluator_module,
        args.expected_evaluator_sha256,
        dart_binary=args.dart,
        expected_dart_hash=args.expected_dart_sha256,
        validate_dart=True,
    )
    if evaluator_record["sha256"] != provenance["evaluator"]["sha256"]:
        raise PreflightError("evaluator changed after sealed preflight")
    evaluator = evaluator_module.evaluate_dart_jit_tests_detail
    by_task = {str(plan["task_id"]): plan for plan in plans}
    attempt_rows = _journal_rows(
        out, "batch_slot_attempts.jsonl", config_sha, unique="custom_id"
    )
    terminal_rows = _journal_rows(
        out, "terminal_slots.jsonl", config_sha, unique="custom_id"
    )
    outcome_rows = _journal_rows(out, "outcomes.jsonl", config_sha, unique="custom_id")
    attempts = {str(row["custom_id"]): row for row in attempt_rows}
    terminals = {str(row["custom_id"]): row for row in terminal_rows}
    outcomes = {str(row["custom_id"]): row for row in outcome_rows}
    attempt_journal = JsonlJournal(out / "batch_slot_attempts.jsonl")
    terminal_journal = JsonlJournal(out / "terminal_slots.jsonl")
    outcome_journal = JsonlJournal(out / "outcomes.jsonl")
    error_path = out / "openai_batch_request_errors.jsonl"
    prior_error_rows = (
        load_jsonl(error_path, "OpenAI Batch request errors")
        if error_path.is_file()
        else []
    )
    request_error_records: list[dict[str, Any]] = []
    error_by_custom_id: dict[str, dict[str, Any]] = {}
    for error_row in prior_error_rows:
        custom_id = str(error_row.get("custom_id") or "")
        if (
            error_row.get("schema") != SCHEMA
            or error_row.get("config_sha256") != config_sha
            or not custom_id
            or custom_id in error_by_custom_id
        ):
            raise audited.RunFailure("foreign/duplicate persisted Batch request error")
        error_by_custom_id[custom_id] = error_row

    for custom_id in sorted(
        expected,
        key=lambda value: (
            int(expected[value]["sample_index"]),
            int(expected[value]["task_index"]),
        ),
    ):
        spec = expected[custom_id]
        row = all_results[custom_id]
        response = row.get("response")
        status_code = (
            response.get("status_code") if isinstance(response, Mapping) else None
        )
        body = response.get("body") if isinstance(response, Mapping) else None
        error = row.get("error")
        if (
            status_code != 200
            or not isinstance(body, Mapping)
            or error not in (None, {})
        ):
            error_payload: Any = error
            if error_payload in (None, {}) and isinstance(body, Mapping):
                error_payload = body.get("error", body)
            error_code = ""
            if isinstance(error_payload, Mapping):
                error_code = str(
                    error_payload.get("code")
                    or error_payload.get("type")
                    or error_payload.get("error")
                    or ""
                )
            error_record = {
                "schema": SCHEMA,
                "config_sha256": config_sha,
                "batch_id": batch_id,
                "custom_id": custom_id,
                "task_id": spec["task_id"],
                "task_index": spec["task_index"],
                "sample_index": spec["sample_index"],
                "status_code": status_code,
                "error": error,
                "native_error_code": error_code or None,
                "native_content_filter": error_code
                in {"content_filter", "safety", "policy_violation"},
                "native_batch_record_sha256": stable_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "_source_file_kind"
                    }
                ),
                "source_file_kind": row["_source_file_kind"],
            }
            prior_error = error_by_custom_id.get(custom_id)
            if prior_error is not None and stable_sha256(prior_error) != stable_sha256(
                error_record
            ):
                raise audited.RunFailure("persisted Batch request error changed")
            error_by_custom_id[custom_id] = error_record
            continue
        normalized = normalize_responses_body(body)
        terminal = audited.classify_terminal_provider_response(
            normalized,
            expected_model=args.model,
            max_prompt_tokens=PROVIDER_INPUT_TOKEN_CAP,
            requested_max_tokens=MAX_OUTPUT_TOKENS,
        )
        native = native_response_metadata(body)
        if custom_id not in attempts:
            attempt = {
                "schema": SCHEMA,
                "record_type": "batch_slot_attempt",
                "recorded_at": utc_now(),
                "config_sha256": config_sha,
                "batch_id": batch_id,
                "custom_id": custom_id,
                "task_id": str(spec["task_id"]),
                "task_index": int(spec["task_index"]),
                "sample_index": int(spec["sample_index"]),
                "status_code": status_code,
                "response_id": terminal.response_id,
                "resolved_model": terminal.response_model,
                "finish_reason": terminal.finish_reason,
                **native,
                "usage": terminal.usage,
                "normalized_response": normalized,
                "native_batch_record_sha256": stable_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "_source_file_kind"
                    }
                ),
                "raw_native_response_retained_in": raw_artifacts[
                    row["_source_file_kind"]
                ],
            }
            attempt_journal.append(attempt)
            attempts[custom_id] = attempt
        if custom_id not in outcomes:
            evaluation, performed = _evaluate_terminal(
                args,
                evaluator=evaluator,
                plan=by_task[str(spec["task_id"])],
                terminal=terminal,
                task_id=str(spec["task_id"]),
                sample_index=int(spec["sample_index"]),
            )
            outcome = {
                "schema": SCHEMA,
                "record_type": "logical_slot_outcome",
                "evaluated_at": utc_now(),
                "config_sha256": config_sha,
                "batch_id": batch_id,
                "custom_id": custom_id,
                "task_id": str(spec["task_id"]),
                "task_index": int(spec["task_index"]),
                "sample_index": int(spec["sample_index"]),
                "requested_max_output_tokens": MAX_OUTPUT_TOKENS,
                "response_id": terminal.response_id,
                "finish_reason": terminal.finish_reason,
                **native,
                "candidate_valid": bool(terminal.candidate_valid),
                "terminal_reason": terminal.terminal_reason,
                "code_sha256": terminal.code_sha256,
                "evaluator_sha256": evaluator_record["sha256"],
                "evaluator_entrypoint": evaluator_record["entrypoint"],
                "evaluation_performed": performed,
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
            }
            outcome_journal.append(outcome)
            outcomes[custom_id] = outcome
        if custom_id not in terminals:
            terminal_row = {
                "schema": SCHEMA,
                "record_type": "terminal_logical_slot",
                "recorded_at": utc_now(),
                "config_sha256": config_sha,
                "batch_id": batch_id,
                "custom_id": custom_id,
                "task_id": str(spec["task_id"]),
                "task_index": int(spec["task_index"]),
                "sample_index": int(spec["sample_index"]),
                "response_id": terminal.response_id,
                "resolved_model": terminal.response_model,
                "finish_reason": terminal.finish_reason,
                **native,
                "candidate_valid": bool(terminal.candidate_valid),
                "terminal_reason": terminal.terminal_reason,
                "code": terminal.code,
                "code_sha256": terminal.code_sha256,
                "usage": terminal.usage,
            }
            terminal_journal.append(terminal_row)
            terminals[custom_id] = terminal_row

    request_error_records = [
        error_by_custom_id[key] for key in sorted(error_by_custom_id)
    ]
    atomic_write_jsonl(error_path, request_error_records)
    result = _write_progress_or_summary(
        args,
        out=out,
        plans=plans,
        config_sha=config_sha,
        provenance=provenance,
        evaluator_record=evaluator_record,
        batch_id=batch_id,
        request_error_rows=request_error_records,
        all_shards_harvested=(
            int(active["shard_index"]) + 1 == int(active["shard_count"])
        ),
    )
    event = {
        "schema": SCHEMA,
        "event_type": "batch_harvested",
        "recorded_at": utc_now(),
        "config_sha256": config_sha,
        "batch_id": batch_id,
        "shard_index": int(active["shard_index"]),
        "shard_count": int(active["shard_count"]),
        "batch_status": status,
        "request_count": len(all_results),
        "terminal_provider_responses": len(terminals),
        "provider_request_errors": len(request_error_records),
        "raw_native_artifacts": raw_artifacts,
        "result_status": result["status"],
    }
    JsonlJournal(out / "openai_batch_events.jsonl").append(event)
    return result


def _dispatch(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if args.action == "submit":
        return _submit(args, out=out, plans=plans, config_sha=config_sha)
    if args.action == "status":
        return _status(args, out=out, config_sha=config_sha)
    if args.action == "harvest":
        return _harvest(
            args,
            out=out,
            plans=plans,
            config_sha=config_sha,
            provenance=provenance,
        )
    if args.action == "auto":
        active = _active_submission(_batch_events(out, config_sha))
        if active is None:
            return _submit(args, out=out, plans=plans, config_sha=config_sha)
        status = _status(args, out=out, config_sha=config_sha)
        if status["status"] in {"completed", "failed", "expired", "cancelled"}:
            return _harvest(
                args,
                out=out,
                plans=plans,
                config_sha=config_sha,
                provenance=provenance,
            )
        return status
    raise audited.RunFailure(f"unsupported action: {args.action}")


def main(argv: Sequence[str] | None = None) -> int:
    install_hooks()
    args = parse_args(argv)
    out = audited.choose_output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    try:
        audited.enforce_output_state_policy(args, out)
    except Exception as exc:
        print(
            f"OPENAI56_BATCH_FAILED_CLOSED error={type(exc).__name__}: "
            f"{exc} out={out}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    with audited.RunLock(out / ".run.lock"):
        try:
            tokenizer, plans, prompt_map, config_sha, provenance = prepare_run(
                args, out
            )
            del tokenizer, prompt_map
            specs = request_specs(plans)
            requests = batch_requests(args, plans, specs)
            _write_or_verify_jsonl(out / "openai_batch_requests.jsonl", requests)
            absolute = absolute_preflight_cost(args, len(requests))
            atomic_write_json(out / "openai_batch_absolute_cost_bound.json", absolute)
            print(
                f"OPENAI56_BATCH_PREFLIGHT_OK model={args.model} "
                f"pair_arm={args.pair_arm_key} tasks={len(plans)} K={K} "
                f"requests={len(requests)} max_output={MAX_OUTPUT_TOKENS} "
                f"reasoning={REASONING_EFFORT} "
                f"absolute_bound_usd={absolute['exact_worst_case_total_usd']} "
                f"out={out}",
                flush=True,
            )
            if args.action == "preflight":
                provenance["status"] = "preflight_only_complete_no_api_calls"
                provenance["completed_at"] = utc_now()
                provenance["absolute_cost_bound"] = file_record(
                    out / "openai_batch_absolute_cost_bound.json"
                )
                provenance["exact_batch_requests"] = file_record(
                    out / "openai_batch_requests.jsonl"
                )
                atomic_write_json(out / "provenance.json", provenance)
                return 0
            result = _dispatch(
                args,
                out=out,
                plans=plans,
                config_sha=config_sha,
                provenance=provenance,
            )
        except Exception as exc:
            failure = {
                "schema": SCHEMA,
                "status": "failed_closed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(out / "failure.json", failure)
            print(
                f"OPENAI56_BATCH_FAILED_CLOSED error={type(exc).__name__}: "
                f"{exc} out={out}",
                file=sys.stderr,
                flush=True,
            )
            return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
