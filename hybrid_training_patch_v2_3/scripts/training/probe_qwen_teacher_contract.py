#!/usr/bin/env python3
"""Make one short paid call to validate the Qwen logprob response contract."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.qwen_direct_compact_teacher_artifact import (  # noqa: E402
    DEFAULT_MODEL,
    NEGATIVE_TAIL_TOLERANCE,
    TOP_LOGPROBS,
    ArtifactError,
    StudentTokenizerBinding,
    audit_candidate_tokens,
    atomic_write_json,
    backend_identity,
    build_messages,
    file_record,
    load_f2_prompt_contract,
    load_verified_prompt_rows,
    normalize_response,
    sha256_text,
    stable_sha256,
    utc_now,
    validate_alibaba_model_studio_base_url,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prompt-jsonl", required=True, type=Path)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--expected-prompt-rows", required=True, type=int)
    parser.add_argument("--prompt-manifest", required=True, type=Path)
    parser.add_argument("--expected-prompt-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_ENDPOINT", ""))
    parser.add_argument(
        "--token-plan-automation-authorized",
        action="store_true",
        help=(
            "Attest that Alibaba explicitly authorized automated research use "
            "of this account's Token Plan endpoint."
        ),
    )
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--student-tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-student-tokenizer-sha256", required=True)
    parser.add_argument("--student-eos-token-id", required=True, type=int)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--task-id",
        default="",
        help=(
            "Probe this exact task from the sealed prompt set. When omitted, "
            "the shortest sealed prompt is used."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=128,
        help="Probe-only thinking budget; production uses 8192.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("require_top5", "sequence_only"),
        default="require_top5",
        help=(
            "require_top5 fails closed unless every returned content token has "
            "raw bytes and exactly five alternatives. sequence_only explicitly "
            "tests only the sampled-sequence API contract."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request Qwen thinking mode. Use --no-enable-thinking only for a "
            "standard-endpoint logprob/parser control probe."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_url = validate_alibaba_model_studio_base_url(
        args.base_url,
        token_plan_automation_authorized=(
            args.token_plan_automation_authorized
        ),
    )
    if args.objective_mode == "require_top5" and args.enable_thinking:
        raise ArtifactError(
            "require_top5 cannot probe thinking-mode content logprobs: they are "
            "conditioned on hidden reasoning tokens unavailable to the student; "
            "pass --no-enable-thinking"
        )
    if args.timeout_seconds <= 0 or not 1 <= args.max_tokens <= 12288:
        raise ArtifactError("probe timeout/max-token budget is invalid")
    if args.thinking_budget <= 0 or args.thinking_budget >= args.max_tokens:
        raise ArtifactError(
            "--thinking-budget must be positive and below --max-tokens"
        )
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ArtifactError(
            f"API key environment variable {args.api_key_env!r} is not set"
        )
    prompts, prompt_record = load_verified_prompt_rows(
        args.prompt_jsonl,
        expected_sha256=args.expected_prompt_sha256,
        expected_rows=args.expected_prompt_rows,
    )
    binding = StudentTokenizerBinding.from_file(
        args.student_tokenizer_json,
        expected_sha256=args.expected_student_tokenizer_sha256,
        eos_token_id=args.student_eos_token_id,
    )
    system_prompt, prompt_manifest_record, _prompt_manifest = (
        load_f2_prompt_contract(
            args.prompt_manifest,
            expected_sha256=args.expected_prompt_manifest_sha256,
            prompt_record=prompt_record,
            expected_rows=args.expected_prompt_rows,
            student_tokenizer_sha256=binding.tokenizer_record["sha256"],
        )
    )
    system_prompt_sha256 = sha256_text(system_prompt)
    if any(
        prompt.representation_schema != "lossless-semantic-f2"
        or prompt.system_prompt_sha256 != system_prompt_sha256
        for prompt in prompts
    ):
        raise ArtifactError("probe rows differ from the manifest-bound F2 prompt")
    if args.task_id:
        selected = [row for row in prompts if row.task_id == args.task_id]
        if len(selected) != 1:
            raise ArtifactError(
                f"--task-id {args.task_id!r} is not in the sealed prompt set"
            )
        prompt = selected[0]
        selection = "explicit_task_id"
    else:
        # The shortest sealed prompt minimizes paid input while testing the
        # endpoint, request fields, bytes, and returned backend identity.
        prompt = min(prompts, key=lambda row: len(row.text))
        selection = "shortest_sealed_prompt"
    messages = build_messages(system_prompt, prompt)
    request_parameters = {
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": int(args.max_tokens),
        "seed": 44,
        "extra_body": {
            "enable_thinking": bool(args.enable_thinking),
            "top_k": 101,
        },
    }
    if args.objective_mode == "require_top5":
        request_parameters.update(
            {"logprobs": True, "top_logprobs": TOP_LOGPROBS}
        )
    if args.enable_thinking:
        request_parameters["extra_body"]["thinking_budget"] = int(
            args.thinking_budget
        )
    try:
        from openai import OpenAI
    except Exception as exc:
        raise ArtifactError("the openai package is required for the probe") from exc
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=float(args.timeout_seconds),
        max_retries=0,
    )
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            **request_parameters,
        )
        candidate = normalize_response(
            response,
            task_id=prompt.task_id,
            sample_index=0,
            prompt_sha256=stable_sha256(messages),
            requested_model=args.model,
            request_parameters=request_parameters,
            required_function="fn0",
        )
    except Exception as exc:
        body = getattr(exc, "body", None)
        if not isinstance(body, dict):
            body = {}
        provider_message = str(body.get("message") or "")
        contract_error = (
            str(exc)
            if isinstance(exc, ArtifactError)
            else ""
        )
        failure_artifact = {
            "schema": "qwen-direct-compact-teacher-contract-probe-v1",
            "created_at": utc_now(),
            "contract_ok": False,
            "production_collection_started": False,
            "prompt_artifact": prompt_record,
            "prompt_manifest": prompt_manifest_record,
            "selected_task_id": prompt.task_id,
            "task_selection": selection,
            "request_messages_sha256": stable_sha256(messages),
            "requested_model": args.model,
            "objective_mode": args.objective_mode,
            "base_url": args.base_url.rstrip("/"),
            "request_parameters": request_parameters,
            "provider_error": {
                "exception_type": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None),
                "provider_code": body.get("code") or getattr(exc, "code", None),
                "request_id": getattr(exc, "request_id", None),
                "message_length": len(provider_message),
                "message_sha256": hashlib.sha256(
                    provider_message.encode("utf-8")
                ).hexdigest(),
            },
            "response_contract_error": contract_error,
        }
        atomic_write_json(args.output, failure_artifact)
        print(
            "QWEN_CONTRACT_PROBE_PROVIDER_ERROR "
            f"type={type(exc).__name__} "
            f"status={getattr(exc, 'status_code', None)} "
            f"code={failure_artifact['provider_error']['provider_code']} "
            f"output={args.output}",
            flush=True,
        )
        return 2

    validation_errors: list[str] = []
    tokens = candidate["chosen_tokens_with_top_logprobs"]
    chosen_raw_bytes_complete = bool(tokens)
    exact_top5_each_position = bool(tokens)
    top5_raw_bytes_present = bool(tokens)
    finite_top5_logprobs = bool(tokens)
    nonnegative_inferred_tail_each_position = bool(tokens)
    inferred_tail_masses: list[float] = []
    if not tokens and args.objective_mode == "require_top5":
        validation_errors.append("no_content_token_logprobs")
    reconstructed = bytearray()
    for index, token in enumerate(tokens):
        raw = token.get("bytes")
        top = token.get("top_logprobs")
        if not isinstance(raw, list):
            chosen_raw_bytes_complete = False
            if args.objective_mode == "require_top5":
                validation_errors.append(f"token_{index}_missing_raw_bytes")
            raw = []
        if not isinstance(top, list) or len(top) != TOP_LOGPROBS:
            exact_top5_each_position = False
            if args.objective_mode == "require_top5":
                validation_errors.append(f"token_{index}_not_exact_top5")
            top = top if isinstance(top, list) else []
        reconstructed.extend(int(value) for value in raw)
        for alternative in top:
            if alternative.get("bytes") is None:
                top5_raw_bytes_present = False
                if args.objective_mode == "require_top5":
                    validation_errors.append(
                        f"token_{index}_top_alternative_missing_raw_bytes"
                    )
            try:
                finite = math.isfinite(float(alternative.get("logprob")))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                finite_top5_logprobs = False
                if args.objective_mode == "require_top5":
                    validation_errors.append(
                        f"token_{index}_top_alternative_nonfinite_logprob"
                    )
        if len(top) == TOP_LOGPROBS and all(
            isinstance(alternative, dict)
            and isinstance(alternative.get("logprob"), (int, float))
            and math.isfinite(float(alternative["logprob"]))
            for alternative in top
        ):
            inferred_tail = 1.0 - math.fsum(
                math.exp(float(alternative["logprob"]))
                for alternative in top
            )
            inferred_tail_masses.append(inferred_tail)
            if inferred_tail < -NEGATIVE_TAIL_TOLERANCE:
                nonnegative_inferred_tail_each_position = False
                if args.objective_mode == "require_top5":
                    validation_errors.append(
                        f"token_{index}_materially_negative_inferred_tail"
                    )
        elif args.objective_mode == "require_top5":
            nonnegative_inferred_tail_each_position = False
    raw_content = str(candidate["response"].get("raw_content") or "")
    raw_content_bytes_reconstruct = bool(
        tokens
        and chosen_raw_bytes_complete
        and bytes(reconstructed) == raw_content.encode("utf-8")
    )
    nonempty_content = bool(raw_content.strip())
    if not nonempty_content:
        validation_errors.append("empty_sampled_content")
    if (
        args.objective_mode == "require_top5"
        and not raw_content_bytes_reconstruct
    ):
        validation_errors.append("content_token_bytes_do_not_reconstruct")
    usage = candidate["response"].get("usage") or {}
    nonzero_usage = int(usage.get("total_tokens") or 0) > 0
    if not nonzero_usage:
        validation_errors.append("provider_usage_zero_or_missing")
    provider_reported_seed = candidate["response"].get(
        "provider_reported_seed"
    )
    provider_seed_honor_attested = bool(provider_reported_seed == 44)
    identity = backend_identity(candidate)
    returned_backend_identity_present = bool(
        str(identity.get("returned_model") or "")
    )
    if not returned_backend_identity_present:
        validation_errors.append("returned_backend_identity_missing")
    returned_model_matches_requested = bool(
        identity.get("returned_model") == args.model
    )
    if not returned_model_matches_requested:
        validation_errors.append("returned_model_differs_from_exact_request")
    token_mapping_audit = (
        audit_candidate_tokens(candidate, binding)
        if args.objective_mode == "require_top5"
        else None
    )
    if token_mapping_audit is not None:
        mapping_summary = token_mapping_audit["summary"]
        if not mapping_summary["chosen_mapping_complete"]:
            validation_errors.append(
                "chosen_provider_tokens_do_not_map_one_to_one_to_student"
            )
        if not mapping_summary["top_mapping_complete"]:
            validation_errors.append(
                "top5_provider_tokens_do_not_map_one_to_one_to_student"
            )
        if not mapping_summary["tail_valid"]:
            validation_errors.append("materially_negative_inferred_tail")
    validation_errors = sorted(set(validation_errors))
    artifact = {
        "schema": "qwen-direct-compact-teacher-contract-probe-v1",
        "created_at": utc_now(),
        "contract_ok": not validation_errors,
        "production_collection_started": False,
        "prompt_artifact": prompt_record,
        "prompt_manifest": prompt_manifest_record,
        "selected_task_id": prompt.task_id,
        "task_selection": selection,
        "request_messages_sha256": stable_sha256(messages),
        "requested_model": args.model,
        "objective_mode": args.objective_mode,
        "base_url": args.base_url.rstrip("/"),
        "request_parameters": request_parameters,
        "validated": {
            "one_synchronous_n1_call": True,
            "nonempty_sampled_content": nonempty_content,
            "content_token_logprobs_present": bool(tokens),
            "nonzero_usage": nonzero_usage,
            "raw_content_bytes_reconstruct": raw_content_bytes_reconstruct,
            "exact_top5_each_position": exact_top5_each_position,
            "top5_raw_bytes_present": top5_raw_bytes_present,
            "finite_logprobs": finite_top5_logprobs,
            "nonnegative_inferred_tail_each_position": (
                nonnegative_inferred_tail_each_position
            ),
            "negative_tail_tolerance": NEGATIVE_TAIL_TOLERANCE,
            "returned_backend_identity_present": (
                returned_backend_identity_present
            ),
            "returned_model_matches_requested": (
                returned_model_matches_requested
            ),
            "provider_seed_honor_attested": (
                provider_seed_honor_attested
            ),
            "provider_seed_honor_assumed": False,
        },
        "validation_errors": validation_errors,
        "response_shape": {
            "content_characters": len(raw_content),
            "reasoning_characters": len(
                str(candidate["response"].get("raw_reasoning_content") or "")
            ),
            "content_logprob_positions": len(tokens),
            "minimum_inferred_tail_mass": (
                min(inferred_tail_masses)
                if inferred_tail_masses
                else None
            ),
        },
        "candidate": candidate,
        "backend_identity": identity,
        "student_tokenizer": binding.tokenizer_record,
        "student_eos_token_id": binding.eos_token_id,
        "student_tokenizer_probe": {
            "scope": "observed_probe_content_tokens_only",
            "tokenizer_identity_proven": False,
            "mapping_audit": token_mapping_audit,
        },
    }
    atomic_write_json(args.output, artifact)
    if validation_errors:
        print(
            "QWEN_CONTRACT_PROBE_FAIL "
            f"model={identity['returned_model']} mode={args.objective_mode} "
            f"tokens={len(tokens)} "
            f"errors={','.join(validation_errors)} "
            f"output={args.output}",
            flush=True,
        )
        return 2
    print(
        "QWEN_CONTRACT_PROBE_OK "
        f"model={identity['returned_model']} mode={args.objective_mode} "
        f"tokens={len(tokens)} "
        f"finish_reason={candidate['response']['finish_reason']} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
