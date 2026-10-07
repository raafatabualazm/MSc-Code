#!/usr/bin/env python3
"""OpenAI Batch API state machine for direct-compact ChatGPT RS-SFT.

Commands are restart-safe:

``prepare`` creates one immutable JSONL batch round,
``submit`` uploads and creates the 24-hour batch,
``retrieve`` polls and downloads terminal output,
``ingest`` independently verifies returned Dart candidates, and
``finalize`` merges rounds and enforces the production coverage floor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.training.collect_chatgpt_compact_rs import (
    SCHEMA as SYNC_SCHEMA,
    atomic_json,
    atomic_jsonl,
    count_prompt_tokens,
    evaluate_dart_jit_tests_detail,
    extract_code,
    load_env_file,
    load_jsonl,
    load_object,
    object_dict,
    response_text,
    sha256_file,
    stable_sha256,
    utc_now,
    validate_failure_inputs,
    validate_serialized_inputs,
)


SCHEMA = "direct-compact-chatgpt-rs-openai-batch-v1"
TERMINAL_BATCH_STATES = {
    "completed",
    "failed",
    "expired",
    "cancelled",
}
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_BYTES = 190 * 1024 * 1024


def write_bytes_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sdk_content_bytes(value: Any) -> bytes:
    content = getattr(value, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    if hasattr(value, "read"):
        result = value.read()
        if isinstance(result, bytes):
            return result
        if isinstance(result, str):
            return result.encode("utf-8")
    if hasattr(value, "text"):
        text = value.text
        if callable(text):
            text = text()
        if isinstance(text, str):
            return text.encode("utf-8")
    raise ValueError("SDK file response cannot be converted to bytes")


def load_verified(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            continue
        for row in load_jsonl(path):
            if row.get("ok") is not True or not str(row.get("task_id") or ""):
                raise ValueError(f"{path}: contains an unverified repair row")
            rows.append(row)
    return rows


def common_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--serialized_inputs", required=True)
    parser.add_argument("--serialized_manifest", default="")
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--score_report", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--round_dir", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--prior_verified", action="append", default=[])
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--samples_per_task", type=int, default=4)
    parser.add_argument("--max_output_tokens", type=int, default=3072)
    parser.add_argument("--max_prompt_tokens", type=int, default=12000)
    parser.add_argument("--chat_overhead_reserve", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0)


def prepare_command(args: argparse.Namespace) -> int:
    if args.round <= 0 or args.samples_per_task <= 0:
        raise ValueError("round and samples_per_task must be positive")
    if args.max_output_tokens <= 0 or args.max_prompt_tokens <= 0:
        raise ValueError("token caps must be positive")
    if args.limit < 0 or args.chat_overhead_reserve < 0:
        raise ValueError("limit and chat overhead reserve must be non-negative")
    if not 0 < args.temperature <= 2:
        raise ValueError("temperature must lie in (0, 2]")

    serialized_path = Path(args.serialized_inputs).expanduser().resolve()
    manifest_path = (
        Path(args.serialized_manifest).expanduser().resolve()
        if args.serialized_manifest
        else Path(str(serialized_path) + ".manifest.json")
    )
    tokenizer_path = Path(args.tokenizer_json).expanduser().resolve()
    train_path = Path(args.train_file).expanduser().resolve()
    score_path = Path(args.score_report).expanduser().resolve()
    prediction_path = Path(args.predictions).expanduser().resolve()
    round_dir = Path(args.round_dir).expanduser().resolve()
    for path in (
        serialized_path,
        manifest_path,
        tokenizer_path,
        train_path,
        score_path,
        prediction_path,
        Path(str(prediction_path) + ".provenance.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    serialized, f2_system_prompt, _serialized_manifest = (
        validate_serialized_inputs(
            serialized_path, train_path, manifest_path, tokenizer_path
        )
    )
    _tests, task_results, _predictions = validate_failure_inputs(
        train_path=train_path,
        score_path=score_path,
        prediction_path=prediction_path,
    )
    if set(serialized) != set(task_results):
        raise ValueError("serialized inputs and score report have different tasks")
    already_solved = {
        str(row["task_id"])
        for row in load_verified(
            [Path(value).expanduser().resolve() for value in args.prior_verified]
        )
    }
    failed_ids = sorted(
        task_id
        for task_id, result in task_results.items()
        if not bool(result.get("pass_at_k")) and task_id not in already_solved
    )
    if args.limit:
        failed_ids = failed_ids[: args.limit]
    if not failed_ids:
        print("BATCH_PREPARE_NO_UNSOLVED_TASKS", flush=True)
        return 4
    prior_records = [
        {
            "path": str(Path(value).expanduser().resolve()),
            "sha256": sha256_file(Path(value).expanduser().resolve()),
        }
        for value in args.prior_verified
    ]
    prepare_config_sha256 = stable_sha256(
        {
            "round": args.round,
            "model": args.model,
            "samples_per_task": args.samples_per_task,
            "max_output_tokens": args.max_output_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "chat_overhead_reserve": args.chat_overhead_reserve,
            "temperature": args.temperature,
            "limit": args.limit,
            "selected_task_ids": failed_ids,
            "prior_verified": prior_records,
            "serialized_sha256": sha256_file(serialized_path),
            "train_sha256": sha256_file(train_path),
            "score_sha256": sha256_file(score_path),
            "predictions_sha256": sha256_file(prediction_path),
        }
    )

    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError("install tokenizers for prompt preflight") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    input_path = round_dir / "input.jsonl"
    index_path = round_dir / "request_index.jsonl"
    round_manifest_path = round_dir / "manifest.json"
    if round_manifest_path.is_file():
        manifest = load_object(round_manifest_path)
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("round") != args.round
            or manifest.get("requested_model") != args.model
            or manifest.get("prepare_config_sha256")
            != prepare_config_sha256
            or manifest.get("input", {}).get("sha256") != sha256_file(input_path)
            or manifest.get("request_index", {}).get("sha256")
            != sha256_file(index_path)
        ):
            raise ValueError("existing batch round does not match requested config")
        print(
            f"BATCH_PREPARE_RESUME round={args.round} "
            f"requests={manifest['requests']} input={input_path}",
            flush=True,
        )
        return 0
    if round_dir.exists() and any(round_dir.iterdir()):
        raise ValueError(f"round directory is non-empty: {round_dir}")
    round_dir.mkdir(parents=True, exist_ok=True)

    requests: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    prompt_counts: list[int] = []
    sample_offset = (args.round - 1) * args.samples_per_task
    for position, task_id in enumerate(failed_ids):
        messages = [
            {"role": "developer", "content": f2_system_prompt},
            {
                "role": "user",
                "content": (
                    serialized[task_id]["text"].rstrip()
                    + "\nSTUDENT_PRIVATE_VERIFIER_RESULT: no sampled student "
                    "candidate passed. Previous teacher candidates, if any, also "
                    "failed private verification.\nReturn the corrected "
                    "self-contained Dart compilation-unit fragment, including "
                    "required imports/helpers and fn0; do not include main or tests."
                ),
            },
        ]
        token_count = count_prompt_tokens(
            tokenizer, messages, args.chat_overhead_reserve
        )
        if token_count > args.max_prompt_tokens:
            raise RuntimeError(
                f"{task_id}: prompt needs {token_count} sealed-Qwen tokens, cap is "
                f"{args.max_prompt_tokens}; refusing truncation"
            )
        prompt_counts.append(token_count)
        prompt_sha = stable_sha256(messages)
        for local_sample in range(args.samples_per_task):
            sample_index = sample_offset + local_sample
            custom_id = (
                f"r{args.round:03d}-t{position:05d}-s{sample_index:03d}"
            )
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": args.model,
                        "input": messages,
                        "max_output_tokens": args.max_output_tokens,
                        "temperature": args.temperature,
                        "store": False,
                    },
                }
            )
            index_rows.append(
                {
                    "custom_id": custom_id,
                    "round": args.round,
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "prompt_sha256": prompt_sha,
                    "prompt_tokens_estimate": token_count,
                    "serialized_text_sha256": serialized[task_id][
                        "text_sha256"
                    ],
                }
            )
    if len(requests) > MAX_BATCH_REQUESTS:
        raise RuntimeError(
            f"round has {len(requests)} requests, Batch limit is "
            f"{MAX_BATCH_REQUESTS}; split the cohort"
        )
    atomic_jsonl(input_path, requests)
    atomic_jsonl(index_path, index_rows)
    input_bytes = input_path.stat().st_size
    if input_bytes > MAX_BATCH_BYTES:
        input_path.unlink()
        index_path.unlink()
        raise RuntimeError(
            f"batch input is {input_bytes} bytes, conservative cap is "
            f"{MAX_BATCH_BYTES}; split the cohort"
        )
    manifest = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "round": args.round,
        "prepare_config_sha256": prepare_config_sha256,
        "requested_model": args.model,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "tasks": len(failed_ids),
        "requests": len(requests),
        "samples_per_task": args.samples_per_task,
        "sample_index_offset": sample_offset,
        "request_parameters": {
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "store": False,
        },
        "prompt": {
            "minimum_tokens": min(prompt_counts),
            "maximum_tokens": max(prompt_counts),
            "max_prompt_tokens": args.max_prompt_tokens,
            "chat_overhead_reserve": args.chat_overhead_reserve,
            "private_tests_exposed_to_api": False,
            "gold_source_exposed_to_api": False,
            "student_candidate_exposed_to_api": False,
            "representation": (
                "verified compressed enriched assembly plus explicit compressed CFG"
            ),
        },
        "prior_verified": prior_records,
        "inputs": {
            "serialized": {
                "path": str(serialized_path),
                "sha256": sha256_file(serialized_path),
            },
            "serialized_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
            },
            "score": {
                "path": str(score_path),
                "sha256": sha256_file(score_path),
            },
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "bytes": input_bytes,
        },
        "request_index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
        },
    }
    atomic_json(round_manifest_path, manifest)
    print(
        f"BATCH_PREPARED round={args.round} tasks={len(failed_ids)} "
        f"requests={len(requests)} bytes={input_bytes} input={input_path}",
        flush=True,
    )
    return 0


def client_from_args(args: argparse.Namespace) -> Any:
    load_env_file(Path(args.env_file).expanduser().resolve())
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing {args.api_key_env}")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("install the official OpenAI SDK") from exc
    return OpenAI(
        api_key=api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=args.api_timeout,
    )


def validate_round(round_dir: Path) -> dict[str, Any]:
    manifest = load_object(round_dir / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("batch round manifest has an unknown schema")
    input_path = round_dir / "input.jsonl"
    index_path = round_dir / "request_index.jsonl"
    if manifest.get("input", {}).get("sha256") != sha256_file(input_path):
        raise ValueError("batch input file changed after preparation")
    if manifest.get("request_index", {}).get("sha256") != sha256_file(
        index_path
    ):
        raise ValueError("batch request index changed after preparation")
    if len(load_jsonl(input_path)) != int(manifest.get("requests", -1)):
        raise ValueError("batch input row count changed")
    return manifest


def submit_command(args: argparse.Namespace) -> int:
    round_dir = Path(args.round_dir).expanduser().resolve()
    manifest = validate_round(round_dir)
    submission_path = round_dir / "submission.json"
    if submission_path.is_file():
        submission = load_object(submission_path)
        if (
            submission.get("schema") != SCHEMA
            or submission.get("input_sha256")
            != manifest["input"]["sha256"]
            or not submission.get("batch", {}).get("id")
        ):
            raise ValueError("existing batch submission is invalid")
        print(
            f"BATCH_SUBMIT_RESUME id={submission['batch']['id']} "
            f"status={submission['batch'].get('status')}",
            flush=True,
        )
        return 0
    client = client_from_args(args)
    with (round_dir / "input.jsonl").open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    uploaded_raw = object_dict(uploaded)
    file_id = str(uploaded_raw.get("id") or "")
    if not file_id:
        raise RuntimeError("OpenAI file upload returned no file id")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "purpose": "direct-compact-rs-sft",
            "round": str(manifest["round"]),
            "input_sha256": manifest["input"]["sha256"],
        },
    )
    batch_raw = object_dict(batch)
    if not str(batch_raw.get("id") or ""):
        raise RuntimeError("OpenAI batch creation returned no batch id")
    submission = {
        "schema": SCHEMA,
        "submitted_at": utc_now(),
        "host": socket.gethostname(),
        "api_base_url": args.base_url.rstrip("/"),
        "input_sha256": manifest["input"]["sha256"],
        "uploaded_file": uploaded_raw,
        "batch": batch_raw,
    }
    atomic_json(submission_path, submission)
    print(
        f"BATCH_SUBMITTED id={batch_raw['id']} status={batch_raw.get('status')} "
        f"round={manifest['round']}",
        flush=True,
    )
    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    round_dir = Path(args.round_dir).expanduser().resolve()
    validate_round(round_dir)
    submission = load_object(round_dir / "submission.json")
    batch_id = str(submission.get("batch", {}).get("id") or "")
    if not batch_id:
        raise ValueError("submission has no batch id")
    client = client_from_args(args)
    history_path = round_dir / "status_history.jsonl"
    while True:
        batch_raw = object_dict(client.batches.retrieve(batch_id))
        append_jsonl(
            history_path,
            {
                "schema": SCHEMA,
                "observed_at": utc_now(),
                "batch": batch_raw,
            },
        )
        atomic_json(
            round_dir / "latest_status.json",
            {
                "schema": SCHEMA,
                "observed_at": utc_now(),
                "batch": batch_raw,
            },
        )
        status = str(batch_raw.get("status") or "")
        print(
            f"BATCH_STATUS id={batch_id} status={status} "
            f"counts={batch_raw.get('request_counts')}",
            flush=True,
        )
        if status in TERMINAL_BATCH_STATES:
            break
        if not args.wait:
            return 3
        time.sleep(args.poll_seconds)

    for field, filename in (
        ("output_file_id", "output.jsonl"),
        ("error_file_id", "errors.jsonl"),
    ):
        file_id = str(batch_raw.get(field) or "")
        destination = round_dir / filename
        if file_id and not destination.exists():
            data = sdk_content_bytes(client.files.content(file_id))
            write_bytes_exclusive(destination, data)
    if status != "completed":
        return 2
    output_path = round_dir / "output.jsonl"
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("completed batch has no downloaded output file")
    print(
        f"BATCH_RETRIEVED id={batch_id} output_sha256={sha256_file(output_path)}",
        flush=True,
    )
    return 0


def validate_batch_response_body(body: Mapping[str, Any]) -> dict[str, Any]:
    if str(body.get("status") or "") != "completed":
        raise ValueError(f"response status is {body.get('status')!r}")
    response_id = str(body.get("id") or "")
    resolved_model = str(body.get("model") or "")
    content = response_text(body, body)
    code = extract_code(content)
    usage = body.get("usage")
    if not response_id or not resolved_model or not content or not code:
        raise ValueError("response lacks id, model, content, or code")
    if not isinstance(usage, Mapping):
        raise ValueError("response has no usage")
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    for name, value in (
        ("input", input_tokens),
        ("output", output_tokens),
        ("total", total_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"response {name} token usage is invalid")
    if total_tokens < input_tokens + output_tokens:
        raise ValueError("response token usage is internally inconsistent")
    return {
        "response_id": response_id,
        "resolved_model": resolved_model,
        "content": content,
        "code": code,
        "usage": dict(usage),
    }


def ingest_command(args: argparse.Namespace) -> int:
    round_dir = Path(args.round_dir).expanduser().resolve()
    manifest = validate_round(round_dir)
    output_path = round_dir / "output.jsonl"
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    index_rows = load_jsonl(round_dir / "request_index.jsonl")
    index: dict[str, dict[str, Any]] = {}
    for row in index_rows:
        custom_id = str(row.get("custom_id") or "")
        if not custom_id or custom_id in index:
            raise ValueError("request index has missing/duplicate custom id")
        index[custom_id] = row
    output_rows = load_jsonl(output_path)
    responses: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        custom_id = str(row.get("custom_id") or "")
        if not custom_id or custom_id in responses:
            raise ValueError("batch output has missing/duplicate custom id")
        responses[custom_id] = row
    error_rows = (
        load_jsonl(round_dir / "errors.jsonl")
        if (round_dir / "errors.jsonl").is_file()
        and (round_dir / "errors.jsonl").stat().st_size
        else []
    )
    errors: dict[str, dict[str, Any]] = {}
    for row in error_rows:
        custom_id = str(row.get("custom_id") or "")
        if custom_id:
            if custom_id in errors or custom_id in responses:
                raise ValueError("duplicate custom id across output/error files")
            errors[custom_id] = row
    unknown = (set(responses) | set(errors)) - set(index)
    if unknown:
        raise ValueError(f"batch returned unknown custom ids: {sorted(unknown)[:5]}")

    train_path = Path(manifest["inputs"]["train"]["path"])
    if sha256_file(train_path) != manifest["inputs"]["train"]["sha256"]:
        raise ValueError("private train file changed before batch ingestion")
    tests: dict[str, str] = {}
    for row in load_jsonl(train_path):
        task_id = str(row.get("task_id") or "")
        test_code = str(
            row.get("acceptance_tests")
            or row.get("tests")
            or row.get("feedback_tests")
            or ""
        )
        if not task_id or task_id in tests or not test_code:
            raise ValueError("private train tests are missing or duplicated")
        tests[task_id] = test_code

    def verify(custom_id: str) -> dict[str, Any]:
        request = index[custom_id]
        raw_output = responses[custom_id]
        response = raw_output.get("response")
        if (
            not isinstance(response, Mapping)
            or int(response.get("status_code", -1)) != 200
            or not isinstance(response.get("body"), Mapping)
            or raw_output.get("error") not in (None, {})
        ):
            return {
                "custom_id": custom_id,
                "task_id": request["task_id"],
                "sample_index": request["sample_index"],
                "api_ok": False,
                "error": "non_200_or_malformed_batch_response",
                "raw_output": raw_output,
            }
        try:
            parsed = validate_batch_response_body(response["body"])
            compiled, passed, diagnostic, _source = (
                evaluate_dart_jit_tests_detail(
                    parsed["code"],
                    tests[str(request["task_id"])],
                    f"{request['task_id']}_batch_{request['sample_index']}",
                    timeout=args.eval_timeout,
                    stability_runs=args.stability_runs,
                )
            )
            return {
                "custom_id": custom_id,
                "task_id": request["task_id"],
                "sample_index": request["sample_index"],
                "prompt_sha256": request["prompt_sha256"],
                "api_ok": True,
                **parsed,
                "compiled": bool(compiled),
                "passed": bool(passed),
                "verifier_diagnostic": str(diagnostic)[:2000],
                "raw_output": raw_output,
            }
        except Exception as exc:
            return {
                "custom_id": custom_id,
                "task_id": request["task_id"],
                "sample_index": request["sample_index"],
                "api_ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "raw_output": raw_output,
            }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        future_map = {
            pool.submit(verify, custom_id): custom_id
            for custom_id in sorted(responses)
        }
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())
    for custom_id, row in errors.items():
        request = index[custom_id]
        results.append(
            {
                "custom_id": custom_id,
                "task_id": request["task_id"],
                "sample_index": request["sample_index"],
                "api_ok": False,
                "error": "batch_error_file",
                "raw_output": row,
            }
        )
    results.sort(key=lambda row: row["custom_id"])
    atomic_jsonl(round_dir / "attempts.jsonl", results)

    verified = [
        {
            "schema": SCHEMA,
            "provider": "openai-batch",
            "requested_model": manifest["requested_model"],
            "resolved_model": row["resolved_model"],
            "response_id": row["response_id"],
            "custom_id": row["custom_id"],
            "task_id": row["task_id"],
            "sample_index": row["sample_index"],
            "prompt_sha256": row["prompt_sha256"],
            "code": row["code"],
            "code_sha256": hashlib.sha256(row["code"].encode()).hexdigest(),
            "ok": True,
            "independently_completion_attested": True,
            "stability_runs": args.stability_runs,
        }
        for row in results
        if row.get("passed") is True
    ]
    verified.sort(key=lambda row: (row["task_id"], row["sample_index"]))
    atomic_jsonl(round_dir / "verified_repairs.jsonl", verified)
    prior_rows = load_verified(
        [Path(value).expanduser().resolve() for value in args.prior_verified]
    )
    cumulative_tasks = {
        str(row["task_id"]) for row in [*prior_rows, *verified]
    }
    missing_ids = sorted(set(index) - set(responses) - set(errors))
    report = {
        "schema": SCHEMA,
        "status": "complete" if not missing_ids else "incomplete",
        "ingested_at": utc_now(),
        "round": manifest["round"],
        "requested_model": manifest["requested_model"],
        "requests": len(index),
        "responses": len(responses),
        "batch_errors": len(errors),
        "missing_custom_ids": missing_ids,
        "api_valid_responses": sum(row.get("api_ok") is True for row in results),
        "verified_candidates": len(verified),
        "verified_tasks_this_round": len(
            {str(row["task_id"]) for row in verified}
        ),
        "cumulative_verified_tasks": len(cumulative_tasks),
        "minimum_verified_tasks": args.min_verified_tasks,
        "production_coverage_met": (
            len(cumulative_tasks) >= args.min_verified_tasks
        ),
        "resolved_models": sorted(
            {
                str(row.get("resolved_model") or "")
                for row in results
                if row.get("resolved_model")
            }
        ),
        "usage": {
            key: sum(
                int(row.get("usage", {}).get(key, 0))
                for row in results
                if isinstance(row.get("usage"), Mapping)
            )
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
        "private_tests_exposed_to_api": False,
        "gold_source_exposed_to_api": False,
        "evaluator": {
            "completion_attestation": (
                "per-run-256-bit-marker-exactly-once-v1"
            ),
            "stability_runs": args.stability_runs,
        },
        "artifacts": {
            "input_sha256": manifest["input"]["sha256"],
            "output_sha256": sha256_file(output_path),
            "attempts_sha256": sha256_file(round_dir / "attempts.jsonl"),
            "verified_repairs_sha256": sha256_file(
                round_dir / "verified_repairs.jsonl"
            ),
        },
    }
    atomic_json(round_dir / "ingest_report.json", report)
    print(
        f"BATCH_INGESTED round={manifest['round']} "
        f"verified_tasks={report['verified_tasks_this_round']} "
        f"cumulative={report['cumulative_verified_tasks']} "
        f"required={args.min_verified_tasks} missing={len(missing_ids)}",
        flush=True,
    )
    return 0 if not missing_ids else 2


def finalize_command(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    round_dirs = sorted(
        path
        for path in output_root.glob("round_*")
        if (path / "ingest_report.json").is_file()
    )
    if not round_dirs:
        raise RuntimeError("no ingested batch round exists")
    rows: list[dict[str, Any]] = []
    reports = []
    for round_dir in round_dirs:
        report = load_object(round_dir / "ingest_report.json")
        if report.get("schema") != SCHEMA or report.get("status") != "complete":
            raise ValueError(f"{round_dir}: round is not completely ingested")
        artifact = round_dir / "verified_repairs.jsonl"
        if report.get("artifacts", {}).get(
            "verified_repairs_sha256"
        ) != sha256_file(artifact):
            raise ValueError(f"{round_dir}: verified repairs changed")
        if artifact.stat().st_size:
            rows.extend(load_verified([artifact]))
        reports.append(report)
    distinct: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task_id"]), str(row["code_sha256"]))
        distinct.setdefault(key, row)
    merged = sorted(
        distinct.values(),
        key=lambda row: (
            row["task_id"],
            row["sample_index"],
            row["code_sha256"],
        ),
    )
    verified_path = output_root / "verified_repairs.jsonl"
    atomic_jsonl(verified_path, merged)
    unique_tasks = len({row["task_id"] for row in merged})
    report = {
        "schema": SCHEMA,
        "finalized_at": utc_now(),
        "rounds": len(round_dirs),
        "verified_candidates": len(merged),
        "unique_verified_tasks": unique_tasks,
        "minimum_unique_verified_tasks": args.min_verified_tasks,
        "production_coverage_met": unique_tasks >= args.min_verified_tasks,
        "round_reports": [
            {
                "path": str(round_dir / "ingest_report.json"),
                "sha256": sha256_file(round_dir / "ingest_report.json"),
            }
            for round_dir in round_dirs
        ],
        "verified_repairs": {
            "path": str(verified_path),
            "sha256": sha256_file(verified_path),
        },
    }
    atomic_json(output_root / "report.json", report)
    print(
        f"BATCH_FINALIZED rounds={len(round_dirs)} "
        f"verified_tasks={unique_tasks} required={args.min_verified_tasks}",
        flush=True,
    )
    return 0 if report["production_coverage_met"] else 2


def add_client_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--round_dir", required=True)
    parser.add_argument("--env_file", default="/workspace/OpenAI.env")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--api_timeout", type=int, default=600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", allow_abbrev=False)
    common_prepare_arguments(prepare)
    prepare.set_defaults(func=prepare_command)

    submit = subparsers.add_parser("submit", allow_abbrev=False)
    add_client_arguments(submit)
    submit.set_defaults(func=submit_command)

    retrieve = subparsers.add_parser("retrieve", allow_abbrev=False)
    add_client_arguments(retrieve)
    retrieve.add_argument("--wait", action="store_true")
    retrieve.add_argument("--poll_seconds", type=int, default=60)
    retrieve.set_defaults(func=retrieve_command)

    ingest = subparsers.add_parser("ingest", allow_abbrev=False)
    ingest.add_argument("--round_dir", required=True)
    ingest.add_argument("--prior_verified", action="append", default=[])
    ingest.add_argument("--workers", type=int, default=32)
    ingest.add_argument("--eval_timeout", type=int, default=30)
    ingest.add_argument("--stability_runs", type=int, default=2)
    ingest.add_argument("--min_verified_tasks", type=int, default=400)
    ingest.set_defaults(func=ingest_command)

    finalize = subparsers.add_parser("finalize", allow_abbrev=False)
    finalize.add_argument("--output_root", required=True)
    finalize.add_argument("--min_verified_tasks", type=int, default=400)
    finalize.set_defaults(func=finalize_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(args, "api_timeout") and args.api_timeout <= 0:
        raise ValueError("api_timeout must be positive")
    if hasattr(args, "poll_seconds") and not 10 <= args.poll_seconds <= 600:
        raise ValueError("poll_seconds must lie in [10, 600]")
    if hasattr(args, "workers") and args.workers <= 0:
        raise ValueError("workers must be positive")
    if hasattr(args, "eval_timeout") and args.eval_timeout <= 0:
        raise ValueError("eval_timeout must be positive")
    if hasattr(args, "stability_runs") and args.stability_runs <= 0:
        raise ValueError("stability_runs must be positive")
    if (
        hasattr(args, "min_verified_tasks")
        and args.min_verified_tasks <= 0
    ):
        raise ValueError("min_verified_tasks must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
