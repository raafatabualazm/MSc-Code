#!/usr/bin/env python3
"""Audited, fail-closed frontier pass@k evaluation.

The default arm uses the exact sealed compact source supplied to the student,
verifies every artifact hash and round trip, expands opaque atoms to readable
normalized instructions, and preserves the compact CFG and real binary
constant prefix.  ``raw`` and ``raw_constants`` are separately labelled
controls over the same pinned held-out cohort.

Legacy environment variables remain accepted:
  PROVIDER, MODEL, K, WORKERS, LIMIT, MAXTOK, BUDGET, DEV, DSET, OUT
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from frontier_core import (
    SCHEMA_VERSION,
    CompactArtifactBundle,
    InvalidCompletion,
    JsonlJournal,
    PreflightError,
    TokenBudget,
    atomic_write_json,
    atomic_write_jsonl,
    build_messages,
    candidate_safety_reasons,
    complete_raw_disassembly,
    count_prompt_tokens,
    file_record,
    load_jsonl,
    public_dataclass,
    sha256_file,
    sha256_text,
    stable_sha256,
    utc_now,
    validate_completion,
    wilson_interval,
)

WORKSPACE = Path("/workspace")
RB = WORKSPACE / "artifacts" / "compact_fn0_rebuild"
DEFAULT_DEV = RB / "dev_fn0_real.jsonl"
DEFAULT_CONTRACT = RB / "fn0_contract.json"
DEFAULT_CONSTANTS = RB / "real_constants.jsonl"
DEFAULT_CODEBOOK = (
    WORKSPACE
    / "direct_compact_stage"
    / "scrubbed_master_v2_release"
    / "direct_compact_split_v1"
    / "compact_qwen_confirmatory_v1"
    / "codebook.json"
)
DEFAULT_CODEC = (
    WORKSPACE / "direct_compact_stage" / "scripts" / "data" / "build_compact_qwen_v1.py"
)
DEFAULT_TOKENIZER = (
    WORKSPACE
    / ".hf_home"
    / "hub"
    / "models--Qwen--Qwen3-8B"
    / "snapshots"
    / "b968826d9c46dd6066d109eabc6255188de91218"
    / "tokenizer.json"
)
DEFAULT_EVALUATOR = (
    WORKSPACE
    / "hybrid_training_patch_v2_3"
    / "scripts"
    / "evaluation"
    / "graph_compile_at_k_antigravity.py"
)
DEFAULT_DART = WORKSPACE / "dart-3.12.2" / "usr" / "bin" / "dart"
PINNED_DEV_SHA256 = "a4ed1cf185d52c3d212e2d7348fdb2a1dffd0035f4c395e2e897fd072fa70001"
PINNED_CONSTANTS_SHA256 = (
    "ec9b7086f03f1099cee31903cb4933c326df4f39160cd6820ebc47cd94860b13"
)
REQUIRED_ATTESTATION_ID = "per-run-256-bit-marker-exactly-once-v1"
MAIN_STUB = (
    "\n\nvoid main() { int frontierStub = 0; "
    "for (int i = 0; i < 3; i++) { frontierStub += i; } "
    "print(frontierStub); }\n"
)


class RunFailure(RuntimeError):
    pass


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {raw!r}") from exc


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise PreflightError(f"malformed environment line {path}:{line_number}")
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_args() -> argparse.Namespace:
    provider_default = os.environ.get("PROVIDER", "qwen").strip().lower()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--provider",
        choices=["qwen", "deepseek"],
        default=provider_default,
    )
    parser.add_argument("--model", default=os.environ.get("MODEL", ""))
    parser.add_argument(
        "--arm",
        choices=["compact", "raw", "raw_constants"],
        default=os.environ.get("ARM", "compact"),
    )
    parser.add_argument("--k", type=int, default=env_int("K", 10))
    parser.add_argument("--workers", type=int, default=env_int("WORKERS", 10))
    parser.add_argument("--limit", type=int, default=env_int("LIMIT", 0))
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=env_int("MAXTOK", 0),
        help="Completion cap; defaults to 8192 for Qwen and 12000 for DeepSeek.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=env_int("MAX_PROMPT_TOKENS", 12000),
    )
    parser.add_argument(
        "--chat-overhead-reserve",
        type=int,
        default=env_int("CHAT_OVERHEAD_RESERVE", 256),
    )
    parser.add_argument("--budget", type=int, default=env_int("BUDGET", 0))
    parser.add_argument(
        "--temperature", type=float, default=env_float("TEMPERATURE", 0.8)
    )
    parser.add_argument("--top-p", type=float, default=env_float("TOP_P", 0.95))
    parser.add_argument(
        "--timeout-seconds", type=int, default=env_int("API_TIMEOUT", 600)
    )
    parser.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=env_int("MAX_ATTEMPTS_PER_SAMPLE", 6),
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=env_float("RETRY_BASE_SECONDS", 2.0),
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=env_float("RETRY_MAX_SECONDS", 30.0),
    )
    parser.add_argument(
        "--eval-timeout-seconds", type=int, default=env_int("EVAL_TIMEOUT", 30)
    )
    parser.add_argument(
        "--eval-stability-runs", type=int, default=env_int("EVAL_STABILITY_RUNS", 2)
    )
    parser.add_argument("--dev", type=Path, default=Path(os.environ.get("DEV", DEFAULT_DEV)))
    parser.add_argument("--dataset-label", default=os.environ.get("DSET", "common175"))
    parser.add_argument(
        "--expected-dev-sha256",
        default=os.environ.get("EXPECTED_DEV_SHA256", PINNED_DEV_SHA256),
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=env_int("EXPECTED_TASK_COUNT", 175),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument(
        "--expected-constants-sha256",
        default=os.environ.get(
            "EXPECTED_CONSTANTS_SHA256", PINNED_CONSTANTS_SHA256
        ),
    )
    parser.add_argument("--evaluator-module", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument(
        "--expected-evaluator-sha256",
        default=os.environ.get("EXPECTED_EVALUATOR_SHA256", ""),
    )
    parser.add_argument("--dart", type=Path, default=DEFAULT_DART)
    parser.add_argument(
        "--expected-dart-sha256",
        default=os.environ.get("EXPECTED_DART_SHA256", ""),
    )
    parser.add_argument(
        "--raw-cache-dir", type=Path, default=RB / "frontier_raw_cache_v1"
    )
    parser.add_argument("--qwen-env-file", type=Path, default=WORKSPACE / "Qwen.env")
    parser.add_argument(
        "--deepseek-env-file", type=Path, default=WORKSPACE / "data.env"
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--extra-body-json", default=os.environ.get("EXTRA_BODY_JSON", "")
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify artifacts and write prompts without calling an API.",
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.max_output_tokens == 0:
        args.max_output_tokens = 8192 if args.provider == "qwen" else 12000
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.max_prompt_tokens <= 0:
        parser.error("--max-prompt-tokens must be positive")
    if args.chat_overhead_reserve < 0:
        parser.error("--chat-overhead-reserve cannot be negative")
    if args.budget < 0:
        parser.error("--budget cannot be negative")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be in [0,2]")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0,1]")
    if args.timeout_seconds <= 0 or args.eval_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    if args.max_attempts_per_sample <= 0 or args.eval_stability_runs <= 0:
        parser.error("attempt and stability counts must be positive")
    if args.retry_base_seconds < 0 or args.retry_max_seconds < 0:
        parser.error("retry delays cannot be negative")
    if args.retry_base_seconds > args.retry_max_seconds:
        parser.error("--retry-base-seconds cannot exceed --retry-max-seconds")
    if args.expected_task_count <= 0:
        parser.error("--expected-task-count must be positive")
    for option, value in (
        ("--expected-dev-sha256", args.expected_dev_sha256),
        ("--expected-constants-sha256", args.expected_constants_sha256),
        ("--expected-evaluator-sha256", args.expected_evaluator_sha256),
        ("--expected-dart-sha256", args.expected_dart_sha256),
    ):
        normalized = value.strip().lower()
        if normalized and not re.fullmatch(r"[0-9a-f]{64}", normalized):
            parser.error(f"{option} must be a 64-character hexadecimal SHA-256")
    if not args.model:
        args.model = (
            "qwen3.8-max-preview"
            if args.provider == "qwen"
            else "deepseek-v4-pro"
        )
    if args.extra_body_json:
        try:
            extra = json.loads(args.extra_body_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--extra-body-json is invalid: {exc}")
        if not isinstance(extra, dict):
            parser.error("--extra-body-json must decode to an object")
        args.extra_body = extra
    else:
        args.extra_body = {}
    return args


def safe_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return sanitized[:80] or "unnamed"


def choose_output_dir(args: argparse.Namespace) -> Path:
    if args.out is not None:
        return args.out.expanduser().resolve()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = "-".join(
        [
            stamp,
            safe_label(args.dataset_label),
            args.provider,
            safe_label(args.model),
            args.arm,
            uuid.uuid4().hex[:8],
        ]
    )
    return (RB / "frontier_eval" / name).resolve()


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at": utc_now(),
            },
            sort_keys=True,
        )
        try:
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as exc:
            raise RunFailure(
                f"run directory is locked: {self.path}. Remove the lock only after "
                "confirming no runner owns it."
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def import_evaluator(
    path: Path,
    expected_hash: str,
    *,
    dart_binary: Path,
    expected_dart_hash: str,
    validate_dart: bool,
) -> tuple[Any, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PreflightError(f"evaluator module does not exist: {path}")
    before = sha256_file(path)
    if expected_hash and before != expected_hash.strip().lower():
        raise PreflightError(
            f"evaluator hash mismatch: expected {expected_hash}, got {before}"
        )
    spec = importlib.util.spec_from_file_location(
        f"frontier_evaluator_{before[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot import evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    after = sha256_file(path)
    if before != after:
        raise PreflightError("evaluator module changed while it was imported")
    required_functions = (
        "evaluate_dart_jit_tests_detail",
        "prepare_dart_test_completion_attestation",
        "dart_test_completion_observed",
    )
    missing = [
        name for name in required_functions if not callable(getattr(module, name, None))
    ]
    if missing:
        raise PreflightError(
            "hardened evaluator is missing required function(s): "
            + ", ".join(missing)
        )
    attestation_id = str(getattr(module, "COMPLETION_ATTESTATION_ID", "") or "")
    if attestation_id != REQUIRED_ATTESTATION_ID:
        raise PreflightError(
            "hardened evaluator attestation identity mismatch: "
            f"expected {REQUIRED_ATTESTATION_ID!r}, got {attestation_id!r}"
        )
    dart_binary = dart_binary.expanduser().resolve()
    if not dart_binary.is_file():
        raise PreflightError(f"pinned Dart binary does not exist: {dart_binary}")
    module.DART_BIN = str(dart_binary)
    dart_record = file_record(dart_binary)
    if (
        expected_dart_hash
        and dart_record["sha256"] != expected_dart_hash.strip().lower()
    ):
        raise PreflightError(
            "Dart binary hash mismatch: expected "
            f"{expected_dart_hash}, got {dart_record['sha256']}"
        )
    if validate_dart:
        try:
            dart_version = subprocess.run(
                [str(dart_binary), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            raise PreflightError(f"pinned Dart binary is not runnable: {exc}") from exc
        if dart_version.returncode != 0:
            raise PreflightError(
                "pinned Dart binary failed --version: "
                f"{(dart_version.stderr or dart_version.stdout or '')[:500]}"
            )
        dart_record["version"] = (
            dart_version.stdout or dart_version.stderr or ""
        ).strip()
    record = file_record(path)
    record.update(
        {
            "entrypoint": "evaluate_dart_jit_tests_detail",
            "completion_attestation_id": attestation_id,
            "required_functions": list(required_functions),
            "legacy_returncode_only_evaluator_used": False,
            "dart_binary": dart_record,
        }
    )
    return module, record


def validate_dataset(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    actual_hash = sha256_file(args.dev.expanduser().resolve())
    expected_hash = args.expected_dev_sha256.strip().lower()
    if actual_hash != expected_hash:
        raise PreflightError(
            f"dataset hash mismatch: expected {expected_hash}, got {actual_hash}. "
            "An alternate cohort requires an explicit reviewed "
            "--expected-dev-sha256."
        )
    if len(rows) != args.expected_task_count:
        raise PreflightError(
            f"dataset has {len(rows)} rows, expected {args.expected_task_count}"
        )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise PreflightError(f"dataset row {index} has no task_id")
        if task_id in seen:
            raise PreflightError(f"duplicate dataset task_id: {task_id}")
        seen.add(task_id)
        if str(row.get("function") or "") != "fn0":
            raise PreflightError(f"task {task_id} target function is not fn0")
        if str(row.get("lang") or "").lower() != "dart":
            raise PreflightError(f"task {task_id} target language is not Dart")
        tests = row.get("tests")
        acceptance = row.get("acceptance_tests")
        if not isinstance(tests, str) or not tests.strip():
            raise PreflightError(f"task {task_id} has no tests")
        if not isinstance(acceptance, str) or not acceptance.strip():
            raise PreflightError(f"task {task_id} has no acceptance_tests")
        if tests != acceptance:
            raise PreflightError(
                f"task {task_id} tests and acceptance_tests differ; this runner "
                "requires the pinned common evaluator suite"
            )
        if not isinstance(row.get("dart_source"), str) or not row["dart_source"].strip():
            raise PreflightError(f"task {task_id} has no dart_source provenance")
    if args.limit:
        return rows[: args.limit]
    return rows


def config_for_hash(args: argparse.Namespace) -> dict[str, Any]:
    _api_key, base_url = resolve_api_configuration(args)
    return {
        "schema": SCHEMA_VERSION,
        "provider": args.provider,
        "model_requested": args.model,
        "arm": args.arm,
        "k": args.k,
        "workers": args.workers,
        "limit": args.limit,
        "max_output_tokens": args.max_output_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "chat_overhead_reserve": args.chat_overhead_reserve,
        "budget": args.budget,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout_seconds": args.timeout_seconds,
        "max_attempts_per_sample": args.max_attempts_per_sample,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds,
        "eval_timeout_seconds": args.eval_timeout_seconds,
        "eval_stability_runs": args.eval_stability_runs,
        "dataset_label": args.dataset_label,
        "expected_dev_sha256": args.expected_dev_sha256,
        "expected_task_count": args.expected_task_count,
        "expected_constants_sha256": args.expected_constants_sha256,
        "extra_body": args.extra_body,
        "evaluator_module": str(args.evaluator_module.expanduser().resolve()),
        "expected_evaluator_sha256": args.expected_evaluator_sha256.strip().lower(),
        "dart_binary": str(args.dart.expanduser().resolve()),
        "expected_dart_sha256": args.expected_dart_sha256.strip().lower(),
        "api_base_url_sha256": sha256_text(base_url.rstrip("/")),
        "api_base_url_redacted": redact_api_endpoint(base_url),
    }


def prepare_run(
    args: argparse.Namespace,
    out: Path,
) -> tuple[
    CompactArtifactBundle,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
    dict[str, Any],
]:
    dev_path = args.dev.expanduser().resolve()
    if not dev_path.is_file():
        raise PreflightError(f"dataset does not exist: {dev_path}")
    full_rows = load_jsonl(dev_path, "frontier dataset")
    rows = validate_dataset(args, full_rows)
    bundle = CompactArtifactBundle(
        contract_path=args.contract,
        codebook_path=args.codebook,
        tokenizer_path=args.tokenizer_json,
        codec_path=args.codec,
        constants_path=args.constants,
        expected_constants_sha256=args.expected_constants_sha256,
    )
    config = config_for_hash(args)
    config_sha = stable_sha256(config)

    plans: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    prompt_map: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        task_id = str(row["task_id"])
        prepared = bundle.prepare(row)
        raw_disassembly = None
        raw_provenance = None
        if args.arm in {"raw", "raw_constants"}:
            raw_disassembly, raw_provenance = complete_raw_disassembly(
                task_id=task_id,
                dart_source=row["dart_source"],
                dart_binary=args.dart,
                cache_dir=args.raw_cache_dir,
                main_stub=MAIN_STUB,
            )
        messages = build_messages(
            arm=args.arm,
            prepared=prepared,
            raw_disassembly=raw_disassembly,
        )
        prompt_tokens = count_prompt_tokens(
            messages,
            bundle.tokenizer,
            chat_overhead_reserve=args.chat_overhead_reserve,
        )
        if prompt_tokens["estimated_prompt_tokens"] > args.max_prompt_tokens:
            raise PreflightError(
                f"task {task_id} prompt is "
                f"{prompt_tokens['estimated_prompt_tokens']} sealed-Qwen tokens "
                f"including reserve, cap is {args.max_prompt_tokens}; refusing to "
                "truncate"
            )
        prompt_sha = stable_sha256(messages)
        source_hash = sha256_text(row["dart_source"])
        tests_hash = sha256_text(row["tests"])
        task_record = {
            "schema": SCHEMA_VERSION,
            "task_index": index,
            "task_id": task_id,
            "function": "fn0",
            "language": "Dart",
            "dart_source_sha256": source_hash,
            "tests_sha256": tests_hash,
            "acceptance_tests_sha256": sha256_text(row["acceptance_tests"]),
            "tests_equal_acceptance_tests": True,
            "compact_ids_sha256": prepared.compact_ids_sha256,
            "compact_text_sha256": prepared.compact_text_sha256,
            "canonical_sha256": prepared.canonical_sha256,
            "constants_record_sha256": prepared.constants_record_sha256,
            "constants_extraction_error": prepared.constants_extraction_error,
            "constant_prefix_tokens": len(prepared.constant_prefix_ids),
            "graph_tokens": len(prepared.graph_ids),
            "raw_control": (
                {
                    key: value
                    for key, value in raw_provenance.items()
                    if key not in {"cache_hit", "cache_path"}
                }
                if raw_provenance is not None
                else None
            ),
        }
        prompt_record = {
            "schema": SCHEMA_VERSION,
            "task_id": task_id,
            "arm": args.arm,
            "prompt_sha256": prompt_sha,
            "messages": messages,
            "token_count": prompt_tokens,
            "tokenizer_sha256": bundle.tokenizer_sha256,
            "never_truncated": True,
            "tests_exposed": False,
        }
        plans.append(
            {
                "task_id": task_id,
                "row": row,
                "messages": messages,
                "prompt_sha256": prompt_sha,
                "estimated_prompt_tokens": prompt_tokens[
                    "estimated_prompt_tokens"
                ],
            }
        )
        task_records.append(task_record)
        prompt_records.append(prompt_record)
        prompt_map[task_id] = prompt_record

    task_set_sha = stable_sha256([plan["task_id"] for plan in plans])
    constant_error_tasks = [
        record["task_id"]
        for record in task_records
        if record["constants_extraction_error"] is not None
    ]
    dataset_record = file_record(dev_path)
    evaluator_module, evaluator_record = import_evaluator(
        args.evaluator_module,
        args.expected_evaluator_sha256,
        dart_binary=args.dart,
        expected_dart_hash=args.expected_dart_sha256,
        validate_dart=False,
    )
    for plan in plans:
        task_id = plan["task_id"]
        ok, diagnostic, _instrumented_source, marker = (
            evaluator_module.prepare_dart_test_completion_attestation(
                plan["row"]["acceptance_tests"]
            )
        )
        if not ok or not marker:
            raise PreflightError(
                f"task {task_id} acceptance-test harness cannot be completion "
                f"attested: {diagnostic or 'no marker generated'}"
            )
    del evaluator_module
    provenance = {
        "schema": SCHEMA_VERSION,
        "status": "preflight_complete",
        "created_at": utc_now(),
        "run_id": out.name,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "config": config,
        "config_sha256": config_sha,
        "task_set_sha256": task_set_sha,
        "tasks_selected": len(plans),
        "binary_constant_extraction_errors": {
            "count": len(constant_error_tasks),
            "task_ids": constant_error_tasks,
            "interpretation": (
                "The exact student prefix is still reproduced; an extraction "
                "error means that task received no additional successfully "
                "recovered constants."
            ),
        },
        "dataset_rows_before_limit": len(full_rows),
        "dataset": dataset_record,
        "artifacts": bundle.artifact_records(),
        "evaluator": evaluator_record,
        "runner": file_record(Path(__file__)),
        "core": file_record(Path(__file__).with_name("frontier_core.py")),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "preflight_invariants": {
            "dataset_sha256_pinned": True,
            "expected_dataset_rows_verified": True,
            "unique_task_ids": True,
            "tests_equal_acceptance_tests": True,
            "completion_attested_evaluator_contract_verified": True,
            "evaluator_sha256_pinned": bool(
                args.expected_evaluator_sha256.strip()
            ),
            "dart_binary_sha256_pinned": bool(args.expected_dart_sha256.strip()),
            "acceptance_test_main_attestable_for_every_task": True,
            "legacy_returncode_only_evaluator_used": False,
            "compact_artifact_hashes_verified": True,
            "per_row_compact_hashes_verified": True,
            "codec_roundtrip_verified_for_every_task": True,
            "student_constant_prefix_reconstructed_for_every_task": True,
            "binary_constant_extraction_succeeded_for_every_task": not constant_error_tasks,
            "opaque_custom_ids_not_sent_to_api": True,
            "prompt_token_cap_checked_for_every_task": True,
            "prompts_never_truncated": True,
            "tests_not_exposed_to_teacher": True,
            "raw_arm_is_control_only": args.arm != "compact",
        },
    }
    existing_provenance = out / "provenance.json"
    if existing_provenance.is_file():
        prior = json.loads(existing_provenance.read_text(encoding="utf-8"))
        if not args.resume:
            raise RunFailure(f"output already exists and --no-resume was requested: {out}")
        if prior.get("config_sha256") != config_sha:
            raise RunFailure(
                "resume config does not match existing provenance: "
                f"{prior.get('config_sha256')} != {config_sha}"
            )
        if prior.get("task_set_sha256") != task_set_sha:
            raise RunFailure("resume task set does not match existing provenance")
        if (prior.get("dataset") or {}).get("sha256") != dataset_record["sha256"]:
            raise RunFailure("resume dataset hash does not match existing provenance")
        provenance["created_at"] = prior.get("created_at", provenance["created_at"])
        provenance["resumed_at"] = utc_now()
    atomic_write_json(out / "provenance.json", provenance)
    atomic_write_jsonl(out / "tasks.jsonl", task_records)
    atomic_write_jsonl(out / "prompts.jsonl", prompt_records)
    return bundle, plans, prompt_map, config_sha, provenance


def redact_api_endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        return "(unset)"
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return "(unparseable)"
    hostname = parsed.hostname or ""
    if not hostname:
        return "(unparseable)"
    netloc = hostname
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", "")
    )


def resolve_api_configuration(args: argparse.Namespace) -> tuple[str, str]:
    if args.provider == "qwen":
        loaded = read_env_file(args.qwen_env_file)
        key = (
            args.api_key
            or os.environ.get("QWEN_API_KEY", "")
            or loaded.get("API_KEY", "")
        )
        base = (
            args.base_url
            or os.environ.get("QWEN_BASE_URL", "")
            or loaded.get("DASHSCOPE_ENDPOINT", "")
        )
    else:
        loaded = read_env_file(args.deepseek_env_file)
        key = (
            args.api_key
            or os.environ.get("DEEPSEEK_API_KEY", "")
            or loaded.get("DEEPSEEK_API_KEY", "")
        )
        base = (
            args.base_url
            or os.environ.get("DEEPSEEK_BASE_URL", "")
            or loaded.get("DEEPSEEK_BASE_URL", "")
            or "https://api.deepseek.com"
        )
    return key, base.rstrip("/")


def api_credentials(args: argparse.Namespace) -> tuple[str, str]:
    key, base = resolve_api_configuration(args)
    if not key:
        raise PreflightError(f"no API key configured for provider {args.provider}")
    if not base:
        raise PreflightError(f"no API base URL configured for provider {args.provider}")
    return key, base


def response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(response, "dict"):
        dumped = response.dict()
        if isinstance(dumped, dict):
            return dumped
    return {"unserializable_response_type": type(response).__name__}


def usage_total(raw_response: Mapping[str, Any], fallback: int) -> int:
    usage = raw_response.get("usage")
    if not isinstance(usage, Mapping):
        return fallback
    value = usage.get("total_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    total = value
    if total < 0:
        return fallback
    return total


def load_resume_attempts(
    path: Path,
    *,
    config_sha: str,
    prompt_map: Mapping[str, Mapping[str, Any]],
    budget: TokenBudget,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], int]]:
    valid: dict[tuple[str, int], dict[str, Any]] = {}
    next_attempt: dict[tuple[str, int], int] = {}
    if not path.is_file():
        return valid, next_attempt
    for row in load_jsonl(path, "attempt journal"):
        if row.get("config_sha256") != config_sha:
            raise RunFailure("attempt journal contains a foreign config fingerprint")
        task_id = str(row.get("task_id") or "")
        sample_index = int(row.get("sample_index", -1))
        attempt_index = int(row.get("attempt_index", -1))
        key = (task_id, sample_index)
        if task_id not in prompt_map or sample_index < 0 or attempt_index < 0:
            raise RunFailure("attempt journal contains an invalid task/sample index")
        if row.get("prompt_sha256") != prompt_map[task_id].get("prompt_sha256"):
            raise RunFailure("attempt journal prompt fingerprint mismatch")
        next_attempt[key] = max(next_attempt.get(key, 0), attempt_index + 1)
        budget_charge = row.get("budget_charge_tokens")
        if isinstance(budget_charge, bool) or not isinstance(budget_charge, int):
            usage = row.get("usage")
            total = usage.get("total_tokens") if isinstance(usage, Mapping) else None
            budget_charge = total if isinstance(total, int) and total > 0 else 0
        if budget_charge > 0:
            if not budget.reserve(budget_charge):
                raise RunFailure("resumed attempts already exceed token budget")
            budget.settle(budget_charge, budget_charge)
        if row.get("valid"):
            code = str(row.get("code") or "")
            if sha256_text(code) != row.get("code_sha256"):
                raise RunFailure("attempt journal candidate hash mismatch")
            if candidate_safety_reasons(code):
                raise RunFailure("attempt journal contains an unsafe valid candidate")
            if key in valid and valid[key].get("code_sha256") != row.get(
                "code_sha256"
            ):
                raise RunFailure("multiple different valid candidates occupy one sample")
            valid[key] = row
    return valid, next_attempt


def load_resume_outcomes(
    path: Path,
    *,
    config_sha: str,
    evaluator_sha256: str,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    existing: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not path.is_file():
        return existing
    for row in load_jsonl(path, "outcome journal"):
        if row.get("config_sha256") != config_sha:
            raise RunFailure("outcome journal contains a foreign config fingerprint")
        task_id = str(row.get("task_id") or "")
        sample_index = int(row.get("sample_index", -1))
        attempt_id = str(row.get("attempt_id") or "")
        code_sha = str(row.get("code_sha256") or "")
        if not task_id or sample_index < 0 or not attempt_id or not code_sha:
            raise RunFailure("outcome journal contains an invalid identity")
        key = (task_id, sample_index, attempt_id)
        if key in existing:
            if stable_sha256(existing[key]) != stable_sha256(row):
                raise RunFailure("outcome journal contains conflicting duplicates")
            continue
        runs = row.get("stability_runs")
        if not isinstance(runs, list) or not runs:
            raise RunFailure("outcome journal has no stability-run evidence")
        if row.get("evaluator_sha256") != evaluator_sha256:
            raise RunFailure("outcome journal evaluator fingerprint mismatch")
        if row.get("completion_attestation_id") != REQUIRED_ATTESTATION_ID:
            raise RunFailure("outcome journal attestation identity mismatch")
        if row.get("completion_attestation_enforced") is not True:
            raise RunFailure("outcome journal lacks completion-attestation enforcement")
        for run in runs:
            if not isinstance(run, Mapping):
                raise RunFailure("outcome journal has an invalid stability run")
            if run.get("completion_attestation_id") != REQUIRED_ATTESTATION_ID:
                raise RunFailure("stability-run attestation identity mismatch")
            if run.get("completion_attestation_required") is not True:
                raise RunFailure("stability run did not require completion attestation")
            if bool(run.get("completion_attestation_satisfied")) != bool(
                run.get("passed")
            ):
                raise RunFailure(
                    "stability-run pass disagrees with completion attestation"
                )
        all_compiled = all(bool(run.get("compiled")) for run in runs)
        all_passed = all(bool(run.get("passed")) for run in runs)
        if bool(row.get("compiled")) != all_compiled:
            raise RunFailure("outcome compile result disagrees with stability runs")
        if bool(row.get("passed")) != all_passed:
            raise RunFailure("outcome pass result disagrees with stability runs")
        if bool(row.get("completion_attestation_satisfied_all_runs")) != all_passed:
            raise RunFailure("outcome attestation result disagrees with stability runs")
        existing[key] = row
    return existing


def retry_delay(args: argparse.Namespace, attempt_index: int) -> float:
    base = min(
        args.retry_max_seconds,
        args.retry_base_seconds * (2 ** min(attempt_index, 8)),
    )
    return min(args.retry_max_seconds, base * random.uniform(0.8, 1.2))


def make_request(client: Any, args: argparse.Namespace, messages: list[dict[str, str]]) -> Any:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.timeout_seconds,
    }
    if args.extra_body:
        request["extra_body"] = args.extra_body
    return client.chat.completions.create(**request)


def evaluate_candidate_stably(
    evaluator: Any,
    *,
    code: str,
    tests: str,
    task_id: str,
    sample_index: int,
    stability_runs: int,
    timeout: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for stability_index in range(stability_runs):
        evaluation_id = (
            f"{task_id}_frontier_s{sample_index}_r{stability_index}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        try:
            compiled, passed, diagnostic, evaluated_source = evaluator(
                code,
                tests,
                evaluation_id,
                timeout=timeout,
                stability_runs=1,
            )
            compiled = bool(compiled)
            passed = bool(passed)
            diagnostic = str(diagnostic or "")
            if passed and not compiled:
                raise RuntimeError("hardened evaluator returned passed without compiled")
            if diagnostic == "dart_not_found":
                raise RunFailure(
                    "pinned Dart binary disappeared or became unavailable during "
                    f"evaluation of {evaluation_id}"
                )
            runs.append(
                {
                    "stability_index": stability_index,
                    "evaluation_id": evaluation_id,
                    "compiled": compiled,
                    "passed": passed,
                    "diagnostic": diagnostic,
                    "evaluated_source_sha256": sha256_text(
                        str(evaluated_source or "")
                    ),
                    "completion_attestation_id": REQUIRED_ATTESTATION_ID,
                    "completion_attestation_required": True,
                    "completion_attestation_satisfied": passed,
                }
            )
        except RunFailure:
            raise
        except Exception as exc:
            raise RunFailure(
                "completion-attested evaluator raised an internal exception for "
                f"{evaluation_id}: {type(exc).__name__}: {exc}"
            ) from exc
    return {
        "compiled": all(run["compiled"] for run in runs),
        "passed": all(run["passed"] for run in runs),
        "completion_attestation_id": REQUIRED_ATTESTATION_ID,
        "completion_attestation_enforced": True,
        "completion_attestation_satisfied_all_runs": all(
            run["completion_attestation_satisfied"] for run in runs
        ),
        "stability_runs": runs,
    }


def run_api_and_evaluation(
    args: argparse.Namespace,
    *,
    out: Path,
    plans: list[dict[str, Any]],
    prompt_map: dict[str, dict[str, Any]],
    config_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise PreflightError("the openai Python package is required") from exc
    key, base_url = api_credentials(args)
    client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
    if not args.expected_evaluator_sha256.strip():
        raise PreflightError(
            "paid evaluation requires --expected-evaluator-sha256 to pin the "
            "completion-attested harness"
        )
    if not args.expected_dart_sha256.strip():
        raise PreflightError(
            "paid evaluation requires --expected-dart-sha256 to pin the Dart "
            "runtime/compiler"
        )
    evaluator_module, evaluator_record = import_evaluator(
        args.evaluator_module,
        args.expected_evaluator_sha256,
        dart_binary=args.dart,
        expected_dart_hash=args.expected_dart_sha256,
        validate_dart=True,
    )
    evaluator = evaluator_module.evaluate_dart_jit_tests_detail
    provenance = dict(provenance)
    if evaluator_record["sha256"] != provenance["evaluator"]["sha256"]:
        raise PreflightError("evaluator changed after prompt preflight")
    provenance["evaluator"] = evaluator_record
    provenance["api"] = {
        "provider": args.provider,
        "base_url_redacted": redact_api_endpoint(base_url),
        "base_url_sha256": sha256_text(base_url),
        "requested_model": args.model,
        "openai_package_version": importlib.metadata.version("openai"),
        "credentials_persisted": False,
    }
    provenance["status"] = "running"
    provenance["started_at"] = utc_now()
    atomic_write_json(out / "provenance.json", provenance)

    budget = TokenBudget(args.budget)
    attempts_path = out / "attempts.jsonl"
    valid_resume, next_attempt = load_resume_attempts(
        attempts_path,
        config_sha=config_sha,
        prompt_map=prompt_map,
        budget=budget,
    )
    attempts = JsonlJournal(attempts_path)
    outcomes_path = out / "outcomes.jsonl"
    resumed_outcomes = load_resume_outcomes(
        outcomes_path,
        config_sha=config_sha,
        evaluator_sha256=evaluator_record["sha256"],
    )
    valid_attempt_keys = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in valid_resume.items()
    }
    orphan_outcomes = sorted(set(resumed_outcomes) - valid_attempt_keys)
    if orphan_outcomes:
        raise RunFailure(
            f"outcome journal has {len(orphan_outcomes)} orphan record(s); "
            f"first={orphan_outcomes[0]}"
        )
    outcomes = JsonlJournal(outcomes_path)
    stop = threading.Event()
    worst_case_reservation = args.max_prompt_tokens + args.max_output_tokens

    def run_task(plan: dict[str, Any]) -> dict[str, Any]:
        task_id = plan["task_id"]
        valid_candidates: list[dict[str, Any]] = []
        for sample_index in range(args.k):
            key_tuple = (task_id, sample_index)
            resumed = valid_resume.get(key_tuple)
            if resumed is not None:
                valid_candidates.append(
                    {
                        "sample_index": sample_index,
                        "code": resumed["code"],
                        "code_sha256": resumed["code_sha256"],
                        "attempt_id": resumed["attempt_id"],
                        "resumed": True,
                    }
                )
                continue
            first_attempt = next_attempt.get(key_tuple, 0)
            accepted: dict[str, Any] | None = None
            for attempt_index in range(
                first_attempt, first_attempt + args.max_attempts_per_sample
            ):
                if stop.is_set():
                    raise RunFailure(f"task {task_id} stopped after another fatal error")
                attempt_id = (
                    f"{safe_label(task_id)}.s{sample_index}.a{attempt_index}."
                    f"{uuid.uuid4().hex[:10]}"
                )
                base_record: dict[str, Any] = {
                    "schema": SCHEMA_VERSION,
                    "record_type": "api_attempt",
                    "attempt_id": attempt_id,
                    "config_sha256": config_sha,
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "attempt_index": attempt_index,
                    "prompt_sha256": plan["prompt_sha256"],
                    "requested_model": args.model,
                    "provider": args.provider,
                    "started_at": utc_now(),
                }
                if not budget.reserve(worst_case_reservation):
                    record = dict(base_record)
                    record.update(
                        {
                            "finished_at": utc_now(),
                            "valid": False,
                            "invalid_reason": "token_budget_reservation_failed",
                            "budget_charge_tokens": 0,
                            "usage": None,
                            "response": None,
                        }
                    )
                    attempts.append(record)
                    raise RunFailure(
                        f"token budget cannot reserve another request for task {task_id}"
                    )
                response: Any = None
                reservation_open = True
                try:
                    response = make_request(client, args, plan["messages"])
                    raw_response = response_to_dict(response)
                    settled = usage_total(raw_response, worst_case_reservation)
                    budget.settle(worst_case_reservation, settled)
                    reservation_open = False
                    if settled > worst_case_reservation:
                        record = dict(base_record)
                        record.update(
                            {
                                "finished_at": utc_now(),
                                "valid": False,
                                "invalid_reason": (
                                    "provider_usage_exceeded_reserved_worst_case:"
                                    f"{settled}>{worst_case_reservation}"
                                ),
                                "budget_charge_tokens": settled,
                                "usage": raw_response.get("usage"),
                                "response": raw_response,
                            }
                        )
                        attempts.append(record)
                        raise RunFailure(
                            "provider token usage exceeded the requested prompt plus "
                            "completion caps; aborting to preserve the budget contract"
                        )
                    try:
                        completion = validate_completion(
                            response,
                            max_prompt_tokens=args.max_prompt_tokens,
                            max_output_tokens=args.max_output_tokens,
                        )
                    except InvalidCompletion as exc:
                        usage = raw_response.get("usage")
                        record = dict(base_record)
                        record.update(
                            {
                                "finished_at": utc_now(),
                                "valid": False,
                                "invalid_reason": str(exc),
                                "budget_charge_tokens": settled,
                                "usage": usage if isinstance(usage, Mapping) else None,
                                "response": raw_response,
                            }
                        )
                        attempts.append(record)
                    else:
                        record = dict(base_record)
                        record.update(
                            {
                                "finished_at": utc_now(),
                                "valid": True,
                                "invalid_reason": None,
                                "response_id": completion.response_id,
                                "resolved_model": completion.response_model,
                                "response_created": completion.response_created,
                                "finish_reason": completion.finish_reason,
                                "budget_charge_tokens": settled,
                                "usage": completion.usage,
                                "content": completion.content,
                                "reasoning_content": completion.reasoning_content,
                                "code": completion.code,
                                "code_sha256": completion.code_sha256,
                                "response": completion.raw_response,
                            }
                        )
                        attempts.append(record)
                        accepted = record
                        break
                except RunFailure:
                    if reservation_open:
                        # A request may have reached the provider even when the
                        # client raised. Charge the full reservation to keep a
                        # configured budget a true upper bound.
                        budget.settle(
                            worst_case_reservation, worst_case_reservation
                        )
                    raise
                except Exception as exc:
                    if reservation_open:
                        # Unknown API failures have unknown billing. Conservatively
                        # consume the worst-case reservation rather than silently
                        # undercounting or permitting a budget overshoot.
                        budget.settle(
                            worst_case_reservation, worst_case_reservation
                        )
                        reservation_open = False
                    record = dict(base_record)
                    record.update(
                        {
                            "finished_at": utc_now(),
                            "valid": False,
                            "invalid_reason": (
                                f"api_exception:{type(exc).__name__}:{str(exc)[:1000]}"
                            ),
                            "budget_charge_tokens": worst_case_reservation,
                            "usage": None,
                            "response": None,
                        }
                    )
                    attempts.append(record)
                if accepted is None and attempt_index + 1 < (
                    first_attempt + args.max_attempts_per_sample
                ):
                    delay = retry_delay(args, attempt_index)
                    if stop.wait(delay):
                        raise RunFailure(
                            f"task {task_id} stopped during retry backoff"
                        )
            if accepted is None:
                raise RunFailure(
                    f"task {task_id} sample {sample_index} did not yield a valid "
                    f"completion in {args.max_attempts_per_sample} attempts"
                )
            valid_candidates.append(
                {
                    "sample_index": sample_index,
                    "code": accepted["code"],
                    "code_sha256": accepted["code_sha256"],
                    "attempt_id": accepted["attempt_id"],
                    "resumed": False,
                }
            )

        if len(valid_candidates) != args.k:
            raise RunFailure(
                f"task {task_id} has {len(valid_candidates)} valid candidates, "
                f"expected {args.k}"
            )
        candidate_outcomes: list[dict[str, Any]] = []
        for candidate in valid_candidates:
            outcome_key = (
                task_id,
                candidate["sample_index"],
                candidate["attempt_id"],
            )
            resumed_outcome = resumed_outcomes.get(outcome_key)
            if resumed_outcome is not None:
                if resumed_outcome.get("code_sha256") != candidate["code_sha256"]:
                    raise RunFailure(
                        f"resumed outcome code hash mismatch for {outcome_key}"
                    )
                runs = resumed_outcome.get("stability_runs") or []
                if len(runs) != args.eval_stability_runs:
                    raise RunFailure(
                        f"resumed outcome stability count mismatch for {outcome_key}"
                    )
                candidate_outcomes.append(resumed_outcome)
                continue
            evaluation = evaluate_candidate_stably(
                evaluator,
                code=candidate["code"],
                tests=plan["row"]["acceptance_tests"],
                task_id=task_id,
                sample_index=candidate["sample_index"],
                stability_runs=args.eval_stability_runs,
                timeout=args.eval_timeout_seconds,
            )
            outcome = {
                "schema": SCHEMA_VERSION,
                "record_type": "candidate_outcome",
                "config_sha256": config_sha,
                "task_id": task_id,
                "sample_index": candidate["sample_index"],
                "attempt_id": candidate["attempt_id"],
                "code_sha256": candidate["code_sha256"],
                "evaluator_sha256": evaluator_record["sha256"],
                "evaluator_entrypoint": evaluator_record["entrypoint"],
                "completion_attestation_id": evaluation[
                    "completion_attestation_id"
                ],
                "completion_attestation_enforced": evaluation[
                    "completion_attestation_enforced"
                ],
                "completion_attestation_satisfied_all_runs": evaluation[
                    "completion_attestation_satisfied_all_runs"
                ],
                "compiled": evaluation["compiled"],
                "passed": evaluation["passed"],
                "stability_runs": evaluation["stability_runs"],
                "evaluated_at": utc_now(),
            }
            outcomes.append(outcome)
            candidate_outcomes.append(outcome)
        return {
            "task_id": task_id,
            "valid_completions": len(valid_candidates),
            "compiled": any(value["compiled"] for value in candidate_outcomes),
            "passed": any(value["passed"] for value in candidate_outcomes),
            "candidate_outcomes": candidate_outcomes,
        }

    task_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(run_task, plan): plan for plan in plans}
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_map), 1
        ):
            plan = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                stop.set()
                failures.append(
                    {
                        "task_id": plan["task_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                task_results.append(result)
            if completed % 10 == 0 or completed == len(plans):
                print(
                    f"  {completed}/{len(plans)} complete; "
                    f"valid_tasks={len(task_results)} failures={len(failures)} "
                    f"tokens={budget.snapshot()['spent']}",
                    flush=True,
                )
    if failures:
        raise RunFailure(
            f"{len(failures)} task(s) failed; first failure: {failures[0]}"
        )
    if len(task_results) != len(plans):
        raise RunFailure(
            f"only {len(task_results)}/{len(plans)} tasks completed"
        )
    if any(result["valid_completions"] != args.k for result in task_results):
        raise RunFailure("one or more tasks did not receive exactly K valid completions")

    task_order = {plan["task_id"]: index for index, plan in enumerate(plans)}
    task_results.sort(key=lambda result: task_order[result["task_id"]])
    passed = sum(result["passed"] for result in task_results)
    compiled = sum(result["compiled"] for result in task_results)
    resolved_models: set[str] = set()
    response_ids: set[str] = set()
    attempt_rows = load_jsonl(attempts_path, "completed attempt journal")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    recorded_budget_charge = 0
    valid_count = 0
    invalid_count = 0
    invalid_reasons: dict[str, int] = {}
    for row in attempt_rows:
        charge = row.get("budget_charge_tokens")
        if isinstance(charge, bool) or not isinstance(charge, int) or charge < 0:
            raise RunFailure("attempt journal has an invalid budget charge")
        recorded_budget_charge += charge
        if row.get("valid"):
            valid_count += 1
            resolved_models.add(str(row.get("resolved_model") or ""))
            response_id = str(row.get("response_id") or "")
            if not response_id:
                raise RunFailure("one or more valid attempts lacks a response id")
            if response_id in response_ids:
                raise RunFailure(f"duplicate valid response id: {response_id}")
            response_ids.add(response_id)
        else:
            invalid_count += 1
            reason = str(row.get("invalid_reason") or "unknown")
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        row_usage = row.get("usage")
        if isinstance(row_usage, Mapping):
            for key_name in usage:
                value = row_usage.get(key_name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key_name] += int(value)
    if valid_count != len(plans) * args.k:
        raise RunFailure(
            f"attempt journal has {valid_count} valid completions, expected "
            f"{len(plans) * args.k}"
        )
    if "" in resolved_models:
        raise RunFailure("one or more valid attempts lacks a resolved model identity")
    if len(resolved_models) != 1:
        raise RunFailure(
            f"valid completions resolved to multiple model identities: "
            f"{sorted(resolved_models)}"
        )
    if recorded_budget_charge != budget.snapshot()["spent"]:
        raise RunFailure(
            "attempt-journal budget charges disagree with the in-memory ledger: "
            f"{recorded_budget_charge} != {budget.snapshot()['spent']}"
        )
    summary = {
        "schema": SCHEMA_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "run_id": out.name,
        "dataset_label": args.dataset_label,
        "dataset_sha256": args.expected_dev_sha256,
        "task_set_sha256": provenance["task_set_sha256"],
        "arm": args.arm,
        "arm_interpretation": (
            "primary exact decoded student representation"
            if args.arm == "compact"
            else "raw-disassembly control; not a compression-only comparison"
        ),
        "provider": args.provider,
        "requested_model": args.model,
        "resolved_models": sorted(resolved_models),
        "k": args.k,
        "tasks": len(task_results),
        "valid_completions": valid_count,
        "invalid_attempts": invalid_count,
        "invalid_attempt_reasons": invalid_reasons,
        "pass_at_k": {
            "successes": passed,
            "total": len(task_results),
            "rate": passed / len(task_results),
            "wilson_95": wilson_interval(passed, len(task_results)),
        },
        "compile_at_k": {
            "successes": compiled,
            "total": len(task_results),
            "rate": compiled / len(task_results),
            "wilson_95": wilson_interval(compiled, len(task_results)),
        },
        "usage": usage,
        "budget": budget.snapshot(),
        "recorded_budget_charge_tokens": recorded_budget_charge,
        "evaluator": evaluator_record,
        "completion_attestation_id": REQUIRED_ATTESTATION_ID,
        "completion_attestation_enforced_for_every_candidate": True,
        "all_tasks_have_exactly_k_valid_completions": True,
        "early_stopping_used": False,
        "prompt_truncation_used": False,
        "task_results": task_results,
        "artifacts": {
            "tasks": file_record(out / "tasks.jsonl"),
            "prompts": file_record(out / "prompts.jsonl"),
            "attempts": file_record(out / "attempts.jsonl"),
            "outcomes": file_record(out / "outcomes.jsonl"),
        },
    }
    atomic_write_json(out / "summary.json", summary)
    provenance["status"] = "complete"
    provenance["completed_at"] = summary["completed_at"]
    provenance["summary_sha256"] = sha256_file(out / "summary.json")
    atomic_write_json(out / "provenance.json", provenance)
    atomic_write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA_VERSION,
            "created_at": utc_now(),
            "files": {
                name: file_record(out / name)
                for name in (
                    "provenance.json",
                    "tasks.jsonl",
                    "prompts.jsonl",
                    "attempts.jsonl",
                    "outcomes.jsonl",
                    "summary.json",
                )
            },
        },
    )
    return summary


def main() -> int:
    args = parse_args()
    out = choose_output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out / ".run.lock"):
        try:
            bundle, plans, prompt_map, config_sha, provenance = prepare_run(args, out)
            del bundle
            max_estimate = max(
                int(prompt_map[plan["task_id"]]["token_count"]["estimated_prompt_tokens"])
                for plan in plans
            )
            print(
                f"PREFLIGHT_OK arm={args.arm} dataset={args.dataset_label} "
                f"tasks={len(plans)} max_prompt_tokens={max_estimate} "
                f"out={out}",
                flush=True,
            )
            if args.preflight_only:
                provenance["status"] = "preflight_only_complete"
                provenance["completed_at"] = utc_now()
                atomic_write_json(out / "provenance.json", provenance)
                return 0
            summary = run_api_and_evaluation(
                args,
                out=out,
                plans=plans,
                prompt_map=prompt_map,
                config_sha=config_sha,
                provenance=provenance,
            )
        except Exception as exc:
            failure = {
                "schema": SCHEMA_VERSION,
                "status": "failed_closed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(out / "failure.json", failure)
            print(
                f"FRONTIER_FAILED_CLOSED error={type(exc).__name__}: {exc} out={out}",
                file=sys.stderr,
                flush=True,
            )
            return 2
    pass_result = summary["pass_at_k"]
    compile_result = summary["compile_at_k"]
    print(
        f"FRONTIER_PASSK dataset={args.dataset_label} arm={args.arm} "
        f"provider={args.provider} model={args.model} K={args.k} "
        f"tasks={summary['tasks']} pass@{args.k}="
        f"{pass_result['successes']}/{pass_result['total']}="
        f"{pass_result['rate']:.4f} compile@{args.k}="
        f"{compile_result['successes']}/{compile_result['total']}="
        f"{compile_result['rate']:.4f} tokens={summary['usage']['total_tokens']} "
        f"out={out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
