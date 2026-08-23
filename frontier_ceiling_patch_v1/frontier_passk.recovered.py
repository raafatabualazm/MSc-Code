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
    COMPACT_F2_SYSTEM_PROMPT,
    F2_SCHEMA,
    SCHEMA_VERSION,
    CompactArtifactBundle,
    JsonlJournal,
    PreflightError,
    ResponseContractError,
    TokenBudget,
    atomic_write_json,
    atomic_write_jsonl,
    build_messages,
    classify_terminal_provider_response,
    complete_raw_disassembly,
    count_prompt_tokens,
    decode_f2,
    file_record,
    load_json,
    load_jsonl,
    public_dataclass,
    sha256_file,
    sha256_text,
    stable_sha256,
    utc_now,
    wilson_interval,
)

RUN_SCHEMA_VERSION = "audited-frontier-passk-v2"
FIXED_SLOT_POLICY_SCHEMA = "fixed-cap-exact-response-slot-v1"
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
    parser.add_argument(
        "--input-mode",
        choices=["decoded_compact", "prematerialized_f2"],
        default=os.environ.get("INPUT_MODE", "decoded_compact"),
        help=(
            "decoded_compact reconstructs F2 from sealed compact artifacts; "
            "prematerialized_f2 consumes a separately hash-pinned F2 JSONL and "
            "its manifest joined to a hash-pinned measure-only evaluator JSONL "
            "and seal."
        ),
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
    parser.add_argument(
        "--prompt-jsonl",
        type=Path,
        default=(
            Path(os.environ["PROMPT_JSONL"])
            if os.environ.get("PROMPT_JSONL")
            else None
        ),
        help="Pre-materialized lossless-F2 JSONL (prematerialized_f2 mode only).",
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=(
            Path(os.environ["PROMPT_MANIFEST"])
            if os.environ.get("PROMPT_MANIFEST")
            else None
        ),
        help="Manifest that seals --prompt-jsonl (prematerialized_f2 mode only).",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=(
            Path(os.environ["EVAL_JSONL"])
            if os.environ.get("EVAL_JSONL")
            else None
        ),
        help="Private held-out evaluator JSONL (prematerialized_f2 mode only).",
    )
    parser.add_argument(
        "--eval-seal",
        type=Path,
        default=(
            Path(os.environ["EVAL_SEAL"])
            if os.environ.get("EVAL_SEAL")
            else None
        ),
        help="Measure-only split seal for --eval-jsonl.",
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=(
            Path(os.environ["PAIR_MANIFEST"])
            if os.environ.get("PAIR_MANIFEST")
            else None
        ),
        help=(
            "Shared two-arm cohort seal (required by prematerialized_f2)."
        ),
    )
    parser.add_argument(
        "--pair-arm-key",
        choices=["opus_real_fn0_cfg", "codex_multifunction_cfg"],
        default=os.environ.get("PAIR_ARM_KEY", ""),
        help="Arm record in --pair-manifest bound to this run.",
    )
    parser.add_argument(
        "--expected-prompt-jsonl-sha256",
        default=os.environ.get("EXPECTED_PROMPT_JSONL_SHA256", ""),
    )
    parser.add_argument(
        "--expected-prompt-manifest-sha256",
        default=os.environ.get("EXPECTED_PROMPT_MANIFEST_SHA256", ""),
    )
    parser.add_argument(
        "--expected-eval-jsonl-sha256",
        default=os.environ.get("EXPECTED_EVAL_JSONL_SHA256", ""),
    )
    parser.add_argument(
        "--expected-eval-seal-sha256",
        default=os.environ.get("EXPECTED_EVAL_SEAL_SHA256", ""),
    )
    parser.add_argument(
        "--expected-pair-manifest-sha256",
        default=os.environ.get("EXPECTED_PAIR_MANIFEST_SHA256", ""),
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
        (
            "--expected-prompt-jsonl-sha256",
            args.expected_prompt_jsonl_sha256,
        ),
        (
            "--expected-prompt-manifest-sha256",
            args.expected_prompt_manifest_sha256,
        ),
        ("--expected-eval-jsonl-sha256", args.expected_eval_jsonl_sha256),
        ("--expected-eval-seal-sha256", args.expected_eval_seal_sha256),
        (
            "--expected-pair-manifest-sha256",
            args.expected_pair_manifest_sha256,
        ),
    ):
        normalized = value.strip().lower()
        if normalized and not re.fullmatch(r"[0-9a-f]{64}", normalized):
            parser.error(f"{option} must be a 64-character hexadecimal SHA-256")
    prematerialized_values = (
        args.prompt_jsonl,
        args.prompt_manifest,
        args.eval_jsonl,
        args.eval_seal,
        args.pair_manifest,
        args.pair_arm_key,
        args.expected_prompt_jsonl_sha256.strip(),
        args.expected_prompt_manifest_sha256.strip(),
        args.expected_eval_jsonl_sha256.strip(),
        args.expected_eval_seal_sha256.strip(),
        args.expected_pair_manifest_sha256.strip(),
    )
    if args.input_mode == "prematerialized_f2":
        if args.arm != "compact":
            parser.error(
                "--input-mode prematerialized_f2 requires --arm compact"
            )
        if args.limit:
            parser.error(
                "--input-mode prematerialized_f2 forbids --limit; evaluate the "
                "entire sealed cohort"
            )
        if args.expected_task_count != 175:
            parser.error(
                "--input-mode prematerialized_f2 is sealed for the full "
                "175-task paired cohort"
            )
        if any(not value for value in prematerialized_values):
            parser.error(
                "--input-mode prematerialized_f2 requires --prompt-jsonl, "
                "--prompt-manifest, --eval-jsonl, --eval-seal, "
                "--pair-manifest, --pair-arm-key, and all five corresponding "
                "--expected-*-sha256 values"
            )
        if not args.expected_evaluator_sha256.strip():
            parser.error(
                "--input-mode prematerialized_f2 requires "
                "--expected-evaluator-sha256"
            )
        if not args.expected_dart_sha256.strip():
            parser.error(
                "--input-mode prematerialized_f2 requires "
                "--expected-dart-sha256"
            )
    elif any(prematerialized_values):
        parser.error(
            "pre-materialized F2 paths/hashes require "
            "--input-mode prematerialized_f2"
        )
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


def fixed_slot_policy(args: argparse.Namespace) -> dict[str, Any]:
    """Return the complete response-slot sampling contract."""

    return {
        "schema": FIXED_SLOT_POLICY_SCHEMA,
        "requested_model": args.model,
        "resolved_model_must_equal_requested": True,
        "k": args.k,
        "fixed_max_output_tokens": args.max_output_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": args.extra_body,
        "request_timeout_seconds": args.timeout_seconds,
        "max_transport_attempts_per_slot": args.max_attempts_per_sample,
        "every_returned_response_consumes_one_slot": True,
        "retry_only_when_no_provider_response": True,
        "finish_reason_length_consumes_slot": True,
        "finish_reason_does_not_blanket_invalidate_extractable_code": True,
        "safe_extractable_fn0_is_evaluated": True,
        "unusable_candidate_is_terminal_failure": True,
        "no_candidate_resampling": True,
        "duplicate_response_id_is_fatal": True,
        "response_identity_and_usage_contract_is_fatal": True,
        "early_stopping": False,
    }


def runtime_core_source_path() -> Path:
    runner_path = Path(__file__).resolve()
    return runner_path.with_name(
        "frontier_core.recovered.py"
        if runner_path.name.endswith(".recovered.py")
        else "frontier_core.py"
    )


def config_for_hash(args: argparse.Namespace) -> dict[str, Any]:
    _api_key, base_url = resolve_api_configuration(args)
    try:
        openai_sdk_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        openai_sdk_version = None
    runner_path = Path(__file__).resolve()
    core_path = runtime_core_source_path()
    frontier_f2_path = runner_path.with_name("frontier_f2.py")
    slot_policy = fixed_slot_policy(args)
    config = {
        "schema": RUN_SCHEMA_VERSION,
        "provider": args.provider,
        "model_requested": args.model,
        "arm": args.arm,
        "input_mode": args.input_mode,
        "pair_arm_key": (
            args.pair_arm_key
            if args.input_mode == "prematerialized_f2"
            else None
        ),
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
        "expected_task_count": args.expected_task_count,
        "extra_body": args.extra_body,
        "evaluator_module": str(args.evaluator_module.expanduser().resolve()),
        "expected_evaluator_sha256": args.expected_evaluator_sha256.strip().lower(),
        "dart_binary": str(args.dart.expanduser().resolve()),
        "expected_dart_sha256": args.expected_dart_sha256.strip().lower(),
        "api_base_url_sha256": sha256_text(base_url.rstrip("/")),
        "api_base_url_redacted": redact_api_endpoint(base_url),
        "runtime_identity": {
            "runner_sha256": sha256_file(runner_path),
            "core_sha256": sha256_file(core_path),
            "frontier_f2_sha256": sha256_file(frontier_f2_path),
            "openai_sdk_version": openai_sdk_version,
        },
        "slot_policy": slot_policy,
        "slot_policy_sha256": stable_sha256(slot_policy),
    }
    if args.input_mode == "prematerialized_f2":
        config["sealed_inputs"] = {
            "prompt_jsonl": str(args.prompt_jsonl.expanduser().resolve()),
            "prompt_jsonl_sha256": (
                args.expected_prompt_jsonl_sha256.strip().lower()
            ),
            "prompt_manifest": str(
                args.prompt_manifest.expanduser().resolve()
            ),
            "prompt_manifest_sha256": (
                args.expected_prompt_manifest_sha256.strip().lower()
            ),
            "eval_jsonl": str(args.eval_jsonl.expanduser().resolve()),
            "eval_jsonl_sha256": (
                args.expected_eval_jsonl_sha256.strip().lower()
            ),
            "eval_seal": str(args.eval_seal.expanduser().resolve()),
            "eval_seal_sha256": (
                args.expected_eval_seal_sha256.strip().lower()
            ),
            "pair_manifest": str(
                args.pair_manifest.expanduser().resolve()
            ),
            "pair_manifest_sha256": (
                args.expected_pair_manifest_sha256.strip().lower()
            ),
            "pair_arm_key": args.pair_arm_key,
            "tokenizer_json": str(
                args.tokenizer_json.expanduser().resolve()
            ),
        }
    else:
        config.update(
            {
                "expected_dev_sha256": args.expected_dev_sha256,
                "expected_constants_sha256": args.expected_constants_sha256,
            }
        )
    return config


F2_MANIFEST_SCHEMA = "verified-api-readable-compact-v2"
EVAL_SEAL_SCHEMA = "compact-public-private-join-seal-v1"
PAIR_MANIFEST_SCHEMA = "frontier-enrichment-pair-v1"
PAIR_ARM_KEYS = frozenset(
    {"opus_real_fn0_cfg", "codex_multifunction_cfg"}
)
REQUIRED_PAIR_INVARIANTS = (
    "same_ordered_175_tasks",
    "same_acceptance_tests",
    "same_f2_grammar",
    "same_prompt_budget_contract",
    "measure_only_seals_verified",
    "provider_prompts_exclude_gold_and_tests",
    "prompt_artifact_hashes_verified",
    "no_prompt_truncation",
)
REQUIRED_F2_MANIFEST_INVARIANTS = (
    "all_artifact_hashes_verified",
    "all_row_contract_hashes_verified",
    "all_codec_roundtrips_verified",
    "all_student_constant_prefixes_verified",
    "all_f2_semantic_roundtrips_verified",
    "f2_system_prompt_self_contained_and_hashed",
    "all_complete_prompts_within_limit",
    "opaque_source_ids_expanded",
    "cfg_explicit",
)
REQUIRED_F2_ROW_VERIFICATION = {
    "artifact_hashes": True,
    "row_contract_hashes": True,
    "codec_text_roundtrip": True,
    "codec_token_id_roundtrip": True,
    "student_constant_prefix": True,
    "per_task_instruction_dictionary_roundtrip": True,
    "compact_semantic_f2_roundtrip": True,
    "branch_targets_reconstructed_from_cfg": True,
    "visible_task_symbols_one_token": True,
    "opaque_custom_ids_in_text": False,
}
OPTIONAL_STRONG_F2_ROW_VERIFICATION = (
    "all_user_functions_retained",
    "all_external_symbols_retained",
    "transfer_table_redundancy_proven",
)
FORBIDDEN_PROMPT_ROW_FIELDS = frozenset(
    {
        "dart_source",
        "tests",
        "acceptance_tests",
        "feedback_tests",
        "gold",
        "gold_source",
        "reference_solution",
        "solution",
        "target_code",
    }
)


def _pinned_file_record(
    path_value: Path | None,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if path_value is None:
        raise PreflightError(f"{label} path was not supplied")
    path = path_value.expanduser().resolve()
    if not path.is_file():
        raise PreflightError(f"{label} does not exist: {path}")
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PreflightError(f"expected {label} SHA-256 is invalid")
    record = file_record(path)
    if record["sha256"] != expected:
        raise PreflightError(
            f"{label} hash mismatch: expected {expected}, "
            f"got {record['sha256']}"
        )
    return path, record


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreflightError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise PreflightError(f"{label} must be positive")
    return result


def _validate_embedded_file_record(
    value: Any,
    actual: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} file binding is missing")
    embedded_sha = str(value.get("sha256") or "").strip().lower()
    if embedded_sha != actual["sha256"]:
        raise PreflightError(
            f"{label} file binding hash mismatch: expected "
            f"{actual['sha256']}, got {embedded_sha or '(missing)'}"
        )
    for field in ("bytes", "size_bytes"):
        if field in value:
            embedded_size = _nonnegative_int(
                value[field], f"{label}.{field}"
            )
            if embedded_size != actual["bytes"]:
                raise PreflightError(
                    f"{label}.{field} mismatch: expected {actual['bytes']}, "
                    f"got {embedded_size}"
                )


def _load_pinned_tokenizer(
    path_value: Path,
    *,
    expected_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    tokenizer_path = path_value.expanduser().resolve()
    if not tokenizer_path.is_file():
        raise PreflightError(
            f"pre-materialized tokenizer does not exist: {tokenizer_path}"
        )
    tokenizer_record = file_record(tokenizer_path)
    if tokenizer_record["sha256"] != expected_sha256:
        raise PreflightError(
            "pre-materialized tokenizer hash mismatch: expected "
            f"{expected_sha256}, got {tokenizer_record['sha256']}"
        )
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise PreflightError(
            "the 'tokenizers' package is required for sealed prompt "
            "token-count verification"
        ) from exc
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise PreflightError(
            f"cannot load sealed tokenizer {tokenizer_path}: {exc}"
        ) from exc
    return tokenizer, tokenizer_record


def _validate_prematerialized_eval_rows(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(rows) != expected_rows:
        raise PreflightError(
            f"evaluator dataset has {len(rows)} rows, expected {expected_rows}"
        )
    normalized: list[dict[str, Any]] = []
    task_ids: list[str] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        task_id = str(raw_row.get("task_id") or "")
        if not task_id:
            raise PreflightError(f"evaluator row {index} has no task_id")
        if task_id in seen:
            raise PreflightError(f"duplicate evaluator task_id: {task_id}")
        seen.add(task_id)
        task_ids.append(task_id)

        function = str(raw_row.get("function") or "")
        if function != "fn0":
            raise PreflightError(
                f"task {task_id} evaluator target function is not fn0"
            )
        language = str(raw_row.get("lang") or "").lower()
        if language != "dart":
            raise PreflightError(
                f"task {task_id} evaluator target language is not Dart"
            )
        source = raw_row.get("dart_source")
        if not isinstance(source, str) or not source.strip():
            raise PreflightError(
                f"task {task_id} has no dart_source provenance"
            )
        tests = raw_row.get("tests")
        if not isinstance(tests, str) or not tests.strip():
            raise PreflightError(f"task {task_id} has no tests")
        acceptance = raw_row.get("acceptance_tests", tests)
        if not isinstance(acceptance, str) or not acceptance.strip():
            raise PreflightError(
                f"task {task_id} has no acceptance_tests"
            )
        if tests != acceptance:
            raise PreflightError(
                f"task {task_id} tests and acceptance_tests differ"
            )
        row = dict(raw_row)
        row["acceptance_tests"] = acceptance
        normalized.append(row)
    return normalized, task_ids


def validate_prematerialized_f2_inputs(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load and independently revalidate one sealed F2/evaluator pair.

    The prompt file and private evaluator file are intentionally separate.
    Only the prompt row's ``text`` is placed in API messages; source and tests
    remain in the private evaluator row used after completion collection.
    """

    prompt_path, prompt_record = _pinned_file_record(
        args.prompt_jsonl,
        args.expected_prompt_jsonl_sha256,
        "prompt JSONL",
    )
    manifest_path, manifest_record = _pinned_file_record(
        args.prompt_manifest,
        args.expected_prompt_manifest_sha256,
        "prompt manifest",
    )
    eval_path, eval_record = _pinned_file_record(
        args.eval_jsonl,
        args.expected_eval_jsonl_sha256,
        "evaluator JSONL",
    )
    seal_path, seal_record = _pinned_file_record(
        args.eval_seal,
        args.expected_eval_seal_sha256,
        "evaluator seal",
    )
    pair_manifest_path, pair_manifest_record = _pinned_file_record(
        args.pair_manifest,
        args.expected_pair_manifest_sha256,
        "paired comparison manifest",
    )

    prompt_rows = load_jsonl(prompt_path, "pre-materialized F2 prompts")
    eval_rows_raw = load_jsonl(eval_path, "private evaluator dataset")
    manifest = load_json(manifest_path, "pre-materialized F2 manifest")
    seal = load_json(seal_path, "private evaluator seal")
    pair_manifest = load_json(
        pair_manifest_path, "paired comparison manifest"
    )

    if manifest.get("schema") != F2_MANIFEST_SCHEMA:
        raise PreflightError(
            "unexpected pre-materialized F2 manifest schema: "
            f"{manifest.get('schema')!r}"
        )
    if seal.get("schema") != EVAL_SEAL_SCHEMA:
        raise PreflightError(
            f"unexpected evaluator seal schema: {seal.get('schema')!r}"
        )
    if pair_manifest.get("schema") != PAIR_MANIFEST_SCHEMA:
        raise PreflightError(
            "unexpected paired comparison manifest schema: "
            f"{pair_manifest.get('schema')!r}"
        )
    pair_arms = pair_manifest.get("arms")
    if not isinstance(pair_arms, Mapping) or set(pair_arms) != PAIR_ARM_KEYS:
        raise PreflightError(
            "paired comparison manifest does not contain the exact Opus and "
            "Codex arm keys"
        )
    if args.pair_arm_key not in PAIR_ARM_KEYS:
        raise PreflightError(
            f"unsupported paired comparison arm key: {args.pair_arm_key!r}"
        )
    selected_pair_arm = pair_arms[args.pair_arm_key]
    if not isinstance(selected_pair_arm, Mapping):
        raise PreflightError(
            f"paired comparison arm {args.pair_arm_key!r} is malformed"
        )
    pair_invariants = pair_manifest.get("invariants")
    if not isinstance(pair_invariants, Mapping):
        raise PreflightError(
            "paired comparison manifest has no invariant map"
        )
    for invariant_name in REQUIRED_PAIR_INVARIANTS:
        if pair_invariants.get(invariant_name) is not True:
            raise PreflightError(
                f"paired comparison invariant {invariant_name!r} is not true"
            )
    if (
        _positive_int(pair_manifest.get("rows"), "pair manifest rows")
        != args.expected_task_count
    ):
        raise PreflightError(
            "paired comparison manifest row count differs from the requested "
            "cohort"
        )
    for pair_field, actual_record in (
        ("prompts", prompt_record),
        ("prompt_manifest", manifest_record),
        ("eval", eval_record),
        ("seal", seal_record),
    ):
        _validate_embedded_file_record(
            selected_pair_arm.get(pair_field),
            actual_record,
            f"pair manifest arm {args.pair_arm_key}.{pair_field}",
        )
    if seal.get("selected_role") != "measure":
        raise PreflightError(
            "evaluator seal is not selected_role='measure'"
        )
    if "training_allowed" in seal and seal.get("training_allowed") is not False:
        raise PreflightError(
            "evaluator seal does not explicitly forbid training"
        )
    if (
        "heldout_measure_only" in seal
        and seal.get("heldout_measure_only") is not True
    ):
        raise PreflightError(
            "evaluator seal does not identify a held-out measure-only split"
        )
    if (
        "raw_source_names_serialized" in seal
        and seal.get("raw_source_names_serialized") is not False
    ):
        raise PreflightError(
            "evaluator seal says raw private source names were serialized"
        )

    manifest_rows = _positive_int(manifest.get("rows"), "manifest.rows")
    seal_rows = _positive_int(seal.get("rows"), "seal.rows")
    if manifest_rows != args.expected_task_count:
        raise PreflightError(
            f"prompt manifest has {manifest_rows} rows, expected "
            f"{args.expected_task_count}"
        )
    if seal_rows != args.expected_task_count:
        raise PreflightError(
            f"evaluator seal has {seal_rows} rows, expected "
            f"{args.expected_task_count}"
        )
    if len(prompt_rows) != args.expected_task_count:
        raise PreflightError(
            f"prompt JSONL has {len(prompt_rows)} rows, expected "
            f"{args.expected_task_count}"
        )
    eval_rows, eval_task_ids = _validate_prematerialized_eval_rows(
        eval_rows_raw,
        expected_rows=args.expected_task_count,
    )

    _validate_embedded_file_record(
        manifest.get("output"), prompt_record, "manifest.output"
    )
    _validate_embedded_file_record(
        manifest.get("dataset"), eval_record, "manifest.dataset"
    )
    if str(seal.get("output_sha256") or "").lower() != eval_record["sha256"]:
        raise PreflightError(
            "evaluator seal output_sha256 does not bind the evaluator JSONL"
        )
    if "output" in seal:
        _validate_embedded_file_record(
            seal.get("output"), eval_record, "seal.output"
        )
    if "f2_output" in seal:
        _validate_embedded_file_record(
            seal.get("f2_output"), prompt_record, "seal.f2_output"
        )
    if "f2_manifest" in seal:
        _validate_embedded_file_record(
            seal.get("f2_manifest"),
            manifest_record,
            "seal.f2_manifest",
        )
    if (
        "frontier_f2_schema" in seal
        and seal.get("frontier_f2_schema") != F2_SCHEMA
    ):
        raise PreflightError("evaluator seal frontier F2 schema mismatch")
    if (
        "completion_attestation_id" in seal
        and seal.get("completion_attestation_id") != REQUIRED_ATTESTATION_ID
    ):
        raise PreflightError(
            "evaluator seal completion-attestation contract mismatch"
        )

    prompt_contract = manifest.get("f2_prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise PreflightError("manifest has no f2_prompt_contract")
    if prompt_contract.get("representation_schema") != F2_SCHEMA:
        raise PreflightError("manifest representation schema is not lossless F2")
    system_prompt = prompt_contract.get("system_prompt")
    if system_prompt != COMPACT_F2_SYSTEM_PROMPT:
        raise PreflightError(
            "manifest system prompt differs from the audited runner F2 prompt"
        )
    expected_system_sha = sha256_text(COMPACT_F2_SYSTEM_PROMPT)
    if prompt_contract.get("system_prompt_sha256") != expected_system_sha:
        raise PreflightError("manifest system-prompt hash mismatch")
    if pair_manifest.get("system_prompt_sha256") != expected_system_sha:
        raise PreflightError(
            "paired comparison system-prompt hash mismatch"
        )
    tokenizer_sha = str(
        prompt_contract.get("tokenizer_sha256") or ""
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", tokenizer_sha):
        raise PreflightError("manifest tokenizer SHA-256 is invalid")
    tokenizer, tokenizer_record = _load_pinned_tokenizer(
        args.tokenizer_json,
        expected_sha256=tokenizer_sha,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PreflightError("manifest has no artifact bindings")
    _validate_embedded_file_record(
        artifacts.get("tokenizer"),
        tokenizer_record,
        "manifest.artifacts.tokenizer",
    )
    frontier_f2_record = file_record(
        Path(__file__).with_name("frontier_f2.py")
    )
    # The original Opus v1 serializer predates the explicit frontier_f2 file
    # binding. Its caller-pinned output+manifest hashes, row text hashes, full
    # grammar decode, and recomputed tokenizer counts still bind exactly what
    # reaches the API. Newer Codex manifests do carry this stronger binding;
    # when present it is mandatory and exact.
    if "frontier_f2" in artifacts:
        _validate_embedded_file_record(
            artifacts.get("frontier_f2"),
            frontier_f2_record,
            "manifest.artifacts.frontier_f2",
        )
    for artifact_name, artifact_value in artifacts.items():
        if not isinstance(artifact_value, Mapping):
            raise PreflightError(
                f"manifest artifact {artifact_name!r} is not a file binding"
            )
        digest = str(artifact_value.get("sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PreflightError(
                f"manifest artifact {artifact_name!r} has an invalid SHA-256"
            )

    if (
        _positive_int(
            prompt_contract.get("max_prompt_tokens"),
            "manifest max_prompt_tokens",
        )
        != args.max_prompt_tokens
    ):
        raise PreflightError(
            "manifest prompt-token cap differs from the requested run cap"
        )
    if (
        _nonnegative_int(
            prompt_contract.get("chat_overhead_reserve"),
            "manifest chat_overhead_reserve",
        )
        != args.chat_overhead_reserve
    ):
        raise PreflightError(
            "manifest chat-overhead reserve differs from the requested run"
        )
    if prompt_contract.get("all_rows_within_limit") is not True:
        raise PreflightError(
            "manifest does not attest that all prompts fit the token cap"
        )
    invariants = manifest.get("invariants")
    if not isinstance(invariants, Mapping):
        raise PreflightError("manifest has no invariants")
    for name in REQUIRED_F2_MANIFEST_INVARIANTS:
        if invariants.get(name) is not True:
            raise PreflightError(
                f"manifest invariant {name!r} is not true"
            )
    codex_pair_arm = args.pair_arm_key == "codex_multifunction_cfg"
    if codex_pair_arm:
        for name in (
            "all_user_functions_retained",
            "all_external_symbols_retained",
            "transfer_table_redundancy_proven",
            "keyed_private_source_symbol_attestation_used",
            "raw_source_names_not_serialized",
        ):
            if invariants.get(name) is not True:
                raise PreflightError(
                    f"Codex arm manifest invariant {name!r} is not true"
                )
    source_attestation = manifest.get("source_symbol_attestation")
    if codex_pair_arm and not isinstance(source_attestation, Mapping):
        raise PreflightError(
            "Codex arm manifest has no keyed source-symbol attestation"
        )
    if source_attestation is not None:
        if not isinstance(source_attestation, Mapping):
            raise PreflightError(
                "manifest source-symbol attestation is malformed"
            )
        if source_attestation.get("used") is not True:
            raise PreflightError(
                "manifest source-symbol attestation was not used"
            )
        if source_attestation.get("is_keyed") is not True:
            raise PreflightError(
                "manifest source-symbol attestation is not keyed"
            )
        if source_attestation.get("raw_names_serialized") is not False:
            raise PreflightError(
                "manifest source-symbol attestation exposes raw names"
            )

    prompt_task_ids: list[str] = []
    seen_prompt_ids: set[str] = set()
    prompt_counts: list[tuple[str, dict[str, int]]] = []
    for index, row in enumerate(prompt_rows):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise PreflightError(f"prompt row {index} has no task_id")
        if task_id in seen_prompt_ids:
            raise PreflightError(f"duplicate prompt task_id: {task_id}")
        seen_prompt_ids.add(task_id)
        prompt_task_ids.append(task_id)
        leaked_fields = sorted(FORBIDDEN_PROMPT_ROW_FIELDS.intersection(row))
        if leaked_fields:
            raise PreflightError(
                f"prompt row {task_id} contains private field(s): "
                f"{', '.join(leaked_fields)}"
            )
        if row.get("schema") != SCHEMA_VERSION:
            raise PreflightError(
                f"prompt row {task_id} runner schema mismatch"
            )
        if row.get("representation_schema") != F2_SCHEMA:
            raise PreflightError(
                f"prompt row {task_id} representation is not lossless F2"
            )
        if row.get("system_prompt_sha256") != expected_system_sha:
            raise PreflightError(
                f"prompt row {task_id} system-prompt hash mismatch"
            )
        text_value = row.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise PreflightError(f"prompt row {task_id} has no F2 text")
        if "\x00" in text_value:
            raise PreflightError(f"prompt row {task_id} contains a NUL byte")
        if row.get("text_sha256") != sha256_text(text_value):
            raise PreflightError(
                f"prompt row {task_id} text SHA-256 mismatch"
            )
        try:
            _prefix, canonical = decode_f2(text_value)
        except Exception as exc:
            raise PreflightError(
                f"prompt row {task_id} is not valid F2: {exc}"
            ) from exc
        blocks = canonical.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise PreflightError(
                f"prompt row {task_id} F2 graph has no blocks"
            )
        verification = row.get("verified")
        if not isinstance(verification, Mapping):
            raise PreflightError(
                f"prompt row {task_id} has no verification map"
            )
        for name, expected_value in REQUIRED_F2_ROW_VERIFICATION.items():
            if verification.get(name) is not expected_value:
                raise PreflightError(
                    f"prompt row {task_id} verification {name!r} "
                    f"is not {expected_value!r}"
                )
        for name in OPTIONAL_STRONG_F2_ROW_VERIFICATION:
            if name in verification and verification.get(name) is not True:
                raise PreflightError(
                    f"prompt row {task_id} strong verification {name!r} "
                    "is not true"
                )
        if codex_pair_arm:
            for name in (
                *OPTIONAL_STRONG_F2_ROW_VERIFICATION,
                "keyed_source_symbol_attestation_bound",
                "raw_source_names_not_serialized",
            ):
                if verification.get(name) is not True:
                    raise PreflightError(
                        f"Codex prompt row {task_id} verification {name!r} "
                        "is not true"
                    )
            if row.get("source_symbol_attestation_used") is not True:
                raise PreflightError(
                    f"Codex prompt row {task_id} did not use source-symbol "
                    "attestation"
                )
            if row.get("source_symbol_attestation_is_keyed") is not True:
                raise PreflightError(
                    f"Codex prompt row {task_id} source-symbol attestation "
                    "is not keyed"
                )
            binding = row.get("source_symbol_attestation_binding")
            if not isinstance(binding, Mapping):
                raise PreflightError(
                    f"Codex prompt row {task_id} source-symbol binding is "
                    "missing"
                )
            if (
                binding.get("schema") != "dart-user-symbol-attestation-v1"
                or binding.get("complete") is not True
                or binding.get("raw_names_present") is not False
            ):
                raise PreflightError(
                    f"Codex prompt row {task_id} source-symbol binding is "
                    "not complete/private"
                )
            if row.get(
                "source_symbol_attestation_binding_sha256"
            ) != stable_sha256(binding):
                raise PreflightError(
                    f"Codex prompt row {task_id} source-symbol binding hash "
                    "mismatch"
                )
        messages = [
            {"role": "system", "content": COMPACT_F2_SYSTEM_PROMPT},
            {"role": "user", "content": text_value},
        ]
        observed_count = count_prompt_tokens(
            messages,
            tokenizer,
            chat_overhead_reserve=args.chat_overhead_reserve,
        )
        embedded_count = row.get("prompt_preflight")
        if not isinstance(embedded_count, Mapping):
            raise PreflightError(
                f"prompt row {task_id} has no prompt_preflight"
            )
        for field, expected_value in observed_count.items():
            observed_value = _nonnegative_int(
                embedded_count.get(field),
                f"prompt row {task_id} prompt_preflight.{field}",
            )
            if observed_value != expected_value:
                raise PreflightError(
                    f"prompt row {task_id} token count {field} mismatch: "
                    f"expected {expected_value}, got {observed_value}"
                )
        if observed_count["estimated_prompt_tokens"] > args.max_prompt_tokens:
            raise PreflightError(
                f"prompt row {task_id} has "
                f"{observed_count['estimated_prompt_tokens']} sealed-Qwen "
                f"tokenizer-estimate tokens, cap is {args.max_prompt_tokens}; "
                "refusing to truncate"
            )
        prompt_counts.append((task_id, observed_count))

    if prompt_task_ids != eval_task_ids:
        raise PreflightError(
            "ordered prompt task IDs differ from ordered evaluator task IDs"
        )
    task_set_sha = stable_sha256(prompt_task_ids)
    if manifest.get("task_set_sha256") != task_set_sha:
        raise PreflightError("manifest ordered task-set hash mismatch")
    if "task_set_sha256" in seal and seal.get("task_set_sha256") != task_set_sha:
        raise PreflightError("evaluator seal ordered task-set hash mismatch")
    if pair_manifest.get("ordered_task_ids_sha256") != task_set_sha:
        raise PreflightError(
            "paired comparison ordered task-set hash mismatch"
        )
    acceptance_test_hashes = [
        sha256_text(str(row["acceptance_tests"])) for row in eval_rows
    ]
    acceptance_sequence_sha = stable_sha256(acceptance_test_hashes)
    if (
        pair_manifest.get("ordered_acceptance_test_hashes_sha256")
        != acceptance_sequence_sha
    ):
        raise PreflightError(
            "paired comparison acceptance-test sequence hash mismatch"
        )
    for prompt_row, eval_row in zip(prompt_rows, eval_rows):
        task_id = str(prompt_row["task_id"])
        normalized_prompt_text = str(prompt_row["text"]).replace(
            "\r\n", "\n"
        )
        for private_field in (
            "dart_source",
            "tests",
            "acceptance_tests",
        ):
            private_text = str(eval_row[private_field]).replace(
                "\r\n", "\n"
            )
            if private_text and private_text in normalized_prompt_text:
                raise PreflightError(
                    f"prompt row {task_id} contains exact private "
                    f"{private_field}"
                )

    maximum_task_id, maximum_count = max(
        prompt_counts,
        key=lambda item: item[1]["estimated_prompt_tokens"],
    )
    if (
        _positive_int(
            prompt_contract.get("maximum_estimated_prompt_tokens"),
            "manifest maximum_estimated_prompt_tokens",
        )
        != maximum_count["estimated_prompt_tokens"]
    ):
        raise PreflightError(
            "manifest maximum prompt-token count does not match the rows"
        )
    if prompt_contract.get("maximum_task_id") != maximum_task_id:
        raise PreflightError(
            "manifest maximum-token task ID does not match the rows"
        )

    extraction = manifest.get("binary_constant_extraction_errors")
    if not isinstance(extraction, Mapping):
        raise PreflightError(
            "manifest binary-constant extraction accounting is missing"
        )
    error_task_ids = [
        str(row["task_id"])
        for row in prompt_rows
        if row.get("constants_extraction_error") not in (None, "")
    ]
    if _nonnegative_int(extraction.get("count"), "constant error count") != len(
        error_task_ids
    ):
        raise PreflightError(
            "manifest binary-constant extraction error count mismatch"
        )
    if extraction.get("task_ids") != error_task_ids:
        raise PreflightError(
            "manifest binary-constant extraction task IDs mismatch"
        )

    return {
        "prompt_rows": prompt_rows,
        "eval_rows": eval_rows,
        "task_ids": prompt_task_ids,
        "task_set_sha256": task_set_sha,
        "acceptance_test_sequence_sha256": acceptance_sequence_sha,
        "prompt_counts": {
            task_id: count for task_id, count in prompt_counts
        },
        "manifest": manifest,
        "seal": seal,
        "pair_manifest": pair_manifest,
        "pair_arm_key": args.pair_arm_key,
        "tokenizer": tokenizer,
        "records": {
            "prompt_jsonl": prompt_record,
            "prompt_manifest": manifest_record,
            "eval_jsonl": eval_record,
            "eval_seal": seal_record,
            "pair_manifest": pair_manifest_record,
            "tokenizer": tokenizer_record,
            "frontier_f2": frontier_f2_record,
        },
        "binary_constant_extraction_errors": {
            "count": len(error_task_ids),
            "task_ids": error_task_ids,
        },
    }


def _prepare_decoded_compact_run(
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
            "schema": RUN_SCHEMA_VERSION,
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
            "schema": RUN_SCHEMA_VERSION,
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
        "schema": RUN_SCHEMA_VERSION,
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
        "core": file_record(runtime_core_source_path()),
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
        if prior.get("schema") != RUN_SCHEMA_VERSION:
            raise RunFailure(
                "resume provenance uses an incompatible run schema; legacy "
                "response-rejection pilots cannot be resumed into v2"
            )
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


def _prepare_prematerialized_f2_run(
    args: argparse.Namespace,
    out: Path,
) -> tuple[
    Any,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
    dict[str, Any],
]:
    sealed = validate_prematerialized_f2_inputs(args)
    config = config_for_hash(args)
    config_sha = stable_sha256(config)

    plans: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    prompt_map: dict[str, dict[str, Any]] = {}
    for index, (prompt_row, eval_row) in enumerate(
        zip(sealed["prompt_rows"], sealed["eval_rows"])
    ):
        task_id = str(prompt_row["task_id"])
        text_value = str(prompt_row["text"])
        messages = [
            {"role": "system", "content": COMPACT_F2_SYSTEM_PROMPT},
            {"role": "user", "content": text_value},
        ]
        prompt_sha = stable_sha256(messages)
        token_count = sealed["prompt_counts"][task_id]
        task_record = {
            "schema": RUN_SCHEMA_VERSION,
            "task_index": index,
            "task_id": task_id,
            "function": "fn0",
            "language": "Dart",
            "dart_source_sha256": sha256_text(eval_row["dart_source"]),
            "tests_sha256": sha256_text(eval_row["tests"]),
            "acceptance_tests_sha256": sha256_text(
                eval_row["acceptance_tests"]
            ),
            "tests_equal_acceptance_tests": True,
            "input_mode": "prematerialized_f2",
            "representation_schema": prompt_row["representation_schema"],
            "prematerialized_row_sha256": stable_sha256(prompt_row),
            "prematerialized_text_sha256": prompt_row["text_sha256"],
            "compact_ids_sha256": prompt_row.get("compact_ids_sha256"),
            "compact_text_sha256": prompt_row.get("compact_text_sha256"),
            "canonical_sha256": prompt_row.get("canonical_sha256"),
            "constants_record_sha256": prompt_row.get(
                "constants_record_sha256"
            ),
            "constants_extraction_error": prompt_row.get(
                "constants_extraction_error"
            ),
            "constant_prefix_tokens": prompt_row.get(
                "constant_prefix_tokens"
            ),
            "graph_tokens": prompt_row.get("graph_tokens"),
            "raw_control": None,
        }
        prompt_record = {
            "schema": RUN_SCHEMA_VERSION,
            "task_id": task_id,
            "arm": args.arm,
            "input_mode": "prematerialized_f2",
            "representation_schema": F2_SCHEMA,
            "prompt_sha256": prompt_sha,
            "messages": messages,
            "token_count": token_count,
            "token_count_basis": "sealed_qwen_tokenizer_estimate",
            "tokenizer_sha256": sealed["records"]["tokenizer"]["sha256"],
            "source_prompt_jsonl_sha256": sealed["records"][
                "prompt_jsonl"
            ]["sha256"],
            "source_prompt_row_sha256": stable_sha256(prompt_row),
            "never_truncated": True,
            "tests_exposed": False,
            "source_exposed": False,
        }
        plans.append(
            {
                "task_id": task_id,
                "row": eval_row,
                "messages": messages,
                "prompt_sha256": prompt_sha,
                "estimated_prompt_tokens": token_count[
                    "estimated_prompt_tokens"
                ],
            }
        )
        task_records.append(task_record)
        prompt_records.append(prompt_record)
        prompt_map[task_id] = prompt_record

    evaluator_module, evaluator_record = import_evaluator(
        args.evaluator_module,
        args.expected_evaluator_sha256,
        dart_binary=args.dart,
        expected_dart_hash=args.expected_dart_sha256,
        validate_dart=False,
    )
    sealed_evaluator_sha = str(
        sealed["seal"].get("evaluator_sha256") or ""
    ).strip().lower()
    if (
        sealed_evaluator_sha
        and sealed_evaluator_sha != evaluator_record["sha256"]
    ):
        raise PreflightError(
            "evaluator module differs from the evaluator bound by the split seal"
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
        "schema": RUN_SCHEMA_VERSION,
        "status": "preflight_complete",
        "created_at": utc_now(),
        "run_id": out.name,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "config": config,
        "config_sha256": config_sha,
        "task_set_sha256": sealed["task_set_sha256"],
        "acceptance_test_sequence_sha256": sealed[
            "acceptance_test_sequence_sha256"
        ],
        "tasks_selected": len(plans),
        "binary_constant_extraction_errors": {
            **sealed["binary_constant_extraction_errors"],
            "interpretation": (
                "The prompt rows are byte-for-byte sealed serializer outputs; "
                "an extraction error means that source artifact did not add "
                "binary constants for the named task."
            ),
        },
        "dataset_rows_before_limit": len(sealed["eval_rows"]),
        "dataset": sealed["records"]["eval_jsonl"],
        "artifacts": sealed["records"],
        "source_f2_manifest_claims": {
            "schema": sealed["manifest"]["schema"],
            "task_set_sha256": sealed["manifest"]["task_set_sha256"],
            "rows": sealed["manifest"]["rows"],
            "system_prompt_sha256": sealed["manifest"][
                "f2_prompt_contract"
            ]["system_prompt_sha256"],
            "representation_schema": sealed["manifest"][
                "f2_prompt_contract"
            ]["representation_schema"],
        },
        "source_eval_seal_claims": {
            "schema": sealed["seal"]["schema"],
            "selected_role": sealed["seal"]["selected_role"],
            "rows": sealed["seal"]["rows"],
            "output_sha256": sealed["seal"]["output_sha256"],
            "task_set_sha256": sealed["seal"].get("task_set_sha256"),
            "heldout_measure_only": sealed["seal"].get(
                "heldout_measure_only"
            ),
            "training_allowed": sealed["seal"].get("training_allowed"),
        },
        "source_pair_manifest_claims": {
            "schema": sealed["pair_manifest"]["schema"],
            "sha256": sealed["records"]["pair_manifest"]["sha256"],
            "pair_arm_key": sealed["pair_arm_key"],
            "rows": sealed["pair_manifest"]["rows"],
            "ordered_task_ids_sha256": sealed["pair_manifest"][
                "ordered_task_ids_sha256"
            ],
            "ordered_acceptance_test_hashes_sha256": sealed[
                "pair_manifest"
            ]["ordered_acceptance_test_hashes_sha256"],
            "system_prompt_sha256": sealed["pair_manifest"][
                "system_prompt_sha256"
            ],
        },
        "prompt_token_accounting": {
            "preflight_count_basis": "sealed_qwen_tokenizer_estimate",
            "preflight_count_is_provider_authoritative": False,
            "provider_usage_prompt_tokens_checked_after_every_response": True,
            "provider_usage_is_authoritative_for_deepseek_cap": True,
            "provider_prompt_tokens_must_not_exceed": args.max_prompt_tokens,
        },
        "evaluator": evaluator_record,
        "runner": file_record(Path(__file__)),
        "core": file_record(runtime_core_source_path()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "preflight_invariants": {
            "input_mode_is_prematerialized_f2": True,
            "all_four_input_files_sha256_pinned": True,
            "shared_pair_manifest_sha256_pinned": True,
            "selected_pair_arm_artifact_bindings_verified": True,
            "paired_acceptance_test_sequence_sha256_verified": True,
            "paired_system_prompt_sha256_verified": True,
            "prompt_manifest_output_binding_verified": True,
            "prompt_manifest_eval_dataset_binding_verified": True,
            "eval_seal_output_binding_verified": True,
            "eval_split_selected_role_measure": True,
            "eval_split_not_marked_training_allowed": (
                sealed["seal"].get("training_allowed") is not True
            ),
            "expected_dataset_rows_verified": True,
            "unique_task_ids": True,
            "ordered_prompt_eval_task_ids_identical": True,
            "ordered_task_set_sha256_verified": True,
            "tests_equal_acceptance_tests": True,
            "completion_attested_evaluator_contract_verified": True,
            "evaluator_sha256_pinned": bool(
                args.expected_evaluator_sha256.strip()
            ),
            "dart_binary_sha256_pinned": bool(
                args.expected_dart_sha256.strip()
            ),
            "acceptance_test_main_attestable_for_every_task": True,
            "legacy_returncode_only_evaluator_used": False,
            "prematerialized_f2_manifest_invariants_verified": True,
            "per_row_f2_text_sha256_verified": True,
            "per_row_f2_syntax_verified": True,
            "per_row_f2_verification_map_verified": True,
            "tokenizer_sha256_verified": True,
            "per_row_prompt_tokens_recomputed": True,
            "manifest_prompt_maximum_recomputed": True,
            "opaque_custom_ids_not_sent_to_api": True,
            "prompt_token_cap_checked_for_every_task": True,
            "prompts_never_truncated": True,
            "tests_not_exposed_to_teacher": True,
            "source_not_exposed_to_teacher": True,
            "exact_private_source_and_tests_absent_from_f2_text": True,
            "codex_keyed_source_symbol_attestation_verified": (
                sealed["pair_arm_key"] == "codex_multifunction_cfg"
            ),
            "local_prompt_counts_are_sealed_qwen_estimates": True,
            "provider_reported_prompt_tokens_are_authoritative": True,
            "raw_disassembly_generated": False,
            "raw_asm_cache_used": False,
        },
    }
    existing_provenance = out / "provenance.json"
    if existing_provenance.is_file():
        prior = json.loads(existing_provenance.read_text(encoding="utf-8"))
        if not args.resume:
            raise RunFailure(
                f"output already exists and --no-resume was requested: {out}"
            )
        if prior.get("schema") != RUN_SCHEMA_VERSION:
            raise RunFailure(
                "resume provenance uses an incompatible run schema; legacy "
                "response-rejection pilots cannot be resumed into v2"
            )
        if prior.get("config_sha256") != config_sha:
            raise RunFailure(
                "resume config does not match existing provenance: "
                f"{prior.get('config_sha256')} != {config_sha}"
            )
        if prior.get("task_set_sha256") != sealed["task_set_sha256"]:
            raise RunFailure("resume task set does not match existing provenance")
        if (
            (prior.get("dataset") or {}).get("sha256")
            != sealed["records"]["eval_jsonl"]["sha256"]
        ):
            raise RunFailure(
                "resume evaluator-dataset hash does not match provenance"
            )
        prior_artifacts = prior.get("artifacts") or {}
        for artifact_name in (
            "prompt_jsonl",
            "prompt_manifest",
            "eval_jsonl",
            "eval_seal",
            "pair_manifest",
            "tokenizer",
            "frontier_f2",
        ):
            if (
                (prior_artifacts.get(artifact_name) or {}).get("sha256")
                != sealed["records"][artifact_name]["sha256"]
            ):
                raise RunFailure(
                    f"resume {artifact_name} hash does not match provenance"
                )
        provenance["created_at"] = prior.get(
            "created_at", provenance["created_at"]
        )
        provenance["resumed_at"] = utc_now()
    atomic_write_json(out / "provenance.json", provenance)
    atomic_write_jsonl(out / "tasks.jsonl", task_records)
    atomic_write_jsonl(out / "prompts.jsonl", prompt_records)
    return sealed["tokenizer"], plans, prompt_map, config_sha, provenance


def enforce_output_state_policy(
    args: argparse.Namespace,
    out: Path,
) -> None:
    if not args.resume:
        state_names = (
            "provenance.json",
            "tasks.jsonl",
            "prompts.jsonl",
            "attempts.jsonl",
            "outcomes.jsonl",
            "summary.json",
            "manifest.json",
            "failure.json",
        )
        existing_state = [
            name for name in state_names if (out / name).exists()
        ]
        if existing_state:
            raise RunFailure(
                "--no-resume refuses existing run state: "
                + ", ".join(existing_state)
            )


def prepare_run(
    args: argparse.Namespace,
    out: Path,
) -> tuple[
    Any,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
    dict[str, Any],
]:
    enforce_output_state_policy(args, out)
    if args.input_mode == "prematerialized_f2":
        return _prepare_prematerialized_f2_run(args, out)
    return _prepare_decoded_compact_run(args, out)


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
    requested_model: str,
    k: int,
    max_prompt_tokens: int,
    requested_max_tokens: int,
    max_transport_attempts_per_slot: int,
    slot_policy_sha256: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], int]]:
    terminal: dict[tuple[str, int], dict[str, Any]] = {}
    next_attempt: dict[tuple[str, int], int] = {}
    if not path.is_file():
        return terminal, next_attempt
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    response_ids: set[str] = set()
    for row in load_jsonl(path, "attempt journal"):
        if row.get("schema") != RUN_SCHEMA_VERSION:
            raise RunFailure("attempt journal uses an incompatible run schema")
        if row.get("config_sha256") != config_sha:
            raise RunFailure("attempt journal contains a foreign config fingerprint")
        if row.get("slot_policy_sha256") != slot_policy_sha256:
            raise RunFailure("attempt journal contains a foreign slot policy")
        task_id = str(row.get("task_id") or "")
        sample_index = int(row.get("sample_index", -1))
        attempt_index = int(row.get("attempt_index", -1))
        key = (task_id, sample_index)
        if (
            task_id not in prompt_map
            or sample_index < 0
            or sample_index >= k
            or attempt_index < 0
        ):
            raise RunFailure("attempt journal contains an invalid task/sample index")
        if row.get("prompt_sha256") != prompt_map[task_id].get("prompt_sha256"):
            raise RunFailure("attempt journal prompt fingerprint mismatch")
        if row.get("requested_max_tokens") != requested_max_tokens:
            raise RunFailure("attempt journal fixed completion cap mismatch")
        grouped.setdefault(key, []).append(row)
        budget_charge = row.get("budget_charge_tokens")
        if isinstance(budget_charge, bool) or not isinstance(budget_charge, int):
            raise RunFailure("attempt journal has an invalid budget charge")
        if budget_charge < 0:
            raise RunFailure("attempt journal has a negative budget charge")
        if budget_charge > 0:
            if not budget.reserve(budget_charge):
                raise RunFailure("resumed attempts already exceed token budget")
            budget.settle(budget_charge, budget_charge)

    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda value: int(value["attempt_index"]))
        observed_indices = [int(value["attempt_index"]) for value in ordered]
        if observed_indices != list(range(len(ordered))):
            raise RunFailure(
                f"attempt journal indices are not contiguous from zero for {key}"
            )
        if len(ordered) > max_transport_attempts_per_slot:
            raise RunFailure(
                f"attempt journal exceeds the transport-attempt cap for {key}"
            )
        saw_terminal = False
        for row in ordered:
            response_received = row.get("response_received")
            slot_terminal = row.get("slot_terminal")
            if type(response_received) is not bool or type(slot_terminal) is not bool:
                raise RunFailure("attempt journal lacks exact response/terminal flags")
            if saw_terminal:
                raise RunFailure(f"attempt journal has a post-terminal attempt for {key}")
            if response_received:
                if slot_terminal is not True:
                    raise RunFailure("a returned provider response is not terminal")
                raw_response = row.get("response")
                if not isinstance(raw_response, Mapping):
                    raise RunFailure("terminal response is missing its raw response")
                try:
                    classified = classify_terminal_provider_response(
                        dict(raw_response),
                        expected_model=requested_model,
                        max_prompt_tokens=max_prompt_tokens,
                        requested_max_tokens=requested_max_tokens,
                    )
                except ResponseContractError as exc:
                    raise RunFailure(
                        f"resumed terminal response violates response contract: {exc}"
                    ) from exc
                expected_fields = {
                    "response_id": classified.response_id,
                    "resolved_model": classified.response_model,
                    "response_created": classified.response_created,
                    "finish_reason": classified.finish_reason,
                    "candidate_valid": classified.candidate_valid,
                    "terminal_reason": classified.terminal_reason,
                    "content": classified.content,
                    "reasoning_content": classified.reasoning_content,
                    "code": classified.code,
                    "code_sha256": classified.code_sha256,
                    "usage": classified.usage,
                }
                for field, expected in expected_fields.items():
                    if row.get(field) != expected:
                        raise RunFailure(
                            f"attempt journal terminal field {field!r} was tampered"
                        )
                if classified.response_id in response_ids:
                    raise RunFailure(
                        f"duplicate terminal response id: {classified.response_id}"
                    )
                response_ids.add(classified.response_id)
                if row.get("transport_retry") is not False:
                    raise RunFailure("terminal response is marked as a transport retry")
                if row.get("transport_error") is not None:
                    raise RunFailure("terminal response has a transport error")
                if row.get("fatal_response_contract") is not False:
                    raise RunFailure("completed terminal response is marked fatal")
                if row.get("budget_charge_tokens") != classified.usage[
                    "total_tokens"
                ]:
                    raise RunFailure(
                        "terminal response budget charge disagrees with usage"
                    )
                terminal[key] = row
                saw_terminal = True
            else:
                if slot_terminal is not False:
                    raise RunFailure("response-less attempt is marked terminal")
                if row.get("candidate_valid") is not None:
                    raise RunFailure(
                        "response-less attempt has a candidate-validity value"
                    )
                if row.get("terminal_reason") is not None:
                    raise RunFailure("response-less attempt has a terminal reason")
                if row.get("response") is not None or row.get("usage") is not None:
                    raise RunFailure("response-less attempt contains response data")
                if row.get("transport_retry") is not True:
                    raise RunFailure("response-less attempt is not a transport retry")
                if row.get("retryable_transport") is not True:
                    raise RunFailure(
                        "attempt journal contains a non-retryable API exception"
                    )
                if row.get("fatal_response_contract") is not False:
                    raise RunFailure("transport retry is marked response-contract fatal")
                if not str(row.get("transport_error") or ""):
                    raise RunFailure("transport retry lacks an error")
                expected_transport_charge = (
                    max_prompt_tokens + requested_max_tokens
                )
                if row.get("budget_charge_tokens") != expected_transport_charge:
                    raise RunFailure(
                        "transport retry budget charge is not the full reservation"
                    )
        next_attempt[key] = len(ordered)
    return terminal, next_attempt


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
        if row.get("schema") != RUN_SCHEMA_VERSION:
            raise RunFailure("outcome journal uses an incompatible run schema")
        if row.get("config_sha256") != config_sha:
            raise RunFailure("outcome journal contains a foreign config fingerprint")
        task_id = str(row.get("task_id") or "")
        sample_index = int(row.get("sample_index", -1))
        attempt_id = str(row.get("attempt_id") or "")
        if not task_id or sample_index < 0 or not attempt_id:
            raise RunFailure("outcome journal contains an invalid identity")
        key = (task_id, sample_index, attempt_id)
        if key in existing:
            raise RunFailure("outcome journal contains a duplicate outcome")
        candidate_valid = row.get("candidate_valid")
        evaluation_performed = row.get("evaluation_performed")
        if type(candidate_valid) is not bool or type(evaluation_performed) is not bool:
            raise RunFailure("outcome journal lacks candidate/evaluation flags")
        runs = row.get("stability_runs")
        if not isinstance(runs, list):
            raise RunFailure("outcome journal stability runs are malformed")
        if row.get("evaluator_sha256") != evaluator_sha256:
            raise RunFailure("outcome journal evaluator fingerprint mismatch")
        if candidate_valid:
            if evaluation_performed is not True or not runs:
                raise RunFailure("evaluable outcome lacks stability-run evidence")
            if row.get("completion_attestation_id") != REQUIRED_ATTESTATION_ID:
                raise RunFailure("outcome journal attestation identity mismatch")
            if row.get("completion_attestation_enforced") is not True:
                raise RunFailure(
                    "evaluable outcome lacks completion-attestation enforcement"
                )
            for run in runs:
                if not isinstance(run, Mapping):
                    raise RunFailure("outcome journal has an invalid stability run")
                if run.get("completion_attestation_id") != REQUIRED_ATTESTATION_ID:
                    raise RunFailure("stability-run attestation identity mismatch")
                if run.get("completion_attestation_required") is not True:
                    raise RunFailure(
                        "stability run did not require completion attestation"
                    )
                if type(run.get("compiled")) is not bool or type(
                    run.get("passed")
                ) is not bool:
                    raise RunFailure("stability run has non-boolean results")
                if type(run.get("completion_attestation_satisfied")) is not bool:
                    raise RunFailure(
                        "stability run has a non-boolean attestation result"
                    )
                if run.get("completion_attestation_satisfied") != run.get(
                    "passed"
                ):
                    raise RunFailure(
                        "stability-run pass disagrees with completion attestation"
                    )
            all_compiled = all(run["compiled"] for run in runs)
            all_passed = all(run["passed"] for run in runs)
            if row.get("compiled") is not all_compiled:
                raise RunFailure(
                    "outcome compile result disagrees with stability runs"
                )
            if row.get("passed") is not all_passed:
                raise RunFailure(
                    "outcome pass result disagrees with stability runs"
                )
            if (
                row.get("completion_attestation_satisfied_all_runs")
                is not all_passed
            ):
                raise RunFailure(
                    "outcome attestation result disagrees with stability runs"
                )
        else:
            if evaluation_performed is not False or runs:
                raise RunFailure(
                    "terminal invalid candidate must have no evaluator executions"
                )
            if row.get("compiled") is not False or row.get("passed") is not False:
                raise RunFailure("terminal invalid candidate must be a failed outcome")
            if row.get("completion_attestation_enforced") is not False:
                raise RunFailure(
                    "terminal invalid candidate has a false attestation claim"
                )
            if row.get("completion_attestation_satisfied_all_runs") is not False:
                raise RunFailure(
                    "terminal invalid candidate has a false attestation result"
                )
        existing[key] = row
    return existing


def retry_delay(args: argparse.Namespace, attempt_index: int) -> float:
    base = min(
        args.retry_max_seconds,
        args.retry_base_seconds * (2 ** min(attempt_index, 8)),
    )
    return min(args.retry_max_seconds, base * random.uniform(0.8, 1.2))


def is_retryable_api_exception(exc: Exception) -> bool:
    """Return whether an exception represents a response-less retryable call."""

    status = getattr(exc, "status_code", None)
    if isinstance(status, bool):
        status = None
    if isinstance(status, int):
        return status in {408, 409, 429} or status >= 500
    class_names = {base.__name__ for base in type(exc).__mro__}
    return bool(
        class_names.intersection({"APIConnectionError", "APITimeoutError"})
    )


def make_request(
    client: Any,
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    *,
    requested_max_tokens: int,
) -> Any:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": requested_max_tokens,
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
    attempts_path = out / "attempts.jsonl"
    outcomes_path = out / "outcomes.jsonl"
    if not args.resume:
        existing_journals = [
            str(path.name)
            for path in (attempts_path, outcomes_path)
            if path.exists()
        ]
        if existing_journals:
            raise RunFailure(
                "--no-resume refuses existing journals: "
                + ", ".join(existing_journals)
            )
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
    slot_policy_sha256 = provenance["config"]["slot_policy_sha256"]
    terminal_resume, next_attempt = load_resume_attempts(
        attempts_path,
        config_sha=config_sha,
        prompt_map=prompt_map,
        budget=budget,
        requested_model=args.model,
        k=args.k,
        max_prompt_tokens=args.max_prompt_tokens,
        requested_max_tokens=args.max_output_tokens,
        max_transport_attempts_per_slot=args.max_attempts_per_sample,
        slot_policy_sha256=slot_policy_sha256,
    )
    attempts = JsonlJournal(attempts_path)
    resumed_outcomes = load_resume_outcomes(
        outcomes_path,
        config_sha=config_sha,
        evaluator_sha256=evaluator_record["sha256"],
    )
    terminal_attempt_keys = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in terminal_resume.items()
    }
    orphan_outcomes = sorted(set(resumed_outcomes) - terminal_attempt_keys)
    if orphan_outcomes:
        raise RunFailure(
            f"outcome journal has {len(orphan_outcomes)} orphan record(s); "
            f"first={orphan_outcomes[0]}"
        )
    outcomes = JsonlJournal(outcomes_path)
    stop = threading.Event()
    worst_case_reservation = args.max_prompt_tokens + args.max_output_tokens
    response_id_lock = threading.Lock()
    terminal_response_ids = {
        str(row["response_id"]) for row in terminal_resume.values()
    }

    def run_task(plan: dict[str, Any]) -> dict[str, Any]:
        task_id = plan["task_id"]
        terminal_slots: list[dict[str, Any]] = []
        for sample_index in range(args.k):
            key_tuple = (task_id, sample_index)
            resumed = terminal_resume.get(key_tuple)
            if resumed is not None:
                terminal_slots.append(
                    {
                        "sample_index": sample_index,
                        "code": resumed["code"],
                        "code_sha256": resumed["code_sha256"],
                        "attempt_id": resumed["attempt_id"],
                        "response_id": resumed["response_id"],
                        "finish_reason": resumed["finish_reason"],
                        "candidate_valid": resumed["candidate_valid"],
                        "terminal_reason": resumed["terminal_reason"],
                        "resumed": True,
                    }
                )
                continue
            first_attempt = next_attempt.get(key_tuple, 0)
            terminal_record: dict[str, Any] | None = None
            for attempt_index in range(first_attempt, args.max_attempts_per_sample):
                if stop.is_set():
                    raise RunFailure(f"task {task_id} stopped after another fatal error")
                attempt_id = (
                    f"{safe_label(task_id)}.s{sample_index}.a{attempt_index}."
                    f"{uuid.uuid4().hex[:10]}"
                )
                base_record: dict[str, Any] = {
                    "schema": RUN_SCHEMA_VERSION,
                    "record_type": "api_attempt",
                    "attempt_id": attempt_id,
                    "config_sha256": config_sha,
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "attempt_index": attempt_index,
                    "prompt_sha256": plan["prompt_sha256"],
                    "requested_model": args.model,
                    "requested_max_tokens": args.max_output_tokens,
                    "provider": args.provider,
                    "slot_policy_sha256": slot_policy_sha256,
                    "started_at": utc_now(),
                }
                if not budget.reserve(worst_case_reservation):
                    raise RunFailure(
                        f"token budget cannot reserve another request for task {task_id}"
                    )
                reservation_open = True
                try:
                    response = make_request(
                        client,
                        args,
                        plan["messages"],
                        requested_max_tokens=args.max_output_tokens,
                    )
                    raw_response = response_to_dict(response)
                    settled = usage_total(raw_response, worst_case_reservation)
                    budget.settle(worst_case_reservation, settled)
                    reservation_open = False
                    try:
                        terminal = classify_terminal_provider_response(
                            response,
                            expected_model=args.model,
                            max_prompt_tokens=args.max_prompt_tokens,
                            requested_max_tokens=args.max_output_tokens,
                        )
                    except ResponseContractError as exc:
                        record = dict(base_record)
                        record.update(
                            {
                                "finished_at": utc_now(),
                                "response_received": True,
                                "slot_terminal": True,
                                "candidate_valid": False,
                                "terminal_reason": (
                                    "fatal_response_contract:" + str(exc)
                                ),
                                "transport_retry": False,
                                "transport_error": None,
                                "fatal_response_contract": True,
                                "budget_charge_tokens": settled,
                                "usage": (
                                    raw_response.get("usage")
                                    if isinstance(raw_response.get("usage"), Mapping)
                                    else None
                                ),
                                "response": raw_response,
                            }
                        )
                        attempts.append(record)
                        raise RunFailure(
                            f"returned provider response violates the fatal "
                            f"identity/usage contract: {exc}"
                        ) from exc
                    else:
                        with response_id_lock:
                            duplicate_response_id = (
                                terminal.response_id in terminal_response_ids
                            )
                            if not duplicate_response_id:
                                terminal_response_ids.add(terminal.response_id)
                        record = dict(base_record)
                        record.update(
                            {
                                "finished_at": utc_now(),
                                "response_received": True,
                                "slot_terminal": True,
                                "candidate_valid": terminal.candidate_valid,
                                "terminal_reason": terminal.terminal_reason,
                                "transport_retry": False,
                                "transport_error": None,
                                "fatal_response_contract": False,
                                "response_id": terminal.response_id,
                                "resolved_model": terminal.response_model,
                                "response_created": terminal.response_created,
                                "finish_reason": terminal.finish_reason,
                                "budget_charge_tokens": settled,
                                "usage": terminal.usage,
                                "content": terminal.content,
                                "reasoning_content": terminal.reasoning_content,
                                "code": terminal.code,
                                "code_sha256": terminal.code_sha256,
                                "response": terminal.raw_response,
                            }
                        )
                        attempts.append(record)
                        if duplicate_response_id:
                            raise RunFailure(
                                f"duplicate terminal response id: "
                                f"{terminal.response_id}"
                            )
                        terminal_record = record
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
                    retryable = is_retryable_api_exception(exc)
                    record.update(
                        {
                            "finished_at": utc_now(),
                            "response_received": False,
                            "slot_terminal": False,
                            "candidate_valid": None,
                            "terminal_reason": None,
                            "transport_retry": True,
                            "retryable_transport": retryable,
                            "transport_error": (
                                f"api_exception:{type(exc).__name__}:"
                                f"{str(exc)[:1000]}"
                            ),
                            "fatal_response_contract": False,
                            "budget_charge_tokens": worst_case_reservation,
                            "usage": None,
                            "response": None,
                        }
                    )
                    attempts.append(record)
                    if not retryable:
                        raise RunFailure(
                            f"non-retryable API exception before a provider "
                            f"response: {type(exc).__name__}: {exc}"
                        ) from exc
                if (
                    terminal_record is None
                    and attempt_index + 1 < args.max_attempts_per_sample
                ):
                    delay = retry_delay(args, attempt_index)
                    if stop.wait(delay):
                        raise RunFailure(
                            f"task {task_id} stopped during retry backoff"
                        )
            if terminal_record is None:
                raise RunFailure(
                    f"task {task_id} sample {sample_index} did not receive a "
                    f"provider response in {args.max_attempts_per_sample} "
                    f"transport attempts"
                )
            terminal_slots.append(
                {
                    "sample_index": sample_index,
                    "code": terminal_record["code"],
                    "code_sha256": terminal_record["code_sha256"],
                    "attempt_id": terminal_record["attempt_id"],
                    "response_id": terminal_record["response_id"],
                    "finish_reason": terminal_record["finish_reason"],
                    "candidate_valid": terminal_record["candidate_valid"],
                    "terminal_reason": terminal_record["terminal_reason"],
                    "resumed": False,
                }
            )

        if len(terminal_slots) != args.k:
            raise RunFailure(
                f"task {task_id} has {len(terminal_slots)} terminal provider "
                f"responses, expected {args.k}"
            )
        candidate_outcomes: list[dict[str, Any]] = []
        for candidate in terminal_slots:
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
                if (
                    resumed_outcome.get("candidate_valid")
                    != candidate["candidate_valid"]
                    or resumed_outcome.get("response_id")
                    != candidate["response_id"]
                    or resumed_outcome.get("finish_reason")
                    != candidate["finish_reason"]
                    or resumed_outcome.get("terminal_reason")
                    != candidate["terminal_reason"]
                ):
                    raise RunFailure(
                        f"resumed outcome terminal receipt mismatch for {outcome_key}"
                    )
                runs = resumed_outcome.get("stability_runs") or []
                expected_runs = (
                    args.eval_stability_runs if candidate["candidate_valid"] else 0
                )
                if len(runs) != expected_runs:
                    raise RunFailure(
                        f"resumed outcome stability count mismatch for {outcome_key}"
                    )
                candidate_outcomes.append(resumed_outcome)
                continue
            if candidate["candidate_valid"]:
                evaluation = evaluate_candidate_stably(
                    evaluator,
                    code=candidate["code"],
                    tests=plan["row"]["acceptance_tests"],
                    task_id=task_id,
                    sample_index=candidate["sample_index"],
                    stability_runs=args.eval_stability_runs,
                    timeout=args.eval_timeout_seconds,
                )
                evaluation_performed = True
            else:
                evaluation = {
                    "compiled": False,
                    "passed": False,
                    "completion_attestation_id": REQUIRED_ATTESTATION_ID,
                    "completion_attestation_enforced": False,
                    "completion_attestation_satisfied_all_runs": False,
                    "stability_runs": [],
                }
                evaluation_performed = False
            outcome = {
                "schema": RUN_SCHEMA_VERSION,
                "record_type": "candidate_outcome",
                "config_sha256": config_sha,
                "task_id": task_id,
                "sample_index": candidate["sample_index"],
                "attempt_id": candidate["attempt_id"],
                "response_id": candidate["response_id"],
                "finish_reason": candidate["finish_reason"],
                "candidate_valid": candidate["candidate_valid"],
                "terminal_reason": candidate["terminal_reason"],
                "code_sha256": candidate["code_sha256"],
                "evaluator_sha256": evaluator_record["sha256"],
                "evaluator_entrypoint": evaluator_record["entrypoint"],
                "evaluation_performed": evaluation_performed,
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
            "terminal_responses": len(terminal_slots),
            "evaluable_candidates": sum(
                bool(value["candidate_valid"]) for value in terminal_slots
            ),
            "invalid_candidates": sum(
                not bool(value["candidate_valid"]) for value in terminal_slots
            ),
            "length_slots": sum(
                value["finish_reason"] == "length" for value in terminal_slots
            ),
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
                    f"terminal_tasks={len(task_results)} failures={len(failures)} "
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
    if any(result["terminal_responses"] != args.k for result in task_results):
        raise RunFailure(
            "one or more tasks did not receive exactly K terminal provider responses"
        )

    task_order = {plan["task_id"]: index for index, plan in enumerate(plans)}
    task_results.sort(key=lambda result: task_order[result["task_id"]])
    passed = sum(result["passed"] for result in task_results)
    compiled = sum(result["compiled"] for result in task_results)
    resolved_models: set[str] = set()
    attempt_rows = load_jsonl(attempts_path, "completed attempt journal")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    recorded_budget_charge = 0
    terminal_response_count = 0
    evaluable_candidate_count = 0
    invalid_candidate_count = 0
    transport_retry_count = 0
    length_slot_count = 0
    terminal_reasons: dict[str, int] = {}
    for row in attempt_rows:
        charge = row.get("budget_charge_tokens")
        if isinstance(charge, bool) or not isinstance(charge, int) or charge < 0:
            raise RunFailure("attempt journal has an invalid budget charge")
        recorded_budget_charge += charge
        if row.get("response_received") is True:
            terminal_response_count += 1
            resolved_models.add(str(row.get("resolved_model") or ""))
            if row.get("candidate_valid") is True:
                evaluable_candidate_count += 1
            else:
                invalid_candidate_count += 1
            if row.get("finish_reason") == "length":
                length_slot_count += 1
            reason = str(row.get("terminal_reason") or "missing")
            terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
        else:
            transport_retry_count += 1
        row_usage = row.get("usage")
        if isinstance(row_usage, Mapping):
            for key_name in usage:
                value = row_usage.get(key_name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key_name] += int(value)
    audit_budget = TokenBudget(0)
    completed_terminal, _ = load_resume_attempts(
        attempts_path,
        config_sha=config_sha,
        prompt_map=prompt_map,
        budget=audit_budget,
        requested_model=args.model,
        k=args.k,
        max_prompt_tokens=args.max_prompt_tokens,
        requested_max_tokens=args.max_output_tokens,
        max_transport_attempts_per_slot=args.max_attempts_per_sample,
        slot_policy_sha256=slot_policy_sha256,
    )
    expected_slot_keys = {
        (plan["task_id"], sample_index)
        for plan in plans
        for sample_index in range(args.k)
    }
    if set(completed_terminal) != expected_slot_keys:
        raise RunFailure(
            "attempt journal does not contain exactly one terminal provider "
            "response for every task/sample slot"
        )
    if terminal_response_count != len(plans) * args.k:
        raise RunFailure("terminal-response aggregate disagrees with exact K slots")
    completed_outcomes = load_resume_outcomes(
        outcomes_path,
        config_sha=config_sha,
        evaluator_sha256=evaluator_record["sha256"],
    )
    expected_outcome_keys = {
        (task_id, sample_index, str(row["attempt_id"]))
        for (task_id, sample_index), row in completed_terminal.items()
    }
    if set(completed_outcomes) != expected_outcome_keys:
        raise RunFailure(
            "outcome journal does not contain exactly one outcome for every "
            "terminal provider response"
        )
    if "" in resolved_models:
        raise RunFailure("one or more terminal responses lacks a resolved model")
    if resolved_models != {args.model}:
        raise RunFailure(
            f"terminal responses have wrong model identities: "
            f"{sorted(resolved_models)}"
        )
    if recorded_budget_charge != budget.snapshot()["spent"]:
        raise RunFailure(
            "attempt-journal budget charges disagree with the in-memory ledger: "
            f"{recorded_budget_charge} != {budget.snapshot()['spent']}"
        )
    summary = {
        "schema": RUN_SCHEMA_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "run_id": out.name,
        "dataset_label": args.dataset_label,
        "dataset_sha256": provenance["dataset"]["sha256"],
        "task_set_sha256": provenance["task_set_sha256"],
        "arm": args.arm,
        "arm_interpretation": (
            "sealed pre-materialized lossless-F2 representation"
            if args.input_mode == "prematerialized_f2"
            else (
                "primary exact decoded student representation"
                if args.arm == "compact"
                else "raw-disassembly control; not a compression-only comparison"
            )
        ),
        "input_mode": args.input_mode,
        "pair_arm_key": (
            args.pair_arm_key
            if args.input_mode == "prematerialized_f2"
            else None
        ),
        "pair_manifest_sha256": (
            provenance["artifacts"]["pair_manifest"]["sha256"]
            if args.input_mode == "prematerialized_f2"
            else None
        ),
        "acceptance_test_sequence_sha256": provenance.get(
            "acceptance_test_sequence_sha256"
        ),
        "provider": args.provider,
        "requested_model": args.model,
        "resolved_models": sorted(resolved_models),
        "fixed_max_output_tokens": args.max_output_tokens,
        "slot_policy": provenance["config"]["slot_policy"],
        "slot_policy_sha256": slot_policy_sha256,
        "k": args.k,
        "tasks": len(task_results),
        "terminal_responses": terminal_response_count,
        "evaluable_candidates": evaluable_candidate_count,
        "invalid_candidates": invalid_candidate_count,
        "transport_retries": transport_retry_count,
        "length_slots": length_slot_count,
        "model_invalid_responses": invalid_candidate_count,
        "discarded_terminal_responses": 0,
        "terminal_reasons": terminal_reasons,
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
        "completion_attestation_enforced_for_every_evaluable_candidate": True,
        "terminal_invalid_candidates_not_evaluated": True,
        "all_tasks_have_exactly_k_terminal_provider_responses": True,
        "every_terminal_provider_response_has_exactly_one_outcome": True,
        "returned_responses_resampled": False,
        "transport_failures_only_retried": True,
        "early_stopping_used": False,
        "prompt_truncation_used": False,
        "prompt_token_accounting": {
            "preflight_count_basis": (
                "sealed_qwen_tokenizer_estimate"
                if args.input_mode == "prematerialized_f2"
                else "sealed_input_tokenizer_estimate"
            ),
            "provider_usage_prompt_tokens_authoritative": True,
            "provider_prompt_token_cap": args.max_prompt_tokens,
            "every_terminal_response_usage_checked_against_cap": True,
        },
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
            "schema": RUN_SCHEMA_VERSION,
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
    try:
        enforce_output_state_policy(args, out)
    except Exception as exc:
        print(
            f"FRONTIER_FAILED_CLOSED error={type(exc).__name__}: {exc} "
            f"out={out}",
            file=sys.stderr,
            flush=True,
        )
        return 2
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
                f"tasks={len(plans)} "
                + (
                    f"max_sealed_qwen_estimate_tokens={max_estimate} "
                    if args.input_mode == "prematerialized_f2"
                    else f"max_prompt_tokens={max_estimate} "
                )
                + f"out={out}",
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
                "schema": RUN_SCHEMA_VERSION,
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
