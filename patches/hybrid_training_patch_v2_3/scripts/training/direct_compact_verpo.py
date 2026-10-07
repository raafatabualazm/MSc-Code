#!/usr/bin/env python3
"""On-policy VeRPO for the encoder-free direct-compact Qwen student.

This trainer deliberately has no graph encoder, soft prefix, or legacy text
checkpoint path.  A policy checkpoint is the inseparable tuple

    (Qwen LoRA adapter, compact-token input overlay, sealed compact contract).

Every rollout group is sampled from the current tuple, scored by the hardened
Dart completion-attested verifier, optionally graded by a compile-gated
DeepSeek teacher, and consumed by exactly one optimizer update.  Recovery
checkpoints are immutable and self-contained, but are published only at a
predeclared interval (and at the final update) so optimizer state cannot exhaust
pod storage.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from models.direct_compact_causal import (
    CONTRACT_SCHEMA_V3,
    DirectCompactBatchCollator,
    DirectCompactContract,
    restore_source_embedding_overlay,
    resolve_decoder_config_path,
    sha256_artifact,
    sha256_file,
    validate_base_model_vocab,
    validate_join_seal,
)
from scripts.preprocessing.build_multifunction_executable_view import (
    REPRESENTATION_SCHEMA,
)
from scripts.preprocessing.build_verpo_feedback_view import (
    PRODUCTION_ELIGIBLE_TASK_IDS_SHA256,
    PRODUCTION_EXCLUDED_TASK_IDS_SHA256,
    PRODUCTION_EXPECTED_ACCOUNTING,
    ROLLOUT_SCOPE,
    extract_expect_spans,
    harness_with_cases,
    validate_feedback_training_boundary,
)
from scripts.evaluation.graph_compile_at_k_antigravity import (
    COMPLETION_ATTESTATION_ID,
    evaluate_dart_jit_tests_detail,
    validate_dart_binary,
)
from scripts.training.direct_compact_qwen_decompiler import (
    _encode,
    copy_exact_contract,
    direct_prompt,
    target_source,
    validate_warmstart_checkpoint,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (
    load_f2_prompt_contract,
    load_verified_prompt_rows,
    read_jsonl,
    sha256_text,
)


RUN_SCHEMA = "direct-compact-verpo-run-v1"
CHECKPOINT_SCHEMA = "direct-compact-verpo-checkpoint-v1"
JOURNAL_SCHEMA = "direct-compact-verpo-rollout-journal-v1"
STEP_JOURNAL_SCHEMA = "direct-compact-verpo-step-journal-v1"
JOURNAL_ATTESTATION_SCHEMA = "direct-compact-verpo-journal-chain-v1"
TASK_SCHEDULE_SCHEMA = "direct-compact-verpo-task-cycle-v1"
DEEPSEEK_RECEIPT_SCHEMA = "verpo-deepseek-response-receipt-v1"
DEEPSEEK_RECEIPT_ATTESTATION_SCHEMA = (
    "verpo-deepseek-response-receipt-attestation-v1"
)
DEEPSEEK_RESPONSE_IDS_SCHEMA = "verpo-deepseek-response-id-set-v1"
RECEIPT_CHAIN_GENESIS_SHA256 = "0" * 64
ARCHITECTURE = "qwen-causal-compact-tokens-no-encoder"
F2_REPRESENTATION_SCHEMA = "lossless-semantic-f2"
_CHECKPOINT_RE = re.compile(r"^checkpoint-optstep-(\d{6})$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def deterministic_task_schedule(
    task_ids: Sequence[str],
    *,
    seed: int,
    rollout_groups: int,
) -> list[int]:
    """Predeclare shuffled-without-replacement task indices across cycles."""

    normalized = [str(task_id) for task_id in task_ids]
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(not task_id for task_id in normalized)
    ):
        raise ValueError("VeRPO task schedule requires unique nonempty task IDs")
    if rollout_groups <= 0:
        raise ValueError("VeRPO rollout group count must be positive")
    result: list[int] = []
    epoch = 0
    while len(result) < rollout_groups:
        order = sorted(
            range(len(normalized)),
            key=lambda index: canonical_sha256(
                {
                    "schema": TASK_SCHEDULE_SCHEMA,
                    "seed": int(seed),
                    "epoch": epoch,
                    "task_id": normalized[index],
                }
            ),
        )
        result.extend(order[: rollout_groups - len(result)])
        epoch += 1
    return result


def task_sampling_contract(
    task_ids: Sequence[str],
    *,
    seed: int,
    max_updates: int,
    rollout_batch_size: int,
) -> dict[str, Any]:
    rollout_groups = int(max_updates) * int(rollout_batch_size)
    schedule = deterministic_task_schedule(
        task_ids, seed=seed, rollout_groups=rollout_groups
    )
    planned_ids = [str(task_ids[index]) for index in schedule]
    unique = len(set(planned_ids))
    return {
        "schema": TASK_SCHEDULE_SCHEMA,
        "policy": (
            "stable_sha256_epoch_permutation_without_replacement_then_cycle"
        ),
        "seed": int(seed),
        "dataset_rows": len(task_ids),
        "dataset_task_ids_sha256": canonical_sha256(list(task_ids)),
        "planned_rollout_groups": rollout_groups,
        "planned_schedule_task_ids_sha256": canonical_sha256(planned_ids),
        "planned_unique_tasks": unique,
        "planned_unique_fraction": unique / len(task_ids),
        "complete_dataset_cycles": rollout_groups // len(task_ids),
        "partial_cycle_groups": rollout_groups % len(task_ids),
        "with_replacement_within_cycle": False,
        "heldout_in_schedule": False,
    }


@dataclass(frozen=True)
class TeacherVisibleSource:
    """One hash-bound, verifier-free F2 source exposed to API teachers."""

    task_id: str
    text: str
    text_sha256: str
    source_record_sha256: str
    system_prompt: str
    system_prompt_sha256: str


def load_teacher_visible_sources(
    path: str | Path,
    *,
    expected_sha256: str,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    student_tokenizer_sha256: str,
) -> tuple[dict[str, TeacherVisibleSource], dict[str, Any]]:
    """Load the exact Qwen/GPT F2 artifact for DeepSeek task joins."""

    source_path = Path(path).expanduser().resolve()
    prompts, record = load_verified_prompt_rows(
        source_path,
        expected_sha256=expected_sha256,
    )
    raw_rows = read_jsonl(source_path)
    if len(raw_rows) != len(prompts):
        raise AssertionError("verified F2 prompt loader changed row count")
    system_prompt, manifest_record, prompt_manifest = load_f2_prompt_contract(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        prompt_record=record,
        expected_rows=len(prompts),
        student_tokenizer_sha256=student_tokenizer_sha256,
    )
    prompt_contract = prompt_manifest["f2_prompt_contract"]
    expected_system_sha = str(prompt_contract["system_prompt_sha256"])
    sources: dict[str, TeacherVisibleSource] = {}
    for prompt, raw in zip(prompts, raw_rows, strict=True):
        if raw.get("representation_schema") != F2_REPRESENTATION_SCHEMA:
            raise ValueError(
                f"{prompt.task_id}: teacher source is not the lossless F2 schema"
            )
        if raw.get("system_prompt_sha256") != expected_system_sha:
            raise ValueError(
                f"{prompt.task_id}: F2 source grammar differs from the API "
                "teacher grammar"
            )
        if not prompt.text.startswith("F2\n"):
            raise ValueError(f"{prompt.task_id}: F2 source has no F2 header")
        sources[prompt.task_id] = TeacherVisibleSource(
            task_id=prompt.task_id,
            text=prompt.text,
            text_sha256=prompt.text_sha256,
            source_record_sha256=prompt.source_record_sha256,
            system_prompt=system_prompt,
            system_prompt_sha256=expected_system_sha,
        )
    if not sources:
        raise ValueError("teacher F2 prompt artifact is empty")
    return sources, {
        "artifact": record,
        "manifest": manifest_record,
        "rows": len(sources),
        "task_set_sha256": canonical_sha256(sorted(sources)),
        "representation_schema": F2_REPRESENTATION_SCHEMA,
        "system_prompt_sha256": expected_system_sha,
        "f2_prompt_contract": dict(prompt_contract),
        "hidden_tests_exposed": False,
        "gold_dart_exposed": False,
    }


def _read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def write_json_new(path: str | Path, value: Any) -> None:
    """Create a JSON artifact without ever replacing an existing path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_new(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Create and fsync an immutable JSONL artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def create_journal_attempt(
    output_dir: Path,
    *,
    start_step: int,
    run_contract_sha256: str,
    parent_chain_sha256: str,
    parent_receipt_chain_sha256: str = RECEIPT_CHAIN_GENESIS_SHA256,
    receipt_index_offset: int = 0,
) -> Path:
    root = output_dir / "rollout_journal_attempts"
    root.mkdir(exist_ok=True)
    indices: list[int] = []
    for child in root.iterdir():
        match = re.fullmatch(r"attempt-(\d{4})-from-(\d{6})", child.name)
        if match and child.is_dir():
            indices.append(int(match.group(1)))
    index = max(indices, default=0) + 1
    attempt = root / f"attempt-{index:04d}-from-{start_step:06d}"
    attempt.mkdir()
    write_json_new(
        attempt / "attempt_manifest.json",
        {
            "schema": "direct-compact-verpo-journal-attempt-v1",
            "attempt_index": index,
            "resume_from_optimizer_step": start_step,
            "run_contract_sha256": run_contract_sha256,
            "parent_committed_chain_sha256": parent_chain_sha256,
            "parent_deepseek_receipt_chain_sha256": (
                parent_receipt_chain_sha256
            ),
            "deepseek_receipt_index_offset": int(receipt_index_offset),
            "deepseek_receipt_journal": "deepseek_response_receipts.jsonl",
            "append_only": True,
            "orphan_steps_after_a_crash_are_not_committed_by_a_checkpoint": True,
        },
    )
    return attempt


def validate_deepseek_receipt_attestation(
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one optimizer step's exact provider-response hash-chain slice."""

    if attestation.get("schema") != DEEPSEEK_RECEIPT_ATTESTATION_SCHEMA:
        raise ValueError("step journal has unknown DeepSeek receipt schema")
    before = int(attestation.get("receipt_count_before_step", -1))
    count = int(attestation.get("receipt_count_this_step", -1))
    cumulative = int(attestation.get("cumulative_receipt_count", -1))
    if before < 0 or count < 0 or cumulative != before + count:
        raise ValueError("DeepSeek receipt counts are inconsistent")
    previous = str(
        attestation.get("previous_receipt_chain_sha256") or ""
    )
    expected_head = str(
        attestation.get("cumulative_receipt_chain_sha256") or ""
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", previous)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_head)
    ):
        raise ValueError("DeepSeek receipt chain contains an invalid digest")
    receipts = attestation.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != count:
        raise ValueError("DeepSeek receipt list/count mismatch")
    if (
        attestation.get("plaintext_prompts_persisted") is not False
        or attestation.get("plaintext_reasoning_persisted") is not False
    ):
        raise ValueError("DeepSeek receipt attestation violates privacy contract")
    cursor = previous
    response_ids: list[str] = []
    for offset, receipt in enumerate(receipts, start=1):
        if not isinstance(receipt, Mapping):
            raise ValueError("DeepSeek receipt is not an object")
        expected_index = before + offset
        if (
            receipt.get("schema") != DEEPSEEK_RECEIPT_SCHEMA
            or int(receipt.get("receipt_index", -1)) != expected_index
            or receipt.get("previous_receipt_sha256") != cursor
            or receipt.get("plaintext_prompt_persisted") is not False
            or receipt.get("plaintext_reasoning_persisted") is not False
        ):
            raise ValueError("DeepSeek receipt identity/chain is inconsistent")
        receipt_base = dict(receipt)
        observed_hash = str(receipt_base.pop("receipt_sha256", ""))
        if observed_hash != canonical_sha256(receipt_base):
            raise ValueError("DeepSeek receipt payload hash mismatch")
        validation = receipt.get("validation") or {}
        request = receipt.get("request") or {}
        response = receipt.get("response") or {}
        exact_model_required = validation.get(
            "exact_requested_model_required", True
        )
        fingerprint_required = validation.get(
            "system_fingerprint_required", True
        )
        if (
            validation.get("accepted") is not True
            or validation.get("rejection_reasons") != []
            or not isinstance(response.get("id"), str)
            or not response["id"]
            or not isinstance(response.get("model"), str)
            or not response["model"]
            or (
                exact_model_required is True
                and response.get("model") != request.get("model")
            )
            or request.get("reasoning_effort") not in {"high", "max"}
            or (
                fingerprint_required is True
                and (
                    not isinstance(
                        response.get("system_fingerprint"), str
                    )
                    or not response["system_fingerprint"]
                )
            )
            or int(response.get("prompt_tokens") or 0) <= 0
            or int(response.get("completion_tokens") or 0) <= 0
            or int(response.get("total_tokens") or 0)
            != int(response.get("prompt_tokens") or 0)
            + int(response.get("completion_tokens") or 0)
        ):
            raise ValueError("DeepSeek receipt contains an unaccepted response")
        response_ids.append(str(response["id"]))
        cursor = observed_hash
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("DeepSeek response ID repeats within a step")
    if count:
        if (
            int(attestation.get("first_receipt_index", -1)) != before + 1
            or int(attestation.get("last_receipt_index", -1)) != cumulative
        ):
            raise ValueError("DeepSeek receipt index range is inconsistent")
    elif (
        attestation.get("first_receipt_index") is not None
        or attestation.get("last_receipt_index") is not None
    ):
        raise ValueError("empty DeepSeek receipt slice declares an index range")
    if cursor != expected_head:
        raise ValueError("DeepSeek receipt chain head mismatch")
    return {
        "receipt_count_before_step": before,
        "receipt_count_this_step": count,
        "cumulative_receipt_count": cumulative,
        "cumulative_receipt_chain_sha256": cursor,
        "response_id_sha256s_this_step": sorted(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in response_ids
        ),
    }


def validate_cumulative_deepseek_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_head_sha256: str,
    expected_response_id_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Revalidate the complete checkpoint-contained receipt history."""

    rows = [dict(value) for value in receipts]
    attestation = {
        "schema": DEEPSEEK_RECEIPT_ATTESTATION_SCHEMA,
        "receipt_count_before_step": 0,
        "receipt_count_this_step": len(rows),
        "cumulative_receipt_count": len(rows),
        "first_receipt_index": 1 if rows else None,
        "last_receipt_index": len(rows) if rows else None,
        "previous_receipt_chain_sha256": RECEIPT_CHAIN_GENESIS_SHA256,
        "cumulative_receipt_chain_sha256": (
            rows[-1].get("receipt_sha256")
            if rows
            else RECEIPT_CHAIN_GENESIS_SHA256
        ),
        "receipts": rows,
        "plaintext_prompts_persisted": False,
        "plaintext_reasoning_persisted": False,
    }
    summary = validate_deepseek_receipt_attestation(attestation)
    expected_ids = sorted(str(value) for value in expected_response_id_sha256s)
    observed_ids = summary["response_id_sha256s_this_step"]
    if (
        summary["cumulative_receipt_count"] != int(expected_count)
        or summary["cumulative_receipt_chain_sha256"]
        != expected_head_sha256
        or observed_ids != expected_ids
    ):
        raise ValueError(
            "cumulative DeepSeek receipts differ from telemetry/response IDs"
        )
    return summary


def write_step_journal(
    attempt_dir: Path,
    *,
    journal: Mapping[str, Any],
    previous_chain_sha256: str,
) -> dict[str, Any]:
    step = int(journal.get("optimizer_step", -1))
    if step <= 0:
        raise ValueError("step journal has no positive optimizer_step")
    receipt_attestation = journal.get("deepseek_response_receipts")
    if receipt_attestation is not None:
        validate_deepseek_receipt_attestation(receipt_attestation)
    entry_base = {
        "schema": STEP_JOURNAL_SCHEMA,
        "optimizer_step": step,
        "previous_chain_sha256": previous_chain_sha256,
        "journal": dict(journal),
    }
    entry_payload_sha256 = canonical_sha256(entry_base)
    cumulative_chain_sha256 = canonical_sha256(
        {
            "schema": JOURNAL_ATTESTATION_SCHEMA,
            "previous_chain_sha256": previous_chain_sha256,
            "entry_payload_sha256": entry_payload_sha256,
        }
    )
    entry = {
        **entry_base,
        "entry_payload_sha256": entry_payload_sha256,
        "cumulative_chain_sha256": cumulative_chain_sha256,
    }
    path = attempt_dir / f"step-{step:06d}.json"
    write_json_new(path, entry)
    return {
        "schema": JOURNAL_ATTESTATION_SCHEMA,
        "optimizer_step": step,
        "attempt": str(attempt_dir),
        "latest_step_journal": {
            "path": str(path),
            "sha256": sha256_file(path),
        },
        "previous_chain_sha256": previous_chain_sha256,
        "cumulative_chain_sha256": cumulative_chain_sha256,
    }


def publish_completed_run(
    *,
    output_dir: Path,
    max_updates: int,
    checkpoint_interval: int,
    latest_checkpoint: Path,
    run_contract_sha256: str,
    judge_telemetry: Mapping[str, Any],
    published_checkpoints_this_process: int,
    finalize_only_recovery: bool,
) -> None:
    """Publish the immutable completion marker, including finalize-only recovery."""

    checkpoint_provenance = _read_json_object(
        latest_checkpoint / "checkpoint_provenance.json"
    )
    if (
        checkpoint_provenance.get("schema") != CHECKPOINT_SCHEMA
        or int(checkpoint_provenance.get("optimizer_step", -1)) != max_updates
        or checkpoint_provenance.get("run_contract_sha256")
        != run_contract_sha256
    ):
        raise ValueError(
            "final checkpoint is not the exact declared terminal optimizer step"
        )
    checkpoint_judge_telemetry_path = latest_checkpoint / "judge_telemetry.json"
    checkpoint_response_ids_path = latest_checkpoint / "judge_response_ids.json"
    checkpoint_response_receipts_path = (
        latest_checkpoint / "judge_response_receipts.jsonl"
    )
    if (
        not checkpoint_judge_telemetry_path.is_file()
        or not checkpoint_response_ids_path.is_file()
        or not checkpoint_response_receipts_path.is_file()
        or sha256_file(checkpoint_judge_telemetry_path)
        != checkpoint_provenance.get("judge_telemetry_sha256")
        or sha256_file(checkpoint_response_ids_path)
        != checkpoint_provenance.get("judge_response_ids_sha256")
        or sha256_file(checkpoint_response_receipts_path)
        != checkpoint_provenance.get("judge_response_receipts_sha256")
        or _read_json_object(checkpoint_judge_telemetry_path)
        != dict(judge_telemetry)
    ):
        raise ValueError(
            "final checkpoint DeepSeek telemetry/response-ID binding differs"
        )
    write_json_new(
        output_dir / "completed.json",
        {
            "schema": "direct-compact-verpo-completed-v1",
            "optimizer_steps": max_updates,
            "checkpoint_interval": checkpoint_interval,
            "published_checkpoints_this_process": (
                published_checkpoints_this_process
            ),
            "latest_checkpoint": str(latest_checkpoint),
            "latest_checkpoint_provenance_sha256": sha256_file(
                latest_checkpoint / "checkpoint_provenance.json"
            ),
            "run_contract_sha256": run_contract_sha256,
            "rollout_journal_chain_sha256": checkpoint_provenance.get(
                "rollout_journal_chain_sha256"
            ),
            "latest_step_journal_sha256": checkpoint_provenance.get(
                "latest_step_journal_sha256"
            ),
            "deepseek_receipt_count": int(
                judge_telemetry.get("receipt_count", -1)
            ),
            "deepseek_receipt_chain_sha256": judge_telemetry.get(
                "receipt_chain_sha256"
            ),
            "judge_response_ids_sha256": checkpoint_provenance.get(
                "judge_response_ids_sha256"
            ),
            "judge_response_receipts_sha256": checkpoint_provenance.get(
                "judge_response_receipts_sha256"
            ),
            "judge_telemetry": dict(judge_telemetry),
            "judge_telemetry_cumulative": dict(judge_telemetry),
            "finalize_only_recovery": finalize_only_recovery,
        },
    )


_JUDGE_COUNTERS = frozenset(
    {
        "score_requested",
        "critique_requested",
        "group_calls_attempted",
        "group_calls_succeeded",
        "group_calls_skipped_budget",
        "skipped_ineligible",
        "api_calls",
        "api_successes",
        "api_failures",
        "parse_failures",
        "completion_retries",
        "empty_responses",
        "length_responses",
        "reasoning_responses",
        "cache_hits",
    }
)


def cumulative_judge_telemetry(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine resume segments without corrupting identity/configuration fields."""
    result = dict(current)
    for field in _JUDGE_COUNTERS:
        result[field] = int(prior.get(field, 0)) + int(current.get(field, 0))
    result["cache_entries_current_process"] = int(current.get("cache_entries", 0))
    result["segments"] = int(prior.get("segments", 0)) + 1
    if not current.get("last_error") and prior.get("last_error"):
        result["last_error"] = prior["last_error"]
    return result


class DisabledVerpoJudge:
    """Receipt-compatible no-op used when live teacher calls are disabled."""

    def __init__(
        self,
        *,
        receipt_chain_seed: str,
        receipt_index_offset: int,
        prior_response_id_sha256s: Sequence[str],
        receipt_journal_path: str | Path,
        mode: str,
    ) -> None:
        self._receipt_chain_seed = str(receipt_chain_seed)
        self._receipt_index_offset = int(receipt_index_offset)
        self._response_ids = sorted(
            str(value) for value in prior_response_id_sha256s
        )
        self._mode = str(mode)
        journal = Path(receipt_journal_path).expanduser().resolve()
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    def telemetry(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "prompt_schema_version": None,
            "model": None,
            "base_url": None,
            "api_style": None,
            "mode": self._mode,
            "fail_closed": False,
            "score_requested": 0,
            "critique_requested": 0,
            "group_calls_attempted": 0,
            "group_calls_succeeded": 0,
            "group_calls_skipped_budget": 0,
            "skipped_ineligible": 0,
            "api_calls": 0,
            "api_successes": 0,
            "api_failures": 0,
            "parse_failures": 0,
            "completion_retries": 0,
            "empty_responses": 0,
            "length_responses": 0,
            "reasoning_responses": 0,
            "cache_hits": 0,
            "cache_entries": 0,
            "response_receipts_current_process": 0,
            "receipt_count": self._receipt_index_offset,
            "receipt_chain_sha256": self._receipt_chain_seed,
            "unique_response_ids": len(self._response_ids),
            "last_error": None,
        }

    def response_id_sha256s(self) -> list[str]:
        return list(self._response_ids)

    def receipt_attestation_since(
        self, receipt_count: int
    ) -> dict[str, Any]:
        if int(receipt_count) != self._receipt_index_offset:
            raise ValueError("disabled judge receipt cursor is inconsistent")
        return {
            "schema": DEEPSEEK_RECEIPT_ATTESTATION_SCHEMA,
            "receipt_count_before_step": self._receipt_index_offset,
            "receipt_count_this_step": 0,
            "cumulative_receipt_count": self._receipt_index_offset,
            "first_receipt_index": None,
            "last_receipt_index": None,
            "previous_receipt_chain_sha256": self._receipt_chain_seed,
            "cumulative_receipt_chain_sha256": self._receipt_chain_seed,
            "receipts": [],
            "plaintext_prompts_persisted": False,
            "plaintext_reasoning_persisted": False,
        }

    def assert_healthy(self, *, require_success: bool = False) -> None:
        if require_success:
            raise RuntimeError("disabled judge cannot provide live scores")


def verpo_local_rewards(
    group_details: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    density_norm: bool = True,
    epsilon: float = 1e-8,
) -> list[float]:
    """VeRPO Eq. 7-10 density-calibrated local partial-success rewards.

    The Gaussian KDE bandwidth is exactly ``std(rho) / 2``.  Missing test
    entries fail closed. Global full-suite outcomes are intentionally not
    folded into this value: VeRPO centers local and global rewards separately
    before fusing their advantages.
    """
    if not group_details:
        raise ValueError("VeRPO needs a non-empty rollout group")
    if alpha <= 0.0:
        raise ValueError("VeRPO alpha must be positive")
    if epsilon <= 0.0:
        raise ValueError("VeRPO epsilon must be positive")
    n_tests = max(
        (len(detail.get("test_passes") or []) for detail in group_details),
        default=0,
    )
    if n_tests <= 0:
        raise ValueError("VeRPO verifier returned no per-test evidence")

    matrix: list[list[bool]] = []
    for detail in group_details:
        passes = [bool(value) for value in (detail.get("test_passes") or [])]
        if len(passes) != n_tests:
            raise ValueError("VeRPO group has inconsistent per-test vectors")
        matrix.append(passes)

    group_size = len(matrix)
    rho = [
        sum(matrix[group][test] for group in range(group_size)) / group_size
        for test in range(n_tests)
    ]
    weights = [math.exp(-alpha * value) for value in rho]
    if density_norm:
        mean = sum(rho) / n_tests
        variance = sum((value - mean) ** 2 for value in rho) / n_tests
        sigma = math.sqrt(variance) / 2.0
        if sigma <= epsilon:
            densities = [float(n_tests)] * n_tests
        else:
            denominator = 2.0 * sigma * sigma
            densities = [
                sum(
                    math.exp(-((rho[left] - rho[right]) ** 2) / denominator)
                    for right in range(n_tests)
                )
                for left in range(n_tests)
            ]
        weights = [
            weight / (density + epsilon)
            for weight, density in zip(weights, densities)
        ]

    return [
        sum(
            weights[test]
            for test in range(n_tests)
            if matrix[group][test]
        )
        for group in range(group_size)
    ]


def should_query_group_teacher(
    group_details: Sequence[Mapping[str, Any]],
    *,
    group_ordinal: int,
    interval: int,
) -> bool:
    """Gate live teacher work to sparse, unresolved comparison groups."""

    if interval <= 0:
        raise ValueError("teacher interval must be positive")
    if group_ordinal <= 0:
        raise ValueError("group ordinal must be positive")
    if group_ordinal % interval:
        return False
    if any(bool(detail.get("full_pass")) for detail in group_details):
        return False
    compiling_failures = sum(
        bool(detail.get("compiled")) and not bool(detail.get("full_pass"))
        for detail in group_details
    )
    return compiling_failures >= 2


def select_group_teacher_candidates(
    group_details: Sequence[Mapping[str, Any]],
    local_rewards: Sequence[float],
    *,
    top_n: int = 2,
) -> list[int]:
    """Select the strongest compiling failures, stably by rollout index."""

    if len(group_details) != len(local_rewards):
        raise ValueError("teacher selector detail/reward lengths differ")
    if top_n <= 0:
        raise ValueError("teacher selector top_n must be positive")
    values = [float(value) for value in local_rewards]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("teacher selector rewards must be finite")
    eligible = [
        index
        for index, detail in enumerate(group_details)
        if bool(detail.get("compiled")) and not bool(detail.get("full_pass"))
    ]
    eligible.sort(key=lambda index: (-values[index], index))
    return eligible[:top_n]


def sparse_teacher_advantages(
    *,
    group_size: int,
    selected_indices: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[float], list[bool]]:
    """Center only observed teacher scores; missing entries remain exact zero."""

    if group_size <= 0:
        raise ValueError("teacher advantage group_size must be positive")
    indices = [int(index) for index in selected_indices]
    values = [float(value) for value in scores]
    if len(indices) != len(values):
        raise ValueError("selected teacher index/score lengths differ")
    if (
        len(set(indices)) != len(indices)
        or any(index < 0 or index >= group_size for index in indices)
    ):
        raise ValueError("selected teacher indices are invalid")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("teacher scores must be finite values in [0,1]")
    mask = [False] * group_size
    advantages = [0.0] * group_size
    if not values:
        return advantages, mask
    mean = sum(values) / len(values)
    for index, value in zip(indices, values):
        mask[index] = True
        advantages[index] = value - mean
    return advantages, mask


def compile_gated_teacher_signals(
    group_details: Sequence[Mapping[str, Any]],
    judge_scores: Sequence[float],
) -> list[float]:
    """Return the separately centered DeepSeek signal's raw group values.

    A verifier-confirmed full pass is assigned the maximum signal 1 without an
    API call; a non-compiling candidate receives 0; only compiling failures use
    DeepSeek's score. This term is an explicit extension to paper VeRPO and is
    never folded into either verifiable reward before centering.
    """
    if len(group_details) != len(judge_scores):
        raise ValueError("judge detail/score lengths differ")
    signals: list[float] = []
    for detail, raw_score in zip(group_details, judge_scores):
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("judge scores must be finite values in [0,1]")
        if bool(detail.get("full_pass")):
            signals.append(1.0)
        elif bool(detail.get("compiled")):
            signals.append(score)
        else:
            signals.append(0.0)
    return signals


def mean_centered_advantages(
    rewards: Sequence[float],
) -> list[float]:
    """Mean-center a group with VeRPO's constant F_norm=1."""
    if not rewards:
        raise ValueError("cannot center an empty reward group")
    values = [float(value) for value in rewards]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("rewards must be finite")
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def verpo_unified_advantages(
    group_details: Sequence[Mapping[str, Any]],
    local_rewards: Sequence[float],
    teacher_signals: Sequence[float],
    *,
    beta: float,
    teacher_weight: float,
    teacher_mask: Sequence[bool] | None = None,
) -> dict[str, list[float]]:
    """VeRPO Eq. 12-14 plus an explicit centered teacher add-on.

    ``A = A_global + beta * A_local + teacher_weight * A_teacher``.
    Each component is independently mean-centered and uses F_norm=1; no
    population-standard-deviation division is permitted.
    """
    if not (
        len(group_details) == len(local_rewards) == len(teacher_signals)
    ):
        raise ValueError(
            "global/local/teacher VeRPO group lengths differ"
        )
    if not group_details:
        raise ValueError("VeRPO needs a non-empty rollout group")
    if beta < 0.0 or teacher_weight < 0.0:
        raise ValueError(
            "VeRPO beta and teacher coefficient must be non-negative"
        )
    global_rewards = [
        float(bool(detail.get("full_pass"))) for detail in group_details
    ]
    global_advantages = mean_centered_advantages(global_rewards)
    local_advantages = mean_centered_advantages(local_rewards)
    if teacher_mask is None:
        teacher_advantages = mean_centered_advantages(teacher_signals)
        normalized_teacher_mask = [True] * len(teacher_signals)
    else:
        normalized_teacher_mask = [bool(value) for value in teacher_mask]
        if len(normalized_teacher_mask) != len(teacher_signals):
            raise ValueError("teacher signal/mask lengths differ")
        selected_indices = [
            index
            for index, present in enumerate(normalized_teacher_mask)
            if present
        ]
        selected_scores = [
            float(teacher_signals[index]) for index in selected_indices
        ]
        teacher_advantages, rebuilt_mask = sparse_teacher_advantages(
            group_size=len(teacher_signals),
            selected_indices=selected_indices,
            scores=selected_scores,
        )
        if rebuilt_mask != normalized_teacher_mask:
            raise AssertionError("teacher observation mask changed")
    unified = [
        global_advantage
        + beta * local_advantage
        + teacher_weight * teacher_advantage
        for global_advantage, local_advantage, teacher_advantage in zip(
            global_advantages,
            local_advantages,
            teacher_advantages,
        )
    ]
    return {
        "global_rewards": global_rewards,
        "local_rewards": [float(value) for value in local_rewards],
        "teacher_signals": [float(value) for value in teacher_signals],
        "teacher_mask": normalized_teacher_mask,
        "global_advantages": global_advantages,
        "local_advantages": local_advantages,
        "teacher_advantages": teacher_advantages,
        "unified_advantages": unified,
    }


def policy_token_loss(
    current_logprobs: torch.Tensor,
    saved_rollout_logprobs: torch.Tensor,
    advantage: float,
    *,
    ppo_clip: float,
) -> torch.Tensor:
    """Token policy loss, optionally PPO-clipped against saved rollout logits."""
    if current_logprobs.ndim != 1 or current_logprobs.numel() == 0:
        raise ValueError("current token logprobs must be a non-empty vector")
    if saved_rollout_logprobs.shape != current_logprobs.shape:
        raise ValueError("saved/current rollout logprob shapes differ")
    if ppo_clip < 0.0 or ppo_clip >= 1.0:
        raise ValueError("ppo_clip must be zero or lie in (0,1)")
    advantage_tensor = current_logprobs.new_tensor(float(advantage))
    if ppo_clip == 0.0:
        return -(advantage_tensor * current_logprobs).mean()
    ratio = torch.exp(
        (current_logprobs - saved_rollout_logprobs.detach()).clamp(-20.0, 20.0)
    )
    unclipped = ratio * advantage_tensor
    clipped = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip) * advantage_tensor
    return -torch.minimum(unclipped, clipped).mean()


def split_visible_expect_harnesses(test_code: str) -> list[str]:
    """Create one canonical harness per balanced visible ``expect`` case."""

    source = str(test_code)
    spans = extract_expect_spans(source)
    if not spans:
        raise ValueError("visible feedback tests contain no expect cases")
    return [
        harness_with_cases(source, spans, {selected})
        for selected in range(len(spans))
    ]


def score_dart_candidate(
    candidate: str,
    feedback_tests: str,
    task_id: str,
    *,
    timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    """Run full-suite and per-test completion-attested Dart rewards."""
    variants = split_visible_expect_harnesses(feedback_tests)
    compiled, full_pass, diagnostic, _ = evaluate_dart_jit_tests_detail(
        candidate,
        feedback_tests,
        f"{task_id}-full",
        timeout=timeout,
        stability_runs=stability_runs,
    )
    if not compiled:
        test_passes = [False] * len(variants)
    elif full_pass:
        test_passes = [True] * len(variants)
    else:
        test_passes = []
        for index, variant in enumerate(variants):
            _compiled, passed, _diagnostic, _ = evaluate_dart_jit_tests_detail(
                candidate,
                variant,
                f"{task_id}-test-{index}",
                timeout=timeout,
                stability_runs=stability_runs,
            )
            test_passes.append(bool(_compiled and passed))
    return {
        "compiled": bool(compiled),
        "full_pass": bool(compiled and full_pass),
        "test_passes": test_passes,
        "diagnostic": str(diagnostic or "")[:4000],
    }


def judge_payload_from_rollout(
    *,
    source: TeacherVisibleSource,
    feedback_tests: str,
    candidate: str,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    """Whitelist the only fields that may cross the teacher boundary.

    ``source`` is the independently serialized, verifier-free F2 assembly+CFG
    artifact used by the other API teachers. ``feedback_tests`` are the public
    training-time behavioral specification. Hidden acceptance tests and
    supervised/reference Dart are neither accepted by this function nor
    retained in the rollout record object.
    """
    return {
        "source": source.text,
        "source_sha256": source.text_sha256,
        "source_format_guide": source.system_prompt,
        "tests": str(feedback_tests),
        "candidate": str(candidate),
        "diagnostic": str(detail.get("diagnostic") or ""),
        "compiled": bool(detail.get("compiled")),
        "full_pass": bool(detail.get("full_pass")),
    }


def group_judge_payload_from_rollout(
    *,
    source: TeacherVisibleSource,
    feedback_tests: str,
    candidates: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    """Build one whitelist-only request for the selected comparison subset."""

    if len(candidates) != len(details):
        raise ValueError("group teacher candidate/detail lengths differ")
    indices = [int(index) for index in selected_indices]
    if (
        len(indices) < 2
        or len(set(indices)) != len(indices)
        or any(index < 0 or index >= len(candidates) for index in indices)
    ):
        raise ValueError(
            "group teacher needs at least two unique selected candidates"
        )
    selected: list[dict[str, Any]] = []
    for index in indices:
        candidate = candidates[index]
        detail = details[index]
        if not bool(detail.get("compiled")) or bool(detail.get("full_pass")):
            raise ValueError("group teacher selection contains ineligible rollout")
        selected.append(
            {
                "group_index": int(candidate.get("group_index", index)),
                "candidate": str(candidate.get("candidate") or ""),
                "diagnostic": str(detail.get("diagnostic") or ""),
                "compiled": True,
                "full_pass": False,
            }
        )
    return {
        "source": source.text,
        "source_sha256": source.text_sha256,
        "source_format_guide": source.system_prompt,
        "tests": str(feedback_tests),
        "candidates": selected,
    }


def enqueue_teacher_escalation(
    path: str | Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one fsynced, content-addressed escalation at most once."""

    queue_path = Path(path).expanduser().resolve()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    identity = {
        "schema": "direct-compact-verpo-teacher-escalation-key-v1",
        "task_id": str(row.get("task_id") or ""),
        "group_ordinal": int(row.get("group_ordinal", -1)),
        "policy_version": int(row.get("policy_version", -1)),
        "run_contract_sha256": str(row.get("run_contract_sha256") or ""),
        "payload": row.get("payload"),
    }
    if (
        not identity["task_id"]
        or identity["group_ordinal"] <= 0
        or identity["policy_version"] < 0
        or not re.fullmatch(
            r"[0-9a-f]{64}", identity["run_contract_sha256"]
        )
        or not isinstance(identity["payload"], Mapping)
    ):
        raise ValueError("offline teacher escalation identity is invalid")
    escalation_key = canonical_sha256(identity)
    sealed = {
        "schema": "direct-compact-verpo-teacher-escalation-v1",
        **row,
        "escalation_key": escalation_key,
    }
    if queue_path.exists():
        with queue_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{queue_path}:{line_number}: invalid escalation JSONL"
                    ) from exc
                if existing.get("escalation_key") == escalation_key:
                    return dict(existing)
    with queue_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                sealed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return sealed


@dataclass(frozen=True)
class CompactVerpoRecord:
    task_id: str
    prompt_ids: tuple[int, ...]
    compact_ids: tuple[int, ...]
    teacher_source: TeacherVisibleSource
    feedback_tests: str
    replay_target_ids: tuple[int, ...]


class CompactVerpoDataset:
    """Sealed compact rows reduced to a visible-only rollout boundary."""

    def __init__(
        self,
        path: str | Path,
        *,
        tokenizer: Any,
        contract: DirectCompactContract,
        enable_replay: bool,
        max_generation_tokens: int,
        teacher_sources: Mapping[str, TeacherVisibleSource],
    ) -> None:
        if (
            max_generation_tokens <= 0
            or max_generation_tokens > contract.max_target_tokens
        ):
            raise ValueError(
                "VeRPO generation budget exceeds the compact target contract"
            )
        self.path = Path(path)
        self.rows: list[CompactVerpoRecord] = []
        self.task_ids: list[str] = []
        self.pool_use_count = 0
        seen: set[str] = set()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        with self.path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    raise ValueError(f"{self.path}: blank sealed row at {index + 1}")
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{self.path}: row {index + 1} is not an object")
                identity = str(raw.get("task_id") or raw.get("id") or f"row-{index}")
                if not identity or identity in seen:
                    raise ValueError(
                        f"{self.path}: missing/duplicate task identity {identity!r}"
                    )
                seen.add(identity)
                teacher_source = teacher_sources.get(identity)
                if teacher_source is None:
                    raise ValueError(
                        f"{identity}: no exact F2 teacher source is available"
                    )
                compact_ids = contract.validate_row(raw, identity)
                pool = contract.validate_v3_pool_payload(
                    compact_ids, tokenizer, identity
                )
                if contract.schema == CONTRACT_SCHEMA_V3:
                    self.pool_use_count += len(pool["uses"])
                row_function = str(raw.get("function") or "").strip()
                if row_function and row_function != contract.target_function:
                    raise ValueError(
                        f"{identity}: target function differs from compact contract"
                    )
                row_language = str(
                    raw.get("language") or raw.get("lang") or ""
                ).strip()
                if (
                    row_language
                    and row_language.lower()
                    != contract.target_language.lower()
                ):
                    raise ValueError(
                        f"{identity}: target language differs from compact contract"
                    )
                feedback_tests = raw.get("feedback_tests")
                if not isinstance(feedback_tests, str) or not feedback_tests.strip():
                    raise ValueError(
                        f"{identity}: VeRPO requires the explicit feedback_tests field; "
                        "acceptance/hidden tests are never a fallback"
                    )
                # Validate the per-test contract before loading an expensive model.
                split_visible_expect_harnesses(feedback_tests)
                prompt_ids = _encode(
                    tokenizer,
                    direct_prompt(
                        raw,
                        target_function=contract.target_function,
                        target_language=contract.target_language,
                    ),
                    special=True,
                )
                replay_ids: list[int] = []
                if enable_replay:
                    replay_ids = _encode(
                        tokenizer,
                        target_source(raw, identity),
                        special=False,
                    )
                    if eos_id is not None and (
                        not replay_ids or replay_ids[-1] != eos_id
                    ):
                        replay_ids.append(int(eos_id))
                    if len(replay_ids) > contract.max_target_tokens:
                        raise ValueError(
                            f"{identity}: replay target exceeds compact contract"
                        )
                if len(compact_ids) > contract.max_source_tokens:
                    raise ValueError(f"{identity}: compact source exceeds contract")
                if (
                    len(prompt_ids) + len(compact_ids) + max(1, len(replay_ids))
                    > contract.max_total_tokens
                ):
                    raise ValueError(f"{identity}: row exceeds decoder context contract")
                if (
                    len(prompt_ids)
                    + len(compact_ids)
                    + max_generation_tokens
                    > contract.max_total_tokens
                ):
                    raise ValueError(
                        f"{identity}: rollout generation budget exceeds the "
                        "decoder context contract"
                    )
                self.rows.append(
                    CompactVerpoRecord(
                        task_id=identity,
                        prompt_ids=tuple(prompt_ids),
                        compact_ids=tuple(compact_ids),
                        teacher_source=teacher_source,
                        feedback_tests=feedback_tests,
                        replay_target_ids=tuple(replay_ids),
                    )
                )
                self.task_ids.append(identity)
        if not self.rows:
            raise ValueError(f"{self.path}: no VeRPO rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CompactVerpoRecord:
        return self.rows[index]


def validate_rollout_distribution(
    *, group_size: int, top_p: float, top_k: int, temperature: float
) -> None:
    if group_size < 2:
        raise ValueError("VeRPO group_size must be at least 2")
    if top_p != 1.0:
        raise ValueError("on-policy VeRPO requires top_p=1.0 (no nucleus truncation)")
    if top_k != 0:
        raise ValueError("on-policy VeRPO requires top_k=0 (no top-k truncation)")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("generation temperature must be finite and positive")


def _context_limit(config: Any) -> int | None:
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    allowed = ("lora_", "source_embeddings")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not any(token in name for token in allowed)
    ]
    if unexpected:
        raise ValueError(
            "direct-compact VeRPO found trainable base/non-contract parameters: "
            + ", ".join(unexpected[:12])
        )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("direct-compact VeRPO has no trainable adapter/overlay rows")
    return parameters


def completion_token_logprobs(
    model: torch.nn.Module,
    prefix_ids: Sequence[int],
    completion_ids: Sequence[int],
    *,
    temperature: float,
    device: torch.device,
    with_grad: bool,
) -> torch.Tensor:
    """Score completion tokens under the exact untruncated rollout policy."""
    if not prefix_ids or not completion_ids:
        raise ValueError("logprob scoring requires non-empty prefix and completion")
    input_ids = torch.tensor(
        [list(prefix_ids) + list(completion_ids)],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        start = len(prefix_ids) - 1
        end = start + len(completion_ids)
        logits = outputs.logits[0, start:end, :].float() / temperature
        targets = input_ids[0, len(prefix_ids) :]
        values = torch.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
    if values.numel() != len(completion_ids):
        raise RuntimeError("causal logprob alignment failed")
    return values


def _optimizer_to_parameter_device(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(parameter.device)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("invalid VeRPO RNG state")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_states = state.get("torch_cuda") or []
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG device count differs from checkpoint")
        torch.cuda.set_rng_state_all(cuda_states)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        match = _CHECKPOINT_RE.fullmatch(child.name)
        if match and child.is_dir():
            checkpoints.append((int(match.group(1)), child.resolve()))
    return max(checkpoints, default=(0, None))[1]


def validate_resume_checkpoint(
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    run_contract_sha256: str,
    compact_contract_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    root = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if root.parent != output:
        raise ValueError("resume checkpoint must be a direct child of output_dir")
    latest = _latest_checkpoint(output)
    if latest != root:
        raise ValueError("resume must use the latest immutable optimizer checkpoint")
    match = _CHECKPOINT_RE.fullmatch(root.name)
    if not match:
        raise ValueError("resume checkpoint name is not optimizer-step bound")
    required = {
        "checkpoint_provenance": root / "checkpoint_provenance.json",
        "optimizer": root / "optimizer.pt",
        "rng": root / "rng_state.pt",
        "trainer_state": root / "trainer_state.json",
        "journal": root / "rollout_journal.json",
        "judge_telemetry": root / "judge_telemetry.json",
        "judge_response_ids": root / "judge_response_ids.json",
        "judge_response_receipts": root / "judge_response_receipts.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError("resume checkpoint is incomplete: " + ", ".join(missing))
    warmstart = validate_warmstart_checkpoint(
        root, contract_path=compact_contract_path
    )
    provenance = _read_json_object(required["checkpoint_provenance"])
    if provenance.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("resume checkpoint has unknown provenance schema")
    if provenance.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("resume checkpoint belongs to a different VeRPO run contract")
    step = int(provenance.get("optimizer_step", -1))
    if step != int(match.group(1)):
        raise ValueError("resume checkpoint step/name mismatch")
    bindings = {
        "decoder_adapter_sha256": sha256_artifact(warmstart["adapter"]),
        "source_overlay_sha256": sha256_file(warmstart["overlay"]),
        "optimizer_sha256": sha256_file(required["optimizer"]),
        "rng_state_sha256": sha256_file(required["rng"]),
        "trainer_state_sha256": sha256_file(required["trainer_state"]),
        "rollout_journal_sha256": sha256_file(required["journal"]),
        "judge_telemetry_sha256": sha256_file(required["judge_telemetry"]),
        "judge_response_ids_sha256": sha256_file(
            required["judge_response_ids"]
        ),
        "judge_response_receipts_sha256": sha256_file(
            required["judge_response_receipts"]
        ),
    }
    mismatches = [
        key for key, value in bindings.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "resume artifacts differ from checkpoint provenance: "
            + ", ".join(mismatches)
        )
    journal_attestation = _read_json_object(required["journal"])
    judge_telemetry = _read_json_object(required["judge_telemetry"])
    response_ids_record = _read_json_object(required["judge_response_ids"])
    response_id_hashes = response_ids_record.get("response_id_sha256s")
    if (
        response_ids_record.get("schema") != DEEPSEEK_RESPONSE_IDS_SCHEMA
        or not isinstance(response_id_hashes, list)
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in response_id_hashes
        )
        or len(response_id_hashes) != len(set(response_id_hashes))
        or int(response_ids_record.get("count", -1)) != len(response_id_hashes)
        or int(judge_telemetry.get("receipt_count", -1))
        != len(response_id_hashes)
        or int(judge_telemetry.get("unique_response_ids", -1))
        != len(response_id_hashes)
    ):
        raise ValueError("resume DeepSeek response-ID set is incomplete")
    cumulative_receipts = read_jsonl(required["judge_response_receipts"])
    validate_cumulative_deepseek_receipts(
        cumulative_receipts,
        expected_count=int(judge_telemetry["receipt_count"]),
        expected_head_sha256=str(judge_telemetry["receipt_chain_sha256"]),
        expected_response_id_sha256s=response_id_hashes,
    )
    latest_step_record = journal_attestation.get("latest_step_journal") or {}
    latest_step_path = Path(
        str(latest_step_record.get("path") or "")
    ).expanduser().resolve()
    if (
        journal_attestation.get("schema") != JOURNAL_ATTESTATION_SCHEMA
        or int(journal_attestation.get("optimizer_step", -1)) != step
        or not latest_step_path.is_file()
        or sha256_file(latest_step_path) != latest_step_record.get("sha256")
        or provenance.get("latest_step_journal_sha256")
        != latest_step_record.get("sha256")
        or provenance.get("rollout_journal_chain_sha256")
        != journal_attestation.get("cumulative_chain_sha256")
    ):
        raise ValueError("resume rollout journal chain is incomplete or inconsistent")
    latest_step_entry = _read_json_object(latest_step_path)
    latest_receipts = (
        (latest_step_entry.get("journal") or {}).get(
            "deepseek_response_receipts"
        )
        or {}
    )
    receipt_summary = validate_deepseek_receipt_attestation(latest_receipts)
    if (
        receipt_summary["cumulative_receipt_count"]
        != int(judge_telemetry.get("receipt_count", -1))
        or receipt_summary["cumulative_receipt_chain_sha256"]
        != judge_telemetry.get("receipt_chain_sha256")
    ):
        raise ValueError(
            "resume DeepSeek receipt head differs from judge telemetry"
        )
    return provenance, {**warmstart, **required}


def save_optimizer_checkpoint(
    *,
    output_dir: Path,
    optimizer_step: int,
    model: torch.nn.Module,
    overlay: Any,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    contract_path: str | Path,
    run_contract_sha256: str,
    base_provenance: Mapping[str, Any],
    journal_attestation: Mapping[str, Any],
    judge_telemetry: Mapping[str, Any],
    response_id_sha256s: Sequence[str] = (),
    prior_response_receipts_path: str | Path | None = None,
    response_receipts_current_segment: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Atomically publish one immutable, exact-resume optimizer checkpoint."""
    name = f"checkpoint-optstep-{optimizer_step:06d}"
    destination = output_dir / name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite VeRPO checkpoint {name}")
    # Unique incomplete directories make a power loss during publication
    # recoverable without deleting or overwriting forensic state.
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.incomplete-", dir=output_dir)
    )
    try:
        adapter = temporary / "decoder_adapter"
        model.save_pretrained(adapter)
        overlay_path = temporary / "source_embedding_overlay.pt"
        torch.save(overlay.overlay_state(), overlay_path)
        tokenizer.save_pretrained(temporary / "tokenizer")
        copy_exact_contract(contract_path, temporary / "compact_contract.json")

        optimizer_path = temporary / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        rng_path = temporary / "rng_state.pt"
        torch.save(capture_rng_state(), rng_path)
        trainer_state = {
            "schema": "direct-compact-verpo-trainer-state-v1",
            "optimizer_step": optimizer_step,
            "policy_version": optimizer_step,
        }
        write_json_new(temporary / "trainer_state.json", trainer_state)
        if (
            journal_attestation.get("schema") != JOURNAL_ATTESTATION_SCHEMA
            or int(journal_attestation.get("optimizer_step", -1))
            != optimizer_step
        ):
            raise ValueError("checkpoint journal attestation step mismatch")
        latest_step_record = journal_attestation.get("latest_step_journal") or {}
        latest_step_path = Path(
            str(latest_step_record.get("path") or "")
        ).expanduser().resolve()
        if (
            not latest_step_path.is_file()
            or sha256_file(latest_step_path)
            != latest_step_record.get("sha256")
        ):
            raise ValueError("checkpoint latest step journal differs from attestation")
        write_json_new(
            temporary / "rollout_journal.json", journal_attestation
        )
        latest_step_entry = _read_json_object(latest_step_path)
        receipt_attestation = (
            (latest_step_entry.get("journal") or {}).get(
                "deepseek_response_receipts"
            )
        )
        normalized_response_ids = sorted(str(value) for value in response_id_sha256s)
        if receipt_attestation is not None:
            receipt_summary = validate_deepseek_receipt_attestation(
                receipt_attestation
            )
            if (
                receipt_summary["cumulative_receipt_count"]
                != int(judge_telemetry.get("receipt_count", -1))
                or receipt_summary["cumulative_receipt_chain_sha256"]
                != judge_telemetry.get("receipt_chain_sha256")
                or len(normalized_response_ids)
                != int(judge_telemetry.get("receipt_count", -1))
                or len(normalized_response_ids)
                != len(set(normalized_response_ids))
                or any(
                    not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in normalized_response_ids
                )
            ):
                raise ValueError(
                    "checkpoint DeepSeek receipts/telemetry/response IDs differ"
                )
        prior_receipts = (
            []
            if prior_response_receipts_path is None
            else read_jsonl(prior_response_receipts_path)
        )
        cumulative_receipts = prior_receipts + [
            dict(value) for value in response_receipts_current_segment
        ]
        validate_cumulative_deepseek_receipts(
            cumulative_receipts,
            expected_count=int(judge_telemetry.get("receipt_count", 0)),
            expected_head_sha256=str(
                judge_telemetry.get(
                    "receipt_chain_sha256",
                    RECEIPT_CHAIN_GENESIS_SHA256,
                )
            ),
            expected_response_id_sha256s=normalized_response_ids,
        )
        response_receipts_path = temporary / "judge_response_receipts.jsonl"
        write_jsonl_new(response_receipts_path, cumulative_receipts)
        response_ids_path = temporary / "judge_response_ids.json"
        write_json_new(
            response_ids_path,
            {
                "schema": DEEPSEEK_RESPONSE_IDS_SCHEMA,
                "count": len(normalized_response_ids),
                "response_id_sha256s": normalized_response_ids,
            },
        )
        write_json_new(temporary / "judge_telemetry.json", judge_telemetry)

        run_provenance = {
            **dict(base_provenance),
            "schema": "direct-compact-run-provenance-v1",
            "architecture": ARCHITECTURE,
            "stage": "on-policy-direct-compact-verpo",
            "optimizer_step": optimizer_step,
            "run_contract_sha256": run_contract_sha256,
            "contract_sha256": sha256_file(temporary / "compact_contract.json"),
            "source_overlay_sha256": sha256_file(overlay_path),
            "decoder_adapter_sha256": sha256_artifact(adapter),
            "judge_telemetry": dict(judge_telemetry),
            "rollout_journal_chain_sha256": journal_attestation[
                "cumulative_chain_sha256"
            ],
            "latest_step_journal_sha256": latest_step_record["sha256"],
            "judge_response_receipts_sha256": sha256_file(
                response_receipts_path
            ),
        }
        write_json_new(temporary / "run_provenance.json", run_provenance)
        checkpoint_provenance = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": ARCHITECTURE,
            "optimizer_step": optimizer_step,
            "run_contract_sha256": run_contract_sha256,
            "decoder_adapter_sha256": sha256_artifact(adapter),
            "source_overlay_sha256": sha256_file(overlay_path),
            "compact_contract_sha256": sha256_file(
                temporary / "compact_contract.json"
            ),
            "optimizer_sha256": sha256_file(optimizer_path),
            "rng_state_sha256": sha256_file(rng_path),
            "trainer_state_sha256": sha256_file(
                temporary / "trainer_state.json"
            ),
            "rollout_journal_sha256": sha256_file(
                temporary / "rollout_journal.json"
            ),
            "rollout_journal_chain_sha256": journal_attestation[
                "cumulative_chain_sha256"
            ],
            "latest_step_journal_sha256": latest_step_record["sha256"],
            "judge_telemetry_sha256": sha256_file(
                temporary / "judge_telemetry.json"
            ),
            "judge_response_ids_sha256": sha256_file(response_ids_path),
            "judge_response_receipts_sha256": sha256_file(
                response_receipts_path
            ),
            "run_provenance_sha256": sha256_file(
                temporary / "run_provenance.json"
            ),
        }
        write_json_new(
            temporary / "checkpoint_provenance.json", checkpoint_provenance
        )
        temporary.rename(destination)
    except Exception:
        # Preserve incomplete state for forensic inspection.  A subsequent run
        # fails closed rather than deleting or overwriting it.
        raise
    return destination


def build_run_contract(
    args: argparse.Namespace,
    *,
    contract: DirectCompactContract,
    decoder_model: str,
    decoder_revision: str,
    model_config_path: str | Path,
    warmstart: Mapping[str, Path],
    teacher_source_attestation: Mapping[str, Any],
    rollout_task_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "architecture": ARCHITECTURE,
        "predeclared_chain_contract": {
            "path": str(
                Path(args.predeclared_chain_contract).expanduser().resolve()
            ),
            "sha256": sha256_file(args.predeclared_chain_contract),
        },
        "executable_view_report": {
            "path": str(
                Path(args.executable_view_report).expanduser().resolve()
            ),
            "sha256": sha256_file(args.executable_view_report),
        },
        "verpo_feedback_public_manifest": {
            "path": str(
                Path(args.feedback_view_public_manifest).expanduser().resolve()
            ),
            "sha256": sha256_file(args.feedback_view_public_manifest),
        },
        "rollout_file": {
            "path": str(Path(args.rollout_file).expanduser().resolve()),
            "sha256": sha256_file(args.rollout_file),
        },
        "rollout_seal_sha256": sha256_file(args.rollout_seal),
        "compact_contract_sha256": sha256_file(args.contract),
        "codebook_sha256": sha256_file(args.codebook),
        "codec_sha256": sha256_file(args.codec_artifact),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_json),
        "decoder_model": decoder_model,
        "decoder_revision": decoder_revision,
        "model_config_sha256": sha256_file(model_config_path),
        "warmstart": {
            "path": str(warmstart["root"]),
            "decoder_adapter_sha256": sha256_artifact(warmstart["adapter"]),
            "source_overlay_sha256": sha256_file(warmstart["overlay"]),
            "contract_sha256": sha256_file(warmstart["contract"]),
            "provenance_sha256": sha256_file(warmstart["provenance"]),
        },
        "generation": {
            "group_size": args.group_size,
            "rollout_batch_size": args.rollout_batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "task_sampling": task_sampling_contract(
            rollout_task_ids,
            seed=args.seed,
            max_updates=args.max_updates,
            rollout_batch_size=args.rollout_batch_size,
        ),
        "verifier": {
            "reward_test_field": "feedback_tests",
            "workers": args.reward_workers,
            "timeout": args.reward_timeout,
            "stability_runs": args.reward_stability_runs,
            "completion_attestation": COMPLETION_ATTESTATION_ID,
            "full_and_per_test": True,
        },
        "reward": {
            "verpo_alpha": args.verpo_alpha,
            "verpo_beta": args.verpo_beta,
            "kde_bandwidth": "population_std_over_2",
            "global_reward": "verifier_full_suite_binary",
            "local_reward": "density_calibrated_visible_per_test",
            "global_advantage": "global_reward_minus_group_mean",
            "local_advantage": "local_reward_minus_group_mean",
            "advantage_normalization_factor": 1,
            "unified_advantage": (
                "A_global + verpo_beta*A_local + "
                "judge_weight*A_teacher"
            ),
            "judge_weight": args.judge_weight,
            "teacher_signal": {
                "observed": (
                    "selected_compiling_failure_score_in_[0,1]"
                ),
                "unobserved": "missing_with_exact_zero_advantage",
                "advantage": "observed_score_minus_observed_subset_mean",
                "separately_centered_over_observed_subset": True,
                "full_pass_or_non_compiling_teacher_mask": False,
                "paper_extension": True,
            },
            "population_std_advantage_division": False,
        },
        "judge": {
            "mode": args.judge_mode,
            "provider": "provider-neutral",
            "model": args.judge_model,
            "base_url": args.judge_base_url.rstrip("/"),
            "api_style": args.judge_api_style,
            "interval": args.judge_interval,
            "group_top_n": args.judge_group_top_n,
            "deadline_seconds": args.judge_deadline_seconds,
            "failure_policy": args.judge_failure_policy,
            "reasoning_mode": args.judge_reasoning_mode,
            "max_calls": args.judge_max_calls,
            "escalation_queue": str(
                Path(args.judge_escalation_queue).expanduser().resolve()
            ),
            "thinking_mode": args.judge_thinking_mode,
            "reasoning_effort": args.judge_reasoning_effort,
            "max_tokens": args.judge_max_tokens,
            "completion_retries": args.judge_completion_retries,
            "retry_max_tokens": args.judge_retry_max_tokens,
            "timeout_seconds": args.judge_timeout_seconds,
            "max_retries": args.judge_max_retries,
            "concurrency": args.judge_concurrency,
            "selection": (
                "no_full_pass_and_at_least_two_compiling_failures_"
                "then_top_local_reward_stable"
            ),
            "one_request_per_selected_group": True,
            "compile_gated": True,
            "unjudged_is_missing_not_zero_score": True,
            "teacher_source": dict(teacher_source_attestation),
            "compressed_enriched_assembly_exposed": True,
            "compressed_cfg_exposed": True,
            "visible_feedback_tests_only": True,
            "reference_or_acceptance_tests": False,
            "response_provenance": {
                "receipt_schema": DEEPSEEK_RECEIPT_SCHEMA,
                "receipt_attestation_schema": (
                    DEEPSEEK_RECEIPT_ATTESTATION_SCHEMA
                ),
                "exact_response_model_required": (
                    args.judge_api_style != "openai_responses"
                ),
                "unique_response_id_required_across_resumes": True,
                "nonempty_system_fingerprint_required": (
                    args.judge_api_style != "openai_responses"
                ),
                "positive_usage_and_exact_total_equality_required": True,
                "append_fsync_before_optimizer_step": True,
                "receipt_hash_chain_checkpointed": True,
                "response_id_set_checkpointed": True,
                "plaintext_prompts_persisted": False,
                "plaintext_reasoning_persisted": False,
            },
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "ppo_clip": args.ppo_clip,
            "sft_replay_weight": args.sft_replay_weight,
            "on_policy_logprob_tolerance": (
                args.on_policy_logprob_tolerance
            ),
            "max_updates": args.max_updates,
            "checkpoint_interval": args.checkpoint_interval,
            "updates_per_rollout_batch": 1,
        },
        "runtime": {
            "seed": args.seed,
            "attn_implementation": args.attn_implementation,
            "load_4bit": bool(args.load_4bit),
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
            "contract_max_source_tokens": contract.max_source_tokens,
            "contract_max_target_tokens": contract.max_target_tokens,
            "contract_max_total_tokens": contract.max_total_tokens,
            "heldout_loaded_during_training": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--rollout_file", required=True)
    parser.add_argument("--rollout_seal", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codec_artifact", required=True)
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument("--teacher_prompt_jsonl", default="")
    parser.add_argument("--expected_teacher_prompt_sha256", default="")
    parser.add_argument("--teacher_prompt_manifest", default="")
    parser.add_argument("--expected_teacher_prompt_manifest_sha256", default="")
    parser.add_argument("--executable_view_report", required=True)
    parser.add_argument(
        "--expected_executable_view_report_sha256",
        required=True,
    )
    parser.add_argument("--feedback_view_public_manifest", required=True)
    parser.add_argument(
        "--expected_feedback_view_public_manifest_sha256", required=True
    )
    parser.add_argument(
        "--expected_feedback_eligible_rows",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["eligible_rows"],
    )
    parser.add_argument(
        "--expected_feedback_excluded_rows",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["excluded_rows"],
    )
    parser.add_argument(
        "--expected_feedback_source_expect_cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["source_expect_cases"],
    )
    parser.add_argument(
        "--expected_feedback_visible_expect_cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["visible_expect_cases"],
    )
    parser.add_argument(
        "--expected_feedback_holdback_expect_cases",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["holdback_expect_cases"],
    )
    parser.add_argument(
        "--expected_feedback_odd_case_tasks",
        type=int,
        default=PRODUCTION_EXPECTED_ACCOUNTING["odd_case_tasks"],
    )
    parser.add_argument(
        "--expected_feedback_eligible_task_ids_sha256",
        default=PRODUCTION_ELIGIBLE_TASK_IDS_SHA256,
    )
    parser.add_argument(
        "--expected_feedback_excluded_task_ids_sha256",
        default=PRODUCTION_EXCLUDED_TASK_IDS_SHA256,
    )
    parser.add_argument(
        "--derive_feedback_accounting_from_sealed_manifest",
        action="store_true",
        help=(
            "Trust counts and task-ID commitments only through the pinned "
            "trainer-safe feedback manifest."
        ),
    )
    parser.add_argument(
        "--expected_parent_fit_rows",
        type=int,
        default=1580,
        help="Expanded production passes 2776; legacy compatibility is 1580.",
    )
    parser.add_argument("--predeclared_chain_contract", required=True)
    parser.add_argument(
        "--expected_predeclared_chain_sha256", required=True
    )
    parser.add_argument("--warmstart_checkpoint", required=True)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resume_unstarted", action="store_true")
    parser.add_argument("--storage_preflight_only", action="store_true")
    parser.add_argument("--decoder_model", default="")
    parser.add_argument("--decoder_revision", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer_revision", default="")
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--rollout_batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--max_updates", type=int, default=1232)
    parser.add_argument("--checkpoint_interval", type=int, default=154)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ppo_clip", type=float, default=0.0)
    parser.add_argument("--sft_replay_weight", type=float, default=0.05)
    parser.add_argument("--on_policy_logprob_tolerance", type=float, default=2e-4)
    parser.add_argument("--verpo_alpha", type=float, default=2.0)
    parser.add_argument("--verpo_beta", type=float, default=1.0)
    parser.add_argument("--reward_workers", type=int, default=8)
    parser.add_argument("--reward_timeout", type=int, default=30)
    parser.add_argument("--reward_stability_runs", type=int, default=1)
    parser.add_argument("--judge_weight", type=float, default=0.25)
    parser.add_argument(
        "--judge_mode",
        choices=["off", "sparse_inline", "offline_queue"],
        default="sparse_inline",
    )
    parser.add_argument("--judge_interval", type=int, default=8)
    parser.add_argument("--judge_group_top_n", type=int, default=2)
    parser.add_argument("--judge_deadline_seconds", type=float, default=60.0)
    parser.add_argument(
        "--judge_failure_policy",
        choices=["local_only"],
        default="local_only",
    )
    parser.add_argument(
        "--judge_reasoning_mode",
        choices=["standard", "pro"],
        default="standard",
    )
    parser.add_argument("--judge_max_calls", type=int, default=None)
    parser.add_argument(
        "--judge_escalation_queue",
        default="teacher-escalations.jsonl",
    )
    parser.add_argument("--judge_model", default="gpt-5.6-terra")
    parser.add_argument("--judge_base_url", default="https://api.openai.com/v1")
    parser.add_argument(
        "--judge_api_style",
        choices=["openai_responses", "openai_compatible_chat"],
        default="openai_responses",
    )
    parser.add_argument("--judge_concurrency", type=int, default=1)
    parser.add_argument("--judge_max_tokens", type=int, default=12288)
    parser.add_argument("--judge_completion_retries", type=int, default=0)
    parser.add_argument("--judge_retry_max_tokens", type=int, default=12288)
    parser.add_argument("--judge_thinking_mode", default="provider_default")
    parser.add_argument(
        "--judge_reasoning_effort",
        choices=["high", "max"],
        default="high",
    )
    parser.add_argument(
        "--judge_timeout_seconds",
        type=float,
        default=None,
        help="Deprecated alias for --judge_deadline_seconds.",
    )
    parser.add_argument("--judge_max_retries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args(argv)
    if args.judge_timeout_seconds is not None:
        args.judge_deadline_seconds = args.judge_timeout_seconds
    args.judge_timeout_seconds = args.judge_deadline_seconds
    if args.judge_max_calls is None:
        total_groups = args.max_updates * args.rollout_batch_size
        args.judge_max_calls = (
            0
            if args.judge_mode != "sparse_inline"
            else math.ceil(total_groups / max(1, args.judge_interval))
        )
    if (
        args.judge_mode == "sparse_inline"
        and args.judge_reasoning_mode == "pro"
    ):
        parser.error(
            "--judge_reasoning_mode=pro is offline-only and cannot block "
            "--judge_mode=sparse_inline"
        )
    return args


def _validate_args(args: argparse.Namespace, contract: DirectCompactContract) -> None:
    validate_rollout_distribution(
        group_size=args.group_size,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    if args.rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive")
    if args.max_updates <= 0:
        raise ValueError("max_updates must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if (
        not args.derive_feedback_accounting_from_sealed_manifest
        and
        args.max_updates * args.rollout_batch_size
        < args.expected_feedback_eligible_rows
    ):
        raise ValueError(
            "production VeRPO must cover every eligible training task at "
            "least once before completion"
        )
    if args.max_new_tokens <= 0 or args.max_new_tokens > contract.max_target_tokens:
        raise ValueError("max_new_tokens exceeds the compact target contract")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if args.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if args.sft_replay_weight < 0.0:
        raise ValueError("sft_replay_weight must be non-negative")
    if args.on_policy_logprob_tolerance <= 0.0:
        raise ValueError("on-policy tolerance must be positive")
    if args.reward_workers <= 0 or args.reward_timeout <= 0:
        raise ValueError("reward workers/timeout must be positive")
    if args.reward_stability_runs <= 0:
        raise ValueError("reward_stability_runs must be positive")
    if args.judge_weight < 0.0:
        raise ValueError("judge_weight must be non-negative")
    if args.verpo_alpha <= 0.0 or args.verpo_beta < 0.0:
        raise ValueError("VeRPO alpha/beta values are invalid")
    if args.judge_interval <= 0:
        raise ValueError("judge_interval must be positive")
    if args.judge_group_top_n < 2 or args.judge_group_top_n > args.group_size:
        raise ValueError("judge_group_top_n must lie in [2, group_size]")
    if args.judge_max_calls < 0:
        raise ValueError("judge_max_calls must be non-negative")
    if not args.judge_escalation_queue and args.judge_mode != "off":
        raise ValueError("teacher-enabled modes require an escalation queue")
    if (
        args.judge_mode == "sparse_inline"
        and args.judge_reasoning_mode != "standard"
    ):
        raise ValueError("inline VeRPO cannot use pro reasoning mode")
    if args.judge_failure_policy != "local_only":
        raise ValueError("inline teacher failures must fall back to local rewards")
    if not args.judge_model.strip() or not args.judge_base_url.strip():
        raise ValueError("judge model/base URL must be nonempty")
    if args.judge_concurrency <= 0:
        raise ValueError("judge concurrency must be positive")
    if args.judge_max_tokens <= 0 or args.judge_retry_max_tokens <= 0:
        raise ValueError("judge token budgets must be positive")
    if args.judge_retry_max_tokens < args.judge_max_tokens:
        raise ValueError("judge retry token budget cannot shrink")
    if (
        args.judge_completion_retries != 0
        or args.judge_max_retries != 0
        or args.judge_deadline_seconds <= 0.0
        or args.judge_deadline_seconds > 60.0
    ):
        raise ValueError(
            "inline judge requires zero retries and a deadline in (0,60]"
        )
    if args.judge_retry_max_tokens != args.judge_max_tokens:
        raise ValueError("zero-retry judge must use one fixed token budget")
    if args.judge_thinking_mode not in {
        "disabled",
        "enabled",
        "provider_default",
    }:
        raise ValueError("judge thinking mode is invalid")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        args.expected_predeclared_chain_sha256.strip().lower(),
    ):
        raise ValueError("expected predeclared-chain SHA-256 is invalid")
    if (
        sha256_file(args.executable_view_report)
        != args.expected_executable_view_report_sha256.strip().lower()
    ):
        raise ValueError("parent executable-view report hash mismatch")
    if not args.teacher_prompt_jsonl:
        raise ValueError(
            "VeRPO requires --teacher_prompt_jsonl with the exact "
            "F2 enriched assembly+CFG artifact"
        )
    if not re.fullmatch(
        r"[0-9a-f]{64}", args.expected_teacher_prompt_sha256.strip().lower()
    ):
        raise ValueError("--expected_teacher_prompt_sha256 must be a SHA-256")
    if not args.teacher_prompt_manifest:
        raise ValueError(
            "VeRPO requires --teacher_prompt_manifest with the exact "
            "manifest-bound F2 system prompt"
        )
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        args.expected_teacher_prompt_manifest_sha256.strip().lower(),
    ):
        raise ValueError(
            "--expected_teacher_prompt_manifest_sha256 must be a SHA-256"
        )
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.expected_parent_fit_rows <= 0:
        raise ValueError("expected_parent_fit_rows must be positive")
    if args.resume_checkpoint and args.resume_unstarted:
        raise ValueError("checkpoint and unstarted recovery are mutually exclusive")
    if args.ppo_clip < 0.0 or args.ppo_clip >= 1.0:
        raise ValueError("ppo_clip must be zero or lie in (0,1)")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    contract = DirectCompactContract.load(args.contract)
    _validate_args(args, contract)
    predeclared_chain_path = Path(
        args.predeclared_chain_contract
    ).expanduser().resolve()
    if not predeclared_chain_path.is_file():
        raise FileNotFoundError(predeclared_chain_path)
    if (
        sha256_file(predeclared_chain_path)
        != args.expected_predeclared_chain_sha256.strip().lower()
    ):
        raise ValueError("predeclared chain differs from its expected SHA-256")
    predeclared_chain = _read_json_object(predeclared_chain_path)
    predeclared_payload = predeclared_chain.get("payload")
    if (
        predeclared_chain.get("schema")
        != "post-qwen-predeclared-training-chain-v1"
        or not isinstance(predeclared_payload, Mapping)
    ):
        raise ValueError("predeclared chain has an invalid schema/payload")
    validate_dart_binary()

    output_dir = Path(args.output_dir).expanduser().resolve()
    original_warmstart = validate_warmstart_checkpoint(
        args.warmstart_checkpoint, contract_path=args.contract
    )
    if output_dir == original_warmstart["root"]:
        raise ValueError("VeRPO output cannot overwrite its warm-start checkpoint")

    seal = validate_join_seal(
        args.rollout_file,
        args.rollout_seal,
        args.contract,
        expected_role="fit",
    )
    expected_feedback_accounting: Mapping[str, Any] | None = {
        "parent_rows": PRODUCTION_EXPECTED_ACCOUNTING["parent_rows"],
        "eligible_rows": args.expected_feedback_eligible_rows,
        "excluded_rows": args.expected_feedback_excluded_rows,
        "source_expect_cases": (
            args.expected_feedback_source_expect_cases
        ),
        "visible_expect_cases": (
            args.expected_feedback_visible_expect_cases
        ),
        "holdback_expect_cases": (
            args.expected_feedback_holdback_expect_cases
        ),
        "odd_case_tasks": args.expected_feedback_odd_case_tasks,
    }
    expected_eligible_digest: str | None = (
        args.expected_feedback_eligible_task_ids_sha256
    )
    expected_excluded_digest: str | None = (
        args.expected_feedback_excluded_task_ids_sha256
    )
    if args.derive_feedback_accounting_from_sealed_manifest:
        expected_feedback_accounting = None
        expected_eligible_digest = None
        expected_excluded_digest = None
    feedback_view = validate_feedback_training_boundary(
        rollout=args.rollout_file,
        seal=args.rollout_seal,
        f2=args.teacher_prompt_jsonl,
        f2_manifest=args.teacher_prompt_manifest,
        public_manifest=args.feedback_view_public_manifest,
        expected_public_manifest_sha256=(
            args.expected_feedback_view_public_manifest_sha256
        ),
        contract=args.contract,
        expected_accounting=expected_feedback_accounting,
        expected_eligible_task_ids_sha256=expected_eligible_digest,
        expected_excluded_task_ids_sha256=expected_excluded_digest,
    )
    attested_accounting = feedback_view["accounting"]
    args.expected_feedback_eligible_rows = int(
        attested_accounting["eligible_rows"]
    )
    args.expected_feedback_excluded_rows = int(
        attested_accounting["excluded_rows"]
    )
    args.expected_feedback_source_expect_cases = int(
        attested_accounting["source_expect_cases"]
    )
    args.expected_feedback_visible_expect_cases = int(
        attested_accounting["visible_expect_cases"]
    )
    args.expected_feedback_holdback_expect_cases = int(
        attested_accounting["holdback_expect_cases"]
    )
    args.expected_feedback_odd_case_tasks = int(
        attested_accounting["odd_case_tasks"]
    )
    args.expected_feedback_eligible_task_ids_sha256 = feedback_view[
        "task_ids_sha256"
    ]
    args.expected_feedback_excluded_task_ids_sha256 = feedback_view[
        "excluded_task_ids_sha256"
    ]
    if (
        args.max_updates * args.rollout_batch_size
        < args.expected_feedback_eligible_rows
    ):
        raise ValueError(
            "production VeRPO must cover every sealed eligible task at least "
            "once before completion"
        )
    declared_verpo = predeclared_payload.get("verpo")
    declared_executable = predeclared_payload.get("executable_train")
    if (
        not isinstance(declared_verpo, Mapping)
        or not isinstance(declared_executable, Mapping)
        or declared_verpo.get("feedback_accounting")
        != dict(attested_accounting)
        or declared_verpo.get("feedback_eligible_task_ids_sha256")
        != feedback_view["task_ids_sha256"]
        or declared_verpo.get("feedback_excluded_task_ids_sha256")
        != feedback_view["excluded_task_ids_sha256"]
        or declared_verpo.get("rollout_rows") != feedback_view["rows"]
        or declared_verpo.get("parent_safe_rows")
        != feedback_view["parent_rows"]
        or (
            args.derive_feedback_accounting_from_sealed_manifest
            and declared_verpo.get("parent_fit_rows")
            != args.expected_parent_fit_rows
        )
        or (declared_executable.get("report") or {}).get("sha256")
        != sha256_file(args.executable_view_report)
    ):
        raise ValueError(
            "VeRPO invocation differs from the sealed predeclared chain"
        )
    if (
        int(seal.get("rows", -1)) != feedback_view["rows"]
        or seal.get("training_objective_scope") != ROLLOUT_SCOPE
        or seal.get("representation_schema") != REPRESENTATION_SCHEMA
        or seal.get("execution_ineligible_task_ids") != []
        or feedback_view.get("acceptance_tests_exposed") is not False
        or feedback_view.get("reward_holdback_exposed") is not False
    ):
        raise ValueError(
            "VeRPO rollout is not the sealed TRAIN-only feedback subset "
            "derived from the executable fit view"
        )
    teacher_sources, teacher_source_attestation = load_teacher_visible_sources(
        args.teacher_prompt_jsonl,
        expected_sha256=args.expected_teacher_prompt_sha256,
        manifest_path=args.teacher_prompt_manifest,
        expected_manifest_sha256=(
            args.expected_teacher_prompt_manifest_sha256
        ),
        student_tokenizer_sha256=sha256_file(args.tokenizer_json),
    )
    teacher_source_attestation["feedback_view"] = feedback_view
    decoder_model = args.decoder_model.strip() or contract.decoder_model
    decoder_revision = args.decoder_revision.strip() or contract.decoder_revision
    decoder_config_path = resolve_decoder_config_path(
        decoder_model, decoder_revision
    )
    contract.validate_decoder_binding(
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        model_config_path=decoder_config_path,
    )
    run_contract = build_run_contract(
        args,
        contract=contract,
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        model_config_path=decoder_config_path,
        warmstart=original_warmstart,
        teacher_source_attestation=teacher_source_attestation,
        rollout_task_ids=list(teacher_sources),
    )
    run_contract_hash = canonical_sha256(run_contract)

    if args.resume_checkpoint:
        if not output_dir.is_dir():
            raise ValueError("resume output_dir does not exist")
        if (output_dir / "completed.json").exists():
            raise ValueError("the requested VeRPO run is already complete")
        saved_run_contract = _read_json_object(output_dir / "run_contract.json")
        if canonical_sha256(saved_run_contract) != run_contract_hash:
            raise ValueError("resume arguments differ from the sealed run contract")
        resume_provenance, resume_paths = validate_resume_checkpoint(
            args.resume_checkpoint,
            output_dir=output_dir,
            run_contract_sha256=run_contract_hash,
            compact_contract_path=args.contract,
        )
        load_checkpoint = resume_paths
        start_step = int(resume_provenance["optimizer_step"])
        parent_journal_chain = str(
            resume_provenance.get("rollout_journal_chain_sha256") or ""
        )
        if not re.fullmatch(r"[0-9a-f]{64}", parent_journal_chain):
            raise ValueError("resume checkpoint lacks a journal-chain binding")
        prior_judge_telemetry = _read_json_object(
            resume_paths["judge_telemetry"]
        )
        prior_response_id_record = _read_json_object(
            resume_paths["judge_response_ids"]
        )
        prior_response_id_sha256s = list(
            prior_response_id_record["response_id_sha256s"]
        )
        prior_response_receipts_path: Path | None = resume_paths[
            "judge_response_receipts"
        ]
    elif args.resume_unstarted:
        if not output_dir.is_dir():
            raise ValueError("unstarted recovery output_dir does not exist")
        if (output_dir / "completed.json").exists():
            raise ValueError("the requested VeRPO run is already complete")
        if any(
            child.is_dir() and _CHECKPOINT_RE.fullmatch(child.name)
            for child in output_dir.iterdir()
        ):
            raise ValueError(
                "unstarted recovery is forbidden after a checkpoint exists"
            )
        saved_run_contract = _read_json_object(output_dir / "run_contract.json")
        if canonical_sha256(saved_run_contract) != run_contract_hash:
            raise ValueError(
                "unstarted recovery arguments differ from the sealed run contract"
            )
        load_checkpoint = original_warmstart
        start_step = 0
        parent_journal_chain = "0" * 64
        prior_judge_telemetry = {}
        prior_response_id_sha256s = []
        prior_response_receipts_path = None
    else:
        if output_dir.exists():
            raise ValueError(f"fresh VeRPO output path already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        write_json_new(output_dir / "run_contract.json", run_contract)
        load_checkpoint = original_warmstart
        start_step = 0
        parent_journal_chain = "0" * 64
        prior_judge_telemetry = {}
        prior_response_id_sha256s = []
        prior_response_receipts_path = None
    prior_receipt_count = int(prior_judge_telemetry.get("receipt_count", 0))
    prior_receipt_chain = str(
        prior_judge_telemetry.get(
            "receipt_chain_sha256", RECEIPT_CHAIN_GENESIS_SHA256
        )
    )
    if (
        prior_receipt_count != len(prior_response_id_sha256s)
        or not re.fullmatch(r"[0-9a-f]{64}", prior_receipt_chain)
    ):
        raise ValueError(
            "prior DeepSeek receipt count/chain/response IDs are inconsistent"
        )
    if start_step > args.max_updates:
        raise ValueError("resume checkpoint exceeds max_updates")
    if start_step == args.max_updates:
        if not args.resume_checkpoint:
            raise AssertionError("only a final checkpoint can finalize a run")
        final_checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
        publish_completed_run(
            output_dir=output_dir,
            max_updates=args.max_updates,
            checkpoint_interval=args.checkpoint_interval,
            latest_checkpoint=final_checkpoint,
            run_contract_sha256=run_contract_hash,
            judge_telemetry=prior_judge_telemetry,
            published_checkpoints_this_process=0,
            finalize_only_recovery=True,
        )
        print(
            f"VERPO_FINALIZE_ONLY_RECOVERY checkpoint={final_checkpoint}",
            flush=True,
        )
        return

    # Heavy model dependencies remain lazy so pure reward/resume tests run on CPU.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_name = args.tokenizer or decoder_model
    tokenizer_revision = (
        args.tokenizer_revision.strip()
        or (decoder_revision if tokenizer_name == decoder_model else "")
        or None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("VeRPO tokenizer requires EOS or pad token")
        tokenizer.pad_token = tokenizer.eos_token
    contract.validate_artifacts(
        tokenizer=tokenizer,
        tokenizer_json_path=args.tokenizer_json,
        codec_path=args.codec_artifact,
        codebook_path=args.codebook,
    )
    dataset = CompactVerpoDataset(
        args.rollout_file,
        tokenizer=tokenizer,
        contract=contract,
        enable_replay=args.sft_replay_weight > 0.0,
        max_generation_tokens=args.max_new_tokens,
        teacher_sources=teacher_sources,
    )
    if (
        len(dataset) != feedback_view["rows"]
        or len(teacher_sources) != feedback_view["rows"]
        or set(dataset.task_ids) != set(teacher_sources)
    ):
        raise ValueError(
            "VeRPO compact rows and DeepSeek F2 sources must cover the "
            "same exact attested feedback-eligible task subset"
        )
    if contract.schema == CONTRACT_SCHEMA_V3:
        expected_pool_uses = int(seal["pool_metadata"]["total_use_count"])
        if dataset.pool_use_count != expected_pool_uses:
            raise ValueError("decoded v3 pool uses differ from the sealed total")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.bf16 and not args.load_4bit:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16 and not args.load_4bit:
        model_kwargs["torch_dtype"] = torch.float16
    if args.load_4bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit VeRPO requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(
        decoder_model,
        revision=decoder_revision,
        **model_kwargs,
    )
    context_limit = _context_limit(model.config)
    if context_limit is None or contract.max_total_tokens > context_limit:
        raise ValueError("decoder context is smaller than the compact contract")
    validate_base_model_vocab(model, contract)
    model = PeftModel.from_pretrained(
        model, str(load_checkpoint["adapter"]), is_trainable=True
    )
    overlay = restore_source_embedding_overlay(
        model,
        dict(contract.source_token_expansions),
        load_checkpoint["overlay"],
        base_vocab_size=int(contract.base_vocab_size or 0),
    )
    if model.get_output_embeddings().weight.size(0) != contract.base_vocab_size:
        raise RuntimeError("compact overlay unexpectedly resized Qwen LM head")
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if not args.load_4bit:
        model.to(device)
    model.eval()  # Gradients stay enabled; dropout must not break on-policy equality.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    parameters = _trainable_parameters(model)
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    checkpoint_steps = [
        step
        for step in range(start_step + 1, args.max_updates + 1)
        if step % args.checkpoint_interval == 0 or step == args.max_updates
    ]
    # Adapter weights plus AdamW parameter/state tensors dominate each immutable
    # checkpoint.  Twenty bytes per trainable parameter and a 25% reserve is a
    # deliberately conservative preflight; no checkpoint pruning or overwrite
    # is required to stay within the declared plan.
    estimated_bytes_per_checkpoint = max(
        1 << 30, trainable_parameter_count * 20
    )
    required_free_bytes = int(
        estimated_bytes_per_checkpoint * len(checkpoint_steps) * 1.25
        + (2 << 30)
    )
    available_free_bytes = shutil.disk_usage(output_dir).free
    storage_passed = available_free_bytes >= required_free_bytes
    storage_preflight = {
        "schema": "direct-compact-verpo-storage-preflight-v1",
        "run_contract_sha256": run_contract_hash,
        "resume_from_optimizer_step": start_step,
        "max_updates": args.max_updates,
        "checkpoint_interval": args.checkpoint_interval,
        "planned_checkpoint_steps": checkpoint_steps,
        "trainable_parameter_count": trainable_parameter_count,
        "estimated_bytes_per_checkpoint": estimated_bytes_per_checkpoint,
        "required_free_bytes": required_free_bytes,
        "available_free_bytes_at_preflight": available_free_bytes,
        "safety_factor": 1.25,
        "minimum_noncheckpoint_reserve_bytes": 2 << 30,
        "checkpoint_pruning": False,
        "passed": storage_passed,
        "destructive_cleanup_attempted": False,
    }
    storage_preflight_root = output_dir / "storage_preflights"
    storage_preflight_root.mkdir(exist_ok=True)
    prior_preflights = sorted(
        storage_preflight_root.glob(
            f"preflight-from-{start_step:06d}-attempt-*.json"
        )
    )
    storage_preflight_path = storage_preflight_root / (
        f"preflight-from-{start_step:06d}-attempt-"
        f"{len(prior_preflights) + 1:04d}.json"
    )
    write_json_new(storage_preflight_path, storage_preflight)
    print(
        "VERPO_STORAGE_PREFLIGHT "
        f"path={storage_preflight_path} "
        f"required_bytes={required_free_bytes} "
        f"available_bytes={available_free_bytes} "
        f"passed={str(storage_passed).lower()}",
        flush=True,
    )
    if not storage_passed:
        raise RuntimeError(
            "insufficient free disk for immutable VeRPO checkpoint plan: "
            f"report={storage_preflight_path} need={required_free_bytes} "
            f"free={available_free_bytes} "
            f"remaining_checkpoints={len(checkpoint_steps)}; "
            "no cleanup was attempted"
        )
    if args.storage_preflight_only:
        print(
            "VERPO_STORAGE_PREFLIGHT_ONLY_COMPLETE "
            f"report={storage_preflight_path}",
            flush=True,
        )
        return
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.resume_checkpoint:
        optimizer_state = torch.load(
            load_checkpoint["optimizer"], map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(optimizer_state)
        _optimizer_to_parameter_device(optimizer)
        restore_rng_state(load_checkpoint["rng"])
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    journal_attempt = create_journal_attempt(
        output_dir,
        start_step=start_step,
        run_contract_sha256=run_contract_hash,
        parent_chain_sha256=parent_journal_chain,
        parent_receipt_chain_sha256=prior_receipt_chain,
        receipt_index_offset=prior_receipt_count,
    )

    from scripts.training.verpo_judge_antigravity import VerpoJudge

    prior_group_calls = int(
        prior_judge_telemetry.get("group_calls_attempted", 0)
    )
    remaining_group_calls = max(
        0, int(args.judge_max_calls) - prior_group_calls
    )
    receipt_journal_path = (
        journal_attempt / "deepseek_response_receipts.jsonl"
    )
    if args.judge_mode == "sparse_inline":
        judge = VerpoJudge(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_style=args.judge_api_style,
            concurrency=args.judge_concurrency,
            max_tokens=args.judge_max_tokens,
            timeout_seconds=args.judge_deadline_seconds,
            max_retries=args.judge_max_retries,
            completion_retries=args.judge_completion_retries,
            retry_max_tokens=args.judge_retry_max_tokens,
            thinking_mode=args.judge_thinking_mode,
            reasoning_effort=args.judge_reasoning_effort,
            reasoning_mode=args.judge_reasoning_mode,
            max_calls=remaining_group_calls,
            fail_closed=False,
            receipt_chain_seed=prior_receipt_chain,
            receipt_index_offset=prior_receipt_count,
            prior_response_id_sha256s=prior_response_id_sha256s,
            receipt_journal_path=receipt_journal_path,
        )
    else:
        judge = DisabledVerpoJudge(
            receipt_chain_seed=prior_receipt_chain,
            receipt_index_offset=prior_receipt_count,
            prior_response_id_sha256s=prior_response_id_sha256s,
            receipt_journal_path=receipt_journal_path,
            mode=args.judge_mode,
        )
    if args.judge_mode == "sparse_inline" and remaining_group_calls > 0:
        judge.validate_configuration()

    replay_collator = DirectCompactBatchCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        max_source_tokens=contract.max_source_tokens,
        max_target_tokens=contract.max_target_tokens,
        max_total_tokens=contract.max_total_tokens,
        source_token_ids=contract.source_token_ids,
    )
    base_provenance = {
        "decoder_model": decoder_model,
        "decoder_revision": decoder_revision,
        "model_config_sha256": sha256_file(decoder_config_path),
        "attn_implementation": args.attn_implementation,
        "codebook_sha256": sha256_file(args.codebook),
        "codec_sha256": sha256_file(args.codec_artifact),
        "train_file_sha256": sha256_file(args.rollout_file),
        "train_seal_sha256": sha256_file(args.rollout_seal),
        "train_sealed_rows": int(seal["rows"]),
        "teacher_source_artifact": teacher_source_attestation,
        "training_schedule": run_contract["optimizer"],
        "warmstart_checkpoint": run_contract["warmstart"],
        "feedback_view": feedback_view,
        "heldout_loaded_during_training": False,
        "graph_encoder": None,
        "soft_prefix": None,
    }
    rollout_schedule = deterministic_task_schedule(
        dataset.task_ids,
        seed=args.seed,
        rollout_groups=args.max_updates * args.rollout_batch_size,
    )
    planned_schedule_ids = [
        dataset.task_ids[index] for index in rollout_schedule
    ]
    sampling_contract = run_contract["task_sampling"]
    if (
        canonical_sha256(dataset.task_ids)
        != sampling_contract["dataset_task_ids_sha256"]
        or canonical_sha256(planned_schedule_ids)
        != sampling_contract["planned_schedule_task_ids_sha256"]
        or len(set(planned_schedule_ids))
        != sampling_contract["planned_unique_tasks"]
    ):
        raise ValueError("VeRPO deterministic task schedule differs from run contract")
    current_journal_chain = parent_journal_chain

    eos_id = tokenizer.eos_token_id
    total_eligible_judgements = 0
    latest_checkpoint: Path | None = None
    published_checkpoints = 0
    for optimizer_step in range(start_step + 1, args.max_updates + 1):
        receipt_cursor = int(judge.telemetry()["receipt_count"])
        policy_version = optimizer_step - 1
        batch_groups: list[dict[str, Any]] = []
        for batch_index in range(args.rollout_batch_size):
            schedule_position = (
                (optimizer_step - 1) * args.rollout_batch_size + batch_index
            )
            record = dataset[rollout_schedule[schedule_position]]
            prefix = list(record.prompt_ids) + list(record.compact_ids)
            if len(prefix) + args.max_new_tokens > contract.max_total_tokens:
                raise ValueError(
                    f"{record.task_id}: generation would exceed compact context contract"
                )
            prompt_tensor = torch.tensor(
                [prefix], dtype=torch.long, device=device
            )
            attention = torch.ones_like(prompt_tensor)
            candidates: list[dict[str, Any]] = []
            for group_index in range(args.group_size):
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=prompt_tensor,
                        attention_mask=attention,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=1.0,
                        top_k=0,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_id,
                        use_cache=True,
                    )
                completion_ids = generated[0, len(prefix) :].tolist()
                if not completion_ids:
                    raise RuntimeError("rollout produced an empty completion")
                old_logprobs = completion_token_logprobs(
                    model,
                    prefix,
                    completion_ids,
                    temperature=args.temperature,
                    device=device,
                    with_grad=False,
                ).detach().cpu()
                decoded = tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )
                candidates.append(
                    {
                        "group_index": group_index,
                        "completion_ids": completion_ids,
                        "candidate": decoded,
                        "rollout_token_logprobs": [
                            float(value) for value in old_logprobs.tolist()
                        ],
                    }
                )

            def score_one(item: dict[str, Any]) -> dict[str, Any]:
                return score_dart_candidate(
                    item["candidate"],
                    record.feedback_tests,
                    f"{record.task_id}-{optimizer_step}-{item['group_index']}",
                    timeout=args.reward_timeout,
                    stability_runs=args.reward_stability_runs,
                )

            with ThreadPoolExecutor(
                max_workers=min(args.reward_workers, len(candidates))
            ) as pool:
                details = list(pool.map(score_one, candidates))
            local_rewards = verpo_local_rewards(
                details,
                alpha=args.verpo_alpha,
            )
            group_ordinal = schedule_position + 1
            judge_scores: list[float | None] = [None] * len(candidates)
            teacher_signals = [0.0] * len(candidates)
            teacher_advantages, teacher_mask = sparse_teacher_advantages(
                group_size=len(candidates),
                selected_indices=[],
                scores=[],
            )
            selected_indices: list[int] = []
            teacher_due = (
                args.judge_mode != "off"
                and should_query_group_teacher(
                    details,
                    group_ordinal=group_ordinal,
                    interval=args.judge_interval,
                )
            )
            if teacher_due:
                selected_indices = select_group_teacher_candidates(
                    details,
                    local_rewards,
                    top_n=args.judge_group_top_n,
                )
                group_payload = group_judge_payload_from_rollout(
                    source=record.teacher_source,
                    feedback_tests=record.feedback_tests,
                    candidates=candidates,
                    details=details,
                    selected_indices=selected_indices,
                )
                escalation = {
                    "task_id": record.task_id,
                    "group_ordinal": group_ordinal,
                    "policy_version": policy_version,
                    "run_contract_sha256": run_contract_hash,
                    "payload": group_payload,
                }
                if args.judge_mode == "offline_queue":
                    enqueue_teacher_escalation(
                        args.judge_escalation_queue,
                        {
                            **escalation,
                            "reason": "offline_queue",
                        },
                    )
                elif args.judge_mode == "sparse_inline":
                    try:
                        selected_scores = judge.score_group(group_payload)
                        for selected_index, score in zip(
                            selected_indices,
                            selected_scores,
                            strict=True,
                        ):
                            judge_scores[selected_index] = score
                            teacher_signals[selected_index] = score
                        teacher_advantages, teacher_mask = (
                            sparse_teacher_advantages(
                                group_size=len(candidates),
                                selected_indices=selected_indices,
                                scores=selected_scores,
                            )
                        )
                        total_eligible_judgements += len(selected_scores)
                    except Exception as teacher_error:
                        teacher_advantages, teacher_mask = (
                            sparse_teacher_advantages(
                                group_size=len(candidates),
                                selected_indices=[],
                                scores=[],
                            )
                        )
                        enqueue_teacher_escalation(
                            args.judge_escalation_queue,
                            {
                                **escalation,
                                "reason": (
                                    f"{type(teacher_error).__name__}: "
                                    f"{teacher_error}"
                                )[:1000],
                            },
                        )
            selected_index_set = set(selected_indices)
            selected_for_teacher = [
                index in selected_index_set
                for index in range(len(candidates))
            ]
            advantage_components = verpo_unified_advantages(
                details,
                local_rewards,
                teacher_signals,
                beta=args.verpo_beta,
                teacher_weight=args.judge_weight,
                teacher_mask=teacher_mask,
            )
            if (
                advantage_components["teacher_advantages"]
                != teacher_advantages
            ):
                raise AssertionError("sparse teacher advantages changed")
            for (
                candidate,
                detail,
                local_reward,
                judge_score,
                teacher_signal,
                teacher_observed,
                was_selected_for_teacher,
                global_reward,
                global_advantage,
                local_advantage,
                teacher_advantage,
                advantage,
            ) in zip(
                candidates,
                details,
                local_rewards,
                judge_scores,
                [
                    (
                        teacher_signals[index]
                        if teacher_mask[index]
                        else None
                    )
                    for index in range(len(candidates))
                ],
                teacher_mask,
                selected_for_teacher,
                advantage_components["global_rewards"],
                advantage_components["global_advantages"],
                advantage_components["local_advantages"],
                advantage_components["teacher_advantages"],
                advantage_components["unified_advantages"],
            ):
                candidate.update(
                    {
                        "verifier": detail,
                        "verpo_local_reward": local_reward,
                        "verifier_global_reward": global_reward,
                        "judge_score": judge_score,
                        "teacher_signal": teacher_signal,
                        "teacher_score_observed": teacher_observed,
                        "selected_for_teacher": was_selected_for_teacher,
                        "global_advantage": global_advantage,
                        "local_advantage": local_advantage,
                        "teacher_advantage": teacher_advantage,
                        "advantage": advantage,
                    }
                )
            batch_groups.append(
                {
                    "batch_index": batch_index,
                    "task_schedule_position": schedule_position,
                    "task_id": record.task_id,
                    "feedback_tests_sha256": hashlib.sha256(
                        record.feedback_tests.encode("utf-8")
                    ).hexdigest(),
                    "prompt_ids": list(record.prompt_ids),
                    "compact_ids": list(record.compact_ids),
                    "replay_target_ids": list(record.replay_target_ids),
                    "candidates": candidates,
                }
            )

        # One and only one optimizer step consumes this complete fresh batch.
        optimizer.zero_grad(set_to_none=True)
        policy_losses: list[float] = []
        max_policy_drift = 0.0
        candidate_count = sum(len(group["candidates"]) for group in batch_groups)
        for group in batch_groups:
            prefix = group["prompt_ids"] + group["compact_ids"]
            for candidate in group["candidates"]:
                current = completion_token_logprobs(
                    model,
                    prefix,
                    candidate["completion_ids"],
                    temperature=args.temperature,
                    device=device,
                    with_grad=True,
                )
                saved = torch.tensor(
                    candidate["rollout_token_logprobs"],
                    dtype=current.dtype,
                    device=current.device,
                )
                drift = float((current.detach() - saved).abs().max().cpu())
                max_policy_drift = max(max_policy_drift, drift)
                if drift > args.on_policy_logprob_tolerance:
                    raise RuntimeError(
                        "current policy differs from saved rollout logprobs before "
                        f"its update (max drift {drift:.6g})"
                    )
                loss = policy_token_loss(
                    current,
                    saved,
                    candidate["advantage"],
                    ppo_clip=args.ppo_clip,
                )
                (loss / candidate_count).backward()
                policy_losses.append(float(loss.detach().cpu()))
                del current, saved, loss

        replay_losses: list[float] = []
        if args.sft_replay_weight > 0.0:
            for group in batch_groups:
                feature = {
                    "decoder_prompt_input_ids": group["prompt_ids"],
                    "compact_input_ids": group["compact_ids"],
                    "target_input_ids": group["replay_target_ids"],
                }
                batch = {
                    key: value.to(device)
                    for key, value in replay_collator([feature]).items()
                }
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                replay_loss = output.loss
                if replay_loss is None or not torch.isfinite(replay_loss):
                    raise RuntimeError("SFT replay produced a non-finite loss")
                (
                    args.sft_replay_weight
                    * replay_loss
                    / len(batch_groups)
                ).backward()
                replay_losses.append(float(replay_loss.detach().cpu()))
                del output, replay_loss, batch

        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, args.max_grad_norm
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError("VeRPO gradient norm is non-finite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        telemetry = cumulative_judge_telemetry(
            prior_judge_telemetry, judge.telemetry()
        )
        receipt_attestation = judge.receipt_attestation_since(receipt_cursor)
        receipt_summary = validate_deepseek_receipt_attestation(
            receipt_attestation
        )
        if (
            receipt_summary["cumulative_receipt_count"]
            != int(telemetry["receipt_count"])
            or receipt_summary["cumulative_receipt_chain_sha256"]
            != telemetry["receipt_chain_sha256"]
        ):
            raise RuntimeError(
                "DeepSeek step receipts differ from cumulative telemetry"
            )
        journal = {
            "schema": JOURNAL_SCHEMA,
            "optimizer_step": optimizer_step,
            "sampled_policy_version": policy_version,
            "updates_applied_to_rollout": 1,
            "rollout_distribution": run_contract["generation"],
            "groups": batch_groups,
            "optimization": {
                "policy_loss_mean": (
                    sum(policy_losses) / len(policy_losses)
                    if policy_losses
                    else 0.0
                ),
                "sft_replay_loss_mean": (
                    sum(replay_losses) / len(replay_losses)
                    if replay_losses
                    else None
                ),
                "sft_replay_weight": args.sft_replay_weight,
                "ppo_clip": args.ppo_clip,
                "max_preupdate_rollout_logprob_drift": max_policy_drift,
                "grad_norm_before_clip": float(torch.as_tensor(grad_norm).cpu()),
            },
            "judge_telemetry": telemetry,
            "deepseek_response_receipts": receipt_attestation,
        }
        journal_attestation = write_step_journal(
            journal_attempt,
            journal=journal,
            previous_chain_sha256=current_journal_chain,
        )
        current_journal_chain = str(
            journal_attestation["cumulative_chain_sha256"]
        )
        checkpoint_for_step: Path | None = None
        if (
            optimizer_step % args.checkpoint_interval == 0
            or optimizer_step == args.max_updates
        ):
            checkpoint_for_step = save_optimizer_checkpoint(
                output_dir=output_dir,
                optimizer_step=optimizer_step,
                model=model,
                overlay=overlay,
                optimizer=optimizer,
                tokenizer=tokenizer,
                contract_path=args.contract,
                run_contract_sha256=run_contract_hash,
                base_provenance=base_provenance,
                journal_attestation=journal_attestation,
                judge_telemetry=telemetry,
                response_id_sha256s=judge.response_id_sha256s(),
                prior_response_receipts_path=prior_response_receipts_path,
                response_receipts_current_segment=(
                    judge.receipt_attestation_since(
                        prior_receipt_count
                    )["receipts"]
                ),
            )
            latest_checkpoint = checkpoint_for_step
            published_checkpoints += 1
        print(
            json.dumps(
                {
                    "optimizer_step": optimizer_step,
                    "policy_loss_mean": journal["optimization"][
                        "policy_loss_mean"
                    ],
                    "sft_replay_loss_mean": journal["optimization"][
                        "sft_replay_loss_mean"
                    ],
                    "full_passes": sum(
                        candidate["verifier"]["full_pass"]
                        for group in batch_groups
                        for candidate in group["candidates"]
                    ),
                    "compiled": sum(
                        candidate["verifier"]["compiled"]
                        for group in batch_groups
                        for candidate in group["candidates"]
                    ),
                    "checkpoint": (
                        None
                        if checkpoint_for_step is None
                        else str(checkpoint_for_step)
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if latest_checkpoint is None:
        raise RuntimeError("VeRPO completed without an optimizer checkpoint")
    publish_completed_run(
        output_dir=output_dir,
        max_updates=args.max_updates,
        checkpoint_interval=args.checkpoint_interval,
        latest_checkpoint=latest_checkpoint,
        run_contract_sha256=run_contract_hash,
        judge_telemetry=cumulative_judge_telemetry(
            prior_judge_telemetry, judge.telemetry()
        ),
        published_checkpoints_this_process=published_checkpoints,
        finalize_only_recovery=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"direct-compact VeRPO failed closed: {error}", file=sys.stderr)
        raise
