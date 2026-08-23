#!/usr/bin/env python3
"""Discardable, pure-reward VeRPO pilot for the sealed typed Arm-C2 policy.

This profile deliberately leaves the production VeRPO trainer untouched.  It
adapts that audited engine to the opaque typed ``fn0`` input view, admits only
the final Arm-C2 checkpoint, and restricts fresh on-policy groups to the exact
150-task proxy-audit sample.  The proxy candidates are used only to identify
the task cohort; no stored candidate, action, or reward enters an update.

The objective is frozen to independently centered full-pass, density-weighted
visible partial-test, and 0.25 compile rewards.  There is no SFT replay term.
At update 16 a predeclared mechanics gate is evaluated before update 17 is
sampled and the process pauses even on GO. Explicit continuation restores the
same checkpoint, optimizer, and RNG under the immutable 150-update contract.
Later 16-update windows are gated automatically. This pilot can never promote
a checkpoint automatically and never reads the private holdback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.evaluation import durable_evaluation_journal as journal
from scripts.evaluation import t5gemma2_typed_fold_gold_replay_inference_v1 as c2_guard
from scripts.training import t5gemma2_compiler_feedback_verpo as engine
from scripts.training import t5gemma2_typed_contract_sft as typed_sft
from scripts.training import t5gemma2_typed_fold_gold_replay_v1 as arm_c


RUN_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-checkpoint-v1"
ROLLOUT_SCHEMA = "t5gemma2-typed-c2-verpo-pilot150-rollout-v1"
GATE_SCHEMA = "t5gemma2-typed-c2-verpo-window-gate-v1"
SELECTION_SCHEMA = "t5gemma2-typed-c2-verpo-proxy-selection-v1"
TASK_VIEW_SCHEMA = "t5gemma2-typed-c2-verpo-task-view-v1"
TASK_VIEW_MANIFEST_SCHEMA = "t5gemma2-typed-c2-verpo-task-view-manifest-v1"

EXPECTED_PROXY_SUMMARY_SHA256 = (
    "b0d73acb0391adea3844afa6f36589f4035e5fa4e73751f25836b318f43d9435"
)
EXPECTED_PROXY_JOURNAL_SHA256 = (
    "b63250b1db1ca53fdf033cd3935824b4a96a76c37ef4f1f390dabd72370be1f4"
)
EXPECTED_PROXY_CHAIN_HEAD_SHA256 = (
    "76b4bcc98ef7f16fd57a76d0501a7c91617c9ac80d1506b2beeb8763a2ab8172"
)
EXPECTED_PROXY_CONTRACT_SHA256 = (
    "390a30c004b27c3ada39e38b8b92252128a4715e8556346bab802e809331ee43"
)
EXPECTED_PROXY_TASK_IDS_SHA256 = (
    "a2ae70fc0b9a79a36e393045e05e4f271f50d971cbcecba3a437a513e326e6e9"
)
EXPECTED_PROXY_TERMINALS_SHA256 = (
    "0cea6eeecfb696a60e3d55691adf6e3473c63b103ab27d5102dd154680f51887"
)
EXPECTED_PROXY_TASKS = 150
EXPECTED_GATE_UPDATE = 16
EXPECTED_MAX_UPDATES = 150
EXPECTED_GATE_INTERVAL = 16
EXPECTED_EOS_TOKEN_ID = 1
PREREGISTRATION_SHA256 = (
    "bc515b7dd7efb4d2458da3af407028eca572cf2a7a1af6616e0a8f8797c134a9"
)
PREREGISTRATION_SEAL_SHA256 = (
    "5bfbe6359d0b84ecabe43542eaebadd77bde0f9e959a210682ebcfa453445c80"
)
ZERO_EPSILON = 1e-12
LOCAL_RESIDUAL_TOLERANCE_SCALE = 1e-10

_BASE_LOAD_TASKS = engine.load_verpo_tasks
_BASE_VALIDATE_BOUNDARY = engine._validate_feedback_boundary  # noqa: SLF001
_BASE_RUNTIME_PROVENANCE = engine.runtime_provenance
_BASE_RUN_CONTRACT = engine._run_contract  # noqa: SLF001
_BASE_GENERATE_GROUP = engine.generate_group
_BASE_SCORE_CANDIDATE = engine.score_dart_candidate
_BASE_TRAJECTORY_RECORD = engine._trajectory_record  # noqa: SLF001


@dataclass(frozen=True)
class SelectedTask:
    task_id: str
    source_sha256: str
    typed_contract_sha256: str


@dataclass(frozen=True)
class ProxySelection:
    tasks: tuple[SelectedTask, ...]
    summary_sha256: str
    journal_sha256: str
    chain_head_sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class TypedTaskView:
    tasks: tuple[engine.VerpoTask, ...]
    manifest: dict[str, Any]
    manifest_sha256: str
    view_sha256: str


class _MechanicsGateStop(RuntimeError):
    """Internal control flow used after any sealed window STOP decision."""


class _MandatoryGatePause(RuntimeError):
    """Normal phase boundary after the first sealed gate returns GO."""


_ACTIVE_TASK_VIEW: TypedTaskView | None = None
_BASE_GROUPS_STARTED = 0
_GATE_OUTPUT_DIR: Path | None = None
_CONTINUE_AFTER_GATE16 = False


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_digest(path: Path, expected: str, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return observed


def load_proxy_selection(
    summary_path: str | Path,
    journal_path: str | Path,
    chain_head_path: str | Path,
) -> ProxySelection:
    """Load only task identities and typed-source seals from the proxy audit."""

    summary_file = Path(summary_path).expanduser().resolve()
    events_file = Path(journal_path).expanduser().resolve()
    head_file = Path(chain_head_path).expanduser().resolve()
    summary_sha = _require_digest(
        summary_file, EXPECTED_PROXY_SUMMARY_SHA256, "proxy summary"
    )
    events_sha = _require_digest(
        events_file, EXPECTED_PROXY_JOURNAL_SHA256, "proxy journal"
    )
    head_sha = _require_digest(
        head_file, EXPECTED_PROXY_CHAIN_HEAD_SHA256, "proxy chain head"
    )
    summary = _read_object(summary_file, "proxy summary")
    selection = summary.get("selection")
    decision = summary.get("decision")
    source_journal = summary.get("journal")
    if (
        summary.get("schema")
        != "t5gemma2-typed-proxy-reward-audit-summary-v1"
        or summary.get("status") != "complete"
        or summary.get("contract_sha256") != EXPECTED_PROXY_CONTRACT_SHA256
        or not isinstance(selection, Mapping)
        or selection.get("sample_size") != EXPECTED_PROXY_TASKS
        or selection.get("seed") != 42
        or selection.get("ordered_task_ids_sha256")
        != EXPECTED_PROXY_TASK_IDS_SHA256
        or not isinstance(decision, Mapping)
        or decision.get("overall_decision") != "GO"
        or not isinstance(source_journal, Mapping)
        or source_journal.get("sha256") != events_sha
        or source_journal.get("chain_head_sha256") != head_sha
        or source_journal.get("event_count") != EXPECTED_PROXY_TASKS + 2
    ):
        raise ValueError("proxy reward-audit summary contract differs")

    events = journal.load_journal(events_file)
    if len(events) != EXPECTED_PROXY_TASKS + 2:
        raise ValueError("proxy reward-audit journal event count differs")
    header = events[0]
    contract = header.get("contract")
    if (
        header.get("schema")
        != "t5gemma2-typed-proxy-reward-audit-journal-v1"
        or header.get("event") != "header"
        or not isinstance(contract, Mapping)
        or header.get("contract_sha256") != EXPECTED_PROXY_CONTRACT_SHA256
        or journal.canonical_sha256(contract) != EXPECTED_PROXY_CONTRACT_SHA256
        or contract.get("selection") != selection
    ):
        raise ValueError("proxy reward-audit journal header differs")

    tasks: list[SelectedTask] = []
    for position, event in enumerate(events[1:-1]):
        task_id = str(event.get("task_id") or "")
        source_sha = str(event.get("source_sha256") or "")
        contract_sha = str(event.get("typed_contract_sha256") or "")
        if (
            event.get("schema")
            != "t5gemma2-typed-proxy-reward-audit-journal-v1"
            or event.get("event") != "task_terminal"
            or event.get("task_position") != position
            or not task_id
            or len(source_sha) != 64
            or len(contract_sha) != 64
            or any(
                key in event
                for key in (
                    "acceptance_tests",
                    "holdback_tests",
                    "reward_holdback_tests",
                    "private_tests",
                    "candidate",
                    "candidate_code",
                )
            )
        ):
            raise ValueError(f"proxy selection terminal {position} differs")
        tasks.append(SelectedTask(task_id, source_sha, contract_sha))
    if len({task.task_id for task in tasks}) != EXPECTED_PROXY_TASKS:
        raise ValueError("proxy selection task IDs are not unique")
    if len({task.source_sha256 for task in tasks}) != EXPECTED_PROXY_TASKS:
        raise ValueError("proxy selection typed sources are not unique")
    ordered_ids = [task.task_id for task in tasks]
    complete = events[-1]
    if (
        complete.get("schema")
        != "t5gemma2-typed-proxy-reward-audit-journal-v1"
        or complete.get("event") != "complete"
        or complete.get("tasks") != EXPECTED_PROXY_TASKS
        or complete.get("terminal_task_ids_sha256")
        != EXPECTED_PROXY_TASK_IDS_SHA256
        or complete.get("terminal_results_sha256")
        != EXPECTED_PROXY_TERMINALS_SHA256
        or journal.canonical_sha256(ordered_ids)
        != EXPECTED_PROXY_TASK_IDS_SHA256
    ):
        raise ValueError("proxy reward-audit completion differs")
    return ProxySelection(
        tasks=tuple(tasks),
        summary_sha256=summary_sha,
        journal_sha256=events_sha,
        chain_head_sha256=head_sha,
        contract_sha256=EXPECTED_PROXY_CONTRACT_SHA256,
    )


def load_task_view(
    task_view_path: str | Path,
    manifest_path: str | Path,
) -> TypedTaskView:
    """Validate the CPU-built, target-free task view consumed by the trainer."""

    view_path = Path(task_view_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_object(manifest_file, "typed task-view manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    view_record = manifest.get("task_view")
    selection = manifest.get("selection")
    privacy = manifest.get("privacy")
    runtime = manifest.get("runtime")
    source_boundary = manifest.get("source_boundary")
    project_root = Path(__file__).resolve().parents[2]
    builder_path = (
        project_root
        / "scripts/preprocessing/build_t5gemma2_typed_c2_verpo_task_view.py"
    )
    if (
        manifest.get("schema") != TASK_VIEW_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("rows") != EXPECTED_PROXY_TASKS
        or manifest.get("manifest_sha256") != typed_sft.engine.canonical_sha256(body)
        or not isinstance(view_record, Mapping)
        or Path(str(view_record.get("path") or "")).resolve() != view_path
        or view_record.get("rows") != EXPECTED_PROXY_TASKS
        or view_record.get("ordered_task_ids_sha256")
        != EXPECTED_PROXY_TASK_IDS_SHA256
        or not isinstance(selection, Mapping)
        or selection.get("ordered_task_ids_sha256")
        != EXPECTED_PROXY_TASK_IDS_SHA256
        or selection.get("prior_candidates_actions_logprobs_rewards_reused")
        is not False
        or selection.get("proxy_summary_sha256")
        != EXPECTED_PROXY_SUMMARY_SHA256
        or selection.get("proxy_journal_sha256")
        != EXPECTED_PROXY_JOURNAL_SHA256
        or selection.get("proxy_chain_head_sha256")
        != EXPECTED_PROXY_CHAIN_HEAD_SHA256
        or selection.get("proxy_contract_sha256")
        != EXPECTED_PROXY_CONTRACT_SHA256
        or not isinstance(privacy, Mapping)
        or privacy.get("gold_or_target_in_task_view") is not False
        or privacy.get("acceptance_tests_in_task_view") is not False
        or privacy.get("private_holdback_in_task_view") is not False
        or privacy.get("visible_train_feedback_tests_in_task_view") is not True
        or not isinstance(source_boundary, Mapping)
        or source_boundary.get("rows") != 2386
        or source_boundary.get("validated") is not True
        or not isinstance(runtime, Mapping)
        or runtime.get("builder_sha256") != _sha256_file(builder_path)
        or runtime.get("typed_source_builder_sha256")
        != _sha256_file(Path(typed_sft.__file__).resolve())
        or runtime.get("pilot_profile_sha256")
        != _sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("typed task-view manifest contract differs")
    view_sha = _sha256_file(view_path)
    if view_sha != str(view_record.get("sha256") or ""):
        raise ValueError("typed task-view SHA-256 differs")
    rows = engine._read_jsonl(view_path, "typed task view")  # noqa: SLF001
    if len(rows) != EXPECTED_PROXY_TASKS:
        raise ValueError("typed task-view row count differs")
    tasks: list[engine.VerpoTask] = []
    for position, row in enumerate(rows):
        task_id = str(row.get("task_id") or "")
        source = row.get("source")
        tests = row.get("feedback_tests")
        if (
            row.get("schema") != TASK_VIEW_SCHEMA
            or row.get("position") != position
            or not task_id
            or not isinstance(source, str)
            or not source.strip()
            or engine.sha256_text(source) != row.get("source_sha256")
            or len(str(row.get("typed_contract_sha256") or "")) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(row.get("typed_contract_sha256") or "")
            )
            or not isinstance(tests, str)
            or not tests.strip()
            or engine.sha256_text(tests) != row.get("feedback_tests_sha256")
            or row.get("model_visible_fields")
            != ["opaque_typed_contract", "F2.text"]
            or row.get("target_or_gold_present") is not False
            or row.get("private_holdback_present") is not False
            or any(
                key in row
                for key in (
                    "dart_source",
                    "target",
                    "supervised_target",
                    "acceptance_tests",
                    "holdback_tests",
                    "reward_holdback_tests",
                    "private_tests",
                )
            )
        ):
            raise ValueError(f"typed task-view row {position} differs")
        engine.split_visible_expect_harnesses(tests)
        tasks.append(
            engine.VerpoTask(
                task_id=task_id,
                source=source,
                source_sha256=str(row["source_sha256"]),
                feedback_tests=tests,
                replay_target="",
            )
        )
    if (
        len({task.task_id for task in tasks}) != EXPECTED_PROXY_TASKS
        or len({task.source_sha256 for task in tasks}) != EXPECTED_PROXY_TASKS
        or typed_sft.engine.canonical_sha256([task.task_id for task in tasks])
        != EXPECTED_PROXY_TASK_IDS_SHA256
        or typed_sft.engine.canonical_sha256(
            [task.source_sha256 for task in tasks]
        )
        != view_record.get("ordered_source_sha256s_sha256")
        or typed_sft.engine.canonical_sha256(
            [str(row["typed_contract_sha256"]) for row in rows]
        )
        != view_record.get("ordered_contract_sha256s_sha256")
    ):
        raise ValueError("typed task-view identity seal differs")
    return TypedTaskView(
        tasks=tuple(tasks),
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_file),
        view_sha256=view_sha,
    )


def load_typed_selected_tasks(
    rollout_path: str | Path,
    f2_path: str | Path,
) -> list[engine.VerpoTask]:
    del rollout_path, f2_path
    if _ACTIVE_TASK_VIEW is None:
        raise RuntimeError("typed C2 VeRPO task view is not installed")
    return list(_ACTIVE_TASK_VIEW.tasks)


def _profile_boundary(args: argparse.Namespace) -> dict[str, Any]:
    del args
    if _ACTIVE_TASK_VIEW is None:
        raise RuntimeError("typed C2 VeRPO task view is not installed")
    source_boundary = _ACTIVE_TASK_VIEW.manifest.get("source_boundary")
    if (
        not isinstance(source_boundary, Mapping)
        or source_boundary.get("rows") != 2386
        or source_boundary.get("validated") is not True
    ):
        raise ValueError("typed task-view source boundary differs")
    return {
        "schema": TASK_VIEW_MANIFEST_SCHEMA,
        "rows": EXPECTED_PROXY_TASKS,
        "full_boundary_rows": 2386,
        "task_view_sha256": _ACTIVE_TASK_VIEW.view_sha256,
        "task_view_manifest_sha256": _ACTIVE_TASK_VIEW.manifest_sha256,
        "selection": dict(_ACTIVE_TASK_VIEW.manifest["selection"]),
        "trainer_opened_gold_or_target_source": False,
        "private_holdback_exposed": False,
    }


def _profile_runtime_provenance() -> dict[str, Any]:
    record = dict(_BASE_RUNTIME_PROVENANCE())
    project_root = Path(__file__).resolve().parents[2]
    extra = {
        "typed_c2_pilot_profile": Path(
            "scripts/training/t5gemma2_typed_c2_verpo_pilot150.py"
        ),
        "typed_task_view_builder": Path(
            "scripts/preprocessing/build_t5gemma2_typed_c2_verpo_task_view.py"
        ),
        "typed_source_builder": Path(
            "scripts/training/t5gemma2_typed_contract_sft.py"
        ),
        "arm_c2_contract": Path(
            "scripts/training/t5gemma2_typed_fold_gold_replay_v1.py"
        ),
        "arm_c2_guard": Path(
            "scripts/evaluation/t5gemma2_typed_fold_gold_replay_inference_v1.py"
        ),
        "durable_journal": Path(
            "scripts/evaluation/durable_evaluation_journal.py"
        ),
        "preregistration": Path(
            "../analysis_rs_sft_fold/T5GEMMA2_TYPED_C2_VERPO_150_PREREGISTRATION.md"
        ),
        "preregistration_seal": Path(
            "../analysis_rs_sft_fold/T5GEMMA2_TYPED_C2_VERPO_150_PREREGISTRATION.seal.json"
        ),
    }
    code = dict(record["code"])
    for name, relative in extra.items():
        path = (project_root / relative).resolve()
        code[name] = {
            "relative_path": relative.as_posix(),
            "sha256": _sha256_file(path),
        }
    if (
        code["preregistration"]["sha256"] != PREREGISTRATION_SHA256
        or code["preregistration_seal"]["sha256"]
        != PREREGISTRATION_SEAL_SHA256
    ):
        raise ValueError("typed C2 VeRPO preregistration seal differs")
    record["code"] = code
    record["code_bundle_sha256"] = typed_sft.engine.canonical_sha256(code)
    return record


def _profile_run_contract(**kwargs: Any) -> dict[str, Any]:
    if _ACTIVE_TASK_VIEW is None:
        raise RuntimeError("typed C2 VeRPO task view is not installed")
    contract = dict(_BASE_RUN_CONTRACT(**kwargs))
    warm_contract = kwargs["warm_contract"]
    warmstart = Path(kwargs["args"].warmstart_checkpoint).expanduser().resolve()
    # Keep the shared typed checkpoint loader's architecture contract while
    # recording the policy engine's more specific description separately.
    contract["architecture"] = "native_encoder_decoder"
    contract["policy_architecture"] = "native_t5gemma2_encoder_decoder"
    contract["status"] = "training"
    contract["base_model"] = dict(warm_contract["base_model"])
    contract["lora"] = dict(warm_contract["lora"])
    if contract.get("sampling", {}).get("eos_token_ids") != [EXPECTED_EOS_TOKEN_ID]:
        raise ValueError("typed C2 VeRPO EOS token contract differs")
    contract["input_view"] = {
        "view": "opaque_typed_contract_plus_compressed_enriched_F2",
        "function_name": "fn0",
        "parameter_name_policy": "p{zero_based_index}",
        "semantic_names_visible": False,
        "typed_prompt_source_sha256s": typed_sft.engine.canonical_sha256(
            [task.source_sha256 for task in kwargs["tasks"]]
        ),
    }
    contract["selection"] = {
        "schema": SELECTION_SCHEMA,
        "source": "sealed_target_free_typed_task_view",
        "tasks": EXPECTED_PROXY_TASKS,
        "ordered_task_ids_sha256": EXPECTED_PROXY_TASK_IDS_SHA256,
        "task_view_sha256": _ACTIVE_TASK_VIEW.view_sha256,
        "task_view_manifest_sha256": _ACTIVE_TASK_VIEW.manifest_sha256,
        "proxy_summary_sha256": _ACTIVE_TASK_VIEW.manifest["selection"][
            "proxy_summary_sha256"
        ],
        "proxy_journal_sha256": _ACTIVE_TASK_VIEW.manifest["selection"][
            "proxy_journal_sha256"
        ],
        "proxy_chain_head_sha256": _ACTIVE_TASK_VIEW.manifest["selection"][
            "proxy_chain_head_sha256"
        ],
        "proxy_contract_sha256": _ACTIVE_TASK_VIEW.manifest["selection"][
            "proxy_contract_sha256"
        ],
        "stored_candidates_actions_logprobs_rewards_reused": False,
        "trainer_opened_gold_or_target_source": False,
    }
    contract["warmstart"]["checkpoint_files"] = {
        "run_contract_sha256": _sha256_file(warmstart / "run_contract.json"),
        "adapter_weights_sha256": _sha256_file(
            warmstart / "adapter" / "adapter_model.safetensors"
        ),
        "adapter_config_sha256": _sha256_file(
            warmstart / "adapter" / "adapter_config.json"
        ),
        "tokenizer_sha256": _sha256_file(
            warmstart / "tokenizer" / "tokenizer.json"
        ),
    }
    contract["optimization"]["objective_profile"] = "pure_execution_reward"
    contract["optimization"]["gold_or_sft_replay_gradient"] = False
    contract["repair"] = {
        **contract["repair"],
        "eligibility": (
            "all_zero_visible_reward_group_and_candidate_only_neutral_main_"
            "compile_failure"
        ),
        "feedback": "sanitized_candidate_only_neutral_main_dart_compiler",
        "visible_test_diagnostic_used": False,
        "candidate_only_neutral_main": "void main() {}",
    }
    contract["on_policy_invariants"] = {
        "fresh_groups_from_current_c2_policy": True,
        "prior_proxy_candidates_reused": False,
        "decoder_action_ids_captured_at_rollout": True,
        "old_policy_token_logprobs_captured_before_update": True,
        "same_conditioning_temperature_and_sampling_support_recomputed": True,
        "drift_checked_before_optimizer_step": True,
        "ppo_clip": 0.0,
        "importance_ratio_objective_used": False,
        "maximum_absolute_logprob_drift": kwargs[
            "args"
        ].on_policy_logprob_tolerance,
    }
    contract["pilot"] = {
        "disposition": "discardable_mechanics_pilot_not_a_promotion_arm",
        "maximum_updates": EXPECTED_MAX_UPDATES,
        "gate_update": EXPECTED_GATE_UPDATE,
        "gate_evaluated_before_sampling_update": EXPECTED_GATE_UPDATE + 1,
        "gate_interval": EXPECTED_GATE_INTERVAL,
        "mandatory_pause_after_first_gate": True,
        "later_gates_automatic": True,
        "gate": {
            "minimum_base_unified_active_groups": 8,
            "minimum_base_local_noncollinear_groups": 4,
            "maximum_zero_policy_updates": 8,
            "all_integrity_invariants_required": True,
        },
        "automatic_promotion_permitted": False,
        "private_holdback_read": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_seal_sha256": PREREGISTRATION_SEAL_SHA256,
    }
    contract["automatic_promotion_permitted"] = False
    contract["production_floor_eligible"] = False
    contract["private_holdback_exposed"] = False
    lengths = []
    for task in kwargs["tasks"]:
        encoded = kwargs["tokenizer"](
            task.source,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        lengths.append(len(encoded))
    if not lengths or max(lengths) > kwargs["args"].max_source_tokens:
        raise ValueError("typed task view contains an over-cap encoder source")
    contract["source_token_preflight"] = {
        "rows": len(lengths),
        "maximum": max(lengths),
        "minimum": min(lengths),
        "max_source_tokens": kwargs["args"].max_source_tokens,
        "truncated_rows": 0,
        "token_lengths_sha256": typed_sft.engine.canonical_sha256(lengths),
        "tokenizer_add_special_tokens": True,
        "tokenizer_truncation": False,
    }
    return contract


def _orthogonal_residual_squared(
    value: Sequence[float], spans: Sequence[Sequence[float]]
) -> tuple[float, float]:
    vector = [float(item) for item in value]
    basis: list[list[float]] = []
    for raw in spans:
        candidate = [float(item) for item in raw]
        for unit in basis:
            coefficient = sum(a * b for a, b in zip(candidate, unit, strict=True))
            candidate = [
                a - coefficient * b
                for a, b in zip(candidate, unit, strict=True)
            ]
        norm = math.sqrt(sum(item * item for item in candidate))
        if norm > ZERO_EPSILON:
            basis.append([item / norm for item in candidate])
    residual = list(vector)
    for unit in basis:
        coefficient = sum(a * b for a, b in zip(residual, unit, strict=True))
        residual = [
            a - coefficient * b
            for a, b in zip(residual, unit, strict=True)
        ]
    return sum(item * item for item in residual), sum(item * item for item in vector)


def score_dart_candidate_neutral_repair(
    candidate: str,
    feedback_tests: str,
    task_id: str,
    *,
    timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    """Score reward on visible tests but source repair diagnostics neutrally."""

    detail = _BASE_SCORE_CANDIDATE(
        candidate,
        feedback_tests,
        task_id,
        timeout=timeout,
        stability_runs=stability_runs,
    )
    neutral_compiled, _neutral_pass, neutral_diagnostic, _ = (
        engine.evaluate_dart_jit_tests_detail(
            candidate,
            "void main() {}",
            f"{task_id}-candidate-only-neutral-main",
            timeout=timeout,
            stability_runs=stability_runs,
        )
    )
    detail["diagnostic"] = str(neutral_diagnostic or "")[:12000]
    detail["repair_compiled"] = bool(neutral_compiled)
    detail["repair_diagnostic_source"] = "candidate_only_plus_neutral_main"
    return detail


def build_neutral_repair_groups(
    *,
    model: Any,
    tokenizer: Any,
    task: engine.VerpoTask,
    base_group: Sequence[engine.RolloutTrajectory],
    max_parents: int,
    repair_group_size: int,
    max_source_tokens: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[list[engine.RolloutTrajectory]]:
    """Repair only candidates that fail candidate-only neutral compilation."""

    if not engine._group_is_zero_pass(base_group):  # noqa: SLF001
        return []
    eligible = [
        index
        for index, trajectory in enumerate(base_group)
        if trajectory.detail.get("repair_compiled") is False
        and bool(trajectory.candidate.strip())
        and trajectory.detail.get("repair_diagnostic_source")
        == "candidate_only_plus_neutral_main"
    ]
    if not eligible:
        return []
    candidates = [base_group[index].candidate for index in eligible]
    chosen_local = engine.max_min_diverse_indices(
        candidates, min(max_parents, len(candidates))
    )
    groups: list[list[engine.RolloutTrajectory]] = []
    for local_index in chosen_local:
        parent = base_group[eligible[local_index]]
        repair = engine.build_compiler_repair_context(
            task_id=task.task_id,
            source_sha256=task.source_sha256,
            candidate=parent.candidate,
            diagnostic=str(parent.detail.get("diagnostic") or ""),
            compiled=False,
        )
        groups.append(
            engine.generate_group(
                model=model,
                tokenizer=tokenizer,
                task_id=task.task_id,
                source=task.source + "\n" + str(repair["text"]),
                state_kind="compiler_repair",
                group_size=repair_group_size,
                max_source_tokens=max_source_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
                parent_candidate_sha256=engine.sha256_text(parent.candidate),
                feedback_sha256=str(
                    repair["payload"]["compiler_feedback_sha256"]
                ),
            )
        )
    return groups


def _profile_trajectory_record(
    trajectory: engine.RolloutTrajectory,
) -> dict[str, Any]:
    record = _BASE_TRAJECTORY_RECORD(trajectory)
    record["repair_compiled_candidate_only_neutral_main"] = bool(
        trajectory.detail.get("repair_compiled", False)
    )
    record["repair_diagnostic_source"] = str(
        trajectory.detail.get("repair_diagnostic_source")
        or "candidate_only_plus_neutral_main"
    )
    record["raw_diagnostic_persisted"] = False
    return record


def evaluate_mechanics_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_contract_sha256: str,
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    """Evaluate one frozen 16-update gate from public rollout metrics only."""

    if (
        len(rows) != EXPECTED_GATE_INTERVAL
        or window_end - window_start + 1 != EXPECTED_GATE_INTERVAL
        or window_start <= 0
        or window_end % EXPECTED_GATE_INTERVAL != 0
    ):
        raise ValueError("mechanics gate requires one exact 16-update window")
    unified_active = 0
    local_noncollinear = 0
    zero_policy_updates = 0
    for offset, row in enumerate(rows):
        position = window_start + offset
        task_records = row.get("task_records")
        if (
            row.get("schema") != ROLLOUT_SCHEMA
            or row.get("update") != position
            or row.get("run_contract_sha256") != run_contract_sha256
            or not isinstance(task_records, list)
            or len(task_records) != 1
            or not math.isfinite(float(row.get("policy_loss", float("nan"))))
            or not math.isfinite(float(row.get("grad_norm", float("nan"))))
            or not math.isfinite(
                float(row.get("max_on_policy_logprob_drift", float("nan")))
            )
            or float(row["max_on_policy_logprob_drift"]) > 2e-4
            or float(row.get("sft_replay_loss", float("nan"))) != 0.0
            or int(row.get("sampled_pad_before_eos", -1)) != 0
        ):
            raise ValueError(f"mechanics gate metric {position} violates integrity")
        reward = task_records[0].get("base_reward")
        trajectories = task_records[0].get("trajectories")
        if (
            not isinstance(trajectories, list)
            or len(trajectories) != int(row.get("trajectory_count", -1))
        ):
            raise ValueError(
                f"mechanics gate metric {position} trajectory count differs"
            )
        for trajectory_index, trajectory in enumerate(trajectories):
            actions = trajectory.get("actions") if isinstance(trajectory, Mapping) else None
            if (
                not isinstance(trajectory, Mapping)
                or not isinstance(actions, list)
                or not actions
                or len(actions) != int(trajectory.get("action_tokens", -1))
                or len(actions) > 8192
                or actions[-1] != EXPECTED_EOS_TOKEN_ID
                or int(trajectory.get("sampled_pad_before_eos", -1)) != 0
            ):
                raise ValueError(
                    f"mechanics gate metric {position} trajectory "
                    f"{trajectory_index} violates EOS/action invariants"
                )
        if not isinstance(reward, Mapping):
            raise ValueError(f"mechanics gate metric {position} lacks base reward")
        required = (
            "global_advantages",
            "local_advantages",
            "compile_advantages",
            "unified_advantages",
        )
        vectors: dict[str, list[float]] = {}
        for key in required:
            values = reward.get(key)
            if (
                not isinstance(values, list)
                or len(values) != 4
                or any(not math.isfinite(float(item)) for item in values)
            ):
                raise ValueError(
                    f"mechanics gate metric {position} has invalid {key}"
                )
            vectors[key] = [float(item) for item in values]
        unified_active += int(
            any(abs(value) > ZERO_EPSILON for value in vectors["unified_advantages"])
        )
        residual_squared, local_squared = _orthogonal_residual_squared(
            vectors["local_advantages"],
            [vectors["global_advantages"], vectors["compile_advantages"]],
        )
        local_noncollinear += int(
            local_squared > ZERO_EPSILON
            and residual_squared
            > LOCAL_RESIDUAL_TOLERANCE_SCALE * max(1.0, local_squared)
        )
        active = int(row.get("active_policy_trajectories", -1))
        optimizer_step = row.get("optimizer_step")
        if active < 0 or type(optimizer_step) is not bool:
            raise ValueError(f"mechanics gate metric {position} activity differs")
        if active == 0:
            if (
                float(row["policy_loss"]) != 0.0
                or float(row["grad_norm"]) != 0.0
                or optimizer_step is not False
            ):
                raise ValueError(
                    f"mechanics gate metric {position} zero-policy semantics differ"
                )
            zero_policy_updates += 1
        elif (
            optimizer_step is not True
            or float(row["grad_norm"]) <= 0.0
        ):
            raise ValueError(
                f"mechanics gate metric {position} active update is degenerate"
            )

    criteria = {
        "base_unified_active_groups": {
            "observed": unified_active,
            "minimum": 8,
            "pass": unified_active >= 8,
        },
        "base_local_noncollinear_groups": {
            "observed": local_noncollinear,
            "minimum": 4,
            "pass": local_noncollinear >= 4,
        },
        "zero_policy_updates": {
            "observed": zero_policy_updates,
            "maximum": 8,
            "pass": zero_policy_updates <= 8,
        },
        "integrity": {"pass": True},
    }
    decision = "GO" if all(value["pass"] for value in criteria.values()) else "STOP"
    body = {
        "schema": GATE_SCHEMA,
        "status": "pass",
        "window_start_update": window_start,
        "window_end_update": window_end,
        "gate_update": window_end,
        "evaluated_before_sampling_update": window_end + 1,
        "run_contract_sha256": run_contract_sha256,
        "criteria": criteria,
        "decision": decision,
        "automatic_promotion_performed": False,
        "private_holdback_read": False,
    }
    return {**body, "gate_sha256": typed_sft.engine.canonical_sha256(body)}


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank rollout metric at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"rollout metric {line_number} is not an object")
            rows.append(value)
    return rows


def _guarded_generate_group(**kwargs: Any):
    global _BASE_GROUPS_STARTED
    if kwargs.get("state_kind") == "base":
        if (
            _BASE_GROUPS_STARTED > 0
            and _BASE_GROUPS_STARTED % EXPECTED_GATE_INTERVAL == 0
            and _BASE_GROUPS_STARTED < EXPECTED_MAX_UPDATES
        ):
            if _GATE_OUTPUT_DIR is None:
                raise RuntimeError("mechanics gate output directory is absent")
            contract = _read_object(
                _GATE_OUTPUT_DIR / "run_contract.json", "pilot run contract"
            )
            contract_sha = typed_sft.engine.canonical_sha256(contract)
            metrics = _read_metrics(_GATE_OUTPUT_DIR / "rollout_metrics.jsonl")
            if len(metrics) != _BASE_GROUPS_STARTED:
                raise ValueError("checkpoint/metric count differs at window gate")
            window_end = _BASE_GROUPS_STARTED
            window_start = window_end - EXPECTED_GATE_INTERVAL + 1
            gate = evaluate_mechanics_gate(
                metrics[-EXPECTED_GATE_INTERVAL:],
                run_contract_sha256=contract_sha,
                window_start=window_start,
                window_end=window_end,
            )
            engine._atomic_json(  # noqa: SLF001
                _GATE_OUTPUT_DIR / f"pilot_gate_update{window_end:06d}.json", gate
            )
            if gate["decision"] != "GO":
                raise _MechanicsGateStop(
                    f"update-{window_end} mechanics gate returned STOP"
                )
            if window_end == EXPECTED_GATE_UPDATE and not _CONTINUE_AFTER_GATE16:
                phase = engine.bind_run_contract(
                    {
                        "schema": "t5gemma2-typed-c2-verpo-phase-status-v1",
                        "status": "awaiting_explicit_resume_after_gate16",
                        "completed_update": EXPECTED_GATE_UPDATE,
                        "latest_checkpoint": "checkpoint-optstep-000016",
                        "gate_file": "pilot_gate_update000016.json",
                        "gate_decision": "GO",
                        "automatic_promotion_performed": False,
                    },
                    contract,
                )
                engine._atomic_json(  # noqa: SLF001
                    _GATE_OUTPUT_DIR / "phase_status.json", phase
                )
                raise _MandatoryGatePause("mandatory update-16 phase boundary")
        _BASE_GROUPS_STARTED += 1
    trajectories = _BASE_GENERATE_GROUP(**kwargs)
    for index, trajectory in enumerate(trajectories):
        if (
            not trajectory.actions
            or not 1 <= len(trajectory.actions) <= 8192
            or trajectory.actions[-1] != EXPECTED_EOS_TOKEN_ID
            or trajectory.sampled_pad_before_eos != 0
        ):
            raise RuntimeError(
                f"fresh rollout {index} violates sampled EOS/action invariants"
            )
    return trajectories


def _validate_exact_profile(args: argparse.Namespace) -> None:
    expected: dict[str, Any] = {
        "group_size": 4,
        "repair_group_size": 4,
        "max_repair_parents": 2,
        "tasks_per_update": 1,
        "max_updates": EXPECTED_MAX_UPDATES,
        "temperature": 0.8,
        "max_new_tokens": 8192,
        "max_source_tokens": 32768,
        "max_target_tokens": 32768,
        "verpo_alpha": 2.0,
        "local_weight": 1.0,
        "compile_weight": 0.25,
        "learning_rate": 1e-6,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "ppo_clip": 0.0,
        "sft_replay_weight": 0.0,
        "on_policy_logprob_tolerance": 2e-4,
        "reward_workers": 4,
        "reward_timeout": 30,
        "reward_stability_runs": 1,
        "checkpoint_interval": 1,
        "keep_last_checkpoints": 2,
        "seed": 42,
        "attn_implementation": "sdpa",
        "bf16": True,
    }
    differences = {
        key: (getattr(args, key), value)
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if differences:
        raise ValueError(f"typed C2 VeRPO pilot profile differs: {differences}")


def _resume_update(args: argparse.Namespace) -> int:
    if not args.resume_checkpoint:
        return 0
    state = torch.load(
        Path(args.resume_checkpoint) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    update = int(state.get("update", -1))
    if state.get("schema") != CHECKPOINT_SCHEMA or not 1 <= update < EXPECTED_MAX_UPDATES:
        raise ValueError("typed C2 VeRPO resume state differs")
    return update


def _validate_crossed_gates(output_dir: Path, resume_update: int) -> None:
    """Recompute every gate crossed by a resumed policy before loading it."""

    contract = _read_object(output_dir / "run_contract.json", "pilot run contract")
    contract_sha = typed_sft.engine.canonical_sha256(contract)
    metrics = _read_metrics(output_dir / "rollout_metrics.jsonl")
    if len(metrics) != resume_update:
        raise ValueError("resume checkpoint/metric count differs")
    required_boundaries = [EXPECTED_GATE_UPDATE]
    required_boundaries.extend(
        boundary
        for boundary in range(32, EXPECTED_MAX_UPDATES, EXPECTED_GATE_INTERVAL)
        if boundary < resume_update
    )
    for boundary in required_boundaries:
        start = boundary - EXPECTED_GATE_INTERVAL + 1
        expected = evaluate_mechanics_gate(
            metrics[start - 1 : boundary],
            run_contract_sha256=contract_sha,
            window_start=start,
            window_end=boundary,
        )
        observed = _read_object(
            output_dir / f"pilot_gate_update{boundary:06d}.json",
            f"pilot gate update {boundary}",
        )
        if observed != expected or observed.get("decision") != "GO":
            raise ValueError(f"resume crossed an invalid gate at update {boundary}")


def train(
    args: argparse.Namespace,
    task_view: TypedTaskView,
    *,
    continue_after_gate16: bool,
) -> dict[str, Any]:
    """Install the typed-C2 profile, run the engine, and always restore globals."""

    global _ACTIVE_TASK_VIEW, _BASE_GROUPS_STARTED, _GATE_OUTPUT_DIR
    global _CONTINUE_AFTER_GATE16
    _validate_exact_profile(args)
    warm_contract = _read_object(
        Path(args.warmstart_checkpoint) / "run_contract.json",
        "typed Arm-C2 run contract",
    )
    c2_guard._require_arm_c_contract(warm_contract)  # noqa: SLF001
    original = {
        "run_schema": engine.RUN_SCHEMA,
        "checkpoint_schema": engine.CHECKPOINT_SCHEMA,
        "rollout_schema": engine.ROLLOUT_SCHEMA,
        "supported": engine.SUPPORTED_WARMSTART_SCHEMAS,
        "load_tasks": engine.load_verpo_tasks,
        "validate_boundary": engine._validate_feedback_boundary,  # noqa: SLF001
        "runtime": engine.runtime_provenance,
        "run_contract": engine._run_contract,  # noqa: SLF001
        "generate_group": engine.generate_group,
        "score_candidate": engine.score_dart_candidate,
        "build_repair_groups": engine.build_repair_groups,
        "trajectory_record": engine._trajectory_record,  # noqa: SLF001
    }
    _ACTIVE_TASK_VIEW = task_view
    resume_update = _resume_update(args)
    _BASE_GROUPS_STARTED = resume_update
    _GATE_OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
    _CONTINUE_AFTER_GATE16 = bool(continue_after_gate16)
    if continue_after_gate16:
        if resume_update < EXPECTED_GATE_UPDATE:
            raise ValueError(
                "continuation after gate16 requires checkpoint update >=16"
            )
        _validate_crossed_gates(_GATE_OUTPUT_DIR, resume_update)
    elif resume_update > EXPECTED_GATE_UPDATE:
        raise ValueError("update >=16 resume requires --continue_after_gate16")
    engine.RUN_SCHEMA = RUN_SCHEMA
    engine.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    engine.ROLLOUT_SCHEMA = ROLLOUT_SCHEMA
    engine.SUPPORTED_WARMSTART_SCHEMAS = frozenset(
        set(engine.SUPPORTED_WARMSTART_SCHEMAS) | {arm_c.RUN_SCHEMA}
    )
    engine.load_verpo_tasks = load_typed_selected_tasks
    engine._validate_feedback_boundary = _profile_boundary  # type: ignore[assignment]  # noqa: SLF001
    engine.runtime_provenance = _profile_runtime_provenance
    engine._run_contract = _profile_run_contract  # type: ignore[assignment]  # noqa: SLF001
    engine.generate_group = _guarded_generate_group
    engine.score_dart_candidate = score_dart_candidate_neutral_repair
    engine.build_repair_groups = build_neutral_repair_groups
    engine._trajectory_record = _profile_trajectory_record  # type: ignore[assignment]  # noqa: SLF001
    try:
        try:
            result = engine.train(args)
        except _MandatoryGatePause:
            return _read_object(
                _GATE_OUTPUT_DIR / "phase_status.json", "pilot phase status"
            )
        except _MechanicsGateStop:
            contract = _read_object(
                _GATE_OUTPUT_DIR / "run_contract.json", "pilot run contract"
            )
            stopped_update = _BASE_GROUPS_STARTED
            result = engine.bind_run_contract(
                {
                    "schema": RUN_SCHEMA,
                    "status": "stopped_at_window_gate",
                    "updates": stopped_update,
                    "latest_checkpoint": f"checkpoint-optstep-{stopped_update:06d}",
                    "gate_update": stopped_update,
                    "window_gate": "STOP",
                    "automatic_promotion_performed": False,
                    "production_floor_eligible": False,
                    "no_frontier_api": True,
                },
                contract,
            )
            engine._atomic_json(_GATE_OUTPUT_DIR / "result.json", result)  # noqa: SLF001
            return result
        result = {
            **result,
            "mechanics_gate": "GO",
            "window_gates_passed": list(range(16, 145, 16)),
            "automatic_promotion_performed": False,
            "production_floor_eligible": False,
            "pilot_disposition": "discardable_not_for_automatic_promotion",
        }
        engine._atomic_json(_GATE_OUTPUT_DIR / "result.json", result)  # noqa: SLF001
        return result
    finally:
        engine.RUN_SCHEMA = original["run_schema"]
        engine.CHECKPOINT_SCHEMA = original["checkpoint_schema"]
        engine.ROLLOUT_SCHEMA = original["rollout_schema"]
        engine.SUPPORTED_WARMSTART_SCHEMAS = original["supported"]
        engine.load_verpo_tasks = original["load_tasks"]
        engine._validate_feedback_boundary = original["validate_boundary"]  # type: ignore[assignment]  # noqa: SLF001
        engine.runtime_provenance = original["runtime"]
        engine._run_contract = original["run_contract"]  # type: ignore[assignment]  # noqa: SLF001
        engine.generate_group = original["generate_group"]
        engine.score_dart_candidate = original["score_candidate"]
        engine.build_repair_groups = original["build_repair_groups"]
        engine._trajectory_record = original["trajectory_record"]  # type: ignore[assignment]  # noqa: SLF001
        _ACTIVE_TASK_VIEW = None
        _BASE_GROUPS_STARTED = 0
        _GATE_OUTPUT_DIR = None
        _CONTINUE_AFTER_GATE16 = False


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, TypedTaskView, bool]:
    profile = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    profile.add_argument("--typed_task_view", required=True)
    profile.add_argument("--typed_task_view_manifest", required=True)
    profile.add_argument("--continue_after_gate16", action="store_true")
    custom, remaining = profile.parse_known_args(argv)
    args = engine.parse_args(remaining)
    task_view = load_task_view(
        custom.typed_task_view,
        custom.typed_task_view_manifest,
    )
    return args, task_view, bool(custom.continue_after_gate16)


def main(argv: Sequence[str] | None = None) -> int:
    args, task_view, continue_after_gate16 = parse_args(argv)
    result = train(
        args,
        task_view,
        continue_after_gate16=continue_after_gate16,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
