#!/usr/bin/env python3
"""Native T5Gemma 2 VeRPO with an entirely local Dart compiler repair loop.

One optimizer update contains two on-policy states sampled from the same frozen
policy version:

1. ordinary decompilation rollouts from the sealed enriched F2 source; and
2. for all-zero groups, repair rollouts whose encoder input additionally
   contains one failed candidate and its sanitized *compiler* diagnostic.

Repair rewards update only the repair-conditioned trajectories.  They are
never transferred to the original prompt.  Global pass, partial visible-test
reward, and binary compile reward are independently mean-centered.  No API
teacher, LLM judge, acceptance test, or private holdback enters this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from scripts.evaluation.graph_compile_at_k_antigravity import (
    evaluate_dart_jit_tests_detail,
    validate_dart_binary,
)
from scripts.preprocessing.build_verpo_feedback_view import (
    extract_expect_spans,
    harness_with_cases,
    validate_feedback_training_boundary,
)
from scripts.training.seq2seq_verpo_core import (
    build_compiler_repair_context,
    max_min_diverse_indices,
    normalize_generated_seq2seq_ids,
    seq2seq_completion_token_logprobs,
    sha256_text,
    validate_on_policy_logprob_drift,
    verpo_execution_compile_advantages,
)
from scripts.training.t5gemma2_enriched_sft import (
    DEFAULT_MODEL,
    RUN_SCHEMA as SFT_RUN_SCHEMA,
    build_encoder_source,
    canonical_sha256,
)


RUN_SCHEMA = "t5gemma2-compiler-feedback-verpo-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-compiler-feedback-verpo-checkpoint-v1"
ROLLOUT_SCHEMA = "t5gemma2-compiler-feedback-verpo-rollout-v1"
TASK_SCHEDULE_SCHEMA = "t5gemma2-compiler-feedback-verpo-schedule-v1"
RUNTIME_PROVENANCE_SCHEMA = "t5gemma2-compiler-feedback-verpo-runtime-provenance-v1"
MIXED_RS_SFT_RUN_SCHEMA = "t5gemma2-mixed-rs-sft-run-v1"
SUPPORTED_WARMSTART_SCHEMAS = frozenset({SFT_RUN_SCHEMA, MIXED_RS_SFT_RUN_SCHEMA})
_CHECKPOINT_RE = re.compile(r"^checkpoint-optstep-(\d{6})$")
_FORBIDDEN_ROLLOUT_TEST_FIELDS = frozenset(
    {
        "tests",
        "acceptance_tests",
        "hidden_tests",
        "holdback_tests",
        "reward_holdback_tests",
    }
)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _read_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{label}: blank row {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label}: row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label}: no rows")
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance() -> dict[str, Any]:
    """Bind the exact objective/evaluator sources and CUDA runtime."""

    project_root = Path(__file__).resolve().parents[2]
    relative_modules = {
        "trainer": Path("scripts/training/t5gemma2_compiler_feedback_verpo.py"),
        "seq2seq_core": Path("scripts/training/seq2seq_verpo_core.py"),
        "dart_evaluator": Path("scripts/evaluation/graph_compile_at_k_antigravity.py"),
        "feedback_boundary_builder": Path(
            "scripts/preprocessing/build_verpo_feedback_view.py"
        ),
        "enriched_sft_helper": Path("scripts/training/t5gemma2_enriched_sft.py"),
    }
    code: dict[str, dict[str, str]] = {}
    for name, relative in relative_modules.items():
        path = (project_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VeRPO runtime source is missing: {relative}")
        code[name] = {
            "relative_path": relative.as_posix(),
            "sha256": _sha256_file(path),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_index: int | None = None
    device_name: str | None = None
    compute_capability: list[int] | None = None
    total_memory_bytes: int | None = None
    multiprocessor_count: int | None = None
    if cuda_available:
        device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device_index)
        device_name = str(properties.name)
        compute_capability = [int(properties.major), int(properties.minor)]
        total_memory_bytes = int(properties.total_memory)
        multiprocessor_count = int(properties.multi_processor_count)

    cudnn_version = torch.backends.cudnn.version()
    return {
        "schema": RUNTIME_PROVENANCE_SCHEMA,
        "code": code,
        "code_bundle_sha256": canonical_sha256(code),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": {
            "version": str(torch.__version__),
            "git_version": str(getattr(torch.version, "git_version", "") or ""),
        },
        "cuda": {
            "available": cuda_available,
            "build_version": str(torch.version.cuda or ""),
            "cudnn_version": (
                int(cudnn_version) if cudnn_version is not None else None
            ),
            "visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
            "device_count": int(torch.cuda.device_count()),
            "current_device": device_index,
            "device_name": device_name,
            "compute_capability": compute_capability,
            "total_memory_bytes": total_memory_bytes,
            "multiprocessor_count": multiprocessor_count,
        },
    }


def bind_run_contract(
    value: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one artifact record bound to the canonical run contract."""

    if "run_contract_sha256" in value:
        raise ValueError("artifact record already contains a run-contract binding")
    result = dict(value)
    result["run_contract_sha256"] = canonical_sha256(run_contract)
    return result


def deterministic_task_schedule(
    task_ids: Sequence[str],
    *,
    seed: int,
    groups: int,
) -> list[int]:
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("VeRPO schedule requires unique nonempty task IDs")
    if groups <= 0:
        raise ValueError("VeRPO schedule group count must be positive")
    result: list[int] = []
    epoch = 0
    while len(result) < groups:
        order = sorted(
            range(len(task_ids)),
            key=lambda index: canonical_sha256(
                {
                    "schema": TASK_SCHEDULE_SCHEMA,
                    "seed": int(seed),
                    "epoch": epoch,
                    "task_id": task_ids[index],
                }
            ),
        )
        result.extend(order[: groups - len(result)])
        epoch += 1
    return result


def split_visible_expect_harnesses(test_code: str) -> list[str]:
    spans = extract_expect_spans(str(test_code))
    if not spans:
        raise ValueError("visible feedback harness has no expect cases")
    return [
        harness_with_cases(str(test_code), spans, {index})
        for index in range(len(spans))
    ]


def score_dart_candidate(
    candidate: str,
    feedback_tests: str,
    task_id: str,
    *,
    timeout: int,
    stability_runs: int,
) -> dict[str, Any]:
    """Return completion-attested visible execution evidence."""

    variants = split_visible_expect_harnesses(feedback_tests)
    compiled, full_pass, diagnostic, _ = evaluate_dart_jit_tests_detail(
        candidate,
        feedback_tests,
        f"{task_id}-full",
        timeout=timeout,
        stability_runs=stability_runs,
    )
    if full_pass:
        test_passes = [True] * len(variants)
    else:
        test_passes: list[bool] = []
        for index, variant in enumerate(variants):
            one_compiled, passed, _, _ = evaluate_dart_jit_tests_detail(
                candidate,
                variant,
                f"{task_id}-test-{index}",
                timeout=timeout,
                stability_runs=stability_runs,
            )
            test_passes.append(bool(one_compiled and passed))
    return {
        "compiled": bool(compiled),
        "full_pass": bool(compiled and full_pass),
        "test_passes": test_passes,
        # Raw diagnostics stay in memory only until compiler sanitization.
        "diagnostic": str(diagnostic or "")[:12000],
    }


def _target_source(row: Mapping[str, Any], task_id: str) -> str:
    target = str(
        row.get("supervised_target")
        or row.get("dart_source")
        or row.get("source")
        or ""
    ).strip()
    if not target:
        raise ValueError(f"{task_id}: missing replay target")
    return target


@dataclass(frozen=True)
class VerpoTask:
    task_id: str
    source: str
    source_sha256: str
    feedback_tests: str
    replay_target: str


def load_verpo_tasks(
    rollout_path: str | Path,
    f2_path: str | Path,
) -> list[VerpoTask]:
    rollout_rows = _read_jsonl(rollout_path, "VeRPO rollout")
    f2_rows = _read_jsonl(f2_path, "VeRPO F2")
    if len(rollout_rows) != len(f2_rows):
        raise ValueError("VeRPO rollout/F2 row counts differ")
    tasks: list[VerpoTask] = []
    seen: set[str] = set()
    for index, (rollout, f2) in enumerate(zip(rollout_rows, f2_rows, strict=True)):
        task_id = str(rollout.get("task_id") or "").strip()
        f2_task_id = str(f2.get("task_id") or "").strip()
        if not task_id or task_id in seen or task_id != f2_task_id:
            raise ValueError(f"VeRPO task identity mismatch/duplicate at row {index}")
        seen.add(task_id)
        feedback_tests = rollout.get("feedback_tests")
        if not isinstance(feedback_tests, str) or not feedback_tests.strip():
            raise ValueError(f"{task_id}: feedback_tests is missing")
        forbidden = sorted(
            field for field in _FORBIDDEN_ROLLOUT_TEST_FIELDS if field in rollout
        )
        if forbidden:
            raise ValueError(
                f"{task_id}: rollout contains forbidden private/test fields "
                f"{forbidden}"
            )
        split_visible_expect_harnesses(feedback_tests)
        source = build_encoder_source(f2, task_id)
        tasks.append(
            VerpoTask(
                task_id=task_id,
                source=source,
                source_sha256=sha256_text(source),
                feedback_tests=feedback_tests,
                replay_target=_target_source(rollout, task_id),
            )
        )
    return tasks


def _decoder_special_ids(
    model: Any, tokenizer: Any
) -> tuple[int, int, tuple[int, ...]]:
    decoder = getattr(model.config, "decoder", None)
    start = getattr(decoder, "bos_token_id", None)
    if type(start) is not int:
        start = getattr(model.config, "decoder_start_token_id", None)
    pad = getattr(decoder, "pad_token_id", None)
    if type(pad) is not int:
        pad = tokenizer.pad_token_id
    eos = getattr(decoder, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = (int(eos),) if type(eos) is int else tuple(int(v) for v in eos)
    if type(start) is not int or type(pad) is not int or not eos_ids:
        raise ValueError("loaded model lacks decoder BOS/PAD/EOS IDs")
    return int(start), int(pad), eos_ids


def _encode_source(
    tokenizer: Any,
    source: str,
    *,
    max_source_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        source,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("tokenizer did not return one encoder row")
    if input_ids.size(1) > max_source_tokens:
        raise ValueError(
            f"encoder source length {input_ids.size(1)} exceeds "
            f"{max_source_tokens}; repair/source truncation is forbidden"
        )
    return input_ids.to(device), attention_mask.to(device)


def _encode_target(
    tokenizer: Any,
    target: str,
    *,
    max_target_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    values = tokenizer(
        target,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    ids = [int(value) for value in values]
    eos = int(tokenizer.eos_token_id)
    if not ids or ids[-1] != eos:
        ids.append(eos)
    if len(ids) > max_target_tokens:
        raise ValueError(
            f"SFT replay target length {len(ids)} exceeds {max_target_tokens}"
        )
    return torch.tensor([ids], dtype=torch.long, device=device)


def _decode_candidate(tokenizer: Any, actions: Sequence[int]) -> str:
    text = tokenizer.decode(
        list(actions),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    # This is only a deterministic convenience.  The evaluator performs its
    # own hardened code extraction as well.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    fenced = re.search(r"```(?:dart)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    return text


@dataclass
class RolloutTrajectory:
    task_id: str
    state_kind: str
    source: str
    source_sha256: str
    actions: tuple[int, ...]
    candidate: str
    detail: dict[str, Any]
    saved_logprobs: torch.Tensor
    advantage: float = 0.0
    parent_candidate_sha256: str = ""
    feedback_sha256: str = ""
    sampled_pad_before_eos: int = 0


def _generation_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    pad_token_id: int,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Declare the exact untruncated sampling distribution."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")
    eos_ids = tuple(eos_token_ids)
    if (
        not eos_ids
        or len(set(eos_ids)) != len(eos_ids)
        or any(type(value) is not int or value < 0 for value in eos_ids)
    ):
        raise ValueError("eos_token_ids must be non-empty unique token IDs")
    eos_value: int | list[int] = eos_ids[0] if len(eos_ids) == 1 else list(eos_ids)
    return {
        "do_sample": True,
        "temperature": float(temperature),
        # Every truncation/sampling warper is explicitly neutralized. PAD is
        # reserved for synthetic post-EOS padding, so remove it from sampling
        # support and apply the identical mask when recomputing logprobs.
        "top_p": 1.0,
        "top_k": 0,
        "min_p": None,
        "typical_p": 1.0,
        "top_h": None,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "penalty_alpha": None,
        "sequence_bias": None,
        "constraints": None,
        "force_words_ids": None,
        "renormalize_logits": False,
        "remove_invalid_values": False,
        "bad_words_ids": None,
        "suppress_tokens": (None if pad_token_id in eos_ids else [int(pad_token_id)]),
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "watermarking_config": None,
        "stop_strings": None,
        "token_healing": False,
        "guidance_scale": None,
        "dola_layers": None,
        "early_stopping": False,
        "length_penalty": 1.0,
        "max_time": None,
        "min_length": 0,
        "min_new_tokens": 0,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_value,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "output_logits": False,
    }


def generate_group(
    *,
    model: Any,
    tokenizer: Any,
    task_id: str,
    source: str,
    state_kind: str,
    group_size: int,
    max_source_tokens: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    parent_candidate_sha256: str = "",
    feedback_sha256: str = "",
) -> list[RolloutTrajectory]:
    """Generate and pre-update score one fixed-conditioning group."""

    decoder_start, pad_id, eos_ids = _decoder_special_ids(model, tokenizer)
    suppressed_ids = tuple() if pad_id in eos_ids else (pad_id,)
    input_ids, attention_mask = _encode_source(
        tokenizer,
        source,
        max_source_tokens=max_source_tokens,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    prefix = torch.tensor([[decoder_start]], dtype=torch.long, device=device)
    trajectories: list[RolloutTrajectory] = []
    for _ in range(group_size):
        with torch.no_grad():
            generated = model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=prefix,
                **_generation_kwargs(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    pad_token_id=pad_id,
                    eos_token_ids=eos_ids,
                ),
            )
        sequence = generated.sequences[0].detach().cpu()
        actions = normalize_generated_seq2seq_ids(
            sequence,
            decoder_prefix_ids=[decoder_start],
            eos_token_ids=eos_ids,
            pad_token_id=pad_id,
        )
        saved = (
            seq2seq_completion_token_logprobs(
                model,
                actions,
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=attention_mask,
                temperature=temperature,
                pad_token_id=pad_id,
                eos_token_ids=eos_ids,
                decoder_start_token_id=decoder_start,
                suppressed_token_ids=suppressed_ids,
                with_grad=False,
            )
            .detach()
            .cpu()
        )
        trajectories.append(
            RolloutTrajectory(
                task_id=task_id,
                state_kind=state_kind,
                source=source,
                source_sha256=sha256_text(source),
                actions=tuple(actions),
                candidate=_decode_candidate(tokenizer, actions),
                detail={},
                saved_logprobs=saved,
                parent_candidate_sha256=parent_candidate_sha256,
                feedback_sha256=feedback_sha256,
                sampled_pad_before_eos=0,
            )
        )
    return trajectories


def score_group(
    trajectories: Sequence[RolloutTrajectory],
    *,
    feedback_tests: str,
    timeout: int,
    stability_runs: int,
    workers: int,
) -> None:
    if not trajectories:
        raise ValueError("cannot score an empty rollout group")
    with ThreadPoolExecutor(max_workers=min(workers, len(trajectories))) as pool:
        futures = [
            pool.submit(
                score_dart_candidate,
                trajectory.candidate,
                feedback_tests,
                f"{trajectory.task_id}-{trajectory.state_kind}-{index}",
                timeout=timeout,
                stability_runs=stability_runs,
            )
            for index, trajectory in enumerate(trajectories)
        ]
        for trajectory, future in zip(trajectories, futures, strict=True):
            trajectory.detail = future.result()


def assign_advantages(
    trajectories: Sequence[RolloutTrajectory],
    *,
    alpha: float,
    local_weight: float,
    compile_weight: float,
) -> dict[str, list[float]]:
    reward = verpo_execution_compile_advantages(
        [trajectory.detail for trajectory in trajectories],
        alpha=alpha,
        local_weight=local_weight,
        compile_weight=compile_weight,
    )
    for trajectory, advantage in zip(
        trajectories, reward["unified_advantages"], strict=True
    ):
        trajectory.advantage = float(advantage)
    return reward


def _group_is_zero_pass(
    trajectories: Sequence[RolloutTrajectory],
) -> bool:
    return bool(trajectories) and all(
        not trajectory.detail["full_pass"] and not any(trajectory.detail["test_passes"])
        for trajectory in trajectories
    )


def build_repair_groups(
    *,
    model: Any,
    tokenizer: Any,
    task: VerpoTask,
    base_group: Sequence[RolloutTrajectory],
    max_parents: int,
    repair_group_size: int,
    max_source_tokens: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[list[RolloutTrajectory]]:
    """Create compiler-feedback states only from non-compiling zero-pass rows."""

    if not _group_is_zero_pass(base_group):
        return []
    eligible = [
        index
        for index, trajectory in enumerate(base_group)
        if not bool(trajectory.detail["compiled"])
        and bool(trajectory.candidate.strip())
    ]
    if not eligible:
        return []
    candidate_subset = [base_group[index].candidate for index in eligible]
    selected_local = max_min_diverse_indices(
        candidate_subset, min(max_parents, len(candidate_subset))
    )
    selected = [eligible[index] for index in selected_local]
    groups: list[list[RolloutTrajectory]] = []
    for index in selected:
        parent = base_group[index]
        repair = build_compiler_repair_context(
            task_id=task.task_id,
            source_sha256=task.source_sha256,
            candidate=parent.candidate,
            diagnostic=str(parent.detail.get("diagnostic") or ""),
            compiled=False,
        )
        repair_source = task.source + "\n" + str(repair["text"])
        group = generate_group(
            model=model,
            tokenizer=tokenizer,
            task_id=task.task_id,
            source=repair_source,
            state_kind="compiler_repair",
            group_size=repair_group_size,
            max_source_tokens=max_source_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            parent_candidate_sha256=sha256_text(parent.candidate),
            feedback_sha256=str(repair["payload"]["compiler_feedback_sha256"]),
        )
        groups.append(group)
    return groups


def policy_token_loss(
    current_logprobs: torch.Tensor,
    saved_logprobs: torch.Tensor,
    advantage: float,
    *,
    ppo_clip: float,
) -> torch.Tensor:
    if current_logprobs.ndim != 1 or current_logprobs.numel() == 0:
        raise ValueError("current logprobs must be a nonempty vector")
    saved = saved_logprobs.to(
        device=current_logprobs.device,
        dtype=current_logprobs.dtype,
    )
    if saved.shape != current_logprobs.shape:
        raise ValueError("saved/current logprob shapes differ")
    advantage_tensor = current_logprobs.new_tensor(float(advantage))
    if ppo_clip == 0.0:
        return -(advantage_tensor * current_logprobs).mean()
    ratios = torch.exp((current_logprobs - saved.detach()).clamp(-20.0, 20.0))
    unclipped = ratios * advantage_tensor
    clipped = ratios.clamp(1.0 - ppo_clip, 1.0 + ppo_clip) * advantage_tensor
    return -torch.minimum(unclipped, clipped).mean()


def declared_step_trajectory_slots(
    *,
    tasks_per_update: int,
    group_size: int,
    max_repair_parents: int,
    repair_group_size: int,
) -> int:
    """Return the predeclared fixed denominator for one optimizer step."""

    values = {
        "tasks_per_update": tasks_per_update,
        "group_size": group_size,
        "max_repair_parents": max_repair_parents,
        "repair_group_size": repair_group_size,
    }
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("trajectory-slot dimensions must be positive integers")
    return tasks_per_update * (group_size + max_repair_parents * repair_group_size)


def normalize_step_slot_loss(
    loss: torch.Tensor,
    *,
    declared_slots: int,
) -> torch.Tensor:
    """Normalize every realized trajectory by the same sealed step budget."""

    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError("trajectory policy loss must be one scalar tensor")
    if type(declared_slots) is not int or declared_slots <= 0:
        raise ValueError("declared trajectory slots must be positive")
    return loss / declared_slots


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in value:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _load_policy(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any] | None, dict[str, Any]]:
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    warmstart = Path(args.warmstart_checkpoint).expanduser().resolve()
    warm_contract = _read_json(
        warmstart / "run_contract.json", "SFT warmstart run contract"
    )
    warm_schema = str(warm_contract.get("schema") or "")
    if warm_schema not in SUPPORTED_WARMSTART_SCHEMAS:
        raise ValueError(
            "warmstart is not a supported native T5Gemma 2 adapter checkpoint"
        )
    if warm_contract.get("architecture") != "native_encoder_decoder":
        raise ValueError("warmstart is not the native encoder-decoder architecture")
    if warm_schema == MIXED_RS_SFT_RUN_SCHEMA:
        privacy = warm_contract.get("privacy")
        dataset = warm_contract.get("dataset")
        if (
            not isinstance(privacy, Mapping)
            or privacy.get("heldout_overlap") != 0
            or privacy.get("tests_model_visible") is not False
            or privacy.get("private_feedback_model_visible") is not False
            or not isinstance(dataset, Mapping)
            or dataset.get("schema") != "t5gemma2-mixed-rs-sft-dataset-v1"
            or dataset.get("heldout_overlap") != 0
        ):
            raise ValueError("mixed RS-SFT warmstart privacy contract failed")
    base_record = warm_contract.get("base_model") or {}
    model_name = str(base_record.get("name") or DEFAULT_MODEL)
    revision = str(
        base_record.get("resolved_commit")
        or base_record.get("requested_revision")
        or ""
    )
    if not revision:
        raise ValueError("warmstart does not pin an immutable base revision")
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        warmstart / "tokenizer",
        trust_remote_code=False,
    )
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        revision=revision,
        token=token,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    # The SFT warmstart seals its config after gradient checkpointing disables
    # cache. Compare the base in that same normalized state.
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    if hasattr(base.config, "decoder"):
        base.config.decoder.use_cache = False
    if canonical_sha256(base.config.to_dict()) != str(
        base_record.get("config_sha256") or ""
    ):
        raise ValueError("loaded base config differs from SFT warmstart")

    resume_state: dict[str, Any] | None = None
    adapter_path = warmstart / "adapter"
    if args.resume_checkpoint:
        resume = Path(args.resume_checkpoint).expanduser().resolve()
        adapter_path = resume / "adapter"
        state = torch.load(
            resume / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("VeRPO resume checkpoint schema mismatch")
        resume_state = state
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=True,
    )
    if hasattr(model.config, "decoder"):
        model.config.decoder.use_cache = False
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("vision" in name.lower() for name in trainable):
        raise ValueError("VeRPO trainability contract includes no adapters or vision")
    return model, tokenizer, resume_state, warm_contract


def _prune_checkpoints(output_dir: Path, *, keep: int) -> None:
    if type(keep) is not int or keep <= 0:
        raise ValueError("checkpoint retention count must be positive")
    root = output_dir.resolve()
    root_contract_path = root / "run_contract.json"
    if not root_contract_path.is_file():
        raise ValueError("refusing checkpoint pruning without root run contract")
    root_contract = _read_json(root_contract_path, "root run contract")
    if root_contract.get("schema") != RUN_SCHEMA:
        raise ValueError("refusing checkpoint pruning for a foreign root run")
    root_contract_sha256 = canonical_sha256(root_contract)

    checkpoints: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()

    # Validate every checkpoint-shaped directory before deleting any of them.
    # This prevents a mixed/foreign directory from causing a partial prune.
    for _, path in checkpoints:
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError(f"refusing to prune redirected checkpoint {path}")
        contract_path = path / "run_contract.json"
        if not contract_path.is_file():
            raise ValueError(f"refusing to prune unsealed directory {path}")
        contract = _read_json(contract_path, "checkpoint run contract")
        if (
            contract.get("schema") != RUN_SCHEMA
            or canonical_sha256(contract) != root_contract_sha256
        ):
            raise ValueError(f"refusing to prune foreign checkpoint {path}")
        if (
            not (path / "training_state.pt").is_file()
            or not (path / "adapter").is_dir()
            or not (path / "tokenizer").is_dir()
        ):
            raise ValueError(f"refusing to prune incomplete checkpoint {path}")

    for _, path in checkpoints[:-keep]:
        shutil.rmtree(path)


def save_checkpoint(
    *,
    output_dir: Path,
    update: int,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    run_contract: Mapping[str, Any],
    keep: int,
) -> Path:
    destination = output_dir / f"checkpoint-optstep-{update:06d}"
    if destination.exists():
        raise FileExistsError(f"immutable checkpoint exists: {destination}")
    temporary = output_dir / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(temporary / "tokenizer")
    _atomic_json(temporary / "run_contract.json", dict(run_contract))
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "update": int(update),
            "optimizer": optimizer.state_dict(),
            "rng": _rng_state(),
            "run_contract_sha256": canonical_sha256(run_contract),
        },
        temporary / "training_state.pt",
    )
    os.replace(temporary, destination)
    _atomic_json(
        output_dir / "latest_checkpoint.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "update": int(update),
            "path": str(destination.resolve()),
            "run_contract_sha256": canonical_sha256(run_contract),
        },
    )
    _prune_checkpoints(output_dir, keep=keep)
    return destination


def _trajectory_record(trajectory: RolloutTrajectory) -> dict[str, Any]:
    detail = trajectory.detail
    return {
        "state_kind": trajectory.state_kind,
        "source_sha256": trajectory.source_sha256,
        "actions": list(trajectory.actions),
        "action_tokens": len(trajectory.actions),
        "candidate": trajectory.candidate,
        "candidate_sha256": sha256_text(trajectory.candidate),
        "compiled": bool(detail["compiled"]),
        "full_pass": bool(detail["full_pass"]),
        "test_passes": list(detail["test_passes"]),
        "advantage": float(trajectory.advantage),
        "saved_logprobs_sha256": hashlib.sha256(
            trajectory.saved_logprobs.float().numpy().tobytes()
        ).hexdigest(),
        "parent_candidate_sha256": trajectory.parent_candidate_sha256,
        "feedback_sha256": trajectory.feedback_sha256,
        "raw_diagnostic_persisted": False,
        "sampled_pad_before_eos": int(trajectory.sampled_pad_before_eos),
    }


def _validate_feedback_boundary(args: argparse.Namespace) -> dict[str, Any]:
    return validate_feedback_training_boundary(
        rollout=args.rollout_file,
        seal=args.rollout_seal,
        f2=args.f2_jsonl,
        f2_manifest=args.f2_manifest,
        public_manifest=args.feedback_public_manifest,
        expected_public_manifest_sha256=(args.expected_feedback_public_manifest_sha256),
        contract=args.compact_contract,
        expected_accounting=None,
        expected_eligible_task_ids_sha256=None,
        expected_excluded_task_ids_sha256=None,
    )


def _run_contract(
    *,
    args: argparse.Namespace,
    boundary: Mapping[str, Any],
    tasks: Sequence[VerpoTask],
    warm_contract: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    decoder_capacity = int(
        ((warm_contract.get("base_model") or {}).get("decoder_capacity") or 0)
    )
    if decoder_capacity <= 1:
        raise ValueError("SFT warmstart has no sealed decoder capacity")
    if args.max_new_tokens + 1 > decoder_capacity:
        raise ValueError(
            "generation actions plus the decoder-start prefix exceed the "
            f"sealed decoder capacity: {args.max_new_tokens}+1 > "
            f"{decoder_capacity}"
        )
    planned_groups = args.max_updates * args.tasks_per_update
    task_ids = [task.task_id for task in tasks]
    schedule = deterministic_task_schedule(
        task_ids,
        seed=args.seed,
        groups=planned_groups,
    )
    planned_ids = [task_ids[index] for index in schedule]
    _, pad_id, eos_ids = _decoder_special_ids(model, tokenizer)
    suppressed_ids = [] if pad_id in eos_ids else [pad_id]
    return {
        "schema": RUN_SCHEMA,
        "runtime_provenance": runtime_provenance(),
        "architecture": "native_t5gemma2_encoder_decoder",
        "objective": ("on_policy_visible_execution_verpo_plus_local_compiler_repair"),
        "warmstart": {
            "path": str(Path(args.warmstart_checkpoint).resolve()),
            "run_contract_sha256": canonical_sha256(warm_contract),
            "stage_schema": str(warm_contract.get("schema") or ""),
            "production_floor_eligible": (
                warm_contract.get("production_floor_eligible", True) is True
            ),
        },
        "feedback_boundary": dict(boundary),
        "tasks": {
            "rows": len(tasks),
            "task_ids_sha256": canonical_sha256(task_ids),
            "schedule_task_ids_sha256": canonical_sha256(planned_ids),
            "schedule_without_replacement_within_cycle": True,
        },
        "sampling": {
            "group_size": args.group_size,
            "repair_group_size": args.repair_group_size,
            "max_repair_parents": args.max_repair_parents,
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": args.max_new_tokens,
            "max_source_tokens": args.max_source_tokens,
            "decoder_prefix_is_not_an_action": True,
            "sampled_eos_is_an_action": True,
            "pad_token_id": pad_id,
            "eos_token_ids": list(eos_ids),
            "suppressed_token_ids": suppressed_ids,
            "pad_removed_from_sampling_support": bool(suppressed_ids),
            "sampling_support_constraint_exactly_recomputed": True,
            "pad_before_eos_fail_closed": True,
            "distribution_truncated": False,
        },
        "reward": {
            "visible_tests_only": True,
            "global_full_pass": True,
            "density_calibrated_partial_tests": True,
            "verpo_alpha": args.verpo_alpha,
            "local_weight": args.local_weight,
            "compile_weight": args.compile_weight,
            "components_independently_mean_centered": True,
            "diagnostic_text_is_not_a_scalar_reward": True,
        },
        "repair": {
            "trigger": "all_zero_visible_test_group",
            "eligibility": "noncompiling_candidates_only",
            "parent_selection": "deterministic_max_min_token_trigram_distance",
            "feedback": "sanitized_local_dart_compiler_only",
            "repair_advantage_updates_original_state": False,
            "same_policy_version_before_one_update": True,
        },
        "optimization": {
            "max_updates": args.max_updates,
            "tasks_per_update": args.tasks_per_update,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "ppo_clip": args.ppo_clip,
            "sft_replay_weight": args.sft_replay_weight,
            "on_policy_logprob_tolerance": (args.on_policy_logprob_tolerance),
            "declared_trajectory_slots_per_task": (
                declared_step_trajectory_slots(
                    tasks_per_update=1,
                    group_size=args.group_size,
                    max_repair_parents=args.max_repair_parents,
                    repair_group_size=args.repair_group_size,
                )
            ),
        },
        "checkpoint": {
            "interval": args.checkpoint_interval,
            "keep_last": args.keep_last_checkpoints,
            "base_model_duplicated": False,
        },
        "no_frontier_api": True,
        "llm_judge": False,
        "acceptance_tests_exposed": False,
        "private_holdback_exposed": False,
        "seed": args.seed,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("VeRPO requires CUDA")
    validate_dart_binary()
    boundary = _validate_feedback_boundary(args)
    tasks = load_verpo_tasks(args.rollout_file, args.f2_jsonl)
    if len(tasks) != int(boundary["rows"]):
        raise ValueError("loaded VeRPO task count differs from sealed boundary")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, resume_state, warm_contract = _load_policy(args)
    run_contract = _run_contract(
        args=args,
        boundary=boundary,
        tasks=tasks,
        warm_contract=warm_contract,
        model=model,
        tokenizer=tokenizer,
    )
    if resume_state is not None:
        resume_contract = _read_json(
            Path(args.resume_checkpoint) / "run_contract.json",
            "VeRPO resume run contract",
        )
        if resume_state.get("run_contract_sha256") != canonical_sha256(
            resume_contract
        ) or canonical_sha256(resume_contract) != canonical_sha256(run_contract):
            raise ValueError("VeRPO resume contract mismatch")
    _atomic_json(output_dir / "run_contract.json", run_contract)

    device = torch.device("cuda")
    model.to(device)
    model.eval()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    start_update = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        _restore_rng_state(resume_state["rng"])
        start_update = int(resume_state["update"])

    task_ids = [task.task_id for task in tasks]
    schedule = deterministic_task_schedule(
        task_ids,
        seed=args.seed,
        groups=args.max_updates * args.tasks_per_update,
    )
    decoder_start, pad_id, eos_ids = _decoder_special_ids(model, tokenizer)
    suppressed_ids = tuple() if pad_id in eos_ids else (pad_id,)
    declared_slots = declared_step_trajectory_slots(
        tasks_per_update=args.tasks_per_update,
        group_size=args.group_size,
        max_repair_parents=args.max_repair_parents,
        repair_group_size=args.repair_group_size,
    )
    metrics_path = output_dir / "rollout_metrics.jsonl"

    for update_index in range(start_update, args.max_updates):
        optimizer.zero_grad(set_to_none=True)
        step_trajectories: list[RolloutTrajectory] = []
        replay_tasks: list[VerpoTask] = []
        task_records: list[dict[str, Any]] = []
        for task_offset in range(args.tasks_per_update):
            schedule_index = update_index * args.tasks_per_update + task_offset
            task = tasks[schedule[schedule_index]]
            replay_tasks.append(task)
            base_group = generate_group(
                model=model,
                tokenizer=tokenizer,
                task_id=task.task_id,
                source=task.source,
                state_kind="base",
                group_size=args.group_size,
                max_source_tokens=args.max_source_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            )
            score_group(
                base_group,
                feedback_tests=task.feedback_tests,
                timeout=args.reward_timeout,
                stability_runs=args.reward_stability_runs,
                workers=args.reward_workers,
            )
            base_reward = assign_advantages(
                base_group,
                alpha=args.verpo_alpha,
                local_weight=args.local_weight,
                compile_weight=args.compile_weight,
            )
            repair_groups = build_repair_groups(
                model=model,
                tokenizer=tokenizer,
                task=task,
                base_group=base_group,
                max_parents=args.max_repair_parents,
                repair_group_size=args.repair_group_size,
                max_source_tokens=args.max_source_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            )
            repair_rewards: list[dict[str, list[float]]] = []
            for repair_group in repair_groups:
                score_group(
                    repair_group,
                    feedback_tests=task.feedback_tests,
                    timeout=args.reward_timeout,
                    stability_runs=args.reward_stability_runs,
                    workers=args.reward_workers,
                )
                repair_rewards.append(
                    assign_advantages(
                        repair_group,
                        alpha=args.verpo_alpha,
                        local_weight=args.local_weight,
                        compile_weight=args.compile_weight,
                    )
                )
            task_trajectories = list(base_group)
            for group in repair_groups:
                task_trajectories.extend(group)
            step_trajectories.extend(task_trajectories)
            task_records.append(
                {
                    "task_id": task.task_id,
                    "base_zero_pass": _group_is_zero_pass(base_group),
                    "repair_groups": len(repair_groups),
                    "base_reward": base_reward,
                    "repair_rewards": repair_rewards,
                    "trajectories": [
                        _trajectory_record(item) for item in task_trajectories
                    ],
                }
            )

        active_policy_trajectories = 0
        max_drift = 0.0
        policy_loss_value = 0.0
        for trajectory in step_trajectories:
            if trajectory.advantage == 0.0:
                continue
            input_ids, attention_mask = _encode_source(
                tokenizer,
                trajectory.source,
                max_source_tokens=args.max_source_tokens,
                device=device,
            )
            current = seq2seq_completion_token_logprobs(
                model,
                trajectory.actions,
                encoder_input_ids=input_ids,
                encoder_attention_mask=attention_mask,
                temperature=args.temperature,
                pad_token_id=pad_id,
                eos_token_ids=eos_ids,
                decoder_start_token_id=decoder_start,
                suppressed_token_ids=suppressed_ids,
                with_grad=True,
            )
            drift = validate_on_policy_logprob_drift(
                current,
                trajectory.saved_logprobs,
                tolerance=args.on_policy_logprob_tolerance,
                rollout_conditioning_sha256=trajectory.source_sha256,
                current_conditioning_sha256=sha256_text(trajectory.source),
                rollout_temperature=args.temperature,
                current_temperature=args.temperature,
            )
            max_drift = max(max_drift, drift)
            loss = normalize_step_slot_loss(
                policy_token_loss(
                    current,
                    trajectory.saved_logprobs,
                    trajectory.advantage,
                    ppo_clip=args.ppo_clip,
                ),
                declared_slots=declared_slots,
            )
            loss.backward()
            policy_loss_value += float(loss.detach().cpu())
            active_policy_trajectories += 1

        replay_loss_value = 0.0
        if args.sft_replay_weight > 0.0:
            for task in replay_tasks:
                input_ids, attention_mask = _encode_source(
                    tokenizer,
                    task.source,
                    max_source_tokens=args.max_source_tokens,
                    device=device,
                )
                labels = _encode_target(
                    tokenizer,
                    task.replay_target,
                    max_target_tokens=args.max_target_tokens,
                    device=device,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                replay_loss = (
                    outputs.loss * args.sft_replay_weight / args.tasks_per_update
                )
                if not torch.isfinite(replay_loss):
                    raise FloatingPointError("SFT replay loss is non-finite")
                replay_loss.backward()
                replay_loss_value += float(replay_loss.detach().cpu())

        if active_policy_trajectories == 0 and args.sft_replay_weight == 0.0:
            grad_norm = 0.0
            optimizer.zero_grad(set_to_none=True)
            optimizer_step = False
        else:
            grad_norm_tensor = clip_grad_norm_(trainable, args.max_grad_norm)
            grad_norm = float(grad_norm_tensor)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step = True

        completed_update = update_index + 1
        if (
            completed_update % args.checkpoint_interval == 0
            or completed_update == args.max_updates
        ):
            save_checkpoint(
                output_dir=output_dir,
                update=completed_update,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                run_contract=run_contract,
                keep=args.keep_last_checkpoints,
            )

        all_details = [trajectory.detail for trajectory in step_trajectories]
        metric = bind_run_contract(
            {
                "schema": ROLLOUT_SCHEMA,
                "update": completed_update,
                "task_records": task_records,
                "trajectory_count": len(step_trajectories),
                "declared_trajectory_slots": declared_slots,
                "active_policy_trajectories": active_policy_trajectories,
                "compiled": sum(bool(item["compiled"]) for item in all_details),
                "full_pass": sum(bool(item["full_pass"]) for item in all_details),
                "visible_tests_passed": sum(
                    sum(bool(value) for value in item["test_passes"])
                    for item in all_details
                ),
                "sampled_pad_before_eos": sum(
                    int(trajectory.sampled_pad_before_eos)
                    for trajectory in step_trajectories
                ),
                "policy_loss": policy_loss_value,
                "sft_replay_loss": replay_loss_value,
                "max_on_policy_logprob_drift": max_drift,
                "grad_norm": grad_norm,
                "optimizer_step": optimizer_step,
                "no_frontier_api": True,
            },
            run_contract,
        )
        with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    metric,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        summary = {
            key: metric[key]
            for key in (
                "update",
                "trajectory_count",
                "active_policy_trajectories",
                "compiled",
                "full_pass",
                "visible_tests_passed",
                "sampled_pad_before_eos",
                "policy_loss",
                "sft_replay_loss",
                "max_on_policy_logprob_drift",
                "grad_norm",
            )
        }
        print(json.dumps(summary, sort_keys=True), flush=True)

    result = bind_run_contract(
        {
            "schema": RUN_SCHEMA,
            "status": "complete",
            "updates": args.max_updates,
            "latest_checkpoint": (f"checkpoint-optstep-{args.max_updates:06d}"),
            "no_frontier_api": True,
            "production_floor_eligible": (
                run_contract["warmstart"]["production_floor_eligible"] is True
            ),
        },
        run_contract,
    )
    _atomic_json(output_dir / "result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--rollout_file", required=True)
    parser.add_argument("--rollout_seal", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--f2_manifest", required=True)
    parser.add_argument("--feedback_public_manifest", required=True)
    parser.add_argument("--expected_feedback_public_manifest_sha256", required=True)
    parser.add_argument("--compact_contract", required=True)
    parser.add_argument("--warmstart_checkpoint", required=True)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--repair_group_size", type=int, default=4)
    parser.add_argument("--max_repair_parents", type=int, default=2)
    parser.add_argument("--tasks_per_update", type=int, default=1)
    parser.add_argument("--max_updates", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=32767)
    parser.add_argument("--max_source_tokens", type=int, default=65536)
    parser.add_argument("--max_target_tokens", type=int, default=32768)
    parser.add_argument("--verpo_alpha", type=float, default=2.0)
    parser.add_argument("--local_weight", type=float, default=1.0)
    parser.add_argument("--compile_weight", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ppo_clip", type=float, default=0.0)
    parser.add_argument("--sft_replay_weight", type=float, default=0.02)
    parser.add_argument("--on_policy_logprob_tolerance", type=float, default=2e-4)
    parser.add_argument("--reward_workers", type=int, default=4)
    parser.add_argument("--reward_timeout", type=int, default=30)
    parser.add_argument("--reward_stability_runs", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--keep_last_checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa"],
        default="sdpa",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    for name in (
        "group_size",
        "repair_group_size",
        "max_repair_parents",
        "tasks_per_update",
        "max_updates",
        "max_new_tokens",
        "max_source_tokens",
        "max_target_tokens",
        "reward_workers",
        "reward_timeout",
        "reward_stability_runs",
        "checkpoint_interval",
        "keep_last_checkpoints",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        parser.error("--temperature must be finite and positive")
    if args.verpo_alpha <= 0.0:
        parser.error("--verpo_alpha must be positive")
    if args.local_weight < 0.0 or args.compile_weight < 0.0:
        parser.error("reward weights must be non-negative")
    if args.learning_rate <= 0.0 or args.max_grad_norm <= 0.0:
        parser.error("optimization rates must be positive")
    if args.weight_decay < 0.0 or args.sft_replay_weight < 0.0:
        parser.error("weight decay/replay weight must be non-negative")
    if args.ppo_clip < 0.0 or args.ppo_clip >= 1.0:
        parser.error("--ppo_clip must be zero or lie in (0,1)")
    if args.on_policy_logprob_tolerance <= 0.0:
        parser.error("--on_policy_logprob_tolerance must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
