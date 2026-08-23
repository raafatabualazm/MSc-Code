#!/usr/bin/env python3
"""Batched decoder-only generation from compact source tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from models.direct_compact_causal import (
    CONTRACT_SCHEMA_V3,
    DirectCompactContract,
    restore_source_embedding_overlay,
    resolve_decoder_config_path,
    sha256_artifact,
    sha256_file,
    validate_base_model_vocab,
    validate_v3_pool_alignment_metadata,
)
from scripts.training.direct_compact_qwen_decompiler import (
    DIRECT_PROMPT_MODE_CODE_ONLY_V1,
    DIRECT_PROMPT_MODES,
    _encode,
    direct_prompt,
)
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    require_unique_slots,
)


MODEL_PUBLIC_FIELDS = frozenset(
    {
        "compact_input_ids",
        "compact_codec_sha256",
        "compact_codebook_sha256",
        "compact_tokenizer_sha256",
    }
)
INFERENCE_JOURNAL_SCHEMA = "direct-compact-inference-journal-v1"
RESCUE_CONDITIONING_SCHEMA = "direct-compact-verpo-rescue-conditioning-v1"
RESCUE_ARMS = frozenset(
    {
        "plain_resample",
        "compiler_only",
        "diagnosis_only",
        "diagnosis_and_steps",
    }
)


def load_rescue_conditioning_plan(path: str | Path) -> dict[str, Any]:
    """Load one immutable, candidate-rank-specific rescue conditioning plan.

    A plan reserves the same number of repair slots for every task. Rows whose
    structured diagnosis failed validation remain in the plan with
    ``generate=false``; the downstream intention-to-treat scorer counts their
    reserved slots as failures instead of silently shrinking M.
    """

    plan_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"cannot parse rescue conditioning plan: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("rescue conditioning plan must be a JSON object")
    required = {
        "schema",
        "arm",
        "base_candidate_rank",
        "repairs_per_candidate",
        "source_plan_sha256",
        "rows",
    }
    if set(value) != required:
        raise ValueError(
            "rescue conditioning plan fields differ: "
            + ", ".join(sorted(set(value) ^ required))
        )
    if value.get("schema") != RESCUE_CONDITIONING_SCHEMA:
        raise ValueError("rescue conditioning plan schema is unsupported")
    arm = str(value.get("arm") or "")
    if arm not in RESCUE_ARMS:
        raise ValueError(f"unsupported rescue arm {arm!r}")
    candidate_rank = value.get("base_candidate_rank")
    repairs = value.get("repairs_per_candidate")
    if (
        isinstance(candidate_rank, bool)
        or not isinstance(candidate_rank, int)
        or candidate_rank < 0
        or isinstance(repairs, bool)
        or not isinstance(repairs, int)
        or repairs <= 0
    ):
        raise ValueError("rescue candidate rank/repair count is invalid")
    source_sha = str(value.get("source_plan_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("rescue source-plan SHA-256 is invalid")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("rescue conditioning plan rows are empty")
    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    row_fields = {
        "task_id",
        "generate",
        "conditioning",
        "conditioning_sha256",
        "rejection_reasons",
    }
    for position, raw in enumerate(raw_rows):
        if not isinstance(raw, dict) or set(raw) != row_fields:
            raise ValueError(
                f"rescue conditioning row {position} has invalid fields"
            )
        task_id = str(raw.get("task_id") or "")
        if not task_id or task_id in seen:
            raise ValueError(
                f"rescue conditioning row {position} has duplicate/empty task ID"
            )
        seen.add(task_id)
        generate = raw.get("generate")
        conditioning = raw.get("conditioning")
        reasons = raw.get("rejection_reasons")
        if (
            not isinstance(generate, bool)
            or not isinstance(conditioning, str)
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise ValueError(
                f"rescue conditioning row {task_id!r} has invalid values"
            )
        expected_sha = hashlib.sha256(conditioning.encode("utf-8")).hexdigest()
        if raw.get("conditioning_sha256") != expected_sha:
            raise ValueError(
                f"rescue conditioning row {task_id!r} hash mismatch"
            )
        if generate:
            if reasons:
                raise ValueError(
                    f"generatable rescue row {task_id!r} has rejection reasons"
                )
            if arm == "plain_resample" and conditioning:
                raise ValueError(
                    f"plain-resample row {task_id!r} must be unconditioned"
                )
            if arm != "plain_resample" and not conditioning.strip():
                raise ValueError(
                    f"rescue conditioning row {task_id!r} is empty for {arm}"
                )
        if not generate and (conditioning or not reasons):
            raise ValueError(
                f"rejected rescue row {task_id!r} must be empty with reasons"
            )
        normalized_rows.append(
            {
                "task_id": task_id,
                "generate": generate,
                "conditioning": conditioning,
                "conditioning_sha256": expected_sha,
                "rejection_reasons": list(reasons),
            }
        )
    return {
        "schema": RESCUE_CONDITIONING_SCHEMA,
        "path": str(plan_path),
        "sha256": sha256_file(plan_path),
        "arm": arm,
        "base_candidate_rank": candidate_rank,
        "repairs_per_candidate": repairs,
        "source_plan_sha256": source_sha,
        "rows": normalized_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rescue_conditioning_plan",
        default="",
        help=(
            "Optional sealed direct-compact rescue plan. It selects a task "
            "subset and appends exact candidate/feedback conditioning after "
            "the compact source."
        ),
    )
    parser.add_argument(
        "--journal",
        default="",
        help="Append-only exact-resume journal (defaults beside --output).",
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codec_artifact", required=True)
    parser.add_argument("--decoder_model", default="")
    parser.add_argument("--decoder_revision", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer_revision", default="")
    parser.add_argument("--tokenizer_json", required=True)
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--decoder_adapter", default="")
    parser.add_argument("--source_overlay", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Sampling top-k. Zero disables top-k truncation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--role", choices=["fit", "measure", "all"], default="all")
    parser.add_argument(
        "--direct_prompt_mode",
        choices=sorted(DIRECT_PROMPT_MODES),
        default=DIRECT_PROMPT_MODE_CODE_ONLY_V1,
        help=(
            "Checkpoint-conditioned direct prompt mode. This is sealed into "
            "the generation journal and output provenance."
        ),
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(
    path: Path,
    alignment_path: Path,
    contract: DirectCompactContract,
    tokenizer: Any,
    target_budget: int,
    role: str,
    direct_prompt_mode: str = DIRECT_PROMPT_MODE_CODE_ONLY_V1,
    rescue_conditioning: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if direct_prompt_mode not in DIRECT_PROMPT_MODES:
        raise ValueError(
            "unsupported direct prompt mode "
            f"{direct_prompt_mode!r}; expected one of "
            f"{sorted(DIRECT_PROMPT_MODES)}"
        )
    public_rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    alignment_rows = [
        json.loads(line)
        for line in alignment_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(public_rows) != len(alignment_rows):
        raise ValueError(
            "compact public/alignment row-count mismatch: "
            f"{len(public_rows)} != {len(alignment_rows)}"
        )
    conditioning_by_task: dict[str, dict[str, Any]] = {}
    planned_ids: list[str] = []
    planned_generate_ids: list[str] = []
    if rescue_conditioning is not None:
        for item in rescue_conditioning["rows"]:
            conditioning_by_task[item["task_id"]] = item
            planned_ids.append(item["task_id"])
            if item["generate"]:
                planned_generate_ids.append(item["task_id"])
        if not planned_generate_ids:
            raise ValueError("rescue conditioning plan has no generatable rows")
    rows = []
    seen: set[str] = set()
    observed_plan_ids: list[str] = []
    for index, (row, alignment) in enumerate(
        zip(public_rows, alignment_rows, strict=True)
    ):
        if not isinstance(row, dict) or frozenset(row) != MODEL_PUBLIC_FIELDS:
            raise ValueError(
                f"{path}: row {index + 1} must contain exactly the strict "
                "compact model fields"
            )
        if not isinstance(alignment, dict) or alignment.get("model_row") != index:
            raise ValueError(
                f"{alignment_path}: row {index + 1} has an invalid model_row"
            )
        alignment_role = str(alignment.get("role") or "").strip().lower()
        if alignment_role not in {"fit", "measure"}:
            raise ValueError(
                f"{alignment_path}: row {index + 1} has invalid role "
                f"{alignment_role!r}"
            )
        identity_value = alignment.get("task_id") or alignment.get("id")
        if identity_value in (None, ""):
            raise ValueError(
                f"{alignment_path}: row {index + 1} has no private identity"
            )
        identity = str(identity_value)
        conditioning_row = (
            conditioning_by_task.get(identity)
            if rescue_conditioning is not None
            else None
        )
        compact = contract.validate_row(row, identity)
        pool = contract.validate_v3_pool_payload(compact, tokenizer, identity)
        if contract.schema == CONTRACT_SCHEMA_V3:
            pool_metadata = validate_v3_pool_alignment_metadata(
                alignment, f"alignment row {identity!r}"
            )
            if pool_metadata["use_count"] != len(pool["uses"]):
                raise ValueError(
                    f"{identity}: pool_metadata use_count does not match "
                    "the compact pool payload"
                )
        if role != "all" and alignment_role != role:
            continue
        if rescue_conditioning is not None and conditioning_row is None:
            continue
        if identity in seen:
            raise ValueError(f"duplicate selected identity {identity!r}")
        seen.add(identity)
        if rescue_conditioning is not None:
            observed_plan_ids.append(identity)
            if not conditioning_row["generate"]:
                continue
        prompt = _encode(
            tokenizer,
            direct_prompt(
                {**row, "direct_prompt_mode": direct_prompt_mode},
                target_function=contract.target_function,
                target_language=contract.target_language,
            ),
            special=True,
        )
        conditioning_text = (
            str(conditioning_row["conditioning"])
            if conditioning_row is not None
            else ""
        )
        conditioning_ids = (
            _encode(tokenizer, conditioning_text, special=False)
            if conditioning_text
            else []
        )
        if (
            len(prompt)
            + len(compact)
            + len(conditioning_ids)
            + target_budget
            > contract.max_total_tokens
        ):
            raise ValueError(
                f"{identity}: prompt+compact+rescue conditioning leaves "
                "insufficient target budget"
            )
        rows.append(
            {
                "identity": identity,
                "prompt": prompt,
                "compact": compact,
                "conditioning": conditioning_ids,
                "conditioning_sha256": (
                    conditioning_row["conditioning_sha256"]
                    if conditioning_row is not None
                    else None
                ),
                "direct_prompt_mode": direct_prompt_mode,
            }
        )
    if not rows:
        raise ValueError(f"role {role!r} selected no compact rows")
    if rescue_conditioning is not None:
        if set(observed_plan_ids) != set(planned_ids):
            missing = sorted(set(planned_ids) - set(observed_plan_ids))
            raise ValueError(
                "rescue conditioning plan tasks do not exactly belong to "
                f"the selected compact/alignment role; missing={missing[:8]}"
            )
        observed = [str(row["identity"]) for row in rows]
        if set(observed) != set(planned_generate_ids):
            missing = sorted(set(planned_generate_ids) - set(observed))
            raise ValueError(
                "rescue conditioning tasks do not exactly match the selected "
                f"compact/alignment role; missing={missing[:8]}"
            )
    return rows


def _rescue_plan_binding(
    rescue_conditioning: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: rescue_conditioning[key]
        for key in (
            "schema",
            "path",
            "sha256",
            "arm",
            "base_candidate_rank",
            "repairs_per_candidate",
            "source_plan_sha256",
        )
    }


def _build_inference_journal_contract(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    alignment_path: Path,
    selected_ids: list[str],
    contract: DirectCompactContract,
    decoder_model: str,
    decoder_revision: str,
    model_config_sha256: str,
    rescue_conditioning: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema": INFERENCE_JOURNAL_SCHEMA,
        "dataset_sha256": sha256_file(dataset_path),
        "alignment_sha256": sha256_file(alignment_path),
        "selected_role": args.role,
        "selected_task_ids_sha256": canonical_sha256(selected_ids),
        "contract_sha256": sha256_file(args.contract),
        "codebook_sha256": sha256_file(args.codebook),
        "codec_sha256": sha256_file(args.codec_artifact),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_json),
        "decoder_model": decoder_model,
        "decoder_revision": decoder_revision,
        "model_config_sha256": model_config_sha256,
        "attn_implementation": args.attn_implementation,
        "decoder_adapter_sha256": (
            sha256_artifact(args.decoder_adapter)
            if args.decoder_adapter
            else None
        ),
        "source_overlay_sha256": sha256_file(args.source_overlay),
        "num_rows": len(selected_ids),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "direct_prompt_mode": args.direct_prompt_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "precision": (
            "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")
        ),
        "batch_seed_policy": _generation_seed_policy(rescue_conditioning),
        "started_without_terminal_policy": (
            "retry_identical_seeded_batch_with_hash_chained_receipt"
        ),
    }
    if rescue_conditioning is not None:
        payload["rescue_conditioning"] = _rescue_plan_binding(
            rescue_conditioning
        )
        payload["prediction_token_ids_persisted"] = True
        payload["generation_vocab_size"] = int(
            contract.base_vocab_size or 0
        )
    return payload


def validate_existing_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Validate-and-reuse one complete immutable inference artifact."""

    output_path = Path(args.output).expanduser().resolve()
    provenance_path = Path(str(output_path) + ".provenance.json")
    if output_path.exists() != provenance_path.exists():
        raise ValueError(
            "inference output/provenance is only partially present"
        )
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    outputs = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not isinstance(outputs, list) or not outputs:
        raise ValueError("existing inference artifacts have invalid JSON shape")
    dataset_path = Path(args.dataset).expanduser().resolve()
    alignment_path = Path(args.alignment).expanduser().resolve()
    contract = DirectCompactContract.load(args.contract)
    decoder_model = args.decoder_model.strip() or contract.decoder_model
    decoder_revision = (
        args.decoder_revision.strip() or contract.decoder_revision
    )
    if (
        decoder_model != contract.decoder_model
        or decoder_revision != contract.decoder_revision
    ):
        raise ValueError(
            "existing inference decoder override differs from its contract"
        )
    alignment_rows = [
        json.loads(line)
        for line in alignment_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    public_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(public_rows) != len(alignment_rows):
        raise ValueError("existing inference input views are misaligned")
    selected_ids = [
        str(row.get("task_id") or row.get("id") or "")
        for row in alignment_rows
        if args.role == "all" or row.get("role") == args.role
    ]
    selected_alignment_ids = list(selected_ids)
    rescue_conditioning = (
        load_rescue_conditioning_plan(args.rescue_conditioning_plan)
        if args.rescue_conditioning_plan
        else None
    )
    if rescue_conditioning is not None:
        if args.num_samples != rescue_conditioning["repairs_per_candidate"]:
            raise ValueError(
                "num_samples differs from rescue repairs_per_candidate"
            )
        if args.batch_size != 1:
            raise ValueError(
                "rescue inference requires batch_size=1 for paired "
                "task-local RNG"
            )
        if args.limit != 0:
            raise ValueError(
                "rescue inference forbids --limit; seal the intended task "
                "subset into the conditioning plan"
            )
        all_plan_ids = {
            row["task_id"] for row in rescue_conditioning["rows"]
        }
        if not all_plan_ids <= set(selected_alignment_ids):
            raise ValueError(
                "rescue conditioning plan tasks differ from selected "
                "alignment role"
            )
        selected_plan_ids = {
            row["task_id"]
            for row in rescue_conditioning["rows"]
            if row["generate"]
        }
        selected_ids = [
            task_id for task_id in selected_ids if task_id in selected_plan_ids
        ]
        if set(selected_ids) != selected_plan_ids:
            raise ValueError(
                "rescue conditioning tasks differ from selected alignment role"
            )
    if args.limit > 0:
        selected_ids = selected_ids[: args.limit]
    output_ids = [
        str(row.get("id") or "") if isinstance(row, dict) else ""
        for row in outputs
    ]
    conditioning_sha_by_task = (
        {
            row["task_id"]: row["conditioning_sha256"]
            for row in rescue_conditioning["rows"]
            if row["generate"]
        }
        if rescue_conditioning is not None
        else {}
    )
    if (
        not selected_ids
        or output_ids != selected_ids
        or any(
            not _valid_inference_output_row(
                row,
                expected_id=task_id,
                num_samples=args.num_samples,
                rescue=rescue_conditioning is not None,
                vocab_size=int(contract.base_vocab_size or 0),
                conditioning_sha256=conditioning_sha_by_task.get(task_id),
            )
            for task_id, row in zip(selected_ids, outputs, strict=True)
        )
    ):
        raise ValueError("existing inference output task/sample coverage differs")
    expected_journal_contract = _build_inference_journal_contract(
        args=args,
        dataset_path=dataset_path,
        alignment_path=alignment_path,
        selected_ids=selected_ids,
        contract=contract,
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        model_config_sha256=contract.model_config_sha256,
        rescue_conditioning=rescue_conditioning,
    )
    journal_path = Path(
        args.journal or (str(output_path) + ".generation.journal.jsonl")
    ).expanduser().resolve()
    journal_events = load_journal(journal_path)
    terminals, complete, orphan = _inference_journal_state(
        journal_events,
        contract_payload=expected_journal_contract,
        rows=[
            {
                "identity": task_id,
                "conditioning_sha256": conditioning_sha_by_task.get(task_id),
            }
            for task_id in selected_ids
        ],
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    if not complete or orphan is not None:
        raise ValueError("existing inference journal is not complete")
    journal_outputs = [
        prediction
        for batch_index in range(len(terminals))
        for prediction in terminals[batch_index]
    ]
    if journal_outputs != outputs:
        raise ValueError(
            "existing inference output differs from its terminal journal"
        )
    completion_event = journal_events[-1]
    expected = {
        "schema": "direct-compact-inference-v1",
        "dataset_sha256": sha256_file(dataset_path),
        "alignment_sha256": sha256_file(alignment_path),
        "selected_role": args.role,
        "contract_sha256": sha256_file(args.contract),
        "codebook_sha256": sha256_file(args.codebook),
        "codec_sha256": sha256_file(args.codec_artifact),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_json),
        "decoder_model": decoder_model,
        "decoder_revision": decoder_revision,
        "model_config_sha256": contract.model_config_sha256,
        "attn_implementation": args.attn_implementation,
        "decoder_adapter": args.decoder_adapter or None,
        "decoder_adapter_sha256": (
            sha256_artifact(args.decoder_adapter)
            if args.decoder_adapter
            else None
        ),
        "source_overlay_sha256": sha256_file(args.source_overlay),
        "overlay_rows": len(contract.source_token_expansions),
        "lm_head_rows": int(contract.base_vocab_size or 0),
        "num_rows": len(selected_ids),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "direct_prompt_mode": args.direct_prompt_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "precision": (
            "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")
        ),
        "output_sha256": sha256_file(output_path),
        "generation_journal": journal_record(journal_path),
        "sampling_seed_policy": _provenance_seed_policy(
            rescue_conditioning
        ),
        "started_without_terminal_policy": (
            "retry_identical_seeded_batch_with_hash_chained_receipt"
        ),
        "resampled_slots": 0,
        "orphan_retry_events": int(
            completion_event["orphan_retry_events"]
        ),
        "orphan_recomputed_slots": int(
            completion_event["orphan_recomputed_slots"]
        ),
        "encoder": None,
        "soft_prefix": None,
    }
    if rescue_conditioning is not None:
        expected["rescue_conditioning"] = _rescue_plan_binding(
            rescue_conditioning
        )
        expected["prediction_token_ids_persisted"] = True
        expected["generation_vocab_size"] = int(
            contract.base_vocab_size or 0
        )
    mismatches = [
        key for key, value in expected.items()
        if provenance.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "existing inference provenance differs: " + ", ".join(mismatches)
        )
    return provenance


def _batch_seed(
    *, base_seed: int, batch_index: int, task_ids: list[str]
) -> int:
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": "direct-compact-inference-batch-seed-v1",
                "base_seed": int(base_seed),
                "batch_index": int(batch_index),
                "task_ids": task_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    # transformers.set_seed also seeds NumPy, whose legacy seed API requires
    # a 32-bit value.
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _rescue_task_seed(
    *,
    base_seed: int,
    source_plan_sha256: str,
    base_candidate_rank: int,
    task_id: str,
) -> int:
    """Pair a task's random draws across rescue arms and plan sparsity."""

    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": "direct-compact-rescue-task-seed-v1",
                "base_seed": int(base_seed),
                "source_plan_sha256": str(source_plan_sha256),
                "base_candidate_rank": int(base_candidate_rank),
                "task_id": str(task_id),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _generation_seed_policy(
    rescue_conditioning: dict[str, Any] | None,
) -> str:
    if rescue_conditioning is None:
        return "sha256(base_seed,batch_index,ordered_task_ids)"
    return (
        "sha256(base_seed,source_plan_sha256,"
        "base_candidate_rank,task_id)"
    )


def _provenance_seed_policy(
    rescue_conditioning: dict[str, Any] | None,
) -> str:
    if rescue_conditioning is None:
        return "independent_sha256_seed_per_ordered_task_batch"
    return _generation_seed_policy(rescue_conditioning)


def _scheduled_batch_seed(
    *,
    base_seed: int,
    batch_index: int,
    task_ids: list[str],
    rescue_conditioning: dict[str, Any] | None,
) -> int:
    if rescue_conditioning is None:
        return _batch_seed(
            base_seed=base_seed,
            batch_index=batch_index,
            task_ids=task_ids,
        )
    if len(task_ids) != 1:
        raise ValueError(
            "rescue generation seed requires exactly one task per batch"
        )
    return _rescue_task_seed(
        base_seed=base_seed,
        source_plan_sha256=rescue_conditioning["source_plan_sha256"],
        base_candidate_rank=rescue_conditioning["base_candidate_rank"],
        task_id=task_ids[0],
    )


def _trim_generated_token_ids(
    token_ids: list[int],
    *,
    eos_token_id: int | list[int] | tuple[int, ...] | None,
    pad_token_id: int | None,
) -> list[int]:
    """Keep the exact generated completion through its first EOS.

    Batched ``generate`` right-pads shorter returned sequences. Persisting the
    padded matrix would make later on-policy log-probability replay train on
    synthetic padding, so rescue runs retain only the real completion.
    """

    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise ValueError("generated rescue completion has invalid token IDs")
    normalized = list(token_ids)
    if isinstance(eos_token_id, int) and not isinstance(eos_token_id, bool):
        eos_ids = {eos_token_id}
    elif isinstance(eos_token_id, (list, tuple)) and all(
        isinstance(token, int)
        and not isinstance(token, bool)
        and token >= 0
        for token in eos_token_id
    ):
        eos_ids = set(eos_token_id)
    else:
        eos_ids = set()
    eos_positions = [
        position
        for position, token in enumerate(normalized)
        if token in eos_ids
    ]
    if eos_positions:
        return normalized[: eos_positions[0] + 1]
    if pad_token_id is not None:
        while normalized and normalized[-1] == pad_token_id:
            normalized.pop()
    if not normalized:
        raise ValueError("generated rescue completion contains no token IDs")
    return normalized


def _valid_prediction_token_ids(
    row: Any,
    *,
    num_samples: int,
    vocab_size: int | None,
) -> bool:
    if not isinstance(row, dict):
        return False
    values = row.get("prediction_token_ids")
    if not isinstance(values, list) or len(values) != num_samples:
        return False
    if (
        vocab_size is not None
        and (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or vocab_size <= 0
        )
    ):
        return False
    return all(
        isinstance(token_ids, list)
        and bool(token_ids)
        and all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and token >= 0
            and (vocab_size is None or token < vocab_size)
            for token in token_ids
        )
        for token_ids in values
    )


def _valid_inference_output_row(
    row: Any,
    *,
    expected_id: str,
    num_samples: int,
    rescue: bool,
    vocab_size: int | None,
    conditioning_sha256: str | None,
) -> bool:
    if not isinstance(row, dict):
        return False
    expected_fields = {"id", "predictions"}
    if rescue:
        expected_fields.update(
            {"prediction_token_ids", "conditioning_sha256"}
        )
    if set(row) != expected_fields or str(row.get("id") or "") != expected_id:
        return False
    predictions = row.get("predictions")
    if (
        not isinstance(predictions, list)
        or len(predictions) != num_samples
        or any(not isinstance(value, str) for value in predictions)
    ):
        return False
    if not rescue:
        return True
    return (
        isinstance(conditioning_sha256, str)
        and row.get("conditioning_sha256") == conditioning_sha256
        and _valid_prediction_token_ids(
            row,
            num_samples=num_samples,
            vocab_size=vocab_size,
        )
    )


def _inference_batch_seal(
    *,
    batch_index: int,
    task_ids: list[str],
    batch_seed: int,
    slot_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema": "direct-compact-inference-sealed-batch-v1",
        "batch_index": int(batch_index),
        "task_ids": list(task_ids),
        "batch_seed": int(batch_seed),
        "slot_ids": list(slot_ids),
    }


def make_inference_orphan_retry_event(
    orphan: dict[str, Any],
) -> dict[str, Any]:
    """Authorize one exact recomputation of a durably started local batch."""

    started = orphan["started"]
    retries = list(orphan["retries"])
    batch_seal = dict(orphan["batch_seal"])
    previous_attempt_sha256 = (
        retries[-1]["journal_event_sha256"]
        if retries
        else started["journal_event_sha256"]
    )
    return {
        "event": "inference_batch_orphan_retry",
        "schema": INFERENCE_JOURNAL_SCHEMA,
        "batch_index": int(started["batch_index"]),
        "retry_index": len(retries) + 1,
        "started_event_sha256": started["journal_event_sha256"],
        "previous_attempt_event_sha256": previous_attempt_sha256,
        "sealed_batch": batch_seal,
        "sealed_batch_sha256": canonical_sha256(batch_seal),
        "completed_terminal_batches_preserved": int(
            orphan["completed_terminal_batches"]
        ),
        "recovery_reason": "process_interrupted_after_durable_batch_start",
        "recompute_identical_seeded_batch": True,
        "resample_new_random_draws": False,
    }


def _inference_journal_state(
    events: list[dict[str, Any]],
    *,
    contract_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    batch_size: int,
    num_samples: int,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    bool,
    dict[str, Any] | None,
]:
    """Validate completed batches and expose only a trailing sealed orphan."""

    if not events:
        return {}, False, None
    if (
        events[0].get("event") != "inference_header"
        or events[0].get("schema") != INFERENCE_JOURNAL_SCHEMA
        or events[0].get("contract") != contract_payload
        or events[0].get("contract_sha256")
        != canonical_sha256(contract_payload)
    ):
        raise ValueError("inference journal header differs from exact run contract")
    terminals: dict[int, list[dict[str, Any]]] = {}
    retry_event_count = 0
    retry_slot_count = 0
    cursor = 1
    batch_count = (len(rows) + batch_size - 1) // batch_size
    while cursor < len(events):
        event = events[cursor]
        if event.get("event") == "inference_complete":
            if (
                event.get("schema") != INFERENCE_JOURNAL_SCHEMA
                or cursor != len(events) - 1
                or len(terminals) != batch_count
            ):
                raise ValueError("inference completion event is premature")
            ordered = [
                prediction
                for index in range(batch_count)
                for prediction in terminals[index]
            ]
            if (
                event.get("outputs_canonical_sha256")
                != canonical_sha256(ordered)
                or int(event.get("rows", -1)) != len(rows)
                or int(event.get("slots", -1))
                != len(rows) * num_samples
                or int(event.get("resampled_slots", -1)) != 0
                or int(event.get("orphan_retry_events", -1))
                != retry_event_count
                or int(event.get("orphan_recomputed_slots", -1))
                != retry_slot_count
            ):
                raise ValueError("inference completion digest/count differs")
            return terminals, True, None
        batch_index = len(terminals)
        expected_batch = rows[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        expected_ids = [str(row["identity"]) for row in expected_batch]
        expected_seed = _scheduled_batch_seed(
            base_seed=int(contract_payload["seed"]),
            batch_index=batch_index,
            task_ids=expected_ids,
            rescue_conditioning=contract_payload.get("rescue_conditioning"),
        )
        expected_slots = [
            f"{task_id}:{sample_index}"
            for task_id in expected_ids
            for sample_index in range(num_samples)
        ]
        expected_seal = _inference_batch_seal(
            batch_index=batch_index,
            task_ids=expected_ids,
            batch_seed=expected_seed,
            slot_ids=expected_slots,
        )
        if (
            event.get("event") != "inference_batch_started"
            or event.get("schema") != INFERENCE_JOURNAL_SCHEMA
            or int(event.get("batch_index", -1)) != batch_index
            or event.get("task_ids") != expected_ids
            or int(event.get("batch_seed", -1)) != expected_seed
            or event.get("slot_ids") != expected_slots
        ):
            raise ValueError("inference journal batch-start schedule differs")
        started = event
        cursor += 1
        retries: list[dict[str, Any]] = []
        previous_attempt_sha256 = str(
            started.get("journal_event_sha256") or ""
        )
        while (
            cursor < len(events)
            and events[cursor].get("event")
            == "inference_batch_orphan_retry"
        ):
            retry = events[cursor]
            if (
                retry.get("schema") != INFERENCE_JOURNAL_SCHEMA
                or int(retry.get("batch_index", -1)) != batch_index
                or int(retry.get("retry_index", -1)) != len(retries) + 1
                or retry.get("started_event_sha256")
                != started.get("journal_event_sha256")
                or retry.get("previous_attempt_event_sha256")
                != previous_attempt_sha256
                or retry.get("sealed_batch") != expected_seal
                or retry.get("sealed_batch_sha256")
                != canonical_sha256(expected_seal)
                or int(
                    retry.get(
                        "completed_terminal_batches_preserved", -1
                    )
                )
                != len(terminals)
                or retry.get("recovery_reason")
                != "process_interrupted_after_durable_batch_start"
                or retry.get("recompute_identical_seeded_batch") is not True
                or retry.get("resample_new_random_draws") is not False
            ):
                raise ValueError(
                    "inference orphan-retry receipt is inconsistent"
                )
            previous_attempt_sha256 = str(
                retry.get("journal_event_sha256") or ""
            )
            retries.append(retry)
            retry_event_count += 1
            retry_slot_count += len(expected_slots)
            cursor += 1
        if cursor >= len(events):
            return (
                terminals,
                False,
                {
                    "batch_index": batch_index,
                    "started": started,
                    "retries": retries,
                    "batch_seal": expected_seal,
                    "completed_terminal_batches": len(terminals),
                },
            )
        terminal = events[cursor]
        predictions = terminal.get("predictions")
        expected_latest_retry = (
            retries[-1].get("journal_event_sha256") if retries else None
        )
        if (
            terminal.get("event") != "inference_batch_terminal"
            or terminal.get("schema") != INFERENCE_JOURNAL_SCHEMA
            or int(terminal.get("batch_index", -1)) != batch_index
            or terminal.get("started_event_sha256")
            != started.get("journal_event_sha256")
            or int(terminal.get("retry_count", 0)) != len(retries)
            or terminal.get("latest_retry_event_sha256")
            != expected_latest_retry
            or not isinstance(predictions, list)
            or [str(row.get("id") or "") for row in predictions]
            != expected_ids
            or any(
                not _valid_inference_output_row(
                    prediction,
                    expected_id=expected_row["identity"],
                    num_samples=num_samples,
                    rescue=(
                        contract_payload.get("rescue_conditioning")
                        is not None
                    ),
                    vocab_size=contract_payload.get(
                        "generation_vocab_size"
                    ),
                    conditioning_sha256=expected_row.get(
                        "conditioning_sha256"
                    ),
                )
                for expected_row, prediction in zip(
                    expected_batch,
                    predictions,
                    strict=True,
                )
            )
            or terminal.get("predictions_canonical_sha256")
            != canonical_sha256(predictions)
        ):
            raise ValueError("inference journal batch terminal is inconsistent")
        terminals[batch_index] = predictions
        cursor += 1
    return terminals, False, None


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.num_samples > 1 and args.temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    if args.top_k < 0:
        raise ValueError("top_k must be non-negative")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    rescue_conditioning = (
        load_rescue_conditioning_plan(args.rescue_conditioning_plan)
        if args.rescue_conditioning_plan
        else None
    )
    if (
        rescue_conditioning is not None
        and args.num_samples
        != rescue_conditioning["repairs_per_candidate"]
    ):
        raise ValueError(
            "num_samples must equal rescue repairs_per_candidate"
        )
    if rescue_conditioning is not None and args.batch_size != 1:
        raise ValueError(
            "rescue inference requires batch_size=1 for paired task-local RNG"
        )
    if rescue_conditioning is not None and args.limit != 0:
        raise ValueError(
            "rescue inference forbids --limit; seal the intended task subset "
            "into the conditioning plan"
        )
    output_path = Path(args.output).expanduser().resolve()
    provenance_path = Path(str(output_path) + ".provenance.json")
    journal_path = Path(
        args.journal or (str(output_path) + ".generation.journal.jsonl")
    ).expanduser().resolve()
    args.journal = str(journal_path)
    if output_path.is_file() and provenance_path.is_file():
        validate_existing_inference(args)
        print(f"DIRECT_COMPACT_INFERENCE_REUSED output={output_path}", flush=True)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    contract = DirectCompactContract.load(args.contract)
    if args.max_new_tokens > contract.max_target_tokens:
        raise ValueError("max_new_tokens exceeds the sealed target-token budget")
    decoder_model = args.decoder_model.strip() or contract.decoder_model
    decoder_revision = args.decoder_revision.strip() or contract.decoder_revision
    decoder_config_path = resolve_decoder_config_path(decoder_model, decoder_revision)
    contract.validate_decoder_binding(
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        model_config_path=decoder_config_path,
    )
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
        tokenizer.pad_token = tokenizer.eos_token
    contract.validate_artifacts(
        tokenizer=tokenizer,
        tokenizer_json_path=args.tokenizer_json,
        codec_path=args.codec_artifact,
        codebook_path=args.codebook,
    )
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        decoder_model,
        revision=decoder_revision,
        **model_kwargs,
    )
    validate_base_model_vocab(model, contract)
    if args.decoder_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.decoder_adapter)
    model.to(args.device)
    overlay = restore_source_embedding_overlay(
        model,
        dict(contract.source_token_expansions),
        args.source_overlay,
        base_vocab_size=int(contract.base_vocab_size or 0),
    )
    if model.get_output_embeddings().weight.size(0) != contract.base_vocab_size:
        raise RuntimeError("inference setup unexpectedly resized the LM head")
    model.eval()

    dataset_path = Path(args.dataset).resolve()
    alignment_path = Path(args.alignment).resolve()
    rows = load_rows(
        dataset_path,
        alignment_path,
        contract,
        tokenizer,
        target_budget=args.max_new_tokens,
        role=args.role,
        direct_prompt_mode=args.direct_prompt_mode,
        rescue_conditioning=rescue_conditioning,
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    batch_size = max(1, int(args.batch_size))
    slot_ids = [
        f"{row['identity']}:{sample_index}"
        for row in rows
        for sample_index in range(args.num_samples)
    ]
    require_unique_slots(slot_ids)
    journal_contract = _build_inference_journal_contract(
        args=args,
        dataset_path=dataset_path,
        alignment_path=alignment_path,
        selected_ids=[str(row["identity"]) for row in rows],
        contract=contract,
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        model_config_sha256=sha256_file(decoder_config_path),
        rescue_conditioning=rescue_conditioning,
    )
    events = load_journal(journal_path)
    if not events:
        if output_path.exists() or provenance_path.exists():
            raise ValueError(
                "partial inference artifacts exist without a durable journal"
            )
        append_event(
            journal_path,
            {
                "event": "inference_header",
                "schema": INFERENCE_JOURNAL_SCHEMA,
                "contract": journal_contract,
                "contract_sha256": canonical_sha256(journal_contract),
            },
        )
        events = load_journal(journal_path)
    terminals, complete, orphan = _inference_journal_state(
        events,
        contract_payload=journal_contract,
        rows=rows,
        batch_size=batch_size,
        num_samples=args.num_samples,
    )
    batch_count = (len(rows) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_index in range(len(terminals), batch_count):
            start = batch_index * batch_size
            batch = rows[start : start + batch_size]
            task_ids = [str(row["identity"]) for row in batch]
            batch_seed = _scheduled_batch_seed(
                base_seed=args.seed,
                batch_index=batch_index,
                task_ids=task_ids,
                rescue_conditioning=rescue_conditioning,
            )
            expected_slot_ids = [
                f"{task_id}:{sample_index}"
                for task_id in task_ids
                for sample_index in range(args.num_samples)
            ]
            latest_retry: dict[str, Any] | None = None
            if orphan is not None:
                if int(orphan.get("batch_index", -1)) != batch_index:
                    raise RuntimeError(
                        "inference orphan is not the next incomplete batch"
                    )
                started = dict(orphan["started"])
                latest_retry = append_event(
                    journal_path,
                    make_inference_orphan_retry_event(orphan),
                )
                retry_count = len(orphan["retries"]) + 1
                orphan = None
            else:
                started = append_event(
                    journal_path,
                    {
                        "event": "inference_batch_started",
                        "schema": INFERENCE_JOURNAL_SCHEMA,
                        "batch_index": batch_index,
                        "task_ids": task_ids,
                        "batch_seed": batch_seed,
                        "slot_ids": expected_slot_ids,
                    },
                )
                retry_count = 0
            # Every batch has its own sealed seed, so completed batches consume
            # no hidden RNG state needed by a resumed later batch.
            set_seed(batch_seed)
            sequences = [
                row["prompt"] + row["compact"] + row["conditioning"]
                for row in batch
            ]
            width = max(map(len, sequences))
            input_ids = []
            attention_mask = []
            for sequence in sequences:
                padding = width - len(sequence)
                input_ids.append([tokenizer.pad_token_id] * padding + sequence)
                attention_mask.append([0] * padding + [1] * len(sequence))
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=args.device)
            mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=args.device)
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.num_samples > 1,
                "num_return_sequences": args.num_samples,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if args.num_samples > 1:
                generation_kwargs.update(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            generated = model.generate(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                **generation_kwargs,
            )
            generated = generated[:, width:]
            generated_rows = [
                _trim_generated_token_ids(
                    [int(token) for token in sequence],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for sequence in generated.tolist()
            ]
            decoded = tokenizer.batch_decode(
                generated_rows, skip_special_tokens=True
            )
            batch_predictions: list[dict[str, Any]] = []
            for row_index, row in enumerate(batch):
                offset = row_index * args.num_samples
                predictions = decoded[offset : offset + args.num_samples]
                output_row: dict[str, Any] = {
                    "id": row["identity"],
                    "predictions": predictions,
                }
                if rescue_conditioning is not None:
                    output_row["prediction_token_ids"] = generated_rows[
                        offset : offset + args.num_samples
                    ]
                    output_row["conditioning_sha256"] = row[
                        "conditioning_sha256"
                    ]
                batch_predictions.append(output_row)
            append_event(
                journal_path,
                {
                    "event": "inference_batch_terminal",
                    "schema": INFERENCE_JOURNAL_SCHEMA,
                    "batch_index": int(started["batch_index"]),
                    "started_event_sha256": started["journal_event_sha256"],
                    "retry_count": retry_count,
                    "latest_retry_event_sha256": (
                        None
                        if latest_retry is None
                        else latest_retry["journal_event_sha256"]
                    ),
                    "predictions": batch_predictions,
                    "predictions_canonical_sha256": canonical_sha256(
                        batch_predictions
                    ),
                },
            )
            terminals[int(started["batch_index"])] = batch_predictions
    outputs = [
        prediction
        for batch_index in range(batch_count)
        for prediction in terminals[batch_index]
    ]
    if not complete:
        events_before_complete = load_journal(journal_path)
        orphan_retry_events = [
            event
            for event in events_before_complete
            if event.get("event") == "inference_batch_orphan_retry"
        ]
        append_event(
            journal_path,
            {
                "event": "inference_complete",
                "schema": INFERENCE_JOURNAL_SCHEMA,
                "rows": len(rows),
                "slots": len(slot_ids),
                "outputs_canonical_sha256": canonical_sha256(outputs),
                "resampled_slots": 0,
                "orphan_retry_events": len(orphan_retry_events),
                "orphan_recomputed_slots": sum(
                    len((event.get("sealed_batch") or {}).get("slot_ids") or [])
                    for event in orphan_retry_events
                ),
            },
        )
    final_events = load_journal(journal_path)
    terminals, complete, orphan = _inference_journal_state(
        final_events,
        contract_payload=journal_contract,
        rows=rows,
        batch_size=batch_size,
        num_samples=args.num_samples,
    )
    if not complete or orphan is not None:
        raise RuntimeError("inference journal did not reach completion")
    completion_event = final_events[-1]
    outputs = [
        prediction
        for batch_index in range(batch_count)
        for prediction in terminals[batch_index]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require_exact_or_write(output_path, outputs)
    provenance = {
        "schema": "direct-compact-inference-v1",
        "dataset_sha256": sha256_file(dataset_path),
        "alignment_sha256": sha256_file(alignment_path),
        "selected_role": args.role,
        "contract_sha256": sha256_file(args.contract),
        "codebook_sha256": sha256_file(args.codebook),
        "codec_sha256": sha256_file(args.codec_artifact),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_json),
        "decoder_model": decoder_model,
        "decoder_revision": decoder_revision,
        "model_config_sha256": sha256_file(decoder_config_path),
        "attn_implementation": args.attn_implementation,
        "decoder_adapter": args.decoder_adapter or None,
        "decoder_adapter_sha256": (
            sha256_artifact(args.decoder_adapter) if args.decoder_adapter else None
        ),
        "source_overlay_sha256": sha256_file(args.source_overlay),
        "overlay_rows": int(overlay.source_embeddings.num_embeddings),
        "lm_head_rows": int(model.get_output_embeddings().weight.size(0)),
        "num_rows": len(rows),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "direct_prompt_mode": args.direct_prompt_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "precision": (
            "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")
        ),
        "output_sha256": sha256_file(output_path),
        "generation_journal": journal_record(journal_path),
        "sampling_seed_policy": _provenance_seed_policy(
            rescue_conditioning
        ),
        "started_without_terminal_policy": (
            "retry_identical_seeded_batch_with_hash_chained_receipt"
        ),
        "resampled_slots": 0,
        "orphan_retry_events": int(
            completion_event["orphan_retry_events"]
        ),
        "orphan_recomputed_slots": int(
            completion_event["orphan_recomputed_slots"]
        ),
        "encoder": None,
        "soft_prefix": None,
    }
    if rescue_conditioning is not None:
        provenance["rescue_conditioning"] = _rescue_plan_binding(
            rescue_conditioning
        )
        provenance["prediction_token_ids_persisted"] = True
        provenance["generation_vocab_size"] = int(
            contract.base_vocab_size or 0
        )
    require_exact_or_write(provenance_path, provenance)
    validate_existing_inference(args)


if __name__ == "__main__":
    main()
