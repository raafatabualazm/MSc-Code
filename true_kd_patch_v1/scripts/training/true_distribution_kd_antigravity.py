#!/usr/bin/env python3
"""Direct-compact knowledge distillation with explicit, non-interchangeable modes.

Modes
-----
``dense_full_kl``
    Loads a frozen local teacher, requires byte-identical teacher/student
    tokenizers and equal output vocabularies, and computes full-vocabulary
    ``KL(teacher || student)`` at every target position (including EOS).

``sparse_topk_tail_kl``
    Consumes a strictly sealed offline artifact containing top-k token IDs,
    unrounded natural logprobs, and an explicit aggregate tail mass.  This is a
    coarsened KL lower bound and is never reported as full KD.

The script is specialized for the repository's DirectCompactCausalLM contract.
It deliberately refuses legacy API artifacts containing decoded token strings
without sealed tokenizer identity, token IDs, tail mass, and EOS.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_ROOT = SCRIPT_DIR.parents[1]
DATA_SCRIPT_DIR = PATCH_ROOT / "scripts" / "data"
for import_path in (SCRIPT_DIR, DATA_SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from kd_objectives_antigravity import (  # noqa: E402
    accumulation_window_size,
    dense_forward_kl,
    is_accumulation_boundary,
    sparse_topk_tail_forward_kl,
)
from validate_qwen_kd_artifacts import (  # noqa: E402
    sha256_file,
    validate_sparse_topk_tail_dataset,
)


MODES = ("dense_full_kl", "sparse_topk_tail_kl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--project_root",
        required=True,
        help="Path containing models/ and scripts/ from hybrid_training_patch_v2_3",
    )
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--train_seal", default="")
    parser.add_argument("--sparse_manifest", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codec_artifact", required=True)

    parser.add_argument("--student_model", default="")
    parser.add_argument("--student_revision", default="")
    parser.add_argument("--student_tokenizer", default="")
    parser.add_argument("--student_tokenizer_revision", default="")
    parser.add_argument("--student_tokenizer_json", required=True)
    parser.add_argument("--student_warmstart_checkpoint", default="")

    parser.add_argument("--teacher_model", default="")
    parser.add_argument("--teacher_revision", default="")
    parser.add_argument("--teacher_tokenizer", default="")
    parser.add_argument("--teacher_tokenizer_revision", default="")
    parser.add_argument("--teacher_tokenizer_json", default="")
    parser.add_argument("--teacher_checkpoint", default="")

    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--student_load_4bit", action="store_true")
    parser.add_argument("--teacher_load_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--position_chunk_size", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--hard_ce_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_every_steps", type=int, default=0)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "dense_full_kl":
        required = {
            "--train_seal": args.train_seal,
            "--teacher_model": args.teacher_model,
            "--teacher_tokenizer_json": args.teacher_tokenizer_json,
            "--teacher_checkpoint": args.teacher_checkpoint,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"dense_full_kl requires: {', '.join(missing)}")
    else:
        if not args.sparse_manifest:
            raise ValueError("sparse_topk_tail_kl requires --sparse_manifest")
        if args.teacher_checkpoint or args.teacher_model:
            raise ValueError(
                "sparse_topk_tail_kl uses sealed offline distributions, not a "
                "local --teacher_model/--teacher_checkpoint"
            )
        if abs(args.temperature - 1.0) > 1e-12:
            raise ValueError(
                "sparse_topk_tail_kl requires --temperature 1; a hidden aggregate "
                "tail cannot be re-tempered exactly"
            )
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or positive")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("batch_size and grad_accum must be positive")
    if args.position_chunk_size <= 0:
        raise ValueError("position_chunk_size must be positive")
    if args.logging_steps <= 0 or args.save_every_steps < 0:
        raise ValueError("invalid logging/save interval")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer hyperparameters")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must lie in [0, 1)")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        raise ValueError("temperature must be finite and positive")
    if args.kd_weight <= 0 or args.hard_ce_weight < 0:
        raise ValueError("kd_weight must be positive and hard_ce_weight non-negative")
    if args.lora_r <= 0 or args.lora_alpha <= 0 or args.lora_dropout < 0:
        raise ValueError("invalid LoRA configuration")
    if args.student_load_4bit and args.dtype == "fp32":
        raise ValueError("4-bit student training requires bf16 or fp16 compute")


def _hash_directory(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        return sha256_file(root)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"{root}: artifact directory is empty")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _canonical_json_sha256(path: str | Path) -> str:
    """Hash JSON semantics independently of whitespace/key serialization."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _ensure_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{path}: output path exists and is not a directory")
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"{path}: output directory is non-empty; use a new distinct output "
            "directory so an earlier run cannot be mixed with this one"
        )
    path.mkdir(parents=True, exist_ok=True)


def _checkpoint_layout(
    checkpoint: str | Path, *, identity: str
) -> dict[str, Path]:
    root = Path(checkpoint).resolve()
    result = {
        "root": root,
        "adapter": root / "decoder_adapter",
        "overlay": root / "source_embedding_overlay.pt",
        "tokenizer_json": root / "tokenizer" / "tokenizer.json",
        "contract": root / "compact_contract.json",
    }
    missing = [name for name, path in result.items() if name != "root" and not path.exists()]
    if missing:
        raise ValueError(f"{identity} checkpoint is missing: {', '.join(missing)}")
    if not result["adapter"].is_dir():
        raise ValueError(f"{identity} decoder_adapter must be a directory")
    return result


def _validate_checkpoint_binding(
    layout: Mapping[str, Path],
    *,
    identity: str,
    contract_path: str | Path,
    tokenizer_sha256: str,
) -> None:
    if _canonical_json_sha256(layout["contract"]) != _canonical_json_sha256(
        contract_path
    ):
        raise ValueError(f"{identity} checkpoint compact contract differs semantically")
    checkpoint_contract = json.loads(
        layout["contract"].read_text(encoding="utf-8")
    )
    if checkpoint_contract.get("tokenizer_json_sha256") != tokenizer_sha256:
        raise ValueError(
            f"{identity} checkpoint contract binds a different tokenizer JSON"
        )
    # The checkpoint tokenizer is a save_pretrained serialization and is not
    # used to tokenize KD inputs. Transformers may rewrite ByteLevel decoder
    # defaults while saving it, so its file bytes are not a valid training
    # identity. Still require it to be valid JSON for checkpoint completeness.
    json.loads(layout["tokenizer_json"].read_text(encoding="utf-8"))


def _tokenizer_signature(tokenizer: Any) -> dict[str, Any]:
    return {
        "vocab": tokenizer.get_vocab(),
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "additional_special_tokens_ids": list(
            getattr(tokenizer, "additional_special_tokens_ids", []) or []
        ),
    }


def _dtype(args: argparse.Namespace) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]


def _model_kwargs(
    *,
    args: argparse.Namespace,
    load_4bit: bool,
    device: torch.device,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_dtype(args),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": str(device)}
    else:
        kwargs["torch_dtype"] = _dtype(args)
    return kwargs


def _load_teacher(
    args: argparse.Namespace,
    *,
    contract: Any,
    layout: Mapping[str, Path],
    device: torch.device,
    restore_source_embedding_overlay: Any,
    validate_base_model_vocab: Any,
) -> tuple[torch.nn.Module, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        revision=args.teacher_revision or None,
        **_model_kwargs(args=args, load_4bit=args.teacher_load_4bit, device=device),
    )
    validate_base_model_vocab(base, contract)
    teacher = PeftModel.from_pretrained(
        base, str(layout["adapter"]), is_trainable=False
    )
    if not args.teacher_load_4bit:
        teacher.to(device)
    overlay = restore_source_embedding_overlay(
        teacher,
        dict(contract.source_token_expansions),
        layout["overlay"],
        base_vocab_size=int(contract.base_vocab_size),
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False
    return teacher, overlay


def _load_student(
    args: argparse.Namespace,
    *,
    contract: Any,
    decoder_model: str,
    decoder_revision: str,
    warmstart_layout: Mapping[str, Path] | None,
    device: torch.device,
    install_source_embedding_overlay: Any,
    restore_source_embedding_overlay: Any,
    validate_base_model_vocab: Any,
) -> tuple[torch.nn.Module, Any]:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        decoder_model,
        revision=decoder_revision or None,
        **_model_kwargs(args=args, load_4bit=args.student_load_4bit, device=device),
    )
    validate_base_model_vocab(base, contract)
    if args.student_load_4bit:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if warmstart_layout is not None:
        student = PeftModel.from_pretrained(
            base, str(warmstart_layout["adapter"]), is_trainable=True
        )
    else:
        student = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    if not args.student_load_4bit:
        student.to(device)
    if warmstart_layout is not None:
        overlay = restore_source_embedding_overlay(
            student,
            dict(contract.source_token_expansions),
            warmstart_layout["overlay"],
            base_vocab_size=int(contract.base_vocab_size),
        )
    else:
        overlay = install_source_embedding_overlay(
            student,
            dict(contract.source_token_expansions),
            base_vocab_size=int(contract.base_vocab_size),
        )
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    if hasattr(student.config, "use_cache"):
        student.config.use_cache = False
    student.train()
    return student, overlay


def _causal_body_and_head(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(base, "model", None)
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(body, torch.nn.Module) or not isinstance(head, torch.nn.Module):
        raise TypeError(
            "dense KD needs a decoder model exposing `.model` hidden states and "
            "`get_output_embeddings()` so logits can be restricted to target positions"
        )
    return body, head


def _supervised_hidden_states(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    body, _ = _causal_body_and_head(model)
    output = body(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state
    shifted_labels = batch["labels"][:, 1:]
    supervised = shifted_labels.ne(-100)
    if not torch.any(supervised):
        raise ValueError("batch contains no supervised next-token positions")
    return hidden[:, :-1, :][supervised], shifted_labels[supervised]


class SparseCompactDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        tokenizer: Any,
        contract: Any,
        direct_prompt: Any,
        encode: Any,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        eos_id = tokenizer.eos_token_id
        for index, row in enumerate(rows):
            identity = str(row.get("task_id") or row.get("id") or f"row-{index}")
            compact_ids = contract.validate_row(row, identity)
            target_ids = [int(value) for value in row["target_input_ids"]]
            if eos_id is None or target_ids[-1] != int(eos_id):
                raise ValueError(f"{identity}: target does not end with tokenizer EOS")
            prompt_ids = encode(
                tokenizer,
                direct_prompt(
                    row,
                    target_function=contract.target_function,
                    target_language=contract.target_language,
                ),
                special=True,
            )
            if len(compact_ids) > contract.max_source_tokens:
                raise ValueError(f"{identity}: source exceeds the contract limit")
            if len(target_ids) > contract.max_target_tokens:
                raise ValueError(f"{identity}: target exceeds the contract limit")
            if len(prompt_ids) + len(compact_ids) + len(target_ids) > contract.max_total_tokens:
                raise ValueError(f"{identity}: row exceeds the contract context limit")
            self.rows.append(
                {
                    "decoder_prompt_input_ids": prompt_ids,
                    "compact_input_ids": compact_ids,
                    "target_input_ids": target_ids,
                    "teacher_positions": row["teacher_positions"],
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class SparseCompactCollator:
    def __init__(self, base_collator: Any) -> None:
        self.base_collator = base_collator

    def __call__(
        self, features: Sequence[Mapping[str, Any]]
    ) -> dict[str, torch.Tensor]:
        batch = self.base_collator(features)
        max_target = max(len(feature["teacher_positions"]) for feature in features)
        max_topk = max(
            len(position["top_token_ids"])
            for feature in features
            for position in feature["teacher_positions"]
        )
        top_ids = torch.full(
            (len(features), max_target, max_topk), -1, dtype=torch.long
        )
        top_logprobs = torch.zeros(
            (len(features), max_target, max_topk), dtype=torch.float64
        )
        top_mask = torch.zeros(
            (len(features), max_target, max_topk), dtype=torch.bool
        )
        tail_mass = torch.zeros((len(features), max_target), dtype=torch.float64)
        position_mask = torch.zeros(
            (len(features), max_target), dtype=torch.bool
        )
        for row_index, feature in enumerate(features):
            for position_index, position in enumerate(feature["teacher_positions"]):
                ids = position["top_token_ids"]
                logprobs = position["top_logprobs"]
                count = len(ids)
                top_ids[row_index, position_index, :count] = torch.tensor(
                    ids, dtype=torch.long
                )
                top_logprobs[row_index, position_index, :count] = torch.tensor(
                    logprobs, dtype=torch.float64
                )
                top_mask[row_index, position_index, :count] = True
                tail_mass[row_index, position_index] = float(position["tail_mass"])
                position_mask[row_index, position_index] = True
        batch.update(
            {
                "teacher_top_token_ids": top_ids,
                "teacher_top_logprobs": top_logprobs,
                "teacher_top_mask": top_mask,
                "teacher_tail_mass": tail_mass,
                "teacher_position_mask": position_mask,
            }
        )
        return batch


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _autocast(args: argparse.Namespace, device: torch.device) -> Any:
    if args.dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=_dtype(args),
        enabled=True,
    )


def _batch_loss(
    args: argparse.Namespace,
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    with _autocast(args, batch["input_ids"].device):
        student_hidden, target_ids = _supervised_hidden_states(student, batch)
    if teacher is not None:
        with torch.no_grad(), _autocast(args, batch["input_ids"].device):
            teacher_hidden, teacher_targets = _supervised_hidden_states(teacher, batch)
        if not torch.equal(target_ids, teacher_targets):
            raise RuntimeError("teacher/student supervised target alignment changed")
    else:
        teacher_hidden = None

    _, student_head = _causal_body_and_head(student)
    teacher_head = (
        None if teacher is None else _causal_body_and_head(teacher)[1]
    )
    position_count = int(target_ids.numel())
    kd_sum: torch.Tensor | None = None
    ce_sum: torch.Tensor | None = None

    if args.mode == "sparse_topk_tail_kl":
        position_mask = batch["teacher_position_mask"]
        top_ids = batch["teacher_top_token_ids"][position_mask]
        top_logprobs = batch["teacher_top_logprobs"][position_mask]
        top_mask = batch["teacher_top_mask"][position_mask]
        tail_mass = batch["teacher_tail_mass"][position_mask]
        if top_ids.size(0) != position_count:
            raise RuntimeError("sparse distributions do not align with labels")

    for start in range(0, position_count, args.position_chunk_size):
        end = min(position_count, start + args.position_chunk_size)

        if args.mode == "dense_full_kl":
            assert teacher_hidden is not None and teacher_head is not None

            def dense_chunk(
                student_hidden_chunk: torch.Tensor,
                teacher_hidden_chunk: torch.Tensor,
                target_chunk: torch.Tensor,
            ) -> torch.Tensor:
                with _autocast(args, student_hidden_chunk.device):
                    student_logits = student_head(student_hidden_chunk).float()
                with torch.no_grad(), _autocast(
                    args, teacher_hidden_chunk.device
                ):
                    teacher_logits = teacher_head(teacher_hidden_chunk).float()
                if teacher_logits.size(-1) != student_logits.size(-1):
                    raise RuntimeError("teacher/student output vocabularies differ")
                kd_value = dense_forward_kl(
                    student_logits,
                    teacher_logits,
                    temperature=args.temperature,
                    reduction="sum",
                )
                ce_value = (
                    F.cross_entropy(
                        student_logits,
                        target_chunk,
                        reduction="sum",
                    )
                    if args.hard_ce_weight > 0
                    else student_logits.new_zeros(())
                )
                return torch.stack((kd_value, ce_value))

            # Without checkpointing, every [chunk, vocabulary] student softmax
            # remains live until the final backward and chunking does not actually
            # bound memory. Recompute each frozen-teacher/student-head chunk during
            # backward so only hidden states, not all full-vocabulary tensors, are
            # retained.
            chunk_values = torch.utils.checkpoint.checkpoint(
                dense_chunk,
                student_hidden[start:end],
                teacher_hidden[start:end],
                target_ids[start:end],
                use_reentrant=False,
            )
        else:

            def sparse_chunk(
                student_hidden_chunk: torch.Tensor,
                target_chunk: torch.Tensor,
                top_ids_chunk: torch.Tensor,
                top_logprobs_chunk: torch.Tensor,
                tail_mass_chunk: torch.Tensor,
                top_mask_chunk: torch.Tensor,
            ) -> torch.Tensor:
                with _autocast(args, student_hidden_chunk.device):
                    student_logits = student_head(student_hidden_chunk).float()
                kd_value = sparse_topk_tail_forward_kl(
                    student_logits,
                    top_ids_chunk,
                    top_logprobs_chunk,
                    tail_mass_chunk,
                    teacher_top_mask=top_mask_chunk,
                    temperature=args.temperature,
                    reduction="sum",
                )
                ce_value = (
                    F.cross_entropy(
                        student_logits,
                        target_chunk,
                        reduction="sum",
                    )
                    if args.hard_ce_weight > 0
                    else student_logits.new_zeros(())
                )
                return torch.stack((kd_value, ce_value))

            chunk_values = torch.utils.checkpoint.checkpoint(
                sparse_chunk,
                student_hidden[start:end],
                target_ids[start:end],
                top_ids[start:end],
                top_logprobs[start:end],
                tail_mass[start:end],
                top_mask[start:end],
                use_reentrant=False,
            )
        chunk_kd, chunk_ce = chunk_values.unbind()
        kd_sum = chunk_kd if kd_sum is None else kd_sum + chunk_kd
        ce_sum = chunk_ce if ce_sum is None else ce_sum + chunk_ce
    assert kd_sum is not None and ce_sum is not None
    kd_loss = kd_sum / position_count
    ce_loss = ce_sum / position_count
    total = args.kd_weight * kd_loss + args.hard_ce_weight * ce_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "ce_loss": float(ce_loss.detach().cpu()),
        "positions": float(position_count),
    }


def _save_checkpoint(
    destination: Path,
    *,
    args: argparse.Namespace,
    student: torch.nn.Module,
    overlay: Any,
    tokenizer: Any,
    contract_path: str | Path,
    provenance: Mapping[str, Any],
    status: str,
) -> None:
    if status not in {"checkpoint", "complete"}:
        raise ValueError("checkpoint status must be checkpoint or complete")
    destination.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(destination / "decoder_adapter")
    torch.save(overlay.overlay_state(), destination / "source_embedding_overlay.pt")
    tokenizer.save_pretrained(destination / "tokenizer")
    shutil.copyfile(contract_path, destination / "compact_contract.json")
    if sha256_file(contract_path) != sha256_file(
        destination / "compact_contract.json"
    ):
        raise RuntimeError("saved compact contract is not byte-identical")
    final_provenance = dict(provenance)
    final_provenance.update(
        {
            "decoder_adapter_sha256": _hash_directory(
                destination / "decoder_adapter"
            ),
            "source_embedding_overlay_sha256": sha256_file(
                destination / "source_embedding_overlay.pt"
            ),
            "saved_contract_sha256": sha256_file(
                destination / "compact_contract.json"
            ),
            "status": status,
        }
    )
    (destination / "run_provenance.json").write_text(
        json.dumps(final_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models.direct_compact_causal import (
        DirectCompactBatchCollator,
        DirectCompactContract,
        install_source_embedding_overlay,
        resolve_decoder_config_path,
        restore_source_embedding_overlay,
        validate_base_model_vocab,
        validate_join_seal,
    )
    from scripts.training.direct_compact_qwen_decompiler import (
        CompactJsonlDataset,
        _encode,
        direct_prompt,
    )
    from transformers import (
        AutoTokenizer,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir).resolve()
    _ensure_output_directory(output_dir)

    contract = DirectCompactContract.load(args.contract)
    contract_sha = sha256_file(args.contract)
    decoder_model = args.student_model.strip() or contract.decoder_model
    decoder_revision = args.student_revision.strip() or contract.decoder_revision
    decoder_config_path = resolve_decoder_config_path(
        decoder_model, decoder_revision
    )
    contract.validate_decoder_binding(
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        model_config_path=decoder_config_path,
    )

    student_tokenizer_name = args.student_tokenizer or decoder_model
    student_tokenizer_revision = (
        args.student_tokenizer_revision
        or (decoder_revision if student_tokenizer_name == decoder_model else "")
        or None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        student_tokenizer_name,
        revision=student_tokenizer_revision,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("true KD requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    contract.validate_artifacts(
        tokenizer=tokenizer,
        tokenizer_json_path=args.student_tokenizer_json,
        codec_path=args.codec_artifact,
        codebook_path=args.codebook,
    )
    student_tokenizer_sha = sha256_file(args.student_tokenizer_json)

    warmstart_layout = None
    if args.student_warmstart_checkpoint:
        warmstart_layout = _checkpoint_layout(
            args.student_warmstart_checkpoint, identity="student warm-start"
        )
        _validate_checkpoint_binding(
            warmstart_layout,
            identity="student warm-start",
            contract_path=args.contract,
            tokenizer_sha256=student_tokenizer_sha,
        )

    teacher = None
    teacher_layout = None
    teacher_tokenizer_sha = None
    if args.mode == "dense_full_kl":
        teacher_tokenizer_sha = sha256_file(args.teacher_tokenizer_json)
        if teacher_tokenizer_sha != student_tokenizer_sha:
            raise ValueError(
                "dense_full_kl requires byte-identical teacher/student tokenizer JSON"
            )
        teacher_tokenizer_name = args.teacher_tokenizer or args.teacher_model
        teacher_tokenizer_revision = (
            args.teacher_tokenizer_revision
            or (args.teacher_revision if teacher_tokenizer_name == args.teacher_model else "")
            or None
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_tokenizer_name,
            revision=teacher_tokenizer_revision,
            trust_remote_code=True,
        )
        if teacher_tokenizer.eos_token_id is None:
            raise ValueError("dense teacher tokenizer has no EOS token")
        if teacher_tokenizer.pad_token_id is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        if _tokenizer_signature(teacher_tokenizer) != _tokenizer_signature(tokenizer):
            raise ValueError(
                "teacher and student tokenizer vocabulary/special-token mappings differ"
            )
        teacher_layout = _checkpoint_layout(
            args.teacher_checkpoint, identity="teacher"
        )
        _validate_checkpoint_binding(
            teacher_layout,
            identity="teacher",
            contract_path=args.contract,
            tokenizer_sha256=student_tokenizer_sha,
        )
        if warmstart_layout is not None:
            if (
                _hash_directory(teacher_layout["adapter"])
                == _hash_directory(warmstart_layout["adapter"])
                and sha256_file(teacher_layout["overlay"])
                == sha256_file(warmstart_layout["overlay"])
            ):
                raise ValueError(
                    "teacher and student warm-start are identical; dense KL would "
                    "begin as a no-op"
                )

    train_seal = None
    sparse_manifest = None
    if args.mode == "dense_full_kl":
        train_seal = validate_join_seal(
            args.train_file,
            args.train_seal,
            args.contract,
            expected_role="fit",
        )
        train_dataset = CompactJsonlDataset(
            args.train_file, tokenizer=tokenizer, contract=contract
        )
    else:
        sparse_manifest, sparse_rows = validate_sparse_topk_tail_dataset(
            args.train_file,
            args.sparse_manifest,
            student_tokenizer_json=args.student_tokenizer_json,
            expected_contract_sha256=contract_sha,
        )
        if int(sparse_manifest["student_vocab_size"]) != int(
            contract.base_vocab_size
        ):
            raise ValueError("sparse manifest vocabulary does not match the contract")
        if int(sparse_manifest["eos_token_id"]) != int(tokenizer.eos_token_id):
            raise ValueError("sparse manifest EOS does not match the tokenizer")
        train_dataset = SparseCompactDataset(
            sparse_rows,
            tokenizer=tokenizer,
            contract=contract,
            direct_prompt=direct_prompt,
            encode=_encode,
        )

    base_collator = DirectCompactBatchCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        max_source_tokens=contract.max_source_tokens,
        max_target_tokens=contract.max_target_tokens,
        max_total_tokens=contract.max_total_tokens,
        pad_to_multiple_of=8,
        source_token_ids=contract.source_token_ids,
    )
    collator = (
        base_collator
        if args.mode == "dense_full_kl"
        else SparseCompactCollator(base_collator)
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    if len(data_loader) == 0:
        raise ValueError("training data loader is empty")

    if teacher_layout is not None:
        teacher, _ = _load_teacher(
            args,
            contract=contract,
            layout=teacher_layout,
            device=device,
            restore_source_embedding_overlay=restore_source_embedding_overlay,
            validate_base_model_vocab=validate_base_model_vocab,
        )
    student, student_overlay = _load_student(
        args,
        contract=contract,
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        warmstart_layout=warmstart_layout,
        device=device,
        install_source_embedding_overlay=install_source_embedding_overlay,
        restore_source_embedding_overlay=restore_source_embedding_overlay,
        validate_base_model_vocab=validate_base_model_vocab,
    )
    student_vocab = int(student.get_output_embeddings().weight.size(0))
    if student_vocab != int(contract.base_vocab_size):
        raise RuntimeError("student output vocabulary does not match the contract")
    if teacher is not None:
        teacher_vocab = int(teacher.get_output_embeddings().weight.size(0))
        if teacher_vocab != student_vocab:
            raise ValueError(
                "dense_full_kl requires identical teacher/student output vocabulary"
            )

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("student has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(data_loader) / args.grad_accum)
    planned_steps = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)
    if planned_steps <= 0:
        raise ValueError("training schedule contains no optimizer steps")
    warmup_steps = int(planned_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_steps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and args.dtype == "fp16")
    )

    provenance: dict[str, Any] = {
        "schema": "true-distribution-kd-run-provenance-v1",
        "mode": args.mode,
        "objective": (
            "full_vocab_forward_kl_teacher_to_student"
            if args.mode == "dense_full_kl"
            else "topk_plus_aggregate_tail_forward_kl_lower_bound"
        ),
        "is_full_distribution_kd": args.mode == "dense_full_kl",
        "student_model": decoder_model,
        "student_revision": decoder_revision,
        "student_tokenizer_sha256": student_tokenizer_sha,
        "student_vocab_size": student_vocab,
        "student_warmstart_checkpoint": (
            str(warmstart_layout["root"]) if warmstart_layout else None
        ),
        "student_warmstart_adapter_sha256": (
            _hash_directory(warmstart_layout["adapter"])
            if warmstart_layout
            else None
        ),
        "student_warmstart_overlay_sha256": (
            sha256_file(warmstart_layout["overlay"])
            if warmstart_layout
            else None
        ),
        "student_warmstart_contract_file_sha256": (
            sha256_file(warmstart_layout["contract"])
            if warmstart_layout
            else None
        ),
        "student_warmstart_saved_tokenizer_json_sha256": (
            sha256_file(warmstart_layout["tokenizer_json"])
            if warmstart_layout
            else None
        ),
        "teacher_model": args.teacher_model or (
            sparse_manifest["teacher_model"] if sparse_manifest else None
        ),
        "teacher_revision": args.teacher_revision or (
            sparse_manifest["teacher_revision"] if sparse_manifest else None
        ),
        "teacher_tokenizer_sha256": teacher_tokenizer_sha or (
            sparse_manifest["teacher_tokenizer_sha256"]
            if sparse_manifest
            else None
        ),
        "teacher_checkpoint": (
            str(teacher_layout["root"]) if teacher_layout else None
        ),
        "teacher_adapter_sha256": (
            _hash_directory(teacher_layout["adapter"]) if teacher_layout else None
        ),
        "teacher_overlay_sha256": (
            sha256_file(teacher_layout["overlay"]) if teacher_layout else None
        ),
        "teacher_contract_file_sha256": (
            sha256_file(teacher_layout["contract"]) if teacher_layout else None
        ),
        "teacher_saved_tokenizer_json_sha256": (
            sha256_file(teacher_layout["tokenizer_json"])
            if teacher_layout
            else None
        ),
        "contract_sha256": contract_sha,
        "contract_canonical_json_sha256": _canonical_json_sha256(args.contract),
        "codebook_sha256": sha256_file(args.codebook),
        "codec_sha256": sha256_file(args.codec_artifact),
        "train_file_sha256": sha256_file(args.train_file),
        "train_seal_sha256": (
            sha256_file(args.train_seal) if args.train_seal else None
        ),
        "sparse_manifest_sha256": (
            sha256_file(args.sparse_manifest) if args.sparse_manifest else None
        ),
        "sealed_train_rows": (
            int(train_seal["rows"]) if train_seal is not None else len(train_dataset)
        ),
        "eos_token_id": int(tokenizer.eos_token_id),
        "eos_supervised": True,
        "temperature": args.temperature,
        "temperature_squared_scaling": args.mode == "dense_full_kl",
        "kd_weight": args.kd_weight,
        "hard_ce_weight": args.hard_ce_weight,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "partial_accumulation_windows_use_true_divisor": True,
        "position_chunk_size": args.position_chunk_size,
        "full_vocab_position_chunks_are_activation_checkpointed": True,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "student_load_4bit": args.student_load_4bit,
        "teacher_load_4bit": args.teacher_load_4bit,
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "objectives_sha256": sha256_file(
            SCRIPT_DIR / "kd_objectives_antigravity.py"
        ),
    }

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    observed_batches = 0
    rolling = {"loss": 0.0, "kd_loss": 0.0, "ce_loss": 0.0, "positions": 0.0}
    stop = False
    for epoch in range(args.epochs):
        for batch_index, raw_batch in enumerate(data_loader):
            batch = _move_batch(raw_batch, device)
            loss, metrics = _batch_loss(
                args,
                student=student,
                teacher=teacher,
                batch=batch,
            )
            divisor = accumulation_window_size(
                batch_index,
                total_batches=len(data_loader),
                gradient_accumulation_steps=args.grad_accum,
            )
            scaler.scale(loss / divisor).backward()
            observed_batches += 1
            for key in rolling:
                rolling[key] += metrics[key]

            if is_accumulation_boundary(
                batch_index,
                total_batches=len(data_loader),
                gradient_accumulation_steps=args.grad_accum,
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                if optimizer_step % args.logging_steps == 0:
                    denominator = float(observed_batches)
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "optimizer_step": optimizer_step,
                                "loss": rolling["loss"] / denominator,
                                "kd_loss": rolling["kd_loss"] / denominator,
                                "ce_loss": rolling["ce_loss"] / denominator,
                                "supervised_positions": int(rolling["positions"]),
                                "learning_rate": scheduler.get_last_lr()[0],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    observed_batches = 0
                    rolling = {
                        "loss": 0.0,
                        "kd_loss": 0.0,
                        "ce_loss": 0.0,
                        "positions": 0.0,
                    }
                if (
                    args.save_every_steps
                    and optimizer_step % args.save_every_steps == 0
                ):
                    checkpoint_provenance = dict(provenance)
                    checkpoint_provenance["optimizer_steps"] = optimizer_step
                    _save_checkpoint(
                        output_dir / f"checkpoint-{optimizer_step}",
                        args=args,
                        student=student,
                        overlay=student_overlay,
                        tokenizer=tokenizer,
                        contract_path=args.contract,
                        provenance=checkpoint_provenance,
                        status="checkpoint",
                    )
                if args.max_steps > 0 and optimizer_step >= args.max_steps:
                    stop = True
                    break
        if stop:
            break
    if optimizer_step <= 0:
        raise RuntimeError("training completed without an optimizer step")
    provenance["optimizer_steps"] = optimizer_step
    _save_checkpoint(
        output_dir,
        args=args,
        student=student,
        overlay=student_overlay,
        tokenizer=tokenizer,
        contract_path=args.contract,
        provenance=provenance,
        status="complete",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": args.mode,
                "optimizer_steps": optimizer_step,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
