#!/usr/bin/env python3
"""Native T5Gemma 2 SFT on the sealed enriched F2 binary representation.

This is deliberately a text-to-text trainer.  The encoder receives the
audited, API-readable F2 representation (constants, external calls,
instructions, and explicit control-flow structure); the decoder is supervised
only on the reference Dart source.  Test harnesses are never serialized into
the model input.

Production runs are hash-pinned and fail closed on truncation.  Checkpoints
contain only the LoRA adapter, tokenizer, optimizer/scheduler/RNG state, and
the immutable run contract--never a duplicate copy of the 9B base model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn.utils import clip_grad_norm_


RUN_SCHEMA = "t5gemma2-enriched-sft-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-enriched-sft-checkpoint-v1"
REPRESENTATION_SCHEMA = "lossless-semantic-f2"
F2_ROW_SCHEMA = "audited-frontier-passk-v1"
DEFAULT_MODEL = "google/t5gemma-2-4b-4b"
SOURCE_PREAMBLE = (
    "Decompile the following enriched compact binary representation to Dart.\n"
    "Use the recovered constants, external-call identities, instruction view, "
    "and control-flow structure together. Return exactly one self-contained "
    "compilable Dart source-unit fragment. It must define a top-level function "
    "named fn0, may include required imports and top-level helpers, and must "
    "not contain markdown, prose, tests, demos, or main.\n"
    "<enriched_binary>\n"
)
SOURCE_SUFFIX = "\n</enriched_binary>\n"
_FORBIDDEN_SOURCE_FIELDS = frozenset(
    {
        "acceptance_tests",
        "tests",
        "feedback_tests",
        "holdback_tests",
        "dart_source",
        "supervised_target",
        "reference",
        "gold",
    }
)
_REQUIRED_F2_ATTESTATIONS = {
    "artifact_hashes": True,
    "row_contract_hashes": True,
    "codec_text_roundtrip": True,
    "codec_token_id_roundtrip": True,
    "student_constant_prefix": True,
    "per_task_instruction_dictionary_roundtrip": True,
    "compact_semantic_f2_roundtrip": True,
    "branch_targets_reconstructed_from_cfg": True,
    "visible_task_symbols_one_token": True,
    # This is an absence attestation, so the correct sealed value is false.
    "opaque_custom_ids_in_text": False,
}
_ALLOWED_LORA_SUFFIXES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_NAME_RE = re.compile(r"checkpoint-optstep-([0-9]{6,})\Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{source}: blank row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}: row {line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{source}: no rows")
    return rows


def _pin_file(
    path: str | Path,
    expected_sha256: str,
    *,
    allow_unpinned: bool,
) -> str:
    observed = sha256_file(path)
    expected = expected_sha256.strip().lower()
    if not expected and not allow_unpinned:
        raise ValueError(f"{path}: production input requires an expected SHA-256")
    if expected and observed != expected:
        raise ValueError(
            f"{path}: SHA-256 mismatch; expected {expected}, observed {observed}"
        )
    return observed


def _identity(row: Mapping[str, Any], index: int) -> str:
    task_id = str(row.get("task_id") or row.get("id") or "").strip()
    if not task_id:
        raise ValueError(f"row {index}: missing task_id")
    return task_id


def _target_source(row: Mapping[str, Any], task_id: str) -> str:
    target = str(
        row.get("supervised_target")
        or row.get("dart_source")
        or row.get("source")
        or ""
    ).strip()
    if not target:
        raise ValueError(f"{task_id}: missing supervised Dart source")
    return target


def build_encoder_source(f2_row: Mapping[str, Any], task_id: str) -> str:
    """Return the only text visible to the native encoder."""

    if str(f2_row.get("schema") or "") != F2_ROW_SCHEMA:
        raise ValueError(f"{task_id}: F2 producer schema is not sealed")
    if f2_row.get("representation_schema") != REPRESENTATION_SCHEMA:
        raise ValueError(f"{task_id}: source is not the sealed F2 representation")
    if str(f2_row.get("task_id") or "").strip() != task_id:
        raise ValueError(f"{task_id}: F2 row identity mismatch")
    for field in _FORBIDDEN_SOURCE_FIELDS:
        if field in f2_row:
            raise ValueError(f"{task_id}: forbidden field {field!r} in F2 source")
    text = f2_row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{task_id}: F2 source text is empty")
    declared = str(f2_row.get("text_sha256") or "").strip().lower()
    observed = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not _HEX_SHA256.fullmatch(declared) or declared != observed:
        raise ValueError(f"{task_id}: F2 text digest mismatch")
    verified = f2_row.get("verified")
    if not isinstance(verified, dict) or not verified:
        raise ValueError(f"{task_id}: F2 verification record is missing")
    failed = sorted(
        key
        for key, expected in _REQUIRED_F2_ATTESTATIONS.items()
        if verified.get(key) is not expected
    )
    if failed:
        raise ValueError(f"{task_id}: F2 verification contract failed: {failed}")
    return SOURCE_PREAMBLE + text + SOURCE_SUFFIX


@dataclass(frozen=True)
class TextPair:
    task_id: str
    source: str
    target: str
    source_sha256: str
    target_sha256: str


def load_text_pairs(
    dataset_path: str | Path,
    f2_path: str | Path,
    *,
    expected_dataset_sha256: str,
    expected_f2_sha256: str,
    expected_rows: int,
    allow_unpinned_inputs: bool = False,
) -> tuple[list[TextPair], dict[str, Any]]:
    """Hash-pin and join the enriched source to the reference Dart target."""

    dataset_sha = _pin_file(
        dataset_path,
        expected_dataset_sha256,
        allow_unpinned=allow_unpinned_inputs,
    )
    f2_sha = _pin_file(
        f2_path,
        expected_f2_sha256,
        allow_unpinned=allow_unpinned_inputs,
    )
    dataset_rows = _read_jsonl(dataset_path)
    f2_rows = _read_jsonl(f2_path)
    if expected_rows > 0 and (
        len(dataset_rows) != expected_rows or len(f2_rows) != expected_rows
    ):
        raise ValueError(
            "row-count mismatch: "
            f"dataset={len(dataset_rows)}, f2={len(f2_rows)}, "
            f"expected={expected_rows}"
        )
    if len(dataset_rows) != len(f2_rows):
        raise ValueError("dataset and F2 row counts differ")

    dataset_ids = [_identity(row, index) for index, row in enumerate(dataset_rows)]
    f2_ids = [_identity(row, index) for index, row in enumerate(f2_rows)]
    if len(set(dataset_ids)) != len(dataset_ids) or len(set(f2_ids)) != len(f2_ids):
        raise ValueError("dataset or F2 contains duplicate task IDs")
    if dataset_ids != f2_ids:
        raise ValueError(
            "dataset and F2 rows are not in the same hash-bound task order"
        )

    pairs: list[TextPair] = []
    for task_id, dataset_row, f2_row in zip(
        dataset_ids, dataset_rows, f2_rows, strict=True
    ):
        source = build_encoder_source(f2_row, task_id)
        target = _target_source(dataset_row, task_id)
        pairs.append(
            TextPair(
                task_id=task_id,
                source=source,
                target=target,
                source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                target_sha256=hashlib.sha256(target.encode("utf-8")).hexdigest(),
            )
        )
    manifest = {
        "schema": RUN_SCHEMA,
        # Identity is content-addressed so an immutable checkpoint can resume
        # after the sealed inputs move to another host or mount point.
        "dataset": {"sha256": dataset_sha},
        "f2": {"sha256": f2_sha},
        "rows": len(pairs),
        "task_ids_sha256": canonical_sha256(dataset_ids),
        "source_sha256s_sha256": canonical_sha256(
            [pair.source_sha256 for pair in pairs]
        ),
        "target_sha256s_sha256": canonical_sha256(
            [pair.target_sha256 for pair in pairs]
        ),
        "model_visible_fields": ["F2.text"],
        "model_hidden_fields": sorted(_FORBIDDEN_SOURCE_FIELDS),
        "reference_target_field": "dart_source",
    }
    return pairs, manifest


@dataclass(frozen=True)
class TokenizedPair:
    task_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def _tokenize_text(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    return [int(token) for token in encoded["input_ids"]]


def tokenize_pairs(
    tokenizer: Any,
    pairs: Sequence[TextPair],
    *,
    max_source_tokens: int,
    max_target_tokens: int,
) -> tuple[list[TokenizedPair], dict[str, Any]]:
    """Tokenize without truncation and fail closed when capacity is exceeded."""

    if max_source_tokens <= 0 or max_target_tokens <= 0:
        raise ValueError("source and target capacities must be positive")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("T5Gemma tokenizer has no EOS token")
    rows: list[TokenizedPair] = []
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    for pair in pairs:
        input_ids = _tokenize_text(tokenizer, pair.source)
        # Decoder labels are the Dart bytes/tokens followed by exactly one EOS.
        # Letting the tokenizer inject BOS here would train the model to predict
        # BOS as source code even though the model already right-shifts labels
        # with its configured decoder-start token.
        labels = _tokenize_text(
            tokenizer,
            pair.target,
            add_special_tokens=False,
        )
        if int(eos_id) in labels:
            raise ValueError(f"{pair.task_id}: target contains an embedded EOS token")
        labels.append(int(eos_id))
        if len(input_ids) > max_source_tokens:
            raise ValueError(
                f"{pair.task_id}: source length {len(input_ids)} exceeds "
                f"{max_source_tokens}; source truncation is forbidden"
            )
        if len(labels) > max_target_tokens:
            raise ValueError(
                f"{pair.task_id}: target length {len(labels)} exceeds "
                f"{max_target_tokens}; target truncation is forbidden"
            )
        rows.append(
            TokenizedPair(
                task_id=pair.task_id,
                input_ids=tuple(input_ids),
                labels=tuple(labels),
            )
        )
        source_lengths.append(len(input_ids))
        target_lengths.append(len(labels))
    report = {
        "rows": len(rows),
        "source_tokens": {
            "max": max(source_lengths),
            "mean": sum(source_lengths) / len(source_lengths),
        },
        "target_tokens": {
            "max": max(target_lengths),
            "mean": sum(target_lengths) / len(target_lengths),
        },
        "truncated_rows": 0,
        "eos_supervised": True,
        "target_special_tokens": "none_plus_exactly_one_terminal_eos",
    }
    return rows, report


def collate_pairs(
    rows: Sequence[TokenizedPair],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    max_source = max(len(row.input_ids) for row in rows)
    max_target = max(len(row.labels) for row in rows)
    input_ids = torch.full(
        (len(rows), max_source),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full(
        (len(rows), max_target),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        source = torch.tensor(row.input_ids, dtype=torch.long, device=device)
        target = torch.tensor(row.labels, dtype=torch.long, device=device)
        input_ids[index, : source.numel()] = source
        attention_mask[index, : source.numel()] = 1
        labels[index, : target.numel()] = target
    return {
        "task_ids": [row.task_id for row in rows],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def deterministic_epoch_order(
    rows: Sequence[TokenizedPair], *, seed: int, epoch: int
) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda index: canonical_sha256(
            {
                "schema": RUN_SCHEMA,
                "seed": int(seed),
                "epoch": int(epoch),
                "task_id": rows[index].task_id,
            }
        ),
    )


def calculate_training_schedule(
    *,
    rows: int,
    epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    max_updates: int,
    warmup_ratio: float,
) -> dict[str, int]:
    """Return the exact optimizer-step schedule implemented by ``train``."""

    if rows <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("rows, epochs, and batch size must be positive")
    if gradient_accumulation <= 0 or max_updates < 0:
        raise ValueError("invalid accumulation or maximum-update count")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup ratio must lie in [0,1)")
    microbatches_per_epoch = math.ceil(rows / batch_size)
    updates_per_epoch = math.ceil(microbatches_per_epoch / gradient_accumulation)
    available_updates = epochs * updates_per_epoch
    planned_updates = (
        min(available_updates, max_updates) if max_updates > 0 else available_updates
    )
    warmup_updates = min(
        planned_updates,
        math.ceil(planned_updates * warmup_ratio),
    )
    return {
        "microbatches_per_epoch": microbatches_per_epoch,
        "updates_per_epoch": updates_per_epoch,
        "available_updates": available_updates,
        "planned_updates": planned_updates,
        "warmup_updates": warmup_updates,
    }


def cosine_schedule_multiplier(
    step: int,
    *,
    warmup_updates: int,
    total_updates: int,
) -> float:
    if step < 0 or warmup_updates < 0 or total_updates <= 0:
        raise ValueError("invalid scheduler step or horizon")
    if warmup_updates > total_updates:
        raise ValueError("warmup exceeds the scheduler horizon")
    if warmup_updates > 0 and step < warmup_updates:
        return max(1e-8, step / warmup_updates)
    progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
    progress = min(1.0, max(0.0, progress))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _checkpoint_name(update: int) -> str:
    return f"checkpoint-optstep-{update:06d}"


def save_checkpoint(
    *,
    output_dir: Path,
    update: int,
    epoch: int,
    next_row: int,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    run_contract: Mapping[str, Any],
) -> Path:
    """Publish one immutable, update-boundary adapter checkpoint."""

    destination = output_dir / _checkpoint_name(update)
    if destination.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {destination}")
    temporary = output_dir / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary / "adapter").mkdir()
    model.save_pretrained(
        temporary / "adapter",
        safe_serialization=True,
    )
    tokenizer.save_pretrained(temporary / "tokenizer")
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "update": int(update),
            "epoch": int(epoch),
            "next_row": int(next_row),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
            "run_contract_sha256": canonical_sha256(run_contract),
        },
        temporary / "training_state.pt",
    )
    _atomic_json(temporary / "run_contract.json", dict(run_contract))
    os.replace(temporary, destination)
    _atomic_json(
        output_dir / "latest_checkpoint.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "path": str(destination.resolve()),
            "update": int(update),
            "run_contract_sha256": canonical_sha256(run_contract),
        },
    )
    return destination


def _load_tokenizer(model_name: str, revision: str, token: str | None) -> Any:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "use_fast": True,
    }
    if token:
        kwargs["token"] = token
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token_id is None:
        raise ValueError("T5Gemma tokenizer does not define a pad token")
    if tokenizer.eos_token_id is None:
        raise ValueError("T5Gemma tokenizer does not define an EOS token")
    return tokenizer


def _load_base_model(args: argparse.Namespace, token: str | None) -> Any:
    from transformers import AutoModelForSeq2SeqLM

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    if token:
        kwargs["token"] = token
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **kwargs)
    _validate_t5gemma2_config(model.config)
    return model


def _validate_t5gemma2_config(config: Any) -> None:
    if str(getattr(config, "model_type", "")) != "t5gemma2":
        raise ValueError("loaded checkpoint is not a native T5Gemma 2 model")
    if not bool(getattr(config, "is_encoder_decoder", False)):
        raise ValueError("loaded checkpoint is not a native encoder-decoder")
    encoder = getattr(config, "encoder", None)
    if encoder is None or getattr(encoder, "text_config", None) is None:
        raise ValueError("T5Gemma 2 encoder text configuration is missing")
    if getattr(config, "decoder", None) is None:
        raise ValueError("T5Gemma 2 decoder configuration is missing")


def _resolve_lora_targets(model: Any, args: argparse.Namespace) -> list[str]:
    """Resolve text-only modules; never attach adapters to the vision tower."""

    requested_items = [
        item.strip() for item in args.lora_target_modules.split(",") if item.strip()
    ]
    requested = set(requested_items)
    if not requested:
        raise ValueError("LoRA target suffix list is empty")
    if len(requested) != len(requested_items):
        raise ValueError("LoRA target suffix list contains duplicates")
    unsupported = sorted(requested - _ALLOWED_LORA_SUFFIXES)
    if unsupported:
        raise ValueError(
            f"unsupported LoRA target suffixes: {unsupported}; "
            f"allowed={sorted(_ALLOWED_LORA_SUFFIXES)}"
        )

    targets_by_side: dict[str, dict[str, list[str]]] = {
        "encoder": {suffix: [] for suffix in requested},
        "decoder": {suffix: [] for suffix in requested},
    }
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if not isinstance(module, torch.nn.Linear) or suffix not in requested:
            continue
        is_text_encoder = ".encoder.text_model.layers." in f".{name}"
        is_decoder = ".decoder.layers." in f".{name}"
        if is_text_encoder:
            targets_by_side["encoder"][suffix].append(name)
        elif is_decoder:
            targets_by_side["decoder"][suffix].append(name)

    missing_by_side = {
        side: sorted(suffix for suffix, names in suffixes.items() if not names)
        for side, suffixes in targets_by_side.items()
    }
    if any(missing_by_side.values()):
        available = sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        )
        raise ValueError(
            "could not resolve exact text encoder+decoder LoRA modules; "
            f"missing_by_side={missing_by_side}, "
            f"linear sample={available[:40]}"
        )
    targets = [
        name
        for side in ("encoder", "decoder")
        for suffix in sorted(requested)
        for name in targets_by_side[side][suffix]
    ]
    if any("vision" in name.lower() for name in targets):
        raise AssertionError("vision-tower module entered the LoRA target set")
    return sorted(targets)


def _attach_lora(model: Any, args: argparse.Namespace, targets: Sequence[str]) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    if not targets:
        raise ValueError("no LoRA target modules resolved")
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
    )
    return get_peft_model(model, config)


def _adapter_weight_target_modules(checkpoint: Path) -> set[str]:
    """Recover the actual adapted modules from the saved weight keys.

    PEFT minimizes ``target_modules`` when serializing a config, so that field
    is a suffix-matching program rather than a byte-for-byte copy of the
    explicit module list supplied at construction time.  The adapter weight
    keys remain exact and are therefore the authoritative checkpoint audit.
    """

    safetensors_path = checkpoint / "adapter" / "adapter_model.safetensors"
    bin_path = checkpoint / "adapter" / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors import safe_open

        with safe_open(
            safetensors_path,
            framework="pt",
            device="cpu",
        ) as handle:
            keys = list(handle.keys())
    elif bin_path.is_file():
        payload = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("resume adapter binary is not a state mapping")
        keys = [str(key) for key in payload]
    else:
        raise FileNotFoundError("resume checkpoint lacks adapter weights")

    pattern = re.compile(
        r"^(?:base_model\.model\.)?"
        r"(?P<module>.+)\.lora_(?P<branch>A|B)"
        r"(?:\.[^.]+)?\.weight$"
    )
    branches: dict[str, set[str]] = {}
    unexpected_lora_keys: list[str] = []
    for key in keys:
        match = pattern.fullmatch(str(key))
        if match is None:
            if ".lora_" in str(key):
                unexpected_lora_keys.append(str(key))
            continue
        module = match.group("module")
        branches.setdefault(module, set()).add(match.group("branch"))
    if unexpected_lora_keys:
        raise ValueError(
            "resume adapter has unrecognized LoRA weight keys: "
            f"{unexpected_lora_keys[:10]}"
        )
    incomplete = sorted(
        module for module, observed in branches.items() if observed != {"A", "B"}
    )
    if not branches or incomplete:
        raise ValueError(
            "resume adapter lacks complete LoRA A/B pairs: "
            f"incomplete={incomplete[:10]}"
        )
    return set(branches)


def _load_resume_artifacts(
    checkpoint: Path,
    *,
    exact_targets: Sequence[str],
    weights_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"resume checkpoint is not a directory: {checkpoint}")
    required = [
        checkpoint / "run_contract.json",
        checkpoint / "training_state.pt",
        checkpoint / "adapter" / "adapter_config.json",
        checkpoint / "tokenizer" / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    adapter_weights = [
        checkpoint / "adapter" / "adapter_model.safetensors",
        checkpoint / "adapter" / "adapter_model.bin",
    ]
    if not any(path.is_file() for path in adapter_weights):
        missing.append(str(checkpoint / "adapter" / "adapter_model.{safetensors,bin}"))
    if missing:
        raise FileNotFoundError(f"resume checkpoint is incomplete; missing={missing}")
    with (checkpoint / "run_contract.json").open("r", encoding="utf-8") as handle:
        saved_contract = json.load(handle)
    if (
        not isinstance(saved_contract, dict)
        or saved_contract.get("schema") != RUN_SCHEMA
        or saved_contract.get("status") != "training"
    ):
        raise ValueError("resume run contract schema/status mismatch")
    saved_targets = saved_contract.get("lora", {}).get("targets")
    if (
        not isinstance(saved_targets, list)
        or len(saved_targets) != len(exact_targets)
        or set(map(str, saved_targets)) != set(map(str, exact_targets))
    ):
        raise ValueError(
            "resume run contract target modules differ from the exact text-only set"
        )
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=weights_only,
    )
    if not isinstance(state, dict) or state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("resume checkpoint schema mismatch")
    if state.get("run_contract_sha256") != canonical_sha256(saved_contract):
        raise ValueError("resume checkpoint run-contract hash mismatch")
    update = int(state.get("update", -1))
    if update < 0 or checkpoint.name != _checkpoint_name(update):
        raise ValueError("resume checkpoint path/update identity mismatch")
    for key in ("optimizer", "scheduler", "rng"):
        if key not in state:
            raise ValueError(f"resume checkpoint lacks {key!r} state")

    with (checkpoint / "adapter" / "adapter_config.json").open(
        "r", encoding="utf-8"
    ) as handle:
        adapter_config = json.load(handle)
    configured_targets = adapter_config.get("target_modules")
    if (
        not isinstance(configured_targets, list)
        or not configured_targets
        or any(not isinstance(name, str) or not name for name in configured_targets)
    ):
        raise ValueError("resume adapter target-module program is invalid")
    if str(adapter_config.get("task_type") or "") != "SEQ_2_SEQ_LM":
        raise ValueError("resume adapter is not a seq2seq LoRA policy")
    weighted_targets = _adapter_weight_target_modules(checkpoint)
    if weighted_targets != set(map(str, exact_targets)):
        raise ValueError(
            "resume adapter weights differ from the exact text-only target set"
        )
    return saved_contract, state


def _is_link_like(path: Path) -> bool:
    """Treat POSIX symlinks and Windows directory junctions identically."""

    try:
        is_junction = getattr(path, "is_junction", None)
        return path.is_symlink() or bool(is_junction is not None and is_junction())
    except OSError as exc:
        raise ValueError(f"cannot inspect checkpoint path safely: {path}") from exc


def _path_identity(path: Path) -> tuple[int, int, int]:
    stat = path.stat(follow_symlinks=False)
    return int(stat.st_dev), int(stat.st_ino), int(stat.st_mode)


def _assert_real_checkpoint_tree(root: Path, checkpoint: Path) -> None:
    if _is_link_like(checkpoint):
        raise ValueError(f"checkpoint cannot be a symlink/junction: {checkpoint}")
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint candidate is not a directory: {checkpoint}")
    resolved = checkpoint.resolve(strict=True)
    if resolved.parent != root or checkpoint.parent != root:
        raise ValueError(f"checkpoint candidate escapes the output root: {checkpoint}")
    for entry in checkpoint.rglob("*"):
        if _is_link_like(entry):
            raise ValueError(f"checkpoint tree contains a symlink/junction: {entry}")
        resolved_entry = entry.resolve(strict=True)
        if resolved_entry != resolved and resolved not in resolved_entry.parents:
            raise ValueError(f"checkpoint entry escapes its directory: {entry}")


def _validated_retention_candidates(
    output_dir: Path,
    *,
    run_contract: Mapping[str, Any],
) -> list[tuple[int, Path, tuple[int, int, int]]]:
    """Validate every checkpoint-like child before returning any deletion set."""

    if _is_link_like(output_dir):
        raise ValueError("checkpoint output root cannot be a symlink/junction")
    root = output_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint output root is not a directory")
    root_contract_path = root / "run_contract.json"
    if _is_link_like(root_contract_path) or not root_contract_path.is_file():
        raise ValueError("root run contract is missing or linked")
    with root_contract_path.open("r", encoding="utf-8") as handle:
        root_contract = json.load(handle)
    expected_contract_sha = canonical_sha256(run_contract)
    accepted_contract_shas = {expected_contract_sha}
    migration = run_contract.get("resume_migration")
    if migration is not None:
        if (
            not isinstance(migration, Mapping)
            or migration.get("schema") != "trainer-source-only-resume-v1"
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(migration.get("accepted_contract_sha256") or ""),
            )
        ):
            raise ValueError("root run contract has an invalid resume migration")
        accepted_contract_shas.add(str(migration["accepted_contract_sha256"]))
    if canonical_sha256(root_contract) != expected_contract_sha:
        raise ValueError("root run contract differs from the active run")
    exact_targets = run_contract.get("lora", {}).get("targets")
    if not isinstance(exact_targets, list) or not exact_targets:
        raise ValueError("root run contract lacks exact LoRA targets")

    candidates: list[tuple[int, Path, tuple[int, int, int]]] = []
    checkpoint_like = sorted(
        (
            child
            for child in root.iterdir()
            if child.name.startswith("checkpoint-optstep-")
        ),
        key=lambda child: child.name,
    )
    for checkpoint in checkpoint_like:
        match = _CHECKPOINT_NAME_RE.fullmatch(checkpoint.name)
        if match is None:
            raise ValueError(
                f"malformed checkpoint-like entry in output root: {checkpoint}"
            )
        update = int(match.group(1))
        if checkpoint.name != _checkpoint_name(update):
            raise ValueError(f"non-canonical checkpoint name: {checkpoint.name}")
        _assert_real_checkpoint_tree(root, checkpoint)
        saved_contract, state = _load_resume_artifacts(
            checkpoint,
            exact_targets=[str(name) for name in exact_targets],
            weights_only=True,
        )
        if canonical_sha256(saved_contract) not in accepted_contract_shas:
            raise ValueError(f"foreign checkpoint run contract: {checkpoint.name}")
        if int(state.get("update", -1)) != update:
            raise ValueError(
                f"checkpoint update differs from its name: {checkpoint.name}"
            )
        candidates.append((update, checkpoint, _path_identity(checkpoint)))
    return sorted(candidates, key=lambda item: item[0])


def prune_checkpoints(
    *,
    output_dir: Path,
    keep_last: int,
    run_contract: Mapping[str, Any],
) -> list[Path]:
    """Delete only old, fully validated checkpoints from this exact run."""

    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    first_pass = _validated_retention_candidates(
        output_dir,
        run_contract=run_contract,
    )
    if len(first_pass) <= keep_last:
        return []
    # Revalidate the complete set before the first destructive operation. This
    # also catches a candidate swapped between discovery and pruning.
    second_pass = _validated_retention_candidates(
        output_dir,
        run_contract=run_contract,
    )
    first_snapshot = [
        (update, path.name, identity) for update, path, identity in first_pass
    ]
    second_snapshot = [
        (update, path.name, identity) for update, path, identity in second_pass
    ]
    if first_snapshot != second_snapshot:
        raise ValueError("checkpoint set changed during retention validation")

    removed: list[Path] = []
    for _update, checkpoint, identity in second_pass[:-keep_last]:
        if _is_link_like(checkpoint) or _path_identity(checkpoint) != identity:
            raise ValueError(
                f"checkpoint changed immediately before pruning: {checkpoint}"
            )
        shutil.rmtree(checkpoint)
        removed.append(checkpoint)
    return removed


def _validate_trainable_adapter_parameters(model: Any) -> None:
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("LoRA policy has no trainable parameters")
    invalid = [
        name
        for name in trainable
        if "lora_" not in name.lower() or "vision" in name.lower()
    ]
    encoder = [
        name for name in trainable if ".encoder.text_model.layers." in f".{name}"
    ]
    decoder = [name for name in trainable if ".decoder.layers." in f".{name}"]
    if invalid or not encoder or not decoder:
        raise ValueError(
            "trainable parameters are not an exact text-only encoder+decoder "
            f"LoRA set: invalid={invalid[:20]}, encoder={len(encoder)}, "
            f"decoder={len(decoder)}"
        )


def _load_or_create_policy(
    args: argparse.Namespace, *, token: str | None
) -> tuple[Any, dict[str, Any] | None, list[str]]:
    model = _load_base_model(args, token)
    exact_targets = _resolve_lora_targets(model, args)
    resume_state: dict[str, Any] | None = None
    if args.resume_checkpoint:
        from peft import PeftModel

        checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
        saved_contract, state = _load_resume_artifacts(
            checkpoint,
            exact_targets=exact_targets,
        )
        model = PeftModel.from_pretrained(
            model,
            checkpoint / "adapter",
            is_trainable=True,
        )
        resume_state = {"state": state, "contract": saved_contract}
    else:
        model = _attach_lora(model, args, exact_targets)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "decoder"):
        model.config.decoder.use_cache = False
    _validate_trainable_adapter_parameters(model)
    return model, resume_state, exact_targets


def _config_position_capacities(model: Any) -> tuple[int, int]:
    config = model.config
    _validate_t5gemma2_config(config)
    encoder = getattr(config, "encoder", None)
    encoder_text = getattr(encoder, "text_config", encoder)
    decoder = getattr(config, "decoder", None)
    source = int(getattr(encoder_text, "max_position_embeddings", 0) or 0)
    target = int(getattr(decoder, "max_position_embeddings", 0) or 0)
    if source <= 0 or target <= 0:
        raise ValueError(
            "loaded T5Gemma 2 config does not declare encoder/decoder capacity"
        )
    return source, target


def _optimizer_and_scheduler(
    model: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    warmup_updates: int,
    total_updates: int,
) -> tuple[torch.optim.Optimizer, Any]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("model has no trainable adapter parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    def multiplier(step: int) -> float:
        return cosine_schedule_multiplier(
            step,
            warmup_updates=warmup_updates,
            total_updates=total_updates,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def _tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or len(vocabulary) != len(tokenizer):
        raise ValueError("tokenizer vocabulary cannot be sealed exactly")
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "vocab_sha256": canonical_sha256(vocabulary),
        "special_tokens_map_sha256": canonical_sha256(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "resolved_commit": str(init_kwargs.get("_commit_hash") or ""),
    }


def _resolved_model_commit(
    config: Any,
    *,
    requested_revision: str,
    allow_unpinned: bool,
) -> tuple[str, str]:
    observed = str(getattr(config, "_commit_hash", None) or "").strip().lower()
    requested = requested_revision.strip().lower()
    exact_requested = bool(re.fullmatch(r"[0-9a-f]{40}", requested))
    if observed:
        if not re.fullmatch(r"[0-9a-f]{40}", observed):
            raise ValueError("Hub loader returned a malformed model commit hash")
        if exact_requested and observed != requested:
            raise ValueError("loaded model commit differs from --model_revision")
        return observed, "hub_resolved_commit"
    if exact_requested:
        return requested, "exact_requested_commit"
    if allow_unpinned:
        return "", "explicitly_unpinned"
    raise ValueError(
        "the Hub loader did not expose an immutable model commit; pass an "
        "exact 40-hex --model_revision commit or explicitly allow an "
        "unpinned model"
    )


def _runtime_contract() -> dict[str, str]:
    import peft
    import transformers

    return {
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "cuda": str(torch.version.cuda or ""),
    }


def _bind_resume_contract(
    current_contract: Mapping[str, Any],
    saved_contract: Mapping[str, Any],
    *,
    expected_legacy_trainer_sha256: str,
) -> dict[str, Any]:
    """Bind an exact resume, or one audited trainer-source-only migration."""

    current = copy.deepcopy(dict(current_contract))
    saved = copy.deepcopy(dict(saved_contract))
    current_trainer = str(current.get("runtime", {}).get("trainer_sha256") or "")
    saved_trainer = str(saved.get("runtime", {}).get("trainer_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", current_trainer) or not re.fullmatch(
        r"[0-9a-f]{64}", saved_trainer
    ):
        raise ValueError("resume contract lacks a valid trainer source hash")

    if saved_trainer == current_trainer:
        saved_migration = saved.get("resume_migration")
        if saved_migration is not None:
            current["resume_migration"] = copy.deepcopy(saved_migration)
        if canonical_sha256(current) != canonical_sha256(saved):
            raise ValueError("resume contract differs from the current run")
        return current

    if saved_trainer != expected_legacy_trainer_sha256:
        raise ValueError(
            "resume trainer source changed without the exact expected legacy hash"
        )
    compatible = copy.deepcopy(current)
    compatible["runtime"]["trainer_sha256"] = saved_trainer
    if canonical_sha256(compatible) != canonical_sha256(saved):
        raise ValueError(
            "resume migration changes more than the sealed trainer source hash"
        )
    current["resume_migration"] = {
        "schema": "trainer-source-only-resume-v1",
        "accepted_contract_sha256": canonical_sha256(saved),
        "from_trainer_sha256": saved_trainer,
        "to_trainer_sha256": current_trainer,
        "reason": "validate-peft-minimized-targets-from-exact-adapter-weight-keys",
    }
    return current


def _write_preflight(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    data_manifest: Mapping[str, Any],
    token_report: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA,
        "status": "preflight_complete",
        "architecture": "native_encoder_decoder",
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime": _runtime_contract(),
        "dataset": dict(data_manifest),
        "tokenization": dict(token_report),
        "tokenizer": _tokenizer_contract(tokenizer),
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "source_truncation": False,
        "target_truncation": False,
    }
    _atomic_json(output_dir / "preflight.json", record)
    return record


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.preflight_only:
        raise RuntimeError("T5Gemma 2 training requires CUDA")
    if args.bf16 and not args.preflight_only and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    token = os.environ.get("HF_TOKEN") or None
    tokenizer = _load_tokenizer(args.model, args.model_revision, token)
    pairs, data_manifest = load_text_pairs(
        args.train_jsonl,
        args.f2_jsonl,
        expected_dataset_sha256=args.expected_train_sha256,
        expected_f2_sha256=args.expected_f2_sha256,
        expected_rows=args.expected_rows,
        allow_unpinned_inputs=args.allow_unpinned_inputs,
    )
    tokenized, token_report = tokenize_pairs(
        tokenizer,
        pairs,
        max_source_tokens=args.max_source_tokens,
        max_target_tokens=args.max_target_tokens,
    )
    preflight = _write_preflight(
        output_dir,
        args=args,
        data_manifest=data_manifest,
        token_report=token_report,
        tokenizer=tokenizer,
    )
    if args.preflight_only:
        return preflight

    schedule = calculate_training_schedule(
        rows=len(tokenized),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_updates=args.max_updates,
        warmup_ratio=args.warmup_ratio,
    )
    total_updates = schedule["planned_updates"]
    warmup_updates = schedule["warmup_updates"]
    model, resume, exact_lora_targets = _load_or_create_policy(args, token=token)
    source_capacity, target_capacity = _config_position_capacities(model)
    if (
        int(token_report["source_tokens"]["max"]) > source_capacity
        or int(token_report["target_tokens"]["max"]) > target_capacity
    ):
        raise ValueError(
            "tokenized data exceeds loaded model capacity: "
            f"observed source/target="
            f"{token_report['source_tokens']['max']}/"
            f"{token_report['target_tokens']['max']}, "
            f"model={source_capacity}/{target_capacity}"
        )
    resolved_commit, commit_source = _resolved_model_commit(
        model.config,
        requested_revision=args.model_revision,
        allow_unpinned=args.allow_unpinned_model,
    )
    base_model_record = {
        "name": args.model,
        "requested_revision": args.model_revision,
        "resolved_commit": resolved_commit,
        "commit_source": commit_source,
        "config_sha256": canonical_sha256(model.config.to_dict()),
        "encoder_capacity": source_capacity,
        "decoder_capacity": target_capacity,
        "is_encoder_decoder": bool(model.config.is_encoder_decoder),
        "attn_implementation": args.attn_implementation,
    }
    run_contract = {
        **preflight,
        "status": "training",
        "base_model": base_model_record,
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_updates": warmup_updates,
            "planned_updates": total_updates,
            "microbatches_per_epoch": schedule["microbatches_per_epoch"],
            "updates_per_epoch": schedule["updates_per_epoch"],
            "available_updates": schedule["available_updates"],
            "seed": args.seed,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "attn_implementation": args.attn_implementation,
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": exact_lora_targets,
            "encoder_and_decoder_trainable": True,
            "vision_trainable": False,
        },
        "checkpointing": {
            "interval": args.checkpoint_interval,
            "keep_last": args.keep_last_checkpoints,
            "retention": "validate-all-then-prune-oldest-v1",
        },
    }
    if resume is not None:
        run_contract = _bind_resume_contract(
            run_contract,
            resume["contract"],
            expected_legacy_trainer_sha256=args.resume_from_trainer_sha256,
        )
    _atomic_json(output_dir / "run_contract.json", run_contract)

    device = torch.device("cuda")
    model.to(device)
    optimizer, scheduler = _optimizer_and_scheduler(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_updates=warmup_updates,
        total_updates=total_updates,
    )
    start_epoch = 0
    start_row = 0
    update = 0
    if resume is not None:
        state = resume["state"]
        start_epoch = int(state.get("epoch", -1))
        start_row = int(state.get("next_row", -1))
        update = int(state.get("update", -1))
        if (
            update < 0
            or update > total_updates
            or start_epoch < 0
            or start_epoch > args.epochs
            or start_row < 0
            or start_row > len(tokenized)
            or (start_epoch == args.epochs and start_row != 0)
            or (start_epoch < args.epochs and start_row == len(tokenized))
        ):
            raise ValueError("resume optimizer/epoch/row position lies outside the run")
        completed_microbatches = (
            math.ceil(start_row / args.batch_size) if start_row else 0
        )
        if (
            start_row % args.batch_size != 0
            or completed_microbatches % args.gradient_accumulation != 0
        ):
            raise ValueError("resume row is not an optimizer-step boundary")
        expected_update = (
            start_epoch * schedule["updates_per_epoch"]
            + completed_microbatches // args.gradient_accumulation
        )
        if update != expected_update:
            raise ValueError(
                "resume update does not match its deterministic epoch/row "
                f"position: state={update}, expected={expected_update}"
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng_state(state["rng"])

    if update == total_updates:
        final = {
            "schema": RUN_SCHEMA,
            "status": "complete",
            "updates": update,
            "planned_updates": total_updates,
            "rows": len(tokenized),
            "latest_checkpoint": _checkpoint_name(update),
            "no_frontier_api": True,
            "resumed_complete_checkpoint": True,
        }
        _atomic_json(output_dir / "result.json", final)
        return final

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    running_loss = 0.0
    journal_path = output_dir / "train_metrics.jsonl"
    stop = False
    for epoch in range(start_epoch, args.epochs):
        order = deterministic_epoch_order(tokenized, seed=args.seed, epoch=epoch)
        position = start_row if epoch == start_epoch else 0
        while position < len(order):
            indices = order[position : position + args.batch_size]
            batch_rows = [tokenized[index] for index in indices]
            batch = collate_pairs(
                batch_rows,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite SFT loss at epoch={epoch}, row={position}"
                )
            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.detach().cpu())
            accumulated += 1
            position += len(indices)
            epoch_finished = position >= len(order)
            should_step = accumulated >= args.gradient_accumulation or epoch_finished
            if not should_step:
                continue
            if accumulated < args.gradient_accumulation:
                # Losses were divided by the full accumulation factor. Restore
                # the true mean for a short final window before clipping.
                correction = args.gradient_accumulation / accumulated
                for parameter in model.parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        parameter.grad.mul_(correction)
            grad_norm = clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            metric = {
                "schema": RUN_SCHEMA,
                "update": update,
                "epoch": epoch,
                "next_row": position,
                "loss": running_loss / accumulated,
                "grad_norm": float(grad_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "microbatches": accumulated,
                "source_tokens": int(batch["attention_mask"].sum().item()),
                "target_tokens": int((batch["labels"] != -100).sum().item()),
            }
            with journal_path.open("a", encoding="utf-8", newline="\n") as handle:
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
            print(json.dumps(metric, sort_keys=True), flush=True)
            accumulated = 0
            running_loss = 0.0

            next_epoch = epoch
            next_row = position
            if next_row >= len(order):
                next_epoch = epoch + 1
                next_row = 0
            if update % args.checkpoint_interval == 0 or update >= total_updates:
                save_checkpoint(
                    output_dir=output_dir,
                    update=update,
                    epoch=next_epoch,
                    next_row=next_row,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    run_contract=run_contract,
                )
                prune_checkpoints(
                    output_dir=output_dir,
                    keep_last=args.keep_last_checkpoints,
                    run_contract=run_contract,
                )
            if update >= total_updates:
                stop = True
                break
        if stop:
            break
        start_row = 0

    final = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "updates": update,
        "planned_updates": total_updates,
        "rows": len(tokenized),
        "latest_checkpoint": _checkpoint_name(update),
        "no_frontier_api": True,
    }
    _atomic_json(output_dir / "result.json", final)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--expected_train_sha256", default="")
    parser.add_argument("--expected_f2_sha256", default="")
    parser.add_argument("--expected_rows", type=int, default=2776)
    parser.add_argument("--allow_unpinned_inputs", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default="")
    parser.add_argument("--allow_unpinned_model", action="store_true")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resume_from_trainer_sha256", default="")
    parser.add_argument("--max_source_tokens", type=int, default=32768)
    parser.add_argument("--max_target_tokens", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--max_updates", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_interval", type=int, default=25)
    parser.add_argument("--keep_last_checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa"],
        default="sdpa",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_rows <= 0:
        parser.error("--expected_rows must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch_size must be positive")
    if args.gradient_accumulation <= 0:
        parser.error("--gradient_accumulation must be positive")
    if args.max_updates < 0:
        parser.error("--max_updates cannot be negative")
    if args.learning_rate <= 0.0 or args.max_grad_norm <= 0.0:
        parser.error("learning rate and max grad norm must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must lie in [0,1)")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint_interval must be positive")
    if args.keep_last_checkpoints <= 0:
        parser.error("--keep_last_checkpoints must be positive")
    if args.resume_from_trainer_sha256 and not re.fullmatch(
        r"[0-9a-f]{64}", args.resume_from_trainer_sha256
    ):
        parser.error("--resume_from_trainer_sha256 must be a lowercase SHA-256")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("LoRA rank/alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora_dropout must lie in [0,1)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
