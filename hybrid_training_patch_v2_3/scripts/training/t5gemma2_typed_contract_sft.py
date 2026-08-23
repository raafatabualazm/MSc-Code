#!/usr/bin/env python3
"""Fresh-base T5Gemma 2 SFT with a gold-derived opaque Dart contract.

The encoder sees the same sealed F2 representation as the original native
T5Gemma arm, preceded by exactly one type-and-arity contract such as
``Set<int> fn0(List<String> p0)``.  Only ``fn0`` and neutral ``pN`` parameter
names are admitted.  The 175-task development partition is used solely for a
hash-pinned overlap check; none of its text is serialized into model input.

This module deliberately reuses the audited optimizer/checkpoint engine while
installing distinct run/checkpoint schemas and binding both engine sources in
the immutable runtime contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.training import hybrid_data_controls as controls
from scripts.training import t5gemma2_enriched_sft as engine


RUN_SCHEMA = "t5gemma2-typed-opaque-contract-sft-run-v1"
CHECKPOINT_SCHEMA = "t5gemma2-typed-opaque-contract-sft-checkpoint-v1"
CONTRACT_SCHEMA = "dart-typed-opaque-contract-v1"
CONTRACT_INSTRUCTION = (
    "Implement a top-level fn0 callable through this opaque required contract "
    "(return type and required parameter types only; parameter names are neutral): "
    "{signature}.\n"
)
_MARKER = "<enriched_binary>\n"
_TYPE_TEXT = re.compile(r"[A-Za-z_][A-Za-z0-9_<>,? ]*\Z")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")
_SAFE_DEFAULT = re.compile(
    r"(?:true|false|null|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\Z"
)

_BASE_LOAD_TEXT_PAIRS = engine.load_text_pairs
_BASE_RUNTIME_CONTRACT = engine._runtime_contract  # noqa: SLF001


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parameter_sections(text: str) -> list[tuple[str, str]]:
    """Return ``(required|optional|named, declaration)`` in source order."""

    value = text.strip()
    if not value:
        return []
    group_start = -1
    opener = ""
    depth = 0
    for index, character in enumerate(value):
        if character in "<(":
            depth += 1
        elif character in ">)":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced fn0 parameter type")
        elif character in "[{" and depth == 0:
            group_start = index
            opener = character
            break
    sections: list[tuple[str, str]] = []
    if group_start >= 0:
        closer = "]" if opener == "[" else "}"
        if not value.endswith(closer):
            raise ValueError("optional/named fn0 parameter group is malformed")
        prefix = value[:group_start].rstrip().rstrip(",").strip()
        if prefix:
            sections.extend(
                ("required", item)
                for item in controls._split_parameters(prefix)  # noqa: SLF001
                if item.strip()
            )
        inner = value[group_start + 1 : -1]
        mode = "optional" if opener == "[" else "named"
        sections.extend(
            (mode, item)
            for item in controls._split_parameters(inner)  # noqa: SLF001
            if item.strip()
        )
    else:
        sections.extend(
            ("required", item)
            for item in controls._split_parameters(value)  # noqa: SLF001
            if item.strip()
        )
    if depth != 0:
        raise ValueError("unbalanced fn0 parameter type")
    return sections


def opaque_contract_signature(gold_source: str) -> tuple[str, dict[str, Any]]:
    """Extract fn0's typed call contract while removing parameter semantics."""

    if not isinstance(gold_source, str) or not gold_source.strip():
        raise ValueError("gold source is absent for typed-contract derivation")
    source_signature = controls.extract_source_signature(gold_source, "fn0")
    if not source_signature:
        raise ValueError("could not recover fn0's source signature")
    match = re.fullmatch(r"\s*(.+?)\s+fn0\s*\((.*)\)\s*", source_signature, re.S)
    if match is None:
        raise ValueError("fn0 signature is outside the sealed grammar")
    return_type = re.sub(r"\s+", " ", match.group(1).strip())
    if not _TYPE_TEXT.fullmatch(return_type):
        raise ValueError("fn0 return type is outside the sealed type grammar")

    parsed: list[dict[str, str]] = []
    original_names: list[str] = []
    source_parameter_count = 0
    omitted_optional_parameter_count = 0
    for mode, declaration in _parameter_sections(match.group(2)):
        source_parameter_count += 1
        value = declaration.strip()
        if mode == "named" and value.startswith("required "):
            value = value[len("required ") :].strip()
            required_named = True
        else:
            required_named = False
        pieces = value.split("=", 1)
        declaration_without_default = pieces[0].strip()
        default = pieces[1].strip() if len(pieces) == 2 else ""
        explicit = re.fullmatch(
            r"(.+?)\s+([A-Za-z_]\w*)", declaration_without_default, re.S
        )
        if explicit is None:
            if not _IDENTIFIER.fullmatch(declaration_without_default):
                raise ValueError("fn0 parameter declaration is outside the sealed grammar")
            type_text = "dynamic"
            original_name = declaration_without_default
        else:
            type_text = re.sub(r"\s+", " ", explicit.group(1).strip())
            original_name = explicit.group(2)
        if not _TYPE_TEXT.fullmatch(type_text):
            raise ValueError("fn0 parameter type is outside the sealed type grammar")
        if default and not _SAFE_DEFAULT.fullmatch(default):
            raise ValueError("fn0 default value is outside the non-semantic scalar grammar")
        if mode == "required" and default:
            raise ValueError("required fn0 parameter unexpectedly has a default")
        if mode == "optional" and not default and not type_text.endswith("?"):
            raise ValueError("non-nullable optional fn0 parameter lacks a default")
        # Optional parameters are not part of the minimum callable contract.
        # Omitting them avoids leaking their default constants while remaining
        # truthful about every invocation accepted through this contract.
        if mode == "optional" or (mode == "named" and not required_named):
            omitted_optional_parameter_count += 1
            continue
        parsed.append(
            {
                "mode": mode,
                "type": type_text,
                "default": default,
                "required_named": str(required_named).lower(),
            }
        )
        original_names.append(original_name)

    required_parts: list[str] = []
    optional_parts: list[str] = []
    named_parts: list[str] = []
    parameter_types: list[str] = []
    for index, parameter in enumerate(parsed):
        type_text = parameter["type"]
        parameter_types.append(type_text)
        rendered = f"{type_text} p{index}"
        if parameter["mode"] == "required":
            required_parts.append(rendered)
        elif parameter["mode"] == "optional":
            optional_parts.append(rendered)
        else:
            if parameter["required_named"] == "true":
                rendered = "required " + rendered
            named_parts.append(rendered)
    groups = list(required_parts)
    if optional_parts:
        groups.append("[" + ", ".join(optional_parts) + "]")
    if named_parts:
        groups.append("{" + ", ".join(named_parts) + "}")
    rendered = f"{return_type} fn0(" + ", ".join(groups) + ")"

    for index, name in enumerate(original_names):
        if name != f"p{index}" and re.search(rf"\b{re.escape(name)}\b", rendered):
            raise ValueError("semantic parameter name survived opaque rendering")
    if rendered.count("fn0") != 1:
        raise ValueError("opaque contract did not render exactly one fn0")
    record = {
        "schema": CONTRACT_SCHEMA,
        "source_signature_sha256": _sha256_text(source_signature),
        "opaque_signature": rendered,
        "opaque_signature_sha256": _sha256_text(rendered),
        "return_type": return_type,
        "parameter_types": parameter_types,
        "arity": len(parsed),
        "source_parameter_count": source_parameter_count,
        "omitted_optional_parameter_count": omitted_optional_parameter_count,
        "parameter_modes": [item["mode"] for item in parsed],
        "function_name": "fn0",
        "parameter_name_policy": "p{zero_based_index}",
        "semantic_parameter_names_exposed": False,
    }
    return rendered, record


def build_typed_encoder_source(
    f2_row: Mapping[str, Any], task_id: str, target: str
) -> tuple[str, dict[str, Any]]:
    base_source = engine.build_encoder_source(f2_row, task_id)
    if base_source.count(_MARKER) != 1:
        raise ValueError(f"{task_id}: enriched-binary marker is not unique")
    signature, record = opaque_contract_signature(target)
    instruction = CONTRACT_INSTRUCTION.format(signature=signature)
    source = base_source.replace(_MARKER, instruction + _MARKER, 1)
    if source.count(instruction) != 1:
        raise ValueError(f"{task_id}: typed contract was not inserted exactly once")
    return source, record


def _heldout_record(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    allow_unpinned_inputs: bool,
) -> dict[str, Any]:
    observed_sha = engine._pin_file(  # noqa: SLF001
        heldout_path, expected_sha256, allow_unpinned=allow_unpinned_inputs
    )
    heldout_rows = engine._read_jsonl(heldout_path)  # noqa: SLF001
    if len(heldout_rows) != expected_rows:
        raise ValueError(
            f"heldout row-count mismatch: observed={len(heldout_rows)}, "
            f"expected={expected_rows}"
        )
    train_ids = [engine._identity(row, index) for index, row in enumerate(train_rows)]  # noqa: SLF001
    heldout_ids = [
        engine._identity(row, index) for index, row in enumerate(heldout_rows)  # noqa: SLF001
    ]
    if len(set(heldout_ids)) != len(heldout_ids):
        raise ValueError("heldout contains duplicate task IDs")
    overlap = sorted(set(train_ids) & set(heldout_ids))
    if overlap:
        raise ValueError(f"training/heldout task-ID overlap: {overlap[:10]}")
    train_targets = {
        _sha256_text(engine._target_source(row, task_id))  # noqa: SLF001
        for task_id, row in zip(train_ids, train_rows, strict=True)
    }
    heldout_targets = {
        _sha256_text(engine._target_source(row, task_id))  # noqa: SLF001
        for task_id, row in zip(heldout_ids, heldout_rows, strict=True)
    }
    source_overlap = train_targets & heldout_targets
    if source_overlap:
        raise ValueError("training/heldout exact gold-source overlap")

    def acceptance_hash(row: Mapping[str, Any]) -> str:
        value = row.get("acceptance_tests") or row.get("tests")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("row is missing acceptance tests for contamination audit")
        return _sha256_text(value)

    train_acceptance = {acceptance_hash(row) for row in train_rows}
    heldout_acceptance = {acceptance_hash(row) for row in heldout_rows}
    acceptance_overlap = train_acceptance & heldout_acceptance
    if acceptance_overlap:
        raise ValueError("training/heldout exact acceptance-test overlap")
    return {
        "sha256": observed_sha,
        "rows": len(heldout_rows),
        "task_ids_sha256": engine.canonical_sha256(heldout_ids),
        "task_id_overlap": 0,
        "exact_gold_source_overlap": 0,
        "exact_acceptance_test_overlap": 0,
        "model_visible": False,
    }


def load_typed_text_pairs(
    dataset_path: str | Path,
    f2_path: str | Path,
    *,
    expected_dataset_sha256: str,
    expected_f2_sha256: str,
    expected_rows: int,
    heldout_path: str | Path,
    expected_heldout_sha256: str,
    expected_heldout_rows: int,
    exclude_train_task_ids: Sequence[str] = (),
    allow_unpinned_inputs: bool = False,
) -> tuple[list[engine.TextPair], dict[str, Any]]:
    pairs, manifest = _BASE_LOAD_TEXT_PAIRS(
        dataset_path,
        f2_path,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_f2_sha256=expected_f2_sha256,
        expected_rows=expected_rows,
        allow_unpinned_inputs=allow_unpinned_inputs,
    )
    dataset_rows = engine._read_jsonl(dataset_path)  # noqa: SLF001
    f2_rows = engine._read_jsonl(f2_path)  # noqa: SLF001

    requested_exclusions = list(exclude_train_task_ids)
    if len(set(requested_exclusions)) != len(requested_exclusions):
        raise ValueError("duplicate requested training exclusion")
    dataset_ids = [
        engine._identity(row, index) for index, row in enumerate(dataset_rows)  # noqa: SLF001
    ]
    missing_exclusions = sorted(set(requested_exclusions) - set(dataset_ids))
    if missing_exclusions:
        raise ValueError(f"requested training exclusions are absent: {missing_exclusions}")
    keep = [task_id not in set(requested_exclusions) for task_id in dataset_ids]
    excluded_rows = [
        row for row, retained in zip(dataset_rows, keep, strict=True) if not retained
    ]
    pairs = [pair for pair, retained in zip(pairs, keep, strict=True) if retained]
    dataset_rows = [
        row for row, retained in zip(dataset_rows, keep, strict=True) if retained
    ]
    f2_rows = [row for row, retained in zip(f2_rows, keep, strict=True) if retained]
    retained_ids = [
        task_id for task_id, retained in zip(dataset_ids, keep, strict=True) if retained
    ]
    heldout = _heldout_record(
        dataset_rows,
        heldout_path,
        expected_sha256=expected_heldout_sha256,
        expected_rows=expected_heldout_rows,
        allow_unpinned_inputs=allow_unpinned_inputs,
    )
    typed_pairs: list[engine.TextPair] = []
    contracts: list[dict[str, Any]] = []
    for pair, f2_row in zip(pairs, f2_rows, strict=True):
        source, contract = build_typed_encoder_source(f2_row, pair.task_id, pair.target)
        typed_pairs.append(
            replace(
                pair,
                source=source,
                source_sha256=_sha256_text(source),
            )
        )
        contracts.append(contract)
    manifest = {
        **manifest,
        "schema": RUN_SCHEMA,
        "input_rows": manifest["rows"],
        "rows": len(typed_pairs),
        "task_ids_sha256": engine.canonical_sha256(retained_ids),
        "source_sha256s_sha256": engine.canonical_sha256(
            [pair.source_sha256 for pair in typed_pairs]
        ),
        "target_sha256s_sha256": engine.canonical_sha256(
            [pair.target_sha256 for pair in typed_pairs]
        ),
        "training_exclusions": {
            "count": len(excluded_rows),
            "task_ids": requested_exclusions,
            "task_ids_sha256": engine.canonical_sha256(requested_exclusions),
            "acceptance_tests_sha256": [
                _sha256_text(str(row.get("acceptance_tests") or row.get("tests")))
                for row in excluded_rows
            ],
            "reason": "exact heldout acceptance-test duplicate",
        },
        "model_visible_fields": ["opaque_typed_contract", "F2.text"],
        "opaque_contract": {
            "schema": CONTRACT_SCHEMA,
            "rows": len(contracts),
            "contracts_sha256": engine.canonical_sha256(contracts),
            "arity_histogram": dict(
                sorted(Counter(str(item["arity"]) for item in contracts).items())
            ),
            "parameter_name_policy": "p{zero_based_index}",
            "semantic_function_name_exposed": False,
            "semantic_parameter_names_exposed": False,
        },
        "heldout": heldout,
        "reference_target_field": "dart_source",
    }
    return typed_pairs, manifest


def _typed_runtime_contract() -> dict[str, str]:
    record = dict(_BASE_RUNTIME_CONTRACT())
    record["training_engine_sha256"] = record["trainer_sha256"]
    record["trainer_sha256"] = engine.sha256_file(Path(__file__).resolve())
    record["trainer_profile"] = "typed_opaque_contract_fresh_base"
    return record


def train(args: argparse.Namespace) -> dict[str, Any]:
    original = {
        "run_schema": engine.RUN_SCHEMA,
        "checkpoint_schema": engine.CHECKPOINT_SCHEMA,
        "load_text_pairs": engine.load_text_pairs,
        "runtime_contract": engine._runtime_contract,  # noqa: SLF001
    }

    def profile_loader(*positional: Any, **keywords: Any):
        return load_typed_text_pairs(
            *positional,
            **keywords,
            heldout_path=args.heldout_jsonl,
            expected_heldout_sha256=args.expected_heldout_sha256,
            expected_heldout_rows=args.expected_heldout_rows,
            exclude_train_task_ids=args.exclude_train_task_id,
        )

    engine.RUN_SCHEMA = RUN_SCHEMA
    engine.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    engine.load_text_pairs = profile_loader
    engine._runtime_contract = _typed_runtime_contract  # noqa: SLF001
    try:
        return engine.train(args)
    finally:
        engine.RUN_SCHEMA = original["run_schema"]
        engine.CHECKPOINT_SCHEMA = original["checkpoint_schema"]
        engine.load_text_pairs = original["load_text_pairs"]
        engine._runtime_contract = original["runtime_contract"]  # noqa: SLF001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--heldout_jsonl", required=True)
    parser.add_argument("--expected_train_sha256", required=True)
    parser.add_argument("--expected_f2_sha256", required=True)
    parser.add_argument("--expected_heldout_sha256", required=True)
    parser.add_argument("--expected_rows", type=int, default=2776)
    parser.add_argument("--expected_heldout_rows", type=int, default=175)
    parser.add_argument("--exclude_train_task_id", action="append", default=[])
    parser.add_argument("--allow_unpinned_inputs", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=engine.DEFAULT_MODEL)
    parser.add_argument("--model_revision", default="")
    parser.add_argument("--allow_unpinned_model", action="store_true")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--resume_from_trainer_sha256", default="")
    parser.add_argument("--max_source_tokens", type=int, default=32768)
    parser.add_argument("--max_target_tokens", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--max_updates", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--keep_last_checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_rows <= 0 or args.expected_heldout_rows <= 0:
        parser.error("expected row counts must be positive")
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        parser.error("epochs, batch size, and gradient accumulation must be positive")
    if args.max_updates < 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0:
        parser.error("invalid update or optimizer settings")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.lora_dropout < 1:
        parser.error("warmup ratio and LoRA dropout must lie in [0,1)")
    if args.checkpoint_interval <= 0 or args.keep_last_checkpoints <= 0:
        parser.error("checkpoint settings must be positive")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("LoRA rank/alpha must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
