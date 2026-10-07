#!/usr/bin/env python3
"""Train Qwen directly on compact source tokens, with no neural encoder.

The compact codec emits ``compact_input_ids``.  These IDs are concatenated with
ordinary prompt and Dart target IDs and passed through one causal decoder.  This
entry point deliberately does not import graph, CFG, PyG, prefix, or encoder
modules.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from models.direct_compact_causal import (
    CONTRACT_SCHEMA_V3,
    DirectCompactBatchCollator,
    DirectCompactCausalLM,
    DirectCompactContract,
    install_source_embedding_overlay,
    migrate_source_embedding_overlay,
    restore_source_embedding_overlay,
    resolve_decoder_config_path,
    sha256_artifact,
    sha256_file,
    validate_base_model_vocab,
    validate_join_seal,
)

DIRECT_PROMPT_MODE_CODE_ONLY_V1 = "code_only_v1"
DIRECT_PROMPT_MODE_QWEN_COT_V1 = "qwen_cot_v1"
DIRECT_PROMPT_MODES = frozenset(
    {
        DIRECT_PROMPT_MODE_CODE_ONLY_V1,
        DIRECT_PROMPT_MODE_QWEN_COT_V1,
    }
)
DIRECT_TRAINER_RESUME_SCHEMA = "direct-compact-trainer-resume-v1"
DIRECT_TRAINER_RESUME_FILENAME = "direct_trainer_resume.json"
_FORBIDDEN_ROOT_MODEL_STATE_NAMES = frozenset(
    {
        "adapter_model.bin",
        "adapter_model.safetensors",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
)


def _encode(tokenizer: Any, text: str, *, special: bool) -> list[int]:
    result = tokenizer(
        text,
        add_special_tokens=special,
        truncation=False,
        padding=False,
    )
    return list(result["input_ids"])


def direct_prompt(
    row: Mapping[str, Any], *, target_function: str | None = None,
    target_language: str | None = None,
) -> str:
    prompt_mode = str(
        row.get("direct_prompt_mode") or DIRECT_PROMPT_MODE_CODE_ONLY_V1
    ).strip()
    if prompt_mode not in DIRECT_PROMPT_MODES:
        raise ValueError(
            "unsupported direct_prompt_mode "
            f"{prompt_mode!r}; expected one of {sorted(DIRECT_PROMPT_MODES)}"
        )
    language = str(target_language or row.get("language") or row.get("lang") or "Dart")
    function_name = str(
        target_function or row.get("function") or row.get("name") or "fn0"
    )
    code_only_prompt = (
        f"Decompile the following compact binary representation to {language}.\n"
        "Return one self-contained compilable source-unit fragment, including "
        "required imports and top-level helpers, without markdown, prose, "
        "tests, or demos.\n"
        f"The fragment must define a top-level function named exactly {function_name}.\n"
        f"Do not define main. Infer {function_name}'s return type and complete parameter "
        "contract from the binary.\n"
        "Compact binary tokens follow:\n"
    )
    if prompt_mode == DIRECT_PROMPT_MODE_CODE_ONLY_V1:
        # This is intentionally the byte-identical pre-mode prompt. Production
        # inference and all non-CoT training rows default to this branch.
        return code_only_prompt
    return (
        f"Decompile the following compact binary representation to {language}.\n"
        "First reason about the reconstruction inside exactly one "
        "<think>...</think> block. Then return one self-contained compilable "
        "source-unit fragment, including required imports and top-level "
        "helpers, without markdown, prose after </think>, tests, or demos.\n"
        f"The fragment must define a top-level function named exactly {function_name}.\n"
        f"Do not define main. Infer {function_name}'s return type and complete parameter "
        "contract from the binary.\n"
        "Compact binary tokens follow:\n"
    )


def target_source(row: Mapping[str, Any], identity: str) -> str:
    value = (
        row.get("supervised_target")
        or row.get("dart_source")
        or row.get("source")
        or ""
    )
    prompt_mode = str(
        row.get("direct_prompt_mode") or DIRECT_PROMPT_MODE_CODE_ONLY_V1
    ).strip()
    if prompt_mode not in DIRECT_PROMPT_MODES:
        raise ValueError(
            f"{identity}: unsupported direct_prompt_mode {prompt_mode!r}"
        )
    raw = str(value)
    # Qwen CoT artifacts bind the exact raw reasoning and final-content byte
    # strings. Do not apply the ordinary outer-whitespace normalization to
    # those mode-conditioned targets.
    result = (
        raw
        if prompt_mode == DIRECT_PROMPT_MODE_QWEN_COT_V1
        else raw.strip()
    )
    if not result or not result.strip():
        raise ValueError(f"{identity}: missing supervised Dart source")
    return result


def copy_exact_contract(source: str | Path, destination: str | Path) -> None:
    """Copy the sealed input contract byte-for-byte into a checkpoint.

    ``DirectCompactContract`` intentionally models only fields consumed by the
    trainer.  Re-serializing that model would silently omit release provenance,
    producing a plausible-looking but unsealed reload contract.  Checkpoints
    therefore carry the exact validated input artifact instead.
    """

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    if sha256_file(source_path) != sha256_file(destination_path):
        raise RuntimeError("checkpoint contract copy is not byte-identical")


def _canonical_json_sha256(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write a JSON seal last, without exposing a plausible partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise ValueError(f"checkpoint artifact is neither a file nor directory: {path}")
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file()
    )


def _checkpoint_step(checkpoint: str | Path) -> int:
    root = Path(checkpoint).expanduser().resolve()
    match = re.fullmatch(r"checkpoint-([0-9]+)", root.name)
    if match is None:
        raise ValueError(
            "direct trainer resume checkpoint must be named checkpoint-N"
        )
    step = int(match.group(1))
    if step <= 0:
        raise ValueError("direct trainer resume checkpoint step must be positive")
    return step


def _checkpoint_artifact_bindings(checkpoint: str | Path) -> dict[str, Any]:
    """Hash every state component needed for an exact Trainer continuation."""

    root = Path(checkpoint).expanduser().resolve()
    required: dict[str, Path] = {
        "decoder_adapter": root / "decoder_adapter",
        "source_embedding_overlay": root / "source_embedding_overlay.pt",
        "compact_contract": root / "compact_contract.json",
        "optimizer": root / "optimizer.pt",
        "scheduler": root / "scheduler.pt",
        "rng_state": root / "rng_state.pth",
        "trainer_state": root / "trainer_state.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ValueError(
            "resume checkpoint lacks exact-continuation artifacts: "
            + ", ".join(missing)
        )
    if not required["decoder_adapter"].is_dir():
        raise ValueError("resume decoder_adapter must be a directory")
    if not (required["decoder_adapter"] / "adapter_config.json").is_file():
        raise ValueError("resume decoder_adapter has no adapter_config.json")
    for name, path in required.items():
        if name != "decoder_adapter" and not path.is_file():
            raise ValueError(f"resume checkpoint artifact is not a file: {path}")
    bindings = {
        name: {
            "sha256": (
                sha256_artifact(path) if path.is_dir() else sha256_file(path)
            ),
            "size_bytes": _artifact_size(path),
        }
        for name, path in required.items()
    }
    scaler = root / "scaler.pt"
    if scaler.exists():
        if not scaler.is_file():
            raise ValueError("resume scaler state is not a file")
        bindings["scaler"] = {
            "sha256": sha256_file(scaler),
            "size_bytes": scaler.stat().st_size,
        }
    return bindings


def _mapping_difference_paths(
    expected: Any,
    observed: Any,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        differences: list[str] = []
        keys = sorted(set(expected) | set(observed), key=str)
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in observed:
                differences.append(path)
            else:
                differences.extend(
                    _mapping_difference_paths(
                        expected[key],
                        observed[key],
                        prefix=path,
                    )
                )
        return differences
    if expected != observed:
        return [prefix or "<root>"]
    return []


def write_direct_trainer_resume_metadata(
    checkpoint: str | Path,
    *,
    compatibility: Mapping[str, Any],
    global_step: int,
) -> Path:
    """Seal model, optimizer, scheduler, RNG, and immutable launch inputs."""

    root = Path(checkpoint).expanduser().resolve()
    step = _checkpoint_step(root)
    if int(global_step) != step:
        raise ValueError(
            f"checkpoint directory step {step} differs from trainer step "
            f"{global_step}"
        )
    trainer_state = _load_json_object(
        root / "trainer_state.json", "direct trainer state"
    )
    if int(trainer_state.get("global_step", -1)) != step:
        raise ValueError(
            "trainer_state global_step differs from its checkpoint directory"
        )
    compatibility_object = dict(compatibility)
    metadata = {
        "schema": DIRECT_TRAINER_RESUME_SCHEMA,
        "architecture": "qwen-causal-compact-tokens-no-encoder",
        "global_step": step,
        "compatibility": compatibility_object,
        "compatibility_sha256": _canonical_mapping_sha256(
            compatibility_object
        ),
        "checkpoint_artifacts": _checkpoint_artifact_bindings(root),
    }
    destination = root / DIRECT_TRAINER_RESUME_FILENAME
    _write_json_atomic(destination, metadata)
    return destination


def validate_direct_trainer_resume_checkpoint(
    checkpoint: str | Path,
    *,
    expected_compatibility: Mapping[str, Any],
) -> dict[str, Path]:
    """Fail closed unless a checkpoint is an exact continuation of this run."""

    root = Path(checkpoint).expanduser().resolve()
    step = _checkpoint_step(root)
    unexpected_root_weights = sorted(
        name for name in _FORBIDDEN_ROOT_MODEL_STATE_NAMES
        if (root / name).exists()
    )
    if unexpected_root_weights:
        raise ValueError(
            "resume checkpoint contains ambiguous root model state; direct "
            "checkpoints must store only decoder_adapter/: "
            + ", ".join(unexpected_root_weights)
        )
    metadata_path = root / DIRECT_TRAINER_RESUME_FILENAME
    if not metadata_path.is_file():
        raise ValueError(
            f"resume checkpoint has no {DIRECT_TRAINER_RESUME_FILENAME} seal"
        )
    metadata = _load_json_object(
        metadata_path, "direct trainer resume metadata"
    )
    if metadata.get("schema") != DIRECT_TRAINER_RESUME_SCHEMA:
        raise ValueError("resume checkpoint has an unknown resume schema")
    if metadata.get("architecture") != (
        "qwen-causal-compact-tokens-no-encoder"
    ):
        raise ValueError("resume checkpoint architecture differs")
    if int(metadata.get("global_step", -1)) != step:
        raise ValueError("resume metadata global_step differs from checkpoint")
    observed_compatibility = metadata.get("compatibility")
    if not isinstance(observed_compatibility, Mapping):
        raise ValueError("resume metadata compatibility must be an object")
    expected_object = dict(expected_compatibility)
    expected_hash = _canonical_mapping_sha256(expected_object)
    if metadata.get("compatibility_sha256") != _canonical_mapping_sha256(
        observed_compatibility
    ):
        raise ValueError("resume compatibility payload differs from its hash")
    differences = _mapping_difference_paths(
        expected_object, observed_compatibility
    )
    if differences or metadata.get("compatibility_sha256") != expected_hash:
        shown = ", ".join(differences[:12]) or "compatibility_sha256"
        if len(differences) > 12:
            shown += f", ... (+{len(differences) - 12})"
        raise ValueError(
            "resume checkpoint is incompatible with immutable launch inputs: "
            + shown
        )
    observed_artifacts = metadata.get("checkpoint_artifacts")
    if not isinstance(observed_artifacts, Mapping):
        raise ValueError("resume checkpoint artifact bindings must be an object")
    current_artifacts = _checkpoint_artifact_bindings(root)
    artifact_differences = _mapping_difference_paths(
        current_artifacts, observed_artifacts
    )
    if artifact_differences:
        raise ValueError(
            "resume checkpoint state differs from its artifact bindings: "
            + ", ".join(artifact_differences[:12])
        )
    trainer_state = _load_json_object(
        root / "trainer_state.json", "direct trainer state"
    )
    if int(trainer_state.get("global_step", -1)) != step:
        raise ValueError(
            "trainer_state global_step differs from checkpoint metadata"
        )
    return {
        "root": root,
        "adapter": root / "decoder_adapter",
        "overlay": root / "source_embedding_overlay.pt",
        "contract": root / "compact_contract.json",
        "metadata": metadata_path,
    }


def make_direct_trainer_class(
    trainer_base: type,
    *,
    adapter_model: Any,
    overlay: Any,
    tokenizer: Any,
    contract_path: str | Path,
    authorized_resume_checkpoint: str | Path | None,
    resume_compatibility: Mapping[str, Any],
) -> type:
    """Build the Trainer shim that owns the split adapter+overlay checkpoint."""

    authorized = (
        None
        if authorized_resume_checkpoint is None
        else Path(authorized_resume_checkpoint).expanduser().resolve()
    )

    class DirectTrainer(trainer_base):
        _direct_resume_model_state_verified = False

        def _save(self, output_dir: str | None = None, state_dict=None) -> None:
            del state_dict
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            # PEFT saves only adapter weights. The compact overlay is separately
            # persisted and never causes the base embedding or LM head to expand.
            if not hasattr(adapter_model, "save_pretrained"):
                raise RuntimeError("direct checkpoint adapter cannot be saved")
            adapter_model.save_pretrained(destination / "decoder_adapter")
            torch.save(
                overlay.overlay_state(),
                destination / "source_embedding_overlay.pt",
            )
            tokenizer.save_pretrained(destination / "tokenizer")
            copy_exact_contract(
                contract_path,
                destination / "compact_contract.json",
            )

        def _save_checkpoint(self, model_to_save, trial) -> None:
            # The base Trainer writes optimizer/scheduler/RNG/trainer_state.
            # Our seal is deliberately written only after all of those succeed.
            super()._save_checkpoint(model_to_save, trial)
            if not self.args.should_save:
                return
            run_dir = Path(self._get_output_dir(trial=trial))
            destination = run_dir / f"checkpoint-{self.state.global_step}"
            write_direct_trainer_resume_metadata(
                destination,
                compatibility=resume_compatibility,
                global_step=int(self.state.global_step),
            )

        def _load_from_checkpoint(
            self,
            resume_from_checkpoint: str,
            model: Any | None = None,
        ) -> None:
            # main() has already constructed PEFT and the compact overlay from
            # the sealed nested artifacts. Never delegate to Trainer: it only
            # understands root model/adapter files and would either reject this
            # layout or load the wrong object while silently leaving the overlay
            # at its warm-start state.
            del model
            candidate = Path(resume_from_checkpoint).expanduser().resolve()
            if authorized is None or candidate != authorized:
                raise ValueError(
                    "Trainer attempted to load an unauthorized direct checkpoint"
                )
            validate_direct_trainer_resume_checkpoint(
                candidate,
                expected_compatibility=resume_compatibility,
            )
            self._direct_resume_model_state_verified = True

    return DirectTrainer


def validate_self_sealed_checkpoint(
    checkpoint: str | Path,
) -> dict[str, Path]:
    """Validate a direct-compact checkpoint against its own sealed contract."""

    root = Path(checkpoint).expanduser().resolve()
    if root == Path.cwd().resolve():
        raise ValueError("warm-start checkpoint may not be the current directory")
    paths = {
        "root": root,
        "adapter": root / "decoder_adapter",
        "overlay": root / "source_embedding_overlay.pt",
        "contract": root / "compact_contract.json",
        "provenance": root / "run_provenance.json",
    }
    missing = [
        name
        for name, path in paths.items()
        if name != "root" and not path.exists()
    ]
    if missing:
        raise ValueError(
            "direct-compact warm-start checkpoint is incomplete; missing "
            + ", ".join(missing)
        )
    if not paths["adapter"].is_dir():
        raise ValueError("warm-start decoder_adapter must be a directory")
    provenance = _load_json_object(
        paths["provenance"], "warm-start run provenance"
    )
    if provenance.get("schema") != "direct-compact-run-provenance-v1":
        raise ValueError("warm-start checkpoint has an unknown provenance schema")
    if provenance.get("architecture") != "qwen-causal-compact-tokens-no-encoder":
        raise ValueError("warm-start checkpoint is not a direct-compact causal model")
    contract = DirectCompactContract.load(paths["contract"])
    expected_bindings = {
        "decoder_model": contract.decoder_model,
        "decoder_revision": contract.decoder_revision,
        "contract_sha256": sha256_file(paths["contract"]),
        "source_overlay_sha256": sha256_file(paths["overlay"]),
        "decoder_adapter_sha256": sha256_artifact(paths["adapter"]),
    }
    mismatches = [
        field
        for field, expected in expected_bindings.items()
        if provenance.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "warm-start checkpoint artifacts do not match their sealed provenance: "
            + ", ".join(mismatches)
        )
    return paths


def validate_warmstart_checkpoint(
    checkpoint: str | Path,
    *,
    contract_path: str | Path,
) -> dict[str, Path]:
    """Validate the adapter+overlay checkpoint used for continued SFT.

    Direct-compact checkpoints are a three-part object: decoder LoRA, the
    input-only compact-token overlay, and the exact compact contract.  Loading
    only the LoRA silently resets the representation learned by the student,
    while accepting a different contract changes the conditioning domain.
    """

    paths = validate_self_sealed_checkpoint(checkpoint)
    current_contract = Path(contract_path).expanduser().resolve()
    if _canonical_json_sha256(paths["contract"]) != _canonical_json_sha256(
        current_contract
    ):
        raise ValueError(
            "warm-start compact contract differs from the selected training contract"
        )
    return paths


def validate_overlay_migration_contracts(
    old_contract_path: str | Path,
    new_contract_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless only embedding bindings or context limits changed."""

    old_path = Path(old_contract_path).expanduser().resolve()
    new_path = Path(new_contract_path).expanduser().resolve()
    old_raw = _load_json_object(old_path, "old compact contract")
    new_raw = _load_json_object(new_path, "new compact contract")
    old_contract = DirectCompactContract.from_mapping(old_raw)
    new_contract = DirectCompactContract.from_mapping(new_raw)
    migratable_fields = frozenset(
        {
            "codec_sha256",
            "codebook_sha256",
            "source_token_expansions",
            "max_target_tokens",
            "max_total_tokens",
        }
    )
    old_fixed = {
        key: value for key, value in old_raw.items() if key not in migratable_fields
    }
    new_fixed = {
        key: value for key, value in new_raw.items() if key not in migratable_fields
    }
    if old_fixed != new_fixed:
        changed = sorted(
            key
            for key in set(old_fixed).union(new_fixed)
            if old_fixed.get(key) != new_fixed.get(key)
        )
        raise ValueError(
            "compact contracts are not overlay-migration compatible; "
            "non-migratable fields changed: "
            + ", ".join(changed)
        )
    if old_contract.source_token_ids != new_contract.source_token_ids:
        raise ValueError(
            "compact contracts changed the stable source-token ID sequence"
        )
    if old_contract.base_vocab_size != new_contract.base_vocab_size:
        raise ValueError("compact contracts changed the frozen base vocabulary")
    old_expansions = dict(old_contract.source_token_expansions)
    new_expansions = dict(new_contract.source_token_expansions)
    identical = [
        source_id
        for source_id in new_contract.source_token_ids
        if old_expansions[source_id] == new_expansions[source_id]
    ]
    changed = [
        source_id
        for source_id in new_contract.source_token_ids
        if old_expansions[source_id] != new_expansions[source_id]
    ]
    return {
        "schema": "direct-compact-overlay-migration-compatibility-v1",
        "old_contract_sha256": sha256_file(old_path),
        "new_contract_sha256": sha256_file(new_path),
        "allowed_changed_fields": sorted(migratable_fields),
        "observed_changed_fields": sorted(
            key
            for key in migratable_fields
            if old_raw.get(key) != new_raw.get(key)
        ),
        "source_token_rows": len(new_contract.source_token_ids),
        "identical_expansion_rows": len(identical),
        "changed_expansion_rows": len(changed),
        "identical_expansion_source_token_ids": identical,
        "changed_expansion_source_token_ids": changed,
        "all_non_migratable_contract_fields_identical": True,
        "stable_source_token_id_sequence_identical": True,
        "base_vocab_size_identical": True,
    }


def materialize_overlay_migrated_checkpoint(
    *,
    model: torch.nn.Module,
    source_checkpoint: str | Path,
    new_contract_path: str | Path,
    codebook_path: str | Path,
    codec_path: str | Path,
    output_dir: str | Path,
    stage_contract_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an adapter-identical checkpoint under a refitted contract.

    No optimizer step is taken.  The adapter tree is copied byte-for-byte, and
    the source overlay is migrated with the exact-expansion-only policy.
    """

    source = validate_self_sealed_checkpoint(source_checkpoint)
    compatibility = validate_overlay_migration_contracts(
        source["contract"], new_contract_path
    )
    old_contract = DirectCompactContract.load(source["contract"])
    new_contract = DirectCompactContract.load(new_contract_path)
    if compatibility["source_token_rows"] <= 0:
        raise ValueError("overlay migration has no source-token rows")
    lm_head = model.get_output_embeddings()
    if lm_head is None or int(lm_head.weight.size(0)) != int(
        new_contract.base_vocab_size or 0
    ):
        raise ValueError("base model LM head differs from the new compact contract")
    overlay, migration = migrate_source_embedding_overlay(
        model,
        old_source_token_expansions=dict(
            old_contract.source_token_expansions
        ),
        new_source_token_expansions=dict(
            new_contract.source_token_expansions
        ),
        checkpoint=source["overlay"],
        base_vocab_size=int(new_contract.base_vocab_size or 0),
    )
    if (
        migration["reused_source_token_ids"]
        != compatibility["identical_expansion_source_token_ids"]
        or migration["reinitialized_source_token_ids"]
        != compatibility["changed_expansion_source_token_ids"]
    ):
        raise RuntimeError(
            "overlay migration implementation disagrees with contract audit"
        )
    if int(model.get_output_embeddings().weight.size(0)) != int(
        new_contract.base_vocab_size or 0
    ):
        raise RuntimeError("overlay migration unexpectedly resized the LM head")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite overlay-migration output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.migration.",
            dir=destination.parent,
        )
    )
    try:
        shutil.copytree(source["adapter"], temporary / "decoder_adapter")
        if sha256_artifact(temporary / "decoder_adapter") != sha256_artifact(
            source["adapter"]
        ):
            raise RuntimeError("migration changed the decoder adapter tree")
        torch.save(
            overlay.overlay_state(),
            temporary / "source_embedding_overlay.pt",
        )
        copy_exact_contract(
            new_contract_path, temporary / "compact_contract.json"
        )
        receipt = {
            "schema": "direct-compact-overlay-migration-receipt-v1",
            "created_at": _utc_now(),
            "training_steps": 0,
            "source_checkpoint": {
                "path": str(source["root"]),
                "decoder_adapter_sha256": sha256_artifact(
                    source["adapter"]
                ),
                "source_overlay_sha256": sha256_file(source["overlay"]),
                "contract_sha256": sha256_file(
                    source["contract"]
                ),
                "provenance_sha256": sha256_file(
                    source["provenance"]
                ),
            },
            "contract_compatibility": compatibility,
            "overlay_migration": migration,
            "outputs": {
                "decoder_adapter_sha256": sha256_artifact(
                    temporary / "decoder_adapter"
                ),
                "source_overlay_sha256": sha256_file(
                    temporary / "source_embedding_overlay.pt"
                ),
                "compact_contract_sha256": sha256_file(
                    temporary / "compact_contract.json"
                ),
                "codebook_sha256": sha256_file(codebook_path),
                "codec_sha256": sha256_file(codec_path),
            },
            "invariants": {
                "no_training_or_optimizer_step_performed": True,
                "decoder_adapter_tree_byte_identical": True,
                "old_overlay_row_reused_only_for_identical_expansion": True,
                "changed_rows_use_new_codebook_mean_initialization": True,
                "new_contract_copied_byte_identically": True,
                "heldout_data_opened": False,
            },
        }
        _write_json(temporary / "overlay_migration_receipt.json", receipt)
        provenance = {
            "schema": "direct-compact-run-provenance-v1",
            "architecture": "qwen-causal-compact-tokens-no-encoder",
            "checkpoint_stage": "contract-overlay-migration-only",
            "decoder_model": new_contract.decoder_model,
            "decoder_revision": new_contract.decoder_revision,
            "model_config_sha256": new_contract.model_config_sha256,
            "contract_sha256": sha256_file(
                temporary / "compact_contract.json"
            ),
            "codebook_sha256": sha256_file(codebook_path),
            "codec_sha256": sha256_file(codec_path),
            "source_overlay_sha256": sha256_file(
                temporary / "source_embedding_overlay.pt"
            ),
            "decoder_adapter_sha256": sha256_artifact(
                temporary / "decoder_adapter"
            ),
            "source_embedding_overlay_rows": len(
                new_contract.source_token_ids
            ),
            "lm_head_rows": int(
                model.get_output_embeddings().weight.size(0)
            ),
            "training_performed": False,
            "heldout_loaded_during_migration": False,
            "stage_contract": (
                None
                if stage_contract_record is None
                else dict(stage_contract_record)
            ),
            "overlay_migration_receipt_sha256": sha256_file(
                temporary / "overlay_migration_receipt.json"
            ),
            "warmstart_checkpoint": receipt["source_checkpoint"],
        }
        _write_json(temporary / "run_provenance.json", provenance)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_warmstart_checkpoint(
        destination, contract_path=new_contract_path
    )
    return {
        "checkpoint": str(destination),
        "receipt": str(destination / "overlay_migration_receipt.json"),
        "receipt_sha256": sha256_file(
            destination / "overlay_migration_receipt.json"
        ),
        "source_overlay_sha256": sha256_file(
            destination / "source_embedding_overlay.pt"
        ),
        "decoder_adapter_sha256": sha256_artifact(
            destination / "decoder_adapter"
        ),
        "reused_rows": migration["rows"]["reused_identical_expansion"],
        "reinitialized_rows": migration["rows"][
            "reinitialized_new_codebook_mean"
        ],
    }


def validate_overlay_migrated_checkpoint(
    *,
    checkpoint: str | Path,
    source_checkpoint: str | Path,
    new_contract_path: str | Path,
    codebook_path: str | Path,
    codec_path: str | Path,
) -> dict[str, Any]:
    """Validate and reuse a completed overlay-only migration checkpoint."""

    destination = validate_warmstart_checkpoint(
        checkpoint, contract_path=new_contract_path
    )
    source = validate_self_sealed_checkpoint(source_checkpoint)
    compatibility = validate_overlay_migration_contracts(
        source["contract"], new_contract_path
    )
    receipt_path = destination["root"] / "overlay_migration_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("migrated checkpoint has no migration receipt")
    receipt = _load_json_object(receipt_path, "overlay migration receipt")
    provenance = _load_json_object(
        destination["provenance"], "migrated checkpoint provenance"
    )
    new_contract = DirectCompactContract.load(new_contract_path)
    codebook_sha = sha256_file(codebook_path)
    codec_sha = sha256_file(codec_path)
    if (
        codebook_sha != new_contract.codebook_sha256
        or codec_sha != new_contract.codec_sha256
    ):
        raise ValueError(
            "migration codebook/codec differs from the new compact contract"
        )
    expected_source = {
        "path": str(source["root"]),
        "decoder_adapter_sha256": sha256_artifact(source["adapter"]),
        "source_overlay_sha256": sha256_file(source["overlay"]),
        "contract_sha256": sha256_file(source["contract"]),
        "provenance_sha256": sha256_file(source["provenance"]),
    }
    expected_outputs = {
        "decoder_adapter_sha256": sha256_artifact(
            destination["adapter"]
        ),
        "source_overlay_sha256": sha256_file(destination["overlay"]),
        "compact_contract_sha256": sha256_file(
            destination["contract"]
        ),
        "codebook_sha256": codebook_sha,
        "codec_sha256": codec_sha,
    }
    if (
        receipt.get("schema")
        != "direct-compact-overlay-migration-receipt-v1"
        or receipt.get("training_steps") != 0
        or receipt.get("source_checkpoint") != expected_source
        or receipt.get("contract_compatibility") != compatibility
        or receipt.get("outputs") != expected_outputs
    ):
        raise ValueError("overlay migration receipt artifact bindings differ")
    expected_invariants = {
        "no_training_or_optimizer_step_performed": True,
        "decoder_adapter_tree_byte_identical": True,
        "old_overlay_row_reused_only_for_identical_expansion": True,
        "changed_rows_use_new_codebook_mean_initialization": True,
        "new_contract_copied_byte_identically": True,
        "heldout_data_opened": False,
    }
    if receipt.get("invariants") != expected_invariants:
        raise ValueError("overlay migration receipt invariants differ")
    if expected_outputs["decoder_adapter_sha256"] != expected_source[
        "decoder_adapter_sha256"
    ]:
        raise ValueError("overlay migration changed the decoder adapter")

    migration = receipt.get("overlay_migration")
    if not isinstance(migration, Mapping):
        raise ValueError("overlay migration receipt has no row audit")
    identical_ids = compatibility[
        "identical_expansion_source_token_ids"
    ]
    changed_ids = compatibility["changed_expansion_source_token_ids"]
    rows = migration.get("rows")
    if (
        migration.get("schema")
        != "source-token-overlay-expansion-migration-v1"
        or migration.get("reused_source_token_ids") != identical_ids
        or migration.get("reinitialized_source_token_ids") != changed_ids
        or not isinstance(rows, Mapping)
        or int(rows.get("total", -1))
        != compatibility["source_token_rows"]
        or int(rows.get("reused_identical_expansion", -1))
        != len(identical_ids)
        or int(rows.get("reinitialized_new_codebook_mean", -1))
        != len(changed_ids)
    ):
        raise ValueError("overlay migration receipt row accounting differs")

    old_state = torch.load(
        source["overlay"], map_location="cpu", weights_only=True
    )
    new_state = torch.load(
        destination["overlay"], map_location="cpu", weights_only=True
    )
    expected_ids = tuple(new_contract.source_token_ids)
    for label, state in (("source", old_state), ("migrated", new_state)):
        if (
            not isinstance(state, Mapping)
            or state.get("schema")
            != "source-token-embedding-overlay-v1"
            or int(state.get("base_vocab_size", -1))
            != int(new_contract.base_vocab_size or 0)
            or tuple(state.get("source_token_ids") or ()) != expected_ids
            or not isinstance(state.get("source_embeddings"), torch.Tensor)
        ):
            raise ValueError(f"{label} overlay state contract differs")
    old_weights = old_state["source_embeddings"]
    new_weights = new_state["source_embeddings"]
    if old_weights.shape != new_weights.shape:
        raise ValueError("source/migrated overlay tensor shapes differ")
    row_by_id = {
        source_id: index for index, source_id in enumerate(expected_ids)
    }
    if any(
        not torch.equal(
            old_weights[row_by_id[source_id]],
            new_weights[row_by_id[source_id]],
        )
        for source_id in identical_ids
    ):
        raise ValueError(
            "an identical-expansion learned overlay row was not preserved"
        )

    expected_provenance = {
        "checkpoint_stage": "contract-overlay-migration-only",
        "decoder_model": new_contract.decoder_model,
        "decoder_revision": new_contract.decoder_revision,
        "contract_sha256": expected_outputs["compact_contract_sha256"],
        "codebook_sha256": codebook_sha,
        "codec_sha256": codec_sha,
        "source_overlay_sha256": expected_outputs[
            "source_overlay_sha256"
        ],
        "decoder_adapter_sha256": expected_outputs[
            "decoder_adapter_sha256"
        ],
        "source_embedding_overlay_rows": len(expected_ids),
        "lm_head_rows": int(new_contract.base_vocab_size or 0),
        "training_performed": False,
        "heldout_loaded_during_migration": False,
        "overlay_migration_receipt_sha256": sha256_file(receipt_path),
        "warmstart_checkpoint": expected_source,
    }
    mismatches = [
        field
        for field, expected in expected_provenance.items()
        if provenance.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "migrated checkpoint provenance differs: "
            + ", ".join(mismatches)
        )
    return {
        "checkpoint": str(destination["root"]),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "reused_rows": len(identical_ids),
        "reinitialized_rows": len(changed_ids),
    }


class CompactJsonlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        tokenizer: Any,
        contract: DirectCompactContract,
    ) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, list[int]]] = []
        self.pool_use_count = 0
        eos_id = getattr(tokenizer, "eos_token_id", None)
        with self.path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = str(row.get("task_id") or row.get("id") or f"row-{index}")
                compact_ids = contract.validate_row(row, identity)
                pool = contract.validate_v3_pool_payload(
                    compact_ids, tokenizer, identity
                )
                if contract.schema == CONTRACT_SCHEMA_V3:
                    self.pool_use_count += len(pool["uses"])
                row_function = str(row.get("function") or "").strip()
                if row_function and row_function != contract.target_function:
                    raise ValueError(
                        f"{identity}: target function {row_function!r} does not match "
                        f"the contract {contract.target_function!r}"
                    )
                row_language = str(
                    row.get("language") or row.get("lang") or ""
                ).strip()
                if row_language and row_language.lower() != contract.target_language.lower():
                    raise ValueError(
                        f"{identity}: target language {row_language!r} does not match "
                        f"the contract {contract.target_language!r}"
                    )
                prompt_ids = _encode(
                    tokenizer,
                    direct_prompt(
                        row,
                        target_function=contract.target_function,
                        target_language=contract.target_language,
                    ),
                    special=True,
                )
                target_ids = _encode(
                    tokenizer, target_source(row, identity), special=False
                )
                if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                    target_ids.append(int(eos_id))
                if len(target_ids) > contract.max_target_tokens:
                    raise ValueError(
                        f"{identity}: target needs {len(target_ids)} tokens, exceeding "
                        f"the no-truncation limit {contract.max_target_tokens}"
                    )
                total = len(prompt_ids) + len(compact_ids) + len(target_ids)
                if total > contract.max_total_tokens:
                    raise ValueError(
                        f"{identity}: prompt+source+target needs {total} tokens, "
                        f"exceeding {contract.max_total_tokens}"
                    )
                self.rows.append(
                    {
                        "decoder_prompt_input_ids": prompt_ids,
                        "compact_input_ids": compact_ids,
                        "target_input_ids": target_ids,
                    }
                )
        if not self.rows:
            raise ValueError(f"{self.path}: no usable compact training rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def _context_limit(config: Any) -> int | None:
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="")
    parser.add_argument("--eval_file", default="")
    parser.add_argument("--train_seal", default="")
    parser.add_argument("--eval_seal", default="")
    parser.add_argument(
        "--no_eval_during_training",
        action="store_true",
        help=(
            "Do not load a measure split during fitting. Use this for "
            "predeclared post-Qwen stage training so held-out tasks cannot "
            "influence launch, checkpointing, or stopping."
        ),
    )
    parser.add_argument(
        "--stage_contract",
        default="",
        help=(
            "Optional immutable orchestration contract to bind into checkpoint "
            "provenance. Supply together with its expected SHA-256."
        ),
    )
    parser.add_argument("--expected_stage_contract_sha256", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        default="",
        help=(
            "Resume an interrupted trainer checkpoint inside --output_dir. "
            "Use 'auto' to select the highest checkpoint-* step. A started "
            "stage is never deleted or silently restarted."
        ),
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
        "--warmstart_checkpoint",
        default="",
        help=(
            "Continue from a complete direct-compact adapter+overlay checkpoint. "
            "Both components and the semantic compact contract are validated."
        ),
    )
    parser.add_argument(
        "--migrate_warmstart_only",
        action="store_true",
        help=(
            "Take no optimizer steps. Copy the warm-start adapter and migrate "
            "its compact overlay to --contract, reusing a learned row only "
            "when the old/new ordered source expansion is identical. Changed "
            "rows retain new-codebook-mean initialization."
        ),
    )
    parser.add_argument(
        "--validate_migrated_warmstart_only",
        action="store_true",
        help=(
            "Perform no model load or training. Validate an existing "
            "--output_dir overlay-migration checkpoint and its sealed receipt "
            "against --warmstart_checkpoint and the selected v2 artifacts."
        ),
    )
    parser.add_argument(
        "--sparse_topk_tail_manifest",
        default="",
        help=(
            "Optional sealed coarsened top-k+tail auxiliary manifest. "
            "Sequence target NLL remains the primary objective."
        ),
    )
    parser.add_argument(
        "--sparse_topk_tail_weight",
        type=float,
        default=0.0,
        help="Auxiliary coefficient in (0,1); zero disables the auxiliary.",
    )
    parser.add_argument(
        "--sparse_topk_tail_position_chunk_size",
        type=int,
        default=32,
        help="Maximum sparse positions materialized against the vocabulary at once.",
    )
    parser.add_argument(
        "--sequence_distribution_nll",
        action="store_true",
        help=(
            "Use equal-draw, EOS-inclusive summed sequence NLL. This is the "
            "primary Monte Carlo forward-KL objective for sampled Qwen "
            "sequences; ordinary SFT retains the default token-mean loss."
        ),
    )
    parser.add_argument(
        "--sequence_nll_position_chunk_size",
        type=int,
        default=512,
        help=(
            "Position chunk size for checkpointed FP32 sequence-NLL "
            "cross-entropy. Used only with --sequence_distribution_nll."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--eval_strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_contract_record: dict[str, Any] | None = None
    if bool(args.stage_contract) != bool(args.expected_stage_contract_sha256):
        raise ValueError(
            "stage contract path and expected SHA-256 must be supplied together"
        )
    if args.stage_contract:
        expected_stage_sha = args.expected_stage_contract_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_stage_sha):
            raise ValueError("expected stage-contract SHA-256 is invalid")
        stage_contract_path = Path(args.stage_contract).expanduser().resolve()
        if not stage_contract_path.is_file():
            raise FileNotFoundError(stage_contract_path)
        observed_stage_sha = sha256_file(stage_contract_path)
        if observed_stage_sha != expected_stage_sha:
            raise ValueError("stage contract differs from its expected SHA-256")
        stage_contract_record = {
            "path": str(stage_contract_path),
            "sha256": observed_stage_sha,
            "size_bytes": stage_contract_path.stat().st_size,
        }
    if args.migrate_warmstart_only and args.validate_migrated_warmstart_only:
        raise ValueError(
            "migration creation and validation-only modes are mutually exclusive"
        )
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if args.logging_steps <= 0:
        raise ValueError("logging_steps must be positive")
    sparse_enabled = bool(args.sparse_topk_tail_manifest.strip())
    if sparse_enabled != (args.sparse_topk_tail_weight > 0.0):
        raise ValueError(
            "sparse top-k+tail manifest and positive auxiliary weight must "
            "be supplied together"
        )
    if (
        not math.isfinite(args.sparse_topk_tail_weight)
        or args.sparse_topk_tail_weight < 0.0
        or args.sparse_topk_tail_weight >= 1.0
    ):
        raise ValueError(
            "sparse top-k+tail weight must lie in [0,1); sequence NLL is primary"
        )
    if args.sparse_topk_tail_position_chunk_size <= 0:
        raise ValueError("sparse top-k+tail position chunk size must be positive")
    contract = DirectCompactContract.load(args.contract)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.validate_migrated_warmstart_only:
        if not args.warmstart_checkpoint:
            raise ValueError(
                "--validate_migrated_warmstart_only requires "
                "--warmstart_checkpoint"
            )
        forbidden_validation_inputs = {
            "--train_file": args.train_file,
            "--train_seal": args.train_seal,
            "--eval_file": args.eval_file,
            "--eval_seal": args.eval_seal,
            "--sparse_topk_tail_manifest": args.sparse_topk_tail_manifest,
        }
        populated = [
            name
            for name, value in forbidden_validation_inputs.items()
            if str(value or "").strip()
        ]
        if populated or args.sequence_distribution_nll:
            raise ValueError(
                "migration validation-only mode may not receive training, "
                "evaluation, or teacher-distribution inputs"
            )
        result = validate_overlay_migrated_checkpoint(
            checkpoint=output_dir,
            source_checkpoint=args.warmstart_checkpoint,
            new_contract_path=args.contract,
            codebook_path=args.codebook,
            codec_path=args.codec_artifact,
        )
        print(
            "DIRECT_COMPACT_OVERLAY_MIGRATION_VALID "
            f"checkpoint={result['checkpoint']} "
            f"reused_rows={result['reused_rows']} "
            f"reinitialized_rows={result['reinitialized_rows']} "
            f"receipt_sha256={result['receipt_sha256']}",
            flush=True,
        )
        return

    # Decoder imports are lazy so contract/collator/migration validation unit
    # tests require only CPU PyTorch and never import Transformers.
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    resume_checkpoint: Path | None = None
    output_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if args.resume_from_checkpoint and (
        args.migrate_warmstart_only
        or args.validate_migrated_warmstart_only
    ):
        raise ValueError("overlay migration modes cannot resume optimizer state")
    if output_nonempty:
        if not args.resume_from_checkpoint:
            raise ValueError(
                f"output directory is non-empty: {output_dir}; pass "
                "--resume_from_checkpoint auto to resume an owned stage"
            )
        if args.resume_from_checkpoint == "auto":
            candidates = sorted(
                (
                    path
                    for path in output_dir.glob("checkpoint-*")
                    if path.is_dir()
                    and re.fullmatch(r"checkpoint-[0-9]+", path.name)
                ),
                key=lambda path: int(path.name.rsplit("-", 1)[1]),
            )
            if not candidates:
                raise ValueError(
                    "non-empty output has no resumable checkpoint-* directory"
                )
            resume_checkpoint = candidates[-1].resolve()
        else:
            resume_checkpoint = Path(
                args.resume_from_checkpoint
            ).expanduser().resolve()
        if (
            resume_checkpoint.parent != output_dir
            or not re.fullmatch(
                r"checkpoint-[0-9]+", resume_checkpoint.name
            )
        ):
            raise ValueError(
                "resume checkpoint must be an owned checkpoint-* directory "
                "directly under --output_dir"
            )
        for required_resume in (
            resume_checkpoint / "trainer_state.json",
            resume_checkpoint / "optimizer.pt",
            resume_checkpoint / "scheduler.pt",
            resume_checkpoint / "rng_state.pth",
            resume_checkpoint / "source_embedding_overlay.pt",
            resume_checkpoint / "compact_contract.json",
            resume_checkpoint / DIRECT_TRAINER_RESUME_FILENAME,
            resume_checkpoint / "decoder_adapter" / "adapter_config.json",
        ):
            if not required_resume.is_file():
                raise ValueError(
                    f"resume checkpoint is incomplete: {required_resume}"
                )
        if (
            sha256_file(resume_checkpoint / "compact_contract.json")
            != sha256_file(args.contract)
        ):
            raise ValueError("resume checkpoint compact contract changed")
    elif args.resume_from_checkpoint:
        raise ValueError(
            "--resume_from_checkpoint requires an existing non-empty output"
        )
    if args.save_steps <= 0:
        raise ValueError("--save_steps must be positive")
    if args.migrate_warmstart_only:
        if not args.warmstart_checkpoint:
            raise ValueError(
                "--migrate_warmstart_only requires --warmstart_checkpoint"
            )
        forbidden_migration_inputs = {
            "--train_file": args.train_file,
            "--train_seal": args.train_seal,
            "--eval_file": args.eval_file,
            "--eval_seal": args.eval_seal,
            "--sparse_topk_tail_manifest": args.sparse_topk_tail_manifest,
        }
        populated = [
            name
            for name, value in forbidden_migration_inputs.items()
            if str(value or "").strip()
        ]
        if populated:
            raise ValueError(
                "overlay-only migration may not open training/evaluation or "
                "teacher-distribution artifacts: "
                + ", ".join(populated)
            )
        if args.sequence_distribution_nll:
            raise ValueError(
                "overlay-only migration may not select a training loss"
            )
        if output_dir.exists():
            raise ValueError(
                "overlay-only migration requires an absent output directory"
            )
        warmstart = validate_self_sealed_checkpoint(
            args.warmstart_checkpoint
        )
        validate_overlay_migration_contracts(
            warmstart["contract"], args.contract
        )
    elif args.warmstart_checkpoint:
        warmstart = validate_warmstart_checkpoint(
            args.warmstart_checkpoint,
            contract_path=args.contract,
        )
        if output_dir == warmstart["root"]:
            raise ValueError("output directory must differ from warm-start checkpoint")
    else:
        warmstart = None
    if not args.migrate_warmstart_only and (
        not args.train_file or not args.train_seal
    ):
        raise ValueError(
            "--train_file and --train_seal are required for training"
        )
    if args.migrate_warmstart_only:
        train_seal = None
        eval_seal = None
    else:
        train_seal = validate_join_seal(
            args.train_file,
            args.train_seal,
            args.contract,
            expected_role="fit",
        )
    if args.migrate_warmstart_only:
        pass
    elif args.no_eval_during_training:
        if args.eval_file or args.eval_seal:
            raise ValueError(
                "--no_eval_during_training forbids --eval_file/--eval_seal"
            )
        if args.eval_strategy != "no":
            raise ValueError(
                "--no_eval_during_training requires --eval_strategy no"
            )
        eval_seal = None
    else:
        if not args.eval_file or not args.eval_seal:
            raise ValueError(
                "--eval_file and --eval_seal are required unless "
                "--no_eval_during_training is set"
            )
        eval_seal = validate_join_seal(
            args.eval_file,
            args.eval_seal,
            args.contract,
            expected_role="measure",
        )
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
        if tokenizer.eos_token_id is None:
            raise ValueError("direct causal training requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    contract.validate_artifacts(
        tokenizer=tokenizer,
        tokenizer_json_path=args.tokenizer_json,
        codec_path=args.codec_artifact,
        codebook_path=args.codebook,
    )
    sparse_manifest = None
    sparse_helpers = None
    if sparse_enabled:
        from scripts.training import direct_compact_sparse_topk_tail as sparse_helpers

        if tokenizer.eos_token_id is None:
            raise ValueError("sparse top-k+tail training requires student EOS")
        sparse_manifest = sparse_helpers.validate_sparse_manifest(
            args.train_file,
            args.sparse_topk_tail_manifest,
            contract_path=args.contract,
            tokenizer_json_path=args.tokenizer_json,
        )
        if int(sparse_manifest.get("student_output_vocab_size", -1)) != int(
            contract.base_vocab_size or 0
        ):
            raise ValueError(
                "sparse auxiliary output vocabulary differs from compact contract"
            )

    warmstart_binding = (
        None
        if warmstart is None
        else {
            "decoder_adapter_sha256": sha256_artifact(warmstart["adapter"]),
            "source_overlay_sha256": sha256_file(warmstart["overlay"]),
            "contract_sha256": sha256_file(warmstart["contract"]),
            "provenance_sha256": sha256_file(warmstart["provenance"]),
        }
    )
    resume_compatibility: dict[str, Any] = {
        "schema": "direct-compact-trainer-launch-compatibility-v1",
        "implementation": {
            "trainer_sha256": sha256_file(Path(__file__).resolve()),
            "direct_compact_causal_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "models"
                / "direct_compact_causal.py"
            ),
            "sparse_topk_tail_sha256": (
                None
                if sparse_helpers is None
                else sha256_file(Path(sparse_helpers.__file__).resolve())
            ),
            "torch_version": str(torch.__version__),
            "transformers_version": importlib.metadata.version("transformers"),
            "peft_version": importlib.metadata.version("peft"),
            "accelerate_version": importlib.metadata.version("accelerate"),
        },
        "decoder": {
            "model": decoder_model,
            "revision": decoder_revision,
            "model_config_sha256": sha256_file(decoder_config_path),
            "attn_implementation": args.attn_implementation,
            "tokenizer": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
        },
        "compact_artifacts": {
            "contract_sha256": sha256_file(args.contract),
            "codebook_sha256": sha256_file(args.codebook),
            "codec_sha256": sha256_file(args.codec_artifact),
            "tokenizer_json_sha256": sha256_file(args.tokenizer_json),
        },
        "datasets": {
            "train_file_sha256": sha256_file(args.train_file),
            "train_seal_sha256": sha256_file(args.train_seal),
            "train_sealed_rows": int(train_seal["rows"]),
            "eval_file_sha256": (
                None
                if args.no_eval_during_training
                else sha256_file(args.eval_file)
            ),
            "eval_seal_sha256": (
                None
                if args.no_eval_during_training
                else sha256_file(args.eval_seal)
            ),
            "eval_sealed_rows": (
                None if eval_seal is None else int(eval_seal["rows"])
            ),
            "no_eval_during_training": bool(args.no_eval_during_training),
            "stage_contract": (
                None
                if stage_contract_record is None
                else {
                    "sha256": stage_contract_record["sha256"],
                    "size_bytes": int(stage_contract_record["size_bytes"]),
                }
            ),
            "sparse_topk_tail_manifest_sha256": (
                None
                if not sparse_enabled
                else sha256_file(args.sparse_topk_tail_manifest)
            ),
        },
        "loss": {
            "sequence_distribution_nll": bool(
                args.sequence_distribution_nll
            ),
            "sequence_nll_position_chunk_size": int(
                args.sequence_nll_position_chunk_size
            ),
            "sparse_topk_tail_weight": float(
                args.sparse_topk_tail_weight
            ),
            "sparse_topk_tail_position_chunk_size": int(
                args.sparse_topk_tail_position_chunk_size
            ),
        },
        "optimization": {
            "learning_rate": float(args.learning_rate),
            "epochs": float(args.epochs),
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "seed": int(args.seed),
            "eval_strategy": args.eval_strategy,
            "logging_steps": int(args.logging_steps),
            "save_steps": int(args.save_steps),
            "save_total_limit": 2,
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "load_4bit": bool(args.load_4bit),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
        },
        "initialization": {
            "warmstart_checkpoint": warmstart_binding,
        },
    }
    resume_state = (
        None
        if resume_checkpoint is None
        else validate_direct_trainer_resume_checkpoint(
            resume_checkpoint,
            expected_compatibility=resume_compatibility,
        )
    )
    # save_total_limit may rotate the checkpoint we resumed from after newer
    # checkpoints are committed. Capture its already-validated identity now.
    resume_source_binding = (
        None
        if resume_state is None
        else {
            "path": str(resume_state["root"]),
            "global_step": _checkpoint_step(resume_state["root"]),
            "resume_metadata_sha256": sha256_file(
                resume_state["metadata"]
            ),
            "decoder_adapter_sha256": sha256_artifact(
                resume_state["adapter"]
            ),
            "source_overlay_sha256": sha256_file(
                resume_state["overlay"]
            ),
        }
    )

    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.no_lora:
        raise ValueError(
            "--no_lora is not supported by the adapter+overlay checkpoint contract"
        )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.bf16 and not args.load_4bit:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16 and not args.load_4bit:
        model_kwargs["torch_dtype"] = torch.float16
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        decoder_model,
        revision=decoder_revision,
        **model_kwargs,
    )
    context_limit = _context_limit(model.config)
    if context_limit is None:
        raise ValueError("decoder config does not expose a verifiable context limit")
    if contract.max_total_tokens > context_limit:
        raise ValueError(
            f"contract permits {contract.max_total_tokens} tokens but decoder context "
            f"is only {context_limit}"
        )
    validate_base_model_vocab(model, contract)
    if args.migrate_warmstart_only:
        result = materialize_overlay_migrated_checkpoint(
            model=model,
            source_checkpoint=args.warmstart_checkpoint,
            new_contract_path=args.contract,
            codebook_path=args.codebook,
            codec_path=args.codec_artifact,
            output_dir=output_dir,
            stage_contract_record=stage_contract_record,
        )
        print(
            "DIRECT_COMPACT_OVERLAY_MIGRATION_COMPLETE "
            f"checkpoint={result['checkpoint']} "
            f"reused_rows={result['reused_rows']} "
            f"reinitialized_rows={result['reinitialized_rows']} "
            f"receipt_sha256={result['receipt_sha256']}",
            flush=True,
        )
        return

    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if args.load_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    adapter_state = resume_state if resume_state is not None else warmstart
    if adapter_state is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_state["adapter"]),
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            model,
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

    expansions = dict(contract.source_token_expansions)
    if adapter_state is not None:
        overlay = restore_source_embedding_overlay(
            model,
            expansions,
            adapter_state["overlay"],
            base_vocab_size=int(contract.base_vocab_size or 0),
        )
    else:
        overlay = install_source_embedding_overlay(
            model,
            expansions,
            base_vocab_size=int(contract.base_vocab_size or 0),
        )
    if model.get_output_embeddings().weight.size(0) != contract.base_vocab_size:
        raise RuntimeError("source-token setup unexpectedly resized the LM head")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_dataset = CompactJsonlDataset(
        args.train_file, tokenizer=tokenizer, contract=contract
    )
    eval_dataset = (
        None
        if args.no_eval_during_training
        else CompactJsonlDataset(
            args.eval_file, tokenizer=tokenizer, contract=contract
        )
    )
    sparse_attachment = None
    if sparse_enabled:
        assert sparse_helpers is not None and sparse_manifest is not None
        sparse_attachment = sparse_helpers.attach_sparse_metadata(
            train_dataset,
            args.train_file,
            tokenizer=tokenizer,
            eos_token_id=int(tokenizer.eos_token_id),
            output_vocab_size=int(contract.base_vocab_size or 0),
            expected_rows_with_auxiliary=int(
                sparse_manifest["rows_with_sparse_auxiliary"]
            ),
            expected_sparse_positions=int(sparse_manifest["sparse_positions"]),
        )
    if contract.schema == CONTRACT_SCHEMA_V3:
        split_datasets = [("train", train_dataset, train_seal)]
        if eval_dataset is not None and eval_seal is not None:
            split_datasets.append(("eval", eval_dataset, eval_seal))
        for split, dataset, seal in split_datasets:
            expected_uses = seal["pool_metadata"]["total_use_count"]
            if dataset.pool_use_count != expected_uses:
                raise ValueError(
                    f"{split}: decoded pool-use count {dataset.pool_use_count} "
                    f"does not match sealed count {expected_uses}"
                )
    collator = DirectCompactBatchCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_tokens=contract.max_source_tokens,
        max_target_tokens=contract.max_target_tokens,
        max_total_tokens=contract.max_total_tokens,
        pad_to_multiple_of=8,
        source_token_ids=contract.source_token_ids,
    )
    if sparse_enabled:
        assert sparse_helpers is not None
        collator = sparse_helpers.SparseTopKTailCollator(collator)
        wrapped_model = sparse_helpers.DirectCompactSparseTopKTailCausalLM(
            model,
            auxiliary_weight=args.sparse_topk_tail_weight,
            position_chunk_size=args.sparse_topk_tail_position_chunk_size,
            sequence_sum_nll=args.sequence_distribution_nll,
        )
    else:
        wrapped_model = DirectCompactCausalLM(
            model,
            sequence_sum_nll=args.sequence_distribution_nll,
            sequence_nll_position_chunk_size=(
                args.sequence_nll_position_chunk_size
            ),
        )
    DirectTrainer = make_direct_trainer_class(
        Trainer,
        adapter_model=model,
        overlay=overlay,
        tokenizer=tokenizer,
        contract_path=args.contract,
        authorized_resume_checkpoint=resume_checkpoint,
        resume_compatibility=resume_compatibility,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy=args.eval_strategy,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = DirectTrainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    if (
        resume_checkpoint is not None
        and not trainer._direct_resume_model_state_verified
    ):
        raise RuntimeError(
            "Trainer did not execute the sealed direct-checkpoint load hook"
        )
    trainer.save_model(str(output_dir))
    overlay_path = output_dir / "source_embedding_overlay.pt"
    adapter_path = output_dir / "decoder_adapter"
    provenance = {
        "schema": "direct-compact-run-provenance-v1",
        "architecture": "qwen-causal-compact-tokens-no-encoder",
        "decoder_model": decoder_model,
        "decoder_revision": decoder_revision,
        "model_config_sha256": sha256_file(decoder_config_path),
        "attn_implementation": args.attn_implementation,
        "loss_contract": {
            "sequence_distribution_nll": bool(
                args.sequence_distribution_nll
            ),
            "sequence_target_suffix_logits_only": bool(
                args.sequence_distribution_nll
            ),
            "sequence_nll_position_chunk_size": (
                int(args.sequence_nll_position_chunk_size)
                if args.sequence_distribution_nll
                else None
            ),
            "primary_reduction": (
                "equal_weight_mean_of_eos_inclusive_per_sequence_nll_sums"
                if args.sequence_distribution_nll
                else "base_causal_lm_token_mean"
            ),
        },
        "contract_sha256": sha256_file(args.contract),
        "codebook_sha256": sha256_file(args.codebook),
        "codec_sha256": sha256_file(args.codec_artifact),
        "train_file_sha256": sha256_file(args.train_file),
        "eval_file_sha256": (
            None if args.no_eval_during_training else sha256_file(args.eval_file)
        ),
        "train_seal_sha256": sha256_file(args.train_seal),
        "eval_seal_sha256": (
            None if args.no_eval_during_training else sha256_file(args.eval_seal)
        ),
        "train_sealed_rows": int(train_seal["rows"]),
        "eval_sealed_rows": (
            None if eval_seal is None else int(eval_seal["rows"])
        ),
        "heldout_loaded_during_training": not args.no_eval_during_training,
        "stage_contract": stage_contract_record,
        "source_overlay_sha256": sha256_file(overlay_path),
        "decoder_adapter_sha256": sha256_artifact(adapter_path),
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "eval_strategy": args.eval_strategy,
        "training_schedule": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "load_4bit": bool(args.load_4bit),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
        },
        "source_embedding_overlay_rows": len(contract.source_token_ids),
        "lm_head_rows": int(model.get_output_embeddings().weight.size(0)),
        "graph_encoder": None,
        "soft_prefix": None,
        "resumed_from_trainer_checkpoint": resume_source_binding,
        "trainer_final_global_step": int(trainer.state.global_step),
        "trainer_resume_compatibility_sha256": _canonical_mapping_sha256(
            resume_compatibility
        ),
        "warmstart_checkpoint": (
            None
            if warmstart is None
            else {
                "path": str(warmstart["root"]),
                **warmstart_binding,
            }
        ),
        "sparse_topk_tail_auxiliary": (
            None
            if not sparse_enabled
            else {
                "objective": "coarsened_topk_plus_tail_forward_kl",
                "data_processing_lower_bound_only": True,
                "dense_full_vocabulary_kl": False,
                "full_vocabulary_kd": False,
                "sequence_monte_carlo_forward_kl_nll_primary": True,
                "weight": args.sparse_topk_tail_weight,
                "position_chunk_size": args.sparse_topk_tail_position_chunk_size,
                "manifest_path": str(
                    Path(args.sparse_topk_tail_manifest).expanduser().resolve()
                ),
                "manifest_sha256": sha256_file(
                    args.sparse_topk_tail_manifest
                ),
                "rows_with_sparse_auxiliary": sparse_attachment[
                    "rows_with_sparse_auxiliary"
                ],
                "sparse_positions": sparse_attachment["sparse_positions"],
                "teacher_eos_distribution_available": False,
                "sparse_auxiliary_applied_to_eos": False,
                "student_eos_supervised_by_primary_sequence_nll": True,
            }
        ),
    }
    (output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
