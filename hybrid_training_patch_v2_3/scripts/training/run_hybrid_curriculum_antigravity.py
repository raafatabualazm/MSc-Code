#!/usr/bin/env python3
"""Run the fail-closed all-length hybrid decompiler curriculum (v2.3).

The ordering is deliberate:

1. CPU-only leakage, deduplication, neutral-contract, hidden-test, and reference
   parity gates run before any GPU training.
2. Every approved length stratum enters direct whole-function SFT.
3. Bridge and >=200-instruction rows receive deterministic CFG-region planning
   and plan-conditioned reconstruction supervision.
4. A code-only recovery stage returns the policy to the deployed task while
   oversampling the long-function strata.
5. Free-running graph permutation gates, frontier repair, hidden verification,
   matched-control RS-SFT, and optional VeRPO remain fail closed.

Only a deterministic, length-stratified development slice is withheld. Long
functions are no longer routed out of supervised training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 6


# Only architecture/representation settings are imported from checkpoint
# provenance. Runtime, dataset, output, HF-upload, and objective settings are
# deliberately excluded so replaying provenance cannot trigger side effects.
ARCHITECTURE_ENV_KEYS = frozenset({
    "GRAPH_ENCODER_MODEL", "GRAPH_DECODER_MODEL",
    "GRAPH_ENCODER_REVISION", "GRAPH_DECODER_REVISION",
    "GRAPH_ENCODER_PEFT", "GRAPH_DECODER_PEFT",
    "GRAPH_FREEZE_ENCODER", "GRAPH_FREEZE_DECODER",
    "GRAPH_LORA_R", "GRAPH_LORA_ALPHA", "GRAPH_QWEN_LORA_TARGETS",
    "GRAPH_ATTN_IMPLEMENTATION",
    "GRAPH_DFG_MODE", "GRAPH_EDGE_ABLATION", "GRAPH_ADD_REVERSE_EDGES",
    "GRAPH_MAX_DATAFLOW_EDGES", "GRAPH_GNN_ABLATION", "GRAPH_GNN_LAYERS",
    "GRAPH_GLOBAL_ATTENTION_ABLATION", "GRAPH_POSITION_SCHEME",
    "GRAPH_CAUSAL_POSITION_IDS", "GRAPH_AUTO_CFG", "GRAPH_MAX_BLOCK_INSTRS",
    "GRAPH_BLOCK_POOLING", "GRAPH_BLOCK_VECTORS_PER_BLOCK",
    "GRAPH_BLOCK_POSITION_MODE", "GRAPH_QWEN_PREFIX_TOKENS",
    "GRAPH_QWEN_PREFIX_HEADS", "GRAPH_QWEN_PREFIX_DYNAMIC",
    "GRAPH_QWEN_PREFIX_MIN_TOKENS", "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2",
    "GRAPH_QWEN_PREFIX_GATE_MODE", "GRAPH_QWEN_PREFIX_GATE_INIT",
    "GRAPH_QWEN_PREFIX_RMS_MATCH", "GRAPH_QWEN_PREFIX_DROPOUT",
    "GRAPH_REGION_COMPRESSION", "GRAPH_REGION_MAX_BLOCKS",
    "GRAPH_STRICT_GRAPH", "GRAPH_DECODER_PROMPT_MAX_LENGTH",
    "GRAPH_PROMPT_CLEAN_ASM",
    "GRAPH_PROMPT_FIT_ASSEMBLY", "GRAPH_USE_REASONING",
})

ESSENTIAL_WARM_START_KEYS = frozenset({
    # Model/adapter construction.
    "GRAPH_ENCODER_MODEL", "GRAPH_DECODER_MODEL",
    "GRAPH_ENCODER_REVISION", "GRAPH_DECODER_REVISION",
    "GRAPH_ENCODER_PEFT", "GRAPH_DECODER_PEFT",
    "GRAPH_FREEZE_ENCODER", "GRAPH_FREEZE_DECODER",
    "GRAPH_LORA_R", "GRAPH_LORA_ALPHA",
    # Graph construction and forward semantics. Several ablations deliberately
    # retain the same parameter shapes, so state-dict compatibility alone cannot
    # detect their drift.
    "GRAPH_DFG_MODE", "GRAPH_EDGE_ABLATION", "GRAPH_ADD_REVERSE_EDGES",
    "GRAPH_MAX_DATAFLOW_EDGES", "GRAPH_GNN_ABLATION", "GRAPH_GNN_LAYERS",
    "GRAPH_GLOBAL_ATTENTION_ABLATION", "GRAPH_POSITION_SCHEME",
    "GRAPH_CAUSAL_POSITION_IDS", "GRAPH_AUTO_CFG", "GRAPH_MAX_BLOCK_INSTRS",
    "GRAPH_BLOCK_POOLING", "GRAPH_BLOCK_VECTORS_PER_BLOCK",
    "GRAPH_BLOCK_POSITION_MODE", "GRAPH_REGION_COMPRESSION",
    "GRAPH_REGION_MAX_BLOCKS", "GRAPH_STRICT_GRAPH",
    # Continuous-prefix construction and active-slot semantics.
    "GRAPH_QWEN_PREFIX_TOKENS", "GRAPH_QWEN_PREFIX_DYNAMIC",
    "GRAPH_QWEN_PREFIX_MIN_TOKENS", "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2",
    "GRAPH_QWEN_PREFIX_GATE_MODE", "GRAPH_QWEN_PREFIX_RMS_MATCH",
    # Decoder input contract inherited unless a stage explicitly changes a
    # modality flag such as GRAPH_PROMPT_ASSEMBLY_MODE.
    "GRAPH_DECODER_PROMPT_MAX_LENGTH", "GRAPH_PROMPT_CLEAN_ASM",
    "GRAPH_PROMPT_FIT_ASSEMBLY", "GRAPH_USE_REASONING",
})

# Historical July-2026 checkpoints recorded only explicitly exported graph
# settings. For these exact trainer source revisions, an absent key means the
# listed source-code default. Never apply this migration to an unknown or
# changed trainer: structural defaults are part of the checkpoint contract.
_LEGACY_ARCH_DEFAULTS = {
    "GRAPH_AUTO_CFG": "0",
    "GRAPH_CAUSAL_POSITION_IDS": "cumsum",
    "GRAPH_EDGE_ABLATION": "full",
    "GRAPH_ENCODER_PEFT": "none",
    "GRAPH_FREEZE_DECODER": "0",
    "GRAPH_FREEZE_ENCODER": "1",
    "GRAPH_GLOBAL_ATTENTION_ABLATION": "full",
    "GRAPH_GNN_ABLATION": "full",
    "GRAPH_GNN_LAYERS": "4",
    "GRAPH_MAX_BLOCK_INSTRS": "0",
    "GRAPH_MAX_DATAFLOW_EDGES": "4096",
    "GRAPH_QWEN_PREFIX_GATE_MODE": "scalar",
    "GRAPH_STRICT_GRAPH": "0",
    "GRAPH_USE_REASONING": "0",
}
_LEGACY_ARCH_DEFAULT_SOURCE_HASHES = frozenset({
    # Remote trainer used by text_arm_v2_s44 02t/08b.
    "cd6325012dcdb76e22cc027040a23c1f94afa6e5490c27aafcbbef062d56a3b4",
    # Packaged trainer with the same defaults, used by local regression tests.
    "409a697751ed56532e102c130485f0620b0655484773c60b4805988188c3b418",
})


def _checkpoint_provenance_candidates(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.architecture_env_json:
        candidates.append(Path(args.architecture_env_json).expanduser().resolve())
    for raw in (
        args.probe_checkpoint,
        args.initial_checkpoint,
        args.stage1_checkpoint,
        getattr(args, "text_sft_checkpoint", ""),
        getattr(args, "verpo_checkpoint", ""),
    ):
        if not raw:
            continue
        checkpoint = Path(raw).expanduser().resolve()
        sibling = checkpoint.parent / "run_provenance.json"
        if sibling not in candidates:
            candidates.append(sibling)
    return candidates


def load_architecture_environment(
    args: argparse.Namespace,
) -> tuple[dict[str, str], Path | None]:
    """Load the structural ``graph_environment`` bound to a warm checkpoint."""
    explicit = bool(args.architecture_env_json)
    for candidate in _checkpoint_provenance_candidates(args):
        if not candidate.is_file():
            if explicit and candidate == Path(args.architecture_env_json).expanduser().resolve():
                raise FileNotFoundError(candidate)
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        environment = payload.get("graph_environment")
        if not isinstance(environment, dict):
            nested = payload.get("provenance")
            environment = nested.get("graph_environment") if isinstance(nested, dict) else None
        if not isinstance(environment, dict):
            if explicit:
                raise ValueError(
                    f"{candidate} does not contain a graph_environment object"
                )
            continue
        environment = _complete_known_legacy_architecture(
            payload=payload,
            environment={
                str(key): str(value)
                for key, value in environment.items()
                if isinstance(key, str) and value is not None
            },
            args=args,
            provenance_path=candidate,
        )
        filtered = {
            key: str(value)
            for key, value in environment.items()
            if key in ARCHITECTURE_ENV_KEYS and value is not None
        }
        if not filtered:
            raise ValueError(f"{candidate} contains no recognised architecture settings")
        # Before v2.1 Qwen adapters were unconditionally attention-only and the
        # target set was not recorded. This legacy value is deterministic, not
        # guessed; Stage 1 may then perform an explicit zero-output MLP expansion.
        if (
            "GRAPH_QWEN_LORA_TARGETS" not in filtered
            and "qwen" in filtered.get("GRAPH_DECODER_MODEL", "").lower()
            and filtered.get("GRAPH_DECODER_PEFT", "none").lower() in {"lora", "dora"}
        ):
            filtered["GRAPH_QWEN_LORA_TARGETS"] = "attention"
            print(
                f"Architecture provenance {candidate} predates Qwen LoRA target tracking; "
                "binding it to the historical attention-only target set."
            )
        return filtered, candidate
    return {}, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_known_legacy_architecture(
    *,
    payload: dict[str, Any],
    environment: dict[str, str],
    args: argparse.Namespace,
    provenance_path: Path,
) -> dict[str, str]:
    missing = ESSENTIAL_WARM_START_KEYS - set(environment)
    fillable = missing & set(_LEGACY_ARCH_DEFAULTS)
    if not fillable:
        return environment

    source_files = payload.get("source_files")
    if not isinstance(source_files, list):
        nested = payload.get("provenance")
        source_files = (
            nested.get("source_files") if isinstance(nested, dict) else None
        )
    trainer_records = [
        record
        for record in (source_files or [])
        if isinstance(record, dict)
        and str(record.get("path", "")).replace("\\", "/").endswith(
            "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py"
        )
    ]
    if len(trainer_records) != 1:
        return environment
    recorded_sha = str(trainer_records[0].get("sha256", "")).lower()
    current_trainer = (
        Path(args.project_root).expanduser().resolve()
        / "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py"
    )
    if (
        recorded_sha not in _LEGACY_ARCH_DEFAULT_SOURCE_HASHES
        or not current_trainer.is_file()
        or sha256_file(current_trainer) != recorded_sha
    ):
        return environment

    completed = dict(environment)
    for key in sorted(fillable):
        completed[key] = _LEGACY_ARCH_DEFAULTS[key]
    print(
        f"Architecture provenance {provenance_path} predates explicit default "
        f"tracking; recovered {len(fillable)} keys from exact trainer source "
        f"{recorded_sha}.",
        file=sys.stderr,
    )
    return completed


def file_record(path: str | Path, required: bool = True) -> dict[str, Any] | None:
    value = Path(path).expanduser().resolve()
    if not value.exists():
        if required:
            raise FileNotFoundError(value)
        return None
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256_file(value),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def effective_generation_budgets(args: argparse.Namespace) -> dict[str, int]:
    """Resolve every code-generation ceiling used by the curriculum."""
    long_floor = int(args.long_generation_max_tokens)
    return {
        "gate": max(int(args.gate_max_new_tokens), long_floor),
        "rollout": max(int(args.rollout_max_new_tokens), long_floor),
        "grpo": max(int(args.grpo_max_new_tokens), long_floor),
    }


def validate_target_generation_budget(args: argparse.Namespace) -> dict[str, int]:
    budgets = effective_generation_budgets(args)
    target = int(args.long_target_max_tokens)
    minimum = min(budgets.values())
    if target > minimum:
        raise ValueError(
            "long_target_max_tokens cannot exceed any effective generation budget; "
            f"target={target}, effective={budgets}. Increase long_generation_max_tokens "
            "or lower the admitted target ceiling."
        )
    return budgets


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def env_snapshot(env: dict[str, str]) -> dict[str, str]:
    prefixes = (
        "GRAPH_",
        "GRPO_",
        "VERPO_",
        "HF_",
        "CUDA_",
        "PYTORCH_",
        "HYBRID_",
    )
    secret_fragments = ("TOKEN", "KEY", "SECRET", "PASSWORD")
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(prefixes)
        and not any(fragment in key.upper() for fragment in secret_fragments)
    }


def _split_existing_paths(value: str) -> list[Path]:
    paths: list[Path] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            paths.append(candidate.resolve())
    return paths


def input_fingerprints(command: list[str], env: dict[str, str]) -> list[dict[str, Any]]:
    paths: dict[str, Path] = {}
    for value in command[1:]:
        for path in _split_existing_paths(str(value)):
            paths[str(path)] = path
    for key, value in env.items():
        if key.endswith(("_FILE", "_CHECKPOINT")) or key in {
            "GRPO_TRAIN_FILE",
            "GRAPH_TRAIN_FILE",
            "GRAPH_EVAL_FILE",
            "GRPO_VERIFIED_ANCHOR_FILE",
        }:
            for path in _split_existing_paths(value):
                paths[str(path)] = path
    return [file_record(path) for path in sorted(paths.values(), key=str)]


def stage_signature(command: list[str], env: dict[str, str], inputs: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "environment": env_snapshot(env),
        "inputs": inputs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "pytorch_model.bin"


def validate_verpo_repair_dataset(path: str | Path) -> dict[str, Any]:
    """Validate the V2 prompt contract without exposing hidden tests."""
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    rows = 0
    task_ids: set[str] = set()
    with value.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [
                field
                for field in ("prior_attempt", "repair_feedback", "feedback_tests")
                if not str(row.get(field) or "").strip()
            ]
            if missing or not bool(row.get("verpo_repair")):
                raise ValueError(
                    f"invalid VeRPO repair row {line_number}: missing "
                    + ", ".join(missing or ["verpo_repair=true"])
                )
            leaked = [
                field
                for field in (
                    "tests",
                    "acceptance_tests",
                    "hidden_tests",
                    "private_tests",
                )
                if str(row.get(field) or "").strip()
            ]
            if leaked:
                raise ValueError(
                    f"invalid VeRPO repair row {line_number}: hidden/legacy "
                    f"harness fields present: {', '.join(leaked)}"
                )
            identity = row.get("task_id", row.get("id", row.get("source_line")))
            if identity is not None:
                task_ids.add(str(identity))
            rows += 1
    if rows == 0:
        raise ValueError(f"VeRPO repair dataset is empty: {value}")
    return {
        "path": str(value),
        "rows": rows,
        "unique_tasks": len(task_ids),
        "sha256": sha256_file(value),
    }


def mark_rows_evaluation_only(path: Path) -> None:
    """Flag every row's ``hybrid_metadata.evaluation_only=True`` in a JSONL file.

    The functional gates fail closed unless each evaluation row carries this
    flag. The long dev subset is produced by ``build_hierarchical_long_training``
    (which predates the flag) and is held out from supervised training, so
    marking it is accurate. Idempotent; safe to run whether 00g ran or reused.
    """
    if not path.is_file():
        return
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("hybrid_metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata["evaluation_only"] = True
            row["hybrid_metadata"] = metadata
            rows.append(row)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CurriculumRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.project_root).expanduser().resolve()
        self.output = Path(args.output_root).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.python = args.python or sys.executable
        self.base_env = os.environ.copy()
        architecture_env, architecture_source = load_architecture_environment(args)
        self.architecture_env = architecture_env
        self.architecture_env_source = architecture_source
        self.base_env.update(architecture_env)
        self.warm_qwen_lora_targets = architecture_env.get(
            "GRAPH_QWEN_LORA_TARGETS", args.qwen_lora_targets
        )
        self.target_qwen_lora_targets = args.qwen_lora_targets

        # CLI values are explicit overrides; otherwise resolve from checkpoint
        # provenance, then the current environment, then stable model defaults.
        args.decoder_model = (
            args.decoder_model
            or architecture_env.get("GRAPH_DECODER_MODEL")
            or os.environ.get("GRAPH_DECODER_MODEL")
            or "Qwen/Qwen3-8B"
        )
        args.encoder_model = (
            args.encoder_model
            or architecture_env.get("GRAPH_ENCODER_MODEL")
            or os.environ.get("GRAPH_ENCODER_MODEL")
            or "microsoft/graphcodebert-base"
        )
        args.decoder_revision = (
            args.decoder_revision
            or architecture_env.get("GRAPH_DECODER_REVISION")
            or os.environ.get("GRAPH_DECODER_REVISION", "")
        )
        args.encoder_revision = (
            args.encoder_revision
            or architecture_env.get("GRAPH_ENCODER_REVISION")
            or os.environ.get("GRAPH_ENCODER_REVISION", "")
        )
        if args.decoder_prompt_max_length <= 0:
            args.decoder_prompt_max_length = int(
                architecture_env.get(
                    "GRAPH_DECODER_PROMPT_MAX_LENGTH",
                    os.environ.get("GRAPH_DECODER_PROMPT_MAX_LENGTH", "768"),
                )
            )

        warm_start_requested = any(
            value
            for value in (
                args.probe_checkpoint,
                args.initial_checkpoint,
                args.stage1_checkpoint,
                args.text_sft_checkpoint,
                args.verpo_checkpoint,
            )
        )
        if warm_start_requested and not args.dry_run and not args.allow_unpinned_architecture:
            if architecture_source is None:
                raise SystemExit(
                    "A warm checkpoint was supplied without structural provenance. "
                    "Pass --architecture_env_json pointing to its run_provenance.json "
                    "or prediction provenance; --allow_unpinned_architecture is a "
                    "research-only escape hatch."
                )
            missing_contract = sorted(ESSENTIAL_WARM_START_KEYS - set(architecture_env))
            decoder_is_qwen = "qwen" in str(args.decoder_model).lower()
            decoder_peft = str(
                architecture_env.get(
                    "GRAPH_DECODER_PEFT",
                    self.base_env.get("GRAPH_DECODER_PEFT", "none"),
                )
            ).strip().lower()
            if (
                decoder_is_qwen
                and decoder_peft in {"lora", "dora"}
                and "GRAPH_QWEN_LORA_TARGETS" not in architecture_env
            ):
                missing_contract.append("GRAPH_QWEN_LORA_TARGETS")
            if missing_contract:
                raise SystemExit(
                    f"architecture provenance {architecture_source} is incomplete; missing: "
                    + ", ".join(sorted(set(missing_contract)))
                )

        qwen_lora_contract_active = (
            "qwen" in str(args.decoder_model).lower()
            and str(
                architecture_env.get(
                    "GRAPH_DECODER_PEFT",
                    self.base_env.get("GRAPH_DECODER_PEFT", "none"),
                )
            ).strip().lower() in {"lora", "dora"}
        )
        if (
            qwen_lora_contract_active
            and self.warm_qwen_lora_targets != self.target_qwen_lora_targets
        ):
            supported_expansion = (
                self.warm_qwen_lora_targets == "attention"
                and self.target_qwen_lora_targets == "attention_mlp"
                and bool(args.initial_checkpoint)
                and not bool(args.stage1_checkpoint)
            )
            if not supported_expansion and not args.dry_run:
                raise SystemExit(
                    "Qwen LoRA target architecture mismatch: warm checkpoint uses "
                    f"{self.warm_qwen_lora_targets!r}, requested pipeline uses "
                    f"{self.target_qwen_lora_targets!r}. Only an attention -> "
                    "attention_mlp expansion during a newly trained Stage 1 is supported. "
                    "Supply the old checkpoint as --initial_checkpoint, not --stage1_checkpoint."
                )

        self.base_env["PYTHONPATH"] = str(self.root) + os.pathsep + self.base_env.get("PYTHONPATH", "")
        self.base_env["GRAPH_DECODER_MODEL"] = args.decoder_model
        self.base_env["GRAPH_ENCODER_MODEL"] = args.encoder_model
        self.base_env["GRAPH_DECODER_REVISION"] = args.decoder_revision
        self.base_env["GRAPH_ENCODER_REVISION"] = args.encoder_revision
        self.base_env["GRAPH_DECODER_PROMPT_MAX_LENGTH"] = str(args.decoder_prompt_max_length)
        self.base_env["GRAPH_QWEN_LORA_TARGETS"] = self.target_qwen_lora_targets
        self.base_env.update({
            "GRAPH_REGION_COMPRESSION": args.region_compression,
            "GRAPH_REGION_MAX_BLOCKS": str(args.region_max_blocks),
            "GRAPH_BLOCK_POOLING": args.block_pooling,
            "GRAPH_BLOCK_VECTORS_PER_BLOCK": str(args.block_vectors_per_block),
            "GRAPH_QWEN_PREFIX_TOKENS": str(args.long_prefix_tokens),
            "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
            "GRAPH_QWEN_PREFIX_MIN_TOKENS": str(args.long_prefix_min_tokens),
            "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": str(args.long_prefix_tokens_per_log2),
            "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
            "GRAPH_ADD_REVERSE_EDGES": "1",
            "GRAPH_BLOCK_POSITION_MODE": "sinusoidal",
            "GRAPH_DFG_MODE": "edges",
            "GRAPH_POSITION_SCHEME": "roberta",
            "GRAPH_GRADIENT_CHECKPOINTING": "1",
            "GRAPH_PROMPT_CLEAN_ASM": "1",
            "GRAPH_PROMPT_FIT_ASSEMBLY": "1",
            "GRAPH_MAX_TARGET_LENGTH": str(args.long_target_max_tokens),
            "GRAPH_DECODER_PROMPT_MAX_LENGTH": str(args.long_prompt_max_tokens),
        })
        self.base_env["GRAPH_SEED"] = str(args.seed)
        self.base_env.setdefault("GRAPH_QUIET", "1")
        self.base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.history: list[dict[str, Any]] = []

        required = [
            self.root / "scripts/training/hybrid_data_controls.py",
            self.root / "scripts/training/checkpoint_contract.py",
            self.root / "scripts/training/prepare_hybrid_training_data_antigravity.py",
            self.root / "scripts/training/build_hierarchical_long_training_antigravity.py",
            self.root / "scripts/training/build_balanced_sft_mix_antigravity.py",
            self.root / "models/graph_data_collator.py",
            self.root / "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
            self.root / "scripts/training/graph_grpo_decompiler_antigravity.py",
            self.root / "scripts/training/verpo_judge_antigravity.py",
            self.root / "scripts/training/build_verpo_repair_dataset_antigravity.py",
            self.root / "scripts/training/teacher_repair_dataset_antigravity.py",
            self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py",
            self.root / "scripts/evaluation/prepare_neutral_evaluation_antigravity.py",
            self.root / "scripts/evaluation/probe_graph_representations_antigravity.py",
            self.root / "scripts/evaluation/functional_graph_gate_antigravity.py",
            self.root / "scripts/evaluation/report_token_lengths_antigravity.py",
            self.root / "scripts/evaluation/graph_inference_antigravity.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Overlay the v2.3 patch into the project root first. Missing:\n  "
                + "\n  ".join(missing)
            )
        code_digest = hashlib.sha256()
        for path in sorted(required, key=str):
            code_digest.update(path.read_bytes())
        self.base_env["GRAPH_HYBRID_PATCH_SHA256"] = code_digest.hexdigest()

    def run(
        self,
        stage: str,
        command: list[str],
        env_updates: dict[str, str] | None = None,
        expected: Path | None = None,
        *,
        force: bool = False,
    ) -> None:
        done_path = self.output / "state" / f"{stage}.done.json"
        env = self.base_env.copy()
        if env_updates:
            env.update({key: str(value) for key, value in env_updates.items()})
        inputs = input_fingerprints(command, env)
        signature = stage_signature(command, env, inputs)

        if (
            self.args.resume
            and not self.args.force
            and not force
            and done_path.is_file()
            and (expected is None or expected.exists())
        ):
            try:
                previous = json.loads(done_path.read_text(encoding="utf-8"))
            except Exception:
                previous = {}
            prior_artifact = previous.get("expected_artifact")
            artifact_matches = expected is None or (
                isinstance(prior_artifact, dict)
                and prior_artifact.get("sha256") == sha256_file(expected)
            )
            if previous.get("stage_signature") == signature and artifact_matches:
                print(f"[{stage}] already complete; reusing {expected or done_path}")
                previous["reused"] = True
                self.history.append(previous)
                return
            print(
                f"[{stage}] previous state or output digest differs from current "
                "inputs/config; rebuilding"
            )

        print(f"\n[{stage}] {display_command(command)}")
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "command": command,
            "environment": env_snapshot(env),
            "input_fingerprints": inputs,
            "stage_signature": signature,
            "expected_artifact": str(expected.resolve()) if expected else None,
        }
        if self.args.dry_run:
            record["dry_run"] = True
            self.history.append(record)
            return

        started = time.time()
        subprocess.run(command, cwd=self.root, env=env, check=True)
        if expected is not None and not expected.exists():
            raise RuntimeError(f"stage {stage} completed but expected artifact is missing: {expected}")
        record["elapsed_seconds"] = time.time() - started
        record["expected_artifact"] = file_record(expected) if expected else None
        write_json(done_path, record)
        self.history.append(record)

    def record_reuse(self, stage: str, artifact: Path, note: str) -> None:
        if not self.args.dry_run:
            file_record(artifact)
        self.history.append(
            {
                "schema_version": SCHEMA_VERSION,
                "stage": stage,
                "command": ["reuse", str(artifact)],
                "environment": {},
                "reused_artifact": file_record(artifact, required=False),
                "note": note,
            }
        )

    def sft_stage(
        self,
        *,
        stage: str,
        train_file: Path,
        eval_file: Path,
        output_dir: Path,
        checkpoint: Path | None,
        prompt_mode: str,
        binary_evidence: bool,
        dependence_weight: float,
        dependence_every: int,
        learning_rate: float,
        epochs: float,
        allow_qwen_lora_expansion: bool = False,
        max_target_length: int | None = None,
        max_prompt_length: int | None = None,
    ) -> Path:
        env = {
            "GRAPH_TRAIN_FILE": str(train_file),
            # Never use the frozen benchmark or the training rows themselves
            # as Trainer eval data. Phase 0 emits a deterministic approved dev
            # split; functional held-out evaluation remains the separate gate.
            "GRAPH_EVAL_FILE": str(eval_file),
            "GRAPH_OUTPUT_DIR": str(output_dir),
            "GRAPH_PROMPT_ASSEMBLY_MODE": prompt_mode,
            "GRAPH_PROMPT_BINARY_EVIDENCE": "1" if binary_evidence else "0",
            "GRAPH_FACTS_FIRST_TARGET": "1",
            "GRAPH_REQUIRE_PHASE0_APPROVED": "1",
            "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "1",
            "GRAPH_NEUTRAL_FUNCTION_NAME": "fn0",
            "GRAPH_PREFIX_DEPENDENCE_WEIGHT": str(dependence_weight),
            "GRAPH_PREFIX_DEPENDENCE_MARGIN": str(self.args.prefix_dependence_margin),
            "GRAPH_PREFIX_DEPENDENCE_EVERY": str(dependence_every),
            "GRAPH_PREFIX_NEGATIVE_BANK_SIZE": str(self.args.prefix_negative_bank_size),
            "GRAPH_PREFIX_GATE_FLOOR": str(self.args.prefix_gate_floor),
            "GRAPH_PREFIX_GATE_FLOOR_WEIGHT": str(self.args.prefix_gate_floor_weight),
            "GRAPH_LR": str(learning_rate),
            "GRAPH_EPOCHS": str(epochs),
            "GRAPH_BATCH_SIZE": str(self.args.sft_batch_size),
            "GRAPH_GRAD_ACCUM": str(self.args.sft_grad_accum),
            "GRAPH_LOAD_4BIT": str(int(self.args.load_4bit)),
            "GRAPH_MAX_STEPS": str(self.args.max_steps),
            "GRAPH_QWEN_LORA_TARGETS": self.target_qwen_lora_targets,
            # All SFT stages instantiate one consistent long-function architecture.
            # This avoids adding region/prefix parameters halfway through training.
            "GRAPH_REGION_COMPRESSION": self.args.region_compression,
            "GRAPH_REGION_MAX_BLOCKS": str(self.args.region_max_blocks),
            "GRAPH_BLOCK_POOLING": self.args.block_pooling,
            "GRAPH_BLOCK_VECTORS_PER_BLOCK": str(self.args.block_vectors_per_block),
            # Env-overridable so a decoder-only run can force prefix=0 (no graph
            # encoder / GNN / prefix adapter -> a plain Qwen3-8B+LoRA text model).
            "GRAPH_QWEN_PREFIX_TOKENS": os.environ.get("GRAPH_QWEN_PREFIX_TOKENS", str(self.args.long_prefix_tokens)),
            "GRAPH_QWEN_PREFIX_DYNAMIC": os.environ.get("GRAPH_QWEN_PREFIX_DYNAMIC", "1"),
            "GRAPH_QWEN_PREFIX_MIN_TOKENS": os.environ.get("GRAPH_QWEN_PREFIX_MIN_TOKENS", str(self.args.long_prefix_min_tokens)),
            "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": str(self.args.long_prefix_tokens_per_log2),
            "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
            "GRAPH_ADD_REVERSE_EDGES": "1",
            "GRAPH_BLOCK_POSITION_MODE": "sinusoidal",
            "GRAPH_DFG_MODE": "edges",
            "GRAPH_POSITION_SCHEME": "roberta",
            "GRAPH_GRADIENT_CHECKPOINTING": "1",
            "GRAPH_PROMPT_CLEAN_ASM": "1",
            "GRAPH_PROMPT_FIT_ASSEMBLY": "1",
            "GRAPH_ALLOW_TARGET_TRUNCATION": "0",
            "GRAPH_ALLOW_PROMPT_TRUNCATION": "0",
            "GRAPH_MAX_TARGET_LENGTH": str(max_target_length or self.args.long_target_max_tokens),
            "GRAPH_DECODER_PROMPT_MAX_LENGTH": str(max_prompt_length or self.args.long_prompt_max_tokens),
        }
        if allow_qwen_lora_expansion:
            env["GRAPH_ALLOW_QWEN_LORA_EXPANSION"] = "1"
        if checkpoint is not None:
            env["GRAPH_CHECKPOINT"] = str(checkpoint)
        target = checkpoint_path(output_dir)
        self.run(
            stage,
            [self.python, "-m", "scripts.training.graph_encoder_decoder_decompiler_v2_antigravity"],
            env,
            target,
        )
        return target

    def functional_gate_command(
        self,
        *,
        dataset: Path,
        checkpoint: Path,
        output_dir: Path,
        report: Path,
        performance_prompt_mode: str,
        min_permutation_drop_pp: float,
        min_facts_drop_pp: float,
        min_improvement_pp: float,
        baseline_checkpoint: Path | None = None,
        baseline_pass_at_k: float = -1.0,
        require_paired_baseline: bool = False,
        run_deployment_arm: bool = True,
        run_causality_arm: bool = True,
        min_rows: int | None = None,
    ) -> list[str]:
        command = [
            self.python,
            str(self.root / "scripts/evaluation/functional_graph_gate_antigravity.py"),
            "--project_root", str(self.root),
            "--dataset", str(dataset),
            "--checkpoint", str(checkpoint),
            "--output_dir", str(output_dir),
            "--report", str(report),
            "--decoder_model", self.args.decoder_model,
            "--decoder_revision", self.args.decoder_revision,
            "--encoder_revision", self.args.encoder_revision,
            "--seed", str(self.args.seed),
            "--num_samples", str(self.args.gate_num_samples),
            "--k", str(self.args.gate_k),
            "--generation_batch_size", str(self.args.gate_generation_batch_size),
            "--max_new_tokens", str(max(self.args.gate_max_new_tokens, self.args.long_generation_max_tokens)),
            "--decoder_prompt_max_length", str(max(self.args.decoder_prompt_max_length, self.args.long_prompt_max_tokens)),
            "--causality_prompt_mode", "graph_only",
            "--performance_prompt_mode", performance_prompt_mode,
            "--run_deployment_arm" if run_deployment_arm else "--no-run_deployment_arm",
            "--run_causality_arm" if run_causality_arm else "--no-run_causality_arm",
            "--min_rows", str(min_rows or self.args.min_gate_rows),
            "--bootstrap_iterations", str(self.args.gate_bootstrap_iterations),
            "--bootstrap_seed", str(self.args.seed),
            "--statistical_confidence", str(self.args.gate_statistical_confidence),
            "--max_sign_test_p_value", str(self.args.gate_max_sign_test_p_value),
            "--min_causal_effective_pairs", str(self.args.gate_min_causal_effective_pairs),
            "--min_deployment_effective_pairs", str(self.args.gate_min_deployment_effective_pairs),
            "--min_causal_permutation_ci_lower_pp", str(self.args.gate_min_permutation_ci_lower_pp),
            "--min_causal_null_ci_lower_pp", str(self.args.gate_min_null_ci_lower_pp),
            "--min_facts_permutation_ci_lower_pp", str(self.args.gate_min_facts_ci_lower_pp),
            "--min_deployment_ci_lower_pp", str(self.args.gate_min_deployment_ci_lower_pp),
            "--require_facts_statistics" if self.args.gate_require_facts_statistics else "--no-require_facts_statistics",
            "--min_causal_permutation_drop_pp", str(min_permutation_drop_pp),
            "--min_facts_permutation_drop_pp", str(min_facts_drop_pp),
            "--min_causal_task_losses", str(self.args.gate_min_causal_task_losses),
            "--min_deployment_improvement_pp", str(min_improvement_pp),
            "--load_4bit", str(int(self.args.load_4bit)),
            "--workers", str(self.args.verifier_workers),
            "--timeout", str(self.args.test_timeout),
        ]
        if require_paired_baseline:
            command += ["--require_paired_baseline"]
        if self.args.limit_tasks:
            command += ["--limit", str(self.args.limit_tasks)]
        if baseline_checkpoint is not None:
            command += ["--baseline_checkpoint", str(baseline_checkpoint)]
        elif baseline_pass_at_k >= 0.0:
            command += ["--baseline_pass_at_k", str(baseline_pass_at_k)]
        return command

    def write_manifest(self, status: str, artifacts: dict[str, Path | None]) -> Path:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "pipeline": (
                "Phase0 all-length approval -> representation probe -> direct all-length SFT -> "
                "region-plan + plan-conditioned hierarchical SFT -> code-only recovery -> "
                "free-running graph gate -> frontier repair -> hidden verification -> "
                "exact 50/50 RS-SFT + matched-step gold control -> +6pp kill switch -> optional VeRPO"
            ),
            "project_root": str(self.root),
            "output_root": str(self.output),
            "arguments": vars(self.args),
            "architecture_contract": {
                "source": file_record(self.architecture_env_source, required=False)
                if self.architecture_env_source else None,
                "environment": self.architecture_env,
                "warm_qwen_lora_targets": self.warm_qwen_lora_targets,
                "pipeline_qwen_lora_targets": self.target_qwen_lora_targets,
                "allow_unpinned": bool(self.args.allow_unpinned_architecture),
            },
            "history": self.history,
            "artifacts": {
                name: file_record(path, required=False) if path is not None else None
                for name, path in artifacts.items()
            },
            "limitations": [
                "The counterfactual NLL objective is diagnostic/auxiliary; it is not a success gate.",
                "All approved instruction strata, including >=200, are used in SFT and hierarchical recovery.",
                "A small deterministic length-stratified development slice remains withheld for model selection.",
                "Long-function success must still be reported separately by instruction stratum.",
            ],
        }
        path = self.output / "hybrid_curriculum_manifest.json"
        write_json(path, manifest)
        return path

    def execute(self) -> int:
        args = self.args
        train_file = Path(args.train_file).expanduser().resolve()
        eval_file = Path(args.eval_file).expanduser().resolve()
        functional_eval_file = Path(args.functional_eval_file or eval_file).expanduser().resolve()
        file_record(train_file)
        file_record(eval_file)
        file_record(functional_eval_file)
        forbidden_evals = [eval_file]
        if functional_eval_file not in forbidden_evals:
            forbidden_evals.append(functional_eval_file)
        for value in args.frozen_eval_file:
            path = Path(value).expanduser().resolve()
            file_record(path)
            if path not in forbidden_evals:
                forbidden_evals.append(path)

        # --- Standalone gate-eval mode (no training): evaluate an existing
        # checkpoint on an explicit dataset with the standard functional gate,
        # inheriting the orchestrator's base_env (patch SHA, PEFT config, prompt
        # budget). Used for the CLEAN-assembly long-budget re-test (02a3) that
        # pairs against 02a2 with only {assembly text, prompt budget} changed.
        if args.eval_only_dataset:
            eval_dataset = Path(args.eval_only_dataset).expanduser().resolve()
            eval_checkpoint = Path(args.eval_only_checkpoint).expanduser().resolve()
            eval_dir = self.output / args.eval_only_output
            eval_report = eval_dir / "report.json"
            eval_cmd = self.functional_gate_command(
                dataset=eval_dataset,
                checkpoint=eval_checkpoint,
                output_dir=eval_dir,
                report=eval_report,
                performance_prompt_mode="full",
                min_permutation_drop_pp=0.0,
                min_facts_drop_pp=0.0,
                min_improvement_pp=0.0,
                run_deployment_arm=True,
                run_causality_arm=False,
                min_rows=args.min_long_gate_rows,
            )
            print(f"\n[{args.eval_only_output}] {display_command(eval_cmd)}")
            if not args.dry_run:
                eval_dir.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(eval_cmd, cwd=self.root, env=self.base_env.copy())
                print(f"\nEval-only gate exit={completed.returncode}; report: {eval_report}")
                return completed.returncode
            return 0

        initial_checkpoint = Path(args.initial_checkpoint).expanduser().resolve() if args.initial_checkpoint else None
        stage1_supplied = Path(args.stage1_checkpoint).expanduser().resolve() if args.stage1_checkpoint else None
        probe_checkpoint = Path(args.probe_checkpoint).expanduser().resolve() if args.probe_checkpoint else (initial_checkpoint or stage1_supplied)
        if not args.dry_run:
            if initial_checkpoint is not None:
                file_record(initial_checkpoint)
            if stage1_supplied is not None:
                file_record(stage1_supplied)
        # With the new hierarchical architecture a legacy Regions16 checkpoint
        # may not contain the region/prefix modules. An explicit compatible
        # checkpoint is probed before training; otherwise the representation
        # probe runs immediately after direct all-length SFT.

        phase0_dir = self.output / "00_phase0"
        raw_token_length_report = phase0_dir / "raw_token_length_distribution.json"
        effective_generation_budget = min(
            validate_target_generation_budget(args).values()
        )
        self.run(
            "00_raw_token_length_measurement",
            [
                self.python,
                str(self.root / "scripts/evaluation/report_token_lengths_antigravity.py"),
                "--dataset", str(train_file),
                "--output", str(raw_token_length_report),
                "--decoder_model", args.decoder_model,
                "--decoder_revision", args.decoder_revision,
                "--max_target_tokens", str(args.long_target_max_tokens),
                "--max_prompt_tokens", str(args.long_prompt_max_tokens),
                "--max_generation_tokens", str(effective_generation_budget),
                "--short_max", str(args.max_train_instructions),
                "--bridge_max", str(args.max_bridge_instructions),
                "--historical_limits", "768,1024,2048,3072",
                "--prompt_mode", "full",
                "--no-fail_on_overflow",
            ],
            {
                "GRAPH_PROMPT_CLEAN_ASM": "1",
                "GRAPH_PROMPT_FIT_ASSEMBLY": "1",
                "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "0",
            },
            expected=raw_token_length_report,
        )

        approved_all = phase0_dir / "approved_all_length_train.jsonl"
        approved_dev = phase0_dir / "approved_all_length_dev.jsonl"
        short_train = phase0_dir / "approved_short_le150.jsonl"
        bridge_train = phase0_dir / "approved_bridge_151_199.jsonl"
        long_train = phase0_dir / "approved_long_ge200.jsonl"
        phase0_report = phase0_dir / "phase0_report.json"
        phase0_cmd = [
            self.python,
            str(self.root / "scripts/training/prepare_hybrid_training_data_antigravity.py"),
            "--input", str(train_file),
            "--output", str(approved_all),
            "--dev_output", str(approved_dev),
            "--short_output", str(short_train),
            "--dev_fraction", str(args.dev_fraction),
            "--min_dev_rows", str(args.min_dev_rows),
            "--bridge_output", str(bridge_train),
            "--long_output", str(long_train),
            "--report", str(phase0_report),
            "--neutral_name", "fn0",
            "--data_role", args.data_role,
            "--feedback_fraction", str(args.feedback_fraction),
            "--max_instructions", str(args.max_train_instructions),
            "--max_bridge_instructions", str(args.max_bridge_instructions),
            "--min_short_rows", str(args.min_short_rows),
            "--min_long_rows", str(args.min_long_rows),
            "--seed", str(args.seed),
            "--timeout", str(args.test_timeout),
            "--workers", str(args.verifier_workers),
            # Auto-drop the rare non-deterministic reference (e.g. differential
            # master oracles whose stdout embeds the current date) instead of
            # aborting the whole gate. Such rows are already excluded from the
            # approved/training set; this only stops the strict tripwire from
            # failing on them, and they cannot be reliably pre-stripped because
            # their flakiness is time/random dependent.
            "--drop_invalid_references",
        ]
        for path in forbidden_evals:
            phase0_cmd += ["--forbidden_eval", str(path)]
        self.run("00a_phase0_prepare", phase0_cmd, expected=phase0_report)

        phase0_audit = phase0_dir / "approved_all_length_reference_audit.json"
        self.run(
            "00b_phase0_reference_audit",
            [
                self.python,
                str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                "--dataset", str(approved_all),
                "--test_fields", "feedback_tests,acceptance_tests,tests",
                "--run_references", "-1",
                "--workers", str(args.verifier_workers),
                "--timeout", str(args.test_timeout),
                "--require_phase0_approved",
                "--require_neutral_contract",
                "--report", str(phase0_audit),
            ],
            expected=phase0_audit,
        )

        phase0_dev_audit = phase0_dir / "approved_all_length_dev_reference_audit.json"
        self.run(
            "00b2_phase0_dev_reference_audit",
            [
                self.python,
                str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                "--dataset", str(approved_dev),
                "--test_fields", "feedback_tests,acceptance_tests,tests",
                "--run_references", "-1",
                "--workers", str(args.verifier_workers),
                "--timeout", str(args.test_timeout),
                "--require_phase0_approved",
                "--require_neutral_contract",
                "--report", str(phase0_dev_audit),
            ],
            expected=phase0_dev_audit,
        )

        hierarchy_dir = self.output / "00f_hierarchical_data"
        direct_train = hierarchy_dir / "direct_all_length_train.jsonl"
        hierarchy_train = hierarchy_dir / "region_plan_multitask_train.jsonl"
        hierarchy_control_train = hierarchy_dir / "matched_direct_control_train.jsonl"
        recovery_train = hierarchy_dir / "code_recovery_train.jsonl"
        hierarchy_report = hierarchy_dir / "train_report.json"
        self.run(
            "00f_build_hierarchical_train",
            [
                self.python,
                str(self.root / "scripts/training/build_hierarchical_long_training_antigravity.py"),
                "--input", str(approved_all),
                "--direct_output", str(direct_train),
                "--hierarchical_output", str(hierarchy_train),
                "--matched_control_output", str(hierarchy_control_train),
                "--recovery_output", str(recovery_train),
                "--report", str(hierarchy_report),
                "--short_max", str(args.max_train_instructions),
                "--bridge_max", str(args.max_bridge_instructions),
                "--max_region_blocks", str(args.region_max_blocks),
                "--short_replay_fraction", str(args.hierarchical_short_replay_fraction),
                "--short_repeat", str(args.recovery_short_repeat),
                "--bridge_repeat", str(args.recovery_bridge_repeat),
                "--long_repeat", str(args.recovery_long_repeat),
                "--min_long_rows", str(args.min_long_rows),
                "--seed", str(args.seed),
            ],
            expected=hierarchy_report,
        )

        direct_dev = hierarchy_dir / "direct_all_length_dev.jsonl"
        hierarchy_dev = hierarchy_dir / "region_plan_multitask_dev.jsonl"
        hierarchy_control_dev = hierarchy_dir / "matched_direct_control_dev.jsonl"
        recovery_dev = hierarchy_dir / "code_recovery_dev.jsonl"
        short_dev = hierarchy_dir / "short_dev.jsonl"
        bridge_dev = hierarchy_dir / "bridge_dev.jsonl"
        long_dev = hierarchy_dir / "long_dev_ge200.jsonl"
        hierarchy_dev_report = hierarchy_dir / "dev_report.json"
        self.run(
            "00g_build_hierarchical_dev",
            [
                self.python,
                str(self.root / "scripts/training/build_hierarchical_long_training_antigravity.py"),
                "--input", str(approved_dev),
                "--direct_output", str(direct_dev),
                "--hierarchical_output", str(hierarchy_dev),
                "--matched_control_output", str(hierarchy_control_dev),
                "--recovery_output", str(recovery_dev),
                "--short_subset_output", str(short_dev),
                "--bridge_subset_output", str(bridge_dev),
                "--long_subset_output", str(long_dev),
                "--report", str(hierarchy_dev_report),
                "--short_max", str(args.max_train_instructions),
                "--bridge_max", str(args.max_bridge_instructions),
                "--max_region_blocks", str(args.region_max_blocks),
                "--short_replay_fraction", str(args.hierarchical_short_replay_fraction),
                "--short_repeat", "1",
                "--bridge_repeat", "1",
                "--long_repeat", "1",
                "--min_long_rows", "0",
                "--seed", str(args.seed),
            ],
            expected=hierarchy_dev_report,
        )
        # The functional gates (02a2/02e) fail closed unless every eval row is
        # marked evaluation_only. build_hierarchical predates that requirement, so
        # flag the gate-consumed long dev subset here in the orchestrator (not a
        # required-files-hashed module, so this does not resign the training
        # stages). Runs whether 00g executed or was reused.
        if not args.dry_run:
            mark_rows_evaluation_only(long_dev)

        token_length_report = phase0_dir / "token_length_distribution.json"
        self.run(
            "00h_token_length_preflight",
            [
                self.python,
                str(self.root / "scripts/evaluation/report_token_lengths_antigravity.py"),
                "--dataset", str(direct_train),
                "--dataset", str(hierarchy_train),
                "--dataset", str(recovery_train),
                "--dataset", str(direct_dev),
                "--output", str(token_length_report),
                "--decoder_model", args.decoder_model,
                "--decoder_revision", args.decoder_revision,
                "--max_target_tokens", str(args.long_target_max_tokens),
                "--max_prompt_tokens", str(args.long_prompt_max_tokens),
                "--max_generation_tokens", str(effective_generation_budget),
                "--short_max", str(args.max_train_instructions),
                "--bridge_max", str(args.max_bridge_instructions),
                "--historical_limits", "768,1024,2048,3072",
                "--prompt_mode", "full",
                "--fail_on_overflow",
            ],
            {
                "GRAPH_PROMPT_CLEAN_ASM": "1",
                "GRAPH_PROMPT_FIT_ASSEMBLY": "1",
                "GRAPH_PROMPT_BINARY_EVIDENCE": "1",
                "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "1",
                "GRAPH_NEUTRAL_FUNCTION_NAME": "fn0",
            },
            expected=token_length_report,
        )

        neutral_eval = phase0_dir / "neutral_functional_eval.jsonl"
        neutral_report = phase0_dir / "neutral_functional_eval_report.json"
        self.run(
            "00c_prepare_neutral_gate",
            [
                self.python,
                str(self.root / "scripts/evaluation/prepare_neutral_evaluation_antigravity.py"),
                "--input", str(functional_eval_file),
                "--output", str(neutral_eval),
                "--report", str(neutral_report),
                "--neutral_name", "fn0",
                "--min_rows", str(args.min_gate_rows),
                "--timeout", str(args.test_timeout),
                "--workers", str(args.verifier_workers),
            ],
            expected=neutral_eval,
        )

        neutral_audit = phase0_dir / "neutral_functional_eval_audit.json"
        self.run(
            "00d_neutral_gate_audit",
            [
                self.python,
                str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                "--dataset", str(neutral_eval),
                "--test_fields", "tests",
                "--run_references", "-1",
                "--workers", str(args.verifier_workers),
                "--timeout", str(args.test_timeout),
                "--require_neutral_contract",
                "--report", str(neutral_audit),
            ],
            expected=neutral_audit,
        )

        # --- Clean text-only arm (no graph curriculum) -----------------------
        # The graph channel is causally dead (forced-probe null 2026-07-20) and
        # truncation is not the >=200 wall (02a3 null). This arm trains ONE text
        # SFT (assembly->Dart, full prompt) from base on the sealed all-length
        # corpus, then runs the long gate + rollouts + verifier feedback (stages
        # 02e/03/04). It skips graph-only 01, the 02a control chain, the dead 02b
        # gate, and hierarchical 02c. The graph prefix module is still built but
        # is behaviorally inert, so "full" prompt mode == text conditioning.
        if args.text_only:
            text_dir = self.output / "02t_text_sft"
            if args.text_sft_checkpoint:
                text_ckpt = Path(args.text_sft_checkpoint).expanduser().resolve()
                if not args.dry_run and not text_ckpt.is_file():
                    raise FileNotFoundError(f"--text_sft_checkpoint not found: {text_ckpt}")
                self.record_reuse("02t_text_sft", text_ckpt, "supplied text SFT checkpoint")
            else:
                text_ckpt = self.sft_stage(
                    stage="02t_text_sft",
                    train_file=recovery_train,
                    eval_file=recovery_dev,
                    output_dir=text_dir,
                    checkpoint=initial_checkpoint,
                    prompt_mode="full",
                    binary_evidence=True,
                    dependence_weight=0.0,
                    dependence_every=1,
                    learning_rate=args.recovery_lr,
                    epochs=args.recovery_epochs,
                    max_target_length=args.long_target_max_tokens,
                    max_prompt_length=args.long_prompt_max_tokens,
                )

            # Finish path: 06 harvest already done -> run 07 mix -> 08b RS-SFT ->
            # 09 functional gate on the sealed dev, reusing the exact sft_stage /
            # functional_gate_command config. The +6pp kill switch always uses a
            # separately generated matched-step checkpoint baseline; an aggregate
            # pass@k scalar cannot provide paired task uncertainty.
            if getattr(args, "text_sft_only", False):
                print(f"\n[text_sft_only] decoder SFT checkpoint ready: {text_ckpt}")
                return 0

            if getattr(args, "text_finish_rs_sft", False):
                curriculum_checkpoint = text_ckpt
                raw_rs_sft = (
                    Path(args.rs_sft_file).expanduser().resolve()
                    if args.rs_sft_file
                    else self.output / "04_teacher_harvest" / "verified_rs_sft.jsonl"
                )
                if not args.dry_run and not raw_rs_sft.is_file():
                    raise FileNotFoundError(f"--rs_sft_file not found: {raw_rs_sft}")
                self.record_reuse(
                    "06_build_verified_rs_sft",
                    raw_rs_sft,
                    "supplied RS-SFT curriculum pending independent re-certification",
                )
                recertify_dir = self.output / "06a_recertified_verified_rs_sft"
                rs_sft = recertify_dir / "verified_rs_sft.jsonl"
                recertify_report = recertify_dir / "report.json"
                self.run(
                    "06a_recertify_verified_rs_sft",
                    [
                        self.python,
                        str(self.root / "scripts/training/teacher_repair_dataset_antigravity.py"),
                        "recertify",
                        "--input", str(raw_rs_sft),
                        "--out", str(rs_sft),
                        "--report", str(recertify_report),
                        "--timeout", str(args.test_timeout),
                        "--workers", str(args.verifier_workers),
                        "--facts_gate_mode", args.rs_sft_recertify_facts_gate_mode,
                        "--min_verified_rows", str(args.min_verified_rows),
                        "--min_verified_unique_tasks", str(args.min_verified_unique_tasks),
                        "--min_verified_length_bins", str(args.min_verified_length_bins),
                    ],
                    expected=rs_sft,
                )
                verified_audit = self.output / "04_teacher_harvest" / "verified_rs_sft_audit.json"
                self.run(
                    "06b_verified_rs_sft_audit",
                    [
                        self.python,
                        str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                        "--dataset", str(rs_sft),
                        "--test_fields", "feedback_tests,acceptance_tests,tests",
                        "--run_references", "-1",
                        "--workers", str(args.verifier_workers),
                        "--timeout", str(args.test_timeout),
                        "--require_phase0_approved",
                        "--require_neutral_contract",
                        "--require_verified_origin",
                        "--report", str(verified_audit),
                    ],
                    expected=verified_audit,
                )
                mix_dir = self.output / "07_balanced_rs_sft"
                balanced_mix = mix_dir / "balanced_50_50.jsonl"
                gold_control = mix_dir / "gold_only_matched_steps.jsonl"
                mix_report = mix_dir / "balanced_50_50_report.json"
                mix_cmd = [
                    self.python,
                    str(self.root / "scripts/training/build_balanced_sft_mix_antigravity.py"),
                    "--gold", str(approved_all),
                    "--verified", str(rs_sft),
                    "--out", str(balanced_mix),
                    "--gold_control_out", str(gold_control),
                    "--report", str(mix_report),
                    "--seed", str(args.seed),
                    "--min_verified_rows", str(args.min_verified_rows),
                    "--min_verified_unique_tasks", str(args.min_verified_unique_tasks),
                    "--min_verified_length_bins", str(args.min_verified_length_bins),
                    "--max_verified_oversample_factor", str(args.max_verified_oversample_factor),
                ]
                if args.rs_sft_rows_per_epoch:
                    mix_cmd += ["--rows_per_epoch", str(args.rs_sft_rows_per_epoch)]
                if getattr(args, "rs_sft_allow_partial_gold", False):
                    mix_cmd += ["--allow_partial_gold_coverage"]
                self.run("07_build_balanced_sft_mix", mix_cmd, expected=balanced_mix)

                balanced_audit = mix_dir / "balanced_mix_audit.json"
                self.run(
                    "07b_balanced_mix_audit",
                    [
                        self.python,
                        str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                        "--dataset", str(balanced_mix),
                        "--test_fields", "feedback_tests,acceptance_tests,tests",
                        "--run_references", "-1",
                        "--workers", str(args.verifier_workers),
                        "--timeout", str(args.test_timeout),
                        "--require_phase0_approved",
                        "--require_neutral_contract",
                        "--report", str(balanced_audit),
                    ],
                    expected=balanced_audit,
                )

                baseline_checkpoint: Path
                if args.rs_sft_baseline_checkpoint:
                    baseline_checkpoint = Path(
                        args.rs_sft_baseline_checkpoint
                    ).expanduser().resolve()
                    if not args.dry_run:
                        file_record(baseline_checkpoint)
                    self.record_reuse(
                        "08a_gold_only_full_sft_control",
                        baseline_checkpoint,
                        "supplied external matched RS-SFT control checkpoint",
                    )
                else:
                    control_dir = self.output / "08a_gold_only_full_sft_control"
                    baseline_checkpoint = self.sft_stage(
                        stage="08a_gold_only_full_sft_control",
                        train_file=gold_control,
                        eval_file=approved_dev,
                        output_dir=control_dir,
                        checkpoint=curriculum_checkpoint,
                        prompt_mode="full",
                        binary_evidence=True,
                        dependence_weight=0.0,
                        dependence_every=1,
                        learning_rate=args.stage2_lr,
                        epochs=args.stage2_epochs,
                        max_target_length=args.long_target_max_tokens,
                        max_prompt_length=args.long_prompt_max_tokens,
                    )
                stage2_dir = self.output / "08b_rejection_sampling_sft"
                stage2_checkpoint = self.sft_stage(
                    stage="08b_rejection_sampling_sft",
                    train_file=balanced_mix,
                    eval_file=approved_dev,
                    output_dir=stage2_dir,
                    checkpoint=curriculum_checkpoint,
                    prompt_mode="full",
                    binary_evidence=True,
                    dependence_weight=0.0,
                    dependence_every=1,
                    learning_rate=args.stage2_lr,
                    epochs=args.stage2_epochs,
                    max_target_length=args.long_target_max_tokens,
                    max_prompt_length=args.long_prompt_max_tokens,
                )
                gate_dataset = (
                    Path(args.text_eval_file).expanduser().resolve()
                    if args.text_eval_file else long_dev
                )
                stage2_gate_dir = self.output / "09_rs_sft_functional_kill_switch"
                stage2_gate_report = stage2_gate_dir / "report.json"
                self.run(
                    "09_rs_sft_functional_kill_switch",
                    self.functional_gate_command(
                        dataset=gate_dataset,
                        checkpoint=stage2_checkpoint,
                        output_dir=stage2_gate_dir,
                        report=stage2_gate_report,
                        performance_prompt_mode="full",
                        min_permutation_drop_pp=0.0,
                        min_facts_drop_pp=0.0,
                        min_improvement_pp=args.rs_sft_min_improvement_pp,
                        baseline_checkpoint=baseline_checkpoint,
                        require_paired_baseline=True,
                        run_deployment_arm=True,
                        run_causality_arm=False,
                        min_rows=args.min_long_gate_rows,
                    ),
                    expected=stage2_gate_report,
                )
                manifest = self.write_manifest(
                    "dry_run" if args.dry_run else "completed",
                    {
                        "supplied_rs_sft": raw_rs_sft,
                        "verified_rs_sft": rs_sft,
                        "verified_rs_sft_recertification": recertify_report,
                        "verified_rs_sft_audit": verified_audit,
                        "balanced_rs_sft_mix": balanced_mix,
                        "balanced_mix_report": mix_report,
                        "balanced_mix_audit": balanced_audit,
                        "gold_only_matched_step_control": gold_control,
                        "gold_only_control_checkpoint": baseline_checkpoint,
                        "stage2_checkpoint": stage2_checkpoint,
                        "stage2_functional_gate": stage2_gate_report,
                    },
                )
                print(
                    f"\n[text_finish_rs_sft] 08b={stage2_checkpoint} "
                    f"paired_gate={stage2_gate_report} manifest={manifest}"
                )
                return 0

            if getattr(args, "text_verpo", False):
                verpo_ckpt = (
                    Path(args.verpo_checkpoint).expanduser().resolve()
                    if args.verpo_checkpoint
                    else checkpoint_path(self.output / "08b_rejection_sampling_sft")
                )
                if not args.dry_run and not Path(verpo_ckpt).is_file():
                    raise FileNotFoundError(f"--verpo_checkpoint / 08b init not found: {verpo_ckpt}")
                rs_sft = (
                    Path(args.rs_sft_file).expanduser().resolve()
                    if args.rs_sft_file
                    else
                    self.output
                    / "06a_recertified_verified_rs_sft"
                    / "verified_rs_sft.jsonl"
                )
                self.record_reuse("08b_rejection_sampling_sft", Path(verpo_ckpt), "supplied VeRPO init checkpoint")
                grpo_dir = self.output / "10_verpo"
                grpo_env = {
                    "GRPO_TRAIN_FILE": str(approved_all),
                    "GRPO_VERIFIED_ANCHOR_FILE": str(rs_sft),
                    "GRAPH_CHECKPOINT": str(verpo_ckpt),
                    "GRAPH_OUTPUT_DIR": str(grpo_dir),
                    "GRAPH_PROMPT_ASSEMBLY_MODE": "full",
                    "GRAPH_PROMPT_BINARY_EVIDENCE": "1",
                    "GRAPH_REQUIRE_PHASE0_APPROVED": "1",
                    "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "1",
                    "GRAPH_NEUTRAL_FUNCTION_NAME": "fn0",
                    "GRAPH_FACTS_FIRST_TARGET": "1",
                    "GRAPH_LOAD_4BIT": str(int(args.load_4bit)),
                    "GRAPH_MAX_TARGET_LENGTH": str(args.long_target_max_tokens),
                    "GRAPH_DECODER_PROMPT_MAX_LENGTH": str(args.long_prompt_max_tokens),
                    "GRAPH_REGION_COMPRESSION": args.region_compression,
                    "GRAPH_REGION_MAX_BLOCKS": str(args.region_max_blocks),
                    "GRAPH_BLOCK_POOLING": args.block_pooling,
                    "GRAPH_BLOCK_VECTORS_PER_BLOCK": str(args.block_vectors_per_block),
                    "GRAPH_QWEN_PREFIX_TOKENS": str(args.long_prefix_tokens),
                    "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
                    "GRAPH_QWEN_PREFIX_MIN_TOKENS": str(args.long_prefix_min_tokens),
                    "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": str(args.long_prefix_tokens_per_log2),
                    "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
                    "GRAPH_ADD_REVERSE_EDGES": "1",
                    "GRAPH_BLOCK_POSITION_MODE": "sinusoidal",
                    "GRAPH_DFG_MODE": "edges",
                    "GRAPH_POSITION_SCHEME": "roberta",
                    "GRAPH_BATCH_SIZE": str(args.grpo_batch_size),
                    "GRAPH_GRAD_ACCUM": str(args.grpo_grad_accum),
                    "GRPO_REWARD_MODE": args.grpo_reward_mode,
                    "GRPO_VERPO_JUDGE": "1" if args.verpo_judge else "0",
                    "GRPO_VERPO_JUDGE_WEIGHT": str(args.verpo_judge_weight),
                    "GRPO_VERPO_FULL_PASS_MARGIN": str(args.verpo_full_pass_margin),
                    "GRPO_VERPO_REPAIR": "1" if args.verpo_repair else "0",
                    "VERPO_JUDGE_MODEL": args.verpo_judge_model,
                    "VERPO_JUDGE_BASE_URL": args.verpo_judge_base_url,
                    "VERPO_JUDGE_CONCURRENCY": str(args.verpo_judge_concurrency),
                    "VERPO_JUDGE_TIMEOUT_SECONDS": str(args.verpo_judge_timeout_seconds),
                    "VERPO_JUDGE_MAX_RETRIES": str(args.verpo_judge_max_retries),
                    "VERPO_JUDGE_FAIL_CLOSED": "1",
                    "VERPO_JUDGE_REQUIRE_SUCCESS": "1",
                    "GRPO_REWARD_TEST_FIELD": "feedback_tests",
                    "GRPO_GROUP_SIZE": str(args.grpo_group_size),
                    "GRPO_EPOCHS": str(args.grpo_epochs),
                    "GRPO_LR": str(args.grpo_lr),
                    "GRPO_TEST_TIMEOUT": str(args.test_timeout),
                    "GRPO_MAX_NEW_TOKENS": str(max(args.grpo_max_new_tokens, args.long_generation_max_tokens)),
                    "GRPO_REWARD_WORKERS": str(args.verifier_workers),
                    "GRPO_ADV_NORM": "mean",
                    "GRPO_MIN_REWARD_RANGE": str(args.grpo_min_reward_range),
                    "GRPO_PASSK_K": str(args.grpo_passk_k),
                    "GRPO_LOSS_POOLING": "seq",
                    "GRPO_SCORE_CHUNK_SIZE": str(args.grpo_score_chunk_size),
                    "GRPO_GENERATION_CHUNK_SIZE": str(args.grpo_generation_chunk_size),
                    # Scoring uses the full temperature-scaled softmax, so
                    # rollout sampling must not apply nucleus truncation.
                    "GRPO_GEN_TOP_P": "1.0",
                    # This checkpoint retains a 64-token graph-prefix contract.
                    # Freezing the glue removes its backward graph but does not
                    # pretend graph encoding is skipped. Rollout chunking bounds
                    # the decoder KV cache without changing checkpoint semantics.
                    "GRPO_TRAIN_GRAPH_GLUE": os.environ.get("GRPO_TRAIN_GRAPH_GLUE", "0"),
                    "GRPO_KL_COEF": "0.0",
                    "GRPO_SFT_ANCHOR_COEF": str(args.grpo_sft_anchor_coef),
                    "GRPO_SFT_ANCHOR_ON_NO_SIGNAL": "0",
                    "GRPO_DYNAMIC_RESAMPLE_ATTEMPTS": str(args.grpo_dynamic_resample_attempts),
                    "GRAPH_SAVE_STRATEGY": "steps",
                    "GRAPH_SAVE_STEPS": str(args.grpo_save_steps),
                    "GRAPH_SAVE_TOTAL_LIMIT": "2",
                    "GRAPH_MAX_STEPS": str(args.max_steps),
                }
                if args.verpo_repair:
                    grpo_env["GRPO_TRAIN_FILE"] = str(
                        Path(args.verpo_repair_file).expanduser().resolve()
                    )
                self.run(
                    "10_verpo",
                    [self.python, "-m", "scripts.training.graph_grpo_decompiler_antigravity"],
                    grpo_env,
                    checkpoint_path(grpo_dir),
                )
                print(f"\n[text_verpo] 10_verpo checkpoint: {checkpoint_path(grpo_dir)}")
                return 0

            text_gate_dir = self.output / "02te_text_long_gate"
            text_gate_report = text_gate_dir / "report.json"
            text_gate_dataset = (
                Path(args.text_eval_file).expanduser().resolve()
                if args.text_eval_file else long_dev
            )
            if args.text_eval_file and not args.dry_run:
                mark_rows_evaluation_only(text_gate_dataset)
            self.run(
                "02te_text_long_gate",
                self.functional_gate_command(
                    dataset=text_gate_dataset,
                    checkpoint=text_ckpt,
                    output_dir=text_gate_dir,
                    report=text_gate_report,
                    performance_prompt_mode="full",
                    min_permutation_drop_pp=0.0,
                    min_facts_drop_pp=0.0,
                    min_improvement_pp=0.0,
                    run_deployment_arm=True,
                    run_causality_arm=False,
                    min_rows=args.min_long_gate_rows,
                ),
                expected=text_gate_report,
            )

            text_predictions = self.output / "03_train_rollouts" / "predictions.json"
            if args.predictions:
                text_predictions = Path(args.predictions).expanduser().resolve()
                self.record_reuse("03_train_rollouts", text_predictions, "supplied rollout pool")
            else:
                text_infer_env = {
                    "GRAPH_PROMPT_ASSEMBLY_MODE": "full",
                    "GRAPH_PROMPT_BINARY_EVIDENCE": "1",
                    "GRAPH_REGION_COMPRESSION": args.region_compression,
                    "GRAPH_REGION_MAX_BLOCKS": str(args.region_max_blocks),
                    "GRAPH_BLOCK_POOLING": args.block_pooling,
                    "GRAPH_BLOCK_VECTORS_PER_BLOCK": str(args.block_vectors_per_block),
                    "GRAPH_QWEN_PREFIX_TOKENS": str(args.long_prefix_tokens),
                    "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
                    "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
                    "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "1",
                    "GRAPH_NEUTRAL_FUNCTION_NAME": "fn0",
                    "GRAPH_LOAD_4BIT": str(int(args.load_4bit)),
                    "GRAPH_MAX_TARGET_LENGTH": str(args.long_target_max_tokens),
                    "GRAPH_DECODER_PROMPT_MAX_LENGTH": str(args.long_prompt_max_tokens),
                }
                text_infer_cmd = [
                    self.python,
                    str(self.root / "scripts/evaluation/graph_inference_antigravity.py"),
                    "--dataset", str(approved_all),
                    "--decoder_model", args.decoder_model,
                    "--checkpoint", str(text_ckpt),
                    "--output", str(text_predictions),
                    "--num_samples", str(args.rollout_samples),
                    "--generation_batch_size", str(args.rollout_generation_batch_size),
                    "--max_new_tokens", str(max(args.rollout_max_new_tokens, args.long_generation_max_tokens)),
                    "--decoder_prompt_max_length", str(args.long_prompt_max_tokens),
                    "--decoder_revision", args.decoder_revision,
                    "--encoder_revision", args.encoder_revision,
                    "--seed", str(args.seed),
                    "--graph_input_ablation", "none",
                ]
                if args.limit_tasks:
                    text_infer_cmd += ["--limit", str(args.limit_tasks)]
                self.run("03_train_rollouts", text_infer_cmd, text_infer_env, text_predictions)

            text_teacher_dir = self.output / "04_teacher_harvest"
            text_collected = text_teacher_dir / "collected.jsonl"
            text_collect_cmd = [
                self.python,
                str(self.root / "scripts/training/teacher_repair_dataset_antigravity.py"),
                "collect",
                "--dataset", str(approved_all),
                "--predictions", str(text_predictions),
                "--prediction_provenance", str(text_predictions) + ".provenance.json",
                "--expected_checkpoint", str(text_ckpt),
                "--phase0_report", str(phase0_report),
                "--data_role", args.data_role,
                "--out", str(text_collected),
                "--timeout", str(args.test_timeout),
                "--workers", str(args.verifier_workers),
                "--max_failures_per_task", str(args.max_failures_per_task),
                "--max_positives_per_task", str(args.max_positives_per_task),
            ]
            if args.limit_candidates:
                text_collect_cmd += ["--limit_candidates", str(args.limit_candidates)]
            self.run("04_collect_verifier_feedback", text_collect_cmd, expected=text_collected)

            print(f"\n[text_only] complete through stage 04. gate={text_gate_report} "
                  f"rollouts={text_predictions} feedback={text_collected}")
            print("[text_only] stage 05 (frontier teacher, paid API) NOT started — rerun without --text_only or add a teacher stage when ready.")
            return 0

        probe_report = phase0_dir / "representation_probe.json"
        if args.run_probe and probe_checkpoint is not None:
            self.run(
                "00e_representation_probe",
                [
                    self.python,
                    str(self.root / "scripts/evaluation/probe_graph_representations_antigravity.py"),
                    "--dataset", str(approved_all),
                    "--checkpoint", str(probe_checkpoint),
                    "--report", str(probe_report),
                    "--decoder_model", args.decoder_model,
                    "--encoder_model", args.encoder_model,
                    "--decoder_revision", args.decoder_revision,
                    "--encoder_revision", args.encoder_revision,
                    "--seed", str(args.seed),
                    "--max_rows", str(args.probe_max_rows),
                    "--min_rows", str(args.probe_min_rows),
                    "--min_prefix_semantic_r2", str(args.probe_min_prefix_semantic_r2),
                    "--min_prefix_mean_lift", str(args.probe_min_prefix_mean_lift),
                    "--min_retrieval_above_chance", str(args.probe_min_retrieval_above_chance),
                ],
                {"GRAPH_QWEN_LORA_TARGETS": self.warm_qwen_lora_targets},
                expected=probe_report,
            )
        elif not args.run_probe:
            self.history.append({"stage": "00e_representation_probe", "note": "probe explicitly disabled"})

        stage1_dir = self.output / "01_graph_only_sft"
        if stage1_supplied is not None:
            stage1_checkpoint = stage1_supplied
            self.record_reuse("01_graph_only_sft", stage1_checkpoint, "supplied graph-only checkpoint")
        else:
            stage1_checkpoint = self.sft_stage(
                stage="01_graph_only_sft",
                train_file=direct_train,
                eval_file=direct_dev,
                output_dir=stage1_dir,
                checkpoint=initial_checkpoint,
                prompt_mode="graph_only",
                binary_evidence=False,
                dependence_weight=args.stage1_dependence_weight,
                dependence_every=args.stage1_dependence_every,
                learning_rate=args.stage1_lr,
                epochs=args.stage1_epochs,
                allow_qwen_lora_expansion=(
                    initial_checkpoint is not None
                    and self.warm_qwen_lora_targets == "attention"
                    and self.target_qwen_lora_targets == "attention_mlp"
                ),
            )

        if args.run_probe and probe_checkpoint is None:
            self.run(
                "01b_post_direct_sft_representation_probe",
                [
                    self.python,
                    str(self.root / "scripts/evaluation/probe_graph_representations_antigravity.py"),
                    "--dataset", str(approved_all),
                    "--checkpoint", str(stage1_checkpoint),
                    "--report", str(probe_report),
                    "--decoder_model", args.decoder_model,
                    "--encoder_model", args.encoder_model,
                    "--decoder_revision", args.decoder_revision,
                    "--encoder_revision", args.encoder_revision,
                    "--seed", str(args.seed),
                    "--max_rows", str(args.probe_max_rows),
                    "--min_rows", str(args.probe_min_rows),
                    "--min_prefix_semantic_r2", str(args.probe_min_prefix_semantic_r2),
                    "--min_prefix_mean_lift", str(args.probe_min_prefix_mean_lift),
                    "--min_retrieval_above_chance", str(args.probe_min_retrieval_above_chance),
                ],
                expected=probe_report,
            )

        # Focused Stage-1 graph-causality re-test: after training Stage 1 (with
        # whatever forcing --prefix_gate_floor/--stage1_dependence_weight supply),
        # run ONLY the causality gate on Stage 1 and stop, skipping the control,
        # hierarchy, and long stages. Run non-fatally so the report is captured
        # even when causality is not significant (the gate exits 1 but writes the
        # report first). Use --limit_tasks to keep it fast.
        if args.causality_probe_only:
            probe_gate_dir = self.output / "02b_forced_causality_probe"
            probe_gate_report = probe_gate_dir / "report.json"
            probe_cmd = self.functional_gate_command(
                dataset=neutral_eval,
                checkpoint=stage1_checkpoint,
                output_dir=probe_gate_dir,
                report=probe_gate_report,
                performance_prompt_mode="graph_only",
                min_permutation_drop_pp=args.stage1_min_permutation_drop_pp,
                min_facts_drop_pp=args.stage1_min_facts_drop_pp,
                min_improvement_pp=0.0,
                run_deployment_arm=False,
                run_causality_arm=True,
            )
            print(f"\n[02b_forced_causality_probe] {display_command(probe_cmd)}")
            if not args.dry_run:
                probe_gate_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(probe_cmd, cwd=self.root, env=self.base_env.copy())
            print(f"\nForced-causality probe report: {probe_gate_report}")
            return 0

        # Run the matched direct-code control regardless of whether the latent
        # graph channel later passes its causal-use gate. This independently
        # measures how much of the historical >=200 cliff came from target
        # truncation and insufficient generation budgets.
        hierarchy_control_dir = self.output / "02a0_matched_direct_control"
        hierarchy_control_checkpoint = self.sft_stage(
            stage="02a0_matched_direct_control",
            train_file=hierarchy_control_train,
            eval_file=hierarchy_control_dev,
            output_dir=hierarchy_control_dir,
            checkpoint=stage1_checkpoint,
            prompt_mode="full",
            binary_evidence=True,
            dependence_weight=0.0,
            dependence_every=1,
            learning_rate=args.hierarchical_lr,
            epochs=args.hierarchical_epochs,
            max_target_length=args.long_target_max_tokens,
            max_prompt_length=args.long_prompt_max_tokens,
        )

        control_recovery_dir = self.output / "02a1_matched_control_recovery"
        control_recovery_checkpoint = self.sft_stage(
            stage="02a1_matched_control_recovery",
            train_file=recovery_train,
            eval_file=recovery_dev,
            output_dir=control_recovery_dir,
            checkpoint=hierarchy_control_checkpoint,
            prompt_mode="full",
            binary_evidence=True,
            dependence_weight=0.0,
            dependence_every=1,
            learning_rate=args.recovery_lr,
            epochs=args.recovery_epochs,
            max_target_length=args.long_target_max_tokens,
            max_prompt_length=args.long_prompt_max_tokens,
        )

        direct_control_eval_dir = self.output / "02a2_direct_control_long_eval"
        direct_control_eval_report = direct_control_eval_dir / "report.json"
        self.run(
            "02a2_direct_control_long_eval",
            self.functional_gate_command(
                dataset=long_dev,
                checkpoint=control_recovery_checkpoint,
                output_dir=direct_control_eval_dir,
                report=direct_control_eval_report,
                performance_prompt_mode="full",
                min_permutation_drop_pp=0.0,
                min_facts_drop_pp=0.0,
                min_improvement_pp=0.0,
                run_deployment_arm=True,
                run_causality_arm=False,
                min_rows=args.min_long_gate_rows,
            ),
            expected=direct_control_eval_report,
        )

        stage1_gate_dir = self.output / "02b_graph_only_functional_gate"
        stage1_gate_report = stage1_gate_dir / "report.json"
        self.run(
            "02b_graph_only_functional_gate",
            self.functional_gate_command(
                dataset=neutral_eval,
                checkpoint=stage1_checkpoint,
                output_dir=stage1_gate_dir,
                report=stage1_gate_report,
                performance_prompt_mode="graph_only",
                min_permutation_drop_pp=args.stage1_min_permutation_drop_pp,
                min_facts_drop_pp=args.stage1_min_facts_drop_pp,
                min_improvement_pp=0.0,
                run_deployment_arm=False,
                run_causality_arm=True,
            ),
            expected=stage1_gate_report,
        )

        hierarchical_dir = self.output / "02c_hierarchical_region_sft"
        hierarchical_checkpoint = self.sft_stage(
            stage="02c_hierarchical_region_sft",
            train_file=hierarchy_train,
            eval_file=hierarchy_dev,
            output_dir=hierarchical_dir,
            checkpoint=stage1_checkpoint,
            prompt_mode="full",
            binary_evidence=True,
            dependence_weight=0.0,
            dependence_every=1,
            learning_rate=args.hierarchical_lr,
            epochs=args.hierarchical_epochs,
            max_target_length=args.long_target_max_tokens,
            max_prompt_length=args.long_prompt_max_tokens,
        )

        recovery_dir = self.output / "02d_all_length_code_recovery"
        curriculum_checkpoint = self.sft_stage(
            stage="02d_all_length_code_recovery",
            train_file=recovery_train,
            eval_file=recovery_dev,
            output_dir=recovery_dir,
            checkpoint=hierarchical_checkpoint,
            prompt_mode="full",
            binary_evidence=True,
            dependence_weight=0.0,
            dependence_every=1,
            learning_rate=args.recovery_lr,
            epochs=args.recovery_epochs,
            max_target_length=args.long_target_max_tokens,
            max_prompt_length=args.long_prompt_max_tokens,
        )

        long_gate_dir = self.output / "02e_long_hierarchy_functional_gate"
        long_gate_report = long_gate_dir / "report.json"
        self.run(
            "02e_long_hierarchy_functional_gate",
            self.functional_gate_command(
                dataset=long_dev,
                checkpoint=curriculum_checkpoint,
                output_dir=long_gate_dir,
                report=long_gate_report,
                performance_prompt_mode="full",
                min_permutation_drop_pp=args.stage2_min_permutation_drop_pp,
                min_facts_drop_pp=args.stage2_min_facts_drop_pp,
                min_improvement_pp=args.hierarchical_long_min_improvement_pp,
                baseline_checkpoint=control_recovery_checkpoint,
                run_deployment_arm=True,
                min_rows=args.min_long_gate_rows,
            ),
            expected=long_gate_report,
        )

        predictions = Path(args.predictions).expanduser().resolve() if args.predictions else self.output / "03_train_rollouts" / "predictions.json"
        if args.predictions:
            self.record_reuse("03_train_rollouts", predictions, "supplied rollout pool")
        else:
            infer_env = {
                "GRAPH_PROMPT_ASSEMBLY_MODE": "full",
                "GRAPH_PROMPT_BINARY_EVIDENCE": "1",
                "GRAPH_REGION_COMPRESSION": args.region_compression,
                "GRAPH_REGION_MAX_BLOCKS": str(args.region_max_blocks),
                "GRAPH_BLOCK_POOLING": args.block_pooling,
                "GRAPH_BLOCK_VECTORS_PER_BLOCK": str(args.block_vectors_per_block),
                "GRAPH_QWEN_PREFIX_TOKENS": str(args.long_prefix_tokens),
                "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
                "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
                "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "1",
                "GRAPH_NEUTRAL_FUNCTION_NAME": "fn0",
                "GRAPH_LOAD_4BIT": str(int(args.load_4bit)),
                "GRAPH_MAX_TARGET_LENGTH": str(args.long_target_max_tokens),
                "GRAPH_DECODER_PROMPT_MAX_LENGTH": str(args.long_prompt_max_tokens),
            }
            infer_cmd = [
                self.python,
                str(self.root / "scripts/evaluation/graph_inference_antigravity.py"),
                "--dataset", str(approved_all),
                "--decoder_model", args.decoder_model,
                "--checkpoint", str(curriculum_checkpoint),
                "--output", str(predictions),
                "--num_samples", str(args.rollout_samples),
                "--generation_batch_size", str(args.rollout_generation_batch_size),
                "--max_new_tokens", str(max(args.rollout_max_new_tokens, args.long_generation_max_tokens)),
                "--decoder_prompt_max_length", str(args.long_prompt_max_tokens),
                "--decoder_revision", args.decoder_revision,
                "--encoder_revision", args.encoder_revision,
                "--seed", str(args.seed),
                "--graph_input_ablation", "none",
            ]
            if args.limit_tasks:
                infer_cmd += ["--limit", str(args.limit_tasks)]
            self.run("03_train_rollouts", infer_cmd, infer_env, predictions)

        teacher_dir = self.output / "04_teacher_harvest"
        collected = teacher_dir / "collected.jsonl"
        collect_cmd = [
            self.python,
            str(self.root / "scripts/training/teacher_repair_dataset_antigravity.py"),
            "collect",
            "--dataset", str(approved_all),
            "--predictions", str(predictions),
            "--prediction_provenance", str(predictions) + ".provenance.json",
            "--expected_checkpoint", str(curriculum_checkpoint),
            "--phase0_report", str(phase0_report),
            "--data_role", args.data_role,
            "--out", str(collected),
            "--timeout", str(args.test_timeout),
            "--workers", str(args.verifier_workers),
            "--max_failures_per_task", str(args.max_failures_per_task),
            "--max_positives_per_task", str(args.max_positives_per_task),
        ]
        if args.limit_candidates:
            collect_cmd += ["--limit_candidates", str(args.limit_candidates)]
        self.run("04_collect_verifier_feedback", collect_cmd, expected=collected)

        teacher_responses = Path(args.teacher_responses).expanduser().resolve() if args.teacher_responses else teacher_dir / (
            "teacher_responses.jsonl" if args.teacher_mode == "sync" else "teacher_batch_requests.jsonl"
        )
        if args.teacher_responses:
            self.record_reuse("05_frontier_teacher", teacher_responses, "supplied completed teacher responses")
        else:
            if not args.teacher_model:
                raise SystemExit("--teacher_model is required unless --teacher_responses is supplied")
            teacher_cmd = [
                self.python,
                str(self.root / "scripts/training/teacher_repair_dataset_antigravity.py"),
                "teacher",
                "--collected", str(collected),
                "--out", str(teacher_responses),
                "--mode", args.teacher_mode,
                "--model", args.teacher_model,
                "--test_visibility", args.teacher_test_visibility,
                "--max_output_tokens", str(args.teacher_max_output_tokens),
                "--concurrency", str(args.teacher_concurrency),
            ]
            if args.teacher_base_url:
                teacher_cmd += ["--base_url", args.teacher_base_url]
            if args.teacher_limit:
                teacher_cmd += ["--limit", str(args.teacher_limit)]
            self.run("05_frontier_teacher", teacher_cmd, expected=teacher_responses)

        artifacts: dict[str, Path | None] = {
            "phase0_all_length_train": approved_all,
            "phase0_all_length_dev": approved_dev,
            "short_training_stratum": short_train,
            "bridge_training_stratum": bridge_train,
            "long_training_stratum": long_train,
            "direct_all_length_train": direct_train,
            "hierarchical_multitask_train": hierarchy_train,
            "matched_direct_control_train": hierarchy_control_train,
            "code_recovery_train": recovery_train,
            "long_development_stratum": long_dev,
            "phase0_report": phase0_report,
            "raw_token_length_distribution": raw_token_length_report,
            "token_length_distribution": token_length_report,
            "neutral_eval": neutral_eval,
            "probe_report": probe_report if args.run_probe else None,
            "stage1_checkpoint": stage1_checkpoint,
            "stage1_functional_gate": stage1_gate_report,
            "direct_control_long_evaluation": direct_control_eval_report,
            "hierarchical_checkpoint": hierarchical_checkpoint,
            "matched_direct_control_checkpoint": hierarchy_control_checkpoint,
            "matched_control_recovery_checkpoint": control_recovery_checkpoint,
            "all_length_recovery_checkpoint": curriculum_checkpoint,
            "long_hierarchy_functional_gate": long_gate_report,
            "rollouts": predictions,
            "collected": collected,
            "teacher_responses": teacher_responses,
        }
        if args.teacher_mode == "batch" and not args.teacher_responses and not args.dry_run:
            manifest = self.write_manifest("awaiting_batch_responses", artifacts)
            print(
                "Batch request JSONL was produced. Submit it, download completed responses, "
                f"then rerun with --teacher_responses. Manifest: {manifest}"
            )
            return 2

        rs_sft = teacher_dir / "verified_rs_sft.jsonl"
        preferences = teacher_dir / "preferences.jsonl"
        build_report = teacher_dir / "verified_build_report.json"
        build_cmd = [
            self.python,
            str(self.root / "scripts/training/teacher_repair_dataset_antigravity.py"),
            "build",
            "--collected", str(collected),
            "--phase0_report", str(phase0_report),
            "--teacher_responses", str(teacher_responses),
            "--data_role", args.data_role,
            "--out_sft", str(rs_sft),
            "--out_preferences", str(preferences),
            "--report", str(build_report),
            "--timeout", str(args.test_timeout),
            "--workers", str(args.verifier_workers),
            "--facts_gate_mode", args.facts_gate_mode,
            "--teacher_test_visibility", args.teacher_test_visibility,
            "--min_verified_rows", str(args.min_verified_rows),
            "--min_verified_unique_tasks", str(args.min_verified_unique_tasks),
            "--min_verified_length_bins", str(args.min_verified_length_bins),
        ]
        self.run("06_build_verified_rs_sft", build_cmd, expected=rs_sft)

        verified_audit = teacher_dir / "verified_rs_sft_audit.json"
        self.run(
            "06b_verified_rs_sft_audit",
            [
                self.python,
                str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                "--dataset", str(rs_sft),
                "--test_fields", "feedback_tests,acceptance_tests,tests",
                "--run_references", "-1",
                "--workers", str(args.verifier_workers),
                "--timeout", str(args.test_timeout),
                "--require_phase0_approved",
                "--require_neutral_contract",
                "--require_verified_origin",
                "--report", str(verified_audit),
            ],
            expected=verified_audit,
        )

        mix_dir = self.output / "07_balanced_rs_sft"
        balanced_mix = mix_dir / "balanced_50_50.jsonl"
        gold_control = mix_dir / "gold_only_matched_steps.jsonl"
        mix_report = mix_dir / "balanced_50_50_report.json"
        self.run(
            "07_build_balanced_sft_mix",
            [
                self.python,
                str(self.root / "scripts/training/build_balanced_sft_mix_antigravity.py"),
                "--gold", str(approved_all),
                "--verified", str(rs_sft),
                "--out", str(balanced_mix),
                "--gold_control_out", str(gold_control),
                "--report", str(mix_report),
                "--seed", str(args.seed),
                "--min_verified_rows", str(args.min_verified_rows),
                "--min_verified_unique_tasks", str(args.min_verified_unique_tasks),
                "--min_verified_length_bins", str(args.min_verified_length_bins),
                "--max_verified_oversample_factor", str(args.max_verified_oversample_factor),
            ],
            expected=balanced_mix,
        )

        balanced_audit = mix_dir / "balanced_mix_audit.json"
        self.run(
            "07b_balanced_mix_audit",
            [
                self.python,
                str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                "--dataset", str(balanced_mix),
                "--test_fields", "feedback_tests,acceptance_tests,tests",
                "--run_references", "-1",
                "--workers", str(args.verifier_workers),
                "--timeout", str(args.test_timeout),
                "--require_phase0_approved",
                "--require_neutral_contract",
                "--report", str(balanced_audit),
            ],
            expected=balanced_audit,
        )

        # A +6 pp RS-SFT claim requires a like-for-like baseline: a
        # matched-step, matched-modality control. A graph-only Stage-1 score is
        # modality-mismatched and cannot serve as the full-prompt RS-SFT baseline.
        # Unless an external frozen baseline is supplied, train gold-only control
        # from the same Stage-1 checkpoint with the same full prompt, LR, epochs,
        # and exact number of examples as the 50/50 run.
        gold_control_checkpoint: Path | None = None
        needs_gold_control = not args.rs_sft_baseline_checkpoint
        if needs_gold_control:
            control_dir = self.output / "08a_gold_only_full_sft_control"
            gold_control_checkpoint = self.sft_stage(
                stage="08a_gold_only_full_sft_control",
                train_file=gold_control,
                eval_file=approved_dev,
                output_dir=control_dir,
                checkpoint=curriculum_checkpoint,
                prompt_mode="full",
                binary_evidence=True,
                dependence_weight=0.0,
                dependence_every=1,
                learning_rate=args.stage2_lr,
                epochs=args.stage2_epochs,
                max_target_length=args.long_target_max_tokens,
                max_prompt_length=args.long_prompt_max_tokens,
            )

        stage2_dir = self.output / "08b_rejection_sampling_sft"
        stage2_checkpoint = self.sft_stage(
            stage="08b_rejection_sampling_sft",
            train_file=balanced_mix,
            eval_file=approved_dev,
            output_dir=stage2_dir,
            checkpoint=curriculum_checkpoint,
            prompt_mode="full",
            binary_evidence=True,
            dependence_weight=0.0,  # full-text wrong-graph margin is confounded
            dependence_every=1,
            learning_rate=args.stage2_lr,
            epochs=args.stage2_epochs,
            max_target_length=args.long_target_max_tokens,
            max_prompt_length=args.long_prompt_max_tokens,
        )

        stage2_gate_dir = self.output / "09_rs_sft_functional_kill_switch"
        stage2_gate_report = stage2_gate_dir / "report.json"
        baseline_checkpoint: Path | None = None
        if args.rs_sft_baseline_checkpoint:
            baseline_checkpoint = Path(
                args.rs_sft_baseline_checkpoint
            ).expanduser().resolve()
            if not args.dry_run:
                file_record(baseline_checkpoint)
        else:
            if gold_control_checkpoint is None:
                raise RuntimeError(
                    "internal error: no external baseline and no matched-step gold control"
                )
            baseline_checkpoint = gold_control_checkpoint
        self.run(
            "09_rs_sft_functional_kill_switch",
            self.functional_gate_command(
                dataset=neutral_eval,
                checkpoint=stage2_checkpoint,
                output_dir=stage2_gate_dir,
                report=stage2_gate_report,
                performance_prompt_mode="full",
                min_permutation_drop_pp=args.stage2_min_permutation_drop_pp,
                min_facts_drop_pp=args.stage2_min_facts_drop_pp,
                min_improvement_pp=args.rs_sft_min_improvement_pp,
                baseline_checkpoint=baseline_checkpoint,
                require_paired_baseline=True,
            ),
            expected=stage2_gate_report,
        )

        final_checkpoint = stage2_checkpoint
        final_gate_report: Path | None = stage2_gate_report
        if args.run_grpo:
            grpo_dir = self.output / "10_verpo"
            grpo_env = {
                "GRPO_TRAIN_FILE": str(approved_all),
                "GRPO_VERIFIED_ANCHOR_FILE": str(rs_sft),
                "GRAPH_CHECKPOINT": str(stage2_checkpoint),
                "GRAPH_OUTPUT_DIR": str(grpo_dir),
                "GRAPH_PROMPT_ASSEMBLY_MODE": "full",
                "GRAPH_PROMPT_BINARY_EVIDENCE": "1",
                "GRAPH_REQUIRE_PHASE0_APPROVED": "1",
                "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "1",
                "GRAPH_NEUTRAL_FUNCTION_NAME": "fn0",
                # Stage 1 and RS-SFT are trained with FACTS_JSON + DART targets.
                # Keep the verified-only anchor in exactly the same target
                # schema; otherwise its CE term would pull the policy back
                # toward plain source-only formatting.
                "GRAPH_FACTS_FIRST_TARGET": "1",
                "GRAPH_LOAD_4BIT": str(int(args.load_4bit)),
                "GRAPH_MAX_TARGET_LENGTH": str(args.long_target_max_tokens),
                "GRAPH_DECODER_PROMPT_MAX_LENGTH": str(args.long_prompt_max_tokens),
                "GRAPH_REGION_COMPRESSION": args.region_compression,
                "GRAPH_REGION_MAX_BLOCKS": str(args.region_max_blocks),
                "GRAPH_BLOCK_POOLING": args.block_pooling,
                "GRAPH_BLOCK_VECTORS_PER_BLOCK": str(args.block_vectors_per_block),
                "GRAPH_QWEN_PREFIX_TOKENS": str(args.long_prefix_tokens),
                "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
                "GRAPH_QWEN_PREFIX_MIN_TOKENS": str(args.long_prefix_min_tokens),
                "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": str(args.long_prefix_tokens_per_log2),
                "GRAPH_QWEN_PREFIX_RMS_MATCH": "1",
                "GRAPH_ADD_REVERSE_EDGES": "1",
                "GRAPH_BLOCK_POSITION_MODE": "sinusoidal",
                "GRAPH_DFG_MODE": "edges",
                "GRAPH_POSITION_SCHEME": "roberta",
                "GRAPH_BATCH_SIZE": str(args.grpo_batch_size),
                "GRAPH_GRAD_ACCUM": str(args.grpo_grad_accum),
                "GRPO_REWARD_MODE": args.grpo_reward_mode,
                # --- VeRPO hybrid feedback (compiler + LLM) ---
                # V1 reward densifier: LLM judge scores functional progress on
                # compiling-but-failing candidates -> within-group variance where
                # the compiler reward is flat. V2 repair: prompt carries the prior
                # attempt + feedback (build_decoder_prompt) so the student reads
                # what broke. Judge auth flows via OPENAI_API_KEY in base_env.
                "GRPO_VERPO_JUDGE": "1" if args.verpo_judge else "0",
                "GRPO_VERPO_JUDGE_WEIGHT": str(args.verpo_judge_weight),
                "GRPO_VERPO_FULL_PASS_MARGIN": str(args.verpo_full_pass_margin),
                "GRPO_VERPO_REPAIR": "1" if args.verpo_repair else "0",
                "VERPO_JUDGE_MODEL": args.verpo_judge_model,
                "VERPO_JUDGE_BASE_URL": args.verpo_judge_base_url,
                "VERPO_JUDGE_CONCURRENCY": str(args.verpo_judge_concurrency),
                "VERPO_JUDGE_TIMEOUT_SECONDS": str(args.verpo_judge_timeout_seconds),
                "VERPO_JUDGE_MAX_RETRIES": str(args.verpo_judge_max_retries),
                "VERPO_JUDGE_FAIL_CLOSED": "1",
                "VERPO_JUDGE_REQUIRE_SUCCESS": "1",
                "GRPO_REWARD_TEST_FIELD": "feedback_tests",
                "GRPO_GROUP_SIZE": str(args.grpo_group_size),
                "GRPO_EPOCHS": str(args.grpo_epochs),
                "GRPO_LR": str(args.grpo_lr),
                "GRPO_TEST_TIMEOUT": str(args.test_timeout),
                "GRPO_MAX_NEW_TOKENS": str(max(args.grpo_max_new_tokens, args.long_generation_max_tokens)),
                "GRPO_REWARD_WORKERS": str(args.verifier_workers),
                "GRPO_ADV_NORM": "mean",
                "GRPO_MIN_REWARD_RANGE": str(args.grpo_min_reward_range),
                "GRPO_PASSK_K": str(args.grpo_passk_k),
                "GRPO_LOSS_POOLING": "seq",
                "GRPO_SCORE_CHUNK_SIZE": str(args.grpo_score_chunk_size),
                "GRPO_GENERATION_CHUNK_SIZE": str(args.grpo_generation_chunk_size),
                # Scoring uses the full temperature-scaled softmax, so
                # rollout sampling must not apply nucleus truncation.
                "GRPO_GEN_TOP_P": "1.0",
                "GRPO_TRAIN_GRAPH_GLUE": "1",
                "GRPO_KL_COEF": "0.0",
                "GRPO_SFT_ANCHOR_COEF": str(args.grpo_sft_anchor_coef),
                "GRPO_SFT_ANCHOR_ON_NO_SIGNAL": "0",
                "GRPO_DYNAMIC_RESAMPLE_ATTEMPTS": str(args.grpo_dynamic_resample_attempts),
                "GRAPH_SAVE_STRATEGY": "steps",
                "GRAPH_SAVE_STEPS": str(args.grpo_save_steps),
                "GRAPH_SAVE_TOTAL_LIMIT": "2",
                "GRAPH_MAX_STEPS": str(args.max_steps),
            }
            if args.verpo_repair:
                grpo_env["GRPO_TRAIN_FILE"] = str(
                    Path(args.verpo_repair_file).expanduser().resolve()
                )
            final_checkpoint = checkpoint_path(grpo_dir)
            self.run(
                "10_verpo",
                [self.python, "-m", "scripts.training.graph_grpo_decompiler_antigravity"],
                grpo_env,
                final_checkpoint,
            )

            final_gate_dir = self.output / "11_final_functional_gate"
            final_gate_report = final_gate_dir / "report.json"
            self.run(
                "11_final_functional_gate",
                self.functional_gate_command(
                    dataset=neutral_eval,
                    checkpoint=final_checkpoint,
                    output_dir=final_gate_dir,
                    report=final_gate_report,
                    performance_prompt_mode="full",
                    min_permutation_drop_pp=args.stage2_min_permutation_drop_pp,
                    min_facts_drop_pp=args.stage2_min_facts_drop_pp,
                    min_improvement_pp=args.final_min_improvement_pp,
                    baseline_checkpoint=stage2_checkpoint,
                ),
                expected=final_gate_report,
            )

        artifacts.update(
            {
                "verified_rs_sft": rs_sft,
                "preferences": preferences,
                "verified_build_report": build_report,
                "balanced_mix": balanced_mix,
                "gold_only_matched_step_control": gold_control,
                "gold_only_control_checkpoint": gold_control_checkpoint,
                "stage2_checkpoint": stage2_checkpoint,
                "stage2_functional_gate": stage2_gate_report,
                "final_checkpoint": final_checkpoint,
                "final_functional_gate": final_gate_report,
            }
        )
        manifest = self.write_manifest("dry_run" if args.dry_run else "completed", artifacts)
        print(f"\nFinal checkpoint: {final_checkpoint}")
        print(f"Manifest: {manifest}")
        print(f"Bridge training stratum 151-{args.max_bridge_instructions}: {bridge_train}")
        print(f"Long training stratum >{args.max_bridge_instructions}: {long_train}")
        print("All approved long functions participated in direct SFT, hierarchical region supervision, and code-only recovery.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True, help="Frozen evaluation file; never used as SFT data")
    parser.add_argument("--functional_eval_file", default="", help="Held-out gate file; defaults to --eval_file")
    parser.add_argument("--frozen_eval_file", action="append", default=[], help="Additional frozen eval/blocklist file")
    parser.add_argument("--initial_checkpoint", default="")
    parser.add_argument("--probe_checkpoint", default="")
    parser.add_argument("--stage1_checkpoint", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--teacher_responses", default="")
    parser.add_argument(
        "--architecture_env_json",
        default="",
        help=(
            "run_provenance.json or prediction provenance containing the exact "
            "graph_environment used by a warm checkpoint. Adjacent "
            "checkpoint-parent/run_provenance.json is auto-discovered."
        ),
    )
    parser.add_argument(
        "--allow_unpinned_architecture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Research-only: allow a warm checkpoint without complete graph_environment provenance.",
    )

    parser.add_argument("--decoder_model", default="")
    parser.add_argument("--encoder_model", default="")
    parser.add_argument("--decoder_revision", default="")
    parser.add_argument("--encoder_revision", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_4bit", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--qwen_lora_targets", choices=["attention", "attention_mlp"],
        default="attention_mlp",
        help="Qwen decoder LoRA target set. New modality-alignment runs default to attention+MLP.",
    )
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--data_role", choices=["train", "development", "dev"], default="train")
    parser.add_argument("--feedback_fraction", type=float, default=0.60)
    parser.add_argument("--dev_fraction", type=float, default=0.10)
    parser.add_argument("--min_dev_rows", type=int, default=32)
    parser.add_argument("--max_train_instructions", type=int, default=150)
    parser.add_argument("--max_bridge_instructions", type=int, default=199)
    parser.add_argument("--min_short_rows", type=int, default=64)
    parser.add_argument("--min_long_rows", type=int, default=64,
                        help="Minimum >=200-instruction rows retained for supervised training")
    parser.add_argument("--min_gate_rows", type=int, default=96)
    parser.add_argument("--min_long_gate_rows", type=int, default=64)
    parser.add_argument("--test_timeout", type=int, default=20)
    parser.add_argument("--verifier_workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 1)))

    parser.add_argument("--run_probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe_max_rows", type=int, default=256)
    parser.add_argument("--probe_min_rows", type=int, default=32)
    parser.add_argument("--probe_min_prefix_semantic_r2", type=float, default=-1.0)
    parser.add_argument("--probe_min_prefix_mean_lift", type=float, default=0.02)
    parser.add_argument("--probe_min_retrieval_above_chance", type=float, default=0.0)

    parser.add_argument("--stage1_dependence_weight", type=float, default=0.25)
    parser.add_argument("--stage1_dependence_every", type=int, default=1)
    parser.add_argument("--prefix_dependence_margin", type=float, default=0.15)
    parser.add_argument("--prefix_negative_bank_size", type=int, default=32)
    parser.add_argument("--stage1_lr", type=float, default=5e-6)
    parser.add_argument("--stage1_epochs", type=float, default=1.0)
    parser.add_argument("--sft_batch_size", type=int, default=1)
    parser.add_argument("--sft_grad_accum", type=int, default=32)

    parser.add_argument("--hierarchical_lr", type=float, default=3e-6)
    parser.add_argument("--hierarchical_epochs", type=float, default=1.0)
    parser.add_argument("--recovery_lr", type=float, default=2e-6)
    parser.add_argument("--recovery_epochs", type=float, default=1.0)
    parser.add_argument("--hierarchical_short_replay_fraction", type=float, default=0.33)
    parser.add_argument("--recovery_short_repeat", type=int, default=1)
    parser.add_argument("--recovery_bridge_repeat", type=int, default=2)
    parser.add_argument("--recovery_long_repeat", type=int, default=3)
    parser.add_argument("--region_compression", choices=["linear_residual"], default="linear_residual")
    parser.add_argument("--region_max_blocks", type=int, default=8)
    parser.add_argument("--block_pooling", choices=["multi_query"], default="multi_query")
    parser.add_argument("--block_vectors_per_block", type=int, default=4)
    parser.add_argument("--long_prefix_tokens", type=int, default=64)
    parser.add_argument("--long_prefix_min_tokens", type=int, default=16)
    parser.add_argument("--long_prefix_tokens_per_log2", type=int, default=8)
    parser.add_argument("--long_target_max_tokens", type=int, default=3072)
    parser.add_argument("--long_prompt_max_tokens", type=int, default=3072)
    parser.add_argument("--long_generation_max_tokens", type=int, default=3072)

    parser.add_argument("--gate_num_samples", type=int, default=10)
    parser.add_argument("--gate_k", type=int, default=10)
    parser.add_argument("--gate_generation_batch_size", type=int, default=4)
    parser.add_argument("--gate_max_new_tokens", type=int, default=3072)
    parser.add_argument(
        "--decoder_prompt_max_length", type=int, default=0,
        help="0 inherits GRAPH_DECODER_PROMPT_MAX_LENGTH from architecture provenance (else 768).",
    )
    parser.add_argument(
        "--stage1_min_permutation_drop_pp", type=float, default=0.0,
        help=(
            "Optional pre-registered practical-effect floor for graph permutation. "
            "Default 0 relies on the exact paired test and bootstrap lower bound; "
            "set a positive value only from an external repeatability/noise study."
        ),
    )
    parser.add_argument("--stage1_min_facts_drop_pp", type=float, default=0.0)
    parser.add_argument(
        "--stage2_min_permutation_drop_pp", type=float, default=0.0,
        help=(
            "Optional pre-registered Stage-2 graph-causality effect floor. "
            "The exact paired test and bootstrap lower bound remain mandatory."
        ),
    )
    parser.add_argument("--stage2_min_facts_drop_pp", type=float, default=0.0)
    parser.add_argument("--gate_bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--gate_statistical_confidence", type=float, default=0.95)
    parser.add_argument("--gate_max_sign_test_p_value", type=float, default=0.05)
    parser.add_argument("--gate_min_causal_effective_pairs", type=int, default=8)
    parser.add_argument("--gate_min_deployment_effective_pairs", type=int, default=8)
    parser.add_argument("--gate_min_permutation_ci_lower_pp", type=float, default=0.0)
    parser.add_argument("--gate_min_null_ci_lower_pp", type=float, default=0.0)
    parser.add_argument("--gate_min_facts_ci_lower_pp", type=float, default=0.0)
    parser.add_argument("--gate_min_deployment_ci_lower_pp", type=float, default=0.0)
    parser.add_argument(
        "--gate_require_facts_statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--gate_min_causal_task_losses",
        type=int,
        default=0,
        help="Deprecated extra count floor; exact sign test and paired bootstrap are authoritative.",
    )
    parser.add_argument("--hierarchical_long_min_improvement_pp", type=float, default=0.0,
                        help="Minimum paired pass@k improvement over the matched-example/step direct-SFT control on >=200 rows")
    parser.add_argument("--rs_sft_min_improvement_pp", type=float, default=6.0)
    parser.add_argument("--final_min_improvement_pp", type=float, default=0.0)
    parser.add_argument(
        "--baseline_pass_at_k",
        type=float,
        default=-1.0,
        help=(
            "Deprecated for curriculum runs. Aggregate baselines do not support "
            "paired RS-SFT inference; let the runner train/signature-resume its "
            "matched gold-only control."
        ),
    )
    parser.add_argument(
        "--rs_sft_baseline_checkpoint",
        default="",
        help=(
            "Deprecated fail-closed option. The runner now trains (or signature-resumes) "
            "its own matched-step gold-only control because an arbitrary external "
            "checkpoint cannot prove the matched training intervention."
        ),
    )
    parser.add_argument("--limit_tasks", type=int, default=0)
    # Graph-forcing knobs for the Stage-1 causality re-test. Defaults (0.0)
    # preserve the shipped weak-forcing behaviour; raising them forces the graph
    # prefix gate open (floor) and penalises graph-invariant output (via the
    # existing --stage1_dependence_weight counterfactual hinge).
    parser.add_argument("--prefix_gate_floor", type=float, default=0.0,
                        help="Force the graph-prefix gate >= this floor (0.0 = off)")
    parser.add_argument("--prefix_gate_floor_weight", type=float, default=0.0,
                        help="Penalty weight for the prefix-gate floor (0.0 = off)")
    parser.add_argument("--causality_probe_only", action="store_true",
                        help="Train Stage 1 (+probe), run ONLY the graph causality gate, then stop")
    parser.add_argument("--eval_only_dataset", default="",
                        help="Standalone gate-eval mode: dataset JSONL to evaluate (skips all training)")
    parser.add_argument("--eval_only_checkpoint", default="",
                        help="Checkpoint .bin to evaluate in eval-only mode")
    parser.add_argument("--eval_only_output", default="eval_only_gate",
                        help="Output subdirectory name under output_root for the eval-only gate")
    parser.add_argument("--text_only", action="store_true",
                        help="Clean text-only arm: one text SFT (full prompt) then stages 02e/03/04; "
                             "skips graph-only 01, the 02a control chain, the dead 02b gate, and 02c")
    parser.add_argument("--text_eval_file", default="",
                        help="text_only: explicit >=200 gate dataset (e.g. sealed 326 dev) so the "
                             "text arm scores the SAME rows as the compact arm; must be evaluation_only")
    parser.add_argument("--text_sft_checkpoint", default="",
                        help="text_only: skip 02t SFT and use this existing checkpoint (guaranteed "
                             "reuse when Phase-0 reruns non-idempotently); runs only gate/03/04")
    parser.add_argument("--text_finish_rs_sft", action="store_true",
                        help="text_only: skip 02te/03/04 and run 07 balanced mix -> 08b RS-SFT retrain "
                             "-> 09 functional gate, using --rs_sft_file as the verified curriculum")
    parser.add_argument("--rs_sft_file", default="",
                        help="text_finish_rs_sft: path to the verified RS-SFT curriculum JSONL "
                             "(default: <output>/04_teacher_harvest/verified_rs_sft.jsonl)")
    parser.add_argument("--rs_sft_rows_per_epoch", type=int, default=0,
                        help="text_finish_rs_sft: 07 mix epoch size (0 = build default 2*len(gold)); "
                             "use 2*4*n_verified to keep the 4x oversample cap when verified is scarce")
    parser.add_argument("--rs_sft_allow_partial_gold", action="store_true",
                        help="text_finish_rs_sft: allow 07 mix to subsample gold (verified-scarce 50/50)")
    parser.add_argument("--text_verpo", action="store_true",
                        help="skip to 10_verpo from an existing RS-SFT checkpoint; preserves the "
                             "checkpoint's graph-prefix architecture even though graph glue is frozen")
    parser.add_argument("--verpo_checkpoint", default="",
                        help="text_verpo: init checkpoint for GRPO (default: the 08b RS-SFT checkpoint)")
    parser.add_argument("--text_sft_only", action="store_true",
                        help="text_only: train (or reuse) the 02t decoder SFT and stop before 02te/03 "
                             "(use with GRAPH_QWEN_PREFIX_TOKENS=0 for a decoder-only text model)")

    parser.add_argument("--rollout_samples", type=int, default=16)
    parser.add_argument("--rollout_generation_batch_size", type=int, default=4)
    parser.add_argument("--rollout_max_new_tokens", type=int, default=3072)
    parser.add_argument("--limit_candidates", type=int, default=0)
    parser.add_argument("--max_failures_per_task", type=int, default=2)
    parser.add_argument("--max_positives_per_task", type=int, default=3)

    parser.add_argument("--teacher_model", default="")
    parser.add_argument("--teacher_mode", choices=["sync", "batch"], default="sync")
    parser.add_argument("--teacher_test_visibility", choices=["none", "summary", "diagnostics"], default="summary")
    parser.add_argument("--teacher_concurrency", type=int, default=4)
    parser.add_argument("--teacher_max_output_tokens", type=int, default=6000)
    parser.add_argument("--teacher_limit", type=int, default=0)
    parser.add_argument("--teacher_base_url", default="")
    parser.add_argument("--facts_gate_mode", choices=["signature", "conservative", "strict"], default="conservative")
    parser.add_argument(
        "--rs_sft_recertify_facts_gate_mode",
        choices=["signature", "conservative", "strict"],
        default="signature",
        help=(
            "Facts gate used only when re-certifying a supplied legacy RS-SFT "
            "artifact. Signature is the migration default because executable "
            "feedback/acceptance/full suites are replayed and compiler-derived "
            "numeric immediates are not a sound semantic requirement."
        ),
    )
    parser.add_argument("--min_verified_rows", type=int, default=64)
    parser.add_argument("--min_verified_unique_tasks", type=int, default=64)
    parser.add_argument("--min_verified_length_bins", type=int, default=3)
    parser.add_argument("--max_verified_oversample_factor", type=float, default=4.0)

    parser.add_argument("--stage2_lr", type=float, default=3e-6)
    parser.add_argument("--stage2_epochs", type=float, default=1.0)

    parser.add_argument("--run_grpo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grpo_reward_mode", choices=["binary", "shaped", "verpo"], default="verpo")
    # VeRPO hybrid feedback signals (both build on the shaped/verpo compiler reward)
    parser.add_argument(
        "--verpo_judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add the fail-closed compile-gated LLM teacher reward (default: enabled)",
    )
    parser.add_argument("--verpo_judge_weight", type=float, default=0.25,
                        help="maximum additive semantic-progress reward")
    parser.add_argument("--verpo_full_pass_margin", type=float, default=0.001,
                        help="minimum reward margin preserving verifier-confirmed full-pass dominance")
    parser.add_argument("--verpo_repair", action="store_true",
                        help="V2: in-context repair — prompt carries prior attempt + feedback")
    parser.add_argument("--verpo_repair_file", default="",
                        help="V2: repair dataset (prior_attempt+repair_feedback) from build_verpo_repair_dataset")
    parser.add_argument("--verpo_judge_model", default="deepseek-chat")
    parser.add_argument("--verpo_judge_base_url", default="https://api.deepseek.com")
    parser.add_argument("--verpo_judge_concurrency", type=int, default=8)
    parser.add_argument("--verpo_judge_timeout_seconds", type=float, default=60.0)
    parser.add_argument("--verpo_judge_max_retries", type=int, default=2)
    parser.add_argument("--grpo_group_size", type=int, default=8)
    parser.add_argument("--grpo_batch_size", type=int, default=1)
    parser.add_argument("--grpo_grad_accum", type=int, default=8)
    parser.add_argument("--grpo_epochs", type=int, default=1)
    parser.add_argument("--grpo_lr", type=float, default=1e-6)
    parser.add_argument("--grpo_max_new_tokens", type=int, default=3072)
    parser.add_argument("--grpo_min_reward_range", type=float, default=0.0,
                        help="non-paper reward deadband; default 0 preserves every non-uniform group")
    parser.add_argument("--grpo_passk_k", type=int, default=10)
    parser.add_argument("--grpo_score_chunk_size", type=int, default=4)
    parser.add_argument("--grpo_generation_chunk_size", type=int, default=1,
                        help="rollouts generated simultaneously; 1 bounds Qwen KV-cache memory")
    parser.add_argument("--grpo_save_steps", type=int, default=1,
                        help="write bounded recovery checkpoints every N GRPO batches")
    parser.add_argument("--grpo_sft_anchor_coef", type=float, default=0.10)
    parser.add_argument("--grpo_dynamic_resample_attempts", type=int, default=2)

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    positive = (
        "max_train_instructions", "max_bridge_instructions", "min_short_rows", "min_long_rows", "min_dev_rows", "min_gate_rows", "min_long_gate_rows", "test_timeout",
        "verifier_workers", "probe_max_rows", "probe_min_rows", "stage1_dependence_every",
        "prefix_negative_bank_size", "sft_batch_size", "sft_grad_accum",
        "recovery_short_repeat", "recovery_bridge_repeat", "recovery_long_repeat",
        "region_max_blocks", "block_vectors_per_block", "long_prefix_tokens",
        "long_prefix_min_tokens", "long_prefix_tokens_per_log2", "long_target_max_tokens",
        "long_prompt_max_tokens", "long_generation_max_tokens", "gate_num_samples",
        "gate_bootstrap_iterations", "gate_min_causal_effective_pairs",
        "gate_min_deployment_effective_pairs",
        "gate_k", "gate_generation_batch_size", "gate_max_new_tokens", "rollout_samples",
        "rollout_generation_batch_size", "rollout_max_new_tokens", "teacher_concurrency",
        "min_verified_rows", "min_verified_unique_tasks", "min_verified_length_bins",
        "grpo_group_size", "grpo_batch_size", "grpo_grad_accum", "grpo_epochs",
        "grpo_max_new_tokens", "grpo_score_chunk_size",
        "grpo_generation_chunk_size", "grpo_save_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not 0.0 <= args.hierarchical_short_replay_fraction <= 1.0:
        parser.error("--hierarchical_short_replay_fraction must be in [0,1]")
    if args.long_prefix_min_tokens > args.long_prefix_tokens:
        parser.error("--long_prefix_min_tokens cannot exceed --long_prefix_tokens")
    if args.max_bridge_instructions < args.max_train_instructions:
        parser.error("--max_bridge_instructions must be >= --max_train_instructions")
    if not 0.0 < args.dev_fraction < 1.0:
        parser.error("--dev_fraction must be strictly between 0 and 1")
    if args.gate_k > args.gate_num_samples:
        parser.error("--gate_k cannot exceed --gate_num_samples")
    try:
        validate_target_generation_budget(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.gate_bootstrap_iterations < 100:
        parser.error("--gate_bootstrap_iterations must be at least 100")
    if not 0.5 < args.gate_statistical_confidence < 1.0:
        parser.error("--gate_statistical_confidence must be in (0.5,1)")
    if not 0.0 < args.gate_max_sign_test_p_value < 1.0:
        parser.error("--gate_max_sign_test_p_value must be in (0,1)")
    if not 0.0 < args.feedback_fraction < 1.0:
        parser.error("--feedback_fraction must be in (0,1)")
    if args.stage1_dependence_weight < 0.0 or args.grpo_sft_anchor_coef < 0.0:
        parser.error("objective coefficients must be non-negative")
    if args.grpo_dynamic_resample_attempts < 0:
        parser.error("--grpo_dynamic_resample_attempts must be non-negative")
    if args.grpo_min_reward_range < 0.0:
        parser.error("--grpo_min_reward_range must be non-negative")
    if args.verpo_judge:
        if args.verpo_judge_weight <= 0.0:
            parser.error("--verpo_judge_weight must be positive")
        if args.verpo_full_pass_margin <= 0.0:
            parser.error("--verpo_full_pass_margin must be positive")
        if args.verpo_judge_timeout_seconds <= 0.0:
            parser.error("--verpo_judge_timeout_seconds must be positive")
        if args.verpo_judge_concurrency <= 0:
            parser.error("--verpo_judge_concurrency must be positive")
        if args.verpo_judge_max_retries < 0:
            parser.error("--verpo_judge_max_retries must be non-negative")
    if args.verpo_repair and not args.verpo_repair_file:
        parser.error("--verpo_repair requires --verpo_repair_file")
    if args.verpo_repair_file and not args.verpo_repair:
        parser.error("--verpo_repair_file requires --verpo_repair")
    if args.verpo_repair:
        try:
            repair_contract = validate_verpo_repair_dataset(args.verpo_repair_file)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        args.verpo_repair_file = repair_contract["path"]
    if args.gate_min_causal_task_losses < 0:
        parser.error("--gate_min_causal_task_losses must be non-negative")
    if args.baseline_pass_at_k > 1.0:
        parser.error("--baseline_pass_at_k must be <=1 or negative to disable")
    if args.baseline_pass_at_k >= 0.0:
        parser.error(
            "--baseline_pass_at_k is not accepted for RS-SFT curriculum gates; "
            "omit it so the runner trains/signature-resumes its matched gold-only control"
        )
    if args.rs_sft_baseline_checkpoint:
        parser.error(
            "--rs_sft_baseline_checkpoint is not accepted because its matched-step "
            "training contract cannot be verified. Omit it; --resume will reuse the "
            "runner-owned 08a gold-only control only when its stage signature matches."
        )
    runner = CurriculumRunner(args)
    raise SystemExit(runner.execute())


if __name__ == "__main__":
    main()
