#!/usr/bin/env python3
"""Run the fail-closed hybrid decompiler curriculum (v2.1).

The ordering is deliberate:

1. CPU-only leakage, deduplication, neutral-contract, hidden-test, and reference
   parity gates run before any GPU training.
2. A frozen representation probe asks whether facts survive the encoder/GNN/
   projection/prefix path.
3. Graph-only SFT may use the counterfactual NLL objective, but a held-out,
   free-running pass@k permutation gate is the authoritative graph-use test.
4. Frontier repairs receive only redacted feedback tests. Hidden acceptance tests
   and a deterministic FACTS gate decide whether a repair enters training.
5. RS-SFT uses an explicitly materialised 50/50 gold-to-verified sampler and no
   confounded full-text dependence loss.
6. VeRPO is optional and uses dynamic resampling plus a separate verified-only
   anchor loader; ordinary RL-row references are never anchor labels.

Long functions above the Phase-0 instruction cap are retained as a separate
holdout. This curriculum does not claim to solve the >=200-instruction cliff.
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

SCHEMA_VERSION = 3


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


def _checkpoint_provenance_candidates(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.architecture_env_json:
        candidates.append(Path(args.architecture_env_json).expanduser().resolve())
    for raw in (args.probe_checkpoint, args.initial_checkpoint, args.stage1_checkpoint):
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


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def env_snapshot(env: dict[str, str]) -> dict[str, str]:
    prefixes = ("GRAPH_", "GRPO_", "HF_", "CUDA_", "PYTORCH_", "HYBRID_")
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
            value for value in (args.probe_checkpoint, args.initial_checkpoint, args.stage1_checkpoint)
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
        self.base_env["GRAPH_SEED"] = str(args.seed)
        self.base_env.setdefault("GRAPH_QUIET", "1")
        self.base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.history: list[dict[str, Any]] = []

        required = [
            self.root / "scripts/training/hybrid_data_controls.py",
            self.root / "scripts/training/checkpoint_contract.py",
            self.root / "scripts/training/prepare_hybrid_training_data_antigravity.py",
            self.root / "scripts/training/build_balanced_sft_mix_antigravity.py",
            self.root / "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
            self.root / "scripts/training/graph_grpo_decompiler_antigravity.py",
            self.root / "scripts/training/teacher_repair_dataset_antigravity.py",
            self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py",
            self.root / "scripts/evaluation/prepare_neutral_evaluation_antigravity.py",
            self.root / "scripts/evaluation/probe_graph_representations_antigravity.py",
            self.root / "scripts/evaluation/functional_graph_gate_antigravity.py",
            self.root / "scripts/evaluation/graph_inference_antigravity.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Overlay the v2.1 patch into the project root first. Missing:\n  "
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
            if previous.get("stage_signature") == signature:
                print(f"[{stage}] already complete; reusing {expected or done_path}")
                previous["reused"] = True
                self.history.append(previous)
                return
            print(f"[{stage}] previous state differs from current inputs/config; rebuilding")

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
            "GRAPH_PREFIX_GATE_FLOOR": "0.0",
            "GRAPH_PREFIX_GATE_FLOOR_WEIGHT": "0.0",
            "GRAPH_LR": str(learning_rate),
            "GRAPH_EPOCHS": str(epochs),
            "GRAPH_BATCH_SIZE": str(self.args.sft_batch_size),
            "GRAPH_GRAD_ACCUM": str(self.args.sft_grad_accum),
            "GRAPH_LOAD_4BIT": str(int(self.args.load_4bit)),
            "GRAPH_MAX_STEPS": str(self.args.max_steps),
            "GRAPH_QWEN_LORA_TARGETS": self.target_qwen_lora_targets,
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
        run_deployment_arm: bool = True,
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
            "--max_new_tokens", str(self.args.gate_max_new_tokens),
            "--decoder_prompt_max_length", str(self.args.decoder_prompt_max_length),
            "--causality_prompt_mode", "graph_only",
            "--performance_prompt_mode", performance_prompt_mode,
            "--run_deployment_arm" if run_deployment_arm else "--no-run_deployment_arm",
            "--min_rows", str(self.args.min_gate_rows),
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
                "Phase0 -> representation probe -> graph-only SFT -> free-running graph gate -> "
                "frontier repair -> hidden verification -> exact 50/50 RS-SFT + matched-step gold control -> "
                "+6pp kill switch -> optional VeRPO"
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
                "Functions above the Phase-0 instruction cap remain a separate long-function holdout.",
                "No claim is made that one-shot whole-function decoding solves the >=200-instruction cliff.",
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

        initial_checkpoint = Path(args.initial_checkpoint).expanduser().resolve() if args.initial_checkpoint else None
        stage1_supplied = Path(args.stage1_checkpoint).expanduser().resolve() if args.stage1_checkpoint else None
        probe_checkpoint = Path(args.probe_checkpoint).expanduser().resolve() if args.probe_checkpoint else (initial_checkpoint or stage1_supplied)
        if not args.dry_run:
            if initial_checkpoint is not None:
                file_record(initial_checkpoint)
            if stage1_supplied is not None:
                file_record(stage1_supplied)
            if args.run_probe and probe_checkpoint is None:
                raise SystemExit(
                    "The mandatory pre-training representation probe needs --probe_checkpoint, "
                    "--initial_checkpoint, or a supplied --stage1_checkpoint."
                )
        if probe_checkpoint is None:
            # Dry-run placeholder only; the real path is rejected above.
            probe_checkpoint = self.output / "00_phase0" / "PROBE_CHECKPOINT_REQUIRED.pt"

        phase0_dir = self.output / "00_phase0"
        approved_short = phase0_dir / "approved_short_train.jsonl"
        approved_dev = phase0_dir / "approved_short_dev.jsonl"
        bridge_holdout = phase0_dir / "approved_bridge_151_199.jsonl"
        long_holdout = phase0_dir / "approved_long_ge200.jsonl"
        phase0_report = phase0_dir / "phase0_report.json"
        phase0_cmd = [
            self.python,
            str(self.root / "scripts/training/prepare_hybrid_training_data_antigravity.py"),
            "--input", str(train_file),
            "--output", str(approved_short),
            "--dev_output", str(approved_dev),
            "--dev_fraction", str(args.dev_fraction),
            "--min_dev_rows", str(args.min_dev_rows),
            "--bridge_output", str(bridge_holdout),
            "--long_output", str(long_holdout),
            "--report", str(phase0_report),
            "--neutral_name", "fn0",
            "--data_role", args.data_role,
            "--feedback_fraction", str(args.feedback_fraction),
            "--max_instructions", str(args.max_train_instructions),
            "--max_bridge_instructions", str(args.max_bridge_instructions),
            "--min_short_rows", str(args.min_short_rows),
            "--seed", str(args.seed),
            "--timeout", str(args.test_timeout),
            "--workers", str(args.verifier_workers),
        ]
        for path in forbidden_evals:
            phase0_cmd += ["--forbidden_eval", str(path)]
        self.run("00a_phase0_prepare", phase0_cmd, expected=phase0_report)

        phase0_audit = phase0_dir / "approved_short_reference_audit.json"
        self.run(
            "00b_phase0_reference_audit",
            [
                self.python,
                str(self.root / "scripts/evaluation/audit_grpo_reward_antigravity.py"),
                "--dataset", str(approved_short),
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

        phase0_dev_audit = phase0_dir / "approved_short_dev_reference_audit.json"
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

        probe_report = phase0_dir / "representation_probe.json"
        if args.run_probe:
            self.run(
                "00e_representation_probe",
                [
                    self.python,
                    str(self.root / "scripts/evaluation/probe_graph_representations_antigravity.py"),
                    "--dataset", str(approved_short),
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
        else:
            self.record_reuse("00e_representation_probe", probe_checkpoint, "probe explicitly disabled")

        stage1_dir = self.output / "01_graph_only_sft"
        if stage1_supplied is not None:
            stage1_checkpoint = stage1_supplied
            self.record_reuse("01_graph_only_sft", stage1_checkpoint, "supplied graph-only checkpoint")
        else:
            stage1_checkpoint = self.sft_stage(
                stage="01_graph_only_sft",
                train_file=approved_short,
                eval_file=approved_dev,
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

        stage1_gate_dir = self.output / "02a_graph_only_functional_gate"
        stage1_gate_report = stage1_gate_dir / "report.json"
        self.run(
            "02a_graph_only_functional_gate",
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
            ),
            expected=stage1_gate_report,
        )

        predictions = Path(args.predictions).expanduser().resolve() if args.predictions else self.output / "03_train_rollouts" / "predictions.json"
        if args.predictions:
            self.record_reuse("03_train_rollouts", predictions, "supplied rollout pool")
        else:
            infer_env = {
                "GRAPH_PROMPT_ASSEMBLY_MODE": "graph_only",
                "GRAPH_PROMPT_BINARY_EVIDENCE": "0",
                "GRAPH_REQUIRE_NEUTRAL_CONTRACT": "1",
                "GRAPH_NEUTRAL_FUNCTION_NAME": "fn0",
                "GRAPH_LOAD_4BIT": str(int(args.load_4bit)),
            }
            infer_cmd = [
                self.python,
                str(self.root / "scripts/evaluation/graph_inference_antigravity.py"),
                "--dataset", str(approved_short),
                "--decoder_model", args.decoder_model,
                "--checkpoint", str(stage1_checkpoint),
                "--output", str(predictions),
                "--num_samples", str(args.rollout_samples),
                "--generation_batch_size", str(args.rollout_generation_batch_size),
                "--max_new_tokens", str(args.rollout_max_new_tokens),
                "--decoder_prompt_max_length", str(args.decoder_prompt_max_length),
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
            "--dataset", str(approved_short),
            "--predictions", str(predictions),
            "--prediction_provenance", str(predictions) + ".provenance.json",
            "--expected_checkpoint", str(stage1_checkpoint),
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
            "phase0_short_train": approved_short,
            "phase0_short_dev": approved_dev,
            "bridge_function_holdout": bridge_holdout,
            "long_function_holdout": long_holdout,
            "phase0_report": phase0_report,
            "neutral_eval": neutral_eval,
            "probe_report": probe_report if args.run_probe else None,
            "stage1_checkpoint": stage1_checkpoint,
            "stage1_functional_gate": stage1_gate_report,
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
                "--gold", str(approved_short),
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
        needs_gold_control = (
            not args.rs_sft_baseline_checkpoint and args.baseline_pass_at_k < 0.0
        )
        if needs_gold_control:
            control_dir = self.output / "08a_gold_only_full_sft_control"
            gold_control_checkpoint = self.sft_stage(
                stage="08a_gold_only_full_sft_control",
                train_file=gold_control,
                eval_file=approved_dev,
                output_dir=control_dir,
                checkpoint=stage1_checkpoint,
                prompt_mode="full",
                binary_evidence=True,
                dependence_weight=0.0,
                dependence_every=1,
                learning_rate=args.stage2_lr,
                epochs=args.stage2_epochs,
            )

        stage2_dir = self.output / "08b_rejection_sampling_sft"
        stage2_checkpoint = self.sft_stage(
            stage="08b_rejection_sampling_sft",
            train_file=balanced_mix,
            eval_file=approved_dev,
            output_dir=stage2_dir,
            checkpoint=stage1_checkpoint,
            prompt_mode="full",
            binary_evidence=True,
            dependence_weight=0.0,  # full-text wrong-graph margin is confounded
            dependence_every=1,
            learning_rate=args.stage2_lr,
            epochs=args.stage2_epochs,
        )

        stage2_gate_dir = self.output / "09_rs_sft_functional_kill_switch"
        stage2_gate_report = stage2_gate_dir / "report.json"
        baseline_checkpoint: Path | None = None
        baseline_pass_at_k = args.baseline_pass_at_k
        if args.rs_sft_baseline_checkpoint:
            baseline_checkpoint = Path(
                args.rs_sft_baseline_checkpoint
            ).expanduser().resolve()
            if not args.dry_run:
                file_record(baseline_checkpoint)
        elif baseline_pass_at_k < 0.0:
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
                baseline_pass_at_k=baseline_pass_at_k,
            ),
            expected=stage2_gate_report,
        )

        final_checkpoint = stage2_checkpoint
        final_gate_report: Path | None = stage2_gate_report
        if args.run_grpo:
            grpo_dir = self.output / "10_verpo"
            grpo_env = {
                "GRPO_TRAIN_FILE": str(approved_short),
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
                "GRAPH_BATCH_SIZE": str(args.grpo_batch_size),
                "GRAPH_GRAD_ACCUM": str(args.grpo_grad_accum),
                "GRPO_REWARD_MODE": args.grpo_reward_mode,
                "GRPO_REWARD_TEST_FIELD": "feedback_tests",
                "GRPO_GROUP_SIZE": str(args.grpo_group_size),
                "GRPO_EPOCHS": str(args.grpo_epochs),
                "GRPO_LR": str(args.grpo_lr),
                "GRPO_TEST_TIMEOUT": str(args.test_timeout),
                "GRPO_MAX_NEW_TOKENS": str(args.grpo_max_new_tokens),
                "GRPO_REWARD_WORKERS": str(args.verifier_workers),
                "GRPO_ADV_NORM": "mean",
                "GRPO_MIN_REWARD_RANGE": str(args.grpo_min_reward_range),
                "GRPO_PASSK_K": str(args.grpo_passk_k),
                "GRPO_LOSS_POOLING": "seq",
                "GRPO_SCORE_CHUNK_SIZE": str(args.grpo_score_chunk_size),
                "GRPO_TRAIN_GRAPH_GLUE": "1",
                "GRPO_KL_COEF": "0.0",
                "GRPO_SFT_ANCHOR_COEF": str(args.grpo_sft_anchor_coef),
                "GRPO_SFT_ANCHOR_ON_NO_SIGNAL": "0",
                "GRPO_DYNAMIC_RESAMPLE_ATTEMPTS": str(args.grpo_dynamic_resample_attempts),
                "GRAPH_MAX_STEPS": str(args.max_steps),
            }
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
        print(f"Bridge holdout 151-{args.max_bridge_instructions} instructions (not trained): {bridge_holdout}")
        print(f"Long-function holdout >{args.max_bridge_instructions} instructions (not trained): {long_holdout}")
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
    parser.add_argument("--min_gate_rows", type=int, default=96)
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

    parser.add_argument("--gate_num_samples", type=int, default=10)
    parser.add_argument("--gate_k", type=int, default=10)
    parser.add_argument("--gate_generation_batch_size", type=int, default=4)
    parser.add_argument("--gate_max_new_tokens", type=int, default=1024)
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
    parser.add_argument("--rs_sft_min_improvement_pp", type=float, default=6.0)
    parser.add_argument("--final_min_improvement_pp", type=float, default=0.0)
    parser.add_argument(
        "--baseline_pass_at_k",
        type=float,
        default=-1.0,
        help=(
            "Frozen neutral-exact pass@k baseline as a fraction, e.g. 0.0455. "
            "When omitted, the runner trains a gold-only matched-step/full-prompt "
            "control from the same Stage-1 checkpoint."
        ),
    )
    parser.add_argument(
        "--rs_sft_baseline_checkpoint",
        default="",
        help=(
            "Optional external checkpoint evaluated with the same full-text gate prompt. "
            "When absent, a matched-step gold-only control is trained automatically."
        ),
    )
    parser.add_argument("--limit_tasks", type=int, default=0)

    parser.add_argument("--rollout_samples", type=int, default=16)
    parser.add_argument("--rollout_generation_batch_size", type=int, default=4)
    parser.add_argument("--rollout_max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit_candidates", type=int, default=0)
    parser.add_argument("--max_failures_per_task", type=int, default=2)
    parser.add_argument("--max_positives_per_task", type=int, default=3)

    parser.add_argument("--teacher_model", default="")
    parser.add_argument("--teacher_mode", choices=["sync", "batch"], default="sync")
    parser.add_argument("--teacher_test_visibility", choices=["none", "summary", "diagnostics"], default="summary")
    parser.add_argument("--teacher_concurrency", type=int, default=4)
    parser.add_argument("--teacher_max_output_tokens", type=int, default=3000)
    parser.add_argument("--teacher_limit", type=int, default=0)
    parser.add_argument("--teacher_base_url", default="")
    parser.add_argument("--facts_gate_mode", choices=["signature", "conservative", "strict"], default="conservative")
    parser.add_argument("--min_verified_rows", type=int, default=64)
    parser.add_argument("--min_verified_unique_tasks", type=int, default=64)
    parser.add_argument("--min_verified_length_bins", type=int, default=3)
    parser.add_argument("--max_verified_oversample_factor", type=float, default=4.0)

    parser.add_argument("--stage2_lr", type=float, default=3e-6)
    parser.add_argument("--stage2_epochs", type=float, default=1.0)

    parser.add_argument("--run_grpo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grpo_reward_mode", choices=["binary", "shaped", "verpo"], default="verpo")
    parser.add_argument("--grpo_group_size", type=int, default=8)
    parser.add_argument("--grpo_batch_size", type=int, default=1)
    parser.add_argument("--grpo_grad_accum", type=int, default=8)
    parser.add_argument("--grpo_epochs", type=int, default=1)
    parser.add_argument("--grpo_lr", type=float, default=1e-6)
    parser.add_argument("--grpo_max_new_tokens", type=int, default=768)
    parser.add_argument("--grpo_min_reward_range", type=float, default=0.05)
    parser.add_argument("--grpo_passk_k", type=int, default=10)
    parser.add_argument("--grpo_score_chunk_size", type=int, default=4)
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
        "max_train_instructions", "max_bridge_instructions", "min_short_rows", "min_dev_rows", "min_gate_rows", "test_timeout",
        "verifier_workers", "probe_max_rows", "probe_min_rows", "stage1_dependence_every",
        "prefix_negative_bank_size", "sft_batch_size", "sft_grad_accum", "gate_num_samples",
        "gate_bootstrap_iterations", "gate_min_causal_effective_pairs",
        "gate_min_deployment_effective_pairs",
        "gate_k", "gate_generation_batch_size", "gate_max_new_tokens", "rollout_samples",
        "rollout_generation_batch_size", "rollout_max_new_tokens", "teacher_concurrency",
        "min_verified_rows", "min_verified_unique_tasks", "min_verified_length_bins",
        "grpo_group_size", "grpo_batch_size", "grpo_grad_accum", "grpo_epochs",
        "grpo_max_new_tokens", "grpo_score_chunk_size",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_bridge_instructions < args.max_train_instructions:
        parser.error("--max_bridge_instructions must be >= --max_train_instructions")
    if not 0.0 < args.dev_fraction < 1.0:
        parser.error("--dev_fraction must be strictly between 0 and 1")
    if args.gate_k > args.gate_num_samples:
        parser.error("--gate_k cannot exceed --gate_num_samples")
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
    if args.gate_min_causal_task_losses < 0:
        parser.error("--gate_min_causal_task_losses must be non-negative")
    if args.baseline_pass_at_k > 1.0:
        parser.error("--baseline_pass_at_k must be <=1 or negative to disable")
    runner = CurriculumRunner(args)
    raise SystemExit(runner.execute())


if __name__ == "__main__":
    main()
