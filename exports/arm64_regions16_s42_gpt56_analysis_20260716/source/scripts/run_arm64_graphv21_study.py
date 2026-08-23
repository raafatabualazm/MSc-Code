#!/usr/bin/env python3
"""Resumable leakage-free ARM64 Graph-v2.1 replication study.

The runner keeps ARM64 external-validity results separate from the x86 study,
pins the immutable prepared split, and defaults to a dry plan.  The pilot
contains the complete causal architecture matrix; later phases repeat the
scientifically useful arms and apply inference-only causal controls.
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
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_leakage_free_study import (  # noqa: E402
    DECODER_REV,
    ENCODER_REV,
    EXPERIMENT,
    MODEL_STEM,
    arm_args,
)
from scripts.run_graphv2_followups import (  # noqa: E402
    CLAP_ENCODER_REV,
    CLAP_MODEL_STEM,
    gate_slug,
)


PYTHON = sys.executable
RUNNER = ROOT / "configs" / "run_sweeps_antigravity.py"
ARM_DIR = "data/datasets/arm64_graphv2"
ARM_TRAIN = f"{ARM_DIR}/flutter_train_graphv2.jsonl"
ARM_EVAL = f"{ARM_DIR}/flutter_eval_graphv2.jsonl"
ARM_FULL = f"{ARM_DIR}/flutter_function_assembly_pool_graphv2.jsonl"
ARM_SUMMARY = f"{ARM_DIR}/flutter_function_assembly_pool_graphv2.summary.json"
ARM_SPLIT_MANIFEST = f"{ARM_DIR}/flutter_split_graphv2.manifest.json"
ARM_STEM = MODEL_STEM + "_arm64v21"
CLAP_ARM_STEM = CLAP_MODEL_STEM + "_arm64v21"

EXPECTED_FILES = {
    ARM_FULL: (1714, "bd64a1e8d24dc93a89f05d7f58cbaa9b4a09c7232e0a85555561f9dbeaa1519b"),
    ARM_TRAIN: (1371, "f21782dd60edc11988867659dd2d16a5f6b6d2f550594cae09ad7cf92b68dcb7"),
    ARM_EVAL: (343, "864dc0bb7e9ee305ba0fc4be6e5d5ecbbeb7c17fd15bb3e41cfcc6d9aaf65fac"),
}

PILOT_ARMS = (
    "signature_only",
    "text",
    "prefix_no_edges",
    "prefix_cfg",
    "prefix_cfg_dfg",
    "prefix_shuffled",
    "prefix_no_gine",
)
DEFAULT_REPEAT_ARMS = (
    "text",
    "prefix_no_edges",
    "prefix_cfg",
    "prefix_cfg_dfg",
    "prefix_no_gine",
)

# ARM64 replication is gated on the x86 screen. These specifications preserve
# the winning x86 architecture exactly while changing only the ISA dataset.
SELECTED_ARCHITECTURES = {
    "prefix_no_gine": {
        "base_arm": "prefix_no_gine",
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "estimate_hours": 2.5,
        "expected_encoder": "microsoft/graphcodebert-base",
    },
    "prefix_no_edges": {
        "base_arm": "prefix_no_edges",
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "estimate_hours": 2.5,
        "expected_encoder": "microsoft/graphcodebert-base",
    },
    "prefix_cfg": {
        "base_arm": "prefix_cfg",
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "estimate_hours": 2.5,
        "expected_encoder": "microsoft/graphcodebert-base",
    },
    "prefix_cfg_dfg": {
        "base_arm": "prefix_cfg_dfg",
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "estimate_hours": 2.5,
        "expected_encoder": "microsoft/graphcodebert-base",
    },
    "prefix_no_gine_regions": {
        "base_arm": "prefix_no_gine",
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "linear_residual",
        "region_max_blocks": 8,
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "estimate_hours": 2.7,
        "expected_encoder": "microsoft/graphcodebert-base",
    },
    "prefix_no_gine_regions16": {
        "base_arm": "prefix_no_gine",
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "linear_residual",
        "region_max_blocks": 16,
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "estimate_hours": 2.7,
        "expected_encoder": "microsoft/graphcodebert-base",
    },
    "prefix_no_gine_multivector4": {
        "base_arm": "prefix_no_gine",
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "block_pooling": "multi_query",
        "vectors_per_block": 4,
        "estimate_hours": 3.0,
        "expected_encoder": "microsoft/graphcodebert-base",
    },
    "prefix_no_gine_clap": {
        "base_arm": "prefix_no_gine",
        "encoder": "clap",
        "encoder_revision": CLAP_ENCODER_REV,
        "region_compression": "off",
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "estimate_hours": 2.7,
        "expected_encoder": "hustcw/clap-asm",
    },
}

for _vector_count in (2, 8):
    SELECTED_ARCHITECTURES[f"prefix_no_gine_multivector{_vector_count}"] = {
        **SELECTED_ARCHITECTURES["prefix_no_gine_multivector4"],
        "vectors_per_block": _vector_count,
    }


@dataclass
class Stage:
    name: str
    model_name: str
    command: list[str]
    estimate_hours: float
    expected_graph_ablation: str = "none"
    expected_region_compression: str = "off"
    expected_region_max_blocks: int = 8
    expected_block_pooling: str = "cls"
    expected_encoder_model: str = "microsoft/graphcodebert-base"
    expected_prefix_density: int | None = None
    expected_gate_init: float | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def set_option(command: list[str], option: str, value: str) -> None:
    while option in command:
        index = command.index(option)
        del command[index:index + 2]
    command.extend([option, value])


def validate_prepared_data() -> None:
    for relative, (expected_rows, expected_sha) in EXPECTED_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Missing immutable ARM64 file: {path}")
        actual_rows = jsonl_rows(path)
        actual_sha = file_sha256(path)
        if actual_rows != expected_rows or actual_sha != expected_sha:
            raise SystemExit(
                f"ARM64 data mismatch for {relative}: rows={actual_rows} "
                f"sha256={actual_sha}; expected rows={expected_rows} sha256={expected_sha}"
            )

    summary = json.loads((ROOT / ARM_SUMMARY).read_text(encoding="utf-8"))
    split = json.loads((ROOT / ARM_SPLIT_MANIFEST).read_text(encoding="utf-8"))
    if (
        summary.get("graph_schema_version") != "antigravity-graph-v2.1"
        or summary.get("max_block_instrs") != 20
        or summary.get("max_dataflow_edges") != 0
        or summary.get("output_sha256") != EXPECTED_FILES[ARM_FULL][1]
    ):
        raise SystemExit("ARM64 Graph-v2.1 summary is stale or invalid")
    if (
        split.get("cross_split_component_count") != 0
        or split.get("seed") != 42
        or split.get("train_rows") != 1371
        or split.get("eval_rows") != 343
    ):
        raise SystemExit("ARM64 train/evaluation split manifest is stale or invalid")

    required_reports = {
        "results/arm64_graph_v2_dataset_audit.json": "passed",
        "results/arm64_graphv2_train_eval_overlap.json": "overlap",
        "results/arm64_graphv2_x86_benchmark_overlap.json": "overlap",
    }
    for relative in required_reports:
        if not (ROOT / relative).is_file():
            raise SystemExit(f"Missing ARM64 audit report: {relative}")
    structural = json.loads(
        (ROOT / "results/arm64_graph_v2_dataset_audit.json").read_text(encoding="utf-8")
    )
    if structural.get("passed") is not True:
        raise SystemExit("ARM64 structural/tokenizer audit did not pass")
    for relative in (
        "results/arm64_graphv2_train_eval_overlap.json",
        "results/arm64_graphv2_x86_benchmark_overlap.json",
    ):
        report = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        if report.get("exact_overlap_pairs", 0) != 0 or report.get("near_overlap_pairs", 0) != 0:
            raise SystemExit(f"ARM64 overlap audit failed: {relative}")
    print("ARM64 Graph-v2.1 immutable data and leakage audits: VERIFIED")


def architecture_stem(architecture: str) -> str:
    return CLAP_ARM_STEM if architecture == "prefix_no_gine_clap" else ARM_STEM


def model_name(seed: int, label: str, architecture: str | None = None) -> str:
    stem = architecture_stem(architecture or label)
    return f"{stem}_s{seed}_{label}"


def selected_label(args: argparse.Namespace) -> str:
    if (
        args.selected_prefix_density == 4
        and abs(args.selected_gate_init - 0.2) <= 1e-9
    ):
        return args.selected_architecture
    return (
        f"{args.selected_architecture}_ppl{args.selected_prefix_density}_"
        f"gate{gate_slug(args.selected_gate_init)}"
    )


def apply_selected_architecture(command: list[str], architecture: str) -> dict:
    try:
        spec = SELECTED_ARCHITECTURES[architecture]
    except KeyError as exc:
        raise SystemExit(f"Unknown selected architecture: {architecture}") from exc

    command.extend(arm_args(str(spec["base_arm"])))
    set_option(command, "--encoder", str(spec["encoder"]))
    set_option(command, "--encoder_revision", str(spec["encoder_revision"]))
    set_option(command, "--region_compression", str(spec["region_compression"]))
    set_option(
        command,
        "--region_max_blocks",
        str(spec.get("region_max_blocks", 8)),
    )
    set_option(command, "--block_pooling", str(spec["block_pooling"]))
    set_option(
        command,
        "--block_vectors_per_block",
        str(spec["vectors_per_block"]),
    )
    return spec


def base_command(seed: int, args: argparse.Namespace) -> list[str]:
    return [
        PYTHON,
        str(RUNNER),
        "--experiment", EXPERIMENT,
        "--encoder", "gcb",
        "--max_risk", "high",
        "--hardware_profile", "rtx6000",
        "--force_rerun",
        "--decoder_revision", DECODER_REV,
        "--encoder_revision", ENCODER_REV,
        "--seed", str(seed),
        "--train_file", ARM_TRAIN,
        "--eval_file", ARM_EVAL,
        "--compile_dataset", ARM_EVAL,
        "--pass_dataset", ARM_EVAL,
        "--compile_mode", "jit_tests",
        "--epochs", str(args.epochs),
        "--sft_lr", args.sft_lr,
        "--lora_r", "64",
        "--lora_alpha", "128",
        "--load_4bit", "0",
        "--attn_implementation", "sdpa",
        "--gradient_checkpointing", "1",
        "--train_batch_size", str(args.train_batch_size),
        "--grad_accum", str(args.grad_accum),
        "--qwen_prefix_gate_init", "0.2",
        "--qwen_prefix_gate_mode", "token",
        "--qwen_prefix_rms_match", "1",
        "--decoder_prompt_max_length", "2048",
        "--prompt_fit_assembly", "1",
        "--auto_cfg", "0",
        "--max_block_instrs", "20",
        "--position_scheme", "roberta",
        "--causal_position_ids", "cumsum",
        "--use_reasoning", "0",
        "--eval_max_new_tokens", "768",
        "--generation_batch_size", str(args.generation_batch_size),
        "--num_samples", "10",
        "--pass_num_samples", "10",
        "--metric_workers", str(args.metric_workers),
        # Keep final adapters but not optimizer-heavy epoch snapshots. The
        # prepared data and final results are the durable scientific artifacts.
        "--save_strategy", "no",
        "--save_total_limit", "1",
        "--gnn_layers", "4",
        "--global_attention_ablation", "full",
        "--region_compression", "off",
        "--region_max_blocks", "8",
        "--block_pooling", "cls",
        "--block_vectors_per_block", "4",
    ]


def add_hf(command: list[str], repo: str) -> None:
    if repo:
        command.extend([
            "--hf_repo", repo,
            "--hf_private", "1",
            "--hf_upload_checkpoints", "1",
        ])


def make_training_stage(
    seed: int,
    arm: str,
    args: argparse.Namespace,
    *,
    use_selected_config: bool = False,
) -> Stage:
    label = selected_label(args) if use_selected_config else arm
    name = model_name(
        seed,
        label,
        architecture=arm if arm in SELECTED_ARCHITECTURES else None,
    )
    command = base_command(seed, args)
    command.extend(["--name_suffix", f"_arm64v21_s{seed}_{label}"])
    if arm in SELECTED_ARCHITECTURES:
        spec = apply_selected_architecture(command, arm)
    else:
        command.extend(arm_args(arm))
        spec = {
            "region_compression": "off",
            "block_pooling": "cls",
            "expected_encoder": "microsoft/graphcodebert-base",
            "estimate_hours": 2.5,
        }
    if use_selected_config:
        set_option(
            command,
            "--qwen_prefix_tokens_per_log2",
            str(args.selected_prefix_density),
        )
        set_option(
            command,
            "--qwen_prefix_gate_init",
            str(args.selected_gate_init),
        )
    if arm == "signature_only":
        command.append("--skip_training")
    add_hf(command, args.hf_repo)
    estimate = (
        1.8 if arm == "signature_only" else float(spec["estimate_hours"])
    )
    return Stage(
        f"arm64_s{seed}_{label}",
        name,
        command,
        estimate,
        expected_region_compression=str(spec["region_compression"]),
        expected_region_max_blocks=int(spec.get("region_max_blocks", 8)),
        expected_block_pooling=str(spec["block_pooling"]),
        expected_encoder_model=str(spec["expected_encoder"]),
        expected_prefix_density=(
            args.selected_prefix_density if use_selected_config else None
        ),
        expected_gate_init=(
            args.selected_gate_init if use_selected_config else None
        ),
    )


def make_pilot(args: argparse.Namespace) -> list[Stage]:
    return [make_training_stage(args.seed, arm, args) for arm in PILOT_ARMS]


def make_repeats(args: argparse.Namespace) -> list[Stage]:
    allowed = set(PILOT_ARMS) | set(SELECTED_ARCHITECTURES)
    arms = csv_strings(args.repeat_arms)
    unknown = sorted(set(arms) - allowed)
    if unknown:
        raise SystemExit("Unknown --repeat_arms: " + ", ".join(unknown))
    return [
        make_training_stage(seed, arm, args)
        for seed in csv_ints(args.repeat_seeds)
        for arm in arms
    ]


def make_selected(args: argparse.Namespace) -> list[Stage]:
    """Replicate only the architecture selected by the x86 screen."""
    return [
        make_training_stage(
            seed,
            args.selected_architecture,
            args,
            use_selected_config=True,
        )
        for seed in csv_ints(args.selected_seeds)
    ]


def make_causal_controls(args: argparse.Namespace) -> list[Stage]:
    controls = (
        ("gate_zero", "none", ["--qwen_prefix_gate_override", "0.0"]),
        (
            "prefix_permuted",
            "cyclic_shift",
            [
                "--eval_graph_input_ablation", "cyclic_shift",
                "--eval_graph_ablation_seed", str(args.ablation_seed),
            ],
        ),
        (
            "block_order_shuffled",
            "shuffle_blocks",
            [
                "--eval_graph_input_ablation", "shuffle_blocks",
                "--eval_graph_ablation_seed", str(args.ablation_seed),
            ],
        ),
    )
    stages: list[Stage] = []
    source_label = selected_label(args)
    for seed in csv_ints(args.causal_seeds):
        source = model_name(
            seed,
            source_label,
            architecture=args.selected_architecture,
        )
        checkpoint = ROOT / "artifacts" / source / "pytorch_model.bin"
        for control, expected_ablation, extras in controls:
            arm = f"{source_label}_eval_{control}"
            name = model_name(
                seed,
                arm,
                architecture=args.selected_architecture,
            )
            command = base_command(seed, args)
            command.extend([
                "--name_suffix", f"_arm64v21_s{seed}_{arm}",
                "--skip_training",
                "--sft_checkpoint", str(checkpoint),
            ])
            spec = apply_selected_architecture(
                command,
                args.selected_architecture,
            )
            set_option(
                command,
                "--qwen_prefix_tokens_per_log2",
                str(args.selected_prefix_density),
            )
            set_option(
                command,
                "--qwen_prefix_gate_init",
                str(args.selected_gate_init),
            )
            command.extend(extras)
            add_hf(command, args.hf_repo)
            stages.append(Stage(
                f"arm64_causal_s{seed}_{control}",
                name,
                command,
                1.6,
                expected_graph_ablation=expected_ablation,
                expected_region_compression=str(spec["region_compression"]),
                expected_region_max_blocks=int(spec.get("region_max_blocks", 8)),
                expected_block_pooling=str(spec["block_pooling"]),
                expected_encoder_model=str(spec["expected_encoder"]),
                expected_prefix_density=args.selected_prefix_density,
                expected_gate_init=args.selected_gate_init,
            ))
    return stages


def result_complete(stage: Stage) -> bool:
    summary = ROOT / "results/sweeps_antigravity" / f"{stage.model_name}.json"
    predictions = ROOT / "results" / f"{stage.model_name}_pass_predictions.json"
    provenance = Path(str(predictions) + ".provenance.json")
    if not (summary.is_file() and predictions.is_file() and provenance.is_file()):
        return False
    try:
        rows = json.loads(predictions.read_text(encoding="utf-8"))
        prov = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    graph_environment = prov.get("graph_environment", {})
    required = (
        len(rows) == 343
        and all(len(row.get("predictions", [])) == 10 for row in rows)
        and prov.get("prompt_schema_version") == "antigravity-v2-no-test-hints"
        and prov.get("scoring_tests_visible_to_policy") is False
        and prov.get("dataset", {}).get("sha256") == EXPECTED_FILES[ARM_EVAL][1]
        and prov.get("graph_input_ablation", {}).get("mode", "none")
        == stage.expected_graph_ablation
        and graph_environment.get("GRAPH_REGION_COMPRESSION", "off")
        == stage.expected_region_compression
        and int(graph_environment.get("GRAPH_REGION_MAX_BLOCKS", "8"))
        == stage.expected_region_max_blocks
        and graph_environment.get("GRAPH_BLOCK_POOLING", "cls")
        == stage.expected_block_pooling
        and graph_environment.get(
            "GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base"
        )
        == stage.expected_encoder_model
    )
    if not required:
        return False
    if stage.expected_prefix_density is not None:
        try:
            density = int(graph_environment["GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2"])
        except (KeyError, TypeError, ValueError):
            return False
        if density != stage.expected_prefix_density:
            return False
    if stage.expected_gate_init is not None:
        try:
            gate_init = float(graph_environment["GRAPH_QWEN_PREFIX_GATE_INIT"])
        except (KeyError, TypeError, ValueError):
            return False
        if abs(gate_init - stage.expected_gate_init) > 1e-9:
            return False
    return True


def run_stages(args: argparse.Namespace, stages: list[Stage]) -> None:
    validate_prepared_data()
    started = time.monotonic()
    planned = sum(stage.estimate_hours for stage in stages)
    env = os.environ.copy()
    env.update({
        "GRAPH_STRICT_GRAPH": "1",
        "GRAPH_MAX_DATAFLOW_EDGES": "0",
        "GRAPH_ADD_REVERSE_EDGES": "1",
        "GRAPH_BLOCK_POSITION_MODE": "sinusoidal",
        "EVAL_PASS_STABILITY_RUNS": "3",
    })
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for stage in stages:
        print(f"\n=== {stage.name} (estimate {stage.estimate_hours:.2f} h) ===")
        print(shell_join(stage.command))
        if result_complete(stage):
            print("SKIP: complete ARM64 result and leakage-free provenance already exist")
            continue
        if not args.execute:
            continue
        elapsed = (time.monotonic() - started) / 3600
        if elapsed + stage.estimate_hours > args.budget_hours:
            raise SystemExit(
                f"Budget stop before {stage.name}: elapsed={elapsed:.2f} h, "
                f"next={stage.estimate_hours:.2f} h, budget={args.budget_hours:.2f} h"
            )
        log = ROOT / "logs/arm64_graphv21" / f"{stage.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\n$ " + shell_join(stage.command) + "\n")
            process = subprocess.Popen(
                stage.command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                handle.write(line)
            code = process.wait()
        if code:
            raise SystemExit(f"{stage.name} failed ({code}); see {log}")
        if not result_complete(stage):
            raise SystemExit(
                f"{stage.name} exited successfully but result/provenance is incomplete"
            )
    print(f"\nPlanned estimate: {planned:.2f} GPU/wall-clock hours")
    if not args.execute:
        print("Dry plan only. Add --execute to launch the selected ARM64 phase.")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--phase",
        choices=["pilot", "repeat", "selected", "causal", "all"],
        default="selected",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat_seeds", default="43,44")
    parser.add_argument("--repeat_arms", default=",".join(DEFAULT_REPEAT_ARMS))
    parser.add_argument(
        "--selected_seeds",
        default="42,43,44,45,46",
        help="ARM64 training seeds for the x86-selected architecture.",
    )
    parser.add_argument(
        "--causal_seeds",
        default="42,43,44,45,46",
        help="ARM64 selected-checkpoint seeds for inference-only causal controls.",
    )
    parser.add_argument(
        "--selected_architecture",
        choices=sorted(SELECTED_ARCHITECTURES),
        default="prefix_no_gine",
    )
    parser.add_argument("--selected_prefix_density", type=int, default=4)
    parser.add_argument("--selected_gate_init", type=float, default=0.2)
    parser.add_argument("--ablation_seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sft_lr", default="5e-6")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--generation_batch_size", type=int, default=10)
    parser.add_argument("--metric_workers", type=int, default=64)
    parser.add_argument("--budget_hours", type=float, default=40.0)
    parser.add_argument("--hf_repo", default=os.environ.get("HF_OUT_REPO", ""))
    args = parser.parse_args()

    if args.selected_prefix_density < 1:
        parser.error("--selected_prefix_density must be positive")
    if not 0.0 < args.selected_gate_init < 1.0:
        parser.error("--selected_gate_init must be between zero and one")

    builders = {
        "pilot": lambda: make_pilot(args),
        "repeat": lambda: make_repeats(args),
        "selected": lambda: make_selected(args),
        "causal": lambda: make_causal_controls(args),
    }
    if args.phase == "all":
        stages = []
        for phase in ("pilot", "repeat", "selected", "causal"):
            stages.extend(builders[phase]())
    else:
        stages = builders[args.phase]()
    run_stages(args, stages)


if __name__ == "__main__":
    main()
