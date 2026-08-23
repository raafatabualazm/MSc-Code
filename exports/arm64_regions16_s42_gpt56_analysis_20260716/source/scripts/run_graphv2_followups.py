#!/usr/bin/env python3
"""Resumable follow-up matrix for the leakage-free x86 Graph-v2 study.

The script is dry-run by default. It creates the confirmatory seed repeats,
causal inference controls, post-GINE architecture isolation, and a meaningful
dynamic-prefix capacity sweep. Expensive branches are separate phases so a
completed no-GINE result can select the next scientific question.
"""

from __future__ import annotations

import argparse
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
    BENCHMARK,
    DECODER_REV,
    ENCODER_REV,
    EXPERIMENT,
    MODEL_STEM,
    SFT_TRAIN,
    SFT_VALID,
    arm_args,
    common_args,
)

CLAP_ENCODER_REV = "620f4beba2edce172e8f35e263399716494950c9"
CLAP_MODEL_STEM = MODEL_STEM.replace("_gcb_a128", "_clap_a128")
CLAP_EXPERIMENT = EXPERIMENT.removesuffix("_gcb") + "_clap"
CLAP_FROZEN_ENCODER_EXPERIMENT = CLAP_EXPERIMENT.replace(
    "_lora_enc_dec_r16_", "_freeze_enc_lora_dec_"
)
CLAP_FROZEN_ENCODER_MODEL_STEM = CLAP_MODEL_STEM.replace(
    "_lora_enc_dec_r64_", "_freeze_enc_lora_dec_"
)
FROZEN_ENCODER_EXPERIMENT = EXPERIMENT.replace(
    "_lora_enc_dec_r16_", "_freeze_enc_lora_dec_"
)
FROZEN_ENCODER_MODEL_STEM = MODEL_STEM.replace(
    "_lora_enc_dec_r64_", "_freeze_enc_lora_dec_"
)


REPRESENTATION_SPECS = {
    "prefix_no_gine": {
        "base_arm": "prefix_no_gine",
        "model_stem": MODEL_STEM,
        "experiment": EXPERIMENT,
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "region_max_blocks": 8,
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "expected_encoder": "microsoft/graphcodebert-base",
        "estimate_hours": 1.4,
    },
    "prefix_no_edges": {
        "base_arm": "prefix_no_edges",
        "model_stem": MODEL_STEM,
        "experiment": EXPERIMENT,
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "region_max_blocks": 8,
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "expected_encoder": "microsoft/graphcodebert-base",
        "estimate_hours": 1.4,
    },
    "prefix_no_gine_regions": {
        "base_arm": "prefix_no_gine",
        "model_stem": MODEL_STEM,
        "experiment": EXPERIMENT,
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "linear_residual",
        "region_max_blocks": 8,
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "expected_encoder": "microsoft/graphcodebert-base",
        "estimate_hours": 1.5,
    },
    "prefix_no_gine_clap": {
        "base_arm": "prefix_no_gine",
        "model_stem": CLAP_MODEL_STEM,
        "experiment": CLAP_EXPERIMENT,
        "encoder": "clap",
        "encoder_revision": CLAP_ENCODER_REV,
        "region_compression": "off",
        "region_max_blocks": 8,
        "block_pooling": "cls",
        "vectors_per_block": 4,
        "expected_encoder": "hustcw/clap-asm",
        "estimate_hours": 1.5,
    },
}

for _vector_count in (2, 4, 8):
    REPRESENTATION_SPECS[f"prefix_no_gine_multivector{_vector_count}"] = {
        "base_arm": "prefix_no_gine",
        "model_stem": MODEL_STEM,
        "experiment": EXPERIMENT,
        "encoder": "gcb",
        "encoder_revision": ENCODER_REV,
        "region_compression": "off",
        "region_max_blocks": 8,
        "block_pooling": "multi_query",
        "vectors_per_block": _vector_count,
        "expected_encoder": "microsoft/graphcodebert-base",
        "estimate_hours": 1.7,
    }


@dataclass
class Stage:
    name: str
    model_name: str
    command: list[str]
    estimate_hours: float
    expected_graph_ablation: str = "none"
    expected_region_compression: str = "off"
    expected_block_pooling: str = "cls"
    expected_encoder_model: str = "microsoft/graphcodebert-base"
    expected_rows: int = 154
    expected_region_max_blocks: int | None = None
    expected_global_attention: str | None = None
    expected_block_position_mode: str | None = None
    expected_gnn_layers: int | None = None
    expected_prefix_density: int | None = None
    expected_gate_init: float | None = None
    expected_freeze_encoder: bool | None = None
    expected_encoder_peft: str | None = None
    expected_dfg_mode: str | None = None
    expected_edge_ablation: str | None = None
    expected_gnn_ablation: str | None = None


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def set_option(command: list[str], option: str, value: str) -> None:
    while option in command:
        index = command.index(option)
        del command[index:index + 2]
    command.extend([option, value])


def result_complete(stage: Stage) -> bool:
    summary = ROOT / "results" / "sweeps_antigravity" / f"{stage.model_name}.json"
    predictions = ROOT / "results" / f"{stage.model_name}_pass_predictions.json"
    provenance = Path(str(predictions) + ".provenance.json")
    if not (summary.is_file() and predictions.is_file() and provenance.is_file()):
        return False
    try:
        rows = json.loads(predictions.read_text(encoding="utf-8"))
        prov = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    graph_ablation = prov.get("graph_input_ablation", {}).get("mode", "none")
    graph_environment = prov.get("graph_environment", {})
    region_compression = graph_environment.get(
        "GRAPH_REGION_COMPRESSION", "off"
    )
    block_pooling = graph_environment.get("GRAPH_BLOCK_POOLING", "cls")
    encoder_model = graph_environment.get(
        "GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base"
    )
    required = (
        len(rows) == stage.expected_rows
        and all(len(row.get("predictions", [])) == 10 for row in rows)
        and prov.get("prompt_schema_version") == "antigravity-v2-no-test-hints"
        and prov.get("scoring_tests_visible_to_policy") is False
        and graph_ablation == stage.expected_graph_ablation
        and region_compression == stage.expected_region_compression
        and block_pooling == stage.expected_block_pooling
        and encoder_model == stage.expected_encoder_model
    )
    if not required:
        return False

    optional_environment_checks = (
        (
            stage.expected_region_max_blocks,
            "GRAPH_REGION_MAX_BLOCKS",
            int,
        ),
        (
            stage.expected_global_attention,
            "GRAPH_GLOBAL_ATTENTION_ABLATION",
            str,
        ),
        (
            stage.expected_block_position_mode,
            "GRAPH_BLOCK_POSITION_MODE",
            str,
        ),
        (stage.expected_gnn_layers, "GRAPH_GNN_LAYERS", int),
        (
            stage.expected_prefix_density,
            "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2",
            int,
        ),
        (
            stage.expected_gate_init,
            "GRAPH_QWEN_PREFIX_GATE_INIT",
            float,
        ),
        (
            stage.expected_encoder_peft,
            "GRAPH_ENCODER_PEFT",
            str,
        ),
        (stage.expected_dfg_mode, "GRAPH_DFG_MODE", str),
        (stage.expected_edge_ablation, "GRAPH_EDGE_ABLATION", str),
        (stage.expected_gnn_ablation, "GRAPH_GNN_ABLATION", str),
    )
    for expected, key, converter in optional_environment_checks:
        if expected is None:
            continue
        try:
            actual = converter(graph_environment[key])
        except (KeyError, TypeError, ValueError):
            return False
        if converter is float:
            if abs(float(actual) - float(expected)) > 1e-9:
                return False
        elif actual != expected:
            return False

    if stage.expected_freeze_encoder is not None:
        actual = graph_environment.get("GRAPH_FREEZE_ENCODER") == "1"
        if actual != stage.expected_freeze_encoder:
            return False
    return True


def wait_for_result(model_name: str, poll_seconds: int, timeout_hours: float) -> None:
    stage = Stage(
        name="wait_dependency",
        model_name=model_name,
        command=[],
        estimate_hours=0.0,
    )
    started = time.monotonic()
    while not result_complete(stage):
        elapsed = (time.monotonic() - started) / 3600
        if elapsed >= timeout_hours:
            raise SystemExit(
                f"Timed out after {elapsed:.2f} h waiting for complete result: {model_name}"
            )
        print(
            f"Waiting for {model_name}: elapsed={elapsed:.2f} h; "
            f"checking again in {poll_seconds}s",
            flush=True,
        )
        time.sleep(poll_seconds)
    print(f"Dependency result is complete: {model_name}", flush=True)


def base_command(seed: int, workers: int) -> list[str]:
    command = common_args(seed, SFT_TRAIN, SFT_VALID, 4, "5e-6", workers)
    # The rented host has limited ephemeral disk. Follow-up trainings are short;
    # retain the final state only instead of two optimizer-heavy epoch snapshots.
    set_option(command, "--save_strategy", "no")
    set_option(command, "--save_total_limit", "1")
    command.extend([
        "--gnn_layers", "4",
        "--global_attention_ablation", "full",
        "--region_compression", "off",
        "--region_max_blocks", "8",
        "--block_pooling", "cls",
        "--block_vectors_per_block", "4",
    ])
    return command


def architecture_args(architecture: str) -> list[str]:
    if architecture not in {"prefix_no_gine", "prefix_no_edges"}:
        raise ValueError(f"unsupported selected architecture: {architecture}")
    return arm_args(architecture)


def add_hf(command: list[str], repo: str) -> None:
    if repo:
        command.extend([
            "--hf_repo", repo,
            "--hf_private", "1",
            "--hf_upload_checkpoints", "1",
        ])


def make_seed_repeats(args: argparse.Namespace) -> list[Stage]:
    stages: list[Stage] = []
    for seed in csv_ints(args.repeat_seeds):
        for arm in ("signature_only", "text", "prefix_no_gine"):
            model_name = f"{MODEL_STEM}_graphv2_clean_s{seed}_{arm}"
            command = base_command(seed, args.metric_workers)
            command.extend(["--name_suffix", f"_graphv2_clean_s{seed}_{arm}"])
            command.extend(arm_args(arm))
            if arm == "signature_only":
                command.append("--skip_training")
            add_hf(command, args.hf_repo)
            stages.append(Stage(
                name=f"x86_repeat_s{seed}_{arm}",
                model_name=model_name,
                command=command,
                estimate_hours=1.1 if arm == "signature_only" else 1.4,
            ))
    return stages


def selected_model(seed: int, architecture: str) -> str:
    return f"{MODEL_STEM}_graphv2_clean_s{seed}_{architecture}"


def make_causal_controls(args: argparse.Namespace) -> list[Stage]:
    seed = args.seed
    controls = (
        ("gate_zero", "none", ["--qwen_prefix_gate_override", "0.0"]),
        ("prefix_permuted", "cyclic_shift", [
            "--eval_graph_input_ablation", "cyclic_shift",
            "--eval_graph_ablation_seed", str(args.ablation_seed),
        ]),
        ("block_order_shuffled", "shuffle_blocks", [
            "--eval_graph_input_ablation", "shuffle_blocks",
            "--eval_graph_ablation_seed", str(args.ablation_seed),
        ]),
    )
    stages: list[Stage] = []
    architectures = csv_strings(args.causal_architectures)
    unknown = sorted(set(architectures) - {"prefix_no_gine", "prefix_no_edges"})
    if unknown:
        raise SystemExit("Unknown --causal_architectures: " + ", ".join(unknown))
    for architecture in architectures:
        source_name = selected_model(seed, architecture)
        checkpoint = ROOT / "artifacts" / source_name / "pytorch_model.bin"
        for label, graph_ablation, extras in controls:
            suffix = f"_graphv2_clean_s{seed}_{architecture}_eval_{label}"
            model_name = MODEL_STEM + suffix
            command = base_command(seed, args.metric_workers)
            command.extend([
                "--name_suffix", suffix,
                "--skip_training",
                "--sft_checkpoint", str(checkpoint),
            ])
            command.extend(architecture_args(architecture))
            command.extend(extras)
            add_hf(command, args.hf_repo)
            stages.append(Stage(
                name=f"x86_causal_{architecture}_{label}",
                model_name=model_name,
                command=command,
                estimate_hours=0.8,
                expected_graph_ablation=graph_ablation,
            ))
    return stages


def make_isolation(args: argparse.Namespace) -> list[Stage]:
    seed = args.seed
    specifications = [
        (
            "prefix_no_gine_no_attention",
            "prefix_no_gine",
            ["--global_attention_ablation", "identity"],
        ),
        (
            "prefix_no_edges_gine2",
            "prefix_no_edges",
            ["--gnn_layers", "2"],
        ),
        (
            "prefix_no_gine_regions",
            "prefix_no_gine",
            [
                "--region_compression", "linear_residual",
                "--region_max_blocks", "8",
            ],
        ),
        (
            "prefix_no_gine_no_positions",
            "prefix_no_gine",
            ["--block_position_mode", "off"],
        ),
        (
            "prefix_no_gine_frozen_encoder",
            "prefix_no_gine",
            ["--experiment", FROZEN_ENCODER_EXPERIMENT],
        ),
    ]
    selected = set(csv_strings(args.isolation_variants))
    known = {label for label, _, _ in specifications}
    unknown = sorted(selected - known)
    if unknown:
        raise SystemExit("Unknown --isolation_variants: " + ", ".join(unknown))
    specifications = [
        specification
        for specification in specifications
        if specification[0] in selected
    ]
    stages: list[Stage] = []
    for label, arm, extras in specifications:
        suffix = f"_graphv2_clean_s{seed}_{label}"
        model_stem = (
            FROZEN_ENCODER_MODEL_STEM
            if label == "prefix_no_gine_frozen_encoder"
            else MODEL_STEM
        )
        model_name = model_stem + suffix
        command = base_command(seed, args.metric_workers)
        command.extend(["--name_suffix", suffix])
        command.extend(arm_args(arm))
        if len(extras) % 2:
            raise ValueError(f"Isolation options must be key/value pairs: {extras}")
        for index in range(0, len(extras), 2):
            set_option(command, extras[index], extras[index + 1])
        add_hf(command, args.hf_repo)
        stages.append(Stage(
            name=f"x86_isolation_{label}",
            model_name=model_name,
            command=command,
            estimate_hours=1.5 if label == "prefix_no_gine_regions" else 1.4,
            expected_region_compression=(
                "linear_residual" if label == "prefix_no_gine_regions" else "off"
            ),
            expected_region_max_blocks=(
                8 if label == "prefix_no_gine_regions" else None
            ),
            expected_global_attention=(
                "identity" if label == "prefix_no_gine_no_attention" else None
            ),
            expected_block_position_mode=(
                "off" if label == "prefix_no_gine_no_positions" else None
            ),
            expected_gnn_layers=(
                2 if label == "prefix_no_edges_gine2" else None
            ),
            expected_freeze_encoder=(
                True if label == "prefix_no_gine_frozen_encoder" else None
            ),
            expected_encoder_peft=(
                "none" if label == "prefix_no_gine_frozen_encoder" else None
            ),
        ))
    return stages


def make_encoder_controls(args: argparse.Namespace) -> list[Stage]:
    seed = args.seed
    specifications = {
        "prefix_no_gine_clap": {
            "model_stem": CLAP_MODEL_STEM,
            "experiment": CLAP_EXPERIMENT,
            "encoder": "clap",
            "encoder_revision": CLAP_ENCODER_REV,
            "block_pooling": "cls",
            "vectors_per_block": 4,
            "estimate_hours": 1.5,
            "expected_encoder": "hustcw/clap-asm",
        },
        "prefix_no_gine_multivector4": {
            "model_stem": MODEL_STEM,
            "experiment": EXPERIMENT,
            "encoder": "gcb",
            "encoder_revision": ENCODER_REV,
            "block_pooling": "multi_query",
            "vectors_per_block": 4,
            "estimate_hours": 1.7,
            "expected_encoder": "microsoft/graphcodebert-base",
        },
    }
    selected = csv_strings(args.encoder_variants)
    unknown = sorted(set(selected) - set(specifications))
    if unknown:
        raise SystemExit("Unknown --encoder_variants: " + ", ".join(unknown))

    stages: list[Stage] = []
    for label in selected:
        spec = specifications[label]
        suffix = f"_graphv2_clean_s{seed}_{label}"
        model_name = str(spec["model_stem"]) + suffix
        command = base_command(seed, args.metric_workers)
        command.extend(["--name_suffix", suffix])
        command.extend(arm_args("prefix_no_gine"))
        set_option(command, "--experiment", str(spec["experiment"]))
        set_option(command, "--encoder", str(spec["encoder"]))
        set_option(command, "--encoder_revision", str(spec["encoder_revision"]))
        set_option(command, "--block_pooling", str(spec["block_pooling"]))
        set_option(
            command,
            "--block_vectors_per_block",
            str(spec["vectors_per_block"]),
        )
        add_hf(command, args.hf_repo)
        stages.append(Stage(
            name=f"x86_encoder_{label}",
            model_name=model_name,
            command=command,
            estimate_hours=float(spec["estimate_hours"]),
            expected_block_pooling=str(spec["block_pooling"]),
            expected_encoder_model=str(spec["expected_encoder"]),
        ))
    return stages


def make_interaction_controls(args: argparse.Namespace) -> list[Stage]:
    seed = args.seed
    specifications = {
        "prefix_no_gine_clap_frozen_encoder": {
            "arm": "prefix_no_gine",
            "model_stem": CLAP_FROZEN_ENCODER_MODEL_STEM,
            "experiment": CLAP_FROZEN_ENCODER_EXPERIMENT,
            "freeze_encoder": True,
            "encoder_peft": "none",
            "dfg_mode": "edges",
            "edge_ablation": "full",
            "gnn_ablation": "identity",
        },
        "prefix_cfg_clap": {
            "arm": "prefix_cfg",
            "model_stem": CLAP_MODEL_STEM,
            "experiment": CLAP_EXPERIMENT,
            "freeze_encoder": False,
            "encoder_peft": "lora",
            "dfg_mode": "edges",
            "edge_ablation": "cfg",
            "gnn_ablation": "full",
        },
    }
    selected = csv_strings(args.interaction_variants)
    unknown = sorted(set(selected) - set(specifications))
    if unknown:
        raise SystemExit("Unknown --interaction_variants: " + ", ".join(unknown))
    if len(selected) != len(set(selected)):
        raise SystemExit("--interaction_variants must not contain duplicates")

    stages: list[Stage] = []
    for label in selected:
        spec = specifications[label]
        suffix = f"_graphv2_clean_s{seed}_{label}"
        model_name = str(spec["model_stem"]) + suffix
        command = base_command(seed, args.metric_workers)
        command.extend(["--name_suffix", suffix])
        command.extend(arm_args(str(spec["arm"])))
        set_option(command, "--experiment", str(spec["experiment"]))
        set_option(command, "--encoder", "clap")
        set_option(command, "--encoder_revision", CLAP_ENCODER_REV)
        add_hf(command, args.hf_repo)
        stages.append(Stage(
            name=f"x86_interaction_{label}",
            model_name=model_name,
            command=command,
            estimate_hours=1.5,
            expected_encoder_model="hustcw/clap-asm",
            expected_freeze_encoder=bool(spec["freeze_encoder"]),
            expected_encoder_peft=str(spec["encoder_peft"]),
            expected_dfg_mode=str(spec["dfg_mode"]),
            expected_edge_ablation=str(spec["edge_ablation"]),
            expected_gnn_ablation=str(spec["gnn_ablation"]),
        ))
    return stages


def representation_spec(label: str) -> dict[str, object]:
    try:
        return REPRESENTATION_SPECS[label]
    except KeyError as exc:
        raise SystemExit(f"Unknown --selected_representation: {label}") from exc


def apply_representation(command: list[str], label: str) -> dict[str, object]:
    spec = representation_spec(label)
    command.extend(arm_args(str(spec["base_arm"])))
    set_option(command, "--experiment", str(spec["experiment"]))
    set_option(command, "--encoder", str(spec["encoder"]))
    set_option(command, "--encoder_revision", str(spec["encoder_revision"]))
    set_option(
        command,
        "--region_compression",
        str(spec["region_compression"]),
    )
    set_option(
        command,
        "--region_max_blocks",
        str(spec["region_max_blocks"]),
    )
    set_option(command, "--block_pooling", str(spec["block_pooling"]))
    set_option(
        command,
        "--block_vectors_per_block",
        str(spec["vectors_per_block"]),
    )
    return spec


def gate_slug(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def configured_label(
    representation: str,
    prefix_density: int,
    gate_init: float,
) -> str:
    if prefix_density == 4 and abs(gate_init - 0.2) <= 1e-9:
        return representation
    return (
        f"{representation}_ppl{prefix_density}_gate{gate_slug(gate_init)}"
    )


def configured_model_name(
    seed: int,
    representation: str,
    prefix_density: int,
    gate_init: float,
) -> str:
    spec = representation_spec(representation)
    label = configured_label(representation, prefix_density, gate_init)
    return f"{spec['model_stem']}_graphv2_clean_s{seed}_{label}"


def configured_stage(
    args: argparse.Namespace,
    *,
    seed: int,
    representation: str,
    prefix_density: int,
    gate_init: float,
) -> Stage:
    spec = representation_spec(representation)
    label = configured_label(representation, prefix_density, gate_init)
    suffix = f"_graphv2_clean_s{seed}_{label}"
    command = base_command(seed, args.metric_workers)
    command.extend(["--name_suffix", suffix])
    apply_representation(command, representation)
    set_option(command, "--qwen_prefix_tokens_per_log2", str(prefix_density))
    set_option(command, "--qwen_prefix_gate_init", str(gate_init))
    add_hf(command, args.hf_repo)
    is_existing_default = prefix_density == 4 and abs(gate_init - 0.2) <= 1e-9
    return Stage(
        name=f"x86_config_s{seed}_{label}",
        model_name=str(spec["model_stem"]) + suffix,
        command=command,
        estimate_hours=float(spec["estimate_hours"]),
        expected_region_compression=str(spec["region_compression"]),
        expected_region_max_blocks=(
            None
            if is_existing_default and spec["region_compression"] == "off"
            else int(spec["region_max_blocks"])
        ),
        expected_block_pooling=str(spec["block_pooling"]),
        expected_encoder_model=str(spec["expected_encoder"]),
        # Older archived default-cell provenance predates these two fields.
        expected_prefix_density=None if is_existing_default else prefix_density,
        expected_gate_init=None if is_existing_default else gate_init,
    )


def make_vector_sweep(args: argparse.Namespace) -> list[Stage]:
    values = csv_ints(args.vector_values)
    if any(value not in {2, 4, 8} for value in values):
        raise SystemExit("--vector_values must be a subset of 2,4,8")
    stages: list[Stage] = []
    for value in values:
        label = f"prefix_no_gine_multivector{value}"
        spec = representation_spec(label)
        suffix = f"_graphv2_clean_s{args.seed}_{label}"
        command = base_command(args.seed, args.metric_workers)
        command.extend(["--name_suffix", suffix])
        apply_representation(command, label)
        add_hf(command, args.hf_repo)
        stages.append(Stage(
            name=f"x86_vector_s{args.seed}_v{value}",
            model_name=str(spec["model_stem"]) + suffix,
            command=command,
            estimate_hours=float(spec["estimate_hours"]),
            expected_block_pooling="multi_query",
            expected_encoder_model=str(spec["expected_encoder"]),
        ))
    return stages


def make_region_sweep(args: argparse.Namespace) -> list[Stage]:
    values = csv_ints(args.region_values)
    if any(value < 2 for value in values):
        raise SystemExit("--region_values must contain integers >= 2")
    stages: list[Stage] = []
    for value in values:
        label = (
            "prefix_no_gine_regions"
            if value == 8
            else f"prefix_no_gine_regions{value}"
        )
        suffix = f"_graphv2_clean_s{args.seed}_{label}"
        command = base_command(args.seed, args.metric_workers)
        command.extend(["--name_suffix", suffix])
        command.extend(arm_args("prefix_no_gine"))
        set_option(command, "--region_compression", "linear_residual")
        set_option(command, "--region_max_blocks", str(value))
        add_hf(command, args.hf_repo)
        stages.append(Stage(
            name=f"x86_region_s{args.seed}_max{value}",
            model_name=MODEL_STEM + suffix,
            command=command,
            estimate_hours=1.5,
            expected_region_compression="linear_residual",
            expected_region_max_blocks=value,
        ))
    return stages


def make_prefix_grid(args: argparse.Namespace) -> list[Stage]:
    densities = csv_ints(args.prefix_density_values)
    gates = csv_floats(args.prefix_gate_values)
    if any(value < 1 for value in densities):
        raise SystemExit("prefix density values must be positive")
    if any(not 0.0 < value < 1.0 for value in gates):
        raise SystemExit("prefix gate values must be between zero and one")
    return [
        configured_stage(
            args,
            seed=args.seed,
            representation=args.selected_representation,
            prefix_density=density,
            gate_init=gate,
        )
        for density in densities
        for gate in gates
    ]


def make_confirmatory_repeats(args: argparse.Namespace) -> list[Stage]:
    return [
        configured_stage(
            args,
            seed=seed,
            representation=args.selected_representation,
            prefix_density=args.selected_prefix_density,
            gate_init=args.selected_gate_init,
        )
        for seed in csv_ints(args.confirm_seeds)
    ]


def make_confirmatory_causal(args: argparse.Namespace) -> list[Stage]:
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
    spec = representation_spec(args.selected_representation)
    stages: list[Stage] = []
    for seed in csv_ints(args.confirm_seeds):
        source_label = configured_label(
            args.selected_representation,
            args.selected_prefix_density,
            args.selected_gate_init,
        )
        source_name = configured_model_name(
            seed,
            args.selected_representation,
            args.selected_prefix_density,
            args.selected_gate_init,
        )
        checkpoint = ROOT / "artifacts" / source_name / "pytorch_model.bin"
        for control, graph_ablation, extras in controls:
            label = f"{source_label}_eval_{control}"
            suffix = f"_graphv2_clean_s{seed}_{label}"
            command = base_command(seed, args.metric_workers)
            command.extend([
                "--name_suffix", suffix,
                "--skip_training",
                "--sft_checkpoint", str(checkpoint),
            ])
            apply_representation(command, args.selected_representation)
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
                name=f"x86_confirm_causal_s{seed}_{control}",
                model_name=str(spec["model_stem"]) + suffix,
                command=command,
                estimate_hours=0.8,
                expected_graph_ablation=graph_ablation,
                expected_region_compression=str(spec["region_compression"]),
                expected_region_max_blocks=int(spec["region_max_blocks"]),
                expected_block_pooling=str(spec["block_pooling"]),
                expected_encoder_model=str(spec["expected_encoder"]),
                expected_prefix_density=args.selected_prefix_density,
                expected_gate_init=args.selected_gate_init,
            ))
    return stages


def make_capacity(args: argparse.Namespace, repeat: bool = False) -> list[Stage]:
    values = (
        [args.capacity_winner]
        if repeat
        else csv_ints(args.capacity_values)
    )
    if any(value is None or value < 1 for value in values):
        raise SystemExit("capacity values must be positive integers")
    seeds = csv_ints(args.repeat_seeds) if repeat else [args.seed]
    stages: list[Stage] = []
    for seed in seeds:
        for value in values:
            suffix = (
                f"_graphv2_clean_s{seed}_{args.selected_architecture}"
                f"_tokens_per_log2_{value}"
            )
            model_name = MODEL_STEM + suffix
            command = base_command(seed, args.metric_workers)
            command.extend(["--name_suffix", suffix])
            command.extend(architecture_args(args.selected_architecture))
            set_option(command, "--qwen_prefix_tokens_per_log2", str(value))
            add_hf(command, args.hf_repo)
            stages.append(Stage(
                name=f"x86_capacity_s{seed}_ppl{value}",
                model_name=model_name,
                command=command,
                estimate_hours=1.4,
            ))
    return stages


def run_stages(args: argparse.Namespace, stages: list[Stage]) -> None:
    started = time.monotonic()
    env = os.environ.copy()
    env["GRAPH_STRICT_GRAPH"] = "1"
    env["GRAPH_MAX_DATAFLOW_EDGES"] = "0"
    planned = 0.0
    for stage in stages:
        planned += stage.estimate_hours
        print(f"\n=== {stage.name} (estimate {stage.estimate_hours:.2f} h) ===")
        print(shell_join(stage.command))
        if result_complete(stage):
            print("SKIP: complete leakage-free result and provenance already exist")
            continue
        if not args.execute:
            continue
        elapsed = (time.monotonic() - started) / 3600
        if elapsed + stage.estimate_hours > args.budget_hours:
            raise SystemExit(
                f"Budget stop before {stage.name}: elapsed={elapsed:.2f} h, "
                f"next={stage.estimate_hours:.2f} h, budget={args.budget_hours:.2f} h"
            )
        log = ROOT / "logs" / "graphv2_followups" / f"{stage.name}.log"
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
                f"{stage.name} exited successfully but its result/provenance is incomplete"
            )
    print(f"\nPlanned estimate: {planned:.2f} GPU/wall-clock hours")
    if not args.execute:
        print("Dry plan only. Add --execute after selecting the scientifically justified phase.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "x86-repeats",
            "x86-causal",
            "x86-isolation",
            "x86-encoder",
            "x86-interaction",
            "x86-vector",
            "x86-region",
            "x86-prefix-grid",
            "x86-confirm",
            "x86-confirm-causal",
            "x86-comprehensive-screen",
            "x86-capacity",
            "x86-capacity-repeat",
            "all",
        ],
        default="x86-causal",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget_hours", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat_seeds", default="43,44")
    parser.add_argument("--confirm_seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--selected_architecture",
        choices=["prefix_no_gine", "prefix_no_edges"],
        default="prefix_no_gine",
    )
    parser.add_argument(
        "--selected_representation",
        choices=sorted(REPRESENTATION_SPECS),
        default="prefix_no_gine",
    )
    parser.add_argument("--selected_prefix_density", type=int, default=4)
    parser.add_argument("--selected_gate_init", type=float, default=0.2)
    parser.add_argument("--ablation_seed", type=int, default=20260713)
    parser.add_argument(
        "--causal_architectures",
        default="prefix_no_gine,prefix_no_edges",
        help="Comma-separated trained architectures for inference-only causal controls.",
    )
    parser.add_argument("--capacity_values", default="2,6")
    parser.add_argument("--vector_values", default="2,4,8")
    parser.add_argument("--region_values", default="4,8,16")
    parser.add_argument("--prefix_density_values", default="2,4,6")
    parser.add_argument("--prefix_gate_values", default="0.1,0.2,0.3")
    parser.add_argument(
        "--isolation_variants",
        default=(
            "prefix_no_gine_no_attention,prefix_no_edges_gine2,"
            "prefix_no_gine_regions,prefix_no_gine_no_positions,"
            "prefix_no_gine_frozen_encoder"
        ),
        help="Comma-separated subset of x86-isolation stages.",
    )
    parser.add_argument(
        "--encoder_variants",
        default="prefix_no_gine_clap,prefix_no_gine_multivector4",
        help="Comma-separated subset of CLAP-ASM and multi-vector encoder controls.",
    )
    parser.add_argument(
        "--interaction_variants",
        default="prefix_no_gine_clap_frozen_encoder,prefix_cfg_clap",
        help="Comma-separated subset of the seed-42 encoder interaction cells.",
    )
    parser.add_argument("--capacity_winner", type=int)
    parser.add_argument("--metric_workers", type=int, default=64)
    parser.add_argument("--hf_repo", default="")
    parser.add_argument(
        "--wait_for_model",
        default="",
        help="Do not start the matrix until this model has a complete 154x10 result.",
    )
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--wait_timeout_hours", type=float, default=4.0)
    args = parser.parse_args()

    if args.phase == "x86-capacity-repeat" and args.capacity_winner is None:
        raise SystemExit("--capacity_winner is required for x86-capacity-repeat")
    if args.selected_prefix_density < 1:
        raise SystemExit("--selected_prefix_density must be positive")
    if not 0.0 < args.selected_gate_init < 1.0:
        raise SystemExit("--selected_gate_init must be between zero and one")
    if args.poll_seconds < 5:
        raise SystemExit("--poll_seconds must be at least 5")
    if args.wait_for_model:
        wait_for_result(
            args.wait_for_model,
            args.poll_seconds,
            args.wait_timeout_hours,
        )

    builders = {
        "x86-repeats": lambda: make_seed_repeats(args),
        "x86-causal": lambda: make_causal_controls(args),
        "x86-isolation": lambda: make_isolation(args),
        "x86-encoder": lambda: make_encoder_controls(args),
        "x86-interaction": lambda: make_interaction_controls(args),
        "x86-vector": lambda: make_vector_sweep(args),
        "x86-region": lambda: make_region_sweep(args),
        "x86-prefix-grid": lambda: make_prefix_grid(args),
        "x86-confirm": lambda: make_confirmatory_repeats(args),
        "x86-confirm-causal": lambda: make_confirmatory_causal(args),
        "x86-capacity": lambda: make_capacity(args),
        "x86-capacity-repeat": lambda: make_capacity(args, repeat=True),
    }
    if args.phase == "x86-comprehensive-screen":
        stages = []
        for phase in (
            "x86-isolation",
            "x86-encoder",
            "x86-vector",
            "x86-region",
        ):
            stages.extend(builders[phase]())
    elif args.phase == "all":
        stages = []
        for phase in (
            "x86-repeats",
            "x86-causal",
            "x86-isolation",
            "x86-encoder",
            "x86-capacity",
        ):
            stages.extend(builders[phase]())
    else:
        stages = builders[args.phase]()
    run_stages(args, stages)


if __name__ == "__main__":
    main()
