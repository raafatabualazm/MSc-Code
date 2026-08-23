#!/usr/bin/env python3
"""Resumable leakage-free x86 study runbook for one 96 GB GPU.

The default 20-hour profile prioritizes the causal architecture controls. It
does not train on either evaluation corpus (126 standalone rows or 154
HumanEval-Dart rows). GRPO is optional and uses a 256-task synthetic training
subset, then evaluates on all 154 untouched benchmark tasks.

Examples:
  python scripts/run_leakage_free_study.py --phase preflight --execute
  python scripts/run_leakage_free_study.py --phase gpu --execute
  python scripts/run_leakage_free_study.py --phase core --execute
  python scripts/run_leakage_free_study.py --phase expanded --execute
  python scripts/run_leakage_free_study.py --phase reward-preflight --execute
  python scripts/run_leakage_free_study.py --phase grpo --execute
  python scripts/run_leakage_free_study.py --phase all --execute --run-grpo

Without --execute the script prints the exact commands and estimated budget.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUNNER = ROOT / "configs" / "run_sweeps_antigravity.py"

EXPERIMENT = "qwen3-8b-base_lora_enc_dec_r16_5e6_gcb"
MODEL_STEM = "qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128"
DECODER_REV = "b968826d9c46dd6066d109eabc6255188de91218"
ENCODER_REV = "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"

SFT_TRAIN_REBUILT = "data/datasets/dart_all_rebuilt_train_unique.jsonl"
SFT_TRAIN_RAW = "data/datasets/dart_all_rebuilt_train_no_validation_overlap.jsonl"
SFT_VALID_RAW = "data/datasets/dart_all_rebuilt_validation_unique.jsonl"
BENCHMARK_RAW = "data/testing/grpo_data_cfg.jsonl"
SYN_INPUT_RAW = "data/datasets/synthetic_pool_clean_cfg.jsonl"
SFT_TRAIN = "data/datasets/dart_all_graphv2_train.jsonl"
SFT_VALID = "data/datasets/dart_all_graphv2_validation.jsonl"
BENCHMARK = "data/testing/grpo_data_graphv2.jsonl"
LEGACY_126 = "data/testing/compile-test2_cfg.jsonl"
SYN_INPUT = "data/datasets/synthetic_pool_graphv2.jsonl"
SYN_CLEAN = "data/datasets/synthetic_pool_reward_clean_graphv2.jsonl"
SYN_DIR = "data/datasets/synthetic_reward_graphv2_splits"


def audit_transfer_manifest() -> None:
    """Fail early when the clean-study archive omits a local dependency."""
    manifest_path = ROOT / "upload_clean_study_filelist.txt"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing transfer manifest: {manifest_path}")

    entries = {
        line.strip().replace("\\", "/")
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(entry for entry in entries if not (ROOT / entry).is_file())
    omitted_imports: set[tuple[str, str]] = set()

    for relpath in sorted(entry for entry in entries if entry.endswith(".py")):
        tree = ast.parse(
            (ROOT / relpath).read_text(encoding="utf-8"), filename=relpath
        )
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.split(".", 1)[0] not in {"models", "scripts", "configs"}:
                    continue
                candidate = module.replace(".", "/") + ".py"
                if (ROOT / candidate).is_file() and candidate not in entries:
                    omitted_imports.add((relpath, candidate))

    if missing or omitted_imports:
        details = []
        if missing:
            details.append("missing listed files:\n  " + "\n  ".join(missing))
        if omitted_imports:
            details.append(
                "unlisted local imports:\n  "
                + "\n  ".join(f"{owner} -> {dependency}" for owner, dependency in sorted(omitted_imports))
            )
        raise SystemExit("Transfer manifest audit failed:\n" + "\n".join(details))
    print(f"Transfer manifest audit passed: {len(entries)} files")
SYN_TRAIN = f"{SYN_DIR}/synthetic_reward_graphv2_train.jsonl"
SYN_VALID = f"{SYN_DIR}/synthetic_reward_graphv2_validation.jsonl"
SYN_TEST = f"{SYN_DIR}/synthetic_reward_graphv2_test.jsonl"
SYN_GRPO = f"{SYN_DIR}/synthetic_reward_graphv2_grpo256.jsonl"

CONFIRMATORY_ARMS = ("text", "prefix_no_edges", "prefix_cfg", "prefix_cfg_dfg")
PRIMARY_REPEAT_ARMS = ("prefix_no_edges", "prefix_cfg", "prefix_cfg_dfg")
PRIORITY_AUXILIARY_ARMS = ("prefix_shuffled", "prefix_cfg_dfg_text")
FULL_AUXILIARY_ARMS = ("prefix_shuffled", "prefix_cfg_dfg_text", "prefix_no_gine")
ARM_FLAGS = {
    "signature_only": (0, "none", 0, "off", "none", "identity"),
    "text": (0, "full", 1, "off", "none", "identity"),
    "prefix_no_edges": (64, "none", 0, "edges", "none", "full"),
    "prefix_shuffled": (64, "none", 0, "edges", "shuffle", "full"),
    "prefix_cfg": (64, "none", 0, "edges", "cfg", "full"),
    "prefix_cfg_dfg": (64, "none", 0, "edges", "full", "full"),
    "prefix_cfg_dfg_text": (64, "full", 1, "edges", "full", "full"),
    "prefix_no_gine": (64, "none", 0, "edges", "full", "identity"),
}

ESTIMATED_HOURS = {
    "preflight": 0.4,
    # Historical 154x10 generation pools took roughly 1-2.5 h. The runner now
    # reuses one pool for compile and pass metrics, but these remain cautious
    # RTX PRO 6000 estimates rather than H100/B200 timings.
    "reference": 1.1,
    "core_arm": 1.25,
    "aux_arm": 1.25,
    "expanded": 2.0,
    "reward_preflight": 0.5,
    "grpo": 4.0,
}


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_name(seed: int, arm: str, expanded: bool = False) -> str:
    suffix = f"_graphv2_clean_s{seed}_{arm}"
    if expanded:
        suffix += "_expanded"
    return MODEL_STEM + suffix


def common_args(seed: int, train_file: str, eval_file: str, epochs: int, lr: str, workers: int) -> list[str]:
    return [
        PYTHON, str(RUNNER),
        "--experiment", EXPERIMENT,
        "--encoder", "gcb",
        "--max_risk", "high",
        "--hardware_profile", "rtx6000",
        "--force_rerun",
        "--decoder_revision", DECODER_REV,
        "--encoder_revision", ENCODER_REV,
        "--seed", str(seed),
        "--train_file", train_file,
        "--eval_file", eval_file,
        "--compile_dataset", BENCHMARK,
        "--pass_dataset", BENCHMARK,
        "--compile_mode", "jit_tests",
        "--epochs", str(epochs),
        "--sft_lr", lr,
        "--lora_r", "64",
        "--lora_alpha", "128",
        "--load_4bit", "0",
        "--attn_implementation", "sdpa",
        "--gradient_checkpointing", "1",
        "--train_batch_size", "4",
        "--grad_accum", "16",
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
        "--generation_batch_size", "10",
        "--num_samples", "10",
        "--pass_num_samples", "10",
        "--metric_workers", str(workers),
        "--save_strategy", "epoch",
        "--save_total_limit", "2",
    ]


def arm_args(arm: str) -> list[str]:
    prefix, prompt_mode, clean_asm, dfg, edge, gnn = ARM_FLAGS[arm]
    return [
        "--qwen_prefix_tokens", str(prefix),
        "--qwen_prefix_dynamic", "1" if prefix > 0 else "0",
        "--qwen_prefix_min_tokens", "4",
        "--qwen_prefix_tokens_per_log2", "4",
        "--prompt_assembly_mode", prompt_mode,
        "--prompt_fit_assembly", "0" if prompt_mode == "none" else "1",
        "--prompt_clean_asm", str(clean_asm),
        "--dfg_mode", dfg,
        "--edge_ablation", edge,
        "--gnn_ablation", gnn,
        "--add_reverse_edges", "1",
        "--block_position_mode", "sinusoidal",
    ]


def summary_path(name: str, grpo: bool = False) -> Path:
    suffix = "_grpo" if grpo else ""
    return ROOT / "results" / "sweeps_antigravity" / f"{name}{suffix}.json"


def predictions_paths(name: str, grpo: bool = False) -> tuple[Path, Path]:
    suffix = "_grpo" if grpo else ""
    return (
        ROOT / "results" / f"{name}{suffix}_compile_predictions.json",
        ROOT / "results" / f"{name}{suffix}_pass_predictions.json",
    )


def checkpoint_path(name: str, grpo: bool = False) -> Path:
    suffix = "_grpo" if grpo else ""
    return ROOT / "artifacts" / f"{name}{suffix}" / "pytorch_model.bin"


def valid_provenance(prediction: Path, seed: int) -> bool:
    sidecar = Path(str(prediction) + ".provenance.json")
    if not prediction.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("scoring_tests_visible_to_policy") is False
        and int(payload.get("seed", -1)) == seed
        and payload.get("prompt_schema_version") == "antigravity-v2-no-test-hints"
        and payload.get("generation", {}).get("use_cache") is True
        and payload.get("generation", {}).get("decoder_gradient_checkpointing") is False
    )


def complete(name: str, seed: int, grpo: bool = False) -> bool:
    if not summary_path(name, grpo).is_file():
        return False
    return all(valid_provenance(path, seed) for path in predictions_paths(name, grpo))


class Runbook:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.started = time.monotonic()
        self.log_dir = ROOT / "logs" / "leakage_free_graphv2"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.log_dir / "state.json"
        self.state: dict[str, Any] = {"completed": {}, "failed": {}}
        if self.state_path.is_file():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        self.planned_hours = 0.0

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def require_preflight(self, *, full_reward_parity: bool) -> None:
        def load_report(path: Path) -> dict[str, Any]:
            if not path.is_file():
                raise SystemExit(f"Required preflight report is missing: {path}")
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Invalid preflight report {path}: {exc}") from exc

        graph_reports = (
            ("data/testing/grpo_data_graphv2.summary.json", BENCHMARK, 154),
            ("data/datasets/dart_all_graphv2_train.summary.json", SFT_TRAIN, 770),
            ("data/datasets/dart_all_graphv2_validation.summary.json", SFT_VALID, 83),
            ("data/datasets/synthetic_pool_graphv2.summary.json", SYN_INPUT, 1726),
        )
        for report_path, dataset_path, expected_rows in graph_reports:
            report = load_report(ROOT / report_path)
            if (
                report.get("graph_schema_version") != "antigravity-graph-v2"
                or report.get("output_rows") != expected_rows
                or report.get("max_block_instrs") != 20
                or report.get("max_dataflow_edges") != 0
                or report.get("output_sha256") != file_sha256(ROOT / dataset_path)
            ):
                raise SystemExit(
                    f"Graph-v2 build report is stale or failed: {report_path}"
                )

        graph_audit = load_report(ROOT / "results/graph_v2_dataset_audit.json")
        if (
            graph_audit.get("passed") is not True
            or graph_audit.get("max_block_instrs") != 20
            or graph_audit.get("max_code_tokens") != 510
        ):
            raise SystemExit("Graph-v2 structural/tokenizer audit failed")
        split_filter = load_report(
            ROOT / "results/sft_source_train_validation_filter_graphv2.json"
        )
        if (
            split_filter.get("filtered_train_rows") != 770
            or split_filter.get("dropped_train_rows") != 3
            or split_filter.get("train_sha256")
            != file_sha256(ROOT / SFT_TRAIN_REBUILT)
            or split_filter.get("benchmark_sha256")
            != file_sha256(ROOT / SFT_VALID_RAW)
            or split_filter.get("filtered_train_sha256")
            != file_sha256(ROOT / SFT_TRAIN_RAW)
        ):
            raise SystemExit("SFT train/validation near-overlap filter is stale or failed")
        audited_hashes = {
            item.get("path", "").replace("\\", "/"): item.get("sha256")
            for item in graph_audit.get("datasets", [])
        }
        for dataset_path in (BENCHMARK, SFT_TRAIN, SFT_VALID, SYN_INPUT):
            if audited_hashes.get(dataset_path) != file_sha256(ROOT / dataset_path):
                raise SystemExit(
                    f"Graph-v2 audit is stale for {dataset_path}; rerun --phase preflight"
                )

        complexity = load_report(ROOT / "results/graph_v2_complexity_audit.json")
        complexity_hashes = {
            item.get("path", "").replace("\\", "/"): item.get("sha256")
            for item in complexity.get("datasets", [])
        }
        for dataset_path in (BENCHMARK, SFT_TRAIN, SFT_VALID, SYN_INPUT):
            if complexity_hashes.get(dataset_path) != file_sha256(ROOT / dataset_path):
                raise SystemExit(
                    f"Graph complexity audit is stale for {dataset_path}; rerun --phase preflight"
                )

        prompt = load_report(ROOT / "results/protocol_audit_154_graphv2.json")
        if (
            prompt.get("passed") is not True
            or prompt.get("rows_leaking_scoring_tests") != 0
            or prompt.get("prompt_schema_version") != "antigravity-v2-no-test-hints"
            or prompt.get("dataset", {}).get("sha256") != file_sha256(ROOT / BENCHMARK)
        ):
            raise SystemExit("Prompt-integrity preflight is stale or failed; rerun --phase preflight")

        for filename in (
            "sft_benchmark_overlap_graphv2.json",
            "sft_train_validation_overlap_graphv2.json",
            "synthetic_benchmark_overlap_graphv2.json",
        ):
            overlap = load_report(ROOT / "results" / filename)
            if (
                overlap.get("exact_overlap_pairs") != 0
                or overlap.get("near_overlap_pairs") != 0
                or overlap.get("duplicate_train_source_rows") != 0
            ):
                raise SystemExit(f"Dataset-overlap preflight failed: {filename}")

        manifest = load_report(ROOT / SYN_DIR / "manifest.json")
        if manifest.get("rows") != 1725 or manifest.get("output_sha256") != file_sha256(ROOT / SYN_CLEAN):
            raise SystemExit("Synthetic manifest is stale or failed; rerun --phase preflight")
        for split_name, split_path in (
            ("train", SYN_TRAIN), ("validation", SYN_VALID), ("test", SYN_TEST)
        ):
            expected = manifest.get("splits", {}).get(split_name, {}).get("sha256")
            if not expected or expected != file_sha256(ROOT / split_path):
                raise SystemExit(f"Synthetic {split_name} split hash does not match the manifest")

        schema = load_report(ROOT / SYN_DIR / "schema_validation.json")
        static_reward = load_report(ROOT / SYN_DIR / "reward_static_audit.json")
        if schema.get("rows") != 1725 or schema.get("problems"):
            raise SystemExit("Synthetic schema validation failed")
        if static_reward.get("static_failures") != 0:
            raise SystemExit("Synthetic reward static audit failed")

        if full_reward_parity:
            parity = load_report(ROOT / SYN_DIR / "reward_full_parity.json")
            result = parity.get("reference_reward_parity", {})
            if result.get("checked") != 1725 or result.get("passed") != 1725 or result.get("failures"):
                raise SystemExit("Full executable reward parity failed; do not start GRPO/GPU phase")

        print("Preflight reports and dataset hashes: VERIFIED")

    def run(self, stage: str, command: list[str], estimate: float, *, skip: bool = False) -> None:
        if (
            self.args.hf_repo
            and len(command) > 1
            and Path(command[1]).resolve() == RUNNER.resolve()
            and "--hf_repo" not in command
        ):
            command = [
                *command,
                "--hf_repo", self.args.hf_repo,
                "--hf_private", str(self.args.hf_private),
                "--hf_upload_checkpoints", "1",
            ]
        self.planned_hours += estimate
        print(f"\n=== {stage} (estimate {estimate:.2f} h) ===")
        print(shell_join(command))
        if skip:
            print("SKIP: valid summary and test-free provenance already exist")
            return
        if not self.args.execute:
            return
        elapsed_hours = (time.monotonic() - self.started) / 3600
        if elapsed_hours + estimate > self.args.budget_hours:
            raise SystemExit(
                f"Stopping before {stage}: elapsed {elapsed_hours:.2f} h + estimate {estimate:.2f} h "
                f"exceeds budget {self.args.budget_hours:.2f} h"
            )
        log_path = self.log_dir / f"{stage}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + shell_join(command) + "\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
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
                log.write(line)
            code = process.wait()
        if code != 0:
            self.state.setdefault("failed", {})[stage] = {"exit_code": code, "log": str(log_path)}
            self.save_state()
            raise SystemExit(f"Stage failed ({code}): {stage}; see {log_path}")
        self.state.setdefault("completed", {})[stage] = {"log": str(log_path), "time": time.time()}
        self.save_state()

    def preflight(self) -> None:
        audit_transfer_manifest()
        required = [SFT_TRAIN_REBUILT, SFT_VALID_RAW, BENCHMARK_RAW, SYN_INPUT_RAW]
        missing = [path for path in required if not (ROOT / path).is_file()]
        if missing:
            raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

        compile_files = [
            "scripts/training/graph_grpo_decompiler_antigravity.py",
            "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
            "scripts/evaluation/graph_pass_at_k_antigravity.py",
            "scripts/evaluation/graph_compile_at_k_antigravity.py",
            "scripts/evaluation/compile_statistical_results_antigravity.py",
            "scripts/data/prepare_synthetic_reward_pool.py",
            "scripts/data/rebuild_sft_assembly.py",
            "scripts/data/merge_jsonl_by_source.py",
            "scripts/data/build_graph_v2_jsonl.py",
            "scripts/data/audit_graph_v2_dataset.py",
            "scripts/data/analyze_graph_complexity_v2.py",
            "scripts/data/cfg_extractor.py",
            "scripts/data/dfg_extractor.py",
            "scripts/data/test_graph_v2_integrity.py",
            "scripts/data/test_graph_preprocessing_fixes.py",
            "scripts/data/test_graph_architecture_ablation.py",
            "scripts/training/audit_grpo_reward_antigravity.py",
            "configs/run_sweeps_antigravity.py",
            "generate_synthetic_tasks.py",
            "generate_synthetic_tasks_parallel.py",
            "scripts/run_leakage_free_study.py",
            "scripts/run_graphv2_followups.py",
            "scripts/run_arm64_graphv21_study.py",
            "scripts/evaluation/analyze_graphv2_clean_study.py",
            "scripts/evaluation/analyze_arm64_graphv21_study.py",
            "scripts/evaluation/test_graph_input_ablation_antigravity.py",
        ]
        commands = [
            [PYTHON, "-m", "py_compile", *compile_files],
            [PYTHON, "-m", "unittest", "scripts.evaluation.test_protocol_integrity_antigravity"],
            [PYTHON, "-m", "unittest", "scripts.evaluation.test_graph_input_ablation_antigravity"],
            [PYTHON, "-m", "unittest", "scripts.data.test_graph_v2_integrity"],
            [PYTHON, "-m", "unittest", "scripts.data.test_graph_architecture_ablation"],
            [PYTHON, "scripts/training/grpo_selfcheck.py"],
            [PYTHON, "generate_synthetic_tasks_parallel.py", "--self-test"],
            [PYTHON, "scripts/data/audit_dataset_overlap_antigravity.py",
             "--train", SFT_TRAIN_REBUILT, "--benchmark", SFT_VALID_RAW,
             "--output", "results/sft_source_train_validation_filter_graphv2.json",
             "--filtered_train", SFT_TRAIN_RAW,
             "--expected_filtered_rows", "770"],
            [PYTHON, "scripts/data/build_graph_v2_jsonl.py",
             "--input", BENCHMARK_RAW, "--output", BENCHMARK,
             "--rejected", "data/testing/grpo_data_graphv2.rejected.jsonl",
             "--summary", "data/testing/grpo_data_graphv2.summary.json",
             "--expected_input_rows", "154", "--expected_output_rows", "154",
             "--max_block_instrs", "20", "--max_dataflow_edges", "0"],
            [PYTHON, "scripts/data/build_graph_v2_jsonl.py",
             "--input", SFT_TRAIN_RAW, "--output", SFT_TRAIN,
             "--rejected", "data/datasets/dart_all_graphv2_train.rejected.jsonl",
             "--summary", "data/datasets/dart_all_graphv2_train.summary.json",
             "--expected_input_rows", "770", "--expected_output_rows", "770",
             "--max_block_instrs", "20",
             "--max_dataflow_edges", "0"],
            [PYTHON, "scripts/data/build_graph_v2_jsonl.py",
             "--input", SFT_VALID_RAW, "--output", SFT_VALID,
             "--rejected", "data/datasets/dart_all_graphv2_validation.rejected.jsonl",
             "--summary", "data/datasets/dart_all_graphv2_validation.summary.json",
             "--expected_input_rows", "83", "--expected_output_rows", "83",
             "--max_block_instrs", "20",
             "--max_dataflow_edges", "0"],
            [PYTHON, "scripts/data/build_graph_v2_jsonl.py",
             "--input", SYN_INPUT_RAW, "--output", SYN_INPUT,
             "--rejected", "data/datasets/synthetic_pool_graphv2.rejected.jsonl",
             "--summary", "data/datasets/synthetic_pool_graphv2.summary.json",
             "--expected_input_rows", "1726", "--expected_output_rows", "1726",
             "--max_block_instrs", "20", "--max_dataflow_edges", "0"],
            [PYTHON, "scripts/data/test_graph_preprocessing_fixes.py"],
            [PYTHON, "scripts/data/audit_graph_v2_dataset.py",
             "--dataset", f"{BENCHMARK}=154",
             "--dataset", f"{SFT_TRAIN}=770",
             "--dataset", f"{SFT_VALID}=83",
             "--dataset", f"{SYN_INPUT}=1726",
             "--max_block_instrs", "20",
             "--tokenizer_revision", ENCODER_REV,
             "--output", "results/graph_v2_dataset_audit.json"],
            [PYTHON, "scripts/data/analyze_graph_complexity_v2.py",
             "--dataset", BENCHMARK,
             "--dataset", SFT_TRAIN,
             "--dataset", SFT_VALID,
             "--dataset", SYN_INPUT,
             "--output", "results/graph_v2_complexity_audit.json",
             "--top", "20"],
            [PYTHON, "scripts/evaluation/audit_prompt_integrity_antigravity.py",
             "--dataset", BENCHMARK, "--output", "results/protocol_audit_154_graphv2.json"],
            [PYTHON, "scripts/data/audit_dataset_overlap_antigravity.py",
             "--train", SFT_TRAIN, "--benchmark", BENCHMARK,
             "--output", "results/sft_benchmark_overlap_graphv2.json"],
            [PYTHON, "scripts/data/audit_dataset_overlap_antigravity.py",
             "--train", SFT_TRAIN, "--benchmark", SFT_VALID,
             "--output", "results/sft_train_validation_overlap_graphv2.json"],
            [PYTHON, "scripts/data/prepare_synthetic_reward_pool.py",
             "--input", SYN_INPUT, "--output", SYN_CLEAN,
             "--split_dir", SYN_DIR, "--split_prefix", "synthetic_reward_graphv2",
             "--seed", "42", "--drop_invalid", "--report", f"{SYN_DIR}/manifest.json"],
            [PYTHON, "scripts/data/audit_dataset_overlap_antigravity.py",
             "--train", SYN_CLEAN, "--benchmark", BENCHMARK,
             "--output", "results/synthetic_benchmark_overlap_graphv2.json"],
            [PYTHON, "scripts/data/style_match_synthetic_pool.py",
             "--synthetic", SYN_TRAIN, "--real", BENCHMARK,
             "--output", SYN_GRPO, "--summary", f"{SYN_GRPO}.summary.json",
             "--target_rows", "256", "--max_source_words", "220",
             "--max_asm_lines", "700", "--min_expects", "4",
             "--allow_extra_return_types", "1", "--seed", "42"],
            [PYTHON, "scripts/training/audit_grpo_reward_antigravity.py",
             "--dataset", SYN_CLEAN, "--report", f"{SYN_DIR}/reward_static_audit.json"],
            [PYTHON, "scripts/data/validate_synthetic_pool.py",
             "--pool", SYN_CLEAN, "--eval_data", BENCHMARK,
             "--run_tests", "0", "--report", f"{SYN_DIR}/schema_validation.json"],
        ]
        per_command = ESTIMATED_HOURS["preflight"] / len(commands)
        for index, command in enumerate(commands, start=1):
            self.run(f"preflight_{index:02d}", command, per_command)

        if self.args.full_reward_audit or self.args.phase == "all":
            self.run(
                "preflight_full_reward_parity",
                [PYTHON, "scripts/training/audit_grpo_reward_antigravity.py",
                 "--dataset", SYN_CLEAN, "--run_references", "-1",
                 "--workers", str(self.args.reward_workers),
                 "--report", f"{SYN_DIR}/reward_full_parity.json"],
                0.6,
            )

    def reference(self, seed: int) -> None:
        name = model_name(seed, "untuned")
        command = common_args(seed, SFT_TRAIN, SFT_VALID, 4, "5e-6", self.args.metric_workers)
        command += [
            "--name_suffix", f"_graphv2_clean_s{seed}_untuned",
            "--skip_training",
            *arm_args("text"),
        ]
        preds = predictions_paths(name)
        if not complete(name, seed) and all(valid_provenance(path, seed) for path in preds):
            command.append("--skip_inference")
        self.run(f"reference_s{seed}", command, ESTIMATED_HOURS["reference"], skip=complete(name, seed))

    def signature_reference(self, seed: int) -> None:
        name = model_name(seed, "signature_only_base")
        command = common_args(seed, SFT_TRAIN, SFT_VALID, 4, "5e-6", self.args.metric_workers)
        command += [
            "--name_suffix", f"_graphv2_clean_s{seed}_signature_only_base",
            "--skip_training",
            *arm_args("signature_only"),
        ]
        preds = predictions_paths(name)
        if not complete(name, seed) and all(valid_provenance(path, seed) for path in preds):
            command.append("--skip_inference")
        self.run(
            f"signature_reference_s{seed}", command, ESTIMATED_HOURS["reference"],
            skip=complete(name, seed),
        )

    def core_arm(self, seed: int, arm: str, auxiliary: bool = False) -> None:
        name = model_name(seed, arm)
        command = common_args(seed, SFT_TRAIN, SFT_VALID, 4, "5e-6", self.args.metric_workers)
        command += ["--name_suffix", f"_graphv2_clean_s{seed}_{arm}", *arm_args(arm)]
        checkpoint = checkpoint_path(name)
        preds = predictions_paths(name)
        if checkpoint.is_file() and not complete(name, seed):
            command += ["--skip_training", "--sft_checkpoint", str(checkpoint)]
            if all(valid_provenance(path, seed) for path in preds):
                command.append("--skip_inference")
        estimate = ESTIMATED_HOURS["aux_arm" if auxiliary else "core_arm"]
        self.run(f"core_s{seed}_{arm}", command, estimate, skip=complete(name, seed))

    def expanded(self, seed: int) -> None:
        arm = "prefix_cfg_dfg"
        name = model_name(seed, arm, expanded=True)
        command = common_args(
            seed,
            f"{SFT_TRAIN},{SYN_TRAIN}",
            f"{SFT_VALID},{SYN_VALID}",
            2,
            "2e-6",
            self.args.metric_workers,
        )
        command += ["--name_suffix", f"_graphv2_clean_s{seed}_{arm}_expanded", *arm_args(arm)]
        checkpoint = checkpoint_path(name)
        preds = predictions_paths(name)
        if checkpoint.is_file() and not complete(name, seed):
            command += ["--skip_training", "--sft_checkpoint", str(checkpoint)]
            if all(valid_provenance(path, seed) for path in preds):
                command.append("--skip_inference")
        self.run(f"expanded_s{seed}", command, ESTIMATED_HOURS["expanded"], skip=complete(name, seed))

    def grpo_command(self, seed: int, preflight_batches: int = 0) -> tuple[str, list[str]]:
        source_name = model_name(seed, "prefix_cfg_dfg", expanded=True)
        source_checkpoint = checkpoint_path(source_name)
        if not source_checkpoint.is_file() and self.args.execute:
            raise SystemExit(f"Expanded SFT checkpoint required before GRPO: {source_checkpoint}")
        suffix = f"_graphv2_clean_s{seed}_cfgdfg_expanded_synrl"
        if preflight_batches:
            suffix += "_reward_preflight"
        name = MODEL_STEM + suffix
        command = [
            PYTHON, str(RUNNER),
            "--experiment", EXPERIMENT,
            "--name_suffix", suffix,
            "--encoder", "gcb", "--max_risk", "high", "--hardware_profile", "rtx6000",
            "--force_rerun", "--use_grpo",
            "--grpo_checkpoint", str(source_checkpoint),
            "--grpo_train_file", SYN_GRPO,
            "--compile_dataset", BENCHMARK, "--pass_dataset", BENCHMARK,
            "--compile_mode", "jit_tests",
            "--decoder_revision", DECODER_REV, "--encoder_revision", ENCODER_REV,
            "--seed", str(seed), "--lora_r", "64", "--lora_alpha", "128",
            "--load_4bit", "0", "--attn_implementation", "sdpa",
            "--qwen_prefix_tokens", "64", "--qwen_prefix_gate_init", "0.2",
            "--qwen_prefix_gate_mode", "token",
            "--qwen_prefix_dynamic", "1", "--qwen_prefix_min_tokens", "4",
            "--qwen_prefix_tokens_per_log2", "4",
            "--qwen_prefix_rms_match", "1", "--prompt_assembly_mode", "none",
            "--decoder_prompt_max_length", "2048", "--prompt_fit_assembly", "0",
            "--auto_cfg", "0", "--max_block_instrs", "20", "--dfg_mode", "edges",
            "--edge_ablation", "full", "--gnn_ablation", "full",
            "--add_reverse_edges", "1", "--block_position_mode", "sinusoidal",
            "--position_scheme", "roberta", "--causal_position_ids", "cumsum",
            "--use_reasoning", "0",
            "--grpo_group_size", "16", "--grpo_score_chunk_size", "1",
            "--grpo_gen_top_p", "1.0",
            "--grpo_epochs", "1", "--grpo_lr", "5e-7",
            "--grpo_reward_mode", "binary", "--grpo_binary_fail_reward", "-1.0",
            "--grpo_perfect_base_reward", "1.0", "--grpo_perfect_bonus", "0.0",
            "--grpo_kl_coef", "0.0", "--grpo_clip_eps", "0.15",
            "--grpo_adv_norm", "mean", "--grpo_loss_pooling", "seq",
            "--grpo_simko_k", "0", "--grpo_passk_k", "0",
            "--grpo_unique_test_bonus", "0.0", "--grpo_duplicate_penalty", "0.0",
            "--grpo_entropy_coef", "0.0", "--grpo_overlong_filter", "1",
            "--grpo_max_new_tokens", "768", "--grpo_test_timeout", "8",
            "--grpo_reward_workers", str(self.args.reward_workers),
            "--train_batch_size", "1", "--grad_accum", "8",
            "--eval_max_new_tokens", "768", "--generation_batch_size", "10",
            "--num_samples", "10", "--pass_num_samples", "10",
            "--metric_workers", str(self.args.metric_workers),
            "--save_strategy", "epoch", "--save_total_limit", "2",
        ]
        if preflight_batches:
            command += ["--grpo_reward_preflight_batches", str(preflight_batches)]
        return name, command

    def reward_preflight(self, seed: int) -> None:
        name, command = self.grpo_command(seed, self.args.reward_preflight_batches)
        report = ROOT / "artifacts" / f"{name}_grpo" / "reward_preflight.json"
        skip = False
        if report.is_file():
            try:
                existing = json.loads(report.read_text(encoding="utf-8"))
                self._validate_reward_preflight(existing, seed, report)
                skip = True
            except (SystemExit, json.JSONDecodeError, OSError) as exc:
                print(f"Stale reward preflight will be regenerated: {exc}")
        self.run(
            f"reward_preflight_s{seed}", command, ESTIMATED_HOURS["reward_preflight"],
            skip=skip,
        )
        if report.is_file():
            data = json.loads(report.read_text(encoding="utf-8"))
            print("Reward preflight:", json.dumps(data, indent=2))
            self._validate_reward_preflight(data, seed, report)
            if data.get("signal_group_rate_mean", 0.0) < self.args.min_signal_group_rate:
                raise SystemExit(
                    "GRPO not approved: signal_group_rate_mean is below "
                    f"{self.args.min_signal_group_rate:.3f}. Use RS-SFT/distillation instead."
                )

    def _validate_reward_preflight(self, data: dict[str, Any], seed: int, report: Path) -> None:
        source_checkpoint = checkpoint_path(model_name(seed, "prefix_cfg_dfg", expanded=True))
        expected = {
            "seed": seed,
            "group_size": 16,
            "reward_mode": "binary",
            "prompt_schema_version": "antigravity-v2-no-test-hints",
            "no_optimizer_update": True,
            "pass_stability_runs": 3,
        }
        mismatches = [key for key, value in expected.items() if data.get(key) != value]
        if data.get("checkpoint", {}).get("sha256") != file_sha256(source_checkpoint):
            mismatches.append("checkpoint.sha256")
        if data.get("dataset", {}).get("sha256") != file_sha256(ROOT / SYN_GRPO):
            mismatches.append("dataset.sha256")
        implementation = data.get("reward_implementation", {})
        if implementation.get("trainer", {}).get("sha256") != file_sha256(
            ROOT / "scripts" / "training" / "graph_grpo_decompiler_antigravity.py"
        ):
            mismatches.append("reward_implementation.trainer.sha256")
        if implementation.get("jit_evaluator", {}).get("sha256") != file_sha256(
            ROOT / "scripts" / "evaluation" / "graph_compile_at_k_antigravity.py"
        ):
            mismatches.append("reward_implementation.jit_evaluator.sha256")
        if data.get("reward") != {"perfect": 1.0, "binary_fail": -1.0}:
            mismatches.append("reward")
        generation = data.get("generation", {})
        if generation != {"temperature": 0.7, "top_p": 1.0, "max_new_tokens": 768}:
            mismatches.append("generation")
        if mismatches:
            raise SystemExit(
                f"GRPO not approved: stale/mismatched reward preflight {report}: "
                + ", ".join(mismatches)
            )

    def grpo(self, seed: int) -> None:
        preflight_name, _ = self.grpo_command(seed, self.args.reward_preflight_batches)
        report = ROOT / "artifacts" / f"{preflight_name}_grpo" / "reward_preflight.json"
        if not report.is_file():
            raise SystemExit(
                "GRPO not approved: the matching no-update reward preflight report is missing. "
                "Run --phase reward-preflight first. Expected: " + str(report)
            )
        try:
            preflight = json.loads(report.read_text(encoding="utf-8"))
            signal_rate = float(preflight["signal_group_rate_mean"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"GRPO not approved: invalid reward preflight report {report}: {exc}") from exc
        self._validate_reward_preflight(preflight, seed, report)
        if signal_rate < self.args.min_signal_group_rate:
            raise SystemExit(
                "GRPO not approved: signal_group_rate_mean "
                f"{signal_rate:.3f} is below {self.args.min_signal_group_rate:.3f}. "
                "Use RS-SFT/distillation instead."
            )
        name, command = self.grpo_command(seed, 0)
        self.run(f"grpo_s{seed}", command, ESTIMATED_HOURS["grpo"], skip=complete(name, seed, grpo=True))


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--phase", choices=["plan", "preflight", "gpu", "core", "expanded", "reward-preflight", "grpo", "all"], default="plan")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--budget_hours", type=float, default=20.0)
    parser.add_argument("--metric_workers", type=int, default=max(1, min(64, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--reward_workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--full_matrix", action="store_true", help="Run auxiliary controls for every seed, not seed 42 only")
    parser.add_argument("--run_grpo", action="store_true", help="Include GRPO after its reward preflight in --phase all")
    parser.add_argument("--full_reward_audit", action="store_true", help="Execute all 1,725 reference rewards during preflight")
    parser.add_argument("--hf_repo", default=os.environ.get("HF_OUT_REPO", ""),
                        help="Optional Hugging Face repo for automatic epoch checkpoints and results")
    parser.add_argument("--hf_private", type=int, choices=[0, 1], default=1)
    parser.add_argument("--reward_preflight_batches", type=int, default=8)
    parser.add_argument("--min_signal_group_rate", type=float, default=0.05)
    args = parser.parse_args()

    # Confirmatory graph-v2 runs fail closed and retain every DFG edge. A
    # positive cap would reintroduce source-order selection bias.
    os.environ["GRAPH_STRICT_GRAPH"] = "1"
    os.environ["GRPO_STRICT_GRAPH"] = "1"
    os.environ["GRAPH_MAX_DATAFLOW_EDGES"] = "0"
    os.environ["GRAPH_ADD_REVERSE_EDGES"] = "1"
    os.environ["GRAPH_BLOCK_POSITION_MODE"] = "sinusoidal"
    # Re-run only initially successful candidates. This rejects stochastic
    # programs that happen to satisfy deterministic tests once (for example,
    # generated Random().nextBool() tie-breaking) without tripling all Dart
    # executions.
    os.environ["EVAL_PASS_STABILITY_RUNS"] = "3"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    seeds = csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("At least one seed is required")
    runbook = Runbook(args)

    if args.phase == "all":
        phases = {"preflight", "core", "expanded", "reward-preflight"}
    elif args.phase == "gpu":
        phases = {"core", "expanded", "reward-preflight"}
    else:
        phases = {args.phase}
    if args.phase == "plan":
        phases = {"preflight", "core", "expanded", "reward-preflight"}

    if "preflight" in phases:
        runbook.preflight()
    paid_or_training_phase = bool(phases & {"core", "expanded", "reward-preflight"}) or args.phase == "grpo"
    if args.execute and paid_or_training_phase:
        runbook.require_preflight(
            full_reward_parity=("reward-preflight" in phases or args.phase == "grpo")
        )
    if "core" in phases:
        for seed_index, seed in enumerate(seeds):
            # One 20-hour reservation prioritizes the causal graph comparison.
            # Raw-base/text baselines and auxiliary controls run once; the
            # no-edge/CFG/CFG+DFG arms receive independent training seeds.
            if seed_index == 0 or args.full_matrix:
                runbook.reference(seed)
                runbook.signature_reference(seed)
                arms = CONFIRMATORY_ARMS
            else:
                arms = PRIMARY_REPEAT_ARMS
            for arm in arms:
                runbook.core_arm(seed, arm)
            if args.full_matrix or seed == seeds[0]:
                auxiliary_arms = FULL_AUXILIARY_ARMS if args.full_matrix else PRIORITY_AUXILIARY_ARMS
                for arm in auxiliary_arms:
                    runbook.core_arm(seed, arm, auxiliary=True)
    if "expanded" in phases:
        runbook.expanded(seeds[0])
    if "reward-preflight" in phases:
        runbook.reward_preflight(seeds[0])
    if args.phase == "grpo" or (args.phase in {"all", "gpu"} and args.run_grpo):
        runbook.grpo(seeds[0])

    print(f"\nPlanned stage estimate: {runbook.planned_hours:.2f} GPU/wall-clock hours")
    if not args.execute:
        print("Dry plan only. Add --execute to run it.")


if __name__ == "__main__":
    main()
