"""Regression tests for leakage and evaluator-schema symmetry."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.pyg_cfg_dataset import cfg_to_pyg
from configs.run_sweeps_antigravity import copy_prediction_pool
from scripts.run_graphv2_followups import (
    CLAP_ENCODER_REV,
    CLAP_EXPERIMENT,
    CLAP_FROZEN_ENCODER_EXPERIMENT,
    CLAP_FROZEN_ENCODER_MODEL_STEM,
    CLAP_MODEL_STEM,
    make_encoder_controls,
    make_interaction_controls,
    make_isolation,
    make_prefix_grid,
    make_vector_sweep,
    result_complete,
)
from scripts.run_leakage_free_study import EXPERIMENT
from scripts.run_arm64_graphv21_study import (
    SELECTED_ARCHITECTURES as ARM64_SELECTED_ARCHITECTURES,
    selected_label as arm64_selected_label,
)
from scripts.evaluation.codebleu import CodeBLEUCalculator
from scripts.evaluation.graph_pass_at_k_antigravity import should_force_zero
from scripts.evaluation.graph_compile_at_k_antigravity import (
    _is_dart_jit_static_error,
    dart_test_completion_observed,
    disallowed_dart_test_runtime_library,
    evaluate_dart_jit_tests_detail,
    prepare_dart_test_completion_attestation,
)
from scripts.evaluation.graph_inference_antigravity import GraphInferenceModel
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
    PROMPT_SCHEMA_VERSION,
    build_decoder_prompt,
)


class PromptIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "language": "Dart",
            "signature": "int example(int x)",
            "assembly": "mov eax, edi",
            "tests": "expect(candidate(7), 42);",
        }

    def test_default_prompt_hides_scoring_tests(self) -> None:
        prompt = build_decoder_prompt(self.row)
        self.assertNotIn("candidate(7)", prompt)
        self.assertNotIn("42", prompt)
        self.assertNotIn("Unit-test harness excerpt", prompt)

    def test_oracle_hint_requires_explicit_opt_in_and_label(self) -> None:
        prompt = build_decoder_prompt(self.row, include_oracle_test_hints=True)
        self.assertIn("ORACLE DIAGNOSTIC ONLY", prompt)
        self.assertIn("expect(candidate(7), 42);", prompt)

    def test_prompt_schema_records_matched_function_contract(self) -> None:
        self.assertEqual(PROMPT_SCHEMA_VERSION, "antigravity-v3-matched-function-contract")

    def test_frozen_exact_comparator_prompt_stream_is_byte_preserved(self) -> None:
        dataset = Path(__file__).resolve().parents[2] / "data/testing/grpo_data_graphv2.jsonl"
        rows = [
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        digest = hashlib.sha256()
        with patch.dict(
            os.environ,
            {
                "GRAPH_PROMPT_ASSEMBLY_MODE": "none",
                "GRAPH_QWEN_PREFIX_TOKENS": "64",
                "GRAPH_USE_REASONING": "0",
            },
        ):
            for index, row in enumerate(rows):
                prompt = build_decoder_prompt(row, None, 2048)
                digest.update(
                    json.dumps(
                        [str(row.get("task_id", index)), prompt],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
        self.assertEqual(
            digest.hexdigest(),
            "55adb80e8a24df956c82c2eed260523a2f6c1b6e00a566bfb7b269c7eab75d0d",
        )

    def test_exact_and_name_only_prompts_share_structural_constraints(self) -> None:
        row = {
            **self.row,
            "function": "candidate",
            "signature": "int candidate(int value)",
        }
        exact_prompt = build_decoder_prompt({**row, "prompt_signature_mode": "exact"})
        name_only_prompt = build_decoder_prompt(
            {**row, "prompt_signature_mode": "name_only"}
        )
        shared_constraints = (
            "Do not replace it with only a void main() demo.",
            "Do not define the required function inside main(); define it at top level.",
        )

        for constraint in shared_constraints:
            with self.subTest(constraint=constraint):
                self.assertEqual(exact_prompt.count(constraint), 1)
                self.assertEqual(name_only_prompt.count(constraint), 1)

    def test_name_only_prompt_requires_exact_name_and_infers_hidden_interface(self) -> None:
        hidden_signature = (
            "Map<String, int> semanticSecret("
            "List<double> sensitiveValues, bool hiddenFlag)"
        )
        prompt = build_decoder_prompt({
            "language": "Dart",
            "function": "candidate",
            "signature": hidden_signature,
            "prompt_signature_mode": "name_only",
            "assembly": "mov rax, rdi",
        })

        self.assertIn(
            "Implement a top-level Dart function named exactly candidate.", prompt
        )
        self.assertIn(
            "Infer the return type and complete parameter list "
            "(types, order, and arity) from the binary representation.",
            prompt,
        )
        self.assertNotIn("Implement this exact top-level Dart signature", prompt)
        self.assertNotIn(hidden_signature, prompt)
        self.assertNotIn("semanticSecret", prompt)
        self.assertNotIn("sensitiveValues", prompt)
        self.assertNotIn("hiddenFlag", prompt)

    def test_signature_only_control_does_not_claim_a_graph_channel(self) -> None:
        with patch.dict(
            "os.environ",
            {"GRAPH_PROMPT_ASSEMBLY_MODE": "none", "GRAPH_QWEN_PREFIX_TOKENS": "0"},
            clear=False,
        ):
            prompt = build_decoder_prompt(self.row)
        self.assertIn("binary representation withheld", prompt)
        self.assertNotIn("assembly provided via graph channel", prompt)
        self.assertNotIn("mov eax", prompt)


class EvaluatorSchemaTests(unittest.TestCase):
    def test_no_task_exception_for_local_schema(self) -> None:
        self.assertFalse(should_force_zero({"filename": "160.dart", "source_line": 152}))

    def test_no_task_exception_for_frontier_schema(self) -> None:
        self.assertFalse(should_force_zero({"id": "160", "task_id": "160"}))

    def test_unrelated_task_is_not_forced(self) -> None:
        self.assertFalse(should_force_zero({"id": "159", "filename": "159.dart"}))


class EdgeAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "edges": [
                {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
                {"source": 1, "target": 2, "edge_type": "conditional_true"},
                {"source": 0, "target": 2, "edge_type": "dataflow"},
            ]
        }
        self.nodes = torch.zeros((3, 768))

    def _edge_count(self, mode: str) -> int:
        with patch.dict(
            "os.environ",
            {"GRAPH_EDGE_ABLATION": mode, "GRAPH_ADD_REVERSE_EDGES": "0"},
            clear=False,
        ):
            return cfg_to_pyg(self.record, self.nodes).edge_index.size(1)

    def test_none_removes_all_edges(self) -> None:
        self.assertEqual(self._edge_count("none"), 0)

    def test_cfg_and_dfg_filters_are_complementary(self) -> None:
        self.assertEqual(self._edge_count("cfg"), 2)
        self.assertEqual(self._edge_count("dfg"), 1)

    def test_shuffle_is_deterministic_and_preserves_count(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "GRAPH_EDGE_ABLATION": "shuffle",
                "GRAPH_SEED": "42",
                "GRAPH_ADD_REVERSE_EDGES": "0",
            },
            clear=False,
        ):
            first = cfg_to_pyg(self.record, self.nodes).edge_index.clone()
            second = cfg_to_pyg(self.record, self.nodes).edge_index.clone()
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.size(1), 3)


class CandidatePoolReuseTests(unittest.TestCase):
    def test_copy_updates_output_path_and_records_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "compile.json"
            destination = root / "pass.json"
            source.write_text('[{"predictions": ["x"]}]', encoding="utf-8")
            Path(str(source) + ".provenance.json").write_text(
                json.dumps({"output": {"path": str(source)}, "seed": 42}),
                encoding="utf-8",
            )

            copy_prediction_pool(str(source), str(destination))

            self.assertEqual(destination.read_bytes(), source.read_bytes())
            provenance = json.loads(
                Path(str(destination) + ".provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["output"]["path"], str(destination.resolve()))
            self.assertEqual(
                provenance["reused_candidate_pool_from"], str(source.resolve())
            )

    def test_copy_refuses_pool_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "compile.json"
            destination = Path(tmp) / "pass.json"
            source.write_text("[]", encoding="utf-8")
            with self.assertRaises(SystemExit):
                copy_prediction_pool(str(source), str(destination))
            self.assertFalse(destination.exists())


class EncoderFollowupCommandTests(unittest.TestCase):
    @staticmethod
    def _option(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def test_encoder_controls_select_matching_experiment_configs(self) -> None:
        args = argparse.Namespace(
            seed=42,
            encoder_variants=(
                "prefix_no_gine_clap,prefix_no_gine_multivector4"
            ),
            metric_workers=64,
            hf_repo="",
        )

        stages = {stage.name: stage for stage in make_encoder_controls(args)}
        clap = stages["x86_encoder_prefix_no_gine_clap"].command
        multivector = stages[
            "x86_encoder_prefix_no_gine_multivector4"
        ].command

        self.assertEqual(self._option(clap, "--experiment"), CLAP_EXPERIMENT)
        self.assertEqual(self._option(clap, "--encoder"), "clap")
        self.assertEqual(self._option(multivector, "--experiment"), EXPERIMENT)
        self.assertEqual(self._option(multivector, "--encoder"), "gcb")

    def test_interaction_cells_have_exact_factors_and_unique_model_names(self) -> None:
        args = argparse.Namespace(
            seed=42,
            interaction_variants=(
                "prefix_no_gine_clap_frozen_encoder,prefix_cfg_clap"
            ),
            metric_workers=64,
            hf_repo="",
        )

        stages = make_interaction_controls(args)
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "x86_interaction_prefix_no_gine_clap_frozen_encoder",
                "x86_interaction_prefix_cfg_clap",
            ],
        )
        self.assertEqual(len({stage.model_name for stage in stages}), 2)

        frozen, cfg = stages
        self.assertEqual(
            frozen.model_name,
            CLAP_FROZEN_ENCODER_MODEL_STEM
            + "_graphv2_clean_s42_prefix_no_gine_clap_frozen_encoder",
        )
        self.assertEqual(
            cfg.model_name,
            CLAP_MODEL_STEM + "_graphv2_clean_s42_prefix_cfg_clap",
        )
        for stage in stages:
            self.assertEqual(self._option(stage.command, "--encoder"), "clap")
            self.assertEqual(
                self._option(stage.command, "--encoder_revision"),
                CLAP_ENCODER_REV,
            )
            self.assertEqual(stage.expected_encoder_model, "hustcw/clap-asm")

        self.assertEqual(
            self._option(frozen.command, "--experiment"),
            CLAP_FROZEN_ENCODER_EXPERIMENT,
        )
        self.assertEqual(self._option(frozen.command, "--dfg_mode"), "edges")
        self.assertEqual(self._option(frozen.command, "--edge_ablation"), "full")
        self.assertEqual(self._option(frozen.command, "--gnn_ablation"), "identity")
        self.assertTrue(frozen.expected_freeze_encoder)
        self.assertEqual(frozen.expected_encoder_peft, "none")

        self.assertEqual(self._option(cfg.command, "--experiment"), CLAP_EXPERIMENT)
        self.assertEqual(self._option(cfg.command, "--dfg_mode"), "edges")
        self.assertEqual(self._option(cfg.command, "--edge_ablation"), "cfg")
        self.assertEqual(self._option(cfg.command, "--gnn_ablation"), "full")
        self.assertFalse(cfg.expected_freeze_encoder)
        self.assertEqual(cfg.expected_encoder_peft, "lora")

    def test_interaction_completion_rejects_wrong_factor_provenance(self) -> None:
        args = argparse.Namespace(
            seed=42,
            interaction_variants="prefix_no_gine_clap_frozen_encoder",
            metric_workers=64,
            hf_repo="",
        )
        stage = make_interaction_controls(args)[0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results"
            sweep_dir = result_dir / "sweeps_antigravity"
            sweep_dir.mkdir(parents=True)
            (sweep_dir / f"{stage.model_name}.json").write_text(
                "{}", encoding="utf-8"
            )
            predictions = result_dir / f"{stage.model_name}_pass_predictions.json"
            predictions.write_text(
                json.dumps([{"predictions": ["x"] * 10} for _ in range(154)]),
                encoding="utf-8",
            )
            provenance = {
                "prompt_schema_version": "antigravity-v2-no-test-hints",
                "scoring_tests_visible_to_policy": False,
                "graph_input_ablation": {"mode": "none"},
                "graph_environment": {
                    "GRAPH_REGION_COMPRESSION": "off",
                    "GRAPH_BLOCK_POOLING": "cls",
                    "GRAPH_ENCODER_MODEL": "hustcw/clap-asm",
                    "GRAPH_FREEZE_ENCODER": "1",
                    "GRAPH_ENCODER_PEFT": "none",
                    "GRAPH_DFG_MODE": "edges",
                    "GRAPH_EDGE_ABLATION": "full",
                    "GRAPH_GNN_ABLATION": "identity",
                },
            }
            sidecar = Path(str(predictions) + ".provenance.json")
            sidecar.write_text(json.dumps(provenance), encoding="utf-8")

            with patch("scripts.run_graphv2_followups.ROOT", root):
                self.assertTrue(result_complete(stage))
                provenance["graph_environment"]["GRAPH_GNN_ABLATION"] = "full"
                sidecar.write_text(json.dumps(provenance), encoding="utf-8")
                self.assertFalse(result_complete(stage))

    def test_comprehensive_vector_sweep_covers_two_four_and_eight(self) -> None:
        args = argparse.Namespace(
            seed=42,
            vector_values="2,4,8",
            metric_workers=64,
            hf_repo="",
        )
        stages = make_vector_sweep(args)

        self.assertEqual(len(stages), 3)
        self.assertEqual(
            {
                self._option(stage.command, "--block_vectors_per_block")
                for stage in stages
            },
            {"2", "4", "8"},
        )
        self.assertTrue(
            all(self._option(stage.command, "--block_pooling") == "multi_query"
                for stage in stages)
        )

    def test_prefix_grid_is_full_three_by_three_and_reuses_default_label(self) -> None:
        args = argparse.Namespace(
            seed=42,
            selected_representation="prefix_no_gine",
            prefix_density_values="2,4,6",
            prefix_gate_values="0.1,0.2,0.3",
            metric_workers=64,
            hf_repo="",
        )
        stages = make_prefix_grid(args)

        self.assertEqual(len(stages), 9)
        pairs = {
            (
                self._option(stage.command, "--qwen_prefix_tokens_per_log2"),
                self._option(stage.command, "--qwen_prefix_gate_init"),
            )
            for stage in stages
        }
        self.assertEqual(
            pairs,
            {
                (density, gate)
                for density in ("2", "4", "6")
                for gate in ("0.1", "0.2", "0.3")
            },
        )
        default = next(
            stage
            for stage in stages
            if self._option(stage.command, "--qwen_prefix_tokens_per_log2") == "4"
            and self._option(stage.command, "--qwen_prefix_gate_init") == "0.2"
        )
        self.assertTrue(default.model_name.endswith("_s42_prefix_no_gine"))

    def test_component_isolation_includes_position_and_encoder_controls(self) -> None:
        args = argparse.Namespace(
            seed=42,
            isolation_variants=(
                "prefix_no_gine_no_attention,prefix_no_edges_gine2,"
                "prefix_no_gine_no_positions,prefix_no_gine_frozen_encoder"
            ),
            metric_workers=64,
            hf_repo="",
        )
        stages = {stage.name: stage for stage in make_isolation(args)}

        self.assertEqual(len(stages), 4)
        self.assertEqual(
            self._option(
                stages["x86_isolation_prefix_no_gine_no_positions"].command,
                "--block_position_mode",
            ),
            "off",
        )
        frozen = stages["x86_isolation_prefix_no_gine_frozen_encoder"]
        self.assertIn("freeze_enc_lora_dec", self._option(frozen.command, "--experiment"))
        self.assertTrue(frozen.expected_freeze_encoder)

    def test_arm64_accepts_all_multivector_counts_and_frozen_prefix_settings(self) -> None:
        self.assertTrue(
            {
                "prefix_no_gine_multivector2",
                "prefix_no_gine_multivector4",
                "prefix_no_gine_multivector8",
                "prefix_no_gine_regions16",
            }.issubset(ARM64_SELECTED_ARCHITECTURES)
        )
        self.assertEqual(
            ARM64_SELECTED_ARCHITECTURES["prefix_no_gine_regions16"]["region_max_blocks"],
            16,
        )
        label = arm64_selected_label(argparse.Namespace(
            selected_architecture="prefix_no_gine_multivector8",
            selected_prefix_density=6,
            selected_gate_init=0.3,
        ))
        self.assertEqual(
            label,
            "prefix_no_gine_multivector8_ppl6_gate0p3",
        )


class CodeBLEUCompatibilityTests(unittest.TestCase):
    def test_project_dart_calculator_is_available(self) -> None:
        calculator = CodeBLEUCalculator("dart")
        result = calculator.compute_codebleu(
            "int addOne(int x) => x + 1;",
            "int addOne(int x) => x + 1;",
        )
        self.assertIn("codebleu", result)
        self.assertGreaterEqual(result["codebleu"], 0.0)
        self.assertLessEqual(result["codebleu"], 1.0)


class InferenceConfigurationTests(unittest.TestCase):
    def test_generation_explicitly_enables_kv_cache(self) -> None:
        source = inspect.getsource(GraphInferenceModel.generate)
        self.assertEqual(source.count("use_cache=True"), 2)


class AlignedCompileClassificationTests(unittest.TestCase):
    def test_static_error_classifier_handles_front_end_crash_not_runtime_failure(self) -> None:
        self.assertTrue(_is_dart_jit_static_error("Error: Expected ';' after this."))
        self.assertTrue(
            _is_dart_jit_static_error(
                "Crash when compiling: RangeError (index): Invalid value: 4"
            )
        )
        self.assertFalse(_is_dart_jit_static_error("Unhandled exception: 3 != 2"))

    def test_success_must_repeat_for_stability(self) -> None:
        tests = """void main() {
  final candidate = foo;
  expect(candidate(1), 2);
}
void expect(dynamic a, dynamic b) { if (a != b) throw '$a != $b'; }
"""
        nonce = "a" * 64
        marker = f"__ANTIGRAVITY_DART_TEST_COMPLETED_{nonce}__"
        responses = [
            subprocess.CompletedProcess([], 0, stdout=marker + "\n", stderr=""),
            subprocess.CompletedProcess(
                [], 255, stdout="", stderr="Unhandled exception: stochastic failure"
            ),
        ]
        with (
            patch(
                "scripts.evaluation.graph_compile_at_k_antigravity.secrets.token_hex",
                return_value=nonce,
            ),
            patch(
                "scripts.evaluation.graph_compile_at_k_antigravity.subprocess.run",
                side_effect=responses,
            ) as run,
        ):
            compiled, passed, diagnostic, _ = evaluate_dart_jit_tests_detail(
                "int foo(int x) => x + 1;",
                tests,
                "stochastic",
                timeout=30,
                stability_runs=3,
            )
        self.assertEqual((compiled, passed), (True, False))
        self.assertIn("stability_run_2/3", diagnostic)
        self.assertEqual(run.call_count, 2)

    def test_completion_attestation_is_unique_and_rejects_process_capabilities(self) -> None:
        nonce = "b" * 64
        source = "void main() { return; }\n"
        with patch(
            "scripts.evaluation.graph_compile_at_k_antigravity.secrets.token_hex",
            return_value=nonce,
        ):
            ok, diagnostic, instrumented, marker = (
                prepare_dart_test_completion_attestation(source)
            )
        self.assertTrue(ok, diagnostic)
        self.assertNotIn("void main() { return; }", instrumented)
        self.assertIn("Future<void> main(List<String>", instrumented)
        self.assertTrue(dart_test_completion_observed(marker + "\n", marker))
        self.assertFalse(
            dart_test_completion_observed(marker + "\n" + marker + "\n", marker)
        )
        self.assertEqual(
            disallowed_dart_test_runtime_library(r"import 'dart:\x69o';"),
            "dart:io",
        )

    @unittest.skipUnless(shutil.which("dart"), "Dart is required for executable protocol test")
    def test_one_jit_run_returns_nested_compile_and_pass_outcomes(self) -> None:
        tests = """void main() {
  final candidate = foo;
  expect(candidate(1), 2);
}
void expect(dynamic a, dynamic b) { if (a != b) throw '$a != $b'; }
"""
        compiled, passed, diagnostic, _ = evaluate_dart_jit_tests_detail(
            "int foo(int x) => x + 1;", tests, "pass", timeout=30
        )
        self.assertEqual((compiled, passed), (True, True), diagnostic)

        compiled, passed, _, _ = evaluate_dart_jit_tests_detail(
            "int foo(int x) => x + 2;", tests, "runtime_fail", timeout=30
        )
        self.assertEqual((compiled, passed), (True, False))

        compiled, passed, _, _ = evaluate_dart_jit_tests_detail(
            "int foo(int x) => ;", tests, "static_fail", timeout=30
        )
        self.assertEqual((compiled, passed), (False, False))

        async_tests = """Future<void> main() async {
  await Future<void>.delayed(Duration.zero);
  final candidate = foo;
  expect(await Future.value(candidate(1)), 2);
}
void expect(dynamic a, dynamic b) { if (a != b) throw '$a != $b'; }
"""
        compiled, passed, diagnostic, _ = evaluate_dart_jit_tests_detail(
            "int foo(int x) => x + 1;", async_tests, "async_pass", timeout=30
        )
        self.assertEqual((compiled, passed), (True, True), diagnostic)

    @unittest.skipUnless(shutil.which("dart"), "Dart is required for executable protocol test")
    def test_candidate_controlled_successful_termination_never_passes(self) -> None:
        tests = """void main() {
  final candidate = foo;
  expect(candidate(1), 2);
}
void expect(dynamic a, dynamic b) { if (a != b) throw '$a != $b'; }
"""
        attacks = {
            "exit_zero": (
                "import 'dart:io';\nint foo(int x) { exit(0); }",
                "completion_attestation_disallowed_library",
            ),
            "isolate_exit": (
                "import 'dart:isolate';\n"
                "int foo(int x) { Isolate.exit(); }",
                "",
            ),
        }
        for task_id, (candidate, required_diagnostic) in attacks.items():
            with self.subTest(task_id=task_id):
                compiled, passed, diagnostic, _ = evaluate_dart_jit_tests_detail(
                    candidate, tests, task_id, timeout=30
                )
                self.assertFalse(passed)
                if required_diagnostic:
                    self.assertIn(required_diagnostic, diagnostic)


if __name__ == "__main__":
    unittest.main()
