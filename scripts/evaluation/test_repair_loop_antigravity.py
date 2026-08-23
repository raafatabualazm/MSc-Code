"""Focused tests for the compile/test/repair diagnostic driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation.repair_loop_antigravity import (
    _next_chain_round_limit,
    _normalize_assertions,
    aggregate,
    build_repair_prompt,
    failure_topology,
    generate_model_candidate,
    load_visible_tests,
    pass_at_k,
    resolve_run_configuration,
    run_repair_chain,
    validate_base_prompt,
    validate_visible_test_boundary,
    visible_tests_for_row,
)


class PassAtKTests(unittest.TestCase):
    def test_estimator_and_aggregate(self) -> None:
        self.assertAlmostEqual(pass_at_k(10, 1, 1), 0.1)
        self.assertEqual(pass_at_k(10, 0, 10), 0.0)
        self.assertEqual(pass_at_k(10, 1, 10), 1.0)

        metrics = aggregate([[1, 0], [0, 0]], ks=(1, 2))
        self.assertAlmostEqual(metrics["pass_at_1"], 0.25)
        self.assertAlmostEqual(metrics["pass_at_2"], 0.5)
        self.assertEqual(metrics["solved_any"], 1)


class FailureTopologyTests(unittest.TestCase):
    def test_unsolved_tasks_are_split_into_three_compile_categories(self) -> None:
        tasks = [
            {"compile": [1, 0], "pass": [1, 0]},
            {"compile": [0, 0], "pass": [0, 0]},
            {"compile": [1, 0], "pass": [0, 0]},
            {"compile": [1, 1], "pass": [0, 0]},
        ]

        topology = failure_topology(tasks)

        self.assertEqual(topology["unsolved_tasks"], 3)
        self.assertEqual(topology["no_compiling_candidates_tasks"], 1)
        self.assertEqual(topology["mixed_compile_outcomes_tasks"], 1)
        self.assertEqual(topology["all_candidates_compile_wrong_tasks"], 1)
        self.assertEqual(topology["tasks_with_base_compile_feedback"], 2)
        self.assertEqual(topology["noncompiling_candidates_on_unsolved_tasks"], 3)
        self.assertEqual(topology["compiling_wrong_candidates_on_unsolved_tasks"], 3)

    def test_mismatched_candidate_vectors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "compile flags"):
            failure_topology([{"compile": [1], "pass": [0, 0]}])


class RepairBoundaryTests(unittest.TestCase):
    def test_assertion_scan_is_main_scoped_and_balanced(self) -> None:
        harness = """void main() {
  final candidate = combine;
  expect(
    candidate([1, 2], {'close': ')'}),
    3,
  );
  assert(candidate([], {}) == 0);
}

void expect(dynamic a, dynamic b) {
  if (a is List && b is List) expect(a[0], b[0]);
}
"""
        assertions = _normalize_assertions(harness)
        self.assertEqual(len(assertions), 2)
        self.assertTrue(any("candidate([1,2]" in item for item in assertions))
        self.assertFalse(any("a[0]" in item for item in assertions))

    def test_generation_budget_caps_the_last_chain(self) -> None:
        self.assertEqual(_next_chain_round_limit(3, 10, 0, 0, 40), 3)
        self.assertEqual(_next_chain_round_limit(3, 10, 12, 39, 40), 0)
        self.assertIsNone(_next_chain_round_limit(3, 10, 13, 40, 40))
        self.assertEqual(_next_chain_round_limit(3, 2, 0, 0, None), 3)
        self.assertIsNone(_next_chain_round_limit(3, 2, 2, 8, None))

    def test_base_prompt_rejects_hidden_test_payload(self) -> None:
        tests = "expect(candidate(7), 42);"
        validate_base_prompt("Convert assembly to Dart.", tests)
        with self.assertRaisesRegex(RuntimeError, "verbatim scoring tests"):
            validate_base_prompt(f"Convert assembly.\n{tests}", tests)
        scoring = "void main() {\n  expect(candidate(7), 42);\n}"
        with self.assertRaisesRegex(RuntimeError, "scoring-test assertion"):
            validate_base_prompt(
                "Convert assembly.\nexpect(candidate(7), 42);",
                scoring,
            )

    def test_compile_and_oracle_prompts_are_distinct(self) -> None:
        base = "Convert assembly to Dart."
        compile_prompt = build_repair_prompt(
            base,
            "int f(){",
            "error: expected '}'",
            feedback_kind="compile",
        )
        oracle_prompt = build_repair_prompt(
            base,
            "int f()=>0;",
            "Expected: 5 Actual: 0",
            feedback_kind="oracle_tests",
        )

        self.assertIn("Compiler diagnostic", compile_prompt)
        self.assertNotIn("ORACLE", compile_prompt)
        self.assertIn("ORACLE hidden-test diagnostic", oracle_prompt)
        self.assertIn("contaminated", oracle_prompt)

    def test_chain_uses_feedback_then_stops_on_success(self) -> None:
        prompts: list[str] = []

        def generate(prompt: str, generation_index: int) -> str:
            prompts.append(prompt)
            return "int f()=>0;" if generation_index == 0 else "int f()=>5;"

        def evaluate(candidate: str) -> tuple[bool, str]:
            ok = "=>5;" in candidate
            return ok, "" if ok else "Expected: 5 Actual: 0"

        candidate, trace, terminal = run_repair_chain(
            generate,
            evaluate,
            "Convert assembly to Dart.",
            max_repair_rounds=3,
            feedback_kind="oracle_tests",
        )

        self.assertEqual(candidate, "int f()=>5;")
        self.assertEqual(len(trace), 2)
        self.assertEqual(terminal, "tests_passed")
        self.assertIn("Expected: 5 Actual: 0", prompts[1])

    def test_generation_adapter_calls_real_generate_surface(self) -> None:
        class FakeTokenizer:
            truncation_side = "right"

            def __init__(self) -> None:
                self.calls: list[dict] = []

            def __call__(self, prompt: str, **kwargs):
                self.calls.append(
                    {
                        "prompt": prompt,
                        "truncation_side": self.truncation_side,
                        **kwargs,
                    }
                )
                return {"input_ids": "ids", "attention_mask": "mask"}

        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = None

            def generate(self, *args, **kwargs):
                self.kwargs = kwargs
                return ["int f()=>5;"]

        tokenizer = FakeTokenizer()
        model = FakeModel()
        candidate = generate_model_candidate(
            model,
            tokenizer,
            block_tensors="blocks",
            graph_data="graph",
            prompt="repair feedback at the end",
            device="cuda",
            max_new_tokens=768,
            prompt_max_length=2048,
            do_sample=True,
            preserve_prompt_suffix=True,
        )

        self.assertEqual(candidate, "int f()=>5;")
        self.assertEqual(tokenizer.calls[0]["truncation_side"], "left")
        self.assertEqual(tokenizer.truncation_side, "right")
        self.assertEqual(model.kwargs["decoder_prompt_input_ids"], "ids")
        self.assertEqual(model.kwargs["decoder_prompt_attention_mask"], "mask")
        self.assertTrue(model.kwargs["do_sample"])
        self.assertEqual(model.kwargs["num_samples"], 1)


class VisibleTestBoundaryTests(unittest.TestCase):
    def test_loader_requires_visible_tests_field_and_matches_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visible.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "task-7",
                        "filename": "7.dart",
                        "visible_tests": (
                            "void main(){final candidate = square;"
                            "expect(candidate(2), 4);}"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            mapping = load_visible_tests(path)

        self.assertEqual(
            visible_tests_for_row(mapping, {"task_id": "task-7"}, 0),
            "void main(){final candidate = square;expect(candidate(2), 4);}",
        )
        self.assertEqual(
            visible_tests_for_row(mapping, {"filename": "7.dart"}, 0),
            "void main(){final candidate = square;expect(candidate(2), 4);}",
        )

    def test_loader_rejects_hidden_fields_and_identifierless_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visible.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "task-7",
                        "visible_tests": (
                            "void main(){final candidate = square;"
                            "expect(candidate(2), 4);}"
                        ),
                        "tests": "void main(){expect(candidate(2), 4);}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "forbidden hidden/scoring"):
                load_visible_tests(path)

            path.write_text(
                json.dumps(
                    {
                        "visible_tests": (
                            "void main(){final candidate = square;"
                            "expect(candidate(2), 4);}"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no stable task_id"):
                load_visible_tests(path)

    def test_visible_assertions_must_be_disjoint_from_scoring_tests(self) -> None:
        visible = (
            "void main(){final candidate = square;expect(candidate(2), 4);}"
        )
        scoring = (
            "void main(){final candidate = square;expect(candidate(2), 4);"
            "expect(candidate(3), 9);}"
        )
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            validate_visible_test_boundary(visible, scoring, "task-7")

        validate_visible_test_boundary(
            "void main(){final candidate = square;expect(candidate(5), 25);}",
            scoring,
            "task-7",
        )

    def test_candidate_bindings_and_inputs_must_be_disjoint(self) -> None:
        scoring = (
            "void main(){final candidate = square;"
            "expect(candidate(2), 4);}"
        )
        with self.assertRaisesRegex(RuntimeError, "bindings differ"):
            validate_visible_test_boundary(
                "void main(){final candidate = cube;expect(candidate(3), 27);}",
                scoring,
                "task-7",
            )
        with self.assertRaisesRegex(RuntimeError, "candidate inputs overlap"):
            validate_visible_test_boundary(
                "void main(){final candidate = square;"
                "expect(candidate(2), 2 + 2);}",
                scoring,
                "task-7",
            )


class SourceProvenanceTests(unittest.TestCase):
    @staticmethod
    def _record(path: Path) -> dict:
        data = path.read_bytes()
        return {
            "path": f"/workspace/{path.name}",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def test_configuration_replays_environment_and_verifies_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            dataset = root / "dataset.jsonl"
            provenance_path = root / "predictions.json.provenance.json"
            checkpoint.write_bytes(b"checkpoint")
            dataset.write_text("{}\n", encoding="utf-8")
            provenance_path.write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "checkpoint": self._record(checkpoint),
                        "dataset": self._record(dataset),
                        "graph_environment": {
                            "GRAPH_QWEN_PREFIX_TOKENS": "16",
                            "GRAPH_PROMPT_ASSEMBLY_MODE": "none",
                        },
                        "models": {
                            "decoder": {
                                "requested_id": "Qwen/Qwen3-8B",
                                "requested_revision": "decoder-sha",
                            },
                            "encoder": {
                                "requested_id": "microsoft/graphcodebert-base",
                                "requested_revision": "encoder-sha",
                            },
                        },
                        "generation": {
                            "max_new_tokens": 768,
                            "decoder_prompt_max_length": 2048,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                source_provenance=str(provenance_path),
                allow_unverified_environment=False,
                decoder_model=None,
                encoder_model=None,
                decoder_revision=None,
                encoder_revision=None,
                seed=None,
                max_new_tokens=None,
                decoder_prompt_max_length=None,
                checkpoint=str(checkpoint),
                pass_dataset=str(dataset),
            )

            with patch.dict(os.environ, {}, clear=False):
                config = resolve_run_configuration(args)
                self.assertEqual(os.environ["GRAPH_QWEN_PREFIX_TOKENS"], "16")
                self.assertEqual(os.environ["GRAPH_PROMPT_ASSEMBLY_MODE"], "none")

            self.assertEqual(config["decoder_model"], "Qwen/Qwen3-8B")
            self.assertEqual(config["decoder_revision"], "decoder-sha")
            self.assertEqual(config["encoder_revision"], "encoder-sha")
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["prompt_max_length"], 2048)

    def test_checkpoint_hash_mismatch_stops_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            dataset = root / "dataset.jsonl"
            provenance_path = root / "predictions.json.provenance.json"
            checkpoint.write_bytes(b"changed")
            dataset.write_text("{}\n", encoding="utf-8")
            provenance_path.write_text(
                json.dumps(
                    {
                        "checkpoint": {
                            "size_bytes": len(b"changed"),
                            "sha256": hashlib.sha256(b"original").hexdigest(),
                        },
                        "dataset": self._record(dataset),
                        "graph_environment": {},
                        "models": {
                            "decoder": {"requested_id": "Qwen/Qwen3-8B"},
                            "encoder": {
                                "requested_id": "microsoft/graphcodebert-base"
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                source_provenance=str(provenance_path),
                allow_unverified_environment=False,
                decoder_model=None,
                encoder_model=None,
                decoder_revision=None,
                encoder_revision=None,
                seed=None,
                max_new_tokens=None,
                decoder_prompt_max_length=None,
                checkpoint=str(checkpoint),
                pass_dataset=str(dataset),
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint sha256"):
                resolve_run_configuration(args)

    def test_split_manifest_bridges_original_and_hidden_dataset_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            original = root / "original.jsonl"
            hidden = root / "hidden.jsonl"
            visible = root / "visible.jsonl"
            split_manifest = root / "split.manifest.json"
            provenance_path = root / "predictions.json.provenance.json"
            checkpoint.write_bytes(b"checkpoint")
            original.write_text('{"tests":"all"}\n', encoding="utf-8")
            hidden.write_text('{"tests":"hidden"}\n', encoding="utf-8")
            visible.write_text('{"visible_tests":"visible"}\n', encoding="utf-8")
            split_manifest.write_text(
                json.dumps(
                    {
                        "stage": "repair_test_split",
                        "input": self._record(original),
                        "hidden_output": self._record(hidden),
                        "visible_output": self._record(visible),
                    }
                ),
                encoding="utf-8",
            )
            provenance_path.write_text(
                json.dumps(
                    {
                        "checkpoint": self._record(checkpoint),
                        "dataset": self._record(original),
                        "graph_environment": {},
                        "models": {
                            "decoder": {"requested_id": "Qwen/Qwen3-8B"},
                            "encoder": {
                                "requested_id": "microsoft/graphcodebert-base"
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                source_provenance=str(provenance_path),
                allow_unverified_environment=False,
                decoder_model=None,
                encoder_model=None,
                decoder_revision=None,
                encoder_revision=None,
                seed=None,
                max_new_tokens=None,
                decoder_prompt_max_length=None,
                checkpoint=str(checkpoint),
                pass_dataset=str(hidden),
                test_split_manifest=str(split_manifest),
            )

            config = resolve_run_configuration(args)

            self.assertEqual(
                config["dataset_record"]["sha256"],
                hashlib.sha256(hidden.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                config["test_split_manifest"]["input"]["sha256"],
                hashlib.sha256(original.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
