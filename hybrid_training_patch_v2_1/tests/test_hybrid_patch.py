from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

PATCH_ROOT = Path(__file__).resolve().parents[1]
TRAINING = PATCH_ROOT / "scripts" / "training"
EVALUATION = PATCH_ROOT / "scripts" / "evaluation"
SFT = TRAINING / "graph_encoder_decoder_decompiler_v2_antigravity.py"
GRPO = TRAINING / "graph_grpo_decompiler_antigravity.py"
TEACHER = TRAINING / "teacher_repair_dataset_antigravity.py"
RUNNER = TRAINING / "run_hybrid_curriculum_antigravity.py"
CONTROLS = TRAINING / "hybrid_data_controls.py"
BALANCED = TRAINING / "build_balanced_sft_mix_antigravity.py"
PREP = TRAINING / "prepare_hybrid_training_data_antigravity.py"
AUDIT = EVALUATION / "audit_grpo_reward_antigravity.py"
FUNCTIONAL_GATE = EVALUATION / "functional_graph_gate_antigravity.py"
NEUTRAL_EVAL = EVALUATION / "prepare_neutral_evaluation_antigravity.py"
INFERENCE = EVALUATION / "graph_inference_antigravity.py"
PROBE = EVALUATION / "probe_graph_representations_antigravity.py"
CHECKPOINT_CONTRACT = TRAINING / "checkpoint_contract.py"

if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from scripts.training.hybrid_data_controls import (  # noqa: E402
    alpha_normalized_source,
    attach_test_partition,
    candidate_assertion_count,
    candidate_fact_match,
    facts_comment,
    infer_function_name,
    mechanical_facts,
    neutralize_training_row,
    sanitize_verifier_diagnostic,
    source_fingerprints,
    verified_origin,
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def copy_python_package_file(source: Path, root: Path) -> Path:
    relative = source.relative_to(PATCH_ROOT)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    current = target.parent
    while current != root:
        (current / "__init__.py").touch()
        current = current.parent
    return target




def phase0_metadata(*, length_bin: str = "lt60", origin: str | None = None) -> dict:
    metadata = {
        "phase0_approved": True,
        "neutralized": True,
        "neutral_contract": True,
        "evaluation_only": False,
        "data_role": "train",
        "source_overlap_hash": "f" * 64,
        "source_overlap_hashes": {
            "neutral_sha256": "f" * 64,
            "alpha_structural_sha256": "e" * 64,
        },
        "reference_test_replay": {"passed": True},
        "length_bin": length_bin,
    }
    if origin is not None:
        metadata.update(
            {
                "origin": origin,
                "verifier_replayed": True,
                "feedback_replayed": True,
                "feedback_tests_passed": True,
                "verifier_full_pass": True,
                "hidden_acceptance_replayed": True,
                "acceptance_tests_passed": True,
                "facts_gate_passed": True,
            }
        )
    return metadata

def write_fake_evaluator(root: Path, body: str) -> None:
    target = root / "scripts/evaluation/graph_compile_at_k_antigravity.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    (root / "scripts/__init__.py").touch()
    (root / "scripts/evaluation/__init__.py").touch()
    target.write_text(body, encoding="utf-8")


class HybridPatchV2Tests(unittest.TestCase):
    maxDiff = None

    def test_all_python_files_parse(self):
        paths = sorted(PATCH_ROOT.rglob("*.py"))
        self.assertGreaterEqual(len(paths), 12)
        for path in paths:
            if "__pycache__" in path.parts:
                continue
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_neutral_contract_preserves_literals_and_partitions_tests(self):
        row = {
            "task_id": "semantic-task",
            "function": "semanticName",
            "signature": "int semanticName(int count)",
            "source": "int semanticName(int count) { const s = 'semanticName'; return count + 1; }",
            "assembly": "semanticName:\n  bl semanticName\n  cmp x0, 0\n  add x0, x0, 1\n  ret",
            "cfg": [{"id": 0, "instructions": ["bl semanticName", "ret"]}],
            "tests": """void main() {
  final candidate = semanticName;
  expect(candidate(0), 1);
  expect(candidate(2), 3);
  expect(candidate(5), 6);
}
""",
        }
        neutral = neutralize_training_row(row)
        self.assertEqual(infer_function_name(neutral), "fn0")
        self.assertIn("int fn0(int a)", neutral["signature"])
        self.assertIn("'semanticName'", neutral["source"])
        self.assertNotIn("semanticName:", neutral["assembly"])
        self.assertIn("fn0:", neutral["assembly"])
        self.assertEqual(neutral["cfg"][0]["instructions"][0], "bl fn0")
        self.assertTrue(neutral["hybrid_metadata"]["binary_symbol_neutralized"])
        self.assertNotIn("final candidate = semanticName", neutral["tests"])
        self.assertIn("final candidate = fn0", neutral["tests"])

        split_a = attach_test_partition(neutral, feedback_fraction=2 / 3, seed=42)
        split_b = attach_test_partition(neutral, feedback_fraction=2 / 3, seed=42)
        self.assertEqual(split_a["feedback_tests"], split_b["feedback_tests"])
        self.assertEqual(split_a["acceptance_tests"], split_b["acceptance_tests"])
        self.assertEqual(candidate_assertion_count(split_a["feedback_tests"]), 2)
        self.assertEqual(candidate_assertion_count(split_a["acceptance_tests"]), 1)
        feedback_asserts = {
            line.strip()
            for line in split_a["feedback_tests"].splitlines()
            if line.strip().startswith("expect(")
        }
        acceptance_asserts = {
            line.strip()
            for line in split_a["acceptance_tests"].splitlines()
            if line.strip().startswith("expect(")
        }
        self.assertFalse(feedback_asserts & acceptance_asserts)

    def test_diagnostic_redaction_removes_oracle_values(self):
        raw = """/tmp/task/test.dart: Expected: <42>
Actual: <7>
Input: [1, 2, 3]
output: secret
Unhandled exception: test failed for value 99
"""
        clean = sanitize_verifier_diagnostic(raw)
        lowered = clean.lower()
        for token in ("expected", "actual", "input:", "output:", "secret", "42", "<7>"):
            self.assertNotIn(token.lower(), lowered)
        self.assertIn("assertion mismatch values redacted", lowered)

    def test_complete_facts_gate_and_return_contract(self):
        task = {
            "function": "fn0",
            "signature": "int fn0(int a)",
            "assembly": "cmp x0, 0\nadd x0, x0, 1\nret",
        }
        facts = mechanical_facts(task)
        task["binary_facts"] = facts
        candidate = "int fn0(int a) => a + 1;"
        exact_claim = dict(facts)
        ok, reasons = candidate_fact_match(
            task,
            candidate,
            mode="conservative",
            teacher_claim=exact_claim,
            require_claims=True,
        )
        self.assertTrue(ok, reasons)

        incomplete = dict(exact_claim)
        incomplete.pop("comparisons")
        ok, reasons = candidate_fact_match(
            task,
            candidate,
            mode="conservative",
            teacher_claim=incomplete,
            require_claims=True,
        )
        self.assertFalse(ok)
        self.assertTrue(any("omitted comparisons" in reason for reason in reasons))

        bad_return = "double fn0(int a) => a + 1.0;"
        ok, reasons = candidate_fact_match(task, bad_return, mode="signature")
        self.assertFalse(ok)
        self.assertTrue(any("return type" in reason for reason in reasons))

    def test_alpha_fingerprint_catches_local_renaming_but_preserves_literals(self):
        left = {
            "function": "sumValues",
            "source": "int sumValues(int value) { final result = value + 7; return result; }",
        }
        renamed = {
            "function": "compute",
            "source": "int compute(int x) { final y = x + 7; return y; }",
        }
        changed = {
            "function": "compute",
            "source": "int compute(int x) { final y = x + 8; return y; }",
        }
        self.assertNotEqual(
            source_fingerprints(left)["neutral_sha256"],
            source_fingerprints(renamed)["neutral_sha256"],
        )
        self.assertEqual(
            source_fingerprints(left)["alpha_structural_sha256"],
            source_fingerprints(renamed)["alpha_structural_sha256"],
        )
        self.assertNotEqual(
            source_fingerprints(renamed)["alpha_structural_sha256"],
            source_fingerprints(changed)["alpha_structural_sha256"],
        )
        self.assertIn("7", alpha_normalized_source(left))

    def test_semantic_assembly_facts_exclude_runtime_layout_numbers(self):
        row = {
            "function": "fn0",
            "signature": "int fn0(int a)",
            "assembly": """
fn0:
  stp x29, x30, [sp, #-0x10]!
  sub sp, sp, #0x18
  ldr x0, [x26, #0x48]
  b.ls 0x2264f8098
  add x3, x22, #0x20
  sub x5, x4, #0x30
  cmp x1, #0x32
  mov x0, #0x64
  ret
""",
        }
        facts = mechanical_facts(row)
        constants = facts["salient_numeric_constants"]
        self.assertEqual(constants, [48, 50, 100])
        for runtime_value in (-16, 24, 32, 72, 0x2264F8098):
            self.assertNotIn(runtime_value, constants)
        self.assertEqual(facts["facts_extractor_version"], 2)

    def test_verified_anchor_provenance_is_strict(self):
        gold = {
            "hybrid_metadata": {
                "origin": "verified_reference",
                "verifier_replayed": True,
                "verifier_full_pass": True,
            }
        }
        repair = {"hybrid_metadata": phase0_metadata(origin="teacher_repair")}
        self.assertFalse(verified_origin(gold))
        self.assertTrue(verified_origin(repair))

    def test_full_text_dependence_objective_is_guarded(self):
        tree = ast.parse(SFT.read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PrefixDependenceSeq2SeqTrainer"
        )
        module_ast = ast.Module(body=[class_node], type_ignores=[])
        ast.fix_missing_locations(module_ast)

        import copy
        import torch
        import torch.nn.functional as F

        class StubTrainer:
            def __init__(self, *args, **kwargs):
                self.state = types.SimpleNamespace(global_step=0)
                self.args = types.SimpleNamespace(logging_steps=10)

            def log(self, value):
                del value

        namespace = {
            "AntigravitySeq2SeqTrainer": StubTrainer,
            "torch": torch,
            "F": F,
            "copy": copy,
            "os": os,
            "_model_output": lambda output, key: output[key],
            "_per_sequence_cross_entropy": lambda logits, labels: torch.zeros(logits.size(0)),
        }
        old = dict(os.environ)
        try:
            os.environ["GRAPH_PREFIX_DEPENDENCE_WEIGHT"] = "0.1"
            os.environ["GRAPH_PROMPT_ASSEMBLY_MODE"] = "full"
            os.environ.pop("GRAPH_ALLOW_CONFOUNDED_DEPENDENCE", None)
            exec(compile(module_ast, str(SFT), "exec"), namespace)
            with self.assertRaises(ValueError):
                namespace["PrefixDependenceSeq2SeqTrainer"]()
            os.environ["GRAPH_PROMPT_ASSEMBLY_MODE"] = "graph_only"
            trainer = namespace["PrefixDependenceSeq2SeqTrainer"]()
            self.assertAlmostEqual(trainer.prefix_dependence_weight, 0.1)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_balanced_rs_sft_is_exactly_fifty_fifty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_path = root / "gold.jsonl"
            verified_path = root / "verified.jsonl"
            out = root / "mixed.jsonl"
            gold_control = root / "gold_control.jsonl"
            report = root / "report.json"
            gold_rows = []
            for index in range(4):
                row = {
                    "task_id": f"g{index}",
                    "function": "fn0",
                    "signature": "int fn0()",
                    "assembly": "mov eax, 0\nret",
                    "source": "int fn0() => 0;",
                    "feedback_tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "acceptance_tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "hybrid_metadata": phase0_metadata(),
                }
                row["binary_facts"] = mechanical_facts(row)
                gold_rows.append(row)
            verified_rows = []
            for index, bin_name in enumerate(("lt60", "60_89")):
                row = {
                    "task_id": f"v{index}",
                    "function": "fn0",
                    "signature": "int fn0()",
                    "assembly": f"mov eax, {index}\nret",
                    "source": f"int fn0() => {index};",
                    "feedback_tests": f"void main(){{ final candidate = fn0; expect(candidate(), {index}); }}",
                    "acceptance_tests": f"void main(){{ final candidate = fn0; expect(candidate(), {index}); }}",
                    "tests": f"void main(){{ final candidate = fn0; expect(candidate(), {index}); }}",
                    "hybrid_metadata": phase0_metadata(
                        length_bin=bin_name, origin="teacher_repair"
                    ),
                }
                row["binary_facts"] = mechanical_facts(row)
                verified_rows.append(row)
            gold_path.write_text("".join(json.dumps(row) + "\n" for row in gold_rows), encoding="utf-8")
            verified_path.write_text("".join(json.dumps(row) + "\n" for row in verified_rows), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(BALANCED),
                    "--gold", str(gold_path),
                    "--verified", str(verified_path),
                    "--out", str(out),
                    "--gold_control_out", str(gold_control),
                    "--report", str(report),
                    "--min_verified_rows", "2",
                    "--min_verified_unique_tasks", "2",
                    "--min_verified_length_bins", "2",
                    "--max_verified_oversample_factor", "2",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONPATH": str(PATCH_ROOT)},
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            buckets = [row["hybrid_metadata"]["sft_mix_bucket"] for row in rows]
            self.assertEqual(buckets.count("gold"), 4)
            self.assertEqual(buckets.count("verified"), 4)
            report_data = json.loads(report.read_text())
            self.assertEqual(report_data["realized_verified_ratio"], 0.5)
            self.assertEqual(report_data["gold_coverage_fraction"], 1.0)
            self.assertTrue(report_data["gold_full_coverage_enforced"])
            control_rows = [
                json.loads(line)
                for line in gold_control.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(control_rows), len(rows))
            self.assertTrue(
                all(
                    row["hybrid_metadata"]["sft_mix_bucket"] == "gold_control"
                    for row in control_rows
                )
            )
            self.assertTrue(report_data["gold_control_matches_training_examples"])

    def test_balanced_rs_sft_rejects_silent_partial_gold_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_path = root / "gold.jsonl"
            verified_path = root / "verified.jsonl"
            gold_rows = []
            for index in range(4):
                row = {
                    "task_id": f"g{index}",
                    "function": "fn0",
                    "signature": "int fn0()",
                    "assembly": "mov eax, 0\nret",
                    "source": "int fn0() => 0;",
                    "feedback_tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "acceptance_tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "hybrid_metadata": phase0_metadata(),
                }
                row["binary_facts"] = mechanical_facts(row)
                gold_rows.append(row)
            verified_rows = []
            for index, bin_name in enumerate(("lt60", "60_89")):
                row = {
                    "task_id": f"v{index}",
                    "function": "fn0",
                    "signature": "int fn0()",
                    "assembly": f"mov eax, {index}\nret",
                    "source": f"int fn0() => {index};",
                    "feedback_tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "acceptance_tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "tests": "void main(){ final candidate = fn0; expect(candidate(), 0); }",
                    "hybrid_metadata": phase0_metadata(length_bin=bin_name, origin="teacher_repair"),
                }
                row["binary_facts"] = mechanical_facts(row)
                verified_rows.append(row)
            gold_path.write_text("".join(json.dumps(row) + "\n" for row in gold_rows), encoding="utf-8")
            verified_path.write_text("".join(json.dumps(row) + "\n" for row in verified_rows), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(BALANCED),
                    "--gold", str(gold_path),
                    "--verified", str(verified_path),
                    "--out", str(root / "mixed.jsonl"),
                    "--report", str(root / "report.json"),
                    "--rows_per_epoch", "4",
                    "--min_verified_rows", "2",
                    "--min_verified_unique_tasks", "2",
                    "--min_verified_length_bins", "2",
                    "--max_verified_oversample_factor", "2",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONPATH": str(PATCH_ROOT)},
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("silently omit training rows", proc.stderr + proc.stdout)

    def test_teacher_prompt_and_batch_request_do_not_expose_tests(self):
        module = load_module(TEACHER, "teacher_repair_v2_test")
        task = {
            "task_id": "t0",
            "function": "fn0",
            "signature": "int fn0(int a)",
            "assembly": "cmp x0, 0\nret",
            "tests": "void main(){ final candidate = fn0; expect(candidate(1), 2); }",
        }
        task["binary_facts"] = mechanical_facts(task)
        record = {
            "failure_id": "f0",
            "task_key": "t0",
            "task": task,
            "candidate": "int fn0(int a) => 0;",
            "verifier": {
                "compiled": True,
                "passed": False,
                "pass_ratio": 0.0,
                "passed_count": 0,
                "total_count": 1,
                "test_passes": [False],
                "diagnostic": sanitize_verifier_diagnostic("Expected: 2\nActual: 0"),
            },
        }
        prompt = module.make_teacher_prompt(record, "diagnostics", 2000, 2000)
        self.assertNotIn("expect(candidate", prompt)
        self.assertNotIn("Expected: 2", prompt)
        self.assertNotIn("Actual: 0", prompt)
        self.assertIn("feedback_pass_vector", prompt)
        batch = module.openai_batch_row(record, "frontier-model", prompt, 1200)
        self.assertEqual(batch["url"], "/v1/responses")
        self.assertEqual(batch["custom_id"], "f0")
        self.assertFalse(batch["body"]["store"])

    def test_rollout_provenance_binds_dataset_output_checkpoint_and_cardinality(self):
        module = load_module(TEACHER, "teacher_rollout_provenance_test")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "train.jsonl"
            dataset.write_text(json.dumps({"task_id": "t0"}) + "\n", encoding="utf-8")
            predictions = root / "predictions.json"
            prediction_rows = [
                {
                    "id": "t0",
                    "source_line": 1,
                    "predictions": ["int fn0() => 0;", "int fn0() => 1;"],
                }
            ]
            predictions.write_text(json.dumps(prediction_rows), encoding="utf-8")
            checkpoint = root / "pytorch_model.bin"
            checkpoint.write_bytes(b"checkpoint")
            provenance = {
                "prompt_schema_version": "fixture-v1",
                "scoring_tests_visible_to_policy": False,
                "row_count": 1,
                "generation": {"num_samples": 2},
                "graph_input_ablation": {
                    "mode": "none",
                    "final_context_zeroed": False,
                },
                "graph_prefix_gate": {"override_requested": None},
                "dataset": module.file_record(dataset),
                "output": module.file_record(predictions),
                "checkpoint": module.file_record(checkpoint),
                "checkpoint_load": {"status": "passed"},
            }
            sidecar = Path(str(predictions) + ".provenance.json")
            sidecar.write_text(json.dumps(provenance), encoding="utf-8")
            report = module.validate_prediction_artifact(
                dataset,
                predictions,
                prediction_rows,
                expected_checkpoint=checkpoint,
                required=True,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["unique_source_line_count"], 1)

            provenance["dataset"]["sha256"] = "0" * 64
            sidecar.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset SHA-256 mismatch"):
                module.validate_prediction_artifact(
                    dataset,
                    predictions,
                    prediction_rows,
                    expected_checkpoint=checkpoint,
                    required=True,
                )

    def test_teacher_pipeline_replays_hidden_acceptance_and_facts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            teacher_script = copy_python_package_file(TEACHER, root)
            copy_python_package_file(CONTROLS, root)
            write_fake_evaluator(
                root,
                """
def evaluate_dart_jit_tests_detail(raw, test_code, task_id, timeout=30, stability_runs=None):
    compiled = 'SYNTAX_ERROR' not in raw
    returns_one = compiled and ('=> 1;' in raw or 'return 1;' in raw)
    returns_two = compiled and ('=> 2;' in raw or 'return 2;' in raw)
    if 'FEEDBACK_ONLY' in test_code:
        passed = returns_one or returns_two
    else:
        passed = returns_one
    diagnostic = '' if passed else ('compile failed' if not compiled else 'Expected: 1\\nActual: 0')
    return compiled, passed, diagnostic, raw + '\\n' + test_code
""",
            )
            row = {
                "task_id": "t0",
                "function": "fn0",
                "signature": "int fn0()",
                "source": "int fn0() => 1;",
                "assembly": "mov eax, 1\nret",
                "feedback_tests": "// FEEDBACK_ONLY\nvoid main() {\n final candidate = fn0;\n expect(candidate(), 1);\n}\n",
                "acceptance_tests": "// HIDDEN_ONLY\nvoid main() {\n final candidate = fn0;\n expect(candidate(), 1);\n}\n",
                "tests": "// FULL\nvoid main() {\n final candidate = fn0;\n expect(candidate(), 1);\n}\n",
                "hybrid_metadata": phase0_metadata(),
            }
            row["binary_facts"] = mechanical_facts(row)
            row["facts_target_comment"] = facts_comment(row["binary_facts"])
            dataset = root / "train.jsonl"
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps(
                    [
                        {
                            "id": "t0",
                            "source_line": 1,
                            "predictions": [
                                "int fn0() => 0;",
                                "int fn0() => 2;",
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            collected = root / "collected.jsonl"
            env = {**os.environ, "PYTHONPATH": str(root)}
            subprocess.run(
                [
                    sys.executable,
                    str(teacher_script),
                    "collect",
                    "--dataset", str(dataset),
                    "--predictions", str(predictions),
                    "--data_role", "train",
                    "--out", str(collected),
                    "--workers", "1",
                ],
                cwd=root,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            collected_rows = [json.loads(line) for line in collected.read_text().splitlines()]
            failure = next(item for item in collected_rows if item["disposition"] == "needs_teacher")
            self.assertTrue(
                all(
                    "Expected" not in item["verifier"]["diagnostic"]
                    and "Actual" not in item["verifier"]["diagnostic"]
                    for item in collected_rows
                )
            )
            responses = root / "teacher.jsonl"
            responses.write_text(
                json.dumps(
                    {
                        "failure_id": failure["failure_id"],
                        "model": "fake-frontier",
                        "parsed": {
                            "failure_class": "arithmetic",
                            "confidence": 0.9,
                            "fact_claims": row["binary_facts"],
                            "behavioral_facts": ["returns an integer"],
                            "failure_evidence": ["feedback assertions failed"],
                            "repair_actions": ["return the assembly constant"],
                            "repaired_code": "int fn0() => 1;",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out_sft = root / "rs_sft.jsonl"
            out_pref = root / "preferences.jsonl"
            report = root / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(teacher_script),
                    "build",
                    "--collected", str(collected),
                    "--teacher_responses", str(responses),
                    "--data_role", "train",
                    "--out_sft", str(out_sft),
                    "--out_preferences", str(out_pref),
                    "--report", str(report),
                    "--workers", "1",
                    "--facts_gate_mode", "conservative",
                    "--min_verified_rows", "1",
                    "--min_verified_unique_tasks", "1",
                    "--min_verified_length_bins", "1",
                ],
                cwd=root,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            built = [json.loads(line) for line in out_sft.read_text().splitlines()]
            self.assertEqual(len(built), 1)
            metadata = built[0]["hybrid_metadata"]
            self.assertEqual(metadata["origin"], "teacher_repair")
            self.assertTrue(metadata["hidden_acceptance_replayed"])
            self.assertTrue(metadata["facts_gate_passed"])
            report_data = json.loads(report.read_text())
            self.assertEqual(report_data["rejections"].get("hidden_acceptance"), 1)


    def test_phase0_preflight_stratifies_short_bridge_long_and_full_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            prep_script = copy_python_package_file(PREP, root)
            audit_script = copy_python_package_file(AUDIT, root)
            copy_python_package_file(CONTROLS, root)
            write_fake_evaluator(
                root,
                """
def evaluate_dart_jit_tests_detail(raw, test_code, task_id, timeout=30, stability_runs=None):
    compiled = 'fn0' in raw
    passed = compiled and 'final candidate = fn0' in test_code and 'expect(candidate' in test_code
    return compiled, passed, '' if passed else 'Expected: hidden\\nActual: hidden', raw + '\\n' + test_code
""",
            )

            def row(index: int, instructions: int) -> dict:
                name = f"semantic{index}"
                assembly = "\n".join(
                    [f"add x0, x0, #{value}" for value in range(instructions - 1)]
                    + ["ret"]
                )
                return {
                    "task_id": f"t{index}",
                    "function": name,
                    "signature": f"int {name}(int value)",
                    "source": f"int {name}(int value) => value + {index};",
                    "assembly": assembly,
                    "tests": (
                        f"void main() {{\n final candidate = {name};\n"
                        f" expect(candidate(0), {index});\n"
                        f" expect(candidate(1), {index + 1});\n}}\n"
                    ),
                }

            source = root / "train.jsonl"
            source.write_text(
                "".join(json.dumps(row(i, n)) + "\n" for i, n in enumerate((3, 5, 7))),
                encoding="utf-8",
            )
            frozen = root / "frozen.jsonl"
            frozen.write_text(
                json.dumps(
                    {
                        "task_id": "eval-only",
                        "function": "other",
                        "signature": "int other()",
                        "source": "int other() => 99;",
                        "assembly": "mov x0, #99\nret",
                        "tests": "void main(){ final candidate = other; expect(candidate(), 99); }",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            short = root / "short.jsonl"
            bridge = root / "bridge.jsonl"
            long = root / "long.jsonl"
            report = root / "phase0.json"
            env = {**os.environ, "PYTHONPATH": str(root)}
            phase0_run = subprocess.run(
                [
                    sys.executable,
                    str(prep_script),
                    "--input", str(source),
                    "--forbidden_eval", str(frozen),
                    "--output", str(short),
                    "--bridge_output", str(bridge),
                    "--long_output", str(long),
                    "--report", str(report),
                    "--max_instructions", "3",
                    "--max_bridge_instructions", "5",
                    "--feedback_fraction", "0.5",
                    "--min_short_rows", "1",
                    "--workers", "1",
                ],
                cwd=root,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(
                phase0_run.returncode, 0,
                msg=f"stdout:\n{phase0_run.stdout}\nstderr:\n{phase0_run.stderr}",
            )
            self.assertEqual(len(short.read_text().splitlines()), 1)
            self.assertEqual(len(bridge.read_text().splitlines()), 1)
            self.assertEqual(len(long.read_text().splitlines()), 1)
            report_data = json.loads(report.read_text())
            self.assertEqual(report_data["short_rows"], 1)
            self.assertEqual(report_data["bridge_rows"], 1)
            self.assertEqual(report_data["long_rows"], 1)

            audit_report = root / "audit.json"
            subprocess.run(
                [
                    sys.executable,
                    str(audit_script),
                    "--dataset", str(short),
                    "--test_fields", "feedback_tests,acceptance_tests,tests",
                    "--run_references", "-1",
                    "--workers", "1",
                    "--require_phase0_approved",
                    "--require_neutral_contract",
                    "--report", str(audit_report),
                ],
                cwd=root,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            audited = json.loads(audit_report.read_text())
            self.assertEqual(audited["static_failures"], 0)
            self.assertEqual(audited["reference_reward_parity"]["checked_harnesses"], 3)

    def test_phase0_blocks_alpha_renamed_frozen_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            prep_script = copy_python_package_file(PREP, root)
            copy_python_package_file(CONTROLS, root)
            write_fake_evaluator(
                root,
                """
def evaluate_dart_jit_tests_detail(raw, test_code, task_id, timeout=30, stability_runs=None):
    return True, True, '', raw + '\n' + test_code
""",
            )
            train = root / "train.jsonl"
            frozen = root / "frozen.jsonl"
            train_row = {
                "task_id": "train-copy",
                "function": "calculate",
                "signature": "int calculate(int value)",
                "source": "int calculate(int value) { final result = value + 7; return result; }",
                "assembly": "calculate:\n add x0, x0, #7\n ret",
                "tests": "void main(){ final candidate = calculate; expect(candidate(1), 8); expect(candidate(2), 9); }",
            }
            eval_row = {
                "task_id": "frozen-original",
                "function": "answer",
                "signature": "int answer(int x)",
                "source": "int answer(int x) { final y = x + 7; return y; }",
                "assembly": "answer:\n add x0, x0, #7\n ret",
                "tests": "void main(){ final candidate = answer; expect(candidate(1), 8); expect(candidate(2), 9); }",
            }
            train.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            frozen.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")
            report = root / "phase0.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(prep_script),
                    "--input", str(train),
                    "--forbidden_eval", str(frozen),
                    "--output", str(root / "short.jsonl"),
                    "--bridge_output", str(root / "bridge.jsonl"),
                    "--long_output", str(root / "long.jsonl"),
                    "--report", str(report),
                    "--min_short_rows", "1",
                    "--workers", "1",
                ],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            report_data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_data["frozen_eval_overlap_rows"], 1)
            self.assertEqual(report_data["status"], "failed")
            overlap = report_data["frozen_eval_overlaps"][0]
            self.assertIn("alpha_structural_sha256", overlap["matched_fingerprint_kinds"])

    def test_neutral_evaluation_copy_is_evaluation_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            neutral_script = copy_python_package_file(NEUTRAL_EVAL, root)
            copy_python_package_file(CONTROLS, root)
            write_fake_evaluator(
                root,
                """
def evaluate_dart_jit_tests_detail(raw, test_code, task_id, timeout=30, stability_runs=None):
    return ('fn0' in raw), ('fn0' in raw and 'fn0' in test_code), '', raw + '\\n' + test_code
""",
            )
            source = root / "eval.jsonl"
            rows = []
            for index in range(2):
                rows.append(
                    {
                        "task_id": f"e{index}",
                        "function": f"meaningful{index}",
                        "signature": f"int meaningful{index}(int value)",
                        "source": f"int meaningful{index}(int value) => value + {index};",
                        "assembly": f"meaningful{index}:\n add x0, x0, {index}\n ret",
                        "tests": f"void main() {{\n final candidate = meaningful{index};\n expect(candidate(1), {index + 1});\n}}\n",
                    }
                )
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "neutral.jsonl"
            report = root / "neutral_report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(neutral_script),
                    "--input", str(source),
                    "--output", str(output),
                    "--report", str(report),
                    "--workers", "1",
                    "--min_rows", "2",
                ],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root)},
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            neutral_rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(neutral_rows), 2)
            self.assertTrue(all(infer_function_name(row) == "fn0" for row in neutral_rows))
            self.assertTrue(all(row["hybrid_metadata"]["evaluation_only"] for row in neutral_rows))
            self.assertTrue(all(not row["hybrid_metadata"]["phase0_approved"] for row in neutral_rows))

    def test_runner_dry_run_orders_fail_closed_gates_and_disables_stage2_dependence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            shutil.copytree(PATCH_ROOT / "scripts", root / "scripts")
            (root / "scripts/__init__.py").touch()
            (root / "scripts/training/__init__.py").touch()
            (root / "scripts/evaluation/__init__.py").touch()
            (root / "scripts/evaluation/graph_pass_at_k_antigravity.py").write_text(
                "print('{\"pass_at_10\": 0.0}')\n", encoding="utf-8"
            )
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            row = {
                "task_id": "t0",
                "function": "foo",
                "signature": "int foo()",
                "assembly": "foo:\n ret",
                "source": "int foo() => 0;",
                "tests": "void main() {\n final candidate = foo;\n expect(candidate(), 0);\n expect(candidate(), 0);\n}\n",
            }
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps({**row, "task_id": "e0"}) + "\n", encoding="utf-8")
            output = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/training/run_hybrid_curriculum_antigravity.py"),
                    "--project_root", str(root),
                    "--output_root", str(output),
                    "--train_file", str(train),
                    "--eval_file", str(eval_path),
                    "--teacher_model", "frontier-model",
                    "--no-run_grpo",
                    "--dry_run",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            manifest = json.loads((output / "hybrid_curriculum_manifest.json").read_text())
            stages = [item["stage"] for item in manifest["history"]]
            expected_prefix = [
                "00a_phase0_prepare",
                "00b_phase0_reference_audit",
                "00b2_phase0_dev_reference_audit",
                "00c_prepare_neutral_gate",
                "00d_neutral_gate_audit",
                "00e_representation_probe",
                "01_graph_only_sft",
            ]
            self.assertEqual(stages[: len(expected_prefix)], expected_prefix)
            self.assertIn("02a_graph_only_functional_gate", stages)
            collect = next(
                item for item in manifest["history"]
                if item["stage"] == "04_collect_verifier_feedback"
            )
            self.assertIn("--prediction_provenance", collect["command"])
            self.assertIn("--expected_checkpoint", collect["command"])
            self.assertIn("08a_gold_only_full_sft_control", stages)
            self.assertIn("08b_rejection_sampling_sft", stages)
            self.assertIn("09_rs_sft_functional_kill_switch", stages)
            self.assertNotIn("10_verpo", stages)

            control = next(
                item for item in manifest["history"]
                if item["stage"] == "08a_gold_only_full_sft_control"
            )
            stage2 = next(
                item for item in manifest["history"]
                if item["stage"] == "08b_rejection_sampling_sft"
            )
            for stage in (control, stage2):
                self.assertEqual(
                    stage["environment"]["GRAPH_PREFIX_DEPENDENCE_WEIGHT"], "0.0"
                )
                self.assertIn(
                    "approved_short_dev.jsonl",
                    stage["environment"]["GRAPH_EVAL_FILE"].replace("\\", "/"),
                )
                self.assertNotEqual(
                    stage["environment"]["GRAPH_TRAIN_FILE"],
                    stage["environment"]["GRAPH_EVAL_FILE"],
                )
                self.assertEqual(
                    stage["environment"]["GRAPH_PROMPT_ASSEMBLY_MODE"], "full"
                )
            for key in ("GRAPH_CHECKPOINT", "GRAPH_LR", "GRAPH_EPOCHS"):
                self.assertEqual(control["environment"][key], stage2["environment"][key])
            self.assertNotEqual(
                control["environment"]["GRAPH_TRAIN_FILE"],
                stage2["environment"]["GRAPH_TRAIN_FILE"],
            )

            mix = next(
                item for item in manifest["history"]
                if item["stage"] == "07_build_balanced_sft_mix"
            )
            self.assertIn("--gold_control_out", mix["command"])

            gate = next(
                item for item in manifest["history"]
                if item["stage"] == "02a_graph_only_functional_gate"
            )
            self.assertIn("--causality_prompt_mode", gate["command"])
            self.assertIn("--min_causal_permutation_drop_pp", gate["command"])
            self.assertIn("--min_causal_task_losses", gate["command"])
            self.assertIn("--bootstrap_iterations", gate["command"])
            self.assertIn("--max_sign_test_p_value", gate["command"])
            self.assertIn("--min_causal_effective_pairs", gate["command"])
            self.assertIn("--min_rows", gate["command"])

            kill_switch = next(
                item for item in manifest["history"]
                if item["stage"] == "09_rs_sft_functional_kill_switch"
            )
            baseline_index = kill_switch["command"].index("--baseline_checkpoint") + 1
            self.assertIn(
                "08a_gold_only_full_sft_control/pytorch_model.bin",
                kill_switch["command"][baseline_index].replace("\\", "/"),
            )

    def test_grpo_anchor_is_separate_verified_only_and_resamples(self):
        text = GRPO.read_text(encoding="utf-8")
        self.assertIn("GRPO_VERIFIED_ANCHOR_FILE", text)
        self.assertIn("ordinary RL-row references are never", text)
        self.assertIn("GRPO_DYNAMIC_RESAMPLE_ATTEMPTS", text)
        self.assertIn("GRPO_REWARD_TEST_FIELD", text)
        self.assertIn("GRPO_SFT_ANCHOR_ON_NO_SIGNAL", text)
        self.assertIn('os.environ.get("GRPO_SFT_ANCHOR_ON_NO_SIGNAL", "0")', text)
        self.assertIn("while True:", text)
        self.assertIn("resample_attempts", text)

    def test_curriculum_keeps_facts_first_anchor_and_uses_matched_step_baseline(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"GRAPH_FACTS_FIRST_TARGET": "1"', text)
        self.assertIn("gold_only_matched_steps.jsonl", text)
        self.assertIn("08a_gold_only_full_sft_control", text)
        self.assertIn("matched-step, matched-modality control", text)
        self.assertIn("baseline_checkpoint = gold_control_checkpoint", text)
        self.assertNotIn("baseline_checkpoint = Path(args.initial_checkpoint", text)

    def test_functional_gate_is_free_running_graph_only_causality(self):
        gate_text = FUNCTIONAL_GATE.read_text(encoding="utf-8")
        inference_text = INFERENCE.read_text(encoding="utf-8")
        self.assertIn("causality_prompt_mode", gate_text)
        self.assertIn('"graph_only"', gate_text)
        self.assertIn("matched_permutation", gate_text)
        self.assertIn("performance_prompt_mode", gate_text)
        self.assertIn("min_improvement_pp", gate_text)
        self.assertIn("prompt_stream_sha256", gate_text)
        self.assertIn("different text prompts", gate_text)
        self.assertIn("matched_permutation", inference_text)
        self.assertIn("shape-matched BIJECTION", inference_text)
        self.assertIn("'null'", inference_text)
        self.assertIn("force_null_graph", inference_text)
        self.assertIn("final_context_zeroed", inference_text)
        self.assertIn('"null"', gate_text)

    def test_functional_gate_requires_complete_candidate_coverage(self):
        module = load_module(FUNCTIONAL_GATE, "functional_gate_coverage_test")
        module.load_evaluator = lambda: (
            lambda raw, tests, task_id, timeout=30: (True, True, "", raw + tests)
        )
        dataset = [
            {
                "task_id": f"t{index}",
                "function": "fn0",
                "signature": "int fn0()",
                "assembly": "mov x0, #7\nret",
                "tests": "void main(){ final candidate = fn0; expect(candidate(), 7); }",
                "binary_facts": mechanical_facts(
                    {
                        "function": "fn0",
                        "signature": "int fn0()",
                        "assembly": "mov x0, #7\nret",
                    }
                ),
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            incomplete = root / "incomplete.json"
            incomplete.write_text(
                json.dumps([{"source_line": 1, "predictions": ["a", "b"]}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "1 rows for a 2-row"):
                module.evaluate_predictions(
                    dataset,
                    incomplete,
                    k_values=[2],
                    workers=1,
                    timeout=1,
                    expected_candidates=2,
                )

            wrong_k = root / "wrong_k.json"
            wrong_k.write_text(
                json.dumps(
                    [
                        {"source_line": 1, "predictions": ["a", "b"]},
                        {"source_line": 2, "predictions": ["a"]},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected exactly 2"):
                module.evaluate_predictions(
                    dataset,
                    wrong_k,
                    k_values=[2],
                    workers=1,
                    timeout=1,
                    expected_candidates=2,
                )

    def test_probe_installs_architecture_environment_before_model_imports(self):
        text = PROBE.read_text(encoding="utf-8")
        env_index = text.index('os.environ["GRAPH_DECODER_MODEL"]')
        model_import_index = text.index(
            "from scripts.evaluation.graph_inference_antigravity import build_blocks"
        )
        self.assertLess(env_index, model_import_index)
        self.assertIn("train-fitted ridge map", text)
        self.assertIn('"numeric_constant": 16', text)
        self.assertIn('f"{group}:hash_{index}"', text)
        self.assertIn("permuted-label control", text)

    def test_matched_permutation_is_deranged_and_swaps_donor_assembly(self):
        tree = ast.parse(INFERENCE.read_text(encoding="utf-8"))
        wanted = {"_matched_derangement", "_copy_graph_payload"}
        nodes = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
        ]
        module_ast = ast.Module(body=nodes, type_ignores=[])
        ast.fix_missing_locations(module_ast)
        import copy
        import hashlib
        namespace = {
            "copy": copy,
            "hashlib": hashlib,
            "_row_identity": lambda row, index: row["id"],
            "_graph_complexity": lambda row: (row["blocks"], row["instructions"]),
        }
        exec(compile(module_ast, str(INFERENCE), "exec"), namespace)
        rows = [
            {"id": f"t{i}", "blocks": i + 1, "instructions": 10 + i, "assembly": f"asm-{i}"}
            for i in range(5)
        ]
        mapping = namespace["_matched_derangement"](rows, 42)
        self.assertEqual(sorted(mapping), list(range(len(rows))))
        self.assertTrue(all(index != donor for index, donor in enumerate(mapping)))
        target = {"id": "target", "assembly": "target-asm", "tests": "target-tests"}
        donor = {"id": "donor", "assembly": "donor-asm", "cfg": [{"id": 0}], "edges": []}
        swapped = namespace["_copy_graph_payload"](target, donor)
        self.assertEqual(swapped["assembly"], "donor-asm")
        self.assertEqual(swapped["tests"], "target-tests")

    def test_checkpoint_contract_accepts_frozen_missing_and_rejects_drift(self):
        import torch

        module = torch.nn.Module()
        module.register_parameter("trainable", torch.nn.Parameter(torch.ones(2)))
        module.register_parameter(
            "frozen", torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        )
        contract = load_module(CHECKPOINT_CONTRACT, "checkpoint_contract_test")

        state = {"trainable": torch.full((2,), 3.0)}
        missing, unexpected = module.load_state_dict(state, strict=False)
        report = contract.validate_trainable_checkpoint_load(
            module, state, missing_keys=missing, unexpected_keys=unexpected
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["absent_trainable_tensor_count"], 0)
        self.assertEqual(report["missing_frozen_tensor_count"], 1)

        with self.assertRaises(RuntimeError):
            contract.validate_trainable_checkpoint_load(
                module, {"frozen": torch.zeros(2)}, context="missing adapter"
            )
        with self.assertRaises(RuntimeError):
            contract.validate_trainable_checkpoint_load(
                module,
                {"trainable": torch.ones(2), "obsolete_adapter": torch.ones(1)},
                unexpected_keys=["obsolete_adapter"],
                context="wrong prefix architecture",
            )
        overridden = contract.validate_trainable_checkpoint_load(
            module,
            {"frozen": torch.zeros(2)},
            context="research override",
            allow_partial=True,
        )
        self.assertEqual(overridden["status"], "overridden")

    def test_architecture_provenance_loader_filters_side_effect_settings(self):
        runner = load_module(RUNNER, "hybrid_runner_arch_test")
        with tempfile.TemporaryDirectory() as temp:
            provenance = Path(temp) / "run_provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "graph_environment": {
                            "GRAPH_DECODER_MODEL": "Qwen/Qwen3-8B",
                            "GRAPH_ENCODER_PEFT": "lora",
                            "GRAPH_DECODER_PEFT": "lora",
                            "GRAPH_LORA_R": "64",
                            "GRAPH_LORA_ALPHA": "128",
                            "GRAPH_QWEN_PREFIX_TOKENS": "64",
                            "GRAPH_DECODER_PROMPT_MAX_LENGTH": "2048",
                            "GRAPH_OUTPUT_DIR": "/should/not/import",
                            "GRAPH_TRAIN_FILE": "/should/not/import.jsonl",
                            "HF_TOKEN": "secret",
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = types.SimpleNamespace(
                architecture_env_json=str(provenance),
                probe_checkpoint="",
                initial_checkpoint="",
                stage1_checkpoint="",
            )
            environment, source = runner.load_architecture_environment(args)
            self.assertEqual(source, provenance.resolve())
            self.assertEqual(environment["GRAPH_LORA_R"], "64")
            self.assertEqual(environment["GRAPH_QWEN_PREFIX_TOKENS"], "64")
            self.assertEqual(environment["GRAPH_QWEN_LORA_TARGETS"], "attention")
            self.assertEqual(environment["GRAPH_DECODER_PROMPT_MAX_LENGTH"], "2048")
            self.assertNotIn("GRAPH_OUTPUT_DIR", environment)
            self.assertNotIn("GRAPH_TRAIN_FILE", environment)
            self.assertNotIn("HF_TOKEN", environment)

    def test_checkpoint_validation_is_used_by_every_model_entrypoint(self):
        for path in (SFT, GRPO, INFERENCE, PROBE):
            text = path.read_text(encoding="utf-8")
            self.assertIn("validate_trainable_checkpoint_load", text, path.name)
        installer = (PATCH_ROOT / "apply_hybrid_patch.py").read_text(encoding="utf-8")
        self.assertIn("scripts/training/checkpoint_contract.py", installer)


    def test_statistical_gate_rejects_one_task_noise_and_accepts_consistent_effect(self):
        module = load_module(FUNCTIONAL_GATE, "functional_gate_statistics_test")
        self.assertAlmostEqual(module.exact_one_sided_sign_p_value(1, 0), 0.5)
        self.assertAlmostEqual(module.exact_one_sided_sign_p_value(5, 0), 0.03125)

        noisy_left = [
            {"task_key": f"t{index}", "pass@10": 1.0 if index == 0 else 0.0}
            for index in range(96)
        ]
        noisy_right = [
            {"task_key": f"t{index}", "pass@10": 0.0}
            for index in range(96)
        ]
        noisy = module.paired_task_comparison(
            noisy_left,
            noisy_right,
            metric="pass@10",
            iterations=2000,
            confidence=0.95,
            seed=42,
        )
        failures = module._paired_statistical_failures(
            "one-task effect",
            noisy,
            minimum_effective_pairs=8,
            maximum_p_value=0.05,
            minimum_lower_bound_pp=0.0,
        )
        self.assertTrue(any("only 1" in failure for failure in failures))
        self.assertTrue(any("p=0.5" in failure for failure in failures))

        strong_left = [
            {"task_key": f"t{index}", "pass@10": 1.0 if index < 8 else 0.0}
            for index in range(96)
        ]
        strong = module.paired_task_comparison(
            strong_left,
            noisy_right,
            metric="pass@10",
            iterations=2000,
            confidence=0.95,
            seed=42,
        )
        self.assertLess(strong["exact_one_sided_sign_p_value"], 0.05)
        self.assertGreater(strong["bootstrap"]["one_sided_lower_pp"], 0.0)
        self.assertEqual(
            module._paired_statistical_failures(
                "consistent effect",
                strong,
                minimum_effective_pairs=8,
                maximum_p_value=0.05,
                minimum_lower_bound_pp=0.0,
            ),
            [],
        )

    def test_qwen_lora_defaults_to_attention_plus_mlp(self):
        source = SFT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        wanted = {"decoder_lora_target_modules"}
        nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "QWEN_LORA_TARGET_MODES" for target in node.targets)
        ]
        module_ast = ast.Module(body=assignments + nodes, type_ignores=[])
        ast.fix_missing_locations(module_ast)
        namespace = {"os": os}
        exec(compile(module_ast, str(SFT), "exec"), namespace)
        previous = os.environ.pop("GRAPH_QWEN_LORA_TARGETS", None)
        try:
            defaults = namespace["decoder_lora_target_modules"](
                "Qwen/Qwen3-8B", is_causal=True
            )
            self.assertEqual(
                defaults,
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            os.environ["GRAPH_QWEN_LORA_TARGETS"] = "attention"
            historical = namespace["decoder_lora_target_modules"](
                "Qwen/Qwen3-8B", is_causal=True
            )
            self.assertEqual(historical, ["q_proj", "k_proj", "v_proj", "o_proj"])
        finally:
            if previous is None:
                os.environ.pop("GRAPH_QWEN_LORA_TARGETS", None)
            else:
                os.environ["GRAPH_QWEN_LORA_TARGETS"] = previous

    def test_checkpoint_contract_allows_only_explicit_new_trainables(self):
        import torch

        module = torch.nn.Module()
        module.register_parameter("existing", torch.nn.Parameter(torch.ones(1)))
        module.register_parameter("new_adapter", torch.nn.Parameter(torch.zeros(1)))
        contract = load_module(CHECKPOINT_CONTRACT, "checkpoint_contract_expansion_test")
        state = {"existing": torch.ones(1)}
        missing, unexpected = module.load_state_dict(state, strict=False)
        report = contract.validate_trainable_checkpoint_load(
            module,
            state,
            missing_keys=missing,
            unexpected_keys=unexpected,
            allowed_absent_trainable_keys=["new_adapter"],
            context="documented adapter expansion",
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["allowed_absent_trainable_tensor_count"], 1)
        self.assertEqual(report["unapproved_absent_trainable_tensor_count"], 0)
        with self.assertRaises(RuntimeError):
            contract.validate_trainable_checkpoint_load(
                module,
                state,
                missing_keys=missing,
                unexpected_keys=unexpected,
                context="undeclared adapter expansion",
            )



    def test_qwen_attention_checkpoint_expansion_is_zero_output_and_explicit(self):
        import torch

        source = SFT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "qwen_lora_expansion_allowed_keys"
        ]
        module_ast = ast.Module(body=nodes, type_ignores=[])
        ast.fix_missing_locations(module_ast)
        namespace = {"os": os, "torch": torch}
        exec(compile(module_ast, str(SFT), "exec"), namespace)

        class LoraProjection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({
                    "default": torch.nn.Linear(2, 1, bias=False)
                })
                self.lora_B = torch.nn.ModuleDict({
                    "default": torch.nn.Linear(1, 2, bias=False)
                })
                torch.nn.init.zeros_(self.lora_B["default"].weight)

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = LoraProjection()
                self.up_proj = LoraProjection()
                self.down_proj = LoraProjection()

        model = DummyModel()
        warm_state = {
            "decoder.layers.0.q_proj.lora_A.default.weight": torch.ones(1, 2),
            "decoder.layers.0.q_proj.lora_B.default.weight": torch.zeros(2, 1),
        }
        previous = {
            key: os.environ.get(key)
            for key in (
                "GRAPH_ALLOW_QWEN_LORA_EXPANSION",
                "GRAPH_DECODER_PEFT",
                "GRAPH_QWEN_LORA_TARGETS",
            )
        }
        try:
            os.environ["GRAPH_ALLOW_QWEN_LORA_EXPANSION"] = "1"
            os.environ["GRAPH_DECODER_PEFT"] = "lora"
            os.environ["GRAPH_QWEN_LORA_TARGETS"] = "attention_mlp"
            allowed = namespace["qwen_lora_expansion_allowed_keys"](
                model, warm_state
            )
            self.assertEqual(len(allowed), 6)
            self.assertTrue(all("lora_" in key for key in allowed))
            self.assertTrue(all(
                any(module in key for module in ("gate_proj", "up_proj", "down_proj"))
                for key in allowed
            ))

            with torch.no_grad():
                model.gate_proj.lora_B["default"].weight.fill_(1.0)
            with self.assertRaisesRegex(RuntimeError, "not zero-initialized"):
                namespace["qwen_lora_expansion_allowed_keys"](model, warm_state)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_qwen_lora_target_contract_is_not_required_for_non_qwen_models(self):
        runner = load_module(RUNNER, "hybrid_runner_non_qwen_contract_test")
        self.assertNotIn(
            "GRAPH_QWEN_LORA_TARGETS", runner.ESSENTIAL_WARM_START_KEYS
        )




if __name__ == "__main__":
    unittest.main(verbosity=2)
