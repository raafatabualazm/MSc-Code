from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = (
    PATCH_ROOT
    / "scripts"
    / "preprocessing"
    / "prepare_phase0_2776_expansion.py"
)
SANITIZE_PATH = (
    PATCH_ROOT
    / "scripts"
    / "preprocessing"
    / "sanitize_phase0_supplemental_targets.py"
)
BUILD_PATH = (
    PATCH_ROOT
    / "scripts"
    / "preprocessing"
    / "build_phase0_2776_multifunction_expansion.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("phase0_2776_prepare_test", PREPARE_PATH)
sanitize = load_module("phase0_2776_sanitize_test", SANITIZE_PATH)
builder = load_module("phase0_2776_builder_test", BUILD_PATH)


class _CapturingEvaluator:
    def __init__(self) -> None:
        self.raw = ""

    def evaluate_dart_jit_tests_detail(
        self,
        raw,
        tests,
        task_id,
        *,
        timeout,
        stability_runs,
    ):
        del tests, task_id, timeout, stability_runs
        self.raw = raw
        return True, True, "", raw


class Phase02776ExpansionTests(unittest.TestCase):
    def test_set_arithmetic_and_supplemental_family_contract(self) -> None:
        self.assertEqual(
            prepare.PHASE0_TRAIN_ROWS - prepare.HELDOUT_ROWS,
            prepare.EXPANDED_FIT_ROWS,
        )
        self.assertEqual(
            prepare.PARENT_FIT_ROWS + prepare.SUPPLEMENTAL_ROWS,
            prepare.EXPANDED_FIT_ROWS,
        )
        self.assertEqual(
            sum(prepare.SUPPLEMENTAL_FAMILY_COUNTS.values()),
            prepare.SUPPLEMENTAL_ROWS,
        )

    def test_private_target_keeps_supporting_declarations_and_uses_fn0(
        self,
    ) -> None:
        raw_sha = "a" * 64
        prepared_source = (
            "enum TrafficLight { red, green }\n"
            "extension Label on TrafficLight {\n"
            "  String get text => name;\n"
            "}\n"
            "String candidate() => TrafficLight.red.text;\n"
        )
        result = prepare._base_target_row(
            label={
                "task_id": "sigless_test",
                "family": "master",
                "dart_source": prepared_source,
            },
            build_row={
                "task_id": "sigless_test",
                "function_source": prepared_source,
                "function_source_sha256": "b" * 64,
                "analysis_program_sha256": "c" * 64,
                "split_row": 7,
                "phase0_manifest_line": 9,
                "compact_private_metadata": {
                    "input_row_sha256": raw_sha,
                },
            },
            source_row={
                "task_id": "sigless_test",
                "function": "solve",
                "tests": (
                    "void main() {\n"
                    "  final candidate = solve();\n"
                    "  if (candidate != 'red') throw StateError('solve');\n"
                    "}\n"
                ),
            },
            source_raw_sha256=raw_sha,
            contract={
                "codec_sha256": "d" * 64,
                "codebook_sha256": "e" * 64,
                "tokenizer_json_sha256": "f" * 64,
            },
        )
        self.assertTrue(result["dart_source"].startswith("enum TrafficLight"))
        self.assertIn("String fn0()", result["dart_source"])
        self.assertNotIn("String candidate()", result["dart_source"])
        self.assertIn("final candidate = fn0();", result["tests"])
        # String contents are not rewritten by the lexical identifier control.
        self.assertIn("StateError('solve')", result["tests"])

    def test_gold_evaluation_uses_lossless_full_program_fence(self) -> None:
        evaluator = _CapturingEvaluator()
        source = (
            "enum TrafficLight { red }\n"
            "String fn0() => TrafficLight.red.name;\n"
        )
        tests = "void main() { if (fn0() != 'red') throw StateError('x'); }\n"
        outcome = sanitize.evaluate_row(
            evaluator,
            {
                "task_id": "sigless_test",
                "dart_source": source,
                "tests": tests,
                "acceptance_tests": tests,
            },
            task_suffix="unit",
            timeout=2,
            stability_runs=1,
        )
        self.assertTrue(outcome["passed"])
        self.assertEqual(
            evaluator.raw,
            f"```dart\n{source.rstrip()}\n```",
        )

    def test_expanded_files_are_byte_appends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.jsonl"
            supplement = root / "supplement.jsonl"
            expanded = root / "expanded.jsonl"
            parent_bytes = b'{"task_id":"parent"}\n'
            supplement_bytes = b'{"task_id":"supplement"}\n'
            parent.write_bytes(parent_bytes)
            supplement.write_bytes(supplement_bytes)
            builder.atomic_concat(expanded, parent, supplement)
            payload = expanded.read_bytes()
            self.assertEqual(payload, parent_bytes + supplement_bytes)
            self.assertEqual(payload[: len(parent_bytes)], parent_bytes)


if __name__ == "__main__":
    unittest.main()
