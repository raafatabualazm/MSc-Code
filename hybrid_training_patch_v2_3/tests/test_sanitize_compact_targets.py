from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parents[1]
SANITIZER_PATH = (
    PATCH_ROOT / "scripts" / "preprocessing" / "sanitize_compact_targets.py"
)
EVALUATOR_PATH = (
    PATCH_ROOT / "scripts" / "evaluation" / "graph_compile_at_k_antigravity.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sanitizer = load_module("compact_target_sanitizer_test", SANITIZER_PATH)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def row(task_id: str, source: str, expected: int) -> dict:
    tests = (
        "void main() {\n"
        f"  if (fn0() != {expected}) throw StateError('bad result');\n"
        "}\n"
    )
    return {
        "task_id": task_id,
        "dart_source": source,
        "tests": tests,
        "acceptance_tests": tests,
        "feedback_tests": "",
    }


@unittest.skipUnless(shutil.which("dart"), "Dart is required")
class CompactTargetSanitationTests(unittest.TestCase):
    def test_unused_forbidden_imports_are_stripped_and_used_import_is_quarantined(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "contract.json"
            write_json(contract, {"schema": "test-contract"})
            contract_sha = sanitizer.sha256_file(contract)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            write_jsonl(
                train,
                [
                    row(
                        "train_safe",
                        "import 'dart:io';\nint fn0() => 2;\n",
                        2,
                    ),
                    row(
                        "train_unsafe",
                        "import 'dart:io';\n"
                        "int fn0() => Platform.isWindows ? 2 : 2;\n",
                        2,
                    ),
                ],
            )
            write_jsonl(
                dev,
                [
                    row(
                        "dev_safe",
                        "import 'dart:io';\nint fn0() => 3;\n",
                        3,
                    )
                ],
            )
            train_seal = root / "train.seal.json"
            dev_seal = root / "dev.seal.json"
            write_json(
                train_seal,
                {
                    "schema": sanitizer.SEAL_SCHEMA,
                    "selected_role": "fit",
                    "output_sha256": sanitizer.sha256_file(train),
                    "contract_sha256": contract_sha,
                    "rows": 2,
                },
            )
            write_json(
                dev_seal,
                {
                    "schema": sanitizer.SEAL_SCHEMA,
                    "selected_role": "measure",
                    "output_sha256": sanitizer.sha256_file(dev),
                    "contract_sha256": contract_sha,
                    "rows": 1,
                },
            )
            evaluator = sanitizer.load_evaluator(EVALUATOR_PATH)
            args = argparse.Namespace(
                input_train=train,
                expected_input_train_sha256=sanitizer.sha256_file(train),
                input_train_seal=train_seal,
                expected_input_train_seal_sha256=sanitizer.sha256_file(
                    train_seal
                ),
                input_dev=dev,
                expected_input_dev_sha256=sanitizer.sha256_file(dev),
                input_dev_seal=dev_seal,
                expected_input_dev_seal_sha256=sanitizer.sha256_file(dev_seal),
                contract=contract,
                expected_contract_sha256=contract_sha,
                evaluator=EVALUATOR_PATH,
                expected_evaluator_sha256=sanitizer.sha256_file(EVALUATOR_PATH),
                output_imitation_train=root / "imitation.train.jsonl",
                output_imitation_train_seal=(
                    root / "imitation.train.seal.json"
                ),
                output_train=root / "clean.train.jsonl",
                output_train_seal=root / "clean.train.seal.json",
                output_dev=root / "clean.dev.jsonl",
                output_dev_seal=root / "clean.dev.seal.json",
                quarantine=root / "quarantine.jsonl",
                report=root / "report.json",
                expected_input_train_rows=2,
                expected_output_train_rows=1,
                expected_dev_rows=1,
                expected_sanitized_train_task_ids="train_safe",
                expected_quarantined_train_task_ids="train_unsafe",
                expected_sanitized_dev_task_ids="dev_safe",
                expected_dart_version=sanitizer.dart_version(
                    str(evaluator.DART_BIN)
                ),
                timeout=30,
                stability_runs=1,
                workers=2,
            )
            report = sanitizer.build(args)
            self.assertEqual(report["counts"]["output_imitation_train"], 2)
            self.assertEqual(report["counts"]["output_train"], 1)
            self.assertEqual(report["counts"]["output_dev"], 1)
            self.assertEqual(
                report["task_sets"]["quarantined_train"], ["train_unsafe"]
            )
            clean_train = sanitizer.read_jsonl(
                args.output_train, "clean train"
            )
            imitation_train = sanitizer.read_jsonl(
                args.output_imitation_train, "imitation train"
            )
            clean_dev = sanitizer.read_jsonl(args.output_dev, "clean dev")
            self.assertNotIn("dart:io", clean_train[0]["dart_source"])
            self.assertNotIn("dart:io", clean_dev[0]["dart_source"])
            self.assertEqual(
                {item["task_id"] for item in imitation_train},
                {"train_safe", "train_unsafe"},
            )
            unsafe_imitation = next(
                item
                for item in imitation_train
                if item["task_id"] == "train_unsafe"
            )
            self.assertIn("dart:io", unsafe_imitation["dart_source"])
            quarantine = sanitizer.read_jsonl(
                args.quarantine, "quarantine"
            )
            self.assertIn(
                "Platform",
                quarantine[0]["original_row"]["dart_source"],
            )
            self.assertTrue(
                report["policy"]["all_emitted_gold_targets_recertified"]
            )


if __name__ == "__main__":
    unittest.main()
