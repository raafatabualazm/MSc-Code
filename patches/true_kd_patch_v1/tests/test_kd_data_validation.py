from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(DATA_DIR))

from validate_qwen_kd_artifacts import (  # noqa: E402
    SPARSE_MANIFEST_SCHEMA,
    SPARSE_ROW_SCHEMA,
    audit_legacy_qwen_logprobs,
    sha256_file,
    validate_sparse_topk_tail_dataset,
)


class LegacyQwenAuditTests(unittest.TestCase):
    def test_legacy_topk_strings_are_rejected_for_kd_but_keep_verified_code(
        self,
    ) -> None:
        row = {
            "task_id": "task-1",
            "ok": True,
            "repair": "ab",
            "code": "ab",
            "teacher_tokens": ["a", "b"],
            "teacher_logprobs": [
                {
                    "t": "a",
                    "lp": -0.1,
                    "top": [
                        {"t": "a", "lp": -0.1},
                        {"t": "x", "lp": -3.0},
                    ],
                },
                {
                    "t": "b",
                    "lp": -0.2,
                    "top": [
                        {"t": "b", "lp": -0.2},
                        {"t": "y", "lp": -2.0},
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = audit_legacy_qwen_logprobs(
                path, expected_training_rows=10
            )
        self.assertFalse(report["usable_for_dense_full_kl"])
        self.assertFalse(report["usable_for_sparse_topk_tail_kl"])
        self.assertTrue(report["usable_verified_code_for_rs_sft"])
        self.assertEqual(report["row_coverage"], 0.1)
        self.assertIn("teacher_tokenizer_sha256", report["missing_provenance_fields"])


class StrictSparseValidationTests(unittest.TestCase):
    def _write_valid_artifact(
        self, root: Path
    ) -> tuple[Path, Path, Path, str]:
        tokenizer = root / "tokenizer.json"
        tokenizer.write_text('{"sealed":"tokenizer"}\n', encoding="utf-8")
        contract_sha = "c" * 64
        row = {
            "schema": SPARSE_ROW_SCHEMA,
            "task_id": "task-1",
            "compact_input_ids": [100],
            "compact_codec_sha256": "a" * 64,
            "compact_codebook_sha256": "b" * 64,
            "compact_tokenizer_sha256": "d" * 64,
            "function": "fn0",
            "lang": "Dart",
            "target_input_ids": [1, 2],
            "teacher_positions": [
                {
                    "observed_token_id": 1,
                    "top_token_ids": [1, 3],
                    "top_logprobs": [math.log(0.6), math.log(0.3)],
                    "tail_mass": 0.1,
                },
                {
                    "observed_token_id": 2,
                    "top_token_ids": [2, 0],
                    "top_logprobs": [math.log(0.7), math.log(0.2)],
                    "tail_mass": 0.1,
                },
            ],
        }
        data = root / "sparse.jsonl"
        data.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest = {
            "schema": SPARSE_MANIFEST_SCHEMA,
            "data_sha256": sha256_file(data),
            "rows": 1,
            "source_dataset_sha256": "1" * 64,
            "contract_sha256": contract_sha,
            "collector_sha256": "2" * 64,
            "teacher_model": "teacher",
            "teacher_revision": "revision",
            "teacher_tokenizer_sha256": sha256_file(tokenizer),
            "student_tokenizer_sha256": sha256_file(tokenizer),
            "student_vocab_size": 4,
            "logprob_base": "natural",
            "probability_temperature": 1.0,
            "logprob_rounding": "none",
            "includes_eos": True,
            "eos_token_id": 2,
            "topk_tail_partition": True,
        }
        manifest_path = root / "sparse.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return data, manifest_path, tokenizer, contract_sha

    def test_valid_sparse_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data, manifest, tokenizer, contract_sha = self._write_valid_artifact(
                Path(temporary)
            )
            parsed_manifest, rows = validate_sparse_topk_tail_dataset(
                data,
                manifest,
                student_tokenizer_json=tokenizer,
                expected_contract_sha256=contract_sha,
            )
        self.assertEqual(parsed_manifest["rows"], 1)
        self.assertEqual(len(rows), 1)

    def test_tokenizer_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, tokenizer, contract_sha = self._write_valid_artifact(root)
            parsed = json.loads(manifest.read_text(encoding="utf-8"))
            parsed["teacher_tokenizer_sha256"] = "0" * 64
            manifest.write_text(json.dumps(parsed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                validate_sparse_topk_tail_dataset(
                    data,
                    manifest,
                    student_tokenizer_json=tokenizer,
                    expected_contract_sha256=contract_sha,
                )

    def test_missing_eos_distribution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, tokenizer, contract_sha = self._write_valid_artifact(root)
            row = json.loads(data.read_text(encoding="utf-8"))
            row["target_input_ids"] = [1]
            row["teacher_positions"] = row["teacher_positions"][:1]
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            parsed = json.loads(manifest.read_text(encoding="utf-8"))
            parsed["data_sha256"] = sha256_file(data)
            manifest.write_text(json.dumps(parsed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "end with"):
                validate_sparse_topk_tail_dataset(
                    data,
                    manifest,
                    student_tokenizer_json=tokenizer,
                    expected_contract_sha256=contract_sha,
                )

    def test_unsealed_probability_mass_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, tokenizer, contract_sha = self._write_valid_artifact(root)
            row = json.loads(data.read_text(encoding="utf-8"))
            row["teacher_positions"][0]["tail_mass"] = 0.2
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            parsed = json.loads(manifest.read_text(encoding="utf-8"))
            parsed["data_sha256"] = sha256_file(data)
            manifest.write_text(json.dumps(parsed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not one"):
                validate_sparse_topk_tail_dataset(
                    data,
                    manifest,
                    student_tokenizer_json=tokenizer,
                    expected_contract_sha256=contract_sha,
                )


if __name__ == "__main__":
    unittest.main()
