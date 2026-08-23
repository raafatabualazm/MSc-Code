from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.verify_fixed_scrub_v3_comparator import (
    EXPECTED_GRAPH_ENV,
    candidate_digest,
    sha256,
    verify,
)


class VerifyFixedScrubV3ComparatorTests(unittest.TestCase):
    def fixture(self, root: Path) -> argparse.Namespace:
        predictions_path = root / "comparator_predictions.json"
        provenance_path = root / "comparator_predictions.json.provenance.json"
        dataset_path = root / "benchmark.jsonl"
        source = {
            "task_id": "7",
            "filename": "7.dart",
            "dart_source": "int target() => 7;",
            "tests": "void main() {}",
        }
        dataset_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
        predictions = [
            {
                "id": "7",
                "source_line": 1,
                "filename": "7.dart",
                "predictions": ["int target() => 7;"],
                "reference": source["dart_source"],
                "tests": source["tests"],
                "graph_input_ablation": {
                    "mode": "none",
                    "target_id": "7",
                    "donor_id": "7",
                },
            }
        ]
        predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
        decoder_revision = "decoder-revision"
        encoder_revision = "encoder-revision"
        renderer_hash = "a" * 64
        inference_hash = "b" * 64
        graph_env = dict(EXPECTED_GRAPH_ENV)
        graph_env["GRAPH_DECODER_REVISION"] = decoder_revision
        graph_env["GRAPH_ENCODER_REVISION"] = encoder_revision
        provenance = {
            "schema_version": 1,
            "prompt_schema_version": "antigravity-v2-no-test-hints",
            "prompt_stream_sha256": "c" * 64,
            "scoring_tests_visible_to_policy": False,
            "row_count": 1,
            "seed": 42,
            "generation": {
                "num_samples": 1,
                "generation_batch_size": 1,
                "max_new_tokens": 768,
                "decoder_prompt_max_length": 2048,
                "use_cache": True,
                "decoder_gradient_checkpointing": False,
            },
            "output": {
                "sha256": sha256(predictions_path),
                "size_bytes": predictions_path.stat().st_size,
            },
            "dataset": {
                "sha256": sha256(dataset_path),
                "size_bytes": dataset_path.stat().st_size,
            },
            "checkpoint": {"sha256": "d" * 64, "size_bytes": 123},
            "models": {
                "decoder": {
                    "requested_id": "Qwen/Qwen3-8B",
                    "requested_revision": decoder_revision,
                    "resolved_commit": decoder_revision,
                },
                "encoder": {
                    "requested_id": "microsoft/graphcodebert-base",
                    "requested_revision": encoder_revision,
                    "resolved_commit": encoder_revision,
                },
            },
            "graph_environment": graph_env,
            "graph_input_ablation": {"mode": "none", "seed": 42, "self_mapped_rows": 1},
            "source_files": [
                {
                    "path": "/workspace/scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py",
                    "sha256": renderer_hash,
                    "size_bytes": 10,
                },
                {
                    "path": "/workspace/scripts/evaluation/graph_inference_antigravity.py",
                    "sha256": inference_hash,
                    "size_bytes": 11,
                },
            ],
        }
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        return argparse.Namespace(
            predictions=predictions_path,
            provenance=provenance_path,
            dataset=dataset_path,
            expected_rows=1,
            expected_samples=1,
            seed=42,
            expected_predictions_sha256=sha256(predictions_path),
            expected_provenance_sha256=sha256(provenance_path),
            expected_dataset_sha256=sha256(dataset_path),
            checkpoint_sha256="d" * 64,
            expected_prompt_stream_sha256="c" * 64,
            decoder_revision=decoder_revision,
            encoder_revision=encoder_revision,
            renderer_sha256=renderer_hash,
            inference_source_sha256=inference_hash,
        )

    @staticmethod
    def reseal_provenance(args: argparse.Namespace) -> None:
        args.expected_provenance_sha256 = sha256(args.provenance)

    def test_accepts_and_reports_legacy_runtime_observation_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            result = verify(args)
            self.assertTrue(result["verified"])
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["candidate_stream_sha256"], candidate_digest(json.loads(args.predictions.read_text())))
            self.assertFalse(result["runtime_observations"]["checkpoint_load"]["verified"])
            self.assertFalse(result["runtime_observations"]["graph_prefix_gate"]["verified"])

    def test_rejects_prediction_not_bound_by_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            rows = json.loads(args.predictions.read_text(encoding="utf-8"))
            rows[0]["predictions"][0] = "int target() => 8;"
            args.predictions.write_text(json.dumps(rows), encoding="utf-8")
            args.expected_predictions_sha256 = sha256(args.predictions)
            with self.assertRaisesRegex(SystemExit, "does not bind prediction output"):
                verify(args)

    def test_rejects_prompt_stream_change_even_when_provenance_is_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
            provenance["prompt_stream_sha256"] = "e" * 64
            args.provenance.write_text(json.dumps(provenance), encoding="utf-8")
            self.reseal_provenance(args)
            with self.assertRaisesRegex(SystemExit, "prompt stream"):
                verify(args)

    def test_validates_runtime_observations_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
            provenance["checkpoint_load"] = {
                "strict": False,
                "missing_keys_count": 602,
                "unexpected_keys_count": 0,
            }
            provenance["graph_prefix_gate"] = {"mean_sigmoid": 0.2}
            args.provenance.write_text(json.dumps(provenance), encoding="utf-8")
            self.reseal_provenance(args)
            result = verify(args)
            self.assertTrue(result["runtime_observations"]["checkpoint_load"]["verified"])
            self.assertTrue(result["runtime_observations"]["graph_prefix_gate"]["verified"])

            provenance["graph_prefix_gate"]["mean_sigmoid"] = 0.9
            args.provenance.write_text(json.dumps(provenance), encoding="utf-8")
            self.reseal_provenance(args)
            with self.assertRaisesRegex(SystemExit, "prefix gate out of range"):
                verify(args)


if __name__ == "__main__":
    unittest.main()
