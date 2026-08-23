from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.verify_fixed_scrub_v3_inference import (
    EXPECTED_GRAPH_ENV,
    sha256,
    verify,
)


class VerifyFixedScrubV3InferenceTests(unittest.TestCase):
    def fixture(self, root: Path) -> argparse.Namespace:
        public_path = root / "public.jsonl"
        raw_path = root / "raw.json"
        provenance_path = root / "raw.json.provenance.json"
        public_path.write_text(json.dumps({"task_id": "opaque_1"}) + "\n", encoding="utf-8")
        raw_path.write_text(
            json.dumps(
                [
                    {
                        "id": "opaque_1",
                        "source_line": 1,
                        "predictions": ["int fn0() => 1;"],
                        "tests": "",
                        "reference": "",
                        "graph_input_ablation": {
                            "mode": "none",
                            "target_id": "opaque_1",
                            "donor_id": "opaque_1",
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
        provenance = {
            "prompt_schema_version": "antigravity-v3-matched-function-contract",
            "scoring_tests_visible_to_policy": False,
            "prompt_stream_sha256": "a" * 64,
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
            "dataset": {"sha256": sha256(public_path)},
            "output": {"sha256": sha256(raw_path)},
            "checkpoint": {"sha256": "c" * 64},
            "checkpoint_load": {
                "strict": False,
                "missing_keys_count": 602,
                "unexpected_keys_count": 0,
            },
            "graph_prefix_gate": {"mean_sigmoid": 0.2},
            "graph_environment": dict(EXPECTED_GRAPH_ENV),
            "graph_input_ablation": {"mode": "none", "self_mapped_rows": 1},
            "models": {
                "decoder": {"requested_revision": "decoder-rev"},
                "encoder": {"requested_revision": "encoder-rev"},
            },
        }
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        return argparse.Namespace(
            public_dataset=public_path,
            raw_predictions=raw_path,
            provenance=provenance_path,
            checkpoint_sha256="c" * 64,
            expected_prompt_stream_sha256="a" * 64,
            prompt_schema="antigravity-v3-matched-function-contract",
            expected_rows=1,
            expected_samples=1,
            seed=42,
            missing_keys=602,
            unexpected_keys=0,
            gate_min=0.15,
            gate_max=0.25,
            decoder_revision="decoder-rev",
            encoder_revision="encoder-rev",
        )

    def test_accepts_complete_matched_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = verify(self.fixture(Path(temp)))
            self.assertTrue(result["verified"])
            self.assertEqual(result["checkpoint_load"]["missing_keys_count"], 602)

    def test_rejects_checkpoint_load_signature_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            payload = json.loads(args.provenance.read_text(encoding="utf-8"))
            payload["checkpoint_load"]["unexpected_keys_count"] = 1
            args.provenance.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "unexpected-key signature"):
                verify(args)


if __name__ == "__main__":
    unittest.main()
