from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WriteFixedScrubV3ScoringProvenanceTests(unittest.TestCase):
    def test_writes_analyzer_required_fields_for_public_only_arm(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = root / "scripts/evaluation/write_fixed_scrub_v3_scoring_provenance.py"
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            checkpoint = work / "checkpoint.bin"
            predictions = work / "predictions.json"
            stats = work / "stats.csv"
            public = work / "public.jsonl"
            scorer = work / "scorer.py"
            dart = work / "dart.txt"
            inference = work / "raw.provenance.json"
            join = work / "join.provenance.json"
            output = work / "stats.csv.provenance.json"
            checkpoint.write_bytes(b"weights")
            predictions.write_text("[]", encoding="utf-8")
            stats.write_text("problem_id\n", encoding="utf-8")
            public.write_text("{}\n", encoding="utf-8")
            scorer.write_text("# scorer\n", encoding="utf-8")
            dart.write_text(
                "Dart SDK version: 3.11.5 (stable) (test) on linux_x64\n",
                encoding="utf-8",
            )

            import hashlib

            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            public_hash = hashlib.sha256(public.read_bytes()).hexdigest()
            inference.write_text(
                json.dumps(
                    {
                        "checkpoint": {"sha256": checkpoint_hash},
                        "dataset": {"sha256": public_hash},
                        "seed": 42,
                        "prompt_schema_version": "antigravity-v3-matched-function-contract",
                        "scoring_tests_visible_to_policy": False,
                    }
                ),
                encoding="utf-8",
            )
            join.write_text(
                json.dumps({"policy_input_verified_public_only": True}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--arm",
                    "name_only",
                    "--predictions",
                    str(predictions),
                    "--stats",
                    str(stats),
                    "--checkpoint",
                    str(checkpoint),
                    "--inference_provenance",
                    str(inference),
                    "--join_provenance",
                    str(join),
                    "--public_dataset",
                    str(public),
                    "--scorer",
                    str(scorer),
                    "--dart_version_file",
                    str(dart),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["compile_mode"], "jit_tests")
            self.assertTrue(payload["policy_input_verified_public_only"])
            self.assertEqual(payload["generation_seed"], 42)
            self.assertEqual(payload["checkpoint"]["sha256"], checkpoint_hash)


if __name__ == "__main__":
    unittest.main()
