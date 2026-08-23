from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.project_fixed_scrub_v3_standalone_pool import (
    EXPECTED_ROWS,
    EXPECTED_SAMPLES,
    candidate_stream_sha256,
    load_candidate_pool,
    project_predictions,
    sha256_file,
)


def prediction_rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"semanticTask{index}",
            "source_line": index + 1,
            "filename": f"semanticTask{index}.dart",
            "predictions": [
                f"```dart\nint candidate_{index}_{sample}() => {sample}; // café\n```"
                for sample in range(EXPECTED_SAMPLES)
            ],
            "reference": f"int secretSemanticName{index}() => 1;",
            "tests": f"void main() {{ secretSemanticName{index}(); }}",
            "dart_function_signature": f"int secretSemanticName{index}()",
            "evaluation_only_dart_function_signature": f"int secretSemanticName{index}()",
            "source": "withheld source",
            "language": "dart",
            "graph_input_ablation": {"target_id": f"semanticTask{index}"},
        }
        for index in range(EXPECTED_ROWS)
    ]


class ProjectFixedScrubV3StandalonePoolTests(unittest.TestCase):
    def test_projects_actual_joined_schema_without_changing_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            source = work / "joined.json"
            output = work / "candidate_pool.json"
            provenance = work / "candidate_pool.json.provenance.json"
            rows = prediction_rows()
            source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

            payload = project_predictions(source, output, provenance)
            pool = load_candidate_pool(output)
            expected_candidates = [row["predictions"] for row in rows]

            self.assertEqual([row["predictions"] for row in pool], expected_candidates)
            self.assertTrue(all(set(row) == {"id", "predictions"} for row in pool))
            self.assertEqual(pool[0]["id"], "standalone_row_0001")
            self.assertEqual(pool[-1]["id"], "standalone_row_0154")
            self.assertEqual(
                payload["candidate_stream_sha256"],
                candidate_stream_sha256(expected_candidates),
            )
            self.assertEqual(payload["output"]["candidate_pool"]["sha256"], sha256_file(output))
            self.assertFalse(payload["semantic_identifiers_preserved"])
            rendered_pool = output.read_text(encoding="utf-8")
            self.assertNotIn("secretSemanticName", rendered_pool)
            self.assertNotIn('"tests"', rendered_pool)
            self.assertNotIn('"reference"', rendered_pool)

    def test_rejects_wrong_candidate_count_and_non_string_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            source = work / "bad.json"
            output = work / "pool.json"
            provenance = work / "pool.provenance.json"
            rows = prediction_rows()
            rows[3]["predictions"] = ["one"]
            source.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 10 candidates"):
                project_predictions(source, output, provenance)

            rows = prediction_rows()
            rows[3]["predictions"][2] = {"not": "a string"}  # type: ignore[index]
            source.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every candidate must be a string"):
                project_predictions(source, output, provenance)

    def test_rejects_output_aliasing_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "predictions.json"
            source.write_text(json.dumps(prediction_rows()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                project_predictions(source, source, Path(temp) / "sidecar.json")


if __name__ == "__main__":
    unittest.main()
