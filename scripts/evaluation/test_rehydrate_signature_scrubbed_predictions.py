from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation.rehydrate_signature_scrubbed_predictions import (
    candidate_digest,
    main,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RehydrateSignatureScrubbedPredictionsTest(unittest.TestCase):
    def fixture(
        self,
        root: Path,
        *,
        signature_mode: str = "name_only",
        target_name: str = "candidate",
    ) -> tuple[Path, Path, Path, Path]:
        public = root / "public.jsonl"
        private = root / "private.jsonl"
        predictions = root / "predictions.json"
        output = root / "scored.json"
        if signature_mode == "neutral_exact":
            signature_fields = {
                "dart_function_signature": f"int {target_name}(int a)",
                "prompt_signature_mode": "exact",
                "public_prompt_signature": f"int {target_name}(int a)",
            }
        else:
            signature_fields = {
                "dart_function_signature": "",
                "prompt_signature_mode": "name_only",
                "public_prompt_signature": "",
            }
        public_row = {
            "task_id": "neutral-1",
            "filename": "neutral-1.dart",
            "function": target_name,
            "camel_case_function_name": target_name,
            **signature_fields,
        }
        private_row = {
            **public_row,
            "dart_source": f"int {target_name}(int x) => x + 1;",
            "tests": f"void main() {{ assert({target_name}(1) == 2); }}",
            "evaluation_only_dart_function_signature": f"int {target_name}(int x)",
        }
        prediction_rows = [{
            "id": "neutral-1",
            "filename": "neutral-1.dart",
            "predictions": [f"int {target_name}(int x) => x + 1;"],
            "reference": "",
            "tests": "",
        }]
        write_jsonl(public, [public_row])
        write_jsonl(private, [private_row])
        predictions.write_text(json.dumps(prediction_rows), encoding="utf-8")
        provenance = {
            "row_count": 1,
            "scoring_tests_visible_to_policy": False,
            "dataset": {
                "sha256": __import__("hashlib").sha256(public.read_bytes()).hexdigest(),
            },
            "generation": {"num_samples": 1},
        }
        Path(str(predictions) + ".provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        return public, private, predictions, output

    def run_main(
        self,
        public: Path,
        private: Path,
        predictions: Path,
        output: Path,
        *,
        signature_mode: str = "name_only",
        target_name: str | None = None,
    ) -> None:
        argv = [
            "rehydrate",
            "--predictions", str(predictions),
            "--public_dataset", str(public),
            "--private_dataset", str(private),
            "--output", str(output),
            "--expected_rows", "1",
            "--expected_samples", "1",
            "--expected_signature_mode", signature_mode,
        ]
        if target_name is not None:
            argv.extend(["--expected_target_name", target_name])
        with patch("sys.argv", argv):
            main()

    def test_joins_labels_without_changing_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(Path(tmp))
            before = candidate_digest(json.loads(predictions.read_text(encoding="utf-8")))
            self.run_main(public, private, predictions, output)
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("candidate", rows[0]["reference"])
            self.assertIn("assert", rows[0]["tests"])
            self.assertEqual(before, candidate_digest(rows))
            provenance_path = Path(str(output) + ".provenance.json")
            self.assertTrue(provenance_path.is_file())
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(
                provenance["schema_version"],
                "signature-scrubbed-post-inference-join-v3",
            )
            self.assertEqual(provenance["expected_target_name"], "candidate")

    def test_rejects_hidden_labels_in_inference_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(Path(tmp))
            rows = json.loads(predictions.read_text(encoding="utf-8"))
            rows[0]["tests"] = "leaked"
            predictions.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already contains hidden labels"):
                self.run_main(public, private, predictions, output)

    def test_rejects_private_field_in_public_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(Path(tmp))
            rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines()]
            rows[0]["dart_source"] = "leaked"
            write_jsonl(public, rows)
            provenance_path = Path(str(predictions) + ".provenance.json")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["dataset"]["sha256"] = __import__("hashlib").sha256(
                public.read_bytes()
            ).hexdigest()
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exposes private fields"):
                self.run_main(public, private, predictions, output)

    def _rewrite_public(self, public: Path, predictions: Path, mutate) -> None:
        rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines()]
        mutate(rows[0])
        write_jsonl(public, rows)
        provenance_path = Path(str(predictions) + ".provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["dataset"]["sha256"] = __import__("hashlib").sha256(
            public.read_bytes()
        ).hexdigest()
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    def test_neutral_exact_mode_joins_and_validates_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(
                Path(tmp), signature_mode="neutral_exact"
            )
            self.run_main(
                public, private, predictions, output, signature_mode="neutral_exact"
            )
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("candidate", rows[0]["reference"])

    def test_fn0_name_only_mode_joins_and_records_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(
                Path(tmp), target_name="fn0"
            )
            before = candidate_digest(json.loads(predictions.read_text(encoding="utf-8")))
            self.run_main(
                public, private, predictions, output, target_name="fn0"
            )
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("fn0", rows[0]["reference"])
            self.assertEqual(before, candidate_digest(rows))
            provenance = json.loads(
                Path(str(output) + ".provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["expected_target_name"], "fn0")

    def test_fn0_neutral_exact_mode_joins_and_validates_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(
                Path(tmp), signature_mode="neutral_exact", target_name="fn0"
            )
            self.run_main(
                public,
                private,
                predictions,
                output,
                signature_mode="neutral_exact",
                target_name="fn0",
            )
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("int fn0(int x)", rows[0]["reference"])

    def test_fn0_rejects_public_target_field_mismatches(self) -> None:
        for field in ("function", "camel_case_function_name"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                public, private, predictions, output = self.fixture(
                    Path(tmp), target_name="fn0"
                )
                self._rewrite_public(
                    public,
                    predictions,
                    lambda row, field=field: row.update({field: "candidate"}),
                )
                with self.assertRaisesRegex(ValueError, "does not match expected target"):
                    self.run_main(
                        public, private, predictions, output, target_name="fn0"
                    )

    def test_fn0_neutral_exact_rejects_candidate_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(
                Path(tmp), signature_mode="neutral_exact", target_name="fn0"
            )
            self._rewrite_public(
                public,
                predictions,
                lambda row: row.update(
                    dart_function_signature="int candidate(int a)",
                    public_prompt_signature="int candidate(int a)",
                ),
            )
            with self.assertRaisesRegex(ValueError, "expected target 'fn0'"):
                self.run_main(
                    public,
                    private,
                    predictions,
                    output,
                    signature_mode="neutral_exact",
                    target_name="fn0",
                )

    def test_fn0_name_only_rejects_candidate_hidden_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(
                Path(tmp), target_name="fn0"
            )
            rows = [
                json.loads(line)
                for line in private.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["evaluation_only_dart_function_signature"] = (
                "int candidate(int x)"
            )
            write_jsonl(private, rows)
            with self.assertRaisesRegex(ValueError, "hidden signature does not target 'fn0'"):
                self.run_main(
                    public, private, predictions, output, target_name="fn0"
                )

    def test_name_only_rejects_public_prompt_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(Path(tmp))
            self._rewrite_public(
                public,
                predictions,
                lambda row: row.update(public_prompt_signature="int candidate(int a)"),
            )
            with self.assertRaisesRegex(ValueError, "exposes an exact signature"):
                self.run_main(public, private, predictions, output)

    def test_neutral_exact_rejects_signature_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(
                Path(tmp), signature_mode="neutral_exact"
            )
            self._rewrite_public(
                public,
                predictions,
                lambda row: row.update(dart_function_signature="int candidate(String a)"),
            )
            with self.assertRaisesRegex(ValueError, "does not match the sealed"):
                self.run_main(
                    public, private, predictions, output, signature_mode="neutral_exact"
                )

    def test_name_only_mode_rejects_neutral_exact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            public, private, predictions, output = self.fixture(
                Path(tmp), signature_mode="neutral_exact"
            )
            with self.assertRaisesRegex(ValueError, "is not name-only"):
                self.run_main(public, private, predictions, output)

    def test_rejects_fingerprint_hashes_in_v2_and_v3_public_rows(self) -> None:
        for schema in ("dart-signature-scrubbed-v2", "dart-signature-scrubbed-v3"):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                public, private, predictions, output = self.fixture(Path(tmp))
                self._rewrite_public(
                    public,
                    predictions,
                    lambda row, schema=schema: row.update(
                        benchmark_protocol={
                            "schema": schema,
                            "semantic_function_name_sha256": "f" * 64,
                        }
                    ),
                )
                with self.assertRaisesRegex(ValueError, "exposes fingerprint hashes"):
                    self.run_main(public, private, predictions, output)


if __name__ == "__main__":
    unittest.main()
