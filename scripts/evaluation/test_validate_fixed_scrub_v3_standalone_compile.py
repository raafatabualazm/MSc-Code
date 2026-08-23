from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.project_fixed_scrub_v3_standalone_pool import (
    EXPECTED_ROWS,
    EXPECTED_SAMPLES,
    project_predictions,
    sha256_file,
)
from scripts.evaluation.test_project_fixed_scrub_v3_standalone_pool import prediction_rows
from scripts.evaluation.validate_fixed_scrub_v3_standalone_compile import (
    EXPECTED_K,
    compile_at_k_estimator,
    validate_and_seal,
)


def render_stdout(counts: list[int], overrides: dict[str, float] | None = None) -> str:
    lines = [
        f"[{index}/{EXPECTED_ROWS}] n={EXPECTED_SAMPLES}, compiling={count}, "
        f"compile@1={count / EXPECTED_SAMPLES:.4f}"
        for index, count in enumerate(counts, 1)
    ]
    result: dict[str, float | int] = {
        f"compile_at_{k}": sum(
            compile_at_k_estimator(EXPECTED_SAMPLES, count, k) for count in counts
        )
        / EXPECTED_ROWS
        for k in EXPECTED_K
    }
    result["total_problems"] = EXPECTED_ROWS
    if overrides:
        result.update(overrides)
    lines.append(json.dumps(result, indent=2))
    return "\n".join(lines) + "\n"


class ValidateFixedScrubV3StandaloneCompileTests(unittest.TestCase):
    def fixture(self, root: Path) -> argparse.Namespace:
        source = root / "joined.json"
        pool = root / "pool.json"
        projection = root / "pool.json.provenance.json"
        stdout = root / "standalone_compile_at_k.txt"
        scorer = root / "graph_compile_at_k_antigravity.py"
        projector = Path(__file__).with_name("project_fixed_scrub_v3_standalone_pool.py")
        dart = root / "dart_version.txt"
        output = root / "standalone_compile_at_k.txt.provenance.json"

        source.write_text(json.dumps(prediction_rows()), encoding="utf-8")
        project_predictions(source, pool, projection)
        stdout.write_text(render_stdout([index % 11 for index in range(EXPECTED_ROWS)]), encoding="utf-8")
        scorer.write_text("# pinned legacy scorer fixture\n", encoding="utf-8")
        dart.write_text(
            "Dart SDK version: 3.11.5 (stable) (fixture) on linux_x64\n",
            encoding="utf-8",
        )
        return argparse.Namespace(
            source_predictions=source,
            candidate_pool=pool,
            projection_provenance=projection,
            compile_stdout=stdout,
            scorer=scorer,
            projector=projector,
            dart_version_file=dart,
            expected_pool_sha256=sha256_file(pool),
            expected_projection_provenance_sha256=sha256_file(projection),
            expected_scorer_sha256=sha256_file(scorer),
            output=output,
        )

    def test_validates_metrics_and_writes_separate_provenance_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            payload = validate_and_seal(args)
            written = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(payload, written)
            self.assertEqual(payload["standalone_compile_mode"], "legacy")
            self.assertNotIn("compile_mode", payload)
            self.assertEqual(payload["k_values"], [1, 5, 10])
            self.assertTrue(
                payload["validated_invariants"]["compile_at_k_recomputed_from_row_counts"]
            )
            self.assertEqual(
                payload["inputs"]["projection_provenance"]["sha256"],
                args.expected_projection_provenance_sha256,
            )

    def test_rejects_tampered_metric_even_when_json_is_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            counts = [index % 11 for index in range(EXPECTED_ROWS)]
            args.compile_stdout.write_text(
                render_stdout(counts, {"compile_at_5": 0.0}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "compile_at_5 does not match"):
                validate_and_seal(args)

    def test_rejects_unpinned_pool_scorer_or_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            args.expected_pool_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "candidate pool pin mismatch"):
                validate_and_seal(args)

            args = self.fixture(Path(temp))
            args.expected_scorer_sha256 = "1" * 64
            with self.assertRaisesRegex(ValueError, "scorer pin mismatch"):
                validate_and_seal(args)

            args = self.fixture(Path(temp))
            args.expected_projection_provenance_sha256 = "2" * 64
            with self.assertRaisesRegex(ValueError, "projection provenance pin mismatch"):
                validate_and_seal(args)

    def test_rejects_wrong_dart_or_hidden_pool_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            args.dart_version_file.write_text(
                "Dart SDK version: 3.12.2 (stable) on linux_x64\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unexpected Dart SDK"):
                validate_and_seal(args)

            args = self.fixture(Path(temp))
            pool = json.loads(args.candidate_pool.read_text(encoding="utf-8"))
            pool[0]["tests"] = "hidden"
            args.candidate_pool.write_text(json.dumps(pool), encoding="utf-8")
            args.expected_pool_sha256 = sha256_file(args.candidate_pool)
            projection_payload = json.loads(args.projection_provenance.read_text(encoding="utf-8"))
            projection_payload["output"]["candidate_pool"] = {
                "path": str(args.candidate_pool.resolve()),
                "sha256": sha256_file(args.candidate_pool),
                "size_bytes": args.candidate_pool.stat().st_size,
            }
            args.projection_provenance.write_text(
                json.dumps(projection_payload), encoding="utf-8"
            )
            args.expected_projection_provenance_sha256 = sha256_file(
                args.projection_provenance
            )
            with self.assertRaisesRegex(ValueError, "expected only"):
                validate_and_seal(args)

    def test_rejects_sidecar_output_aliasing_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            args.output = args.compile_stdout
            with self.assertRaisesRegex(ValueError, "must not overwrite an input"):
                validate_and_seal(args)


if __name__ == "__main__":
    unittest.main()
