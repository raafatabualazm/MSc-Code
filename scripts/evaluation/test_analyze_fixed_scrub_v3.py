from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_fixed_scrub_v3.py")
SPEC = importlib.util.spec_from_file_location("analyze_fixed_scrub_v3", MODULE_PATH)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


CHECKPOINT = "a" * 64
SCORER = "b" * 64


def tests_for(name: str, value: int) -> str:
    return (
        "void main() {\n"
        f"  final implementation = {name};\n"
        f"  expect(implementation({value}), {value + 1});\n"
        "}\n"
    )


def reference_for(name: str) -> str:
    return f"int {name}(int a) {{ return a + 1; }}"


def provenance(seed: int = 42, checkpoint: str = CHECKPOINT) -> dict:
    return {
        "checkpoint": {"sha256": checkpoint},
        "seed": seed,
        "row_count": 2,
        "samples_per_row": 3,
        "scoring": {
            "compile_mode": "jit_tests",
            "scorer_sha256": SCORER,
            "dart_sdk_version": "3.11.5",
        },
        "prompt_schema_version": "fixed-scrub-v3",
        "scoring_tests_visible_to_policy": False,
        "policy_input_verified_public_only": True,
    }


def file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": analyzer.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class FixedScrubV3AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_arm(
        self,
        label: str,
        ids: list[str],
        target: str,
        flags: list[tuple[list[int], list[int]]],
        order: list[int],
        embedded: bool = True,
        provenance_payload: dict | None = None,
    ) -> tuple[Path, Path | None]:
        rows = []
        stats_rows = []
        candidates_by_task = [
            [
                f"int {target}(int a) => a + 1;",
                f"class Box {{ int {target}(int a) => a; }}",
                "void main() {}",
            ],
            [
                f"int {target}() => 0;",
                f"int {target}(int a) {{ return a; }}",
                f"int {target}(int a, int b) => a + b;",
            ],
        ]
        for output_index, task_index in enumerate(order):
            compiled, passed = flags[task_index]
            row = {
                "id": ids[task_index],
                "predictions": candidates_by_task[task_index],
                "reference": reference_for(target),
                "tests": tests_for(target, task_index + 10),
                "evaluation_only_dart_function_signature": f"int {target}(int a)",
            }
            if embedded:
                row["compile_flags"] = compiled
                row["pass_flags"] = passed
            rows.append(row)
            stats_row = {"problem_id": ids[task_index], "language": "dart"}
            for candidate in range(1, 4):
                stats_row[f"cand_{candidate}_compile"] = compiled[candidate - 1]
                stats_row[f"cand_{candidate}_pass"] = passed[candidate - 1]
            stats_rows.append(stats_row)
        prediction_path = self.root / f"{label}.json"
        prediction_path.write_text(json.dumps(rows), encoding="utf-8")
        inference_provenance = provenance_payload or provenance()
        prov_path = Path(str(prediction_path) + ".provenance.json")
        prov_path.write_text(json.dumps(inference_provenance), encoding="utf-8")
        stats_path = self.root / f"{label}.csv"
        fieldnames = ["problem_id", "language"] + [
            field
            for candidate in range(1, 4)
            for field in (f"cand_{candidate}_compile", f"cand_{candidate}_pass")
        ]
        with stats_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_rows)
        arm = {
            "comparator": "comparator",
            "neutral": "neutral_exact",
            "name": "name_only",
        }[label]
        scoring_provenance = {
            "schema_version": analyzer.SCORING_PROVENANCE_SCHEMA,
            "arm": arm,
            "checkpoint": inference_provenance["checkpoint"],
            "generation_seed": inference_provenance["seed"],
            "compile_mode": "jit_tests",
            "scorer_sha256": SCORER,
            "dart_sdk_version": "3.11.5",
            "prompt_schema_version": inference_provenance["prompt_schema_version"],
            "scoring_tests_visible_to_policy": False,
            "policy_input_verified_public_only": arm != "comparator",
            "row_count": 2,
            "samples_per_row": 3,
            "inputs": {
                "predictions": file_record(prediction_path),
                "stats": file_record(stats_path),
            },
        }
        Path(str(stats_path) + ".provenance.json").write_text(
            json.dumps(scoring_provenance), encoding="utf-8"
        )
        return prediction_path, stats_path

    @staticmethod
    def refresh_scoring_bindings(arm: tuple[Path, Path | None]) -> None:
        prediction_path, stats_path = arm
        assert stats_path is not None
        sidecar = Path(str(stats_path) + ".provenance.json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["inputs"]["predictions"] = file_record(prediction_path)
        payload["inputs"]["stats"] = file_record(stats_path)
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

    def args(self, comparator, neutral, name, **overrides):
        values = {
            "comparator": comparator[0],
            "neutral_exact": neutral[0],
            "name_only": name[0],
            "comparator_stats": comparator[1],
            "neutral_exact_stats": neutral[1],
            "name_only_stats": name[1],
            "comparator_provenance": [],
            "neutral_exact_provenance": [],
            "name_only_provenance": [],
            "pair_manifest": None,
            "broken_tasks": None,
            "target_name": "fn0",
            "expected_tasks": 2,
            "expected_candidates": 3,
            "output_json": None,
            "output_markdown": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_pairs_by_normalized_tests_and_computes_metrics(self) -> None:
        comparator_flags = [([1, 1, 0], [1, 0, 0]), ([1, 1, 1], [0, 0, 0])]
        neutral_flags = [([1, 0, 0], [0, 0, 0]), ([1, 1, 1], [0, 1, 0])]
        name_flags = [([0, 0, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", comparator_flags, [0, 1])
        # Both v3 files are deliberately shuffled and use unrelated public IDs.
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", neutral_flags, [1, 0])
        name = self.write_arm("name", ["x0", "x1"], "fn0", name_flags, [1, 0], embedded=False)

        report = analyzer.analyze(self.args(comparator, neutral, name))

        self.assertEqual(report["pairing"]["method"], "normalized_hidden_test_sha256")
        self.assertAlmostEqual(report["arms"]["comparator"]["all_tasks"]["metrics"]["pass_at_1"], 1 / 6)
        self.assertEqual(report["arms"]["name_only"]["all_tasks"]["candidate_compile_count"], 1)
        comparison = report["comparisons_to_comparator"]["neutral_exact"]["all_tasks"]["pass_at_10"]
        self.assertEqual((comparison["solved_task_gains"], comparison["solved_task_losses"]), (1, 1))
        shape = report["arms"]["neutral_exact"]["static_output_shape"]
        self.assertEqual(shape["top_level_target_definitions"], 4)
        self.assertEqual(shape["top_level_target_arity_matches"], 2)

    def test_explicit_manifest_and_broken_denominator(self) -> None:
        flags = [([1, 1, 1], [1, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", flags, [0, 1])
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", flags, [1, 0])
        name = self.write_arm("name", ["x0", "x1"], "fn0", flags, [0, 1])
        manifest = self.root / "pairs.json"
        manifest.write_text(
            json.dumps(
                {
                    "rows": [
                        {"comparator_id": "c0", "neutral_exact_id": "n0", "name_only_id": "x0"},
                        {"comparator_id": "c1", "neutral_exact_id": "n1", "name_only_id": "x1"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        broken = self.root / "broken.json"
        broken.write_text(json.dumps(["c1"]), encoding="utf-8")

        report = analyzer.analyze(
            self.args(comparator, neutral, name, pair_manifest=manifest, broken_tasks=broken)
        )

        self.assertEqual(report["pairing"]["method"], "explicit_manifest")
        self.assertEqual(report["denominators"]["valid_tasks"], 1)
        self.assertEqual(report["denominators"]["excluded_broken_tasks"], 1)
        markdown = analyzer.markdown_report(report)
        self.assertIn("## Metrics (all tasks)", markdown)
        self.assertIn("## Sensitivity metrics (valid tasks only; n=1)", markdown)
        self.assertIn("excludes 1 inherited benchmark contract defects", markdown)
        self.assertIn(
            "## Sensitivity paired outcomes versus comparator (valid tasks only)",
            markdown,
        )
        self.assertIn("### neutral_exact (valid tasks)", markdown)
        self.assertIn("| comparator | 16.6667%", markdown)
        self.assertIn("| comparator | 33.3333%", markdown)

    def test_fails_closed_on_provenance_mismatch(self) -> None:
        flags = [([1, 0, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", flags, [0, 1])
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", flags, [0, 1])
        bad = provenance(checkpoint="c" * 64)
        name = self.write_arm("name", ["x0", "x1"], "fn0", flags, [0, 1], provenance_payload=bad)
        with self.assertRaisesRegex(analyzer.AnalysisError, "checkpoint_sha256"):
            analyzer.analyze(self.args(comparator, neutral, name))

    def test_fails_closed_on_test_mismatch_even_with_manifest(self) -> None:
        flags = [([1, 0, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", flags, [0, 1])
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", flags, [0, 1])
        name = self.write_arm("name", ["x0", "x1"], "fn0", flags, [0, 1])
        name_rows = json.loads(name[0].read_text(encoding="utf-8"))
        name_rows[0]["tests"] = tests_for("fn0", 999)
        name[0].write_text(json.dumps(name_rows), encoding="utf-8")
        self.refresh_scoring_bindings(name)
        manifest = self.root / "pairs.json"
        manifest.write_text(
            json.dumps(
                [
                    {"comparator_id": "c0", "neutral_exact_id": "n0", "name_only_id": "x0"},
                    {"comparator_id": "c1", "neutral_exact_id": "n1", "name_only_id": "x1"},
                ]
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(analyzer.AnalysisError, "equal hidden tests"):
            analyzer.analyze(self.args(comparator, neutral, name, pair_manifest=manifest))

    def test_rejects_wrong_candidate_count_and_pass_without_compile(self) -> None:
        flags = [([1, 0, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", flags, [0, 1])
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", flags, [0, 1])
        invalid_flags = [([0, 0, 0], [1, 0, 0]), flags[1]]
        name = self.write_arm("name", ["x0", "x1"], "fn0", invalid_flags, [0, 1])
        with self.assertRaisesRegex(analyzer.AnalysisError, "pass flag"):
            analyzer.analyze(self.args(comparator, neutral, name))

    def test_rejects_prediction_bytes_not_bound_by_scoring_provenance(self) -> None:
        flags = [([1, 0, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", flags, [0, 1])
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", flags, [0, 1])
        name = self.write_arm("name", ["x0", "x1"], "fn0", flags, [0, 1])
        name[0].write_text(name[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")

        with self.assertRaisesRegex(analyzer.AnalysisError, "predictions SHA-256 mismatch"):
            analyzer.analyze(self.args(comparator, neutral, name))

    def test_rejects_stats_bytes_not_bound_by_scoring_provenance(self) -> None:
        flags = [([1, 0, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", flags, [0, 1])
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", flags, [0, 1])
        name = self.write_arm("name", ["x0", "x1"], "fn0", flags, [0, 1])
        assert name[1] is not None
        name[1].write_text(name[1].read_text(encoding="utf-8") + "\n", encoding="utf-8")

        with self.assertRaisesRegex(analyzer.AnalysisError, "stats SHA-256 mismatch"):
            analyzer.analyze(self.args(comparator, neutral, name))

    def test_rejects_malformed_scoring_file_record(self) -> None:
        flags = [([1, 0, 0], [0, 0, 0]), ([1, 0, 0], [0, 0, 0])]
        comparator = self.write_arm("comparator", ["c0", "c1"], "semantic", flags, [0, 1])
        neutral = self.write_arm("neutral", ["n0", "n1"], "fn0", flags, [0, 1])
        name = self.write_arm("name", ["x0", "x1"], "fn0", flags, [0, 1])
        assert name[1] is not None
        sidecar = Path(str(name[1]) + ".provenance.json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["inputs"]["stats"]["sha256"] = "not-a-sha256"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(analyzer.AnalysisError, "stats SHA-256 is malformed"):
            analyzer.analyze(self.args(comparator, neutral, name))

    def test_static_shape_ignores_strings_comments_and_methods(self) -> None:
        candidate = """
        // int fn0(int a) => a;
        const example = 'int fn0(int a) => a;';
        class C { int fn0(int a) => a; }
        int fn0(int a, int b) => a + b;
        """
        self.assertEqual(analyzer.top_level_target_arities(candidate, "fn0"), [2])
        self.assertEqual(
            analyzer.top_level_target_arities("int fn0(int a, {int b = 1, List<int> c = const [1, 2],}) => a;", "fn0"),
            [3],
        )
        self.assertEqual(
            analyzer.parameter_arity("[1, 2], {'a': 1, 'b': 2}", allow_optional_groups=False),
            2,
        )


if __name__ == "__main__":
    unittest.main()
