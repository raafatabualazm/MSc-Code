"""Focused tests for deterministic repair-test splitting."""

from __future__ import annotations

import unittest

from scripts.evaluation.build_repair_test_split import (
    build_split_rows,
    candidate_case_spans,
    exclude_reference_failures,
    split_harness,
)
from scripts.evaluation.repair_loop_antigravity import (
    _candidate_calls,
    validate_visible_test_boundary,
)


HARNESS = """void main() {
  final candidate = square;
  expect(candidate(0), 0);
  expect(candidate(1), 1);
  expect(candidate(2), 4);
  expect(candidate(3), 9);
}

void expect(dynamic actual, dynamic expected) {
  if (actual != expected) throw '$actual != $expected';
}
"""


class HarnessSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_preserves_helpers(self) -> None:
        first = split_harness(HARNESS, "task-square", seed=42)
        second = split_harness(HARNESS, "task-square", seed=42)
        self.assertEqual(first, second)

        visible, hidden, metadata = first
        self.assertEqual(metadata["visible_cases"], 2)
        self.assertEqual(metadata["hidden_cases"], 2)
        self.assertEqual(len(_candidate_calls(visible)), 2)
        self.assertEqual(len(_candidate_calls(hidden)), 2)
        self.assertIn("void expect(dynamic actual", visible)
        self.assertIn("void expect(dynamic actual", hidden)
        validate_visible_test_boundary(visible, hidden, "task-square")

    def test_multiline_nested_candidate_call_is_one_case(self) -> None:
        harness = """void main() {
  final candidate = combine;
  expect(
    candidate([1, 2], {'x': '(value)'}),
    3,
  );
  expect(candidate([], {}), 0);
}
"""
        self.assertEqual(len(candidate_case_spans(harness)), 2)
        visible, hidden, _ = split_harness(harness, "nested", seed=7)
        validate_visible_test_boundary(visible, hidden, "nested")

    def test_duplicate_candidate_inputs_stay_on_the_same_side(self) -> None:
        harness = """void main() {
  final candidate = square;
  expect(candidate(2), 4);
  expect(candidate(3), 9);
  expect(candidate(2), 2 + 2);
}
"""
        visible, hidden, metadata = split_harness(harness, "duplicates", seed=42)
        self.assertEqual(metadata["total_cases"], 3)
        self.assertEqual(metadata["unique_input_groups"], 2)
        self.assertFalse(_candidate_calls(visible) & _candidate_calls(hidden))
        validate_visible_test_boundary(visible, hidden, "duplicates")

    def test_minimum_input_group_constraints_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 5"):
            split_harness(HARNESS, "task-square", 42, min_visible=2, min_hidden=3)

        five_cases = HARNESS.replace(
            "  expect(candidate(3), 9);",
            "  expect(candidate(3), 9);\n  expect(candidate(4), 16);",
        )
        visible, hidden, metadata = split_harness(
            five_cases,
            "task-square-five",
            42,
            min_visible=2,
            min_hidden=3,
        )
        self.assertEqual(metadata["visible_input_groups"], 2)
        self.assertEqual(metadata["hidden_input_groups"], 3)
        self.assertFalse(_candidate_calls(visible) & _candidate_calls(hidden))

    def test_unsplittable_rows_fail_closed_or_are_explicitly_dropped(self) -> None:
        single = """void main() {
  final candidate = square;
  expect(candidate(2), 4);
}
"""
        row = {
            "task_id": "single",
            "filename": "single.dart",
            "function": "square",
            "tests": single,
        }
        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_split_rows([row], seed=42, drop_unsplittable=False)

        visible, hidden, tasks, dropped = build_split_rows(
            [row],
            seed=42,
            drop_unsplittable=True,
        )
        self.assertEqual((visible, hidden, tasks), ([], [], []))
        self.assertEqual(dropped[0]["task_id"], "single")

    def test_reference_failures_are_explicitly_excluded(self) -> None:
        source_rows = [
            {"task_id": "good", "tests": HARNESS},
            {"task_id": "bad", "tests": HARNESS},
        ]
        visible_rows = [
            {"task_id": "good", "visible_tests": HARNESS},
            {"task_id": "bad", "visible_tests": HARNESS},
        ]
        hidden_rows = [
            {"task_id": "good", "tests": HARNESS},
            {"task_id": "bad", "tests": HARNESS},
        ]
        task_records = [{"task_id": "good"}, {"task_id": "bad"}]
        dropped: list[dict] = []
        validation = {
            "preexisting_reference_failures": [
                {"task_id": "bad", "original_diagnostic": "broken label"}
            ]
        }
        visible, hidden, tasks = exclude_reference_failures(
            source_rows,
            visible_rows,
            hidden_rows,
            task_records,
            dropped,
            validation,
        )
        self.assertEqual([row["task_id"] for row in visible], ["good"])
        self.assertEqual([row["task_id"] for row in hidden], ["good"])
        self.assertEqual([row["task_id"] for row in tasks], ["good"])
        self.assertEqual(dropped[0]["task_id"], "bad")
        self.assertEqual(validation["excluded_reference_failures"], 1)


if __name__ == "__main__":
    unittest.main()
