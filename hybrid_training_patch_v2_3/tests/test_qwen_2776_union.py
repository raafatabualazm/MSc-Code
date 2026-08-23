from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.training.build_qwen_quality_gate import build as build_quality_gate
from scripts.training.prepare_qwen_2776_supplement import (
    _shared_compatibility,
    derive_partition,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (
    ArtifactError,
    sha256_text,
    stable_sha256,
)
from scripts.training.union_qwen_2776_training_artifacts import (
    SEQUENCE_SCHEDULE_SCHEMA,
    _deterministic_union_order,
    validate_exact_grid,
)


class _NoopContract:
    def validate_row(self, row, identity):
        del row, identity


def _compact(task_id: str, token: int, target: str = "int fn0() => 0;"):
    return {
        "task_id": task_id,
        "compact_input_ids": [token],
        "dart_source": target,
    }


def _prompt(task_id: str, compact_sha: str, text: str):
    return {
        "task_id": task_id,
        "text": text,
        "text_sha256": sha256_text(text),
        "compact_ids_sha256": compact_sha,
        "representation_schema": "lossless-semantic-f2",
        "system_prompt_sha256": "a" * 64,
    }


class Qwen2776UnionTests(unittest.TestCase):
    def test_partition_is_exact_and_order_preserving(self):
        legacy = [f"legacy-{index:04d}" for index in range(1580)]
        supplement = [f"supplement-{index:04d}" for index in range(1196)]
        heldout = [f"heldout-{index:03d}" for index in range(175)]
        candidate = legacy[:10] + heldout + supplement + legacy[10:]
        fit, observed_supplement = derive_partition(
            candidate_order=candidate,
            legacy_order=legacy,
            heldout_order=heldout,
        )
        self.assertEqual(len(fit), 2776)
        self.assertEqual(set(fit), set(legacy) | set(supplement))
        self.assertEqual(observed_supplement, supplement)
        self.assertFalse(set(fit) & set(heldout))

        filtered_fit, filtered_supplement = derive_partition(
            candidate_order=fit,
            legacy_order=legacy,
            heldout_order=heldout,
        )
        self.assertEqual(filtered_fit, fit)
        self.assertEqual(filtered_supplement, supplement)

    def test_partition_rejects_heldout_leak_and_overlap(self):
        legacy = [f"legacy-{index:04d}" for index in range(1580)]
        supplement = [f"supplement-{index:04d}" for index in range(1196)]
        heldout = [f"heldout-{index:03d}" for index in range(175)]
        leaked = legacy + supplement
        leaked[-1] = heldout[0]
        with self.assertRaisesRegex(ArtifactError, "heldout"):
            derive_partition(
                candidate_order=leaked,
                legacy_order=legacy,
                heldout_order=heldout,
            )

    def test_shared_parent_requires_identical_compact_and_prompt(self):
        old_compact = {"a": _compact("a", 7)}
        fit_compact = {"a": _compact("a", 7, "int fn0() => 1;")}
        compact_sha = stable_sha256([7])
        old_prompt = {"a": _prompt("a", compact_sha, "same prompt")}
        fit_prompt = {"a": _prompt("a", compact_sha, "same prompt")}
        result = _shared_compatibility(
            legacy_ids=["a"],
            legacy_compact=old_compact,
            fit_compact=fit_compact,
            legacy_prompts=old_prompt,
            fit_prompts=fit_prompt,
        )
        self.assertTrue(result["all_compact_ids_byte_identical"])
        self.assertTrue(result["all_api_prompt_text_byte_identical"])
        self.assertEqual(result["gold_target_difference_count"], 1)

        fit_prompt["a"] = _prompt("a", compact_sha, "changed prompt")
        with self.assertRaisesRegex(ArtifactError, "F2 prompt differs"):
            _shared_compatibility(
                legacy_ids=["a"],
                legacy_compact=old_compact,
                fit_compact=fit_compact,
                legacy_prompts=old_prompt,
                fit_prompts=fit_prompt,
            )

    def test_exact_sequence_grid_rejects_missing_or_duplicate_slot(self):
        tasks = {"a", "b"}
        fit = {
            "a": (0, _compact("a", 10)),
            "b": (1, _compact("b", 11)),
        }
        rows = []
        schedule = []
        for task_id in sorted(tasks):
            compact_sha = stable_sha256(fit[task_id][1]["compact_input_ids"])
            for sample in range(8):
                target = f"int fn0() => {sample};"
                rows.append(_compact(task_id, fit[task_id][1]["compact_input_ids"][0], target))
                schedule.append(
                    {
                        "schema": SEQUENCE_SCHEDULE_SCHEMA,
                        "position": len(schedule),
                        "kind": "teacher_draw",
                        "task_id": task_id,
                        "sample_index": sample,
                        "candidate_id": f"{task_id}-{sample}",
                        "compact_ids_sha256": compact_sha,
                        "target_sha256": sha256_text(target),
                    }
                )
        paired = validate_exact_grid(
            label="parent",
            rows=rows,
            schedule=schedule,
            expected_task_ids=tasks,
            samples_per_task=8,
            schedule_schema=SEQUENCE_SCHEDULE_SCHEMA,
            contract=_NoopContract(),
            fit_by_task=fit,
            cot=False,
        )
        self.assertEqual(len(paired), 16)
        with self.assertRaisesRegex(ArtifactError, "exact K=8"):
            validate_exact_grid(
                label="parent",
                rows=rows[:-1],
                schedule=schedule[:-1],
                expected_task_ids=tasks,
                samples_per_task=8,
                schedule_schema=SEQUENCE_SCHEDULE_SCHEMA,
                contract=_NoopContract(),
                fit_by_task=fit,
                cot=False,
            )

    def test_union_order_is_independent_of_teacher_candidate_payload(self):
        original = [
            {
                "task_id": f"task-{task}",
                "sample_index": sample,
                "candidate_id": f"candidate-before-{task}-{sample}",
            }
            for task in range(5)
            for sample in range(8)
        ]
        mutated = [
            {
                **item,
                "candidate_id": "payload-dependent-after-"
                + item["candidate_id"],
            }
            for item in reversed(original)
        ]
        expected = [
            (item["task_id"], item["sample_index"])
            for item in _deterministic_union_order(
                original, seed=44, objective="sequence"
            )
        ]
        observed = [
            (item["task_id"], item["sample_index"])
            for item in _deterministic_union_order(
                mutated, seed=44, objective="sequence"
            )
        ]
        self.assertEqual(observed, expected)

    def test_quality_gate_is_built_only_from_exact_real_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            verified = root / "verified.jsonl"
            output = root / "gate.json"
            unique = {f"task-{index}": 8 for index in range(16)}
            audit.write_text(
                json.dumps(
                    {
                        "coverage": {
                            "candidates": 128,
                            "parseable_candidates": 100,
                        },
                        "sampling": {
                            "unique_final_sequences_per_task": unique,
                            "pathological_all_tasks_have_identical_k8_draws": False,
                        },
                        "target_length_gate": {
                            "passed": True,
                            "targets_checked": 128,
                            "overflow_count": 0,
                            "non_code_target_count": 100,
                            "evidence_sha256": "b" * 64,
                            "target_contract": {
                                "trainer_contract": {"sha256": "c" * 64},
                                "max_target_tokens": 24576,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            verified.write_text(
                "\n".join(
                    json.dumps({"task_id": f"task-{index}"})
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
            gate = build_quality_gate(
                argparse.Namespace(
                    pilot_audit=audit,
                    pilot_verified_only=verified,
                    output=output,
                    pilot_tasks=16,
                    minimum_verified_tasks=1,
                    minimum_parseable_fraction=0.5,
                )
            )
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["candidates"], 128)
            self.assertEqual(gate["sampling_diversity"][
                "tasks_with_all_k8_draws_identical"
            ], 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["schema"],
                "qwen-teacher-quality-gate-v1",
            )
            verified.write_text("", encoding="utf-8")
            diagnostic_gate = build_quality_gate(
                argparse.Namespace(
                    pilot_audit=audit,
                    pilot_verified_only=verified,
                    output=output,
                    pilot_tasks=16,
                    minimum_verified_tasks=0,
                    minimum_parseable_fraction=0.5,
                )
            )
            self.assertTrue(diagnostic_gate["passed"])
            self.assertEqual(diagnostic_gate["verified_tasks"], 0)
            self.assertTrue(
                diagnostic_gate[
                    "verified_correctness_is_diagnostic_only"
                ]
            )

    def test_launchers_keep_supplement_journal_separate(self):
        workspace = Path(__file__).resolve().parents[2]
        supplemental = (
            workspace
            / "fixed_training_launchers"
            / "run_qwen38_supplemental_harvest.sh"
        ).read_text(encoding="utf-8")
        union = (
            workspace
            / "fixed_training_launchers"
            / "run_qwen38_union_2776.sh"
        ).read_text(encoding="utf-8")
        training = (
            workspace
            / "fixed_training_launchers"
            / "run_qwen38_train_union_2776.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("direct_compact_qwen38_supplement1196", supplemental)
        self.assertIn("supplement_teacher_slots == 9568", supplemental)
        self.assertIn(
            'QWEN_PILOT_MIN_VERIFIED_TASKS:-0',
            supplemental,
        )
        self.assertIn("live_journal_modified=false", supplemental)
        self.assertIn("sequence_rows=22208", union)
        self.assertIn("cot_rows=5552", union)
        self.assertIn(
            'SUPPLEMENT_TRAIN="${SUPPLEMENT_TRAIN:-', training
        )
        self.assertIn(
            'supplemental_gold "${EXPANDED_GOLD}"', training
        )
        self.assertNotIn(
            'train_multifunction_binary.jsonl" 2776 token_mean',
            training,
        )


if __name__ == "__main__":
    unittest.main()
