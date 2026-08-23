from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation import direct_compact_qwen_inference as inference
from scripts.evaluation import score_direct_compact_passk as scoring
from scripts.evaluation import seal_post_qwen_evaluation_suite as suite
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
)


class DurableJournalTests(unittest.TestCase):
    def test_hash_chain_and_external_head_detect_tail_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "events.jsonl"
            append_event(journal, {"event": "one"})
            append_event(journal, {"event": "two"})
            self.assertEqual(len(load_journal(journal)), 2)
            record = journal_record(journal)
            self.assertEqual(record["event_count"], 2)
            lines = journal.read_text(encoding="utf-8").splitlines()
            journal.write_text(lines[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "chain head"):
                load_journal(journal)

    def test_exact_publish_never_replaces_different_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            require_exact_or_write(output, {"value": 1})
            require_exact_or_write(output, {"value": 1})
            with self.assertRaisesRegex(ValueError, "differs"):
                require_exact_or_write(output, {"value": 2})

    def test_suite_revalidates_terminal_pairing_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "generation.jsonl"
            append_event(journal, {"event": "inference_header"})
            started = append_event(
                journal, {"event": "inference_batch_started"}
            )
            append_event(
                journal,
                {
                    "event": "inference_batch_terminal",
                    "started_event_sha256": started[
                        "journal_event_sha256"
                    ],
                },
            )
            append_event(
                journal,
                {
                    "event": "inference_complete",
                    "resampled_slots": 0,
                    "orphan_retry_events": 0,
                    "orphan_recomputed_slots": 0,
                },
            )
            record = journal_record(journal)
            self.assertEqual(
                suite.completed_evaluation_journal(
                    record, kind="generation"
                ),
                record,
            )


class InferenceResumeTests(unittest.TestCase):
    def _contract(self):
        return {"schema": inference.INFERENCE_JOURNAL_SCHEMA, "seed": 44}

    def _rows(self):
        return [
            {"identity": "task-a"},
            {"identity": "task-b"},
        ]

    def _events(self):
        contract = self._contract()
        task_ids = ["task-a", "task-b"]
        started = {
            "event": "inference_batch_started",
            "schema": inference.INFERENCE_JOURNAL_SCHEMA,
            "batch_index": 0,
            "task_ids": task_ids,
            "batch_seed": inference._batch_seed(
                base_seed=44, batch_index=0, task_ids=task_ids
            ),
            "slot_ids": [
                "task-a:0",
                "task-a:1",
                "task-b:0",
                "task-b:1",
            ],
            "journal_event_sha256": "a" * 64,
        }
        predictions = [
            {"id": "task-a", "predictions": ["a0", "a1"]},
            {"id": "task-b", "predictions": ["b0", "b1"]},
        ]
        return [
            {
                "event": "inference_header",
                "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
            started,
            {
                "event": "inference_batch_terminal",
                "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                "batch_index": 0,
                "started_event_sha256": "a" * 64,
                "retry_count": 0,
                "latest_retry_event_sha256": None,
                "predictions": predictions,
                "predictions_canonical_sha256": canonical_sha256(
                    predictions
                ),
            },
            {
                "event": "inference_complete",
                "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                "outputs_canonical_sha256": canonical_sha256(predictions),
                "rows": 2,
                "slots": 4,
                "resampled_slots": 0,
                "orphan_retry_events": 0,
                "orphan_recomputed_slots": 0,
            },
        ]

    def test_complete_batches_are_exactly_reusable(self):
        terminals, complete, orphan = inference._inference_journal_state(
            self._events(),
            contract_payload=self._contract(),
            rows=self._rows(),
            batch_size=2,
            num_samples=2,
        )
        self.assertTrue(complete)
        self.assertIsNone(orphan)
        self.assertEqual(len(terminals[0]), 2)

    def test_started_without_terminal_exposes_exact_sealed_orphan(self):
        terminals, complete, orphan = inference._inference_journal_state(
            self._events()[:2],
            contract_payload=self._contract(),
            rows=self._rows(),
            batch_size=2,
            num_samples=2,
        )
        self.assertEqual(terminals, {})
        self.assertFalse(complete)
        self.assertEqual(orphan["batch_index"], 0)
        self.assertEqual(
            orphan["batch_seal"]["slot_ids"],
            ["task-a:0", "task-a:1", "task-b:0", "task-b:1"],
        )

    def test_orphan_retry_receipt_completes_without_discarding_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "inference.jsonl"
            contract = self._contract()
            append_event(
                journal,
                {
                    "event": "inference_header",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "contract": contract,
                    "contract_sha256": canonical_sha256(contract),
                },
            )
            task_ids = ["task-a", "task-b"]
            started = append_event(
                journal,
                {
                    "event": "inference_batch_started",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "batch_index": 0,
                    "task_ids": task_ids,
                    "batch_seed": inference._batch_seed(
                        base_seed=44,
                        batch_index=0,
                        task_ids=task_ids,
                    ),
                    "slot_ids": [
                        "task-a:0",
                        "task-a:1",
                        "task-b:0",
                        "task-b:1",
                    ],
                },
            )
            terminals, complete, orphan = inference._inference_journal_state(
                load_journal(journal),
                contract_payload=contract,
                rows=self._rows(),
                batch_size=2,
                num_samples=2,
            )
            self.assertEqual(terminals, {})
            self.assertFalse(complete)
            retry = append_event(
                journal,
                inference.make_inference_orphan_retry_event(orphan),
            )
            predictions = [
                {"id": "task-a", "predictions": ["a0", "a1"]},
                {"id": "task-b", "predictions": ["b0", "b1"]},
            ]
            append_event(
                journal,
                {
                    "event": "inference_batch_terminal",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "batch_index": 0,
                    "started_event_sha256": started[
                        "journal_event_sha256"
                    ],
                    "retry_count": 1,
                    "latest_retry_event_sha256": retry[
                        "journal_event_sha256"
                    ],
                    "predictions": predictions,
                    "predictions_canonical_sha256": canonical_sha256(
                        predictions
                    ),
                },
            )
            append_event(
                journal,
                {
                    "event": "inference_complete",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "outputs_canonical_sha256": canonical_sha256(
                        predictions
                    ),
                    "rows": 2,
                    "slots": 4,
                    "resampled_slots": 0,
                    "orphan_retry_events": 1,
                    "orphan_recomputed_slots": 4,
                },
            )
            terminals, complete, orphan = inference._inference_journal_state(
                load_journal(journal),
                contract_payload=contract,
                rows=self._rows(),
                batch_size=2,
                num_samples=2,
            )
            self.assertTrue(complete)
            self.assertIsNone(orphan)
            self.assertEqual(terminals[0], predictions)

    def test_duplicate_or_tampered_inference_retry_is_rejected(self):
        contract = self._contract()
        started = self._events()[1]
        _, _, orphan = inference._inference_journal_state(
            self._events()[:2],
            contract_payload=contract,
            rows=self._rows(),
            batch_size=2,
            num_samples=2,
        )
        retry = inference.make_inference_orphan_retry_event(orphan)
        retry["journal_event_sha256"] = "d" * 64
        duplicate = dict(retry)
        duplicate["journal_event_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "retry receipt"):
            inference._inference_journal_state(
                [self._events()[0], started, retry, duplicate],
                contract_payload=contract,
                rows=self._rows(),
                batch_size=2,
                num_samples=2,
            )
        tampered = dict(retry)
        tampered["sealed_batch_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "retry receipt"):
            inference._inference_journal_state(
                [self._events()[0], started, tampered],
                contract_payload=contract,
                rows=self._rows(),
                batch_size=2,
                num_samples=2,
            )


class ScoreResumeTests(unittest.TestCase):
    def _jobs(self):
        return [
            {
                "task_id": "task-a",
                "sample_index": index,
                "raw_sha256": str(index) * 64,
                "code_sha256": str(index + 1) * 64,
                "code": f"int f()=>{index};",
            }
            for index in range(2)
        ]

    def _contract(self):
        return {"schema": scoring.SCORE_JOURNAL_SCHEMA}

    def test_cot_reasoning_prefix_is_removed_before_scoring(self):
        code = "int fn0() => 7;"
        self.assertEqual(
            scoring.extract_scored_code(
                "<think>\ninspect CFG first\n</think>\n" + code
            ),
            code,
        )
        self.assertEqual(
            scoring.extract_scored_code(
                "Inspect the CFG and constants first.\n" + code
            ),
            code,
        )
        self.assertEqual(
            scoring.extract_scored_code("<think>\nunfinished reasoning"),
            "",
        )

    def test_started_score_batch_exposes_exact_sealed_orphan(self):
        jobs = self._jobs()
        projection = [
            {key: value for key, value in job.items() if key != "code"}
            for job in jobs
        ]
        events = [
            {
                "event": "score_header",
                "schema": scoring.SCORE_JOURNAL_SCHEMA,
                "contract": self._contract(),
                "contract_sha256": canonical_sha256(self._contract()),
            },
            {
                "event": "score_batch_started",
                "schema": scoring.SCORE_JOURNAL_SCHEMA,
                "batch_index": 0,
                "slot_ids": ["task-a:0", "task-a:1"],
                "jobs_canonical_sha256": canonical_sha256(projection),
                "journal_event_sha256": "b" * 64,
            },
        ]
        terminals, complete, orphan = scoring._score_journal_state(
            events,
            contract_payload=self._contract(),
            jobs=jobs,
            batch_size=2,
        )
        self.assertEqual(terminals, {})
        self.assertFalse(complete)
        self.assertEqual(orphan["batch_index"], 0)

    def test_terminal_score_batch_is_exactly_reusable(self):
        jobs = self._jobs()
        projection = [
            {key: value for key, value in job.items() if key != "code"}
            for job in jobs
        ]
        results = [
            {
                **projection[index],
                "compiled": bool(index),
                "passed": False,
                "diagnostic": "test",
            }
            for index in range(2)
        ]
        events = [
            {
                "event": "score_header",
                "schema": scoring.SCORE_JOURNAL_SCHEMA,
                "contract": self._contract(),
                "contract_sha256": canonical_sha256(self._contract()),
            },
            {
                "event": "score_batch_started",
                "schema": scoring.SCORE_JOURNAL_SCHEMA,
                "batch_index": 0,
                "slot_ids": ["task-a:0", "task-a:1"],
                "jobs_canonical_sha256": canonical_sha256(projection),
                "journal_event_sha256": "c" * 64,
            },
            {
                "event": "score_batch_terminal",
                "schema": scoring.SCORE_JOURNAL_SCHEMA,
                "batch_index": 0,
                "started_event_sha256": "c" * 64,
                "retry_count": 0,
                "latest_retry_event_sha256": None,
                "candidate_results": results,
                "candidate_results_canonical_sha256": canonical_sha256(
                    results
                ),
            },
            {
                "event": "score_complete",
                "schema": scoring.SCORE_JOURNAL_SCHEMA,
                "slots": 2,
                "candidate_results_canonical_sha256": canonical_sha256(
                    results
                ),
                "rerun_slots": 0,
                "orphan_retry_events": 0,
                "orphan_rerun_slots": 0,
            },
        ]
        terminals, complete, orphan = scoring._score_journal_state(
            events,
            contract_payload=self._contract(),
            jobs=jobs,
            batch_size=2,
        )
        self.assertTrue(complete)
        self.assertIsNone(orphan)
        self.assertEqual(terminals[0], results)

    def test_score_orphan_retry_receipt_completes_same_slots(self):
        jobs = self._jobs()
        projection = [
            {key: value for key, value in job.items() if key != "code"}
            for job in jobs
        ]
        results = [
            {
                **projection[index],
                "compiled": bool(index),
                "passed": False,
                "diagnostic": "test",
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "score.jsonl"
            contract = self._contract()
            append_event(
                journal,
                {
                    "event": "score_header",
                    "schema": scoring.SCORE_JOURNAL_SCHEMA,
                    "contract": contract,
                    "contract_sha256": canonical_sha256(contract),
                },
            )
            started = append_event(
                journal,
                {
                    "event": "score_batch_started",
                    "schema": scoring.SCORE_JOURNAL_SCHEMA,
                    "batch_index": 0,
                    "slot_ids": ["task-a:0", "task-a:1"],
                    "jobs_canonical_sha256": canonical_sha256(projection),
                },
            )
            _, _, orphan = scoring._score_journal_state(
                load_journal(journal),
                contract_payload=contract,
                jobs=jobs,
                batch_size=2,
            )
            retry = append_event(
                journal,
                scoring.make_score_orphan_retry_event(orphan),
            )
            append_event(
                journal,
                {
                    "event": "score_batch_terminal",
                    "schema": scoring.SCORE_JOURNAL_SCHEMA,
                    "batch_index": 0,
                    "started_event_sha256": started[
                        "journal_event_sha256"
                    ],
                    "retry_count": 1,
                    "latest_retry_event_sha256": retry[
                        "journal_event_sha256"
                    ],
                    "candidate_results": results,
                    "candidate_results_canonical_sha256": canonical_sha256(
                        results
                    ),
                },
            )
            append_event(
                journal,
                {
                    "event": "score_complete",
                    "schema": scoring.SCORE_JOURNAL_SCHEMA,
                    "slots": 2,
                    "candidate_results_canonical_sha256": canonical_sha256(
                        results
                    ),
                    "rerun_slots": 0,
                    "orphan_retry_events": 1,
                    "orphan_rerun_slots": 2,
                },
            )
            terminals, complete, orphan = scoring._score_journal_state(
                load_journal(journal),
                contract_payload=contract,
                jobs=jobs,
                batch_size=2,
            )
            self.assertTrue(complete)
            self.assertIsNone(orphan)
            self.assertEqual(terminals[0], results)

    def test_tampered_score_retry_receipt_is_rejected(self):
        jobs = self._jobs()
        projection = [
            {key: value for key, value in job.items() if key != "code"}
            for job in jobs
        ]
        header = {
            "event": "score_header",
            "schema": scoring.SCORE_JOURNAL_SCHEMA,
            "contract": self._contract(),
            "contract_sha256": canonical_sha256(self._contract()),
        }
        started = {
            "event": "score_batch_started",
            "schema": scoring.SCORE_JOURNAL_SCHEMA,
            "batch_index": 0,
            "slot_ids": ["task-a:0", "task-a:1"],
            "jobs_canonical_sha256": canonical_sha256(projection),
            "journal_event_sha256": "f" * 64,
        }
        _, _, orphan = scoring._score_journal_state(
            [header, started],
            contract_payload=self._contract(),
            jobs=jobs,
            batch_size=2,
        )
        retry = scoring.make_score_orphan_retry_event(orphan)
        retry["journal_event_sha256"] = "1" * 64
        retry["sealed_batch"]["slot_ids"] = ["task-a:0"]
        with self.assertRaisesRegex(ValueError, "retry receipt"):
            scoring._score_journal_state(
                [header, started, retry],
                contract_payload=self._contract(),
                jobs=jobs,
                batch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
