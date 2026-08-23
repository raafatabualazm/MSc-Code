from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAINER_PATH = ROOT / "scripts" / "training" / "direct_compact_verpo.py"
JUDGE_PATH = ROOT / "scripts" / "training" / "verpo_judge_antigravity.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def parse_base_args(verpo, *extra: str):
    return verpo.parse_args(
        [
            "--rollout_file",
            "train.jsonl",
            "--rollout_seal",
            "train.seal.json",
            "--output_dir",
            "out",
            "--contract",
            "contract.json",
            "--codebook",
            "codebook.json",
            "--codec_artifact",
            "codec.py",
            "--tokenizer_json",
            "tokenizer.json",
            "--executable_view_report",
            "executable.build.json",
            "--expected_executable_view_report_sha256",
            "a" * 64,
            "--feedback_view_public_manifest",
            "feedback.public.json",
            "--expected_feedback_view_public_manifest_sha256",
            "c" * 64,
            "--predeclared_chain_contract",
            "chain.json",
            "--expected_predeclared_chain_sha256",
            "b" * 64,
            "--warmstart_checkpoint",
            "warmstart",
            "--judge_escalation_queue",
            "teacher-escalations.jsonl",
            *extra,
        ]
    )


def failed_detail(
    *,
    compiled: bool = True,
    full_pass: bool = False,
    diagnostic: str = "visible test failed",
):
    return {
        "compiled": compiled,
        "full_pass": full_pass,
        "test_passes": [False],
        "diagnostic": diagnostic,
    }


class SparseInlineCliContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verpo = load_module(TRAINER_PATH, "verpo_sparse_cli_test")

    def test_live_defaults_are_sparse_bounded_and_non_pro(self):
        args = parse_base_args(self.verpo)
        self.assertEqual(args.judge_mode, "sparse_inline")
        self.assertEqual(args.judge_interval, 8)
        self.assertEqual(args.judge_group_top_n, 2)
        self.assertEqual(args.judge_deadline_seconds, 60.0)
        self.assertEqual(args.judge_failure_policy, "local_only")
        self.assertEqual(args.judge_reasoning_mode, "standard")
        self.assertEqual(
            Path(args.judge_escalation_queue),
            Path("teacher-escalations.jsonl"),
        )

    def test_inline_pro_mode_is_rejected_before_training(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_base_args(
                self.verpo,
                "--judge_mode",
                "sparse_inline",
                "--judge_reasoning_mode",
                "pro",
            )
        self.assertNotIn("unrecognized arguments", stderr.getvalue())

    def test_pro_is_reserved_for_offline_queue_processing(self):
        args = parse_base_args(
            self.verpo,
            "--judge_mode",
            "offline_queue",
            "--judge_reasoning_mode",
            "pro",
        )
        self.assertEqual(args.judge_mode, "offline_queue")
        self.assertEqual(args.judge_reasoning_mode, "pro")

    def test_failure_policy_cannot_turn_teacher_outage_into_run_failure(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_base_args(
                self.verpo,
                "--judge_failure_policy",
                "abort",
            )
        self.assertNotIn("unrecognized arguments", stderr.getvalue())


class SparseGroupSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verpo = load_module(TRAINER_PATH, "verpo_sparse_group_test")

    def test_group_gate_requires_interval_and_two_compiling_failures(self):
        eligible = [
            failed_detail(),
            failed_detail(diagnostic="different failure"),
            failed_detail(compiled=False),
        ]
        self.assertTrue(
            self.verpo.should_query_group_teacher(
                eligible,
                group_ordinal=8,
                interval=8,
            )
        )
        self.assertFalse(
            self.verpo.should_query_group_teacher(
                eligible,
                group_ordinal=7,
                interval=8,
            )
        )
        self.assertFalse(
            self.verpo.should_query_group_teacher(
                [failed_detail(), failed_detail(compiled=False)],
                group_ordinal=8,
                interval=8,
            )
        )

    def test_any_verified_full_pass_suppresses_the_teacher(self):
        details = [
            failed_detail(),
            failed_detail(),
            {
                "compiled": True,
                "full_pass": True,
                "test_passes": [True],
                "diagnostic": "",
            },
        ]
        self.assertFalse(
            self.verpo.should_query_group_teacher(
                details,
                group_ordinal=8,
                interval=8,
            )
        )

    def test_selector_chooses_top_two_compiling_failures_stably(self):
        details = [
            failed_detail(diagnostic="weak"),
            failed_detail(diagnostic="best"),
            failed_detail(diagnostic="second"),
            failed_detail(compiled=False, diagnostic="not eligible"),
        ]
        selected = self.verpo.select_group_teacher_candidates(
            details,
            [0.2, 0.9, 0.7, 1.0],
            top_n=2,
        )
        self.assertEqual(selected, [1, 2])

        # A deterministic group router must not let completion order break ties.
        tied = self.verpo.select_group_teacher_candidates(
            details[:3],
            [0.5, 0.5, 0.5],
            top_n=2,
        )
        self.assertEqual(tied, [0, 1])

    def test_group_payload_contains_selected_top_two_in_one_request(self):
        source_text = "F2\nC0\n\nAx86_64\nEä¸€\nD\nB\nä¸€ret|\nX\n"
        source = self.verpo.TeacherVisibleSource(
            task_id="task-1",
            text=source_text,
            text_sha256=hashlib.sha256(
                source_text.encode("utf-8")
            ).hexdigest(),
            source_record_sha256="a" * 64,
            system_prompt="F2 format guide",
            system_prompt_sha256=hashlib.sha256(
                b"F2 format guide"
            ).hexdigest(),
        )
        candidates = [
            {"group_index": 0, "candidate": "int fn0() => 0;"},
            {"group_index": 1, "candidate": "int fn0() => 2;"},
            {"group_index": 2, "candidate": "int fn0() => 3;"},
            {"group_index": 3, "candidate": "this does not compile"},
        ]
        details = [
            failed_detail(diagnostic="got 0"),
            failed_detail(diagnostic="got 2"),
            failed_detail(diagnostic="got 3"),
            failed_detail(compiled=False, diagnostic="syntax error"),
        ]
        selected = self.verpo.select_group_teacher_candidates(
            details,
            [0.2, 0.9, 0.7, 1.0],
            top_n=2,
        )
        payload = self.verpo.group_judge_payload_from_rollout(
            source=source,
            feedback_tests="void main() { expect(fn0(), 1); }",
            candidates=candidates,
            details=details,
            selected_indices=selected,
        )
        submitted = payload["candidates"]
        self.assertEqual(
            [row["group_index"] for row in submitted],
            [1, 2],
        )
        self.assertEqual(
            [row["candidate"] for row in submitted],
            [candidates[1]["candidate"], candidates[2]["candidate"]],
        )
        self.assertEqual(payload["source"], source_text)
        self.assertEqual(payload["source_format_guide"], "F2 format guide")
        serialized = json.dumps(payload, sort_keys=True)
        self.assertIn("expect(fn0(), 1)", serialized)
        self.assertNotIn(candidates[0]["candidate"], serialized)
        self.assertNotIn("this does not compile", serialized)
        self.assertNotIn("acceptance_tests", serialized)
        self.assertNotIn("dart_source", serialized)


class GroupJudgeRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.judge_module = load_module(JUDGE_PATH, "verpo_group_judge_test")

    def item(self, index: int):
        source = "F2\nC0\n\nAx86_64\nEä¸€\nD\nB\nä¸€ret|\nX\n"
        return {
            "group_index": index,
            "source": source,
            "source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "source_format_guide": "F2 format guide",
            "tests": "void main() { expect(fn0(), 1); }",
            "candidate": f"int fn0() => {index};",
            "diagnostic": f"got {index}",
            "compiled": True,
            "full_pass": False,
        }

    def test_score_group_uses_exactly_one_api_call_for_selected_candidates(self):
        judge = self.judge_module.VerpoJudge(
            fail_closed=True,
            thinking_mode="provider_default",
            reasoning_effort="high",
        )
        calls = []

        def fake_call(system, user):
            calls.append((system, user))
            return json.dumps({"scores": [80, 20]})

        judge._call = fake_call
        items = [self.item(index) for index in range(2)]
        self.assertEqual(judge.score_group(items), [0.8, 0.2])
        self.assertEqual(len(calls), 1)
        for item in items:
            self.assertIn(item["candidate"], calls[0][1])
            self.assertIn(item["diagnostic"], calls[0][1])

    def test_score_group_accepts_the_production_group_payload_in_one_call(self):
        judge = self.judge_module.VerpoJudge(
            fail_closed=True,
            api_style="openai_responses",
            reasoning_mode="standard",
            reasoning_effort="high",
        )
        calls = []

        def fake_call(system, user):
            calls.append((system, user))
            return json.dumps({"scores": [75, 25]})

        judge._call = fake_call
        items = [self.item(index) for index in range(2)]
        payload = {
            "source": items[0]["source"],
            "source_sha256": items[0]["source_sha256"],
            "source_format_guide": items[0]["source_format_guide"],
            "tests": items[0]["tests"],
            "candidates": [
                {
                    "group_index": item["group_index"],
                    "candidate": item["candidate"],
                    "diagnostic": item["diagnostic"],
                    "compiled": True,
                    "full_pass": False,
                }
                for item in items
            ],
        }
        self.assertEqual(judge.score_group(payload), [0.75, 0.25])
        self.assertEqual(len(calls), 1)

    def test_standard_responses_request_never_enables_pro_mode(self):
        standard = self.judge_module.VerpoJudge(
            api_style="openai_responses",
            reasoning_mode="standard",
            reasoning_effort="high",
        )
        options = standard._request_options(4096)
        self.assertEqual(options["reasoning"], {"effort": "high"})
        self.assertNotIn("mode", options["reasoning"])

        offline = self.judge_module.VerpoJudge(
            api_style="openai_responses",
            reasoning_mode="pro",
            reasoning_effort="max",
        )
        self.assertEqual(
            offline._request_options(4096)["reasoning"],
            {"effort": "max", "mode": "pro"},
        )

    def test_responses_api_group_call_is_sealed_and_uses_standard_reasoning(self):
        calls = []

        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    id="resp-group-1",
                    model="gpt-5.6-terra-2026-07-01",
                    status="completed",
                    incomplete_details=None,
                    output_text=json.dumps({"scores": [70, 30]}),
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=20,
                        total_tokens=120,
                        output_tokens_details=SimpleNamespace(
                            reasoning_tokens=12
                        ),
                    ),
                )

        judge = self.judge_module.VerpoJudge(
            model="gpt-5.6-terra",
            api_style="openai_responses",
            reasoning_mode="standard",
            reasoning_effort="high",
            max_tokens=4096,
            completion_retries=0,
            retry_max_tokens=4096,
            fail_closed=True,
        )
        judge._client = SimpleNamespace(responses=Responses())
        items = [self.item(index) for index in range(2)]
        self.assertEqual(judge.score_group(items), [0.7, 0.3])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "gpt-5.6-terra")
        self.assertEqual(calls[0]["reasoning"], {"effort": "high"})
        self.assertNotIn("mode", calls[0]["reasoning"])
        receipt = judge.receipt_attestation_since(0)["receipts"][0]
        self.assertTrue(receipt["validation"]["accepted"])
        self.assertFalse(
            receipt["validation"]["exact_requested_model_required"]
        )
        self.assertEqual(receipt["request"]["reasoning_mode"], "standard")
        self.assertEqual(receipt["response"]["finish_reason"], "stop")
        self.assertTrue(receipt["response"]["reasoning_content_present"])

    def test_score_group_rejects_positional_score_count_mismatch(self):
        judge = self.judge_module.VerpoJudge(
            fail_closed=True,
            thinking_mode="provider_default",
            reasoning_effort="high",
        )
        judge._call = lambda *_args: json.dumps({"scores": [80]})
        with self.assertRaises(self.judge_module.VerpoJudgeError):
            judge.score_group([self.item(0), self.item(1)])


class SparseTeacherAdvantageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verpo = load_module(TRAINER_PATH, "verpo_sparse_advantage_test")

    def test_only_observed_scores_are_centered_and_unselected_stay_zero(self):
        advantages, mask = self.verpo.sparse_teacher_advantages(
            group_size=4,
            selected_indices=[1, 3],
            scores=[0.8, 0.2],
        )
        self.assertEqual(mask, [False, True, False, True])
        self.assertEqual(advantages[0], 0.0)
        self.assertEqual(advantages[2], 0.0)
        self.assertAlmostEqual(advantages[1], 0.3)
        self.assertAlmostEqual(advantages[3], -0.3)
        self.assertAlmostEqual(
            sum(value for value, present in zip(advantages, mask) if present),
            0.0,
        )

    def test_timeout_or_missing_teacher_has_no_advantage_or_mask(self):
        advantages, mask = self.verpo.sparse_teacher_advantages(
            group_size=4,
            selected_indices=[],
            scores=[],
        )
        self.assertEqual(advantages, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(mask, [False, False, False, False])

    def test_single_observation_cannot_create_a_relative_advantage(self):
        advantages, mask = self.verpo.sparse_teacher_advantages(
            group_size=4,
            selected_indices=[2],
            scores=[0.9],
        )
        self.assertEqual(advantages, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(mask, [False, False, True, False])


class TrainerFallbackStructureTests(unittest.TestCase):
    def test_production_uses_group_api_once_and_not_per_candidate_score(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("judge.score_group("), 1)
        self.assertNotIn("judge.score(judge_items)", source)

    def test_group_teacher_failure_is_caught_queued_and_marked_missing(self):
        tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))

        def calls_named(nodes, name: str):
            return [
                node
                for parent in nodes
                for node in ast.walk(parent)
                if isinstance(node, ast.Call)
                and (
                    (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == name
                    )
                    or (
                        isinstance(node.func, ast.Name)
                        and node.func.id == name
                    )
                )
            ]

        guarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if calls_named(node.body, "score_group"):
                guarded.append(node)
        self.assertEqual(
            len(guarded),
            1,
            "the single inline group request must be protected by local fallback",
        )
        handler_nodes = [
            child
            for handler in guarded[0].handlers
            for child in handler.body
        ]
        self.assertTrue(
            calls_named(handler_nodes, "enqueue_teacher_escalation"),
            "teacher failure must enqueue the complete group for offline repair",
        )
        self.assertTrue(
            calls_named(handler_nodes, "sparse_teacher_advantages"),
            "teacher failure must be represented as missing, not zero-score data",
        )

    def test_candidate_journal_distinguishes_selected_observed_and_missing(self):
        tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
        update_keys = set()
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "update"
                or not isinstance(node.func.value, ast.Name)
                or node.func.value.id != "candidate"
            ):
                continue
            for argument in node.args:
                if not isinstance(argument, ast.Dict):
                    continue
                update_keys.update(
                    key.value
                    for key in argument.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                )
        self.assertIn("selected_for_teacher", update_keys)
        self.assertIn("teacher_score_observed", update_keys)


class EscalationQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verpo = load_module(TRAINER_PATH, "verpo_escalation_queue_test")

    def test_escalation_queue_is_jsonl_and_idempotent_by_group_key(self):
        record = {
            "task_id": "task-1",
            "group_ordinal": 8,
            "policy_version": 7,
            "run_contract_sha256": "a" * 64,
            "reason": "TimeoutError: inline deadline exceeded",
            "payload": {
                "source_sha256": "b" * 64,
                "tests": "visible tests",
                "candidates": [
                    {
                        "group_index": 0,
                        "candidate": "int fn0() => 0;",
                        "diagnostic": "got 0",
                    },
                    {
                        "group_index": 1,
                        "candidate": "int fn0() => 2;",
                        "diagnostic": "got 2",
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "escalations.jsonl"
            first = self.verpo.enqueue_teacher_escalation(queue, record)
            second = self.verpo.enqueue_teacher_escalation(queue, record)
            rows = [
                json.loads(line)
                for line in queue.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.assertEqual(first["escalation_key"], second["escalation_key"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], "task-1")
        self.assertEqual(len(rows[0]["payload"]["candidates"]), 2)
        self.assertRegex(rows[0]["escalation_key"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
