from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "scripts" / "training"
GRPO = TRAINING / "graph_grpo_decompiler_antigravity.py"
JUDGE = TRAINING / "verpo_judge_antigravity.py"
REPAIR = TRAINING / "build_verpo_repair_dataset_antigravity.py"
RUNNER = TRAINING / "run_hybrid_curriculum_antigravity.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_grpo_pure_functions():
    tree = ast.parse(GRPO.read_text(encoding="utf-8"))
    wanted = {
        "verpo_group_rewards",
        "apply_compile_gated_judge_rewards",
    }
    nodes = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    nodes.extend(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    )
    namespace = {"math": math}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(GRPO), "exec"), namespace)
    return namespace


def detail(passes, *, compiled=True, full_pass=False):
    return {
        "test_passes": list(passes),
        "compiled": compiled,
        "full_pass": full_pass,
    }


def provider_response(
    response_id,
    *,
    content,
    finish_reason="stop",
    reasoning_content="",
    model="deepseek-chat",
    prompt_tokens=10,
    completion_tokens=5,
    total_tokens=15,
    system_fingerprint="deepseek-test-fingerprint",
):
    return SimpleNamespace(
        id=response_id,
        model=model,
        system_fingerprint=system_fingerprint,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                ),
            )
        ],
    )


class VeRPORewardIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_grpo_pure_functions()

    def test_judge_is_applied_after_group_reward_and_changes_failures(self):
        verpo = self.functions["verpo_group_rewards"]
        apply_judge = self.functions["apply_compile_gated_judge_rewards"]
        group = [
            detail([True, True], full_pass=True),
            detail([True, False]),
            detail([False, True]),
            detail([False, False], compiled=False),
        ]
        base = verpo(group, [-1.0] * 4, alpha=2.0, anchor_weight=1.0)
        updated, bonuses = apply_judge(
            group,
            base,
            [0.0, 0.9, 0.1, 1.0],
            weight=0.25,
            full_pass_margin=0.001,
        )
        self.assertGreater(updated[1], base[1])
        self.assertGreater(bonuses[1], bonuses[2])
        self.assertEqual(updated[3], base[3])
        self.assertGreater(updated[0], max(updated[1:]))

    def test_full_pass_dominates_even_with_large_teacher_weight(self):
        apply_judge = self.functions["apply_compile_gated_judge_rewards"]
        group = [
            detail([True], full_pass=True),
            detail([False]),
        ]
        updated, _ = apply_judge(
            group,
            [1.01, 1.0],
            [0.0, 1.0],
            weight=10.0,
            full_pass_margin=0.001,
        )
        self.assertAlmostEqual(updated[1], 1.009)
        self.assertGreater(updated[0], updated[1])

    def test_kde_uses_paper_std_over_two_bandwidth(self):
        verpo = self.functions["verpo_group_rewards"]
        group = [
            detail([True, True, True, True], full_pass=True),
            detail([True, True, True, False]),
            detail([True, True, False, False]),
            detail([True, False, False, False]),
        ]
        actual = verpo(
            group, [-1.0] * 4, alpha=2.0, anchor_weight=0.0, density_norm=True
        )
        rho = [1.0, 0.75, 0.5, 0.25]
        mean = sum(rho) / len(rho)
        std = math.sqrt(sum((x - mean) ** 2 for x in rho) / len(rho))
        sigma = std / 2.0
        densities = [
            sum(
                math.exp(-((rho[j] - rho[k]) ** 2) / (2.0 * sigma * sigma))
                for k in range(len(rho))
            )
            for j in range(len(rho))
        ]
        weights = [
            math.exp(-2.0 * rho[j]) / (densities[j] + 1e-8)
            for j in range(len(rho))
        ]
        expected = [
            sum(weight for weight, passed in zip(weights, row["test_passes"]) if passed)
            for row in group
        ]
        for got, wanted in zip(actual, expected):
            self.assertAlmostEqual(got, wanted, places=12)

    def test_reward_deadband_default_is_zero(self):
        tree = ast.parse(GRPO.read_text(encoding="utf-8"))
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "group_advantages"
        )
        self.assertEqual(ast.literal_eval(fn.args.defaults[-1]), 0.0)

    def test_rollout_distribution_is_not_top_p_or_top_k_truncated(self):
        source = GRPO.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("GRPO_GEN_TOP_P", "1.0")', source)
        self.assertGreaterEqual(source.count("top_k=0"), 2)
        self.assertIn("--gen_top_p must be 1.0", source)

    def test_recovery_snapshots_follow_optimizer_updates(self):
        source = GRPO.read_text(encoding="utf-8")
        self.assertIn('bool(stats.get("optimizer_stepped", 0.0))', source)
        self.assertIn("checkpoint-optstep-", source)
        self.assertNotIn('f"checkpoint-step-{step_count}"', source)


class JudgeIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(JUDGE, "verpo_judge_integrity_test")

    def item(self):
        source = "F2\nC0\n\nAx86_64\nE一\nD\nB\n一{ret}|\nX\n"
        return {
            "source": source,
            "source_sha256": self.module.hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "source_format_guide": "F2 test format guide",
            "tests": "void main() { assert(fn0() == 1); }",
            "candidate": "int fn0() => 0;",
            "diagnostic": "expected 1",
            "compiled": True,
            "full_pass": False,
        }

    def test_cache_key_covers_diagnostic_and_provider_identity(self):
        key = self.module._key
        base = key(
            "t",
            "c",
            "score",
            source="s",
            source_format_guide="g",
            diagnostic="d",
            model="m",
            base_url="u",
        )
        self.assertNotEqual(
            base,
            key(
                "t",
                "c",
                "score",
                source="s2",
                source_format_guide="g",
                diagnostic="d",
                model="m",
                base_url="u",
            ),
        )
        self.assertNotEqual(
            base,
            key(
                "t",
                "c",
                "score",
                source="s",
                source_format_guide="g",
                diagnostic="d2",
                model="m",
                base_url="u",
            ),
        )
        self.assertNotEqual(
            base,
            key(
                "t",
                "c",
                "score",
                source="s",
                source_format_guide="g",
                diagnostic="d",
                model="m2",
                base_url="u",
            ),
        )
        self.assertNotEqual(
            base,
            key(
                "t",
                "c",
                "score",
                source="s",
                source_format_guide="g",
                diagnostic="d",
                model="m",
                base_url="u2",
            ),
        )
        self.assertNotEqual(
            base,
            key(
                "t",
                "c",
                "score",
                source="s",
                source_format_guide="g",
                diagnostic="d",
                model="m",
                base_url="u",
                reasoning_effort="max",
            ),
        )

    def test_parse_failure_is_not_silently_cached_as_zero(self):
        judge = self.module.VerpoJudge(fail_closed=True)
        judge._call = lambda *_args: "score: 80"
        with self.assertRaises(self.module.VerpoJudgeError):
            judge.score([self.item()])
        telemetry = judge.telemetry()
        self.assertEqual(telemetry["parse_failures"], 1)
        self.assertEqual(telemetry["api_failures"], 1)
        self.assertEqual(telemetry["cache_entries"], 0)

    def test_success_is_cached_and_telemetry_proves_one_call(self):
        judge = self.module.VerpoJudge(fail_closed=True)
        calls = []

        def fake_call(*_args):
            calls.append(True)
            return "80"

        judge._call = fake_call
        self.assertEqual(judge.score([self.item()]), [0.8])
        self.assertEqual(judge.score([self.item()]), [0.8])
        self.assertEqual(len(calls), 1)
        telemetry = judge.telemetry()
        self.assertEqual(telemetry["api_successes"], 1)
        self.assertEqual(telemetry["cache_hits"], 1)

    def test_score_prompt_contains_exact_f2_source_and_guide(self):
        judge = self.module.VerpoJudge(fail_closed=True)
        calls = []

        def fake_call(system, user):
            calls.append((system, user))
            return "80"

        judge._call = fake_call
        row = self.item()
        self.assertEqual(judge.score([row]), [0.8])
        self.assertEqual(len(calls), 1)
        self.assertIn(row["source_format_guide"], calls[0][1])
        self.assertIn(row["source"], calls[0][1])
        self.assertIn(row["tests"], calls[0][1])
        self.assertIn(row["candidate"], calls[0][1])

    def test_source_hash_mismatch_fails_closed_before_api(self):
        judge = self.module.VerpoJudge(fail_closed=True)
        judge._call = lambda *_args: self.fail("API must not be called")
        row = self.item()
        row["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            self.module.VerpoJudgeError, "source hash mismatch"
        ):
            judge.score([row])

    def test_configuration_has_explicit_timeout(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            judge = self.module.VerpoJudge(timeout_seconds=17, max_retries=0)
            judge.validate_configuration()
            self.assertEqual(judge.timeout_seconds, 17)
            self.assertEqual(judge.max_retries, 0)

    def test_deepseek_thinking_defaults_have_large_bounded_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            judge = self.module.VerpoJudge()
        self.assertEqual(judge.thinking_mode, "enabled")
        self.assertEqual(judge.reasoning_effort, "max")
        self.assertEqual(judge.max_tokens, 12288)
        self.assertEqual(judge.retry_max_tokens, 32768)
        self.assertEqual(judge._retry_token_budgets(), [12288, 32768, 32768])

    def test_reasoning_length_response_retries_with_larger_budget(self):
        responses = [
            provider_response(
                "response-length",
                content="",
                finish_reason="length",
                reasoning_content="unfinished hidden reasoning",
            ),
            provider_response(
                "response-stop",
                content="80",
            ),
        ]
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return responses.pop(0)

        judge = self.module.VerpoJudge(
            max_tokens=128,
            completion_retries=2,
            retry_max_tokens=2048,
            thinking_mode="disabled",
            fail_closed=True,
        )
        judge._client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        self.assertEqual(judge.score([self.item()]), [0.8])
        self.assertEqual([call["max_tokens"] for call in calls], [128, 512])
        self.assertEqual(
            calls[0]["extra_body"], {"thinking": {"type": "disabled"}}
        )
        self.assertEqual(calls[0]["reasoning_effort"], "max")
        self.assertNotIn("temperature", calls[0])
        self.assertNotIn("top_p", calls[0])
        telemetry = judge.telemetry()
        self.assertEqual(telemetry["api_calls"], 2)
        self.assertEqual(telemetry["completion_retries"], 1)
        self.assertEqual(telemetry["length_responses"], 1)
        self.assertEqual(telemetry["empty_responses"], 1)
        self.assertEqual(telemetry["reasoning_responses"], 1)
        self.assertEqual(telemetry["api_successes"], 1)
        self.assertEqual(telemetry["api_failures"], 0)

    def test_exhausted_empty_completions_fail_closed_and_are_not_cached(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return provider_response(
                    f"response-length-{len(calls)}",
                    content="",
                    finish_reason="length",
                    reasoning_content="still thinking",
                )

        judge = self.module.VerpoJudge(
            max_tokens=128,
            completion_retries=1,
            retry_max_tokens=512,
            thinking_mode="provider_default",
            fail_closed=True,
        )
        judge._client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        with self.assertRaisesRegex(
            self.module.VerpoJudgeError, "no complete final content"
        ):
            judge.score([self.item()])
        self.assertEqual([call["max_tokens"] for call in calls], [128, 512])
        self.assertNotIn("extra_body", calls[0])
        telemetry = judge.telemetry()
        self.assertEqual(telemetry["api_calls"], 2)
        self.assertEqual(telemetry["api_failures"], 1)
        self.assertEqual(telemetry["cache_entries"], 0)

    def test_response_receipt_binds_identity_usage_and_hides_prompt(self):
        row = self.item()

        class Completions:
            def create(self, **_kwargs):
                return provider_response("response-receipt", content="80")

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "receipts.jsonl"
            judge = self.module.VerpoJudge(
                fail_closed=True,
                receipt_journal_path=receipt_path,
            )
            judge._client = SimpleNamespace(
                chat=SimpleNamespace(completions=Completions())
            )
            self.assertEqual(judge.score([row]), [0.8])
            attestation = judge.receipt_attestation_since(0)
            self.assertEqual(attestation["receipt_count_this_step"], 1)
            receipt = attestation["receipts"][0]
            self.assertEqual(receipt["response"]["id"], "response-receipt")
            self.assertEqual(receipt["response"]["model"], "deepseek-chat")
            self.assertEqual(receipt["response"]["prompt_tokens"], 10)
            self.assertTrue(receipt["validation"]["accepted"])
            serialized = json.dumps(attestation, sort_keys=True)
            self.assertNotIn(row["source"], serialized)
            self.assertNotIn(row["candidate"], serialized)
            self.assertNotIn("unfinished hidden reasoning", serialized)
            self.assertEqual(len(receipt_path.read_text().splitlines()), 1)

    def test_response_model_and_positive_usage_fail_closed(self):
        responses = [
            provider_response(
                "wrong-model",
                content="80",
                model="not-the-requested-model",
            ),
            provider_response(
                "zero-usage",
                content="80",
                prompt_tokens=0,
                total_tokens=5,
            ),
            provider_response(
                "unequal-total",
                content="80",
                total_tokens=16,
            ),
        ]

        class Completions:
            def create(self, **_kwargs):
                return responses.pop(0)

        judge = self.module.VerpoJudge(fail_closed=True)
        judge._client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        with self.assertRaisesRegex(
            self.module.VerpoJudgeError, "response_model_mismatch"
        ):
            judge._call("system", "user")
        with self.assertRaisesRegex(
            self.module.VerpoJudgeError, "invalid_prompt_tokens"
        ):
            judge._call("system", "user")
        with self.assertRaisesRegex(
            self.module.VerpoJudgeError, "invalid_total_tokens"
        ):
            judge._call("system", "user")
        attestation = judge.receipt_attestation_since(0)
        self.assertEqual(attestation["receipt_count_this_step"], 3)
        self.assertTrue(
            all(
                receipt["validation"]["accepted"] is False
                for receipt in attestation["receipts"]
            )
        )

    def test_duplicate_response_id_is_rejected_and_resume_chain_continues(self):
        responses = [
            provider_response("unique-one", content="80"),
            provider_response("unique-one", content="80"),
        ]

        class Completions:
            def create(self, **_kwargs):
                return responses.pop(0)

        judge = self.module.VerpoJudge(fail_closed=True)
        judge._client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        self.assertEqual(judge._call("system", "user"), "80")
        with self.assertRaisesRegex(
            self.module.VerpoJudgeError, "duplicate_response_id"
        ):
            judge._call("system", "user")

        first = self.module.VerpoJudge(fail_closed=True)
        first._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: provider_response(
                        "resume-one", content="80"
                    )
                )
            )
        )
        self.assertEqual(first._call("system", "user"), "80")
        prior = first.telemetry()
        second = self.module.VerpoJudge(
            fail_closed=True,
            receipt_chain_seed=prior["receipt_chain_sha256"],
            receipt_index_offset=prior["receipt_count"],
            prior_response_id_sha256s=first.response_id_sha256s(),
        )
        second._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: provider_response(
                        "resume-two", content="80"
                    )
                )
            )
        )
        self.assertEqual(second._call("system", "user"), "80")
        resumed = second.receipt_attestation_since(1)
        self.assertEqual(resumed["first_receipt_index"], 2)
        self.assertEqual(
            resumed["previous_receipt_chain_sha256"],
            prior["receipt_chain_sha256"],
        )


class RepairAndRunnerIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repair = load_module(REPAIR, "verpo_repair_integrity_test")
        cls.runner = load_module(RUNNER, "verpo_runner_integrity_test")

    def test_task_id_joins_prediction_id(self):
        prediction = {"id": "task-7", "predictions": ["x"]}
        index, ambiguous = self.repair._prediction_index([prediction])
        matched = self.repair._match_prediction(
            {"task_id": "task-7"}, index, ambiguous
        )
        self.assertIs(matched, prediction)

    def test_ambiguous_prediction_identity_fails_closed(self):
        index, ambiguous = self.repair._prediction_index(
            [
                {"id": "same", "predictions": ["a"]},
                {"task_id": "same", "predictions": ["b"]},
            ]
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.repair._match_prediction({"task_id": "same"}, index, ambiguous)

    def test_repair_contract_rejects_hidden_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair.jsonl"
            row = {
                "task_id": "t",
                "prior_attempt": "int fn0()=>0;",
                "repair_feedback": "wrong value",
                "feedback_tests": "assert(fn0()==1);",
                "acceptance_tests": "assert(fn0()==99);",
                "verpo_repair": True,
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "acceptance_tests"):
                self.runner.validate_verpo_repair_dataset(path)

    def test_repair_contract_accepts_visible_only_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair.jsonl"
            row = {
                "task_id": "t",
                "prior_attempt": "int fn0()=>0;",
                "repair_feedback": "wrong value",
                "feedback_tests": "assert(fn0()==1);",
                "verpo_repair": True,
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = self.runner.validate_verpo_repair_dataset(path)
            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["unique_tasks"], 1)

    def test_resume_signature_includes_nonsecret_judge_identity(self):
        base = {
            "GRAPH_OUTPUT_DIR": "out",
            "VERPO_JUDGE_MODEL": "deepseek-chat",
            "DEEPSEEK_API_KEY": "secret",
        }
        snapshot = self.runner.env_snapshot(base)
        self.assertEqual(snapshot["VERPO_JUDGE_MODEL"], "deepseek-chat")
        self.assertNotIn("DEEPSEEK_API_KEY", snapshot)
        sig1 = self.runner.stage_signature(["python", "-m", "trainer"], base, [])
        changed = dict(base, VERPO_JUDGE_MODEL="deepseek-reasoner")
        sig2 = self.runner.stage_signature(["python", "-m", "trainer"], changed, [])
        self.assertNotEqual(sig1, sig2)

    def test_shortcut_checkpoints_participate_in_architecture_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_checkpoint = root / "text" / "pytorch_model.bin"
            verpo_checkpoint = root / "verpo" / "pytorch_model.bin"
            args = SimpleNamespace(
                architecture_env_json="",
                probe_checkpoint="",
                initial_checkpoint="",
                stage1_checkpoint="",
                text_sft_checkpoint=str(text_checkpoint),
                verpo_checkpoint=str(verpo_checkpoint),
            )
            self.assertEqual(
                self.runner._checkpoint_provenance_candidates(args),
                [
                    text_checkpoint.resolve().parent / "run_provenance.json",
                    verpo_checkpoint.resolve().parent / "run_provenance.json",
                ],
            )

    def test_text_verpo_defaults_to_recertified_anchor(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            '"06a_recertified_verified_rs_sft"\n'
            '                    / "verified_rs_sft.jsonl"',
            source,
        )

    def test_known_legacy_provenance_recovers_only_source_bound_defaults(self):
        trainer = TRAINING / "graph_encoder_decoder_decompiler_v2_antigravity.py"
        trainer_sha = self.runner.sha256_file(trainer)
        self.assertIn(
            trainer_sha, self.runner._LEGACY_ARCH_DEFAULT_SOURCE_HASHES
        )
        payload = {
            "source_files": [
                {
                    "path": str(trainer),
                    "sha256": trainer_sha,
                }
            ]
        }
        completed = self.runner._complete_known_legacy_architecture(
            payload=payload,
            environment={},
            args=SimpleNamespace(project_root=str(ROOT)),
            provenance_path=Path("legacy.json"),
        )
        self.assertEqual(completed, self.runner._LEGACY_ARCH_DEFAULTS)

        payload["source_files"][0]["sha256"] = "0" * 64
        rejected = self.runner._complete_known_legacy_architecture(
            payload=payload,
            environment={},
            args=SimpleNamespace(project_root=str(ROOT)),
            provenance_path=Path("unknown.json"),
        )
        self.assertEqual(rejected, {})

    def test_installer_ships_judge_and_repair_builder(self):
        installer = load_module(ROOT / "apply_hybrid_patch.py", "installer_integrity_test")
        self.assertIn(
            "scripts/training/verpo_judge_antigravity.py", installer.FILES
        )
        self.assertIn(
            "scripts/training/build_verpo_repair_dataset_antigravity.py",
            installer.FILES,
        )


if __name__ == "__main__":
    unittest.main()
