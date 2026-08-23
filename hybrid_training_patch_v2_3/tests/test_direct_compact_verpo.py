from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import direct_compact_verpo as verpo


def detail(passes, *, compiled=True, full_pass=False):
    return {
        "test_passes": list(passes),
        "compiled": compiled,
        "full_pass": full_pass,
        "diagnostic": "visible failure",
    }


class RewardTests(unittest.TestCase):
    def test_kde_bandwidth_is_population_std_over_two(self):
        group = [
            detail([True, True, True, True], full_pass=True),
            detail([True, True, True, False]),
            detail([True, True, False, False]),
            detail([True, False, False, False]),
        ]
        actual = verpo.verpo_local_rewards(group, alpha=2.0)
        rho = [1.0, 0.75, 0.5, 0.25]
        mean = sum(rho) / len(rho)
        sigma = math.sqrt(
            sum((value - mean) ** 2 for value in rho) / len(rho)
        ) / 2.0
        densities = [
            sum(
                math.exp(
                    -((rho[left] - rho[right]) ** 2)
                    / (2.0 * sigma * sigma)
                )
                for right in range(len(rho))
            )
            for left in range(len(rho))
        ]
        weights = [
            math.exp(-2.0 * value) / (density + 1e-8)
            for value, density in zip(rho, densities)
        ]
        expected = [
            sum(
                weight
                for weight, passed in zip(weights, row["test_passes"])
                if passed
            )
            for row in group
        ]
        for observed, wanted in zip(actual, expected):
            self.assertAlmostEqual(observed, wanted, places=12)

    def test_compile_gated_teacher_signal_preserves_verifier_endpoints(self):
        group = [
            detail([True], full_pass=True),
            detail([False]),
            detail([False], compiled=False),
        ]
        signals = verpo.compile_gated_teacher_signals(
            group, [0.0, 1.0, 1.0]
        )
        self.assertEqual(signals, [1.0, 1.0, 0.0])
        components = verpo.verpo_unified_advantages(
            group,
            [1.0, 0.0, 0.0],
            signals,
            beta=1.0,
            teacher_weight=10.0,
        )
        advantages = components["unified_advantages"]
        self.assertGreater(advantages[0], advantages[1])
        self.assertGreater(advantages[1], advantages[2])

    def test_verpo_uses_fnorm_one_and_separate_centered_components(self):
        advantages = verpo.mean_centered_advantages([1.0, 2.0, 3.0])
        self.assertEqual(advantages, [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertEqual(
            verpo.mean_centered_advantages([7.0, 7.0]), [0.0, 0.0]
        )
        group = [
            detail([True], full_pass=True),
            detail([False]),
            detail([False], compiled=False),
        ]
        components = verpo.verpo_unified_advantages(
            group,
            [3.0, 1.0, 0.0],
            [1.0, 0.6, 0.0],
            beta=2.0,
            teacher_weight=0.5,
        )
        global_expected = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]
        local_expected = [5.0 / 3.0, -1.0 / 3.0, -4.0 / 3.0]
        teacher_expected = [
            1.0 - 1.6 / 3.0,
            0.6 - 1.6 / 3.0,
            -1.6 / 3.0,
        ]
        for observed, expected in zip(
            components["global_advantages"], global_expected
        ):
            self.assertAlmostEqual(observed, expected)
        for observed, expected in zip(
            components["local_advantages"], local_expected
        ):
            self.assertAlmostEqual(observed, expected)
        for observed, expected in zip(
            components["teacher_advantages"], teacher_expected
        ):
            self.assertAlmostEqual(observed, expected)
        unified_expected = [
            global_value + 2.0 * local_value + 0.5 * teacher_value
            for global_value, local_value, teacher_value in zip(
                global_expected, local_expected, teacher_expected
            )
        ]
        for observed, expected in zip(
            components["unified_advantages"], unified_expected
        ):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(
            sum(components["unified_advantages"]), 0.0
        )

    def test_ppo_ratio_uses_saved_rollout_logprobs(self):
        current = torch.tensor([-1.1, -0.4], requires_grad=True)
        old = torch.tensor([-1.0, -0.5])
        loss = verpo.policy_token_loss(
            current, old, 1.0, ppo_clip=0.2
        )
        expected = -torch.minimum(
            torch.exp(current.detach() - old),
            torch.exp(current.detach() - old).clamp(0.8, 1.2),
        ).mean()
        self.assertAlmostEqual(float(loss.detach()), float(expected))
        loss.backward()
        self.assertIsNotNone(current.grad)


class VisibleVerifierBoundaryTests(unittest.TestCase):
    TESTS = """void main() {
  final candidate = fn0;
  expect(candidate(1), 2);
  expect(candidate(2), 3);
}

void expect(dynamic a, dynamic b) {
  if (a == b) return;
  throw '$a != $b';
}

void expectList(List a, List b) {
  for (var i = 0; i < a.length; i++) {
    expect(a[i], b[i]);
  }
}
"""

    def test_per_test_split_only_selects_main_assertions(self):
        variants = verpo.split_visible_expect_harnesses(self.TESTS)
        self.assertEqual(len(variants), 2)
        for variant in variants:
            # Definition, one behavioral assertion, and one recursive helper call.
            self.assertEqual(variant.count("expect("), 3)
            self.assertIn("expect(a[i], b[i]);", variant)

    def test_full_and_per_test_use_completion_attested_evaluator(self):
        answers = [
            (True, False, "full failed", "source"),
            (True, True, "", "source"),
            (True, False, "one failed", "source"),
        ]
        with patch.object(
            verpo, "evaluate_dart_jit_tests_detail", side_effect=answers
        ) as evaluate:
            result = verpo.score_dart_candidate(
                "int fn0(int x) => x + 1;",
                self.TESTS,
                "task",
                timeout=19,
                stability_runs=2,
            )
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(result["test_passes"], [True, False])
        self.assertTrue(result["compiled"])
        self.assertFalse(result["full_pass"])
        for call in evaluate.call_args_list:
            self.assertEqual(call.kwargs["timeout"], 19)
            self.assertEqual(call.kwargs["stability_runs"], 2)

    def test_judge_payload_is_a_strict_visible_only_whitelist(self):
        source_text = "F2\nC0\n\nAx86_64\nE一\nD\nB\n一ret|\nX\n"
        payload = verpo.judge_payload_from_rollout(
            source=verpo.TeacherVisibleSource(
                task_id="task",
                text=source_text,
                text_sha256=verpo.sha256_text(source_text),
                source_record_sha256="a" * 64,
                system_prompt="F2 test format guide",
                system_prompt_sha256=verpo.sha256_text(
                    "F2 test format guide"
                ),
            ),
            feedback_tests="PUBLIC_SENTINEL",
            candidate="CANDIDATE_SENTINEL",
            detail={
                "compiled": True,
                "full_pass": False,
                "diagnostic": "DIAGNOSTIC_SENTINEL",
                "acceptance_tests": "HIDDEN_SENTINEL",
                "dart_source": "REFERENCE_SENTINEL",
            },
        )
        self.assertEqual(
            set(payload),
            {
                "source",
                "source_sha256",
                "source_format_guide",
                "tests",
                "candidate",
                "diagnostic",
                "compiled",
                "full_pass",
            },
        )
        serialized = json.dumps(payload)
        self.assertNotIn("HIDDEN_SENTINEL", serialized)
        self.assertNotIn("REFERENCE_SENTINEL", serialized)
        self.assertIn("PUBLIC_SENTINEL", serialized)
        self.assertEqual(payload["source"], source_text)
        self.assertEqual(
            payload["source_format_guide"], "F2 test format guide"
        )

    def test_teacher_source_uses_exact_manifest_bound_f2_guide(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt_path = root / "prompts.jsonl"
            manifest_path = root / "prompts.jsonl.manifest.json"
            system_prompt = "exact manifest-bound F2 guide"
            system_sha = verpo.sha256_text(system_prompt)
            text = "F2\nC0\n\nAx86_64\nE一\nD\nB\n一ret|\nX\n"
            row = {
                "schema": "verified-api-readable-compact-row-v2",
                "task_id": "task",
                "text": text,
                "text_sha256": verpo.sha256_text(text),
                "representation_schema": "lossless-semantic-f2",
                "system_prompt_sha256": system_sha,
                "verified": {
                    "artifact_hashes": True,
                    "row_contract_hashes": True,
                    "codec_text_roundtrip": True,
                    "codec_token_id_roundtrip": True,
                    "student_constant_prefix": True,
                    "per_task_instruction_dictionary_roundtrip": True,
                    "compact_semantic_f2_roundtrip": True,
                    "branch_targets_reconstructed_from_cfg": True,
                    "visible_task_symbols_one_token": True,
                    "opaque_custom_ids_in_text": False,
                },
            }
            prompt_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            prompt_record = {
                "path": str(prompt_path),
                "size_bytes": prompt_path.stat().st_size,
                "sha256": verpo.sha256_file(prompt_path),
            }
            tokenizer_sha = "d" * 64
            manifest = {
                "schema": "verified-api-readable-compact-v2",
                "rows": 1,
                "output": prompt_record,
                "f2_prompt_contract": {
                    "representation_schema": "lossless-semantic-f2",
                    "system_prompt": system_prompt,
                    "system_prompt_sha256": system_sha,
                    "tokenizer_sha256": tokenizer_sha,
                    "all_rows_within_limit": True,
                },
                "invariants": {
                    "all_artifact_hashes_verified": True,
                    "all_row_contract_hashes_verified": True,
                    "all_codec_roundtrips_verified": True,
                    "all_student_constant_prefixes_verified": True,
                    "all_f2_semantic_roundtrips_verified": True,
                    "f2_system_prompt_self_contained_and_hashed": True,
                    "all_complete_prompts_within_limit": True,
                    "opaque_source_ids_expanded": True,
                    "cfg_explicit": True,
                    "all_user_functions_retained": True,
                    "all_external_symbols_retained": True,
                    "transfer_table_redundancy_proven": True,
                    "train_dev_representation_contract_identical": True,
                },
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            sources, attestation = verpo.load_teacher_visible_sources(
                prompt_path,
                expected_sha256=verpo.sha256_file(prompt_path),
                manifest_path=manifest_path,
                expected_manifest_sha256=verpo.sha256_file(manifest_path),
                student_tokenizer_sha256=tokenizer_sha,
            )
            self.assertEqual(sources["task"].system_prompt, system_prompt)
            self.assertEqual(
                attestation["manifest"]["sha256"],
                verpo.sha256_file(manifest_path),
            )


class ProductionContractTests(unittest.TestCase):
    def test_task_schedule_is_without_replacement_and_resume_stable(self):
        task_ids = [f"task-{index}" for index in range(11)]
        first = verpo.deterministic_task_schedule(
            task_ids, seed=42, rollout_groups=17
        )
        second = verpo.deterministic_task_schedule(
            task_ids, seed=42, rollout_groups=17
        )
        self.assertEqual(first, second)
        self.assertEqual(len(set(first[:11])), 11)
        self.assertEqual(len(set(first[11:])), 6)
        contract = verpo.task_sampling_contract(
            task_ids,
            seed=42,
            max_updates=17,
            rollout_batch_size=1,
        )
        self.assertEqual(contract["planned_unique_tasks"], 11)
        self.assertEqual(contract["complete_dataset_cycles"], 1)
        self.assertEqual(contract["partial_cycle_groups"], 6)
        self.assertFalse(contract["with_replacement_within_cycle"])

    def test_production_defaults_are_fresh_untruncated_groups_of_eight(self):
        args = verpo.parse_args(
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
            ]
        )
        self.assertEqual(args.group_size, 8)
        self.assertEqual(args.top_p, 1.0)
        self.assertEqual(args.top_k, 0)
        self.assertEqual(args.judge_mode, "sparse_inline")
        self.assertEqual(args.judge_interval, 8)
        self.assertEqual(args.judge_group_top_n, 2)
        self.assertEqual(args.judge_deadline_seconds, 60.0)
        self.assertEqual(args.judge_failure_policy, "local_only")
        self.assertEqual(args.judge_reasoning_mode, "standard")
        self.assertEqual(args.max_updates, 1232)
        self.assertEqual(args.checkpoint_interval, 154)
        self.assertEqual(args.verpo_beta, 1.0)
        self.assertGreaterEqual(
            args.max_updates * args.rollout_batch_size,
            args.expected_feedback_eligible_rows,
        )
        self.assertEqual(args.ppo_clip, 0.0)
        self.assertGreater(args.sft_replay_weight, 0.0)

    def test_truncated_rollout_distributions_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "top_p=1.0"):
            verpo.validate_rollout_distribution(
                group_size=8, top_p=0.95, top_k=0, temperature=0.8
            )
        with self.assertRaisesRegex(ValueError, "top_k=0"):
            verpo.validate_rollout_distribution(
                group_size=8, top_p=1.0, top_k=50, temperature=0.8
            )

    def test_one_static_optimizer_update_and_optimizer_step_checkpoint(self):
        source = (ROOT / "scripts/training/direct_compact_verpo.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(source.count("optimizer.step()"), 1)
        self.assertIn("checkpoint-optstep-", source)
        self.assertIn("rollout_token_logprobs", source)
        self.assertIn("updates_applied_to_rollout", source)

    def test_checkpoint_publication_is_immutable_and_journaled(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

            def save_pretrained(self, destination):
                path = Path(destination)
                path.mkdir()
                (path / "adapter_config.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                torch.save({"weight": self.weight.detach()}, path / "adapter.pt")

        class FakeOverlay:
            def overlay_state(self):
                return {
                    "schema": "source-token-embedding-overlay-v1",
                    "base_vocab_size": 1,
                    "source_token_ids": [1],
                    "source_embeddings": torch.ones(1, 1),
                }

        class FakeTokenizer:
            def save_pretrained(self, destination):
                path = Path(destination)
                path.mkdir()
                (path / "tokenizer_config.json").write_text(
                    "{}\n", encoding="utf-8"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "contract.json"
            contract.write_text('{"schema":"test"}\n', encoding="utf-8")
            output = root / "output"
            output.mkdir()
            model = FakeModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            attempt = verpo.create_journal_attempt(
                output,
                start_step=0,
                run_contract_sha256="a" * 64,
                parent_chain_sha256="0" * 64,
            )
            journal_attestation = verpo.write_step_journal(
                attempt,
                journal={
                    "schema": verpo.JOURNAL_SCHEMA,
                    "optimizer_step": 1,
                    "groups": [],
                    "deepseek_response_receipts": {
                        "schema": verpo.DEEPSEEK_RECEIPT_ATTESTATION_SCHEMA,
                        "receipt_count_before_step": 0,
                        "receipt_count_this_step": 0,
                        "cumulative_receipt_count": 0,
                        "first_receipt_index": None,
                        "last_receipt_index": None,
                        "previous_receipt_chain_sha256": "0" * 64,
                        "cumulative_receipt_chain_sha256": "0" * 64,
                        "receipts": [],
                        "plaintext_prompts_persisted": False,
                        "plaintext_reasoning_persisted": False,
                    },
                },
                previous_chain_sha256="0" * 64,
            )
            checkpoint = verpo.save_optimizer_checkpoint(
                output_dir=output,
                optimizer_step=1,
                model=model,
                overlay=FakeOverlay(),
                optimizer=optimizer,
                tokenizer=FakeTokenizer(),
                contract_path=contract,
                run_contract_sha256="a" * 64,
                base_provenance={
                    "decoder_model": "Qwen/Qwen3-8B",
                    "decoder_revision": "revision",
                },
                journal_attestation=journal_attestation,
                judge_telemetry={
                    "api_successes": 0,
                    "receipt_count": 0,
                    "unique_response_ids": 0,
                    "receipt_chain_sha256": "0" * 64,
                },
                response_id_sha256s=[],
            )
            self.assertTrue((checkpoint / "rollout_journal.json").is_file())
            self.assertTrue((checkpoint / "judge_response_ids.json").is_file())
            self.assertTrue(
                (checkpoint / "judge_response_receipts.jsonl").is_file()
            )
            self.assertTrue(
                (checkpoint / "checkpoint_provenance.json").is_file()
            )
            with self.assertRaises(FileExistsError):
                verpo.save_optimizer_checkpoint(
                    output_dir=output,
                    optimizer_step=1,
                    model=model,
                    overlay=FakeOverlay(),
                    optimizer=optimizer,
                    tokenizer=FakeTokenizer(),
                    contract_path=contract,
                    run_contract_sha256="a" * 64,
                    base_provenance={},
                    journal_attestation=journal_attestation,
                    judge_telemetry={},
                )


if __name__ == "__main__":
    unittest.main()
