from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.preprocessing import build_multifunction_executable_view as view
from scripts.preprocessing import build_verpo_feedback_view as feedback


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class ExecutableMultifunctionViewTests(unittest.TestCase):
    def _parent_fixture(self, root: Path) -> Path:
        contract = root / "contract.json"
        contract.write_text('{"schema":"test-contract"}\n', encoding="utf-8")
        codebook = root / "codebook.json"
        codebook.write_text('{"schema":"test-codebook"}\n', encoding="utf-8")
        excluded = sorted(view.EXECUTION_INELIGIBLE_TASK_IDS)
        safe = [f"safe-{index:04d}" for index in range(1578)]
        train_ids = safe[:300] + [excluded[0]] + safe[300:900] + [
            excluded[1]
        ] + safe[900:]
        self.assertEqual(len(train_ids), 1580)
        dev_ids = [f"heldout-{index:03d}" for index in range(175)]

        train = root / "train.jsonl"
        train_f2 = root / "train_f2.jsonl"
        dev = root / "dev.jsonl"
        write_jsonl(
            train,
            [
                {
                    "task_id": task_id,
                    "binary_multifunction_schema": view.REPRESENTATION_SCHEMA,
                    "compact_input_ids": [1, index],
                    "tests": (
                        "import 'package:test/test.dart';\n"
                        "void main() {\n"
                        "  expect(fn0(1), 1);\n"
                        "  expect(fn0(2), 2);\n"
                        "  expect(\n"
                        "    fn0(3),\n"
                        "    3,\n"
                        "  );\n"
                        "  expect(fn0(4), 4);\n"
                        "}\n"
                    ),
                    "acceptance_tests": "must-never-enter-feedback-output",
                }
                for index, task_id in enumerate(train_ids)
            ],
        )
        write_jsonl(
            train_f2,
            [
                {
                    "task_id": task_id,
                    "representation_schema": view.F2_REPRESENTATION_SCHEMA,
                    "text": f"F2\n{index}\n",
                }
                for index, task_id in enumerate(train_ids)
            ],
        )
        write_jsonl(
            dev,
            [
                {
                    "task_id": task_id,
                    "binary_multifunction_schema": view.REPRESENTATION_SCHEMA,
                    "compact_input_ids": [2, index],
                }
                for index, task_id in enumerate(dev_ids)
            ],
        )

        common_seal = {
            "schema": view.JOIN_SEAL_SCHEMA,
            "contract_sha256": view.sha256_file(contract),
            "representation_schema": view.REPRESENTATION_SCHEMA,
            "frontier_f2_schema": view.F2_REPRESENTATION_SCHEMA,
            "adapter_contract_sha256": "1" * 64,
            "adapter_script_sha256": "2" * 64,
            "source_function_bundles_sha256": "3" * 64,
            "source_symbol_attestation_used": True,
            "source_symbol_attestation_is_keyed": True,
            "source_symbol_attestation_file_sha256": "4" * 64,
            "source_symbol_attestation_key_id_sha256": "5" * 64,
            "raw_source_names_serialized": False,
            "sanitation_schema": "compact-target-harness-sanitation-v1",
            "sanitizer_sha256": "6" * 64,
            "evaluator_sha256": "7" * 64,
            "completion_attestation_id": "attested",
            "dart_version": "3.12.2",
            "stability_runs": 2,
            "quarantine_sha256": "8" * 64,
        }
        train_seal = root / "train.seal.json"
        write_json(
            train_seal,
            common_seal
            | {
                "selected_role": "fit",
                "training_allowed": True,
                "heldout_measure_only": False,
                "training_objective_scope": view.PARENT_TRAIN_SCOPE,
                "rows": 1580,
                "output_sha256": view.sha256_file(train),
                "executable_reward_eligible_rows": 1578,
                "execution_ineligible_task_ids": excluded,
            },
        )
        dev_seal = root / "dev.seal.json"
        write_json(
            dev_seal,
            common_seal
            | {
                "selected_role": "measure",
                "training_allowed": False,
                "heldout_measure_only": True,
                "rows": 175,
                "output_sha256": view.sha256_file(dev),
            },
        )
        f2_manifest = root / "train_f2.jsonl.manifest.json"
        write_json(
            f2_manifest,
            {
                "schema": view.F2_MANIFEST_SCHEMA,
                "rows": 1580,
                "dataset": view.file_record(train),
                "output": view.file_record(train_f2),
                "f2_prompt_contract": {
                    "representation_schema": view.F2_REPRESENTATION_SCHEMA,
                    "system_prompt": "format",
                    "system_prompt_sha256": hashlib.sha256(
                        b"format"
                    ).hexdigest(),
                    "tokenizer_sha256": "9" * 64,
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
            },
        )

        parent = root / "parent.build.json"
        write_json(
            parent,
            {
                "schema": view.PARENT_BUILD_SCHEMA,
                "representation_schema": view.REPRESENTATION_SCHEMA,
                "passed": True,
                "counts": {
                    "train_rows": 1580,
                    "dev_rows": 175,
                    "excluded_rows": 0,
                    "truncated_rows": 0,
                },
                "invariants": {
                    "all_user_functions_retained": True,
                    "all_machine_instructions_retained": True,
                    "all_cfg_edges_retained_with_global_offsets": True,
                    "all_external_aliases_and_exact_definitions_retained": True,
                    "source_token_id_set_preserved_from_parent": True,
                    "block_and_control_token_ids_preserved_from_parent": True,
                    "instruction_codebook_refit_from_train_only": True,
                    "heldout_rows_used_for_instruction_codebook_fit": 0,
                    "warmstart_overlay_rows_reusable_only_when_expansions_match": True,
                    "inline_cfg_source_is_current_containing_block": True,
                    "inline_cfg_omits_only_redundant_edge_source_tokens": True,
                    "all_inline_cfg_text_and_token_roundtrips_verified": True,
                    "all_f2_semantic_roundtrips_verified": True,
                    "all_student_rows_within_9000": True,
                    "all_api_prompts_within_12000": True,
                    "zero_excluded_rows": True,
                    "zero_truncated_rows": True,
                    "train_dev_task_sets_disjoint": True,
                    "dev_is_measure_only_and_not_training": True,
                    "train_dev_representation_contract_identical": True,
                },
                "inputs": {"contract": view.file_record(contract)},
                "derived_representation": {
                    "contract": view.file_record(contract),
                    "codebook": view.file_record(codebook),
                },
                "outputs": {
                    "contract": view.file_record(contract),
                    "codebook": view.file_record(codebook),
                    "train": view.file_record(train),
                    "train_seal": view.file_record(train_seal),
                    "train_f2": view.file_record(train_f2),
                    "train_f2_manifest": view.file_record(f2_manifest),
                    "dev": view.file_record(dev),
                    "dev_seal": view.file_record(dev_seal),
                },
            },
        )
        return parent

    def test_exact_safe1578_derivation_preserves_parent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self._parent_fixture(root)
            output = root / "executable"
            result = view.build_executable_view(
                parent_build_report=parent,
                expected_parent_build_report_sha256=view.sha256_file(parent),
                output_dir=output,
            )
            self.assertEqual(result["counts"]["executable_train_rows"], 1578)
            rows = view.load_jsonl(
                result["outputs"]["dataset"]["path"], "derived"
            )
            self.assertFalse(
                {row["task_id"] for row in rows}.intersection(
                    view.EXECUTION_INELIGIBLE_TASK_IDS
                )
            )
            validated = view.validate_executable_view(
                dataset=result["outputs"]["dataset"]["path"],
                seal=result["outputs"]["seal"]["path"],
                f2=result["outputs"]["f2"]["path"],
                f2_manifest=result["outputs"]["f2_manifest"]["path"],
                build_report=output / "executable_view.build.json",
                expected_build_report_sha256=view.sha256_file(
                    output / "executable_view.build.json"
                ),
                contract=result["outputs"]["contract"]["path"],
                verify_heldout=True,
            )
            self.assertEqual(validated["rows"], 1578)
            self.assertEqual(validated["heldout_rows"], 175)
            self.assertTrue(validated["heldout_bytes_opened_during_validation"])

    def test_train_side_validation_does_not_open_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self._parent_fixture(root)
            output = root / "executable"
            result = view.build_executable_view(
                parent_build_report=parent,
                expected_parent_build_report_sha256=view.sha256_file(parent),
                output_dir=output,
            )
            heldout = Path(result["heldout_measure_only"]["dataset"]["path"])
            heldout.unlink()
            validated = view.validate_executable_view(
                dataset=result["outputs"]["dataset"]["path"],
                seal=result["outputs"]["seal"]["path"],
                f2=result["outputs"]["f2"]["path"],
                f2_manifest=result["outputs"]["f2_manifest"]["path"],
                build_report=output / "executable_view.build.json",
                expected_build_report_sha256=view.sha256_file(
                    output / "executable_view.build.json"
                ),
                contract=result["outputs"]["contract"]["path"],
                verify_heldout=False,
            )
            self.assertFalse(
                validated["heldout_bytes_opened_during_validation"]
            )
            with self.assertRaises(FileNotFoundError):
                view.validate_executable_view(
                    dataset=result["outputs"]["dataset"]["path"],
                    seal=result["outputs"]["seal"]["path"],
                    f2=result["outputs"]["f2"]["path"],
                    f2_manifest=result["outputs"]["f2_manifest"]["path"],
                    build_report=output / "executable_view.build.json",
                    contract=result["outputs"]["contract"]["path"],
                    verify_heldout=True,
                )


class LauncherContractTests(unittest.TestCase):
    def test_post_qwen_launchers_have_no_old_dataset_or_checkpoint_routes(self):
        names = (
            "run_collect_chatgpt_compact_rs.sh",
            "run_finish_rs_sft.sh",
            "run_verpo_v2.sh",
            "run_rs_sft_then_verpo.sh",
        )
        for name in names:
            source = (ROOT / "fixed_training_launchers" / name).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("compact_fn0", source, name)
            self.assertNotIn("whole_real", source, name)
            self.assertNotIn("text_arm", source, name)

    def test_heldout_is_opened_only_after_verpo_training(self):
        finish = (
            ROOT / "fixed_training_launchers" / "run_finish_rs_sft.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--no_eval_during_training", finish)
        self.assertNotIn("--eval_file", finish)
        verpo = (
            ROOT / "fixed_training_launchers" / "run_verpo_v2.sh"
        ).read_text(encoding="utf-8")
        train_call = verpo.index(
            '"${PYTHON}" -m scripts.training.direct_compact_verpo'
        )
        heldout_open = verpo.index(
            "# All fitting is now complete. Only now may the measure-only split be opened."
        )
        self.assertLess(train_call, heldout_open)
        self.assertNotIn("functional_gate", verpo)

    def test_checkpoint_conditioned_prompt_modes_are_predeclared_per_arm(self):
        collect = (
            ROOT
            / "fixed_training_launchers"
            / "run_collect_chatgpt_compact_rs.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--direct_prompt_mode qwen_cot_v1", collect)
        verpo = (
            ROOT / "fixed_training_launchers" / "run_verpo_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"${EVALUATION_ROOT}/predictions/qwen.json" qwen_cot_v1',
            verpo,
        )
        for artifact in ("control.json", "rs_sft.json", "verpo.json"):
            self.assertIn(
                f'"${{EVALUATION_ROOT}}/predictions/{artifact}" code_only_v1',
                verpo,
            )
        chain = (
            ROOT
            / "hybrid_training_patch_v2_3"
            / "scripts"
            / "training"
            / "seal_post_qwen_chain.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"direct_prompt_modes": {', chain)
        self.assertIn(
            "Qwen CoT=qwen_cot_v1 and control/RS/VeRPO=code_only_v1",
            chain,
        )


class VerpoFeedbackViewTests(ExecutableMultifunctionViewTests):
    def test_train_only_half_split_is_attested_and_excludes_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_report = self._parent_fixture(root)
            executable = root / "executable"
            result = view.build_executable_view(
                parent_build_report=parent_report,
                expected_parent_build_report_sha256=view.sha256_file(
                    parent_report
                ),
                output_dir=executable,
            )
            safe_rows = view.load_jsonl(
                result["outputs"]["dataset"]["path"], "fixture safe rows"
            )
            safe_ids = [str(row["task_id"]) for row in safe_rows]
            expected_accounting = {
                "parent_rows": 1578,
                "eligible_rows": 1578,
                "excluded_rows": 0,
                "source_expect_cases": 6312,
                "visible_expect_cases": 3156,
                "holdback_expect_cases": 3156,
                "odd_case_tasks": 0,
            }
            eligible_digest = feedback.stable_sha256(safe_ids)
            excluded_digest = feedback.stable_sha256([])
            output = root / "feedback"
            report = feedback.build_feedback_view(
                executable_dataset=result["outputs"]["dataset"]["path"],
                executable_seal=result["outputs"]["seal"]["path"],
                executable_f2=result["outputs"]["f2"]["path"],
                executable_f2_manifest=result["outputs"]["f2_manifest"][
                    "path"
                ],
                executable_view_report=(
                    executable / "executable_view.build.json"
                ),
                expected_executable_view_report_sha256=view.sha256_file(
                    executable / "executable_view.build.json"
                ),
                contract=result["outputs"]["contract"]["path"],
                output_dir=output,
                seed=42,
                expected_accounting=expected_accounting,
                expected_eligible_task_ids_sha256=eligible_digest,
                expected_excluded_task_ids_sha256=excluded_digest,
            )
            self.assertEqual(report["accounting"]["eligible_rows"], 1578)
            self.assertEqual(report["accounting"]["visible_expect_cases"], 3156)
            self.assertEqual(report["accounting"]["holdback_expect_cases"], 3156)
            rollout = view.load_jsonl(
                output / "verpo_rollout_feedback.jsonl", "feedback rollout"
            )
            self.assertNotIn("acceptance_tests", rollout[0])
            self.assertNotIn("tests", rollout[0])
            self.assertNotIn("reward_holdback_tests", rollout[0])
            self.assertEqual(
                len(feedback.extract_expect_spans(rollout[0]["feedback_tests"])),
                2,
            )
            validated = feedback.validate_feedback_view(
                rollout=output / "verpo_rollout_feedback.jsonl",
                seal=output / "verpo_rollout_feedback.seal.json",
                f2=output / "verpo_teacher_f2.jsonl",
                f2_manifest=output / "verpo_teacher_f2.jsonl.manifest.json",
                build_report=output / "verpo_feedback_view.build.json",
                expected_build_report_sha256=view.sha256_file(
                    output / "verpo_feedback_view.build.json"
                ),
                public_manifest=output / "verpo_feedback_view.public.json",
                expected_public_manifest_sha256=view.sha256_file(
                    output / "verpo_feedback_view.public.json"
                ),
                executable_dataset=result["outputs"]["dataset"]["path"],
                executable_seal=result["outputs"]["seal"]["path"],
                executable_f2=result["outputs"]["f2"]["path"],
                executable_f2_manifest=result["outputs"]["f2_manifest"][
                    "path"
                ],
                executable_view_report=(
                    executable / "executable_view.build.json"
                ),
                expected_executable_view_report_sha256=view.sha256_file(
                    executable / "executable_view.build.json"
                ),
                contract=result["outputs"]["contract"]["path"],
                expected_accounting=expected_accounting,
                expected_eligible_task_ids_sha256=eligible_digest,
                expected_excluded_task_ids_sha256=excluded_digest,
            )
            self.assertFalse(validated["acceptance_tests_exposed"])
            self.assertFalse(validated["reward_holdback_exposed"])
            # The trainer-boundary validator must not resolve/open parent or
            # private holdback bytes. Move both away as sentinels.
            parent_dataset = Path(result["outputs"]["dataset"]["path"])
            parent_dataset.rename(root / "parent-do-not-open.jsonl")
            Path(result["outputs"]["seal"]["path"]).rename(
                root / "parent-seal-do-not-open.json"
            )
            Path(result["outputs"]["f2"]["path"]).rename(
                root / "parent-f2-do-not-open.jsonl"
            )
            Path(result["outputs"]["f2_manifest"]["path"]).rename(
                root / "parent-f2-manifest-do-not-open.json"
            )
            (output / "reward_holdback.private.jsonl").rename(
                root / "holdback-do-not-open.jsonl"
            )
            (output / "verpo_feedback_view.build.json").rename(
                root / "private-report-do-not-open.json"
            )
            boundary = feedback.validate_feedback_training_boundary(
                rollout=output / "verpo_rollout_feedback.jsonl",
                seal=output / "verpo_rollout_feedback.seal.json",
                f2=output / "verpo_teacher_f2.jsonl",
                f2_manifest=output / "verpo_teacher_f2.jsonl.manifest.json",
                public_manifest=output / "verpo_feedback_view.public.json",
                expected_public_manifest_sha256=view.sha256_file(
                    output / "verpo_feedback_view.public.json"
                ),
                contract=result["outputs"]["contract"]["path"],
                expected_accounting=expected_accounting,
                expected_eligible_task_ids_sha256=eligible_digest,
                expected_excluded_task_ids_sha256=excluded_digest,
            )
            self.assertFalse(
                boundary["parent_or_private_bytes_opened_during_validation"]
            )

    def test_feedback_membership_expectation_mismatch_fails_before_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_report = self._parent_fixture(root)
            executable = root / "executable"
            result = view.build_executable_view(
                parent_build_report=parent_report,
                expected_parent_build_report_sha256=view.sha256_file(
                    parent_report
                ),
                output_dir=executable,
            )
            with self.assertRaisesRegex(
                feedback.FeedbackViewError,
                "independently pinned production membership",
            ):
                feedback.build_feedback_view(
                    executable_dataset=result["outputs"]["dataset"]["path"],
                    executable_seal=result["outputs"]["seal"]["path"],
                    executable_f2=result["outputs"]["f2"]["path"],
                    executable_f2_manifest=result["outputs"]["f2_manifest"][
                        "path"
                    ],
                    executable_view_report=(
                        executable / "executable_view.build.json"
                    ),
                    expected_executable_view_report_sha256=view.sha256_file(
                        executable / "executable_view.build.json"
                    ),
                    contract=result["outputs"]["contract"]["path"],
                    output_dir=root / "must-not-exist",
                    seed=42,
                    expected_accounting={
                        "parent_rows": 1578,
                        "eligible_rows": 1577,
                        "excluded_rows": 1,
                        "source_expect_cases": 6312,
                        "visible_expect_cases": 3156,
                        "holdback_expect_cases": 3156,
                        "odd_case_tasks": 0,
                    },
                    expected_eligible_task_ids_sha256="0" * 64,
                    expected_excluded_task_ids_sha256="1" * 64,
                )
            self.assertFalse((root / "must-not-exist").exists())

    def test_single_or_no_expect_harness_is_excluded(self):
        for tests in (
            "void main() { print('manual'); }\n",
            "void main() { expect(fn0(), 1); }\n",
        ):
            with self.assertRaisesRegex(
                feedback.FeedbackViewError, "fewer than two"
            ):
                feedback.split_train_harness(
                    task_id="task", tests=tests, seed=42
                )


if __name__ == "__main__":
    unittest.main()
