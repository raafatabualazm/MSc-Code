from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from models.direct_compact_causal import DirectCompactContract, sha256_file
from scripts.evaluation import durable_evaluation_journal as durable_journal
from scripts.training import build_direct_compact_rs_sft as builder
from scripts.training import (
    import_direct_compact_rs_hard_targets as hard_target_importer,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def contract_fixture(path: Path) -> DirectCompactContract:
    digest = "a" * 64
    contract = DirectCompactContract(
        schema="direct-compact-causal-v1",
        codec_sha256=digest,
        codebook_sha256="b" * 64,
        tokenizer_json_sha256="c" * 64,
        tokenizer_fingerprint_sha256="d" * 64,
        model_config_sha256="e" * 64,
        decoder_model="Qwen/test",
        decoder_revision="immutable-revision",
        target_function="fn0",
        target_language="Dart",
        dfg_extractor_sha256="f" * 64,
        lossless_domain="scrubbed_canonical_graph",
        base_vocab_size=4,
        source_token_ids=(4,),
        source_token_expansions=((4, (2,)),),
    )
    path.write_text(
        json.dumps(contract.as_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def compact_row(
    contract: DirectCompactContract, task_id: str
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "compact_input_ids": [4, 2],
        "compact_codec_sha256": contract.codec_sha256,
        "compact_codebook_sha256": contract.codebook_sha256,
        "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
        "dart_source": "int fn0(int x) => x + 1;",
        "acceptance_tests": "assert(fn0(1) == 2);",
    }


class DirectCompactRsSftBuilderTests(unittest.TestCase):
    def test_builds_exact_matched_half_repair_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract = contract_fixture(contract_path)
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            repairs = root / "repairs.jsonl"
            repairs_report = root / "repairs.report.json"
            executable_report = root / "executable.build.json"
            output = root / "output"
            write_jsonl(train, [compact_row(contract, "fit-1")])
            write_jsonl(heldout, [compact_row(contract, "measure-1")])
            write_jsonl(
                repairs,
                [{"task_id": "fit-1", "code": "int fn0(int x) => 1 + x;"}],
            )
            seal = {
                "schema": "compact-public-private-join-seal-v1",
                "selected_role": "fit",
                "output_sha256": sha256_file(train),
                "contract_sha256": sha256_file(contract_path),
                "rows": 1,
                "adapter_contract_sha256": "1" * 64,
                "adapter_script_sha256": "2" * 64,
                "source_function_bundles_sha256": "3" * 64,
                "source_symbol_attestation_used": True,
                "source_symbol_attestation_is_keyed": True,
                "source_symbol_attestation_file_sha256": "4" * 64,
                "source_symbol_attestation_key_id_sha256": "5" * 64,
                "raw_source_names_serialized": False,
                "sanitation_schema": "test",
                "sanitizer_sha256": "6" * 64,
                "evaluator_sha256": "7" * 64,
                "completion_attestation_id": "test",
                "dart_version": "test",
                "stability_runs": 2,
                "quarantine_sha256": "8" * 64,
            }
            seal_path = root / "train.seal.json"
            seal_path.write_text(json.dumps(seal), encoding="utf-8")
            executable_report.write_text("{}\n", encoding="utf-8")
            repairs_report.write_text(
                json.dumps(
                    {
                        "schema": "direct-compact-openai-rs-harvest-v2",
                        "status": "complete",
                        "provider": "openai",
                        "api": "responses",
                        "base_url": "https://api.openai.com/v1",
                        "requested_model": "gpt-5.6-sol",
                        "production_coverage_met": True,
                        "inputs": {
                            "train_file": {"sha256": sha256_file(train)},
                            "train_seal": {"sha256": sha256_file(seal_path)},
                            "executable_view": {
                                "report": {
                                    "sha256": sha256_file(executable_report)
                                }
                            },
                        },
                        "outputs": {
                            "verified_repairs_sha256": sha256_file(repairs)
                        },
                    }
                ),
                encoding="utf-8",
            )

            evaluator = types.ModuleType(
                "graph_compile_at_k_antigravity"
            )
            evaluator.evaluate_dart_jit_tests_detail = (
                lambda code, tests, identity, timeout, stability_runs: (
                    True,
                    True,
                    "ok",
                    code + tests + identity,
                )
            )
            argv = [
                "build_direct_compact_rs_sft.py",
                "--base_train",
                str(train),
                "--base_train_seal",
                str(seal_path),
                "--contract",
                str(contract_path),
                "--executable_view_report",
                str(executable_report),
                "--expected_executable_view_report_sha256",
                sha256_file(executable_report),
                "--repairs",
                f"chatgpt={repairs}",
                "--repair_report",
                f"chatgpt={repairs_report}",
                "--output_dir",
                str(output),
                "--rows_per_arm",
                "4",
                "--min_unique_repairs",
                "1",
                "--workers",
                "1",
                "--allow_low_coverage_smoke",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(
                sys.modules,
                {"graph_compile_at_k_antigravity": evaluator},
            ), mock.patch.object(
                builder,
                "validate_executable_view",
                return_value={
                    "heldout": {"path": str(heldout), "sha256": sha256_file(heldout)},
                    "heldout_seal": {"path": "heldout.seal", "sha256": "9" * 64},
                    "heldout_rows": 1,
                    "heldout_task_ids_sha256": "a" * 64,
                },
            ):
                builder.main()

            intervention = [
                json.loads(line)
                for line in (output / "rs_sft_50_50.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            control = [
                json.loads(line)
                for line in (output / "gold_only_matched.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            schedule = [
                json.loads(line)
                for line in (output / "schedule.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(intervention), 4)
            self.assertEqual(len(control), 4)
            self.assertEqual(
                sum(row["kind"] == "repair" for row in schedule), 2
            )
            self.assertEqual(
                [row["compact_input_ids"] for row in intervention],
                [row["compact_input_ids"] for row in control],
            )
            self.assertEqual(
                sum(
                    row["dart_source"] == "int fn0(int x) => 1 + x;"
                    for row in intervention
                ),
                2,
            )
            report = json.loads(
                (output / "build_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["unique_recertified_tasks"], 1)
            self.assertTrue(
                report["arms"]["source_sequence_exactly_matched"]
            )

    def test_repair_selection_is_deterministic_across_input_order(self) -> None:
        rows = [
            {
                "task_id": "t",
                "provider": "qwen",
                "code": "longer code",
            },
            {
                "task_id": "t",
                "provider": "chatgpt",
                "code": "short",
            },
        ]
        for row in rows:
            row["code_sha256"] = hashlib.sha256(
                row["code"].encode()
            ).hexdigest()
        first, alternatives = builder.choose_repairs(rows)
        second, _ = builder.choose_repairs(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(first["t"]["code"], "short")
        self.assertEqual(alternatives["t"], 2)

    def test_exact_duplicate_retains_all_provider_provenance(self) -> None:
        code = "int fn0(int x) => x + 1;"
        rows = [
            {
                "task_id": "t",
                "provider": "qwen37_max",
                "provider_identity": {
                    "key": "qwen37_max",
                    "requested_model": "qwen3.7-max",
                },
                "artifact_sha256": "1" * 64,
                "repair_report_sha256": "2" * 64,
                "artifact_row": 7,
                "source_row_sha256": "3" * 64,
                "source_schema": "direct-compact-rs-hard-target-v1",
                "provider_provenance": {"request_id": "qwen-request"},
                "code": code,
            },
            {
                "task_id": "t",
                "provider": "chatgpt",
                "provider_identity": {
                    "key": "chatgpt",
                    "requested_model": "gpt-5.6-sol",
                },
                "artifact_sha256": "4" * 64,
                "repair_report_sha256": "5" * 64,
                "artifact_row": 11,
                "source_row_sha256": "6" * 64,
                "source_schema": "direct-compact-openai-rs-harvest-v2",
                "provider_provenance": {"response_id": "openai-response"},
                "code": code,
            },
        ]
        for row in rows:
            row["code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
        forward, alternatives = builder.choose_repairs(rows)
        reverse, _ = builder.choose_repairs(list(reversed(rows)))
        self.assertEqual(forward, reverse)
        selected = forward["t"]
        self.assertEqual(alternatives["t"], 1)
        self.assertEqual(
            {
                value["provider"]
                for value in selected["dedupe_contributors"]
            },
            {"qwen37_max", "chatgpt"},
        )
        self.assertEqual(len(selected["dedupe_contributors"]), 2)

    def test_reasoning_payload_screen_is_recursive_but_allows_hashes(self) -> None:
        self.assertTrue(
            hard_target_importer.contains_prohibited_reasoning(
                {"source": {"reasoning_content": "private chain"}}
            )
        )
        self.assertFalse(
            hard_target_importer.contains_prohibited_reasoning(
                {
                    "source": {
                        "raw_reasoning_sha256": "a" * 64,
                        "reasoning_characters": 4096,
                    }
                }
            )
        )

    def test_imports_and_revalidates_sealed_provider_hard_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract = contract_fixture(contract_path)
            train = root / "train.jsonl"
            write_jsonl(train, [compact_row(contract, "fit-1")])
            executable_report = root / "executable.build.json"
            executable_report.write_text("{}\n", encoding="utf-8")
            evaluator = root / "production_evaluator.py"
            evaluator.write_text(
                "COMPLETION_ATTESTATION_ID = 'test-attestation'\n"
                "def evaluate_dart_jit_tests_detail("
                "code, tests, identity, timeout, stability_runs):\n"
                "    ok = code.startswith('```dart\\nclass Helper')\n"
                "    return ok, ok, 'ok' if ok else 'not-fenced', code\n",
                encoding="utf-8",
            )
            evaluator_sha = sha256_file(evaluator)
            seal = {
                "schema": "compact-public-private-join-seal-v1",
                "selected_role": "fit",
                "output_sha256": sha256_file(train),
                "contract_sha256": sha256_file(contract_path),
                "rows": 1,
                "adapter_contract_sha256": "1" * 64,
                "adapter_script_sha256": "2" * 64,
                "source_function_bundles_sha256": "3" * 64,
                "source_symbol_attestation_used": True,
                "source_symbol_attestation_is_keyed": True,
                "source_symbol_attestation_file_sha256": "4" * 64,
                "source_symbol_attestation_key_id_sha256": "5" * 64,
                "raw_source_names_serialized": False,
                "sanitation_schema": "test",
                "sanitizer_sha256": "6" * 64,
                "evaluator_sha256": evaluator_sha,
                "completion_attestation_id": "test-attestation",
                "dart_version": "Dart test",
                "stability_runs": 2,
                "quarantine_sha256": "8" * 64,
            }
            seal_path = root / "train.seal.json"
            seal_path.write_text(json.dumps(seal), encoding="utf-8")
            source_report = root / "qwen.report.json"
            provider = {
                "key": "qwen37_max",
                "organization": "Alibaba Cloud",
                "api": "chat.completions",
                "requested_model": "qwen3.7-max",
                "returned_model_must_equal_requested": True,
                "returned_models": ["qwen3.7-max"],
            }
            executable_view = {
                "rows": 1,
                "parent_rows": 2776,
                "heldout": {"path": "never-opened-heldout.jsonl"},
                "heldout_seal": {"path": "never-opened-heldout.seal.json"},
                "heldout_rows": 175,
                "heldout_task_ids_sha256": "9" * 64,
                "heldout_bytes_opened_during_validation": False,
            }
            fit_binding = {
                "base_train_sha256": sha256_file(train),
                "base_train_seal_sha256": sha256_file(seal_path),
                "contract_sha256": sha256_file(contract_path),
                "executable_view_report_sha256": sha256_file(
                    executable_report
                ),
            }
            fit_universe = {
                **fit_binding,
                "parent_fit_rows": 2776,
                "executable_rows": 1,
                "heldout_rows": 175,
                "heldout_task_ids_sha256": "9" * 64,
                "heldout_intersection_count": 0,
                "heldout_bytes_opened_during_harvest": False,
            }
            verification = {
                "evaluator_sha256": evaluator_sha,
                "completion_attestation_id": "test-attestation",
                "dart_version": "Dart test",
                "stability_runs": 2,
                "compiled": True,
                "passed": True,
                "acceptance_holdback_exposed_to_provider": False,
                "heldout_tests_exposed_to_provider": False,
            }
            code = (
                "class Helper { static int add(int x) => x + 1; }\n"
                "int fn0(int x) => Helper.add(x);"
            )
            provenance = {
                "request_id": "qwen-request-1",
                "candidate_index": 0,
            }
            source_row = {
                "schema": hard_target_importer.SOURCE_ROW_SCHEMA,
                "provider_key": "qwen37_max",
                "provider": provider,
                "task_id": "fit-1",
                "code": code,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "fit_bindings": fit_binding,
                "verification": verification,
                "source_provenance": provenance,
                "source_provenance_sha256": (
                    hard_target_importer.stable_sha256(provenance)
                ),
            }
            source_targets = root / "qwen.targets.jsonl"
            write_jsonl(source_targets, [source_row])
            source_report.write_text(
                json.dumps(
                    {
                        "schema": hard_target_importer.SOURCE_REPORT_SCHEMA,
                        "status": "complete",
                        "provider_key": "qwen37_max",
                        "provider": provider,
                        "source_targets_sha256": sha256_file(source_targets),
                        "verifier_implementation_sha256": evaluator_sha,
                        "code_only": True,
                        "reasoning_is_not_training_target": True,
                    }
                ),
                encoding="utf-8",
            )
            source_seal = {
                "schema": hard_target_importer.SOURCE_SEAL_SCHEMA,
                "status": "complete",
                "provider_key": "qwen37_max",
                "provider": provider,
                "code_only": True,
                "reasoning_is_not_training_target": True,
                "rows": 1,
                "output_sha256": sha256_file(source_targets),
                "source_report_sha256": sha256_file(source_report),
                "fit_universe": fit_universe,
                "verifier": {
                    "evaluator_sha256": evaluator_sha,
                    "completion_attestation_id": "test-attestation",
                    "dart_version": "Dart test",
                    "stability_runs": 2,
                    "all_candidates_compiled": True,
                    "all_candidates_passed": True,
                    "acceptance_holdback_exposed_to_provider": False,
                    "heldout_tests_exposed_to_provider": False,
                    "verifier_implementation_sha256": evaluator_sha,
                },
                "task_set_sha256": hard_target_importer.stable_sha256(
                    ["fit-1"]
                ),
                "ordered_candidate_keys_sha256": (
                    hard_target_importer.stable_sha256(
                        [
                            {
                                "task_id": "fit-1",
                                "code_sha256": source_row["code_sha256"],
                            }
                        ]
                    )
                ),
                "source_journal_chain_head_sha256": "b" * 64,
            }
            source_seal_path = root / "qwen.targets.seal.json"
            source_seal_path.write_text(
                json.dumps(source_seal), encoding="utf-8"
            )
            output = root / "imported"
            args = types.SimpleNamespace(
                base_train=train,
                base_train_seal=seal_path,
                contract=contract_path,
                executable_view_report=executable_report,
                expected_executable_view_report_sha256=sha256_file(
                    executable_report
                ),
                provider_key="qwen37_max",
                source_targets=source_targets,
                source_seal=source_seal_path,
                expected_source_seal_sha256=sha256_file(source_seal_path),
                source_report=source_report,
                expected_source_report_sha256=sha256_file(source_report),
                evaluator=evaluator,
                expected_evaluator_sha256=evaluator_sha,
                output_dir=output,
                expected_parent_fit_rows=2776,
                workers=1,
                timeout=30,
                stability_runs=2,
            )
            with mock.patch.object(
                hard_target_importer,
                "validate_executable_view",
                return_value=executable_view,
            ), mock.patch.object(
                hard_target_importer,
                "observe_dart_version",
                return_value="Dart test",
            ):
                manifest = hard_target_importer.import_targets(args)

            repairs_path = output / "verified_repairs.jsonl"
            manifest_path = output / "import_manifest.json"
            imported = json.loads(
                repairs_path.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(imported["code"], code)
            self.assertEqual(imported["provider"], provider)
            self.assertNotIn("reasoning", imported)
            self.assertTrue(
                manifest["invariants"]["fit2776_membership_bound"]
            )
            builder.validate_imported_repair_manifest(
                provider="qwen37_max",
                repair_path=repairs_path,
                report_path=manifest_path,
                report=manifest,
                base_path=train,
                base_seal_path=seal_path,
                contract_path=contract_path,
                executable_view_report=executable_report,
                base_seal=seal,
                executable_view=executable_view,
            )

    def test_imports_native_qwen37_auxiliary_collector_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "compact.contract.json"
            compact_contract = contract_fixture(contract_path)
            train = root / "executable.jsonl"
            write_jsonl(train, [compact_row(compact_contract, "fit-1")])
            evaluator = root / "production_evaluator.py"
            evaluator.write_text(
                "COMPLETION_ATTESTATION_ID = "
                "'per-run-256-bit-marker-exactly-once-v1'\n"
                "DART_BIN = 'dart'\n"
                "def evaluate_dart_jit_tests_detail("
                "code, tests, identity, timeout, stability_runs):\n"
                "    ok = code.startswith('```dart\\nclass Helper')\n"
                "    return ok, ok, 'ok' if ok else 'not-fenced', code\n",
                encoding="utf-8",
            )
            evaluator_sha = sha256_file(evaluator)
            base_seal = {
                "schema": "compact-public-private-join-seal-v1",
                "selected_role": "fit",
                "output_sha256": sha256_file(train),
                "contract_sha256": sha256_file(contract_path),
                "rows": 1,
                "adapter_contract_sha256": "1" * 64,
                "adapter_script_sha256": "2" * 64,
                "source_function_bundles_sha256": "3" * 64,
                "source_symbol_attestation_used": True,
                "source_symbol_attestation_is_keyed": True,
                "source_symbol_attestation_file_sha256": "4" * 64,
                "source_symbol_attestation_key_id_sha256": "5" * 64,
                "raw_source_names_serialized": False,
                "sanitation_schema": "test",
                "sanitizer_sha256": "6" * 64,
                "evaluator_sha256": evaluator_sha,
                "completion_attestation_id": (
                    "per-run-256-bit-marker-exactly-once-v1"
                ),
                "dart_version": "Dart production test",
                "stability_runs": 2,
                "quarantine_sha256": "8" * 64,
            }
            base_seal_path = root / "executable.seal.json"
            base_seal_path.write_text(
                json.dumps(base_seal), encoding="utf-8"
            )

            parent_fit = root / "fit2776.jsonl"
            write_jsonl(
                parent_fit, [compact_row(compact_contract, "fit-1")]
            )
            frozen_split = root / "frozen.split.json"
            frozen_split.write_text(
                json.dumps({"schema": "frozen-test"}), encoding="utf-8"
            )
            parent_seal = root / "fit2776.seal.json"
            parent_seal.write_text(
                json.dumps(
                    {
                        "schema": "compact-public-private-join-seal-v1",
                        "selected_role": "fit",
                        "training_allowed": True,
                        "heldout_measure_only": False,
                        "rows": 2776,
                        "output_sha256": sha256_file(parent_fit),
                        "contract_sha256": sha256_file(frozen_split),
                        "heldout_commitment": {
                            "task_set_sha256": "9" * 64
                        },
                    }
                ),
                encoding="utf-8",
            )
            executable_report = root / "executable.report.json"
            executable_report_value = {
                "parent": {
                    "train": hard_target_importer.file_record(parent_fit),
                    "train_seal": hard_target_importer.file_record(
                        parent_seal
                    ),
                },
                "outputs": {},
            }
            executable_report.write_text(
                json.dumps(executable_report_value), encoding="utf-8"
            )
            executable_view = {
                "rows": 1,
                "parent_rows": 2776,
                "heldout": {"path": "never-opened-heldout.jsonl"},
                "heldout_seal": {"path": "never-opened-heldout.seal.json"},
                "heldout_rows": 175,
                "heldout_task_ids_sha256": "a" * 64,
                "heldout_bytes_opened_during_validation": False,
                "excluded_task_ids": [],
            }

            model = "qwen3.7-max-2026-06-08"
            endpoint = (
                "https://dashscope-intl.aliyuncs.com/"
                "compatible-mode/v1"
            )
            run_contract = {
                "schema": hard_target_importer.QWEN37_RUN_CONTRACT_SCHEMA,
                "exact_pinned_model": model,
                "returned_model_must_equal_requested": True,
                "endpoint": endpoint,
                "inputs": {
                    "fit": hard_target_importer.file_record(parent_fit),
                    "fit_seal": hard_target_importer.file_record(
                        parent_seal
                    ),
                    "frozen_contract": hard_target_importer.file_record(
                        frozen_split
                    ),
                    "heldout_commitment_sha256": "b" * 64,
                    "heldout_artifact_opened": False,
                },
                "transport": {
                    "api": "synchronous_chat_completions",
                    "n": 1,
                    "workers": 1,
                    "one_terminal_logical_draw_per_task": True,
                },
                "verifier": {
                    "implementation_sha256": evaluator_sha,
                    "stability_runs": 2,
                    "completion_attestation": (
                        "per-run-256-bit-marker-exactly-once-v1"
                    ),
                },
                "contamination_contract": {
                    "fit_rows": 2776,
                    "heldout_artifact_opened": False,
                    "tests_in_provider_messages": False,
                    "gold_in_provider_messages": False,
                    "raw_diagnostic_in_provider_messages": False,
                    "compressed_enriched_assembly_in_provider_messages": True,
                    "compressed_cfg_in_provider_messages": True,
                },
                "training_compatibility": {
                    "qwen38_sequence_kl": False,
                    "qwen38_cot": False,
                    "qwen38_union": False,
                    "auxiliary_verified_rs_sft_hard_targets_only": True,
                },
                "budget": {"cap_tokens": 1000},
                "mode": "auxiliary_verified_rs_sft_hard_targets_only",
            }
            run_contract_path = root / "run_contract.json"
            durable_journal.require_exact_or_write(
                run_contract_path, run_contract
            )
            run_contract_digest = durable_journal.canonical_sha256(
                run_contract
            )
            journal_path = root / "attempts.journal.jsonl"
            durable_journal.append_event(
                journal_path,
                {
                    "event": "repair_header",
                    "schema": hard_target_importer.QWEN37_JOURNAL_SCHEMA,
                    "run_contract": hard_target_importer.file_record(
                        run_contract_path
                    ),
                    "run_contract_sha256": run_contract_digest,
                    "model": model,
                    "endpoint": endpoint,
                },
            )
            started = durable_journal.append_event(
                journal_path,
                {
                    "event": "repair_slot_started",
                    "schema": hard_target_importer.QWEN37_JOURNAL_SCHEMA,
                    "task_id": "fit-1",
                    "reservation_tokens": 100,
                    "requested_model": model,
                    "endpoint": endpoint,
                    "candidate": {
                        "task_id": "fit-1",
                        "code_sha256": "f" * 64,
                        "priority_name": "compiled_failed",
                    },
                },
            )
            code = (
                "class Helper { static int add(int x) => x + 1; }\n"
                "int fn0(int x) => Helper.add(x);"
            )
            code_sha = hashlib.sha256(code.encode()).hexdigest()
            tests_sha = hashlib.sha256(
                "assert(fn0(1) == 2);".encode()
            ).hexdigest()
            durable_journal.append_event(
                journal_path,
                {
                    "event": "repair_slot_terminal",
                    "schema": hard_target_importer.QWEN37_JOURNAL_SCHEMA,
                    "task_id": "fit-1",
                    "start_event_sha256": started[
                        "journal_event_sha256"
                    ],
                    "reservation_tokens": 100,
                    "provider_usage": {
                        "prompt_tokens": 6,
                        "completion_tokens": 4,
                        "total_tokens": 10,
                    },
                    "budget_debit_tokens": 10,
                    "requested_model": model,
                    "returned_model": model,
                    "returned_model_matches_requested": True,
                    "endpoint": endpoint,
                    "status": "verified_pass",
                    "system_fingerprint": "qwen-fingerprint",
                    "provider_request_id": "qwen-request-1",
                    "provider_response_sha256": "c" * 64,
                    "raw_content_sha256": "d" * 64,
                    "raw_reasoning_sha256": "e" * 64,
                    "code": code,
                    "code_sha256": code_sha,
                    "verification": {
                        "compiled": True,
                        "passed": True,
                        "harness_completion_attested": True,
                        "completion_attestation": (
                            "per-run-256-bit-marker-exactly-once-v1"
                        ),
                        "tests_sha256": tests_sha,
                        "verifier_sha256": evaluator_sha,
                        "stability_runs": 2,
                    },
                },
            )
            durable_journal.append_event(
                journal_path,
                {
                    "event": "collection_complete",
                    "schema": hard_target_importer.QWEN37_JOURNAL_SCHEMA,
                    "budget_debit_tokens": 10,
                },
            )
            journal_receipt = durable_journal.journal_record(journal_path)
            verified_path = root / "verified_repairs.jsonl"
            write_jsonl(
                verified_path,
                [
                    {
                        "schema": hard_target_importer.QWEN37_OUTPUT_SCHEMA,
                        "task_id": "fit-1",
                        "target": code,
                        "target_sha256": code_sha,
                        "target_mode": "final_dart_code_only",
                        "reasoning_in_target": False,
                        "training_use": (
                            "auxiliary_verified_rs_sft_hard_target_only"
                        ),
                        "source": {
                            "model": model,
                            "endpoint": endpoint,
                            "system_fingerprint": "qwen-fingerprint",
                            "provider_request_id": "qwen-request-1",
                            "provider_response_sha256": "c" * 64,
                            "raw_content_sha256": "d" * 64,
                            "raw_reasoning_sha256": "e" * 64,
                            "failed_candidate_code_sha256": "f" * 64,
                            "priority": "compiled_failed",
                        },
                        "attestation": {
                            "tests_sha256": tests_sha,
                            "verifier_sha256": evaluator_sha,
                            "completion_attestation": (
                                "per-run-256-bit-marker-exactly-once-v1"
                            ),
                            "stability_runs": 2,
                            "passed": True,
                        },
                    }
                ],
            )
            ledger_path = root / "token_ledger.json"
            durable_journal.require_exact_or_write(
                ledger_path,
                {
                    "schema": hard_target_importer.QWEN37_LEDGER_SCHEMA,
                    "model": model,
                    "endpoint": endpoint,
                    "budget_cap_tokens": 1000,
                    "budget_debit_tokens": 10,
                    "provider_reported_actual_tokens": 10,
                    "unknown_usage_slots_charged_at_full_reservation": 0,
                    "remaining_tokens": 990,
                    "logical_draws": 1,
                    "journal": journal_receipt,
                },
            )
            source_report = root / "build_report.json"
            source_report_value = {
                "schema": hard_target_importer.QWEN37_REPORT_SCHEMA,
                "model": model,
                "endpoint": endpoint,
                "run_contract_sha256": run_contract_digest,
                "eligible_failures": 1,
                "logical_draws": 1,
                "verified_repairs": 1,
                "terminal_statuses": {"verified_pass": 1},
                "verified_repairs_artifact": (
                    hard_target_importer.file_record(verified_path)
                ),
                "token_ledger": hard_target_importer.file_record(
                    ledger_path
                ),
                "journal": journal_receipt,
                "contamination_controls": {
                    "fit_rows": 2776,
                    "heldout_artifact_opened": False,
                    "provider_received_tests": False,
                    "provider_received_gold": False,
                    "provider_received_raw_compiler_diagnostics": False,
                    "provider_received_compressed_enriched_assembly": True,
                    "provider_received_compressed_cfg": True,
                },
                "compatibility": {
                    "qwen38_sequence_kl_import_allowed": False,
                    "qwen38_cot_import_allowed": False,
                    "qwen38_union_import_allowed": False,
                    "auxiliary_verified_rs_sft_hard_target_import_allowed": (
                        True
                    ),
                },
            }
            source_report.write_text(
                json.dumps(source_report_value), encoding="utf-8"
            )
            output = root / "imported"
            args = types.SimpleNamespace(
                base_train=train,
                base_train_seal=base_seal_path,
                contract=contract_path,
                executable_view_report=executable_report,
                expected_executable_view_report_sha256=sha256_file(
                    executable_report
                ),
                provider_key="qwen37_snapshot",
                source_targets=verified_path,
                source_seal=run_contract_path,
                expected_source_seal_sha256=sha256_file(
                    run_contract_path
                ),
                source_report=source_report,
                expected_source_report_sha256=sha256_file(source_report),
                evaluator=evaluator,
                expected_evaluator_sha256=evaluator_sha,
                output_dir=output,
                expected_parent_fit_rows=2776,
                workers=1,
                timeout=30,
                stability_runs=2,
            )
            with mock.patch.object(
                hard_target_importer,
                "validate_executable_view",
                return_value=executable_view,
            ), mock.patch.object(
                hard_target_importer,
                "observe_dart_version",
                return_value="Dart production test",
            ):
                manifest = hard_target_importer.import_targets(args)
            imported = json.loads(
                (output / "verified_repairs.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(imported["code"], code)
            self.assertEqual(
                imported["provider"]["requested_model"], model
            )
            self.assertEqual(
                imported["provider_provenance"][
                    "source_journal_chain_head_sha256"
                ],
                journal_receipt["head_event_sha256"],
            )
            self.assertEqual(
                manifest["source_schema"],
                hard_target_importer.QWEN37_REPORT_SCHEMA,
            )


if __name__ == "__main__":
    unittest.main()
