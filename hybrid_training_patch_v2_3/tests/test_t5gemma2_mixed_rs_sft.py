from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluation import t5gemma2_f2_passk_inference as inference
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    journal_record,
)
from scripts.training import t5gemma2_enriched_sft as base_sft
from scripts.training import t5gemma2_mixed_rs_sft as mixed
from scripts.training import t5gemma2_mixed_rs_sft_resume_compat as resume_compat
from scripts.training.seq2seq_verpo_core import build_compiler_repair_context


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def file_record(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": base_sft.sha256_file(path),
        "rows": rows,
    }


def f2_row(task_id: str) -> dict[str, object]:
    text = f"F2\nC0\n// {task_id}\nAx86_64\nX\n"
    return {
        "schema": base_sft.F2_ROW_SCHEMA,
        "representation_schema": base_sft.REPRESENTATION_SCHEMA,
        "task_id": task_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "verified": dict(base_sft._REQUIRED_F2_ATTESTATIONS),
    }


def warmstart() -> mixed.WarmstartIdentity:
    return mixed.WarmstartIdentity(
        checkpoint_name="checkpoint-optstep-000348",
        update=348,
        run_contract_sha256="a" * 64,
        adapter_weights_sha256="b" * 64,
        adapter_config_sha256="c" * 64,
        model="google/t5gemma-2-4b-4b",
        model_revision="d" * 40,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.05,
        exact_lora_targets=("model.encoder.text_model.layers.0.q_proj",),
    )


def make_gold_inputs(
    root: Path, *, heldout_task_id: str = "heldout-1"
) -> tuple[Path, Path, Path]:
    task_ids = ["local-1", "api-1", "gold-1", "gold-2", "gold-3", "gold-4"]
    gold = root / "gold.jsonl"
    f2 = root / "gold_f2.jsonl"
    heldout = root / "heldout.jsonl"
    write_jsonl(
        gold,
        [
            {
                "task_id": task_id,
                "dart_source": f"int fn0(int x) => x + {index};",
                # These fields prove that the loader selects only the target
                # and never serializes test text into the encoder source.
                "acceptance_tests": f"assert(fn0(1) == {index + 1});",
            }
            for index, task_id in enumerate(task_ids)
        ],
    )
    write_jsonl(f2, [f2_row(task_id) for task_id in task_ids])
    write_jsonl(heldout, [{"task_id": heldout_task_id}])
    return gold, f2, heldout


def make_local_report(
    root: Path,
    identity: mixed.WarmstartIdentity,
    *,
    adapter_weights_sha256: str | None = None,
) -> Path:
    directory = root / "local"
    directory.mkdir()
    targets = directory / "rs_sft_repairs.jsonl"
    f2 = directory / "rs_sft_repairs_f2.jsonl"
    write_jsonl(
        targets,
        [{"task_id": "local-1", "dart_source": "int fn0(int x) => x + 7;"}],
    )
    write_jsonl(f2, [f2_row("local-1")])
    run_contract_sha = "1" * 64
    journal = directory / "harvest.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": "t5gemma2-local-rs-sft-pilot-journal-v1",
            "contract_sha256": run_contract_sha,
        },
    )
    code = "int fn0(int x) => x + 7;"
    selected_source = base_sft.build_encoder_source(f2_row("local-1"), "local-1")
    append_event(
        journal,
        {
            "event": "task_terminal",
            "schema": "t5gemma2-local-rs-sft-pilot-journal-v1",
            "task_id": "local-1",
            "selected_target": {
                "schema": "t5gemma2-local-rs-sft-target-v1",
                "task_id": "local-1",
                "code": code,
                "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                "source_sha256": hashlib.sha256(
                    selected_source.encode("utf-8")
                ).hexdigest(),
                "visible_passed": True,
                "private_gate_passed": True,
            },
        },
    )
    append_event(
        journal,
        {
            "event": "complete",
            "schema": "t5gemma2-local-rs-sft-pilot-journal-v1",
            "tasks": 1,
        },
    )
    report = {
        "schema": mixed.LOCAL_REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": run_contract_sha,
        "checkpoint": {
            "name": identity.model,
            "revision": identity.model_revision,
            "adapter": {
                "run_contract_sha256": identity.run_contract_sha256,
                "adapter_weights_sha256": (
                    adapter_weights_sha256 or identity.adapter_weights_sha256
                ),
                "adapter_config_sha256": identity.adapter_config_sha256,
            },
        },
        "pilot": {
            "tasks": 1,
            "accepted_unique_targets": 1,
            "production_floor_met": True,
        },
        "outputs": {
            "repairs": file_record(targets, 1),
            "repairs_f2": file_record(f2, 1),
        },
        "privacy_invariants": {
            "heldout_175_opened": False,
            "frontier_api_calls": False,
            "private_holdback_text_in_model_input": False,
            "private_holdback_text_in_outputs": False,
            "private_diagnostics_persisted": False,
            "all_generation_precedes_private_gate_per_task": True,
            "private_gate_can_only_reject_transfer": True,
        },
        "journal": journal_record(journal),
    }
    path = directory / "harvest_report.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def make_api_report(
    root: Path,
    *,
    exploratory: bool,
    tests_present: bool = False,
) -> Path:
    directory = root / "api"
    directory.mkdir()
    task_id = "api-1"
    f2 = f2_row(task_id)
    original_source = base_sft.build_encoder_source(f2, task_id)
    target_code = "int fn0(int x) => x * 2;"
    candidate = "int fn0(String x) => x.length;"
    diagnostic = "test.dart:1: Error: int expected"
    context = build_compiler_repair_context(
        task_id=task_id,
        source_sha256=hashlib.sha256(original_source.encode("utf-8")).hexdigest(),
        candidate=candidate,
        diagnostic=diagnostic,
        compiled=False,
    )
    encoder_source = original_source + "\n" + str(context["text"])
    safe_diagnostic = str(context["payload"]["compiler_feedback"])
    run_contract_sha = "e" * 64
    journal = directory / "api_rescue.journal.jsonl"
    append_event(
        journal,
        {
            "event": "header",
            "schema": "t5gemma2-api-rs-sft-rescue-journal-v1",
            "contract_sha256": run_contract_sha,
        },
    )
    append_event(
        journal,
        {
            "event": "task_verification",
            "schema": "t5gemma2-api-rs-sft-rescue-journal-v1",
            "task_id": task_id,
            "source_sha256": hashlib.sha256(
                original_source.encode("utf-8")
            ).hexdigest(),
            "all_api_generation_completed_before_private_gate": True,
            "holdback_failure_triggers_generation": False,
            "private_diagnostics_persisted": False,
            "private_feedback_serialized_to_model": False,
            "selected_target": {
                "schema": mixed.API_DIRECT_TARGET_SCHEMA,
                "task_id": task_id,
                "code": target_code,
                "code_sha256": hashlib.sha256(target_code.encode("utf-8")).hexdigest(),
                "source_sha256": hashlib.sha256(
                    original_source.encode("utf-8")
                ).hexdigest(),
                "feedback_source_sha256": hashlib.sha256(
                    encoder_source.encode("utf-8")
                ).hexdigest(),
                "parent_code_sha256": hashlib.sha256(
                    candidate.encode("utf-8")
                ).hexdigest(),
                "diagnostic_sha256": hashlib.sha256(
                    safe_diagnostic.encode("utf-8")
                ).hexdigest(),
                "visible_passed": True,
                "private_gate_passed": True,
                "exploratory_prefix": exploratory,
                "production_floor_eligible": not exploratory,
            },
        },
    )
    append_event(
        journal,
        {
            "event": "complete",
            "schema": "t5gemma2-api-rs-sft-rescue-journal-v1",
            "verified_targets": 1,
            "exploratory_prefix": exploratory,
            "production_floor_eligible": not exploratory,
        },
    )
    direct_targets = directory / "direct_hard_targets.jsonl"
    direct_f2 = directory / "direct_hard_targets_f2.jsonl"
    repair_sources = directory / "repair_policy_sources.jsonl"
    repair_targets = directory / "repair_policy_targets.jsonl"
    repair_id = f"{task_id}::api-rescue::000000"
    write_jsonl(
        direct_targets,
        [
            {
                "schema": mixed.API_DIRECT_TARGET_SCHEMA,
                "task_id": task_id,
                "dart_source": target_code,
                "dart_source_sha256": hashlib.sha256(
                    target_code.encode("utf-8")
                ).hexdigest(),
                "source_sha256": hashlib.sha256(
                    original_source.encode("utf-8")
                ).hexdigest(),
                "visible_passed": True,
                "private_gate_passed": True,
                "exploratory_prefix": exploratory,
                "production_floor_eligible": not exploratory,
                "provenance": {
                    "run_contract_sha256": run_contract_sha,
                    "slot_position": 0,
                    "parent_code_sha256": hashlib.sha256(
                        candidate.encode("utf-8")
                    ).hexdigest(),
                    "diagnostic_sha256": hashlib.sha256(
                        safe_diagnostic.encode("utf-8")
                    ).hexdigest(),
                },
            }
        ],
    )
    write_jsonl(direct_f2, [f2])
    write_jsonl(
        repair_sources,
        [
            {
                "schema": mixed.API_REPAIR_PAIR_SCHEMA,
                "task_id": repair_id,
                "source_task_id": task_id,
                "encoder_source": encoder_source,
                "encoder_source_sha256": hashlib.sha256(
                    encoder_source.encode("utf-8")
                ).hexdigest(),
                "original_f2_source_sha256": hashlib.sha256(
                    original_source.encode("utf-8")
                ).hexdigest(),
                "parent_code_sha256": hashlib.sha256(
                    candidate.encode("utf-8")
                ).hexdigest(),
                "compiler_diagnostic": safe_diagnostic,
                "compiler_diagnostic_sha256": hashlib.sha256(
                    safe_diagnostic.encode("utf-8")
                ).hexdigest(),
                "private_feedback_present": False,
                "tests_present": tests_present,
                "gold_present": False,
                "exploratory_prefix": exploratory,
                "production_floor_eligible": not exploratory,
            }
        ],
    )
    write_jsonl(
        repair_targets,
        [
            {
                "schema": mixed.API_REPAIR_PAIR_SCHEMA,
                "task_id": repair_id,
                "source_task_id": task_id,
                "dart_source": target_code,
                "dart_source_sha256": hashlib.sha256(
                    target_code.encode("utf-8")
                ).hexdigest(),
                "exploratory_prefix": exploratory,
                "production_floor_eligible": not exploratory,
            }
        ],
    )
    direct_manifest = {
        "schema": mixed.API_DIRECT_MANIFEST_SCHEMA,
        "run_contract_sha256": run_contract_sha,
        "rows": 1,
        "targets": file_record(direct_targets, 1),
        "f2": file_record(direct_f2, 1),
        "mapping": "original_sealed_F2_to_visible_and_private_verified_Dart",
        "compatible_trainer": "t5gemma2_enriched_sft.py",
        "unique_source_tasks": True,
        "exploratory_prefix": exploratory,
        "production_floor_eligible": not exploratory,
        "may_count_toward_production_min_unique_targets": not exploratory,
    }
    repair_manifest = {
        "schema": mixed.API_REPAIR_MANIFEST_SCHEMA,
        "run_contract_sha256": run_contract_sha,
        "rows": 1,
        "targets": file_record(repair_targets, 1),
        "prebuilt_encoder_sources": file_record(repair_sources, 1),
        "mapping": (
            "exact_original_F2_plus_failed_candidate_plus_sanitized_compiler_"
            "diagnostic_to_the_same_visible_and_private_verified_Dart"
        ),
        "source_is_exact_model_input": True,
        "requires_prebuilt_encoder_source_loader": True,
        "private_feedback_present": False,
        "exploratory_prefix": exploratory,
        "production_floor_eligible": not exploratory,
        "may_count_toward_production_min_unique_targets": not exploratory,
    }
    (directory / "direct_manifest.json").write_text(
        json.dumps(direct_manifest, sort_keys=True), encoding="utf-8"
    )
    (directory / "repair_policy_manifest.json").write_text(
        json.dumps(repair_manifest, sort_keys=True), encoding="utf-8"
    )
    report = {
        "schema": mixed.API_REPORT_SCHEMA,
        "status": "complete",
        "run_contract_sha256": run_contract_sha,
        "heldout_175_opened": False,
        "exploratory_prefix": exploratory,
        "production_floor_eligible": not exploratory,
        "may_count_toward_production_min_unique_targets": not exploratory,
        "local_source": {
            "mode": (
                "exploratory_terminal_prefix" if exploratory else "completed_local_run"
            ),
            "source_journal_modified": False,
            "exploratory_prefix": exploratory,
            "production_floor_eligible": not exploratory,
        },
        "privacy_invariants": {
            "api_credentials_persisted": False,
            "gold_sent_to_provider": False,
            "plaintext_reasoning_persisted": False,
            "private_holdback_sent_to_provider": False,
            "visible_training_tests_in_training_outputs": False,
            "visible_training_tests_sent_to_provider": True,
            "api_input_fields": [
                "original_test_free_F2",
                "failed_student_code",
                "sanitized_compiler_diagnostic",
                "visible_training_tests_provider_only",
            ],
        },
        "direct_manifest": direct_manifest,
        "repair_policy_manifest": repair_manifest,
        "verification": {"verified_unique_hard_targets": 1},
        "provider": {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "credential_source": "environment_value_not_persisted",
        },
        "journal": journal_record(journal),
    }
    path = directory / "api_rescue_report.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def build_fixture(
    root: Path,
    *,
    exploratory: bool = False,
    allow_exploratory: bool = False,
    tests_present: bool = False,
    heldout_task_id: str = "heldout-1",
    bad_local_warmstart: bool = False,
) -> tuple[list[mixed.MixedPair], dict[str, object]]:
    identity = warmstart()
    gold, f2, heldout = make_gold_inputs(root, heldout_task_id=heldout_task_id)
    local_report = make_local_report(
        root,
        identity,
        adapter_weights_sha256=("f" * 64 if bad_local_warmstart else None),
    )
    api_report = make_api_report(
        root,
        exploratory=exploratory,
        tests_present=tests_present,
    )
    return mixed.build_mixed_pairs(
        gold_train_jsonl=gold,
        gold_f2_jsonl=f2,
        expected_gold_train_sha256=base_sft.sha256_file(gold),
        expected_gold_f2_sha256=base_sft.sha256_file(f2),
        expected_gold_rows=6,
        heldout_jsonl=heldout,
        expected_heldout_sha256=base_sft.sha256_file(heldout),
        expected_heldout_rows=1,
        local_reports=[(local_report, base_sft.sha256_file(local_report))],
        api_reports=[(api_report, base_sft.sha256_file(api_report))],
        warmstart=identity,
        gold_replay_ratio=1.0,
        gold_replay_rows=-1,
        min_verified_direct_targets=2,
        min_repair_conditioned_targets=1,
        allow_exploratory_inputs=allow_exploratory,
        require_local_production_floor=True,
        seed=42,
    )


class MixedRsSftTests(unittest.TestCase):
    def test_builds_direct_repair_and_gold_replay_without_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pairs, manifest = build_fixture(Path(directory))
        self.assertEqual(
            manifest["composition"],
            {
                "verified_direct": 2,
                "repair_conditioned": 1,
                "gold_replay": 3,
            },
        )
        self.assertEqual(len(pairs), 6)
        self.assertEqual(manifest["heldout_overlap"], 0)
        self.assertFalse(manifest["tests_model_visible"])
        self.assertTrue(manifest["production_floor_eligible"])
        repair = next(pair for pair in pairs if pair.kind == "repair_conditioned")
        self.assertIn("COMPILER_REPAIR_CONTEXT_JSON", repair.source)
        self.assertNotIn("assert(fn0", repair.source)

    def test_exploratory_api_prefix_is_explicit_and_taints_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exploratory API prefix"):
                build_fixture(Path(directory), exploratory=True)
        with tempfile.TemporaryDirectory() as directory:
            _pairs, manifest = build_fixture(
                Path(directory),
                exploratory=True,
                allow_exploratory=True,
            )
        self.assertTrue(manifest["exploratory_inputs"])
        self.assertFalse(manifest["production_floor_eligible"])

    def test_repair_source_with_tests_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "API repair row"):
                build_fixture(Path(directory), tests_present=True)

    def test_any_train_heldout_overlap_fails_before_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "overlaps held-out"):
                build_fixture(Path(directory), heldout_task_id="local-1")

    def test_local_report_must_bind_exact_warmstart_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "warm-start binding"):
                build_fixture(Path(directory), bad_local_warmstart=True)

    def test_conflicting_targets_for_one_encoder_input_fail_closed(self) -> None:
        first = mixed._make_pair(
            pair_id="task::first",
            source_task_id="task",
            kind="verified_direct",
            source="sealed source",
            target="void f() {}",
            provenance=(),
        )
        second = mixed._make_pair(
            pair_id="task::second",
            source_task_id="task",
            kind="verified_direct",
            source="sealed source",
            target="void f() { return; }",
            provenance=(),
        )
        with self.assertRaisesRegex(ValueError, "conflicting verified targets"):
            mixed._deduplicate_pairs([first, second])

    def test_schedule_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = warmstart()
            gold, f2, heldout = make_gold_inputs(root)
            local_report = make_local_report(root, identity)
            api_report = make_api_report(root, exploratory=False)
            kwargs = {
                "gold_train_jsonl": gold,
                "gold_f2_jsonl": f2,
                "expected_gold_train_sha256": base_sft.sha256_file(gold),
                "expected_gold_f2_sha256": base_sft.sha256_file(f2),
                "expected_gold_rows": 6,
                "heldout_jsonl": heldout,
                "expected_heldout_sha256": base_sft.sha256_file(heldout),
                "expected_heldout_rows": 1,
                "local_reports": [(local_report, base_sft.sha256_file(local_report))],
                "api_reports": [(api_report, base_sft.sha256_file(api_report))],
                "warmstart": identity,
                "gold_replay_ratio": 1.0,
                "gold_replay_rows": -1,
                "min_verified_direct_targets": 2,
                "min_repair_conditioned_targets": 1,
                "allow_exploratory_inputs": False,
                "require_local_production_floor": True,
                "seed": 42,
            }
            first, first_manifest = mixed.build_mixed_pairs(**kwargs)
            second, second_manifest = mixed.build_mixed_pairs(**kwargs)
        self.assertEqual(
            [pair.pair_id for pair in first],
            [pair.pair_id for pair in second],
        )
        self.assertEqual(
            first_manifest["schedule_sha256"],
            second_manifest["schedule_sha256"],
        )

    def test_mixed_checkpoint_is_valid_adapter_only_parent_warmstart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-optstep-000426"
            adapter = checkpoint / "adapter"
            tokenizer = checkpoint / "tokenizer"
            adapter.mkdir(parents=True)
            tokenizer.mkdir()
            target = "model.decoder.layers.0.self_attn.q_proj"
            contract = {
                "schema": mixed.RUN_SCHEMA,
                "status": "training",
                "architecture": "native_encoder_decoder",
                "model": "google/t5gemma-2-4b-4b",
                "model_revision": "d" * 40,
                "base_model": {"is_encoder_decoder": True},
                "dataset": {
                    "schema": mixed.DATASET_SCHEMA,
                    "heldout_overlap": 0,
                },
                "tokenizer": {"class": "GemmaTokenizer"},
                "privacy": {
                    "heldout_overlap": 0,
                    "heldout_content_model_visible": False,
                    "tests_model_visible": False,
                    "private_feedback_model_visible": False,
                },
                "lora": {
                    "rank": 64,
                    "alpha": 128,
                    "dropout": 0.05,
                    "targets": [target],
                    "encoder_and_decoder_trainable": True,
                    "vision_trainable": False,
                    "new_adapter_attached": False,
                    "warmstart_weights_continued": True,
                },
            }
            (checkpoint / "run_contract.json").write_text(
                json.dumps(contract, sort_keys=True), encoding="utf-8"
            )
            mixed.torch.save(
                {
                    "schema": mixed.CHECKPOINT_SCHEMA,
                    "update": 426,
                    "run_contract_sha256": mixed._canonical_sha256(contract),
                    "optimizer": {},
                    "scheduler": {},
                    "rng": {},
                },
                checkpoint / "training_state.pt",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"sealed adapter")
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "r": 64,
                        "lora_alpha": 128,
                        "lora_dropout": 0.05,
                        "task_type": "SEQ_2_SEQ_LM",
                        "target_modules": [target],
                    }
                ),
                encoding="utf-8",
            )
            (tokenizer / "tokenizer_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with mock.patch.object(
                base_sft,
                "_adapter_weight_target_modules",
                return_value={target},
            ):
                identity, saved_contract = mixed.validate_warmstart(
                    checkpoint,
                    expected_update=426,
                    expected_run_contract_sha256=mixed._canonical_sha256(contract),
                    expected_adapter_weights_sha256=base_sft.sha256_file(
                        adapter / "adapter_model.safetensors"
                    ),
                    expected_adapter_config_sha256=base_sft.sha256_file(
                        adapter / "adapter_config.json"
                    ),
                    model="google/t5gemma-2-4b-4b",
                    model_revision="d" * 40,
                )
        self.assertEqual(identity.update, 426)
        self.assertEqual(identity.exact_lora_targets, (target,))
        self.assertEqual(saved_contract["schema"], mixed.RUN_SCHEMA)

    def test_resume_contract_accepts_exact_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-optstep-000021"
            (checkpoint / "adapter").mkdir(parents=True)
            (checkpoint / "tokenizer").mkdir()
            (checkpoint / "adapter" / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            (checkpoint / "tokenizer" / "tokenizer_config.json").write_text(
                "{}", encoding="utf-8"
            )
            run_contract = {
                "schema": mixed.RUN_SCHEMA,
                "warmstart": {
                    "exact_lora_targets": ("encoder.q_proj", "decoder.q_proj")
                },
            }
            (checkpoint / "run_contract.json").write_text(
                json.dumps(run_contract, sort_keys=True), encoding="utf-8"
            )
            mixed.torch.save(
                {
                    "schema": mixed.CHECKPOINT_SCHEMA,
                    "update": 21,
                    "run_contract_sha256": mixed._canonical_sha256(run_contract),
                    "optimizer": {},
                    "scheduler": {},
                    "rng": {},
                },
                checkpoint / "training_state.pt",
            )
            targets = ("encoder.q_proj", "decoder.q_proj")
            with mock.patch.object(
                base_sft,
                "_adapter_weight_target_modules",
                return_value=set(targets),
            ):
                state = mixed._load_stage_checkpoint(
                    checkpoint,
                    run_contract=run_contract,
                    exact_targets=targets,
                )
                self.assertEqual(state["update"], 21)

                changed_contract = dict(run_contract)
                changed_contract["schema"] = "different"
                with self.assertRaisesRegex(
                    ValueError, "resume run contract differs"
                ):
                    mixed._load_stage_checkpoint(
                        checkpoint,
                        run_contract=changed_contract,
                        exact_targets=targets,
                    )

    def test_legacy_resume_wrapper_normalizes_only_warmstart_targets(self) -> None:
        identity = warmstart()
        converted = resume_compat._json_compatible_asdict(identity)
        self.assertEqual(
            converted["exact_lora_targets"],
            list(identity.exact_lora_targets),
        )
        self.assertIsInstance(converted["exact_lora_targets"], list)

    def test_legacy_resume_wrapper_retains_latest_two_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory).resolve()
            checkpoints = [
                output_dir / f"checkpoint-optstep-{update:06d}"
                for update in (5, 10, 15)
            ]
            for checkpoint in checkpoints:
                checkpoint.mkdir()
                (checkpoint / "sentinel").write_text("sealed", encoding="utf-8")
            (output_dir / "latest_checkpoint.json").write_text(
                json.dumps({"path": str(checkpoints[-1])}),
                encoding="utf-8",
            )
            removed = resume_compat._prune_superseded_checkpoints(
                output_dir,
                checkpoints[-1],
            )
            self.assertEqual(removed, ["checkpoint-optstep-000005"])
            self.assertFalse(checkpoints[0].exists())
            self.assertTrue(checkpoints[1].exists())
            self.assertTrue(checkpoints[2].exists())


class MixedRsSftDeployTests(unittest.TestCase):
    def test_launcher_targets_two_epoch_native_checkpoint(self) -> None:
        project = Path(__file__).resolve().parents[1]
        launcher = (project / "deploy" / "vast" / "t5gemma2_mixed_rs_sft.sh").read_text(
            encoding="utf-8"
        )
        config = (project / "deploy" / "vast" / "t5gemma2-mixed-rs-sft.conf").read_text(
            encoding="utf-8"
        )
        self.assertIn("checkpoint-optstep-000348", launcher)
        self.assertIn("t5gemma2_mixed_rs_sft.py", launcher)
        self.assertIn("--resume_checkpoint", launcher)
        self.assertIn("t5gemma2_mixed_rs_sft_resume_compat.py", launcher)
        self.assertIn("--allow_exploratory_inputs", launcher)
        self.assertNotIn("graph", launcher.lower())
        self.assertIn("mixed_rs_sft_exploratory_v1", config)
        self.assertIn('ALLOW_EXPLORATORY_INPUTS="1"', config)

    def test_launcher_validates_both_minimum_target_counts(self) -> None:
        project = Path(__file__).resolve().parents[1]
        launcher = (
            project / "deploy" / "vast" / "t5gemma2_mixed_rs_sft.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'if ! [[ "${MIN_DIRECT_TARGETS}" =~ ^[0-9]+$ ]] \\\n'
            '  || ! [[ "${MIN_REPAIR_TARGETS}" =~ ^[0-9]+$ ]]; then',
            launcher,
        )
        self.assertNotIn(
            '"${MIN_DIRECT_TARGETS}" =~ ^[0-9]+$ || '
            '"${MIN_REPAIR_TARGETS}" =~ ^[0-9]+$',
            launcher,
        )

    def test_kimi_pass2_is_api_only_and_adapter_only_parent_continuation(self) -> None:
        project = Path(__file__).resolve().parents[1]
        generic = (
            project / "deploy" / "vast" / "t5gemma2_mixed_rs_sft.sh"
        ).read_text(encoding="utf-8")
        launcher = (
            project
            / "deploy"
            / "vast"
            / "t5gemma2_mixed_rs_sft_kimi_pass2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ -n "${LOCAL_REPORT_SPECS}" ]]; then', generic)
        self.assertIn('if [[ -n "${API_REPORT_SPECS}" ]]; then', generic)
        self.assertIn("checkpoint-optstep-000426", launcher)
        self.assertIn("T5GEMMA_MIXED_LOCAL_REPORT_SPECS=", launcher)
        self.assertIn("T5GEMMA_MIXED_MIN_DIRECT_TARGETS=13", launcher)
        self.assertIn("T5GEMMA_MIXED_MIN_REPAIR_TARGETS=13", launcher)
        self.assertIn("T5GEMMA_MIXED_EPOCHS=1", launcher)
        self.assertIn("T5GEMMA_MIXED_LEARNING_RATE=1e-5", launcher)
        self.assertNotIn("RESUME_COMPAT=1", launcher)

    def test_heldout_inference_accepts_sealed_mixed_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-optstep-000021"
            (checkpoint / "adapter").mkdir(parents=True)
            (checkpoint / "tokenizer").mkdir()
            (checkpoint / "adapter" / "adapter_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(
                b"adapter"
            )
            (checkpoint / "tokenizer" / "tokenizer.json").write_text(
                "{}\n", encoding="utf-8"
            )
            contract = {
                "schema": mixed.RUN_SCHEMA,
                "status": "training",
                "architecture": "native_encoder_decoder",
                "base_model": {
                    "name": inference.MODEL_NAME,
                    "resolved_commit": inference.MODEL_REVISION,
                    "is_encoder_decoder": True,
                    "config_sha256": "f" * 64,
                },
                "lora": {"targets": ["model.decoder.layers.0.q_proj"]},
                "dataset": {
                    "schema": mixed.DATASET_SCHEMA,
                    "heldout_overlap": 0,
                },
                "privacy": {
                    "heldout_overlap": 0,
                    "heldout_content_model_visible": False,
                    "tests_model_visible": False,
                    "private_feedback_model_visible": False,
                },
                "production_floor_eligible": False,
            }
            (checkpoint / "run_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            with mock.patch.object(
                inference,
                "_adapter_weight_target_modules",
                return_value={"model.decoder.layers.0.q_proj"},
            ):
                _contract, record = inference._checkpoint_record(checkpoint, "sft")
        self.assertEqual(record["training_stage_schema"], mixed.RUN_SCHEMA)
        self.assertFalse(record["production_floor_eligible"])


if __name__ == "__main__":
    unittest.main()
