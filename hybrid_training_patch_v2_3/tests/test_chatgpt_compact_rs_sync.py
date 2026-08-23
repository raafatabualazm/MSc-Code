from __future__ import annotations

import ast
import hashlib
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from models.direct_compact_causal import sha256_artifact
from scripts.training.collect_chatgpt_compact_rs import (
    QWEN_AUDIT_SCHEMA,
    QWEN_BUILD_SCHEMA,
    QWEN_COT_BUILD_SCHEMA,
    QWEN_COT_PROMPT_MODE,
    QWEN_COT_SCHEDULE_SCHEMA,
    QWEN_COT_THINK_CLOSE_ID,
    QWEN_COT_THINK_OPEN_ID,
    QWEN_STAGE_MODEL,
    build_repair_prompt,
    escalated_output_token_budget,
    ensure_run_contract,
    file_record,
    run_sample_major_bounded,
    sha256_file,
    stable_sha256,
    validate_capacity_only_contracts,
    validate_openai_base_url,
    validate_qwen_student_checkpoint,
)


class OpenAISyncRSCollectorTests(unittest.TestCase):
    class _CharTokenizer:
        class _Encoding:
            def __init__(self, count: int) -> None:
                self.ids = list(range(count))

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return self._Encoding(len(text))

    def _qwen_checkpoint(self, root: Path) -> tuple[Path, Path]:
        stage = root / "qwen_stage"
        checkpoint = stage / "direct_compact_qwen_sequence_warmstart"
        adapter = checkpoint / "decoder_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        overlay = checkpoint / "source_embedding_overlay.pt"
        overlay.write_bytes(b"overlay")
        contract = checkpoint / "compact_contract.json"
        contract_value = {
            "schema": "direct-compact-causal-v1",
            "base_vocab_size": 100,
            "codec_sha256": "a" * 64,
            "codebook_sha256": "b" * 64,
            "max_source_tokens": 9000,
            "max_target_tokens": 24576,
            "max_total_tokens": 36864,
            "source_token_expansions": {
                "100": [1, 2],
                "101": [3, 4],
            },
        }
        contract.write_text(
            json.dumps(contract_value, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        train = stage / "qwen_mc_sequence_train.jsonl"
        schedule = stage / "qwen_mc_sequence_train.schedule.jsonl"
        teacher_schedule: list[dict[str, object]] = []
        train_rows: list[dict[str, object]] = []
        position = 0
        for task_index in range(1580):
            for sample_index in range(8):
                teacher_schedule.append(
                    {
                        "schema": (
                            "direct-compact-mc-sequence-forward-kl-nll-"
                            "schedule-v1"
                        ),
                        "position": position,
                        "kind": "teacher_draw",
                        "task_id": f"t{task_index}",
                        "candidate_id": f"candidate-{task_index}-{sample_index}",
                        "sample_index": sample_index,
                        "base_row_index": task_index,
                        "compact_ids_sha256": "1" * 64,
                        "target_sha256": "2" * 64,
                        "draw_weight": 1.0,
                    }
                )
                train_rows.append({"task_id": f"t{task_index}"})
                position += 1
        train.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in train_rows
            ),
            encoding="utf-8",
        )
        schedule.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in teacher_schedule
            ),
            encoding="utf-8",
        )
        train_seal = stage / "qwen_mc_sequence_train.seal.json"
        train_seal.write_text(
            json.dumps(
                {
                    "schema": "compact-public-private-join-seal-v1",
                    "selected_role": "fit",
                    "rows": 12640,
                    "output_sha256": sha256_file(train),
                    "output_size_bytes": train.stat().st_size,
                    "contract_sha256": sha256_file(contract),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        compact_train = stage / "train_multifunction_binary.jsonl"
        compact_train.write_text(
            "".join(
                json.dumps(
                    {
                        "task_id": f"t{index}",
                        "binary_multifunction_schema": (
                            "binary-multifunction-v1-semantic-adapter-v1"
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
                for index in range(1580)
            ),
            encoding="utf-8",
        )
        compact_train_seal = stage / "train_multifunction_binary.seal.json"
        compact_train_seal.write_text(
            json.dumps(
                {
                    "schema": "compact-public-private-join-seal-v1",
                    "selected_role": "fit",
                    "rows": 1580,
                    "output_sha256": sha256_file(compact_train),
                    "output_size_bytes": compact_train.stat().st_size,
                    "contract_sha256": sha256_file(contract),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        gold = stage / "direct_compact_multifunction_gold_sft"
        gold_adapter = gold / "decoder_adapter"
        gold_adapter.mkdir(parents=True)
        (gold_adapter / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (gold_adapter / "adapter_model.safetensors").write_bytes(b"gold")
        gold_overlay = gold / "source_embedding_overlay.pt"
        gold_overlay.write_bytes(b"gold-overlay")
        gold_contract = gold / "compact_contract.json"
        gold_contract.write_text(contract.read_text(encoding="utf-8"), encoding="utf-8")
        gold_provenance = gold / "run_provenance.json"
        gold_provenance.write_text(
            json.dumps(
                {
                    "schema": "direct-compact-run-provenance-v1",
                    "architecture": "qwen-causal-compact-tokens-no-encoder",
                    "decoder_adapter_sha256": sha256_artifact(gold_adapter),
                    "source_overlay_sha256": sha256_file(gold_overlay),
                    "contract_sha256": sha256_file(gold_contract),
                    "loss_contract": {
                        "sequence_distribution_nll": False,
                        "primary_reduction": "base_causal_lm_token_mean",
                    },
                    "train_file_sha256": sha256_file(compact_train),
                    "train_seal_sha256": sha256_file(compact_train_seal),
                    "train_sealed_rows": 1580,
                    "heldout_loaded_during_training": False,
                    "eval_file_sha256": None,
                    "eval_seal_sha256": None,
                    "eval_sealed_rows": None,
                    "eval_strategy": "no",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        prompts = stage / "train_multifunction_binary_f2.jsonl"
        prompts.write_text(
            "".join(
                json.dumps({"task_id": f"t{index}", "text": "F2\n"})
                + "\n"
                for index in range(1580)
            ),
            encoding="utf-8",
        )
        prompt_manifest = stage / (
            "train_multifunction_binary_f2.jsonl.manifest.json"
        )
        prompt_manifest.write_text(
            json.dumps(
                {
                    "schema": "verified-api-readable-compact-v2",
                    "rows": 1580,
                    "dataset": file_record(compact_train),
                    "output": file_record(prompts),
                    "f2_prompt_contract": {
                        "representation_schema": "lossless-semantic-f2",
                        "system_prompt_sha256": "c" * 64,
                    },
                    "invariants": {
                        "all_user_functions_retained": True,
                        "all_external_symbols_retained": True,
                        "transfer_table_redundancy_proven": True,
                        "train_dev_representation_contract_identical": True,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        audit = stage / "qwen_teacher.audit.json"
        teacher_parseable = stage / "qwen_teacher.sequence.jsonl"
        teacher_parseable.write_text(
            "".join(
                json.dumps(
                    {
                        "candidate_id": (
                            f"candidate-{task_index}-{sample_index}"
                        ),
                        "task_id": f"t{task_index}",
                        "sample_index": sample_index,
                    },
                    sort_keys=True,
                )
                + "\n"
                for task_index in range(1580)
                for sample_index in range(8)
            ),
            encoding="utf-8",
        )
        audit.write_text(
            json.dumps(
                {
                    "schema": QWEN_AUDIT_SCHEMA,
                    "objective_mode": "sequence_only",
                    "production_readiness": {
                        "mc_sequence_forward_kl_nll": True
                    },
                    "capabilities": {
                        "dense_full_vocabulary_kl": False
                    },
                    "coverage": {
                        "candidates": 12640,
                        "sequence_candidates": 12640,
                    },
                    "homogeneous_backend_shards": [
                        {
                            "backend_identity": {
                                "requested_model": QWEN_STAGE_MODEL,
                                "returned_model": QWEN_STAGE_MODEL,
                                "system_fingerprint": "sealed",
                            }
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        build = stage / "qwen_mc_sequence_train.build.json"
        build.write_text(
            json.dumps(
                {
                    "schema": QWEN_BUILD_SCHEMA,
                    "objective": {
                        "name": "monte_carlo_sequence_forward_kl_nll",
                        "every_teacher_draw_emitted_exactly_once": True,
                        "all_k8_draws_required_and_emitted": True,
                        "parseability_filtering": False,
                        "correctness_filtering": False,
                        "gold_targets_mixed_into_sequence_objective": False,
                        "target_transform": "trim_outer_whitespace",
                        "objective_mode": "sequence_only",
                        "teacher_sampling": {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": 101,
                            "tempered": False,
                            "truncated": False,
                        },
                        "dense_full_vocabulary_kl": False,
                    },
                    "gold_replay": {
                        "requested_final_fraction": 0.0,
                        "realized_final_fraction": 0.0,
                        "rows": 0,
                        "required_zero_for_sequence_only": True,
                    },
                    "counts": {
                        "teacher_draw_rows": 12640,
                        "gold_replay_rows": 0,
                        "output_rows": 12640,
                        "unique_teacher_candidate_ids": 12640,
                    },
                    "inputs": {
                        "teacher_audit": file_record(audit),
                        "teacher_parseable": file_record(teacher_parseable),
                        "compact_train": file_record(compact_train),
                        "compact_train_seal": file_record(compact_train_seal),
                        "contract": file_record(contract),
                        "prompt_artifact": file_record(prompts),
                        "prompt_manifest": file_record(prompt_manifest),
                        "f2_prompt_contract": {
                            "representation_schema": "lossless-semantic-f2",
                            "system_prompt_sha256": "c" * 64,
                        },
                    },
                    "outputs": {
                        "dataset": file_record(train),
                        "standard_direct_compact_seal": file_record(train_seal),
                        "schedule": file_record(schedule),
                    },
                    "schedule_sha256": stable_sha256(teacher_schedule),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = checkpoint / "run_provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "schema": "direct-compact-run-provenance-v1",
                    "architecture": "qwen-causal-compact-tokens-no-encoder",
                    "decoder_adapter_sha256": sha256_artifact(adapter),
                    "source_overlay_sha256": sha256_file(overlay),
                    "contract_sha256": sha256_file(contract),
                    "train_file_sha256": sha256_file(train),
                    "train_seal_sha256": sha256_file(train_seal),
                    "train_sealed_rows": 12640,
                    "loss_contract": {
                        "sequence_distribution_nll": True,
                        "primary_reduction": (
                            "equal_weight_mean_of_eos_inclusive_"
                            "per_sequence_nll_sums"
                        ),
                    },
                    "heldout_loaded_during_training": False,
                    "eval_file_sha256": None,
                    "eval_seal_sha256": None,
                    "eval_sealed_rows": None,
                    "eval_strategy": "no",
                    "sparse_topk_tail_auxiliary": None,
                    "warmstart_checkpoint": {
                        "path": str(gold.resolve()),
                        "decoder_adapter_sha256": sha256_artifact(gold_adapter),
                        "source_overlay_sha256": sha256_file(gold_overlay),
                        "contract_sha256": sha256_file(gold_contract),
                        "provenance_sha256": sha256_file(gold_provenance),
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return checkpoint, build

    def _cot_checkpoint(self, root: Path) -> tuple[Path, Path]:
        sequence_checkpoint, sequence_build = self._qwen_checkpoint(root)
        stage = sequence_checkpoint.parent
        sequence_build_value = json.loads(
            sequence_build.read_text(encoding="utf-8")
        )
        sequence_inputs = sequence_build_value["inputs"]

        teacher_journal = stage / "qwen_teacher.journal.jsonl"
        teacher_journal.write_text('{"schema":"sealed-journal"}\n', encoding="utf-8")
        teacher_chain_head = stage / "qwen_teacher.journal.jsonl.chain-head.json"
        teacher_chain_head.write_text(
            '{"schema":"sealed-chain-head"}\n', encoding="utf-8"
        )
        student_tokenizer = stage / "student_tokenizer.json"
        student_tokenizer.write_text(
            json.dumps(
                {
                    "schema": "sealed-student-tokenizer",
                    "<think>": QWEN_COT_THINK_OPEN_ID,
                    "</think>": QWEN_COT_THINK_CLOSE_ID,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        sequence_inputs.update(
            {
                "teacher_journal": file_record(teacher_journal),
                "student_tokenizer": file_record(student_tokenizer),
            }
        )
        sequence_build.write_text(
            json.dumps(sequence_build_value, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        cot_checkpoint = stage / "direct_compact_qwen_cot_sft"
        cot_adapter = cot_checkpoint / "decoder_adapter"
        cot_adapter.mkdir(parents=True)
        (cot_adapter / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (cot_adapter / "adapter_model.safetensors").write_bytes(b"cot-adapter")
        cot_overlay = cot_checkpoint / "source_embedding_overlay.pt"
        cot_overlay.write_bytes(b"cot-overlay")
        cot_contract = cot_checkpoint / "compact_contract.json"
        shutil.copyfile(
            Path(sequence_inputs["contract"]["path"]),
            cot_contract,
        )

        cot_train = stage / "qwen_cot_sft_train.jsonl"
        cot_schedule = stage / "qwen_cot_sft_train.schedule.jsonl"
        cot_rows: list[dict[str, object]] = []
        schedule_rows: list[dict[str, object]] = []
        for task_index in range(1580):
            for sample_index in (0, 1):
                position = len(schedule_rows)
                reasoning = f"reasoning-{task_index}-{sample_index}"
                final = f"int fn{task_index}() => {sample_index};"
                target = f"<think>\n{reasoning}\n</think>\n{final}"
                target_sha = hashlib.sha256(target.encode("utf-8")).hexdigest()
                evidence = {
                    "schema": "qwen-target-length-evidence-v1",
                    "sequence_target_sha256": target_sha,
                    "content_token_count": len(target),
                    "eos_inclusive_target_token_count": len(target) + 1,
                    "max_target_tokens": 24576,
                    "within_contract": True,
                    "overflow_by_tokens": 0,
                    "eos_token_id": 99,
                    "eos_appended": True,
                    "final_token_is_eos": True,
                    "add_special_tokens": False,
                    "truncated": False,
                    "prompt_token_count": 10,
                    "compact_source_token_count": 2,
                    "prompt_source_target_token_count": len(target) + 13,
                    "max_total_tokens": 36864,
                    "within_total_contract": True,
                }
                candidate_id = f"candidate-{task_index}-{sample_index}"
                cot_rows.append(
                    {
                        "task_id": f"t{task_index}",
                        "binary_multifunction_schema": (
                            "binary-multifunction-v1-semantic-adapter-v1"
                        ),
                        "dart_source": target,
                        "direct_prompt_mode": QWEN_COT_PROMPT_MODE,
                    }
                )
                schedule_rows.append(
                    {
                        "schema": QWEN_COT_SCHEDULE_SCHEMA,
                        "position": position,
                        "task_id": f"t{task_index}",
                        "sample_index": sample_index,
                        "candidate_id": candidate_id,
                        "base_row_index": task_index,
                        "compact_ids_sha256": "1" * 64,
                        "reasoning_content_sha256": hashlib.sha256(
                            reasoning.encode("utf-8")
                        ).hexdigest(),
                        "raw_final_content_sha256": hashlib.sha256(
                            final.encode("utf-8")
                        ).hexdigest(),
                        "cot_target_sha256": target_sha,
                        "reasoning_content_empty": False,
                        "target_length_evidence": evidence,
                        "selection_rule": "sealed_sample_index_in_[0,1]",
                        "selected_without_outcome_inspection": True,
                    }
                )
        cot_train.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in cot_rows
            ),
            encoding="utf-8",
        )
        cot_schedule.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in schedule_rows
            ),
            encoding="utf-8",
        )
        cot_seal = stage / "qwen_cot_sft_train.seal.json"
        cot_seal.write_text(
            json.dumps(
                {
                    "schema": "compact-public-private-join-seal-v1",
                    "selected_role": "fit",
                    "rows": 3160,
                    "output_sha256": sha256_file(cot_train),
                    "output_size_bytes": cot_train.stat().st_size,
                    "contract_sha256": sha256_file(cot_contract),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        coverage_gate = {
            "passed": True,
            "expected_tasks": 1580,
            "selected_tasks": 1580,
            "expected_rows": 3160,
            "selected_rows": 3160,
            "exact_kcot_coverage_fraction": 1.0,
            "nonempty_reasoning_rows": 3160,
            "empty_reasoning_rows": 0,
            "nonempty_reasoning_fraction": 1.0,
            "empty_reasoning_diagnostics": [],
            "max_target_tokens": 24576,
            "max_total_tokens": 36864,
            "overflow_rows": 0,
            "overflow_diagnostics": [],
            "target_length_evidence_sha256": stable_sha256(
                [row["target_length_evidence"] for row in schedule_rows]
            ),
            "minimum_nonempty_reasoning_fraction": 0.9,
            "pilot_prior": {
                "selected_rows": 128,
                "nonempty_reasoning_rows": 128,
                "nonempty_reasoning_fraction": 1.0,
                "binding": "informational only",
            },
            "empty_rows_retained_if_gate_passes": True,
            "overflow_rows_retained_or_replaced": False,
            "overflow_policy": "abort_build_without_filtering_or_resampling",
        }
        cot_build = stage / "qwen_cot_sft_train.build.json"
        cot_build.write_text(
            json.dumps(
                {
                    "schema": QWEN_COT_BUILD_SCHEMA,
                    "build_completed": True,
                    "objective": {
                        "name": "qwen_cot_hard_sft",
                        "ordinary_hard_sft": True,
                        "dense_token_kl": False,
                        "sequence_forward_kl": False,
                        "reasoning_logprobs_available": False,
                        "pure_sequence_kl_artifact_modified": False,
                        "direct_prompt_mode": QWEN_COT_PROMPT_MODE,
                        "target_template": (
                            "<think>\\n + raw_reasoning_content + "
                            "\\n</think>\\n + raw_final_content"
                        ),
                        "target_transform": (
                            "none_byte_exact_provider_strings"
                        ),
                        "samples_per_task": 2,
                        "selected_sample_indices": [0, 1],
                        "selection_depends_only_on": [
                            "task_id",
                            "sample_index",
                        ],
                        "correctness_filtering": False,
                        "compile_filtering": False,
                        "parseability_filtering": False,
                        "confidence_filtering": False,
                        "logprob_filtering": False,
                        "empty_reasoning_filtering": False,
                        "resampling": False,
                        "provider_calls": False,
                    },
                    "coverage_gate": coverage_gate,
                    "inputs": {
                        "compact_train": sequence_inputs["compact_train"],
                        "compact_train_seal": sequence_inputs[
                            "compact_train_seal"
                        ],
                        "contract": sequence_inputs["contract"],
                        "prompt_artifact": sequence_inputs["prompt_artifact"],
                        "prompt_manifest": sequence_inputs["prompt_manifest"],
                        "f2_prompt_contract": sequence_inputs[
                            "f2_prompt_contract"
                        ],
                        "teacher_journal": file_record(teacher_journal),
                        "teacher_journal_chain_head": file_record(
                            teacher_chain_head
                        ),
                        "teacher_audit": sequence_inputs["teacher_audit"],
                        "student_tokenizer": file_record(student_tokenizer),
                        "native_think_tokens": {
                            "open_token_id": QWEN_COT_THINK_OPEN_ID,
                            "close_token_id": QWEN_COT_THINK_CLOSE_ID,
                        },
                    },
                    "counts": {
                        "tasks": 1580,
                        "rows": 3160,
                        "rows_per_task": 2,
                        "unique_candidate_ids": 3160,
                        "empty_reasoning_rows_retained": 0,
                    },
                    "outputs": {
                        "dataset": file_record(cot_train),
                        "standard_direct_compact_seal": file_record(cot_seal),
                        "schedule": file_record(cot_schedule),
                    },
                    "schedule_sha256": stable_sha256(schedule_rows),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        sequence_provenance = sequence_checkpoint / "run_provenance.json"
        sequence_binding = {
            "path": str(sequence_checkpoint.resolve()),
            "decoder_adapter_sha256": sha256_artifact(
                sequence_checkpoint / "decoder_adapter"
            ),
            "source_overlay_sha256": sha256_file(
                sequence_checkpoint / "source_embedding_overlay.pt"
            ),
            "contract_sha256": sha256_file(
                sequence_checkpoint / "compact_contract.json"
            ),
            "provenance_sha256": sha256_file(sequence_provenance),
        }
        cot_provenance = cot_checkpoint / "run_provenance.json"
        cot_provenance.write_text(
            json.dumps(
                {
                    "schema": "direct-compact-run-provenance-v1",
                    "architecture": "qwen-causal-compact-tokens-no-encoder",
                    "decoder_adapter_sha256": sha256_artifact(cot_adapter),
                    "source_overlay_sha256": sha256_file(cot_overlay),
                    "contract_sha256": sha256_file(cot_contract),
                    "train_file_sha256": sha256_file(cot_train),
                    "train_seal_sha256": sha256_file(cot_seal),
                    "train_sealed_rows": 3160,
                    "loss_contract": {
                        "sequence_distribution_nll": False,
                        "sequence_target_suffix_logits_only": False,
                        "primary_reduction": "base_causal_lm_token_mean",
                    },
                    "heldout_loaded_during_training": False,
                    "eval_file_sha256": None,
                    "eval_seal_sha256": None,
                    "eval_sealed_rows": None,
                    "eval_strategy": "no",
                    "sparse_topk_tail_auxiliary": None,
                    "stage_contract": {
                        "path": str(cot_build.resolve()),
                        "sha256": sha256_file(cot_build),
                        "size_bytes": cot_build.stat().st_size,
                    },
                    "warmstart_checkpoint": sequence_binding,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return cot_checkpoint, sequence_build

    def _capacity_migrate_gold_fixture(
        self,
        checkpoint: Path,
        build: Path,
    ) -> Path:
        stage = checkpoint.parent
        current_contract = checkpoint / "compact_contract.json"
        current_contract_value = json.loads(
            current_contract.read_text(encoding="utf-8")
        )
        build_value = json.loads(build.read_text(encoding="utf-8"))
        current_seal_source = Path(
            build_value["inputs"]["compact_train_seal"]["path"]
        )
        current_seal = stage / "train_multifunction_binary_target24k.seal.json"
        shutil.copyfile(current_seal_source, current_seal)
        build_value["inputs"]["compact_train_seal"] = file_record(current_seal)

        source_gold = stage / "direct_compact_multifunction_gold_sft"
        source_contract = source_gold / "compact_contract.json"
        old_contract_value = dict(current_contract_value)
        old_contract_value["max_target_tokens"] = 4096
        old_contract_value["max_total_tokens"] = 16384
        source_contract.write_text(
            json.dumps(old_contract_value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        old_seal_value = json.loads(current_seal.read_text(encoding="utf-8"))
        old_seal_value["contract_sha256"] = sha256_file(source_contract)
        current_seal_source.write_text(
            json.dumps(old_seal_value, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        source_adapter = source_gold / "decoder_adapter"
        source_overlay = source_gold / "source_embedding_overlay.pt"
        source_provenance_path = source_gold / "run_provenance.json"
        source_provenance = json.loads(
            source_provenance_path.read_text(encoding="utf-8")
        )
        source_provenance.update(
            {
                "decoder_adapter_sha256": sha256_artifact(source_adapter),
                "source_overlay_sha256": sha256_file(source_overlay),
                "contract_sha256": sha256_file(source_contract),
                "train_seal_sha256": sha256_file(current_seal_source),
            }
        )
        source_provenance_path.write_text(
            json.dumps(source_provenance, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_binding = {
            "path": str(source_gold.resolve()),
            "decoder_adapter_sha256": sha256_artifact(source_adapter),
            "source_overlay_sha256": sha256_file(source_overlay),
            "contract_sha256": sha256_file(source_contract),
            "provenance_sha256": sha256_file(source_provenance_path),
        }

        migrated = stage / "direct_compact_multifunction_gold_sft_target24k"
        migrated_adapter = migrated / "decoder_adapter"
        shutil.copytree(source_adapter, migrated_adapter)
        migrated_overlay = migrated / "source_embedding_overlay.pt"
        shutil.copyfile(source_overlay, migrated_overlay)
        migrated_contract = migrated / "compact_contract.json"
        shutil.copyfile(current_contract, migrated_contract)

        source_ids = sorted(
            int(source_id)
            for source_id in current_contract_value[
                "source_token_expansions"
            ]
        )
        expansions = {
            int(source_id): list(expansion)
            for source_id, expansion in current_contract_value[
                "source_token_expansions"
            ].items()
        }
        expansion_sha = stable_sha256(
            {
                str(source_id): expansions[source_id]
                for source_id in sorted(expansions)
            }
        )
        compatibility = {
            "schema": "direct-compact-overlay-migration-compatibility-v1",
            "old_contract_sha256": sha256_file(source_contract),
            "new_contract_sha256": sha256_file(migrated_contract),
            "allowed_changed_fields": sorted(
                {
                    "codec_sha256",
                    "codebook_sha256",
                    "source_token_expansions",
                    "max_target_tokens",
                    "max_total_tokens",
                }
            ),
            "observed_changed_fields": [
                "max_target_tokens",
                "max_total_tokens",
            ],
            "source_token_rows": len(source_ids),
            "identical_expansion_rows": len(source_ids),
            "changed_expansion_rows": 0,
            "identical_expansion_source_token_ids": source_ids,
            "changed_expansion_source_token_ids": [],
            "all_non_migratable_contract_fields_identical": True,
            "stable_source_token_id_sequence_identical": True,
            "base_vocab_size_identical": True,
        }
        migration = {
            "schema": "source-token-overlay-expansion-migration-v1",
            "policy": (
                "reuse_learned_row_iff_source_id_and_ordered_base_token_"
                "expansion_are_identical_else_new_codebook_mean"
            ),
            "base_vocab_size": current_contract_value["base_vocab_size"],
            "source_token_ids": source_ids,
            "source_token_ids_sha256": stable_sha256(source_ids),
            "old_source_token_expansions_sha256": expansion_sha,
            "new_source_token_expansions_sha256": expansion_sha,
            "rows": {
                "total": len(source_ids),
                "reused_identical_expansion": len(source_ids),
                "reinitialized_new_codebook_mean": 0,
            },
            "reused_source_token_ids": source_ids,
            "reinitialized_source_token_ids": [],
            "invariants": {
                "stable_source_token_id_set_identical": True,
                "changed_expansion_rows_copied_from_old_overlay": False,
                "changed_expansion_rows_initialized_from_new_codebook_mean": True,
                "base_embedding_and_lm_head_not_resized": True,
            },
        }
        receipt = {
            "schema": "direct-compact-overlay-migration-receipt-v1",
            "created_at": "2026-07-24T00:00:00Z",
            "training_steps": 0,
            "source_checkpoint": source_binding,
            "contract_compatibility": compatibility,
            "overlay_migration": migration,
            "outputs": {
                "decoder_adapter_sha256": sha256_artifact(migrated_adapter),
                "source_overlay_sha256": sha256_file(migrated_overlay),
                "compact_contract_sha256": sha256_file(migrated_contract),
                "codebook_sha256": current_contract_value["codebook_sha256"],
                "codec_sha256": current_contract_value["codec_sha256"],
            },
            "invariants": {
                "no_training_or_optimizer_step_performed": True,
                "decoder_adapter_tree_byte_identical": True,
                "old_overlay_row_reused_only_for_identical_expansion": True,
                "changed_rows_use_new_codebook_mean_initialization": True,
                "new_contract_copied_byte_identically": True,
                "heldout_data_opened": False,
            },
        }
        receipt_path = migrated / "overlay_migration_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        migrated_provenance = {
            "schema": "direct-compact-run-provenance-v1",
            "architecture": "qwen-causal-compact-tokens-no-encoder",
            "checkpoint_stage": "contract-overlay-migration-only",
            "contract_sha256": sha256_file(migrated_contract),
            "decoder_adapter_sha256": sha256_artifact(migrated_adapter),
            "source_overlay_sha256": sha256_file(migrated_overlay),
            "codebook_sha256": current_contract_value["codebook_sha256"],
            "codec_sha256": current_contract_value["codec_sha256"],
            "source_embedding_overlay_rows": len(source_ids),
            "lm_head_rows": current_contract_value["base_vocab_size"],
            "training_performed": False,
            "heldout_loaded_during_migration": False,
            "overlay_migration_receipt_sha256": sha256_file(receipt_path),
            "warmstart_checkpoint": source_binding,
        }
        migrated_provenance_path = migrated / "run_provenance.json"
        migrated_provenance_path.write_text(
            json.dumps(migrated_provenance, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        migrated_binding = {
            "path": str(migrated.resolve()),
            "decoder_adapter_sha256": sha256_artifact(migrated_adapter),
            "source_overlay_sha256": sha256_file(migrated_overlay),
            "contract_sha256": sha256_file(migrated_contract),
            "provenance_sha256": sha256_file(migrated_provenance_path),
        }
        sequence_provenance_path = checkpoint / "run_provenance.json"
        sequence_provenance = json.loads(
            sequence_provenance_path.read_text(encoding="utf-8")
        )
        sequence_provenance["warmstart_checkpoint"] = migrated_binding
        sequence_provenance_path.write_text(
            json.dumps(sequence_provenance, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        build.write_text(
            json.dumps(build_value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return migrated

    def test_failed_candidate_is_included_whole_or_omitted_per_task(self):
        tokenizer = self._CharTokenizer()
        source = "F2|ASM=lossless|CFG=explicit"
        candidate = "int fn0() => 1;"
        base_messages, base_tokens, base_inclusion = build_repair_prompt(
            tokenizer=tokenizer,
            serialized_text=source,
            student_predictions=[candidate],
            include_student_candidate=False,
            max_prompt_tokens=10000,
            chat_overhead_reserve=0,
            task_id="task-a",
            system_prompt="sealed F2 grammar",
        )
        self.assertFalse(base_inclusion["included"])
        self.assertEqual(
            base_messages,
            [
                {"role": "developer", "content": "sealed F2 grammar"},
                {"role": "user", "content": source},
            ],
        )
        omitted_messages, omitted_tokens, omitted = build_repair_prompt(
            tokenizer=tokenizer,
            serialized_text=source,
            student_predictions=[candidate],
            include_student_candidate=True,
            max_prompt_tokens=base_tokens,
            chat_overhead_reserve=0,
            task_id="task-a",
            system_prompt="sealed F2 grammar",
        )
        self.assertEqual(omitted_tokens, base_tokens)
        self.assertEqual(omitted_messages, base_messages)
        self.assertEqual(omitted["reason"], "optional_candidate_exceeds_cap")

        included_messages, included_tokens, included = build_repair_prompt(
            tokenizer=tokenizer,
            serialized_text=source,
            student_predictions=[candidate],
            include_student_candidate=True,
            max_prompt_tokens=10000,
            chat_overhead_reserve=0,
            task_id="task-b",
            system_prompt="sealed F2 grammar",
        )
        self.assertTrue(included["included"])
        self.assertGreater(included_tokens, base_tokens)
        self.assertEqual(included_messages[:2], base_messages)
        self.assertEqual(len(included_messages), 3)
        self.assertIn(candidate, included_messages[2]["content"])
        self.assertEqual(included_messages[1]["content"], source)

    def test_qwen_stage_checkpoint_and_inference_are_exactly_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            inference = {
                "schema": "direct-compact-inference-v1",
                "selected_role": "fit",
                "decoder_adapter": str(checkpoint / "decoder_adapter"),
                "decoder_adapter_sha256": sha256_artifact(
                    checkpoint / "decoder_adapter"
                ),
                "source_overlay_sha256": sha256_file(
                    checkpoint / "source_embedding_overlay.pt"
                ),
            }
            result = validate_qwen_student_checkpoint(
                checkpoint,
                qwen_build_manifest=build,
                inference_provenance=inference,
            )
            self.assertEqual(result["requested_teacher_model"], QWEN_STAGE_MODEL)
            inference["decoder_adapter"] = str(Path(temporary) / "old_student")
            with self.assertRaisesRegex(
                ValueError, "not generated from the Qwen checkpoint"
            ):
                validate_qwen_student_checkpoint(
                    checkpoint,
                    qwen_build_manifest=build,
                    inference_provenance=inference,
                )

    def test_final_qwen_cot_checkpoint_recursively_validates_sequence_parent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, sequence_build = self._cot_checkpoint(Path(temporary))
            inference = {
                "schema": "direct-compact-inference-v1",
                "selected_role": "fit",
                "decoder_adapter": str(checkpoint / "decoder_adapter"),
                "decoder_adapter_sha256": sha256_artifact(
                    checkpoint / "decoder_adapter"
                ),
                "source_overlay_sha256": sha256_file(
                    checkpoint / "source_embedding_overlay.pt"
                ),
            }
            result = validate_qwen_student_checkpoint(
                checkpoint,
                qwen_build_manifest=sequence_build,
                inference_provenance=inference,
            )
            self.assertEqual(
                result["stage"],
                (
                    "qwen3.8-max-preview-mc-sequence-forward-kl-nll-"
                    "plus-cot-hard-sft"
                ),
            )
            self.assertEqual(
                result["qwen_cot_train_dataset"]["sha256"],
                sha256_file(Path(result["qwen_train_paths"]["dataset"])),
            )
            self.assertEqual(
                result["qwen_sequence_stage"]["stage"],
                "qwen3.8-max-preview-mc-sequence-forward-kl-nll",
            )
            self.assertEqual(
                result["qwen_sequence_warmstart"]["path"],
                str(
                    (
                        Path(temporary)
                        / "qwen_stage"
                        / "direct_compact_qwen_sequence_warmstart"
                    ).resolve()
                ),
            )

    def test_qwen_cot_manifest_objective_native_tokens_and_coverage_fail_closed(
        self,
    ) -> None:
        mutations = (
            (
                "objective",
                lambda value: value["objective"].__setitem__(
                    "ordinary_hard_sft", False
                ),
                "objective contract",
            ),
            (
                "native-think",
                lambda value: value["inputs"]["native_think_tokens"].__setitem__(
                    "open_token_id", 0
                ),
                "native Qwen3 think token",
            ),
            (
                "coverage",
                lambda value: value["coverage_gate"].__setitem__(
                    "selected_rows", 3159
                ),
                "coverage/24K",
            ),
        )
        for name, mutate, expected in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                checkpoint, sequence_build = self._cot_checkpoint(
                    Path(temporary)
                )
                cot_build = checkpoint.parent / "qwen_cot_sft_train.build.json"
                value = json.loads(cot_build.read_text(encoding="utf-8"))
                mutate(value)
                cot_build.write_text(
                    json.dumps(value, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                provenance_path = checkpoint / "run_provenance.json"
                provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8")
                )
                provenance["stage_contract"] = {
                    "path": str(cot_build.resolve()),
                    "sha256": sha256_file(cot_build),
                    "size_bytes": cot_build.stat().st_size,
                }
                provenance_path.write_text(
                    json.dumps(provenance, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, expected):
                    validate_qwen_student_checkpoint(
                        checkpoint,
                        qwen_build_manifest=sequence_build,
                    )

    def test_qwen_cot_exact_input_and_output_hashes_fail_closed(self) -> None:
        for artifact_key, expected in (
            ("student_tokenizer", "sealed SHA-256"),
            ("dataset", "sealed SHA-256"),
        ):
            with (
                self.subTest(artifact=artifact_key),
                tempfile.TemporaryDirectory() as temporary,
            ):
                checkpoint, sequence_build = self._cot_checkpoint(
                    Path(temporary)
                )
                cot_build_path = (
                    checkpoint.parent / "qwen_cot_sft_train.build.json"
                )
                cot_build = json.loads(
                    cot_build_path.read_text(encoding="utf-8")
                )
                if artifact_key == "student_tokenizer":
                    artifact = Path(
                        cot_build["inputs"]["student_tokenizer"]["path"]
                    )
                else:
                    artifact = Path(cot_build["outputs"]["dataset"]["path"])
                with artifact.open("a", encoding="utf-8") as handle:
                    handle.write("tamper\n")
                with self.assertRaisesRegex(ValueError, expected):
                    validate_qwen_student_checkpoint(
                        checkpoint,
                        qwen_build_manifest=sequence_build,
                    )

    def test_qwen_cot_recursively_rejects_truncated_sequence_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, sequence_build = self._cot_checkpoint(Path(temporary))
            build_value = json.loads(
                sequence_build.read_text(encoding="utf-8")
            )
            build_value["counts"]["teacher_draw_rows"] = 1
            sequence_build.write_text(
                json.dumps(build_value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "sealed Qwen sequence stage"
            ):
                validate_qwen_student_checkpoint(
                    checkpoint,
                    qwen_build_manifest=sequence_build,
                )

    def test_capacity_only_migrated_gold_checkpoint_is_fully_traced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            migrated = self._capacity_migrate_gold_fixture(checkpoint, build)
            result = validate_qwen_student_checkpoint(
                checkpoint, qwen_build_manifest=build
            )
            migration = result["qwen_gold_adaptation"][
                "capacity_only_migration"
            ]
            self.assertEqual(
                result["qwen_gold_adaptation"]["path"],
                str(migrated.resolve()),
            )
            self.assertEqual(
                migration["changed_contract_fields"],
                ["max_target_tokens", "max_total_tokens"],
            )
            self.assertEqual(migration["reused_source_token_rows"], 2)
            self.assertEqual(migration["reinitialized_source_token_rows"], 0)
            nested = migration["nested_source_gold_sft"]
            self.assertEqual(
                nested["compact_contract_sha256"],
                sha256_file(Path(nested["path"]) / "compact_contract.json"),
            )

    def test_capacity_migration_receipt_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            migrated = self._capacity_migrate_gold_fixture(checkpoint, build)
            receipt_path = migrated / "overlay_migration_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["contract_compatibility"]["observed_changed_fields"] = [
                "codec_sha256",
                "max_target_tokens",
                "max_total_tokens",
            ]
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "migration receipt contract failed"
            ):
                validate_qwen_student_checkpoint(
                    checkpoint, qwen_build_manifest=build
                )

    def test_capacity_contract_gate_rejects_semantic_or_shrinking_changes(
        self,
    ) -> None:
        old = {
            "codec_sha256": "a" * 64,
            "max_target_tokens": 4096,
            "max_total_tokens": 16384,
        }
        current = {
            **old,
            "max_target_tokens": 24576,
            "max_total_tokens": 36864,
        }
        self.assertEqual(
            validate_capacity_only_contracts(old, current),
            ["max_target_tokens", "max_total_tokens"],
        )
        with self.assertRaisesRegex(ValueError, "not capacity-only"):
            validate_capacity_only_contracts(
                old, {**current, "codec_sha256": "b" * 64}
            )
        with self.assertRaisesRegex(ValueError, "monotonically increase"):
            validate_capacity_only_contracts(
                old,
                {
                    **old,
                    "max_target_tokens": 2048,
                    "max_total_tokens": 36864,
                },
            )

    def test_capacity_migration_requires_complete_nested_gold_sft(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            migrated = self._capacity_migrate_gold_fixture(checkpoint, build)
            receipt = json.loads(
                (migrated / "overlay_migration_receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            source = Path(receipt["source_checkpoint"]["path"])
            (source / "decoder_adapter" / "adapter_config.json").unlink()
            with self.assertRaisesRegex(
                ValueError, "nested source gold-SFT checkpoint is incomplete"
            ):
                validate_qwen_student_checkpoint(
                    checkpoint, qwen_build_manifest=build
                )

    def test_current_compact_reseal_must_bind_current_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            self._capacity_migrate_gold_fixture(checkpoint, build)
            build_value = json.loads(build.read_text(encoding="utf-8"))
            seal_path = Path(
                build_value["inputs"]["compact_train_seal"]["path"]
            )
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["contract_sha256"] = "0" * 64
            seal_path.write_text(
                json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8"
            )
            build_value["inputs"]["compact_train_seal"] = file_record(seal_path)
            build.write_text(
                json.dumps(build_value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "current contract"):
                validate_qwen_student_checkpoint(
                    checkpoint, qwen_build_manifest=build
                )

    def test_sequence_reseal_must_bind_current_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            build_value = json.loads(build.read_text(encoding="utf-8"))
            seal_path = Path(
                build_value["outputs"]["standard_direct_compact_seal"]["path"]
            )
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["contract_sha256"] = "0" * 64
            seal_path.write_text(
                json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8"
            )
            build_value["outputs"]["standard_direct_compact_seal"] = (
                file_record(seal_path)
            )
            provenance_path = checkpoint / "run_provenance.json"
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            provenance["train_seal_sha256"] = sha256_file(seal_path)
            provenance_path.write_text(
                json.dumps(provenance, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            build.write_text(
                json.dumps(build_value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "current contract"):
                validate_qwen_student_checkpoint(
                    checkpoint, qwen_build_manifest=build
                )

    def test_non_qwen_or_wrong_qwen_teacher_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            build_value = json.loads(build.read_text(encoding="utf-8"))
            audit_path = Path(build_value["inputs"]["teacher_audit"]["path"])
            audit_value = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_value["homogeneous_backend_shards"][0][
                "backend_identity"
            ]["requested_model"] = "qwen3.7-max"
            audit_path.write_text(
                json.dumps(audit_value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            build_value["inputs"]["teacher_audit"] = file_record(audit_path)
            build.write_text(
                json.dumps(build_value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model other than"):
                validate_qwen_student_checkpoint(
                    checkpoint, qwen_build_manifest=build
                )

    def test_truncated_qwen_k8_manifest_is_rejected_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, build = self._qwen_checkpoint(Path(temporary))
            build_value = json.loads(build.read_text(encoding="utf-8"))
            build_value["counts"]["teacher_draw_rows"] = 1
            build_value["counts"]["unique_teacher_candidate_ids"] = 1
            build.write_text(
                json.dumps(build_value, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "sealed Qwen sequence stage"
            ):
                validate_qwen_student_checkpoint(
                    checkpoint, qwen_build_manifest=build
                )

    def test_scheduler_is_bounded_sample_major_and_stops_at_coverage(self) -> None:
        failed = [f"t{index:02d}" for index in range(12)]
        calls: list[tuple[str, int]] = []
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def harvest(job: tuple[str, int]):
            nonlocal active, maximum_active
            with lock:
                calls.append(job)
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return {"terminal": True, "passed": True}

        result = run_sample_major_bounded(
            failed_ids=failed,
            samples_per_task=4,
            terminal_slots=set(),
            verified_task_ids=set(),
            minimum_verified_tasks=4,
            workers=2,
            harvest=harvest,
        )
        self.assertEqual(result["stop_reason"], "coverage_target_reached")
        self.assertGreaterEqual(len(result["verified_task_ids"]), 4)
        self.assertLessEqual(len(calls), 5)
        self.assertEqual({sample for _task, sample in calls}, {0})
        self.assertLessEqual(maximum_active, 2)

    def test_scheduler_finishes_a_task_wide_round_before_next_round(self) -> None:
        failed = [f"t{index}" for index in range(5)]
        calls: list[tuple[str, int]] = []

        def harvest(job: tuple[str, int]):
            calls.append(job)
            return {"terminal": True, "passed": job[1] == 1}

        run_sample_major_bounded(
            failed_ids=failed,
            samples_per_task=3,
            terminal_slots=set(),
            verified_task_ids=set(),
            minimum_verified_tasks=2,
            workers=2,
            harvest=harvest,
        )
        first_second_round = next(
            index for index, job in enumerate(calls) if job[1] == 1
        )
        self.assertEqual(first_second_round, len(failed))
        self.assertTrue(all(sample == 0 for _task, sample in calls[:5]))

    def test_resume_contract_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_contract.json"
            ensure_run_contract(path, {"model": "gpt-5.6-sol"})
            ensure_run_contract(path, {"model": "gpt-5.6-sol"})
            with self.assertRaisesRegex(ValueError, "run contract changed"):
                ensure_run_contract(path, {"model": "other"})

    def test_only_official_openai_endpoint_is_allowed(self) -> None:
        self.assertEqual(
            validate_openai_base_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1",
        )
        with self.assertRaisesRegex(ValueError, "official OpenAI endpoint"):
            validate_openai_base_url(
                "https://example-resource.openai.azure.com/openai/v1"
            )

    def test_responses_call_has_reasoning_and_no_temperature(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "training"
            / "collect_chatgpt_compact_rs.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        matching_calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "create"
                and isinstance(function.value, ast.Attribute)
                and function.value.attr == "responses"
            ):
                matching_calls.append(node)
        self.assertEqual(len(matching_calls), 1)
        keywords = {keyword.arg for keyword in matching_calls[0].keywords}
        self.assertIn("reasoning", keywords)
        self.assertNotIn("temperature", keywords)

    def test_output_budget_escalates_only_for_explicit_max_token_incomplete(
        self,
    ) -> None:
        self.assertEqual(
            escalated_output_token_budget(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
                current_budget=8192,
                ceiling_budget=12288,
            ),
            12288,
        )
        for status, details in (
            ("completed", {"reason": "max_output_tokens"}),
            ("incomplete", {"reason": "content_filter"}),
            ("failed", {"reason": "max_output_tokens"}),
            ("incomplete", None),
        ):
            self.assertIsNone(
                escalated_output_token_budget(
                    status=status,
                    incomplete_details=details,
                    current_budget=8192,
                    ceiling_budget=12288,
                )
            )
        self.assertIsNone(
            escalated_output_token_budget(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
                current_budget=12288,
                ceiling_budget=12288,
            )
        )


if __name__ == "__main__":
    unittest.main()
