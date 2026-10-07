from __future__ import annotations

import argparse
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from models.direct_compact_causal import (
    CONTRACT_SCHEMA,
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.training.build_qwen_cot_sft import (
    QWEN3_THINK_CLOSE_ID,
    QWEN3_THINK_OPEN_ID,
    _require_native_think_tokens,
    build,
    compose_cot_target,
    cot_coverage_gate,
    selected_candidates,
)
from scripts.training.direct_compact_qwen_decompiler import (
    DIRECT_PROMPT_MODE_CODE_ONLY_V1,
    DIRECT_PROMPT_MODE_QWEN_COT_V1,
    direct_prompt,
    target_source,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (
    DEFAULT_MODEL,
    OBJECTIVE_MODE_SEQUENCE_ONLY,
    SAMPLE_SEED_ALGORITHM,
    ArtifactError,
    JournalState,
    PromptRow,
    StudentTokenizerBinding,
    build_messages,
    collect_candidates,
    file_record,
    materialize_artifacts,
    read_jsonl,
    sha256_text,
    stable_sha256,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = list(ids)


class _ThinkTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<think>":
            return _Encoding([QWEN3_THINK_OPEN_ID])
        if text == "</think>":
            return _Encoding([QWEN3_THINK_CLOSE_ID])
        return _Encoding([])


class _SequenceCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **payload):
        index = len(self.calls)
        self.calls.append(payload)
        return types.SimpleNamespace(
            id=f"request-{index}",
            model=DEFAULT_MODEL,
            created=1000 + index,
            system_fingerprint="backend",
            service_tier="default",
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            choices=[
                types.SimpleNamespace(
                    finish_reason="stop",
                    message=types.SimpleNamespace(
                        content=f"int fn0() => {index};",
                        reasoning_content=f"reasoning draw {index}",
                    ),
                    logprobs=types.SimpleNamespace(content=[]),
                )
            ],
        )


class _SequenceClient:
    def __init__(self) -> None:
        completions = _SequenceCompletions()
        self.completions = completions
        self.chat = types.SimpleNamespace(completions=completions)


def _journal(task_ids: list[str]) -> JournalState:
    candidates = {}
    for task_id in task_ids:
        for sample_index in range(8):
            candidate_id = f"{task_id}-draw-{sample_index}"
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "task_id": task_id,
                "sample_index": sample_index,
                "prompt_sha256": f"prompt-{task_id}",
                "completion_attested": True,
                # Deliberately vary outcome-like fields. Selection must ignore
                # all of them and still choose slot numbers 0 and 1.
                "parse": {"parseable": sample_index % 2 == 0},
                "chosen_tokens_with_top_logprobs": (
                    [{"logprob": 0.0}] if sample_index == 7 else []
                ),
                "response": {
                    "raw_reasoning_content": (
                        "" if sample_index == 0 else f"reason-{sample_index}"
                    ),
                    "raw_content": f"final-{sample_index}",
                },
            }
    return JournalState(
        header=None,
        candidates=candidates,
        rejections={},
        starts={},
        terminals={},
        slots={},
        verifications={},
        error_counts={},
    )


class QwenCotSftTests(unittest.TestCase):
    def test_code_only_prompt_is_byte_identical_to_pre_mode_contract(self):
        row = {"lang": "Dart", "function": "fn0"}
        expected = (
            "Decompile the following compact binary representation to Dart.\n"
            "Return one self-contained compilable source-unit fragment, including "
            "required imports and top-level helpers, without markdown, prose, "
            "tests, or demos.\n"
            "The fragment must define a top-level function named exactly fn0.\n"
            "Do not define main. Infer fn0's return type and complete parameter "
            "contract from the binary.\n"
            "Compact binary tokens follow:\n"
        )
        self.assertEqual(direct_prompt(row), expected)
        self.assertEqual(
            direct_prompt(
                {**row, "direct_prompt_mode": DIRECT_PROMPT_MODE_CODE_ONLY_V1}
            ),
            expected,
        )

    def test_cot_mode_is_distinct_and_preserves_exact_raw_target_bytes(self):
        raw_reasoning = "  inspect eax\nkeep trailing space  "
        raw_final = "\nint fn0() => 7;\n  "
        target = compose_cot_target(raw_reasoning, raw_final)
        self.assertEqual(
            target,
            "<think>\n  inspect eax\nkeep trailing space  "
            "\n</think>\n\nint fn0() => 7;\n  ",
        )
        row = {
            "function": "fn0",
            "lang": "Dart",
            "direct_prompt_mode": DIRECT_PROMPT_MODE_QWEN_COT_V1,
            "dart_source": target,
        }
        self.assertEqual(target_source(row, "task"), target)
        self.assertIn("<think>...</think>", direct_prompt(row))
        self.assertNotEqual(
            direct_prompt(row),
            direct_prompt({"function": "fn0", "lang": "Dart"}),
        )

    def test_empty_reasoning_target_is_retained_exactly(self):
        self.assertEqual(
            compose_cot_target("", "int fn0() => 1;"),
            "<think>\n\n</think>\nint fn0() => 1;",
        )

    def test_selection_is_exact_slots_zero_and_one_not_outcome_based(self):
        tasks = ["a", "b"]
        selected = selected_candidates(
            state=_journal(tasks),
            task_ids=tasks,
            prompt_hashes={task: f"prompt-{task}" for task in tasks},
        )
        self.assertEqual(
            [(row["task_id"], row["sample_index"]) for row in selected],
            [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
        )
        # Slot 0 has empty reasoning and slot 1 has no favorable score. Both
        # remain selected; later "better" draws are not substituted.
        self.assertEqual(selected[0]["response"]["raw_reasoning_content"], "")
        self.assertEqual(selected[1]["chosen_tokens_with_top_logprobs"], [])

    def test_selection_fails_closed_on_incomplete_k8_without_resampling(self):
        state = _journal(["a"])
        state.candidates.pop("a-draw-6")
        with self.assertRaisesRegex(ArtifactError, "complete sealed K=8"):
            selected_candidates(
                state=state,
                task_ids=["a"],
                prompt_hashes={"a": "prompt-a"},
            )

    def test_coverage_gate_counts_empty_rows_and_enforces_90_percent(self):
        schedule = [
            {
                "task_id": f"task-{index // 2}",
                "target_length_evidence": {"tokens": index + 1},
            }
            for index in range(20)
        ]
        empty = [{"task_id": "task-0"}, {"task_id": "task-1"}]
        gate = cot_coverage_gate(
            task_count=10,
            schedule_rows=schedule,
            empty_reasoning=empty,
            overflow=[],
            min_nonempty_reasoning_fraction=0.90,
            max_target_tokens=24576,
            max_total_tokens=36864,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["selected_rows"], 20)
        self.assertEqual(gate["empty_reasoning_rows"], 2)
        self.assertEqual(gate["nonempty_reasoning_fraction"], 0.9)

        failed = cot_coverage_gate(
            task_count=10,
            schedule_rows=schedule,
            empty_reasoning=empty + [{"task_id": "task-2"}],
            overflow=[],
            min_nonempty_reasoning_fraction=0.90,
            max_target_tokens=24576,
            max_total_tokens=36864,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["empty_reasoning_rows"], 3)

    def test_any_24k_or_total_context_overflow_fails_gate(self):
        schedule = [
            {
                "task_id": "task",
                "target_length_evidence": {
                    "eos_inclusive_target_token_count": 24577,
                    "max_target_tokens": 24576,
                },
            },
            {
                "task_id": "task",
                "target_length_evidence": {
                    "eos_inclusive_target_token_count": 10,
                    "max_target_tokens": 24576,
                },
            },
        ]
        overflow = [{"task_id": "task", "sample_index": 0}]
        gate = cot_coverage_gate(
            task_count=1,
            schedule_rows=schedule,
            empty_reasoning=[],
            overflow=overflow,
            min_nonempty_reasoning_fraction=0.90,
            max_target_tokens=24576,
            max_total_tokens=36864,
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["overflow_rows"], 1)
        self.assertEqual(gate["overflow_diagnostics"], overflow)

    def test_qwen_native_think_token_ids_are_required(self):
        self.assertEqual(
            _require_native_think_tokens(_ThinkTokenizer()),
            {
                "open_token_id": QWEN3_THINK_OPEN_ID,
                "close_token_id": QWEN3_THINK_CLOSE_ID,
            },
        )

        class WrongTokenizer(_ThinkTokenizer):
            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return _Encoding([42])

        with self.assertRaisesRegex(ArtifactError, "native think tokens"):
            _require_native_think_tokens(WrongTokenizer())

    def test_end_to_end_builder_emits_two_raw_cot_rows_and_standard_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from tokenizers import Tokenizer, models

            tokenizer_path = root / "tokenizer.json"
            Tokenizer(
                models.WordLevel(
                    {
                        "<unk>": 0,
                        "<eos>": 1,
                        **{f"int fn0() => {index};": index + 2 for index in range(8)},
                        "<think>": 10,
                        "</think>": 11,
                    },
                    unk_token="<unk>",
                )
            ).save(str(tokenizer_path))
            tokenizer_sha = sha256_file(tokenizer_path)
            contract = DirectCompactContract(
                schema=CONTRACT_SCHEMA,
                codec_sha256="a" * 64,
                codebook_sha256="b" * 64,
                tokenizer_json_sha256=tokenizer_sha,
                tokenizer_fingerprint_sha256="c" * 64,
                model_config_sha256="d" * 64,
                decoder_model="fake/qwen",
                decoder_revision="immutable",
                target_function="fn0",
                target_language="Dart",
                dfg_extractor_sha256="e" * 64,
                lossless_domain="scrubbed_canonical_graph",
                max_source_tokens=128,
                max_target_tokens=128,
                max_total_tokens=512,
                base_vocab_size=12,
                source_token_ids=(12,),
                source_token_expansions=((12, (2,)),),
            )
            contract_path = root / "contract.json"
            contract_path.write_text(
                json.dumps(contract.as_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            train_path = root / "train.jsonl"
            train_row = {
                "task_id": "task",
                "lang": "Dart",
                "function": "fn0",
                "dart_source": "int fn0() => 99;",
                "compact_input_ids": [12],
                "compact_codec_sha256": contract.codec_sha256,
                "compact_codebook_sha256": contract.codebook_sha256,
                "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
            }
            train_path.write_text(
                json.dumps(train_row, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            train_seal = root / "train.seal.json"
            train_seal.write_text(
                json.dumps(
                    {
                        "schema": "compact-public-private-join-seal-v1",
                        "selected_role": "fit",
                        "output_sha256": sha256_file(train_path),
                        "contract_sha256": sha256_file(contract_path),
                        "rows": 1,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            system_prompt = "sealed system"
            prompt_text = "F2\nBLOCKS\n0:!ret\nEND\n"
            prompt_row = {
                "schema": "frontier-compact-api-readable-v1",
                "representation_schema": "lossless-semantic-f2",
                "system_prompt_sha256": sha256_text(system_prompt),
                "task_id": "task",
                "text": prompt_text,
                "text_sha256": sha256_text(prompt_text),
                "compact_ids_sha256": stable_sha256([12]),
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
            prompt_path = root / "prompts.jsonl"
            prompt_path.write_text(
                json.dumps(prompt_row, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            prompt = PromptRow(
                task_id="task",
                text=prompt_text,
                text_sha256=sha256_text(prompt_text),
                source_record_sha256=stable_sha256(prompt_row),
                source_schema=prompt_row["schema"],
                representation_schema="lossless-semantic-f2",
                system_prompt_sha256=sha256_text(system_prompt),
            )
            tokenizer_record = file_record(tokenizer_path)
            generation = {
                "n": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 12288,
                "extra_body": {
                    "top_k": 101,
                    "enable_thinking": True,
                    "thinking_budget": 8192,
                },
            }
            messages_sha = stable_sha256(build_messages(system_prompt, prompt))
            header = {
                "collection_mode": "offline_precompute_only_no_gradient_loop",
                "prompt_artifact": file_record(prompt_path),
                "prompt_manifest": {
                    "path": "manifest.json",
                    "sha256": "f" * 64,
                    "size_bytes": 1,
                },
                "f2_prompt_contract": {
                    "representation_schema": "lossless-semantic-f2",
                    "system_prompt_sha256": sha256_text(system_prompt),
                },
                "task_ids": ["task"],
                "prompt_bindings": [
                    {
                        "task_id": "task",
                        "serializer_text_sha256": prompt.text_sha256,
                        "source_record_sha256": prompt.source_record_sha256,
                        "request_messages_sha256": messages_sha,
                    }
                ],
                "samples_per_task": 8,
                "requested_model": DEFAULT_MODEL,
                "returned_model_must_equal_requested": True,
                "objective_mode": OBJECTIVE_MODE_SEQUENCE_ONLY,
                "generation_parameters": generation,
                "transport": {
                    "length_capped_response_policy": {
                        "same_task_draw_only": True,
                        "completed_draws_reissued": False,
                        "max_token_capacities": [12288, 16384, 24576],
                        "capped_responses_retained_by_hash": True,
                    }
                },
                "sampling_seed_contract": {
                    "algorithm": SAMPLE_SEED_ALGORITHM,
                    "seed_base": 44,
                    "unique_seed_required_per_task_draw": True,
                    "provider_seed_honor_not_assumed": True,
                    "response_seed_echo_required_to_attest_honor": True,
                },
                "target_length_contract": {
                    "schema": "qwen-sequence-target-length-contract-v1",
                    "trainer_contract": file_record(contract_path),
                    "trainer_contract_schema": CONTRACT_SCHEMA,
                    "max_target_tokens": 128,
                    "student_tokenizer": tokenizer_record,
                    "student_eos_token_id": 1,
                    "tokenization": {
                        "add_special_tokens": False,
                        "eos_policy": ("append_exactly_once_if_final_token_is_not_eos"),
                        "matches_trainer_dataset_loader": True,
                        "truncation_permitted": False,
                        "overflow_filtering_permitted": False,
                        "overflow_resampling_permitted": False,
                    },
                    "target_source": {
                        "field": "choice.message.content",
                        "reasoning_field": ("choice.message.reasoning_content"),
                        "reasoning_excluded": True,
                        "final_dart_code_only_required": False,
                    },
                },
            }
            journal = root / "teacher.journal.jsonl"
            collect_candidates(
                prompts=[prompt],
                client=_SequenceClient(),
                journal_path=journal,
                header_payload=header,
                system_prompt=system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=generation,
                required_function="fn0",
                verifier=lambda candidate: {
                    "compiled": False,
                    "passed": False,
                    "harness_completion_attested": True,
                    "diagnostic": f"ignored-{candidate['sample_index']}",
                    "verifier_id": "test",
                    "verifier_sha256": "1" * 64,
                    "tests_sha256": "2" * 64,
                },
                seed_base=44,
                require_returned_model_exact=True,
            )
            sequence_rows = root / "sequence.jsonl"
            rs_rows = root / "rs.jsonl"
            audit_path = root / "audit.json"
            materialize_artifacts(
                journal_path=journal,
                binding=StudentTokenizerBinding(
                    Tokenizer.from_file(str(tokenizer_path)),
                    eos_token_id=1,
                    tokenizer_record=tokenizer_record,
                ),
                parseable_output=sequence_rows,
                rs_sft_output=rs_rows,
                audit_output=audit_path,
            )

            output = root / "cot.jsonl"
            output_seal = root / "cot.seal.json"
            schedule = root / "cot.schedule.jsonl"
            manifest_path = root / "cot.build.json"
            with patch(
                "scripts.training.build_qwen_cot_sft." "_require_native_think_tokens",
                return_value={
                    "open_token_id": QWEN3_THINK_OPEN_ID,
                    "close_token_id": QWEN3_THINK_CLOSE_ID,
                },
            ):
                manifest = build(
                    argparse.Namespace(
                        compact_train_jsonl=train_path,
                        compact_train_seal=train_seal,
                        contract=contract_path,
                        prompt_jsonl=prompt_path,
                        expected_prompt_sha256=sha256_file(prompt_path),
                        teacher_journal=journal,
                        expected_teacher_journal_sha256=sha256_file(journal),
                        teacher_audit_json=audit_path,
                        expected_teacher_audit_sha256=sha256_file(audit_path),
                        student_tokenizer_json=tokenizer_path,
                        expected_student_tokenizer_sha256=tokenizer_sha,
                        output_jsonl=output,
                        output_seal=output_seal,
                        schedule_output=schedule,
                        build_manifest=manifest_path,
                        min_nonempty_reasoning_fraction=0.90,
                    )
                )
            self.assertTrue(manifest["build_completed"])
            self.assertEqual(manifest["counts"]["rows"], 2)
            self.assertFalse(manifest["objective"]["correctness_filtering"])
            rows = read_jsonl(output)
            self.assertEqual(
                [row["direct_prompt_mode"] for row in rows],
                [DIRECT_PROMPT_MODE_QWEN_COT_V1] * 2,
            )
            self.assertEqual(
                rows[0]["dart_source"],
                "<think>\nreasoning draw 0\n</think>\nint fn0() => 0;",
            )
            self.assertEqual(
                rows[1]["dart_source"],
                "<think>\nreasoning draw 1\n</think>\nint fn0() => 1;",
            )
            self.assertEqual(
                [row["sample_index"] for row in read_jsonl(schedule)],
                [0, 1],
            )
            validate_join_seal(output, output_seal, contract_path, expected_role="fit")


if __name__ == "__main__":
    unittest.main()
