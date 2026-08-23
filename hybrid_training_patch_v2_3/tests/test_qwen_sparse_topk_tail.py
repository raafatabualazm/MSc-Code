from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from models.direct_compact_causal import (
    CONTRACT_SCHEMA,
    DirectCompactBatchCollator,
    DirectCompactContract,
    sha256_file,
    validate_join_seal,
)
from scripts.training.build_qwen_sparse_topk_tail_auxiliary import (
    IneligibleSparseDraw,
    _sparse_payload,
    build as build_sparse_artifact,
)
from scripts.training.direct_compact_sparse_topk_tail import (
    SPARSE_FIELD,
    SPARSE_MANIFEST_SCHEMA,
    SPARSE_ROW_SCHEMA,
    DirectCompactSparseTopKTailCausalLM,
    SparseTopKTailCollator,
    attach_sparse_metadata,
    coarsened_topk_tail_forward_kl,
    validate_sparse_manifest,
)
from scripts.training.qwen_direct_compact_teacher_artifact import (
    file_record,
    read_jsonl,
    stable_sha256,
)


class SparseObjectiveTests(unittest.TestCase):
    def test_matches_manual_coarsened_forward_kl(self) -> None:
        student = torch.tensor([[0.4, 0.1, 0.3, 0.2]]).log()
        actual = coarsened_topk_tail_forward_kl(
            student,
            torch.tensor([[0, 2]]),
            torch.tensor([[math.log(0.5), math.log(0.2)]], dtype=torch.float64),
            torch.tensor([0.3], dtype=torch.float64),
        )
        expected = (
            0.5 * math.log(0.5 / 0.4)
            + 0.2 * math.log(0.2 / 0.3)
            + 0.3 * math.log(0.3 / 0.3)
        )
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_teacher_tail_is_never_clamped(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            coarsened_topk_tail_forward_kl(
                torch.zeros(1, 4),
                torch.tensor([[0]]),
                torch.tensor([[math.log(0.9)]], dtype=torch.float64),
                torch.tensor([-0.1], dtype=torch.float64),
            )

    def test_duplicate_partition_categories_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            coarsened_topk_tail_forward_kl(
                torch.zeros(1, 4),
                torch.tensor([[0, 0]]),
                torch.tensor(
                    [[math.log(0.2), math.log(0.2)]], dtype=torch.float64
                ),
                torch.tensor([0.6], dtype=torch.float64),
            )


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits_parameter = torch.nn.Parameter(
            torch.tensor([0.0, 0.4, -0.2, 0.1])
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        del attention_mask, kwargs
        logits = self.logits_parameter.view(1, 1, -1).expand(
            input_ids.size(0), input_ids.size(1), -1
        )
        loss = (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            if labels is not None
            else None
        )
        return {"loss": loss, "logits": logits}


class SparseDirectIntegrationTests(unittest.TestCase):
    def feature(self):
        return {
            "decoder_prompt_input_ids": [0],
            "compact_input_ids": [4],
            "target_input_ids": [1, 2],
            SPARSE_FIELD: {
                "target_token_ids": [1],
                "teacher_positions": [
                    {
                        "observed_token_id": 1,
                        "top_token_ids": [1, 3],
                        "top_logprobs": [math.log(0.6), math.log(0.3)],
                        "tail_probability_mass": 0.1,
                    }
                ],
            },
        }

    def test_primary_nll_plus_subunit_auxiliary_and_eos_is_not_sparse(self):
        base = DirectCompactBatchCollator(
            pad_token_id=0,
            max_source_tokens=8,
            max_target_tokens=8,
            max_total_tokens=16,
            source_token_ids=(4,),
        )
        batch = SparseTopKTailCollator(base)([self.feature()])
        self.assertEqual(int(batch["sparse_teacher_position_mask"].sum()), 1)
        self.assertEqual(int(batch["labels"].ne(-100).sum()), 2)

        model = DirectCompactSparseTopKTailCausalLM(
            TinyCausalLM(),
            auxiliary_weight=0.1,
            sequence_sum_nll=True,
        )
        output = model(**batch)
        primary = output["primary_sequence_nll"]
        auxiliary = output["sparse_topk_tail_forward_kl"]
        token_mean = F.cross_entropy(
            output["logits"][:, :-1].reshape(-1, output["logits"].size(-1)),
            batch["labels"][:, 1:].reshape(-1),
            ignore_index=-100,
        )
        self.assertAlmostEqual(
            float(primary.detach()),
            float((2.0 * token_mean).detach()),
            places=6,
        )
        self.assertAlmostEqual(
            float(output["loss"].detach()),
            float((primary + 0.1 * auxiliary).detach()),
            places=6,
        )
        output["loss"].backward()
        self.assertIsNotNone(model.causal_lm.logits_parameter.grad)

    def test_attempting_sparse_eos_supervision_fails(self):
        base = DirectCompactBatchCollator(
            pad_token_id=0,
            max_source_tokens=8,
            max_target_tokens=8,
            max_total_tokens=16,
            source_token_ids=(4,),
        )
        batch = SparseTopKTailCollator(base)([self.feature()])
        # Forge a second sparse position corresponding to the appended EOS.
        batch["sparse_teacher_position_mask"] = torch.tensor([[True, True]])
        batch["sparse_teacher_top_token_ids"] = torch.tensor([[[1, 3], [2, 0]]])
        batch["sparse_teacher_top_logprobs"] = torch.tensor(
            [
                [
                    [math.log(0.6), math.log(0.3)],
                    [math.log(0.7), math.log(0.2)],
                ]
            ],
            dtype=torch.float64,
        )
        batch["sparse_teacher_top_mask"] = torch.ones((1, 2, 2), dtype=torch.bool)
        batch["sparse_teacher_tail_mass"] = torch.tensor(
            [[0.1, 0.1]], dtype=torch.float64
        )
        batch["sparse_teacher_observed_ids"] = torch.tensor([[1, 2]])
        model = DirectCompactSparseTopKTailCausalLM(
            TinyCausalLM(), auxiliary_weight=0.1
        )
        with self.assertRaisesRegex(ValueError, "EOS"):
            model(**batch)


class MappingAndSealTests(unittest.TestCase):
    @staticmethod
    def tokenizer():
        from tokenizers import Tokenizer, models

        return Tokenizer(
            models.WordLevel(
                {
                    "<pad>": 0,
                    "int fn0() => 1;": 1,
                    "alt-a": 2,
                    "alt-b": 3,
                    "alt-c": 4,
                    "alt-d": 5,
                    "<eos>": 6,
                    "<unk>": 7,
                },
                unk_token="<unk>",
            )
        )

    @staticmethod
    def teacher_row():
        code = "int fn0() => 1;"
        top = [
            (code, math.log(0.5), 1),
            ("alt-a", math.log(0.15), 2),
            ("alt-b", math.log(0.1), 3),
            ("alt-c", math.log(0.08), 4),
            ("alt-d", math.log(0.07), 5),
        ]
        top_mass = math.fsum(math.exp(value[1]) for value in top)
        return {
            "schema": "qwen-direct-compact-mc-sequence-v1",
            "candidate_id": "a" * 64,
            "raw_content": code,
            "chosen_tokens_with_top_logprobs": [
                {
                    "token": code,
                    "bytes": list(code.encode()),
                    "logprob": math.log(0.5),
                    "top_logprobs": [
                        {
                            "token": token,
                            "bytes": list(token.encode()),
                            "logprob": logprob,
                        }
                        for token, logprob, _token_id in top
                    ],
                }
            ],
            "student_token_mapping_audit": {
                "summary": {
                    "chosen_bytes_reconstruct_raw_content": True,
                    "chosen_mapping_complete": True,
                    "top_mapping_complete": True,
                    "top5_count_complete": True,
                    "materially_negative_tail_positions": 0,
                    "logged_eos_covered": False,
                },
                "tokens": [
                    {
                        "chosen_student_token_id": 1,
                        "top_alternative_mappings": [
                            {"student_token_id": token_id, "mapping_error": None}
                            for _token, _logprob, token_id in top
                        ],
                        "tail_probability_mass_raw": 1.0 - top_mass,
                    }
                ],
            },
        }

    def test_exact_full_sequence_mapping_builds_content_only_payload(self):
        payload = _sparse_payload(
            self.teacher_row(),
            "int fn0() => 1;",
            tokenizer=self.tokenizer(),
            vocab_size=8,
            eos_token_id=6,
        )
        self.assertEqual(payload["target_token_ids"], [1])
        self.assertEqual(len(payload["teacher_positions"]), 1)
        self.assertFalse(
            payload["eos_policy"]["sparse_auxiliary_applied_to_eos"]
        )

    def test_trailing_whitespace_is_omitted_at_provider_token_boundary(self):
        row = self.teacher_row()
        code_position = row["chosen_tokens_with_top_logprobs"][0]
        code_audit = row["student_token_mapping_audit"]["tokens"][0]
        suffix = "\n"
        row["raw_content"] = "int fn0() => 1;" + suffix
        row["chosen_tokens_with_top_logprobs"] = [
            code_position,
            {"bytes": list(suffix.encode()), "top_logprobs": []},
        ]
        row["student_token_mapping_audit"]["tokens"] = [
            code_audit,
            {"chosen_student_token_id": 7},
        ]
        payload = _sparse_payload(
            row,
            "int fn0() => 1;",
            tokenizer=self.tokenizer(),
            vocab_size=8,
            eos_token_id=6,
        )
        self.assertEqual(payload["target_token_ids"], [1])
        self.assertEqual(len(payload["teacher_positions"]), 1)
        self.assertEqual(
            payload["target_alignment"]["leading_provider_tokens_omitted"], 0
        )
        self.assertEqual(
            payload["target_alignment"]["trailing_provider_tokens_omitted"], 1
        )

    def test_leading_whitespace_is_ineligible_even_at_token_boundary(self):
        row = self.teacher_row()
        code_position = row["chosen_tokens_with_top_logprobs"][0]
        code_audit = row["student_token_mapping_audit"]["tokens"][0]
        prefix = "\n"
        row["raw_content"] = prefix + "int fn0() => 1;"
        row["chosen_tokens_with_top_logprobs"] = [
            {"bytes": list(prefix.encode()), "top_logprobs": []},
            code_position,
        ]
        row["student_token_mapping_audit"]["tokens"] = [
            {"chosen_student_token_id": 7},
            code_audit,
        ]
        with self.assertRaisesRegex(IneligibleSparseDraw, "changes the teacher prefix"):
            _sparse_payload(
                row,
                "int fn0() => 1;",
                tokenizer=self.tokenizer(),
                vocab_size=8,
                eos_token_id=6,
            )

    def test_outer_whitespace_that_splits_provider_token_is_ineligible(self):
        row = self.teacher_row()
        raw = " " + row["raw_content"]
        row["raw_content"] = raw
        row["chosen_tokens_with_top_logprobs"][0]["bytes"] = list(raw.encode())
        with self.assertRaisesRegex(IneligibleSparseDraw, "cuts a provider token"):
            _sparse_payload(
                row,
                "int fn0() => 1;",
                tokenizer=self.tokenizer(),
                vocab_size=8,
                eos_token_id=6,
            )

    def test_response_transform_or_negative_tail_is_ineligible(self):
        with self.assertRaisesRegex(IneligibleSparseDraw, "normalization"):
            _sparse_payload(
                self.teacher_row(),
                "int fn0() => 2;",
                tokenizer=self.tokenizer(),
                vocab_size=8,
                eos_token_id=6,
            )
        row = self.teacher_row()
        row["student_token_mapping_audit"]["tokens"][0][
            "tail_probability_mass_raw"
        ] = -1e-12
        with self.assertRaisesRegex(IneligibleSparseDraw, "tail"):
            _sparse_payload(
                row,
                "int fn0() => 1;",
                tokenizer=self.tokenizer(),
                vocab_size=8,
                eos_token_id=6,
            )

    def test_attach_requires_exact_target_then_appended_eos(self):
        sparse = {
            "schema": SPARSE_ROW_SCHEMA,
            "target_token_ids": [1],
            "teacher_positions": [
                {
                    "observed_token_id": 1,
                    "top_token_ids": [1, 3],
                    "top_logprobs": [math.log(0.6), math.log(0.3)],
                    "tail_probability_mass": 0.1,
                }
            ],
            "target_alignment": {
                "transform": "trim_trailing_outer_whitespace",
                "trim_on_provider_token_boundaries": True,
                "leading_provider_tokens_omitted": 0,
                "trailing_provider_tokens_omitted": 0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text(
                json.dumps({"dart_source": "x", SPARSE_FIELD: sparse}) + "\n",
                encoding="utf-8",
            )
            dataset = type("Dataset", (), {"rows": [{"target_input_ids": [1, 6]}]})()
            report = attach_sparse_metadata(
                dataset,
                path,
                tokenizer=self.tokenizer(),
                eos_token_id=6,
                output_vocab_size=8,
                expected_rows_with_auxiliary=1,
                expected_sparse_positions=1,
            )
            self.assertEqual(report["sparse_positions"], 1)

    def test_manifest_forbids_dense_claim_and_requires_eos_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "train.jsonl"
            contract = root / "contract.json"
            tokenizer = root / "tokenizer.json"
            dataset.write_text("{}\n", encoding="utf-8")
            contract.write_text("{}\n", encoding="utf-8")
            tokenizer.write_text("{}\n", encoding="utf-8")
            manifest = {
                "schema": SPARSE_MANIFEST_SCHEMA,
                "dataset_sha256": sha256_file(dataset),
                "contract_sha256": sha256_file(contract),
                "student_tokenizer_json_sha256": sha256_file(tokenizer),
                "student_output_vocab_size": 8,
                "objective": "coarsened_topk_plus_tail_forward_kl",
                "sequence_monte_carlo_forward_kl_nll_primary": True,
                "dense_full_vocabulary_kl": True,
                "full_vocabulary_kd": False,
                "rows": 1,
                "rows_with_sparse_auxiliary": 1,
                "sparse_positions": 1,
                "eos_policy": {
                    "teacher_eos_distribution_available": False,
                    "sparse_auxiliary_applied_to_eos": False,
                    "student_eos_supervised_by_primary_sequence_nll": True,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dense-KL"):
                validate_sparse_manifest(
                    dataset,
                    manifest_path,
                    contract_path=contract,
                    tokenizer_json_path=tokenizer,
                )

    def test_offline_builder_attaches_only_sealed_direct_compact_auxiliary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = self.tokenizer()
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            tokenizer_sha = sha256_file(tokenizer_path)
            contract = DirectCompactContract(
                schema=CONTRACT_SCHEMA,
                codec_sha256="a" * 64,
                codebook_sha256="b" * 64,
                tokenizer_json_sha256=tokenizer_sha,
                tokenizer_fingerprint_sha256="c" * 64,
                model_config_sha256="d" * 64,
                decoder_model="fake/student",
                decoder_revision="immutable",
                target_function="fn0",
                target_language="Dart",
                dfg_extractor_sha256="e" * 64,
                lossless_domain="scrubbed_canonical_graph",
                base_vocab_size=8,
                source_token_ids=(8,),
                source_token_expansions=((8, (1,)),),
            )
            contract_path = root / "contract.json"
            contract_path.write_text(
                json.dumps(contract.as_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            sequence_row = {
                "lang": "Dart",
                "function": "fn0",
                "dart_source": "int fn0() => 1;",
                "compact_input_ids": [8],
                "compact_codec_sha256": contract.codec_sha256,
                "compact_codebook_sha256": contract.codebook_sha256,
                "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
            }
            sequence = root / "sequence.jsonl"
            sequence.write_text(json.dumps(sequence_row) + "\n", encoding="utf-8")
            sequence_seal = root / "sequence.seal.json"
            sequence_seal.write_text(
                json.dumps(
                    {
                        "schema": "compact-public-private-join-seal-v1",
                        "selected_role": "fit",
                        "output_sha256": sha256_file(sequence),
                        "contract_sha256": sha256_file(contract_path),
                        "rows": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            teacher_row = self.teacher_row()
            teacher = root / "teacher.jsonl"
            teacher.write_text(json.dumps(teacher_row) + "\n", encoding="utf-8")
            schedule = root / "schedule.jsonl"
            schedule.write_text(
                json.dumps(
                    {
                        "kind": "teacher_draw",
                        "candidate_id": teacher_row["candidate_id"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            identity = {
                "requested_model": "qwen3.8-max-preview",
                "returned_model": "qwen3.8-max-preview-202607",
                "system_fingerprint": "backend-a",
            }
            identity_sha = stable_sha256(identity)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema": "qwen-direct-compact-teacher-audit-v1",
                        "objective_mode": "require_top5",
                        "production_readiness": {
                            "sparse_top5_plus_tail": True
                        },
                        "capabilities": {
                            "content_logprob_prefix_fully_visible_to_student": True
                        },
                        "student_tokenizer": file_record(tokenizer_path),
                        "homogeneous_backend_shards": [
                            {
                                "backend_identity_sha256": identity_sha,
                                "backend_identity": identity,
                                "parseable_output": file_record(teacher),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sequence_manifest = root / "sequence.build.json"
            sequence_manifest.write_text(
                json.dumps(
                    {
                        "schema": (
                            "direct-compact-mc-sequence-forward-kl-nll-build-v1"
                        ),
                        "objective": {"dense_token_kl": False},
                        "outputs": {
                            "dataset": file_record(sequence),
                            "schedule": file_record(schedule),
                        },
                        "inputs": {
                            "teacher_parseable": file_record(teacher),
                            "teacher_audit": file_record(audit),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            attestation = root / "attestation.json"
            attestation.write_text(
                json.dumps(
                    {
                        "schema": "qwen-provider-shared-tokenizer-attestation-v1",
                        "backend_identity_sha256": identity_sha,
                        "requested_model": identity["requested_model"],
                        "returned_model": identity["returned_model"],
                        "teacher_tokenizer_json_sha256": tokenizer_sha,
                        "student_tokenizer_json_sha256": tokenizer_sha,
                        "student_eos_token_id": 6,
                        "teacher_output_vocab_size": 8,
                        "student_output_vocab_size": 8,
                        "tokenizer_files_byte_identical": True,
                        "full_vocabulary_identical": True,
                        "segmentation_rules_identical": True,
                        "special_token_ids_identical": True,
                        "provider_model_tokenizer_binding_attested": True,
                        "provider_model_tokenizer_binding_source": (
                            "sealed provider release artifact"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "sparse.jsonl"
            output_seal = root / "sparse.seal.json"
            output_manifest = root / "sparse.manifest.json"
            args = argparse.Namespace(
                minimum_eligible_fraction=1.0,
                contract=contract_path,
                sequence_train_jsonl=sequence,
                sequence_train_seal=sequence_seal,
                sequence_schedule_jsonl=schedule,
                sequence_build_manifest=sequence_manifest,
                expected_sequence_build_manifest_sha256=sha256_file(
                    sequence_manifest
                ),
                teacher_parseable_jsonl=teacher,
                expected_teacher_parseable_sha256=sha256_file(teacher),
                teacher_audit_json=audit,
                expected_teacher_audit_sha256=sha256_file(audit),
                student_tokenizer_json=tokenizer_path,
                expected_student_tokenizer_sha256=tokenizer_sha,
                teacher_tokenizer_json=tokenizer_path,
                expected_teacher_tokenizer_sha256=tokenizer_sha,
                tokenizer_attestation=attestation,
                expected_tokenizer_attestation_sha256=sha256_file(attestation),
                student_eos_token_id=6,
                output_jsonl=output,
                output_seal=output_seal,
                output_manifest=output_manifest,
            )
            manifest = build_sparse_artifact(args)
            self.assertEqual(manifest["rows_with_sparse_auxiliary"], 1)
            self.assertFalse(manifest["dense_full_vocabulary_kl"])
            self.assertFalse(
                manifest["global_provider_tokenizer_identity_claimed"]
            )
            self.assertIn(SPARSE_FIELD, read_jsonl(output)[0])
            validate_join_seal(
                output, output_seal, contract_path, expected_role="fit"
            )


if __name__ == "__main__":
    unittest.main()
