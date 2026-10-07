from __future__ import annotations

import ast
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from models.direct_compact_causal import (
    COMPOSITE_REPRESENTATION_V1,
    GRAPH_LITERAL_OMISSION_POLICY_V1,
    CONTRACT_SCHEMA,
    CONTRACT_SCHEMA_V2,
    CONTRACT_SCHEMA_V3,
    JOIN_SEAL_SCHEMA_V2,
    LOSSLESS_DOMAIN_V3,
    POOL_ALIGNMENT_SCHEMA_V1,
    POOL_ENCODING_V1,
    POOL_POSITIONAL_ENCODING_V2,
    POOL_PAYLOAD_SCHEMA_V1,
    POOL_PROJECTION_V1,
    POOL_SCOPE_V1,
    NON_GRAPH_AOT_XREF_POLICY_V1,
    NESTED_NONLITERAL_DESCRIPTORS_V1,
    DirectCompactBatchCollator,
    DirectCompactCausalLM,
    DirectCompactContract,
    SourceTokenEmbeddingOverlay,
    canonical_v3_pool_json,
    install_source_embedding_overlay,
    matched_permutation_indices,
    migrate_source_embedding_overlay,
    per_sequence_causal_nll,
    per_sequence_causal_nll_sum,
    restore_source_embedding_overlay,
    sha256_artifact,
    sha256_file,
    tokenizer_fingerprint,
    validate_base_model_vocab,
    validate_join_seal,
    validate_v3_pool_alignment_metadata,
)
from scripts.training.join_compact_public_private import build_join
from scripts.training.direct_compact_qwen_decompiler import (
    DIRECT_TRAINER_RESUME_FILENAME,
    copy_exact_contract,
    make_direct_trainer_class,
    materialize_overlay_migrated_checkpoint,
    validate_direct_trainer_resume_checkpoint,
    validate_overlay_migration_contracts,
    validate_overlay_migrated_checkpoint,
    validate_self_sealed_checkpoint,
    validate_warmstart_checkpoint,
)
from scripts.evaluation.direct_compact_qwen_inference import load_rows as load_inference_rows
from scripts.evaluation.validate_direct_compact_training_stage import validate_stage


class FakeTokenizer:
    def __init__(self) -> None:
        # Compact source ID 4 is deliberately external to the base tokenizer.
        self._vocab = {"<pad>": 0, "<eos>": 1, "a": 2, "b": 3}
        self.special_tokens_map = {"pad_token": "<pad>", "eos_token": "<eos>"}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = None

    def get_vocab(self):
        return dict(self._vocab)


class FakeCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 4, hidden_size: int = 6) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = types.SimpleNamespace(vocab_size=vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self, input_ids=None, inputs_embeds=None, attention_mask=None,
        labels=None, **kwargs
    ):
        logits_to_keep = int(kwargs.pop("logits_to_keep", 0) or 0)
        del attention_mask, kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.lm_head(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


def make_contract(tokenizer: FakeTokenizer) -> DirectCompactContract:
    digest = "a" * 64
    return DirectCompactContract(
        schema=CONTRACT_SCHEMA,
        codec_sha256=digest,
        codebook_sha256=digest,
        tokenizer_json_sha256="b" * 64,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(tokenizer),
        model_config_sha256="c" * 64,
        decoder_model="fake/decoder",
        decoder_revision="revision-1",
        target_function="fn0",
        target_language="Dart",
        dfg_extractor_sha256="d" * 64,
        lossless_domain="scrubbed_canonical_graph",
        max_source_tokens=9000,
        max_target_tokens=16,
        max_total_tokens=12000,
        base_vocab_size=4,
        source_token_ids=(4,),
        source_token_expansions=((4, (2, 3)),),
    )


def make_v3_contract(tokenizer: FakeTokenizer) -> DirectCompactContract:
    digest = "a" * 64
    routes = DirectCompactContractTests.v2_routes()
    return DirectCompactContract(
        schema=CONTRACT_SCHEMA_V3,
        codec_sha256=digest,
        codebook_sha256="b" * 64,
        tokenizer_json_sha256="c" * 64,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(tokenizer),
        model_config_sha256="d" * 64,
        decoder_model="fake/decoder",
        decoder_revision="revision-3",
        target_function="candidate",
        target_language="Dart",
        dfg_extractor_sha256=None,
        extractor_routes=routes,
        runtime_symbol_policy_sha256="e" * 64,
        pool_extractor_sha256="f" * 64,
        dart_toolchain_manifest_sha256="1" * 64,
        aot_manifest_sha256="2" * 64,
        pool_reconciliation_manifest_sha256="4" * 64,
        graph_codec_dependency_sha256="3" * 64,
        target_architecture="x86_64",
        pool_schema=POOL_PAYLOAD_SCHEMA_V1,
        pool_encoding=POOL_ENCODING_V1,
        pool_positional_encoding=POOL_POSITIONAL_ENCODING_V2,
        pool_scope=POOL_SCOPE_V1,
        pool_projection=POOL_PROJECTION_V1,
        all_encoded_pool_uses_reference_canonical_graph_instructions=True,
        raw_disassembly_unreachable_islands_in_lossless_domain=False,
        non_graph_aot_xrefs=NON_GRAPH_AOT_XREF_POLICY_V1,
        graph_retained_literal_use_omission_policy=(
            GRAPH_LITERAL_OMISSION_POLICY_V1
        ),
        pool_order_and_duplicates_preserved=True,
        string_representation="ordered-dart-utf16-code-units",
        integer_representation="canonical-signed-decimal",
        double_representation="exact-ieee754-binary64-bits-lower-hex",
        composite_representation=COMPOSITE_REPRESENTATION_V1,
        nested_nonliteral_descriptors=NESTED_NONLITERAL_DESCRIPTORS_V1,
        stream_marker_ids={
            "<G2C3>": 4,
            "<CFG>": 5,
            "<PX0>": 6,
            "<PEND>": 7,
            "<END>": 8,
        },
        lossless_domain=LOSSLESS_DOMAIN_V3,
        base_vocab_size=4,
        source_token_ids=(4, 5, 6, 7, 8),
        source_token_expansions=tuple(
            (token_id, (2,)) for token_id in range(4, 9)
        ),
    )


def valid_v3_pool_metadata(*, use_count: int = 1) -> dict[str, object]:
    return {
        "schema": POOL_ALIGNMENT_SCHEMA_V1,
        "receipt_sha256": "3" * 64,
        "projection_sha256": "4" * 64,
        "use_count": use_count,
        "source_blind": True,
        "target_function": "candidate",
    }


class DirectCompactContractTests(unittest.TestCase):
    @staticmethod
    def v2_routes() -> dict[str, dict[str, object]]:
        return {
            "current_combined_v2": {
                "allow_call_edges": True,
                "cfg_extractor_sha256": "1" * 64,
                "combined_hash_algorithm": "sha256(filename || bytes)",
                "dfg_extractor_sha256": "2" * 64,
                "dfg_metadata": "locations_and_dependency_count",
                "graph_extractor_sha256": "3" * 64,
                "route_atom": "<DX1>",
            },
            "legacy_release_v1": {
                "allow_call_edges": False,
                "cfg_extractor_sha256": "4" * 64,
                "combined_hash_algorithm": "sha256(cfg || dfg)",
                "dfg_extractor_sha256": "5" * 64,
                "dfg_metadata": "endpoints_only",
                "graph_extractor_sha256": "6" * 64,
                "route_atom": "<DX0>",
            },
        }

    def test_external_source_ids_are_absent_from_base_tokenizer(self) -> None:
        tokenizer = FakeTokenizer()
        contract = make_contract(tokenizer)
        self.assertNotIn(4, tokenizer.get_vocab().values())
        contract.validate_artifacts(tokenizer=tokenizer)

    def test_tokenizer_vocab_may_be_smaller_than_reserved_model_vocab(self) -> None:
        tokenizer = FakeTokenizer()  # 4 exposed tokenizer entries
        digest = "a" * 64
        contract = DirectCompactContract(
            schema=CONTRACT_SCHEMA,
            codec_sha256=digest,
            codebook_sha256=digest,
            tokenizer_json_sha256="b" * 64,
            tokenizer_fingerprint_sha256=tokenizer_fingerprint(tokenizer),
            model_config_sha256="c" * 64,
            decoder_model="fake/decoder",
            decoder_revision="revision-1",
            target_function="fn0",
            target_language="Dart",
            dfg_extractor_sha256="d" * 64,
            lossless_domain="scrubbed_canonical_graph",
            base_vocab_size=6,
            source_token_ids=(6,),
            source_token_expansions=((6, (2, 3)),),
        )
        model = FakeCausalLM(vocab_size=6)
        contract.validate_artifacts(tokenizer=tokenizer)
        self.assertEqual(validate_base_model_vocab(model, contract), 6)
        overlay = install_source_embedding_overlay(
            model, dict(contract.source_token_expansions), base_vocab_size=6
        )
        self.assertEqual(overlay.source_token_ids, (6,))
        self.assertEqual(model.lm_head.weight.size(0), 6)

    def test_contract_and_row_hashes_are_fail_closed(self) -> None:
        tokenizer = FakeTokenizer()
        contract = make_contract(tokenizer)
        row = {
            "compact_input_ids": [4, 2],
            "compact_codec_sha256": contract.codec_sha256,
            "compact_codebook_sha256": contract.codebook_sha256,
            "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
        }
        self.assertEqual(contract.validate_row(row, "r0"), [4, 2])
        row["compact_codebook_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "codebook_sha256 mismatch"):
            contract.validate_row(row, "r0")

    def test_source_gate_is_exactly_no_truncation(self) -> None:
        tokenizer = FakeTokenizer()
        contract = make_contract(tokenizer)
        row = {
            "compact_input_ids": [2] * 9001,
            "compact_codec_sha256": contract.codec_sha256,
            "compact_codebook_sha256": contract.codebook_sha256,
            "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
        }
        with self.assertRaisesRegex(ValueError, "refusing truncation|no-truncation"):
            contract.validate_row(row, "too-long")

    def test_tokenizer_and_artifact_hashes_are_checked(self) -> None:
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            codec = Path(directory) / "codec.py"
            codebook = Path(directory) / "codebook.json"
            tokenizer_json = Path(directory) / "tokenizer.json"
            codec.write_text("codec", encoding="utf-8")
            tokenizer_json.write_text('{"base":true}', encoding="utf-8")
            model_config_sha256 = "c" * 64
            decoder_model = "fake/decoder"
            decoder_revision = "revision-1"
            codebook.write_text(
                json.dumps(
                    {
                        "tokenizer_json_sha256": sha256_file(tokenizer_json),
                        "model_config_sha256": model_config_sha256,
                        "decoder_model": decoder_model,
                        "decoder_revision": decoder_revision,
                        "base_vocab_size": 4,
                        "dfg_extractor_sha256": "d" * 64,
                        "source_token_expansions": {"4": [2]},
                    }
                ),
                encoding="utf-8",
            )
            contract = DirectCompactContract(
                schema=CONTRACT_SCHEMA,
                codec_sha256=sha256_file(codec),
                codebook_sha256=sha256_file(codebook),
                tokenizer_json_sha256=sha256_file(tokenizer_json),
                tokenizer_fingerprint_sha256=tokenizer_fingerprint(tokenizer),
                model_config_sha256=model_config_sha256,
                decoder_model=decoder_model,
                decoder_revision=decoder_revision,
                target_function="fn0",
                target_language="Dart",
                dfg_extractor_sha256="d" * 64,
                lossless_domain="scrubbed_canonical_graph",
                base_vocab_size=4,
                source_token_ids=(4,),
                source_token_expansions=((4, (2,)),),
            )
            contract.validate_artifacts(
                tokenizer=tokenizer,
                tokenizer_json_path=tokenizer_json,
                codec_path=codec,
                codebook_path=codebook,
            )
            codebook.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "codebook SHA-256 mismatch"):
                contract.validate_artifacts(
                    tokenizer=tokenizer,
                    tokenizer_json_path=tokenizer_json,
                    codec_path=codec,
                    codebook_path=codebook,
                )

    def test_json_contract_round_trip_preserves_expansions(self) -> None:
        original = make_contract(FakeTokenizer())
        rebuilt = DirectCompactContract.from_mapping(original.as_dict())
        self.assertEqual(rebuilt, original)

    def test_v2_contract_round_trip_preserves_versioned_extractors(self) -> None:
        tokenizer = FakeTokenizer()
        value = {
            "schema": CONTRACT_SCHEMA_V2,
            "codec_sha256": "a" * 64,
            "codebook_sha256": "b" * 64,
            "tokenizer_json_sha256": "c" * 64,
            "tokenizer_fingerprint_sha256": tokenizer_fingerprint(tokenizer),
            "model_config_sha256": "d" * 64,
            "decoder_model": "fake/decoder",
            "decoder_revision": "revision-2",
            "target_function": "candidate",
            "target_language": "Dart",
            "extractor_routes": self.v2_routes(),
            "runtime_symbol_policy_sha256": "e" * 64,
            "lossless_domain": "scrubbed_canonical_graph_v2",
            "base_vocab_size": 4,
            "source_token_ids": [4],
            "source_token_expansions": {"4": [2, 3]},
        }
        contract = DirectCompactContract.from_mapping(value)
        self.assertEqual(contract.as_dict(), value | {
            "max_source_tokens": 9000,
            "max_target_tokens": 3072,
            "max_total_tokens": 12288,
            "source_embedding_init": "codebook_mean",
        })
        self.assertEqual(
            DirectCompactContract.from_mapping(contract.as_dict()), contract
        )

    def test_v2_contract_rejects_single_or_implicit_extractor(self) -> None:
        tokenizer = FakeTokenizer()
        base = {
            "schema": CONTRACT_SCHEMA_V2,
            "codec_sha256": "a" * 64,
            "codebook_sha256": "b" * 64,
            "tokenizer_json_sha256": "c" * 64,
            "tokenizer_fingerprint_sha256": tokenizer_fingerprint(tokenizer),
            "model_config_sha256": "d" * 64,
            "decoder_model": "fake/decoder",
            "decoder_revision": "revision-2",
            "target_function": "candidate",
            "target_language": "Dart",
            "runtime_symbol_policy_sha256": "e" * 64,
            "lossless_domain": "scrubbed_canonical_graph_v2",
        }
        with self.assertRaisesRegex(ValueError, "both extractor routes"):
            DirectCompactContract.from_mapping(
                base | {"extractor_routes": {"only": next(iter(self.v2_routes().values()))}}
            )
        with self.assertRaisesRegex(ValueError, "use extractor_routes"):
            DirectCompactContract.from_mapping(
                base
                | {
                    "extractor_routes": self.v2_routes(),
                    "dfg_extractor_sha256": "f" * 64,
                }
            )

    def test_v3_contract_round_trip_and_marker_validation(self) -> None:
        contract = make_v3_contract(FakeTokenizer())
        rebuilt = DirectCompactContract.from_mapping(contract.as_dict())
        self.assertEqual(rebuilt, contract)
        valid = {
            "compact_input_ids": [4, 5, 6, 2, 7, 8],
            "compact_codec_sha256": contract.codec_sha256,
            "compact_codebook_sha256": contract.codebook_sha256,
            "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
        }
        self.assertEqual(contract.validate_row(valid, "v3-ok"), valid["compact_input_ids"])

        malformed = dict(valid)
        malformed["compact_input_ids"] = [4, 5, 6, 7, 8]
        with self.assertRaisesRegex(ValueError, "pool JSON payload may not be empty"):
            contract.validate_row(malformed, "v3-empty")

        malformed["compact_input_ids"] = [4, 5, 6, 4, 7, 8]
        with self.assertRaisesRegex(ValueError, "must occur exactly once|base-tokenizer"):
            contract.validate_row(malformed, "v3-source-in-payload")

        malformed["compact_input_ids"] = [4, 6, 2, 5, 7, 8]
        with self.assertRaisesRegex(ValueError, "invalid v3 marker order"):
            contract.validate_row(malformed, "v3-order")

    def test_v3_contract_requires_binary_pool_bindings(self) -> None:
        value = make_v3_contract(FakeTokenizer()).as_dict()
        del value["pool_extractor_sha256"]
        with self.assertRaisesRegex(ValueError, "pool_extractor_sha256"):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        del value["pool_reconciliation_manifest_sha256"]
        with self.assertRaisesRegex(
            ValueError, "pool_reconciliation_manifest_sha256"
        ):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        value["pool_reconciliation_manifest_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(
            ValueError, "pool_reconciliation_manifest_sha256"
        ):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        value["pool_scope"] = "whole-disassembly"
        with self.assertRaisesRegex(ValueError, "pool_scope must be"):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        value["raw_disassembly_unreachable_islands_in_lossless_domain"] = True
        with self.assertRaisesRegex(ValueError, "must be false"):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        del value["nested_nonliteral_descriptors"]
        with self.assertRaisesRegex(ValueError, "nested_nonliteral_descriptors"):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        value["nested_nonliteral_descriptors"][
            "profile_type_to_nonliteral_kind"
        ]["SecretClass"] = "runtime_object"
        with self.assertRaisesRegex(ValueError, "finite source-blind descriptor"):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        del value["pool_positional_encoding"]
        with self.assertRaisesRegex(ValueError, "pool_positional_encoding"):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        value["pool_positional_encoding"]["kind_to_tag"]["string"] = 99
        with self.assertRaisesRegex(ValueError, "positional grammar"):
            DirectCompactContract.from_mapping(value)
        value = make_v3_contract(FakeTokenizer()).as_dict()
        value["stream_marker_ids"] = value["stream_marker_ids"] | {"<OTHER>": 9}
        with self.assertRaisesRegex(ValueError, "stream_marker_ids field mismatch"):
            DirectCompactContract.from_mapping(value)

    def test_v3_pool_payload_must_decode_to_canonical_ascii_json(self) -> None:
        contract = make_v3_contract(FakeTokenizer())
        ids = [4, 5, 6, 2, 7, 8]

        class Decoder:
            def __init__(self, value: str) -> None:
                self.value = value

            def decode(self, token_ids, **kwargs):
                del kwargs
                self.last_ids = token_ids
                return self.value

        canonical_pool = {
            "schema": POOL_PAYLOAD_SCHEMA_V1,
            "target_function": "candidate",
            "uses": [],
        }
        canonical_text = canonical_v3_pool_json(canonical_pool)
        decoder = Decoder(canonical_text)
        self.assertEqual(
            contract.validate_v3_pool_payload(ids, decoder, "canonical"),
            canonical_pool,
        )
        self.assertEqual(decoder.last_ids, [2])
        with self.assertRaisesRegex(ValueError, "not canonical"):
            contract.validate_v3_pool_payload(
                ids,
                Decoder(canonical_text.replace("[", "[ ", 1)),
                "spaced",
            )
        with self.assertRaisesRegex(ValueError, "positional pool root"):
            contract.validate_v3_pool_payload(ids, Decoder("{}"), "object")

    def test_v3_pool_payload_accepts_complete_recursive_composite_exactly(self) -> None:
        contract = make_v3_contract(FakeTokenizer())
        ids = [4, 5, 6, 2, 7, 8]
        composite = {
            "schema": POOL_PAYLOAD_SCHEMA_V1,
            "target_function": "candidate",
            "uses": [
                {
                    "pp_offset": 1935,
                    "kind": "composite",
                    "payload": {
                        "complete": True,
                        "composite_type": "array_storage",
                        "elements": [
                            {
                                "index": 0,
                                "value": {
                                    "kind": "int",
                                    "payload": {"decimal": "1"},
                                },
                            },
                            {
                                "index": 0,
                                "value": {
                                    "kind": "composite",
                                    "payload": {
                                        "complete": True,
                                        "composite_type": "map_storage",
                                        "elements": [
                                            {
                                                "index": 0,
                                                "value": {
                                                    "kind": "string",
                                                    "payload": {
                                                        "code_units": [107, 101, 121]
                                                    },
                                                },
                                            },
                                            {
                                                "index": 1,
                                                "value": {
                                                    "kind": "null",
                                                    "payload": {},
                                                },
                                            },
                                            {
                                                "index": 2,
                                                "value": {
                                                    "kind": "nonliteral",
                                                    "payload": {
                                                        "nonliteral_kind": "type_metadata",
                                                        "profile_type": "TypeArguments",
                                                    },
                                                },
                                            },
                                            {
                                                "index": 3,
                                                "value": {
                                                    "kind": "nonliteral",
                                                    "payload": {
                                                        "nonliteral_kind": "runtime_object",
                                                        "profile_type": "Instance",
                                                    },
                                                },
                                            },
                                        ],
                                        "omitted_edge_counts": {"weak": 1},
                                    },
                                },
                            },
                        ],
                        "omitted_edge_counts": {"property": 2, "internal": 1},
                    },
                    "use_sites": [{"block": 0, "instruction": 6}],
                }
            ],
        }

        class Decoder:
            def decode(self, token_ids, **kwargs):
                del token_ids, kwargs
                return canonical_v3_pool_json(composite)

        validated = contract.validate_v3_pool_payload(
            ids, Decoder(), "composite"
        )
        payload = validated["uses"][0]["payload"]
        self.assertEqual([item["index"] for item in payload["elements"]], [0, 0])
        self.assertEqual(
            payload["elements"][1]["value"]["payload"]["composite_type"],
            "map_storage",
        )
        nested_elements = payload["elements"][1]["value"]["payload"]["elements"]
        self.assertEqual(
            nested_elements[2]["value"],
            {
                "kind": "nonliteral",
                "payload": {
                    "nonliteral_kind": "type_metadata",
                    "profile_type": "TypeArguments",
                },
            },
        )
        self.assertEqual(
            payload["omitted_edge_counts"], {"internal": 1, "property": 2}
        )

    def test_v3_pool_payload_rejects_incomplete_or_unresolved_composite(self) -> None:
        contract = make_v3_contract(FakeTokenizer())
        ids = [4, 5, 6, 2, 7, 8]

        def envelope(payload):
            return {
                "schema": POOL_PAYLOAD_SCHEMA_V1,
                "target_function": "candidate",
                "uses": [
                    {
                        "pp_offset": 1,
                        "kind": "composite",
                        "payload": payload,
                        "use_sites": [{"block": 0, "instruction": 0}],
                    }
                ],
            }

        class Decoder:
            def __init__(self, value):
                self.value = value

            def decode(self, token_ids, **kwargs):
                del token_ids, kwargs
                return canonical_v3_pool_json(self.value)

        incomplete = {
            "complete": False,
            "composite_type": "array_storage",
            "elements": [],
            "omitted_edge_counts": {},
        }
        with self.assertRaisesRegex(ValueError, "composite must be complete"):
            contract.validate_v3_pool_payload(
                ids, Decoder(envelope(incomplete)), "incomplete"
            )
        unresolved = {
            "complete": True,
            "composite_type": "array_storage",
            "elements": [
                {
                    "index": 0,
                    "value": {"kind": "reference", "payload": {}},
                }
            ],
            "omitted_edge_counts": {},
        }
        with self.assertRaisesRegex(ValueError, "unsupported literal kind"):
            contract.validate_v3_pool_payload(
                ids, Decoder(envelope(unresolved)), "unresolved"
            )

    def test_v3_pool_payload_restricts_nonliteral_to_source_blind_nested_pairs(self) -> None:
        contract = make_v3_contract(FakeTokenizer())
        ids = [4, 5, 6, 2, 7, 8]

        class Decoder:
            def __init__(self, value):
                self.value = value

            def decode(self, token_ids, **kwargs):
                del token_ids, kwargs
                return canonical_v3_pool_json(self.value)

        def envelope(kind, payload):
            return {
                "schema": POOL_PAYLOAD_SCHEMA_V1,
                "target_function": "candidate",
                "uses": [
                    {
                        "pp_offset": 1,
                        "kind": kind,
                        "payload": payload,
                        "use_sites": [{"block": 0, "instruction": 0}],
                    }
                ],
            }

        descriptor = {
            "nonliteral_kind": "runtime_object",
            "profile_type": "Instance",
        }
        with self.assertRaisesRegex(ValueError, "top-level nonliteral"):
            contract.validate_v3_pool_payload(
                ids, Decoder(envelope("nonliteral", descriptor)), "top-level"
            )

        def nested(payload):
            return envelope(
                "composite",
                {
                    "complete": True,
                    "composite_type": "array_storage",
                    "elements": [
                        {
                            "index": 0,
                            "value": {"kind": "nonliteral", "payload": payload},
                        }
                    ],
                    "omitted_edge_counts": {},
                },
            )

        for profile_type in ("Instance", "Record"):
            contract.validate_v3_pool_payload(
                ids,
                Decoder(
                    nested(
                        {
                            "nonliteral_kind": "runtime_object",
                            "profile_type": profile_type,
                        }
                    )
                ),
                f"allowed-{profile_type}",
            )

        bad_values = (
            descriptor | {"name": "SecretClass"},
            {"nonliteral_kind": "callable", "profile_type": "TypeArguments"},
            {"nonliteral_kind": "runtime_object", "profile_type": "UnknownClass"},
            {"nonliteral_kind": "runtime_object", "profile_type": "0x1234"},
        )
        for index, bad in enumerate(bad_values):
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "nonliteral payload fields|descriptor pair"
            ):
                contract.validate_v3_pool_payload(
                    ids, Decoder(nested(bad)), f"bad-{index}"
                )

    def test_v3_pool_payload_rejects_excessive_composite_depth(self) -> None:
        contract = make_v3_contract(FakeTokenizer())
        ids = [4, 5, 6, 2, 7, 8]
        nested = {"kind": "null", "payload": {}}
        for _ in range(66):
            nested = {
                "kind": "composite",
                "payload": {
                    "complete": True,
                    "composite_type": "array_storage",
                    "elements": [{"index": 0, "value": nested}],
                    "omitted_edge_counts": {},
                },
            }
        envelope = {
            "schema": POOL_PAYLOAD_SCHEMA_V1,
            "target_function": "candidate",
            "uses": [
                {
                    "pp_offset": 1,
                    "kind": nested["kind"],
                    "payload": nested["payload"],
                    "use_sites": [{"block": 0, "instruction": 0}],
                }
            ],
        }

        class Decoder:
            def decode(self, token_ids, **kwargs):
                del token_ids, kwargs
                return canonical_v3_pool_json(envelope)

        with self.assertRaisesRegex(ValueError, "composite depth limit"):
            contract.validate_v3_pool_payload(ids, Decoder(), "deep")

    def test_v3_pool_payload_applies_depth_limit_to_nonliteral_terminal(self) -> None:
        contract = make_v3_contract(FakeTokenizer())
        ids = [4, 5, 6, 2, 7, 8]
        nested = {
            "kind": "nonliteral",
            "payload": {
                "nonliteral_kind": "runtime_object",
                "profile_type": "Instance",
            },
        }
        # The outer composite is depth zero.  Sixty-five wrappers therefore
        # place this otherwise-valid terminal descriptor at depth 65.
        for _ in range(65):
            nested = {
                "kind": "composite",
                "payload": {
                    "complete": True,
                    "composite_type": "array_storage",
                    "elements": [{"index": 0, "value": nested}],
                    "omitted_edge_counts": {},
                },
            }
        envelope = {
            "schema": POOL_PAYLOAD_SCHEMA_V1,
            "target_function": "candidate",
            "uses": [
                {
                    "pp_offset": 1,
                    "kind": nested["kind"],
                    "payload": nested["payload"],
                    "use_sites": [{"block": 0, "instruction": 0}],
                }
            ],
        }

        class Decoder:
            def decode(self, token_ids, **kwargs):
                del token_ids, kwargs
                return canonical_v3_pool_json(envelope)

        with self.assertRaisesRegex(ValueError, "composite depth limit"):
            contract.validate_v3_pool_payload(ids, Decoder(), "deep-nonliteral")

    def test_v3_codebook_must_bind_schema_pool_and_marker_ids(self) -> None:
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_json = root / "tokenizer.json"
            codec = root / "codec.py"
            codebook = root / "codebook.json"
            tokenizer_json.write_text("{}", encoding="utf-8")
            codec.write_text("codec-v3", encoding="utf-8")
            base = make_v3_contract(tokenizer).as_dict()
            base["codec_sha256"] = sha256_file(codec)
            base["tokenizer_json_sha256"] = sha256_file(tokenizer_json)
            codebook_value = {
                "schema": "compact-qwen-v3-codebook",
                "tokenizer_json_sha256": base["tokenizer_json_sha256"],
                "model_config_sha256": base["model_config_sha256"],
                "decoder_model": base["decoder_model"],
                "decoder_revision": base["decoder_revision"],
                "base_vocab_size": base["base_vocab_size"],
                "runtime_symbol_policy_sha256": base["runtime_symbol_policy_sha256"],
                "extractor_routes": base["extractor_routes"],
                "pool_extractor_sha256": base["pool_extractor_sha256"],
                "dart_toolchain_manifest_sha256": base[
                    "dart_toolchain_manifest_sha256"
                ],
                "aot_manifest_sha256": base["aot_manifest_sha256"],
                "pool_reconciliation_manifest_sha256": base[
                    "pool_reconciliation_manifest_sha256"
                ],
                "graph_codec_dependency_sha256": base[
                    "graph_codec_dependency_sha256"
                ],
                "target_architecture": base["target_architecture"],
                "pool_schema": base["pool_schema"],
                "pool_encoding": base["pool_encoding"],
                "pool_positional_encoding": base["pool_positional_encoding"],
                "pool_scope": base["pool_scope"],
                "pool_projection": base["pool_projection"],
                "all_encoded_pool_uses_reference_canonical_graph_instructions": base[
                    "all_encoded_pool_uses_reference_canonical_graph_instructions"
                ],
                "raw_disassembly_unreachable_islands_in_lossless_domain": base[
                    "raw_disassembly_unreachable_islands_in_lossless_domain"
                ],
                "non_graph_aot_xrefs": base["non_graph_aot_xrefs"],
                "graph_retained_literal_use_omission_policy": base[
                    "graph_retained_literal_use_omission_policy"
                ],
                "pool_order_and_duplicates_preserved": base[
                    "pool_order_and_duplicates_preserved"
                ],
                "string_representation": base["string_representation"],
                "integer_representation": base["integer_representation"],
                "double_representation": base["double_representation"],
                "composite_representation": base["composite_representation"],
                "nested_nonliteral_descriptors": base[
                    "nested_nonliteral_descriptors"
                ],
                "source_atom_ids": base["stream_marker_ids"],
                "source_token_expansions": base["source_token_expansions"],
            }
            codebook.write_text(json.dumps(codebook_value), encoding="utf-8")
            base["codebook_sha256"] = sha256_file(codebook)
            contract = DirectCompactContract.from_mapping(base)
            contract.validate_artifacts(
                tokenizer=tokenizer,
                tokenizer_json_path=tokenizer_json,
                codec_path=codec,
                codebook_path=codebook,
            )
            codebook_value["source_atom_ids"]["<PX0>"] = 99
            codebook.write_text(json.dumps(codebook_value), encoding="utf-8")
            changed = contract.as_dict()
            changed["codebook_sha256"] = sha256_file(codebook)
            changed_contract = DirectCompactContract.from_mapping(changed)
            with self.assertRaisesRegex(ValueError, "stream marker IDs mismatch"):
                changed_contract.validate_artifacts(
                    tokenizer=tokenizer,
                    tokenizer_json_path=tokenizer_json,
                    codec_path=codec,
                    codebook_path=codebook,
                )


class DirectCompactCollatorTests(unittest.TestCase):
    def test_prompt_and_source_are_masked_and_padding_is_terminal(self) -> None:
        collator = DirectCompactBatchCollator(
            pad_token_id=0,
            max_source_tokens=4,
            max_target_tokens=4,
            max_total_tokens=12,
            source_token_ids=(4,),
        )
        batch = collator(
            [
                {
                    "decoder_prompt_input_ids": [2, 3],
                    "compact_input_ids": [4, 2],
                    "target_input_ids": [3, 1],
                },
                {
                    "decoder_prompt_input_ids": [2],
                    "compact_input_ids": [4],
                    "target_input_ids": [3, 2, 1],
                },
            ]
        )
        self.assertEqual(batch["input_ids"].tolist()[0], [2, 3, 4, 2, 3, 1])
        self.assertEqual(batch["labels"].tolist()[0], [-100, -100, -100, -100, 3, 1])
        self.assertEqual(batch["input_ids"].tolist()[1], [2, 4, 3, 2, 1, 0])
        self.assertEqual(batch["attention_mask"].tolist()[1], [1, 1, 1, 1, 1, 0])
        self.assertEqual(batch["labels"].tolist()[1], [-100, -100, 3, 2, 1, -100])

    def test_source_tokens_are_forbidden_in_targets(self) -> None:
        collator = DirectCompactBatchCollator(
            pad_token_id=0, source_token_ids=(4,)
        )
        with self.assertRaisesRegex(ValueError, "source token IDs occur in target"):
            collator(
                [{
                    "decoder_prompt_input_ids": [2],
                    "compact_input_ids": [4],
                    "target_input_ids": [4, 1],
                }]
            )

    def test_overflow_never_truncates(self) -> None:
        collator = DirectCompactBatchCollator(
            pad_token_id=0,
            max_source_tokens=2,
            max_target_tokens=2,
            max_total_tokens=5,
        )
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            collator(
                [{
                    "decoder_prompt_input_ids": [2],
                    "compact_input_ids": [2, 3, 2],
                    "target_input_ids": [1],
                }]
            )


class SourceEmbeddingOverlayTests(unittest.TestCase):
    def test_overlay_uses_codebook_mean_and_keeps_lm_head_small(self) -> None:
        torch.manual_seed(3)
        model = FakeCausalLM()
        expected = model.embed.weight[[2, 3]].detach().mean(dim=0)
        output_rows = model.lm_head.weight.size(0)
        overlay = install_source_embedding_overlay(
            model, {4: [2, 3]}, base_vocab_size=4
        )
        self.assertIsInstance(model.get_input_embeddings(), SourceTokenEmbeddingOverlay)
        self.assertTrue(torch.allclose(overlay.source_embeddings.weight[0], expected))
        self.assertEqual(model.lm_head.weight.size(0), output_rows)
        self.assertFalse(overlay.base_embedding.weight.requires_grad)
        self.assertTrue(overlay.source_embeddings.weight.requires_grad)

    def test_tiny_decoder_forward_backpropagates_only_to_overlay_and_lm(self) -> None:
        torch.manual_seed(5)
        model = FakeCausalLM()
        overlay = install_source_embedding_overlay(
            model, {4: [2, 3]}, base_vocab_size=4
        )
        wrapper = DirectCompactCausalLM(model)
        collator = DirectCompactBatchCollator(
            pad_token_id=0,
            source_token_ids=(4,),
            max_source_tokens=4,
            max_target_tokens=4,
            max_total_tokens=10,
        )
        batch = collator(
            [{
                "decoder_prompt_input_ids": [2],
                "compact_input_ids": [4],
                "target_input_ids": [3, 1],
            }]
        )
        outputs = wrapper(**batch)
        outputs["loss"].backward()
        self.assertIsNotNone(overlay.source_embeddings.weight.grad)
        self.assertGreater(float(overlay.source_embeddings.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(overlay.base_embedding.weight.grad)
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_unknown_nonbase_token_fails(self) -> None:
        model = FakeCausalLM()
        install_source_embedding_overlay(model, {4: [2]}, base_vocab_size=4)
        with self.assertRaisesRegex(ValueError, "outside base and source vocabularies"):
            model.get_input_embeddings()(torch.tensor([[5]]))

    def test_overlay_checkpoint_restores_without_lm_head_growth(self) -> None:
        model = FakeCausalLM()
        overlay = install_source_embedding_overlay(
            model, {4: [2]}, base_vocab_size=4
        )
        with torch.no_grad():
            overlay.source_embeddings.weight.fill_(7.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.pt"
            torch.save(overlay.overlay_state(), path)
            restored_model = FakeCausalLM()
            restored = restore_source_embedding_overlay(
                restored_model, {4: [2]}, path, base_vocab_size=4
            )
        self.assertTrue(
            torch.equal(
                restored.source_embeddings.weight,
                torch.full_like(restored.source_embeddings.weight, 7.0),
            )
        )
        self.assertEqual(restored_model.lm_head.weight.size(0), 4)

    def test_overlay_migration_reuses_only_identical_expansions(self) -> None:
        old_model = FakeCausalLM()
        with torch.no_grad():
            old_model.embed.weight.copy_(
                torch.arange(24, dtype=torch.float32).reshape(4, 6)
            )
        frozen_base = old_model.embed.weight.detach().clone()
        old_overlay = install_source_embedding_overlay(
            old_model,
            {4: [2], 5: [3]},
            base_vocab_size=4,
        )
        with torch.no_grad():
            old_overlay.source_embeddings.weight[0].fill_(7.0)
            old_overlay.source_embeddings.weight[1].fill_(9.0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "overlay.pt"
            torch.save(old_overlay.overlay_state(), checkpoint)
            new_model = FakeCausalLM()
            with torch.no_grad():
                new_model.embed.weight.copy_(frozen_base)
            migrated, report = migrate_source_embedding_overlay(
                new_model,
                old_source_token_expansions={4: [2], 5: [3]},
                new_source_token_expansions={4: [2], 5: [2, 3]},
                checkpoint=checkpoint,
                base_vocab_size=4,
            )

        self.assertTrue(
            torch.equal(
                migrated.source_embeddings.weight[0],
                torch.full_like(
                    migrated.source_embeddings.weight[0], 7.0
                ),
            )
        )
        expected_changed = frozen_base[[2, 3]].float().mean(dim=0)
        self.assertTrue(
            torch.equal(
                migrated.source_embeddings.weight[1],
                expected_changed,
            )
        )
        self.assertFalse(
            torch.equal(
                migrated.source_embeddings.weight[1],
                torch.full_like(
                    migrated.source_embeddings.weight[1], 9.0
                ),
            )
        )
        self.assertEqual(report["reused_source_token_ids"], [4])
        self.assertEqual(report["reinitialized_source_token_ids"], [5])
        self.assertEqual(new_model.lm_head.weight.size(0), 4)

    def test_overlay_migration_rejects_source_id_abi_change(self) -> None:
        model = FakeCausalLM()
        overlay = install_source_embedding_overlay(
            model, {4: [2]}, base_vocab_size=4
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "overlay.pt"
            torch.save(overlay.overlay_state(), checkpoint)
            with self.assertRaisesRegex(
                ValueError, "stable source-token ID set"
            ):
                migrate_source_embedding_overlay(
                    FakeCausalLM(),
                    old_source_token_expansions={4: [2]},
                    new_source_token_expansions={5: [2]},
                    checkpoint=checkpoint,
                    base_vocab_size=4,
                )

    def test_overlay_lookup_follows_preplaced_base_embedding_device(self) -> None:
        # Reload inference moves the PEFT-wrapped decoder before restoring the
        # input overlay.  A meta device is sufficient to exercise placement
        # without making this regression test depend on CUDA availability.
        model = FakeCausalLM().to("meta")
        overlay = install_source_embedding_overlay(
            model,
            {4: [2]},
            base_vocab_size=4,
            initialize_from_expansions=False,
        )
        self.assertEqual(overlay.source_id_to_row.device.type, "meta")
        self.assertEqual(overlay.source_embeddings.weight.device.type, "meta")


class CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_contract_copy_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sealed-contract.json"
            destination = root / "checkpoint" / "compact_contract.json"
            payload = b'{"consumed":true,"release_only":{"seal":"abc"}}\r\n'
            source.write_bytes(payload)
            copy_exact_contract(source, destination)
            self.assertEqual(destination.read_bytes(), payload)

    def test_warmstart_checkpoint_is_bound_to_exact_saved_artifacts(self) -> None:
        tokenizer = FakeTokenizer()
        contract = make_contract(tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            adapter = checkpoint / "decoder_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(
                '{"peft_type":"LORA"}\n', encoding="utf-8"
            )
            overlay = checkpoint / "source_embedding_overlay.pt"
            overlay.write_bytes(b"sealed-overlay")
            contract_path = root / "contract.json"
            contract_path.write_text(
                json.dumps(contract.as_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            copy_exact_contract(
                contract_path, checkpoint / "compact_contract.json"
            )
            provenance = {
                "schema": "direct-compact-run-provenance-v1",
                "architecture": "qwen-causal-compact-tokens-no-encoder",
                "decoder_model": contract.decoder_model,
                "decoder_revision": contract.decoder_revision,
                "contract_sha256": sha256_file(
                    checkpoint / "compact_contract.json"
                ),
                "source_overlay_sha256": sha256_file(overlay),
                "decoder_adapter_sha256": sha256_artifact(adapter),
            }
            (checkpoint / "run_provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )

            paths = validate_warmstart_checkpoint(
                checkpoint, contract_path=contract_path
            )
            self.assertEqual(paths["root"], checkpoint.resolve())

            overlay.write_bytes(b"tampered-overlay")
            with self.assertRaisesRegex(ValueError, "sealed provenance"):
                validate_warmstart_checkpoint(
                    checkpoint, contract_path=contract_path
                )

    def test_direct_trainer_resume_continues_step_one_to_two_exactly(self) -> None:
        class ToyAdapter:
            def __init__(self, value: float) -> None:
                self.value = torch.tensor(value)

            def save_pretrained(self, destination: Path) -> None:
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "adapter_config.json").write_text(
                    '{"peft_type":"LORA"}\n', encoding="utf-8"
                )
                torch.save(
                    {"value": self.value.clone()},
                    destination / "adapter_model.bin",
                )

            @classmethod
            def restore(cls, checkpoint: Path) -> "ToyAdapter":
                state = torch.load(
                    checkpoint / "decoder_adapter" / "adapter_model.bin",
                    map_location="cpu",
                    weights_only=True,
                )
                return cls(float(state["value"]))

        class ToyOverlay:
            def __init__(self, value: float) -> None:
                self.value = torch.tensor(value)

            def overlay_state(self) -> dict[str, torch.Tensor]:
                return {"value": self.value.clone()}

            @classmethod
            def restore(cls, checkpoint: Path) -> "ToyOverlay":
                state = torch.load(
                    checkpoint / "source_embedding_overlay.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                return cls(float(state["value"]))

        class ToyTokenizer:
            def save_pretrained(self, destination: Path) -> None:
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "tokenizer.json").write_text(
                    '{"toy":true}\n', encoding="utf-8"
                )

        class ToyTrainerBase:
            def __init__(
                self,
                *,
                output_dir: Path,
                global_step: int,
                optimizer_momentum: float,
                scheduler_step: int,
                rng_marker: int,
            ) -> None:
                self.args = types.SimpleNamespace(
                    output_dir=str(output_dir),
                    should_save=True,
                )
                self.state = types.SimpleNamespace(global_step=global_step)
                self.optimizer_momentum = optimizer_momentum
                self.scheduler_step = scheduler_step
                self.rng_marker = rng_marker

            def _get_output_dir(self, trial=None) -> str:
                del trial
                return self.args.output_dir

            def save_model(
                self, output_dir: str, _internal_call: bool = False
            ) -> None:
                del _internal_call
                self._save(output_dir)

            def _save_checkpoint(self, model, trial) -> None:
                del model, trial
                destination = (
                    Path(self.args.output_dir)
                    / f"checkpoint-{self.state.global_step}"
                )
                self.save_model(str(destination), _internal_call=True)
                torch.save(
                    {"momentum": self.optimizer_momentum},
                    destination / "optimizer.pt",
                )
                torch.save(
                    {"step": self.scheduler_step},
                    destination / "scheduler.pt",
                )
                torch.save(
                    {"marker": self.rng_marker},
                    destination / "rng_state.pth",
                )
                (destination / "trainer_state.json").write_text(
                    json.dumps({"global_step": self.state.global_step}) + "\n",
                    encoding="utf-8",
                )

            def _load_from_checkpoint(self, checkpoint, model=None) -> None:
                del checkpoint, model
                raise AssertionError(
                    "root-model Trainer loading must never be delegated"
                )

            def restore_non_model_state(self, checkpoint: Path) -> None:
                optimizer = torch.load(
                    checkpoint / "optimizer.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                scheduler = torch.load(
                    checkpoint / "scheduler.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                rng = torch.load(
                    checkpoint / "rng_state.pth",
                    map_location="cpu",
                    weights_only=True,
                )
                state = json.loads(
                    (checkpoint / "trainer_state.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.optimizer_momentum = float(optimizer["momentum"])
                self.scheduler_step = int(scheduler["step"])
                self.rng_marker = int(rng["marker"])
                self.state.global_step = int(state["global_step"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            output.mkdir()
            contract = root / "compact_contract.json"
            contract.write_bytes(b'{"sealed":"contract"}\n')
            compatibility = {
                "schema": "direct-compact-trainer-launch-compatibility-v1",
                "data": {"sha256": "a" * 64},
                "loss": {"sequence_distribution_nll": True},
                "optimization": {
                    "learning_rate": 5e-6,
                    "max_steps": 2,
                },
            }

            first_adapter = ToyAdapter(1.0)
            first_overlay = ToyOverlay(10.0)
            FirstTrainer = make_direct_trainer_class(
                ToyTrainerBase,
                adapter_model=first_adapter,
                overlay=first_overlay,
                tokenizer=ToyTokenizer(),
                contract_path=contract,
                authorized_resume_checkpoint=None,
                resume_compatibility=compatibility,
            )
            first = FirstTrainer(
                output_dir=output,
                global_step=1,
                optimizer_momentum=0.75,
                scheduler_step=1,
                rng_marker=101,
            )
            first._save_checkpoint(None, None)
            checkpoint_one = output / "checkpoint-1"
            self.assertTrue(
                (checkpoint_one / DIRECT_TRAINER_RESUME_FILENAME).is_file()
            )
            validate_direct_trainer_resume_checkpoint(
                checkpoint_one,
                expected_compatibility=compatibility,
            )

            resumed_adapter = ToyAdapter.restore(checkpoint_one)
            resumed_overlay = ToyOverlay.restore(checkpoint_one)
            ResumedTrainer = make_direct_trainer_class(
                ToyTrainerBase,
                adapter_model=resumed_adapter,
                overlay=resumed_overlay,
                tokenizer=ToyTokenizer(),
                contract_path=contract,
                authorized_resume_checkpoint=checkpoint_one,
                resume_compatibility=compatibility,
            )
            resumed = ResumedTrainer(
                output_dir=output,
                global_step=0,
                optimizer_momentum=-1.0,
                scheduler_step=-1,
                rng_marker=-1,
            )
            resumed._load_from_checkpoint(str(checkpoint_one))
            resumed.restore_non_model_state(checkpoint_one)
            self.assertTrue(resumed._direct_resume_model_state_verified)
            self.assertEqual(resumed.state.global_step, 1)
            self.assertEqual(float(resumed_adapter.value), 1.0)
            self.assertEqual(float(resumed_overlay.value), 10.0)
            self.assertEqual(resumed.optimizer_momentum, 0.75)
            self.assertEqual(resumed.scheduler_step, 1)
            self.assertEqual(resumed.rng_marker, 101)

            # The second update must build on checkpoint-1 state, not the
            # original construction values.
            resumed_adapter.value.add_(1.0)
            resumed_overlay.value.add_(2.0)
            resumed.optimizer_momentum += 0.125
            resumed.scheduler_step += 1
            resumed.rng_marker += 1
            resumed.state.global_step += 1
            resumed._save_checkpoint(None, None)
            checkpoint_two = output / "checkpoint-2"
            validate_direct_trainer_resume_checkpoint(
                checkpoint_two,
                expected_compatibility=compatibility,
            )
            self.assertEqual(
                float(ToyAdapter.restore(checkpoint_two).value), 2.0
            )
            self.assertEqual(
                float(ToyOverlay.restore(checkpoint_two).value), 12.0
            )
            optimizer_two = torch.load(
                checkpoint_two / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(float(optimizer_two["momentum"]), 0.875)
            self.assertEqual(
                json.loads(
                    (checkpoint_two / "trainer_state.json").read_text(
                        encoding="utf-8"
                    )
                )["global_step"],
                2,
            )

            changed = json.loads(json.dumps(compatibility))
            changed["optimization"]["learning_rate"] = 1e-5
            with self.assertRaisesRegex(
                ValueError, "immutable launch inputs"
            ):
                validate_direct_trainer_resume_checkpoint(
                    checkpoint_two,
                    expected_compatibility=changed,
                )
            (checkpoint_two / "source_embedding_overlay.pt").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(
                ValueError, "artifact bindings"
            ):
                validate_direct_trainer_resume_checkpoint(
                    checkpoint_two,
                    expected_compatibility=compatibility,
                )

    def test_contract_overlay_migration_is_sealed_and_adapter_exact(self) -> None:
        tokenizer = FakeTokenizer()
        seed_contract = make_contract(tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codebook = root / "new_codebook.json"
            codec = root / "new_codec.py"
            codebook.write_text('{"codebook":"new"}\n', encoding="utf-8")
            codec.write_text("# new inline-CFG codec\n", encoding="utf-8")

            old_raw = seed_contract.as_dict()
            old_raw["source_token_ids"] = [4, 5]
            old_raw["source_token_expansions"] = {
                "4": [2],
                "5": [3],
            }
            old_contract = DirectCompactContract.from_mapping(old_raw)
            new_raw = json.loads(json.dumps(old_raw))
            new_raw["codec_sha256"] = sha256_file(codec)
            new_raw["codebook_sha256"] = sha256_file(codebook)
            new_raw["source_token_expansions"]["5"] = [2, 3]
            new_contract = DirectCompactContract.from_mapping(new_raw)
            old_contract_path = root / "old_contract.json"
            new_contract_path = root / "new_contract.json"
            old_contract_path.write_text(
                json.dumps(old_contract.as_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            new_contract_path.write_text(
                json.dumps(new_contract.as_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )

            source = root / "source"
            adapter = source / "decoder_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(
                '{"peft_type":"LORA"}\n', encoding="utf-8"
            )
            (adapter / "adapter_model.safetensors").write_bytes(
                b"adapter-exact"
            )
            copy_exact_contract(
                old_contract_path, source / "compact_contract.json"
            )
            old_model = FakeCausalLM()
            old_overlay = install_source_embedding_overlay(
                old_model,
                dict(old_contract.source_token_expansions),
                base_vocab_size=4,
            )
            with torch.no_grad():
                old_overlay.source_embeddings.weight[0].fill_(7.0)
                old_overlay.source_embeddings.weight[1].fill_(9.0)
            torch.save(
                old_overlay.overlay_state(),
                source / "source_embedding_overlay.pt",
            )
            source_provenance = {
                "schema": "direct-compact-run-provenance-v1",
                "architecture": "qwen-causal-compact-tokens-no-encoder",
                "decoder_model": old_contract.decoder_model,
                "decoder_revision": old_contract.decoder_revision,
                "contract_sha256": sha256_file(
                    source / "compact_contract.json"
                ),
                "source_overlay_sha256": sha256_file(
                    source / "source_embedding_overlay.pt"
                ),
                "decoder_adapter_sha256": sha256_artifact(adapter),
            }
            (source / "run_provenance.json").write_text(
                json.dumps(source_provenance), encoding="utf-8"
            )
            validate_self_sealed_checkpoint(source)

            compatibility = validate_overlay_migration_contracts(
                old_contract_path, new_contract_path
            )
            self.assertEqual(
                compatibility["identical_expansion_source_token_ids"], [4]
            )
            self.assertEqual(
                compatibility["changed_expansion_source_token_ids"], [5]
            )

            migration_model = FakeCausalLM()
            with torch.no_grad():
                migration_model.embed.weight.copy_(
                    torch.arange(24, dtype=torch.float32).reshape(4, 6)
                )
            output = root / "migrated"
            result = materialize_overlay_migrated_checkpoint(
                model=migration_model,
                source_checkpoint=source,
                new_contract_path=new_contract_path,
                codebook_path=codebook,
                codec_path=codec,
                output_dir=output,
            )
            validated_migration = validate_overlay_migrated_checkpoint(
                checkpoint=output,
                source_checkpoint=source,
                new_contract_path=new_contract_path,
                codebook_path=codebook,
                codec_path=codec,
            )
            self.assertEqual(
                validated_migration["receipt_sha256"],
                result["receipt_sha256"],
            )
            validate_warmstart_checkpoint(
                output, contract_path=new_contract_path
            )
            self.assertEqual(
                sha256_artifact(output / "decoder_adapter"),
                sha256_artifact(adapter),
            )
            receipt = json.loads(
                (output / "overlay_migration_receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                receipt["overlay_migration"]["reused_source_token_ids"],
                [4],
            )
            self.assertEqual(
                receipt["overlay_migration"][
                    "reinitialized_source_token_ids"
                ],
                [5],
            )
            self.assertTrue(
                receipt["invariants"]["heldout_data_opened"] is False
            )
            provenance = json.loads(
                (output / "run_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                provenance["overlay_migration_receipt_sha256"],
                result["receipt_sha256"],
            )

            capacity_raw = json.loads(json.dumps(new_raw))
            capacity_raw["max_target_tokens"] = 32
            capacity_raw["max_total_tokens"] = (
                int(capacity_raw["max_source_tokens"]) + 512
            )
            capacity_contract = root / "capacity_contract.json"
            capacity_contract.write_text(
                json.dumps(capacity_raw), encoding="utf-8"
            )
            capacity_compatibility = validate_overlay_migration_contracts(
                new_contract_path, capacity_contract
            )
            self.assertEqual(
                capacity_compatibility["observed_changed_fields"],
                ["max_target_tokens", "max_total_tokens"],
            )
            self.assertEqual(
                capacity_compatibility["changed_expansion_rows"], 0
            )

            incompatible_raw = json.loads(json.dumps(new_raw))
            incompatible_raw["decoder_revision"] = "changed-revision"
            incompatible = root / "incompatible_contract.json"
            incompatible.write_text(
                json.dumps(incompatible_raw), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "non-migratable fields changed"
            ):
                validate_overlay_migration_contracts(
                    old_contract_path, incompatible
                )

    def test_training_stage_gate_binds_gold_data_base_and_loss(self) -> None:
        tokenizer = FakeTokenizer()
        contract = make_contract(tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract_path.write_text(
                json.dumps(contract.as_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )

            def make_checkpoint(
                path: Path, extra: dict | None = None
            ) -> Path:
                adapter = path / "decoder_adapter"
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text(
                    '{"peft_type":"LORA"}\n', encoding="utf-8"
                )
                overlay = path / "source_embedding_overlay.pt"
                overlay.write_bytes(b"sealed-overlay-" + path.name.encode())
                copy_exact_contract(
                    contract_path, path / "compact_contract.json"
                )
                provenance = {
                    "schema": "direct-compact-run-provenance-v1",
                    "architecture": "qwen-causal-compact-tokens-no-encoder",
                    "decoder_model": contract.decoder_model,
                    "decoder_revision": contract.decoder_revision,
                    "contract_sha256": sha256_file(
                        path / "compact_contract.json"
                    ),
                    "source_overlay_sha256": sha256_file(overlay),
                    "decoder_adapter_sha256": sha256_artifact(adapter),
                }
                provenance.update(extra or {})
                (path / "run_provenance.json").write_text(
                    json.dumps(provenance), encoding="utf-8"
                )
                return path

            base = make_checkpoint(root / "base")
            train = root / "train.jsonl"
            eval_file = root / "dev.jsonl"
            train.write_text('{"task_id":"train"}\n', encoding="utf-8")
            eval_file.write_text('{"task_id":"dev"}\n', encoding="utf-8")

            def write_seal(path: Path, output: Path, role: str) -> None:
                path.write_text(
                    json.dumps(
                        {
                            "schema": "compact-public-private-join-seal-v1",
                            "selected_role": role,
                            "output_sha256": sha256_file(output),
                            "contract_sha256": sha256_file(contract_path),
                            "rows": 1,
                        }
                    ),
                    encoding="utf-8",
                )

            train_seal = root / "train.seal.json"
            eval_seal = root / "dev.seal.json"
            write_seal(train_seal, train, "fit")
            write_seal(eval_seal, eval_file, "measure")
            base_paths = validate_warmstart_checkpoint(
                base, contract_path=contract_path
            )
            gold = make_checkpoint(
                root / "gold",
                {
                    "loss_contract": {
                        "sequence_distribution_nll": False,
                        "primary_reduction": "base_causal_lm_token_mean",
                    },
                    "train_file_sha256": sha256_file(train),
                    "eval_file_sha256": sha256_file(eval_file),
                    "train_seal_sha256": sha256_file(train_seal),
                    "eval_seal_sha256": sha256_file(eval_seal),
                    "train_sealed_rows": 1,
                    "eval_sealed_rows": 1,
                    "heldout_loaded_during_training": True,
                    "eval_strategy": "epoch",
                    "warmstart_checkpoint": {
                        "decoder_adapter_sha256": sha256_artifact(
                            base_paths["adapter"]
                        ),
                        "source_overlay_sha256": sha256_file(
                            base_paths["overlay"]
                        ),
                        "contract_sha256": sha256_file(
                            base_paths["contract"]
                        ),
                        "provenance_sha256": sha256_file(
                            base_paths["provenance"]
                        ),
                    },
                    "sparse_topk_tail_auxiliary": None,
                },
            )
            args = types.SimpleNamespace(
                checkpoint=gold,
                contract=contract_path,
                train_file=train,
                train_seal=train_seal,
                eval_file=eval_file,
                eval_seal=eval_seal,
                expected_train_rows=1,
                expected_eval_rows=1,
                loss_mode="token_mean",
                base_warmstart=base,
            )
            result = validate_stage(args)
            self.assertTrue(result["valid"])
            args.loss_mode = "sequence_sum"
            with self.assertRaisesRegex(ValueError, "loss contract"):
                validate_stage(args)


class ConditioningProbePrimitiveTests(unittest.TestCase):
    def test_matched_permutation_is_deterministic_and_has_no_fixed_points(self) -> None:
        lengths = [10, 11, 100, 101]
        first = matched_permutation_indices(lengths, seed=9)
        second = matched_permutation_indices(lengths, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(len(lengths))))
        self.assertTrue(all(index != donor for index, donor in enumerate(first)))

    def test_matched_permutation_minimizes_length_mismatch(self) -> None:
        lengths = [198, 31, 140, 644, 280, 822, 99, 264]
        mapping = matched_permutation_indices(lengths, seed=42)
        cost = sum(abs(lengths[index] - lengths[donor]) for index, donor in enumerate(mapping))
        self.assertEqual(cost, 640)

    def test_causal_nll_uses_only_shifted_unmasked_targets(self) -> None:
        logits = torch.zeros((1, 4, 3))
        labels = torch.tensor([[-100, -100, 1, 2]])
        logits[0, 1, 1] = 10.0
        logits[0, 2, 2] = 10.0
        nll = per_sequence_causal_nll(logits, labels)
        self.assertLess(float(nll[0]), 0.001)

    def test_sequence_distribution_nll_sums_tokens_then_averages_draws(self) -> None:
        logits = torch.zeros((2, 3, 3))
        labels = torch.tensor(
            [
                [-100, 1, 2],
                [-100, -100, 2],
            ]
        )
        normalized = per_sequence_causal_nll(logits, labels)
        summed = per_sequence_causal_nll_sum(logits, labels)
        self.assertTrue(
            torch.allclose(normalized, torch.full((2,), math.log(3.0)))
        )
        self.assertTrue(
            torch.allclose(
                summed,
                torch.tensor([2.0 * math.log(3.0), math.log(3.0)]),
            )
        )

        model = FakeCausalLM(vocab_size=3)
        wrapper = DirectCompactCausalLM(model, sequence_sum_nll=True)
        outputs = wrapper(
            input_ids=torch.tensor([[0, 1, 2], [0, 1, 0]]),
            attention_mask=torch.ones((2, 3), dtype=torch.long),
            labels=labels,
        )
        expected = per_sequence_causal_nll_sum(
            outputs["logits"], labels
        ).mean()
        self.assertTrue(torch.allclose(outputs["loss"], expected))

    def test_null_arm_may_remove_source_explicitly(self) -> None:
        collator = DirectCompactBatchCollator(
            pad_token_id=0, allow_empty_source=True
        )
        batch = collator([{
            "decoder_prompt_input_ids": [2],
            "compact_input_ids": [],
            "target_input_ids": [3, 1],
        }])
        self.assertEqual(batch["source_lengths"].tolist(), [0])

    def test_zero_source_ablation_preserves_positions_and_masks(self) -> None:
        model = FakeCausalLM()
        install_source_embedding_overlay(model, {4: [2]}, base_vocab_size=4)
        wrapper = DirectCompactCausalLM(model)
        collator = DirectCompactBatchCollator(
            pad_token_id=0, source_token_ids=(4,)
        )
        batch = collator([{
            "decoder_prompt_input_ids": [2],
            "compact_input_ids": [4, 4],
            "target_input_ids": [3, 1],
        }])
        correct = wrapper(**batch)["logits"]
        ablated = wrapper(**batch, zero_source_embeddings=True)["logits"]
        self.assertEqual(correct.shape, ablated.shape)
        self.assertEqual(batch["source_lengths"].tolist(), [2])
        self.assertFalse(torch.equal(correct, ablated))


class ArtifactBindingTests(unittest.TestCase):
    def test_directory_hash_covers_paths_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_text("one", encoding="utf-8")
            first = sha256_artifact(root)
            (root / "a").write_text("two", encoding="utf-8")
            self.assertNotEqual(first, sha256_artifact(root))
            (root / "nested").mkdir()
            (root / "nested" / "b").write_text("three", encoding="utf-8")
            self.assertNotEqual(first, sha256_artifact(root))

    def test_join_seal_binds_dataset_contract_role_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "train.jsonl"
            contract = root / "contract.json"
            seal = root / "train.seal.json"
            dataset.write_text('{"row":1}\n', encoding="utf-8")
            contract.write_text('{"contract":1}\n', encoding="utf-8")
            seal.write_text(json.dumps({
                "schema": "compact-public-private-join-seal-v1",
                "selected_role": "fit",
                "output_sha256": sha256_file(dataset),
                "contract_sha256": sha256_file(contract),
                "rows": 1,
            }), encoding="utf-8")
            validate_join_seal(
                dataset, seal, contract, expected_role="fit"
            )
            dataset.write_text('{"row":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed dataset SHA-256 mismatch"):
                validate_join_seal(
                    dataset, seal, contract, expected_role="fit"
                )


class NoEncoderDependencyTests(unittest.TestCase):
    def test_direct_files_do_not_import_graph_or_encoder_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "models/direct_compact_causal.py",
            root / "scripts/training/direct_compact_qwen_decompiler.py",
            root / "scripts/training/run_direct_compact_curriculum.py",
            root / "scripts/training/join_compact_public_private.py",
            root / "scripts/evaluation/direct_compact_qwen_inference.py",
            root / "scripts/evaluation/probe_direct_compact_conditioning.py",
        ]
        forbidden_roots = {
            "torch_geometric",
            "models.graphcodebert_tensor_builder",
            "models.hierarchical_graph_encoder_antigravity",
            "scripts.data.cfg_extractor",
            "scripts.data.dfg_extractor",
        }
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            for forbidden in forbidden_roots:
                self.assertNotIn(forbidden, imported, path)
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("AutoModel.from_pretrained", source)


class SealedJoinTests(unittest.TestCase):
    def test_join_uses_task_mapping_but_emits_no_alignment_or_oracle_fields(self) -> None:
        tokenizer = FakeTokenizer()
        contract = make_contract(tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            public = directory / "public.jsonl"
            private = directory / "private.jsonl"
            alignment = directory / "alignment.jsonl"
            output = directory / "training.jsonl"
            seal = directory / "seal.json"
            contract_path = directory / "contract.json"
            contract_path.write_text(
                json.dumps(contract.as_dict()), encoding="utf-8"
            )
            public_rows = [
                {
                    "compact_input_ids": [4, 2],
                    "compact_codec_sha256": contract.codec_sha256,
                    "compact_codebook_sha256": contract.codebook_sha256,
                    "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
                },
                {
                    "compact_input_ids": [4, 3],
                    "compact_codec_sha256": contract.codec_sha256,
                    "compact_codebook_sha256": contract.codebook_sha256,
                    "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
                },
            ]
            private_rows = [
                {
                    "task_id": "b",
                    "function": "fn0",
                    "dart_source": "int fn0()=>2;",
                    "tests": "secret",
                    "family": "topup_s46",
                },
                {
                    "task_id": "a",
                    "function": "fn0",
                    "dart_source": "int fn0()=>1;",
                    "tests": "secret",
                    "family": "master",
                },
            ]
            alignment_rows = [
                {
                    "model_row": 0,
                    "task_id": "a",
                    "role": "fit",
                    "compact_text": "private-audit",
                    "family": "master",
                    "source_pool": None,
                    "dfg_route": "legacy",
                },
                {
                    "model_row": 1,
                    "task_id": "b",
                    "role": "measure",
                    "compact_text": "private-audit",
                    "family": "topup_s46",
                    "source_pool": "topup_s46",
                    "dfg_route": "current",
                },
            ]
            public.write_text(
                "".join(json.dumps(row) + "\n" for row in public_rows),
                encoding="utf-8",
            )
            private.write_text(
                "".join(json.dumps(row) + "\n" for row in private_rows),
                encoding="utf-8",
            )
            alignment.write_text(
                "".join(json.dumps(row) + "\n" for row in alignment_rows),
                encoding="utf-8",
            )
            result = build_join(
                public,
                alignment,
                private,
                output,
                seal,
                contract_path,
                require_bijective_private=True,
            )
            measure_output = directory / "measure.jsonl"
            measure_seal = directory / "measure.seal.json"
            measure_result = build_join(
                public,
                alignment,
                private,
                measure_output,
                measure_seal,
                contract_path,
                role="measure",
            )
            with self.assertRaisesRegex(
                ValueError, "strict private-label bijection failed"
            ):
                build_join(
                    public,
                    alignment,
                    private,
                    directory / "strict-measure.jsonl",
                    directory / "strict-measure.seal.json",
                    contract_path,
                    role="measure",
                    require_bijective_private=True,
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            measure_rows = [
                json.loads(line)
                for line in measure_output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual([row["dart_source"] for row in rows], ["int fn0()=>1;", "int fn0()=>2;"])
        for row in rows:
            self.assertNotIn("task_id", row)
            self.assertNotIn("tests", row)
            self.assertNotIn("assembly", row)
            self.assertNotIn("signature", row)
            self.assertNotIn("public_line", row)
            self.assertNotIn("private_line", row)
            self.assertNotIn("family", row)
            self.assertNotIn("source_pool", row)
            self.assertNotIn("dfg_route", row)
        self.assertEqual(result["mapping"][0]["private_line"], 1)
        self.assertEqual(result["mapping"][1]["private_line"], 0)
        self.assertTrue(result["private_bijection"]["verified"])
        self.assertEqual(
            result["private_metadata_counts"],
            {
                "rows": 2,
                "family": {
                    "counts": {"master": 1, "topup_s46": 1},
                    "missing_rows": 0,
                },
                "source_pool": {
                    "counts": {"topup_s46": 1},
                    "missing_rows": 1,
                },
                "extractor_route": {
                    "counts": {"current": 1, "legacy": 1},
                    "missing_rows": 0,
                },
            },
        )
        self.assertEqual(measure_result["rows"], 1)
        self.assertEqual(measure_result["skipped_rows"], 1)
        self.assertEqual(
            measure_result["private_metadata_counts"]["family"]["counts"],
            {"topup_s46": 1},
        )
        self.assertEqual([row["dart_source"] for row in measure_rows], ["int fn0()=>2;"])

    def test_join_rejects_extra_model_visible_public_fields(self) -> None:
        tokenizer = FakeTokenizer()
        contract = make_contract(tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            public = directory / "public.jsonl"
            alignment = directory / "alignment.jsonl"
            private = directory / "private.jsonl"
            output = directory / "training.jsonl"
            seal = directory / "seal.json"
            contract_path = directory / "contract.json"
            contract_path.write_text(json.dumps(contract.as_dict()), encoding="utf-8")
            public.write_text(
                json.dumps(
                    {
                        "compact_input_ids": [4],
                        "compact_codec_sha256": contract.codec_sha256,
                        "compact_codebook_sha256": contract.codebook_sha256,
                        "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
                        "task_id": "leak",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            alignment.write_text(
                json.dumps({"model_row": 0, "task_id": "a", "role": "fit"}) + "\n",
                encoding="utf-8",
            )
            private.write_text(
                json.dumps(
                    {
                        "task_id": "a",
                        "function": "fn0",
                        "dart_source": "int fn0()=>1;",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                build_join(
                    public, alignment, private, output, seal, contract_path
                )


class V3SealedJoinTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        contract = make_v3_contract(FakeTokenizer())
        contract_path = root / "contract.json"
        public = root / "public.jsonl"
        alignment = root / "alignment.jsonl"
        private = root / "private.jsonl"
        contract_path.write_text(json.dumps(contract.as_dict()), encoding="utf-8")
        public.write_text(
            json.dumps(
                {
                    "compact_input_ids": [4, 5, 6, 2, 7, 8],
                    "compact_codec_sha256": contract.codec_sha256,
                    "compact_codebook_sha256": contract.codebook_sha256,
                    "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        alignment.write_text(
            json.dumps(
                {
                    "model_row": 0,
                    "task_id": "v3-a",
                    "role": "fit",
                    "family": "master",
                    "source_pool": None,
                    "dfg_route": "legacy_release_v1",
                    "pool_metadata": valid_v3_pool_metadata(use_count=2),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        private.write_text(
            json.dumps(
                {
                    "task_id": "v3-a",
                    "function": "candidate",
                    "lang": "Dart",
                    "family": "master",
                    "dart_source": "int candidate()=>1;",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return public, alignment, private, contract_path

    def test_v3_join_requires_pool_sidecar_and_keeps_it_model_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public, alignment, private, contract_path = self._fixture(root)
            output = root / "train.jsonl"
            seal_path = root / "train.seal.json"
            seal = build_join(
                public,
                alignment,
                private,
                output,
                seal_path,
                contract_path,
                role="fit",
            )
            self.assertEqual(seal["schema"], JOIN_SEAL_SCHEMA_V2)
            self.assertEqual(seal["pool_metadata"]["rows"], 1)
            self.assertEqual(seal["pool_metadata"]["total_use_count"], 2)
            validate_join_seal(
                output, seal_path, contract_path, expected_role="fit"
            )
            model_row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                set(model_row),
                {
                    "lang",
                    "function",
                    "dart_source",
                    "compact_input_ids",
                    "compact_codec_sha256",
                    "compact_codebook_sha256",
                    "compact_tokenizer_sha256",
                },
            )
            self.assertNotIn("pool_metadata", model_row)

    def test_v3_join_rejects_missing_or_nonblind_pool_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public, alignment, private, contract_path = self._fixture(root)
            row = json.loads(alignment.read_text(encoding="utf-8"))
            del row["pool_metadata"]
            alignment.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no pool_metadata object"):
                build_join(
                    public,
                    alignment,
                    private,
                    root / "bad.jsonl",
                    root / "bad.seal.json",
                    contract_path,
                    role="fit",
                )

            _, alignment, _, _ = self._fixture(root)
            row = json.loads(alignment.read_text(encoding="utf-8"))
            row["pool_metadata"]["source_blind"] = False
            alignment.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_blind must be true"):
                build_join(
                    public,
                    alignment,
                    private,
                    root / "bad2.jsonl",
                    root / "bad2.seal.json",
                    contract_path,
                    role="fit",
                )

    def test_pool_metadata_validator_rejects_extra_fields(self) -> None:
        metadata = valid_v3_pool_metadata()
        metadata["literal_payload"] = "secret"
        with self.assertRaisesRegex(ValueError, "pool_metadata field mismatch"):
            validate_v3_pool_alignment_metadata(
                {"pool_metadata": metadata}, "row"
            )


class StrictInferenceAlignmentTests(unittest.TestCase):
    def test_alignment_supplies_identity_and_role_but_not_prompt_content(self) -> None:
        contract = make_contract(FakeTokenizer())

        class RecordingTokenizer:
            def __init__(self) -> None:
                self.texts = []

            def __call__(self, text, **kwargs):
                del kwargs
                self.texts.append(text)
                return {"input_ids": [2, 3]}

        tokenizer = RecordingTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            public = directory / "public.jsonl"
            alignment = directory / "alignment.jsonl"
            public_rows = [
                {
                    "compact_input_ids": [4, 2],
                    "compact_codec_sha256": contract.codec_sha256,
                    "compact_codebook_sha256": contract.codebook_sha256,
                    "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
                },
                {
                    "compact_input_ids": [4, 3],
                    "compact_codec_sha256": contract.codec_sha256,
                    "compact_codebook_sha256": contract.codebook_sha256,
                    "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
                },
            ]
            alignment_rows = [
                {"model_row": 0, "task_id": "secret-a", "role": "fit"},
                {"model_row": 1, "task_id": "secret-b", "role": "measure"},
            ]
            public.write_text(
                "".join(json.dumps(row) + "\n" for row in public_rows),
                encoding="utf-8",
            )
            alignment.write_text(
                "".join(json.dumps(row) + "\n" for row in alignment_rows),
                encoding="utf-8",
            )
            rows = load_inference_rows(
                public,
                alignment,
                contract,
                tokenizer,
                target_budget=2,
                role="measure",
            )
            cot_rows = load_inference_rows(
                public,
                alignment,
                contract,
                tokenizer,
                target_budget=2,
                role="measure",
                direct_prompt_mode="qwen_cot_v1",
            )
        self.assertEqual([row["identity"] for row in rows], ["secret-b"])
        self.assertEqual(
            [row["direct_prompt_mode"] for row in cot_rows],
            ["qwen_cot_v1"],
        )
        self.assertTrue(all("secret-" not in text for text in tokenizer.texts))
        self.assertTrue(all("fn0" in text for text in tokenizer.texts))
        self.assertTrue(
            any("First reason about the reconstruction" in text for text in tokenizer.texts)
        )

    def test_v3_inference_requires_valid_pool_alignment_metadata(self) -> None:
        contract = make_v3_contract(FakeTokenizer())

        class TinyTokenizer:
            def __call__(self, text, **kwargs):
                del text, kwargs
                return {"input_ids": [2]}

            def decode(self, token_ids, **kwargs):
                del kwargs
                if token_ids != [2]:
                    raise AssertionError(token_ids)
                return canonical_v3_pool_json(
                    {
                        "schema": POOL_PAYLOAD_SCHEMA_V1,
                        "target_function": "candidate",
                        "uses": [],
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public.jsonl"
            alignment = root / "alignment.jsonl"
            public.write_text(
                json.dumps(
                    {
                        "compact_input_ids": [4, 5, 6, 2, 7, 8],
                        "compact_codec_sha256": contract.codec_sha256,
                        "compact_codebook_sha256": contract.codebook_sha256,
                        "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            alignment_row = {
                "model_row": 0,
                "task_id": "v3-eval",
                "role": "measure",
                "pool_metadata": valid_v3_pool_metadata(use_count=0),
            }
            alignment.write_text(json.dumps(alignment_row) + "\n", encoding="utf-8")
            rows = load_inference_rows(
                public,
                alignment,
                contract,
                TinyTokenizer(),
                target_budget=2,
                role="measure",
            )
            self.assertEqual([row["identity"] for row in rows], ["v3-eval"])
            alignment_row["pool_metadata"]["use_count"] = 1
            alignment.write_text(json.dumps(alignment_row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "use_count does not match"):
                load_inference_rows(
                    public,
                    alignment,
                    contract,
                    TinyTokenizer(),
                    target_budget=2,
                    role="measure",
                )
            alignment_row["pool_metadata"]["use_count"] = 0
            del alignment_row["pool_metadata"]
            alignment.write_text(json.dumps(alignment_row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no pool_metadata object"):
                load_inference_rows(
                    public,
                    alignment,
                    contract,
                    TinyTokenizer(),
                    target_budget=2,
                    role="measure",
                )


if __name__ == "__main__":
    unittest.main()
