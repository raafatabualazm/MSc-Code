from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from models.direct_compact_causal import sha256_file
from scripts.training import chatgpt_compact_rs_batch as batch


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class ChatGptCompactBatchTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> argparse.Namespace:
        train = root / "train.jsonl"
        write_jsonl(
            train,
            [
                {
                    "task_id": "fit-1",
                    "acceptance_tests": "assert(fn0(1) == 2);",
                    "dart_source": "int fn0(int x) => x + 1;",
                }
            ],
        )
        predictions = root / "predictions.json"
        predictions.write_text(
            json.dumps([{"id": "fit-1", "predictions": ["bad candidate"]}]),
            encoding="utf-8",
        )
        prediction_provenance = {
            "schema": "direct-compact-inference-v1",
            "output_sha256": sha256_file(predictions),
        }
        provenance_path = Path(str(predictions) + ".provenance.json")
        provenance_path.write_text(
            json.dumps(prediction_provenance), encoding="utf-8"
        )
        score = root / "score.json"
        score.write_text(
            json.dumps(
                {
                    "schema": "direct-compact-attested-passk-v1",
                    "evaluation": {"sha256": sha256_file(train)},
                    "predictions": {
                        "path": str(predictions.resolve()),
                        "sha256": sha256_file(predictions),
                        "provenance_sha256": sha256_file(provenance_path),
                    },
                    "evaluator": {
                        "completion_attestation": (
                            "per-run-256-bit-marker-exactly-once-v1"
                        )
                    },
                    "k": 1,
                    "task_results": [
                        {"task_id": "fit-1", "pass_at_k": False}
                    ],
                }
            ),
            encoding="utf-8",
        )
        tokenizer = Tokenizer(
            models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer_path = root / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        system_prompt = "Sealed lossless F2 grammar; return Dart fn0 only."
        system_prompt_sha256 = batch.hashlib.sha256(
            system_prompt.encode()
        ).hexdigest()
        serialized = root / "serialized.jsonl"
        text = "F2\nC0\n\nAx86_64\nE一\nD\nB\n一ret|\nX\n"
        write_jsonl(
            serialized,
            [
                {
                    "task_id": "fit-1",
                    "text": text,
                    "text_sha256": batch.hashlib.sha256(
                        text.encode()
                    ).hexdigest(),
                    "verified": {
                        "per_task_instruction_dictionary_roundtrip": True,
                        "compact_semantic_f2_roundtrip": True,
                        "branch_targets_reconstructed_from_cfg": True,
                        "opaque_custom_ids_in_text": False,
                    },
                    "representation_schema": "lossless-semantic-f2",
                    "system_prompt_sha256": system_prompt_sha256,
                }
            ],
        )
        serialized_manifest = Path(str(serialized) + ".manifest.json")
        serialized_manifest.write_text(
            json.dumps(
                {
                    "schema": "verified-api-readable-compact-v2",
                    "output": {
                        "sha256": sha256_file(serialized),
                        "size_bytes": serialized.stat().st_size,
                    },
                    "dataset": {"sha256": sha256_file(train)},
                    "rows": 1,
                    "training_objective_scope": "executable_reward_only",
                    "derivation": {
                        "schema": "binary-multifunction-executable-subset-v1",
                        "output_rows": 1,
                    },
                    "f2_prompt_contract": {
                        "representation_schema": "lossless-semantic-f2",
                        "system_prompt": system_prompt,
                        "system_prompt_sha256": system_prompt_sha256,
                        "tokenizer_sha256": sha256_file(tokenizer_path),
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
                        "exact_audited_execution_exclusions_applied": True,
                        "all_remaining_rows_byte_identical_to_parent": True,
                        "heldout_175_disjoint_and_untouched": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        return argparse.Namespace(
            serialized_inputs=str(serialized),
            serialized_manifest=str(serialized_manifest),
            tokenizer_json=str(tokenizer_path),
            train_file=str(train),
            score_report=str(score),
            predictions=str(predictions),
            round_dir=str(root / "round_001"),
            round=1,
            prior_verified=[],
            model="chat-latest",
            samples_per_task=4,
            max_output_tokens=3072,
            max_prompt_tokens=12000,
            chat_overhead_reserve=256,
            temperature=0.8,
            limit=0,
        )

    def test_prepare_builds_four_independent_responses_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            self.assertEqual(batch.prepare_command(args), 0)
            round_dir = Path(args.round_dir)
            requests = [
                json.loads(line)
                for line in (round_dir / "input.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(requests), 4)
            self.assertEqual(len({row["custom_id"] for row in requests}), 4)
            self.assertTrue(
                all(row["url"] == "/v1/responses" for row in requests)
            )
            self.assertTrue(
                all(row["body"]["model"] == "chat-latest" for row in requests)
            )
            encoded = json.dumps(requests)
            self.assertNotIn("assert(fn0", encoded)
            self.assertNotIn("int fn0(int x) => x + 1", encoded)
            manifest = json.loads(
                (round_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["requests"], 4)
            self.assertFalse(manifest["prompt"]["private_tests_exposed_to_api"])
            self.assertEqual(batch.prepare_command(args), 0)

    def test_response_validation_rejects_zero_usage(self) -> None:
        body = {
            "status": "completed",
            "id": "resp-1",
            "model": "chat-latest",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "int fn0(int x) => x;",
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 1,
                "total_tokens": 1,
            },
        }
        with self.assertRaisesRegex(ValueError, "input token usage"):
            batch.validate_batch_response_body(body)


if __name__ == "__main__":
    unittest.main()
