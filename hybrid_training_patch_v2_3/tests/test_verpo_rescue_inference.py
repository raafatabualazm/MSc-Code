from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from models.direct_compact_causal import DirectCompactContract, sha256_file
from scripts.evaluation import direct_compact_qwen_inference as inference
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
)


class StubContract:
    schema = "stub-v1"
    target_function = "fn0"
    target_language = "Dart"
    max_total_tokens = 12

    def validate_row(self, row, _identity):
        return list(row["compact_input_ids"])

    def validate_v3_pool_payload(self, _compact, _tokenizer, _identity):
        return {"uses": []}


class CountingTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        truncation,
        padding,
    ):
        del truncation, padding
        if "Compact binary tokens follow:" in text:
            values = [2, 3]
        else:
            values = [5] * len(text)
        if add_special_tokens:
            values = [1] + values
        return {"input_ids": values}


def public_row() -> dict[str, object]:
    return {
        "compact_input_ids": [7, 8],
        "compact_codec_sha256": "a" * 64,
        "compact_codebook_sha256": "b" * 64,
        "compact_tokenizer_sha256": "c" * 64,
    }


def normalized_plan(
    *,
    arm: str,
    rows: list[dict[str, object]],
    source_plan_sha256: str = "d" * 64,
    base_candidate_rank: int = 2,
    repairs_per_candidate: int = 4,
) -> dict[str, object]:
    return {
        "schema": inference.RESCUE_CONDITIONING_SCHEMA,
        "path": "plan.json",
        "sha256": "e" * 64,
        "arm": arm,
        "base_candidate_rank": base_candidate_rank,
        "repairs_per_candidate": repairs_per_candidate,
        "source_plan_sha256": source_plan_sha256,
        "rows": rows,
    }


def normalized_row(
    task_id: str,
    *,
    generate: bool,
    conditioning: str,
    reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "generate": generate,
        "conditioning": conditioning,
        "conditioning_sha256": hashlib.sha256(
            conditioning.encode("utf-8")
        ).hexdigest(),
        "rejection_reasons": list(reasons or []),
    }


class RescuePlanContractTests(unittest.TestCase):
    def write_plan(self, root: Path, value: dict[str, object]) -> Path:
        path = root / "plan.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def raw_plan(
        self,
        *,
        arm: str,
        row: dict[str, object],
    ) -> dict[str, object]:
        return {
            "schema": inference.RESCUE_CONDITIONING_SCHEMA,
            "arm": arm,
            "base_candidate_rank": 0,
            "repairs_per_candidate": 4,
            "source_plan_sha256": "a" * 64,
            "rows": [row],
        }

    def test_plain_resample_is_strictly_unconditioned(self):
        row = normalized_row(
            "task-a",
            generate=True,
            conditioning="feedback contamination",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_plan(
                Path(temporary),
                self.raw_plan(arm="plain_resample", row=row),
            )
            with self.assertRaisesRegex(ValueError, "must be unconditioned"):
                inference.load_rescue_conditioning_plan(path)

    def test_generatable_row_cannot_carry_rejection_reasons(self):
        row = normalized_row(
            "task-a",
            generate=True,
            conditioning="compiler diagnostic",
            reasons=["judge_invalid"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_plan(
                Path(temporary),
                self.raw_plan(arm="compiler_only", row=row),
            )
            with self.assertRaisesRegex(ValueError, "rejection reasons"):
                inference.load_rescue_conditioning_plan(path)


class RescueLoadingTests(unittest.TestCase):
    def write_views(self, root: Path):
        dataset = root / "public.jsonl"
        alignment = root / "alignment.jsonl"
        dataset.write_text(
            json.dumps(public_row()) + "\n",
            encoding="utf-8",
        )
        alignment.write_text(
            json.dumps(
                {
                    "model_row": 0,
                    "task_id": "task-a",
                    "role": "measure",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return dataset, alignment

    def test_conditioning_tokens_count_against_total_context(self):
        plan = normalized_plan(
            arm="compiler_only",
            rows=[
                normalized_row(
                    "task-a",
                    generate=True,
                    conditioning="abcd",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            dataset, alignment = self.write_views(Path(temporary))
            with self.assertRaisesRegex(
                ValueError,
                "rescue conditioning leaves insufficient target budget",
            ):
                inference.load_rows(
                    dataset,
                    alignment,
                    StubContract(),
                    CountingTokenizer(),
                    target_budget=4,
                    role="measure",
                    rescue_conditioning=plan,
                )

    def test_rejected_plan_task_must_exist_in_selected_alignment_role(self):
        plan = normalized_plan(
            arm="diagnosis_only",
            rows=[
                normalized_row(
                    "task-a",
                    generate=True,
                    conditioning="diagnosis",
                ),
                normalized_row(
                    "task-missing",
                    generate=False,
                    conditioning="",
                    reasons=["schema_rejected"],
                ),
            ],
        )
        contract = StubContract()
        contract.max_total_tokens = 100
        with tempfile.TemporaryDirectory() as temporary:
            dataset, alignment = self.write_views(Path(temporary))
            with self.assertRaisesRegex(
                ValueError,
                "plan tasks do not exactly belong",
            ):
                inference.load_rows(
                    dataset,
                    alignment,
                    contract,
                    CountingTokenizer(),
                    target_budget=4,
                    role="measure",
                    rescue_conditioning=plan,
                )


class PairedSeedAndTokenTests(unittest.TestCase):
    def test_seed_is_task_local_and_arm_independent(self):
        common = {
            "base_seed": 42,
            "source_plan_sha256": "a" * 64,
            "base_candidate_rank": 3,
            "task_id": "task-z",
        }
        seed = inference._rescue_task_seed(**common)
        first = normalized_plan(
            arm="compiler_only",
            rows=[],
            source_plan_sha256="a" * 64,
            base_candidate_rank=3,
        )
        second = normalized_plan(
            arm="diagnosis_and_steps",
            rows=[],
            source_plan_sha256="a" * 64,
            base_candidate_rank=3,
        )
        self.assertEqual(
            inference._scheduled_batch_seed(
                base_seed=42,
                batch_index=0,
                task_ids=["task-z"],
                rescue_conditioning=first,
            ),
            seed,
        )
        self.assertEqual(
            inference._scheduled_batch_seed(
                base_seed=42,
                batch_index=99,
                task_ids=["task-z"],
                rescue_conditioning=second,
            ),
            seed,
        )
        self.assertNotEqual(
            seed,
            inference._rescue_task_seed(
                **{**common, "task_id": "task-other"}
            ),
        )

    def test_rescue_seed_rejects_multi_task_batches(self):
        plan = normalized_plan(arm="plain_resample", rows=[])
        with self.assertRaisesRegex(ValueError, "exactly one task"):
            inference._scheduled_batch_seed(
                base_seed=42,
                batch_index=0,
                task_ids=["task-a", "task-b"],
                rescue_conditioning=plan,
            )

    def test_nonrescue_seed_and_provenance_policy_remain_compatible(self):
        self.assertEqual(
            inference._batch_seed(
                base_seed=42,
                batch_index=0,
                task_ids=["task-a", "task-b"],
            ),
            3099127134,
        )
        self.assertEqual(
            inference._provenance_seed_policy(None),
            "independent_sha256_seed_per_ordered_task_batch",
        )

    def test_token_trimming_supports_multiple_eos_and_rejects_bool(self):
        self.assertEqual(
            inference._trim_generated_token_ids(
                [4, 5, 9, 0, 0],
                eos_token_id=[8, 9],
                pad_token_id=0,
            ),
            [4, 5, 9],
        )
        with self.assertRaisesRegex(ValueError, "invalid token IDs"):
            inference._trim_generated_token_ids(
                [4, True],
                eos_token_id=9,
                pad_token_id=0,
            )

    def test_persisted_tokens_are_bounded_by_generation_vocab(self):
        row = {
            "prediction_token_ids": [[1, 2], [3]],
        }
        self.assertTrue(
            inference._valid_prediction_token_ids(
                row,
                num_samples=2,
                vocab_size=4,
            )
        )
        row["prediction_token_ids"][1] = [4]
        self.assertFalse(
            inference._valid_prediction_token_ids(
                row,
                num_samples=2,
                vocab_size=4,
            )
        )

    def test_rescue_output_binds_conditioning_and_exact_fields(self):
        row = {
            "id": "task-a",
            "predictions": ["code-a", "code-b"],
            "prediction_token_ids": [[1], [2]],
            "conditioning_sha256": "a" * 64,
        }
        self.assertTrue(
            inference._valid_inference_output_row(
                row,
                expected_id="task-a",
                num_samples=2,
                rescue=True,
                vocab_size=4,
                conditioning_sha256="a" * 64,
            )
        )
        row["conditioning_sha256"] = "b" * 64
        self.assertFalse(
            inference._valid_inference_output_row(
                row,
                expected_id="task-a",
                num_samples=2,
                rescue=True,
                vocab_size=4,
                conditioning_sha256="a" * 64,
            )
        )


class ExistingArtifactCompatibilityTests(unittest.TestCase):
    def make_contract(self) -> DirectCompactContract:
        return DirectCompactContract(
            codec_sha256="1" * 64,
            codebook_sha256="2" * 64,
            tokenizer_json_sha256="3" * 64,
            tokenizer_fingerprint_sha256="4" * 64,
            model_config_sha256="5" * 64,
            decoder_model="fake/decoder",
            decoder_revision="revision-1",
            target_function="fn0",
            target_language="Dart",
            dfg_extractor_sha256="6" * 64,
            lossless_domain="scrubbed_canonical_graph",
            max_source_tokens=10,
            max_target_tokens=8,
            max_total_tokens=20,
            base_vocab_size=10,
            source_token_ids=(10,),
            source_token_expansions=((10, (2,)),),
        )

    def touch(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_default_decoder_identity_reuses_exact_legacy_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = self.make_contract()
            contract_path = root / "contract.json"
            contract_path.write_text(
                json.dumps(contract.as_dict()),
                encoding="utf-8",
            )
            dataset = root / "public.jsonl"
            alignment = root / "alignment.jsonl"
            self.touch(dataset, json.dumps(public_row()) + "\n")
            self.touch(
                alignment,
                json.dumps(
                    {
                        "model_row": 0,
                        "task_id": "task-a",
                        "role": "measure",
                    }
                )
                + "\n",
            )
            codebook = root / "codebook.json"
            codec = root / "codec.py"
            tokenizer_json = root / "tokenizer.json"
            overlay = root / "overlay.pt"
            for path, content in (
                (codebook, "{}"),
                (codec, "# codec"),
                (tokenizer_json, "{}"),
                (overlay, "overlay"),
            ):
                self.touch(path, content)

            output = root / "predictions.json"
            outputs = [{"id": "task-a", "predictions": ["int fn0()=>1;"]}]
            output.write_text(json.dumps(outputs), encoding="utf-8")
            journal = root / "generation.jsonl"
            args = argparse.Namespace(
                dataset=str(dataset),
                alignment=str(alignment),
                output=str(output),
                rescue_conditioning_plan="",
                journal=str(journal),
                contract=str(contract_path),
                codebook=str(codebook),
                codec_artifact=str(codec),
                decoder_model="",
                decoder_revision="",
                tokenizer="",
                tokenizer_revision="",
                tokenizer_json=str(tokenizer_json),
                attn_implementation="eager",
                decoder_adapter="",
                source_overlay=str(overlay),
                batch_size=1,
                max_new_tokens=8,
                num_samples=1,
                temperature=0.8,
                top_p=0.95,
                top_k=0,
                seed=42,
                limit=0,
                role="measure",
                direct_prompt_mode="code_only_v1",
                bf16=False,
                fp16=False,
                device="cpu",
            )
            journal_contract = inference._build_inference_journal_contract(
                args=args,
                dataset_path=dataset,
                alignment_path=alignment,
                selected_ids=["task-a"],
                contract=contract,
                decoder_model=contract.decoder_model,
                decoder_revision=contract.decoder_revision,
                model_config_sha256=contract.model_config_sha256,
                rescue_conditioning=None,
            )
            header = append_event(
                journal,
                {
                    "event": "inference_header",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "contract": journal_contract,
                    "contract_sha256": canonical_sha256(journal_contract),
                },
            )
            del header
            seed = inference._batch_seed(
                base_seed=42,
                batch_index=0,
                task_ids=["task-a"],
            )
            started = append_event(
                journal,
                {
                    "event": "inference_batch_started",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "batch_index": 0,
                    "task_ids": ["task-a"],
                    "batch_seed": seed,
                    "slot_ids": ["task-a:0"],
                },
            )
            append_event(
                journal,
                {
                    "event": "inference_batch_terminal",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "batch_index": 0,
                    "started_event_sha256": started[
                        "journal_event_sha256"
                    ],
                    "retry_count": 0,
                    "latest_retry_event_sha256": None,
                    "predictions": outputs,
                    "predictions_canonical_sha256": canonical_sha256(
                        outputs
                    ),
                },
            )
            append_event(
                journal,
                {
                    "event": "inference_complete",
                    "schema": inference.INFERENCE_JOURNAL_SCHEMA,
                    "rows": 1,
                    "slots": 1,
                    "outputs_canonical_sha256": canonical_sha256(outputs),
                    "resampled_slots": 0,
                    "orphan_retry_events": 0,
                    "orphan_recomputed_slots": 0,
                },
            )
            provenance = {
                "schema": "direct-compact-inference-v1",
                "dataset_sha256": sha256_file(dataset),
                "alignment_sha256": sha256_file(alignment),
                "selected_role": "measure",
                "contract_sha256": sha256_file(contract_path),
                "codebook_sha256": sha256_file(codebook),
                "codec_sha256": sha256_file(codec),
                "tokenizer_json_sha256": sha256_file(tokenizer_json),
                "decoder_model": contract.decoder_model,
                "decoder_revision": contract.decoder_revision,
                "model_config_sha256": contract.model_config_sha256,
                "attn_implementation": "eager",
                "decoder_adapter": None,
                "decoder_adapter_sha256": None,
                "source_overlay_sha256": sha256_file(overlay),
                "overlay_rows": 1,
                "lm_head_rows": 10,
                "num_rows": 1,
                "num_samples": 1,
                "max_new_tokens": 8,
                "direct_prompt_mode": "code_only_v1",
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 0,
                "batch_size": 1,
                "limit": 0,
                "seed": 42,
                "bf16": False,
                "fp16": False,
                "precision": "fp32",
                "output_sha256": sha256_file(output),
                "generation_journal": journal_record(journal),
                "sampling_seed_policy": (
                    "independent_sha256_seed_per_ordered_task_batch"
                ),
                "started_without_terminal_policy": (
                    "retry_identical_seeded_batch_with_hash_chained_receipt"
                ),
                "resampled_slots": 0,
                "orphan_retry_events": 0,
                "orphan_recomputed_slots": 0,
                "encoder": None,
                "soft_prefix": None,
            }
            Path(str(output) + ".provenance.json").write_text(
                json.dumps(provenance),
                encoding="utf-8",
            )
            self.assertEqual(
                inference.validate_existing_inference(args),
                provenance,
            )


if __name__ == "__main__":
    unittest.main()
