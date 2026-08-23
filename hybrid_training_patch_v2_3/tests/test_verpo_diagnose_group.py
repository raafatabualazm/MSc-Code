from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = ROOT / "scripts" / "training" / "verpo_judge_antigravity.py"
GROUNDING_PATH = ROOT / "scripts" / "training" / "verpo_rescue_grounding.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class DiagnoseGroupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(JUDGE_PATH, "verpo_diagnose_group_test")
        cls.grounding = load_module(
            GROUNDING_PATH,
            "verpo_diagnose_grounding_integration_test",
        )

    def catalog(self, candidate_count: int = 2):
        candidates = []
        candidate_refs = []
        diagnostic_refs = []
        for position in range(candidate_count):
            prefix = f"C{position:03d}"
            candidate_refs.extend(
                [f"{prefix}:BOF", f"{prefix}:L0001", f"{prefix}:EOF"]
            )
            diagnostic_refs.append(f"{prefix}:DIAGNOSTIC")
            candidates.append(
                {
                    "candidate_index": position,
                    "lines": [
                        {
                            "ref": f"{prefix}:L0001",
                            "text": f"int fn0() => {position};",
                        }
                    ],
                    "diagnostic": {
                        "ref": f"{prefix}:DIAGNOSTIC",
                        "text": f"got {position}",
                    },
                }
            )
        body = {
            "schema": "test-grounding-catalog-v1",
            "allowed_refs": {
                "f2_block": ["F2B000"],
                "f2_instruction": ["F2B000:I000"],
                "f2_edge": ["F2E000"],
                "candidate_line": candidate_refs,
                "diagnostic": diagnostic_refs,
            },
            "blocks": [{"ref": "F2B000"}],
            "instructions": [{"ref": "F2B000:I000"}],
            "edges": [{"ref": "F2E000", "source": "F2B000"}],
            "candidates": candidates,
        }
        return {**body, "catalog_sha256": canonical_sha256(body)}

    def payload(self, candidate_count: int = 2, *, guidance_mode=None):
        source = "F2\nC0\n\nAx86_64\nEone\nD\nB\noneret|\nX\n"
        catalog = self.catalog(candidate_count)
        payload = {
            "source": source,
            "source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "source_format_guide": "F2 format guide",
            "tests": "void main() { expect(fn0(), 1); }",
            "reference_catalog": catalog,
            "reference_catalog_sha256": catalog["catalog_sha256"],
            "candidates": [
                {
                    "group_index": 10 + position,
                    "candidate": f"int fn0() => {position};",
                    "diagnostic": f"got {position}",
                    "compiled": True,
                    "full_pass": False,
                }
                for position in range(candidate_count)
            ],
        }
        if guidance_mode is not None:
            payload["guidance_mode"] = guidance_mode
        return payload

    def diagnosis(
        self,
        *,
        group_index: int,
        position: int,
        repair_steps=None,
        evidence_ref: str = "F2E000",
    ):
        return {
            "group_index": group_index,
            "fault_class": "wrong_branch",
            "edit_location": {
                "operation": "insert_after",
                "anchor_ref": f"C{position:03d}:L0001",
                "start_ref": None,
                "end_ref": None,
                "anchor_text": f"int fn0() => {position};",
            },
            "evidence": [
                {
                    "kind": "f2_edge",
                    "ref": evidence_ref,
                    "claim": "The conditional successor is not represented.",
                }
            ],
            "explanation": "The candidate omits the observed branch behavior.",
            "repair_steps": [] if repair_steps is None else repair_steps,
        }

    def response(self, *diagnoses):
        return json.dumps(
            {
                "schema": self.module.DIAGNOSE_RESPONSE_SCHEMA,
                "diagnoses": list(diagnoses),
            }
        )

    def judge(self, **kwargs):
        return self.module.VerpoJudge(
            fail_closed=True,
            thinking_mode="provider_default",
            reasoning_effort="high",
            **kwargs,
        )

    def test_valid_group_is_versioned_cached_and_contains_no_raw_reasoning(self):
        judge = self.judge()
        calls = []

        def fake_call(system, user, **kwargs):
            self.assertEqual(
                kwargs["json_schema"],
                self.module._DIAGNOSE_JSON_SCHEMA,
            )
            calls.append((system, user))
            return self.response(
                self.diagnosis(group_index=10, position=0),
                self.diagnosis(group_index=11, position=1),
            )

        judge._call = fake_call
        payload = self.payload()
        first = judge.diagnose_group(payload)
        second = judge.diagnose_group(payload)

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["schema"], self.module.DIAGNOSE_RESULT_SCHEMA)
        self.assertEqual(
            first["response_schema"],
            self.module.DIAGNOSE_RESPONSE_SCHEMA,
        )
        self.assertEqual(
            first["prompt_schema_version"],
            self.module.DIAGNOSE_PROMPT_SCHEMA_VERSION,
        )
        self.assertEqual(first["guidance_mode"], "diagnosis_only")
        self.assertTrue(all(row["accepted"] for row in first["diagnoses"]))
        self.assertNotIn("reasoning", json.dumps(first).lower())
        self.assertEqual(calls[0][1].count("int fn0() => 0;"), 1)
        self.assertEqual(calls[0][1].count("got 0"), 1)
        telemetry = judge.telemetry()
        self.assertEqual(telemetry["diagnose_group_calls_attempted"], 1)
        self.assertEqual(telemetry["diagnose_group_calls_succeeded"], 1)
        self.assertEqual(telemetry["diagnose_results_accepted"], 2)
        self.assertEqual(telemetry["cache_hits"], 1)

    def test_invalid_semantic_item_rejects_only_it_and_erases_feedback(self):
        judge = self.judge()
        poisoned = self.diagnosis(
            group_index=11,
            position=1,
            evidence_ref="F2E999",
            repair_steps=[],
        )
        poisoned["explanation"] = "MALICIOUS UNGROUNDED FEEDBACK"
        judge._call = lambda *_args, **_kwargs: self.response(
            self.diagnosis(group_index=10, position=0),
            poisoned,
        )

        result = judge.diagnose_group(self.payload())
        good, rejected = result["diagnoses"]
        self.assertTrue(good["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertIn("evidence_ref_invalid", rejected["rejection_reasons"])
        self.assertIsNone(rejected["fault_class"])
        self.assertIsNone(rejected["edit_location"])
        self.assertEqual(rejected["evidence"], [])
        self.assertEqual(rejected["explanation"], "")
        self.assertEqual(rejected["repair_steps"], [])
        self.assertNotIn("MALICIOUS", json.dumps(result))

        telemetry = judge.telemetry()
        self.assertEqual(telemetry["diagnose_results_accepted"], 1)
        self.assertEqual(telemetry["diagnose_results_rejected"], 1)
        self.assertEqual(telemetry["diagnose_semantic_rejections"], 1)
        self.assertEqual(
            telemetry["diagnose_rejection_causes"]["evidence_ref_invalid"],
            1,
        )

    def test_rescue_scalar_is_forbidden_by_the_exact_contract(self):
        judge = self.judge()
        with_scalar = self.diagnosis(group_index=10, position=0)
        with_scalar["rescue_likelihood_score"] = 90
        judge._call = lambda *_args, **_kwargs: self.response(
            with_scalar,
            self.diagnosis(group_index=11, position=1),
        )
        result = judge.diagnose_group(self.payload())
        self.assertFalse(result["diagnoses"][0]["accepted"])
        self.assertEqual(
            result["diagnoses"][0]["rejection_reasons"],
            ["diagnosis_keys_invalid"],
        )
        self.assertTrue(result["diagnoses"][1]["accepted"])
        self.assertNotIn("rescue_likelihood", json.dumps(result))

    def test_structured_output_transport_is_explicit_and_cache_bound(self):
        responses = self.module.VerpoJudge(
            api_style="openai_responses",
            reasoning_effort="medium",
            thinking_mode="provider_default",
        )
        response_options = responses._request_options(
            4096,
            json_schema=self.module._DIAGNOSE_JSON_SCHEMA,
            json_schema_name=self.module._DIAGNOSE_JSON_SCHEMA_NAME,
        )
        self.assertEqual(response_options["reasoning"], {"effort": "medium"})
        self.assertEqual(
            response_options["text"]["format"]["type"],
            "json_schema",
        )
        self.assertTrue(response_options["text"]["format"]["strict"])
        self.assertEqual(
            responses._structured_output_mode(
                self.module._DIAGNOSE_JSON_SCHEMA
            ),
            "responses_text_json_schema",
        )

        fallback = self.module.VerpoJudge(
            api_style="openai_compatible_chat",
            reasoning_effort="medium",
            thinking_mode="provider_default",
            chat_json_schema=False,
        )
        fallback_options = fallback._request_options(
            4096,
            json_schema=self.module._DIAGNOSE_JSON_SCHEMA,
            json_schema_name=self.module._DIAGNOSE_JSON_SCHEMA_NAME,
        )
        self.assertNotIn("response_format", fallback_options)
        self.assertEqual(
            fallback._structured_output_mode(
                self.module._DIAGNOSE_JSON_SCHEMA
            ),
            "validated_json_fallback",
        )

        strict_chat = self.module.VerpoJudge(
            api_style="openai_compatible_chat",
            reasoning_effort="xhigh",
            thinking_mode="provider_default",
            chat_json_schema=True,
        )
        chat_options = strict_chat._request_options(
            4096,
            json_schema=self.module._DIAGNOSE_JSON_SCHEMA,
            json_schema_name=self.module._DIAGNOSE_JSON_SCHEMA_NAME,
        )
        self.assertEqual(
            chat_options["response_format"]["type"],
            "json_schema",
        )
        self.assertEqual(
            strict_chat._structured_output_mode(
                self.module._DIAGNOSE_JSON_SCHEMA
            ),
            "compatible_chat_json_schema",
        )

    def test_diagnosis_only_forbids_steps_but_steps_mode_requires_them(self):
        diagnosis = self.diagnosis(
            group_index=10,
            position=0,
            repair_steps=["Add the missing conditional successor."],
        )
        diagnosis_two = self.diagnosis(
            group_index=11,
            position=1,
            repair_steps=["Add the missing conditional successor."],
        )
        judge = self.judge()
        judge._call = lambda *_args, **_kwargs: self.response(
            diagnosis, diagnosis_two
        )
        diagnosis_only = judge.diagnose_group(self.payload())
        self.assertTrue(
            all(not row["accepted"] for row in diagnosis_only["diagnoses"])
        )
        self.assertTrue(
            all(
                "repair_steps_forbidden" in row["rejection_reasons"]
                for row in diagnosis_only["diagnoses"]
            )
        )

        with_steps = judge.diagnose_group(
            self.payload(guidance_mode="diagnosis_and_steps")
        )
        self.assertTrue(
            all(row["accepted"] for row in with_steps["diagnoses"])
        )

    def test_external_grounding_rejection_does_not_abort_valid_sibling(self):
        judge = self.judge()
        judge._call = lambda *_args, **_kwargs: self.response(
            self.diagnosis(group_index=10, position=0),
            self.diagnosis(group_index=11, position=1),
        )

        def validator(item, _catalog, *, expected_candidate_index):
            self.assertEqual(
                item["group_index"],
                10 + expected_candidate_index,
            )
            if expected_candidate_index == 1:
                return {
                    "accepted": False,
                    "rejection_reasons": ["anchor_text_mismatch"],
                }
            return {"accepted": True, "rejection_reasons": []}

        result = judge.diagnose_group(
            self.payload(),
            item_validator=validator,
            validator_schema_version="test-grounding-v1",
        )
        self.assertTrue(result["diagnoses"][0]["accepted"])
        self.assertFalse(result["diagnoses"][1]["accepted"])
        self.assertEqual(
            result["diagnoses"][1]["rejection_reasons"],
            ["anchor_text_mismatch"],
        )

    def test_live_grounding_validator_enforces_exact_anchor_text(self):
        grounding = self.grounding
        candidates = tuple(
            grounding.CandidateReference(
                candidate_index=position,
                source_sha256=hashlib.sha256(
                    f"int fn0() => {position};".encode("utf-8")
                ).hexdigest(),
                bof_ref=f"C{position:03d}:BOF",
                lines=(
                    grounding.CandidateLineReference(
                        ref=f"C{position:03d}:L0001",
                        line_number=1,
                        text=f"int fn0() => {position};",
                    ),
                ),
                eof_ref=f"C{position:03d}:EOF",
                diagnostic_ref=f"C{position:03d}:DIAGNOSTIC",
                diagnostic=f"got {position}",
            )
            for position in range(2)
        )
        catalog = grounding.GroundingCatalog(
            frontier_f2_sha256="1" * 64,
            f2_source_sha256="2" * 64,
            constant_prefix_sha256="3" * 64,
            f2_schema=grounding.EXPECTED_F2_SCHEMA,
            architecture="x86_64",
            entry_block_refs=("F2B000",),
            blocks=(
                grounding.F2BlockReference(
                    ref="F2B000",
                    block_id=0,
                    is_entry=True,
                    instruction_refs=("F2B000:I000",),
                ),
            ),
            instructions=(
                grounding.F2InstructionReference(
                    ref="F2B000:I000",
                    block_ref="F2B000",
                    block_id=0,
                    instruction_index=0,
                    text="ret",
                ),
            ),
            edges=(
                grounding.F2EdgeReference(
                    ref="F2E000",
                    edge_index=0,
                    source_ref="F2B000",
                    target_ref="F2B000",
                    edge_type="return",
                ),
            ),
            candidates=candidates,
        )
        payload = self.payload()
        payload["reference_catalog"] = catalog
        payload["reference_catalog_sha256"] = catalog.catalog_sha256

        wrong_anchor = self.diagnosis(group_index=11, position=1)
        wrong_anchor["edit_location"]["anchor_text"] = "not the cited line"
        judge = self.judge()
        judge._call = lambda *_args, **_kwargs: self.response(
            self.diagnosis(group_index=10, position=0),
            wrong_anchor,
        )
        result = judge.diagnose_group(
            payload,
            item_validator=grounding.validate_diagnosis_item,
            validator_schema_version=grounding.GROUNDING_SCHEMA,
        )
        self.assertTrue(result["diagnoses"][0]["accepted"])
        self.assertFalse(result["diagnoses"][1]["accepted"])
        self.assertEqual(
            result["diagnoses"][1]["rejection_reasons"],
            ["insert_anchor_text_mismatch"],
        )

    def test_top_level_contract_failure_is_not_cached(self):
        judge = self.judge()
        calls = []

        def fake_call(*_args, **_kwargs):
            calls.append(True)
            return json.dumps({"diagnoses": []})

        judge._call = fake_call
        with self.assertRaises(self.module.VerpoJudgeError):
            judge.diagnose_group(self.payload())
        with self.assertRaises(self.module.VerpoJudgeError):
            judge.diagnose_group(self.payload())
        self.assertEqual(len(calls), 2)
        telemetry = judge.telemetry()
        self.assertEqual(telemetry["diagnose_response_schema_failures"], 2)
        self.assertEqual(telemetry["parse_failures"], 2)
        self.assertEqual(telemetry["cache_entries"], 0)

    def test_reference_catalog_hash_and_candidate_set_are_cache_bound(self):
        judge = self.judge()
        calls = []

        def fake_call(_system, user, **_kwargs):
            calls.append(user)
            if '"group_index":12' in user:
                return self.response(
                    self.diagnosis(group_index=10, position=0),
                    self.diagnosis(group_index=11, position=1),
                    self.diagnosis(group_index=12, position=2),
                )
            return self.response(
                self.diagnosis(group_index=10, position=0),
                self.diagnosis(group_index=11, position=1),
            )

        judge._call = fake_call
        judge.diagnose_group(self.payload())
        judge.diagnose_group(self.payload(candidate_count=3))
        self.assertEqual(len(calls), 2)

        tampered = self.payload()
        tampered["reference_catalog"]["edges"][0]["source"] = "F2B999"
        with self.assertRaisesRegex(
            self.module.VerpoJudgeError,
            "catalogue hash mismatch",
        ):
            judge.diagnose_group(tampered)

    def test_receipt_persists_only_reasoning_presence_not_plaintext(self):
        secret = "DO-NOT-PERSIST-PRIVATE-THINKING"
        content = self.response(
            self.diagnosis(group_index=10, position=0),
            self.diagnosis(group_index=11, position=1),
        )

        class Completions:
            def create(self, **_kwargs):
                return SimpleNamespace(
                    id="diagnose-response-1",
                    model="deepseek-chat",
                    system_fingerprint="fp-test",
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content=content,
                                reasoning_content=secret,
                            ),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=100,
                        completion_tokens=50,
                        total_tokens=150,
                    ),
                )

        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "receipts.jsonl"
            judge = self.judge(
                model="deepseek-chat",
                api_style="openai_compatible_chat",
                max_tokens=4096,
                completion_retries=0,
                retry_max_tokens=4096,
                receipt_journal_path=journal,
            )
            judge._client = SimpleNamespace(
                chat=SimpleNamespace(completions=Completions())
            )
            result = judge.diagnose_group(self.payload())
            self.assertTrue(
                all(row["accepted"] for row in result["diagnoses"])
            )
            persisted = journal.read_text(encoding="utf-8")
            self.assertNotIn(secret, persisted)
            receipt = judge.receipt_attestation_since(0)["receipts"][0]
            self.assertTrue(
                receipt["response"]["reasoning_content_present"]
            )
            self.assertFalse(receipt["plaintext_reasoning_persisted"])
            self.assertEqual(
                receipt["request"]["structured_output_mode"],
                "validated_json_fallback",
            )
            self.assertRegex(
                receipt["request"]["structured_output_schema_sha256"],
                r"^[0-9a-f]{64}$",
            )


if __name__ == "__main__":
    unittest.main()
