from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = ROOT / "scripts" / "training" / "verpo_judge_antigravity.py"


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


class NativeAnthropicJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            JUDGE_PATH,
            "verpo_judge_anthropic_native_test",
        )

    def payload(self):
        source = "F2\nC0\n\nAx86_64\nEone\nD\nB\noneret|\nX\n"
        catalog_body = {
            "schema": "test-grounding-catalog-v1",
            "allowed_refs": {
                "f2_block": ["F2B000"],
                "f2_instruction": ["F2B000:I000"],
                "f2_edge": ["F2E000"],
                "candidate_line": [
                    "C000:BOF",
                    "C000:L0001",
                    "C000:EOF",
                ],
                "diagnostic": ["C000:DIAGNOSTIC"],
            },
            "blocks": [{"ref": "F2B000"}],
            "instructions": [{"ref": "F2B000:I000"}],
            "edges": [{"ref": "F2E000", "source": "F2B000"}],
            "candidates": [
                {
                    "candidate_index": 0,
                    "lines": [
                        {
                            "ref": "C000:L0001",
                            "text": "int fn0() => 0;",
                        }
                    ],
                    "diagnostic": {
                        "ref": "C000:DIAGNOSTIC",
                        "text": "got 0",
                    },
                }
            ],
        }
        catalog = {
            **catalog_body,
            "catalog_sha256": canonical_sha256(catalog_body),
        }
        return {
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
                    "group_index": 7,
                    "candidate": "int fn0() => 0;",
                    "diagnostic": "got 0",
                    "compiled": True,
                    "full_pass": False,
                }
            ],
        }

    def response_text(self):
        return json.dumps(
            {
                "schema": self.module.DIAGNOSE_RESPONSE_SCHEMA,
                "diagnoses": [
                    {
                        "group_index": 7,
                        "fault_class": "wrong_constant",
                        "edit_location": {
                            "operation": "replace_range",
                            "anchor_ref": None,
                            "start_ref": "C000:L0001",
                            "end_ref": "C000:L0001",
                            "anchor_text": "int fn0() => 0;",
                        },
                        "evidence": [
                            {
                                "kind": "f2_instruction",
                                "ref": "F2B000:I000",
                                "claim": "The cited instruction disagrees with the literal.",
                            }
                        ],
                        "explanation": "The returned literal is incorrect.",
                        "repair_steps": [],
                    }
                ],
            }
        )

    def judge(self, journal: Path):
        return self.module.VerpoJudge(
            model="claude-sonnet-5",
            base_url="https://api.anthropic.com",
            api_style="anthropic_messages",
            max_tokens=4096,
            timeout_seconds=180,
            max_retries=0,
            completion_retries=0,
            retry_max_tokens=4096,
            thinking_mode="adaptive",
            reasoning_effort="high",
            reasoning_mode="standard",
            chat_json_schema=False,
            max_calls=1,
            fail_closed=True,
            receipt_journal_path=journal,
        )

    def anthropic_response(self, *, stop_reason: str):
        return SimpleNamespace(
            id="msg_native_anthropic_1",
            model="claude-sonnet-5",
            stop_reason=stop_reason,
            content=[
                SimpleNamespace(
                    type="thinking",
                    thinking="DO-NOT-PERSIST-ANTHROPIC-THINKING",
                ),
                SimpleNamespace(type="text", text=self.response_text()),
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                cache_creation_input_tokens=20,
                cache_read_input_tokens=30,
                output_tokens=50,
            ),
        )

    def test_native_messages_call_is_grounded_sealed_and_non_retrying(self):
        calls = []
        owner = self

        class Messages:
            def create(self, **options):
                calls.append(options)
                return owner.anthropic_response(stop_reason="end_turn")

        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "receipts.jsonl"
            judge = self.judge(journal)
            judge._client = SimpleNamespace(messages=Messages())

            result = judge.diagnose_group(self.payload())

            self.assertTrue(result["diagnoses"][0]["accepted"])
            self.assertEqual(len(calls), 1)
            request = calls[0]
            self.assertEqual(request["model"], "claude-sonnet-5")
            self.assertEqual(request["max_tokens"], 4096)
            self.assertEqual(request["thinking"], {"type": "adaptive"})
            self.assertEqual(
                request["output_config"],
                {
                    "effort": "high",
                    "format": {
                        "type": "json_schema",
                        "schema": self.module._DIAGNOSE_JSON_SCHEMA,
                    },
                },
            )
            self.assertNotIn("extra_body", request)
            self.assertEqual(
                request["messages"],
                [{"role": "user", "content": request["messages"][0]["content"]}],
            )
            self.assertNotIn("reasoning_effort", request)
            self.assertNotIn("response_format", request)
            self.assertIn("diagnose failed", request["system"])

            receipt = judge.receipt_attestation_since(0)["receipts"][0]
            self.assertTrue(receipt["validation"]["accepted"])
            self.assertTrue(
                receipt["validation"]["exact_requested_model_required"]
            )
            self.assertFalse(
                receipt["validation"]["system_fingerprint_required"]
            )
            self.assertEqual(receipt["response"]["prompt_tokens"], 150)
            self.assertEqual(receipt["response"]["completion_tokens"], 50)
            self.assertEqual(receipt["response"]["total_tokens"], 200)
            self.assertEqual(
                receipt["response"]["total_tokens_source"],
                "derived_from_input_plus_cache_plus_output",
            )
            self.assertEqual(receipt["response"]["finish_reason"], "stop")
            self.assertTrue(
                receipt["response"]["reasoning_content_present"]
            )
            self.assertEqual(
                receipt["request"]["structured_output_mode"],
                "anthropic_output_config_json_schema",
            )
            self.assertEqual(
                receipt["request"]["reasoning_effort"],
                "high",
            )
            self.assertRegex(
                receipt["request"]["structured_output_schema_sha256"],
                r"^[0-9a-f]{64}$",
            )
            persisted = journal.read_text(encoding="utf-8")
            self.assertNotIn(
                "DO-NOT-PERSIST-ANTHROPIC-THINKING",
                persisted,
            )
            telemetry = judge.telemetry()
            self.assertEqual(telemetry["api_calls"], 1)
            self.assertEqual(telemetry["api_successes"], 1)
            self.assertEqual(telemetry["completion_retries"], 0)

    def test_max_tokens_is_one_terminal_billed_attempt(self):
        calls = []
        owner = self

        class Messages:
            def create(self, **options):
                calls.append(options)
                return owner.anthropic_response(stop_reason="max_tokens")

        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "receipts.jsonl"
            judge = self.judge(journal)
            judge._client = SimpleNamespace(messages=Messages())

            with self.assertRaisesRegex(
                self.module.VerpoJudgeError,
                "no complete final content",
            ):
                judge.diagnose_group(self.payload())

            self.assertEqual(len(calls), 1)
            receipt = judge.receipt_attestation_since(0)["receipts"][0]
            self.assertEqual(receipt["response"]["finish_reason"], "length")
            telemetry = judge.telemetry()
            self.assertEqual(telemetry["api_calls"], 1)
            self.assertEqual(telemetry["length_responses"], 1)
            self.assertEqual(telemetry["completion_retries"], 0)

    def test_anthropic_credential_source_is_provider_specific(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "anthropic-fixture",
                "DEEPSEEK_API_KEY": "deepseek-fixture",
                "OPENAI_API_KEY": "openai-fixture",
            },
            clear=True,
        ):
            judge = self.module.VerpoJudge(
                model="claude-sonnet-5",
                base_url="https://api.anthropic.com",
                api_style="anthropic_messages",
                thinking_mode="adaptive",
                reasoning_effort="high",
                max_retries=0,
                completion_retries=0,
                retry_max_tokens=4096,
                max_tokens=4096,
            )
            self.assertEqual(judge._api_key(), "anthropic-fixture")
            judge.validate_configuration()
            with patch("anthropic.Anthropic") as constructor:
                client = SimpleNamespace(messages=SimpleNamespace())
                constructor.return_value = client
                self.assertIs(judge._get_client(), client)
                constructor.assert_called_once_with(
                    api_key="anthropic-fixture",
                    base_url="https://api.anthropic.com",
                    timeout=60.0,
                    max_retries=0,
                )

        with self.assertRaisesRegex(ValueError, "thinking_mode=adaptive"):
            self.module.VerpoJudge(
                model="claude-sonnet-5",
                base_url="https://api.anthropic.com",
                api_style="anthropic_messages",
                thinking_mode="enabled",
            )


if __name__ == "__main__":
    unittest.main()
