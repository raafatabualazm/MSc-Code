from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

from models.direct_compact_causal import (
    CONTRACT_SCHEMA,
    DirectCompactContract,
    sha256_file,
    tokenizer_fingerprint,
    validate_join_seal,
)
from scripts.training.build_qwen_sequence_kd import build as build_sequence_kd
from scripts.training.qwen_direct_compact_teacher_artifact import (
    DEFAULT_MODEL,
    OBJECTIVE_MODE_SEQUENCE_ONLY,
    SAMPLE_SEED_ALGORITHM,
    SAMPLES_PER_TASK,
    ArtifactError,
    JournalState,
    PromptRow,
    StudentTokenizerBinding,
    append_event,
    backend_identity,
    build_messages,
    collect_candidates,
    count_prompt_tokens,
    derived_sample_seed,
    ensure_run_header,
    file_record,
    load_f2_prompt_contract,
    make_orphan_reissue_authorization_event,
    make_orphan_reissue_attempt_event,
    make_slot_started_event,
    materialize_artifacts,
    normalize_response,
    read_jsonl,
    sha256_text,
    stable_sha256,
    validate_alibaba_model_studio_base_url,
    validate_qwen38_sequence_sampling,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class MappingTokenizer:
    def __init__(self, values: list[str]) -> None:
        self.values = list(values)
        self.to_id = {value: index for index, value in enumerate(self.values)}

    def get_vocab_size(self):
        return len(self.values)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return _Encoding(
            [self.to_id[text]] if text in self.to_id else []
        )

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self.values[int(index)] for index in ids)


class ContractTokenizer:
    def __init__(self) -> None:
        self._vocab = {"<pad>": 0, "<eos>": 1, "a": 2, "b": 3}
        self.special_tokens_map = {"pad_token": "<pad>", "eos_token": "<eos>"}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = None

    def get_vocab(self):
        return dict(self._vocab)


def make_contract(
    tokenizer_json_sha256: str = "c" * 64,
) -> DirectCompactContract:
    tokenizer = ContractTokenizer()
    return DirectCompactContract(
        schema=CONTRACT_SCHEMA,
        codec_sha256="a" * 64,
        codebook_sha256="b" * 64,
        tokenizer_json_sha256=tokenizer_json_sha256,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(tokenizer),
        model_config_sha256="d" * 64,
        decoder_model="fake/qwen-student",
        decoder_revision="immutable-revision",
        target_function="fn0",
        target_language="Dart",
        dfg_extractor_sha256="e" * 64,
        lossless_domain="scrubbed_canonical_graph",
        max_source_tokens=128,
        max_target_tokens=128,
        max_total_tokens=512,
        base_vocab_size=4,
        source_token_ids=(4,),
        source_token_expansions=((4, (2, 3)),),
    )


def logprob_token(token: str, logprob: float, alternatives: list[tuple[str, float]]):
    return types.SimpleNamespace(
        token=token,
        bytes=list(token.encode("utf-8")),
        logprob=logprob,
        top_logprobs=[
            types.SimpleNamespace(
                token=value,
                bytes=list(value.encode("utf-8")),
                logprob=score,
            )
            for value, score in alternatives
        ],
    )


def fake_response(
    request_index: int,
    *,
    model: str = DEFAULT_MODEL,
    content: str = "int fn0() => 1;",
    finish_reason: str = "stop",
):
    alternatives = [
        (content, -1.123456789012345),
        ("alt-a", -2.0),
        ("alt-b", -2.1),
        ("alt-c", -2.2),
        ("alt-d", -2.3),
    ]
    return types.SimpleNamespace(
        id=f"request-{request_index}",
        model=model,
        created=123456 + request_index,
        system_fingerprint="backend-a",
        service_tier="default",
        usage=types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ),
        choices=[
            types.SimpleNamespace(
                finish_reason=finish_reason,
                message=types.SimpleNamespace(content=content),
                logprobs=types.SimpleNamespace(
                    content=[
                        logprob_token(
                            content, -1.123456789012345, alternatives
                        )
                    ]
                ),
            )
        ],
    )


class FakeCompletions:
    def __init__(self, model_for_call=None) -> None:
        self.calls: list[dict] = []
        self.model_for_call = model_for_call or (lambda _: DEFAULT_MODEL)

    def create(self, **payload):
        self.calls.append(payload)
        index = len(self.calls) - 1
        return fake_response(index, model=self.model_for_call(index))


class FakeClient:
    def __init__(self, model_for_call=None) -> None:
        self.completions = FakeCompletions(model_for_call)
        self.chat = types.SimpleNamespace(completions=self.completions)


class FakeModerationError(Exception):
    code = "data_inspection_failed"
    status_code = 400
    body = {
        "code": "data_inspection_failed",
        "message": "blocked arbitrary codebook glyph",
    }


class ModerationThenSuccessCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **payload):
        self.calls.append(payload)
        index = len(self.calls) - 1
        if index == 0:
            raise FakeModerationError()
        return fake_response(index)


class ModerationThenSuccessClient:
    def __init__(self) -> None:
        self.completions = ModerationThenSuccessCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


class ConcurrentCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def create(self, **payload):
        with self.lock:
            index = len(self.calls)
            self.calls.append(payload)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return fake_response(index)
        finally:
            with self.lock:
                self.active -= 1


class ConcurrentClient:
    def __init__(self) -> None:
        self.completions = ConcurrentCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


class QwenTeacherArtifactTests(unittest.TestCase):
    system_prompt = "sealed-system"
    generation = {
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 12288,
        "extra_body": {"top_k": 101, "enable_thinking": False},
        "logprobs": True,
        "top_logprobs": 5,
    }

    def prompt(self) -> PromptRow:
        text = "F1\nBLOCKS\n0:!ret\nEND\n"
        return PromptRow(
            task_id="task-1",
            text=text,
            text_sha256=sha256_text(text),
            source_record_sha256="f" * 64,
            source_schema="frontier-compact-api-readable-v1",
        )

    def header(
        self, prompt: PromptRow, prompt_record=None, *, seed_base=None,
        objective_mode="require_top5", returned_model_exact=False,
        max_target_tokens=128, contract_record=None, tokenizer_record=None,
    ) -> dict:
        messages_hash = stable_sha256(build_messages(self.system_prompt, prompt))
        result = {
            "collection_mode": "offline_precompute_only_no_gradient_loop",
            "prompt_artifact": prompt_record
            or {"path": "prompt.jsonl", "sha256": "1" * 64, "size_bytes": 1},
            "prompt_manifest": {
                "path": "prompt.jsonl.manifest.json",
                "sha256": "2" * 64,
                "size_bytes": 1,
            },
            "f2_prompt_contract": {
                "representation_schema": "lossless-semantic-f2",
                "system_prompt_sha256": sha256_text(self.system_prompt),
            },
            "task_ids": [prompt.task_id],
            "prompt_bindings": [
                {
                    "task_id": prompt.task_id,
                    "serializer_text_sha256": prompt.text_sha256,
                    "source_record_sha256": prompt.source_record_sha256,
                    "request_messages_sha256": messages_hash,
                }
            ],
            "samples_per_task": SAMPLES_PER_TASK,
            "requested_model": DEFAULT_MODEL,
            "returned_model_must_equal_requested": bool(
                returned_model_exact
            ),
            "objective_mode": objective_mode,
            "generation_parameters": dict(self.generation),
            "transport": {
                "length_capped_response_policy": {
                    "same_task_draw_only": True,
                    "completed_draws_reissued": False,
                    "max_token_capacities": [
                        int(self.generation["max_tokens"]),
                        16384,
                        24576,
                    ],
                    "capped_responses_retained_by_hash": True,
                }
            },
            "target_length_contract": {
                "schema": "qwen-sequence-target-length-contract-v1",
                "trainer_contract": contract_record
                or {
                    "path": "mock-contract",
                    "sha256": "9" * 64,
                    "size_bytes": 1,
                },
                "trainer_contract_schema": CONTRACT_SCHEMA,
                "max_target_tokens": max_target_tokens,
                "student_tokenizer": tokenizer_record or {
                    "path": "mock-tokenizer",
                    "sha256": "4" * 64,
                },
                "student_eos_token_id": 5,
                "tokenization": {
                    "add_special_tokens": False,
                    "eos_policy": (
                        "append_exactly_once_if_final_token_is_not_eos"
                    ),
                    "matches_trainer_dataset_loader": True,
                    "truncation_permitted": False,
                    "overflow_filtering_permitted": False,
                    "overflow_resampling_permitted": False,
                },
                "target_source": {
                    "field": "choice.message.content",
                    "reasoning_field": "choice.message.reasoning_content",
                    "reasoning_excluded": True,
                    "final_dart_code_only_required": True,
                },
            },
        }
        if objective_mode == OBJECTIVE_MODE_SEQUENCE_ONLY:
            result["target_length_contract"]["target_source"][
                "final_dart_code_only_required"
            ] = False
            result["generation_parameters"].pop("logprobs", None)
            result["generation_parameters"].pop("top_logprobs", None)
            result["generation_parameters"]["extra_body"] = {
                "top_k": 101,
                "enable_thinking": True,
                "thinking_budget": 8192,
            }
        if seed_base is not None:
            result["sampling_seed_contract"] = {
                "algorithm": SAMPLE_SEED_ALGORITHM,
                "seed_base": seed_base,
                "unique_seed_required_per_task_draw": True,
                "provider_seed_honor_not_assumed": True,
                "response_seed_echo_required_to_attest_honor": True,
            }
        return result

    @staticmethod
    def verifier(candidate):
        passed = int(candidate["sample_index"]) % 2 == 0
        return {
            "compiled": passed,
            "passed": passed,
            "harness_completion_attested": passed,
            "diagnostic": "",
            "verifier_id": "mock-completion-verifier-v1",
            "verifier_sha256": "2" * 64,
            "tests_sha256": "3" * 64,
        }

    def binding(self) -> StudentTokenizerBinding:
        values = [
            "int fn0() => 1;",
            "alt-a",
            "alt-b",
            "alt-c",
            "alt-d",
            "<eos>",
        ]
        return StudentTokenizerBinding(
            MappingTokenizer(values),
            eos_token_id=5,
            tokenizer_record={"path": "mock-tokenizer", "sha256": "4" * 64},
        )

    def test_f2_contract_rejects_legacy_single_function_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "prompts.jsonl"
            prompt_path.write_text("{}\n", encoding="utf-8")
            prompt_record = file_record(prompt_path)
            manifest_path = root / "prompts.jsonl.manifest.json"
            manifest = {
                "schema": "verified-api-readable-compact-v2",
                "rows": 1,
                "output": prompt_record,
                "f2_prompt_contract": {
                    "representation_schema": "lossless-semantic-f2",
                    "system_prompt": self.system_prompt,
                    "system_prompt_sha256": sha256_text(self.system_prompt),
                    "tokenizer_sha256": "4" * 64,
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
                    # Old fn0-only manifests had the same outer F2 schema but
                    # none of the complete-user-function attestations.
                },
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ArtifactError, "F2 prompt manifest contract failed"
            ):
                load_f2_prompt_contract(
                    manifest_path,
                    expected_sha256=sha256_file(manifest_path),
                    prompt_record=prompt_record,
                    expected_rows=1,
                    student_tokenizer_sha256="4" * 64,
                )

    def collect(self, root: Path, *, prompt_record=None):
        prompt = self.prompt()
        client = FakeClient()
        journal = root / "teacher.journal.jsonl"
        state = collect_candidates(
            prompts=[prompt],
            client=client,
            journal_path=journal,
            header_payload=self.header(prompt, prompt_record),
            system_prompt=self.system_prompt,
            requested_model=DEFAULT_MODEL,
            generation_parameters=self.generation,
            required_function="fn0",
            verifier=self.verifier,
        )
        return prompt, client, journal, state

    def test_k8_is_independent_resumable_and_preserves_raw_logprobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, client, journal, state = self.collect(root)
            self.assertEqual(len(client.completions.calls), SAMPLES_PER_TASK)
            self.assertEqual(len(state.candidates), SAMPLES_PER_TASK)
            self.assertEqual(len(state.verifications), SAMPLES_PER_TASK)
            self.assertTrue(
                all(call["n"] == 1 for call in client.completions.calls)
            )
            candidate = next(iter(state.candidates.values()))
            token = candidate["chosen_tokens_with_top_logprobs"][0]
            self.assertEqual(token["bytes"], list(b"int fn0() => 1;"))
            self.assertEqual(token["logprob"], -1.123456789012345)
            self.assertEqual(len(token["top_logprobs"]), 5)
            self.assertEqual(candidate["response"]["request_id"], "request-0")
            self.assertEqual(candidate["response"]["returned_model"], DEFAULT_MODEL)
            self.assertEqual(
                backend_identity(candidate)["system_fingerprint"], "backend-a"
            )

            resumed = FakeClient()
            state2 = collect_candidates(
                prompts=[prompt],
                client=resumed,
                journal_path=journal,
                header_payload=self.header(prompt),
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
            )
            self.assertEqual(len(resumed.completions.calls), 0)
            self.assertEqual(len(state2.candidates), SAMPLES_PER_TASK)

    def test_input_moderation_uses_lossless_transport_and_keeps_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "F2\nD\n一mov:A,B\nS\n一\nB2\n一:一\nX\n"
            prompt = PromptRow(
                task_id="task-moderated",
                text=text,
                text_sha256=sha256_text(text),
                source_record_sha256="e" * 64,
                source_schema="verified-api-readable-compact-v2",
            )
            client = ModerationThenSuccessClient()
            journal = root / "teacher.journal.jsonl"
            state = collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=journal,
                header_payload=self.header(prompt),
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                max_retries=2,
            )
            self.assertEqual(
                len(client.completions.calls),
                SAMPLES_PER_TASK + 1,
            )
            self.assertEqual(len(state.candidates), SAMPLES_PER_TASK)
            first = next(
                candidate
                for candidate in state.candidates.values()
                if candidate["sample_index"] == 0
            )
            transport = first["request_transport"]
            self.assertTrue(transport["roundtrip_proven"])
            self.assertFalse(
                transport["canonical_raw_byte_request_executed"]
            )
            self.assertEqual(
                transport["moderation_error_code"],
                "data_inspection_failed",
            )
            canonical_user = client.completions.calls[0]["messages"][1][
                "content"
            ]
            transported_user = client.completions.calls[1]["messages"][1][
                "content"
            ]
            self.assertEqual(canonical_user, text)
            self.assertNotIn("一", transported_user)
            self.assertIn("~U4E00;", transported_user)
            self.assertEqual(
                stable_sha256(client.completions.calls[0]["messages"]),
                transport["canonical_messages_sha256"],
            )
            self.assertEqual(
                stable_sha256(client.completions.calls[1]["messages"]),
                transport["transport_messages_sha256"],
            )

    def orphan_journal(
        self,
        root: Path,
        *,
        preauthorize: bool = False,
        preattempt: bool = False,
    ):
        prompt = self.prompt()
        header = self.header(prompt, seed_base=44)
        header["provider_authorization"] = {
            "token_plan_automation_authorized": True,
            "attested_by": "workspace_operator",
            "scope": "automated_research_teacher_harvest",
        }
        header["implementation"] = {
            "collector": {"path": "collector.py", "sha256": "a" * 64},
            "artifact_core": {"path": "artifact.py", "sha256": "b" * 64},
        }
        journal = root / "orphan.jsonl"
        ensure_run_header(journal, header)
        request_parameters = {
            **self.generation,
            "seed": derived_sample_seed(44, prompt.task_id, 0),
        }
        started = make_slot_started_event(
            task_id=prompt.task_id,
            sample_index=0,
            prompt_sha256=stable_sha256(
                build_messages(self.system_prompt, prompt)
            ),
            request_parameters=request_parameters,
        )
        append_event(journal, started)
        if preauthorize:
            authorization = make_orphan_reissue_authorization_event(
                started,
                original_run_header_sha256=stable_sha256(header),
                original_collector_implementation=header[
                    "implementation"
                ],
                recovery_collector_implementation=header[
                    "implementation"
                ],
            )
            append_event(
                journal,
                authorization,
            )
            if preattempt:
                append_event(
                    journal,
                    make_orphan_reissue_attempt_event(
                        started,
                        authorization,
                        attempt_index=1,
                    ),
                )
        return prompt, header, journal, started

    def test_recovery_may_only_increase_operational_parallelism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()
            header = self.header(prompt)
            header["transport"] |= {
                "timeout_seconds": 600.0,
                "api_workers": 16,
                "local_verifier_workers": 16,
            }
            journal = root / "parallelism.jsonl"
            ensure_run_header(journal, header)

            increased = json.loads(json.dumps(header))
            increased["transport"]["api_workers"] = 32
            increased["transport"]["local_verifier_workers"] = 32
            with self.assertRaisesRegex(
                ArtifactError, "resume configuration/header differs"
            ):
                ensure_run_header(journal, increased)
            ensure_run_header(
                journal,
                increased,
                allow_recovery_implementation_change=True,
            )

            decreased = json.loads(json.dumps(header))
            decreased["transport"]["api_workers"] = 8
            with self.assertRaisesRegex(
                ArtifactError, "resume configuration/header differs"
            ):
                ensure_run_header(
                    journal,
                    decreased,
                    allow_recovery_implementation_change=True,
                )

            changed_timeout = json.loads(json.dumps(increased))
            changed_timeout["transport"]["timeout_seconds"] = 601.0
            with self.assertRaisesRegex(
                ArtifactError, "resume configuration/header differs"
            ):
                ensure_run_header(
                    journal,
                    changed_timeout,
                    allow_recovery_implementation_change=True,
                )

    def test_orphan_recovery_is_opt_in_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, header, journal, started = self.orphan_journal(root)

            default_client = FakeClient()
            with self.assertRaisesRegex(
                ArtifactError, "automatic reissue is forbidden"
            ):
                collect_candidates(
                    prompts=[prompt],
                    client=default_client,
                    journal_path=journal,
                    header_payload=header,
                    system_prompt=self.system_prompt,
                    requested_model=DEFAULT_MODEL,
                    generation_parameters=self.generation,
                    required_function="fn0",
                    verifier=self.verifier,
                    seed_base=44,
                )
            self.assertEqual(default_client.completions.calls, [])

            recovery_header = json.loads(json.dumps(header))
            recovery_header["implementation"] = {
                "collector": {"path": "collector.py", "sha256": "c" * 64},
                "artifact_core": {
                    "path": "artifact.py",
                    "sha256": "d" * 64,
                },
            }
            recovery_client = FakeClient()
            state = collect_candidates(
                prompts=[prompt],
                client=recovery_client,
                journal_path=journal,
                header_payload=recovery_header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
                authorize_orphan_reissue_with_duplicate_billing_risk=True,
            )
            self.assertEqual(len(recovery_client.completions.calls), 8)
            self.assertEqual(
                recovery_client.completions.calls[0]["seed"],
                started["request_parameters"]["seed"],
            )
            self.assertEqual(
                {
                    key: value
                    for key, value in recovery_client.completions.calls[0].items()
                    if key not in {"model", "messages"}
                },
                started["request_parameters"],
            )
            self.assertEqual(len(state.candidates), SAMPLES_PER_TASK)
            self.assertEqual(len(state.slots), SAMPLES_PER_TASK)
            self.assertEqual(len(state.terminals), SAMPLES_PER_TASK)
            self.assertEqual(len(state.reissue_authorizations), 1)
            self.assertEqual(len(next(iter(state.reissue_attempts.values()))), 1)
            authorization = next(iter(state.reissue_authorizations.values()))
            self.assertTrue(
                authorization[
                    "original_provider_request_may_have_billed_or_completed"
                ]
            )
            self.assertTrue(
                authorization["duplicate_provider_billing_risk_acknowledged"]
            )
            self.assertEqual(
                authorization["original_collector_implementation"],
                header["implementation"],
            )
            self.assertEqual(
                authorization["recovery_collector_implementation"],
                recovery_header["implementation"],
            )

            events_before = read_jsonl(journal)
            restarted_client = FakeClient()
            restarted = collect_candidates(
                prompts=[prompt],
                client=restarted_client,
                journal_path=journal,
                header_payload=recovery_header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
                authorize_orphan_reissue_with_duplicate_billing_risk=True,
            )
            self.assertEqual(restarted_client.completions.calls, [])
            self.assertEqual(read_jsonl(journal), events_before)
            self.assertEqual(len(restarted.terminals), SAMPLES_PER_TASK)
            self.assertEqual(
                sum(
                    row.get("event")
                    == "teacher_slot_orphan_reissue_authorized"
                    for row in events_before
                ),
                1,
            )
            self.assertEqual(
                sum(
                    row.get("event")
                    == "teacher_slot_orphan_reissue_attempt_started"
                    for row in events_before
                ),
                1,
            )
            self.assertEqual(
                sum(
                    row.get("event") == "teacher_slot_terminal"
                    for row in events_before
                ),
                SAMPLES_PER_TASK,
            )

            audit = materialize_artifacts(
                journal_path=journal,
                binding=self.binding(),
                parseable_output=root / "sequence.jsonl",
                rs_sft_output=root / "verified.jsonl",
                audit_output=root / "audit.json",
            )
            self.assertTrue(audit["production_ready"])
            self.assertEqual(audit["coverage"]["candidates"], SAMPLES_PER_TASK)

    def test_preauthorized_orphan_does_not_append_duplicate_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, header, journal, _ = self.orphan_journal(
                root, preauthorize=True
            )
            state = collect_candidates(
                prompts=[prompt],
                client=FakeClient(),
                journal_path=journal,
                header_payload=header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
                authorize_orphan_reissue_with_duplicate_billing_risk=True,
            )
            self.assertEqual(len(state.reissue_authorizations), 1)
            self.assertEqual(len(next(iter(state.reissue_attempts.values()))), 1)
            self.assertEqual(
                sum(
                    row.get("event")
                    == "teacher_slot_orphan_reissue_authorized"
                    for row in read_jsonl(journal)
                ),
                1,
            )

    def test_each_crash_recovery_gets_a_new_attempt_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, header, journal, _ = self.orphan_journal(
                root,
                preauthorize=True,
                preattempt=True,
            )
            state = collect_candidates(
                prompts=[prompt],
                client=FakeClient(),
                journal_path=journal,
                header_payload=header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
                authorize_orphan_reissue_with_duplicate_billing_risk=True,
            )
            attempts = next(iter(state.reissue_attempts.values()))
            self.assertEqual([row["attempt_index"] for row in attempts], [1, 2])
            self.assertEqual(
                attempts[0]["request_parameters"],
                attempts[1]["request_parameters"],
            )
            self.assertEqual(
                attempts[0]["requested_seed"],
                attempts[1]["requested_seed"],
            )
            slot = next(iter(state.reissue_attempts))
            self.assertEqual(
                state.terminals[slot]["orphan_reissue_attempt_id"],
                attempts[-1]["orphan_reissue_attempt_id"],
            )

    def test_recovery_transport_retry_gets_its_own_attempt_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, header, journal, _ = self.orphan_journal(root)

            class FailFirstCompletions:
                def __init__(inner_self):
                    inner_self.calls = []

                def create(inner_self, **payload):
                    inner_self.calls.append(payload)
                    if len(inner_self.calls) == 1:
                        raise TimeoutError("simulated indeterminate transport")
                    return fake_response(len(inner_self.calls) - 1)

            completions = FailFirstCompletions()
            client = types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=completions)
            )
            state = collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=journal,
                header_payload=header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
                max_retries=1,
                authorize_orphan_reissue_with_duplicate_billing_risk=True,
            )
            attempts = next(iter(state.reissue_attempts.values()))
            self.assertEqual([row["attempt_index"] for row in attempts], [1, 2])
            self.assertEqual(
                attempts[0]["request_parameters"],
                attempts[1]["request_parameters"],
            )
            self.assertEqual(len(completions.calls), SAMPLES_PER_TASK + 1)

    def test_existing_authorization_is_append_only_reauthorized_on_new_implementation(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, header, journal, _ = self.orphan_journal(
                root,
                preauthorize=True,
            )
            initial_events = read_jsonl(journal)
            prior_authorization = next(
                row
                for row in initial_events
                if row.get("event")
                == "teacher_slot_orphan_reissue_authorized"
            )
            changed_header = json.loads(json.dumps(header))
            changed_header["implementation"]["artifact_core"]["sha256"] = (
                "e" * 64
            )
            client = FakeClient()
            state = collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=journal,
                header_payload=changed_header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
                authorize_orphan_reissue_with_duplicate_billing_risk=True,
            )
            events = read_jsonl(journal)
            reauthorizations = [
                row
                for row in events
                if row.get("event")
                == "teacher_slot_orphan_reissue_implementation_reauthorized"
            ]
            self.assertEqual(len(reauthorizations), 1)
            reauthorization = reauthorizations[0]
            self.assertEqual(
                reauthorization["prior_orphan_reissue_authorization_id"],
                prior_authorization["orphan_reissue_authorization_id"],
            )
            self.assertEqual(
                reauthorization[
                    "previous_recovery_collector_implementation"
                ],
                header["implementation"],
            )
            self.assertEqual(
                reauthorization["recovery_collector_implementation"],
                changed_header["implementation"],
            )
            self.assertEqual(
                reauthorization["reissue_request_parameters"],
                prior_authorization["reissue_request_parameters"],
            )
            self.assertEqual(
                reauthorization["requested_seed"],
                prior_authorization["reissue_request_parameters"]["seed"],
            )
            self.assertEqual(len(client.completions.calls), SAMPLES_PER_TASK)
            self.assertEqual(len(state.terminals), SAMPLES_PER_TASK)
            self.assertEqual(len(state.reissue_authorizations), 1)
            self.assertEqual(
                next(iter(state.reissue_authorizations.values()))[
                    "orphan_reissue_authorization_id"
                ],
                reauthorization["orphan_reissue_authorization_id"],
            )

    def test_orphan_reissue_authorization_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, header, journal, started = self.orphan_journal(root)
            authorization = make_orphan_reissue_authorization_event(
                started,
                original_run_header_sha256=stable_sha256(header),
                original_collector_implementation=header["implementation"],
                recovery_collector_implementation=header["implementation"],
            )
            authorization["duplicate_provider_billing_risk_acknowledged"] = False
            append_event(journal, authorization)
            with self.assertRaisesRegex(
                ArtifactError,
                "orphan-reissue authorization payload hash mismatch",
            ):
                JournalState.load(
                    journal,
                    allow_indeterminate_slots=True,
                )

    def test_orphan_reissue_attempt_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, journal, _ = self.orphan_journal(
                root,
                preauthorize=True,
            )
            state = JournalState.load(
                journal,
                allow_indeterminate_slots=True,
            )
            slot = next(iter(state.starts))
            attempt = make_orphan_reissue_attempt_event(
                state.starts[slot],
                state.reissue_authorizations[slot],
                attempt_index=1,
            )
            attempt["requested_seed"] = -1
            append_event(journal, attempt)
            with self.assertRaisesRegex(
                ArtifactError,
                "orphan-reissue attempt payload hash mismatch",
            ):
                JournalState.load(
                    journal,
                    allow_indeterminate_slots=True,
                )

    def test_orphan_reissue_requires_sealed_token_plan_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, header, journal, _ = self.orphan_journal(root)
            header = {
                key: value
                for key, value in header.items()
                if key != "provider_authorization"
            }
            with self.assertRaisesRegex(
                ArtifactError, "sealed Token Plan automation authorization"
            ):
                collect_candidates(
                    prompts=[prompt],
                    client=FakeClient(),
                    journal_path=journal,
                    header_payload=header,
                    system_prompt=self.system_prompt,
                    requested_model=DEFAULT_MODEL,
                    generation_parameters=self.generation,
                    required_function="fn0",
                    verifier=self.verifier,
                    seed_base=44,
                    authorize_orphan_reissue_with_duplicate_billing_risk=True,
                )

    def test_collection_uses_bounded_parallel_calls_with_single_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()
            client = ConcurrentClient()
            state = collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=root / "parallel.jsonl",
                header_payload=self.header(prompt),
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                workers=4,
                verifier_workers=2,
            )
            self.assertEqual(len(state.candidates), SAMPLES_PER_TASK)
            self.assertEqual(len(state.verifications), SAMPLES_PER_TASK)
            self.assertGreater(client.completions.max_active, 1)

    def test_k8_uses_distinct_sealed_provider_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()
            client = FakeClient()
            collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=root / "seeded.jsonl",
                header_payload=self.header(prompt, seed_base=44),
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
            )
            seeds = [call["seed"] for call in client.completions.calls]
            self.assertEqual(len(seeds), SAMPLES_PER_TASK)
            self.assertEqual(len(set(seeds)), SAMPLES_PER_TASK)
            self.assertTrue(all(0 <= seed < 2**31 for seed in seeds))
            audit = materialize_artifacts(
                journal_path=root / "seeded.jsonl",
                binding=self.binding(),
                parseable_output=root / "seeded.sequence.jsonl",
                rs_sft_output=root / "seeded.rs.jsonl",
                audit_output=root / "seeded.audit.json",
            )
            self.assertTrue(audit["production_ready"])

    def test_tempered_or_nucleus_sampling_is_rejected_as_non_kl(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = self.prompt()
            generation = dict(self.generation)
            generation["top_p"] = 0.95
            with self.assertRaisesRegex(
                ArtifactError, "temperature=1.0, top_p=1.0"
            ):
                collect_candidates(
                    prompts=[prompt],
                    client=FakeClient(),
                    journal_path=Path(directory) / "invalid-sampling.jsonl",
                    header_payload={
                        **self.header(prompt),
                        "generation_parameters": generation,
                    },
                    system_prompt=self.system_prompt,
                    requested_model=DEFAULT_MODEL,
                    generation_parameters=generation,
                    required_function="fn0",
                    verifier=self.verifier,
                )

    def test_top5_rejects_hidden_reasoning_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = self.prompt()
            generation = {
                **self.generation,
                "extra_body": {
                    **self.generation["extra_body"],
                    "enable_thinking": True,
                    "thinking_budget": 8192,
                },
            }
            with self.assertRaisesRegex(ArtifactError, "same visible prefix"):
                collect_candidates(
                    prompts=[prompt],
                    client=FakeClient(),
                    journal_path=Path(directory) / "hidden-prefix.jsonl",
                    header_payload={
                        **self.header(prompt),
                        "generation_parameters": generation,
                    },
                    system_prompt=self.system_prompt,
                    requested_model=DEFAULT_MODEL,
                    generation_parameters=generation,
                    required_function="fn0",
                    verifier=self.verifier,
                )

    def test_sequence_only_is_explicit_and_does_not_claim_sparse_or_dense_kl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()

            class NoLogprobCompletions:
                def __init__(self):
                    self.calls = []

                def create(inner_self, **payload):
                    inner_self.calls.append(payload)
                    response = fake_response(len(inner_self.calls) - 1)
                    response.choices[0].logprobs = None
                    response.choices[0].message.reasoning_content = (
                        "private chain of thought that must never be a target"
                    )
                    return response

            completions = NoLogprobCompletions()
            client = types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=completions)
            )
            sequence_header = self.header(
                prompt, objective_mode=OBJECTIVE_MODE_SEQUENCE_ONLY
            )
            generation = dict(sequence_header["generation_parameters"])
            journal = root / "sequence-only.jsonl"
            collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=journal,
                header_payload=sequence_header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=generation,
                required_function="fn0",
                verifier=self.verifier,
            )
            self.assertTrue(
                all("logprobs" not in call for call in completions.calls)
            )
            audit = materialize_artifacts(
                journal_path=journal,
                binding=self.binding(),
                parseable_output=root / "sequence.jsonl",
                rs_sft_output=root / "rs.jsonl",
                audit_output=root / "audit.json",
            )
            self.assertTrue(
                audit["production_readiness"]["mc_sequence_forward_kl_nll"]
            )
            self.assertFalse(
                audit["production_readiness"]["sparse_top5_plus_tail"]
            )
            self.assertFalse(
                audit["capabilities"]["dense_full_vocabulary_kl"]
            )
            rows = read_jsonl(root / "sequence.jsonl")
            self.assertTrue(
                all(
                    row["sequence_target"] == "int fn0() => 1;"
                    and row["target_content_contract"]["reasoning_excluded"]
                    is True
                    and "private chain" not in row["sequence_target"]
                    for row in rows
                )
            )

    def test_sequence_sampling_settings_are_exactly_pinned(self):
        exact = {
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
        validate_qwen38_sequence_sampling(DEFAULT_MODEL, exact)
        for mutation in (
            {"n": 8},
            {"max_tokens": 12287},
            {"extra_body": {**exact["extra_body"], "thinking_budget": 8191}},
        ):
            changed = {**exact, **mutation}
            with self.assertRaisesRegex(ArtifactError, "exact qwen3.8"):
                validate_qwen38_sequence_sampling(DEFAULT_MODEL, changed)
        with self.assertRaisesRegex(ArtifactError, "exact qwen3.8"):
            validate_qwen38_sequence_sampling("qwen3.8-max", exact)

    def test_length_response_escalates_only_that_slot_and_keeps_completed_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = self.prompt()

            class LengthOnceCompletions:
                def __init__(inner_self):
                    inner_self.calls = []

                def create(inner_self, **payload):
                    inner_self.calls.append(payload)
                    index = len(inner_self.calls) - 1
                    return fake_response(
                        index,
                        finish_reason="length" if index == 0 else "stop",
                    )

            completions = LengthOnceCompletions()
            client = types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=completions)
            )
            state = collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=Path(directory) / "length-escalation.jsonl",
                header_payload=self.header(prompt),
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
            )
            self.assertEqual(len(state.candidates), SAMPLES_PER_TASK)
            self.assertEqual(len(completions.calls), SAMPLES_PER_TASK + 1)
            self.assertEqual(completions.calls[0]["max_tokens"], 12288)
            self.assertEqual(completions.calls[1]["max_tokens"], 16384)
            first_draw = next(
                row
                for row in state.candidates.values()
                if row["sample_index"] == 0
            )
            self.assertEqual(
                first_draw["request_parameters"]["max_tokens"], 16384
            )

    def test_qwen_endpoint_rejects_token_plan_and_arbitrary_hosts(self):
        self.assertEqual(
            validate_alibaba_model_studio_base_url(
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/"
            ),
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(
            validate_alibaba_model_studio_base_url(
                "https://workspace.ap-southeast-1.maas.aliyuncs.com/"
                "compatible-mode/v1"
            ),
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1",
        )
        for value in (
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1",
            "http://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "https://evil.example/compatible-mode/v1",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1?q=x",
            "https://user@dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ):
            with self.assertRaisesRegex(
                ArtifactError, "automation-capable"
            ):
                validate_alibaba_model_studio_base_url(value)

        self.assertEqual(
            validate_alibaba_model_studio_base_url(
                "https://token-plan.ap-southeast-1.maas.aliyuncs.com/"
                "compatible-mode/v1",
                token_plan_automation_authorized=True,
            ),
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1",
        )

    def test_sequence_only_retains_non_code_draws_without_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()

            class NonparseableCompletions:
                def __init__(self):
                    self.calls = []

                def create(inner_self, **payload):
                    inner_self.calls.append(payload)
                    return fake_response(
                        len(inner_self.calls) - 1,
                        content="Here is the answer: int fn0() => 1;",
                    )

            completions = NonparseableCompletions()
            client = types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=completions)
            )
            journal = root / "nonparseable.jsonl"
            sequence_header = self.header(
                prompt, objective_mode=OBJECTIVE_MODE_SEQUENCE_ONLY
            )
            sequence_generation = dict(
                sequence_header["generation_parameters"]
            )
            collect_candidates(
                prompts=[prompt],
                client=client,
                journal_path=journal,
                header_payload=sequence_header,
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=sequence_generation,
                required_function="fn0",
                verifier=self.verifier,
            )
            sequence_path = root / "sequence.jsonl"
            audit = materialize_artifacts(
                journal_path=journal,
                binding=StudentTokenizerBinding(
                    MappingTokenizer(
                        [
                            "Here is the answer: int fn0() => 1;",
                            "alt-a",
                            "alt-b",
                            "alt-c",
                            "alt-d",
                            "<eos>",
                        ]
                    ),
                    eos_token_id=5,
                    tokenizer_record={
                        "path": "mock-tokenizer",
                        "sha256": "4" * 64,
                    },
                ),
                parseable_output=sequence_path,
                rs_sft_output=root / "rs.jsonl",
                audit_output=root / "audit.json",
            )
            rows = read_jsonl(sequence_path)
            self.assertEqual(len(rows), SAMPLES_PER_TASK)
            self.assertEqual(
                audit["coverage"]["parseable_candidates"], SAMPLES_PER_TASK
            )
            self.assertEqual(
                audit["coverage"]["sequence_candidates"], SAMPLES_PER_TASK
            )
            self.assertTrue(audit["production_ready"])
            self.assertEqual(
                audit["target_length_gate"]["non_code_target_count"],
                SAMPLES_PER_TASK,
            )
            self.assertTrue(audit["target_length_gate"]["passed"])
            self.assertFalse(
                audit["target_length_gate"][
                    "final_dart_code_only_required"
                ]
            )
            self.assertTrue(
                all(
                    row["sequence_target"]
                    == "Here is the answer: int fn0() => 1;"
                    for row in rows
                )
            )

    def test_prompt_budget_count_includes_sealed_reserve(self):
        prompt = self.prompt()
        messages = build_messages(self.system_prompt, prompt)
        tokenizer = MappingTokenizer(
            [self.system_prompt, messages[1]["content"], "<eos>"]
        )
        count = count_prompt_tokens(
            messages, tokenizer, chat_overhead_reserve=256
        )
        self.assertEqual(count["system_tokens"], 1)
        self.assertEqual(count["user_tokens"], 1)
        self.assertEqual(count["estimated_prompt_tokens"], 258)

    def test_materializer_splits_parseable_and_verified_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, journal, _ = self.collect(root)
            parseable = root / "parseable.jsonl"
            rs_sft = root / "verified.jsonl"
            audit_path = root / "audit.json"
            audit = materialize_artifacts(
                journal_path=journal,
                binding=self.binding(),
                parseable_output=parseable,
                rs_sft_output=rs_sft,
                audit_output=audit_path,
            )
            self.assertTrue(audit["production_ready"])
            self.assertEqual(len(read_jsonl(parseable)), 8)
            self.assertEqual(len(read_jsonl(rs_sft)), 4)
            self.assertEqual(audit["coverage"]["chosen_mapping_coverage"], 1.0)
            self.assertEqual(audit["coverage"]["top_mapping_coverage"], 1.0)
            self.assertEqual(
                audit["coverage"][
                    "sequences_whose_chosen_bytes_reconstruct_raw_content"
                ],
                8,
            )
            self.assertEqual(
                audit["coverage"]["logged_eos_sequence_coverage"], 0.0
            )
            self.assertFalse(
                audit["capabilities"]["dense_full_vocabulary_kl"]
            )
            self.assertEqual(
                audit["sampling"]["unique_final_sequences_per_task"],
                {"task-1": 1},
            )
            self.assertFalse(
                audit["sampling"][
                    "pathological_all_tasks_have_identical_k8_draws"
                ]
            )
            self.assertFalse(
                audit["sampling"]["provider_seed_honor_assumed"]
            )
            self.assertFalse(
                audit["sampling"][
                    "duplicates_filtered_from_sequence_training"
                ]
            )
            self.assertTrue(audit["target_length_gate"]["passed"])
            self.assertEqual(
                audit["target_length_gate"]["targets_checked"],
                SAMPLES_PER_TASK,
            )

    def test_eos_inclusive_target_overflow_fails_with_draw_diagnostics(self):
        class SplitTargetTokenizer(MappingTokenizer):
            def encode(inner_self, text, add_special_tokens=False):
                if text == "int fn0() => 1;":
                    return _Encoding([0, 0])
                return super().encode(text, add_special_tokens=add_special_tokens)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()
            journal = root / "overflow.jsonl"
            collect_candidates(
                prompts=[prompt],
                client=FakeClient(),
                journal_path=journal,
                header_payload=self.header(prompt, max_target_tokens=2),
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
            )
            binding = StudentTokenizerBinding(
                SplitTargetTokenizer(
                    [
                        "int fn0() => 1;",
                        "alt-a",
                        "alt-b",
                        "alt-c",
                        "alt-d",
                        "<eos>",
                    ]
                ),
                eos_token_id=5,
                tokenizer_record={
                    "path": "mock-tokenizer",
                    "sha256": "4" * 64,
                },
            )
            audit = materialize_artifacts(
                journal_path=journal,
                binding=binding,
                parseable_output=root / "sequence.jsonl",
                rs_sft_output=root / "rs.jsonl",
                audit_output=root / "audit.json",
            )
            self.assertFalse(audit["production_ready"])
            gate = audit["target_length_gate"]
            self.assertEqual(gate["overflow_count"], SAMPLES_PER_TASK)
            self.assertEqual(len(gate["overflow_diagnostics"]), SAMPLES_PER_TASK)
            first = gate["overflow_diagnostics"][0]
            self.assertEqual(first["eos_inclusive_target_token_count"], 3)
            self.assertEqual(first["max_target_tokens"], 2)
            self.assertEqual(first["overflow_by_tokens"], 1)
            self.assertIn("task_id", first)
            self.assertIn("sample_index", first)
            self.assertFalse(gate["failure_policy"]["truncate"])
            self.assertFalse(gate["failure_policy"]["filter_draw"])
            self.assertFalse(gate["failure_policy"]["resample_draw"])

    def test_launcher_overlaps_gold_gpu_only_after_dry_preflight_and_migration(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_qwen_sequence_kd_warmstart.sh"
        ).read_text(encoding="utf-8")
        invocations = [
            index
            for index, line in enumerate(launcher.splitlines())
            if line.strip() == "start_gold_adapt"
        ]
        self.assertEqual(len(invocations), 2)
        lines = launcher.splitlines()
        preflight_line = next(
            index
            for index, line in enumerate(lines)
            if '"${PREFLIGHT_ARGS[@]}"' in line
        )
        migration_line = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == "prepare_overlay_migration"
        )
        probe_line = next(
            index
            for index, line in enumerate(lines)
            if "-m scripts.training.probe_qwen_teacher_contract" in line
        )
        build_line = next(
            index
            for index, line in enumerate(lines)
            if "-m scripts.training.build_qwen_sequence_kd" in line
        )
        finish_line = max(
            index
            for index, line in enumerate(lines)
            if line.strip() == "finish_gold_adapt"
        )
        self.assertLess(preflight_line, migration_line)
        self.assertLess(migration_line, invocations[0])
        self.assertLess(invocations[0], probe_line)
        self.assertLess(build_line, invocations[1])
        self.assertLess(invocations[0], build_line)
        self.assertGreater(finish_line, build_line)
        self.assertIn("--migrate_warmstart_only", launcher)
        self.assertIn("--validate_migrated_warmstart_only", launcher)
        self.assertIn('--target-contract "${COMPACT_CONTRACT}"', launcher)
        self.assertIn(".target_length_gate.passed == true", launcher)

    def test_provider_seed_echo_is_audited_and_mismatch_fails(self):
        request = dict(self.generation)
        request["seed"] = 123
        matching = fake_response(0)
        matching.seed = 123
        candidate = normalize_response(
            matching,
            task_id="task-1",
            sample_index=0,
            prompt_sha256="5" * 64,
            requested_model=DEFAULT_MODEL,
            request_parameters=request,
            required_function="fn0",
        )
        self.assertEqual(
            candidate["response"]["provider_reported_seed"], 123
        )
        mismatched = fake_response(1)
        mismatched.seed = 124
        with self.assertRaisesRegex(
            ArtifactError, "seed different from the request"
        ):
            normalize_response(
                mismatched,
                task_id="task-1",
                sample_index=0,
                prompt_sha256="5" * 64,
                requested_model=DEFAULT_MODEL,
                request_parameters=request,
                required_function="fn0",
            )

    def test_material_negative_tail_is_not_clamped(self):
        response = fake_response(0)
        for alternative in response.choices[0].logprobs.content[0].top_logprobs:
            alternative.logprob = 0.0
        with self.assertRaisesRegex(ArtifactError, "negative inferred tail"):
            normalize_response(
                response,
                task_id="task-1",
                sample_index=0,
                prompt_sha256="5" * 64,
                requested_model=DEFAULT_MODEL,
                request_parameters=self.generation,
                required_function="fn0",
            )

    def test_require_top5_rejects_missing_logprobs_immediately(self):
        response = fake_response(0)
        response.choices[0].logprobs = None
        with self.assertRaisesRegex(
            ArtifactError, "omitted requested content logprobs"
        ):
            normalize_response(
                response,
                task_id="task-1",
                sample_index=0,
                prompt_sha256="5" * 64,
                requested_model=DEFAULT_MODEL,
                request_parameters=self.generation,
                required_function="fn0",
            )

    def test_backend_drift_stops_and_preserves_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()
            client = FakeClient(
                lambda index: DEFAULT_MODEL if index == 0 else "different-backend"
            )
            journal = root / "drift.jsonl"
            with self.assertRaisesRegex(ArtifactError, "different model/backend"):
                collect_candidates(
                    prompts=[prompt],
                    client=client,
                    journal_path=journal,
                    header_payload=self.header(prompt),
                    system_prompt=self.system_prompt,
                    requested_model=DEFAULT_MODEL,
                    generation_parameters=self.generation,
                    required_function="fn0",
                    verifier=self.verifier,
                )
            candidates = [
                row
                for row in read_jsonl(journal)
                if row.get("event") == "teacher_candidate"
            ]
            self.assertEqual(len(candidates), 2)

    def test_exact_returned_model_policy_rejects_alias_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = self.prompt()
            client = FakeClient(lambda _: "different-model")
            journal = root / "exact-model.jsonl"
            with self.assertRaisesRegex(
                ArtifactError, "invalid_teacher_response"
            ):
                collect_candidates(
                    prompts=[prompt],
                    client=client,
                    journal_path=journal,
                    header_payload=self.header(
                        prompt, returned_model_exact=True
                    ),
                    system_prompt=self.system_prompt,
                    requested_model=DEFAULT_MODEL,
                    generation_parameters=self.generation,
                    required_function="fn0",
                    verifier=self.verifier,
                    require_returned_model_exact=True,
                )
            events = read_jsonl(journal)
            self.assertEqual(
                sum(
                    row.get("event") == "teacher_rejected_draw"
                    for row in events
                ),
                1,
            )
            self.assertEqual(
                sum(
                    row.get("event") == "teacher_slot_terminal"
                    for row in events
                ),
                1,
            )
            resumed = FakeClient()
            with self.assertRaisesRegex(
                ArtifactError, "permanently failed"
            ):
                collect_candidates(
                    prompts=[prompt],
                    client=resumed,
                    journal_path=journal,
                    header_payload=self.header(
                        prompt, returned_model_exact=True
                    ),
                    system_prompt=self.system_prompt,
                    requested_model=DEFAULT_MODEL,
                    generation_parameters=self.generation,
                    required_function="fn0",
                    verifier=self.verifier,
                    require_returned_model_exact=True,
                )
            self.assertEqual(resumed.completions.calls, [])

    def test_journal_chain_detects_deleted_durable_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, journal, _ = self.collect(root)
            rows = journal.read_text(encoding="utf-8").splitlines()
            candidate_index = next(
                index
                for index, line in enumerate(rows)
                if json.loads(line).get("event") == "teacher_candidate"
            )
            del rows[candidate_index]
            journal.write_text(
                "\n".join(rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ArtifactError, "hash chain|chain head"
            ):
                JournalState.load(journal)

    def test_sealed_sequence_builder_uses_all_draws_with_equal_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from tokenizers import Tokenizer, models

            tokenizer_path = root / "tokenizer.json"
            Tokenizer(
                models.WordLevel(
                    {
                        "int fn0() => 1;": 0,
                        "alt-a": 1,
                        "alt-b": 2,
                        "alt-c": 3,
                        "alt-d": 4,
                        "<eos>": 5,
                    },
                    unk_token="alt-a",
                )
            ).save(str(tokenizer_path))
            tokenizer_sha256 = sha256_file(tokenizer_path)
            contract = make_contract(tokenizer_sha256)
            contract_path = root / "contract.json"
            contract_path.write_text(
                json.dumps(contract.as_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            train = root / "train.jsonl"
            train_row = {
                "task_id": "task-1",
                "lang": "Dart",
                "function": "fn0",
                "dart_source": "int fn0() => 7;",
                "compact_input_ids": [4],
                "compact_codec_sha256": contract.codec_sha256,
                "compact_codebook_sha256": contract.codebook_sha256,
                "compact_tokenizer_sha256": contract.tokenizer_json_sha256,
            }
            train.write_text(
                json.dumps(train_row, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            train_seal = root / "train.seal.json"
            train_seal.write_text(
                json.dumps(
                    {
                        "schema": "compact-public-private-join-seal-v1",
                        "selected_role": "fit",
                        "output_sha256": sha256_file(train),
                        "contract_sha256": sha256_file(contract_path),
                        "rows": 1,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            prompt_text = self.prompt().text
            prompt_row = {
                "schema": "frontier-compact-api-readable-v1",
                "representation_schema": "lossless-semantic-f2",
                "system_prompt_sha256": sha256_text(self.system_prompt),
                "task_id": "task-1",
                "text": prompt_text,
                "text_sha256": sha256_text(prompt_text),
                "compact_ids_sha256": stable_sha256([4]),
                "compact_text_sha256": "6" * 64,
                "canonical_sha256": "7" * 64,
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
            verified_prompt = PromptRow(
                task_id="task-1",
                text=prompt_text,
                text_sha256=sha256_text(prompt_text),
                source_record_sha256=stable_sha256(prompt_row),
                source_schema=prompt_row["schema"],
                representation_schema="lossless-semantic-f2",
                system_prompt_sha256=sha256_text(self.system_prompt),
            )
            prompt_record = file_record(prompt_path)
            client = FakeClient()
            journal_path = root / "teacher.jsonl"
            collect_candidates(
                prompts=[verified_prompt],
                client=client,
                journal_path=journal_path,
                header_payload=self.header(
                    verified_prompt,
                    prompt_record,
                    seed_base=44,
                    returned_model_exact=True,
                    contract_record=file_record(contract_path),
                    tokenizer_record=file_record(tokenizer_path),
                ),
                system_prompt=self.system_prompt,
                requested_model=DEFAULT_MODEL,
                generation_parameters=self.generation,
                required_function="fn0",
                verifier=self.verifier,
                seed_base=44,
                require_returned_model_exact=True,
            )
            parseable = root / "parseable.jsonl"
            rs_sft = root / "rs.jsonl"
            audit_path = root / "audit.json"
            materialize_artifacts(
                journal_path=journal_path,
                binding=StudentTokenizerBinding(
                    Tokenizer.from_file(str(tokenizer_path)),
                    eos_token_id=5,
                    tokenizer_record=file_record(tokenizer_path),
                ),
                parseable_output=parseable,
                rs_sft_output=rs_sft,
                audit_output=audit_path,
            )
            output = root / "sequence.jsonl"
            output_seal = root / "sequence.seal.json"
            schedule = root / "sequence.schedule.jsonl"
            manifest_path = root / "sequence.build.json"
            args = argparse.Namespace(
                compact_train_jsonl=train,
                compact_train_seal=train_seal,
                contract=contract_path,
                prompt_jsonl=prompt_path,
                expected_prompt_sha256=sha256_file(prompt_path),
                teacher_parseable_jsonl=parseable,
                expected_teacher_parseable_sha256=sha256_file(parseable),
                teacher_journal=journal_path,
                expected_teacher_journal_sha256=sha256_file(journal_path),
                teacher_audit_json=audit_path,
                expected_teacher_audit_sha256=sha256_file(audit_path),
                output_jsonl=output,
                output_seal=output_seal,
                schedule_output=schedule,
                build_manifest=manifest_path,
                student_tokenizer_json=tokenizer_path,
                expected_student_tokenizer_sha256=tokenizer_sha256,
                gold_replay_fraction=0.2,
                seed=44,
            )
            manifest = build_sequence_kd(args)
            self.assertEqual(manifest["counts"]["teacher_draw_rows"], 8)
            self.assertEqual(manifest["counts"]["gold_replay_rows"], 2)
            self.assertEqual(manifest["counts"]["output_rows"], 10)
            self.assertFalse(manifest["objective"]["confidence_weighting"])
            self.assertFalse(manifest["objective"]["dense_token_kl"])
            schedule_rows = read_jsonl(schedule)
            teacher_schedule = [
                row for row in schedule_rows if row["kind"] == "teacher_draw"
            ]
            self.assertEqual(len(teacher_schedule), 8)
            self.assertEqual(
                len({row["candidate_id"] for row in teacher_schedule}), 8
            )
            self.assertTrue(
                all(row["draw_weight"] == 1.0 for row in teacher_schedule)
            )
            validate_join_seal(
                output, output_seal, contract_path, expected_role="fit"
            )


if __name__ == "__main__":
    unittest.main()
