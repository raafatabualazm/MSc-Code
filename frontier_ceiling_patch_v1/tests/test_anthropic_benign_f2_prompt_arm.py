from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"
sys.path.insert(0, str(PATCH))

import anthropic_benign_f2_prompt_arm as arm
import anthropic_opus5_o1_benign_batch as opus_runner
import frontier_passk_anthropic_benign_batch as runner
import frontier_core as core
from frontier_f2 import F2_SYSTEM_PROMPT, serialize_f2


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _CharacterTokenizer:
    """Tiny reversible tokenizer sufficient for a focused F2 fixture."""

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        return _Encoding([ord(value) for value in text])

    def encode_batch(
        self, values: list[str], add_special_tokens: bool = False
    ) -> list[_Encoding]:
        return [
            self.encode(value, add_special_tokens=add_special_tokens)
            for value in values
        ]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(value) for value in ids)


def _f2() -> str:
    canonical = {
        "architecture": "x86_64",
        "entry_blocks": [0],
        "blocks": [
            {"id": 0, "instructions": ["mov rax,0x1", "jne @B2"]},
            {"id": 1, "instructions": ["xor rax,rax", "ret"]},
            {"id": 2, "instructions": ["ret"]},
        ],
        "cfg_edges": [
            {"source": 0, "target": 2, "edge_type": "conditional_true"},
            {"source": 0, "target": 1, "edge_type": "conditional_false"},
        ],
    }
    return serialize_f2(
        "// constants: [1]\n",
        canonical,
        tokenizer=_CharacterTokenizer(),
        visible_symbols=[chr(value) for value in range(0x4E00, 0x4E80)],
    )


def _source_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": F2_SYSTEM_PROMPT},
        {"role": "user", "content": _f2()},
    ]


def _provenance() -> dict:
    invariants = {
        name: True
        for name in (
            "input_mode_is_prematerialized_f2",
            "selected_pair_arm_artifact_bindings_verified",
            "paired_acceptance_test_sequence_sha256_verified",
            "ordered_prompt_eval_task_ids_identical",
            "per_row_f2_syntax_verified",
            "per_row_f2_verification_map_verified",
            "prompts_never_truncated",
            "tests_not_exposed_to_teacher",
            "source_not_exposed_to_teacher",
            "exact_private_source_and_tests_absent_from_f2_text",
        )
    }
    return {
        "task_set_sha256": "1" * 64,
        "acceptance_test_sequence_sha256": "2" * 64,
        "source_pair_manifest_claims": {
            "schema": "frontier-enrichment-pair-v1",
            "sha256": "3" * 64,
            "pair_arm_key": "opus_real_fn0_cfg",
            "rows": 175,
        },
        "source_eval_seal_claims": {
            "selected_role": "measure",
        },
        "source_f2_manifest_claims": {
            "representation_schema": "lossless-semantic-f2",
        },
        "preflight_invariants": invariants,
    }


def _native_attempt(custom_id: str, stop_reason: str) -> dict:
    return {
        "custom_id": custom_id,
        "native_batch_result": {
            "result": {
                "type": "succeeded",
                "message": {"stop_reason": stop_reason},
            }
        },
    }


def _terminal(
    task_id: str, sample_index: int, finish_reason: str
) -> dict:
    return {
        "task_id": task_id,
        "sample_index": sample_index,
        "custom_id": f"{task_id}-{sample_index}",
        "finish_reason": finish_reason,
    }


def test_overlay_preserves_exact_f2_bytes_and_decoded_semantics():
    source = _source_messages()
    transformed, proof = arm.build_benign_messages(source)

    recovered = arm.extract_exact_f2_payload(transformed[1]["content"])
    assert recovered.encode("utf-8") == source[1]["content"].encode("utf-8")
    assert transformed[0]["content"].endswith(F2_SYSTEM_PROMPT)
    assert proof["f2_payload_utf8_bytes_identical"] is True
    assert proof["decoded_canonical_semantics_identical"] is True
    assert proof["source_f2_payload_sha256"] == core.sha256_text(recovered)


def test_overlay_is_distinct_and_hash_bound():
    source = _source_messages()
    transformed, _proof = arm.build_benign_messages(source)
    contract = arm.arm_contract()

    assert transformed != source
    assert contract["arm_label"] == arm.ARM_LABEL
    assert contract["source_system_prompt_sha256"] == core.sha256_text(
        F2_SYSTEM_PROMPT
    )
    assert contract["runtime_system_prompt_sha256"] == core.sha256_text(
        transformed[0]["content"]
    )
    assert (
        contract["authorization_evidence"]["artifact_verified"] is False
    )
    assert contract["required_refusal_reporting"][
        "unconditional_pass_at_k_remains_primary"
    ] is True


def test_source_system_or_payload_tamper_fails_closed():
    wrong_system = _source_messages()
    wrong_system[0]["content"] += " changed"
    with pytest.raises(core.PreflightError, match="sealed original"):
        arm.build_benign_messages(wrong_system)

    transformed, _proof = arm.build_benign_messages(_source_messages())
    transformed[1]["content"] = transformed[1]["content"][:-1]
    with pytest.raises(core.PreflightError, match="malformed"):
        arm.extract_exact_f2_payload(transformed[1]["content"])


def test_provenance_gate_distinguishes_proof_from_operator_attestation():
    evidence = arm.verify_source_provenance(_provenance())
    assert evidence["verified_from_sealed_artifacts"]["rows"] == 175
    assert evidence["verified_from_sealed_artifacts"]["evaluation_role"] == (
        "measure"
    )
    assert evidence["operator_attested_not_artifact_verified"][
        "analysis_is_authorized"
    ] is True
    assert evidence["artifact_provenance_limits"]

    bad = _provenance()
    bad["preflight_invariants"]["tests_not_exposed_to_teacher"] = False
    with pytest.raises(core.PreflightError, match="tests_not_exposed"):
        arm.verify_source_provenance(bad)


def test_complete_arm_application_writes_hash_bound_manifest(tmp_path: Path):
    payload = _f2()
    plans = []
    prompt_map = {}
    for index in range(175):
        task_id = f"task-{index:03d}"
        messages = [
            {"role": "system", "content": F2_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
        prompt_sha = core.stable_sha256(messages)
        plans.append(
            {
                "task_id": task_id,
                "messages": messages,
                "prompt_sha256": prompt_sha,
            }
        )
        prompt_map[task_id] = {
            "task_id": task_id,
            "messages": messages,
            "prompt_sha256": prompt_sha,
        }
    provenance = _provenance()
    provenance["artifacts"] = {"pair_manifest": {"sha256": "3" * 64}}
    args = SimpleNamespace(
        operator_attests_authorized_benchmark=True,
        chat_overhead_reserve=256,
        max_prompt_tokens=100_000,
    )

    transformed, records, manifest = arm.apply_prompt_arm(
        tokenizer=_CharacterTokenizer(),
        plans=plans,
        prompt_map=prompt_map,
        config_sha256="6" * 64,
        provenance=provenance,
        args=args,
        out=tmp_path,
    )

    assert len(transformed) == len(records) == manifest["tasks"] == 175
    assert manifest["all_f2_payload_utf8_bytes_identical"] is True
    assert manifest["all_decoded_f2_semantics_identical"] is True
    assert (tmp_path / "benign_prompt_arm_manifest.json").is_file()
    assert len(
        core.load_jsonl(tmp_path / "prompts.jsonl", "test prompts")
    ) == 175
    for plan in transformed:
        recovered = arm.extract_exact_f2_payload(
            plan["messages"][1]["content"]
        )
        assert recovered.encode("utf-8") == payload.encode("utf-8")


def test_runner_requires_explicit_operator_attestation(monkeypatch):
    monkeypatch.delenv(
        "ANTHROPIC_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK", raising=False
    )
    monkeypatch.setattr(
        runner,
        "_ORIGINAL_BATCH_PARSE_ARGS",
        lambda values: SimpleNamespace(
            parsed_values=values, dataset_label="common175_test"
        ),
    )
    with pytest.raises(SystemExit):
        runner.parse_args([])

    args = runner.parse_args(
        ["--attest-authorized-benchmark", "--action", "preflight"]
    )
    assert args.operator_attests_authorized_benchmark is True
    assert args.parsed_values == ["--action", "preflight"]
    assert args.dataset_label.endswith(arm.ARM_LABEL)


def test_opus_runner_requires_attestation_and_installs_separate_engine(
    monkeypatch,
):
    monkeypatch.delenv(
        "ANTHROPIC_OPERATOR_ATTESTS_AUTHORIZED_BENCHMARK", raising=False
    )
    monkeypatch.setattr(
        opus_runner,
        "_ORIGINAL_OPUS_PARSE_ARGS",
        lambda values: SimpleNamespace(
            parsed_values=values, dataset_label="common175_opus_test"
        ),
    )
    with pytest.raises(SystemExit):
        opus_runner.parse_args([])
    args = opus_runner.parse_args(
        ["--attest-authorized-benchmark", "--action", "preflight"]
    )
    assert args.operator_attests_authorized_benchmark is True
    assert args.dataset_label.endswith(arm.ARM_LABEL)

    original_prepare = opus_runner.audited.prepare_run
    with opus_runner.configured_engine():
        assert opus_runner.batch.parse_args is opus_runner.parse_args
        assert (
            opus_runner.batch.config_for_hash
            is opus_runner.config_for_hash
        )
        assert opus_runner.audited.prepare_run is opus_runner.prepare_run
    assert opus_runner.audited.prepare_run is original_prepare


def test_refusal_report_separates_native_refusals_from_other_failures():
    tasks = ["a", "b"]
    terminals = [
        _terminal("a", 0, "content_filter"),
        _terminal("a", 1, "content_filter"),
        _terminal("b", 0, "content_filter"),
        _terminal("b", 1, "stop"),
    ]
    attempts = [
        _native_attempt("a-0", "refusal"),
        _native_attempt("a-1", "refusal"),
        _native_attempt("b-0", "refusal"),
        _native_attempt("b-1", "end_turn"),
    ]
    report = arm.build_refusal_report(
        terminals=terminals,
        attempts=attempts,
        task_ids=tasks,
        k=2,
        config_sha256="4" * 64,
    )

    assert report["native_refusal_slots"] == 3
    assert report["native_refusal_rate_among_terminal_slots"] == 0.75
    assert report["tasks_with_any_native_refusal"] == 2
    assert report["tasks_with_all_k_slots_native_refusal"] == 1
    assert report[
        "native_refusal_and_normalized_content_filter_sets_identical"
    ] is True
    assert report["metric_interpretation"][
        "unconditional_pass_at_k_remains_primary"
    ] is True


def test_refusal_normalization_mismatch_fails_closed():
    with pytest.raises(core.PreflightError, match="not normalized"):
        arm.build_refusal_report(
            terminals=[_terminal("a", 0, "stop")],
            attempts=[_native_attempt("a-0", "refusal")],
            task_ids=["a"],
            k=1,
            config_sha256="5" * 64,
        )
