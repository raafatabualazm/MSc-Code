from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "frontier_ceiling_patch_v1"

sys.path.insert(0, str(PATCH))
import frontier_f2 as f2
import frontier_core as core
import frontier_passk as runner


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_fixture(tmp_path: Path):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    preamble = core.constant_preamble(["hello"], [7])
    tokenizer.train_from_iterator(
        [preamble, "mov rax,rbx", "entry basic blocks cfg"], trainer=trainer
    )
    tokenizer.add_tokens([chr(value) for value in range(0x4E00, 0x4E40)])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    base_vocab = tokenizer.get_vocab_size(with_added_tokens=True)

    codec_path = ROOT / "scripts" / "data" / "build_compact_qwen_v1.py"
    codec_spec = importlib.util.spec_from_file_location("fixture_codec", codec_path)
    assert codec_spec and codec_spec.loader
    codec = importlib.util.module_from_spec(codec_spec)
    codec_spec.loader.exec_module(codec)

    expansions = ["mov rax,rbx"]
    atoms = ["<I0>", "<B0>"] + list(codec.CONTROL)
    atom_ids = {atom: base_vocab + index for index, atom in enumerate(atoms)}
    source_expansions = {str(token_id): [0] for token_id in atom_ids.values()}
    codebook = {
        "schema": "compact-qwen-v1-codebook",
        "expansions": expansions,
        "source_atom_ids": atom_ids,
        "source_token_expansions": source_expansions,
        "model_vocab_size": base_vocab,
        "base_vocab_size": base_vocab,
        "tokenizer_json_sha256": sha(tokenizer_path),
    }
    codebook_path = tmp_path / "codebook.json"
    write_json(codebook_path, codebook)

    contract = {
        "schema": "direct-compact-causal-v1",
        "codec_sha256": sha(codec_path),
        "codebook_sha256": sha(codebook_path),
        "tokenizer_json_sha256": sha(tokenizer_path),
        "base_vocab_size": base_vocab,
        "max_source_tokens": 9000,
        "source_token_ids": sorted(atom_ids.values()),
        "source_token_expansions": source_expansions,
    }
    contract_path = tmp_path / "contract.json"
    write_json(contract_path, contract)

    constants = {
        "task_id": "heldout_1",
        "strings": ["hello"],
        "numbers": [7],
        "err": None,
    }
    constants_path = tmp_path / "real_constants.jsonl"
    constants_path.write_text(
        json.dumps(constants, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    canonical = {
        "architecture": "x86_64",
        "entry_blocks": [0],
        "blocks": [{"id": 0, "instructions": ["mov rax,rbx", "mov rax,rbx"]}],
        "cfg_edges": [],
    }
    graph_text = codec.encode(canonical, {"mov rax,rbx": 0})
    graph_ids = codec.compact_ids(graph_text, tokenizer, atom_ids)
    prefix_ids = [
        token_id
        for token_id in tokenizer.encode(preamble).ids
        if token_id < base_vocab
    ][:256]
    row = {
        "task_id": "heldout_1",
        "compact_input_ids": prefix_ids + graph_ids,
        "compact_codec_sha256": contract["codec_sha256"],
        "compact_codebook_sha256": contract["codebook_sha256"],
        "compact_tokenizer_sha256": contract["tokenizer_json_sha256"],
    }
    bundle = core.CompactArtifactBundle(
        contract_path=contract_path,
        codebook_path=codebook_path,
        tokenizer_path=tokenizer_path,
        codec_path=codec_path,
        constants_path=constants_path,
        expected_constants_sha256=sha(constants_path),
    )
    return bundle, row


def test_hash_checked_compact_serialization_is_readable_and_explicit(tmp_path: Path):
    bundle, row = build_fixture(tmp_path)
    record = core.prepare_api_readable_compact(bundle, row)
    text = record["text"]
    assert record["verified"]["codec_token_id_roundtrip"] is True
    assert record["verified"]["compact_semantic_f2_roundtrip"] is True
    assert record["representation_schema"] == "lossless-semantic-f2"
    assert record["system_prompt_sha256"] == core.sha256_text(
        core.COMPACT_F2_SYSTEM_PROMPT
    )
    assert "hello" in text
    assert "numbers: 7" in text
    assert text.startswith("F2\nC")
    assert "\nAx86_64\n" in text
    assert "\nD\n" in text
    assert "\nB2\n" in text
    prefix, canonical = core.decode_f2(text)
    assert prefix == bundle.prepare(row).constant_prefix_text
    assert canonical == bundle.prepare(row).canonical
    assert "<I0>" not in text
    assert "<B0>" not in text


def test_f2_roundtrips_sequences_cfg_targets_and_arbitrary_constants(tmp_path: Path):
    bundle, _ = build_fixture(tmp_path)
    repeated = [
        "mov rax,rbx",
        "add rax,0x1",
        "mov rcx,rax",
        "sub rcx,0x2",
    ]
    canonical = {
        "architecture": "x86_64",
        "entry_blocks": [0],
        "blocks": [
            {"id": 0, "instructions": repeated + ["jne @B2"]},
            {"id": 1, "instructions": repeated + ["jmp @B3"]},
            {"id": 2, "instructions": repeated + ["jne @B0"]},
            {
                "id": 3,
                "instructions": ["call @U2>", "call @SELF", "ret"],
            },
        ],
        "cfg_edges": [
            {"source": 0, "target": 2, "edge_type": "conditional_true"},
            {"source": 0, "target": 1, "edge_type": "conditional_false"},
            {"source": 1, "target": 3, "edge_type": "unconditional_jump"},
            {"source": 2, "target": 0, "edge_type": "loop_backedge"},
            {"source": 2, "target": 3, "edge_type": "conditional_false"},
        ],
    }
    prefix = "literal {|#~\\n\\t\\x1b世界\\n"
    text = core.serialize_compact_graph(
        prefix,
        canonical,
        {},
        tokenizer=bundle.tokenizer,
        visible_symbols=bundle.visible_symbols,
    )
    decoded_prefix, decoded = core.decode_f2(text)
    assert decoded_prefix == prefix
    assert decoded == canonical
    assert "\nS\n" in text
    sequence_section = text.split("\nS\n", 1)[1].split("\nB2\n", 1)[0]
    assert sequence_section
    assert "@B0" not in text
    assert "@B2" not in text
    assert "@B3" not in text
    assert "@SELF" in text
    assert "U2>" in text


def test_f2_roundtrips_multifunction_and_external_markers(tmp_path: Path):
    bundle, _ = build_fixture(tmp_path)
    canonical = {
        "architecture": "x86_64",
        "entry_blocks": [0],
        "blocks": [
            {
                "id": 0,
                "instructions": [
                    "fn @SELF",
                    "call @U0",
                    "call @X0",
                    "lea rax,@X0",
                    "lea @X0,rax",
                    "lea r10,[rip+0x0] # @U0+0x45",
                    "mov rax,~scratch%23",
                    "call @SELF+0x4",
                    "rep movs BYTE PTR es:[rdi],BYTE PTR ds:[rsi]",
                    "ret",
                ],
            },
            {
                "id": 1,
                "instructions": ["fn @U0", "call @SELF", "ret"],
            },
        ],
        "cfg_edges": [],
    }
    text = core.serialize_compact_graph(
        "// @X0=\"dart:core__StringBase._interpolate\"\n",
        canonical,
        {},
        tokenizer=bundle.tokenizer,
        visible_symbols=bundle.visible_symbols,
    )
    prefix, decoded = core.decode_f2(text)
    assert prefix == "// @X0=\"dart:core__StringBase._interpolate\"\n"
    assert decoded == canonical
    assert "fn:@SELF" in text
    assert "fn:U0" in text
    assert "call:X0" in text
    assert "%23" in text
    assert "%7E" in text
    assert "%25" in text


def test_f2_profitable_text_macros_roundtrip_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, _ = build_fixture(tmp_path)
    monkeypatch.setattr(f2, "TEXT_MACRO_TRIGGER_TOKENS", 0)
    canonical = {
        "architecture": "x86_64",
        "entry_blocks": [0],
        "blocks": [
            {
                "id": block_id,
                "instructions": [
                    (
                        "mov rax,QWORD PTR "
                        f"[rbp-0x{0x100 + block_id:x}]"
                    )
                ],
            }
            for block_id in range(24)
        ],
        "cfg_edges": [],
    }
    text = core.serialize_compact_graph(
        "",
        canonical,
        {},
        tokenizer=bundle.tokenizer,
        visible_symbols=bundle.visible_symbols,
    )
    assert "\nM\n" in text
    decoded_prefix, decoded = core.decode_f2(text)
    assert decoded_prefix == ""
    assert decoded == canonical


def test_f2_visible_symbols_are_exact_single_tokens(tmp_path: Path):
    bundle, _ = build_fixture(tmp_path)
    assert bundle.visible_symbols
    for symbol in bundle.visible_symbols:
        encoded = bundle.tokenizer.encode(symbol, add_special_tokens=False)
        assert len(encoded.ids) == 1
        assert bundle.tokenizer.decode(encoded.ids) == symbol


@pytest.mark.parametrize(
    "canonical,match",
    [
        (
            {
                "architecture": "x86_64",
                "entry_blocks": [1],
                "blocks": [{"id": 1, "instructions": ["ret"]}],
                "cfg_edges": [],
            },
            "contiguous block IDs",
        ),
        (
            {
                "architecture": "x86_64",
                "entry_blocks": [0],
                "blocks": [{"id": 0, "instructions": ["mov rax,{rbx}"]}],
                "cfg_edges": [],
            },
            "stream delimiter",
        ),
    ],
)
def test_f2_ambiguous_inputs_fail_closed(tmp_path: Path, canonical, match):
    bundle, _ = build_fixture(tmp_path)
    with pytest.raises(core.PreflightError, match=match):
        core.serialize_compact_graph(
            "",
            canonical,
            {},
            tokenizer=bundle.tokenizer,
            visible_symbols=bundle.visible_symbols,
        )


def test_row_contract_hash_mismatch_fails_closed(tmp_path: Path):
    bundle, row = build_fixture(tmp_path)
    row["compact_codec_sha256"] = "0" * 64
    with pytest.raises(core.PreflightError, match="compact_codec_sha256 mismatch"):
        bundle.prepare(row)


def test_student_constant_prefix_mismatch_fails_closed(tmp_path: Path):
    bundle, row = build_fixture(tmp_path)
    row["compact_input_ids"] = row["compact_input_ids"][1:]
    with pytest.raises(core.PreflightError, match="constant prefix"):
        bundle.prepare(row)


def test_candidate_safety_masks_strings_but_rejects_real_exit():
    harmless = """
    import 'dart:io';
    String fn0() {
      // exit(0) is text, not an action.
      return 'Process.killPid(1)';
    }
    """
    assert core.candidate_safety_reasons(harmless) == []
    malicious = "import 'dart:io'; dynamic fn0() { exit(0); }"
    assert "calls process exit" in core.candidate_safety_reasons(malicious)


def valid_response() -> dict:
    return {
        "id": "response-1",
        "model": "resolved-model-revision",
        "created": 123,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "dynamic fn0() => 7;",
                    "reasoning_content": "private reasoning",
                    "refusal": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


def test_valid_completion_records_identity_usage_and_reasoning():
    result = core.validate_completion(
        valid_response(), max_prompt_tokens=12000, max_output_tokens=8192
    )
    assert result.response_id == "response-1"
    assert result.response_model == "resolved-model-revision"
    assert result.reasoning_content == "private reasoning"
    assert result.usage["total_tokens"] == 120
    assert result.code == "dynamic fn0() => 7;"


def test_embedded_thinking_is_not_compiled_as_dart():
    response = valid_response()
    response["choices"][0]["message"]["reasoning_content"] = None
    response["choices"][0]["message"]["content"] = (
        "<think>derive behavior carefully</think>\n"
        "dynamic fn0() => 7;"
    )
    result = core.validate_completion(
        response, max_prompt_tokens=12000, max_output_tokens=8192
    )
    assert result.reasoning_content == "derive behavior carefully"
    assert result.code == "dynamic fn0() => 7;"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_non_stop_completion_is_not_valid_k(finish_reason):
    response = valid_response()
    response["choices"][0]["finish_reason"] = finish_reason
    with pytest.raises(core.InvalidCompletion, match="finish_reason"):
        core.validate_completion(
            response, max_prompt_tokens=12000, max_output_tokens=8192
        )


def test_zero_token_response_is_invalid():
    response = valid_response()
    response["usage"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    with pytest.raises(core.InvalidCompletion, match="zero"):
        core.validate_completion(
            response, max_prompt_tokens=12000, max_output_tokens=8192
        )


def test_zero_completion_usage_is_invalid_even_with_nonempty_content():
    response = valid_response()
    response["usage"] = {
        "prompt_tokens": 100,
        "completion_tokens": 0,
        "total_tokens": 100,
    }
    with pytest.raises(core.InvalidCompletion, match="zero"):
        core.validate_completion(
            response, max_prompt_tokens=12000, max_output_tokens=8192
        )


def test_completion_usage_over_requested_cap_is_invalid():
    response = valid_response()
    response["usage"] = {
        "prompt_tokens": 100,
        "completion_tokens": 8193,
        "total_tokens": 8293,
    }
    with pytest.raises(core.InvalidCompletion, match="completion tokens"):
        core.validate_completion(
            response, max_prompt_tokens=12000, max_output_tokens=8192
        )


def test_prompt_count_has_explicit_overhead_reserve(tmp_path: Path):
    bundle, _ = build_fixture(tmp_path)
    count = core.count_prompt_tokens(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        bundle.tokenizer,
        chat_overhead_reserve=256,
    )
    assert count["estimated_prompt_tokens"] == (
        count["system_tokens"] + count["user_tokens"] + 256
    )


def test_budget_reserves_worst_case_across_concurrency():
    budget = core.TokenBudget(100)
    assert budget.reserve(60) is True
    assert budget.reserve(60) is False
    budget.settle(60, 40)
    assert budget.reserve(60) is True
    budget.settle(60, 60)
    assert budget.snapshot() == {"limit": 100, "spent": 100, "reserved": 0}


def test_gdb_symbol_parser_keeps_anonymous_closure():
    output = """
    All functions matching regular expression "^fn0":
    0x0000000000010000  fn0()
    0x0000000000010100  fn0.<anonymous closure>()
    0x0000000000010200  unrelated()
    """
    assert core._gdb_function_names(output) == [
        "fn0",
        "fn0.<anonymous closure>",
    ]


def write_fake_evaluator(tmp_path: Path, source: str) -> tuple[Path, Path]:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(source, encoding="utf-8")
    dart = tmp_path / "dart"
    dart.write_bytes(b"pinned-test-dart")
    return evaluator, dart


def test_evaluator_contract_rejects_legacy_returncode_only_module(tmp_path: Path):
    evaluator, dart = write_fake_evaluator(
        tmp_path,
        "def evaluate_candidate(*args, **kwargs):\n"
        "    return True\n",
    )
    with pytest.raises(core.PreflightError, match="missing required function"):
        runner.import_evaluator(
            evaluator,
            sha(evaluator),
            dart_binary=dart,
            expected_dart_hash=sha(dart),
            validate_dart=False,
        )


def test_evaluator_contract_persists_hash_entrypoint_and_attestation(tmp_path: Path):
    evaluator, dart = write_fake_evaluator(
        tmp_path,
        "COMPLETION_ATTESTATION_ID = "
        "'per-run-256-bit-marker-exactly-once-v1'\n"
        "def evaluate_dart_jit_tests_detail(*args, **kwargs):\n"
        "    return True, True, '', 'source'\n"
        "def prepare_dart_test_completion_attestation(source):\n"
        "    return True, '', source, 'marker'\n"
        "def dart_test_completion_observed(stdout, marker):\n"
        "    return True\n",
    )
    module, record = runner.import_evaluator(
        evaluator,
        sha(evaluator),
        dart_binary=dart,
        expected_dart_hash=sha(dart),
        validate_dart=False,
    )
    assert callable(module.evaluate_dart_jit_tests_detail)
    assert record["sha256"] == sha(evaluator)
    assert record["entrypoint"] == "evaluate_dart_jit_tests_detail"
    assert record["completion_attestation_id"] == runner.REQUIRED_ATTESTATION_ID
    assert record["legacy_returncode_only_evaluator_used"] is False
    assert record["dart_binary"]["sha256"] == sha(dart)


def test_stable_evaluation_requires_attestation_on_every_pass():
    calls: list[str] = []

    def evaluator(code, tests, task_id, *, timeout, stability_runs):
        calls.append(task_id)
        assert code == "dynamic fn0() => 7;"
        assert tests == "void main() {}"
        assert timeout == 30
        assert stability_runs == 1
        return True, True, "", "dynamic fn0() => 7;\nvoid main() {}"

    result = runner.evaluate_candidate_stably(
        evaluator,
        code="dynamic fn0() => 7;",
        tests="void main() {}",
        task_id="heldout_1",
        sample_index=3,
        stability_runs=2,
        timeout=30,
    )
    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert result["compiled"] is True
    assert result["passed"] is True
    assert result["completion_attestation_enforced"] is True
    assert result["completion_attestation_satisfied_all_runs"] is True
    assert all(
        run["completion_attestation_satisfied"]
        and run["completion_attestation_id"] == runner.REQUIRED_ATTESTATION_ID
        for run in result["stability_runs"]
    )


@pytest.mark.parametrize(
    "evaluator",
    [
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda *args, **kwargs: (False, False, "dart_not_found", ""),
    ],
)
def test_evaluator_infrastructure_failure_aborts_run(evaluator):
    with pytest.raises(runner.RunFailure):
        runner.evaluate_candidate_stably(
            evaluator,
            code="dynamic fn0() => 7;",
            tests="void main() {}",
            task_id="heldout_1",
            sample_index=0,
            stability_runs=1,
            timeout=30,
        )
