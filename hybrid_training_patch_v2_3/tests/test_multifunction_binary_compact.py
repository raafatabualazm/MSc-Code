from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer

from hybrid_training_patch_v2_3.scripts.preprocessing import (
    build_multifunction_binary_compact as builder,
)
from scripts.data import extract_dart_aot_user_function_bundle as extractor
from scripts.data import build_dart_user_symbol_attestation as attestation_builder
from scripts.data.test_extract_dart_aot_user_function_bundle import (
    DISASSEMBLIES,
    INFO_FUNCTIONS,
)


ROOT = Path(__file__).resolve().parents[2]
CODEC_PATH = ROOT / "scripts/data/build_compact_qwen_v1.py"
INLINE_CODEC_PATH = ROOT / "scripts/data/build_multifunction_compact_v2.py"
CODEBOOK_PATH = (
    ROOT
    / "scrubbed_master_v2_release/direct_compact_split_v1"
    / "compact_qwen_confirmatory_v1/codebook.json"
)
TOKENIZER_PATH = ROOT / "temp/qwen3_8b_tokenizer.remote.json"
F2_PATH = ROOT / "frontier_ceiling_patch_v1/frontier_f2.py"
CONTRACT_PATH = (
    ROOT / "pod_sync_20260723/artifacts/compact_fn0_rebuild/fn0_contract.json"
)
ATTESTATION_KEY = bytes(range(32))
ATTESTATION_FILE_SHA256 = "4" * 64


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_only_contract() -> dict[str, str]:
    return {
        "contract": extractor.SCAFFOLD_CONTRACT,
        "build_input_schema": extractor.SOURCE_ONLY_BUILD_INPUT_SCHEMA,
        "aot_row_schema": extractor.SOURCE_ONLY_AOT_ROW_SCHEMA,
        "analysis_program_sha256": "1" * 64,
        "function_source_sha256": "2" * 64,
        "producer_script_sha256": "3" * 64,
    }


def attested_symbols(
    task_id: str,
    *,
    function_symbols: list[str] | None = None,
    type_symbols: list[str] | None = None,
) -> extractor.AttestedSymbols:
    contract = source_only_contract()
    function_symbols = function_symbols or [
        "candidate",
        "helper",
    ]
    type_symbols = type_symbols or ["SecretType"]
    salt = attestation_builder.row_salt(
        ATTESTATION_KEY,
        task_id=task_id,
        analysis_program_sha256=contract["analysis_program_sha256"],
    )
    function_entries = [
        {
            "alias": f"AF{index}",
            "digest": attestation_builder.symbol_digest(
                ATTESTATION_KEY,
                task_id=task_id,
                salt_hex=salt,
                kind="function",
                index=index,
                symbol=symbol,
            ),
        }
        for index, symbol in enumerate(function_symbols)
    ]
    type_entries = [
        {
            "alias": f"T{index}",
            "digest": attestation_builder.symbol_digest(
                ATTESTATION_KEY,
                task_id=task_id,
                salt_hex=salt,
                kind="type",
                index=index,
                symbol=symbol,
            ),
        }
        for index, symbol in enumerate(type_symbols)
    ]
    row = {
        "schema": attestation_builder.SCHEMA,
        "task_id": task_id,
        "analysis_program_sha256": contract["analysis_program_sha256"],
        "function_source_sha256": contract["function_source_sha256"],
        "producer_script_sha256": contract["producer_script_sha256"],
        "key_id_sha256": attestation_builder.key_id_sha256(ATTESTATION_KEY),
        "salt_hex": salt,
        "function_symbols": function_entries,
        "type_symbols": type_entries,
        "completeness": {
            "complete_source_symbols_projection": True,
            "source_symbols_bound_to_transform_metadata": True,
            "only_dart_scheme_imports": True,
            "ordered_function_count": len(function_entries),
            "ordered_type_count": len(type_entries),
            "ordered_commitment": attestation_builder.ordered_commitment(
                ATTESTATION_KEY,
                task_id=task_id,
                salt_hex=salt,
                function_digests=[
                    entry["digest"] for entry in function_entries
                ],
                type_digests=[entry["digest"] for entry in type_entries],
            ),
        },
    }
    return extractor.AttestedSymbols(row, ATTESTATION_KEY)


def representative_bundle(
    task_id: str = "sigless_test",
    *,
    external_annotation: str = "dart:core__StringBase._interpolate",
) -> dict:
    private_file, symbols = extractor.select_same_file_symbols(INFO_FUNCTIONS)
    disassemblies = DISASSEMBLIES.replace(
        "dart:core__StringBase._interpolate",
        external_annotation,
    )
    parsed = extractor.parse_combined_gdb_disassemblies(
        disassemblies, symbols
    )
    return extractor.build_user_function_bundle(
        task_id=task_id,
        root_symbol="candidate",
        private_file_identity=private_file,
        selected_symbols=symbols,
        parsed_by_symbol=parsed,
        info_output_sha256="a" * 64,
        aot_sha256="b" * 64,
        aot_size_bytes=1234,
        source_only_contract=source_only_contract(),
        symbol_attestation=attested_symbols(task_id),
        symbol_attestation_file_sha256=ATTESTATION_FILE_SHA256,
        trusted_runtime_symbols=(
            {"dart:core__StringBase._interpolate"}
            if external_annotation == "dart:core__StringBase._interpolate"
            else set()
        ),
        split="train",
        split_row=17,
    )


def reseal_model_projection(bundle: dict) -> None:
    bundle["model_projection_sha256"] = extractor.canonical_sha256(
        extractor.model_projection(bundle)
    )


def test_combiner_offsets_all_functions_and_preserves_alias_domains() -> None:
    bundle = representative_bundle()
    canonical, projection = builder.combine_user_function_bundle(
        bundle, extractor
    )

    # The producer-owned exact main was disassembled/accounted, but is not a
    # user helper. F0 + closure + helper are retained.
    assert len(projection["functions"]) == 3
    assert [
        function["model_alias"] for function in projection["functions"]
    ] == ["@SELF", "@U0", "@U1"]
    assert canonical["entry_blocks"] == [0]
    assert canonical["blocks"][0]["instructions"][0] == "fn @SELF"
    helper_entries = [
        function["global_entry_blocks"][0]
        for function in projection["functions"][1:]
    ]
    assert canonical["blocks"][helper_entries[0]]["instructions"][0] == "fn @U0"
    assert canonical["blocks"][helper_entries[1]]["instructions"][0] == "fn @U1"

    instructions = [
        instruction
        for block in canonical["blocks"]
        for instruction in block["instructions"]
    ]
    assert "call @U1" in instructions
    assert "call @X0" in instructions
    assert any(instruction.startswith("jne @B") for instruction in instructions)
    assert not any("@F" in instruction or "@L+" in instruction for instruction in instructions)

    assert projection["external_symbols"] == [
        {
            "external_id": "X0",
            "symbol": "dart:core__StringBase._interpolate",
            "symbol_class": "trusted_runtime",
        }
    ]
    transfer = projection["transfer_semantics"]
    assert transfer["call_instruction_count"] == 3
    assert transfer["transfer_row_count"] == 3
    assert transfer["byte_coordinates_omitted_after_one_to_one_proof"] is True
    source_attestation = projection["source_symbol_attestation"]
    assert source_attestation["used"] is True
    assert source_attestation["is_keyed"] is True
    assert source_attestation["raw_names_serialized"] is False
    assert source_attestation["type_aliases"] == ["T0"]
    assert source_attestation["function_attestation_aliases"] == [
        "AF0",
        "AF1",
    ]
    assert (
        source_attestation["binding_sha256"]
        == extractor.canonical_sha256(source_attestation["binding"])
    )


def test_recursive_attested_function_accounting_is_accepted() -> None:
    private_file, symbols = extractor.select_same_file_symbols(INFO_FUNCTIONS)
    task_id = "sigless_recursive_constructor"
    initial = DISASSEMBLIES.replace(
        "dart:core__StringBase._interpolate", "new SecretType"
    )
    parsed = extractor.parse_combined_gdb_disassemblies(initial, symbols)
    attestation = attested_symbols(task_id)
    targets = extractor.discover_attested_direct_callees(
        parsed, attestation
    )
    recovered = extractor.parse_combined_gdb_disassemblies_by_address(
        """Dump of assembler code for function new SecretType:
   0x0000000000001310 <+0>:\te9 eb 00 00 00\tjmp 0x1400 <stub AllocateObject>
End of assembler dump.
""",
        targets,
    )
    parsed.update(recovered)
    selected = [*symbols, *recovered]
    bundle = extractor.build_user_function_bundle(
        task_id=task_id,
        root_symbol="candidate",
        private_file_identity=private_file,
        selected_symbols=selected,
        parsed_by_symbol=parsed,
        info_output_sha256="a" * 64,
        aot_sha256="b" * 64,
        aot_size_bytes=1234,
        source_only_contract=source_only_contract(),
        symbol_attestation=attestation,
        symbol_attestation_file_sha256=ATTESTATION_FILE_SHA256,
        gdb_file_symbols=symbols,
        split="train",
        split_row=17,
    )
    accounting = bundle["accounting"]
    assert accounting["selected_function_count"] == 5
    assert accounting["gdb_file_function_count"] == 4
    assert accounting["attested_recursive_function_count"] == 1
    canonical, projection = builder.combine_user_function_bundle(
        bundle, extractor
    )
    assert canonical["blocks"]
    assert len(projection["functions"]) == accounting["user_function_count"]


def test_selected_function_accounting_mismatch_fails_closed() -> None:
    bundle = representative_bundle()
    bundle["accounting"]["selected_function_count"] += 1
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="accounting equality",
    ):
        builder.combine_user_function_bundle(bundle, extractor)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_text_read", True),
        ("raw_source_names_serialized", True),
        ("raw_source_paths_serialized", True),
        ("source_symbol_attestation_used", False),
        ("source_symbol_attestation_is_keyed", False),
    ],
)
def test_attestation_truth_field_drift_fails_closed(
    field: str, value: bool
) -> None:
    bundle = representative_bundle()
    bundle[field] = value
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="truth fields|keyed source-symbol attestation",
    ):
        builder.combine_user_function_bundle(bundle, extractor)


def test_attestation_binding_cannot_smuggle_a_raw_name() -> None:
    bundle = representative_bundle()
    bundle["symbol_attestation_binding"]["raw_name"] = "SecretType"
    reseal_model_projection(bundle)
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="binding shape is not exact/name-free",
    ):
        builder.combine_user_function_bundle(bundle, extractor)


def test_attestation_alias_inventory_cannot_smuggle_a_raw_name() -> None:
    bundle = representative_bundle()
    bundle["type_aliases"][0]["raw_name"] = "SecretType"
    reseal_model_projection(bundle)
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="row is not name-free",
    ):
        builder.combine_user_function_bundle(bundle, extractor)


def test_attested_type_assertion_and_neutralized_non_user_are_supported() -> None:
    bundle = representative_bundle(
        "sigless_type_assert",
        external_annotation="assert type is SecretType<int>",
    )
    _, projection = builder.combine_user_function_bundle(bundle, extractor)
    preamble = builder.binary_enrichment_preamble(
        [], [], projection["external_symbols"]
    )
    assert "assert type is @T0<int>" in preamble
    assert "SecretType" not in preamble

    bundle = representative_bundle(
        "sigless_attested_non_user",
        external_annotation="new List<SecretType>",
    )
    _, projection = builder.combine_user_function_bundle(bundle, extractor)
    assert projection["external_symbols"][0]["symbol"] is None
    assert (
        projection["external_symbols"][0]["symbol_class"]
        == "neutralized_untrusted_runtime"
    )


def test_external_attestation_alias_must_exist_in_private_binding() -> None:
    bundle = representative_bundle()
    bundle["external_symbols"][0] = {
        "external_id": "X0",
        "symbol": "assert type is @T999",
        "symbol_class": "trusted_runtime",
    }
    reseal_model_projection(bundle)
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="unattested type",
    ):
        builder.combine_user_function_bundle(bundle, extractor)


def test_attested_type_alias_outside_type_assertion_fails_closed() -> None:
    bundle = representative_bundle()
    bundle["external_symbols"][0] = {
        "external_id": "X0",
        "symbol": "runtime label @T0",
        "symbol_class": "trusted_runtime",
    }
    reseal_model_projection(bundle)
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="outside a type assertion",
    ):
        builder.combine_user_function_bundle(bundle, extractor)


def test_binary_prefix_roundtrips_exact_external_dictionary() -> None:
    bundle = representative_bundle()
    _, projection = builder.combine_user_function_bundle(bundle, extractor)
    preamble = builder.binary_enrichment_preamble(
        ["hello"], [7], projection["external_symbols"]
    )
    assert '"hello"' in preamble
    assert '// numbers ["7"]' in preamble
    assert "// externals[X=index,T=runtime,N=neutralized]:T|" in preamble
    assert builder.parse_external_dictionary_from_preamble(preamble) == projection[
        "external_symbols"
    ]

    redacted = [
        {
            "external_id": "X0",
            "symbol": None,
            "symbol_class": "neutralized_untrusted_runtime",
        }
    ]
    redacted_preamble = builder.binary_enrichment_preamble([], [], redacted)
    assert builder.parse_external_dictionary_from_preamble(
        redacted_preamble
    ) == redacted
    assert ":N|[null]" in redacted_preamble


def test_binary_prefix_ascii_escapes_tokenizer_normalization_sensitive_unicode() -> None:
    # Qwen's tokenizer decode NFC-normalizes this decomposed circumflex.  The
    # serialized binary constant must therefore carry JSON's ASCII escape,
    # which is reversible and byte-stable through tokenizer encode/decode.
    decomposed = " -> y\u0302="
    preamble = builder.binary_enrichment_preamble(
        [decomposed],
        [],
        [
            {
                "external_id": "X0",
                "symbol": "type is @T0 \u03a9",
                "symbol_class": "trusted_runtime",
            }
        ],
    )
    assert "\\u0302" in preamble
    assert "\\u03a9" in preamble
    assert decomposed not in preamble
    strings_payload = preamble.splitlines()[0].removeprefix("// strings ")
    assert json.loads(strings_payload) == [decomposed]


def test_binary_prefix_preserves_complex_numbers_and_rejects_bad_external_order() -> None:
    numeric_preamble = builder.binary_enrichment_preamble(
        [], ["Float32x4(1.0, 2.0, 3.0, 4.0)"], []
    )
    assert (
        '// numbers ["Float32x4(1.0, 2.0, 3.0, 4.0)"]'
        in numeric_preamble
    )
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="contiguous X-index",
    ):
        builder.binary_enrichment_preamble(
            [],
            [],
            [
                {
                    "external_id": "X1",
                    "symbol": "print",
                    "symbol_class": "trusted_runtime",
                }
            ],
        )


def test_binary_prefix_rejects_malformed_external_class_stream() -> None:
    malformed = (
        "// externals[X=index,T=runtime,N=neutralized]:Q|[\"print\"]\n"
    )
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="invalid shape",
    ):
        builder.parse_external_dictionary_from_preamble(malformed)


def test_combined_graph_roundtrips_inline_cfg_and_f2() -> None:
    if not TOKENIZER_PATH.is_file():
        pytest.skip("pinned local Qwen tokenizer artifact is unavailable")
    bundle = representative_bundle()
    canonical, projection = builder.combine_user_function_bundle(
        bundle, extractor
    )
    preamble = builder.binary_enrichment_preamble(
        ["hello"], [7], projection["external_symbols"]
    )

    codec = import_module(
        INLINE_CODEC_PATH, "test_multifunction_inline_cfg_codec"
    )
    f2 = import_module(F2_PATH, "test_multifunction_f2")
    codebook = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    code = {
        instruction: index
        for index, instruction in enumerate(codebook["expansions"])
    }
    graph_text = codec.encode(canonical, code)
    assert codec.decode(graph_text, codebook["expansions"]) == canonical
    graph_ids = codec.compact_ids(
        graph_text, tokenizer, codebook["source_atom_ids"]
    )
    assert graph_ids

    text = f2.serialize_f2(preamble, canonical, tokenizer=tokenizer)
    decoded_prefix, decoded = f2.decode_f2(text)
    assert decoded_prefix == preamble
    assert decoded == canonical
    assert "fn:@SELF" in text
    assert "fn:U0" in text
    assert "call:X0" in text


def test_train_only_codebook_refit_preserves_stable_ids() -> None:
    class TinyTokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [1 + (ord(value) % 89) for value in text]

    tokenizer = TinyTokenizer()
    old = ["old train", "old unused", "old remains"]
    atom_ids = {f"<I{index}>": 100 + index for index in range(3)}
    source_expansions = {
        str(atom_ids[f"<I{index}>"]): tokenizer.encode(instruction)
        for index, instruction in enumerate(old)
    }
    parent_codebook = {
        "schema": builder.CODEBOOK_SCHEMA,
        "codebook_size": 3,
        "expansions": old,
        "source_atom_ids": atom_ids,
        "source_token_expansions": source_expansions,
        "base_vocab_size": 100,
    }
    parent_contract = {
        "base_vocab_size": 100,
        "source_token_ids": list(atom_ids.values()),
        "source_token_expansions": source_expansions,
    }
    graph = {
        "blocks": [
            {
                "id": 0,
                "instructions": ["old train", "new twice", "new twice"],
            }
        ]
    }
    refit, stats = builder.build_train_only_stable_codebook(
        parent_codebook=parent_codebook,
        parent_contract=parent_contract,
        train_canonicals=[graph],
        train_task_ids=["train-only"],
        tokenizer=tokenizer,
        tokenizer_sha256="a" * 64,
        parent_codebook_sha256="b" * 64,
        parent_contract_sha256="c" * 64,
        inline_cfg_codec_sha256="d" * 64,
        function_bundles_sha256="e" * 64,
        builder_script_sha256="f" * 64,
    )
    assert refit["expansions"] == [
        "old train",
        "new twice",
        "old remains",
    ]
    assert refit["source_atom_ids"] == atom_ids
    assert refit["heldout_rows_used_for_fit"] == 0
    assert stats["changed_instruction_slots"] == 1
    assert (
        refit["source_token_expansions"]["101"]
        == tokenizer.encode("new twice")
    )
    assert refit["source_token_expansions"]["100"] == source_expansions["100"]
    assert refit["source_token_expansions"]["102"] == source_expansions["102"]


def test_combiner_rejects_any_user_exclusion() -> None:
    bundle = representative_bundle()
    bundle["accounting"]["excluded_user_instruction_count"] = 1
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="zero-exclusion",
    ):
        builder.combine_user_function_bundle(bundle, extractor)


def test_transfer_table_must_match_every_call() -> None:
    bundle = representative_bundle()
    bundle["interfunction_transfers"] = bundle[
        "interfunction_transfers"
    ][1:]
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="call instructions have no transfer row",
    ):
        builder.prove_transfer_table_redundant(bundle)


def test_local_target_must_be_a_real_block_entry() -> None:
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="not a block entry",
    ):
        builder._rewrite_instruction(
            "jne @L+0x99",
            function_count=1,
            local_start_to_global_block={0: 0},
            external_count=0,
        )


def test_sanitized_train_seal_rejects_executable_reward_subset(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"task_id":"t"}\n', encoding="utf-8")
    seal_path = tmp_path / "train.seal.json"
    seal_path.write_text(
        json.dumps(
            {
                "schema": builder.SPLIT_SEAL_SCHEMA,
                "selected_role": "fit",
                "sanitation_schema": builder.SANITATION_SCHEMA,
                "sanitizer_sha256": "b" * 64,
                "evaluator_sha256": "c" * 64,
                "quarantine_sha256": "d" * 64,
                "completion_attestation_id": "attestation-v1",
                "dart_version": "Dart test",
                "stability_runs": 2,
                "training_objective_scope": "executable_reward_only",
                "output_sha256": file_sha(dataset_path),
                "contract_sha256": "a" * 64,
                "rows": 1,
                "executable_reward_eligible_rows": 1,
                "execution_ineligible_task_ids": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        builder.MultiFunctionBuildError,
        match="all-1580 sequence-imitation view",
    ):
        builder.validate_sanitized_base_seal(
            dataset_path=dataset_path,
            seal_path=seal_path,
            dataset_record={"sha256": file_sha(dataset_path)},
            contract_sha256="a" * 64,
            role="fit",
            expected_rows=1,
        )


def test_end_to_end_build_emits_role_seals_and_loadable_f2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not TOKENIZER_PATH.is_file():
        pytest.skip("pinned local Qwen tokenizer artifact is unavailable")
    monkeypatch.setattr(builder, "EXPECTED_TRAIN_ROWS", 1)
    monkeypatch.setattr(builder, "EXPECTED_DEV_ROWS", 1)

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    base_rows = []
    bundles = []
    constants = []
    for task_id in ("train_task", "dev_task"):
        base_rows.append(
            {
                "task_id": task_id,
                "compact_input_ids": [1],
                "dart_source": "int fn0() => 1;\n",
                "tests": "void main() {}\n",
                "compact_codec_sha256": contract["codec_sha256"],
                "compact_codebook_sha256": contract["codebook_sha256"],
                "compact_tokenizer_sha256": contract[
                    "tokenizer_json_sha256"
                ],
            }
        )
        bundle = representative_bundle(task_id)
        bundle["producer"] = {
            "script_sha256": file_sha(
                ROOT / "scripts/data/extract_dart_aot_user_function_bundle.py"
            ),
            "gdb": "/usr/bin/gdb",
        }
        bundles.append(bundle)
        constants.append(
            {
                "schema": "dart-aot-attested-pool-constants-v1",
                "task_id": task_id,
                "strings": ["hello"],
                "numbers": ["7"],
                "err": None,
                "noff": 1,
                "pool_offsets_sha256": "e" * 64,
                "accounting": {
                    "supported_string_objects": 1,
                    "supported_number_objects": 1,
                    "inline_float32_entries": 0,
                    "inline_float64_entries": 0,
                    "inline_float32x4_entries": 0,
                    "tagged_sentinel_entries": 0,
                    "metadata_strings_rejected": 0,
                    "unsupported_or_immediate_entries": 0,
                    "unreadable_entries": 0,
                },
            }
        )

    train_path = tmp_path / "base_train.jsonl"
    dev_path = tmp_path / "base_dev.jsonl"
    train_seal_path = tmp_path / "base_train.seal.json"
    dev_seal_path = tmp_path / "base_dev.seal.json"
    bundles_path = tmp_path / "bundles.jsonl"
    constants_path = tmp_path / "constants.jsonl"
    train_path.write_text(json.dumps(base_rows[0]) + "\n", encoding="utf-8")
    dev_path.write_text(json.dumps(base_rows[1]) + "\n", encoding="utf-8")
    train_seal_path.write_text(
        json.dumps(
            {
                "schema": "compact-public-private-join-seal-v1",
                "selected_role": "fit",
                "sanitation_schema": builder.SANITATION_SCHEMA,
                "sanitizer_sha256": "b" * 64,
                "evaluator_sha256": "c" * 64,
                "quarantine_sha256": "d" * 64,
                "completion_attestation_id": "attestation-v1",
                "dart_version": "Dart test",
                "stability_runs": 2,
                "training_objective_scope": "sequence_imitation_all_train",
                "output_sha256": file_sha(train_path),
                "contract_sha256": file_sha(CONTRACT_PATH),
                "rows": 1,
                "executable_reward_eligible_rows": 1,
                "execution_ineligible_task_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dev_seal_path.write_text(
        json.dumps(
            {
                "schema": "compact-public-private-join-seal-v1",
                "selected_role": "measure",
                "sanitation_schema": builder.SANITATION_SCHEMA,
                "sanitizer_sha256": "b" * 64,
                "evaluator_sha256": "c" * 64,
                "quarantine_sha256": "d" * 64,
                "completion_attestation_id": "attestation-v1",
                "dart_version": "Dart test",
                "stability_runs": 2,
                "output_sha256": file_sha(dev_path),
                "contract_sha256": file_sha(CONTRACT_PATH),
                "rows": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bundles_path.write_text(
        "".join(
            json.dumps(value, sort_keys=True) + "\n" for value in bundles
        ),
        encoding="utf-8",
    )
    constants_path.write_text(
        "".join(json.dumps(value) + "\n" for value in constants),
        encoding="utf-8",
    )
    extractor_path = (
        ROOT / "scripts/data/extract_dart_aot_user_function_bundle.py"
    )
    output_dir = tmp_path / "output"
    args = SimpleNamespace(
        base_train=train_path,
        expected_base_train_sha256=file_sha(train_path),
        base_train_seal=train_seal_path,
        expected_base_train_seal_sha256=file_sha(train_seal_path),
        base_dev=dev_path,
        expected_base_dev_sha256=file_sha(dev_path),
        base_dev_seal=dev_seal_path,
        expected_base_dev_seal_sha256=file_sha(dev_seal_path),
        function_bundles=bundles_path,
        expected_function_bundles_sha256=file_sha(bundles_path),
        constants=constants_path,
        expected_constants_sha256=file_sha(constants_path),
        extractor_script=extractor_path,
        expected_extractor_script_sha256=file_sha(extractor_path),
        contract=CONTRACT_PATH,
        expected_contract_sha256=file_sha(CONTRACT_PATH),
        codebook=CODEBOOK_PATH,
        expected_codebook_sha256=file_sha(CODEBOOK_PATH),
        tokenizer_json=TOKENIZER_PATH,
        expected_tokenizer_sha256=file_sha(TOKENIZER_PATH),
        codec=CODEC_PATH,
        expected_codec_sha256=file_sha(CODEC_PATH),
        inline_cfg_codec=INLINE_CODEC_PATH,
        expected_inline_cfg_codec_sha256=file_sha(INLINE_CODEC_PATH),
        frontier_f2=F2_PATH,
        expected_frontier_f2_sha256=file_sha(F2_PATH),
        output_dir=output_dir,
        expected_train_rows=1,
        expected_dev_rows=1,
        student_token_limit=builder.STUDENT_TOKEN_LIMIT,
        api_prompt_token_limit=builder.API_PROMPT_TOKEN_LIMIT,
        chat_overhead_reserve=builder.CHAT_OVERHEAD_RESERVE,
    )
    report = builder.build(args)
    assert report["passed"] is True
    assert report["counts"]["train_rows"] == 1
    assert report["counts"]["dev_rows"] == 1

    train_output = output_dir / "train_multifunction_binary.jsonl"
    dev_output = output_dir / "dev_multifunction_binary.jsonl"
    train_seal = json.loads(
        (output_dir / "train_multifunction_binary.seal.json").read_text(
            encoding="utf-8"
        )
    )
    dev_seal = json.loads(
        (output_dir / "dev_multifunction_binary.seal.json").read_text(
            encoding="utf-8"
        )
    )
    assert train_seal["schema"] == "compact-public-private-join-seal-v1"
    assert train_seal["selected_role"] == "fit"
    assert (
        train_seal["training_objective_scope"]
        == "sequence_imitation_all_train"
    )
    assert train_seal["executable_reward_eligible_rows"] == 1
    assert train_seal["execution_ineligible_task_ids"] == []
    assert train_seal["output_sha256"] == file_sha(train_output)
    assert dev_seal["selected_role"] == "measure"
    assert dev_seal["heldout_measure_only"] is True
    assert dev_seal["output_sha256"] == file_sha(dev_output)
    train_row = json.loads(
        train_output.read_text(encoding="utf-8").strip()
    )
    public_binding = train_row[
        "binary_source_symbol_attestation_binding"
    ]
    assert train_row["binary_source_symbol_attestation_used"] is True
    assert train_row["binary_source_symbol_attestation_is_keyed"] is True
    assert public_binding["raw_names_present"] is False
    assert "SecretType" not in json.dumps(train_row, sort_keys=True)
    assert train_seal["source_symbol_attestation_used"] is True
    assert train_seal["source_symbol_attestation_is_keyed"] is True
    assert (
        train_seal["source_symbol_attestation_file_sha256"]
        == ATTESTATION_FILE_SHA256
    )

    report = json.loads(
        (output_dir / "build_report.json").read_text(encoding="utf-8")
    )
    assert report["source_symbol_attestation"]["used"] is True
    assert report["source_symbol_attestation"]["is_keyed"] is True
    assert (
        report["invariants"][
            "keyed_private_source_symbol_attestation_used_to_build_representation"
        ]
        is True
    )
    assert (
        report["invariants"][
            "raw_source_or_user_names_serialized_in_model_inputs"
        ]
        is False
    )
    assert (
        "source_contents_or_user_names_used_to_build_input"
        not in report["invariants"]
    )

    teacher_artifact = import_module(
        ROOT
        / "hybrid_training_patch_v2_3/scripts/training"
        / "qwen_direct_compact_teacher_artifact.py",
        "test_teacher_artifact_loader",
    )
    train_f2 = output_dir / "train_multifunction_binary_f2.jsonl"
    manifest = train_f2.with_suffix(train_f2.suffix + ".manifest.json")
    f2_row = json.loads(train_f2.read_text(encoding="utf-8").strip())
    assert f2_row["source_symbol_attestation_used"] is True
    assert f2_row["source_symbol_attestation_is_keyed"] is True
    assert "SecretType" not in f2_row["text"]
    assert public_binding["key_id_sha256"] not in f2_row["text"]
    assert "symbol_attestation_binding" not in f2_row["text"]
    prompts, prompt_record = teacher_artifact.load_verified_prompt_rows(
        train_f2,
        expected_sha256=file_sha(train_f2),
        expected_rows=1,
    )
    system_prompt, _, loaded_manifest = (
        teacher_artifact.load_f2_prompt_contract(
            manifest,
            expected_sha256=file_sha(manifest),
            prompt_record=prompt_record,
            expected_rows=1,
            student_tokenizer_sha256=file_sha(TOKENIZER_PATH),
        )
    )
    assert prompts[0].task_id == "train_task"
    assert system_prompt
    assert loaded_manifest["f2_prompt_contract"][
        "all_rows_within_limit"
    ] is True
    assert loaded_manifest["source_symbol_attestation"]["used"] is True
    assert (
        loaded_manifest["invariants"][
            "keyed_private_source_symbol_attestation_used"
        ]
        is True
    )
