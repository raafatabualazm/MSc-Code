import importlib.util
import sys
from pathlib import Path

import pytest


CODEC_PATH = Path(__file__).with_name("build_compact_qwen_v3.py")
SPEC = importlib.util.spec_from_file_location("compact_qwen_v3_codec_test", CODEC_PATH)
CODEC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CODEC
SPEC.loader.exec_module(CODEC)


def row_with_pool(*, pool=None, call_edge=True):
    instructions = [
        "mov rax,QWORD PTR [r15+0x787]",
        "mov rbx,QWORD PTR [r15-0x8]",
        "mov rcx,QWORD PTR [r15+0x797]",
        "mov rdx,QWORD PTR [r15+0x7a7]",
        "mov rsi,QWORD PTR [r15+0x7af]",
        "mov rdi,QWORD PTR [r15+0x7b7]",
        "mov r8,QWORD PTR [r15+0x7bf]",
        "call 0x20 <candidate>",
    ]
    edges = []
    if call_edge:
        edges.append({"source": 0, "target": 1, "edge_type": "call"})
    return {
        "task_id": "pool-unit",
        "function": "candidate",
        "graph_v2": {
            "extractor_sha256": CODEC.ROUTE_SPECS[
                CODEC.ROUTE_CURRENT
            ].combined_sha256
        },
        "cfg": [
            {"id": 0, "start_address": "0x10", "instructions": instructions},
            {"id": 1, "start_address": "0x20", "instructions": ["ret"]},
        ],
        "edges": edges,
        "integrity": {"entry_blocks": [0]},
        "binary_pool_uses": complete_pool() if pool is None else pool,
    }


def complete_pool():
    # Deliberately unsorted, with repeated use-sites and a duplicate record.
    return [
        {
            "pp_offset": 0x7B7,
            "kind": "string",
            "payload": {
                "code_units": [0x3C, 0x3E, 0xD83D, 0xDE00, 0xD800],
            },
            "use_sites": [
                {"block": 0, "instruction": 5},
                {"block": 0, "instruction": 5},
            ],
        },
        {
            "pp_offset": 0x787,
            "kind": "int",
            "payload": {"decimal": "-9223372036854775808"},
            "use_sites": [{"block": 0, "instruction": 0}],
        },
        {
            "pp_offset": -8,
            "kind": "double",
            "payload": {"bits_hex": "8000000000000000"},
            "use_sites": [{"block": 0, "instruction": 1}],
        },
        {
            "pp_offset": 0x797,
            "kind": "double",
            "payload": {"bits_hex": "7ff8000000000001"},
            "use_sites": [{"block": 0, "instruction": 2}],
        },
        {
            "pp_offset": 0x7A7,
            "kind": "null",
            "payload": {},
            "use_sites": [{"block": 0, "instruction": 3}],
        },
        {
            "pp_offset": 0x7AF,
            "kind": "bool",
            "payload": {"value": True},
            "use_sites": [{"block": 0, "instruction": 4}],
        },
        {
            "pp_offset": 0x7AF,
            "kind": "bool",
            "payload": {"value": True},
            "use_sites": [{"block": 0, "instruction": 4}],
        },
        {
            "pp_offset": 0x7BF,
            "kind": "composite",
            "payload": {
                "complete": True,
                "composite_type": "array_storage",
                "elements": [
                    {
                        "index": 0,
                        "value": {"kind": "int", "payload": {"decimal": "1"}},
                    },
                    {
                        "index": 1,
                        "value": {"kind": "int", "payload": {"decimal": "3"}},
                    },
                    {
                        "index": 1,
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
                                            "payload": {"code_units": [107, 101, 121]},
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "value": {
                                            "kind": "double",
                                            "payload": {"bits_hex": "4008000000000000"},
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
                    {
                        "index": 3,
                        "value": {"kind": "int", "payload": {"decimal": "3"}},
                    },
                    {"index": 4, "value": {"kind": "null", "payload": {}}},
                ],
                "omitted_edge_counts": {"property": 2, "internal": 1},
            },
            "use_sites": [{"block": 0, "instruction": 6}],
        },
    ]


def test_v3_roundtrip_preserves_pool_order_duplicates_and_call_edge():
    canonical = CODEC.canonicalize(row_with_pool())
    expansions = [
        instruction
        for block in canonical["blocks"]
        for instruction in block["instructions"]
    ]
    text = CODEC.encode(
        canonical, {instruction: index for index, instruction in enumerate(expansions)}
    )
    assert text.startswith("<G2C3><AX64><DX1>")
    assert "<CC><B0><B1>" in text
    assert text.endswith("<PEND><END>")
    assert text.count("<PX0>") == 1
    # Exact UTF-16 strings are JSON-escaped; literal angle brackets can never
    # become compact control markers.
    pool_text = text.split("<PX0>", 1)[1].split("<PEND>", 1)[0]
    assert "<" not in pool_text and ">" not in pool_text
    assert pool_text.startswith("[[")
    assert "code_units" not in pool_text
    assert "target_function" not in pool_text

    decoded = CODEC.decode(text, expansions)
    assert decoded == {key: canonical[key] for key in decoded}
    assert decoded["binary_pool"]["target_function"] == "candidate"
    assert decoded["binary_pool"]["uses"] == complete_pool()
    assert decoded["binary_pool"]["uses"][0]["use_sites"] == [
        {"block": 0, "instruction": 5},
        {"block": 0, "instruction": 5},
    ]
    assert decoded["binary_pool"]["uses"][-2] == decoded["binary_pool"]["uses"][-3]
    composite = decoded["binary_pool"]["uses"][-1]["payload"]
    assert composite["composite_type"] == "array_storage"
    assert [item["index"] for item in composite["elements"]] == [0, 1, 1, 3, 4]
    assert composite["elements"][2]["value"]["payload"]["composite_type"] == "map_storage"
    assert composite["omitted_edge_counts"] == {"internal": 1, "property": 2}
    nested_map = composite["elements"][2]["value"]["payload"]
    assert nested_map["elements"][2]["value"] == {
        "kind": "nonliteral",
        "payload": {
            "nonliteral_kind": "type_metadata",
            "profile_type": "TypeArguments",
        },
    }
    assert nested_map["elements"][3]["value"]["payload"] == {
        "nonliteral_kind": "runtime_object",
        "profile_type": "Instance",
    }


def test_empty_literal_pool_is_explicit_and_roundtrips():
    canonical = CODEC.canonicalize(row_with_pool(pool=[]))
    text = CODEC.encode(canonical, {})
    decoded = CODEC.decode(text, [])
    assert decoded["binary_pool"] == {
        "schema": CODEC.POOL_SCHEMA,
        "target_function": "candidate",
        "uses": [],
    }


def test_exact_pilot_array_storage_projection_is_lossless():
    record = {
        "pp_offset": 0x78F,
        "kind": "composite",
        "payload": {
            "complete": True,
            "composite_type": "array_storage",
            "elements": [
                {
                    "index": index,
                    "value": (
                        {"kind": "null", "payload": {}}
                        if value is None
                        else {"kind": "int", "payload": {"decimal": str(value)}}
                    ),
                }
                for index, value in enumerate([1, 3, 3, 3, None])
            ],
            "omitted_edge_counts": {"internal": 2},
        },
        "use_sites": [{"block": 0, "instruction": 0}],
    }
    assert CODEC.canonicalize_pool_uses([record]) == [record]


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda uses: uses[0].pop("payload"), "binary_pool_record_0_keys"),
        (lambda uses: uses[0].update(extra=1), "binary_pool_record_0_keys"),
        (lambda uses: uses[0].update(pp_offset=True), "pp_offset_0_must_be_integer"),
        (lambda uses: uses[0].update(kind="array"), "unsupported_pool_literal_kind"),
        (
            lambda uses: uses[0].update(payload={"code_units": [0x10000]}),
            "string_code_unit_out_of_range",
        ),
        (
            lambda uses: uses[1].update(payload={"decimal": "01"}),
            "int_decimal_not_canonical",
        ),
        (
            lambda uses: uses[2].update(payload={"bits_hex": "800000000000000A"}),
            "double_bits_not_canonical",
        ),
        (lambda uses: uses[4].update(payload={"value": None}), "pool_null_payload_keys"),
        (lambda uses: uses[5].update(payload={"value": 1}), "bool_value_must_be_boolean"),
        (lambda uses: uses[0].update(use_sites=[]), "use_sites_must_be_nonempty"),
        (
            lambda uses: uses[0]["use_sites"][0].update(instruction=0),
            "use_offset_not_present_at_site",
        ),
    ],
)
def test_malformed_pool_input_fails_closed(mutate, error):
    uses = complete_pool()
    mutate(uses)
    with pytest.raises(ValueError, match=error):
        CODEC.canonicalize(row_with_pool(pool=uses))


@pytest.mark.parametrize("bad_kind", ["reference", "truncated", "unresolved", "code"])
def test_unsupported_nested_composite_nodes_fail_closed(bad_kind):
    uses = complete_pool()
    uses[-1]["payload"]["elements"][0]["value"] = {
        "kind": bad_kind,
        "payload": {},
    }
    with pytest.raises(ValueError, match="unsupported_pool_literal_kind"):
        CODEC.canonicalize(row_with_pool(pool=uses))


def test_top_level_nonliteral_record_remains_forbidden():
    record = {
        "pp_offset": 1,
        "kind": "nonliteral",
        "payload": {
            "nonliteral_kind": "runtime_object",
            "profile_type": "Instance",
        },
        "use_sites": [{"block": 0, "instruction": 0}],
    }
    with pytest.raises(ValueError, match="top_level_nonliteral_pool_record_not_supported"):
        CODEC.canonicalize_pool_uses([record])


@pytest.mark.parametrize(
    "payload,error",
    [
        (
            {
                "nonliteral_kind": "type_metadata",
                "profile_type": "TypeArguments",
                "name": "List<int>",
            },
            "pool_nonliteral_payload_keys",
        ),
        (
            {"nonliteral_kind": "callable", "profile_type": "TypeArguments"},
            "unsupported_pool_nonliteral_descriptor_pair",
        ),
        (
            {"nonliteral_kind": "runtime_object", "profile_type": "UnknownClass"},
            "unsupported_pool_nonliteral_descriptor_pair",
        ),
        (
            {"nonliteral_kind": "runtime_object", "profile_type": "0x1234"},
            "unsupported_pool_nonliteral_descriptor_pair",
        ),
    ],
)
def test_nested_nonliteral_descriptor_is_strict_source_blind_allowlist(payload, error):
    uses = complete_pool()
    nested_map = uses[-1]["payload"]["elements"][2]["value"]["payload"]
    nested_map["elements"][2]["value"]["payload"] = payload
    with pytest.raises(ValueError, match=error):
        CODEC.canonicalize(row_with_pool(pool=uses))


@pytest.mark.parametrize(
    "mutate,error",
    [
        (
            lambda payload: payload.update(complete=False),
            "pool_composite_must_be_complete",
        ),
        (
            lambda payload: payload.update(composite_type="set_storage"),
            "unsupported_pool_composite_type",
        ),
        (
            lambda payload: payload["elements"][0].update(index=-1),
            "pool_composite_element_index_negative",
        ),
        (
            lambda payload: payload["elements"][0].update(extra=True),
            "pool_composite_element_0_keys",
        ),
        (
            lambda payload: payload.update(omitted_edge_counts={"weak": -1}),
            "pool_composite_omitted_edge_count_negative",
        ),
        (
            lambda payload: payload.update(omitted_edge_counts={"weak": True}),
            "pool_composite_omitted_edge_count_weak_must_be_integer",
        ),
    ],
)
def test_malformed_composite_input_fails_closed(mutate, error):
    uses = complete_pool()
    mutate(uses[-1]["payload"])
    with pytest.raises(ValueError, match=error):
        CODEC.canonicalize(row_with_pool(pool=uses))


def test_cyclic_composite_fails_closed():
    payload = {
        "complete": True,
        "composite_type": "array_storage",
        "elements": [],
        "omitted_edge_counts": {},
    }
    payload["elements"].append(
        {"index": 0, "value": {"kind": "composite", "payload": payload}}
    )
    record = {
        "pp_offset": 1,
        "kind": "composite",
        "payload": payload,
        "use_sites": [{"block": 0, "instruction": 0}],
    }
    with pytest.raises(ValueError, match="pool_composite_cycle_detected"):
        CODEC.canonicalize_pool_uses([record])


def test_composite_depth_limit_is_fail_closed_not_truncating():
    node = {"kind": "null", "payload": {}}
    for _ in range(CODEC.MAX_COMPOSITE_DEPTH + 1):
        node = {
            "kind": "composite",
            "payload": {
                "complete": True,
                "composite_type": "array_storage",
                "elements": [{"index": 0, "value": node}],
                "omitted_edge_counts": {},
            },
        }
    record = {
        "pp_offset": 1,
        "kind": node["kind"],
        "payload": node["payload"],
        "use_sites": [{"block": 0, "instruction": 0}],
    }
    with pytest.raises(ValueError, match="pool_composite_depth_limit_exceeded"):
        CODEC.canonicalize_pool_uses([record])


def test_noncanonical_or_injected_pool_stream_fails_closed():
    canonical = CODEC.canonicalize(row_with_pool())
    expansions = [
        instruction
        for block in canonical["blocks"]
        for instruction in block["instructions"]
    ]
    code = {instruction: index for index, instruction in enumerate(expansions)}
    text = CODEC.encode(canonical, code)
    with pytest.raises(ValueError, match="json_not_canonical"):
        CODEC.decode(text.replace('<PX0>[', '<PX0>[ ', 1), expansions)
    with pytest.raises(ValueError, match="pool_marker_count_mismatch"):
        CODEC.decode(text.replace('<PX0>[', '<PX0><PX0>[', 1), expansions)


def test_use_site_requires_exact_signed_fixed_r15_displacement():
    assert CODEC.fixed_r15_offsets("mov rax,[r15+0x10]") == [16]
    assert CODEC.fixed_r15_offsets("mov rax,[r15-8]") == [-8]
    assert CODEC.fixed_r15_offsets("mov rax,[r15]") == [0]
    assert CODEC.fixed_r15_offsets("mov rax,[r15+rcx*8+0xf]") == []


def test_non_graph_or_fabricated_pool_use_site_cannot_enter_stream():
    uses = complete_pool()
    uses[0]["use_sites"] = [{"block": 9, "instruction": 0}]
    with pytest.raises(ValueError, match="binary_pool_use_block_out_of_range"):
        CODEC.canonicalize(row_with_pool(pool=uses))


def test_v3_contract_binds_pool_aot_toolchain_and_v2_graph_codec():
    digest = "a" * 64
    contract = CODEC.codec_contract(
        codec_sha256=digest,
        codebook_sha256=digest,
        tokenizer_json_sha256=digest,
        pool_extractor_sha256=digest,
        aot_manifest_sha256=digest,
        dart_toolchain_manifest_sha256=digest,
        pool_reconciliation_manifest_sha256=digest,
    )
    assert contract["schema"] == "direct-compact-causal-v3"
    assert contract["target_function"] == "candidate"
    assert contract["pool_order_and_duplicates_preserved"] is True
    assert len(contract["graph_codec_dependency_sha256"]) == 64
    assert contract["double_representation"].startswith("exact-ieee754")
    assert contract["composite_representation"]["types"] == [
        "array_storage",
        "map_storage",
    ]
    assert contract["composite_representation"]["omitted_edge_counts_preserved"] is True
    assert contract["pool_scope"] == "canonical-graph-retained-fixed-r15-uses-v1"
    assert contract["pool_encoding"] == "canonical-positional-json-delta-v2"
    positional = contract["pool_positional_encoding"]
    assert positional["kind_to_tag"]["string"] == 0
    assert positional["kind_to_tag"]["nonliteral"] == 6
    assert positional["composite_type_to_tag"] == {
        "array_storage": 0,
        "map_storage": 1,
    }
    assert positional["record_pp_offsets"].startswith("signed-delta")
    assert contract["raw_disassembly_unreachable_islands_in_lossless_domain"] is False
    assert contract["graph_retained_literal_use_omission_policy"] == (
        "reject-via-hash-bound-private-reconciliation"
    )
    assert contract["pool_reconciliation_manifest_sha256"] == digest
    nested = contract["nested_nonliteral_descriptors"]
    assert nested["top_level_records"] == "reject"
    assert nested["profile_type_to_nonliteral_kind"]["Instance"] == "runtime_object"
    assert nested["profile_type_to_nonliteral_kind"]["Record"] == "runtime_object"
    assert nested["profile_type_to_nonliteral_kind"]["TypeArguments"] == "type_metadata"
    assert nested["names_symbols_addresses_offsets_cids_and_hashes"] == "unrepresentable"


def test_pool_envelope_target_and_schema_tampering_fail_closed():
    canonical = CODEC.canonicalize(row_with_pool(pool=[]))
    canonical["binary_pool"]["target_function"] = "other"
    with pytest.raises(ValueError, match="target_function_must_be_candidate"):
        CODEC.encode(canonical, {})
    canonical["binary_pool"]["target_function"] = "candidate"
    canonical["binary_pool"]["schema"] = "unknown"
    with pytest.raises(ValueError, match="schema_mismatch"):
        CODEC.encode(canonical, {})
