from scripts.data.build_phase0_binary_pool_graphs import (
    BuildError,
    NESTED_NONLITERAL_PROFILE_KIND,
    _reconciled_value,
    composite_value,
    model_pool_uses,
)
from scripts.data.build_compact_qwen_v3 import (
    NESTED_NONLITERAL_PROFILE_KIND as CODEC_NONLITERAL_PROFILE_KIND,
)

import pytest


def test_pool_projection_accounts_pruned_xref_without_fabricating_site() -> None:
    static = {
        "entries": [
            {
                "pool_offset": "0x787",
                "category": "literal",
                "literal": {"type": "string", "code_units": [65], "value": "A"},
                "uses": [{"function_id": "candidate", "pc": "0x10"}],
            },
            {
                "pool_offset": "0x78f",
                "category": "literal",
                "literal": {"type": "int", "decimal": "7"},
                "uses": [{"function_id": "candidate", "pc": "0x20"}],
            },
        ]
    }
    runtime = {
        "entries": [
            {
                "pp_offset": 0x787,
                "category": "literal",
                "literal": {"type": "string", "code_units": [65]},
                "uses": [{"function_id": "candidate", "pc": "0x1010"}],
            },
            {
                "pp_offset": 0x78F,
                "category": "nonliteral",
                "nonliteral_kind": "untagged_or_smi",
                "uses": [{"function_id": "candidate", "pc": "0x1020"}],
            },
        ]
    }
    cfg = [
        {
            "id": 0,
            "instructions": ["mov rax,QWORD PTR [r15+0x787]"],
        }
    ]
    records, accounting = model_pool_uses(
        static,
        runtime,
        candidate_base=0x1000,
        sites={0x1010: {"block": 0, "instruction": 0}},
        cfg=cfg,
    )
    assert records == [
        {
            "pp_offset": 0x787,
            "kind": "string",
            "payload": {"code_units": [65]},
            "use_sites": [{"block": 0, "instruction": 0}],
        }
    ]
    assert accounting["target_exact_xrefs"] == 2
    assert accounting["graph_retained_xrefs"] == 1
    assert accounting["excluded_non_graph_xref_count"] == 1
    assert accounting["all_target_xrefs_accounted"] is True


def test_complete_composite_is_recursive_and_exact() -> None:
    value = composite_value(
        {
            "category": "composite",
            "complete": True,
            "composite_type": "array_storage",
            "elements": [
                {
                    "index": 0,
                    "value": {
                        "category": "literal",
                        "literal": {"type": "int", "decimal": "3"},
                    },
                },
                {
                    "index": 1,
                    "value": {
                        "category": "literal",
                        "literal": {"type": "null", "value": None},
                    },
                },
            ],
            "omitted_edge_counts": {"property": 1},
        }
    )
    assert value["kind"] == "composite"
    assert value["payload"]["elements"][0]["value"] == {
        "kind": "int",
        "payload": {"decimal": "3"},
    }
    assert value["payload"]["elements"][1]["value"] == {
        "kind": "null",
        "payload": {},
    }


def test_nested_source_blind_nonliteral_is_preserved_and_allowlists_match() -> None:
    assert NESTED_NONLITERAL_PROFILE_KIND == CODEC_NONLITERAL_PROFILE_KIND
    for profile_type in ("Instance", "Record"):
        value = composite_value(
            {
                "category": "composite",
                "complete": True,
                "composite_type": "array_storage",
                "elements": [
                    {
                        "index": 0,
                        "value": {
                            "category": "nonliteral",
                            "nonliteral_kind": "runtime_object",
                            "profile_type": profile_type,
                        },
                    }
                ],
                "omitted_edge_counts": {},
            }
        )
        assert value["payload"]["elements"][0]["value"] == {
            "kind": "nonliteral",
            "payload": {
                "nonliteral_kind": "runtime_object",
                "profile_type": profile_type,
            },
        }


@pytest.mark.parametrize(
    "node",
    [
        {
            "category": "nonliteral",
            "nonliteral_kind": "runtime_object",
            "profile_type": "UserClassName",
        },
        {
            "category": "nonliteral",
            "nonliteral_kind": "callable",
            "profile_type": "Instance",
        },
        {
            "category": "nonliteral",
            "nonliteral_kind": "runtime_object",
            "profile_type": "Instance",
            "name": "candidate",
        },
    ],
)
def test_nested_nonliteral_unknown_mismatch_or_extra_field_fails_closed(node) -> None:
    with pytest.raises(BuildError):
        composite_value(node)


def test_runtime_double_bits_are_authoritative_for_nan() -> None:
    value = _reconciled_value(
        {
            "category": "literal",
            "literal": {"type": "double", "value": "NaN"},
        },
        {
            "category": "literal",
            "literal": {"type": "double", "bits_hex": "7ff8000000000042"},
        },
    )
    assert value == {
        "kind": "double",
        "payload": {"bits_hex": "7ff8000000000042"},
    }
