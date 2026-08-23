import copy
import hashlib
from pathlib import Path

import pytest

from scripts.data import build_compact_qwen_v3 as base
from scripts.data import build_compact_qwen_v3_cfgtypes_v2 as codec


def graph_row():
    return {
        "task_id": "inline-cfg-unit",
        "function": "candidate",
        "graph_v2": {
            "extractor_sha256": codec.ROUTE_SPECS[
                codec.ROUTE_CURRENT
            ].combined_sha256
        },
        "cfg": [
            {
                "id": 0,
                "start_address": "0x10",
                "instructions": ["cmp eax,0x0", "je 0x30 <candidate+0x20>"],
            },
            {
                "id": 1,
                "start_address": "0x20",
                "instructions": ["call 0x30 <candidate+0x20>"],
            },
            {"id": 2, "start_address": "0x30", "instructions": ["ret"]},
        ],
        "edges": [
            {"source": 0, "target": 2, "edge_type": "conditional_true"},
            {"source": 0, "target": 1, "edge_type": "conditional_false"},
            {"source": 1, "target": 2, "edge_type": "call"},
            {"source": 1, "target": 2, "edge_type": "linear_fallthrough"},
        ],
        "integrity": {"entry_blocks": [0]},
        "binary_pool_uses": [],
    }


def instruction_code(canonical):
    instructions = [
        instruction
        for block in canonical["blocks"]
        for instruction in block["instructions"]
    ]
    return instructions, {
        instruction: index for index, instruction in enumerate(instructions)
    }


def test_inline_cfg_roundtrip_preserves_explicit_call_and_branch_targets():
    canonical = codec.canonicalize(graph_row())
    expansions, code = instruction_code(canonical)
    text = codec.encode(canonical, code)

    assert text.startswith("<G2C3><AX64><DX1><ENTRY><B0><BLOCKS>")
    assert "<B0><I0><I1><CT><B2><CF><B1>" in text
    assert "<B1><I2><CC><B2><CN><B2>" in text
    assert "<CT><B0><B2>" not in text
    assert "<CC><B1><B2>" not in text
    assert text.count("<CC>") == 1
    assert text.count("<CFG>") == 1

    decoded = codec.decode(text, expansions)
    assert decoded == {key: canonical[key] for key in decoded}
    assert decoded["cfg_edges"] == canonical["cfg_edges"]


def test_inline_cfg_is_shorter_than_edge_triples_without_changing_pool():
    canonical = codec.canonicalize(graph_row())
    expansions, code = instruction_code(canonical)
    compact = codec.encode(canonical, code)
    original = base.encode(canonical, code)

    compact_graph, compact_pool = compact.split(codec.POOL_START, 1)
    original_graph, original_pool = original.split(base.POOL_START, 1)
    assert compact_pool == original_pool
    assert len(list(base.TAG_RE.finditer(compact_graph))) < len(
        list(base.TAG_RE.finditer(original_graph))
    )


@pytest.mark.parametrize("edge_type", ["conditional_false", "linear_fallthrough"])
def test_inline_cfg_rejects_non_next_implicit_target(edge_type):
    canonical = codec.canonicalize(graph_row())
    edge = next(item for item in canonical["cfg_edges"] if item["edge_type"] == edge_type)
    edge["target"] = 0
    with pytest.raises(ValueError, match="target_is_not_next_block"):
        codec.encode(canonical, {})


def test_inline_cfg_rejects_non_grouped_sources():
    canonical = codec.canonicalize(graph_row())
    canonical["cfg_edges"] = [
        canonical["cfg_edges"][2],
        *canonical["cfg_edges"][:2],
        canonical["cfg_edges"][3],
    ]
    with pytest.raises(ValueError, match="not_source_grouped"):
        codec.encode(canonical, {})


def test_raw_instruction_fallback_roundtrips_with_inline_edges():
    canonical = codec.canonicalize(graph_row())
    text = codec.encode(canonical, {})
    decoded = codec.decode(text, [])
    assert decoded == {key: canonical[key] for key in decoded}
    assert text.count("<R>") == sum(
        len(block["instructions"]) for block in canonical["blocks"]
    )


def test_inline_cfg_contract_is_versioned_and_binds_base_pool_codec():
    values = {name: hashlib.sha256(name.encode()).hexdigest() for name in (
        "codec_sha256",
        "codebook_sha256",
        "tokenizer_json_sha256",
        "pool_extractor_sha256",
        "aot_manifest_sha256",
        "dart_toolchain_manifest_sha256",
        "pool_reconciliation_manifest_sha256",
    )}
    contract = codec.codec_contract(**values)
    assert contract["cfg_encoding"] == codec.CFG_ENCODING
    assert contract["cfg_inline_encoding"]["call_edges"].startswith("explicit_<CC>")
    assert contract["base_pool_codec_sha256"] == hashlib.sha256(
        Path(base.__file__).read_bytes()
    ).hexdigest()


def test_original_v3_codec_remains_edge_triple_compatible():
    canonical = base.canonicalize(copy.deepcopy(graph_row()))
    expansions, code = instruction_code(canonical)
    text = base.encode(canonical, code)
    assert "<CT><B0><B2><CF><B0><B1>" in text
    assert "<CC><B1><B2><CN><B1><B2>" in text
    decoded = base.decode(text, expansions)
    assert decoded == {key: canonical[key] for key in decoded}
