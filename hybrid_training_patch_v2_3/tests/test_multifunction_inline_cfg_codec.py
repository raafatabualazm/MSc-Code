from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CODEC_PATH = ROOT / "scripts/data/build_multifunction_compact_v2.py"


def load_codec():
    spec = importlib.util.spec_from_file_location(
        "test_multifunction_inline_cfg_codec", CODEC_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


codec = load_codec()


def representative_graph() -> dict:
    return {
        "architecture": "x86_64",
        "entry_blocks": [0, 2],
        "blocks": [
            {"id": 0, "instructions": ["mov rax,rbx", "jne @B2"]},
            {"id": 1, "instructions": ["call @X0", "ret"]},
            {"id": 2, "instructions": ["fn @U0", "ret"]},
        ],
        "cfg_edges": [
            {"source": 0, "target": 2, "edge_type": "conditional_true"},
            {"source": 0, "target": 1, "edge_type": "conditional_false"},
            {"source": 1, "target": 2, "edge_type": "linear_fallthrough"},
        ],
    }


def test_inline_cfg_roundtrip_and_source_elision() -> None:
    graph = representative_graph()
    expansions = ["mov rax,rbx", "jne @B2", "ret"]
    text = codec.encode(graph, {value: index for index, value in enumerate(expansions)})
    assert text.startswith("<G2C1><CFG><AX64>")
    assert text.count("<CT>") == 1
    assert text.count("<CF>") == 1
    assert text.count("<CN>") == 1
    # Three edges carry only their target blocks. Their source blocks occur
    # once as block definitions, rather than once again per edge.
    assert codec.decode(text, expansions) == graph


def test_inline_cfg_keeps_raw_fallback_exact() -> None:
    graph = representative_graph()
    text = codec.encode(graph, {})
    assert "<R>call @X0<E>" in text
    assert codec.decode(text, []) == graph


def test_inline_cfg_rejects_non_source_grouped_edges() -> None:
    graph = representative_graph()
    graph["cfg_edges"] = [
        graph["cfg_edges"][2],
        graph["cfg_edges"][0],
        graph["cfg_edges"][1],
    ]
    with pytest.raises(ValueError, match="source-block order"):
        codec.encode(graph, {})


def test_inline_cfg_rejects_raw_compact_delimiter() -> None:
    graph = representative_graph()
    graph["blocks"][0]["instructions"][0] = "mov <bad>"
    with pytest.raises(ValueError, match="compact delimiter"):
        codec.encode(graph, {})

