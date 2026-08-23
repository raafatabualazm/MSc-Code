import importlib.util
import sys
from pathlib import Path


CODEC_PATH = Path(__file__).with_name("build_compact_qwen_v2.py")
CODEC_SPEC = importlib.util.spec_from_file_location("compact_qwen_v2_codec_test", CODEC_PATH)
CODEC = importlib.util.module_from_spec(CODEC_SPEC)
sys.modules[CODEC_SPEC.name] = CODEC
CODEC_SPEC.loader.exec_module(CODEC)

PREPARER_PATH = CODEC_PATH.parents[2] / "scrubbed_master_v2_release" / "prepare_phase0_compact_qwen_v2.py"
PREPARER_SPEC = importlib.util.spec_from_file_location(
    "compact_qwen_v2_preparer_test", PREPARER_PATH
)
PREPARER = importlib.util.module_from_spec(PREPARER_SPEC)
sys.modules[PREPARER_SPEC.name] = PREPARER
PREPARER_SPEC.loader.exec_module(PREPARER)


def current_row(*, opcode="dec rdx", include_call=True):
    edges = [
        {
            "source": 0,
            "target": 1,
            "edge_type": "dataflow",
            "locations": ["rdx"],
            "dependency_count": 1,
        }
    ]
    if include_call:
        edges.insert(0, {"source": 0, "target": 1, "edge_type": "call"})
    return {
        "task_id": "fresh-eval-unit",
        "function": "candidate",
        "graph_v2": {"extractor_sha256": CODEC.ROUTE_SPECS[CODEC.ROUTE_CURRENT].combined_sha256},
        "cfg": [
            {"id": 0, "start_address": "0x10", "instructions": [opcode, "call 0x20 <candidate>"]},
            {"id": 1, "start_address": "0x20", "instructions": ["ret"]},
        ],
        "edges": edges,
        "integrity": {"entry_blocks": [0]},
    }


def test_current_call_edge_and_extractor_route_roundtrip():
    canonical = CODEC.canonicalize(current_row())
    expansions = ["dec rdx", "call @SELF", "ret"]
    text = CODEC.encode(canonical, {value: index for index, value in enumerate(expansions)})
    assert "<DX1>" in text
    assert "<CC><B0><B1>" in text
    decoded = CODEC.decode(text, expansions)
    assert decoded == {key: canonical[key] for key in decoded}
    assert canonical["dfg_edges"] == [
        {
            "source": 0,
            "target": 1,
            "edge_type": "dataflow",
            "locations": ["rdx"],
            "dependency_count": 1,
        }
    ]


def test_call_edge_cannot_be_silently_routed_through_legacy_codec():
    row = current_row()
    row["graph_v2"]["extractor_sha256"] = CODEC.ROUTE_SPECS[
        CODEC.ROUTE_LEGACY
    ].combined_sha256
    row["edges"] = [{"source": 0, "target": 1, "edge_type": "call"}]
    try:
        CODEC.canonicalize(row)
    except ValueError as error:
        assert "call_edge_not_allowed_for_legacy_route" in str(error)
    else:
        raise AssertionError("legacy route silently accepted a v2 call edge")


def test_route_atom_tampering_is_fail_closed():
    canonical = CODEC.canonicalize(current_row())
    expansions = ["dec rdx", "call @SELF", "ret"]
    text = CODEC.encode(canonical, {value: index for index, value in enumerate(expansions)})
    tampered = text.replace("<DX1>", "<DX0>", 1)
    try:
        CODEC.decode(tampered, expansions)
    except ValueError as error:
        assert "call_edge_not_allowed_for_legacy_route" in str(error)
    else:
        raise AssertionError("route-atom tampering was not detected")


def test_dec_is_valid_but_local_opcode_still_quarantines():
    assert CODEC.canonicalize(current_row())["blocks"][0]["instructions"][0] == "dec rdx"
    try:
        CODEC.canonicalize(current_row(opcode="local_0 rax,rax", include_call=False))
    except ValueError as error:
        assert "unknown_or_corrupt_mnemonic:local_0" in str(error)
    else:
        raise AssertionError("corrupt local_N mnemonic was accepted")


def test_family_metadata_policy_is_explicit_and_three_valued():
    assert PREPARER.normalize_family("sigless_unit", None) == "master"
    assert PREPARER.normalize_family("fresh-eval-unit", "base_llm") == "topup_s45"
    assert PREPARER.normalize_family("fresh-eval-unit", "topup_s44") == "topup_s45"
    assert PREPARER.normalize_family("fresh-eval-unit", "topup_s45") == "topup_s45"
    assert PREPARER.normalize_family("fresh-eval-unit", "topup_s46") == "topup_s46"
    assert {
        PREPARER.normalize_family("sigless_unit", None),
        PREPARER.normalize_family("fresh-eval-unit", "topup_s45"),
        PREPARER.normalize_family("fresh-eval-unit", "topup_s46"),
    } == {"master", "topup_s45", "topup_s46"}
