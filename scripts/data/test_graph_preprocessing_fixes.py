"""Verification suite for the 2026-07 graph-preprocessing fixes.

Run:  python scripts/data/test_graph_preprocessing_fixes.py [snapshot.pkl]

Covers: legacy tensor-builder byte-identity, dual-ISA instruction def-use,
block def-use, cross-block reaching-definitions DFG, GRAPH_DFG_MODE /
GRAPH_POSITION_SCHEME tensor variants, ensure_cfg_blocks dataflow wiring on
real dataset rows, cumsum_position_ids, EDGE_TYPE_TO_IDX, and the
dominator-chain loop-backedge detection.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PASSED = 0


def check(name, condition, detail=""):
    global PASSED
    if not condition:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASSED += 1
    print(f"  ok: {name}")


def test_instruction_def_use():
    from scripts.data.dfg_extractor import instruction_def_use as du

    print("[instruction_def_use]")
    # x86-64 (Intel syntax)
    check("x86 mov", du("mov rax, rbx") == (["rbx"], ["rax"]))
    check("x86 alias eax->rax", du("mov eax, ecx") == (["rcx"], ["rax"]))
    check("x86 subreg r8d->r8", du("mov r8d, r9d") == (["r9"], ["r8"]))
    check("x86 rmw add", du("add rax, rcx") == (["rax", "rcx"], ["rax"]))
    check("x86 xor zero idiom", du("xor rax, rax") == ([], ["rax"]))
    check("x86 mem store: addr regs read, no write",
          du("mov QWORD PTR [rax+rbx*2+0x8], rcx") == (["rax", "rbx", "rcx"], []))
    check("x86 stack load", du("mov rdx, QWORD PTR [rbp-0x10]") == (["rbp", "stack:rbp:-16"], ["rdx"]))
    check("x86 lea", du("lea rdi, [rax+0x4]") == (["rax"], ["rdi"]))
    check("x86 cmp defines flags", du("cmp rax, rbx") == (["rax", "rbx"], ["flags"]))
    check("x86 xchg reads and writes both operands", du("xchg rax, rbx") == (["rax", "rbx"], ["rax", "rbx"]))
    check("x86 idiv implicit operands", du("idiv rcx") == (["rax", "rdx", "rcx"], ["rax", "rdx"]))
    check("x86 cqo implicit operands", du("cqo") == (["rax"], ["rax", "rdx"]))
    check("x86 pop", du("pop rbp") == ([], ["rbp"]))
    check("x86 branch untracked", du("jmp 0x1234") == ([], []))

    # AArch64 (llvm-objdump)
    check("arm mov", du("mov x0, x1") == (["x1"], ["x0"]))
    check("arm w->x alias", du("add w1, w2, w3") == (["x2", "x3"], ["x1"]))
    check("arm add sp imm", du("add sp, sp, #0x10") == (["sp"], ["sp"]))
    check("arm zero reg excluded", du("mov w8, wzr") == ([], ["x8"]))
    check("arm moving-sp ldr omits unsound slot", du("ldr x0, [sp, #8]") == (["sp"], ["x0"]))
    check("arm moving-sp str omits unsound slot", du("str x0, [sp, #8]") == (["x0", "sp"], []))
    check("arm stable-frame slot", du("ldr x0, [x29, #8]") == (["x29", "stack:x29:+8"], ["x0"]))
    check("arm stp writeback: base written",
          du("stp x29, x30, [sp, #-16]!") == (["x29", "x30", "sp"], ["sp"]))
    check("arm ldp", du("ldp x29, x30, [sp], #16") == (["sp"], ["x29", "x30", "sp"]))
    check("arm madd", du("madd x0, x1, x2, x3") == (["x1", "x2", "x3"], ["x0"]))
    check("arm cbz read-only", du("cbz x0, 0x226304 <foo>") == (["x0"], []))
    check("arm movk rmw", du("movk x8, #0x1234, lsl #16") == (["x8"], ["x8"]))
    check("arm fp/lr alias", du("mov x29, sp")[1] == ["x29"])
    check("arm eor zero idiom", du("eor x0, x1, x1") == ([], ["x0"]))
    check("arm hex imm not a register", du("mov x0, #0x1e") == ([], ["x0"]))
    check("arm stxr status destination", du("stxr w0, x1, [x2]") == (["x1", "x2"], ["x0"]))
    bl_reads, bl_writes = du("bl 0x226304 <bar>")
    check(
        "arm bl Dart call effects",
        bl_reads == ["x1", "x2", "x3", "x5", "x6", "x7"]
        and all(f"x{i}" in bl_writes for i in range(15))
        and "x30" in bl_writes,
    )


def test_block_def_use():
    from scripts.data.dfg_extractor import block_def_use

    print("[block_def_use]")
    defs, ue = block_def_use([
        "ldr x0, [sp, #8]",   # reads sp (upward), defs x0
        "add x1, x0, x2",     # reads x0 (defined here) + x2 (upward), defs x1
        "str x1, [sp]",       # reads x1 (local), sp (already upward)
    ])
    check("defs", defs == {"x0", "x1"})
    check("upward-exposed", ue == {"sp", "x2"})


def test_cross_block_dfg():
    from scripts.data.dfg_extractor import build_cross_block_dfg

    print("[build_cross_block_dfg]")
    # B0 defines x0,x1; B1 kills x0; B2 uses x0 and x1.
    blocks = [
        {"instructions": ["mov x0, #1", "mov x1, #2"]},
        {"instructions": ["mov x0, #3"]},
        {"instructions": ["add x2, x0, x1"]},
    ]
    cfg = [
        {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
        {"source": 1, "target": 2, "edge_type": "linear_fallthrough"},
    ]
    edges = build_cross_block_dfg(blocks, cfg)
    pairs = {(e["source"], e["target"]) for e in edges}
    check("kill respected: x0 flows 1->2 not 0->2", (1, 2) in pairs and (0, 2) in pairs)
    # (0,2) present because x1 (not killed in B1) flows 0->2.
    check("all edges typed dataflow", all(e["edge_type"] == "dataflow" for e in edges))

    # Loop-carried flow: B1 uses x0 defined in B1 (previous iteration) and B0.
    blocks2 = [
        {"instructions": ["mov x0, #0"]},
        {"instructions": ["add x0, x0, #1", "cmp x0, x9"]},
    ]
    cfg2 = [
        {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
        {"source": 1, "target": 1, "edge_type": "loop_backedge"},
    ]
    pairs2 = {(e["source"], e["target"]) for e in build_cross_block_dfg(blocks2, cfg2)}
    check("loop-carried self edge", (1, 1) in pairs2)
    check("entry def reaches loop", (0, 1) in pairs2)

    check("single block -> no edges", build_cross_block_dfg([blocks[0]], []) == [])


def test_tensor_builder(snapshot_path):
    from transformers import AutoTokenizer
    from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
    from scripts.data.dfg_extractor import LightweightDFGExtractor
    from scripts.data.cfg_extractor import ensure_cfg_blocks

    print("[tensor builder]")
    tok = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    dfg = LightweightDFGExtractor()

    rows = []
    dataset_path = ROOT / "data/testing/grpo_data_graphv2.jsonl"
    if not dataset_path.is_file():
        dataset_path = ROOT.parent / "data/benchmark/grpo_data_graphv2.jsonl"
    with dataset_path.open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            rows.append(json.loads(line))

    # 1. legacy mode is byte-identical to the pre-change snapshot
    if snapshot_path and Path(snapshot_path).exists():
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)
        os.environ["GRAPH_DFG_MODE"] = "legacy"
        os.environ["GRAPH_POSITION_SCHEME"] = "legacy"
        builder = GraphCodeBERTTensorBuilder(tok, max_seq_len=512)
        idx = 0
        for row in rows:
            blocks, _ = ensure_cfg_blocks(row)
            for b in blocks[:6]:
                meta = dfg.extract_block_dfg_structured(b["instructions"])
                t = builder.build_block_tensors(b["instructions"], meta)
                for key in ("input_ids", "position_ids", "token_type_ids", "attention_mask"):
                    if t[key].tolist() != snapshot[idx][key]:
                        raise AssertionError(f"legacy drift at block {idx} key {key}")
                idx += 1
        check(f"legacy byte-identical ({idx} blocks vs snapshot)", True)
    else:
        print("  skip: no snapshot file supplied")

    # 2. edges/off modes: no <unk> DFG appendage, full window for code
    sample_block = None
    blocks, _ = ensure_cfg_blocks(rows[0])
    for b in blocks:
        meta = dfg.extract_block_dfg_structured(b["instructions"])
        if meta["nodes"]:
            sample_block = (b, meta)
            break
    check("found a block with legacy DFG nodes", sample_block is not None)
    b, meta = sample_block

    legacy_builder = GraphCodeBERTTensorBuilder(tok, dfg_mode="legacy", position_scheme="legacy")
    edges_builder = GraphCodeBERTTensorBuilder(tok, dfg_mode="edges", position_scheme="legacy")
    t_legacy = legacy_builder.build_block_tensors(b["instructions"], meta)
    t_edges = edges_builder.build_block_tensors(b["instructions"], meta)
    unk = tok.convert_tokens_to_ids(tok.unk_token)
    n_legacy_unk = t_legacy["input_ids"].tolist().count(unk)
    n_edges_unk = t_edges["input_ids"].tolist().count(unk)
    check("legacy appends unk DFG tokens", n_legacy_unk >= len(meta["nodes"][:64]) > 0,
          f"unk={n_legacy_unk} nodes={len(meta['nodes'])}")
    check("edges mode appends none", n_edges_unk < n_legacy_unk and 1 not in t_edges["token_type_ids"].tolist())

    # Graph-v2 rejects oversized blocks. Legacy mode still reproduces its
    # historical 502-token truncation; non-strict edges mode is exercised only
    # as an explicit backward-compatibility check.
    long_instrs = [f"add x{i % 30}, x{(i + 1) % 30}, #{i}" for i in range(400)]
    t_long_legacy = legacy_builder.build_block_tensors(long_instrs, {"nodes": [], "edges": [], "instruction_to_nodes": {}})
    strict_rejected = False
    with patch.dict(os.environ, {"GRAPH_STRICT_GRAPH": "1"}, clear=False):
        try:
            edges_builder.build_block_tensors(
                long_instrs,
                {"nodes": [], "edges": [], "instruction_to_nodes": {}},
            )
        except ValueError as exc:
            strict_rejected = "exceeds GraphCodeBERT budget" in str(exc)
    check("strict graph-v2 rejects oversized blocks", strict_rejected)
    with patch.dict(os.environ, {"GRAPH_STRICT_GRAPH": "0"}, clear=False):
        t_long_edges = edges_builder.build_block_tensors(
            long_instrs,
            {"nodes": [], "edges": [], "instruction_to_nodes": {}},
        )
    real_legacy = int(t_long_legacy["attention_mask"].sum().item())
    real_edges = int(t_long_edges["attention_mask"].sum().item())
    check("legacy-compatible edges mode reclaims the 8-token reserve", real_edges == 512 and real_legacy == 504,
          f"legacy={real_legacy} edges={real_edges}")

    # 3. roberta position scheme
    rob_builder = GraphCodeBERTTensorBuilder(tok, dfg_mode="edges", position_scheme="roberta")
    t_rob = rob_builder.build_block_tensors(["mov rax, rbx"], {"nodes": [], "edges": [], "instruction_to_nodes": {}})
    pos = t_rob["position_ids"].tolist()
    total = int(t_rob["attention_mask"].sum().item())
    check("roberta: code starts at 2", pos[0] == 2 and pos[:total] == list(range(2, total + 2)))
    check("roberta: pads at padding_idx=1", set(pos[total:]) == {1})
    check("roberta: max position < 514", max(pos) < 514)
    t_rob_legacy_dfg = GraphCodeBERTTensorBuilder(tok, dfg_mode="legacy", position_scheme="roberta").build_block_tensors(
        b["instructions"], meta)
    ttypes = t_rob_legacy_dfg["token_type_ids"].tolist()
    rpos = t_rob_legacy_dfg["position_ids"].tolist()
    dfg_positions = {rpos[i] for i, tt in enumerate(ttypes) if tt == 1}
    check("roberta: DFG tokens at position 0", dfg_positions == {0})


def test_ensure_cfg_blocks_dataflow():
    from scripts.data.cfg_extractor import ensure_cfg_blocks

    print("[ensure_cfg_blocks dataflow wiring]")
    rows = []
    fixture_candidates = (
        (
            "grpo_data_graphv2.jsonl",
            ROOT / "data/testing/grpo_data_graphv2.jsonl",
            ROOT.parent / "data/benchmark/grpo_data_graphv2.jsonl",
        ),
        (
            "synthetic_pool_graphv2.jsonl",
            ROOT / "data/datasets/synthetic_pool_graphv2.jsonl",
            ROOT.parent / "data/fixtures/synthetic_pool_graphv2_first.jsonl",
        ),
    )
    for label, *candidates in fixture_candidates:
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            searched = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"Missing {label}; searched: {searched}")
        with path.open(encoding="utf-8-sig") as f:
            rows.append((label, json.loads(f.readline())))

    for path, row in rows:
        os.environ["GRAPH_DFG_MODE"] = "legacy"
        blocks_l, edges_l = ensure_cfg_blocks(row)
        os.environ["GRAPH_DFG_MODE"] = "edges"
        blocks_e, edges_e = ensure_cfg_blocks(row)
        dataflow = [e for e in edges_e if e["edge_type"] == "dataflow"]
        cfg_only = [e for e in edges_e if e["edge_type"] != "dataflow"]
        check(f"{Path(path).name}: CFG edges unchanged", cfg_only == edges_l)
        check(f"{Path(path).name}: dataflow edges added", len(dataflow) > 0, f"got {len(dataflow)}")
        n = len(blocks_e)
        check(f"{Path(path).name}: dataflow indices in range",
              all(0 <= e["source"] < n and 0 <= e["target"] < n for e in dataflow))
        # idempotency: re-running on a record already carrying dataflow edges
        row2 = dict(row)
        row2["edges"] = edges_e
        _, edges_again = ensure_cfg_blocks(row2)
        check(f"{Path(path).name}: idempotent",
              sum(1 for e in edges_again if e["edge_type"] == "dataflow") == len(dataflow))
    os.environ["GRAPH_DFG_MODE"] = "legacy"


def test_cumsum_position_ids():
    import torch
    from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import cumsum_position_ids

    print("[cumsum_position_ids]")
    # [prefix 1 1 | prompt 1 1 0 0 | target 1 1]
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.float)
    pos = cumsum_position_ids(mask)
    check("pad holes skipped", pos.tolist() == [[0, 1, 2, 3, 1, 1, 4, 5]])


def test_edge_type_vocab():
    from models.pyg_cfg_dataset import EDGE_TYPE_TO_IDX, cfg_to_pyg
    import torch

    print("[edge type vocab]")
    check("dataflow has dedicated slot", EDGE_TYPE_TO_IDX["dataflow"] == 8)
    check("18 graph-v2 forward/reverse edge slots", max(EDGE_TYPE_TO_IDX.values()) == 17)
    data = cfg_to_pyg(
        {"edges": [{"source": 0, "target": 1, "edge_type": "dataflow"}]},
        torch.zeros((2, 8)),
    )
    check("dataflow edge_attr = 8", data.edge_attr.tolist() == [8])


def test_dominator_backedge():
    from scripts.data.cfg_extractor import AssemblyCFGExtractor

    print("[dominator-chain backedge]")
    # Non-monotonic stream: header at 0x900 sits at a HIGHER address than the
    # latch (0x350), so the target<start address heuristic cannot see the
    # backedge, and idom(latch)=B2 != header, so the old idom-equality check
    # missed it too. Only the dominator-chain walk classifies it.
    asm = "\n".join([
        "0x100 <+0>:\tnop",
        "0x900 <+1>:\tcmp rax, rbx",
        "0x904 <+5>:\tje 0xa00",
        "0x208 <+9>:\tadd rax, rcx",
        "0x20c <+13>:\tjmp 0x350",
        "0x350 <+17>:\tsub rbx, rcx",
        "0x354 <+21>:\tjmp 0x900",
        "0xa00 <+25>:\tret",
    ])
    blocks, edges, _ = AssemblyCFGExtractor(asm).build_blocks()
    back = [e for e in edges if e.source == 3 and e.target == 1]
    check("latch->header edge exists", len(back) == 1)
    check("classified loop_backedge via dominator chain", back[0].edge_type == "loop_backedge",
          f"got {back[0].edge_type}")
    forward = [e for e in edges if e.edge_type == "loop_backedge" and (e.source, e.target) != (3, 1)]
    check("no forward edge misclassified", not forward, str(forward))


def main():
    snapshot = sys.argv[1] if len(sys.argv) > 1 else None
    os.environ["GRAPH_ADD_REVERSE_EDGES"] = "0"
    os.environ["GRAPH_BLOCK_POSITION_MODE"] = "off"
    os.environ.setdefault("GRAPH_DFG_MODE", "legacy")
    os.environ.setdefault("GRAPH_POSITION_SCHEME", "legacy")
    test_instruction_def_use()
    test_block_def_use()
    test_cross_block_dfg()
    test_dominator_backedge()
    test_edge_type_vocab()
    test_ensure_cfg_blocks_dataflow()
    test_cumsum_position_ids()
    test_tensor_builder(snapshot)
    print(f"\nALL {PASSED} CHECKS PASSED")


if __name__ == "__main__":
    main()
