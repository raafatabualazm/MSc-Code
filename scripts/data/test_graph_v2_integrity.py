"""Adversarial regression tests for the graph-v2 extraction contract."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.pyg_cfg_dataset import (
    EDGE_TYPE_TO_IDX,
    build_linear_region_ids,
    cfg_to_pyg,
)
from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from models.hierarchical_graph_encoder_antigravity import (
    GraphPoolingEncoder,
    LearnedBlockQueryPool,
)
from scripts.data.cfg_extractor import AssemblyCFGExtractor, ensure_cfg_blocks
from scripts.data.dfg_extractor import build_cross_block_dfg, instruction_def_use
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import QwenGraphPrefixAdapter
from generate_synthetic_tasks_parallel import run as run_generator_command


def extract(lines: list[str]):
    with patch.dict(os.environ, {"GRAPH_MAX_BLOCK_INSTRS": "24"}, clear=False):
        return AssemblyCFGExtractor("\n".join(lines)).build_blocks()


class AssemblyCaptureTests(unittest.TestCase):
    def test_full_output_mode_preserves_disassembly_head(self) -> None:
        payload = "HEAD" + ("x" * 6000) + "TAIL"
        with tempfile.TemporaryDirectory(prefix="ag_output_test_") as tmp:
            command = [sys.executable, "-c", f"print({payload!r})"]
            code, full = run_generator_command(command, str(Path(tmp)), output_limit=0)
            self.assertEqual(code, 0)
            self.assertIn("HEAD", full)
            self.assertIn("TAIL", full)
            code, diagnostic_tail = run_generator_command(command, str(Path(tmp)))
            self.assertEqual(code, 0)
            self.assertNotIn("HEAD", diagnostic_tail)
            self.assertIn("TAIL", diagnostic_tail)


class CFGV2Tests(unittest.TestCase):
    def test_missing_networkx_fails_graph_integrity_closed(self) -> None:
        with patch("scripts.data.cfg_extractor.nx", None):
            _, _, integrity = AssemblyCFGExtractor(
                "0x100 <+0>:\tcmp rax,rbx\n"
                "0x104 <+4>:\tjne 0x100\n"
                "0x108 <+8>:\tret"
            ).build_blocks()
        self.assertFalse(integrity["networkx_available"])
        self.assertFalse(integrity["valid"])

    def test_gdb_symbol_list_is_not_an_instruction(self) -> None:
        blocks, _, integrity = extract([
            'All functions matching regular expression "foo":',
            '1: static void foo(void);',
            'Dump of assembler code for function foo:',
            '0x100 <+0>:\tpush rbp',
            '0x101 <+1>:\tret',
            'End of assembler dump.',
        ])
        instructions = [item for block in blocks for item in block.instructions]
        self.assertEqual(instructions, ["push rbp", "ret"])
        self.assertEqual(integrity["entry_address"], "0x100")

    def test_arm_internal_call_preserves_wrapper_body_and_continuation(self) -> None:
        blocks, edges, integrity = extract([
            "100:\tbl 0x120",
            "104:\tret",
            "120:\tmov x0,x1",
            "124:\tret",
        ])
        signatures = {(edge.source, edge.target, edge.edge_type) for edge in edges}
        self.assertIn((0, 1, "linear_fallthrough"), signatures)
        self.assertIn((0, 2, "call"), signatures)
        self.assertEqual(integrity["internal_direct_call_count"], 1)
        self.assertEqual(integrity["pruned_unreachable_block_count"], 0)
        self.assertTrue(integrity["valid"])

    def test_arm_symbol_slices_are_valid_multi_entry_components(self) -> None:
        extractor = AssemblyCFGExtractor(
            "080:\tmov x0,#0x0\n"
            "084:\tret\n"
            "100:\tmov x1,x2\n"
            "104:\tret\n"
            "120:\tmov x3,x4\n"
            "124:\tret",
            entry_addresses=["0x100", "0x80", "0x120"],
        )
        blocks, _, integrity = extractor.build_blocks()
        self.assertEqual(len(blocks), 3)
        self.assertEqual(integrity["entry_block"], 1)
        self.assertEqual(integrity["entry_blocks"], [1, 0, 2])
        self.assertEqual(integrity["pruned_unreachable_block_count"], 0)
        self.assertTrue(integrity["valid"])

    def test_extended_x86_condition_has_two_successors(self) -> None:
        blocks, edges, _ = extract([
            '0x100 <+0>:\tadd rax,rbx',
            '0x104 <+4>:\tjno 0x110',
            '0x108 <+8>:\tmov rcx,rax',
            '0x10c <+12>:\tret',
            '0x110 <+16>:\tret',
        ])
        outgoing = {(edge.target, edge.edge_type) for edge in edges if edge.source == 0}
        self.assertEqual(outgoing, {(1, "conditional_false"), (2, "conditional_true")})
        self.assertEqual(blocks[0].edge_types, [edge.edge_type for edge in edges if edge.source == 0])

    def test_indirect_jump_displacement_does_not_become_target(self) -> None:
        blocks, edges, integrity = extract([
            '0x100 <+0>:\tjmp QWORD PTR [rip+0x108]',
            '0x108 <+8>:\tret',
        ])
        self.assertFalse(edges)
        self.assertEqual(integrity["indirect_branch_count"], 1)
        self.assertEqual(blocks[0].successors, [])

    def test_current_instruction_marker_and_real_entry_are_preserved(self) -> None:
        blocks, edges, integrity = extract([
            '0x300 <+20>:\tret',
            '=> 0x100 <+0>:\tjmp 0x300',
        ])
        self.assertEqual(integrity["entry_block"], 0)
        self.assertTrue(any(edge.source == 0 and edge.target == 1 for edge in edges))

    def test_discarded_symbol_region_cannot_suppress_reused_addresses(self) -> None:
        blocks, _, integrity = extract([
            '0x100 <+0>:\tud2',
            '0x101 <+1>:\tint3',
            '0x100 <+0>:\tmov rax, 1',
            '0x101 <+1>:\tret',
        ])
        instructions = [item for block in blocks for item in block.instructions]
        self.assertEqual(instructions, ["mov rax, 1", "ret"])
        self.assertEqual(integrity["duplicate_address_count"], 0)

    def test_arm_objdump_bytes_and_bare_target(self) -> None:
        blocks, edges, _ = extract([
            '226304: a9bf7bfd stp x29, x30, [sp, #-16]!',
            '226308: 54000040 b.eq 226310',
            '22630c: d503201f nop',
            '226310: d65f03c0 ret',
        ])
        self.assertTrue(blocks[0].instructions[0].startswith("stp "))
        self.assertEqual({edge.edge_type for edge in edges if edge.source == 0},
                         {"conditional_true", "conditional_false"})

    def test_trap_has_no_fallthrough(self) -> None:
        blocks, edges, _ = extract([
            '0x100 <+0>:\tud2',
            '0x101 <+1>:\tret',
        ])
        self.assertEqual(blocks[0].block_type, "trap")
        self.assertFalse(any(edge.source == 0 for edge in edges))

    def test_closed_unreachable_epilogue_is_pruned_and_reported(self) -> None:
        blocks, edges, integrity = extract([
            '0x100 <+0>:\tjmp 0x104',
            '0x102 <+2>:\tret',
            '0x104 <+4>:\tret',
        ])
        self.assertTrue(integrity["valid"])
        self.assertEqual(integrity["pruned_unreachable_block_count"], 1)
        self.assertEqual(len(blocks), 2)
        self.assertEqual([(edge.source, edge.target) for edge in edges], [(0, 1)])

    def test_unresolved_branch_is_never_laundered_by_pruning(self) -> None:
        _, _, integrity = extract([
            '0x100 <+0>:\tjmp 0x999',
            '0x104 <+4>:\tret',
        ])
        self.assertFalse(integrity["valid"])
        self.assertEqual(integrity["unresolved_direct_branch_count"], 1)
        self.assertEqual(integrity["pruned_unreachable_block_count"], 0)

    def test_external_tail_jump_is_recorded_not_called_truncation(self) -> None:
        blocks, edges, integrity = extract([
            "Dump of assembler code for function foo:",
            "0x100 <+0>:\tmov rax, 1",
            "0x104 <+4>:\tjmp 0x900 <stub _iso_stub_ReturnAsyncNotFutureStub>",
            "End of assembler dump.",
        ])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(edges, [])
        self.assertEqual(integrity["external_direct_branch_count"], 1)
        self.assertEqual(integrity["unresolved_direct_branch_count"], 0)
        self.assertTrue(integrity["valid"])

    def test_missing_same_function_target_remains_invalid(self) -> None:
        _, _, integrity = extract([
            "Dump of assembler code for function foo:",
            "0x100 <+0>:\tjmp 0x900 <foo+2048>",
            "End of assembler dump.",
        ])
        self.assertEqual(integrity["external_direct_branch_count"], 0)
        self.assertEqual(integrity["unresolved_direct_branch_count"], 1)
        self.assertFalse(integrity["valid"])

    def test_strict_loader_rejects_missing_or_invalid_graphs(self) -> None:
        strict = {"GRAPH_STRICT_GRAPH": "1", "GRPO_STRICT_GRAPH": "0"}
        with patch.dict(os.environ, strict, clear=False):
            with self.assertRaises(ValueError):
                ensure_cfg_blocks({"assembly": "0x100 <+0>:\tret"}, auto_extract=False)
            with self.assertRaises(ValueError):
                ensure_cfg_blocks({
                    "cfg": [{"instructions": ["ret"]}],
                    "edges": [],
                    "integrity": {"valid": False},
                })

    def test_dfg_off_removes_pinned_dataflow_edges(self) -> None:
        record = {
            "cfg": [
                {"instructions": ["mov rax, 1"]},
                {"instructions": ["ret"]},
            ],
            "edges": [
                {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
                {"source": 0, "target": 1, "edge_type": "dataflow"},
            ],
            "integrity": {"valid": True},
        }
        with patch.dict(os.environ, {"GRAPH_DFG_MODE": "off"}, clear=False):
            _, edges = ensure_cfg_blocks(record)
        self.assertEqual([edge["edge_type"] for edge in edges], ["linear_fallthrough"])


class DFGV2Tests(unittest.TestCase):
    def test_dart_x64_call_convention(self) -> None:
        reads, writes = instruction_def_use("call 0x500 <foo>")
        self.assertEqual(reads, ["rdi", "rsi", "rdx", "rbx", "r8", "r9"])
        self.assertEqual(
            writes,
            ["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"],
        )

    def test_dart_arm64_call_convention(self) -> None:
        reads, writes = instruction_def_use("bl 0x226304 <foo>")
        self.assertEqual(reads, ["x1", "x2", "x3", "x5", "x6", "x7"])
        self.assertEqual(writes, [f"x{i}" for i in range(15)] + ["x30"])

    def test_eight_bit_setcc_and_return_are_visible(self) -> None:
        self.assertEqual(instruction_def_use("setne al"), (["flags"], ["rax"]))
        reads, writes = instruction_def_use("ret")
        self.assertEqual(writes, [])
        self.assertIn("rax", reads)
        self.assertIn("x0", reads)

    def test_call_kills_stale_return_register(self) -> None:
        blocks = [
            {"instructions": ["mov rax, 1"]},
            {"instructions": ["call 0x500 <foo>"]},
            {"instructions": ["add rbx, rax"]},
        ]
        cfg = [
            {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
            {"source": 1, "target": 2, "edge_type": "linear_fallthrough"},
        ]
        edges = build_cross_block_dfg(blocks, cfg)
        pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertIn((1, 2), pairs)
        self.assertNotIn((0, 2), pairs)

    def test_stack_store_load_produces_dependency(self) -> None:
        blocks = [
            {"instructions": ["mov QWORD PTR [rbp-0x8], rax"]},
            {"instructions": ["mov rcx, QWORD PTR [rbp-0x8]"]},
        ]
        cfg = [{"source": 0, "target": 1, "edge_type": "linear_fallthrough"}]
        edges = build_cross_block_dfg(blocks, cfg)
        self.assertEqual(len(edges), 1)
        self.assertIn("stack:rbp:-8", edges[0]["locations"])

    def test_moving_stack_pointer_offsets_are_not_false_slots(self) -> None:
        reads, writes = instruction_def_use("mov rcx, QWORD PTR [rsp+0x8]")
        self.assertIn("rsp", reads)
        self.assertFalse(any(item.startswith("stack:") for item in reads + writes))

    def test_return_value_reaches_return_block(self) -> None:
        blocks = [
            {"instructions": ["mov rax, 42"]},
            {"instructions": ["ret"]},
        ]
        cfg = [{"source": 0, "target": 1, "edge_type": "linear_fallthrough"}]
        self.assertIn((0, 1), {
            (edge["source"], edge["target"])
            for edge in build_cross_block_dfg(blocks, cfg)
        })

    def test_reverse_ordered_hundred_block_chain_converges(self) -> None:
        blocks = [{"instructions": ["nop"]} for _ in range(100)]
        blocks[99] = {"instructions": ["mov x0, #7"]}
        blocks[0] = {"instructions": ["add x1, x0, #1"]}
        cfg = [
            {"source": index, "target": index - 1, "edge_type": "linear_fallthrough"}
            for index in range(99, 0, -1)
        ]
        pairs = {(edge["source"], edge["target"]) for edge in build_cross_block_dfg(blocks, cfg)}
        self.assertIn((99, 0), pairs)


class GraphConsumerV2Tests(unittest.TestCase):
    def test_clap_tensor_builder_preserves_instruction_token_ids(self) -> None:
        class FakeAsmTokenizer:
            name_or_path = "hustcw/clap-asm"
            pad_token_id = 1
            unk_token_id = 3
            unk_token = "<unk>"

            def encode_function(self, function):
                self.function = function
                return {
                    "input_ids": [10, 11, 12],
                    "token_type_ids": [100, 100, 101],
                }

        tokenizer = FakeAsmTokenizer()
        builder = GraphCodeBERTTensorBuilder(tokenizer, max_seq_len=8, dfg_mode="edges")
        tensors = builder.build_block_tensors(["mov rax, 1", "ret"], {})
        self.assertEqual(tokenizer.function, {"0": "mov rax, 1", "1": "ret"})
        self.assertEqual(tensors["input_ids"].tolist(), [10, 11, 12, 1, 1, 1, 1, 1])
        self.assertEqual(
            tensors["token_type_ids"].tolist(),
            [100, 100, 101, 1, 1, 1, 1, 1],
        )
        self.assertEqual(tensors["attention_mask"].sum().item(), 3.0)

    def test_multi_query_pool_uses_unmasked_tokens_and_retains_four_vectors(self) -> None:
        torch.manual_seed(19)
        pool = LearnedBlockQueryPool(
            hidden_size=8, num_vectors=4, num_heads=2, dropout=0.0
        ).eval()
        token_states = torch.randn(2, 6, 8, requires_grad=True)
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ])
        pooled = pool(token_states, attention_mask)
        perturbed = token_states.detach().clone()
        perturbed[attention_mask == 0] = 10000.0
        pooled_perturbed = pool(perturbed, attention_mask)
        self.assertEqual(tuple(pooled.shape), (2, 4, 8))
        self.assertTrue(torch.allclose(pooled.detach(), pooled_perturbed, atol=1e-5))
        self.assertGreater(
            float((pooled[:, 1:] - pooled[:, :-1]).abs().max().detach()),
            1e-6,
        )
        pooled.square().mean().backward()
        self.assertGreater(float(pool.query_tokens.grad.abs().sum()), 0.0)
        self.assertGreater(float(token_states.grad.abs().sum()), 0.0)

    def test_graph_encoder_retains_multi_vector_block_states(self) -> None:
        env = {
            "GRAPH_GNN_ABLATION": "identity",
            "GRAPH_GLOBAL_ATTENTION_ABLATION": "identity",
            "GRAPH_BLOCK_POSITION_MODE": "off",
            "GRAPH_REGION_COMPRESSION": "off",
        }
        with patch.dict(os.environ, env, clear=False):
            encoder = GraphPoolingEncoder(hidden_size=8, num_edge_types=18)
            graph_states, graph_mask = encoder(
                torch.randn(5, 4, 8),
                list_of_B_i=[2, 3],
            )
        self.assertEqual(tuple(graph_states.shape), (2, 12, 8))
        self.assertEqual(graph_mask.sum(dim=1).tolist(), [8.0, 12.0])

    def test_multi_vector_pooling_does_not_inflate_dynamic_prefix_budget(self) -> None:
        env = {
            "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
            "GRAPH_QWEN_PREFIX_MIN_TOKENS": "4",
            "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": "4",
            "GRAPH_QWEN_PREFIX_GATE_MODE": "token",
            "GRAPH_BLOCK_POOLING": "multi_query",
            "GRAPH_BLOCK_VECTORS_PER_BLOCK": "4",
        }
        with patch.dict(os.environ, env, clear=False):
            adapter = QwenGraphPrefixAdapter(
                hidden_dim=8, num_prefix_tokens=64, num_heads=2, dropout=0.0
            ).eval()
        with torch.no_grad():
            _, prefix_mask = adapter(torch.randn(1, 100, 8), torch.ones(1, 100))
        # 100 graph states represent 25 blocks, so the normal scale-4 budget is 20.
        self.assertEqual(prefix_mask.sum().item(), 20.0)

    def test_linear_regions_stop_at_branches_joins_and_size_boundaries(self) -> None:
        edges = [
            {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
            {"source": 1, "target": 2, "edge_type": "linear_fallthrough"},
            {"source": 2, "target": 3, "edge_type": "conditional_true"},
            {"source": 2, "target": 4, "edge_type": "conditional_false"},
            {"source": 3, "target": 5, "edge_type": "linear_fallthrough"},
            {"source": 4, "target": 5, "edge_type": "linear_fallthrough"},
            {"source": 5, "target": 6, "edge_type": "unconditional_jump"},
            {"source": 6, "target": 7, "edge_type": "linear_fallthrough"},
            # Data flow must not split or join control regions.
            {"source": 0, "target": 7, "edge_type": "dataflow"},
        ]
        region_ids = build_linear_region_ids(8, edges, max_region_blocks=2)
        self.assertEqual(region_ids.tolist(), [0, 0, 1, 2, 3, 4, 4, 5])

    def test_pyg_retains_regions_when_message_passing_edges_are_ablated(self) -> None:
        record = {
            "edges": [
                {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
                {"source": 1, "target": 2, "edge_type": "linear_fallthrough"},
            ],
        }
        with patch.dict(
            os.environ,
            {"GRAPH_EDGE_ABLATION": "none", "GRAPH_REGION_MAX_BLOCKS": "2"},
            clear=False,
        ):
            data = cfg_to_pyg(record, torch.zeros((3, 8)))
        self.assertEqual(data.edge_index.size(1), 0)
        self.assertEqual(data.region_id.tolist(), [0, 0, 1])

        call_record = {
            "cfg": [
                {"block_type": "call"},
                {"block_type": "return"},
            ],
            "edges": [
                {"source": 0, "target": 1, "edge_type": "linear_fallthrough"},
            ],
        }
        with patch.dict(os.environ, {"GRAPH_EDGE_ABLATION": "none"}, clear=False):
            call_data = cfg_to_pyg(call_record, torch.zeros((2, 8)))
        self.assertEqual(call_data.region_id.tolist(), [0, 1])

    def test_region_context_preserves_blocks_and_receives_gradients(self) -> None:
        common = {
            "GRAPH_GNN_ABLATION": "identity",
            "GRAPH_GLOBAL_ATTENTION_ABLATION": "identity",
            "GRAPH_BLOCK_POSITION_MODE": "off",
        }
        with patch.dict(
            os.environ,
            {**common, "GRAPH_REGION_COMPRESSION": "off"},
            clear=False,
        ):
            baseline = GraphPoolingEncoder(hidden_size=8, num_edge_types=18)
        with patch.dict(
            os.environ,
            {**common, "GRAPH_REGION_COMPRESSION": "linear_residual"},
            clear=False,
        ):
            regions = GraphPoolingEncoder(hidden_size=8, num_edge_types=18)
        regions.load_state_dict(baseline.state_dict())

        block_states = torch.randn(4, 8, requires_grad=True)
        # Region IDs are local to each PyG graph and therefore restart at zero
        # after batching.
        region_ids = torch.tensor([0, 0, 0, 0])
        with patch.dict(os.environ, common, clear=False):
            baseline_states, baseline_mask = baseline(
                block_states.detach(), list_of_B_i=[2, 2]
            )
            region_states, region_mask = regions(
                block_states,
                list_of_B_i=[2, 2],
                region_ids=region_ids,
            )
            region_states.square().mean().backward()

        self.assertEqual(region_states.shape, baseline_states.shape)
        self.assertTrue(torch.equal(region_mask, baseline_mask))
        self.assertFalse(torch.allclose(region_states.detach(), baseline_states))
        gradients = {
            "region_score": regions.region_score.weight.grad,
            "region_projection": regions.region_projection.weight.grad,
            "region_gate": regions.region_gate_logit.grad,
            "block_states": block_states.grad,
        }
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_out_of_range_edge_fails_before_pyg_batching(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside this graph"):
            cfg_to_pyg(
                {"edges": [{"source": 0, "target": 5, "edge_type": "dataflow"}]},
                torch.zeros((2, 8)),
            )

    def test_dataflow_has_a_dedicated_edge_slot(self) -> None:
        self.assertEqual(EDGE_TYPE_TO_IDX["dataflow"], 8)
        self.assertNotEqual(EDGE_TYPE_TO_IDX["dataflow"], EDGE_TYPE_TO_IDX["call"])

    def test_reverse_edges_are_distinct_and_bidirectional(self) -> None:
        record = {
            "edges": [{"source": 0, "target": 1, "edge_type": "dataflow"}],
        }
        with patch.dict(
            os.environ,
            {"GRAPH_ADD_REVERSE_EDGES": "1", "GRAPH_EDGE_ABLATION": "full"},
            clear=False,
        ):
            data = cfg_to_pyg(record, torch.zeros((2, 8)))
        self.assertEqual(data.edge_index.t().tolist(), [[0, 1], [1, 0]])
        self.assertEqual(
            data.edge_attr.tolist(),
            [EDGE_TYPE_TO_IDX["dataflow"], EDGE_TYPE_TO_IDX["reverse_dataflow"]],
        )

    def test_block_positions_survive_permutation_in_prefix(self) -> None:
        torch.manual_seed(7)
        encoder = GraphPoolingEncoder(hidden_size=8, num_edge_types=18).eval()
        adapter = QwenGraphPrefixAdapter(
            hidden_dim=8, num_prefix_tokens=2, num_heads=2, dropout=0.0
        ).eval()
        blocks = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        env = {"GRAPH_GNN_ABLATION": "identity", "GRAPH_BLOCK_POSITION_MODE": "off"}
        with patch.dict(os.environ, env, clear=False), torch.no_grad():
            off_a, mask = encoder(blocks, list_of_B_i=[2])
            off_b, _ = encoder(blocks.flip(0), list_of_B_i=[2])
            prefix_off_a, _ = adapter(off_a, mask)
            prefix_off_b, _ = adapter(off_b, mask)
        self.assertTrue(torch.allclose(prefix_off_a, prefix_off_b, atol=1e-6))

        env["GRAPH_BLOCK_POSITION_MODE"] = "sinusoidal"
        with patch.dict(os.environ, env, clear=False), torch.no_grad():
            pos_a, mask = encoder(blocks, list_of_B_i=[2])
            pos_b, _ = encoder(blocks.flip(0), list_of_B_i=[2])
            prefix_pos_a, _ = adapter(pos_a, mask)
            prefix_pos_b, _ = adapter(pos_b, mask)
        self.assertFalse(torch.allclose(prefix_pos_a, prefix_pos_b, atol=1e-6))

    def test_single_node_prefix_slots_retain_identity(self) -> None:
        torch.manual_seed(7)
        with patch.dict(
            os.environ, {"GRAPH_QWEN_PREFIX_GATE_MODE": "token"}, clear=False
        ):
            adapter = QwenGraphPrefixAdapter(
                hidden_dim=16, num_prefix_tokens=4, num_heads=4, dropout=0.0
            ).eval()
        prefix, _ = adapter(torch.randn(1, 1, 16), torch.ones(1, 1))
        self.assertGreater(
            float((prefix[:, 1:] - prefix[:, :-1]).abs().max().detach()),
            1e-6,
        )
        prefix.sum().backward()
        self.assertEqual(tuple(adapter.gate_logit.shape), (4, 1))
        self.assertTrue(torch.isfinite(adapter.gate_logit.grad).all())

    def test_dynamic_prefix_scales_with_graph_size(self) -> None:
        env = {
            "GRAPH_QWEN_PREFIX_DYNAMIC": "1",
            "GRAPH_QWEN_PREFIX_MIN_TOKENS": "4",
            "GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2": "4",
            "GRAPH_QWEN_PREFIX_GATE_MODE": "token",
        }
        with patch.dict(os.environ, env, clear=False):
            adapter = QwenGraphPrefixAdapter(
                hidden_dim=8, num_prefix_tokens=64, num_heads=2, dropout=0.0
            ).eval()
        states = torch.randn(3, 458, 8)
        graph_mask = torch.zeros(3, 458)
        graph_mask[0, :1] = 1
        graph_mask[1, :25] = 1
        graph_mask[2, :458] = 1
        with torch.no_grad():
            prefix, prefix_mask = adapter(states, graph_mask)
        self.assertEqual(prefix_mask.sum(dim=1).tolist(), [4.0, 20.0, 36.0])
        self.assertTrue(torch.equal(prefix[prefix_mask == 0], torch.zeros_like(prefix[prefix_mask == 0])))

    def test_graph_glue_gradients_are_finite_and_nonzero(self) -> None:
        torch.manual_seed(11)
        encoder = GraphPoolingEncoder(hidden_size=8, num_edge_types=18)
        with patch.dict(
            os.environ,
            {
                "GRAPH_GNN_ABLATION": "full",
                "GRAPH_BLOCK_POSITION_MODE": "sinusoidal",
                "GRAPH_QWEN_PREFIX_GATE_MODE": "token",
            },
            clear=False,
        ):
            adapter = QwenGraphPrefixAdapter(
                hidden_dim=8, num_prefix_tokens=4, num_heads=2, dropout=0.0
            )
            block_states = torch.randn(3, 8, requires_grad=True)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
            edge_attr = torch.tensor([0, 9, 0, 9])
            graph_states, graph_mask = encoder(
                block_states,
                edge_index=edge_index,
                edge_attr=edge_attr,
                list_of_B_i=[3],
            )
            prefix, _ = adapter(graph_states, graph_mask)
            prefix.square().mean().backward()

        gradients = {
            "edge_embedding": encoder.edge_embedding.weight.grad,
            "gine": next(encoder.convs[0].parameters()).grad,
            "projection": encoder.projection.weight.grad,
            "block_position_scale": encoder.block_position_scale.grad,
            "prefix_queries": adapter.query_tokens.grad,
            "prefix_gate": adapter.gate_logit.grad,
            "block_states": block_states.grad,
        }
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
