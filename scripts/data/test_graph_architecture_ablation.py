from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from models.hierarchical_graph_encoder_antigravity import GraphPoolingEncoder


class GraphArchitectureAblationTests(unittest.TestCase):
    def test_gnn_depth_is_configurable(self) -> None:
        with patch.dict(os.environ, {"GRAPH_GNN_LAYERS": "2"}, clear=False):
            encoder = GraphPoolingEncoder(hidden_size=8, num_edge_types=18)
        self.assertEqual(encoder.gnn_layers, 2)
        self.assertEqual(len(encoder.convs), 2)
        self.assertEqual(len(encoder.norms), 2)

    def test_global_attention_identity_retains_block_sequence(self) -> None:
        env = {
            "GRAPH_GNN_LAYERS": "2",
            "GRAPH_GNN_ABLATION": "identity",
            "GRAPH_GLOBAL_ATTENTION_ABLATION": "identity",
            "GRAPH_BLOCK_POSITION_MODE": "off",
        }
        with patch.dict(os.environ, env, clear=False):
            encoder = GraphPoolingEncoder(hidden_size=8, num_edge_types=18)
            blocks = torch.randn(3, 8)
            output, mask = encoder(blocks, list_of_B_i=[3])
            expected = encoder.projection(blocks.unsqueeze(0))

        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(mask, torch.ones((1, 3)))

    def test_invalid_gnn_depth_fails_loudly(self) -> None:
        with patch.dict(os.environ, {"GRAPH_GNN_LAYERS": "0"}, clear=False):
            with self.assertRaises(ValueError):
                GraphPoolingEncoder(hidden_size=8, num_edge_types=18)


if __name__ == "__main__":
    unittest.main()
