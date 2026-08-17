
"""
Hierarchical graph encoder components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
from torch_geometric.nn import GINEConv


class LocalBlockEncoder(nn.Module):
    def __init__(self, model_name='microsoft/graphcodebert-base'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        old_embeddings = self.encoder.embeddings.token_type_embeddings

        if old_embeddings.num_embeddings < 2:
            new_embeddings = nn.Embedding(
                2,
                old_embeddings.embedding_dim,
            )

            with torch.no_grad():
                new_embeddings.weight[0] = old_embeddings.weight[0]
                new_embeddings.weight[1] = old_embeddings.weight[0]

            self.encoder.embeddings.token_type_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        if token_type_ids is not None and token_type_ids.dim() == 1:
            token_type_ids = token_type_ids.unsqueeze(0)

        if input_ids.numel() == 0 or input_ids.shape[-1] == 0:
            batch = input_ids.shape[0] if input_ids.dim() > 0 else 1
            return torch.zeros((batch, self.encoder.config.hidden_size), device=input_ids.device)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        return outputs.last_hidden_state[:, 0, :]


class GraphPoolingEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_edge_types=8):
        super().__init__()

        self.edge_embedding = nn.Embedding(num_edge_types, hidden_size)

        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )

        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, block_embeddings, edge_index=None, edge_attr=None):
        if edge_index is not None and edge_attr is not None:
            edge_embeddings = self.edge_embedding(edge_attr.to(block_embeddings.device))
            edge_index = edge_index.to(block_embeddings.device)

            block_embeddings = self.conv(
                block_embeddings,
                edge_index,
                edge_embeddings,
            )

        if block_embeddings.dim() == 2:
            block_embeddings = block_embeddings.unsqueeze(0)

        attended, _ = self.attention(
            block_embeddings,
            block_embeddings,
            block_embeddings,
        )

        pooled = attended.mean(dim=1)

        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)

        return self.projection(pooled)
