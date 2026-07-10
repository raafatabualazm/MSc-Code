"""
Hierarchical graph encoder components (Antigravity version).
Preserves block sequences to prevent information bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
from torch_geometric.nn import GINEConv


import os


class LocalBlockEncoder(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        if model_name is None:
            model_name = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Locate the embeddings module dynamically
        embeddings = None
        if hasattr(self.encoder, "embeddings"):
            embeddings = self.encoder.embeddings
        elif hasattr(self.encoder, "jroformer") and hasattr(self.encoder.jroformer, "embeddings"):
            embeddings = self.encoder.jroformer.embeddings
        elif hasattr(self.encoder, "roberta") and hasattr(self.encoder.roberta, "embeddings"):
            embeddings = self.encoder.roberta.embeddings
        else:
            for name, module in self.encoder.named_modules():
                if name.endswith("embeddings") and hasattr(module, "token_type_embeddings"):
                    embeddings = module
                    break

        if embeddings is None:
            raise ValueError(f"Could not find embeddings module in model {model_name}")

        if hasattr(embeddings, "token_type_embeddings") and embeddings.token_type_embeddings is not None:
            old_embeddings = embeddings.token_type_embeddings
            if old_embeddings.num_embeddings < 2:
                new_embeddings = nn.Embedding(
                    2,
                    old_embeddings.embedding_dim,
                )

                with torch.no_grad():
                    new_embeddings.weight[0] = old_embeddings.weight[0]
                    new_embeddings.weight[1] = old_embeddings.weight[0]

                embeddings.token_type_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Resolve signature of base model if wrapped by PEFT
        base_model = self.encoder
        if hasattr(self.encoder, "get_base_model"):
            base_model = self.encoder.get_base_model()

        encoder_class_name = base_model.__class__.__name__.lower()
        needs_1d_mask = "asmencoder" in encoder_class_name or "roformer" in encoder_class_name

        if needs_1d_mask:
            pad_token_id = getattr(base_model.config, "pad_token_id", 1)
            if pad_token_id is None:
                pad_token_id = 1
            attention_mask = (input_ids != pad_token_id).to(dtype=torch.float32)
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.float)

            if attention_mask.dim() == 4 and attention_mask.size(1) == 1:
                attention_mask = attention_mask.squeeze(1)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            if attention_mask.shape[0] != input_ids.shape[0]:
                if attention_mask.shape[0] == 1:
                    attention_mask = attention_mask.expand(input_ids.shape[0], *attention_mask.shape[1:])

            # GraphCodeBERTTensorBuilder emits a 2D token-visibility mask
            # (1.0 real / 0.0 pad), which is used as-is. The collapse logic
            # below is defensive compatibility for older cached tensors that
            # still carry the historical square additive graph mask (0.0
            # visible / -10000.0 masked): reduce it to "this token
            # participates in at least one visible relation".
            if attention_mask.dim() == 3:
                if attention_mask.min() < 0:
                    attention_mask = (attention_mask >= 0).any(dim=-1)
                else:
                    attention_mask = (attention_mask > 0).any(dim=-1)
            elif attention_mask.dim() == 2 and attention_mask.min() < 0:
                attention_mask = attention_mask >= 0

            attention_mask = attention_mask.to(dtype=torch.float32)

        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.shape[0] != input_ids.shape[0]:
                if position_ids.shape[0] == 1:
                    position_ids = position_ids.expand(input_ids.shape[0], -1)

        if token_type_ids is not None:
            if token_type_ids.dim() == 1:
                token_type_ids = token_type_ids.unsqueeze(0)
            if token_type_ids.shape[0] != input_ids.shape[0]:
                if token_type_ids.shape[0] == 1:
                    token_type_ids = token_type_ids.expand(input_ids.shape[0], -1)

        if input_ids.numel() == 0 or input_ids.shape[-1] == 0:
            batch = input_ids.shape[0] if input_ids.dim() > 0 else 1
            return torch.zeros((batch, self.encoder.config.hidden_size), device=input_ids.device)

        import inspect
        sig = inspect.signature(base_model.forward)
        
        kwargs = {}
        if "input_ids" in sig.parameters:
            kwargs["input_ids"] = input_ids
        else:
            kwargs["input_ids"] = input_ids

        if "attention_mask" in sig.parameters and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        if "position_ids" in sig.parameters and position_ids is not None:
            kwargs["position_ids"] = position_ids

        if "token_type_ids" in sig.parameters and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        if isinstance(outputs, torch.Tensor):
            return outputs
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, tuple):
            return outputs[0][:, 0, :]
        else:
            raise ValueError(f"Unknown encoder output format from {self.encoder.__class__.__name__}: {type(outputs)}")



class GraphPoolingEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_edge_types=8):
        super().__init__()

        self.edge_embedding = nn.Embedding(num_edge_types, hidden_size)

        # 4-layer GNN to capture 4-hop control-flow neighborhoods
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            ) for _ in range(4)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(4)
        ])

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )

        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, block_embeddings, edge_index=None, edge_attr=None, list_of_B_i=None):
        # Keep original embeddings for global residual skip connection
        initial_embeddings = block_embeddings

        # 1. Run GNN update
        if edge_index is not None and edge_attr is not None:
            edge_embeddings = self.edge_embedding(edge_attr.to(block_embeddings.device))
            edge_index = edge_index.to(block_embeddings.device)

            for conv, norm in zip(self.convs, self.norms):
                x = conv(block_embeddings, edge_index, edge_embeddings)
                block_embeddings = norm(block_embeddings + x)

            # Global residual connection back to initial block embeddings
            block_embeddings = block_embeddings + initial_embeddings

        # 2. De-concatenate batch into padded sequence of shape (batch_size, max_blocks, hidden_size)
        if list_of_B_i is not None:
            batch_size = len(list_of_B_i)
            max_blocks = max(list_of_B_i)
            
            # Split the continuous node embeddings tensor into a list of tensors for each sample
            split_embeddings = torch.split(block_embeddings, list_of_B_i, dim=0)
            
            padded_embeddings = []
            encoder_attention_mask = []
            
            for emb in split_embeddings:
                num_blocks = emb.size(0)
                pad_len = max_blocks - num_blocks
                if pad_len > 0:
                    padding = torch.zeros((pad_len, emb.size(1)), device=emb.device, dtype=emb.dtype)
                    padded_emb = torch.cat([emb, padding], dim=0)
                    mask = torch.cat([
                        torch.ones(num_blocks, device=emb.device, dtype=torch.float),
                        torch.zeros(pad_len, device=emb.device, dtype=torch.float)
                    ], dim=0)
                else:
                    padded_emb = emb
                    mask = torch.ones(num_blocks, device=emb.device, dtype=torch.float)
                    
                padded_embeddings.append(padded_emb)
                encoder_attention_mask.append(mask)
                
            padded_embeddings = torch.stack(padded_embeddings, dim=0)
            encoder_attention_mask = torch.stack(encoder_attention_mask, dim=0)
        else:
            # Single-example case (inference)
            batch_size = 1
            max_blocks = block_embeddings.size(0)
            padded_embeddings = block_embeddings.unsqueeze(0)
            encoder_attention_mask = torch.ones((1, max_blocks), device=block_embeddings.device, dtype=torch.float)

        # 3. Multihead Self-Attention with padding mask
        # key_padding_mask expects True for positions to be masked (padded)
        key_padding_mask = (encoder_attention_mask == 0)
        
        attended, _ = self.attention(
            padded_embeddings,
            padded_embeddings,
            padded_embeddings,
            key_padding_mask=key_padding_mask,
        )

        # 4. Project and return (shape: [batch_size, max_blocks, hidden_size] and [batch_size, max_blocks])
        return self.projection(attended), encoder_attention_mask
