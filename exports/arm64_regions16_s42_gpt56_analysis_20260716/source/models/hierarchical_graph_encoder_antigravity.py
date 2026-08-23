"""
Hierarchical graph encoder components (Antigravity version).
Preserves block sequences to prevent information bottleneck.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torch_geometric.nn import GINEConv
from torch_geometric.utils import softmax as pyg_softmax


class LearnedBlockQueryPool(nn.Module):
    """Resample token-level block states into several learned role vectors."""

    def __init__(
        self,
        hidden_size: int,
        num_vectors: int = 4,
        num_heads: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        if not 2 <= num_vectors <= 8:
            raise ValueError(
                f"GRAPH_BLOCK_VECTORS_PER_BLOCK must be in [2, 8], got {num_vectors}"
            )
        heads = max(1, min(num_heads, hidden_size))
        while hidden_size % heads != 0 and heads > 1:
            heads -= 1

        self.num_vectors = num_vectors
        self.query_tokens = nn.Parameter(torch.empty(num_vectors, hidden_size))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_value_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if token_states.dim() != 3:
            raise ValueError(
                f"Expected token states [batch, tokens, hidden], got {token_states.shape}"
            )
        batch_size = token_states.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, -1)
            if attention_mask.size(1) != token_states.size(1):
                raise ValueError(
                    "Block token mask length does not match token-level encoder states"
                )
            key_padding_mask = attention_mask <= 0
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        key_values = self.key_value_norm(token_states)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            key_values,
            key_values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = queries + attended
        return pooled + self.ffn(pooled)


class LocalBlockEncoder(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        if model_name is None:
            model_name = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")
        encoder_revision = os.environ.get("GRAPH_ENCODER_REVISION", "").strip() or None
        if "clap-asm" in model_name.lower():
            # CLAP's remote AsmEncoder predates Transformers 5 and does not
            # declare the loader bookkeeping added in that release. Resolve
            # its dynamic class once and add the empty declaration before the
            # real checkpoint load. The checkpoint's word/token-type embedding
            # aliases remain tied after loading.
            encoder_config = AutoConfig.from_pretrained(
                model_name,
                revision=encoder_revision,
                trust_remote_code=True,
            )
            class_probe = AutoModel.from_config(
                encoder_config,
                trust_remote_code=True,
            )
            encoder_class = class_probe.__class__
            if not hasattr(encoder_class, "all_tied_weights_keys"):
                encoder_class.all_tied_weights_keys = {}
            del class_probe
            self.encoder = encoder_class.from_pretrained(
                model_name,
                revision=encoder_revision,
                config=encoder_config,
            )
        else:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                revision=encoder_revision,
                trust_remote_code=True,
            )

        self.block_pooling = os.environ.get(
            "GRAPH_BLOCK_POOLING", "cls"
        ).strip().lower()
        if self.block_pooling not in {"cls", "multi_query"}:
            raise ValueError(
                f"Unknown GRAPH_BLOCK_POOLING={self.block_pooling!r}; "
                "use cls or multi_query"
            )
        self.block_vectors_per_block = int(
            os.environ.get("GRAPH_BLOCK_VECTORS_PER_BLOCK", "4")
        )
        self.multi_vector_pool = LearnedBlockQueryPool(
            hidden_size=self.encoder.config.hidden_size,
            num_vectors=self.block_vectors_per_block,
            num_heads=8,
            dropout=0.05,
        )

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
            output_shape = (batch, self.encoder.config.hidden_size)
            if self.block_pooling == "multi_query":
                output_shape = (
                    batch,
                    self.block_vectors_per_block,
                    self.encoder.config.hidden_size,
                )
            return torch.zeros(output_shape, device=input_ids.device)

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
            encoded = outputs
        elif hasattr(outputs, "last_hidden_state"):
            encoded = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            encoded = outputs[0]
        else:
            raise ValueError(
                f"Unknown encoder output format from "
                f"{self.encoder.__class__.__name__}: {type(outputs)}"
            )

        if encoded.dim() == 2:
            if self.block_pooling == "multi_query":
                raise ValueError(
                    "multi_query block pooling requires token-level encoder states, "
                    f"but {base_model.__class__.__name__} returned one vector per block"
                )
            return encoded
        if encoded.dim() != 3:
            raise ValueError(
                f"Expected pooled or token-level encoder output, got {encoded.shape}"
            )
        if self.block_pooling == "multi_query":
            return self.multi_vector_pool(encoded, attention_mask)
        return encoded[:, 0, :]



class GraphPoolingEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_edge_types=None):
        super().__init__()

        if num_edge_types is None:
            num_edge_types = 18 if os.environ.get("GRAPH_ADD_REVERSE_EDGES", "0") == "1" else 9

        self.edge_embedding = nn.Embedding(num_edge_types, hidden_size)
        self.block_position_scale = nn.Parameter(torch.tensor(0.1))

        self.gnn_layers = int(os.environ.get("GRAPH_GNN_LAYERS", "4"))
        if not 1 <= self.gnn_layers <= 8:
            raise ValueError(
                f"GRAPH_GNN_LAYERS must be in [1, 8], got {self.gnn_layers}"
            )

        # Four layers preserve historical checkpoints. New controlled runs can
        # select fewer layers without changing any other graph component.
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            ) for _ in range(self.gnn_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(self.gnn_layers)
        ])

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )

        self.region_compression = os.environ.get(
            "GRAPH_REGION_COMPRESSION", "off"
        ).strip().lower()
        if self.region_compression not in {"off", "linear_residual"}:
            raise ValueError(
                "Unknown GRAPH_REGION_COMPRESSION="
                f"{self.region_compression!r}; use off or linear_residual"
            )

        # These parameters are always present so newly trained on/off controls
        # are parameter-matched. In linear_residual mode, bounded straight-line
        # regions are attention-compressed and fused back into every member
        # block. The block sequence itself is retained for the prefix adapter.
        self.region_score = nn.Linear(hidden_size, 1, bias=False)
        self.region_projection = nn.Linear(hidden_size, hidden_size)
        self.region_norm = nn.LayerNorm(hidden_size)
        self.region_gate_logit = nn.Parameter(torch.logit(torch.tensor(0.2)))

        self.projection = nn.Linear(hidden_size, hidden_size)

    def _inject_region_context(
        self,
        block_embeddings: torch.Tensor,
        region_ids: torch.Tensor | None,
        list_of_B_i: list[int] | None,
    ) -> torch.Tensor:
        if self.region_compression == "off":
            return block_embeddings
        if region_ids is None:
            raise ValueError(
                "linear_residual region compression requires per-block region_ids"
            )

        counts = list_of_B_i or [block_embeddings.size(0)]
        if sum(counts) != block_embeddings.size(0):
            raise ValueError(
                f"Region split sizes {counts} do not match "
                f"{block_embeddings.size(0)} block embeddings"
            )
        region_ids = region_ids.reshape(-1).to(
            device=block_embeddings.device, dtype=torch.long
        )
        if region_ids.numel() != block_embeddings.size(0):
            raise ValueError(
                f"Expected {block_embeddings.size(0)} region IDs, "
                f"received {region_ids.numel()}"
            )

        split_embeddings = torch.split(block_embeddings, counts, dim=0)
        split_region_ids = torch.split(region_ids, counts, dim=0)
        fused_graphs = []
        gate = torch.sigmoid(self.region_gate_logit).to(block_embeddings.dtype)
        for embeddings, graph_regions in zip(split_embeddings, split_region_ids):
            if graph_regions.numel() == 0 or torch.any(graph_regions < 0):
                raise ValueError("Region IDs must be non-empty and non-negative")
            unique_regions = torch.unique(graph_regions, sorted=True)
            expected_regions = torch.arange(
                int(unique_regions[-1].item()) + 1,
                device=graph_regions.device,
                dtype=graph_regions.dtype,
            )
            if not torch.equal(unique_regions, expected_regions):
                raise ValueError(
                    "Each graph must use contiguous local region IDs beginning at zero"
                )

            num_regions = expected_regions.numel()
            scores = self.region_score(embeddings).squeeze(-1)
            weights = pyg_softmax(scores, graph_regions, num_nodes=num_regions)
            weighted = embeddings * weights.unsqueeze(-1)
            pooled = embeddings.new_zeros((num_regions, embeddings.size(-1)))
            pooled = pooled.index_add(0, graph_regions, weighted)
            region_context = self.region_norm(self.region_projection(pooled))
            fused_graphs.append(embeddings + gate * region_context[graph_regions])

        return torch.cat(fused_graphs, dim=0)

    def forward(
        self,
        block_embeddings,
        edge_index=None,
        edge_attr=None,
        list_of_B_i=None,
        region_ids=None,
    ):
        block_detail_states = None
        vectors_per_block = 1
        if block_embeddings.dim() == 3:
            # Multi-query local pooling retains K role vectors per block. Their
            # mean is the topology-aligned node state consumed by GINE; graph
            # and region updates are broadcast back into all K detail vectors.
            block_detail_states = block_embeddings
            vectors_per_block = block_embeddings.size(1)
            block_embeddings = block_embeddings.mean(dim=1)
        elif block_embeddings.dim() != 2:
            raise ValueError(
                "GraphPoolingEncoder expects [blocks, hidden] or "
                f"[blocks, vectors, hidden], got {block_embeddings.shape}"
            )

        # Keep original node embeddings for graph and region residuals.
        initial_embeddings = block_embeddings
        gnn_ablation = os.environ.get("GRAPH_GNN_ABLATION", "full").strip().lower()
        if gnn_ablation not in {"full", "identity"}:
            raise ValueError(
                f"Unknown GRAPH_GNN_ABLATION={gnn_ablation!r}; use full or identity"
            )

        # 1. Run GNN update
        if gnn_ablation == "full" and edge_index is not None and edge_attr is not None:
            edge_embeddings = self.edge_embedding(edge_attr.to(block_embeddings.device))
            edge_index = edge_index.to(block_embeddings.device)

            for conv, norm in zip(self.convs, self.norms):
                x = conv(block_embeddings, edge_index, edge_embeddings)
                block_embeddings = norm(block_embeddings + x)

            # Global residual connection back to initial block embeddings
            block_embeddings = block_embeddings + initial_embeddings

        # 2. Compress each bounded straight-line region to a learned summary,
        # then broadcast that context back to its member blocks. This creates a
        # block -> region -> function hierarchy without dropping block states or
        # changing the dynamic graph-prefix token schedule.
        block_embeddings = self._inject_region_context(
            block_embeddings,
            region_ids,
            list_of_B_i,
        )

        sequence_counts = list_of_B_i
        if block_detail_states is not None:
            contextual_delta = (block_embeddings - initial_embeddings).unsqueeze(1)
            block_embeddings = (
                block_detail_states + contextual_delta
            ).reshape(-1, block_detail_states.size(-1))
            if list_of_B_i is not None:
                sequence_counts = [count * vectors_per_block for count in list_of_B_i]

        # 3. De-concatenate batch into padded sequence of shape (batch_size, max_blocks, hidden_size)
        if sequence_counts is not None:
            batch_size = len(sequence_counts)
            max_blocks = max(sequence_counts)
            
            # Split the continuous node embeddings tensor into a list of tensors for each sample
            split_embeddings = torch.split(block_embeddings, sequence_counts, dim=0)
            
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

        position_mode = os.environ.get("GRAPH_BLOCK_POSITION_MODE", "off").strip().lower()
        if position_mode == "sinusoidal":
            positions = torch.arange(
                max_blocks, device=padded_embeddings.device, dtype=torch.float32
            ).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(
                    0,
                    padded_embeddings.size(-1),
                    2,
                    device=padded_embeddings.device,
                    dtype=torch.float32,
                )
                * (-math.log(10000.0) / padded_embeddings.size(-1))
            )
            positional = torch.zeros(
                (max_blocks, padded_embeddings.size(-1)),
                device=padded_embeddings.device,
                dtype=torch.float32,
            )
            positional[:, 0::2] = torch.sin(positions * frequencies)
            positional[:, 1::2] = torch.cos(positions * frequencies)
            positional = positional.to(dtype=padded_embeddings.dtype).unsqueeze(0)
            positional = positional * encoder_attention_mask.unsqueeze(-1).to(positional.dtype)
            padded_embeddings = (
                padded_embeddings
                + self.block_position_scale.to(padded_embeddings.dtype) * positional
            )
        elif position_mode != "off":
            raise ValueError(
                f"Unknown GRAPH_BLOCK_POSITION_MODE={position_mode!r}; "
                "use off or sinusoidal"
            )

        # 4. Multihead self-attention with a parameter-matched identity
        # ablation. The attention module remains instantiated so the control
        # changes computation, not model construction or checkpoint layout.
        attention_ablation = os.environ.get(
            "GRAPH_GLOBAL_ATTENTION_ABLATION", "full"
        ).strip().lower()
        if attention_ablation == "full":
            # key_padding_mask expects True for padded positions.
            key_padding_mask = encoder_attention_mask == 0
            attended, _ = self.attention(
                padded_embeddings,
                padded_embeddings,
                padded_embeddings,
                key_padding_mask=key_padding_mask,
            )
        elif attention_ablation == "identity":
            attended = padded_embeddings
        else:
            raise ValueError(
                "Unknown GRAPH_GLOBAL_ATTENTION_ABLATION="
                f"{attention_ablation!r}; use full or identity"
            )

        # 5. Project and return (shape: [batch_size, max_blocks, hidden_size] and [batch_size, max_blocks])
        return self.projection(attended), encoder_attention_mask
