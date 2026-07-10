
from __future__ import annotations

import os

import torch


class GraphCodeBERTTensorBuilder:
    """Per-block token tensors for the local (GraphCodeBERT-family) encoder.

    GRAPH_DFG_MODE:
      legacy (default) - byte-identical historical behavior: intra-block DFG
        node NAMES are appended after [SEP]. NOTE these names are not in the
        vocab, so they all become <unk>; with the structure-aware mask gone
        (see attention-mask note below) they carry no dataflow information and
        only spend window budget. Kept solely so old checkpoints reproduce.
      off / edges - nothing is appended; the full window (max_len-2) goes to
        assembly tokens. In 'edges' mode dataflow instead enters the model as
        cross-block 'dataflow' edges on the CFG graph (see
        scripts/data/dfg_extractor.build_cross_block_dfg, wired in
        ensure_cfg_blocks), which the GNN consumes directly.

    GRAPH_POSITION_SCHEME:
      legacy (default) - code positions start at 0, DFG tokens share the [SEP]
        position, pads get 0. Historical behavior; note RoBERTa positions
        0 and 1 are untrained/zeroed in the pretrained checkpoint.
      roberta - pretraining-faithful: code tokens start at padding_idx+1 = 2
        (matching create_position_ids_from_input_ids), DFG tokens get 0
        (original GraphCodeBERT convention), pads get padding_idx = 1.
        Use for new runs, especially with a frozen encoder.
    """

    def __init__(self, tokenizer, max_seq_len=512, max_dfg_nodes=64,
                 dfg_mode=None, position_scheme=None):
        self.tokenizer = tokenizer
        self.max_len = min(max_seq_len, 512)
        self.max_dfg = max_dfg_nodes
        self.dfg_mode = (dfg_mode or os.environ.get("GRAPH_DFG_MODE", "legacy")).lower()
        self.position_scheme = (
            position_scheme or os.environ.get("GRAPH_POSITION_SCHEME", "legacy")
        ).lower()

    def build_block_tensors(self, block_instructions, dfg_meta):
        append_dfg_tokens = self.dfg_mode == 'legacy'

        instruction_token_spans = {}

        code_tokens = []
        current_position = 1

        for instruction_index, instruction in enumerate(block_instructions):
            tokens = self.tokenizer.tokenize(instruction)

            instruction_token_spans[instruction_index] = list(
                range(current_position, current_position + len(tokens))
            )

            current_position += len(tokens)
            code_tokens.extend(tokens)

        if not code_tokens:
            code_tokens = [self.tokenizer.unk_token]

        available_slots = self.max_len - 2

        if append_dfg_tokens:
            if len(code_tokens) > (available_slots - 8):
                code_tokens = code_tokens[:(available_slots - 8)]

            remaining_dfg_budget = min(
                self.max_dfg,
                available_slots - len(code_tokens)
            )

            raw_dfg_nodes = (dfg_meta or {}).get('nodes', [])
            raw_dfg_edges = (dfg_meta or {}).get('edges', [])

            register_priority = {
                'rax': 5,
                'rbx': 4,
                'rcx': 4,
                'rdx': 4,
                'rbp': 5,
                'rsp': 5,
            }

            node_scores = {
                node: 0
                for node in raw_dfg_nodes
            }

            for source, target in raw_dfg_edges:
                if source in node_scores:
                    node_scores[source] += 1

                if target in node_scores:
                    node_scores[target] += 1

            for node in raw_dfg_nodes:
                register_name = node.split('_')[0]
                node_scores[node] += register_priority.get(register_name, 0)

                if register_name in ['rax', 'rbx']:
                    node_scores[node] += 2

            sorted_nodes = sorted(
                raw_dfg_nodes,
                key=lambda node: node_scores.get(node, 0),
                reverse=True,
            )

            pruned_dfg_nodes = sorted_nodes[:remaining_dfg_budget]
        else:
            # Dataflow lives on the graph ('edges' mode) or is disabled ('off');
            # spend the entire window on real assembly tokens.
            if len(code_tokens) > available_slots:
                code_tokens = code_tokens[:available_slots]
            pruned_dfg_nodes = []

        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]

        code_len = len(tokens)

        tokens = tokens + pruned_dfg_nodes
        total_len = len(tokens)

        if total_len > self.max_len:
            tokens = tokens[:self.max_len]
            total_len = self.max_len

        if total_len <= 2:
            tokens.append(self.tokenizer.unk_token)
            total_len = len(tokens)

        padding_len = self.max_len - total_len

        padded_tokens = tokens + [self.tokenizer.pad_token] * padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)

        sep_position = code_len - 1

        if self.position_scheme == 'roberta':
            # Pretraining-faithful RoBERTa positions: real tokens from
            # padding_idx+1 (=2), DFG tokens at 0 (GraphCodeBERT convention),
            # pads at padding_idx (=1, the zeroed embedding row).
            # max index = code_len+1 <= 513 < 514 embedding rows.
            position_ids = (
                list(range(2, code_len + 2)) +
                [0] * len(pruned_dfg_nodes)
            )[:total_len]
            position_ids += [1] * padding_len
        else:
            position_ids = (
                list(range(code_len)) +
                [sep_position] * len(pruned_dfg_nodes)
            )[:total_len]
            position_ids += [0] * padding_len

        token_type_ids = (
            [0] * code_len +
            [1] * len(pruned_dfg_nodes)
        )[:total_len]

        token_type_ids += [0] * padding_len

        # The local encoder (LocalBlockEncoder.forward) consumes a 2D
        # token-visibility mask: 1.0 for real tokens, 0.0 for padding. The old
        # per-block (max_len, max_len) square DFG mask was dropped because it
        # dominated CPU/RAM once functions became many-block (median 17, up to
        # ~450 blocks) and the encoder collapsed it to this 2D form anyway.
        attention_mask = torch.zeros(self.max_len, dtype=torch.float32)
        attention_mask[:total_len] = 1.0

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': attention_mask.unsqueeze(0),
        }
