
from __future__ import annotations

import torch


class GraphCodeBERTTensorBuilder:
    def __init__(self, tokenizer, max_seq_len=512, max_dfg_nodes=64):
        self.tokenizer = tokenizer
        self.max_len = min(max_seq_len, 512)
        self.max_dfg = max_dfg_nodes

    def build_block_tensors(self, block_instructions, dfg_meta):
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

        if len(code_tokens) > (available_slots - 8):
            code_tokens = code_tokens[:(available_slots - 8)]

        remaining_dfg_budget = min(
            self.max_dfg,
            available_slots - len(code_tokens)
        )

        raw_dfg_nodes = dfg_meta['nodes']
        raw_dfg_edges = dfg_meta['edges']

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
        pruned_node_set = set(pruned_dfg_nodes)

        pruned_dfg_edges = [
            (source, target)
            for source, target in raw_dfg_edges
            if source in pruned_node_set and target in pruned_node_set
        ]

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

        attention_mask = torch.full(
            (self.max_len, self.max_len),
            -10000.0,
        )

        attention_mask[:code_len, :code_len] = 0.0

        node_to_idx = {
            node: code_len + idx
            for idx, node in enumerate(pruned_dfg_nodes)
            if (code_len + idx) < self.max_len
        }

        for instruction_index, nodes in dfg_meta['instruction_to_nodes'].items():
            token_positions = instruction_token_spans.get(instruction_index, [])

            for node in nodes:
                if node not in node_to_idx:
                    continue

                node_idx = node_to_idx[node]
                attention_mask[node_idx, node_idx] = 0.0

                for token_position in token_positions:
                    if token_position >= self.max_len:
                        continue

                    attention_mask[token_position, node_idx] = 0.0
                    attention_mask[node_idx, token_position] = 0.0

        for source_node, target_node in pruned_dfg_edges:
            if source_node in node_to_idx and target_node in node_to_idx:
                source_idx = node_to_idx[source_node]
                target_idx = node_to_idx[target_node]

                attention_mask[source_idx, target_idx] = 0.0
                attention_mask[target_idx, source_idx] = 0.0

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': attention_mask.unsqueeze(0),
        }
