
from __future__ import annotations

import torch
from torch_geometric.data import Batch as PyGBatch


class GraphDataCollator:
    def __init__(self, target_tokenizer, pad_to_multiple_of=None):
        self.target_tokenizer = target_tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        batch = {}

        if 'labels' in features[0]:
            # Labels may already contain -100 (ignore_index) for padded positions,
            # which tokenizer.pad cannot handle. Pad manually with -100 so the loss
            # ignores padded positions.
            label_lists = [list(feature['labels']) for feature in features]
            max_len = max(len(ids) for ids in label_lists)
            padded = [ids + [-100] * (max_len - len(ids)) for ids in label_lists]
            batch['labels'] = torch.tensor(padded, dtype=torch.long)

        if 'pyg_data' in features[0]:
            pyg_graphs = [f['pyg_data'] for f in features]
            batch['pyg_graph'] = PyGBatch.from_data_list(pyg_graphs)

        passthrough = [
            'cfg',
            'edges',
            'block_inputs',
        ]

        for key in passthrough:
            if key in features[0]:
                batch[key] = [f[key] for f in features]

        tensor_keys = [
            'input_ids',
            'attention_mask',
            'position_ids',
            'token_type_ids',
            'decoder_prompt_input_ids',
            'decoder_prompt_attention_mask',
        ]

        for key in tensor_keys:
            if key in features[0]:
                values = [torch.tensor(f[key]) for f in features]
                batch[key] = torch.stack(values)

        return batch
