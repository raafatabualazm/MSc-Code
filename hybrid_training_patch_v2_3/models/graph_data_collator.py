from __future__ import annotations

import math

import torch
from torch_geometric.data import Batch as PyGBatch


class GraphDataCollator:
    """Dynamically pad decoder-side sequences to the current batch maximum.

    Graph/block tensors keep their existing representation. Labels are padded with
    ``-100`` and causal prompt tokens are right-padded with the decoder pad token.
    This avoids allocating every example at the global 3k-token safety ceiling.
    """

    def __init__(self, target_tokenizer, pad_to_multiple_of=None):
        self.target_tokenizer = target_tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _padded_length(self, lengths: list[int]) -> int:
        maximum = max(lengths, default=0)
        if self.pad_to_multiple_of and maximum:
            multiple = int(self.pad_to_multiple_of)
            maximum = int(math.ceil(maximum / multiple) * multiple)
        return maximum

    def __call__(self, features):
        if not features:
            raise ValueError("GraphDataCollator received an empty feature list")

        batch = {}

        if "labels" in features[0]:
            label_lists = [list(feature["labels"]) for feature in features]
            max_len = self._padded_length([len(ids) for ids in label_lists])
            padded = [ids + [-100] * (max_len - len(ids)) for ids in label_lists]
            batch["labels"] = torch.tensor(padded, dtype=torch.long)

        if "decoder_prompt_input_ids" in features[0]:
            pad_id = self.target_tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.target_tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0

            prompt_ids = [list(feature["decoder_prompt_input_ids"]) for feature in features]
            prompt_masks = [
                list(feature.get("decoder_prompt_attention_mask") or [1] * len(ids))
                for feature, ids in zip(features, prompt_ids)
            ]
            for index, (ids, mask) in enumerate(zip(prompt_ids, prompt_masks)):
                if len(ids) != len(mask):
                    raise ValueError(
                        f"decoder prompt ids/mask length mismatch at batch index {index}: "
                        f"ids={len(ids)} mask={len(mask)}"
                    )
            max_len = self._padded_length([len(ids) for ids in prompt_ids])
            batch["decoder_prompt_input_ids"] = torch.tensor(
                [ids + [pad_id] * (max_len - len(ids)) for ids in prompt_ids],
                dtype=torch.long,
            )
            batch["decoder_prompt_attention_mask"] = torch.tensor(
                [mask + [0] * (max_len - len(mask)) for mask in prompt_masks],
                dtype=torch.long,
            )

        if "pyg_data" in features[0]:
            pyg_graphs = [feature["pyg_data"] for feature in features]
            batch["pyg_graph"] = PyGBatch.from_data_list(pyg_graphs)

        passthrough = (
            "cfg",
            "edges",
            "block_inputs",
            "language",
            "tests",
        )
        for key in passthrough:
            if key in features[0]:
                batch[key] = [feature[key] for feature in features]

        # Encoder-side tensors are already fixed-width by the block tensor builder.
        tensor_keys = (
            "input_ids",
            "attention_mask",
            "position_ids",
            "token_type_ids",
        )
        for key in tensor_keys:
            if key in features[0]:
                values = [torch.as_tensor(feature[key]) for feature in features]
                batch[key] = torch.stack(values)

        return batch
