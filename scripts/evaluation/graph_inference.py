
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from models.hierarchical_graph_encoder import LocalBlockEncoder, GraphPoolingEncoder
from models.pyg_cfg_dataset import cfg_to_pyg
from scripts.data.dfg_extractor import LightweightDFGExtractor

ENCODER_MODEL = "microsoft/graphcodebert-base"
DECODER_MODEL = "t5-small"


class GraphInferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = LocalBlockEncoder()
        self.graph_encoder = GraphPoolingEncoder()
        self.t5 = T5ForConditionalGeneration.from_pretrained(DECODER_MODEL)
        self.proj = torch.nn.Linear(768, self.t5.config.d_model)

    @torch.no_grad()
    def encode_function(self, block_tensors, graph_data, device):
        embeddings = []

        for block in block_tensors:
            output = self.local_encoder(
                input_ids=block['input_ids'].to(device),
                attention_mask=torch.ones_like(block['input_ids']).to(device),
                position_ids=block['position_ids'].to(device),
                token_type_ids=block['token_type_ids'].to(device),
            )
            embeddings.append(output.squeeze(0))

        if not embeddings:
            embeddings = [torch.zeros(768, device=device)]

        node_states = torch.stack(embeddings)

        pooled = self.graph_encoder(
            node_states,
            graph_data.edge_index.to(device) if graph_data.edge_index is not None else None,
            graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None,
        )

        return self.proj(pooled.unsqueeze(1))

    @torch.no_grad()
    def generate(self, block_tensors, graph_data, tokenizer, device, max_new_tokens=256):
        encoder_hidden_states = self.encode_function(block_tensors, graph_data, device)

        generated = self.t5.generate(
            input_ids=torch.tensor([[self.t5.config.decoder_start_token_id]], device=device),
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            max_new_tokens=max_new_tokens,
        )

        return tokenizer.decode(generated[0], skip_special_tokens=True)


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_blocks(record, tensor_builder, dfg_extractor):
    cfg = record.get('cfg', [])

    if not cfg:
        assembly = record['assembly'].splitlines()
        cfg = [{
            'block_id': 'entry',
            'block_type': 'entry',
            'instructions': assembly,
        }]

    block_tensors = []

    for block in cfg:
        dfg = dfg_extractor.extract_block_dfg_structured(block['instructions'])

        tensors = tensor_builder.build_block_tensors(
            block['instructions'],
            dfg,
        )

        block_tensors.append(tensors)

    graph_data = cfg_to_pyg({
        'cfg': cfg,
        'edges': record.get('edges', []),
    }, torch.zeros((max(len(cfg), 1), 768)))

    return block_tensors, graph_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=165)
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    tensor_builder = GraphCodeBERTTensorBuilder(tokenizer)
    dfg_extractor = LightweightDFGExtractor()

    model = GraphInferenceModel().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'loaded checkpoint: {args.checkpoint}')
    model.eval()

    rows = load_jsonl(args.dataset)[:args.limit]
    outputs = []

    for idx, row in enumerate(rows):
        block_tensors, graph_data = build_blocks(
            row,
            tensor_builder,
            dfg_extractor,
        )

        prediction = model.generate(
            block_tensors,
            graph_data,
            tokenizer,
            device,
        )

        outputs.append({
            'id': idx,
            'prediction': prediction,
            'reference': row.get('source', ''),
            'language': row.get('language', 'dart'),
        })

        print(f'[{idx + 1}/{len(rows)}] generated')

    Path(args.output).write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
