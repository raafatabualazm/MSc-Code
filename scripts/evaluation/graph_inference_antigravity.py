"""
Graph-aware decompiler inference (Antigravity version).
Supports T5/CodeT5 and Qwen decoders, multi-sample generation (K samples), and correct attention masks.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast, PreTrainedModel
    _old_add_tokens = PreTrainedTokenizerFast._add_tokens
    
    def _patched_add_tokens(self, new_tokens, special_tokens=False):
        dict_or_attr = lambda o, k, d: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
        conv_tokens = []
        for t in new_tokens:
            if isinstance(t, str):
                conv_tokens.append(t)
            else:
                conv_tokens.append(tokenizers.AddedToken(
                    dict_or_attr(t, 'content', str(t)),
                    single_word=dict_or_attr(t, 'single_word', False),
                    lstrip=dict_or_attr(t, 'lstrip', False),
                    rstrip=dict_or_attr(t, 'rstrip', False),
                    normalized=dict_or_attr(t, 'normalized', True),
                    special=dict_or_attr(t, 'special', False)
                ))
        if special_tokens:
            return self._tokenizer.add_special_tokens(conv_tokens)
        return self._tokenizer.add_tokens(conv_tokens)
        
    PreTrainedTokenizerFast._add_tokens = _patched_add_tokens
    
    # Support newer transformers quantizer tie check on older Salesforce model
    PreTrainedModel.all_tied_weights_keys = property(
        lambda self: {k: None for k in getattr(self, "_all_tied_weights_keys", [])} if isinstance(getattr(self, "_all_tied_weights_keys", None), (list, set)) else getattr(self, "_all_tied_weights_keys", {}),
        lambda self, val: setattr(self, "_all_tied_weights_keys", val)
    )
    
    PreTrainedModel.get_head_mask = lambda self, head_mask, num_hidden_layers, is_attention_chunked=False: \
        head_mask if head_mask is not None else [None] * num_hidden_layers
    
    # Patch CodeT5pEncoderDecoderModel.tie_weights dynamically inside PreTrainedModel.__init__
    _old_init = PreTrainedModel.__init__
    def _patched_init(self, *args, **kwargs):
        _old_init(self, *args, **kwargs)
        cls = self.__class__
        if cls.__name__ == "CodeT5pEncoderDecoderModel" and not getattr(cls, "_tie_weights_patched", False):
            old_tie_weights = cls.tie_weights
            def patched_tie_weights(self, *args, **kwargs):
                return old_tie_weights(self)
            cls.tie_weights = patched_tie_weights
            cls._tie_weights_patched = True
            print("Successfully patched CodeT5pEncoderDecoderModel.tie_weights dynamically!")
    PreTrainedModel.__init__ = _patched_init
except Exception as e:
    pass

import argparse
import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput

from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from models.hierarchical_graph_encoder_antigravity import LocalBlockEncoder, GraphPoolingEncoder
from models.pyg_cfg_dataset import cfg_to_pyg
from scripts.data.dfg_extractor import LightweightDFGExtractor
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import GraphCodeBERTT5Seq2Seq

ENCODER_MODEL = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")


class GraphInferenceModel(torch.nn.Module):
    def __init__(self, decoder_model_name):
        super().__init__()
        # Use our training model wrapper directly to keep architecture, PEFT loading, and causal/seq2seq logic unified!
        os.environ["GRAPH_DECODER_MODEL"] = decoder_model_name
        self.decompiler = GraphCodeBERTT5Seq2Seq()
        self.is_causal = self.decompiler.is_causal
        self.t5_model = self.decompiler.t5_model
        
    @torch.no_grad()
    def encode_function(self, block_tensors, graph_data, device):
        embeddings = []

        for block in block_tensors:
            output = self.decompiler.local_encoder(
                input_ids=block['input_ids'].to(device),
                attention_mask=block['attention_mask'].to(device),
                position_ids=block['position_ids'].to(device),
                token_type_ids=block['token_type_ids'].to(device),
            )
            embeddings.append(output.squeeze(0))

        if not embeddings:
            embeddings = [torch.zeros(768, device=device)]

        node_states = torch.stack(embeddings)

        # GraphPoolingEncoder in antigravity mode preserves the sequence of blocks
        pooled, encoder_attention_mask = self.decompiler.graph_encoder(
            node_states,
            graph_data.edge_index.to(device) if graph_data.edge_index is not None else None,
            graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None,
            list_of_B_i=None, # None indicates single example inference
        ) # shapes: (1, num_blocks, 768) and (1, num_blocks)

        projected = self.decompiler.projection(pooled) # (1, num_blocks, decoder_dim)
        return projected, encoder_attention_mask

    @torch.no_grad()
    def generate(self, block_tensors, graph_data, decoder_tokenizer, device, max_new_tokens=256, num_samples=1):
        encoder_hidden_states, encoder_attention_mask = self.encode_function(block_tensors, graph_data, device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.t5_model.dtype)
        
        do_sample = num_samples > 1
        temp = 0.7 if do_sample else None
        top_p = 0.95 if do_sample else None

        if self.is_causal:
            # Causal Prefix Injection
            start_token_id = decoder_tokenizer.bos_token_id or decoder_tokenizer.eos_token_id or 0
            input_ids = torch.tensor([[start_token_id]], device=device)
            
            if num_samples > 1:
                input_ids = input_ids.expand(num_samples, -1)
                encoder_hidden_states = encoder_hidden_states.expand(num_samples, -1, -1)
                encoder_attention_mask = encoder_attention_mask.expand(num_samples, -1)
                
            base_model = self.decompiler.base_decoder_model
            input_embeds = base_model.get_input_embeddings()(input_ids)
            
            inputs_embeds = torch.cat([encoder_hidden_states, input_embeds], dim=1)
            inputs_embeds = inputs_embeds.to(dtype=base_model.dtype)
            prefix_len = encoder_hidden_states.size(1)
            
            combined_mask = torch.cat([
                encoder_attention_mask,
                torch.ones((encoder_attention_mask.size(0), 1), dtype=encoder_attention_mask.dtype, device=device)
            ], dim=1)
            
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p,
                eos_token_id=decoder_tokenizer.eos_token_id,
                pad_token_id=decoder_tokenizer.pad_token_id or decoder_tokenizer.eos_token_id,
            )
            
            predictions = []
            for out in outputs:
                pred = decoder_tokenizer.decode(out, skip_special_tokens=True)
                predictions.append(pred)
            return predictions

        else:
            # Seq2Seq Decoder
            cfg = self.t5_model.config
            start_token_id = getattr(cfg, 'decoder_start_token_id', None)
            if start_token_id is None and hasattr(cfg, 'decoder'):
                start_token_id = getattr(cfg.decoder, 'decoder_start_token_id', None)
            if start_token_id is None:
                start_token_id = getattr(cfg, 'pad_token_id', None)
            if start_token_id is None:
                start_token_id = decoder_tokenizer.pad_token_id or 0
            input_ids = torch.tensor([[start_token_id]], device=device)
            
            if num_samples > 1:
                input_ids = input_ids.expand(num_samples, -1)
                encoder_hidden_states = encoder_hidden_states.expand(num_samples, -1, -1)
                encoder_attention_mask = encoder_attention_mask.expand(num_samples, -1)
                
            outputs = self.t5_model.generate(
                decoder_input_ids=input_ids,
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
                attention_mask=encoder_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p,
                eos_token_id=decoder_tokenizer.eos_token_id,
                pad_token_id=decoder_tokenizer.pad_token_id or decoder_tokenizer.eos_token_id,
            )
            
            predictions = []
            for out in outputs:
                pred = decoder_tokenizer.decode(out, skip_special_tokens=True)
                predictions.append(pred)
            return predictions


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
    parser.add_argument('--decoder_model', default='t5-small')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=165)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--num_samples', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set default PEFT env variables based on checkpoint path to ensure correct model architecture loading
    if args.checkpoint:
        checkpoint_lower = args.checkpoint.lower()
        if "lora" in checkpoint_lower:
            os.environ.setdefault("GRAPH_ENCODER_PEFT", "lora")
            os.environ.setdefault("GRAPH_DECODER_PEFT", "lora")
        else:
            os.environ.setdefault("GRAPH_ENCODER_PEFT", "none")
            os.environ.setdefault("GRAPH_DECODER_PEFT", "none")

    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, trust_remote_code=True)
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model, trust_remote_code=True)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=512)
    dfg_extractor = LightweightDFGExtractor()

    model = GraphInferenceModel(args.decoder_model).to(device)
    if args.checkpoint:
        # Load custom PyTorch state dict
        state = torch.load(args.checkpoint, map_location=device)
        
        # Load directly into decompiler wrapper if keys match decompiler module name layout
        if any(k.startswith('local_encoder') for k in state.keys()) and not hasattr(model, 'local_encoder'):
            missing, unexpected = model.decompiler.load_state_dict(state, strict=False)
        else:
            missing, unexpected = model.load_state_dict(state, strict=False)
            
        print(f'loaded checkpoint: {args.checkpoint}')
        print(f'Loaded parameters with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
    model.eval()

    rows = load_jsonl(args.dataset)[:args.limit]
    outputs = []

    for idx, row in enumerate(rows):
        block_tensors, graph_data = build_blocks(
            row,
            tensor_builder,
            dfg_extractor,
        )

        predictions = model.generate(
            block_tensors,
            graph_data,
            decoder_tokenizer,
            device,
            max_new_tokens=256,
            num_samples=args.num_samples,
        )

        outputs.append({
            'id': idx,
            'predictions': predictions,
            'reference': row.get('source', ''),
            'language': row.get('language', 'dart'),
        })

        print(f'[{idx + 1}/{len(rows)}] generated {len(predictions)} candidates')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
