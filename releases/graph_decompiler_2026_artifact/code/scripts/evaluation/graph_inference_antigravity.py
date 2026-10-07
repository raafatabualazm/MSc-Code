"""
Graph-aware decompiler inference (Antigravity version).
Supports T5/CodeT5 and Qwen decoders, multi-sample generation (K samples), and correct attention masks.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Quiet mode (default on): suppress framework warning spam so generation
# progress logs stay readable. Errors are NOT suppressed. GRAPH_QUIET=0 to
# see everything.
import os
if os.environ.get("GRAPH_QUIET", "1") != "0":
    import warnings
    for _cat in (FutureWarning, UserWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=_cat)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast, PreTrainedModel
    from transformers.configuration_utils import PretrainedConfig
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

    # Salesforce CodeT5+ custom config cannot be constructed with no args.
    # Transformers 4.57 calls `self.__class__()` while logging config diffs,
    # which raises before weights load. Fall back to the full dict for repr/json.
    if not getattr(PretrainedConfig, "_codet5p_to_diff_patched", False):
        _old_to_diff_dict = PretrainedConfig.to_diff_dict
        _old_get_non_default_generation_parameters = getattr(
            PretrainedConfig,
            "_get_non_default_generation_parameters",
            None,
        )

        def _full_json_safe_config_dict(config):
            config_dict = config.to_dict()
            if hasattr(config, "dict_torch_dtype_to_str"):
                config.dict_torch_dtype_to_str(config_dict)
            elif hasattr(config, "dict_dtype_to_str"):
                config.dict_dtype_to_str(config_dict)
            return config_dict

        def _patched_to_diff_dict(self):
            try:
                return _old_to_diff_dict(self)
            except AssertionError:
                return _full_json_safe_config_dict(self)

        PretrainedConfig.to_diff_dict = _patched_to_diff_dict

        if _old_get_non_default_generation_parameters is not None:
            def _patched_get_non_default_generation_parameters(self):
                try:
                    return _old_get_non_default_generation_parameters(self)
                except AssertionError:
                    return {}

            PretrainedConfig._get_non_default_generation_parameters = _patched_get_non_default_generation_parameters
        PretrainedConfig._codet5p_to_diff_patched = True
    
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

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from models.hierarchical_graph_encoder_antigravity import LocalBlockEncoder, GraphPoolingEncoder
from models.pyg_cfg_dataset import cfg_to_pyg
from scripts.data.dfg_extractor import LightweightDFGExtractor
from scripts.data.cfg_extractor import ensure_cfg_blocks
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import GraphCodeBERTT5Seq2Seq
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import build_decoder_prompt
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import maybe_override_qwen_prefix_gate

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
        # Text-only mode (causal + GRAPH_QWEN_PREFIX_TOKENS=0): zero-width
        # prefix, no graph compute — keeps ablation arms honest (before
        # 2026-07-04 this path prepended the raw block states instead).
        if self.decompiler.is_causal and self.decompiler.qwen_prefix_tokens == 0:
            return (
                torch.zeros((1, 0, self.decompiler.decoder_dim), device=device),
                torch.zeros((1, 0), device=device),
            )

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
        projected, encoder_attention_mask = self.decompiler.prepare_decoder_context(
            projected,
            encoder_attention_mask,
        )
        return projected, encoder_attention_mask

    @torch.no_grad()
    def generate(
        self,
        block_tensors,
        graph_data,
        decoder_tokenizer,
        device,
        max_new_tokens=256,
        num_samples=1,
        decoder_prompt_input_ids=None,
        decoder_prompt_attention_mask=None,
        do_sample=None,
    ):
        encoder_hidden_states, encoder_attention_mask = self.encode_function(block_tensors, graph_data, device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.t5_model.dtype)

        # do_sample must reflect the TOTAL candidate count, not this chunk's
        # size: with generation_batch_size=2 and num_samples=5, the last
        # chunk has size 1 and used to silently fall back to greedy decoding,
        # mixing one greedy candidate into an otherwise sampled pool.
        if do_sample is None:
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

            prompt_embeds = None
            if decoder_prompt_input_ids is not None:
                decoder_prompt_input_ids = decoder_prompt_input_ids.to(device)
                prompt_embeds = base_model.get_input_embeddings()(decoder_prompt_input_ids)
                if num_samples > 1:
                    prompt_embeds = prompt_embeds.expand(num_samples, -1, -1)
                    decoder_prompt_attention_mask = decoder_prompt_attention_mask.expand(num_samples, -1)
            
            embed_parts = [encoder_hidden_states]
            if prompt_embeds is not None:
                embed_parts.append(prompt_embeds)
            embed_parts.append(input_embeds)
            inputs_embeds = torch.cat(embed_parts, dim=1)
            inputs_embeds = inputs_embeds.to(dtype=base_model.dtype)
            
            mask_parts = [encoder_attention_mask]
            if decoder_prompt_attention_mask is not None:
                mask_parts.append(decoder_prompt_attention_mask.to(device=device, dtype=encoder_attention_mask.dtype))
            mask_parts.append(torch.ones((encoder_attention_mask.size(0), 1), dtype=encoder_attention_mask.dtype, device=device))
            combined_mask = torch.cat(mask_parts, dim=1)
            
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.15,
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
                repetition_penalty=1.15,
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
    # Same CFG resolution as training (precomputed cfg -> inline auto-extract ->
    # single-block fallback) so inference graphs match what the model trained on.
    cfg, edges = ensure_cfg_blocks(record)

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
        'edges': edges,
    }, torch.zeros((max(len(cfg), 1), 768)))

    return block_tensors, graph_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--decoder_model', default='t5-small')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help="Maximum rows to evaluate; 0 means all rows")
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument(
        '--generation_batch_size',
        type=int,
        default=int(os.environ.get('GRAPH_EVAL_GENERATION_BATCH_SIZE', '5')),
        help="Maximum candidates to generate in one forward batch; preserves total num_samples while reducing VRAM spikes",
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=int(os.environ.get('GRAPH_EVAL_MAX_NEW_TOKENS', '768')),
        help="Generation budget per candidate. Keep aligned with SFT max_target_length; too small gives high prefix CodeBLEU but uncompilable truncated Dart.",
    )
    parser.add_argument(
        '--decoder_prompt_max_length',
        type=int,
        default=int(os.environ.get('GRAPH_DECODER_PROMPT_MAX_LENGTH', '768')),
        help="Max tokens for the Qwen text prompt used alongside the graph prefix.",
    )
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
    maybe_override_qwen_prefix_gate(model.decompiler)
    model.eval()

    rows = load_jsonl(args.dataset)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    outputs = []

    progress = None
    row_iter = rows
    if tqdm is not None:
        progress = tqdm(rows, desc="generate", unit="row", dynamic_ncols=True)
        row_iter = progress
    log_line = progress.write if progress is not None else print

    for idx, row in enumerate(row_iter):
        block_tensors, graph_data = build_blocks(
            row,
            tensor_builder,
            dfg_extractor,
        )
        prompt_tensors = None
        if model.is_causal:
            prompt_tensors = decoder_tokenizer(
                build_decoder_prompt(row, decoder_tokenizer, args.decoder_prompt_max_length),
                max_length=args.decoder_prompt_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        predictions = []
        remaining = args.num_samples
        generation_batch_size = max(1, args.generation_batch_size)
        while remaining > 0:
            batch_n = min(generation_batch_size, remaining)
            predictions.extend(model.generate(
                block_tensors,
                graph_data,
                decoder_tokenizer,
                device,
                max_new_tokens=args.max_new_tokens,
                num_samples=batch_n,
                decoder_prompt_input_ids=prompt_tensors["input_ids"] if prompt_tensors is not None else None,
                decoder_prompt_attention_mask=prompt_tensors["attention_mask"] if prompt_tensors is not None else None,
                do_sample=args.num_samples > 1,
            ))
            remaining -= batch_n

        outputs.append({
            'id': row.get('task_id', idx),
            'source_line': idx + 1,
            'filename': row.get('filename', ''),
            'predictions': predictions,
            'reference': row.get('source', row.get('dart_source', '')),
            'language': row.get('language', row.get('lang', 'dart')),
            'tests': row.get('tests', ''),
        })

        log_line(f'[{idx + 1}/{len(rows)}] generated {len(predictions)} candidates')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
