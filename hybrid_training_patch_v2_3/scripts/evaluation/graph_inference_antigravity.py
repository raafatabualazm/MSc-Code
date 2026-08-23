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
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, set_seed
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
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import PROMPT_SCHEMA_VERSION
from scripts.training.hybrid_data_controls import instruction_count
from scripts.training.checkpoint_contract import validate_trainable_checkpoint_load
from scripts.provenance_antigravity import (
    file_record,
    git_state,
    graph_environment,
    model_commit,
    runtime_record,
    write_json,
)

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
    def encode_function(self, block_tensors, graph_data, device, force_null_graph=False):
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
            region_ids=graph_data.region_id.to(device),
        ) # shapes: (1, num_blocks, 768) and (1, num_blocks)

        projected = self.decompiler.projection(pooled) # (1, num_blocks, decoder_dim)
        projected, encoder_attention_mask = self.decompiler.prepare_decoder_context(
            projected,
            encoder_attention_mask,
        )
        if force_null_graph:
            # Preserve the exact prefix length/mask/positions while removing all
            # graph content. This isolates content use from a generic soft-prefix
            # mode switch and works for both causal and seq2seq decoders.
            projected = torch.zeros_like(projected)
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
        force_null_graph=False,
    ):
        encoder_hidden_states, encoder_attention_mask = self.encode_function(
            block_tensors, graph_data, device, force_null_graph=force_null_graph
        )
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
                use_cache=True,
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
                use_cache=True,
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


def _row_identity(row: dict, index: int) -> str:
    return str(row.get('task_id', row.get('filename', index)))


def _remap_index_list(values, old_to_new):
    if not isinstance(values, list):
        return values
    return [old_to_new[value] for value in values if value in old_to_new]


def _shuffle_graph_blocks(row: dict, seed: int, index: int) -> tuple[dict, dict]:
    graph_row = copy.deepcopy(row)
    cfg = graph_row.get('cfg')
    edges = graph_row.get('edges')
    if not isinstance(cfg, list) or not cfg:
        raise ValueError(
            "shuffle_blocks requires a non-empty precomputed cfg for every row"
        )
    if not isinstance(edges, list):
        raise ValueError("shuffle_blocks requires precomputed edges for every row")

    identity = _row_identity(row, index)
    seed_material = f"{seed}|{index}|{identity}|{len(cfg)}"
    local_seed = int.from_bytes(
        hashlib.sha256(seed_material.encode('utf-8')).digest()[:8], 'big'
    )
    permutation = list(range(len(cfg)))  # new index -> old index
    random.Random(local_seed).shuffle(permutation)
    if len(permutation) > 1 and permutation == list(range(len(cfg))):
        permutation = permutation[1:] + permutation[:1]
    old_to_new = {old: new for new, old in enumerate(permutation)}

    shuffled_cfg = []
    for new_index, old_index in enumerate(permutation):
        block = copy.deepcopy(cfg[old_index])
        block['id'] = new_index
        block['label'] = f'block_{new_index}'
        block['predecessors'] = _remap_index_list(
            block.get('predecessors', []), old_to_new
        )
        block['successors'] = _remap_index_list(
            block.get('successors', []), old_to_new
        )
        shuffled_cfg.append(block)

    shuffled_edges = []
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source not in old_to_new or target not in old_to_new:
            raise ValueError(
                f"edge endpoint outside cfg while shuffling row {identity}: "
                f"{source}->{target}"
            )
        shuffled_edges.append({
            **edge,
            'source': old_to_new[source],
            'target': old_to_new[target],
        })

    graph_row['cfg'] = shuffled_cfg
    graph_row['edges'] = shuffled_edges
    integrity = copy.deepcopy(graph_row.get('integrity') or {})
    if isinstance(integrity.get('entry_block'), int):
        integrity['entry_block'] = old_to_new[integrity['entry_block']]
    for key in ('entry_blocks', 'isolated_nodes', 'isolated_nonentry_nodes', 'unreachable_nodes'):
        if key in integrity:
            integrity[key] = _remap_index_list(integrity[key], old_to_new)
    graph_row['integrity'] = integrity

    permutation_sha = hashlib.sha256(
        json.dumps(permutation, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return graph_row, {
        'mode': 'shuffle_blocks',
        'target_id': identity,
        'donor_id': identity,
        'block_count': len(cfg),
        'permutation_sha256': permutation_sha,
        'changed': permutation != list(range(len(cfg))),
    }


def _graph_complexity(row: dict) -> tuple[int, int]:
    cfg = row.get('cfg')
    block_count = len(cfg) if isinstance(cfg, list) and cfg else 1
    return block_count, instruction_count(row)


def _matched_derangement(rows: list[dict], seed: int) -> list[int]:
    """Return a deterministic, shape-matched BIJECTION with no self donors.

    Rows are sorted by (block count, instruction count, seeded tie break), then
    the cyclic offset with the smallest total shape distance is selected. This
    is a one-to-one donor map, unlike independent nearest-neighbour sampling.
    """
    if len(rows) < 2:
        raise ValueError('matched_permutation requires at least two rows')
    identities = [_row_identity(row, index) for index, row in enumerate(rows)]
    if len(set(identities)) != len(identities):
        raise ValueError('matched_permutation requires unique task identities')
    complexities = [_graph_complexity(row) for row in rows]
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            complexities[index][0],
            complexities[index][1],
            hashlib.sha256(f"{seed}|{identities[index]}".encode()).hexdigest(),
        ),
    )
    best: tuple[int, int] | None = None
    best_map: list[int] | None = None
    for offset in range(1, len(rows)):
        mapping = [0] * len(rows)
        cost = 0
        for position, target_index in enumerate(order):
            donor_index = order[(position + offset) % len(order)]
            mapping[target_index] = donor_index
            left, right = complexities[target_index], complexities[donor_index]
            cost += 1000 * abs(left[0] - right[0]) + abs(left[1] - right[1])
        score = (cost, offset)
        if best is None or score < best:
            best, best_map = score, mapping
    assert best_map is not None
    return best_map


def _copy_graph_payload(target: dict, donor: dict) -> dict:
    graph_row = copy.deepcopy(target)
    # build_blocks may derive CFG inline from assembly, so the donor assembly is
    # part of the graph payload. The decoder prompt is built from the untouched
    # target row elsewhere, preventing prompt leakage from the donor.
    for key in ('assembly', 'cfg', 'edges', 'integrity', 'graph_v2'):
        if key in donor:
            graph_row[key] = copy.deepcopy(donor[key])
        else:
            graph_row.pop(key, None)
    return graph_row


def apply_graph_input_ablation(
    rows: list[dict],
    mode: str,
    seed: int,
) -> tuple[list[dict], list[dict], str]:
    """Build graph-channel inputs while leaving target prompts/tests untouched."""
    mode = mode.strip().lower()
    if mode not in {'none', 'null', 'cyclic_shift', 'matched_permutation', 'shuffle_blocks'}:
        raise ValueError(
            f"Unknown graph input ablation {mode!r}; use none, null, cyclic_shift, matched_permutation, or shuffle_blocks"
        )

    graph_rows: list[dict] = []
    records: list[dict] = []
    if mode in {'cyclic_shift', 'matched_permutation'} and len(rows) < 2:
        raise ValueError(f"{mode} requires at least two dataset rows")
    offset = 0 if mode != 'cyclic_shift' else 1 + (seed % (len(rows) - 1))
    matched_donors = _matched_derangement(rows, seed) if mode == 'matched_permutation' else None

    for index, row in enumerate(rows):
        target_id = _row_identity(row, index)
        if mode == 'none':
            graph_row = row
            record = {
                'mode': 'none',
                'target_id': target_id,
                'donor_id': target_id,
                'final_context_zeroed': False,
            }
        elif mode == 'null':
            graph_row = row
            record = {
                'mode': 'null',
                'target_id': target_id,
                'donor_id': target_id,
                'final_context_zeroed': True,
            }
        elif mode in {'cyclic_shift', 'matched_permutation'}:
            donor_index = (
                (index + offset) % len(rows)
                if mode == 'cyclic_shift'
                else int(matched_donors[index])
            )
            donor = rows[donor_index]
            graph_row = _copy_graph_payload(row, donor)
            target_complexity = _graph_complexity(row)
            donor_complexity = _graph_complexity(donor)
            record = {
                'mode': mode,
                'target_id': target_id,
                'donor_id': _row_identity(donor, donor_index),
                'donor_index': donor_index,
                'offset': offset if mode == 'cyclic_shift' else None,
                'target_block_count': target_complexity[0],
                'target_instruction_count': target_complexity[1],
                'donor_block_count': donor_complexity[0],
                'donor_instruction_count': donor_complexity[1],
            }
            if record['target_id'] == record['donor_id']:
                raise ValueError(f"{mode} produced an identity collision")
        else:
            graph_row, record = _shuffle_graph_blocks(row, seed, index)
        graph_rows.append(graph_row)
        records.append(record)

    mapping_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return graph_rows, records, mapping_sha


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
    parser.add_argument(
        '--decoder_revision',
        default=os.environ.get('GRAPH_DECODER_REVISION', ''),
        help='Immutable Hugging Face decoder revision/commit SHA.',
    )
    parser.add_argument(
        '--encoder_revision',
        default=os.environ.get('GRAPH_ENCODER_REVISION', ''),
        help='Immutable Hugging Face encoder revision/commit SHA.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=int(os.environ.get('GRAPH_SEED', '42')),
        help='Generation seed recorded in the provenance sidecar.',
    )
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
    parser.add_argument(
        '--graph_input_ablation',
        choices=['none', 'null', 'cyclic_shift', 'matched_permutation', 'shuffle_blocks'],
        default='none',
        help=(
            "Inference-only graph-channel control. null zeroes the final graph context "
            "while preserving prefix length/mask; cyclic_shift supplies each target "
            "with another task's graph; matched_permutation uses a shape-matched one-to-one derangement; shuffle_blocks changes canonical block order "
            "while remapping edges to preserve topology."
        ),
    )
    parser.add_argument(
        '--graph_ablation_seed',
        type=int,
        default=None,
        help='Deterministic graph-ablation seed; defaults to --seed.',
    )
    args = parser.parse_args()
    os.environ['GRAPH_DECODER_REVISION'] = args.decoder_revision
    os.environ['GRAPH_ENCODER_REVISION'] = args.encoder_revision
    os.environ['GRAPH_SEED'] = str(args.seed)
    set_seed(args.seed)

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

    encoder_revision = args.encoder_revision.strip() or None
    decoder_revision = args.decoder_revision.strip() or None
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_MODEL,
        revision=encoder_revision,
        trust_remote_code=True,
    )
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        args.decoder_model,
        revision=decoder_revision,
        trust_remote_code=True,
    )
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=512)
    dfg_extractor = LightweightDFGExtractor()

    model = GraphInferenceModel(args.decoder_model).to(device)
    checkpoint_load = None
    if args.checkpoint:
        # Load custom PyTorch state dict
        state = torch.load(args.checkpoint, map_location=device)
        
        # Select the load target by actual key overlap. Decoder-only adapters do
        # not necessarily contain a ``local_encoder`` key, so prefix heuristics
        # can silently choose the wrapper and reject every unprefixed tensor.
        decompiler_keys = set(model.decompiler.state_dict())
        wrapper_keys = set(model.state_dict())
        decompiler_overlap = len(set(state) & decompiler_keys)
        wrapper_overlap = len(set(state) & wrapper_keys)
        load_target = model.decompiler if decompiler_overlap >= wrapper_overlap else model
        if max(decompiler_overlap, wrapper_overlap) == 0:
            raise RuntimeError(
                "inference checkpoint has no tensors recognised by either the "
                "decompiler or wrapper architecture"
            )
        missing, unexpected = load_target.load_state_dict(state, strict=False)
        checkpoint_load = validate_trainable_checkpoint_load(
            load_target,
            state,
            missing_keys=missing,
            unexpected_keys=unexpected,
            context="inference checkpoint",
        )
        checkpoint_load['load_target'] = (
            'GraphCodeBERTT5Seq2Seq' if load_target is model.decompiler else 'GraphInferenceModel'
        )
        print(f'loaded checkpoint: {args.checkpoint}')
        print(
            'Validated checkpoint architecture: '
            f"recognised={checkpoint_load['recognised_checkpoint_tensor_count']} "
            f"missing_frozen={checkpoint_load['missing_frozen_tensor_count']} "
            f"unexpected={checkpoint_load['unexpected_tensor_count']}"
        )
    maybe_override_qwen_prefix_gate(model.decompiler)
    prefix_adapter = getattr(model.decompiler, 'qwen_prefix_adapter', None)
    prefix_gate_tensor = (
        getattr(prefix_adapter, 'gate_logit', None)
        if prefix_adapter is not None
        else None
    )
    prefix_gate_value = None
    if prefix_gate_tensor is not None:
        with torch.no_grad():
            prefix_gate_value = float(
                torch.sigmoid(prefix_gate_tensor.float()).mean().detach().cpu().item()
            )
    model.eval()

    # Training and inference share the wrapper/environment. Training may need
    # activation checkpointing, but generation needs the decoder KV cache;
    # inheriting GRAPH_GRADIENT_CHECKPOINTING=1 previously forced use_cache
    # off and made every autoregressive step recompute the full prefix.
    for decoder in (model.t5_model, model.decompiler.base_decoder_model):
        if hasattr(decoder, "gradient_checkpointing_disable"):
            decoder.gradient_checkpointing_disable()
        if hasattr(decoder, "config") and hasattr(decoder.config, "use_cache"):
            decoder.config.use_cache = True
    model.decompiler.gradient_checkpointing = False
    print("Inference mode: decoder gradient checkpointing OFF; KV cache ON")

    rows = load_jsonl(args.dataset)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    graph_ablation_seed = (
        args.seed if args.graph_ablation_seed is None else args.graph_ablation_seed
    )
    graph_rows, graph_ablation_records, graph_ablation_sha = apply_graph_input_ablation(
        rows,
        args.graph_input_ablation,
        graph_ablation_seed,
    )
    if args.graph_input_ablation != 'none':
        print(
            f"Graph input ablation: {args.graph_input_ablation} "
            f"(seed={graph_ablation_seed}, mapping_sha256={graph_ablation_sha})"
        )
    outputs = []
    prompt_digest = hashlib.sha256()

    progress = None
    row_iter = rows
    if tqdm is not None:
        progress = tqdm(rows, desc="generate", unit="row", dynamic_ncols=True)
        row_iter = progress
    log_line = progress.write if progress is not None else print

    for idx, row in enumerate(row_iter):
        block_tensors, graph_data = build_blocks(
            graph_rows[idx],
            tensor_builder,
            dfg_extractor,
        )
        prompt_tensors = None
        if model.is_causal:
            prompt_text = build_decoder_prompt(
                row,
                decoder_tokenizer,
                args.decoder_prompt_max_length,
            )
            if "Unit-test harness excerpt" in prompt_text or "ORACLE DIAGNOSTIC" in prompt_text:
                raise RuntimeError("Scoring-test content reached the evaluation policy prompt")
            prompt_digest.update(
                json.dumps(
                    [str(row.get('task_id', idx)), prompt_text],
                    ensure_ascii=False,
                    separators=(',', ':'),
                ).encode('utf-8')
            )
            prompt_tensors = decoder_tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )
            prompt_tokens = int(prompt_tensors["input_ids"].size(1))
            if prompt_tokens > args.decoder_prompt_max_length:
                raise RuntimeError(
                    f"row {idx + 1} prompt needs {prompt_tokens} tokens but "
                    f"--decoder_prompt_max_length={args.decoder_prompt_max_length}; "
                    "refusing silent inference truncation"
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
                force_null_graph=args.graph_input_ablation == 'null',
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
            'graph_input_ablation': graph_ablation_records[idx],
        })

        log_line(f'[{idx + 1}/{len(rows)}] generated {len(predictions)} candidates')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    decoder_config = model.decompiler.base_decoder_model.config
    encoder_config = model.decompiler.local_encoder.encoder.config
    source_paths = [
        Path(__file__),
        ROOT / 'scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py',
        ROOT / 'models/hierarchical_graph_encoder_antigravity.py',
        ROOT / 'models/graphcodebert_tensor_builder.py',
        ROOT / 'models/pyg_cfg_dataset.py',
        ROOT / 'scripts/data/cfg_extractor.py',
        ROOT / 'scripts/data/dfg_extractor.py',
    ]
    provenance = {
        'schema_version': 1,
        'prompt_schema_version': PROMPT_SCHEMA_VERSION,
        'scoring_tests_visible_to_policy': False,
        'prompt_stream_sha256': prompt_digest.hexdigest(),
        'row_count': len(outputs),
        'seed': args.seed,
        'generation': {
            'num_samples': args.num_samples,
            'generation_batch_size': args.generation_batch_size,
            'max_new_tokens': args.max_new_tokens,
            'decoder_prompt_max_length': args.decoder_prompt_max_length,
            'use_cache': True,
            'decoder_gradient_checkpointing': False,
        },
        'graph_input_ablation': {
            'mode': args.graph_input_ablation,
            'seed': graph_ablation_seed,
            'mapping_sha256': graph_ablation_sha,
            'self_mapped_rows': sum(
                record['target_id'] == record['donor_id']
                for record in graph_ablation_records
            ),
            'final_context_zeroed': args.graph_input_ablation == 'null',
        },
        'models': {
            'decoder': {
                'requested_id': args.decoder_model,
                'requested_revision': decoder_revision,
                'resolved_name_or_path': getattr(decoder_config, '_name_or_path', None),
                'resolved_commit': model_commit(decoder_config),
            },
            'encoder': {
                'requested_id': ENCODER_MODEL,
                'requested_revision': encoder_revision,
                'resolved_name_or_path': getattr(encoder_config, '_name_or_path', None),
                'resolved_commit': model_commit(encoder_config),
            },
        },
        'dataset': file_record(args.dataset),
        'checkpoint': file_record(args.checkpoint, required=False) if args.checkpoint else None,
        'checkpoint_load': checkpoint_load,
        'graph_prefix_gate': {
            'mean_sigmoid': prefix_gate_value,
            'override_requested': os.environ.get(
                'GRAPH_QWEN_PREFIX_GATE_OVERRIDE', ''
            ).strip() or None,
        },
        'output': file_record(out_path),
        'source_files': [file_record(path) for path in source_paths],
        'git': git_state(ROOT),
        'graph_environment': graph_environment(),
        'runtime': runtime_record(),
    }
    provenance_path = Path(str(out_path) + '.provenance.json')
    write_json(provenance_path, provenance)
    print(f'wrote {args.output}')
    print(f'wrote {provenance_path}')


if __name__ == '__main__':
    main()
