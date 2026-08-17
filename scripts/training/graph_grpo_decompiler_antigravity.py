"""
Graph-aware GRPO (Group Relative Policy Optimization) training script for neural decompilation (Antigravity version).
Optimized for low VRAM using PEFT adapter-disabling context to get reference model logits without parameter duplication.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast
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
except Exception as e:
    pass

import os
import re
import json
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from torch_geometric.data import Batch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)

# Insert root to system path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.pyg_cfg_dataset import cfg_to_pyg
from models.hierarchical_graph_encoder_antigravity import LocalBlockEncoder, GraphPoolingEncoder
from scripts.data.dfg_extractor import LightweightDFGExtractor
from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
    GraphDecompilerConfig,
    GraphCodeBERTT5Seq2Seq,
    canonicalize_source,
    build_dataset,
    tokenize_dataset,
)

ENCODER_MODEL = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")


def compute_overlap_reward(reference: str, completion: str) -> float:
    ref_tokens = set(reference.lower().split())
    comp_tokens = set(completion.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = len(ref_tokens & comp_tokens) / len(ref_tokens)
    return overlap

def compile_dart(code: str) -> bool:
    if not shutil.which("dart"):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        try:
            path.write_text(code, encoding='utf-8')
            result = subprocess.run(
                ['dart', 'analyze', str(path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

def calculate_rewards(completions: List[str], references: List[str], languages: List[str]) -> torch.Tensor:
    rewards = []
    for comp, ref, lang in zip(completions, references, languages):
        comp_strip = comp.strip()
        # Heavily penalize empty or tiny outputs
        if not comp_strip or len(comp_strip) < 10:
            rewards.append(-1.0)
            continue
            
        overlap = compute_overlap_reward(ref, comp)
        reward_score = overlap
        
        # Add compilation reward if language is dart
        if lang.lower() == 'dart':
            is_compilable = compile_dart(comp)
            if is_compilable:
                reward_score += 0.5  # Compile bonus!
            else:
                reward_score -= 0.2  # Compile penalty!
                
        rewards.append(reward_score)
    return torch.tensor(rewards, dtype=torch.float32)

class GRPOTrainer:
    def __init__(self, model: GraphCodeBERTT5Seq2Seq, tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        
        # Set up optimizer only on trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        
    def train_step(self, batch: Dict[str, Any], device: str) -> float:
        self.model.eval() # Generation uses eval mode
        
        cfg = batch.get('cfg')
        edges = batch.get('edges')
        block_inputs = batch.get('block_inputs')
        labels = batch.get('labels')
        
        # 1. Forward pass through Graph Encoder
        # Since we generate multiple completions per example, we want to run the GNN once and then replicate
        with torch.no_grad():
            block_embeddings_batch = []
            list_of_B_i = []
            
            for block_group in block_inputs:
                if not block_group:
                    list_of_B_i.append(1)
                    block_embeddings_batch.append(torch.zeros((1, 768), device=device))
                    continue
                
                group_input_ids = torch.stack([
                    torch.tensor(b['input_ids'], dtype=torch.long, device=device)
                    for b in block_group
                ])
                group_attention_mask = torch.stack([
                    torch.tensor(b['attention_mask'], dtype=torch.float, device=device).squeeze(0)
                    for b in block_group
                ])
                group_position_ids = torch.stack([
                    torch.tensor(b['position_ids'], dtype=torch.long, device=device)
                    for b in block_group
                ])
                group_token_type_ids = torch.stack([
                    torch.tensor(b['token_type_ids'], dtype=torch.long, device=device)
                    for b in block_group
                ])

                block_embeddings = self.model.local_encoder(
                    group_input_ids,
                    group_attention_mask,
                    group_position_ids,
                    group_token_type_ids,
                )
                block_embeddings_batch.append(block_embeddings)
                list_of_B_i.append(block_embeddings.size(0))

            edge_index = None
            edge_attr = None
            if cfg is not None and edges is not None:
                try:
                    batch_graphs = []
                    for batch_index in range(len(cfg)):
                        node_embeddings = block_embeddings_batch[batch_index]
                        graph_record = {'cfg': cfg[batch_index], 'edges': edges[batch_index]}
                        batch_graphs.append(cfg_to_pyg(graph_record, node_embeddings))
                    pyg_batch = Batch.from_data_list(batch_graphs)
                    edge_index = pyg_batch.edge_index
                    edge_attr = pyg_batch.edge_attr
                except Exception:
                    pass

            with torch.cuda.amp.autocast(enabled=False):
                graph_inputs = torch.cat(block_embeddings_batch, dim=0).float()
                encoder_hidden_states, encoder_attention_mask = self.model.graph_encoder(
                    graph_inputs,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    list_of_B_i=list_of_B_i,
                )
                encoder_hidden_states = self.model.projection(encoder_hidden_states)
        
        # Repeat the hidden states G times for Group Sampling
        B, max_blocks, d_model = encoder_hidden_states.shape
        G = self.args.group_size
        
        # Reshape to group dimension
        # (B * G, max_blocks, d_model)
        grp_hidden_states = encoder_hidden_states.repeat_interleave(G, dim=0)
        grp_attention_mask = encoder_attention_mask.repeat_interleave(G, dim=0)
        
        # 2. Sample completions from the policy model
        base_model = self.model.base_decoder_model
        
        if self.model.is_causal:
            start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
            input_ids = torch.tensor([[start_token_id]], device=device).expand(B * G, -1)
            
            input_embeds = base_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([grp_hidden_states, input_embeds], dim=1)
            inputs_embeds = inputs_embeds.to(dtype=base_model.dtype)
            
            combined_mask = torch.cat([
                grp_attention_mask,
                torch.ones((grp_attention_mask.size(0), 1), dtype=grp_attention_mask.dtype, device=device)
            ], dim=1)
            
            with torch.no_grad():
                outputs = self.model.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=combined_mask,
                    max_new_tokens=128, # Safe limit for training speed
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        else:
            # Seq2Seq
            start_token_id = self.model.t5_model.config.decoder_start_token_id or self.tokenizer.pad_token_id
            input_ids = torch.tensor([[start_token_id]], device=device).expand(B * G, -1)
            
            from transformers.modeling_outputs import BaseModelOutput
            with torch.no_grad():
                outputs = self.model.t5_model.generate(
                    decoder_input_ids=input_ids,
                    encoder_outputs=BaseModelOutput(last_hidden_state=grp_hidden_states),
                    attention_mask=grp_attention_mask,
                    max_new_tokens=128, # Safe limit for training speed
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        
        # 3. Calculate rewards for the generated sequences
        # Decode reference texts
        ref_texts = []
        languages = []
        for i in range(B):
            ref_id = labels[i]
            # Replace pad/ignore index
            ref_id_clean = [tok for tok in ref_id if tok != -100]
            ref_texts.append(self.tokenizer.decode(ref_id_clean, skip_special_tokens=True))
            languages.append(batch.get('language', ['dart'])[i])
            
        completions = []
        for out_seq in outputs:
            completions.append(self.tokenizer.decode(out_seq, skip_special_tokens=True))
            
        # Replicate references G times to match completions batching
        repeated_refs = [ref for ref in ref_texts for _ in range(G)]
        repeated_langs = [lang for lang in languages for _ in range(G)]
        
        rewards = calculate_rewards(completions, repeated_refs, repeated_langs)
        rewards = rewards.to(device)
        
        # 4. Group Normalization of Rewards
        rewards_reshaped = rewards.view(B, G)
        mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
        std_rewards = rewards_reshaped.std(dim=1, keepdim=True).clamp(min=1e-5)
        advantages = ((rewards_reshaped - mean_rewards) / std_rewards).view(-1)
        
        # 5. Calculate policy log-probs & reference log-probs
        self.model.train() # Switch back to train mode for backpropagation
        self.optimizer.zero_grad()
        
        from peft import PeftModel
        
        if self.model.is_causal:
            target_ids = outputs[:, 1:]
            target_len = target_ids.size(1)
            prefix_len = grp_hidden_states.size(1)
            
            # Re-compute embeds of inputs (all tokens except the last one)
            input_ids = outputs[:, :-1]
            target_embeds = base_model.get_input_embeddings()(input_ids)
            combined_inputs_embeds = torch.cat([grp_hidden_states, target_embeds], dim=1)
            combined_inputs_embeds = combined_inputs_embeds.to(dtype=base_model.dtype)
            
            forward_mask = torch.cat([
                grp_attention_mask,
                torch.ones((grp_attention_mask.size(0), input_ids.size(1)), dtype=grp_attention_mask.dtype, device=device)
            ], dim=1)
            
            policy_outputs = self.model.t5_model(
                inputs_embeds=combined_inputs_embeds,
                attention_mask=forward_mask,
            )
            policy_logits = policy_outputs.logits[:, prefix_len:, :]
            
            # reference policy pass
            if isinstance(self.model.t5_model, PeftModel):
                with self.model.t5_model.disable_adapter():
                    ref_outputs = self.model.t5_model(
                        inputs_embeds=combined_inputs_embeds,
                        attention_mask=forward_mask,
                    )
                    ref_logits = ref_outputs.logits[:, prefix_len:, :]
            else:
                ref_logits = None
        else:
            # Seq2Seq
            target_ids = outputs
            target_len = target_ids.size(1)
            decoder_input_ids = self.model.base_decoder_model._shift_right(target_ids)
            
            policy_outputs = self.model.t5_model(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=(grp_hidden_states,),
                attention_mask=grp_attention_mask,
            )
            policy_logits = policy_outputs.logits
            
            # reference policy pass
            if isinstance(self.model.t5_model, PeftModel):
                with self.model.t5_model.disable_adapter():
                    ref_outputs = self.model.t5_model(
                        decoder_input_ids=decoder_input_ids,
                        encoder_outputs=(grp_hidden_states,),
                        attention_mask=grp_attention_mask,
                    )
                    ref_logits = ref_outputs.logits
            else:
                ref_logits = None
                
        # Calculate log probabilities
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        token_log_probs = torch.gather(policy_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        if ref_logits is not None:
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            token_ref_log_probs = torch.gather(ref_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        else:
            token_ref_log_probs = token_log_probs.detach()
            
        # Create mask for valid tokens (up to first EOS)
        eos_token_id = self.tokenizer.eos_token_id
        is_eos = (target_ids == eos_token_id)
        cum_eos = torch.cumsum(is_eos.to(torch.int32), dim=1)
        valid_mask = (cum_eos == 0) | ((cum_eos == 1) & is_eos)
        valid_mask = valid_mask.to(device)
        
        # 6. Compute GRPO loss
        ratio = torch.exp(token_log_probs - token_ref_log_probs)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2)
        
        # KL penalty: KL(ref || policy)
        kl = torch.exp(token_ref_log_probs - token_log_probs) - (token_ref_log_probs - token_log_probs) - 1.0
        
        # Combine losses
        loss_matrix = (policy_loss + self.args.kl_coef * kl) * valid_mask.float()
        loss = loss_matrix.sum() / valid_mask.float().sum().clamp(min=1.0)
        
        # 7. Backward pass
        loss.backward()
        # Gradient clipping for RL stability
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_size', type=int, default=4)
    parser.add_argument('--kl_coef', type=float, default=0.01)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=int(os.environ.get("GRAPH_BATCH_SIZE", "2")))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=int(os.environ.get("GRAPH_MAX_STEPS", -1)))
    parser.add_argument('--output_dir', default=os.environ.get("GRAPH_OUTPUT_DIR", "artifacts/qwen-grpo"))
    parser.add_argument('--checkpoint', default=os.environ.get("GRAPH_CHECKPOINT", ""))
    args = parser.parse_args()

    config = GraphDecompilerConfig()
    # If output_dir was set in environment, prioritize it
    config.output_dir = args.output_dir
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    
    dfg_extractor = LightweightDFGExtractor()

    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, trust_remote_code=True)
    decoder_model_name = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name, trust_remote_code=True)
    
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=config.max_input_length)

    print("Loading datasets...")
    train_records, eval_records = build_dataset(config)

    # Re-tokenize datasets with standard token types
    print("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)
    
    # Custom collate fn for data loader to keep dictionaries intact
    def grpo_collate(batch_list):
        collated = {
            'labels': torch.stack([torch.tensor(x['labels'], dtype=torch.long) for x in batch_list]),
            'block_inputs': [x['block_inputs'] for x in batch_list],
            'cfg': [x['cfg'] for x in batch_list],
            'edges': [x['edges'] for x in batch_list],
            'language': [x.get('language', 'dart') for x in batch_list],
        }
        return collated

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=grpo_collate
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GraphCodeBERTT5Seq2Seq().to(device)
    if args.checkpoint:
        print(f"Loading SFT checkpoint from: {args.checkpoint}")
        try:
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded SFT checkpoint weights.")
        except Exception as e:
            print(f"Warning: Failed to load SFT checkpoint: {e}")

    trainer = GRPOTrainer(model, decoder_tokenizer, args)
    
    max_steps = args.max_steps
    step_count = 0
    print("Starting GRPO Reinforcement Learning loop...")
    
    for epoch in range(args.epochs):
        for idx, batch in enumerate(train_loader):
            # Move labels to device
            batch['labels'] = batch['labels'].to(device)
            
            try:
                loss_val = trainer.train_step(batch, device)
                print(f"Epoch {epoch+1} | Step {step_count+1} | Batch {idx+1}/{len(train_loader)} | Loss: {loss_val:.4f}")
            except Exception as e:
                print(f"Error at step {step_count+1}: {e}")
                
            step_count += 1
            if max_steps > 0 and step_count >= max_steps:
                break
        if max_steps > 0 and step_count >= max_steps:
            break

    print(f"Saving GRPO-tuned model to {config.output_dir}...")
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save standard PyTorch weights via state dict filtering
    state_dict = model.state_dict()
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    trainable_state_dict = {k: v for k, v in state_dict.items() if k in trainable_keys}

    torch.save(trainable_state_dict, os.path.join(config.output_dir, "pytorch_model.bin"))
    decoder_tokenizer.save_pretrained(config.output_dir)
    print("GRPO training completed successfully.")

if __name__ == '__main__':
    main()
