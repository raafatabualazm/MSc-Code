"""
Graph-aware encoder-decoder training pipeline for neural decompilation (Antigravity version).
Supports T5/CodeT5 and Qwen (Causal LM) decoders, and PEFT (LoRA/DoRA) configuration.
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

import json
import re
import os
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import Dataset
from torch_geometric.data import Batch
from models.pyg_cfg_dataset import cfg_to_pyg
from models.hierarchical_graph_encoder_antigravity import LocalBlockEncoder, GraphPoolingEncoder
from scripts.data.dfg_extractor import LightweightDFGExtractor
from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from models.graph_data_collator import GraphDataCollator
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ENCODER_MODEL = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")
DECODER_MODEL = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")


@dataclass
class GraphDecompilerConfig:
    train_file: str = "data/datasets/dart_all.jsonl"
    eval_file: str = "data/datasets/test-set.jsonl"
    output_dir: str = os.environ.get("GRAPH_OUTPUT_DIR", "artifacts/graph-decompiler-v1")
    max_input_length: int = 512  # GraphCodeBERT max block length
    max_target_length: int = 768
    learning_rate: float = float(os.environ.get("GRAPH_LR", "5e-6"))
    batch_size: int = int(os.environ.get("GRAPH_BATCH_SIZE", "2"))  # Support custom batch size!
    epochs: int = int(os.environ.get("GRAPH_EPOCHS", "1"))
    use_reasoning: bool = False
    inject_cfg_tags: bool = True
    use_graph_encoder: bool = True


def canonicalize_source(source: str) -> str:
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"\s+", " ", source)

    variable_map: Dict[str, str] = {}
    variable_counter = 0

    def normalize_identifier(match):
        nonlocal variable_counter
        token = match.group(0)

        reserved = {
            "if", "else", "while", "for", "return",
            "class", "void", "int", "double",
            "String", "bool"
        }

        if token in reserved:
            return token

        if token not in variable_map:
            variable_map[token] = f"var_{variable_counter}"
            variable_counter += 1

        return variable_map[token]

    source = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", normalize_identifier, source)
    return source.strip()


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "assembly" in record and "source" in record:
                rows.append(record)
    return rows


def build_dataset(config: GraphDecompilerConfig):
    # Simply load raw records. Formatting and tokenization happen in tokenize_dataset.
    return (
        load_jsonl(config.train_file),
        load_jsonl(config.eval_file),
    )


def tokenize_dataset(records, tensor_builder: GraphCodeBERTTensorBuilder, dfg_extractor, decoder_tokenizer: PreTrainedTokenizerBase, config):
    converted = []

    for idx, record in enumerate(records):
        cfg_blocks = record.get('cfg', [])
        edges = record.get('edges', [])

        if not cfg_blocks:
            assembly_lines = record['assembly'].splitlines()
            cfg_blocks = [{
                'block_id': 'entry',
                'block_type': 'entry',
                'instructions': assembly_lines,
            }]

        block_tensors = []
        for block in cfg_blocks:
            dfg = dfg_extractor.extract_block_dfg_structured(block['instructions'])
            tensors = tensor_builder.build_block_tensors(
                block['instructions'],
                dfg,
            )
            block_tensors.append({
                'input_ids': tensors['input_ids'].tolist(),
                'position_ids': tensors['position_ids'].tolist(),
                'token_type_ids': tensors['token_type_ids'].tolist(),
                'attention_mask': tensors['attention_mask'].tolist(),
            })

        target_text = canonicalize_source(record["source"])
        labels = decoder_tokenizer(
            target_text,
            max_length=config.max_target_length,
            truncation=True,
            padding="max_length",
        )

        converted.append({
            "block_inputs": block_tensors,
            "labels": labels["input_ids"],
            "cfg": cfg_blocks,
            "edges": edges,
        })

        if (idx + 1) % 500 == 0:
            print(f"Tokenized {idx + 1}/{len(records)} examples")

    return Dataset.from_list(converted)


class GraphCodeBERTT5Seq2Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.local_encoder = LocalBlockEncoder()
        self.graph_encoder = GraphPoolingEncoder()

        # Check if decoder is causal
        # Qwen has standard causal LM architecture
        decoder_model_name = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")
        decoder_peft = os.environ.get("GRAPH_DECODER_PEFT", "none").lower()
        
        # Check if we should load the decoder in 4-bit to fit in 6GB VRAM
        is_large = "770m" in decoder_model_name.lower() or "2b" in decoder_model_name.lower()
        load_in_4bit = is_large and (decoder_peft in ["lora", "dora"])
        
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = quantization_config
            kwargs["device_map"] = "auto"
            print(f"Loading decoder {decoder_model_name} with BitsAndBytesConfig NF4 for VRAM efficiency...")

        if "codet5" in decoder_model_name.lower():
            kwargs["use_safetensors"] = False

        is_qwen = "qwen" in decoder_model_name.lower()
        if is_qwen:
            self.t5_model = AutoModelForCausalLM.from_pretrained(decoder_model_name, trust_remote_code=True, **kwargs)
            self.is_causal = True
        else:
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(decoder_model_name, trust_remote_code=True, **kwargs)
            self.is_causal = False

        self.encoder_dim = self.local_encoder.encoder.config.hidden_size
        if hasattr(self.t5_model.config, "decoder") and self.t5_model.config.decoder is not None:
            decoder_cfg = self.t5_model.config.decoder
            if hasattr(decoder_cfg, "hidden_size"):
                self.decoder_dim = decoder_cfg.hidden_size
            elif hasattr(decoder_cfg, "d_model"):
                self.decoder_dim = decoder_cfg.d_model
            elif hasattr(decoder_cfg, "n_embd"):
                self.decoder_dim = decoder_cfg.n_embd
            else:
                raise ValueError("Could not determine decoder hidden dimension from config.decoder")
        elif hasattr(self.t5_model.config, "hidden_size"):
            self.decoder_dim = self.t5_model.config.hidden_size
        elif hasattr(self.t5_model.config, "d_model"):
            self.decoder_dim = self.t5_model.config.d_model
        elif hasattr(self.t5_model.config, "n_embd"):
            self.decoder_dim = self.t5_model.config.n_embd
        else:
            raise ValueError("Could not determine decoder hidden dimension from config")

        # Upgrade to LayerNorm + MLP bridge for robust alignment
        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.encoder_dim),
            torch.nn.Linear(self.encoder_dim, self.decoder_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        )

        # Setup PEFT & Freezing Configs
        encoder_peft = os.environ.get("GRAPH_ENCODER_PEFT", "none").lower()
        freeze_encoder = os.environ.get("GRAPH_FREEZE_ENCODER", "1") == "1"
        freeze_decoder = os.environ.get("GRAPH_FREEZE_DECODER", "0") == "1"

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        lora_r = int(os.environ.get("GRAPH_LORA_R", "16"))
        lora_alpha = int(os.environ.get("GRAPH_LORA_ALPHA", "32"))

        # Apply PEFT/Freezing to Local Encoder
        if encoder_peft in ["lora", "dora"]:
            use_dora = (encoder_peft == "dora")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
                use_dora=use_dora,
            )
            self.local_encoder.encoder = get_peft_model(self.local_encoder.encoder, lora_config)
            print(f"Applied PEFT {encoder_peft.upper()} (r={lora_r}, alpha={lora_alpha}) to LocalBlockEncoder")
        elif freeze_encoder:
            for param in self.local_encoder.parameters():
                param.requires_grad = False
            print("Froze LocalBlockEncoder weights")

        # Apply PEFT/Freezing to Decoder
        if decoder_peft in ["lora", "dora"]:
            use_dora = (decoder_peft == "dora")
            if "codet5p-2b" in decoder_model_name.lower():
                target_modules = ["qkv_proj", "q_attn"]
            else:
                target_modules = ["q_proj", "v_proj"] if self.is_causal else ["q", "v"]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                use_dora=use_dora,
            )
            if load_in_4bit:
                self.t5_model = prepare_model_for_kbit_training(self.t5_model, use_gradient_checkpointing=False)
            self.t5_model = get_peft_model(self.t5_model, lora_config)
            print(f"Applied PEFT {decoder_peft.upper()} (r={lora_r}, alpha={lora_alpha}) to Decoder")
        elif freeze_decoder:
            for param in self.t5_model.parameters():
                param.requires_grad = False
            print("Froze Decoder weights")

        # Fix Salesforce modeling_codet5p meta device scale_attn bug by re-creating it
        for name, module in self.t5_model.named_modules():
            if module.__class__.__name__ == "CodeT5pAttention":
                if hasattr(module, "scale_attn") and isinstance(module.scale_attn, torch.Tensor):
                    device = next(module.parameters()).device if list(module.parameters()) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    head_dim = getattr(module, "head_dim", 32)
                    module.scale_attn = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=device)).to(dtype=self.t5_model.dtype)
                    print(f"Re-created scale_attn in {name} on device: {device}")

        # Patch self.t5_model.enc_to_dec_proj to Identity if it exists to avoid shape mismatch (e.g. codet5p-2b)
        base_model = self.base_decoder_model
        if hasattr(base_model, "enc_to_dec_proj"):
            base_model.enc_to_dec_proj = torch.nn.Identity()
            print("Successfully patched self.t5_model.enc_to_dec_proj dynamically to torch.nn.Identity()!")

    @property
    def base_decoder_model(self):
        from peft import PeftModel
        if isinstance(self.t5_model, PeftModel):
            return self.t5_model.base_model.model
        return self.t5_model

    @property
    def decoder(self):
        if hasattr(self.base_decoder_model, 'decoder'):
            return self.base_decoder_model.decoder
        return None

    @property
    def lm_head(self):
        if hasattr(self.base_decoder_model, 'lm_head'):
            return self.base_decoder_model.lm_head
        return None

    def forward(self, input_ids=None, attention_mask=None, labels=None, decoder_input_ids=None, cfg=None, edges=None, block_inputs=None, **kwargs):
        block_embeddings_batch = []
        list_of_B_i = []

        # 1. Forward pass through Local Block Encoder
        if block_inputs is not None:
            for block_group in block_inputs:
                if not block_group:
                    list_of_B_i.append(1)
                    block_embeddings_batch.append(torch.zeros((1, 768), device=labels.device if labels is not None else "cuda"))
                    continue

                group_input_ids = torch.stack([
                    torch.tensor(b['input_ids'], dtype=torch.long, device=labels.device if labels is not None else "cuda")
                    for b in block_group
                ])
                group_attention_mask = torch.stack([
                    torch.tensor(b['attention_mask'], dtype=torch.float, device=labels.device if labels is not None else "cuda").squeeze(0)
                    for b in block_group
                ])
                group_position_ids = torch.stack([
                    torch.tensor(b['position_ids'], dtype=torch.long, device=labels.device if labels is not None else "cuda")
                    for b in block_group
                ])
                group_token_type_ids = torch.stack([
                    torch.tensor(b['token_type_ids'], dtype=torch.long, device=labels.device if labels is not None else "cuda")
                    for b in block_group
                ])

                block_embeddings = self.local_encoder(
                    group_input_ids,
                    group_attention_mask,
                    group_position_ids,
                    group_token_type_ids,
                )

                block_embeddings_batch.append(block_embeddings)
                list_of_B_i.append(block_embeddings.size(0))

        # 2. Construct PyG graphs for CFG GNN
        edge_index = None
        edge_attr = None

        if cfg is not None and edges is not None:
            try:
                batch_graphs = []
                for batch_index in range(len(cfg)):
                    node_embeddings = block_embeddings_batch[batch_index]
                    graph_record = {
                        'cfg': cfg[batch_index],
                        'edges': edges[batch_index],
                    }
                    batch_graphs.append(
                        cfg_to_pyg(graph_record, node_embeddings)
                    )

                pyg_batch = Batch.from_data_list(batch_graphs)
                edge_index = pyg_batch.edge_index
                edge_attr = pyg_batch.edge_attr
            except Exception:
                edge_index = None
                edge_attr = None

        # 3. Concatenate node features and update with GNN and padding sequences
        # Run GNN in Float32 to ensure numerical stability and prevent CUBLAS internal errors in half-precision
        with torch.cuda.amp.autocast(enabled=False):
            graph_inputs = torch.cat(block_embeddings_batch, dim=0).float()

            encoder_hidden_states, encoder_attention_mask = self.graph_encoder(
                graph_inputs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                list_of_B_i=list_of_B_i,
            )

            encoder_hidden_states = self.projection(encoder_hidden_states)

        encoder_hidden_states = encoder_hidden_states.to(dtype=self.t5_model.dtype)

        # 4. Forward through decoder
        if self.is_causal:
            # Causal Prefix Injection
            if labels is not None and decoder_input_ids is None:
                decoder_input_ids = labels

            decoder_embeds = self.base_decoder_model.get_input_embeddings()(decoder_input_ids)

            inputs_embeds = torch.cat([encoder_hidden_states, decoder_embeds], dim=1)
            inputs_embeds = inputs_embeds.to(dtype=self.base_decoder_model.dtype)

            prefix_len = encoder_hidden_states.size(1)
            target_len = decoder_input_ids.size(1)

            combined_mask = torch.cat([
                encoder_attention_mask,
                torch.ones((encoder_attention_mask.size(0), target_len), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)
            ], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
            )
            logits = outputs.logits
            target_logits = logits[:, prefix_len:, :]

            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    target_logits.reshape(-1, target_logits.size(-1)),
                    labels.view(-1),
                )

            return {"loss": loss, "logits": target_logits}

        else:
            # Seq2Seq Decoder
            if labels is not None and decoder_input_ids is None:
                if hasattr(self.base_decoder_model, '_shift_right'):
                    decoder_input_ids = self.base_decoder_model._shift_right(labels)
                else:
                    cfg = self.base_decoder_model.config
                    decoder_start_token_id = getattr(cfg, 'decoder_start_token_id', None)
                    if decoder_start_token_id is None and hasattr(cfg, 'decoder'):
                        decoder_start_token_id = getattr(cfg.decoder, 'decoder_start_token_id', None)
                    if decoder_start_token_id is None:
                        decoder_start_token_id = getattr(cfg, 'pad_token_id', 0)
                        
                    pad_token_id = getattr(cfg, 'pad_token_id', 0)
                    if pad_token_id is None and hasattr(cfg, 'decoder'):
                        pad_token_id = getattr(cfg.decoder, 'pad_token_id', 0)
                    if pad_token_id is None:
                        pad_token_id = 0
                        
                    shifted_input_ids = labels.new_zeros(labels.shape)
                    shifted_input_ids[..., 1:] = labels[..., :-1].clone()
                    shifted_input_ids[..., 0] = decoder_start_token_id
                    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
                    decoder_input_ids = shifted_input_ids

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

            if hasattr(decoder_outputs, 'logits') and decoder_outputs.logits is not None:
                logits = decoder_outputs.logits
            else:
                sequence_output = decoder_outputs.last_hidden_state
                sequence_output = sequence_output * (self.decoder_dim ** -0.5)
                logits = self.lm_head(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            return {"loss": loss, "logits": logits}


def initialize_model(tokenizer):
    return GraphCodeBERTT5Seq2Seq()


def main():
    config = GraphDecompilerConfig()
    dfg_extractor = LightweightDFGExtractor()

    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, trust_remote_code=True)
    decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL, trust_remote_code=True)
    
    # Configure pad token if it's missing (e.g. Qwen/Causal LMs)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=config.max_input_length)

    print("Loading datasets...")
    train_records, eval_records = build_dataset(config)

    print("Tokenizing datasets (using unified tensor builder format)...")
    train_dataset = tokenize_dataset(train_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)
    eval_dataset = tokenize_dataset(eval_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)

class AntigravitySeq2SeqTrainer(Seq2SeqTrainer):
    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"AntigravitySeq2SeqTrainer: Saving model state dict to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        # Save only trainable parameters (PEFT adapters, GNN, projection layers) to reduce size and avoid 4-bit serialization issues
        trainable_keys = {name for name, param in self.model.named_parameters() if param.requires_grad}
        trainable_state_dict = {k: v for k, v in state_dict.items() if k in trainable_keys}

        torch.save(trainable_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        if (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    config = GraphDecompilerConfig()
    dfg_extractor = LightweightDFGExtractor()

    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, trust_remote_code=True)
    decoder_model_name = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name, trust_remote_code=True)
    
    # Configure pad token if it's missing (e.g. Qwen/Causal LMs)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=config.max_input_length)

    print("Loading datasets...")
    train_records, eval_records = build_dataset(config)

    print("Tokenizing datasets (using unified tensor builder format)...")
    train_dataset = tokenize_dataset(train_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)
    eval_dataset = tokenize_dataset(eval_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)

    model = initialize_model(encoder_tokenizer)

    max_steps = int(os.environ.get("GRAPH_MAX_STEPS", -1))

    grad_accum = int(os.environ.get("GRAPH_GRAD_ACCUM", "1"))
    print(f"Gradient accumulation steps: {grad_accum}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs if max_steps < 0 else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=grad_accum,
        predict_with_generate=True,
        eval_strategy="steps" if max_steps > 0 else "epoch",
        eval_steps=max_steps if max_steps > 0 else None,
        save_strategy="no",
        logging_steps=1 if max_steps > 0 else 10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,  # VERY IMPORTANT to keep custom graph columns!
    )


    data_collator = GraphDataCollator(decoder_tokenizer)

    trainer = AntigravitySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print(f"Saving model to {config.output_dir}...")
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
