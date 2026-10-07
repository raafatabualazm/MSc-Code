"""
Graph-aware encoder-decoder training pipeline for neural decompilation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import Dataset
from torch_geometric.data import Batch
from models.pyg_cfg_dataset import cfg_to_pyg
from models.hierarchical_graph_encoder import LocalBlockEncoder, GraphPoolingEncoder
from scripts.data.dfg_extractor import LightweightDFGExtractor
from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from models.graph_data_collator import GraphDataCollator
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

ENCODER_MODEL = "microsoft/graphcodebert-base"
DECODER_MODEL = "Salesforce/codet5-small"


@dataclass
class GraphDecompilerConfig:
    train_file: str = "data/matched/all_dart_matched.jsonl"
    eval_file: str = "data/datasets/test-set.jsonl"
    output_dir: str = "artifacts/graph-decompiler-v1"
    max_input_length: int = 2048
    max_target_length: int = 768
    learning_rate: float = 2e-5
    batch_size: int = 1
    epochs: int = 3
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


def build_encoder_input(record: dict, use_reasoning: bool, inject_cfg_tags: bool) -> str:
    assembly = record["assembly"]
    reasoning = record.get("reasoning", "")

    pseudo_cfg = []

    for line in assembly.splitlines():
        lowered = line.lower()

        if "jmp" in lowered or "br" in lowered:
            pseudo_cfg.append("[branch]")

        if "call" in lowered:
            pseudo_cfg.append("[call]")

        if "ret" in lowered:
            pseudo_cfg.append("[return]")

    sections = ["<asm>", assembly, "</asm>"]

    if inject_cfg_tags and pseudo_cfg:
        sections.extend([
            "<cfg>",
            " ".join(sorted(set(pseudo_cfg))),
            "</cfg>",
        ])

    if use_reasoning and reasoning:
        sections.extend([
            "<reasoning>",
            reasoning[:1200],
            "</reasoning>",
        ])

    return "\n".join(sections)


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
    def convert(records):
        converted = []

        for record in records:
            cfg_blocks = record.get('cfg', [])
            edges = record.get('edges', [])

            block_texts = []

            for block in cfg_blocks:
                header = f"<block type={block['block_type']}>"
                body = "\n".join(block['instructions'])

                dfg_edges = dfg_extractor.extract_block_dfg_structured(
                    block['instructions']
                )

                dfg_summary = "\n".join([
                    f"<dfg {edge[0]}->{edge[1]}>"
                    for edge in dfg_edges['edges']
                ])

                block_texts.append(
                    header + "\n" + body + "\n" + dfg_summary
                )

            converted.append({
                "block_texts": block_texts,
                "input_text": "\n\n".join(block_texts) if block_texts else build_encoder_input(
                    record,
                    config.use_reasoning,
                    config.inject_cfg_tags,
                ),
                "target_text": canonicalize_source(record["source"]),
                "cfg": cfg_blocks,
                "edges": edges,
            })

        return Dataset.from_list(converted)

    return (
        convert(load_jsonl(config.train_file)),
        convert(load_jsonl(config.eval_file)),
    )


def tokenize_dataset(dataset, tokenizer: PreTrainedTokenizerBase, config):
    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=config.max_input_length,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            batch["target_text"],
            max_length=config.max_target_length,
            truncation=True,
            padding="max_length",
        )

        block_inputs = []

        for block_group in batch.get("block_texts", []):
            block_group = [b for b in block_group if isinstance(b, str) and b.strip()]

            if not block_group:
                block_inputs.append({"input_ids": [], "attention_mask": []})
                continue

            encoded_blocks = tokenizer(
                block_group,
                max_length=256,
                truncation=True,
                padding="max_length",
            )

            block_inputs.append({
                "input_ids": encoded_blocks["input_ids"],
                "attention_mask": encoded_blocks["attention_mask"],
            })

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["cfg"] = batch.get("cfg", [])
        model_inputs["edges"] = batch.get("edges", [])
        model_inputs["block_inputs"] = block_inputs

        return model_inputs

    return dataset.map(tokenize, batched=True)


class GraphCodeBERTT5Seq2Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.local_encoder = LocalBlockEncoder()
        self.graph_encoder = GraphPoolingEncoder()

        self.encoder = AutoModel.from_pretrained(ENCODER_MODEL)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(DECODER_MODEL)
        self.decoder = self.t5_model.decoder
        self.lm_head = self.t5_model.lm_head

        self.encoder_dim = self.encoder.config.hidden_size
        self.decoder_dim = self.t5_model.config.d_model

        if self.encoder_dim != self.decoder_dim:
            self.projection = torch.nn.Linear(self.encoder_dim, self.decoder_dim)
        else:
            self.projection = torch.nn.Identity()

    def forward(self, input_ids=None, attention_mask=None, labels=None, decoder_input_ids=None, cfg=None, edges=None, block_inputs=None, **kwargs):
        block_embeddings_batch = []

        if block_inputs is not None:
            for block_group in block_inputs:
                if not block_group:
                    continue

                block_input_ids = torch.tensor(
                    block_group['input_ids'],
                    dtype=torch.long,
                    device=input_ids.device,
                )

                block_attention_mask = torch.tensor(
                    block_group['attention_mask'],
                    dtype=torch.float,
                    device=input_ids.device,
                )

                block_embeddings = self.local_encoder(
                    block_input_ids,
                    block_attention_mask,
                )

                block_embeddings_batch.append(block_embeddings)

        edge_index = None
        edge_attr = None

        if cfg is not None and edges is not None:
            try:
                batch_graphs = []

                for batch_index in range(len(cfg)):
                    node_count = max(len(cfg[batch_index]), 1)

                    if batch_index < len(block_embeddings_batch):
                        node_embeddings = block_embeddings_batch[batch_index]
                    else:
                        continue

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

        pooled_graph = self.graph_encoder(
            local_states,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        encoder_hidden_states = self.projection(
            pooled_graph.unsqueeze(1)
        )

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self.t5_model._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

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

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    tensor_builder = GraphCodeBERTTensorBuilder(tokenizer)

    train_dataset, eval_dataset = build_dataset(config)

    train_dataset = tokenize_dataset(train_dataset, tokenizer, config)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, config)

    model = initialize_model(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = GraphDataCollator(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
