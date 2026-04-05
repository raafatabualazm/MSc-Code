from __future__ import annotations

import gc
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ChunkedTrainingConfig:
    model_dir: str = "Qwen/Qwen3-4B-Thinking-2507"

    # Sequence / chunking
    max_seq_length: Optional[int] = 90112  # None = no hard cap
    chunk_size: int = 16384
    overlap_size: int = 0

    # KV-cache forwarding (bounded-history causal attention)
    use_kv_cache_forwarding: bool = True
    max_kv_cache_tokens: int = 32768  # 0 = unlimited

    # Training
    output_dir: str = "decompiler-kvcache-fixed"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Logging / checkpointing
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3

    # Memory / speed
    gradient_checkpointing: bool = False
    use_flash_attention: bool = True

    # Dataset
    data_file: str = "all_swift_matched.jsonl"
    test_size: float = 0.1

    def validate(self) -> None:
        if self.per_device_train_batch_size != 1:
            raise ValueError("Chunked training requires per_device_train_batch_size=1")
        if self.overlap_size < 0 or self.overlap_size >= self.chunk_size:
            raise ValueError("overlap_size must satisfy 0 <= overlap_size < chunk_size")
        if self.use_kv_cache_forwarding and self.overlap_size != 0:
            raise ValueError(
                "Set overlap_size=0 when use_kv_cache_forwarding=True to avoid double-counting context"
            )
        if self.use_kv_cache_forwarding and self.gradient_checkpointing:
            raise ValueError(
                "gradient_checkpointing=True with use_kv_cache_forwarding=True is intentionally disabled here. "
                "Many HF causal LM implementations do not safely support training-time use_cache with checkpointing."
            )


# ============================================================================
# Prompt Templates
# ============================================================================


TRAIN_PROMPT_STYLE = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and {}.
Please convert the following assembly code to idiomatic and clear {} code.

### Assembly:
{}

### Response:
<think>
{}
</think>
```{}
{}
```"""


# ============================================================================
# KV-cache utilities
# ============================================================================


def detach_kv_cache(past_key_values: Tuple[Tuple[torch.Tensor, ...], ...]) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    return tuple(tuple(t.detach() for t in layer_cache) for layer_cache in past_key_values)



def truncate_kv_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
    max_length: int,
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    if max_length <= 0:
        return past_key_values
    current_length = past_key_values[0][0].size(2)
    if current_length <= max_length:
        return past_key_values
    return tuple(tuple(t[:, :, -max_length:, :] for t in layer_cache) for layer_cache in past_key_values)



def get_kv_cache_length(past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]]) -> int:
    if past_key_values is None or len(past_key_values) == 0:
        return 0
    return int(past_key_values[0][0].size(2))


# ============================================================================
# Dataset formatting / tokenization
# ============================================================================


def format_prompts(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List[str]]:
    texts: List[str] = []
    for assembly, source, lang, reasoning in zip(
        examples["assembly"],
        examples["source"],
        examples["language"],
        examples["reasoning"],
    ):
        if tokenizer.eos_token and not source.endswith(tokenizer.eos_token):
            source += tokenizer.eos_token
        text = TRAIN_PROMPT_STYLE.format(lang, lang, assembly, reasoning, lang.lower(), source)
        texts.append(text)
    return {"text": texts}



def tokenize_examples(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    hard_cap_tokens: Optional[int] = None,
) -> Dict[str, List[List[int]]]:
    all_input_ids: List[List[int]] = []
    all_attention_masks: List[List[int]] = []
    lengths: List[int] = []

    for text in examples["text"]:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if hard_cap_tokens is not None and len(input_ids) > hard_cap_tokens:
            # Tail-preserving cap so the response target is retained.
            input_ids = input_ids[-hard_cap_tokens:]
            attention_mask = attention_mask[-hard_cap_tokens:]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        lengths.append(len(input_ids))

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "seq_len": lengths,
    }


# ============================================================================
# Chunked data collator
# ============================================================================


class ChunkedDataCollator:
    def __init__(self, chunk_size: int, overlap_size: int = 0):
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.stride = chunk_size - overlap_size

    def create_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        seq_len = int(input_ids.size(0))
        chunks: List[Dict[str, Any]] = []

        if seq_len <= self.chunk_size:
            chunks.append(
                {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0),
                    "labels": labels.unsqueeze(0),
                    "chunk_idx": 0,
                    "seq_position_offset": 0,
                }
            )
            chunks[0]["total_chunks"] = 1
            return chunks

        start = 0
        chunk_idx = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)

            chunk_input_ids = input_ids[start:end]
            chunk_attention_mask = attention_mask[start:end]
            chunk_labels = labels[start:end].clone()

            # Only relevant when overlap_size > 0 and KV forwarding is disabled.
            if start > 0 and self.overlap_size > 0:
                overlap = min(self.overlap_size, int(chunk_labels.size(0)))
                chunk_labels[:overlap] = -100

            chunks.append(
                {
                    "input_ids": chunk_input_ids.unsqueeze(0),
                    "attention_mask": chunk_attention_mask.unsqueeze(0),
                    "labels": chunk_labels.unsqueeze(0),
                    "chunk_idx": chunk_idx,
                    "seq_position_offset": start,
                }
            )

            if end >= seq_len:
                break

            start += self.stride
            chunk_idx += 1

        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        return chunks

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError(f"Expected batch_size=1, got {len(features)}")

        feature = features[0]
        input_ids = torch.tensor(feature["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(feature["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()

        chunks = self.create_chunks(input_ids, attention_mask, labels)
        return {
            "chunks": chunks,
            "seq_len": int(input_ids.size(0)),
            "num_chunks": len(chunks),
        }


# ============================================================================
# Trainer
# ============================================================================


class ChunkedSFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        train_dataset,
        eval_dataset,
        config: ChunkedTrainingConfig,
        accelerator: Accelerator,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.accelerator = accelerator
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        self.data_collator = ChunkedDataCollator(
            chunk_size=config.chunk_size,
            overlap_size=config.overlap_size,
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True,
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.gradient_accumulation_steps
        )
        self.max_train_steps = self.num_update_steps_per_epoch * config.num_train_epochs
        self.lr_scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.max_train_steps * config.warmup_ratio),
            num_training_steps=self.max_train_steps,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.saved_checkpoints: List[str] = []

    def _get_cosine_schedule_with_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
    ) -> torch.Tensor:
        if past_key_values is None:
            return attention_mask
        cache_len = get_kv_cache_length(past_key_values)
        cache_mask = torch.ones(
            (attention_mask.size(0), cache_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return torch.cat([cache_mask, attention_mask], dim=1)

    def forward_chunk(
        self,
        chunk: Dict[str, Any],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
    ) -> Tuple[torch.Tensor, int, Optional[Tuple[Tuple[torch.Tensor, ...], ...]], torch.Tensor]:
        device = self.accelerator.device
        input_ids = chunk["input_ids"].to(device, non_blocking=True)
        attention_mask = chunk["attention_mask"].to(device, non_blocking=True)
        labels = chunk["labels"].to(device, non_blocking=True)

        seq_len = int(input_ids.size(1))
        seq_offset = int(chunk["seq_position_offset"])
        use_cache = bool(self.config.use_kv_cache_forwarding)

        attention_mask = self._prepare_attention_mask(attention_mask, past_key_values)
        position_ids = torch.arange(
            seq_offset,
            seq_offset + seq_len,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid_tokens = int((shift_labels != -100).sum().item())

        if valid_tokens > 0:
            loss_sum = self.loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        else:
            loss_sum = outputs.logits.new_zeros(())

        next_logits = outputs.logits[:, -1, :].contiguous()
        new_past_kv = outputs.past_key_values if use_cache else None
        return loss_sum, valid_tokens, new_past_kv, next_logits

    def process_one_sample(
        self,
        chunks: List[Dict[str, Any]],
        backward: bool = True,
    ) -> Tuple[float, int]:
        if not chunks:
            return 0.0, 0

        # Count all valid targets, including one boundary target between adjacent chunks.
        total_valid_tokens = 0
        for idx, chunk in enumerate(chunks):
            labels = chunk["labels"]
            total_valid_tokens += int((labels[:, 1:] != -100).sum().item())
            if idx + 1 < len(chunks):
                next_first_label = chunks[idx + 1]["labels"][:, 0]
                total_valid_tokens += int((next_first_label != -100).sum().item())

        if total_valid_tokens == 0:
            return 0.0, 0

        total_loss_sum_value = 0.0
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None

        for idx, chunk in enumerate(chunks):
            loss_sum, valid_tokens, new_past_kv, next_token_logits = self.forward_chunk(
                chunk,
                past_key_values=past_key_values,
            )

            boundary_valid_tokens = 0
            if idx + 1 < len(chunks):
                next_first_label = chunks[idx + 1]["labels"][:, 0].to(self.accelerator.device)
                boundary_valid_tokens = int((next_first_label != -100).sum().item())
                if boundary_valid_tokens > 0:
                    boundary_loss = self.loss_fct(next_token_logits, next_first_label)
                    loss_sum = loss_sum + boundary_loss

            chunk_total_valid = valid_tokens + boundary_valid_tokens

            if backward and chunk_total_valid > 0:
                self.accelerator.backward(loss_sum / total_valid_tokens)

            total_loss_sum_value += float(loss_sum.detach().float().item())

            if self.config.use_kv_cache_forwarding and new_past_kv is not None:
                past_key_values = detach_kv_cache(new_past_kv)
                if self.config.max_kv_cache_tokens > 0:
                    past_key_values = truncate_kv_cache(
                        past_key_values,
                        self.config.max_kv_cache_tokens,
                    )
            else:
                past_key_values = None

            del loss_sum, new_past_kv, next_token_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sample_mean_loss = total_loss_sum_value / total_valid_tokens
        return sample_mean_loss, total_valid_tokens

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss_weighted = 0.0
        total_tokens = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}",
            disable=not self.accelerator.is_local_main_process,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch in progress_bar:
            chunks = batch["chunks"]

            with self.accelerator.accumulate(self.model):
                sample_loss, sample_tokens = self.process_one_sample(chunks, backward=True)
                total_loss_weighted += sample_loss * sample_tokens
                total_tokens += sample_tokens

                if self.accelerator.sync_gradients:
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss_weighted / max(total_tokens, 1)
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        progress_bar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{current_lr:.2e}",
                                "step": self.global_step,
                                "tokens": total_tokens,
                            }
                        )
                        logger.info(
                            "Step %s: loss=%0.4f lr=%0.2e tokens=%s",
                            self.global_step,
                            avg_loss,
                            current_lr,
                            total_tokens,
                        )

                    if self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        logger.info("Step %s: eval_loss=%0.4f", self.global_step, eval_loss)
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint(is_best=True)
                        self.model.train()

                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

            del chunks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss_weighted / max(total_tokens, 1)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss_weighted = 0.0
        total_tokens = 0

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            chunks = batch["chunks"]
            sample_loss, sample_tokens = self.process_one_sample(chunks, backward=False)
            total_loss_weighted += sample_loss * sample_tokens
            total_tokens += sample_tokens

            del chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss_weighted / max(total_tokens, 1)

    def save_checkpoint(self, is_best: bool = False) -> None:
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = os.path.join(
            self.config.output_dir,
            "best-model" if is_best else f"checkpoint-{self.global_step}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            save_function=self.accelerator.save,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(checkpoint_dir)

        torch.save(
            {
                "global_step": self.global_step,
                "best_eval_loss": self.best_eval_loss,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, "training_state.pt"),
        )

        logger.info("Saved checkpoint to %s", checkpoint_dir)

        if not is_best:
            self.saved_checkpoints.append(checkpoint_dir)
            if len(self.saved_checkpoints) > self.config.save_total_limit:
                oldest = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest):
                    import shutil

                    shutil.rmtree(oldest)
                    logger.info("Removed old checkpoint: %s", oldest)

    def train(self) -> float:
        logger.info("=" * 60)
        logger.info("Starting bounded-history chunked SFT training")
        logger.info("=" * 60)
        logger.info("Train examples: %s", len(self.train_dataset))
        logger.info("Eval examples: %s", len(self.eval_dataset))
        logger.info("Epochs: %s", self.config.num_train_epochs)
        logger.info("Chunk size: %s", self.config.chunk_size)
        logger.info("Max seq length: %s", self.config.max_seq_length)
        logger.info("KV forwarding: %s", self.config.use_kv_cache_forwarding)
        logger.info("Max KV cache tokens: %s", self.config.max_kv_cache_tokens)
        logger.info("Grad accumulation: %s", self.config.gradient_accumulation_steps)
        logger.info("Total optimization steps: %s", self.max_train_steps)

        for epoch in range(self.config.num_train_epochs):
            train_loss = self.train_epoch(epoch)
            logger.info("Epoch %s train loss: %0.4f", epoch + 1, train_loss)

            eval_loss = self.evaluate()
            logger.info("Epoch %s eval loss: %0.4f", epoch + 1, eval_loss)

            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.save_checkpoint(is_best=True)

        self.save_checkpoint()
        logger.info("Training completed")
        return self.best_eval_loss


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    config = ChunkedTrainingConfig(
        model_dir="Qwen/Qwen3-4B-Thinking-2507",
        max_seq_length=90112,
        chunk_size=16384,
        overlap_size=0,
        use_kv_cache_forwarding=True,
        max_kv_cache_tokens=32768,
        output_dir="decompiler-kvcache-fixed",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=False,
        data_file="all_swift_matched.jsonl",
        test_size=0.1,
    )
    config.validate()

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    os.makedirs(config.output_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading tokenizer from %s", config.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s", config.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
        quantization_config=bnb_config,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.07,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    logger.info("Applying LoRA configuration")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset from %s", config.data_file)
    dataset = load_dataset("json", data_files=config.data_file, split="train")
    dataset_split = dataset.train_test_split(test_size=config.test_size, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    logger.info("Formatting prompts")
    train_dataset = train_dataset.map(
        lambda x: format_prompts(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_prompts(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    logger.info("Pretokenizing")
    train_dataset = train_dataset.map(
        lambda x: tokenize_examples(x, tokenizer, hard_cap_tokens=config.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_examples(x, tokenizer, hard_cap_tokens=config.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    train_lengths = train_dataset["seq_len"]
    logger.info(
        "Train samples=%s Eval samples=%s Min=%s Max=%s Mean=%0.0f",
        len(train_dataset),
        len(eval_dataset),
        min(train_lengths),
        max(train_lengths),
        sum(train_lengths) / len(train_lengths),
    )

    trainer = ChunkedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        accelerator=accelerator,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_loss = trainer.train()
    logger.info("Best eval loss: %0.4f", best_loss)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(trainer.model)
        unwrapped_model.save_pretrained(config.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(config.output_dir)

        try:
            merged_model = unwrapped_model.merge_and_unload()
            full_dir = f"{config.output_dir}-full"
            merged_model.save_pretrained(full_dir, safe_serialization=True)
            tokenizer.save_pretrained(full_dir)
            logger.info("Saved merged model to %s", full_dir)
        except Exception as exc:
            logger.warning("Could not merge adapter weights: %s", exc)


if __name__ == "__main__":
    main()
