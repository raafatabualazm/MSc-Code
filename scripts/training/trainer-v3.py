"""
Chunked SFT Training for Ultra-Long Sequences (90K+ tokens)
Handles sequences up to 90112 tokens by splitting into processable chunks
with gradient accumulation across chunks.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import math
import gc
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ChunkedTrainingConfig:
    """Configuration for chunked training."""
    # Model settings
    model_dir: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    
    # Chunk settings
    max_seq_length: int = 90112  # Maximum sequence length to handle
    chunk_size: int = 4096       # Size of each chunk for processing
    overlap_size: int = 256      # Overlap between chunks for context continuity
    
    # Training settings
    output_dir: str = "decompiler-chunked-v1"
    per_device_train_batch_size: int = 1  # Must be 1 for chunked processing
    gradient_accumulation_steps: int = 8  # Accumulate across samples
    chunks_accumulation: int = 1          # How many chunks before optimizer step (auto-calculated)
    num_train_epochs: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Dataset
    data_file: str = "data/matched/all_swift_matched.jsonl"
    test_size: float = 0.1


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

INFERENCE_PROMPT_STYLE = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a reverse engineering expert with advanced knowledge in assembly and {}. 
Please convert the following assembly code to idiomatic and clear {} code. 

### Assembly:
{}

### Response:
<think>
"""


# ============================================================================
# Chunked Data Collator
# ============================================================================

class ChunkedDataCollator:
    """
    Data collator that handles ultra-long sequences by chunking.
    Returns chunk information for custom training loop.
    """
    
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 90112,
        chunk_size: int = 4096,
        overlap_size: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.effective_chunk_size = chunk_size - overlap_size
        
    def create_chunks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        """Split a single sequence into overlapping chunks."""
        seq_len = input_ids.size(0)
        chunks = []
        
        if seq_len <= self.chunk_size:
            # No chunking needed
            chunks.append({
                'input_ids': input_ids.unsqueeze(0),
                'attention_mask': attention_mask.unsqueeze(0),
                'labels': labels.unsqueeze(0),
                'is_first_chunk': True,
                'is_last_chunk': True,
                'chunk_idx': 0,
                'total_chunks': 1,
            })
        else:
            # Calculate number of chunks
            num_chunks = math.ceil((seq_len - self.overlap_size) / self.effective_chunk_size)
            
            for i in range(num_chunks):
                start_idx = i * self.effective_chunk_size
                end_idx = min(start_idx + self.chunk_size, seq_len)
                
                # Adjust start for overlap (except first chunk)
                if i > 0:
                    start_idx = max(0, end_idx - self.chunk_size)
                
                chunk_input_ids = input_ids[start_idx:end_idx]
                chunk_attention_mask = attention_mask[start_idx:end_idx]
                chunk_labels = labels[start_idx:end_idx].clone()
                
                # Mask overlap region labels (except for last chunk overlap)
                # This prevents double-counting loss on overlapping tokens
                if i > 0 and self.overlap_size > 0:
                    overlap_to_mask = min(self.overlap_size, chunk_labels.size(0))
                    chunk_labels[:overlap_to_mask] = -100
                
                chunks.append({
                    'input_ids': chunk_input_ids.unsqueeze(0),
                    'attention_mask': chunk_attention_mask.unsqueeze(0),
                    'labels': chunk_labels.unsqueeze(0),
                    'is_first_chunk': (i == 0),
                    'is_last_chunk': (i == num_chunks - 1),
                    'chunk_idx': i,
                    'total_chunks': num_chunks,
                })
        
        return chunks
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function that prepares chunked batches.
        For chunked training, we process one sample at a time.
        """
        all_chunks = []
        
        for feature in features:
            text = feature['text']
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            
            # Create labels (same as input_ids for causal LM)
            labels = input_ids.clone()
            
            # Create chunks for this sample
            chunks = self.create_chunks(input_ids, attention_mask, labels)
            all_chunks.extend(chunks)
        
        return {
            'chunks': all_chunks,
            'num_samples': len(features),
        }


# ============================================================================
# Chunked Trainer
# ============================================================================

class ChunkedSFTTrainer:
    """
    Custom trainer for ultra-long sequence SFT with chunked processing.
    Handles gradient accumulation across chunks and samples.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config: ChunkedTrainingConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision='bf16',
        )
        
        # Data collator
        self.data_collator = ChunkedDataCollator(
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            chunk_size=config.chunk_size,
            overlap_size=config.overlap_size,
        )
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0,  # Must be 0 for custom collator with chunks
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
        
        # Calculate training steps
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / config.gradient_accumulation_steps
        )
        self.max_train_steps = self.num_update_steps_per_epoch * config.num_train_epochs
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler
        num_warmup_steps = int(self.max_train_steps * config.warmup_ratio)
        self.lr_scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler
            )
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.saved_checkpoints = []
        
    def _get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create cosine learning rate schedule with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    def compute_chunk_loss(self, chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a single chunk."""
        input_ids = chunk['input_ids'].to(self.accelerator.device)
        attention_mask = chunk['attention_mask'].to(self.accelerator.device)
        labels = chunk['labels'].to(self.accelerator.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        
        return outputs.loss
    
    def process_sample_chunks(
        self,
        chunks: List[Dict[str, torch.Tensor]],
        accumulate_gradients: bool = True,
    ) -> float:
        """
        Process all chunks for a sample, accumulating gradients.
        Returns the average loss across chunks.
        """
        total_loss = 0.0
        num_chunks = len(chunks)
        
        for chunk in chunks:
            with self.accelerator.accumulate(self.model):
                loss = self.compute_chunk_loss(chunk)
                
                # Scale loss by number of chunks to normalize
                scaled_loss = loss / num_chunks
                
                if accumulate_gradients:
                    self.accelerator.backward(scaled_loss)
                
                total_loss += loss.item()
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / num_chunks
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}",
            disable=not self.accelerator.is_local_main_process,
        )
        
        accumulated_samples = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            chunks = batch['chunks']
            num_samples = batch['num_samples']
            
            # Process each sample's chunks
            sample_start = 0
            for sample_idx in range(num_samples):
                # Find chunks belonging to this sample
                sample_chunks = []
                for chunk in chunks[sample_start:]:
                    sample_chunks.append(chunk)
                    if chunk['is_last_chunk']:
                        break
                sample_start += len(sample_chunks)
                
                # Process chunks with gradient accumulation
                sample_loss = self.process_sample_chunks(sample_chunks, accumulate_gradients=True)
                total_loss += sample_loss
                num_batches += 1
                accumulated_samples += 1
                
                # Update weights after gradient_accumulation_steps
                if accumulated_samples >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    accumulated_samples = 0
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / num_batches
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'step': self.global_step,
                        })
                        logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        logger.info(f"Step {self.global_step}: eval_loss={eval_loss:.4f}")
                        
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint(is_best=True)
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
            
            # Memory cleanup
            del chunks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Handle remaining accumulated gradients
        if accumulated_samples > 0:
            if self.config.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on the eval dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            chunks = batch['chunks']
            
            for chunk in chunks:
                loss = self.compute_chunk_loss(chunk)
                total_loss += loss.item()
                num_batches += 1
            
            # Memory cleanup
            del chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save a checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}" if not is_best else "best-model"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Unwrap model and save
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            save_function=self.accelerator.save,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Manage checkpoint limit
        if not is_best:
            self.saved_checkpoints.append(checkpoint_dir)
            if len(self.saved_checkpoints) > self.config.save_total_limit:
                oldest = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest):
                    import shutil
                    shutil.rmtree(oldest)
                    logger.info(f"Removed old checkpoint: {oldest}")
    
    def train(self):
        """Run the full training loop."""
        logger.info("Starting training...")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size = {self.config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        logger.info(f"  Chunk size = {self.config.chunk_size}")
        logger.info(f"  Max sequence length = {self.config.max_seq_length}")
        
        for epoch in range(self.config.num_train_epochs):
            epoch_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss:.4f}")
            
            # End of epoch evaluation
            eval_loss = self.evaluate()
            logger.info(f"Epoch {epoch + 1} eval loss: {eval_loss:.4f}")
            
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.save_checkpoint(is_best=True)
        
        # Final save
        self.save_checkpoint()
        logger.info("Training completed!")
        
        return self.best_eval_loss


# ============================================================================
# Main Training Script
# ============================================================================

def format_prompts(examples, tokenizer):
    """Format examples into training prompts."""
    inputs = examples["assembly"]
    outputs = examples["source"]
    langs = examples["language"]
    reasonings = examples["reasoning"]
    texts = []
    
    for assembly, source, lang, reasoning in zip(inputs, outputs, langs, reasonings):
        if not source.endswith(tokenizer.eos_token):
            source += tokenizer.eos_token
        text = TRAIN_PROMPT_STYLE.format(lang, lang, assembly, reasoning, lang.lower(), source)
        texts.append(text)
    
    return {"text": texts}


def main():
    # Configuration
    config = ChunkedTrainingConfig(
        model_dir="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        max_seq_length=90112,
        chunk_size=4096,          # Process 4K tokens at a time
        overlap_size=256,         # 256 token overlap for context
        output_dir="decompiler-chunked-v1",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=True,
        data_file="data/matched/all_swift_matched.jsonl",
        test_size=0.1,
    )
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model from {config.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
        quantization_config=bnb_config,
    )
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
    
    model.config.use_cache = False
    
    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.07,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ],
    )
    
    # Apply LoRA
    logger.info("Applying LoRA configuration")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    logger.info(f"Loading dataset from {config.data_file}")
    dataset = load_dataset("json", data_files=config.data_file, split="train")
    
    # Split dataset
    dataset_split = dataset.train_test_split(test_size=config.test_size, seed=42)
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]
    
    # Format datasets
    train_dataset = train_dataset.map(
        lambda x: format_prompts(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        lambda x: format_prompts(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Analyze sequence lengths
    logger.info("Analyzing sequence lengths...")
    sample_lengths = []
    for i in range(min(100, len(train_dataset))):
        tokens = tokenizer(train_dataset[i]['text'], return_tensors='pt')
        sample_lengths.append(tokens['input_ids'].size(1))
    
    logger.info(f"Sample sequence lengths - Min: {min(sample_lengths)}, Max: {max(sample_lengths)}, "
                f"Mean: {sum(sample_lengths)/len(sample_lengths):.0f}")
    
    # Initialize trainer
    trainer = ChunkedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        config=config,
    )
    
    # Train
    gc.collect()
    torch.cuda.empty_cache()
    
    best_loss = trainer.train()
    logger.info(f"Training complete. Best eval loss: {best_loss:.4f}")
    
    # Merge and save final model
    logger.info("Merging LoRA weights and saving final model...")
    merged_model = model.merge_and_unload()
    full_dir = f"{config.output_dir}-full"
    merged_model.save_pretrained(full_dir, safe_serialization=True)
    tokenizer.save_pretrained(full_dir)
    logger.info(f"Saved merged model to {full_dir}")
    
    # Test inference
    logger.info("Running test inference...")
    test_assembly = train_dataset[0]['text'].split("### Assembly:\n")[1].split("\n\n### Response:")[0]
    test_lang = "Swift"
    
    inputs = tokenizer(
        INFERENCE_PROMPT_STYLE.format(test_lang, test_lang, test_assembly[:2000]),  # Truncate for test
        return_tensors="pt"
    ).to(merged_model.device)
    
    with torch.no_grad():
        outputs = merged_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Test inference completed successfully!")
    print("\n" + "=" * 50)
    print("Sample Output:")
    print(response.split("### Response:")[1][:1000] if "### Response:" in response else response[:1000])
    print("=" * 50)


if __name__ == "__main__":
    main()

